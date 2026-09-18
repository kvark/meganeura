// K-split GEMV for M=1 against a GGML K-quant weight (Q4_K, Q5_K, Q6_K or
// Q3_K superblocks), with the activation row quantized to Q8_1 so the inner
// product uses packed integer arithmetic. Where the device reports
// `shader_integer_dot_product` the four-byte dot products are the hardware
// `dot4I8Packed` (DP4A); otherwise the exact scalar expansion runs.
//
// A superblock spans 256 elements in eight 32-element sub-blocks, which
// pair one-to-one with Q8_1 activation blocks, so the loop below stays
// per-32-elements like the Q4_0 and Q8 kernels; `$BLOCK_DOT` derives the
// superblock index from the activation-block index and applies the
// format's per-sub-block scale arithmetic. GGML's CUDA `vec_dot_*_K_q8_1`
// kernels compute exactly this shape of sum: an integer dot of the raw
// quants against the activation quants, scaled per sub-block, minus one
// constant term per sub-block times the activation block's raw sum.
//
//   Q4_K: value = d * sc * q - dmin * mn
//         → (d*sc)*d8*sumi - (dmin*mn)*s8
//   Q5_K: value = d * sc * (nib + 16*hi) - dmin * mn
//         → (d*sc)*d8*(sumi_nib + 16*sumi_hi) - (dmin*mn)*s8
//   Q6_K: value = d * sc * (q6 - 32), two 16-element sub-blocks per block
//         → Σ_j (d*sc_j)*d8*(sumi_j - 32*qsum)
//   Q3_K: value = d * sc * (q2 - (hbit ? 0 : 4)), likewise two sub-blocks
//
// Every integer dot is exact in i32: the packed operands stay within one
// signed byte and at most 32 products per dot, so nothing overflows.
//
// The activation is quantized in the kernel rather than in a dispatch of
// its own. Each workgroup covers four output columns and every thread takes
// whole 32-element blocks, so a block is quantized once per workgroup and
// reused across its four columns; no workgroup memory and no barrier are
// needed for it. That repeats the work across workgroups the way the
// RmsNorm-fused GEMV repeats its prologue, and for the same reason: the
// activation is a few kilobytes that every workgroup is reading at once.
//
// When a source RmsNorm folds in, the prologue computes `inv_rms` from the
// *unnormalized* row and `a_val` applies the scale on the way to the
// quantizer, so the integer arithmetic sees exactly the row the unfused
// path would have quantized.

struct Params {
    m: u32,
    n: u32,
    k: u32,
$PARAM_PAD
}

var<storage> matrix_a: array<f32>;
$NORM_DECL
var<storage> matrix_b: array<u32>;
var<storage, read_write> matrix_c: array<vec4<f32>>;
$ADDEND_DECL
var<uniform> params: Params;
// Workgroup width; see `matmul_gemv.wgsl`.
const LANES: u32 = 256u;

var<workgroup> reduce_buf: array<vec4<f32>, LANES>;

// One byte at `byte_base + off`, and four consecutive bytes as one word.
// The 210- and 110-byte superblocks are not whole numbers of words, so
// every read is byte-addressed and the unaligned ones stitch two words;
// the word-aligned Q4_K and Q5_K take the shift-zero path for free.
fn kq_byte_at(byte_base: u32, off: u32) -> u32 {
    let at = byte_base + off;
    return (matrix_b[at / 4u] >> ((at % 4u) * 8u)) & 0xFFu;
}

fn kq_word4(byte_base: u32, off: u32) -> u32 {
    let at = byte_base + off;
    let word = matrix_b[at / 4u];
    let shift = (at % 4u) * 8u;
    if shift == 0u {
        return word;
    }
    return (word >> shift) | (matrix_b[at / 4u + 1u] << (32u - shift));
}

$PACKED_DOT_HELPER

$WEIGHT_HELPERS

$A_FN_DECL

@compute @workgroup_size(LANES)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let col4 = wgid.x;
    let lane = lid.x;
    let n_v4 = params.n / 4u;
    let k = params.k;
$NORM_PROLOGUE
    if col4 >= n_v4 { return; }
    let blocks = k / 32u;

    var acc = vec4<f32>(0.0);
    var blk = lane;
    loop {
        if blk >= blocks { break; }

        // Quantize this block of the activation to Q8_1.
        let abase = blk * 32u;
        var amax = 0.0;
        for (var i = 0u; i < 32u; i++) {
            amax = max(amax, abs(a_val(abase + i)));
        }
        let d8 = amax / 127.0;
        // An all-zero block has no scale; leave it at zero rather than
        // dividing by it, which would put NaN into every column.
        let inv = select(0.0, 1.0 / d8, amax > 0.0);
        var u: array<u32, 8>;
        var qsum = 0;
        for (var j = 0u; j < 8u; j++) {
            var word = 0u;
            for (var b = 0u; b < 4u; b++) {
                // With finite input this cannot exceed 127 — every value
                // is at most `amax` and the divisor is `amax / 127` — so the
                // clamp is a guard on the float-to-int conversion for a row
                // carrying an infinity or a NaN, not a rounding case.
                let q = clamp(i32(round(a_val(abase + j * 4u + b) * inv)), -127, 127);
                qsum += q;
                word |= (bitcast<u32>(q) & 0xFFu) << (b * 8u);
            }
            u[j] = word;
        }
        let s8 = d8 * f32(qsum);

        // Four columns share the quantized block.
        let col = col4 * 4u;
        for (var c = 0u; c < 4u; c++) {
            acc[c] += $BLOCK_DOT(col + c, blk, &u, d8, s8);
        }
        blk += LANES;
    }

    reduce_buf[lane] = acc;
    workgroupBarrier();
    if lane < 128u { reduce_buf[lane] += reduce_buf[lane + 128u]; }
    workgroupBarrier();
    if lane < 64u { reduce_buf[lane] += reduce_buf[lane + 64u]; }
    workgroupBarrier();
    if lane < 32u { reduce_buf[lane] += reduce_buf[lane + 32u]; }
    workgroupBarrier();
    if lane < 16u { reduce_buf[lane] += reduce_buf[lane + 16u]; }
    workgroupBarrier();
    if lane < 8u  { reduce_buf[lane] += reduce_buf[lane + 8u];  }
    workgroupBarrier();
    if lane < 4u  { reduce_buf[lane] += reduce_buf[lane + 4u];  }
    workgroupBarrier();
    if lane < 2u  { reduce_buf[lane] += reduce_buf[lane + 2u];  }
    workgroupBarrier();
    if lane == 0u {
        matrix_c[col4] = reduce_buf[0] + reduce_buf[1]$ADDEND;
    }
}
