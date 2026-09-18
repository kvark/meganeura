// K-split GEMV for M=1 against a Meganeura Q8 weight (GGML Q8_0 after its
// host repack), with the activation row quantized to Q8_1 so the inner
// product uses packed integer arithmetic. Where the device reports
// `shader_integer_dot_product` the four-byte dot products are the hardware
// `dot4I8Packed` (DP4A); otherwise the exact scalar expansion runs.
//
// The arithmetic is llama.cpp's `vec_dot_q8_0_q8_1`. A Q8 block stores
// `value = d * q8w` for an int8 quant and an f16 scale, and a Q8_1 block
// stores int8 quants with `d8` and `s8 = d8 * sum(q8a)`. Over one block:
//
//     sum_i q8w_i * d * q8a_i * d8 = d * (d8 * sum_i(q8w_i * q8a_i))
//
// so the whole 32-element block reduces to eight integer dot products and
// one scale product. Q8_0 is symmetric about zero, so — unlike Q4_0 — there
// is no -8 bias and no `s8` correction term. The sum is exact in i32: the
// operands are bounded by 127, and 32 such products cannot overflow.
//
// Meganeura's Q8 block is nine words: the f16 scale in the low half of word
// zero and the thirty-two int8 quants in words one through eight, four per
// word with element 4j + b in byte b. Every word is whole and aligned, so
// each pairs *directly* with one word of the Q8_1 activation quants — no
// nibble splitting, no unaligned stitching.
//
// The activation is quantized in the kernel rather than in a dispatch of its
// own. Each workgroup covers four output columns and every thread takes
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

$PACKED_DOT_HELPER

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
            let base = ((col + c) * blocks + blk) * 9u;
            var sumi = 0;
            for (var j = 0u; j < 8u; j++) {
                sumi += dot_q8_q8_packed(matrix_b[base + 1u + j], u[j]);
            }
            // The f16 scale rides in the low half of the block's first
            // word; `unpack2x16float` decodes it exactly as the plain
            // decoder's `decode_f16` does.
            acc[c] += unpack2x16float(matrix_b[base] & 0xFFFFu).x * (d8 * f32(sumi));
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
