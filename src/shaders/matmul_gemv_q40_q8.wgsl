// K-split GEMV for M=1 against a GGML Q4_0 weight, with the activation row
// quantized to Q8_1 so the inner product runs on `dot4I8Packed` instead of
// f32 multiply-add.
//
// The arithmetic is llama.cpp's `vec_dot_q4_0_q8_1`. A Q4_0 block stores
// `value = d4 * (q - 8)` for a nibble `q`, and a Q8_1 block stores int8
// quants with `d8` and `s8 = d8 * sum(q8)`. Over one block:
//
//     sum_i (q4_i - 8) * d4 * q8_i * d8
//   = d4 * (d8 * sum_i(q4_i * q8_i) - 8 * s8)
//
// so the whole 32-element block reduces to eight integer dot products and
// one correction term. `sum_i(q4_i * q8_i)` is exact in i32: the operands
// are bounded by 15 and 127, and 32 such products cannot overflow.
//
// GGML's split-nibble layout is what makes this work. One 4-byte word of
// `qs` holds eight nibbles: four low ones for elements 4i..4i+3 and four
// high ones for elements 16+4i..16+4i+3. Masking gives two int8x4 vectors of
// *consecutive* elements, each of which pairs with one word of the Q8_1
// quants. Meganeura's own Q4, which pairs neighbouring elements in a byte,
// could not feed this without a shuffle.
//
// The activation is quantized in the kernel rather than in a dispatch of its
// own. Each workgroup covers four output columns and every thread takes
// whole 32-element blocks, so a block is quantized once per workgroup and
// reused across its four columns; no workgroup memory and no barrier are
// needed for it. That repeats the work across workgroups the way the
// RmsNorm-fused GEMV repeats its prologue, and for the same reason: the
// activation is a few kilobytes that every workgroup is reading at once.

struct Params {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
}

var<storage> matrix_a: array<f32>;
var<storage> matrix_b: array<u32>;
var<storage, read_write> matrix_c: array<vec4<f32>>;
var<uniform> params: Params;
// Workgroup width; see `matmul_gemv.wgsl`.
const LANES: u32 = 256u;

var<workgroup> reduce_buf: array<vec4<f32>, LANES>;

// The f16 scale at the head of the Q4_0 block starting at `byte_base`.
fn q40_scale(byte_base: u32) -> f32 {
    let lo = (matrix_b[byte_base / 4u] >> ((byte_base % 4u) * 8u)) & 0xFFu;
    let at = byte_base + 1u;
    let hi = (matrix_b[at / 4u] >> ((at % 4u) * 8u)) & 0xFFu;
    return unpack2x16float(lo | (hi << 8u)).x;
}

// Nibble word `i` of the block: bytes 2 + 4i .. 2 + 4i + 4.
//
// Blocks are 18 bytes, so this is word-aligned for odd blocks and offset by
// two bytes for even ones; the unaligned case stitches two words. The second
// word is always in bounds: the last read of an even block ends at
// `byte_base + 19`, and a buffer whose final block is even has an odd block
// count, which leaves `18 * blocks` two bytes short of a word and so gets
// padded by exactly the two bytes needed.
fn q40_nibbles(byte_base: u32, i: u32) -> u32 {
    let at = byte_base + 2u + i * 4u;
    let word = matrix_b[at / 4u];
    let shift = (at % 4u) * 8u;
    if shift == 0u {
        return word;
    }
    return (word >> shift) | (matrix_b[at / 4u + 1u] << (32u - shift));
}

@compute @workgroup_size(LANES)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let col4 = wgid.x;
    let lane = lid.x;
    let n_v4 = params.n / 4u;
    if col4 >= n_v4 { return; }
    let blocks = params.k / 32u;

    var acc = vec4<f32>(0.0);
    var blk = lane;
    loop {
        if blk >= blocks { break; }

        // Quantize this block of the activation to Q8_1.
        let abase = blk * 32u;
        var amax = 0.0;
        for (var i = 0u; i < 32u; i++) {
            amax = max(amax, abs(matrix_a[abase + i]));
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
                let q = clamp(i32(round(matrix_a[abase + j * 4u + b] * inv)), -127, 127);
                qsum += q;
                word |= (bitcast<u32>(q) & 0xFFu) << (b * 8u);
            }
            u[j] = word;
        }
        let s8 = d8 * f32(qsum);

        // Four columns share the quantized block.
        let col = col4 * 4u;
        for (var c = 0u; c < 4u; c++) {
            let byte_base = ((col + c) * blocks + blk) * 18u;
            var sumi = 0;
            for (var i = 0u; i < 4u; i++) {
                let vi = q40_nibbles(byte_base, i);
                // Low nibbles are elements 4i..4i+3, high nibbles are
                // 16+4i..16+4i+3, so they pair with quant words i and 4+i.
                sumi += dot4I8Packed(vi & 0x0F0F0F0Fu, u[i]);
                sumi += dot4I8Packed((vi >> 4u) & 0x0F0F0F0Fu, u[4u + i]);
            }
            acc[c] += q40_scale(byte_base) * (d8 * f32(sumi) - 8.0 * s8);
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
        matrix_c[col4] = reduce_buf[0] + reduce_buf[1];
    }
}
