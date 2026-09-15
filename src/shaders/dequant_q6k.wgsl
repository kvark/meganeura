// Byte `off` of the superblock starting at absolute byte `byte_base`.
fn q6k_byte(byte_base: u32, off: u32) -> u32 {
    let at = byte_base + off;
    return (matrix_b[at / 4u] >> ((at % 4u) * 8u)) & 0xFFu;
}

// scales[] is int8_t, and these do go negative.
fn q6k_scale(byte_base: u32, i: u32) -> f32 {
    let raw = q6k_byte(byte_base, 192u + i);
    return f32(i32(raw) - select(0, 256, raw >= 128u));
}

fn q6k_d(byte_base: u32) -> f32 {
    let lo = q6k_byte(byte_base, 208u);
    let hi = q6k_byte(byte_base, 209u);
    return decode_f16(lo | (hi << 8u));
}

// The 6-bit quant for element `e`, before scaling.
fn q6k_quant(byte_base: u32, e: u32) -> i32 {
    let half = e / 128u;
    let within = e % 128u;
    let j = within / 32u;
    let l = within % 32u;
    let ql = q6k_byte(byte_base, half * 64u + l + (j & 1u) * 32u);
    let lo = select(ql >> 4u, ql & 0xFu, j < 2u);
    let qh = q6k_byte(byte_base, 128u + half * 32u + l);
    let hi = (qh >> (j * 2u)) & 3u;
    return i32(lo | (hi << 4u)) - 32;
}

// Which of the sixteen sub-block scales covers element `e`.
fn q6k_scale_index(e: u32) -> u32 {
    let half = e / 128u;
    let within = e % 128u;
    return half * 8u + (within / 32u) * 2u + (within % 32u) / 16u;
}

fn dequant_q6k(k_idx: u32, n_idx: u32) -> f32 {
    let byte_base = (n_idx * (params.k / 256u) + k_idx / 256u) * 210u;
    let e = k_idx % 256u;
    let scale = q6k_d(byte_base) * q6k_scale(byte_base, q6k_scale_index(e));
    return scale * f32(q6k_quant(byte_base, e));
}

fn dequant_q6k_pack8(k_base: u32, n_idx: u32) -> array<f32, 8> {
    // Eight 8-aligned elements share a superblock, a 32-element stride and
    // a 16-element scale group, so the f16 decode and the scale byte are
    // read once for all of them. The payload stays per element: ql and qh
    // live in separate regions at arbitrary word alignment.
    let byte_base = (n_idx * (params.k / 256u) + k_base / 256u) * 210u;
    let e = k_base % 256u;
    let scale = q6k_d(byte_base) * q6k_scale(byte_base, q6k_scale_index(e));
    var out: array<f32, 8>;
    for (var i = 0u; i < 8u; i++) {
        out[i] = scale * f32(q6k_quant(byte_base, e + i));
    }
    return out;
}
