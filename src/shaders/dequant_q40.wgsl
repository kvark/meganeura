// Byte `off` of the block starting at absolute byte `byte_base`.
fn q40_byte(byte_base: u32, off: u32) -> u32 {
    let at = byte_base + off;
    return (matrix_b[at / 4u] >> ((at % 4u) * 8u)) & 0xFFu;
}

fn q40_d(byte_base: u32) -> f32 {
    let lo = q40_byte(byte_base, 0u);
    let hi = q40_byte(byte_base, 1u);
    return decode_f16(lo | (hi << 8u));
}

// The 4-bit quant for element `e`, biased by -8. GGML splits a block's
// nibbles across halves: byte `j` carries element `j` in its low nibble and
// element `j + 16` in its high nibble, where Meganeura's own Q4 pairs
// adjacent elements into one byte.
fn q40_quant(byte_base: u32, e: u32) -> i32 {
    let raw = q40_byte(byte_base, 2u + e % 16u);
    let nibble = select(raw & 0xFu, raw >> 4u, e >= 16u);
    return i32(nibble) - 8;
}

fn dequant_q40(k_idx: u32, n_idx: u32) -> f32 {
    let byte_base = (n_idx * (params.k / 32u) + k_idx / 32u) * 18u;
    return q40_d(byte_base) * f32(q40_quant(byte_base, k_idx % 32u));
}

fn dequant_q40_pack8(k_base: u32, n_idx: u32) -> array<f32, 8> {
    // Eight 8-aligned elements share a block, so the f16 scale is read once
    // for all of them. The nibbles are not shared: the split-half layout
    // pairs element `e` with `e + 16`, never with `e + 1`, so eight
    // consecutive elements come from eight separate bytes.
    let byte_base = (n_idx * (params.k / 32u) + k_base / 32u) * 18u;
    let e = k_base % 32u;
    let d = q40_d(byte_base);
    var out: array<f32, 8>;
    for (var i = 0u; i < 8u; i++) {
        out[i] = d * f32(q40_quant(byte_base, e + i));
    }
    return out;
}
