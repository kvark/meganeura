fn q3k_byte(byte_base: u32, off: u32) -> u32 {
    let at = byte_base + off;
    return (matrix_b[at / 4u] >> ((at % 4u) * 8u)) & 0xFFu;
}

// Scale `i` of sixteen, biased by 32. Mirrors the kmask1/kmask2 shuffle in
// dequantize_row_q3_K: groups 0 and 1 take low nibbles of scales[0..8],
// groups 2 and 3 the high nibbles, and each borrows two more bits from
// scales[8..12].
fn q3k_scale(byte_base: u32, i: u32) -> f32 {
    let b = i % 4u;
    let g = i / 4u;
    let src = select(4u + b, b, (g % 2u) == 0u);
    let raw = q3k_byte(byte_base, 96u + src);
    let nib = select(raw >> 4u, raw & 0xFu, g < 2u);
    let hi = (q3k_byte(byte_base, 96u + 8u + b) >> (g * 2u)) & 3u;
    return f32(i32(nib | (hi << 4u)) - 32);
}

// The 2-bit quant for element `e`, with the inverted high bit applied.
//
// `l` is the element's position within its 32-element stride, which is
// `e % 32` once the half and stride terms cancel; `hmask` is shared across
// halves and strides, and the bit that selects between them is `e / 32`.
fn q3k_quant(byte_base: u32, e: u32) -> i32 {
    let h = e / 128u;
    let j = (e % 128u) / 32u;
    let l = e % 32u;
    let q = (q3k_byte(byte_base, 32u + h * 32u + l) >> (j * 2u)) & 3u;
    let hbit = (q3k_byte(byte_base, l) >> (e / 32u)) & 1u;
    return i32(q) - select(4, 0, hbit == 1u);
}

fn q3k_d(byte_base: u32) -> f32 {
    let lo = q3k_byte(byte_base, 108u);
    let hi = q3k_byte(byte_base, 109u);
    return decode_f16(lo | (hi << 8u));
}

fn dequant_q3k(k_idx: u32, n_idx: u32) -> f32 {
    let byte_base = (n_idx * (params.k / 256u) + k_idx / 256u) * 110u;
    let e = k_idx % 256u;
    // One scale per 16 elements.
    let scale = q3k_d(byte_base) * q3k_scale(byte_base, e / 16u);
    return scale * f32(q3k_quant(byte_base, e));
}

fn dequant_q3k_pack8(k_base: u32, n_idx: u32) -> array<f32, 8> {
    // Eight 8-aligned elements share a superblock and a 16-element scale
    // group, so the f16 decode and the scale shuffle run once for all of
    // them. The payload stays per element: qs and hmask live in separate
    // regions.
    let byte_base = (n_idx * (params.k / 256u) + k_base / 256u) * 110u;
    let e = k_base % 256u;
    let scale = q3k_d(byte_base) * q3k_scale(byte_base, e / 16u);
    var out: array<f32, 8>;
    for (var i = 0u; i < 8u; i++) {
        out[i] = scale * f32(q3k_quant(byte_base, e + i));
    }
    return out;
}
