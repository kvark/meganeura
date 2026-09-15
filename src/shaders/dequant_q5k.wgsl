// The 5-bit quant for element `e`: nibble from qs, plus 16 if this
// sub-block's bit is set in qh.
fn q5k_quant(base: u32, e: u32) -> u32 {
    let l = e % 32u;
    let byte = kq_byte(base, 48u + (e / 64u) * 32u + l);
    let nib = select(byte >> 4u, byte & 0xFu, (e % 64u) < 32u);
    let hi = (kq_byte(base, 16u + l) >> (e / 32u)) & 1u;
    return nib + hi * 16u;
}

fn dequant_q5k(k_idx: u32, n_idx: u32) -> f32 {
    let base = (n_idx * (params.k / 256u) + k_idx / 256u) * 44u;
    let hdr = decode_f16_pair(matrix_b[base]);
    let e = k_idx % 256u;
    let sm = kq_scale_min(base, e / 32u);
    return hdr.x * sm.x * f32(q5k_quant(base, e)) - hdr.y * sm.y;
}

fn dequant_q5k_pack8(k_base: u32, n_idx: u32) -> array<f32, 8> {
    // Eight 8-aligned elements share a sub-block, so the header and the
    // scale pair are read once. The payload stays per element: the nibble
    // and its high bit come from two separate regions.
    let base = (n_idx * (params.k / 256u) + k_base / 256u) * 44u;
    let hdr = decode_f16_pair(matrix_b[base]);
    let e = k_base % 256u;
    let sm = kq_scale_min(base, e / 32u);
    let scale = hdr.x * sm.x;
    let offset = hdr.y * sm.y;
    var out: array<f32, 8>;
    for (var i = 0u; i < 8u; i++) {
        out[i] = scale * f32(q5k_quant(base, e + i)) - offset;
    }
    return out;
}
