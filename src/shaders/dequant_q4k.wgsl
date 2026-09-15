fn dequant_q4k(k_idx: u32, n_idx: u32) -> f32 {
    let base = (n_idx * (params.k / 256u) + k_idx / 256u) * 36u;
    let hdr = decode_f16_pair(matrix_b[base]);
    let d = hdr.x;
    let dmin = hdr.y;
    let e = k_idx % 256u;
    let sm = kq_scale_min(base, e / 32u);
    let byte = kq_byte(base, 16u + (e / 64u) * 32u + e % 32u);
    let q = select(byte >> 4u, byte & 0xFu, (e % 64u) < 32u);
    return d * sm.x * f32(q) - dmin * sm.y;
}

fn dequant_q4k_pack8(k_base: u32, n_idx: u32) -> array<f32, 8> {
    let base = (n_idx * (params.k / 256u) + k_base / 256u) * 36u;
    let hdr = decode_f16_pair(matrix_b[base]);
    let d = hdr.x;
    let dmin = hdr.y;
    let e = k_base % 256u;
    let sm = kq_scale_min(base, e / 32u);
    let scale = d * sm.x;
    let offset = dmin * sm.y;
    // Eight 8-aligned elements share a sub-block and a nibble half, and
    // span eight consecutive bytes - two words, because GGML pairs e with
    // e + 32 rather than e with e + 1.
    let low = (e % 64u) < 32u;
    let wbase = base + (16u + (e / 64u) * 32u + e % 32u) / 4u;
    let w0 = matrix_b[wbase];
    let w1 = matrix_b[wbase + 1u];
    var out: array<f32, 8>;
    for (var i = 0u; i < 8u; i++) {
        let w = select(w1, w0, i < 4u);
        let byte = (w >> ((i % 4u) * 8u)) & 0xFFu;
        let q = select(byte >> 4u, byte & 0xFu, low);
        out[i] = scale * f32(q) - offset;
    }
    return out;
}
