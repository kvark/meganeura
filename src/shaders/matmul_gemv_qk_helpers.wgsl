$F16_DECODE_FN
$K_SCALE_MIN_FN

// Q4_K: word-aligned 144-byte superblocks and paired nibbles.
fn q4k_block_dot(
    col: u32,
    blk: u32,
    u: ptr<function, array<u32, 8u>>,
    d8: f32,
    s8: f32,
) -> f32 {
    let sblocks = params.k / 256u;
    let base = (col * sblocks + blk / 8u) * 36u;
    let hdr = decode_f16_pair(matrix_b[base]);
    let sm = kq_scale_min(base, blk % 8u);
    let even = (blk % 2u) == 0u;
    let qbase = base * 4u + 16u + ((blk % 8u) / 2u) * 32u;
    var sumi = 0;
    for (var i = 0u; i < 4u; i++) {
        let w_lo = kq_word4(qbase, i * 4u);
        let w_hi = kq_word4(qbase, 16u + i * 4u);
        let v_lo = select((w_lo >> 4u) & 0x0F0F0F0Fu, w_lo & 0x0F0F0F0Fu, even);
        let v_hi = select((w_hi >> 4u) & 0x0F0F0F0Fu, w_hi & 0x0F0F0F0Fu, even);
        sumi += dot_qk_packed(v_lo, (*u)[i]) + dot_qk_packed(v_hi, (*u)[4u + i]);
    }
    return hdr.x * sm.x * (d8 * f32(sumi)) - hdr.y * sm.y * s8;
}

// Q5_K is Q4_K plus one high bit per quant.
fn q5k_block_dot(
    col: u32,
    blk: u32,
    u: ptr<function, array<u32, 8u>>,
    d8: f32,
    s8: f32,
) -> f32 {
    let sblocks = params.k / 256u;
    let base = (col * sblocks + blk / 8u) * 44u;
    let hdr = decode_f16_pair(matrix_b[base]);
    let sm = kq_scale_min(base, blk % 8u);
    let even = (blk % 2u) == 0u;
    let hshift = blk % 8u;
    let qhbase = base * 4u + 16u;
    let qsbase = base * 4u + 48u + ((blk % 8u) / 2u) * 32u;
    var sumi = 0;
    for (var i = 0u; i < 4u; i++) {
        let wl_lo = kq_word4(qsbase, i * 4u);
        let wl_hi = kq_word4(qsbase, 16u + i * 4u);
        let nib_lo = select((wl_lo >> 4u) & 0x0F0F0F0Fu, wl_lo & 0x0F0F0F0Fu, even);
        let nib_hi = select((wl_hi >> 4u) & 0x0F0F0F0Fu, wl_hi & 0x0F0F0F0Fu, even);
        let hb_lo = (kq_word4(qhbase, i * 4u) >> hshift) & 0x01010101u;
        let hb_hi = (kq_word4(qhbase, 16u + i * 4u) >> hshift) & 0x01010101u;
        sumi += dot_qk_packed(nib_lo | (hb_lo << 4u), (*u)[i]);
        sumi += dot_qk_packed(nib_hi | (hb_hi << 4u), (*u)[4u + i]);
    }
    return hdr.x * sm.x * (d8 * f32(sumi)) - hdr.y * sm.y * s8;
}

// Q6_K: two 16-element sub-blocks per activation block.
fn q6k_scale(base: u32, i: u32) -> f32 {
    let raw = kq_byte_at(base, 192u + i);
    return f32(i32(raw) - select(0, 256, raw >= 128u));
}

fn q6k_block_dot(
    col: u32,
    blk: u32,
    u: ptr<function, array<u32, 8u>>,
    d8: f32,
    s8: f32,
) -> f32 {
    let sblocks = params.k / 256u;
    let base = (col * sblocks + blk / 8u) * 210u;
    let j8 = blk % 8u;
    let half = j8 / 4u;
    let j6 = j8 % 4u;
    let qlbase = base + half * 64u + (j6 & 1u) * 32u;
    let qhbase = base + 128u + half * 32u;
    var sumi_a = 0;
    var sumi_b = 0;
    var usum_a = 0;
    var usum_b = 0;
    for (var i = 0u; i < 4u; i++) {
        let la = kq_word4(qlbase, i * 4u);
        let lb = kq_word4(qlbase, 16u + i * 4u);
        let lo_a = select((la >> 4u) & 0x0F0F0F0Fu, la & 0x0F0F0F0Fu, j6 < 2u);
        let lo_b = select((lb >> 4u) & 0x0F0F0F0Fu, lb & 0x0F0F0F0Fu, j6 < 2u);
        let hi_a = ((kq_word4(qhbase, i * 4u) >> (j6 * 2u)) & 0x03030303u) << 4u;
        let hi_b = ((kq_word4(qhbase, 16u + i * 4u) >> (j6 * 2u)) & 0x03030303u) << 4u;
        sumi_a += dot_qk_packed(lo_a | hi_a, (*u)[i]);
        sumi_b += dot_qk_packed(lo_b | hi_b, (*u)[4u + i]);
        usum_a += dot_qk_packed(0x01010101u, (*u)[i]);
        usum_b += dot_qk_packed(0x01010101u, (*u)[4u + i]);
    }
    let d = decode_f16(kq_byte_at(base, 208u) | (kq_byte_at(base, 209u) << 8u));
    let sc_a = d * q6k_scale(base, half * 8u + j6 * 2u);
    let sc_b = d * q6k_scale(base, half * 8u + j6 * 2u + 1u);
    return sc_a * (d8 * f32(sumi_a - 32 * usum_a))
        + sc_b * (d8 * f32(sumi_b - 32 * usum_b));
}

// Q3_K: the hmask bit is inverted, so a clear bit subtracts four.
fn q3k_scale(base: u32, i: u32) -> f32 {
    let b = i % 4u;
    let g = i / 4u;
    let src = select(4u + b, b, (g % 2u) == 0u);
    let raw = kq_byte_at(base, 96u + src);
    let nib = select(raw >> 4u, raw & 0xFu, g < 2u);
    let hi = (kq_byte_at(base, 96u + 8u + b) >> (g * 2u)) & 3u;
    return f32(i32(nib | (hi << 4u)) - 32);
}

fn q3k_block_dot(
    col: u32,
    blk: u32,
    u: ptr<function, array<u32, 8u>>,
    d8: f32,
    s8: f32,
) -> f32 {
    let sblocks = params.k / 256u;
    let base = (col * sblocks + blk / 8u) * 110u;
    let j8 = blk % 8u;
    let h = j8 / 4u;
    let j3 = j8 % 4u;
    let qshift = j3 * 2u;
    let hshift = blk % 8u;
    var sumi_a = 0;
    var sumi_b = 0;
    for (var i = 0u; i < 4u; i++) {
        let q_a = (kq_word4(base, 32u + h * 32u + i * 4u) >> qshift) & 0x03030303u;
        let q_b = (kq_word4(base, 32u + h * 32u + 16u + i * 4u) >> qshift) & 0x03030303u;
        let hb_a = (kq_word4(base, i * 4u) >> hshift) & 0x01010101u;
        let hb_b = (kq_word4(base, 16u + i * 4u) >> hshift) & 0x01010101u;
        sumi_a += dot_qk_packed(q_a, (*u)[i])
            - 4 * dot_qk_packed(hb_a ^ 0x01010101u, (*u)[i]);
        sumi_b += dot_qk_packed(q_b, (*u)[4u + i])
            - 4 * dot_qk_packed(hb_b ^ 0x01010101u, (*u)[4u + i]);
    }
    let d = decode_f16(kq_byte_at(base, 108u) | (kq_byte_at(base, 109u) << 8u));
    return d * q3k_scale(base, (blk % 8u) * 2u) * (d8 * f32(sumi_a))
        + d * q3k_scale(base, (blk % 8u) * 2u + 1u) * (d8 * f32(sumi_b));
}
