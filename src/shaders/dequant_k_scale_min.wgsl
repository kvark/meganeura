fn kq_byte(base: u32, off: u32) -> u32 {
    let w = matrix_b[base + off / 4u];
    return (w >> ((off % 4u) * 8u)) & 0xFFu;
}

fn kq_scale_min(base: u32, j: u32) -> vec2<f32> {
    var sc: u32;
    var mn: u32;
    if j < 4u {
        sc = kq_byte(base, 4u + j) & 63u;
        mn = kq_byte(base, 8u + j) & 63u;
    } else {
        let hi = kq_byte(base, j + 8u);
        sc = (hi & 0xFu) | ((kq_byte(base, j) >> 6u) << 4u);
        mn = (hi >> 4u) | ((kq_byte(base, 4u + j) >> 6u) << 4u);
    }
    return vec2<f32>(f32(sc), f32(mn));
}
