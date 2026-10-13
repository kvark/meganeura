fn round_bf16(value: f32) -> bf16 {
    let bits = bitcast<u32>(value);
    var rounded = bits + 0x7fffu + ((bits >> 16u) & 1u);
    if (bits & 0x7fffffffu) > 0x7f800000u {
        rounded = bits | 0x00400000u;
    }
    return bf16(bitcast<f32>(rounded & 0xffff0000u));
}
