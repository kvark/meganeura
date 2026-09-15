fn decode_f16(bits: u32) -> f32 {
    return unpack2x16float(bits & 0xFFFFu).x;
}

// Both halves of a packed (low, high) scale pair in one instruction.
fn decode_f16_pair(bits: u32) -> vec2<f32> {
    return unpack2x16float(bits);
}
