fn dot_q4_q8_packed(q4: u32, q8: u32) -> i32 {
    var sum: i32 = 0;
    for (var shift = 0u; shift < 32u; shift += 8u) {
        let w = i32((q4 >> shift) & 0xFFu);
        let byte = i32((q8 >> shift) & 0xFFu);
        let a = select(byte, byte - 256, byte >= 128);
        sum += w * a;
    }
    return sum;
}

fn dot_q8_q8_packed(a: u32, b: u32) -> i32 {
    var sum: i32 = 0;
    for (var shift = 0u; shift < 32u; shift += 8u) {
        let x = i32((a >> shift) & 0xFFu);
        let y = i32((b >> shift) & 0xFFu);
        sum += select(x, x - 256, x >= 128) * select(y, y - 256, y >= 128);
    }
    return sum;
}

fn dot_qk_packed(a: u32, b: u32) -> i32 {
    var sum: i32 = 0;
    for (var shift = 0u; shift < 32u; shift += 8u) {
        let x = i32((a >> shift) & 0xFFu);
        let y = i32((b >> shift) & 0xFFu);
        sum += select(x, x - 256, x >= 128) * select(y, y - 256, y >= 128);
    }
    return sum;
}
