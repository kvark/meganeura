fn dot_q4_q8_packed(q4: u32, q8: u32) -> i32 {
    return dot4I8Packed(q4, q8);
}

fn dot_q8_q8_packed(a: u32, b: u32) -> i32 {
    return dot4I8Packed(a, b);
}

fn dot_qk_packed(a: u32, b: u32) -> i32 {
    return dot4I8Packed(a, b);
}
