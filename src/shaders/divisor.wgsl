fn multiply_high(a: u32, b: u32) -> u32 {
    let a_low = a & 0xffffu;
    let a_high = a >> 16u;
    let b_low = b & 0xffffu;
    let b_high = b >> 16u;
    let low = a_low * b_low;
    let middle0 = a_high * b_low + (low >> 16u);
    let middle1 = a_low * b_high + (middle0 & 0xffffu);
    return a_high * b_high + (middle0 >> 16u) + (middle1 >> 16u);
}

// For d > 1, multiplier = floor(2^32 / d). The estimate is at most one low.
fn divide_exact(value: u32, divisor: u32, multiplier: u32) -> u32 {
    if divisor == 1u {
        return value;
    }
    let quotient = multiply_high(value, multiplier);
    let remainder = value - quotient * divisor;
    return quotient + u32(remainder >= divisor);
}
