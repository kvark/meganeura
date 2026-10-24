// One binary scale per reduction block. Unscale each partial matrix product
// before accumulating across blocks in F32; scales must not vary within a dot.
// Precision is relative to the block maximum, not each row: much smaller rows
// can lose mantissa bits or underflow. Whole-model qualification bounds the error.
var<workgroup> scale_maxima: array<vec2<f32>, 32>;

fn exponent_for(maximum: f32) -> i32 {
    let bits = bitcast<u32>(maximum) & 0x7fffffffu;
    if bits == 0u {
        return 0; // An all-zero block uses a scale of one.
    }
    let exponent = i32((bits >> 23u) & 255u);
    if exponent == 0 {
        return 163 - i32(firstLeadingBit(bits));
    }
    return 141 - exponent; // 127 + 14: scaled maximum is in [2^14, 2^15).
}

fn scale_power(value: f32, exponent: i32) -> f32 {
    if exponent >= -120 && exponent <= 120 {
        return value * bitcast<f32>(u32(exponent + 127) << 23u);
    }
    // Split extreme scales into normal F32 factors, including when restoring
    // the sum of two operand exponents after a matrix product.
    let first = clamp(exponent, -120, 120);
    let second = clamp(exponent - first, -120, 120);
    let third = exponent - first - second;
    let a = bitcast<f32>(u32(first + 127) << 23u);
    let b = bitcast<f32>(u32(second + 127) << 23u);
    let c = bitcast<f32>(u32(third + 127) << 23u);
    return ((value * a) * b) * c;
}

// The caller must reach a workgroup barrier before calling this again,
// so all reads of scale_maxima finish before another reduction overwrites it.
fn block_exponents(local: vec2<f32>, subgroup: u32, subgroup_width: u32, subgroup_lane: u32) -> vec2<i32> {
    let largest = subgroupMax(local);
    if subgroup_lane == 0u {
        scale_maxima[subgroup] = largest;
    }
    workgroupBarrier();
    var maximum = vec2<f32>(0.0);
    for (var group = 0u; group < 128u / subgroup_width; group++) {
        maximum = max(maximum, scale_maxima[group]);
    }
    return vec2<i32>(exponent_for(maximum.x), exponent_for(maximum.y));
}
