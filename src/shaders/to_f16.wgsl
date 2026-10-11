// Cast f32 → f16, elementwise. Backward is the identity (mixed-precision
// straight-through): the f32 master weights stay the trainable parameter;
// this produces the f16 copy a bandwidth-bound forward read (embedding_f16)
// consumes.
struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

var<storage> src: array<f32>;
var<storage, read_write> dst: array<u32>;
var<uniform> params: Params;

// Round to nearest, ties to even, independently of the driver's half packing.
fn half_bits(value: f32) -> u32 {
    let bits = bitcast<u32>(value);
    let sign = (bits >> 16u) & 0x8000u;
    let exponent = (bits >> 23u) & 0xffu;
    let mantissa = bits & 0x7fffffu;
    if exponent == 255u {
        if mantissa != 0u { return sign | 0x7e00u | (mantissa >> 13u); }
        return sign | 0x7c00u;
    }
    if exponent > 142u { return sign | 0x7c00u; }
    if exponent < 102u { return sign; }
    if exponent < 113u {
        let significand = mantissa | 0x800000u;
        let shift = 126u - exponent;
        let bias = (1u << (shift - 1u)) - 1u;
        return sign | ((significand + bias + ((significand >> shift) & 1u)) >> shift);
    }
    let rounded = (bits & 0x7fffffffu) + 0xfffu + ((bits >> 13u) & 1u);
    return sign | ((rounded - 0x38000000u) >> 13u);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x * 2u;
    if i >= params.len { return; }
    var packed = half_bits(src[i]);
    if i + 1u < params.len {
        packed |= half_bits(src[i + 1u]) << 16u;
    }
    dst[gid.x] = packed;
}
