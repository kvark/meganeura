// In-place scale a gradient buffer by `min(1, max_norm / sqrt(acc))`,
// reading `acc` (the global gradient-norm-squared accumulator filled
// by `grad_clip_norm_sq`).
//
struct Params {
    len: u32,
    max_norm: f32,
    _pad0: u32,
    _pad1: u32,
}

var<storage, read_write> grad: array<f32>;
var<storage> acc: array<f32>;
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.len { return; }

    let norm_sq = acc[0];
    // norm_sq could be negative due to f32 rounding edge cases or
    // partial NaN if a prior backward overflowed; guard with max(0).
    let norm = sqrt(max(norm_sq, 0.0));
    var scale: f32 = 1.0;
    if norm > params.max_norm && params.max_norm > 0.0 {
        scale = params.max_norm / norm;
    }
    grad[i] = grad[i] * scale;
}
