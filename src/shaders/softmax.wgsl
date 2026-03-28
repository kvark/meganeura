struct Params {
    batch: u32,
    features: u32,
    _pad0: u32,
    _pad1: u32,
}

var<storage> src: array<f32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if row >= params.batch { return; }
    let offset = row * params.features;

    // Find max for numerical stability
    var max_val = src[offset];
    for (var j = 1u; j < params.features; j++) {
        max_val = max(max_val, src[offset + j]);
    }

    // Compute exp(x - max) and sum
    var sum_exp = 0.0;
    for (var j = 0u; j < params.features; j++) {
        let exp_val = exp(src[offset + j] - max_val);
        dst[offset + j] = exp_val;
        sum_exp += exp_val;
    }

    // Normalize
    for (var j = 0u; j < params.features; j++) {
        dst[offset + j] /= sum_exp;
    }
}
