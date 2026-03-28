struct Params {
    rows: u32,
    cols: u32,
    eps_bits: u32,
    _pad: u32,
}

var<storage> src: array<f32>;
var<storage> src_b: array<f32>;  // weight
var<storage> bias: array<f32>;   // bias
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if row >= params.rows { return; }
    let offset = row * params.cols;
    let eps = bitcast<f32>(params.eps_bits);

    // Pass 1: Compute mean
    var sum = 0.0;
    for (var j = 0u; j < params.cols; j++) {
        sum += src[offset + j];
    }
    let mean = sum / f32(params.cols);

    // Pass 2: Compute variance
    var var_sum = 0.0;
    for (var j = 0u; j < params.cols; j++) {
        let diff = src[offset + j] - mean;
        var_sum += diff * diff;
    }
    let rstd = inverseSqrt(var_sum / f32(params.cols) + eps);

    // Pass 3: Normalize and apply affine
    for (var j = 0u; j < params.cols; j++) {
        let normed = (src[offset + j] - mean) * rstd;
        dst[offset + j] = normed * src_b[j] + bias[j];
    }
}
