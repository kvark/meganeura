// LayerNorm gradient shaders.
// Two entry points:
//   layer_norm_grad_wb: gradient wrt weight and bias
//   layer_norm_grad_x: gradient wrt input
// Params: m=rows, n=cols, k=eps_bits

struct Params {
    m: u32,
    n: u32,
    k: u32,    // eps_bits
    _pad: u32,
}

var<storage> src_a: array<f32>;  // dy (grad_output)
var<storage> src_b: array<f32>;  // x (input)
var<storage> bias: array<f32>;   // w (weight)
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;

// grad_weight[j] = sum_i(dy[i,j] * normed[i,j])
// grad_bias[j] = sum_i(dy[i,j])
// Output layout: dst[0..cols] = grad_weight, dst[cols..2*cols] = grad_bias
//
// Dispatch: [cols, 1, 1]
@compute @workgroup_size(1)
fn layer_norm_grad_wb(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    let rows = params.m;
    let cols = params.n;
    let eps = bitcast<f32>(params.k);
    if j >= cols { return; }

    var grad_w = 0.0;

    for (var i = 0u; i < rows; i++) {
        let offset = i * cols;
        // Recompute mean and rstd for this row
        var sum = 0.0;
        for (var c = 0u; c < cols; c++) {
            sum += src_b[offset + c];
        }
        let mean = sum / f32(cols);
        var var_sum = 0.0;
        for (var c = 0u; c < cols; c++) {
            let diff = src_b[offset + c] - mean;
            var_sum += diff * diff;
        }
        let rstd = inverseSqrt(var_sum / f32(cols) + eps);
        let normed = (src_b[offset + j] - mean) * rstd;

        grad_w += src_a[offset + j] * normed;
    }

    dst[j] = grad_w;
}

// grad_x[i,j] = rstd * (dy[i,j]*w[j] - normed[i,j]*s_i - mean(dy*w)/cols)
// where s_i = sum_j(dy[i,j]*w[j]*normed[i,j]) / cols
//
// Dispatch: [rows, 1, 1]
@compute @workgroup_size(1)
fn layer_norm_grad_x(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let rows = params.m;
    let cols = params.n;
    let eps = bitcast<f32>(params.k);
    if row >= rows { return; }

    let offset = row * cols;

    // Recompute mean, rstd
    var sum = 0.0;
    for (var j = 0u; j < cols; j++) {
        sum += src_b[offset + j];
    }
    let mean = sum / f32(cols);
    var var_sum = 0.0;
    for (var j = 0u; j < cols; j++) {
        let diff = src_b[offset + j] - mean;
        var_sum += diff * diff;
    }
    let rstd = inverseSqrt(var_sum / f32(cols) + eps);

    // Compute dot products for the backward formula
    var dot_dy_w = 0.0;    // sum_j(dy[j] * w[j])
    var dot_dy_w_norm = 0.0; // sum_j(dy[j] * w[j] * normed[j])
    for (var j = 0u; j < cols; j++) {
        let dy_w = src_a[offset + j] * bias[j];
        let normed = (src_b[offset + j] - mean) * rstd;
        dot_dy_w += dy_w;
        dot_dy_w_norm += dy_w * normed;
    }

    // Write grad_x
    let inv_n = 1.0 / f32(cols);
    for (var j = 0u; j < cols; j++) {
        let normed = (src_b[offset + j] - mean) * rstd;
        let dy_w = src_a[offset + j] * bias[j];
        dst[offset + j] = rstd * (dy_w - inv_n * dot_dy_w - normed * inv_n * dot_dy_w_norm);
    }
}
