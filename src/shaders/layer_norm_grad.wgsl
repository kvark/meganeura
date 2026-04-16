// LayerNorm gradient shaders (parallel).
// Two entry points:
//   layer_norm_grad_wb: gradient wrt weight and bias — row-parallel
//   layer_norm_grad_x: gradient wrt input — 256-thread per row
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
var<workgroup> wg_data: array<f32, 256>;

// grad_weight[j] = sum_i(dy[i,j] * normed[i,j])
// grad_bias[j] = sum_i(dy[i,j])
// For row-parallel: each WG handles one row, writes partial grad_w and grad_bias.
// Output: dst[row * cols * 2 + j] = partial_grad_w, dst[row * cols * 2 + cols + j] = partial_grad_bias
//
// Dispatch: [rows, 1, 1], workgroup_size(256)
// Followed by a SumRows dispatch to reduce rows → final grad_w, grad_bias.
@compute @workgroup_size(256)
fn layer_norm_grad_wb(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wgid.x;
    let tid = lid.x;
    let cols = params.n;
    let eps = bitcast<f32>(params.k);
    let offset = row * cols;

    // Cooperative mean: grid-stride sum
    var s = 0.0;
    var j = tid;
    loop {
        if j >= cols { break; }
        s += src_b[offset + j];
        j += 256u;
    }
    wg_data[tid] = s;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if tid < stride { wg_data[tid] += wg_data[tid + stride]; }
        workgroupBarrier();
    }
    let mean = wg_data[0] / f32(cols);
    workgroupBarrier();

    // Cooperative variance
    var v = 0.0;
    j = tid;
    loop {
        if j >= cols { break; }
        let diff = src_b[offset + j] - mean;
        v += diff * diff;
        j += 256u;
    }
    wg_data[tid] = v;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if tid < stride { wg_data[tid] += wg_data[tid + stride]; }
        workgroupBarrier();
    }
    let rstd = inverseSqrt(wg_data[0] / f32(cols) + eps);

    // Write partial grad_w and grad_bias for this row
    let out_base = row * cols;
    j = tid;
    loop {
        if j >= cols { break; }
        let normed = (src_b[offset + j] - mean) * rstd;
        dst[out_base + j] = src_a[offset + j] * normed;
        j += 256u;
    }
}

// grad_x[i,j] = rstd * (dy[i,j]*w[j] - normed[i,j]*s_i - mean(dy*w)/cols)
// where s_i = sum_j(dy[i,j]*w[j]*normed[i,j]) / cols
//
// Dispatch: [rows, 1, 1], workgroup_size(256)
@compute @workgroup_size(256)
fn layer_norm_grad_x(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wgid.x;
    let tid = lid.x;
    let rows = params.m;
    let cols = params.n;
    let eps = bitcast<f32>(params.k);
    if row >= rows { return; }

    let offset = row * cols;

    // Phase 1: Cooperative mean
    var s = 0.0;
    var j = tid;
    loop {
        if j >= cols { break; }
        s += src_b[offset + j];
        j += 256u;
    }
    wg_data[tid] = s;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if tid < stride { wg_data[tid] += wg_data[tid + stride]; }
        workgroupBarrier();
    }
    let mean = wg_data[0] / f32(cols);
    workgroupBarrier();

    // Phase 2: Cooperative variance
    var v = 0.0;
    j = tid;
    loop {
        if j >= cols { break; }
        let diff = src_b[offset + j] - mean;
        v += diff * diff;
        j += 256u;
    }
    wg_data[tid] = v;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if tid < stride { wg_data[tid] += wg_data[tid + stride]; }
        workgroupBarrier();
    }
    let rstd = inverseSqrt(wg_data[0] / f32(cols) + eps);
    workgroupBarrier();

    // Phase 3: Cooperative dot products (dot_dy_w and dot_dy_w_norm)
    var d1 = 0.0;
    var d2 = 0.0;
    j = tid;
    loop {
        if j >= cols { break; }
        let dy_w = src_a[offset + j] * bias[j];
        let normed = (src_b[offset + j] - mean) * rstd;
        d1 += dy_w;
        d2 += dy_w * normed;
        j += 256u;
    }
    // Reduce d1 and d2 together
    wg_data[tid] = d1;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if tid < stride { wg_data[tid] += wg_data[tid + stride]; }
        workgroupBarrier();
    }
    let dot_dy_w = wg_data[0];
    workgroupBarrier();

    wg_data[tid] = d2;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if tid < stride { wg_data[tid] += wg_data[tid + stride]; }
        workgroupBarrier();
    }
    let dot_dy_w_norm = wg_data[0];

    // Phase 4: Write grad_x
    let inv_n = 1.0 / f32(cols);
    j = tid;
    loop {
        if j >= cols { break; }
        let normed = (src_b[offset + j] - mean) * rstd;
        let dy_w = src_a[offset + j] * bias[j];
        dst[offset + j] = rstd * (dy_w - inv_n * dot_dy_w - normed * inv_n * dot_dy_w_norm);
        j += 256u;
    }
}
