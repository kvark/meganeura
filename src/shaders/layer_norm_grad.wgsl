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

// grad_weight[j] = sum_i(dy[i,j] * normed[i,j]). Each workgroup handles
// `_pad` consecutive rows (a power of two, at most 32); every row of the
// block gets 256 / block lanes, so all rows' mean and rstd take one tree
// each. It then writes dst[block * cols + j] = sum over its rows of
// dy * normed. A SumRows dispatch reduces the blocks unless there is only
// one. The bias gradient is a plain SumRows of dy.
//
// Dispatch: [ceil(rows / block), 1, 1], workgroup_size(256)
const MAX_BLOCK: u32 = 32u;
var<workgroup> row_mean: array<f32, MAX_BLOCK>;
var<workgroup> row_rstd: array<f32, MAX_BLOCK>;

@compute @workgroup_size(256)
fn layer_norm_grad_wb(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let cols = params.n;
    let eps = bitcast<f32>(params.k);
    let block = clamp(params._pad, 1u, MAX_BLOCK);
    let row_begin = wgid.x * block;
    let row_end = min(row_begin + block, params.m);

    // Lanes [slot * lanes, (slot + 1) * lanes) reduce row `row_begin + slot`.
    let lanes = 256u / block;
    let slot = tid / lanes;
    let lane = tid % lanes;
    let row = row_begin + slot;
    let live = row < row_end;
    let offset = row * cols;

    // Cooperative mean: grid-stride sum
    var s = 0.0;
    if live {
        var j = lane;
        loop {
            if j >= cols { break; }
            s += src_b[offset + j];
            j += lanes;
        }
    }
    wg_data[tid] = s;
    workgroupBarrier();
    // A fixed eight-step tree; steps wider than a row's lane group idle.
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if stride < lanes && lane < stride { wg_data[tid] += wg_data[tid + stride]; }
        workgroupBarrier();
    }
    let mean = wg_data[tid - lane] / f32(cols);
    workgroupBarrier();

    // Cooperative variance
    var v = 0.0;
    if live {
        var j = lane;
        loop {
            if j >= cols { break; }
            let diff = src_b[offset + j] - mean;
            v += diff * diff;
            j += lanes;
        }
    }
    wg_data[tid] = v;
    workgroupBarrier();
    // A fixed eight-step tree; steps wider than a row's lane group idle.
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if stride < lanes && lane < stride { wg_data[tid] += wg_data[tid + stride]; }
        workgroupBarrier();
    }
    if lane == 0u && live {
        row_mean[slot] = mean;
        row_rstd[slot] = inverseSqrt(wg_data[tid] / f32(cols) + eps);
    }
    workgroupBarrier();

    var j = tid;
    loop {
        if j >= cols { break; }
        var acc = 0.0;
        // A constant bound keeps `k` a plain unrolled index: lavapipe's
        // threaded JIT crashes on a shared-array index that follows a
        // runtime loop bound.
        for (var k = 0u; k < MAX_BLOCK; k++) {
            let r = row_begin + k;
            if r >= row_end { break; }
            let index = r * cols + j;
            acc += src_a[index] * (src_b[index] - row_mean[k]) * row_rstd[k];
        }
        dst[wgid.x * cols + j] = acc;
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
