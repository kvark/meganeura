// GroupNorm backward w.r.t. input.
// Dispatch: [N * num_groups, 1, 1]  workgroup_size(256)
// grad_input[i] = inv_std * (w[c] * dout[i] - mean(w*dout) - xhat[i] * mean(w*dout*xhat)) / 1
// where xhat = (x - mean) * inv_std
//
// Inputs: grad_out (src_a), input (src_b), weight (bias), grad_input (dst)
// Params encode the same as forward.

struct Params {
    batch: u32,
    channels: u32,
    spatial: u32,
    num_groups: u32,
    eps_bits: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

var<storage> src_a: array<f32>;     // grad_output
var<storage> src_b: array<f32>;     // input x
var<storage> bias: array<f32>;      // weight[C]
var<storage, read_write> dst: array<f32>;  // grad_input
var<uniform> params: Params;
var<workgroup> wg_data: array<f32, 256>;
var<workgroup> wg_data2: array<f32, 256>;

@compute @workgroup_size(256)
fn grad_input(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let ng = wgid.x;
    if ng >= params.batch * params.num_groups { return; }

    let n = ng / params.num_groups;
    let group = ng % params.num_groups;
    let tid = lid.x;
    let eps = bitcast<f32>(params.eps_bits);

    let channels_per_group = params.channels / params.num_groups;
    let group_size = channels_per_group * params.spatial;
    let c_start = group * channels_per_group;

    // Pass 1: compute mean and variance of x within group
    var sum_x = 0.0;
    var sum_x2 = 0.0;
    var j = tid;
    loop {
        if j >= group_size { break; }
        let c_local = j / params.spatial;
        let hw = j % params.spatial;
        let c = c_start + c_local;
        let idx = ((n * params.channels + c) * params.spatial) + hw;
        let v = src_b[idx];
        sum_x += v;
        sum_x2 += v * v;
        j += 256u;
    }
    wg_data[tid] = sum_x;
    wg_data2[tid] = sum_x2;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if stride == 0u { break; }
        if tid < stride {
            wg_data[tid] += wg_data[tid + stride];
            wg_data2[tid] += wg_data2[tid + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }
    let mean = wg_data[0] / f32(group_size);
    let variance = wg_data2[0] / f32(group_size) - mean * mean;
    let inv_std = inverseSqrt(variance + eps);
    workgroupBarrier();

    // Pass 2: compute sum(w * dout) and sum(w * dout * xhat) within group
    var sum_wdy = 0.0;
    var sum_wdy_xhat = 0.0;
    j = tid;
    loop {
        if j >= group_size { break; }
        let c_local = j / params.spatial;
        let hw = j % params.spatial;
        let c = c_start + c_local;
        let idx = ((n * params.channels + c) * params.spatial) + hw;
        let dy = src_a[idx];
        let w = bias[c];
        let xhat = (src_b[idx] - mean) * inv_std;
        sum_wdy += w * dy;
        sum_wdy_xhat += w * dy * xhat;
        j += 256u;
    }
    wg_data[tid] = sum_wdy;
    wg_data2[tid] = sum_wdy_xhat;
    workgroupBarrier();

    stride = 128u;
    loop {
        if stride == 0u { break; }
        if tid < stride {
            wg_data[tid] += wg_data[tid + stride];
            wg_data2[tid] += wg_data2[tid + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }
    let mean_wdy = wg_data[0] / f32(group_size);
    let mean_wdy_xhat = wg_data2[0] / f32(group_size);

    // Pass 3: compute grad_input
    j = tid;
    loop {
        if j >= group_size { break; }
        let c_local = j / params.spatial;
        let hw = j % params.spatial;
        let c = c_start + c_local;
        let idx = ((n * params.channels + c) * params.spatial) + hw;
        let dy = src_a[idx];
        let w = bias[c];
        let xhat = (src_b[idx] - mean) * inv_std;
        dst[idx] = inv_std * (w * dy - mean_wdy - xhat * mean_wdy_xhat);
        j += 256u;
    }
}

// GroupNorm backward w.r.t. weight and bias.
// Dispatch: [C, 1, 1]  workgroup_size(1)
// grad_weight[c] = sum_{n,hw} grad_out[n,c,hw] * xhat[n,c,hw]
// grad_bias[c] = sum_{n,hw} grad_out[n,c,hw]
// This is a simple serial accumulation per channel (can be parallelized later).
// dst layout: [grad_weight[C], grad_bias[C]] = 2*C elements

@compute @workgroup_size(1)
fn grad_weight_bias(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if c >= params.channels { return; }

    let eps = bitcast<f32>(params.eps_bits);
    let channels_per_group = params.channels / params.num_groups;
    let group = c / channels_per_group;

    var sum_dw = 0.0;
    var sum_db = 0.0;

    for (var n = 0u; n < params.batch; n++) {
        // Compute mean and variance for this (n, group)
        let c_start = group * channels_per_group;
        let group_size = channels_per_group * params.spatial;
        var sum_x = 0.0;
        var sum_x2 = 0.0;
        for (var j = 0u; j < group_size; j++) {
            let cc = c_start + j / params.spatial;
            let hw = j % params.spatial;
            let idx = ((n * params.channels + cc) * params.spatial) + hw;
            let v = src_b[idx];
            sum_x += v;
            sum_x2 += v * v;
        }
        let mean = sum_x / f32(group_size);
        let variance = sum_x2 / f32(group_size) - mean * mean;
        let inv_std = inverseSqrt(variance + eps);

        for (var hw = 0u; hw < params.spatial; hw++) {
            let idx = ((n * params.channels + c) * params.spatial) + hw;
            let dy = src_a[idx];
            let xhat = (src_b[idx] - mean) * inv_std;
            sum_dw += dy * xhat;
            sum_db += dy;
        }
    }

    dst[c] = sum_dw;
    dst[params.channels + c] = sum_db;
}
