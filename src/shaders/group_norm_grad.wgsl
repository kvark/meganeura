// GroupNorm backward, in three entry points sharing one module.
//
//   grad_stats:       (mean, inv_std) per (n, group) → stats buffer (as dst)
//   grad_input:       grad wrt input, reading those statistics
//   grad_weight_bias: grad wrt weight and bias, reading those statistics
//
// Computing the statistics once matters most for the weight/bias gradient:
// it runs one workgroup per channel, and deriving the group statistics
// there re-read the whole group for every channel in it.
//
// Inputs: grad_out (src_a), input (src_b), weight (bias), stats.
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
var<storage, read_write> dst: array<f32>;
var<storage> stats: array<f32>;     // (mean, inv_std) per (n, group)
var<uniform> params: Params;
var<workgroup> wg_data: array<f32, 256>;
var<workgroup> wg_data2: array<f32, 256>;

// Same two-pass mean and variance, in the same order, as the single-pass
// forward kernel, so both directions agree on inv_std.
// Dispatch: [N * num_groups, 1, 1]
@compute @workgroup_size(256)
fn grad_stats(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let ng = wgid.x;
    if ng >= params.batch * params.num_groups { return; }

    let n = ng / params.num_groups;
    let group = ng % params.num_groups;
    let tid = lid.x;
    let eps = bitcast<f32>(params.eps_bits);
    let channels_per_group = params.channels / params.num_groups;
    let group_size = channels_per_group * params.spatial;
    // Each group is contiguous in NCHW.
    let base = (n * params.channels + group * channels_per_group) * params.spatial;

    var sum_val = 0.0;
    var j = tid;
    loop {
        if j >= group_size { break; }
        sum_val += src_b[base + j];
        j += 256u;
    }
    wg_data[tid] = sum_val;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if stride == 0u { break; }
        if tid < stride {
            wg_data[tid] += wg_data[tid + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }
    let mean = wg_data[0] / f32(group_size);
    workgroupBarrier();

    var var_val = 0.0;
    j = tid;
    loop {
        if j >= group_size { break; }
        let d = src_b[base + j] - mean;
        var_val += d * d;
        j += 256u;
    }
    wg_data[tid] = var_val;
    workgroupBarrier();
    stride = 128u;
    loop {
        if stride == 0u { break; }
        if tid < stride {
            wg_data[tid] += wg_data[tid + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }
    if tid == 0u {
        dst[ng * 2u] = mean;
        dst[ng * 2u + 1u] = inverseSqrt(wg_data[0] / f32(group_size) + eps);
    }
}

// grad_input[i] = inv_std * (w[c] * dout[i] - mean(w*dout) - xhat[i] * mean(w*dout*xhat))
// where xhat = (x - mean) * inv_std.
// Dispatch: [N * num_groups, 1, 1]
@compute @workgroup_size(256)
fn grad_input(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let ng = wgid.x;
    if ng >= params.batch * params.num_groups { return; }

    let n = ng / params.num_groups;
    let group = ng % params.num_groups;
    let tid = lid.x;

    let channels_per_group = params.channels / params.num_groups;
    let group_size = channels_per_group * params.spatial;
    let c_start = group * channels_per_group;

    let mean = stats[ng * 2u];
    let inv_std = stats[ng * 2u + 1u];

    // Pass 1: compute sum(w * dout) and sum(w * dout * xhat) within group
    var sum_wdy = 0.0;
    var sum_wdy_xhat = 0.0;
    var j = tid;
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
    let mean_wdy = wg_data[0] / f32(group_size);
    let mean_wdy_xhat = wg_data2[0] / f32(group_size);

    // Pass 2: compute grad_input
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
// Dispatch: [C, 1, 1]  workgroup_size(256)
// grad_weight[c] = sum_{n,hw} grad_out[n,c,hw] * xhat[n,c,hw]
// grad_bias[c] = sum_{n,hw} grad_out[n,c,hw]
// dst layout: [grad_weight[C], grad_bias[C]] = 2*C elements
//
// Each workgroup handles one channel c and reads only that channel's planes.
@compute @workgroup_size(256)
fn grad_weight_bias(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let c = wgid.x;
    if c >= params.channels { return; }

    let tid = lid.x;
    let channels_per_group = params.channels / params.num_groups;
    let group = c / channels_per_group;

    var local_dw = 0.0;
    var local_db = 0.0;
    for (var n = 0u; n < params.batch; n++) {
        let ng = n * params.num_groups + group;
        let mean = stats[ng * 2u];
        let inv_std = stats[ng * 2u + 1u];
        let base = (n * params.channels + c) * params.spatial;
        var j = tid;
        loop {
            if j >= params.spatial { break; }
            let dy = src_a[base + j];
            local_dw += dy * (src_b[base + j] - mean) * inv_std;
            local_db += dy;
            j += 256u;
        }
    }
    wg_data[tid] = local_dw;
    wg_data2[tid] = local_db;
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

    if tid == 0u {
        dst[c] = wg_data[0];
        dst[params.channels + c] = wg_data2[0];
    }
}
