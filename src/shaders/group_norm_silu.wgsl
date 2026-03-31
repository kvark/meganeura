// Fused GroupNorm + SiLU forward: input[N, C, H, W] → output[N, C, H, W]
// Same as group_norm.wgsl but applies SiLU(x) = x / (1 + exp(-x)) after normalization.
// Eliminates one intermediate buffer read/write and one dispatch barrier.
// Dispatch: [N * num_groups, 1, 1]  workgroup_size(256)

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

var<storage> src: array<f32>;
var<storage> src_b: array<f32>;     // weight[C]
var<storage> bias: array<f32>;      // bias[C]
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;
var<workgroup> wg_data: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let ng = wgid.x;
    if ng >= params.batch * params.num_groups { return; }

    let n = ng / params.num_groups;
    let group = ng % params.num_groups;
    let tid = lid.x;
    let eps = bitcast<f32>(params.eps_bits);

    let channels_per_group = params.channels / params.num_groups;
    let group_size = channels_per_group * params.spatial;
    let c_start = group * channels_per_group;

    // Phase 1: compute mean
    var sum_val = 0.0;
    var j = tid;
    loop {
        if j >= group_size { break; }
        let c_local = j / params.spatial;
        let hw = j % params.spatial;
        let c = c_start + c_local;
        let idx = ((n * params.channels + c) * params.spatial) + hw;
        sum_val += src[idx];
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

    // Phase 2: compute variance
    var var_val = 0.0;
    j = tid;
    loop {
        if j >= group_size { break; }
        let c_local = j / params.spatial;
        let hw = j % params.spatial;
        let c = c_start + c_local;
        let idx = ((n * params.channels + c) * params.spatial) + hw;
        let d = src[idx] - mean;
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
    let variance = wg_data[0] / f32(group_size);
    let inv_std = inverseSqrt(variance + eps);

    // Phase 3: normalize, scale, shift, then SiLU
    j = tid;
    loop {
        if j >= group_size { break; }
        let c_local = j / params.spatial;
        let hw = j % params.spatial;
        let c = c_start + c_local;
        let idx = ((n * params.channels + c) * params.spatial) + hw;
        let normalized = (src[idx] - mean) * inv_std;
        let x = normalized * src_b[c] + bias[c];
        dst[idx] = x / (1.0 + exp(-x));  // SiLU
        j += 256u;
    }
}
