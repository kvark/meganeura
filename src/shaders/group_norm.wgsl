// GroupNorm forward: input[N, C, H, W] → output[N, C, H, W]
// Groups channels into num_groups sets, normalizes per (n, group).
// Dispatch: [N * num_groups, 1, 1]  workgroup_size(256)
//
// weight[C], bias[C] are per-channel scale and shift (in src_b and bias buffers).

struct Params {
    batch: u32,
    channels: u32,
    spatial: u32,       // H * W
    num_groups: u32,
    eps_bits: u32,
    chunks: u32,
    apply_silu: u32,
    _pad2: u32,
}

var<storage> src: array<f32>;
var<storage> src_b: array<f32>;     // weight[C]
var<storage> bias: array<f32>;      // bias[C]
var<storage, read_write> dst: array<f32>;
var<storage> partials: array<f32>;  // (sum, sumsq) per slice
var<uniform> params: Params;
var<workgroup> wg_data: array<f32, 256>;
var<workgroup> wg_data_sq: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let ng = wgid.x;  // n * num_groups + group
    if ng >= params.batch * params.num_groups { return; }

    let n = ng / params.num_groups;
    let group = ng % params.num_groups;
    let tid = lid.x;
    let eps = bitcast<f32>(params.eps_bits);

    let channels_per_group = params.channels / params.num_groups;
    let group_size = channels_per_group * params.spatial;  // elements per (n, group)
    let c_start = group * channels_per_group;

    // Phase 1: compute mean via strided accumulation
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

    // Tree reduction for mean
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

    // Phase 3: normalize, scale, shift
    j = tid;
    loop {
        if j >= group_size { break; }
        let c_local = j / params.spatial;
        let hw = j % params.spatial;
        let c = c_start + c_local;
        let idx = ((n * params.channels + c) * params.spatial) + hw;
        let normalized = (src[idx] - mean) * inv_std;
        dst[idx] = normalized * src_b[c] + bias[c];
        j += 256u;
    }
}

// Parallel-image GroupNorm, second of two passes. The generated reduction
// writes a (sum, sumsq) pair for every slice; each workgroup combines its
// group's pairs and normalises the slice it owns.
// Dispatch: [N * num_groups * chunks, 1, 1]
@compute @workgroup_size(256)
fn apply(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let chunks = params.chunks;
    let total_slices = params.batch * params.num_groups * chunks;
    if wgid.x >= total_slices { return; }

    let chunk = wgid.x % chunks;
    let ng = wgid.x / chunks;
    let n = ng / params.num_groups;
    let group = ng % params.num_groups;
    let tid = lid.x;
    let eps = bitcast<f32>(params.eps_bits);

    let channels_per_group = params.channels / params.num_groups;
    let group_size = channels_per_group * params.spatial;
    let c_start = group * channels_per_group;

    var sum_val = 0.0;
    var sumsq_val = 0.0;
    var p = tid;
    loop {
        if p >= chunks { break; }
        sum_val += partials[(ng * chunks + p) * 2u];
        sumsq_val += partials[(ng * chunks + p) * 2u + 1u];
        p += 256u;
    }
    wg_data[tid] = sum_val;
    wg_data_sq[tid] = sumsq_val;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if stride == 0u { break; }
        if tid < stride {
            wg_data[tid] += wg_data[tid + stride];
            wg_data_sq[tid] += wg_data_sq[tid + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }

    let count = f32(group_size);
    let mean = wg_data[0] / count;
    let variance = max(wg_data_sq[0] / count - mean * mean, 0.0);
    let inv_std = inverseSqrt(variance + eps);

    let chunk_size = group_size / chunks;
    let begin = chunk * chunk_size;
    let end = begin + chunk_size;
    var j = begin + tid;
    loop {
        if j >= end { break; }
        let c_local = j / params.spatial;
        let hw = j % params.spatial;
        let c = c_start + c_local;
        let idx = ((n * params.channels + c) * params.spatial) + hw;
        var v = (src[idx] - mean) * inv_std * src_b[c] + bias[c];
        if params.apply_silu != 0u {
            v = v / (1.0 + exp(-v));
        }
        dst[idx] = v;
        j += 256u;
    }
}
