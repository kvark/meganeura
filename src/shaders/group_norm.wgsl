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
var<storage> partials: array<f32>;  // (sum, M2) per slice
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

// Parallel-image GroupNorm, first of two passes: one workgroup per slice of
// a group writes (sum, M2), M2 being the slice's squared deviations from its
// own mean. Two reductions over the slice rather than (sum, Σx²): that form
// cancels for groups whose mean is large next to their spread.
// Dispatch: [N * num_groups * chunks, 1, 1]
@compute @workgroup_size(256)
fn stats(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let chunks = params.chunks;
    if wgid.x >= params.batch * params.num_groups * chunks { return; }
    let chunk = wgid.x % chunks;
    let ng = wgid.x / chunks;
    let n = ng / params.num_groups;
    let group = ng % params.num_groups;
    let tid = lid.x;
    let channels_per_group = params.channels / params.num_groups;
    let chunk_size = channels_per_group * params.spatial / chunks;
    // A group is contiguous in NCHW, so its slice is too.
    let base = (n * params.channels + group * channels_per_group) * params.spatial
        + chunk * chunk_size;

    var s = 0.0;
    var j = tid;
    loop {
        if j >= chunk_size { break; }
        s += src[base + j];
        j += 256u;
    }
    wg_data[tid] = s;
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
    let sum = wg_data[0];
    let mean = sum / f32(chunk_size);
    workgroupBarrier();

    var m2 = 0.0;
    j = tid;
    loop {
        if j >= chunk_size { break; }
        let d = src[base + j] - mean;
        m2 += d * d;
        j += 256u;
    }
    wg_data[tid] = m2;
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
        dst[wgid.x * 2u] = sum;
        dst[wgid.x * 2u + 1u] = wg_data[0];
    }
}

// Parallel-image GroupNorm, second of two passes. Each workgroup combines
// its group's (sum, M2) pairs exactly (Chan et al.: M2 = Σ M2ᵢ + nᵢ·(meanᵢ −
// mean)² over equal slices of nᵢ elements) and normalises the slice it owns.
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

    let chunk_size = group_size / chunks;
    var sum_val = 0.0;
    var p = tid;
    loop {
        if p >= chunks { break; }
        sum_val += partials[(ng * chunks + p) * 2u];
        p += 256u;
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
    let count = f32(group_size);
    let mean = wg_data[0] / count;
    workgroupBarrier();

    var m2_val = 0.0;
    p = tid;
    loop {
        if p >= chunks { break; }
        let chunk_mean = partials[(ng * chunks + p) * 2u] / f32(chunk_size);
        let d = chunk_mean - mean;
        m2_val += partials[(ng * chunks + p) * 2u + 1u] + f32(chunk_size) * d * d;
        p += 256u;
    }
    wg_data_sq[tid] = m2_val;
    workgroupBarrier();
    stride = 128u;
    loop {
        if stride == 0u { break; }
        if tid < stride {
            wg_data_sq[tid] += wg_data_sq[tid + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }

    let variance = wg_data_sq[0] / count;
    let inv_std = inverseSqrt(variance + eps);

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
