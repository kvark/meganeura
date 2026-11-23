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

const WG_SIZE: u32 = 128u;

// LayerNorm with subgroup reductions, for devices with subgroups: one
// workgroup per row as in `layer_norm.wgsl`, but each sum is a subgroup sum
// plus one exchange of the subgroup totals, two barriers where the tree takes
// eight. Two reductions, the mean and then the mean squared deviation (the
// one-pass form cancels for rows with a large mean).
var<workgroup> totals: array<f32, WG_SIZE>;

// The workgroup's sum of `value`. These one-dimensional workgroups fill
// consecutive subgroups, including a possible partial last one; count them
// from the width (some drivers report one subgroup when several run).
fn workgroup_sum(value: f32, wave_id: u32, sg_id: u32, wave_size: u32) -> f32 {
    let waves = (WG_SIZE + wave_size - 1u) / wave_size;
    let wave_total = subgroupAdd(value);
    if waves == 1u {
        return wave_total;
    }
    if sg_id == subgroupBroadcastFirst(sg_id) {
        totals[wave_id] = wave_total;
    }
    workgroupBarrier();
    var total = 0.0;
    for (var w = 0u; w < waves; w += 1u) {
        total += totals[w];
    }
    // The next sum reuses the totals.
    workgroupBarrier();
    return total;
}

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_id: u32,
    @builtin(subgroup_id) wave_id: u32,
    @builtin(subgroup_size) wave_size: u32,
) {
    let row = wgid.x;
    if row >= params.rows { return; }
    let offset = row * params.cols;
    let cols = params.cols;
    let eps = bitcast<f32>(params.eps_bits);
    let tid = lid.x;

    var s = 0.0;
    for (var j = tid; j < cols; j = j + WG_SIZE) {
        s += src[offset + j];
    }
    let mean = workgroup_sum(s, wave_id, sg_id, wave_size) / f32(cols);

    var ss = 0.0;
    for (var j = tid; j < cols; j = j + WG_SIZE) {
        let d = src[offset + j] - mean;
        ss += d * d;
    }
    let rstd = inverseSqrt(workgroup_sum(ss, wave_id, sg_id, wave_size) / f32(cols) + eps);

    for (var j = tid; j < cols; j = j + WG_SIZE) {
        let normed = (src[offset + j] - mean) * rstd;
        dst[offset + j] = normed * src_b[j] + bias[j];
    }
}
