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

// Workgroup-cooperative LayerNorm: one workgroup per row, threads
// stride across the row's columns. Replaces the previous
// one-thread-per-row kernel that ran ~24x off the card's memory
// bandwidth (Whisper profile: 180us per 1500x384 norm).
//
// Two reductions: the mean, then the mean squared deviation from it. The
// one-pass E[x²] − E[x]² cancels catastrophically for rows whose mean is
// large next to their spread and can go negative, making rsqrt NaN. The
// row is small next to the cache, so the second read is cheap.
var<workgroup> partial: array<f32, WG_SIZE>;
var<workgroup> wg_mean: f32;
var<workgroup> wg_rstd: f32;

fn reduce_partial(tid: u32) {
    var stride = WG_SIZE / 2u;
    while stride > 0u {
        if tid < stride {
            partial[tid] += partial[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
}

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
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
    partial[tid] = s;
    workgroupBarrier();
    reduce_partial(tid);
    if tid == 0u {
        wg_mean = partial[0] / f32(cols);
    }
    workgroupBarrier();
    let mean = wg_mean;

    var ss = 0.0;
    for (var j = tid; j < cols; j = j + WG_SIZE) {
        let d = src[offset + j] - mean;
        ss += d * d;
    }
    partial[tid] = ss;
    workgroupBarrier();
    reduce_partial(tid);
    if tid == 0u {
        wg_rstd = inverseSqrt(partial[0] / f32(cols) + eps);
    }
    workgroupBarrier();
    let rstd = wg_rstd;

    for (var j = tid; j < cols; j = j + WG_SIZE) {
        let normed = (src[offset + j] - mean) * rstd;
        dst[offset + j] = normed * src_b[j] + bias[j];
    }
}
