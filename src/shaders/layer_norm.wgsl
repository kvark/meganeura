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
// Each thread accumulates partial sum and sum-of-squares for its
// stride, then a tree reduction collapses them across the workgroup.
// E[x²] - E[x]² is the cheaper variance form. For typical transformer
// activations the precision loss vs Welford is negligible — values
// are roughly zero-mean and well-bounded.
var<workgroup> partial_sum: array<f32, WG_SIZE>;
var<workgroup> partial_sumsq: array<f32, WG_SIZE>;
var<workgroup> wg_mean: f32;
var<workgroup> wg_rstd: f32;

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

    // Phase 1: accumulate partial sum and sum-of-squares over the
    // thread's strided slice. Each thread processes ceil(cols/WG_SIZE)
    // elements, so for cols=384 (Whisper) that's 3 elements/thread.
    var s = 0.0;
    var ss = 0.0;
    for (var j = tid; j < cols; j = j + WG_SIZE) {
        let v = src[offset + j];
        s += v;
        ss += v * v;
    }
    partial_sum[tid] = s;
    partial_sumsq[tid] = ss;
    workgroupBarrier();

    // Phase 2: tree reduction. Halve the active range each step until
    // thread 0 holds the full row sum.
    var stride = WG_SIZE / 2u;
    while stride > 0u {
        if tid < stride {
            partial_sum[tid] += partial_sum[tid + stride];
            partial_sumsq[tid] += partial_sumsq[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if tid == 0u {
        let mean = partial_sum[0] / f32(cols);
        let var_ = partial_sumsq[0] / f32(cols) - mean * mean;
        wg_mean = mean;
        wg_rstd = inverseSqrt(var_ + eps);
    }
    workgroupBarrier();

    let mean = wg_mean;
    let rstd = wg_rstd;

    // Phase 3: normalize and apply affine, same striding pattern.
    for (var j = tid; j < cols; j = j + WG_SIZE) {
        let normed = (src[offset + j] - mean) * rstd;
        dst[offset + j] = normed * src_b[j] + bias[j];
    }
}
