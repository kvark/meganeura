// Precompute rsqrt for RmsNorm: rsqrt[row] = inverseSqrt(sum(x²)/K + eps)
//
// Dispatch: [rows, 1, 1], WG=256
// Each workgroup cooperatively computes rsqrt for one row using
// shared-memory tree reduction over the K dimension.

struct Params {
    rows: u32,
    cols: u32,
    eps_bits: u32,
    _pad: u32,
}

var<storage> src: array<f32>;              // X [rows, cols]
var<storage, read_write> dst: array<f32>;  // rsqrt [rows]
var<uniform> params: Params;
var<workgroup> wg_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wgid.x;
    let tid = lid.x;
    let cols = params.cols;
    let eps = bitcast<f32>(params.eps_bits);

    if row >= params.rows { return; }

    // Cooperative sum of squares: each thread sums a stride of the row
    let offset = row * cols;
    var ss = 0.0;
    var j = tid;
    loop {
        if j >= cols { break; }
        let v = src[offset + j];
        ss += v * v;
        j += 256u;
    }
    wg_sum[tid] = ss;
    workgroupBarrier();

    // Tree reduction
    var stride = 128u;
    loop {
        if stride == 0u { break; }
        if tid < stride {
            wg_sum[tid] += wg_sum[tid + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }

    if tid == 0u {
        dst[row] = inverseSqrt(wg_sum[0] / f32(cols) + eps);
    }
}
