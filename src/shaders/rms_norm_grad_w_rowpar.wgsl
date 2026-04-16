// RmsNormGradW pass 1: row-parallel partial gradient computation.
//
// Each workgroup handles one row: computes rsqrt via tree reduction,
// then writes partial[row, col] = dy[row,col] * x[row,col] * rsqrt.
//
// Dispatch: [rows, 1, 1], workgroup_size(256)
// Output: [rows, cols] partial gradient (to be summed over rows by SumRows)

struct Params {
    m: u32,     // rows
    n: u32,     // cols
    k: u32,     // eps_bits
    _pad: u32,
}

var<storage> src_a: array<f32>;  // dy [rows, cols]
var<storage> src_b: array<f32>;  // x  [rows, cols]
var<storage> bias: array<f32>;   // w  [cols] (unused in this pass)
var<storage, read_write> dst: array<f32>;  // partial [rows, cols]
var<uniform> params: Params;
var<workgroup> wg_data: array<f32, 256>;

@compute @workgroup_size(256)
fn rms_norm_grad_w_rowpar(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wgid.x;
    let tid = lid.x;
    let cols = params.n;
    let eps = bitcast<f32>(params.k);
    let offset = row * cols;

    // Cooperative rsqrt: grid-stride sum-of-squares
    var ss = 0.0;
    var j = tid;
    loop {
        if j >= cols { break; }
        let v = src_b[offset + j];
        ss += v * v;
        j += 256u;
    }
    wg_data[tid] = ss;
    workgroupBarrier();

    // Tree reduction
    for (var s = 128u; s > 0u; s >>= 1u) {
        if tid < s {
            wg_data[tid] += wg_data[tid + s];
        }
        workgroupBarrier();
    }
    let rsqrt_val = inverseSqrt(wg_data[0] / f32(cols) + eps);

    // Write partial gradient: dy[row,col] * x[row,col] * rsqrt
    j = tid;
    loop {
        if j >= cols { break; }
        dst[offset + j] = src_a[offset + j] * src_b[offset + j] * rsqrt_val;
        j += 256u;
    }
}
