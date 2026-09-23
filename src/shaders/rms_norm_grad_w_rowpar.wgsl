// RmsNormGradW pass 1: row-block partial gradient computation.
//
// Each workgroup handles `_pad` consecutive rows (a power of two, at most
// 32). Every row of the block gets 256 / block lanes that reduce its sum of
// squares side by side, so the rsqrt of all rows takes one tree; then each
// column writes partial[block, col] = sum over the block's rows of
// dy[row,col] * x[row,col] * rsqrt. Larger blocks shrink the partial buffer
// SumRows reads back; a block of one is the plain row-parallel kernel.
//
// Dispatch: [ceil(rows / block), 1, 1], workgroup_size(256)
// Output: [ceil(rows / block), cols] partial gradient (summed by SumRows)

struct Params {
    m: u32,     // rows
    n: u32,     // cols
    k: u32,     // eps_bits
    _pad: u32,  // rows per workgroup
}

var<storage> src_a: array<f32>;  // dy [rows, cols]
var<storage> src_b: array<f32>;  // x  [rows, cols]
var<storage> bias: array<f32>;   // w  [cols] (unused in this pass)
var<storage, read_write> dst: array<f32>;  // partial [blocks, cols]
var<uniform> params: Params;
var<workgroup> wg_data: array<f32, 256>;

const MAX_BLOCK: u32 = 32u;
var<workgroup> row_rsqrt: array<f32, MAX_BLOCK>;

@compute @workgroup_size(256)
fn rms_norm_grad_w_rowpar(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let cols = params.n;
    let eps = bitcast<f32>(params.k);
    let block = clamp(params._pad, 1u, MAX_BLOCK);
    let row_begin = wgid.x * block;
    let row_end = min(row_begin + block, params.m);

    // Lanes [slot * lanes, (slot + 1) * lanes) reduce row `row_begin + slot`.
    let lanes = 256u / block;
    let slot = tid / lanes;
    let lane = tid % lanes;
    let row = row_begin + slot;
    var ss = 0.0;
    if row < row_end {
        var j = lane;
        loop {
            if j >= cols { break; }
            let v = src_b[row * cols + j];
            ss += v * v;
            j += lanes;
        }
    }
    wg_data[tid] = ss;
    workgroupBarrier();
    // A fixed eight-step tree; steps wider than a row's lane group idle.
    for (var s = 128u; s > 0u; s >>= 1u) {
        if s < lanes && lane < s {
            wg_data[tid] += wg_data[tid + s];
        }
        workgroupBarrier();
    }
    if lane == 0u && row < row_end {
        row_rsqrt[slot] = inverseSqrt(wg_data[tid] / f32(cols) + eps);
    }
    workgroupBarrier();

    // Partial gradient: dy * x * rsqrt summed over the block's rows.
    var j = tid;
    loop {
        if j >= cols { break; }
        var acc = 0.0;
        // A constant bound keeps `k` a plain unrolled index: lavapipe's
        // threaded JIT crashes on a shared-array index that follows a
        // runtime loop bound.
        for (var k = 0u; k < MAX_BLOCK; k++) {
            let r = row_begin + k;
            if r >= row_end { break; }
            let index = r * cols + j;
            acc += src_a[index] * src_b[index] * row_rsqrt[k];
        }
        dst[wgid.x * cols + j] = acc;
        j += 256u;
    }
}
