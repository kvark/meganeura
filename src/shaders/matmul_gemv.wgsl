// GEMV kernel for M=1 matmul: C[1, N] = A[1, K] × B[K, N]
//
// Optimized for the LM-decode hot path where our 64×64-tile matmul wastes
// 63/64 of the workgroup. One thread per output column.
//
// Uses workgroup size 32 (one subgroup/warp on all known GPUs) so:
//   * No workgroup barriers needed — threads in a warp execute in lockstep.
//   * 8× more workgroups than a WG=256 design, helping occupancy at
//     small N (e.g. 576/32 = 18 workgroups vs 3 workgroups at WG=256).
//
// Memory pattern: within a warp, threads t=0..31 access
// B[k*N + col_base..col_base+31] for the same k, giving one coalesced
// 128-byte load per warp per k. A is read directly from global memory
// (broadcast across all threads; on NVIDIA this coalesces to a single load
// via the L1 cache / driver broadcast detection).
//
// Dispatch: [ceil(N/32), 1, 1]. Binding layout matches matmul.wgsl /
// MatMulData.

struct Params {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
}

var<storage> matrix_a: array<f32>;           // A: [1, K]
var<storage> matrix_b: array<f32>;           // B: [K, N]
var<storage, read_write> matrix_c: array<f32>;  // C: [1, N]
var<uniform> params: Params;

@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let col = wgid.x * 32u + lid.x;
    if col >= params.n { return; }
    let n = params.n;
    let k = params.k;

    // Unrolled by 4 so the compiler has an easier time issuing FMAs and
    // pipelining B loads. With K usually ≥ 64 in real transformers this
    // cuts the outer loop iteration count by 4×.
    var acc0 = 0.0;
    var acc1 = 0.0;
    var acc2 = 0.0;
    var acc3 = 0.0;
    var kk = 0u;
    loop {
        if kk + 4u > k { break; }
        let a0 = matrix_a[kk];
        let a1 = matrix_a[kk + 1u];
        let a2 = matrix_a[kk + 2u];
        let a3 = matrix_a[kk + 3u];
        acc0 += a0 * matrix_b[kk * n + col];
        acc1 += a1 * matrix_b[(kk + 1u) * n + col];
        acc2 += a2 * matrix_b[(kk + 2u) * n + col];
        acc3 += a3 * matrix_b[(kk + 3u) * n + col];
        kk += 4u;
    }
    // Tail for K not multiple of 4.
    loop {
        if kk >= k { break; }
        acc0 += matrix_a[kk] * matrix_b[kk * n + col];
        kk += 1u;
    }

    matrix_c[col] = (acc0 + acc1) + (acc2 + acc3);
}
