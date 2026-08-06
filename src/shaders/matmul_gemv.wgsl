// K-split GEMV for M=1: C[1, N] = A[1, K] × B[K, N].
//
// One workgroup handles 4 output columns (one vec4). The 256 threads in
// the workgroup K-split the reduction: each thread accumulates a partial
// vec4 over K/256 rows of B. A tree reduction across the threads (in
// shared memory) combines the partials. Final write: one vec4 to C.
//
// Compared to the previous N-parallel-only GEMV:
//   Old: N/128 workgroups × 32 threads   e.g. N=576 → 5 WG × 32 = 160 threads
//   New: N/4 workgroups   × 256 threads  e.g. N=576 → 144 WG × 256 = 36864 threads
// Memory-level parallelism rises accordingly (the previous version was
// at ~10% of memory-BW peak because occupancy was too low to hide DRAM
// latency).
//
// The 256-wide workgroup is not about steady-state bandwidth: measured
// free-running on an RTX 5080, 32 threads is actually faster (1205-1409
// GB/s vs 1002-1111). It wins on barrier recovery - 1.6-2.9us to reach
// full rate after a barrier versus 2.05-11.37us for the narrow form -
// and decode is a chain of short kernels separated by barriers, so ramp
// and drain dominate.
//
// Requires N % 4 == 0. Tradeoff vs old: within a warp, threads now
// access the same col4 at different k rows (strided by N/4 vec4s), so
// the per-warp coalescing is lost. L2 cache still captures reuse across
// adjacent col4 workgroups (they share the same k rows of B), so the
// net memory throughput is higher in aggregate.

struct Params {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
}

var<storage> matrix_a: array<f32>;
var<storage> matrix_b: array<vec4<f32>>;
var<storage, read_write> matrix_c: array<vec4<f32>>;
var<uniform> params: Params;
var<workgroup> reduce_buf: array<vec4<f32>, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let col4 = wgid.x;
    let lane = lid.x;
    let n_v4 = params.n / 4u;
    if col4 >= n_v4 { return; }
    let k = params.k;

    // Each thread accumulates a partial sum over its K-stride slice.
    var acc = vec4<f32>(0.0);
    var kk = lane;
    loop {
        if kk >= k { break; }
        let a = matrix_a[kk];
        let b = matrix_b[kk * n_v4 + col4];
        acc = acc + vec4<f32>(a) * b;
        kk += 256u;
    }

    // Tree reduction via shared memory. The last three levels stay within
    // one warp on NVIDIA, where workgroupBarrier compiles to near-free
    // subgroup sync instructions.
    reduce_buf[lane] = acc;
    workgroupBarrier();
    if lane < 128u { reduce_buf[lane] += reduce_buf[lane + 128u]; }
    workgroupBarrier();
    if lane < 64u { reduce_buf[lane] += reduce_buf[lane + 64u]; }
    workgroupBarrier();
    if lane < 32u { reduce_buf[lane] += reduce_buf[lane + 32u]; }
    workgroupBarrier();
    if lane < 16u { reduce_buf[lane] += reduce_buf[lane + 16u]; }
    workgroupBarrier();
    if lane < 8u  { reduce_buf[lane] += reduce_buf[lane + 8u];  }
    workgroupBarrier();
    if lane < 4u  { reduce_buf[lane] += reduce_buf[lane + 4u];  }
    workgroupBarrier();
    if lane < 2u  { reduce_buf[lane] += reduce_buf[lane + 2u];  }
    workgroupBarrier();
    if lane == 0u {
        matrix_c[col4] = reduce_buf[0] + reduce_buf[1];
    }
}
