// K-split GEMV for M=1: C[1, N] = A[1, K] × B[K, N].
//
// One workgroup handles 4 output columns (one vec4). The 32 threads in
// the workgroup K-split the reduction: each thread accumulates a partial
// vec4 over K/32 rows of B. A tree reduction across the 32 threads (in
// shared memory) combines the partials. Final write: one vec4 to C.
//
// Compared to the previous N-parallel-only GEMV:
//   Old: N/128 workgroups × 32 threads   e.g. N=576 → 5 WG × 32 = 160 threads
//   New: N/4 workgroups   × 32 threads   e.g. N=576 → 144 WG × 32 = 4608 threads
// ~28× more concurrent threads on RTX 3050. Memory-level parallelism
// rises accordingly (the previous version was at ~10% of memory-BW
// peak because occupancy was too low to hide DRAM latency).
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
var<workgroup> reduce_buf: array<vec4<f32>, 32>;

@compute @workgroup_size(32)
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
        kk += 32u;
    }

    // Warp-wide reduction via shared memory. WG=32 means one warp; on
    // NVIDIA the workgroupBarrier calls compile to near-free subgroup
    // sync instructions.
    reduce_buf[lane] = acc;
    workgroupBarrier();
    if lane < 16u { reduce_buf[lane] = reduce_buf[lane] + reduce_buf[lane + 16u]; }
    workgroupBarrier();
    if lane < 8u  { reduce_buf[lane] = reduce_buf[lane] + reduce_buf[lane + 8u];  }
    workgroupBarrier();
    if lane < 4u  { reduce_buf[lane] = reduce_buf[lane] + reduce_buf[lane + 4u];  }
    workgroupBarrier();
    if lane < 2u  { reduce_buf[lane] = reduce_buf[lane] + reduce_buf[lane + 2u];  }
    workgroupBarrier();
    if lane == 0u {
        matrix_c[col4] = reduce_buf[0] + reduce_buf[1];
    }
}
