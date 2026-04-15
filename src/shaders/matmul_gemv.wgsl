// GEMV kernel for M=1 matmul: C[1, N] = A[1, K] × B[K, N]
//
// Optimized for the LM-decode hot path where our 64×64-tile matmul wastes
// 63/64 of the workgroup. One thread per output column: thread t in
// workgroup w writes C[w * 256 + t]. Threads in a warp access consecutive
// columns of B, giving coalesced B loads. A is loaded cooperatively into
// shared memory in chunks of 256 and broadcast-read per k-iteration.
//
// Dispatch: [ceil(N/256), 1, 1], workgroup size 256.
//
// Same binding layout as matmul.wgsl so runtime can reuse MatMulData.

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
var<workgroup> shared_a: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let col = wgid.x * 256u + tid;
    let n = params.n;
    let k = params.k;

    var acc = 0.0;
    var k_base = 0u;
    loop {
        if k_base >= k { break; }
        // Cooperatively stage up to 256 A-row elements into shared memory.
        let k_end = min(k_base + 256u, k);
        let k_chunk = k_end - k_base;
        if tid < k_chunk {
            shared_a[tid] = matrix_a[k_base + tid];
        }
        workgroupBarrier();

        // Accumulate this chunk's contribution to our output column.
        // All threads in a warp read B[(k_base+kk) * n + col] for
        // consecutive `col` values → one coalesced load per kk.
        if col < n {
            for (var kk = 0u; kk < k_chunk; kk++) {
                acc += shared_a[kk] * matrix_b[(k_base + kk) * n + col];
            }
        }
        workgroupBarrier();
        k_base = k_end;
    }

    if col < n {
        matrix_c[col] = acc;
    }
}
