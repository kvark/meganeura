// GEMV kernel for M=1 matmul: C[1, N] = A[1, K] × B[K, N].
//
// Requires N % 4 == 0. Each thread handles one 4-column vec4 of output.
// B is viewed as `array<vec4<f32>>` so every B load is a vec4 (128-bit
// storage load in SPIR-V → `buffer_load_dwordx4`-equivalent on the GPU),
// quadrupling in-flight memory bandwidth per thread vs scalar f32 loads.
//
// WG=32 (one subgroup/warp), no barriers. Within a warp, threads t=0..31
// read vec4s at consecutive indices — one coalesced 512-byte transaction
// per warp per k-iteration.
//
// Inner K loop is unrolled by 4 so the compiler has 4 in-flight B loads
// and 4 accumulators per thread, raising memory-level parallelism.
//
// Dispatch: [ceil((N/4)/32), 1, 1] = [ceil(N/128), 1, 1].
// Binding layout matches matmul.wgsl / MatMulData.

struct Params {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
}

var<storage> matrix_a: array<f32>;
// N must be a multiple of 4 — compile::Op::MatMul gates on this before
// selecting `ShaderEntry::MatMulGemv`.
var<storage> matrix_b: array<vec4<f32>>;
var<storage, read_write> matrix_c: array<vec4<f32>>;
var<uniform> params: Params;

@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let col4 = wgid.x * 32u + lid.x;
    let n_v4 = params.n / 4u;
    if col4 >= n_v4 { return; }
    let k = params.k;

    // 4 independent vec4 accumulators → 4 in-flight B loads per thread.
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    var kk = 0u;
    loop {
        if kk + 4u > k { break; }
        let a0 = matrix_a[kk];
        let a1 = matrix_a[kk + 1u];
        let a2 = matrix_a[kk + 2u];
        let a3 = matrix_a[kk + 3u];
        let b0 = matrix_b[kk * n_v4 + col4];
        let b1 = matrix_b[(kk + 1u) * n_v4 + col4];
        let b2 = matrix_b[(kk + 2u) * n_v4 + col4];
        let b3 = matrix_b[(kk + 3u) * n_v4 + col4];
        acc0 = acc0 + vec4<f32>(a0) * b0;
        acc1 = acc1 + vec4<f32>(a1) * b1;
        acc2 = acc2 + vec4<f32>(a2) * b2;
        acc3 = acc3 + vec4<f32>(a3) * b3;
        kk += 4u;
    }
    // Tail for K not multiple of 4.
    loop {
        if kk >= k { break; }
        let a = matrix_a[kk];
        let b = matrix_b[kk * n_v4 + col4];
        acc0 = acc0 + vec4<f32>(a) * b;
        kk += 1u;
    }

    matrix_c[col4] = (acc0 + acc1) + (acc2 + acc3);
}
