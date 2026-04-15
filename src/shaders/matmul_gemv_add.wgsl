// GEMV with fused residual-add for M=1 matmul:
//   C[1, N] = A[1, K] × B[K, N] + D[1, N]
//
// Identical structure to matmul_gemv.wgsl — same vec4 loads, unroll-by-4,
// WG=32 — plus one extra vec4 load from D (residual) at the end.
// Requires N % 4 == 0. Binding layout matches FusedMatMulAddData: the
// addend `src` is bound as an additional storage input.

struct Params {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
}

var<storage> matrix_a: array<f32>;
var<storage> matrix_b: array<vec4<f32>>;
var<storage, read_write> matrix_c: array<vec4<f32>>;
var<storage> src: array<vec4<f32>>;  // D: residual addend [1, N]
var<uniform> params: Params;

@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let col4 = wgid.x * 32u + lid.x;
    let n_v4 = params.n / 4u;
    if col4 >= n_v4 { return; }
    let k = params.k;

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
    loop {
        if kk >= k { break; }
        let a = matrix_a[kk];
        let b = matrix_b[kk * n_v4 + col4];
        acc0 = acc0 + vec4<f32>(a) * b;
        kk += 1u;
    }

    matrix_c[col4] = (acc0 + acc1) + (acc2 + acc3) + src[col4];
}
