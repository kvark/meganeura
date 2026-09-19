// 4-column GEMV-BT: C[1, N] = A[1, K] × B[N, K]ᵀ.
//
// One workgroup writes 4 consecutive outputs so dispatch is N/4, matching
// the K-split GEMV grid. B is stored [N, K]; each output row is contiguous
// along K. Threads K-split with vec4 A/B loads (llama.cpp MUL_MAT_VEC's
// access pattern) and reuse A across the four rows.
//
// Used when a graph MatMul still has B as logical [K, N] but compile
// physically transposes the weight so this kernel can run. Native graph
// MatMulBT (lm_head) keeps the 1-column shader.
//
// Requires K % 4 == 0. Dispatch: ceil(N/4) workgroups.

struct Params {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
}

var<storage> matrix_a: array<vec4<f32>>;
var<storage> matrix_b: array<vec4<f32>>;
var<storage, read_write> matrix_c: array<f32>;
var<uniform> params: Params;
const LANES: u32 = 64u;

var<workgroup> reduce_buf: array<vec4<f32>, LANES>;

@compute @workgroup_size(LANES)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let col0 = wgid.x * 4u;
    let lane = lid.x;
    let k_v4 = params.k / 4u;
    let n = params.n;

    var acc = vec4<f32>(0.0);
    var kk_v4 = lane;
    loop {
        if kk_v4 >= k_v4 { break; }
        let a = matrix_a[kk_v4];
        acc.x += dot(a, matrix_b[(col0 + 0u) * k_v4 + kk_v4]);
        if col0 + 1u < n {
            acc.y += dot(a, matrix_b[(col0 + 1u) * k_v4 + kk_v4]);
        }
        if col0 + 2u < n {
            acc.z += dot(a, matrix_b[(col0 + 2u) * k_v4 + kk_v4]);
        }
        if col0 + 3u < n {
            acc.w += dot(a, matrix_b[(col0 + 3u) * k_v4 + kk_v4]);
        }
        kk_v4 += LANES;
    }

    reduce_buf[lane] = acc;
    workgroupBarrier();
    if lane < 32u { reduce_buf[lane] += reduce_buf[lane + 32u]; }
    workgroupBarrier();
    if lane < 16u { reduce_buf[lane] += reduce_buf[lane + 16u]; }
    workgroupBarrier();
    if lane < 8u { reduce_buf[lane] += reduce_buf[lane + 8u]; }
    workgroupBarrier();
    if lane < 4u { reduce_buf[lane] += reduce_buf[lane + 4u]; }
    workgroupBarrier();
    if lane < 2u { reduce_buf[lane] += reduce_buf[lane + 2u]; }
    workgroupBarrier();
    if lane == 0u {
        let total = reduce_buf[0] + reduce_buf[1];
        matrix_c[col0] = total.x;
        if col0 + 1u < n { matrix_c[col0 + 1u] = total.y; }
        if col0 + 2u < n { matrix_c[col0 + 2u] = total.z; }
        if col0 + 3u < n { matrix_c[col0 + 3u] = total.w; }
    }
}
