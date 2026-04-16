// K-split GEMV for M=1 MatMulBT: C[1, N] = A[1, K] × B[N, K]ᵀ.
//
// For MatMulBT, B is stored [N, K] row-major — meaning each row of B
// (one output col's weights) is contiguous along K. K-split thus gives
// us naturally coalesced memory access:
//
//   WG handles output col `c`. The 32 threads in the WG K-split; thread
//   lane l reads vec4s at indices {l, l+32, l+64, ...} inside row c.
//   Within a warp, lanes 0..31 read 32 consecutive vec4s = 512 B = 4
//   cache lines per warp-step, perfectly coalesced.
//
// Both A and B are loaded as `array<vec4<f32>>` so each iteration
// consumes one 128-bit A load + one 128-bit B load + one dot4 product.
// That's 4 FMAs per iteration per thread vs 1 in the scalar version,
// and 128-bit memory transactions at the load level.
//
// Requires K % 4 == 0. Dispatch: N workgroups × 32 threads.
// Binding layout matches MatMulData (matrix_a / matrix_b / matrix_c /
// params); the shader reinterprets the storage buffers as vec4 arrays.

struct Params {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
}

var<storage> matrix_a: array<vec4<f32>>;     // A: [1, K] as K/4 vec4s
var<storage> matrix_b: array<vec4<f32>>;     // B: [N, K] as N × K/4 vec4s
var<storage, read_write> matrix_c: array<f32>;  // C: [1, N]
var<uniform> params: Params;
var<workgroup> reduce_buf: array<f32, 32>;

@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let col = wgid.x;
    let lane = lid.x;
    if col >= params.n { return; }
    let k_v4 = params.k / 4u;
    let row_off = col * k_v4;

    var acc = 0.0;
    var kk_v4 = lane;
    loop {
        if kk_v4 >= k_v4 { break; }
        let a = matrix_a[kk_v4];
        let b = matrix_b[row_off + kk_v4];
        acc = acc + dot(a, b);
        kk_v4 += 32u;
    }

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
        matrix_c[col] = reduce_buf[0] + reduce_buf[1];
    }
}
