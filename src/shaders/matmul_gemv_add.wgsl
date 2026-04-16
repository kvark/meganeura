// K-split GEMV with residual add: C[1, N] = A × B + D[1, N].
// Structure mirrors matmul_gemv.wgsl; only difference is the final vec4
// load from D and the add into the reduce result. Dispatch: N/4 WGs ×
// 32 threads. Requires N % 4 == 0.

struct Params {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
}

var<storage> matrix_a: array<f32>;
var<storage> matrix_b: array<vec4<f32>>;
var<storage, read_write> matrix_c: array<vec4<f32>>;
var<storage> src: array<vec4<f32>>;
var<uniform> params: Params;
var<workgroup> reduce_buf: array<vec4<f32>, 32>;

@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let col4 = wgid.x;
    let lane = lid.x;
    let n_v4 = params.n / 4u;
    if col4 >= n_v4 { return; }
    let k = params.k;

    var acc = vec4<f32>(0.0);
    var kk = lane;
    loop {
        if kk >= k { break; }
        let a = matrix_a[kk];
        let b = matrix_b[kk * n_v4 + col4];
        acc = acc + vec4<f32>(a) * b;
        kk += 32u;
    }

    reduce_buf[lane] = acc;
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
        matrix_c[col4] = reduce_buf[0] + reduce_buf[1] + src[col4];
    }
}
