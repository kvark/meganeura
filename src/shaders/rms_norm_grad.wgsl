// Two entry points:
//   rms_norm_grad_w: gradient wrt weight, dispatch [ceil(cols/256), 1, 1]
//   rms_norm_grad_x: gradient wrt input, dispatch [rows, 1, 1]
// Params field names: m=rows, n=cols, k=eps_bits

struct Params {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
}

var<storage> src_a: array<f32>;  // dy
var<storage> src_b: array<f32>;  // x
var<storage> bias: array<f32>;   // w (weight)
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;
var<workgroup> wg_data: array<f32, 256>;

// grad_w[col] = sum_i( dy[i*n+col] * x[i*n+col] * rsqrt_i )
@compute @workgroup_size(256)
fn rms_norm_grad_w(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let rows = params.m;
    let cols = params.n;
    let eps = bitcast<f32>(params.k);
    if col >= cols { return; }

    var acc = 0.0;
    for (var i = 0u; i < rows; i++) {
        let offset = i * cols;
        // Compute rsqrt for row i
        var ss = 0.0;
        for (var j = 0u; j < cols; j++) {
            let v = src_b[offset + j];
            ss += v * v;
        }
        let rsqrt_i = inverseSqrt(ss / f32(cols) + eps);
        acc += src_a[offset + col] * src_b[offset + col] * rsqrt_i;
    }
    dst[col] = acc;
}

// grad_x[i,j] = rsqrt_i * (dy[i,j]*w[j] - x[i,j] * s_i)
// where s_i = (rsqrt_i^2 / cols) * sum_j(dy[i,j]*w[j]*x[i,j])
@compute @workgroup_size(256)
fn rms_norm_grad_x(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wgid.x;
    let tid = lid.x;
    let rows = params.m;
    let cols = params.n;
    let eps = bitcast<f32>(params.k);
    if row >= rows { return; }
    let offset = row * cols;

    // Phase 1: Compute rsqrt via shared memory reduction
    var ss = 0.0;
    var j = tid;
    loop {
        if j >= cols { break; }
        let v = src_b[offset + j];
        ss += v * v;
        j += 256u;
    }
    wg_data[tid] = ss;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if stride == 0u { break; }
        if tid < stride {
            wg_data[tid] += wg_data[tid + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }
    let rsqrt_val = inverseSqrt(wg_data[0] / f32(cols) + eps);

    // Phase 2: Compute s_i = (rsqrt^2 / cols) * sum_j(dy[j]*w[j]*x[j])
    var dot = 0.0;
    j = tid;
    loop {
        if j >= cols { break; }
        dot += src_a[offset + j] * bias[j] * src_b[offset + j];
        j += 256u;
    }
    wg_data[tid] = dot;
    workgroupBarrier();

    stride = 128u;
    loop {
        if stride == 0u { break; }
        if tid < stride {
            wg_data[tid] += wg_data[tid + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }
    let s_i = (rsqrt_val * rsqrt_val / f32(cols)) * wg_data[0];

    // Phase 3: Write output
    j = tid;
    loop {
        if j >= cols { break; }
        let dy = src_a[offset + j];
        let w = bias[j];
        let x = src_b[offset + j];
        dst[offset + j] = rsqrt_val * (dy * w - x * s_i);
        j += 256u;
    }
}
