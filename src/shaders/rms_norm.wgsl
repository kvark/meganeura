struct Params {
    rows: u32,
    cols: u32,
    eps_bits: u32,
    _pad: u32,
}

var<storage> src: array<f32>;
var<storage> bias: array<f32>;  // weight vector
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;
var<workgroup> wg_data: array<f32, 256>;

// One workgroup per row. Dispatch: [rows, 1, 1]
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wgid.x;
    let tid = lid.x;
    if row >= params.rows { return; }
    let offset = row * params.cols;
    let eps = bitcast<f32>(params.eps_bits);

    // Phase 1: Strided accumulation of squared values
    var ss = 0.0;
    var j = tid;
    loop {
        if j >= params.cols { break; }
        let val = src[offset + j];
        ss += val * val;
        j += 256u;
    }
    wg_data[tid] = ss;
    workgroupBarrier();

    // Phase 2: Tree reduction
    var stride = 128u;
    loop {
        if stride == 0u { break; }
        if tid < stride {
            wg_data[tid] += wg_data[tid + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }

    // Phase 3: Compute inverse square root
    let mean_sq = wg_data[0] / f32(params.cols);
    let rsqrt_val = inverseSqrt(mean_sq + eps);

    // Phase 4: Normalize and scale
    var j2 = tid;
    loop {
        if j2 >= params.cols { break; }
        dst[offset + j2] = src[offset + j2] * rsqrt_val * bias[j2];
        j2 += 256u;
    }
}
