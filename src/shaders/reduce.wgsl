struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

var<storage> src: array<f32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;
var<workgroup> wg_data: array<f32, 256>;

@compute @workgroup_size(256)
fn sum_all(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    // Strided accumulation
    var acc = 0.0;
    var idx = tid;
    loop {
        if idx >= params.len { break; }
        acc += src[idx];
        idx += 256u;
    }
    wg_data[tid] = acc;
    workgroupBarrier();

    // Tree reduction
    var stride = 128u;
    loop {
        if stride == 0u { break; }
        if tid < stride {
            wg_data[tid] += wg_data[tid + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }

    if tid == 0u {
        dst[0] = wg_data[0];
    }
}

@compute @workgroup_size(256)
fn mean_all(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    // Strided accumulation
    var acc = 0.0;
    var idx = tid;
    loop {
        if idx >= params.len { break; }
        acc += src[idx];
        idx += 256u;
    }
    wg_data[tid] = acc;
    workgroupBarrier();

    // Tree reduction
    var stride = 128u;
    loop {
        if stride == 0u { break; }
        if tid < stride {
            wg_data[tid] += wg_data[tid + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }

    if tid == 0u {
        dst[0] = wg_data[0] / f32(params.len);
    }
}
