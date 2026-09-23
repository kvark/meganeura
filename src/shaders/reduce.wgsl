struct Params {
    len: u32,
    // Mean divisor; zero means `len`. Set when this dispatch finishes a mean
    // over partial sums, whose count is not the mean's element count.
    divisor: u32,
    _pad1: u32,
    _pad2: u32,
}

var<storage> src: array<f32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;
var<workgroup> wg_data: array<f32, 256>;

// Sum of a grid-strided slice of `src`. A single workgroup covers the whole
// input; with several, each covers its share and the caller finishes the
// partial sums with a second, single-workgroup dispatch.
fn slice_sum(tid: u32, group: u32, groups: u32) -> f32 {
    var acc = 0.0;
    var idx = group * 256u + tid;
    let stride = groups * 256u;
    loop {
        if idx >= params.len { break; }
        acc += src[idx];
        idx += stride;
    }
    wg_data[tid] = acc;
    workgroupBarrier();

    // Tree reduction
    var half = 128u;
    loop {
        if half == 0u { break; }
        if tid < half {
            wg_data[tid] += wg_data[tid + half];
        }
        workgroupBarrier();
        half >>= 1u;
    }
    return wg_data[0];
}

@compute @workgroup_size(256)
fn sum_all(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let total = slice_sum(lid.x, wgid.x, groups.x);
    if lid.x == 0u {
        dst[wgid.x] = total;
    }
}

@compute @workgroup_size(256)
fn mean_all(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let total = slice_sum(lid.x, wgid.x, groups.x);
    if lid.x == 0u {
        if groups.x == 1u {
            dst[0] = total / f32(select(params.len, params.divisor, params.divisor != 0u));
        } else {
            // A partial sum; the finishing dispatch divides.
            dst[wgid.x] = total;
        }
    }
}
