// Reduction: sum_all, mean_all

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

var<storage, read> src: array<f32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;

const WG_SIZE: u32 = 256u;
var<workgroup> wg_data: array<f32, 256>;

@compute @workgroup_size(256)
fn sum_all(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = gid.x;
    let local_id = lid.x;

    if i < params.len {
        wg_data[local_id] = src[i];
    } else {
        wg_data[local_id] = 0.0;
    }
    workgroupBarrier();

    for (var stride = WG_SIZE / 2u; stride > 0u; stride /= 2u) {
        if local_id < stride {
            wg_data[local_id] += wg_data[local_id + stride];
        }
        workgroupBarrier();
    }

    if local_id == 0u {
        dst[gid.x / WG_SIZE] = wg_data[0];
    }
}

@compute @workgroup_size(256)
fn mean_all(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = gid.x;
    let local_id = lid.x;

    if i < params.len {
        wg_data[local_id] = src[i];
    } else {
        wg_data[local_id] = 0.0;
    }
    workgroupBarrier();

    for (var stride = WG_SIZE / 2u; stride > 0u; stride /= 2u) {
        if local_id < stride {
            wg_data[local_id] += wg_data[local_id + stride];
        }
        workgroupBarrier();
    }

    if local_id == 0u {
        dst[gid.x / WG_SIZE] = wg_data[0] / f32(params.len);
    }
}
