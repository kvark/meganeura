struct Params {
    len: u32,
    columns: u32,
    stride: u32,
    offset: u32,
    rate: f32,
    decay: f32,
    copy: u32,
    _pad: u32,
}
var<storage, read> src: array<f32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let block = group.x + group.y * groups.x;
    let blocks = params.len / 256u + select(0u, 1u, params.len % 256u != 0u);
    if block >= blocks { return; }
    let i = block * 256u + local.x;
    if i >= params.len { return; }
    let j = params.offset + (i / params.columns) * params.stride + i % params.columns;
    if params.copy != 0u {
        dst[j] = src[i];
    } else {
        dst[j] = params.rate * src[i] + params.decay * dst[j];
    }
}
