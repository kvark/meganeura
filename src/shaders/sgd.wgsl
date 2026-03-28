struct Params {
    len: u32,
    lr: f32,
    _pad0: u32,
    _pad1: u32,
}

var<storage> param: array<f32>;
var<storage> grad: array<f32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.len { return; }
    dst[i] = param[i] - params.lr * grad[i];
}
