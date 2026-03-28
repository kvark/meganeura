struct Params {
    seq: u32,
    hidden: u32,
    _pad0: u32,
    _pad1: u32,
}

var<storage> indices: array<u32>;
var<storage> src: array<f32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let total = params.seq * params.hidden;
    if i >= total { return; }
    let row = i / params.hidden;
    let col = i % params.hidden;
    let token_id = indices[row];
    dst[i] = src[token_id * params.hidden + col];
}
