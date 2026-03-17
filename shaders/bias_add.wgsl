// Bias add: dst[i] = src[i] + bias[i % bias_len]

struct Params {
    len: u32,
    bias_len: u32,
    _pad0: u32,
    _pad1: u32,
}

var<storage, read> src: array<f32>;
var<storage, read> bias: array<f32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.len { return; }
    dst[i] = src[i] + bias[i % params.bias_len];
}
