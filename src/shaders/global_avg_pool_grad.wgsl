// GlobalAvgPool backward: broadcast + scale
// grad_input[i] = grad_output[i / spatial] / spatial
// Dispatch: [ceil(total/256), 1, 1] where total = batch * channels * spatial

struct Params {
    total: u32,     // batch * channels * spatial
    spatial: u32,   // H * W
    _pad0: u32,
    _pad1: u32,
}

var<storage> src: array<f32>;              // grad_output [batch * channels]
var<storage, read_write> dst: array<f32>;  // grad_input [batch * channels * spatial]
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.total { return; }
    let spatial = params.spatial;
    dst[i] = src[i / spatial] / f32(spatial);
}
