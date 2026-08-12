// Row broadcast used by GlobalAvgPool backward and BroadcastInner.
// Dispatch: [ceil(total/256), 1, 1] where total = batch * channels * spatial

struct Params {
    total: u32,     // batch * channels * spatial
    spatial: u32,   // H * W
    broadcast: u32, // non-zero keeps the broadcast unscaled
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
    let value = src[i / spatial];
    dst[i] = select(value / f32(spatial), value, params.broadcast != 0u);
}
