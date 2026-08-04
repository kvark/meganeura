// Row broadcast used by GlobalAvgPool backward and BroadcastInner.
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

@compute @workgroup_size(256)
fn broadcast_inner(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.total { return; }
    dst[i] = src[i / params.spatial];
}

@compute @workgroup_size(256)
fn tile_inner(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.total { return; }
    let inner = params.spatial;
    let repeated_inner = inner * params._pad0;
    let row = i / repeated_inner;
    dst[i] = src[row * inner + i % inner];
}

@compute @workgroup_size(256)
fn tile_inner_grad(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.total { return; }
    let inner = params.spatial;
    let repeats = params._pad0;
    let row = i / inner;
    let column = i % inner;
    let base = row * inner * repeats + column;
    var sum = 0.0;
    for (var repeat = 0u; repeat < repeats; repeat += 1u) {
        sum += src[base + repeat * inner];
    }
    dst[i] = sum;
}
