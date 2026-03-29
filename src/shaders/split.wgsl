// Split (backward of Concat): extract first Ca channels from [N, Ca+Cb, H, W].
// Dispatch: [ceil(total_out / 256), 1, 1]  workgroup_size(256)
// Entry point split_a extracts channels [0..Ca), split_b extracts [Ca..Ca+Cb).

struct Params {
    batch: u32,
    channels_a: u32,
    channels_b: u32,
    spatial: u32,  // H * W
}

var<storage> src: array<f32>;        // grad_output [N, Ca+Cb, H, W]
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;

@compute @workgroup_size(256)
fn split_a(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let total_a = params.batch * params.channels_a * params.spatial;
    if i >= total_a { return; }

    let hw = i % params.spatial;
    let c = (i / params.spatial) % params.channels_a;
    let n = i / (params.channels_a * params.spatial);

    let total_channels = params.channels_a + params.channels_b;
    let src_idx = ((n * total_channels + c) * params.spatial) + hw;
    dst[i] = src[src_idx];
}

@compute @workgroup_size(256)
fn split_b(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let total_b = params.batch * params.channels_b * params.spatial;
    if i >= total_b { return; }

    let hw = i % params.spatial;
    let cb = (i / params.spatial) % params.channels_b;
    let n = i / (params.channels_b * params.spatial);

    let total_channels = params.channels_a + params.channels_b;
    let src_idx = ((n * total_channels + (params.channels_a + cb)) * params.spatial) + hw;
    dst[i] = src[src_idx];
}
