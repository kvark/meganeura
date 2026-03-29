// Concat two tensors along the channel dimension (NCHW layout).
// input_a[N, Ca, H, W] ++ input_b[N, Cb, H, W] → output[N, Ca+Cb, H, W]
// Dispatch: [ceil(total_out / 256), 1, 1]  workgroup_size(256)

struct Params {
    batch: u32,
    channels_a: u32,
    channels_b: u32,
    spatial: u32,  // H * W
}

var<storage> src_a: array<f32>;
var<storage> src_b: array<f32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let total_channels = params.channels_a + params.channels_b;
    let total = params.batch * total_channels * params.spatial;
    if i >= total { return; }

    // Decode NCHW index
    let hw = i % params.spatial;
    let c = (i / params.spatial) % total_channels;
    let n = i / (total_channels * params.spatial);

    if c < params.channels_a {
        let src_idx = ((n * params.channels_a + c) * params.spatial) + hw;
        dst[i] = src_a[src_idx];
    } else {
        let cb = c - params.channels_a;
        let src_idx = ((n * params.channels_b + cb) * params.spatial) + hw;
        dst[i] = src_b[src_idx];
    }
}
