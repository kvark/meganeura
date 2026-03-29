// Nearest-neighbor 2x upsample: input[N, C, H, W] → output[N, C, 2H, 2W]
// Dispatch: [ceil(total_out / 256), 1, 1]  workgroup_size(256)

struct Params {
    batch: u32,
    channels: u32,
    in_h: u32,
    in_w: u32,
}

var<storage> src: array<f32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let out_h = params.in_h * 2u;
    let out_w = params.in_w * 2u;
    let total = params.batch * params.channels * out_h * out_w;
    if i >= total { return; }

    // Decode output NCHW index
    let ow = i % out_w;
    let oh = (i / out_w) % out_h;
    let c = (i / (out_w * out_h)) % params.channels;
    let n = i / (params.channels * out_h * out_w);

    // Nearest neighbor: map to input
    let ih = oh / 2u;
    let iw = ow / 2u;
    let src_idx = ((n * params.channels + c) * params.in_h + ih) * params.in_w + iw;
    dst[i] = src[src_idx];
}
