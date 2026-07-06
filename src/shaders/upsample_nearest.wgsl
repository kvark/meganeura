// Nearest-neighbor upsample with separate H/W scale factors.
// input[N, C, H, W] → output[N, C, H*scale_h, W*scale_w]
// Dispatch: [ceil(total_out / 256), 1, 1]  workgroup_size(256)

struct Params {
    batch: u32,
    channels: u32,
    in_h: u32,
    in_w: u32,
    scale_h: u32,
    scale_w: u32,
    _pad0: u32,
    _pad1: u32,
}

var<storage> src: array<f32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let out_h = params.in_h * params.scale_h;
    let out_w = params.in_w * params.scale_w;
    let total = params.batch * params.channels * out_h * out_w;
    if i >= total { return; }

    // Decode NCHW output index.
    let ow = i % out_w;
    let oh = (i / out_w) % out_h;
    let c = (i / (out_w * out_h)) % params.channels;
    let n = i / (params.channels * out_h * out_w);

    // Nearest neighbor mapping.
    let ih = oh / params.scale_h;
    let iw = ow / params.scale_w;
    let src_idx = ((n * params.channels + c) * params.in_h + ih) * params.in_w + iw;
    dst[i] = src[src_idx];
}
