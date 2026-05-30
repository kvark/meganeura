// Spatial crop on NCHW.
// input[N,C,in_h,in_w] → output[N,C, in_h-start_h-end_h, in_w-start_w-end_w]
// Dispatch: [ceil(total_out / 256), 1, 1]  workgroup_size(256)

struct Params {
    batch: u32,
    channels: u32,
    in_h: u32,
    in_w: u32,
    start_h: u32,
    end_h: u32,
    start_w: u32,
    end_w: u32,
}

var<storage> src: array<f32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let out_h = params.in_h - params.start_h - params.end_h;
    let out_w = params.in_w - params.start_w - params.end_w;
    let total = params.batch * params.channels * out_h * out_w;
    if i >= total { return; }

    let ow = i % out_w;
    let oh = (i / out_w) % out_h;
    let c = (i / (out_w * out_h)) % params.channels;
    let n = i / (params.channels * out_h * out_w);

    let ih = oh + params.start_h;
    let iw = ow + params.start_w;
    let src_idx = ((n * params.channels + c) * params.in_h + ih) * params.in_w + iw;
    dst[i] = src[src_idx];
}
