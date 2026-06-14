// Dilate-with-zeros along the H axis.
// input[N,C,H,W] → output[N,C, H*stride_h - (stride_h - 1), W]
//
//   output[n, c, oh, w] = input[n, c, oh / stride_h, w]    if oh % stride_h == 0
//                       = 0                                otherwise
//
// Used together with `dilate_zeros_w` to lift ConvTranspose2D with
// stride > 1 on both axes to an equivalent forward Conv2D (dilate → pad →
// forward conv). Dispatch: [ceil(total / 256), 1, 1], workgroup_size(256).

struct Params {
    batch: u32,
    channels: u32,
    in_h: u32,
    in_w: u32,
    stride_h: u32,
    out_h: u32,
    _pad0: u32,
    _pad1: u32,
}

var<storage> src: array<f32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let total = params.batch * params.channels * params.out_h * params.in_w;
    if i >= total { return; }

    let w = i % params.in_w;
    let oh = (i / params.in_w) % params.out_h;
    let c = (i / (params.in_w * params.out_h)) % params.channels;
    let n = i / (params.channels * params.out_h * params.in_w);

    if (oh % params.stride_h) != 0u {
        dst[i] = 0.0;
        return;
    }
    let ih = oh / params.stride_h;
    let src_idx = ((n * params.channels + c) * params.in_h + ih) * params.in_w + w;
    dst[i] = src[src_idx];
}
