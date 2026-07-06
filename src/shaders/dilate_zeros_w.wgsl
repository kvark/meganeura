// Dilate-with-zeros along the W axis.
// input[N,C,H,W] → output[N,C,H, W*stride_w - (stride_w - 1)]
//
//   output[n, c, h, ow] = input[n, c, h, ow / stride_w]    if ow % stride_w == 0
//                       = 0                                 otherwise
//
// Used to lift ConvTranspose2D to an equivalent forward Conv2D
// (`dilate → pad → forward conv`). Dispatch: [ceil(total / 256), 1, 1],
// workgroup_size(256).

struct Params {
    batch: u32,
    channels: u32,
    in_h: u32,
    in_w: u32,
    stride_w: u32,
    out_w: u32,
    _pad0: u32,
    _pad1: u32,
}

var<storage> src: array<f32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let total = params.batch * params.channels * params.in_h * params.out_w;
    if i >= total { return; }

    let ow = i % params.out_w;
    let h = (i / params.out_w) % params.in_h;
    let c = (i / (params.out_w * params.in_h)) % params.channels;
    let n = i / (params.channels * params.in_h * params.out_w);

    if (ow % params.stride_w) != 0u {
        dst[i] = 0.0;
        return;
    }
    let iw = ow / params.stride_w;
    let src_idx = ((n * params.channels + c) * params.in_h + h) * params.in_w + iw;
    dst[i] = src[src_idx];
}
