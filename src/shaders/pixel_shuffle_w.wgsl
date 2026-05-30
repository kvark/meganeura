// PixelShuffleW: channel-to-width pixel-shuffle for NCHW tensors.
// [B, C, H, W] → [B, C/factor, H, W*factor]
// Mapping: out[b, c, h, factor*w + k] = in[b, k*(C/factor) + c, h, w]

struct Params {
    batch: u32,
    channels: u32, // input channels
    in_h: u32,
    in_w: u32,
    factor: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

var<storage> src: array<f32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let out_w = params.in_w * params.factor;
    let out_c = params.channels / params.factor;
    let total = params.batch * out_c * params.in_h * out_w;
    if i >= total { return; }

    let ow = i % out_w;
    let oh = (i / out_w) % params.in_h;
    let oc = (i / (out_w * params.in_h)) % out_c;
    let n = i / (out_c * params.in_h * out_w);

    let w_in = ow / params.factor;
    let k = ow % params.factor;
    let c_in = k * out_c + oc;
    let src_idx = ((n * params.channels + c_in) * params.in_h + oh) * params.in_w + w_in;
    dst[i] = src[src_idx];
}
