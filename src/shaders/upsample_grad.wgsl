// Backward of nearest-neighbor 2x upsample: grad_output[N,C,2H,2W] → grad_input[N,C,H,W]
// Each input pixel receives the sum of 4 output gradients (the 2×2 block).
// Dispatch: [ceil(total_in / 256), 1, 1]  workgroup_size(256)

struct Params {
    batch: u32,
    channels: u32,
    in_h: u32,
    in_w: u32,
}

var<storage> src: array<f32>;        // grad_output [N,C,2H,2W]
var<storage, read_write> dst: array<f32>;  // grad_input [N,C,H,W]
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let total = params.batch * params.channels * params.in_h * params.in_w;
    if i >= total { return; }

    let iw = i % params.in_w;
    let ih = (i / params.in_w) % params.in_h;
    let c = (i / (params.in_w * params.in_h)) % params.channels;
    let n = i / (params.channels * params.in_h * params.in_w);

    let out_h = params.in_h * 2u;
    let out_w = params.in_w * 2u;
    let oh = ih * 2u;
    let ow = iw * 2u;

    // Sum the 2×2 block in grad_output
    let base = (n * params.channels + c) * out_h;
    let v00 = src[(base + oh) * out_w + ow];
    let v01 = src[(base + oh) * out_w + ow + 1u];
    let v10 = src[(base + oh + 1u) * out_w + ow];
    let v11 = src[(base + oh + 1u) * out_w + ow + 1u];

    dst[i] = v00 + v01 + v10 + v11;
}
