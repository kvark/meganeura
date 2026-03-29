// Conv2d forward: input[N,Ci,H,W] * kernel[Co,Ci,kH,kW] → output[N,Co,oH,oW]
// Dispatch: [ceil(oW/16), ceil(oH/16), N*Co]  workgroup_size(16,16,1)

struct Params {
    batch: u32,
    in_channels: u32,
    in_h: u32,
    in_w: u32,
    out_channels: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride: u32,
    padding: u32,
    out_h: u32,
    out_w: u32,
    _pad: u32,
}

var<storage> src: array<f32>;       // input
var<storage> weight: array<f32>;    // kernel
var<storage, read_write> dst: array<f32>;  // output
var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ow = gid.x;
    let oh = gid.y;
    let nco = gid.z;  // n * out_channels + co

    if ow >= params.out_w || oh >= params.out_h { return; }

    let n = nco / params.out_channels;
    let co = nco % params.out_channels;
    if n >= params.batch { return; }

    var sum = 0.0;

    for (var ci = 0u; ci < params.in_channels; ci++) {
        for (var kh = 0u; kh < params.kernel_h; kh++) {
            for (var kw = 0u; kw < params.kernel_w; kw++) {
                let ih = oh * params.stride + kh;
                let iw = ow * params.stride + kw;

                // Check padding bounds
                let h = i32(ih) - i32(params.padding);
                let w = i32(iw) - i32(params.padding);

                if h >= 0 && u32(h) < params.in_h && w >= 0 && u32(w) < params.in_w {
                    let in_idx = ((n * params.in_channels + ci) * params.in_h + u32(h)) * params.in_w + u32(w);
                    let k_idx = ((co * params.in_channels + ci) * params.kernel_h + kh) * params.kernel_w + kw;
                    sum += src[in_idx] * weight[k_idx];
                }
            }
        }
    }

    let out_idx = ((n * params.out_channels + co) * params.out_h + oh) * params.out_w + ow;
    dst[out_idx] = sum;
}
