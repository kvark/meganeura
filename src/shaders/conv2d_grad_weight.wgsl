// Conv2d backward w.r.t. kernel: grad_output[N,Co,oH,oW] × input[N,Ci,H,W] → grad_kernel[Co,Ci,kH,kW]
// Dispatch: [ceil(kW * Ci / 16), ceil(kH / 16), Co]  workgroup_size(16,16,1)
// Accumulates over batch and spatial output positions.

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

var<storage> grad_out: array<f32>;    // grad_output [N,Co,oH,oW]
var<storage> src: array<f32>;         // input [N,Ci,H,W]
var<storage, read_write> dst: array<f32>;  // grad_kernel [Co,Ci,kH,kW]
var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cikw = gid.x;  // ci * kernel_w + kw
    let kh = gid.y;
    let co = gid.z;

    let ci = cikw / params.kernel_w;
    let kw = cikw % params.kernel_w;

    if ci >= params.in_channels || kh >= params.kernel_h || co >= params.out_channels { return; }

    var sum = 0.0;

    for (var n = 0u; n < params.batch; n++) {
        for (var oh = 0u; oh < params.out_h; oh++) {
            for (var ow = 0u; ow < params.out_w; ow++) {
                let ih = i32(oh * params.stride + kh) - i32(params.padding);
                let iw = i32(ow * params.stride + kw) - i32(params.padding);

                if ih >= 0 && u32(ih) < params.in_h && iw >= 0 && u32(iw) < params.in_w {
                    let go_idx = ((n * params.out_channels + co) * params.out_h + oh) * params.out_w + ow;
                    let in_idx = ((n * params.in_channels + ci) * params.in_h + u32(ih)) * params.in_w + u32(iw);
                    sum += grad_out[go_idx] * src[in_idx];
                }
            }
        }
    }

    let k_idx = ((co * params.in_channels + ci) * params.kernel_h + kh) * params.kernel_w + kw;
    dst[k_idx] = sum;
}
