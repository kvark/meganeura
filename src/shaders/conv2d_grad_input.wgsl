// Conv2d backward w.r.t. input: grad_output[N,Co,oH,oW] × kernel[Co,Ci,kH,kW] → grad_input[N,Ci,H,W]
// This is a "full convolution" of grad_output with flipped kernel.
// Dispatch: [ceil(W/16), ceil(H/16), N*Ci]  workgroup_size(16,16,1)

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
var<storage> weight: array<f32>;      // kernel [Co,Ci,kH,kW]
var<storage, read_write> dst: array<f32>;  // grad_input [N,Ci,H,W]
var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let iw = gid.x;
    let ih = gid.y;
    let nci = gid.z;  // n * in_channels + ci

    if iw >= params.in_w || ih >= params.in_h { return; }

    let n = nci / params.in_channels;
    let ci = nci % params.in_channels;
    if n >= params.batch { return; }

    var sum = 0.0;

    for (var co = 0u; co < params.out_channels; co++) {
        for (var kh = 0u; kh < params.kernel_h; kh++) {
            for (var kw = 0u; kw < params.kernel_w; kw++) {
                // Which output position (oh, ow) used input (ih, iw) with kernel (kh, kw)?
                // ih = oh * stride + kh - padding  =>  oh = (ih + padding - kh) / stride
                let h_off = i32(ih) + i32(params.padding) - i32(kh);
                let w_off = i32(iw) + i32(params.padding) - i32(kw);

                if h_off >= 0 && w_off >= 0 && (h_off % i32(params.stride)) == 0 && (w_off % i32(params.stride)) == 0 {
                    let oh = u32(h_off) / params.stride;
                    let ow = u32(w_off) / params.stride;

                    if oh < params.out_h && ow < params.out_w {
                        let go_idx = ((n * params.out_channels + co) * params.out_h + oh) * params.out_w + ow;
                        let k_idx = ((co * params.in_channels + ci) * params.kernel_h + kh) * params.kernel_w + kw;
                        sum += grad_out[go_idx] * weight[k_idx];
                    }
                }
            }
        }
    }

    let in_idx = ((n * params.in_channels + ci) * params.in_h + ih) * params.in_w + iw;
    dst[in_idx] = sum;
}
