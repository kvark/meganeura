// Conv2d backward w.r.t. input: grad_output[N,Co,oH,oW] × kernel[Co,Ci,kH,kW] → grad_input[N,Ci,H,W]
// This is a "full convolution" of grad_output with flipped kernel.
// Dispatch: [ceil(W/16), ceil(H/16), N*Ci]  workgroup_size(16,16,1)
//
// Optimizations:
//   - Stride-1 fast path: eliminates modulo/division per iteration
//   - Shared memory for weights: one cooperative load per (co, kernel) instead of 256 redundant reads

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

// Shared memory for kernel weights: up to 7×7 kernel (practically ≤ 3×3)
var<workgroup> wg_weight: array<f32, 49>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let iw = gid.x;
    let ih = gid.y;
    let nci = gid.z;  // n * in_channels + ci

    let n = nci / params.in_channels;
    let ci = nci % params.in_channels;
    let in_bounds = iw < params.in_w && ih < params.in_h && n < params.batch;

    let tid = lid.y * 16u + lid.x;
    let kernel_size = params.kernel_h * params.kernel_w;
    let i_padding = i32(params.padding);

    var sum = 0.0;

    if params.stride == 1u {
        // Fast path for stride=1: no modulo/division, simplified bounds
        for (var co = 0u; co < params.out_channels; co++) {
            // Cooperative weight load into shared memory
            if tid < kernel_size {
                wg_weight[tid] = weight[(co * params.in_channels + ci) * kernel_size + tid];
            }
            workgroupBarrier();

            if in_bounds {
                let go_base = (n * params.out_channels + co) * params.out_h * params.out_w;

                for (var kh = 0u; kh < params.kernel_h; kh++) {
                    let oh = i32(ih) + i_padding - i32(kh);
                    if oh >= 0 && u32(oh) < params.out_h {
                        for (var kw = 0u; kw < params.kernel_w; kw++) {
                            let ow = i32(iw) + i_padding - i32(kw);
                            if ow >= 0 && u32(ow) < params.out_w {
                                sum += grad_out[go_base + u32(oh) * params.out_w + u32(ow)]
                                     * wg_weight[kh * params.kernel_w + kw];
                            }
                        }
                    }
                }
            }

            workgroupBarrier();
        }
    } else {
        // General path for stride > 1
        for (var co = 0u; co < params.out_channels; co++) {
            if tid < kernel_size {
                wg_weight[tid] = weight[(co * params.in_channels + ci) * kernel_size + tid];
            }
            workgroupBarrier();

            if in_bounds {
                let go_base = (n * params.out_channels + co) * params.out_h * params.out_w;
                let i_stride = i32(params.stride);

                for (var kh = 0u; kh < params.kernel_h; kh++) {
                    let h_off = i32(ih) + i_padding - i32(kh);
                    if h_off >= 0 && (h_off % i_stride) == 0 {
                        let oh = u32(h_off) / params.stride;
                        if oh < params.out_h {
                            for (var kw = 0u; kw < params.kernel_w; kw++) {
                                let w_off = i32(iw) + i_padding - i32(kw);
                                if w_off >= 0 && (w_off % i_stride) == 0 {
                                    let ow = u32(w_off) / params.stride;
                                    if ow < params.out_w {
                                        sum += grad_out[go_base + oh * params.out_w + ow]
                                             * wg_weight[kh * params.kernel_w + kw];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            workgroupBarrier();
        }
    }

    if in_bounds {
        let in_idx = ((n * params.in_channels + ci) * params.in_h + ih) * params.in_w + iw;
        dst[in_idx] = sum;
    }
}
