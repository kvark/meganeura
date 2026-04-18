// Conv2d forward: input[N,Ci,H,W] * kernel[Co,Ci,kH,kW] → output[N,Co,oH,oW]
// Dispatch: [ceil(oW/16), ceil(oH/16), N*Co]  workgroup_size(16,16,1)
//
// Optimizations:
//   - Shared memory for weights: one cooperative load per (ci, kernel) instead of 256 redundant reads

struct Params {
    batch: u32,
    in_channels: u32,
    in_h: u32,
    in_w: u32,
    out_channels: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride: u32,
    padding_h: u32,
    out_h: u32,
    out_w: u32,
    padding_w: u32,
    inv_kernel_w: f32,
    inv_kernel_hw: f32,
    inv_col_w: f32,
    inv_go_spatial: f32,
}

var<storage> src: array<f32>;       // input
var<storage> weight: array<f32>;    // kernel
var<storage, read_write> dst: array<f32>;  // output
var<uniform> params: Params;

// Shared memory for kernel weights: up to 7×7 kernel
var<workgroup> wg_weight: array<f32, 49>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let ow = gid.x;
    let oh = gid.y;
    let nco = gid.z;  // n * out_channels + co

    let n = nco / params.out_channels;
    let co = nco % params.out_channels;
    let in_bounds = ow < params.out_w && oh < params.out_h && n < params.batch;

    let tid = lid.y * 16u + lid.x;
    let kernel_size = params.kernel_h * params.kernel_w;
    let i_padding_h = i32(params.padding_h);
    let i_padding_w = i32(params.padding_w);

    var sum = 0.0;

    for (var ci = 0u; ci < params.in_channels; ci++) {
        // Cooperative weight load into shared memory
        if tid < kernel_size {
            wg_weight[tid] = weight[(co * params.in_channels + ci) * kernel_size + tid];
        }
        workgroupBarrier();

        if in_bounds {
            for (var kh = 0u; kh < params.kernel_h; kh++) {
                for (var kw = 0u; kw < params.kernel_w; kw++) {
                    let ih = i32(oh * params.stride + kh) - i_padding_h;
                    let iw = i32(ow * params.stride + kw) - i_padding_w;

                    if ih >= 0 && u32(ih) < params.in_h && iw >= 0 && u32(iw) < params.in_w {
                        let in_idx = ((n * params.in_channels + ci) * params.in_h + u32(ih)) * params.in_w + u32(iw);
                        sum += src[in_idx] * wg_weight[kh * params.kernel_w + kw];
                    }
                }
            }
        }

        workgroupBarrier();
    }

    if in_bounds {
        let out_idx = ((n * params.out_channels + co) * params.out_h + oh) * params.out_w + ow;
        dst[out_idx] = sum;
    }
}
