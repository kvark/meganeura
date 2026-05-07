// Depthwise Conv2d forward (groups == channels).
//
// input shape:  [N, C, H, W]
// weight shape: [C, 1, kH, kW]   — one filter per channel
// output shape: [N, C, oH, oW]
//
// Each output element reads ONE input channel and convolves with that
// channel's kH×kW filter. No cross-channel summation (contrast: regular
// Conv2d in conv2d.wgsl loops over all in_channels).
//
// Dispatch: [ceil(oW/16), ceil(oH/16), N*C]  workgroup_size(16,16,1)
//
// Used by EfficientNet's MBConv blocks (depthwise k=3 or k=5) — see
// docs/kindle_local_dev.md Phase 2a.

struct Params {
    batch: u32,
    channels: u32,
    in_h: u32,
    in_w: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride: u32,
    padding_h: u32,
    out_h: u32,
    out_w: u32,
    padding_w: u32,
    _pad: u32,
}

var<storage> src: array<f32>;       // input  [N, C, H, W]
var<storage> weight: array<f32>;    // kernel [C, kH, kW] (Ci=1 collapsed)
var<storage, read_write> dst: array<f32>;  // output [N, C, oH, oW]
var<uniform> params: Params;

// Shared memory for the per-channel kernel (≤ 7×7 = 49 f32).
// One cooperative load per (n, c) workgroup instead of 256 redundant reads.
var<workgroup> wg_weight: array<f32, 49>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let ow = gid.x;
    let oh = gid.y;
    let nc = gid.z;  // n * channels + c

    let n = nc / params.channels;
    let c = nc % params.channels;
    let in_bounds = ow < params.out_w && oh < params.out_h && n < params.batch;

    let tid = lid.y * 16u + lid.x;
    let kernel_size = params.kernel_h * params.kernel_w;
    let i_padding_h = i32(params.padding_h);
    let i_padding_w = i32(params.padding_w);

    // Cooperative weight load — one filter (one channel) into shared memory.
    if tid < kernel_size {
        wg_weight[tid] = weight[c * kernel_size + tid];
    }
    workgroupBarrier();

    if !in_bounds {
        return;
    }

    var sum = 0.0;
    for (var kh = 0u; kh < params.kernel_h; kh++) {
        for (var kw = 0u; kw < params.kernel_w; kw++) {
            let ih = i32(oh * params.stride + kh) - i_padding_h;
            let iw = i32(ow * params.stride + kw) - i_padding_w;

            if ih >= 0 && u32(ih) < params.in_h && iw >= 0 && u32(iw) < params.in_w {
                let in_idx = ((n * params.channels + c) * params.in_h + u32(ih)) * params.in_w + u32(iw);
                sum += src[in_idx] * wg_weight[kh * params.kernel_w + kw];
            }
        }
    }

    let out_idx = ((n * params.channels + c) * params.out_h + oh) * params.out_w + ow;
    dst[out_idx] = sum;
}
