struct Params {
    batch: u32,
    in_channels: u32,
    in_h: u32,
    in_w: u32,
    out_channels: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride_h: u32,
    padding_h: u32,
    out_h: u32,
    out_w: u32,
    padding_w: u32,
    stride_w: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

var<storage> grad_out: array<f32>;
var<storage> weight: array<f32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;

var<workgroup> wg_weight: array<f32, 49>;

@compute @workgroup_size(16, 16, 1)
fn conv_t(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let iw = gid.x;
    let ih = gid.y;
    let nci = gid.z;
    let n = nci / params.in_channels;
    let ci = nci % params.in_channels;
    let in_bounds = iw < params.in_w && ih < params.in_h && n < params.batch;
    let tid = lid.y * 16u + lid.x;
    let kernel_size = params.kernel_h * params.kernel_w;
    let i_padding_h = i32(params.padding_h);
    let i_padding_w = i32(params.padding_w);
    let i_stride_h = i32(params.stride_h);
    let i_stride_w = i32(params.stride_w);
    var sum = 0.0;
    for (var co = 0u; co < params.out_channels; co++) {
        if tid < kernel_size {
            wg_weight[tid] = weight[(co * params.in_channels + ci) * kernel_size + tid];
        }
        workgroupBarrier();
        if in_bounds {
            let go_base = (n * params.out_channels + co) * params.out_h * params.out_w;
            for (var kh = 0u; kh < params.kernel_h; kh++) {
                let h_off = i32(ih) + i_padding_h - i32(kh);
                if h_off >= 0 && (h_off % i_stride_h) == 0 {
                    let oh = u32(h_off) / params.stride_h;
                    if oh < params.out_h {
                        for (var kw = 0u; kw < params.kernel_w; kw++) {
                            let w_off = i32(iw) + i_padding_w - i32(kw);
                            if w_off >= 0 && (w_off % i_stride_w) == 0 {
                                let ow = u32(w_off) / params.stride_w;
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
    if in_bounds {
        let in_idx = ((n * params.in_channels + ci) * params.in_h + ih) * params.in_w + iw;
        dst[in_idx] = sum;
    }
}

// Slice2D — input[N,C,in_h,in_w] → output[N,C, in_h-start_h-end_h, in_w-start_w-end_w]
struct SliceParams {
    batch: u32,
    channels: u32,
    in_h: u32,
    in_w: u32,
    start_h: u32,
    end_h: u32,
    start_w: u32,
    end_w: u32,
}

var<storage> slice_src: array<f32>;
var<storage, read_write> slice_dst: array<f32>;
var<uniform> slice_params: SliceParams;

@compute @workgroup_size(256)
fn slice(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let out_h = slice_params.in_h - slice_params.start_h - slice_params.end_h;
    let out_w = slice_params.in_w - slice_params.start_w - slice_params.end_w;
    let total = slice_params.batch * slice_params.channels * out_h * out_w;
    if i >= total { return; }

    let ow = i % out_w;
    let oh = (i / out_w) % out_h;
    let c = (i / (out_w * out_h)) % slice_params.channels;
    let n = i / (slice_params.channels * out_h * out_w);

    let ih = oh + slice_params.start_h;
    let iw = ow + slice_params.start_w;
    let src_idx = ((n * slice_params.channels + c) * slice_params.in_h + ih) * slice_params.in_w + iw;
    slice_dst[i] = slice_src[src_idx];
}
