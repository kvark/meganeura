struct Params {
    batch: u32,
    channels: u32,
    in_h: u32,
    in_w: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride: u32,
    padding: u32,
    out_h: u32,
    out_w: u32,
    _pad0: u32,
    _pad1: u32,
}

var<storage> src: array<f32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;

@compute @workgroup_size(256)
fn max_pool_2d(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.batch * params.channels * params.out_h * params.out_w;
    if idx >= total { return; }

    let ow = idx % params.out_w;
    let oh = (idx / params.out_w) % params.out_h;
    let c = (idx / (params.out_w * params.out_h)) % params.channels;
    let n = idx / (params.out_w * params.out_h * params.channels);

    var max_val = -3.402823e+38;
    for (var kh = 0u; kh < params.kernel_h; kh++) {
        for (var kw = 0u; kw < params.kernel_w; kw++) {
            let ih = oh * params.stride + kh - params.padding;
            let iw = ow * params.stride + kw - params.padding;
            if ih < params.in_h && iw < params.in_w {
                let src_idx = ((n * params.channels + c) * params.in_h + ih) * params.in_w + iw;
                max_val = max(max_val, src[src_idx]);
            }
        }
    }
    dst[idx] = max_val;
}
