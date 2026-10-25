var<storage> grad_out: array<$A_TYPE>;
var<storage> src: array<$B_TYPE>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;

fn load_gradient(channel: u32, reduction: u32) -> $A_TYPE {
    let spatial = params.out_h * params.out_w;
    let image = divide_exact(reduction, spatial, params.output_spatial_multiplier);
    let position = reduction - image * spatial;
    return grad_out[(image * params.out_channels + channel) * spatial + position];
}

fn load_input(reduction: u32, column: u32) -> $B_TYPE {
    let channel = column / (KH * KW);
    let kernel = column % (KH * KW);
    let spatial = params.out_h * params.out_w;
    let image = divide_exact(reduction, spatial, params.output_spatial_multiplier);
    let position = reduction - image * spatial;
    let oh = divide_exact(position, params.out_w, params.column_width_multiplier);
    let ow = position - oh * params.out_w;
    let ih = i32(oh * STRIDE + kernel / KW) - i32(params.padding_h);
    let iw = i32(ow * STRIDE + kernel % KW) - i32(params.padding_w);
    if ih < 0 || iw < 0 || ih >= i32(params.in_h) || iw >= i32(params.in_w) {
        return $B_TYPE(0.0f);
    }
    return src[((image * params.in_channels + channel) * params.in_h + u32(ih)) * params.in_w + u32(iw)];
}
