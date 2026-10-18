var<storage> src: array<$B_TYPE>;
var<storage> weight: array<$A_TYPE>;
var<storage, read_write> dst: array<$OUT_TYPE>;
var<uniform> params: Params;

fn load_pixel(image: u32, reduction: u32, column: u32) -> $B_TYPE {
    let channel = reduction / (KH * KW);
    let kernel = reduction % (KH * KW);
    let oh = divide_exact(column, params.out_w, params.column_width_multiplier);
    let ow = column - oh * params.out_w;
    let ih = i32(oh * STRIDE + kernel / KW) - i32(params.padding_h);
    let iw = i32(ow * STRIDE + kernel % KW) - i32(params.padding_w);
    if ih < 0 || iw < 0 || ih >= i32(params.in_h) || iw >= i32(params.in_w) {
        return $B_TYPE(0.0f);
    }
    return src[((image * params.in_channels + channel) * params.in_h + u32(ih)) * params.in_w + u32(iw)];
}
