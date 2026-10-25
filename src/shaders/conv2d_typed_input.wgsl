var<storage> grad_out: array<$B_TYPE>;
var<storage> weight: array<$A_TYPE>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;

fn load_weight(channel: u32, reduction: u32) -> $A_TYPE {
    let co = reduction / (KH * KW);
    let kernel = reduction % (KH * KW);
    return weight[(co * params.in_channels + channel) * KH * KW + kernel];
}

fn load_gradient(image: u32, reduction: u32, column: u32) -> $B_TYPE {
    let co = reduction / (KH * KW);
    let kernel = reduction % (KH * KW);
    let ih = divide_exact(column, params.in_w, params.column_width_multiplier);
    let iw = column - ih * params.in_w;
    let oh = i32(ih + params.padding_h) - i32(kernel / KW);
    let ow = i32(iw + params.padding_w) - i32(kernel % KW);
    if oh < 0 || ow < 0 || oh % i32(STRIDE) != 0 || ow % i32(STRIDE) != 0 {
        return $B_TYPE(0.0f);
    }
    let h = u32(oh) / STRIDE;
    let w = u32(ow) / STRIDE;
    if h >= params.out_h || w >= params.out_w { return $B_TYPE(0.0f); }
    return grad_out[((image * params.out_channels + co) * params.out_h + h) * params.out_w + w];
}
