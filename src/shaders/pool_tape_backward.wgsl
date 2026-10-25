fn covering_lo(i: u32, kernel: u32) -> u32 {
    let shifted = i + params.padding;
    return select(0u, (shifted + params.stride - kernel) / params.stride, shifted + 1u > kernel);
}

fn gradient(at: u32) -> f32 {
    let plane = at / (params.in_h * params.in_w);
    let ih = (at / params.in_w) % params.in_h;
    let iw = at % params.in_w;
    let hi_h = min((ih + params.padding) / params.stride, params.out_h - 1u);
    let hi_w = min((iw + params.padding) / params.stride, params.out_w - 1u);
    let base = plane * params.out_h * params.out_w;
    var acc = 0.0;
    for (var oh = covering_lo(ih, params.kernel_h); oh <= hi_h; oh++) {
        for (var ow = covering_lo(iw, params.kernel_w); ow <= hi_w; ow++) {
            let index = base + oh * params.out_w + ow;
            if winner[index] == at { acc += dy[index]; }
        }
    }
    return acc;
}
