let column = col + lane % 32u;
let input_channel = column / (KH * KW);
let kernel = column % (KH * KW);
let base_row = i32(kernel / KW) - i32(params.padding_h);
let base_column = i32(kernel % KW) - i32(params.padding_w);
let lower = vec2<u32>(u32(max(0, (-base_row + i32(STRIDE) - 1) / i32(STRIDE))),
    u32(max(0, (-base_column + i32(STRIDE) - 1) / i32(STRIDE))));
let upper = vec2<u32>(u32(max(0, (i32(params.in_h) - base_row + i32(STRIDE) - 1) / i32(STRIDE))),
    u32(max(0, (i32(params.in_w) - base_column + i32(STRIDE) - 1) / i32(STRIDE))));
let spatial = params.out_h * params.out_w;
let iterator_grid = vec2<u32>(params.out_h, params.out_w);
let iterator_step = vec2<u32>((32u % spatial) / params.out_w, 32u % params.out_w);
let image_step = 32u / spatial;
let image_stride = params.in_channels * params.in_h * params.in_w;
let iterator_delta = vec3<u32>(image_step * image_stride + STRIDE * (iterator_step.x * params.in_w + iterator_step.y),
    STRIDE * (params.in_w - params.out_w), image_stride - params.out_h * STRIDE * params.in_w);
let first_gradient = first + lane % 32u;
let first_image = divide_exact(first_gradient, spatial, params.output_spatial_multiplier);
var gradient_position = first_gradient - first_image * spatial;
var gradient_offset = first_image * params.out_channels * spatial + gradient_position;
let gradient_step = (32u / spatial) * params.out_channels * spatial + 32u % spatial;
let gradient_wrap = (params.out_channels - 1u) * spatial;
