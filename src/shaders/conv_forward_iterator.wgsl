let column = col + lane % 32u;
let output_row = divide_exact(column, params.out_w, params.column_width_multiplier);
let output_column = column - output_row * params.out_w;
let base_row = i32(output_row * STRIDE) - i32(params.padding_h);
let base_column = i32(output_column * STRIDE) - i32(params.padding_w);
let lower = vec2<u32>(u32(max(0, -base_row)), u32(max(0, -base_column)));
let upper = vec2<u32>(u32(max(0, i32(params.in_h) - base_row)), u32(max(0, i32(params.in_w) - base_column)));
let iterator_grid = vec2<u32>(KH, KW);
let iterator_step = vec2<u32>((32u % (KH * KW)) / KW, 32u % KW);
let channel_step = 32u / (KH * KW);
let iterator_delta = vec3<u32>(channel_step * params.in_h * params.in_w + iterator_step.x * params.in_w + iterator_step.y,
    params.in_w - KW, params.in_h * params.in_w - KH * params.in_w);
let image_offset = group.z * params.in_channels * params.in_h * params.in_w;
