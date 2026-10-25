pub(super) struct Staging {
    pub initialize: String,
    pub load: String,
    pub advance: String,
}

pub(super) fn generate(weight_gradient: bool) -> Staging {
    let mut initialize = if weight_gradient {
        include_str!("../../shaders/conv_weight_iterator.wgsl")
    } else {
        include_str!("../../shaders/conv_forward_iterator.wgsl")
    }
    .to_owned();
    let mut load = String::new();
    let mut advance = String::new();
    for slot in 0..8 {
        // Constant names/indices keep the eight independent cursors in registers.
        initialize += &format!(
            "let reduction_{slot} = first + lane / 32u + {offset}u;\n",
            offset = 4 * slot
        );
        if weight_gradient {
            initialize += &format!(
                "let image_{slot} = divide_exact(reduction_{slot}, spatial, params.output_spatial_multiplier);\n\
                 let position_{slot} = reduction_{slot} - image_{slot} * spatial;\n\
                 let h_{slot} = divide_exact(position_{slot}, params.out_w, params.column_width_multiplier);\n\
                 let w_{slot} = position_{slot} - h_{slot} * params.out_w;\n\
                 var address_{slot} = TileAddress(h_{slot}, w_{slot},\n\
                   (image_{slot} * params.in_channels + input_channel) * params.in_h * params.in_w +\n\
                   u32(i32(h_{slot} * STRIDE) + base_row) * params.in_w + u32(i32(w_{slot} * STRIDE) + base_column));\n"
            );
        } else {
            initialize += &format!(
                "let channel_{slot} = reduction_{slot} / (KH * KW);\n\
                 let kernel_{slot} = reduction_{slot} % (KH * KW);\n\
                 let h_{slot} = kernel_{slot} / KW;\n\
                 let w_{slot} = kernel_{slot} % KW;\n\
                 var address_{slot} = TileAddress(h_{slot}, w_{slot},\n\
                   image_offset + channel_{slot} * params.in_h * params.in_w +\n\
                   u32(base_row + i32(h_{slot})) * params.in_w + u32(base_column + i32(w_{slot})));\n"
            );
        }
        load += &format!(
            "{{\nlet index = lane + {offset}u;\n\
             let ar = row + index / 32u;\nlet ak = start + index % 32u;\n\
             let bk = start + index / 32u;\n\
             var a = $MULTIPLY_TYPE(0.0f);\nvar b = $MULTIPLY_TYPE(0.0f);\n\
             if ar < m && ak < k {{ a = $A_CAST($A_LOAD); }}\n\
             if bk < k && column < n && valid_address(address_{slot}, lower, upper) {{ b = $B_CAST(src[address_{slot}.offset]); }}\n\
             stage_a[index] = a;\nstage_b[index] = b;\n}}\n",
            offset = 128 * slot
        );
        advance += &format!(
            "address_{slot} = advance_address(address_{slot}, iterator_grid, iterator_step, iterator_delta);\n"
        );
    }
    if weight_gradient {
        advance += "gradient_position += 32u % spatial;\ngradient_offset += gradient_step;\n\
                    if gradient_position >= spatial { gradient_position -= spatial; gradient_offset += gradient_wrap; }\n";
    }
    Staging {
        initialize,
        load,
        advance,
    }
}
