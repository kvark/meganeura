use super::{DType, MatrixTypes, ShaderGroup};

pub(super) struct Staging {
    pub declaration: String,
    pub initialize: String,
    pub load: String,
    pub advance: String,
    pub multiply: String,
}

pub(super) fn generate(group: ShaderGroup, types: MatrixTypes) -> Staging {
    let (dx, dw) = match group {
        ShaderGroup::Conv2dBF16 { .. } => (false, false),
        ShaderGroup::Conv2dGradientTyped {
            weight_gradient, ..
        } => (!weight_gradient, weight_gradient),
        _ => unreachable!(),
    };
    let pitch = if types.multiply == DType::F32 { 36 } else { 40 };
    let vectors = pitch / 4;
    let mut initialize = String::from("let vc = (lane % 8u) * 4u;\n");
    let mut load = String::new();
    let mut advance = String::new();
    if dw {
        initialize += "let spatial = params.out_h * params.out_w;\n\
            let iterator_grid = vec2<u32>(params.out_h, params.out_w);\n\
            let iterator_step = vec2<u32>((32u % spatial) / params.out_w, 32u % params.out_w);\n\
            let image_stride = params.in_channels * params.in_h * params.in_w;\n\
            let iterator_delta = vec3<u32>((32u / spatial) * image_stride + STRIDE * (iterator_step.x * params.in_w + iterator_step.y),\n\
                STRIDE * (params.in_w - params.out_w), image_stride - params.out_h * STRIDE * params.in_w);\n\
            let gradient_step = (32u / spatial) * params.out_channels * spatial + 32u % spatial;\n\
            let gradient_wrap = (params.out_channels - 1u) * spatial;\n";
    } else {
        initialize += "let iterator_grid = vec2<u32>(KH, KW);\n\
            let iterator_step = vec2<u32>((32u % (KH * KW)) / KW, 32u % KW);\n\
            let channel_step = 32u / (KH * KW);\n";
        initialize += if dx {
            "let gradient_spatial = params.out_h * params.out_w;\n\
             let image_offset = group.z * params.out_channels * gradient_spatial;\n\
             let iterator_delta = vec3<u32>(channel_step * gradient_spatial, 0u, gradient_spatial);\n"
        } else {
            "let image_offset = group.z * params.in_channels * params.in_h * params.in_w;\n\
             let iterator_delta = vec3<u32>(channel_step * params.in_h * params.in_w + iterator_step.x * params.in_w + iterator_step.y,\n\
                params.in_w - KW, params.in_h * params.in_w - KH * params.in_w);\n"
        };
    }
    for component in 0..4 {
        initialize += &format!("let column_{component} = col + vc + {component}u;\n");
        if dx {
            initialize += &format!(
                "let ih_{component} = divide_exact(column_{component}, params.in_w, params.column_width_multiplier);\n\
                 let iw_{component} = column_{component} - ih_{component} * params.in_w;\n\
                 let base_row_{component} = i32(ih_{component} + params.padding_h);\n\
                 let base_column_{component} = i32(iw_{component} + params.padding_w);\n"
            );
        } else {
            if dw {
                initialize += &format!(
                    "let channel_{component} = column_{component} / (KH * KW);\n\
                     let kernel_{component} = column_{component} % (KH * KW);\n\
                     let base_row_{component} = i32(kernel_{component} / KW) - i32(params.padding_h);\n\
                     let base_column_{component} = i32(kernel_{component} % KW) - i32(params.padding_w);\n\
                     let base_offset_{component} = channel_{component} * params.in_h * params.in_w + u32(base_row_{component}) * params.in_w + u32(base_column_{component});\n\
                     let gradient_first_{component} = first + vc + {component}u;\n\
                     let gradient_image_{component} = divide_exact(gradient_first_{component}, spatial, params.output_spatial_multiplier);\n\
                     var gradient_position_{component} = gradient_first_{component} - gradient_image_{component} * spatial;\n\
                     var gradient_offset_{component} = gradient_image_{component} * params.out_channels * spatial + gradient_position_{component};\n"
                );
                advance += &format!(
                    "gradient_position_{component} += 32u % spatial;\n\
                     gradient_offset_{component} += gradient_step;\n\
                     if gradient_position_{component} >= spatial {{ gradient_position_{component} -= spatial; gradient_offset_{component} += gradient_wrap; }}\n"
                );
            } else {
                initialize += &format!(
                    "let oh_{component} = divide_exact(column_{component}, params.out_w, params.column_width_multiplier);\n\
                     let ow_{component} = column_{component} - oh_{component} * params.out_w;\n\
                     let base_row_{component} = i32(oh_{component} * STRIDE) - i32(params.padding_h);\n\
                     let base_column_{component} = i32(ow_{component} * STRIDE) - i32(params.padding_w);\n\
                     let base_offset_{component} = u32(base_row_{component}) * params.in_w + u32(base_column_{component});\n"
                );
            }
            let divisor = if dw { "i32(STRIDE)" } else { "1i" };
            initialize += &format!(
                "let lower_{component} = vec2<u32>(u32(max(0, (-base_row_{component} + {divisor} - 1) / {divisor})), u32(max(0, (-base_column_{component} + {divisor} - 1) / {divisor})));\n\
                 let upper_{component} = vec2<u32>(u32(max(0, (i32(params.in_h) - base_row_{component} + {divisor} - 1) / {divisor})), u32(max(0, (i32(params.in_w) - base_column_{component} + {divisor} - 1) / {divisor})));\n"
            );
        }
    }
    for part in 0..2 {
        initialize += &format!(
            "let reduction_{part} = first + lane / 8u + {}u;\n",
            part * 16
        );
        if dw {
            initialize += &format!(
                "let image_{part} = divide_exact(reduction_{part}, spatial, params.output_spatial_multiplier);\n\
                 let position_{part} = reduction_{part} - image_{part} * spatial;\n\
                 let h_{part} = divide_exact(position_{part}, params.out_w, params.column_width_multiplier);\n\
                 let w_{part} = position_{part} - h_{part} * params.out_w;\n\
                 var address_{part} = TileAddress(h_{part}, w_{part}, image_{part} * image_stride + STRIDE * (h_{part} * params.in_w + w_{part}));\n"
            );
        } else {
            let offset = if dx {
                format!("image_offset + channel_{part} * gradient_spatial")
            } else {
                format!(
                    "image_offset + channel_{part} * params.in_h * params.in_w + h_{part} * params.in_w + w_{part}"
                )
            };
            initialize += &format!(
                "let channel_{part} = reduction_{part} / (KH * KW);\n\
                 let kernel_{part} = reduction_{part} % (KH * KW);\n\
                 let h_{part} = kernel_{part} / KW;\nlet w_{part} = kernel_{part} % KW;\n\
                 var address_{part} = TileAddress(h_{part}, w_{part}, {offset});\n"
            );
        }
        advance += &format!(
            "address_{part} = advance_address(address_{part}, iterator_grid, iterator_step, iterator_delta);\n"
        );
        load += &format!(
            "{{ let vr = lane / 8u + {}u;\nlet ar = row + vr;\nlet bk = start + vr;\nvar a = vec4<$MULTIPLY_TYPE>($MULTIPLY_TYPE(0.0f));\nvar b = vec4<$MULTIPLY_TYPE>($MULTIPLY_TYPE(0.0f));\n",
            part * 16
        );
        for (component, field) in ["x", "y", "z", "w"].into_iter().enumerate() {
            let a = if dw {
                format!("grad_out[gradient_offset_{component} + ar * spatial]")
            } else if dx {
                "load_weight(ar, ak)".to_owned()
            } else {
                "weight[ar * k + ak]".to_owned()
            };
            load += &format!(
                "{{ let ak = start + vc + {component}u;\nif all(vec2<bool>((ar < m), (ak < k))) {{ a.{field} = $A_CAST({a}); }}\n"
            );
            if dx {
                load += &format!(
                    "let oh = base_row_{component} - i32(address_{part}.row);\n\
                     let ow = base_column_{component} - i32(address_{part}.column);\n\
                     if all(vec4<bool>((bk < k), (column_{component} < n), (oh >= 0), (ow >= 0))) {{\n\
                       if all(vec2<bool>((oh % i32(STRIDE) == 0), (ow % i32(STRIDE) == 0))) {{\n\
                       let h = u32(oh) / STRIDE; let w = u32(ow) / STRIDE;\n\
                       if all(vec2<bool>((h < params.out_h), (w < params.out_w))) {{ b.{field} = $B_CAST(grad_out[address_{part}.offset + h * params.out_w + w]); }}\n}}\n}}\n"
                );
            } else {
                load += &format!(
                    "if all(vec3<bool>((bk < k), (column_{component} < n), valid_address(address_{part}, lower_{component}, upper_{component}))) {{ b.{field} = $B_CAST(src[address_{part}.offset + base_offset_{component}]); }}\n"
                );
            }
            load += "}\n";
        }
        load += &format!(
            "stage_a[vr * {vectors}u + lane % 8u] = a;\nstage_b[vr * {vectors}u + lane % 8u] = b;\n}}\n"
        );
    }
    let mut multiply = String::new();
    for part in 0..2 {
        multiply += &format!(
            "let a{part} = coopLoadT<coop_mat16x16<$MULTIPLY_TYPE, A>>(&stage_a[(((tile % 4u) / 2u) * 16u * {pitch}u + {k}u) / 4u], {vectors}u);\n\
             let b{part} = coopLoadT<coop_mat16x16<$MULTIPLY_TYPE, B>>(&stage_b[({k}u * {pitch}u + b_offset) / 4u], {vectors}u);\n\
             accumulator = coopMultiplyAdd(a{part}, b{part}, accumulator);\n",
            k = part * 16
        );
    }
    Staging {
        declaration: format!(
            "var<workgroup> stage_a: array<vec4<$MULTIPLY_TYPE>, {count}>;\nvar<workgroup> stage_b: array<vec4<$MULTIPLY_TYPE>, {count}>;",
            count = 32 * vectors
        ),
        initialize,
        load,
        advance,
        multiply,
    }
}
