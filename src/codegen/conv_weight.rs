use super::{CoopConfig, ShaderModule, preprocess};

pub(crate) fn generate_conv_weight_coop(config: &CoopConfig) -> ShaderModule {
    let tile = config.tile_size;
    assert!(matches!(tile, 8 | 16));
    let source = preprocess(
        include_str!("../shaders/conv2d_grad_weight_coop.wgsl"),
        &[
            (
                "$ENABLE_F16",
                if config.use_f16_input {
                    "enable f16;"
                } else {
                    ""
                },
            ),
            ("$ELEM", if config.use_f16_input { "f16" } else { "f32" }),
            ("$PARAMS", include_str!("../shaders/conv2d_params.wgsl")),
            ("$DIVISOR", crate::divisor::SHADER),
            ("$DIGITS", include_str!("../shaders/digits.wgsl")),
            ("$TILE", &tile.to_string()),
            ("$WIDTH", &(tile * 2).to_string()),
            ("$OPERAND_SIZE", &(tile * tile * 2).to_string()),
            ("$RESULT_SIZE", &(tile * tile * 4).to_string()),
        ],
    );
    ShaderModule::new(&source)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modules_validate() {
        for (tile_size, use_f16_input) in [(8, false), (16, false), (16, true)] {
            let module = generate_conv_weight_coop(&CoopConfig {
                tile_size,
                use_f16_input,
                compensated: false,
            });
            let flags =
                naga::valid::ValidationFlags::all() ^ naga::valid::ValidationFlags::BINDINGS;
            let capabilities = naga::valid::Capabilities::COOPERATIVE_MATRIX
                | naga::valid::Capabilities::SHADER_FLOAT16
                | naga::valid::Capabilities::SUBGROUP;
            naga::valid::Validator::new(flags, capabilities)
                .validate(&module.module)
                .unwrap();
        }
    }
}
