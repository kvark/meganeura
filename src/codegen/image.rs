use crate::DType;
use std::{
    fmt::Write,
    hash::{Hash, Hasher},
};

/// Channel normalization/activation with an optional bias and 2x2 spatial
/// reduction. Physical images remain NCHW; arithmetic and saved values are F32.
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ImageKernel {
    pub channels: u32,
    pub pool: bool,
    pub bias: bool,
    pub backward: bool,
    /// Forward input or backward output storage. No arithmetic policy change.
    pub storage: DType,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub view: Option<ImageView>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ImageView {
    pub storage: DType,
    pub packed: bool,
    pub expansion_width: u32,
    pub reduction_width: u32,
}

impl ImageKernel {
    pub fn rows_per_workgroup(&self) -> u32 {
        assert!((1..=256).contains(&self.channels));
        256 / self.channels.next_power_of_two()
    }

    pub(crate) fn hash_key(&self) -> u64 {
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        self.channels.hash(&mut hash);
        self.pool.hash(&mut hash);
        self.bias.hash(&mut hash);
        self.backward.hash(&mut hash);
        self.storage.hash(&mut hash);
        if let Some(view) = self.view.as_ref() {
            view.hash(&mut hash);
        }
        hash.finish()
    }

    pub(crate) fn binding_names(&self) -> Vec<&'static str> {
        if self.backward {
            let mut names = vec!["dy", "pooled", "inverse", "scale"];
            if self.pool {
                names.push("winner");
            }
            names.extend(["dst", "partial_scale"]);
            if self.bias {
                names.push("partial_bias");
            }
            names
        } else {
            let mut names = vec!["src", "scale"];
            if self.bias {
                names.push("bias");
            }
            names.extend(["dst", "pooled", "inverse"]);
            if self.pool {
                names.push("winner");
            }
            names
        }
    }

    pub fn generate(&self) -> super::ShaderModule {
        let ty = match self.storage {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            _ => panic!("image storage {:?}", self.storage),
        };
        let output_storage = self.view.as_ref().map_or(
            if self.backward {
                self.storage
            } else {
                DType::F32
            },
            |v| v.storage,
        );
        let output_ty = match output_storage {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            _ => panic!("image output storage {output_storage:?}"),
        };
        let mut code = String::new();
        if self.storage == DType::BF16 || output_storage == DType::BF16 {
            code.push_str("enable wgpu_bfloat16;\n");
        }
        if self.storage == DType::F16 || output_storage == DType::F16 {
            code.push_str("enable f16;\n");
        }
        if self.storage == DType::BF16 || output_storage == DType::BF16 {
            code.push_str(include_str!("../shaders/bf16_round.wgsl"));
        }
        writeln!(code, "const CHANNELS: u32 = {}u;\nconst ROWS: u32 = {}u;\nconst LANES: u32 = {}u;\nconst WINDOW: u32 = {}u;",
            self.channels, self.rows_per_workgroup(), self.channels.next_power_of_two(), if self.pool {2} else {1}).unwrap();
        code.push_str(include_str!("../shaders/image_region.wgsl"));
        if self.backward {
            code.push_str("var<storage> dy: array<f32>;\nvar<storage> pooled: array<f32>;\nvar<storage> inverse: array<f32>;\nvar<storage> scale: array<f32>;\n");
            if self.pool {
                code.push_str("var<storage> winner: array<u32>;\n");
            }
            writeln!(code, "var<storage, read_write> dst: array<{output_ty}>;\nvar<storage, read_write> partial_scale: array<f32>;").unwrap();
            if self.bias {
                code.push_str("var<storage, read_write> partial_bias: array<f32>;\n");
            }
            let store = match self.storage {
                DType::BF16 => "round_bf16(value)",
                DType::F16 => "f16(value)",
                _ => "value",
            };
            let winner = if self.pool { "winner[at]" } else { "0u" };
            let bias_store = if self.bias {
                "partial_bias[group.x * CHANNELS + channel] = scratch[lane];"
            } else {
                ""
            };
            code.push_str(
                &include_str!("../shaders/image_region_backward.wgsl")
                    .replace("$WINNER", winner)
                    .replace("$STORE", store)
                    .replace("$BIAS_STORE", bias_store),
            );
        } else {
            writeln!(
                code,
                "var<storage> src: array<{ty}>;\nvar<storage> scale: array<f32>;"
            )
            .unwrap();
            if self.bias {
                code.push_str("var<storage> bias: array<f32>;\n");
            }
            writeln!(code, "var<storage, read_write> dst: array<{output_ty}>;\nvar<storage, read_write> pooled: array<f32>;\nvar<storage, read_write> inverse: array<f32>;").unwrap();
            if self.pool {
                code.push_str("var<storage, read_write> winner: array<u32>;\n");
            }
            code.push_str(
                &include_str!("../shaders/image_region_forward.wgsl")
                    .replace("$BIAS", if self.bias { "bias[channel]" } else { "0.0" })
                    .replace(
                        "$SAVE_WINNER",
                        if self.pool {
                            "winner[at] = chosen;"
                        } else {
                            ""
                        },
                    ),
            );
        }
        if let Some(view) = self.view.as_ref() {
            assert!(!self.backward || view.expansion_width == 0);
            assert!(self.backward || view.reduction_width == 0);
            assert!(view.reduction_width == 0 || !self.pool);
            if self.backward {
                let assignment = format!(
                    "dst[source_index(row, channel, offset)] = {};",
                    match self.storage {
                        DType::BF16 => "round_bf16(value)",
                        DType::F16 => "f16(value)",
                        _ => "value",
                    }
                );
                assert_eq!(code.matches(&assignment).count(), 1);
                code = code.replace(
                    &assignment,
                    "store_image_output(source_index(row, channel, offset), value);",
                );
                if view.reduction_width != 0 {
                    code = code.replace("dn = dy[at] *", "dn = load_image_gradient(at) *");
                    let width = view.reduction_width;
                    writeln!(code, "fn load_image_gradient(index: u32) -> f32 {{\n    let base = (index / {width}u * 2u) * {}u + index % {width}u * 2u;\n    return ((dy[base] + dy[base + 1u]) + dy[base + {}u]) + dy[base + {}u];\n}}", width * 2, width * 2, width * 2 + 1).unwrap();
                }
            } else {
                let assignment = "dst[at] = normalized / (1.0 + exp(-normalized));";
                assert_eq!(code.matches(assignment).count(), 1);
                code = code.replace(
                    assignment,
                    "store_image_output(at, normalized / (1.0 + exp(-normalized)));",
                );
            }
            let conversion = match output_storage {
                DType::BF16 => "round_bf16(value)",
                DType::F16 => "f16(value)",
                _ => "value",
            };
            code.push_str("fn store_image_output(index: u32, value: f32) {\n");
            if !view.packed && view.expansion_width == 0 {
                writeln!(code, "    dst[index] = {conversion};\n}}").unwrap();
            } else {
                let shape = if self.backward {
                    "params.in_h * params.in_w"
                } else {
                    "params.out_h * params.out_w"
                };
                writeln!(code, "    let pixels = {shape};\n    let channel = index / pixels % CHANNELS;\n    let batch = index / (pixels * CHANNELS);\n    let pixel = index % pixels;").unwrap();
                if view.expansion_width != 0 {
                    let w = view.expansion_width;
                    writeln!(code, "    let output_pixels = pixels * 4u;\n    for (var offset = 0u; offset < 4u; offset += 1u) {{\n        let position = (pixel / {w}u * 2u + offset / 2u) * {}u + pixel % {w}u * 2u + offset % 2u;", w * 2).unwrap();
                } else {
                    code.push_str(
                        "    let output_pixels = pixels;\n    {\n        let position = pixel;\n",
                    );
                }
                if view.packed {
                    let padded = self.channels.div_ceil(32) * 32;
                    writeln!(code, "        let base = (batch * output_pixels + position) * {padded}u;\n        dst[base + channel] = {conversion};\n        if channel == 0u {{\n            for (var padding = CHANNELS; padding < {padded}u; padding += 1u) {{ dst[base + padding] = {output_ty}(0.0); }}\n        }}").unwrap();
                } else {
                    writeln!(code, "        dst[(batch * CHANNELS + channel) * output_pixels + position] = {conversion};").unwrap();
                }
                code.push_str("    }\n}\n");
            }
        }
        super::ShaderModule::new(&code)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_view_storage_and_layout_modules() {
        let mut count = 0;
        for channels in [3, 17, 32, 48, 64, 256] {
            for storage in [DType::F32, DType::F16, DType::BF16] {
                for output in [DType::F32, DType::F16, DType::BF16] {
                    for backward in [false, true] {
                        for pool in [false, true] {
                            for packed in [false, true] {
                                let kernel = ImageKernel {
                                    channels,
                                    storage,
                                    backward,
                                    pool,
                                    bias: true,
                                    view: Some(ImageView {
                                        storage: output,
                                        packed,
                                        expansion_width: if backward { 0 } else { 5 },
                                        reduction_width: if backward && !pool { 5 } else { 0 },
                                    }),
                                };
                                let mut shader = kernel.generate();
                                let mut binding = 0;
                                for (_, var) in shader.module.global_variables.iter_mut() {
                                    if matches!(
                                        var.space,
                                        naga::AddressSpace::Storage { .. }
                                            | naga::AddressSpace::Uniform
                                    ) {
                                        var.binding =
                                            Some(naga::ResourceBinding { group: 0, binding });
                                        binding += 1;
                                    }
                                }
                                let info = naga::valid::Validator::new(
                                    naga::valid::ValidationFlags::all(),
                                    naga::valid::Capabilities::all(),
                                )
                                .validate(&shader.module)
                                .unwrap();
                                naga::back::spv::write_vec(
                                    &shader.module,
                                    &info,
                                    &naga::back::spv::Options::default(),
                                    None,
                                )
                                .unwrap();
                                count += 1;
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(count, 432);
    }

    #[test]
    fn image_default_view_preserves_key() {
        #[derive(Hash)]
        struct Original {
            channels: u32,
            pool: bool,
            bias: bool,
            backward: bool,
            storage: DType,
        }
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        Original {
            channels: 48,
            pool: true,
            bias: true,
            backward: false,
            storage: DType::BF16,
        }
        .hash(&mut hash);
        let kernel = ImageKernel {
            channels: 48,
            pool: true,
            bias: true,
            backward: false,
            storage: DType::BF16,
            view: None,
        };
        assert_eq!(kernel.hash_key(), hash.finish());
        let serialized = serde_json::to_value(&kernel).unwrap();
        assert!(serialized.get("view").is_none());
    }

    #[test]
    fn image_region_shader_types() {
        for channels in [3, 17, 32, 48, 64, 256] {
            for storage in [DType::F32, DType::F16, DType::BF16] {
                for pool in [false, true] {
                    for bias in [false, true] {
                        for backward in [false, true] {
                            let mut shader = ImageKernel {
                                channels,
                                storage,
                                pool,
                                bias,
                                backward,
                                view: None,
                            }
                            .generate();
                            let mut binding = 0;
                            for (_, var) in shader.module.global_variables.iter_mut() {
                                if matches!(
                                    var.space,
                                    naga::AddressSpace::Storage { .. }
                                        | naga::AddressSpace::Uniform
                                ) {
                                    var.binding = Some(naga::ResourceBinding { group: 0, binding });
                                    binding += 1;
                                }
                            }
                            naga::valid::Validator::new(
                                naga::valid::ValidationFlags::all(),
                                naga::valid::Capabilities::all(),
                            )
                            .validate(&shader.module)
                            .unwrap();
                        }
                    }
                }
            }
        }
    }
}
