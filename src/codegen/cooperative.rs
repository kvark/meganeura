use super::{MatMulCoopVariant, MatrixTypes, ShaderGroup, ShaderModule, coop_shape, preprocess};
use crate::{DType, kernels::conv::Params};

// Keep partition boundaries identical across native tile shapes.
// A 32-element block contains whole 8-wide or 16-wide K tiles.
pub(super) const SPLIT_RANGE: &str = "let blocks = (k + 31u) / 32u;\n\
    let base = blocks / params.splits;\n\
    let extra = blocks % params.splits;\n\
    let first = (wgid.z * base + min(wgid.z, extra)) * 32u;\n\
    let count = base + select(0u, 1u, wgid.z < extra);\n\
    let end = min(first + count * 32u, k);";

/// Dense and implicit-GEMM convolutions share the same checked cooperative
/// loop. Operands are staged in the multiplication type; the accumulator is
/// F32, and so are partials until their reduction.
pub(super) fn generate(group: ShaderGroup, types: MatrixTypes, shape: [u32; 3]) -> ShaderModule {
    generate_with_params(group, types, shape, None)
}

pub(super) fn generate_with_params(
    group: ShaderGroup,
    types: MatrixTypes,
    shape: [u32; 3],
    params: Option<&Params>,
) -> ShaderModule {
    types.validate();
    assert!(
        super::checked_cooperative_shape(&[shape]).is_some(),
        "unsupported cooperative tile {shape:?}"
    );
    let [m, n, k] = shape;
    let split_range = SPLIT_RANGE.replace("wgid.", "group.");
    let vector_conv = params.is_some_and(Params::vectorized_forward);
    let (declaration, dimensions, range, a, b, index, destination, add) = match group {
        ShaderGroup::MatMul
        | ShaderGroup::MatMulAT
        | ShaderGroup::MatMulBT
        | ShaderGroup::MatMulAdd
        | ShaderGroup::MatMulATAdd
        | ShaderGroup::MatMulBTAdd => {
            let (addend, variant) = coop_shape(group).unwrap();
            let mut declaration = String::from(
                "struct Params { m: u32, n: u32, k: u32, splits: u32 }\n\
                 var<storage> matrix_a: array<$A_TYPE>;\n\
                 var<storage> matrix_b: array<$B_TYPE>;\n\
                 var<storage, read_write> matrix_c: array<$OUT_TYPE>;\n\
                 var<uniform> params: Params;\n",
            );
            if addend {
                declaration.push_str("var<storage> src: array<f32>;\n");
            }
            (
                declaration,
                "let m = params.m; let n = params.n; let k = params.k;",
                split_range.as_str(),
                if variant == MatMulCoopVariant::AT {
                    "matrix_a[ak * m + ar]"
                } else {
                    "matrix_a[ar * k + ak]"
                },
                if variant == MatMulCoopVariant::BT {
                    "matrix_b[bc * k + bk]"
                } else {
                    "matrix_b[bk * n + bc]"
                },
                "(group.z * m + r) * n + c",
                "matrix_c",
                if addend { " + src[index]" } else { "" },
            )
        }
        ShaderGroup::Conv2dBF16 {
            kernel_h,
            kernel_w,
            stride,
            ..
        }
        | ShaderGroup::Conv2dChecked {
            kernel_h,
            kernel_w,
            stride,
        } => {
            assert!(kernel_h > 0 && kernel_w > 0 && stride > 0);
            if let ShaderGroup::Conv2dBF16 { output_bf16, .. } = group {
                assert_eq!(
                    types,
                    MatrixTypes {
                        output: if output_bf16 { DType::BF16 } else { DType::F32 },
                        ..MatrixTypes::BF16
                    }
                );
            }
            let declaration = format!(
                "{}\n{}\nconst KH: u32 = {kernel_h}u;\nconst KW: u32 = {kernel_w}u;\n\
                 const STRIDE: u32 = {stride}u;\n{}",
                include_str!("../shaders/conv2d_params.wgsl"),
                if params.is_some() {
                    crate::divisor::NATIVE_SHADER
                } else {
                    crate::divisor::SHADER
                },
                if vector_conv {
                    "var<storage> src: array<vec4<$B_TYPE>>;\nvar<storage> weight: array<vec4<$A_TYPE>>;\nvar<storage, read_write> dst: array<$OUT_TYPE>;"
                } else {
                    include_str!("../shaders/conv2d_checked.wgsl")
                }
            );
            let params_decl = match params {
                Some(p) => format!(
                    "const params = Params({}u, {}u, {}u, {}u, {}u, {}u, {}u, {}u, {}u, {}u, {}u, {}u, {}u, {}u, {}u, {}u);",
                    p.batch,
                    p.in_channels,
                    p.in_h,
                    p.in_w,
                    p.out_channels,
                    p.kernel_h,
                    p.kernel_w,
                    p.stride,
                    p.padding_h,
                    p.out_h,
                    p.out_w,
                    p.padding_w,
                    p.kernel_w_multiplier,
                    p.kernel_hw_multiplier,
                    p.column_width_multiplier,
                    p.output_spatial_multiplier,
                ),
                None => "var<uniform> params: Params;".to_owned(),
            };
            (
                format!("{declaration}\n{params_decl}"),
                "let m = params.out_channels; let n = params.out_h * params.out_w; let k = params.in_channels * KH * KW;",
                "let first = 0u; let end = k;",
                "weight[ar * k + ak]",
                "load_pixel(group.z, bk, bc)",
                "(group.z * m + r) * n + c",
                "dst",
                "",
            )
        }
        _ => unreachable!(),
    };
    let mut extensions = String::from("enable wgpu_cooperative_matrix;\n");
    if types.uses(DType::F16) {
        extensions.push_str("enable f16;\n");
    }
    if types.uses(DType::BF16) {
        extensions.push_str("enable wgpu_bfloat16;\n");
    }
    // F32 storage reaches a BF16 multiplication through nearest-even
    // rounding; every other operand conversion is exact or a plain cast.
    let cast = |storage| {
        if types.multiply == DType::BF16 && storage == DType::F32 {
            "round_bf16"
        } else {
            MatrixTypes::name(types.multiply)
        }
    };
    let (a_cast, b_cast) = (cast(types.a), cast(types.b));
    let convert = |value: &str, storage| {
        let cast = cast(storage);
        if types.multiply == DType::BF16 && storage == DType::F32 {
            format!(
                "vec4<bf16>({cast}({value}.x), {cast}({value}.y), {cast}({value}.z), {cast}({value}.w))"
            )
        } else {
            format!("vec4<{cast}>({value})")
        }
    };
    let stage = if vector_conv {
        include_str!("../shaders/conv2d_checked_vector.wgsl")
    } else {
        include_str!("../shaders/matmul_checked_stage.wgsl")
    };
    let rounds = types.output == DType::BF16
        || (types.multiply == DType::BF16 && (types.a == DType::F32 || types.b == DType::F32));
    let multiply = MatrixTypes::name(types.multiply);
    ShaderModule::new(&preprocess(
        include_str!("../shaders/matmul_checked.wgsl"),
        &[
            ("$STAGE", stage),
            ("$THREADS", "128"),
            ("$A_VECTOR", &convert("v", types.a)),
            ("$B_VECTOR", &convert("v", types.b)),
            ("$EXTENSIONS", &extensions),
            ("$DECL", &declaration),
            ("$DIMS", dimensions),
            ("$RANGE", range),
            ("$A_LOAD", a),
            ("$B_LOAD", b),
            ("$INDEX", index),
            ("$DEST", destination),
            (
                "$ROUND",
                if rounds {
                    include_str!("../shaders/bf16_round.wgsl")
                } else {
                    ""
                },
            ),
            ("$A_TYPE", MatrixTypes::name(types.a)),
            ("$B_TYPE", MatrixTypes::name(types.b)),
            ("$A_CAST", a_cast),
            ("$B_CAST", b_cast),
            ("$OUT_TYPE", MatrixTypes::name(types.output)),
            (
                "$STORE",
                if types.output == DType::BF16 {
                    "round_bf16"
                } else {
                    MatrixTypes::name(types.output)
                },
            ),
            ("$ADD", add),
            // WGSL names cooperative matrices columns × rows, whereas device
            // capabilities and the product shape are M × N × K.
            ("$A_TILE", &format!("coop_mat{k}x{m}<{multiply}, A>")),
            ("$B_TILE", &format!("coop_mat{n}x{k}<{multiply}, B>")),
            ("$C_TILE", &format!("coop_mat{n}x{m}<f32, C>")),
            ("$MULTIPLY_TYPE", multiply),
            ("$TILE_M", &m.to_string()),
            ("$TILE_N", &n.to_string()),
            ("$TILE_K", &k.to_string()),
        ],
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_shape_selection_requires_an_implemented_product() {
        use crate::codegen::checked_cooperative_shape as select;
        assert_eq!(select(&[]), None);
        assert_eq!(select(&[[16, 8, 16], [8, 16, 32]]), None);
        assert_eq!(select(&[[8, 8, 8]]), Some([8, 8, 8]));
        assert_eq!(select(&[[8, 16, 16]]), Some([8, 16, 16]));
        assert_eq!(select(&[[8, 8, 8], [8, 16, 16]]), Some([8, 16, 16]));
        assert_eq!(select(&[[8, 16, 16], [16, 16, 16]]), Some([16, 16, 16]));
    }

    #[test]
    fn typed_operand_combinations_are_validated() {
        for types in [
            MatrixTypes::BF16,
            MatrixTypes::reduced(DType::F16),
            MatrixTypes::reduced(DType::F32),
            MatrixTypes {
                a: DType::F32,
                b: DType::F16,
                multiply: DType::F16,
                output: DType::BF16,
            },
            MatrixTypes {
                a: DType::BF16,
                b: DType::F16,
                multiply: DType::F32,
                output: DType::F32,
            },
        ] {
            assert!(types.is_valid(), "{types:?}");
        }
        for types in [
            // Neither half-precision format converts exactly to the other.
            MatrixTypes {
                b: DType::F16,
                ..MatrixTypes::BF16
            },
            MatrixTypes {
                output: DType::U32,
                ..MatrixTypes::BF16
            },
        ] {
            assert!(!types.is_valid(), "{types:?}");
        }
    }

    #[test]
    fn convolution_and_dense_modules_validate_and_emit_spirv() {
        let capabilities = naga::valid::Capabilities::SHADER_BFLOAT16
            | naga::valid::Capabilities::SHADER_FLOAT16
            | naga::valid::Capabilities::COOPERATIVE_MATRIX
            | naga::valid::Capabilities::SUBGROUP;
        let mut cases = Vec::new();
        for shape in [[16, 16, 16], [8, 16, 16], [8, 8, 8]] {
            for output_bf16 in [false, true] {
                let output = if output_bf16 { DType::BF16 } else { DType::F32 };
                cases.push((
                    ShaderGroup::Conv2dBF16 {
                        kernel_h: 2,
                        kernel_w: 4,
                        stride: 2,
                        output_bf16,
                    },
                    MatrixTypes {
                        output,
                        ..MatrixTypes::BF16
                    },
                    shape,
                ));
                for storage in [DType::BF16, DType::F16, DType::F32] {
                    for multiply in [DType::BF16, DType::F16, DType::F32] {
                        let types = MatrixTypes {
                            a: storage,
                            b: DType::F32,
                            multiply,
                            output,
                        };
                        if types.is_valid() {
                            for group in [
                                ShaderGroup::Conv2dChecked {
                                    kernel_h: 2,
                                    kernel_w: 4,
                                    stride: 2,
                                },
                                ShaderGroup::MatMul,
                                ShaderGroup::MatMulAT,
                                ShaderGroup::MatMulBT,
                                ShaderGroup::MatMulAdd,
                                ShaderGroup::MatMulATAdd,
                                ShaderGroup::MatMulBTAdd,
                            ] {
                                cases.push((group, types, shape));
                            }
                        }
                    }
                }
            }
        }
        for (group, types, shape) in cases {
            for layout in [
                super::super::TypedMatmulLayout::default(),
                super::super::TypedMatmulLayout {
                    vectorized: true,
                    direct_output: false,
                },
                super::super::TypedMatmulLayout::for_product(
                    types,
                    coop_shape(group).is_some_and(|(addend, _)| addend),
                    [64, 64, 64],
                ),
            ] {
                let mut module = super::super::generate_module_coop_typed_with_layout(
                    group, types, shape, layout,
                );
                let mut binding = 0;
                for (_, global) in module.module.global_variables.iter_mut() {
                    if matches!(
                        global.space,
                        naga::AddressSpace::Uniform | naga::AddressSpace::Storage { .. }
                    ) {
                        global.binding = Some(naga::ResourceBinding { group: 0, binding });
                        binding += 1;
                    }
                }
                let info =
                    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), capabilities)
                        .validate(&module.module)
                        .unwrap_or_else(|error| panic!("{group:?} {types:?} {shape:?}: {error:?}"));
                let options = naga::back::spv::Options {
                    lang_version: (1, 3),
                    flags: naga::back::spv::WriterFlags::empty(),
                    ..Default::default()
                };
                let words =
                    naga::back::spv::write_vec(&module.module, &info, &options, None).unwrap();
                assert!(!words.is_empty());
            }
        }
    }

    #[test]
    fn specialized_convolutions_validate() {
        let capabilities = naga::valid::Capabilities::SHADER_BFLOAT16
            | naga::valid::Capabilities::SHADER_FLOAT16
            | naga::valid::Capabilities::COOPERATIVE_MATRIX
            | naga::valid::Capabilities::SUBGROUP;
        for shape in [[16, 16, 16], [8, 16, 16], [8, 8, 8]] {
            for multiply in [DType::F16, DType::BF16] {
                for (h, w, ci, kh, kw, stride, pad) in [
                    (14, 14, 64, 1, 1, 1, 0),
                    (7, 7, 64, 1, 1, 1, 0),
                    (11, 13, 3, 3, 3, 2, 1),
                ] {
                    let params = Params {
                        batch: 2,
                        in_channels: ci,
                        in_h: h,
                        in_w: w,
                        out_channels: 65,
                        kernel_h: kh,
                        kernel_w: kw,
                        stride,
                        padding_h: pad,
                        out_h: (h + 2 * pad - kh) / stride + 1,
                        out_w: (w + 2 * pad - kw) / stride + 1,
                        padding_w: pad,
                        kernel_w_multiplier: 0,
                        kernel_hw_multiplier: 0,
                        column_width_multiplier: 0,
                        output_spatial_multiplier: 0,
                    };
                    let mut module = generate_with_params(
                        ShaderGroup::Conv2dChecked {
                            kernel_h: kh,
                            kernel_w: kw,
                            stride,
                        },
                        MatrixTypes {
                            multiply,
                            ..MatrixTypes::F32_AS_F16
                        },
                        shape,
                        Some(&params),
                    );
                    let mut binding = 0;
                    for (_, global) in module.module.global_variables.iter_mut() {
                        if matches!(
                            global.space,
                            naga::AddressSpace::Uniform | naga::AddressSpace::Storage { .. }
                        ) {
                            global.binding = Some(naga::ResourceBinding { group: 0, binding });
                            binding += 1;
                        }
                    }
                    let info = naga::valid::Validator::new(
                        naga::valid::ValidationFlags::all(),
                        capabilities,
                    )
                    .validate(&module.module)
                    .unwrap();
                    naga::back::spv::write_vec(
                        &module.module,
                        &info,
                        &naga::back::spv::Options::default(),
                        None,
                    )
                    .unwrap();
                }
            }
        }
    }

    #[test]
    fn typed_dense_uses_shared_staging_and_register_tiles() {
        use super::super::{TypedMatmulLayout, generate_module_coop_typed_with_layout};
        for storage in [DType::F16, DType::BF16] {
            let types = MatrixTypes::reduced(storage);
            let layout = TypedMatmulLayout::for_product(types, false, [64, 64, 64]);
            let module = generate_module_coop_typed_with_layout(
                ShaderGroup::MatMul,
                types,
                [16, 16, 16],
                layout,
            );
            assert!(module.source.contains(&format!(
                "matrix_a: array<vec4<{}>>",
                MatrixTypes::name(storage)
            )));
            assert!(
                module
                    .source
                    .contains("acc11 = coopMultiplyAdd(a1, b1, acc11)")
            );
            assert!(!module.source.contains("var<workgroup> shared_c"));
            assert!(module.source.contains("wgid.z * m * n"));
        }
    }
}
