use super::{MatrixTypes, ShaderGroup, ShaderModule, preprocess};
use crate::DType;
#[path = "cooperative/convolution.rs"]
mod convolution;
#[path = "cooperative/convolution_words.rs"]
mod convolution_words;
#[path = "cooperative/matrix.rs"]
mod matrix;

/// Sum short matrix-accumulator blocks with F32 compensation. This bounds the
/// dependency chain without increasing global partial storage or dispatches.
pub(super) fn compensated_partials(
    names: &[String],
    first: &str,
    next: &str,
    end: &str,
) -> (String, String, String) {
    let mut init = String::new();
    let mut step = format!("if {next} >= {end} || ({next} - {first}) % 256u == 0u {{\n");
    let mut finish = String::new();
    for name in names {
        init += &format!(
            "var total_{name} = coop_mat16x16<f32,C>();\nvar error_{name} = coop_mat16x16<f32,C>();\n"
        );
        step += &format!(
            "let corrected_{name} = {name} - error_{name};\nlet sum_{name} = total_{name} + corrected_{name};\nerror_{name} = (sum_{name} - total_{name}) - corrected_{name};\ntotal_{name} = sum_{name};\n{name} = coop_mat16x16<f32,C>();\n"
        );
        finish += &format!("{name} = total_{name};\n");
    }
    step += "}\n";
    (init, step, finish)
}

/// Dense and implicit-GEMM convolutions share the same checked cooperative loop.
pub(super) fn generate(group: ShaderGroup) -> ShaderModule {
    if matches!(
        group,
        ShaderGroup::Conv2dBF16 { .. } | ShaderGroup::Conv2dGradientTyped { .. }
    ) && std::env::var("MEGANEURA_CONVOLUTION_WORDS").as_deref() == Ok("1")
    {
        return generate_impl(group, true, false, true, None);
    }
    generate_dense(group, true)
}

pub(super) fn generate_dense(group: ShaderGroup, partitioned: bool) -> ShaderModule {
    generate_impl(group, partitioned, false, false, None)
}

pub(crate) fn generate_vector(group: ShaderGroup) -> ShaderModule {
    generate_impl(group, true, true, false, None)
}

pub(crate) fn generate_reuse(
    group: ShaderGroup,
    shape: super::ConvolutionReuseShape,
) -> ShaderModule {
    generate_impl(group, true, false, false, Some(shape))
}

fn generate_impl(
    group: ShaderGroup,
    partitioned: bool,
    vectorized: bool,
    convolution_words: bool,
    reuse: Option<super::ConvolutionReuseShape>,
) -> ShaderModule {
    let (declaration, dimensions, range, a, b, index, destination, add, types) = match group {
        ShaderGroup::BlockMatMulTyped {
            transpose_a,
            transpose_b,
            types,
        } => {
            assert!(!(transpose_a && transpose_b));
            (
                format!(
                    "{}\nvar<storage> matrix_a: array<$A_STORAGE>;\nvar<storage> matrix_b: array<$B_STORAGE>;\nvar<storage, read_write> matrix_c: array<$OUT_TYPE>;\nvar<uniform> params: Params;\n",
                    include_str!("../shaders/matmul_grouped_params.wgsl")
                ),
                "let m = params.m; let n = params.n; let k = params.k;",
                "let block_id = group.z / params.splits; let part_id = group.z % params.splits;\nlet span = grouped_span(k, params.splits, part_id); let first = span.x; let end = span.y;",
                if transpose_a {
                    "matrix_a[(ak * params.groups + block_id) * m + ar]"
                } else {
                    "matrix_a[(ar * params.groups + block_id) * k + ak]"
                },
                if transpose_a {
                    "matrix_b[(bk * params.groups + block_id) * n + bc]"
                } else if transpose_b {
                    "matrix_b[(block_id * n + bc) * k + bk]"
                } else {
                    "matrix_b[(block_id * k + bk) * n + bc]"
                },
                if transpose_a {
                    "((part_id * params.groups + block_id) * m + r) * n + c"
                } else {
                    "((part_id * m + r) * params.groups + block_id) * n + c"
                },
                "matrix_c",
                "",
                types,
            )
        }
        ShaderGroup::MatMulTyped {
            transpose_a,
            transpose_b,
            addend,
            types,
        } => {
            assert!(!(transpose_a && transpose_b));
            let mut declaration = String::from(
                "struct Params { m: u32, n: u32, k: u32, splits: u32 }\n\
                 var<storage> matrix_a: array<$A_STORAGE>;\n\
                 var<storage> matrix_b: array<$B_STORAGE>;\n\
                 var<storage, read_write> matrix_c: array<$OUT_TYPE>;\n\
                 var<uniform> params: Params;\n",
            );
            if addend {
                declaration.push_str("var<storage> src: array<f32>;\n");
            }
            (
                declaration,
                "let m = params.m; let n = params.n; let k = params.k;",
                if partitioned {
                    "let chunk = (k / params.splits / 32u) * 32u;\n\
                 let first = group.z * chunk;\n\
                 let end = select(first + chunk, k, group.z + 1u == params.splits);"
                } else {
                    "let first = 0u; let end = k;"
                },
                if transpose_a {
                    "matrix_a[ak * m + ar]"
                } else {
                    "matrix_a[ar * k + ak]"
                },
                if transpose_b {
                    "matrix_b[bc * k + bk]"
                } else {
                    "matrix_b[bk * n + bc]"
                },
                "(group.z * m + r) * n + c",
                "matrix_c",
                if addend { " + src[index]" } else { "" },
                types,
            )
        }
        ShaderGroup::Conv2dBF16 {
            kernel_h,
            kernel_w,
            stride,
            output_bf16,
        } => {
            assert!(kernel_h > 0 && kernel_w > 0 && stride > 0);
            let declaration = format!(
                "{}\n{}\nconst KH: u32 = {kernel_h}u;\nconst KW: u32 = {kernel_w}u;\n\
                 const STRIDE: u32 = {stride}u;\n{}",
                include_str!("../shaders/conv2d_params.wgsl"),
                crate::divisor::SHADER,
                include_str!("../shaders/conv2d_bf16.wgsl")
            );
            (
                declaration,
                "let m = params.out_channels; let n = params.out_h * params.out_w; let k = params.in_channels * KH * KW;",
                "let first = 0u; let end = k;",
                "weight[ar * k + ak]",
                "load_pixel(group.z, bk, bc)",
                "(group.z * m + r) * n + c",
                "dst",
                "",
                MatrixTypes {
                    output: if output_bf16 { DType::BF16 } else { DType::F32 },
                    ..MatrixTypes::BF16
                },
            )
        }
        ShaderGroup::Conv2dGradientTyped {
            weight_gradient,
            kernel_h,
            kernel_w,
            stride,
            splits,
            types,
        } => {
            assert!(kernel_h > 0 && kernel_w > 0 && stride > 0 && splits > 0);
            assert_eq!(types.output, DType::F32);
            let declaration = format!(
                "{}\n{}\nconst KH: u32 = {kernel_h}u;\nconst KW: u32 = {kernel_w}u;\nconst STRIDE: u32 = {stride}u;\nconst SPLITS: u32 = {splits}u;\n{}",
                include_str!("../shaders/conv2d_params.wgsl"),
                crate::divisor::SHADER,
                if weight_gradient {
                    include_str!("../shaders/conv2d_typed_weight.wgsl")
                } else {
                    include_str!("../shaders/conv2d_typed_input.wgsl")
                }
            );
            if weight_gradient {
                (
                    declaration,
                    "let m = params.out_channels; let n = params.in_channels * KH * KW;",
                    "let total_k = params.batch * params.out_h * params.out_w;\nlet tiles = (total_k + 15u) / 16u;\nlet count = tiles / SPLITS;\nlet extra = tiles % SPLITS;\nlet first_tile = group.z * count + min(group.z, extra);\nlet first = first_tile * 16u;\nlet end = min(total_k, (first_tile + count + select(0u, 1u, group.z < extra)) * 16u);\nlet k = end;",
                    "load_gradient(ar, ak)",
                    "load_input(bk, bc)",
                    "(group.z * m + r) * n + c",
                    "dst",
                    "",
                    types,
                )
            } else {
                assert_eq!(splits, 1);
                (
                    declaration,
                    "let m = params.in_channels; let n = params.in_h * params.in_w; let k = params.out_channels * KH * KW;",
                    "let first = 0u; let end = k;",
                    "load_weight(ar, ak)",
                    "load_gradient(group.z, bk, bc)",
                    "(group.z * m + r) * n + c",
                    "dst",
                    "",
                    types,
                )
            }
        }
        _ => unreachable!(),
    };
    if let Some(shape) = reuse {
        types.validate();
        return super::convolution_reuse::generate(
            group,
            types,
            shape,
            &declaration,
            dimensions,
            range,
        );
    }
    let words = convolution_words.then(|| convolution_words::generate(group, types));
    let staged = match group {
        ShaderGroup::Conv2dBF16 { .. } => Some(convolution::generate(false)),
        ShaderGroup::Conv2dGradientTyped {
            weight_gradient: true,
            ..
        } => Some(convolution::generate(true)),
        _ => None,
    };
    let declaration = if staged.is_some() || words.is_some() {
        format!(
            "{}\n{}",
            declaration,
            include_str!("../shaders/conv_tile_address.wgsl")
        )
    } else {
        declaration
    };
    let a = if matches!(
        group,
        ShaderGroup::Conv2dGradientTyped {
            weight_gradient: true,
            ..
        }
    ) {
        "grad_out[gradient_offset + ar * spatial]"
    } else {
        a
    };
    types.validate();
    let vector = vectorized.then(|| matrix::generate(group));
    let storage = |dtype| {
        let name = MatrixTypes::name(dtype);
        if vectorized {
            format!("vec4<{name}>")
        } else {
            name.to_owned()
        }
    };
    let mut extensions = String::from("enable wgpu_cooperative_matrix;\n");
    let fields = [types.a, types.b, types.multiply, types.output];
    if fields.contains(&DType::F16) {
        extensions.push_str("enable f16;\n");
    }
    if fields.contains(&DType::BF16) {
        extensions.push_str("enable wgpu_bfloat16;\n");
    }
    let (partial_init, partial_step, partial_finish) = if matches!(
        group,
        ShaderGroup::Conv2dGradientTyped {
            weight_gradient: true,
            ..
        }
    ) {
        compensated_partials(&["accumulator".to_owned()], "first", "(start + 32u)", "end")
    } else {
        Default::default()
    };
    ShaderModule::new(&preprocess(
        include_str!("../shaders/matmul_checked_qualified.wgsl"),
        &[
            ("$EXTENSIONS", &extensions),
            ("$DECL", &declaration),
            ("$DIMS", dimensions),
            ("$RANGE", range),
            ("$A_STORAGE", &storage(types.a)),
            ("$B_STORAGE", &storage(types.b)),
            ("$STAGE_DECL", words.as_ref().map_or_else(|| vector.as_ref().map_or(
                "var<workgroup> stage_a: array<$MULTIPLY_TYPE, 1024>;\nvar<workgroup> stage_b: array<$MULTIPLY_TYPE, 1024>;",
                |v| v.declaration.as_str()), |w| w.declaration.as_str())),
            ("$MATRIX_PRODUCT", words.as_ref().map_or_else(|| vector.as_ref().map_or(
                include_str!("../shaders/matmul_checked_multiply.wgsl"),
                |v| v.multiply.as_str()), |w| w.multiply.as_str())),
            (
                "$STAGE_INIT",
                words.as_ref().map_or_else(|| staged.as_ref().map_or("", |s| s.initialize.as_str()), |w| w.initialize.as_str()),
            ),
            (
                "$STAGE_LOAD",
                words.as_ref().map_or_else(|| vector.as_ref().map_or_else(|| staged.as_ref().map_or(
                    include_str!("../shaders/matmul_checked_load.wgsl"), |s| s.load.as_str()),
                    |v| v.load.as_str()), |w| w.load.as_str()),
            ),
            (
                "$STAGE_ADVANCE",
                words.as_ref().map_or_else(|| staged.as_ref().map_or("", |s| s.advance.as_str()), |w| w.advance.as_str()),
            ),
            ("$A_LOAD", a),
            ("$B_LOAD", b),
            ("$INDEX", index),
            ("$PARTIAL_INIT", &partial_init),
            ("$PARTIAL_STEP", &partial_step),
            ("$PARTIAL_FINISH", &partial_finish),
            ("$DEST", destination),
            (
                "$ROUND",
                if types.output == DType::BF16 || types.multiply == DType::BF16 {
                    include_str!("../shaders/bf16_round.wgsl")
                } else {
                    ""
                },
            ),
            ("$A_TYPE", MatrixTypes::name(types.a)),
            ("$B_TYPE", MatrixTypes::name(types.b)),
            (
                "$A_CAST",
                if types.multiply == DType::BF16 && types.a == DType::F32 {
                    "round_bf16"
                } else {
                    MatrixTypes::name(types.multiply)
                },
            ),
            (
                "$B_CAST",
                if types.multiply == DType::BF16 && types.b == DType::F32 {
                    "round_bf16"
                } else {
                    MatrixTypes::name(types.multiply)
                },
            ),
            ("$MULTIPLY_TYPE", MatrixTypes::name(types.multiply)),
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
        ],
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_convolution_derivatives_validate() {
        for multiply in [DType::F32, DType::F16, DType::BF16] {
            for weight_gradient in [false, true] {
                for splits in [1, 3] {
                    if !weight_gradient && splits != 1 {
                        continue;
                    }
                    let mut module = generate(ShaderGroup::Conv2dGradientTyped {
                        weight_gradient,
                        kernel_h: 2,
                        kernel_w: 4,
                        stride: 2,
                        splits,
                        types: MatrixTypes {
                            a: multiply,
                            b: multiply,
                            multiply,
                            output: DType::F32,
                        },
                    });
                    let mut binding = 0;
                    for (_, var) in module.module.global_variables.iter_mut() {
                        if matches!(
                            var.space,
                            naga::AddressSpace::Uniform | naga::AddressSpace::Storage { .. }
                        ) {
                            var.binding = Some(naga::ResourceBinding { group: 0, binding });
                            binding += 1;
                        }
                    }
                    let info = naga::valid::Validator::new(
                        naga::valid::ValidationFlags::all(),
                        naga::valid::Capabilities::all(),
                    )
                    .validate(&module.module)
                    .unwrap();
                    let words = naga::back::spv::write_vec(
                        &module.module,
                        &info,
                        &naga::back::spv::Options {
                            lang_version: (1, 3),
                            flags: naga::back::spv::WriterFlags::empty(),
                            ..Default::default()
                        },
                        None,
                    )
                    .unwrap();
                    assert!(!words.is_empty());
                }
            }
        }
    }

    #[test]
    fn convolution_and_dense_modules_validate_and_emit_spirv() {
        let capabilities = naga::valid::Capabilities::SHADER_BFLOAT16
            | naga::valid::Capabilities::COOPERATIVE_MATRIX
            | naga::valid::Capabilities::SUBGROUP;
        for output_bf16 in [false, true] {
            for group in [
                ShaderGroup::Conv2dBF16 {
                    kernel_h: 2,
                    kernel_w: 4,
                    stride: 2,
                    output_bf16,
                },
                ShaderGroup::MatMulTyped {
                    transpose_a: false,
                    transpose_b: true,
                    addend: true,
                    types: MatrixTypes {
                        output: if output_bf16 { DType::BF16 } else { DType::F32 },
                        ..MatrixTypes::BF16
                    },
                },
            ] {
                let mut module = generate(group);
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
                        .unwrap();
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
}
