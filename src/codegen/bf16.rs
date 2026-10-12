use super::{ShaderGroup, ShaderModule, preprocess};

/// Dense and implicit-GEMM convolutions share the same checked cooperative loop.
pub(super) fn generate(group: ShaderGroup) -> ShaderModule {
    let (declaration, dimensions, range, a, b, index, destination, add, output_bf16) = match group {
        ShaderGroup::MatMulBF16 {
            transpose_a,
            transpose_b,
            addend,
            output_bf16,
        } => {
            assert!(!(transpose_a && transpose_b));
            let mut declaration = String::from(
                "struct Params { m: u32, n: u32, k: u32, splits: u32 }\n\
                 var<storage> matrix_a: array<bf16>;\n\
                 var<storage> matrix_b: array<bf16>;\n\
                 var<storage, read_write> matrix_c: array<$OUT_TYPE>;\n\
                 var<uniform> params: Params;\n",
            );
            if addend {
                declaration.push_str("var<storage> src: array<f32>;\n");
            }
            (
                declaration,
                "let m = params.m; let n = params.n; let k = params.k;",
                "let chunk = (k / params.splits / 32u) * 32u;\n\
                 let first = group.z * chunk;\n\
                 let end = select(first + chunk, k, group.z + 1u == params.splits);",
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
                output_bf16,
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
                output_bf16,
            )
        }
        _ => unreachable!(),
    };
    ShaderModule::new(&preprocess(
        include_str!("../shaders/matmul_bf16.wgsl"),
        &[
            ("$DECL", &declaration),
            ("$DIMS", dimensions),
            ("$RANGE", range),
            ("$A_LOAD", a),
            ("$B_LOAD", b),
            ("$INDEX", index),
            ("$DEST", destination),
            ("$ROUND", include_str!("../shaders/bf16_round.wgsl")),
            ("$OUT_TYPE", if output_bf16 { "bf16" } else { "f32" }),
            ("$STORE", if output_bf16 { "round_bf16" } else { "f32" }),
            ("$ADD", add),
        ],
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convolution_and_dense_modules_validate_and_emit_spirv() {
        let capabilities = naga::valid::Capabilities::SHADER_BFLOAT16
            | naga::valid::Capabilities::COOPERATIVE_MATRIX
            | naga::valid::Capabilities::COOPERATIVE_MATRIX_BFLOAT16
            | naga::valid::Capabilities::SUBGROUP;
        for output_bf16 in [false, true] {
            for group in [
                ShaderGroup::Conv2dBF16 {
                    kernel_h: 2,
                    kernel_w: 4,
                    stride: 2,
                    output_bf16,
                },
                ShaderGroup::MatMulBF16 {
                    transpose_a: false,
                    transpose_b: true,
                    addend: true,
                    output_bf16,
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
