use crate::{
    DType,
    codegen::{MatrixTypes, ShaderGroup},
};

pub(super) struct Staging {
    pub declaration: String,
    pub load: String,
    pub multiply: String,
}

/// Shared tiles keep each operand's physical major order, so vector loads and
/// stores remain contiguous even for transposed products.
pub(super) fn generate(group: ShaderGroup) -> Staging {
    let (at, bt, grouped, types) = match group {
        ShaderGroup::MatMulTyped {
            transpose_a,
            transpose_b,
            types,
            ..
        } => (transpose_a, transpose_b, false, types),
        ShaderGroup::BlockMatMulTyped {
            transpose_a,
            transpose_b,
            types,
        } => (transpose_a, transpose_b, true, types),
        _ => panic!("vector checked staging requires a typed matrix"),
    };
    assert!(!(at && bt));
    types.validate();
    let multiply = MatrixTypes::name(types.multiply);
    let mut load = String::new();
    for operand in [true, false] {
        let name = if operand { "a" } else { "b" };
        let transpose = if operand { at } else { bt };
        let storage = if operand { types.a } else { types.b };
        let (r, c, condition) = match (operand, transpose) {
            (true, false) => ("row + vr", "start + vc", "r < m && c < k"),
            (true, true) => ("start + vr", "row + vc", "r < k && c < m"),
            (false, false) => ("start + vr", "col + vc", "r < k && c < n"),
            (false, true) => ("col + vr", "start + vc", "r < n && c < k"),
        };
        let index = match (operand, transpose, grouped) {
            (true, false, false) | (false, true, false) => "r * k + c",
            (true, true, false) => "r * m + c",
            (false, false, false) => "r * n + c",
            (true, false, true) => "(r * params.groups + block_id) * k + c",
            (true, true, true) => "(r * params.groups + block_id) * m + c",
            (false, true, true) => "(block_id * n + r) * k + c",
            (false, false, true) if at => "(r * params.groups + block_id) * n + c",
            (false, false, true) => "(block_id * k + r) * n + c",
        };
        let value = if storage == types.multiply {
            "stored".to_owned()
        } else if types.multiply == DType::BF16 {
            "vec4<bf16>(round_bf16(stored.x), round_bf16(stored.y), round_bf16(stored.z), round_bf16(stored.w))".to_owned()
        } else {
            format!("vec4<{multiply}>(stored)")
        };
        for part in 0..2 {
            load += &format!(
                "{{ let packet = lane + {}u;\nlet vr = packet / 8u; let vc = (packet % 8u) * 4u;\nlet r = {r}; let c = {c};\nvar value = vec4<{multiply}>(0.0);\nif {condition} {{ let stored = matrix_{name}[({index}) / 4u]; value = {value}; }}\nstage_{name}[packet] = value; }}\n",
                part * 128,
            );
        }
    }
    let mut products = String::new();
    for part in 0..2 {
        let k = part * 16;
        let ai = if at {
            format!("({k}u * 32u + ((tile % 4u) / 2u) * 16u) / 4u")
        } else {
            format!("(a_offset + {k}u) / 4u")
        };
        let bi = if bt {
            format!("(b_offset * 32u + {k}u) / 4u")
        } else {
            format!("({k}u * 32u + b_offset) / 4u")
        };
        let al = if at { "coopLoad" } else { "coopLoadT" };
        let bl = if bt { "coopLoad" } else { "coopLoadT" };
        products += &format!(
            "let a{part} = {al}<coop_mat16x16<{multiply}, A>>(&stage_a[{ai}], 8u);\nlet b{part} = {bl}<coop_mat16x16<{multiply}, B>>(&stage_b[{bi}], 8u);\naccumulator = coopMultiplyAdd(a{part}, b{part}, accumulator);\n"
        );
    }
    Staging {
        declaration: format!(
            "var<workgroup> stage_a: array<vec4<{multiply}>, 256>;\nvar<workgroup> stage_b: array<vec4<{multiply}>, 256>;"
        ),
        load,
        multiply: products,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_vector_shaders_validate() {
        let mut cases = 0;
        for multiply in [DType::F32, DType::F16, DType::BF16] {
            let storage = if multiply == DType::F32 {
                vec![DType::F32]
            } else {
                vec![DType::F32, multiply]
            };
            for &a in &storage {
                for &b in &storage {
                    for output in [DType::F32, DType::F16, DType::BF16] {
                        for (at, bt) in [(false, false), (true, false), (false, true)] {
                            let types = MatrixTypes {
                                a,
                                b,
                                multiply,
                                output,
                            };
                            for group in [
                                ShaderGroup::MatMulTyped {
                                    transpose_a: at,
                                    transpose_b: bt,
                                    addend: false,
                                    types,
                                },
                                ShaderGroup::MatMulTyped {
                                    transpose_a: at,
                                    transpose_b: bt,
                                    addend: true,
                                    types,
                                },
                                ShaderGroup::BlockMatMulTyped {
                                    transpose_a: at,
                                    transpose_b: bt,
                                    types,
                                },
                            ] {
                                let mut module = super::super::generate_vector(group);
                                let mut binding = 0;
                                for (_, var) in module.module.global_variables.iter_mut() {
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
                                cases += 1;
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(cases, 243);
    }
}
