use crate::{DType, schedule::PointwiseDAG};
use std::{
    fmt::Write,
    hash::{Hash, Hasher},
};

/// A window reduction with a saved discrete derivative. Storage types are
/// explicit; pooling and partial sums always compute in F32.
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum PoolKernel {
    Forward {
        prologue: PointwiseDAG,
        inputs: Vec<DType>,
    },
    Backward {
        output: DType,
        sum: bool,
    },
}

impl PoolKernel {
    pub(crate) fn hash_key(&self) -> u64 {
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        self.hash(&mut hash);
        hash.finish()
    }

    pub(crate) fn binding_names(&self) -> Vec<&'static str> {
        match *self {
            Self::Forward { ref prologue, .. } => {
                let mut names = PointwiseDAG::input_binding_names(prologue.n_inputs).to_vec();
                names.extend(["dst", "winner"]);
                names
            }
            Self::Backward { sum, .. } => {
                let mut names = vec!["dy", "winner", "dst"];
                if sum {
                    names.push("partial");
                }
                names
            }
        }
    }

    pub fn generate(&self) -> super::ShaderModule {
        let types = match *self {
            Self::Forward { ref inputs, .. } => inputs.as_slice(),
            Self::Backward { ref output, .. } => std::slice::from_ref(output),
        };
        let mut src = String::new();
        if types.contains(&DType::BF16) {
            src.push_str("enable wgpu_bfloat16;\n");
        }
        if types.contains(&DType::F16) {
            src.push_str("enable f16;\n");
        }
        if types.contains(&DType::BF16) {
            src.push_str(include_str!("../shaders/bf16_round.wgsl"));
        }
        src.push_str(
            include_str!("../shaders/max_pool_2d.wgsl")
                .split("var<storage>")
                .next()
                .unwrap(),
        );
        src.push_str("var<uniform> params: Params;\n");
        match *self {
            Self::Forward {
                ref prologue,
                ref inputs,
            } => {
                assert_eq!(inputs.len(), prologue.n_inputs as usize);
                let names = PointwiseDAG::input_binding_names(prologue.n_inputs);
                for (name, ty) in names.iter().zip(inputs) {
                    writeln!(src, "var<storage> {name}: array<{}>;", scalar(*ty)).unwrap();
                }
                src.push_str("var<storage, read_write> dst: array<f32>;\nvar<storage, read_write> winner: array<u32>;\nfn sample(at: u32) -> f32 {\n");
                src.push_str(&prologue.emit_body_with(
                    |i| format!("f32({}[at])", names[i as usize]),
                    |i, divisor, modulus| {
                        format!("f32({}[(at / {divisor}u) % {modulus}u])", names[i as usize])
                    },
                ));
                writeln!(src, "return v{};\n}}", prologue.output).unwrap();
                src.push_str(include_str!("../shaders/pool_tape_forward.wgsl"));
            }
            Self::Backward { output, sum } => {
                src.push_str("var<storage> dy: array<f32>;\nvar<storage> winner: array<u32>;\n");
                writeln!(
                    src,
                    "var<storage, read_write> dst: array<{}>;",
                    scalar(output)
                )
                .unwrap();
                src.push_str(include_str!("../shaders/pool_tape_backward.wgsl"));
                let value = match output {
                    DType::BF16 => "round_bf16(value)",
                    DType::F16 => "f16(value)",
                    DType::F32 => "value",
                    _ => unreachable!(),
                };
                if sum {
                    src.push_str("var<storage, read_write> partial: array<f32>;\nvar<workgroup> scratch: array<f32, 256>;\n@compute @workgroup_size(256)\nfn pool(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {\nlet lane = local.x;\nlet plane = group.x;\nlet inner = params.in_h * params.in_w;\nvar acc = 0.0;\nfor (var col = lane; col < inner; col += 256u) {\nlet at = plane * inner + col;\nlet value = gradient(at);\n");
                    writeln!(src, "dst[at] = {value};\nacc += value;\n}}\nscratch[lane] = acc;\nworkgroupBarrier();").unwrap();
                    let mut stride = 128;
                    while stride != 0 {
                        writeln!(src, "if lane < {stride}u {{ scratch[lane] += scratch[lane + {stride}u]; }}\nworkgroupBarrier();").unwrap();
                        stride /= 2;
                    }
                    src.push_str("if lane == 0u { partial[plane] = scratch[0]; }\n}\n");
                } else {
                    src.push_str("@compute @workgroup_size(256)\nfn pool(@builtin(global_invocation_id) gid: vec3<u32>) {\nlet at = gid.x;\nif at >= params.batch * params.channels * params.in_h * params.in_w { return; }\nlet value = gradient(at);\n");
                    writeln!(src, "dst[at] = {value};\n}}").unwrap();
                }
            }
        }
        super::ShaderModule::new(&src)
    }
}

fn scalar(ty: DType) -> &'static str {
    match ty {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        _ => panic!("pool scalar storage {ty:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schedule::Pw;

    fn validate(mut shader: super::super::ShaderModule) {
        let mut binding = 0;
        for (_, var) in shader.module.global_variables.iter_mut() {
            if matches!(
                var.space,
                naga::AddressSpace::Storage { .. } | naga::AddressSpace::Uniform
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

    #[test]
    fn spatial_tape_shader_types() {
        let mut count = 0;
        for ty in [DType::F32, DType::F16, DType::BF16] {
            for inputs in [vec![ty], vec![ty, DType::F32], vec![ty, DType::BF16]] {
                let prologue = if inputs.len() == 1 {
                    PointwiseDAG {
                        n_inputs: 1,
                        ops: vec![Pw::LoadInput(0)],
                        output: 0,
                    }
                } else {
                    PointwiseDAG {
                        n_inputs: 2,
                        ops: vec![
                            Pw::LoadInput(0),
                            Pw::LoadBroadcast {
                                input: 1,
                                divisor: 49,
                                modulus: 3,
                            },
                            Pw::Add(0, 1),
                        ],
                        output: 2,
                    }
                };
                let module = PoolKernel::Forward { prologue, inputs }.generate();
                validate(module);
                count += 1;
            }
            for sum in [false, true] {
                let module = PoolKernel::Backward { output: ty, sum }.generate();
                validate(module);
                count += 1;
            }
        }
        assert_eq!(count, 15);
    }
}
