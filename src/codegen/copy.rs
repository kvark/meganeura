use crate::{DType, schedule::PointwiseDAG};
use std::{
    fmt::Write,
    hash::{Hash, Hasher},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum CopySource {
    Input(u8),
    Node(u32),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct CopyNode {
    pub inputs: [CopySource; 2],
    pub widths: [u32; 2],
    pub inner: u32,
}

/// Index composition only. The optional final conversion keeps its original
/// storage boundary; no floating-point arithmetic crosses it.
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct CopyKernel {
    pub n_inputs: u8,
    pub nodes: Vec<CopyNode>,
    pub root: CopySource,
    pub storage: DType,
}

impl CopyKernel {
    pub(crate) fn hash_key(&self) -> u64 {
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        self.hash(&mut hash);
        hash.finish()
    }

    pub fn generate(&self) -> super::ShaderModule {
        let names = PointwiseDAG::input_binding_names(self.n_inputs);
        let (ty, convert, enable) = match self.storage {
            DType::F32 => ("f32", "value", ""),
            DType::F16 => ("u32", "", ""),
            DType::BF16 => ("bf16", "round_bf16(value)", "enable wgpu_bfloat16;\n"),
            _ => panic!("copy output storage {:?}", self.storage),
        };
        let mut code = enable.to_string();
        if self.storage == DType::BF16 {
            code.push_str(include_str!("../shaders/bf16_round.wgsl"));
        }
        if self.storage == DType::F16 {
            let original = include_str!("../shaders/to_f16.wgsl");
            code.push_str(
                &original
                    [original.find("fn half_bits").unwrap()..original.find("@compute").unwrap()],
            );
        }
        for name in names {
            writeln!(code, "var<storage> {name}: array<f32>;").unwrap();
        }
        writeln!(code, "var<storage, read_write> dst: array<{ty}>;\nstruct Params {{ len: u32, pad0: u32, pad1: u32, pad2: u32 }}\nvar<uniform> params: Params;").unwrap();
        let load = |source: CopySource, at: &str| match source {
            CopySource::Input(i) => format!("{}[{at}]", names[i as usize]),
            CopySource::Node(i) => format!("copy_{i}({at})"),
        };
        for (i, node) in self.nodes.iter().enumerate() {
            for source in node.inputs {
                if let CopySource::Node(child) = source {
                    assert!((child as usize) < i);
                }
            }
            let left = node.widths[0].checked_mul(node.inner).unwrap();
            let right = node.widths[1].checked_mul(node.inner).unwrap();
            let span = left.checked_add(right).unwrap();
            assert!(left > 0 && right > 0);
            writeln!(code, "fn copy_{i}(at: u32) -> f32 {{\n    let row = at / {span}u;\n    let offset = at % {span}u;").unwrap();
            if node.inputs[0] == node.inputs[1] && left == right {
                writeln!(
                    code,
                    "    return {};\n}}",
                    load(node.inputs[0], &format!("row * {left}u + offset % {left}u"))
                )
                .unwrap();
            } else {
                writeln!(
                    code,
                    "    if offset < {left}u {{ return {}; }}\n    return {};\n}}",
                    load(node.inputs[0], &format!("row * {left}u + offset")),
                    load(
                        node.inputs[1],
                        &format!("row * {right}u + offset - {left}u")
                    )
                )
                .unwrap();
            }
        }
        if self.storage == DType::F16 {
            writeln!(code, "@compute @workgroup_size(256)\nfn copy_main(@builtin(global_invocation_id) gid: vec3<u32>) {{\n    let at = gid.x * 2u;\n    if at >= params.len {{ return; }}\n    var packed = half_bits({});\n    if at + 1u < params.len {{ packed |= half_bits({}) << 16u; }}\n    dst[gid.x] = packed;\n}}", load(self.root, "at"), load(self.root, "at + 1u")).unwrap();
        } else {
            writeln!(code, "@compute @workgroup_size(256)\nfn copy_main(@builtin(global_invocation_id) gid: vec3<u32>) {{\n    if gid.x >= params.len {{ return; }}\n    let value = {};\n    dst[gid.x] = {convert};\n}}", load(self.root, "gid.x")).unwrap();
        }
        super::ShaderModule::new(&code)
    }
}
