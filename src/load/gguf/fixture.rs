//! Synthetic GGUF models for tests.
//!
//! Building a real file would mean shipping weights; these assemble a
//! [`GgufModel`] in memory instead, with every tensor the architecture
//! implies at the shape it implies, so the builder and the weight loader
//! can be exercised against a complete inventory.

use std::collections::HashMap;

use super::arch::ModelConfig;
use super::graph;
use super::{GgmlType, GgufModel, GgufTensor, GgufValue};

/// A model whose metadata describes `arch` at small but realistic
/// dimensions, and whose tensors are whatever [`ModelConfig`] then implies.
///
/// Values are a cheap deterministic ramp rather than zeros: a zeroed model
/// makes transposes and orderings invisible, so a test could pass on a
/// wrong layout.
pub fn model(arch: &str) -> GgufModel {
    let mut m = GgufModel {
        metadata: super::arch::test_metadata(arch),
        tensors: HashMap::new(),
    };
    // A first pass over metadata alone settles the dimensions; the tensor
    // list then follows from them.
    let config = ModelConfig::from_gguf(&m).expect("fixture metadata describes a known model");
    for (name, dims) in tensor_shapes(&config) {
        m.tensors.insert(name, f32_tensor(dims));
    }
    m
}

/// [`model`], then whatever `edit` does to it before the config is read
/// again. For tests that need one tensor at a different type or shape.
pub fn model_with(arch: &str, edit: impl FnOnce(&mut GgufModel)) -> GgufModel {
    let mut m = model(arch);
    edit(&mut m);
    m
}

/// Set a metadata key on the architecture's own namespace.
pub fn set_arch_key(m: &mut GgufModel, suffix: &str, value: GgufValue) {
    let arch = m
        .architecture()
        .expect("fixture always declares an architecture")
        .to_string();
    m.metadata.insert(format!("{arch}.{suffix}"), value);
}

/// Every tensor the architecture needs, in GGUF's own dimension order
/// (fastest-varying first).
pub fn tensor_shapes(config: &ModelConfig) -> Vec<(String, Vec<usize>)> {
    let hidden = config.hidden_size;
    let ffn = config.intermediate_size;
    let q_dim = config.q_dim();
    let kv_dim = config.kv_dim();
    let head_dim = config.head_dim as usize;

    graph::parameter_names(config)
        .into_iter()
        .map(|name| {
            let dims = match name.rsplit_once('.').map(|(_, last)| last) {
                // Norm weights and biases are one row wide, named by
                // whatever they normalize.
                Some("bias") => vec![bias_width(&name, config)],
                _ if name.ends_with("_norm.weight") || name == "output_norm.weight" => {
                    if name.contains("attn_q_norm") || name.contains("attn_k_norm") {
                        vec![head_dim]
                    } else {
                        vec![hidden]
                    }
                }
                _ => match () {
                    _ if name == graph::TOKEN_EMBD => vec![hidden, config.vocab_size],
                    _ if name == graph::OUTPUT => vec![hidden, config.vocab_size],
                    _ if name.ends_with("attn_q.weight") => vec![hidden, q_dim],
                    _ if name.ends_with("attn_k.weight") => vec![hidden, kv_dim],
                    _ if name.ends_with("attn_v.weight") => vec![hidden, kv_dim],
                    _ if name.ends_with("attn_output.weight") => vec![q_dim, hidden],
                    _ if name.ends_with("ffn_gate.weight") => vec![hidden, ffn],
                    _ if name.ends_with("ffn_up.weight") => vec![hidden, ffn],
                    _ if name.ends_with("ffn_down.weight") => vec![ffn, hidden],
                    _ => unreachable!("fixture has no shape for {name}"),
                },
            };
            (name, dims)
        })
        .collect()
}

fn bias_width(name: &str, config: &ModelConfig) -> usize {
    if name.ends_with("attn_q.bias") {
        config.q_dim()
    } else if name.ends_with("attn_k.bias") || name.ends_with("attn_v.bias") {
        config.kv_dim()
    } else if name.ends_with("ffn_up.bias") {
        config.intermediate_size
    } else {
        config.hidden_size
    }
}

/// An f32 tensor of `dims`, filled with a bounded deterministic ramp.
pub fn f32_tensor(dims: Vec<usize>) -> GgufTensor {
    let count: usize = dims.iter().product();
    let mut bytes = Vec::with_capacity(count * 4);
    for i in 0..count {
        // Bounded and sign-alternating, so a wrong transpose shows up as a
        // different value rather than a different magnitude.
        let v = ((i % 17) as f32 - 8.0) / 32.0;
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    GgufTensor::new(dims, GgmlType::F32, bytes)
}
