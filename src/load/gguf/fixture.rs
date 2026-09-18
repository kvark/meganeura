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
        let tensor = if name == "rope_freqs.weight" {
            let mut factors = vec![1.0e30; dims[0]];
            factors[..(config.rope_dim / 8) as usize].fill(1.0);
            f32_values_tensor(dims, factors)
        } else if name.ends_with(".layer_output_scale.weight") {
            filled_f32_tensor(dims, 1.0)
        } else {
            f32_tensor(dims)
        };
        m.tensors.insert(name, tensor);
    }
    if config.architecture.uses_per_layer_embeddings() {
        let span = config.num_layers * config.per_layer_embed_size;
        m.tensors.insert(
            "per_layer_token_embd.weight".to_string(),
            f32_tensor(vec![span, config.vocab_size]),
        );
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
    let mut names = graph::parameter_names(config);
    names.extend(optional_names_present(config));
    names
        .into_iter()
        .map(|name| {
            let layer = name
                .strip_prefix("blk.")
                .and_then(|rest| rest.split('.').next())
                .and_then(|number| number.parse::<usize>().ok());
            let ffn = layer
                .map(|index| config.layer_ffn_size(index))
                .unwrap_or(config.intermediate_size);
            let head_dim = layer
                .map(|index| config.head_dim_at(index) as usize)
                .unwrap_or(config.head_dim as usize);
            let q_dim = layer
                .map(|index| config.num_heads as usize * config.head_dim_at(index) as usize)
                .unwrap_or_else(|| config.q_dim());
            let kv_dim = layer
                .map(|index| config.kv_dim_at(index))
                .unwrap_or_else(|| config.kv_dim_at(0));
            let dims = match name.rsplit_once('.').map(|(_, last)| last) {
                // Norm weights and biases are one row wide, named by
                // whatever they normalize.
                Some("bias") => vec![bias_width(&name, config)],
                _ if name == "rope_freqs.weight" => vec![(config.rope_dim / 2) as usize],
                _ if name.ends_with(".layer_output_scale.weight") => vec![1],
                _ if name == "per_layer_proj_norm.weight" => {
                    vec![config.per_layer_embed_size]
                }
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
                    _ if name == graph::PER_LAYER_PROJ => {
                        vec![hidden, config.num_layers * config.per_layer_embed_size]
                    }
                    _ if name.ends_with(".inp_gate.weight") => {
                        vec![hidden, config.per_layer_embed_size]
                    }
                    _ if name.ends_with(".proj.weight") => {
                        vec![config.per_layer_embed_size, hidden]
                    }
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

fn filled_f32_tensor(dims: Vec<usize>, value: f32) -> GgufTensor {
    let count: usize = dims.iter().product();
    let bytes = (0..count).flat_map(|_| value.to_le_bytes()).collect();
    GgufTensor::new(dims, GgmlType::F32, bytes)
}

fn f32_values_tensor(dims: Vec<usize>, values: Vec<f32>) -> GgufTensor {
    assert_eq!(dims.iter().product::<usize>(), values.len());
    let bytes = values.into_iter().flat_map(f32::to_le_bytes).collect();
    GgufTensor::new(dims, GgmlType::F32, bytes)
}

/// The optional tensors a representative GGUF file of this architecture carries.
///
/// Not simply every optional name: Qwen2 biases Q, K and V but leaves the
/// attention output unbiased, and Gemma4's layer-output scale is optional.
/// A fixture that invented a bias would hide exactly the mismatch that
/// broke real Qwen2 files.
fn optional_names_present(config: &ModelConfig) -> Vec<String> {
    let arch = config.architecture;
    let mut names = Vec::new();
    for layer in 0..config.num_layers {
        let p = format!("blk.{layer}");
        if matches!(
            arch,
            super::arch::Architecture::Qwen2 | super::arch::Architecture::Phi2
        ) {
            for part in ["attn_q", "attn_k", "attn_v"] {
                names.push(format!("{p}.{part}.bias"));
            }
        }
        if arch == super::arch::Architecture::Phi2 {
            names.push(format!("{p}.attn_output.bias"));
            names.push(format!("{p}.ffn_up.bias"));
            names.push(format!("{p}.ffn_down.bias"));
        }
        if arch.scales_block_outputs() {
            names.push(format!("{p}.layer_output_scale.weight"));
        }
    }
    names
}

fn bias_width(name: &str, config: &ModelConfig) -> usize {
    if name.ends_with("attn_q.bias") {
        config.q_dim()
    } else if name.ends_with("attn_k.bias") || name.ends_with("attn_v.bias") {
        config.kv_dim_at(0)
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
