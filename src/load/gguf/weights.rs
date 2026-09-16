//! Filling a compiled session from the file.
//!
//! [`graph::build`](super::graph::build) declared every parameter under
//! GGUF's own tensor name and in the dtype the file stores it in, so loading
//! is mostly a matter of handing each tensor's bytes straight over:
//! [`Session::set_parameter_packed`] for the block formats, which keeps the
//! file's own encoding and does not requantize, and
//! [`Session::set_parameter`] for the rest.
//!
//! Three tensors do need work on the way in, and each is a property of the
//! model rather than of the container:
//!
//! - The **embedding table** is dequantized to f16, because the gather has
//!   no block-quantized variant. It is also the one tensor read in GGUF's
//!   own row order rather than transposed — see
//!   [`GgufTensor::to_f32_rows`](super::GgufTensor::to_f32_rows).
//! - **Gemma's norm weights** are stored centred on zero and applied as
//!   `1 + w`. Folding the `+1` in here keeps every shader unaware of it.
//! - **KV caches** are not in the file at all. They are session state, and
//!   [`reset_caches`] zeroes them between generations.

use crate::Session;

use super::arch::ModelConfig;
use super::graph::{self, ModelGraph};
use super::{GgufError, GgufModel, GgufTensor};

/// What a load actually did, for callers that want to report it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LoadReport {
    /// Tensors handed over in the file's own block encoding.
    pub packed: usize,
    /// Tensors that went through f32 — the norms, biases and any weight
    /// the file stores unquantized.
    pub dequantized: usize,
    /// Parameters the graph declared but the compiled plan no longer has,
    /// the optimizer having folded them away. Named so a surprising count
    /// can be investigated rather than guessed at.
    pub skipped: Vec<String>,
}

/// Fill every weight `config` implies from `model`.
///
/// The KV caches are left alone; they are state rather than weights, and
/// [`reset_caches`] owns them.
pub fn load(
    session: &mut Session,
    model: &GgufModel,
    config: &ModelConfig,
) -> Result<LoadReport, GgufError> {
    let mut report = LoadReport::default();
    for name in graph::parameter_names(config) {
        // A parameter the optimizer folded away has nowhere to go, and
        // `set_parameter` would panic rather than say so.
        if !session.has_parameter(&name) {
            report.skipped.push(name);
            continue;
        }
        let tensor = model
            .tensors
            .get(&name)
            .ok_or_else(|| GgufError::MissingTensor(format!("the file has no `{name}`")))?;
        load_one(session, &name, tensor, config, &mut report)?;
    }
    Ok(report)
}

/// Hand one tensor over in whichever form its parameter was declared.
fn load_one(
    session: &mut Session,
    name: &str,
    tensor: &GgufTensor,
    config: &ModelConfig,
    report: &mut LoadReport,
) -> Result<(), GgufError> {
    // The table is a list of rows, already in the orientation the gather
    // wants, and is declared f16 however the file stores it.
    if name == graph::TOKEN_EMBD {
        let values = tensor.to_f32_rows()?;
        expect_len(name, values.len(), config.vocab_size * config.hidden_size)?;
        session.set_parameter(name, &values);
        report.dequantized += 1;
        return Ok(());
    }

    // Norms and biases are one-dimensional, so `to_f32` has nothing to
    // transpose and the values arrive in order.
    if tensor.dims.len() == 1 {
        let mut values = tensor.to_f32()?;
        if is_norm_weight(name) && config.architecture.norm_weight_offset_by_one() {
            for v in &mut values {
                *v += 1.0;
            }
        }
        session.set_parameter(name, &values);
        report.dequantized += 1;
        return Ok(());
    }

    match graph::weight_dtype(tensor)? {
        crate::graph::DType::F32 | crate::graph::DType::F16 => {
            // Declared f32 or f16; either way `set_parameter` takes f32 and
            // the runtime narrows it if the buffer is half-width.
            let values = tensor.to_f32()?;
            session.set_parameter(name, &values);
            report.dequantized += 1;
        }
        _ => {
            let (_dtype, bytes) = tensor.to_packed()?;
            session.set_parameter_packed(name, &bytes);
            report.packed += 1;
        }
    }
    Ok(())
}

/// Whether a name is a norm's scale, as against a bias or a projection.
///
/// Gemma's `1 + w` applies to the scales only, so this must not catch
/// `output.weight` — hence the `_norm.` rather than a bare `norm`.
fn is_norm_weight(name: &str) -> bool {
    name.ends_with("_norm.weight") || name == "output_norm.weight"
}

fn expect_len(name: &str, got: usize, want: usize) -> Result<(), GgufError> {
    if got == want {
        return Ok(());
    }
    Err(GgufError::MissingTensor(format!(
        "`{name}` dequantized to {got} values, but the architecture makes it {want}"
    )))
}

/// Zero every KV cache, so the next generation starts from an empty
/// prefix.
///
/// Attention reads `valid` rows from `position`, so stale rows beyond the
/// current prefix are never consulted — but a cache carried across a
/// *shorter* prompt would otherwise leave the tail of a previous
/// conversation attendable.
pub fn reset_caches(session: &mut Session, built: &ModelGraph, config: &ModelConfig) {
    let zeros = vec![0.0f32; built.max_seq_len * config.kv_dim()];
    for layer in 0..config.num_layers {
        for name in [
            ModelGraph::k_cache_name(layer),
            ModelGraph::v_cache_name(layer),
        ] {
            if session.has_parameter(&name) {
                session.set_parameter(&name, &zeros);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load::gguf::fixture;
    use crate::load::gguf::{GgmlType, GgufValue};

    #[test]
    fn a_norm_weight_is_recognized_but_a_projection_is_not() {
        assert!(is_norm_weight("output_norm.weight"));
        assert!(is_norm_weight("blk.0.attn_norm.weight"));
        assert!(is_norm_weight("blk.0.ffn_norm.weight"));
        assert!(is_norm_weight("blk.0.attn_q_norm.weight"));
        assert!(!is_norm_weight("output.weight"));
        assert!(!is_norm_weight("blk.0.attn_q.weight"));
        assert!(!is_norm_weight("blk.0.attn_norm.bias"));
    }

    /// The load path and the graph must agree about every tensor, or the
    /// loader would hand packed bytes to an f32 buffer.
    #[test]
    fn the_loader_and_the_builder_agree_on_every_dtype() {
        for arch in ["llama", "qwen2", "qwen3", "gemma", "gemma2"] {
            let model = fixture::model(arch);
            let config = ModelConfig::from_gguf(&model).unwrap();
            for name in graph::parameter_names(&config) {
                let tensor = &model.tensors[&name];
                if name == graph::TOKEN_EMBD || tensor.dims.len() == 1 {
                    continue;
                }
                // A fixture stores f32, so both sides must say f32.
                assert_eq!(
                    graph::weight_dtype(tensor).unwrap(),
                    crate::graph::DType::F32,
                    "{arch}/{name}"
                );
            }
        }
    }

    #[test]
    fn the_embedding_table_is_read_in_row_order_not_transposed() {
        let model = fixture::model("llama");
        let config = ModelConfig::from_gguf(&model).unwrap();
        let table = &model.tensors[graph::TOKEN_EMBD];
        let rows = table.to_f32_rows().unwrap();
        let transposed = table.to_f32().unwrap();
        assert_eq!(rows.len(), config.vocab_size * config.hidden_size);
        assert_eq!(rows.len(), transposed.len());
        assert_ne!(
            rows, transposed,
            "the fixture ramp must distinguish the orientations"
        );
        // Row `v` of the table is element (0..hidden, v) in GGUF order,
        // which is a contiguous run.
        let hidden = config.hidden_size;
        for v in 0..4 {
            for h in 0..hidden {
                assert_eq!(
                    rows[v * hidden + h],
                    transposed[h * config.vocab_size + v],
                    "row {v} column {h}"
                );
            }
        }
    }

    #[test]
    fn gemma_norm_weights_gain_the_one_that_llama_does_not() {
        // The fold is applied on load, so the shaders never learn about it.
        for (arch, offset) in [("llama", false), ("gemma", true)] {
            let model = fixture::model(arch);
            let config = ModelConfig::from_gguf(&model).unwrap();
            assert_eq!(config.architecture.norm_weight_offset_by_one(), offset);

            let stored = model.tensors["blk.0.attn_norm.weight"].to_f32().unwrap();
            let expected: Vec<f32> = stored
                .iter()
                .map(|v| if offset { v + 1.0 } else { *v })
                .collect();
            // Mirror what `load_one` does for a 1-D norm weight.
            let mut values = stored.clone();
            if is_norm_weight("blk.0.attn_norm.weight")
                && config.architecture.norm_weight_offset_by_one()
            {
                for v in &mut values {
                    *v += 1.0;
                }
            }
            assert_eq!(values, expected, "{arch}");
        }
    }

    #[test]
    fn a_missing_tensor_is_reported_by_name() {
        // `load` needs a Session, which needs a GPU; this covers the same
        // lookup the loader performs first.
        let model = fixture::model_with("llama", |m| {
            m.tensors.remove("blk.0.attn_v.weight");
        });
        let config = ModelConfig::from_gguf(&model).unwrap();
        let missing: Vec<String> = graph::parameter_names(&config)
            .into_iter()
            .filter(|n| !model.tensors.contains_key(n))
            .collect();
        assert_eq!(missing, vec!["blk.0.attn_v.weight".to_string()]);
    }

    #[test]
    fn a_quantized_weight_reaches_the_packed_path() {
        let model = fixture::model_with("llama", |m| {
            m.tensors.insert(
                "blk.0.attn_q.weight".to_string(),
                crate::load::gguf::GgufTensor::new(
                    vec![64, 64],
                    GgmlType::Q4_0,
                    vec![0; (64 * 64 / 32) * 18],
                ),
            );
        });
        let tensor = &model.tensors["blk.0.attn_q.weight"];
        let dtype = graph::weight_dtype(tensor).unwrap();
        assert!(
            crate::compile::WeightFormat::from_dtype(dtype).uses_reduced_storage(),
            "a Q4_0 tensor must take set_parameter_packed, not set_parameter"
        );
        // And the packed bytes must be exactly what the parameter expects.
        let (packed_dtype, bytes) = tensor.to_packed().unwrap();
        assert_eq!(packed_dtype, dtype);
        assert!(!bytes.is_empty());
    }

    #[test]
    fn every_fixture_architecture_has_a_tensor_for_every_parameter() {
        for arch in ["llama", "qwen2", "qwen3", "gemma", "gemma2"] {
            let model = fixture::model(arch);
            let config = ModelConfig::from_gguf(&model).unwrap();
            for name in graph::parameter_names(&config) {
                assert!(
                    model.tensors.contains_key(&name),
                    "{arch} fixture is missing {name}"
                );
            }
        }
    }

    #[test]
    fn the_report_counts_what_it_did() {
        let mut report = LoadReport::default();
        report.packed += 1;
        report.dequantized += 2;
        report.skipped.push("blk.0.ffn_gate.weight".to_string());
        assert_eq!(report.packed, 1);
        assert_eq!(report.dequantized, 2);
        assert_eq!(report.skipped.len(), 1);
    }

    #[test]
    fn an_untied_head_is_a_projection_and_keeps_its_own_dtype() {
        let model = fixture::model_with("llama", |m| {
            m.tensors.insert(
                graph::OUTPUT.to_string(),
                crate::load::gguf::GgufTensor::new(
                    vec![64, 32 * 8],
                    GgmlType::Q8_0,
                    vec![0; (64 * 32 * 8 / 32) * 34],
                ),
            );
            fixture::set_arch_key(m, "vocab_size", GgufValue::U32(256));
        });
        let tensor = &model.tensors[graph::OUTPUT];
        assert_eq!(
            graph::weight_dtype(tensor).unwrap(),
            crate::graph::DType::Q8_0,
            "the head is a weight, not a lookup table, so it stays packed"
        );
    }
}
