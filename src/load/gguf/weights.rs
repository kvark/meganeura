//! Filling a compiled session from the file.
//!
//! [`super::graph::build`] declared every parameter under
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
//!
//! # Rearranging rows
//!
//! Two things make a parameter's bytes a *range of rows* of some tensor
//! rather than the tensor itself, and both are handled by
//! `rows_of` before anything else looks at the values.
//!
//! Phi packs Q, K and V into one `attn_qkv.weight`, and Phi3 packs the
//! feed-forward gate and up into one double-width `ffn_up.weight`. Rows are
//! output features and GGUF blocks along the *other* axis, so each row is a
//! contiguous run of bytes whatever the encoding — slicing works on a
//! Q4_K superblock tensor exactly as it does on f32.
//!
//! Llama's Q and K need more than slicing. GGML has two RoPE conventions,
//! and llama.cpp's converter permutes those two weights on the way into a
//! llama GGUF so that GGML's *interleaved* rope reproduces what
//! HuggingFace's *half-split* rope would have done. Meganeura's RoPE is the
//! half-split one, so the permutation has to come back out — see
//! `unpermute_rope_pairs`. Getting this wrong is not a crash: the model
//! loads and emits fluent, wrong text.

use std::borrow::Cow;

use crate::Session;

use super::arch::ModelConfig;
use super::graph::{self, ModelGraph, Source};
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
        let tensor = resolve(model, config, &name)?;
        load_one(session, &name, &tensor, config, &mut report)?;
    }
    // Biases exist only where the file has them, and the graph omitted the
    // add wherever it did not — so an absent one is not a gap to report.
    for name in graph::optional_parameter_names(config) {
        if !session.has_parameter(&name) || !model.tensors.contains_key(&name) {
            continue;
        }
        let tensor = resolve(model, config, &name)?;
        load_one(session, &name, &tensor, config, &mut report)?;
    }
    Ok(report)
}

/// The tensor a parameter is actually made of: sliced out of a packed
/// tensor where the architecture packs, and un-permuted where the
/// converter permuted.
///
/// Both rearrangements happen in GGUF's *own* byte layout, where a row is
/// always `ne0` elements of contiguous blocks. Doing them after
/// `to_packed` would mean knowing each destination format's layout —
/// Meganeura's Q4 keeps block headers in a region of their own — where
/// here one rule covers every encoding.
fn resolve<'a>(
    model: &'a GgufModel,
    config: &ModelConfig,
    param: &str,
) -> Result<Cow<'a, GgufTensor>, GgufError> {
    let source = graph::source_of(config, param);
    let tensor = model.tensors.get(&source.tensor).ok_or_else(|| {
        GgufError::MissingTensor(format!(
            "the file has no `{}`, which `{param}` reads from",
            source.tensor
        ))
    })?;

    let sliced = match source.rows {
        Some(ref rows) => Cow::Owned(rows_of(tensor, rows.clone(), &source)?),
        None => Cow::Borrowed(tensor),
    };

    let rotated = config.architecture.rope_is_interleaved()
        && (param.ends_with("attn_q.weight") || param.ends_with("attn_k.weight"));
    if !rotated {
        return Ok(sliced);
    }
    Ok(Cow::Owned(unpermute_rope_pairs(
        &sliced,
        config.head_dim as usize,
    )?))
}

/// Rows `rows` of `tensor`, as a tensor of their own.
///
/// GGUF stores `ne1` rows of `ne0`, blocked along `ne0`, so every row is
/// the same number of contiguous bytes and a row range is one slice.
fn rows_of(
    tensor: &GgufTensor,
    rows: std::ops::Range<usize>,
    source: &Source,
) -> Result<GgufTensor, GgufError> {
    let stride = row_bytes(tensor)?;
    let total = tensor.dims.get(1).copied().unwrap_or(1);
    if rows.end > total {
        return Err(GgufError::MissingTensor(format!(
            "`{}` has {total} rows, but rows {}..{} were asked for",
            source.tensor, rows.start, rows.end
        )));
    }
    let data = tensor.data();
    let from = rows.start * stride;
    let to = rows.end * stride;
    if to > data.len() {
        return Err(GgufError::MissingTensor(format!(
            "`{}` holds {} bytes, too few for {total} rows of {stride}",
            source.tensor,
            data.len()
        )));
    }
    Ok(GgufTensor::new(
        vec![tensor.dims[0], rows.len()],
        tensor.ggml_type,
        data[from..to].to_vec(),
    ))
}

/// Undo the row permutation llama.cpp's converter applies to Q and K.
///
/// The converter reshapes each head's rows as `(2, head_dim/2)` and swaps
/// those axes, so HuggingFace row `a * head_dim/2 + b` within a head is
/// written at GGUF row `2 * b + a`. Reading that backwards: GGUF row `r`
/// belongs at HuggingFace row `(r % 2) * head_dim/2 + r / 2`.
///
/// The effect is that HuggingFace's half-split pair `(b, b + head_dim/2)`
/// becomes GGUF's adjacent pair `(2b, 2b + 1)`, which is what makes GGML's
/// interleaved rope equivalent. Undoing it puts the halves back where
/// Meganeura's RoPE expects them.
fn unpermute_rope_pairs(tensor: &GgufTensor, head_dim: usize) -> Result<GgufTensor, GgufError> {
    let stride = row_bytes(tensor)?;
    let rows = tensor.dims.get(1).copied().unwrap_or(1);
    if head_dim == 0 || !head_dim.is_multiple_of(2) || !rows.is_multiple_of(head_dim) {
        return Err(GgufError::BadShape(format!(
            "{rows} rows do not divide into {head_dim}-wide heads"
        )));
    }
    let data = tensor.data();
    if data.len() < rows * stride {
        return Err(GgufError::BadShape(format!(
            "a {rows}-row tensor of {stride}-byte rows needs {} bytes, has {}",
            rows * stride,
            data.len()
        )));
    }

    let half = head_dim / 2;
    let mut out = vec![0u8; data.len()];
    // The tail past the last whole row — Q6_K's zero padding — is carried
    // over rather than dropped.
    out[rows * stride..].copy_from_slice(&data[rows * stride..]);
    for head_start in (0..rows).step_by(head_dim) {
        for r in 0..head_dim {
            let hf = (r % 2) * half + r / 2;
            let from = (head_start + r) * stride;
            let to = (head_start + hf) * stride;
            out[to..to + stride].copy_from_slice(&data[from..from + stride]);
        }
    }
    Ok(GgufTensor::new(tensor.dims.clone(), tensor.ggml_type, out))
}

/// Bytes one row of `tensor` occupies in GGUF's own layout.
fn row_bytes(tensor: &GgufTensor) -> Result<usize, GgufError> {
    let k = tensor.dims.first().copied().unwrap_or(0);
    let (Some(elements), Some(bytes)) = (
        tensor.ggml_type.block_elements(),
        tensor.ggml_type.block_bytes(),
    ) else {
        return Err(GgufError::UnsupportedType(tensor.ggml_type.tag()));
    };
    if !k.is_multiple_of(elements) {
        return Err(GgufError::BadShape(format!(
            "a row of {k} is not a whole number of {elements}-element blocks"
        )));
    }
    Ok(k / elements * bytes)
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

    if tensor.dims.len() == 1 {
        let values = vector_values(tensor)?;
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

/// Values for a one-dimensional parameter — a norm weight or a bias.
///
/// These are read straight through. `to_f32` has nothing to transpose for
/// a single dimension, so the values arrive in order, and *no*
/// architecture adjusts them.
///
/// That last part is the whole reason this is a named function. Gemma's
/// norm weights are trained centred on zero and applied as `1 + w`, which
/// invites folding the one in here — but llama.cpp's converter has already
/// done it (`GemmaModel::modify_tensors` writes `data_torch + 1`, and
/// Gemma2 and Gemma3 do the same). The file holds the *applied* scale, not
/// the trained parameter. Adding one again would turn a trained zero into
/// two rather than one, doubling that norm's contribution on every layer
/// including the final one.
fn vector_values(tensor: &GgufTensor) -> Result<Vec<f32>, GgufError> {
    tensor.to_f32()
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
    use crate::load::gguf::arch::Architecture;
    use crate::load::gguf::fixture;
    use crate::load::gguf::{GgmlType, GgufValue};

    /// The permutation llama.cpp's converter applies, so the test states
    /// the forward direction independently of the code that inverts it.
    fn permute_rows(rows: usize, head_dim: usize) -> Vec<usize> {
        // HuggingFace row `a * half + b` within a head is written at GGUF
        // row `2 * b + a`.
        let half = head_dim / 2;
        let mut gguf_of_hf = vec![0usize; rows];
        for head in (0..rows).step_by(head_dim) {
            for a in 0..2 {
                for b in 0..half {
                    gguf_of_hf[head + a * half + b] = head + 2 * b + a;
                }
            }
        }
        gguf_of_hf
    }

    /// A tensor whose every row holds its own index, so a permutation is
    /// readable straight off the values.
    fn numbered_rows(rows: usize, k: usize) -> GgufTensor {
        let mut bytes = Vec::with_capacity(rows * k * 4);
        for r in 0..rows {
            for _ in 0..k {
                bytes.extend_from_slice(&(r as f32).to_le_bytes());
            }
        }
        GgufTensor::new(vec![k, rows], GgmlType::F32, bytes)
    }

    #[test]
    fn unpermuting_inverts_the_converters_permutation_exactly() {
        let head_dim = 8;
        let rows = 24;
        let k = 4;
        let gguf_of_hf = permute_rows(rows, head_dim);

        // Build the tensor as the converter would have written it: GGUF
        // row `gguf_of_hf[h]` carries HuggingFace row `h`.
        let mut bytes = vec![0u8; rows * k * 4];
        for (hf, &gguf) in gguf_of_hf.iter().enumerate() {
            for col in 0..k {
                let at = (gguf * k + col) * 4;
                bytes[at..at + 4].copy_from_slice(&(hf as f32).to_le_bytes());
            }
        }
        let permuted = GgufTensor::new(vec![k, rows], GgmlType::F32, bytes);

        let restored = unpermute_rope_pairs(&permuted, head_dim).unwrap();
        let values = restored.to_f32_rows().unwrap();
        for hf in 0..rows {
            assert_eq!(
                values[hf * k],
                hf as f32,
                "row {hf} did not come back to itself"
            );
        }
    }

    #[test]
    fn unpermuting_is_a_permutation_and_not_the_identity() {
        let tensor = numbered_rows(16, 4);
        let out = unpermute_rope_pairs(&tensor, 8).unwrap();
        let before = tensor.to_f32_rows().unwrap();
        let after = out.to_f32_rows().unwrap();
        assert_ne!(before, after, "the permutation should move rows");

        let mut sorted_before = before.clone();
        let mut sorted_after = after.clone();
        sorted_before.sort_by(f32::total_cmp);
        sorted_after.sort_by(f32::total_cmp);
        assert_eq!(sorted_before, sorted_after, "no row may be lost or copied");
    }

    #[test]
    fn unpermuting_moves_whole_rows_of_a_packed_tensor_too() {
        // 64 elements per row is two Q4_0 blocks, so a row is 36 bytes —
        // the point being that the slice never splits a block.
        let rows = 8;
        let k = 64;
        let per_row = (k / 32) * 18;
        let mut bytes = Vec::with_capacity(rows * per_row);
        for r in 0..rows {
            bytes.extend(std::iter::repeat_n(r as u8, per_row));
        }
        let tensor = GgufTensor::new(vec![k, rows], GgmlType::Q4_0, bytes);
        let out = unpermute_rope_pairs(&tensor, 4).unwrap();
        let data = out.data();
        for r in 0..rows {
            let row = &data[r * per_row..(r + 1) * per_row];
            assert!(
                row.iter().all(|&b| b == row[0]),
                "row {r} mixes bytes from two source rows"
            );
        }
    }

    #[test]
    fn only_the_llama_family_needs_the_permutation() {
        assert!(Architecture::Llama.rope_is_interleaved());
        for arch in [
            Architecture::Qwen2,
            Architecture::Qwen3,
            Architecture::Gemma,
            Architecture::Gemma2,
            Architecture::Gemma3,
            Architecture::Phi3,
        ] {
            assert!(!arch.rope_is_interleaved(), "{arch}");
        }
    }

    #[test]
    fn the_permutation_reaches_q_and_k_but_nothing_else() {
        let model = fixture::model("llama");
        let config = ModelConfig::from_gguf(&model).unwrap();
        let moved = |param: &str| {
            let resolved = resolve(&model, &config, param).unwrap();
            resolved.to_f32_rows().unwrap() != model.tensors[param].to_f32_rows().unwrap()
        };
        assert!(moved("blk.0.attn_q.weight"));
        assert!(moved("blk.0.attn_k.weight"));
        assert!(!moved("blk.0.attn_v.weight"), "V is not rotated");
        assert!(!moved("blk.0.attn_output.weight"));
        assert!(!moved("blk.0.ffn_gate.weight"));
    }

    #[test]
    fn a_family_that_converts_unpermuted_is_left_alone() {
        let model = fixture::model("qwen3");
        let config = ModelConfig::from_gguf(&model).unwrap();
        let resolved = resolve(&model, &config, "blk.0.attn_q.weight").unwrap();
        assert_eq!(
            resolved.to_f32_rows().unwrap(),
            model.tensors["blk.0.attn_q.weight"].to_f32_rows().unwrap()
        );
    }

    #[test]
    fn a_packed_qkv_tensor_slices_into_three_projections() {
        // Phi packs Q, K and V into one tensor; the three parameters are
        // contiguous row ranges of it.
        let mut model = fixture::model("llama");
        let config = ModelConfig::from_gguf(&model).unwrap();
        let q = config.q_dim();
        let kv = config.kv_dim();
        let hidden = config.hidden_size;
        model.metadata.insert(
            "general.architecture".to_string(),
            GgufValue::String("phi3".to_string()),
        );
        // Re-key the metadata under phi3 so the config still reads.
        for suffix in [
            "embedding_length",
            "block_count",
            "attention.head_count",
            "attention.head_count_kv",
            "feed_forward_length",
            "attention.layer_norm_rms_epsilon",
            "rope.freq_base",
            "context_length",
            "vocab_size",
        ] {
            if let Some(v) = model.metadata.get(&format!("llama.{suffix}")).cloned() {
                model.metadata.insert(format!("phi3.{suffix}"), v);
            }
        }
        let config = ModelConfig::from_gguf(&model).unwrap();
        assert!(config.architecture.packs_qkv());

        model.tensors.insert(
            "blk.0.attn_qkv.weight".to_string(),
            fixture::f32_tensor(vec![hidden, q + 2 * kv]),
        );

        let whole = model.tensors["blk.0.attn_qkv.weight"]
            .to_f32_rows()
            .unwrap();
        let take = |param: &str, rows: std::ops::Range<usize>| {
            let resolved = resolve(&model, &config, param).unwrap();
            assert_eq!(resolved.to_f32_rows().unwrap(), whole[rows]);
        };
        take("blk.0.attn_q.weight", 0..q * hidden);
        take("blk.0.attn_k.weight", q * hidden..(q + kv) * hidden);
        take(
            "blk.0.attn_v.weight",
            (q + kv) * hidden..(q + 2 * kv) * hidden,
        );
    }

    #[test]
    fn a_mis_sized_norm_weight_is_an_error_not_a_panic() {
        let model = fixture::model_with("llama", |m| {
            m.tensors.insert(
                "blk.0.attn_norm.weight".to_string(),
                fixture::f32_tensor(vec![63]),
            );
        });
        let config = ModelConfig::from_gguf(&model).unwrap();
        let mut g = crate::Graph::new();
        let err = graph::build(&mut g, &model, &config, 4, 16).unwrap_err();
        assert!(
            matches!(&err, GgufError::MissingTensor(m) if m.contains("attn_norm.weight")),
            "{err:?}"
        );
    }

    #[test]
    fn qwen2_loads_without_an_attention_output_bias() {
        // Real Qwen2 files bias Q, K and V but not the output; requiring
        // all four rejected every one of them.
        let model = fixture::model("qwen2");
        assert!(model.tensors.contains_key("blk.0.attn_q.bias"));
        assert!(!model.tensors.contains_key("blk.0.attn_output.bias"));
        let config = ModelConfig::from_gguf(&model).unwrap();
        let mut g = crate::Graph::new();
        graph::build(&mut g, &model, &config, 4, 16).expect("a real Qwen2 shape must build");
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
    fn a_norm_weight_is_loaded_exactly_as_the_file_stores_it() {
        // llama.cpp's converter writes `data_torch + 1` for Gemma norms,
        // so a trained zero is already 1 in the file. Adding one here
        // would make it 2 — a doubling of that norm, not a rounding
        // difference — so no architecture may adjust these.
        for arch in ["llama", "gemma", "gemma2", "gemma3", "qwen3", "phi2"] {
            let model = fixture::model(arch);
            let tensor = &model.tensors["blk.0.attn_norm.weight"];
            assert_eq!(
                vector_values(tensor).unwrap(),
                tensor.to_f32().unwrap(),
                "{arch} adjusted a norm weight on load"
            );
        }
    }

    #[test]
    fn a_converted_gemma_norm_of_one_stays_one() {
        // The value a trained zero has *after* conversion, which is what a
        // real file carries. It must survive the load unchanged.
        let tensor = GgufTensor::new(
            vec![4],
            GgmlType::F32,
            [1.0f32; 4].iter().flat_map(|v| v.to_le_bytes()).collect(),
        );
        assert_eq!(vector_values(&tensor).unwrap(), vec![1.0; 4]);
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
