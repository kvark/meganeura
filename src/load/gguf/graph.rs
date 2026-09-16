//! The graph a GGUF file implies.
//!
//! GGUF stores no operations, only a description — an architecture name and
//! a set of dimensions — plus tensors named by a convention
//! (`blk.0.attn_q.weight`). This module turns that description into a
//! [`Graph`], declaring each parameter under the file's own tensor name and
//! in the dtype the file already stores it in, so [`super::weights`] can
//! hand the bytes over without converting them.
//!
//! # One graph for prefill and decode
//!
//! The graph takes a *block* of `block_size` token slots, of which the first
//! `valid` are real, starting at absolute position `position`. Three inputs
//! carry that:
//!
//! | input | type | meaning |
//! |-------|------|---------|
//! | `token_ids` | `[block_size]` u32 | the block's tokens, unused slots arbitrary |
//! | `position`  | `[1]` u32 | absolute position of row 0 |
//! | `valid`     | `[1]` u32 | how many rows are real |
//!
//! Row `i` is at absolute position `position + i`, which is what
//! [`Graph::rope_dynamic_offset`] and [`Graph::cached_block_attention`] both
//! assume, so one shape serves a whole prompt (`valid = prompt_len`) and a
//! single decode step (`valid = 1`) alike. The K/V caches live in the
//! session as parameters and [`Graph::cache_write_prefix`] updates them in
//! place, so nothing round-trips through the host between steps.
//!
//! Only the last valid row can predict the next token, so
//! [`Graph::prefix_last`] narrows the residual to one row *before* the output
//! head. The head is the widest matmul in the model — `hidden × vocab` — and
//! this way it runs once per step rather than `block_size` times.
//!
//! # What is refused
//!
//! An architecture whose graph cannot be expressed exactly is rejected by
//! name rather than approximated, because a decoder that is subtly wrong
//! still emits fluent text and gives nothing to debug against. Two such
//! limits exist today, both in the ops rather than here:
//!
//! - **Partial RoPE** (`rope.dimension_count < head_dim`, as Phi2 uses)
//!   would need to rotate part of each head and leave the rest, which needs
//!   a strided split the IR has no op for.
//! - **Attention logit softcapping** (Gemma2's `attn_logit_softcapping`) has
//!   no parameter on the cached attention ops. *Final* logit softcapping is
//!   supported, being expressible after the head.

use crate::graph::{DType, Graph, NodeId};

use super::arch::ModelConfig;
use super::{GgufError, GgufModel};

/// The graph's interface: what to read out, and what the caller must set.
///
/// Returned by [`build`]. The caller marks [`Self::outputs`] on the graph;
/// the individual node lists are kept so a caller that wants the caches at
/// different output indices can arrange its own.
#[derive(Clone, Debug)]
pub struct ModelGraph {
    /// Logits for the last valid row, `[1, vocab_size]`.
    pub logits: NodeId,
    /// Per-layer key caches after this block's write, `[max_seq_len, kv_dim]`.
    pub k_caches: Vec<NodeId>,
    /// Per-layer value caches after this block's write.
    pub v_caches: Vec<NodeId>,
    /// Token slots per step.
    pub block_size: usize,
    /// Cache depth — the longest sequence this graph can attend over.
    pub max_seq_len: usize,
}

impl ModelGraph {
    /// Graph outputs in the order [`super::weights`] and
    /// [`super::generate`] expect: logits, then every key cache, then every
    /// value cache.
    pub fn outputs(&self) -> Vec<NodeId> {
        let mut out = Vec::with_capacity(1 + self.k_caches.len() + self.v_caches.len());
        out.push(self.logits);
        out.extend_from_slice(&self.k_caches);
        out.extend_from_slice(&self.v_caches);
        out
    }

    /// The parameter name of layer `layer`'s key cache.
    pub fn k_cache_name(layer: usize) -> String {
        format!("cache.{layer}.k")
    }

    /// The parameter name of layer `layer`'s value cache.
    pub fn v_cache_name(layer: usize) -> String {
        format!("cache.{layer}.v")
    }
}

/// Build the decoder `config` describes into `g`.
///
/// `block_size` is how many token slots one step consumes and `max_seq_len`
/// how far back attention can reach; the latter bounds the KV cache and so
/// the session's memory.
///
/// Parameters are declared under GGUF's own tensor names and in the dtype
/// `model` stores each one in, which is why the file is a parameter here:
/// the graph is a reading of it, not a template it is poured into.
pub fn build(
    g: &mut Graph,
    model: &GgufModel,
    config: &ModelConfig,
    block_size: usize,
    max_seq_len: usize,
) -> Result<ModelGraph, GgufError> {
    check_expressible(config)?;
    if block_size == 0 {
        return Err(GgufError::BadMetadata("block_size must be > 0".into()));
    }
    if max_seq_len == 0 {
        return Err(GgufError::BadMetadata("max_seq_len must be > 0".into()));
    }

    let arch = config.architecture;
    let hidden = config.hidden_size;
    let kv_dim = config.kv_dim();
    let q_dim = config.q_dim();
    let head_dim = config.head_dim;
    let eps = config.norm_eps;
    let theta = config.rope_theta;

    let token_ids = g.input_u32("token_ids", &[block_size]);
    let position = g.input_u32("position", &[1]);
    let valid = g.input_u32("valid", &[1]);

    // The embedding table is declared f16 whatever the file stores, because
    // the gather has no block-quantized variant — and because a tied output
    // head reads this same tensor through `matmul_bt`, which block formats
    // cannot serve either (their blocks would run along N there, not K).
    // f16 satisfies both and halves the bytes read per gathered row.
    let embed = g.parameter_f16(TOKEN_EMBD, &[config.vocab_size, hidden]);
    let mut x = g.embedding_f16(token_ids, embed);

    // Gemma scales the embedding by sqrt(n_embd) on the way in. It is a
    // constant factor, so it folds into the graph rather than the weights —
    // the table is shared with the output head, which must not see it.
    if arch.scales_embeddings() {
        x = g.scale(x, (hidden as f32).sqrt());
    }

    let mut k_caches = Vec::with_capacity(config.num_layers);
    let mut v_caches = Vec::with_capacity(config.num_layers);

    for layer in 0..config.num_layers {
        let p = format!("blk.{layer}");

        let normed = norm(g, model, config, &format!("{p}.attn_norm"), x, eps)?;

        // Phi2 feeds one normed input to attention and feed-forward
        // together; everything else re-norms between them.
        let attn_in = normed;

        let q = projection(g, model, &format!("{p}.attn_q.weight"), &[hidden, q_dim])?;
        let k = projection(g, model, &format!("{p}.attn_k.weight"), &[hidden, kv_dim])?;
        let v = projection(g, model, &format!("{p}.attn_v.weight"), &[hidden, kv_dim])?;

        let mut q = g.matmul(attn_in, q);
        let mut k = g.matmul(attn_in, k);
        let mut v = g.matmul(attn_in, v);

        if arch.qkv_bias() {
            q = bias(g, model, &format!("{p}.attn_q.bias"), q, q_dim)?;
            k = bias(g, model, &format!("{p}.attn_k.bias"), k, kv_dim)?;
            v = bias(g, model, &format!("{p}.attn_v.bias"), v, kv_dim)?;
        }

        // Qwen3 norms each head of Q and K before the rotation. The weight
        // is one head wide and shared across heads, which is exactly what
        // rms_norm over a [rows*heads, head_dim] view does.
        if arch.qk_norm() {
            q = per_head_norm(
                g,
                model,
                &format!("{p}.attn_q_norm.weight"),
                q,
                block_size,
                config.num_heads,
                head_dim,
                eps,
            )?;
            k = per_head_norm(
                g,
                model,
                &format!("{p}.attn_k_norm.weight"),
                k,
                block_size,
                config.num_kv_heads,
                head_dim,
                eps,
            )?;
        }

        // Row i sits at absolute position `position + i`, which is the
        // offset form's own rule, so one call covers a whole prompt block
        // and a single decode step alike.
        let q = g.rope_dynamic_offset(q, theta, position, head_dim);
        let k = g.rope_dynamic_offset(k, theta, position, head_dim);

        let k_cache = g.parameter(&ModelGraph::k_cache_name(layer), &[max_seq_len, kv_dim]);
        let v_cache = g.parameter(&ModelGraph::v_cache_name(layer), &[max_seq_len, kv_dim]);

        // Only the valid rows enter the cache, and attention must read the
        // *written* caches: with no data dependency the scheduler may order
        // this block's write after the attention that should see it.
        let k_written = g.cache_write_prefix(k, k_cache, position, valid);
        let v_written = g.cache_write_prefix(v, v_cache, position, valid);

        // `window_size = 0` is the op's spelling of "attend to the whole
        // prefix"; Gemma2 alternates, so the layer index decides.
        let window = if config.layer_is_windowed(layer) {
            config.sliding_window.unwrap_or(0) as u32
        } else {
            0
        };
        let attn = g.cached_block_attention(
            q,
            k_written,
            v_written,
            position,
            valid,
            config.num_heads,
            config.num_kv_heads,
            head_dim,
            window,
        );

        k_caches.push(k_written);
        v_caches.push(v_written);

        let wo = projection(
            g,
            model,
            &format!("{p}.attn_output.weight"),
            &[q_dim, hidden],
        )?;
        let mut attn_out = g.matmul(attn, wo);
        if arch.qkv_bias() {
            attn_out = bias(g, model, &format!("{p}.attn_output.bias"), attn_out, hidden)?;
        }
        // Gemma2 norms each block's output before it rejoins the residual.
        if arch.post_block_norms() {
            attn_out = norm(
                g,
                model,
                config,
                &format!("{p}.post_attention_norm"),
                attn_out,
                eps,
            )?;
        }

        if arch.parallel_residual() {
            // Attention and feed-forward read the same normed input and
            // both add into the residual, rather than composing.
            let ffn_out = feed_forward(g, model, config, &p, attn_in, block_size)?;
            let both = g.add(attn_out, ffn_out);
            x = g.add(x, both);
        } else {
            x = g.add(x, attn_out);
            let ffn_in = norm(g, model, config, &format!("{p}.ffn_norm"), x, eps)?;
            let mut ffn_out = feed_forward(g, model, config, &p, ffn_in, block_size)?;
            if arch.post_block_norms() {
                ffn_out = norm(
                    g,
                    model,
                    config,
                    &format!("{p}.post_ffw_norm"),
                    ffn_out,
                    eps,
                )?;
            }
            x = g.add(x, ffn_out);
        }
    }

    let x = norm(g, model, config, OUTPUT_NORM, x, eps)?;

    // Narrow to the one row that can predict the next token before the
    // output head, the widest matmul in the model, rather than after.
    let last = g.prefix_last(x, valid);

    let mut logits = if config.tie_word_embeddings {
        // The table is [vocab, hidden] and `matmul_bt` wants B as [N, K],
        // which is the same orientation — no transposed copy is needed.
        g.matmul_bt(last, embed)
    } else {
        let head = projection(g, model, OUTPUT, &[hidden, config.vocab_size])?;
        g.matmul(last, head)
    };

    if let Some(cap) = config.final_logit_softcap {
        logits = softcap(g, logits, cap);
    }

    Ok(ModelGraph {
        logits,
        k_caches,
        v_caches,
        block_size,
        max_seq_len,
    })
}

/// The embedding table's tensor name, which is also the tied output head.
pub const TOKEN_EMBD: &str = "token_embd.weight";
/// The final norm before the output head.
pub const OUTPUT_NORM: &str = "output_norm";
/// The output head, absent when the model ties it to [`TOKEN_EMBD`].
pub const OUTPUT: &str = "output.weight";

/// Reject what the ops cannot express, naming the reason.
fn check_expressible(config: &ModelConfig) -> Result<(), GgufError> {
    if config.rope_dim != config.head_dim {
        return Err(GgufError::UnsupportedArchitecture(format!(
            "{}: rope.dimension_count {} is narrower than the {}-wide head, and \
             rotating part of a head needs a strided split with no op behind it",
            config.architecture, config.rope_dim, config.head_dim
        )));
    }
    if !config.head_dim.is_multiple_of(2) {
        return Err(GgufError::UnsupportedArchitecture(format!(
            "{}: RoPE needs an even head width, got {}",
            config.architecture, config.head_dim
        )));
    }
    if config.head_dim > 512 {
        return Err(GgufError::UnsupportedArchitecture(format!(
            "{}: cached attention supports heads up to 512 wide, got {}",
            config.architecture, config.head_dim
        )));
    }
    if config.attn_logit_softcap.is_some() {
        return Err(GgufError::UnsupportedArchitecture(format!(
            "{}: attn_logit_softcapping has no parameter on the cached attention \
             ops, and dropping it would change every attention distribution",
            config.architecture
        )));
    }
    Ok(())
}

/// `cap * tanh(x / cap)` — a smooth ceiling on the logits.
fn softcap(g: &mut Graph, x: NodeId, cap: f32) -> NodeId {
    let scaled = g.scale(x, 1.0 / cap);
    let squashed = g.tanh(scaled);
    g.scale(squashed, cap)
}

/// The block's feed-forward, gated or not.
fn feed_forward(
    g: &mut Graph,
    model: &GgufModel,
    config: &ModelConfig,
    prefix: &str,
    input: NodeId,
    _block_size: usize,
) -> Result<NodeId, GgufError> {
    let arch = config.architecture;
    let hidden = config.hidden_size;
    let ffn = config.intermediate_size;

    let hidden_act = if arch.gated_ffn() {
        let w_gate = projection(
            g,
            model,
            &format!("{prefix}.ffn_gate.weight"),
            &[hidden, ffn],
        )?;
        let w_up = projection(g, model, &format!("{prefix}.ffn_up.weight"), &[hidden, ffn])?;
        let gate = g.matmul(input, w_gate);
        let up = g.matmul(input, w_up);
        if arch.scales_embeddings() {
            // Gemma gates with GELU where llama gates with SiLU. The
            // multiply is the same; only the activation differs, so this
            // cannot go through the fused `swiglu`.
            let activated = g.gelu(gate);
            g.mul(activated, up)
        } else {
            g.swiglu(gate, up)
        }
    } else {
        let w_up = projection(g, model, &format!("{prefix}.ffn_up.weight"), &[hidden, ffn])?;
        let mut up = g.matmul(input, w_up);
        up = bias(g, model, &format!("{prefix}.ffn_up.bias"), up, ffn)?;
        g.gelu(up)
    };

    let w_down = projection(
        g,
        model,
        &format!("{prefix}.ffn_down.weight"),
        &[ffn, hidden],
    )?;
    let mut out = g.matmul(hidden_act, w_down);
    if !arch.gated_ffn() {
        out = bias(g, model, &format!("{prefix}.ffn_down.bias"), out, hidden)?;
    }
    Ok(out)
}

/// Whichever normalization the architecture uses, under `name.weight`
/// (and `name.bias` for LayerNorm).
fn norm(
    g: &mut Graph,
    model: &GgufModel,
    config: &ModelConfig,
    name: &str,
    x: NodeId,
    eps: f32,
) -> Result<NodeId, GgufError> {
    let hidden = g.node(x).ty.shape[1];
    let weight_name = format!("{name}.weight");
    require_tensor(model, &weight_name)?;
    let w = g.parameter(&weight_name, &[hidden]);
    if config.architecture.uses_layer_norm() {
        let bias_name = format!("{name}.bias");
        require_tensor(model, &bias_name)?;
        let b = g.parameter(&bias_name, &[hidden]);
        Ok(g.layer_norm(x, w, b, eps))
    } else {
        Ok(g.rms_norm(x, w, eps))
    }
}

/// RMSNorm applied within each head rather than across the row.
///
/// Qwen3's Q/K norms are one head wide and shared by every head, so the
/// rows are reshaped to `[rows * heads, head_dim]`, normed, and reshaped
/// back — the norm's own per-row behaviour then *is* the per-head one.
#[allow(clippy::too_many_arguments)]
fn per_head_norm(
    g: &mut Graph,
    model: &GgufModel,
    name: &str,
    x: NodeId,
    rows: usize,
    heads: u32,
    head_dim: u32,
    eps: f32,
) -> Result<NodeId, GgufError> {
    require_tensor(model, name)?;
    let w = g.parameter(name, &[head_dim as usize]);
    let wide = g.reshape(x, &[rows * heads as usize, head_dim as usize]);
    let normed = g.rms_norm(wide, w, eps);
    Ok(g.reshape(normed, &[rows, heads as usize * head_dim as usize]))
}

/// Add a bias vector, which GGUF stores as a plain f32 row.
fn bias(
    g: &mut Graph,
    model: &GgufModel,
    name: &str,
    x: NodeId,
    width: usize,
) -> Result<NodeId, GgufError> {
    require_tensor(model, name)?;
    let b = g.parameter(name, &[width]);
    Ok(g.bias_add(x, b))
}

/// Declare a projection weight in the dtype the file stores it in.
///
/// GGUF names dimensions fastest-first, so a `[K, N]` tensor there is a
/// `[K, N]` Meganeura parameter — the orientations agree for weights, and
/// [`super::GgufTensor::to_packed`] performs the transpose that the packed
/// layouts imply.
fn projection(
    g: &mut Graph,
    model: &GgufModel,
    name: &str,
    shape: &[usize; 2],
) -> Result<NodeId, GgufError> {
    let tensor = require_tensor(model, name)?;
    if tensor.dims.len() != 2 || tensor.dims[0] != shape[0] || tensor.dims[1] != shape[1] {
        return Err(GgufError::MissingTensor(format!(
            "`{name}` is {:?}, but the architecture makes it {shape:?}",
            tensor.dims
        )));
    }
    let dtype = match tensor.ggml_type {
        super::GgmlType::F32 => DType::F32,
        super::GgmlType::F16 => DType::F16,
        // Anything block-packed keeps the file's own encoding: going
        // through f32 would requantize on the way back in and roughly
        // double the error the file already carries.
        _ => tensor.packed_dtype()?,
    };
    Ok(parameter_of(g, name, shape, dtype))
}

/// Declare a parameter of a given dtype, choosing the constructor that
/// carries the block-size assertions for it.
fn parameter_of(g: &mut Graph, name: &str, shape: &[usize; 2], dtype: DType) -> NodeId {
    match dtype {
        DType::F32 => g.parameter(name, shape),
        DType::F16 => g.parameter_f16(name, shape),
        DType::Q40 => g.parameter_q40(name, shape),
        DType::Q4_0 => g.parameter_q4(name, shape),
        DType::Q8_0 => g.parameter_q8(name, shape),
        DType::Q4K => g.parameter_q4k(name, shape),
        DType::Q6K => g.parameter_q6k(name, shape),
        DType::Q5K => g.parameter_q5k(name, shape),
        DType::Q3K => g.parameter_q3k(name, shape),
        // `projection` only ever produces the dtypes above; a new one
        // reaching here means a missing arm rather than a bad file.
        other => unreachable!("no parameter constructor for {other:?}"),
    }
}

fn require_tensor<'a>(
    model: &'a GgufModel,
    name: &str,
) -> Result<&'a super::GgufTensor, GgufError> {
    model
        .tensors
        .get(name)
        .ok_or_else(|| GgufError::MissingTensor(format!("the file has no `{name}`")))
}

/// Every parameter name [`build`] declares, in no particular order.
///
/// Useful for checking a file against an architecture before compiling
/// anything, and for reporting which tensors a load will consume.
pub fn parameter_names(config: &ModelConfig) -> Vec<String> {
    let arch = config.architecture;
    let mut names = vec![TOKEN_EMBD.to_string(), format!("{OUTPUT_NORM}.weight")];
    if arch.uses_layer_norm() {
        names.push(format!("{OUTPUT_NORM}.bias"));
    }
    if !config.tie_word_embeddings {
        names.push(OUTPUT.to_string());
    }
    for layer in 0..config.num_layers {
        let p = format!("blk.{layer}");
        names.push(format!("{p}.attn_norm.weight"));
        if arch.uses_layer_norm() {
            names.push(format!("{p}.attn_norm.bias"));
        }
        for part in ["attn_q", "attn_k", "attn_v", "attn_output"] {
            names.push(format!("{p}.{part}.weight"));
            if arch.qkv_bias() {
                names.push(format!("{p}.{part}.bias"));
            }
        }
        if arch.qk_norm() {
            names.push(format!("{p}.attn_q_norm.weight"));
            names.push(format!("{p}.attn_k_norm.weight"));
        }
        if arch.post_block_norms() {
            names.push(format!("{p}.post_attention_norm.weight"));
            names.push(format!("{p}.post_ffw_norm.weight"));
        }
        if !arch.parallel_residual() {
            names.push(format!("{p}.ffn_norm.weight"));
        }
        if arch.gated_ffn() {
            names.push(format!("{p}.ffn_gate.weight"));
            names.push(format!("{p}.ffn_up.weight"));
        } else {
            names.push(format!("{p}.ffn_up.weight"));
            names.push(format!("{p}.ffn_up.bias"));
            names.push(format!("{p}.ffn_down.bias"));
        }
        names.push(format!("{p}.ffn_down.weight"));
    }
    names
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load::gguf::fixture;
    use crate::load::gguf::{GgmlType, GgufTensor, GgufValue};
    use std::collections::HashMap;

    /// Build the graph for a fixture model, returning the graph and its
    /// interface.
    fn build_fixture(arch: &str) -> (Graph, ModelGraph, ModelConfig) {
        let model = fixture::model(arch);
        let config = ModelConfig::from_gguf(&model).unwrap();
        let mut g = Graph::new();
        let built = build(&mut g, &model, &config, 4, 16).unwrap();
        g.set_outputs(built.outputs());
        (g, built, config)
    }

    /// Every `Op::Parameter` the graph declares, with its declared shape.
    fn declared_parameters(g: &Graph) -> HashMap<String, Vec<usize>> {
        g.nodes()
            .iter()
            .filter_map(|node| match node.op {
                crate::graph::Op::Parameter { ref name } => {
                    Some((name.clone(), node.ty.shape.clone()))
                }
                _ => None,
            })
            .collect()
    }

    #[test]
    fn the_llama_graph_declares_exactly_the_files_tensors() {
        let (g, _, config) = build_fixture("llama");
        let declared = declared_parameters(&g);
        let mut expected: Vec<String> = parameter_names(&config);
        // The caches are parameters too, but they are state rather than
        // weights and so are not in the file.
        for layer in 0..config.num_layers {
            expected.push(ModelGraph::k_cache_name(layer));
            expected.push(ModelGraph::v_cache_name(layer));
        }
        expected.sort();
        let mut actual: Vec<String> = declared.keys().cloned().collect();
        actual.sort();
        assert_eq!(actual, expected);
    }

    #[test]
    fn projection_shapes_follow_the_files_dimensions() {
        let (g, _, config) = build_fixture("llama");
        let declared = declared_parameters(&g);
        assert_eq!(
            declared["blk.0.attn_q.weight"],
            vec![config.hidden_size, config.q_dim()]
        );
        assert_eq!(
            declared["blk.0.attn_k.weight"],
            vec![config.hidden_size, config.kv_dim()]
        );
        assert_eq!(
            declared["blk.0.ffn_down.weight"],
            vec![config.intermediate_size, config.hidden_size]
        );
        assert_eq!(
            declared[TOKEN_EMBD],
            vec![config.vocab_size, config.hidden_size],
            "the table is a list of rows, not a [K, N] weight"
        );
    }

    #[test]
    fn the_embedding_table_is_f16_whatever_the_file_stores() {
        // Q4_K would be the natural reading of a quantized table, but the
        // gather has no block variant and a tied head could not read it
        // through matmul_bt either.
        let model = fixture::model_with("llama", |m| {
            let hidden = 64;
            let vocab = 32;
            m.tensors.insert(
                TOKEN_EMBD.to_string(),
                GgufTensor::new(
                    vec![hidden, vocab],
                    GgmlType::Q4K,
                    vec![0; (hidden * vocab / 256) * 144],
                ),
            );
        });
        let config = ModelConfig::from_gguf(&model).unwrap();
        let mut g = Graph::new();
        build(&mut g, &model, &config, 4, 16).unwrap();
        let table = g
            .nodes()
            .iter()
            .find(
                |n| matches!(n.op, crate::graph::Op::Parameter { ref name } if name == TOKEN_EMBD),
            )
            .unwrap();
        assert_eq!(table.ty.dtype, crate::graph::DType::F16);
    }

    #[test]
    fn a_quantized_projection_keeps_the_files_own_encoding() {
        let model = fixture::model_with("llama", |m| {
            // Q4_0 blocks 32 elements, which divides the fixture's 64-wide
            // reduction; Q4_K's 256-element superblock would not.
            m.tensors.insert(
                "blk.0.attn_q.weight".to_string(),
                GgufTensor::new(vec![64, 64], GgmlType::Q4_0, vec![0; (64 * 64 / 32) * 18]),
            );
        });
        let config = ModelConfig::from_gguf(&model).unwrap();
        let mut g = Graph::new();
        build(&mut g, &model, &config, 4, 16).unwrap();
        let w = g
            .nodes()
            .iter()
            .find(|n| {
                matches!(n.op, crate::graph::Op::Parameter { ref name }
                    if name == "blk.0.attn_q.weight")
            })
            .unwrap();
        assert_eq!(
            w.ty.dtype,
            crate::graph::DType::Q40,
            "going through f32 would requantize and double the error"
        );
    }

    #[test]
    fn the_logits_are_one_row_wide_however_big_the_block() {
        let model = fixture::model("llama");
        let config = ModelConfig::from_gguf(&model).unwrap();
        for block in [1, 4, 32] {
            let mut g = Graph::new();
            let built = build(&mut g, &model, &config, block, 64).unwrap();
            assert_eq!(
                g.node(built.logits).ty.shape,
                vec![1, config.vocab_size],
                "block {block} should still predict one token"
            );
        }
    }

    #[test]
    fn caches_are_sized_by_max_seq_len_not_the_block() {
        let model = fixture::model("llama");
        let config = ModelConfig::from_gguf(&model).unwrap();
        let mut g = Graph::new();
        let built = build(&mut g, &model, &config, 4, 128).unwrap();
        let declared = declared_parameters(&g);
        assert_eq!(
            declared[&ModelGraph::k_cache_name(0)],
            vec![128, config.kv_dim()]
        );
        assert_eq!(built.k_caches.len(), config.num_layers);
        assert_eq!(built.v_caches.len(), config.num_layers);
    }

    #[test]
    fn outputs_are_logits_then_keys_then_values() {
        let (_, built, config) = build_fixture("llama");
        let outputs = built.outputs();
        assert_eq!(outputs.len(), 1 + 2 * config.num_layers);
        assert_eq!(outputs[0], built.logits);
        assert_eq!(&outputs[1..1 + config.num_layers], &built.k_caches[..]);
        assert_eq!(&outputs[1 + config.num_layers..], &built.v_caches[..]);
    }

    #[test]
    fn a_tied_model_declares_no_output_head() {
        let (g, _, config) = build_fixture("llama");
        assert!(
            config.tie_word_embeddings,
            "the fixture omits output.weight"
        );
        assert!(!declared_parameters(&g).contains_key(OUTPUT));
    }

    #[test]
    fn an_untied_model_declares_its_own_head() {
        let model = fixture::model_with("llama", |m| {
            m.tensors
                .insert(OUTPUT.to_string(), fixture::f32_tensor(vec![64, 32]));
        });
        let config = ModelConfig::from_gguf(&model).unwrap();
        assert!(!config.tie_word_embeddings);
        let mut g = Graph::new();
        build(&mut g, &model, &config, 4, 16).unwrap();
        assert_eq!(
            declared_parameters(&g)[OUTPUT],
            vec![config.hidden_size, config.vocab_size]
        );
    }

    #[test]
    fn qwen2_adds_qkv_biases_and_qwen3_replaces_them_with_head_norms() {
        let (qwen2, _, _) = build_fixture("qwen2");
        let d2 = declared_parameters(&qwen2);
        assert!(d2.contains_key("blk.0.attn_q.bias"));
        assert!(!d2.contains_key("blk.0.attn_q_norm.weight"));

        let (qwen3, _, config) = build_fixture("qwen3");
        let d3 = declared_parameters(&qwen3);
        assert!(!d3.contains_key("blk.0.attn_q.bias"));
        assert_eq!(
            d3["blk.0.attn_q_norm.weight"],
            vec![config.head_dim as usize],
            "the Q/K norm is one head wide and shared across heads"
        );
    }

    #[test]
    fn gemma2_adds_a_second_pair_of_norms_per_block() {
        let (gemma, _, _) = build_fixture("gemma");
        let dg = declared_parameters(&gemma);
        assert!(!dg.contains_key("blk.0.post_attention_norm.weight"));

        let (gemma2, _, _) = build_fixture("gemma2");
        let d2 = declared_parameters(&gemma2);
        assert!(d2.contains_key("blk.0.post_attention_norm.weight"));
        assert!(d2.contains_key("blk.0.post_ffw_norm.weight"));
    }

    #[test]
    fn a_missing_tensor_names_itself_rather_than_panicking() {
        let model = fixture::model_with("llama", |m| {
            m.tensors.remove("blk.1.ffn_down.weight");
        });
        let config = ModelConfig::from_gguf(&model).unwrap();
        let mut g = Graph::new();
        let err = build(&mut g, &model, &config, 4, 16).unwrap_err();
        assert!(
            matches!(&err, GgufError::MissingTensor(m) if m.contains("blk.1.ffn_down.weight")),
            "{err:?}"
        );
    }

    #[test]
    fn a_tensor_of_the_wrong_shape_is_reported_against_the_architecture() {
        let model = fixture::model_with("llama", |m| {
            m.tensors.insert(
                "blk.0.attn_q.weight".to_string(),
                fixture::f32_tensor(vec![64, 63]),
            );
        });
        let config = ModelConfig::from_gguf(&model).unwrap();
        let mut g = Graph::new();
        let err = build(&mut g, &model, &config, 4, 16).unwrap_err();
        assert!(
            matches!(&err, GgufError::MissingTensor(m)
                if m.contains("attn_q.weight") && m.contains("[64, 63]")),
            "{err:?}"
        );
    }

    #[test]
    fn a_partial_rope_is_refused_rather_than_rotated_whole() {
        let model = fixture::model_with("llama", |m| {
            fixture::set_arch_key(m, "rope.dimension_count", GgufValue::U32(8));
        });
        let config = ModelConfig::from_gguf(&model).unwrap();
        let mut g = Graph::new();
        let err = build(&mut g, &model, &config, 4, 16).unwrap_err();
        assert!(
            matches!(&err, GgufError::UnsupportedArchitecture(m)
                if m.contains("rope.dimension_count")),
            "{err:?}"
        );
    }

    #[test]
    fn attention_softcapping_is_refused_rather_than_dropped() {
        let model = fixture::model_with("gemma2", |m| {
            fixture::set_arch_key(m, "attn_logit_softcapping", GgufValue::F32(50.0));
        });
        let config = ModelConfig::from_gguf(&model).unwrap();
        let mut g = Graph::new();
        let err = build(&mut g, &model, &config, 4, 16).unwrap_err();
        assert!(
            matches!(&err, GgufError::UnsupportedArchitecture(m)
                if m.contains("attn_logit_softcapping")),
            "{err:?}"
        );
    }

    #[test]
    fn final_softcapping_is_expressible_and_so_is_applied() {
        let plain = fixture::model("gemma2");
        let config_plain = ModelConfig::from_gguf(&plain).unwrap();
        let mut g0 = Graph::new();
        build(&mut g0, &plain, &config_plain, 4, 16).unwrap();

        let capped = fixture::model_with("gemma2", |m| {
            fixture::set_arch_key(m, "final_logit_softcapping", GgufValue::F32(30.0));
        });
        let config = ModelConfig::from_gguf(&capped).unwrap();
        assert_eq!(config.final_logit_softcap, Some(30.0));
        let mut g1 = Graph::new();
        build(&mut g1, &capped, &config, 4, 16).unwrap();

        let tanhs = |g: &Graph| {
            g.nodes()
                .iter()
                .filter(|n| matches!(n.op, crate::graph::Op::Tanh))
                .count()
        };
        assert_eq!(tanhs(&g0), 0);
        assert_eq!(tanhs(&g1), 1, "cap * tanh(x / cap) after the head");
    }

    #[test]
    fn a_sliding_window_reaches_the_attention_op() {
        let model = fixture::model_with("gemma2", |m| {
            fixture::set_arch_key(m, "attention.sliding_window", GgufValue::U32(8));
        });
        let config = ModelConfig::from_gguf(&model).unwrap();
        let mut g = Graph::new();
        build(&mut g, &model, &config, 4, 16).unwrap();
        let windows: Vec<u32> = g
            .nodes()
            .iter()
            .filter_map(|n| match n.op {
                crate::graph::Op::CachedBlockAttention { window_size, .. } => Some(window_size),
                _ => None,
            })
            .collect();
        // Gemma2 alternates, starting windowed.
        assert_eq!(windows, vec![8, 0], "layer 0 windowed, layer 1 full");
    }

    #[test]
    fn a_model_with_no_window_attends_fully_in_every_layer() {
        let (g, _, _) = build_fixture("llama");
        let windows: Vec<u32> = g
            .nodes()
            .iter()
            .filter_map(|n| match n.op {
                crate::graph::Op::CachedBlockAttention { window_size, .. } => Some(window_size),
                _ => None,
            })
            .collect();
        assert_eq!(windows, vec![0, 0]);
    }

    #[test]
    fn gemma_scales_its_embeddings_and_llama_does_not() {
        let scales = |arch: &str| {
            let (g, _, _) = build_fixture(arch);
            g.nodes()
                .iter()
                .filter(|n| matches!(n.op, crate::graph::Op::Scale { .. }))
                .count()
        };
        assert_eq!(scales("llama"), 0);
        assert!(scales("gemma") >= 1);
    }

    #[test]
    fn gemma_gates_with_gelu_where_llama_uses_the_fused_swiglu() {
        let count = |arch: &str, f: fn(&crate::graph::Op) -> bool| {
            let (g, _, _) = build_fixture(arch);
            g.nodes().iter().filter(|n| f(&n.op)).count()
        };
        assert_eq!(
            count("llama", |op| matches!(op, crate::graph::Op::SwiGLU)),
            2
        );
        assert_eq!(count("llama", |op| matches!(op, crate::graph::Op::Gelu)), 0);
        assert_eq!(
            count("gemma", |op| matches!(op, crate::graph::Op::SwiGLU)),
            0
        );
        assert_eq!(count("gemma", |op| matches!(op, crate::graph::Op::Gelu)), 2);
    }

    #[test]
    fn a_zero_block_or_cache_is_rejected() {
        let model = fixture::model("llama");
        let config = ModelConfig::from_gguf(&model).unwrap();
        let mut g = Graph::new();
        assert!(build(&mut g, &model, &config, 0, 16).is_err());
        assert!(build(&mut g, &model, &config, 4, 0).is_err());
    }
}
