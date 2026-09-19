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
//! # Where a parameter's bytes come from
//!
//! A parameter is usually a tensor of its own, but not always: Phi packs Q,
//! K and V into one `attn_qkv.weight`, and Phi3 packs the feed-forward gate
//! and up into one double-width `ffn_up.weight`. `source_of` resolves a
//! parameter name to the tensor and row range backing it, and both this
//! module and [`super::weights`] go through it, so the shape a parameter is
//! declared with and the bytes it is filled from cannot disagree.
//!
//! Biases are *optional*. GGUF families are inconsistent about them in ways
//! not worth a variant each — Qwen2 biases Q, K and V but leaves the
//! attention output unbiased — and an absent bias is a zero bias, which is
//! the graph without the add.
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
    check_expressible(model, config)?;
    if block_size == 0 {
        return Err(GgufError::BadMetadata("block_size must be > 0".into()));
    }
    if max_seq_len == 0 {
        return Err(GgufError::BadMetadata("max_seq_len must be > 0".into()));
    }

    let arch = config.architecture;
    let hidden = config.hidden_size;
    let eps = config.norm_eps;

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

    // Gemma4's per-layer embeddings: every block projects the residual
    // through one wide projection into all layers' PLE rows, normed per
    // row, and adds the token's gathered per-layer embedding — the latter
    // read on the host, since the table is block-quantized and the gather
    // has no quantized variant.
    // Token-major rows: row (t, l) is token t's layer-l PLE row, which is
    // what a one-hot matmul over the row axis selects from later.
    let ple_rows = config.num_layers * block_size;
    let ple = if arch.uses_per_layer_embeddings() {
        let ple_size = config.per_layer_embed_size;
        if ple_size == 0 {
            return Err(GgufError::BadMetadata(
                "the architecture carries per-layer embeddings but embedding_length_per_layer_input is absent or zero"
                    .to_string(),
            ));
        }
        let table = require_tensor(model, "per_layer_token_embd.weight")?;
        let expected_ple_width = config.num_layers * ple_size;
        if table.dims.as_slice() != [expected_ple_width, config.vocab_size] {
            return Err(GgufError::BadShape(format!(
                "`per_layer_token_embd.weight` has shape {:?}, expected GGUF dimensions [{expected_ple_width}, {}]",
                table.dims, config.vocab_size
            )));
        }
        let w_proj = projection(
            g,
            model,
            config,
            PER_LAYER_PROJ,
            &[hidden, config.num_layers * ple_size],
        )?;
        let proj_norm_name = "per_layer_proj_norm.weight";
        check_vector(
            proj_norm_name,
            require_tensor(model, proj_norm_name)?,
            ple_size,
        )?;
        let proj_norm = g.parameter(proj_norm_name, &[ple_size]);
        let layered = project(g, x, w_proj);
        let layered = g.scale(layered, 1.0 / (hidden as f32).sqrt());
        let layered = g.reshape(layered, &[ple_rows, ple_size]);
        let layered = g.rms_norm(layered, proj_norm, eps);
        let ple_in = g.input("ple", &[ple_rows, ple_size]);
        let layered = g.add(layered, ple_in);
        Some(g.scale(layered, 1.0 / 2.0f32.sqrt()))
    } else {
        None
    };
    let rope_factors = if arch.uses_rope_factors() {
        Some(g.parameter("rope_freqs.weight", &[(config.rope_dim / 2) as usize]))
    } else {
        None
    };

    let mut k_caches = Vec::with_capacity(config.num_layers);
    let mut v_caches = Vec::with_capacity(config.num_layers);
    let mut k_updated: Vec<Option<NodeId>> = vec![None; config.num_layers];
    let mut v_updated: Vec<Option<NodeId>> = vec![None; config.num_layers];

    for layer in 0..config.num_layers {
        let p = format!("blk.{layer}");
        let head_dim = config.head_dim_at(layer);
        let kv_dim = config.kv_dim_at(layer);
        let q_dim = config.num_heads as usize * head_dim as usize;

        let normed = norm(g, model, config, &format!("{p}.attn_norm"), x, eps)?;

        // Phi2 feeds one normed input to attention and feed-forward
        // together; everything else re-norms between them.
        let attn_in = normed;

        let q = projection(
            g,
            model,
            config,
            &format!("{p}.attn_q.weight"),
            &[hidden, q_dim],
        )?;
        let mut q = project(g, attn_in, q);

        // Optional rather than gated on the architecture: Qwen2 biases Q,
        // K and V but not the attention output, and requiring all four
        // would fail on every real Qwen2 file. K/V biases apply where the
        // projections exist; a shared-KV layer has none of either.
        q = optional_bias(g, model, &format!("{p}.attn_q.bias"), q, q_dim)?;

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
        }

        // Row i sits at absolute position `position + i`, which is the
        // offset form's own rule, so one call covers a whole prompt block
        // and a single decode step alike.
        // Gemma3 rotates its local layers with a smaller base than its
        // global ones, so the base is read per layer rather than once.
        let layer_theta = config.layer_rope_theta(layer);
        let freqs = if config.layer_is_windowed(layer) {
            None
        } else {
            rope_factors
        };
        let q = match freqs {
            Some(freqs) => g.rope_dynamic_offset_factors(q, layer_theta, position, head_dim, freqs),
            None => g.rope_dynamic_offset(q, layer_theta, position, head_dim),
        };

        // Layers sharing a KV cache have no K/V projections and no cache
        // write: their attention reads the owning layer's cache, written
        // earlier in the same block.
        let source = config.kv_source(layer);
        let (k_written, v_written) = if source == layer {
            let w_k = projection(
                g,
                model,
                config,
                &format!("{p}.attn_k.weight"),
                &[hidden, kv_dim],
            )?;
            let w_v = projection(
                g,
                model,
                config,
                &format!("{p}.attn_v.weight"),
                &[hidden, kv_dim],
            )?;
            let mut k = project(g, attn_in, w_k);
            let mut v = project(g, attn_in, w_v);
            if arch.qk_norm() {
                k = per_head_norm(
                    g,
                    model,
                    &format!("{p}.attn_k_norm.weight"),
                    k,
                    block_size,
                    config.num_kv_heads_at(layer),
                    head_dim,
                    eps,
                )?;
            }
            if arch.norms_values() {
                // Gemma4 norms V with an all-ones weight — the same
                // head-wide rsqrt as Q and K, but no learned scale.
                let ones = g.constant(vec![1.0; head_dim as usize], &[head_dim as usize]);
                let wide = g.reshape(
                    v,
                    &[
                        block_size * config.num_kv_heads_at(layer) as usize,
                        head_dim as usize,
                    ],
                );
                let normed = g.rms_norm(wide, ones, eps);
                v = g.reshape(normed, &[block_size, kv_dim]);
            }
            let k = match freqs {
                Some(freqs) => {
                    g.rope_dynamic_offset_factors(k, layer_theta, position, head_dim, freqs)
                }
                None => g.rope_dynamic_offset(k, layer_theta, position, head_dim),
            };
            let k_cache = g.parameter(&ModelGraph::k_cache_name(layer), &[max_seq_len, kv_dim]);
            let v_cache = g.parameter(&ModelGraph::v_cache_name(layer), &[max_seq_len, kv_dim]);
            // Only the valid rows enter the cache, and attention must read
            // the *written* caches: with no data dependency the scheduler
            // may order this block's write after the attention that should
            // see it.
            let k_written = g.cache_write_prefix(k, k_cache, position, valid);
            let v_cache_w = g.cache_write_prefix(v, v_cache, position, valid);
            (k_written, Some(v_cache_w))
        } else {
            (
                k_updated[source].expect("a shared layer's source is always earlier"),
                None,
            )
        };

        let v_written = match v_written {
            Some(w) => w,
            None => v_updated[source].expect("a shared layer's source is always earlier"),
        };

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
            config.num_kv_heads_at(layer),
            head_dim,
            window,
        );

        k_updated[layer] = Some(k_written);
        v_updated[layer] = Some(v_written);
        if source == layer {
            k_caches.push(k_written);
            v_caches.push(v_written);
        }

        let wo = projection(
            g,
            model,
            config,
            &format!("{p}.attn_output.weight"),
            &[q_dim, hidden],
        )?;
        let mut attn_out = project(g, attn, wo);
        attn_out = optional_bias(g, model, &format!("{p}.attn_output.bias"), attn_out, hidden)?;
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

        // Gemma4's per-layer embedding mixing: the block gates its own
        // output against the layer's PLE row, projects, normed, and adds —
        // the one-hot matmul selects layer `layer`'s token-major rows.
        if let Some(ple) = ple {
            let ple_size = config.per_layer_embed_size;
            let w_gate = projection(
                g,
                model,
                config,
                &format!("{p}.inp_gate.weight"),
                &[hidden, ple_size],
            )?;
            let w_proj = projection(
                g,
                model,
                config,
                &format!("{p}.proj.weight"),
                &[ple_size, hidden],
            )?;
            let post_name = format!("{p}.post_norm.weight");
            check_vector(&post_name, require_tensor(model, &post_name)?, hidden)?;
            let post = g.parameter(&post_name, &[hidden]);

            let mut sel = vec![0.0f32; block_size * ple_rows];
            for (t, slot) in sel.chunks_mut(ple_rows).enumerate() {
                slot[t * config.num_layers + layer] = 1.0;
            }
            let sel = g.constant(sel, &[block_size, ple_rows]);
            let selected = g.matmul(sel, ple);
            let ple_layer = g.reshape(selected, &[block_size, ple_size]);

            let gated = project(g, x, w_gate);
            let gated = g.gelu(gated);
            let mixed = g.mul(gated, ple_layer);
            let mixed = project(g, mixed, w_proj);
            let mixed = g.rms_norm(mixed, post, eps);
            x = g.add(x, mixed);
        }
        if arch.scales_block_outputs() {
            let name = format!("{p}.layer_output_scale.weight");
            if let Some(tensor) = model.tensors.get(&name) {
                check_vector(&name, tensor, 1)?;
                let scale = tensor.to_f32()?;
                x = g.scale(x, scale[0]);
            }
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
        let head = projection(g, model, config, OUTPUT, &[hidden, config.vocab_size])?;
        project(g, last, head)
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
/// The per-layer embedding projection: all layers' PLE rows in one wide
/// output, which each block normed per row and added the gathered table
/// row onto.
pub const PER_LAYER_PROJ: &str = "per_layer_model_proj.weight";

/// Reject what the ops cannot express, naming the reason.
fn check_expressible(model: &GgufModel, config: &ModelConfig) -> Result<(), GgufError> {
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
    for layer in 0..config.num_layers {
        let head_dim = config.head_dim_at(layer);
        if !head_dim.is_multiple_of(2) {
            return Err(GgufError::UnsupportedArchitecture(format!(
                "{}: RoPE needs an even head width at layer {layer}, got {head_dim}",
                config.architecture
            )));
        }
        if head_dim > 512 {
            return Err(GgufError::UnsupportedArchitecture(format!(
                "{}: cached attention supports heads up to 512 wide, got {head_dim} at layer {layer}",
                config.architecture
            )));
        }
        let rope_dim = config.rope_dim_at(layer);
        if rope_dim != head_dim {
            return Err(GgufError::UnsupportedArchitecture(format!(
                "{}: RoPE rotates {rope_dim} dimensions of the {head_dim}-wide head at layer {layer}; this graph requires whole-head rotation",
                config.architecture
            )));
        }
    }
    if config.attn_logit_softcap.is_some() {
        return Err(GgufError::UnsupportedArchitecture(format!(
            "{}: attn_logit_softcapping has no parameter on the cached attention \
             ops, and dropping it would change every attention distribution",
            config.architecture
        )));
    }
    // Position scaling changes the angle at every position, so reading the
    // base and ignoring the scheme loads a different model that still
    // produces fluent text. Refuse rather than approximate.
    if let Some(ref scaling) = config.rope_scaling {
        return Err(GgufError::UnsupportedArchitecture(format!(
            "{}: the file declares RoPE scaling ({scaling}), which the graph \
             has no op for — it emits base-theta rotation only, so honouring \
             the base alone would rotate every position wrongly",
            config.architecture
        )));
    }
    if config.architecture.uses_rope_factors() {
        // Gemma4's file-wide factors encode proportional RoPE on full-
        // attention layers. Those layers require exactly this tensor.
        let name = "rope_freqs.weight";
        let tensor = model
            .tensors
            .get(name)
            .ok_or_else(|| GgufError::MissingTensor(format!("the file has no `{name}`")))?;
        let freqs = tensor.to_f32()?;
        let expected = (config.rope_dim / 2) as usize;
        if tensor.dims.as_slice() != [expected] || freqs.len() != expected {
            return Err(GgufError::BadShape(format!(
                "{name} has shape {:?}, expected [{expected}] for the {}-wide full-attention head",
                tensor.dims, config.rope_dim
            )));
        }
        for (i, &f) in freqs.iter().enumerate() {
            if !f.is_finite() || f <= 0.0 {
                return Err(GgufError::BadMetadata(format!(
                    "{name} pair {i} carries {f}, which is not a rope divisor"
                )));
            }
        }
        if let Some(extra) = model.tensors.keys().find(|candidate| {
            candidate.as_str() != name
                && (candidate.contains("rope_freqs") || candidate.contains("rope_factors"))
        }) {
            return Err(GgufError::UnsupportedArchitecture(format!(
                "{}: the extra RoPE correction `{extra}` has no graph path",
                config.architecture
            )));
        }
    } else if let Some(name) = rope_factor_tensor(model) {
        return Err(GgufError::UnsupportedArchitecture(format!(
            "{}: the file carries `{name}`, a per-frequency RoPE correction \
             that the graph cannot apply; dropping it would rotate long \
             positions wrongly",
            config.architecture
        )));
    }
    for key in ["expert_count", "expert_used_count"] {
        if let Some(value) = model.arch_key(key) {
            let count = value.as_u64().ok_or_else(|| {
                GgufError::BadMetadata(format!("{key} is not an unsigned integer: {value:?}"))
            })?;
            if count > 0 {
                return Err(GgufError::UnsupportedArchitecture(format!(
                    "{}: MoE metadata `{key} = {count}` has no graph path",
                    config.architecture
                )));
            }
        }
    }
    if let Some(name) = model.tensors.keys().find(|name| {
        name.contains(".experts.")
            || name.contains("ffn_gate_inp")
            || name.contains("ffn_gate_exps")
            || name.contains("ffn_up_exps")
            || name.contains("ffn_down_exps")
            || name.contains("ffn_gate_up_exps")
    }) {
        return Err(GgufError::UnsupportedArchitecture(format!(
            "{}: expert tensor `{name}` has no graph path",
            config.architecture
        )));
    }
    Ok(())
}

/// The name of a RoPE correction tensor, if the file carries one.
///
/// Llama 3.1 writes `rope_freqs.weight` and Phi3's long-context variants
/// write per-layer `rope_factors_long` / `rope_factors_short`. Upstream
/// passes these into `ggml_rope_ext`; there is nowhere to put them here,
/// and a file that ships them is not a plain-RoPE model however ordinary
/// its architecture name looks. Gemma4's `rope_freqs.weight` is handled
/// separately because its factors encode proportional RoPE on full layers.
fn rope_factor_tensor<'a>(model: &'a GgufModel) -> Option<&'a str> {
    let mut found: Option<&str> = None;
    for name in model.tensors.keys() {
        if name.contains("rope_freqs") || name.contains("rope_factors") {
            // Deterministic across runs, since `tensors` is a HashMap.
            found = Some(match found {
                Some(seen) if seen <= name.as_str() => seen,
                _ => name.as_str(),
            });
        }
    }
    found
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
    let ffn = config.layer_ffn_size(layer_index(prefix));

    let hidden_act = if arch.gated_ffn() {
        let w_gate = projection(
            g,
            model,
            config,
            &format!("{prefix}.ffn_gate.weight"),
            &[hidden, ffn],
        )?;
        let w_up = projection(
            g,
            model,
            config,
            &format!("{prefix}.ffn_up.weight"),
            &[hidden, ffn],
        )?;
        let gate = project(g, input, w_gate);
        let up = project(g, input, w_up);
        if arch.gates_with_gelu() {
            // Gemma gates with GELU where llama gates with SiLU. The
            // multiply is the same; only the activation differs, so this
            // cannot go through the fused `swiglu`.
            let activated = g.gelu(gate);
            g.mul(activated, up)
        } else {
            g.swiglu(gate, up)
        }
    } else {
        let w_up = projection(
            g,
            model,
            config,
            &format!("{prefix}.ffn_up.weight"),
            &[hidden, ffn],
        )?;
        let mut up = project(g, input, w_up);
        up = optional_bias(g, model, &format!("{prefix}.ffn_up.bias"), up, ffn)?;
        g.gelu(up)
    };

    let w_down = projection(
        g,
        model,
        config,
        &format!("{prefix}.ffn_down.weight"),
        &[ffn, hidden],
    )?;
    let mut out = project(g, hidden_act, w_down);
    out = optional_bias(g, model, &format!("{prefix}.ffn_down.bias"), out, hidden)?;
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
    check_vector(&weight_name, require_tensor(model, &weight_name)?, hidden)?;
    let w = g.parameter(&weight_name, &[hidden]);
    if config.architecture.uses_layer_norm() {
        let bias_name = format!("{name}.bias");
        check_vector(&bias_name, require_tensor(model, &bias_name)?, hidden)?;
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
    check_vector(name, require_tensor(model, name)?, head_dim as usize)?;
    let w = g.parameter(name, &[head_dim as usize]);
    let wide = g.reshape(x, &[rows * heads as usize, head_dim as usize]);
    let normed = g.rms_norm(wide, w, eps);
    Ok(g.reshape(normed, &[rows, heads as usize * head_dim as usize]))
}

/// Add a bias vector, if the file carries one.
///
/// Biases are optional because GGUF families are inconsistent about them
/// in ways that are not worth a variant each: Qwen2 biases Q, K and V but
/// *not* the attention output, and producers differ over the
/// feed-forward. An absent bias is a zero bias, which is exactly the graph
/// without the add — so a missing tensor here means "no bias", not an
/// error. A bias of the wrong *width* is still an error.
fn optional_bias(
    g: &mut Graph,
    model: &GgufModel,
    name: &str,
    x: NodeId,
    width: usize,
) -> Result<NodeId, GgufError> {
    let Some(tensor) = model.tensors.get(name) else {
        return Ok(x);
    };
    check_vector(name, tensor, width)?;
    let b = g.parameter(name, &[width]);
    Ok(g.bias_add(x, b))
}

/// A 1-D tensor must be exactly as wide as the thing it applies to.
///
/// Without this a norm weight of the wrong length reaches the runtime's
/// buffer-size assertion and panics, where a mis-shaped *projection*
/// already returns an error.
fn check_vector(name: &str, tensor: &super::GgufTensor, width: usize) -> Result<(), GgufError> {
    if tensor.dims.as_slice() != [width] {
        return Err(GgufError::MissingTensor(format!(
            "`{name}` is {:?}, but the architecture makes it [{width}]",
            tensor.dims
        )));
    }
    Ok(())
}

/// Where a parameter's values live in the file.
///
/// Usually a tensor of its own, but Phi packs Q, K and V into one
/// `attn_qkv.weight` and Phi3 packs the feed-forward gate and up into one
/// double-width `ffn_up.weight`. Those parameters name a *row range*
/// within the packed tensor, which [`super::weights`] slices out. Rows are
/// output features, and GGUF blocks along the other axis, so a row range
/// is a contiguous run of bytes whatever the encoding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Source {
    /// The tensor to read, as the file names it.
    pub tensor: String,
    /// Output rows to take from it, or `None` for all of them.
    pub rows: Option<std::ops::Range<usize>>,
}

impl Source {
    fn whole(name: &str) -> Self {
        Self {
            tensor: name.to_string(),
            rows: None,
        }
    }

    fn slice(name: &str, rows: std::ops::Range<usize>) -> Self {
        Self {
            tensor: name.to_string(),
            rows: Some(rows),
        }
    }
}

/// Resolve a parameter name to the tensor and rows backing it.
///
/// Both the builder and the loader go through this, so they cannot
/// disagree about which bytes a parameter is made of.
pub(super) fn layer_index(prefix: &str) -> usize {
    prefix
        .strip_prefix("blk.")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

pub(super) fn source_of(config: &ModelConfig, param: &str) -> Source {
    let arch = config.architecture;
    if !arch.packs_qkv() && !arch.packs_gate_up() {
        return Source::whole(param);
    }
    // `blk.0.attn_q.weight` splits into the block prefix `blk.0` and the
    // leaf `attn_q.weight` — two dotted components, not one, which is why
    // a single `rsplit_once` is not enough.
    let Some((prefix, leaf)) = split_block_leaf(param) else {
        return Source::whole(param);
    };

    let q = config.q_dim();
    let kv = config.kv_dim_at(0);
    let ffn = config.intermediate_size;

    if arch.packs_qkv() {
        let packed = format!("{prefix}.attn_qkv.weight");
        match leaf {
            "attn_q.weight" => return Source::slice(&packed, 0..q),
            "attn_k.weight" => return Source::slice(&packed, q..q + kv),
            "attn_v.weight" => return Source::slice(&packed, q + kv..q + 2 * kv),
            _ => {}
        }
    }
    if arch.packs_gate_up() {
        // Gate first, then up — the order llama.cpp's converter writes
        // them in. Reversed, the gate would silently become the value it
        // is supposed to multiply.
        let packed = format!("{prefix}.ffn_up.weight");
        match leaf {
            "ffn_gate.weight" => return Source::slice(&packed, 0..ffn),
            "ffn_up.weight" => return Source::slice(&packed, ffn..2 * ffn),
            _ => {}
        }
    }
    Source::whole(param)
}

/// Split `blk.<n>.<part>.<kind>` into `blk.<n>` and `<part>.<kind>`.
fn split_block_leaf(param: &str) -> Option<(&str, &str)> {
    let (head, kind) = param.rsplit_once('.')?;
    let (prefix, part) = head.rsplit_once('.')?;
    let leaf_start = param.len() - (part.len() + 1 + kind.len());
    Some((prefix, &param[leaf_start..]))
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
    config: &ModelConfig,
    name: &str,
    shape: &[usize; 2],
) -> Result<NodeId, GgufError> {
    let source = source_of(config, name);
    let tensor = require_tensor(model, &source.tensor)?;
    // A sliced source only has to *contain* the rows taken from it — the
    // other slices account for the rest. A whole one must match exactly:
    // nothing would read the surplus rows of an oversized tensor, so
    // accepting it would silently load a differently shaped model.
    let fits = tensor.dims.len() == 2
        && tensor.dims[0] == shape[0]
        && match source.rows.as_ref() {
            Some(rows) => tensor.dims[1] >= rows.end && rows.len() == shape[1],
            None => tensor.dims[1] == shape[1],
        };
    if !fits {
        let needed = match source.rows.as_ref() {
            Some(rows) => format!("rows {}..{} of [{}, _]", rows.start, rows.end, shape[0]),
            None => format!("{shape:?}"),
        };
        return Err(GgufError::MissingTensor(format!(
            "`{}` is {:?}, but `{name}` needs {needed}",
            source.tensor, tensor.dims,
        )));
    }
    let dtype = weight_dtype(tensor)?;
    let shape = if matches!(dtype, DType::F32 | DType::F16) {
        [shape[1], shape[0]]
    } else {
        *shape
    };
    Ok(parameter_of(g, name, &shape, dtype))
}

fn project(g: &mut Graph, input: NodeId, weight: NodeId) -> NodeId {
    if matches!(g.node(weight).ty.dtype, DType::F32 | DType::F16) {
        g.matmul_bt(input, weight)
    } else {
        g.matmul(input, weight)
    }
}

/// The dtype a projection weight is declared — and so must be *filled* —
/// with.
///
/// [`super::weights`] reads this too, so the loader cannot disagree with
/// the graph about whether a tensor arrives packed or as f32.
pub(super) fn weight_dtype(tensor: &super::GgufTensor) -> Result<DType, GgufError> {
    Ok(match tensor.ggml_type {
        super::GgmlType::F32 => DType::F32,
        super::GgmlType::F16 => DType::F16,
        // BF16 has no packed Meganeura form; f32 holds every bf16 value
        // exactly, so that is the lossless spelling.
        super::GgmlType::BF16 => DType::F32,
        // Anything block-packed keeps the file's own encoding: going
        // through f32 would requantize on the way back in and roughly
        // double the error the file already carries.
        _ => tensor.packed_dtype()?,
    })
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
    if arch.uses_rope_factors() && config.rope_dim > 0 {
        names.push("rope_freqs.weight".to_string());
    }
    if arch.uses_per_layer_embeddings() {
        names.push(PER_LAYER_PROJ.to_string());
        names.push("per_layer_proj_norm.weight".to_string());
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
        // Layers sharing a KV cache declare no K/V projections of their
        // own: their attention reads the owning layer's cache, which
        // `weights::load` fills from the owner's tensors.
        let has_kv = config.kv_source(layer) == layer;
        for (part, needed) in [
            ("attn_q", true),
            ("attn_k", has_kv),
            ("attn_v", has_kv),
            ("attn_output", true),
        ] {
            if needed {
                names.push(format!("{p}.{part}.weight"));
            }
        }
        if arch.qk_norm() {
            names.push(format!("{p}.attn_q_norm.weight"));
            if has_kv {
                names.push(format!("{p}.attn_k_norm.weight"));
            }
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
        }
        names.push(format!("{p}.ffn_up.weight"));
        names.push(format!("{p}.ffn_down.weight"));
        if arch.uses_per_layer_embeddings() {
            for part in ["inp_gate.weight", "proj.weight", "post_norm.weight"] {
                names.push(format!("{p}.{part}"));
            }
        }
    }
    names
}

/// The bias parameters [`build`] will add *if* the file carries them.
///
/// Separate from [`parameter_names`] because these are not required: an
/// absent bias is a zero bias and the graph simply omits the add. The
/// loader needs the list to fill the ones that are present.
pub fn optional_parameter_names(config: &ModelConfig) -> Vec<String> {
    let mut names = Vec::new();
    for layer in 0..config.num_layers {
        let p = format!("blk.{layer}");
        for part in ["attn_q", "attn_k", "attn_v", "attn_output"] {
            names.push(format!("{p}.{part}.bias"));
        }
        names.push(format!("{p}.ffn_up.bias"));
        names.push(format!("{p}.ffn_down.bias"));
    }
    names
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_declared_rope_scaling_is_refused_rather_than_ignored() {
        // The graph emits base-theta rotation only. Honouring the base and
        // dropping the scheme would rotate every position wrongly while
        // still producing fluent text.
        for (kind, factor) in [("linear", 4.0), ("yarn", 8.0), ("longrope", 2.0)] {
            let model = fixture::model_with("llama", |m| {
                fixture::set_arch_key(m, "rope.scaling.type", GgufValue::String(kind.into()));
                fixture::set_arch_key(m, "rope.scaling.factor", GgufValue::F32(factor));
            });
            let config = ModelConfig::from_gguf(&model).unwrap();
            let mut g = crate::Graph::new();
            let err = build(&mut g, &model, &config, 4, 16).unwrap_err();
            assert!(format!("{err}").contains(kind), "{err}");
        }
    }

    #[test]
    fn an_identity_scaling_declaration_is_not_a_refusal() {
        // Producers write the keys even when they mean "none"; a factor of
        // one changes nothing and must not reject the file.
        for (kind, factor) in [("none", 8.0), ("linear", 1.0)] {
            let model = fixture::model_with("llama", |m| {
                fixture::set_arch_key(m, "rope.scaling.type", GgufValue::String(kind.into()));
                fixture::set_arch_key(m, "rope.scaling.factor", GgufValue::F32(factor));
            });
            let config = ModelConfig::from_gguf(&model).unwrap();
            assert_eq!(config.rope_scaling, None, "{kind} {factor}");
            let mut g = crate::Graph::new();
            build(&mut g, &model, &config, 4, 16).expect("identity scaling should build");
        }
    }

    #[test]
    fn a_legacy_linear_scale_is_refused_too() {
        let model = fixture::model_with("llama", |m| {
            fixture::set_arch_key(m, "rope.scale_linear", GgufValue::F32(4.0));
        });
        let config = ModelConfig::from_gguf(&model).unwrap();
        let mut g = crate::Graph::new();
        assert!(build(&mut g, &model, &config, 4, 16).is_err());
    }

    #[test]
    fn a_rope_correction_tensor_is_refused_by_name() {
        // Llama 3.1 ships `rope_freqs.weight`; Phi3's long-context
        // variants ship per-layer factors. Either makes the file something
        // other than a plain-RoPE model, whatever its architecture says.
        for name in ["rope_freqs.weight", "blk.0.rope_factors_long.weight"] {
            let model = fixture::model_with("llama", |m| {
                m.tensors
                    .insert(name.to_string(), fixture::f32_tensor(vec![8]));
            });
            let config = ModelConfig::from_gguf(&model).unwrap();
            let mut g = crate::Graph::new();
            let err = build(&mut g, &model, &config, 4, 16).unwrap_err();
            assert!(format!("{err}").contains(name), "{err}");
        }
    }

    #[test]
    fn an_oversized_whole_tensor_is_rejected() {
        // Nothing reads the surplus rows, so accepting it would load a
        // differently shaped model than the one declared.
        let model = fixture::model_with("llama", |m| {
            let dims = m.tensors["blk.0.attn_q.weight"].dims.clone();
            m.tensors.insert(
                "blk.0.attn_q.weight".to_string(),
                fixture::f32_tensor(vec![dims[0], dims[1] + 1]),
            );
        });
        let config = ModelConfig::from_gguf(&model).unwrap();
        let mut g = crate::Graph::new();
        let err = build(&mut g, &model, &config, 4, 16).unwrap_err();
        assert!(format!("{err}").contains("attn_q.weight"), "{err}");
    }
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
    fn the_gemma4_graph_declares_every_weight_the_loader_must_fill() {
        let (g, _, config) = build_fixture("gemma4");
        let declared = declared_parameters(&g);
        let mut expected = parameter_names(&config);
        for layer in 0..config.num_layers {
            if config.kv_source(layer) == layer {
                expected.push(ModelGraph::k_cache_name(layer));
                expected.push(ModelGraph::v_cache_name(layer));
            }
        }
        expected.sort();
        let mut actual: Vec<String> = declared.keys().cloned().collect();
        actual.sort();
        assert_eq!(actual, expected);
        for name in [
            PER_LAYER_PROJ,
            "per_layer_proj_norm.weight",
            "blk.0.inp_gate.weight",
            "blk.0.proj.weight",
            "blk.0.post_norm.weight",
        ] {
            assert!(
                parameter_names(&config).iter().any(|param| param == name),
                "{name}"
            );
        }
    }

    #[test]
    fn gemma4_rope_factors_apply_only_to_full_attention_layers() {
        let (g, _, config) = build_fixture("gemma4");
        let declared = declared_parameters(&g);
        assert_eq!(
            declared["blk.0.attn_k.weight"],
            vec![config.hidden_size, config.kv_dim_at(0)]
        );
        assert_eq!(
            declared["blk.1.attn_k.weight"],
            vec![config.hidden_size, config.kv_dim_at(1)]
        );
        let kv_heads: Vec<_> = g
            .nodes()
            .iter()
            .filter_map(|node| match node.op {
                crate::graph::Op::CachedBlockAttention { num_kv_heads, .. } => Some(num_kv_heads),
                _ => None,
            })
            .collect();
        assert_eq!(kv_heads, vec![1, 2]);
        let factors: Vec<_> = g
            .nodes()
            .iter()
            .filter_map(|node| match node.op {
                crate::graph::Op::RoPE {
                    head_dim,
                    freq_factors: true,
                    ..
                } => Some(head_dim),
                _ => None,
            })
            .collect();
        assert_eq!(factors, vec![config.head_dim; 2]);
    }

    #[test]
    fn gemma4_refuses_unsupported_local_rotary_width_and_moe() {
        let mut missing_factors = fixture::model("gemma4");
        missing_factors.tensors.remove("rope_freqs.weight");
        let config = ModelConfig::from_gguf(&missing_factors).unwrap();
        assert!(matches!(
            build(&mut Graph::new(), &missing_factors, &config, 4, 16),
            Err(GgufError::MissingTensor(_))
        ));

        let mut local_rope = fixture::model("gemma4");
        fixture::set_arch_key(
            &mut local_rope,
            "rope.dimension_count_swa",
            GgufValue::U32(4),
        );
        let config = ModelConfig::from_gguf(&local_rope).unwrap();
        assert!(matches!(
            build(&mut Graph::new(), &local_rope, &config, 4, 16),
            Err(GgufError::UnsupportedArchitecture(_))
        ));

        for key in ["expert_count", "expert_used_count"] {
            let mut moe = fixture::model("gemma4");
            fixture::set_arch_key(&mut moe, key, GgufValue::U32(128));
            let config = ModelConfig::from_gguf(&moe).unwrap();
            assert!(matches!(
                build(&mut Graph::new(), &moe, &config, 4, 16),
                Err(GgufError::UnsupportedArchitecture(_))
            ));
        }
    }

    #[test]
    fn gemma4_layer_output_scale_is_optional_but_must_be_scalar_when_present() {
        let mut model = fixture::model("gemma4");
        model.tensors.remove("blk.0.layer_output_scale.weight");
        let config = ModelConfig::from_gguf(&model).unwrap();
        assert!(build(&mut Graph::new(), &model, &config, 4, 16).is_ok());

        model.tensors.insert(
            "blk.0.layer_output_scale.weight".to_string(),
            fixture::f32_tensor(vec![2]),
        );
        assert!(matches!(
            build(&mut Graph::new(), &model, &config, 4, 16),
            Err(GgufError::MissingTensor(_))
        ));
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
            vec![config.hidden_size, config.kv_dim_at(0)]
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
            vec![128, config.kv_dim_at(0)]
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
