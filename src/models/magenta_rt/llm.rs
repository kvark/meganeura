//! Magenta-RT LLM: T5 1.1 encoder-decoder ("Depthformer").
//!
//! Architecture (large variant, from `depthformer/configs/mrt_merged_large.gin`):
//!
//! - **Encoder** (bidirectional): 24 layers, embed=1024, mlp=2816, 16 heads × 64 head_dim.
//!   Stack of `[T5LayerNorm → self-attn(rel_pos_bias) → residual → T5LayerNorm → GeGLU MLP → residual]`.
//! - **Decoder** ("Depthformer", hierarchical autoregressive):
//!   - 20 *temporal* layers operating over 50 frames (sees cross-attention to encoder).
//!   - 4 *depth* layers operating over 16 RVQ levels per frame (no cross-attention).
//!   - The decoded sequence is `50 frames × 16 RVQ = 800 tokens`.
//! - **Embeddings**: shared token embedder (29698 × 1024), added to non-learned
//!   sinusoidal absolute position embeddings (FixedEmbed, max_length=1006).
//! - **Rel-pos bias**: per-layer learned T5 buckets — encoder/temporal use
//!   (32 buckets, max 128), depth uses (16, 16). All bidirectional except
//!   causal decoder self-attn.
//! - **T5LayerNorm** = RMSNorm without subtract-mean; use [`Graph::rms_norm`].
//! - **DenseGeneral with `use_bias=False`** throughout — every projection
//!   is a bare matmul with no bias term.
//!
//! Sampling (inference): classifier-free guidance (batch=2: positive + negative
//! style), temperature 1.1, top-k 40. See [`super::sampling`].

use crate::graph::{Graph, NodeId};

/// Hyperparameters for the Depthformer encoder-decoder LLM.
///
/// Constructor methods correspond to the two published checkpoint sizes.
#[derive(Clone, Debug)]
pub struct LlmConfig {
    /// 1024 (large) or 768 (base).
    pub embed_dim: u32,
    /// 64.
    pub head_dim: u32,
    /// 16 (large) or 12 (base).
    pub num_heads: u32,
    /// 2816 (large) or 2048 (base) — the GeGLU intermediate dimension.
    pub mlp_dim: u32,
    /// 24 (large) or 12 (base).
    pub num_encoder_layers: u32,
    /// 20 (large) or 10 (base) — temporal decoder stack.
    pub num_temporal_decoder_layers: u32,
    /// 4 — depth decoder stack (operates per-frame over the 16 RVQ levels).
    pub num_depth_decoder_layers: u32,
    /// 16 — number of RVQ levels per frame (depth-decoder unrolling dim).
    pub num_levels: u32,
    /// 29698 — padded vocab the LLM was trained with.
    pub vocab_size: u32,
    /// 1006 — encoder input length: `250 frames × 4 codec RVQ + 6 style tokens`.
    pub encoder_seq_len: u32,
    /// 800 — decoder output length: `50 frames × 16 RVQ levels`.
    pub decoder_seq_len: u32,
    /// 32 — T5 rel-pos buckets for encoder + temporal decoder.
    pub rel_pos_num_buckets: u32,
    /// 128 — T5 rel-pos max distance for encoder + temporal decoder.
    pub rel_pos_max_distance: u32,
    /// 16 — rel-pos buckets for depth decoder (one bucket per RVQ level).
    pub depth_rel_pos_num_buckets: u32,
    /// 16 — rel-pos max distance for depth decoder.
    pub depth_rel_pos_max_distance: u32,
    /// T5LayerNorm epsilon (flaxformer default 1e-6).
    pub layer_norm_eps: f32,
}

impl LlmConfig {
    /// Large checkpoint (~800M params).
    pub fn large() -> Self {
        Self {
            embed_dim: 1024,
            head_dim: 64,
            num_heads: 16,
            mlp_dim: 2816,
            num_encoder_layers: 24,
            num_temporal_decoder_layers: 20,
            num_depth_decoder_layers: 4,
            num_levels: 16,
            vocab_size: 29698,
            encoder_seq_len: 1006,
            decoder_seq_len: 800,
            rel_pos_num_buckets: 32,
            rel_pos_max_distance: 128,
            depth_rel_pos_num_buckets: 16,
            depth_rel_pos_max_distance: 16,
            layer_norm_eps: 1e-6,
        }
    }

    /// Base checkpoint (~325M params). Layer counts and dims inferred from the
    /// gin `size_base.gin` overrides; verify against the actual manifest.
    pub fn base() -> Self {
        Self {
            embed_dim: 768,
            head_dim: 64,
            num_heads: 12,
            mlp_dim: 2048,
            num_encoder_layers: 12,
            num_temporal_decoder_layers: 10,
            num_depth_decoder_layers: 4,
            num_levels: 16,
            vocab_size: 29698,
            encoder_seq_len: 1006,
            decoder_seq_len: 800,
            rel_pos_num_buckets: 32,
            rel_pos_max_distance: 128,
            depth_rel_pos_num_buckets: 16,
            depth_rel_pos_max_distance: 16,
            layer_norm_eps: 1e-6,
        }
    }
}

/// Build one T5 1.1 encoder layer: pre-norm self-attn + pre-norm GeGLU MLP, both with residual.
///
/// `rel_pos_bias_table` is the learned `[num_heads * num_buckets]` parameter
/// for this layer's attention (T5 does NOT share the rel-pos table across
/// layers per the gin config, so callers register a fresh table per layer).
///
/// Parameter names follow this convention (chosen for clarity; the Colab dumper's
/// manifest will use flaxformer's naming and we'll remap during weight loading):
///   `{prefix}.pre_attn_norm`, `{prefix}.attn.q`, `{prefix}.attn.k`,
///   `{prefix}.attn.v`, `{prefix}.attn.o`, `{prefix}.attn.rel_pos_bias_table`,
///   `{prefix}.pre_mlp_norm`, `{prefix}.mlp.w_gate`, `{prefix}.mlp.w_up`,
///   `{prefix}.mlp.w_down`.
pub fn build_encoder_layer(
    g: &mut Graph,
    cfg: &LlmConfig,
    x: NodeId,
    rel_pos_bias_table: NodeId,
    prefix: &str,
) -> NodeId {
    let embed = cfg.embed_dim as usize;
    let mlp_dim = cfg.mlp_dim as usize;
    let attn_dim = (cfg.num_heads * cfg.head_dim) as usize;

    // --- Pre-attention RMSNorm (T5LayerNorm has weight only, no bias, no centering) ---
    let ln1_w = g.parameter(&format!("{prefix}.pre_attn_norm"), &[embed]);
    let h = g.rms_norm(x, ln1_w, cfg.layer_norm_eps);

    // --- Self-attention (no biases anywhere in T5 1.1, but rel-pos bias is in QK^T) ---
    let wq = g.parameter(&format!("{prefix}.attn.q"), &[embed, attn_dim]);
    let wk = g.parameter(&format!("{prefix}.attn.k"), &[embed, attn_dim]);
    let wv = g.parameter(&format!("{prefix}.attn.v"), &[embed, attn_dim]);
    let q = g.matmul(h, wq);
    let k = g.matmul(h, wk);
    let v = g.matmul(h, wv);
    let attn = g.full_attention_with_rel_pos_bias(
        q,
        k,
        v,
        rel_pos_bias_table,
        cfg.num_heads,
        cfg.num_heads,
        cfg.head_dim,
        cfg.rel_pos_num_buckets,
        cfg.rel_pos_max_distance,
        true,  // bidirectional encoder
        false, // not causal
    );

    let wo = g.parameter(&format!("{prefix}.attn.o"), &[attn_dim, embed]);
    let attn_out = g.matmul(attn, wo);
    let x = g.add(x, attn_out);

    // --- Pre-MLP RMSNorm + GeGLU FFN ---
    let ln2_w = g.parameter(&format!("{prefix}.pre_mlp_norm"), &[embed]);
    let h = g.rms_norm(x, ln2_w, cfg.layer_norm_eps);
    let w_gate = g.parameter(&format!("{prefix}.mlp.w_gate"), &[embed, mlp_dim]);
    let w_up = g.parameter(&format!("{prefix}.mlp.w_up"), &[embed, mlp_dim]);
    let w_down = g.parameter(&format!("{prefix}.mlp.w_down"), &[mlp_dim, embed]);
    let gate = g.matmul(h, w_gate);
    let up = g.matmul(h, w_up);
    let ffn = g.geglu(gate, up);
    let ffn_out = g.matmul(ffn, w_down);
    g.add(x, ffn_out)
}

/// Build the encoder forward pass: token-embed → +sinusoidal pos → 24 layers → final norm.
///
/// Returns the encoder output `[seq_len, embed_dim]` for cross-attention into the decoder.
///
/// The token embedding table is shared with the decoder and the output projection
/// (weight tying), so this graph also produces it as a side-channel via the
/// `shared_token_embedder` parameter name.
///
/// TODO: actually implement once the weight manifest is available — current
/// placeholder builds only one layer to validate the loop structure and shape
/// contracts.
pub fn build_encoder_graph(g: &mut Graph, cfg: &LlmConfig, batch: usize) -> NodeId {
    // For CFG, batch=2 (pos + neg). Encoder runs both rows in parallel; we
    // currently emulate that by concatenating batched inputs along seq, but
    // proper batched attention would be cleaner — TODO.
    assert!(batch == 1 || batch == 2, "encoder batch is 1 or 2 (CFG)");
    let seq = cfg.encoder_seq_len as usize;
    let embed = cfg.embed_dim as usize;

    // Token IDs: [batch * seq] flat (with batch=2 stacked on the seq dim)
    let token_ids = g.input_u32("encoder_input_tokens", &[batch * seq]);
    let embed_w = g.parameter("shared_token_embedder", &[cfg.vocab_size as usize, embed]);
    let mut x = g.embedding(token_ids, embed_w);

    // Sinusoidal absolute position embeddings (loaded as a constant, not learned).
    let pos_embed = g.parameter("encoder.pos_embed", &[seq, embed]);
    // TODO(broadcast): when batch=2 we need to add the same pos_embed to both
    // rows. Once a broadcast-add op is in place, do that. For now, panic if batch != 1.
    assert_eq!(batch, 1, "TODO: broadcast pos_embed across CFG batch");
    x = g.add(x, pos_embed);

    for i in 0..cfg.num_encoder_layers {
        let prefix = format!("encoder.layers.{i}");
        // Rel-pos bias table for this layer (per gin: not shared across layers).
        // The fused FullAttentionRelPosBias kernel does the bucket lookup inline,
        // so we pass the small [num_heads, num_buckets] table directly.
        let bias_table = g.parameter(
            &format!("{prefix}.attn.rel_pos_bias_table"),
            &[(cfg.num_heads * cfg.rel_pos_num_buckets) as usize],
        );
        x = build_encoder_layer(g, cfg, x, bias_table, &prefix);
    }

    let final_ln_w = g.parameter("encoder.final_norm", &[embed]);
    g.rms_norm(x, final_ln_w, cfg.layer_norm_eps)
}

// TODO(decoder): build_decoder_layer (with cross-attention to encoder + causal
// self-attn with rel_pos_bias), then build_depth_decoder_layer (per-frame inner
// transformer over 16 RVQ levels), then build_decoder_graph that combines them.
// Decoder is autoregressive; needs KV cache for inference. Defer until weights
// are in place so we can validate against the Colab reference at each layer.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_large_matches_gin() {
        let c = LlmConfig::large();
        assert_eq!(c.embed_dim, 1024);
        assert_eq!(c.num_heads * c.head_dim, c.embed_dim);
        assert_eq!(c.num_encoder_layers, 24);
        assert_eq!(c.num_temporal_decoder_layers + c.num_depth_decoder_layers, 24);
        assert_eq!(c.encoder_seq_len, 1006);
        assert_eq!(c.decoder_seq_len, c.num_levels * 50);
    }

    #[test]
    fn encoder_graph_builds_with_expected_param_count() {
        // Smoke test: just confirm the graph constructs without panicking
        // and registers a sensible number of parameters.
        let cfg = LlmConfig::large();
        let mut g = Graph::new();
        let _logits = build_encoder_graph(&mut g, &cfg, 1);
        // Per-layer: 4 attn matmuls + 3 mlp matmuls + 2 layer norms + 1 rel-pos
        // table = 10 params per layer × 24 layers = 240
        // Plus shared: token embed + pos embed + final norm = 243.
        // Allow a small fudge for graph internals.
        assert!(
            g.nodes().len() > 24 * 10,
            "encoder graph should have ≥ ~240 parameter nodes plus ops, got {} nodes",
            g.nodes().len()
        );
    }
}
