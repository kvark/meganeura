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

    /// Base checkpoint (~325M params). Config confirmed against the actual
    /// HuggingFace checkpoint `llm_base_x4286_c1860k`:
    ///   embed=768, heads=12, head_dim=64, mlp=2048, vocab=29824
    ///   encoder=12, temporal_decoder=20, depth_decoder=4
    /// Despite the "base" label, the decoder is the LARGE variant's 20 layers.
    pub fn base() -> Self {
        Self {
            embed_dim: 768,
            head_dim: 64,
            num_heads: 12,
            mlp_dim: 2048,
            num_encoder_layers: 12,
            num_temporal_decoder_layers: 20,
            num_depth_decoder_layers: 4,
            num_levels: 16,
            vocab_size: 29824,
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
/// Builds all `num_encoder_layers` and the final norm. Structurally complete
/// and wired to real ops, but not yet verified against a weight dump, and
/// batch=1 only (see the broadcast TODO inside).
pub fn build_encoder_graph(g: &mut Graph, cfg: &LlmConfig, batch: usize) -> NodeId {
    // CFG ultimately needs batch=2 (positive + negative style), but adding the
    // shared pos_embed to both rows requires a broadcast-add op that isn't in
    // place yet — so the encoder is batch=1 only for now.
    // TODO(broadcast): add a broadcast-add and lift this to batch ∈ {1, 2}.
    assert_eq!(batch, 1, "encoder is batch=1 until a broadcast-add op lands (CFG batch=2 TODO)");
    let seq = cfg.encoder_seq_len as usize;
    let embed = cfg.embed_dim as usize;

    let token_ids = g.input_u32("encoder_input_tokens", &[batch * seq]);
    let embed_w = g.parameter("shared_token_embedder", &[cfg.vocab_size as usize, embed]);
    let mut x = g.embedding(token_ids, embed_w);

    // Sinusoidal absolute position embeddings (loaded as a constant, not learned).
    let pos_embed = g.parameter("encoder.pos_embed", &[seq, embed]);
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

/// Build one Depthformer *temporal* decoder layer — a standard T5 1.1 decoder
/// layer: pre-norm causal self-attention (with rel-pos bias), pre-norm
/// cross-attention to the encoder output, and a pre-norm GeGLU MLP, each with a
/// residual. T5 uses RMSNorm and no projection biases (DenseGeneral
/// `use_bias=False`). Cross-attention carries no rel-pos bias (T5 encoder-
/// decoder attention is plain scaled-dot-product).
///
/// `self_rel_pos_table` is this layer's learned `[num_heads * num_buckets]`
/// causal-self-attention bias table (registered per-layer, like the encoder).
///
/// Param names: `{prefix}.pre_self_attn_norm`, `.self_attn.{q,k,v,o}`,
/// `.pre_cross_attn_norm`, `.cross_attn.{q,k,v,o}`, `.pre_mlp_norm`,
/// `.mlp.{w_gate,w_up,w_down}`.
pub fn build_decoder_layer(
    g: &mut Graph,
    cfg: &LlmConfig,
    x: NodeId,
    encoder_out: NodeId,
    self_rel_pos_table: NodeId,
    prefix: &str,
) -> NodeId {
    let embed = cfg.embed_dim as usize;
    let attn_dim = (cfg.num_heads * cfg.head_dim) as usize;
    let mlp_dim = cfg.mlp_dim as usize;

    // --- 1. Causal self-attention ---
    let ln1 = g.parameter(&format!("{prefix}.pre_self_attn_norm"), &[embed]);
    let h = g.rms_norm(x, ln1, cfg.layer_norm_eps);
    let wq = g.parameter(&format!("{prefix}.self_attn.q"), &[embed, attn_dim]);
    let wk = g.parameter(&format!("{prefix}.self_attn.k"), &[embed, attn_dim]);
    let wv = g.parameter(&format!("{prefix}.self_attn.v"), &[embed, attn_dim]);
    let q = g.matmul(h, wq);
    let k = g.matmul(h, wk);
    let v = g.matmul(h, wv);
    let sa = g.full_attention_with_rel_pos_bias(
        q,
        k,
        v,
        self_rel_pos_table,
        cfg.num_heads,
        cfg.num_heads,
        cfg.head_dim,
        cfg.rel_pos_num_buckets,
        cfg.rel_pos_max_distance,
        false, // not bidirectional
        true,  // causal
    );
    let wo = g.parameter(&format!("{prefix}.self_attn.o"), &[attn_dim, embed]);
    let sa_out = g.matmul(sa, wo);
    let x = g.add(x, sa_out);

    // --- 2. Cross-attention to encoder output (no rel-pos bias) ---
    let ln2 = g.parameter(&format!("{prefix}.pre_cross_attn_norm"), &[embed]);
    let h = g.rms_norm(x, ln2, cfg.layer_norm_eps);
    let wcq = g.parameter(&format!("{prefix}.cross_attn.q"), &[embed, attn_dim]);
    let wck = g.parameter(&format!("{prefix}.cross_attn.k"), &[embed, attn_dim]);
    let wcv = g.parameter(&format!("{prefix}.cross_attn.v"), &[embed, attn_dim]);
    let cq = g.matmul(h, wcq);
    let ck = g.matmul(encoder_out, wck);
    let cv = g.matmul(encoder_out, wcv);
    let ca = g.cross_attention(cq, ck, cv, cfg.num_heads, cfg.num_heads, cfg.head_dim);
    let wco = g.parameter(&format!("{prefix}.cross_attn.o"), &[attn_dim, embed]);
    let ca_out = g.matmul(ca, wco);
    let x = g.add(x, ca_out);

    // --- 3. GeGLU MLP ---
    let ln3 = g.parameter(&format!("{prefix}.pre_mlp_norm"), &[embed]);
    let h = g.rms_norm(x, ln3, cfg.layer_norm_eps);
    let w_gate = g.parameter(&format!("{prefix}.mlp.w_gate"), &[embed, mlp_dim]);
    let w_up = g.parameter(&format!("{prefix}.mlp.w_up"), &[embed, mlp_dim]);
    let w_down = g.parameter(&format!("{prefix}.mlp.w_down"), &[mlp_dim, embed]);
    let gate = g.matmul(h, w_gate);
    let up = g.matmul(h, w_up);
    let ffn = g.geglu(gate, up);
    let ffn_out = g.matmul(ffn, w_down);
    g.add(x, ffn_out)
}

// TODO(decoder): the temporal decoder layer above is the verifiable core. Still
// to reverse-engineer + build (see tools/magenta_rt/LLM_FINDINGS.md):
//   - build_depth_decoder_layer: the per-frame inner transformer over the 16
//     RVQ levels (no cross-attention; conditioned on the temporal output).
//   - build_decoder_graph: token-embed + position encoding + the temporal
//     stack + the depth module + the weight-tied logits head.
//   - KV cache for autoregressive decode, and the generation loop wiring in
//     `super::sampling`. The exact position-encoding scheme (T5 rel-pos vs the
//     absolute sinusoidal PE the numpy ref assumes) is unresolved and needs the
//     checkpoint tensor names to settle.

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
    fn config_base_matches_checkpoint() {
        let c = LlmConfig::base();
        // Verified against weights_llm_base.safetensors manifest.
        assert_eq!(c.embed_dim, 768);
        assert_eq!(c.num_heads * c.head_dim, c.embed_dim);
        assert_eq!(c.num_encoder_layers, 12);
        assert_eq!(c.num_temporal_decoder_layers, 20);
        assert_eq!(c.num_depth_decoder_layers, 4);
        assert_eq!(c.vocab_size, 29824);
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
