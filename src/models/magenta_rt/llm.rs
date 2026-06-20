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
//! - **Embeddings**: shared token embedder (`vocab × embed`). The reference
//!   `depthformer.py` encoder embedder applies `Scale(sqrt(embed))` and adds
//!   **no** position embedding (the position-embed slot is a no-op by default),
//!   and the `llm_base` checkpoint carries no learned PE tensor.
//! - **Rel-pos bias**: learned T5 buckets, shared across each sub-decoder's
//!   layers — `temporal_decoder.relpos_bias` `[heads, 128]` and
//!   `depth_decoder.relpos_bias_depth` `[heads, 16]`. The **encoder** has no
//!   rel-pos tensor in the checkpoint (open question — see
//!   `tools/magenta_rt/LLM_FINDINGS.md`). Decoder self-attn is causal.
//! - **Logits**: a dedicated `decoder.logits_dense.kernel` `[embed, vocab]`
//!   (NOT weight-tied to the token embedder), applied after the shared
//!   `decoder_norm` to the depth-decoder output.
//! - **T5LayerNorm** = RMSNorm without subtract-mean; use [`Graph::rms_norm`].
//! - **DenseGeneral with `use_bias=False`** throughout — every projection
//!   is a bare matmul with no bias term.
//!
//! Sampling (inference): classifier-free guidance (batch=2: positive + negative
//! style), temperature 1.1, top-k 40. See [`super::sampling`].

use crate::graph::{Graph, NodeId};
use crate::runtime::Session;

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
            rel_pos_num_buckets: 128,
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
            // 128 temporal rel-pos buckets — confirmed by the checkpoint manifest
            // (`temporal_decoder.relpos_bias.rel_embedding` is `[12, 128]`).
            rel_pos_num_buckets: 128,
            rel_pos_max_distance: 128,
            depth_rel_pos_num_buckets: 16,
            depth_rel_pos_max_distance: 16,
            layer_norm_eps: 1e-6,
        }
    }
}

/// Build one encoder layer: pre-norm **bidirectional** self-attention + pre-norm
/// GeGLU MLP, both residual. T5 1.1 RMSNorm, no projection biases.
///
/// The `llm_base` checkpoint carries **no encoder rel-pos tensor** (only the two
/// decoders do), so the encoder self-attention here is plain scaled-dot-product
/// (no rel-pos bias) — position enters via the sinusoidal PE added once in
/// [`build_encoder_graph`] (per the in-repo `llm_numpy_ref.py`). This is the one
/// LLM piece not yet checked against real weights; see `LLM_FINDINGS.md`.
///
/// Param names: `{prefix}.pre_attn_norm`, `.attn.{q,k,v,o}`, `.pre_mlp_norm`,
/// `.mlp.{w_gate,w_up,w_down}` — all backed by checkpoint tensors.
pub fn build_encoder_layer(g: &mut Graph, cfg: &LlmConfig, x: NodeId, prefix: &str) -> NodeId {
    let embed = cfg.embed_dim as usize;
    let mlp_dim = cfg.mlp_dim as usize;
    let attn_dim = (cfg.num_heads * cfg.head_dim) as usize;

    // --- Pre-attention RMSNorm (T5LayerNorm: weight only, no bias/centering) ---
    let ln1_w = g.parameter(&format!("{prefix}.pre_attn_norm"), &[embed]);
    let h = g.rms_norm(x, ln1_w, cfg.layer_norm_eps);

    // --- Bidirectional self-attention (no rel-pos bias; plain SDPA) ---
    let wq = g.parameter(&format!("{prefix}.attn.q"), &[embed, attn_dim]);
    let wk = g.parameter(&format!("{prefix}.attn.k"), &[embed, attn_dim]);
    let wv = g.parameter(&format!("{prefix}.attn.v"), &[embed, attn_dim]);
    let q = g.matmul(h, wq);
    let k = g.matmul(h, wk);
    let v = g.matmul(h, wv);
    // `cross_attention` is non-causal scaled-dot-product with no rel-pos bias —
    // exactly bidirectional self-attention when Q/K/V come from the same input.
    let attn = g.cross_attention(q, k, v, cfg.num_heads, cfg.num_heads, cfg.head_dim);

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

/// Fixed sinusoidal absolute position embedding `[seq, embed]` (standard
/// Transformer convention, `10000` timescale; matches `llm_numpy_ref.py`):
/// `pe[p, 2i] = sin(p / 10000^(2i/embed))`, `pe[p, 2i+1] = cos(...)`.
fn sinusoidal_pos_embed(seq: usize, embed: usize) -> Vec<f32> {
    let mut pe = vec![0.0_f32; seq * embed];
    for p in 0..seq {
        for i in 0..embed / 2 {
            let inv_freq = 1.0_f64 / 10000.0_f64.powf(2.0 * i as f64 / embed as f64);
            let angle = p as f64 * inv_freq;
            pe[p * embed + 2 * i] = angle.sin() as f32;
            pe[p * embed + 2 * i + 1] = angle.cos() as f32;
        }
    }
    pe
}

/// Build the encoder forward pass: token-embed → +sinusoidal pos → 24 layers → final norm.
///
/// Returns the encoder output `[seq_len, embed_dim]` for cross-attention into the decoder.
///
/// The token embedding table is shared with the decoder and the output projection
/// (weight tying), so this graph also produces it as a side-channel via the
/// `shared_token_embedder` parameter name.
///
/// Builds all `num_encoder_layers` and the final norm. Every parameter is backed
/// by a checkpoint tensor (the sinusoidal PE is computed, not stored); only the
/// position scheme itself is unverified against real weights (`LLM_FINDINGS.md`).
/// `seq` lets the caller use a length other than `cfg.encoder_seq_len` (e.g. for
/// tests); pass `cfg.encoder_seq_len` for the real model. Batch=1 only.
pub fn build_encoder_graph(g: &mut Graph, cfg: &LlmConfig, seq: usize) -> NodeId {
    let embed = cfg.embed_dim as usize;

    let token_ids = g.input_u32("encoder_input_tokens", &[seq]);
    let embed_w = g.parameter("shared_token_embedder", &[cfg.vocab_size as usize, embed]);
    let mut x = g.embedding(token_ids, embed_w);

    // Fixed sinusoidal absolute position embedding (computed, not a checkpoint
    // tensor — the encoder has no learned PE; per `llm_numpy_ref.py`). NOTE: the
    // v2 `depthformer.py` embedder also applies `Scale(sqrt(embed))`, which the
    // numpy ref omits — an open detail to settle against real weights.
    let pe = g.constant(sinusoidal_pos_embed(seq, embed), &[seq, embed]);
    x = g.add(x, pe);

    for i in 0..cfg.num_encoder_layers {
        let prefix = format!("encoder.layers.{i}");
        x = build_encoder_layer(g, cfg, x, &prefix);
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

/// Build the full-sequence temporal decoder forward pass: token embed → N
/// temporal decoder layers (each cross-attending to `encoder_out`) → final
/// RMSNorm → weight-tied logits.
///
/// This is the parallel (teacher-forcing / prefix-scoring) form; autoregressive
/// inference reuses the same layers with a KV cache (TODO — see
/// `tools/magenta_rt/LLM_FINDINGS.md`). No absolute position embedding is added:
/// standard T5 1.1 relies on the per-layer rel-pos bias, and the absolute-PE
/// question is unresolved (LLM_FINDINGS.md). Logits are weight-tied to the
/// shared token embedder via a transpose.
///
/// Inputs:  `decoder_input_tokens` u32 `[seq_len]`, `encoder_out` `[enc_seq, embed]`.
/// Output:  logits `[seq_len, vocab_size]`.
pub fn build_temporal_decoder_stack(
    g: &mut Graph,
    cfg: &LlmConfig,
    decoder_input_tokens: NodeId,
    encoder_out: NodeId,
) -> NodeId {
    let embed = cfg.embed_dim as usize;
    let table = g.parameter("shared_token_embedder", &[cfg.vocab_size as usize, embed]);
    let mut x = g.embedding(decoder_input_tokens, table);
    for i in 0..cfg.num_temporal_decoder_layers {
        let prefix = format!("decoder.temporal_layers.{i}");
        let rel = g.parameter(
            &format!("{prefix}.self_attn.rel_pos_bias_table"),
            &[(cfg.num_heads * cfg.rel_pos_num_buckets) as usize],
        );
        x = build_decoder_layer(g, cfg, x, encoder_out, rel, &prefix);
    }
    let final_norm = g.parameter("decoder.decoder_norm", &[embed]);
    let x = g.rms_norm(x, final_norm, cfg.layer_norm_eps);
    // Weight-tied logits: x @ token_embedder^T → [seq_len, vocab].
    let logits_w = g.transpose(table);
    g.matmul(x, logits_w)
}

/// Mean-pool a frame's RVQ level embeddings down to one vector per frame:
/// `embedded [num_frames*num_levels, embed] → [num_frames, embed]`.
///
/// This is `embedded.mean(axis=levels)` from `depthformer.py`'s
/// `MultivariateDecoder` — the temporal body sees the average of a frame's 16
/// level embeddings. Implemented as a single matmul by a constant pooling
/// matrix `P [num_frames, num_frames*num_levels]` carrying `1/num_levels` on
/// each frame's contiguous level block; this folds the per-frame grouping and
/// the average into one op (meganeura reductions are scalar/inner-axis only and
/// `mul` has no scalar broadcast, so a pooling matmul is the clean primitive).
fn build_level_mean_pool(
    g: &mut Graph,
    cfg: &LlmConfig,
    embedded: NodeId,
    num_frames: usize,
) -> NodeId {
    let levels = cfg.num_levels as usize;
    let cols = num_frames * levels;
    let inv = 1.0 / levels as f32;
    let mut pool = vec![0.0_f32; num_frames * cols];
    for f in 0..num_frames {
        for l in 0..levels {
            pool[f * cols + f * levels + l] = inv;
        }
    }
    let p = g.constant(pool, &[num_frames, cols]);
    g.matmul(p, embedded)
}

/// Build the full Depthformer *temporal* decoder forward: the per-frame RVQ
/// token grid → shared token embed → mean-pool over the `num_levels` RVQ levels
/// → `num_temporal_decoder_layers` temporal layers (each causal self-attention +
/// cross-attention to `encoder_out`) → the per-frame temporal states
/// `[num_frames, embed]`.
///
/// This is the real temporal input construction (mean-pooled level embeddings,
/// per `depthformer.py`'s `MultivariateDecoder`), unlike
/// [`build_temporal_decoder_stack`], which embeds a flat token sequence directly
/// and appends a placeholder weight-tied logits head. The temporal output here
/// is **not** normalized and carries **no** logits: per the checkpoint the
/// single `decoder_norm` + `logits_dense` head sits after the *depth* module,
/// and these raw temporal states feed the depth module as its level-0 input
/// (see [`build_depth_decoder_stack`]).
///
/// `decoder_input_tokens` is the SOS-shifted grid `[num_frames * num_levels]`
/// (the caller prepends the SOS frame and drops the last per the teacher-forcing
/// shift). The rel-pos bias table is **shared across the temporal layers** — the
/// checkpoint stores one `temporal_decoder.relpos_bias.rel_embedding` (`[heads,
/// 128]`) for the whole sub-decoder, not one per layer.
///
/// Inputs:  `decoder_input_tokens` u32 `[num_frames * num_levels]`,
///          `encoder_out` `[enc_seq, embed]`.
/// Output:  temporal states `[num_frames, embed]`.
pub fn build_temporal_decoder(
    g: &mut Graph,
    cfg: &LlmConfig,
    decoder_input_tokens: NodeId,
    encoder_out: NodeId,
    num_frames: usize,
) -> NodeId {
    let embed = cfg.embed_dim as usize;
    let table = g.parameter("shared_token_embedder", &[cfg.vocab_size as usize, embed]);
    let embedded = g.embedding(decoder_input_tokens, table);
    let mut x = build_level_mean_pool(g, cfg, embedded, num_frames);
    // One shared rel-pos bias table for the whole temporal sub-decoder.
    let rel = g.parameter(
        "decoder.temporal_decoder.rel_pos_bias_table",
        &[(cfg.num_heads * cfg.rel_pos_num_buckets) as usize],
    );
    for i in 0..cfg.num_temporal_decoder_layers {
        let prefix = format!("decoder.temporal_layers.{i}");
        x = build_decoder_layer(g, cfg, x, encoder_out, rel, &prefix);
    }
    x
}

/// Build one Depthformer *depth* decoder layer — the per-frame inner transformer
/// over the 16 RVQ levels. Topology resolved from the checkpoint manifest
/// (`tools/magenta_rt/llm_base_manifest.json`) and the reference
/// `magenta_rt/jax/depthformer.py`: it is a standard T5 1.1 decoder layer
/// **without** cross-attention — pre-norm causal self-attention (its own,
/// depth-specific rel-pos buckets) → pre-norm GeGLU MLP, each residual. The
/// checkpoint's `depth_layers_*` carry exactly `self_attention.{q,k,v,out}`,
/// `mlp.{wi_0,wi_1,wo}`, `pre_self_attention_layer_norm`, `pre_mlp_layer_norm`
/// — no `encoder_decoder_attention`, confirming the absence of cross-attention.
///
/// The depth module conditions on the temporal output via its *input* (the
/// temporal per-frame vector is prepended as level 0; see
/// [`build_depth_decoder_stack`]), not via cross-attention.
///
/// Param names: `{prefix}.pre_self_attn_norm`, `.self_attn.{q,k,v,o}`,
/// `.pre_mlp_norm`, `.mlp.{w_gate,w_up,w_down}`.
pub fn build_depth_decoder_layer(
    g: &mut Graph,
    cfg: &LlmConfig,
    x: NodeId,
    self_rel_pos_table: NodeId,
    prefix: &str,
) -> NodeId {
    let p = register_depth_layer_params(g, cfg, prefix);
    depth_layer_forward(g, cfg, x, self_rel_pos_table, &p)
}

/// Pre-created parameter node handles for one depth layer. Hoisting the params
/// lets the parallel decoder reuse a single set of nodes across all frames (the
/// runtime binds each parameter buffer to exactly one node, so a param name must
/// map to a single graph node — duplicating the nodes would leave all but the
/// first unbound).
struct DepthLayerParams {
    ln1: NodeId,
    q: NodeId,
    k: NodeId,
    v: NodeId,
    o: NodeId,
    ln2: NodeId,
    w_gate: NodeId,
    w_up: NodeId,
    w_down: NodeId,
}

fn register_depth_layer_params(g: &mut Graph, cfg: &LlmConfig, prefix: &str) -> DepthLayerParams {
    let embed = cfg.embed_dim as usize;
    let attn_dim = (cfg.num_heads * cfg.head_dim) as usize;
    let mlp_dim = cfg.mlp_dim as usize;
    DepthLayerParams {
        ln1: g.parameter(&format!("{prefix}.pre_self_attn_norm"), &[embed]),
        q: g.parameter(&format!("{prefix}.self_attn.q"), &[embed, attn_dim]),
        k: g.parameter(&format!("{prefix}.self_attn.k"), &[embed, attn_dim]),
        v: g.parameter(&format!("{prefix}.self_attn.v"), &[embed, attn_dim]),
        o: g.parameter(&format!("{prefix}.self_attn.o"), &[attn_dim, embed]),
        ln2: g.parameter(&format!("{prefix}.pre_mlp_norm"), &[embed]),
        w_gate: g.parameter(&format!("{prefix}.mlp.w_gate"), &[embed, mlp_dim]),
        w_up: g.parameter(&format!("{prefix}.mlp.w_up"), &[embed, mlp_dim]),
        w_down: g.parameter(&format!("{prefix}.mlp.w_down"), &[mlp_dim, embed]),
    }
}

/// One depth layer's compute, given pre-registered params: pre-norm causal
/// self-attention (depth rel-pos buckets, no cross-attention) → pre-norm GeGLU
/// MLP, each residual.
fn depth_layer_forward(
    g: &mut Graph,
    cfg: &LlmConfig,
    x: NodeId,
    self_rel_pos_table: NodeId,
    p: &DepthLayerParams,
) -> NodeId {
    // --- 1. Causal self-attention over the RVQ levels (own rel-pos buckets) ---
    let h = g.rms_norm(x, p.ln1, cfg.layer_norm_eps);
    let q = g.matmul(h, p.q);
    let k = g.matmul(h, p.k);
    let v = g.matmul(h, p.v);
    let sa = g.full_attention_with_rel_pos_bias(
        q,
        k,
        v,
        self_rel_pos_table,
        cfg.num_heads,
        cfg.num_heads,
        cfg.head_dim,
        cfg.depth_rel_pos_num_buckets,
        cfg.depth_rel_pos_max_distance,
        false, // not bidirectional
        true,  // causal: levels decoded low → high
    );
    let sa_out = g.matmul(sa, p.o);
    let x = g.add(x, sa_out);

    // --- 2. GeGLU MLP (no cross-attention in the depth module) ---
    let h = g.rms_norm(x, p.ln2, cfg.layer_norm_eps);
    let gate = g.matmul(h, p.w_gate);
    let up = g.matmul(h, p.w_up);
    let ffn = g.geglu(gate, up);
    let ffn_out = g.matmul(ffn, p.w_down);
    g.add(x, ffn_out)
}

/// Pre-created handles for the whole depth sub-decoder (shared rel-pos table,
/// the depth layers, the shared final norm, and the logits projection).
struct DepthParams {
    rel: NodeId,
    layers: Vec<DepthLayerParams>,
    decoder_norm: NodeId,
    logits: NodeId,
}

fn register_depth_params(g: &mut Graph, cfg: &LlmConfig) -> DepthParams {
    let embed = cfg.embed_dim as usize;
    // One shared rel-pos table for the whole depth sub-decoder — the checkpoint
    // stores a single `depth_decoder.relpos_bias_depth.rel_embedding` `[heads,
    // 16]`, not one table per depth layer.
    let rel = g.parameter(
        "decoder.depth_decoder.rel_pos_bias_table",
        &[(cfg.num_heads * cfg.depth_rel_pos_num_buckets) as usize],
    );
    let layers = (0..cfg.num_depth_decoder_layers)
        .map(|i| register_depth_layer_params(g, cfg, &format!("decoder.depth_layers.{i}")))
        .collect();
    // Shared final norm (target.decoder.decoder_norm) then the dedicated,
    // non-tied logits projection (target.decoder.logits_dense.kernel).
    let decoder_norm = g.parameter("decoder.decoder_norm", &[embed]);
    let logits = g.parameter("decoder.logits_dense", &[embed, cfg.vocab_size as usize]);
    DepthParams {
        rel,
        layers,
        decoder_norm,
        logits,
    }
}

/// Depth-decoder forward for one frame, given pre-registered params: depth
/// layers → shared `decoder_norm` → non-tied `logits_dense` → `[levels, vocab]`.
fn depth_forward(g: &mut Graph, cfg: &LlmConfig, depth_inputs: NodeId, dp: &DepthParams) -> NodeId {
    let mut x = depth_inputs;
    for p in &dp.layers {
        x = depth_layer_forward(g, cfg, x, dp.rel, p);
    }
    let x = g.rms_norm(x, dp.decoder_norm, cfg.layer_norm_eps);
    g.matmul(x, dp.logits)
}

/// Build the per-frame depth decoder forward: the `[num_levels, embed]` depth
/// input sequence → 4 depth layers → final `decoder_norm` → `logits_dense`
/// projection → `[num_levels, vocab]` logits.
///
/// `depth_inputs` is the assembled `[num_levels, embed]` sequence for one frame.
/// Per `depthformer.py`'s `MultivariateDecoder`, it is
/// `concat([temporal_output[None], embed(levels 0..num_levels-2)], axis=levels)`
/// — i.e. the temporal per-frame vector at position 0, then the embeddings of
/// the already-decoded RVQ levels 0..14 (the RQ-Transformer prefix). The full
/// decoder ([`build_decoder`]) assembles this and runs the stack per frame; this
/// builder is the verifiable per-frame core.
///
/// **Logits are NOT weight-tied.** The checkpoint carries a dedicated
/// `decoder.logits_dense.kernel` `[embed, vocab]` alongside
/// `token_embedder.embedding` `[vocab, embed]` (two distinct tensors), and the
/// final `decoder_norm` is shared (it sits at `target.decoder.*`, above both the
/// temporal and depth sub-decoders, and is applied to the depth output before
/// the logits projection).
///
/// Output: logits `[num_levels, vocab_size]`.
pub fn build_depth_decoder_stack(g: &mut Graph, cfg: &LlmConfig, depth_inputs: NodeId) -> NodeId {
    let dp = register_depth_params(g, cfg);
    depth_forward(g, cfg, depth_inputs, &dp)
}

/// Slice rows `[start, start+len)` of a row-major `[rows, dim]` tensor →
/// `[len, dim]` (a contiguous row range; `slice_2d` over the H axis).
fn slice_rows(
    g: &mut Graph,
    x: NodeId,
    rows: usize,
    dim: usize,
    start: usize,
    len: usize,
) -> NodeId {
    let out = g.slice_2d(
        x,
        1,
        1,
        rows as u32,
        dim as u32,
        start as u32,
        (rows - start - len) as u32,
        0,
        0,
    );
    g.reshape(out, &[len, dim])
}

/// Concatenate two row-major tensors `[ra, dim]` and `[rb, dim]` along rows →
/// `[ra + rb, dim]` (channel-axis concat with `spatial = dim`).
fn concat_rows(g: &mut Graph, a: NodeId, b: NodeId, ra: usize, rb: usize, dim: usize) -> NodeId {
    let out = g.concat(a, b, 1, ra as u32, rb as u32, dim as u32);
    g.reshape(out, &[ra + rb, dim])
}

/// Build the full parallel ("teacher-forcing") Depthformer decoder forward:
/// SOS-padded RVQ token grid → per-frame logits `[num_frames * num_levels,
/// vocab]`. This joins [`build_temporal_decoder`]'s temporal path with the
/// per-frame depth stack, exactly per `depthformer.py`'s `MultivariateDecoder`:
///
/// 1. embed the padded grid once → `[(num_frames+1) * num_levels, embed]`;
/// 2. mean-pool the levels of padded frames `0..num_frames-1` → temporal input
///    `[num_frames, embed]`; run the temporal layers (shared rel-pos table,
///    cross-attending to `encoder_out`) → temporal states `[num_frames, embed]`;
/// 3. for each output frame `t`, assemble the depth input
///    `concat([temporal_state[t], embed(target-frame-t levels 0..num_levels-2)])`
///    `→ [num_levels, embed]` and run the depth stack → `[num_levels, vocab]`;
/// 4. concatenate the per-frame logits → `[num_frames * num_levels, vocab]`.
///
/// The depth stack is run **per frame** because its self-attention is causal
/// *within* a frame's levels (a flattened `[num_frames*num_levels, embed]` pass
/// would leak across frames); the depth params are registered once and shared
/// across frames. This parallel form is for teacher-forcing / verification —
/// production inference is the autoregressive step loop (KV-cache; future work).
///
/// `decoder_input_tokens` is the SOS-padded grid `[(num_frames+1) * num_levels]`
/// u32: frame 0 is the SOS frame, frames `1..=num_frames` are the targets.
///
/// Inputs:  `decoder_input_tokens` u32 `[(num_frames+1) * num_levels]`,
///          `encoder_out` `[enc_seq, embed]`.
/// Output:  logits `[num_frames * num_levels, vocab_size]`.
pub fn build_decoder(
    g: &mut Graph,
    cfg: &LlmConfig,
    decoder_input_tokens: NodeId,
    encoder_out: NodeId,
    num_frames: usize,
) -> NodeId {
    let embed = cfg.embed_dim as usize;
    let levels = cfg.num_levels as usize;
    let vocab = cfg.vocab_size as usize;
    let padded_rows = (num_frames + 1) * levels;

    let table = g.parameter("shared_token_embedder", &[vocab, embed]);
    let embedded = g.embedding(decoder_input_tokens, table); // [(F+1)*L, embed]

    // --- Temporal path: mean-pool padded frames 0..F-1 → [F, embed] ---
    let temporal_src = slice_rows(g, embedded, padded_rows, embed, 0, num_frames * levels);
    let mut x = build_level_mean_pool(g, cfg, temporal_src, num_frames);
    let trel = g.parameter(
        "decoder.temporal_decoder.rel_pos_bias_table",
        &[(cfg.num_heads * cfg.rel_pos_num_buckets) as usize],
    );
    for i in 0..cfg.num_temporal_decoder_layers {
        let prefix = format!("decoder.temporal_layers.{i}");
        x = build_decoder_layer(g, cfg, x, encoder_out, trel, &prefix);
    }
    let temporal_out = x; // [F, embed]

    // --- Depth path: per-frame, sharing one set of registered params ---
    let dp = register_depth_params(g, cfg);
    let mut logits: Option<NodeId> = None;
    for t in 0..num_frames {
        // Level 0: this frame's temporal state.
        let trow = slice_rows(g, temporal_out, num_frames, embed, t, 1); // [1, embed]
                                                                         // Levels 1..L-1: embeddings of target-frame-t levels 0..L-2 — i.e. padded
                                                                         // frame t+1's first L-1 rows (the RQ-Transformer teacher-forcing prefix).
        let prefix = slice_rows(
            g,
            embedded,
            padded_rows,
            embed,
            (t + 1) * levels,
            levels - 1,
        );
        let depth_in = concat_rows(g, trow, prefix, 1, levels - 1, embed); // [L, embed]
        let frame_logits = depth_forward(g, cfg, depth_in, &dp); // [L, vocab]
        logits = Some(match logits {
            None => frame_logits,
            Some(acc) => concat_rows(g, acc, frame_logits, t * levels, levels, vocab),
        });
    }
    logits.expect("num_frames must be ≥ 1")
}

/// Build one **incremental** (autoregressive) temporal decode step: given the
/// just-decoded frame's RVQ tokens, advance the temporal decoder by one frame
/// using a KV cache, returning that frame's temporal state `[1, embed]`.
///
/// This is the efficient inference counterpart to [`build_temporal_decoder`]'s
/// parallel forward: the graph is built once (fixed, single-frame shape) and
/// run once per frame, with the per-layer K/V caches persisting across `step()`
/// calls as mutable `parameter` buffers (the smollm2 decode pattern). The
/// self-attention reads the growing cache via [`Graph::cached_attention`];
/// cross-attention to the (fixed) encoder output is recomputed each step. The
/// self-attention applies the learned T5 rel-pos bias via
/// [`Graph::cached_attention_rel_pos`] (shared
/// `decoder.temporal_decoder.rel_pos_bias_table`), so the incremental step
/// matches the parallel forward bit-for-bit (not just at zero rel-pos).
///
/// Caches are named `decoder.temporal_kv_cache.{layer}.{k,v}` `[max_frames,
/// attn_dim]` — zero-initialise them before the first step. The layer weights
/// reuse the same names as [`build_decoder_layer`]
/// (`decoder.temporal_layers.{i}.*`), so one weight set drives both paths.
///
/// `head_dim` must be 64 (`cached_attention` reduces the dot over its 64-lane
/// workgroup) — the real Magenta-RT value.
///
/// Inputs:  `step_tokens` u32 `[num_levels]` (the input frame's RVQ tokens —
///          the SOS frame for the first step, else the previous decoded frame),
///          `encoder_out` `[enc_seq, embed]`, `kv_pos` u32 scalar (the frame
///          index / number of valid cache rows).
/// Output:  temporal state `[1, embed]` for this frame.
pub fn build_temporal_decode_step(
    g: &mut Graph,
    cfg: &LlmConfig,
    step_tokens: NodeId,
    encoder_out: NodeId,
    kv_pos: NodeId,
    max_frames: usize,
) -> NodeId {
    let embed = cfg.embed_dim as usize;
    let attn_dim = (cfg.num_heads * cfg.head_dim) as usize;
    let mlp_dim = cfg.mlp_dim as usize;

    // Mean-pool this frame's level embeddings → the temporal input [1, embed].
    let table = g.parameter("shared_token_embedder", &[cfg.vocab_size as usize, embed]);
    let embedded = g.embedding(step_tokens, table);
    let mut x = build_level_mean_pool(g, cfg, embedded, 1);

    // Shared temporal rel-pos table (same as the parallel path), applied inside
    // the cached self-attention so incremental decode matches the full forward.
    let rel = g.parameter(
        "decoder.temporal_decoder.rel_pos_bias_table",
        &[(cfg.num_heads * cfg.rel_pos_num_buckets) as usize],
    );

    for i in 0..cfg.num_temporal_decoder_layers {
        let prefix = format!("decoder.temporal_layers.{i}");

        // --- 1. Cached causal self-attention over frames ---
        let ln1 = g.parameter(&format!("{prefix}.pre_self_attn_norm"), &[embed]);
        let h = g.rms_norm(x, ln1, cfg.layer_norm_eps);
        let wq = g.parameter(&format!("{prefix}.self_attn.q"), &[embed, attn_dim]);
        let wk = g.parameter(&format!("{prefix}.self_attn.k"), &[embed, attn_dim]);
        let wv = g.parameter(&format!("{prefix}.self_attn.v"), &[embed, attn_dim]);
        let q = g.matmul(h, wq);
        let k = g.matmul(h, wk);
        let v = g.matmul(h, wv);
        let k_cache = g.parameter(
            &format!("decoder.temporal_kv_cache.{i}.k"),
            &[max_frames, attn_dim],
        );
        let v_cache = g.parameter(
            &format!("decoder.temporal_kv_cache.{i}.v"),
            &[max_frames, attn_dim],
        );
        // Write this frame's K/V into the cache at `kv_pos`, then attend to the
        // valid rows. Thread the updated-cache nodes into `cached_attention` so
        // the writes are scheduled before the read (a data dependency, rather
        // than relying on emit order).
        let k_cache = g.cache_write(k, k_cache, kv_pos);
        let v_cache = g.cache_write(v, v_cache, kv_pos);
        let sa = g.cached_attention_rel_pos(
            q,
            k_cache,
            v_cache,
            kv_pos,
            rel,
            cfg.num_heads,
            cfg.num_heads,
            cfg.head_dim,
            cfg.rel_pos_num_buckets,
            cfg.rel_pos_max_distance,
            false, // causal temporal self-attention
        );
        let wo = g.parameter(&format!("{prefix}.self_attn.o"), &[attn_dim, embed]);
        let sa_out = g.matmul(sa, wo);
        let x1 = g.add(x, sa_out);

        // --- 2. Cross-attention to the (fixed) encoder output ---
        let ln2 = g.parameter(&format!("{prefix}.pre_cross_attn_norm"), &[embed]);
        let h = g.rms_norm(x1, ln2, cfg.layer_norm_eps);
        let wcq = g.parameter(&format!("{prefix}.cross_attn.q"), &[embed, attn_dim]);
        let wck = g.parameter(&format!("{prefix}.cross_attn.k"), &[embed, attn_dim]);
        let wcv = g.parameter(&format!("{prefix}.cross_attn.v"), &[embed, attn_dim]);
        let cq = g.matmul(h, wcq);
        let ck = g.matmul(encoder_out, wck);
        let cv = g.matmul(encoder_out, wcv);
        let ca = g.cross_attention(cq, ck, cv, cfg.num_heads, cfg.num_heads, cfg.head_dim);
        let wco = g.parameter(&format!("{prefix}.cross_attn.o"), &[attn_dim, embed]);
        let ca_out = g.matmul(ca, wco);
        let x2 = g.add(x1, ca_out);

        // --- 3. GeGLU MLP ---
        let ln3 = g.parameter(&format!("{prefix}.pre_mlp_norm"), &[embed]);
        let h = g.rms_norm(x2, ln3, cfg.layer_norm_eps);
        let w_gate = g.parameter(&format!("{prefix}.mlp.w_gate"), &[embed, mlp_dim]);
        let w_up = g.parameter(&format!("{prefix}.mlp.w_up"), &[embed, mlp_dim]);
        let w_down = g.parameter(&format!("{prefix}.mlp.w_down"), &[mlp_dim, embed]);
        let gate = g.matmul(h, w_gate);
        let up = g.matmul(h, w_up);
        let ffn = g.geglu(gate, up);
        let ffn_out = g.matmul(ffn, w_down);
        x = g.add(x2, ffn_out);
    }
    x
}

/// Options for [`decode`]: sequence length, sampling, and classifier-free
/// guidance.
#[derive(Clone, Debug)]
pub struct DecodeOptions {
    /// Number of frames to generate (each yields `num_levels` RVQ tokens).
    pub num_frames: usize,
    /// Start-of-sequence token id (the frame-0 temporal input is all `sos_id`).
    pub sos_id: u32,
    /// CFG guidance weight `w`; the guided logits are `neg + w*(pos - neg)`.
    /// Only used when a negative temporal session is supplied.
    pub guidance_weight: f32,
    /// Sampling temperature; `<= 0` selects greedy argmax (deterministic).
    pub temperature: f32,
    /// Top-k for temperature sampling (ignored when greedy).
    pub top_k: usize,
    /// PRNG seed for temperature sampling.
    pub seed: u64,
}

impl DecodeOptions {
    /// Greedy (argmax) decode, no CFG.
    pub fn greedy(num_frames: usize, sos_id: u32) -> Self {
        Self {
            num_frames,
            sos_id,
            guidance_weight: 0.0,
            temperature: 0.0,
            top_k: 0,
            seed: 0,
        }
    }
}

/// Run one incremental temporal decode step → the frame's temporal state `[embed]`.
fn temporal_step(s: &mut Session, prev_frame: &[u32], t: usize, embed: usize) -> Vec<f32> {
    s.set_input_u32("step_tokens", prev_frame);
    s.set_input_u32("kv_pos", &[t as u32]);
    s.step();
    s.wait();
    s.read_output(embed)
}

/// Run the depth stack over a depth input → logits `[num_levels * vocab]`.
fn depth_logits(s: &mut Session, depth_in: &[f32], levels: usize, vocab: usize) -> Vec<f32> {
    s.set_input("depth_inputs", depth_in);
    s.step();
    s.wait();
    s.read_output(levels * vocab)
}

/// Host-driven autoregressive decode with optional classifier-free guidance and
/// temperature/top-k sampling. Returns the generated `[num_frames * num_levels]`
/// RVQ token grid.
///
/// Sessions (built once by the caller, weights + caches already set):
/// - `temporal_pos`: a [`build_temporal_decode_step`] graph conditioned on the
///   positive encoder output (set its `enc_out` and zero its KV caches first).
/// - `temporal_neg`: an optional second temporal session conditioned on the
///   negative (mask-style) encoder output, for CFG. `None` ⇒ no guidance.
/// - `depth`: a [`build_depth_decoder_stack`] graph (no persistent state, reused
///   for both the positive and negative depth passes).
///
/// CFG is done entirely host-side: the positive and negative passes are run as
/// two independent batch=1 decodes whose per-level logits are combined with
/// [`super::sampling::cfg_combine`] before sampling — so no graph-level batch=2
/// (or encoder broadcast-add) is required. The sampled token is fed back into
/// both passes so they stay in lock-step.
///
/// `embed_table` is a CPU copy of the `[vocab, embed]` shared token embedder,
/// used to embed sampled tokens into the depth input (level `q+1`'s slot).
pub fn decode(
    cfg: &LlmConfig,
    temporal_pos: &mut Session,
    mut temporal_neg: Option<&mut Session>,
    depth: &mut Session,
    embed_table: &[f32],
    opts: &DecodeOptions,
) -> Vec<u32> {
    let embed = cfg.embed_dim as usize;
    let levels = cfg.num_levels as usize;
    let vocab = cfg.vocab_size as usize;

    let mut rng = super::sampling::Xorshift64::new(opts.seed);
    let sample = |logits: &[f32], rng: &mut super::sampling::Xorshift64| -> u32 {
        if opts.temperature <= 0.0 {
            super::sampling::argmax(logits)
        } else {
            super::sampling::top_k_sample(logits, opts.temperature, opts.top_k, rng)
        }
    };

    let mut out = Vec::with_capacity(opts.num_frames * levels);
    // The temporal input for frame 0 is the SOS frame; thereafter the previous
    // decoded frame.
    let mut prev_frame = vec![opts.sos_id; levels];

    for t in 0..opts.num_frames {
        let pos_state = temporal_step(temporal_pos, &prev_frame, t, embed);
        let neg_state = temporal_neg
            .as_deref_mut()
            .map(|s| temporal_step(s, &prev_frame, t, embed));

        // Depth input position 0 is the temporal state; position q+1 is the
        // embedding of the just-decoded level q (shared across pos/neg).
        let mut depth_pos = vec![0.0_f32; levels * embed];
        depth_pos[0..embed].copy_from_slice(&pos_state);
        let mut depth_neg = neg_state.map(|st| {
            let mut d = vec![0.0_f32; levels * embed];
            d[0..embed].copy_from_slice(&st);
            d
        });

        let mut frame = Vec::with_capacity(levels);
        for q in 0..levels {
            let pos = depth_logits(depth, &depth_pos, levels, vocab);
            let pos_row = &pos[q * vocab..(q + 1) * vocab];
            let tok = match depth_neg.as_ref() {
                Some(dn) => {
                    let neg = depth_logits(depth, dn, levels, vocab);
                    let neg_row = &neg[q * vocab..(q + 1) * vocab];
                    let mut combined = vec![0.0_f32; vocab];
                    super::sampling::cfg_combine(
                        pos_row,
                        neg_row,
                        opts.guidance_weight,
                        &mut combined,
                    );
                    sample(&combined, &mut rng)
                }
                None => sample(pos_row, &mut rng),
            };
            frame.push(tok);
            if q + 1 < levels {
                let e = tok as usize * embed;
                let slot = (q + 1) * embed..(q + 2) * embed;
                depth_pos[slot.clone()].copy_from_slice(&embed_table[e..e + embed]);
                if let Some(dn) = depth_neg.as_mut() {
                    dn[slot].copy_from_slice(&embed_table[e..e + embed]);
                }
            }
        }
        out.extend_from_slice(&frame);
        prev_frame = frame;
    }
    out
}

/// Greedy (argmax) autoregressive decode, batch=1, no CFG — a thin wrapper over
/// [`decode`]. See [`decode`] for the session contract.
pub fn decode_greedy(
    cfg: &LlmConfig,
    temporal: &mut Session,
    depth: &mut Session,
    embed_table: &[f32],
    num_frames: usize,
    sos_id: u32,
) -> Vec<u32> {
    decode(
        cfg,
        temporal,
        None,
        depth,
        embed_table,
        &DecodeOptions::greedy(num_frames, sos_id),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_large_matches_gin() {
        let c = LlmConfig::large();
        assert_eq!(c.embed_dim, 1024);
        assert_eq!(c.num_heads * c.head_dim, c.embed_dim);
        assert_eq!(c.num_encoder_layers, 24);
        assert_eq!(
            c.num_temporal_decoder_layers + c.num_depth_decoder_layers,
            24
        );
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
    fn depth_decoder_stack_builds() {
        // Smoke test: the per-frame depth stack constructs without panicking.
        // GPU-vs-CPU correctness (à la tests/llm_decoder_stack_correctness.rs)
        // needs a Vulkan device; this only checks graph composition.
        let cfg = LlmConfig::base();
        let mut g = Graph::new();
        let depth_in = g.input(
            "depth_inputs",
            &[cfg.num_levels as usize, cfg.embed_dim as usize],
        );
        let _logits = build_depth_decoder_stack(&mut g, &cfg, depth_in);
        // 4 depth layers × (4 self-attn matmuls + 3 mlp matmuls + 2 norms) plus
        // 1 shared rel-pos table + decoder_norm + logits_dense params, and ops.
        assert!(
            g.nodes().len() > cfg.num_depth_decoder_layers as usize * 10,
            "depth stack should have ≥ ~42 parameter nodes plus ops, got {} nodes",
            g.nodes().len()
        );
    }

    #[test]
    fn encoder_graph_builds_with_expected_param_count() {
        // Smoke test: just confirm the graph constructs without panicking
        // and registers a sensible number of parameters.
        let cfg = LlmConfig::large();
        let mut g = Graph::new();
        let _out = build_encoder_graph(&mut g, &cfg, cfg.encoder_seq_len as usize);
        // Per-layer: 4 attn matmuls + 3 mlp matmuls + 2 layer norms = 9 params
        // × 24 layers, plus shared token embed + final norm (PE is a constant).
        assert!(
            g.nodes().len() > 24 * 9,
            "encoder graph should have ≥ ~218 parameter nodes plus ops, got {} nodes",
            g.nodes().len()
        );
    }
}
