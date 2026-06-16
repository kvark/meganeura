//! MusicCoCa text encoder: text prompt → 768-d style embedding → 12 RVQ tokens.
//!
//! Magenta-RT conditions generation on a style embedding produced by MusicCoCa
//! (a CoCa-style joint text/audio embedder). For text prompts we only need the
//! text tower + the RVQ quantizer; the LLM consumes the first 6 of the 12 RVQ
//! levels as `encoder_style_rvq_depth` tokens.
//!
//! ## Architecture (reverse-engineered; see `tools/magenta_rt/MUSICCOCA_FINDINGS.md`)
//!
//! A standard 12-layer pre-norm Transformer encoder followed by single-query
//! attention pooling, verified against the published `embed_text` SavedModel at
//! **0.9993 mean cosine** over the 26 reference prompts (93.3% end-to-end RVQ
//! token match). The forward pass mirrors `musiccoca_numpy_ref.py`:
//!
//! ```text
//! x = embed_table[ids] * sqrt(768)                 # token embed, CoCa sqrt-d scale
//! x = x + sinusoidal_pe(concat[sin, cos])          # absolute positions
//! for layer in 0..12:                              # pre-norm, GeLU MLP
//!     h = LayerNorm(x); attn = MHA(h) (12 heads × 64); x += O(attn)
//!     h = LayerNorm(x); x += Wo(gelu(Wi(h)))
//! pooled = CrossAttn(query=[1,768], kv=x)          # 12 heads × 256 head_dim
//! embed  = LayerNorm(Wo(pooled))                   # [1, 768] contrastive embedding
//! ```
//!
//! ### Padding
//! The SavedModel runs a fixed `seq=128` with a padding mask. Running each
//! prompt **unpadded** (`seq = real token count`) is mathematically identical
//! for the pooled embedding — masked-out positions contribute nothing to
//! attention and the pool — so no masked-attention op is needed.
//!
//! ### Three load-time weight transforms (so the graph stays plain)
//! 1. **sqrt(d) scale** folded into the embedding table: `table *= sqrt(768)`.
//! 2. **LayerNorm +1 offset**: flaxformer stores the trainable scale as a
//!    deviation from identity, so every LN scale loads as `arg + 1.0`.
//! 3. **RVQ codebook order**: the 12 codebooks are stored in *alphabetical*
//!    string order (`0,1,10,11,2,…`); reorder to numeric before quantizing
//!    (see [`numeric_codebook_order`]). Missing this drops the match to ~17%.
//!
//! ## `tf_var_leaves.N` → parameter-name mapping (text branch)
//! From `MUSICCOCA_FINDINGS.md`. The per-layer args are `[12, …]` stacked over
//! layers; a loader slices `[i]` for `text_encoder.layers.{i}`:
//!
//! | param name                              | tf_var_leaves arg | raw shape         |
//! |-----------------------------------------|-------------------|-------------------|
//! | `text_encoder.embed_table`              | 27 (transposed)   | `[768, 64000]`    |
//! | `…layers.{i}.pre_attn_norm.{scale,bias}`| 71, 70            | `[12, 768]`       |
//! | `…layers.{i}.attn.{q,k,v,o}.kernel`     | 77, 73, 79, 75    | `[12, 768, 12, 64]`|
//! | `…layers.{i}.attn.{q,k,v}.bias`         | 76, 72, 78        | `[12, 12, 64]`    |
//! | `…layers.{i}.attn.o.bias`               | 74                | `[12, 768]`       |
//! | `…layers.{i}.pre_mlp_norm.{scale,bias}` | 69, 68            | `[12, 768]`       |
//! | `…layers.{i}.mlp.wi.{kernel,bias}`      | 65, 64            | `[12,768,3072]`/`[12,3072]` |
//! | `…layers.{i}.mlp.wo.{kernel,bias}`      | 67, 66            | `[12,3072,768]`/`[12,768]` |
//! | `text_encoder.pool.query`               | 24                | `[1, 768]`        |
//! | `text_encoder.pool.{q,k,v,o}.kernel`    | 19, 14, 21, 17    | `[768, 12, 256]`  |
//! | `text_encoder.pool.{q,k,v,o}.bias`      | 18, 13, 20, 16    | `[12, 256]`/`[768]`|
//! | `text_encoder.final_norm.{scale,bias}`  | 23, 22            | `[768]`           |
//! | `quantizer.codebooks` (numeric order)   | 12× `[768, 1024]` | (alphabetical!)   |

use crate::graph::{Graph, NodeId};

/// MusicCoCa text-tower dimensions (Lyria Team paper §2.2 + dumped shapes).
#[derive(Clone, Debug)]
pub struct MusicCoCaConfig {
    /// Joint embedding dimension (768).
    pub embed_dim: u32,
    /// Encoder attention heads (12).
    pub num_heads: u32,
    /// Encoder head dimension (64); `num_heads * head_dim == embed_dim`.
    pub head_dim: u32,
    /// MLP hidden dimension (3072).
    pub mlp_dim: u32,
    /// Number of transformer layers (12).
    pub num_layers: u32,
    /// Attention-pool head dimension (256). The pool uses `num_heads` heads of
    /// this size (12 × 256 = 3072) rather than the encoder's 64.
    pub pool_head_dim: u32,
    /// Text vocabulary size (64000).
    pub vocab_size: u32,
    /// RVQ depth — number of codebooks / output tokens (12).
    pub rvq_depth: u32,
    /// RVQ codebook size (1024).
    pub codebook_size: u32,
    /// LayerNorm epsilon (1e-6, flaxformer default).
    pub layer_norm_eps: f32,
}

impl Default for MusicCoCaConfig {
    fn default() -> Self {
        Self {
            embed_dim: 768,
            num_heads: 12,
            head_dim: 64,
            mlp_dim: 3072,
            num_layers: 12,
            pool_head_dim: 256,
            vocab_size: 64000,
            rvq_depth: 12,
            codebook_size: 1024,
            layer_norm_eps: 1e-6,
        }
    }
}

/// One pre-norm transformer encoder layer: LayerNorm → MHA → residual →
/// LayerNorm → GeLU MLP → residual. Projection biases are added explicitly
/// (`bias_add`) since meganeura's matmul has no fused bias.
fn text_encoder_layer(
    g: &mut Graph,
    cfg: &MusicCoCaConfig,
    x: NodeId,
    prefix: &str,
) -> NodeId {
    let embed = cfg.embed_dim as usize;
    let attn_dim = (cfg.num_heads * cfg.head_dim) as usize;
    let mlp = cfg.mlp_dim as usize;

    // Pre-attention LayerNorm (scale carries the +1 offset from load).
    let ln1_w = g.parameter(&format!("{prefix}.pre_attn_norm.scale"), &[embed]);
    let ln1_b = g.parameter(&format!("{prefix}.pre_attn_norm.bias"), &[embed]);
    let h = g.layer_norm(x, ln1_w, ln1_b, cfg.layer_norm_eps);

    // Q/K/V projections [embed, attn_dim] + per-head bias [attn_dim].
    let proj = |g: &mut Graph, h: NodeId, name: &str| -> NodeId {
        let w = g.parameter(&format!("{prefix}.attn.{name}.kernel"), &[embed, attn_dim]);
        let b = g.parameter(&format!("{prefix}.attn.{name}.bias"), &[attn_dim]);
        let m = g.matmul(h, w);
        g.bias_add(m, b)
    };
    let q = proj(g, h, "q");
    let k = proj(g, h, "k");
    let v = proj(g, h, "v");
    let attn = g.full_attention(q, k, v, cfg.num_heads, cfg.num_heads, cfg.head_dim);

    // Output projection [attn_dim, embed] + bias [embed], then residual.
    let wo = g.parameter(&format!("{prefix}.attn.o.kernel"), &[attn_dim, embed]);
    let bo = g.parameter(&format!("{prefix}.attn.o.bias"), &[embed]);
    let attn_proj = g.matmul(attn, wo);
    let attn_out = g.bias_add(attn_proj, bo);
    let x = g.add(x, attn_out);

    // Pre-MLP LayerNorm → GeLU MLP → residual.
    let ln2_w = g.parameter(&format!("{prefix}.pre_mlp_norm.scale"), &[embed]);
    let ln2_b = g.parameter(&format!("{prefix}.pre_mlp_norm.bias"), &[embed]);
    let h = g.layer_norm(x, ln2_w, ln2_b, cfg.layer_norm_eps);
    let wi = g.parameter(&format!("{prefix}.mlp.wi.kernel"), &[embed, mlp]);
    let bi = g.parameter(&format!("{prefix}.mlp.wi.bias"), &[mlp]);
    let up = g.matmul(h, wi);
    let up = g.bias_add(up, bi);
    let act = g.gelu(up);
    let wo_mlp = g.parameter(&format!("{prefix}.mlp.wo.kernel"), &[mlp, embed]);
    let bo_mlp = g.parameter(&format!("{prefix}.mlp.wo.bias"), &[embed]);
    let down = g.matmul(act, wo_mlp);
    let mlp_out = g.bias_add(down, bo_mlp);
    g.add(x, mlp_out)
}

/// Build the text-encoder graph for a prompt of `seq_len` (unpadded) tokens.
///
/// Input:  `text_tokens` — u32 `[seq_len]` SentencePiece ids (SOS-prefixed).
/// Output: the `[1, embed_dim]` contrastive style embedding (pre-L2-norm).
/// Quantize it with [`rvq_quantize`] to get the 12 style tokens; the LLM uses
/// the first `encoder_style_rvq_depth` (6) of them.
pub fn build_text_encoder_graph(g: &mut Graph, cfg: &MusicCoCaConfig, seq_len: usize) -> NodeId {
    let embed = cfg.embed_dim as usize;

    let ids = g.input_u32("text_tokens", &[seq_len]);
    // Embedding table [vocab, embed]; the sqrt(embed) CoCa scale is folded in
    // at load, so the lookup output is already scaled.
    let table = g.parameter("text_encoder.embed_table", &[cfg.vocab_size as usize, embed]);
    let mut x = g.embedding(ids, table); // [seq_len, embed]

    // Add concatenated sin/cos positional encoding (deterministic constant).
    let pe = sinusoidal_pos_embedding(seq_len, embed);
    let pe_node = g.constant(pe, &[seq_len, embed]);
    x = g.add(x, pe_node);

    for i in 0..cfg.num_layers {
        let prefix = format!("text_encoder.layers.{i}");
        x = text_encoder_layer(g, cfg, x, &prefix);
    }

    // Attention pooling: one learned query attends over all encoder positions,
    // with 12 heads of pool_head_dim (256). No final encoder LN precedes this
    // (the only post-encoder LN is after the pool — confirmed via TFLite).
    let pool_dim = (cfg.num_heads * cfg.pool_head_dim) as usize;
    let query = g.parameter("text_encoder.pool.query", &[1, embed]);
    let pool_proj = |g: &mut Graph, src: NodeId, name: &str| -> NodeId {
        let w = g.parameter(&format!("text_encoder.pool.{name}.kernel"), &[embed, pool_dim]);
        let b = g.parameter(&format!("text_encoder.pool.{name}.bias"), &[pool_dim]);
        let m = g.matmul(src, w);
        g.bias_add(m, b)
    };
    let pool_q = pool_proj(g, query, "q"); // [1, pool_dim]
    let pool_k = pool_proj(g, x, "k"); // [seq_len, pool_dim]
    let pool_v = pool_proj(g, x, "v"); // [seq_len, pool_dim]
    let pooled = g.cross_attention(
        pool_q,
        pool_k,
        pool_v,
        cfg.num_heads,
        cfg.num_heads,
        cfg.pool_head_dim,
    ); // [1, pool_dim]
    let po_w = g.parameter("text_encoder.pool.o.kernel", &[pool_dim, embed]);
    let po_b = g.parameter("text_encoder.pool.o.bias", &[embed]);
    let po = g.matmul(pooled, po_w);
    let pool_out = g.bias_add(po, po_b); // [1, embed]

    // Final LayerNorm at the end of embed_text (scale carries +1 offset).
    let fln_w = g.parameter("text_encoder.final_norm.scale", &[embed]);
    let fln_b = g.parameter("text_encoder.final_norm.bias", &[embed]);
    g.layer_norm(pool_out, fln_w, fln_b, cfg.layer_norm_eps) // [1, embed]
}

/// Concatenated sinusoidal positional encoding: `[sin(pos·invf), cos(pos·invf)]`
/// where `invf[i] = 10000^(-2i/dim)`, `i in 0..dim/2`. Layout is `[seq_len, dim]`
/// row-major. **Concatenated, not interleaved** — confirmed against the TFLite
/// `CONCATENATION(sin, cos)` node.
pub fn sinusoidal_pos_embedding(seq_len: usize, dim: usize) -> Vec<f32> {
    assert!(dim.is_multiple_of(2), "pos-embedding dim must be even");
    let half = dim / 2;
    let mut pe = vec![0.0_f32; seq_len * dim];
    for pos in 0..seq_len {
        for i in 0..half {
            let inv_freq = 10000_f32.powf(-2.0 * i as f32 / dim as f32);
            let angle = pos as f32 * inv_freq;
            pe[pos * dim + i] = angle.sin();
            pe[pos * dim + half + i] = angle.cos();
        }
    }
    pe
}

/// Permutation mapping numeric RVQ level → index in the alphabetically-sorted
/// codebook dump. The safetensors stores levels keyed by stringified index, so
/// `sorted()` yields `0,1,10,11,2,3,…,9`. `numeric_codebook_order()[level]` is
/// the position of `level`'s codebook in that sorted list.
pub fn numeric_codebook_order(depth: usize) -> Vec<usize> {
    let mut keys: Vec<usize> = (0..depth).collect();
    keys.sort_by_key(|k| k.to_string());
    // keys[sorted_pos] = numeric_level; invert to numeric_level -> sorted_pos.
    let mut order = vec![0usize; depth];
    for (sorted_pos, &numeric_level) in keys.iter().enumerate() {
        order[numeric_level] = sorted_pos;
    }
    order
}

/// Residual vector quantization of a `dim`-length embedding into `depth` tokens.
///
/// `codebooks` is `depth` concatenated `[dim, codebook_size]` blocks **in
/// numeric level order** (apply [`numeric_codebook_order`] to the raw dump
/// first). At each level the nearest centroid (Euclidean) is selected, its
/// index emitted, and its vector subtracted from the running residual.
pub fn rvq_quantize(
    embedding: &[f32],
    codebooks: &[f32],
    depth: usize,
    dim: usize,
    codebook_size: usize,
) -> Vec<u32> {
    assert_eq!(embedding.len(), dim, "embedding length must equal dim");
    assert_eq!(
        codebooks.len(),
        depth * dim * codebook_size,
        "codebooks must be depth × dim × codebook_size"
    );
    let mut residual = embedding.to_vec();
    let mut tokens = Vec::with_capacity(depth);
    for level in 0..depth {
        let cb = &codebooks[level * dim * codebook_size..(level + 1) * dim * codebook_size];
        // cb is [dim, codebook_size]; centroid e's d-th component = cb[d*cs + e].
        let mut best_idx = 0usize;
        let mut best_dist = f32::INFINITY;
        for e in 0..codebook_size {
            let mut dist = 0.0_f32;
            for d in 0..dim {
                let diff = residual[d] - cb[d * codebook_size + e];
                dist += diff * diff;
            }
            if dist < best_dist {
                best_dist = dist;
                best_idx = e;
            }
        }
        tokens.push(best_idx as u32);
        for d in 0..dim {
            residual[d] -= cb[d * codebook_size + best_idx];
        }
    }
    tokens
}

/// L2-normalize a vector (the `contrastive_txt_embed_l2_normalized` output).
pub fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter().map(|x| x / norm).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pos_embedding_first_position_is_sin0_cos1() {
        let dim = 8;
        let pe = sinusoidal_pos_embedding(3, dim);
        // pos 0: sin(0)=0 for the first half, cos(0)=1 for the second half.
        for i in 0..dim / 2 {
            assert!(pe[i].abs() < 1e-6, "sin half at pos0 should be 0");
            assert!((pe[dim / 2 + i] - 1.0).abs() < 1e-6, "cos half at pos0 should be 1");
        }
        // pos 1, channel 0: sin(1*1)=sin(1); cos half channel 0: cos(1).
        assert!((pe[dim + 0] - 1.0_f32.sin()).abs() < 1e-6);
        assert!((pe[dim + dim / 2] - 1.0_f32.cos()).abs() < 1e-6);
    }

    #[test]
    fn pos_embedding_layout_is_concatenated_not_interleaved() {
        // Highest-frequency (i=0) and a lower one (i=1) must live in distinct
        // first/second halves, not interleaved pairs.
        let dim = 4;
        let pe = sinusoidal_pos_embedding(2, dim);
        // Row for pos=1: [sin(a0), sin(a1), cos(a0), cos(a1)].
        let a0 = 1.0_f32 * 10000_f32.powf(0.0);
        let a1 = 1.0_f32 * 10000_f32.powf(-2.0 / 4.0);
        assert!((pe[4 + 0] - a0.sin()).abs() < 1e-6);
        assert!((pe[4 + 1] - a1.sin()).abs() < 1e-6);
        assert!((pe[4 + 2] - a0.cos()).abs() < 1e-6);
        assert!((pe[4 + 3] - a1.cos()).abs() < 1e-6);
    }

    #[test]
    fn codebook_order_is_alphabetical_to_numeric() {
        // For depth 12 the dump's sorted order is 0,1,10,11,2,3,4,5,6,7,8,9.
        let order = numeric_codebook_order(12);
        assert_eq!(order[0], 0); // "0" sorts first
        assert_eq!(order[1], 1); // "1" sorts second
        assert_eq!(order[2], 4); // "2" sorts after "10","11"
        assert_eq!(order[10], 2); // "10" sorts third
        assert_eq!(order[11], 3); // "11" sorts fourth
        assert_eq!(order[9], 11); // "9" sorts last
    }

    #[test]
    fn rvq_quantize_picks_nearest_then_residualizes() {
        // dim=2, depth=2, codebook_size=2. Level 0 centroids: e0=(1,0) e1=(0,4).
        // Level 1 centroids: e0=(0.1,0) e1=(0,0.1).
        // embed=(1.05, 0). Level 0: nearest is e0=(1,0) → token 0; residual=(0.05,0).
        // Level 1: nearest is e0=(0.1,0) → token 0.
        let dim = 2;
        let cs = 2;
        // [dim, codebook_size] per level, d-major: cb[d*cs + e].
        let cb0 = [/*d0*/ 1.0, 0.0, /*d1*/ 0.0, 4.0]; // e0=(1,0) e1=(0,4)
        let cb1 = [/*d0*/ 0.1, 0.0, /*d1*/ 0.0, 0.1]; // e0=(0.1,0) e1=(0,0.1)
        let codebooks: Vec<f32> = cb0.iter().chain(cb1.iter()).copied().collect();
        let tokens = rvq_quantize(&[1.05, 0.0], &codebooks, 2, dim, cs);
        assert_eq!(tokens, vec![0, 0]);
    }

    #[test]
    fn l2_normalize_unit_norm() {
        let out = l2_normalize(&[3.0, 4.0]);
        assert!((out[0] - 0.6).abs() < 1e-6);
        assert!((out[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn text_encoder_graph_builds_with_expected_output_shape() {
        let cfg = MusicCoCaConfig::default();
        let mut g = Graph::new();
        let out = build_text_encoder_graph(&mut g, &cfg, 5);
        g.set_outputs(vec![out]);
        // Contrastive style embedding is [1, 768].
        assert_eq!(g.node(out).ty.shape, vec![1, cfg.embed_dim as usize]);
    }

    #[test]
    fn text_encoder_graph_lowers_to_a_plan() {
        // Compiling exercises op→dispatch lowering for the whole encoder
        // (embedding, layer_norm, full/cross attention, gelu MLP), catching
        // integration issues the graph-build shape asserts don't. No GPU.
        let cfg = MusicCoCaConfig::default();
        let mut g = Graph::new();
        let out = build_text_encoder_graph(&mut g, &cfg, 5);
        g.set_outputs(vec![out]);
        let plan = crate::compile::compile(&g);
        assert!(!plan.dispatches.is_empty());
        assert!(plan.input_buffers.iter().any(|(n, _)| n == "text_tokens"));
    }
}
