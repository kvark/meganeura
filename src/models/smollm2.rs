//! SmolLM2 model definition for meganeura.
//!
//! Builds the computation graph for SmolLM2-135M inference.
//! Architecture: decoder-only transformer with GQA, RoPE, RMSNorm, SwiGLU.

use crate::graph::{Graph, NodeId};

/// Hyperparameters for a SmolLM2 model instance.
///
/// Values correspond to the `config.json` published alongside the
/// HuggingFace model weights.
pub struct SmolLM2Config {
    /// Vocabulary size (number of token embeddings).
    pub vocab_size: usize,
    /// Dimensionality of the transformer hidden state.
    pub hidden_size: usize,
    /// Number of transformer decoder blocks.
    pub num_hidden_layers: usize,
    /// Number of query heads in grouped-query attention (GQA).
    pub num_attention_heads: u32,
    /// Number of key/value heads (fewer than query heads for GQA).
    pub num_key_value_heads: u32,
    /// Inner dimension of the SwiGLU feed-forward network.
    pub intermediate_size: usize,
    /// Epsilon for RMSNorm numerical stability.
    pub rms_norm_eps: f32,
    /// Base frequency for Rotary Position Embeddings (RoPE).
    pub rope_theta: f32,
}

impl SmolLM2Config {
    /// SmolLM2-135M configuration.
    pub fn smollm2_135m() -> Self {
        Self {
            vocab_size: 49152,
            hidden_size: 576,
            num_hidden_layers: 30,
            num_attention_heads: 9,
            num_key_value_heads: 3,
            intermediate_size: 1536,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
        }
    }

    pub fn head_dim(&self) -> u32 {
        self.hidden_size as u32 / self.num_attention_heads
    }

    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads as usize * self.head_dim() as usize
    }
}

/// Build the SmolLM2 inference graph.
///
/// Returns the logits output node ID. The graph expects:
/// - Input "token_ids": U32 tensor of shape `[seq_len]`
/// - Parameters named following the safetensors convention:
///   `model.embed_tokens.weight`, `model.layers.{i}.input_layernorm.weight`, etc.
pub fn build_graph(g: &mut Graph, config: &SmolLM2Config, seq_len: usize) -> NodeId {
    let hidden = config.hidden_size;
    let kv_dim = config.kv_dim();
    let ffn = config.intermediate_size;
    let eps = config.rms_norm_eps;
    let theta = config.rope_theta;

    // Token embedding
    let token_ids = g.input_u32("token_ids", &[seq_len]);
    let embed_weight = g.parameter("model.embed_tokens.weight", &[config.vocab_size, hidden]);
    let mut x = g.embedding(token_ids, embed_weight);

    // Transformer layers
    for i in 0..config.num_hidden_layers {
        let prefix = format!("model.layers.{}", i);

        // Pre-attention RMSNorm
        let ln1_w = g.parameter(&format!("{}.input_layernorm.weight", prefix), &[hidden]);
        let h = g.rms_norm(x, ln1_w, eps);

        // QKV projections (weights are [out, in] in HF, we transpose during loading)
        let wq = g.parameter(
            &format!("{}.self_attn.q_proj.weight", prefix),
            &[hidden, hidden],
        );
        let wk = g.parameter(
            &format!("{}.self_attn.k_proj.weight", prefix),
            &[hidden, kv_dim],
        );
        let wv = g.parameter(
            &format!("{}.self_attn.v_proj.weight", prefix),
            &[hidden, kv_dim],
        );

        let q = g.matmul(h, wq); // [seq, hidden]
        let k = g.matmul(h, wk); // [seq, kv_dim]
        let v = g.matmul(h, wv); // [seq, kv_dim]

        // RoPE
        let q = g.rope(q, theta);
        let k = g.rope(k, theta);

        // Causal attention with GQA
        let attn = g.causal_attention(
            q,
            k,
            v,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim(),
        );

        // Output projection
        let wo = g.parameter(
            &format!("{}.self_attn.o_proj.weight", prefix),
            &[hidden, hidden],
        );
        let attn_out = g.matmul(attn, wo);

        // Residual connection
        x = g.add(x, attn_out);

        // Post-attention RMSNorm
        let ln2_w = g.parameter(
            &format!("{}.post_attention_layernorm.weight", prefix),
            &[hidden],
        );
        let h = g.rms_norm(x, ln2_w, eps);

        // SwiGLU FFN
        let w_gate = g.parameter(&format!("{}.mlp.gate_proj.weight", prefix), &[hidden, ffn]);
        let w_up = g.parameter(&format!("{}.mlp.up_proj.weight", prefix), &[hidden, ffn]);
        let w_down = g.parameter(&format!("{}.mlp.down_proj.weight", prefix), &[ffn, hidden]);

        let gate = g.matmul(h, w_gate); // [seq, ffn]
        let gate = g.silu(gate);
        let up = g.matmul(h, w_up); // [seq, ffn]
        let ffn_out = g.mul(gate, up);
        let ffn_out = g.matmul(ffn_out, w_down); // [seq, hidden]

        // Residual connection
        x = g.add(x, ffn_out);
    }

    // Final RMSNorm
    let final_ln_w = g.parameter("model.norm.weight", &[hidden]);
    x = g.rms_norm(x, final_ln_w, eps);

    // LM head (often tied to embed_tokens, loaded separately)
    let lm_head = g.parameter("lm_head.weight", &[hidden, config.vocab_size]);
    g.matmul(x, lm_head) // [seq, vocab]
}

/// Get all weight parameter names for SmolLM2.
pub fn weight_names(config: &SmolLM2Config) -> Vec<String> {
    let mut names = Vec::new();
    names.push("model.embed_tokens.weight".to_string());

    for i in 0..config.num_hidden_layers {
        let p = format!("model.layers.{}", i);
        names.push(format!("{}.input_layernorm.weight", p));
        names.push(format!("{}.self_attn.q_proj.weight", p));
        names.push(format!("{}.self_attn.k_proj.weight", p));
        names.push(format!("{}.self_attn.v_proj.weight", p));
        names.push(format!("{}.self_attn.o_proj.weight", p));
        names.push(format!("{}.post_attention_layernorm.weight", p));
        names.push(format!("{}.mlp.gate_proj.weight", p));
        names.push(format!("{}.mlp.up_proj.weight", p));
        names.push(format!("{}.mlp.down_proj.weight", p));
    }

    names.push("model.norm.weight".to_string());
    names.push("lm_head.weight".to_string());
    names
}

/// Names of weight tensors that need transposing (linear layer weights).
pub fn transposed_weight_names(config: &SmolLM2Config) -> Vec<String> {
    let mut names = Vec::new();
    for i in 0..config.num_hidden_layers {
        let p = format!("model.layers.{}", i);
        names.push(format!("{}.self_attn.q_proj.weight", p));
        names.push(format!("{}.self_attn.k_proj.weight", p));
        names.push(format!("{}.self_attn.v_proj.weight", p));
        names.push(format!("{}.self_attn.o_proj.weight", p));
        names.push(format!("{}.mlp.gate_proj.weight", p));
        names.push(format!("{}.mlp.up_proj.weight", p));
        names.push(format!("{}.mlp.down_proj.weight", p));
    }
    names
}
