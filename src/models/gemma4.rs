//! Gemma-4 model definition for meganeura.
//!
//! Builds the computation graph for Google Gemma-4 text decoder inference.
//! Architecture: decoder-only transformer with GQA, RoPE, RMSNorm, SwiGLU,
//! hybrid sliding-window / global attention, QK-norm, and logit soft-capping.

use crate::graph::{Graph, NodeId};

/// Hyperparameters for a Gemma-4 model instance.
///
/// Values correspond to the `config.json` published alongside the
/// HuggingFace model weights (e.g. `google/gemma-4-12b-pt`).
pub struct Gemma4Config {
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
    /// Sliding window size for local attention layers.
    pub sliding_window_size: u32,
    /// Period for global attention layers. Every `global_attn_period`-th layer
    /// (starting from `global_attn_offset`) uses full causal attention; all
    /// other layers use sliding-window attention.
    pub global_attn_period: usize,
    /// Layer offset for the first global attention layer.
    pub global_attn_offset: usize,
    /// Whether to apply RMSNorm to Q and K before attention (QK-norm).
    pub use_qk_norm: bool,
}

impl Gemma4Config {
    /// Gemma-4-1B configuration.
    pub fn gemma4_1b() -> Self {
        Self {
            vocab_size: 262144,
            hidden_size: 1856,
            num_hidden_layers: 26,
            num_attention_heads: 14,
            num_key_value_heads: 7,
            intermediate_size: 7424,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            sliding_window_size: 4096,
            global_attn_period: 4,
            global_attn_offset: 3,
            use_qk_norm: true,
        }
    }

    /// Gemma-4-4B configuration.
    pub fn gemma4_4b() -> Self {
        Self {
            vocab_size: 262144,
            hidden_size: 2560,
            num_hidden_layers: 34,
            num_attention_heads: 20,
            num_key_value_heads: 10,
            intermediate_size: 10240,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            sliding_window_size: 4096,
            global_attn_period: 4,
            global_attn_offset: 3,
            use_qk_norm: true,
        }
    }

    /// Gemma-4-12B configuration.
    pub fn gemma4_12b() -> Self {
        Self {
            vocab_size: 262144,
            hidden_size: 3840,
            num_hidden_layers: 48,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            intermediate_size: 15360,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            sliding_window_size: 4096,
            global_attn_period: 4,
            global_attn_offset: 3,
            use_qk_norm: true,
        }
    }

    /// Gemma-4-27B configuration.
    pub fn gemma4_27b() -> Self {
        Self {
            vocab_size: 262144,
            hidden_size: 4608,
            num_hidden_layers: 62,
            num_attention_heads: 32,
            num_key_value_heads: 16,
            intermediate_size: 18432,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            sliding_window_size: 4096,
            global_attn_period: 4,
            global_attn_offset: 3,
            use_qk_norm: true,
        }
    }

    /// Tiny configuration for unit tests (fast compilation and execution).
    pub fn small_test() -> Self {
        Self {
            vocab_size: 64,
            hidden_size: 32,
            num_hidden_layers: 4,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            intermediate_size: 64,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            sliding_window_size: 8,
            global_attn_period: 4,
            global_attn_offset: 3,
            use_qk_norm: true,
        }
    }

    pub fn head_dim(&self) -> u32 {
        self.hidden_size as u32 / self.num_attention_heads
    }

    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads as usize * self.head_dim() as usize
    }

    fn is_global_layer(&self, layer: usize) -> bool {
        layer >= self.global_attn_offset
            && (layer - self.global_attn_offset) % self.global_attn_period == 0
    }
}

/// Build the Gemma-4 inference graph.
///
/// Returns the logits output node ID. The graph expects:
/// - Input "token_ids": U32 tensor of shape `[seq_len]`
/// - Parameters named following the HuggingFace safetensors convention:
///   `model.embed_tokens.weight`, `model.layers.{i}.input_layernorm.weight`, etc.
pub fn build_graph(g: &mut Graph, config: &Gemma4Config, seq_len: usize) -> NodeId {
    let hidden = config.hidden_size;
    let kv_dim = config.kv_dim();
    let ffn = config.intermediate_size;
    let eps = config.rms_norm_eps;
    let theta = config.rope_theta;
    let head_dim = config.head_dim();

    // Token embedding
    let token_ids = g.input_u32("token_ids", &[seq_len]);
    let embed_weight = g.parameter("model.embed_tokens.weight", &[config.vocab_size, hidden]);
    let mut x = g.embedding(token_ids, embed_weight);

    // Transformer layers
    for i in 0..config.num_hidden_layers {
        let prefix = format!("model.layers.{}", i);
        let is_global = config.is_global_layer(i);

        // Pre-attention RMSNorm
        let ln1_w = g.parameter(&format!("{}.input_layernorm.weight", prefix), &[hidden]);
        let h = g.rms_norm(x, ln1_w, eps);

        // QKV projections
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

        // QK-norm: apply RMSNorm to Q and K before RoPE
        let (q, k) = if config.use_qk_norm {
            let qn_w = g.parameter(&format!("{}.self_attn.q_norm.weight", prefix), &[hidden]);
            let kn_w = g.parameter(&format!("{}.self_attn.k_norm.weight", prefix), &[kv_dim]);
            (g.rms_norm(q, qn_w, eps), g.rms_norm(k, kn_w, eps))
        } else {
            (q, k)
        };

        // RoPE
        let q = g.rope(q, theta, head_dim);
        let k = g.rope(k, theta, head_dim);

        // Attention: global (full causal) or sliding-window
        let attn = if is_global {
            g.causal_attention(
                q,
                k,
                v,
                config.num_attention_heads,
                config.num_key_value_heads,
                head_dim,
            )
        } else {
            g.sliding_window_attention(
                q,
                k,
                v,
                config.num_attention_heads,
                config.num_key_value_heads,
                head_dim,
                config.sliding_window_size,
            )
        };

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
        let up = g.matmul(h, w_up); // [seq, ffn]
        let ffn_out = g.swiglu(gate, up);
        let ffn_out = g.matmul(ffn_out, w_down); // [seq, hidden]

        // Residual connection
        x = g.add(x, ffn_out);
    }

    // Final RMSNorm
    let final_ln_w = g.parameter("model.norm.weight", &[hidden]);
    x = g.rms_norm(x, final_ln_w, eps);

    // LM head
    let lm_head = g.parameter("lm_head.weight", &[hidden, config.vocab_size]);
    g.matmul(x, lm_head) // [seq, vocab]
}

/// Get all weight parameter names for Gemma-4.
pub fn weight_names(config: &Gemma4Config) -> Vec<String> {
    let mut names = Vec::new();
    names.push("model.embed_tokens.weight".to_string());

    for i in 0..config.num_hidden_layers {
        let p = format!("model.layers.{}", i);
        names.push(format!("{}.input_layernorm.weight", p));
        names.push(format!("{}.self_attn.q_proj.weight", p));
        names.push(format!("{}.self_attn.k_proj.weight", p));
        names.push(format!("{}.self_attn.v_proj.weight", p));
        names.push(format!("{}.self_attn.o_proj.weight", p));
        if config.use_qk_norm {
            names.push(format!("{}.self_attn.q_norm.weight", p));
            names.push(format!("{}.self_attn.k_norm.weight", p));
        }
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
pub fn transposed_weight_names(config: &Gemma4Config) -> Vec<String> {
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
    names.push("lm_head.weight".to_string());
    names
}
