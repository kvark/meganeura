//! Mistral model definition for meganeura.
//!
//! Builds the computation graph for Mistral inference.
//! Architecture: decoder-only transformer with GQA, RoPE, RMSNorm, SwiGLU,
//! and sliding-window attention on all layers.

use crate::graph::{Graph, NodeId};

/// Hyperparameters for a Mistral model instance.
pub struct MistralConfig {
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Hidden state dimensionality.
    pub hidden_size: usize,
    /// Number of transformer layers.
    pub num_hidden_layers: usize,
    /// Number of query heads (GQA).
    pub num_attention_heads: u32,
    /// Number of key/value heads.
    pub num_key_value_heads: u32,
    /// SwiGLU FFN inner dimension.
    pub intermediate_size: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// RoPE base frequency.
    pub rope_theta: f32,
    /// Sliding window size for attention.
    pub sliding_window_size: u32,
}

impl MistralConfig {
    /// Mistral-7B-v0.3 configuration.
    pub fn mistral_7b() -> Self {
        Self {
            vocab_size: 32768,
            hidden_size: 4096,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            intermediate_size: 14336,
            rms_norm_eps: 1e-5,
            rope_theta: 1_000_000.0,
            sliding_window_size: 4096,
        }
    }

    /// Mistral-Nemo-12B configuration.
    pub fn mistral_nemo_12b() -> Self {
        Self {
            vocab_size: 131072,
            hidden_size: 5120,
            num_hidden_layers: 40,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            intermediate_size: 14336,
            rms_norm_eps: 1e-5,
            rope_theta: 1_000_000.0,
            sliding_window_size: 4096,
        }
    }

    /// Tiny configuration for unit tests.
    pub fn small_test() -> Self {
        Self {
            vocab_size: 64,
            hidden_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            intermediate_size: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            sliding_window_size: 8,
        }
    }

    pub fn head_dim(&self) -> u32 {
        self.hidden_size as u32 / self.num_attention_heads
    }

    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads as usize * self.head_dim() as usize
    }
}

/// Build the Mistral inference graph.
///
/// Returns the logits output node ID. Uses sliding-window attention on all layers.
pub fn build_graph(g: &mut Graph, config: &MistralConfig, seq_len: usize) -> NodeId {
    let hidden = config.hidden_size;
    let kv_dim = config.kv_dim();
    let ffn = config.intermediate_size;
    let eps = config.rms_norm_eps;
    let theta = config.rope_theta;
    let head_dim = config.head_dim();

    let token_ids = g.input_u32("token_ids", &[seq_len]);
    let embed_weight = g.parameter("model.embed_tokens.weight", &[config.vocab_size, hidden]);
    let mut x = g.embedding(token_ids, embed_weight);

    for i in 0..config.num_hidden_layers {
        let prefix = format!("model.layers.{}", i);

        let ln1_w = g.parameter(&format!("{}.input_layernorm.weight", prefix), &[hidden]);
        let h = g.rms_norm(x, ln1_w, eps);

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

        let q = g.matmul(h, wq);
        let k = g.matmul(h, wk);
        let v = g.matmul(h, wv);

        let q = g.rope(q, theta, head_dim);
        let k = g.rope(k, theta, head_dim);

        let attn = g.sliding_window_attention(
            q,
            k,
            v,
            config.num_attention_heads,
            config.num_key_value_heads,
            head_dim,
            config.sliding_window_size,
        );

        let wo = g.parameter(
            &format!("{}.self_attn.o_proj.weight", prefix),
            &[hidden, hidden],
        );
        let attn_out = g.matmul(attn, wo);
        x = g.add(x, attn_out);

        let ln2_w = g.parameter(
            &format!("{}.post_attention_layernorm.weight", prefix),
            &[hidden],
        );
        let h = g.rms_norm(x, ln2_w, eps);

        let w_gate = g.parameter(&format!("{}.mlp.gate_proj.weight", prefix), &[hidden, ffn]);
        let w_up = g.parameter(&format!("{}.mlp.up_proj.weight", prefix), &[hidden, ffn]);
        let w_down = g.parameter(&format!("{}.mlp.down_proj.weight", prefix), &[ffn, hidden]);

        let gate = g.matmul(h, w_gate);
        let up = g.matmul(h, w_up);
        let ffn_out = g.swiglu(gate, up);
        let ffn_out = g.matmul(ffn_out, w_down);

        x = g.add(x, ffn_out);
    }

    let final_ln_w = g.parameter("model.norm.weight", &[hidden]);
    x = g.rms_norm(x, final_ln_w, eps);

    let lm_head = g.parameter("lm_head.weight", &[hidden, config.vocab_size]);
    g.matmul(x, lm_head)
}

/// Get all weight parameter names for Mistral.
pub fn weight_names(config: &MistralConfig) -> Vec<String> {
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

/// Names of weight tensors that need transposing.
pub fn transposed_weight_names(config: &MistralConfig) -> Vec<String> {
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
