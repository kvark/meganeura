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
        let q = g.rope(q, theta, config.head_dim());
        let k = g.rope(k, theta, config.head_dim());

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
        let up = g.matmul(h, w_up); // [seq, ffn]
        let ffn_out = g.swiglu(gate, up);
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

/// Build the SmolLM2 prefill graph.
///
/// Processes the full prompt and outputs logits plus K/V tensors per layer
/// as graph outputs for initializing the decode cache.
///
/// Returns (logits, k_outputs, v_outputs) where k/v_outputs are per-layer.
pub fn build_prefill_graph(
    g: &mut Graph,
    config: &SmolLM2Config,
    seq_len: usize,
) -> (NodeId, Vec<NodeId>, Vec<NodeId>) {
    let hidden = config.hidden_size;
    let kv_dim = config.kv_dim();
    let ffn = config.intermediate_size;
    let eps = config.rms_norm_eps;
    let theta = config.rope_theta;

    let token_ids = g.input_u32("token_ids", &[seq_len]);
    let embed_weight = g.parameter("model.embed_tokens.weight", &[config.vocab_size, hidden]);
    let mut x = g.embedding(token_ids, embed_weight);

    let mut k_outputs = Vec::new();
    let mut v_outputs = Vec::new();

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

        let q = g.rope(q, theta, config.head_dim());
        let k = g.rope(k, theta, config.head_dim());

        // Save K/V for cache initialization
        k_outputs.push(k);
        v_outputs.push(v);

        let attn = g.causal_attention(
            q,
            k,
            v,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim(),
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
    let logits = g.matmul(x, lm_head);

    (logits, k_outputs, v_outputs)
}

/// Build the SmolLM2 decode graph (single-token with KV cache).
///
/// Processes 1 token, reads/writes KV cache, returns logits for 1 position.
/// `max_seq_len` is the pre-allocated cache size.
///
/// Returns (logits, k_cache_params, v_cache_params) where cache params are
/// the parameter NodeIds for pre-allocated cache buffers.
pub fn build_decode_graph(
    g: &mut Graph,
    config: &SmolLM2Config,
    max_seq_len: usize,
) -> (NodeId, Vec<NodeId>, Vec<NodeId>) {
    let hidden = config.hidden_size;
    let kv_dim = config.kv_dim();
    let ffn = config.intermediate_size;
    let eps = config.rms_norm_eps;
    let theta = config.rope_theta;

    // Single token input
    let token_ids = g.input_u32("token_ids", &[1]);
    // Dynamic position for cache write (u32 scalar)
    let kv_pos = g.input_u32("kv_pos", &[1]);

    let embed_weight = g.parameter("model.embed_tokens.weight", &[config.vocab_size, hidden]);
    let mut x = g.embedding(token_ids, embed_weight);

    let mut k_cache_params = Vec::new();
    let mut v_cache_params = Vec::new();

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

        let q = g.matmul(h, wq); // [1, hidden]
        let k = g.matmul(h, wk); // [1, kv_dim]
        let v = g.matmul(h, wv); // [1, kv_dim]

        // RoPE with dynamic position offset from kv_pos input
        let q = g.rope_dynamic_offset(q, theta, kv_pos, config.head_dim());
        let k = g.rope_dynamic_offset(k, theta, kv_pos, config.head_dim());

        // Pre-allocated KV cache buffers (treated as mutable parameters)
        let k_cache = g.parameter(&format!("kv_cache.layer.{}.k", i), &[max_seq_len, kv_dim]);
        let v_cache = g.parameter(&format!("kv_cache.layer.{}.v", i), &[max_seq_len, kv_dim]);
        k_cache_params.push(k_cache);
        v_cache_params.push(v_cache);

        // Write new K/V into cache at kv_pos
        let _k_updated = g.cache_write(k, k_cache, kv_pos);
        let _v_updated = g.cache_write(v, v_cache, kv_pos);

        // Cached attention: Q attends to full cache
        let attn = g.cached_attention(
            q,
            k_cache,
            v_cache,
            kv_pos,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim(),
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
    let logits = g.matmul(x, lm_head); // [1, vocab]

    (logits, k_cache_params, v_cache_params)
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
    names.push("lm_head.weight".to_string());
    names
}
