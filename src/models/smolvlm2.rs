//! SmolVLM2 model definition for meganeura.
//!
//! Builds the computation graph for SmolVLM2-500M-Video-Instruct inference.
//! Architecture: SigLIP vision encoder + pixel shuffle connector + LLaMA-3 text decoder.
//!
//! The vision encoder processes image patches through a vision transformer,
//! then the connector projects vision features into the text model's embedding space.

use crate::graph::{Graph, NodeId};

/// Vision encoder configuration (SigLIP-like ViT).
pub struct VisionConfig {
    /// Image size in pixels (square).
    pub image_size: usize,
    /// Patch size in pixels (square).
    pub patch_size: usize,
    /// Hidden dimensionality of vision transformer.
    pub hidden_size: usize,
    /// Number of attention heads in vision transformer.
    pub num_attention_heads: u32,
    /// Number of vision transformer layers.
    pub num_hidden_layers: usize,
    /// Vision MLP intermediate size (4x hidden).
    pub intermediate_size: usize,
    /// Layer norm epsilon.
    pub layer_norm_eps: f32,
}

impl VisionConfig {
    /// Number of patches per image (image_size / patch_size)².
    pub fn num_patches(&self) -> usize {
        let p = self.image_size / self.patch_size;
        p * p
    }

    /// Dimension of each flattened patch: channels * patch_size².
    pub fn patch_dim(&self) -> usize {
        3 * self.patch_size * self.patch_size
    }

    pub fn head_dim(&self) -> u32 {
        self.hidden_size as u32 / self.num_attention_heads
    }
}

/// Text model configuration (LLaMA-3 variant).
pub struct TextConfig {
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
    /// Epsilon for RMSNorm.
    pub rms_norm_eps: f32,
    /// Base frequency for RoPE.
    pub rope_theta: f32,
}

impl TextConfig {
    pub fn head_dim(&self) -> u32 {
        self.hidden_size as u32 / self.num_attention_heads
    }

    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads as usize * self.head_dim() as usize
    }
}

/// Full SmolVLM2 configuration.
pub struct SmolVLM2Config {
    pub vision: VisionConfig,
    pub text: TextConfig,
    /// Pixel shuffle scale factor (spatial downsampling before projection).
    pub scale_factor: usize,
}

impl SmolVLM2Config {
    /// SmolVLM2-500M-Video-Instruct configuration.
    pub fn smolvlm2_500m() -> Self {
        Self {
            vision: VisionConfig {
                image_size: 512,
                patch_size: 16,
                hidden_size: 768,
                num_attention_heads: 12,
                num_hidden_layers: 12,
                intermediate_size: 3072,
                layer_norm_eps: 1e-6,
            },
            text: TextConfig {
                vocab_size: 49280,
                hidden_size: 960,
                num_hidden_layers: 32,
                num_attention_heads: 15,
                num_key_value_heads: 5,
                intermediate_size: 2560,
                rms_norm_eps: 1e-5,
                rope_theta: 100000.0,
            },
            scale_factor: 4,
        }
    }

    /// Number of vision tokens after pixel shuffle downsampling.
    pub fn num_vision_tokens(&self) -> usize {
        self.vision.num_patches() / (self.scale_factor * self.scale_factor)
    }

    /// Connector input dimension: vision_hidden * scale_factor².
    pub fn connector_input_dim(&self) -> usize {
        self.vision.hidden_size * self.scale_factor * self.scale_factor
    }
}

/// Build the vision encoder graph.
///
/// Takes pre-processed image patches as input and returns vision features.
///
/// Input: "image_patches" — F32 tensor of shape `[num_patches, patch_dim]`
///   where patch_dim = 3 * patch_size² (patches extracted and flattened on CPU).
///
/// Returns the vision feature tensor node, shape `[num_patches, vision_hidden]`.
pub fn build_vision_encoder(
    g: &mut Graph,
    config: &VisionConfig,
    num_patches: usize,
) -> NodeId {
    let hidden = config.hidden_size;
    let eps = config.layer_norm_eps;
    let num_heads = config.num_attention_heads;
    let head_dim = config.head_dim();

    // Patch embedding: linear projection of flattened patches
    let patches = g.input("image_patches", &[num_patches, config.patch_dim()]);
    let patch_weight = g.parameter(
        "model.vision_model.embeddings.patch_embedding.weight",
        &[config.patch_dim(), hidden],
    );
    let patch_bias = g.parameter(
        "model.vision_model.embeddings.patch_embedding.bias",
        &[hidden],
    );
    let mut x = g.matmul(patches, patch_weight);
    x = g.bias_add(x, patch_bias);

    // Position embedding (learned, added to patch embeddings)
    let pos_embed = g.parameter(
        "model.vision_model.embeddings.position_embedding.weight",
        &[num_patches, hidden],
    );
    x = g.add(x, pos_embed);

    // Vision transformer layers
    for i in 0..config.num_hidden_layers {
        let prefix = format!("model.vision_model.encoder.layers.{}", i);

        // Pre-attention LayerNorm
        let ln1_w = g.parameter(&format!("{}.layer_norm1.weight", prefix), &[hidden]);
        let ln1_b = g.parameter(&format!("{}.layer_norm1.bias", prefix), &[hidden]);
        let h = g.layer_norm(x, ln1_w, ln1_b, eps);

        // Self-attention (non-causal, all heads equal for vision)
        let wq = g.parameter(
            &format!("{}.self_attn.q_proj.weight", prefix),
            &[hidden, hidden],
        );
        let bq = g.parameter(
            &format!("{}.self_attn.q_proj.bias", prefix),
            &[hidden],
        );
        let wk = g.parameter(
            &format!("{}.self_attn.k_proj.weight", prefix),
            &[hidden, hidden],
        );
        let bk = g.parameter(
            &format!("{}.self_attn.k_proj.bias", prefix),
            &[hidden],
        );
        let wv = g.parameter(
            &format!("{}.self_attn.v_proj.weight", prefix),
            &[hidden, hidden],
        );
        let bv = g.parameter(
            &format!("{}.self_attn.v_proj.bias", prefix),
            &[hidden],
        );

        let q = g.matmul(h, wq);
        let q = g.bias_add(q, bq);
        let k = g.matmul(h, wk);
        let k = g.bias_add(k, bk);
        let v = g.matmul(h, wv);
        let v = g.bias_add(v, bv);

        // Full (non-causal) attention — vision patches attend to all other patches
        let attn = g.full_attention(q, k, v, num_heads, num_heads, head_dim);

        // Output projection
        let wo = g.parameter(
            &format!("{}.self_attn.out_proj.weight", prefix),
            &[hidden, hidden],
        );
        let bo = g.parameter(
            &format!("{}.self_attn.out_proj.bias", prefix),
            &[hidden],
        );
        let attn_out = g.matmul(attn, wo);
        let attn_out = g.bias_add(attn_out, bo);

        // Residual connection
        x = g.add(x, attn_out);

        // Post-attention LayerNorm + MLP
        let ln2_w = g.parameter(&format!("{}.layer_norm2.weight", prefix), &[hidden]);
        let ln2_b = g.parameter(&format!("{}.layer_norm2.bias", prefix), &[hidden]);
        let h = g.layer_norm(x, ln2_w, ln2_b, eps);

        // MLP: fc1 → GELU → fc2 (standard, not SwiGLU)
        let w1 = g.parameter(
            &format!("{}.mlp.fc1.weight", prefix),
            &[hidden, config.intermediate_size],
        );
        let b1 = g.parameter(
            &format!("{}.mlp.fc1.bias", prefix),
            &[config.intermediate_size],
        );
        let w2 = g.parameter(
            &format!("{}.mlp.fc2.weight", prefix),
            &[config.intermediate_size, hidden],
        );
        let b2 = g.parameter(&format!("{}.mlp.fc2.bias", prefix), &[hidden]);

        let mlp = g.matmul(h, w1);
        let mlp = g.bias_add(mlp, b1);
        let mlp = g.gelu(mlp);
        let mlp = g.matmul(mlp, w2);
        let mlp = g.bias_add(mlp, b2);

        // Residual connection
        x = g.add(x, mlp);
    }

    // Post-encoder layer norm
    let post_ln_w = g.parameter("model.vision_model.post_layernorm.weight", &[hidden]);
    let post_ln_b = g.parameter("model.vision_model.post_layernorm.bias", &[hidden]);
    g.layer_norm(x, post_ln_w, post_ln_b, eps)
}

/// Build the text decoder graph.
///
/// Input `x` is the combined token + vision embedding sequence.
/// Returns logits of shape `[seq_len, vocab_size]`.
pub fn build_text_decoder(
    g: &mut Graph,
    config: &TextConfig,
    mut x: NodeId,
    _seq_len: usize,
) -> NodeId {
    let hidden = config.hidden_size;
    let kv_dim = config.kv_dim();
    let ffn = config.intermediate_size;
    let eps = config.rms_norm_eps;
    let theta = config.rope_theta;

    for i in 0..config.num_hidden_layers {
        let prefix = format!("model.text_model.layers.{}", i);

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

        let q = g.matmul(h, wq);
        let k = g.matmul(h, wk);
        let v = g.matmul(h, wv);

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

        // Residual
        x = g.add(x, attn_out);

        // Post-attention RMSNorm + SwiGLU FFN
        let ln2_w = g.parameter(
            &format!("{}.post_attention_layernorm.weight", prefix),
            &[hidden],
        );
        let h = g.rms_norm(x, ln2_w, eps);

        let w_gate = g.parameter(&format!("{}.mlp.gate_proj.weight", prefix), &[hidden, ffn]);
        let w_up = g.parameter(&format!("{}.mlp.up_proj.weight", prefix), &[hidden, ffn]);
        let w_down = g.parameter(&format!("{}.mlp.down_proj.weight", prefix), &[ffn, hidden]);

        let gate = g.matmul(h, w_gate);
        let gate = g.silu(gate);
        let up = g.matmul(h, w_up);
        let gate_up = g.mul(gate, up);
        let ffn_out = g.matmul(gate_up, w_down);

        x = g.add(x, ffn_out);
    }

    // Final RMSNorm
    let final_ln_w = g.parameter("model.text_model.norm.weight", &[hidden]);
    x = g.rms_norm(x, final_ln_w, eps);

    // LM head
    let lm_head = g.parameter("lm_head.weight", &[hidden, config.vocab_size]);
    g.matmul(x, lm_head)
}

/// Build the full SmolVLM2 inference graph.
///
/// Inputs:
/// - "token_ids": U32 `[text_seq_len]` — text token IDs
/// - "image_patches": F32 `[num_patches, patch_dim]` — pre-processed image patches
/// - "vision_features_shuffled": F32 `[num_vision_tokens, connector_input_dim]` — pixel-shuffled features
/// - "combined_embeds": F32 `[total_seq_len, text_hidden]` — combined vision + text embeddings
///
/// Returns logits node, shape `[total_seq_len, vocab_size]`.
pub fn build_graph(
    g: &mut Graph,
    config: &SmolVLM2Config,
    text_seq_len: usize,
) -> NodeId {
    let num_patches = config.vision.num_patches();
    let num_vision_tokens = config.num_vision_tokens();
    let total_seq_len = num_vision_tokens + text_seq_len;

    // Vision encoder
    let vision_features = build_vision_encoder(g, &config.vision, num_patches);
    // vision_features: [num_patches, vision_hidden]

    // Pixel shuffle connector: spatially rearrange and project
    // After pixel shuffle with scale_factor=4:
    //   [1024, 768] → [1024/16, 768*16] = [64, 12288]
    // Then linear projection to text hidden:
    //   [64, 12288] → [64, 960]
    let connector_input_dim = config.connector_input_dim();
    let connector_weight = g.parameter(
        "model.connector.modality_projection.proj.weight",
        &[connector_input_dim, config.text.hidden_size],
    );

    // Input: pixel-shuffled vision features [num_vision_tokens, connector_input_dim]
    let shuffled_features = g.input(
        "vision_features_shuffled",
        &[num_vision_tokens, connector_input_dim],
    );
    let vision_projected = g.matmul(shuffled_features, connector_weight);
    // vision_projected: [num_vision_tokens, text_hidden]

    // Text token embeddings
    let token_ids = g.input_u32("token_ids", &[text_seq_len]);
    let embed_weight = g.parameter(
        "model.text_model.embed_tokens.weight",
        &[config.text.vocab_size, config.text.hidden_size],
    );
    let _text_embeds = g.embedding(token_ids, embed_weight);

    // Combined sequence embedding input
    let combined_embeds = g.input(
        "combined_embeds",
        &[total_seq_len, config.text.hidden_size],
    );

    // Text decoder
    let logits = build_text_decoder(g, &config.text, combined_embeds, total_seq_len);

    // Suppress unused warnings
    let _ = vision_features;
    let _ = vision_projected;

    logits
}

/// Get all weight parameter names for SmolVLM2.
pub fn weight_names(config: &SmolVLM2Config) -> Vec<String> {
    let mut names = Vec::new();

    // Vision encoder
    names.push("model.vision_model.embeddings.patch_embedding.weight".into());
    names.push("model.vision_model.embeddings.patch_embedding.bias".into());
    names.push("model.vision_model.embeddings.position_embedding.weight".into());

    for i in 0..config.vision.num_hidden_layers {
        let p = format!("model.vision_model.encoder.layers.{}", i);
        names.push(format!("{}.layer_norm1.weight", p));
        names.push(format!("{}.layer_norm1.bias", p));
        names.push(format!("{}.self_attn.q_proj.weight", p));
        names.push(format!("{}.self_attn.q_proj.bias", p));
        names.push(format!("{}.self_attn.k_proj.weight", p));
        names.push(format!("{}.self_attn.k_proj.bias", p));
        names.push(format!("{}.self_attn.v_proj.weight", p));
        names.push(format!("{}.self_attn.v_proj.bias", p));
        names.push(format!("{}.self_attn.out_proj.weight", p));
        names.push(format!("{}.self_attn.out_proj.bias", p));
        names.push(format!("{}.layer_norm2.weight", p));
        names.push(format!("{}.layer_norm2.bias", p));
        names.push(format!("{}.mlp.fc1.weight", p));
        names.push(format!("{}.mlp.fc1.bias", p));
        names.push(format!("{}.mlp.fc2.weight", p));
        names.push(format!("{}.mlp.fc2.bias", p));
    }

    names.push("model.vision_model.post_layernorm.weight".into());
    names.push("model.vision_model.post_layernorm.bias".into());

    // Connector
    names.push("model.connector.modality_projection.proj.weight".into());

    // Text model
    names.push("model.text_model.embed_tokens.weight".into());

    for i in 0..config.text.num_hidden_layers {
        let p = format!("model.text_model.layers.{}", i);
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

    names.push("model.text_model.norm.weight".into());
    names.push("lm_head.weight".into());

    names
}

/// Names of weight tensors that need transposing (linear layer weights stored as [out, in]).
pub fn transposed_weight_names(config: &SmolVLM2Config) -> Vec<String> {
    let mut names = Vec::new();

    // Vision encoder linear weights
    // Note: patch_embedding weight is Conv2d [768, 3, 16, 16] — needs special reshape, not transpose
    for i in 0..config.vision.num_hidden_layers {
        let p = format!("model.vision_model.encoder.layers.{}", i);
        names.push(format!("{}.self_attn.q_proj.weight", p));
        names.push(format!("{}.self_attn.k_proj.weight", p));
        names.push(format!("{}.self_attn.v_proj.weight", p));
        names.push(format!("{}.self_attn.out_proj.weight", p));
        names.push(format!("{}.mlp.fc1.weight", p));
        names.push(format!("{}.mlp.fc2.weight", p));
    }

    // Connector
    names.push("model.connector.modality_projection.proj.weight".into());

    // Text model linear weights
    for i in 0..config.text.num_hidden_layers {
        let p = format!("model.text_model.layers.{}", i);
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
