//! Whisper encoder model definition for meganeura.
//!
//! Architecture: 2 Conv1d layers → positional embedding → N transformer layers → LayerNorm.
//! Audio encoder only (no decoder).
//!
//! Reference: <https://arxiv.org/abs/2212.04356>
//! Weights: `openai/whisper-tiny` on HuggingFace

use crate::graph::{Graph, NodeId};

pub struct WhisperConfig {
    pub d_model: usize,
    pub n_heads: u32,
    pub n_layers: usize,
    pub ffn_dim: usize,
    pub n_mels: usize,
    pub max_source_positions: usize,
    pub layer_norm_eps: f32,
}

impl WhisperConfig {
    pub fn whisper_tiny() -> Self {
        Self {
            d_model: 384,
            n_heads: 6,
            n_layers: 4,
            ffn_dim: 1536,
            n_mels: 80,
            max_source_positions: 1500,
            layer_norm_eps: 1e-5,
        }
    }

    pub fn head_dim(&self) -> u32 {
        self.d_model as u32 / self.n_heads
    }
}

/// Build the Whisper encoder inference graph.
///
/// Input: `"mel"` with shape `[batch * n_mels * mel_len]` (flat NCHW with H=1).
/// Output: encoder hidden states `[batch * seq_len, d_model]` where
/// `seq_len = mel_len / 2` (due to stride-2 in conv2).
///
/// Weight names follow the HuggingFace convention:
/// `model.encoder.conv1.weight`, `model.encoder.layers.0.self_attn.q_proj.weight`, etc.
pub fn build_encoder(g: &mut Graph, config: &WhisperConfig, batch: u32, mel_len: u32) -> NodeId {
    let d = config.d_model;
    let prefix = "model.encoder";

    // --- Conv stem ---
    // Conv1d is emulated as Conv2d with W=1, temporal axis on H.
    // Input layout: [batch, n_mels, mel_len, 1] (NCHW with W=1).
    // kernel: [out_c, in_c, kernel_h=3, kernel_w=1], padding on H only.
    let mel = g.input("mel", &[(batch * config.n_mels as u32 * mel_len) as usize]);

    // Conv1: (n_mels → d_model, kernel=3, stride=1, padding=1)
    let conv1_w = g.parameter(&format!("{prefix}.conv1.weight"), &[d * config.n_mels * 3]);
    let conv1_b = g.parameter(
        &format!("{prefix}.conv1.fused_bias"),
        &[(batch as usize * d * mel_len as usize)],
    );
    // Conv1d as Conv2d with H=mel_len, W=1. Padding on H only (temporal axis).
    let x = g.conv2d_hw(
        mel,
        conv1_w,
        batch,
        config.n_mels as u32,
        mel_len,
        1,
        d as u32,
        3,
        1,
        1,
        1,
        0,
    );
    let x = g.add(x, conv1_b);
    let x = g.gelu(x);

    // Conv2: (d_model → d_model, kernel=3, stride=2, padding=1 on H only)
    let seq_len = (mel_len + 2 - 3) / 2 + 1;
    let conv2_w = g.parameter(&format!("{prefix}.conv2.weight"), &[d * d * 3]);
    let conv2_b = g.parameter(
        &format!("{prefix}.conv2.fused_bias"),
        &[(batch as usize * d * seq_len as usize)],
    );
    let x = g.conv2d_hw(
        x, conv2_w, batch, d as u32, mel_len, 1, d as u32, 3, 1, 2, 1, 0,
    );
    let x = g.add(x, conv2_b);
    let x = g.gelu(x);

    // The conv output is [batch * d_model * seq_len] in NCHW(flat).
    // We need it as [batch * seq_len, d_model] for the transformer.
    // This is a reshape/transpose — since our IR is flat, we treat the
    // conv output as already shaped for the transformer by transposing
    // the channel and spatial dims. For now, use a parameter-based
    // positional embedding that adds directly.

    // Conv output is [d * seq_len] (1D flat, NCHW with batch=1, W=1).
    // Reshape to [d, seq_len] then transpose to [seq_len, d] for transformer.
    assert_eq!(batch, 1, "Whisper encoder currently supports batch=1 only");
    let x = g.reshape(x, &[d, seq_len as usize]);
    let x = g.transpose(x); // [d, seq_len] → [seq_len, d]

    // Add positional embedding
    let pos_embed = g.parameter(
        &format!("{prefix}.embed_positions.weight"),
        &[seq_len as usize, d],
    );
    let mut x = g.add(x, pos_embed);

    for i in 0..config.n_layers {
        let lname = format!("{prefix}.layers.{i}");
        let ln1_w = g.parameter(&format!("{lname}.self_attn_layer_norm.weight"), &[d]);
        let ln1_b = g.parameter(&format!("{lname}.self_attn_layer_norm.bias"), &[d]);
        let h = g.layer_norm(x, ln1_w, ln1_b, config.layer_norm_eps);

        // Self-attention: Q, K, V projections (k_proj has no bias in Whisper)
        let wq = g.parameter(&format!("{lname}.self_attn.q_proj.weight"), &[d, d]);
        let wk = g.parameter(&format!("{lname}.self_attn.k_proj.weight"), &[d, d]);
        let wv = g.parameter(&format!("{lname}.self_attn.v_proj.weight"), &[d, d]);
        let q_b = g.parameter(&format!("{lname}.self_attn.q_proj.bias"), &[d]);
        let v_b = g.parameter(&format!("{lname}.self_attn.v_proj.bias"), &[d]);

        let q = g.matmul(h, wq);
        let q = g.bias_add(q, q_b);
        let k = g.matmul(h, wk);
        let v = g.matmul(h, wv);
        let v = g.bias_add(v, v_b);

        let attn = g.full_attention(
            q,
            k,
            v,
            config.n_heads,
            config.n_heads, // no GQA in whisper
            config.head_dim(),
        );

        let wo = g.parameter(&format!("{lname}.self_attn.out_proj.weight"), &[d, d]);
        let wo_b = g.parameter(&format!("{lname}.self_attn.out_proj.bias"), &[d]);
        let attn_out = g.matmul(attn, wo);
        let attn_out = g.bias_add(attn_out, wo_b);
        x = g.add(x, attn_out);

        // FFN: LayerNorm → Linear → GELU → Linear → residual
        let ln2_w = g.parameter(&format!("{lname}.final_layer_norm.weight"), &[d]);
        let ln2_b = g.parameter(&format!("{lname}.final_layer_norm.bias"), &[d]);
        let h = g.layer_norm(x, ln2_w, ln2_b, config.layer_norm_eps);

        let ff1_w = g.parameter(&format!("{lname}.fc1.weight"), &[d, config.ffn_dim]);
        let ff1_b = g.parameter(&format!("{lname}.fc1.bias"), &[config.ffn_dim]);
        let h = g.matmul(h, ff1_w);
        let h = g.bias_add(h, ff1_b);
        let h = g.gelu(h);

        let ff2_w = g.parameter(&format!("{lname}.fc2.weight"), &[config.ffn_dim, d]);
        let ff2_b = g.parameter(&format!("{lname}.fc2.bias"), &[d]);
        let h = g.matmul(h, ff2_w);
        let h = g.bias_add(h, ff2_b);
        x = g.add(x, h);
    }

    // Final layer norm
    let final_ln_w = g.parameter(&format!("{prefix}.layer_norm.weight"), &[d]);
    let final_ln_b = g.parameter(&format!("{prefix}.layer_norm.bias"), &[d]);
    g.layer_norm(x, final_ln_w, final_ln_b, config.layer_norm_eps)
}

/// Build a Whisper encoder training graph (forward + MSE loss on encoder output).
///
/// This is a simplified training setup: the encoder output is projected to a
/// fixed dimension and compared against a target via MSE loss. Useful for
/// fine-tuning or benchmarking training throughput.
pub fn build_training_graph(config: &WhisperConfig, batch: u32, mel_len: u32) -> Graph {
    let mut g = Graph::new();
    let encoder_out = build_encoder(&mut g, config, batch, mel_len);
    // Project encoder output to a small dimension, then cross-entropy loss.
    // This exercises the full backward through attention + LayerNorm + GELU.
    let seq_len = (mel_len / 2) as usize;
    let num_classes = 64;
    let proj_w = g.parameter("train_proj.weight", &[config.d_model, num_classes]);
    let logits = g.matmul(encoder_out, proj_w); // [seq, num_classes]
    let labels = g.input("labels", &[seq_len, num_classes]);
    let loss = g.cross_entropy_loss(logits, labels);
    g.set_outputs(vec![loss]);
    g
}

/// Names of parameters that need transposing when loaded from HuggingFace.
///
/// HuggingFace stores Linear weights as `[out, in]`; meganeura expects `[in, out]`.
pub fn transposed_weight_names(config: &WhisperConfig) -> Vec<String> {
    let prefix = "model.encoder";
    let mut names = Vec::new();
    for i in 0..config.n_layers {
        let l = format!("{prefix}.layers.{i}");
        for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            names.push(format!("{l}.self_attn.{proj}.weight"));
        }
        names.push(format!("{l}.fc1.weight"));
        names.push(format!("{l}.fc2.weight"));
    }
    names
}
