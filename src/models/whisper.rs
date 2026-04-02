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
    // Conv1: (n_mels → d_model, kernel=3, stride=1, padding=1)
    // Treated as Conv2d with H=1: input [batch, n_mels, 1, mel_len]
    let mel = g.input("mel", &[(batch * config.n_mels as u32 * mel_len) as usize]);
    let conv1_w = g.parameter(
        &format!("{prefix}.conv1.weight"),
        &[d * config.n_mels * 3], // [d_model, n_mels, 3] flattened
    );
    // Conv bias is stored pre-expanded to [batch * d_model * 1 * mel_len] at load time
    // (NCHW channel bias requires spatial broadcast, like BatchNorm fusion)
    let conv1_out_spatial = mel_len; // stride=1, padding=1, kernel=3 → same length
    let conv1_b = g.parameter(
        &format!("{prefix}.conv1.fused_bias"),
        &[(batch as usize * d * conv1_out_spatial as usize)],
    );
    let x = g.conv2d(
        mel,
        conv1_w,
        batch,
        config.n_mels as u32,
        1,
        mel_len,
        d as u32,
        1,
        3,
        1,
        1,
    );
    let x = g.add(x, conv1_b);
    let x = g.gelu(x);

    // Conv2: (d_model → d_model, kernel=3, stride=2, padding=1)
    let seq_len = (mel_len + 2 - 3) / 2 + 1; // after stride-2 conv with padding=1
    let conv2_w = g.parameter(&format!("{prefix}.conv2.weight"), &[d * d * 3]);
    let conv2_b = g.parameter(
        &format!("{prefix}.conv2.fused_bias"),
        &[(batch as usize * d * seq_len as usize)],
    );
    let x = g.conv2d(
        x, conv2_w, batch, d as u32, 1, mel_len, d as u32, 1, 3, 2, 1,
    );
    let x = g.add(x, conv2_b);
    let x = g.gelu(x);

    // The conv output is [batch * d_model * seq_len] in NCHW(flat).
    // We need it as [batch * seq_len, d_model] for the transformer.
    // This is a reshape/transpose — since our IR is flat, we treat the
    // conv output as already shaped for the transformer by transposing
    // the channel and spatial dims. For now, use a parameter-based
    // positional embedding that adds directly.

    // Conv output is [d_model, seq_len] (NCHW flat with batch=1, H=1).
    // Transpose to [seq_len, d_model] for the transformer layers.
    assert_eq!(batch, 1, "Whisper encoder currently supports batch=1 only");
    let x = g.transpose(x); // [d_model, seq_len] → [seq_len, d_model]

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
