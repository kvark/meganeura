//! SmolVLA model definition for meganeura.
//!
//! SmolVLA is a Vision-Language-Action model for robotics that combines
//! SmolVLM2 (vision-language backbone) with an action expert decoder.
//!
//! Architecture:
//! - SmolVLM2 backbone processes images + language into hidden states
//! - Action expert: 16 transformer layers alternating self-attention and cross-attention
//!   - Even layers: self-attention over action tokens
//!   - Odd layers: cross-attention from action tokens to VLM hidden states
//! - Flow matching action decoder with timestep conditioning
//!
//! Reference: <https://huggingface.co/lerobot/smolvla_base>

use crate::graph::{Graph, NodeId};
use crate::models::smolvlm2::{Config as VlmConfig, TextConfig, VisionConfig};

/// Action expert configuration.
pub struct ExpertConfig {
    /// Expert hidden size (= text_hidden * expert_width_multiplier).
    pub hidden_size: usize,
    /// Number of expert transformer layers.
    pub num_layers: usize,
    /// Number of query heads (same as text model).
    pub num_attention_heads: u32,
    /// Number of key/value heads (same as text model).
    pub num_key_value_heads: u32,
    /// Head dimension (same as text model).
    pub head_dim: u32,
    /// FFN intermediate size.
    pub intermediate_size: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// Number of self-attention layers between cross-attention layers.
    pub self_attn_every_n_layers: usize,
}

impl ExpertConfig {
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads as usize * self.head_dim as usize
    }
}

/// Full SmolVLA configuration.
pub struct Config {
    /// VLM backbone (SmolVLM2).
    pub vlm: VlmConfig,
    /// Action expert decoder.
    pub expert: ExpertConfig,
    /// Maximum action dimension.
    pub max_action_dim: usize,
    /// Maximum state dimension.
    pub max_state_dim: usize,
    /// Number of action steps per chunk.
    pub chunk_size: usize,
    /// Number of flow matching denoising steps.
    pub num_steps: usize,
    /// Number of VLM layers to use (may be fewer than total text layers).
    pub num_vlm_layers: usize,
}

impl Config {
    /// Small configuration for unit/smoke tests (no download needed).
    pub fn small_test() -> Self {
        Self {
            vlm: VlmConfig {
                vision: VisionConfig {
                    image_size: 32,
                    patch_size: 16,
                    hidden_size: 64,
                    num_attention_heads: 2,
                    num_hidden_layers: 2,
                    intermediate_size: 128,
                    layer_norm_eps: 1e-6,
                },
                text: TextConfig {
                    vocab_size: 256,
                    hidden_size: 64,
                    num_hidden_layers: 2,
                    num_attention_heads: 2,
                    num_key_value_heads: 2,
                    intermediate_size: 128,
                    rms_norm_eps: 1e-5,
                    rope_theta: 10000.0,
                },
                scale_factor: 1,
            },
            expert: ExpertConfig {
                hidden_size: 64,
                num_layers: 2,
                num_attention_heads: 2,
                num_key_value_heads: 2,
                head_dim: 32,
                intermediate_size: 128,
                rms_norm_eps: 1e-5,
                self_attn_every_n_layers: 2,
            },
            max_action_dim: 8,
            max_state_dim: 8,
            chunk_size: 4,
            num_steps: 2,
            num_vlm_layers: 2,
        }
    }

    /// Default SmolVLA base configuration (lerobot/smolvla_base).
    pub fn smolvla_base() -> Self {
        Self {
            vlm: VlmConfig {
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
            },
            expert: ExpertConfig {
                hidden_size: 720, // 960 * 0.75
                num_layers: 16,
                num_attention_heads: 15,
                num_key_value_heads: 5,
                head_dim: 64,
                intermediate_size: 2048,
                rms_norm_eps: 1e-5,
                self_attn_every_n_layers: 2,
            },
            max_action_dim: 32,
            max_state_dim: 32,
            chunk_size: 50,
            num_steps: 10,
            num_vlm_layers: 16,
        }
    }
}

/// Build the action expert graph.
///
/// The action expert takes noisy action tokens and VLM hidden states,
/// and predicts denoised action velocities via flow matching.
///
/// Inputs:
/// - `noisy_actions`: `[action_seq_len, max_action_dim]`
/// - `timestep`: sinusoidal embedding `[1, expert_hidden * 2]`
/// - `vlm_kv_layer_i` for each cross-attention layer: `[vlm_seq_len, kv_dim]`
///
/// Returns: denoised action prediction `[chunk_size, action_dim]`
pub fn build_action_expert(
    g: &mut Graph,
    config: &Config,
    action_seq_len: usize,
    vlm_seq_len: usize,
) -> NodeId {
    action_expert(g, config, action_seq_len, vlm_seq_len, false)
}

fn action_expert(
    g: &mut Graph,
    config: &Config,
    action_seq_len: usize,
    vlm_seq_len: usize,
    training: bool,
) -> NodeId {
    let expert = &config.expert;
    let expert_hidden = expert.hidden_size;
    let kv_dim = expert.kv_dim();
    let attn_dim = expert.num_attention_heads as usize * expert.head_dim as usize;
    let eps = expert.rms_norm_eps;

    // Inputs
    let noisy_actions = g.input("noisy_actions", &[action_seq_len, config.max_action_dim]);
    let timestep = g.input("timestep", &[1, expert_hidden * 2]);

    // Action input projection: [chunk, action_dim] → [chunk, expert_hidden]
    let action_in_w = g.parameter(
        "model.action_in_proj.weight",
        &[config.max_action_dim, expert_hidden],
    );
    let action_in_b = g.parameter("model.action_in_proj.bias", &[expert_hidden]);
    let mut x = g.matmul(noisy_actions, action_in_w);
    x = g.bias_add(x, action_in_b);

    // Timestep conditioning via MLP: t → [scale, shift]
    // Input timestep is already encoded as sinusoidal embedding
    let time_in_w = g.parameter(
        "model.action_time_mlp_in.weight",
        &[expert_hidden * 2, expert_hidden],
    );
    let time_in_b = g.parameter("model.action_time_mlp_in.bias", &[expert_hidden]);
    let time_out_w = g.parameter(
        "model.action_time_mlp_out.weight",
        &[expert_hidden, expert_hidden],
    );
    let time_out_b = g.parameter("model.action_time_mlp_out.bias", &[expert_hidden]);

    let time_h = g.matmul(timestep, time_in_w);
    let time_h = g.bias_add(time_h, time_in_b);
    let time_h = g.silu(time_h);
    let time_h = g.matmul(time_h, time_out_w);
    let time_embed = g.bias_add(time_h, time_out_b);
    // time_embed: [1, expert_hidden] — broadcast-added to action tokens
    x = g.broadcast_add(x, time_embed);

    // Expert transformer layers
    for i in 0..expert.num_layers {
        let prefix = format!("model.vlm_with_expert.lm_expert.layers.{}", i);
        let is_cross_attn = i % expert.self_attn_every_n_layers != 0;

        // Pre-attention RMSNorm
        let ln1_w = g.parameter(
            &format!("{}.input_layernorm.weight", prefix),
            &[expert_hidden],
        );
        let h = g.rms_norm(x, ln1_w, eps);

        let (kv_input, kv_input_dim) = if is_cross_attn {
            (
                g.input(&format!("vlm_kv_layer_{}", i), &[vlm_seq_len, kv_dim]),
                kv_dim,
            )
        } else {
            (h, expert_hidden)
        };
        let wq = g.parameter(
            &format!("{}.self_attn.q_proj.weight", prefix),
            &[expert_hidden, attn_dim],
        );
        let wk = g.parameter(
            &format!("{}.self_attn.k_proj.weight", prefix),
            &[kv_input_dim, kv_dim],
        );
        let wv = g.parameter(
            &format!("{}.self_attn.v_proj.weight", prefix),
            &[kv_input_dim, kv_dim],
        );
        let q = g.matmul(h, wq);
        let k = g.matmul(kv_input, wk);
        let v = g.matmul(kv_input, wv);
        let attn = if training {
            g.multi_head_attn(
                q,
                k,
                v,
                expert.num_attention_heads,
                expert.num_key_value_heads,
                expert.head_dim,
                is_cross_attn,
            )
        } else if is_cross_attn {
            g.cross_attention(
                q,
                k,
                v,
                expert.num_attention_heads,
                expert.num_key_value_heads,
                expert.head_dim,
            )
        } else {
            g.causal_attention(
                q,
                k,
                v,
                expert.num_attention_heads,
                expert.num_key_value_heads,
                expert.head_dim,
            )
        };
        let wo = g.parameter(
            &format!("{}.self_attn.o_proj.weight", prefix),
            &[attn_dim, expert_hidden],
        );
        let attn_out = g.matmul(attn, wo);
        x = g.add(x, attn_out);

        // Post-attention RMSNorm + SwiGLU FFN
        let ln2_w = g.parameter(
            &format!("{}.post_attention_layernorm.weight", prefix),
            &[expert_hidden],
        );
        let h = g.rms_norm(x, ln2_w, eps);

        let w_gate = g.parameter(
            &format!("{}.mlp.gate_proj.weight", prefix),
            &[expert_hidden, expert.intermediate_size],
        );
        let w_up = g.parameter(
            &format!("{}.mlp.up_proj.weight", prefix),
            &[expert_hidden, expert.intermediate_size],
        );
        let w_down = g.parameter(
            &format!("{}.mlp.down_proj.weight", prefix),
            &[expert.intermediate_size, expert_hidden],
        );

        let gate = g.matmul(h, w_gate);
        let up = g.matmul(h, w_up);
        let gate_up = g.swiglu(gate, up);
        let ffn_out = g.matmul(gate_up, w_down);

        x = g.add(x, ffn_out);
    }

    // Action output projection: [chunk, expert_hidden] → [chunk, action_dim]
    let action_out_w = g.parameter(
        "model.action_out_proj.weight",
        &[expert_hidden, config.max_action_dim],
    );
    let action_out_b = g.parameter("model.action_out_proj.bias", &[config.max_action_dim]);
    let out = g.matmul(x, action_out_w);
    g.bias_add(out, action_out_b)
}

/// Build the state projection.
///
/// Projects robot state observations into the VLM embedding space.
/// Input: "observation_state" F32 `[1, state_dim]`
/// Returns: projected state token `[1, text_hidden]`
pub fn build_state_projection(g: &mut Graph, config: &Config) -> NodeId {
    let state_input = g.input("observation_state", &[1, config.max_state_dim]);
    let w = g.parameter(
        "model.state_proj.weight",
        &[config.max_state_dim, config.vlm.text.hidden_size],
    );
    let b = g.parameter("model.state_proj.bias", &[config.vlm.text.hidden_size]);
    let proj = g.matmul(state_input, w);
    g.bias_add(proj, b)
}

/// Get all weight parameter names for the SmolVLA action expert.
pub fn expert_weight_names(config: &Config) -> Vec<String> {
    let expert = &config.expert;
    let mut names = vec![
        // State projection
        "model.state_proj.weight".into(),
        "model.state_proj.bias".into(),
        // Action projections
        "model.action_in_proj.weight".into(),
        "model.action_in_proj.bias".into(),
        "model.action_out_proj.weight".into(),
        "model.action_out_proj.bias".into(),
        // Time embedding MLP
        "model.action_time_mlp_in.weight".into(),
        "model.action_time_mlp_in.bias".into(),
        "model.action_time_mlp_out.weight".into(),
        "model.action_time_mlp_out.bias".into(),
    ];

    // Expert layers
    for i in 0..expert.num_layers {
        let p = format!("model.vlm_with_expert.lm_expert.layers.{}", i);
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

    names
}

/// Build a training graph for the action expert with MSE loss.
///
/// Uses full GQA-capable `MultiHeadAttn` op (differentiable, with LSE saved
/// for backward) for both self-attention and cross-attention layers.
///
/// Inputs:
/// - `noisy_actions`: `[action_seq_len, max_action_dim]`
/// - `timestep`:      `[1, expert_hidden * 2]`
/// - `vlm_kv_layer_i` for each cross-attn layer: `[vlm_seq_len, kv_dim]`
/// - `target_actions`: `[action_seq_len, max_action_dim]` (training targets)
///
/// Output: scalar MSE loss.
pub fn build_action_expert_training(
    config: &Config,
    action_seq_len: usize,
    vlm_seq_len: usize,
) -> Graph {
    let mut g = Graph::new();
    let out = action_expert(&mut g, config, action_seq_len, vlm_seq_len, true);
    let target = g.input("target_actions", &[action_seq_len, config.max_action_dim]);
    let loss = g.mse_loss(out, target);
    g.set_outputs(vec![loss]);
    g
}

/// Names of expert weight tensors that need transposing.
pub fn expert_transposed_weight_names(config: &Config) -> Vec<String> {
    let expert = &config.expert;
    let mut names = vec![
        "model.state_proj.weight".into(),
        "model.action_in_proj.weight".into(),
        "model.action_out_proj.weight".into(),
        "model.action_time_mlp_in.weight".into(),
        "model.action_time_mlp_out.weight".into(),
    ];

    for i in 0..expert.num_layers {
        let p = format!("model.vlm_with_expert.lm_expert.layers.{}", i);
        names.push(format!("{}.self_attn.q_proj.weight", p));
        names.push(format!("{}.self_attn.k_proj.weight", p));
        names.push(format!("{}.self_attn.v_proj.weight", p));
        names.push(format!("{}.self_attn.o_proj.weight", p));
        names.push(format!("{}.mlp.gate_up_proj.weight", p));
        names.push(format!("{}.mlp.down_proj.weight", p));
    }

    names
}
