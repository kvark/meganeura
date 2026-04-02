#![allow(dead_code, clippy::too_many_arguments)]
/// Load SmolVLM2-500M from HuggingFace and run forward passes on lavapipe.
///
/// Tests the full pipeline: vision encoder → pixel shuffle → connector → text decoder.
/// Uses a reduced patch count (64 instead of 1024) for the vision encoder to keep
/// FullAttention tractable on software Vulkan (lavapipe).
///
/// Usage:
///   cargo run --release --example smolvlm2
#[allow(dead_code, clippy::too_many_arguments)]
use meganeura::{Graph, NodeId, build_inference_session, data::safetensors::SafeTensorsModel};
use std::collections::HashSet;

const REPO_ID: &str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct";

// ---------------------------------------------------------------------------
// Model: SmolVLM2
// ---------------------------------------------------------------------------

struct VisionConfig {
    image_size: usize,
    patch_size: usize,
    hidden_size: usize,
    num_attention_heads: u32,
    num_hidden_layers: usize,
    intermediate_size: usize,
    layer_norm_eps: f32,
}

impl VisionConfig {
    fn num_patches(&self) -> usize {
        let p = self.image_size / self.patch_size;
        p * p
    }

    fn patch_dim(&self) -> usize {
        3 * self.patch_size * self.patch_size
    }

    fn head_dim(&self) -> u32 {
        self.hidden_size as u32 / self.num_attention_heads
    }
}

struct TextConfig {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    intermediate_size: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
}

impl TextConfig {
    fn head_dim(&self) -> u32 {
        self.hidden_size as u32 / self.num_attention_heads
    }

    fn kv_dim(&self) -> usize {
        self.num_key_value_heads as usize * self.head_dim() as usize
    }
}

struct SmolVLM2Config {
    vision: VisionConfig,
    text: TextConfig,
    scale_factor: usize,
}

impl SmolVLM2Config {
    fn smolvlm2_500m() -> Self {
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

    fn num_vision_tokens(&self) -> usize {
        self.vision.num_patches() / (self.scale_factor * self.scale_factor)
    }

    fn connector_input_dim(&self) -> usize {
        self.vision.hidden_size * self.scale_factor * self.scale_factor
    }
}

fn build_vision_encoder(g: &mut Graph, config: &VisionConfig, num_patches: usize) -> NodeId {
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

        // Self-attention
        let wq = g.parameter(
            &format!("{}.self_attn.q_proj.weight", prefix),
            &[hidden, hidden],
        );
        let bq = g.parameter(&format!("{}.self_attn.q_proj.bias", prefix), &[hidden]);
        let wk = g.parameter(
            &format!("{}.self_attn.k_proj.weight", prefix),
            &[hidden, hidden],
        );
        let bk = g.parameter(&format!("{}.self_attn.k_proj.bias", prefix), &[hidden]);
        let wv = g.parameter(
            &format!("{}.self_attn.v_proj.weight", prefix),
            &[hidden, hidden],
        );
        let bv = g.parameter(&format!("{}.self_attn.v_proj.bias", prefix), &[hidden]);

        let q = g.matmul(h, wq);
        let q = g.bias_add(q, bq);
        let k = g.matmul(h, wk);
        let k = g.bias_add(k, bk);
        let v = g.matmul(h, wv);
        let v = g.bias_add(v, bv);

        let attn = g.full_attention(q, k, v, num_heads, num_heads, head_dim);

        // Output projection
        let wo = g.parameter(
            &format!("{}.self_attn.out_proj.weight", prefix),
            &[hidden, hidden],
        );
        let bo = g.parameter(&format!("{}.self_attn.out_proj.bias", prefix), &[hidden]);
        let attn_out = g.matmul(attn, wo);
        let attn_out = g.bias_add(attn_out, bo);

        x = g.add(x, attn_out);

        // Post-attention LayerNorm + MLP
        let ln2_w = g.parameter(&format!("{}.layer_norm2.weight", prefix), &[hidden]);
        let ln2_b = g.parameter(&format!("{}.layer_norm2.bias", prefix), &[hidden]);
        let h = g.layer_norm(x, ln2_w, ln2_b, eps);

        // MLP: fc1 → GELU → fc2
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

        x = g.add(x, mlp);
    }

    // Post-encoder layer norm
    let post_ln_w = g.parameter("model.vision_model.post_layernorm.weight", &[hidden]);
    let post_ln_b = g.parameter("model.vision_model.post_layernorm.bias", &[hidden]);
    g.layer_norm(x, post_ln_w, post_ln_b, eps)
}

fn build_text_decoder(
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
        let up = g.matmul(h, w_up);
        let gate_up = g.swiglu(gate, up);
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

fn transposed_weight_names(config: &SmolVLM2Config) -> Vec<String> {
    let mut names = Vec::new();

    // Vision encoder linear weights
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

// ---------------------------------------------------------------------------
// Example main
// ---------------------------------------------------------------------------

/// Load a parameter, handling transposition and special cases.
fn load_param(
    session: &mut meganeura::Session,
    name: &str,
    model: &SafeTensorsModel,
    transposed_set: &HashSet<&str>,
) {
    if transposed_set.contains(name) {
        let data = model
            .tensor_f32_auto_transposed(name)
            .unwrap_or_else(|e| panic!("{}: {}", name, e));
        session.set_parameter(name, &data);
    } else {
        let data = model
            .tensor_f32_auto(name)
            .unwrap_or_else(|e| panic!("{}: {}", name, e));
        session.set_parameter(name, &data);
    }
}

fn main() {
    env_logger::init();

    let config = SmolVLM2Config::smolvlm2_500m();

    // --- Download model ---
    println!("downloading {} ...", REPO_ID);
    let model = SafeTensorsModel::download(REPO_ID).expect("failed to download model");

    let mut tensor_names: Vec<_> = model.tensor_info().keys().collect();
    tensor_names.sort();
    println!("{} tensors in checkpoint:", tensor_names.len());
    for name in tensor_names.iter().take(5) {
        let info = &model.tensor_info()[*name];
        println!("  {}: {:?} {:?}", name, info.shape, info.dtype);
    }
    println!("  ... and {} more", tensor_names.len() - 5);

    let transposed = transposed_weight_names(&config);
    let transposed_set: HashSet<&str> = transposed.iter().map(|s| s.as_str()).collect();

    // ======== Vision Encoder (reduced patch count for lavapipe) ========
    println!("\n=== Vision Encoder ===");
    // Use 64 patches (8×8 grid) instead of 1024 (32×32) to keep FullAttention
    // tractable on software Vulkan. All per-channel weights are shared; only
    // the position embedding needs slicing.
    let test_num_patches: usize = 64;
    let hidden = config.vision.hidden_size; // 768

    let mut g = Graph::new();
    let vision_out = build_vision_encoder(&mut g, &config.vision, test_num_patches);
    g.set_outputs(vec![vision_out]);

    println!(
        "compiling vision encoder ({} patches, {} hidden)...",
        test_num_patches, hidden
    );
    let mut vision_session = build_inference_session(&g);
    println!(
        "  {} buffers, {} dispatches",
        vision_session.plan().buffers.len(),
        vision_session.plan().dispatches.len()
    );

    // Load vision weights
    println!("loading vision weights...");
    for (name, _) in vision_session.plan().param_buffers.clone() {
        if name == "model.vision_model.embeddings.patch_embedding.weight" {
            // Conv2d [768, 3, 16, 16] in HF → reshape flat [768, 768] (out, in) → transpose
            let raw = model
                .tensor_f32_auto(&name)
                .unwrap_or_else(|e| panic!("{}: {}", name, e));
            let rows = config.vision.hidden_size;
            let cols = config.vision.patch_dim();
            assert_eq!(raw.len(), rows * cols);
            let mut t = vec![0.0f32; rows * cols];
            for r in 0..rows {
                for c in 0..cols {
                    t[c * rows + r] = raw[r * cols + c];
                }
            }
            vision_session.set_parameter(&name, &t);
        } else if name == "model.vision_model.embeddings.position_embedding.weight" {
            // Full checkpoint has [1024, 768]; we only need [test_num_patches, 768]
            let full = model
                .tensor_f32_auto(&name)
                .unwrap_or_else(|e| panic!("{}: {}", name, e));
            let sliced = &full[..test_num_patches * hidden];
            vision_session.set_parameter(&name, sliced);
        } else {
            load_param(&mut vision_session, &name, &model, &transposed_set);
        }
    }

    // Dummy image patches (sinusoidal pattern)
    let patch_dim = config.vision.patch_dim(); // 768
    let patches: Vec<f32> = (0..test_num_patches * patch_dim)
        .map(|i| (i as f32 * 0.001).sin() * 0.1)
        .collect();
    vision_session.set_input("image_patches", &patches);

    println!("running vision encoder ({} patches)...", test_num_patches);
    vision_session.step();
    vision_session.wait();

    let vision_features = vision_session.read_output(test_num_patches * hidden);
    let finite = vision_features.iter().filter(|x| x.is_finite()).count();
    println!(
        "  output: [{}, {}] — {}/{} finite",
        test_num_patches,
        hidden,
        finite,
        vision_features.len()
    );
    assert_eq!(finite, vision_features.len(), "vision output has NaN/Inf!");
    // Drop vision session to reclaim GPU memory
    drop(vision_session);

    // ======== Pixel Shuffle (CPU) ========
    println!("\n=== Pixel Shuffle ===");
    let sf = config.scale_factor; // 4
    let test_grid = 8; // sqrt(test_num_patches)
    let new_grid = test_grid / sf; // 2
    let test_vision_tokens = new_grid * new_grid; // 4
    let connector_input_dim = config.connector_input_dim(); // 768*16 = 12288

    let mut shuffled = vec![0.0f32; test_vision_tokens * connector_input_dim];
    for ny in 0..new_grid {
        for nx in 0..new_grid {
            let dst_idx = ny * new_grid + nx;
            for dy in 0..sf {
                for dx in 0..sf {
                    let src_idx = (ny * sf + dy) * test_grid + (nx * sf + dx);
                    let ch_offset = (dy * sf + dx) * hidden;
                    shuffled[dst_idx * connector_input_dim + ch_offset
                        ..dst_idx * connector_input_dim + ch_offset + hidden]
                        .copy_from_slice(
                            &vision_features[src_idx * hidden..(src_idx + 1) * hidden],
                        );
                }
            }
        }
    }
    println!(
        "  [{}, {}] → [{}, {}]",
        test_num_patches, hidden, test_vision_tokens, connector_input_dim
    );

    // ======== Connector Projection (CPU) ========
    println!("\n=== Connector ===");
    let text_hidden = config.text.hidden_size; // 960

    let connector_weight = model
        .tensor_f32_auto_transposed("model.connector.modality_projection.proj.weight")
        .unwrap_or_else(|e| panic!("connector weight: {}", e));

    // CPU matmul: [test_vision_tokens, 12288] @ [12288, 960] → [test_vision_tokens, 960]
    let mut vision_projected = vec![0.0f32; test_vision_tokens * text_hidden];
    for i in 0..test_vision_tokens {
        for j in 0..text_hidden {
            let mut sum = 0.0f32;
            for k in 0..connector_input_dim {
                sum +=
                    shuffled[i * connector_input_dim + k] * connector_weight[k * text_hidden + j];
            }
            vision_projected[i * text_hidden + j] = sum;
        }
    }
    let finite = vision_projected.iter().filter(|x| x.is_finite()).count();
    println!(
        "  [{}, {}] — {}/{} finite",
        test_vision_tokens,
        text_hidden,
        finite,
        vision_projected.len()
    );

    // ======== Text Decoder ========
    println!("\n=== Text Decoder ===");
    let text_seq_len = 8;
    let total_seq_len = test_vision_tokens + text_seq_len; // 12

    let mut g = Graph::new();
    let combined_input = g.input("combined_embeds", &[total_seq_len, text_hidden]);
    let logits = build_text_decoder(&mut g, &config.text, combined_input, total_seq_len);
    g.set_outputs(vec![logits]);

    println!(
        "compiling text decoder (seq_len={}, vocab={})...",
        total_seq_len, config.text.vocab_size
    );
    let mut text_session = build_inference_session(&g);
    println!(
        "  {} buffers, {} dispatches",
        text_session.plan().buffers.len(),
        text_session.plan().dispatches.len()
    );

    // Load text weights
    println!("loading text weights...");
    for (name, _) in text_session.plan().param_buffers.clone() {
        if name == "lm_head.weight" {
            if model.tensor_info().contains_key("lm_head.weight") {
                load_param(&mut text_session, &name, &model, &transposed_set);
            } else {
                println!("  lm_head tied to embed_tokens");
                let data = model
                    .tensor_f32_auto_transposed("model.text_model.embed_tokens.weight")
                    .expect("embed_tokens for tied lm_head");
                text_session.set_parameter("lm_head.weight", &data);
            }
        } else {
            load_param(&mut text_session, &name, &model, &transposed_set);
        }
    }

    // Combined embeddings: vision_projected || zeros (dummy text)
    let mut combined_embeds = vec![0.0f32; total_seq_len * text_hidden];
    combined_embeds[..test_vision_tokens * text_hidden].copy_from_slice(&vision_projected);

    text_session.set_input("combined_embeds", &combined_embeds);

    println!("running text decoder...");
    text_session.step();
    text_session.wait();

    let vocab = config.text.vocab_size;
    let all_logits = text_session.read_output(total_seq_len * vocab);
    let finite = all_logits.iter().filter(|x| x.is_finite()).count();
    println!(
        "  logits: [{}, {}] — {}/{} finite",
        total_seq_len,
        vocab,
        finite,
        all_logits.len()
    );
    assert_eq!(finite, all_logits.len(), "logits have NaN/Inf!");

    // Top-5 predicted tokens at last position
    let last_logits = &all_logits[(total_seq_len - 1) * vocab..total_seq_len * vocab];
    let mut indexed: Vec<(usize, f32)> = last_logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("\ntop-5 tokens at last position:");
    for (tok, score) in indexed.iter().take(5) {
        println!("  token {} → {:.4}", tok, score);
    }

    println!("\nAll forward passes completed successfully!");
}
