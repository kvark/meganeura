/// Load SmolVLM2-500M from HuggingFace and run forward passes on lavapipe.
///
/// Tests the full pipeline: vision encoder → pixel shuffle → connector → text decoder.
/// Uses a reduced patch count (64 instead of 1024) for the vision encoder to keep
/// FullAttention tractable on software Vulkan (lavapipe).
///
/// Usage:
///   cargo run --release --example smolvlm2
use meganeura::{
    Graph, build_inference_session,
    data::safetensors::SafeTensorsModel,
    models::smolvlm2::{self, SmolVLM2Config},
};
use std::collections::HashSet;

const REPO_ID: &str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct";

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

    let transposed = smolvlm2::transposed_weight_names(&config);
    let transposed_set: HashSet<&str> = transposed.iter().map(|s| s.as_str()).collect();

    // ======== Vision Encoder (reduced patch count for lavapipe) ========
    println!("\n=== Vision Encoder ===");
    // Use 64 patches (8×8 grid) instead of 1024 (32×32) to keep FullAttention
    // tractable on software Vulkan. All per-channel weights are shared; only
    // the position embedding needs slicing.
    let test_num_patches: usize = 64;
    let hidden = config.vision.hidden_size; // 768

    let mut g = Graph::new();
    let vision_out = smolvlm2::build_vision_encoder(&mut g, &config.vision, test_num_patches);
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
    let logits = smolvlm2::build_text_decoder(&mut g, &config.text, combined_input, total_seq_len);
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
