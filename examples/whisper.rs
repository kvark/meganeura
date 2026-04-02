#![allow(dead_code, clippy::too_many_arguments)]
/// Run Whisper-tiny encoder inference using meganeura.
///
/// Downloads weights from HuggingFace, builds the encoder graph
/// manually, and processes a synthetic mel spectrogram.
///
/// Usage:
///   cargo run --release --example whisper
#[allow(dead_code, clippy::too_many_arguments)]
use meganeura::{Graph, NodeId, build_inference_session, data::safetensors::SafeTensorsModel};

const REPO_ID: &str = "openai/whisper-tiny";

// ---------------------------------------------------------------------------
// Model: Whisper encoder
// ---------------------------------------------------------------------------

struct WhisperConfig {
    d_model: usize,
    n_heads: u32,
    n_layers: usize,
    ffn_dim: usize,
    n_mels: usize,
    max_source_positions: usize,
    layer_norm_eps: f32,
}

impl WhisperConfig {
    fn whisper_tiny() -> Self {
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

    fn head_dim(&self) -> u32 {
        self.d_model as u32 / self.n_heads
    }
}

fn build_encoder(g: &mut Graph, config: &WhisperConfig, batch: u32, mel_len: u32) -> NodeId {
    let d = config.d_model;
    let prefix = "model.encoder";

    // --- Conv stem ---
    // Conv1: (n_mels → d_model, kernel=3, stride=1, padding=1)
    // Treated as Conv2d with H=1: input [batch, 80, 1, mel_len]
    let mel = g.input("mel", &[(batch * config.n_mels as u32 * mel_len) as usize]);
    let conv1_w = g.parameter(
        &format!("{prefix}.conv1.weight"),
        &[d * config.n_mels * 3], // [d_model, n_mels, 3] flattened
    );
    let conv1_b = g.parameter(&format!("{prefix}.conv1.bias"), &[d]);
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
    let x = g.bias_add(x, conv1_b);
    let x = g.gelu(x);
    // After conv1: [batch, d_model, 1, mel_len] → still mel_len temporal

    // Conv2: (d_model → d_model, kernel=3, stride=2, padding=1)
    let conv2_w = g.parameter(
        &format!("{prefix}.conv2.weight"),
        &[d * d * 3], // [d_model, d_model, 3] flattened
    );
    let conv2_b = g.parameter(&format!("{prefix}.conv2.bias"), &[d]);
    let x = g.conv2d(
        x, conv2_w, batch, d as u32, 1, mel_len, d as u32, 1, 3, 2, 1,
    );
    let x = g.bias_add(x, conv2_b);
    let x = g.gelu(x);
    // After conv2: [batch, d_model, 1, seq_len] where seq_len = mel_len/2

    let seq_len = (mel_len + 2 - 3) / 2 + 1; // after stride-2 conv with padding=1

    // Positional embedding: [max_source_positions, d_model]
    let pos_embed = g.parameter(
        &format!("{prefix}.embed_positions.weight"),
        &[seq_len as usize, d],
    );

    let _ = pos_embed;

    // --- Transformer layers ---
    let mut x = x;

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

fn transposed_weight_names(config: &WhisperConfig) -> Vec<String> {
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

// ---------------------------------------------------------------------------
// Example main
// ---------------------------------------------------------------------------

fn main() {
    env_logger::init();

    let config = WhisperConfig::whisper_tiny();
    let batch = 1u32;
    let mel_len = 3000u32; // 30 seconds of audio

    // --- Build graph ---
    println!("building Whisper-tiny encoder graph...");
    let mut g = Graph::new();
    let hidden = build_encoder(&mut g, &config, batch, mel_len);
    g.set_outputs(vec![hidden]);

    // --- Compile ---
    println!("compiling...");
    let mut session = build_inference_session(&g);
    println!(
        "  {} buffers, {} dispatches",
        session.plan().buffers.len(),
        session.plan().dispatches.len()
    );

    // --- Download and load weights ---
    println!("downloading {} weights...", REPO_ID);
    let model = SafeTensorsModel::download(REPO_ID).expect("download failed");

    println!("model tensors:");
    let mut names: Vec<_> = model.tensor_info().keys().collect();
    names.sort();
    for name in names.iter().take(10) {
        let info = &model.tensor_info()[*name];
        println!("  {}: shape={:?}", name, info.shape);
    }
    if names.len() > 10 {
        println!("  ... and {} more", names.len() - 10);
    }

    // Load weights (transpose linear layers from [out, in] to [in, out])
    println!("loading weights...");
    let transposed = transposed_weight_names(&config);
    let transposed_set: std::collections::HashSet<&str> =
        transposed.iter().map(|s| s.as_str()).collect();

    for (name, _) in session.plan().param_buffers.clone() {
        // Skip positional embeddings if not in model file
        if !model.tensor_info().contains_key(&name) {
            eprintln!("  skip: {name} (not in model file)");
            continue;
        }
        if transposed_set.contains(name.as_str()) {
            let data = model
                .tensor_f32_auto_transposed(&name)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            session.set_parameter(&name, &data);
        } else {
            let data = model
                .tensor_f32_auto(&name)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            session.set_parameter(&name, &data);
        }
    }
    println!("weights loaded.");

    // --- Inference on synthetic mel spectrogram ---
    // White noise mel spectrogram (not real audio, just tests the pipeline)
    let mel: Vec<f32> = (0..config.n_mels * mel_len as usize)
        .map(|i| ((i * 17 + 5) % 1000) as f32 / 1000.0 - 0.5)
        .collect();

    println!("running encoder on synthetic mel spectrogram...");
    session.set_input("mel", &mel);
    session.step();
    session.wait();

    // Read output
    let output_len = session.plan().buffers[session.plan().output_buffers[0].0 as usize] / 4;
    let output = session.read_output(output_len);

    let seq_len = output.len() / config.d_model;
    println!(
        "\nencoder output: [{}, {}] ({} elements)",
        seq_len,
        config.d_model,
        output.len()
    );
    if output.len() >= 5 {
        println!(
            "  first 5 values: [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
            output[0], output[1], output[2], output[3], output[4]
        );
    }
}
