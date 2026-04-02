/// Run Whisper-tiny encoder inference using meganeura.
///
/// Downloads weights from HuggingFace, builds the encoder graph
/// manually, and processes a synthetic mel spectrogram.
///
/// Usage:
///   cargo run --release --example whisper
use meganeura::{
    Graph, build_inference_session,
    data::safetensors::SafeTensorsModel,
    models::whisper::{self, WhisperConfig},
};

const REPO_ID: &str = "openai/whisper-tiny";

fn main() {
    env_logger::init();

    let config = WhisperConfig::whisper_tiny();
    let batch = 1u32;
    let mel_len = 3000u32; // 30 seconds of audio

    // --- Build graph ---
    println!("building Whisper-tiny encoder graph...");
    let mut g = Graph::new();
    let hidden = whisper::build_encoder(&mut g, &config, batch, mel_len);
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
    let transposed = whisper::transposed_weight_names(&config);
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
