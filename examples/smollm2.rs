/// Run SmolLM2-135M inference using meganeura.
///
/// Downloads the model and tokenizer from HuggingFace Hub,
/// builds the computation graph, loads weights, and generates text.
///
/// Usage:
///   cargo run --release --example smollm2 [-- "Your prompt here"]
use meganeura::{
    Graph, build_inference_session,
    data::safetensors::SafeTensorsModel,
    models::smollm2::{self, SmolLM2Config},
};

const REPO_ID: &str = "HuggingFaceTB/SmolLM2-135M";

fn main() {
    env_logger::init();

    // Set up Perfetto profiling: MEGANEURA_TRACE=path.pftrace
    let trace_path = std::env::var("MEGANEURA_TRACE").ok();
    if trace_path.is_some() {
        meganeura::profiler::init();
    }

    let prompt = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "The meaning of life is".to_string());
    let max_new_tokens = 32;

    let config = SmolLM2Config::smollm2_135m();

    // --- Download model and tokenizer ---
    println!("downloading {} from HuggingFace Hub...", REPO_ID);
    let model = SafeTensorsModel::download(REPO_ID).expect("failed to download model");

    println!("model tensors:");
    let mut names: Vec<_> = model.tensor_info().keys().collect();
    names.sort();
    for name in names.iter().take(5) {
        let info = &model.tensor_info()[*name];
        println!("  {}: shape={:?} dtype={:?}", name, info.shape, info.dtype);
    }
    if names.len() > 5 {
        println!("  ... and {} more", names.len() - 5);
    }

    // Load tokenizer
    println!("loading tokenizer...");
    let api = hf_hub::api::sync::Api::new().expect("failed to create HF API");
    let repo = api.model(REPO_ID.to_string());
    let tokenizer_path = repo.get("tokenizer.json").expect("failed to download tokenizer.json");
    let tokenizer =
        tokenizers::Tokenizer::from_file(tokenizer_path).expect("failed to load tokenizer");

    // Encode prompt
    let encoding = tokenizer.encode(prompt.as_str(), false).expect("tokenization failed");
    let input_ids: Vec<u32> = encoding.get_ids().to_vec();
    let seq_len = input_ids.len() + max_new_tokens;
    println!(
        "prompt: \"{}\" ({} tokens), generating {} more (seq_len={})",
        prompt,
        input_ids.len(),
        max_new_tokens,
        seq_len
    );

    // --- Build graph ---
    println!("building computation graph...");
    let mut g = Graph::new();
    let logits = smollm2::build_graph(&mut g, &config, seq_len);
    g.set_outputs(vec![logits]);

    // --- Compile ---
    println!("compiling inference session...");
    let mut session = build_inference_session(&g);
    println!(
        "session ready: {} buffers, {} dispatches",
        session.plan().buffers.len(),
        session.plan().dispatches.len()
    );

    // --- Load weights ---
    // HuggingFace Linear layers store weights as (out, in) but meganeura
    // matmul expects (in, out), so certain tensors need transposing.
    println!("loading weights...");
    let transposed = smollm2::transposed_weight_names(&config);
    let transposed_set: std::collections::HashSet<&str> =
        transposed.iter().map(|s| s.as_str()).collect();

    for (name, _) in session.plan().param_buffers.clone() {
        if name == "lm_head.weight" {
            // The language-model head is often weight-tied to the token
            // embedding table. If a dedicated lm_head tensor exists in the
            // file we load it; otherwise we reuse embed_tokens transposed.
            if model.tensor_info().contains_key("lm_head.weight") {
                let data = if transposed_set.contains(name.as_str()) {
                    model.tensor_f32_auto_transposed(&name)
                } else {
                    model.tensor_f32_auto(&name)
                };
                session.set_parameter(&name, &data.unwrap_or_else(|e| panic!("{}: {}", name, e)));
            } else {
                // Tied weights: use embed_tokens transposed
                println!("  lm_head tied to embed_tokens, transposing...");
                let data = model
                    .tensor_f32_auto_transposed("model.embed_tokens.weight")
                    .expect("failed to load embed_tokens for lm_head");
                session.set_parameter("lm_head.weight", &data);
            }
        } else if transposed_set.contains(name.as_str()) {
            let data = model
                .tensor_f32_auto_transposed(&name)
                .unwrap_or_else(|e| panic!("{}: {}", name, e));
            session.set_parameter(&name, &data);
        } else {
            let data = model
                .tensor_f32_auto(&name)
                .unwrap_or_else(|e| panic!("{}: {}", name, e));
            session.set_parameter(&name, &data);
        }
    }
    println!("weights loaded.");

    // --- Generate tokens (greedy, autoregressive) ---
    // Each step feeds the full sequence (prompt + generated so far,
    // zero-padded to seq_len) through the model, then picks the
    // highest-probability next token from the logits at the current
    // position.
    println!("generating...\n");

    let mut tokens = vec![0u32; seq_len];
    tokens[..input_ids.len()].copy_from_slice(&input_ids);

    let mut generated = input_ids.clone();

    for step in 0..max_new_tokens {
        session.set_input_u32("token_ids", &tokens);
        session.step();
        session.wait();

        // The model outputs logits shaped [seq_len, vocab_size].
        // We only care about the position just before the next token
        // we want to predict (causal: position N-1 predicts token N).
        let vocab = config.vocab_size;
        let all_logits = session.read_output(seq_len * vocab);

        let cur_pos = input_ids.len() + step;
        let pos_logits = &all_logits[(cur_pos - 1) * vocab..cur_pos * vocab];

        // Argmax
        let next_token = pos_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0 as u32;

        tokens[cur_pos] = next_token;
        generated.push(next_token);

        // Decode and print incrementally
        let decoded = tokenizer
            .decode(&[next_token], false)
            .unwrap_or_else(|_| "?".to_string());
        print!("{}", decoded);
    }
    println!();

    // Print full output
    let full_text = tokenizer
        .decode(&generated, true)
        .unwrap_or_else(|_| "decode error".to_string());
    println!("\n--- Full output ---\n{}", full_text);

    // Save Perfetto trace when profiling.
    if let Some(ref trace_file) = trace_path {
        let path = std::path::Path::new(trace_file);
        meganeura::profiler::save(path).expect("failed to save profile");
        println!("profile saved to {}", path.display());
    }
}
