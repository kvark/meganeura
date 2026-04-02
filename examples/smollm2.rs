#![allow(dead_code, clippy::too_many_arguments)]
/// Run SmolLM2-135M inference using meganeura.
///
/// Downloads the model and tokenizer from HuggingFace Hub,
/// builds the computation graph, loads weights, and generates text.
///
/// Usage:
///   cargo run --release --example smollm2 [-- "Your prompt here"]
#[allow(dead_code, clippy::too_many_arguments)]
use meganeura::{Graph, build_inference_session, data::safetensors::SafeTensorsModel};

// ---------------------------------------------------------------------------
// Inlined SmolLM2 model
// ---------------------------------------------------------------------------
mod smollm2 {
    use meganeura::graph::{Graph, NodeId};

    pub struct SmolLM2Config {
        pub vocab_size: usize,
        pub hidden_size: usize,
        pub num_hidden_layers: usize,
        pub num_attention_heads: u32,
        pub num_key_value_heads: u32,
        pub intermediate_size: usize,
        pub rms_norm_eps: f32,
        pub rope_theta: f32,
    }

    impl SmolLM2Config {
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

    pub fn build_graph(g: &mut Graph, config: &SmolLM2Config, seq_len: usize) -> NodeId {
        let hidden = config.hidden_size;
        let kv_dim = config.kv_dim();
        let ffn = config.intermediate_size;
        let eps = config.rms_norm_eps;
        let theta = config.rope_theta;

        let token_ids = g.input_u32("token_ids", &[seq_len]);
        let embed_weight = g.parameter("model.embed_tokens.weight", &[config.vocab_size, hidden]);
        let mut x = g.embedding(token_ids, embed_weight);

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
        g.matmul(x, lm_head)
    }

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
}
use smollm2::SmolLM2Config;

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
    let tokenizer_path = repo
        .get("tokenizer.json")
        .expect("failed to download tokenizer.json");
    let tokenizer =
        tokenizers::Tokenizer::from_file(tokenizer_path).expect("failed to load tokenizer");

    // Encode prompt
    let encoding = tokenizer
        .encode(prompt.as_str(), false)
        .expect("tokenization failed");
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
