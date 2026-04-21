/// Run Qwen3 inference using meganeura.
///
/// Downloads the model and tokenizer from HuggingFace Hub,
/// builds the computation graph inline (no model module), loads weights,
/// and generates text.
///
/// Architecture: decoder-only transformer with GQA, RoPE, RMSNorm, SwiGLU.
/// All Qwen3 dense variants share this architecture with different dimensions.
///
/// Usage:
///   cargo run --release --example qwen3                           # default: 0.6B
///   cargo run --release --example qwen3 -- --model 1.7B           # Qwen3-1.7B
///   cargo run --release --example qwen3 -- --model 4B             # Qwen3-4B
///   cargo run --release --example qwen3 -- --prompt "Hello world" # custom prompt
///   cargo run --release --example qwen3 -- --tokens 128           # generate 128 tokens
///   cargo run --release --example qwen3 -- --f16                  # half-precision weights
///   cargo run --release --example qwen3 -- --q4                   # 4-bit quantized weights
use meganeura::{Graph, build_inference_session, data::safetensors::SafeTensorsModel};

struct Qwen3Config {
    repo_id: &'static str,
    vocab_size: usize,
    hidden_size: usize,
    num_layers: usize,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    intermediate_size: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    tie_word_embeddings: bool,
}

impl Qwen3Config {
    fn q_dim(&self) -> usize {
        self.num_heads as usize * self.head_dim as usize
    }
    fn kv_dim(&self) -> usize {
        self.num_kv_heads as usize * self.head_dim as usize
    }
}

fn qwen3_0_6b() -> Qwen3Config {
    Qwen3Config {
        repo_id: "Qwen/Qwen3-0.6B",
        vocab_size: 151936,
        hidden_size: 1024,
        num_layers: 28,
        num_heads: 16,
        num_kv_heads: 8,
        head_dim: 128,
        intermediate_size: 3072,
        rms_norm_eps: 1e-6,
        rope_theta: 1_000_000.0,
        tie_word_embeddings: true,
    }
}

fn qwen3_1_7b() -> Qwen3Config {
    Qwen3Config {
        repo_id: "Qwen/Qwen3-1.7B",
        vocab_size: 151936,
        hidden_size: 2048,
        num_layers: 28,
        num_heads: 16,
        num_kv_heads: 8,
        head_dim: 128,
        intermediate_size: 8960,
        rms_norm_eps: 1e-6,
        rope_theta: 1_000_000.0,
        tie_word_embeddings: true,
    }
}

fn qwen3_4b() -> Qwen3Config {
    Qwen3Config {
        repo_id: "Qwen/Qwen3-4B",
        vocab_size: 151936,
        hidden_size: 2560,
        num_layers: 36,
        num_heads: 32,
        num_kv_heads: 8,
        head_dim: 128,
        intermediate_size: 9728,
        rms_norm_eps: 1e-6,
        rope_theta: 1_000_000.0,
        tie_word_embeddings: true,
    }
}

fn qwen3_8b() -> Qwen3Config {
    Qwen3Config {
        repo_id: "Qwen/Qwen3-8B",
        vocab_size: 151936,
        hidden_size: 4096,
        num_layers: 36,
        num_heads: 32,
        num_kv_heads: 8,
        head_dim: 128,
        intermediate_size: 12288,
        rms_norm_eps: 1e-6,
        rope_theta: 1_000_000.0,
        tie_word_embeddings: false,
    }
}

fn main() {
    env_logger::init();

    let trace_path = std::env::var("MEGANEURA_TRACE").ok();
    if trace_path.is_some() {
        meganeura::profiler::init();
    }

    let args: Vec<String> = std::env::args().collect();

    // Parse --model flag
    let model_size = args
        .iter()
        .position(|a| a == "--model")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("0.6B");

    let config = match model_size {
        "0.6B" | "0.6b" => qwen3_0_6b(),
        "1.7B" | "1.7b" => qwen3_1_7b(),
        "4B" | "4b" => qwen3_4b(),
        "8B" | "8b" => qwen3_8b(),
        other => {
            eprintln!(
                "Unknown model size: {}. Available: 0.6B, 1.7B, 4B, 8B",
                other
            );
            std::process::exit(1);
        }
    };

    let prompt = args
        .iter()
        .position(|a| a == "--prompt")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| "The meaning of life is".to_string());

    let max_new_tokens: usize = args
        .iter()
        .position(|a| a == "--tokens")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);

    let use_q4 = args.iter().any(|a| a == "--q4");
    let use_f16 = args.iter().any(|a| a == "--f16") && !use_q4;
    let max_layers: usize = args
        .iter()
        .position(|a| a == "--layers")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(config.num_layers);
    // NOTE: --q4 compiles and runs but full-model Q4 produces NaN.
    // The Q4 dequant shader is correct (verified by GPU test at model scale),
    // but something in the multi-layer chain causes NaN propagation.
    // The isolated matmul test (q4_matmul_correctness) passes with zero error.
    if use_q4 {
        println!("Q4_0 weight storage enabled (~8x VRAM reduction for weights)");
    } else if use_f16 {
        println!("f16 weight storage enabled (halving VRAM for weights)");
    }

    // --- Download model and tokenizer ---
    println!("downloading {} from HuggingFace Hub...", config.repo_id);
    let model = SafeTensorsModel::download(config.repo_id).expect("failed to download model");

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

    println!("loading tokenizer...");
    let api = hf_hub::api::sync::Api::new().expect("failed to create HF API");
    let repo = api.model(config.repo_id.to_string());
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

    // --- Build computation graph ---
    println!("building computation graph...");
    let mut g = Graph::new();

    let hidden = config.hidden_size;
    let qd = config.q_dim();
    let kv = config.kv_dim();
    let ffn = config.intermediate_size;
    let eps = config.rms_norm_eps;
    let theta = config.rope_theta;
    let head_dim = config.head_dim;

    // Helper: matmul weight parameters use f16 storage when --f16 is set
    let weight = |g: &mut Graph, name: &str, shape: &[usize]| -> meganeura::graph::NodeId {
        if use_q4 {
            g.parameter_q4(name, shape)
        } else if use_f16 {
            g.parameter_f16(name, shape)
        } else {
            g.parameter(name, shape)
        }
    };

    // Token embedding (stays f32 — indexed by token ID, not matmul'd)
    let token_ids = g.input_u32("token_ids", &[seq_len]);
    let embed_weight = g.parameter("model.embed_tokens.weight", &[config.vocab_size, hidden]);
    let mut x = g.embedding(token_ids, embed_weight);

    // Transformer layers
    let num_layers = max_layers.min(config.num_layers);
    if num_layers < config.num_layers {
        println!("using {} of {} layers", num_layers, config.num_layers);
    }
    for i in 0..num_layers {
        let p = format!("model.layers.{}", i);

        // Pre-attention RMSNorm (tiny weights, keep f32)
        let ln1_w = g.parameter(&format!("{}.input_layernorm.weight", p), &[hidden]);
        let h = g.rms_norm(x, ln1_w, eps);

        // QKV projections (q_dim = num_heads * head_dim, kv_dim = num_kv_heads * head_dim)
        let wq = weight(
            &mut g,
            &format!("{}.self_attn.q_proj.weight", p),
            &[hidden, qd],
        );
        let wk = weight(
            &mut g,
            &format!("{}.self_attn.k_proj.weight", p),
            &[hidden, kv],
        );
        let wv = weight(
            &mut g,
            &format!("{}.self_attn.v_proj.weight", p),
            &[hidden, kv],
        );

        let q = g.matmul(h, wq);
        let k = g.matmul(h, wk);
        let v = g.matmul(h, wv);

        // RoPE
        let q = g.rope(q, theta, head_dim);
        let k = g.rope(k, theta, head_dim);

        // Causal attention with GQA
        let attn = g.causal_attention(q, k, v, config.num_heads, config.num_kv_heads, head_dim);

        // Output projection + residual
        let wo = weight(
            &mut g,
            &format!("{}.self_attn.o_proj.weight", p),
            &[qd, hidden],
        );
        let attn_out = g.matmul(attn, wo);
        x = g.add(x, attn_out);

        // Post-attention RMSNorm (tiny weights, keep f32)
        let ln2_w = g.parameter(&format!("{}.post_attention_layernorm.weight", p), &[hidden]);
        let h = g.rms_norm(x, ln2_w, eps);

        // SwiGLU FFN
        let w_gate = weight(
            &mut g,
            &format!("{}.mlp.gate_proj.weight", p),
            &[hidden, ffn],
        );
        let w_up = weight(&mut g, &format!("{}.mlp.up_proj.weight", p), &[hidden, ffn]);
        let w_down = weight(
            &mut g,
            &format!("{}.mlp.down_proj.weight", p),
            &[ffn, hidden],
        );

        let gate = g.matmul(h, w_gate);
        let up = g.matmul(h, w_up);
        let ffn_out = g.swiglu(gate, up);
        let ffn_out = g.matmul(ffn_out, w_down);

        x = g.add(x, ffn_out);
    }

    // Final RMSNorm
    let final_ln_w = g.parameter("model.norm.weight", &[hidden]);
    x = g.rms_norm(x, final_ln_w, eps);

    // LM head
    let logits = if config.tie_word_embeddings {
        g.matmul_bt(x, embed_weight)
    } else {
        let lm_head = weight(&mut g, "lm_head.weight", &[hidden, config.vocab_size]);
        g.matmul(x, lm_head)
    };

    g.set_outputs(vec![logits]);

    // --- Compile ---
    println!("compiling inference session...");
    let mut session = build_inference_session(&g);
    let total_vram: usize = session.plan().buffers.iter().sum();
    println!(
        "session ready: {} buffers, {} dispatches, {:.1} MB VRAM",
        session.plan().buffers.len(),
        session.plan().dispatches.len(),
        total_vram as f64 / (1024.0 * 1024.0),
    );

    // --- Load weights ---
    println!("loading weights...");

    // Linear layer weights are stored as (out, in) in HuggingFace but
    // meganeura matmul expects (in, out), so all projection weights
    // need transposing.
    let mut transposed_names = Vec::new();
    for i in 0..num_layers {
        let p = format!("model.layers.{}", i);
        transposed_names.push(format!("{}.self_attn.q_proj.weight", p));
        transposed_names.push(format!("{}.self_attn.k_proj.weight", p));
        transposed_names.push(format!("{}.self_attn.v_proj.weight", p));
        transposed_names.push(format!("{}.self_attn.o_proj.weight", p));
        transposed_names.push(format!("{}.mlp.gate_proj.weight", p));
        transposed_names.push(format!("{}.mlp.up_proj.weight", p));
        transposed_names.push(format!("{}.mlp.down_proj.weight", p));
    }
    if !config.tie_word_embeddings {
        transposed_names.push("lm_head.weight".to_string());
    }
    let transposed_set: std::collections::HashSet<&str> =
        transposed_names.iter().map(|s| s.as_str()).collect();

    for (name, _) in session.plan().param_buffers.clone() {
        if name == "lm_head.weight" {
            if model.tensor_info().contains_key("lm_head.weight") {
                let data = if transposed_set.contains(name.as_str()) {
                    model.tensor_f32_auto_transposed(&name)
                } else {
                    model.tensor_f32_auto(&name)
                };
                session.set_parameter(&name, &data.unwrap_or_else(|e| panic!("{}: {}", name, e)));
            } else {
                println!("  lm_head tied to embed_tokens, transposing...");
                let data = model
                    .tensor_f32_auto_transposed("model.embed_tokens.weight")
                    .expect("failed to load embed_tokens for lm_head");
                session.set_parameter("lm_head.weight", &data);
            }
        } else if name.contains('+') {
            // Concatenated weights from SwiGLU fusion
            let parts: Vec<&str> = name.split('+').collect();
            let needs_transpose = parts.iter().any(|p| transposed_set.contains(*p));
            let mut combined = Vec::new();
            for part in &parts {
                let data = model
                    .tensor_f32_auto(part)
                    .unwrap_or_else(|e| panic!("{}: {}", part, e));
                combined.extend_from_slice(&data);
            }
            if needs_transpose {
                let info0 = &model.tensor_info()[parts[0]];
                let in_dim = info0.shape.last().copied().unwrap_or(1);
                let out_total = combined.len() / in_dim;
                let mut transposed = vec![0.0f32; combined.len()];
                for r in 0..out_total {
                    for c in 0..in_dim {
                        transposed[c * out_total + r] = combined[r * in_dim + c];
                    }
                }
                session.set_parameter(&name, &transposed);
            } else {
                session.set_parameter(&name, &combined);
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
    println!("generating...\n");

    let mut tokens = vec![0u32; seq_len];
    tokens[..input_ids.len()].copy_from_slice(&input_ids);

    let mut generated = input_ids.clone();

    let gen_start = std::time::Instant::now();

    for step in 0..max_new_tokens {
        session.set_input_u32("token_ids", &tokens);
        session.step();
        session.wait();

        let all_logits = session.read_output(seq_len * config.vocab_size);

        let cur_pos = input_ids.len() + step;
        let pos_logits =
            &all_logits[(cur_pos - 1) * config.vocab_size..cur_pos * config.vocab_size];

        // Argmax
        let next_token = pos_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap()
            .0 as u32;

        tokens[cur_pos] = next_token;
        generated.push(next_token);

        let decoded = tokenizer
            .decode(&[next_token], false)
            .unwrap_or_else(|_| "?".to_string());
        print!("{}", decoded);
    }
    println!();

    let gen_elapsed = gen_start.elapsed();
    let tokens_per_sec = max_new_tokens as f64 / gen_elapsed.as_secs_f64();
    println!(
        "\n--- Performance ---\n{} tokens in {:.2}s = {:.1} tokens/sec ({:.1} ms/token)",
        max_new_tokens,
        gen_elapsed.as_secs_f64(),
        tokens_per_sec,
        gen_elapsed.as_secs_f64() * 1000.0 / max_new_tokens as f64,
    );

    let full_text = tokenizer
        .decode(&generated, true)
        .unwrap_or_else(|_| "decode error".to_string());
    println!("\n--- Full output ---\n{}", full_text);

    if let Some(ref trace_file) = trace_path {
        let path = std::path::Path::new(trace_file);
        meganeura::profiler::save(path).expect("failed to save profile");
        println!("profile saved to {}", path.display());
    }
}
