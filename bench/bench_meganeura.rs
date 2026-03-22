/// Benchmark SmolLM2-135M inference with meganeura.
///
/// Measures per-step latency, tokens/second, time-to-first-token, and
/// prints results as JSON for comparison with bench_pytorch.py.
///
/// Usage:
///   cargo run --release --example bench_meganeura [-- --max-tokens 32 --runs 5]
use std::time::Instant;

use meganeura::{
    Graph, build_inference_session,
    data::safetensors::SafeTensorsModel,
    models::smollm2::{self, SmolLM2Config},
};

const REPO_ID: &str = "HuggingFaceTB/SmolLM2-135M";

fn main() {
    env_logger::init();
    let trace_path = std::env::var("MEGANEURA_TRACE").ok();
    if trace_path.is_some() {
        meganeura::profiler::init();
    }

    let mut args = std::env::args().skip(1);
    let mut prompt = "The meaning of life is".to_string();
    let mut max_tokens: usize = 32;
    let mut warmup: usize = 3;
    let mut runs: usize = 5;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--prompt" => prompt = args.next().expect("--prompt value"),
            "--max-tokens" => {
                max_tokens = args.next().expect("--max-tokens value").parse().unwrap()
            }
            "--warmup" => warmup = args.next().expect("--warmup value").parse().unwrap(),
            "--runs" => runs = args.next().expect("--runs value").parse().unwrap(),
            _ => {
                eprintln!("unknown arg: {}", arg);
                std::process::exit(1);
            }
        }
    }

    let config = SmolLM2Config::smollm2_135m();

    // --- Download model + tokenizer ---
    eprintln!("downloading model...");
    let model = SafeTensorsModel::download(REPO_ID).expect("download failed");

    eprintln!("loading tokenizer...");
    let api = hf_hub::api::sync::Api::new().unwrap();
    let repo = api.model(REPO_ID.to_string());
    let tok_path = repo.get("tokenizer.json").unwrap();
    let tokenizer = tokenizers::Tokenizer::from_file(tok_path).unwrap();

    let encoding = tokenizer.encode(prompt.as_str(), false).unwrap();
    let input_ids: Vec<u32> = encoding.get_ids().to_vec();
    let prompt_len = input_ids.len();
    let seq_len = prompt_len + max_tokens;

    eprintln!(
        "prompt: \"{}\" ({} tokens), seq_len={}",
        prompt, prompt_len, seq_len
    );

    // --- Build & compile graph ---
    eprintln!("building graph...");
    let mut g = Graph::new();
    let logits = smollm2::build_graph(&mut g, &config, seq_len);
    g.set_outputs(vec![logits]);

    eprintln!("compiling...");
    let mut session = build_inference_session(&g);
    eprintln!(
        "ready: {} buffers, {} dispatches",
        session.plan().buffers.len(),
        session.plan().dispatches.len()
    );

    // --- Load weights ---
    eprintln!("loading weights...");
    let transposed = smollm2::transposed_weight_names(&config);
    let transposed_set: std::collections::HashSet<&str> =
        transposed.iter().map(|s| s.as_str()).collect();

    for (name, _) in session.plan().param_buffers.clone() {
        if name == "lm_head.weight" {
            if model.tensor_info().contains_key("lm_head.weight") {
                let data = if transposed_set.contains(name.as_str()) {
                    model.tensor_f32_auto_transposed(&name)
                } else {
                    model.tensor_f32_auto(&name)
                };
                session.set_parameter(&name, &data.unwrap());
            } else {
                let data = model
                    .tensor_f32_auto_transposed("model.embed_tokens.weight")
                    .unwrap();
                session.set_parameter("lm_head.weight", &data);
            }
        } else if transposed_set.contains(name.as_str()) {
            let data = model.tensor_f32_auto_transposed(&name).unwrap();
            session.set_parameter(&name, &data);
        } else {
            let data = model.tensor_f32_auto(&name).unwrap();
            session.set_parameter(&name, &data);
        }
    }
    eprintln!("weights loaded.");

    // --- Generation helper ---
    let vocab = config.vocab_size;

    let generate = |session: &mut meganeura::Session, n_tokens: usize| -> (f64, Vec<u32>) {
        let mut tokens = vec![0u32; seq_len];
        tokens[..input_ids.len()].copy_from_slice(&input_ids);
        let mut generated = input_ids.clone();

        let t0 = Instant::now();
        for step in 0..n_tokens {
            session.set_input_u32("token_ids", &tokens);
            session.step();
            session.wait();

            let all_logits = session.read_output(seq_len * vocab);
            let cur_pos = prompt_len + step;
            let pos_logits = &all_logits[(cur_pos - 1) * vocab..cur_pos * vocab];

            let next = pos_logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0 as u32;

            tokens[cur_pos] = next;
            generated.push(next);
        }
        let elapsed = t0.elapsed().as_secs_f64();
        (elapsed, generated)
    };

    // --- Warmup ---
    eprintln!("warming up ({} runs)...", warmup);
    for _ in 0..warmup {
        generate(&mut session, max_tokens);
    }

    // --- Benchmark ---
    eprintln!(
        "benchmarking ({} runs, {} tokens each)...",
        runs, max_tokens
    );
    let mut latencies = Vec::new();
    let mut ttft_values = Vec::new();
    let mut sample_output = String::new();

    for i in 0..runs {
        let (elapsed, output_ids) = generate(&mut session, max_tokens);
        latencies.push(elapsed);

        // TTFT: generate just 1 token
        let (ttft, _) = generate(&mut session, 1);
        ttft_values.push(ttft);

        let tps = max_tokens as f64 / elapsed;
        eprintln!(
            "  run {}: {:.1}ms, {} tokens, {:.1} tok/s, ttft={:.1}ms",
            i + 1,
            elapsed * 1000.0,
            max_tokens,
            tps,
            ttft * 1000.0
        );

        if i == runs - 1 {
            sample_output = tokenizer.decode(&output_ids, true).unwrap_or_default();
        }
    }

    // --- Statistics ---
    let avg = latencies.iter().sum::<f64>() / runs as f64;
    let mut sorted = latencies.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[runs / 2];
    let stdev = if runs > 1 {
        let var = latencies.iter().map(|l| (l - avg).powi(2)).sum::<f64>() / (runs - 1) as f64;
        var.sqrt()
    } else {
        0.0
    };

    let avg_ttft = ttft_values.iter().sum::<f64>() / runs as f64;
    let mut sorted_ttft = ttft_values.clone();
    sorted_ttft.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_ttft = sorted_ttft[runs / 2];

    let tps = max_tokens as f64 / avg;
    let lpt = avg / max_tokens as f64 * 1000.0;

    // --- JSON output ---
    println!("{{");
    println!("  \"framework\": \"meganeura\",");
    println!("  \"model\": \"{}\",", REPO_ID);
    println!("  \"device\": \"blade-gpu\",");
    println!("  \"dtype\": \"float32\",");
    println!(
        "  \"prompt\": \"{}\",",
        prompt.replace('\\', "\\\\").replace('"', "\\\"")
    );
    println!("  \"prompt_tokens\": {},", prompt_len);
    println!("  \"max_new_tokens\": {},", max_tokens);
    println!("  \"runs\": {},", runs);
    println!("  \"avg_latency_ms\": {:.2},", avg * 1000.0);
    println!("  \"median_latency_ms\": {:.2},", median * 1000.0);
    println!("  \"stdev_latency_ms\": {:.2},", stdev * 1000.0);
    println!("  \"tokens_per_second\": {:.2},", tps);
    println!("  \"latency_per_token_ms\": {:.2},", lpt);
    println!("  \"avg_ttft_ms\": {:.2},", avg_ttft * 1000.0);
    println!("  \"median_ttft_ms\": {:.2},", median_ttft * 1000.0);
    println!(
        "  \"sample_output\": \"{}\"",
        sample_output
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
    );
    println!("}}");

    if let Some(path) = trace_path {
        eprintln!("saving trace to {}...", path);
        meganeura::profiler::save(&path).expect("failed to save trace");
        eprintln!(
            "trace saved ({} events)",
            meganeura::profiler::event_count()
        );
    }
}
