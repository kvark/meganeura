//! Matched GGUF diagnostic; see gguf_latency.cpp for the llama.cpp counterpart.
//! Usage: gguf_latency model.gguf output-prefix [tune-seconds]
//! GPU selection and precision use the usual SessionConfig environment options.

use meganeura::{Graph, Session, SessionConfig, load::gguf};
use std::{
    path::Path,
    time::{Duration, Instant},
};

const PROMPT: usize = 128;
const DECODE: usize = 32;
const CONTEXT: usize = 256;
const SAMPLES: usize = 7;

fn set_inputs(
    session: &mut Session,
    model: &gguf::GgufModel,
    config: &gguf::arch::ModelConfig,
    position: usize,
    count: usize,
) {
    let tokens: Vec<u32> = (position..position + count)
        .map(|i| 42 + (i % 31) as u32)
        .collect();
    session.set_input_u32("token_ids", &tokens);
    session.set_input_u32("position", &[position as u32]);
    session.set_input_u32("valid", &[count as u32]);
    if config.architecture.uses_per_layer_embeddings() {
        let ple = gguf::weights::gather_per_layer_embeddings(
            model
                .tensors
                .get("per_layer_token_embd.weight")
                .expect("per-layer embedding table"),
            config,
            &tokens,
        )
        .expect("per-layer embedding gather");
        session.set_input("ple", &ple);
    }
}

fn run(
    session: &mut Session,
    model: &gguf::GgufModel,
    config: &gguf::arch::ModelConfig,
    position: usize,
    count: usize,
) -> (Vec<f32>, [f64; 2]) {
    set_inputs(session, model, config, position, count);
    let start = Instant::now();
    session.step();
    let submitted = Instant::now();
    let mut logits = vec![0.0; config.vocab_size];
    session.wait_read_output(0, &mut logits);
    assert!(logits.iter().all(|x| x.is_finite()));
    let read = Instant::now();
    (
        logits,
        [
            submitted.duration_since(start).as_secs_f64() * 1000.0,
            read.duration_since(submitted).as_secs_f64() * 1000.0,
        ],
    )
}

fn read_outputs(session: &Session) -> Vec<Vec<f32>> {
    session.read_buffers(&session.plan().output_buffers)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let args: Vec<_> = std::env::args().collect();
    assert!(
        (3..=4).contains(&args.len()),
        "gguf_latency model.gguf output-prefix [tune-seconds]"
    );
    let tune_seconds: u64 = args.get(3).map_or(Ok(0), |s| s.parse())?;
    let started = Instant::now();
    let model = gguf::load_gguf(Path::new(&args[1]))?;
    let config = gguf::arch::ModelConfig::from_gguf(&model)?;
    let f32_activations = std::env::var_os("MEGANEURA_F32_ACTIVATIONS").is_some();
    let mut sessions = Vec::new();
    let mut tuning = Vec::new();
    for block in [PROMPT, 1] {
        let mut graph = Graph::new();
        let built = gguf::graph::build(&mut graph, &model, &config, block, CONTEXT)?;
        graph.set_outputs(built.outputs());
        let mut cfg = match sessions.first() {
            Some(s) => SessionConfig::inference_from_env_on(Session::context(s)),
            None => SessionConfig::inference_from_env(),
        };
        cfg.tune = false;
        if f32_activations {
            cfg.options.quantized_activations = false;
        }
        let mut session = meganeura::build(&graph, cfg).0;
        gguf::weights::load(&mut session, &model, &config)?;
        gguf::weights::reset_caches(&mut session, &built, &config);
        let position = if block == 1 { PROMPT + DECODE / 2 } else { 0 };
        let mut initial_caches = Vec::new();
        if let Some(prefill) = sessions.first_mut() {
            run(prefill, &model, &config, 0, PROMPT);
            for (name, _) in &prefill.plan().param_buffers {
                if name.starts_with("cache.") {
                    let values = prefill.read_params(&[name])[0].clone();
                    session.set_parameter(name, &values);
                    initial_caches.push((name.clone(), values));
                }
            }
        }
        if tune_seconds != 0 {
            if block == 1 {
                for pos in PROMPT..position {
                    run(&mut session, &model, &config, pos, 1);
                }
                for (name, values) in &mut initial_caches {
                    *values = session.read_params(&[name])[0].clone();
                }
            }
            run(&mut session, &model, &config, position, block);
            let expected = read_outputs(&session);
            let mut cfg = SessionConfig::inference_from_env_on(session.context());
            cfg.tune = false;
            cfg.cache = None;
            if f32_activations {
                cfg.options.quantized_activations = false;
            }
            drop(session);
            let (selected, report) = meganeura::train::build_measured(
                &graph,
                cfg,
                meganeura::train::BuildSearchOptions {
                    max_time: Duration::from_secs(tune_seconds),
                    max_plan_bytes: 4 << 30,
                    tuning: meganeura::TuneOptions {
                        max_classes: 64,
                        max_scratch_bytes: 256 << 20,
                        min_improvement: 0.01,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                |s, donor| {
                    if let Some(source) = donor.or_else(|| sessions.first_mut()).filter(|source| {
                        s.plan().param_buffers.iter().all(|(name, _)| {
                            name.starts_with("cache.") || source.has_parameter(name)
                        })
                    }) {
                        for (name, _) in s.plan().param_buffers.clone() {
                            if !name.starts_with("cache.") {
                                s.share_parameter_from(source, &name)
                                    .map_err(|e| e.to_string())?;
                            }
                        }
                    } else {
                        gguf::weights::load(s, &model, &config).map_err(|e| e.to_string())?;
                    }
                    gguf::weights::reset_caches(s, &built, &config);
                    for (name, values) in &initial_caches {
                        s.set_parameter(name, values);
                    }
                    set_inputs(s, &model, &config, position, block);
                    Ok(())
                },
                |s| {
                    let actual = read_outputs(s);
                    for (index, (a, b)) in actual.iter().zip(&expected).enumerate() {
                        if a.len() != b.len()
                            || a.iter().zip(b).any(|(&a, &b)| {
                                !a.is_finite() || (a - b).abs() > 1e-5 + 1e-4 * b.abs()
                            })
                        {
                            return Err(format!(
                                "full output/cache {index} differs from untuned reference"
                            ));
                        }
                    }
                    Ok(())
                },
            )?;
            session = selected;
            tuning.push(report);
        }
        if let Some(prefill) = sessions.first_mut() {
            for (name, _) in session.plan().param_buffers.clone() {
                if prefill.has_parameter(&name) {
                    session.share_parameter_from(prefill, &name)?;
                }
            }
        }
        sessions.push(session);
    }
    sessions.swap(0, 1);
    let prepare_ms = started.elapsed().as_secs_f64() * 1000.0;
    let mut prefill_ms = Vec::new();
    let mut decode_ms = Vec::new();
    let mut decode_parts_ms = Vec::new();
    let mut outputs = Vec::new();
    for sample in 0..SAMPLES + 3 {
        // Every run overwrites the same cache prefix. Attention masks the suffix.
        let start = Instant::now();
        let (logits, _) = run(&mut sessions[1], &model, &config, 0, PROMPT);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        if sample >= 3 {
            prefill_ms.push(elapsed);
        }
        if sample == 3 {
            outputs.extend(logits);
        }
        for pos in PROMPT..PROMPT + DECODE {
            let start = Instant::now();
            let (logits, parts) = run(&mut sessions[0], &model, &config, pos, 1);
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            if sample >= 3 {
                decode_ms.push(elapsed);
                decode_parts_ms.push(parts);
            }
            if sample == 3 {
                outputs.extend(logits);
            }
        }
    }
    let data: Vec<u8> = outputs.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(format!("{}.logits.f32", args[2]), data)?;
    let result = serde_json::json!({
        "engine": "meganeura", "model": args[1],
        "device": sessions[0].context().device_information().device_name,
        "prompt": PROMPT, "decode": DECODE, "context": CONTEXT, "cache": "f32",
        "vocab": config.vocab_size, "prepare_ms": prepare_ms,
        "activations": if f32_activations { "f32" } else { "q8_1" },
        "prefill_ms": prefill_ms, "decode_ms": decode_ms,
        "decode_record_finish_ms": decode_parts_ms,
        "dispatches": [sessions[1].plan().dispatches.len(), sessions[0].plan().dispatches.len()],
        "tuning": tuning,
    });
    std::fs::write(
        format!("{}.json", args[2]),
        serde_json::to_vec_pretty(&result)?,
    )?;
    if std::env::var_os("MEGANEURA_GPU_TIMING").is_some() {
        for (i, session) in sessions.iter_mut().enumerate() {
            let profile = meganeura::profiler::capture_session_profile(
                session,
                |_| {},
                meganeura::profiler::CaptureOptions::default(),
            )?;
            meganeura::profiler::save_session_profile_json(
                Path::new(&format!("{}.profile-{i}.json", args[2])),
                &profile,
            )?;
        }
    }
    Ok(())
}
