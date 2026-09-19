//! Matched GGUF diagnostic; see gguf_latency.cpp for the llama.cpp counterpart.
//! Usage: gguf_latency model.gguf output-prefix [tune-seconds] [all|dense|attention]
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

fn run(session: &mut Session, position: usize, count: usize, vocab: usize) -> (Vec<f32>, [f64; 3]) {
    let tokens: Vec<u32> = (position..position + count)
        .map(|i| 42 + (i % 31) as u32)
        .collect();
    session.set_input_u32("token_ids", &tokens);
    session.set_input_u32("position", &[position as u32]);
    session.set_input_u32("valid", &[count as u32]);
    let start = Instant::now();
    session.step();
    let submitted = Instant::now();
    session.wait();
    let finished = Instant::now();
    let mut logits = vec![0.0; vocab];
    session.read_output_by_index(0, &mut logits);
    assert!(logits.iter().all(|x| x.is_finite()));
    let read = Instant::now();
    (
        logits,
        [
            submitted.duration_since(start).as_secs_f64() * 1000.0,
            finished.duration_since(submitted).as_secs_f64() * 1000.0,
            read.duration_since(finished).as_secs_f64() * 1000.0,
        ],
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let args: Vec<_> = std::env::args().collect();
    assert!(
        (3..=5).contains(&args.len()),
        "gguf_latency model.gguf output-prefix [tune-seconds] [all|dense|attention]"
    );
    let tune_seconds: u64 = args.get(3).map_or(Ok(0), |s| s.parse())?;
    let scope = match args.get(4).map(String::as_str).unwrap_or("all") {
        "all" => meganeura::tune::TuneScope::All,
        "dense" => meganeura::tune::TuneScope::Dense,
        "attention" => meganeura::tune::TuneScope::Attention,
        _ => return Err("expected all, dense or attention tuning scope".into()),
    };
    let started = Instant::now();
    let model = gguf::load_gguf(Path::new(&args[1]))?;
    let config = gguf::arch::ModelConfig::from_gguf(&model)?;
    assert!(
        !config.architecture.uses_per_layer_embeddings(),
        "PLE needs host gathering"
    );
    let mut sessions = Vec::new();
    let mut tuning = Vec::new();
    for block in [1, PROMPT] {
        let mut graph = Graph::new();
        let built = gguf::graph::build(&mut graph, &model, &config, block, CONTEXT)?;
        graph.set_outputs(built.outputs());
        let mut cfg = match sessions.first() {
            Some(s) => SessionConfig::inference_from_env_on(Session::context(s)),
            None => SessionConfig::inference_from_env(),
        };
        cfg.tune = false;
        let mut session = meganeura::build(&graph, cfg).0;
        session.set_submission_chunks(1);
        if tune_seconds != 0 {
            tuning.push(session.tune_with(meganeura::tune::TuneOptions {
                scope,
                max_time: Duration::from_secs(tune_seconds),
                max_classes: 64,
                max_scratch_bytes: 256 * 1024 * 1024,
                ..Default::default()
            })?);
        }
        if let Some(decode) = sessions.first_mut() {
            for (name, _) in session.plan().param_buffers.clone() {
                if decode.has_parameter(&name) {
                    session.share_parameter_from(decode, &name)?;
                }
            }
        } else {
            gguf::weights::load(&mut session, &model, &config)?;
            gguf::weights::reset_caches(&mut session, &built, &config);
        }
        sessions.push(session);
    }
    let mut scheduling = Vec::new();
    if tune_seconds != 0 && matches!(scope, meganeura::tune::TuneScope::All) {
        run(&mut sessions[1], 0, PROMPT, config.vocab_size);
        for pos in PROMPT..PROMPT + DECODE {
            run(&mut sessions[0], pos, 1, config.vocab_size);
        }
        for session in &mut sessions {
            scheduling.push(
                session.tune_submissions(meganeura::tune::TuneSubmissionOptions {
                    max_scratch_bytes: 256 * 1024 * 1024,
                    min_improvement: 0.01,
                    ..Default::default()
                })?,
            );
        }
    }
    let prepare_ms = started.elapsed().as_secs_f64() * 1000.0;
    let mut prefill_ms = Vec::new();
    let mut decode_ms = Vec::new();
    let mut decode_parts_ms = Vec::new();
    let mut outputs = Vec::new();
    for sample in 0..SAMPLES + 3 {
        // Every run overwrites the same cache prefix. Attention masks the suffix.
        let start = Instant::now();
        let (logits, _) = run(&mut sessions[1], 0, PROMPT, config.vocab_size);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        if sample >= 3 {
            prefill_ms.push(elapsed);
        }
        if sample == 3 {
            outputs.extend(logits);
        }
        for pos in PROMPT..PROMPT + DECODE {
            let start = Instant::now();
            let (logits, parts) = run(&mut sessions[0], pos, 1, config.vocab_size);
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
        "prefill_ms": prefill_ms, "decode_ms": decode_ms,
        "decode_record_wait_read_ms": decode_parts_ms,
        "dispatches": [sessions[1].plan().dispatches.len(), sessions[0].plan().dispatches.len()],
        "tuning": tuning,
        "submission_tuning": scheduling,
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
