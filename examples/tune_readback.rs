//! Shared versus read-optimized private staging; no whole-step speed claim.
#[path = "support/experiment_io.rs"]
mod experiment_io;
#[path = "support/gpu_monitor.rs"]
mod gpu_monitor;
#[path = "support/tuning_measurement.rs"]
#[allow(dead_code)] // Whole-step timing arithmetic is only used by the cohort replay.
mod measurement;
#[path = "support/readback_measurement.rs"]
mod readback;
#[path = "support/holdout_workloads.rs"]
mod workloads;

use experiment_io::{command, host_sample, sha256, unix_ms, write_record};
use measurement::TensorComparison;
use meganeura::{CoopPolicy, Session, TuneOptions, TuneStaging};
use readback::{FINAL, MIDDLE, PREFIX, PROCESSES, costs, search_order};
use serde_json::{Value, json};
use std::{error::Error, fs::OpenOptions, path::Path, sync::Arc, time::Duration};
use workloads::*;

fn snapshots(sessions: &mut [Session; 2], case: &Case) -> [Snapshot; 2] {
    let [left, right] = sessions;
    [snapshot(left, case), snapshot(right, case)]
}

fn run_case(
    case: &Case,
    gpu: &Arc<blade_graphics::Context>,
    seed: usize,
    case_index: usize,
) -> Result<Value, Box<dyn Error>> {
    let (shared, shared_build) = make_session(case, gpu, CoopPolicy::Disabled);
    let (download, download_build) = make_session(case, gpu, CoopPolicy::Disabled);
    let mut sessions = [shared, download];
    let keys = sessions[0].dispatch_pipeline_keys();
    assert_eq!(keys, sessions[1].dispatch_pipeline_keys());
    let mut record = json!({"name": case.name, "work": case.work, "description": case.description,
        "status": "running", "build": [shared_build, download_build], "host_start": host_sample(),
        "initial_pipeline_keys": keys, "search_order": search_order(seed, case_index),
        "searches": [null, null], "loss_trajectory": []});
    for _ in 0..PREFIX {
        for session in &mut sessions {
            session.step();
            session.wait();
        }
    }
    let before = snapshots(&mut sessions, case);
    let prefix = compare(
        "prefix",
        case.work.adam_step_after(PREFIX),
        &before[0],
        &before[1],
    );
    let mut valid = prefix.passed;
    record["prefix_comparison"] = serde_json::to_value(prefix)?;
    if !valid {
        record["status"] = json!("prefix_failed");
        return Ok(record);
    }
    for index in search_order(seed, case_index) {
        let staging = [TuneStaging::Shared, TuneStaging::Download][index];
        let session = &mut sessions[index];
        let memory_before = memory(session);
        let host = host_sample();
        let started = unix_ms();
        let report = session.tune_with(TuneOptions {
            staging,
            max_time: Duration::from_secs(10),
            ..Default::default()
        })?;
        let finished = unix_ms();
        let after = snapshot(session, case);
        let comparison = compare(
            "search",
            case.work.adam_step_after(PREFIX),
            &before[index],
            &after,
        );
        valid &= comparison.passed && comparison.exact;
        let costs = costs(&report);
        valid &= costs.is_ok();
        let search_costs = match costs {
            Ok(value) => serde_json::to_value(value)?,
            Err(error) => json!({"error": error}),
        };
        record["searches"][index] = json!({"staging": staging, "host_start": host,
            "started_unix_ms": started, "finished_unix_ms": finished, "report": report,
            "state_comparison": comparison, "cost_ms": search_costs,
            "selected_pipeline_keys": session.dispatch_pipeline_keys(),
            "memory_before": memory_before, "memory_after": memory(session)});
    }
    drop(before);
    for age in PREFIX + 1..=FINAL {
        let first = (age + seed) % 2;
        for index in [first, 1 - first] {
            sessions[index].step();
            sessions[index].wait();
        }
        if case.work != Work::Inference {
            let a = sessions[0].read_loss();
            let b = sessions[1].read_loss();
            valid &= TensorComparison::new("loss".into(), &[a], &[b]).passed;
            record["loss_trajectory"]
                .as_array_mut()
                .unwrap()
                .push(json!({"step": age, "shared": a, "download": b}));
        }
        if age == MIDDLE || age == FINAL {
            let state = snapshots(&mut sessions, case);
            let comparison = compare(
                if age == MIDDLE { "middle" } else { "final" },
                case.work.adam_step_after(age),
                &state[0],
                &state[1],
            );
            valid &= comparison.passed;
            record[if age == MIDDLE {
                "middle_comparison"
            } else {
                "final_comparison"
            }] = serde_json::to_value(comparison)?;
        }
    }
    record["final_memory"] = json!([memory(&sessions[0]), memory(&sessions[1])]);
    record["final_age"] = json!(FINAL);
    record["host_finish"] = host_sample();
    record["status"] = json!(if valid {
        "complete"
    } else {
        "validation_failed"
    });
    eprintln!(
        "{}: shared {} ms, download {} ms; {}",
        case.name,
        record["searches"][0]["cost_ms"]["elapsed"],
        record["searches"][1]["cost_ms"]["elapsed"],
        record["status"]
    );
    Ok(record)
}

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    let mut args = std::env::args_os().skip(1);
    let path = args
        .next()
        .ok_or("usage: tune_readback <new-output.json> <seed 1..6>")?;
    let seed: usize = args
        .next()
        .ok_or("missing seed")?
        .to_str()
        .ok_or("non-UTF8 seed")?
        .parse()?;
    if !(1..=PROCESSES).contains(&seed) || args.next().is_some() {
        return Err("expected one output and seed 1..6".into());
    }
    if !command("git", &["status", "--porcelain", "--untracked-files=no"])?.is_empty() {
        return Err("commit tracked source before measuring".into());
    }
    let mut output = OpenOptions::new().write(true).create_new(true).open(path)?;
    let before = command("nvidia-smi", &[]).ok();
    let monitor = gpu_monitor::Monitor::start();
    let gpu = Arc::new(meganeura::init_gpu_context()?);
    let info = gpu.device_information();
    let caps = &gpu.capabilities().cooperative_matrix;
    let mut document = json!({"schema_version": 1, "protocol": "readback-2026-09-06", "status": "running",
        "metadata": {"revision": command("git", &["rev-parse", "HEAD"])?, "tracked_source_clean": true,
            "cargo_lock_sha256": sha256(&Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.lock"))?,
            "executable_sha256": sha256(&std::env::current_exe()?)?, "rustc": command("rustc", &["--version"])?,
            "seed": seed, "process_id": std::process::id(), "started_unix_ms": unix_ms(),
            "device": {"name": info.device_name, "driver": info.driver_info, "f32_tile": caps.f32_tile, "f16_tile": caps.f16_tile},
            "cooperative_policy": "Disabled", "compile_options": compile_options(),
            "runtime_options": format!("{:?}", runtime_options(CoopPolicy::Disabled)), "optimize": meganeura::optimize::OptimizeConfig::default(),
            "prefix": PREFIX, "middle": MIDDLE, "final": FINAL,
            "contract": "strict f32; private staging placement only; full ordinary/tiny qualification unchanged; no whole-step speed claim",
            "nvidia_smi_before": before, "rustflags": std::env::var("RUSTFLAGS").ok(),
            "optimizer": {"adam_lr": 1e-4, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8, "clip_norm": 1.0, "clip_every": 1, "accumulation": false, "decay": 0.0}}, "cases": []});
    write_record(&mut output, &document)?;
    let mut valid = true;
    for (index, case) in crossover_cases().into_iter().enumerate() {
        eprintln!("running {}", case.name);
        let record = run_case(&case, &gpu, seed, index).unwrap_or_else(
            |e| json!({"name": case.name, "status": "error", "error": e.to_string()}),
        );
        valid &= record["status"] == "complete";
        document["cases"].as_array_mut().unwrap().push(record);
        write_record(&mut output, &document)?;
    }
    document["telemetry"] = monitor.finish();
    document["finished_unix_ms"] = json!(unix_ms());
    document["status"] = json!(if valid {
        "complete"
    } else {
        "validation_failed"
    });
    write_record(&mut output, &document)?;
    if !valid {
        return Err("validation failed; completed records retained".into());
    }
    Ok(())
}
