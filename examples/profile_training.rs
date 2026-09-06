//! Fixed whole-step localization, not an optimization or cross-engine benchmark.
#[path = "support/experiment_io.rs"]
mod experiment_io;
#[path = "support/gpu_monitor.rs"]
mod gpu_monitor;
#[path = "support/tuning_measurement.rs"]
#[allow(dead_code)] // Shares tensor arithmetic, not the candidate-pair decision rule.
mod measurement;
#[path = "support/holdout_workloads.rs"]
mod workloads;

use experiment_io::{command, host_sample, sha256, unix_ms, write_record};
use measurement::{TensorComparison, median};
use meganeura::{
    CoopPolicy, GpuOptions, Session, TuneOptions,
    profiler::{CaptureOptions, capture_session_profile},
};
use serde_json::{Value, json};
use std::{error::Error, fs::OpenOptions, path::Path, sync::Arc, time::Duration};
use workloads::*;

const WARMUP: usize = 30;
const SETTLING: usize = 5;
const NORMAL_SAMPLES: usize = 20;
const PROFILE_SAMPLES: usize = 5;

fn profile_cases() -> Vec<Case> {
    cases()
        .into_iter()
        .filter(|case| matches!(case.name, "smollm2-adam" | "whisper-sgd" | "resnet50-flb"))
        .map(|mut case| {
            case.name = match case.name {
                "smollm2-adam" => "smollm2-flb",
                "whisper-sgd" => "whisper-flb",
                name => name,
            };
            case.work = Work::ForwardLossBackward;
            case
        })
        .collect()
}

fn advance(session: &mut Session, count: usize) {
    for _ in 0..count {
        session.step();
        session.wait();
    }
}

fn normal_samples(session: &mut Session, expected_loss: f32) -> Value {
    advance(session, SETTLING);
    let started = host_sample();
    let mut samples = Vec::new();
    let mut losses = Vec::new();
    for _ in 0..NORMAL_SAMPLES {
        samples.push(step_ms(session));
        losses.push(session.read_loss());
    }
    let comparison = TensorComparison::new(
        "loss_trajectory".into(),
        &[expected_loss; NORMAL_SAMPLES],
        &losses,
    );
    json!({"host_start": started, "host_finish": host_sample(), "samples_ms": samples,
        "median_ms": median(&samples), "losses": losses, "loss_comparison": comparison})
}

fn run_case(case: &Case, gpu: &Arc<blade_graphics::Context>) -> Result<Value, Box<dyn Error>> {
    let (mut session, build) = make_session(case, gpu, CoopPolicy::Disabled);
    let keys = session.dispatch_pipeline_keys();
    let mut record = json!({"name": case.name, "description": case.description, "work": case.work,
        "status": "running", "build": build, "host_start": host_sample(),
        "initial_memory": memory(&session), "pipeline_keys": keys,
        "dispatch_contracts": session.plan().dispatches});
    // A zero deadline reports exact eligibility without compiling, allocating or selecting.
    record["search_census"] = serde_json::to_value(session.tune_with(TuneOptions {
        scope: meganeura::TuneScope::Dense,
        max_time: Duration::ZERO,
        ..Default::default()
    })?)?;
    advance(&mut session, WARMUP);
    let reference = snapshot(&mut session, case);
    let expected_loss = session.read_loss();
    let prefix = compare(
        "reference",
        case.work.adam_step_after(WARMUP),
        &reference,
        &reference,
    );
    let mut valid = prefix.passed
        && prefix
            .tensors
            .iter()
            .any(|t| t.name.starts_with("gradient.") && t.reference_sum_sq > 0.0);
    record["reference_comparison"] = serde_json::to_value(prefix)?;
    if !valid {
        record["status"] = json!("reference_failed");
        return Ok(record);
    }
    record["normal_before"] = normal_samples(&mut session, expected_loss);
    let state = snapshot(&mut session, case);
    let before = compare("normal_before", 0, &reference, &state);
    valid &= before.passed;
    record["normal_before_comparison"] = serde_json::to_value(before)?;
    drop(state);

    advance(&mut session, SETTLING);
    let mut preparations = 0;
    let mut comparisons = Vec::new();
    let profile_start = host_sample();
    let profile = capture_session_profile(
        &mut session,
        |session| {
            // The first ring-advance callback sees the retained profiled result,
            // before an ordinary execution can overwrite it. Readbacks use their
            // own encoders and are outside the retained step's wall timer.
            if preparations % 3 == 1 {
                let state = snapshot(session, case);
                let comparison = compare("profiled", 0, &reference, &state);
                valid &= comparison.passed;
                comparisons.push(comparison);
            }
            preparations += 1;
        },
        CaptureOptions {
            samples: PROFILE_SAMPLES,
            unprofiled_median_ms: record["normal_before"]["median_ms"].as_f64(),
            include_pipeline_statistics: true,
        },
    );
    record["profile_host_start"] = profile_start;
    record["profile_host_finish"] = host_sample();
    record["profile_comparisons"] = serde_json::to_value(comparisons)?;
    record["profile_prepare_calls"] = json!(preparations);
    let profile = match profile {
        Ok(profile) => profile,
        Err(error) => {
            record["status"] = json!("profile_error");
            record["error"] = json!(error.to_string());
            return Ok(record);
        }
    };
    assert_eq!(preparations, PROFILE_SAMPLES * 3);
    assert_eq!(
        record["profile_comparisons"].as_array().unwrap().len(),
        PROFILE_SAMPLES
    );
    record["profile"] = serde_json::to_value(profile)?;
    record["normal_after"] = normal_samples(&mut session, expected_loss);
    let after = snapshot(&mut session, case);
    let comparison = compare("normal_after", 0, &reference, &after);
    valid &= comparison.passed
        && record["normal_before"]["loss_comparison"]["passed"] == true
        && record["normal_after"]["loss_comparison"]["passed"] == true;
    record["normal_after_comparison"] = serde_json::to_value(comparison)?;
    assert_eq!(keys, session.dispatch_pipeline_keys());
    record["final_memory"] = memory(&session);
    let same_memory = record["initial_memory"]
        .as_object()
        .unwrap()
        .iter()
        .filter(|(key, _)| key.as_str() != "process_api_sample")
        .all(|(key, value)| value == &record["final_memory"][key]);
    let no_update_state = ["adam_bytes", "accumulator_bytes"]
        .iter()
        .all(|key| record["final_memory"][key] == 0)
        && record["final_memory"]["auxiliary_bytes"] == 4;
    valid &= same_memory && no_update_state;
    record["same_memory"] = json!(same_memory);
    record["host_finish"] = host_sample();
    record["status"] = json!(if valid {
        "complete"
    } else {
        "numerical_failure"
    });
    Ok(record)
}

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    let mut args = std::env::args_os().skip(1);
    let path = args
        .next()
        .ok_or("usage: profile_training <new-output.json> <seed 1..3>")?;
    let seed: usize = args
        .next()
        .ok_or("missing seed")?
        .to_str()
        .ok_or("non-UTF8 seed")?
        .parse()?;
    if !(1..=3).contains(&seed) || args.next().is_some() {
        return Err("expected one output and seed 1..3".into());
    }
    if !command("git", &["status", "--porcelain", "--untracked-files=no"])?.is_empty() {
        return Err("commit tracked source before profiling".into());
    }
    let mut output = OpenOptions::new().write(true).create_new(true).open(path)?;
    let started = unix_ms();
    let before = command("nvidia-smi", &[]).ok();
    let monitor = gpu_monitor::Monitor::start();
    let gpu = Arc::new(meganeura::init_gpu_context_with(GpuOptions {
        timing: true,
        ..Default::default()
    })?);
    let info = gpu.device_information();
    let caps = &gpu.capabilities().cooperative_matrix;
    let mut cases = profile_cases();
    cases.rotate_left(seed - 1);
    let mut document = json!({"schema_version": 1, "protocol": "training-profile-2026-09-06", "status": "running",
        "metadata": {"revision": command("git", &["rev-parse", "HEAD"])?, "tracked_source_clean": true,
            "cargo_lock_sha256": sha256(&Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.lock"))?,
            "executable_sha256": sha256(&std::env::current_exe()?)?, "rustc": command("rustc", &["--version"])?,
            "seed": seed, "process_id": std::process::id(), "started_unix_ms": started,
            "case_order": cases.iter().map(|c| c.name).collect::<Vec<_>>(),
            "device": {"name": info.device_name, "driver": info.driver_info, "f32_tile": caps.f32_tile, "f16_tile": caps.f16_tile},
            "cooperative_policy": "Disabled", "compile_options": compile_options(),
            "runtime_options": format!("{:?}", runtime_options(CoopPolicy::Disabled)), "optimize": meganeura::optimize::OptimizeConfig::default(),
            "warmup": WARMUP, "settling": SETTLING, "normal_samples": NORMAL_SAMPLES, "profile_samples": PROFILE_SAMPLES,
            "contract": "strict scalar f32; fixed-input F+L+B, no optimizer/clip/accumulation; normal step+wait before and after one-pass-per-dispatch capture; all profiled full states checked before ring advance; telemetry active",
            "gpu_timing": true, "nvidia_smi_before": before, "rustflags": std::env::var("RUSTFLAGS").ok()}, "cases": []});
    write_record(&mut output, &document)?;
    let mut valid = true;
    for case in cases {
        eprintln!("profiling {}", case.name);
        let record = run_case(&case, &gpu).unwrap_or_else(
            |error| json!({"name": case.name, "status": "error", "error": error.to_string()}),
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
        return Err("profile or validation failed; completed records retained".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_profile_boundaries_and_balanced_case_order() {
        let cases = profile_cases();
        assert_eq!(
            cases.iter().map(|c| c.name).collect::<Vec<_>>(),
            ["smollm2-flb", "whisper-flb", "resnet50-flb"]
        );
        assert!(cases.iter().all(|c| c.work == Work::ForwardLossBackward));
        for index in 0..3 {
            let mut positions: Vec<_> = (1..=3).map(|seed| (index + 3 - (seed - 1)) % 3).collect();
            positions.sort_unstable();
            assert_eq!(positions, [0, 1, 2]);
        }
    }

    #[test]
    #[ignore = "GPU numerical prefix and timestamp/readback-ring qualification, not retained performance"]
    fn profiling_reads_retained_results_without_advancing_the_session_ring() {
        let gpu = Arc::new(
            meganeura::init_gpu_context_with(GpuOptions {
                timing: true,
                ..Default::default()
            })
            .unwrap(),
        );
        for case in profile_cases() {
            let (mut session, _) = make_session(&case, &gpu, CoopPolicy::Disabled);
            advance(&mut session, 3);
            let reference = snapshot(&mut session, &case);
            let mut preparations = 0;
            let mut checked = 0;
            let profile = capture_session_profile(
                &mut session,
                |session| {
                    if preparations % 3 == 1 {
                        let state = snapshot(session, &case);
                        assert!(compare("profiled", 0, &reference, &state).passed);
                        checked += 1;
                    }
                    preparations += 1;
                },
                CaptureOptions {
                    samples: 2,
                    include_pipeline_statistics: false,
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(checked, 2);
            assert_eq!(preparations, 6);
            assert_eq!(profile.plan.dispatch_count, session.plan().dispatches.len());
            assert_eq!(profile.plan.adam_state_bytes, 0);
            assert_eq!(profile.plan.optimizer_aux_bytes, 4);
            assert!(
                profile
                    .dispatches
                    .iter()
                    .all(|d| d.timing_samples_ms.len() == 2)
            );
        }
    }
}
