//! Fixed A/A and role-reversed whole-step confirmation. See the retained protocol.
#[path = "support/crossover_measurement.rs"]
mod crossover;
#[path = "support/experiment_io.rs"]
mod experiment_io;
#[path = "support/gpu_monitor.rs"]
mod gpu_monitor;
#[path = "support/tuning_measurement.rs"]
mod measurement;
#[path = "support/holdout_workloads.rs"]
mod workloads;

use crossover::{BLOCK_PAIRS, Block, CONTROL_PAIRS, Confirmation, PREFIX, Pair, SETTLING, WARMUP};
use experiment_io::{command, host_sample, sha256, unix_ms, write_record};
use measurement::TensorComparison;
use meganeura::{CoopPolicy, Session, TuneOptions};
use serde_json::{Value, json};
use std::{
    error::Error,
    fs::OpenOptions,
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};
use workloads::*;

fn snapshots(sessions: &mut [Session; 2], case: &Case) -> [Snapshot; 2] {
    let [left, right] = sessions;
    [snapshot(left, case), snapshot(right, case)]
}

fn advance(sessions: &mut [Session; 2], steps: usize, age: &mut usize) {
    for _ in 0..steps {
        for session in sessions.iter_mut() {
            session.step();
            session.wait();
        }
        *age += 1;
    }
}

fn sample_pairs(
    sessions: &mut [Session; 2],
    work: Work,
    count: usize,
    seed: usize,
    age: &mut usize,
) -> Vec<Pair> {
    (0..count)
        .map(|index| {
            let first_session = (index + seed) % 2;
            let mut ms = [0.0; 2];
            ms[first_session] = step_ms(&mut sessions[first_session]);
            ms[1 - first_session] = step_ms(&mut sessions[1 - first_session]);
            *age += 1;
            Pair {
                step: *age,
                first_session,
                left_ms: ms[0],
                right_ms: ms[1],
                left_loss: (work != Work::Inference).then(|| sessions[0].read_loss()),
                right_loss: (work != Work::Inference).then(|| sessions[1].read_loss()),
            }
        })
        .collect()
}

fn valid_losses(pairs: &[Pair], work: Work) -> bool {
    pairs.iter().all(|p| match (p.left_loss, p.right_loss) {
        (None, None) => work == Work::Inference,
        (Some(a), Some(b)) => {
            work != Work::Inference && TensorComparison::new("loss".into(), &[a], &[b]).passed
        }
        _ => false,
    })
}

fn run_case(
    case: &Case,
    gpu: &Arc<blade_graphics::Context>,
    policy: CoopPolicy,
    seed: usize,
) -> Result<Value, Box<dyn Error>> {
    let (left, left_build) = make_session(case, gpu, policy);
    let (right, right_build) = make_session(case, gpu, policy);
    let mut sessions = [left, right];
    let baseline_keys = sessions[0].dispatch_pipeline_keys();
    assert_eq!(baseline_keys, sessions[1].dispatch_pipeline_keys());
    let mut record = json!({"name": case.name, "description": case.description, "work": case.work,
        "seed": seed, "status": "running", "build": [left_build, right_build],
        "baseline_pipeline_keys": baseline_keys, "initial_memory": [memory(&sessions[0]), memory(&sessions[1])],
        "host_start": host_sample(), "blocks": []});
    let mut age = 0;
    advance(&mut sessions, PREFIX, &mut age);
    let prefix = snapshots(&mut sessions, case);
    let prefix_comparison = compare(
        "prefix",
        case.work.adam_step_after(age),
        &prefix[0],
        &prefix[1],
    );
    let mut valid = prefix_comparison.passed;
    record["prefix_comparison"] = serde_json::to_value(prefix_comparison)?;
    drop(prefix);
    if !valid {
        record["status"] = json!("prefix_failed");
        return Ok(record);
    }
    advance(&mut sessions, WARMUP, &mut age);
    let warm = snapshots(&mut sessions, case);
    let warm_comparison = compare("warmup", case.work.adam_step_after(age), &warm[0], &warm[1]);
    valid &= warm_comparison.passed;
    record["warmup_comparison"] = serde_json::to_value(warm_comparison)?;
    drop(warm);
    if !valid {
        record["status"] = json!("warmup_failed");
        return Ok(record);
    }
    {
        let [left, right] = &mut sessions;
        assert_eq!(left.swap_tuning_with(right)?, 0);
    }
    let aa_host_start = host_sample();
    advance(&mut sessions, SETTLING, &mut age);
    let aa_started_ms = unix_ms();
    let control = sample_pairs(&mut sessions, case.work, CONTROL_PAIRS, seed, &mut age);
    let aa_finished_ms = unix_ms();
    let mut previous = snapshots(&mut sessions, case);
    let aa_comparison = compare(
        "control",
        case.work.adam_step_after(age),
        &previous[0],
        &previous[1],
    );
    valid &= aa_comparison.passed && valid_losses(&control, case.work);
    record["control"] = json!({"host_start": aa_host_start, "started_unix_ms": aa_started_ms,
        "finished_unix_ms": aa_finished_ms, "pairs": control, "comparison": aa_comparison});
    let first_winner = seed % 2;
    let search = sessions[first_winner].tune_with(TuneOptions {
        max_time: Duration::from_secs(10),
        ..Default::default()
    })?;
    let after_search = snapshot(&mut sessions[first_winner], case);
    let isolation = compare(
        "search",
        case.work.adam_step_after(age),
        &previous[first_winner],
        &after_search,
    );
    let isolated = isolation.exact && isolation.passed;
    record["search"] = serde_json::to_value(search)?;
    record["search_state_comparison"] = serde_json::to_value(isolation)?;
    drop(after_search);
    if !isolated {
        record["status"] = json!("search_changed_state");
        return Ok(record);
    }
    let winner_keys = sessions[first_winner].dispatch_pipeline_keys();
    let changed = baseline_keys
        .iter()
        .zip(&winner_keys)
        .filter(|(a, b)| a != b)
        .count();
    record["winner_pipeline_keys"] = json!(winner_keys);
    record["dispatches_changed"] = json!(changed);
    let mut current_winner = first_winner;
    let mut blocks = Vec::new();
    for (index, winner_session) in crossover::winner_order(first_winner)
        .into_iter()
        .enumerate()
    {
        let mut swap_record = Value::Null;
        if winner_session != current_winner {
            let start = Instant::now();
            let [left, right] = &mut sessions;
            assert_eq!(left.swap_tuning_with(right)?, changed);
            let swap_ms = start.elapsed().as_secs_f64() * 1e3;
            let after = snapshots(&mut sessions, case);
            let a = compare(
                "swap_left",
                case.work.adam_step_after(age),
                &previous[0],
                &after[0],
            );
            let b = compare(
                "swap_right",
                case.work.adam_step_after(age),
                &previous[1],
                &after[1],
            );
            let isolated = a.exact && a.passed && b.exact && b.passed;
            swap_record = json!({"elapsed_ms": swap_ms, "left": a, "right": b});
            if !isolated {
                record["status"] = json!("swap_changed_state");
                record["failed_swap"] = swap_record;
                return Ok(record);
            }
            current_winner = winner_session;
        }
        assert_eq!(
            sessions[winner_session].dispatch_pipeline_keys(),
            winner_keys
        );
        assert_eq!(
            sessions[1 - winner_session].dispatch_pipeline_keys(),
            baseline_keys
        );
        let host_start = host_sample();
        advance(&mut sessions, SETTLING, &mut age);
        let started_ms = unix_ms();
        let pairs = sample_pairs(
            &mut sessions,
            case.work,
            BLOCK_PAIRS,
            seed + index,
            &mut age,
        );
        let finished_ms = unix_ms();
        valid &= valid_losses(&pairs, case.work);
        let block = Block {
            winner_session,
            pairs,
        };
        previous = snapshots(&mut sessions, case);
        let comparison = compare(
            "crossover",
            case.work.adam_step_after(age),
            &previous[0],
            &previous[1],
        );
        valid &= comparison.passed;
        record["blocks"].as_array_mut().unwrap().push(
            json!({"index": index, "host_start": host_start,
            "started_unix_ms": started_ms, "finished_unix_ms": finished_ms, "swap": swap_record,
            "samples": block, "comparison": comparison}),
        );
        blocks.push(block);
    }
    let confirmation = Confirmation::new(&control, &blocks, changed, valid)?;
    eprintln!(
        "{}: {} changed; {:.3}x / {:.3}x by winner side; {}",
        case.name,
        changed,
        confirmation.left_winner.speedup,
        confirmation.right_winner.speedup,
        confirmation.decision
    );
    record["confirmation"] = serde_json::to_value(confirmation)?;
    record["status"] = json!(if valid {
        "complete"
    } else {
        "numerical_failure"
    });
    record["final_age"] = json!(age);
    record["final_memory"] = json!([memory(&sessions[0]), memory(&sessions[1])]);
    record["host_finish"] = host_sample();
    Ok(record)
}

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    let mut args = std::env::args_os().skip(1);
    let path = args
        .next()
        .ok_or("usage: tune_crossover <new-output.json> <seed 1..6>")?;
    let seed: usize = args
        .next()
        .ok_or("missing seed")?
        .to_str()
        .ok_or("non-UTF8 seed")?
        .parse()?;
    if !(1..=6).contains(&seed) || args.next().is_some() {
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
    let policy = if caps.f32_tile > 0 {
        CoopPolicy::Auto
    } else {
        CoopPolicy::Disabled
    };
    let mut document = json!({"schema_version": 1, "protocol": "crossover-2026-09-06", "status": "running",
        "metadata": {"revision": command("git", &["rev-parse", "HEAD"])?, "tracked_source_clean": true,
            "cargo_lock_sha256": sha256(&Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.lock"))?,
            "executable_sha256": sha256(&std::env::current_exe()?)?, "rustc": command("rustc", &["--version"])?,
            "seed": seed, "process_id": std::process::id(), "started_unix_ms": unix_ms(),
            "device": {"name": info.device_name, "driver": info.driver_info, "f32_tile": caps.f32_tile, "f16_tile": caps.f16_tile},
            "cooperative_policy": format!("{policy:?}"), "compile_options": compile_options(),
            "runtime_options": format!("{:?}", runtime_options(policy)), "optimize": meganeura::optimize::OptimizeConfig::default(),
            "prefix": PREFIX, "warmup": WARMUP, "settling": SETTLING, "control_pairs": CONTROL_PAIRS, "block_pairs": BLOCK_PAIRS,
            "contract": "strict f32; matched evolving states; step+wait including optimizer/clip; telemetry active; swapping/search/readbacks/settling excluded",
            "nvidia_smi_before": before, "rustflags": std::env::var("RUSTFLAGS").ok(),
            "optimizer": {"adam_lr": 1e-4, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8, "clip_norm": 1.0, "clip_every": 1, "accumulation": false, "decay": 0.0}}, "cases": []});
    write_record(&mut output, &document)?;
    let mut valid = true;
    for (index, case) in crossover_cases().into_iter().enumerate() {
        eprintln!("running {}", case.name);
        let record = run_case(&case, &gpu, policy, seed + index).unwrap_or_else(
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_roster_and_balanced_roles() {
        assert_eq!(
            crossover_cases().iter().map(|c| c.name).collect::<Vec<_>>(),
            ["dense-inference", "mlp-adam", "resnet50-flb"]
        );
        for case in 0..3 {
            assert_eq!((1..=6).filter(|seed| (seed + case) % 2 == 0).count(), 3);
        }
    }

    #[test]
    #[ignore = "GPU numerical preflight; no search or performance confirmation"]
    fn crossover_prefix_and_noop_swap_are_numerically_sound() {
        let gpu = Arc::new(meganeura::init_gpu_context().unwrap());
        for case in crossover_cases() {
            let mut sessions = [
                make_session(&case, &gpu, CoopPolicy::Disabled).0,
                make_session(&case, &gpu, CoopPolicy::Disabled).0,
            ];
            let mut age = 0;
            advance(&mut sessions, PREFIX, &mut age);
            let before = snapshots(&mut sessions, &case);
            assert!(
                compare(
                    "prefix",
                    case.work.adam_step_after(age),
                    &before[0],
                    &before[1]
                )
                .passed
            );
            let [left, right] = &mut sessions;
            assert_eq!(left.swap_tuning_with(right).unwrap(), 0);
            let after = snapshots(&mut sessions, &case);
            for index in 0..2 {
                let result = compare(
                    "swap",
                    case.work.adam_step_after(age),
                    &before[index],
                    &after[index],
                );
                assert!(result.exact && result.passed);
            }
        }
    }
}
