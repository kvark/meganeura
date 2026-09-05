//! Isolated kernel search followed by independent whole-step confirmation.
//!
//! Run on an idle GPU, from a clean tracked checkout:
//! `cargo run --release --example tune_session -- results.json`
//! The output path must not exist. This is a synthetic f32 experiment, not a
//! model benchmark or a comparison with another engine. Native-f32 coverage
//! is reported explicitly; f16-input hardware is not silently substituted.

use meganeura::{
    CoopPolicy, Graph, MatmulTile, Mode, Session, SessionConfig, SessionOptions, TuneOptions, build,
};
use serde_json::{Value, json};
use std::{
    error::Error,
    fs::OpenOptions,
    path::Path,
    process::Command,
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const WARMUPS: usize = 30;
const SAMPLE_PAIRS: usize = 40;

fn command(program: &str, args: &[&str]) -> Result<String, Box<dyn Error>> {
    let output = Command::new(program)
        .args(args)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "{program} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

fn sha256(path: &Path) -> Result<String, Box<dyn Error>> {
    let path = path.to_str().ok_or("non-UTF8 path")?;
    let output =
        command("sha256sum", &[path]).or_else(|_| command("shasum", &["-a", "256", path]))?;
    Ok(output
        .split_whitespace()
        .next()
        .ok_or("missing hash")?
        .to_owned())
}

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    let path = std::env::args_os()
        .nth(1)
        .ok_or("usage: tune_session <new-output.json>")?;
    if path == "--device" {
        let gpu = meganeura::init_gpu_context()?;
        let device = gpu.device_information();
        let caps = &gpu.capabilities().cooperative_matrix;
        println!(
            "{}; {} {}; f32_tile={}; f16_tile={}",
            device.device_name,
            device.driver_name,
            device.driver_info,
            caps.f32_tile,
            caps.f16_tile
        );
        return Ok(());
    }
    let source_status = command("git", &["status", "--porcelain", "--untracked-files=no"])?;
    if !source_status.is_empty() {
        return Err(
            "commit tracked source changes before measuring; results need an exact revision".into(),
        );
    }
    // Reserve the path before doing GPU work. Never overwrite frozen results.
    let output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)?;
    let gpu = Arc::new(meganeura::init_gpu_context()?);
    let device = gpu.device_information();
    let caps = &gpu.capabilities().cooperative_matrix;
    let policy = if caps.f32_tile > 0 {
        CoopPolicy::Auto
    } else {
        CoopPolicy::Disabled
    };
    eprintln!(
        "{}: f32_tile={}, f16_tile={}, policy={policy:?}",
        device.device_name, caps.f32_tile, caps.f16_tile
    );
    let metadata = json!({
        "revision": command("git", &["rev-parse", "HEAD"] )?,
        "tracked_source_clean": true,
        "cargo_lock_sha256": sha256(&Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.lock"))?,
        "executable_sha256": sha256(&std::env::current_exe()?)?,
        "rustc": command("rustc", &["--version"] )?,
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "available_parallelism": std::thread::available_parallelism()?.get(),
        "started_unix_seconds": SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        "device": {
            "name": device.device_name,
            "driver_name": device.driver_name,
            "driver_info": device.driver_info,
            "software_emulated": device.is_software_emulated,
            "f32_tile": caps.f32_tile,
            "f16_tile": caps.f16_tile,
        },
        "cooperative_policy": format!("{policy:?}"),
        "contract": "strict f32 operands/storage; deterministic nonzero dense chains; inference; normal step+wait wall time; uploads, compilation and search excluded from steady-state samples",
        "warmups_per_session": WARMUPS,
        "whole_step_sample_pairs": SAMPLE_PAIRS,
        "order": "alternating baseline/tuned then tuned/baseline; raw arrays remain paired",
    });
    let mut cases = Vec::new();
    for (rows, input_width, width, layers) in [
        (33, 17, 65, 4),
        (32, 256, 256, 8),
        (128, 512, 512, 8),
        (64, 1024, 1024, 4),
    ] {
        cases.push(run_case(&gpu, policy, rows, input_width, width, layers)?);
    }
    serde_json::to_writer_pretty(
        output,
        &json!({"schema_version": 1, "metadata": metadata, "cases": cases}),
    )?;
    eprintln!("wrote {}", Path::new(&path).display());
    Ok(())
}

fn data(count: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(747796405).wrapping_add(2891336453);
            let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277803737);
            (((((word >> 22) ^ word) >> 8) as f32 / 16_777_216.0) - 0.5) * scale
        })
        .collect()
}

fn step_ms(session: &mut Session) -> f64 {
    let start = Instant::now();
    session.step();
    session.wait();
    start.elapsed().as_secs_f64() * 1e3
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) * 0.5
    } else {
        sorted[mid]
    }
}

fn parity(reference: &[f32], output: &[f32]) -> Result<Value, Box<dyn Error>> {
    let mut square_error = 0.0;
    let mut square_reference = 0.0;
    let mut max_abs = 0.0_f64;
    if reference.len() != output.len() {
        return Err("output length changed".into());
    }
    for (&a, &b) in reference.iter().zip(output) {
        if !a.is_finite() || !b.is_finite() {
            return Err("nonfinite graph output".into());
        }
        let difference = (a as f64 - b as f64).abs();
        square_error += difference * difference;
        square_reference += (a as f64).powi(2);
        max_abs = max_abs.max(difference);
    }
    let relative_l2 = (square_error / square_reference.max(1e-30)).sqrt();
    if relative_l2 > 2e-4 {
        return Err(format!("whole-graph parity failed: relative L2 {relative_l2}").into());
    }
    Ok(
        json!({"relative_l2": relative_l2, "max_abs": max_abs, "elements": output.len(), "passed": true}),
    )
}

fn run_case(
    gpu: &Arc<blade_graphics::Context>,
    policy: CoopPolicy,
    rows: usize,
    input_width: usize,
    width: usize,
    layers: usize,
) -> Result<Value, Box<dyn Error>> {
    let mut graph = Graph::new();
    let mut y = graph.input("input", &[rows, input_width]);
    for layer in 0..layers {
        let k = if layer == 0 { input_width } else { width };
        let weight = graph.parameter(&format!("weight_{layer}"), &[k, width]);
        y = graph.matmul(y, weight);
    }
    graph.set_outputs(vec![y]);
    let make_session = || {
        let (mut session, _) = build(
            &graph,
            SessionConfig {
                mode: Mode::Inference,
                gpu: Some(Arc::clone(gpu)),
                runtime: SessionOptions {
                    coop: policy,
                    ..Default::default()
                },
                ..Default::default()
            },
        );
        session.set_input("input", &data(rows * input_width, 1, 1.0));
        for layer in 0..layers {
            let k = if layer == 0 { input_width } else { width };
            session.set_parameter(
                &format!("weight_{layer}"),
                &data(k * width, layer as u32 + 2, (12.0 / k as f32).sqrt()),
            );
        }
        session
    };
    let mut baseline = make_session();
    let mut tuned = make_session();
    let initial_keys = baseline.dispatch_pipeline_keys();
    assert_eq!(initial_keys, tuned.dispatch_pipeline_keys());
    step_ms(&mut baseline);
    step_ms(&mut tuned);
    let reference = baseline.read_output(rows * width);
    let before = tuned.read_output(rows * width);
    let initial_parity = parity(&reference, &before)?;
    let report = tuned.tune_with(TuneOptions {
        max_time: Duration::from_secs(10),
        ..Default::default()
    })?;
    let unchanged = tuned
        .read_output(rows * width)
        .iter()
        .zip(&before)
        .all(|(a, b)| a.to_bits() == b.to_bits());
    if !unchanged {
        return Err("tuning changed live output".into());
    }
    let native_f32_qualified = report.outcomes.iter().any(|o| {
        o.qualified
            && (matches!(o.initial, MatmulTile::CooperativeF32 { .. })
                || matches!(o.candidate, MatmulTile::CooperativeF32 { .. }))
    });
    for _ in 0..WARMUPS {
        step_ms(&mut baseline);
        step_ms(&mut tuned);
    }
    let mut baseline_ms = vec![0.0; SAMPLE_PAIRS];
    let mut tuned_ms = vec![0.0; SAMPLE_PAIRS];
    for pair in 0..SAMPLE_PAIRS {
        if pair % 2 == 0 {
            baseline_ms[pair] = step_ms(&mut baseline);
            tuned_ms[pair] = step_ms(&mut tuned);
        } else {
            tuned_ms[pair] = step_ms(&mut tuned);
            baseline_ms[pair] = step_ms(&mut baseline);
        }
    }
    let final_parity = parity(&reference, &tuned.read_output(rows * width))?;
    let baseline_median = median(&baseline_ms);
    let tuned_median = median(&tuned_ms);
    let differences: Vec<_> = baseline_ms
        .iter()
        .zip(&tuned_ms)
        .map(|(a, b)| a - b)
        .collect();
    let difference_median = median(&differences);
    let noise_margin = 2.0
        * median(
            &differences
                .iter()
                .map(|d| (d - difference_median).abs())
                .collect::<Vec<_>>(),
        );
    let final_keys = tuned.dispatch_pipeline_keys();
    let changed = initial_keys
        .iter()
        .zip(&final_keys)
        .filter(|(a, b)| a != b)
        .count();
    eprintln!(
        "{rows}x{input_width}->{width}, {layers} layers: {changed} dispatches changed; {baseline_median:.4} -> {tuned_median:.4} ms ({:.3}x); search {:.3}s",
        baseline_median / tuned_median,
        report.elapsed.as_secs_f64()
    );
    Ok(json!({
        "rows": rows, "input_width": input_width, "width": width, "layers": layers,
        "baseline_pipeline_keys": initial_keys, "tuned_pipeline_keys": final_keys,
        "dispatches_changed": changed, "native_f32_qualified": native_f32_qualified,
        "initial_parity": initial_parity, "final_parity": final_parity,
        "live_output_unchanged_by_search": unchanged, "search": report,
        "baseline_ms": baseline_ms, "tuned_ms": tuned_ms,
        "baseline_median_ms": baseline_median, "tuned_median_ms": tuned_median,
        "speedup": baseline_median / tuned_median,
        "paired_difference_median_ms": difference_median, "paired_noise_margin_ms": noise_margin,
        "whole_step_improvement_exceeds_guard": baseline_median - tuned_median > baseline_median * 0.05 + noise_margin && difference_median > baseline_median * 0.05 + noise_margin,
    }))
}
