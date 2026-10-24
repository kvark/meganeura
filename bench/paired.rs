//! Shared completed-work comparison for the existing application benchmarks.
use meganeura::{Graph, Session, SessionConfig};
use std::{
    collections::BTreeSet,
    process::Command,
    sync::Arc,
    time::{Duration, Instant},
};

#[derive(Clone, Copy)]
pub struct Options {
    pub enabled: bool,
    pub workers: u32,
    pub warmup: usize,
    pub runs: usize,
    pub warmup_ms: u64,
    pub diagnostics: bool,
    pub device_scheduling: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            enabled: false,
            workers: 64,
            warmup: 10,
            runs: 50,
            warmup_ms: 1000,
            diagnostics: false,
            device_scheduling: false,
        }
    }
}

impl Options {
    pub fn argument(&mut self, arg: &str, args: &mut impl Iterator<Item = String>) -> bool {
        match arg {
            "--paired" => self.enabled = true,
            "--workers" => self.workers = args.next().expect("--workers value").parse().unwrap(),
            "--warmup" => self.warmup = args.next().expect("--warmup value").parse().unwrap(),
            "--runs" => self.runs = args.next().expect("--runs value").parse().unwrap(),
            "--warmup-ms" => {
                self.warmup_ms = args.next().expect("--warmup-ms value").parse().unwrap()
            }
            "--diagnostics" => self.diagnostics = true,
            "--device-scheduling" => {
                self.enabled = true;
                self.device_scheduling = true;
            }
            _ => return false,
        }
        true
    }

    pub fn context(self) -> Arc<blade_graphics::Context> {
        assert!(self.runs > 0 && (1..=65535).contains(&self.workers));
        Arc::new(
            meganeura::init_gpu_context_with(meganeura::GpuOptions {
                timing: self.diagnostics,
                ..meganeura::GpuOptions::from_env()
            })
            .expect("benchmark GPU context"),
        )
    }

    pub fn config(
        self,
        gpu: Arc<blade_graphics::Context>,
        training: bool,
        persistent: bool,
    ) -> SessionConfig<'static> {
        let mut config = SessionConfig::from_env_with_gpu(Some(gpu));
        config.mode = if training {
            meganeura::Mode::Training
        } else {
            meganeura::Mode::Inference
        };
        // Both sides use the same normal compilation and native variant selection.
        // No stochastic per-kernel search can select different arithmetic here.
        config.tune = false;
        config.runtime.gpu_timing = self.diagnostics;
        config.runtime.persistent = persistent.then_some(meganeura::PersistentOptions {
            workers: self.workers,
            allow_fallback: true,
            track_workers: self.diagnostics,
            device_scheduling: self.device_scheduling,
        });
        config
    }

    pub fn build(
        self,
        graph: &Graph,
        gpu: Arc<blade_graphics::Context>,
        training: bool,
        persistent: bool,
    ) -> (Session, f64) {
        let start = Instant::now();
        let session = meganeura::build(graph, self.config(gpu, training, persistent)).0;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        eprintln!("built training={training} persistent={persistent} in {elapsed_ms:.1} ms");
        (session, elapsed_ms)
    }
}

fn quantile(values: &[f64], fraction: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = fraction * (sorted.len() - 1) as f64;
    let low = index.floor() as usize;
    let high = index.ceil() as usize;
    sorted[low] + (sorted[high] - sorted[low]) * index.fract()
}

fn distribution(samples: &[f64]) -> serde_json::Value {
    serde_json::json!({ "samples": samples, "median": quantile(samples, 0.5),
        "p25": quantile(samples, 0.25), "p75": quantile(samples, 0.75), "p95": quantile(samples, 0.95) })
}

fn metadata(session: &mut Session, construction_ms: f64) -> serde_json::Value {
    let memory = session.memory_summary();
    let execution = session.execution_report();
    let workers = session.read_persistent_worker_tasks();
    if let Some(ref rows) = workers {
        for (row, region) in rows.iter().zip(&execution.regions) {
            if region.workers != 0 {
                assert_eq!(
                    row.iter().map(|&count| u64::from(count)).sum::<u64>(),
                    region.logical_tasks
                );
            }
        }
    }
    let statistics: Vec<_> = session
        .get_pipeline_statistics()
        .into_iter()
        .map(|(pipeline, executables)| {
            let executables: Vec<_> = executables.into_iter().map(|executable| {
            let values: Vec<_> = executable.statistics.into_iter().map(|value| serde_json::json!({
                "name": value.name, "description": value.description, "value": value.value,
            })).collect();
            serde_json::json!({"name": executable.name, "statistics": values})
        }).collect();
            serde_json::json!({"pipeline": pipeline, "executables": executables})
        })
        .collect();
    let formats: BTreeSet<_> = session
        .plan()
        .dispatches
        .iter()
        .map(|dispatch| format!("{:?}", dispatch.weight_format))
        .collect();
    serde_json::json!({
        "construction_ms": construction_ms, "execution": execution,
        "compiled_operations": session.plan().dispatches.len(),
        "selected_pipeline_keys": session.dispatch_pipeline_keys(), "weight_formats": formats,
        "selected_implementation_keys": session.dispatch_implementation_keys(),
        "logical_buffer_bytes": memory.total_buffer_bytes, "allocated_buffer_bytes": memory.allocated_buffer_bytes,
        "resident_buffer_bytes": memory.total_allocated_bytes(), "scheduler_bytes": memory.scheduler_bytes,
        "worker_tasks_last_step": workers, "compiled_pipeline_statistics": statistics,
    })
}

/// `prepare` restores matched state outside timing. `execute` owns the exact
/// application boundary and must wait for completion. `observe` qualifies
/// outputs/state after timing, without mutating the next execution's seed.
pub fn measure(
    name: &str,
    ordinary: &mut Session,
    persistent: &mut Session,
    construction_ms: [f64; 2],
    options: Options,
    prepare: impl Fn(&mut Session),
    execute: impl Fn(&mut Session),
    observe: impl Fn(&mut Session) -> Vec<f32>,
) -> serde_json::Value {
    eprintln!("{name}: warming paired sessions");
    assert_eq!(
        ordinary.plan(),
        persistent.plan(),
        "paired sessions must use identical compiled arithmetic and precision"
    );
    assert_eq!(
        ordinary.dispatch_implementation_keys(),
        persistent.dispatch_implementation_keys(),
        "paired sessions must select identical arithmetic implementations"
    );
    let warmup_start = Instant::now();
    let mut warmed = 0;
    while warmed < options.warmup
        || warmup_start.elapsed() < Duration::from_millis(options.warmup_ms)
    {
        for session in [&mut *ordinary, &mut *persistent] {
            prepare(session);
            execute(session);
            session.validate_persistent_completion().unwrap();
        }
        warmed += 1;
    }
    let warmup_elapsed_ms = warmup_start.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "{name}: {warmed} warmup pairs in {warmup_elapsed_ms:.1} ms; measuring {} pairs",
        options.runs
    );
    let mut samples = [Vec::new(), Vec::new()];
    let mut gpu_samples = [Vec::new(), Vec::new()];
    let mut last_gpu_passes = [Vec::new(), Vec::new()];
    let mut max_absolute_error = 0.0f64;
    let mut max_budget_fraction = 0.0f64;
    for pair in 0..options.runs {
        let mut observations = [Vec::new(), Vec::new()];
        for index in [pair % 2, 1 - pair % 2] {
            let session = if index == 0 {
                &mut *ordinary
            } else {
                &mut *persistent
            };
            prepare(session);
            let start = Instant::now();
            execute(session);
            samples[index].push(start.elapsed().as_secs_f64() * 1000.0);
            session.validate_persistent_completion().unwrap();
            if options.diagnostics {
                let times = session.gpu_timings();
                assert!(
                    !times.is_empty(),
                    "diagnostic run has no GPU pass intervals"
                );
                gpu_samples[index].push(
                    times
                        .iter()
                        .map(|entry| entry.1.as_secs_f64() * 1000.0)
                        .sum::<f64>(),
                );
                last_gpu_passes[index] = times.into_iter().map(|(label, duration)| {
                    serde_json::json!({"label": label, "interval_ms": duration.as_secs_f64() * 1000.0})
                }).collect();
            }
            observations[index] = observe(session);
        }
        assert_eq!(observations[0].len(), observations[1].len());
        assert!(
            !observations[0].is_empty(),
            "benchmark must qualify outputs or state"
        );
        for (element, (&a, &b)) in observations[0].iter().zip(&observations[1]).enumerate() {
            assert!(
                a.is_finite() && b.is_finite(),
                "{name} pair {pair}, element {element} is nonfinite"
            );
            let error = (f64::from(a) - f64::from(b)).abs();
            let budget = 1e-5 + 1e-4 * f64::from(a.abs().max(b.abs()));
            max_absolute_error = max_absolute_error.max(error);
            max_budget_fraction = max_budget_fraction.max(error / budget);
            assert!(
                error <= budget,
                "{name} pair {pair}, element {element}: {a} vs {b}"
            );
        }
        if (pair + 1) % 10 == 0 || pair + 1 == options.runs {
            eprintln!("{name}: qualified {}/{} pairs", pair + 1, options.runs);
        }
    }
    let ratios: Vec<_> = samples[0]
        .iter()
        .zip(&samples[1])
        .map(|(a, b)| b / a)
        .collect();
    let command_output = |program: &str, arguments: &[&str]| {
        Command::new(program)
            .args(arguments)
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .output()
            .ok()
            .filter(|output| output.status.success())
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
    };
    let device = ordinary.device_information().clone();
    serde_json::json!({
        "case": name, "sample_pairs": options.runs, "order": "alternating AB / BA",
        "warmup_pairs": warmed, "warmup_elapsed_ms": warmup_elapsed_ms,
        "minimum_warmup_pairs": options.warmup, "minimum_warmup_ms": options.warmup_ms,
        "diagnostic_instrumentation": options.diagnostics, "kernel_search_budget_ms": 0,
        "ordinary_wall_ms": distribution(&samples[0]), "persistent_wall_ms": distribution(&samples[1]),
        "persistent_over_ordinary": distribution(&ratios),
        "ordinary_gpu_pass_intervals_ms": gpu_samples[0], "persistent_gpu_pass_intervals_ms": gpu_samples[1],
        "ordinary_last_gpu_passes": last_gpu_passes[0], "persistent_last_gpu_passes": last_gpu_passes[1],
        "gpu_timing_contract": "Diagnostic only: calibrated pass-start intervals through final submission completion; excludes input uploads and separate readbacks",
        "max_absolute_error": max_absolute_error, "max_budget_fraction": max_budget_fraction,
        "comparison_atol": 1e-5, "comparison_rtol": 1e-4,
        "device": device.device_name, "driver": device.driver_name, "driver_info": device.driver_info,
        "source_revision": command_output("git", &["rev-parse", "HEAD"]),
        "source_status": command_output("git", &["status", "--porcelain"]),
        "rustc": command_output("rustc", &["-Vv"]), "arguments": std::env::args().collect::<Vec<_>>(),
        "construction_contract": "Observed with one shared GPU context and ordinary sessions built first; shader cache reuse is not a matched cold-start comparison",
        "ordinary": metadata(ordinary, construction_ms[0]), "persistent": metadata(persistent, construction_ms[1]),
    })
}
