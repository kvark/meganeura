//! Experimental whole-program selection with private, resettable state.
use crate::{
    Session,
    compile::ExecutionPlan,
    runtime::{
        SessionOptions,
        search_state::{SearchState, persistent_writes},
    },
    tune::{TuneDecision, TuneOptions, TuneOutcome, TuneReport, decide, measure_pairs},
};
use serde::Serialize;
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

pub struct Program {
    pub description: String,
    pub plan: ExecutionPlan,
}

#[derive(Clone, Serialize)]
pub struct Options {
    /// Per-program kernel tuning and paired whole-step decision policy.
    pub tuning: TuneOptions,
    /// Whole-program warmup pairs, independent of private kernel warmup.
    pub warmup_runs: u32,
    /// Soft total deadline, including construction, initialization and validation.
    /// In-flight driver work and caller validation cannot be preempted.
    pub max_time: Duration,
    pub max_programs: usize,
    /// Sum of declared logical bytes and persistent-state snapshots for both
    /// incumbent and challenger. This is not
    /// a driver-heap bound: padding, pipelines, staging and tuning are additional.
    pub max_plan_bytes: usize,
}

#[derive(Serialize)]
pub struct Trial {
    pub description: String,
    /// Full session construction, including allocations, not shader-only compile time.
    pub construction_time: Duration,
    pub initialization_time: Duration,
    /// State capture and restoration, excluded from execution samples.
    pub state_copy_time: Duration,
    pub snapshot_bytes: usize,
    pub qualification_time: Duration,
    pub kernel_tuning: Option<TuneReport>,
    pub outcome: TuneOutcome<(), usize>,
}

#[derive(Serialize)]
pub struct Report {
    pub options: Options,
    pub selected: usize,
    pub trials: Vec<Trial>,
    pub elapsed: Duration,
    pub truncated: bool,
}

fn plan_bytes(plan: &ExecutionPlan) -> Result<usize, String> {
    plan.buffers
        .iter()
        .chain(
            persistent_writes(plan)
                .iter()
                .map(|b| &plan.buffers[b.0 as usize]),
        )
        .try_fold(0usize, |sum, bytes| sum.checked_add((*bytes).max(4)))
        .ok_or_else(|| "declared program bytes overflow".into())
}

fn restore(
    session: &mut Session,
    state: &SearchState,
    elapsed: &mut Duration,
) -> Result<(), String> {
    let start = Instant::now();
    let result = state.restore(session);
    *elapsed += start.elapsed();
    result
}

fn run(session: &mut Session, state: &SearchState, elapsed: &mut Duration) -> Result<f64, String> {
    restore(session, state, elapsed)?;
    let start = Instant::now();
    session.step();
    session.wait();
    Ok(start.elapsed().as_secs_f64() * 1000.0)
}

fn validate(
    session: &mut Session,
    state: &SearchState,
    trial: &mut Trial,
    check: &mut impl FnMut(&mut Session) -> Result<(), String>,
) -> Result<(), String> {
    restore(session, state, &mut trial.state_copy_time)?;
    let start = Instant::now();
    let result = check(session);
    trial.qualification_time += start.elapsed();
    restore(session, state, &mut trial.state_copy_time)?;
    result
}

/// Search complete, legal implementations of the same graph.
/// Each has private state and is kernel-tuned *before* comparing graph forms.
/// `initialize` writes representative inputs/weights once. Its optional idle
/// incumbent permits sharing immutable, identically represented parameters via
/// `Session::share_parameter_from`; it must not modify the incumbent's contents.
/// Inputs, outputs and intermediate buffers remain private. Persistent inputs,
/// parameters and constants written by the plan are snapshotted and reset before
/// every trial, outside timing. Shared mutable parameters are rejected. Runtime
/// optimizers/accumulation must be configured after search; updates explicitly in
/// the plan are supported. Callbacks must not import external writable memory or
/// change execution policy. `qualify` must not upload new inputs or weights; it
/// executes one step and checks every observable
/// output and state change against the
/// caller's numerical contract before tuning, after tuning and after measurements.
/// Completed kernel-class searches are reused only inside this call, with exact
/// geometry, placement, precision, knobs and candidate order. Whole-program
/// validation and timing are never reused.
/// The selected session retains its initialized persistent state, not a tuning
/// trial's advanced cache or weights. Output buffers hold the last qualified step.
///
/// Samples include fresh recording, submission and wait, not output readback.
/// Existing paired-order/noise guards select the incumbent; incomplete pairs
/// never win. Only two sessions coexist. This is bounded exploration, not a
/// global optimum; its selection samples are not held-out benchmark results.
pub fn select(
    programs: impl IntoIterator<Item = Program>,
    gpu: Arc<blade_graphics::Context>,
    runtime: SessionOptions,
    options: Options,
    mut initialize: impl FnMut(&mut Session, Option<&mut Session>) -> Result<(), String>,
    mut qualify: impl FnMut(&mut Session) -> Result<(), String>,
) -> Result<(Session, Report), String> {
    options
        .tuning
        .validate()
        .map_err(|error| error.to_string())?;
    if options.max_programs == 0 || options.max_time.is_zero() {
        return Err("program search needs a positive program and time budget".into());
    }
    let start = Instant::now();
    let mut report = Report {
        options: options.clone(),
        selected: 0,
        trials: Vec::new(),
        elapsed: Duration::ZERO,
        truncated: false,
    };
    let mut incumbent: Option<(Session, SearchState)> = None;
    let mut incumbent_bytes = 0usize;
    let mut kernels = crate::runtime::KernelMemo::default();
    let mut programs = programs.into_iter();
    for index in 0..options.max_programs {
        if start.elapsed() >= options.max_time {
            report.truncated = true;
            break;
        }
        let Some(program) = programs.next() else {
            break;
        };
        let trial_start = Instant::now();
        let bytes = plan_bytes(&program.plan)?;
        let mut trial = Trial {
            description: program.description,
            construction_time: Duration::ZERO,
            initialization_time: Duration::ZERO,
            state_copy_time: Duration::ZERO,
            snapshot_bytes: 0,
            qualification_time: Duration::ZERO,
            kernel_tuning: None,
            outcome: TuneOutcome::new((), program.plan.dispatches.len(), report.selected, index),
        };
        trial.outcome.phase_times = None;
        if bytes
            .checked_add(incumbent_bytes)
            .is_none_or(|sum| sum > options.max_plan_bytes)
        {
            trial.outcome.decision = TuneDecision::ScratchLimit;
        } else {
            let build = Instant::now();
            let mut candidate =
                Session::with_context_opts(program.plan, gpu.clone(), runtime.clone());
            trial.construction_time = build.elapsed();
            let result = (|| {
                let init = Instant::now();
                let initialized =
                    initialize(&mut candidate, incumbent.as_mut().map(|entry| &mut entry.0));
                trial.initialization_time = init.elapsed();
                initialized?;
                let copy_start = Instant::now();
                let state = SearchState::capture(&mut candidate, options.max_plan_bytes)?;
                trial.state_copy_time += copy_start.elapsed();
                trial.snapshot_bytes = state.bytes();
                validate(&mut candidate, &state, &mut trial, &mut qualify)?;
                let mut policy = options.tuning.clone();
                policy.max_time = policy
                    .max_time
                    .min(options.max_time.saturating_sub(start.elapsed()));
                trial.kernel_tuning = Some(
                    candidate
                        .tune_with_memo(policy, Some(&mut kernels))
                        .map_err(|error| error.to_string())?,
                );
                validate(&mut candidate, &state, &mut trial, &mut qualify)?;
                trial.outcome.qualified = true;
                if let Some((ref mut baseline, ref baseline_state)) = incumbent {
                    for _ in 0..options.warmup_runs {
                        if start.elapsed() >= options.max_time {
                            break;
                        }
                        run(baseline, baseline_state, &mut trial.state_copy_time)?;
                        run(&mut candidate, &state, &mut trial.state_copy_time)?;
                    }
                    let mut failure = None;
                    (trial.outcome.baseline_ms, trial.outcome.candidate_ms) =
                        measure_pairs(options.tuning.sample_pairs, |alternative| {
                            if start.elapsed() >= options.max_time {
                                return None;
                            }
                            let result = if alternative {
                                run(&mut candidate, &state, &mut trial.state_copy_time)
                            } else {
                                run(baseline, baseline_state, &mut trial.state_copy_time)
                            };
                            match result {
                                Ok(ms) => Some(ms),
                                Err(error) => {
                                    failure = Some(error);
                                    None
                                }
                            }
                        });
                    restore(baseline, baseline_state, &mut trial.state_copy_time)?;
                    if let Some(error) = failure {
                        return Err(error);
                    }
                    validate(&mut candidate, &state, &mut trial, &mut qualify)?;
                    decide(&mut trial.outcome, &options.tuning);
                } else {
                    trial.outcome.selected = index;
                }
                Ok::<_, String>(state)
            })();
            // A failing challenger is discarded. A failing incumbent invalidates
            // the search: it must not survive as the supposedly safe fallback.
            if let Some((ref mut baseline, ref state)) = incumbent {
                validate(baseline, state, &mut trial, &mut qualify)
                    .map_err(|error| format!("incumbent failed repeated qualification: {error}"))?;
            }
            match result {
                Ok(state) if incumbent.is_none() || trial.outcome.selected == index => {
                    incumbent_bytes = plan_bytes(candidate.plan())?;
                    incumbent = Some((candidate, state));
                    report.selected = index;
                }
                Ok(_) => {}
                Err(error) => {
                    if let Some((ref mut baseline, ref state)) = incumbent {
                        restore(baseline, state, &mut trial.state_copy_time)?;
                    }
                    log::warn!("program {index} qualification failed: {error}");
                    trial.outcome.qualified = false;
                    trial.outcome.selected = report.selected;
                    trial.outcome.decision = TuneDecision::InvalidOutput;
                    trial.outcome.failure = Some(error);
                }
            }
        }
        trial.outcome.elapsed = trial_start.elapsed();
        log::info!(
            "program {index}: {:?}, selected={}, {:?}/{:?} ms",
            trial.outcome.decision,
            trial.outcome.selected,
            trial.outcome.baseline_median_ms,
            trial.outcome.candidate_median_ms
        );
        report.trials.push(trial);
    }
    report.truncated |= start.elapsed() >= options.max_time || programs.next().is_some();
    report.elapsed = start.elapsed();
    incumbent
        .map(|(mut session, state)| {
            state.restore(&mut session)?;
            Ok((session, report))
        })
        .ok_or_else(|| "no qualified program within the search bounds".into())
        .and_then(|result| result)
}

#[cfg(test)]
mod tests {
    #[test]
    #[ignore = "GPU qualification of whole-program search and local kernel reuse"]
    fn repeated_programs_reuse_only_completed_kernel_searches() {
        use super::*;
        let gpu = Arc::new(crate::init_gpu_context_with(crate::GpuOptions::from_env()).unwrap());
        let mut graph = crate::Graph::new();
        let x = graph.input("x", &[33, 17]);
        let w = graph.parameter("w", &[17, 65]);
        let y = graph.matmul(x, w);
        graph.set_outputs(vec![y]);
        let plan = crate::compile::compile(&graph);
        for budget in [Duration::ZERO, Duration::from_secs(30)] {
            let programs = (0..2).map(|i| Program {
                description: i.to_string(),
                plan: plan.clone(),
            });
            let mut shares = 0;
            let (mut session, report) = select(
                programs,
                gpu.clone(),
                SessionOptions {
                    coop: crate::CoopPolicy::Disabled,
                    ..Default::default()
                },
                Options {
                    warmup_runs: 1,
                    tuning: TuneOptions {
                        max_time: budget,
                        sample_pairs: 4,
                        warmup_runs: 1,
                        dispatches_per_sample: 2,
                        ..Default::default()
                    },
                    max_time: Duration::from_secs(60),
                    max_programs: 2,
                    max_plan_bytes: 1 << 20,
                },
                |s, incumbent| {
                    s.set_input("x", &vec![0.25; 33 * 17]);
                    if let Some(incumbent) = incumbent {
                        s.share_parameter_from(incumbent, "w").unwrap();
                        shares += 1;
                    } else {
                        s.set_parameter("w", &vec![0.125; 17 * 65]);
                    }
                    Ok(())
                },
                |s| {
                    s.step();
                    s.wait();
                    let mut output = vec![0.0; 33 * 65];
                    s.read_output_by_index(0, &mut output);
                    if output.iter().all(|&v| v == 17.0 / 32.0) {
                        Ok(())
                    } else {
                        Err("whole-output reference mismatch".into())
                    }
                },
            )
            .unwrap();
            assert_eq!(shares, 1);
            session.step();
            session.wait();
            assert!(
                session
                    .read_output(33 * 65)
                    .iter()
                    .all(|&v| v == 17.0 / 32.0)
            );
            assert!(!report.truncated);
            let first = report.trials[0].kernel_tuning.as_ref().unwrap();
            let second = report.trials[1].kernel_tuning.as_ref().unwrap();
            assert_eq!(first.outcomes.is_empty(), budget.is_zero());
            assert!(first.outcomes.iter().all(|o| o.qualified));
            assert!(first.reused_classes.is_empty());
            assert_eq!(
                second.reused_classes.len(),
                if budget.is_zero() {
                    0
                } else {
                    first.eligible_classes
                }
            );
            assert!(second.outcomes.is_empty());
            assert_eq!(session.read_params(&["w"])[0], vec![0.125; 17 * 65]);
        }
    }

    #[test]
    fn charges_persistent_state_before_allocation() {
        use crate::compile::{BufferRef, Dispatch};
        let mut graph = crate::Graph::new();
        let input = graph.input("x", &[2]);
        let output = graph.neg(input);
        graph.set_outputs(vec![output]);
        let mut plan = crate::compile::compile(&graph);
        let bytes = super::plan_bytes(&plan).unwrap();
        plan.param_grad_pairs.push((BufferRef(0), BufferRef(1)));
        assert_eq!(super::plan_bytes(&plan).unwrap(), bytes);
        plan.param_grad_pairs.clear();
        plan.dispatches.push(Dispatch {
            output_buffer: plan.input_buffers[0].1,
            ..Default::default()
        });
        assert_eq!(super::plan_bytes(&plan).unwrap(), bytes + 8);
    }

    #[test]
    #[ignore = "GPU qualification of stateful complete-plan attention search"]
    fn cached_attention_search_restores_state() {
        use super::*;
        let gpu = Arc::new(crate::init_gpu_context_with(crate::GpuOptions::from_env()).unwrap());
        for (capacity, block, position, window, valid) in
            [(7, 3, 0, 3, 2), (96, 1, 95, 37, 1), (96, 3, 93, 37, 2)]
        {
            let mut graph = crate::Graph::new();
            let q = graph.input("q", &[block, 8]);
            let k = graph.parameter("k", &[capacity, 4]);
            let v = graph.parameter("v", &[capacity, 4]);
            let pos = graph.input_u32("position", &[1]);
            let len = graph.input_u32("valid", &[1]);
            // At position zero a missing reset doubles K and halves V again.
            let new_k = graph.scale(k, 2.0);
            let new_v = graph.scale(v, 0.5);
            let k = graph.cache_write_prefix(new_k, k, pos, len);
            let v = graph.cache_write_prefix(new_v, v, pos, len);
            let output = graph.cached_block_attention(q, k, v, pos, len, 2, 1, 4, window);
            graph.set_outputs(vec![output]);
            let plan = crate::compile::compile(&graph);
            let first = plan.attention_sequences()[0].first;
            let inputs: Vec<f32> = (0..block * 8).map(|i| (i as f32 * 0.3).sin()).collect();
            let original: Vec<Vec<f32>> = [0.2, 0.7]
                .iter()
                .map(|factor| {
                    (0..capacity * 4)
                        .map(|i| (i as f32 * factor).cos())
                        .collect()
                })
                .collect();
            let mut changed = original.clone();
            for (i, &factor) in [2.0, 0.5].iter().enumerate() {
                for row in 0..valid {
                    for col in 0..4 {
                        changed[i][(position + row) * 4 + col] =
                            original[i][row * 4 + col] * factor;
                    }
                }
            }
            let mut expected = Vec::new();
            for row in 0..valid {
                let end = position + row + 1;
                let begin = end.saturating_sub(window as usize);
                for head in 0..2 {
                    let scores: Vec<f64> = (begin..end)
                        .map(|token| {
                            (0..4)
                                .map(|col| {
                                    f64::from(inputs[(row * 2 + head) * 4 + col])
                                        * f64::from(changed[0][token * 4 + col])
                                })
                                .sum::<f64>()
                                / 2.0
                        })
                        .collect();
                    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let weights: Vec<_> = scores.iter().map(|s| (s - max).exp()).collect();
                    let sum: f64 = weights.iter().sum();
                    for col in 0..4 {
                        expected.push(
                            weights
                                .iter()
                                .enumerate()
                                .map(|(i, w)| w * f64::from(changed[1][(begin + i) * 4 + col]))
                                .sum::<f64>()
                                / sum,
                        );
                    }
                }
            }
            let programs = [1, 1, 2, 4, 8, 16, 1].into_iter().map(|splits| {
                let mut plan = plan.clone();
                plan.set_attention_splits(&[(first, splits)], 1 << 20)
                    .unwrap();
                Program {
                    description: splits.to_string(),
                    plan,
                }
            });
            let mut index = 0;
            let (mut selected, report) = select(
                programs,
                gpu.clone(),
                SessionOptions {
                    coop: crate::CoopPolicy::Disabled,
                    ..Default::default()
                },
                Options {
                    tuning: TuneOptions {
                        sample_pairs: 4,
                        max_time: Duration::from_secs(1),
                        ..Default::default()
                    },
                    warmup_runs: 2,
                    max_time: Duration::from_secs(60),
                    max_programs: 7,
                    max_plan_bytes: 1 << 20,
                },
                |s, incumbent| {
                    if let Some(ref baseline) = incumbent {
                        assert_eq!(baseline.read_params(&["k", "v"]), original);
                    }
                    s.set_input("q", &inputs);
                    s.set_input_u32("position", &[position as u32]);
                    s.set_input_u32("valid", &[valid as u32]);
                    s.set_parameter("k", &original[0]);
                    s.set_parameter("v", &original[1]);
                    if index == 0 {
                        s.set_input("q", &vec![f32::NAN; inputs.len()]);
                    } else if index == 6 {
                        s.share_parameter_from(incumbent.unwrap(), "k").unwrap();
                    }
                    index += 1;
                    Ok(())
                },
                |s| {
                    assert_eq!(
                        s.read_params(&["k", "v"]),
                        original,
                        "a trial advanced the cache"
                    );
                    s.step();
                    s.wait();
                    let actual = s.read_output(valid * 8);
                    assert_eq!(s.read_params(&["k", "v"]), changed);
                    if actual
                        .iter()
                        .zip(&expected)
                        .all(|(&a, &b)| a.is_finite() && (f64::from(a) - b).abs() < 2e-5)
                    {
                        Ok(())
                    } else {
                        Err("independent complete-output attention reference mismatch".into())
                    }
                },
            )
            .unwrap();
            assert_eq!(report.trials.len(), 7);
            assert!(!report.truncated);
            assert!(!report.trials[0].outcome.qualified);
            assert!(
                report.trials[1..6]
                    .iter()
                    .all(|t| t.outcome.qualified && t.snapshot_bytes == capacity * 4 * 4 * 2)
            );
            assert!(report.trials[1..6].iter().all(|t| {
                t.kernel_tuning
                    .as_ref()
                    .unwrap()
                    .attention_outcomes
                    .is_empty()
            }));
            assert!(!report.trials[6].outcome.qualified);
            assert!(
                report.trials[6]
                    .outcome
                    .failure
                    .as_ref()
                    .unwrap()
                    .contains("private writable")
            );
            assert_eq!(selected.read_params(&["k", "v"]), original);
            selected.step();
            selected.wait();
            assert_eq!(selected.read_params(&["k", "v"]), changed);
            assert!(
                SearchState::capture(&mut selected, 0)
                    .err()
                    .unwrap()
                    .contains("byte budget")
            );
            selected.set_learning_rate(0.1);
            assert!(
                SearchState::capture(&mut selected, 1 << 20)
                    .err()
                    .unwrap()
                    .contains("optimizers")
            );
        }
    }
}
