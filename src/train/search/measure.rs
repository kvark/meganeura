use super::BuildSearchReport;
use crate::{
    Session,
    compile::ExecutionPlan,
    runtime::{
        SessionOptions,
        search_state::{SearchState, persistent_writes},
    },
    tune::{TuneDecision, TuneOutcome, TuneReport, decide, measure_pairs},
};
use serde::Serialize;
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

pub(super) struct Program {
    pub description: String,
    pub plan: ExecutionPlan,
    pub submission_chunks: usize,
}

#[derive(Serialize)]
pub struct BuildSearchTrial {
    pub description: String,
    /// CPU generation of this plan, when the input iterator lowers lazily.
    pub lowering_time: Duration,
    /// Full session construction, including allocations, not shader-only compile time.
    pub construction_time: Duration,
    pub initialization_time: Duration,
    /// State capture and restoration, excluded from execution samples.
    pub state_copy_time: Duration,
    pub snapshot_bytes: usize,
    pub qualification_time: Duration,
    pub kernel_tuning: Option<TuneReport>,
    /// Private kernel choices rejected by the whole-program check and rolled back.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kernel_tuning_rejection: Option<String>,
    pub outcome: TuneOutcome<(), usize>,
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
    trial: &mut BuildSearchTrial,
    check: &mut impl FnMut(&Session) -> Result<(), String>,
) -> Result<(), String> {
    restore(session, state, &mut trial.state_copy_time)?;
    let start = Instant::now();
    session.step();
    session.wait();
    let result = check(session);
    trial.qualification_time += start.elapsed();
    restore(session, state, &mut trial.state_copy_time)?;
    result
}

pub(super) fn select(
    programs: impl IntoIterator<Item = Program>,
    gpu: Arc<blade_graphics::Context>,
    runtime: SessionOptions,
    mut report: BuildSearchReport,
    start: Instant,
    mut initialize: impl FnMut(&mut Session, Option<&mut Session>) -> Result<(), String>,
    mut qualify: impl FnMut(&Session) -> Result<(), String>,
) -> Result<(Session, BuildSearchReport), String> {
    let options = report.options.clone();
    options
        .tuning
        .validate()
        .map_err(|error| error.to_string())?;
    if options.max_programs == 0 || options.max_time.is_zero() {
        return Err("program search needs a positive program and time budget".into());
    }
    let mut incumbent: Option<(Session, SearchState)> = None;
    let mut incumbent_bytes = 0usize;
    let mut kernels = crate::runtime::KernelMemo::default();
    let mut programs = programs.into_iter();
    for index in 0..options.max_programs {
        if start.elapsed() >= options.max_time {
            report.truncated = true;
            break;
        }
        let lowering = Instant::now();
        let Some(program) = programs.next() else {
            break;
        };
        let lowering_time = lowering.elapsed();
        if start.elapsed() >= options.max_time {
            report.truncated = true;
            break;
        }
        let trial_start = Instant::now();
        let bytes = plan_bytes(&program.plan)?;
        let mut trial = BuildSearchTrial {
            description: program.description,
            lowering_time,
            construction_time: Duration::ZERO,
            initialization_time: Duration::ZERO,
            state_copy_time: Duration::ZERO,
            snapshot_bytes: 0,
            qualification_time: Duration::ZERO,
            kernel_tuning: None,
            kernel_tuning_rejection: None,
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
            candidate.set_submission_chunks(program.submission_chunks);
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
                let untuned = candidate.plan().dispatches.clone();
                let mut policy = options.tuning.clone();
                policy.max_time = policy
                    .max_time
                    .min(options.max_time.saturating_sub(start.elapsed()));
                trial.kernel_tuning = Some(
                    candidate
                        .tune_with_memo(policy, Some(&mut kernels))
                        .map_err(|error| error.to_string())?,
                );
                if let Err(error) = validate(&mut candidate, &state, &mut trial, &mut qualify) {
                    log::warn!("program {index} kernel tuning rejected: {error}");
                    trial.kernel_tuning_rejection = Some(error);
                    candidate.restore_tuning(untuned)?;
                    validate(&mut candidate, &state, &mut trial, &mut qualify)?;
                }
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
                    if incumbent.is_none() {
                        return Err(format!("initial program failed qualification: {error}"));
                    }
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
    // Do not lower one more program just to discover whether a bounded search
    // is truncated. Lazy producers can allocate and compile in `next()`.
    report.truncated |= start.elapsed() >= options.max_time
        || (report.trials.len() == options.max_programs && programs.size_hint().1 != Some(0));
    incumbent
        .map(|(mut session, state)| {
            state.restore(&mut session)?;
            report.elapsed = start.elapsed();
            Ok((session, report))
        })
        .ok_or_else(|| "no qualified program within the search bounds".into())
        .and_then(|result| result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TuneOptions;
    use crate::train::BuildSearchOptions;

    fn select(
        programs: impl IntoIterator<Item = Program>,
        gpu: Arc<blade_graphics::Context>,
        runtime: SessionOptions,
        options: BuildSearchOptions,
        initialize: impl FnMut(&mut Session, Option<&mut Session>) -> Result<(), String>,
        qualify: impl FnMut(&Session) -> Result<(), String>,
    ) -> Result<(Session, BuildSearchReport), String> {
        super::select(
            programs,
            gpu,
            runtime,
            BuildSearchReport {
                options,
                ..Default::default()
            },
            Instant::now(),
            initialize,
            qualify,
        )
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
    fn program_search_preserves_update_and_failure_contracts() {
        use crate::compile::{Dispatch, ShaderEntry};
        let gpu = Arc::new(crate::init_gpu_context_with(crate::GpuOptions::from_env()).unwrap());
        let mut graph = crate::Graph::new();
        let x = graph.input("x", &[2]);
        let w = graph.parameter("w", &[2]);
        let y = graph.mul(x, w);
        let loss = graph.sum_all(y);
        graph.set_outputs(vec![loss]);
        let mut plan = crate::compile::compile(&crate::autodiff::differentiate(&graph));
        let (w, grad) = plan.param_grad_pairs[0];
        plan.dispatches.push(Dispatch {
            shader: ShaderEntry::SgdUpdate,
            input_buffers: vec![w, grad],
            output_buffer: w,
            workgroups: [1, 1, 1],
            params: vec![2, 0.25f32.to_bits(), 0, 0],
            ..Default::default()
        });
        for failure in ["none", "tuning", "restoration", "incumbent"] {
            let initialized = std::cell::Cell::new(0);
            let mut checked = 0;
            let programs = ["baseline", "candidate"].map(|name| {
                let mut plan = plan.clone();
                plan.dispatches[0].label = name.into();
                Program {
                    description: name.into(),
                    plan,
                    submission_chunks: 1,
                }
            });
            let result = select(
                programs,
                gpu.clone(),
                SessionOptions::default(),
                BuildSearchOptions {
                    tuning: TuneOptions {
                        max_time: Duration::ZERO,
                        sample_pairs: 4,
                        ..Default::default()
                    },
                    warmup_runs: 2,
                    max_time: Duration::from_secs(30),
                    max_programs: 2,
                    max_plan_bytes: 1 << 20,
                    ..Default::default()
                },
                |s, _| {
                    initialized.set(initialized.get() + 1);
                    s.set_input("x", &[2.0, 4.0]);
                    s.set_parameter("w", &[3.0, 5.0]);
                    Ok(())
                },
                |s| {
                    checked += 1;
                    if checked == 2 && matches!(failure, "tuning" | "restoration") {
                        return Err("injected post-tuning qualification failure".into());
                    }
                    if checked == 3 && failure == "restoration" {
                        return Err("injected restored-program qualification failure".into());
                    }
                    if failure == "incumbent"
                        && initialized.get() == 2
                        && s.plan().dispatches.iter().any(|d| d.label == "baseline")
                    {
                        return Err("injected incumbent qualification failure".into());
                    }
                    assert_eq!(s.read_loss(), 26.0);
                    let mut gradient = [0.0; 2];
                    s.read_param_grad("w", &mut gradient);
                    assert_eq!(gradient, [2.0, 4.0]);
                    assert_eq!(s.read_params(&["w"])[0], [2.5, 4.0]);
                    Ok(())
                },
            );
            if failure == "incumbent" {
                assert!(
                    result
                        .err()
                        .unwrap()
                        .contains("incumbent failed repeated qualification")
                );
            } else if failure == "restoration" {
                assert!(
                    result
                        .err()
                        .unwrap()
                        .contains("restored-program qualification failure")
                );
            } else {
                let (mut selected, report) = result.unwrap();
                assert!(report.trials.iter().all(|t| t.outcome.qualified));
                assert_eq!(
                    report.trials[0].kernel_tuning_rejection.is_some(),
                    failure == "tuning"
                );
                assert_eq!(selected.read_params(&["w"])[0], [3.0, 5.0]);
                selected.step();
                selected.wait();
                assert_eq!(selected.read_params(&["w"])[0], [2.5, 4.0]);
            }
        }
    }

    #[test]
    fn cached_attention_search_restores_state() {
        let gpu = Arc::new(crate::init_gpu_context_with(crate::GpuOptions::from_env()).unwrap());
        for (capacity, block, position, window, valid, dim) in
            [(7, 3, 0, 3, 2, 4), (96, 1, 95, 37, 1, 4)]
        {
            let mut graph = crate::Graph::new();
            let q = graph.input("q", &[block, 2 * dim]);
            let k = graph.parameter("k", &[capacity, dim]);
            let v = graph.parameter("v", &[capacity, dim]);
            let pos = graph.input_u32("position", &[1]);
            let len = graph.input_u32("valid", &[1]);
            // At position zero a missing reset doubles K and halves V again.
            let new_k = graph.scale(k, 2.0);
            let new_v = graph.scale(v, 0.5);
            let k = graph.cache_write_prefix(new_k, k, pos, len);
            let v = graph.cache_write_prefix(new_v, v, pos, len);
            let output = graph.cached_block_attention(q, k, v, pos, len, 2, 1, dim as u32, window);
            graph.set_outputs(vec![output]);
            // Kernel numerics (unequal scores, ragged windows, wide heads) are
            // covered by gemma_inference_ops. Here uniform attention isolates
            // the search runner's mutation/reset and rejection contract.
            let inputs = vec![0.0; block * 2 * dim];
            let original: Vec<Vec<f32>> = [0.2, 0.7]
                .iter()
                .map(|factor| {
                    (0..capacity * dim)
                        .map(|i| (i as f32 * factor).cos())
                        .collect()
                })
                .collect();
            let mut changed = original.clone();
            for (i, &factor) in [2.0, 0.5].iter().enumerate() {
                for row in 0..valid {
                    for col in 0..dim {
                        changed[i][(position + row) * dim + col] =
                            original[i][row * dim + col] * factor;
                    }
                }
            }
            let mut expected = Vec::new();
            for row in 0..valid {
                let end = position + row + 1;
                let begin = if window == 0 {
                    0
                } else {
                    end.saturating_sub(window as usize)
                };
                for col in 0..2 * dim {
                    expected.push(
                        (begin..end)
                            .map(|token| f64::from(changed[1][token * dim + col % dim]))
                            .sum::<f64>()
                            / (end - begin) as f64,
                    );
                }
            }
            let programs = [1, 1, 2, 4, 8, 16, 1].into_iter().map(|splits| {
                let plan = crate::compile::compile_with(
                    &graph,
                    &crate::compile::CompileOptions {
                        cached_attention_splits: Some(splits),
                        ..Default::default()
                    },
                );
                Program {
                    description: splits.to_string(),
                    plan,
                    submission_chunks: splits as usize,
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
                BuildSearchOptions {
                    tuning: TuneOptions {
                        sample_pairs: 4,
                        max_time: Duration::from_secs(1),
                        ..Default::default()
                    },
                    warmup_runs: 2,
                    max_time: Duration::from_secs(60),
                    max_programs: 7,
                    max_plan_bytes: 1 << 20,
                    ..Default::default()
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
                    if index == 1 {
                        s.set_input("q", &vec![f32::NAN; inputs.len()]);
                    } else if index == 6 {
                        s.share_parameter_from(incumbent.unwrap(), "k").unwrap();
                    }
                    index += 1;
                    Ok(())
                },
                |s| {
                    let actual = s.read_output(valid * 2 * dim);
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
            assert!(report.trials[0].outcome.qualified);
            assert!(!report.trials[1].outcome.qualified);
            assert!(
                report.trials[2..6]
                    .iter()
                    .all(|t| t.outcome.qualified && t.snapshot_bytes == capacity * dim * 4 * 2)
            );
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
