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
    collections::VecDeque,
    sync::Arc,
    time::{Duration, Instant},
};

pub(super) struct Program {
    pub description: String,
    pub plan: ExecutionPlan,
    /// Unspecified candidates retain the incumbent's measured submission choice.
    pub submission_chunks: Option<usize>,
    /// Graph form and physical choice indices of a coordinate frontier.
    pub origin: Option<Origin>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Origin {
    pub form: usize,
    pub choice: usize,
}

/// Candidate programs, possibly ordered by which program currently leads.
pub(super) trait Frontier {
    /// The next program to measure; `incumbent` is the selected program's origin.
    fn next(&mut self, incumbent: Option<Origin>) -> Option<Program>;
    /// Whether a later call could still return a program.
    fn may_remain(&self) -> bool;
}

impl<I: Iterator<Item = Program>> Frontier for std::iter::Fuse<I> {
    fn next(&mut self, _: Option<Origin>) -> Option<Program> {
        Iterator::next(self)
    }

    fn may_remain(&self) -> bool {
        self.size_hint().1 != Some(0)
    }
}

#[derive(Serialize)]
pub struct BuildSearchTrial {
    pub description: String,
    /// CPU generation of this plan, when the input iterator lowers lazily.
    pub lowering_time: Duration,
    /// Full session construction, including allocations, not shader-only compile time.
    pub construction_time: Duration,
    pub initialization_time: Duration,
    /// Per-step input preparation, excluded from execution samples.
    pub input_refresh_time: Duration,
    /// State capture and restoration, excluded from execution samples.
    pub state_copy_time: Duration,
    pub snapshot_bytes: usize,
    pub qualification_time: Duration,
    /// Complete executions checked by the caller's `qualify`.
    pub qualifications: u32,
    /// Time left for whole-program warmup, paired sampling and qualification
    /// when assigning the private kernel budget. An estimate, not extra time.
    pub comparison_time_reserved: Duration,
    /// Absent for the initial, untuned fallback.
    pub kernel_tuning: Option<TuneReport>,
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

fn prepare(
    session: &mut Session,
    state: &SearchState,
    trial: &mut BuildSearchTrial,
    refresh_inputs: &mut impl FnMut(&mut Session) -> Result<(), String>,
) -> Result<(), String> {
    restore(session, state, &mut trial.state_copy_time)?;
    let start = Instant::now();
    let result = refresh_inputs(session);
    trial.input_refresh_time += start.elapsed();
    result
}

fn run(
    session: &mut Session,
    state: &SearchState,
    trial: &mut BuildSearchTrial,
    refresh_inputs: &mut impl FnMut(&mut Session) -> Result<(), String>,
) -> Result<f64, String> {
    prepare(session, state, trial, refresh_inputs)?;
    let start = Instant::now();
    session.step();
    session.wait();
    Ok(start.elapsed().as_secs_f64() * 1000.0)
}

fn validate(
    session: &mut Session,
    state: &SearchState,
    trial: &mut BuildSearchTrial,
    refresh_inputs: &mut impl FnMut(&mut Session) -> Result<(), String>,
    check: &mut impl FnMut(&Session) -> Result<(), String>,
) -> Result<Duration, String> {
    let preparation_before = trial.state_copy_time + trial.input_refresh_time;
    prepare(session, state, trial, refresh_inputs)?;
    let start = Instant::now();
    session.step();
    session.wait();
    let execution = start.elapsed();
    let result = check(session);
    trial.qualification_time += start.elapsed();
    trial.qualifications += 1;
    restore(session, state, &mut trial.state_copy_time)?;
    result
        .map(|()| execution + trial.state_copy_time + trial.input_refresh_time - preparation_before)
}

fn comparison_reserve(
    options: &super::BuildSearchOptions,
    step: Duration,
    qualification: Duration,
) -> Duration {
    // Allow each challenger execution twice the observed incumbent cost.
    // State restoration and input refresh are included in `step`;
    // readback/checking is not.
    let pair = step.saturating_mul(3);
    options
        .warmup_time
        .max(pair.saturating_mul(options.warmup_runs))
        .saturating_add(pair.saturating_mul(options.tuning.sample_pairs as u32))
        .saturating_add(qualification.saturating_mul(2))
}

pub(super) fn select(
    mut programs: impl Frontier,
    gpu: Arc<blade_graphics::Context>,
    runtime: SessionOptions,
    mut report: BuildSearchReport,
    start: Instant,
    callbacks: (
        impl FnMut(&mut Session, Option<&mut Session>) -> Result<(), String>,
        impl FnMut(&mut Session) -> Result<(), String>,
        impl FnMut(&Session) -> Result<(), String>,
    ),
) -> Result<(Session, BuildSearchReport), String> {
    let (mut initialize, mut refresh_inputs, mut qualify) = callbacks;
    let options = report.options.clone();
    options
        .tuning
        .validate()
        .map_err(|error| error.to_string())?;
    if options.max_programs == 0 || options.max_time.is_zero() {
        return Err("program search needs a positive program and time budget".into());
    }
    let mut selection = options.tuning.clone();
    selection.min_improvement = options.min_improvement;
    selection.validate().map_err(|error| error.to_string())?;
    let mut incumbent: Option<(Session, SearchState)> = None;
    let mut incumbent_bytes = 0usize;
    let mut selected_chunks = 1;
    let mut selected_description = String::new();
    let mut selected_plan: Option<ExecutionPlan> = None;
    let mut pending_submissions = VecDeque::new();
    let mut pending_tuned = None;
    let mut initial_tuning_queued = false;
    let mut incumbent_step = Duration::ZERO;
    let mut incumbent_qualification = Duration::ZERO;
    let mut probe_submission = true;
    let mut private = crate::runtime::KernelMemo::default();
    let mut shared = options.kernel_memo.as_ref().map(|memo| {
        memo.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    });
    let kernels = shared.as_deref_mut().unwrap_or(&mut private);
    kernels.bind(&gpu, &options.tuning);
    // Recent complete comparisons, without their private kernel probes.
    let mut comparison_costs = VecDeque::new();
    // Nothing executed since the incumbent's last qualification.
    let mut incumbent_checked = false;
    let mut incumbent_origin = None;
    for index in 0..options.max_programs {
        if start.elapsed() >= options.max_time {
            report.truncated = true;
            break;
        }
        if options
            .patience
            .is_some_and(|patience| index - report.selected > patience)
        {
            report.patience_exhausted = true;
            break;
        }
        // An incomplete comparison cannot select its program.
        if comparison_costs
            .iter()
            .min()
            .is_some_and(|&cost| start.elapsed() + cost > options.max_time)
        {
            report.truncated = true;
            report.deadline_reserved = true;
            break;
        }
        let lowering = Instant::now();
        // Cover the first layouts, then alternate the two frontiers so submission
        // probes cannot consume the program budget before useful layouts arrive.
        let program = pending_tuned.take().or_else(|| {
            (index < super::INITIAL_LAYOUTS + usize::from(initial_tuning_queued)
                || !probe_submission)
                .then(|| programs.next(incumbent_origin))
                .flatten()
                .or_else(|| {
                    pending_submissions
                        .pop_front()
                        .map(|submission_chunks| Program {
                            description: selected_description.clone(),
                            plan: selected_plan.as_ref().unwrap().clone(),
                            submission_chunks: Some(submission_chunks),
                            origin: incumbent_origin,
                        })
                })
                .or_else(|| programs.next(incumbent_origin))
        });
        let Some(program) = program else {
            break;
        };
        probe_submission = program.submission_chunks.is_none();
        let lowering_time = lowering.elapsed();
        if start.elapsed() >= options.max_time {
            report.truncated = true;
            break;
        }
        let trial_start = Instant::now();
        let bytes = plan_bytes(&program.plan)?;
        let submission_chunks = program.submission_chunks.unwrap_or(selected_chunks);
        let mut trial = BuildSearchTrial {
            description: format!(
                "{}, {}submission_chunks={submission_chunks}",
                program.description,
                if incumbent.is_none() { "untuned, " } else { "" }
            ),
            lowering_time,
            construction_time: Duration::ZERO,
            initialization_time: Duration::ZERO,
            input_refresh_time: Duration::ZERO,
            state_copy_time: Duration::ZERO,
            snapshot_bytes: 0,
            qualification_time: Duration::ZERO,
            qualifications: 0,
            comparison_time_reserved: Duration::ZERO,
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
            // Preserve the original kernel candidates for memo reuse by submission probes.
            let original_plan = program.plan.clone();
            let mut candidate = match incumbent {
                Some((ref mut baseline, _)) => Session::with_context_opts_inheriting(
                    program.plan,
                    gpu.clone(),
                    runtime.clone(),
                    baseline,
                ),
                None => Session::with_context_opts(program.plan, gpu.clone(), runtime.clone()),
            };
            candidate.set_submission_chunks(submission_chunks);
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
                if incumbent.is_some() {
                    let mut policy = options.tuning.clone();
                    // Do not spend kernel probes on an invalid program.
                    // Memoized winners still need whole-program acceptance.
                    let mut step = incumbent_step;
                    if !policy.max_time.is_zero()
                        && candidate.kernel_probes_pending(&policy, Some(kernels))
                    {
                        step = step.max(validate(
                            &mut candidate,
                            &state,
                            &mut trial,
                            &mut refresh_inputs,
                            &mut qualify,
                        )?);
                    }
                    let qualification = incumbent_qualification.max(trial.qualification_time);
                    trial.comparison_time_reserved =
                        comparison_reserve(&options, step, qualification);
                    policy.max_time = policy.max_time.min(
                        options
                            .max_time
                            .saturating_sub(start.elapsed())
                            .saturating_sub(trial.comparison_time_reserved),
                    );
                    trial.kernel_tuning = Some(
                        candidate
                            .tune_with_memo(policy, Some(&mut *kernels))
                            .map_err(|error| error.to_string())?,
                    );
                }
                let step = validate(
                    &mut candidate,
                    &state,
                    &mut trial,
                    &mut refresh_inputs,
                    &mut qualify,
                )?;
                trial.outcome.qualified = true;
                if let Some((ref mut baseline, ref baseline_state)) = incumbent {
                    let warmup_start = Instant::now();
                    let mut warmed = 0;
                    while warmed < options.warmup_runs
                        || warmup_start.elapsed() < options.warmup_time
                    {
                        if start.elapsed() >= options.max_time {
                            break;
                        }
                        run(baseline, baseline_state, &mut trial, &mut refresh_inputs)?;
                        run(&mut candidate, &state, &mut trial, &mut refresh_inputs)?;
                        warmed += 1;
                    }
                    let mut failure = None;
                    (trial.outcome.baseline_ms, trial.outcome.candidate_ms) =
                        measure_pairs(options.tuning.sample_pairs, |alternative| {
                            if start.elapsed() >= options.max_time {
                                return None;
                            }
                            let result = if alternative {
                                run(&mut candidate, &state, &mut trial, &mut refresh_inputs)
                            } else {
                                run(baseline, baseline_state, &mut trial, &mut refresh_inputs)
                            };
                            match result {
                                Ok(ms) if start.elapsed() < options.max_time => Some(ms),
                                Ok(_) => None,
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
                    decide(&mut trial.outcome, &selection);
                    // A rejected challenger is discarded either way. Confirm a
                    // winner after its repeated execution, before it replaces
                    // the incumbent.
                    if trial.outcome.selected == index {
                        validate(
                            &mut candidate,
                            &state,
                            &mut trial,
                            &mut refresh_inputs,
                            &mut qualify,
                        )?;
                    }
                } else {
                    trial.outcome.selected = index;
                }
                Ok::<_, String>((state, step))
            })();
            // A challenger executed beside the incumbent, sharing its weights.
            incumbent_checked = false;
            match result {
                Ok((state, step)) if incumbent.is_none() || trial.outcome.selected == index => {
                    // The first program is a qualified ordinary fallback.
                    // Tuning it, including memo reuse, is a separate challenger
                    // charged to the same program, time and memory bounds.
                    if incumbent.is_none() && candidate.has_tunable_kernels(&options.tuning) {
                        pending_tuned = Some(Program {
                            description: program.description.clone(),
                            plan: original_plan.clone(),
                            submission_chunks: program.submission_chunks,
                            origin: program.origin,
                        });
                        initial_tuning_queued = true;
                    }
                    incumbent_step = step;
                    incumbent_qualification = trial.qualification_time / trial.qualifications;
                    // Its last execution was its qualification.
                    incumbent_checked = true;
                    incumbent_bytes = plan_bytes(candidate.plan())?;
                    incumbent = Some((candidate, state));
                    selected_chunks = submission_chunks;
                    selected_description = program.description;
                    selected_plan = Some(original_plan);
                    // A program without an origin keeps the coordinate frontier
                    // where it was.
                    incumbent_origin = program.origin.or(incumbent_origin);
                    if program.submission_chunks.is_none() {
                        pending_submissions =
                            std::iter::successors(Some(1usize), |n| n.checked_mul(2))
                                .take_while(|&n| n <= options.max_submission_chunks)
                                .filter(|&n| n != selected_chunks)
                                .collect();
                    }
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
        if trial.outcome.baseline_ms.len() == options.tuning.sample_pairs
            && trial.outcome.candidate_ms.len() == options.tuning.sample_pairs
        {
            let probes = trial
                .kernel_tuning
                .as_ref()
                .map_or(Duration::ZERO, |t| t.elapsed);
            comparison_costs
                .push_back((trial.lowering_time + trial.outcome.elapsed).saturating_sub(probes));
            if comparison_costs.len() > 3 {
                comparison_costs.pop_front();
            }
        }
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
        || pending_tuned.is_some()
        || !pending_submissions.is_empty()
        || ((report.patience_exhausted || report.trials.len() == options.max_programs)
            && programs.may_remain());
    let (mut session, state) = incumbent.ok_or("no qualified program within the search bounds")?;
    // Requalify an incumbent that challengers ran beside since its own check.
    // A failing incumbent invalidates the search: it must not survive as the
    // supposedly safe fallback. A replaced one was discarded already.
    if !incumbent_checked {
        let check = Instant::now();
        state.restore(&mut session)?;
        refresh_inputs(&mut session)
            .map_err(|error| format!("incumbent input refresh failed: {error}"))?;
        session.step();
        session.wait();
        let result = qualify(&session);
        report.final_qualification_time = check.elapsed();
        result.map_err(|error| format!("incumbent failed repeated qualification: {error}"))?;
    }
    state.restore(&mut session)?;
    report.elapsed = start.elapsed();
    Ok((session, report))
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
            programs.into_iter().fuse(),
            gpu,
            runtime,
            BuildSearchReport {
                options,
                ..Default::default()
            },
            Instant::now(),
            (initialize, |_| Ok(()), qualify),
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
        use crate::compile::{Dispatch, Kernel, ShaderEntry};
        use crate::schedule::{PointwiseDAG, Pw};
        let gpu = Arc::new(crate::init_gpu_context_with(crate::GpuOptions::from_env()).unwrap());
        let mut graph = crate::Graph::new();
        let x = graph.input("x", &[2]);
        let w = graph.parameter("w", &[2]);
        let y = graph.mul(x, w);
        let loss = graph.sum_all(y);
        graph.set_outputs(vec![loss]);
        let mut plan = crate::compile::compile(&crate::autodiff::differentiate(&graph));
        // An in-program update: w -= 0.25 * grad.
        let (w, grad) = plan.param_grad_pairs[0];
        plan.dispatches.push(Dispatch {
            shader: ShaderEntry::Generated,
            input_buffers: vec![w, grad],
            output_buffer: w,
            workgroups: [1, 1, 1],
            params: vec![2, 0, 0, 0],
            kernel: Kernel::Pointwise(PointwiseDAG {
                n_inputs: 2,
                ops: vec![
                    Pw::LoadInput(0),
                    Pw::LoadInput(1),
                    Pw::Const(0.25f32.to_bits()),
                    Pw::Mul(1, 2),
                    Pw::Sub(0, 3),
                ],
                output: 4,
            }),
            ..Default::default()
        });
        for invalidate in [false, true] {
            let initialized = std::cell::Cell::new(0);
            let programs = ["baseline", "candidate"].map(|name| {
                let mut plan = plan.clone();
                plan.dispatches[0].label = name.into();
                Program {
                    description: name.into(),
                    plan,
                    submission_chunks: (name == "baseline").then_some(2),
                    origin: None,
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
                    // Identical programs: keep the first, so the injected
                    // failure reaches the final incumbent check.
                    min_improvement: 0.99,
                    warmup_runs: 2,
                    warmup_time: Duration::from_millis(1),
                    max_time: Duration::from_secs(30),
                    max_programs: 2,
                    max_submission_chunks: 2,
                    max_plan_bytes: 1 << 20,
                    ..Default::default()
                },
                |s, incumbent| {
                    // An in-program update makes `w` search state, never shared.
                    assert!(
                        incumbent.is_none_or(|incumbent| !s.inherits_parameter(incumbent, "w"))
                    );
                    initialized.set(initialized.get() + 1);
                    s.set_input("x", &[2.0, 4.0]);
                    s.set_parameter("w", &[3.0, 5.0]);
                    Ok(())
                },
                |s| {
                    if invalidate
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
            if invalidate {
                assert!(
                    result
                        .err()
                        .unwrap()
                        .contains("incumbent failed repeated qualification")
                );
            } else {
                let (mut selected, report) = result.unwrap();
                assert!(report.trials.iter().all(|t| t.outcome.qualified));
                // Nothing to probe, and the challenger lost: one check each.
                assert!(report.trials.iter().all(|t| t.qualifications == 1));
                assert!(
                    report
                        .trials
                        .iter()
                        .all(|t| t.description.ends_with("submission_chunks=2"))
                );
                assert_eq!(selected.read_params(&["w"])[0], [3.0, 5.0]);
                selected.step();
                selected.wait();
                assert_eq!(selected.read_params(&["w"])[0], [2.5, 4.0]);
            }
        }
    }

    #[test]
    fn challengers_inherit_immutable_and_derived_parameters() {
        let gpu = Arc::new(crate::init_gpu_context_with(crate::GpuOptions::from_env()).unwrap());
        let (rows, inner, cols) = (3, 12, 8);
        let mut graph = crate::Graph::new();
        let x = graph.input("x", &[rows, inner]);
        let gate = graph.parameter("gate", &[inner, cols]);
        let up = graph.parameter("up", &[inner, cols]);
        let (gate, up) = (graph.matmul(x, gate), graph.matmul(x, up));
        let y = graph.swiglu(gate, up);
        graph.set_outputs(vec![y]);
        // The ordinary optimizer packs both projections into one derived weight.
        let (optimized, _) = crate::optimize::optimize_with_config(&graph, Default::default());
        let plan = crate::compile::compile(&optimized);
        assert_eq!(plan.derived_params.len(), 1);
        let values = |seed: usize, n: usize| -> Vec<f32> {
            (0..n)
                .map(|i| ((i * 7 + seed) % 11) as f32 * 0.1 - 0.5)
                .collect()
        };
        let (input, weights) = (
            values(0, rows * inner),
            [
                ("gate", values(1, inner * cols)),
                ("up", values(2, inner * cols)),
            ],
        );
        let product = |w: &[f32], r: usize, c: usize| -> f64 {
            (0..inner)
                .map(|k| f64::from(input[r * inner + k]) * f64::from(w[k * cols + c]))
                .sum()
        };
        let expected: Vec<f64> = (0..rows * cols)
            .map(|i| {
                let (a, b) = (
                    product(&weights[0].1, i / cols, i % cols),
                    product(&weights[1].1, i / cols, i % cols),
                );
                a / (1.0 + (-a).exp()) * b
            })
            .collect();
        let programs = (0..3).map(|index| Program {
            description: index.to_string(),
            plan: plan.clone(),
            submission_chunks: Some(1),
            origin: None,
        });
        let uploads = std::cell::Cell::new(0);
        let (mut selected, report) = select(
            programs,
            gpu,
            SessionOptions::default(),
            BuildSearchOptions {
                tuning: TuneOptions {
                    max_time: Duration::ZERO,
                    sample_pairs: 4,
                    ..Default::default()
                },
                warmup_runs: 1,
                warmup_time: Duration::from_millis(1),
                max_time: Duration::from_secs(30),
                max_programs: 3,
                max_plan_bytes: 1 << 20,
                ..Default::default()
            },
            |s, incumbent| {
                s.set_input("x", &input);
                for &(name, ref data) in &weights {
                    if incumbent
                        .as_deref()
                        .is_some_and(|incumbent| s.inherits_parameter(incumbent, name))
                    {
                        continue;
                    }
                    uploads.set(uploads.get() + 1);
                    s.set_parameter(name, data);
                }
                Ok(())
            },
            |s| {
                let actual = s.read_output(rows * cols);
                if actual
                    .iter()
                    .zip(&expected)
                    .all(|(&a, &b)| (f64::from(a) - b).abs() < 1e-4)
                {
                    Ok(())
                } else {
                    Err(format!("SwiGLU mismatch: {actual:?} vs {expected:?}"))
                }
            },
        )
        .unwrap();
        // Only the first program uploads: challengers inherit both sources and
        // the weight derived from them.
        assert_eq!(uploads.get(), weights.len());
        assert_eq!(report.trials.len(), 3);
        assert!(report.trials.iter().all(|t| t.outcome.qualified));
        selected.step();
        selected.wait();
        assert!(
            selected
                .read_output(rows * cols)
                .iter()
                .zip(&expected)
                .all(|(&a, &b)| (f64::from(a) - b).abs() < 1e-4)
        );
    }

    #[test]
    fn partial_reuploads_preserve_inherited_packed_parameters() {
        let gpu = Arc::new(crate::init_gpu_context_with(crate::GpuOptions::from_env()).unwrap());
        let (rows, inner, cols) = (3, 32, 3);
        let expected = 4.0 / (1.0 + (-4.0_f32).exp()) * 2.0;
        for f16 in [true, false] {
            let mut graph = crate::Graph::new();
            let x = graph.input("x", &[rows, inner]);
            let mut parameter = |name| {
                if f16 {
                    graph.parameter_f16(name, &[inner, cols])
                } else {
                    graph.parameter_q4(name, &[inner, cols])
                }
            };
            let (gate, up) = (parameter("gate"), parameter("up"));
            let (gate, up) = (graph.matmul(x, gate), graph.matmul(x, up));
            let y = graph.swiglu(gate, up);
            graph.set_outputs(vec![y]);
            let (optimized, _) = crate::optimize::optimize_with_config(&graph, Default::default());
            let plan = crate::compile::compile(&optimized);
            assert_eq!(plan.derived_params.len(), 1);
            for device in [false, true] {
                let initialized = std::cell::Cell::new(0);
                let skipped = std::cell::Cell::new(0);
                let programs = (0..3).map(|index| Program {
                    description: index.to_string(),
                    plan: plan.clone(),
                    submission_chunks: Some(1),
                    origin: None,
                });
                let (mut selected, report) = select(
                    programs,
                    gpu.clone(),
                    SessionOptions {
                        coop: crate::CoopPolicy::Disabled,
                        device_parameters: device.then_some(blade_graphics::Memory::Device),
                        ..Default::default()
                    },
                    BuildSearchOptions {
                        tuning: TuneOptions {
                            max_time: Duration::ZERO,
                            sample_pairs: 4,
                            ..Default::default()
                        },
                        min_improvement: 0.99,
                        warmup_runs: 1,
                        warmup_time: Duration::from_millis(1),
                        max_time: Duration::from_secs(30),
                        max_programs: 3,
                        max_plan_bytes: 1 << 20,
                        ..Default::default()
                    },
                    |s, incumbent| {
                        let trial = initialized.get();
                        initialized.set(trial + 1);
                        s.set_input("x", &vec![0.25; rows * inner]);
                        for (index, (name, value)) in
                            [("gate", 0.5), ("up", 0.25)].into_iter().enumerate()
                        {
                            if let Some(donor) = incumbent.as_deref() {
                                assert!(s.inherits_parameter(donor, name));
                                // Each challenger reuploads one source, with
                                // unchanged values, and skips its sibling.
                                if index == trial % 2 {
                                    skipped.set(skipped.get() + 1);
                                    continue;
                                }
                            }
                            let data = vec![value; inner * cols];
                            if trial == 2 {
                                let packed = if f16 {
                                    data.iter()
                                        .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
                                        .collect()
                                } else {
                                    crate::runtime::quantize_q4_0(&data, inner, cols)
                                };
                                s.set_parameter_packed(name, &packed);
                            } else {
                                s.set_parameter(name, &data);
                            }
                        }
                        Ok(())
                    },
                    |s| {
                        let actual = s.read_output(rows * cols);
                        if actual.iter().all(|&v| (v - expected).abs() < 1e-5) {
                            Ok(())
                        } else {
                            Err(format!("SwiGLU mismatch: {actual:?} vs {expected}"))
                        }
                    },
                )
                .unwrap();
                assert_eq!(skipped.get(), 2);
                assert_eq!(report.trials.len(), 3);
                assert!(report.trials.iter().all(|t| t.outcome.qualified));
                selected.step();
                selected.wait();
                assert!(
                    selected
                        .read_output(rows * cols)
                        .iter()
                        .all(|&v| (v - expected).abs() < 1e-5)
                );
            }
        }
    }

    #[test]
    fn later_searches_resume_shared_kernel_decisions() {
        let gpu = Arc::new(crate::init_gpu_context_with(crate::GpuOptions::from_env()).unwrap());
        let mut graph = crate::Graph::new();
        let x = graph.input("x", &[33, 17]);
        let w = graph.parameter("w", &[17, 65]);
        let y = graph.matmul(x, w);
        graph.set_outputs(vec![y]);
        let plan = crate::compile::compile(&graph);
        let memo = Arc::new(std::sync::Mutex::new(crate::runtime::KernelMemo::default()));
        let search = || {
            select(
                [Program {
                    description: "ordinary".into(),
                    plan: plan.clone(),
                    submission_chunks: Some(1),
                    origin: None,
                }],
                gpu.clone(),
                SessionOptions {
                    coop: crate::CoopPolicy::Disabled,
                    ..Default::default()
                },
                BuildSearchOptions {
                    tuning: TuneOptions {
                        max_time: Duration::from_secs(30),
                        sample_pairs: 4,
                        target_sample_time: Some(Duration::from_nanos(1)),
                        ..Default::default()
                    },
                    max_time: Duration::from_secs(60),
                    max_programs: 2,
                    // Private kernel winners must not bypass whole-model
                    // selection, including when loaded from the shared memo.
                    min_improvement: 0.99,
                    max_plan_bytes: 1 << 20,
                    kernel_memo: Some(memo.clone()),
                    ..Default::default()
                },
                |s, _| {
                    s.set_input("x", &[0.25; 33 * 17]);
                    s.set_parameter("w", &[0.125; 17 * 65]);
                    Ok(())
                },
                |s| {
                    let output = s.read_output(33 * 65);
                    if output.iter().all(|&v| (v - 0.53125).abs() < 1e-5) {
                        Ok(())
                    } else {
                        Err("matmul mismatch".into())
                    }
                },
            )
            .unwrap()
            .1
        };
        let first = search();
        assert_eq!(first.selected, 0);
        assert!(first.trials[0].kernel_tuning.is_none());
        assert_eq!(first.trials[0].qualifications, 1);
        let tuning = first.trials[1].kernel_tuning.as_ref().unwrap();
        assert!(!tuning.outcomes.is_empty() && !tuning.time_budget_exhausted);
        assert_eq!(first.trials[1].qualifications, 2);
        assert_eq!(first.trials[1].outcome.decision, TuneDecision::KeepBaseline);
        assert_eq!(first.trials[1].outcome.baseline_ms.len(), 4);
        assert_eq!(first.trials[1].outcome.candidate_ms.len(), 4);
        assert!(!first.final_qualification_time.is_zero());
        assert_eq!(memo.lock().unwrap().len(), 1);
        let second = search();
        assert_eq!(second.selected, 0);
        assert!(second.trials[0].kernel_tuning.is_none());
        let tuning = second.trials[1].kernel_tuning.as_ref().unwrap();
        assert!(tuning.outcomes.is_empty());
        assert_eq!(tuning.reused_classes.len(), 1);
        assert_eq!(second.trials[1].qualifications, 1);
        assert_eq!(
            second.trials[1].outcome.decision,
            TuneDecision::KeepBaseline
        );
        assert_eq!(second.trials[1].outcome.baseline_ms.len(), 4);
        assert_eq!(second.trials[1].outcome.candidate_ms.len(), 4);
    }

    #[test]
    fn untuned_fallback_survives_bounds_and_failed_tuning() {
        let gpu = Arc::new(crate::init_gpu_context_with(crate::GpuOptions::from_env()).unwrap());
        let mut graph = crate::Graph::new();
        let x = graph.input("x", &[33, 17]);
        let w = graph.parameter("w", &[17, 65]);
        let y = graph.matmul(x, w);
        graph.set_outputs(vec![y]);
        let plan = crate::compile::compile(&graph);
        let runtime = SessionOptions {
            coop: crate::CoopPolicy::Disabled,
            ..Default::default()
        };
        // Keep cold shader compilation outside the deadline test.
        let mut warm = Session::with_context_opts(plan.clone(), gpu.clone(), runtime.clone());
        warm.step();
        warm.wait();
        drop(warm);
        for case in ["program", "memory", "qualification", "deadline"] {
            let initialized = std::cell::Cell::new(0);
            let mut checks = 0;
            let mut untuned = Vec::new();
            let (mut session, report) = select(
                [Program {
                    description: "ordinary".into(),
                    plan: plan.clone(),
                    submission_chunks: Some(1),
                    origin: None,
                }],
                gpu.clone(),
                runtime.clone(),
                BuildSearchOptions {
                    max_programs: if case == "program" { 1 } else { 2 },
                    max_plan_bytes: if case == "memory" {
                        plan_bytes(&plan).unwrap()
                    } else {
                        1 << 20
                    },
                    max_time: Duration::from_secs(if case == "deadline" { 2 } else { 30 }),
                    warmup_runs: 1,
                    warmup_time: Duration::ZERO,
                    tuning: TuneOptions {
                        max_time: Duration::from_secs(1),
                        sample_pairs: 4,
                        target_sample_time: Some(Duration::from_nanos(1)),
                        ..Default::default()
                    },
                    ..Default::default()
                },
                |session, incumbent| {
                    initialized.set(initialized.get() + 1);
                    if incumbent.is_none() {
                        untuned = session.plan().dispatches.clone();
                    } else if case == "deadline" {
                        std::thread::sleep(Duration::from_secs(2));
                    }
                    session.set_input("x", &[0.25; 33 * 17]);
                    session.set_parameter("w", &[0.125; 17 * 65]);
                    Ok(())
                },
                |session| {
                    checks += 1;
                    assert!(
                        session
                            .read_output(33 * 65)
                            .iter()
                            .all(|&x| (x - 0.53125).abs() < 1e-5)
                    );
                    // The baseline and pre-tuning check pass; reject the tuned
                    // candidate, then requalify the preserved baseline.
                    if case == "qualification" && checks == 3 {
                        Err("injected post-tuning qualification failure".into())
                    } else {
                        Ok(())
                    }
                },
            )
            .unwrap();
            assert_eq!(report.selected, 0, "{case}");
            assert!(report.trials[0].outcome.qualified);
            assert!(report.trials[0].kernel_tuning.is_none());
            assert_eq!(session.plan().dispatches, untuned, "{case}");
            match case {
                "program" => {
                    assert_eq!(report.trials.len(), 1);
                    assert!(report.truncated);
                }
                "memory" => {
                    assert_eq!(initialized.get(), 1);
                    assert_eq!(
                        report.trials[1].outcome.decision,
                        TuneDecision::ScratchLimit
                    );
                }
                "qualification" => {
                    assert_eq!(report.trials[1].qualifications, 2);
                    assert_eq!(
                        report.trials[1].outcome.decision,
                        TuneDecision::InvalidOutput
                    );
                    assert!(!report.final_qualification_time.is_zero());
                }
                "deadline" => {
                    assert_eq!(report.trials[1].outcome.decision, TuneDecision::TimeBudget);
                    assert!(report.trials[1].outcome.baseline_ms.is_empty());
                    assert!(report.truncated);
                }
                _ => unreachable!(),
            }
            session.step();
            session.wait();
            assert!(
                session
                    .read_output(33 * 65)
                    .iter()
                    .all(|&x| (x - 0.53125).abs() < 1e-5)
            );
        }
    }

    /// Identical programs never replace the first, so only the stop rules end
    /// these searches.
    fn identical_programs(count: usize) -> impl Iterator<Item = Program> {
        let mut graph = crate::Graph::new();
        let x = graph.input("x", &[4]);
        let y = graph.neg(x);
        graph.set_outputs(vec![y]);
        let plan = crate::compile::compile(&graph);
        (0..count).map(move |index| Program {
            description: index.to_string(),
            plan: plan.clone(),
            submission_chunks: Some(1),
            origin: None,
        })
    }

    #[test]
    fn refreshes_inputs_for_every_execution_and_rejects_failed_refreshes() {
        let gpu = Arc::new(crate::init_gpu_context_with(crate::GpuOptions::from_env()).unwrap());
        for fail_challenger in [false, true] {
            let mut refreshed = 0;
            let mut checked = 0;
            let programs = identical_programs(2).map(|mut program| {
                program.plan.dispatches[0].label = program.description.clone();
                program
            });
            let options = BuildSearchOptions {
                tuning: TuneOptions {
                    sample_pairs: 4,
                    ..Default::default()
                },
                max_programs: 2,
                max_time: Duration::from_secs(30),
                min_improvement: 0.99,
                warmup_runs: 2,
                warmup_time: Duration::ZERO,
                ..Default::default()
            };
            let (mut session, report) = super::select(
                programs.fuse(),
                gpu.clone(),
                SessionOptions::default(),
                BuildSearchReport {
                    options,
                    ..Default::default()
                },
                Instant::now(),
                (
                    |session: &mut Session, _: Option<&mut Session>| {
                        // Qualification cannot pass without the refresh callback.
                        session.set_input("x", &[99.0; 4]);
                        Ok(())
                    },
                    |session: &mut Session| {
                        refreshed += 1;
                        if fail_challenger && session.plan().dispatches[0].label == "1" {
                            return Err("injected input refresh failure".into());
                        }
                        session.set_input("x", &[1.0, 2.0, 3.0, 4.0]);
                        Ok(())
                    },
                    |session: &Session| {
                        checked += 1;
                        assert_eq!(session.read_output(4), [-1.0, -2.0, -3.0, -4.0]);
                        Ok(())
                    },
                ),
            )
            .unwrap();
            assert_eq!(report.selected, 0);
            assert!(
                report
                    .trials
                    .iter()
                    .all(|t| !t.input_refresh_time.is_zero())
            );
            if fail_challenger {
                assert_eq!(refreshed, 3);
                assert_eq!(checked, 2);
                assert_eq!(
                    report.trials[1].outcome.decision,
                    TuneDecision::InvalidOutput
                );
                assert!(
                    report.trials[1]
                        .outcome
                        .failure
                        .as_ref()
                        .unwrap()
                        .contains("input refresh")
                );
            } else {
                assert_eq!(refreshed, checked + 2 * 2 + 2 * 4);
                assert_eq!(report.trials[1].outcome.baseline_ms.len(), 4);
                assert_eq!(report.trials[1].outcome.candidate_ms.len(), 4);
            }
            session.step();
            session.wait();
            assert_eq!(session.read_output(4), [-1.0, -2.0, -3.0, -4.0]);
        }
    }

    fn stop_rule_search(
        options: BuildSearchOptions,
        delay: Duration,
    ) -> (Session, BuildSearchReport) {
        let gpu = Arc::new(crate::init_gpu_context_with(crate::GpuOptions::from_env()).unwrap());
        // Compile and run the kernels before the search clock starts, so a cold
        // pipeline cache does not spend the deadline under test.
        let mut warm = Session::with_context_opts(
            identical_programs(1).next().unwrap().plan,
            gpu.clone(),
            SessionOptions::default(),
        );
        warm.step();
        warm.wait();
        drop(warm);
        select(
            identical_programs(8),
            gpu,
            SessionOptions::default(),
            BuildSearchOptions {
                tuning: TuneOptions {
                    sample_pairs: 4,
                    ..Default::default()
                },
                min_improvement: 0.99,
                warmup_runs: 1,
                warmup_time: Duration::from_millis(1),
                max_programs: 8,
                max_plan_bytes: 1 << 20,
                ..options
            },
            |s, _| {
                std::thread::sleep(delay);
                s.set_input("x", &[1.0, 2.0, 3.0, 4.0]);
                Ok(())
            },
            |s| {
                if s.read_output(4) == [-1.0, -2.0, -3.0, -4.0] {
                    Ok(())
                } else {
                    Err("negation mismatch".into())
                }
            },
        )
        .unwrap()
    }

    #[test]
    fn patience_stops_after_trials_without_a_new_incumbent() {
        let (_, report) = stop_rule_search(
            BuildSearchOptions {
                max_time: Duration::from_secs(60),
                patience: Some(2),
                ..Default::default()
            },
            Duration::ZERO,
        );
        assert_eq!(report.selected, 0);
        assert_eq!(report.trials.len(), 3);
        assert!(report.patience_exhausted && report.truncated);
        assert!(!report.deadline_reserved);
        // Challengers ran beside the incumbent after its own check.
        assert!(!report.final_qualification_time.is_zero());
    }

    #[test]
    fn no_comparison_starts_that_cannot_finish() {
        // A comparison costs at least its 0.4 s initialization, so after two
        // programs a third could not finish by 1.15 s. The remaining 0.35 s
        // absorbs execution with warm kernels.
        let (_, report) = stop_rule_search(
            BuildSearchOptions {
                max_time: Duration::from_millis(1150),
                ..Default::default()
            },
            Duration::from_millis(400),
        );
        assert!(report.deadline_reserved && report.truncated);
        assert_eq!(report.trials.len(), 2);
        assert_eq!(
            report.trials[1].outcome.decision,
            TuneDecision::KeepBaseline
        );
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
                    submission_chunks: Some(splits as usize),
                    origin: None,
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
                    warmup_time: Duration::from_millis(1),
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
