//! Experimental whole-program selection for immutable inference.
use crate::{
    Session,
    compile::ExecutionPlan,
    runtime::SessionOptions,
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

pub struct Options {
    /// Per-program kernel tuning and paired whole-step decision policy.
    pub tuning: TuneOptions,
    /// Soft total deadline, including construction, initialization and validation.
    /// In-flight driver work and caller validation cannot be preempted.
    pub max_time: Duration,
    pub max_programs: usize,
    /// Sum of declared logical bytes for incumbent and challenger. This is not
    /// a driver-heap bound: padding, pipelines, staging and tuning are additional.
    pub max_plan_bytes: usize,
}

#[derive(Serialize)]
pub struct Trial {
    pub description: String,
    /// Full session construction, including allocations, not shader-only compile time.
    pub construction_time: Duration,
    pub initialization_time: Duration,
    pub qualification_time: Duration,
    pub kernel_tuning: Option<TuneReport>,
    pub outcome: TuneOutcome<(), usize>,
}

#[derive(Serialize)]
pub struct Report {
    pub selected: usize,
    pub trials: Vec<Trial>,
    pub elapsed: Duration,
    pub truncated: bool,
}

fn plan_bytes(plan: &ExecutionPlan) -> Result<usize, String> {
    if !plan.param_grad_pairs.is_empty() {
        return Err("program search currently accepts immutable inference only".into());
    }
    let protected: std::collections::HashSet<_> = plan
        .input_buffers
        .iter()
        .chain(&plan.param_buffers)
        .map(|&(_, buffer)| buffer)
        .chain(plan.constant_buffers.iter().map(|&(buffer, _)| buffer))
        .collect();
    for dispatch in &plan.dispatches {
        if std::iter::once(&dispatch.output_buffer)
            .chain(&dispatch.extra_outputs)
            .any(|buffer| protected.contains(buffer))
        {
            return Err("program search cannot mutate inputs, parameters or constants".into());
        }
    }
    plan.buffers
        .iter()
        .try_fold(0usize, |sum, bytes| sum.checked_add((*bytes).max(4)))
        .ok_or_else(|| "declared program bytes overflow".into())
}

/// Search complete, legal implementations of the same immutable inference graph.
/// Each has private buffers and is kernel-tuned *before* comparing graph forms.
/// `initialize` writes representative inputs/weights once into each private
/// session. `qualify` executes and checks all observable outputs against the
/// caller's numerical contract before tuning, after tuning and after measurements.
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
    mut initialize: impl FnMut(&mut Session) -> Result<(), String>,
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
        selected: 0,
        trials: Vec::new(),
        elapsed: Duration::ZERO,
        truncated: false,
    };
    let mut incumbent: Option<Session> = None;
    let mut incumbent_bytes = 0usize;
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
                let initialized = initialize(&mut candidate);
                trial.initialization_time = init.elapsed();
                initialized?;
                let mut validate = |session: &mut Session| {
                    let began = Instant::now();
                    let result = qualify(session);
                    trial.qualification_time += began.elapsed();
                    result
                };
                validate(&mut candidate)?;
                let mut policy = options.tuning.clone();
                policy.max_time = policy
                    .max_time
                    .min(options.max_time.saturating_sub(start.elapsed()));
                trial.kernel_tuning = Some(
                    candidate
                        .tune_with(policy)
                        .map_err(|error| error.to_string())?,
                );
                validate(&mut candidate)?;
                trial.outcome.qualified = true;
                if let Some(ref mut baseline) = incumbent {
                    for _ in 0..options.tuning.warmup_runs {
                        if start.elapsed() >= options.max_time {
                            break;
                        }
                        baseline.step();
                        baseline.wait();
                        candidate.step();
                        candidate.wait();
                    }
                    (trial.outcome.baseline_ms, trial.outcome.candidate_ms) =
                        measure_pairs(options.tuning.sample_pairs, |alternative| {
                            if start.elapsed() >= options.max_time {
                                return None;
                            }
                            let session = if alternative {
                                &mut candidate
                            } else {
                                &mut *baseline
                            };
                            let sample = Instant::now();
                            session.step();
                            session.wait();
                            Some(sample.elapsed().as_secs_f64() * 1000.0)
                        });
                    validate(&mut candidate)?;
                    decide(&mut trial.outcome, &options.tuning);
                } else {
                    trial.outcome.selected = index;
                }
                Ok::<(), String>(())
            })();
            match result {
                Ok(()) if incumbent.is_none() || trial.outcome.selected == index => {
                    incumbent_bytes = plan_bytes(candidate.plan())?;
                    incumbent = Some(candidate);
                    report.selected = index;
                }
                Ok(()) => {}
                Err(error) => {
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
        .map(|session| (session, report))
        .ok_or_else(|| "no qualified program within the search bounds".into())
}

#[cfg(test)]
mod tests {
    #[test]
    fn rejects_mutable_plans_before_allocation() {
        use crate::compile::{BufferRef, Dispatch};
        let mut graph = crate::Graph::new();
        let input = graph.input("x", &[2]);
        let output = graph.neg(input);
        graph.set_outputs(vec![output]);
        let mut plan = crate::compile::compile(&graph);
        assert!(super::plan_bytes(&plan).is_ok());
        plan.param_grad_pairs.push((BufferRef(0), BufferRef(1)));
        assert!(super::plan_bytes(&plan).is_err());
        plan.param_grad_pairs.clear();
        plan.dispatches.push(Dispatch {
            output_buffer: plan.input_buffers[0].1,
            ..Default::default()
        });
        assert!(super::plan_bytes(&plan).is_err());
    }
}
