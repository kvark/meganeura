//! Measured construction: graph alternatives survive until their kernels are tuned.
use super::{SessionConfig, prepare_graph};
use crate::{
    Graph, Session, TuneOptions,
    compile::{self, ExecutionPlan},
    optimize, runtime,
};
use serde::Serialize;

mod measure;
pub use measure::BuildSearchTrial;
use std::{
    collections,
    time::{Duration, Instant},
};

const INITIAL_LAYOUTS: usize = 3;

#[derive(Clone, Serialize)]
pub struct BuildSearchOptions {
    /// Bound on graph/implementation forms, including ordinary extraction.
    pub max_graphs: usize,
    /// Per-program kernel tuning and shared paired sampling policy.
    pub tuning: TuneOptions,
    /// Minimum whole-program improvement, in addition to the timing noise margin.
    /// Independent of the private kernel threshold in `tuning`.
    pub min_improvement: f64,
    /// Whole-program warmup pairs, independent of private kernel warmup.
    pub warmup_runs: u32,
    /// Minimum paired warmup duration. A fixed step count alone can leave
    /// short workloads in a different operating state from sustained execution.
    pub warmup_time: Duration,
    /// Soft total deadline, including construction, initialization and validation.
    /// In-flight driver work and caller validation cannot be preempted. An
    /// incomplete comparison cannot select its program, so the search does not
    /// start one when even its cheapest recent complete comparison would not
    /// finish in time.
    pub max_time: Duration,
    pub max_programs: usize,
    /// Stop once this many consecutive trials have not replaced the incumbent.
    /// `None` searches until another bound ends it.
    pub patience: Option<usize>,
    /// Explore power-of-two submission chunk counts up to this bound (1..=64).
    /// One keeps every candidate on a single submission.
    pub max_submission_chunks: usize,
    /// Sum of declared logical bytes and persistent-state snapshots for both
    /// incumbent and challenger. This is not a driver-heap bound: padding,
    /// pipelines, staging and kernel-probe scratch are additional.
    pub max_plan_bytes: usize,
    /// Private kernel decisions shared with other searches, which resume them
    /// on the same device, driver and decision policy. `None` keeps them
    /// private to this search. The search holds the lock while it runs.
    #[serde(skip)]
    pub kernel_memo: Option<std::sync::Arc<std::sync::Mutex<runtime::KernelMemo>>>,
}

impl Default for BuildSearchOptions {
    fn default() -> Self {
        Self {
            max_graphs: 16,
            tuning: TuneOptions::default(),
            min_improvement: 0.05,
            warmup_runs: 2,
            warmup_time: Duration::from_millis(250),
            max_time: Duration::from_secs(30),
            max_programs: 64,
            patience: None,
            max_submission_chunks: 1,
            max_plan_bytes: 512 << 20,
            kernel_memo: None,
        }
    }
}

#[derive(Default, Serialize)]
pub struct BuildSearchReport {
    pub options: BuildSearchOptions,
    /// Extracted forms, indexed by the `graph=N` trial descriptions.
    pub graphs: Vec<String>,
    pub extraction_truncated: bool,
    pub skipped_regions: Vec<String>,
    pub preparation_time: Duration,
    pub selected: usize,
    pub trials: Vec<BuildSearchTrial>,
    pub truncated: bool,
    /// Stopped by [`BuildSearchOptions::patience`].
    pub patience_exhausted: bool,
    /// Stopped early because no further comparison could finish by the deadline.
    pub deadline_reserved: bool,
    /// Requalification of the selected program after the last trial; zero
    /// when nothing ran after its own check.
    pub final_qualification_time: Duration,
    pub elapsed: Duration,
}

/// Build from representative inputs, measuring logical and physical alternatives.
///
/// Unlike [`super::build`], this executes private candidate sessions.
/// `initialize` writes representative inputs and weights once per candidate,
/// the same values every time. A challenger is constructed on its idle
/// incumbent's identically stored parameters that neither program writes;
/// [`Session::inherits_parameter`] reports them so the initializer can skip
/// their upload. The incumbent must not be modified. Inputs and writable state
/// remain private. Configure runtime optimizers, accumulation and external
/// bindings after construction, not inside either callback.
///
/// The runner executes one step before each read-only `qualify` call. Check every
/// observable output, gradient and persistent update against your numerical
/// contract. Every program is qualified after kernel tuning, before it is timed;
/// one with kernel classes left to probe is also qualified before tuning. A
/// winning challenger is qualified again after measurement, and the selected
/// program once more before construction returns. A failing challenger is
/// discarded; a failing final incumbent aborts construction.
/// Written inputs, parameters and constants are reset before each step, outside
/// timing. The returned session retains its initialized persistent state; outputs
/// hold its last qualified step. Timing includes fresh recording/submission/wait,
/// not readback. These search samples are not held-out benchmark results.
///
/// The ordinary optimizer supplies the first candidate. Egglog retains alternatives
/// from the original forward graph (one bounded region, repeated where verified).
/// They are not greedily optimized again. Each training form is differentiated
/// separately, so parameter transformations and gradients stay consistent.
/// Matrix tile, split-K and independent attention-gradient layouts are egglog
/// equalities. Lowering emits the extracted schedule. Complete plans vary
/// dispatch fusion, cached-attention splits, low-occupancy convolution
/// weight-gradient splits, and caller-enabled submission chunk counts. Unlocked
/// dispatches are kernel-tuned before comparison. This bounded search does not
/// promise a global optimum.
/// After the first layouts, alternate layout exploration with submission probes
/// on the incumbent. A winning layout reopens those probes; later layouts inherit
/// its measured count rather than expanding a layout/submission cross product.
///
/// `options.tuning` replaces `cfg.tune`. Build-plan caching is not yet supported:
/// calibration data and measured policy are not part of the ordinary cache key.
pub fn build_measured(
    forward_graph: &Graph,
    mut cfg: SessionConfig<'_>,
    options: BuildSearchOptions,
    initialize: impl FnMut(&mut Session, Option<&mut Session>) -> Result<(), String>,
    qualify: impl FnMut(&Session) -> Result<(), String>,
) -> Result<(Session, BuildSearchReport), String> {
    let start = Instant::now();
    let _span = tracing::info_span!("build_measured").entered();
    if cfg.cache.is_some() {
        return Err("measured construction does not use the ordinary build-plan cache".into());
    }
    if options.max_graphs == 0 || options.max_programs == 0 || options.max_time.is_zero() {
        return Err("measured construction needs positive graph, program and time bounds".into());
    }
    if !(1..=64).contains(&options.max_submission_chunks) {
        return Err("measured construction needs a submission chunk bound in 1..=64".into());
    }
    if !options.min_improvement.is_finite() || !(0.0..1.0).contains(&options.min_improvement) {
        return Err("whole-program min_improvement must be finite and in [0, 1)".into());
    }
    options.tuning.validate().map_err(|e| e.to_string())?;
    if cfg.runtime.debug {
        cfg.options.fuse_dispatches = false;
    }
    let gpu = cfg.gpu.take().unwrap_or_else(runtime::default_gpu_context);
    let caps = cfg
        .runtime
        .coop
        .filter_caps(runtime::auto_tune(&gpu, 0).coop_caps);
    let shared_memory_bytes = gpu.capabilities().max_compute_shared_memory_size;
    let target = optimize::search::Target {
        caps,
        shared_memory_bytes,
        forward_coop: cfg.options.flash_forward_coop,
    };
    let (ordinary, _) = prepare_graph(
        forward_graph,
        cfg.mode,
        cfg.optimize,
        cfg.skip_full_optimize,
    );
    let mut graphs = vec![optimize::search::Candidate {
        graph: ordinary,
        expression: "ordinary extraction".into(),
    }];
    let mut extraction_truncated = false;
    let mut skipped_regions = Vec::new();
    if cfg.optimize.mode != optimize::OptimizeMode::Off
        && options.max_graphs > 1
        && start.elapsed() < options.max_time
    {
        let source = super::recognize(forward_graph, cfg.mode, &cfg.optimize).into_toposort();
        let limit = options.max_graphs - 1;
        let space = optimize::search::candidates(&source, cfg.optimize, limit, target);
        let spaces = if space.is_ok()
            || source.nodes().len()
                <= cfg
                    .optimize
                    .saturation_cutoff
                    .min(optimize::SATURATION_CUTOFF)
        {
            vec![space]
        } else {
            // Reuse outlining, not model names or a second pattern matcher.
            let regions = crate::outline::detect_repeated_regions(&source);
            extraction_truncated |= regions.len() > 1;
            if regions.is_empty() {
                skipped_regions.push("large graph has no bounded repeated region".into());
            }
            regions
                .into_iter()
                .take(1)
                .map(|region| {
                    optimize::search::repeated_candidates(
                        &source,
                        region,
                        cfg.optimize,
                        limit,
                        target,
                    )
                })
                .collect()
        };
        for space in spaces {
            match space {
                Ok(space) => {
                    extraction_truncated |= space.truncated;
                    let unfused = optimize::OptimizeConfig {
                        mode: optimize::OptimizeMode::Off,
                        ..cfg.optimize
                    };
                    for form in space.candidates {
                        if start.elapsed() >= options.max_time {
                            extraction_truncated = true;
                            break;
                        }
                        graphs.push(optimize::search::Candidate {
                            graph: prepare_graph(
                                &form.graph,
                                cfg.mode,
                                unfused,
                                cfg.skip_full_optimize,
                            )
                            .0,
                            expression: form.expression,
                        });
                    }
                }
                Err(error) => skipped_regions.push(error),
            }
        }
    }
    if cfg.mode == crate::Mode::Training
        && cfg.optimize.mode != optimize::OptimizeMode::Off
        && options.max_graphs > 1
    {
        let space = backward_layouts(
            graphs,
            cfg.optimize,
            target,
            options.max_graphs,
            start + options.max_time,
            &mut skipped_regions,
        );
        graphs = space.candidates;
        extraction_truncated |= space.truncated;
    }
    let expressions = graphs.iter().map(|form| form.expression.clone()).collect();
    // Compile graph forms only once. Physical alternatives remain lazy, and are
    // interleaved across forms rather than spending the budget on the first form.
    let mut seeds: Vec<Seed> = Vec::new();
    for (index, form) in graphs.into_iter().enumerate() {
        if start.elapsed() >= options.max_time {
            extraction_truncated = true;
            break;
        }
        let graph = std::sync::Arc::new(form.graph);
        let plan = compile::compile_with_caps(&graph, &cfg.options, caps, shared_memory_bytes);
        if !seeds.iter().any(|seed| seed.plan == plan) {
            seeds.push(Seed {
                description: format!("graph={index}"),
                plan,
                graph,
                options: cfg.options.clone(),
            });
        }
    }
    let preparation_time = start.elapsed();
    let programs = implementations(seeds, caps, shared_memory_bytes, options.max_plan_bytes);
    measure::select(
        programs,
        gpu,
        cfg.runtime,
        BuildSearchReport {
            options,
            graphs: expressions,
            extraction_truncated,
            skipped_regions,
            preparation_time,
            ..Default::default()
        },
        start,
        initialize,
        qualify,
    )
}

/// Cover one backward layout and one joint graph/layout choice early, then
/// interleave both axes so neither displaces the other under a deadline.
fn backward_layouts(
    graphs: Vec<optimize::search::Candidate>,
    config: optimize::OptimizeConfig,
    target: optimize::search::Target,
    limit: usize,
    deadline: Instant,
    skipped: &mut Vec<String>,
) -> optimize::search::SearchSpace {
    let mut layouts: Vec<Option<collections::VecDeque<optimize::search::Candidate>>> =
        (0..graphs.len()).map(|_| None).collect();
    let mut candidates = Vec::new();
    let mut truncated = false;
    let prefix = [(0, 0), (0, 1), (1, 1), (1, 0)];
    let diagonal = (1..limit)
        .flat_map(|rank| (0..graphs.len().min(rank + 1)).map(move |index| (index, rank - index)));
    let mut visited = collections::HashSet::new();
    for (index, layout) in prefix.into_iter().chain(diagonal) {
        if index >= graphs.len() || !visited.insert((index, layout)) {
            continue;
        }
        if candidates.len() == limit || Instant::now() >= deadline {
            return optimize::search::SearchSpace {
                candidates,
                truncated: true,
            };
        }
        let graph = &graphs[index];
        if layout == 0 {
            candidates.push(optimize::search::Candidate {
                graph: graph.graph.deep_clone(),
                expression: graph.expression.clone(),
            });
            continue;
        }
        let pending = layouts[index].get_or_insert_with(|| {
            match optimize::search::backward_candidates(&graph.graph, config, limit - 1, target) {
                Ok(space) => {
                    truncated |= space.truncated;
                    space.candidates.into()
                }
                Err(error) => {
                    skipped.push(format!("backward layouts for form {index}: {error}"));
                    collections::VecDeque::new()
                }
            }
        });
        if let Some(form) = pending.pop_front() {
            candidates.push(optimize::search::Candidate {
                graph: form.graph,
                expression: format!("{}; backward={}", graph.expression, form.expression),
            });
        }
    }
    optimize::search::SearchSpace {
        candidates,
        truncated: truncated || layouts.iter().flatten().any(|pending| !pending.is_empty()),
    }
}

struct Seed {
    graph: std::sync::Arc<Graph>,
    options: compile::CompileOptions,
    plan: ExecutionPlan,
    description: String,
}

type AttentionChoice = (u32, Option<(u32, crate::codegen::FlashAttentionShape)>);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct AxisChoice {
    attention: AttentionChoice,
    fuse_dispatches: bool,
}

fn early_physical_cover(baseline: AxisChoice, attention: &[AttentionChoice]) -> Vec<AxisChoice> {
    let mut cover = Vec::new();
    if baseline.fuse_dispatches {
        cover.push(AxisChoice {
            fuse_dispatches: false,
            ..baseline
        });
    }
    if let Some(&choice) = attention.get(1) {
        cover.push(AxisChoice {
            attention: choice,
            ..baseline
        });
    }
    cover
}

/// Cover early layouts, then graph rank × physical rank. Neither a larger graph
/// frontier nor more physical variants should starve the other under a budget.
fn physical_program_order(
    n_seeds: usize,
    baseline: AxisChoice,
    cover: &[AxisChoice],
    tail: &[AxisChoice],
) -> Vec<(usize, AxisChoice)> {
    let mut order: Vec<_> = (0..n_seeds.min(INITIAL_LAYOUTS))
        .map(|seed| (seed, baseline))
        .collect();
    let choices: Vec<_> = std::iter::once(baseline)
        .chain(cover.iter().copied())
        .chain(tail.iter().copied())
        .collect();
    for rank in 0..n_seeds + choices.len() - 1 {
        for (physical, &choice) in choices.iter().take(rank + 1).enumerate() {
            let seed = rank - physical;
            if seed < n_seeds && !(physical == 0 && seed < INITIAL_LAYOUTS) {
                order.push((seed, choice));
            }
        }
    }
    order
}

fn implementations(
    seeds: Vec<Seed>,
    caps: crate::codegen::CoopCaps,
    shared_memory_bytes: u32,
    max_partial_bytes: usize,
) -> impl Iterator<Item = measure::Program> {
    let cached_attention = seeds.iter().any(|p| {
        p.graph
            .nodes()
            .iter()
            .any(|n| matches!(n.op, crate::graph::Op::CachedBlockAttention { .. }))
    });
    let heads: Vec<_> = seeds
        .iter()
        .flat_map(|seed| &seed.plan.dispatches)
        .filter(|d| d.shader.is_attention())
        .map(|d| d.params[3])
        .collect();
    let mut attention = vec![(0, None)];
    if !heads.is_empty() {
        let mut layouts = Vec::new();
        for (e, ept) in [32, 16, 8].into_iter().enumerate() {
            for (t, threads) in [256, 128].into_iter().enumerate() {
                for (k, keys) in [8, 16, 4].into_iter().enumerate() {
                    for (l, interleave) in [false, true].into_iter().enumerate() {
                        let shape = crate::codegen::FlashAttentionShape {
                            threads,
                            keys,
                            interleave,
                        };
                        if heads
                            .iter()
                            .all(|&hd| shape.shared_bytes(hd) <= u64::from(shared_memory_bytes))
                        {
                            layouts.push((e + t + k + l, (0, Some((ept, shape)))));
                        }
                    }
                }
            }
        }
        layouts.sort_by_key(|&(rank, _)| rank);
        attention.extend(layouts.into_iter().map(|(_, layout)| layout));
    }
    if cached_attention {
        attention.extend([1, 2, 4, 8, 16].map(|splits| (splits, None)));
    }
    let baseline = AxisChoice {
        attention: (0, None),
        fuse_dispatches: seeds
            .first()
            .is_some_and(|seed| seed.options.fuse_dispatches),
    };
    // Rank the first alternative on each axis and their interaction early.
    let cover = early_physical_cover(baseline, &attention);
    let mut seen = collections::HashSet::from([baseline]);
    seen.extend(cover.iter().copied());
    let fusion: &[bool] = if baseline.fuse_dispatches {
        &[true, false]
    } else {
        &[false]
    };
    let tail: Vec<_> = attention
        .iter()
        .flat_map(|&attention| {
            fusion.iter().map(move |&fuse_dispatches| AxisChoice {
                attention,
                fuse_dispatches,
            })
        })
        .filter(|&choice| seen.insert(choice))
        .collect();
    let order = physical_program_order(seeds.len(), baseline, &cover, &tail);
    let mut cursor = 0;
    let mut split_plans = Vec::new();
    let mut queued_splits = false;
    std::iter::from_fn(move || {
        loop {
            if let Some(plan) = split_plans.pop() {
                return Some(plan);
            }
            if !queued_splits && cursor > 0 {
                queued_splits = true;
                // After the ordinary incumbent. Popped in reverse, so 4 splits
                // is measured before 8.
                split_plans = low_occupancy_weight_splits(seeds.first(), max_partial_bytes);
                split_plans.reverse();
                if let Some(plan) = split_plans.pop() {
                    return Some(plan);
                }
            }
            if cursor == order.len() {
                return None;
            }
            let (seed_index, current) = order[cursor];
            cursor += 1;
            let seed = seeds.get(seed_index)?;
            let AxisChoice {
                attention: (splits, flash),
                fuse_dispatches,
            } = current;
            let plan = if splits == 0
                && flash.is_none()
                && fuse_dispatches == seed.options.fuse_dispatches
            {
                seed.plan.clone()
            } else {
                let mut options = seed.options.clone();
                options.fuse_dispatches = fuse_dispatches;
                if splits != 0 {
                    options.cached_attention_splits = Some(splits);
                }
                if let Some((ept, shape)) = flash {
                    options.knobs.flash_ept_cap = ept;
                    options.knobs.flash = shape;
                    options.flash_forward_coop = false;
                }
                let plan =
                    compile::compile_with_caps(&seed.graph, &options, caps, shared_memory_bytes);
                if plan == seed.plan
                    || (flash.is_some()
                        && !plan
                            .dispatches
                            .iter()
                            .any(|d| d.shader == compile::ShaderEntry::FlashAttention))
                {
                    continue;
                }
                plan
            };
            return Some(measure::Program {
                description: format!(
                    "{}, dispatch_fusion={fuse_dispatches}, attention_splits={splits}, flash={flash:?}",
                    seed.description
                ),
                plan,
                submission_chunks: None,
            });
        }
    })
}

/// Weight-gradient split-K for convolutions that launch only a few dozen
/// workgroups. Measured whole-step, not installed unless it wins.
fn low_occupancy_weight_splits(
    seed: Option<&Seed>,
    max_partial_bytes: usize,
) -> Vec<measure::Program> {
    let Some(seed) = seed else {
        return Vec::new();
    };
    let mut programs = Vec::new();
    for splits in [4u32, 8] {
        let mut plan = seed.plan.clone();
        let selections: Vec<(usize, u32)> = plan
            .dispatches
            .iter()
            .enumerate()
            .filter_map(|(index, dispatch)| {
                let weight = matches!(
                    dispatch.shader,
                    compile::ShaderEntry::Conv2dGradWeightGemm
                        | compile::ShaderEntry::Conv2dGradWeightGemmSmall
                        | compile::ShaderEntry::Conv2dGradWeightGemm16
                );
                let groups = dispatch.workgroups[0].saturating_mul(dispatch.workgroups[1]);
                let k = dispatch
                    .params
                    .first()
                    .copied()
                    .unwrap_or(0)
                    .saturating_mul(dispatch.params.get(9).copied().unwrap_or(0))
                    .saturating_mul(dispatch.params.get(10).copied().unwrap_or(0));
                (weight
                    && (1..48).contains(&groups)
                    && matches!(dispatch.conv_k_tile(), None | Some(16))
                    && splits <= k.div_ceil(16))
                .then_some((index, splits))
            })
            .collect();
        if selections.is_empty()
            || plan
                .split_conv_weight_gradients(&selections, max_partial_bytes)
                .is_err()
        {
            continue;
        }
        programs.push(measure::Program {
            description: format!(
                "{}, dispatch_fusion={}, conv_dw_splits={splits}, workgroups<48",
                seed.description, seed.options.fuse_dispatches
            ),
            plan,
            submission_chunks: None,
        });
    }
    programs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fusion_alternatives_lead_the_physical_cover() {
        let baseline = AxisChoice {
            attention: (0, None),
            fuse_dispatches: true,
        };
        let unfused = AxisChoice {
            fuse_dispatches: false,
            ..baseline
        };
        let cover = early_physical_cover(baseline, &[]);
        assert_eq!(cover, vec![unfused]);
        let order = physical_program_order(8, baseline, &cover, &[]);
        assert!(order[..4].contains(&(0, unfused)));
        for seed in 0..8 {
            for choice in [baseline, unfused] {
                assert_eq!(
                    order
                        .iter()
                        .filter(|&&entry| entry == (seed, choice))
                        .count(),
                    1
                );
            }
        }
        assert!(early_physical_cover(unfused, &[]).is_empty());
    }

    #[test]
    fn attention_layouts_fit_alongside_submission_probes() {
        let mut graph = Graph::new();
        let q = graph.input("q", &[129, 64]);
        let k = graph.input("k", &[129, 64]);
        let v = graph.input("v", &[129, 64]);
        let y = graph.full_attention(q, k, v, 1, 1, 64);
        graph.set_outputs(vec![y]);
        let graph = std::sync::Arc::new(graph);
        let options = compile::CompileOptions::default();
        let caps = crate::codegen::CoopCaps::default();
        let plan = compile::compile_with_caps(&graph, &options, caps, 65536);
        let seeds = (0..16)
            .map(|index| Seed {
                graph: graph.clone(),
                options: options.clone(),
                plan: plan.clone(),
                description: format!("graph={index}"),
            })
            .collect();
        // Reserve half the ordinary 64-program allowance for submission probes.
        // Even with a full graph frontier, distinct attention layouts must fit.
        let programs: Vec<_> = implementations(seeds, caps, 65536, 512 << 20)
            .take(BuildSearchOptions::default().max_programs / 2)
            .collect();
        assert!(programs.iter().all(|p| p.submission_chunks.is_none()));
        assert!(programs.iter().any(|p| {
            p.description.starts_with("graph=0,")
                && p.plan.knobs.flash.interleave
                && p.plan
                    .dispatches
                    .iter()
                    .any(|d| d.shader == compile::ShaderEntry::FlashAttention)
        }));
    }

    #[test]
    fn measured_native_attention_qualifies_forward_and_backward_candidates() {
        use crate::compile::ShaderEntry;
        use crate::kernels::attention_grad::{AttentionGrad, Operands, Part, Path};
        use crate::{CoopPolicy, Mode, reference};
        let gpu = reference::gpu::shared_context();
        if !gpu
            .capabilities()
            .cooperative_matrix
            .f32_shapes
            .contains(&[16, 16, 16])
        {
            return;
        }
        let mut graph = Graph::new();
        let q = graph.parameter("q", &[129, 128]);
        let k = graph.parameter("k", &[129, 64]);
        let v = graph.parameter("v", &[129, 64]);
        let y = graph.sliding_window_attention(q, k, v, 2, 1, 64, 17);
        let loss = reference::gradients::weighted_loss(&mut graph, y, 11, 0.7);
        graph.set_outputs(vec![loss, y]);
        let full = crate::autodiff::differentiate(&graph);
        let mut feeds = reference::Feeds::new();
        feeds.fill_random(&graph, 20261009, 0.5);
        let values = reference::evaluate(&full, &feeds).unwrap();
        let scales = reference::error_scales(&full, &values).unwrap();
        let native_parts = |session: &crate::Session| {
            session.plan().dispatches.iter().fold(0u8, |mask, d| {
                mask | match d.shader {
                    ShaderEntry::FlashAttentionCoopF32 => 1,
                    ShaderEntry::AttentionGrad(AttentionGrad {
                        part,
                        path: Path::Cooperative(Operands::F32),
                    }) => match part {
                        Part::Q => 2,
                        Part::KV => 4,
                    },
                    _ => 0,
                }
            })
        };
        // Also exercise the rejection path: a caller's stricter contract may
        // reject an otherwise correct candidate, which must never be installed.
        for reject_native in [false, true] {
            let mut observed = std::collections::HashSet::new();
            let (mut session, report) = build_measured(
                &graph,
                SessionConfig {
                    mode: Mode::Training,
                    gpu: Some(gpu.clone()),
                    runtime: crate::SessionOptions {
                        coop: CoopPolicy::NativeF32,
                        poison: true,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                BuildSearchOptions {
                    max_graphs: 8,
                    max_programs: 12,
                    max_time: Duration::from_secs(30),
                    warmup_runs: 1,
                    warmup_time: Duration::ZERO,
                    tuning: TuneOptions {
                        max_classes: 0,
                        sample_pairs: 4,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                |session, _| {
                    for name in ["q", "k", "v"] {
                        session.set_parameter(name, &feeds.f32(name).unwrap());
                    }
                    Ok(())
                },
                |session| {
                    let mask = native_parts(session);
                    observed.insert(mask);
                    for (index, &id) in full.outputs().iter().enumerate() {
                        let want = &values[id as usize].data;
                        let mut got = vec![0.0; want.len()];
                        if index < full.num_user_outputs() {
                            session.read_output_by_index(index, &mut got);
                        } else {
                            session.read_param_grad(
                                ["q", "k", "v"][index - full.num_user_outputs()],
                                &mut got,
                            );
                        }
                        reference::check(&got, want, &scales[id as usize], Default::default())
                            .map_err(|error| format!("output {index}: {error}"))?;
                    }
                    if reject_native && mask != 0 {
                        Err("test contract rejects native candidates".into())
                    } else {
                        Ok(())
                    }
                },
            )
            .unwrap();
            assert!(
                observed.contains(&0) && observed.contains(&7),
                "{observed:?}"
            );
            assert!(
                report.skipped_regions.is_empty(),
                "{:?}",
                report.skipped_regions
            );
            if reject_native {
                assert_eq!(native_parts(&session), 0);
                assert!(report.trials.iter().any(|t| !t.outcome.qualified));
            } else {
                assert!(
                    report.trials.iter().all(|t| t.outcome.qualified),
                    "{}",
                    serde_json::to_string(&report).unwrap()
                );
            }
            // Timing is deliberately not an assertion. The selected session is
            // runnable and its ordinary incumbent was qualified just as strictly.
            session.step();
            session.wait();
            let mut got = vec![0.0; 129 * 128];
            session.read_output_by_index(1, &mut got);
            reference::check(
                &got,
                &values[y as usize].data,
                &scales[y as usize],
                Default::default(),
            )
            .unwrap();
        }
    }

    #[test]
    fn measured_attention_reaches_backward_layouts_with_a_small_budget() {
        use crate::{CoopPolicy, Mode, reference};
        let mut graph = Graph::new();
        let x = graph.input("x", &[33, 64]);
        let q = graph.parameter("q", &[64, 128]);
        let q = graph.matmul(x, q);
        let k = graph.parameter("k", &[65, 64]);
        let v = graph.parameter("v", &[65, 64]);
        let y = graph.multi_head_attn(q, k, v, 2, 1, 64, true);
        let loss = reference::gradients::weighted_loss(&mut graph, y, 11, 0.7);
        graph.set_outputs(vec![loss, y]);
        let full = crate::autodiff::differentiate(&graph);
        let mut feeds = reference::Feeds::new();
        feeds.fill_random(&graph, 20261006, 0.5);
        let values = reference::evaluate(&full, &feeds).unwrap();
        let scales = reference::error_scales(&full, &values).unwrap();
        let mut observed = std::collections::HashSet::new();
        let mut joint_schedule = false;
        let (_, report) = build_measured(
            &graph,
            SessionConfig {
                mode: Mode::Training,
                gpu: Some(reference::gpu::shared_context()),
                runtime: crate::SessionOptions {
                    coop: CoopPolicy::Disabled,
                    ..Default::default()
                },
                ..Default::default()
            },
            BuildSearchOptions {
                max_graphs: 6,
                max_programs: 6,
                max_time: Duration::from_secs(30),
                warmup_runs: 1,
                warmup_time: Duration::ZERO,
                tuning: TuneOptions {
                    max_classes: 0,
                    sample_pairs: 4,
                    ..Default::default()
                },
                ..Default::default()
            },
            |session, _| {
                session.set_input("x", &feeds.f32("x").unwrap());
                for name in ["q", "k", "v"] {
                    session.set_parameter(name, &feeds.f32(name).unwrap());
                }
                Ok(())
            },
            |session| {
                joint_schedule |= session.plan().dispatches.iter().any(|d| d.schedule_locked)
                    && session.plan().dispatches.iter().any(|d| {
                        matches!(d.kernel, compile::Kernel::AttentionBackward { ept_cap: 4 })
                    });
                observed.extend(
                    session
                        .dispatch_pipeline_keys()
                        .into_iter()
                        .filter(|key| key.contains("-flash)")),
                );
                for (index, &id) in full.outputs().iter().enumerate() {
                    let want = &values[id as usize].data;
                    let mut got = vec![0.0; want.len()];
                    if index < full.num_user_outputs() {
                        session.read_output_by_index(index, &mut got);
                    } else {
                        session.read_param_grad(
                            ["q", "k", "v"][index - full.num_user_outputs()],
                            &mut got,
                        );
                    }
                    reference::check(&got, want, &scales[id as usize], Default::default())
                        .map_err(|error| format!("output {index}: {error}"))?;
                }
                Ok(())
            },
        )
        .unwrap();
        assert!(report.graphs.len() <= 6 && report.trials.len() <= 6);
        assert!(
            report.skipped_regions.is_empty(),
            "{:?}",
            report.skipped_regions
        );
        assert!(report.trials.iter().all(|trial| trial.outcome.qualified));
        assert!(
            joint_schedule,
            "the bounded search lost the matrix/layout combination"
        );
        assert!(
            report
                .trials
                .iter()
                .any(|trial| trial.description.starts_with("graph=1,"))
        );
        for name in ["AttentionGrad(dQ-flash)", "AttentionGrad(dKV-flash)"] {
            assert!(
                observed
                    .iter()
                    .any(|key| key.starts_with(name) && key.ends_with("Some(4)")),
                "{observed:?}"
            );
        }
    }

    #[test]
    fn measured_build_keeps_graph_choices_and_training_state() {
        use crate::{Mode, TuneOptions};
        let gpu = std::sync::Arc::new(
            crate::init_gpu_context_with(crate::GpuOptions::from_env()).unwrap(),
        );
        let mut graph = Graph::new();
        let x = graph.input("x", &[3, 33]);
        let w = graph.parameter("w", &[33, 5]);
        let b = graph.input("b", &[3, 5]);
        let y = graph.matmul(x, w);
        let y = graph.add(y, b);
        let loss = graph.mean_all(y);
        graph.set_outputs(vec![loss]);
        for mode in [Mode::Inference, Mode::Training] {
            let (mut session, report) = build_measured(
                &graph,
                SessionConfig {
                    mode,
                    gpu: Some(gpu.clone()),
                    runtime: crate::SessionOptions {
                        coop: crate::CoopPolicy::Disabled,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                BuildSearchOptions {
                    max_graphs: 4,
                    tuning: TuneOptions {
                        max_time: Duration::from_secs(1),
                        sample_pairs: 4,
                        warmup_runs: 1,
                        dispatches_per_sample: 1,
                        ..Default::default()
                    },
                    warmup_runs: 1,
                    warmup_time: Duration::from_millis(1),
                    max_time: Duration::from_secs(60),
                    max_programs: 24,
                    max_submission_chunks: 64,
                    min_improvement: 0.01,
                    max_plan_bytes: 4 << 20,
                    patience: None,
                    kernel_memo: None,
                },
                |s, _| {
                    s.set_input("x", &[0.25; 99]);
                    s.set_input("b", &[0.0625; 15]);
                    s.set_parameter("w", &[0.125; 165]);
                    Ok(())
                },
                |s| {
                    if (s.read_output(1)[0] - 1.09375).abs() > 2e-6 {
                        return Err("forward result changed".into());
                    }
                    assert_eq!(s.read_params(&["w"])[0], [0.125; 165]);
                    if mode == Mode::Training {
                        let mut grad = [0.0; 165];
                        s.read_param_grad("w", &mut grad);
                        if grad.iter().any(|g| (g - 0.05).abs() > 2e-6) {
                            return Err("parameter gradient changed".into());
                        }
                    }
                    Ok(())
                },
            )
            .unwrap();
            assert!(report.graphs.len() >= 3);
            // Four tile schedules plus the unfused form do not fit in max_graphs.
            assert!(report.extraction_truncated);
            assert!(report.trials.len() <= 24);
            assert!(report.skipped_regions.is_empty());
            assert!(report.trials.len() >= report.graphs.len());
            assert!(report.trials.iter().all(|t| t.outcome.qualified));
            for chunks in [2, 4, 8, 16, 32, 64] {
                assert!(report.trials.iter().any(|trial| {
                    trial
                        .description
                        .ends_with(&format!("submission_chunks={chunks}"))
                }));
            }
            assert_eq!(session.read_params(&["w"])[0], [0.125; 165]);
            if mode == Mode::Training {
                session.set_learning_rate(0.1);
                session.step();
                session.wait();
                assert!(
                    session.read_params(&["w"])[0]
                        .iter()
                        .all(|w| (w - 0.12).abs() < 2e-6)
                );
            }
        }
    }
}
