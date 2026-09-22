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
use std::time::{Duration, Instant};

#[derive(Clone, Serialize)]
pub struct BuildSearchOptions {
    /// Bound on logical forms, including ordinary extraction.
    pub max_graphs: usize,
    /// Per-program kernel tuning and paired whole-step decision policy.
    pub tuning: TuneOptions,
    /// Whole-program warmup pairs, independent of private kernel warmup.
    pub warmup_runs: u32,
    /// Soft total deadline, including construction, initialization and validation.
    /// In-flight driver work and caller validation cannot be preempted.
    pub max_time: Duration,
    pub max_programs: usize,
    /// Sum of declared logical bytes and persistent-state snapshots for both
    /// incumbent and challenger. This is not a driver-heap bound: padding,
    /// pipelines, staging and kernel-probe scratch are additional.
    pub max_plan_bytes: usize,
}

impl Default for BuildSearchOptions {
    fn default() -> Self {
        Self {
            max_graphs: 4,
            tuning: TuneOptions::default(),
            warmup_runs: 2,
            max_time: Duration::from_secs(30),
            max_programs: 64,
            max_plan_bytes: 512 << 20,
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
    pub elapsed: Duration,
}

/// Build from representative inputs, measuring logical and physical alternatives.
///
/// Unlike [`super::build`], this executes private candidate sessions.
/// `initialize` writes representative inputs and weights once per candidate. Its
/// optional idle incumbent can donate identically represented immutable parameters
/// via [`Session::share_parameter_from`], but must not be modified. Inputs and
/// writable state must remain private. Configure runtime optimizers, accumulation
/// and external bindings after construction, not inside either callback.
///
/// The runner executes one step before each read-only `qualify` call. Check every
/// observable output, gradient and persistent update against your numerical
/// contract. Qualification runs before/after kernel tuning and after measurement;
/// a failing challenger is discarded, a failing incumbent aborts construction.
/// Written inputs, parameters and constants are reset before each step, outside
/// timing. The returned session retains its initialized persistent state; outputs
/// hold its last qualified step. Timing includes fresh recording/submission/wait,
/// not readback. These search samples are not held-out benchmark results.
///
/// The ordinary optimizer supplies the first candidate. Egglog retains alternatives
/// from the original forward graph (one bounded region, repeated where verified).
/// They are not greedily optimized again. Each training form is differentiated
/// separately, so parameter transformations and gradients stay consistent.
/// NN tile and split-K choices are egglog equalities. Lowering emits the
/// extracted schedule and locks it. Complete plans still vary dispatch fusion,
/// cached-attention splits, low-occupancy convolution weight-gradient splits,
/// and submission chunk counts. Unlocked dispatches are kernel-tuned before
/// comparison. This bounded search does not promise a global optimum.
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
    options.tuning.validate().map_err(|e| e.to_string())?;
    if cfg.runtime.debug {
        cfg.options.fuse_dispatches = false;
    }
    let gpu = cfg.gpu.take().unwrap_or_else(runtime::default_gpu_context);
    let caps = cfg
        .runtime
        .coop
        .filter_caps(runtime::auto_tune(&gpu, 0).coop_caps);
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
        let source = forward_graph.toposort();
        let limit = options.max_graphs - 1;
        let spaces = if source.nodes().len()
            <= cfg
                .optimize
                .saturation_cutoff
                .min(optimize::SATURATION_CUTOFF)
        {
            vec![optimize::search::candidates(&source, cfg.optimize, limit)]
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
                    optimize::search::repeated_candidates(&source, region, cfg.optimize, limit)
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
        for &fusion in if cfg.options.fuse_dispatches {
            &[true, false][..]
        } else {
            &[false][..]
        } {
            let options = compile::CompileOptions {
                fuse_dispatches: fusion,
                ..cfg.options.clone()
            };
            let plan = compile::compile_with_caps(&graph, &options, caps);
            if !seeds.iter().any(|seed| seed.plan == plan) {
                seeds.push(Seed {
                    description: format!("graph={index}, dispatch_fusion={fusion}"),
                    plan,
                    graph: graph.clone(),
                    options,
                });
            }
        }
    }
    let preparation_time = start.elapsed();
    let programs = implementations(
        seeds,
        caps,
        gpu.capabilities().max_compute_shared_memory_size,
        options.max_plan_bytes,
    );
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

struct Seed {
    graph: std::sync::Arc<Graph>,
    options: compile::CompileOptions,
    plan: ExecutionPlan,
    description: String,
}

type AxisChoice = (
    (u32, Option<(u32, crate::codegen::FlashAttentionShape)>),
    usize,
);

fn early_physical_cover(
    chunks: &[usize],
    attention: &[(u32, Option<(u32, crate::codegen::FlashAttentionShape)>)],
) -> Vec<AxisChoice> {
    let mut cover = Vec::new();
    if let Some(&chunk) = chunks.get(1) {
        cover.push(((0, None), chunk));
    }
    if let Some(&choice) = attention.get(1) {
        cover.push((choice, 1));
        if let Some(&chunk) = chunks.get(1) {
            cover.push((choice, chunk));
        }
    }
    cover
}

/// Baseline on every logical form, then the early physical cover on the ordinary
/// graph, then that cover on the other forms, then the remaining schedules.
fn physical_program_order(
    n_seeds: usize,
    baseline: AxisChoice,
    cover: &[AxisChoice],
    tail: &[AxisChoice],
) -> Vec<(usize, AxisChoice)> {
    let mut order = Vec::new();
    for seed in 0..n_seeds {
        order.push((seed, baseline));
    }
    for &choice in cover {
        order.push((0, choice));
    }
    for &choice in cover {
        for seed in 1..n_seeds {
            order.push((seed, choice));
        }
    }
    for &choice in tail {
        for seed in 0..n_seeds {
            order.push((seed, choice));
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
    let chunks = [1, 2, 4, 8, 16, 32, 64];
    let baseline = ((0, None), 1usize);
    // The first alternative on each axis, then those axes crossed, on the
    // ordinary graph before other seeds repeat them.
    let cover = early_physical_cover(&chunks, &attention);
    let axis_len = attention.len().max(chunks.len());
    let single_axis = (0..axis_len).flat_map(|i| {
        [
            attention.get(i + 1).copied().map(|choice| (choice, 1)),
            chunks.get(i + 1).copied().map(|chunk| ((0, None), chunk)),
        ]
        .into_iter()
        .flatten()
    });
    let product = chunks.into_iter().flat_map(|chunk| {
        attention
            .clone()
            .into_iter()
            .filter_map(move |attention_choice| {
                let axes = usize::from(chunk != 1) + usize::from(attention_choice != (0, None));
                (axes >= 2).then_some((attention_choice, chunk))
            })
    });
    let mut seen = vec![baseline];
    seen.extend(cover.iter().copied());
    let tail: Vec<_> = single_axis
        .chain(product)
        .filter(|choice| !seen.contains(choice))
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
            let ((splits, flash), chunks) = current;
            let plan = if splits == 0 && flash.is_none() {
                seed.plan.clone()
            } else {
                let mut options = seed.options.clone();
                if splits != 0 {
                    options.cached_attention_splits = Some(splits);
                }
                if let Some((ept, shape)) = flash {
                    options.knobs.flash_ept_cap = ept;
                    options.knobs.flash = shape;
                    options.flash_forward_coop = false;
                }
                let plan = compile::compile_with_caps(&seed.graph, &options, caps);
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
                    "{}, attention_splits={splits}, flash={flash:?}, submission_chunks={chunks}",
                    seed.description
                ),
                plan,
                submission_chunks: chunks,
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
                "{}, conv_dw_splits={splits}, workgroups<48",
                seed.description
            ),
            plan,
            submission_chunks: 1,
        });
    }
    programs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn submission_chunks_lead_the_physical_cover() {
        let chunks = [1, 2, 4, 8, 16, 32, 64];
        let baseline = ((0, None), 1usize);
        let cover = early_physical_cover(&chunks, &[]);
        assert_eq!(cover, vec![((0, None), 2)]);
        let order = physical_program_order(8, baseline, &cover, &[((0, None), 4)]);
        let chunk = order
            .iter()
            .position(|&(seed, choice)| seed == 0 && choice == ((0, None), 2))
            .expect("two submission chunks on the ordinary graph");
        assert!(chunk < 16, "chunk plan landed at program {chunk}");
        assert_eq!(
            order[..8]
                .iter()
                .map(|&(seed, choice)| (seed, choice == baseline))
                .collect::<Vec<_>>(),
            (0..8).map(|seed| (seed, true)).collect::<Vec<_>>()
        );
        let cover_end = 8 + cover.len();
        assert!(
            order[8..cover_end].iter().all(|&(seed, _)| seed == 0),
            "other graphs wait until the ordinary graph has seen the cover"
        );
        assert_eq!(order[cover_end].0, 1);
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
                    max_time: Duration::from_secs(60),
                    max_programs: 24,
                    max_plan_bytes: 4 << 20,
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
            assert!(report.trials.len() > report.graphs.len());
            assert!(report.trials.iter().all(|t| t.outcome.qualified));
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
