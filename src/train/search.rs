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
/// Complete plans explore dispatch fusion and cached-attention splits before
/// allocation, plus submission chunk counts. Each plan is kernel-tuned before
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
        .filter(|d| {
            matches!(
                d.shader,
                compile::ShaderEntry::FlashAttention
                    | compile::ShaderEntry::MultiHeadAttn
                    | compile::ShaderEntry::FlashAttentionCoop
            )
        })
        .map(|d| d.params[3])
        .collect();
    let mut attention = vec![(0, None)];
    if !heads.is_empty() {
        for ept in [32, 16, 8] {
            for threads in [256, 128] {
                for keys in [8, 16, 4] {
                    for interleave in [false, true] {
                        let shape = crate::codegen::FlashAttentionShape {
                            threads,
                            keys,
                            interleave,
                        };
                        if heads
                            .iter()
                            .all(|&hd| shape.shared_bytes(hd) <= u64::from(shared_memory_bytes))
                        {
                            attention.push((0, Some((ept, shape))));
                        }
                    }
                }
            }
        }
    }
    if cached_attention {
        attention.extend([1, 2, 4, 8, 16].map(|splits| (splits, None)));
    }
    let chunks = [1, 2, 4, 8, 16, 32, 64];
    // Explore kernel layouts across logical forms before multiplying them by
    // submission choices. Lower only the next candidate.
    let matrices = [
        (0, 32),
        (8, 64),
        (4, 64),
        (2, 64),
        (8, 32),
        (4, 32),
        (2, 32),
    ];
    let single_axis: Vec<_> = matrices
        .into_iter()
        .map(|matrix| ((0, None), matrix, 1))
        .chain(
            attention
                .iter()
                .copied()
                .skip(1)
                .map(|attention| (attention, (0, 32), 1)),
        )
        .collect();
    let mut choices = single_axis
        .into_iter()
        .chain(chunks.into_iter().flat_map(move |chunks| {
            attention.clone().into_iter().flat_map(move |attention| {
                matrices.into_iter().filter_map(move |matrix| {
                    (chunks != 1 || (attention != (0, None) && matrix.0 != 0))
                        .then_some((attention, matrix, chunks))
                })
            })
        }));
    let mut current = ((0, None), (0, 32), 1);
    let mut seed_index = seeds.len();
    std::iter::from_fn(move || {
        loop {
            if seed_index == seeds.len() {
                current = choices.next()?;
                seed_index = 0;
            }
            let seed = seeds.get(seed_index)?;
            seed_index += 1;
            let ((splits, flash), (matrix_splits, tile_size), chunks) = current;
            let mut plan = if splits == 0 && flash.is_none() {
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
            if matrix_splits != 0 {
                let shape = crate::codegen::ScalarMatmulShape {
                    tile_size,
                    k_stage: seed.options.knobs.matmul_k_stage,
                    interleave_columns: seed.options.knobs.matmul_interleave_columns,
                };
                let original_buffers = plan.buffers.len();
                let mut remaining = max_partial_bytes;
                for index in (0..plan.dispatches.len()).rev() {
                    if plan
                        .split_matmul(index, shape, matrix_splits, remaining)
                        .is_ok()
                    {
                        remaining -= plan.buffers.last().unwrap();
                    }
                }
                if plan.buffers.len() == original_buffers {
                    continue;
                }
            }
            return Some(measure::Program {
                description: format!(
                    "{}, attention_splits={splits}, flash={flash:?}, matrix_splits={matrix_splits}, matrix_tile={tile_size}, submission_chunks={chunks}",
                    seed.description
                ),
                plan,
                submission_chunks: chunks,
            });
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
            assert!(!report.extraction_truncated);
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
