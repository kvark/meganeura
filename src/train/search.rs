//! Measured construction: graph alternatives survive until their kernels are tuned.
use super::{SessionConfig, prepare_graph};
use crate::{
    Graph, Session, TuneOptions,
    compile::{self, ExecutionPlan},
    optimize, runtime,
};
use serde::Serialize;

mod measure;
mod storage;
mod winograd;
pub use measure::BuildSearchTrial;
use std::{
    collections,
    time::{Duration, Instant},
};

const INITIAL_LAYOUTS: usize = 3;
const ARITHMETIC_LAYOUTS: usize = 5;

#[derive(Clone, Serialize)]
pub struct BuildSearchOptions {
    /// Fresh executions checked at each qualification checkpoint, outside
    /// timing. More than one lets a caller verify a program's repeatability
    /// independently of numerical differences between candidate programs.
    /// Persistent writes and representative inputs are reset before each run.
    /// Repeated checks require `refresh_inputs` to upload identical data.
    pub qualification_runs: u32,
    /// Fresh qualification executions required after search selects its final
    /// program, even when that program was just checked. Zero (the default)
    /// adds no checks beyond requalifying an incumbent disturbed by challengers.
    /// State and identical representative inputs are restored for every run.
    /// Estimated cost is reserved within the soft search deadline.
    pub final_qualification_runs: u32,
    /// Bound on graph/implementation forms, including ordinary extraction.
    pub max_graphs: usize,
    /// Per-challenger kernel tuning and shared paired sampling policy.
    /// The initial fallback always retains its ordinary kernels.
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
    /// Includes the initial untuned fallback and its tuned challenger. One
    /// returns only the fallback, without probing or applying kernel decisions.
    pub max_programs: usize,
    /// Stop once this many consecutive trials have not replaced the incumbent.
    /// `None` searches until another bound ends it.
    pub patience: Option<usize>,
    /// Explore power-of-two submission chunk counts up to this bound (1..=64).
    /// One keeps every candidate on a single submission.
    pub max_submission_chunks: usize,
    /// Sum of declared logical bytes and persistent-state snapshots for both
    /// incumbent and challenger, counting inherited immutable parameters once.
    /// Written parameters and their snapshots remain private to each program.
    /// This is not a driver-heap bound: padding, pipelines, staging and
    /// kernel-probe scratch are additional.
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
            qualification_runs: 1,
            final_qualification_runs: 0,
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
    /// Additional arithmetic candidates allowed by the runtime cooperative policy.
    /// Unsupported device types are listed separately below.
    pub matrix_inputs: Vec<crate::DType>,
    /// Extracted forms, indexed by the `graph=N` trial descriptions.
    pub graphs: Vec<String>,
    pub extraction_truncated: bool,
    pub skipped_regions: Vec<String>,
    /// Requested arithmetic rejected by the device capability/numerical probe.
    pub unsupported_matrix_inputs: Vec<crate::DType>,
    pub preparation_time: Duration,
    /// Private, precision-preserving attention probe; its time is included in preparation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attention_reduction_profile: Option<AttentionReductionProfile>,
    pub selected: usize,
    pub trials: Vec<BuildSearchTrial>,
    pub truncated: bool,
    /// Stopped by [`BuildSearchOptions::patience`].
    pub patience_exhausted: bool,
    /// Stopped early because recent costs predict insufficient comparison time.
    pub deadline_reserved: bool,
    /// Requalification of the selected program after the last trial; zero
    /// when nothing ran after its own check and no final checks were requested.
    pub final_qualification_time: Duration,
    /// Successful fresh executions checked after selection, before returning.
    pub final_qualifications: u32,
    /// Measured once, before the first trial that probes kernels, when tuning
    /// calibrates; every later trial's tuning is given it. Its time is in
    /// `elapsed`, and a profile cached for the context is reused.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub machine_profile: Option<crate::tune::MachineProfile>,
    pub elapsed: Duration,
}

#[derive(Serialize)]
pub struct AttentionReductionProfile {
    /// q_seq, kv_seq, query heads, KV heads, head dimension, elements per lane.
    pub shape: [u32; 6],
    pub portable_ms: Vec<f64>,
    pub subgroup_ms: Vec<f64>,
    pub selected_subgroups: bool,
    pub elapsed: Duration,
}

fn profile_attention_reduction(
    gpu: &std::sync::Arc<blade_graphics::Context>,
    deadline: Instant,
    params: &[u32],
    cap: u32,
    scratch_limit: usize,
) -> Option<AttentionReductionProfile> {
    if deadline.saturating_duration_since(Instant::now()) < Duration::from_secs(1) {
        return None;
    }
    let start = Instant::now();
    let head_dim = params[3];
    let groups = (params[2] >> 16) / (params[2] & 0xFFFF);
    let kv_heads = (params[2] & 0xFFFF).min((16 / groups).max(1));
    let heads = kv_heads * groups;
    let q_width = heads * head_dim;
    let kv_width = kv_heads * head_dim;
    let q_seq = params[0].min(128).min((262144 / q_width).max(1));
    let causal = params.get(5).is_some_and(|&rows| rows > 0) || params[1] == 0;
    let kv_seq = if causal {
        q_seq
    } else {
        params[1].min(128).min((262144 / kv_width).max(1))
    };
    let q_len = (q_seq * q_width) as usize;
    let kv_len = (kv_seq * kv_width) as usize;
    if 32 * (q_len + kv_len) + 4096 > scratch_limit {
        return None;
    }
    let mut graph = Graph::new();
    let q = graph.parameter("q", &[q_seq as usize, q_width as usize]);
    let k = graph.parameter("k", &[kv_seq as usize, kv_width as usize]);
    let v = graph.parameter("v", &[kv_seq as usize, kv_width as usize]);
    let y = if causal {
        graph.causal_attention(q, k, v, heads, kv_heads, head_dim)
    } else {
        graph.multi_head_attn(q, k, v, heads, kv_heads, head_dim, q_seq != kv_seq)
    };
    let loss = graph.mean_all(y);
    graph.set_outputs(vec![loss]);
    let graph = crate::autodiff::differentiate(&graph);
    let mut sessions = Vec::new();
    for subgroups in [false, true] {
        let mut options = compile::CompileOptions {
            flash_forward_coop: false,
            prefer_attention_grad: Some(crate::kernels::attention_grad::Path::Flash),
            ..Default::default()
        };
        options.knobs.flash_grad_q_ept_cap = cap;
        options.knobs.flash_grad_kv_ept_cap = cap;
        options.knobs.flash_subgroup_tiles = subgroups;
        let plan = compile::compile_with(&graph, &options);
        let mut session = Session::with_context_opts(
            plan,
            gpu.clone(),
            crate::SessionOptions {
                coop: crate::CoopPolicy::NativeF32,
                ..Default::default()
            },
        );
        for name in ["q", "k", "v"] {
            let len = if name == "q" { q_len } else { kv_len };
            let data: Vec<_> = (0..len).map(|i| (i % 97) as f32 * 0.01 - 0.48).collect();
            session.set_parameter(name, &data);
        }
        session.step();
        session.wait();
        sessions.push(session);
    }
    let mut encoder = gpu.create_command_encoder(blade_graphics::CommandEncoderDesc {
        name: "attention-reduction-profile",
        buffer_count: 1,
        manual_barriers: false,
    });
    let mut measure = |subgroups| {
        if Instant::now() >= deadline {
            return None;
        }
        let start = Instant::now();
        encoder.start();
        for _ in 0..16 {
            sessions[usize::from(subgroups)].record(&mut encoder).ok()?;
        }
        let sync = gpu.submit(&mut encoder);
        if !gpu.wait_for(&sync, !0).ok()? {
            return None;
        }
        Some(start.elapsed().as_secs_f64() * 1000.0 / 16.0)
    };
    let warm = Instant::now();
    while warm.elapsed() < Duration::from_millis(25) {
        if measure(false).is_none() || measure(true).is_none() {
            break;
        }
    }
    let (portable_ms, subgroup_ms) = crate::tune::measure_pairs(4, &mut measure);
    gpu.destroy_command_encoder(&mut encoder);
    let selected_subgroups = portable_ms.len() == 4
        && crate::tune::median(&subgroup_ms) < 0.98 * crate::tune::median(&portable_ms);
    Some(AttentionReductionProfile {
        shape: [q_seq, kv_seq, heads, kv_heads, head_dim, cap],
        portable_ms,
        subgroup_ms,
        selected_subgroups,
        elapsed: start.elapsed(),
    })
}

/// Build from representative inputs, measuring logical and physical alternatives.
///
/// Unlike [`super::build`], this executes private candidate sessions: the
/// ordinary program, bounded egglog graph forms (each differentiated
/// separately for training), and physical choices such as dispatch fusion,
/// attention layouts, checked matrix arithmetic, weight-gradient splits and
/// submission chunks, in `CoordinateOrder`. Challengers are kernel-tuned, but
/// only a whole-program paired comparison (fresh recording, submission and
/// wait; not readback) can replace the incumbent. No global optimum is promised.
///
/// `initialize` writes the same representative inputs and weights every time.
/// A challenger shares its idle incumbent's unwritten parameters
/// ([`Session::inherits_parameter`]); the incumbent must not be modified.
/// Configure optimizers, accumulation and bindings afterwards. `qualify` checks
/// every output, gradient and persistent update after one step, repeated
/// `qualification_runs` times; different plans, including one plan before and
/// after kernel tuning, may use different arithmetic. The untuned ordinary
/// program is qualified first and is the fallback. Challengers are qualified
/// before timing and again on a win, and are discarded on failure. The selected
/// program is rechecked if others ran since, or per `final_qualification_runs`;
/// that failure aborts. State is reset before each step, outside timing; the
/// result keeps its initialized state and its last qualified outputs.
///
/// `options` bound graphs, programs, plan bytes and a soft total deadline, and
/// `options.tuning` replaces `cfg.tune`. The build-plan cache is unsupported.
/// Inputs stay resident; [`build_measured_with_inputs`] uploads them each step.
pub fn build_measured(
    forward_graph: &Graph,
    cfg: SessionConfig<'_>,
    options: BuildSearchOptions,
    initialize: impl FnMut(&mut Session, Option<&mut Session>) -> Result<(), String>,
    qualify: impl FnMut(&Session) -> Result<(), String>,
) -> Result<(Session, BuildSearchReport), String> {
    build_measured_with_inputs(forward_graph, cfg, options, initialize, |_| Ok(()), qualify)
}

/// Measured construction with representative input uploads before every step.
///
/// Follows [`build_measured`]'s qualification and state contract. After restoring
/// persistent state, `refresh_inputs` must upload the same representative inputs
/// for each qualification, warmup and timing execution. It must not modify
/// parameters, optimizer configuration or other session state. Its time is
/// outside execution samples but inside the common search deadline and recorded
/// in [`BuildSearchTrial::input_refresh_time`] (the final incumbent refresh is
/// part of [`BuildSearchReport::final_qualification_time`]). A refresh failure
/// rejects the challenger; failure on the final incumbent invalidates construction.
///
/// Match how the application feeds inputs: host uploads can change subsequent
/// execution costs even when the upload itself is outside the step timer.
/// `refresh_inputs` must upload identical representative data on every call,
/// including repeated qualification runs; it must not advance through a dataset.
pub fn build_measured_with_inputs(
    forward_graph: &Graph,
    mut cfg: SessionConfig<'_>,
    options: BuildSearchOptions,
    initialize: impl FnMut(&mut Session, Option<&mut Session>) -> Result<(), String>,
    refresh_inputs: impl FnMut(&mut Session) -> Result<(), String>,
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
    if options.qualification_runs == 0 {
        return Err("measured construction needs positive qualification runs".into());
    }
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
        cfg.options.bf16_matrix_gradients,
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
                                cfg.options.bf16_matrix_gradients,
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
    let mut backward_ranks = vec![0; graphs.len()];
    if cfg.mode == crate::Mode::Training
        && cfg.optimize.mode != optimize::OptimizeMode::Off
        && options.max_graphs > 1
    {
        let (space, ranks) = backward_layouts(
            graphs,
            cfg.optimize,
            target,
            options.max_graphs,
            start + options.max_time,
            &mut skipped_regions,
        );
        backward_ranks = ranks;
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
                backward_rank: backward_ranks[index],
                description: format!("graph={index}"),
                plan,
                graph,
                options: cfg.options.clone(),
            });
        }
    }
    let attention_reduction_profile = seeds
        .iter()
        .find_map(|seed| {
            seed.plan.dispatches.iter().find_map(|d| {
                flash_reduction_cap(d, &seed.plan.knobs, gpu.capabilities().subgroup_size)
                    .map(|cap| (d.params.clone(), cap))
            })
        })
        .and_then(|(params, cap)| {
            profile_attention_reduction(
                &gpu,
                start + options.max_time,
                &params,
                cap,
                options.tuning.max_scratch_bytes,
            )
        });
    if let Some(ref profile) = attention_reduction_profile {
        for seed in &mut seeds {
            seed.options.knobs.flash_subgroup_tiles = profile.selected_subgroups;
            seed.plan.knobs.flash_subgroup_tiles = profile.selected_subgroups;
        }
    }
    // Arithmetic permission belongs to the runtime policy, not a second opt-in.
    // Checked F16 reaches rectangular tiles and extracted scalar layouts that
    // the ordinary square-tile heuristic leaves alone.
    let matrix_inputs = matrix_input_order(cfg.runtime.coop, caps);
    let (supported, unsupported_matrix_inputs): (Vec<_>, Vec<_>) = matrix_inputs
        .iter()
        .copied()
        .partition(|&ty| runtime::typed_matrix_supported(&gpu, ty));
    let mut programs = implementations(seeds, caps, shared_memory_bytes, options.max_plan_bytes);
    // Arithmetic alternatives may replace this search's extracted scalar layouts,
    // but keep layouts already present in the caller's graph.
    programs.replace_extracted_matrices = forward_graph
        .nodes()
        .iter()
        .all(|n| n.matmul_impl.is_none());
    programs.replace_extracted_attention = forward_graph
        .nodes()
        .iter()
        .all(|n| n.attention_ept_cap.is_none() && n.attention_impl.is_none());
    programs.allow_matrix_storage = !cfg.runtime.debug && cfg.runtime.pin_buffers.is_none();
    programs.enable_matrix_inputs(&supported, &gpu.capabilities());
    programs.enable_gradient_inputs(&supported);
    programs.attention_subgroup_size = gpu.capabilities().subgroup_size;
    if cfg.mode == crate::Mode::Inference && cfg.options.flash_forward_coop {
        programs.enable_forward_products(gpu.capabilities().subgroup_size);
    }
    programs.enable_flash_reductions();
    if cfg.mode == crate::Mode::Inference
        && matches!(
            cfg.runtime.coop,
            crate::CoopPolicy::Auto | crate::CoopPolicy::AllowF16
        )
        && gpu.capabilities().shader_float16
    {
        programs.enable_flash_tiles();
    }
    if cfg.mode == crate::Mode::Training {
        programs.enable_flash_backward(
            matches!(
                cfg.runtime.coop,
                crate::CoopPolicy::Auto | crate::CoopPolicy::AllowF16
            ) && gpu.capabilities().shader_float16,
        );
    }
    let preparation_time = start.elapsed();
    measure::select(
        programs,
        gpu,
        cfg.runtime,
        BuildSearchReport {
            options,
            graphs: expressions,
            extraction_truncated,
            skipped_regions,
            matrix_inputs,
            unsupported_matrix_inputs,
            preparation_time,
            attention_reduction_profile,
            ..Default::default()
        },
        start,
        (initialize, refresh_inputs, qualify),
    )
}

fn matrix_input_order(
    policy: crate::CoopPolicy,
    caps: crate::codegen::CoopCaps,
) -> Vec<crate::DType> {
    use crate::{CoopPolicy, DType};
    match policy {
        // Rectangular and eight-wide checked products can improve the ordinary
        // F32/scalar path. Prefer F16's mantissa first, but retain BF16
        // as an independent candidate, including when F16 fails qualification.
        CoopPolicy::Auto if matches!(caps.f16_tile, 0 | 8) => vec![DType::F16, DType::BF16],
        CoopPolicy::Auto => vec![DType::BF16, DType::F16],
        CoopPolicy::AllowF16 => vec![DType::F16],
        CoopPolicy::NativeF32 | CoopPolicy::Disabled => Vec::new(),
    }
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
) -> (optimize::search::SearchSpace, Vec<usize>) {
    let mut ranks = Vec::new();
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
            return (
                optimize::search::SearchSpace {
                    candidates,
                    truncated: true,
                },
                ranks,
            );
        }
        let graph = &graphs[index];
        if layout == 0 {
            ranks.push(layout);
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
            ranks.push(layout);
            candidates.push(optimize::search::Candidate {
                graph: form.graph,
                expression: format!("{}; backward={}", graph.expression, form.expression),
            });
        }
    }
    (
        optimize::search::SearchSpace {
            candidates,
            truncated: truncated || layouts.iter().flatten().any(|pending| !pending.is_empty()),
        },
        ranks,
    )
}

struct Seed {
    backward_rank: usize,
    graph: std::sync::Arc<Graph>,
    options: compile::CompileOptions,
    plan: ExecutionPlan,
    description: String,
}

type AttentionChoice = (u32, Option<(u32, crate::codegen::FlashAttentionShape)>);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
struct AxisChoice {
    attention: AttentionChoice,
    fuse_dispatches: bool,
    matrix_inputs: Option<crate::DType>,
    matrix_storage: bool,
    vector_convolutions_only: bool,
    winograd: Option<crate::DType>,
    f16_attention: bool,
    /// Scalar flash staging as f16. Forward V and backward dO stay f32.
    f16_flash_tiles: bool,
    backward_ept: Option<u32>,
    subgroup_tiles: Option<bool>,
    gradient_inputs: Option<crate::DType>,
    forward_products: bool,
}

fn flash_reduction_cap(
    d: &compile::Dispatch,
    knobs: &compile::TuningKnobs,
    subgroup_size: u32,
) -> Option<u32> {
    if d.shader == compile::ShaderEntry::FlashAttention {
        return (subgroup_size > 0).then_some(4);
    }
    let compile::ShaderEntry::AttentionGrad(k) = d.shader else {
        return None;
    };
    if k.path != crate::kernels::attention_grad::Path::Flash {
        return None;
    }
    let &head_dim = d.params.get(3)?;
    let cap = match d.kernel {
        compile::Kernel::AttentionBackward { ept_cap } => ept_cap,
        _ => match k.part {
            crate::kernels::attention_grad::Part::Q => knobs.flash_grad_q_ept_cap,
            crate::kernels::attention_grad::Part::KV => knobs.flash_grad_kv_ept_cap,
        },
    };
    let lanes = crate::codegen::attention_lanes(head_dim, cap).1;
    (lanes > 8 && subgroup_size >= lanes && subgroup_size.is_multiple_of(lanes)).then_some(cap)
}

fn has_flash_reduction(plan: &compile::ExecutionPlan, subgroup_size: u32) -> bool {
    plan.dispatches
        .iter()
        .any(|d| flash_reduction_cap(d, &plan.knobs, subgroup_size).is_some())
}

/// Coordinate order over graph forms and physical choices (indices into the
/// seed and choice lists). The first layouts run with the baseline choice,
/// except when early arithmetic coverage is requested after the first form.
/// After that, steps alternate between the next untried graph form, run with
/// the incumbent's physical choice, and the next untried physical choice on
/// the incumbent's form. A winner on either axis carries over to the other:
/// later forms inherit a winning choice, and a winning form restarts the
/// choices it has not yet run with.
///
/// Physical choices such as attention layouts mostly act independently of the
/// graph form. A diagonal over form × choice rank spends a bounded budget
/// re-measuring the same few choices on every form; on graphs with many
/// forms it never reaches the later choices at all. Pairs off both axes
/// follow, diagonally, once the axes around the incumbent are exhausted.
#[derive(Debug)]
struct CoordinateOrder {
    seeds: usize,
    choices: usize,
    /// Stable choice indices, prioritized around the incumbent's physical axes.
    choice_order: Vec<usize>,
    form_order: Vec<usize>,
    tried: collections::HashSet<(usize, usize)>,
    /// Forms that have run with any choice.
    visited_forms: Vec<bool>,
    initial: usize,
    physical_next: bool,
    priority_choices: collections::VecDeque<usize>,
    /// Queued indices retain their original intent; resolved candidates are
    /// recomputed for the latest incumbent before each trial.
    rebased_choices: collections::HashMap<usize, usize>,
    /// Forms to cover before the first arithmetic alternatives. Eight-wide
    /// products need early coverage of arithmetic, not just scalar layouts.
    arithmetic_layouts: usize,
    priority_attention: Option<usize>,
    deferred: collections::HashSet<(usize, usize)>,
}

impl CoordinateOrder {
    fn new(seeds: usize, choices: usize) -> Self {
        Self {
            seeds,
            choices,
            choice_order: (0..choices).collect(),
            form_order: (0..seeds).collect(),
            tried: collections::HashSet::new(),
            visited_forms: vec![false; seeds],
            initial: 0,
            physical_next: false,
            priority_choices: collections::VecDeque::new(),
            rebased_choices: collections::HashMap::new(),
            arithmetic_layouts: ARITHMETIC_LAYOUTS,
            priority_attention: None,
            deferred: collections::HashSet::new(),
        }
    }

    fn candidate_choice(&self, template: usize) -> usize {
        self.rebased_choices
            .get(&template)
            .copied()
            .unwrap_or(template)
    }

    fn has_priority(&self, candidate: usize) -> bool {
        self.priority_choices
            .iter()
            .any(|&template| self.candidate_choice(template) == candidate)
    }

    /// The next (form, choice) to measure, given the incumbent's.
    fn next(&mut self, incumbent: (usize, usize)) -> Option<(usize, usize)> {
        if self.seeds == 0 || self.choices == 0 {
            return None;
        }
        // Retain ordinary graph/derivative coverage on the existing targets.
        // Eight-wide products cover arithmetic after the ordinary fallback,
        // before layout and low-occupancy split exploration spend the deadline.
        if self
            .visited_forms
            .iter()
            .filter(|&&visited| visited)
            .count()
            >= self.seeds.min(self.arithmetic_layouts)
        {
            while let Some(choice) = self.priority_choices.pop_front() {
                let choice = self.rebased_choices.remove(&choice).unwrap_or(choice);
                let entry = (incumbent.0, choice);
                if self.deferred.contains(&entry) {
                    continue;
                }
                if self.mark(entry) {
                    return Some(entry);
                }
            }
            if let Some(choice) = self.priority_attention.take() {
                let entry = (incumbent.0, choice);
                if self.mark(entry) {
                    return Some(entry);
                }
            }
        }
        while self.initial < self.seeds.min(INITIAL_LAYOUTS) {
            let entry = (self.initial, 0);
            self.initial += 1;
            if self.mark(entry) {
                return Some(entry);
            }
        }
        let (form, choice) = incumbent;
        let available = |entry: &(usize, usize)| self.available(*entry);
        let physical = self
            .choice_order
            .iter()
            .copied()
            .map(|c| (form, c))
            .find(available);
        // Unvisited forms first; once every form has run, revisit them with the
        // incumbent's choice.
        let graph = self
            .form_order
            .iter()
            .copied()
            .map(|s| (s, choice))
            .find(|entry| !self.visited_forms[entry.0] && available(entry))
            .or_else(|| (0..self.seeds).map(|s| (s, choice)).find(available));
        // Both axes exhausted around the incumbent: the remaining pairs,
        // diagonally, for budgets large enough to cover them.
        let rest = |include_deferred| {
            (0..self.seeds + self.choices - 1)
                .flat_map(|rank| (0..self.choices.min(rank + 1)).map(move |c| (rank - c, c)))
                .find(|&(s, c)| {
                    s < self.seeds
                        && !self.tried.contains(&(s, c))
                        && (include_deferred || available(&(s, c)))
                })
        };
        let entry = if self.physical_next {
            physical.or(graph)
        } else {
            graph.or(physical)
        }
        .or_else(|| rest(false))
        .or_else(|| rest(true))?;
        self.physical_next = !self.physical_next;
        self.mark(entry);
        Some(entry)
    }

    /// Untried and not deferred.
    fn available(&self, entry: (usize, usize)) -> bool {
        !self.tried.contains(&entry) && !self.deferred.contains(&entry)
    }

    fn mark(&mut self, entry: (usize, usize)) -> bool {
        self.visited_forms[entry.0] = true;
        self.tried.insert(entry)
    }

    fn exhausted(&self) -> bool {
        self.tried.len() == self.seeds * self.choices
    }
}

fn implementations(
    seeds: Vec<Seed>,
    caps: crate::codegen::CoopCaps,
    shared_memory_bytes: u32,
    max_partial_bytes: usize,
) -> Implementations {
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
        .map(|d| {
            use crate::compile::{AttentionParams, DispatchParams};
            AttentionParams::from_words(&d.params)
                .expect("invalid attention parameters")
                .head_dim
        })
        .collect();
    let mut attention = vec![(0, None)];
    if !heads.is_empty() {
        let mut layouts = Vec::new();
        for (e, ept) in [32, 16, 8].into_iter().enumerate() {
            for (t, threads) in [256, 128].into_iter().enumerate() {
                for (k, keys) in [8, 16, 4].into_iter().enumerate() {
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
                            // Cover both lane assignments at each EPT before
                            // combining tile and workgroup changes.
                            layouts.push(((e + t + k, t + k), (0, Some((ept, shape)))));
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
    // The seed's own fusion setting first, so each axis's first alternative
    // (unfused dispatches, then the next attention layout) ranks early.
    let fusion: &[bool] = if seeds
        .first()
        .is_some_and(|seed| seed.options.fuse_dispatches)
    {
        &[true, false]
    } else {
        &[false]
    };
    let mut choices: Vec<_> = attention
        .iter()
        .flat_map(|&attention| {
            fusion.iter().map(move |&fuse_dispatches| AxisChoice {
                attention,
                fuse_dispatches,
                ..Default::default()
            })
        })
        .collect();
    let f16_attention = caps.f16_attention_rows != 0
        && seeds.iter().any(|seed| {
            seed.plan
                .dispatches
                .iter()
                .any(|d| matches!(d.shader, compile::ShaderEntry::AttentionGrad(_)))
        });
    let first_choice = choices.len();
    if f16_attention {
        let ordinary = choices.clone();
        choices.extend(ordinary.into_iter().map(|choice| AxisChoice {
            f16_attention: true,
            ..choice
        }));
    }
    let mut order = CoordinateOrder::new(seeds.len(), choices.len());
    if f16_attention {
        order.priority_attention = Some(first_choice);
    }
    let split_plans = Vec::new();
    Implementations {
        order,
        choices,
        seeds,
        caps,
        shared_memory_bytes,
        max_partial_bytes,
        split_plans,
        queued_splits: false,
        replace_extracted_matrices: false,
        replace_extracted_attention: false,
        attention_subgroup_size: 0,
        defer_bf16: false,
        allow_matrix_storage: false,
        matrix_shapes: collections::HashMap::new(),
    }
}

/// Programs in [`CoordinateOrder`], lowered lazily.
struct Implementations {
    order: CoordinateOrder,
    choices: Vec<AxisChoice>,
    seeds: Vec<Seed>,
    caps: crate::codegen::CoopCaps,
    shared_memory_bytes: u32,
    max_partial_bytes: usize,
    split_plans: Vec<measure::Program>,
    queued_splits: bool,
    replace_extracted_matrices: bool,
    replace_extracted_attention: bool,
    attention_subgroup_size: u32,
    defer_bf16: bool,
    allow_matrix_storage: bool,
    matrix_shapes: collections::HashMap<crate::DType, ([u32; 3], u32)>,
}

impl Implementations {
    fn enable_gradient_inputs(&mut self, inputs: &[crate::DType]) {
        let ordinary = self.choices.clone();
        for multiply in [crate::DType::BF16, crate::DType::F16] {
            if !inputs.contains(&multiply)
                || !self.matrix_shapes.contains_key(&multiply)
                || !self.seeds.iter().any(|seed| {
                    fill_gradient_candidates(
                        &mut seed.plan.clone(),
                        multiply,
                        self.replace_extracted_matrices,
                    ) != 0
                })
            {
                continue;
            }
            // Derivatives prefer BF16's range when available. Keep both choices
            // measurable; a single F16 qualification cannot prove future range.
            self.order.priority_choices.push_back(self.choices.len());
            self.choices
                .extend(ordinary.iter().map(|choice| AxisChoice {
                    gradient_inputs: Some(multiply),
                    ..*choice
                }));
        }
        self.order.choices = self.choices.len();
        self.order.choice_order = (0..self.choices.len()).collect();
    }

    fn enable_flash_reductions(&mut self) {
        if self
            .seeds
            .iter()
            .any(|seed| has_flash_reduction(&seed.plan, self.attention_subgroup_size))
        {
            self.order.priority_choices.push_back(self.choices.len());
            let ordinary = self.choices.clone();
            self.choices.extend(
                ordinary
                    .into_iter()
                    .filter(|c| !c.f16_attention)
                    .map(|choice| AxisChoice {
                        subgroup_tiles: Some(!self.seeds[0].options.knobs.flash_subgroup_tiles),
                        ..choice
                    }),
            );
        }
        self.order.choices = self.choices.len();
        self.order.choice_order = (0..self.choices.len()).collect();
    }

    fn enable_flash_backward(&mut self, allow_f16: bool) {
        if !self.seeds.iter().any(|seed| {
            seed.plan.dispatches.iter().any(|d| {
                matches!(d.shader, compile::ShaderEntry::AttentionGrad(k)
                if k.path == crate::kernels::attention_grad::Path::Flash)
            })
        }) {
            return;
        }
        let ordinary = self.choices.clone();
        // Preserve precision coverage before adding backward layouts. Spending
        // the deadline on geometry can hide a larger derivative-product win.
        for cap in [16, 32, 8] {
            self.order.priority_choices.push_back(self.choices.len());
            self.choices
                .extend(
                    ordinary
                        .iter()
                        .filter(|c| !c.f16_attention)
                        .map(|choice| AxisChoice {
                            backward_ept: Some(cap),
                            ..*choice
                        }),
                );
        }
        if allow_f16 {
            self.enable_flash_tiles();
        }
        self.order.choices = self.choices.len();
        self.order.choice_order = (0..self.choices.len()).collect();
    }

    fn enable_flash_tiles(&mut self) {
        if !self.seeds.iter().any(|seed| {
            seed.plan.dispatches.iter().any(|d| {
                matches!(
                    d.shader,
                    compile::ShaderEntry::FlashAttention | compile::ShaderEntry::AttentionGrad(_)
                )
            })
        }) {
            return;
        }
        let ordinary = self.choices.clone();
        self.order.priority_choices.push_back(self.choices.len());
        self.choices.extend(
            ordinary
                .into_iter()
                .filter(|c| !c.f16_attention)
                .map(|choice| AxisChoice {
                    f16_flash_tiles: true,
                    forward_products: false,
                    ..choice
                }),
        );
        self.order.choices = self.choices.len();
        self.order.choice_order = (0..self.choices.len()).collect();
    }

    fn enable_forward_products(&mut self, subgroup_size: u32) {
        self.attention_subgroup_size = subgroup_size;
        if !self.seeds.iter().any(|seed| {
            fill_attention_products(
                &mut seed.plan.clone(),
                self.caps,
                self.shared_memory_bytes,
                subgroup_size,
            ) > 0
        }) {
            return;
        }
        // Cover ordinary arithmetic before attention products, then compare
        // products on its winner and on each early arithmetic alternative.
        let priorities = self.order.priority_choices.clone();
        let ordinary = self.choices.clone();
        for (index, choice) in ordinary.into_iter().enumerate() {
            if choice.attention.1.is_some() {
                continue;
            }
            if index == 0 || priorities.contains(&index) {
                self.order.priority_choices.push_back(self.choices.len());
            }
            self.choices.push(AxisChoice {
                forward_products: true,
                ..choice
            });
        }
        self.order.choices = self.choices.len();
        self.order.choice_order = (0..self.choices.len()).collect();
    }

    fn enable_matrix_inputs(
        &mut self,
        inputs: &[crate::DType],
        caps: &blade_graphics::Capabilities,
    ) {
        // Rectangular subgroup kernels share their tile/layout choices. BF16
        // stays reachable, but need not interrupt layout search after F16
        // qualified on the same form and physical choice.
        self.defer_bf16 =
            crate::codegen::checked_cooperative_shape(&caps.cooperative_matrix.f16_f32_shapes)
                == Some([8, 16, 16])
                && crate::codegen::checked_cooperative_shape(
                    &caps.cooperative_matrix.bf16_f32_shapes,
                ) == Some([8, 16, 16])
                && caps.cooperative_matrix.workgroup.granularities.is_empty();
        let ordinary = self.choices.clone();
        let mut dense = vec![(None, false, false)];
        let mut winograd = vec![None];
        for &multiply in inputs {
            let shapes = match multiply {
                crate::DType::F16 if caps.shader_float16 => &caps.cooperative_matrix.f16_f32_shapes,
                crate::DType::BF16 if caps.shader_bfloat16 => {
                    &caps.cooperative_matrix.bf16_f32_shapes
                }
                _ => continue,
            };
            let Some(shape) = crate::codegen::checked_cooperative_shape(shapes) else {
                continue;
            };
            if shape == [8, 8, 8] {
                self.order.arithmetic_layouts = 1;
            }
            if self
                .matrix_shapes
                .insert(multiply, (shape, caps.subgroup_size))
                .is_some()
            {
                continue;
            }
            if self.seeds.iter().any(|s| {
                winograd::fill(&mut s.plan.clone(), multiply, shape, caps.subgroup_size) > 0
            }) {
                winograd.push(Some(multiply));
            }
            if self.seeds.iter().any(|s| {
                fill_matrix_candidates(
                    &mut s.plan.clone(),
                    multiply,
                    self.replace_extracted_matrices,
                ) > 0
            }) {
                let mixed = self.seeds.iter().any(|seed| {
                    let vector = fill_matrix_candidates_scoped(
                        &mut seed.plan.clone(),
                        multiply,
                        self.replace_extracted_matrices,
                        true,
                    );
                    vector > 0
                        && vector
                            < fill_matrix_candidates(
                                &mut seed.plan.clone(),
                                multiply,
                                self.replace_extracted_matrices,
                            )
                });
                if mixed {
                    dense.push((Some(multiply), true, false));
                }
                dense.push((Some(multiply), false, false));
                if self.allow_matrix_storage
                    && self.seeds.iter().any(|seed| {
                        let mut plan = seed.plan.clone();
                        fill_matrix_candidates(
                            &mut plan,
                            multiply,
                            self.replace_extracted_matrices,
                        );
                        storage::fill(&mut plan) != 0
                    })
                {
                    dense.push((Some(multiply), false, true));
                }
            }
        }
        // Independent precision coordinates keep both single-family candidates
        // and their combinations reachable, including mixed F16/BF16 programs.
        for &(matrix_inputs, vector_convolutions_only, matrix_storage) in &dense {
            for &winograd in &winograd {
                if matrix_inputs.is_none() && winograd.is_none() {
                    continue;
                }
                if !matrix_storage && (matrix_inputs.is_none() || winograd.is_none()) {
                    self.order.priority_choices.push_back(self.choices.len());
                }
                self.choices
                    .extend(ordinary.iter().map(|choice| AxisChoice {
                        matrix_inputs,
                        matrix_storage,
                        vector_convolutions_only,
                        winograd,
                        ..*choice
                    }));
            }
        }
        // Winograd's transforms amplify rounding error. Prefer F16's extra
        // mantissa bits here, independently of the dense-product type order.
        self.order
            .priority_choices
            .make_contiguous()
            .sort_by_key(|&index| {
                let choice = self.choices[index];
                (
                    choice.winograd.is_none(),
                    choice.winograd == Some(crate::DType::BF16),
                )
            });
        self.order.choices = self.choices.len();
        self.order.choice_order = (0..self.choices.len()).collect();
    }
}

/// Switch eligible inference attention to F16 QK/PV products.
/// Q/K/V values outside the F16 range can produce non-finite output. Admission
/// relies on the caller's quality check, which covers only the search inputs.
fn fill_attention_products(
    plan: &mut ExecutionPlan,
    caps: crate::codegen::CoopCaps,
    shared_memory_bytes: u32,
    subgroup_size: u32,
) -> usize {
    use compile::{DispatchParams as _, ShaderEntry};
    let mut count = 0;
    for dispatch in &mut plan.dispatches {
        if dispatch.requires_full_precision
            || dispatch.schedule_locked
            || !matches!(
                dispatch.shader,
                ShaderEntry::MultiHeadAttn
                    | ShaderEntry::FlashAttention
                    | ShaderEntry::FlashAttentionCoop
            )
        {
            continue;
        }
        let p = compile::AttentionParams::from_words(&dispatch.params)
            .expect("invalid attention parameters");
        if !crate::kernels::attention::admits_products(
            p.q_seq,
            p.head_dim,
            p.packed_heads >> 16,
            caps,
            shared_memory_bytes,
            subgroup_size,
        ) {
            continue;
        }
        dispatch.shader = ShaderEntry::FlashAttentionProducts;
        dispatch.label = format!(
            "FlashAttentionProducts {}x{} hd{} heads{}",
            p.q_seq,
            p.kv_seq,
            p.head_dim,
            p.packed_heads >> 16
        );
        dispatch.workgroups = [p.q_seq.div_ceil(16), p.packed_heads >> 16, 1];
        count += 1;
    }
    count
}

/// Checked typed products preserve the existing F32 or F16 weight bindings,
/// including tails. Conversion happens inside each timed dispatch. This reaches
/// rectangular subgroup shapes and workgroup tuning without the legacy heuristic.
fn fill_matrix_candidates(
    plan: &mut ExecutionPlan,
    multiply: crate::DType,
    replace_extracted: bool,
) -> usize {
    fill_matrix_candidates_scoped(plan, multiply, replace_extracted, false)
}

fn fill_matrix_candidates_scoped(
    plan: &mut ExecutionPlan,
    multiply: crate::DType,
    replace_extracted: bool,
    vector_convolutions_only: bool,
) -> usize {
    let mut count = 0;
    for d in &mut plan.dispatches {
        if d.shader.is_conv2d_gemm()
            && (!vector_convolutions_only
                || crate::kernels::conv::Params::from(&*d).vectorized_forward())
            && !d.full_precision()
            && !d.schedule_locked
            && d.matmul_epilogue.is_none()
            && d.extra_outputs.is_empty()
            && d.weight_format == compile::WeightFormat::F32
        {
            let p = &d.params;
            let Some(positions) = p[9].checked_mul(p[10]) else {
                continue;
            };
            let grid = [p[4].div_ceil(32), positions.div_ceil(32), p[0]];
            if !compile::workgroups_within_portable_limits(grid) {
                continue;
            }
            d.shader = compile::ShaderEntry::Conv2dGemmCoopGen(p[5], p[6], p[7]);
            d.kernel = compile::Kernel::CooperativeTyped {
                types: crate::codegen::MatrixTypes {
                    multiply,
                    ..crate::codegen::MatrixTypes::F32_AS_F16
                },
                splits: 1,
            };
            d.workgroups = grid;
            count += 1;
            continue;
        }

        if !d.full_precision() && fill_dense_product(d, multiply, replace_extracted) {
            count += 1;
        }
    }
    count
}

/// Only AT/BT derivative products were qualified by the dense controls. In
/// particular, SmolLM2's long-K NN gradient fails the unchanged F16 gate.
/// Bindings, partials and automatic derivative protection remain F32.
/// Reduced inputs are admitted by the caller's quality check on search-time
/// data only. F16 can lose small derivative values as training converges;
/// passing that check does not establish range or accuracy for future steps.
fn fill_gradient_candidates(
    plan: &mut ExecutionPlan,
    multiply: crate::DType,
    replace_extracted: bool,
) -> usize {
    let mut count = 0;
    for d in &mut plan.dispatches {
        if d.is_derivative
            && !d.requires_full_precision
            && matches!(
                d.shader,
                compile::ShaderEntry::MatMulAT | compile::ShaderEntry::MatMulBT
            )
            && matches!(
                d.kernel,
                compile::Kernel::Default
                    | compile::Kernel::ScalarMatmul(_)
                    | compile::Kernel::SplitMatmul { .. }
            )
            && d.input_buffers.len() == 2
            && d.extra_outputs.is_empty()
            && fill_dense_product(d, multiply, replace_extracted)
        {
            count += 1;
        }
    }
    count
}

fn fill_dense_product(
    d: &mut compile::Dispatch,
    multiply: crate::DType,
    replace_extracted: bool,
) -> bool {
    let b = match (d.weight_format, multiply) {
        (compile::WeightFormat::F32, _) => crate::DType::F32,
        (compile::WeightFormat::F16, crate::DType::F16) if !d.is_derivative => crate::DType::F16,
        _ => return false,
    };
    if !d.shader.is_matmul()
        || !matches!(
            d.kernel,
            compile::Kernel::Default
                | compile::Kernel::SmallTile
                | compile::Kernel::Cooperative
                | compile::Kernel::ScalarMatmul(_)
                | compile::Kernel::SplitMatmul { .. }
        )
        || (d.schedule_locked && !replace_extracted)
        || d.horizontal_batch > 1
        || d.matmul_prologue.is_some()
        || d.matmul_epilogue.is_some()
    {
        return false;
    }
    let Some((m, n, k)) = d.mnk() else {
        return false;
    };
    if m <= 1 || n == 0 || k == 0 {
        return false;
    }
    let splits = match d.kernel {
        compile::Kernel::SplitMatmul { splits, .. } => splits,
        _ => 1,
    };
    // Split products keep F32 partials and cannot fuse an addend.
    if splits > 1 && (d.input_buffers.len() != 2 || splits > k.div_ceil(32)) {
        return false;
    }
    let grid = [m.div_ceil(32), n.div_ceil(32), splits];
    if !compile::workgroups_within_portable_limits(grid) {
        return false;
    }
    // Legacy F16 parameters already contain packed half values. Describe
    // that binding explicitly; keep plan.weight_buffers for host uploads.
    d.weight_format = compile::WeightFormat::F32;
    d.kernel = compile::Kernel::CooperativeTyped {
        types: crate::codegen::MatrixTypes {
            b,
            multiply,
            ..crate::codegen::MatrixTypes::F32_AS_F16
        },
        splits,
    };
    d.schedule_locked = false;
    d.workgroups = grid;
    true
}

impl measure::Frontier for Implementations {
    fn early_programs_remaining(&self, incumbent: Option<measure::Origin>) -> usize {
        if self.order.arithmetic_layouts != 1 {
            return 0;
        }
        let form = incumbent.map_or(0, |origin| origin.form);
        self.order
            .priority_choices
            .iter()
            .filter(|&&choice| {
                self.order
                    .available((form, self.order.candidate_choice(choice)))
            })
            .count()
    }

    fn observe(&mut self, origin: measure::Origin, qualified: bool) {
        let choice = self.choices[origin.choice];
        if qualified && choice.matrix_inputs.is_some() && !choice.matrix_storage {
            let stored = AxisChoice {
                matrix_storage: true,
                ..choice
            };
            if let Some(index) = self.choices.iter().position(|&c| c == stored) {
                if !self.order.tried.contains(&(origin.form, index))
                    && !self.order.has_priority(index)
                {
                    self.order.priority_choices.push_back(index);
                }
            }
        }
        if !self.defer_bf16 {
            return;
        }
        // Forward storage families retain their existing fallback ordering.
        // Never defer BF16 derivatives after one successful F16 qualification.
        for winograd in [false, true] {
            let mut bf16 = choice;
            let family = if winograd {
                &mut bf16.winograd
            } else {
                &mut bf16.matrix_inputs
            };
            if family.replace(crate::DType::BF16) != Some(crate::DType::F16) {
                continue;
            }
            if let Some(index) = self.choices.iter().position(|&c| c == bf16) {
                let entry = (origin.form, index);
                if qualified {
                    self.order.deferred.insert(entry);
                } else {
                    self.order.deferred.remove(&entry);
                }
            }
        }
    }

    fn next(&mut self, incumbent: Option<measure::Origin>) -> Option<measure::Program> {
        loop {
            if let Some(plan) = self.split_plans.pop() {
                return Some(plan);
            }
            if !self.queued_splits
                && !self.order.tried.is_empty()
                && self.early_programs_remaining(incumbent) == 0
            {
                self.queued_splits = true;
                // After the ordinary incumbent and any early arithmetic cover.
                // Popped in reverse, so 4 splits is measured before 8.
                self.split_plans =
                    low_occupancy_weight_splits(self.seeds.first(), self.max_partial_bytes);
                self.split_plans.reverse();
                continue;
            }
            let incumbent = incumbent.map_or((0, 0), |origin| (origin.form, origin.choice));
            // Forward forms and derivative layouts are separate coordinates.
            // Try the incumbent's derivative rank on other forward forms before
            // spending the deadline on more derivative variants of early forms.
            // This is only an ordering hint: every form remains reachable and
            // must pass qualification and the whole-program timing gate.
            // Cover the first derivative alternative across forward forms even
            // if a faster ordinary kernel keeps the initial form incumbent.
            // Otherwise that improvement can hide a better combined layout.
            let cover_alternative = self.order.visited_forms.iter().filter(|&&v| v).count()
                < self.order.seeds.min(self.order.arithmetic_layouts);
            let backward_rank = if cover_alternative && self.seeds.len() > 1 {
                self.seeds[1].backward_rank
            } else {
                self.seeds.get(incumbent.0)?.backward_rank
            };
            self.order
                .form_order
                .sort_by_key(|&index| (self.seeds[index].backward_rank != backward_rank, index));
            let active = self.choices[incumbent.1];
            // Carry winning attention into arithmetic trials. The products-only
            // trial instead keeps the arithmetic winner; explicit combined
            // trials still compare each arithmetic alternative with products.
            // Qualification can queue private-storage alternatives later.
            // Keep that arithmetic coverage ahead of the products family too.
            self.order
                .priority_choices
                .make_contiguous()
                .sort_by_key(|&index| self.choices[index].forward_products);
            let arithmetic_pending = self.order.priority_choices.iter().any(|&index| {
                !self.choices[index].forward_products
                    && self
                        .order
                        .available((incumbent.0, self.order.candidate_choice(index)))
            });
            for index in &mut self.order.priority_choices {
                let arithmetic = self.choices[*index];
                self.order.rebased_choices.remove(index);
                let derivative_trial = (arithmetic.gradient_inputs.is_some()
                    && arithmetic.gradient_inputs != active.gradient_inputs)
                    || (arithmetic.backward_ept.is_some()
                        && arithmetic.backward_ept != active.backward_ept)
                    || arithmetic.f16_flash_tiles != active.f16_flash_tiles
                    || arithmetic.subgroup_tiles != active.subgroup_tiles;
                let products_only = arithmetic.forward_products
                    && arithmetic.matrix_inputs.is_none()
                    && arithmetic.winograd.is_none();
                // Do not bind this trial to an interim arithmetic winner.
                if products_only && arithmetic_pending {
                    continue;
                }
                let base = if derivative_trial || products_only {
                    active
                } else {
                    arithmetic
                };
                let neighbor = AxisChoice {
                    gradient_inputs: arithmetic.gradient_inputs.or(active.gradient_inputs),
                    backward_ept: arithmetic.backward_ept.or(active.backward_ept),
                    subgroup_tiles: arithmetic.subgroup_tiles.or(active.subgroup_tiles),
                    f16_flash_tiles: arithmetic.f16_flash_tiles || active.f16_flash_tiles,
                    forward_products: !arithmetic.f16_flash_tiles
                        && arithmetic.subgroup_tiles.is_none()
                        && (arithmetic.forward_products || active.forward_products),
                    attention: active.attention,
                    fuse_dispatches: active.fuse_dispatches,
                    f16_attention: if arithmetic.backward_ept.is_some()
                        || arithmetic.f16_flash_tiles
                        || arithmetic.subgroup_tiles.is_some()
                    {
                        false
                    } else {
                        active.f16_attention
                    },
                    ..base
                };
                if let Some(next) = self.choices.iter().position(|&choice| choice == neighbor) {
                    self.order.rebased_choices.insert(*index, next);
                }
            }
            // Try the other arithmetic family on top of a winning family
            // before spending the remaining deadline on attention layouts.
            let combined: Vec<_> = self
                .order
                .priority_choices
                .iter()
                .filter_map(|&index| {
                    let trial = self.choices[self.order.candidate_choice(index)];
                    if trial.forward_products != active.forward_products
                        || trial.matrix_inputs.is_some() == trial.winograd.is_some()
                    {
                        return None;
                    }
                    let choice = AxisChoice {
                        matrix_inputs: trial.matrix_inputs.or(active.matrix_inputs),
                        matrix_storage: if trial.matrix_inputs.is_some() {
                            trial.matrix_storage
                        } else {
                            active.matrix_storage
                        },
                        vector_convolutions_only: if trial.matrix_inputs.is_some() {
                            trial.vector_convolutions_only
                        } else {
                            active.vector_convolutions_only
                        },
                        winograd: trial.winograd.or(active.winograd),
                        ..active
                    };
                    self.choices.iter().position(|&c| c == choice)
                })
                .collect();
            for index in combined.into_iter().rev() {
                if self.order.available((incumbent.0, index)) && !self.order.has_priority(index) {
                    self.order.priority_choices.push_front(index);
                }
            }
            // Test the incumbent attention layout's other fusion setting, then
            // explore attention with its winning fusion setting. Expanding both
            // fusion settings for every losing layout spends the bounded search
            // before it reaches later attention layouts. Keep every combination
            // available after these coordinate neighbors.
            self.order.choice_order.sort_by_key(|&index| {
                let choice = self.choices[index];
                (
                    usize::from(
                        (
                            choice.matrix_inputs,
                            choice.vector_convolutions_only,
                            choice.matrix_storage,
                        ) != (
                            active.matrix_inputs,
                            active.vector_convolutions_only,
                            active.matrix_storage,
                        ),
                    ) + usize::from(choice.winograd != active.winograd)
                        + usize::from(choice.gradient_inputs != active.gradient_inputs),
                    choice.forward_products != active.forward_products,
                    choice.f16_attention != active.f16_attention,
                    choice.subgroup_tiles != active.subgroup_tiles,
                    choice.attention != active.attention,
                    choice.f16_flash_tiles != active.f16_flash_tiles,
                    choice.backward_ept != active.backward_ept,
                    choice.fuse_dispatches != active.fuse_dispatches,
                    index,
                )
            });
            if let Some(index) = self.order.priority_attention {
                // Compare cooperative backward with the winning dense precision,
                // after the independent arithmetic choices have had coverage.
                self.choices[index] = AxisChoice {
                    f16_attention: true,
                    backward_ept: None,
                    f16_flash_tiles: false,
                    subgroup_tiles: None,
                    ..active
                };
            }
            let (seed_index, choice_index) = self.order.next(incumbent)?;
            let seed = &self.seeds[seed_index];
            let AxisChoice {
                attention: (splits, flash),
                fuse_dispatches,
                matrix_inputs,
                matrix_storage,
                vector_convolutions_only,
                winograd,
                f16_attention,
                f16_flash_tiles,
                backward_ept,
                subgroup_tiles,
                gradient_inputs,
                forward_products,
            } = self.choices[choice_index];
            let mut plan = if splits == 0
                && !f16_attention
                && !f16_flash_tiles
                && subgroup_tiles.is_none()
                && backward_ept.is_none()
                && flash.is_none()
                && fuse_dispatches == seed.options.fuse_dispatches
            {
                seed.plan.clone()
            } else {
                let mut options = seed.options.clone();
                options.fuse_dispatches = fuse_dispatches;
                if f16_attention {
                    options.flash_backward_coop = true;
                    options.prefer_attention_grad =
                        Some(crate::kernels::attention_grad::Path::Cooperative(
                            crate::kernels::attention_grad::Operands::F16,
                        ));
                }
                if splits != 0 {
                    options.cached_attention_splits = Some(splits);
                }
                if let Some((ept, shape)) = flash {
                    options.knobs.flash_ept_cap = ept;
                    options.knobs.flash = shape;
                    options.flash_forward_coop = false;
                }
                if let Some(cap) = backward_ept {
                    options.knobs.flash_grad_q_ept_cap = cap;
                    options.knobs.flash_grad_kv_ept_cap = cap;
                }
                if f16_flash_tiles {
                    options.flash_f16_tiles = true;
                    options.flash_forward_coop = false;
                }
                if let Some(subgroups) = subgroup_tiles {
                    options.knobs.flash_subgroup_tiles = subgroups;
                    options.flash_forward_coop = false;
                }
                let scalar_backward =
                    backward_ept.is_some() || f16_flash_tiles || subgroup_tiles.is_some();
                if scalar_backward {
                    options.flash_backward_coop = false;
                    options.prefer_attention_grad =
                        Some(crate::kernels::attention_grad::Path::Flash);
                }
                let mut attention_graph;
                let graph = if (f16_attention || backward_ept.is_some() || f16_flash_tiles)
                    && self.replace_extracted_attention
                {
                    attention_graph = seed.graph.deep_clone();
                    for node in attention_graph.nodes_mut() {
                        if matches!(
                            node.op,
                            crate::graph::Op::MultiHeadAttnGradQ { .. }
                                | crate::graph::Op::MultiHeadAttnGradK { .. }
                                | crate::graph::Op::MultiHeadAttnGradV { .. }
                        ) {
                            if f16_attention || backward_ept.is_some() {
                                node.attention_ept_cap = None;
                            }
                            node.attention_impl = None;
                        }
                    }
                    &attention_graph
                } else {
                    &seed.graph
                };
                let plan = compile::compile_with_caps(
                    graph,
                    &options,
                    self.caps,
                    self.shared_memory_bytes,
                );
                if subgroup_tiles.is_some()
                    && !has_flash_reduction(&plan, self.attention_subgroup_size)
                {
                    continue;
                }
                let has_flash_backward = plan.dispatches.iter().any(|d| {
                    matches!(d.shader, compile::ShaderEntry::AttentionGrad(k)
                        if k.path == crate::kernels::attention_grad::Path::Flash)
                });
                let has_flash_forward = plan
                    .dispatches
                    .iter()
                    .any(|d| d.shader == compile::ShaderEntry::FlashAttention);
                let has_f16_flash = plan.dispatches.iter().any(|d| d.attention_f16_tiles);
                if (backward_ept.is_some() && !has_flash_backward)
                    || (subgroup_tiles.is_some() && !has_flash_backward && !has_flash_forward)
                    || (f16_flash_tiles && !has_f16_flash)
                {
                    continue;
                }
                if plan == seed.plan
                    || (f16_attention
                        && !plan.dispatches.iter().any(|d| {
                            matches!(d.shader, compile::ShaderEntry::AttentionGrad(kernel)
                            if kernel.path == crate::kernels::attention_grad::Path::Cooperative(
                                crate::kernels::attention_grad::Operands::F16))
                        }))
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
            if forward_products
                && fill_attention_products(
                    &mut plan,
                    self.caps,
                    self.shared_memory_bytes,
                    self.attention_subgroup_size,
                ) == 0
            {
                continue;
            }
            if let Some(multiply) = matrix_inputs {
                if fill_matrix_candidates_scoped(
                    &mut plan,
                    multiply,
                    self.replace_extracted_matrices,
                    vector_convolutions_only,
                ) == 0
                {
                    continue;
                }
            }
            if let Some(multiply) = gradient_inputs {
                if fill_gradient_candidates(&mut plan, multiply, self.replace_extracted_matrices)
                    == 0
                {
                    continue;
                }
            }
            if matrix_storage && storage::fill(&mut plan) == 0 {
                continue;
            }
            if let Some(storage) = winograd {
                let (shape, subgroup) = self.matrix_shapes[&storage];
                if winograd::fill(&mut plan, storage, shape, subgroup) == 0 {
                    continue;
                }
            }
            let mut description = format!(
                "{}, dispatch_fusion={fuse_dispatches}, attention_splits={splits}, flash={flash:?}",
                seed.description
            );
            if f16_attention {
                description.push_str(", f16_attention=true");
            }
            if let Some(cap) = backward_ept {
                description.push_str(&format!(", backward_ept={cap}"));
            }
            if let Some(subgroups) = subgroup_tiles {
                description.push_str(&format!(", subgroup_tiles={subgroups}"));
            }
            if f16_flash_tiles {
                description.push_str(", f16_flash_tiles=true");
            }
            if forward_products {
                description.push_str(", forward_products=true");
            }
            if let Some(dtype) = matrix_inputs {
                description.push_str(&format!(", matrix_inputs={dtype:?}"));
                if matrix_storage {
                    description.push_str(&format!(", matrix_storage={dtype:?}"));
                }
                if vector_convolutions_only {
                    description.push_str(", vector_convolutions_only=true");
                }
            }
            if let Some(dtype) = gradient_inputs {
                description.push_str(&format!(", gradient_inputs={dtype:?}"));
            }
            if let Some(dtype) = winograd {
                description.push_str(&format!(", winograd={dtype:?}"));
            }
            return Some(measure::Program {
                description,
                plan,
                submission_chunks: None,
                origin: Some(measure::Origin {
                    form: seed_index,
                    choice: choice_index,
                }),
            });
        }
    }

    fn may_remain(&self) -> bool {
        !self.queued_splits || !self.split_plans.is_empty() || !self.order.exhausted()
    }
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
            origin: None,
        });
    }
    programs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix_frontier() -> Implementations {
        let mut graph = Graph::new();
        let x = graph.input("x", &[32, 32]);
        let w = graph.parameter("w", &[32, 32]);
        let y = graph.matmul(x, w);
        graph.set_outputs(vec![y]);
        graph_frontier(graph)
    }

    fn gradient_frontier() -> Implementations {
        let mut graph = Graph::new();
        let x = graph.input("x", &[32, 64]);
        let w = graph.parameter("w", &[64, 32]);
        let y = graph.matmul(x, w);
        let loss = graph.mean_all(y);
        graph.set_outputs(vec![loss]);
        graph_frontier(crate::autodiff::differentiate(&graph))
    }

    #[test]
    fn queued_reduction_uses_the_latest_gradient_winner() {
        use measure::Frontier as _;
        let mut frontier = gradient_frontier();
        for index in 1..3 {
            let seed = &frontier.seeds[0];
            frontier.seeds.push(Seed {
                graph: seed.graph.clone(),
                options: seed.options.clone(),
                plan: seed.plan.clone(),
                description: format!("graph={index}"),
                backward_rank: 0,
            });
        }
        let ordinary = frontier.choices[0];
        let bf16 = frontier.choices.len();
        frontier.choices.push(AxisChoice {
            gradient_inputs: Some(crate::DType::BF16),
            ..ordinary
        });
        let f16 = frontier.choices.len();
        frontier.choices.push(AxisChoice {
            gradient_inputs: Some(crate::DType::F16),
            ..ordinary
        });
        let reduction = frontier.choices.len();
        frontier.choices.push(AxisChoice {
            subgroup_tiles: Some(true),
            ..ordinary
        });
        for dtype in [crate::DType::BF16, crate::DType::F16] {
            frontier.choices.push(AxisChoice {
                gradient_inputs: Some(dtype),
                subgroup_tiles: Some(true),
                ..ordinary
            });
        }
        frontier.order = CoordinateOrder::new(frontier.seeds.len(), frontier.choices.len());
        frontier.order.priority_choices.push_back(reduction);
        // Two arithmetic winners arrive while initial graph coverage keeps the
        // reduction trial pending. It must use F16, not its earlier BF16 snapshot.
        for (choice, dtype) in [(bf16, crate::DType::BF16), (f16, crate::DType::F16)] {
            frontier
                .next(Some(measure::Origin { form: 0, choice }))
                .unwrap();
            let queued = frontier.order.priority_choices.front().unwrap();
            assert_eq!(
                frontier.choices[frontier.order.candidate_choice(*queued)].gradient_inputs,
                Some(dtype)
            );
        }
    }

    #[test]
    fn measured_f16_staging_can_replace_cooperative_forward_attention() {
        use measure::Frontier as _;
        let mut graph = Graph::new();
        let q = graph.input("q", &[128, 64]);
        let k = graph.input("k", &[128, 64]);
        let v = graph.input("v", &[128, 64]);
        let y = graph.full_attention(q, k, v, 1, 1, 64);
        graph.set_outputs(vec![y]);
        let mut frontier = graph_frontier(graph);
        let products = frontier.choices.len();
        frontier.choices.push(AxisChoice {
            forward_products: true,
            ..frontier.choices[0]
        });
        frontier.enable_flash_tiles();
        let _baseline = frontier.next(None).unwrap();
        let candidate = frontier
            .next(Some(measure::Origin {
                form: 0,
                choice: products,
            }))
            .unwrap();
        assert!(choice_of(&frontier, &candidate).f16_flash_tiles);
        assert!(!choice_of(&frontier, &candidate).forward_products);
        assert!(
            candidate
                .plan
                .dispatches
                .iter()
                .any(|d| d.shader == compile::ShaderEntry::FlashAttention && d.attention_f16_tiles)
        );
    }

    #[test]
    fn wide_backward_reduction_keeps_extracted_ept() {
        use measure::Frontier as _;
        let mut graph = Graph::new();
        let q = graph.parameter("q", &[50, 960]);
        let k = graph.parameter("k", &[50, 320]);
        let v = graph.parameter("v", &[50, 320]);
        let y = graph.causal_attention(q, k, v, 15, 5, 64);
        let loss = graph.mean_all(y);
        graph.set_outputs(vec![loss]);
        let mut graph = crate::autodiff::differentiate(&graph);
        for node in graph.nodes_mut() {
            if matches!(
                node.op,
                crate::graph::Op::MultiHeadAttnGradQ { .. }
                    | crate::graph::Op::MultiHeadAttnGradK { .. }
                    | crate::graph::Op::MultiHeadAttnGradV { .. }
            ) {
                node.attention_ept_cap = Some(4);
            }
        }
        let mut frontier = graph_frontier(graph);
        frontier.replace_extracted_attention = true;
        frontier.attention_subgroup_size = 32;
        frontier.enable_flash_reductions();
        frontier.enable_flash_backward(true);
        let baseline = frontier.next(None).unwrap();
        let alternative = frontier.next(baseline.origin).unwrap();
        assert_eq!(
            choice_of(&frontier, &alternative).subgroup_tiles,
            Some(true)
        );
        assert!(alternative.plan.knobs.flash_subgroup_tiles);
        assert!(
            alternative
                .plan
                .dispatches
                .iter()
                .any(|d| { matches!(d.kernel, compile::Kernel::AttentionBackward { ept_cap: 4 }) })
        );
        assert_eq!(
            alternative.plan.knobs.flash_grad_q_ept_cap,
            baseline.plan.knobs.flash_grad_q_ept_cap
        );
        let precision = (0..8)
            .find_map(|_| {
                let candidate = frontier.next(baseline.origin)?;
                choice_of(&frontier, &candidate)
                    .f16_flash_tiles
                    .then_some(candidate)
            })
            .expect("backward precision must preserve the extracted layout");
        assert!(precision.plan.dispatches.iter().any(|d| {
            d.attention_f16_tiles
                && matches!(d.kernel, compile::Kernel::AttentionBackward { ept_cap: 4 })
        }));
    }

    #[test]
    fn flash_backward_shape_and_precision_are_independent_early_choices() {
        use measure::Frontier as _;
        let mut graph = Graph::new();
        let q = graph.parameter("q", &[128, 64]);
        let k = graph.parameter("k", &[128, 64]);
        let v = graph.parameter("v", &[128, 64]);
        let y = graph.multi_head_attn(q, k, v, 1, 1, 64, false);
        let loss = graph.mean_all(y);
        graph.set_outputs(vec![loss]);
        let mut frontier = graph_frontier(crate::autodiff::differentiate(&graph));
        // Start from a wider backward layout so the first shape neighbor
        // exercises an actual change, including from a cooperative winner.
        let seed = &mut frontier.seeds[0];
        seed.options.knobs.flash_grad_q_ept_cap = 32;
        seed.options.knobs.flash_grad_kv_ept_cap = 32;
        seed.plan = compile::compile_with_caps(&seed.graph, &seed.options, frontier.caps, 65536);
        let cooperative_index = frontier.choices.len();
        frontier.choices.push(AxisChoice {
            f16_attention: true,
            ..frontier.choices[0]
        });
        frontier.enable_flash_backward(true);
        let baseline = frontier.next(None).unwrap();
        let cooperative = measure::Origin {
            form: 0,
            choice: cooperative_index,
        };
        let shaped = frontier.next(Some(cooperative)).unwrap();
        let choice = choice_of(&frontier, &shaped);
        assert_eq!(choice.backward_ept, Some(16));
        assert!(!choice.f16_flash_tiles);
        assert!(!choice.f16_attention);
        assert!(shaped.plan.dispatches.iter().any(|d| matches!(d.shader, compile::ShaderEntry::AttentionGrad(k) if k.path == crate::kernels::attention_grad::Path::Flash)));
        assert_eq!(
            shaped.plan.knobs.flash_ept_cap,
            baseline.plan.knobs.flash_ept_cap
        );
        assert_eq!(shaped.plan.knobs.flash_grad_q_ept_cap, 16);
        assert_eq!(shaped.plan.knobs.flash_grad_kv_ept_cap, 16);
        assert_eq!(
            choice.fuse_dispatches,
            choice_of(&frontier, &baseline).fuse_dispatches
        );
        let mixed = (0..32)
            .find_map(|_| {
                let candidate = frontier.next(shaped.origin)?;
                choice_of(&frontier, &candidate)
                    .f16_flash_tiles
                    .then_some(candidate)
            })
            .expect("the precision neighbor must stay reachable");
        let choice = choice_of(&frontier, &mixed);
        assert_eq!(choice.backward_ept, Some(16));
        assert_eq!(choice.attention, choice_of(&frontier, &shaped).attention);
        assert_eq!(
            choice.fuse_dispatches,
            choice_of(&frontier, &shaped).fuse_dispatches
        );

        // A new forward winner must revisit backward precision before
        // exhausting all other forward layouts with the old precision.
        let forward_index = frontier
            .choices
            .iter()
            .position(|c| {
                c.attention.1.is_some()
                    && !c.f16_attention
                    && !c.f16_flash_tiles
                    && c.backward_ept.is_none()
            })
            .unwrap();
        frontier.order.priority_choices.clear();
        frontier.queued_splits = true;
        frontier.order.tried.insert((0, forward_index));
        let forward = measure::Origin {
            form: 0,
            choice: forward_index,
        };
        let neighbor = (0..8)
            .find_map(|_| {
                let candidate = frontier.next(Some(forward))?;
                let choice = choice_of(&frontier, &candidate);
                choice.f16_flash_tiles.then_some(choice)
            })
            .expect("revisit backward precision on the forward winner");
        assert_eq!(
            neighbor.attention,
            frontier.choices[forward_index].attention
        );
    }

    #[test]
    fn derivative_arithmetic_keeps_pins_storage_and_unmeasured_nn() {
        use crate::DType::{BF16, F16};
        let baseline = gradient_frontier().seeds.remove(0).plan;
        let products: Vec<_> = baseline
            .dispatches
            .iter()
            .enumerate()
            .filter_map(|(i, d)| (d.is_derivative && d.shader.is_matmul()).then_some(i))
            .collect();
        assert_eq!(products.len(), 2);
        for dtype in [F16, BF16] {
            let mut candidate = baseline.clone();
            assert_eq!(fill_gradient_candidates(&mut candidate, dtype, true), 2);
            for (index, d) in candidate.dispatches.iter().enumerate() {
                if products.contains(&index) {
                    assert!(d.is_derivative && d.full_precision() && !d.requires_full_precision);
                    assert!(
                        matches!(d.kernel, compile::Kernel::CooperativeTyped { types, .. }
                        if types.multiply == dtype && types.a == crate::DType::F32
                        && types.b == crate::DType::F32 && types.output == crate::DType::F32)
                    );
                } else {
                    assert_eq!(d, &baseline.dispatches[index]);
                }
            }
            let mut protected = baseline.clone();
            for &index in &products {
                protected.dispatches[index].requires_full_precision = true;
            }
            assert_eq!(fill_gradient_candidates(&mut protected, dtype, true), 0);
            let mut nn = baseline.clone();
            for &index in &products {
                nn.dispatches[index].shader = compile::ShaderEntry::MatMul;
            }
            assert_eq!(fill_gradient_candidates(&mut nn, dtype, true), 0);
            let mut locked = baseline.clone();
            for &index in &products {
                locked.dispatches[index].schedule_locked = true;
            }
            assert_eq!(fill_gradient_candidates(&mut locked, dtype, false), 0);
            assert_eq!(fill_gradient_candidates(&mut locked, dtype, true), 2);
        }
    }

    #[test]
    fn derivative_precision_is_independent_of_forward_precision_and_policy() {
        use crate::DType::{BF16, F16};
        use measure::Frontier as _;
        let mut frontier = gradient_frontier();
        let caps = matrix_caps(&[[8, 16, 16]], &[[8, 16, 16]]);
        frontier.enable_matrix_inputs(&[], &caps);
        frontier.enable_gradient_inputs(&[]);
        assert!(frontier.choices.iter().all(|c| c.gradient_inputs.is_none()));
        frontier.enable_matrix_inputs(&[BF16, F16], &caps);
        frontier.enable_gradient_inputs(&[BF16, F16]);
        let derivative_order: Vec<_> = frontier
            .order
            .priority_choices
            .iter()
            .filter_map(|&i| frontier.choices[i].gradient_inputs)
            .collect();
        assert_eq!(derivative_order, [BF16, F16]);
        let f16 = frontier
            .choices
            .iter()
            .position(|c| c.gradient_inputs == Some(F16) && c.matrix_inputs.is_none())
            .unwrap();
        let bf16 = frontier
            .choices
            .iter()
            .position(|c| {
                *c == AxisChoice {
                    gradient_inputs: Some(BF16),
                    ..frontier.choices[f16]
                }
            })
            .unwrap();
        frontier.defer_bf16 = true;
        frontier.observe(
            measure::Origin {
                form: 0,
                choice: f16,
            },
            true,
        );
        assert!(!frontier.order.deferred.contains(&(0, bf16)));
        let baseline = frontier.next(None).unwrap();
        assert!(choice_of(&frontier, &baseline).gradient_inputs.is_none());
        assert!(
            baseline
                .plan
                .dispatches
                .iter()
                .filter(|d| d.is_derivative)
                .all(|d| !matches!(d.kernel, compile::Kernel::CooperativeTyped { .. }))
        );
        // The order is exhaustive, so the incumbent does not change what is reached.
        let mut reached = collections::HashSet::new();
        while let Some(program) = frontier.next(baseline.origin) {
            let choice = choice_of(&frontier, &program);
            reached.insert((choice.matrix_inputs, choice.gradient_inputs));
        }
        for forward in [None, Some(F16), Some(BF16)] {
            for derivative in [Some(F16), Some(BF16)] {
                assert!(
                    reached.contains(&(forward, derivative)),
                    "missing {forward:?}/{derivative:?}"
                );
            }
        }
    }

    #[test]
    fn attention_products_follow_arithmetic_and_keep_explicit_pins() {
        use measure::Frontier as _;
        let mut graph = Graph::new();
        let q = graph.input("q", &[33, 144]);
        let weights = graph.parameter("weights", &[144, 144]);
        let q = graph.matmul(q, weights);
        let q = graph.matmul(q, weights);
        let k = graph.input("k", &[19, 48]);
        let v = graph.input("v", &[19, 48]);
        let y = graph.multi_head_attn(q, k, v, 3, 1, 48, true);
        graph.set_outputs(vec![y]);
        let caps = crate::codegen::CoopCaps {
            f16_attention_rows: 8,
            ..Default::default()
        };
        let make_frontier = |shape| {
            let mut frontier = graph_frontier(graph.deep_clone());
            frontier.caps = caps;
            frontier.allow_matrix_storage = true;
            let mut matrix_caps = blade_graphics::Capabilities {
                shader_float16: true,
                shader_bfloat16: true,
                ..Default::default()
            };
            matrix_caps.cooperative_matrix.f16_f32_shapes.push(shape);
            matrix_caps.cooperative_matrix.bf16_f32_shapes.push(shape);
            frontier.enable_matrix_inputs(&[crate::DType::F16, crate::DType::BF16], &matrix_caps);
            frontier.enable_forward_products(32);
            frontier
        };
        let mut frontier = make_frontier([16, 16, 16]);
        let baseline = frontier.next(None).unwrap();
        assert!(
            baseline
                .plan
                .dispatches
                .iter()
                .all(|d| d.shader != compile::ShaderEntry::FlashAttentionProducts)
        );
        let arithmetic = frontier.next(baseline.origin).unwrap();
        assert!(arithmetic.description.contains("matrix_inputs=F16"));
        assert!(!arithmetic.description.contains("forward_products=true"));
        frontier.observe(arithmetic.origin.unwrap(), true);
        let other = frontier.next(arithmetic.origin).unwrap();
        assert!(other.description.contains("matrix_inputs=BF16"));
        assert!(!other.description.contains("forward_products=true"));
        frontier.observe(other.origin.unwrap(), true);
        let stored_f16 = frontier.next(other.origin).unwrap();
        assert!(stored_f16.description.contains("matrix_storage=F16"));
        let stored_bf16 = frontier.next(other.origin).unwrap();
        assert!(stored_bf16.description.contains("matrix_storage=BF16"));
        // The final arithmetic/storage winner must carry into products.
        let candidate = frontier.next(stored_bf16.origin).unwrap();
        assert!(candidate.description.contains("matrix_storage=BF16"));
        assert!(candidate.description.contains("matrix_inputs=BF16"));
        assert!(
            candidate
                .plan
                .dispatches
                .iter()
                .any(|d| d.shader == compile::ShaderEntry::FlashAttentionProducts
                    && d.workgroups == [3, 3, 1]
                    && d.label.starts_with("FlashAttentionProducts"))
        );
        let combined = frontier.next(candidate.origin).unwrap();
        assert!(combined.description.contains("matrix_inputs=F16"));
        assert!(combined.description.contains("forward_products=true"));
        let mut pinned = baseline.plan;
        for d in &mut pinned.dispatches {
            d.requires_full_precision = true;
        }
        assert_eq!(fill_attention_products(&mut pinned, caps, 65536, 32), 0);
        for width in [1, 5] {
            assert!(!crate::kernels::attention::admits_products(
                33, width, 3, caps, 65536, 32
            ));
        }
        // Intel can defer BF16 after an F16 pass. A deferred ordinary entry
        // must not send the products-only trial back to unreduced arithmetic.
        let mut deferred = make_frontier([8, 16, 16]);
        let baseline = deferred.next(None).unwrap();
        let arithmetic = deferred.next(baseline.origin).unwrap();
        deferred.observe(arithmetic.origin.unwrap(), true);
        let stored = deferred.next(arithmetic.origin).unwrap();
        assert!(stored.description.contains("matrix_storage=F16"));
        let candidate = deferred.next(stored.origin).unwrap();
        assert!(candidate.description.contains("matrix_inputs=F16"));
        assert!(candidate.description.contains("forward_products=true"));
    }

    #[test]
    fn search_rejects_attention_products_when_values_exceed_f16_range() {
        use std::cell::Cell;
        let gpu = crate::reference::gpu::shared_context();
        let native = runtime::auto_tune(&gpu, 0).coop_caps;
        if !crate::kernels::attention::admits_products(
            17,
            64,
            1,
            native,
            gpu.capabilities().max_compute_shared_memory_size,
            gpu.capabilities().subgroup_size,
        ) {
            return;
        }
        let mut graph = Graph::new();
        let q = graph.input("q", &[17, 64]);
        let k = graph.input("k", &[17, 64]);
        let v = graph.input("v", &[17, 64]);
        let y = graph.causal_attention(q, k, v, 1, 1, 64);
        graph.set_outputs(vec![y]);
        let tried = Cell::new(false);
        let (session, _) = build_measured(
            &graph,
            SessionConfig {
                gpu: Some(gpu),
                mode: crate::Mode::Inference,
                tune: false,
                cache: None,
                ..Default::default()
            },
            BuildSearchOptions {
                max_graphs: 1,
                max_programs: 2,
                max_time: Duration::from_secs(10),
                warmup_runs: 1,
                warmup_time: Duration::ZERO,
                tuning: TuneOptions {
                    max_classes: 0,
                    ..Default::default()
                },
                ..Default::default()
            },
            |s, _| {
                s.set_input("q", &[0.01; 17 * 64]);
                s.set_input("k", &[0.01; 17 * 64]);
                s.set_input("v", &[3e5; 17 * 64]);
                Ok(())
            },
            |s| {
                if s.plan()
                    .dispatches
                    .iter()
                    .any(|d| d.shader == compile::ShaderEntry::FlashAttentionProducts)
                {
                    tried.set(true);
                }
                // A convex average of constant V has this exact finite value.
                if s.read_output(17 * 64)
                    .iter()
                    .all(|x| x.is_finite() && (*x - 3e5).abs() < 1.0)
                {
                    Ok(())
                } else {
                    Err("attention value overflow".into())
                }
            },
        )
        .unwrap();
        assert!(
            tried.get(),
            "the unsafe candidate must actually be qualified"
        );
        assert!(
            session
                .plan()
                .dispatches
                .iter()
                .all(|d| d.shader != compile::ShaderEntry::FlashAttentionProducts)
        );
    }

    #[test]
    fn matrix_arithmetic_order_preserves_policy_and_bf16_range_fallback() {
        use crate::{CoopPolicy, DType};
        for f16_tile in [0, 8, 16] {
            let caps = crate::codegen::CoopCaps {
                f16_tile,
                ..Default::default()
            };
            let expected = if f16_tile == 16 {
                [DType::BF16, DType::F16]
            } else {
                [DType::F16, DType::BF16]
            };
            assert_eq!(matrix_input_order(CoopPolicy::Auto, caps), expected);
            assert_eq!(matrix_input_order(CoopPolicy::AllowF16, caps), [DType::F16]);
            for policy in [CoopPolicy::NativeF32, CoopPolicy::Disabled] {
                assert!(matrix_input_order(policy, caps).is_empty());
            }
        }
    }

    #[test]
    fn initial_graph_cover_reaches_alternative_layouts_when_baseline_keeps_winning() {
        use measure::Frontier as _;
        let mut frontier = matrix_frontier();
        for index in 1..16 {
            let seed = &frontier.seeds[0];
            frontier.seeds.push(Seed {
                graph: seed.graph.clone(),
                options: seed.options.clone(),
                plan: seed.plan.clone(),
                description: format!("graph={index}"),
                backward_rank: usize::from([1, 2, 5, 13].contains(&index)),
            });
        }
        frontier.order = CoordinateOrder::new(frontier.seeds.len(), frontier.choices.len());
        let baseline = frontier.next(None).unwrap();
        let mut forms = vec![0];
        while forms.len() < ARITHMETIC_LAYOUTS {
            let program = frontier.next(baseline.origin).unwrap();
            let form = program.origin.unwrap().form;
            if !forms.contains(&form) {
                forms.push(form);
            }
        }
        assert_eq!(forms, [0, 1, 2, 5, 13]);
    }

    #[test]
    fn eight_wide_arithmetic_precedes_layout_and_weight_split_exploration() {
        use measure::Frontier as _;
        let mut graph = Graph::new();
        let x = graph.input("x", &[16 * 8 * 8]);
        let w = graph.parameter("w", &[32 * 16 * 9]);
        let conv = graph.conv2d(x, w, 1, 16, 8, 8, 32, 3, 3, 1, 1);
        let loss = graph.mean_all(conv);
        graph.set_outputs(vec![loss]);
        let mut frontier = graph_frontier(crate::autodiff::differentiate(&graph));
        assert!(!low_occupancy_weight_splits(frontier.seeds.first(), 512 << 20).is_empty());
        for index in 1..8 {
            let seed = &frontier.seeds[0];
            frontier.seeds.push(Seed {
                graph: seed.graph.clone(),
                options: seed.options.clone(),
                plan: seed.plan.clone(),
                description: format!("graph={index}"),
                backward_rank: 0,
            });
        }
        frontier.order = CoordinateOrder::new(frontier.seeds.len(), frontier.choices.len());
        let caps = matrix_caps(&[[8, 8, 8]], &[[8, 8, 8]]);
        frontier.enable_matrix_inputs(&[crate::DType::F16, crate::DType::BF16], &caps);
        let first = frontier.next(None).unwrap();
        assert!(choice_of(&frontier, &first).matrix_inputs.is_none());
        let incumbent = first.origin;
        assert_eq!(frontier.early_programs_remaining(incumbent), 2);
        let f16 = frontier.next(incumbent).unwrap();
        assert_eq!(f16.origin.unwrap().form, 0);
        assert_eq!(
            choice_of(&frontier, &f16).matrix_inputs,
            Some(crate::DType::F16)
        );
        assert_eq!(frontier.early_programs_remaining(incumbent), 1);
        let bf16 = frontier.next(incumbent).unwrap();
        assert_eq!(
            choice_of(&frontier, &bf16).matrix_inputs,
            Some(crate::DType::BF16)
        );
        assert_eq!(frontier.early_programs_remaining(incumbent), 0);
        let split = frontier.next(incumbent).unwrap();
        assert!(split.description.contains("conv_dw_splits=4"));
        // Early arithmetic must not remove ordinary graph/layout choices.
        assert!(
            std::iter::from_fn(|| frontier.next(incumbent))
                .any(|p| p.description.starts_with("graph=1,"))
        );
    }

    fn graph_frontier(graph: Graph) -> Implementations {
        let options = compile::CompileOptions::default();
        let caps = crate::codegen::CoopCaps::default();
        let plan = compile::compile_with_caps(&graph, &options, caps, 65536);
        implementations(
            vec![Seed {
                graph: std::sync::Arc::new(graph),
                options,
                plan,
                description: "graph=0".into(),
                backward_rank: 0,
            }],
            caps,
            65536,
            512 << 20,
        )
    }

    /// Capabilities advertising these F16 and BF16 cooperative shapes; an
    /// empty list leaves that type unsupported.
    fn matrix_caps(f16: &[[u32; 3]], bf16: &[[u32; 3]]) -> blade_graphics::Capabilities {
        let mut caps = blade_graphics::Capabilities {
            shader_float16: !f16.is_empty(),
            shader_bfloat16: !bf16.is_empty(),
            subgroup_size: 32,
            ..Default::default()
        };
        caps.cooperative_matrix.f16_f32_shapes = f16.to_vec();
        caps.cooperative_matrix.bf16_f32_shapes = bf16.to_vec();
        caps
    }

    /// The physical choice a frontier program was lowered with.
    fn choice_of(frontier: &Implementations, program: &measure::Program) -> AxisChoice {
        frontier.choices[program.origin.unwrap().choice]
    }

    #[test]
    fn winograd_and_dense_precision_choices_compose_and_qualify_independently() {
        use crate::DType::{BF16, F16};
        use measure::Frontier as _;
        let mut graph = Graph::new();
        let x = graph.input("x", &[64 * 8 * 8]);
        let w = graph.parameter("w", &[64 * 64 * 9]);
        let conv = graph.conv2d(x, w, 1, 64, 8, 8, 64, 3, 3, 1, 1);
        let a = graph.input("a", &[32, 32]);
        let b = graph.parameter("b", &[32, 32]);
        let dense = graph.matmul(a, b);
        let vector_weights = graph.parameter("vector_weights", &[64 * 64]);
        let vector = graph.conv2d(x, vector_weights, 1, 64, 8, 8, 64, 1, 1, 1, 0);
        let gather = graph.conv2d(x, w, 1, 64, 8, 8, 64, 3, 3, 2, 1);
        let q = graph.input("q", &[33, 16]);
        let k = graph.input("k", &[33, 16]);
        let v = graph.input("v", &[33, 16]);
        let dy = graph.input("dy", &[33, 16]);
        let forward = graph.multi_head_attn(q, k, v, 1, 1, 16, true);
        let dq = graph.add_raw_node(
            crate::graph::Op::MultiHeadAttnGradQ {
                fwd_node: forward,
                num_heads: 1,
                num_kv_heads: 1,
                head_dim: 16,
                is_cross: true,
            },
            vec![dy, q, k, v],
            graph.node(q).ty.clone(),
        );
        graph.set_outputs(vec![conv, dense, vector, gather, dq]);
        crate::optimize::apply_winograd_conv_fusions(
            &mut graph,
            &mut Vec::new(),
            &Default::default(),
        );
        let frontier = graph_frontier(graph);
        let mut frontier = implementations(
            frontier.seeds,
            crate::CoopCaps {
                f16_attention_rows: 8,
                ..Default::default()
            },
            65536,
            512 << 20,
        );
        let caps = matrix_caps(&[[8, 16, 16]], &[[8, 16, 16]]);
        frontier.enable_matrix_inputs(&[BF16, F16], &caps);
        let mut expected = collections::HashSet::new();
        for dense in [None, Some(F16), Some(BF16)] {
            for winograd in [None, Some(F16), Some(BF16)] {
                for vector_only in [false, true]
                    .into_iter()
                    .take(1 + usize::from(dense.is_some()))
                {
                    for f16_attention in [false, true] {
                        expected.insert((dense, vector_only, winograd, f16_attention));
                    }
                }
            }
        }
        let mut incumbent = None;
        let mut reached = collections::HashSet::new();
        // Stop once every combination ran; the rest of the order adds nothing.
        while !expected.is_subset(&reached) {
            let Some(program) = frontier.next(incumbent) else {
                break;
            };
            let origin = program.origin.unwrap();
            let choice = frontier.choices[origin.choice];
            if choice.matrix_inputs.is_some()
                && !reached
                    .iter()
                    .any(|&(dense, _, _, _)| matches!(dense, Some(_)))
            {
                if let Some(incumbent) = incumbent {
                    let winner = frontier.choices[incumbent.choice];
                    assert_eq!(
                        choice.winograd, winner.winograd,
                        "retain the incumbent for the first dense trial"
                    );
                    assert_eq!(
                        choice.f16_attention, winner.f16_attention,
                        "retain winning attention arithmetic"
                    );
                }
            }
            reached.insert((
                choice.matrix_inputs,
                choice.vector_convolutions_only,
                choice.winograd,
                choice.f16_attention,
            ));
            assert_eq!(
                program.plan.dispatches.iter().any(|d| {
                    matches!(d.shader, compile::ShaderEntry::AttentionGrad(kernel)
                        if kernel.path == crate::kernels::attention_grad::Path::Cooperative(
                            crate::kernels::attention_grad::Operands::F16))
                }),
                choice.f16_attention,
            );
            let gather = program
                .plan
                .dispatches
                .iter()
                .find(|d| d.origin.contains(&gather))
                .unwrap();
            assert_eq!(
                matches!(gather.kernel, compile::Kernel::CooperativeTyped { .. }),
                choice.matrix_inputs.is_some() && !choice.vector_convolutions_only,
            );
            let dense = program.plan.dispatches.iter().find_map(|d| match d.kernel {
                compile::Kernel::CooperativeTyped { types, .. } => Some(types.multiply),
                _ => None,
            });
            let winograd = program.plan.dispatches.iter().find_map(|d| match d.kernel {
                compile::Kernel::WinogradMatmul { storage, .. } => Some(storage),
                _ => None,
            });
            assert_eq!((dense, winograd), (choice.matrix_inputs, choice.winograd));
            frontier.observe(origin, true);
            // A winning Winograd family must still reach both dense families,
            // even when qualified F16 defers the matching BF16 combination.
            if choice.f16_attention && (incumbent.is_none() || choice.winograd == Some(F16)) {
                incumbent = Some(origin);
            }
        }
        let missing: Vec<_> = expected.difference(&reached).collect();
        assert!(
            missing.is_empty(),
            "missing (dense, vector_only, winograd, f16_attention): {missing:?}"
        );
    }

    #[test]
    fn rectangular_matrix_candidates_require_advertised_arithmetic() {
        let mut frontier = matrix_frontier();
        let caps = matrix_caps(&[[8, 16, 16]], &[]);
        frontier.enable_matrix_inputs(&[crate::DType::F16, crate::DType::BF16], &caps);
        assert_eq!(
            measure::Frontier::early_programs_remaining(&frontier, None),
            0
        );
        let mut visited = Vec::new();
        while let Some(program) = measure::Frontier::next(&mut frontier, None) {
            visited.push(program);
        }
        let inputs: Vec<_> = visited
            .iter()
            .map(|p| choice_of(&frontier, p).matrix_inputs)
            .collect();
        assert_eq!(inputs[..2], [None, Some(crate::DType::F16)]);
        assert!(!inputs.contains(&Some(crate::DType::BF16)));
    }

    #[test]
    fn qualified_rectangular_f16_defers_only_its_matching_bf16_candidate() {
        use measure::Frontier as _;
        for (f16_rows, bf16_rows, qualified, form, deferred) in [
            (8, 8, true, 0, true),
            (8, 8, false, 0, false),
            (8, 8, true, 1, false),
            (16, 16, true, 0, false),
            (8, 16, true, 0, false),
        ] {
            let mut frontier = matrix_frontier();
            frontier.seeds.push(matrix_frontier().seeds.pop().unwrap());
            frontier.order = CoordinateOrder::new(2, frontier.choices.len());
            let mut caps = blade_graphics::Capabilities {
                shader_float16: true,
                shader_bfloat16: true,
                ..Default::default()
            };
            caps.cooperative_matrix
                .f16_f32_shapes
                .push([f16_rows, 16, 16]);
            caps.cooperative_matrix
                .bf16_f32_shapes
                .push([bf16_rows, 16, 16]);
            frontier.enable_matrix_inputs(&[crate::DType::F16, crate::DType::BF16], &caps);
            let f16 = std::iter::from_fn(|| frontier.next(None))
                .find(|p| p.description.contains("matrix_inputs=F16"))
                .unwrap();
            frontier.observe(f16.origin.unwrap(), qualified);
            // F16 need not win: qualification alone can defer the duplicate
            // arithmetic candidate, and only on the form that actually ran it.
            let incumbent = Some(measure::Origin { form, choice: 0 });
            let next = frontier.next(incumbent).unwrap();
            assert_eq!(
                next.description.contains("matrix_inputs=BF16"),
                !deferred,
                "F16 rows={f16_rows}, BF16 rows={bf16_rows}, qualified={qualified}, form={form}"
            );
            if deferred {
                assert!(
                    std::iter::from_fn(|| frontier.next(incumbent))
                        .any(|p| p.description.contains("matrix_inputs=BF16")),
                    "deferral must leave BF16 reachable"
                );
            }
        }
    }

    #[test]
    fn storage_candidates_follow_staged_arithmetic_and_respect_pins() {
        use measure::Frontier as _;
        for allow_storage in [false, true] {
            let mut graph = Graph::new();
            let x = graph.input("x", &[32, 32]);
            let w = graph.parameter("w", &[32, 32]);
            let x = graph.relu(x);
            let y = graph.matmul(x, w);
            graph.set_outputs(vec![y]);
            let mut frontier = graph_frontier(graph);
            frontier.allow_matrix_storage = allow_storage;
            let caps = matrix_caps(&[[16, 16, 16]], &[[16, 16, 16]]);
            frontier.enable_matrix_inputs(&[crate::DType::F16, crate::DType::BF16], &caps);
            let baseline = frontier.next(None).unwrap().origin;
            let staged = frontier.next(baseline).unwrap();
            assert!(staged.description.contains("matrix_inputs=F16"));
            assert!(!staged.description.contains("matrix_storage="));
            frontier.observe(staged.origin.unwrap(), true);
            let other = frontier.next(baseline).unwrap();
            assert!(other.description.contains("matrix_inputs=BF16"));
            assert!(!other.description.contains("matrix_storage="));
            let stored = std::iter::from_fn(|| frontier.next(baseline))
                .find(|p| p.description.contains("matrix_storage=F16"));
            assert_eq!(stored.is_some(), allow_storage);
            if let Some(stored) = stored {
                assert!(
                    stored
                        .plan
                        .dispatches
                        .iter()
                        .any(|d| d.pointwise_storage.f16_output)
                );
            }
        }
    }

    #[test]
    fn typed_candidates_replace_automatic_arithmetic_and_keep_f16_weight_storage() {
        use crate::{DType, compile::WeightFormat};
        let mut graph = Graph::new();
        let a = graph.input("a", &[64, 1024]);
        let b = graph.parameter_f16("b", &[1024, 128]);
        let y = graph.matmul(a, b);
        graph.set_outputs(vec![y]);
        let original = compile::compile(&graph);
        let dispatch = &original.dispatches[0];
        assert_eq!(dispatch.weight_format, WeightFormat::F16);
        let mut candidate = original.clone();
        assert_eq!(
            fill_matrix_candidates(&mut candidate, DType::F16, false),
            1,
            "a legacy F16 weight must not hide a legal typed mixed-storage alternative"
        );
        assert_eq!(
            candidate.dispatches[0].weight_format,
            WeightFormat::F32,
            "typed storage replaces opaque-format dispatch metadata"
        );
        assert!(matches!(candidate.dispatches[0].kernel,
            compile::Kernel::CooperativeTyped {types, splits:1}
                if types.a == DType::F32 && types.b == DType::F16
                    && types.multiply == DType::F16 && types.output == DType::F32));
        let mut bf16 = original.clone();
        assert_eq!(
            fill_matrix_candidates(&mut bf16, DType::BF16, false),
            0,
            "F16 storage cannot be reinterpreted as BF16"
        );
        assert_eq!(bf16, original);

        // Automatic cooperative arithmetic gives way to the typed candidate.
        let mut plan = matrix_frontier().seeds.remove(0).plan;
        let index = plan
            .dispatches
            .iter()
            .position(|d| d.mnk().is_some())
            .unwrap();
        plan.dispatches[index].kernel = compile::Kernel::Cooperative;
        assert_eq!(fill_matrix_candidates(&mut plan, DType::BF16, false), 1);
        assert!(matches!(plan.dispatches[index].kernel,
            compile::Kernel::CooperativeTyped { types, splits: 1 }
                if types.multiply == DType::BF16 && types.a == DType::F32
                    && types.b == DType::F32 && types.output == DType::F32));
    }

    #[test]
    fn arithmetic_alternatives_keep_extracted_partials_and_caller_pins() {
        let mut graph = Graph::new();
        let a = graph.input("a", &[32, 128]);
        let b = graph.parameter("b", &[128, 32]);
        let y = graph.matmul(a, b);
        graph.set_outputs(vec![y]);
        let mut plan = compile::compile(&graph);
        let shape = crate::codegen::ScalarMatmulShape {
            tile_size: 32,
            tile_n: 0,
            k_stage: 16,
            interleave_columns: false,
            unroll_k: false,
        };
        plan.split_matmul(0, shape, 4, 1 << 20).unwrap();
        plan.dispatches[0].schedule_locked = true;
        let original = plan.clone();
        assert_eq!(
            fill_matrix_candidates(&mut plan, crate::DType::F16, false),
            0
        );
        assert_eq!(plan, original);
        assert_eq!(
            fill_matrix_candidates(&mut plan, crate::DType::F16, true),
            1
        );
        assert!(matches!(
            plan.dispatches[0].kernel,
            compile::Kernel::CooperativeTyped { splits: 4, .. }
        ));
        assert!(!plan.dispatches[0].schedule_locked);
        assert_eq!(plan.dispatches[0].workgroups, [1, 1, 4]);
        assert_eq!(plan.buffers, original.buffers);
        assert_eq!(
            plan.dispatches[0].output_buffer,
            original.dispatches[0].output_buffer
        );
        assert_eq!(plan.dispatches[1..], original.dispatches[1..]);

        for protected in [true, false] {
            let mut plan = original.clone();
            if protected {
                plan.dispatches[0].requires_full_precision = true;
            } else {
                plan.dispatches[0].params[1] = 32; // Too few K blocks for four parts.
            }
            let before = plan.clone();
            assert_eq!(
                fill_matrix_candidates(&mut plan, crate::DType::BF16, true),
                0
            );
            assert_eq!(plan, before);
        }
        // Split typed numerics: tests/typed_matmul.rs and tests/bf16_split_k.rs.
    }

    #[test]
    fn matrix_candidates_preserve_dispatches_outside_portable_limits() {
        let frontier = matrix_frontier();
        for convolution in [false, true] {
            for (extent, valid) in [(0, false), (65_535 * 32, true), (65_535 * 32 + 1, false)] {
                let mut plan = frontier.seeds[0].plan.clone();
                let dispatch = plan
                    .dispatches
                    .iter_mut()
                    .find(|d| d.mnk().is_some())
                    .unwrap();
                if convolution {
                    dispatch.shader = compile::ShaderEntry::Conv2dGemm;
                    dispatch.params = vec![1, 1, 1, 1, 32, 1, 1, 1, 0, extent, 1];
                } else {
                    dispatch.params[0] = extent;
                }
                let original = plan.dispatches.clone();
                assert_eq!(
                    fill_matrix_candidates(&mut plan, crate::DType::BF16, false),
                    usize::from(valid)
                );
                if !valid {
                    assert_eq!(plan.dispatches, original);
                }
            }
        }
        let mut plan = frontier.seeds[0].plan.clone();
        let dispatch = plan
            .dispatches
            .iter_mut()
            .find(|d| d.mnk().is_some())
            .unwrap();
        dispatch.shader = compile::ShaderEntry::Conv2dGemm;
        dispatch.params = vec![1, 1, 1, 1, 32, 1, 1, 1, 0, u32::MAX, 2];
        let original = plan.dispatches.clone();
        assert_eq!(
            fill_matrix_candidates(&mut plan, crate::DType::BF16, false),
            0
        );
        assert_eq!(plan.dispatches, original);
    }

    #[test]
    fn auto_searches_bf16_beyond_fp16_range_without_relaxing_the_gate() {
        let gpu = crate::reference::gpu::shared_context();
        if !gpu.capabilities().shader_bfloat16 {
            return;
        }
        let expected_inputs = if matches!(runtime::auto_tune(&gpu, 0).coop_caps.f16_tile, 0 | 8) {
            vec![crate::DType::F16, crate::DType::BF16]
        } else {
            vec![crate::DType::BF16, crate::DType::F16]
        };
        let frontier = matrix_frontier();
        let graph = &frontier.seeds[0].graph;
        let (_, report) = build_measured(
            graph,
            SessionConfig {
                gpu: Some(gpu),
                mode: crate::Mode::Inference,
                ..Default::default()
            },
            BuildSearchOptions {
                max_graphs: 1,
                max_programs: 8,
                max_time: Duration::from_secs(15),
                warmup_runs: 1,
                warmup_time: Duration::ZERO,
                tuning: TuneOptions {
                    max_time: Duration::from_millis(100),
                    ..Default::default()
                },
                ..Default::default()
            },
            |s, _| {
                s.set_input("x", &[65536.0; 1024]);
                s.set_parameter("w", &[1.0 / 65536.0; 1024]);
                Ok(())
            },
            |s| {
                if s.read_output(1024)
                    .iter()
                    .all(|&x| x.is_finite() && (x - 32.0).abs() < 1e-5)
                {
                    Ok(())
                } else {
                    Err("matrix output differs from exact F32 reference".into())
                }
            },
        )
        .unwrap();
        assert_eq!(report.matrix_inputs, expected_inputs);
        if report.matrix_inputs.contains(&crate::DType::F16)
            && !report
                .unsupported_matrix_inputs
                .contains(&crate::DType::F16)
        {
            assert!(
                report.trials.iter().any(|t| {
                    t.description.contains("matrix_inputs=F16") && !t.outcome.qualified
                })
            );
        }
        assert!(
            report
                .trials
                .iter()
                .any(|t| t.description.contains("matrix_inputs=BF16") && t.outcome.qualified)
        );
    }

    #[test]
    fn checked_convolution_candidates_keep_batch_padding_and_partial_tiles() {
        let mut graph = Graph::new();
        let x = graph.input("x", &[2 * 3 * 7 * 9]);
        let w = graph.parameter("w", &[5 * 3 * 3 * 3]);
        let y = graph.conv2d(x, w, 2, 3, 7, 9, 5, 3, 3, 2, 1);
        graph.set_outputs(vec![y]);
        let mut plan = compile::compile(&graph);
        assert_eq!(
            fill_matrix_candidates(&mut plan, crate::DType::BF16, false),
            1
        );
        let gpu = crate::reference::gpu::shared_context();
        if !runtime::typed_matrix_supported(&gpu, crate::DType::BF16) {
            return;
        }
        let input: Vec<f32> = (0..2 * 3 * 7 * 9).map(|i| (i % 11) as f32 / 16.0).collect();
        let weights: Vec<f32> = (0..5 * 3 * 3 * 3).map(|i| (i % 5) as f32 / 16.0).collect();
        let expected: Vec<f32> = crate::reference::evaluate_outputs(
            &graph,
            crate::reference::Feeds::new()
                .set("x", &input)
                .set("w", &weights),
        )
        .unwrap()
        .remove(0)
        .data
        .into_iter()
        .map(|v| v as f32)
        .collect();
        let mut session = Session::with_context(plan, gpu);
        session.set_input("x", &input);
        session.set_parameter("w", &weights);
        session.step();
        session.wait();
        assert_eq!(
            session.read_output(expected.len()),
            expected,
            "checked convolution differs from exact CPU result"
        );
    }

    #[test]
    fn matrix_search_keeps_mixed_and_complete_convolution_choices() {
        use measure::Frontier as _;
        let mut graph = Graph::new();
        let x = graph.input("x", &[2 * 4 * 7 * 4]);
        let w1 = graph.parameter("w1", &[5 * 4]);
        let w3 = graph.parameter("w3", &[5 * 4 * 3 * 3]);
        let a = graph.conv2d(x, w1, 2, 4, 7, 4, 5, 1, 1, 1, 0);
        let b = graph.conv2d(x, w3, 2, 4, 7, 4, 5, 3, 3, 1, 1);
        graph.set_outputs(vec![a, b]);
        let caps = matrix_caps(&[[8, 16, 16]], &[]);
        for lock_gather in [false, true] {
            let mut frontier = matrix_frontier();
            frontier.seeds[0].plan = compile::compile(&graph);
            frontier.seeds[0].graph = std::sync::Arc::new(graph.deep_clone());
            if lock_gather {
                for d in &mut frontier.seeds[0].plan.dispatches {
                    if d.shader.is_conv2d_gemm()
                        && !crate::kernels::conv::Params::from(&*d).vectorized_forward()
                    {
                        d.schedule_locked = true;
                    }
                }
            }
            frontier.enable_matrix_inputs(&[crate::DType::F16], &caps);
            assert_eq!(
                frontier.choices.iter().any(|c| c.vector_convolutions_only),
                !lock_gather,
                "an ineligible gather must not create a duplicate mixed family"
            );
            if lock_gather {
                continue;
            }
            let mut counts = std::collections::HashSet::new();
            while let Some(program) = frontier.next(None) {
                counts.insert(
                    program
                        .plan
                        .dispatches
                        .iter()
                        .filter(|d| matches!(d.kernel, compile::Kernel::CooperativeTyped { .. }))
                        .count(),
                );
            }
            assert_eq!(counts, std::collections::HashSet::from([0, 1, 2]));
        }
    }

    #[test]
    fn vector_convolution_keeps_batch_channel_and_tile_tails() {
        let gpu = crate::reference::gpu::shared_context();
        for dtype in [crate::DType::F16, crate::DType::BF16] {
            if !runtime::typed_matrix_supported(&gpu, dtype) {
                continue;
            }
            for (ci, co, h, w) in [(12, 35, 7, 4), (4, 17, 5, 4)] {
                let mut graph = Graph::new();
                let x = graph.input("x", &[2 * ci * h * w]);
                let weight = graph.parameter("w", &[co * ci]);
                let output = graph.conv2d(
                    x, weight, 2, ci as u32, h as u32, w as u32, co as u32, 1, 1, 1, 0,
                );
                graph.set_outputs(vec![output]);
                let input: Vec<f32> = (0..2 * ci * h * w)
                    .map(|i| ((i % 17) as f32 - 8.0) / 32.0)
                    .collect();
                let weights: Vec<f32> = (0..co * ci)
                    .map(|i| ((i % 11) as f32 - 5.0) / 16.0)
                    .collect();
                let expected: Vec<f32> = crate::reference::evaluate_outputs(
                    &graph,
                    crate::reference::Feeds::new()
                        .set("x", &input)
                        .set("w", &weights),
                )
                .unwrap()
                .remove(0)
                .data
                .into_iter()
                .map(|v| v as f32)
                .collect();
                let mut plan = compile::compile(&graph);
                assert_eq!(
                    fill_matrix_candidates_scoped(&mut plan, dtype, false, true),
                    1
                );
                let mut session = Session::with_context(plan, gpu.clone());
                session.set_input("x", &input);
                session.set_parameter("w", &weights);
                for scale in [1.0, 2.0f32.powi(-15)] {
                    let scaled: Vec<_> = input.iter().map(|x| x * scale).collect();
                    session.set_input("x", &scaled);
                    session.step();
                    session.wait();
                    let want: Vec<_> = expected.iter().map(|x| x * scale).collect();
                    assert_eq!(
                        session.read_output(want.len()),
                        want,
                        "{dtype:?} {ci} {co} {h} {w} {scale}"
                    );
                }
            }
        }
    }

    #[test]
    fn explicit_policies_search_only_their_allowed_arithmetic() {
        let gpu = crate::reference::gpu::shared_context();
        let frontier = matrix_frontier();
        // Policies without typed arithmetic have an empty `matrix_input_order`.
        let coop = crate::CoopPolicy::AllowF16;
        let (_, report) = build_measured(
            &frontier.seeds[0].graph,
            SessionConfig {
                gpu: Some(gpu.clone()),
                mode: crate::Mode::Inference,
                runtime: crate::SessionOptions {
                    coop,
                    ..Default::default()
                },
                ..Default::default()
            },
            BuildSearchOptions {
                max_programs: 8,
                max_graphs: 1,
                warmup_runs: 1,
                warmup_time: Duration::ZERO,
                tuning: TuneOptions {
                    max_time: Duration::from_millis(100),
                    ..Default::default()
                },
                ..Default::default()
            },
            |s, _| {
                s.set_input("x", &[0.25; 1024]);
                s.set_parameter("w", &[0.5; 1024]);
                Ok(())
            },
            |s| {
                assert_eq!(s.read_output(1024), [4.0; 1024]);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(report.matrix_inputs, [crate::DType::F16]);
        if runtime::typed_matrix_supported(&gpu, crate::DType::F16) {
            assert!(report.unsupported_matrix_inputs.is_empty());
            assert!(
                report.trials.iter().any(|t| {
                    t.description.contains("matrix_inputs=F16") && t.outcome.qualified
                })
            );
        } else {
            assert_eq!(report.unsupported_matrix_inputs, [crate::DType::F16]);
            assert!(
                report
                    .trials
                    .iter()
                    .all(|t| !t.description.contains("matrix_inputs="))
            );
        }
        assert!(
            report
                .trials
                .iter()
                .all(|t| !t.description.contains("matrix_inputs=BF16"))
        );
    }

    #[test]
    fn qualification_cannot_be_disabled() {
        let result = build_measured(
            &Graph::new(),
            SessionConfig::default(),
            BuildSearchOptions {
                qualification_runs: 0,
                ..Default::default()
            },
            |_, _| panic!("invalid options must fail before initialization"),
            |_| panic!("invalid options must fail before execution"),
        );
        assert!(
            result
                .err()
                .unwrap()
                .contains("positive qualification runs")
        );
    }

    #[test]
    fn fusion_alternatives_lead_the_physical_choices() {
        let frontier = matrix_frontier();
        let unfused = AxisChoice {
            fuse_dispatches: false,
            ..frontier.choices[0]
        };
        assert!(frontier.choices[0].fuse_dispatches);
        assert_eq!(frontier.choices[1], unfused);
        // Choice 1 is measured on the incumbent's form right after the first
        // layouts and one more form.
        let mut order = CoordinateOrder::new(8, 2);
        let visits: Vec<_> = std::iter::from_fn(|| order.next((0, 0))).collect();
        assert!(visits[..5].contains(&(0, 1)));
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
        let frontier = || {
            let seeds = (0..16)
                .map(|index| Seed {
                    backward_rank: 0,
                    graph: graph.clone(),
                    options: options.clone(),
                    plan: plan.clone(),
                    description: format!("graph={index}"),
                })
                .collect();
            implementations(seeds, caps, 65536, 512 << 20)
        };
        assert!(
            frontier().choices.iter().take(16).any(|choice| {
                choice.attention.1.is_some_and(|(ept, shape)| {
                    ept == 16 && shape.interleave && shape.threads == 256 && shape.keys == 8
                })
            }),
            "cover paired lane layouts before deeper geometry combinations"
        );
        // Reserve half the ordinary 64-program allowance for submission probes.
        // Even with a full graph frontier, distinct attention layouts must fit.
        let budget = BuildSearchOptions::default().max_programs / 2;
        let mut ordinary = frontier();
        let programs: Vec<_> = (0..budget)
            .map_while(|_| measure::Frontier::next(&mut ordinary, None))
            .collect();
        assert!(programs.iter().all(|p| p.submission_chunks.is_none()));
        assert!(programs.iter().any(|p| {
            p.origin.is_some_and(|origin| origin.form == 0)
                && p.plan.knobs.flash.interleave
                && p.plan
                    .dispatches
                    .iter()
                    .any(|d| d.shader == compile::ShaderEntry::FlashAttention)
        }));
        // An unfused incumbent must get the same early attention coverage;
        // ranking every fused combination first would only help one policy.
        let mut frontier = frontier();
        let choice = frontier
            .choices
            .iter()
            .position(|c| c.attention == (0, None) && !c.fuse_dispatches)
            .unwrap();
        let winner = Some(measure::Origin { form: 0, choice });
        let mut layouts = collections::HashSet::new();
        for _ in 0..budget {
            let Some(program) = measure::Frontier::next(&mut frontier, winner) else {
                break;
            };
            let choice = choice_of(&frontier, &program);
            if choice.attention.1.is_some() {
                assert!(!choice.fuse_dispatches);
                layouts.insert(choice.attention);
            }
        }
        assert!(layouts.len() >= 8, "too few distinct attention layouts");
    }

    #[test]
    fn physical_choices_are_not_starved_by_many_graph_forms() {
        // Whisper-like: a dozen forms and dozens of attention layouts under the
        // default program bound. Half the budget goes to the incumbent's choices.
        let mut order = CoordinateOrder::new(12, 72);
        let budget = BuildSearchOptions::default().max_programs;
        let visits: Vec<_> = (0..budget).map_while(|_| order.next((0, 0))).collect();
        let choices = visits.iter().filter(|&&(form, _)| form == 0).count();
        assert!(
            choices >= budget / 2 - INITIAL_LAYOUTS,
            "{choices} choices on form 0"
        );
        assert!((0..12).all(|form| visits.contains(&(form, 0))));
    }

    #[test]
    fn arithmetic_cover_preserves_initial_layouts_and_uses_their_winner() {
        let mut order = CoordinateOrder::new(8, 6);
        order.priority_choices.push_back(5);
        order.priority_attention = Some(4);
        for form in 0..INITIAL_LAYOUTS {
            assert_eq!(order.next((0, 0)), Some((form, 0)));
        }
        let mut visits = INITIAL_LAYOUTS;
        while order.visited_forms.iter().filter(|&&v| v).count() < ARITHMETIC_LAYOUTS {
            let (_, choice) = order.next((1, 0)).unwrap();
            assert!(choice < 4, "arithmetic displaced ordinary layout coverage");
            visits += 1;
        }
        assert_eq!(order.next((1, 0)), Some((1, 5)));
        assert_eq!(order.next((1, 5)), Some((1, 4)));
        let rest: Vec<_> = std::iter::from_fn(|| order.next((1, 0))).collect();
        assert_eq!(rest.len() + visits + 2, 8 * 6);
        assert!(order.exhausted());
    }

    #[test]
    fn coordinate_winners_carry_over() {
        let mut order = CoordinateOrder::new(6, 5);
        for _ in 0..INITIAL_LAYOUTS {
            order.next((0, 0));
        }
        // Choice 3 won on form 0: the next untried form inherits it.
        let next = std::iter::from_fn(|| order.next((0, 3)))
            .find(|&(form, _)| form != 0)
            .unwrap();
        assert_eq!(next, (3, 3));
        // Form 3 won: choices continue on it, skipping the one it ran with.
        let next = std::iter::from_fn(|| order.next((3, 3)))
            .find(|&(form, _)| form == 3)
            .unwrap();
        assert_eq!(next, (3, 0));
        // Graph priority can carry a winning derivative layout to a later
        // forward form without changing the initial cover or physical axis.
        order.form_order.reverse();
        let next = std::iter::from_fn(|| order.next((3, 3)))
            .find(|&(form, _)| form > 3)
            .unwrap();
        assert_eq!(next, (5, 3));
    }

    #[test]
    fn coordinate_order_exhausts_each_pair_once_as_winners_change() {
        for early_arithmetic in [false, true] {
            for seeds in 0..8 {
                for choices in 0..12 {
                    let mut order = CoordinateOrder::new(seeds, choices);
                    if early_arithmetic {
                        order.arithmetic_layouts = 1;
                        order.priority_choices.extend((1..choices).rev());
                    } else if choices > 1 {
                        order.priority_attention = Some(1);
                    }
                    let mut incumbent = (0, 0);
                    let mut visited = collections::HashSet::new();
                    while let Some(entry) = order.next(incumbent) {
                        assert!(entry.0 < seeds && entry.1 < choices);
                        assert!(visited.insert(entry));
                        if visited.len().is_multiple_of(3) {
                            if !early_arithmetic {
                                order.form_order.reverse();
                            }
                            incumbent = entry;
                        }
                    }
                    assert_eq!(visited.len(), seeds * choices);
                    assert!(order.exhausted());
                    assert!(order.next(incumbent).is_none());
                }
            }
        }
    }

    /// Checks every user output, and the gradients of parameters q, k and v,
    /// against the f64 reference values of the differentiated graph.
    fn check_attention_training(
        session: &crate::Session,
        full: &Graph,
        values: &[crate::reference::Tensor],
        scales: &[Vec<f64>],
    ) -> Result<(), String> {
        for (index, &id) in full.outputs().iter().enumerate() {
            let want = &values[id as usize].data;
            let mut got = vec![0.0; want.len()];
            if index < full.num_user_outputs() {
                session.read_output_by_index(index, &mut got);
            } else {
                session.read_param_grad(["q", "k", "v"][index - full.num_user_outputs()], &mut got);
            }
            crate::reference::check(&got, want, &scales[id as usize], Default::default())
                .map_err(|error| format!("output {index}: {error}"))?;
        }
        Ok(())
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
        // Rejected challengers are never installed: see `measure`'s tests.
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
                observed.insert(native_parts(session));
                check_attention_training(session, &full, &values, &scales)
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
        assert!(
            report.trials.iter().all(|t| t.outcome.qualified),
            "{}",
            serde_json::to_string(&report).unwrap()
        );
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
                check_attention_training(session, &full, &values, &scales)
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
        let gpu = crate::reference::gpu::shared_context();
        let mut graph = Graph::new();
        let x = graph.input("x", &[3, 33]);
        let w = graph.parameter("w", &[33, 5]);
        let b = graph.input("b", &[3, 5]);
        let y = graph.matmul(x, w);
        let y = graph.add(y, b);
        let loss = graph.mean_all(y);
        graph.set_outputs(vec![loss]);
        // Training covers the inference checks and adds gradients and updates.
        let (mut session, report) = build_measured(
            &graph,
            SessionConfig {
                mode: Mode::Training,
                gpu: Some(gpu),
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
                ..Default::default()
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
                let mut grad = [0.0; 165];
                s.read_param_grad("w", &mut grad);
                if grad.iter().any(|g| (g - 0.05).abs() > 2e-6) {
                    return Err("parameter gradient changed".into());
                }
                Ok(())
            },
        )
        .unwrap();
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
