//! Measured construction: graph alternatives survive until their kernels are tuned.
use super::{SessionConfig, prepare_graph};
use crate::{
    Graph, Session, TuneOptions,
    compile::{self, ExecutionPlan},
    optimize, runtime,
};
use serde::Serialize;

mod attention_blocks;
mod measure;
mod storage;
mod weight_cache;
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
/// The runner executes one step before each read-only `qualify` call, repeating
/// each checkpoint `options.qualification_runs` times. Check every
/// observable output, gradient and persistent update against your numerical
/// contract. The ordinary untuned program is qualified first and retained as
/// the initial incumbent. Its tuned version, including memoized decisions,
/// must win a whole-program comparison to replace it. Every challenger is
/// qualified after kernel tuning, before it is timed;
/// one with kernel classes left to probe is also qualified before tuning.
/// Repeatability checks should compare executions of the same complete plan;
/// different plans can use different arithmetic and must each satisfy the
/// caller's independent numerical contract. Kernel tuning can change the plan
/// between the pre-tuning and post-tuning checkpoints.
/// A challenger with an identical execution plan and submission count keeps the incumbent
/// without warmup or paired timing. A winning challenger is qualified again after
/// measurement. The selected program is checked before returning if another
/// challenger ran since its last check or `final_qualification_runs` requests it. A failing challenger is discarded; a
/// failing final incumbent aborts construction.
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
/// dispatches are kernel-tuned before comparison, reserving estimated time for
/// whole-program warmup, sampling and qualification within the common deadline.
/// Private kernel wins alone never replace the incumbent. This bounded search
/// does not promise a global optimum.
/// After initial layout/arithmetic coverage, alternate layout exploration with
/// submission probes on the incumbent. A winning layout reopens those probes;
/// later layouts inherit its measured count rather than expanding a
/// layout/submission cross product.
///
/// `options.tuning` replaces `cfg.tune`. Build-plan caching is not yet supported:
/// calibration data and measured policy are not part of the ordinary cache key.
/// Inputs remain resident between steps. When deployment uploads inputs before
/// each step, use [`build_measured_with_inputs`] to measure the same workload.
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
    programs.weight_cache.enabled = cfg.mode == crate::Mode::Inference
        && programs.allow_matrix_storage
        && supported.contains(&crate::DType::F16);
    programs.enable_matrix_inputs(&supported, &gpu.capabilities());
    if cfg.options.flash_forward_coop {
        programs.enable_forward_products(gpu.capabilities().subgroup_size);
    }
    programs.enable_gradient_inputs(&supported);
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
        // F32/compensated/scalar path. Prefer F16's mantissa first, but retain BF16
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct AxisChoice {
    attention: AttentionChoice,
    fuse_dispatches: bool,
    matrix_inputs: Option<crate::DType>,
    matrix_storage: bool,
    vector_convolutions_only: bool,
    winograd: Option<crate::DType>,
    f16_attention: bool,
    forward_products: bool,
    gradient_inputs: Option<crate::DType>,
    bf16_nn_gradients: bool,
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
            arithmetic_layouts: ARITHMETIC_LAYOUTS,
            priority_attention: None,
            deferred: collections::HashSet::new(),
        }
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
            // Keep the ordinary graph cover ahead of reduced attention too:
            // an early kernel win can hide a larger graph-layout improvement.
            if let Some(choice) = self.priority_attention.take() {
                let entry = (incumbent.0, choice);
                if self.mark(entry) {
                    return Some(entry);
                }
            }
            while let Some(choice) = self.priority_choices.pop_front() {
                let entry = (incumbent.0, choice);
                if self.deferred.contains(&entry) {
                    continue;
                }
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
        let available =
            |entry: &(usize, usize)| !self.tried.contains(entry) && !self.deferred.contains(entry);
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
        matrix_inputs: None,
        matrix_storage: false,
        vector_convolutions_only: false,
        winograd: None,
        f16_attention: false,
        forward_products: false,
        gradient_inputs: None,
        bf16_nn_gradients: false,
        fuse_dispatches: seeds
            .first()
            .is_some_and(|seed| seed.options.fuse_dispatches),
    };
    // Rank the first alternative on each axis early.
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
                matrix_inputs: None,
                matrix_storage: false,
                vector_convolutions_only: false,
                winograd: None,
                f16_attention: false,
                forward_products: false,
                gradient_inputs: None,
                bf16_nn_gradients: false,
            })
        })
        .filter(|&choice| seen.insert(choice))
        .collect();
    let mut choices: Vec<_> = std::iter::once(baseline).chain(cover).chain(tail).collect();
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
        attention_blocks: attention_blocks::Refinements::new(shared_memory_bytes),
        weight_cache: weight_cache::Refinements::new(max_partial_bytes),
        refine_next: false,
        prepared_source: None,
        queued_splits: false,
        replace_extracted_matrices: false,
        replace_extracted_attention: false,
        defer_bf16: false,
        allow_matrix_storage: true,
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
    attention_blocks: attention_blocks::Refinements,
    weight_cache: weight_cache::Refinements,
    refine_next: bool,
    prepared_source: Option<(measure::Origin, ExecutionPlan)>,
    replace_extracted_matrices: bool,
    replace_extracted_attention: bool,
    defer_bf16: bool,
    allow_matrix_storage: bool,
    matrix_shapes: collections::HashMap<crate::DType, ([u32; 3], u32)>,
}

impl Implementations {
    fn enable_forward_products(&mut self, subgroup_size: u32) {
        if !(4..=64).contains(&subgroup_size) || !subgroup_size.is_power_of_two() {
            return;
        }
        if !self.seeds.iter().any(|seed| {
            fill_attention_products(&mut seed.plan.clone(), self.caps, self.shared_memory_bytes) > 0
        }) {
            return;
        }
        let first = self.choices.len();
        if self.order.priority_attention.is_none() {
            self.order.priority_attention = Some(first);
        } else {
            // Keep the backward trial first. The forward trial inherits its
            // winner, just like the later matrix arithmetic alternatives.
            self.order.priority_choices.push_front(first);
        }
        let ordinary = self.choices.clone();
        self.choices.extend(
            ordinary
                .into_iter()
                .filter(|choice| choice.attention.1.is_none())
                .map(|choice| AxisChoice {
                    forward_products: true,
                    ..choice
                }),
        );
        self.order.choices = self.choices.len();
        self.order.choice_order = (0..self.choices.len()).collect();
    }

    fn enable_gradient_inputs(&mut self, inputs: &[crate::DType]) {
        let ordinary = self.choices.clone();
        for multiply in [crate::DType::F16, crate::DType::BF16] {
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
            // Keep ordinary graph coverage and backward attention first.
            // Try F16 forward + derivatives together when both are available;
            // separate choices remain reachable if the combination fails.
            if multiply == crate::DType::F16 {
                let forward = ordinary.iter().position(|c| {
                    c.matrix_inputs == Some(multiply)
                        && !c.matrix_storage
                        && !c.vector_convolutions_only
                        && c.winograd.is_none()
                });
                self.order
                    .priority_choices
                    .push_front(self.choices.len() + forward.unwrap_or(0));
            } else {
                self.order.priority_choices.push_back(self.choices.len());
            }
            self.choices
                .extend(ordinary.iter().map(|choice| AxisChoice {
                    gradient_inputs: Some(multiply),
                    ..*choice
                }));
        }
        if inputs.contains(&crate::DType::BF16)
            && self.matrix_shapes.contains_key(&crate::DType::BF16)
            && self.seeds.iter().any(|seed| {
                fill_nn_gradient_candidates(&mut seed.plan.clone(), self.replace_extracted_matrices)
                    != 0
            })
        {
            // Cover a different derivative family before revisiting the
            // transposed family's dtype. Inherit its measured F16 winner.
            let priority = self
                .order
                .priority_choices
                .iter()
                .position(|&index| self.choices[index].gradient_inputs == Some(crate::DType::F16))
                .map_or(0, |index| index + 1);
            self.order
                .priority_choices
                .insert(priority, self.choices.len());
            let ordinary = self.choices.clone();
            self.choices
                .extend(ordinary.into_iter().map(|choice| AxisChoice {
                    bf16_nn_gradients: true,
                    ..choice
                }));
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

/// Retain the staged/scalar candidate and measure this forward algorithm
/// independently of backward precision. Parameter/gradient types and all F32
/// output/LSE bindings stay unchanged; qualification belongs to the caller.
fn fill_attention_products(
    plan: &mut ExecutionPlan,
    caps: crate::codegen::CoopCaps,
    shared_memory_bytes: u32,
) -> usize {
    use compile::{DispatchParams as _, ShaderEntry};
    let mut count = 0;
    for dispatch in &mut plan.dispatches {
        if dispatch.full_precision()
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
        ) {
            continue;
        }
        dispatch.shader = ShaderEntry::FlashAttentionProducts;
        dispatch.workgroups = [p.q_seq.div_ceil(16), p.packed_heads >> 16, 1];
        count += 1;
    }
    count
}

/// Checked typed products preserve the F32 binding contract, including tails.
/// Conversion happens inside each timed dispatch. This also reaches rectangular
/// subgroup shapes and workgroup tuning without the legacy square-tile heuristic.
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

/// Plain F16 NN fails the long-K vocabulary-gradient gate. Transposed
/// products and BF16 NN are independent choices; every program still qualifies.
fn fill_gradient_candidates(
    plan: &mut ExecutionPlan,
    multiply: crate::DType,
    replace_extracted: bool,
) -> usize {
    fill_gradient_products(
        plan,
        multiply,
        replace_extracted,
        &[
            compile::ShaderEntry::MatMulAT,
            compile::ShaderEntry::MatMulBT,
        ],
    )
}

fn fill_nn_gradient_candidates(plan: &mut ExecutionPlan, replace_extracted: bool) -> usize {
    fill_gradient_products(
        plan,
        crate::DType::BF16,
        replace_extracted,
        &[compile::ShaderEntry::MatMul],
    )
}

fn fill_gradient_products(
    plan: &mut ExecutionPlan,
    multiply: crate::DType,
    replace_extracted: bool,
    entries: &[compile::ShaderEntry],
) -> usize {
    let mut count = 0;
    for d in &mut plan.dispatches {
        if d.is_derivative
            && !d.requires_full_precision
            && entries.contains(&d.shader)
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
        || d.weight_format != compile::WeightFormat::F32
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
    // Keep the existing partial buffer and SumRows contract, adding a
    // fused addend only in the first partition. Each part needs a K block.
    if splits > 1 && splits > k.div_ceil(32) {
        return false;
    }
    let grid = [m.div_ceil(32), n.div_ceil(32), splits];
    if !compile::workgroups_within_portable_limits(grid) {
        return false;
    }
    d.kernel = compile::Kernel::CooperativeTyped {
        types: crate::codegen::MatrixTypes {
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
        let covered = self.order.visited_forms.iter().filter(|&&v| v).count()
            >= self.order.seeds.min(self.order.arithmetic_layouts);
        if self.order.arithmetic_layouts != 1 && !covered {
            return self.weight_cache.early_remaining();
        }
        let form = incumbent.map_or(0, |origin| origin.form);
        let nn_selected =
            incumbent.is_some_and(|origin| self.choices[origin.choice].bf16_nn_gradients);
        let storage_selected =
            incumbent.is_some_and(|origin| self.choices[origin.choice].matrix_storage);
        self.order
            .priority_choices
            .iter()
            .filter(|&&choice| {
                // Once graph coverage is complete, reserve one comparison for
                // the independent NN family before further submission probes.
                (self.order.arithmetic_layouts == 1
                    || (self.choices[choice].bf16_nn_gradients && !nn_selected)
                    || (self.choices[choice].matrix_storage
                        && !storage_selected
                        && (self.choices[choice].gradient_inputs.is_some()
                            || self.choices[choice].bf16_nn_gradients)))
                    && !self.order.tried.contains(&(form, choice))
                    && !self.order.deferred.contains(&(form, choice))
            })
            .count()
            + self.weight_cache.early_remaining()
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
                    && !self.order.priority_choices.contains(&index)
                {
                    self.order.priority_choices.push_back(index);
                }
            }
        }
        if !self.defer_bf16 {
            return;
        }
        for family in 0..3 {
            let dtype = match family {
                0 => choice.matrix_inputs,
                1 => choice.winograd,
                _ => choice.gradient_inputs,
            };
            if dtype != Some(crate::DType::F16) {
                continue;
            }
            let mut bf16 = choice;
            match family {
                0 => bf16.matrix_inputs = Some(crate::DType::BF16),
                1 => bf16.winograd = Some(crate::DType::BF16),
                _ => bf16.gradient_inputs = Some(crate::DType::BF16),
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

    fn selected(&mut self, origin: Option<measure::Origin>, plan: &ExecutionPlan) {
        self.attention_blocks.selected(origin, plan);
        self.weight_cache.selected(origin, plan);
        // Changes to private storage need the parent's F32 temporary sizes.
        // The cached incumbent remains available for the actual comparison.
        self.prepared_source = origin.map(|origin| {
            (
                origin,
                plan.f16_weight_parent.as_deref().unwrap_or(plan).clone(),
            )
        });
        let Some(origin) = origin else {
            return;
        };
        let active = self.choices[origin.choice];
        if !self.allow_matrix_storage
            || active.matrix_storage
            || !plan.dispatches.iter().any(|d| {
                d.is_derivative
                    && !d.requires_full_precision
                    && matches!(d.kernel,
                    compile::Kernel::CooperativeTyped { types, .. }
                    | compile::Kernel::CooperativeWorkgroup { types: Some(types), .. }
                    if matches!(types.multiply, crate::DType::F16 | crate::DType::BF16))
            })
            || storage::fill(&mut plan.clone()) == 0
        {
            return;
        }
        // Add only the actual winner's storage neighbor, after its derivative
        // multiplication has qualified. Do not expand the initial cross product.
        let stored = AxisChoice {
            matrix_storage: true,
            ..active
        };
        let index = self
            .choices
            .iter()
            .position(|&c| c == stored)
            .unwrap_or_else(|| {
                let index = self.choices.len();
                self.choices.push(stored);
                self.order.choices = self.choices.len();
                self.order.choice_order.push(index);
                index
            });
        if !self.order.tried.contains(&(origin.form, index))
            && !self.order.deferred.contains(&(origin.form, index))
            && !self.order.priority_choices.contains(&index)
        {
            // This is reserved work: keep it directly after a pending NN trial,
            // ahead of soft arithmetic alternatives that would block submissions.
            let priority = if active.bf16_nn_gradients {
                None
            } else {
                self.order.priority_choices.iter().position(|&other| {
                    self.choices[other].bf16_nn_gradients
                        && !self.order.tried.contains(&(origin.form, other))
                        && !self.order.deferred.contains(&(origin.form, other))
                })
            }
            .map_or(0, |position| position + 1);
            self.order.priority_choices.insert(priority, index);
        }
    }

    fn next(&mut self, incumbent: Option<measure::Origin>) -> Option<measure::Program> {
        let form = incumbent.map_or(0, |origin| origin.form);
        let storage_pending = self.order.priority_choices.iter().any(|&choice| {
            self.choices[choice].matrix_storage
                && !self.order.tried.contains(&(form, choice))
                && !self.order.deferred.contains(&(form, choice))
        });
        if !storage_pending && let Some(program) = self.weight_cache.next_early() {
            return Some(program);
        }
        if self.refine_next && self.early_programs_remaining(incumbent) == 0 {
            self.refine_next = false;
            if let Some(program) = self.attention_blocks.next() {
                return Some(program);
            }
        }
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
            let backward_rank = self.seeds.get(incumbent.0)?.backward_rank;
            self.order
                .form_order
                .sort_by_key(|&index| (self.seeds[index].backward_rank != backward_rank, index));
            let active = self.choices[incumbent.1];
            // Carry the winning attention choice into arithmetic trials.
            for index in &mut self.order.priority_choices {
                let arithmetic = self.choices[*index];
                let nn_trial = arithmetic.bf16_nn_gradients && !active.bf16_nn_gradients;
                let derivative_trial = nn_trial
                    || (arithmetic.gradient_inputs.is_some()
                        && arithmetic.gradient_inputs != active.gradient_inputs);
                let neighbor = AxisChoice {
                    matrix_inputs: if derivative_trial {
                        active.matrix_inputs.or(arithmetic.matrix_inputs)
                    } else {
                        arithmetic.matrix_inputs
                    },
                    matrix_storage: if derivative_trial {
                        active.matrix_storage
                    } else {
                        arithmetic.matrix_storage
                    },
                    vector_convolutions_only: if derivative_trial {
                        active.vector_convolutions_only
                    } else {
                        arithmetic.vector_convolutions_only
                    },
                    winograd: if derivative_trial {
                        active.winograd
                    } else {
                        arithmetic.winograd
                    },
                    gradient_inputs: if nn_trial {
                        active.gradient_inputs
                    } else {
                        arithmetic.gradient_inputs.or(active.gradient_inputs)
                    },
                    bf16_nn_gradients: arithmetic.bf16_nn_gradients || active.bf16_nn_gradients,
                    forward_products: arithmetic.forward_products || active.forward_products,
                    ..active
                };
                if let Some(next) = self.choices.iter().position(|&choice| choice == neighbor) {
                    *index = next;
                } else if active.matrix_storage && derivative_trial {
                    // A lazily added storage choice still needs independent
                    // derivative neighbors, including NN after storage wins.
                    *index = self.choices.len();
                    self.choices.push(neighbor);
                    self.order.choices = self.choices.len();
                    self.order.choice_order.push(*index);
                }
            }
            // Try the other arithmetic family on top of a winning family
            // before spending the remaining deadline on attention layouts.
            let combined: Vec<_> = self
                .order
                .priority_choices
                .iter()
                .filter_map(|&index| {
                    let trial = self.choices[index];
                    if trial.matrix_inputs.is_some() == trial.winograd.is_some() {
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
                let entry = (incumbent.0, index);
                if !self.order.tried.contains(&entry)
                    && !self.order.deferred.contains(&entry)
                    && !self.order.priority_choices.contains(&index)
                {
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
                        + usize::from(choice.gradient_inputs != active.gradient_inputs)
                        + usize::from(choice.bf16_nn_gradients != active.bf16_nn_gradients),
                    choice.forward_products != active.forward_products,
                    choice.f16_attention != active.f16_attention,
                    choice.attention != active.attention,
                    choice.fuse_dispatches != active.fuse_dispatches,
                    index,
                )
            });
            let Some((seed_index, choice_index)) = self.order.next(incumbent) else {
                // Vector rounding changes arithmetic. Let ordinary layouts
                // settle first so recording retains their selected parent.
                return self
                    .attention_blocks
                    .next()
                    .or_else(|| self.weight_cache.next());
            };
            let seed = &self.seeds[seed_index];
            let AxisChoice {
                attention: (splits, flash),
                fuse_dispatches,
                matrix_inputs,
                matrix_storage,
                vector_convolutions_only,
                winograd,
                f16_attention,
                forward_products,
                gradient_inputs,
                bf16_nn_gradients,
            } = self.choices[choice_index];
            // Refine the measured winner's NN multiplication or private storage.
            // Rebuilding its graph would discard unrelated kernel/layout tuning.
            let source = self.prepared_source.as_ref().filter(|entry| {
                let origin = entry.0;
                let selected = self.choices[origin.choice];
                origin.form == seed_index
                    && ((!selected.bf16_nn_gradients
                        && bf16_nn_gradients
                        && AxisChoice {
                            bf16_nn_gradients: true,
                            ..selected
                        } == self.choices[choice_index])
                        || (!selected.matrix_storage
                            && matrix_storage
                            && AxisChoice {
                                matrix_storage: true,
                                ..selected
                            } == self.choices[choice_index]))
            });
            let nn_trial = source
                .is_some_and(|s| !self.choices[s.0.choice].bf16_nn_gradients && bf16_nn_gradients);
            let storage_trial =
                source.is_some_and(|s| !self.choices[s.0.choice].matrix_storage && matrix_storage);
            let prepared = source.is_some();
            let mut plan = if let Some(source) = source {
                source.1.clone()
            } else if splits == 0
                && !f16_attention
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
                let mut attention_graph;
                let graph = if f16_attention && self.replace_extracted_attention {
                    attention_graph = seed.graph.deep_clone();
                    for node in attention_graph.nodes_mut() {
                        if matches!(
                            node.op,
                            crate::graph::Op::MultiHeadAttnGradQ { .. }
                                | crate::graph::Op::MultiHeadAttnGradK { .. }
                                | crate::graph::Op::MultiHeadAttnGradV { .. }
                        ) {
                            node.attention_ept_cap = None;
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
            if !prepared {
                if forward_products
                    && fill_attention_products(&mut plan, self.caps, self.shared_memory_bytes) == 0
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
                    if fill_gradient_candidates(
                        &mut plan,
                        multiply,
                        self.replace_extracted_matrices,
                    ) == 0
                    {
                        continue;
                    }
                }
            }
            if bf16_nn_gradients
                && (!prepared || nn_trial)
                && fill_nn_gradient_candidates(&mut plan, self.replace_extracted_matrices) == 0
            {
                continue;
            }
            if matrix_storage {
                let groups = prepared.then(|| plan.groups.clone());
                let changed = storage::fill(&mut plan);
                if let Some(groups) = groups {
                    // Storage changes leave dependencies and dispatch order intact.
                    plan.groups = groups;
                }
                if changed == 0 && (!prepared || storage_trial) {
                    continue;
                }
            }
            if !prepared {
                if let Some(storage) = winograd {
                    let (shape, subgroup) = self.matrix_shapes[&storage];
                    if winograd::fill(&mut plan, storage, shape, subgroup) == 0 {
                        continue;
                    }
                }
            }
            self.attention_blocks.apply(&mut plan);
            self.refine_next = true;
            let mut description = format!(
                "{}, dispatch_fusion={fuse_dispatches}, attention_splits={splits}, flash={flash:?}",
                seed.description
            );
            if f16_attention {
                description.push_str(", f16_attention=true");
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
            if matrix_storage && matrix_inputs.is_none() {
                description.push_str(", private_matrix_storage=true");
            }
            if let Some(dtype) = gradient_inputs {
                description.push_str(&format!(", gradient_inputs={dtype:?}"));
            }
            if bf16_nn_gradients {
                description.push_str(", nn_gradient_inputs=BF16");
            }
            if let Some(dtype) = winograd {
                description.push_str(&format!(", winograd={dtype:?}"));
            }
            return Some(measure::Program {
                prepared,
                retune: false,
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
        !self.queued_splits
            || !self.split_plans.is_empty()
            || !self.order.exhausted()
            || self.attention_blocks.may_remain()
            || self.weight_cache.may_remain()
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
            prepared: false,
            retune: false,
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

    #[test]
    fn attention_products_keep_fallback_and_precision_policy() {
        use measure::Frontier as _;
        let mut graph = Graph::new();
        let q = graph.input("q", &[33, 144]);
        let k = graph.input("k", &[19, 48]);
        let v = graph.input("v", &[19, 48]);
        let y = graph.multi_head_attn(q, k, v, 3, 1, 48, true);
        graph.set_outputs(vec![y]);
        let caps = crate::codegen::CoopCaps {
            f16_attention_rows: 8,
            ..Default::default()
        };
        let mut frontier = graph_frontier(graph);
        frontier.caps = caps;
        frontier.enable_forward_products(32);
        let baseline = frontier.next(None).unwrap();
        assert!(
            baseline
                .plan
                .dispatches
                .iter()
                .all(|d| d.shader != compile::ShaderEntry::FlashAttentionProducts)
        );
        let candidate = frontier.next(baseline.origin).unwrap();
        assert!(
            candidate
                .plan
                .dispatches
                .iter()
                .any(|d| d.shader == compile::ShaderEntry::FlashAttentionProducts
                    && d.workgroups == [3, 3, 1])
        );
        assert_ne!(baseline.plan, candidate.plan);
        assert!(compile::ShaderEntry::FlashAttentionProducts.is_attention());
        let restored: ExecutionPlan =
            serde_json::from_str(&serde_json::to_string(&candidate.plan).unwrap()).unwrap();
        assert_eq!(candidate.plan, restored);
        let mut protected = baseline.plan.clone();
        for d in &mut protected.dispatches {
            d.requires_full_precision = true;
        }
        assert_eq!(fill_attention_products(&mut protected, caps, 65536), 0);
        let mut locked = baseline.plan.clone();
        for d in &mut locked.dispatches {
            d.schedule_locked = true;
        }
        assert_eq!(fill_attention_products(&mut locked, caps, 65536), 0);
        assert_eq!(
            fill_attention_products(&mut baseline.plan.clone(), Default::default(), 65536),
            0
        );
        assert_eq!(
            fill_attention_products(&mut baseline.plan.clone(), caps, 8192),
            0
        );
        let mut unsupported = graph_frontier((*frontier.seeds[0].graph).deep_clone());
        unsupported.caps = caps;
        unsupported.enable_forward_products(128);
        assert!(unsupported.choices.iter().all(|c| !c.forward_products));
    }

    #[test]
    fn forward_products_preserve_backward_choices_and_parameter_storage() {
        use measure::Frontier as _;
        let mut graph = Graph::new();
        let q = graph.parameter("q", &[33, 48]);
        let k = graph.parameter("k", &[19, 16]);
        let v = graph.parameter("v", &[19, 16]);
        let y = graph.multi_head_attn(q, k, v, 3, 1, 16, true);
        let weights = graph.input("weights", &[33, 48]);
        let weighted = graph.mul(y, weights);
        let loss = graph.sum_all(weighted);
        graph.set_outputs(vec![loss]);
        let graph = std::sync::Arc::new(crate::autodiff::differentiate(&graph));
        let caps = crate::codegen::CoopCaps {
            f16_attention_rows: 8,
            ..Default::default()
        };
        let options = compile::CompileOptions {
            flash_backward_coop: false,
            ..Default::default()
        };
        let plan = compile::compile_with_caps(&graph, &options, caps, 65536);
        let mut frontier = implementations(
            vec![Seed {
                graph,
                options,
                plan,
                description: "training".into(),
                backward_rank: 0,
            }],
            caps,
            65536,
            512 << 20,
        );
        frontier.enable_forward_products(32);
        let baseline = frontier.next(None).unwrap();
        let backward = frontier.next(baseline.origin).unwrap();
        let backward_choice = frontier.choices[backward.origin.unwrap().choice];
        assert!(backward_choice.f16_attention && !backward_choice.forward_products);
        let forward = frontier.next(backward.origin).unwrap();
        let forward_choice = frontier.choices[forward.origin.unwrap().choice];
        assert!(forward_choice.f16_attention && forward_choice.forward_products);
        assert_eq!(
            forward.plan.param_grad_pairs,
            backward.plan.param_grad_pairs
        );
        assert_eq!(forward.plan.param_types, backward.plan.param_types);
        let mut expected = backward.plan;
        assert!(fill_attention_products(&mut expected, caps, 65536) > 0);
        assert_eq!(forward.plan, expected);
        let mut reached = collections::HashSet::from([(false, false), (false, true), (true, true)]);
        while let Some(program) = frontier.next(baseline.origin) {
            let choice = frontier.choices[program.origin.unwrap().choice];
            reached.insert((choice.forward_products, choice.f16_attention));
        }
        assert_eq!(
            reached.len(),
            4,
            "both forward/backward choices stay reachable"
        );
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

    fn nn_gradient_frontier() -> Implementations {
        let mut graph = Graph::new();
        let x = graph.input("x", &[32, 64]);
        let w = graph.parameter("w", &[16, 64]);
        let y = graph.matmul_bt(x, w);
        let loss = graph.mean_all(y);
        graph.set_outputs(vec![loss]);
        graph_frontier(crate::autodiff::differentiate(&graph))
    }

    #[test]
    fn nn_bf16_keeps_storage_pins_and_transposed_products() {
        let baseline = nn_gradient_frontier().seeds.remove(0).plan;
        let nn: Vec<_> = baseline
            .dispatches
            .iter()
            .enumerate()
            .filter_map(|(i, d)| {
                (d.is_derivative && d.shader == compile::ShaderEntry::MatMul).then_some(i)
            })
            .collect();
        assert_eq!(nn.len(), 1);
        let mut candidate = baseline.clone();
        assert_eq!(fill_nn_gradient_candidates(&mut candidate, true), 1);
        for &i in &nn {
            let d = &candidate.dispatches[i];
            assert!(d.full_precision() && d.is_derivative && !d.requires_full_precision);
            assert!(
                matches!(d.kernel, compile::Kernel::CooperativeTyped { types, .. }
                if types.a == crate::DType::F32 && types.b == crate::DType::F32
                    && types.output == crate::DType::F32 && types.multiply == crate::DType::BF16)
            );
            candidate.dispatches[i].kernel = baseline.dispatches[i].kernel.clone();
            candidate.dispatches[i].workgroups = baseline.dispatches[i].workgroups;
        }
        assert_eq!(candidate, baseline);
        for pin in [true, false] {
            let mut protected = baseline.clone();
            for &i in &nn {
                protected.dispatches[i].requires_full_precision = true;
                protected.dispatches[i].is_derivative = pin;
            }
            assert_eq!(fill_nn_gradient_candidates(&mut protected, true), 0);
        }
        let mut locked = baseline.clone();
        for &i in &nn {
            locked.dispatches[i].schedule_locked = true;
        }
        assert_eq!(fill_nn_gradient_candidates(&mut locked, false), 0);
        assert_eq!(fill_nn_gradient_candidates(&mut locked, true), 1);
    }

    #[test]
    fn derivative_storage_refines_the_actual_winner_and_keeps_nn_priority() {
        use crate::DType::{BF16, F16};
        use measure::Frontier as _;
        for (allow_storage, nn_selected) in [(true, false), (true, true), (false, true)] {
            let mut g = Graph::new();
            let x = g.input("x", &[17, 23]);
            let a = g.parameter("a", &[38, 23]);
            let b = g.parameter("b", &[7, 19]);
            let y = g.matmul_bt(x, a);
            let y = g.swiglu_concat(y);
            let y = g.matmul_bt(y, b);
            let loss = g.mean_all(y);
            g.set_outputs(vec![loss]);
            let mut frontier = graph_frontier(crate::autodiff::differentiate(&g));
            frontier.allow_matrix_storage = allow_storage;
            let mut caps = blade_graphics::Capabilities {
                shader_float16: true,
                shader_bfloat16: true,
                subgroup_size: 32,
                ..Default::default()
            };
            caps.cooperative_matrix.f16_f32_shapes = vec![[8, 16, 16]];
            caps.cooperative_matrix.bf16_f32_shapes = vec![[8, 16, 16]];
            frontier.enable_matrix_inputs(&[F16, BF16], &caps);
            frontier.enable_gradient_inputs(&[F16, BF16]);
            let choice = frontier
                .choices
                .iter()
                .position(|c| {
                    c.matrix_inputs == Some(F16)
                        && c.gradient_inputs == Some(F16)
                        && c.bf16_nn_gradients == nn_selected
                        && !c.matrix_storage
                        && c.attention == (0, None)
                        && c.fuse_dispatches
                })
                .unwrap();
            let origin = measure::Origin { form: 0, choice };
            let mut winner = frontier.seeds[0].plan.clone();
            assert!(fill_matrix_candidates(&mut winner, F16, false) > 0);
            assert!(fill_gradient_candidates(&mut winner, F16, false) > 0);
            if nn_selected {
                assert!(fill_nn_gradient_candidates(&mut winner, false) > 0);
            }
            winner.dispatches[0].label.push_str(":measured");
            winner.groups = (0..winner.dispatches.len()).map(|i| i..i + 1).collect();
            frontier.order.mark((0, choice));
            frontier.order.initial = INITIAL_LAYOUTS;
            frontier.queued_splits = true;
            frontier.order.priority_choices.clear();
            let pending_nn = (!nn_selected).then(|| {
                frontier
                    .choices
                    .iter()
                    .position(|&c| {
                        c == AxisChoice {
                            bf16_nn_gradients: true,
                            ..frontier.choices[choice]
                        }
                    })
                    .unwrap()
            });
            if let Some(index) = pending_nn {
                frontier.order.priority_choices.clear();
                frontier.order.priority_choices.push_back(index);
            }
            let soft = frontier
                .choices
                .iter()
                .position(|&c| {
                    c == AxisChoice {
                        gradient_inputs: Some(BF16),
                        ..frontier.choices[choice]
                    }
                })
                .unwrap();
            frontier.order.priority_choices.push_back(soft);
            let before = frontier.choices.len();
            frontier.selected(Some(origin), &winner);
            if !allow_storage {
                assert_eq!(frontier.choices.len(), before);
                assert_eq!(frontier.early_programs_remaining(Some(origin)), 0);
                continue;
            }
            assert_eq!(
                frontier.choices.len(),
                before + 1,
                "only one local neighbor is added"
            );
            assert_eq!(frontier.order.choices, frontier.choices.len());
            if let Some(index) = pending_nn {
                assert_eq!(frontier.order.priority_choices.front(), Some(&index));
                assert_eq!(frontier.early_programs_remaining(Some(origin)), 2);
                // Model an NN trial that did not replace this incumbent.
                frontier.order.priority_choices.pop_front();
            }
            assert_eq!(frontier.early_programs_remaining(Some(origin)), 1);
            assert_eq!(
                frontier.order.priority_choices.front(),
                Some(&before),
                "reserved storage must precede soft arithmetic alternatives"
            );
            let candidate = frontier.next(Some(origin)).unwrap();
            assert!(candidate.prepared);
            let mut expected = winner.clone();
            assert!(storage::fill(&mut expected) > 0);
            expected.groups = winner.groups;
            assert_eq!(candidate.plan, expected);
            assert_eq!(
                frontier.choices[candidate.origin.unwrap().choice],
                AxisChoice {
                    matrix_storage: true,
                    ..frontier.choices[choice]
                }
            );
            frontier.selected(candidate.origin, &candidate.plan);
            assert_eq!(frontier.choices.len(), before + 1);
            assert_eq!(frontier.early_programs_remaining(candidate.origin), 0);
            if let Some(index) = pending_nn {
                frontier.order.priority_choices.push_back(index);
                let nn = frontier.next(candidate.origin).unwrap();
                assert!(nn.prepared);
                let mut expected_nn = candidate.plan.clone();
                assert!(fill_nn_gradient_candidates(&mut expected_nn, false) > 0);
                let groups = expected_nn.groups.clone();
                storage::fill(&mut expected_nn);
                expected_nn.groups = groups;
                assert_eq!(nn.plan, expected_nn);
                let selected = frontier.choices[nn.origin.unwrap().choice];
                assert!(selected.bf16_nn_gradients && selected.matrix_storage);
            }
        }
    }

    #[test]
    fn nn_trial_preserves_the_measured_winners_other_kernels() {
        use measure::Frontier as _;
        let mut frontier = nn_gradient_frontier();
        let mut caps = blade_graphics::Capabilities {
            shader_float16: true,
            shader_bfloat16: true,
            subgroup_size: 32,
            ..Default::default()
        };
        caps.cooperative_matrix.f16_f32_shapes = vec![[8, 16, 16]];
        caps.cooperative_matrix.bf16_f32_shapes = vec![[8, 16, 16]];
        frontier.enable_matrix_inputs(&[crate::DType::F16, crate::DType::BF16], &caps);
        frontier.enable_gradient_inputs(&[crate::DType::F16, crate::DType::BF16]);
        let mut winner = frontier.next(None).unwrap();
        // Represent a runtime-tuned choice that rebuilding the graph would lose.
        winner.plan.dispatches[0].label.push_str(":measured");
        let origin = winner.origin.unwrap();
        frontier.selected(Some(origin), &winner.plan);
        let choice = frontier
            .choices
            .iter()
            .position(|&c| {
                c == AxisChoice {
                    bf16_nn_gradients: true,
                    ..frontier.choices[origin.choice]
                }
            })
            .unwrap();
        frontier.order.priority_choices.clear();
        frontier.order.priority_choices.push_back(choice);
        assert_eq!(frontier.early_programs_remaining(Some(origin)), 1);
        let candidate = frontier.next(Some(origin)).unwrap();
        assert!(candidate.prepared);
        let mut expected = winner.plan.clone();
        assert_eq!(fill_nn_gradient_candidates(&mut expected, false), 1);
        assert_eq!(candidate.plan, expected);
        assert_eq!(frontier.early_programs_remaining(Some(origin)), 0);
    }

    #[test]
    fn nn_bf16_is_independent_of_forward_and_transposed_precision() {
        use crate::DType::{BF16, F16};
        use measure::Frontier as _;
        let mut caps = blade_graphics::Capabilities {
            shader_float16: true,
            shader_bfloat16: true,
            subgroup_size: 32,
            ..Default::default()
        };
        caps.cooperative_matrix.f16_f32_shapes = vec![[8, 16, 16]];
        caps.cooperative_matrix.bf16_f32_shapes = vec![[8, 16, 16]];
        for allowed in [vec![], vec![F16], vec![BF16], vec![F16, BF16]] {
            let mut frontier = nn_gradient_frontier();
            frontier.enable_matrix_inputs(&allowed, &caps);
            frontier.enable_gradient_inputs(&allowed);
            assert_eq!(
                frontier.choices.iter().any(|c| c.bf16_nn_gradients),
                allowed.contains(&BF16)
            );
            if allowed != [F16, BF16] {
                continue;
            }
            let priority: Vec<_> = frontier
                .order
                .priority_choices
                .iter()
                .map(|&i| frontier.choices[i])
                .collect();
            assert_eq!(priority[0].gradient_inputs, Some(F16));
            assert!(priority[1].bf16_nn_gradients);
            let mut reached = collections::HashSet::new();
            let mut incumbent = None;
            while let Some(program) = frontier.next(incumbent) {
                let choice = frontier.choices[program.origin.unwrap().choice];
                reached.insert((
                    choice.matrix_inputs,
                    choice.gradient_inputs,
                    choice.bf16_nn_gradients,
                ));
                for d in program
                    .plan
                    .dispatches
                    .iter()
                    .filter(|d| d.is_derivative && d.shader == compile::ShaderEntry::MatMul)
                {
                    if let compile::Kernel::CooperativeTyped { types, .. } = d.kernel {
                        assert_eq!(types.multiply, BF16, "F16 NN remains unqualified");
                    }
                }
                if choice.gradient_inputs == Some(F16) {
                    incumbent = program.origin;
                }
            }
            for forward in [None, Some(F16), Some(BF16)] {
                for transposed in [None, Some(F16), Some(BF16)] {
                    assert!(
                        reached.contains(&(forward, transposed, true)),
                        "missing {forward:?}/{transposed:?}/BF16 NN"
                    );
                }
            }
        }
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
            assert_eq!(candidate.buffers, baseline.buffers);
            assert_eq!(candidate.param_grad_pairs, baseline.param_grad_pairs);
            assert_eq!(candidate.param_types, baseline.param_types);
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
            // Old plans have only the blanket bit and remain pinned.
            for &index in &products {
                protected.dispatches[index].is_derivative = false;
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
        let mut caps = blade_graphics::Capabilities {
            shader_float16: true,
            shader_bfloat16: true,
            subgroup_size: 32,
            ..Default::default()
        };
        caps.cooperative_matrix.f16_f32_shapes = vec![[8, 16, 16]];
        caps.cooperative_matrix.bf16_f32_shapes = vec![[8, 16, 16]];
        frontier.enable_matrix_inputs(&[], &caps);
        frontier.enable_gradient_inputs(&[]);
        assert!(frontier.choices.iter().all(|c| c.gradient_inputs.is_none()));
        frontier.enable_matrix_inputs(&[BF16, F16], &caps);
        frontier.enable_gradient_inputs(&[BF16, F16]);
        let baseline = frontier.next(None).unwrap();
        assert!(!baseline.description.contains("gradient_inputs="));
        assert!(
            baseline
                .plan
                .dispatches
                .iter()
                .filter(|d| d.is_derivative)
                .all(|d| !matches!(d.kernel, compile::Kernel::CooperativeTyped { .. }))
        );
        let mut reached = collections::HashSet::new();
        let mut incumbent = baseline.origin;
        while let Some(program) = frontier.next(incumbent) {
            let choice = frontier.choices[program.origin.unwrap().choice];
            reached.insert((choice.matrix_inputs, choice.gradient_inputs));
            // Simulate accepting F16 forward first, then a derivative choice.
            if choice.matrix_inputs == Some(F16)
                && (incumbent == baseline.origin || choice.gradient_inputs == Some(F16))
            {
                incumbent = program.origin;
            }
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
    fn eight_wide_arithmetic_precedes_layout_and_weight_split_exploration() {
        use measure::Frontier as _;
        for qualified in [false, true] {
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
            let mut caps = blade_graphics::Capabilities {
                shader_float16: true,
                shader_bfloat16: true,
                subgroup_size: 32,
                ..Default::default()
            };
            caps.cooperative_matrix.f16_f32_shapes.push([8, 8, 8]);
            caps.cooperative_matrix.bf16_f32_shapes.push([8, 8, 8]);
            frontier.enable_matrix_inputs(&[crate::DType::F16, crate::DType::BF16], &caps);
            let first = frontier.next(None).unwrap();
            assert!(!first.description.contains("matrix_inputs="));
            let incumbent = first.origin;
            assert_eq!(frontier.early_programs_remaining(incumbent), 2);
            let f16 = frontier.next(incumbent).unwrap();
            assert!(f16.description.starts_with("graph=0,"));
            assert!(f16.description.contains("matrix_inputs=F16"));
            frontier.observe(f16.origin.unwrap(), qualified);
            assert_eq!(frontier.early_programs_remaining(incumbent), 1);
            let bf16 = frontier.next(incumbent).unwrap();
            assert!(bf16.description.contains("matrix_inputs=BF16"));
            assert_eq!(frontier.early_programs_remaining(incumbent), 0);
            let split = frontier.next(incumbent).unwrap();
            assert!(split.description.contains("conv_dw_splits=4"));
            // Early arithmetic must not remove ordinary graph/layout choices.
            assert!(
                std::iter::from_fn(|| frontier.next(incumbent))
                    .any(|p| p.description.starts_with("graph=1,"))
            );
        }
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
        let mut caps = blade_graphics::Capabilities {
            shader_float16: true,
            shader_bfloat16: true,
            subgroup_size: 32,
            ..Default::default()
        };
        caps.cooperative_matrix.f16_f32_shapes.push([8, 16, 16]);
        caps.cooperative_matrix.bf16_f32_shapes.push([8, 16, 16]);
        frontier.enable_matrix_inputs(&[BF16, F16], &caps);
        let mut incumbent = None;
        let mut reached = collections::HashSet::new();
        while let Some(program) = frontier.next(incumbent) {
            let origin = program.origin.unwrap();
            let choice = frontier.choices[origin.choice];
            if choice.matrix_inputs.is_some()
                && !reached
                    .iter()
                    .any(|&(dense, _, _, _)| matches!(dense, Some(_)))
            {
                assert_eq!(
                    choice.winograd,
                    Some(F16),
                    "retain the incumbent for the first dense trial"
                );
                assert!(choice.f16_attention, "retain winning attention arithmetic");
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
        for dense in [None, Some(F16), Some(BF16)] {
            for winograd in [None, Some(F16), Some(BF16)] {
                for vector_only in
                    [false, true]
                        .into_iter()
                        .take(if dense.is_some() { 2 } else { 1 })
                {
                    for f16_attention in [false, true] {
                        assert!(
                            reached.contains(&(dense, vector_only, winograd, f16_attention)),
                            "{dense:?}, vector_only={vector_only}, {winograd:?}, f16_attention={f16_attention}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn rectangular_matrix_candidates_require_advertised_arithmetic() {
        let mut frontier = matrix_frontier();
        let mut caps = blade_graphics::Capabilities {
            shader_float16: true,
            ..Default::default()
        };
        caps.cooperative_matrix.f16_f32_shapes.push([8, 16, 16]);
        frontier.enable_matrix_inputs(&[crate::DType::F16, crate::DType::BF16], &caps);
        assert_eq!(
            measure::Frontier::early_programs_remaining(&frontier, None),
            0
        );
        let mut visited = Vec::new();
        while let Some(program) = measure::Frontier::next(&mut frontier, None) {
            visited.push(program);
        }
        assert!(!visited[0].description.contains("matrix_inputs="));
        assert!(visited[1].description.contains("matrix_inputs=F16"));
        assert!(visited.iter().all(|p| !p.description.contains("BF16")));
        let mut plan = visited[0].plan.clone();
        for d in &mut plan.dispatches {
            d.requires_full_precision = true;
        }
        assert_eq!(
            fill_matrix_candidates(&mut plan, crate::DType::F16, false),
            0
        );
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
            let mut caps = blade_graphics::Capabilities {
                shader_float16: true,
                shader_bfloat16: true,
                ..Default::default()
            };
            caps.cooperative_matrix.f16_f32_shapes.push([16, 16, 16]);
            caps.cooperative_matrix.bf16_f32_shapes.push([16, 16, 16]);
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
    fn matrix_candidates_replace_automatic_cooperative_arithmetic() {
        let frontier = matrix_frontier();
        let mut plan = frontier.seeds[0].plan.clone();
        let index = plan
            .dispatches
            .iter()
            .position(|d| d.mnk().is_some())
            .unwrap();
        plan.dispatches[index].kernel = compile::Kernel::Cooperative;
        let mut protected = plan.clone();
        protected.dispatches[index].requires_full_precision = true;
        assert_eq!(
            fill_matrix_candidates(&mut protected, crate::DType::BF16, false),
            0
        );
        assert_eq!(
            fill_matrix_candidates(&mut plan, crate::DType::BF16, false),
            1
        );
        assert!(matches!(plan.dispatches[index].kernel,
            compile::Kernel::CooperativeTyped { types, splits: 1 }
                if types.multiply == crate::DType::BF16 && types.a == crate::DType::F32
                    && types.b == crate::DType::F32 && types.output == crate::DType::F32));
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

        let gpu = crate::reference::gpu::shared_context();
        let a: Vec<f32> = (0..32 * 128)
            .map(|i| (i % 7) as f32 / 16.0 - 0.1875)
            .collect();
        let b: Vec<f32> = (0..128 * 32).map(|i| (i % 5) as f32 / 8.0 - 0.25).collect();
        for dtype in [crate::DType::F16, crate::DType::BF16] {
            if !runtime::typed_matrix_supported(&gpu, dtype) {
                continue;
            }
            let mut plan = original.clone();
            assert_eq!(fill_matrix_candidates(&mut plan, dtype, true), 1);
            let mut session = Session::with_context_opts(
                plan,
                gpu.clone(),
                crate::SessionOptions {
                    poison: true,
                    ..Default::default()
                },
            );
            session.set_input("a", &a);
            session.set_parameter("b", &b);
            session.step();
            session.wait();
            for (i, got) in session.read_output(32 * 32).into_iter().enumerate() {
                let expected: f64 = (0..128)
                    .map(|k| f64::from(a[i / 32 * 128 + k]) * f64::from(b[k * 32 + i % 32]))
                    .sum();
                assert_eq!(f64::from(got), expected, "{dtype:?} output {i}");
            }
        }
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
        let gpu = crate::reference::gpu::shared_context();
        if !runtime::typed_matrix_supported(&gpu, crate::DType::BF16) {
            return;
        }
        let mut graph = Graph::new();
        let x = graph.input("x", &[2 * 3 * 7 * 9]);
        let w = graph.parameter("w", &[5 * 3 * 3 * 3]);
        let y = graph.conv2d(x, w, 2, 3, 7, 9, 5, 3, 3, 2, 1);
        graph.set_outputs(vec![y]);
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
        let (_, report) = build_measured(
            &graph,
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
                s.set_input("x", &input);
                s.set_parameter("w", &weights);
                Ok(())
            },
            |s| {
                let actual = s.read_output(expected.len());
                if actual == expected {
                    Ok(())
                } else {
                    Err("checked convolution differs from exact CPU result".into())
                }
            },
        )
        .unwrap();
        assert!(
            report
                .trials
                .iter()
                .any(|t| t.description.contains("matrix_inputs=BF16") && t.outcome.qualified)
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
        let mut caps = blade_graphics::Capabilities {
            shader_float16: true,
            ..Default::default()
        };
        caps.cooperative_matrix.f16_f32_shapes.push([8, 16, 16]);
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
        for coop in [
            crate::CoopPolicy::Disabled,
            crate::CoopPolicy::NativeF32,
            crate::CoopPolicy::AllowF16,
        ] {
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
            if coop == crate::CoopPolicy::AllowF16 {
                assert_eq!(report.matrix_inputs, [crate::DType::F16]);
                if runtime::typed_matrix_supported(&gpu, crate::DType::F16) {
                    assert!(report.unsupported_matrix_inputs.is_empty());
                    assert!(report.trials.iter().any(|t| {
                        t.description.contains("matrix_inputs=F16") && t.outcome.qualified
                    }));
                } else {
                    assert_eq!(report.unsupported_matrix_inputs, [crate::DType::F16]);
                    assert!(
                        report
                            .trials
                            .iter()
                            .all(|t| !t.description.contains("matrix_inputs="))
                    );
                }
            } else {
                assert!(
                    report
                        .trials
                        .iter()
                        .all(|t| !t.description.contains("coherent_weight_cache="))
                );
                assert!(report.matrix_inputs.is_empty());
                assert!(report.unsupported_matrix_inputs.is_empty());
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
    fn fusion_alternatives_lead_the_physical_cover() {
        let baseline = AxisChoice {
            attention: (0, None),
            fuse_dispatches: true,
            matrix_inputs: None,
            matrix_storage: false,
            vector_convolutions_only: false,
            winograd: None,
            f16_attention: false,
            forward_products: false,
            gradient_inputs: None,
            bf16_nn_gradients: false,
        };
        let unfused = AxisChoice {
            fuse_dispatches: false,
            ..baseline
        };
        let cover = early_physical_cover(baseline, &[]);
        assert_eq!(cover, vec![unfused]);
        // Choice 1 is the unfused cover, measured on the incumbent's form
        // right after the first layouts and one more form.
        let mut order = CoordinateOrder::new(8, 2);
        let visits: Vec<_> = std::iter::from_fn(|| order.next((0, 0))).collect();
        assert!(visits[..5].contains(&(0, 1)));
        for seed in 0..8 {
            for choice in 0..2 {
                assert_eq!(
                    visits
                        .iter()
                        .filter(|&&entry| entry == (seed, choice))
                        .count(),
                    1
                );
            }
        }
        assert!(order.exhausted());
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
                backward_rank: 0,
                graph: graph.clone(),
                options: options.clone(),
                plan: plan.clone(),
                description: format!("graph={index}"),
            })
            .collect();
        // Reserve half the ordinary 64-program allowance for submission probes.
        // Even with a full graph frontier, distinct attention layouts must fit.
        let mut frontier = implementations(seeds, caps, 65536, 512 << 20);
        let programs: Vec<_> = (0..BuildSearchOptions::default().max_programs / 2)
            .map_while(|_| measure::Frontier::next(&mut frontier, None))
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
        // The graph1 Whisper winner was absent from the old 64-program
        // diagonal search. Reserve submission probes and still reach it.
        assert!(programs.iter().any(|p| {
            p.origin.is_some_and(|origin| origin.form == 0)
                && p.plan.knobs.flash_ept_cap == 32
                && p.plan.knobs.flash
                    == crate::codegen::FlashAttentionShape {
                        threads: 128,
                        keys: 16,
                        interleave: false,
                    }
        }));
        // The B570 winner needs both a narrower lane vector and interleaving.
        assert!(programs.iter().any(|p| {
            p.origin.is_some_and(|origin| origin.form == 0)
                && p.plan.knobs.flash_ept_cap == 16
                && p.plan.knobs.flash
                    == crate::codegen::FlashAttentionShape {
                        threads: 256,
                        keys: 8,
                        interleave: true,
                    }
        }));
    }

    #[test]
    fn attention_exploration_inherits_winning_fusion() {
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
                backward_rank: 0,
                graph: graph.clone(),
                options: options.clone(),
                plan: plan.clone(),
                description: format!("graph={index}"),
            })
            .collect();
        let mut frontier = implementations(seeds, caps, 65536, 512 << 20);
        let choice = frontier
            .choices
            .iter()
            .position(|c| c.attention == (0, None) && !c.fuse_dispatches)
            .unwrap();
        let winner = Some(measure::Origin { form: 0, choice });
        let mut layouts = collections::HashSet::new();
        // An unfused incumbent must get the same early attention coverage;
        // ranking every fused combination first would only help one policy.
        for _ in 0..BuildSearchOptions::default().max_programs / 2 {
            let Some(program) = measure::Frontier::next(&mut frontier, winner) else {
                break;
            };
            let choice = frontier.choices[program.origin.unwrap().choice];
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
        assert_eq!(order.next((1, 0)), Some((1, 4)));
        assert_eq!(order.next((1, 4)), Some((1, 5)));
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
        for seeds in 0..8 {
            for choices in 0..12 {
                let mut order = CoordinateOrder::new(seeds, choices);
                if choices > 1 {
                    order.priority_attention = Some(1);
                }
                let mut incumbent = (0, 0);
                let mut visited = collections::HashSet::new();
                while let Some(entry) = order.next(incumbent) {
                    assert!(entry.0 < seeds && entry.1 < choices);
                    assert!(visited.insert(entry));
                    if visited.len().is_multiple_of(3) {
                        order.form_order.reverse();
                        incumbent = entry;
                    }
                }
                assert_eq!(visited.len(), seeds * choices);
                assert!(order.exhausted());
                assert!(order.next(incumbent).is_none());
            }
        }
    }

    #[test]
    fn early_arithmetic_order_exhausts_each_pair_as_winners_change() {
        for seeds in 0..5 {
            for choices in 0..8 {
                let mut order = CoordinateOrder::new(seeds, choices);
                order.arithmetic_layouts = 1;
                order.priority_choices.extend((1..choices).rev());
                let mut incumbent = (0, 0);
                let mut visited = collections::HashSet::new();
                while let Some(entry) = order.next(incumbent) {
                    assert!(entry.0 < seeds && entry.1 < choices);
                    assert!(visited.insert(entry));
                    if visited.len().is_multiple_of(3) {
                        incumbent = entry;
                    }
                }
                assert_eq!(visited.len(), seeds * choices);
                assert!(order.exhausted());
            }
        }
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
                    qualification_runs: 1,
                    final_qualification_runs: 0,
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
