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
const ARITHMETIC_LAYOUTS: usize = 5;

#[derive(Clone, Serialize)]
pub struct BuildSearchOptions {
    /// Optional reduced multiplication types for dense products and forward convolutions.
    /// Storage, outputs and precision-protected derivatives remain F32. Each
    /// complete program must pass the caller's accuracy and timing gates.
    /// Unsupported device types are omitted; the ordinary program stays first.
    pub matrix_inputs: Vec<crate::DType>,
    /// Fresh executions checked at each qualification checkpoint, outside
    /// timing. More than one lets a caller verify a program's repeatability
    /// independently of numerical differences between candidate programs.
    /// Persistent writes and representative inputs are reset before each run.
    /// Repeated checks require `refresh_inputs` to upload identical data.
    pub qualification_runs: u32,
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
            matrix_inputs: Vec::new(),
            qualification_runs: 1,
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
    /// when nothing ran after its own check.
    pub final_qualification_time: Duration,
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
/// measurement. The selected program is checked before returning only if another
/// challenger ran since its last check. A failing challenger is discarded; a
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
/// After the first layouts, alternate layout exploration with submission probes
/// on the incumbent. A winning layout reopens those probes; later layouts inherit
/// its measured count rather than expanding a layout/submission cross product.
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
    if options
        .matrix_inputs
        .iter()
        .any(|ty| !matches!(ty, crate::DType::F16 | crate::DType::BF16))
        || (!options.matrix_inputs.is_empty()
            && !matches!(
                cfg.runtime.coop,
                crate::CoopPolicy::Auto | crate::CoopPolicy::AllowF16
            ))
    {
        return Err(
            "reduced matrix candidates require F16/BF16 and a permissive cooperative policy".into(),
        );
    }
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
        let space = backward_layouts(
            graphs,
            cfg.optimize,
            target,
            options.max_graphs,
            start + options.max_time,
            &mut skipped_regions,
            &mut backward_ranks,
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
                backward_rank: backward_ranks[index],
                description: format!("graph={index}"),
                plan,
                graph,
                options: cfg.options.clone(),
            });
        }
    }
    let (matrix_inputs, unsupported_matrix_inputs): (Vec<_>, Vec<_>) = options
        .matrix_inputs
        .iter()
        .copied()
        .partition(|&ty| runtime::typed_matrix_supported(&gpu, ty));
    let mut programs = implementations(seeds, caps, shared_memory_bytes, options.max_plan_bytes);
    programs.enable_matrix_inputs(&matrix_inputs, &gpu.capabilities());
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
            unsupported_matrix_inputs,
            preparation_time,
            ..Default::default()
        },
        start,
        (initialize, refresh_inputs, qualify),
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
    ranks: &mut Vec<usize>,
) -> optimize::search::SearchSpace {
    ranks.clear();
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
    optimize::search::SearchSpace {
        candidates,
        truncated: truncated || layouts.iter().flatten().any(|pending| !pending.is_empty()),
    }
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
/// seed and choice lists). The first layouts run with the baseline choice.
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
        }
    }

    /// The next (form, choice) to measure, given the incumbent's.
    fn next(&mut self, incumbent: (usize, usize)) -> Option<(usize, usize)> {
        if self.seeds == 0 || self.choices == 0 {
            return None;
        }
        // Reach several ordinary graph/derivative combinations before paying
        // for another arithmetic family. Its tuning and full-gradient checks
        // can otherwise displace a larger layout win under the same deadline.
        if self
            .visited_forms
            .iter()
            .filter(|&&visited| visited)
            .count()
            >= self.seeds.min(ARITHMETIC_LAYOUTS)
        {
            while let Some(choice) = self.priority_choices.pop_front() {
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
        let physical = self
            .choice_order
            .iter()
            .copied()
            .map(|c| (form, c))
            .find(|entry| !self.tried.contains(entry));
        // Unvisited forms first; once every form has run, revisit them with the
        // incumbent's choice.
        let graph = self
            .form_order
            .iter()
            .copied()
            .find(|&s| !self.visited_forms[s])
            .map(|s| (s, choice))
            .filter(|entry| !self.tried.contains(entry))
            .or_else(|| {
                (0..self.seeds)
                    .map(|s| (s, choice))
                    .find(|entry| !self.tried.contains(entry))
            });
        // Both axes exhausted around the incumbent: the remaining pairs,
        // diagonally, for budgets large enough to cover them.
        let rest = || {
            (0..self.seeds + self.choices - 1)
                .flat_map(|rank| (0..self.choices.min(rank + 1)).map(move |c| (rank - c, c)))
                .find(|&(s, c)| s < self.seeds && !self.tried.contains(&(s, c)))
        };
        let entry = if self.physical_next {
            physical.or(graph)
        } else {
            graph.or(physical)
        }
        .or_else(rest)?;
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
            })
        })
        .filter(|&choice| seen.insert(choice))
        .collect();
    let choices: Vec<_> = std::iter::once(baseline).chain(cover).chain(tail).collect();
    let order = CoordinateOrder::new(seeds.len(), choices.len());
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
}

impl Implementations {
    fn enable_matrix_inputs(
        &mut self,
        inputs: &[crate::DType],
        caps: &blade_graphics::Capabilities,
    ) {
        let ordinary = self.choices.clone();
        let mut seen = collections::HashSet::new();
        for &multiply in inputs {
            let shapes = match multiply {
                crate::DType::F16 if caps.shader_float16 => &caps.cooperative_matrix.f16_f32_shapes,
                crate::DType::BF16 if caps.shader_bfloat16 => {
                    &caps.cooperative_matrix.bf16_f32_shapes
                }
                _ => continue,
            };
            if !seen.insert(multiply)
                || crate::codegen::checked_cooperative_shape(shapes).is_none()
                || !self
                    .seeds
                    .iter()
                    .any(|s| matrix_candidates(&mut s.plan.clone(), multiply) > 0)
            {
                continue;
            }
            self.order.priority_choices.push_back(self.choices.len());
            self.choices
                .extend(ordinary.iter().map(|choice| AxisChoice {
                    matrix_inputs: Some(multiply),
                    ..*choice
                }));
        }
        self.order.choices = self.choices.len();
        self.order.choice_order = (0..self.choices.len()).collect();
    }
}

/// Checked typed products preserve the F32 binding contract, including tails.
/// Conversion happens inside each timed dispatch. This also reaches rectangular
/// subgroup shapes and workgroup tuning without the legacy square-tile heuristic.
fn matrix_candidates(plan: &mut ExecutionPlan, multiply: crate::DType) -> usize {
    let mut count = 0;
    for d in &mut plan.dispatches {
        if matches!(
            d.shader,
            compile::ShaderEntry::Conv2dGemm
                | compile::ShaderEntry::Conv2dGemmSmall
                | compile::ShaderEntry::Conv2dGemm16
        ) && !d.requires_full_precision
            && !d.schedule_locked
            && d.matmul_epilogue.is_none()
            && d.extra_outputs.is_empty()
            && d.weight_format == compile::WeightFormat::F32
        {
            let p = &d.params;
            d.shader = compile::ShaderEntry::Conv2dGemmCoopGen(p[5], p[6], p[7]);
            d.kernel = compile::Kernel::CooperativeTyped {
                types: crate::codegen::MatrixTypes {
                    multiply,
                    ..crate::codegen::MatrixTypes::F32_AS_F16
                },
                splits: 1,
            };
            d.workgroups = [p[4].div_ceil(32), (p[9] * p[10]).div_ceil(32), p[0]];
            count += 1;
            continue;
        }

        if !matches!(
            d.shader,
            compile::ShaderEntry::MatMul
                | compile::ShaderEntry::MatMulAT
                | compile::ShaderEntry::MatMulBT
                | compile::ShaderEntry::FusedMatMulAdd
                | compile::ShaderEntry::FusedMatMulATAdd
                | compile::ShaderEntry::FusedMatMulBTAdd
        ) || !matches!(
            d.kernel,
            compile::Kernel::Default | compile::Kernel::SmallTile
        ) || d.requires_full_precision
            || d.schedule_locked
            || d.horizontal_batch > 1
            || d.weight_format != compile::WeightFormat::F32
            || d.matmul_prologue.is_some()
            || d.matmul_epilogue.is_some()
        {
            continue;
        }
        let Some((m, n, k)) = d.mnk() else { continue };
        if m <= 1 || n == 0 || k == 0 {
            continue;
        }
        d.kernel = compile::Kernel::CooperativeTyped {
            types: crate::codegen::MatrixTypes {
                multiply,
                ..crate::codegen::MatrixTypes::F32_AS_F16
            },
            splits: 1,
        };
        d.workgroups = [m.div_ceil(32), n.div_ceil(32), 1];
        count += 1;
    }
    count
}

impl measure::Frontier for Implementations {
    fn next(&mut self, incumbent: Option<measure::Origin>) -> Option<measure::Program> {
        loop {
            if let Some(plan) = self.split_plans.pop() {
                return Some(plan);
            }
            if !self.queued_splits && !self.order.tried.is_empty() {
                self.queued_splits = true;
                // After the ordinary incumbent. Popped in reverse, so 4 splits
                // is measured before 8.
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
            // Test the incumbent attention layout's other fusion setting, then
            // explore attention with its winning fusion setting. Expanding both
            // fusion settings for every losing layout spends the bounded search
            // before it reaches later attention layouts. Keep every combination
            // available after these coordinate neighbors.
            self.order.choice_order.sort_by_key(|&index| {
                let choice = self.choices[index];
                (
                    choice.matrix_inputs != active.matrix_inputs,
                    choice.attention != active.attention,
                    choice.fuse_dispatches != active.fuse_dispatches,
                    index,
                )
            });
            let (seed_index, choice_index) = self.order.next(incumbent)?;
            let seed = &self.seeds[seed_index];
            let AxisChoice {
                attention: (splits, flash),
                fuse_dispatches,
                matrix_inputs,
            } = self.choices[choice_index];
            let mut plan = if splits == 0
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
                let plan = compile::compile_with_caps(
                    &seed.graph,
                    &options,
                    self.caps,
                    self.shared_memory_bytes,
                );
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
            if let Some(multiply) = matrix_inputs {
                if matrix_candidates(&mut plan, multiply) == 0 {
                    continue;
                }
            }
            let mut description = format!(
                "{}, dispatch_fusion={fuse_dispatches}, attention_splits={splits}, flash={flash:?}",
                seed.description
            );
            if let Some(dtype) = matrix_inputs {
                description.push_str(&format!(", matrix_inputs={dtype:?}"));
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
    fn rectangular_matrix_candidates_require_advertised_arithmetic() {
        let mut frontier = matrix_frontier();
        let mut caps = blade_graphics::Capabilities {
            shader_float16: true,
            ..Default::default()
        };
        caps.cooperative_matrix.f16_f32_shapes.push([8, 16, 16]);
        frontier.enable_matrix_inputs(&[crate::DType::F16, crate::DType::BF16], &caps);
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
        assert_eq!(matrix_candidates(&mut plan, crate::DType::F16), 0);
    }

    #[test]
    fn bf16_candidate_survives_fp16_overflow_without_relaxing_the_gate() {
        let gpu = crate::reference::gpu::shared_context();
        if !gpu.capabilities().shader_bfloat16 {
            return;
        }
        let f16_supported = runtime::typed_matrix_supported(&gpu, crate::DType::F16);
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
                matrix_inputs: vec![crate::DType::F16, crate::DType::BF16],
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
        if f16_supported {
            assert!(
                report
                    .trials
                    .iter()
                    .any(|t| t.description.contains("matrix_inputs=F16") && !t.outcome.qualified)
            );
        } else {
            assert!(
                report
                    .unsupported_matrix_inputs
                    .contains(&crate::DType::F16)
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
        let types: Vec<_> = [crate::DType::F16, crate::DType::BF16]
            .into_iter()
            .filter(|&ty| runtime::typed_matrix_supported(&gpu, ty))
            .collect();
        if types.is_empty() {
            return;
        }
        let mut graph = Graph::new();
        let x = graph.input("x", &[2 * 3 * 7 * 9]);
        let w = graph.parameter("w", &[5 * 3 * 3 * 3]);
        let y = graph.conv2d(x, w, 2, 3, 7, 9, 5, 3, 3, 2, 1);
        graph.set_outputs(vec![y]);
        let input: Vec<f32> = (0..2 * 3 * 7 * 9).map(|i| (i % 11) as f32 / 16.0).collect();
        let weights: Vec<f32> = (0..5 * 3 * 3 * 3).map(|i| (i % 5) as f32 / 16.0).collect();
        let mut expected = Vec::new();
        for batch in 0..2 {
            for channel in 0..5 {
                for oh in 0..4 {
                    for ow in 0..5 {
                        let mut total = 0.0;
                        for ic in 0..3 {
                            for kh in 0..3 {
                                for kw in 0..3 {
                                    let ih = oh * 2 + kh - 1i32;
                                    let iw = ow * 2 + kw - 1i32;
                                    if (0..7).contains(&ih) && (0..9).contains(&iw) {
                                        total += input[((batch * 3 + ic) * 7 + ih as usize) * 9
                                            + iw as usize]
                                            * weights[((channel * 3 + ic) * 3 + kh as usize) * 3
                                                + kw as usize];
                                    }
                                }
                            }
                        }
                        expected.push(total);
                    }
                }
            }
        }
        let (_, report) = build_measured(
            &graph,
            SessionConfig {
                gpu: Some(gpu),
                mode: crate::Mode::Inference,
                ..Default::default()
            },
            BuildSearchOptions {
                matrix_inputs: types.clone(),
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
        for ty in types {
            assert!(
                report
                    .trials
                    .iter()
                    .any(|t| t.description.contains(&format!("matrix_inputs={ty:?}"))
                        && t.outcome.qualified)
            );
        }
    }

    #[test]
    fn reduced_candidates_cannot_bypass_a_strict_runtime_policy() {
        for coop in [crate::CoopPolicy::Disabled, crate::CoopPolicy::NativeF32] {
            let result = build_measured(
                &Graph::new(),
                SessionConfig {
                    runtime: crate::SessionOptions {
                        coop,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                BuildSearchOptions {
                    matrix_inputs: vec![crate::DType::BF16],
                    ..Default::default()
                },
                |_, _| panic!("invalid arithmetic must fail before initialization"),
                |_| panic!("invalid arithmetic must fail before execution"),
            );
            assert!(
                result
                    .err()
                    .unwrap()
                    .contains("permissive cooperative policy")
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
        order.priority_choices.extend([4, 5]);
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
                    matrix_inputs: Vec::new(),
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
