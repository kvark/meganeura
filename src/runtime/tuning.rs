use super::{Gpu, Pipelines, Session, Variant, ensure_device_memory_budget};
use crate::compile::{BufferRef, Dispatch, ShaderEntry};
use crate::tune::{
    MatmulTile, TuneClass, TuneDecision, TuneError, TuneOptions, TuneOutcome,
    TuneQualificationTimes, TuneReport, TuneStaging, decide, measure_pairs,
};
use blade_graphics as bg;
use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

struct PhaseTimer<'a> {
    start: Instant,
    elapsed: &'a mut Option<Duration>,
}

impl<'a> PhaseTimer<'a> {
    fn new(elapsed: &'a mut Option<Duration>) -> Self {
        Self {
            start: Instant::now(),
            elapsed,
        }
    }
}

impl Drop for PhaseTimer<'_> {
    fn drop(&mut self) {
        *self.elapsed.get_or_insert(Duration::ZERO) += self.start.elapsed();
    }
}

impl Pipelines {
    fn ensure_tune_tile(
        &mut self,
        gpu: &Gpu,
        entry: &ShaderEntry,
        tile: MatmulTile,
    ) -> Result<(), String> {
        let key = tile_variant(entry, tile);
        if self.map.contains_key(&key) {
            return Ok(());
        }
        let module = match tile {
            MatmulTile::Tile32 => crate::codegen::generate_module_small(entry.shader_group()),
            MatmulTile::Tile64 => crate::codegen::generate_module(entry.shader_group()),
            MatmulTile::CooperativeF32 { .. } => crate::codegen::generate_module_coop(
                entry.shader_group(),
                &tile.coop_config().expect("cooperative candidate"),
            ),
        };
        let shader = gpu
            .try_create_shader(bg::ShaderDesc {
                source: &module.source,
                naga_module: Some(module.module),
            })
            .map_err(|error| error.to_string())?;
        let layout = super::shader_data_layout(entry);
        let pipeline = gpu.create_compute_pipeline(bg::ComputePipelineDesc {
            name: entry.entry_point(),
            data_layouts: &[&layout],
            compute: shader.at(entry.entry_point()),
        });
        self.map.insert(key, pipeline);
        Ok(())
    }
}

fn tile_variant(entry: &ShaderEntry, tile: MatmulTile) -> Variant {
    match tile {
        MatmulTile::Tile32 => Variant::SmallTile(entry.clone()),
        MatmulTile::Tile64 => Variant::Scalar(entry.clone()),
        MatmulTile::CooperativeF32 { .. } => Variant::Coop(entry.clone()),
    }
}

struct SearchClass {
    key: TuneClass,
    initial: MatmulTile,
    members: Vec<usize>,
    challengers: Vec<MatmulTile>,
}

struct SelectionSwap {
    index: usize,
    class: TuneClass,
    left: MatmulTile,
    right: MatmulTile,
}

fn selection_swaps(
    left: &[Dispatch],
    right: &[Dispatch],
    left_classes: &[SearchClass],
    right_classes: &[SearchClass],
) -> Result<Vec<SelectionSwap>, TuneError> {
    if left.len() != right.len() {
        return Err(TuneError("tuning swap requires matching dispatch layouts"));
    }
    let by_index = |classes: &[SearchClass]| {
        let mut indices = vec![None; left.len()];
        for (class_index, class) in classes.iter().enumerate() {
            for &index in &class.members {
                indices[index] = Some(class_index);
            }
        }
        indices
    };
    let left_indices = by_index(left_classes);
    let right_indices = by_index(right_classes);
    let mut swaps = Vec::new();
    for (index, (a, b)) in left.iter().zip(right).enumerate() {
        if a == b {
            continue;
        }
        let (Some(ai), Some(bi)) = (left_indices[index], right_indices[index]) else {
            return Err(TuneError(
                "tuning swap only permits eligible f32 tile changes",
            ));
        };
        let (ac, bc) = (&left_classes[ai], &right_classes[bi]);
        if ac.key != bc.key || !ac.initial.fits(&bc.key) || !bc.initial.fits(&ac.key) {
            return Err(TuneError("tuning swap requires identical legal classes"));
        }
        let (mut normalized_a, mut normalized_b) = (a.clone(), b.clone());
        ac.initial.apply(&mut normalized_a, &ac.key);
        ac.initial.apply(&mut normalized_b, &bc.key);
        if normalized_a != normalized_b {
            return Err(TuneError(
                "tuning swap cannot change bindings, fusion or provenance",
            ));
        }
        swaps.push(SelectionSwap {
            index,
            class: ac.key.clone(),
            left: ac.initial,
            right: bc.initial,
        });
    }
    Ok(swaps)
}

fn collect_classes(
    plan: &crate::compile::ExecutionPlan,
    alias: &crate::memplan::AliasPlan,
    coop_config: Option<&crate::codegen::CoopConfig>,
) -> (Vec<SearchClass>, usize) {
    let mut classes: Vec<SearchClass> = Vec::new();
    let mut indices = HashMap::new();
    let mut excluded = 0;
    for (index, dispatch) in plan.dispatches.iter().enumerate() {
        let Some(mut key) = TuneClass::from_dispatch(dispatch, coop_config) else {
            excluded += 1;
            continue;
        };
        let bindings: Vec<_> = dispatch
            .input_buffers
            .iter()
            .chain(std::iter::once(&dispatch.output_buffer))
            .collect();
        let physical: Vec<_> = bindings.iter().map(|b| alias.map[b.0 as usize]).collect();
        // Do not transfer isolated timings to overlapping bindings, even if
        // readonly input aliasing is legal. It changes the cache working set.
        if physical
            .iter()
            .enumerate()
            .any(|(i, p)| physical[..i].contains(p))
        {
            excluded += 1;
            continue;
        }
        for (i, &p) in physical.iter().enumerate() {
            let slot = if i + 1 == physical.len() { 3 } else { i };
            key.device_local[slot] = alias.device_local[p];
        }
        key.binding_bytes = bindings
            .iter()
            .map(|b| plan.buffers[b.0 as usize])
            .collect();
        let initial = MatmulTile::selected(dispatch, coop_config).expect("checked class geometry");
        if !initial.fits(&key) {
            excluded += 1;
            continue;
        }
        let challengers = key.challengers(initial, coop_config);
        let next_index = classes.len();
        let class_index = *indices.entry((key.clone(), initial)).or_insert(next_index);
        if class_index == next_index {
            classes.push(SearchClass {
                key,
                initial,
                members: Vec::new(),
                challengers,
            });
        }
        classes[class_index].members.push(index);
    }
    // A structural prior, not a predicted runtime: favor repeated, large
    // contractions. Stable ties keep the source dispatch order deterministic.
    classes.sort_by_key(|c| {
        std::cmp::Reverse(
            c.members.len() as u128 * c.key.m as u128 * c.key.n as u128 * c.key.k as u128,
        )
    });
    (classes, excluded)
}

impl Session {
    /// Exchange eligible f32 matmul choices without exchanging tensor state.
    ///
    /// This supports controlled crossover experiments, not automatic confirmation,
    /// arbitrary report application, or persistent/cross-device winner reuse.
    /// Sessions must share the exact GPU context, cooperative policy after smoke
    /// tests, allocation layout, generator knobs and scheduling groups. Dispatches
    /// may differ only in complete legal matmul tile selection. Callers remain
    /// responsible for matching inputs, weights, optimizer settings and training age.
    ///
    /// Preflights the entire swap, prepares missing pipelines, then waits for both
    /// sessions before installing choices. No step, upload, readback or optimizer
    /// update occurs. On error choices and tensors remain unchanged; successfully
    /// prepared pipelines can remain cached. Preparation and waits belong outside
    /// whole-step timers. Returns the changed dispatch count per session.
    pub fn swap_tuning_with(&mut self, other: &mut Self) -> Result<usize, TuneError> {
        if !std::sync::Arc::ptr_eq(&self.gpu, &other.gpu)
            || self.coop_config != other.coop_config
            || self.plan.buffers != other.plan.buffers
            || self.plan.knobs != other.plan.knobs
            || self.alias.map != other.alias.map
            || self.alias.sizes != other.alias.sizes
            || self.alias.device_local != other.alias.device_local
            || self.groups != other.groups
        {
            return Err(TuneError(
                "tuning swap requires compatible sessions on one context",
            ));
        }
        let left_classes = collect_classes(&self.plan, &self.alias, self.coop_config.as_ref()).0;
        let right_classes =
            collect_classes(&other.plan, &other.alias, other.coop_config.as_ref()).0;
        let swaps = selection_swaps(
            &self.plan.dispatches,
            &other.plan.dispatches,
            &left_classes,
            &right_classes,
        )?;
        for swap in &swaps {
            for (session, tile) in [(&mut *self, swap.right), (&mut *other, swap.left)] {
                if let Err(error) =
                    session
                        .pipelines
                        .ensure_tune_tile(&session.gpu, &swap.class.shader, tile)
                {
                    log::warn!("tuning swap pipeline preparation: {error}");
                    return Err(TuneError(
                        "tuning swap pipeline preparation failed; see log",
                    ));
                }
            }
        }
        self.wait();
        other.wait();
        for swap in &swaps {
            swap.right
                .apply(&mut self.plan.dispatches[swap.index], &swap.class);
            swap.left
                .apply(&mut other.plan.dispatches[swap.index], &swap.class);
        }
        Ok(swaps.len())
    }

    /// Bounded f32 tile search with default options; logs skips and returns
    /// per-comparison evidence. Use [`Self::tune_with`] for budgets and full reporting.
    ///
    /// Unlike the former family-wide tuner, this never calls `step()` and
    /// never reads or writes live tensor, optimizer, accumulator, or KV state.
    /// It is opt-in GPU work; do not call it while another workload is timing.
    pub fn tune(&mut self) -> Vec<TuneOutcome> {
        self.tune_with(TuneOptions::default())
            .expect("default tuning options are valid")
            .outcomes
    }

    /// Search scalar tiles and advertised, smoke-tested native-f32 cooperative
    /// matmul for exact eligible classes. Occupancy/large-shape thresholds only
    /// choose the starting implementation; they do not remove challengers.
    ///
    /// Each class uses private scratch with matching memory placement and two
    /// deterministic nonzero input patterns (including tiny f32 operands).
    /// Both candidates must agree elementwise and with sampled f64 reference
    /// dots before alternating, batched `encode+submit+wait` measurements.
    /// The result is an isolated-kernel choice, not an end-to-end speed claim.
    ///
    /// Forward MatMul+Add is supported; other prologues/epilogues, horizontal
    /// packs, f16-input cooperative, reduced-storage, GEMV and overlapping-binding
    /// dispatches are excluded. Winners live in this session, not the plan cache.
    /// Only selected dispatch geometry and pipeline resources change. No graph
    /// execution occurs, including when an optimizer or external buffer is bound.
    /// Cooperative padding must fit each binding's declared size; the live
    /// allocation/alias plan is never resized. Up to two sequential challenger
    /// comparisons per class reuse the latest fully qualified winner as the
    /// incumbent. A soft deadline may be exceeded by one in-flight operation;
    /// an incomplete comparison always retains its incumbent.
    pub fn tune_with(&mut self, options: TuneOptions) -> Result<TuneReport, TuneError> {
        options.validate()?;
        let start = Instant::now();
        let (classes, excluded_dispatches) =
            collect_classes(&self.plan, &self.alias, self.coop_config.as_ref());
        let mut report = TuneReport {
            options: options.clone(),
            eligible_classes: classes.len(),
            excluded_dispatches,
            class_limit_reached: classes.len() > options.max_classes,
            ..Default::default()
        };
        for class in classes.iter().take(options.max_classes) {
            if start.elapsed() >= options.max_time {
                report.time_budget_exhausted = true;
                break;
            }
            report.visited_classes += 1;
            let mut incumbent = class.initial;
            for &candidate in &class.challengers {
                if start.elapsed() >= options.max_time {
                    report.time_budget_exhausted = true;
                    break;
                }
                let mut outcome = TuneOutcome {
                    class: class.key.clone(),
                    dispatches: class.members.len(),
                    initial: incumbent,
                    candidate,
                    selected: incumbent,
                    decision: TuneDecision::KeepBaseline,
                    qualified: false,
                    failure: None,
                    elapsed: std::time::Duration::ZERO,
                    compile_time: std::time::Duration::ZERO,
                    phase_times: Some(Default::default()),
                    baseline_ms: Vec::new(),
                    candidate_ms: Vec::new(),
                    baseline_median_ms: None,
                    candidate_median_ms: None,
                    noise_margin_ms: None,
                };
                let class_start = Instant::now();
                self.measure_candidate(class, &options, start, &mut outcome);
                outcome.elapsed = class_start.elapsed();
                if outcome.selected != outcome.initial {
                    for &index in &class.members {
                        outcome
                            .selected
                            .apply(&mut self.plan.dispatches[index], &class.key);
                    }
                    incumbent = outcome.selected;
                }
                log::info!(
                    "tune: {:?} {}x{}x{} ({} dispatches): {:?} vs {:?} -> {:?}, {:?}, medians {:?}/{:?} ms",
                    class.key.shader,
                    class.key.m,
                    class.key.n,
                    class.key.k,
                    class.members.len(),
                    outcome.initial,
                    outcome.candidate,
                    outcome.selected,
                    outcome.decision,
                    outcome.baseline_median_ms,
                    outcome.candidate_median_ms
                );
                if let Some(ref failure) = outcome.failure {
                    log::warn!("tune: {failure}");
                }
                report.outcomes.push(outcome);
            }
        }
        report.elapsed = start.elapsed();
        report.time_budget_exhausted |= report.elapsed >= options.max_time;
        log::info!(
            "tune: {}/{} classes visited, {} comparisons; {} dispatches excluded; {:.3}s; class limit={}, time limit={}",
            report.visited_classes,
            report.eligible_classes,
            report.outcomes.len(),
            report.excluded_dispatches,
            report.elapsed.as_secs_f64(),
            report.class_limit_reached,
            report.time_budget_exhausted
        );
        Ok(report)
    }

    fn measure_candidate(
        &mut self,
        class: &SearchClass,
        options: &TuneOptions,
        start: Instant,
        outcome: &mut TuneOutcome,
    ) {
        let phases = outcome
            .phase_times
            .as_mut()
            .expect("new outcome has phase timers");
        let preparation = PhaseTimer::new(&mut phases.preparation);
        let logical_sizes = class
            .key
            .buffer_sizes()
            .expect("class collection checked extents");
        let baseline_sizes = outcome
            .initial
            .buffer_sizes(&class.key)
            .expect("legal incumbent");
        let candidate_sizes = outcome
            .candidate
            .buffer_sizes(&class.key)
            .expect("legal challenger");
        let sizes: Vec<_> = baseline_sizes
            .into_iter()
            .zip(candidate_sizes)
            .map(|(a, b)| a.max(b))
            .collect();
        let Some(bytes) = scratch_bytes(&sizes) else {
            outcome.decision = TuneDecision::ScratchLimit;
            return;
        };
        if bytes > options.max_scratch_bytes {
            outcome.decision = TuneDecision::ScratchLimit;
            return;
        }
        let memory = self.gpu.memory_stats();
        if memory.budget != 0
            && bytes as u64 > super::safe_device_memory_remaining(memory.usage, memory.budget)
        {
            outcome.decision = TuneDecision::DeviceMemoryBudget;
            return;
        }
        for tile in [outcome.initial, outcome.candidate] {
            if start.elapsed() >= options.max_time {
                outcome.decision = TuneDecision::TimeBudget;
                return;
            }
            let compile_start = Instant::now();
            let compiled = self
                .pipelines
                .ensure_tune_tile(&self.gpu, &class.key.shader, tile);
            outcome.compile_time += compile_start.elapsed();
            if let Err(error) = compiled {
                outcome.decision = TuneDecision::ShaderRejected;
                outcome.failure = Some(format!("{tile:?}: {error}"));
                return;
            }
        }
        if start.elapsed() >= options.max_time {
            outcome.decision = TuneDecision::TimeBudget;
            return;
        }
        let mut scratch = Scratch::new(&self.gpu, &class.key, &sizes, bytes, options.staging);
        let mut dispatch = self.plan.dispatches[class.members[0]].clone();
        dispatch.input_buffers = (0..sizes.len() - 1).map(|i| BufferRef(i as u32)).collect();
        dispatch.output_buffer = BufferRef((sizes.len() - 1) as u32);
        let mut variants = [dispatch.clone(), dispatch];
        outcome.initial.apply(&mut variants[0], &class.key);
        outcome.candidate.apply(&mut variants[1], &class.key);
        let pipelines = [
            &self.pipelines.map[&tile_variant(&class.key.shader, outcome.initial)],
            &self.pipelines.map[&tile_variant(&class.key.shader, outcome.candidate)],
        ];
        drop(preparation);
        phases.qualification_breakdown = Some(Default::default());
        let details = phases.qualification_breakdown.as_mut().unwrap();
        let qualification = PhaseTimer::new(&mut phases.qualification);
        for pattern in 0..2 {
            if start.elapsed() >= options.max_time {
                outcome.decision = TuneDecision::TimeBudget;
                return;
            }
            let inputs = {
                let _timer = PhaseTimer::new(&mut details.input_preparation);
                let mut inputs = test_inputs(&logical_sizes, pattern);
                for (index, data) in inputs.iter_mut().enumerate() {
                    data.resize(sizes[index] / 4, 0.0);
                }
                inputs
            };
            for (index, data) in inputs.iter().enumerate() {
                scratch.upload(index, data, Some(&mut *details));
            }
            let mut reference = Vec::new();
            for variant in 0..2 {
                if start.elapsed() >= options.max_time {
                    outcome.decision = TuneDecision::TimeBudget;
                    return;
                }
                let sentinel = {
                    let _timer = PhaseTimer::new(&mut details.input_preparation);
                    vec![f32::NAN; sizes[sizes.len() - 1] / 4]
                };
                scratch.upload(sizes.len() - 1, &sentinel, Some(&mut *details));
                drop(sentinel);
                {
                    let _timer = PhaseTimer::new(&mut details.dispatch);
                    scratch.run(pipelines[variant], &variants[variant], 1);
                }
                let output = scratch.read_output(details);
                let scale = if pattern == 0 { 1.0 } else { 1.0e-12 };
                let valid = {
                    let _timer = PhaseTimer::new(&mut details.validation);
                    qualify_output(&class.key, &inputs, &output, scale)
                        && (variant == 0 || outputs_agree(&reference, &output, scale))
                };
                if !valid {
                    outcome.decision = TuneDecision::InvalidOutput;
                    outcome.failure = Some(format!(
                        "{:?}, input pattern {pattern}: reference or cross-variant mismatch",
                        [outcome.initial, outcome.candidate][variant]
                    ));
                    return;
                }
                reference = output;
            }
        }
        outcome.qualified = true;
        drop(qualification);
        let warmup = PhaseTimer::new(&mut phases.warmup);
        // Time ordinary-magnitude data, not a zero-filled or subnormal workload.
        for (index, data) in test_inputs(&logical_sizes, 0).iter_mut().enumerate() {
            data.resize(sizes[index] / 4, 0.0);
            scratch.upload(index, data, None);
        }
        for _ in 0..options.warmup_runs {
            for variant in 0..2 {
                if start.elapsed() >= options.max_time {
                    outcome.decision = TuneDecision::TimeBudget;
                    return;
                }
                scratch.run(pipelines[variant], &variants[variant], 1);
            }
        }
        drop(warmup);
        let sampling = PhaseTimer::new(&mut phases.sampling);
        (outcome.baseline_ms, outcome.candidate_ms) =
            measure_pairs(options.sample_pairs, |alternative| {
                if start.elapsed() >= options.max_time {
                    return None;
                }
                let index = usize::from(alternative);
                let ms = scratch.run(
                    pipelines[index],
                    &variants[index],
                    options.dispatches_per_sample,
                );
                // Do not accept a last pair that only finished after the deadline.
                (start.elapsed() < options.max_time)
                    .then_some(ms / options.dispatches_per_sample as f64)
            });
        drop(sampling);
        decide(outcome, options);
    }
}

fn scratch_bytes(sizes: &[usize]) -> Option<usize> {
    sizes
        .iter()
        .try_fold(*sizes.iter().max()?, |sum, size| sum.checked_add(*size))
}

struct Scratch<'a> {
    gpu: &'a Gpu,
    buffers: Vec<bg::Buffer>,
    staging: bg::Buffer,
    output_elements: usize,
    encoder: bg::CommandEncoder,
}

impl<'a> Scratch<'a> {
    fn new(
        gpu: &'a Gpu,
        class: &TuneClass,
        sizes: &[usize],
        bytes: usize,
        staging: TuneStaging,
    ) -> Self {
        ensure_device_memory_budget(gpu, bytes, "matmul tuning scratch");
        let buffers = sizes
            .iter()
            .enumerate()
            .map(|(i, &size)| {
                let slot = if i + 1 == sizes.len() { 3 } else { i };
                gpu.create_buffer(bg::BufferDesc {
                    name: "tune_scratch",
                    size: size as u64,
                    memory: if class.device_local[slot] {
                        bg::Memory::DeviceTransient
                    } else {
                        bg::Memory::Shared
                    },
                })
            })
            .collect();
        let staging = gpu.create_buffer(bg::BufferDesc {
            name: "tune_staging",
            size: *sizes.iter().max().unwrap() as u64,
            memory: match staging {
                TuneStaging::Shared => bg::Memory::Shared,
                TuneStaging::Download => bg::Memory::Download,
            },
        });
        let encoder = gpu.create_command_encoder(bg::CommandEncoderDesc {
            name: "matmul_tune",
            buffer_count: 1,
            manual_barriers: false,
        });
        Self {
            gpu,
            buffers,
            staging,
            output_elements: class.m as usize * class.n as usize,
            encoder,
        }
    }

    fn upload(&mut self, index: usize, data: &[f32], times: Option<&mut TuneQualificationTimes>) {
        let (host, transfer) = times
            .map(|t| (&mut t.upload_host_copy, &mut t.upload_transfer))
            .unzip();
        let host = host.map(PhaseTimer::new);
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.staging.data().cast(), data.len());
        }
        drop(host);
        let _transfer = transfer.map(PhaseTimer::new);
        self.encoder.start();
        self.encoder.transfer("tune_upload").copy_buffer_to_buffer(
            self.staging.at(0),
            self.buffers[index].at(0),
            std::mem::size_of_val(data) as u64,
        );
        self.submit_wait();
    }

    fn read_output(&mut self, times: &mut TuneQualificationTimes) -> Vec<f32> {
        let transfer = PhaseTimer::new(&mut times.readback_transfer);
        self.encoder.start();
        self.encoder
            .transfer("tune_readback")
            .copy_buffer_to_buffer(
                self.buffers.last().unwrap().at(0),
                self.staging.at(0),
                (self.output_elements * 4) as u64,
            );
        self.submit_wait();
        drop(transfer);
        let _host = PhaseTimer::new(&mut times.readback_host_copy);
        unsafe {
            std::slice::from_raw_parts(self.staging.data().cast::<f32>(), self.output_elements)
                .to_vec()
        }
    }

    fn submit_wait(&mut self) {
        let sync = self.gpu.submit(&mut self.encoder);
        let _ = self.gpu.wait_for(&sync, !0);
    }

    fn run(&mut self, pipeline: &bg::ComputePipeline, dispatch: &Dispatch, repeats: u32) -> f64 {
        let start = Instant::now();
        self.encoder.start();
        for _ in 0..repeats {
            let mut pass = self.encoder.compute("tune_matmul");
            let mut pc = pass.with(pipeline);
            Session::bind_dispatch(&self.buffers, dispatch, &mut pc);
            pc.dispatch(dispatch.workgroups);
        }
        self.submit_wait();
        start.elapsed().as_secs_f64() * 1e3
    }
}

impl Drop for Scratch<'_> {
    fn drop(&mut self) {
        self.gpu.destroy_command_encoder(&mut self.encoder);
        self.gpu.destroy_buffer(self.staging);
        for buffer in self.buffers.drain(..) {
            self.gpu.destroy_buffer(buffer);
        }
    }
}

fn test_inputs(sizes: &[usize], pattern: u32) -> Vec<Vec<f32>> {
    (0..sizes.len() - 1)
        .map(|operand| {
            let scale = if pattern == 1 && operand != 1 {
                1.0e-12
            } else {
                1.0
            };
            (0..sizes[operand] / 4)
                .map(|index| {
                    let mut bits = (index as u32)
                        .wrapping_add(0x9e37_79b9u32.wrapping_mul(operand as u32 + 1));
                    bits ^= bits >> 16;
                    bits = bits.wrapping_mul(0x85eb_ca6b);
                    bits ^= bits >> 13;
                    // All operands differ, including addend; no all-zero qualification.
                    // Exercise full mantissas, not just values already exact
                    // in a reduced-mantissa matrix implementation.
                    ((bits >> 8) as f32 / 16_777_216.0 - 0.5) * scale
                })
                .collect()
        })
        .collect()
}

fn close(reference: f64, actual: f32, scale: f64) -> bool {
    actual.is_finite()
        && (reference - actual as f64).abs() <= scale * 1.0e-5 + reference.abs() * 2.0e-4
}

fn outputs_agree(reference: &[f32], actual: &[f32], scale: f64) -> bool {
    reference.len() == actual.len()
        && reference
            .iter()
            .zip(actual)
            .all(|(&a, &b)| close(a as f64, b, scale))
}

fn reference_dot(class: &TuneClass, inputs: &[Vec<f32>], row: usize, col: usize) -> f64 {
    let (m, n, k) = (class.m as usize, class.n as usize, class.k as usize);
    let mut value = if class.has_addend() {
        inputs[2][row * n + col] as f64
    } else {
        0.0
    };
    for inner in 0..k {
        let a = if class.shader == ShaderEntry::MatMulAT {
            inner * m + row
        } else {
            row * k + inner
        };
        let b = if class.shader == ShaderEntry::MatMulBT {
            col * k + inner
        } else {
            inner * n + col
        };
        value += inputs[0][a] as f64 * inputs[1][b] as f64;
    }
    value
}

fn qualify_output(class: &TuneClass, inputs: &[Vec<f32>], output: &[f32], scale: f64) -> bool {
    if output.iter().any(|x| !x.is_finite()) {
        return false;
    }
    let (m, n) = (class.m as usize, class.n as usize);
    // Explicit tile boundaries and last row/column, then scattered dots.
    let edges = [0, 1, 7, 8, 15, 16, 31, 32, 63, 64, usize::MAX];
    for i in 0..32 {
        let row = if i == edges.len() {
            m - 1
        } else if i == edges.len() + 1 {
            0
        } else if i < edges.len() {
            edges[i].min(m - 1)
        } else {
            i * 104729 % m
        };
        let col = if i == edges.len() {
            n - 1
        } else if i == edges.len() + 1 {
            0
        } else if i < edges.len() {
            edges[edges.len() - 1 - i].min(n - 1)
        } else {
            i * 130363 % n
        };
        if !close(
            reference_dot(class, inputs, row, col),
            output[row * n + col],
            scale,
        ) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swap_preflight_checks_complete_layout_before_any_change() {
        let mut graph = crate::Graph::new();
        let x = graph.input("x", &[128, 512]);
        let w = graph.parameter("w", &[512, 512]);
        let h = graph.matmul(x, w);
        let y = graph.matmul(h, w);
        graph.set_outputs(vec![y]);
        let mut left = crate::compile::compile_with(&graph, &Default::default());
        super::super::select_variants(&mut left, None, false, false);
        let alias = crate::memplan::AliasPlan::identity(&left.buffers);
        let classes = collect_classes(&left, &alias, None).0;
        let mut right = left.clone();
        for class in &classes {
            for &index in &class.members {
                MatmulTile::Tile32.apply(&mut right.dispatches[index], &class.key);
            }
        }
        let right_classes = collect_classes(&right, &alias, None).0;
        let swaps = selection_swaps(
            &left.dispatches,
            &right.dispatches,
            &classes,
            &right_classes,
        )
        .unwrap();
        assert_eq!(swaps.len(), 2);
        let original_left = left.dispatches.clone();
        let mut invalid = right.dispatches.clone();
        invalid[1].origin.push(12345);
        assert!(selection_swaps(&left.dispatches, &invalid, &classes, &right_classes).is_err());
        invalid[1] = right.dispatches[1].clone();
        invalid[1].input_buffers.swap(0, 1);
        assert!(selection_swaps(&left.dispatches, &invalid, &classes, &right_classes).is_err());
        assert_eq!(left.dispatches, original_left);
        for swap in swaps {
            swap.right
                .apply(&mut left.dispatches[swap.index], &swap.class);
            swap.left
                .apply(&mut right.dispatches[swap.index], &swap.class);
        }
        assert_eq!(right.dispatches, original_left);
        assert!(
            selection_swaps(
                &left.dispatches,
                &right.dispatches[..1],
                &classes,
                &right_classes
            )
            .is_err()
        );
    }

    #[test]
    #[ignore = "GPU state/crossover qualification; does not select winners by timing"]
    fn tuning_swap_keeps_distinct_training_states_and_next_updates() {
        use crate::{CoopPolicy, SessionConfig, SessionOptions};
        let gpu = std::sync::Arc::new(crate::init_gpu_context().unwrap());
        let mut graph = crate::Graph::new();
        let x = graph.input("x", &[33, 17]);
        let w = graph.parameter("w", &[17, 65]);
        let y = graph.matmul(x, w);
        let loss = graph.mean_all(y);
        graph.set_outputs(vec![loss]);
        let make = |input: f32, steps: usize| {
            let (mut s, _) = crate::build(
                &graph,
                SessionConfig {
                    gpu: Some(gpu.clone()),
                    runtime: SessionOptions {
                        debug: true,
                        coop: CoopPolicy::Disabled,
                        ..Default::default()
                    },
                    ..Default::default()
                },
            );
            s.set_input("x", &vec![input; 33 * 17]);
            s.set_parameter("w", &vec![0.125; 17 * 65]);
            s.set_adam(0.001, 0.9, 0.999, 1e-8);
            s.set_grad_clip_norm(0.05);
            s.set_grad_clip_every(3);
            s.set_grad_accumulate(2);
            for _ in 0..steps {
                s.step();
                s.wait();
            }
            s
        };
        let (mut a, mut a_control) = (make(0.25, 1), make(0.25, 1));
        let (mut b, mut b_control) = (make(-0.125, 2), make(-0.125, 2));
        let state = |s: &Session| {
            let mut values = vec![s.adam_step_count()];
            for (index, &bytes) in s.plan.buffers.iter().enumerate() {
                let mut data = vec![0.0; bytes / 4];
                s.read_buffer(BufferRef(index as u32), &mut data);
                values.extend(data.into_iter().map(f32::to_bits));
            }
            for (m, v) in s.read_adam_states(&["w"]) {
                values.extend(m.into_iter().chain(v).map(f32::to_bits));
            }
            values
        };
        let class = collect_classes(&b.plan, &b.alias, None).0.remove(0);
        let alternative = if class.initial == MatmulTile::Tile32 {
            MatmulTile::Tile64
        } else {
            MatmulTile::Tile32
        };
        b.pipelines
            .ensure_tune_tile(&gpu, &class.key.shader, alternative)
            .unwrap();
        for &index in &class.members {
            alternative.apply(&mut b.plan.dispatches[index], &class.key);
        }
        let (a_keys, b_keys) = (a.dispatch_pipeline_keys(), b.dispatch_pipeline_keys());
        assert_ne!(a_keys, b_keys);
        for _ in 0..4 {
            let (before_a, before_b) = (state(&a), state(&b));
            assert_ne!(before_a, before_b);
            assert!(a.swap_tuning_with(&mut b).unwrap() > 0);
            assert_eq!(state(&a), before_a);
            assert_eq!(state(&b), before_b);
            for s in [&mut a, &mut b, &mut a_control, &mut b_control] {
                s.step();
                s.wait();
            }
            assert_eq!(state(&a), state(&a_control));
            assert_eq!(state(&b), state(&b_control));
        }
        assert_eq!(a.dispatch_pipeline_keys(), a_keys);
        assert_eq!(b.dispatch_pipeline_keys(), b_keys);
        let before = state(&a);
        b.plan.knobs.flash_ept_cap += 1;
        assert!(a.swap_tuning_with(&mut b).is_err());
        assert_eq!(state(&a), before);
        assert_eq!(a.dispatch_pipeline_keys(), a_keys);
    }

    #[test]
    fn swap_preflight_obeys_native_precision_and_complete_geometry() {
        let mut graph = crate::Graph::new();
        let x = graph.input("x", &[32, 32]);
        let w = graph.parameter("w", &[32, 32]);
        let y = graph.matmul(x, w);
        graph.set_outputs(vec![y]);
        let mut left = crate::compile::compile_with(&graph, &Default::default());
        super::super::select_variants(&mut left, None, false, false);
        let alias = crate::memplan::AliasPlan::identity(&left.buffers);
        for tile_size in [8, 16] {
            let config = crate::codegen::CoopConfig {
                tile_size,
                use_f16_input: false,
                compensated: false,
            };
            let classes = collect_classes(&left, &alias, Some(&config)).0;
            let mut right = left.clone();
            MatmulTile::CooperativeF32 { tile_size }
                .apply(&mut right.dispatches[0], &classes[0].key);
            let right_classes = collect_classes(&right, &alias, Some(&config)).0;
            let swaps = selection_swaps(
                &left.dispatches,
                &right.dispatches,
                &classes,
                &right_classes,
            )
            .unwrap();
            assert_eq!(swaps.len(), 1);
            let mut selected = left.dispatches[0].clone();
            swaps[0].right.apply(&mut selected, &swaps[0].class);
            assert_eq!(selected, right.dispatches[0]);
            assert!(selected.scalar_fallback.is_some());
            let reduced = crate::codegen::CoopConfig {
                use_f16_input: true,
                ..config
            };
            assert!(
                selection_swaps(
                    &left.dispatches,
                    &right.dispatches,
                    &classes,
                    &collect_classes(&right, &alias, Some(&reduced)).0
                )
                .is_err()
            );
            right.dispatches[0].workgroups[0] += 1;
            assert!(
                selection_swaps(
                    &left.dispatches,
                    &right.dispatches,
                    &classes,
                    &collect_classes(&right, &alias, Some(&config)).0
                )
                .is_err()
            );
        }
    }

    #[test]
    fn phase_timer_records_partial_time_on_early_exit() {
        fn exit_early(elapsed: &mut Option<Duration>) -> Result<(), ()> {
            let _timer = PhaseTimer {
                start: Instant::now() - Duration::from_millis(10),
                elapsed,
            };
            Err(())
        }
        let mut elapsed = None;
        assert!(exit_early(&mut elapsed).is_err());
        assert!(elapsed.unwrap() >= Duration::from_millis(10));
        let first = elapsed.unwrap();
        assert!(exit_early(&mut elapsed).is_err());
        assert!(elapsed.unwrap() >= first + Duration::from_millis(10));
    }

    #[test]
    #[ignore = "GPU staging round-trip qualification, not a performance assertion"]
    fn tuning_staging_round_trips_all_bits_and_declared_extents() {
        let gpu = crate::init_gpu_context().unwrap();
        for (m, n, k) in [(3, 7, 5), (33, 65, 17), (2048, 1000, 1)] {
            for device_local in [false, true] {
                let class = TuneClass {
                    shader: ShaderEntry::MatMulAT,
                    m,
                    n,
                    k,
                    requires_full_precision: true,
                    device_local: [device_local; 4],
                    binding_bytes: Vec::new(),
                };
                let sizes = class.buffer_sizes().unwrap();
                let bytes = scratch_bytes(&sizes).unwrap();
                let patterns = [
                    0,
                    0x8000_0000,
                    1,
                    0x0080_0000,
                    0x3f80_0000,
                    0xbf80_0000,
                    0x7f80_0000,
                    0x7fc0_1234,
                ];
                let data: Vec<_> = (0..m as usize * n as usize)
                    .map(|i| f32::from_bits(patterns[i % patterns.len()]))
                    .collect();
                for staging in [TuneStaging::Shared, TuneStaging::Download] {
                    let mut scratch = Scratch::new(&gpu, &class, &sizes, bytes, staging);
                    let mut times = TuneQualificationTimes::default();
                    scratch.upload(sizes.len() - 1, &data, Some(&mut times));
                    let read = scratch.read_output(&mut times);
                    assert_eq!(read.len(), data.len());
                    assert!(
                        read.iter()
                            .zip(&data)
                            .all(|(a, b)| a.to_bits() == b.to_bits())
                    );
                    for time in [
                        times.upload_host_copy,
                        times.upload_transfer,
                        times.readback_transfer,
                        times.readback_host_copy,
                    ] {
                        assert!(time.is_some_and(|time| !time.is_zero()));
                    }
                    assert_eq!(times.validation, None);
                    assert_eq!(times.dispatch, None);
                }
            }
        }
    }

    #[test]
    fn scratch_cap_counts_staging_and_checks_overflow() {
        assert_eq!(scratch_bytes(&[4, 8, 12]), Some(36));
        assert_eq!(scratch_bytes(&[usize::MAX, 4]), None);
    }

    #[test]
    fn initial_small_tile_geometry_is_exact_before_search() {
        let mut graph = crate::Graph::new();
        let x = graph.input("x", &[33, 17]);
        let w = graph.parameter("w", &[17, 65]);
        let y = graph.matmul(x, w);
        graph.set_outputs(vec![y]);
        let mut plan = crate::compile::compile_with(&graph, &crate::CompileOptions::default());
        super::super::select_variants(&mut plan, None, false, false);
        let dispatch = &plan.dispatches[0];
        assert!(dispatch.use_small_tiles);
        assert_eq!(dispatch.workgroups, [3, 2, 1]);
        assert!(TuneClass::from_dispatch(dispatch, None).is_some());
    }

    #[test]
    fn reference_dots_match_independent_rectangular_example() {
        let mut class = TuneClass {
            shader: ShaderEntry::MatMul,
            m: 2,
            n: 2,
            k: 3,
            requires_full_precision: false,
            device_local: [false; 4],
            binding_bytes: Vec::new(),
        };
        let normal = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        ];
        let expected = [58.0, 64.0, 139.0, 154.0];
        for shader in [
            ShaderEntry::MatMul,
            ShaderEntry::MatMulAT,
            ShaderEntry::MatMulBT,
            ShaderEntry::FusedMatMulAdd,
        ] {
            class.shader = shader;
            let mut inputs = normal.clone();
            if class.shader == ShaderEntry::MatMulAT {
                inputs[0] = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
            }
            if class.shader == ShaderEntry::MatMulBT {
                inputs[1] = vec![7.0, 9.0, 11.0, 8.0, 10.0, 12.0];
            }
            if class.has_addend() {
                inputs.push(vec![0.5; 4]);
            }
            for (index, &value) in expected.iter().enumerate() {
                assert_eq!(
                    reference_dot(&class, &inputs, index / 2, index % 2),
                    value + if class.has_addend() { 0.5 } else { 0.0 }
                );
            }
        }
    }

    #[test]
    fn repeated_classes_share_search_but_aliases_and_placements_do_not() {
        let mut graph = crate::Graph::new();
        let x = graph.input("x", &[33, 17]);
        let a = graph.parameter("a", &[17, 65]);
        let b = graph.parameter("b", &[17, 65]);
        let y = graph.matmul(x, a);
        let z = graph.matmul(x, b);
        graph.set_outputs(vec![y, z]);
        let mut plan = crate::compile::compile_with(
            &graph,
            &crate::CompileOptions {
                fuse_dispatches: false,
                ..Default::default()
            },
        );
        let mut alias = crate::memplan::AliasPlan::identity(&plan.buffers);
        let (classes, excluded) = collect_classes(&plan, &alias, None);
        assert_eq!(excluded, 0);
        assert_eq!(classes.len(), 1);
        assert_eq!(classes[0].members.len(), 2);
        alias.device_local[plan.dispatches[1].output_buffer.0 as usize] = true;
        assert_eq!(collect_classes(&plan, &alias, None).0.len(), 2);
        plan.dispatches[0].input_buffers[1] = plan.dispatches[0].output_buffer;
        assert_eq!(collect_classes(&plan, &alias, None).1, 1);
    }

    #[test]
    fn tuning_tiles_generate_valid_exact_binding_layouts() {
        for entry in [
            ShaderEntry::MatMul,
            ShaderEntry::FusedMatMulAdd,
            ShaderEntry::MatMulAT,
            ShaderEntry::MatMulBT,
        ] {
            for tile in [
                MatmulTile::Tile32,
                MatmulTile::Tile64,
                MatmulTile::CooperativeF32 { tile_size: 8 },
                MatmulTile::CooperativeF32 { tile_size: 16 },
            ] {
                let mut module = match tile {
                    MatmulTile::Tile32 => {
                        crate::codegen::generate_module_small(entry.shader_group())
                    }
                    MatmulTile::Tile64 => crate::codegen::generate_module(entry.shader_group()),
                    MatmulTile::CooperativeF32 { .. } => crate::codegen::generate_module_coop(
                        entry.shader_group(),
                        &tile.coop_config().unwrap(),
                    ),
                };
                // Blade assigns resource bindings by ShaderData field name.
                // Assign distinct test bindings before full offline validation.
                for (index, (_, var)) in module.module.global_variables.iter_mut().enumerate() {
                    if matches!(
                        var.space,
                        naga::AddressSpace::Storage { .. } | naga::AddressSpace::Uniform
                    ) {
                        var.binding = Some(naga::ResourceBinding {
                            group: 0,
                            binding: index as u32,
                        });
                    }
                }
                let info = naga::valid::Validator::new(
                    naga::valid::ValidationFlags::all(),
                    naga::valid::Capabilities::all(),
                )
                .validate(&module.module)
                .unwrap();
                naga::back::spv::write_vec(
                    &module.module,
                    &info,
                    &naga::back::spv::Options::default(),
                    Some(&naga::back::spv::PipelineOptions {
                        shader_stage: naga::ShaderStage::Compute,
                        entry_point: entry.entry_point().to_owned(),
                    }),
                )
                .unwrap();
                assert!(
                    module
                        .module
                        .entry_points
                        .iter()
                        .any(|e| e.name == entry.entry_point())
                );
                let mut names: Vec<_> = module
                    .module
                    .global_variables
                    .iter()
                    .filter(|&(_, var)| var.binding.is_some())
                    .map(|(_, var)| var.name.as_deref().unwrap())
                    .collect();
                names.sort_unstable();
                let expected = if entry == ShaderEntry::FusedMatMulAdd {
                    vec!["matrix_a", "matrix_b", "matrix_c", "params", "src"]
                } else {
                    vec!["matrix_a", "matrix_b", "matrix_c", "params"]
                };
                assert_eq!(names, expected);
            }
        }
    }

    #[test]
    fn reference_qualification_checks_layout_and_tiny_operands() {
        for shader in [
            ShaderEntry::MatMul,
            ShaderEntry::MatMulAT,
            ShaderEntry::MatMulBT,
            ShaderEntry::FusedMatMulAdd,
        ] {
            let class = TuneClass {
                shader,
                m: 3,
                n: 5,
                k: 7,
                requires_full_precision: true,
                device_local: [false; 4],
                binding_bytes: Vec::new(),
            };
            let sizes = class.buffer_sizes().unwrap();
            for pattern in 0..2 {
                let inputs = test_inputs(&sizes, pattern);
                let output: Vec<_> = (0..15)
                    .map(|i| reference_dot(&class, &inputs, i / 5, i % 5) as f32)
                    .collect();
                let scale = if pattern == 0 { 1.0 } else { 1.0e-12 };
                assert!(qualify_output(&class, &inputs, &output, scale));
                assert!(!qualify_output(&class, &inputs, &[0.0; 15], scale));
                assert!(!qualify_output(&class, &inputs, &[f32::NAN; 15], scale));
                let mut corrupt = output.clone();
                corrupt[14] += scale as f32;
                assert!(!outputs_agree(&output, &corrupt, scale));
                assert!(!qualify_output(&class, &inputs, &corrupt, scale));
            }
        }
    }
}
