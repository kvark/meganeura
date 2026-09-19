use crate::compile::{BufferRef, CachedBlockAttentionParams, Dispatch, ShaderEntry};
use crate::tune::{
    TuneAttention, TuneDecision, TuneOptions, TuneOutcome, TuneQualificationTimes, TuneReport,
    TuneScope, TuneScratchUsage,
};
use blade_graphics as bg;
use std::{collections::HashMap, sync::Arc, time::Instant};

type Outcome = TuneOutcome<TuneAttention, u32>;

struct Member {
    first: usize,
    combine: Option<usize>,
}

struct Class {
    key: TuneAttention,
    initial: u32,
    members: Vec<Member>,
}

impl TuneAttention {
    fn from_dispatch(d: &Dispatch) -> Option<Self> {
        let CachedBlockAttentionParams {
            window_size,
            num_heads,
            num_kv_heads,
            head_dim,
            block_len,
            max_seq,
            ..
        } = CachedBlockAttentionParams::from_words(&d.params)?;
        if [num_heads, num_kv_heads, head_dim, block_len, max_seq].contains(&0)
            || !num_heads.is_multiple_of(num_kv_heads)
            || head_dim > 512
            || block_len > max_seq
            || num_heads > 0xFFFF
            || block_len > 0xFFFF
            || max_seq > u32::MAX / 2
        {
            return None;
        }
        let last = max_seq - block_len;
        let mut positions = vec![0, last / 2, last];
        positions.dedup();
        let key = Self {
            window_size,
            num_heads,
            num_kv_heads,
            head_dim,
            block_len,
            max_seq,
            positions,
            device_local: [false; 7],
            binding_bytes: Vec::new(),
        };
        key.sizes(1)?;
        Some(key)
    }

    fn sizes(&self, splits: u32) -> Option<Vec<usize>> {
        if !(1..=16).contains(&splits) {
            return None;
        }
        let bytes = |elements: u32| usize::try_from(elements).ok()?.checked_mul(4);
        let query = bytes(
            self.block_len
                .checked_mul(self.num_heads)?
                .checked_mul(self.head_dim)?,
        )?;
        let cache = bytes(
            self.max_seq
                .checked_mul(self.num_kv_heads)?
                .checked_mul(self.head_dim)?,
        )?;
        let partials = if splits == 1 {
            4
        } else {
            bytes(
                self.block_len
                    .checked_mul(self.num_heads)?
                    .checked_mul(splits)?
                    .checked_mul(self.head_dim.checked_add(2)?)?,
            )?
        };
        Some(vec![query, cache, cache, 4, 4, query, partials])
    }

    fn sequence(
        &self,
        source: &Dispatch,
        output: BufferRef,
        partials: BufferRef,
        splits: u32,
    ) -> Vec<Dispatch> {
        let old_entry = format!("{:?}", source.shader);
        let label = |entry: &ShaderEntry| match source.label.rsplit_once(&old_entry) {
            Some((prefix, suffix)) => format!("{prefix}{entry:?}{suffix}"),
            None => format!("{entry:?}"),
        };
        let mut first = source.clone();
        let mut params = CachedBlockAttentionParams::from_words(&source.params)
            .expect("qualified attention parameters");
        params.splits = if splits == 1 { 0 } else { splits };
        params.chunk = if splits == 1 {
            0
        } else {
            self.max_seq.div_ceil(splits)
        };
        first.params = params.to_words();
        first.output_buffer = output;
        first.shader = ShaderEntry::CachedBlockAttention;
        first.label = label(&first.shader);
        first.workgroups = [self.block_len, self.num_heads, 1];
        if splits == 1 {
            return vec![first];
        }
        first.shader = ShaderEntry::CachedBlockAttentionSplit;
        first.label = label(&first.shader);
        first.workgroups = [self.block_len, splits, self.num_heads];
        first.output_buffer = partials;
        let mut combine = first.clone();
        combine.shader = ShaderEntry::CachedBlockAttentionCombine;
        combine.label = label(&combine.shader);
        combine.input_buffers = vec![partials];
        combine.output_buffer = output;
        combine.workgroups = [self.block_len, self.num_heads, 1];
        vec![first, combine]
    }
}

fn plain(d: &Dispatch) -> bool {
    !d.use_coop
        && !d.use_small_tiles
        && !d.use_coop_compensated
        && d.weight_format == crate::compile::WeightFormat::F32
        && d.horizontal_batch < 2
        && d.matmul_prologue.is_none()
        && d.matmul_epilogue.is_none()
        && d.epilogue.is_empty()
        && d.epilogue_buffers.is_empty()
        && d.extra_outputs.is_empty()
        && d.pointwise.is_none()
        && d.reduction.is_none()
        && d.conv_k_tile.is_none()
        && d.gemv_shape.is_none()
        && d.gemv_rmsnorm.is_none()
        && !d.gemv_int_dot
        && d.scalar_fallback.is_none()
}

fn collect(session: &super::Session) -> Vec<Class> {
    let plan = &session.plan;
    let mut classes: Vec<Class> = Vec::new();
    let mut indices = HashMap::new();
    for (first, d) in plan.dispatches.iter().enumerate() {
        if !matches!(
            d.shader,
            ShaderEntry::CachedBlockAttention | ShaderEntry::CachedBlockAttentionSplit
        ) || d.input_buffers.len() != 5
            || !plain(d)
        {
            continue;
        }
        let Some(mut key) = TuneAttention::from_dispatch(d) else {
            continue;
        };
        let combine = if d.shader == ShaderEntry::CachedBlockAttentionSplit {
            let consumers: Vec<_> = plan
                .dispatches
                .iter()
                .enumerate()
                .filter(|&(_, c)| c.input_buffers.contains(&d.output_buffer))
                .collect();
            let &[(index, c)] = consumers.as_slice() else {
                continue;
            };
            if index <= first
                || c.shader != ShaderEntry::CachedBlockAttentionCombine
                || c.input_buffers != [d.output_buffer]
                || c.params != d.params
                || !plain(c)
                || c.workgroups != [key.block_len, key.num_heads, 1]
            {
                continue;
            }
            Some(index)
        } else {
            None
        };
        let initial = if combine.is_some() {
            CachedBlockAttentionParams::from_words(&d.params)
                .unwrap()
                .splits
        } else {
            1
        };
        // The compiler may start at non-power-of-two counts. Keep that legal
        // incumbent as well as the small power-of-two challenger set.
        if initial == 0 || initial > 16 {
            continue;
        }
        let output = plan.dispatches[combine.unwrap_or(first)].output_buffer;
        let expected = key.sequence(d, output, d.output_buffer, initial);
        if d.workgroups != expected[0].workgroups || d.params != expected[0].params {
            continue;
        }
        let mut bindings = d.input_buffers.clone();
        bindings.push(output);
        if combine.is_some() {
            bindings.push(d.output_buffer);
        }
        let physical: Vec<_> = bindings
            .iter()
            .map(|b| session.alias.map[b.0 as usize])
            .collect();
        if physical
            .iter()
            .enumerate()
            .any(|(i, p)| physical[..i].contains(p))
        {
            continue;
        }
        for (i, &p) in physical.iter().enumerate() {
            key.device_local[i] = session.alias.device_local[p];
        }
        if combine.is_none() {
            key.device_local[6] = session
                .attention_partials
                .get(&output)
                .map(|b| session.alias.device_local[session.alias.map[b.0 as usize]])
                .unwrap_or(session.optimizer_device);
        }
        key.binding_bytes = bindings[..6]
            .iter()
            .map(|b| plan.buffers[b.0 as usize])
            .collect();
        if key.sizes(1).unwrap()[..6]
            .iter()
            .zip(&key.binding_bytes)
            .any(|(need, have)| need > have)
        {
            continue;
        }
        let next = classes.len();
        let index = *indices.entry((key.clone(), initial)).or_insert(next);
        if index == next {
            classes.push(Class {
                key,
                initial,
                members: Vec::new(),
            });
        }
        classes[index].members.push(Member { first, combine });
    }
    classes.sort_by_key(|c| {
        std::cmp::Reverse(
            c.members.len() as u128
                * c.key.block_len as u128
                * c.key.num_heads as u128
                * c.key.head_dim as u128
                * c.key.max_seq as u128,
        )
    });
    classes
}

impl super::Pipelines {
    fn ensure_attention(
        &mut self,
        gpu: &super::Gpu,
        entry: ShaderEntry,
        dim: u32,
    ) -> Result<(), String> {
        let key = super::Variant::Attention(entry.clone(), dim);
        if self.map.contains_key(&key) {
            return Ok(());
        }
        let module =
            crate::codegen::generate_cached_attention_module(entry.shader_group(), Some(dim));
        self.insert_tuning_pipeline(gpu, key, module, super::super::shader_data_layout(&entry))
    }
}

impl super::Session {
    fn install_attention(
        &mut self,
        class: &Class,
        splits: u32,
        replacements: &mut HashMap<usize, Vec<Dispatch>>,
    ) {
        self.wait();
        let bytes = class.key.sizes(splits).unwrap()[6];
        let mut shared_physical = None;
        for member in &class.members {
            let source = self.plan.dispatches[member.first].clone();
            let output = self.plan.dispatches[member.combine.unwrap_or(member.first)].output_buffer;
            if member.combine.is_some() {
                self.attention_partials.insert(output, source.output_buffer);
            }
            let partial = if splits == 1 {
                BufferRef(0)
            } else if let Some(&partial) = self.attention_partials.get(&output) {
                self.grow_attention_partial(partial, bytes);
                partial
            } else {
                // These new sequences get separate barrier groups below, so
                // one class-sized physical scratch can serve all its members.
                let physical = *shared_physical.get_or_insert_with(|| {
                    super::ensure_device_memory_budget(&self.gpu, bytes, "attention partials");
                    let device = class.key.device_local[6];
                    let handle = self.gpu.create_buffer(bg::BufferDesc {
                        name: "attention_partials",
                        size: bytes as u64,
                        memory: if device {
                            bg::Memory::DeviceTransient
                        } else {
                            bg::Memory::Shared
                        },
                    });
                    let physical = self.physical_buffers.len();
                    self.physical_buffers
                        .push(Arc::new(super::super::PhysicalBuffer {
                            gpu: Arc::clone(&self.gpu),
                            handle,
                        }));
                    self.alias.sizes.push(bytes);
                    self.alias.device_local.push(device);
                    physical
                });
                let partial = BufferRef(self.plan.buffers.len() as u32);
                self.plan.buffers.push(bytes);
                self.buffers.push(self.physical_buffers[physical].handle);
                self.alias.map.push(physical);
                self.written.push(true);
                self.attention_partials.insert(output, partial);
                partial
            };
            let mut sequence = class.key.sequence(&source, output, partial, splits);
            if let Some(combine) = member.combine {
                let tail = if sequence.len() == 2 {
                    vec![sequence.pop().unwrap()]
                } else {
                    Vec::new()
                };
                replacements.insert(combine, tail);
            }
            replacements.insert(member.first, sequence);
        }
    }

    fn grow_attention_partial(&mut self, partial: BufferRef, bytes: usize) {
        let logical = partial.0 as usize;
        let physical = self.alias.map[logical];
        if bytes > self.alias.sizes[physical] {
            super::ensure_device_memory_budget(&self.gpu, bytes, "attention partial growth");
            let handle = self.gpu.create_buffer(bg::BufferDesc {
                name: "attention_partials",
                size: bytes as u64,
                memory: if self.alias.device_local[physical] {
                    bg::Memory::DeviceTransient
                } else {
                    bg::Memory::Shared
                },
            });
            // Other logical intermediates may reuse this allocation. Preserve
            // every old byte, including values observable between steps.
            self.encoder.start();
            self.encoder
                .transfer("grow_attention_partials")
                .copy_buffer_to_buffer(
                    self.physical_buffers[physical].handle.at(0),
                    handle.at(0),
                    self.alias.sizes[physical] as u64,
                );
            self.sync_point = Some(self.gpu.submit(&mut self.encoder));
            self.wait();
            self.physical_buffers[physical] = Arc::new(super::super::PhysicalBuffer {
                gpu: Arc::clone(&self.gpu),
                handle,
            });
            self.alias.sizes[physical] = bytes;
            for (i, &p) in self.alias.map.iter().enumerate() {
                if p == physical {
                    self.buffers[i] = handle;
                }
            }
            self.readback.borrow_mut().staged.clear();
        }
        self.plan.buffers[logical] = self.plan.buffers[logical].max(bytes);
    }

    fn replace_attention_sequences(&mut self, replacements: HashMap<usize, Vec<Dispatch>>) {
        let mut dispatches = Vec::new();
        let mut groups = Vec::new();
        let mut offsets = vec![0; self.plan.dispatches.len() + 1];
        for old_group in &self.groups {
            let mut begin = dispatches.len();
            for index in old_group.clone() {
                offsets[index] = dispatches.len();
                if let Some(sequence) = replacements.get(&index) {
                    if sequence.len() > 1 {
                        if begin != dispatches.len() {
                            groups.push(begin..dispatches.len());
                        }
                        for d in sequence {
                            begin = dispatches.len();
                            dispatches.push(d.clone());
                            groups.push(begin..dispatches.len());
                        }
                        begin = dispatches.len();
                    } else {
                        dispatches.extend_from_slice(sequence);
                    }
                } else {
                    dispatches.push(self.plan.dispatches[index].clone());
                }
            }
            if begin != dispatches.len() {
                groups.push(begin..dispatches.len());
            }
        }
        *offsets.last_mut().unwrap() = dispatches.len();
        if let Some(window) = self.profile_window.as_mut() {
            *window = offsets[window.start.min(offsets.len() - 1)]
                ..offsets[window.end.min(offsets.len() - 1)];
        }
        // Preserve every original group boundary: the allocation plan relies
        // on those barriers, even where a removed combine would allow merging.
        self.plan.dispatches = dispatches;
        self.groups = groups;
        self.pipelines.select(&self.plan.dispatches);
        self.profiled_pass_map.clear();
        self.last_gpu_timings = None;
    }

    pub(super) fn tune_attention(
        &mut self,
        report: &mut TuneReport,
        start: Instant,
        staging: &mut super::Staging<'_>,
    ) {
        if !matches!(report.options.scope, TuneScope::All | TuneScope::Attention) {
            return;
        }
        let classes = collect(self);
        report.eligible_classes += classes.len();
        report.excluded_dispatches -= classes
            .iter()
            .flat_map(|c| &c.members)
            .map(|m| 1 + usize::from(m.combine.is_some()))
            .sum::<usize>();
        report.class_limit_reached |= report.eligible_classes > report.options.max_classes;
        let mut replacements = HashMap::new();
        for class in classes {
            if report.visited_classes == report.options.max_classes
                || start.elapsed() >= report.options.max_time
            {
                break;
            }
            report.visited_classes += 1;
            let mut selected = class.initial;
            for candidate in [1, 2, 4, 8, 16] {
                if candidate == class.initial {
                    continue;
                }
                if start.elapsed() >= report.options.max_time {
                    break;
                }
                let mut outcome =
                    Outcome::new(class.key.clone(), class.members.len(), selected, candidate);
                let began = Instant::now();
                self.measure_attention(&class, &report.options, start, &mut outcome, staging);
                outcome.elapsed = began.elapsed();
                selected = outcome.selected;
                log::info!(
                    "tune attention: {:?}, {} -> {}: {:?}, {:?}/{:?} ms",
                    class.key,
                    outcome.initial,
                    outcome.candidate,
                    outcome.decision,
                    outcome.baseline_median_ms,
                    outcome.candidate_median_ms
                );
                if let Some(ref error) = outcome.failure {
                    log::warn!("tune attention: {error}");
                }
                report.attention_outcomes.push(outcome);
            }
            if selected != class.initial {
                self.install_attention(&class, selected, &mut replacements);
            }
        }
        if !replacements.is_empty() {
            self.replace_attention_sequences(replacements);
        }
    }

    fn measure_attention(
        &mut self,
        class: &Class,
        options: &TuneOptions,
        start: Instant,
        outcome: &mut Outcome,
        staging: &mut super::Staging<'_>,
    ) {
        let key = &class.key;
        {
            let phases = outcome.phase_times.as_mut().unwrap();
            let sizes;
            let variants;
            let sequences;
            let mut scratch;
            {
                let _preparation = super::PhaseTimer::new(&mut phases.preparation);
                phases.preparation_breakdown = Some(Default::default());
                let prep = phases.preparation_breakdown.as_mut().unwrap();
                let bytes;
                {
                    let _checks = super::PhaseTimer::new(&mut prep.checks);
                    let (Some(a), Some(b)) =
                        (key.sizes(outcome.initial), key.sizes(outcome.candidate))
                    else {
                        outcome.decision = TuneDecision::ScratchLimit;
                        return;
                    };
                    sizes = a
                        .iter()
                        .zip(&b)
                        .map(|(&a, &b)| a.max(b))
                        .collect::<Vec<_>>();
                    let Some(required) = super::scratch_bytes(&sizes) else {
                        outcome.decision = TuneDecision::ScratchLimit;
                        return;
                    };
                    bytes = required;
                    if bytes > options.max_scratch_bytes {
                        outcome.decision = TuneDecision::ScratchLimit;
                        return;
                    }
                }
                {
                    let _timer = super::PhaseTimer::new(&mut prep.staging);
                    staging.discard_unmatched(*sizes.iter().max().unwrap());
                }
                let memory = self.gpu.memory_stats();
                if memory.budget != 0
                    && (bytes - staging.stats.retained_staging_bytes) as u64
                        > super::super::safe_device_memory_remaining(memory.usage, memory.budget)
                {
                    outcome.decision = TuneDecision::DeviceMemoryBudget;
                    return;
                }
                let mut source = self.plan.dispatches[class.members[0].first].clone();
                source.input_buffers = (0..5).map(BufferRef).collect();
                variants = [outcome.initial, outcome.candidate]
                    .map(|s| key.sequence(&source, BufferRef(5), BufferRef(6), s));
                for d in variants.iter().flatten() {
                    if start.elapsed() >= options.max_time {
                        outcome.decision = TuneDecision::TimeBudget;
                        return;
                    }
                    let compiled = {
                        let _timer = super::PhaseTimer::new(&mut prep.pipelines);
                        self.pipelines
                            .ensure_attention(&self.gpu, d.shader.clone(), key.head_dim)
                    };
                    outcome.compile_time = prep.pipelines.unwrap();
                    if let Err(error) = compiled {
                        outcome.decision = TuneDecision::ShaderRejected;
                        outcome.failure = Some(error);
                        return;
                    }
                }
                scratch = super::Scratch::new(
                    &key.device_local,
                    &sizes,
                    (5, sizes[5] / 4),
                    bytes,
                    staging,
                    prep,
                    &mut phases.cleanup,
                );
                outcome.scratch = Some(TuneScratchUsage {
                    binding_bytes: sizes.clone(),
                    staging_bytes: scratch.staging.stats.retained_staging_bytes,
                    staging_reused: scratch.staging_reused,
                });
                sequences = variants
                    .iter()
                    .map(|v| {
                        v.iter()
                            .map(|d| {
                                (
                                    &self.pipelines.map[&super::Variant::Attention(
                                        d.shader.clone(),
                                        key.head_dim,
                                    )],
                                    d,
                                )
                            })
                            .collect()
                    })
                    .collect::<Vec<Vec<_>>>();
            }
            {
                phases.qualification_breakdown = Some(Default::default());
                let details = phases.qualification_breakdown.as_mut().unwrap();
                let _qualification = super::PhaseTimer::new(&mut phases.qualification);
                if let Err(error) = qualify(
                    key,
                    &mut scratch,
                    &sequences,
                    [outcome.initial, outcome.candidate],
                    &sizes,
                    start,
                    options,
                    details,
                ) {
                    outcome.decision = if start.elapsed() >= options.max_time {
                        TuneDecision::TimeBudget
                    } else {
                        TuneDecision::InvalidOutput
                    };
                    outcome.failure = Some(error);
                    return;
                }
                outcome.qualified = true;
            }
            {
                let _warmup = super::PhaseTimer::new(&mut phases.warmup);
                for (i, input) in super::test_inputs(&sizes[..6], 0)
                    .iter()
                    .enumerate()
                    .take(3)
                {
                    scratch.upload(i, input, None);
                }
                for &position in &key.positions {
                    scratch.upload(3, &[f32::from_bits(position)], None);
                    scratch.upload(4, &[f32::from_bits(key.block_len)], None);
                    for _ in 0..options.warmup_runs {
                        for sequence in &sequences {
                            if start.elapsed() >= options.max_time {
                                outcome.decision = TuneDecision::TimeBudget;
                                return;
                            }
                            scratch.run(sequence, 1);
                        }
                    }
                }
            }
            let _sampling = super::PhaseTimer::new(&mut phases.sampling);
            (outcome.baseline_ms, outcome.candidate_ms) =
                super::measure_pairs(options.sample_pairs, |alternative| {
                    let mut elapsed = 0.0;
                    for &position in &key.positions {
                        if start.elapsed() >= options.max_time {
                            return None;
                        }
                        scratch.upload(3, &[f32::from_bits(position)], None);
                        elapsed += scratch.run(
                            &sequences[usize::from(alternative)],
                            options.dispatches_per_sample,
                        );
                    }
                    (start.elapsed() < options.max_time).then_some(
                        elapsed
                            / (key.positions.len() as f64 * options.dispatches_per_sample as f64),
                    )
                });
        }
        super::decide(outcome, options);
    }
}

fn qualify(
    key: &TuneAttention,
    scratch: &mut super::Scratch<'_, '_>,
    sequences: &[Vec<(&bg::ComputePipeline, &Dispatch)>],
    splits: [u32; 2],
    sizes: &[usize],
    start: Instant,
    options: &TuneOptions,
    details: &mut TuneQualificationTimes,
) -> Result<(), String> {
    let mut cases: Vec<_> = key
        .positions
        .iter()
        .rev()
        .map(|&p| (p, key.block_len))
        .collect();
    let short = (0, key.block_len.min(2));
    if !cases.contains(&short) {
        cases.push(short);
    }
    for pattern in 0..2 {
        let inputs = {
            let _timer = super::PhaseTimer::new(&mut details.input_preparation);
            super::test_inputs(&sizes[..6], pattern)
        };
        for (index, input) in inputs.iter().enumerate().take(3) {
            scratch.upload(index, input, Some(details));
        }
        let scale = if pattern == 0 { 1.0 } else { 1.0e-12 };
        let mut references: Vec<Vec<f32>> = Vec::new();
        for variant in 0..2 {
            // Do not clear partials between long and short cases: stale slices
            // must be replaced by the identity partial, not folded into output.
            scratch.upload(6, &vec![f32::NAN; sizes[6] / 4], Some(details));
            for (case, &(position, valid)) in cases.iter().enumerate() {
                if start.elapsed() >= options.max_time {
                    return Err("qualification deadline".into());
                }
                scratch.upload(3, &[f32::from_bits(position)], Some(details));
                scratch.upload(4, &[f32::from_bits(valid)], Some(details));
                scratch.upload(5, &vec![f32::NAN; sizes[5] / 4], Some(details));
                {
                    let _timer = super::PhaseTimer::new(&mut details.dispatch);
                    scratch.run(&sequences[variant], 1);
                }
                let mut output = scratch.read_output(details);
                output.truncate((valid * key.num_heads * key.head_dim) as usize);
                let partials = if splits[variant] > 1 {
                    Some(scratch.read_buffer(
                        6,
                        (valid * key.num_heads * splits[variant] * (key.head_dim + 2)) as usize,
                        details,
                    ))
                } else {
                    None
                };
                let _timer = super::PhaseTimer::new(&mut details.validation);
                if !qualify_reference(key, &inputs, &output, position, valid, scale)
                    || (variant != 0 && !super::outputs_agree(&references[case], &output, scale))
                    || partials.as_ref().is_some_and(|p| {
                        !qualify_partials(key, p, position, valid, splits[variant])
                    })
                {
                    return Err(format!(
                        "variant {variant}, pattern {pattern}, position {position}, valid {valid}: reference/partial mismatch"
                    ));
                }
                if variant == 0 {
                    references.push(output);
                }
            }
        }
    }
    Ok(())
}

fn qualify_reference(
    key: &TuneAttention,
    inputs: &[Vec<f32>],
    output: &[f32],
    position: u32,
    valid: u32,
    scale: f64,
) -> bool {
    if valid == 0
        || valid > key.block_len
        || output.len() != (valid * key.num_heads * key.head_dim) as usize
        || output.iter().any(|x| !x.is_finite())
    {
        return false;
    }
    let dim = key.head_dim as usize;
    let mut rows = vec![0, valid / 2, valid - 1];
    rows.dedup();
    for row in rows {
        let end = (position + row + 1).min(key.max_seq);
        let begin = if key.window_size == 0 {
            0
        } else {
            end.saturating_sub(key.window_size)
        };
        for head in 0..key.num_heads {
            let q_offset = ((row * key.num_heads + head) * key.head_dim) as usize;
            let kv_head = head / (key.num_heads / key.num_kv_heads);
            let kv_offset = |token| ((token * key.num_kv_heads + kv_head) * key.head_dim) as usize;
            let scores: Vec<f64> = (begin..end)
                .map(|token| {
                    (0..dim)
                        .map(|d| {
                            inputs[0][q_offset + d] as f64 * inputs[1][kv_offset(token) + d] as f64
                        })
                        .sum::<f64>()
                        / (dim as f64).sqrt()
                })
                .collect();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<_> = scores.iter().map(|s| (s - max).exp()).collect();
            let sum: f64 = weights.iter().sum();
            for d in 0..dim {
                let expected = (begin..end)
                    .zip(&weights)
                    .map(|(token, w)| w * inputs[2][kv_offset(token) + d] as f64)
                    .sum::<f64>()
                    / sum;
                if !super::close(expected, output[q_offset + d], scale) {
                    return false;
                }
            }
        }
    }
    true
}

fn qualify_partials(
    key: &TuneAttention,
    partials: &[f32],
    position: u32,
    valid: u32,
    splits: u32,
) -> bool {
    if partials.iter().any(|x| !x.is_finite()) {
        return false;
    }
    let width = (key.head_dim + 2) as usize;
    let chunk = key.max_seq.div_ceil(splits);
    for row in 0..valid {
        let end = (position + row + 1).min(key.max_seq);
        let begin = if key.window_size == 0 {
            0
        } else {
            end.saturating_sub(key.window_size)
        };
        for head in 0..key.num_heads {
            for split in 0..splits {
                if begin + split * chunk < end {
                    continue;
                }
                let at = ((row * key.num_heads + head) * splits + split) as usize * width;
                if partials[at..at + width - 2].iter().any(|&v| v != 0.0)
                    || partials[at + width - 2] != -1.0e30
                    || partials[at + width - 1] != 0.0
                {
                    return false;
                }
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::{
        BufferRef, CachedBlockAttentionParams, Dispatch, HashMap, ShaderEntry, TuneAttention,
        collect, qualify_partials, qualify_reference,
    };

    #[test]
    fn attention_contract_and_oracles_cover_geometry_and_corruption() {
        let d = Dispatch {
            shader: ShaderEntry::CachedBlockAttentionSplit,
            label: "node: CachedBlockAttentionSplit[3]".into(),
            input_buffers: (0..5).map(BufferRef).collect(),
            output_buffer: BufferRef(6),
            params: vec![3, 2, 1, 5, 3, 7, 4, 2],
            workgroups: [3, 4, 2],
            ..Default::default()
        };
        let key = TuneAttention::from_dispatch(&d).unwrap();
        assert!(CachedBlockAttentionParams::from_words(&d.params[..7]).is_none());
        assert_eq!(key.positions, [0, 2, 4]);
        assert!(key.sizes(0).is_none());
        assert!(key.sizes(17).is_none());
        for count in [1, 2, 3, 4, 8, 16] {
            let sizes = key.sizes(count).unwrap();
            let sequence = key.sequence(&d, BufferRef(5), BufferRef(6), count);
            assert_eq!(sequence.len(), if count == 1 { 1 } else { 2 });
            assert_eq!(sequence.last().unwrap().output_buffer, BufferRef(5));
            for d in &sequence {
                assert_eq!(d.label, format!("node: {:?}[3]", d.shader));
            }
            assert_eq!(sizes[5], 3 * 2 * 5 * 4);
            if count != 1 {
                assert_eq!(sequence[0].params[6..], [count, 7u32.div_ceil(count)]);
                assert_eq!(sizes[6], (3 * 2 * count * 7 * 4) as usize);
            }
        }
        let mut inputs = super::super::test_inputs(&key.sizes(1).unwrap()[..6], 0);
        inputs[0].fill(0.0);
        for (i, v) in inputs[2].iter_mut().enumerate() {
            *v = i as f32;
        }
        let expected: Vec<_> = (0..3)
            .flat_map(|row| (0..10).map(move |i| ((3 + row) * 5 + i % 5) as f32))
            .collect();
        assert!(qualify_reference(&key, &inputs, &expected, 4, 3, 1.0));
        let mut broken = expected.clone();
        broken[7] += 0.1;
        assert!(!qualify_reference(&key, &inputs, &broken, 4, 3, 1.0));
        broken[7] = f32::NAN;
        assert!(!qualify_reference(&key, &inputs, &broken, 4, 3, 1.0));
        let mut partials = vec![0.0; 3 * 2 * 16 * 7];
        for row in partials.chunks_mut(7) {
            row[5] = -1.0e30;
        }
        assert!(qualify_partials(&key, &partials, 0, 3, 16));
        partials[15 * 7 + 6] = 1.0;
        assert!(!qualify_partials(&key, &partials, 0, 3, 16));
        for (index, value) in [(1, 0), (2, 3), (3, 513), (4, 8), (5, u32::MAX)] {
            let mut invalid = d.clone();
            invalid.params[index] = value;
            assert!(TuneAttention::from_dispatch(&invalid).is_none());
        }
    }

    #[test]
    #[ignore = "GPU attention sequence/storage transitions on an idle device"]
    fn attention_retuning_preserves_live_output_and_reuses_partial_storage() {
        for capacity in [64, 96] {
            let mut graph = crate::Graph::new();
            let q = graph.input("q", &[3, 8]);
            let q = graph.scale(q, 0.5);
            let k = graph.parameter("k", &[capacity, 4]);
            let v = graph.parameter("v", &[capacity, 4]);
            let position = graph.input_u32("position", &[1]);
            let valid = graph.input_u32("valid", &[1]);
            let attended = graph.cached_block_attention(q, k, v, position, valid, 2, 1, 4, 37);
            let output = graph.scale(attended, 0.75);
            graph.set_outputs(vec![output]);
            let mut config = crate::SessionConfig::inference_from_env();
            config.tune = false;
            config.runtime.coop = crate::CoopPolicy::Disabled;
            let mut session = crate::build(&graph, config).0;
            session.set_input(
                "q",
                &(0..24).map(|i| (i as f32 * 0.3).sin()).collect::<Vec<_>>(),
            );
            for (name, factor) in [("k", 0.2), ("v", 0.7)] {
                session.set_parameter(
                    name,
                    &(0..capacity * 4)
                        .map(|i| (i as f32 * factor).cos())
                        .collect::<Vec<_>>(),
                );
            }
            session.set_input_u32("position", &[(capacity - 3) as u32]);
            session.set_input_u32("valid", &[3]);
            session.step();
            session.wait();
            let reference = session.read_output(24);
            let mut footprint = None;
            for splits in [16, 1, 2, 16, 1, 4, 16] {
                let before = session.read_output(24);
                let classes = collect(&session);
                assert_eq!(classes.len(), 1);
                for entry in [
                    ShaderEntry::CachedBlockAttention,
                    ShaderEntry::CachedBlockAttentionSplit,
                    ShaderEntry::CachedBlockAttentionCombine,
                ] {
                    session
                        .pipelines
                        .ensure_attention(&session.gpu, entry, 4)
                        .unwrap();
                }
                session.set_profiling(true);
                let mut replacements = HashMap::new();
                session.install_attention(&classes[0], splits, &mut replacements);
                session.replace_attention_sequences(replacements);
                let size = (session.plan.buffers.len(), session.alias.physical_bytes());
                assert_eq!(
                    *footprint.get_or_insert(size),
                    size,
                    "retuning accumulated storage"
                );
                assert_eq!(
                    session.read_output(24),
                    before,
                    "installation changed a live result"
                );
                assert_eq!(
                    session.groups.last().unwrap().end,
                    session.plan.dispatches.len()
                );
                assert_eq!(
                    session.profile_window.as_ref().unwrap().end,
                    session.plan.dispatches.len()
                );
                session.step();
                session.wait();
                assert!(super::super::outputs_agree(
                    &reference,
                    &session.read_output(24),
                    1.0
                ));
            }
        }
    }
}
