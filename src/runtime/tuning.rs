use super::{Gpu, Pipelines, Session, Variant, ensure_device_memory_budget};
use crate::compile::{BufferRef, Dispatch, ShaderEntry};
use crate::tune::{
    MatmulTile, TuneClass, TuneDecision, TuneError, TuneOptions, TuneOutcome, TuneReport, decide,
    measure_pairs,
};
use blade_graphics as bg;
use std::{collections::HashMap, time::Instant};

impl Pipelines {
    fn ensure_scalar_tile(&mut self, gpu: &Gpu, entry: &ShaderEntry, tile: MatmulTile) {
        let key = tile_variant(entry, tile);
        if self.map.contains_key(&key) {
            return;
        }
        let module = match tile {
            MatmulTile::Tile32 => crate::codegen::generate_module_small(entry.shader_group()),
            MatmulTile::Tile64 => crate::codegen::generate_module(entry.shader_group()),
        };
        let shader = gpu.create_shader(bg::ShaderDesc {
            source: &module.source,
            naga_module: Some(module.module),
        });
        let layout = super::shader_data_layout(entry);
        let pipeline = gpu.create_compute_pipeline(bg::ComputePipelineDesc {
            name: entry.entry_point(),
            data_layouts: &[&layout],
            compute: shader.at(entry.entry_point()),
        });
        self.map.insert(key, pipeline);
    }
}

fn tile_variant(entry: &ShaderEntry, tile: MatmulTile) -> Variant {
    match tile {
        MatmulTile::Tile32 => Variant::SmallTile(entry.clone()),
        MatmulTile::Tile64 => Variant::Scalar(entry.clone()),
    }
}

struct SearchClass {
    key: TuneClass,
    initial: MatmulTile,
    members: Vec<usize>,
}

fn collect_classes(
    plan: &crate::compile::ExecutionPlan,
    alias: &crate::memplan::AliasPlan,
) -> (Vec<SearchClass>, usize) {
    let mut classes: Vec<SearchClass> = Vec::new();
    let mut indices = HashMap::new();
    let mut excluded = 0;
    for (index, dispatch) in plan.dispatches.iter().enumerate() {
        let Some(mut key) = TuneClass::from_dispatch(dispatch) else {
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
        let Some(sizes) = key.buffer_sizes() else {
            excluded += 1;
            continue;
        };
        if bindings
            .iter()
            .zip(sizes)
            .any(|(b, size)| plan.buffers[b.0 as usize] < size)
        {
            excluded += 1;
            continue;
        }
        let initial = if dispatch.use_small_tiles {
            MatmulTile::Tile32
        } else {
            MatmulTile::Tile64
        };
        let next_index = classes.len();
        let class_index = *indices.entry((key.clone(), initial)).or_insert(next_index);
        if class_index == next_index {
            classes.push(SearchClass {
                key,
                initial,
                members: Vec::new(),
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
    /// Bounded scalar-tile search with default options; logs skips and returns
    /// per-class evidence. Use [`Self::tune_with`] for budgets and full reporting.
    ///
    /// Unlike the former family-wide tuner, this never calls `step()` and
    /// never reads or writes live tensor, optimizer, accumulator, or KV state.
    /// It is opt-in GPU work; do not call it while another workload is timing.
    pub fn tune(&mut self) -> Vec<TuneOutcome> {
        self.tune_with(TuneOptions::default())
            .expect("default tuning options are valid")
            .outcomes
    }

    /// Search the two existing scalar f32 matmul tiles for exact eligible
    /// classes. The old occupancy threshold is only the initial choice;
    /// measurements can select either size regardless of that threshold.
    ///
    /// Each class uses private scratch with matching memory placement and two
    /// deterministic nonzero input patterns (including tiny f32 operands).
    /// Both candidates must agree elementwise and with sampled f64 reference
    /// dots before alternating, batched `encode+submit+wait` measurements.
    /// The result is an isolated-kernel choice, not an end-to-end speed claim.
    ///
    /// Forward MatMul+Add is supported; other prologues/epilogues, horizontal
    /// packs, cooperative, reduced-storage, GEMV and overlapping-binding
    /// dispatches are excluded. Winners live in this session, not the plan cache.
    /// Only selected dispatch geometry and pipeline resources change. No graph
    /// execution occurs, including when an optimizer or external buffer is bound.
    /// A soft deadline may be exceeded by one in-flight operation; incomplete
    /// measurements always retain the original choice.
    pub fn tune_with(&mut self, options: TuneOptions) -> Result<TuneReport, TuneError> {
        options.validate()?;
        let start = Instant::now();
        let (classes, excluded_dispatches) = collect_classes(&self.plan, &self.alias);
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
            let mut outcome = TuneOutcome {
                class: class.key.clone(),
                dispatches: class.members.len(),
                initial: class.initial,
                selected: class.initial,
                decision: TuneDecision::KeepBaseline,
                qualified: false,
                elapsed: std::time::Duration::ZERO,
                compile_time: std::time::Duration::ZERO,
                baseline_ms: Vec::new(),
                candidate_ms: Vec::new(),
                baseline_median_ms: None,
                candidate_median_ms: None,
                noise_margin_ms: None,
            };
            let class_start = Instant::now();
            self.measure_scalar_class(class, &options, start, &mut outcome);
            outcome.elapsed = class_start.elapsed();
            if outcome.selected != outcome.initial {
                for &index in &class.members {
                    outcome
                        .selected
                        .apply(&mut self.plan.dispatches[index], &class.key);
                }
            }
            log::info!(
                "tune: {:?} {}x{}x{} ({} dispatches): {:?} -> {:?}, {:?}, medians {:?}/{:?} ms",
                class.key.shader,
                class.key.m,
                class.key.n,
                class.key.k,
                class.members.len(),
                outcome.initial,
                outcome.selected,
                outcome.decision,
                outcome.baseline_median_ms,
                outcome.candidate_median_ms
            );
            report.outcomes.push(outcome);
        }
        report.elapsed = start.elapsed();
        report.time_budget_exhausted |= report.elapsed >= options.max_time;
        log::info!(
            "tune: {}/{} classes visited; {} dispatches excluded; {:.3}s; class limit={}, time limit={}",
            report.outcomes.len(),
            report.eligible_classes,
            report.excluded_dispatches,
            report.elapsed.as_secs_f64(),
            report.class_limit_reached,
            report.time_budget_exhausted
        );
        Ok(report)
    }

    fn measure_scalar_class(
        &mut self,
        class: &SearchClass,
        options: &TuneOptions,
        start: Instant,
        outcome: &mut TuneOutcome,
    ) {
        let sizes = class
            .key
            .buffer_sizes()
            .expect("class collection checked extents");
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
        for tile in [class.initial, class.initial.other()] {
            if start.elapsed() >= options.max_time {
                outcome.decision = TuneDecision::TimeBudget;
                return;
            }
            let compile_start = Instant::now();
            self.pipelines
                .ensure_scalar_tile(&self.gpu, &class.key.shader, tile);
            outcome.compile_time += compile_start.elapsed();
        }
        if start.elapsed() >= options.max_time {
            outcome.decision = TuneDecision::TimeBudget;
            return;
        }
        let mut scratch = Scratch::new(&self.gpu, &class.key, &sizes, bytes);
        let mut dispatch = self.plan.dispatches[class.members[0]].clone();
        dispatch.input_buffers = (0..sizes.len() - 1).map(|i| BufferRef(i as u32)).collect();
        dispatch.output_buffer = BufferRef((sizes.len() - 1) as u32);
        let mut variants = [dispatch.clone(), dispatch];
        class.initial.apply(&mut variants[0], &class.key);
        class.initial.other().apply(&mut variants[1], &class.key);
        let pipelines = [
            &self.pipelines.map[&tile_variant(&class.key.shader, class.initial)],
            &self.pipelines.map[&tile_variant(&class.key.shader, class.initial.other())],
        ];
        for pattern in 0..2 {
            if start.elapsed() >= options.max_time {
                outcome.decision = TuneDecision::TimeBudget;
                return;
            }
            let inputs = test_inputs(&sizes, pattern);
            for (index, data) in inputs.iter().enumerate() {
                scratch.upload(index, data);
            }
            let mut reference = Vec::new();
            for variant in 0..2 {
                if start.elapsed() >= options.max_time {
                    outcome.decision = TuneDecision::TimeBudget;
                    return;
                }
                scratch.upload(sizes.len() - 1, &vec![f32::NAN; sizes[sizes.len() - 1] / 4]);
                scratch.run(pipelines[variant], &variants[variant], 1);
                let output = scratch.read_output();
                let scale = if pattern == 0 { 1.0 } else { 1.0e-12 };
                if !qualify_output(&class.key, &inputs, &output, scale)
                    || (variant == 1 && !outputs_agree(&reference, &output, scale))
                {
                    outcome.decision = TuneDecision::InvalidOutput;
                    return;
                }
                reference = output;
            }
        }
        outcome.qualified = true;
        // Time ordinary-magnitude data, not a zero-filled or subnormal workload.
        for (index, data) in test_inputs(&sizes, 0).iter().enumerate() {
            scratch.upload(index, data);
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
    fn new(gpu: &'a Gpu, class: &TuneClass, sizes: &[usize], bytes: usize) -> Self {
        ensure_device_memory_budget(gpu, bytes, "scalar-tile tuning scratch");
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
            memory: bg::Memory::Shared,
        });
        let encoder = gpu.create_command_encoder(bg::CommandEncoderDesc {
            name: "scalar_tile_tune",
            buffer_count: 1,
            manual_barriers: false,
        });
        Self {
            gpu,
            buffers,
            staging,
            output_elements: sizes[sizes.len() - 1] / 4,
            encoder,
        }
    }

    fn upload(&mut self, index: usize, data: &[f32]) {
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.staging.data().cast(), data.len());
        }
        self.encoder.start();
        self.encoder.transfer("tune_upload").copy_buffer_to_buffer(
            self.staging.at(0),
            self.buffers[index].at(0),
            std::mem::size_of_val(data) as u64,
        );
        self.submit_wait();
    }

    fn read_output(&mut self) -> Vec<f32> {
        self.encoder.start();
        self.encoder
            .transfer("tune_readback")
            .copy_buffer_to_buffer(
                self.buffers.last().unwrap().at(0),
                self.staging.at(0),
                (self.output_elements * 4) as u64,
            );
        self.submit_wait();
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
                    ((bits % 1024) as f32 - 511.5) * (scale / 1024.0)
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
    let edges = [0, 1, 31, 32, 63, 64, usize::MAX];
    for i in 0..32 {
        let row = if i == 7 {
            m - 1
        } else if i == 8 {
            0
        } else if i < edges.len() {
            edges[i].min(m - 1)
        } else {
            i * 104729 % m
        };
        let col = if i == 7 {
            n - 1
        } else if i == 8 {
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
        assert!(TuneClass::from_dispatch(dispatch).is_some());
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
        let (classes, excluded) = collect_classes(&plan, &alias);
        assert_eq!(excluded, 0);
        assert_eq!(classes.len(), 1);
        assert_eq!(classes[0].members.len(), 2);
        alias.device_local[plan.dispatches[1].output_buffer.0 as usize] = true;
        assert_eq!(collect_classes(&plan, &alias).0.len(), 2);
        plan.dispatches[0].input_buffers[1] = plan.dispatches[0].output_buffer;
        assert_eq!(collect_classes(&plan, &alias).1, 1);
    }

    #[test]
    fn both_scalar_tiles_generate_valid_exact_binding_layouts() {
        for entry in [
            ShaderEntry::MatMul,
            ShaderEntry::FusedMatMulAdd,
            ShaderEntry::MatMulAT,
            ShaderEntry::MatMulBT,
        ] {
            for tile in [MatmulTile::Tile32, MatmulTile::Tile64] {
                let mut module = match tile {
                    MatmulTile::Tile32 => {
                        crate::codegen::generate_module_small(entry.shader_group())
                    }
                    MatmulTile::Tile64 => crate::codegen::generate_module(entry.shader_group()),
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
