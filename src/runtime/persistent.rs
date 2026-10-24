use super::{
    Dispatch, ExecutionPlan, Gpu, Session, SharedPipeline, SharedPipelineKey, SharedPipelines,
};
use blade_graphics as bg;
use std::{
    hash::{DefaultHasher, Hash, Hasher},
    sync::Arc,
};

/// Opt-in persistent execution of generated pointwise and reduction tiles.
/// Independent compatible operations share a queue; dependent regions retain
/// hardware dispatch barriers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct PersistentOptions {
    /// Physical workgroups per eligible dispatch, independent of its tile count.
    pub workers: u32,
    /// Permit explicitly reported ordinary dispatches and optimizer work.
    /// False rejects unsupported work instead of claiming persistent coverage.
    pub allow_fallback: bool,
    /// Diagnostic per-worker task counters. Adds a queue atomic per claimed task.
    pub track_workers: bool,
}

impl Default for PersistentOptions {
    fn default() -> Self {
        Self {
            workers: 64,
            allow_fallback: true,
            track_workers: false,
        }
    }
}

impl PersistentOptions {
    fn queue_stride(self) -> usize {
        let bytes = if self.track_workers {
            (self.workers as usize + 1) * 4
        } else {
            4
        };
        bytes.next_multiple_of(crate::memplan::ARENA_ALIGNMENT)
    }
}

/// How a contiguous span of compiled graph operations executes. A zero-tile operation still has
/// a physical launch, but contributes no persistent tasks.
#[derive(Clone, Debug, serde::Serialize)]
pub struct ExecutionRegion {
    pub dispatches: std::ops::Range<usize>,
    pub origin: Vec<crate::NodeId>,
    pub label: String,
    /// Queue claims: a tile, or a matching tile through a worker-owned chain.
    pub logical_tasks: u64,
    pub worker_owned_chain: bool,
    /// Zero denotes ordinary execution.
    pub workers: u32,
    pub workgroup_size: Option<u32>,
    pub fallback_reason: Option<String>,
}

/// Current graph coverage and runtime work appended to each step.
#[derive(Clone, Debug, serde::Serialize)]
pub struct ExecutionReport {
    pub persistent: Option<PersistentOptions>,
    pub regions: Vec<ExecutionRegion>,
    /// Graph launches only; appended optimizer work is reported separately.
    pub graph_dispatches: usize,
    pub persistent_tasks: u64,
    pub scheduler_bytes: usize,
    pub appended_fallback: Vec<&'static str>,
}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Params {
    grids: [[u32; 4]; 3],
    starts: [u32; 4],
    ends: [u32; 4],
}

#[derive(blade_macros::ShaderData)]
struct QueueData {
    persistent_queue: bg::BufferPiece,
    persistent_params: Params,
}

pub(super) struct State {
    gpu: Arc<Gpu>,
    pub options: PersistentOptions,
    allow_chains: bool,
    pub report: ExecutionReport,
    pipelines: Vec<Option<Arc<SharedPipeline>>>,
    keys: Vec<Option<String>>,
    params: Vec<Params>,
    dispatch_regions: Vec<usize>,
    queue: Option<bg::Buffer>,
    queue_stride: usize,
    pub groups: Vec<std::ops::Range<usize>>,
}

struct Handler {
    body: crate::schedule::TileKernel,
    layout: bg::ShaderDataLayout,
}

fn tile(dispatch: &Dispatch) -> Result<Handler, String> {
    let (body, layout) = if let Some(dag) = dispatch.pointwise() {
        (
            crate::schedule::lower_pointwise_tile(
                dag,
                crate::schedule::GridShape::default(),
                dispatch.pointwise_storage,
            ),
            super::pointwise_data_layout(dag.n_inputs),
        )
    } else if let Some(kernel) = dispatch.reduction() {
        (
            crate::schedule::lower_tile(&kernel.to_template()).unwrap(),
            super::reduction_data_layout(kernel),
        )
    } else {
        return Err(format!("{:?} has no reusable tile body", dispatch.shader));
    };
    Ok(Handler { body, layout })
}

fn count(dispatch: &Dispatch) -> u64 {
    dispatch
        .workgroups
        .into_iter()
        .map(u64::from)
        .fold(1, u64::saturating_mul)
}

fn overlaps(
    plan: &ExecutionPlan,
    buffers: &[bg::BufferPiece],
    a: crate::compile::BufferRef,
    b: crate::compile::BufferRef,
) -> bool {
    let (a, b) = (a.0 as usize, b.0 as usize);
    let (ap, bp) = (buffers[a], buffers[b]);
    ap.buffer == bp.buffer
        && ap.offset < bp.offset + plan.buffers[b] as u64
        && bp.offset < ap.offset + plan.buffers[a] as u64
}

fn independent(
    plan: &ExecutionPlan,
    buffers: &[bg::BufferPiece],
    left: &Dispatch,
    right: &Dispatch,
) -> bool {
    !left.written_buffers().any(|a| {
        right
            .read_buffers()
            .chain(right.written_buffers())
            .any(|b| overlaps(plan, buffers, a, b))
    }) && !right
        .written_buffers()
        .any(|b| left.read_buffers().any(|a| overlaps(plan, buffers, a, b)))
}

fn chain_compatible(
    plan: &ExecutionPlan,
    buffers: &[bg::BufferPiece],
    left: &Dispatch,
    right: &Dispatch,
) -> bool {
    let per_element =
        |dispatch: &Dispatch| dispatch.pointwise().is_some_and(|dag| !dag.has_broadcast());
    if !per_element(left)
        || !per_element(right)
        || left.params[0] != right.params[0]
        || left.workgroups != right.workgroups
    {
        return false;
    }
    // Identical logical references preserve the index-to-worker mapping.
    // Aliases between distinct logical references may shift that mapping.
    let owned = |a, b| a == b || !overlaps(plan, buffers, a, b);
    left.written_buffers().all(|a| {
        right
            .read_buffers()
            .chain(right.written_buffers())
            .all(|b| owned(a, b))
    }) && right
        .written_buffers()
        .all(|b| left.read_buffers().all(|a| owned(a, b)))
}

fn check_resources(
    caps: &bg::Capabilities,
    plan: &ExecutionPlan,
    buffers: &[bg::BufferPiece],
    dispatches: std::ops::Range<usize>,
    handlers: &[Handler],
    options: PersistentOptions,
    chain: bool,
) -> Result<u32, String> {
    let limits = caps
        .compute_binding_limits
        .ok_or("backend does not report compute binding limits")?;
    if handlers.len() + 1 > limits.max_bind_groups as usize {
        return Err("region exceeds device bind-group limit".into());
    }
    let mut storage = 1;
    let mut buffer_bindings = 2;
    let mut plain = std::mem::size_of::<Params>() as u32;
    let mut shared = 4u64;
    let mut tasks = 0u64;
    for (index, handler) in dispatches.clone().zip(handlers) {
        if handler.layout.bindings.len() > bg::limits::RESOURCES_IN_GROUP as usize {
            return Err("handler exceeds Blade's resources-per-group limit".into());
        }
        for entry in &handler.layout.bindings {
            match entry.1 {
                bg::ShaderBinding::Buffer => {
                    storage += 1;
                    buffer_bindings += 1;
                }
                bg::ShaderBinding::Plain { size } => {
                    plain += size;
                    buffer_bindings += 1;
                }
                _ => return Err("handler has an unsupported resource binding".into()),
            }
        }
        let dispatch = &plan.dispatches[index];
        tasks = tasks.saturating_add(count(dispatch));
        if let Some(kernel) = dispatch.reduction() {
            shared +=
                (1 + kernel.extra_prologues.len() as u64) * u64::from(kernel.workgroup_size) * 4;
        }
        for buffer in dispatch.read_buffers().chain(dispatch.written_buffers()) {
            let piece = buffers[buffer.0 as usize];
            if piece.buffer.size() - piece.offset > limits.max_storage_buffer_range {
                return Err("tensor binding exceeds device storage-buffer range".into());
            }
        }
    }
    if storage > limits.max_storage_buffers
        || plain > bg::limits::PLAIN_DATA_SIZE
        || buffer_bindings > limits.max_buffer_bindings
    {
        return Err("region exceeds buffer or plain-data binding limit".into());
    }
    if shared > u64::from(caps.max_compute_shared_memory_size) {
        return Err(format!(
            "region needs {shared} shared bytes; device limit is {}",
            caps.max_compute_shared_memory_size
        ));
    }
    // Blade binds the remaining allocation, not an explicitly sized subrange.
    if plan.dispatches.len() as u64 * options.queue_stride() as u64
        > limits.max_storage_buffer_range
    {
        return Err("queue allocation exceeds device storage-buffer range".into());
    }
    if chain {
        tasks = count(&plan.dispatches[dispatches.start]);
    }
    u32::try_from(tasks)
        .ok()
        .filter(|count| count.checked_add(options.workers).is_some())
        .ok_or_else(|| "queue counter would overflow u32".to_string())
}

fn binding_name(slot: usize, name: &str) -> &'static str {
    // Blade layouts require static names. This bounded vocabulary avoids
    // leaking dynamically allocated names for every constructed session.
    macro_rules! names {
        ($($name:literal),*) => { match name {
            $($name => [concat!("h0_", $name), concat!("h1_", $name), concat!("h2_", $name)][slot],)*
            _ => panic!("unknown generated binding {name}"),
        } };
    }
    names!(
        "src",
        "src_a",
        "src_b",
        "src_c",
        "src_idx",
        "src_a_idx",
        "src_b_idx",
        "src_c_idx",
        "per_row_src",
        "per_row_src_a",
        "per_row_src_b",
        "bias",
        "bias_a",
        "bias_b",
        "dst",
        "params"
    )
}

impl State {
    pub fn pipeline(&self, index: usize) -> Option<&bg::ComputePipeline> {
        self.pipelines[self.dispatch_regions[index]]
            .as_ref()
            .map(|pipeline| &pipeline.raw)
    }

    pub fn pipeline_key(&self, index: usize) -> Option<&str> {
        self.keys[self.dispatch_regions[index]].as_deref()
    }

    pub fn new(
        gpu: &Arc<Gpu>,
        plan: &ExecutionPlan,
        options: PersistentOptions,
        groups: &[std::ops::Range<usize>],
        buffers: &[bg::BufferPiece],
        external: bool,
        allow_chains: bool,
        dump_dir: Option<&str>,
    ) -> Self {
        use bg::ShaderData;
        super::validate_precision_plan(gpu, plan);
        assert!(
            (1..=65535).contains(&options.workers),
            "persistent workers must be in 1..=65535"
        );
        let caps = gpu.capabilities();
        let queue_stride = options.queue_stride();
        let mut pipelines = Vec::new();
        let mut params = Vec::new();
        let mut keys = Vec::new();
        let mut dispatch_regions = vec![0; plan.dispatches.len()];
        let mut report = ExecutionReport {
            persistent: Some(options),
            regions: Vec::new(),
            graph_dispatches: 0,
            persistent_tasks: 0,
            scheduler_bytes: 0,
            appended_fallback: Vec::new(),
        };
        let mut index = 0;
        let mut group_index = 0;
        while index < plan.dispatches.len() {
            while groups[group_index].end <= index {
                group_index += 1;
            }
            let start = index;
            let dispatch = &plan.dispatches[index];
            let mut region = ExecutionRegion {
                dispatches: index..index + 1,
                origin: dispatch.origin.clone(),
                label: dispatch.label.clone(),
                logical_tasks: count(dispatch),
                worker_owned_chain: false,
                workers: 0,
                workgroup_size: None,
                fallback_reason: None,
            };
            let generated = tile(dispatch).and_then(|handler| {
                check_resources(
                    &caps,
                    plan,
                    buffers,
                    index..index + 1,
                    std::slice::from_ref(&handler),
                    options,
                    false,
                )?;
                Ok(handler)
            });
            let mut region_params = Params {
                grids: [[0; 4]; 3],
                starts: [0; 4],
                ends: [0; 4],
            };
            match generated {
                Ok(first) => {
                    let workgroup_size = first.body.workgroup_size;
                    let mut handlers = vec![first];
                    let chain = allow_chains
                        && groups[group_index].len() == 1
                        && dispatch.pointwise().is_some();
                    index += 1;
                    // Whole independent operations can be shuffled across workers.
                    // BF16's module-wide helper/preamble is currently singleton-only.
                    while !external && handlers.len() < 3 && index < plan.dispatches.len() {
                        if chain {
                            let next_group = &groups[group_index + handlers.len()];
                            if next_group.start != index || next_group.len() != 1 {
                                break;
                            }
                        } else if index >= groups[group_index].end {
                            break;
                        }
                        let candidate = &plan.dispatches[index];
                        if candidate.pointwise_storage
                            != crate::schedule::PointwiseStorage::default()
                            || dispatch.pointwise_storage
                                != crate::schedule::PointwiseStorage::default()
                            || !(start..index).all(|prior| {
                                if chain {
                                    chain_compatible(
                                        plan,
                                        buffers,
                                        &plan.dispatches[prior],
                                        candidate,
                                    )
                                } else {
                                    independent(plan, buffers, &plan.dispatches[prior], candidate)
                                }
                            })
                        {
                            break;
                        }
                        let Ok(handler) = tile(candidate) else {
                            break;
                        };
                        if handler.body.workgroup_size != workgroup_size {
                            break;
                        }
                        handlers.push(handler);
                        if check_resources(
                            &caps,
                            plan,
                            buffers,
                            start..index + 1,
                            &handlers,
                            options,
                            chain,
                        )
                        .is_err()
                        {
                            handlers.pop();
                            break;
                        }
                        index += 1;
                    }
                    region.worker_owned_chain = chain && handlers.len() > 1;
                    region.dispatches.end = index;
                    region.logical_tasks = 0;
                    region.workers = options.workers;
                    region.workgroup_size = Some(workgroup_size);
                    region.origin.clear();
                    region.label = plan.dispatches[start..index]
                        .iter()
                        .map(|dispatch| dispatch.label.as_str())
                        .collect::<Vec<_>>()
                        .join(if region.worker_owned_chain {
                            " → "
                        } else {
                            " | "
                        });
                    let mut source = String::new();
                    let mut calls = String::new();
                    let mut layouts = Vec::new();
                    for (slot, handler) in handlers.into_iter().enumerate() {
                        let operation = &plan.dispatches[start + slot];
                        region.origin.extend_from_slice(&operation.origin);
                        source.push_str(&handler.body.source(&format!("h{slot}_")));
                        calls.push_str(
                            &include_str!("../shaders/persistent_handler.wgsl")
                                .replace("$SLOT", &slot.to_string())
                                .replace(
                                    "$CHAIN",
                                    if region.worker_owned_chain {
                                        "true"
                                    } else {
                                        "false"
                                    },
                                ),
                        );
                        layouts.push(bg::ShaderDataLayout {
                            bindings: handler
                                .layout
                                .bindings
                                .into_iter()
                                .map(|(name, binding)| (binding_name(slot, name), binding))
                                .collect(),
                        });
                        if region.worker_owned_chain {
                            region_params.starts[slot] = 0;
                            region.logical_tasks = count(operation);
                        } else {
                            region_params.starts[slot] = region.logical_tasks as u32;
                            region.logical_tasks += count(operation);
                        }
                        region_params.ends[slot] = region.logical_tasks as u32;
                        region_params.grids[slot][..3].copy_from_slice(&operation.workgroups);
                    }
                    region.origin.sort_unstable();
                    region.origin.dedup();
                    region_params.ends[3] = region.logical_tasks as u32;
                    report.persistent_tasks += region.logical_tasks;
                    source.push_str(
                        &include_str!("../shaders/persistent_tile.wgsl")
                            .replace("$WORKGROUP", &workgroup_size.to_string())
                            .replace("$HANDLERS", &calls)
                            .replace(
                                "$TRACK_WORKERS",
                                if options.track_workers {
                                    "true"
                                } else {
                                    "false"
                                },
                            ),
                    );
                    let module = crate::codegen::ShaderModule::with_hint(source, "persistent");
                    if let Some(dir) = dump_dir {
                        module.dump(dir);
                    }
                    layouts.push(QueueData::layout());
                    let key = SharedPipelineKey {
                        context: Arc::as_ptr(gpu) as usize,
                        entry_point: "persistent_main",
                        layout: format!("{layouts:?}"),
                        source: module.source.clone(),
                    };
                    let mut hasher = DefaultHasher::new();
                    key.source.hash(&mut hasher);
                    key.layout.hash(&mut hasher);
                    keys.push(Some(format!("persistent:{:016x}", hasher.finish())));
                    let pipeline = SharedPipelines::get(&key).unwrap_or_else(|| {
                        let shader = super::create_gen_shader(gpu, module)
                            .expect("persistent shader validation");
                        let pipeline = Arc::new(SharedPipeline {
                            gpu: Arc::clone(gpu),
                            raw: gpu.create_compute_pipeline(bg::ComputePipelineDesc {
                                name: "persistent region",
                                data_layouts: &layouts.iter().collect::<Vec<_>>(),
                                compute: shader.at("persistent_main"),
                            }),
                        });
                        SharedPipelines::insert(key, &pipeline);
                        pipeline
                    });
                    pipelines.push(Some(pipeline));
                }
                Err(reason) => {
                    assert!(
                        options.allow_fallback,
                        "strict persistent execution rejects dispatch {index} ({}): {reason}",
                        dispatch.label
                    );
                    region.fallback_reason = Some(reason);
                    pipelines.push(None);
                    keys.push(None);
                    index += 1;
                }
            }
            dispatch_regions[start..index].fill(report.regions.len());
            report.regions.push(region);
            params.push(region_params);
        }
        report.graph_dispatches = report.regions.len();
        let queue = if pipelines.iter().any(Option::is_some) {
            report.scheduler_bytes = report.regions.len() * queue_stride;
            super::ensure_device_memory_budget(gpu, report.scheduler_bytes, "persistent queues");
            Some(gpu.create_buffer(bg::BufferDesc {
                name: "persistent queues",
                size: report.scheduler_bytes as u64,
                memory: bg::Memory::Device,
            }))
        } else {
            None
        };
        // Conservative physical barriers also order distinct imported handles
        // that may alias the same allocation and whose overlap is unknowable.
        let groups = (0..report.regions.len()).map(|i| i..i + 1).collect();
        log::info!(
            "persistent execution: {} compiled operations, {} physical graph launches, {} tasks, {} workers",
            plan.dispatches.len(),
            report.graph_dispatches,
            report.persistent_tasks,
            options.workers
        );
        Self {
            gpu: Arc::clone(gpu),
            options,
            allow_chains,
            report,
            pipelines,
            keys,
            params,
            dispatch_regions,
            queue,
            queue_stride,
            groups,
        }
    }

    pub fn reset(&self, encoder: &mut bg::CommandEncoder) -> bool {
        if let Some(queue) = self.queue {
            encoder.transfer("reset persistent queues").fill_buffer(
                queue.at(0),
                self.report.scheduler_bytes as u64,
                0,
            );
            true
        } else {
            false
        }
    }

    pub fn record(
        &self,
        index: usize,
        plan: &ExecutionPlan,
        buffers: &[bg::BufferPiece],
        pass: &mut bg::ComputeCommandEncoder<'_>,
    ) -> bool {
        let Some(ref pipeline) = self.pipelines[index] else {
            return false;
        };
        let mut pc = pass.with(&pipeline.raw);
        let region = &self.report.regions[index];
        for (slot, operation) in region.dispatches.clone().enumerate() {
            assert!(Session::bind_generated_dispatch(
                buffers,
                &plan.dispatches[operation],
                &mut pc,
                slot as u32
            ));
        }
        pc.bind(
            region.dispatches.len() as u32,
            &QueueData {
                persistent_queue: self.queue.unwrap().at((index * self.queue_stride) as u64),
                persistent_params: self.params[index],
            },
        );
        pc.dispatch([self.options.workers, 1, 1]);
        true
    }
}

impl Drop for State {
    fn drop(&mut self) {
        if let Some(queue) = self.queue {
            self.gpu.destroy_buffer(queue);
        }
    }
}

impl Session {
    /// Number of physical graph launches. Persistent regions may implement
    /// several compiled operations; runtime-appended work is separate.
    pub fn execution_dispatch_count(&self) -> usize {
        self.persistent
            .as_ref()
            .map_or(self.plan.dispatches.len(), |state| {
                state.report.regions.len()
            })
    }

    pub(crate) fn execution_groups(&self) -> &[std::ops::Range<usize>] {
        self.persistent
            .as_ref()
            .map_or(&self.groups, |state| &state.groups)
    }

    pub(crate) fn profiling_overhead_passes(&self) -> usize {
        usize::from(
            self.persistent
                .as_ref()
                .is_some_and(|state| state.queue.is_some()),
        )
    }
    pub(super) fn refresh_persistent(&mut self) {
        if let Some(ref state) = self.persistent {
            let options = state.options;
            let allow_chains = state.allow_chains;
            self.wait();
            self.persistent = Some(State::new(
                &self.gpu,
                &self.plan,
                options,
                &self.groups,
                &self.buffers,
                self.physical_buffers.iter().any(|buffer| buffer.external),
                allow_chains,
                self.pipelines.dump_dir.as_deref(),
            ));
        }
    }

    /// Per-worker queue tasks from the most recently completed step, grouped
    /// by execution region. Ordinary regions have empty rows. Returns `None`
    /// when `PersistentOptions::track_workers` is off or no persistent work
    /// is selected. A task is a tile or a tile through a worker-owned chain.
    ///
    /// Call after executing a step. External records must first be submitted
    /// and registered with [`Session::track_submission`]. This waits for tracked
    /// work and downloads counters; use it outside performance measurements.
    pub fn read_persistent_worker_tasks(&mut self) -> Option<Vec<Vec<u32>>> {
        self.wait();
        let state = self.persistent.as_ref()?;
        if !state.options.track_workers {
            return None;
        }
        let queue = state.queue?;
        let mut words = vec![0u32; state.report.scheduler_bytes / 4];
        // The staging helper copies raw words without floating-point operations.
        self.read_staged_f32(
            queue.at(0),
            bytemuck::cast_slice_mut(&mut words),
            &mut self.readback.borrow_mut(),
        );
        Some(
            state
                .report
                .regions
                .iter()
                .enumerate()
                .map(|(index, region)| {
                    let start = index * state.queue_stride / 4 + 1;
                    words[start..start + region.workers as usize].to_vec()
                })
                .collect(),
        )
    }

    /// Report actual selected graph paths and configured appended work.
    pub fn execution_report(&self) -> ExecutionReport {
        let mut report = self.persistent.as_ref().map_or_else(
            || ExecutionReport {
                persistent: None,
                regions: self
                    .plan
                    .dispatches
                    .iter()
                    .enumerate()
                    .map(|(i, d)| ExecutionRegion {
                        dispatches: i..i + 1,
                        origin: d.origin.clone(),
                        label: d.label.clone(),
                        logical_tasks: d.workgroups.into_iter().map(u64::from).product(),
                        worker_owned_chain: false,
                        workers: 0,
                        workgroup_size: None,
                        fallback_reason: None,
                    })
                    .collect(),
                graph_dispatches: self.plan.dispatches.len(),
                persistent_tasks: 0,
                scheduler_bytes: 0,
                appended_fallback: Vec::new(),
            },
            |state| state.report.clone(),
        );
        if self.persistent.is_some() && !self.plan.param_grad_pairs.is_empty() {
            if self.grad_accum_scale.is_some() {
                report.appended_fallback.push("gradient accumulation");
            }
            if self.pending_lr.is_some() || self.pending_adam.is_some() {
                if self.pending_grad_clip.is_some() || self.pending_agc.is_some() {
                    report.appended_fallback.push("gradient clipping");
                }
                report.appended_fallback.push("optimizer update");
            }
        }
        report
    }

    pub(super) fn require_ordinary_optimizer(&self) {
        assert!(
            self.persistent
                .as_ref()
                .is_none_or(|state| state.options.allow_fallback)
                || self.plan.param_grad_pairs.is_empty(),
            "strict persistent execution does not support optimizer, clipping or accumulation passes"
        );
    }
}
