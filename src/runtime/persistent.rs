use super::{
    Dispatch, ExecutionPlan, Gpu, Session, SharedPipeline, SharedPipelineKey, SharedPipelines,
};
use blade_graphics as bg;
use std::sync::Arc;

/// Opt-in persistent execution of generated pointwise and reduction tiles.
/// Operation boundaries remain hardware barriers in this first integration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct PersistentOptions {
    /// Physical workgroups per eligible dispatch, independent of its tile count.
    pub workers: u32,
    /// Permit explicitly reported ordinary dispatches and optimizer work.
    /// False rejects unsupported work instead of claiming persistent coverage.
    pub allow_fallback: bool,
}

impl Default for PersistentOptions {
    fn default() -> Self {
        Self {
            workers: 64,
            allow_fallback: true,
        }
    }
}

/// How one compiled graph operation executes. A zero-tile operation still has
/// a physical launch, but contributes no persistent tasks.
#[derive(Clone, Debug, serde::Serialize)]
pub struct ExecutionRegion {
    pub dispatches: std::ops::Range<usize>,
    pub origin: Vec<crate::NodeId>,
    pub label: String,
    pub logical_tasks: u64,
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
    grid: [u32; 3],
    count: u32,
}

#[derive(blade_macros::ShaderData)]
struct QueueData {
    persistent_queue: bg::BufferPiece,
    persistent_params: Params,
}

pub(super) struct State {
    gpu: Arc<Gpu>,
    pub options: PersistentOptions,
    pub report: ExecutionReport,
    pipelines: Vec<Option<Arc<SharedPipeline>>>,
    queue: Option<bg::Buffer>,
    pub groups: Vec<std::ops::Range<usize>>,
}

fn tile(
    dispatch: &Dispatch,
    shared_limit: u32,
) -> Result<(crate::schedule::TileKernel, bg::ShaderDataLayout), String> {
    if let Some(dag) = dispatch.pointwise() {
        Ok((
            crate::schedule::lower_pointwise_tile(
                dag,
                crate::schedule::GridShape::default(),
                dispatch.pointwise_storage,
            ),
            super::pointwise_data_layout(dag.n_inputs),
        ))
    } else if let Some(kernel) = dispatch.reduction() {
        let scratch =
            (1 + kernel.extra_prologues.len() as u64) * u64::from(kernel.workgroup_size) * 4 + 4;
        if scratch > u64::from(shared_limit) {
            return Err(format!(
                "reduction scratch plus queue ticket needs {scratch} shared bytes; limit is {shared_limit}"
            ));
        }
        Ok((
            crate::schedule::lower_tile(&kernel.to_template()).unwrap(),
            super::reduction_data_layout(kernel),
        ))
    } else {
        Err(format!("{:?} has no reusable tile body", dispatch.shader))
    }
}

impl State {
    pub fn pipeline(&self, index: usize) -> Option<&bg::ComputePipeline> {
        self.pipelines[index].as_ref().map(|pipeline| &pipeline.raw)
    }

    pub fn new(
        gpu: &Arc<Gpu>,
        plan: &ExecutionPlan,
        options: PersistentOptions,
        groups: &[std::ops::Range<usize>],
        dump_dir: Option<&str>,
    ) -> Self {
        use bg::ShaderData;
        super::validate_precision_plan(gpu, plan);
        assert!(
            (1..=65535).contains(&options.workers),
            "persistent workers must be in 1..=65535"
        );
        let mut pipelines = Vec::with_capacity(plan.dispatches.len());
        let mut report = ExecutionReport {
            persistent: Some(options),
            regions: Vec::new(),
            graph_dispatches: plan.dispatches.len(),
            persistent_tasks: 0,
            scheduler_bytes: 0,
            appended_fallback: Vec::new(),
        };
        for (index, dispatch) in plan.dispatches.iter().enumerate() {
            let count = dispatch
                .workgroups
                .into_iter()
                .map(u64::from)
                .product::<u64>();
            let mut region = ExecutionRegion {
                dispatches: index..index + 1,
                origin: dispatch.origin.clone(),
                label: dispatch.label.clone(),
                logical_tasks: count,
                workers: 0,
                workgroup_size: None,
                fallback_reason: None,
            };
            let generated = tile(dispatch, gpu.capabilities().max_compute_shared_memory_size)
                .and_then(|result| {
                    u32::try_from(count)
                        .ok()
                        .and_then(|count| count.checked_add(options.workers))
                        .ok_or_else(|| "queue counter would overflow u32".to_string())?;
                    Ok(result)
                });
            match generated {
                Ok((tile, layout)) => {
                    region.workers = options.workers;
                    region.workgroup_size = Some(tile.workgroup_size);
                    report.persistent_tasks += count;
                    let entry = include_str!("../shaders/persistent_tile.wgsl")
                        .replace("$WORKGROUP", &tile.workgroup_size.to_string());
                    let module = crate::codegen::ShaderModule::with_hint(
                        tile.source("") + &entry,
                        "persistent",
                    );
                    if let Some(dir) = dump_dir {
                        module.dump(dir);
                    }
                    let queue_layout = QueueData::layout();
                    let key = SharedPipelineKey {
                        context: Arc::as_ptr(gpu) as usize,
                        entry_point: "persistent_main",
                        layout: format!("{:?}/{:?}", layout.bindings, queue_layout.bindings),
                        source: module.source.clone(),
                    };
                    let pipeline = SharedPipelines::get(&key).unwrap_or_else(|| {
                        let shader = super::create_gen_shader(gpu, module)
                            .expect("persistent shader validation");
                        let pipeline = Arc::new(SharedPipeline {
                            gpu: Arc::clone(gpu),
                            raw: gpu.create_compute_pipeline(bg::ComputePipelineDesc {
                                name: "persistent tiles",
                                data_layouts: &[&layout, &queue_layout],
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
                }
            }
            report.regions.push(region);
        }
        let queue = if pipelines.iter().any(Option::is_some) {
            report.scheduler_bytes = plan.dispatches.len() * crate::memplan::ARENA_ALIGNMENT;
            super::ensure_device_memory_budget(gpu, report.scheduler_bytes, "persistent queues");
            Some(gpu.create_buffer(bg::BufferDesc {
                name: "persistent queues",
                size: report.scheduler_bytes as u64,
                memory: bg::Memory::Device,
            }))
        } else {
            None
        };
        log::info!(
            "persistent execution: {} / {} graph dispatches, {} logical tasks, {} workers, {} scheduler bytes",
            pipelines.iter().filter(|p| p.is_some()).count(),
            pipelines.len(),
            report.persistent_tasks,
            options.workers,
            report.scheduler_bytes
        );
        Self {
            gpu: Arc::clone(gpu),
            options,
            report,
            pipelines,
            queue,
            groups: groups.to_vec(),
        }
    }

    /// Recorded on the GPU, including when several steps share an encoder.
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
        let dispatch = &plan.dispatches[self.report.regions[index].dispatches.start];
        let mut pc = pass.with(&pipeline.raw);
        Session::bind_dispatch(buffers, dispatch, &mut pc);
        pc.bind(
            1,
            &QueueData {
                persistent_queue: self
                    .queue
                    .unwrap()
                    .at((index * crate::memplan::ARENA_ALIGNMENT) as u64),
                persistent_params: Params {
                    grid: dispatch.workgroups,
                    count: self.report.regions[index].logical_tasks as u32,
                },
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
            self.wait();
            self.persistent = Some(State::new(
                &self.gpu,
                &self.plan,
                options,
                &self.groups,
                self.pipelines.dump_dir.as_deref(),
            ));
        }
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
