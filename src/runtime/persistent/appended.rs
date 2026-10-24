use super::{
    AppendedDispatch, Gpu, Params, PersistentOptions, QueueData, SharedPipeline,
    check_handler_resources, compile_region, tile,
};
use crate::compile::{Dispatch, ExecutionPlan, ShaderEntry};
use blade_graphics as bg;
use std::{collections::HashMap, sync::Arc};

struct Program {
    pipeline: Arc<SharedPipeline>,
    key: String,
    shape: [u32; 3],
}

struct Ready {
    programs: HashMap<ShaderEntry, Program>,
    queue: bg::Buffer,
    stride: usize,
    slots: usize,
}

pub(crate) struct Appended {
    gpu: Arc<Gpu>,
    options: PersistentOptions,
    ready: Result<Ready, String>,
}

impl Appended {
    pub fn new(
        gpu: &Arc<Gpu>,
        plan: &ExecutionPlan,
        options: PersistentOptions,
        dump_dir: Option<&str>,
    ) -> Self {
        let portable = PersistentOptions {
            device_scheduling: false,
            ..options
        };
        let ready = (|| {
            let caps = gpu.capabilities();
            let mut programs = HashMap::new();
            for entry in super::super::optimizer::ENTRIES {
                let dispatch = Dispatch {
                    shader: entry.clone(),
                    ..Default::default()
                };
                let handler = tile(gpu, plan, &dispatch, None)?;
                check_handler_resources(
                    &caps,
                    std::slice::from_ref(&handler),
                    options.device_scheduling,
                )?;
                let shape = handler.body.workgroup_size;
                let (pipeline, key) =
                    compile_region(gpu, vec![handler], portable, false, dump_dir)?;
                programs.insert(
                    entry,
                    Program {
                        pipeline,
                        key,
                        shape,
                    },
                );
            }
            // At most one accumulation pass, three clipping passes and one
            // update. Even after rebinding, each unit contains at least one
            // parameter pair. Separate slots preserve every pass's diagnostics.
            let slots = plan
                .param_grad_pairs
                .len()
                .checked_mul(5)
                .ok_or("optimizer queue slot count overflow")?;
            let stride = portable.queue_stride(0);
            let bytes = slots
                .checked_mul(stride)
                .ok_or("optimizer queue size overflow")?;
            if bytes as u64
                > caps
                    .compute_binding_limits
                    .unwrap()
                    .max_storage_buffer_range
            {
                return Err(
                    "optimizer queue allocation exceeds device storage-buffer range".into(),
                );
            }
            super::super::ensure_device_memory_budget(gpu, bytes, "persistent optimizer queues");
            let queue = gpu.create_buffer(bg::BufferDesc {
                name: "persistent optimizer queues",
                size: bytes as u64,
                memory: bg::Memory::Device,
            });
            Ok(Ready {
                programs,
                queue,
                stride,
                slots,
            })
        })();
        Self {
            gpu: Arc::clone(gpu),
            options: portable,
            ready,
        }
    }

    pub fn error(&self) -> Option<&str> {
        self.ready.as_ref().err().map(String::as_str)
    }

    pub fn queue(&self) -> Option<bg::Buffer> {
        self.ready.as_ref().ok().map(|ready| ready.queue)
    }

    pub fn bytes(&self) -> usize {
        self.queue().map_or(0, |queue| queue.size() as usize)
    }

    pub fn describe(
        &self,
        entry: &ShaderEntry,
        label: &'static str,
        grid: [u32; 3],
    ) -> AppendedDispatch {
        let mut record = AppendedDispatch::ordinary(entry, label, grid);
        match self.ready {
            Ok(ref ready) => {
                let program = &ready.programs[entry];
                record.pipeline_key = program.key.clone();
                record.workers = self.options.workers;
                record.workgroup_shape = Some(program.shape);
            }
            Err(ref reason) => record.fallback_reason = Some(reason.clone()),
        }
        record
    }

    pub fn record<D: bg::ShaderData>(
        &self,
        entry: &ShaderEntry,
        slot: usize,
        grid: [u32; 3],
        data: &D,
        pass: &mut bg::ComputeCommandEncoder<'_>,
    ) {
        let ready = self.ready.as_ref().expect("optimizer tile admission");
        assert!(
            slot < ready.slots,
            "optimizer exceeded its five-pass queue bound"
        );
        let tasks = grid
            .into_iter()
            .try_fold(1u32, u32::checked_mul)
            .filter(|tasks| tasks.checked_add(self.options.workers).is_some())
            .expect("optimizer queue counter capacity");
        let mut pc = pass.with(&ready.programs[entry].pipeline.raw);
        pc.bind(0, data);
        pc.bind(
            1,
            &QueueData {
                persistent_queue: ready.queue.at((slot * ready.stride) as u64),
                persistent_params: Params {
                    grids: [[grid[0], grid[1], grid[2], 0], [0; 4], [0; 4]],
                    starts: [0; 4],
                    ends: [tasks, 0, 0, tasks],
                    successors: [0; 4],
                },
            },
        );
        pc.dispatch([self.options.workers, 1, 1]);
    }

    pub fn check_completion(
        &self,
        words: &[u32],
        records: &[AppendedDispatch],
    ) -> Result<(), String> {
        let Ok(ref ready) = self.ready else {
            return Ok(());
        };
        for (slot, record) in records.iter().enumerate() {
            if record.workers == 0 {
                continue;
            }
            let expected = record.logical_tasks + u64::from(record.workers);
            let actual = words[slot * ready.stride / 4];
            if u64::from(actual) != expected {
                return Err(format!(
                    "{} queue {slot}: {actual} claims, expected {expected}",
                    record.label
                ));
            }
        }
        Ok(())
    }

    pub fn worker_tasks(&self, words: &[u32], records: &[AppendedDispatch]) -> Vec<Vec<u32>> {
        let ready = self.ready.as_ref().expect("optimizer tile admission");
        records
            .iter()
            .enumerate()
            .map(|(slot, record)| {
                let start = slot * ready.stride / 4 + 1;
                words[start..start + record.workers as usize].to_vec()
            })
            .collect()
    }

    pub fn pipelines(&self) -> impl Iterator<Item = (&str, &bg::ComputePipeline)> {
        self.ready.iter().flat_map(|ready| {
            ready
                .programs
                .values()
                .map(|program| (program.key.as_str(), &program.pipeline.raw))
        })
    }
}

impl Drop for Appended {
    fn drop(&mut self) {
        if let Some(queue) = self.queue() {
            self.gpu.destroy_buffer(queue);
        }
    }
}
