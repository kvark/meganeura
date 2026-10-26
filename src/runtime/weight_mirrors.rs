use super::*;
use std::sync::{Weak, atomic::Ordering};

struct Stamp {
    source: Weak<PhysicalBuffer>,
    destination: Weak<PhysicalBuffer>,
    source_offset: u64,
    destination_offset: u64,
    elements: u32,
    epoch: u64,
}

impl Stamp {
    fn matches_live(
        &self,
        source: &Arc<PhysicalBuffer>,
        destination: &Arc<PhysicalBuffer>,
        source_offset: u64,
        destination_offset: u64,
        elements: u32,
        epoch: u64,
    ) -> bool {
        self.source.as_ptr() == Arc::as_ptr(source)
            && self.destination.as_ptr() == Arc::as_ptr(destination)
            && self.source_offset == source_offset
            && self.destination_offset == destination_offset
            && self.elements == elements
            && self.epoch == epoch
    }
}

struct Fallback {
    plan: ExecutionPlan,
    pipelines: Pipelines,
}

pub(super) struct WeightMirrors {
    fallback: Option<Fallback>,
    pipeline: Option<Arc<SharedPipeline>>,
    committed: Vec<Option<Stamp>>,
    pending: Vec<(usize, Stamp)>,
}

impl WeightMirrors {
    pub(super) fn track_sources(
        plan: &ExecutionPlan,
        owners: &[Arc<PhysicalBuffer>],
        alias: &crate::memplan::AliasPlan,
    ) {
        for &(source, _) in &plan.f16_weight_mirrors {
            // Zero means never observed by a cache. Enroll the shared backing
            // without resetting an epoch already used by another session.
            let _ = owners[alias.map[source.0 as usize]]
                .cpu_epoch
                .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire);
        }
    }

    pub(super) fn new(
        gpu: &Arc<Gpu>,
        plan: &ExecutionPlan,
        coop: Option<&crate::codegen::CoopConfig>,
    ) -> Self {
        let pipeline = (!plan.f16_weight_mirrors.is_empty()).then(|| {
            let d = Dispatch {
                shader: ShaderEntry::ToF16,
                ..Default::default()
            };
            Pipelines::compile(gpu, plan.knobs, None, &d, &Pipelines::key(&d), None).unwrap()
        });
        let fallback = (!plan.f16_weight_mirrors.is_empty()).then(|| {
            let parent = *plan
                .f16_weight_parent
                .as_ref()
                .expect("mirror plan must retain its qualified parent")
                .clone();
            assert!(parent.f16_weight_mirrors.is_empty() && parent.f16_weight_parent.is_none());
            assert_eq!(parent.dispatches.len(), plan.dispatches.len());
            assert!(
                parent
                    .buffers
                    .iter()
                    .enumerate()
                    .all(|(i, &bytes)| plan.buffers[i] >= bytes),
                "mirror plan shrunk a buffer needed by its parent"
            );
            let pipelines = Pipelines::new(gpu, &parent, coop, None);
            // Only readonly weight bindings differ. Keep the primary's common
            // dispatch order, barrier groups and alias lifetimes for both paths.
            Fallback {
                plan: parent,
                pipelines,
            }
        });
        Self {
            fallback,
            pipeline,
            committed: (0..plan.f16_weight_mirrors.len()).map(|_| None).collect(),
            pending: Vec::new(),
        }
    }
    pub(super) fn commit(&mut self) {
        for (index, stamp) in self.pending.drain(..) {
            self.committed[index] = Some(stamp);
        }
    }
}

impl Session {
    fn unmanaged_weight_source(&self) -> bool {
        self.plan.f16_weight_mirrors.iter().any(|&(source, _)| {
            self.physical_buffers[self.alias.map[source.0 as usize]]
                .gpu_writes
                .load(Ordering::Acquire)
        })
    }

    /// The executable an owned step will use. Profiling and descriptive keys
    /// must follow the same writer policy as command encoding.
    pub(super) fn step_execution(&self) -> (&ExecutionPlan, &Pipelines) {
        if self.unmanaged_weight_source()
            && let Some(parent) = self.weight_mirrors.fallback.as_ref()
        {
            (&parent.plan, &parent.pipelines)
        } else {
            (&self.plan, &self.pipelines)
        }
    }

    pub(crate) fn step_plan(&self) -> &ExecutionPlan {
        self.step_execution().0
    }

    pub(super) fn encode_weight_policy(
        &mut self,
        chunks: usize,
        window: Option<std::ops::Range<usize>>,
        owned: bool,
    ) {
        let unmanaged = self.unmanaged_weight_source();
        if (!owned || unmanaged)
            && let Some(mut fallback) = self.weight_mirrors.fallback.take()
        {
            std::mem::swap(&mut self.plan, &mut fallback.plan);
            std::mem::swap(&mut self.pipelines, &mut fallback.pipelines);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.encode_step_body(chunks, window)
            }));
            std::mem::swap(&mut self.plan, &mut fallback.plan);
            std::mem::swap(&mut self.pipelines, &mut fallback.pipelines);
            self.weight_mirrors.fallback = Some(fallback);
            if let Err(panic) = result {
                std::panic::resume_unwind(panic);
            }
        } else {
            self.encode_weight_mirrors(false);
            self.encode_step_body(chunks, window);
        }
    }
    pub(super) fn mark_host_write(&self, buffer: BufferRef) {
        // Track the backing owner even in an unmirrored session: a cache in
        // another session may share this parameter. The logical map avoids
        // searching allocations on ordinary input/parameter uploads.
        let epoch = &self.physical_buffers[self.alias.map[buffer.0 as usize]].cpu_epoch;
        if epoch.load(Ordering::Acquire) != 0 {
            epoch.fetch_add(1, Ordering::AcqRel);
        }
    }

    pub(super) fn encode_weight_mirrors(&mut self, start_encoder: bool) -> bool {
        if self.plan.f16_weight_mirrors.is_empty() {
            return false;
        }
        assert_eq!(
            self.plan.f16_weight_mirrors.len(),
            self.weight_mirrors.committed.len()
        );
        self.weight_mirrors.pending.clear();
        let mut copies = Vec::new();
        for (index, &(source, destination)) in self.plan.f16_weight_mirrors.iter().enumerate() {
            let src = self.buffers[source.0 as usize];
            let dst = self.buffers[destination.0 as usize];
            let source_owner = &self.physical_buffers[self.alias.map[source.0 as usize]];
            let destination_owner = &self.physical_buffers[self.alias.map[destination.0 as usize]];
            let elements = u32::try_from(self.plan.buffers[source.0 as usize] / 4).unwrap();
            assert!(
                self.plan.buffers[destination.0 as usize]
                    >= usize::try_from(elements).unwrap().div_ceil(2) * 4
            );
            let epoch = source_owner.cpu_epoch.load(Ordering::Acquire);
            let reusable = !source_owner.gpu_writes.load(Ordering::Acquire);
            if reusable
                && self.weight_mirrors.committed[index]
                    .as_ref()
                    .is_some_and(|old| {
                        old.matches_live(
                            source_owner,
                            destination_owner,
                            src.offset,
                            dst.offset,
                            elements,
                            epoch,
                        )
                    })
            {
                continue;
            }
            copies.push((src, dst, elements));
            if reusable {
                let stamp = Stamp {
                    source: Arc::downgrade(source_owner),
                    destination: Arc::downgrade(destination_owner),
                    source_offset: src.offset,
                    destination_offset: dst.offset,
                    elements,
                    epoch,
                };
                self.weight_mirrors.pending.push((index, stamp));
            }
        }
        if copies.is_empty() {
            return false;
        }
        if start_encoder {
            self.encoder.start();
        }
        let pipeline = self.weight_mirrors.pipeline.as_ref().unwrap().clone();
        {
            let mut pass = self.encoder.compute("weight_mirrors");
            for (src, dst, len) in copies {
                let mut p = pass.with(&pipeline.raw);
                p.bind(
                    0,
                    &UnaryData {
                        src,
                        dst,
                        params: UnaryParams {
                            len,
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
                p.dispatch([len.div_ceil(512), 1, 1]);
            }
        }
        self.profiled_pass_map.push(None);
        true
    }

    pub(super) fn prepare_weight_mirrors(&mut self) {
        if self.plan.f16_weight_mirrors.is_empty() {
            return;
        }
        self.wait();
        if self.encode_weight_mirrors(true) {
            self.sync_point = Some(self.gpu.submit(&mut self.encoder));
            self.weight_mirrors.commit();
            self.wait();
        }
    }
}
