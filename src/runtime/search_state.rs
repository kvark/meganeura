//! Reset explicit persistent state between complete-plan search trials.
use super::{PhysicalBuffer, Session, safe_device_memory_remaining};
use crate::compile::{BufferRef, ExecutionPlan};
use blade_graphics as bg;
use std::{collections::HashSet, sync::Arc};

pub(crate) fn persistent_writes(plan: &ExecutionPlan) -> Vec<BufferRef> {
    let writes: HashSet<_> = plan
        .dispatches
        .iter()
        .flat_map(|d| std::iter::once(d.output_buffer).chain(d.extra_outputs.iter().copied()))
        .collect();
    let mut buffers: Vec<_> = plan
        .input_buffers
        .iter()
        .chain(&plan.param_buffers)
        .map(|entry| entry.1)
        .chain(plan.constant_buffers.iter().map(|entry| entry.0))
        .filter(|b| writes.contains(b))
        .collect();
    buffers.sort_unstable_by_key(|b| b.0);
    buffers.dedup();
    buffers
}

pub(crate) struct SearchState {
    images: Vec<SavedBuffer>,
}

struct SavedBuffer {
    live: Arc<PhysicalBuffer>,
    original: PhysicalBuffer,
    bytes: usize,
}

impl SearchState {
    pub fn capture(session: &mut Session, max_bytes: usize) -> Result<Self, String> {
        // These passes have state outside the compiled plan. Do not silently
        // measure a different workload or advance their host-side counters.
        if session.pending_lr.is_some()
            || session.pending_adam.is_some()
            || session.grad_accum_scale.is_some()
        {
            return Err(
                "configure runtime optimizers and accumulation after program search".into(),
            );
        }
        session.wait();
        let mut physical: Vec<_> = persistent_writes(&session.plan)
            .iter()
            .map(|b| session.alias.map[b.0 as usize])
            .collect();
        physical.sort_unstable();
        physical.dedup();
        let mut bytes = 0usize;
        for &p in &physical {
            if Arc::strong_count(&session.physical_buffers[p]) != 1 {
                return Err("program search requires private writable state; do not share mutable parameters".into());
            }
            bytes = bytes
                .checked_add(session.alias.sizes[p].max(4))
                .filter(|&n| n <= max_bytes)
                .ok_or("program search state snapshot exceeds its byte budget")?;
        }
        let memory = session.gpu.memory_stats();
        if memory.budget != 0
            && bytes as u64 > safe_device_memory_remaining(memory.usage, memory.budget)
        {
            return Err("program search state snapshot exceeds the device memory budget".into());
        }
        let state = Self {
            images: physical
                .into_iter()
                .map(|p| {
                    let size = session.alias.sizes[p].max(4);
                    let saved = PhysicalBuffer {
                        gpu: session.gpu.clone(),
                        handle: session.gpu.create_buffer(bg::BufferDesc {
                            name: "program_search_state",
                            size: size as u64,
                            memory: bg::Memory::DeviceTransient,
                        }),
                    };
                    SavedBuffer {
                        live: session.physical_buffers[p].clone(),
                        original: saved,
                        bytes: size,
                    }
                })
                .collect(),
        };
        state.copy(session, false)?;
        Ok(state)
    }

    pub fn restore(&self, session: &mut Session) -> Result<(), String> {
        self.copy(session, true)
    }

    pub fn bytes(&self) -> usize {
        self.images.iter().map(|image| image.bytes).sum()
    }

    fn copy(&self, session: &mut Session, restore: bool) -> Result<(), String> {
        if self.images.is_empty() {
            return Ok(());
        }
        session.wait();
        session.encoder.start();
        {
            let mut pass = session.encoder.transfer("program_search_state");
            for image in &self.images {
                let (live, saved) = (image.live.handle, image.original.handle);
                let (src, dst) = if restore {
                    (saved, live)
                } else {
                    (live, saved)
                };
                pass.copy_buffer_to_buffer(src.at(0), dst.at(0), image.bytes as u64);
            }
        }
        let sync = session.gpu.submit(&mut session.encoder);
        session.sync_point = Some(sync.clone());
        if !session
            .gpu
            .wait_for(&sync, !0)
            .map_err(|_| "state copy GPU wait failed")?
        {
            return Err("state copy GPU wait did not complete".into());
        }
        session.sync_point = None;
        Ok(())
    }
}
