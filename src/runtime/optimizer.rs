//! Optimizer passes over the parameter arena.
//!
//! Trainable parameters and their gradients live in arena chunks
//! ([`crate::memplan::ArenaChunk`]); moments and gradient accumulators are
//! allocated per chunk with the same layout. Every pass (accumulation,
//! clipping, SGD, Adam) runs one dispatch per chunk. Each workgroup finds
//! its parameter in a table of segments, so per-parameter settings such as
//! learning-rate multipliers need no dispatch of their own.
//!
//! A parameter that was rebound after the session was built (to an external
//! allocation, or shared from another session) leaves its chunk and is
//! updated by a dispatch of its own, reading its moments from the same slot.

use super::{Session, create_optimizer_buffer, wait_for_timed_encoder};
use crate::compile::{ExecutionPlan, ShaderEntry};
use crate::memplan::{ARENA_ALIGNMENT, AliasPlan};
use blade_graphics::{Buffer, BufferPiece};
use bytemuck::Zeroable;

/// Elements one workgroup covers; must match `TILE` in
/// `optimizer_segments.wgsl`.
pub(super) const TILE: u32 = 1024;

/// Workgroups in x before a grid wraps into y.
const GRID_WIDTH: u32 = 32768;

/// Slots of trainable pairs sharing one layout for parameters, gradients,
/// moments and accumulators.
pub(super) struct Chunk {
    /// Bytes spanned by the slots.
    pub bytes: usize,
    /// `(index into param_grad_pairs, byte offset)` of each slot.
    slots: Vec<(usize, usize)>,
    /// Physical allocations of the parameter and gradient arenas, when the
    /// chunk is one; otherwise a single pair in allocations of its own.
    arena: Option<(usize, usize)>,
}

/// The chunks for `plan`: its arena chunks, then one per pair outside them.
pub(super) fn chunks(plan: &ExecutionPlan, alias: &AliasPlan) -> Vec<Chunk> {
    let mut chunks: Vec<Chunk> = alias
        .arena
        .iter()
        .map(|chunk| Chunk {
            bytes: chunk.bytes,
            slots: chunk.slots.clone(),
            arena: Some((chunk.params, chunk.grads)),
        })
        .collect();
    let mut packed = vec![false; plan.param_grad_pairs.len()];
    for chunk in &chunks {
        for &(index, _) in &chunk.slots {
            packed[index] = true;
        }
    }
    for (index, &(param, grad)) in plan.param_grad_pairs.iter().enumerate() {
        if !packed[index] {
            chunks.push(Chunk {
                bytes: plan.buffers[param.0 as usize]
                    .max(plan.buffers[grad.0 as usize])
                    .max(4),
                slots: vec![(index, 0)],
                arena: None,
            });
        }
    }
    chunks
}

/// Per-chunk allocations with the chunks' layout (moments, accumulators),
/// and each pair's piece of them.
pub(super) struct ChunkBuffers {
    pub buffers: Vec<Buffer>,
    pub pieces: Vec<BufferPiece>,
}

impl ChunkBuffers {
    pub fn new(session: &Session, name: &str, device_bufs: &mut Vec<(Buffer, u64)>) -> Self {
        let buffers: Vec<Buffer> = session
            .optimizer_chunks
            .iter()
            .enumerate()
            .map(|(index, chunk)| {
                create_optimizer_buffer(
                    &session.gpu,
                    &format!("{name}_{index}"),
                    chunk.bytes as u64,
                    session.optimizer_device,
                    device_bufs,
                )
            })
            .collect();
        let mut pieces = vec![None; session.plan.param_grad_pairs.len()];
        for (chunk, &buffer) in session.optimizer_chunks.iter().zip(&buffers) {
            for &(index, offset) in &chunk.slots {
                pieces[index] = Some(buffer.at(offset as u64));
            }
        }
        Self {
            buffers,
            pieces: pieces
                .into_iter()
                .map(|piece| piece.expect("every pair has a slot"))
                .collect(),
        }
    }

    pub fn bytes(session: &Session) -> usize {
        session
            .optimizer_chunks
            .iter()
            .map(|chunk| chunk.bytes)
            .sum()
    }
}

/// One entry of the segment table; see `optimizer_segments.wgsl`.
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct Segment {
    offset: u32,
    len: u32,
    first_group: u32,
    index: u32,
    lr_scale: f32,
    _pad: [u32; 3],
}

/// What one dispatch of a pass covers: a whole chunk, or one pair that left
/// its chunk.
struct Unit {
    chunk: usize,
    /// Byte offset of the unit within the chunk's moments and accumulators.
    base: usize,
    param: BufferPiece,
    /// This step's gradients.
    grad: BufferPiece,
    /// The gradients clipping and the update read: `grad`, or the
    /// persistent accumulators when temporal accumulation is active.
    source: BufferPiece,
    segments: BufferPiece,
    count: u32,
    groups: u32,
    /// First partial-norm slot of this unit's workgroups.
    slot: u32,
}

impl Unit {
    fn grid(&self) -> [u32; 3] {
        grid(self.groups)
    }
}

fn grid(groups: u32) -> [u32; 3] {
    let width = groups.clamp(1, GRID_WIDTH);
    [width, groups.div_ceil(width).max(1), 1]
}

/// Partial-norm slots the clip passes need for `plan`: one per workgroup.
pub(super) fn clip_slots(plan: &ExecutionPlan) -> u64 {
    plan.param_grad_pairs
        .iter()
        .map(|&(param, _)| u64::from(Session::optimizer_len(plan, param).div_ceil(TILE)))
        .sum()
}

#[derive(blade_macros::ShaderData)]
pub(super) struct SgdData {
    segments: BufferPiece,
    param: BufferPiece,
    grad: BufferPiece,
    params: SgdParams,
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct SgdParams {
    count: u32,
    groups: u32,
    lr: f32,
    _pad0: u32,
}

#[derive(blade_macros::ShaderData)]
pub(super) struct AdamData {
    segments: BufferPiece,
    param: BufferPiece,
    grad: BufferPiece,
    m: BufferPiece,
    v: BufferPiece,
    grouped_grad_norm: BufferPiece,
    params: AdamParams,
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct AdamParams {
    count: u32,
    groups: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    step: f32,
    wd: f32,
    grad_group_size: u32,
    grouped_index: u32,
    algorithm: u32,
    _pad0: u32,
}

/// Gradient accumulation, global clip norm and global clip scale share one
/// binding layout.
#[derive(blade_macros::ShaderData)]
pub(super) struct GradPassData {
    segments: BufferPiece,
    grad: BufferPiece,
    acc: BufferPiece,
    params: GradPassParams,
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct GradPassParams {
    count: u32,
    groups: u32,
    /// Accumulation scale, clip norm, or partial slot, by pass.
    value: u32,
    /// Clip norm: 1 to square a tile, 0 to total the partials.
    square: u32,
}

#[derive(blade_macros::ShaderData)]
pub(super) struct AdaptiveGradClipData {
    segments: BufferPiece,
    param: BufferPiece,
    grad: BufferPiece,
    partials: BufferPiece,
    scales: BufferPiece,
    params: AdaptiveGradClipParams,
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct AdaptiveGradClipParams {
    count: u32,
    groups: u32,
    slot: u32,
    mode: u32,
    clip: f32,
    pmin: f32,
    _pad0: u32,
    _pad1: u32,
}

/// The update a pass applies after the gradients are final.
pub(super) enum Update {
    Sgd {
        lr: f32,
    },
    Adam {
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        algorithm: super::AdaptiveOptimizer,
    },
}

impl Session {
    /// Split the chunks into dispatch units and write their segment table.
    /// The GPU must be idle: the table is host-visible and rewritten here.
    fn optimizer_units(&mut self, accumulating: bool) -> Vec<Unit> {
        struct Plan {
            chunk: usize,
            base: usize,
            param: BufferPiece,
            grad: BufferPiece,
            /// `(pair, element offset from the unit's base)`.
            pairs: Vec<(usize, u32)>,
        }
        let mut plans = Vec::new();
        for (chunk_index, chunk) in self.optimizer_chunks.iter().enumerate() {
            let pieces = |index: usize| {
                let (param, grad) = self.plan.param_grad_pairs[index];
                (
                    self.buffers[param.0 as usize],
                    self.buffers[grad.0 as usize],
                )
            };
            let mut attached = Vec::new();
            for &(index, offset) in &chunk.slots {
                let (param, grad) = self.plan.param_grad_pairs[index];
                let in_arena = chunk.arena.is_some_and(|(params, grads)| {
                    self.alias.map[param.0 as usize] == params
                        && self.alias.map[grad.0 as usize] == grads
                });
                if in_arena {
                    attached.push((index, (offset / 4) as u32));
                } else {
                    let (param, grad) = pieces(index);
                    plans.push(Plan {
                        chunk: chunk_index,
                        base: offset,
                        param,
                        grad,
                        pairs: vec![(index, 0)],
                    });
                }
            }
            if let (Some((params, grads)), false) = (chunk.arena, attached.is_empty()) {
                plans.push(Plan {
                    chunk: chunk_index,
                    base: 0,
                    param: self.physical_buffers[params].handle.at(0),
                    grad: self.physical_buffers[grads].handle.at(0),
                    pairs: attached,
                });
            }
        }

        // Each unit's table starts at a storage-binding offset.
        let per_block = ARENA_ALIGNMENT / std::mem::size_of::<Segment>();
        let mut table: Vec<Segment> = Vec::new();
        let mut units = Vec::with_capacity(plans.len());
        let mut slot = 0u32;
        for plan in plans {
            table.resize(table.len().next_multiple_of(per_block), Segment::zeroed());
            let start = table.len();
            let mut groups = 0u32;
            for &(index, offset) in &plan.pairs {
                let (param, _) = self.plan.param_grad_pairs[index];
                let len = Self::optimizer_len(&self.plan, param);
                table.push(Segment {
                    offset,
                    len,
                    first_group: groups,
                    index: index as u32,
                    lr_scale: Self::lr_multiplier_for_buf(
                        &self.plan.param_buffers,
                        &self.lr_multipliers,
                        param,
                    ),
                    _pad: [0; 3],
                });
                groups += len.div_ceil(TILE);
            }
            units.push((plan, start, groups, slot));
            slot += groups;
        }
        let bytes = (std::mem::size_of_val(table.as_slice()) as u64).max(4);
        if self
            .optimizer_segments
            .is_none_or(|buffer| buffer.size() < bytes)
        {
            if let Some(buffer) = self.optimizer_segments.take() {
                self.gpu.destroy_buffer(buffer);
            }
            self.optimizer_segments = Some(self.gpu.create_buffer(blade_graphics::BufferDesc {
                name: "optimizer_segments",
                size: bytes,
                memory: blade_graphics::Memory::Shared,
            }));
        }
        let segments = self.optimizer_segments.unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(
                table.as_ptr().cast::<u8>(),
                segments.data(),
                std::mem::size_of_val(table.as_slice()),
            );
        }
        let accumulators = self.grad_accum.as_ref().filter(|_| accumulating);
        units
            .into_iter()
            .map(|(plan, start, groups, slot)| Unit {
                chunk: plan.chunk,
                base: plan.base,
                param: plan.param,
                grad: plan.grad,
                source: accumulators.map_or(plan.grad, |accumulators| {
                    accumulators.buffers[plan.chunk].at(plan.base as u64)
                }),
                segments: segments.at((start * std::mem::size_of::<Segment>()) as u64),
                count: plan.pairs.len() as u32,
                groups,
                slot,
            })
            .collect()
    }

    /// Record the passes that follow backward: gradient accumulation, then
    /// (when due) clipping, then `update`. The encoder must be started and
    /// the GPU idle.
    pub(super) fn encode_optimizer(&mut self, update: Option<Update>, clip: bool) {
        if self.plan.param_grad_pairs.is_empty() {
            return;
        }
        let accumulating = self.grad_accum_scale.is_some();
        let units = self.optimizer_units(accumulating);

        // Temporal grad accumulation: add this step's (overwritten) grads
        // into the persistent accumulators that the clip/optimizer below
        // will read.
        if let Some(scale) = self.grad_accum_scale {
            {
                let pipeline = self.pipelines.scalar(ShaderEntry::GradAccum);
                let mut pass = self.encoder.compute("grad_accum");
                for unit in &units {
                    let mut pc = pass.with(pipeline);
                    pc.bind(
                        0,
                        &GradPassData {
                            segments: unit.segments,
                            grad: unit.grad,
                            acc: unit.source,
                            params: GradPassParams {
                                count: unit.count,
                                groups: unit.groups,
                                value: scale.to_bits(),
                                square: 0,
                            },
                        },
                    );
                    pc.dispatch(unit.grid());
                }
            }
            // Submit + wait so the accumulator this pass wrote is fully
            // durable before the optimizer's separate passes read it.
            // Sharing one submission left a write/read hazard that
            // corrupted the accumulator on the apply step.
            self.sync_point = Some(self.gpu.submit(&mut self.encoder));
            self.wait();
            self.encoder.start();
        }

        if clip {
            if let Some((clip, pmin)) = self.pending_agc {
                self.encode_adaptive_clip(&units, clip, pmin);
            } else if let Some(max_norm) = self.pending_grad_clip {
                self.encode_global_clip(&units, max_norm);
            }
        }

        match update {
            None => {}
            Some(Update::Sgd { lr }) => {
                let pipeline = self.pipelines.scalar(ShaderEntry::SgdUpdate);
                let mut pass = self.encoder.compute("sgd_update");
                for unit in &units {
                    let mut pc = pass.with(pipeline);
                    pc.bind(
                        0,
                        &SgdData {
                            segments: unit.segments,
                            param: unit.param,
                            grad: unit.source,
                            params: SgdParams {
                                count: unit.count,
                                groups: unit.groups,
                                lr,
                                _pad0: 0,
                            },
                        },
                    );
                    pc.dispatch(unit.grid());
                }
            }
            Some(Update::Adam {
                lr,
                beta1,
                beta2,
                eps,
                algorithm,
            }) => {
                self.ensure_adam_state();
                self.adam_step += 1;
                let moments = self.adam_state.as_ref().expect("Adam state");
                let (grouped_grad_norm, grad_group_size, grouped_index) =
                    match self.adam_grouped_grad_norm {
                        Some(ref value) => (
                            value.buffer.at(0),
                            value.group_size,
                            value.param_index as u32,
                        ),
                        None => (
                            self.grad_clip_acc.expect("optimizer scratch").at(0),
                            0,
                            u32::MAX,
                        ),
                    };
                let pipeline = self.pipelines.scalar(ShaderEntry::AdamUpdate);
                let mut pass = self.encoder.compute("adam_update");
                for unit in &units {
                    let mut pc = pass.with(pipeline);
                    pc.bind(
                        0,
                        &AdamData {
                            segments: unit.segments,
                            param: unit.param,
                            grad: unit.source,
                            m: moments.0.buffers[unit.chunk].at(unit.base as u64),
                            v: moments.1.buffers[unit.chunk].at(unit.base as u64),
                            grouped_grad_norm,
                            params: AdamParams {
                                count: unit.count,
                                groups: unit.groups,
                                lr,
                                beta1,
                                beta2,
                                eps,
                                step: self.adam_step as f32,
                                wd: self.adam_wd,
                                grad_group_size,
                                grouped_index,
                                algorithm: algorithm as u32,
                                _pad0: 0,
                            },
                        },
                    );
                    pc.dispatch(unit.grid());
                }
            }
        }
    }

    /// Global clipping in three passes, all in the step's submission:
    /// squared partial sums (every workgroup its own slot, so no barriers
    /// between chunks), their total in one workgroup, then the scaling.
    fn encode_global_clip(&mut self, units: &[Unit], max_norm: f32) {
        let acc = self.grad_clip_acc.expect("optimizer scratch");
        let partials = self.grad_clip_partials.expect("optimizer scratch");
        let pipeline = self.pipelines.scalar(ShaderEntry::GradClipNormSq);
        let slots: u32 = units.iter().map(|unit| unit.groups).sum();
        {
            let mut pass = self.encoder.compute("grad_clip_norm_sq");
            for unit in units {
                let mut pc = pass.with(pipeline);
                pc.bind(
                    0,
                    &GradPassData {
                        segments: unit.segments,
                        grad: unit.source,
                        acc: partials.at(0),
                        params: GradPassParams {
                            count: unit.count,
                            groups: unit.groups,
                            value: unit.slot,
                            square: 1,
                        },
                    },
                );
                pc.dispatch(unit.grid());
            }
        }
        {
            let mut pass = self.encoder.compute("grad_clip_norm_total");
            let mut pc = pass.with(pipeline);
            pc.bind(
                0,
                &GradPassData {
                    segments: units[0].segments,
                    grad: partials.at(0),
                    acc: acc.at(0),
                    params: GradPassParams {
                        count: 0,
                        groups: slots,
                        value: 0,
                        square: 0,
                    },
                },
            );
            pc.dispatch([1, 1, 1]);
        }
        let pipeline = self.pipelines.scalar(ShaderEntry::GradClipScale);
        let mut pass = self.encoder.compute("grad_clip_scale");
        for unit in units {
            let mut pc = pass.with(pipeline);
            pc.bind(
                0,
                &GradPassData {
                    segments: unit.segments,
                    grad: unit.source,
                    acc: acc.at(0),
                    params: GradPassParams {
                        count: unit.count,
                        groups: unit.groups,
                        value: max_norm.to_bits(),
                        square: 0,
                    },
                },
            );
            pc.dispatch(unit.grid());
        }
    }

    /// Adaptive clipping in three passes, as for global clipping:
    /// per-workgroup norm pairs, one scale per parameter, then the scaling.
    fn encode_adaptive_clip(&mut self, units: &[Unit], clip: f32, pmin: f32) {
        let partials = self.grad_clip_partials.expect("optimizer scratch");
        let scales = self.agc_scales.expect("optimizer scratch");
        let pipeline = self.pipelines.scalar(ShaderEntry::AdaptiveGradClip);
        for (mode, label) in [(0u32, "agc_norms"), (1, "agc_factor"), (2, "agc_apply")] {
            let mut pass = self.encoder.compute(label);
            for unit in units {
                let mut pc = pass.with(pipeline);
                pc.bind(
                    0,
                    &AdaptiveGradClipData {
                        segments: unit.segments,
                        param: unit.param,
                        grad: unit.source,
                        partials: partials.at(0),
                        scales: scales.at(0),
                        params: AdaptiveGradClipParams {
                            count: unit.count,
                            groups: unit.groups,
                            slot: unit.slot,
                            mode,
                            clip,
                            pmin,
                            _pad0: 0,
                            _pad1: 0,
                        },
                    },
                );
                pc.dispatch(if mode == 1 {
                    grid(unit.count)
                } else {
                    unit.grid()
                });
            }
        }
    }

    /// Run `update` now, in a submission of its own.
    pub(super) fn run_optimizer(&mut self, update: Update) {
        self.wait();
        self.encoder.start();
        self.encode_optimizer(Some(update), false);
        self.sync_point = Some(self.gpu.submit(&mut self.encoder));
    }

    /// Allocate the Adam moments, zeroed, if they do not exist yet.
    pub(super) fn ensure_adam_state(&mut self) {
        if self.adam_state.is_some() || self.plan.param_grad_pairs.is_empty() {
            return;
        }
        for &(param, _) in &self.plan.param_grad_pairs {
            Self::optimizer_len(&self.plan, param);
        }
        let bytes = ChunkBuffers::bytes(self)
            .checked_mul(2)
            .expect("Adam state size overflow");
        super::ensure_device_memory_budget(&self.gpu, bytes, "Adam state");
        self.wait();
        let mut device_buffers = Vec::new();
        let m = ChunkBuffers::new(self, "adam_m", &mut device_buffers);
        let v = ChunkBuffers::new(self, "adam_v", &mut device_buffers);
        self.adam_state = Some((m, v));
        self.zero_optimizer_buffers(&device_buffers, "zero_adam");
    }

    /// Fill freshly created device-local optimizer buffers with zeros.
    pub(super) fn zero_optimizer_buffers(&mut self, buffers: &[(Buffer, u64)], label: &str) {
        if buffers.is_empty() {
            return;
        }
        let mut encoder = self
            .gpu
            .create_command_encoder(blade_graphics::CommandEncoderDesc {
                name: "zero_optimizer",
                buffer_count: 1,
                manual_barriers: false,
            });
        encoder.start();
        {
            let mut transfer = encoder.transfer(label);
            for &(buffer, size) in buffers {
                transfer.fill_buffer(buffer.at(0), size, 0);
            }
        }
        let sync = self.gpu.submit(&mut encoder);
        let _ = wait_for_timed_encoder(&self.gpu, &sync, &mut encoder, self.gpu_timing);
        self.gpu.destroy_command_encoder(&mut encoder);
    }

    /// Each pair's Adam moment pieces, if the moments exist.
    pub(super) fn adam_moments(&self, index: usize) -> Option<(BufferPiece, BufferPiece)> {
        self.adam_state
            .as_ref()
            .map(|state| (state.0.pieces[index], state.1.pieces[index]))
    }
}
