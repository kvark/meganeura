//! Lifetime-based buffer aliasing and 2D slab sub-allocation for the execution plan.
//!
//! The dispatch sequence is static and partitioned into barrier groups
//! (one compute pass each, with a full barrier between groups). Step-local
//! intermediates are sub-allocated inside shared device-local slabs at
//! 256-byte-aligned byte ranges (`AliasPlan::offsets`): concurrent tenants
//! may share one physical slab at disjoint byte ranges, while two logical
//! buffers may occupy overlapping byte ranges (see [`AliasPlan::overlap`])
//! only when their live intervals are strictly disjoint at *barrier-group*
//! granularity. Dispatches inside a group run concurrently, so disjointness
//! must be strict — the old tenant's last use group must come strictly before
//! the new tenant's first write group, guaranteeing a barrier separates them.
//!
//! Coop-padding safety: only matmul output and addend buffers are
//! padded for full-tile cooperative stores, and those stores write the
//! entire padded region on every step, so a reused allocation's stale
//! bytes are overwritten before any read. A/B-tile staging loads are
//! bounds-guarded and never read past the logical extent. Addend
//! padding garbage can only land in the discarded output padding.
//! Re-check this reasoning if a kernel ever gains unguarded reads.
//!
//! Trainable parameters and their gradients are pinned too, but packed into
//! arenas (see [`ArenaChunk`]) so the optimizer updates many parameters per
//! dispatch. A logical buffer is therefore a byte range of its physical
//! allocation, at [`AliasPlan::offsets`].
//!
//! Memory placement is a second bit, not the same as pinning. Pinned
//! user-facing slots (params, inputs, constants, outputs, loss) stay
//! `Memory::Shared`. Parameter-gradient buffers are pinned so the
//! optimizer can find them after the plan, but they are only touched
//! by GPU clip/Adam/SGD, so they go in `Memory::Device` with the
//! step-local intermediates. On discrete boards that keeps grad and
//! Adam traffic out of the ReBAR heap.

use crate::compile::{BufferRef, ExecutionPlan, ShaderEntry};
use std::{collections::HashMap, ops::Range};

/// Storage buffers may be bound at offsets that are multiples of this
/// (the largest `minStorageBufferOffsetAlignment` Vulkan permits).
pub const ARENA_ALIGNMENT: usize = 256;

/// Largest arena chunk: the smallest `maxStorageBufferRange` Vulkan
/// guarantees, since a buffer bound at an offset sees the rest of it.
pub const ARENA_CHUNK_BYTES: usize = 128 << 20;

/// Trainable parameters, and separately their gradients, packed back to back
/// at the same offsets, so one optimizer dispatch covers the whole chunk.
#[derive(Clone, Debug)]
pub struct ArenaChunk {
    /// Physical allocation holding the parameters.
    pub params: usize,
    /// Physical allocation holding the gradients.
    pub grads: usize,
    /// Bytes spanned by the chunk's slots.
    pub bytes: usize,
    /// `(index into param_grad_pairs, byte offset)` of each slot.
    pub slots: Vec<(usize, usize)>,
}

/// Mapping from the plan's logical buffers onto physical allocations.
pub struct AliasPlan {
    /// Physical allocation index for each logical buffer.
    pub map: Vec<usize>,
    /// Byte offset of each logical buffer within its physical allocation.
    pub offsets: Vec<usize>,
    /// Size in bytes of each physical allocation (furthest tenant end: max of
    /// `offset + size` across its tenants).
    pub sizes: Vec<usize>,
    /// Per physical allocation: every tenant is a step-local
    /// intermediate that no host path ever touches, so the allocation
    /// may live in `Memory::Device` (device-local, not host-visible).
    /// Pinned buffers — anything uploaded, read back, or externally
    /// bindable — must stay host-visible.
    pub device_local: Vec<bool>,
    /// Arenas of trainable parameters and gradients.
    pub arena: Vec<ArenaChunk>,
}

impl AliasPlan {
    /// One physical allocation per logical buffer, all host-visible
    /// (both aliasing and device-local placement disabled).
    pub fn identity(buffer_sizes: &[usize]) -> Self {
        Self {
            map: (0..buffer_sizes.len()).collect(),
            offsets: vec![0; buffer_sizes.len()],
            sizes: buffer_sizes.to_vec(),
            device_local: vec![false; buffer_sizes.len()],
            arena: Vec::new(),
        }
    }

    /// Whether logical buffers `a` and `b` occupy overlapping bytes of one
    /// allocation, given the plan's logical sizes.
    pub fn overlap(&self, sizes: &[usize], a: usize, b: usize) -> bool {
        self.map[a] == self.map[b]
            && self.offsets[a] < self.offsets[b] + sizes[b].max(1)
            && self.offsets[b] < self.offsets[a] + sizes[a].max(1)
    }

    /// Point logical buffer `index` (of `size` bytes) at another allocation,
    /// at byte `offset`, and return the physical index to install it under.
    /// That is its old allocation when it was the only tenant; otherwise
    /// (an arena, for instance) a new entry appended here, leaving the old
    /// allocation to its other tenants.
    pub fn rebind(
        &mut self,
        index: usize,
        size: usize,
        offset: usize,
        device_local: bool,
    ) -> usize {
        let old = self.map[index];
        let arena = self
            .arena
            .iter()
            .any(|chunk| chunk.params == old || chunk.grads == old);
        let shared = self
            .map
            .iter()
            .enumerate()
            .any(|(other, &physical)| physical == old && other != index);
        let target = if arena || shared {
            self.sizes.push(size);
            self.device_local.push(device_local);
            self.sizes.len() - 1
        } else {
            self.device_local[old] = device_local;
            old
        };
        self.map[index] = target;
        self.offsets[index] = offset;
        target
    }

    /// Move the eligible trainable parameters and their gradients out of
    /// their dedicated allocations into arena chunks of at most
    /// `chunk_bytes` (a larger pair gets a chunk of its own).
    ///
    /// A pair is eligible when its parameter is stored as `f32` (the only
    /// storage the optimizer updates) and neither buffer has another role
    /// that needs a whole allocation of its own: an input, a constant, or a
    /// buffer of another pair.
    pub fn pack_arena(&mut self, plan: &ExecutionPlan, chunk_bytes: usize) {
        let mut roles = vec![0u32; plan.buffers.len()];
        for &(p, g) in &plan.param_grad_pairs {
            roles[p.0 as usize] += 1;
            roles[g.0 as usize] += 1;
        }
        for &(_, b) in &plan.input_buffers {
            roles[b.0 as usize] += 2;
        }
        for &(b, _) in &plan.constant_buffers {
            roles[b.0 as usize] += 2;
        }
        for &(b, _) in &plan.uniform_constants {
            roles[b.0 as usize] += 2;
        }
        let eligible = |&(p, g): &(BufferRef, BufferRef)| {
            p != g
                && roles[p.0 as usize] == 1
                && roles[g.0 as usize] == 1
                && plan
                    .param_types
                    .get(&p)
                    .is_none_or(|ty| ty.dtype == crate::graph::DType::F32)
        };
        let mut chunks: Vec<(usize, Vec<(usize, usize)>)> = Vec::new();
        for (index, pair) in plan.param_grad_pairs.iter().enumerate() {
            if !eligible(pair) {
                continue;
            }
            let bytes = plan.buffers[pair.0.0 as usize]
                .max(plan.buffers[pair.1.0 as usize])
                .max(4)
                .next_multiple_of(ARENA_ALIGNMENT);
            match chunks.last_mut() {
                Some(&mut (ref mut used, ref mut slots)) if *used + bytes <= chunk_bytes => {
                    slots.push((index, *used));
                    *used += bytes;
                }
                _ => chunks.push((bytes, vec![(index, 0)])),
            }
        }
        if chunks.is_empty() {
            return;
        }
        let mut retired = vec![false; self.sizes.len()];
        for (bytes, slots) in chunks {
            let params = self.sizes.len();
            let grads = params + 1;
            self.sizes.extend([bytes, bytes]);
            let mut grads_device = true;
            for &(index, offset) in &slots {
                let (p, g) = plan.param_grad_pairs[index];
                let (p, g) = (p.0 as usize, g.0 as usize);
                grads_device &= self.device_local[self.map[g]];
                retired[self.map[p]] = true;
                retired[self.map[g]] = true;
                self.map[p] = params;
                self.map[g] = grads;
                self.offsets[p] = offset;
                self.offsets[g] = offset;
            }
            self.device_local.extend([false, grads_device]);
            self.arena.push(ArenaChunk {
                params,
                grads,
                bytes,
                slots,
            });
        }
        // Drop the dedicated allocations the tenants left.
        retired.resize(self.sizes.len(), false);
        let mut renumber = vec![usize::MAX; self.sizes.len()];
        let mut next = 0;
        for (physical, &gone) in retired.iter().enumerate() {
            if !gone {
                renumber[physical] = next;
                next += 1;
            }
        }
        fn keep<T>(values: &mut Vec<T>, retired: &[bool]) {
            let mut physical = 0;
            values.retain(|_| {
                physical += 1;
                !retired[physical - 1]
            });
        }
        keep(&mut self.sizes, &retired);
        keep(&mut self.device_local, &retired);
        for physical in &mut self.map {
            *physical = renumber[*physical];
        }
        for chunk in &mut self.arena {
            chunk.params = renumber[chunk.params];
            chunk.grads = renumber[chunk.grads];
        }
    }

    pub fn logical_bytes(&self, buffer_sizes: &[usize]) -> usize {
        buffer_sizes.iter().sum()
    }

    pub fn physical_bytes(&self) -> usize {
        self.sizes.iter().sum()
    }

    /// Bytes placed in device-local (non-host-visible) memory.
    pub fn device_local_bytes(&self) -> usize {
        self.sizes
            .iter()
            .zip(&self.device_local)
            .filter_map(|(&s, &d)| d.then_some(s))
            .sum()
    }
}

#[derive(Clone, Copy, Default)]
struct BufferUse {
    first_write: Option<usize>,
    first_read: Option<usize>,
    last_use: Option<usize>,
}

impl BufferUse {
    fn read(&mut self, group: usize) {
        self.first_read = Some(self.first_read.map_or(group, |g| g.min(group)));
        self.last_use = Some(self.last_use.map_or(group, |g| g.max(group)));
    }

    fn write(&mut self, group: usize) {
        self.first_write = Some(self.first_write.map_or(group, |g| g.min(group)));
        self.last_use = Some(self.last_use.map_or(group, |g| g.max(group)));
    }
}

/// Compute a buffer aliasing assignment for `plan`, given the barrier
/// groups (ranges of dispatch indices) in final execution order.
///
/// Pinned buffers (never aliased, each keeps a dedicated allocation):
/// - parameters, inputs, constants, outputs, loss — user-visible and/or
///   persistent across steps, and externally bindable;
/// - gradients — read by the runtime-encoded optimizer and grad-clip
///   passes that run *after* the plan's dispatches (pinned, but
///   device-local: those passes are GPU-only);
/// - derived params and quantized weight buffers — uploaded once;
/// - `CacheWrite` outputs (KV caches) — carry state across steps;
/// - parameter-invariant outputs of plans without gradients — kept until a
///   parameter changes;
/// - `ScatterAdd` outputs — read-modify-write;
/// - any buffer whose first use is a read (live-in from a prior step).
///
/// Compiler-eliminated buffers that no dispatch or session-level role touches
/// all share one placeholder allocation sized for the largest of them.
pub fn plan_buffer_aliasing(
    plan: &ExecutionPlan,
    groups: &[Range<usize>],
    pin_spec: Option<&str>,
) -> AliasPlan {
    let (pinned, uses) = compute_pinned(plan, groups, pin_spec);
    let constant_aliases = uniform_constant_aliases(plan);
    let dead: Vec<bool> = uses
        .iter()
        .enumerate()
        .map(|(i, usage)| {
            !pinned[i]
                && usage.first_write.is_none()
                && usage.first_read.is_none()
                && usage.last_use.is_none()
        })
        .collect();
    let n = plan.buffers.len();

    let mut map = vec![usize::MAX; n];
    let mut sizes = Vec::new();
    let mut device_local = Vec::new();
    let device_ok = device_local_eligible(plan);
    for i in 0..n {
        if pinned[i] && constant_aliases[i].is_none() {
            map[i] = sizes.len();
            sizes.push(plan.buffers[i]);
            device_local.push(device_ok[i]);
        }
    }
    for (logical, &target) in constant_aliases.iter().enumerate() {
        if let Some(target) = target {
            debug_assert!(pinned[target]);
            debug_assert_ne!(map[target], usize::MAX);
            map[logical] = map[target];
        }
    }
    let dead_count = dead.iter().filter(|&&is_dead| is_dead).count();
    if dead_count != 0 {
        let dead_logical_bytes = dead
            .iter()
            .enumerate()
            .filter_map(|(i, &is_dead)| is_dead.then_some(plan.buffers[i]))
            .sum::<usize>();
        let dead_physical_bytes = dead
            .iter()
            .enumerate()
            .filter_map(|(i, &is_dead)| is_dead.then_some(plan.buffers[i]))
            .max()
            .unwrap();
        let physical = sizes.len();
        sizes.push(dead_physical_bytes);
        device_local.push(false);
        for (logical, &is_dead) in dead.iter().enumerate() {
            if is_dead {
                map[logical] = physical;
            }
        }
        log::info!(
            "eliminated-buffer placeholder: {dead_count} logical buffers ({:.1} MB) share {:.1} MB",
            dead_logical_bytes as f64 / 1e6,
            dead_physical_bytes as f64 / 1e6,
        );
    }

    // 2D interval slab sub-allocation over live intervals [first_write, last_use].
    //
    // Sorting intermediates by size descending places the largest tensors at
    // offset 0 first, then packs concurrent smaller tensors into aligned gaps
    // left behind by dead larger tensors (up to ARENA_CHUNK_BYTES for normal
    // slabs, or up to the slab's existing size inside idle oversized slabs).
    // Every tenant offset is a multiple of ARENA_ALIGNMENT (256 B) and
    // conflicting live intervals occupy strictly disjoint byte ranges.
    let mut offsets = vec![0usize; n];
    let mut order: Vec<usize> = (0..n).filter(|&i| !pinned[i] && !dead[i]).collect();
    order.sort_by_key(|&i| {
        (
            std::cmp::Reverse(plan.buffers[i]),
            uses[i].first_write.unwrap(),
            i,
        )
    });

    const BLOCK_CAP: usize = 256;
    struct SlabTenant {
        start: usize,
        end: usize,
        offset: usize,
        aligned_size: usize,
    }
    struct TenantBlock {
        min_start: usize,
        max_end: usize,
        tenants: Vec<SlabTenant>,
    }
    struct IntermediateSlab {
        phys: usize,
        size: usize,
        min_start: usize,
        max_end: usize,
        blocks: Vec<TenantBlock>,
    }
    impl IntermediateSlab {
        fn new(phys: usize, size: usize, first: SlabTenant) -> Self {
            let (min_start, max_end) = (first.start, first.end);
            Self {
                phys,
                size,
                min_start,
                max_end,
                blocks: vec![TenantBlock {
                    min_start,
                    max_end,
                    tenants: vec![first],
                }],
            }
        }

        fn insert(&mut self, tenant: SlabTenant) {
            self.min_start = self.min_start.min(tenant.start);
            self.max_end = self.max_end.max(tenant.end);
            let bi = self
                .blocks
                .partition_point(|b| b.min_start <= tenant.start)
                .saturating_sub(1);
            let block = &mut self.blocks[bi];
            let pos = block.tenants.partition_point(|t| t.start <= tenant.start);
            block.min_start = block.min_start.min(tenant.start);
            block.max_end = block.max_end.max(tenant.end);
            block.tenants.insert(pos, tenant);
            if block.tenants.len() >= BLOCK_CAP {
                let mid = block.tenants.len() / 2;
                let right_tenants = block.tenants.split_off(mid);
                block.min_start = block.tenants[0].start;
                block.max_end = block.tenants.iter().map(|t| t.end).max().unwrap();
                let right = TenantBlock {
                    min_start: right_tenants[0].start,
                    max_end: right_tenants.iter().map(|t| t.end).max().unwrap(),
                    tenants: right_tenants,
                };
                self.blocks.insert(bi + 1, right);
            }
        }

        fn collect_overlapping(&self, start: usize, end: usize, busy: &mut Vec<(usize, usize)>) {
            busy.clear();
            if end < self.min_start || start > self.max_end {
                return;
            }
            for block in &self.blocks {
                if block.min_start > end {
                    break;
                }
                if block.max_end < start {
                    continue;
                }
                for t in &block.tenants {
                    if t.start > end {
                        break;
                    }
                    if start <= t.end {
                        busy.push((t.offset, t.offset + t.aligned_size));
                    }
                }
            }
        }
    }

    let find_slab_offset = |slab: &IntermediateSlab,
                            start: usize,
                            end: usize,
                            need: usize,
                            aligned_size: usize,
                            busy: &mut Vec<(usize, usize)>|
     -> Option<(usize, usize)> {
        let oversized = need > ARENA_CHUNK_BYTES;
        slab.collect_overlapping(start, end, busy);
        if oversized {
            // An oversized tensor can reuse any idle slab (normal or oversized)
            // at offset 0, growing it if needed.
            return busy.is_empty().then_some((0, slab.size.max(need)));
        }
        // Normal tensors can sub-allocate inside normal slabs up to
        // ARENA_CHUNK_BYTES, or reuse free ranges inside an existing oversized
        // slab without growing it past its current size.
        let ceiling = if slab.size > ARENA_CHUNK_BYTES {
            slab.size
        } else {
            ARENA_CHUNK_BYTES
        };
        if busy.is_empty() {
            return (need <= ceiling).then_some((0, slab.size.max(need)));
        }
        busy.sort_unstable();
        let mut cursor = 0usize;
        let mut best_gap: Option<(usize, usize)> = None; // (gap_bytes, offset)
        for &(lo, hi) in busy.iter() {
            if lo > cursor {
                let gap = lo - cursor;
                if gap >= aligned_size && best_gap.is_none_or(|(best_size, _)| gap < best_size) {
                    best_gap = Some((gap, cursor));
                }
            }
            cursor = cursor.max(hi);
        }
        if let Some((_, offset)) = best_gap {
            return Some((offset, slab.size.max(offset + need)));
        }
        let end_offset = cursor.checked_add(need)?;
        if end_offset <= ceiling {
            Some((cursor, slab.size.max(end_offset)))
        } else {
            None
        }
    };

    let mut slabs: Vec<IntermediateSlab> = Vec::new();
    let mut busy: Vec<(usize, usize)> = Vec::new();
    for i in order {
        let start = uses[i].first_write.unwrap();
        let end = uses[i].last_use.unwrap();
        let need = plan.buffers[i];
        let aligned_size = need.max(16).next_multiple_of(ARENA_ALIGNMENT);

        // Prefer a slab that fits with zero growth (smallest slab first,
        // smallest offset on ties); otherwise pick the slab with minimum
        // growth within its ceiling.
        let mut best: Option<(usize, usize, usize, usize)> = None; // (growth, new_size, offset, slab_idx)
        for (si, slab) in slabs.iter().enumerate() {
            let Some((offset, new_size)) =
                find_slab_offset(slab, start, end, need, aligned_size, &mut busy)
            else {
                continue;
            };
            let growth = new_size.saturating_sub(slab.size);
            let candidate = (growth, new_size, offset, si);
            if best.is_none_or(|cur| candidate < cur) {
                best = Some(candidate);
            }
        }

        match best {
            Some((_, new_size, offset, si)) => {
                let slab = &mut slabs[si];
                slab.size = new_size;
                sizes[slab.phys] = new_size;
                slab.insert(SlabTenant {
                    start,
                    end,
                    offset,
                    aligned_size,
                });
                map[i] = slab.phys;
                offsets[i] = offset;
            }
            None => {
                let phys = sizes.len();
                sizes.push(need);
                device_local.push(true);
                slabs.push(IntermediateSlab::new(
                    phys,
                    need,
                    SlabTenant {
                        start,
                        end,
                        offset: 0,
                        aligned_size,
                    },
                ));
                map[i] = phys;
                offsets[i] = 0;
            }
        }
    }

    AliasPlan {
        offsets,
        map,
        sizes,
        device_local,
        arena: Vec::new(),
    }
}

/// Recurrent autodiff emits identical full-size zero, one, and scale tensors
/// at every unrolled step. Bit-identical immutable uniform constants can share
/// one physical allocation while retaining distinct logical/node identities.
/// Buffers written by a dispatch or a session-level mutation path are excluded.
fn uniform_constant_aliases(plan: &ExecutionPlan) -> Vec<Option<usize>> {
    let n = plan.buffers.len();
    let mut mutable = vec![false; n];
    let mut mark = |buffer: BufferRef| mutable[buffer.0 as usize] = true;
    for &(_, buffer) in &plan.param_buffers {
        mark(buffer);
    }
    for &(_, buffer) in &plan.input_buffers {
        mark(buffer);
    }
    for &(parameter, gradient) in &plan.param_grad_pairs {
        mark(parameter);
        mark(gradient);
    }
    for entry in &plan.derived_params {
        mark(entry.0);
    }
    for &buffer in plan.weight_buffers.keys() {
        mark(buffer);
    }
    for dispatch in &plan.dispatches {
        mark(dispatch.output_buffer);
        for &buffer in &dispatch.extra_outputs {
            mark(buffer);
        }
    }

    let mut canonical = HashMap::<(usize, usize, u32), usize>::new();
    let mut aliases = vec![None; n];
    let mut count = 0usize;
    let mut bytes = 0usize;
    for &(buffer, ref data) in &plan.constant_buffers {
        let index = buffer.0 as usize;
        if mutable[index] {
            continue;
        }
        let Some(bits) = data.first().map(|value| value.to_bits()) else {
            continue;
        };
        if !data.iter().all(|value| value.to_bits() == bits) {
            continue;
        }
        let key = (plan.buffers[index], data.len(), bits);
        if let Some(&target) = canonical.get(&key) {
            aliases[index] = Some(target);
            count += 1;
            bytes += plan.buffers[index];
        } else {
            canonical.insert(key, index);
        }
    }
    for &(buffer, value) in &plan.uniform_constants {
        let index = buffer.0 as usize;
        if mutable[index] {
            continue;
        }
        let elems = plan.buffers[index] / std::mem::size_of::<f32>();
        if elems == 0 {
            continue;
        }
        let key = (plan.buffers[index], elems, value.to_bits());
        if let Some(&target) = canonical.get(&key) {
            aliases[index] = Some(target);
            count += 1;
            bytes += plan.buffers[index];
        } else {
            canonical.insert(key, index);
        }
    }
    if count != 0 {
        log::info!(
            "uniform constant aliasing: {count} buffers share canonical allocations ({:.1} MB eliminated)",
            bytes as f64 / 1e6,
        );
    }
    aliases
}

/// One physical allocation per logical buffer (no aliasing), but with
/// the same host-visibility classification as [`plan_buffer_aliasing`]
/// — step-local intermediates are still marked for `Memory::Device`.
/// Used when `MEGANEURA_NO_ALIAS` disables reuse, so the aliasing and
/// device-local decisions can be toggled independently.
pub fn plan_no_alias(
    plan: &ExecutionPlan,
    groups: &[Range<usize>],
    pin_spec: Option<&str>,
) -> AliasPlan {
    let (pinned, _) = compute_pinned(plan, groups, pin_spec);
    let device_ok = device_local_eligible(plan);
    AliasPlan {
        map: (0..plan.buffers.len()).collect(),
        offsets: vec![0; plan.buffers.len()],
        sizes: plan.buffers.clone(),
        device_local: pinned
            .iter()
            .enumerate()
            .map(|(i, &p)| !p || device_ok[i])
            .collect(),
        arena: Vec::new(),
    }
}

/// Pinned buffers that are still eligible for `Memory::Device`.
///
/// Parameter gradients are pinned so clip/Adam can find them after the
/// plan, but those passes are GPU-only. Host diagnostics go through the
/// staging readback path.
fn device_local_eligible(plan: &ExecutionPlan) -> Vec<bool> {
    let mut ok = vec![false; plan.buffers.len()];
    for &(_, grad) in &plan.param_grad_pairs {
        let i = grad.0 as usize;
        if i < ok.len() {
            ok[i] = true;
        }
    }
    // A gradient seed can also be a compile-time constant. Session setup
    // initializes constants through their host pointer, so that role takes
    // precedence over device-local gradient storage.
    for &(buffer, _) in &plan.constant_buffers {
        ok[buffer.0 as usize] = false;
    }
    for &(buffer, _) in &plan.uniform_constants {
        ok[buffer.0 as usize] = false;
    }
    ok
}

/// The pinning analysis shared by aliasing and memory-class decisions:
/// which buffers must keep a dedicated, host-visible allocation (see
/// [`plan_buffer_aliasing`] for the full list), plus the per-buffer
/// barrier-group use intervals.
fn compute_pinned(
    plan: &ExecutionPlan,
    groups: &[Range<usize>],
    pin_spec: Option<&str>,
) -> (Vec<bool>, Vec<BufferUse>) {
    let n = plan.buffers.len();
    let mut pinned = vec![false; n];
    let pin = |b: BufferRef, pinned: &mut Vec<bool>| {
        pinned[b.0 as usize] = true;
    };
    for &(_, b) in &plan.param_buffers {
        pin(b, &mut pinned);
    }
    for &(_, b) in &plan.input_buffers {
        pin(b, &mut pinned);
    }
    for &(b, _) in &plan.constant_buffers {
        pin(b, &mut pinned);
    }
    for &(b, _) in &plan.uniform_constants {
        pin(b, &mut pinned);
    }
    for &b in &plan.output_buffers {
        pin(b, &mut pinned);
    }
    if let Some(b) = plan.loss_buffer {
        pin(b, &mut pinned);
    }
    for &(p, g) in &plan.param_grad_pairs {
        pin(p, &mut pinned);
        pin(g, &mut pinned);
    }
    for entry in &plan.derived_params {
        pin(entry.0, &mut pinned);
    }
    for &b in plan.weight_buffers.keys() {
        pin(b, &mut pinned);
    }

    // Barrier-group index for each dispatch.
    let mut group_of = vec![0usize; plan.dispatches.len()];
    for (gi, range) in groups.iter().enumerate() {
        for i in range.clone() {
            group_of[i] = gi;
        }
    }

    let mut uses = vec![BufferUse::default(); n];
    for (i, d) in plan.dispatches.iter().enumerate() {
        let g = group_of[i];
        for b in d.read_buffers() {
            uses[b.0 as usize].read(g);
        }
        for b in d.write_buffers() {
            uses[b.0 as usize].write(g);
        }
        match d.shader {
            // KV caches persist across steps.
            ShaderEntry::CacheWrite => pinned[d.output_buffer.0 as usize] = true,
            // Kept from the step that last ran it, where a session can skip
            // it: without gradients, no step changes the parameters.
            _ if d.parameter_invariant && plan.param_grad_pairs.is_empty() => {
                pinned[d.output_buffer.0 as usize] = true;
            }
            _ => {}
        }
    }
    // Keep an attention operation's external bindings live across its whole
    // split/combine sequence. Tuning can then replace it with a single dispatch
    // without moving an output onto storage still occupied by its query.
    let partials: HashMap<_, _> = plan
        .dispatches
        .iter()
        .enumerate()
        .filter(|&(_, d)| d.shader == ShaderEntry::CachedBlockAttentionSplit)
        .map(|(i, d)| (d.output_buffer, i))
        .collect();
    for (i, d) in plan.dispatches.iter().enumerate() {
        if d.shader != ShaderEntry::CachedBlockAttentionCombine {
            continue;
        }
        let Some(&producer) = d.input_buffers.first().and_then(|b| partials.get(b)) else {
            continue;
        };
        let first = group_of[producer];
        let last = group_of[i];
        uses[d.output_buffer.0 as usize].write(first);
        for b in &plan.dispatches[producer].input_buffers {
            uses[b.0 as usize].read(last);
        }
    }
    // Debug aid: MEGANEURA_PIN_BUFS="3,17,25-40" force-pins logical
    // buffers, excluding them from aliasing. Used to bisect aliasing
    // corruption down to a single buffer.
    if let Some(spec) = pin_spec {
        for part in spec.split(',').filter(|s| !s.is_empty()) {
            if let Some((a, b)) = part.split_once('-') {
                let (a, b): (usize, usize) = (a.parse().unwrap(), b.parse().unwrap());
                let end = b.min(n - 1);
                if a <= end {
                    pinned[a..=end].fill(true);
                }
            } else {
                let i: usize = part.parse().unwrap();
                if i < n {
                    pinned[i] = true;
                }
            }
        }
    }

    for (i, u) in uses.iter().enumerate() {
        let live_in = match (u.first_read, u.first_write) {
            // Read at or before the first write: contents carry over
            // from a previous step (or from a same-group dispatch the
            // hazard analysis allowed) — don't touch.
            (Some(r), Some(w)) => r <= w,
            (Some(_), None) => true,
            // Written but never read is fine (e.g. unused LSE in
            // inference). Untouched, non-persistent buffers were eliminated
            // by compilation and share a placeholder allocation.
            (None, Some(_)) => false,
            (None, None) => false,
        };
        if live_in {
            pinned[i] = true;
        }
    }
    (pinned, uses)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::Dispatch;

    fn dispatch(inputs: &[u32], output: u32) -> Dispatch {
        Dispatch {
            input_buffers: inputs.iter().map(|&b| BufferRef(b)).collect(),
            output_buffer: BufferRef(output),
            ..Default::default()
        }
    }

    fn plan(buffers: Vec<usize>, dispatches: Vec<Dispatch>) -> ExecutionPlan {
        ExecutionPlan {
            buffers,
            logical_buffers: Vec::new(),
            param_buffers: Vec::new(),
            param_types: Default::default(),
            input_buffers: Vec::new(),
            input_types: Default::default(),
            constant_buffers: Vec::new(),
            uniform_constants: Vec::new(),
            dispatches,
            groups: Vec::new(),
            loss_buffer: None,
            output_buffers: Vec::new(),
            param_grad_pairs: Vec::new(),
            lse_buffers: Vec::new(),
            derived_params: Vec::new(),
            weight_buffers: Default::default(),
            node_buffers: Vec::new(),
            node_names: Vec::new(),
            knobs: Default::default(),
        }
    }

    /// Every pair of logical buffers with overlapping physical bytes must
    /// have strictly disjoint live intervals at group granularity, and every
    /// sub-allocation offset must respect ARENA_ALIGNMENT and slab bounds.
    fn check_disjoint(plan: &ExecutionPlan, groups: &[Range<usize>], alias: &AliasPlan) {
        let mut group_of = vec![0usize; plan.dispatches.len()];
        for (gi, r) in groups.iter().enumerate() {
            for i in r.clone() {
                group_of[i] = gi;
            }
        }
        let mut intervals = vec![(usize::MAX, 0usize); plan.buffers.len()];
        for (i, d) in plan.dispatches.iter().enumerate() {
            let g = group_of[i];
            let mut touch = |b: u32| {
                let iv = &mut intervals[b as usize];
                iv.0 = iv.0.min(g);
                iv.1 = iv.1.max(g);
            };
            for b in d.read_buffers() {
                touch(b.0);
            }
            for b in d.write_buffers() {
                touch(b.0);
            }
        }
        for i in 0..plan.buffers.len() {
            assert_eq!(
                alias.offsets[i] % ARENA_ALIGNMENT,
                0,
                "buffer {i} offset {} not aligned to {ARENA_ALIGNMENT}",
                alias.offsets[i]
            );
            assert!(
                alias.offsets[i] + plan.buffers[i] <= alias.sizes[alias.map[i]],
                "buffer {i} [{}..{}] exceeds physical allocation {}",
                alias.offsets[i],
                alias.offsets[i] + plan.buffers[i],
                alias.sizes[alias.map[i]]
            );
        }
        for a in 0..plan.buffers.len() {
            for b in (a + 1)..plan.buffers.len() {
                if !alias.overlap(&plan.buffers, a, b) {
                    continue;
                }
                let (sa, ea) = intervals[a];
                let (sb, eb) = intervals[b];
                assert!(
                    ea < sb || eb < sa,
                    "buffers {a} [{sa},{ea}] and {b} [{sb},{eb}] overlap in physical memory"
                );
            }
        }
    }

    #[test]
    fn chain_reuses_disjoint_intermediates() {
        // x -> t1 -> t2 -> t3 -> out, one dispatch per barrier group.
        // t1 dies at group 1, t3 is born at group 2: they may overlap in memory.
        // t2 overlaps both ends: its byte range must stay disjoint from both.
        let mut p = plan(
            vec![16, 100, 16, 200, 16],
            vec![
                dispatch(&[0], 1),
                dispatch(&[1], 2),
                dispatch(&[2], 3),
                dispatch(&[3], 4),
            ],
        );
        p.input_buffers.push(("x".into(), BufferRef(0)));
        p.output_buffers.push(BufferRef(4));
        let groups: Vec<Range<usize>> = (0..4).map(|i| i..i + 1).collect();
        let alias = plan_buffer_aliasing(&p, &groups, None);

        assert!(
            alias.overlap(&p.buffers, 1, 3),
            "t1 and t3 should reuse memory"
        );
        assert!(!alias.overlap(&p.buffers, 2, 1), "t2 overlaps t1 in time");
        assert!(!alias.overlap(&p.buffers, 2, 3), "t2 overlaps t3 in time");
        assert_ne!(alias.map[0], alias.map[1], "inputs are pinned");
        assert_ne!(alias.map[4], alias.map[3], "outputs are pinned");
        // All three intermediates pack into one device-local slab.
        assert_eq!(alias.map[1], alias.map[2]);
        assert_eq!(alias.map[1], alias.map[3]);
        assert!(alias.physical_bytes() < alias.logical_bytes(&p.buffers));
        check_disjoint(&p, &groups, &alias);
        // Memory classes: intermediates device-local, pinned host-visible.
        assert!(alias.device_local[alias.map[1]]);
        assert!(alias.device_local[alias.map[2]]);
        assert!(
            !alias.device_local[alias.map[0]],
            "input stays host-visible"
        );
        assert!(
            !alias.device_local[alias.map[4]],
            "output stays host-visible"
        );
        assert_eq!(alias.device_local.len(), alias.sizes.len());

        // A split attention result may not reuse its query's storage:
        // the tuner must be free to evaluate the same operation unsplit.
        p.dispatches[1].shader = ShaderEntry::CachedBlockAttentionSplit;
        p.dispatches[2].shader = ShaderEntry::CachedBlockAttentionCombine;
        let alias = plan_buffer_aliasing(&p, &groups, None);
        assert!(!alias.overlap(&p.buffers, 1, 3));
        check_disjoint(&p, &groups, &alias);
    }

    #[test]
    fn no_alias_keeps_memory_classes() {
        // MEGANEURA_NO_ALIAS path: identity mapping, but intermediates
        // are still classified for device-local placement.
        let mut p = plan(
            vec![16, 100, 16, 200, 16],
            vec![
                dispatch(&[0], 1),
                dispatch(&[1], 2),
                dispatch(&[2], 3),
                dispatch(&[3], 4),
            ],
        );
        p.input_buffers.push(("x".into(), BufferRef(0)));
        p.output_buffers.push(BufferRef(4));
        let groups: Vec<Range<usize>> = (0..4).map(|i| i..i + 1).collect();
        let alias = plan_no_alias(&p, &groups, None);
        assert_eq!(alias.map, vec![0, 1, 2, 3, 4]);
        assert_eq!(alias.sizes, p.buffers);
        assert_eq!(alias.device_local, vec![false, true, true, true, false]);
        assert_eq!(alias.device_local_bytes(), 316);
    }

    #[test]
    fn eliminated_buffers_share_one_placeholder() {
        let mut p = plan(vec![16, 64, 128, 16], vec![dispatch(&[0], 3)]);
        p.input_buffers.push(("x".into(), BufferRef(0)));
        p.output_buffers.push(BufferRef(3));
        let groups = std::iter::once(0..1).collect::<Vec<_>>();

        let alias = plan_buffer_aliasing(&p, &groups, None);

        assert_eq!(alias.map[1], alias.map[2]);
        assert_eq!(alias.sizes[alias.map[1]], 128);
        assert!(!alias.device_local[alias.map[1]]);
        assert_ne!(alias.map[0], alias.map[1]);
        assert_ne!(alias.map[3], alias.map[1]);
    }

    #[test]
    fn unused_parameter_does_not_share_the_placeholder() {
        let mut p = plan(vec![16, 64, 128, 16], vec![dispatch(&[0], 3)]);
        p.input_buffers.push(("x".into(), BufferRef(0)));
        p.param_buffers.push(("unused".into(), BufferRef(1)));
        p.output_buffers.push(BufferRef(3));
        let groups = std::iter::once(0..1).collect::<Vec<_>>();

        let alias = plan_buffer_aliasing(&p, &groups, None);

        assert_ne!(alias.map[1], alias.map[2]);
        assert_eq!(alias.sizes[alias.map[1]], 64);
        assert_eq!(alias.sizes[alias.map[2]], 128);
    }

    #[test]
    fn identical_uniform_constants_share_one_allocation() {
        let mut p = plan(vec![16, 16], Vec::new());
        p.constant_buffers.push((BufferRef(0), vec![1.0; 4]));
        p.constant_buffers.push((BufferRef(1), vec![1.0; 4]));

        let alias = plan_buffer_aliasing(&p, &[], None);

        assert_eq!(alias.map[0], alias.map[1]);
        assert_eq!(alias.sizes, [16]);
    }

    #[test]
    fn different_uniform_constants_remain_distinct() {
        let mut p = plan(vec![16, 16], Vec::new());
        p.constant_buffers.push((BufferRef(0), vec![1.0; 4]));
        p.constant_buffers.push((BufferRef(1), vec![2.0; 4]));

        let alias = plan_buffer_aliasing(&p, &[], None);

        assert_ne!(alias.map[0], alias.map[1]);
    }

    #[test]
    fn mutable_constant_role_is_not_aliased() {
        let mut p = plan(vec![16, 16, 16], Vec::new());
        p.constant_buffers.push((BufferRef(0), vec![1.0; 4]));
        p.constant_buffers.push((BufferRef(1), vec![1.0; 4]));
        p.param_buffers.push(("weight".into(), BufferRef(2)));
        p.param_grad_pairs.push((BufferRef(2), BufferRef(1)));

        let alias = plan_buffer_aliasing(&p, &[], None);

        assert_ne!(alias.map[0], alias.map[1]);
    }

    #[test]
    fn same_group_buffers_never_overlap() {
        // Two independent branches in one barrier group run concurrently;
        // their byte ranges must not overlap even though they may share a slab.
        let mut p = plan(
            vec![16, 64, 64, 16],
            vec![dispatch(&[0], 1), dispatch(&[0], 2), dispatch(&[1, 2], 3)],
        );
        p.input_buffers.push(("x".into(), BufferRef(0)));
        p.output_buffers.push(BufferRef(3));
        let groups = vec![0..2, 2..3];
        let alias = plan_buffer_aliasing(&p, &groups, None);
        assert!(!alias.overlap(&p.buffers, 1, 2));
        check_disjoint(&p, &groups, &alias);
    }

    #[test]
    fn slab_suballocation_packs_concurrent_buffers_into_dead_region() {
        // Phase 0: one 1 MiB intermediate `a` (buf 1) written & consumed by phase 1.
        // Phase 2: two concurrent 512 KiB intermediates `b` (buf 2) and `c` (buf 3)
        //          that are disjoint from `a` (`a` dies at group 1, `b`/`c` born at group 2).
        // Whole-buffer reuse would need 1.5 MiB across 2 allocations; 2D slab
        // sub-allocation places both `b` and `c` side-by-side inside `a`'s 1 MiB slab.
        let mib = 1 << 20;
        let half = 512 << 10;
        let mut p = plan(
            vec![16, mib, half, half, 16, 16],
            vec![
                dispatch(&[0], 1),    // group 0: write a (1 MiB)
                dispatch(&[1], 4),    // group 1: consume a -> pinned out 4
                dispatch(&[0], 2),    // group 2: write b (512 KiB)
                dispatch(&[0], 3),    // group 2: write c (512 KiB)
                dispatch(&[2, 3], 5), // group 3: consume b, c -> pinned out 5
            ],
        );
        p.input_buffers.push(("x".into(), BufferRef(0)));
        p.output_buffers.extend([BufferRef(4), BufferRef(5)]);
        let groups = vec![0..1, 1..2, 2..4, 4..5];
        let alias = plan_buffer_aliasing(&p, &groups, None);

        check_disjoint(&p, &groups, &alias);
        assert_eq!(alias.map[1], alias.map[2]);
        assert_eq!(alias.map[1], alias.map[3]);
        assert!(!alias.overlap(&p.buffers, 2, 3));
        assert_eq!(
            alias.device_local_bytes(),
            mib,
            "two 512 KiB concurrent buffers should pack into the dead 1 MiB slab"
        );
    }

    #[test]
    fn slab_suballocation_respects_chunk_ceiling_and_oversized_buffers() {
        // Two concurrent 80 MiB buffers cannot share a 128 MiB slab, while two
        // disjoint 160 MiB oversized buffers still reuse one 160 MiB allocation.
        let m80 = 80 << 20;
        let m160 = 160 << 20;
        let mut p = plan(
            vec![16, m80, m80, m160, m160, 16],
            vec![
                dispatch(&[0], 1),    // group 0: write 80 MiB #1
                dispatch(&[0], 2),    // group 0: write 80 MiB #2
                dispatch(&[1, 2], 3), // group 1: consume both, write 160 MiB #1
                dispatch(&[3], 5),    // group 2: consume 160 MiB #1
                dispatch(&[0], 4),    // group 3: write 160 MiB #2 (disjoint from #1)
                dispatch(&[4], 5),    // group 4: consume 160 MiB #2
            ],
        );
        p.input_buffers.push(("x".into(), BufferRef(0)));
        p.output_buffers.push(BufferRef(5));
        let groups = vec![0..2, 2..3, 3..4, 4..5, 5..6];
        let alias = plan_buffer_aliasing(&p, &groups, None);

        check_disjoint(&p, &groups, &alias);
        assert_ne!(
            alias.map[1], alias.map[2],
            "160 MiB exceeds 128 MiB slab ceiling"
        );
        assert!(alias.sizes[alias.map[1]] <= ARENA_CHUNK_BYTES);
        assert!(alias.sizes[alias.map[2]] <= ARENA_CHUNK_BYTES);
        assert_eq!(
            alias.map[3], alias.map[4],
            "disjoint oversized buffers still alias"
        );
        assert_eq!(alias.sizes[alias.map[3]], m160);
    }

    #[test]
    fn dead_oversized_slab_reused_by_normal_intermediates() {
        // A 160 MiB logits intermediate dies at group 1; two concurrent 40 MiB
        // backward intermediates at group 2 reuse the dead oversized slab with
        // zero additional device-local memory.
        let m160 = 160 << 20;
        let m40 = 40 << 20;
        let mut p = plan(
            vec![16, m160, m40, m40, 16, 16],
            vec![
                dispatch(&[0], 1),    // group 0: write 160 MiB
                dispatch(&[1], 4),    // group 1: consume 160 MiB -> out 4
                dispatch(&[0], 2),    // group 2: write 40 MiB #1
                dispatch(&[0], 3),    // group 2: write 40 MiB #2
                dispatch(&[2, 3], 5), // group 3: consume both -> out 5
            ],
        );
        p.input_buffers.push(("x".into(), BufferRef(0)));
        p.output_buffers.extend([BufferRef(4), BufferRef(5)]);
        let groups = vec![0..1, 1..2, 2..4, 4..5];
        let alias = plan_buffer_aliasing(&p, &groups, None);

        check_disjoint(&p, &groups, &alias);
        assert_eq!(alias.map[1], alias.map[2]);
        assert_eq!(alias.map[1], alias.map[3]);
        assert!(!alias.overlap(&p.buffers, 2, 3));
        assert_eq!(
            alias.device_local_bytes(),
            m160,
            "normal intermediates should reuse the idle oversized slab with zero growth"
        );
    }

    #[test]
    fn slab_block_splitting_preserves_disjointness_on_long_chains() {
        // 600 sequential intermediates force multiple TenantBlock splits
        // (BLOCK_CAP = 256) and still reuse a single slab cleanly.
        let steps = 600usize;
        let mut buffers = vec![16usize]; // 0 = input
        let mut dispatches = Vec::with_capacity(steps + 1);
        for s in 0..steps {
            buffers.push(4096 + (s % 7) * 512);
            dispatches.push(dispatch(&[s as u32], (s + 1) as u32));
        }
        buffers.push(16); // final output
        dispatches.push(dispatch(&[steps as u32], (steps + 1) as u32));
        let mut p = plan(buffers, dispatches);
        p.input_buffers.push(("x".into(), BufferRef(0)));
        p.output_buffers.push(BufferRef((steps + 1) as u32));
        let groups: Vec<Range<usize>> = (0..=steps).map(|i| i..i + 1).collect();
        let alias = plan_buffer_aliasing(&p, &groups, None);
        check_disjoint(&p, &groups, &alias);
        // At most two intermediates live simultaneously, so slab stays tiny.
        assert!(alias.device_local_bytes() <= 32 * 1024);
    }

    #[test]
    fn read_before_write_is_pinned() {
        // Buffer 1 is read in group 0 and written in group 1 — its
        // contents carry over from the previous step (cache-like).
        let mut p = plan(
            vec![16, 64, 16, 64],
            vec![dispatch(&[1], 2), dispatch(&[0], 1), dispatch(&[2], 3)],
        );
        p.input_buffers.push(("x".into(), BufferRef(0)));
        p.output_buffers.push(BufferRef(3));
        let groups: Vec<Range<usize>> = (0..3).map(|i| i..i + 1).collect();
        let alias = plan_buffer_aliasing(&p, &groups, None);
        // Buffer 2 dies at group 2 and nothing later could reuse buffer 1's
        // slot anyway; the property under test: 1 keeps a dedicated slot.
        assert!(alias.map.iter().filter(|&&m| m == alias.map[1]).count() == 1);
    }

    #[test]
    fn cache_write_output_is_pinned() {
        let mut d = dispatch(&[0], 1);
        d.shader = ShaderEntry::CacheWrite;
        let mut p = plan(
            vec![16, 256, 64, 16],
            vec![d, dispatch(&[1], 2), dispatch(&[2], 3)],
        );
        p.input_buffers.push(("kv".into(), BufferRef(0)));
        p.output_buffers.push(BufferRef(3));
        let groups: Vec<Range<usize>> = (0..3).map(|i| i..i + 1).collect();
        let alias = plan_buffer_aliasing(&p, &groups, None);
        assert!(alias.map.iter().filter(|&&m| m == alias.map[1]).count() == 1);
    }

    #[test]
    fn params_and_grads_are_pinned() {
        let mut p = plan(
            vec![64, 64, 64, 64, 64],
            vec![dispatch(&[0, 1], 2), dispatch(&[2], 3), dispatch(&[3], 4)],
        );
        p.input_buffers.push(("x".into(), BufferRef(0)));
        p.param_buffers.push(("w".into(), BufferRef(1)));
        p.param_grad_pairs.push((BufferRef(1), BufferRef(4)));
        p.output_buffers.push(BufferRef(3));
        let groups: Vec<Range<usize>> = (0..3).map(|i| i..i + 1).collect();
        let alias = plan_buffer_aliasing(&p, &groups, None);
        for pinned in [0usize, 1, 3, 4] {
            assert!(
                alias
                    .map
                    .iter()
                    .filter(|&&m| m == alias.map[pinned])
                    .count()
                    == 1,
                "buffer {pinned} must keep a dedicated allocation"
            );
        }
        assert!(
            alias.device_local[alias.map[4]],
            "parameter gradients are device-local"
        );
        assert!(
            !alias.device_local[alias.map[1]],
            "parameters stay host-visible"
        );
    }

    #[test]
    fn trainable_pairs_pack_into_aligned_arena_chunks() {
        // Three f32 pairs of 100, 600 and 300 bytes, then a pair whose
        // gradient is a constant: that one keeps dedicated allocations.
        let mut p = plan(vec![100, 100, 600, 600, 300, 300, 16, 16], Vec::new());
        for (i, (param, grad)) in [(0, 1), (2, 3), (4, 5), (6, 7)].into_iter().enumerate() {
            p.param_buffers.push((format!("w{i}"), BufferRef(param)));
            p.param_grad_pairs.push((BufferRef(param), BufferRef(grad)));
        }
        p.constant_buffers.push((BufferRef(7), vec![1.0; 4]));
        let mut alias = plan_buffer_aliasing(&p, &[], None);
        let before = alias.sizes.len();
        alias.pack_arena(&p, 1024);

        // 256 + 768 fits one chunk; the third pair starts another.
        assert_eq!(alias.arena.len(), 2);
        assert_eq!(alias.arena[0].slots, [(0, 0), (1, 256)]);
        assert_eq!(alias.arena[0].bytes, 1024);
        assert_eq!(alias.arena[1].slots, [(2, 0)]);
        for chunk in &alias.arena {
            for &(index, offset) in &chunk.slots {
                let (param, grad) = p.param_grad_pairs[index];
                assert_eq!(alias.map[param.0 as usize], chunk.params);
                assert_eq!(alias.map[grad.0 as usize], chunk.grads);
                assert_eq!(alias.offsets[param.0 as usize], offset);
                assert_eq!(alias.offsets[grad.0 as usize], offset);
                assert_eq!(offset % ARENA_ALIGNMENT, 0);
            }
            assert!(!alias.device_local[chunk.params]);
            assert!(alias.device_local[chunk.grads]);
        }
        // Six dedicated allocations became four arena chunks' worth.
        assert_eq!(alias.sizes.len(), before - 6 + 4);
        assert_eq!(alias.sizes.len(), alias.device_local.len());
        for (logical, &physical) in alias.map.iter().enumerate() {
            assert!(physical < alias.sizes.len(), "buffer {logical}");
        }
        // The pair with a constant gradient is not packed.
        assert_eq!(alias.offsets[6], 0);
        assert!(alias.map.iter().filter(|&&m| m == alias.map[6]).count() == 1);
        assert!(!alias.overlap(&p.buffers, 0, 2));
        assert!(alias.overlap(&p.buffers, 0, 0));
    }

    #[test]
    fn constant_gradient_seed_stays_host_visible() {
        let mut p = plan(vec![16, 16], vec![dispatch(&[0], 1)]);
        p.param_buffers.push(("w".into(), BufferRef(0)));
        p.param_grad_pairs.push((BufferRef(0), BufferRef(1)));
        p.constant_buffers.push((BufferRef(1), vec![1.0]));
        let groups = std::iter::once(0..1).collect::<Vec<_>>();

        let alias = plan_buffer_aliasing(&p, &groups, None);
        assert!(!alias.device_local[alias.map[1]]);

        let no_alias = plan_no_alias(&p, &groups, None);
        assert!(!no_alias.device_local[1]);
    }

    #[test]
    fn identity_when_disabled() {
        let sizes = vec![16, 32, 64];
        let alias = AliasPlan::identity(&sizes);
        assert_eq!(alias.map, vec![0, 1, 2]);
        assert_eq!(alias.sizes, sizes);
    }
}
