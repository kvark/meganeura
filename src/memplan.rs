//! Lifetime-based buffer aliasing for the execution plan.
//!
//! The dispatch sequence is static and partitioned into barrier groups
//! (one compute pass each, with a full barrier between groups), so
//! buffer reuse is classic linear-scan allocation: two logical buffers
//! may share one physical GPU allocation when their live intervals are
//! disjoint at *barrier-group* granularity. Dispatches inside a group
//! run concurrently, so disjointness must be strict — the old tenant's
//! last use group must come strictly before the new tenant's first
//! write group, guaranteeing a barrier separates them.
//!
//! Coop-padding safety: only matmul output and addend buffers are
//! padded for full-tile cooperative stores, and those stores write the
//! entire padded region on every step, so a reused allocation's stale
//! bytes are overwritten before any read. A/B-tile staging loads are
//! bounds-guarded and never read past the logical extent. Addend
//! padding garbage can only land in the discarded output padding.
//! Re-check this reasoning if a kernel ever gains unguarded reads.
//!
//! Memory placement is a second bit, not the same as pinning. Pinned
//! user-facing slots (params, inputs, constants, outputs, loss) stay
//! `Memory::Shared`. Parameter-gradient buffers are pinned so the
//! optimizer can find them after the plan, but they are only touched
//! by GPU clip/Adam/SGD, so they go in `Memory::Device` with the
//! step-local intermediates. On discrete boards that keeps grad and
//! Adam traffic out of the ReBAR heap.

use crate::compile::{BufferRef, ExecutionPlan, ShaderEntry};
use std::ops::Range;

/// Mapping from the plan's logical buffers onto physical allocations.
pub struct AliasPlan {
    /// Physical allocation index for each logical buffer.
    pub map: Vec<usize>,
    /// Size in bytes of each physical allocation (max over its tenants).
    pub sizes: Vec<usize>,
    /// Per physical allocation: every tenant is a step-local
    /// intermediate that no host path ever touches, so the allocation
    /// may live in `Memory::Device` (device-local, not host-visible).
    /// Pinned buffers — anything uploaded, read back, or externally
    /// bindable — must stay host-visible.
    pub device_local: Vec<bool>,
}

impl AliasPlan {
    /// One physical allocation per logical buffer, all host-visible
    /// (both aliasing and device-local placement disabled).
    pub fn identity(buffer_sizes: &[usize]) -> Self {
        Self {
            map: (0..buffer_sizes.len()).collect(),
            sizes: buffer_sizes.to_vec(),
            device_local: vec![false; buffer_sizes.len()],
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
/// - `ScatterAdd` outputs — read-modify-write;
/// - any buffer whose first use is a read (live-in from a prior step),
///   and any buffer no dispatch touches.
pub fn plan_buffer_aliasing(
    plan: &ExecutionPlan,
    groups: &[Range<usize>],
    pin_spec: Option<&str>,
) -> AliasPlan {
    let (pinned, uses) = compute_pinned(plan, groups, pin_spec);
    let n = plan.buffers.len();

    let mut map = vec![usize::MAX; n];
    let mut sizes = Vec::new();
    let mut device_local = Vec::new();
    let device_ok = device_local_eligible(plan);
    for i in 0..n {
        if pinned[i] {
            map[i] = sizes.len();
            sizes.push(plan.buffers[i]);
            device_local.push(device_ok[i]);
        }
    }

    // Greedy best-fit over live intervals [first_write, last_use].
    let mut order: Vec<usize> = (0..n).filter(|&i| !pinned[i]).collect();
    order.sort_by_key(|&i| {
        (
            uses[i].first_write.unwrap(),
            std::cmp::Reverse(plan.buffers[i]),
        )
    });
    // Allocations open for reuse: (last group used, physical index).
    let mut pool: Vec<(usize, usize)> = Vec::new();
    for i in order {
        let start = uses[i].first_write.unwrap();
        let end = uses[i].last_use.unwrap();
        let need = plan.buffers[i];
        // Among allocations free strictly before `start`, prefer the
        // smallest that already fits; otherwise the largest (grows least).
        let mut best: Option<usize> = None; // index into pool
        for (pi, &(free_after, phys)) in pool.iter().enumerate() {
            if free_after >= start {
                continue;
            }
            best = Some(match best {
                None => pi,
                Some(bi) => {
                    let (bs, cs) = (sizes[pool[bi].1], sizes[phys]);
                    let better = if bs >= need && cs >= need {
                        cs < bs
                    } else {
                        cs > bs
                    };
                    if better { pi } else { bi }
                }
            });
        }
        match best {
            Some(pi) => {
                let phys = pool[pi].1;
                sizes[phys] = sizes[phys].max(need);
                pool[pi].0 = end;
                map[i] = phys;
            }
            None => {
                map[i] = sizes.len();
                sizes.push(need);
                device_local.push(true);
                pool.push((end, map[i]));
            }
        }
    }

    AliasPlan {
        map,
        sizes,
        device_local,
    }
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
        sizes: plan.buffers.clone(),
        device_local: pinned
            .iter()
            .enumerate()
            .map(|(i, &p)| !p || device_ok[i])
            .collect(),
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
        for b in &d.input_buffers {
            uses[b.0 as usize].read(g);
        }
        for b in &d.epilogue_buffers {
            uses[b.0 as usize].read(g);
        }
        if let Some(epi) = d.matmul_epilogue.as_ref() {
            for &(b, _) in &epi.inputs {
                uses[b.0 as usize].read(g);
            }
        }
        if let Some(pro) = d.matmul_prologue.as_ref() {
            for &(b, _) in &pro.factors {
                uses[b.0 as usize].read(g);
            }
        }
        uses[d.output_buffer.0 as usize].write(g);
        for b in &d.extra_outputs {
            uses[b.0 as usize].write(g);
        }
        match d.shader {
            // KV caches persist across steps.
            ShaderEntry::CacheWrite => pinned[d.output_buffer.0 as usize] = true,
            // Read-modify-write accumulation into the output.
            ShaderEntry::ScatterAdd | ShaderEntry::ScatterAddAtomic => {
                uses[d.output_buffer.0 as usize].read(g);
            }
            _ => {}
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
            // inference); untouched buffers stay dedicated.
            (None, Some(_)) => false,
            (None, None) => true,
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
            param_buffers: Vec::new(),
            input_buffers: Vec::new(),
            constant_buffers: Vec::new(),
            dispatches,
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

    /// Every pair of logical buffers sharing a physical allocation must
    /// have strictly disjoint live intervals at group granularity.
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
            for b in &d.input_buffers {
                touch(b.0);
            }
            touch(d.output_buffer.0);
        }
        for a in 0..plan.buffers.len() {
            for b in (a + 1)..plan.buffers.len() {
                if alias.map[a] != alias.map[b] {
                    continue;
                }
                let (sa, ea) = intervals[a];
                let (sb, eb) = intervals[b];
                assert!(
                    ea < sb || eb < sa,
                    "buffers {a} [{sa},{ea}] and {b} [{sb},{eb}] share an allocation"
                );
            }
        }
    }

    #[test]
    fn chain_reuses_disjoint_intermediates() {
        // x -> t1 -> t2 -> t3 -> out, one dispatch per barrier group.
        // t1 dies at group 1, t3 is born at group 2: they may share.
        // t2 overlaps both ends: it may not.
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

        assert_eq!(alias.map[1], alias.map[3], "t1 and t3 should alias");
        assert_ne!(alias.map[2], alias.map[1], "t2 overlaps t1");
        assert_ne!(alias.map[0], alias.map[1], "inputs are pinned");
        assert_ne!(alias.map[4], alias.map[3], "outputs are pinned");
        // Shared allocation sized for the larger tenant.
        assert_eq!(alias.sizes[alias.map[1]], 200);
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
    fn same_group_buffers_never_share() {
        // Two independent branches in one barrier group run concurrently;
        // their outputs must not share even though dispatch indices differ.
        let mut p = plan(
            vec![16, 64, 64, 16],
            vec![dispatch(&[0], 1), dispatch(&[0], 2), dispatch(&[1, 2], 3)],
        );
        p.input_buffers.push(("x".into(), BufferRef(0)));
        p.output_buffers.push(BufferRef(3));
        let groups = vec![0..2, 2..3];
        let alias = plan_buffer_aliasing(&p, &groups, None);
        assert_ne!(alias.map[1], alias.map[2]);
        check_disjoint(&p, &groups, &alias);
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
    fn identity_when_disabled() {
        let sizes = vec![16, 32, 64];
        let alias = AliasPlan::identity(&sizes);
        assert_eq!(alias.map, vec![0, 1, 2]);
        assert_eq!(alias.sizes, sizes);
    }
}
