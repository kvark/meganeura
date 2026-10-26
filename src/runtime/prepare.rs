//! CPU preparation shared by session construction and prospective plan costs.
//! This module selects kernels and lays out buffers; it never allocates GPU
//! resources, compiles pipelines or executes a graph. Device qualification is
//! supplied by the caller, exactly as it is for a session.

use super::{CoopPolicy, SessionOptions, select_variants, workgroup};
use crate::compile::{AttentionParams, DispatchParams, ExecutionPlan, ShaderEntry};
use std::ops::Range;

/// Preserve measured plans, otherwise select backend variants before grouping.
pub(super) fn schedule(
    mut plan: ExecutionPlan,
    opts: &SessionOptions,
    coop_config: Option<&crate::codegen::CoopConfig>,
    caps: &blade_graphics::Capabilities,
    prepared: bool,
) -> (ExecutionPlan, Vec<Range<usize>>) {
    // A mirrored plan and its independently qualified parent share buffer
    // lifetimes. Preserve their physical order and fusion on every entry
    // point, including public constructors and saved-plan reloads.
    let prepared = prepared || !plan.f16_weight_mirrors.is_empty();

    if !prepared {
        // Per-dispatch kernel-variant selection: one pass, one owner.
        select_variants(
            &mut plan,
            coop_config,
            !opts.debug,
            opts.coop == CoopPolicy::AllowF16,
        );
        workgroup::select(&mut plan, caps, opts);
        if let Some(head_dim) = plan
            .dispatches
            .iter()
            .filter(|dispatch| dispatch.shader == ShaderEntry::FlashAttention)
            .map(|dispatch| {
                AttentionParams::from_words(&dispatch.params)
                    .expect("invalid attention parameters")
                    .head_dim
            })
            .max()
        {
            plan.knobs
                .flash
                .fit_shared_memory(head_dim, caps.max_compute_shared_memory_size);
        }
    }

    // Order the dispatches into barrier groups. The compiler owns this because
    // horizontal fusion changes how many dispatches there are, so the
    // groups cannot be computed independently of it.
    if opts.serial_dispatch {
        log::info!("MEGANEURA_SERIAL_DISPATCH: forcing one dispatch per pass");
    }
    let groups = if prepared {
        // The plan already owns its measured dispatch order and fusion.
        if opts.serial_dispatch {
            (0..plan.dispatches.len()).map(|i| i..i + 1).collect()
        } else {
            crate::compile::compute_groups(&plan.dispatches)
        }
    } else {
        crate::compile::schedule_dispatches(
            &mut plan,
            opts.serial_dispatch,
            !opts.serial_dispatch && !opts.debug,
        );
        std::mem::take(&mut plan.groups)
    };
    log::info!(
        "{} dispatches → {} barrier groups",
        plan.dispatches.len(),
        groups.len()
    );
    (plan, groups)
}

/// Initial allocation layout, before optional donation of existing parameters.
/// A prospective cost must account for donated bindings separately: those may
/// change physical identity, offsets and memory placement.
pub(super) fn memory_layout(
    plan: &ExecutionPlan,
    groups: &[Range<usize>],
    opts: &SessionOptions,
) -> (
    crate::memplan::AliasPlan,
    Vec<Option<blade_graphics::Memory>>,
) {
    let mut alias = if opts.debug || opts.no_alias {
        if opts.debug {
            log::info!("debug session: buffer aliasing disabled");
        } else {
            log::info!("MEGANEURA_NO_ALIAS: buffer aliasing disabled");
        }
        crate::memplan::plan_no_alias(plan, groups, opts.pin_buffers.as_deref())
    } else {
        crate::memplan::plan_buffer_aliasing(plan, groups, opts.pin_buffers.as_deref())
    };
    alias.pack_weight_mirrors(
        plan,
        opts.arena_chunk_bytes
            .unwrap_or(crate::memplan::ARENA_CHUNK_BYTES),
    );
    alias.pack_arena(
        plan,
        opts.arena_chunk_bytes
            .unwrap_or(crate::memplan::ARENA_CHUNK_BYTES),
    );
    let device_parameters = opts.device_parameters;
    // Only a parameter-bearing physical allocation may be
    // relocated, and it must be on its own physical buffer.
    let mut parameter_memory = vec![None; alias.sizes.len()];
    if let Some(memory) = device_parameters {
        for buffer in plan
            .param_buffers
            .iter()
            .map(|entry| entry.1)
            .chain(plan.derived_params.iter().map(|entry| entry.0))
        {
            if plan.input_buffers.iter().any(|entry| entry.1 == buffer)
                || plan.constant_buffers.iter().any(|entry| entry.0 == buffer)
                || plan.loss_buffer == Some(buffer)
                || !plan.buffers[buffer.0 as usize].is_multiple_of(4)
                || plan
                    .param_types
                    .get(&buffer)
                    .is_some_and(|ty| !ty.size_bytes().is_multiple_of(4))
            {
                continue;
            }
            let physical = alias.map[buffer.0 as usize];
            // Its own pinned allocation, or a parameter arena.
            assert!(
                alias.arena.iter().any(|chunk| chunk.params == physical)
                    || alias
                        .map
                        .iter()
                        .enumerate()
                        .all(|(i, &p)| p != physical || i == buffer.0 as usize),
                "device parameter must have its own pinned allocation"
            );
            alias.device_local[physical] = true;
            parameter_memory[physical] = Some(memory);
        }
    }
    // Step-local intermediates default to device-local memory on the
    // theory that host-visible (ReBAR) traffic is slower on discrete
    // boards; kill switch for measurement and UMA debugging.
    if opts.debug || opts.no_device_local {
        if !opts.debug {
            log::info!("MEGANEURA_NO_DEVICE_LOCAL: all buffers host-visible");
        }
        alias.device_local.fill(false);
    }
    (alias, parameter_memory)
}

/// Rebind compatible parameters using only layout metadata. The caller owns
/// any handles and decides how to retain or describe the returned bindings.
pub(super) fn shared_bindings<'a>(
    plan: &'a ExecutionPlan,
    alias: &'a mut crate::memplan::AliasPlan,
    source: &'a ExecutionPlan,
    source_alias: &'a crate::memplan::AliasPlan,
    immutable_only: bool,
) -> impl Iterator<Item = (usize, usize)> + 'a {
    super::shared_parameters(plan, source, immutable_only)
        .into_iter()
        .map(move |(target, source)| {
            let target = target.0 as usize;
            let source = source.0 as usize;
            let source_physical = source_alias.map[source];
            let physical = alias.rebind(
                target,
                plan.buffers[target],
                source_alias.offsets[source],
                source_alias.device_local[source_physical],
            );
            (physical, source_physical)
        })
}

/// Preserve slot indices used by optimizer metadata, without reserving retired
/// parameter arenas. The caller identifies any donated physical allocations.
pub(super) fn retire_unbound(
    alias: &mut crate::memplan::AliasPlan,
    parameter_memory: &mut Vec<Option<blade_graphics::Memory>>,
    provided: impl Fn(usize) -> bool,
) {
    parameter_memory.resize(alias.sizes.len(), None);
    for physical in 0..alias.sizes.len() {
        if !provided(physical) && !alias.map.contains(&physical) {
            alias.sizes[physical] = 0;
            alias.device_local[physical] = false;
            parameter_memory[physical] = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn training_plan() -> ExecutionPlan {
        let mut graph = crate::Graph::new();
        let x = graph.input("x", &[17, 32]);
        let w = graph.parameter("w", &[32, 64]);
        let y = graph.matmul(x, w);
        let y = graph.relu(y);
        let v = graph.parameter("v", &[64, 16]);
        let y = graph.matmul(y, v);
        let loss = graph.mean_all(y);
        graph.set_outputs(vec![loss]);
        crate::compile::compile(&crate::autodiff::differentiate(&graph))
    }

    #[test]
    fn placement_overrides_preserve_pins_arenas_and_host_visible_kill_switch() {
        let caps = blade_graphics::Capabilities::default();
        let opts = SessionOptions {
            coop: CoopPolicy::Disabled,
            ..Default::default()
        };
        let (plan, groups) = schedule(training_plan(), &opts, None, &caps, false);
        let (ordinary, memory) = memory_layout(&plan, &groups, &opts);
        assert!(ordinary.device_local.iter().any(|&v| v));
        assert!(memory.iter().all(Option::is_none));
        assert!(
            !ordinary.arena.is_empty(),
            "training parameters must be packed"
        );
        for &(_, buffer) in &plan.input_buffers {
            assert!(!ordinary.device_local[ordinary.map[buffer.0 as usize]]);
        }
        let device = SessionOptions {
            device_parameters: Some(blade_graphics::Memory::Device),
            ..opts.clone()
        };
        let (placed, memory) = memory_layout(&plan, &groups, &device);
        for chunk in &placed.arena {
            assert!(placed.device_local[chunk.params]);
            assert_eq!(memory[chunk.params], Some(blade_graphics::Memory::Device));
        }
        // A shared donor's placement and nonzero arena offsets override
        // the target's requested device placement; its old arena retires.
        assert!(
            plan.param_buffers
                .iter()
                .any(|(_, p)| ordinary.offsets[p.0 as usize] != 0)
        );
        let mut donated = placed;
        let mut donated_memory = memory;
        let old = donated.map[plan.param_buffers[0].1.0 as usize];
        let bindings: Vec<_> =
            shared_bindings(&plan, &mut donated, &plan, &ordinary, true).collect();
        assert_eq!(bindings.len(), plan.param_buffers.len());
        retire_unbound(&mut donated, &mut donated_memory, |p| {
            bindings.iter().any(|&(target, _)| target == p)
        });
        for (_, parameter) in &plan.param_buffers {
            let p = parameter.0 as usize;
            assert_eq!(donated.offsets[p], ordinary.offsets[p]);
            assert_eq!(
                donated.device_local[donated.map[p]],
                ordinary.device_local[ordinary.map[p]]
            );
        }
        assert!(!donated.map.contains(&old));
        assert_eq!(donated.sizes[old], 0);
        assert!(!donated.device_local[old]);
        assert_eq!(donated_memory[old], None);
        for debug in [false, true] {
            let host = SessionOptions {
                debug,
                no_device_local: true,
                ..opts.clone()
            };
            let (layout, memory) = memory_layout(&plan, &groups, &host);
            assert!(layout.device_local.iter().all(|&v| !v));
            assert!(memory.iter().all(Option::is_none));
        }
    }

    #[test]
    fn prospective_layout_matches_owned_and_shared_sessions() {
        let gpu = std::sync::Arc::new(
            crate::init_gpu_context_with(crate::GpuOptions::from_env()).unwrap(),
        );
        for (debug, no_alias, serial_dispatch, no_device_local) in [
            (false, false, false, false),
            (false, true, true, false),
            (true, false, false, false),
            (false, false, false, true),
        ] {
            let opts = SessionOptions {
                coop: CoopPolicy::Disabled,
                debug,
                no_alias,
                serial_dispatch,
                no_device_local,
                ..Default::default()
            };
            let raw = training_plan();
            let (expected, groups) = schedule(raw.clone(), &opts, None, &gpu.capabilities(), false);
            let (mut alias, mut memory) = memory_layout(&expected, &groups, &opts);
            retire_unbound(&mut alias, &mut memory, |_| false);
            let mut session = crate::Session::with_context_opts(raw, gpu.clone(), opts.clone());
            assert_eq!(session.plan(), &expected);
            assert_eq!(session.groups, groups);
            assert_eq!(session.alias.map, alias.map);
            assert_eq!(session.alias.offsets, alias.offsets);
            assert_eq!(session.alias.sizes, alias.sizes);
            assert_eq!(session.alias.device_local, alias.device_local);
            for (buffer, &bytes) in session.physical_buffers.iter().zip(&alias.sizes) {
                // Blade can pad the allocation; this is a requested-byte layout.
                assert!(buffer.handle.size() >= bytes.max(4) as u64);
            }
            let (prepared, groups) = schedule(
                session.plan().clone(),
                &opts,
                None,
                &gpu.capabilities(),
                true,
            );
            let (mut donated, mut memory) = memory_layout(&prepared, &groups, &opts);
            let bindings: Vec<_> = shared_bindings(
                &prepared,
                &mut donated,
                session.plan(),
                &session.alias,
                true,
            )
            .collect();
            retire_unbound(&mut donated, &mut memory, |p| {
                bindings.iter().any(|&(target, _)| target == p)
            });
            let shared = crate::Session::with_search_plan(
                prepared.clone(),
                gpu.clone(),
                opts,
                Some(&mut session),
                true,
            );
            assert_eq!(shared.plan(), &prepared);
            assert_eq!(shared.groups, groups);
            assert_eq!(shared.alias.map, donated.map);
            assert_eq!(shared.alias.offsets, donated.offsets);
            assert_eq!(shared.alias.sizes, donated.sizes);
            assert_eq!(shared.alias.device_local, donated.device_local);
            for (target, source) in bindings {
                assert!(std::sync::Arc::ptr_eq(
                    &shared.physical_buffers[target],
                    &session.physical_buffers[source]
                ));
            }
        }
    }
}
