//! A bounded local cache neighbor of an already qualified inference executable.
use super::{compile::ExecutionPlan, measure};
use std::collections::HashSet;

pub(super) struct Refinements {
    pub(super) enabled: bool,
    max_bytes: usize,
    tried: HashSet<measure::Origin>,
    pending: Option<(measure::Origin, ExecutionPlan)>,
}
impl Refinements {
    pub(super) fn new(max_bytes: usize) -> Self {
        Self {
            enabled: false,
            max_bytes,
            tried: HashSet::new(),
            pending: None,
        }
    }
    pub(super) fn selected(&mut self, origin: Option<measure::Origin>, plan: &ExecutionPlan) {
        self.pending = None;
        let Some(origin) = origin else { return };
        if !self.enabled
            || self.tried.contains(&origin)
            || !plan.f16_weight_mirrors.is_empty()
            || !plan.param_grad_pairs.is_empty()
            || plan.dispatches.iter().any(|d| d.is_derivative)
        {
            return;
        }
        let mut native = plan.clone();
        if fill(&mut native, self.max_bytes) > 0 {
            // There are no added compute dependencies. Keep all of the actual
            // winner's kernel choices, fusion and pass ranges.
            native.groups = plan.groups.clone();
            self.pending = Some((origin, native));
        }
    }
    pub(super) fn next(&mut self) -> Option<measure::Program> {
        let (origin, plan) = self.pending.take()?;
        self.tried.insert(origin);
        Some(measure::Program {
            prepared: true,
            description: format!("refine={origin:?}, coherent_weight_cache=true"),
            plan,
            submission_chunks: None,
            origin: Some(origin),
        })
    }
    pub(super) fn may_remain(&self) -> bool {
        self.pending.is_some()
    }
}

fn fill(plan: &mut crate::compile::ExecutionPlan, max_bytes: usize) -> usize {
    use crate::{
        DType,
        compile::{BufferRef, Kernel, WeightFormat},
    };
    use std::collections::{BTreeMap, HashMap, HashSet};
    if !plan.f16_weight_mirrors.is_empty() {
        return 0;
    }
    let parameters: HashSet<_> = plan
        .param_buffers
        .iter()
        .map(|p| p.1)
        .chain(plan.derived_params.iter().map(|p| p.0))
        .collect();
    let graph_written: HashSet<_> = plan
        .dispatches
        .iter()
        .flat_map(|d| std::iter::once(d.output_buffer).chain(d.extra_outputs.iter().copied()))
        .collect();
    let mut sources = BTreeMap::new();
    for d in &plan.dispatches {
        let Some(slots) = weight_mirror_operand_slots(d) else {
            continue;
        };
        if d.full_precision() || d.matmul_prologue.is_some() || d.matmul_epilogue.is_some() {
            continue;
        }
        let types = match d.kernel {
            Kernel::CooperativeTyped { types, .. }
            | Kernel::CooperativeWorkgroup {
                types: Some(types), ..
            } => types,
            _ => continue,
        };
        if types.multiply != DType::F16 {
            continue;
        }
        for (slot, storage) in slots.into_iter().zip([types.a, types.b]) {
            let buffer = d.input_buffers[slot];
            if storage != DType::F32
                || !parameters.contains(&buffer)
                || graph_written.contains(&buffer)
            {
                continue;
            }
            if plan
                .param_types
                .get(&buffer)
                .is_none_or(|t| t.dtype != DType::F32)
            {
                continue;
            }
            if plan
                .weight_buffers
                .get(&buffer)
                .is_some_and(|x| x.0 != WeightFormat::F32)
            {
                continue;
            }
            let bytes = plan.buffers[buffer.0 as usize];
            if !bytes.is_multiple_of(4) || bytes == 0 {
                continue;
            }
            sources.insert(buffer.0, (bytes / 4).div_ceil(2) * 4);
        }
    }
    let extra: usize = sources
        .values()
        .map(|bytes| bytes.next_multiple_of(crate::memplan::ARENA_ALIGNMENT))
        .sum();
    if extra > max_bytes {
        return 0;
    }
    let parent = plan.clone();
    let mut mirrors = HashMap::new();
    for (source, bytes) in sources {
        let source = BufferRef(source);
        let destination = BufferRef(u32::try_from(plan.buffers.len()).unwrap());
        plan.buffers
            .push(bytes.next_multiple_of(crate::memplan::ARENA_ALIGNMENT));
        plan.f16_weight_mirrors.push((source, destination));
        mirrors.insert(source, destination);
    }
    for d in &mut plan.dispatches {
        let Some(slots) = weight_mirror_operand_slots(d) else {
            continue;
        };
        if d.full_precision() || d.matmul_prologue.is_some() || d.matmul_epilogue.is_some() {
            continue;
        }
        let types = match d.kernel {
            Kernel::CooperativeTyped { ref mut types, .. }
            | Kernel::CooperativeWorkgroup {
                types: Some(ref mut types),
                ..
            } => types,
            _ => continue,
        };
        if types.multiply != DType::F16 {
            continue;
        }
        for (slot, storage) in slots.into_iter().zip([&mut types.a, &mut types.b]) {
            if *storage == DType::F32
                && let Some(&mirror) = mirrors.get(&d.input_buffers[slot])
            {
                d.input_buffers[slot] = mirror;
                *storage = DType::F16;
            }
        }
    }
    if !mirrors.is_empty() {
        plan.f16_weight_parent = Some(Box::new(parent));
        plan.groups.clear();
    }
    mirrors.len()
}

fn weight_mirror_operand_slots(d: &crate::compile::Dispatch) -> Option<[usize; 2]> {
    if d.shader.is_matmul() {
        Some([0, 1])
    } else if matches!(d.shader, crate::compile::ShaderEntry::Conv2dGemmCoopGen(..)) {
        Some([1, 0])
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, Graph, compile};
    fn plan() -> ExecutionPlan {
        let mut g = Graph::new();
        let x = g.input("x", &[32, 32]);
        let w = g.parameter("w", &[32, 32]);
        let y = g.matmul(x, w);
        g.set_outputs(vec![y]);
        let mut p = compile::compile_with(
            &g,
            &compile::CompileOptions {
                fuse_dispatches: false,
                ..Default::default()
            },
        );
        assert_eq!(
            super::super::fill_matrix_candidates(&mut p, DType::F16, true),
            1
        );
        p
    }
    #[test]
    fn cache_neighbor_is_local_prepared_and_once_per_origin() {
        let origin = Some(measure::Origin { form: 2, choice: 3 });
        let p = plan();
        let mut r = Refinements::new(2048);
        r.selected(origin, &p);
        assert!(r.next().is_none());
        r.enabled = true;
        r.selected(origin, &p);
        let n = r.next().unwrap();
        assert!(n.prepared);
        assert_eq!(n.origin, origin);
        assert!(n.submission_chunks.is_none());
        assert_eq!(n.plan.groups, p.groups);
        assert_eq!(n.plan.f16_weight_parent.as_deref(), Some(&p));
        assert_eq!(n.plan.f16_weight_mirrors.len(), 1);
        assert_eq!(n.plan.param_buffers, p.param_buffers);
        let restored: ExecutionPlan =
            serde_json::from_str(&serde_json::to_string(&n.plan).unwrap()).unwrap();
        assert_eq!(restored, n.plan);
        r.selected(origin, &p);
        assert!(r.next().is_none());
        r.selected(Some(measure::Origin { form: 3, choice: 3 }), &p);
        assert!(r.next().is_some());
        r.selected(origin, &n.plan);
        assert!(!r.may_remain());
    }
    #[test]
    fn cache_neighbor_keeps_bounds_pins_training_and_operand_contract() {
        let origin = Some(measure::Origin { form: 0, choice: 0 });
        for case in 0..6 {
            let mut p = plan();
            let mut r = Refinements::new(if case == 0 { 2047 } else { 2048 });
            r.enabled = true;
            match case {
                1 => p.dispatches[0].requires_full_precision = true,
                2 => p.dispatches[0].is_derivative = true,
                3 => p
                    .param_grad_pairs
                    .push((p.param_buffers[0].1, p.output_buffers[0])),
                4 => {
                    if let compile::Kernel::CooperativeTyped { ref mut types, .. } =
                        p.dispatches[0].kernel
                    {
                        types.multiply = DType::BF16;
                    }
                }
                5 => {
                    let b = p.param_buffers[0].1;
                    p.dispatches.push(compile::Dispatch {
                        output_buffer: b,
                        ..Default::default()
                    });
                }
                _ => {}
            }
            r.selected(origin, &p);
            assert!(r.next().is_none(), "case {case}");
        }
        let mut r = Refinements::new(2048);
        r.enabled = true;
        r.selected(origin, &plan());
        assert!(r.may_remain());
        r.selected(None, &plan());
        assert!(
            !r.may_remain(),
            "a new winner discards the previous pending neighbor"
        );
    }
    #[test]
    fn saved_cache_preserves_master_updates_and_caller_recording() {
        use crate::{Session, SessionOptions};
        let gpu = crate::reference::gpu::shared_context();
        if !crate::runtime::typed_matrix_supported(&gpu, DType::F16) {
            return;
        }
        let mut g = Graph::new();
        let x = g.input("x", &[32, 32]);
        let w = g.parameter("w", &[32, 32]);
        let y = g.matmul(x, w);
        g.set_outputs(vec![y]);
        let mut cached = plan();
        assert_eq!(fill(&mut cached, 2048), 1);
        let file = std::env::temp_dir().join(format!(
            "meganeura-coherent-cache-{}.ron",
            std::process::id()
        ));
        crate::cache::save_plan(&cached, &g, &file).unwrap();
        let restored = crate::cache::load_plan(&g, &file).unwrap().unwrap();
        assert_eq!(cached, restored);
        let mut s = Session::with_context_opts(restored, gpu.clone(), SessionOptions::default());
        let inputs: Vec<f32> = (0..1024).map(|i| (i % 7) as f32 / 8.).collect();
        let weights = |scale: f32| {
            (0..1024)
                .map(|i| if i / 32 == i % 32 { scale } else { 0.0 })
                .collect::<Vec<_>>()
        };
        let check = |s: &Session, scale: f32| {
            let got = s.read_output(1024);
            for (&a, &b) in got.iter().zip(&inputs) {
                assert_eq!(a, b * scale);
            }
        };
        s.set_input("x", &inputs);
        s.set_parameter("w", &weights(1.));
        s.step();
        s.wait();
        check(&s, 1.);
        s.upload_param("w", &weights(2.));
        s.step();
        s.wait();
        check(&s, 2.);
        // An ordinary owner has never counted writes before a cache shares it.
        let mut owner = Session::with_context_opts(plan(), gpu.clone(), SessionOptions::default());
        owner.set_parameter("w", &weights(3.));
        s.share_parameter_from(&mut owner, "w").unwrap();
        s.step();
        s.wait();
        check(&s, 3.);
        owner.upload_param("w", &weights(4.));
        s.step();
        s.wait();
        check(&s, 4.);
        let mut enc = gpu.create_command_encoder(blade_graphics::CommandEncoderDesc {
            name: "coherent-cache-test",
            buffer_count: 1,
            manual_barriers: false,
        });
        enc.start();
        s.record(&mut enc).unwrap();
        owner.set_parameter("w", &weights(5.));
        let sync = gpu.submit(&mut enc);
        s.track_submission(sync);
        s.wait();
        check(&s, 5.);
        s.step();
        s.wait();
        check(&s, 5.);
        gpu.destroy_command_encoder(&mut enc);
        // New schemas must not be decoded through a matching older version.
        let current = std::fs::read_to_string(&file).unwrap();
        let stale = current.replacen("format_version: 39,", "format_version: 38,", 1);
        assert_ne!(current, stale);
        std::fs::write(&file, stale).unwrap();
        assert!(crate::cache::load_plan(&g, &file).unwrap().is_none());
        std::fs::remove_file(&file).unwrap();
        drop(owner);
        s.step();
        s.wait();
        check(&s, 5.);
    }
    #[test]
    fn recorded_parent_preserves_depth_first_alias_lifetimes() {
        use crate::{Session, SessionOptions};
        let gpu = crate::reference::gpu::shared_context();
        if !crate::runtime::typed_matrix_supported(&gpu, DType::F16) {
            return;
        }
        let mut g = Graph::new();
        let x = g.input("x", &[32, 32]);
        let w0 = g.parameter("w0", &[32, 32]);
        let a = g.matmul(x, w0);
        let a = g.relu(a);
        let w1 = g.parameter("w1", &[32, 32]);
        let a = g.matmul(a, w1);
        let a = g.relu(a);
        let w2 = g.parameter("w2", &[32, 32]);
        let b = g.matmul(x, w2);
        let b = g.relu(b);
        let w3 = g.parameter("w3", &[32, 32]);
        let b = g.matmul(b, w3);
        let b = g.relu(b);
        let y = g.add(a, b);
        g.set_outputs(vec![y]);
        let mut parent = compile::compile_with(
            &g,
            &compile::CompileOptions {
                fuse_dispatches: false,
                ..Default::default()
            },
        );
        // Retain a legal depth-first order that differs from level scheduling.
        fn visit(
            i: usize,
            ds: &[compile::Dispatch],
            seen: &mut HashSet<usize>,
            order: &mut Vec<usize>,
        ) {
            if !seen.insert(i) {
                return;
            }
            for input in &ds[i].input_buffers {
                if let Some(j) = ds
                    .iter()
                    .position(|d| d.output_buffer == *input || d.extra_outputs.contains(input))
                {
                    visit(j, ds, seen, order);
                }
            }
            order.push(i);
        }
        let mut order = Vec::new();
        let mut seen = HashSet::new();
        for i in (0..parent.dispatches.len()).rev() {
            visit(i, &parent.dispatches, &mut seen, &mut order);
        }
        let old = parent.dispatches.clone();
        parent.dispatches = order.into_iter().map(|i| old[i].clone()).collect();
        parent.groups.clear();
        assert_eq!(
            super::super::fill_matrix_candidates(&mut parent, DType::F16, true),
            4
        );
        let mut native = parent.clone();
        assert_eq!(fill(&mut native, 1 << 20), 4);
        let mut expected = Session::with_context_opts(
            parent,
            gpu.clone(),
            SessionOptions {
                serial_dispatch: true,
                no_alias: true,
                ..Default::default()
            },
        );
        let mut actual = Session::with_context_opts(
            native.clone(),
            gpu.clone(),
            SessionOptions {
                serial_dispatch: true,
                ..Default::default()
            },
        );
        for s in [&mut expected, &mut actual] {
            s.set_input(
                "x",
                &(0..1024).map(|i| (i % 7) as f32 / 8.).collect::<Vec<_>>(),
            );
            for i in 0..4 {
                s.set_parameter(
                    &format!("w{i}"),
                    &(0..1024)
                        .map(|j| {
                            if j / 32 == j % 32 {
                                (i + 1) as f32 / 8.
                            } else {
                                0.
                            }
                        })
                        .collect::<Vec<_>>(),
                );
            }
            s.step();
            s.wait();
        }
        let wanted = expected.read_output(1024);
        assert_eq!(actual.read_output(1024), wanted);
        let mut encoder = gpu.create_command_encoder(blade_graphics::CommandEncoderDesc {
            name: "depth-first-cache-parent",
            buffer_count: 1,
            manual_barriers: false,
        });
        encoder.start();
        actual.record(&mut encoder).unwrap();
        let sync = gpu.submit(&mut encoder);
        actual.track_submission(sync);
        actual.wait();
        assert_eq!(actual.read_output(1024), wanted);
        assert_eq!(actual.plan(), &native);
        gpu.destroy_command_encoder(&mut encoder);
    }
}
