//! A bounded local cache neighbor of an already qualified inference executable.
use super::{compile::ExecutionPlan, measure};
use std::collections::{HashSet, VecDeque};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Kind {
    Cooperative,
    Vector,
    CooperativeTuned,
    VectorTuned,
}

pub(super) struct Refinements {
    pub(super) enabled: bool,
    max_bytes: usize,
    tried: HashSet<(measure::Origin, Kind)>,
    pending: VecDeque<(measure::Origin, Kind, ExecutionPlan)>,
}
impl Refinements {
    pub(super) fn new(max_bytes: usize) -> Self {
        Self {
            enabled: false,
            max_bytes,
            tried: HashSet::new(),
            pending: VecDeque::new(),
        }
    }
    pub(super) fn selected(&mut self, origin: Option<measure::Origin>, plan: &ExecutionPlan) {
        self.pending.clear();
        let Some(origin) = origin else { return };
        if !self.enabled
            || !plan.param_grad_pairs.is_empty()
            || plan.dispatches.iter().any(|d| d.is_derivative)
        {
            return;
        }
        let mut native = plan.clone();
        if fill(&mut native, self.max_bytes) > 0 {
            native.groups = plan.groups.clone();
            if !self.tried.contains(&(origin, Kind::Cooperative)) {
                self.pending
                    .push_back((origin, Kind::Cooperative, native.clone()));
            }
        }
        if !plan.f16_weight_mirrors.is_empty() {
            let vector = plan.dispatches.iter().any(|d| {
                d.gemv_shape().is_some()
                    && d.weight_format == crate::compile::WeightFormat::F16
                    && plan
                        .f16_weight_mirrors
                        .iter()
                        .any(|entry| d.input_buffers.get(1) == Some(&entry.1))
            });
            let kind = if vector {
                Kind::VectorTuned
            } else {
                Kind::CooperativeTuned
            };
            if !self.tried.contains(&(origin, kind)) {
                self.pending.push_back((origin, kind, plan.clone()));
            }
        }
        // Keep the cooperative-only comparison. Rounding vector B operands
        // changes arithmetic and may fail when cooperative caching passes.
        if !self.tried.contains(&(origin, Kind::Vector)) {
            let mut vector = plan.clone();
            if fill_vectors(&mut vector, self.max_bytes) > 0 {
                vector.groups = plan.groups.clone();
                if vector != native {
                    self.pending.push_back((origin, Kind::Vector, vector));
                }
            }
        }
    }
    pub(super) fn next(&mut self) -> Option<measure::Program> {
        let (origin, kind, plan) = self.pending.pop_front()?;
        self.tried.insert((origin, kind));
        Some(measure::Program {
            prepared: true,
            retune: matches!(kind, Kind::CooperativeTuned | Kind::VectorTuned),
            description: format!("refine={origin:?}, coherent_weight_cache=true, weights={kind:?}"),
            plan,
            submission_chunks: None,
            origin: Some(origin),
        })
    }
    pub(super) fn early_remaining(&self) -> usize {
        self.pending
            .iter()
            .filter(|entry| entry.1 != Kind::Vector)
            .count()
    }
    pub(super) fn next_early(&mut self) -> Option<measure::Program> {
        if self
            .pending
            .front()
            .is_some_and(|entry| entry.1 != Kind::Vector)
        {
            self.next()
        } else {
            None
        }
    }
    pub(super) fn may_remain(&self) -> bool {
        !self.pending.is_empty()
    }
}

fn fill(plan: &mut ExecutionPlan, max_bytes: usize) -> usize {
    fill_impl(plan, max_bytes, false)
}
fn fill_vectors(plan: &mut ExecutionPlan, max_bytes: usize) -> usize {
    fill_impl(plan, max_bytes, true)
}

fn fill_impl(plan: &mut ExecutionPlan, max_bytes: usize, vectors: bool) -> usize {
    use crate::{
        DType,
        compile::{BufferRef, Kernel, WeightFormat},
    };
    use std::collections::{BTreeMap, HashMap};
    if (!vectors && !plan.f16_weight_mirrors.is_empty())
        || !plan.param_grad_pairs.is_empty()
        || plan.dispatches.iter().any(|d| d.is_derivative)
    {
        return 0;
    }
    let parameters: HashSet<_> = plan
        .param_buffers
        .iter()
        .map(|p| p.1)
        .chain(plan.derived_params.iter().map(|p| p.0))
        .collect();
    let written: HashSet<_> = plan
        .dispatches
        .iter()
        .flat_map(|d| std::iter::once(d.output_buffer).chain(d.extra_outputs.iter().copied()))
        .collect();
    let mut sources = BTreeMap::new();
    for d in &plan.dispatches {
        for (slot, _) in mirror_operands(d, vectors).into_iter().flatten() {
            let buffer = d.input_buffers[slot];
            let bytes = plan.buffers[buffer.0 as usize];
            if parameters.contains(&buffer)
                && !written.contains(&buffer)
                && plan
                    .param_types
                    .get(&buffer)
                    .is_some_and(|t| t.dtype == DType::F32)
                && plan
                    .weight_buffers
                    .get(&buffer)
                    .is_none_or(|x| x.0 == WeightFormat::F32)
                && bytes > 0
                && bytes.is_multiple_of(4)
            {
                sources.insert(buffer.0, (bytes / 4).div_ceil(2) * 4);
            }
        }
    }
    let mut mirrors: HashMap<_, _> = plan.f16_weight_mirrors.iter().copied().collect();
    let Some(extra) = plan
        .f16_weight_mirrors
        .iter()
        .map(|entry| plan.buffers[entry.1.0 as usize])
        .chain(
            sources
                .iter()
                .filter(|entry| !mirrors.contains_key(&BufferRef(*entry.0)))
                .map(|entry| entry.1.next_multiple_of(crate::memplan::ARENA_ALIGNMENT)),
        )
        .try_fold(0usize, |sum, bytes| sum.checked_add(bytes))
    else {
        return 0;
    };
    if sources.is_empty() || extra > max_bytes {
        return 0;
    }
    // Extend the actual selected executable, retaining any later kernel
    // refinements. Keep its existing qualified fallback without nesting it.
    let parent = plan
        .f16_weight_parent
        .clone()
        .unwrap_or_else(|| Box::new(plan.clone()));
    for (source, bytes) in sources {
        let source = BufferRef(source);
        if mirrors.contains_key(&source) {
            continue;
        }
        let destination = BufferRef(u32::try_from(plan.buffers.len()).unwrap());
        plan.buffers
            .push(bytes.next_multiple_of(crate::memplan::ARENA_ALIGNMENT));
        plan.f16_weight_mirrors.push((source, destination));
        mirrors.insert(source, destination);
    }
    for d in &mut plan.dispatches {
        for (slot, operand) in mirror_operands(d, vectors).into_iter().flatten() {
            let Some(&mirror) = mirrors.get(&d.input_buffers[slot]) else {
                continue;
            };
            d.input_buffers[slot] = mirror;
            match d.kernel {
                Kernel::CooperativeTyped { ref mut types, .. }
                | Kernel::CooperativeWorkgroup {
                    types: Some(ref mut types),
                    ..
                } => {
                    if operand == 0 {
                        types.a = DType::F16;
                    } else {
                        types.b = DType::F16;
                    }
                }
                Kernel::Gemv { .. } => d.weight_format = WeightFormat::F16,
                _ => unreachable!("mirror operand requires a supported kernel"),
            }
        }
    }
    plan.f16_weight_parent = Some(parent);
    plan.groups.clear();
    mirrors.len()
}

// (binding slot, mathematical operand). Convolution's operands are reversed.
fn mirror_operands(d: &crate::compile::Dispatch, vectors: bool) -> [Option<(usize, usize)>; 2] {
    use crate::{
        DType,
        compile::{Kernel, ShaderEntry, WeightFormat},
    };
    if d.full_precision() || d.matmul_prologue.is_some() || d.matmul_epilogue.is_some() {
        return [None; 2];
    }
    // GEMV's existing decoder unpacks B into F32 before multiplication. A,
    // accumulation, bias and outputs stay F32; the private weight copy rounds.
    if vectors
        && matches!(
            d.shader,
            ShaderEntry::MatMulGemv
                | ShaderEntry::MatMulGemvAdd
                | ShaderEntry::MatMulGemvBT
                | ShaderEntry::MatMulGemvBTAdd
        )
        && matches!(
            d.kernel,
            Kernel::Gemv {
                integer_dot: false,
                ..
            }
        )
        && d.weight_format == WeightFormat::F32
        && d.mnk()
            .is_some_and(|(m, n, k)| m == 1 && n.is_multiple_of(4) && k.is_multiple_of(4))
    {
        return [Some((1, 1)), None];
    }
    let Some(slots) = weight_mirror_operand_slots(d) else {
        return [None; 2];
    };
    let types = match d.kernel {
        Kernel::CooperativeTyped { types, .. }
        | Kernel::CooperativeWorkgroup {
            types: Some(types), ..
        } if types.multiply == DType::F16 => types,
        _ => return [None; 2],
    };
    [
        (types.a == DType::F32).then_some((slots[0], 0)),
        (types.b == DType::F32).then_some((slots[1], 1)),
    ]
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
    fn vector_graph(bt: bool, add: bool, norm: bool) -> (Graph, ExecutionPlan) {
        let mut g = Graph::new();
        let x = g.input("x", &[1, 36]);
        let a = if norm {
            let scale = g.parameter("scale", &[36]);
            g.rms_norm(x, scale, 1e-5)
        } else {
            x
        };
        let w = g.parameter("w", if bt { &[20, 36] } else { &[36, 20] });
        let y = if bt {
            g.matmul_bt(a, w)
        } else {
            g.matmul(a, w)
        };
        if add {
            let bias = g.input("bias", &[1, 20]);
            let n = &mut g.nodes_mut()[y as usize];
            n.op = if bt {
                crate::graph::Op::FusedMatMulBTAdd
            } else {
                crate::graph::Op::FusedMatMulAdd
            };
            n.inputs.push(bias);
        }
        g.set_outputs(vec![y]);
        let mut p = compile::compile(&g);
        if norm {
            compile::fuse_rmsnorm_into_gemv(&mut p);
        }
        assert_eq!(p.dispatches.len(), if norm && add { 2 } else { 1 });
        let vector = p
            .dispatches
            .iter()
            .find(|d| d.gemv_shape().is_some())
            .unwrap();
        assert_eq!(vector.gemv_rmsnorm.is_some(), norm && !add);
        (g, p)
    }
    #[test]
    fn vector_cache_keeps_cooperative_choice_and_shared_mirror() {
        let mut g = Graph::new();
        let x = g.input("x", &[32, 32]);
        let v = g.input("v", &[1, 32]);
        let w = g.parameter("w", &[32, 32]);
        let a = g.matmul(x, w);
        let b = g.matmul(v, w);
        g.set_outputs(vec![a, b]);
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
        let origin = Some(measure::Origin { form: 0, choice: 0 });
        let mut r = Refinements::new(2048);
        r.enabled = true;
        r.selected(origin, &p);
        assert_eq!(r.pending.len(), 2);
        assert_eq!(r.early_remaining(), 1);
        let cooperative = r.next_early().unwrap();
        assert_eq!(cooperative.plan.f16_weight_mirrors.len(), 1);
        assert!(
            cooperative
                .plan
                .dispatches
                .iter()
                .any(|d| d.gemv_shape().is_some() && d.weight_format == compile::WeightFormat::F32)
        );
        let mut refined = cooperative.plan.clone();
        let vector_index = refined
            .dispatches
            .iter()
            .position(|d| d.gemv_shape().is_some())
            .unwrap();
        // Another measured refinement may lock an existing kernel after the
        // cache was installed. A vector neighbor must retain that choice.
        refined.dispatches[vector_index].schedule_locked =
            !refined.dispatches[vector_index].schedule_locked;
        r.selected(origin, &refined);
        assert_eq!(r.pending.len(), 2);
        let tuned = r.next_early().unwrap();
        assert!(tuned.prepared && tuned.retune);
        assert_eq!(tuned.plan, refined);
        assert_eq!(r.pending.len(), 1);
        assert_eq!(r.early_remaining(), 0);
        assert!(
            r.next_early().is_none(),
            "vector rounding waits for ordinary layouts"
        );
        let vector = r.next().unwrap();
        assert_eq!(
            vector.plan.dispatches[vector_index].schedule_locked,
            refined.dispatches[vector_index].schedule_locked,
            "must extend the current executable, not rebuild its older parent"
        );
        assert_eq!(
            vector.plan.f16_weight_mirrors,
            cooperative.plan.f16_weight_mirrors
        );
        assert_eq!(vector.plan.f16_weight_parent.as_deref(), Some(&p));
        assert_eq!(vector.plan.buffers, cooperative.plan.buffers);
        assert!(
            vector
                .plan
                .dispatches
                .iter()
                .any(|d| d.gemv_shape().is_some() && d.weight_format == compile::WeightFormat::F16)
        );
        r.selected(origin, &vector.plan);
        let tuned = r.next_early().unwrap();
        assert!(tuned.prepared && tuned.retune);
        assert_eq!(tuned.plan, vector.plan);
        r.selected(origin, &tuned.plan);
        assert_eq!(r.pending.len(), 0);
        let (_, p) = vector_graph(false, false, false);
        for case in 0..5 {
            let mut p = p.clone();
            let original = p.clone();
            let budget = if case == 0 { 0 } else { usize::MAX };
            match case {
                1 => p.dispatches[0].requires_full_precision = true,
                2 => p.dispatches[0].is_derivative = true,
                3 => p
                    .param_grad_pairs
                    .push((p.param_buffers[0].1, p.output_buffers[0])),
                4 => p.dispatches.push(compile::Dispatch {
                    output_buffer: p.param_buffers[0].1,
                    ..Default::default()
                }),
                _ => {}
            }
            let before = p.clone();
            assert_eq!(fill_vectors(&mut p, budget), 0, "case {case}");
            assert_eq!(p, before);
            assert!(original.f16_weight_mirrors.is_empty());
        }
    }
    #[test]
    fn vector_cache_decoders_preserve_f32_inputs_and_accumulation() {
        use crate::{Session, SessionOptions};
        let gpu = crate::reference::gpu::shared_context();
        for bt in [false, true] {
            for add in [false, true] {
                for norm in [false, true] {
                    let (_, mut p) = vector_graph(bt, add, norm);
                    assert_eq!(fill_vectors(&mut p, 2048), 1);
                    let restored: ExecutionPlan =
                        serde_json::from_str(&serde_json::to_string(&p).unwrap()).unwrap();
                    assert_eq!(restored, p);
                    let mut session =
                        Session::with_context_opts(p, gpu.clone(), SessionOptions::default());
                    // F16 subnormal B and A beyond F16 range expose accidental A
                    // narrowing or F16 arithmetic, independent of a matching GPU parent.
                    let x: Vec<f32> = (0..36)
                        .map(|i| 1048576.0 + (i % 7) as f32 * 1024.0)
                        .collect();
                    let w: Vec<f32> = (0..720)
                        .map(|i| (1.0 + (i % 11) as f32 / 10.0) * 2f32.powi(-20))
                        .collect();
                    let bias: Vec<f32> = (0..20).map(|i| i as f32 / 8.0).collect();
                    session.set_input("x", &x);
                    session.set_parameter("w", &w);
                    if norm {
                        session.set_parameter("scale", &[1.; 36]);
                    }
                    if add {
                        session.set_input("bias", &bias);
                    }
                    session.step();
                    session.wait();
                    let got = session.read_output(20);
                    let divisor = if norm {
                        (x.iter().map(|&a| f64::from(a).powi(2)).sum::<f64>() / 36.0 + 1e-5).sqrt()
                    } else {
                        1.0
                    };
                    for j in 0..20 {
                        let want = (0..36)
                            .map(|k| {
                                f64::from(x[k]) / divisor
                                    * f64::from(
                                        half::f16::from_f32(
                                            w[if bt { j * 36 + k } else { k * 20 + j }],
                                        )
                                        .to_f32(),
                                    )
                            })
                            .sum::<f64>()
                            + if add { f64::from(bias[j]) } else { 0.0 };
                        assert!(
                            (f64::from(got[j]) - want).abs() < 1e-9 + want.abs() * 2e-5,
                            "bt={bt}, add={add}, norm={norm}, j={j}, got={}, want={want}",
                            got[j]
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn vector_rounding_still_rejects_cancellation_and_overflow() {
        use crate::{Session, SessionOptions};
        let gpu = crate::reference::gpu::shared_context();
        for overflow in [false, true] {
            let (_, parent) = vector_graph(false, false, false);
            let mut cached = parent.clone();
            assert_eq!(fill_vectors(&mut cached, 2048), 1);
            let mut original =
                Session::with_context_opts(parent, gpu.clone(), SessionOptions::default());
            let mut candidate =
                Session::with_context_opts(cached, gpu.clone(), SessionOptions::default());
            let mut w = vec![0.; 720];
            for j in 0..20 {
                w[j] = if overflow { 70000. } else { 1.0001 };
                if !overflow {
                    w[20 + j] = -1.;
                }
            }
            for s in [&mut original, &mut candidate] {
                s.set_input("x", &[1.; 36]);
                s.set_parameter("w", &w);
                s.step();
                s.wait();
            }
            let want = original.read_output(20);
            let got = candidate.read_output(20);
            assert!(want.iter().all(|x| x.is_finite() && *x != 0.));
            let relative_rms = (got
                .iter()
                .zip(&want)
                .map(|(&a, &b)| f64::from(a - b).powi(2))
                .sum::<f64>()
                / want.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>())
            .sqrt();
            assert!(
                !relative_rms.is_finite() || relative_rms > 0.01,
                "rounding must remain a qualified alternative, overflow={overflow}"
            );
        }
    }
    #[test]
    fn vector_cache_tracks_restore_share_record_and_gpu_writes() {
        use crate::{Session, SessionOptions};
        let gpu = crate::reference::gpu::shared_context();
        let (graph, mut parent) = vector_graph(false, false, false);
        parent.output_buffers.push(parent.param_buffers[0].1);
        let mut cached = parent.clone();
        assert_eq!(fill_vectors(&mut cached, 2048), 1);
        let path =
            std::env::temp_dir().join(format!("meganeura-vector-cache-{}.ron", std::process::id()));
        crate::cache::save_plan(&cached, &graph, &path).unwrap();
        let restored = crate::cache::load_plan(&graph, &path).unwrap().unwrap();
        assert_eq!(restored, cached);
        std::fs::remove_file(&path).unwrap();
        let mut s = Session::with_context_opts(restored, gpu.clone(), SessionOptions::default());
        s.set_input("x", &[1.; 36]);
        s.set_parameter("w", &[1.; 720]);
        let check = |s: &mut Session, scale: f32| {
            s.step();
            s.wait();
            assert_eq!(s.read_output(20), vec![36. * scale; 20]);
        };
        check(&mut s, 1.);
        let checkpoint = path.with_extension("safetensors");
        s.save_checkpoint(&checkpoint).unwrap();
        s.upload_param("w", &[2.; 720]);
        check(&mut s, 2.);
        s.load_checkpoint(&checkpoint).unwrap();
        check(&mut s, 1.);
        std::fs::remove_file(checkpoint).unwrap();
        let mut owner = Session::with_context_opts(parent, gpu.clone(), SessionOptions::default());
        owner.set_parameter("w", &[3.; 720]);
        s.share_parameter_from(&mut owner, "w").unwrap();
        check(&mut s, 3.);
        owner.upload_param("w", &[4.; 720]);
        check(&mut s, 4.);
        let mut enc = gpu.create_command_encoder(blade_graphics::CommandEncoderDesc {
            name: "vector-cache-test",
            buffer_count: 1,
            manual_barriers: false,
        });
        enc.start();
        s.record(&mut enc).unwrap();
        owner.upload_param("w", &[5.; 720]);
        let sync = gpu.submit(&mut enc);
        s.track_submission(sync);
        s.wait();
        assert_eq!(s.read_output(20), vec![180.; 20]);
        check(&mut s, 5.);
        // Publishing an output that aliases the shared master makes all
        // caches of that allocation use the parent. A real GPU write after
        // the last cache refresh must be observed by the next owned step.
        let exposed = owner.output_buffer(1).unwrap();
        enc.start();
        enc.transfer("replace-vector-weight")
            .fill_buffer(exposed, 2880, 0);
        let sync = gpu.submit(&mut enc);
        s.track_submission(sync);
        s.wait();
        check(&mut s, 0.);
        gpu.destroy_command_encoder(&mut enc);
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
        let tuned = r.next_early().unwrap();
        assert!(tuned.prepared && tuned.retune);
        assert_eq!(tuned.plan, n.plan);
        r.selected(origin, &tuned.plan);
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
