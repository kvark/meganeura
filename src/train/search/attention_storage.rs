//! Keep the final narrowing at private K/V producers. Attention already rounds
//! these values to F16; Q, softmax, accumulation, output and LSE stay F32.
use super::*;
use crate::{DType, schedule::PointwiseStorage};
use compile::{BufferRef, Dispatch, DispatchParams as _, Kernel, ShaderEntry};

fn producer_output(dispatch: &mut Dispatch) -> bool {
    if dispatch.pointwise().is_some() && dispatch.pointwise_storage == PointwiseStorage::default() {
        dispatch.pointwise_storage.output = DType::F16;
        return true;
    }
    if !matches!(
        dispatch.shader,
        ShaderEntry::MatMul
            | ShaderEntry::MatMulAT
            | ShaderEntry::MatMulBT
            | ShaderEntry::FusedMatMulAdd
            | ShaderEntry::FusedMatMulATAdd
            | ShaderEntry::FusedMatMulBTAdd
    ) {
        return false;
    }
    let types = match dispatch.kernel {
        Kernel::CooperativeTyped {
            ref mut types,
            splits: 1,
        }
        | Kernel::CooperativeWorkgroup {
            types: Some(ref mut types),
            splits: 1,
            ..
        } => types,
        _ => return false,
    };
    // Multiplication and accumulation do not change. The reader already
    // rounds the F32 result to F16, independent of the producer arithmetic.
    if types.output != DType::F32 {
        return false;
    }
    types.output = DType::F16;
    true
}

fn writer(
    plan: &ExecutionPlan,
    buffer: BufferRef,
    reader: usize,
    slot: usize,
    elements: usize,
) -> Option<usize> {
    if plan.buffers[buffer.0 as usize] != elements.checked_mul(4)? {
        return None;
    }
    let mut producer = None;
    for (index, dispatch) in plan.dispatches.iter().enumerate() {
        if dispatch.output_buffer == buffer || dispatch.extra_outputs.contains(&buffer) {
            if producer.replace(index).is_some() {
                return None;
            }
        }
        for (input, &b) in dispatch.input_buffers.iter().enumerate() {
            if b == buffer && (index != reader || input != slot) {
                return None;
            }
        }
        if dispatch
            .matmul_prologue
            .iter()
            .flat_map(|p| p.factors.iter().map(|x| x.0))
            .chain(
                dispatch
                    .matmul_epilogue
                    .iter()
                    .flat_map(|p| p.inputs.iter().map(|x| x.0)),
            )
            .chain(dispatch.gemv_rmsnorm.iter().map(|p| p.weight))
            .any(|b| b == buffer)
        {
            return None;
        }
    }
    let index = producer?;
    let d = &plan.dispatches[index];
    if index >= reader
        || d.requires_full_precision
        || d.schedule_locked
        || d.fusion_barrier
        || d.horizontal_batch > 1
        || !d.extra_outputs.is_empty()
        || d.matmul_prologue.is_some()
        || d.matmul_epilogue.is_some()
        || d.gemv_rmsnorm.is_some()
    {
        return None;
    }
    Some(index)
}

/// Retain the staged parent and qualify this as a complete measured neighbor.
/// Both producers must be legal before either buffer is changed.
pub(super) fn fill(plan: &mut ExecutionPlan) -> usize {
    if !plan.param_grad_pairs.is_empty()
        || plan
            .dispatches
            .iter()
            .any(|d| matches!(d.shader, ShaderEntry::AttentionGrad(_)))
    {
        return 0;
    }
    let protected = plan.externally_visible_buffers();
    let mut count = 0;
    for reader in 0..plan.dispatches.len() {
        let d = &plan.dispatches[reader];
        if d.shader != (ShaderEntry::FlashAttentionProducts { kv_f16: false })
            || d.requires_full_precision
            || d.schedule_locked
            || d.fusion_barrier
        {
            continue;
        }
        let Some(p) = compile::AttentionParams::from_words(&d.params) else {
            continue;
        };
        let Some(elements) = (p.kv_seq as usize)
            .checked_mul((p.packed_heads & 0xffff) as usize)
            .and_then(|n| n.checked_mul(p.head_dim as usize))
        else {
            continue;
        };
        if elements == 0 {
            continue;
        }
        let &[_, k, v] = d.input_buffers.as_slice() else {
            continue;
        };
        if k == v || protected.contains(&k) || protected.contains(&v) {
            continue;
        }
        let Some(ki) = writer(plan, k, reader, 1, elements) else {
            continue;
        };
        let Some(vi) = writer(plan, v, reader, 2, elements) else {
            continue;
        };
        if ki == vi {
            continue;
        }
        let (mut kd, mut vd) = (plan.dispatches[ki].clone(), plan.dispatches[vi].clone());
        if !producer_output(&mut kd) || !producer_output(&mut vd) {
            continue;
        }
        plan.dispatches[ki] = kd;
        plan.dispatches[vi] = vd;
        plan.buffers[k.0 as usize] /= 2;
        plan.buffers[v.0 as usize] /= 2;
        plan.dispatches[reader].shader = ShaderEntry::FlashAttentionProducts { kv_f16: true };
        count += 1;
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture(residual: bool) -> (Graph, ExecutionPlan) {
        let mut g = Graph::new();
        let q = g.input("q", &[64, 32]);
        let x = g.input("k", &[32, 32]);
        let w = g.parameter("w", &[32, 32]);
        let k = g.matmul(x, w);
        let x = g.input("v", &[32, 32]);
        let b = g.parameter("b", &[32]);
        let v = if residual {
            let norm = g.rms_norm(x, b, 1e-5);
            let r = g.input("residual", &[32, 32]);
            g.add(norm, r)
        } else {
            g.bias_add(x, b)
        };
        let y = g.multi_head_attn(q, k, v, 2, 2, 16, true);
        g.set_outputs(vec![y]);
        let mut p = compile::compile(&g);
        for d in &mut p.dispatches {
            if d.shader.is_attention() {
                d.shader = ShaderEntry::FlashAttentionProducts { kv_f16: false };
            }
            if d.shader == ShaderEntry::MatMul {
                d.kernel = Kernel::CooperativeTyped {
                    types: crate::codegen::MatrixTypes::F32_AS_F16,
                    splits: 1,
                };
            }
        }
        (g, p)
    }
    #[test]
    fn private_pairs_keep_interfaces_and_both_producers_atomic() {
        let (_, parent) = fixture(false);
        let reader = parent
            .dispatches
            .iter()
            .position(|d| d.shader.is_attention())
            .unwrap();
        let kv = [
            parent.dispatches[reader].input_buffers[1],
            parent.dispatches[reader].input_buffers[2],
        ];
        let mut changed = parent.clone();
        assert_eq!(fill(&mut changed), 1);
        assert_eq!(
            changed.dispatches[reader].shader,
            ShaderEntry::FlashAttentionProducts { kv_f16: true }
        );
        assert_eq!(fill(&mut changed), 0);
        let decoded: ExecutionPlan = ron::from_str(&ron::to_string(&changed).unwrap()).unwrap();
        assert_eq!(changed, decoded);
        let mut reverted = changed;
        for b in kv {
            reverted.buffers[b.0 as usize] *= 2;
            let i = parent
                .dispatches
                .iter()
                .position(|d| d.output_buffer == b)
                .unwrap();
            reverted.dispatches[i] = parent.dispatches[i].clone();
        }
        reverted.dispatches[reader] = parent.dispatches[reader].clone();
        assert_eq!(reverted, parent);
        let mut exposed = parent.clone();
        exposed.output_buffers.push(kv[1]);
        let saved = exposed.clone();
        assert_eq!(fill(&mut exposed), 0);
        assert_eq!(exposed, saved);
        let mut shared = parent.clone();
        let mut extra = parent.dispatches[reader].clone();
        extra.input_buffers[0] = kv[0];
        shared.dispatches.push(extra);
        let saved = shared.clone();
        assert_eq!(fill(&mut shared), 0);
        assert_eq!(shared, saved);
        let mut training = parent.clone();
        training.param_grad_pairs.push((kv[0], kv[1]));
        let saved = training.clone();
        assert_eq!(fill(&mut training), 0);
        assert_eq!(training, saved);
    }
    #[test]
    fn measured_frontier_retains_staged_parent_and_reaches_joint_storage() {
        use measure::Frontier as _;
        let (g, mut plan) = fixture(false);
        for d in &mut plan.dispatches {
            if d.shader.is_attention() {
                d.shader = ShaderEntry::FlashAttention;
            }
        }
        let caps = crate::codegen::CoopCaps {
            f16_attention_rows: 16,
            ..Default::default()
        };
        let mut frontier = implementations(
            vec![Seed {
                graph: std::sync::Arc::new(g),
                options: Default::default(),
                plan,
                description: "private KV fixture".into(),
                backward_rank: 0,
            }],
            caps,
            65536,
            512 << 20,
        );
        frontier.enable_forward_products(32, true);
        assert!(
            frontier
                .choices
                .iter()
                .all(|c| c.forward_products != ForwardProducts::HalfKv)
        );
        let base = frontier.next(None).unwrap();
        let staged = frontier.next(base.origin).unwrap();
        assert!(
            staged
                .plan
                .dispatches
                .iter()
                .any(|d| d.shader == (ShaderEntry::FlashAttentionProducts { kv_f16: false }))
        );
        frontier.observe(staged.origin.unwrap(), true);
        // Keep the original baseline as incumbent: a qualified loser must
        // not hide the joint storage alternative.
        let joint = frontier.next(base.origin).unwrap();
        assert!(
            joint
                .plan
                .dispatches
                .iter()
                .any(|d| d.shader == (ShaderEntry::FlashAttentionProducts { kv_f16: true }))
        );
        assert!(
            frontier
                .choices
                .iter()
                .any(|c| c.forward_products == ForwardProducts::Staged)
        );
        assert!(
            joint
                .plan
                .dispatches
                .iter()
                .any(|d| d.pointwise_storage.output == DType::F16)
        );
    }
    #[test]
    fn normalization_fusion_keeps_the_narrowed_store_boundary() {
        let (_, mut plan) = fixture(true);
        assert_eq!(fill(&mut plan), 1);
        let before = plan.clone();
        compile::fuse_rmsnorm_into_add(&mut plan);
        assert_eq!(plan, before);
    }

    #[test]
    fn inapplicable_storage_keeps_the_complete_frontier_unchanged() {
        use measure::Frontier as _;
        let mut graph = Graph::new();
        let x = graph.input("x", &[32, 32]);
        let w = graph.parameter("w", &[32, 32]);
        let q = graph.matmul(x, w);
        let k = graph.input("k", &[32, 32]);
        let v = graph.input("v", &[32, 32]);
        let y = graph.multi_head_attn(q, k, v, 1, 1, 32, false);
        graph.set_outputs(vec![y]);
        let graph = std::sync::Arc::new(graph);
        let plan = compile::compile(&graph);
        let caps = crate::codegen::CoopCaps {
            f16_attention_rows: 16,
            ..Default::default()
        };
        let mut device = blade_graphics::Capabilities {
            shader_float16: true,
            shader_bfloat16: true,
            subgroup_size: 32,
            ..Default::default()
        };
        device.cooperative_matrix.f16_f32_shapes.push([16, 16, 16]);
        device.cooperative_matrix.bf16_f32_shapes.push([16, 16, 16]);
        let frontier = |private_storage| {
            let seeds = (0..2)
                .map(|index| Seed {
                    graph: graph.clone(),
                    options: Default::default(),
                    plan: plan.clone(),
                    description: format!("graph={index}"),
                    backward_rank: 0,
                })
                .collect();
            let mut f = implementations(seeds, caps, 65536, 512 << 20);
            f.enable_matrix_inputs(&[DType::F16, DType::BF16], &device);
            f.enable_forward_products(32, private_storage);
            f
        };
        let (mut control, mut candidate) = (frontier(false), frontier(true));
        let mut incumbent = None;
        let mut programs = 0;
        loop {
            assert_eq!(control.choices, candidate.choices);
            let (a, b) = (control.next(incumbent), candidate.next(incumbent));
            assert_eq!(control.order.tried, candidate.order.tried);
            match (a, b) {
                (None, None) => break,
                (Some(a), Some(b)) => {
                    assert_eq!(a.plan, b.plan);
                    assert_eq!(a.description, b.description);
                    assert_eq!(a.origin, b.origin);
                    let qualified = programs % 4 != 0;
                    control.observe(a.origin.unwrap(), qualified);
                    candidate.observe(b.origin.unwrap(), qualified);
                    if qualified && programs % 3 == 0 {
                        incumbent = a.origin;
                    }
                    programs += 1;
                    assert!(programs < 1000, "frontier did not exhaust");
                }
                _ => panic!("storage changed frontier exhaustion"),
            }
        }
        assert!(programs > 10);
    }
}
