use super::{BufferRef, Dispatch, ExecutionPlan, Kernel, ShaderEntry};
use crate::{
    DType,
    codegen::PoolKernel,
    schedule::{PointwiseDAG, Pw, ReduceOp, ScalarStorage},
};
use std::collections::HashMap;

fn identity() -> PointwiseDAG {
    PointwiseDAG {
        n_inputs: 1,
        ops: vec![Pw::LoadInput(0)],
        output: 0,
    }
}

pub(super) fn spatial_sum(d: &Dispatch, planes: u32, pixels: u32) -> bool {
    let Some(k) = d.reduction() else {
        return false;
    };
    k.op == ReduceOp::Sum
        && k.prologue == identity()
        && k.extra_prologues.is_empty()
        && k.epilogue.is_none()
        && k.n_per_elem == 1
        && k.n_per_row == 0
        && k.workgroup_size == 256
        && k.rows_per_workgroup == 1
        && k.gather_elem.iter().all(|&x| !x)
        && k.input_row_repeats.iter().all(|&x| x == 1)
        && d.params[..2] == [planes, pixels]
        && d.scalar_storage == ScalarStorage::default()
        && d.extra_outputs.is_empty()
        && d.horizontal_batch <= 1
}

pub(super) fn fuse(plan: &mut ExecutionPlan) {
    loop {
        let protected = plan.externally_visible_buffers();
        let mut producers = HashMap::new();
        let mut readers = HashMap::<BufferRef, Vec<usize>>::new();
        for (i, d) in plan.dispatches.iter().enumerate() {
            producers.insert(d.output_buffer, i);
            for &b in &d.input_buffers {
                readers.entry(b).or_default().push(i);
            }
        }
        let Some((fi, bi)) = plan.dispatches.iter().enumerate().find_map(|(fi, f)| {
            if f.shader != ShaderEntry::MaxPool2d || !f.extra_outputs.is_empty() {
                return None;
            }
            plan.dispatches
                .iter()
                .enumerate()
                .skip(fi + 1)
                .find_map(|(bi, b)| {
                    (b.shader == ShaderEntry::MaxPool2dGrad
                        && b.params == f.params
                        && b.input_buffers[1] == f.input_buffers[0]
                        && b.extra_outputs.is_empty())
                    .then_some((fi, bi))
                })
        }) else {
            break;
        };
        let mut forward = plan.dispatches[fi].clone();
        let mut backward = plan.dispatches[bi].clone();
        let p = &forward.params;
        let planes = p[0].checked_mul(p[1]).expect("pool planes");
        let pixels = p[2].checked_mul(p[3]).expect("pool spatial extent");
        let elements = planes.checked_mul(pixels).expect("pool element count");
        let output_elements = planes
            .checked_mul(p[8])
            .and_then(|x| x.checked_mul(p[9]))
            .expect("pool outputs");
        assert!(p[8] != 0 && p[9] != 0);
        let source = forward.input_buffers[0];
        let mut prologue = identity();
        let mut types = vec![DType::F32];
        let mut remove = Vec::new();
        if let Some(&pi) = producers.get(&source) {
            let d = &plan.dispatches[pi];
            if let Some(dag) = d.pointwise() {
                if pi < fi
                    && !protected.contains(&source)
                    && readers[&source] == [fi, bi]
                    && d.extra_outputs.is_empty()
                    && d.params[0] == elements
                    && !d.scalar_storage.bf16_output
                    && (!d.fusion_barrier || d.scalar_storage != ScalarStorage::default())
                {
                    prologue = dag.clone();
                    types = (0..dag.n_inputs)
                        .map(|i| {
                            if d.scalar_storage.bf16_inputs & (1 << i) != 0 {
                                DType::BF16
                            } else {
                                DType::F32
                            }
                        })
                        .collect();
                    forward.input_buffers = d.input_buffers.clone();
                    forward.origin.extend_from_slice(&d.origin);
                    forward.requires_full_precision |= d.requires_full_precision;
                    remove.push(pi);
                }
            }
        }
        let tape = BufferRef(plan.buffers.len() as u32);
        plan.buffers.push(output_elements as usize * 4);
        forward.shader = ShaderEntry::Generated;
        forward.kernel = Kernel::Pool(PoolKernel::Forward {
            prologue,
            inputs: types,
        });
        forward.extra_outputs.push(tape);
        forward.fusion_barrier = true;
        forward.label = "max_pool_tape".into();

        let dense = backward.output_buffer;
        let uses = readers.get(&dense).map(Vec::as_slice).unwrap_or_default();
        let sum = uses
            .iter()
            .copied()
            .find(|&i| spatial_sum(&plan.dispatches[i], planes, pixels));
        let cast = uses.iter().copied().find(|&i| {
            let d = &plan.dispatches[i];
            d.shader == ShaderEntry::ToBF16 && d.params[0] == elements && d.extra_outputs.is_empty()
        });
        let mut output = DType::F32;
        if let Some(ci) = cast {
            if !protected.contains(&dense) && uses.len() == 1 + usize::from(sum.is_some()) {
                output = DType::BF16;
                backward.output_buffer = plan.dispatches[ci].output_buffer;
                backward
                    .origin
                    .extend_from_slice(&plan.dispatches[ci].origin);
                remove.push(ci);
            }
        }
        if let Some(si) = sum {
            backward
                .extra_outputs
                .push(plan.dispatches[si].output_buffer);
            backward
                .origin
                .extend_from_slice(&plan.dispatches[si].origin);
            backward.requires_full_precision |= plan.dispatches[si].requires_full_precision;
            backward.workgroups = [planes, 1, 1];
            remove.push(si);
        }
        backward.input_buffers[1] = tape;
        backward.shader = ShaderEntry::Generated;
        backward.kernel = Kernel::Pool(PoolKernel::Backward {
            output,
            sum: sum.is_some(),
        });
        backward.fusion_barrier = true;
        backward.label = "max_pool_tape_grad".into();
        plan.dispatches[fi] = forward;
        plan.dispatches[bi] = backward;
        remove.sort_unstable();
        remove.dedup();
        for i in remove.into_iter().rev() {
            plan.dispatches.remove(i);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Graph, TensorType, graph::Op};

    fn plan(protect_source: bool, protect_gradient: bool, backward: bool) -> ExecutionPlan {
        let mut g = Graph::new();
        let x = g.input("x", &[2 * 3 * 7 * 7]);
        let x = g.to_bf16(x);
        let x = g.from_bf16(x);
        let bias = g.parameter("bias", &[3]);
        let c = g.add_per_channel(x, bias, 3, 49);
        let y = g.max_pool_2d(c, 2, 3, 7, 7, 3, 3, 1, 1);
        let mut outputs = vec![y];
        if protect_source {
            outputs.push(c);
        }
        if backward {
            let dy = g.input("dy", &[2 * 3 * 7 * 7]);
            let dx = g.add_raw_node(
                Op::MaxPool2dGrad {
                    channels: 3,
                    in_h: 7,
                    in_w: 7,
                    kernel_h: 3,
                    kernel_w: 3,
                    stride: 1,
                    padding: 1,
                },
                vec![dy, c],
                TensorType::f32(vec![2 * 3 * 7 * 7]),
            );
            let rounded = g.to_bf16(dx);
            let rounded = g.from_bf16(rounded);
            let shaped = g.reshape(dx, &[6, 49]);
            let db = g.sum_inner(shaped);
            outputs.extend([rounded, db]);
            if protect_gradient {
                outputs.push(dx);
            }
        }
        g.set_outputs(outputs);
        let g = crate::optimize::optimize(&g);
        super::super::compile(&g)
    }

    #[test]
    fn spatial_tape_consumer_contracts() {
        let p = plan(false, false, true);
        let f = p
            .dispatches
            .iter()
            .find(|d| matches!(d.kernel, Kernel::Pool(PoolKernel::Forward { .. })))
            .unwrap();
        let b = p
            .dispatches
            .iter()
            .find(|d| matches!(d.kernel, Kernel::Pool(PoolKernel::Backward { .. })))
            .unwrap();
        assert!(
            matches!(&f.kernel, Kernel::Pool(PoolKernel::Forward { inputs, .. }) if inputs == &[DType::BF16, DType::F32])
        );
        assert_eq!(
            b.kernel,
            Kernel::Pool(PoolKernel::Backward {
                output: DType::BF16,
                sum: true
            })
        );
        assert_eq!(b.input_buffers[1], f.extra_outputs[0]);
        assert_eq!(b.extra_outputs.len(), 1);
        assert_eq!(b.workgroups, [6, 1, 1]);
        let protected = plan(true, true, true);
        let f = protected
            .dispatches
            .iter()
            .find(|d| matches!(d.kernel, Kernel::Pool(PoolKernel::Forward { .. })))
            .unwrap();
        let b = protected
            .dispatches
            .iter()
            .find(|d| matches!(d.kernel, Kernel::Pool(PoolKernel::Backward { .. })))
            .unwrap();
        assert!(
            matches!(&f.kernel, Kernel::Pool(PoolKernel::Forward { inputs, prologue }) if inputs == &[DType::F32] && *prologue == identity())
        );
        assert_eq!(
            b.kernel,
            Kernel::Pool(PoolKernel::Backward {
                output: DType::F32,
                sum: true
            })
        );
        assert_eq!(protected.output_buffers[1], f.input_buffers[0]);
        assert!(protected.output_buffers.contains(&b.output_buffer));
        assert!(
            !plan(false, false, false)
                .dispatches
                .iter()
                .any(|d| matches!(d.kernel, Kernel::Pool(_)))
        );
    }
}
