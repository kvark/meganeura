//! Compose row-local scalar work without crossing a stored rounding boundary.

use super::{BufferRef, Dispatch, ExecutionPlan, Kernel, ShaderEntry};
use crate::schedule::ScalarStorage;
use std::collections::HashMap;

fn movable(dispatch: &Dispatch) -> bool {
    !dispatch.fusion_barrier || dispatch.scalar_storage != ScalarStorage::default()
}

fn reads(dispatch: &Dispatch) -> impl Iterator<Item = BufferRef> + '_ {
    dispatch
        .input_buffers
        .iter()
        .copied()
        .chain(
            dispatch
                .matmul_epilogue
                .iter()
                .flat_map(|e| e.inputs.iter().map(|v| v.0)),
        )
        .chain(
            dispatch
                .matmul_prologue
                .iter()
                .flat_map(|p| p.factors.iter().map(|v| v.0)),
        )
        .chain(dispatch.gemv_rmsnorm.iter().map(|n| n.weight))
}

fn writes(dispatch: &Dispatch) -> impl Iterator<Item = BufferRef> + '_ {
    std::iter::once(dispatch.output_buffer).chain(dispatch.extra_outputs.iter().copied())
}

fn plain_copy(dispatch: &Dispatch) -> bool {
    dispatch.kernel == Kernel::Default
        && !dispatch.fusion_barrier
        && dispatch.scalar_storage == ScalarStorage::default()
        && dispatch.input_buffers.len() == 1
        && dispatch.extra_outputs.is_empty()
        && dispatch.matmul_epilogue.is_none()
        && dispatch.matmul_prologue.is_none()
        && dispatch.gemv_rmsnorm.is_none()
        && dispatch.horizontal_batch <= 1
}

/// Compose a repeated-row view with scalar arithmetic, including typed stores.
/// No arithmetic or stored rounding boundary is moved or removed.
pub(super) fn fold_repeated_inputs(plan: &mut ExecutionPlan) {
    loop {
        let protected = plan.externally_visible_buffers();
        let mut producers = HashMap::new();
        let mut uses = HashMap::<BufferRef, usize>::new();
        for (i, d) in plan.dispatches.iter().enumerate() {
            for output in writes(d) {
                producers.insert(output, i);
            }
            for input in reads(d) {
                *uses.entry(input).or_default() += 1;
            }
        }
        let mut folded = false;
        'next: for ci in 0..plan.dispatches.len() {
            let c = &plan.dispatches[ci];
            let Some(kernel) = c.reduction() else {
                continue;
            };
            if !movable(c) || kernel.gather_elem.iter().any(|&g| g) {
                continue;
            }
            let outer = c.params[0];
            let inner = c.params[1];
            let per_elem = kernel.n_per_elem as usize;
            for slot in 0..per_elem {
                if kernel.input_row_repeats.get(slot).copied().unwrap_or(1) != 1
                    || c.scalar_storage.bf16_inputs & (1 << slot) != 0
                    || kernel.prologue.broadcasts_input(slot as u8)
                    || kernel
                        .extra_prologues
                        .iter()
                        .any(|p| p.broadcasts_input(slot as u8))
                    || kernel
                        .epilogue
                        .as_ref()
                        .is_some_and(|e| e.dag.broadcasts_input(slot as u8))
                {
                    continue;
                }
                let buffer = c.input_buffers[slot];
                let Some(&ti) = producers.get(&buffer) else {
                    continue;
                };
                let t = &plan.dispatches[ti];
                if ti >= ci
                    || !plain_copy(t)
                    || t.shader != ShaderEntry::Transpose
                    || t.params.len() != 4
                    || t.params[0] != inner
                    || t.params[2..] != [0, 0]
                {
                    continue;
                }
                let repeats = t.params[1];
                let batch = t.workgroups[2];
                if repeats <= 1
                    || batch == 0
                    || inner == 0
                    || batch.checked_mul(repeats) != Some(outer)
                {
                    continue;
                }
                let expanded = t.input_buffers[0];
                let Some(&bi) = producers.get(&expanded) else {
                    continue;
                };
                let b = &plan.dispatches[bi];
                if bi >= ti
                    || !plain_copy(b)
                    || !b.is_inner_broadcast()
                    || b.params[1] != repeats
                    || outer.checked_mul(inner) != b.params.first().copied()
                {
                    continue;
                }
                let source = b.input_buffers[0];
                let bytes = u64::from(batch) * u64::from(inner) * 4;
                if (plan.buffers[source.0 as usize] as u64) < bytes
                    || plan.dispatches[bi..=ci]
                        .iter()
                        .any(|d| writes(d).any(|w| w == source))
                {
                    continue;
                }
                let origin = t
                    .origin
                    .iter()
                    .chain(&b.origin)
                    .copied()
                    .collect::<Vec<_>>();
                let c = &mut plan.dispatches[ci];
                let kernel = c.reduction_mut().unwrap();
                kernel.input_row_repeats.resize(per_elem, 1);
                kernel.input_row_repeats[slot] = repeats;
                c.input_buffers[slot] = source;
                c.origin.extend(origin);
                if uses.get(&buffer) == Some(&1) && !protected.contains(&buffer) {
                    plan.dispatches.remove(ti);
                    if uses.get(&expanded) == Some(&1) && !protected.contains(&expanded) {
                        plan.dispatches.remove(bi);
                    }
                }
                folded = true;
                break 'next;
            }
        }
        if !folded {
            break;
        }
    }
}

pub(super) fn fuse(plan: &mut ExecutionPlan) {
    loop {
        let protected = plan.externally_visible_buffers();
        let mut producers = HashMap::<BufferRef, usize>::new();
        let mut reads = HashMap::<BufferRef, usize>::new();
        for (i, dispatch) in plan.dispatches.iter().enumerate() {
            producers.insert(dispatch.output_buffer, i);
            for input in &dispatch.input_buffers {
                *reads.entry(*input).or_default() += 1;
            }
        }
        let mut remove = None;
        'next: for ci in 0..plan.dispatches.len() {
            let consumer = &plan.dispatches[ci];
            if !movable(consumer) {
                continue;
            }
            if let Some(kernel) = consumer.reduction() {
                if kernel.gather_elem.iter().any(|&v| v)
                    || kernel.input_row_repeats.iter().any(|&v| v != 1)
                {
                    continue;
                }
                let elements = consumer.params[0].checked_mul(consumer.params[1]);
                for slot in 0..kernel.n_per_elem as usize {
                    let buffer = consumer.input_buffers[slot];
                    if protected.contains(&buffer) || reads.get(&buffer) != Some(&1) {
                        continue;
                    }
                    let Some(&pi) = producers.get(&buffer) else {
                        continue;
                    };
                    let producer = &plan.dispatches[pi];
                    let Some(dag) = producer.pointwise() else {
                        continue;
                    };
                    let streams = kernel.n_per_elem as usize + dag.n_inputs as usize - 1;
                    if pi >= ci
                        || !movable(producer)
                        || dag.n_inputs == 0
                        || streams > 3
                        || producer.params.first().copied() != elements
                        || producer.scalar_storage.bf16_output
                        || consumer.scalar_storage.bf16_inputs & (1 << slot) != 0
                        || kernel.prologue.broadcasts_input(slot as u8)
                        || kernel
                            .extra_prologues
                            .iter()
                            .any(|p| p.broadcasts_input(slot as u8))
                        || kernel
                            .epilogue
                            .as_ref()
                            .is_some_and(|e| e.dag.broadcasts_input(slot as u8))
                    {
                        continue;
                    }
                    let producer = producer.clone();
                    let dag = producer.pointwise().unwrap();
                    let consumer = &mut plan.dispatches[ci];
                    let old_storage = consumer.scalar_storage;
                    let mut inputs = producer.input_buffers.clone();
                    let mut storage = ScalarStorage {
                        bf16_inputs: producer.scalar_storage.bf16_inputs,
                        bf16_output: old_storage.bf16_output,
                    };
                    for (i, &input) in consumer.input_buffers.iter().enumerate() {
                        if i != slot {
                            if old_storage.bf16_inputs & (1 << i) != 0 {
                                storage.bf16_inputs |= 1 << inputs.len();
                            }
                            inputs.push(input);
                        }
                    }
                    let kernel = consumer.reduction_mut().unwrap();
                    kernel.prologue = kernel.prologue.fuse_input(slot as u8, dag);
                    for extra in &mut kernel.extra_prologues {
                        *extra = extra.fuse_input(slot as u8, dag);
                    }
                    if let Some(epilogue) = kernel.epilogue.as_mut() {
                        epilogue.dag = epilogue.dag.fuse_input(slot as u8, dag);
                    }
                    kernel.n_per_elem = streams as u8;
                    kernel.gather_elem.clear();
                    kernel.input_row_repeats.clear();
                    consumer.input_buffers = inputs;
                    consumer.scalar_storage = storage;
                    consumer.requires_full_precision |= producer.requires_full_precision;
                    consumer.origin.extend(producer.origin);
                    consumer.fusion_barrier |= storage != ScalarStorage::default();
                    remove = Some(pi);
                    break 'next;
                }
            } else if let Some(dag) = consumer.pointwise() {
                if dag.n_inputs != 1
                    || dag.has_broadcast()
                    || consumer.scalar_storage.bf16_inputs != 0
                {
                    continue;
                }
                let buffer = consumer.input_buffers[0];
                if protected.contains(&buffer) || reads.get(&buffer) != Some(&1) {
                    continue;
                }
                let Some(&pi) = producers.get(&buffer) else {
                    continue;
                };
                let producer = &plan.dispatches[pi];
                if pi >= ci
                    || !movable(producer)
                    || producer.scalar_storage.bf16_output
                    || producer.reduction().is_none_or(|k| k.epilogue.is_none())
                    || producer.params[0].checked_mul(producer.params[1])
                        != consumer.params.first().copied()
                {
                    continue;
                }
                let consumer = consumer.clone();
                let producer = &mut plan.dispatches[pi];
                let epilogue = producer.reduction_mut().unwrap().epilogue.as_mut().unwrap();
                epilogue.dag = consumer.pointwise().unwrap().fuse_input(0, &epilogue.dag);
                producer.output_buffer = consumer.output_buffer;
                producer.scalar_storage.bf16_output = consumer.scalar_storage.bf16_output;
                producer.requires_full_precision |= consumer.requires_full_precision;
                producer.fusion_barrier |= producer.scalar_storage != ScalarStorage::default();
                producer.origin.extend(consumer.origin);
                remove = Some(ci);
                break 'next;
            }
        }
        match remove {
            Some(index) => {
                plan.dispatches.remove(index);
            }
            None => break,
        }
    }
}

#[cfg(test)]
mod repeated_tests {
    use super::*;
    use crate::{
        CompileOptions, Graph,
        compile::{
            self, EpilogueLoadKind, GemvRmsNorm, MatMulEpilogue, MatMulPrologue, PrologueLoadKind,
        },
        schedule::{PointwiseDAG, Pw},
    };

    fn plan() -> ExecutionPlan {
        let mut g = Graph::new();
        let x = g.input("x", &[3, 19]);
        let flat = g.reshape(x, &[57, 1]);
        let expanded = g.broadcast_inner(flat, 5);
        let shaped = g.reshape(expanded, &[3, 19, 5]);
        let transposed = g.transpose(shaped);
        let rows = g.reshape(transposed, &[15, 19]);
        let output = g.sum_inner(rows);
        g.set_outputs(vec![output]);
        let mut p = compile::compile_with(
            &g,
            &CompileOptions {
                fuse_dispatches: false,
                ..Default::default()
            },
        );
        assert_eq!(p.dispatches.len(), 3);
        p.dispatches[2].scalar_storage.bf16_output = true;
        p.dispatches[2].fusion_barrier = true;
        p
    }

    #[test]
    fn typed_and_visible_views() {
        for visible in 0..3 {
            let mut p = plan();
            let original = p.dispatches[2].clone();
            if visible > 0 {
                p.output_buffers
                    .push(p.dispatches[visible - 1].output_buffer);
            }
            fold_repeated_inputs(&mut p);
            let d = p.dispatches.last().unwrap();
            assert_eq!(d.reduction().unwrap().input_row_repeats, [5]);
            assert_eq!(d.scalar_storage, original.scalar_storage);
            assert_eq!(d.fusion_barrier, original.fusion_barrier);
            assert_eq!(d.params, original.params);
            assert_eq!(d.workgroups, original.workgroups);
            assert_eq!(p.dispatches.len(), visible + 1);
        }
    }

    #[test]
    fn unsafe_views_remain_materialized() {
        for case in 0..10 {
            let mut p = plan();
            match case {
                0 => p.dispatches[0].fusion_barrier = true,
                1 => p.dispatches[1].fusion_barrier = true,
                2 => p.dispatches[2].scalar_storage = ScalarStorage::default(),
                3 => p.dispatches[2].scalar_storage.bf16_inputs = 1,
                4 => p.dispatches[0].scalar_storage.bf16_output = true,
                5 => p.dispatches[2].reduction_mut().unwrap().gather_elem = vec![true],
                6 => {
                    p.dispatches[2].reduction_mut().unwrap().prologue.ops[0] = Pw::LoadBroadcast {
                        input: 0,
                        divisor: 1,
                        modulus: 19,
                    }
                }
                7 => {
                    let mut overwrite = p.dispatches[0].clone();
                    overwrite.output_buffer = p.dispatches[0].input_buffers[0];
                    overwrite.extra_outputs.clear();
                    p.dispatches.insert(2, overwrite);
                }
                8 => p.dispatches[1].workgroups[2] = 4,
                9 => {
                    let source = p.dispatches[0].input_buffers[0];
                    p.buffers[source.0 as usize] = 56 * 4;
                }
                _ => unreachable!(),
            }
            let before = p.dispatches.clone();
            fold_repeated_inputs(&mut p);
            assert_eq!(p.dispatches, before, "case {case}");
        }
    }

    #[test]
    fn metadata_consumers_keep_their_copy() {
        for kind in 0..4 {
            let mut p = plan();
            let copy = p.dispatches[1].output_buffer;
            let extra = BufferRef(p.buffers.len() as u32);
            p.buffers.push(4);
            let mut d = Dispatch {
                output_buffer: extra,
                ..Default::default()
            };
            match kind {
                0 => d.input_buffers.push(copy),
                1 => {
                    d.matmul_epilogue = Some(MatMulEpilogue {
                        dag: PointwiseDAG {
                            n_inputs: 2,
                            ops: vec![Pw::LoadInput(1)],
                            output: 0,
                        },
                        inputs: vec![(copy, EpilogueLoadKind::PerElement)],
                    })
                }
                2 => {
                    d.matmul_prologue = Some(MatMulPrologue {
                        factors: vec![(copy, PrologueLoadKind::PerRow)],
                    })
                }
                3 => {
                    d.gemv_rmsnorm = Some(GemvRmsNorm {
                        weight: copy,
                        eps_bits: 0,
                    })
                }
                _ => unreachable!(),
            }
            p.dispatches.push(d);
            fold_repeated_inputs(&mut p);
            assert_eq!(p.dispatches.len(), 4, "metadata {kind}");
            assert_eq!(p.dispatches[2].reduction().unwrap().input_row_repeats, [5]);
        }
    }
}
