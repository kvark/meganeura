//! Narrow private producers whose readers already require one reduced format.
//! Multiplication types are already chosen; this pass does not choose precision.
//! A derivative reader may use a reduced binding when its multiplication was
//! already selected in that format. Derivative producers and explicit pins
//! retain F32 stores.

use crate::{
    DType,
    compile::{ExecutionPlan, Kernel, ShaderEntry},
};
use std::collections::{HashMap, HashSet};

pub(super) fn fill(plan: &mut ExecutionPlan) -> usize {
    let mut writers = HashMap::new();
    let mut reads = HashMap::<_, usize>::new();
    let mut requested = HashMap::<_, (usize, usize)>::new();
    for (index, d) in plan.dispatches.iter().enumerate() {
        for buffer in std::iter::once(d.output_buffer).chain(d.extra_outputs.iter().copied()) {
            writers
                .entry(buffer)
                .and_modify(|writer| *writer = None)
                .or_insert(Some(index));
        }
        for &buffer in &d.input_buffers {
            *reads.entry(buffer).or_default() += 1;
        }
        // Fused auxiliary bindings can observe an F32 producer too.
        for buffer in d
            .matmul_epilogue
            .iter()
            .flat_map(|e| e.inputs.iter().map(|x| x.0))
            .chain(
                d.matmul_prologue
                    .iter()
                    .flat_map(|p| p.factors.iter().map(|x| x.0)),
            )
            .chain(d.gemv_rmsnorm.iter().map(|n| n.weight))
        {
            *reads.entry(buffer).or_default() += 1;
        }
        let types = match d.kernel {
            Kernel::CooperativeTyped { types, .. }
            | Kernel::CooperativeWorkgroup {
                types: Some(types), ..
            } => types,
            _ => continue,
        };
        if !d.shader.is_matmul()
            || d.requires_full_precision
            || d.matmul_prologue.is_some()
            || d.matmul_epilogue.is_some()
        {
            continue;
        }
        for (&buffer, storage) in d.input_buffers.iter().zip([types.a, types.b]) {
            if storage == DType::F32 && matches!(types.multiply, DType::F16 | DType::BF16) {
                let request = requested
                    .entry((buffer, types.multiply))
                    .or_insert((index, 0));
                request.0 = request.0.min(index);
                request.1 += 1;
            }
        }
    }
    let protected = plan.externally_visible_buffers();
    let lse: HashSet<_> = plan.lse_buffers.iter().map(|entry| entry.1).collect();
    let mut requests: Vec<_> = requested.into_iter().collect();
    // Keep the transformation order reproducible for inspection.
    requests.sort_by_key(|&((buffer, dtype), (first, _))| (first, buffer.0, dtype == DType::BF16));
    let mut conversions = HashSet::new();
    for ((source, dtype), (first, count)) in requests {
        let bytes = plan.buffers[source.0 as usize];
        if bytes == 0 || !bytes.is_multiple_of(4) {
            continue;
        }
        let Some(Some(index)) = writers.get(&source).copied() else {
            continue;
        };
        if index >= first {
            continue;
        }
        let eligible = {
            let d = &plan.dispatches[index];
            // The primary attention output may narrow independently of LSE.
            // Its readers already round to F16; softmax/statistics remain F32.
            let attention = dtype == DType::F16
                && matches!(
                    d.shader,
                    ShaderEntry::FlashAttentionCoop | ShaderEntry::FlashAttentionProducts
                )
                && d.extra_outputs.iter().all(|buffer| lse.contains(buffer));
            !protected.contains(&source)
                && reads[&source] == count
                && !d.full_precision()
                && (d.extra_outputs.is_empty() || attention)
                && d.matmul_prologue.is_none()
                && d.matmul_epilogue.is_none()
                && (matches!(d.kernel,
                    Kernel::CooperativeTyped { types, splits: 1 }
                    | Kernel::CooperativeWorkgroup { types: Some(types), splits: 1, .. }
                    if types.output == DType::F32)
                    && d.shader.is_matmul()
                    || ((d.pointwise().is_some()
                        || d.reduction().is_some_and(|r| r.epilogue.is_some())
                        || matches!(
                            d.shader,
                            ShaderEntry::SwiGLUConcat | ShaderEntry::GeGLUConcat
                        )
                        || attention)
                        && !d.fusion_barrier
                        && !d.pointwise_storage.bf16_output
                        && !d.pointwise_storage.f16_output))
        };
        if !eligible {
            continue;
        }
        let d = &mut plan.dispatches[index];
        if d.pointwise().is_some()
            || d.reduction().is_some()
            || matches!(
                d.shader,
                ShaderEntry::SwiGLUConcat
                    | ShaderEntry::GeGLUConcat
                    | ShaderEntry::FlashAttentionCoop
                    | ShaderEntry::FlashAttentionProducts
            )
        {
            d.pointwise_storage.f16_output = dtype == DType::F16;
            d.pointwise_storage.bf16_output = dtype == DType::BF16;
        } else {
            let types = match d.kernel {
                Kernel::CooperativeTyped { ref mut types, .. }
                | Kernel::CooperativeWorkgroup {
                    types: Some(ref mut types),
                    ..
                } => types,
                _ => unreachable!(),
            };
            types.output = dtype;
        }
        plan.buffers[source.0 as usize] = bytes / 2;
        conversions.insert((source, dtype));
    }
    for d in &mut plan.dispatches {
        let types = match d.kernel {
            Kernel::CooperativeTyped { ref mut types, .. }
            | Kernel::CooperativeWorkgroup {
                types: Some(ref mut types),
                ..
            } => types,
            _ => continue,
        };
        if !d.shader.is_matmul()
            || d.requires_full_precision
            || d.matmul_prologue.is_some()
            || d.matmul_epilogue.is_some()
        {
            continue;
        }
        for (slot, storage) in [&mut types.a, &mut types.b].into_iter().enumerate() {
            if conversions.contains(&(d.input_buffers[slot], types.multiply)) {
                *storage = types.multiply;
            }
        }
    }
    if !conversions.is_empty() {
        plan.groups.clear(); // Session construction schedules the revised plan.
    }
    conversions.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Graph, Session,
        compile::{Dispatch, ShaderEntry},
    };

    fn chain(training: bool, pointwise: u8) -> Graph {
        let mut g = Graph::new();
        let x = g.input("x", &[17, 23]);
        let width = if pointwise >= 3 { 38 } else { 19 };
        let a = g.parameter("a", &[23, width]);
        let b = g.parameter("b", &[19, 7]);
        let first = g.matmul(x, a);
        let first = match pointwise {
            1 => g.relu(first),
            2 => {
                let weight = g.parameter("norm", &[19]);
                g.rms_norm(first, weight, 1e-5)
            }
            3 => g.swiglu_concat(first),
            4 => g.geglu_concat(first),
            _ => first,
        };
        let last = g.matmul(first, b);
        if training {
            let target = g.input("target", &[17, 7]);
            let loss = g.mse_loss(last, target);
            g.set_outputs(vec![loss]);
            crate::autodiff::differentiate(&g)
        } else {
            g.set_outputs(vec![last]);
            g
        }
    }

    fn plan(g: &Graph, dtype: DType) -> ExecutionPlan {
        let mut p = crate::compile::compile_with(
            g,
            &crate::compile::CompileOptions {
                fuse_dispatches: false,
                ..Default::default()
            },
        );
        super::super::fill_matrix_candidates(&mut p, dtype, false);
        p
    }

    #[test]
    fn attention_store_narrows_only_its_private_f16_output() {
        for protection in 0..5 {
            let mut g = Graph::new();
            let q = g.input("q", &[33, 144]);
            let k = g.input("k", &[19, 48]);
            let v = g.input("v", &[19, 48]);
            let attention = g.multi_head_attn(q, k, v, 3, 1, 48, true);
            let w = g.parameter("w", &[144, 32]);
            let output = g.matmul(attention, w);
            let mut outputs = vec![output];
            if protection == 1 {
                outputs.push(g.neg(attention)); // A reader still needs F32.
            } else if protection == 2 {
                outputs.push(attention); // An externally observed value.
            }
            g.set_outputs(outputs);
            let mut p = plan(&g, DType::F16);
            super::super::fill_attention_products(
                &mut p,
                crate::codegen::CoopCaps {
                    f16_attention_rows: 8,
                    ..Default::default()
                },
                32 << 10,
            );
            let index = p
                .dispatches
                .iter()
                .position(|d| d.shader == ShaderEntry::FlashAttentionProducts)
                .unwrap();
            p.dispatches[index].requires_full_precision = protection == 3;
            if protection == 4 {
                p.lse_buffers.clear(); // Unrecognized auxiliary output.
            }
            let buffer = p.dispatches[index].output_buffer;
            let before = p.clone();
            assert_eq!(fill(&mut p), usize::from(protection == 0));
            if protection != 0 {
                assert_eq!(p, before);
                continue;
            }
            assert!(p.dispatches[index].pointwise_storage.f16_output);
            assert_eq!(
                p.buffers[buffer.0 as usize],
                before.buffers[buffer.0 as usize] / 2
            );
            assert_eq!(
                p.dispatches[index].extra_outputs,
                before.dispatches[index].extra_outputs
            );
            for &extra in &p.dispatches[index].extra_outputs {
                assert_eq!(
                    p.buffers[extra.0 as usize],
                    before.buffers[extra.0 as usize]
                );
            }
            let reader = p.dispatches.iter().find(|d| d.shader.is_matmul()).unwrap();
            assert!(
                matches!(reader.kernel, Kernel::CooperativeTyped { types, .. } if types.a == DType::F16)
            );
            let path = std::env::temp_dir().join(format!(
                "meganeura-attention-half-store-{}.ron",
                std::process::id()
            ));
            crate::cache::save_plan(&p, &g, &path).unwrap();
            assert_eq!(crate::cache::load_plan(&g, &path).unwrap().unwrap(), p);
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    #[ignore = "GPU qualification of private attention stores and LSE"]
    fn attention_half_store_preserves_f16_readers_and_lse() {
        let gpu = crate::reference::gpu::shared_context();
        let caps = gpu.capabilities();
        let Some(rows) = caps
            .cooperative_matrix
            .f16_f32_shapes
            .iter()
            .find_map(|&[m, n, k]| (matches!(m, 8 | 16) && n == 16 && k == 16).then_some(m))
        else {
            return;
        };
        for (queries, keys, heads, kv_heads, dim, window) in
            [(33, 19, 3, 1, 48, 7), (16, 16, 2, 2, 64, 0)]
        {
            let mut g = Graph::new();
            let q = g.input("q", &[queries, heads * dim]);
            let k = g.input("k", &[keys, kv_heads * dim]);
            let v = g.input("v", &[keys, kv_heads * dim]);
            let y = g.multi_head_attn(
                q,
                k,
                v,
                heads as u32,
                kv_heads as u32,
                dim as u32,
                window != 0,
            );
            let w = g.parameter("w", &[heads * dim, 17]);
            let output = g.matmul(y, w);
            g.set_outputs(vec![output]);
            let mut baseline = plan(&g, DType::F16);
            super::super::fill_attention_products(
                &mut baseline,
                crate::codegen::CoopCaps {
                    f16_attention_rows: rows,
                    ..Default::default()
                },
                caps.max_compute_shared_memory_size,
            );
            let attention = baseline
                .dispatches
                .iter_mut()
                .find(|d| d.shader == ShaderEntry::FlashAttentionProducts)
                .unwrap();
            attention.params[4] = window;
            let lse = attention.extra_outputs[0];
            let mut half = baseline.clone();
            assert_eq!(fill(&mut half), 1);
            let options = crate::SessionOptions {
                coop: crate::CoopPolicy::Disabled,
                no_alias: true,
                poison: true,
                ..Default::default()
            };
            let mut baseline = Session::with_context_opts(baseline, gpu.clone(), options.clone());
            let mut half = Session::with_context_opts(half, gpu.clone(), options);
            let values = |n: usize, scale: f32| {
                (0..n)
                    .map(|i| (i as f32 * 0.17).sin() * scale)
                    .collect::<Vec<_>>()
            };
            for session in [&mut baseline, &mut half] {
                session.set_input("q", &values(queries * heads * dim, 0.7));
                session.set_input("k", &values(keys * kv_heads * dim, 0.4));
                session.set_input("v", &values(keys * kv_heads * dim, 0.9));
                session.set_parameter("w", &values(heads * dim * 17, 0.03));
                session.step();
                session.wait();
            }
            assert_eq!(
                baseline.read_output(queries * 17),
                half.read_output(queries * 17)
            );
            let count = baseline.plan().buffers[lse.0 as usize] / 4;
            let (mut original, mut narrowed) = (vec![0.0; count], vec![0.0; count]);
            baseline.read_buffer(lse, &mut original);
            half.read_buffer(lse, &mut narrowed);
            assert_eq!(
                original.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                narrowed.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn storage_narrows_private_producers_and_preserves_f32_readers() {
        for training in [false, true] {
            let mut p = plan(&chain(training, 0), DType::F16);
            let external = p.externally_visible_buffers();
            let sizes: Vec<_> = external
                .iter()
                .map(|b| (*b, p.buffers[b.0 as usize]))
                .collect();
            let first = p
                .dispatches
                .iter()
                .find(|d| d.shader.is_matmul())
                .unwrap()
                .output_buffer;
            let count = fill(&mut p);
            assert_eq!(count > 0, !training);
            let producer = p.dispatches.iter().find(|d| d.shader.is_matmul()).unwrap();
            assert_eq!(producer.output_buffer, first);
            assert!(
                matches!(producer.kernel, Kernel::CooperativeTyped { types, .. }
                if types.output == if training { DType::F32 } else { DType::F16 })
            );
            for (buffer, bytes) in sizes {
                assert_eq!(p.buffers[buffer.0 as usize], bytes);
            }
            let once = p.clone();
            assert_eq!(fill(&mut p), 0);
            assert_eq!(p, once);
        }

        let mut g = Graph::new();
        let x = g.input("x", &[32, 32]);
        let w = g.parameter("w", &[32, 32]);
        let x = g.relu(x);
        let a = g.matmul(x, w);
        let b = g.matmul_at(x, w);
        let y = g.add(a, b);
        g.set_outputs(vec![y]);
        let mut p = plan(&g, DType::BF16);
        let dispatches = p.dispatches.len();
        assert_eq!(fill(&mut p), 1);
        assert_eq!(p.dispatches.len(), dispatches);
        assert!(p.dispatches.iter().any(|d| d.pointwise_storage.bf16_output));
        let matrices: Vec<_> = p
            .dispatches
            .iter()
            .filter(|d| d.shader.is_matmul())
            .collect();
        assert_eq!(matrices[0].input_buffers, matrices[1].input_buffers);
    }

    #[test]
    fn multiple_writers_keep_values_in_f32() {
        let mut p = plan(&chain(false, 0), DType::BF16);
        let first = p
            .dispatches
            .iter()
            .find(|d| d.shader.is_matmul())
            .unwrap()
            .output_buffer;
        p.dispatches.insert(
            0,
            Dispatch {
                shader: ShaderEntry::Generated,
                output_buffer: first,
                ..Default::default()
            },
        );
        assert_eq!(fill(&mut p), 0);
        assert!(
            matches!(p.dispatches[1].kernel, Kernel::CooperativeTyped { types, .. }
            if types.output == DType::F32)
        );
    }

    #[test]
    fn reduced_derivative_readers_share_private_stores_without_narrowing_gradients() {
        for dtype in [DType::F16, DType::BF16] {
            let mut p = plan(&chain(true, 3), dtype);
            assert!(super::super::fill_gradient_candidates(&mut p, dtype, false) > 0);
            let before = p.clone();
            assert!(fill(&mut p) > 0);
            assert_eq!(fill(&mut p), 0, "storage refinement is idempotent");
            for buffer in before.externally_visible_buffers() {
                assert_eq!(
                    p.buffers[buffer.0 as usize],
                    before.buffers[buffer.0 as usize]
                );
            }
            assert!(p.dispatches.iter().any(|d| {
                d.is_derivative
                    && matches!(d.kernel, Kernel::CooperativeTyped { types, .. }
                    if types.a == dtype || types.b == dtype)
            }));
            for (old, new) in before.dispatches.iter().zip(&p.dispatches) {
                assert_eq!(old.is_derivative, new.is_derivative);
                assert_eq!(old.requires_full_precision, new.requires_full_precision);
                if old.is_derivative {
                    assert_eq!(
                        p.buffers[old.output_buffer.0 as usize],
                        before.buffers[old.output_buffer.0 as usize]
                    );
                }
            }
        }
    }

    #[test]
    fn mixed_precision_or_pinned_derivative_readers_prevent_narrowing() {
        let mut mixed = plan(&chain(true, 3), DType::F16);
        super::super::fill_gradient_candidates(&mut mixed, DType::BF16, false);
        assert_eq!(fill(&mut mixed), 0);

        let mut pinned = plan(&chain(true, 3), DType::F16);
        super::super::fill_gradient_candidates(&mut pinned, DType::F16, false);
        for d in &mut pinned.dispatches {
            if d.is_derivative {
                d.requires_full_precision = true;
            }
        }
        assert_eq!(fill(&mut pinned), 0);
    }

    #[test]
    fn native_storage_keeps_chain_outputs_gradients_and_parameter_updates() {
        let gpu = crate::reference::gpu::shared_context();
        for dtype in [DType::F16, DType::BF16] {
            if !crate::runtime::typed_matrix_supported(&gpu, dtype) {
                continue;
            }
            for (training, pointwise, reduced_derivatives) in [
                (false, 0, false),
                (false, 1, false),
                (false, 2, false),
                (false, 3, false),
                (false, 4, false),
                (true, 0, false),
                (true, 3, false),
                (true, 0, true),
                (true, 3, true),
            ] {
                let g = chain(training, pointwise);
                let mut staged = plan(&g, dtype);
                if reduced_derivatives {
                    assert!(super::super::fill_gradient_candidates(&mut staged, dtype, false) > 0);
                }
                let mut native = staged.clone();
                assert_eq!(
                    fill(&mut native) > 0,
                    !training || reduced_derivatives,
                    "{dtype:?}, training={training}, pointwise={pointwise}"
                );
                let reads = staged
                    .output_buffers
                    .iter()
                    .copied()
                    .chain(staged.param_grad_pairs.iter().map(|x| x.1))
                    .collect::<Vec<_>>();
                let mut sessions = [staged, native].map(|p| Session::with_context(p, gpu.clone()));
                for update in [1.0, 0.5] {
                    for s in &mut sessions {
                        let input: Vec<_> = (0..17 * 23)
                            .map(|i| {
                                ((i % 13) as f32 - 6.0)
                                    * 0.125
                                    * if pointwise >= 3 {
                                        1.0
                                    } else {
                                        2.0f32.powi(-15)
                                    }
                            })
                            .collect();
                        s.set_input("x", &input);
                        s.set_parameter(
                            "a",
                            &(0..23 * if pointwise >= 3 { 38 } else { 19 })
                                .map(|i| ((i % 7) as f32 - 3.0) * 0.125 * update)
                                .collect::<Vec<_>>(),
                        );
                        s.set_parameter(
                            "b",
                            &(0..19 * 7)
                                .map(|i| ((i % 11) as f32 - 5.0) * 0.125)
                                .collect::<Vec<_>>(),
                        );
                        if pointwise == 2 {
                            s.set_parameter("norm", &[1.0; 19]);
                        }
                        if training {
                            s.set_input("target", &[0.125; 17 * 7]);
                        }
                        s.step();
                        s.wait();
                    }
                    for &buffer in &reads {
                        let len = sessions[0].plan().buffers[buffer.0 as usize] / 4;
                        let mut a = vec![0.0; len];
                        let mut b = vec![0.0; len];
                        sessions[0].read_buffer(buffer, &mut a);
                        sessions[1].read_buffer(buffer, &mut b);
                        assert_eq!(
                            a, b,
                            "{dtype:?}, training={training}, update={update}, {buffer:?}"
                        );
                    }
                }
            }
        }
    }
}
