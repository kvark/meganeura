//! F32 Winograd weight gradients, admitted only by whole-program qualification.
use crate::compile::{BufferRef, Dispatch, ExecutionPlan, ShaderEntry};

/// Replace eligible 3x3 stride-1 dW producers, including private split-K pairs.
/// The caller rebuilds groups after this pass and checks the complete plan budget.
pub(super) fn fill(plan: &mut ExecutionPlan, max_scratch: usize) -> usize {
    let mut protected = plan.externally_visible_buffers();
    protected.extend(plan.node_buffers.iter().map(|&(_, b)| b));
    let mut reused = Vec::new();
    let mut removed = std::collections::HashSet::new();
    let mut replacements = Vec::new();
    let mut added: Vec<usize> = Vec::new();
    let mut scratch = 0usize;
    for (index, d) in plan.dispatches.iter().enumerate() {
        let split = match d.shader {
            ShaderEntry::Conv2dGradWeightGemm
            | ShaderEntry::Conv2dGradWeightGemmSmall
            | ShaderEntry::Conv2dGradWeightGemm16 => false,
            ShaderEntry::Conv2dGradWeightGemmSplit
            | ShaderEntry::Conv2dGradWeightGemmSplitSmall
            | ShaderEntry::Conv2dGradWeightGemmSplit16 => true,
            _ => continue,
        };
        if d.params.len() != 12
            || d.input_buffers.len() != 2
            || !d.extra_outputs.is_empty()
            || d.schedule_locked
            || d.matmul_epilogue.is_some()
            || d.weight_format != crate::compile::WeightFormat::F32
        {
            continue;
        }
        if crate::tune::TuneClass::from_dispatch(d, None).is_none() {
            continue;
        }
        let p = &d.params;
        let (ci, co, oh, ow) = (p[1], p[4], p[9], p[10]);
        if p[5..8] != [3, 3, 1] || p[8] != p[11] || ci < 16 || co < 16 {
            continue;
        }
        let tiles_h = oh.div_ceil(4);
        let tiles_w = ow.div_ceil(4);
        let Some(tiles) = p[0]
            .checked_mul(tiles_h)
            .and_then(|v| v.checked_mul(tiles_w))
        else {
            continue;
        };
        if tiles == 0 {
            continue;
        }
        let splits = tiles.div_ceil(256).clamp(1, 4);
        let Some(lengths) = [
            ci.checked_mul(tiles),
            co.checked_mul(tiles),
            co.checked_mul(ci).and_then(|v| v.checked_mul(splits)),
        ]
        .into_iter()
        .map(|n| {
            n.and_then(|n| n.checked_mul(36))
                .and_then(|n| (n as usize).checked_mul(4))
        })
        .collect::<Option<Vec<_>>>() else {
            continue;
        };
        let Some(total) = lengths
            .iter()
            .try_fold(scratch, |n, &v| n.checked_add(v))
            .filter(|&n| n <= max_scratch)
        else {
            continue;
        };
        let grids = [
            [(ci * tiles).div_ceil(256), 1, 1],
            [(co * tiles).div_ceil(256), 1, 1],
            [ci.div_ceil(64), co.div_ceil(64), 36 * splits],
            [(co * ci).div_ceil(256), 1, 1],
        ];
        if !grids
            .iter()
            .all(|&g| crate::compile::workgroups_within_portable_limits(g))
        {
            continue;
        }
        let mut output = d.output_buffer;
        let mut reuse = None;
        if split {
            if protected.contains(&output)
                || plan.dispatches.iter().enumerate().any(|(i, other)| {
                    i != index
                        && (other.output_buffer == output || other.extra_outputs.contains(&output))
                })
            {
                continue;
            }
            let mut consumers = plan
                .dispatches
                .iter()
                .enumerate()
                .filter(|&(_, c)| c.input_buffers.contains(&output));
            let Some((sum_index, sum)) = consumers.next() else {
                continue;
            };
            if consumers.next().is_some()
                || sum_index <= index
                || sum.shader != ShaderEntry::SumRows
                || sum.kernel != crate::compile::Kernel::Default
                || sum.pointwise_storage != Default::default()
                || sum.input_buffers != [output]
                || sum.params.len() != 4
                || sum.params[..2] != [d.workgroups[2], co * ci * 9]
                || !sum.extra_outputs.is_empty()
            {
                continue;
            }
            output = sum.output_buffer;
            removed.insert(sum_index);
            reuse = Some(d.output_buffer);
        }
        let start = plan.buffers.len() + added.len();
        let (v, dy, u) = if let Some(partial) = reuse {
            reused.push((partial, lengths[0]));
            (
                partial,
                BufferRef(start as u32),
                BufferRef(start as u32 + 1),
            )
        } else {
            (
                BufferRef(start as u32),
                BufferRef(start as u32 + 1),
                BufferRef(start as u32 + 2),
            )
        };
        let make = |shader, workgroups, inputs, output_buffer, params| Dispatch {
            shader,
            workgroups,
            input_buffers: inputs,
            output_buffer,
            params,
            label: format!("{} Winograd dW", d.label),
            origin: d.origin.clone(),
            is_derivative: d.is_derivative,
            requires_full_precision: d.requires_full_precision,
            fusion_barrier: d.fusion_barrier,
            ..Default::default()
        };
        let sequence = vec![
            make(
                ShaderEntry::WinogradGradInputTransform,
                grids[0],
                vec![d.input_buffers[1]],
                v,
                vec![ci, p[2], p[3], p[8], tiles_h, tiles_w, tiles, 0],
            ),
            make(
                ShaderEntry::WinogradGradDeltaTransform,
                grids[1],
                vec![d.input_buffers[0]],
                dy,
                vec![co, oh, ow, 0, tiles_h, tiles_w, tiles, 0],
            ),
            make(
                ShaderEntry::WinogradGradProduct(splits),
                grids[2],
                vec![dy, v],
                u,
                vec![co, ci, tiles, 0],
            ),
            make(
                ShaderEntry::WinogradGradOutputTransform,
                grids[3],
                vec![u],
                output,
                vec![co, ci, 0, 0, 0, 0, 0, splits],
            ),
        ];
        scratch = total;
        added.extend(if reuse.is_some() {
            &lengths[1..]
        } else {
            &lengths[..]
        });
        replacements.push((index, sequence));
    }
    let count = replacements.len();
    for (buffer, bytes) in reused {
        plan.buffers[buffer.0 as usize] = bytes;
    }
    plan.buffers.extend(added);
    let mut replacements = replacements.into_iter().peekable();
    let old = std::mem::take(&mut plan.dispatches);
    for (index, d) in old.into_iter().enumerate() {
        if replacements.peek().is_some_and(|r| r.0 == index) {
            plan.dispatches.extend(replacements.next().unwrap().1);
        } else if !removed.contains(&index) {
            plan.dispatches.push(d);
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    fn graph(ci: u32, co: u32, h: u32, w: u32, batch: u32, pad: u32) -> crate::Graph {
        let mut g = crate::Graph::new();
        let x = g.input("x", &[(batch * ci * h * w) as usize]);
        let dy = g.input(
            "dy",
            &[(batch * co * (h + 2 * pad - 2) * (w + 2 * pad - 2)) as usize],
        );
        let dw = g.conv2d_grad_weight(dy, x, ci, h, w, co, 3, 3, 1, pad, pad);
        g.set_outputs(vec![dw]);
        g
    }
    #[test]
    fn budget_and_split_consumers_preserve_fallback() {
        let g = graph(64, 64, 17, 19, 2, 1);
        let base = crate::compile::compile(&g);
        let mut small = base.clone();
        assert_eq!(fill(&mut small, 1), 0);
        assert_eq!(small, base);
        let mut p = base.clone();
        assert_eq!(fill(&mut p, usize::MAX), 1);
        assert_eq!(p.output_buffers, base.output_buffers);
        assert_eq!(p.dispatches.len(), 4);
        assert!(
            !p.dispatches
                .iter()
                .any(|d| d.shader == ShaderEntry::SumRows)
        );
        let before = p.clone();
        assert_eq!(fill(&mut p, usize::MAX), 0);
        assert_eq!(p, before);
        let producer = base
            .dispatches
            .iter()
            .position(|d| {
                matches!(
                    d.shader,
                    ShaderEntry::Conv2dGradWeightGemmSplit
                        | ShaderEntry::Conv2dGradWeightGemmSplitSmall
                        | ShaderEntry::Conv2dGradWeightGemmSplit16
                )
            })
            .unwrap();
        for visible in [false, true] {
            let mut p = base.clone();
            let partial = p.dispatches[producer].output_buffer;
            if visible {
                p.output_buffers.push(partial);
            } else {
                p.dispatches.push(Dispatch {
                    input_buffers: vec![partial],
                    ..Default::default()
                });
            }
            let old = p.clone();
            assert_eq!(fill(&mut p, usize::MAX), 0);
            assert_eq!(p, old);
        }
    }
    #[test]
    fn generated_adjoint_modules_validate_and_emit_spirv() {
        for group in [
            crate::codegen::ShaderGroup::WinogradGradInputTransform,
            crate::codegen::ShaderGroup::WinogradGradDeltaTransform,
            crate::codegen::ShaderGroup::WinogradGradOutputTransform,
            crate::codegen::ShaderGroup::WinogradGradProduct(1),
            crate::codegen::ShaderGroup::WinogradGradProduct(4),
        ] {
            let mut module = crate::codegen::winograd::weight_gradient(group);
            crate::codegen::validate_shader_for_test(
                &mut module.module,
                naga::valid::Capabilities::empty(),
                (1, 3),
            )
            .unwrap();
        }
        for group in [
            crate::codegen::ShaderGroup::Conv2dGradWeightGemm16,
            crate::codegen::ShaderGroup::Conv2dGradWeightGemmSmall,
            crate::codegen::ShaderGroup::Conv2dGradWeightGemm,
            crate::codegen::ShaderGroup::Conv2dGradWeightGemmSplit,
        ] {
            for k in [16, 32] {
                for interleave in [false, true] {
                    if matches!(
                        group,
                        crate::codegen::ShaderGroup::Conv2dGradWeightGemmSplit
                    ) && k == 32
                    {
                        continue;
                    }
                    let mut module = crate::codegen::generate_conv_weight_module(
                        group,
                        k,
                        interleave,
                        crate::codegen::ConvWeightStaging::Columns,
                        None,
                    );
                    crate::codegen::validate_shader_for_test(
                        &mut module.module,
                        naga::valid::Capabilities::empty(),
                        (1, 3),
                    )
                    .unwrap();
                }
            }
        }
    }
    #[test]
    #[ignore = "GPU: F64 oracle for F(4,3) adjoints and both implicit-GEMM staging layouts"]
    fn adjoints_and_staging_match_f64() {
        use crate::reference::{Feeds, Tolerance, check, evaluate, magnitudes};
        let gpu = std::sync::Arc::new(
            crate::init_gpu_context_with(crate::GpuOptions::from_env()).unwrap(),
        );
        for (ci, co, h, w, batch, pad) in [
            (17, 19, 7, 9, 2, 0),
            (17, 19, 7, 9, 2, 1),
            (16, 16, 5, 7, 1, 2),
            (16, 16, 66, 65, 2, 1),
        ] {
            let g = graph(ci, co, h, w, batch, pad);
            let mut feeds = Feeds::new();
            feeds.fill_random(&g, 709, 0.25);
            let values = evaluate(&g, &feeds).unwrap();
            let id = g.outputs()[0];
            let node = g.node(id);
            let inputs = node
                .inputs
                .iter()
                .map(|&i| &values[i as usize])
                .collect::<Vec<_>>();
            let want = &values[id as usize];
            let magnitude = magnitudes(&g, node, &inputs, want).unwrap();
            let base = crate::compile::compile(&g);
            let mut plans = Vec::new();
            let mut p = base.clone();
            assert_eq!(fill(&mut p, usize::MAX), 1);
            plans.push(p);
            if h == 7 {
                for staging in [
                    crate::codegen::ConvWeightStaging::Columns,
                    crate::codegen::ConvWeightStaging::Reduction,
                ] {
                    for interleave_columns in [false, true] {
                        let mut p = base.clone();
                        for d in &mut p.dispatches {
                            if matches!(
                                d.shader,
                                ShaderEntry::Conv2dGradWeightGemm
                                    | ShaderEntry::Conv2dGradWeightGemmSmall
                                    | ShaderEntry::Conv2dGradWeightGemm16
                                    | ShaderEntry::Conv2dGradWeightGemmSplit
                                    | ShaderEntry::Conv2dGradWeightGemmSplitSmall
                                    | ShaderEntry::Conv2dGradWeightGemmSplit16
                            ) {
                                d.kernel = crate::compile::Kernel::ConvWeight {
                                    k_tile: 16,
                                    interleave_columns,
                                    staging,
                                };
                            }
                        }
                        plans.push(p);
                    }
                }
            }
            for mut p in plans {
                crate::compile::schedule_dispatches(&mut p, false, false);
                let label = p
                    .dispatches
                    .iter()
                    .map(|d| format!("{:?}/{:?}", d.shader, d.kernel))
                    .collect::<Vec<_>>();
                let mut s = crate::Session::with_context_opts(
                    p,
                    gpu.clone(),
                    crate::SessionOptions {
                        poison: true,
                        coop: crate::CoopPolicy::Disabled,
                        ..Default::default()
                    },
                );
                for name in ["x", "dy"] {
                    s.set_input(name, &feeds.f32(name).unwrap());
                }
                for _ in 0..2 {
                    s.step();
                    s.wait();
                    let mut got = vec![0.; want.len()];
                    s.read_output_by_index(0, &mut got);
                    assert!(
                        check(&got, &want.data, &magnitude, Tolerance::default()).is_ok(),
                        "{label:?} ({ci},{co},{h},{w},{batch},{pad})"
                    );
                }
            }
        }
    }
}
