use crate::{
    DType,
    compile::{ExecutionPlan, Kernel, ShaderEntry},
};

/// Narrow only private transform buffers consumed by one aligned product.
/// Everything else, including incomplete tiles and protected derivatives,
/// retains its original F32 implementation.
pub(super) fn fill(
    plan: &mut ExecutionPlan,
    storage: DType,
    native: [u32; 3],
    subgroup: u32,
) -> usize {
    if subgroup == 0 || subgroup > 128 {
        return 0;
    }
    let mut protected = plan.externally_visible_buffers();
    protected.extend(plan.node_buffers.iter().map(|&(_, buffer)| buffer));
    let mut uses = std::collections::HashMap::new();
    for d in &plan.dispatches {
        for buffer in &d.input_buffers {
            *uses.entry(*buffer).or_insert(0) += 1;
        }
    }
    let mut count = 0;
    for i in 0..plan.dispatches.len() {
        let d = &plan.dispatches[i];
        if d.shader != ShaderEntry::WinogradBatchedMatMul
            || d.requires_full_precision
            || d.schedule_locked
            || !matches!(d.kernel, Kernel::Default | Kernel::ScalarMatmul(_))
            || d.input_buffers.len() != 2
            || !d.extra_outputs.is_empty()
        {
            continue;
        }
        let [m, n, k, _] = d.params[..] else { continue };
        let tile = if n == 16 {
            [native[0], 16]
        } else {
            [32.min(256 / subgroup / 2 * native[0]), 32]
        };
        if m == 0
            || n == 0
            || k == 0
            || !m.is_multiple_of(tile[0])
            || !n.is_multiple_of(tile[1])
            || !k.is_multiple_of(native[2])
        {
            continue;
        }
        let grid = [n / tile[1], m / tile[0], 16];
        if !crate::compile::workgroups_within_portable_limits(grid) {
            continue;
        }
        let mut producers = Vec::new();
        for (&buffer, shader) in d.input_buffers.iter().zip([
            ShaderEntry::WinogradWeightTransform,
            ShaderEntry::WinogradInputTransform,
        ]) {
            if protected.contains(&buffer) || uses.get(&buffer) != Some(&1) {
                break;
            }
            let writers: Vec<_> = plan
                .dispatches
                .iter()
                .enumerate()
                .filter(|&(_, p)| p.output_buffer == buffer)
                .collect();
            let [(index, p)] = writers[..] else { break };
            if index >= i
                || p.shader != shader
                || p.kernel != Kernel::Default
                || p.requires_full_precision
                || p.schedule_locked
                || !p.extra_outputs.is_empty()
            {
                break;
            }
            producers.push(index);
        }
        if producers.len() != 2 {
            continue;
        }
        for (index, elements) in producers.into_iter().zip([
            16 * u64::from(m) * u64::from(k),
            16 * u64::from(k) * u64::from(n),
        ]) {
            let p = &mut plan.dispatches[index];
            p.kernel = Kernel::WinogradTransform(storage);
            plan.buffers[p.output_buffer.0 as usize] = (elements * 2) as usize;
        }
        let d = &mut plan.dispatches[i];
        d.kernel = Kernel::WinogradMatmul { storage, tile };
        d.workgroups = grid;
        count += 1;
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::{BufferRef, Dispatch};

    fn plan(m: u32, n: u32, k: u32) -> ExecutionPlan {
        ExecutionPlan {
            buffers: vec![
                16 * (m * k) as usize * 4,
                16 * (k * n) as usize * 4,
                16 * (m * n) as usize * 4,
            ],
            dispatches: vec![
                Dispatch {
                    shader: ShaderEntry::WinogradWeightTransform,
                    output_buffer: BufferRef(0),
                    ..Default::default()
                },
                Dispatch {
                    shader: ShaderEntry::WinogradInputTransform,
                    output_buffer: BufferRef(1),
                    ..Default::default()
                },
                Dispatch {
                    shader: ShaderEntry::WinogradBatchedMatMul,
                    input_buffers: vec![BufferRef(0), BufferRef(1)],
                    output_buffer: BufferRef(2),
                    params: vec![m, n, k, 0],
                    ..Default::default()
                },
            ],
            ..crate::compile::compile(&crate::Graph::new())
        }
    }

    #[test]
    fn narrowing_preserves_shared_visible_and_full_precision_buffers() {
        for mutation in 0..5 {
            let mut p = plan(64, 16, 64);
            match mutation {
                0 => p.output_buffers.push(BufferRef(0)),
                1 => p.dispatches.push(Dispatch {
                    input_buffers: vec![BufferRef(1)],
                    ..Default::default()
                }),
                2 => p.dispatches[0].requires_full_precision = true,
                3 => p.dispatches[2].requires_full_precision = true,
                _ => p.node_buffers.push((0, BufferRef(1))),
            }
            let original = p.clone();
            assert_eq!(fill(&mut p, DType::F16, [8, 16, 16], 32), 0);
            assert_eq!(p, original);
        }
    }

    #[test]
    fn native_tiles_require_alignment_and_shrink_only_their_operands() {
        for rows in [8, 16] {
            for (m, n, k, expected) in [
                (64, 16, 64, 1),
                (64, 64, 64, 1),
                (63, 16, 64, 0),
                (64, 17, 64, 0),
                (64, 16, 65, 0),
            ] {
                let mut p = plan(m, n, k);
                let sizes = p.buffers.clone();
                assert_eq!(fill(&mut p, DType::F16, [rows, 16, 16], 32), expected);
                assert_eq!(p.buffers[2], sizes[2]);
                if expected == 1 {
                    assert_eq!(&p.buffers[..2], &[sizes[0] / 2, sizes[1] / 2]);
                    let Kernel::WinogradMatmul { tile, .. } = p.dispatches[2].kernel else {
                        panic!()
                    };
                    assert_eq!(p.dispatches[2].workgroups, [n / tile[1], m / tile[0], 16]);
                } else {
                    assert_eq!(p.buffers, sizes);
                }
            }
        }
    }

    #[test]
    fn native_products_match_exact_convolution_with_updated_weights() {
        let gpu = crate::reference::gpu::shared_context();
        let caps = gpu.capabilities();
        for (h, w) in [(3, 7), (7, 15)] {
            let mut graph = crate::Graph::new();
            let x = graph.input("x", &[2 * 64 * h * w]);
            let weight = graph.parameter("w", &[64 * 64 * 9]);
            let y = graph.conv2d(x, weight, 2, 64, h as u32, w as u32, 64, 3, 3, 1, 1);
            graph.set_outputs(vec![y]);
            // These dyadic values and their transforms are exact in both half
            // formats. Include batches, padding and partial spatial tiles.
            let input: Vec<f32> = (0..2 * 64 * h * w)
                .map(|i| (i % 5) as f32 / 32.0 - 0.0625)
                .collect();
            let weights: Vec<f32> = (0..64 * 64 * 9)
                .map(|i| (i % 7) as f32 / 64.0 - 0.046875)
                .collect();
            let expected = crate::reference::evaluate_outputs(
                &graph,
                crate::reference::Feeds::new()
                    .set("x", &input)
                    .set("w", &weights),
            )
            .unwrap()
            .remove(0)
            .data;
            crate::optimize::apply_winograd_conv_fusions(
                &mut graph,
                &mut Vec::new(),
                &Default::default(),
            );
            for storage in [DType::F16, DType::BF16] {
                if !crate::runtime::typed_matrix_supported(&gpu, storage) {
                    continue;
                }
                let shapes = if storage == DType::F16 {
                    &caps.cooperative_matrix.f16_f32_shapes
                } else {
                    &caps.cooperative_matrix.bf16_f32_shapes
                };
                let shape = crate::codegen::checked_cooperative_shape(shapes).unwrap();
                let mut plan = crate::compile::compile(&graph);
                assert_eq!(fill(&mut plan, storage, shape, caps.subgroup_size), 1);
                let mut session = crate::Session::with_context_opts(
                    plan,
                    gpu.clone(),
                    crate::SessionOptions {
                        poison: true,
                        ..Default::default()
                    },
                );
                session.set_input("x", &input);
                for sign in [1.0, -1.0] {
                    session
                        .set_parameter("w", &weights.iter().map(|v| v * sign).collect::<Vec<_>>());
                    session.step();
                    session.wait();
                    for (i, (actual, expected)) in session
                        .read_output(expected.len())
                        .into_iter()
                        .zip(&expected)
                        .enumerate()
                    {
                        assert_eq!(
                            f64::from(actual),
                            expected * f64::from(sign),
                            "{storage:?}, {h}x{w}, sign={sign}, output={i}"
                        );
                    }
                }
            }
        }
    }
}
