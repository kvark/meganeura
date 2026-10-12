use meganeura::{Graph, compile::ShaderEntry};

fn graph(m: usize, n: usize, k: usize, layout: usize, add: bool, round: bool) -> Graph {
    let mut graph = Graph::new();
    let a = graph.parameter("a", &if layout == 1 { [k, m] } else { [m, k] });
    let b = graph.parameter("b", &if layout == 2 { [n, k] } else { [k, n] });
    let a = graph.to_bf16(a);
    let b = graph.to_bf16(b);
    let y = match layout {
        0 => graph.matmul(a, b),
        1 => graph.matmul_at(a, b),
        _ => graph.matmul_bt(a, b),
    };
    if add {
        let bias = graph.parameter("bias", &[m, n]);
        let node = &mut graph.nodes_mut()[y as usize];
        node.op = match layout {
            0 => meganeura::graph::Op::FusedMatMulAdd,
            1 => meganeura::graph::Op::FusedMatMulATAdd,
            _ => meganeura::graph::Op::FusedMatMulBTAdd,
        };
        node.inputs.push(bias);
    }
    let y = if round {
        let y = graph.to_bf16(y);
        graph.from_bf16(y)
    } else {
        y
    };
    graph.set_outputs(vec![y]);
    graph
}

fn values(len: usize, seed: u32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = (i as u32).wrapping_mul(1_664_525).wrapping_add(seed);
            (((x ^ (x >> 16)) % 61) as f32 - 30.0) / 64.0
        })
        .collect()
}

#[test]
fn partitions_reduce_before_rounding_and_leave_addends_unsplit() {
    for layout in 0..3 {
        for add in [false, true] {
            let plan = meganeura::compile::compile(&graph(33, 65, 1031, layout, add, true));
            let index = plan
                .dispatches
                .iter()
                .position(|d| matches!(d.shader, ShaderEntry::MatMulBF16 { .. }))
                .unwrap();
            let product = &plan.dispatches[index];
            let splits = if add { 1 } else { 4 };
            assert_eq!(product.params, [33, 65, 1031, splits]);
            assert_eq!(product.workgroups, [2, 3, splits]);
            assert!(matches!(product.shader,
                ShaderEntry::MatMulBF16 { addend, output_bf16, .. }
                if addend == add && output_bf16 == add));
            if !add {
                let sum = &plan.dispatches[index + 1];
                assert_eq!(sum.shader, ShaderEntry::SumRows);
                assert_eq!(sum.params, [4, 33 * 65, 1, 0]);
                assert_eq!(sum.input_buffers, [product.output_buffer]);
                assert_eq!(
                    plan.buffers[product.output_buffer.0 as usize],
                    4 * 33 * 65 * 4
                );
                assert_ne!(sum.output_buffer, product.output_buffer);
            }
        }
    }
}

#[test]
fn partitions_and_outputs_match_full_f64() {
    let gpu = crate::support::gpu::gpu();
    if !gpu
        .capabilities()
        .cooperative_matrix
        .bf16_f32_shapes
        .contains(&[16, 16, 16])
    {
        eprintln!("native BF16 matrices unavailable; skipping device-specific coverage");
        return;
    }
    let check_budget = || {
        let memory = gpu.memory_stats();
        assert!(memory.budget.saturating_sub(memory.usage) >= 2 << 30);
    };
    let mut cases = 0;
    let mut checked = 0;
    for (m, n, k, layouts) in [
        (8, 256, 2048, 1),
        (128, 256, 2560, 1),
        (33, 65, 1031, 3),
        (1, 7, 4099, 3),
    ] {
        for layout in 0..layouts {
            for round in [false, true] {
                check_budget();
                let mut config = crate::support::gpu::inference_config();
                config.tune = false;
                config.runtime.poison = true;
                config.runtime.no_alias = true;
                let (mut session, _) =
                    meganeura::build(&graph(m, n, k, layout, false, round), config);
                let product = session
                    .plan()
                    .dispatches
                    .iter()
                    .find(|d| matches!(d.shader, ShaderEntry::MatMulBF16 { .. }))
                    .unwrap();
                let splits = product.params[3] as usize;
                let partial_buffer = product.output_buffer;
                assert!(splits > 1);
                let chunk = k / splits / 32 * 32;
                // Change feeds and reuse the same poisoned session, not just fresh allocations.
                for seed in [1_013_904_223, 1_987_654_321] {
                    let a = values(m * k, seed);
                    let b = values(n * k, seed ^ 0x12345678);
                    session.set_parameter("a", &a);
                    session.set_parameter("b", &b);
                    session.step();
                    session.wait();
                    let mut partial = vec![0.0; splits * m * n];
                    session.read_buffer(partial_buffer, &mut partial);
                    let mut output = vec![0.0; m * n];
                    session.read_output_by_index(0, &mut output);
                    for row in 0..m {
                        for col in 0..n {
                            let index = row * n + col;
                            let mut total = 0.0;
                            for split in 0..splits {
                                let end = if split + 1 == splits {
                                    k
                                } else {
                                    (split + 1) * chunk
                                };
                                let mut expected = 0.0_f64;
                                for reduce in split * chunk..end {
                                    let ai = if layout == 1 {
                                        reduce * m + row
                                    } else {
                                        row * k + reduce
                                    };
                                    let bi = if layout == 2 {
                                        col * k + reduce
                                    } else {
                                        reduce * n + col
                                    };
                                    expected += f64::from(a[ai]) * f64::from(b[bi]);
                                }
                                assert_eq!(
                                    f64::from(partial[split * m * n + index]),
                                    expected,
                                    "partial {m}x{n}x{k} layout={layout} split={split} at {index}"
                                );
                                total += expected;
                                checked += 1;
                            }
                            let expected = if round {
                                half::bf16::from_f32(total as f32).to_f32()
                            } else {
                                total as f32
                            };
                            assert_eq!(
                                output[index], expected,
                                "output {m}x{n}x{k} layout={layout} at {index}"
                            );
                            checked += 1;
                        }
                    }
                    cases += 1;
                }
                check_budget();
            }
        }
    }
    println!("bf16_split_f64_verified cases={cases} values={checked} reused_poisoned_buffers=true");
}

#[test]
fn long_products_preserve_master_and_reduced_gradients() {
    let gpu = crate::support::gpu::gpu();
    if !gpu
        .capabilities()
        .cooperative_matrix
        .bf16_f32_shapes
        .contains(&[16, 16, 16])
    {
        eprintln!("native BF16 matrices unavailable; skipping device-specific coverage");
        return;
    }
    let mut cases = 0;
    for layout in 0..3 {
        for add in [false, true] {
            for reduced in [false, true] {
                let mut graph = graph(3, 5, 1031, layout, add, true);
                let y = graph.outputs()[0];
                let weights = graph.input("weights", &[3, 5]);
                let weighted = graph.mul(y, weights);
                let loss = graph.sum_all(weighted);
                graph.set_outputs(vec![loss, y]);
                let mut feeds = meganeura::reference::Feeds::new();
                for (name, len, seed) in [("a", 3 * 1031, 101), ("b", 5 * 1031, 207)] {
                    let data: Vec<_> = values(len, seed).into_iter().map(|v| v / 1.137).collect();
                    feeds.set(name, &data);
                }
                feeds.set("weights", &values(15, 509));
                if add {
                    feeds.set("bias", &[0.125; 15]);
                }
                let mut options = meganeura::reference::gpu::Options::default();
                options.compile.bf16_matrix_gradients = reduced;
                let report =
                    meganeura::reference::gpu::check_training(&graph, &feeds, &options).unwrap();
                report.assert_passed("split BF16 product and master gradients");
                cases += 1;
            }
        }
    }
    let memory = gpu.memory_stats();
    assert!(memory.budget.saturating_sub(memory.usage) >= 2 << 30);
    println!("bf16_split_gradients_verified cases={cases}");
}
