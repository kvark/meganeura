use meganeura::{DType, Graph, compile::Kernel};

fn graph(m: usize, n: usize, k: usize, layout: usize, add: bool, round: bool) -> Graph {
    crate::support::data::layout_graph(DType::BF16, layout, add, round, [m, n, k])
}

fn values(len: usize, seed: u32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = (i as u32).wrapping_mul(1_664_525).wrapping_add(seed);
            (((x ^ (x >> 16)) % 61) as f32 - 30.0) / 64.0
        })
        .collect()
}

fn partition(k: usize, splits: usize, split: usize) -> std::ops::Range<usize> {
    let blocks = k.div_ceil(32);
    let base = blocks / splits;
    let extra = blocks % splits;
    let first = (split * base + split.min(extra)) * 32;
    let count = base + usize::from(split < extra);
    first..(first + count * 32).min(k)
}

/// `typed_matmul::explicit_storage_selects_the_typed_schedule` checks the
/// partitioned plan; this checks every partial and the reduced output.
#[test]
fn partitions_and_outputs_meet_native_accuracy_budget() {
    let gpu = crate::support::gpu::gpu();
    if meganeura::codegen::checked_cooperative_shape(
        &gpu.capabilities().cooperative_matrix.bf16_f32_shapes,
    )
    .is_none()
    {
        eprintln!("native BF16 matrices unavailable; skipping device-specific coverage");
        return;
    }
    let check_budget = || crate::support::gpu::assert_protocol_headroom(&gpu);
    let mut cases = 0;
    let mut checked = 0;
    for (m, n, k, layouts) in [(8, 256, 2048, 1), (33, 65, 1031, 3), (1, 7, 4099, 3)] {
        for layout in 0..layouts {
            // Rounding applies after the reduction, whatever the layout.
            for round in [false, true] {
                if round && layout != 0 {
                    continue;
                }
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
                    .find(|d| matches!(d.kernel, Kernel::CooperativeTyped { .. }))
                    .unwrap();
                let Kernel::CooperativeTyped { splits, .. } = product.kernel else {
                    unreachable!()
                };
                let splits = splits as usize;
                let partial_buffer = product.output_buffer;
                assert!(splits > 1);
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
                            let mut total_abs = 0.0;
                            for split in 0..splits {
                                let mut expected = 0.0_f64;
                                let mut sum_abs = 0.0;
                                let range = partition(k, splits, split);
                                let terms = range.len();
                                for reduce in range {
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
                                    let term = f64::from(a[ai]) * f64::from(b[bi]);
                                    expected += term;
                                    sum_abs += term.abs();
                                }
                                let got = f64::from(partial[split * m * n + index]);
                                let bound = meganeura::reference::native_matrix::error_bound(
                                    terms, sum_abs,
                                )
                                .unwrap();
                                assert!(
                                    got.is_finite() && (got - expected).abs() <= bound,
                                    "partial {m}x{n}x{k} layout={layout} split={split} at {index}: {got} != {expected}, bound={bound}"
                                );
                                total += expected;
                                total_abs += sum_abs;
                                checked += 1;
                            }
                            let bound =
                                meganeura::reference::native_matrix::error_bound(k, total_abs)
                                    .unwrap();
                            // Round the interval, not just its centre: native
                            // accumulation error can cross a BF16 midpoint.
                            // Conversion itself remains bit-exactly tested.
                            let endpoint = |value: f64| {
                                if round {
                                    f64::from(half::bf16::from_f32(value as f32).to_f32())
                                } else {
                                    value
                                }
                            };
                            let (lo, hi) = (endpoint(total - bound), endpoint(total + bound));
                            let got = f64::from(output[index]);
                            assert!(
                                got.is_finite() && got >= lo && got <= hi,
                                "output {m}x{n}x{k} layout={layout} at {index}: {got} outside [{lo}, {hi}]"
                            );
                            if !round {
                                // Also check the reduction independently of
                                // the matrix arithmetic's larger error budget.
                                let values =
                                    (0..splits).map(|s| f64::from(partial[s * m * n + index]));
                                let sum: f64 = values.clone().sum();
                                let scale: f64 = values.map(f64::abs).sum();
                                let reduction_bound =
                                    meganeura::reference::native_matrix::error_bound(splits, scale)
                                        .unwrap();
                                assert!(
                                    (got - sum).abs() <= reduction_bound,
                                    "partial reduction at {index}"
                                );
                            };
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
    if meganeura::codegen::checked_cooperative_shape(
        &gpu.capabilities().cooperative_matrix.bf16_f32_shapes,
    )
    .is_none()
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
    crate::support::gpu::assert_protocol_headroom(&gpu);
    println!("bf16_split_gradients_verified cases={cases}");
}
