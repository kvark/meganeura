//! F32 cooperative tiles must cover every SIMD-group quadrant,
//! transpose and partial K/M tile without silently reducing precision.
use meganeura::{CoopPolicy, Graph, build};

fn supported() -> bool {
    let shapes = &crate::support::gpu::gpu()
        .capabilities()
        .cooperative_matrix
        .f32_shapes;
    shapes.contains(&[8, 8, 8]) || shapes.contains(&[16, 16, 16])
}

#[test]
fn native_f32_staging_preserves_split_reduction_order() {
    use meganeura::{Session, SessionOptions, compile::Kernel};
    let gpu = crate::support::gpu::gpu();
    if !gpu
        .capabilities()
        .cooperative_matrix
        .f32_shapes
        .contains(&[16, 16, 16])
    {
        return;
    }
    // K/4 ends halfway through a 32-wide staging tile; 544 also gives an
    // unequal last partition. Exercise both contiguous and transposed loads.
    for k in [544, 576, 960] {
        let (m, n) = (64, 256);
        for kind in 0..4 {
            let mut graph = Graph::new();
            let a = graph.input("a", &if kind == 1 { [k, m] } else { [m, k] });
            let b = graph.input("b", &if kind == 2 { [n, k] } else { [k, n] });
            let y = match kind {
                1 => graph.matmul_at(a, b),
                2 => graph.matmul_bt(a, b),
                _ => graph.matmul(a, b),
            };
            let y = if kind == 3 {
                let c = graph.input("c", &[m, n]);
                graph.add(y, c)
            } else {
                y
            };
            graph.set_outputs(vec![y]);
            let mut config = crate::support::gpu::inference_config();
            config.runtime.coop = CoopPolicy::NativeF32;
            config.tune = false;
            let (session, _) = build(&graph, config);
            let plan = session.plan().clone();
            drop(session);
            let index = plan
                .dispatches
                .iter()
                .position(|d| matches!(d.kernel, Kernel::CooperativeTiled { splits: 4, .. }))
                .unwrap_or_else(|| {
                    panic!(
                        "split native16 product kind={kind}, K={k}: {:?}",
                        plan.dispatches
                    )
                });
            let a: Vec<_> = (0..m * k)
                .map(|i| ((i * 17 % 101) as f32 - 50.0) * 0.013)
                .collect();
            let b: Vec<_> = (0..n * k)
                .map(|i| ((i * 31 % 97) as f32 - 48.0) * 0.017)
                .collect();
            let c: Vec<_> = (0..m * n).map(|i| (i % 7) as f32 * 0.1).collect();
            let mut reference = vec![0.0f64; m * n];
            for row in 0..m {
                for col in 0..n {
                    let mut value = if kind == 3 {
                        f64::from(c[row * n + col])
                    } else {
                        0.0
                    };
                    for inner in 0..k {
                        let ai = if kind == 1 {
                            inner * m + row
                        } else {
                            row * k + inner
                        };
                        let bi = if kind == 2 {
                            col * k + inner
                        } else {
                            inner * n + col
                        };
                        value += f64::from(a[ai]) * f64::from(b[bi]);
                    }
                    reference[row * n + col] = value;
                }
            }
            let mut baseline = None;
            for columns in [64, 128] {
                for stage in [16, 32] {
                    for prefetch in [false, true] {
                        let mut variant = plan.clone();
                        let Kernel::CooperativeTiled { ref mut shape, .. } =
                            variant.dispatches[index].kernel
                        else {
                            unreachable!()
                        };
                        shape.columns = columns;
                        shape.k_stage = stage;
                        shape.prefetch = prefetch;
                        variant.dispatches[index].workgroups[1] = n as u32 / columns;
                        let mut session = Session::with_context_opts(
                            variant,
                            gpu.clone(),
                            SessionOptions {
                                coop: CoopPolicy::NativeF32,
                                ..Default::default()
                            },
                        );
                        session.set_input("a", &a);
                        session.set_input("b", &b);
                        if kind == 3 {
                            session.set_input("c", &c);
                        }
                        session.step();
                        session.wait();
                        let actual = session.read_output(m * n);
                        for (&value, &expected) in actual.iter().zip(&reference) {
                            assert!(
                                (f64::from(value) - expected).abs()
                                    <= 1e-5 * (1.0 + expected.abs()),
                                "kind={kind}, K={k}, stage={stage}, prefetch={prefetch}: {value} != {expected}"
                            );
                        }
                        if let Some(ref baseline) = baseline {
                            assert_eq!(
                                &actual, baseline,
                                "kind={kind}, K={k}, columns={columns}, stage={stage}, prefetch={prefetch}"
                            );
                        } else {
                            baseline = Some(actual);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn coop_f32_tiles_cover_transposes_edges_and_f32_exponents() {
    if !supported() {
        eprintln!("native f32 cooperative matrices unavailable; skipping device-specific coverage");
        return;
    }
    for (m, n, k) in [
        (33, 512, 17),
        (64, 256, 33),
        (65, 256, 65),
        (65, 272, 17),
        (32, 512, 1031),
        (64, 256, 2064),
        (64, 256, 4112),
    ] {
        for addend in [false, true] {
            if addend && k < 2048 {
                continue;
            }
            for transpose in [0, 1, 2] {
                let mut graph = Graph::new();
                let a_shape = if transpose == 1 { [k, m] } else { [m, k] };
                let b_shape = if transpose == 2 { [n, k] } else { [k, n] };
                let a = graph.input("a", &a_shape);
                let b = graph.input("b", &b_shape);
                let y = match transpose {
                    1 => graph.matmul_at(a, b),
                    2 => graph.matmul_bt(a, b),
                    _ => graph.matmul(a, b),
                };
                let y = if addend {
                    let src = graph.input("src", &[m, n]);
                    graph.add(y, src)
                } else {
                    y
                };
                graph.set_outputs(vec![y]);
                let mut config = crate::support::gpu::inference_config();
                config.runtime.coop = CoopPolicy::NativeF32;
                config.tune = false;
                let (mut session, _) = build(&graph, config);
                assert!(
                    session.plan().dispatches.iter().any(|d| d.use_coop()),
                    "test must exercise f32 cooperative execution: {m}x{n}x{k}, transpose={transpose}"
                );
                if m == 64
                    && k >= 2048
                    && crate::support::gpu::gpu()
                        .capabilities()
                        .cooperative_matrix
                        .f32_shapes
                        .contains(&[16, 16, 16])
                {
                    assert!(
                        session.plan().dispatches.iter().any(|d| matches!(
                            d.kernel,
                            meganeura::compile::Kernel::CooperativeTiled { splits: 16, .. }
                        )),
                        "native f32 tiles must be split along K"
                    );
                    assert!(
                        session
                            .plan()
                            .dispatches
                            .iter()
                            .any(|d| d.shader == meganeura::compile::ShaderEntry::SumRows)
                    );
                }
                // Both tiny and large operands exceed f16's usable exponent
                // range. Their products and all accumulated outputs remain f32.
                let a: Vec<f32> = (0..m * k)
                    .map(|i| ((i * 17 % 101) as f32 - 50.0) * 1e-12)
                    .collect();
                let b: Vec<f32> = (0..n * k)
                    .map(|i| ((i * 31 % 97) as f32 - 48.0) * 1e5)
                    .collect();
                session.set_input("a", &a);
                session.set_input("b", &b);
                let src: Vec<_> = (0..m * n)
                    .map(|i| ((i * 7 % 31) as f32 - 15.0) * 0.001)
                    .collect();
                if addend {
                    session.set_input("src", &src);
                }
                session.step();
                session.wait();
                let got = session.read_output(m * n);
                let mut max_error = 0.0_f64;
                let mut max_expected = 0.0_f64;
                for row in 0..m {
                    for col in 0..n {
                        let mut expected = if addend {
                            f64::from(src[row * n + col])
                        } else {
                            0.0_f64
                        };
                        for inner in 0..k {
                            let ai = if transpose == 1 {
                                inner * m + row
                            } else {
                                row * k + inner
                            };
                            let bi = if transpose == 2 {
                                col * k + inner
                            } else {
                                inner * n + col
                            };
                            expected += f64::from(a[ai]) * f64::from(b[bi]);
                        }
                        let actual = f64::from(got[row * n + col]);
                        assert!(actual.is_finite());
                        max_error = max_error.max((actual - expected).abs());
                        max_expected = max_expected.max(expected.abs());
                    }
                }
                assert!(
                    max_error < max_expected * 3e-5,
                    "{m}x{n}x{k} transpose={transpose}: max error {max_error}, max reference {max_expected}"
                );
            }
        }
    }
}

#[test]
fn coop_f32_prologue_addend_and_epilogue_cover_edge_rows() {
    if !supported() {
        return;
    }
    for (m, n, k) in [(65, 256, 33), (64, 256, 2064), (64, 256, 4112)] {
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i * 17 % 101) as f32 - 50.0) * 0.002)
            .collect();
        let b: Vec<f32> = (0..n * k)
            .map(|i| ((i * 31 % 97) as f32 - 48.0) * 0.002)
            .collect();
        let norm: Vec<f32> = (0..k).map(|i| 0.5 + i as f32 * 0.01).collect();
        let addend: Vec<f32> = (0..m * n)
            .map(|i| ((i * 7 % 31) as f32 - 15.0) * 0.001)
            .collect();
        for kind in ["prologue", "addend", "epilogue"] {
            let mut graph = Graph::new();
            let input = graph.input("a", &[m, k]);
            let weight = graph.input("b", &[k, n]);
            let lhs = if kind == "prologue" {
                let w = graph.input("norm", &[k]);
                graph.rms_norm(input, w, 1e-5)
            } else {
                input
            };
            let product = graph.matmul(lhs, weight);
            let output = match kind {
                "addend" => {
                    let src = graph.input("src", &[m, n]);
                    graph.add(product, src)
                }
                "epilogue" => graph.relu(product),
                _ => product,
            };
            graph.set_outputs(vec![output]);
            let mut config = crate::support::gpu::inference_config();
            config.runtime.coop = CoopPolicy::NativeF32;
            config.tune = false;
            let (mut session, _) = build(&graph, config);
            let dispatch = session
                .plan()
                .dispatches
                .iter()
                .find(|d| d.use_coop())
                .expect("cooperative f32 matrix dispatch");
            match kind {
                "prologue" => assert!(dispatch.matmul_prologue.is_some()),
                "epilogue" => assert!(dispatch.matmul_epilogue.is_some()),
                _ => assert_eq!(
                    dispatch.shader,
                    meganeura::compile::ShaderEntry::FusedMatMulAdd
                ),
            }
            if m == 64
                && crate::support::gpu::gpu()
                    .capabilities()
                    .cooperative_matrix
                    .f32_shapes
                    .contains(&[16, 16, 16])
            {
                assert_eq!(
                    matches!(
                        dispatch.kernel,
                        meganeura::compile::Kernel::CooperativeTiled { .. }
                    ),
                    kind != "epilogue"
                );
            }
            session.set_input("a", &a);
            session.set_input("b", &b);
            if kind == "prologue" {
                session.set_input("norm", &norm);
            }
            if kind == "addend" {
                session.set_input("src", &addend);
            }
            session.step();
            session.wait();
            let got = session.read_output(m * n);
            let mut squared_error = 0.0_f64;
            let mut squared_reference = 0.0_f64;
            for row in 0..m {
                let inverse_rms = (a[row * k..(row + 1) * k]
                    .iter()
                    .map(|&v| f64::from(v).powi(2))
                    .sum::<f64>()
                    / k as f64
                    + 1e-5)
                    .sqrt()
                    .recip();
                for col in 0..n {
                    let mut expected = if kind == "addend" {
                        f64::from(addend[row * n + col])
                    } else {
                        0.0
                    };
                    for inner in 0..k {
                        let scale = if kind == "prologue" {
                            inverse_rms * f64::from(norm[inner])
                        } else {
                            1.0
                        };
                        expected +=
                            f64::from(a[row * k + inner]) * scale * f64::from(b[inner * n + col]);
                    }
                    if kind == "epilogue" {
                        expected = expected.max(0.0);
                    }
                    squared_error += (f64::from(got[row * n + col]) - expected).powi(2);
                    squared_reference += expected.powi(2);
                }
            }
            assert!(
                (squared_error / squared_reference).sqrt() < 1e-5,
                "{kind}: error {squared_error}, reference {squared_reference}"
            );
        }
    }
}

#[test]
fn coop_f32_horizontal_tiles_keep_all_outputs_independent() {
    if !supported() {
        return;
    }
    let (m, n, k) = (65, 256, 33);
    for copies in [2, 3] {
        let mut graph = Graph::new();
        let a = graph.input("a", &[m, k]);
        let outputs = (0..copies)
            .map(|copy| {
                let b = graph.input(&format!("b{copy}"), &[k, n]);
                graph.matmul(a, b)
            })
            .collect();
        graph.set_outputs(outputs);
        let mut config = crate::support::gpu::inference_config();
        config.runtime.coop = CoopPolicy::NativeF32;
        config.tune = false;
        let (mut session, _) = build(&graph, config);
        assert!(
            session
                .plan()
                .dispatches
                .iter()
                .any(|dispatch| dispatch.use_coop() && dispatch.horizontal_batch == copies)
        );
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i * 17 % 101) as f32 - 50.0) * 0.002)
            .collect();
        let weights: Vec<Vec<f32>> = (0..copies)
            .map(|copy| {
                (0..n * k)
                    .map(|i| ((i * 31 % 97) as f32 - 48.0) * (copy + 1) as f32 * 0.002)
                    .collect()
            })
            .collect();
        session.set_input("a", &a);
        for (copy, b) in weights.iter().enumerate() {
            session.set_input(&format!("b{copy}"), b);
        }
        session.step();
        session.wait();
        for (copy, b) in weights.iter().enumerate() {
            let mut got = vec![0.0; m * n];
            session.read_output_by_index(copy, &mut got);
            let mut squared_error = 0.0_f64;
            let mut squared_reference = 0.0_f64;
            for row in 0..m {
                for col in 0..n {
                    let expected: f64 = (0..k)
                        .map(|inner| f64::from(a[row * k + inner]) * f64::from(b[inner * n + col]))
                        .sum();
                    squared_error += (f64::from(got[row * n + col]) - expected).powi(2);
                    squared_reference += expected.powi(2);
                }
            }
            assert!((squared_error / squared_reference).sqrt() < 1e-5);
        }
    }
}
