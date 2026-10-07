//! F32 cooperative tiles must cover every SIMD-group quadrant,
//! transpose and partial K/M tile without silently reducing precision.
use meganeura::{CoopPolicy, Graph, build};

fn supported() -> bool {
    crate::support::gpu::gpu()
        .capabilities()
        .cooperative_matrix
        .f32_shapes
        .contains(&[8, 8, 8])
}

#[test]
fn coop_f32_tiles_cover_transposes_edges_and_f32_exponents() {
    if !supported() {
        eprintln!(
            "cooperative f32 8x8 f32 matrices unavailable; skipping device-specific coverage"
        );
        return;
    }
    for (m, n, k) in [
        (33, 512, 17),
        (64, 256, 33),
        (65, 256, 65),
        (65, 272, 17),
        (32, 512, 1031),
    ] {
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
            graph.set_outputs(vec![y]);
            let mut config = crate::support::gpu::inference_config();
            config.runtime.coop = CoopPolicy::NativeF32;
            config.tune = false;
            let (mut session, _) = build(&graph, config);
            assert!(
                session.plan().dispatches.iter().any(|d| d.use_coop()),
                "test must exercise f32 cooperative execution: {m}x{n}x{k}, transpose={transpose}"
            );
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
            session.step();
            session.wait();
            let got = session.read_output(m * n);
            let mut max_error = 0.0_f64;
            let mut max_expected = 0.0_f64;
            for row in 0..m {
                for col in 0..n {
                    let mut expected = 0.0_f64;
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

#[test]
fn coop_f32_prologue_addend_and_epilogue_cover_edge_rows() {
    if !supported() {
        return;
    }
    let (m, n, k) = (65, 256, 33);
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

/// Two sessions, not a shape-by-layout sweep. Reuse each compiled kernel
/// across a changed addend and a zero product that isolates the add path.
#[test]
fn coop_f32_transposed_addends_cover_edges_and_replay() {
    use meganeura::compile::ShaderEntry;

    if !supported() {
        return;
    }
    let (m, n, k) = (65, 272, 33);
    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i * 17 % 101) as f32 - 50.0) * 0.002)
        .collect();
    let b: Vec<f32> = (0..n * k)
        .map(|i| ((i * 31 % 97) as f32 - 48.0) * 0.002)
        .collect();
    let zeros = vec![0.0; a.len()];
    for transposed_a in [true, false] {
        let mut graph = Graph::new();
        let x = graph.input("a", &if transposed_a { [k, m] } else { [m, k] });
        let w = graph.input("b", &if transposed_a { [k, n] } else { [n, k] });
        let src = graph.input("src", &[m, n]);
        let product = if transposed_a {
            graph.matmul_at(x, w)
        } else {
            graph.matmul_bt(x, w)
        };
        let y = graph.add(product, src);
        graph.set_outputs(vec![y]);
        let mut config = crate::support::gpu::inference_config();
        config.runtime.coop = CoopPolicy::NativeF32;
        config.runtime.poison = true;
        config.tune = false;
        let (mut session, _) = build(&graph, config);
        let expected_shader = if transposed_a {
            ShaderEntry::FusedMatMulATAdd
        } else {
            ShaderEntry::FusedMatMulBTAdd
        };
        assert_eq!(session.plan().dispatches.len(), 1);
        let dispatch = &session.plan().dispatches[0];
        assert_eq!(dispatch.shader, expected_shader);
        assert!(
            dispatch.use_coop(),
            "must exercise fused cooperative execution"
        );
        // Checked addend staging must not grow a logical external input.
        let addend_buffer = dispatch.input_buffers[2].0 as usize;
        assert_eq!(session.plan().buffers[addend_buffer], m * n * 4);
        session.set_input("b", &b);

        // Compute the product once, independently in f64, for all three feeds.
        let expected_product: Vec<f64> = (0..m * n)
            .map(|i| {
                let (row, col) = (i / n, i % n);
                (0..k)
                    .map(|inner| {
                        let ai = if transposed_a {
                            inner * m + row
                        } else {
                            row * k + inner
                        };
                        let bi = if transposed_a {
                            inner * n + col
                        } else {
                            col * k + inner
                        };
                        f64::from(a[ai]) * f64::from(b[bi])
                    })
                    .sum()
            })
            .collect();
        for (feed, input) in [a.as_slice(), a.as_slice(), zeros.as_slice()]
            .into_iter()
            .enumerate()
        {
            let addend: Vec<f32> = (0..m * n)
                .map(|i| (((i * 7 + feed * 11) % 31) as f32 - 15.0) * 0.003)
                .collect();
            session.set_input("a", input);
            session.set_input("src", &addend);
            session.step();
            session.wait();
            let got = session.read_output(m * n);
            let mut error = 0.0_f64;
            let mut magnitude = 0.0_f64;
            for i in 0..got.len() {
                assert!(got[i].is_finite());
                let expected =
                    f64::from(addend[i]) + if feed == 2 { 0.0 } else { expected_product[i] };
                error += (f64::from(got[i]) - expected).powi(2);
                magnitude += expected.powi(2);
            }
            assert!(magnitude > 0.0);
            assert!(
                (error / magnitude).sqrt() < 1e-5,
                "{expected_shader:?}, feed {feed}: error {error}, reference {magnitude}"
            );
        }
    }
}
