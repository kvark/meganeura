//! Numeric coverage for the generic operators used by multimodal encoders.

use meganeura::Graph;

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "output[{index}]={actual}, expected {expected}"
        );
    }
}

#[test]
fn clamp_is_inclusive() {
    let mut graph = Graph::new();
    let input = graph.input("input", &[6]);
    let output = graph.clamp(input, -1.0, 2.0);
    graph.set_outputs(vec![output]);

    let mut session = meganeura::build(&graph, meganeura::SessionConfig::inference_from_env()).0;
    session.set_input("input", &[-3.0, -1.0, -0.5, 0.0, 2.0, 4.0]);
    session.step();
    session.wait();
    let actual = session.read_output(6);
    assert_close(&actual, &[-1.0, -1.0, -0.5, 0.0, 2.0, 2.0], 1.0e-6);
}

#[test]
fn clamp_preserves_values_inside_large_bounds() {
    let input_values = [-123.5, -1.0, 0.0, 1.0, 456.25];
    let mut graph = Graph::new();
    let input = graph.input("input", &[input_values.len()]);
    let output = graph.clamp(input, -10_000_000_000.0, 10_000_000_000.0);
    graph.set_outputs(vec![output]);

    let mut session = meganeura::build(&graph, meganeura::SessionConfig::inference_from_env()).0;
    session.set_input("input", &input_values);
    session.step();
    session.wait();
    assert_close(
        &session.read_output(input_values.len()),
        &input_values,
        1.0e-6,
    );
}

#[test]
fn rope_accepts_independent_row_positions() {
    let rows = 3;
    let dim = 8;
    let theta = 100.0_f32;
    let input: Vec<_> = (0..rows * dim).map(|i| i as f32 * 0.125 - 1.0).collect();
    let positions = [4_u32, 1, 9];

    let mut graph = Graph::new();
    let input_node = graph.input("input", &[rows, dim]);
    let positions_node = graph.input_u32("positions", &[rows]);
    let output = graph.rope_with_positions(input_node, theta, positions_node, dim as u32);
    graph.set_outputs(vec![output]);
    let mut session = meganeura::build(&graph, meganeura::SessionConfig::inference_from_env()).0;
    session.set_input("input", &input);
    session.set_input_u32("positions", &positions);
    session.step();
    session.wait();
    let actual = session.read_output(rows * dim);

    let mut expected = vec![0.0; input.len()];
    for (row, &position) in positions.iter().enumerate().take(rows) {
        for pair in 0..dim / 2 {
            let angle = position as f32 * theta.powf(-2.0 * pair as f32 / dim as f32);
            let (sin, cos) = angle.sin_cos();
            let first = row * dim + pair;
            let second = first + dim / 2;
            expected[first] = input[first] * cos - input[second] * sin;
            expected[second] = input[first] * sin + input[second] * cos;
        }
    }
    assert_close(&actual, &expected, 2.0e-5);
}

#[test]
fn bias_mul_broadcasts_over_rows() {
    let mut graph = Graph::new();
    let input = graph.input("input", &[3, 4]);
    let scale = graph.parameter("scale", &[4]);
    let output = graph.bias_mul(input, scale);
    graph.set_outputs(vec![output]);
    let mut session = meganeura::build(&graph, meganeura::SessionConfig::inference_from_env()).0;
    session.set_parameter("scale", &[2.0, -1.0, 0.5, 0.0]);
    session.set_input(
        "input",
        &[
            1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0, 8.0, 6.0, 4.0, 2.0,
        ],
    );
    session.step();
    session.wait();
    assert_close(
        &session.read_output(12),
        &[
            2.0, -2.0, 1.5, 0.0, -2.0, 2.0, -1.5, 0.0, 16.0, -6.0, 2.0, 0.0,
        ],
        1.0e-6,
    );
}

#[test]
fn batched_f16_matmul_bt_preserves_fused_clamp_bounds() {
    const M: usize = 13;
    const K: usize = 32;
    const N: usize = 20;
    let a: Vec<_> = (0..M * K)
        .map(|i| ((i * 7 % 29) as f32 - 14.0) * 0.125)
        .collect();
    let b: Vec<_> = (0..N * K)
        .map(|i| ((i * 11 % 31) as f32 - 15.0) * 0.1)
        .collect();
    let b_rounded: Vec<_> = b
        .iter()
        .map(|&value| half::f16::from_f32(value).to_f32())
        .collect();
    let mut expected = vec![0.0; M * N];
    for row in 0..M {
        for column in 0..N {
            expected[row * N + column] = (0..K)
                .map(|inner| a[row * K + inner] * b_rounded[column * K + inner])
                .sum::<f32>()
                .clamp(-3.25, 4.5);
        }
    }
    assert!(expected.contains(&-3.25));
    assert!(expected.contains(&4.5));
    assert!(expected.iter().any(|value| (-3.25..4.5).contains(value)));

    let mut graph = Graph::new();
    let an = graph.input("a", &[M, K]);
    let bn = graph.parameter_f16("b", &[N, K]);
    let product = graph.matmul_bt(an, bn);
    let output = graph.clamp(product, -3.25, 4.5);
    graph.set_outputs(vec![output]);
    let mut session = meganeura::build(&graph, meganeura::SessionConfig::inference_from_env()).0;
    session.set_input("a", &a);
    session.set_parameter("b", &b);
    session.step();
    session.wait();
    assert_close(&session.read_output(M * N), &expected, 5.0e-3);
}

#[test]
fn chunked_relative_attention_matches_blocked_reference() {
    const SEQ: usize = 8;
    const HEADS: usize = 2;
    const HEAD_DIM: usize = 4;
    const DIM: usize = HEADS * HEAD_DIM;
    const LEFT: usize = 3;
    const SOFTCAP: f32 = 5.0;

    let q: Vec<_> = (0..SEQ * DIM)
        .map(|i| ((i * 7 % 19) as f32 - 9.0) * 0.07)
        .collect();
    let k: Vec<_> = (0..SEQ * DIM)
        .map(|i| ((i * 11 % 23) as f32 - 11.0) * 0.05)
        .collect();
    let v: Vec<_> = (0..SEQ * DIM)
        .map(|i| ((i * 5 % 17) as f32 - 8.0) * 0.11)
        .collect();
    let relative: Vec<_> = (0..LEFT * DIM)
        .map(|i| ((i * 3 % 13) as f32 - 6.0) * 0.03)
        .collect();

    let mut expected = vec![0.0; SEQ * DIM];
    for query in 0..SEQ {
        // The blocked Gemma mask permits distances `< left_context - 1`.
        let first_key = query.saturating_sub(LEFT - 2);
        for head in 0..HEADS {
            let mut scores = Vec::new();
            for key in first_key..=query {
                let distance = query - key;
                let relative_row = LEFT - 1 - distance;
                let mut score = 0.0;
                for d in 0..HEAD_DIM {
                    let column = head * HEAD_DIM + d;
                    let relative_value = relative[relative_row * DIM + column];
                    score += q[query * DIM + column] * (k[key * DIM + column] + relative_value);
                }
                scores.push(SOFTCAP * (score / SOFTCAP).tanh());
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denominator: f32 = scores.iter().map(|score| (score - max).exp()).sum();
            for (offset, key) in (first_key..=query).enumerate() {
                let weight = (scores[offset] - max).exp() / denominator;
                for d in 0..HEAD_DIM {
                    let column = head * HEAD_DIM + d;
                    expected[query * DIM + column] += weight * v[key * DIM + column];
                }
            }
        }
    }

    let mut graph = Graph::new();
    let qn = graph.input("q", &[SEQ, DIM]);
    let kn = graph.input("k", &[SEQ, DIM]);
    let vn = graph.input("v", &[SEQ, DIM]);
    let rn = graph.input("relative", &[LEFT, DIM]);
    let output = graph.chunked_relative_attention(
        qn,
        kn,
        vn,
        rn,
        HEADS as u32,
        HEAD_DIM as u32,
        LEFT as u32,
        SOFTCAP,
    );
    graph.set_outputs(vec![output]);
    let mut session = meganeura::build(&graph, meganeura::SessionConfig::inference_from_env()).0;
    session.set_input("q", &q);
    session.set_input("k", &k);
    session.set_input("v", &v);
    session.set_input("relative", &relative);
    session.step();
    session.wait();
    assert_close(&session.read_output(SEQ * DIM), &expected, 2.0e-5);
}

#[test]
fn cached_block_writes_only_valid_rows_and_selects_last() {
    for (max_seq, window) in [(6, 0), (96, 0), (96, 37)] {
        let block = 3;
        let dim = 4;
        let mut graph = Graph::new();
        let q = graph.input("q", &[block, 2 * dim]);
        let new_k = graph.input("new_k", &[block, dim]);
        let new_v = graph.input("new_v", &[block, dim]);
        let k_cache = graph.parameter("k", &[max_seq, dim]);
        let v_cache = graph.parameter("v", &[max_seq, dim]);
        let position = graph.input_u32("position", &[1]);
        let valid = graph.input_u32("valid", &[1]);
        let k_cache = graph.cache_write_prefix(new_k, k_cache, position, valid);
        let v_cache = graph.cache_write_prefix(new_v, v_cache, position, valid);
        let attended = graph.cached_block_attention(
            q, k_cache, v_cache, position, valid, 2, 1, dim as u32, window,
        );
        let output = graph.prefix_last(attended, valid);
        graph.set_outputs(vec![output]);

        let mut session =
            meganeura::build(&graph, meganeura::SessionConfig::inference_from_env()).0;
        session.set_input("q", &[0.0; 24]);
        session.set_input(
            "new_k",
            &[
                3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 1.0e6, 1.0e6, 1.0e6, 1.0e6,
            ],
        );
        session.set_input(
            "new_v",
            &[
                3.0, 6.0, 9.0, 12.0, 4.0, 8.0, 12.0, 16.0, 1.0e6, 1.0e6, 1.0e6, 1.0e6,
            ],
        );
        let mut initial_k = vec![0.0; max_seq * dim];
        let mut initial_v = vec![0.0; max_seq * dim];
        for row in 0..2 {
            for col in 0..dim {
                initial_k[row * dim + col] = (row + 1) as f32;
                initial_v[row * dim + col] = (row + 1) as f32 * (col + 1) as f32;
            }
        }
        session.set_parameter("k", &initial_k);
        session.set_parameter("v", &initial_v);
        session.set_input_u32("position", &[2]);
        session.set_input_u32("valid", &[2]);
        session.step();
        session.wait();
        let actual = session.read_output(2 * dim);
        // The second valid query sees cache rows 0..3. Q=0 makes them uniform.
        assert_close(&actual, &[2.5, 5.0, 7.5, 10.0, 2.5, 5.0, 7.5, 10.0], 1e-5);

        let queries: Vec<_> = (0..block * 2 * dim)
            .map(|i| (i as f32 * 0.7).sin())
            .collect();
        session.set_input("q", &queries);
        for (i, (k, v)) in initial_k.iter_mut().zip(&mut initial_v).enumerate() {
            *k = (i as f32 * 0.3).sin() * 4.0;
            *v = (i as f32 * 0.4).cos();
        }
        session.set_parameter("k", &initial_k);
        session.set_parameter("v", &initial_v);
        // Reuse partial storage from a long context in a short one. This also
        // covers unequal scores, multiple reduction tiles and ragged windows.
        for position in [max_seq - 2, 2] {
            session.set_input_u32("position", &[position as u32]);
            for row in 0..2 {
                for col in 0..dim {
                    initial_k[(position + row) * dim + col] = (row + 3) as f32;
                    initial_v[(position + row) * dim + col] = (row + 3) as f32 * (col + 1) as f32;
                }
            }
            let end = position + 2;
            let start = if window == 0 {
                0
            } else {
                end.saturating_sub(window as usize)
            };
            let mut expected = Vec::new();
            for head in 0..2 {
                let query = &queries[(2 + head) * dim..(3 + head) * dim];
                let scores: Vec<f64> = (start..end)
                    .map(|row| {
                        query
                            .iter()
                            .enumerate()
                            .map(|(col, &q)| f64::from(q) * f64::from(initial_k[row * dim + col]))
                            .sum::<f64>()
                            / (dim as f64).sqrt()
                    })
                    .collect();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<_> = scores.iter().map(|s| (s - max).exp()).collect();
                let sum: f64 = weights.iter().sum();
                for col in 0..dim {
                    expected.push(
                        (weights
                            .iter()
                            .enumerate()
                            .map(|(row, &w)| w * f64::from(initial_v[(start + row) * dim + col]))
                            .sum::<f64>()
                            / sum) as f32,
                    );
                }
            }
            session.step();
            session.wait();
            assert_close(&session.read_output(2 * dim), &expected, 1e-5);
        }
    }
}

#[test]
fn fused_rms_norm_add_matches_cpu_for_prefill_and_decode() {
    const EPS: f32 = 1.0e-6;

    for (rows, cols) in [(1, 1536), (64, 1536), (17, 3)] {
        let input = (0..rows * cols)
            .map(|index| ((index * 17 % 101) as f32 - 50.0) / 23.0)
            .collect::<Vec<_>>();
        let weight = (0..cols)
            .map(|index| 0.75 + (index * 7 % 19) as f32 / 32.0)
            .collect::<Vec<_>>();
        let residual = (0..rows * cols)
            .map(|index| ((index * 29 % 83) as f32 - 41.0) / 31.0)
            .collect::<Vec<_>>();
        let mut expected = vec![0.0; input.len()];
        for row in 0..rows {
            let offset = row * cols;
            let mean_square = input[offset..offset + cols]
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                / cols as f32;
            let inverse_rms = (mean_square + EPS).sqrt().recip();
            for (column, &scale) in weight.iter().enumerate() {
                let index = offset + column;
                expected[index] = input[index] * inverse_rms * scale + residual[index];
            }
        }

        let mut graph = Graph::new();
        let input_node = graph.input("input", &[rows, cols]);
        let weight_node = graph.parameter("weight", &[cols]);
        let residual_node = graph.input("residual", &[rows, cols]);
        let normalized = graph.rms_norm(input_node, weight_node, EPS);
        let output = graph.add(normalized, residual_node);
        graph.set_outputs(vec![output]);

        let mut session =
            meganeura::build(&graph, meganeura::SessionConfig::inference_from_env()).0;
        assert_eq!(
            session
                .plan()
                .dispatches
                .iter()
                .filter(|dispatch| {
                    dispatch.shader == meganeura::compile::ShaderEntry::RmsNormAdd
                })
                .count(),
            1,
            "rows={rows}: expected the fused RMSNorm+add dispatch"
        );
        session.set_input("input", &input);
        session.set_parameter("weight", &weight);
        session.set_input("residual", &residual);
        session.step();
        session.wait();
        assert_eq!(
            session.read_node(residual_node).unwrap(),
            residual,
            "rows={rows}: residual input was not uploaded"
        );
        assert_close(&session.read_output(input.len()), &expected, 2.0e-5);
    }
}
