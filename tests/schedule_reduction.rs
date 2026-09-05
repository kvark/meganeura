//! Parity tests for the `fuse_reduction_chains` pass: a graph's output
//! must match within f32 round-off whether the reduction runs unfused
//! (separate mul / embedding / sum_inner dispatches) or fused into a
//! single reduction kernel (pointwise prologue + gather streams).
//!
//! This is the end-to-end gate for the gather-fusion path: it exercises
//! the reduction gather codegen (schedule.rs) and the dynamic runtime
//! binding (runtime.rs) that only fire for fused reductions with
//! n_per_elem > 1 and/or gather streams.

use meganeura::compile::{ShaderEntry, compile_with};
use meganeura::{CompileOptions, Graph, Mode, NodeId, Session, SessionConfig};

/// Exact-equality checks between a fused path and its explicit expansion.
///
/// The contract is real on the author's hardware, but any adapter free to
/// reassociate the accumulation lands a last ULP out: lavapipe on
/// `normalize_inner_sum` and `pairwise_squared_distance`, Metal on
/// `narrow_sum_inner_matches_scalar_f32_order` and softplus's gradient.
/// CI sets `MEGANEURA_SKIP_BIT_EXACT=1` so the whole class is gated in one
/// place instead of a name-by-name skip list that grows every time another
/// adapter is added; local runs keep it at full strength.
fn skip_bit_exact() -> bool {
    if std::env::var("MEGANEURA_SKIP_BIT_EXACT").unwrap_or_default() == "1" {
        eprintln!("MEGANEURA_SKIP_BIT_EXACT set — skipping fused-vs-expanded exact equality");
        return true;
    }
    false
}

type F32Inputs = Vec<(&'static str, Vec<f32>)>;
type U32Inputs = Vec<(&'static str, Vec<u32>)>;
type BuildResult = (NodeId, F32Inputs, U32Inputs);

fn run(build: &dyn Fn(&mut Graph) -> BuildResult, n_out: usize, fuse: bool) -> Vec<f32> {
    let mut g = Graph::new();
    let (out, f_inputs, u_inputs) = build(&mut g);
    g.set_outputs(vec![out]);

    // Isolate the reduction-fusion pass; keep pointwise fusion on both
    // sides so only the reduction folding differs.
    let opts = CompileOptions {
        use_schedule_reduction: fuse,
        ..CompileOptions::default()
    };
    let mut session: Session = meganeura::build(
        &g,
        SessionConfig {
            mode: Mode::Inference,
            options: opts,
            ..SessionConfig::from_env()
        },
    )
    .0;
    for (name, data) in &f_inputs {
        session.set_input(name, data);
    }
    for (name, data) in &u_inputs {
        session.set_input_u32(name, data);
    }
    session.step();
    session.wait();
    session.read_output(n_out)
}

fn assert_parity(build: impl Fn(&mut Graph) -> BuildResult, n_out: usize) {
    let unfused = run(&build, n_out, false);
    let fused = run(&build, n_out, true);
    assert_eq!(unfused.len(), fused.len());
    for (i, (a, b)) in unfused.iter().zip(fused.iter()).enumerate() {
        assert!(
            (a - b).abs() <= a.abs().max(b.abs()) * 1e-6 + 1e-7,
            "reduction-fusion parity mismatch at [{i}]: unfused={a}, fused={b}",
        );
    }
}

/// Phase 1: `sum_inner(mul(a, b))` — fold a pointwise (binary) producer
/// into the reduction prologue. n_per_elem 1→2 (no gather) exercises the
/// dynamic binding path.
#[test]
fn sum_inner_of_mul_parity() {
    let m = 6usize;
    let n = 64usize;
    assert_parity(
        |g| {
            let a = g.input("a", &[m, n]);
            let b = g.input("b", &[m, n]);
            let prod = g.mul(a, b);
            let y = g.sum_inner(prod); // [m, 1]
            let a_data: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.13 - 2.0).collect();
            let b_data: Vec<f32> = (0..m * n).map(|i| (i as f32) * -0.07 + 1.0).collect();
            (y, vec![("a", a_data), ("b", b_data)], vec![])
        },
        m,
    );
}

/// Narrow rows use one lane per row and preserve the scalar column-order sum
/// while the final partial workgroup exercises inactive packed rows. The
/// pointwise product is folded into the reduction prologue.
#[test]
fn narrow_sum_inner_matches_scalar_f32_order() {
    if skip_bit_exact() {
        return;
    }
    let m = 259usize;
    let n = 9usize;
    let a_data: Vec<f32> = (0..m * n)
        .map(|i| ((i * 17 % 101) as f32 - 50.0) * 0.013)
        .collect();
    let b_data: Vec<f32> = (0..m * n)
        .map(|i| ((i * 29 % 97) as f32 - 48.0) * -0.021)
        .collect();
    let actual = run(
        &|g| {
            let a = g.input("a", &[m, n]);
            let b = g.input("b", &[m, n]);
            let product = g.mul(a, b);
            let output = g.sum_inner(product);
            (
                output,
                vec![("a", a_data.clone()), ("b", b_data.clone())],
                vec![],
            )
        },
        m,
        true,
    );
    for (row, &actual_row) in actual.iter().enumerate().take(m) {
        let mut expected = 0.0_f32;
        for col in 0..n {
            let index = row * n + col;
            expected += a_data[index] * b_data[index];
        }
        assert_eq!(
            actual_row.to_bits(),
            expected.to_bits(),
            "packed row {row}: {} != {expected}",
            actual_row
        );
    }
}

#[test]
fn narrow_sum_inner_matches_ones_matmul_bit_exactly() {
    if skip_bit_exact() {
        return;
    }
    let m = 259usize;
    let n = 9usize;
    let input_data: Vec<f32> = (0..m * n)
        .map(|i| ((i * 37 % 211) as f32 - 105.0) * 0.0073)
        .collect();
    let mut graph = Graph::new();
    let input = graph.input("input", &[m, n]);
    let ones = graph.constant(vec![1.0; n], &[n, 1]);
    let matmul = graph.matmul(input, ones);
    let packed = graph.sum_inner(input);
    graph.set_outputs(vec![matmul, packed]);
    let mut session = meganeura::build(
        &graph,
        SessionConfig {
            mode: Mode::Inference,
            ..SessionConfig::default()
        },
    )
    .0;
    session.set_input("input", &input_data);
    session.step();
    session.wait();
    let mut matmul_output = vec![0.0; m];
    let mut packed_output = vec![0.0; m];
    session.read_output_by_index(0, &mut matmul_output);
    session.read_output_by_index(1, &mut packed_output);
    for row in 0..m {
        assert_eq!(
            packed_output[row].to_bits(),
            matmul_output[row].to_bits(),
            "packed row {row}: {} != matmul {}",
            packed_output[row],
            matmul_output[row]
        );
    }
}

#[test]
fn sum_inner_gradient_repeats_each_row_bit_exactly() {
    if skip_bit_exact() {
        return;
    }
    const ROWS: usize = 513;
    const COLS: usize = 16;
    let row_weights = (0..ROWS)
        .map(|row| ((row * 31 % 127) as f32 - 63.0) * 0.017)
        .collect::<Vec<_>>();

    let mut graph = Graph::new();
    let input = graph.parameter("input", &[ROWS, COLS]);
    let reduced = graph.sum_inner(input);
    let weights = graph.input("weights", &[ROWS, 1]);
    let weighted = graph.mul(reduced, weights);
    let loss = graph.sum_all(weighted);
    graph.set_outputs(vec![loss]);

    let mut session = meganeura::build_session(&graph);
    session.set_parameter("input", &vec![0.0; ROWS * COLS]);
    session.set_input("weights", &row_weights);
    session.step();
    session.wait();

    let mut gradient = vec![0.0; ROWS * COLS];
    session.read_param_grad("input", &mut gradient);
    for row in 0..ROWS {
        for col in 0..COLS {
            assert_eq!(
                gradient[row * COLS + col].to_bits(),
                row_weights[row].to_bits(),
                "gradient mismatch at [{row}, {col}]"
            );
        }
    }
}

#[test]
fn unit_column_matmul_matches_generic_forward_and_gradient_bits() {
    if skip_bit_exact() {
        return;
    }
    const ROWS: usize = 513;
    const COLS: usize = 3;
    let input_data = (0..ROWS * COLS)
        .map(|index| ((index * 37 % 211) as f32 - 105.0) * 0.0073)
        .collect::<Vec<_>>();
    let row_weights = (0..ROWS)
        .map(|row| ((row * 31 % 127) as f32 - 63.0) * 0.017)
        .collect::<Vec<_>>();

    let mut graph = Graph::new();
    let input = graph.parameter("input", &[ROWS, COLS]);
    let ones = graph.constant(vec![1.0; COLS], &[COLS, 1]);
    let reduced = graph.matmul(input, ones);
    let weights = graph.input("weights", &[ROWS, 1]);
    let weighted = graph.mul(reduced, weights);
    let loss = graph.sum_all(weighted);
    graph.set_outputs(vec![loss, reduced]);

    let mut session = meganeura::build_session(&graph);
    session.set_parameter("input", &input_data);
    session.set_input("weights", &row_weights);
    session.step();
    session.wait();

    let mut reduced_output = vec![0.0; ROWS];
    session.read_output_by_index(1, &mut reduced_output);
    let mut gradient = vec![0.0; ROWS * COLS];
    session.read_param_grad("input", &mut gradient);

    // Supplying the same unit column at runtime prevents the compiler from
    // taking the constant specialization and provides the generic MatMul /
    // MatMulBT reference path.
    let mut reference_graph = Graph::new();
    let reference_input = reference_graph.parameter("input", &[ROWS, COLS]);
    let reference_ones = reference_graph.input("ones", &[COLS, 1]);
    let reference_reduced = reference_graph.matmul(reference_input, reference_ones);
    let reference_weights = reference_graph.input("weights", &[ROWS, 1]);
    let reference_weighted = reference_graph.mul(reference_reduced, reference_weights);
    let reference_loss = reference_graph.sum_all(reference_weighted);
    reference_graph.set_outputs(vec![reference_loss, reference_reduced]);

    let mut reference_session = meganeura::build_session(&reference_graph);
    reference_session.set_parameter("input", &input_data);
    reference_session.set_input("ones", &[1.0; COLS]);
    reference_session.set_input("weights", &row_weights);
    reference_session.step();
    reference_session.wait();

    let mut reference_output = vec![0.0; ROWS];
    reference_session.read_output_by_index(1, &mut reference_output);
    let mut reference_gradient = vec![0.0; ROWS * COLS];
    reference_session.read_param_grad("input", &mut reference_gradient);
    for row in 0..ROWS {
        let mut expected = 0.0_f32;
        for col in 0..COLS {
            expected += input_data[row * COLS + col];
            assert_eq!(
                gradient[row * COLS + col].to_bits(),
                row_weights[row].to_bits(),
                "gradient mismatch at [{row}, {col}]"
            );
            assert_eq!(
                gradient[row * COLS + col].to_bits(),
                reference_gradient[row * COLS + col].to_bits(),
                "specialized/generic gradient mismatch at [{row}, {col}]"
            );
        }
        assert_eq!(
            reduced_output[row].to_bits(),
            expected.to_bits(),
            "reduction mismatch at row {row}"
        );
        assert_eq!(
            reduced_output[row].to_bits(),
            reference_output[row].to_bits(),
            "specialized/generic forward mismatch at row {row}"
        );
    }
}

#[test]
fn broadcast_inner_forward_and_gradient_match_explicit_repetition() {
    if skip_bit_exact() {
        return;
    }
    const ROWS: usize = 513;
    const COLS: usize = 8;
    let input_data = (0..ROWS)
        .map(|row| ((row * 31 % 127) as f32 - 63.0) * 0.017)
        .collect::<Vec<_>>();
    let weights_data = (0..ROWS * COLS)
        .map(|index| ((index * 17 % 41) as f32 - 20.0) * 0.031)
        .collect::<Vec<_>>();

    let mut graph = Graph::new();
    let input = graph.parameter("input", &[ROWS, 1]);
    let repeated = graph.broadcast_inner(input, COLS);
    let weights = graph.input("weights", &[ROWS, COLS]);
    let weighted = graph.mul(repeated, weights);
    let loss = graph.sum_all(weighted);
    graph.set_outputs(vec![loss, repeated]);

    let mut session = meganeura::build_session(&graph);
    session.set_parameter("input", &input_data);
    session.set_input("weights", &weights_data);
    session.step();
    session.wait();

    let mut repeated_output = vec![0.0; ROWS * COLS];
    session.read_output_by_index(1, &mut repeated_output);
    let mut gradient = vec![0.0; ROWS];
    session.read_param_grad("input", &mut gradient);
    for row in 0..ROWS {
        let mut expected_gradient = 0.0_f32;
        for col in 0..COLS {
            let index = row * COLS + col;
            assert_eq!(repeated_output[index].to_bits(), input_data[row].to_bits());
            expected_gradient += weights_data[index];
        }
        assert_eq!(
            gradient[row].to_bits(),
            expected_gradient.to_bits(),
            "gradient mismatch at row {row}",
        );
    }
}

#[test]
fn global_avg_pool_gradient_keeps_spatial_normalization() {
    const BATCH: usize = 2;
    const CHANNELS: usize = 3;
    const SPATIAL: usize = 5;

    let mut graph = Graph::new();
    let input = graph.parameter("input", &[BATCH * CHANNELS * SPATIAL]);
    let pooled = graph.global_avg_pool(input, BATCH as u32, CHANNELS as u32, SPATIAL as u32);
    let loss = graph.sum_all(pooled);
    graph.set_outputs(vec![loss]);

    let mut session = meganeura::build_session(&graph);
    let input_data = [1.0; BATCH * CHANNELS * SPATIAL];
    session.set_parameter("input", &input_data);
    session.step();
    session.wait();

    let mut gradient = vec![0.0; BATCH * CHANNELS * SPATIAL];
    session.read_param_grad("input", &mut gradient);
    for value in gradient {
        assert!((value - (SPATIAL as f32).recip()).abs() <= 1.0e-7);
    }
}

#[test]
fn normalize_inner_sum_matches_explicit_forward_and_gradient() {
    if skip_bit_exact() {
        return;
    }
    const ROWS: usize = 257;
    const INNER: usize = 8;
    const FLOOR: f32 = 0.125;
    let mut input_data = (0..ROWS * INNER)
        .map(|index| ((index * 31 % 101) as f32 + 1.0) * 0.004)
        .collect::<Vec<_>>();
    input_data[0..INNER].fill(0.0);
    input_data[INNER..2 * INNER].fill(0.005);
    input_data[2 * INNER..3 * INNER].fill(FLOOR / INNER as f32);
    let weights_data = (0..ROWS * INNER)
        .map(|index| ((index * 17 % 41) as f32 - 20.0) * 0.031)
        .collect::<Vec<_>>();

    let mut graph = Graph::new();
    let input = graph.parameter("input", &[ROWS, INNER]);
    let normalized = graph.normalize_inner_sum(input, FLOOR);
    let weights = graph.input("weights", &[ROWS, INNER]);
    let weighted = graph.mul(normalized, weights);
    let loss = graph.sum_all(weighted);
    graph.set_outputs(vec![loss, normalized]);

    let mut session = meganeura::build_session(&graph);
    session.set_parameter("input", &input_data);
    session.set_input("weights", &weights_data);
    session.step();
    session.wait();

    let mut normalized_output = vec![0.0; ROWS * INNER];
    session.read_output_by_index(1, &mut normalized_output);
    let mut gradient = vec![0.0; ROWS * INNER];
    session.read_param_grad("input", &mut gradient);
    for row in 0..ROWS {
        let begin = row * INNER;
        let mut sum = 0.0_f32;
        for column in 0..INNER {
            let index = begin + column;
            sum += input_data[index];
        }
        let above_floor = sum - FLOOR;
        let denominator = above_floor.max(0.0) + FLOOR;
        let inverse = 1.0 / denominator;
        let negative_inverse_squared = -(inverse * inverse);
        let mut denominator_gradient = 0.0_f32;
        if above_floor > 0.0 {
            for column in 0..INNER {
                let index = begin + column;
                let reciprocal_gradient = weights_data[index] * input_data[index];
                denominator_gradient += reciprocal_gradient * negative_inverse_squared;
            }
        }
        for column in 0..INNER {
            let index = begin + column;
            let expected_output = input_data[index] * inverse;
            assert!(
                (normalized_output[index] - expected_output).abs()
                    <= normalized_output[index].abs().max(expected_output.abs()) * 2.0e-6 + 2.0e-7,
                "normalized value mismatch at [{row}, {column}]: {} != {}",
                normalized_output[index],
                expected_output,
            );
            let expected_gradient = weights_data[index] * inverse + denominator_gradient;
            assert!(
                (gradient[index] - expected_gradient).abs()
                    <= gradient[index].abs().max(expected_gradient.abs()) * 2.0e-6 + 2.0e-7,
                "gradient mismatch at [{row}, {column}]: {} != {}",
                gradient[index],
                expected_gradient,
            );
        }
    }

    let mut expanded_graph = Graph::new();
    let expanded_input = expanded_graph.parameter("input", &[ROWS, INNER]);
    let sum = expanded_graph.sum_inner(expanded_input);
    let negative_floor = expanded_graph.constant(vec![-FLOOR; ROWS], &[ROWS, 1]);
    let above_floor = expanded_graph.add(sum, negative_floor);
    let above_floor = expanded_graph.relu(above_floor);
    let floor = expanded_graph.constant(vec![FLOOR; ROWS], &[ROWS, 1]);
    let denominator = expanded_graph.add(above_floor, floor);
    let denominator = expanded_graph.broadcast_inner(denominator, INNER);
    let expanded_normalized = expanded_graph.div(expanded_input, denominator);
    let expanded_weights = expanded_graph.input("weights", &[ROWS, INNER]);
    let expanded_weighted = expanded_graph.mul(expanded_normalized, expanded_weights);
    let expanded_loss = expanded_graph.sum_all(expanded_weighted);
    expanded_graph.set_outputs(vec![expanded_loss, expanded_normalized]);

    let mut expanded_session = meganeura::build_session(&expanded_graph);
    expanded_session.set_parameter("input", &input_data);
    expanded_session.set_input("weights", &weights_data);
    expanded_session.step();
    expanded_session.wait();
    let mut expanded_output = vec![0.0; ROWS * INNER];
    expanded_session.read_output_by_index(1, &mut expanded_output);
    let mut expanded_gradient = vec![0.0; ROWS * INNER];
    expanded_session.read_param_grad("input", &mut expanded_gradient);
    for index in 0..ROWS * INNER {
        assert_eq!(
            normalized_output[index].to_bits(),
            expanded_output[index].to_bits(),
            "expanded output mismatch at {index}: {} != {}",
            normalized_output[index],
            expanded_output[index],
        );
        assert_eq!(
            gradient[index].to_bits(),
            expanded_gradient[index].to_bits(),
            "expanded gradient mismatch at {index}: {} != {}",
            gradient[index],
            expanded_gradient[index],
        );
    }
}

#[test]
fn pairwise_squared_distance_matches_explicit_forward_and_gradients() {
    if skip_bit_exact() {
        return;
    }
    const ROWS: usize = 513;
    const INNER: usize = 3;
    const PAIRS: usize = 8;
    let left_data = (0..ROWS * INNER)
        .map(|index| ((index * 31 % 127) as f32 - 63.0) * 0.017)
        .collect::<Vec<_>>();
    let right_data = (0..ROWS * PAIRS * INNER)
        .map(|index| ((index * 19 % 109) as f32 - 54.0) * 0.013)
        .collect::<Vec<_>>();
    let weights_data = (0..ROWS * PAIRS)
        .map(|index| ((index * 17 % 41) as f32 - 20.0) * 0.031)
        .collect::<Vec<_>>();

    let mut graph = Graph::new();
    let left = graph.parameter("left", &[ROWS, INNER]);
    let right = graph.parameter("right", &[ROWS * PAIRS, INNER]);
    let distances = graph.pairwise_squared_distance(left, right, PAIRS);
    let weights = graph.input("weights", &[ROWS, PAIRS]);
    let weighted = graph.mul(distances, weights);
    let loss = graph.sum_all(weighted);
    graph.set_outputs(vec![loss, distances]);

    let mut session = meganeura::build_session(&graph);
    session.set_parameter("left", &left_data);
    session.set_parameter("right", &right_data);
    session.set_input("weights", &weights_data);
    session.step();
    session.wait();

    let mut distances_output = vec![0.0; ROWS * PAIRS];
    session.read_output_by_index(1, &mut distances_output);
    let mut left_gradient = vec![0.0; ROWS * INNER];
    session.read_param_grad("left", &mut left_gradient);
    let mut right_gradient = vec![0.0; ROWS * PAIRS * INNER];
    session.read_param_grad("right", &mut right_gradient);

    for row in 0..ROWS {
        for pair in 0..PAIRS {
            let pair_index = row * PAIRS + pair;
            let mut expected_distance = 0.0_f32;
            for column in 0..INNER {
                let left_index = row * INNER + column;
                let right_index = pair_index * INNER + column;
                let delta = left_data[left_index] - right_data[right_index];
                expected_distance += delta * delta;
                let term = weights_data[pair_index] * delta;
                let expected_gradient = -(term + term);
                assert!(
                    (right_gradient[right_index] - expected_gradient).abs()
                        <= right_gradient[right_index]
                            .abs()
                            .max(expected_gradient.abs())
                            * 1.0e-6
                            + 1.0e-7,
                    "right gradient mismatch at [{row}, {pair}, {column}]: {} != {}",
                    right_gradient[right_index],
                    expected_gradient,
                );
            }
            assert!(
                (distances_output[pair_index] - expected_distance).abs()
                    <= distances_output[pair_index]
                        .abs()
                        .max(expected_distance.abs())
                        * 1.0e-6
                        + 1.0e-7,
                "distance mismatch at [{row}, {pair}]: {} != {}",
                distances_output[pair_index],
                expected_distance,
            );
        }
        for column in 0..INNER {
            let left_index = row * INNER + column;
            let mut expected_gradient = 0.0_f32;
            for pair in 0..PAIRS {
                let pair_index = row * PAIRS + pair;
                let right_index = pair_index * INNER + column;
                let delta = left_data[left_index] - right_data[right_index];
                let term = weights_data[pair_index] * delta;
                expected_gradient += term + term;
            }
            assert!(
                (left_gradient[left_index] - expected_gradient).abs()
                    <= left_gradient[left_index].abs().max(expected_gradient.abs()) * 1.0e-6
                        + 1.0e-7,
                "left gradient mismatch at [{row}, {column}]: {} != {}",
                left_gradient[left_index],
                expected_gradient,
            );
        }
    }
}

#[test]
fn pairwise_vector_rejection_matches_explicit_forward_and_gradients() {
    const ROWS: usize = 257;
    const INNER: usize = 3;
    const PAIRS: usize = 8;
    let vectors_data = (0..ROWS * PAIRS * INNER)
        .map(|index| ((index * 31 % 127) as f32 - 63.0) * 0.017)
        .collect::<Vec<_>>();
    let directions_data = (0..ROWS * INNER)
        .map(|index| ((index * 19 % 109) as f32 - 54.0) * 0.013)
        .collect::<Vec<_>>();
    let weights_data = (0..ROWS * PAIRS * INNER)
        .map(|index| ((index * 17 % 41) as f32 - 20.0) * 0.031)
        .collect::<Vec<_>>();

    let mut graph = Graph::new();
    let vectors = graph.parameter("vectors", &[ROWS * PAIRS, INNER]);
    let directions = graph.parameter("directions", &[ROWS, INNER]);
    let rejected = graph.pairwise_vector_rejection(vectors, directions, PAIRS);
    let weights = graph.input("weights", &[ROWS * PAIRS, INNER]);
    let weighted = graph.mul(rejected, weights);
    let loss = graph.sum_all(weighted);
    graph.set_outputs(vec![loss, rejected]);

    let mut session = meganeura::build_session(&graph);
    session.set_parameter("vectors", &vectors_data);
    session.set_parameter("directions", &directions_data);
    session.set_input("weights", &weights_data);
    session.step();
    session.wait();

    let mut rejected_output = vec![0.0; ROWS * PAIRS * INNER];
    session.read_output_by_index(1, &mut rejected_output);
    let mut vectors_gradient = vec![0.0; ROWS * PAIRS * INNER];
    session.read_param_grad("vectors", &mut vectors_gradient);
    let mut directions_gradient = vec![0.0; ROWS * INNER];
    session.read_param_grad("directions", &mut directions_gradient);

    for row in 0..ROWS {
        let direction_begin = row * INNER;
        for pair in 0..PAIRS {
            let vector_begin = (row * PAIRS + pair) * INNER;
            let mut component = 0.0_f32;
            let mut grad_component = 0.0_f32;
            for column in 0..INNER {
                component +=
                    vectors_data[vector_begin + column] * directions_data[direction_begin + column];
                grad_component +=
                    weights_data[vector_begin + column] * directions_data[direction_begin + column];
            }
            grad_component = -grad_component;
            for column in 0..INNER {
                let index = vector_begin + column;
                let projection = directions_data[direction_begin + column] * component;
                let expected = vectors_data[index] + (-projection);
                assert!(
                    (rejected_output[index] - expected).abs()
                        <= rejected_output[index].abs().max(expected.abs()) * 1.0e-6 + 2.0e-7,
                    "rejection mismatch at [{row}, {pair}, {column}]: {} != {}",
                    rejected_output[index],
                    expected,
                );

                let indirect = directions_data[direction_begin + column] * grad_component;
                let expected_gradient = weights_data[index] + indirect;
                assert!(
                    (vectors_gradient[index] - expected_gradient).abs()
                        <= vectors_gradient[index].abs().max(expected_gradient.abs()) * 1.0e-6
                            + 2.0e-7,
                    "vector gradient mismatch at [{row}, {pair}, {column}]: {} != {}",
                    vectors_gradient[index],
                    expected_gradient,
                );
            }
        }

        for output_column in 0..INNER {
            let mut expected_gradient = 0.0_f32;
            for pair in 0..PAIRS {
                let vector_begin = (row * PAIRS + pair) * INNER;
                let mut component = 0.0_f32;
                let mut grad_component = 0.0_f32;
                for column in 0..INNER {
                    component += vectors_data[vector_begin + column]
                        * directions_data[direction_begin + column];
                    grad_component += weights_data[vector_begin + column]
                        * directions_data[direction_begin + column];
                }
                let direct = (-weights_data[vector_begin + output_column]) * component;
                let indirect = vectors_data[vector_begin + output_column] * (-grad_component);
                expected_gradient += direct + indirect;
            }
            let index = direction_begin + output_column;
            assert!(
                (directions_gradient[index] - expected_gradient).abs()
                    <= directions_gradient[index]
                        .abs()
                        .max(expected_gradient.abs())
                        * 1.0e-6
                        + 2.0e-7,
                "direction gradient mismatch at [{row}, {output_column}]: {} != {}",
                directions_gradient[index],
                expected_gradient,
            );
        }
    }
}

/// Phase 2: `sum_inner(mul(embedding(idx, table), basis))` — the SH colour
/// path. Folds the gather (Embedding) into the reduction as a gather
/// stream. Exercises the full gather codegen + dynamic binding.
#[test]
fn sum_inner_of_gather_times_basis_parity() {
    let vocab = 4usize;
    let m = 6usize; // outer (P*L)
    let n = 64usize; // inner (K) = table hidden
    assert_parity(
        |g| {
            let idx = g.input_u32("idx", &[m]);
            let table = g.input("table", &[vocab, n]);
            let basis = g.input("basis", &[m, n]);
            let gathered = g.embedding(idx, table); // [m, n]
            let prod = g.mul(gathered, basis); // [m, n]
            let y = g.sum_inner(prod); // [m, 1]
            let table_data: Vec<f32> = (0..vocab * n).map(|i| (i as f32) * 0.21 - 1.0).collect();
            let basis_data: Vec<f32> = (0..m * n).map(|i| (i as f32) * 0.05 + 0.3).collect();
            let idx_data: Vec<u32> = (0..m).map(|i| (i * 3 % vocab) as u32).collect();
            (
                y,
                vec![("table", table_data), ("basis", basis_data)],
                vec![("idx", idx_data)],
            )
        },
        m,
    );
}

/// White-box: the two-gather graph must actually COLLAPSE — one reduction
/// dispatch with two gather streams, and zero Embedding / standalone-mul
/// dispatches left. Guards against the parity tests passing on a silent
/// no-op fusion.
#[test]
fn two_gather_reduction_actually_fuses() {
    let vocab = 5usize;
    let m = 6usize;
    let n = 64usize;
    let mut g = Graph::new();
    let idx_a = g.input_u32("idx_a", &[m]);
    let idx_b = g.input_u32("idx_b", &[m]);
    let ta = g.input("ta", &[vocab, n]);
    let tb = g.input("tb", &[vocab, n]);
    let ga = g.embedding(idx_a, ta);
    let gb = g.embedding(idx_b, tb);
    let prod = g.mul(ga, gb);
    let y = g.sum_inner(prod);
    g.set_outputs(vec![y]);

    let opts = CompileOptions {
        use_schedule_reduction: true,
        ..CompileOptions::default()
    };
    let plan = compile_with(&g, &opts);

    let reductions: Vec<_> = plan
        .dispatches
        .iter()
        .filter(|d| d.reduction.is_some())
        .collect();
    assert_eq!(reductions.len(), 1, "expected exactly one fused reduction");
    let k = reductions[0].reduction.as_ref().unwrap();
    assert_eq!(
        k.n_per_elem, 2,
        "mul producer should fold to 2 per-elem streams"
    );
    assert_eq!(
        k.gather_elem,
        vec![true, true],
        "both embedding producers should fold as gather streams"
    );
    // Both indices + both tables present as input buffers (4 streams).
    assert_eq!(reductions[0].input_buffers.len(), 4);
    // No standalone Embedding dispatches remain.
    let embeds = plan
        .dispatches
        .iter()
        .filter(|d| d.shader == ShaderEntry::Embedding && d.reduction.is_none())
        .count();
    assert_eq!(embeds, 0, "embedding dispatches should be folded away");
}

#[test]
fn shared_gather_and_offset_fold_into_each_reduction() {
    let vocab = 5usize;
    let m = 6usize;
    let n = 64usize;
    let mut graph = Graph::new();
    let indices = graph.input_u32("indices", &[m]);
    let table = graph.input("table", &[vocab, n]);
    let offset = graph.input("offset", &[m, n]);
    let factors_a = graph.input("factors_a", &[m, n]);
    let factors_b = graph.input("factors_b", &[m, n]);
    let gathered = graph.embedding(indices, table);
    let relative = graph.add(gathered, offset);
    let terms_a = graph.mul(relative, factors_a);
    let reduced_a = graph.sum_inner(terms_a);
    let terms_b = graph.mul(relative, factors_b);
    let reduced_b = graph.sum_inner(terms_b);
    let output = graph.add(reduced_a, reduced_b);
    graph.set_outputs(vec![output]);

    let opts = CompileOptions {
        use_schedule_reduction: true,
        ..CompileOptions::default()
    };
    let plan = compile_with(&graph, &opts);
    let reductions: Vec<_> = plan
        .dispatches
        .iter()
        .filter(|dispatch| dispatch.reduction.is_some())
        .collect();
    assert_eq!(reductions.len(), 2);
    for reduction in reductions {
        let kernel = reduction.reduction.as_ref().unwrap();
        assert_eq!(kernel.n_per_elem, 3);
        assert_eq!(kernel.gather_elem, vec![true, false, false]);
        assert_eq!(reduction.input_buffers.len(), 4);
    }
    assert!(!plan.dispatches.iter().any(|dispatch| {
        dispatch.shader == ShaderEntry::Embedding && dispatch.reduction.is_none()
    }));
    assert!(!plan.dispatches.iter().any(|dispatch| {
        dispatch.shader == ShaderEntry::Add && dispatch.params[0] == (m * n) as u32
    }));
}

#[test]
fn shared_gather_and_offset_reduction_parity() {
    let vocab = 5usize;
    let m = 6usize;
    let n = 64usize;
    assert_parity(
        |graph| {
            let indices = graph.input_u32("indices", &[m]);
            let table = graph.input("table", &[vocab, n]);
            let offset = graph.input("offset", &[m, n]);
            let factors_a = graph.input("factors_a", &[m, n]);
            let factors_b = graph.input("factors_b", &[m, n]);
            let gathered = graph.embedding(indices, table);
            let relative = graph.add(gathered, offset);
            let terms_a = graph.mul(relative, factors_a);
            let reduced_a = graph.sum_inner(terms_a);
            let terms_b = graph.mul(relative, factors_b);
            let reduced_b = graph.sum_inner(terms_b);
            let output = graph.add(reduced_a, reduced_b);
            let table_data = (0..vocab * n)
                .map(|index| index as f32 * 0.021 - 1.0)
                .collect();
            let offset_data = (0..m * n)
                .map(|index| index as f32 * -0.013 + 0.5)
                .collect();
            let factors_a_data = (0..m * n).map(|index| index as f32 * 0.005 + 0.3).collect();
            let factors_b_data = (0..m * n)
                .map(|index| index as f32 * -0.007 + 0.7)
                .collect();
            let indices_data = (0..m).map(|row| (row * 3 % vocab) as u32).collect();
            (
                output,
                vec![
                    ("table", table_data),
                    ("offset", offset_data),
                    ("factors_a", factors_a_data),
                    ("factors_b", factors_b_data),
                ],
                vec![("indices", indices_data)],
            )
        },
        m,
    );
}

#[test]
fn shared_gather_reductions_accumulate_the_table_gradient() {
    let vocab = 5usize;
    let m = 6usize;
    let n = 64usize;
    let mut graph = Graph::new();
    let indices = graph.input_u32("indices", &[m]);
    let table = graph.parameter("table", &[vocab, n]);
    let offset = graph.input("offset", &[m, n]);
    let factors_a = graph.input("factors_a", &[m, n]);
    let factors_b = graph.input("factors_b", &[m, n]);
    let gathered = graph.embedding(indices, table);
    let relative = graph.add(gathered, offset);
    let terms_a = graph.mul(relative, factors_a);
    let reduced_a = graph.sum_inner(terms_a);
    let terms_b = graph.mul(relative, factors_b);
    let reduced_b = graph.sum_inner(terms_b);
    let rows = graph.add(reduced_a, reduced_b);
    let loss = graph.sum_all(rows);
    graph.set_outputs(vec![loss]);

    let indices_data = (0..m)
        .map(|row| (row * 3 % vocab) as u32)
        .collect::<Vec<_>>();
    let factors_a_data = (0..m * n)
        .map(|index| index as f32 * 0.005 + 0.3)
        .collect::<Vec<_>>();
    let factors_b_data = (0..m * n)
        .map(|index| index as f32 * -0.007 + 0.7)
        .collect::<Vec<_>>();
    let mut session = meganeura::build_session(&graph);
    session.set_parameter("table", &vec![0.0; vocab * n]);
    session.set_input("offset", &vec![0.0; m * n]);
    session.set_input("factors_a", &factors_a_data);
    session.set_input("factors_b", &factors_b_data);
    session.set_input_u32("indices", &indices_data);
    session.set_learning_rate(0.0);
    session.step();
    session.wait();

    let mut actual = vec![0.0; vocab * n];
    session.read_param_grad("table", &mut actual);
    let mut expected = vec![0.0; vocab * n];
    for row in 0..m {
        let table_row = indices_data[row] as usize;
        for column in 0..n {
            expected[table_row * n + column] +=
                factors_a_data[row * n + column] + factors_b_data[row * n + column];
        }
    }
    for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            (actual - expected).abs() <= expected.abs() * 1.0e-6 + 1.0e-6,
            "table gradient mismatch at {index}: {actual} != {expected}"
        );
    }
}

/// Two gather streams (both operands gathered) — closest to the real SH
/// case where both coeff and per-step basis are indexed loads.
#[test]
fn sum_inner_of_two_gathers_parity() {
    let vocab = 5usize;
    let m = 6usize;
    let n = 64usize;
    assert_parity(
        |g| {
            let idx_a = g.input_u32("idx_a", &[m]);
            let idx_b = g.input_u32("idx_b", &[m]);
            let ta = g.input("ta", &[vocab, n]);
            let tb = g.input("tb", &[vocab, n]);
            let ga = g.embedding(idx_a, ta);
            let gb = g.embedding(idx_b, tb);
            let prod = g.mul(ga, gb);
            let y = g.sum_inner(prod);
            let ta_data: Vec<f32> = (0..vocab * n).map(|i| (i as f32) * 0.11 - 0.5).collect();
            let tb_data: Vec<f32> = (0..vocab * n).map(|i| (i as f32) * -0.09 + 0.7).collect();
            let ia: Vec<u32> = (0..m).map(|i| (i % vocab) as u32).collect();
            let ib: Vec<u32> = (0..m).map(|i| ((i * 2 + 1) % vocab) as u32).collect();
            (
                y,
                vec![("ta", ta_data), ("tb", tb_data)],
                vec![("idx_a", ia), ("idx_b", ib)],
            )
        },
        m,
    );
}

/// LayerNorm forward: the two-accumulator archetype (sum + sum-of-squares
/// in one pass) must match the hand-written layer_norm.wgsl shader.
#[test]
fn layer_norm_archetype_parity() {
    assert_parity(
        |g| {
            let x = g.input("x", &[5, 96]);
            let w = g.input("w", &[96]);
            let b = g.input("b", &[96]);
            let y = g.layer_norm(x, w, b, 1e-5);
            let x_data: Vec<f32> = (0..5 * 96)
                .map(|i| ((i * 37 % 101) as f32) * 0.11 - 4.7)
                .collect();
            let w_data: Vec<f32> = (0..96).map(|i| 0.5 + ((i % 7) as f32) * 0.2).collect();
            let b_data: Vec<f32> = (0..96).map(|i| ((i % 5) as f32) * 0.3 - 0.6).collect();
            (y, vec![("x", x_data), ("w", w_data), ("b", b_data)], vec![])
        },
        5 * 96,
    );
}

#[test]
fn gathered_reduction_table_gradient_matches_row_scale_bit_exactly() {
    if skip_bit_exact() {
        return;
    }
    const ROWS: usize = 2049;
    const COLS: usize = 16;
    const VOCAB: usize = 257;

    let mut graph = Graph::new();
    let indices = graph.input_u32("indices", &[ROWS]);
    let table = graph.parameter("table", &[VOCAB, COLS]);
    let factors = graph.input("factors", &[ROWS, COLS]);
    let gathered = graph.embedding(indices, table);
    let terms = graph.mul(gathered, factors);
    let reduced = graph.sum_inner(terms);
    let row_scale = graph.input("row_scale", &[ROWS, 1]);
    let weighted = graph.mul(reduced, row_scale);
    let loss = graph.sum_all(weighted);
    graph.set_outputs(vec![loss]);

    let indices_data = (0..ROWS)
        .map(|row| ((row * 37) % VOCAB) as u32)
        .collect::<Vec<_>>();
    let factors_data = (0..ROWS * COLS)
        .map(|index| {
            let magnitude = (index * 5 % 8 + 1) as f32 * 0.125;
            if index & 8 == 0 {
                magnitude
            } else {
                -magnitude
            }
        })
        .collect::<Vec<_>>();
    let row_scale_data = (0..ROWS)
        .map(|row| {
            let magnitude = (row * 7 % 4 + 1) as f32 * 0.25;
            if row & 4 == 0 { magnitude } else { -magnitude }
        })
        .collect::<Vec<_>>();

    let mut session = meganeura::build_session(&graph);
    session.set_parameter("table", &vec![0.0; VOCAB * COLS]);
    session.set_input_u32("indices", &indices_data);
    session.set_input("factors", &factors_data);
    session.set_input("row_scale", &row_scale_data);
    session.step();
    session.wait();

    let mut actual = vec![0.0; VOCAB * COLS];
    session.read_param_grad("table", &mut actual);
    let mut expected = vec![0.0; VOCAB * COLS];
    for row in 0..ROWS {
        let output_row = indices_data[row] as usize;
        for col in 0..COLS {
            expected[output_row * COLS + col] +=
                row_scale_data[row] * factors_data[row * COLS + col];
        }
    }
    for (index, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "table gradient mismatch at flat index {index}: {actual} != {expected}"
        );
    }
}
