//! Contracts of the `fuse_reduction_chains` pass beyond numerical accuracy:
//! gathers actually fold into their reductions, and a fused reduction is
//! bit-identical to its explicit expansion. Accuracy against the f64
//! reference, under every lowering, is the `oracle` suite's job.

use meganeura::compile::{ShaderEntry, compile_with};
use meganeura::{CompileOptions, Graph, Mode, NodeId, SessionConfig};

/// Exact-equality checks between a fused path and its explicit expansion.
///
/// The contract is real on the author's hardware, but any adapter free to
/// reassociate the accumulation lands a last ULP out: lavapipe on
/// `normalize_inner_sum` and `pairwise_squared_distance`, Metal on the
/// narrow `sum_inner` scalar-order check and softplus's gradient.
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

fn repeated_rows(g: &mut Graph, x: NodeId, rows: usize, cols: usize, repeats: usize) -> NodeId {
    let flat = g.reshape(x, &[rows * cols, 1]);
    let expanded = g.broadcast_inner(flat, repeats);
    let shaped = g.reshape(expanded, &[rows, cols, repeats]);
    let transposed = g.transpose(shaped);
    g.reshape(transposed, &[rows * repeats, cols])
}

#[test]
fn repeated_rows_compile_without_copies() {
    for preserve_copy in [false, true] {
        let mut g = Graph::new();
        let x = g.input("x", &[7, 18]);
        let factors = g.input("factors", &[35, 18]);
        let repeated = repeated_rows(&mut g, x, 7, 18, 5);
        let product = g.mul(repeated, factors);
        let reduced = g.sum_inner(product);
        let second = g.sum_inner(repeated);
        g.set_outputs(if preserve_copy {
            vec![reduced, second, repeated]
        } else {
            vec![reduced, second]
        });
        let plan = compile_with(&g, &CompileOptions::default());
        let reductions = plan
            .dispatches
            .iter()
            .filter_map(|d| d.reduction())
            .collect::<Vec<_>>();
        assert_eq!(reductions.len(), 2);
        assert!(reductions.iter().all(|k| k.input_row_repeats.contains(&5)));
        assert_eq!(
            plan.dispatches
                .iter()
                .filter(|d| d.shader == ShaderEntry::Transpose)
                .count(),
            usize::from(preserve_copy)
        );
        assert_eq!(
            plan.dispatches
                .iter()
                .filter(|d| d.shader == ShaderEntry::GlobalAvgPoolGrad)
                .count(),
            usize::from(preserve_copy)
        );
    }
}

#[test]
fn repeated_rows_forward_and_gradients_match_f64() {
    let gpu = crate::support::gpu::gpu();
    if let Ok(expected) = std::env::var("MEGANEURA_EXPECT_DEVICE_NAME") {
        assert_eq!(gpu.device_information().device_name, expected);
    }
    crate::support::gpu::assert_protocol_headroom(&gpu);
    // A single column, and a multi-workgroup row count.
    for (rows, cols, repeats) in [(3, 1, 2), (259, 3, 5)] {
        let x_data = (0..rows * cols)
            .map(|i| ((i * 17 % 101) as f32 - 50.0) * 0.013)
            .collect::<Vec<_>>();
        let factor_data = (0..rows * repeats * cols)
            .map(|i| ((i * 29 % 97) as f32 - 48.0) * 0.021)
            .collect::<Vec<_>>();
        for fuse in [false, true] {
            let mut g = Graph::new();
            let x = g.parameter("x", &[rows, cols]);
            let factors = g.parameter("factors", &[rows * repeats, cols]);
            let repeated = repeated_rows(&mut g, x, rows, cols, repeats);
            let product = g.mul(repeated, factors);
            let reduced = g.sum_inner(product);
            let loss = g.sum_all(reduced);
            g.set_outputs(vec![loss, reduced]);
            let mut session = meganeura::build(
                &g,
                SessionConfig {
                    mode: Mode::Training,
                    options: CompileOptions {
                        fuse_dispatches: fuse,
                        ..Default::default()
                    },
                    ..crate::support::gpu::config()
                },
            )
            .0;
            session.set_parameter("x", &x_data);
            session.set_parameter("factors", &factor_data);
            session.step();
            session.wait();
            let mut actual = vec![0.0; rows * repeats];
            let mut dx = vec![0.0; x_data.len()];
            let mut df = vec![0.0; factor_data.len()];
            session.read_output_by_index(1, &mut actual);
            session.read_param_grad("x", &mut dx);
            session.read_param_grad("factors", &mut df);
            let mut expected_dx = vec![0.0_f64; x_data.len()];
            for row in 0..rows * repeats {
                let mut expected = 0.0_f64;
                for col in 0..cols {
                    let source = row / repeats * cols + col;
                    let target = row * cols + col;
                    expected += f64::from(x_data[source]) * f64::from(factor_data[target]);
                    expected_dx[source] += f64::from(factor_data[target]);
                    assert_eq!(df[target], x_data[source]);
                }
                assert!((f64::from(actual[row]) - expected).abs() < 2e-5 * (1.0 + expected.abs()));
            }
            for (actual, expected) in dx.into_iter().zip(expected_dx) {
                assert!((f64::from(actual) - expected).abs() < 2e-5 * (1.0 + expected.abs()));
            }
        }
    }
}

/// Narrow rows use one lane per row and preserve the scalar column-order sum
/// while the final partial workgroup exercises inactive packed rows. The
/// pointwise product is folded into the reduction prologue, and a plain row
/// sum matches the product with a ones column bit for bit.
#[test]
fn narrow_sum_inner_matches_scalar_f32_order_and_ones_matmul() {
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
    let mut graph = Graph::new();
    let a = graph.input("a", &[m, n]);
    let b = graph.input("b", &[m, n]);
    let product = graph.mul(a, b);
    let products = graph.sum_inner(product);
    let ones = graph.constant(vec![1.0; n], &[n, 1]);
    let matmul = graph.matmul(a, ones);
    let packed = graph.sum_inner(a);
    graph.set_outputs(vec![products, matmul, packed]);
    let mut session = meganeura::build(
        &graph,
        SessionConfig {
            mode: Mode::Inference,
            options: CompileOptions {
                fuse_dispatches: true,
                ..CompileOptions::default()
            },
            ..crate::support::gpu::config()
        },
    )
    .0;
    session.set_input("a", &a_data);
    session.set_input("b", &b_data);
    session.step();
    session.wait();
    let mut outputs = [vec![0.0; m], vec![0.0; m], vec![0.0; m]];
    for (index, output) in outputs.iter_mut().enumerate() {
        session.read_output_by_index(index, output);
    }
    let [products, matmul, packed] = outputs;
    for row in 0..m {
        let mut expected = 0.0_f32;
        for col in 0..n {
            let index = row * n + col;
            expected += a_data[index] * b_data[index];
        }
        assert_eq!(
            products[row].to_bits(),
            expected.to_bits(),
            "packed row {row}: {} != {expected}",
            products[row]
        );
        assert_eq!(
            packed[row].to_bits(),
            matmul[row].to_bits(),
            "packed row {row}: {} != matmul {}",
            packed[row],
            matmul[row]
        );
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

    // A wide `sum_inner` shares the loss: its gradient repeats each row's
    // weight too.
    const WIDE: usize = 16;
    let mut graph = Graph::new();
    let input = graph.parameter("input", &[ROWS, COLS]);
    let ones = graph.constant(vec![1.0; COLS], &[COLS, 1]);
    let reduced = graph.matmul(input, ones);
    let wide = graph.parameter("wide", &[ROWS, WIDE]);
    let wide_reduced = graph.sum_inner(wide);
    let weights = graph.input("weights", &[ROWS, 1]);
    let weighted = graph.mul(reduced, weights);
    let wide_weighted = graph.mul(wide_reduced, weights);
    let loss = graph.sum_all(weighted);
    let wide_loss = graph.sum_all(wide_weighted);
    let loss = graph.add(loss, wide_loss);
    graph.set_outputs(vec![loss, reduced]);

    let mut session = meganeura::build_session(&graph);
    session.set_parameter("input", &input_data);
    session.set_parameter("wide", &vec![0.0; ROWS * WIDE]);
    session.set_input("weights", &row_weights);
    session.step();
    session.wait();

    let mut reduced_output = vec![0.0; ROWS];
    session.read_output_by_index(1, &mut reduced_output);
    let mut gradient = vec![0.0; ROWS * COLS];
    session.read_param_grad("input", &mut gradient);
    let mut wide_gradient = vec![0.0; ROWS * WIDE];
    session.read_param_grad("wide", &mut wide_gradient);
    for (index, value) in wide_gradient.iter().enumerate() {
        assert_eq!(
            value.to_bits(),
            row_weights[index / WIDE].to_bits(),
            "sum_inner gradient mismatch at [{}, {}]",
            index / WIDE,
            index % WIDE
        );
    }

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
fn normalize_inner_sum_matches_explicit_forward_and_gradient() {
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
    // `oracle` checks the values against the f64 reference; this checks the
    // fused kernel against its explicit expansion.
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

    let opts = CompileOptions::default();
    let plan = compile_with(&g, &opts);

    let reductions: Vec<_> = plan
        .dispatches
        .iter()
        .filter(|d| d.reduction().is_some())
        .collect();
    assert_eq!(reductions.len(), 1, "expected exactly one fused reduction");
    let k = reductions[0].reduction().unwrap();
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
        .filter(|d| d.shader == ShaderEntry::Embedding && d.reduction().is_none())
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

    let opts = CompileOptions::default();
    let plan = compile_with(&graph, &opts);
    let reductions: Vec<_> = plan
        .dispatches
        .iter()
        .filter(|dispatch| dispatch.reduction().is_some())
        .collect();
    assert_eq!(reductions.len(), 2);
    for reduction in reductions {
        let kernel = reduction.reduction().unwrap();
        assert_eq!(kernel.n_per_elem, 3);
        assert_eq!(kernel.gather_elem, vec![true, false, false]);
        assert_eq!(reduction.input_buffers.len(), 4);
    }
    assert!(!plan.dispatches.iter().any(|dispatch| {
        dispatch.shader == ShaderEntry::Embedding && dispatch.reduction().is_none()
    }));
    assert!(!plan.dispatches.iter().any(|dispatch| {
        dispatch.pointwise().is_some() && dispatch.params[0] == (m * n) as u32
    }));
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
