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
            ..SessionConfig::default()
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
fn broadcast_inner_forward_and_gradient_match_explicit_repetition() {
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
fn tile_inner_forward_and_gradient_match_explicit_repetition() {
    const ROWS: usize = 513;
    const INNER: usize = 3;
    const REPEATS: usize = 8;
    let input_data = (0..ROWS * INNER)
        .map(|index| ((index * 31 % 127) as f32 - 63.0) * 0.017)
        .collect::<Vec<_>>();
    let weights_data = (0..ROWS * INNER * REPEATS)
        .map(|index| ((index * 17 % 41) as f32 - 20.0) * 0.031)
        .collect::<Vec<_>>();

    let mut graph = Graph::new();
    let input = graph.parameter("input", &[ROWS, INNER]);
    let repeated = graph.tile_inner(input, REPEATS);
    let weights = graph.input("weights", &[ROWS, INNER * REPEATS]);
    let weighted = graph.mul(repeated, weights);
    let loss = graph.sum_all(weighted);
    graph.set_outputs(vec![loss, repeated]);

    let mut session = meganeura::build_session(&graph);
    session.set_parameter("input", &input_data);
    session.set_input("weights", &weights_data);
    session.step();
    session.wait();

    let mut repeated_output = vec![0.0; ROWS * INNER * REPEATS];
    session.read_output_by_index(1, &mut repeated_output);
    let mut gradient = vec![0.0; ROWS * INNER];
    session.read_param_grad("input", &mut gradient);
    for row in 0..ROWS {
        for column in 0..INNER {
            let mut expected_gradient = 0.0_f32;
            for repeat in 0..REPEATS {
                let output_index = row * INNER * REPEATS + repeat * INNER + column;
                assert_eq!(
                    repeated_output[output_index].to_bits(),
                    input_data[row * INNER + column].to_bits(),
                );
                expected_gradient += weights_data[output_index];
            }
            assert_eq!(
                gradient[row * INNER + column].to_bits(),
                expected_gradient.to_bits(),
                "gradient mismatch at [{row}, {column}]",
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

#[test]
fn gathered_reduction_table_gradient_matches_row_scale_bit_exactly() {
    const ROWS: usize = 2049;
    const COLS: usize = 16;
    const VOCAB: usize = 2053;

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
            expected[output_row * COLS + col] =
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
