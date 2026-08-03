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
