//! Parity tests: a graph's output must match within f32 round-off whether
//! pointwise ops run through the hand-written `unary.wgsl` / `binary.wgsl`
//! shaders or the schedule-template codegen path. This is the gate that
//! lets us retire hand-written pointwise entries one at a time.

use meganeura::{CompileOptions, Graph, Mode, NodeId, Session, SessionConfig};

fn inference(g: &Graph, opts: CompileOptions) -> Session {
    meganeura::build(
        g,
        SessionConfig {
            mode: Mode::Inference,
            options: opts,
            ..SessionConfig::from_env()
        },
    )
    .0
}

/// Build+run a graph once with the given schedule-pointwise setting.
///
/// Baseline (`opts_on=false`) forces `use_schedule_pointwise=false` so we
/// compare against the hand-written shader path regardless of the
/// library's default.
fn run_once(
    input_names: &[&str],
    inputs: &[&[f32]],
    n_out: usize,
    build: &dyn Fn(&mut Graph, &[NodeId]) -> NodeId,
    opts_on: bool,
) -> Vec<f32> {
    run_once_with(input_names, inputs, n_out, build, |o| {
        o.use_schedule_pointwise = opts_on;
    })
}

fn run_once_with(
    input_names: &[&str],
    inputs: &[&[f32]],
    n_out: usize,
    build: &dyn Fn(&mut Graph, &[NodeId]) -> NodeId,
    configure: impl FnOnce(&mut CompileOptions),
) -> Vec<f32> {
    assert_eq!(input_names.len(), inputs.len());
    let mut g = Graph::new();
    let ids: Vec<NodeId> = input_names
        .iter()
        .zip(inputs.iter())
        .map(|(name, data)| g.input(name, &[data.len()]))
        .collect();
    let y = build(&mut g, &ids);
    g.set_outputs(vec![y]);

    let mut opts = CompileOptions::default();
    configure(&mut opts);
    let mut session = inference(&g, opts);
    for (name, data) in input_names.iter().zip(inputs.iter()) {
        session.set_input(name, data);
    }
    session.step();
    session.wait();
    session.read_output(n_out)
}

fn assert_parity(
    input_names: &[&str],
    inputs: &[&[f32]],
    n_out: usize,
    build: impl Fn(&mut Graph, &[NodeId]) -> NodeId,
) {
    let default = run_once(input_names, inputs, n_out, &build, false);
    let schedule = run_once(input_names, inputs, n_out, &build, true);
    assert_eq!(default.len(), schedule.len());
    for (i, (a, b)) in default.iter().zip(schedule.iter()).enumerate() {
        assert!(
            (a - b).abs() <= a.abs().max(b.abs()) * 1e-6 + 1e-7,
            "parity mismatch at [{i}]: default={a}, schedule={b}",
        );
    }
}

// ---- Unary ops ----

#[test]
fn relu_parity() {
    let input: Vec<f32> = (0..256).map(|i| (i as f32) * 0.1 - 12.0).collect();
    assert_parity(&["x"], &[&input], 256, |g, xs| g.relu(xs[0]));
}

#[test]
fn silu_parity() {
    let input: Vec<f32> = (0..256).map(|i| (i as f32) * 0.1 - 12.0).collect();
    assert_parity(&["x"], &[&input], 256, |g, xs| g.silu(xs[0]));
}

#[test]
fn exp_parity() {
    let input: Vec<f32> = (0..256).map(|i| (i as f32) * 0.04 - 5.0).collect();
    assert_parity(&["x"], &[&input], 256, |g, xs| g.exp(xs[0]));
}

#[test]
fn exp_matches_cpu_forward_and_gradient() {
    const LEN: usize = 513;
    let input = (0..LEN)
        .map(|index| ((index * 19 % 101) as f32 - 50.0) * 0.06)
        .collect::<Vec<_>>();
    let weights = (0..LEN)
        .map(|index| ((index * 13 % 47) as f32 - 23.0) * 0.03)
        .collect::<Vec<_>>();

    let mut graph = Graph::new();
    let x = graph.parameter("x", &[LEN]);
    let y = graph.exp(x);
    let weight = graph.input("weight", &[LEN]);
    let weighted = graph.mul(y, weight);
    let loss = graph.sum_all(weighted);
    graph.set_outputs(vec![loss, y]);

    let mut session = meganeura::build_session(&graph);
    session.set_parameter("x", &input);
    session.set_input("weight", &weights);
    session.step();
    session.wait();

    let mut output = vec![0.0_f32; LEN];
    session.read_output_by_index(1, &mut output);
    let mut gradient = vec![0.0_f32; LEN];
    session.read_param_grad("x", &mut gradient);
    for index in 0..LEN {
        let expected = input[index].exp();
        let expected_gradient = weights[index] * expected;
        assert!(
            (output[index] - expected).abs()
                <= output[index].abs().max(expected.abs()) * 1.0e-6 + 1.0e-7,
            "forward mismatch at {index}: {} != {expected}",
            output[index],
        );
        assert!(
            (gradient[index] - expected_gradient).abs()
                <= gradient[index].abs().max(expected_gradient.abs()) * 1.0e-6 + 1.0e-7,
            "gradient mismatch at {index}: {} != {expected_gradient}",
            gradient[index],
        );
    }
}

#[test]
fn softplus_matches_stable_cpu_forward_and_gradient() {
    const BETA: f32 = 10.0;
    const LEN: usize = 513;
    let mut input = (0..LEN)
        .map(|index| ((index * 19 % 101) as f32 - 50.0) * 0.06)
        .collect::<Vec<_>>();
    input[LEN / 2] = 0.0;
    let weights = (0..LEN)
        .map(|index| ((index * 13 % 47) as f32 - 23.0) * 0.03)
        .collect::<Vec<_>>();

    let mut graph = Graph::new();
    let x = graph.parameter("x", &[LEN]);
    let y = graph.softplus(x, BETA);
    let weight = graph.input("weight", &[LEN]);
    let weighted = graph.mul(y, weight);
    let loss = graph.sum_all(weighted);
    graph.set_outputs(vec![loss, y]);

    let mut session = meganeura::build_session(&graph);
    session.set_parameter("x", &input);
    session.set_input("weight", &weights);
    session.step();
    session.wait();

    let mut output = vec![0.0_f32; LEN];
    session.read_output_by_index(1, &mut output);
    let mut gradient = vec![0.0_f32; LEN];
    session.read_param_grad("x", &mut gradient);
    for index in 0..LEN {
        let scaled = BETA * input[index];
        let expected = (scaled.max(0.0) + (-scaled.abs()).exp().ln_1p()) / BETA;
        let expected_gradient = weights[index] / (1.0 + (-scaled).exp());
        assert!(
            (output[index] - expected).abs()
                <= output[index].abs().max(expected.abs()) * 1.0e-6 + 1.0e-7,
            "forward mismatch at {index}: {} != {expected}",
            output[index],
        );
        assert!(
            (gradient[index] - expected_gradient).abs()
                <= gradient[index].abs().max(expected_gradient.abs()) * 1.0e-6 + 1.0e-7,
            "gradient mismatch at {index}: {} != {expected_gradient}",
            gradient[index],
        );
    }
}

#[test]
fn softplus_preserves_expanded_gradient_bits() {
    const BETA: f32 = 10.0;
    const LEN: usize = 513;
    let input = (0..LEN)
        .map(|index| ((index * 19 % 101) as f32 - 50.0) * 0.06)
        .collect::<Vec<_>>();
    let weights = (0..LEN)
        .map(|index| ((index * 13 % 47) as f32 - 23.0) * 0.03)
        .collect::<Vec<_>>();

    let mut graph = Graph::new();
    let expanded_x = graph.parameter("expanded_x", &[LEN]);
    let beta = graph.constant(vec![BETA; LEN], &[LEN]);
    let scaled = graph.mul(expanded_x, beta);
    let positive = graph.relu(scaled);
    let magnitude = graph.abs(scaled);
    let probability = graph.sigmoid(magnitude);
    let log_probability = graph.log(probability);
    let correction = graph.neg(log_probability);
    let stable_sum = graph.add(positive, correction);
    let inverse_beta = graph.constant(vec![BETA.recip(); LEN], &[LEN]);
    let expanded = graph.mul(stable_sum, inverse_beta);

    let fused_x = graph.parameter("fused_x", &[LEN]);
    let fused = graph.softplus(fused_x, BETA);
    let weight = graph.input("weight", &[LEN]);
    let expanded_weighted = graph.mul(expanded, weight);
    let fused_weighted = graph.mul(fused, weight);
    let expanded_loss = graph.sum_all(expanded_weighted);
    let fused_loss = graph.sum_all(fused_weighted);
    let loss = graph.add(expanded_loss, fused_loss);
    graph.set_outputs(vec![loss, expanded, fused]);

    let mut session = meganeura::build_session(&graph);
    session.set_parameter("expanded_x", &input);
    session.set_parameter("fused_x", &input);
    session.set_input("weight", &weights);
    session.step();
    session.wait();

    let mut expanded_output = vec![0.0_f32; LEN];
    session.read_output_by_index(1, &mut expanded_output);
    let mut fused_output = vec![0.0_f32; LEN];
    session.read_output_by_index(2, &mut fused_output);
    for (expanded, fused) in expanded_output.iter().zip(&fused_output) {
        assert!((expanded - fused).abs() <= expanded.abs().max(fused.abs()) * 1.0e-6 + 1.0e-7);
    }

    let mut expanded_gradient = vec![0.0_f32; LEN];
    session.read_param_grad("expanded_x", &mut expanded_gradient);
    let mut fused_gradient = vec![0.0_f32; LEN];
    session.read_param_grad("fused_x", &mut fused_gradient);
    assert_eq!(expanded_gradient, fused_gradient);
}

#[test]
fn softplus_compiles_to_one_pointwise_dispatch() {
    use meganeura::compile::{CompileOptions, compile_with};

    let mut graph = Graph::new();
    let x = graph.input("x", &[513]);
    let y = graph.softplus(x, 10.0);
    graph.set_outputs(vec![y]);

    let plan = compile_with(&graph, &CompileOptions::default());
    assert_eq!(plan.dispatches.len(), 1);
    let dispatch = &plan.dispatches[0];
    assert_eq!(dispatch.input_buffers.len(), 1);
    assert!(dispatch.pointwise.is_some());
}

#[test]
fn chain_relu_neg_parity() {
    // Two unary ops in sequence exercise two separately-generated
    // pointwise pipelines in the same plan.
    let input: Vec<f32> = (0..256).map(|i| (i as f32) * 0.1 - 12.0).collect();
    assert_parity(&["x"], &[&input], 256, |g, xs| {
        let r = g.relu(xs[0]);
        g.neg(r)
    });
}

// ---- Binary ops ----

#[test]
fn add_parity() {
    let a: Vec<f32> = (0..256).map(|i| (i as f32) * 0.05 - 6.0).collect();
    let b: Vec<f32> = (0..256).map(|i| (i as f32) * 0.03 - 4.0).collect();
    assert_parity(&["a", "b"], &[&a, &b], 256, |g, xs| g.add(xs[0], xs[1]));
}

#[test]
fn mul_parity() {
    let a: Vec<f32> = (0..256).map(|i| (i as f32) * 0.05 - 6.0).collect();
    let b: Vec<f32> = (0..256).map(|i| (i as f32) * 0.03 - 4.0).collect();
    assert_parity(&["a", "b"], &[&a, &b], 256, |g, xs| g.mul(xs[0], xs[1]));
}

#[test]
fn greater_parity() {
    let a: Vec<f32> = (0..256).map(|i| (i as f32) * 0.05 - 6.0).collect();
    let b: Vec<f32> = (0..256).map(|i| -(i as f32) * 0.02 + 2.0).collect();
    assert_parity(&["a", "b"], &[&a, &b], 256, |g, xs| g.greater(xs[0], xs[1]));
}

#[test]
fn swiglu_parity() {
    let gate: Vec<f32> = (0..256).map(|i| (i as f32) * 0.05 - 6.0).collect();
    let up: Vec<f32> = (0..256).map(|i| (i as f32) * 0.02 - 2.0).collect();
    assert_parity(&["gate", "up"], &[&gate, &up], 256, |g, xs| {
        g.swiglu(xs[0], xs[1])
    });
}

// ---- Fusion pass ----

/// Confirm the fusion pass collapses a 3-op chain into 1 dispatch.
#[test]
fn fusion_reduces_dispatch_count() {
    use meganeura::compile::{CompileOptions, compile_with};

    let mut g = Graph::new();
    let x = g.input("x", &[256]);
    let a = g.relu(x);
    let b = g.neg(a);
    let c = g.silu(b);
    g.set_outputs(vec![c]);

    let baseline_plan = compile_with(
        &g,
        &CompileOptions {
            use_schedule_pointwise: false,
            ..Default::default()
        },
    );
    let fused_plan = compile_with(
        &g,
        &CompileOptions {
            use_schedule_pointwise: true,
            ..Default::default()
        },
    );

    // Baseline: 3 dispatches (relu, neg, silu).
    assert_eq!(baseline_plan.dispatches.len(), 3);
    // Fused: the three pointwise dispatches should collapse into 1.
    assert_eq!(
        fused_plan.dispatches.len(),
        1,
        "expected pointwise chain to collapse to one dispatch"
    );
    assert!(fused_plan.dispatches[0].pointwise.is_some());
}

/// Parity of a 3-op chain, once the fusion pass has run.
#[test]
fn fused_chain_runtime_parity() {
    let input: Vec<f32> = (0..256).map(|i| (i as f32) * 0.1 - 12.0).collect();
    assert_parity(&["x"], &[&input], 256, |g, xs| {
        let a = g.relu(xs[0]);
        let b = g.neg(a);
        g.silu(b)
    });
}

/// Mixed binary+unary fusion: add(relu(a), b) should become one dispatch
/// with a 2-input fused DAG, and still match the hand-written path.
#[test]
fn fused_relu_into_add_runtime_parity() {
    let a: Vec<f32> = (0..256).map(|i| (i as f32) * 0.05 - 6.0).collect();
    let b: Vec<f32> = (0..256).map(|i| -(i as f32) * 0.02 + 2.0).collect();
    assert_parity(&["a", "b"], &[&a, &b], 256, |g, xs| {
        let r = g.relu(xs[0]);
        g.add(r, xs[1])
    });
}

/// Ternary fusion: `add(mul(a, b), c)` — a producer binary feeds into a
/// consumer binary, producing an arity-3 fused DAG routed through
/// TernaryData. Verifies both the dispatch-count collapse and runtime
/// parity with the unfused path.
#[test]
fn ternary_fusion_add_of_mul() {
    use meganeura::compile::{CompileOptions, compile_with};

    let mut g = Graph::new();
    let a = g.input("a", &[256]);
    let b = g.input("b", &[256]);
    let c = g.input("c", &[256]);
    let ab = g.mul(a, b);
    let out = g.add(ab, c);
    g.set_outputs(vec![out]);

    // Unfused: 2 dispatches (mul, add).
    assert_eq!(
        compile_with(
            &g,
            &CompileOptions {
                use_schedule_pointwise: false,
                ..Default::default()
            },
        )
        .dispatches
        .len(),
        2
    );
    let fused = compile_with(
        &g,
        &CompileOptions {
            use_schedule_pointwise: true,
            ..Default::default()
        },
    );
    assert_eq!(
        fused.dispatches.len(),
        1,
        "expected mul+add to collapse into a single arity-3 dispatch"
    );
    let dag = fused.dispatches[0]
        .pointwise
        .as_ref()
        .expect("fused dispatch should carry a DAG");
    assert_eq!(dag.n_inputs, 3);

    let a_data: Vec<f32> = (0..256).map(|i| (i as f32) * 0.1 - 12.0).collect();
    let b_data: Vec<f32> = (0..256).map(|i| (i as f32) * 0.05 - 6.0).collect();
    let c_data: Vec<f32> = (0..256).map(|i| -(i as f32) * 0.02 + 2.0).collect();
    assert_parity(
        &["a", "b", "c"],
        &[&a_data, &b_data, &c_data],
        256,
        |g, xs| {
            let ab = g.mul(xs[0], xs[1]);
            g.add(ab, xs[2])
        },
    );
}

// ---- Reduction archetype: softmax ----

/// Softmax through the schedule-template reduction archetype must match
/// the hand-written softmax.wgsl within f32 round-off.
#[test]
fn softmax_parity() {
    for (batch, features) in [(4, 64), (513, 8), (259, 3), (257, 10)] {
        let n = batch * features;
        // Non-uniform input so per-row max varies and exp(x - max) is
        // exercised across the subtraction axis.
        let input: Vec<f32> = (0..n)
            .map(|i| ((i as f32) * 0.017 - 3.0).sin() * 2.0)
            .collect();

        let run = |use_schedule: bool| -> Vec<f32> {
            let mut g = Graph::new();
            let x = g.input("x", &[batch, features]);
            let y = g.softmax(x);
            g.set_outputs(vec![y]);
            let opts = CompileOptions {
                use_schedule_reduction: use_schedule,
                ..Default::default()
            };
            let mut s = inference(&g, opts);
            s.set_input("x", &input);
            s.step();
            s.wait();
            s.read_output(n)
        };

        let baseline = run(false);
        let schedule = run(true);

        assert_eq!(baseline.len(), schedule.len());
        for (i, (a, b)) in baseline.iter().zip(schedule.iter()).enumerate() {
            assert!(
                (a - b).abs() <= a.abs().max(b.abs()) * 1e-5 + 1e-7,
                "softmax parity for [{batch}, {features}] at [{i}]: baseline={a}, schedule={b}",
            );
        }
        for row in 0..batch {
            let sum: f32 = schedule[row * features..(row + 1) * features].iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "softmax [{batch}, {features}] row {row} doesn't sum to 1: got {sum}",
            );
        }
    }
}

/// With `use_schedule_reduction`, softmax should compile to two dispatches
/// (max reduction + sum/normalize). Without, it's one.
#[test]
fn softmax_schedule_emits_two_dispatches() {
    use meganeura::compile::{CompileOptions, compile_with};
    let mut g = Graph::new();
    let x = g.input("x", &[4, 64]);
    let y = g.softmax(x);
    g.set_outputs(vec![y]);

    let baseline = compile_with(
        &g,
        &CompileOptions {
            use_schedule_reduction: false,
            ..Default::default()
        },
    );
    let schedule = compile_with(
        &g,
        &CompileOptions {
            use_schedule_reduction: true,
            ..Default::default()
        },
    );
    assert_eq!(baseline.dispatches.len(), 1);
    assert_eq!(schedule.dispatches.len(), 2);
    assert!(schedule.dispatches[0].reduction.is_some());
    assert!(schedule.dispatches[1].reduction.is_some());
}

// ---- Reduction archetype: RmsNorm ----

#[test]
fn rmsnorm_parity() {
    let rows = 4usize;
    let cols = 64usize;
    let n = rows * cols;
    let input: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.013 - 2.0).sin()).collect();
    let weight: Vec<f32> = (0..cols).map(|i| 0.5 + (i as f32) * 0.01).collect();

    let run = |use_schedule: bool| -> Vec<f32> {
        let mut g = Graph::new();
        let x = g.input("x", &[rows, cols]);
        let w = g.parameter("w", &[cols]);
        let y = g.rms_norm(x, w, 1e-5);
        g.set_outputs(vec![y]);
        let opts = CompileOptions {
            use_schedule_reduction: use_schedule,
            ..Default::default()
        };
        let mut s = inference(&g, opts);
        s.set_input("x", &input);
        s.set_parameter("w", &weight);
        s.step();
        s.wait();
        s.read_output(n)
    };

    let baseline = run(false);
    let schedule = run(true);
    assert_eq!(baseline.len(), schedule.len());
    for (i, (a, b)) in baseline.iter().zip(schedule.iter()).enumerate() {
        assert!(
            (a - b).abs() <= a.abs().max(b.abs()) * 1e-5 + 1e-7,
            "rmsnorm parity mismatch at [{i}]: baseline={a}, schedule={b}",
        );
    }
}

#[test]
fn chain_add_relu_parity() {
    // Mixes a binary op with a unary op — exercises both generated-pipeline
    // kinds plus the BinaryData/UnaryData layout switching in one plan.
    let a: Vec<f32> = (0..256).map(|i| (i as f32) * 0.05 - 6.0).collect();
    let b: Vec<f32> = (0..256).map(|i| -(i as f32) * 0.02 + 2.0).collect();
    assert_parity(&["a", "b"], &[&a, &b], 256, |g, xs| {
        let s = g.add(xs[0], xs[1]);
        g.relu(s)
    });
}

#[test]
fn gelu_parity() {
    // Gelu was the last unary op without a DAG mapping; this pins the
    // generated tanh-approx DAG against unary.wgsl's `gelu` entry point.
    let input: Vec<f32> = (0..256).map(|i| (i as f32) * 0.05 - 6.0).collect();
    assert_parity(&["x"], &[&input], 256, |g, xs| g.gelu(xs[0]));
}
