//! Regressions found by the September 2026 shader audit.
//!
//! Every case compares the GPU against a CPU reference or a central finite
//! difference, so each one fails on the defect it names rather than on a
//! changed rounding order.

use meganeura::{Graph, Session};

fn values(n: usize, frequency: f32, phase: f32) -> Vec<f32> {
    (0..n)
        .map(|i| (i as f32 * frequency + phase).sin() * 0.2)
        .collect()
}

/// Central finite difference of the scalar loss for one parameter element.
fn finite_difference(
    session: &mut Session,
    parameters: &[(&str, Vec<f32>)],
    inputs: &[(&str, Vec<f32>)],
    name: &str,
    index: usize,
    step: f32,
) -> f32 {
    let mut evaluate = |delta: f32| {
        for (parameter, data) in parameters {
            if *parameter == name {
                let mut data = data.clone();
                data[index] += delta;
                session.set_parameter(parameter, &data);
            } else {
                session.set_parameter(parameter, data);
            }
        }
        for (input, data) in inputs {
            session.set_input(input, data);
        }
        session.step();
        session.wait();
        session.read_loss()
    };
    (evaluate(step) - evaluate(-step)) / (2.0 * step)
}

fn assert_close(label: &str, actual: f32, expected: f32) {
    let error = (actual - expected).abs();
    let scale = actual.abs().max(expected.abs()).max(1e-5);
    assert!(
        error < 2e-3 || error / scale < 0.05,
        "{label}: analytical={actual:+.6e}, numerical={expected:+.6e}"
    );
}

/// Heads wider than 64 with a sequence shorter than the flash query block
/// used to fall back to the 64-lane scalar kernels, which silently dropped
/// every dimension past 63.
#[test]
fn attention_backward_covers_heads_wider_than_64() {
    let (seq, heads, head_dim) = (4usize, 2u32, 128u32);
    let width = heads as usize * head_dim as usize;
    let mut graph = Graph::new();
    let q = graph.parameter("q", &[seq, width]);
    let k = graph.parameter("k", &[seq, width]);
    let v = graph.parameter("v", &[seq, width]);
    let attention = graph.multi_head_attn(q, k, v, heads, heads, head_dim, false);
    let weights = graph.input("weights", &[seq, width]);
    let weighted = graph.mul(attention, weights);
    let loss = graph.sum_all(weighted);
    graph.set_outputs(vec![loss]);

    let n = seq * width;
    let parameters = [
        ("q", values(n, 0.017, 0.3)),
        ("k", values(n, 0.019, 0.7)),
        ("v", values(n, 0.023, 1.1)),
    ];
    let inputs = [("weights", values(n, 0.013, 1.7))];

    let mut training = meganeura::build(&graph, meganeura::SessionConfig::from_env()).0;
    for (name, data) in &parameters {
        training.set_parameter(name, data);
    }
    training.set_input("weights", &inputs[0].1);
    training.set_learning_rate(0.0);
    training.step();
    training.wait();

    let mut inference = meganeura::build(&graph, meganeura::SessionConfig::inference_from_env()).0;
    for (name, _) in &parameters {
        let mut gradient = vec![0.0; n];
        training.read_param_grad(name, &mut gradient);
        // Both sides of lane 64 in both heads and in a later row.
        for index in [
            0usize,
            63,
            64,
            100,
            127,
            128 + 70,
            width + 90,
            3 * width + 255,
        ] {
            let numerical =
                finite_difference(&mut inference, &parameters, &inputs, name, index, 1e-3);
            assert_close(&format!("{name}[{index}]"), gradient[index], numerical);
        }
    }
}

/// Checks every parameter gradient of `loss(graph)` against finite differences.
fn check_parameter_gradients(
    label: &str,
    graph: &Graph,
    parameters: &[(&str, Vec<f32>)],
    inputs: &[(&str, Vec<f32>)],
    step: f32,
) {
    let mut training = meganeura::build(graph, meganeura::SessionConfig::from_env()).0;
    for (name, data) in parameters {
        training.set_parameter(name, data);
    }
    for (name, data) in inputs {
        training.set_input(name, data);
    }
    training.set_learning_rate(0.0);
    training.step();
    training.wait();

    let mut inference = meganeura::build(graph, meganeura::SessionConfig::inference_from_env()).0;
    for (name, data) in parameters {
        let mut gradient = vec![0.0; data.len()];
        training.read_param_grad(name, &mut gradient);
        for (index, &analytical) in gradient.iter().enumerate() {
            let numerical =
                finite_difference(&mut inference, parameters, inputs, name, index, step);
            assert_close(&format!("{label} {name}[{index}]"), analytical, numerical);
        }
    }
}

/// With fewer than four rows the weight gradient skipped the row reduction
/// and wrote each row's partial product past the end of the `[cols]` output.
#[test]
fn layer_norm_weight_gradient_sums_short_batches() {
    for rows in [1usize, 2, 3, 5] {
        let cols = 6;
        let mut graph = Graph::new();
        let x = graph.parameter("x", &[rows, cols]);
        let w = graph.parameter("w", &[cols]);
        let b = graph.parameter("b", &[cols]);
        let y = graph.layer_norm(x, w, b, 1e-5);
        let target = graph.input("target", &[rows, cols]);
        let weighted = graph.mul(y, target);
        let loss = graph.sum_all(weighted);
        graph.set_outputs(vec![loss]);
        let parameters = [
            (
                "x",
                values(rows * cols, 0.7, 0.1)
                    .iter()
                    .map(|v| v * 10.0)
                    .collect(),
            ),
            (
                "w",
                values(cols, 0.9, 0.4).iter().map(|v| 1.0 + v).collect(),
            ),
            ("b", values(cols, 0.5, 0.2)),
        ];
        let inputs = [(
            "target",
            values(rows * cols, 0.37, 0.9)
                .iter()
                .map(|v| v * 5.0)
                .collect(),
        )];
        check_parameter_gradients(&format!("rows={rows}"), &graph, &parameters, &inputs, 1e-2);
    }
}

/// GroupNorm backward derived its variance as E[x²] − mean², which cancels
/// catastrophically for activations with a large mean and can even go
/// negative. The forward pass uses a stable form, so the two disagreed.
#[test]
fn group_norm_backward_is_stable_for_large_means() {
    // Also with a frozen input, where only the weight and bias gradients
    // ask for the shared statistics.
    for train_input in [true, false] {
        check_group_norm_backward(train_input);
    }
}

fn check_group_norm_backward(train_input: bool) {
    let (batch, channels, spatial, groups) = (2usize, 4usize, 64usize, 2usize);
    let eps = 1e-5f32;
    let n = batch * channels * spatial;
    let x: Vec<f32> = values(n, 0.37, 0.2)
        .iter()
        .map(|v| 50.0 + v * 0.25)
        .collect();
    let w: Vec<f32> = values(channels, 0.9, 0.4).iter().map(|v| 1.0 + v).collect();
    let b = values(channels, 0.5, 0.2);
    let target = values(n, 0.23, 0.9);

    let mut graph = Graph::new();
    let xs = if train_input {
        graph.parameter("x", &[n])
    } else {
        graph.input("x", &[n])
    };
    let ws = graph.parameter("w", &[channels]);
    let bs = graph.parameter("b", &[channels]);
    let y = graph.group_norm(
        xs,
        ws,
        bs,
        batch as u32,
        channels as u32,
        spatial as u32,
        groups as u32,
        eps,
    );
    let t = graph.input("target", &[n]);
    let weighted = graph.mul(y, t);
    let loss = graph.sum_all(weighted);
    graph.set_outputs(vec![loss]);

    let mut session = meganeura::build(&graph, meganeura::SessionConfig::from_env()).0;
    if train_input {
        session.set_parameter("x", &x);
    } else {
        session.set_input("x", &x);
    }
    session.set_parameter("w", &w);
    session.set_parameter("b", &b);
    session.set_input("target", &target);
    session.set_learning_rate(0.0);
    session.step();
    session.wait();
    let mut dx = vec![0.0; n];
    let mut dw = vec![0.0; channels];
    let mut db = vec![0.0; channels];
    if train_input {
        session.read_param_grad("x", &mut dx);
    }
    session.read_param_grad("w", &mut dw);
    session.read_param_grad("b", &mut db);

    // f64 reference.
    let per_group = channels / groups;
    let group_size = per_group * spatial;
    let mut want_dx = vec![0.0f64; n];
    let mut want_dw = vec![0.0f64; channels];
    let mut want_db = vec![0.0f64; channels];
    for image in 0..batch {
        for group in 0..groups {
            let index = |local: usize| {
                let c = group * per_group + local / spatial;
                ((image * channels + c) * spatial + local % spatial, c)
            };
            let mean =
                (0..group_size).map(|j| x[index(j).0] as f64).sum::<f64>() / group_size as f64;
            let var = (0..group_size)
                .map(|j| (x[index(j).0] as f64 - mean).powi(2))
                .sum::<f64>()
                / group_size as f64;
            let inv_std = 1.0 / (var + eps as f64).sqrt();
            let (mut mean_g, mut mean_gx) = (0.0, 0.0);
            for j in 0..group_size {
                let (i, c) = index(j);
                let xhat = (x[i] as f64 - mean) * inv_std;
                let g = target[i] as f64 * w[c] as f64;
                mean_g += g / group_size as f64;
                mean_gx += g * xhat / group_size as f64;
                want_dw[c] += target[i] as f64 * xhat;
                want_db[c] += target[i] as f64;
            }
            for j in 0..group_size {
                let (i, c) = index(j);
                let xhat = (x[i] as f64 - mean) * inv_std;
                let g = target[i] as f64 * w[c] as f64;
                want_dx[i] = inv_std * (g - mean_g - xhat * mean_gx);
            }
        }
    }
    let check = |name: &str, got: &[f32], want: &[f64]| {
        let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g as f64 - w).abs() <= 2e-2 * scale,
                "{name}[{i}]: got {g}, want {w} (scale {scale})"
            );
        }
    };
    if train_input {
        check("dx", &dx, &want_dx);
    }
    check("dw", &dw, &want_dw);
    check("db", &db, &want_db);
}

/// The per-channel bias gradient now reduces each contiguous (batch,
/// channel) plane instead of transposing the whole gradient first.
#[test]
fn per_channel_bias_gradient_matches_reference() {
    for batch in [1usize, 3] {
        let (channels, spatial) = (5usize, 37usize);
        let n = batch * channels * spatial;
        let mut graph = Graph::new();
        let x = graph.input("x", &[n]);
        let b = graph.parameter("b", &[channels]);
        let y = graph.add_per_channel(x, b, channels as u32, spatial as u32);
        let y = graph.relu(y);
        let t = graph.input("t", &[n]);
        let weighted = graph.mul(y, t);
        let loss = graph.sum_all(weighted);
        graph.set_outputs(vec![loss]);
        let bias = values(channels, 0.9, 0.4);
        let xs = values(n, 0.31, 0.2);
        let ts = values(n, 0.17, 0.5);

        let mut session = meganeura::build(&graph, meganeura::SessionConfig::from_env()).0;
        session.set_parameter("b", &bias);
        session.set_input("x", &xs);
        session.set_input("t", &ts);
        session.set_learning_rate(0.0);
        session.step();
        session.wait();
        let mut got = vec![0.0; channels];
        session.read_param_grad("b", &mut got);

        let mut want = vec![0.0f64; channels];
        for i in 0..n {
            let c = (i / spatial) % channels;
            if xs[i] + bias[c] > 0.0 {
                want[c] += ts[i] as f64;
            }
        }
        for c in 0..channels {
            assert!(
                (got[c] as f64 - want[c]).abs() < 1e-4,
                "batch={batch} b[{c}]: got {}, want {}",
                got[c],
                want[c]
            );
        }
    }
}

/// Wide planes pool through a workgroup-per-plane reduction; narrow ones
/// keep the one-thread-per-plane kernel. Both must average exactly.
#[test]
fn global_avg_pool_matches_reference() {
    for (batch, channels, spatial) in [(2usize, 3usize, 7usize), (2, 3, 1000), (1, 32, 12544)] {
        let n = batch * channels * spatial;
        let mut graph = Graph::new();
        let x = graph.input("x", &[n]);
        let y = graph.global_avg_pool(x, batch as u32, channels as u32, spatial as u32);
        graph.set_outputs(vec![y]);
        let (mut session, _) = meganeura::build(
            &graph,
            meganeura::SessionConfig {
                mode: meganeura::Mode::Inference,
                ..meganeura::SessionConfig::default()
            },
        );
        let xs = values(n, 0.013, 0.2);
        session.set_input("x", &xs);
        session.step();
        session.wait();
        let got = session.read_output(batch * channels);
        for (row, value) in got.iter().enumerate() {
            let want = xs[row * spatial..(row + 1) * spatial]
                .iter()
                .map(|&v| v as f64)
                .sum::<f64>()
                / spatial as f64;
            assert!(
                (*value as f64 - want).abs() < 1e-5,
                "{batch}x{channels}x{spatial} row {row}: got {value}, want {want}"
            );
        }
    }
}

/// Global clipping measures the norm with many workgroups per gradient. With
/// SGD the step is exactly `-lr * g * min(1, max_norm / ||g||)`, so the
/// update reveals the norm the GPU computed across every workgroup.
#[test]
fn global_grad_clip_measures_large_gradients() {
    let sizes = [300_001usize, 5, 70_000];
    let mut graph = Graph::new();
    let mut terms = Vec::new();
    for (i, &n) in sizes.iter().enumerate() {
        let p = graph.parameter(&format!("p{i}"), &[n]);
        let t = graph.input(&format!("t{i}"), &[n]);
        let product = graph.mul(p, t);
        terms.push(graph.sum_all(product));
    }
    let loss = terms
        .into_iter()
        .reduce(|a, b| graph.add(a, b))
        .expect("at least one term");
    graph.set_outputs(vec![loss]);

    // d(loss)/d(p_i) = t_i exactly.
    let targets: Vec<Vec<f32>> = sizes
        .iter()
        .enumerate()
        .map(|(i, &n)| values(n, 0.37 + i as f32 * 0.1, 0.3))
        .collect();
    let norm = targets
        .iter()
        .flatten()
        .map(|&v| (v as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let max_norm = (norm * 0.25) as f32;
    let lr = 0.5f32;

    let mut session = meganeura::build(&graph, meganeura::SessionConfig::from_env()).0;
    for (i, &n) in sizes.iter().enumerate() {
        session.set_parameter(&format!("p{i}"), &vec![0.0; n]);
        session.set_input(&format!("t{i}"), &targets[i]);
    }
    session.set_grad_clip_norm(max_norm);
    session.set_learning_rate(lr);
    session.step();
    session.wait();

    let scale = max_norm as f64 / norm;
    for (i, &n) in sizes.iter().enumerate() {
        let mut after = vec![0.0; n];
        session.read_param(&format!("p{i}"), &mut after);
        for j in [0, n / 2, n - 1] {
            let want = -(lr as f64) * targets[i][j] as f64 * scale;
            assert!(
                (after[j] as f64 - want).abs() <= 1e-4 * want.abs().max(1e-3),
                "p{i}[{j}]: got {}, want {want}",
                after[j]
            );
        }
    }
}

/// The tiled transpose must handle edges that do not fill a 16x16 tile.
#[test]
fn transpose_matches_reference_on_ragged_tiles() {
    for (m, n) in [(1usize, 1usize), (3, 40), (17, 33), (64, 16), (100, 7)] {
        let mut graph = Graph::new();
        let x = graph.input("x", &[m, n]);
        let y = graph.transpose(x);
        graph.set_outputs(vec![y]);
        let (mut session, _) = meganeura::build(
            &graph,
            meganeura::SessionConfig {
                mode: meganeura::Mode::Inference,
                ..meganeura::SessionConfig::default()
            },
        );
        let xs: Vec<f32> = (0..m * n).map(|i| i as f32).collect();
        session.set_input("x", &xs);
        session.step();
        session.wait();
        let got = session.read_output(m * n);
        for r in 0..m {
            for c in 0..n {
                assert_eq!(got[c * m + r], xs[r * n + c], "{m}x{n} at ({r}, {c})");
            }
        }
    }
}

/// Cross-entropy combines max, sum-exp and the label sum in one online pass.
/// Wide logit ranges make lanes rescale their running sums repeatedly.
#[test]
fn cross_entropy_online_pass_matches_reference() {
    let (rows, classes) = (3usize, 5000usize);
    let logits: Vec<f32> = (0..rows * classes)
        .map(|i| ((i * 7919) % 1000) as f32 * 0.08 - 40.0 + (i % 5000) as f32 * 0.004)
        .collect();
    let mut labels = vec![0.0f32; rows * classes];
    for r in 0..rows {
        labels[r * classes + (r * 1777) % classes] = 0.75;
        labels[r * classes + (r * 331 + 5) % classes] = 0.5;
    }
    let mut graph = Graph::new();
    let x = graph.parameter("x", &[rows, classes]);
    let y = graph.input("y", &[rows, classes]);
    let loss = graph.cross_entropy_loss(x, y);
    graph.set_outputs(vec![loss]);
    let mut session = meganeura::build(&graph, meganeura::SessionConfig::from_env()).0;
    session.set_parameter("x", &logits);
    session.set_input("y", &labels);
    session.set_learning_rate(0.0);
    session.step();
    session.wait();
    let got_loss = session.read_loss();
    let mut got_grad = vec![0.0; rows * classes];
    session.read_param_grad("x", &mut got_grad);

    let mut want_loss = 0.0f64;
    for r in 0..rows {
        let row = &logits[r * classes..(r + 1) * classes];
        let max = row.iter().fold(f64::MIN, |m, &v| m.max(v as f64));
        let lse = row
            .iter()
            .map(|&v| (v as f64 - max).exp())
            .sum::<f64>()
            .ln()
            + max;
        let label_sum: f64 = labels[r * classes..(r + 1) * classes]
            .iter()
            .map(|&v| v as f64)
            .sum();
        for c in 0..classes {
            let i = r * classes + c;
            let log_softmax = logits[i] as f64 - lse;
            want_loss -= labels[i] as f64 * log_softmax / rows as f64;
            let want = (log_softmax.exp() * label_sum - labels[i] as f64) / rows as f64;
            assert!(
                (got_grad[i] as f64 - want).abs() < 1e-5,
                "grad[{r}, {c}]: got {}, want {want}",
                got_grad[i]
            );
        }
    }
    assert!(
        (got_loss as f64 - want_loss).abs() < 1e-3 * want_loss.abs().max(1.0),
        "loss: got {got_loss}, want {want_loss}"
    );
}

/// The per-channel bias add is a broadcast pointwise kernel that fuses with
/// the activation after it; the fused kernel must index the bias exactly.
#[test]
fn fused_per_channel_bias_and_relu_match_reference() {
    let (batch, channels, spatial) = (3usize, 7usize, 45usize);
    let n = batch * channels * spatial;
    let mut graph = Graph::new();
    let x = graph.input("x", &[n]);
    let b = graph.input("b", &[channels]);
    let y = graph.add_per_channel(x, b, channels as u32, spatial as u32);
    let y = graph.relu(y);
    graph.set_outputs(vec![y]);
    let (mut session, _) = meganeura::build(
        &graph,
        meganeura::SessionConfig {
            mode: meganeura::Mode::Inference,
            ..meganeura::SessionConfig::default()
        },
    );
    let xs = values(n, 0.29, 0.1);
    let bs = values(channels, 1.3, 0.7);
    session.set_input("x", &xs);
    session.set_input("b", &bs);
    session.step();
    session.wait();
    let got = session.read_output(n);
    for i in 0..n {
        let want = (xs[i] + bs[(i / spatial) % channels]).max(0.0);
        assert_eq!(got[i], want, "element {i}");
    }
}

/// Adaptive clipping measures every parameter with many workgroups and
/// scales each by min(1, clip * max(pmin, ||p||) / ||g||). With SGD the step
/// reveals the scale the GPU computed for each parameter.
#[test]
fn adaptive_grad_clip_measures_large_parameters() {
    let sizes = [300_001usize, 5, 70_000];
    let (clip, pmin, lr) = (0.05f32, 1e-3f32, 0.5f32);
    let mut graph = Graph::new();
    let mut terms = Vec::new();
    for (i, &n) in sizes.iter().enumerate() {
        let p = graph.parameter(&format!("p{i}"), &[n]);
        let t = graph.input(&format!("t{i}"), &[n]);
        let product = graph.mul(p, t);
        terms.push(graph.sum_all(product));
    }
    let loss = terms
        .into_iter()
        .reduce(|a, b| graph.add(a, b))
        .expect("at least one term");
    graph.set_outputs(vec![loss]);

    let initial: Vec<Vec<f32>> = sizes
        .iter()
        .enumerate()
        .map(|(i, &n)| values(n, 0.21 + i as f32 * 0.05, 1.1))
        .collect();
    let targets: Vec<Vec<f32>> = sizes
        .iter()
        .enumerate()
        .map(|(i, &n)| values(n, 0.37 + i as f32 * 0.1, 0.3))
        .collect();

    let mut session = meganeura::build(&graph, meganeura::SessionConfig::from_env()).0;
    for (i, _) in sizes.iter().enumerate() {
        session.set_parameter(&format!("p{i}"), &initial[i]);
        session.set_input(&format!("t{i}"), &targets[i]);
    }
    session.set_adaptive_grad_clip(clip, pmin);
    session.set_learning_rate(lr);
    session.step();
    session.wait();

    let norm = |v: &[f32]| v.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    for (i, &n) in sizes.iter().enumerate() {
        let upper = clip as f64 * norm(&initial[i]).max(pmin as f64);
        let scale = (upper / norm(&targets[i])).min(1.0);
        let mut after = vec![0.0; n];
        session.read_param(&format!("p{i}"), &mut after);
        for j in [0, n / 2, n - 1] {
            let want = initial[i][j] as f64 - lr as f64 * targets[i][j] as f64 * scale;
            assert!(
                (after[j] as f64 - want).abs() <= 1e-5 * want.abs().max(1e-2),
                "p{i}[{j}]: got {}, want {want}",
                after[j]
            );
        }
    }
}

/// The flash dK/dV kernels read the per-row dot(dO, O) from a buffer
/// reduced once, in both their register (one thread per KV row) and shared
/// memory (several threads per row) forms, with GQA and a causal mask.
#[test]
fn flash_attention_backward_uses_precomputed_row_dots() {
    use meganeura::compile::ShaderEntry;
    // (seq, heads, kv_heads, head_dim, causal): 64-wide heads split each KV
    // row across two threads; 32-wide heads give each row one thread.
    for (seq, heads, kv_heads, head_dim, causal) in [
        (130usize, 4u32, 2u32, 64u32, false),
        (130, 2, 2, 64, true),
        (260, 2, 1, 32, true),
    ] {
        let q_width = heads as usize * head_dim as usize;
        let kv_width = kv_heads as usize * head_dim as usize;
        let mut graph = Graph::new();
        let q = graph.parameter("q", &[seq, q_width]);
        let k = graph.parameter("k", &[seq, kv_width]);
        let v = graph.parameter("v", &[seq, kv_width]);
        let attention = if causal {
            graph.causal_attention(q, k, v, heads, kv_heads, head_dim)
        } else {
            graph.multi_head_attn(q, k, v, heads, kv_heads, head_dim, false)
        };
        let weights = graph.input("weights", &[seq, q_width]);
        let weighted = graph.mul(attention, weights);
        let loss = graph.sum_all(weighted);
        graph.set_outputs(vec![loss]);

        let parameters = [
            ("q", values(seq * q_width, 0.017, 0.3)),
            ("k", values(seq * kv_width, 0.019, 0.7)),
            ("v", values(seq * kv_width, 0.023, 1.1)),
        ];
        let inputs = [("weights", values(seq * q_width, 0.013, 1.7))];
        let mut training = meganeura::build(&graph, meganeura::SessionConfig::from_env()).0;
        assert!(
            training
                .plan()
                .dispatches
                .iter()
                .any(|d| d.shader == ShaderEntry::FlashGradKV),
            "seq={seq} head_dim={head_dim} did not reach the flash dK/dV kernel"
        );
        for (name, data) in &parameters {
            training.set_parameter(name, data);
        }
        training.set_input("weights", &inputs[0].1);
        training.set_learning_rate(0.0);
        training.step();
        training.wait();

        let mut inference =
            meganeura::build(&graph, meganeura::SessionConfig::inference_from_env()).0;
        for (name, data) in &parameters {
            let mut gradient = vec![0.0; data.len()];
            training.read_param_grad(name, &mut gradient);
            let width = data.len() / seq;
            for index in [0, width + 3, (seq / 2) * width + width - 1, data.len() - 1] {
                let numerical =
                    finite_difference(&mut inference, &parameters, &inputs, name, index, 1e-2);
                assert_close(
                    &format!("seq={seq} hd={head_dim} causal={causal} {name}[{index}]"),
                    gradient[index],
                    numerical,
                );
            }
        }
    }
}

/// Norm weight gradients fold several rows into each partial once there
/// are enough rows; the last block may be short.
#[test]
fn norm_weight_gradients_fold_row_blocks() {
    for rows in [5001usize, 4097] {
        let cols = 96usize;
        let eps = 1e-5f32;
        let xs = values(rows * cols, 0.37, 0.2);
        let ts = values(rows * cols, 0.11, 0.9);
        let w0: Vec<f32> = values(cols, 0.9, 0.4).iter().map(|v| 1.0 + v).collect();
        for layer in [false, true] {
            let mut graph = Graph::new();
            let x = graph.input("x", &[rows, cols]);
            let w = graph.parameter("w", &[cols]);
            let y = if layer {
                let b = graph.parameter("b", &[cols]);
                graph.layer_norm(x, w, b, eps)
            } else {
                graph.rms_norm(x, w, eps)
            };
            let t = graph.input("t", &[rows, cols]);
            let weighted = graph.mul(y, t);
            let loss = graph.sum_all(weighted);
            graph.set_outputs(vec![loss]);
            let mut session = meganeura::build(&graph, meganeura::SessionConfig::from_env()).0;
            session.set_parameter("w", &w0);
            if layer {
                session.set_parameter("b", &vec![0.0; cols]);
            }
            session.set_input("x", &xs);
            session.set_input("t", &ts);
            session.set_learning_rate(0.0);
            session.step();
            session.wait();
            let mut got = vec![0.0; cols];
            session.read_param_grad("w", &mut got);

            let mut want = vec![0.0f64; cols];
            for r in 0..rows {
                let row = &xs[r * cols..(r + 1) * cols];
                let (shift, scale) = if layer {
                    let mean = row.iter().map(|&v| v as f64).sum::<f64>() / cols as f64;
                    let var =
                        row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / cols as f64;
                    (mean, 1.0 / (var + eps as f64).sqrt())
                } else {
                    let ms = row.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / cols as f64;
                    (0.0, 1.0 / (ms + eps as f64).sqrt())
                };
                for c in 0..cols {
                    want[c] += ts[r * cols + c] as f64 * (row[c] as f64 - shift) * scale;
                }
            }
            for c in 0..cols {
                assert!(
                    (got[c] as f64 - want[c]).abs() <= 1e-3 * want[c].abs().max(1.0),
                    "rows={rows} layer={layer} w[{c}]: got {}, want {}",
                    got[c],
                    want[c]
                );
            }
        }
    }
}

/// MaxPool2d routes each output gradient to the input that won its window
/// (the first maximum in row-major window order, as PyTorch does). Its
/// backward used to spread gradients evenly over stride² consecutive
/// elements, which is neither the argmax nor the right window.
#[test]
fn max_pool_gradient_routes_to_argmax() {
    // ResNet's stem pool (3x3, stride 2, padding 1) plus a ragged shape.
    for (batch, channels, h, w, k, stride, padding) in [
        (2usize, 3usize, 12usize, 12usize, 3usize, 2usize, 1usize),
        (1, 2, 9, 7, 2, 2, 0),
        (1, 1, 10, 10, 3, 1, 1),
    ] {
        let out_h = (h + 2 * padding - k) / stride + 1;
        let out_w = (w + 2 * padding - k) / stride + 1;
        let n_in = batch * channels * h * w;
        let n_out = batch * channels * out_h * out_w;
        let mut graph = Graph::new();
        let x = graph.parameter("x", &[n_in]);
        let y = graph.max_pool_2d(
            x,
            batch as u32,
            channels as u32,
            h as u32,
            w as u32,
            k as u32,
            k as u32,
            stride as u32,
            padding as u32,
        );
        let t = graph.input("t", &[n_out]);
        let weighted = graph.mul(y, t);
        let loss = graph.sum_all(weighted);
        graph.set_outputs(vec![loss]);

        // Distinct values, with some ties to exercise first-maximum routing.
        let xs: Vec<f32> = (0..n_in).map(|i| ((i * 37) % 23) as f32 * 0.25).collect();
        let ts = values(n_out, 0.31, 0.4);
        let mut session = meganeura::build(&graph, meganeura::SessionConfig::from_env()).0;
        session.set_parameter("x", &xs);
        session.set_input("t", &ts);
        session.set_learning_rate(0.0);
        session.step();
        session.wait();
        let mut got = vec![0.0; n_in];
        session.read_param_grad("x", &mut got);

        let mut want = vec![0.0f64; n_in];
        for plane in 0..batch * channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut best: Option<(f32, usize)> = None;
                    for kh in 0..k {
                        for kw in 0..k {
                            let ih = (oh * stride + kh) as isize - padding as isize;
                            let iw = (ow * stride + kw) as isize - padding as isize;
                            if ih < 0 || iw < 0 || ih >= h as isize || iw >= w as isize {
                                continue;
                            }
                            let i = plane * h * w + ih as usize * w + iw as usize;
                            if best.is_none_or(|(value, _)| xs[i] > value) {
                                best = Some((xs[i], i));
                            }
                        }
                    }
                    let (_, i) = best.expect("every window overlaps the input");
                    want[i] += ts[(plane * out_h + oh) * out_w + ow] as f64;
                }
            }
        }
        for i in 0..n_in {
            assert!(
                (got[i] as f64 - want[i]).abs() < 1e-5,
                "{h}x{w} k{k} s{stride} p{padding} dx[{i}]: got {}, want {}",
                got[i],
                want[i]
            );
        }
    }
}

/// BCE's backward divided by p(1 - p), which is exactly zero once a
/// sigmoid saturates in f32, while the forward clamps p and stays finite.
/// The denominator is floored the way PyTorch floors it.
#[test]
fn bce_gradient_is_finite_for_saturated_predictions() {
    let mut graph = Graph::new();
    let x = graph.parameter("x", &[4]);
    let p = graph.sigmoid(x);
    let t = graph.input("t", &[4]);
    let loss = graph.bce_loss(p, t);
    graph.set_outputs(vec![loss]);
    let mut session = meganeura::build(&graph, meganeura::SessionConfig::from_env()).0;
    let xs = [30.0f32, -30.0, 0.3, -1.2];
    let ts = [0.0f32, 1.0, 1.0, 0.0];
    session.set_parameter("x", &xs);
    session.set_input("t", &ts);
    session.set_learning_rate(0.0);
    session.step();
    session.wait();
    assert!(session.read_loss().is_finite());
    let mut grad = [0.0f32; 4];
    session.read_param_grad("x", &mut grad);
    assert!(grad.iter().all(|g| g.is_finite()), "gradient {grad:?}");
    // Unsaturated entries keep the exact sigmoid-BCE gradient (p - t) / N.
    for i in 2..4 {
        let p = 1.0 / (1.0 + (-xs[i]).exp());
        let want = (p - ts[i]) / 4.0;
        assert!(
            (grad[i] - want).abs() < 1e-5,
            "grad[{i}] = {}, want {want}",
            grad[i]
        );
    }
}
