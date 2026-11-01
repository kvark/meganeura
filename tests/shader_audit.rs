//! Regressions from the September 2026 shader audit that the `oracle`
//! suite cannot express: gradient-clipping measurements, which reduction
//! the flash backward binds, and finiteness where f32 saturates but f64 does
//! not. The audit's numerical regressions now live in `oracle`.

use meganeura::{Graph, Session};

fn values(n: usize, frequency: f32, phase: f32) -> Vec<f32> {
    (0..n)
        .map(|i| (i as f32 * frequency + phase).sin() * 0.2)
        .collect()
}

/// Clipping measures norms with many workgroups per tensor. With SGD each
/// step reveals the scale the GPU computed: global clipping steps by exactly
/// `-lr * g * min(1, max_norm / ||g||)`, adaptive clipping scales each
/// parameter by `min(1, clip * max(pmin, ||p||) / ||g||)`.
#[test]
fn grad_clip_measures_large_parameters_and_gradients() {
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
    let norm = |v: &[f32]| v.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    let lr = 0.5f32;
    let step = |initial: &[Vec<f32>], clip: &dyn Fn(&mut Session)| {
        let mut session = meganeura::build(&graph, crate::support::gpu::config()).0;
        for (i, initial) in initial.iter().enumerate() {
            session.set_parameter(&format!("p{i}"), initial);
            session.set_input(&format!("t{i}"), &targets[i]);
        }
        clip(&mut session);
        session.set_learning_rate(lr);
        session.step();
        session.wait();
        session
    };

    let global_norm = norm(&targets.concat());
    let max_norm = (global_norm * 0.25) as f32;
    let zeros: Vec<_> = sizes.iter().map(|&n| vec![0.0; n]).collect();
    let session = step(&zeros, &|s| s.set_grad_clip_norm(max_norm));
    let scale = max_norm as f64 / global_norm;
    for (i, &n) in sizes.iter().enumerate() {
        let mut after = vec![0.0; n];
        session.read_param(&format!("p{i}"), &mut after);
        for j in [0, n / 2, n - 1] {
            let want = -(lr as f64) * targets[i][j] as f64 * scale;
            assert!(
                (after[j] as f64 - want).abs() <= 1e-4 * want.abs().max(1e-3),
                "global p{i}[{j}]: got {}, want {want}",
                after[j]
            );
        }
    }

    let (clip, pmin) = (0.05f32, 1e-3f32);
    let initial: Vec<Vec<f32>> = sizes
        .iter()
        .enumerate()
        .map(|(i, &n)| values(n, 0.21 + i as f32 * 0.05, 1.1))
        .collect();
    let session = step(&initial, &|s| s.set_adaptive_grad_clip(clip, pmin));
    for (i, &n) in sizes.iter().enumerate() {
        let upper = clip as f64 * norm(&initial[i]).max(pmin as f64);
        let scale = (upper / norm(&targets[i])).min(1.0);
        let mut after = vec![0.0; n];
        session.read_param(&format!("p{i}"), &mut after);
        for j in [0, n / 2, n - 1] {
            let want = initial[i][j] as f64 - lr as f64 * targets[i][j] as f64 * scale;
            assert!(
                (after[j] as f64 - want).abs() <= 1e-5 * want.abs().max(1e-2),
                "adaptive p{i}[{j}]: got {}, want {want}",
                after[j]
            );
        }
    }
}

/// Flash dQ and dK/dV share one per-row dot(dO, O) reduction, including the
/// register and shared-memory KV variants, GQA and causal masking. The
/// gradients themselves are checked by `oracle`'s
/// `every_backward_path_against_the_reference`.
#[test]
fn flash_attention_backward_uses_precomputed_row_dots() {
    use meganeura::compile::ShaderEntry;
    use meganeura::kernels::attention_grad::{AttentionGrad, Operands, Part, Path};
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

        let plan = meganeura::reference::gpu::plan(
            &graph,
            &Default::default(),
            meganeura::train::Mode::Training,
        );
        let grad = |part| {
            plan.dispatches
                .iter()
                .find(|d| {
                    matches!(
                        d.shader,
                        ShaderEntry::AttentionGrad(AttentionGrad {
                            part: p,
                            path: Path::Flash | Path::Cooperative(Operands::F32),
                        }) if p == part
                    )
                })
                .unwrap_or_else(|| panic!("seq={seq} hd={head_dim}: no flash {part:?} gradient"))
        };
        let (query, kv) = (grad(Part::Q), grad(Part::KV));
        let shared = plan
            .dispatches
            .iter()
            .filter(|d| {
                d.reduction().is_some()
                    && query.input_buffers.contains(&d.output_buffer)
                    && kv.input_buffers.contains(&d.output_buffer)
            })
            .count();
        assert_eq!(
            shared, 1,
            "seq={seq} hd={head_dim} causal={causal}: dQ and dK/dV must share the row reduction"
        );
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
    let mut session = meganeura::build(&graph, crate::support::gpu::config()).0;
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
