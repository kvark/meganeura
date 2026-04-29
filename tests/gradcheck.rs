//! Numerical-gradient verification for backward ops.
//!
//! Catches the class of bug where a backward implementation hardcodes
//! `grad_output = 1.0` instead of multiplying by the actual upstream
//! gradient. Such bugs only manifest when the op is used **mid-graph**
//! (i.e. another node multiplies or scales its output before reaching the
//! final loss); when the op IS the final loss, `grad_output` happens to
//! be 1.0 and the bug is invisible.
//!
//! The discovery that broke kindle: SumAll, MeanAll, CrossEntropyLoss,
//! and BceLoss were all silently dropping `grad_output`. Coefficients
//! like `value_loss_coef`, `recon_loss_coef`, `entropy_beta`, and any
//! `mul(loss, coef)` pattern produced the *correct loss value* but a
//! gradient that ignored the coefficient.
//!
//! Each test here puts the op of interest **mid-graph** (followed by
//! `mul(scalar)` and `add(other_term)`) and compares analytical
//! gradients against finite-difference gradients.

use meganeura::{Graph, build_session};

const EPS: f32 = 5e-3;
const TOL: f32 = 1e-2;

/// Compare analytical gradient (from the backward graph) against a
/// finite-difference numerical gradient for every element of `param`.
fn gradcheck_param(
    session: &mut meganeura::Session,
    param_name: &str,
    n: usize,
    set_inputs: &dyn Fn(&mut meganeura::Session),
) {
    let mut params = vec![0.0f32; n];
    session.read_param(param_name, &mut params);
    let baseline = params.clone();

    // Analytical gradient.
    set_inputs(session);
    session.step();
    session.wait();
    let mut analytical = vec![0.0f32; n];
    session.read_param_grad(param_name, &mut analytical);

    // Numerical gradient via central finite differences.
    let mut numerical = vec![0.0f32; n];
    for i in 0..n {
        let orig = baseline[i];

        params[i] = orig + EPS;
        session.set_parameter(param_name, &params);
        set_inputs(session);
        session.step();
        session.wait();
        let l_plus = session.read_loss();

        params[i] = orig - EPS;
        session.set_parameter(param_name, &params);
        set_inputs(session);
        session.step();
        session.wait();
        let l_minus = session.read_loss();

        params[i] = orig;
        numerical[i] = (l_plus - l_minus) / (2.0 * EPS);
    }
    // Restore params so subsequent tests in the same session start clean.
    session.set_parameter(param_name, &baseline);

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for i in 0..n {
        let a = analytical[i];
        let num = numerical[i];
        let abs_diff = (a - num).abs();
        let rel_diff = abs_diff / a.abs().max(num.abs()).max(1e-6);
        max_abs = max_abs.max(abs_diff);
        max_rel = max_rel.max(rel_diff);
    }
    if max_abs > TOL && max_rel > TOL {
        for i in 0..n {
            eprintln!(
                "  [{}] analytical={:+.6}  numerical={:+.6}  diff={:+.4e}",
                i,
                analytical[i],
                numerical[i],
                analytical[i] - numerical[i]
            );
        }
        panic!(
            "gradcheck failed for `{}`: max_abs_diff={:.3e}, max_rel_diff={:.3e}",
            param_name, max_abs, max_rel
        );
    }
}

/// `mean_all(x*w) * coef`: catches the original bug. If MeanAll backward
/// drops grad_output, the gradient w.r.t. `w` will ignore `coef` entirely
/// and the analytical/numerical gradients will differ by a factor of `coef`.
#[test]
fn mean_all_mid_chain_respects_upstream_coef() {
    let mut g = Graph::new();
    let x = g.input("x", &[2, 3]);
    let w = g.parameter("w", &[2, 3]);
    let prod = g.mul(x, w);
    let mean = g.mean_all(prod);
    let coef = g.scalar(2.5);
    let loss = g.mul(mean, coef);
    g.set_outputs(vec![loss]);

    let mut session = build_session(&g);
    let x_data: Vec<f32> = (0..6).map(|i| (i as f32) * 0.7 - 1.5).collect();
    let w_init: Vec<f32> = (0..6).map(|i| (i as f32) * 0.3 + 0.1).collect();
    session.set_parameter("w", &w_init);
    let set_inputs = |s: &mut meganeura::Session| {
        s.set_input("x", &x_data);
    };
    gradcheck_param(&mut session, "w", 6, &set_inputs);
}

/// `sum_all(x*w) * coef`: same bug class as MeanAll but no `1/N` factor.
#[test]
fn sum_all_mid_chain_respects_upstream_coef() {
    let mut g = Graph::new();
    let x = g.input("x", &[2, 3]);
    let w = g.parameter("w", &[2, 3]);
    let prod = g.mul(x, w);
    let sum = g.sum_all(prod);
    let coef = g.scalar(0.7);
    let loss = g.mul(sum, coef);
    g.set_outputs(vec![loss]);

    let mut session = build_session(&g);
    let x_data: Vec<f32> = (0..6).map(|i| (i as f32) * 0.5 - 1.0).collect();
    let w_init: Vec<f32> = (0..6).map(|i| (i as f32) * 0.2 + 0.05).collect();
    session.set_parameter("w", &w_init);
    let set_inputs = |s: &mut meganeura::Session| {
        s.set_input("x", &x_data);
    };
    gradcheck_param(&mut session, "w", 6, &set_inputs);
}

/// `mse_loss(pred, target) * coef`: real-world pattern from kindle's
/// value head. mse_loss is `mean_all(sq)` internally; multiplying by a
/// coef triggers the same path.
#[test]
fn mse_loss_with_coef_scales_param_gradient() {
    let mut g = Graph::new();
    let x = g.input("x", &[2, 3]);
    let w = g.parameter("w", &[3, 2]);
    let pred = g.matmul(x, w);
    let target = g.input("target", &[2, 2]);
    let mse = g.mse_loss(pred, target);
    let coef = g.scalar(0.1);
    let loss = g.mul(mse, coef);
    g.set_outputs(vec![loss]);

    let mut session = build_session(&g);
    let x_data: Vec<f32> = (0..6).map(|i| (i as f32) * 0.3 + 0.5).collect();
    let target_data: Vec<f32> = (0..4).map(|i| (i as f32) * 0.4 - 0.2).collect();
    let w_init: Vec<f32> = (0..6).map(|i| (i as f32) * 0.25 + 0.1).collect();
    session.set_parameter("w", &w_init);
    let set_inputs = |s: &mut meganeura::Session| {
        s.set_input("x", &x_data);
        s.set_input("target", &target_data);
    };
    gradcheck_param(&mut session, "w", 6, &set_inputs);
}

/// `cross_entropy_loss(logits, labels) * coef`. Uses **batch=1** to avoid
/// a separate meganeura quirk: the CE shader writes a `[batch]`-sized loss
/// buffer but tags the IR output as scalar `[1]`, so any downstream
/// `mul`/`add` only sees element 0 — read_loss then disagrees with the
/// "user-intended" coef·CE for batch>1. Autodiff still computes gradients
/// for the *intended* CE, and matches finite-diff at batch=1.
#[test]
fn cross_entropy_loss_with_coef_scales_param_gradient() {
    let mut g = Graph::new();
    let x = g.input("x", &[1, 3]);
    let w = g.parameter("w", &[3, 4]);
    let logits = g.matmul(x, w);
    let labels = g.input("labels", &[1, 4]);
    let ce = g.cross_entropy_loss(logits, labels);
    let coef = g.scalar(0.3);
    let loss = g.mul(ce, coef);
    g.set_outputs(vec![loss]);

    let mut session = build_session(&g);
    let x_data: Vec<f32> = vec![-0.5, 0.2, 0.7];
    let labels_data: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];
    let w_init: Vec<f32> = (0..12).map(|i| (i as f32) * 0.1 - 0.5).collect();
    session.set_parameter("w", &w_init);
    let set_inputs = |s: &mut meganeura::Session| {
        s.set_input("x", &x_data);
        s.set_input("labels", &labels_data);
    };
    gradcheck_param(&mut session, "w", 12, &set_inputs);
}

/// `bce_loss(sigmoid(logits), labels) * coef`: mirrors the CE test but
/// for binary cross-entropy.
#[test]
fn bce_loss_with_coef_scales_param_gradient() {
    let mut g = Graph::new();
    let x = g.input("x", &[2, 3]);
    let w = g.parameter("w", &[3, 1]);
    let logits = g.matmul(x, w);
    let pred = g.sigmoid(logits);
    let labels = g.input("labels", &[2, 1]);
    let bce = g.bce_loss(pred, labels);
    let coef = g.scalar(0.4);
    let loss = g.mul(bce, coef);
    g.set_outputs(vec![loss]);

    let mut session = build_session(&g);
    let x_data: Vec<f32> = (0..6).map(|i| (i as f32) * 0.2 - 0.4).collect();
    let labels_data: Vec<f32> = vec![0.7, 0.3];
    let w_init: Vec<f32> = vec![0.2, -0.3, 0.5];
    session.set_parameter("w", &w_init);
    let set_inputs = |s: &mut meganeura::Session| {
        s.set_input("x", &x_data);
        s.set_input("labels", &labels_data);
    };
    gradcheck_param(&mut session, "w", 3, &set_inputs);
}

/// Multi-loss combination (kindle policy-graph shape) at **batch=1** to
/// dodge the CE/Add buffer-aliasing quirk. Verifies the value-loss coef
/// (which goes mse → mean_all → mul(coef) → add) actually reaches w_val,
/// and that CE gradient still propagates to w_pol.
#[test]
fn multi_loss_with_independent_coefs() {
    let mut g = Graph::new();
    let x = g.input("x", &[1, 3]);
    let w_pol = g.parameter("w_pol", &[3, 4]);
    let w_val = g.parameter("w_val", &[3, 1]);
    let logits = g.matmul(x, w_pol);
    let labels = g.input("labels", &[1, 4]);
    let value = g.matmul(x, w_val);
    let value_target = g.input("vt", &[1, 1]);

    let policy_loss = g.cross_entropy_loss(logits, labels);
    let value_loss_raw = g.mse_loss(value, value_target);
    let v_coef = g.scalar(0.1);
    let value_loss = g.mul(value_loss_raw, v_coef);
    let total = g.add(policy_loss, value_loss);
    g.set_outputs(vec![total]);

    let mut session = build_session(&g);
    let x_data: Vec<f32> = vec![0.5, 0.7, 0.9];
    let labels_data: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];
    let vt_data: Vec<f32> = vec![1.0];
    session.set_parameter(
        "w_pol",
        &(0..12).map(|i| (i as f32) * 0.1 - 0.5).collect::<Vec<_>>(),
    );
    session.set_parameter("w_val", &vec![0.2, -0.1, 0.3]);
    let set_inputs = |s: &mut meganeura::Session| {
        s.set_input("x", &x_data);
        s.set_input("labels", &labels_data);
        s.set_input("vt", &vt_data);
    };
    gradcheck_param(&mut session, "w_pol", 12, &set_inputs);
    gradcheck_param(&mut session, "w_val", 3, &set_inputs);
}

// ─────────────────────────────────────────────────────────────────────
// Coverage for every elementwise / activation / pointwise op, placed
// mid-graph (followed by `mul(coef)`) so that a backward that drops
// grad_output would fail. These tests exist to prevent regression of
// the bug class described in the autodiff.rs contract comment.
// ─────────────────────────────────────────────────────────────────────

/// Generic helper: build a graph `loss = mul(activation(matmul(x, w)), coef)`
/// reduced via `mean_all`, then gradcheck `w`.
fn check_activation_mid_chain(
    activation: impl Fn(&mut Graph, meganeura::NodeId) -> meganeura::NodeId,
    coef: f32,
) {
    let mut g = Graph::new();
    let x = g.input("x", &[2, 3]);
    let w = g.parameter("w", &[3, 2]);
    let z = g.matmul(x, w);
    let activated = activation(&mut g, z);
    let mean = g.mean_all(activated);
    let coef_node = g.scalar(coef);
    let loss = g.mul(mean, coef_node);
    g.set_outputs(vec![loss]);
    let mut session = build_session(&g);
    let x_data: Vec<f32> = (0..6).map(|i| (i as f32) * 0.3 - 0.5).collect();
    let w_init: Vec<f32> = (0..6).map(|i| (i as f32) * 0.2 + 0.1).collect();
    session.set_parameter("w", &w_init);
    let set_inputs = |s: &mut meganeura::Session| {
        s.set_input("x", &x_data);
    };
    gradcheck_param(&mut session, "w", 6, &set_inputs);
}

#[test]
fn relu_mid_chain() {
    check_activation_mid_chain(|g, z| g.relu(z), 0.7);
}

#[test]
fn sigmoid_mid_chain() {
    check_activation_mid_chain(|g, z| g.sigmoid(z), 0.7);
}

#[test]
fn tanh_mid_chain() {
    check_activation_mid_chain(|g, z| g.tanh(z), 0.7);
}

#[test]
fn neg_mid_chain() {
    check_activation_mid_chain(|g, z| g.neg(z), 0.7);
}

#[test]
fn abs_mid_chain() {
    check_activation_mid_chain(|g, z| g.abs(z), 0.7);
}

/// `log_softmax(...)` mid-chain at **batch > 1**. Originally caught
/// **two** meganeura bugs: (a) LogSoftmax forward dispatched the Softmax
/// shader with no log applied (compile.rs comment was a lie), and
/// (b) LogSoftmax backward reduced the wrong axis (sum over batches
/// instead of features). Both fixed 2026-04-29.
#[test]
fn log_softmax_mid_chain() {
    let mut g = Graph::new();
    let x = g.input("x", &[2, 3]);
    let w = g.parameter("w", &[3, 4]);
    let z = g.matmul(x, w);
    let lsm = g.log_softmax(z);
    let mean = g.mean_all(lsm);
    let coef = g.scalar(0.5);
    let loss = g.mul(mean, coef);
    g.set_outputs(vec![loss]);
    let mut session = build_session(&g);
    let x_data: Vec<f32> = (0..6).map(|i| (i as f32) * 0.3 - 0.5).collect();
    let w_init: Vec<f32> = (0..12).map(|i| (i as f32) * 0.1 - 0.4).collect();
    session.set_parameter("w", &w_init);
    let set_inputs = |s: &mut meganeura::Session| {
        s.set_input("x", &x_data);
    };
    gradcheck_param(&mut session, "w", 12, &set_inputs);
}

/// `softmax(...) * y` — non-trivial test of Softmax backward at
/// **batch > 1**. The original `mean_all(softmax(...))` test passed
/// trivially (mean of softmax = 1/K constant, so gradient is identically
/// 0). This version weights softmax by per-class targets so the loss
/// actually depends on the parameters, exposing whether the backward
/// reduces over the correct axis.
#[test]
fn softmax_mid_chain() {
    let mut g = Graph::new();
    let x = g.input("x", &[2, 3]);
    let w = g.parameter("w", &[3, 4]);
    let z = g.matmul(x, w);
    let sm = g.softmax(z);
    // Weight softmax probabilities by per-element targets, then mean.
    // mean(targets ⊙ softmax(z)) is non-constant in z, so gradient is
    // nonzero — exercises Softmax backward's per-row reduction.
    let targets = g.input("targets", &[2, 4]);
    let weighted = g.mul(sm, targets);
    let mean = g.mean_all(weighted);
    let coef = g.scalar(0.5);
    let loss = g.mul(mean, coef);
    g.set_outputs(vec![loss]);
    let mut session = build_session(&g);
    let x_data: Vec<f32> = (0..6).map(|i| (i as f32) * 0.3 - 0.5).collect();
    let targets_data: Vec<f32> = vec![1.0, 0.5, -0.2, 0.8, -1.0, 0.3, 0.7, -0.4];
    let w_init: Vec<f32> = (0..12).map(|i| (i as f32) * 0.1 - 0.4).collect();
    session.set_parameter("w", &w_init);
    let set_inputs = |s: &mut meganeura::Session| {
        s.set_input("x", &x_data);
        s.set_input("targets", &targets_data);
    };
    gradcheck_param(&mut session, "w", 12, &set_inputs);
}

/// `transpose` is on the gradient path for many matmul-heavy graphs;
/// verify the (now-fixed) chain rule transposes the gradient too.
#[test]
fn transpose_mid_chain() {
    let mut g = Graph::new();
    let x = g.input("x", &[2, 3]);
    let w = g.parameter("w", &[3, 4]);
    let z = g.matmul(x, w); // [2, 4]
    let t = g.transpose(z); // [4, 2]
    let mean = g.mean_all(t);
    let coef = g.scalar(0.5);
    let loss = g.mul(mean, coef);
    g.set_outputs(vec![loss]);
    let mut session = build_session(&g);
    let x_data: Vec<f32> = (0..6).map(|i| (i as f32) * 0.3 - 0.5).collect();
    let w_init: Vec<f32> = (0..12).map(|i| (i as f32) * 0.1 - 0.4).collect();
    session.set_parameter("w", &w_init);
    let set_inputs = |s: &mut meganeura::Session| {
        s.set_input("x", &x_data);
    };
    gradcheck_param(&mut session, "w", 12, &set_inputs);
}

/// `recip(x)` — also exercises Mul backward through the recip identity.
#[test]
fn recip_mid_chain() {
    let mut g = Graph::new();
    let x = g.input("x", &[2, 3]);
    let w = g.parameter("w", &[3, 2]);
    let z = g.matmul(x, w);
    // Add a constant to keep recip away from 0.
    let bias = g.constant(vec![5.0; 4], &[2, 2]);
    let z = g.add(z, bias);
    let r = g.recip(z);
    let mean = g.mean_all(r);
    let coef = g.scalar(0.5);
    let loss = g.mul(mean, coef);
    g.set_outputs(vec![loss]);
    let mut session = build_session(&g);
    let x_data: Vec<f32> = (0..6).map(|i| (i as f32) * 0.3 - 0.5).collect();
    let w_init: Vec<f32> = (0..6).map(|i| (i as f32) * 0.2 + 0.1).collect();
    session.set_parameter("w", &w_init);
    let set_inputs = |s: &mut meganeura::Session| {
        s.set_input("x", &x_data);
    };
    gradcheck_param(&mut session, "w", 6, &set_inputs);
}

/// Regression test for the meganeura buffer-aliasing quirk: with `batch>1`,
/// a `mul(ce, coef)` does NOT compute `coef · CE` in the forward, because
/// the CE shader writes per-batch partials to a `batch*4`-byte buffer
/// while IR claims shape `[1]`. Downstream binary ops (mul/add) only
/// process element 0 and thus only see `partial_loss_0`. read_loss then
/// returns `coef · partial_loss_0`, which **does NOT match** what either
/// a manual `coef · sum(partial_losses)` or the autodiff backward
/// (which uses the FULL gradient (sm·S − labels)/B) computes. Test is
/// `#[ignore]`d — it documents a known meganeura design issue, not a fix
/// the autodiff side can address. Remove the ignore once the buffer/IR
/// shape mismatch is resolved at the compile/runtime layer.
#[test]
#[ignore = "meganeura CE buffer-aliasing: shape [1] but batch*4 bytes — see body"]
fn cross_entropy_buffer_aliasing_at_batch_2() {
    let mut g = Graph::new();
    let x = g.input("x", &[2, 3]);
    let w = g.parameter("w", &[3, 4]);
    let logits = g.matmul(x, w);
    let labels = g.input("labels", &[2, 4]);
    let ce = g.cross_entropy_loss(logits, labels);
    let coef = g.scalar(0.3);
    let loss = g.mul(ce, coef);
    g.set_outputs(vec![loss]);

    let mut session = build_session(&g);
    let x_data: Vec<f32> = (0..6).map(|i| (i as f32) * 0.4 - 1.0).collect();
    let labels_data: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1];
    let w_init: Vec<f32> = (0..12).map(|i| (i as f32) * 0.1 - 0.5).collect();
    session.set_parameter("w", &w_init);
    let set_inputs = |s: &mut meganeura::Session| {
        s.set_input("x", &x_data);
        s.set_input("labels", &labels_data);
    };
    gradcheck_param(&mut session, "w", 12, &set_inputs);
}
