//! Numerical-gradient verification for the vision ops used by diffusion
//! U-Nets (van-world): conv2d (stride 1 and 2, 3x3 and 1x1), group_norm,
//! upsample_2x, concat, embedding, and silu — none of which were covered
//! by `gradcheck.rs`. Same harness: op mid-graph, output weighted by a
//! non-uniform target so the loss is non-degenerate, `* coef` to catch
//! dropped upstream gradients, analytical vs central finite differences.

use meganeura::{Graph, build_session};

const EPS: f32 = 5e-3;
const TOL: f32 = 1e-2;

fn gradcheck_param(
    session: &mut meganeura::Session,
    param_name: &str,
    n: usize,
    set_inputs: &dyn Fn(&mut meganeura::Session),
) {
    let mut params = vec![0.0f32; n];
    session.read_param(param_name, &mut params);
    let baseline = params.clone();

    set_inputs(session);
    session.step();
    session.wait();
    let mut analytical = vec![0.0f32; n];
    session.read_param_grad(param_name, &mut analytical);

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

/// Deterministic pseudo-random-ish values in roughly [-1, 1].
fn ramp(n: usize, scale: f32, offset: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 37 + 11) % 19) as f32 / 9.0 - 1.0)
        .map(|v| v * scale + offset)
        .collect()
}

/// silu mid-chain — used 5x per resblock chain in van-world but absent
/// from the activation gradchecks.
#[test]
fn silu_mid_chain() {
    let mut g = Graph::new();
    let x = g.input("x", &[2, 3]);
    let w = g.parameter("w", &[3, 2]);
    let z = g.matmul(x, w);
    let a = g.silu(z);
    let mean = g.mean_all(a);
    let coef = g.scalar(0.7);
    let loss = g.mul(mean, coef);
    g.set_outputs(vec![loss]);
    let mut session = build_session(&g);
    let x_data = ramp(6, 1.0, 0.0);
    session.set_parameter("w", &ramp(6, 0.5, 0.1));
    let set_inputs = |s: &mut meganeura::Session| s.set_input("x", &x_data);
    gradcheck_param(&mut session, "w", 6, &set_inputs);
}

/// conv2d 3x3 stride 1 pad 1 — gradient w.r.t. the kernel.
/// Matches van-world's resblock convs.
#[test]
fn conv2d_3x3_s1_kernel_grad() {
    let (batch, in_c, h, w_, out_c) = (2u32, 2u32, 4u32, 4u32, 2u32);
    let in_size = (batch * in_c * h * w_) as usize;
    let out_size = (batch * out_c * h * w_) as usize;
    let k_size = (out_c * in_c * 3 * 3) as usize;

    let mut g = Graph::new();
    let x = g.input("x", &[in_size]);
    let k = g.parameter("k", &[k_size]);
    let y = g.conv2d(x, k, batch, in_c, h, w_, out_c, 3, 3, 1, 1);
    let t = g.input("t", &[out_size]);
    let weighted = g.mul(y, t);
    let mean = g.mean_all(weighted);
    let coef = g.scalar(1.3);
    let loss = g.mul(mean, coef);
    g.set_outputs(vec![loss]);

    let mut session = build_session(&g);
    let x_data = ramp(in_size, 1.0, 0.0);
    let t_data = ramp(out_size, 1.0, 0.2);
    session.set_parameter("k", &ramp(k_size, 0.4, 0.0));
    let set_inputs = |s: &mut meganeura::Session| {
        s.set_input("x", &x_data);
        s.set_input("t", &t_data);
    };
    gradcheck_param(&mut session, "k", k_size, &set_inputs);
}

/// conv2d 3x3 stride 2 pad 1 — gradient w.r.t. the kernel.
/// Matches van-world's encoder downsample convs.
#[test]
fn conv2d_3x3_s2_kernel_grad() {
    let (batch, in_c, h, w_, out_c) = (2u32, 2u32, 4u32, 4u32, 2u32);
    let oh = (h + 2 - 3) / 2 + 1; // 2
    let in_size = (batch * in_c * h * w_) as usize;
    let out_size = (batch * out_c * oh * oh) as usize;
    let k_size = (out_c * in_c * 3 * 3) as usize;

    let mut g = Graph::new();
    let x = g.input("x", &[in_size]);
    let k = g.parameter("k", &[k_size]);
    let y = g.conv2d(x, k, batch, in_c, h, w_, out_c, 3, 3, 2, 1);
    let t = g.input("t", &[out_size]);
    let weighted = g.mul(y, t);
    let mean = g.mean_all(weighted);
    let coef = g.scalar(1.3);
    let loss = g.mul(mean, coef);
    g.set_outputs(vec![loss]);

    let mut session = build_session(&g);
    let x_data = ramp(in_size, 1.0, 0.0);
    let t_data = ramp(out_size, 1.0, 0.2);
    session.set_parameter("k", &ramp(k_size, 0.4, 0.0));
    let set_inputs = |s: &mut meganeura::Session| {
        s.set_input("x", &x_data);
        s.set_input("t", &t_data);
    };
    gradcheck_param(&mut session, "k", k_size, &set_inputs);
}

/// conv2d 1x1 — gradient w.r.t. the *input* (via an upstream parameter),
/// exercising Conv2dGradInput. Matches van-world's residual projections.
#[test]
fn conv2d_1x1_input_grad() {
    let (batch, in_c, h, w_, out_c) = (2u32, 2u32, 3u32, 3u32, 3u32);
    let in_size = (batch * in_c * h * w_) as usize;
    let out_size = (batch * out_c * h * w_) as usize;
    let k_size = (out_c * in_c) as usize;

    let mut g = Graph::new();
    let x = g.input("x", &[in_size]);
    let w = g.parameter("w", &[in_size]);
    let xw = g.mul(x, w);
    let k_data = ramp(k_size, 0.6, 0.1);
    let k = g.constant(k_data, &[k_size]);
    let y = g.conv2d(xw, k, batch, in_c, h, w_, out_c, 1, 1, 1, 0);
    let t = g.input("t", &[out_size]);
    let weighted = g.mul(y, t);
    let mean = g.mean_all(weighted);
    let coef = g.scalar(0.9);
    let loss = g.mul(mean, coef);
    g.set_outputs(vec![loss]);

    let mut session = build_session(&g);
    let x_data = ramp(in_size, 1.0, 0.0);
    let t_data = ramp(out_size, 1.0, 0.2);
    session.set_parameter("w", &ramp(in_size, 0.5, 0.2));
    let set_inputs = |s: &mut meganeura::Session| {
        s.set_input("x", &x_data);
        s.set_input("t", &t_data);
    };
    gradcheck_param(&mut session, "w", in_size, &set_inputs);
}

/// conv2d 3x3 stride 1 — gradient w.r.t. the *input* with padding in play.
#[test]
fn conv2d_3x3_s1_input_grad() {
    let (batch, in_c, h, w_, out_c) = (2u32, 2u32, 4u32, 4u32, 2u32);
    let in_size = (batch * in_c * h * w_) as usize;
    let out_size = (batch * out_c * h * w_) as usize;
    let k_size = (out_c * in_c * 3 * 3) as usize;

    let mut g = Graph::new();
    let x = g.input("x", &[in_size]);
    let w = g.parameter("w", &[in_size]);
    let xw = g.mul(x, w);
    let k_data = ramp(k_size, 0.4, 0.0);
    let k = g.constant(k_data, &[k_size]);
    let y = g.conv2d(xw, k, batch, in_c, h, w_, out_c, 3, 3, 1, 1);
    let t = g.input("t", &[out_size]);
    let weighted = g.mul(y, t);
    let mean = g.mean_all(weighted);
    let coef = g.scalar(0.9);
    let loss = g.mul(mean, coef);
    g.set_outputs(vec![loss]);

    let mut session = build_session(&g);
    let x_data = ramp(in_size, 1.0, 0.0);
    let t_data = ramp(out_size, 1.0, 0.2);
    session.set_parameter("w", &ramp(in_size, 0.5, 0.2));
    let set_inputs = |s: &mut meganeura::Session| {
        s.set_input("x", &x_data);
        s.set_input("t", &t_data);
    };
    gradcheck_param(&mut session, "w", in_size, &set_inputs);
}

/// group_norm — gradients w.r.t. the affine weight and bias.
/// C=4, 2 groups, batch 2, matching van-world's GN-heavy resblocks.
#[test]
fn group_norm_weight_bias_grad() {
    let (batch, c, spatial, groups) = (2u32, 4u32, 4u32, 2u32);
    let x_size = (batch * c * spatial) as usize;

    let mut g = Graph::new();
    let x = g.input("x", &[x_size]);
    let w = g.parameter("w", &[c as usize]);
    let b = g.parameter("b", &[c as usize]);
    let y = g.group_norm(x, w, b, batch, c, spatial, groups, 1e-5);
    let t = g.input("t", &[x_size]);
    let weighted = g.mul(y, t);
    let mean = g.mean_all(weighted);
    let coef = g.scalar(1.7);
    let loss = g.mul(mean, coef);
    g.set_outputs(vec![loss]);

    let mut session = build_session(&g);
    let x_data = ramp(x_size, 1.0, 0.0);
    let t_data = ramp(x_size, 1.0, 0.3);
    session.set_parameter("w", &ramp(c as usize, 0.5, 1.0));
    session.set_parameter("b", &ramp(c as usize, 0.3, 0.0));
    let set_inputs = |s: &mut meganeura::Session| {
        s.set_input("x", &x_data);
        s.set_input("t", &t_data);
    };
    gradcheck_param(&mut session, "w", c as usize, &set_inputs);
    gradcheck_param(&mut session, "b", c as usize, &set_inputs);
}

/// group_norm — gradient w.r.t. the *input* (via an upstream parameter),
/// exercising GroupNormGradInput's mean/variance chain rule.
#[test]
fn group_norm_input_grad() {
    let (batch, c, spatial, groups) = (2u32, 4u32, 4u32, 2u32);
    let x_size = (batch * c * spatial) as usize;

    let mut g = Graph::new();
    let x = g.input("x", &[x_size]);
    let w = g.parameter("w", &[x_size]);
    let xw = g.mul(x, w);
    let gn_w = g.constant(vec![1.0; c as usize], &[c as usize]);
    let gn_b = g.constant(vec![0.0; c as usize], &[c as usize]);
    let y = g.group_norm(xw, gn_w, gn_b, batch, c, spatial, groups, 1e-5);
    let t = g.input("t", &[x_size]);
    let weighted = g.mul(y, t);
    let mean = g.mean_all(weighted);
    let coef = g.scalar(1.7);
    let loss = g.mul(mean, coef);
    g.set_outputs(vec![loss]);

    let mut session = build_session(&g);
    let x_data = ramp(x_size, 1.0, 0.5);
    let t_data = ramp(x_size, 1.0, 0.3);
    session.set_parameter("w", &ramp(x_size, 0.5, 1.0));
    let set_inputs = |s: &mut meganeura::Session| {
        s.set_input("x", &x_data);
        s.set_input("t", &t_data);
    };
    gradcheck_param(&mut session, "w", x_size, &set_inputs);
}

/// upsample_2x — gradient w.r.t. the input, exercising Upsample2xGrad's
/// 4-to-1 accumulation. A backward that averages instead of sums (or
/// mis-indexes the 2x2 block) fails here.
#[test]
fn upsample_2x_input_grad() {
    let (batch, c, h, w_) = (2u32, 2u32, 2u32, 2u32);
    let in_size = (batch * c * h * w_) as usize;
    let out_size = in_size * 4;

    let mut g = Graph::new();
    let x = g.input("x", &[in_size]);
    let w = g.parameter("w", &[in_size]);
    let xw = g.mul(x, w);
    let y = g.upsample_2x(xw, batch, c, h, w_);
    let t = g.input("t", &[out_size]);
    let weighted = g.mul(y, t);
    let mean = g.mean_all(weighted);
    let coef = g.scalar(0.8);
    let loss = g.mul(mean, coef);
    g.set_outputs(vec![loss]);

    let mut session = build_session(&g);
    let x_data = ramp(in_size, 1.0, 0.0);
    let t_data = ramp(out_size, 1.0, 0.2);
    session.set_parameter("w", &ramp(in_size, 0.5, 0.3));
    let set_inputs = |s: &mut meganeura::Session| {
        s.set_input("x", &x_data);
        s.set_input("t", &t_data);
    };
    gradcheck_param(&mut session, "w", in_size, &set_inputs);
}

/// concat — gradients w.r.t. *both* branches, exercising the SplitA/SplitB
/// backward routing at batch 2 (channel offsets differ per batch element).
#[test]
fn concat_both_branches_grad() {
    let (batch, ca, cb, spatial) = (2u32, 1u32, 3u32, 4u32);
    let a_size = (batch * ca * spatial) as usize;
    let b_size = (batch * cb * spatial) as usize;
    let out_size = a_size + b_size;

    let mut g = Graph::new();
    let xa = g.input("xa", &[a_size]);
    let xb = g.input("xb", &[b_size]);
    let wa = g.parameter("wa", &[a_size]);
    let wb = g.parameter("wb", &[b_size]);
    let a = g.mul(xa, wa);
    let b = g.mul(xb, wb);
    let y = g.concat(a, b, batch, ca, cb, spatial);
    let t = g.input("t", &[out_size]);
    let weighted = g.mul(y, t);
    let mean = g.mean_all(weighted);
    let coef = g.scalar(1.1);
    let loss = g.mul(mean, coef);
    g.set_outputs(vec![loss]);

    let mut session = build_session(&g);
    let xa_data = ramp(a_size, 1.0, 0.0);
    let xb_data = ramp(b_size, 1.0, 0.1);
    let t_data = ramp(out_size, 1.0, 0.2);
    session.set_parameter("wa", &ramp(a_size, 0.5, 0.2));
    session.set_parameter("wb", &ramp(b_size, 0.5, -0.1));
    let set_inputs = |s: &mut meganeura::Session| {
        s.set_input("xa", &xa_data);
        s.set_input("xb", &xb_data);
        s.set_input("t", &t_data);
    };
    gradcheck_param(&mut session, "wa", a_size, &set_inputs);
    gradcheck_param(&mut session, "wb", b_size, &set_inputs);
}

/// embedding — gradient w.r.t. the table via ScatterAdd, with a repeated
/// index to exercise accumulation (rows 1 appears twice).
#[test]
fn embedding_table_grad() {
    let (vocab, dim) = (5usize, 3usize);
    let indices: Vec<u32> = vec![1, 4, 1, 0];
    let out_size = indices.len() * dim;

    let mut g = Graph::new();
    let idx = g.input_u32("idx", &[indices.len()]);
    let table = g.parameter("table", &[vocab, dim]);
    let y = g.embedding(idx, table);
    let t = g.input("t", &[indices.len(), dim]);
    let weighted = g.mul(y, t);
    let mean = g.mean_all(weighted);
    let coef = g.scalar(1.5);
    let loss = g.mul(mean, coef);
    g.set_outputs(vec![loss]);

    let mut session = build_session(&g);
    let t_data = ramp(out_size, 1.0, 0.2);
    session.set_parameter("table", &ramp(vocab * dim, 0.5, 0.0));
    let set_inputs = |s: &mut meganeura::Session| {
        s.set_input_u32("idx", &indices);
        s.set_input("t", &t_data);
    };
    gradcheck_param(&mut session, "table", vocab * dim, &set_inputs);
}
