//! Large-size gradchecks for spatial reductions. The van-world overfit
//! diagnostic converges at res 8 and freezes at res 64 with identical
//! topology, implicating size-dependent backward bugs. Each test here
//! reruns a gradcheck_vision case at the failing scale (spatial 4096,
//! i.e. 64x64). Element subsets are checked (finite diff at this size is
//! otherwise too slow); tolerances are loose — we hunt factor-of-N or
//! sign errors, not 1% noise.

use meganeura::Graph;

const EPS: f32 = 2e-2;
const TOL: f32 = 5e-2;

fn ramp(n: usize, scale: f32, offset: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 37 + 11) % 19) as f32 / 9.0 - 1.0)
        .map(|v| v * scale + offset)
        .collect()
}

/// Check `n_check` strided elements of `param_name` against central
/// finite differences.
fn gradcheck_sampled(
    session: &mut meganeura::Session,
    param_name: &str,
    n: usize,
    n_check: usize,
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

    let stride = (n / n_check).max(1);
    let mut failures = Vec::new();
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for i in (0..n).step_by(stride).take(n_check) {
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
        let numerical = (l_plus - l_minus) / (2.0 * EPS);
        let a = analytical[i];
        let abs_diff = (a - numerical).abs();
        let rel_diff = abs_diff / a.abs().max(numerical.abs()).max(1e-6);
        max_abs = max_abs.max(abs_diff);
        max_rel = max_rel.max(rel_diff);
        if abs_diff > TOL && rel_diff > TOL {
            failures.push((i, a, numerical));
        }
    }
    session.set_parameter(param_name, &baseline);

    if !failures.is_empty() {
        for (i, a, num) in &failures {
            eprintln!(
                "  {param_name}[{i}] analytical={a:+.6}  numerical={num:+.6}  ratio={:.4}",
                a / num.max(1e-12)
            );
        }
        panic!(
            "gradcheck failed for `{param_name}` at large size: \
             max_abs_diff={max_abs:.3e}, max_rel_diff={max_rel:.3e}"
        );
    }
    println!("{param_name}: max_abs={max_abs:.2e} max_rel={max_rel:.2e} ok");
}

/// group_norm at spatial=4096 (64x64), C=4, 2 groups → group_size 8192,
/// far beyond one 256-thread workgroup stride. Checks input-path,
/// weight, and bias gradients.
#[test]
fn group_norm_large_spatial() {
    let (batch, c, spatial, groups) = (1u32, 4u32, 4096u32, 2u32);
    let x_size = (batch * c * spatial) as usize;

    let mut g = Graph::new();
    let x = g.input("x", &[x_size]);
    let w = g.parameter("w", &[x_size]);
    let xw = g.mul(x, w);
    let gn_w = g.parameter("gn_w", &[c as usize]);
    let gn_b = g.parameter("gn_b", &[c as usize]);
    let y = g.group_norm(xw, gn_w, gn_b, batch, c, spatial, groups, 1e-5);
    let t = g.input("t", &[x_size]);
    let weighted = g.mul(y, t);
    let mean = g.mean_all(weighted);
    let coef = g.scalar(1.7);
    let loss = g.mul(mean, coef);
    g.set_outputs(vec![loss]);

    let mut session = meganeura::build(&g, meganeura::SessionConfig::from_env()).0;
    let x_data = ramp(x_size, 1.0, 0.5);
    let t_data = ramp(x_size, 1.0, 0.3);
    session.set_parameter("w", &ramp(x_size, 0.5, 1.0));
    session.set_parameter("gn_w", &ramp(c as usize, 0.5, 1.0));
    session.set_parameter("gn_b", &ramp(c as usize, 0.3, 0.0));
    let set_inputs = |s: &mut meganeura::Session| {
        s.set_input("x", &x_data);
        s.set_input("t", &t_data);
    };
    gradcheck_sampled(&mut session, "gn_w", c as usize, 4, &set_inputs);
    gradcheck_sampled(&mut session, "gn_b", c as usize, 4, &set_inputs);
    gradcheck_sampled(&mut session, "w", x_size, 8, &set_inputs);
}

/// conv2d 3x3 s1 p1 at 64x64 — kernel gradient reduces over 4096
/// output positions per kernel element.
#[test]
fn conv2d_large_spatial_kernel_grad() {
    let (batch, in_c, h, w_, out_c) = (1u32, 2u32, 64u32, 64u32, 2u32);
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

    let mut session = meganeura::build(&g, meganeura::SessionConfig::from_env()).0;
    let x_data = ramp(in_size, 1.0, 0.0);
    let t_data = ramp(out_size, 1.0, 0.2);
    session.set_parameter("k", &ramp(k_size, 0.4, 0.0));
    let set_inputs = |s: &mut meganeura::Session| {
        s.set_input("x", &x_data);
        s.set_input("t", &t_data);
    };
    gradcheck_sampled(&mut session, "k", k_size, 12, &set_inputs);
}

/// The FiLM broadcast pattern at spatial=4096: emb column [C,1] matmul
/// ones [1,4096]. The backward w.r.t. the column reduces over 4096
/// columns of the plane gradient.
#[test]
fn film_matmul_large_spatial() {
    let (c, spatial) = (8usize, 4096usize);

    let mut g = Graph::new();
    let e = g.parameter("e", &[c, 1]);
    let ones = g.constant(vec![1.0; spatial], &[1, spatial]);
    let plane = g.matmul(e, ones); // [c, spatial]
    let flat = g.reshape(plane, &[c * spatial]);
    let t = g.input("t", &[c * spatial]);
    let weighted = g.mul(flat, t);
    let mean = g.mean_all(weighted);
    let coef = g.scalar(0.9);
    let loss = g.mul(mean, coef);
    g.set_outputs(vec![loss]);

    let mut session = meganeura::build(&g, meganeura::SessionConfig::from_env()).0;
    let t_data = ramp(c * spatial, 1.0, 0.2);
    session.set_parameter("e", &ramp(c, 0.5, 0.1));
    let set_inputs = |s: &mut meganeura::Session| {
        s.set_input("t", &t_data);
    };
    gradcheck_sampled(&mut session, "e", c, 8, &set_inputs);
}

/// mse_loss over 40960 elements (10ch x 64x64) — both the loss readback
/// reduction and its gradient at scale.
#[test]
fn mse_loss_large() {
    let n = 40960usize;
    let mut g = Graph::new();
    let x = g.input("x", &[n]);
    let w = g.parameter("w", &[n]);
    let pred = g.mul(x, w);
    let target = g.input("target", &[n]);
    let loss = g.mse_loss(pred, target);
    g.set_outputs(vec![loss]);

    let mut session = meganeura::build(&g, meganeura::SessionConfig::from_env()).0;
    let x_data = ramp(n, 1.0, 0.0);
    let target_data = ramp(n, 1.0, 0.4);
    session.set_parameter("w", &ramp(n, 0.5, 0.2));
    let set_inputs = |s: &mut meganeura::Session| {
        s.set_input("x", &x_data);
        s.set_input("target", &target_data);
    };
    gradcheck_sampled(&mut session, "w", n, 8, &set_inputs);
}
