//! Regression test for the swapped workgroup axes in the 1x1-conv
//! backward matmul dispatches (compile.rs, Op::Conv2dGradInput /
//! Op::Conv2dGradWeight 1x1 shortcuts).
//!
//! Minimal failing pattern (found by shrinking a diffusion U-Net):
//!   conv3x3 -> conv3x3 -> add(residual) -> conv1x1 -> mse
//! The residual add routes the 1x1 conv's grad_input directly into the
//! first conv's weight gradient. With the swapped dispatch, rows past
//! the first M tile of grad_input were never written and read back
//! stale (aliased) memory — but only when batch*H*W exceeds the tile
//! coverage, so this checks spatial sizes on both sides of it.

use meganeura::Graph;

const EPS: f32 = 1e-2;
const TOL: f32 = 5e-2;

fn ramp(n: usize, scale: f32, offset: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 37 + 11) % 19) as f32 / 9.0 - 1.0)
        .map(|v| v * scale + offset)
        .collect()
}

fn check_at(res: u32) {
    let (in_c, mid_c, out_c) = (3u32, 4u32, 2u32);
    let in_size = (in_c * res * res) as usize;
    let out_size = (out_c * res * res) as usize;

    let mut g = Graph::new();
    let x_in = g.input("x", &[in_size]);
    let w0 = g.parameter("w0", &[(mid_c * in_c * 9) as usize]);
    let x = g.conv2d(x_in, w0, 1, in_c, res, res, mid_c, 3, 3, 1, 1);
    let w1 = g.parameter("w1", &[(mid_c * mid_c * 9) as usize]);
    let h = g.conv2d(x, w1, 1, mid_c, res, res, mid_c, 3, 3, 1, 1);
    let sum = g.add(x, h);
    let w2 = g.parameter("w2", &[(out_c * mid_c) as usize]);
    let pred = g.conv2d(sum, w2, 1, mid_c, res, res, out_c, 1, 1, 1, 0);
    let target = g.input("target", &[out_size]);
    let loss = g.mse_loss(pred, target);
    g.set_outputs(vec![loss]);

    let mut session = meganeura::build(&g, meganeura::SessionConfig::from_env()).0;
    session.set_parameter("w0", &ramp((mid_c * in_c * 9) as usize, 0.3, 0.05));
    session.set_parameter("w1", &ramp((mid_c * mid_c * 9) as usize, 0.3, 0.05));
    session.set_parameter("w2", &ramp((out_c * mid_c) as usize, 0.3, 0.05));
    let x_data = ramp(in_size, 1.0, 0.0);
    let t_data = ramp(out_size, 1.0, 0.3);
    let set_inputs = move |s: &mut meganeura::Session| {
        s.set_input("x", &x_data);
        s.set_input("target", &t_data);
    };

    let n = (mid_c * in_c * 9) as usize;
    set_inputs(&mut session);
    session.step();
    session.wait();
    let mut analytical = vec![0.0f32; n];
    session.read_param_grad("w0", &mut analytical);

    let mut params = vec![0.0f32; n];
    session.read_param("w0", &mut params);
    let baseline = params.clone();

    for i in (0..n).step_by((n / 4).max(1)).take(4) {
        let orig = baseline[i];
        params[i] = orig + EPS;
        session.set_parameter("w0", &params);
        set_inputs(&mut session);
        session.step();
        session.wait();
        let l_plus = session.read_loss();

        params[i] = orig - EPS;
        session.set_parameter("w0", &params);
        set_inputs(&mut session);
        session.step();
        session.wait();
        let l_minus = session.read_loss();

        params[i] = orig;
        session.set_parameter("w0", &params);
        let numerical = (l_plus - l_minus) / (2.0 * EPS);
        let a = analytical[i];
        let abs_diff = (a - numerical).abs();
        let rel_diff = abs_diff / a.abs().max(numerical.abs()).max(1e-6);
        assert!(
            abs_diff <= TOL || rel_diff <= TOL,
            "res={res}: w0[{i}] analytical={a:+.5} numerical={numerical:+.5}"
        );
    }
}

#[test]
fn conv1x1_grad_through_residual_res8() {
    // Below the tile-coverage threshold: passed even with the bug.
    check_at(8);
}

#[test]
fn conv1x1_grad_through_residual_res16() {
    // First failing size with the bug (batch*H*W = 256 > 128 covered rows).
    check_at(16);
}

#[test]
fn conv1x1_grad_through_residual_res64() {
    // The size at which van-world SR training plateaued.
    check_at(64);
}
