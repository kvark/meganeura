//! A deep chain of dense layers for the submission and profiling suites.
//! Every op consumes the previous one's output, so submission and capture
//! boundaries land in the middle of real dependencies.

use meganeura::{Graph, Session};

#[allow(dead_code)] // Each integration target compiles this shared module independently.
pub fn build_chain(layers: usize, rows: usize, dim: usize) -> Graph {
    let mut g = Graph::new();
    let mut x = g.input("x", &[rows, dim]);
    for i in 0..layers {
        let w = g.parameter(&format!("w{i}"), &[dim, dim]);
        let b = g.parameter(&format!("b{i}"), &[dim]);
        x = g.matmul(x, w);
        x = g.bias_add(x, b);
        x = g.gelu(x);
        let ln_w = g.parameter(&format!("ln{i}.w"), &[dim]);
        let ln_b = g.parameter(&format!("ln{i}.b"), &[dim]);
        x = g.layer_norm(x, ln_w, ln_b, 1e-5);
    }
    g.set_outputs(vec![x]);
    g
}

#[allow(dead_code)] // Each integration target compiles this shared module independently.
pub fn seed(session: &mut Session, layers: usize, dim: usize) {
    for i in 0..layers {
        let w: Vec<f32> = (0..dim * dim)
            .map(|k| ((k + i * 7) as f32 * 0.017).sin() * 0.1)
            .collect();
        session.set_parameter(&format!("w{i}"), &w);
        let b: Vec<f32> = (0..dim)
            .map(|k| ((k + i) as f32 * 0.03).cos() * 0.01)
            .collect();
        session.set_parameter(&format!("b{i}"), &b);
        session.set_parameter(&format!("ln{i}.w"), &vec![1.0f32; dim]);
        session.set_parameter(&format!("ln{i}.b"), &vec![0.0f32; dim]);
    }
}

/// The chain's `x` input.
#[allow(dead_code)] // Each integration target compiles this shared module independently.
pub fn input(rows: usize, dim: usize) -> Vec<f32> {
    (0..rows * dim)
        .map(|i| (i as f32 * 0.011).sin() * 0.5)
        .collect()
}
