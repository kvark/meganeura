//! Softplus without losing its small positive value or negative-tail gradient.
use crate::schedule::{PointwiseDAG, Pw};

pub(super) fn forward(beta: f32) -> PointwiseDAG {
    // log1p(t), t = exp(-abs(beta*x)), with compensation for rounded 1+t.
    // If 1+t rounds to 1, return t; otherwise use log(1+t)*t/((1+t)-1).
    // The masked denominator is always nonzero, including exp underflow.
    // Keep max(x,0) outside beta scaling to avoid avoidable positive overflow.
    PointwiseDAG {
        n_inputs: 1,
        ops: vec![
            Pw::LoadInput(0),
            Pw::const_f32(beta),
            Pw::Mul(0, 1),
            Pw::Abs(2),
            Pw::Neg(3),
            Pw::Exp(4),
            Pw::const_f32(1.0),
            Pw::Add(6, 5),
            Pw::Sub(7, 6),
            Pw::const_f32(0.0),
            Pw::Greater(8, 9),
            Pw::Sub(6, 10),
            Pw::Add(8, 11),
            Pw::Div(5, 12),
            Pw::Log(7),
            Pw::Mul(14, 13),
            Pw::Mul(11, 5),
            Pw::Add(15, 16),
            Pw::const_f32(beta.recip()),
            Pw::Mul(17, 18),
            Pw::Relu(0),
            Pw::Add(20, 19),
        ],
        output: 21,
    }
}

pub(super) fn backward(beta: f32) -> PointwiseDAG {
    // sigmoid(beta*x), evaluated without 1-sigmoid(abs(beta*x)) cancellation
    // or exp(-beta*x) overflow. Its value at zero is exactly one half.
    PointwiseDAG {
        n_inputs: 2,
        ops: vec![
            Pw::LoadInput(0),
            Pw::LoadInput(1),
            Pw::const_f32(beta),
            Pw::Mul(1, 2),
            Pw::Abs(3),
            Pw::Neg(4),
            Pw::Exp(5),
            Pw::const_f32(1.0),
            Pw::Add(7, 6),
            Pw::const_f32(0.0),
            Pw::Greater(3, 9),
            Pw::Sub(7, 10),
            Pw::Mul(11, 6),
            Pw::Add(10, 12),
            Pw::Div(13, 8),
            Pw::Mul(0, 14),
        ],
        output: 15,
    }
}
