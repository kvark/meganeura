//! Softplus without losing its small positive value or negative-tail gradient.
use crate::schedule::{PointwiseDAG, Pw};

pub(super) fn forward(beta: f32) -> PointwiseDAG {
    // t = exp(-abs(beta*x)) lies in [0,1]. Evaluate log1p(t) as
    // 2*atanh(t/(2+t)), using eight odd terms. The argument is at most 1/3;
    // the omitted tail is below 1.1e-9 absolute. Unlike compensated (1+t)-1,
    // this spelling cannot be reassociated into a cancellation-prone log(1+t).
    let mut ops = vec![
        Pw::LoadInput(0),
        Pw::const_f32(beta),
        Pw::Mul(0, 1),
        Pw::Abs(2),
        Pw::Neg(3),
        Pw::Exp(4),
        Pw::const_f32(2.0),
        Pw::Add(6, 5),
        Pw::Div(5, 7),
        Pw::Mul(8, 8),
        Pw::const_f32(1.0 / 15.0),
    ];
    let mut polynomial = 10;
    for denominator in [13.0, 11.0, 9.0, 7.0, 5.0, 3.0, 1.0] {
        let coefficient = ops.len() as u16;
        ops.push(Pw::const_f32(1.0 / denominator));
        let product = ops.len() as u16;
        ops.push(Pw::Mul(9, polynomial));
        polynomial = ops.len() as u16;
        ops.push(Pw::Add(coefficient, product));
    }
    let twice_y = ops.len() as u16;
    ops.push(Pw::Mul(6, 8));
    let logarithm = ops.len() as u16;
    ops.push(Pw::Mul(twice_y, polynomial));
    let scaled = ops.len() as u16;
    ops.push(Pw::Div(logarithm, 1));
    // Keep max(x,0) outside beta scaling to avoid avoidable positive overflow.
    let positive = ops.len() as u16;
    ops.push(Pw::Relu(0));
    let output = ops.len() as u16;
    ops.push(Pw::Add(positive, scaled));
    PointwiseDAG { n_inputs: 1, ops, output }
}

pub(super) fn backward(beta: f32) -> PointwiseDAG {
    // sigmoid(beta*x), without 1-sigmoid(abs(beta*x)) cancellation or
    // exp(-beta*x) overflow. Its value at zero is exactly one half.
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
