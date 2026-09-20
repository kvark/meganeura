//! Gaussian-CDF GELU and its analytical derivative in a single pointwise DAG.
//! Abramowitz–Stegun 7.1.26: erf absolute error <1.5e-7. Approximation values
//! stay in shader registers, not full activation buffers retained for autodiff.

use crate::schedule::{PointwiseDAG, Pw};

fn push(ops: &mut Vec<Pw>, value: Pw) -> u16 {
    let index = ops.len() as u16;
    ops.push(value);
    index
}

pub(super) fn pointwise(backward: bool) -> PointwiseDAG {
    let mut ops = vec![
        Pw::LoadInput(u8::from(backward)), // x, second input on backward
        Pw::Abs(0),
        Pw::const_f32(0.327_591_1 * std::f32::consts::FRAC_1_SQRT_2),
        Pw::Mul(1, 2),
        Pw::const_f32(1.0),
        Pw::Add(3, 4),
        Pw::Recip(5),
        Pw::const_f32(1.061_405_4),
        Pw::Mul(6, 7),
    ];
    let mut polynomial = 8;
    for coefficient in [-1.453_152_1, 1.421_413_8, -0.284_496_72, 0.254_829_6] {
        let coefficient = push(&mut ops, Pw::const_f32(coefficient));
        let sum = push(&mut ops, Pw::Add(polynomial, coefficient));
        polynomial = push(&mut ops, Pw::Mul(sum, 6));
    }
    let square = push(&mut ops, Pw::Mul(0, 0));
    let negative_half = push(&mut ops, Pw::const_f32(-0.5));
    let exponent = push(&mut ops, Pw::Mul(square, negative_half));
    let exponential = push(&mut ops, Pw::Exp(exponent));
    let tail = push(&mut ops, Pw::Mul(polynomial, exponential));
    let zero = push(&mut ops, Pw::const_f32(0.0));
    let positive = push(&mut ops, Pw::Greater(0, zero));
    let twice_positive = push(&mut ops, Pw::Add(positive, positive));
    let sign = push(&mut ops, Pw::Sub(4, twice_positive));
    let signed_tail = push(&mut ops, Pw::Mul(sign, tail));
    let half = push(&mut ops, Pw::const_f32(0.5));
    let half_tail = push(&mut ops, Pw::Mul(half, signed_tail));
    // Negative inputs use the tail directly, avoiding 1-erf cancellation.
    let cdf = push(&mut ops, Pw::Add(positive, half_tail));
    let output = if backward {
        let inverse_sqrt_tau = push(&mut ops, Pw::const_f32(0.398_942_3));
        let density = push(&mut ops, Pw::Mul(exponential, inverse_sqrt_tau));
        let x_density = push(&mut ops, Pw::Mul(0, density));
        let derivative = push(&mut ops, Pw::Add(cdf, x_density));
        let upstream = push(&mut ops, Pw::LoadInput(0));
        push(&mut ops, Pw::Mul(upstream, derivative))
    } else {
        push(&mut ops, Pw::Mul(0, cdf))
    };
    PointwiseDAG {
        n_inputs: if backward { 2 } else { 1 },
        ops,
        output,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evaluate(dag: &PointwiseDAG, inputs: &[f32]) -> f32 {
        let mut values = Vec::<f32>::new();
        for op in &dag.ops {
            let at = |i: u16| values[i as usize];
            values.push(match *op {
                Pw::LoadInput(i) => inputs[i as usize],
                Pw::Const(bits) => f32::from_bits(bits),
                Pw::Abs(a) => at(a).abs(),
                Pw::Recip(a) => at(a).recip(),
                Pw::Exp(a) => at(a).exp(),
                Pw::Add(a, b) => at(a) + at(b),
                Pw::Sub(a, b) => at(a) - at(b),
                Pw::Mul(a, b) => at(a) * at(b),
                Pw::Greater(a, b) => u8::from(at(a) > at(b)) as f32,
                _ => panic!("unexpected GELU primitive"),
            });
        }
        values[dag.output as usize]
    }

    #[test]
    fn gaussian_values_derivatives_and_shader_validate_without_gpu() {
        // Independent Gaussian CDF values, including negative tails and zero.
        for (x, cdf) in [
            (-6.0_f32, 9.865876e-10_f32),
            (-3.0, 0.001_349_898),
            (-1.0, 0.158_655_26),
            (0.0, 0.5),
            (1.0, 0.841_344_7),
            (3.0, 0.998_650_1),
            (6.0, 1.0),
        ] {
            let forward = evaluate(&pointwise(false), &[x]);
            let backward = evaluate(&pointwise(true), &[0.7, x]);
            let expected_gradient = 0.7 * (cdf + x * (-x * x / 2.0).exp() * 0.398_942_3);
            assert!((forward - x * cdf).abs() < 5e-7, "x={x}: {forward}");
            assert!(
                (backward - expected_gradient).abs() < 3e-7,
                "x={x}: {backward}"
            );
        }
        for backward in [false, true] {
            let shader = crate::schedule::lower(&crate::schedule::KernelTemplate::Pointwise {
                dag: pointwise(backward),
                grid: crate::schedule::GridShape::default(),
            });
            let module = naga::front::wgsl::parse_str(&shader.source).unwrap();
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all() ^ naga::valid::ValidationFlags::BINDINGS,
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
        }
    }
}
