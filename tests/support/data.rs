//! Deterministic operands, dense references and closeness checks shared by
//! the kernel suites.

/// A PCG-style stream: an LCG step followed by the RXS-M-XS permutation.
#[allow(dead_code)] // Each integration target compiles this shared module independently.
pub struct Pcg(u32);

#[allow(dead_code)] // Each integration target compiles this shared module independently.
impl Pcg {
    pub fn new(seed: u32) -> Self {
        Self(seed | 1)
    }

    pub fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(747796405).wrapping_add(2891336453);
        let w = ((self.0 >> ((self.0 >> 28) + 4)) ^ self.0).wrapping_mul(277803737);
        (w >> 22) ^ w
    }
}

/// `n` deterministic values, uniform in [-1, 1].
#[allow(dead_code)] // Each integration target compiles this shared module independently.
pub fn pcg_inputs(n: usize, seed: u32) -> Vec<f32> {
    let mut rng = Pcg::new(seed);
    (0..n)
        .map(|_| rng.next_u32() as f32 * (2.0 / u32::MAX as f32) - 1.0)
        .collect()
}

/// The f64 dense product, row-major `[m, n]`, by layout: 0 multiplies
/// `A[m, k]` by `B[k, n]`, 1 stores A as `[k, m]` (`MatMulAT`), and 2 stores
/// B as `[n, k]` (`MatMulBT`).
#[allow(dead_code)] // Each integration target compiles this shared module independently.
pub fn matmul_reference(layout: usize, a: &[f32], b: &[f32], [m, n, k]: [usize; 3]) -> Vec<f64> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    let mut c = vec![0.0; m * n];
    for row in 0..m {
        for col in 0..n {
            c[row * n + col] = (0..k)
                .map(|inner| {
                    let ai = if layout == 1 {
                        inner * m + row
                    } else {
                        row * k + inner
                    };
                    let bi = if layout == 2 {
                        col * k + inner
                    } else {
                        inner * n + col
                    };
                    f64::from(a[ai]) * f64::from(b[bi])
                })
                .sum();
        }
    }
    c
}

/// The largest absolute error, relative to the largest reference magnitude,
/// must stay below `tolerance`.
#[allow(dead_code)] // Each integration target compiles this shared module independently.
pub fn assert_close(label: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    // `f32::max` drops NaN, so a non-finite output would otherwise pass.
    assert!(
        actual.iter().all(|v| v.is_finite()),
        "{label}: non-finite output"
    );
    let max_abs = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let scale = expected
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f32, f32::max)
        .max(1e-6);
    let first_bad = actual
        .iter()
        .zip(expected)
        .position(|(a, b)| (a - b).abs() / scale >= tolerance);
    assert!(
        max_abs / scale < tolerance,
        "{label}: relative error {:.3}% (max abs {max_abs:.3e}, scale {scale:.3e}), first bad index {first_bad:?}\n  actual[..8]={:?}\n  expected[..8]={:?}",
        max_abs / scale * 100.0,
        &actual[..8.min(actual.len())],
        &expected[..8.min(expected.len())],
    );
}

/// Relative L2 error of `got` against `want`.
#[allow(dead_code)] // Each integration target compiles this shared module independently.
pub fn relative_l2(got: &[f32], want: &[f32]) -> f64 {
    let error: f64 = got
        .iter()
        .zip(want)
        .map(|(&a, &b)| f64::from(a - b).powi(2))
        .sum();
    let reference: f64 = want.iter().map(|&b| f64::from(b).powi(2)).sum();
    (error / reference.max(f64::EPSILON)).sqrt()
}

/// A dense product by layout (as in `matmul_reference`) of parameters `a`
/// and `b`, optionally with a fused `[m, n]` `bias` and a BF16-rounded output.
/// F16 storage declares F16 parameters; BF16 converts F32 parameters.
#[allow(dead_code)] // Each integration target compiles this shared module independently.
pub fn layout_graph(
    storage: meganeura::DType,
    layout: usize,
    add: bool,
    round: bool,
    [m, n, k]: [usize; 3],
) -> meganeura::Graph {
    use meganeura::{DType, graph::Op};
    let mut g = meganeura::Graph::new();
    let a_shape = if layout == 1 { [k, m] } else { [m, k] };
    let b_shape = if layout == 2 { [n, k] } else { [k, n] };
    let (a, b) = if storage == DType::F16 {
        (
            g.parameter_f16("a", &a_shape),
            g.parameter_f16("b", &b_shape),
        )
    } else {
        let a = g.parameter("a", &a_shape);
        let b = g.parameter("b", &b_shape);
        if storage == DType::BF16 {
            (g.to_bf16(a), g.to_bf16(b))
        } else {
            (a, b)
        }
    };
    let bias = add.then(|| g.parameter("bias", &[m, n]));
    let y = match layout {
        0 => g.matmul(a, b),
        1 => g.matmul_at(a, b),
        _ => g.matmul_bt(a, b),
    };
    if let Some(bias) = bias {
        let node = &mut g.nodes_mut()[y as usize];
        node.op = [
            Op::FusedMatMulAdd,
            Op::FusedMatMulATAdd,
            Op::FusedMatMulBTAdd,
        ][layout]
            .clone();
        node.inputs.push(bias);
    }
    let y = if round {
        let rounded = g.to_bf16(y);
        g.from_bf16(rounded)
    } else {
        y
    };
    g.set_outputs(vec![y]);
    g
}
