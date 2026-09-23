//! Row normalizations, rotary position embeddings and pairwise row ops.

use super::{Error, Tensor};
use crate::graph::{Node, Op, PairwiseGradKind};

fn invalid(node: &Node, reason: impl Into<String>) -> Error {
    Error::Invalid {
        node: node.id,
        reason: reason.into(),
    }
}

/// `[rows, cols]` of a 2D tensor.
fn matrix(node: &Node, t: &Tensor, what: &str) -> Result<(usize, usize), Error> {
    match t.shape[..] {
        [rows, cols] => Ok((rows, cols)),
        ref shape => Err(invalid(node, format!("{what} must be 2D, got {shape:?}"))),
    }
}

fn expect_len(node: &Node, t: &Tensor, len: usize, what: &str) -> Result<(), Error> {
    if t.len() == len {
        Ok(())
    } else {
        Err(invalid(
            node,
            format!("{what} has {} values, expected {len}", t.len()),
        ))
    }
}

/// `1 / sqrt(mean(x²) + eps)`.
fn inv_rms(row: &[f64], eps: f64) -> f64 {
    let mean_sq = row.iter().map(|v| v * v).sum::<f64>() / row.len() as f64;
    1.0 / (mean_sq + eps).sqrt()
}

/// Mean and `1 / sqrt(var + eps)` with the biased (population) variance.
fn moments(row: &[f64], eps: f64) -> (f64, f64) {
    let n = row.len() as f64;
    let mean = row.iter().sum::<f64>() / n;
    let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    (mean, 1.0 / (var + eps).sqrt())
}

/// Inputs `[dy, x, w]` of a norm backward op: checks shapes, returns
/// `(rows, cols)`.
fn norm_grad_inputs(node: &Node, ins: &[&Tensor]) -> Result<(usize, usize), Error> {
    let (dy, x, w) = (ins[0], ins[1], ins[2]);
    let (rows, cols) = matrix(node, x, "x")?;
    expect_len(node, dy, rows * cols, "dy")?;
    expect_len(node, w, cols, "weight")?;
    Ok((rows, cols))
}

/// Where each row of a RoPE input sits in the sequence.
enum Positions<'a> {
    /// `row + offset`.
    Offset(f64),
    /// One absolute position per row.
    PerRow(&'a Tensor),
}

/// The rotation a RoPE-family node applies, in the half-split
/// ("NeoX"/HuggingFace) layout: inside every `head_dim` block, element `i`
/// pairs with `i + head_dim/2`, and pair `i` of a row at position `pos`
/// turns by `pos · theta^(−2i/head_dim) / factor[i]`.
struct Rotation<'a> {
    theta: f64,
    head_dim: usize,
    positions: Positions<'a>,
    factors: Option<&'a Tensor>,
    /// Apply the transposed rotation (the forward rotation's backward).
    inverse: bool,
}

impl Rotation<'_> {
    fn angle(&self, row: usize, pair: usize) -> f64 {
        let pos = match self.positions {
            Positions::Offset(offset) => row as f64 + offset,
            Positions::PerRow(p) => p.data[row],
        };
        let mut freq = self.theta.powf(-2.0 * pair as f64 / self.head_dim as f64);
        if let Some(f) = self.factors {
            freq /= f.data[pair];
        }
        pos * freq
    }
}

/// Read the rotation of a RoPE, RoPEGrad or RoPEPositions node and check
/// its operands. A RoPE row sits at `row + pos_offset`, plus the dynamic
/// offset when a second input is present.
fn rotation<'a>(node: &Node, ins: &[&'a Tensor]) -> Result<Rotation<'a>, Error> {
    let (theta, head_dim, positions, factors, inverse) = match node.op {
        Op::RoPE {
            theta,
            pos_offset,
            head_dim,
            freq_factors,
        } => {
            let dynamic = match ins.get(1) {
                Some(t) if !t.is_empty() => t.data[0],
                Some(_) => return Err(invalid(node, "dynamic offset buffer is empty")),
                None => 0.0,
            };
            let offset = Positions::Offset(f64::from(pos_offset) + dynamic);
            let factors = if freq_factors { Some(ins[2]) } else { None };
            (theta, head_dim, offset, factors, false)
        }
        Op::RoPEGrad {
            theta,
            pos_offset,
            head_dim,
        } => {
            let offset = Positions::Offset(f64::from(pos_offset));
            (theta, head_dim, offset, None, true)
        }
        Op::RoPEPositions { theta, head_dim } => {
            (theta, head_dim, Positions::PerRow(ins[1]), None, false)
        }
        _ => unreachable!("not a RoPE op"),
    };
    let (rows, dim) = matrix(node, ins[0], "input")?;
    let head_dim = head_dim as usize;
    if head_dim == 0 || !head_dim.is_multiple_of(2) || dim % head_dim != 0 {
        return Err(invalid(
            node,
            format!("head_dim {head_dim} must be even and divide {dim}"),
        ));
    }
    if let Some(f) = factors {
        expect_len(node, f, head_dim / 2, "frequency factors")?;
    }
    if let Positions::PerRow(p) = positions {
        expect_len(node, p, rows, "positions")?;
    }
    Ok(Rotation {
        theta: f64::from(theta),
        head_dim,
        positions,
        factors,
        inverse,
    })
}

/// Visit every rotated pair: `f(row, pair_in_head, i0, i1)` with `i0`, `i1`
/// the flat indices of the pair's two elements.
fn for_each_pair(x: &Tensor, head_dim: usize, mut f: impl FnMut(usize, usize, usize, usize)) {
    let (rows, dim) = (x.shape[0], x.shape[1]);
    let half = head_dim / 2;
    for r in 0..rows {
        for head in 0..dim / head_dim {
            let base = r * dim + head * head_dim;
            for i in 0..half {
                f(r, i, base + i, base + i + half);
            }
        }
    }
}

fn rope(x: &Tensor, rot: &Rotation<'_>) -> Vec<f64> {
    let mut out = vec![0.0; x.len()];
    for_each_pair(x, rot.head_dim, |r, i, i0, i1| {
        let (sin, cos) = rot.angle(r, i).sin_cos();
        let sin = if rot.inverse { -sin } else { sin };
        let (a, b) = (x.data[i0], x.data[i1]);
        out[i0] = a * cos - b * sin;
        out[i1] = a * sin + b * cos;
    });
    out
}

/// `(M, D, P)` of a pairwise op: `left`/`directions` is `[M, D]` and the
/// paired tensor is `[M·P, D]`.
fn pairwise_dims(
    node: &Node,
    single: &Tensor,
    paired: &Tensor,
    pairs: u32,
) -> Result<(usize, usize, usize), Error> {
    let (m, d) = matrix(node, single, "unpaired operand")?;
    let p = pairs as usize;
    if paired.shape != [m * p, d] {
        return Err(invalid(
            node,
            format!("paired operand {:?} is not [{}, {d}]", paired.shape, m * p),
        ));
    }
    Ok((m, d, p))
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub(super) fn eval(node: &Node, ins: &[&Tensor]) -> Result<Vec<f64>, Error> {
    let arity = match node.op {
        Op::RmsNorm { .. } | Op::RoPEPositions { .. } => 2,
        Op::RoPE { freq_factors, .. } => {
            if freq_factors {
                3
            } else {
                ins.len().clamp(1, 2)
            }
        }
        Op::RoPEGrad { .. } => 1,
        Op::PairwiseSquaredDistance { .. } | Op::PairwiseVectorRejection { .. } => 2,
        _ => 3,
    };
    if ins.len() != arity {
        return Err(invalid(
            node,
            format!("expected {arity} inputs, got {}", ins.len()),
        ));
    }
    let out = match node.op {
        // y = x / sqrt(mean(x²) + eps) · w, per row.
        Op::RmsNorm { eps } => {
            let (x, w) = (ins[0], ins[1]);
            let (rows, cols) = matrix(node, x, "x")?;
            expect_len(node, w, cols, "weight")?;
            let mut out = vec![0.0; x.len()];
            for r in 0..rows {
                let row = &x.data[r * cols..(r + 1) * cols];
                let s = inv_rms(row, f64::from(eps));
                for c in 0..cols {
                    out[r * cols + c] = row[c] * s * w.data[c];
                }
            }
            out
        }
        // ∂L/∂w_j = Σ_i dy_ij · x_ij · s_i.
        Op::RmsNormGradW { eps } => {
            let (rows, cols) = norm_grad_inputs(node, ins)?;
            let (dy, x) = (ins[0], ins[1]);
            let mut out = vec![0.0; cols];
            for r in 0..rows {
                let row = &x.data[r * cols..(r + 1) * cols];
                let s = inv_rms(row, f64::from(eps));
                for c in 0..cols {
                    out[c] += dy.data[r * cols + c] * row[c] * s;
                }
            }
            out
        }
        // ∂L/∂x_ij = s_i · (dy_ij·w_j − x_ij · s_i² · mean_k(dy_ik·w_k·x_ik)).
        Op::RmsNormGradX { eps } => {
            let (rows, cols) = norm_grad_inputs(node, ins)?;
            let (dy, x, w) = (ins[0], ins[1], ins[2]);
            let mut out = vec![0.0; x.len()];
            for r in 0..rows {
                let at = r * cols;
                let row = &x.data[at..at + cols];
                let s = inv_rms(row, f64::from(eps));
                let proj = (0..cols)
                    .map(|c| dy.data[at + c] * w.data[c] * row[c])
                    .sum::<f64>()
                    / cols as f64;
                for c in 0..cols {
                    out[at + c] = s * (dy.data[at + c] * w.data[c] - row[c] * s * s * proj);
                }
            }
            out
        }
        // y = (x − mean) / sqrt(var + eps) · w + b, per row.
        Op::LayerNorm { eps } => {
            let (x, w, b) = (ins[0], ins[1], ins[2]);
            let (rows, cols) = matrix(node, x, "x")?;
            expect_len(node, w, cols, "weight")?;
            expect_len(node, b, cols, "bias")?;
            let mut out = vec![0.0; x.len()];
            for r in 0..rows {
                let row = &x.data[r * cols..(r + 1) * cols];
                let (mean, rstd) = moments(row, f64::from(eps));
                for c in 0..cols {
                    out[r * cols + c] = (row[c] - mean) * rstd * w.data[c] + b.data[c];
                }
            }
            out
        }
        // The weight gradient only, `[cols]`: Σ_i dy_ij · x̂_ij. (The bias
        // gradient is a separate SumRows; the node has the weight's type.)
        Op::LayerNormGradWB { eps } => {
            let (rows, cols) = norm_grad_inputs(node, ins)?;
            let (dy, x) = (ins[0], ins[1]);
            let mut out = vec![0.0; cols];
            for r in 0..rows {
                let row = &x.data[r * cols..(r + 1) * cols];
                let (mean, rstd) = moments(row, f64::from(eps));
                for c in 0..cols {
                    out[c] += dy.data[r * cols + c] * (row[c] - mean) * rstd;
                }
            }
            out
        }
        // ∂L/∂x = rstd · (g − mean(g) − x̂ · mean(g ⊙ x̂)) with g = dy ⊙ w.
        Op::LayerNormGradX { eps } => {
            let (rows, cols) = norm_grad_inputs(node, ins)?;
            let (dy, x, w) = (ins[0], ins[1], ins[2]);
            let n = cols as f64;
            let mut out = vec![0.0; x.len()];
            for r in 0..rows {
                let at = r * cols;
                let row = &x.data[at..at + cols];
                let (mean, rstd) = moments(row, f64::from(eps));
                let g: Vec<f64> = (0..cols).map(|c| dy.data[at + c] * w.data[c]).collect();
                let xhat: Vec<f64> = row.iter().map(|v| (v - mean) * rstd).collect();
                let mean_g = g.iter().sum::<f64>() / n;
                let mean_gx = dot(&g, &xhat) / n;
                for c in 0..cols {
                    out[at + c] = rstd * (g[c] - mean_g - xhat[c] * mean_gx);
                }
            }
            out
        }
        Op::RoPE { .. } | Op::RoPEGrad { .. } | Op::RoPEPositions { .. } => {
            rope(ins[0], &rotation(node, ins)?)
        }
        // out[m, p] = ‖left[m] − right[m·P + p]‖².
        Op::PairwiseSquaredDistance { pairs } => {
            let (left, right) = (ins[0], ins[1]);
            let (m, d, p) = pairwise_dims(node, left, right, pairs)?;
            let mut out = vec![0.0; m * p];
            for i in 0..m {
                let l = &left.data[i * d..(i + 1) * d];
                for k in 0..p {
                    let r = &right.data[(i * p + k) * d..(i * p + k + 1) * d];
                    out[i * p + k] = l.iter().zip(r).map(|(a, b)| (a - b) * (a - b)).sum();
                }
            }
            out
        }
        // out[m·P + p] = v − (v · u) u with v = vectors[m·P + p] and
        // u = directions[m]. `u` is not normalized here: the op is this
        // bilinear map for any `u`, a projection when `‖u‖ = 1`.
        Op::PairwiseVectorRejection { pairs } => {
            let (vectors, directions) = (ins[0], ins[1]);
            let (m, d, p) = pairwise_dims(node, directions, vectors, pairs)?;
            let mut out = vec![0.0; vectors.len()];
            for i in 0..m {
                let u = &directions.data[i * d..(i + 1) * d];
                for k in 0..p {
                    let at = (i * p + k) * d;
                    let v = &vectors.data[at..at + d];
                    let c = dot(v, u);
                    for j in 0..d {
                        out[at + j] = v[j] - c * u[j];
                    }
                }
            }
            out
        }
        Op::PairwiseGrad { kind, inner, pairs } => {
            let (g, first, second) = (ins[0], ins[1], ins[2]);
            match kind {
                // ∂/∂left[m] = Σ_p 2 g[m, p] (left[m] − right[m·P + p]).
                PairwiseGradKind::DistanceLeft | PairwiseGradKind::DistanceRight => {
                    let (m, d, p) = pairwise_dims(node, first, second, pairs)?;
                    if d != inner as usize {
                        return Err(invalid(node, "inner does not match the operand width"));
                    }
                    expect_len(node, g, m * p, "upstream gradient")?;
                    let left_side = matches!(kind, PairwiseGradKind::DistanceLeft);
                    let mut out = vec![0.0; if left_side { m * d } else { m * p * d }];
                    for i in 0..m {
                        for k in 0..p {
                            let gk = g.data[i * p + k];
                            for j in 0..d {
                                let diff = first.data[i * d + j] - second.data[(i * p + k) * d + j];
                                if left_side {
                                    out[i * d + j] += 2.0 * gk * diff;
                                } else {
                                    out[(i * p + k) * d + j] = -2.0 * gk * diff;
                                }
                            }
                        }
                    }
                    out
                }
                // ∂/∂u[m] = −Σ_p ((v·u) g + (g·u) v) with g, v the rows
                // m·P + p of the upstream gradient and the vectors.
                PairwiseGradKind::RejectionDirections => {
                    let (vectors, directions) = (first, second);
                    let (m, d, p) = pairwise_dims(node, directions, vectors, pairs)?;
                    if d != inner as usize {
                        return Err(invalid(node, "inner does not match the operand width"));
                    }
                    expect_len(node, g, vectors.len(), "upstream gradient")?;
                    let mut out = vec![0.0; m * d];
                    for i in 0..m {
                        let u = &directions.data[i * d..(i + 1) * d];
                        for k in 0..p {
                            let at = (i * p + k) * d;
                            let v = &vectors.data[at..at + d];
                            let gv = &g.data[at..at + d];
                            let (vu, gu) = (dot(v, u), dot(gv, u));
                            for j in 0..d {
                                out[i * d + j] -= vu * gv[j] + gu * v[j];
                            }
                        }
                    }
                    out
                }
            }
        }
        _ => {
            return Err(Error::Unsupported {
                node: node.id,
                reason: format!("{:?} is not a norm, RoPE or pairwise op", node.op),
            });
        }
    };
    Ok(out)
}

/// Error scales for the ops whose result cancels: each is the same
/// expression with every term replaced by its absolute value. Mean
/// subtraction contributes `|mean|`: rounding `x − mean` in `f32` errs by
/// about `ulp(max(|x|, |mean|))`, however small the difference.
pub(super) fn magnitude(node: &Node, ins: &[&Tensor], out: &Tensor) -> Option<Vec<f64>> {
    let rows_cols = |x: &Tensor| (x.len() / x.shape[1].max(1), x.shape[1]);
    match node.op {
        Op::RmsNormGradW { eps } => {
            let (dy, x) = (ins[0], ins[1]);
            let (rows, cols) = rows_cols(x);
            let mut m = vec![0.0; cols];
            for r in 0..rows {
                let row = &x.data[r * cols..(r + 1) * cols];
                let s = inv_rms(row, f64::from(eps));
                for c in 0..cols {
                    m[c] += (dy.data[r * cols + c] * row[c] * s).abs();
                }
            }
            Some(m)
        }
        Op::RmsNormGradX { eps } => {
            let (dy, x, w) = (ins[0], ins[1], ins[2]);
            let (rows, cols) = rows_cols(x);
            let mut m = vec![0.0; x.len()];
            for r in 0..rows {
                let at = r * cols;
                let row = &x.data[at..at + cols];
                let s = inv_rms(row, f64::from(eps));
                let proj = (0..cols)
                    .map(|c| (dy.data[at + c] * w.data[c] * row[c]).abs())
                    .sum::<f64>()
                    / cols as f64;
                for c in 0..cols {
                    m[at + c] =
                        s * ((dy.data[at + c] * w.data[c]).abs() + (row[c] * s * s * proj).abs());
                }
            }
            Some(m)
        }
        Op::LayerNorm { eps } => {
            let (x, w, b) = (ins[0], ins[1], ins[2]);
            let (rows, cols) = rows_cols(x);
            let mut m = vec![0.0; x.len()];
            for r in 0..rows {
                let row = &x.data[r * cols..(r + 1) * cols];
                let (mean, rstd) = moments(row, f64::from(eps));
                for c in 0..cols {
                    m[r * cols + c] = ((row[c] - mean).abs() + mean.abs()) * rstd * w.data[c].abs()
                        + b.data[c].abs();
                }
            }
            Some(m)
        }
        Op::LayerNormGradWB { eps } => {
            let (dy, x) = (ins[0], ins[1]);
            let (rows, cols) = rows_cols(x);
            let mut m = vec![0.0; cols];
            for r in 0..rows {
                let row = &x.data[r * cols..(r + 1) * cols];
                let (mean, rstd) = moments(row, f64::from(eps));
                for c in 0..cols {
                    m[c] +=
                        (dy.data[r * cols + c] * ((row[c] - mean).abs() + mean.abs()) * rstd).abs();
                }
            }
            Some(m)
        }
        Op::LayerNormGradX { eps } => {
            let (dy, x, w) = (ins[0], ins[1], ins[2]);
            let (rows, cols) = rows_cols(x);
            let n = cols as f64;
            let mut m = vec![0.0; x.len()];
            for r in 0..rows {
                let at = r * cols;
                let row = &x.data[at..at + cols];
                let (mean, rstd) = moments(row, f64::from(eps));
                let g: Vec<f64> = (0..cols)
                    .map(|c| (dy.data[at + c] * w.data[c]).abs())
                    .collect();
                let xhat: Vec<f64> = row
                    .iter()
                    .map(|v| ((v - mean).abs() + mean.abs()) * rstd)
                    .collect();
                let mean_g = g.iter().sum::<f64>() / n;
                let mean_gx = dot(&g, &xhat) / n;
                for c in 0..cols {
                    m[at + c] = rstd * (g[c] + mean_g + xhat[c] * mean_gx);
                }
            }
            Some(m)
        }
        // Each rotated component can cancel, so its error scales with the
        // length of its pair. The angle `pos · freq` is itself rounded to
        // `f32` before any sine is taken, which moves both components by
        // `|pair| · ulp(angle)`: 2.4e-4 · |pair| at `|angle| = 4096`, above
        // the default `rtol` however accurate `sin` is. Past `|angle| = 256`
        // the scale grows as `|angle| / 256`, which at the default `rtol`
        // allows about 13 ulps of angle error (product, `pow` and argument
        // reduction).
        Op::RoPE { .. } | Op::RoPEGrad { .. } | Op::RoPEPositions { .. } => {
            let x = ins[0];
            let rot = rotation(node, ins).ok()?;
            let mut m = vec![0.0; x.len()];
            for_each_pair(x, rot.head_dim, |r, i, i0, i1| {
                let scale = x.data[i0].hypot(x.data[i1]) * (rot.angle(r, i).abs() / 256.0).max(1.0);
                m[i0] = scale;
                m[i1] = scale;
            });
            Some(m)
        }
        Op::PairwiseVectorRejection { pairs } => {
            let (vectors, directions) = (ins[0], ins[1]);
            let (m_rows, d) = rows_cols(directions);
            let p = pairs as usize;
            let mut m = vec![0.0; vectors.len()];
            for i in 0..m_rows {
                let u = &directions.data[i * d..(i + 1) * d];
                for k in 0..p {
                    let at = (i * p + k) * d;
                    let v = &vectors.data[at..at + d];
                    let c: f64 = v.iter().zip(u).map(|(a, b)| (a * b).abs()).sum();
                    for j in 0..d {
                        m[at + j] = v[j].abs() + c * u[j].abs();
                    }
                }
            }
            Some(m)
        }
        Op::PairwiseGrad { kind, pairs, .. } => {
            let (g, first, second) = (ins[0], ins[1], ins[2]);
            let p = pairs as usize;
            match kind {
                PairwiseGradKind::DistanceLeft => {
                    let (m_rows, d) = rows_cols(first);
                    let mut m = vec![0.0; out.len()];
                    for i in 0..m_rows {
                        for k in 0..p {
                            for j in 0..d {
                                let diff = first.data[i * d + j] - second.data[(i * p + k) * d + j];
                                m[i * d + j] += (2.0 * g.data[i * p + k] * diff).abs();
                            }
                        }
                    }
                    Some(m)
                }
                PairwiseGradKind::DistanceRight => None,
                PairwiseGradKind::RejectionDirections => {
                    let (vectors, directions) = (first, second);
                    let (m_rows, d) = rows_cols(directions);
                    let mut m = vec![0.0; out.len()];
                    for i in 0..m_rows {
                        let u = &directions.data[i * d..(i + 1) * d];
                        for k in 0..p {
                            let at = (i * p + k) * d;
                            let v = &vectors.data[at..at + d];
                            let gv = &g.data[at..at + d];
                            let vu: f64 = v.iter().zip(u).map(|(a, b)| (a * b).abs()).sum();
                            let gu: f64 = gv.iter().zip(u).map(|(a, b)| (a * b).abs()).sum();
                            for j in 0..d {
                                m[i * d + j] += vu * gv[j].abs() + gu * v[j].abs();
                            }
                        }
                    }
                    Some(m)
                }
            }
        }
        _ => None,
    }
}
