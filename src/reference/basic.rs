//! Elementwise, contraction, reduction, loss, gather and layout ops.

use super::{Error, Tensor};
use crate::graph::{Node, Op};

fn invalid(node: &Node, reason: impl Into<String>) -> Error {
    Error::Invalid {
        node: node.id,
        reason: reason.into(),
    }
}

/// `[rows, cols]` of a tensor viewed as a matrix over its last axis.
fn rows_cols(t: &Tensor) -> (usize, usize) {
    let cols = t.shape.last().copied().unwrap_or(1).max(1);
    (t.len() / cols, cols)
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn silu(x: f64) -> f64 {
    x * sigmoid(x)
}

fn silu_derivative(x: f64) -> f64 {
    let s = sigmoid(x);
    s + x * s * (1.0 - s)
}

const GELU_C: f64 = 0.797_884_560_802_865_4; // sqrt(2/pi)

/// The tanh form of GELU. This is the definition every Meganeura kernel and
/// derivative implements, not an approximation of an erf reference.
pub(super) fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + (GELU_C * (x + 0.044715 * x * x * x)).tanh())
}

pub(super) fn gelu_derivative(x: f64) -> f64 {
    let t = (GELU_C * (x + 0.044715 * x * x * x)).tanh();
    0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * GELU_C * (1.0 + 3.0 * 0.044715 * x * x)
}

/// `C = op(A) · op(B)` with explicit strides: `a(m, k)` and `b(k, n)`.
fn contract(
    m: usize,
    n: usize,
    k: usize,
    a: impl Fn(usize, usize) -> f64,
    b: impl Fn(usize, usize) -> f64,
) -> Vec<f64> {
    let mut out = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            out[i * n + j] = (0..k).map(|p| a(i, p) * b(p, j)).sum();
        }
    }
    out
}

fn matmul_kind(node: &Node, op: &Op, a: &Tensor, b: &Tensor) -> Result<Vec<f64>, Error> {
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err(invalid(node, "matmul operands must be 2D"));
    }
    let (a0, a1) = (a.shape[0], a.shape[1]);
    let (b0, b1) = (b.shape[0], b.shape[1]);
    let out = match *op {
        Op::MatMul | Op::FusedMatMulAdd => {
            if a1 != b0 {
                return Err(invalid(node, "inner dimensions differ"));
            }
            contract(
                a0,
                b1,
                a1,
                |i, p| a.data[i * a1 + p],
                |p, j| b.data[p * b1 + j],
            )
        }
        Op::MatMulAT | Op::FusedMatMulATAdd => {
            if a0 != b0 {
                return Err(invalid(node, "inner dimensions differ"));
            }
            contract(
                a1,
                b1,
                a0,
                |i, p| a.data[p * a1 + i],
                |p, j| b.data[p * b1 + j],
            )
        }
        Op::MatMulBT | Op::FusedMatMulBTAdd => {
            if a1 != b1 {
                return Err(invalid(node, "inner dimensions differ"));
            }
            contract(
                a0,
                b0,
                a1,
                |i, p| a.data[i * a1 + p],
                |p, j| b.data[j * b1 + p],
            )
        }
        _ => unreachable!(),
    };
    Ok(out)
}

fn block_matmul(node: &Node, a: &Tensor, b: &Tensor, transpose_b: bool) -> Result<Vec<f64>, Error> {
    if a.shape.len() != 2 || b.shape.len() != 3 {
        return Err(invalid(node, "block matmul needs 2D A and 3D B"));
    }
    let m = a.shape[0];
    let groups = b.shape[0];
    let (k, n) = if transpose_b {
        (b.shape[2], b.shape[1])
    } else {
        (b.shape[1], b.shape[2])
    };
    if a.shape[1] != groups * k {
        return Err(invalid(node, "block widths differ"));
    }
    let mut out = vec![0.0; m * groups * n];
    for i in 0..m {
        for g in 0..groups {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    let bv = if transpose_b {
                        b.data[(g * n + j) * k + p]
                    } else {
                        b.data[(g * k + p) * n + j]
                    };
                    sum += a.data[i * groups * k + g * k + p] * bv;
                }
                out[i * groups * n + g * n + j] = sum;
            }
        }
    }
    Ok(out)
}

fn block_matmul_at(node: &Node, a: &Tensor, b: &Tensor, groups: usize) -> Result<Vec<f64>, Error> {
    if a.shape.len() != 2 || b.shape.len() != 2 || a.shape[0] != b.shape[0] {
        return Err(invalid(node, "block matmul AT needs [K, G*M] and [K, G*N]"));
    }
    let k = a.shape[0];
    let m = a.shape[1] / groups;
    let n = b.shape[1] / groups;
    let mut out = vec![0.0; groups * m * n];
    for g in 0..groups {
        for i in 0..m {
            for j in 0..n {
                out[(g * m + i) * n + j] = (0..k)
                    .map(|p| {
                        a.data[p * groups * m + g * m + i] * b.data[p * groups * n + g * n + j]
                    })
                    .sum();
            }
        }
    }
    Ok(out)
}

fn zip(
    node: &Node,
    a: &Tensor,
    b: &Tensor,
    f: impl Fn(f64, f64) -> f64,
) -> Result<Vec<f64>, Error> {
    if a.len() != b.len() {
        return Err(invalid(
            node,
            format!("operand lengths differ: {} vs {}", a.len(), b.len()),
        ));
    }
    Ok(a.data.iter().zip(&b.data).map(|(&x, &y)| f(x, y)).collect())
}

fn zip3(
    node: &Node,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    f: impl Fn(f64, f64, f64) -> f64,
) -> Result<Vec<f64>, Error> {
    if a.len() != b.len() || a.len() != c.len() {
        return Err(invalid(node, "operand lengths differ"));
    }
    Ok((0..a.len())
        .map(|i| f(a.data[i], b.data[i], c.data[i]))
        .collect())
}

fn map(a: &Tensor, f: impl Fn(f64) -> f64) -> Vec<f64> {
    a.data.iter().map(|&x| f(x)).collect()
}

/// Broadcast `b` across the rows of `a` (`b` has one row's worth of values).
fn row_broadcast(
    node: &Node,
    a: &Tensor,
    b: &Tensor,
    f: impl Fn(f64, f64) -> f64,
) -> Result<Vec<f64>, Error> {
    let n = b.len();
    if n == 0 || a.len() % n != 0 {
        return Err(invalid(node, "broadcast operand does not tile the input"));
    }
    Ok(a.data
        .iter()
        .enumerate()
        .map(|(i, &x)| f(x, b.data[i % n]))
        .collect())
}

fn softmax_rows(x: &Tensor) -> Vec<f64> {
    let (rows, cols) = rows_cols(x);
    let mut out = vec![0.0; x.len()];
    for r in 0..rows {
        let row = &x.data[r * cols..(r + 1) * cols];
        let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = row.iter().map(|&v| (v - max).exp()).sum();
        for c in 0..cols {
            out[r * cols + c] = (row[c] - max).exp() / sum;
        }
    }
    out
}

fn log_sum_exp(row: &[f64]) -> f64 {
    let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    max + row.iter().map(|&v| (v - max).exp()).sum::<f64>().ln()
}

/// Split `[M, 2N]` into its gate (first half) and up (second half) columns.
fn glu_halves(node: &Node, input: &Tensor) -> Result<(usize, usize), Error> {
    let (rows, cols) = rows_cols(input);
    if cols % 2 != 0 {
        return Err(invalid(node, "concatenated GLU input needs an even width"));
    }
    Ok((rows, cols / 2))
}

fn glu_concat(node: &Node, input: &Tensor, act: impl Fn(f64) -> f64) -> Result<Vec<f64>, Error> {
    let (rows, n) = glu_halves(node, input)?;
    let mut out = vec![0.0; rows * n];
    for r in 0..rows {
        for c in 0..n {
            let gate = input.data[r * 2 * n + c];
            let up = input.data[r * 2 * n + n + c];
            out[r * n + c] = act(gate) * up;
        }
    }
    Ok(out)
}

fn glu_concat_grad(
    node: &Node,
    grad: &Tensor,
    input: &Tensor,
    act: impl Fn(f64) -> f64,
    act_derivative: impl Fn(f64) -> f64,
) -> Result<Vec<f64>, Error> {
    let (rows, n) = glu_halves(node, input)?;
    if grad.len() != rows * n {
        return Err(invalid(node, "GLU gradient does not match half the input"));
    }
    let mut out = vec![0.0; rows * 2 * n];
    for r in 0..rows {
        for c in 0..n {
            let g = grad.data[r * n + c];
            let gate = input.data[r * 2 * n + c];
            let up = input.data[r * 2 * n + n + c];
            out[r * 2 * n + c] = g * up * act_derivative(gate);
            out[r * 2 * n + n + c] = g * act(gate);
        }
    }
    Ok(out)
}

/// `(batch, channels, spatial)` of a flat NCHW tensor.
fn nchw(node: &Node, t: &Tensor, channels: u32, spatial: usize) -> Result<usize, Error> {
    let plane = channels as usize * spatial;
    if plane == 0 || t.len() % plane != 0 {
        return Err(invalid(node, "input is not a whole number of NCHW images"));
    }
    Ok(t.len() / plane)
}

pub(super) fn eval(node: &Node, ins: &[&Tensor]) -> Result<Vec<f64>, Error> {
    let arg = |i: usize| -> Result<&Tensor, Error> {
        ins.get(i)
            .copied()
            .ok_or_else(|| invalid(node, format!("missing input {i}")))
    };
    let out = match node.op {
        Op::MatMul | Op::MatMulAT | Op::MatMulBT => matmul_kind(node, &node.op, arg(0)?, arg(1)?)?,
        Op::FusedMatMulAdd | Op::FusedMatMulATAdd | Op::FusedMatMulBTAdd => {
            let product = matmul_kind(node, &node.op, arg(0)?, arg(1)?)?;
            let d = arg(2)?;
            let n = node.ty.shape.last().copied().unwrap_or(1);
            if d.len() == product.len() {
                product.iter().zip(&d.data).map(|(p, d)| p + d).collect()
            } else if d.len() == n {
                product
                    .iter()
                    .enumerate()
                    .map(|(i, p)| p + d.data[i % n])
                    .collect()
            } else {
                return Err(invalid(
                    node,
                    "addend matches neither the product nor a row",
                ));
            }
        }
        Op::BlockMatMul => block_matmul(node, arg(0)?, arg(1)?, false)?,
        Op::BlockMatMulBT => block_matmul(node, arg(0)?, arg(1)?, true)?,
        Op::BlockMatMulAT { groups } => block_matmul_at(node, arg(0)?, arg(1)?, groups)?,

        Op::Add => zip(node, arg(0)?, arg(1)?, |a, b| a + b)?,
        Op::Mul => zip(node, arg(0)?, arg(1)?, |a, b| a * b)?,
        Op::Greater => zip(node, arg(0)?, arg(1)?, |a, b| if a > b { 1.0 } else { 0.0 })?,

        Op::Relu => map(arg(0)?, |x| x.max(0.0)),
        Op::Sigmoid => map(arg(0)?, sigmoid),
        Op::Tanh => map(arg(0)?, f64::tanh),
        Op::Neg => map(arg(0)?, |x| -x),
        Op::Abs => map(arg(0)?, f64::abs),
        Op::Log => map(arg(0)?, f64::ln),
        Op::Recip => map(arg(0)?, |x| 1.0 / x),
        Op::Exp => map(arg(0)?, f64::exp),
        Op::Softplus { beta } => {
            let beta = f64::from(beta);
            map(arg(0)?, |x| {
                x.max(0.0) + (-(beta * x).abs()).exp().ln_1p() / beta
            })
        }
        Op::SoftplusGrad { beta } => {
            let beta = f64::from(beta);
            zip(node, arg(0)?, arg(1)?, |g, x| g * sigmoid(beta * x))?
        }
        Op::Clamp { min, max } => map(arg(0)?, |x| x.max(f64::from(min)).min(f64::from(max))),
        Op::Scale { factor } => map(arg(0)?, |x| x * f64::from(factor)),

        Op::SumAll => vec![arg(0)?.data.iter().sum()],
        Op::MeanAll => {
            let x = arg(0)?;
            vec![x.data.iter().sum::<f64>() / x.len() as f64]
        }
        Op::SumRows => {
            let x = arg(0)?;
            let n = node.ty.num_elements();
            if n == 0 || x.len() % n != 0 {
                return Err(invalid(node, "output width does not divide the input"));
            }
            let mut out = vec![0.0; n];
            for (i, &v) in x.data.iter().enumerate() {
                out[i % n] += v;
            }
            out
        }
        Op::SumInner => {
            let x = arg(0)?;
            let (rows, cols) = rows_cols(x);
            (0..rows)
                .map(|r| x.data[r * cols..(r + 1) * cols].iter().sum())
                .collect()
        }
        Op::BroadcastInner { inner } => {
            let x = arg(0)?;
            let inner = inner as usize;
            (0..x.len() * inner).map(|i| x.data[i / inner]).collect()
        }
        Op::NormalizeInnerSum { inner, floor } => {
            let x = arg(0)?;
            let (inner, floor) = (inner as usize, f64::from(floor));
            let mut out = vec![0.0; x.len()];
            for r in 0..x.len() / inner {
                let row = &x.data[r * inner..(r + 1) * inner];
                let sum: f64 = row.iter().sum();
                let denominator = (sum - floor).max(0.0) + floor;
                for c in 0..inner {
                    out[r * inner + c] = row[c] / denominator;
                }
            }
            out
        }
        Op::NormalizeInnerSumGrad { inner, floor } => {
            // y_j = x_j / d(s), d = relu(s - floor) + floor, s = Σ x.
            // dx_k = g_k / d - [s > floor] · Σ_j g_j x_j / d².
            let (g, x) = (arg(0)?, arg(1)?);
            let (inner, floor) = (inner as usize, f64::from(floor));
            let mut out = vec![0.0; x.len()];
            for r in 0..x.len() / inner {
                let range = r * inner..(r + 1) * inner;
                let sum: f64 = x.data[range.clone()].iter().sum();
                let d = (sum - floor).max(0.0) + floor;
                let active = if sum > floor { 1.0 } else { 0.0 };
                let dot: f64 = range.clone().map(|i| g.data[i] * x.data[i]).sum();
                for i in range {
                    out[i] = g.data[i] / d - active * dot / (d * d);
                }
            }
            out
        }
        Op::ExclusiveCumsum { reverse } => {
            let x = arg(0)?;
            let (rows, cols) = rows_cols(x);
            let mut out = vec![0.0; x.len()];
            for r in 0..rows {
                let mut acc = 0.0;
                let order: Box<dyn Iterator<Item = usize>> = if reverse {
                    Box::new((0..cols).rev())
                } else {
                    Box::new(0..cols)
                };
                for c in order {
                    out[r * cols + c] = acc;
                    acc += x.data[r * cols + c];
                }
            }
            out
        }
        Op::ShiftInner { offset } => {
            let x = arg(0)?;
            let (rows, cols) = rows_cols(x);
            let mut out = vec![0.0; x.len()];
            for r in 0..rows {
                for c in 0..cols {
                    let source = c as i64 - i64::from(offset);
                    if (0..cols as i64).contains(&source) {
                        out[r * cols + c] = x.data[r * cols + source as usize];
                    }
                }
            }
            out
        }
        Op::Softmax => softmax_rows(arg(0)?),
        Op::LogSoftmax => {
            let x = arg(0)?;
            let (rows, cols) = rows_cols(x);
            let mut out = vec![0.0; x.len()];
            for r in 0..rows {
                let row = &x.data[r * cols..(r + 1) * cols];
                let lse = log_sum_exp(row);
                for c in 0..cols {
                    out[r * cols + c] = row[c] - lse;
                }
            }
            out
        }
        Op::CrossEntropyLoss => {
            // L = -(1/B) Σ_b Σ_j labels · log_softmax(logits)
            let (logits, labels) = (arg(0)?, arg(1)?);
            let (rows, cols) = rows_cols(logits);
            let mut loss = 0.0;
            for r in 0..rows {
                let row = &logits.data[r * cols..(r + 1) * cols];
                let lse = log_sum_exp(row);
                for c in 0..cols {
                    loss -= labels.data[r * cols + c] * (row[c] - lse);
                }
            }
            vec![loss / rows as f64]
        }
        Op::CrossEntropyLogitsGrad => {
            // (softmax · Σlabels − labels) / B
            let (logits, labels) = (arg(0)?, arg(1)?);
            let (rows, cols) = rows_cols(logits);
            let softmax = softmax_rows(logits);
            let mut out = vec![0.0; logits.len()];
            for r in 0..rows {
                let s: f64 = labels.data[r * cols..(r + 1) * cols].iter().sum();
                for c in 0..cols {
                    let i = r * cols + c;
                    out[i] = (softmax[i] * s - labels.data[i]) / rows as f64;
                }
            }
            out
        }
        Op::BceLoss => {
            // -mean(t·log p + (1−t)·log(1−p)), p clamped to [1e-7, 1 − 1e-7].
            let (pred, labels) = (arg(0)?, arg(1)?);
            let eps = f64::from(1e-7f32);
            let hi = f64::from(1.0f32 - 1e-7f32);
            let total: f64 = pred
                .data
                .iter()
                .zip(&labels.data)
                .map(|(&p, &t)| {
                    let p = p.clamp(eps, hi);
                    -(t * p.ln() + (1.0 - t) * (1.0 - p).ln())
                })
                .sum();
            vec![total / pred.len() as f64]
        }
        Op::Transpose => {
            let x = arg(0)?;
            if x.shape.len() != 2 {
                return Err(invalid(node, "transpose needs a 2D input"));
            }
            let (r, c) = (x.shape[0], x.shape[1]);
            let mut out = vec![0.0; x.len()];
            for i in 0..r {
                for j in 0..c {
                    out[j * r + i] = x.data[i * c + j];
                }
            }
            out
        }
        Op::BiasAdd => row_broadcast(node, arg(0)?, arg(1)?, |a, b| a + b)?,
        Op::BiasMul => row_broadcast(node, arg(0)?, arg(1)?, |a, b| a * b)?,
        // A dead node has no defined value; nothing may consume it.
        Op::Nop => vec![f64::NAN; node.ty.num_elements()],
        Op::Identity | Op::Materialize | Op::StopGradient => arg(0)?.data.clone(),
        Op::ScatterAdd { vocab_size } => {
            let (indices, src) = (arg(0)?, arg(1)?);
            let (rows, dim) = rows_cols(src);
            if indices.len() != rows {
                return Err(invalid(node, "one index per source row"));
            }
            let mut out = vec![0.0; vocab_size * dim];
            for r in 0..rows {
                let v = indices.index(r);
                if v >= vocab_size {
                    return Err(invalid(
                        node,
                        format!("index {v} >= vocabulary {vocab_size}"),
                    ));
                }
                for c in 0..dim {
                    out[v * dim + c] += src.data[r * dim + c];
                }
            }
            out
        }
        Op::Silu => map(arg(0)?, silu),
        Op::SiluGrad => zip(node, arg(0)?, arg(1)?, |g, x| g * silu_derivative(x))?,
        Op::Gelu => map(arg(0)?, gelu),
        Op::SwiGLU => zip(node, arg(0)?, arg(1)?, |gate, up| silu(gate) * up)?,
        Op::GeGLU => zip(node, arg(0)?, arg(1)?, |gate, up| gelu(gate) * up)?,
        Op::SwiGLUGradGate => zip3(node, arg(0)?, arg(1)?, arg(2)?, |g, gate, up| {
            g * up * silu_derivative(gate)
        })?,
        Op::SwiGLUGradUp => zip(node, arg(0)?, arg(1)?, |g, gate| g * silu(gate))?,
        Op::SwiGLUConcat => glu_concat(node, arg(0)?, silu)?,
        Op::GeGLUConcat => glu_concat(node, arg(0)?, gelu)?,
        Op::SwiGLUConcatGrad => glu_concat_grad(node, arg(0)?, arg(1)?, silu, silu_derivative)?,
        Op::GeGLUConcatGrad => glu_concat_grad(node, arg(0)?, arg(1)?, gelu, gelu_derivative)?,
        Op::Embedding => {
            let (indices, table) = (arg(0)?, arg(1)?);
            let (vocab, dim) = rows_cols(table);
            let mut out = Vec::with_capacity(indices.len() * dim);
            for i in 0..indices.len() {
                let v = indices.index(i);
                if v >= vocab {
                    return Err(invalid(node, format!("index {v} >= vocabulary {vocab}")));
                }
                out.extend_from_slice(&table.data[v * dim..(v + 1) * dim]);
            }
            out
        }
        Op::ToF16 => map(arg(0)?, super::round_f16),
        Op::CacheWrite | Op::CacheWritePrefix => {
            let (new_kv, cache, pos) = (arg(0)?, arg(1)?, arg(2)?);
            let (rows, dim) = rows_cols(cache);
            let pos = pos.index(0);
            let count = if matches!(node.op, Op::CacheWrite) {
                1
            } else {
                arg(3)?.index(0)
            };
            if count > new_kv.len() / dim || pos + count > rows {
                return Err(invalid(node, "cache write runs past the cache"));
            }
            let mut out = cache.data.clone();
            out[pos * dim..(pos + count) * dim].copy_from_slice(&new_kv.data[..count * dim]);
            out
        }
        Op::PrefixLast => {
            let (x, valid) = (arg(0)?, arg(1)?);
            let (rows, dim) = rows_cols(x);
            let valid = valid.index(0);
            if valid == 0 || valid > rows {
                return Err(invalid(
                    node,
                    format!("valid length {valid} outside 1..={rows}"),
                ));
            }
            x.data[(valid - 1) * dim..valid * dim].to_vec()
        }
        Op::MulPerChannel { channels, spatial } => {
            // dst[n,c,s] = src[n,c,s] · gate[n,c]
            let (src, gate) = (arg(0)?, arg(1)?);
            let spatial = spatial as usize;
            let batch = nchw(node, src, channels, spatial)?;
            if gate.len() != batch * channels as usize {
                return Err(invalid(node, "gate must be [N*C]"));
            }
            map_indexed(src, |i, v| v * gate.data[i / spatial])
        }
        Op::AddPerChannel { channels, spatial } => {
            // dst[n,c,s] = src[n,c,s] + bias[c]
            let (src, bias) = (arg(0)?, arg(1)?);
            let spatial = spatial as usize;
            nchw(node, src, channels, spatial)?;
            if bias.len() != channels as usize {
                return Err(invalid(node, "bias must be [C]"));
            }
            let c = channels as usize;
            map_indexed(src, |i, v| v + bias.data[(i / spatial) % c])
        }
        Op::GlobalAvgPool { channels, spatial } => {
            let x = arg(0)?;
            let spatial = spatial as usize;
            nchw(node, x, channels, spatial)?;
            x.data
                .chunks(spatial)
                .map(|plane| plane.iter().sum::<f64>() / spatial as f64)
                .collect()
        }
        Op::GlobalAvgPoolGrad { spatial, .. } => {
            let g = arg(0)?;
            let spatial = spatial as usize;
            (0..g.len() * spatial)
                .map(|i| g.data[i / spatial] / spatial as f64)
                .collect()
        }
        Op::Concat {
            channels_a,
            channels_b,
            spatial,
        } => {
            let (a, b) = (arg(0)?, arg(1)?);
            let (ca, cb, s) = (channels_a as usize, channels_b as usize, spatial as usize);
            let batch = nchw(node, a, channels_a, s)?;
            if b.len() != batch * cb * s {
                return Err(invalid(node, "concat operands disagree on batch"));
            }
            let mut out = Vec::with_capacity(a.len() + b.len());
            for n in 0..batch {
                out.extend_from_slice(&a.data[n * ca * s..(n + 1) * ca * s]);
                out.extend_from_slice(&b.data[n * cb * s..(n + 1) * cb * s]);
            }
            out
        }
        Op::SplitA {
            channels_a,
            channels_b,
            spatial,
        }
        | Op::SplitB {
            channels_a,
            channels_b,
            spatial,
        } => {
            let x = arg(0)?;
            let (ca, cb, s) = (channels_a as usize, channels_b as usize, spatial as usize);
            let batch = nchw(node, x, channels_a + channels_b, s)?;
            let (start, width) = if matches!(node.op, Op::SplitA { .. }) {
                (0, ca)
            } else {
                (ca, cb)
            };
            let mut out = Vec::with_capacity(batch * width * s);
            for n in 0..batch {
                let base = n * (ca + cb) * s + start * s;
                out.extend_from_slice(&x.data[base..base + width * s]);
            }
            out
        }
        Op::Upsample2x {
            channels,
            in_h,
            in_w,
        } => {
            let x = arg(0)?;
            let (h, w) = (in_h as usize, in_w as usize);
            let planes = nchw(node, x, channels, h * w)? * channels as usize;
            let mut out = vec![0.0; planes * 4 * h * w];
            for p in 0..planes {
                for y in 0..2 * h {
                    for xx in 0..2 * w {
                        out[p * 4 * h * w + y * 2 * w + xx] =
                            x.data[p * h * w + (y / 2) * w + xx / 2];
                    }
                }
            }
            out
        }
        Op::Upsample2xGrad {
            channels,
            in_h,
            in_w,
        } => {
            let g = arg(0)?;
            let (h, w) = (in_h as usize, in_w as usize);
            let planes = nchw(node, g, channels, 4 * h * w)? * channels as usize;
            let mut out = vec![0.0; planes * h * w];
            for p in 0..planes {
                for y in 0..2 * h {
                    for xx in 0..2 * w {
                        out[p * h * w + (y / 2) * w + xx / 2] +=
                            g.data[p * 4 * h * w + y * 2 * w + xx];
                    }
                }
            }
            out
        }
        _ => unreachable!("{:?} is not a basic op", node.op),
    };
    Ok(out)
}

fn map_indexed(a: &Tensor, f: impl Fn(usize, f64) -> f64) -> Vec<f64> {
    a.data.iter().enumerate().map(|(i, &v)| f(i, v)).collect()
}

/// Op-specific error scales where `|result|` understates the rounding error.
pub(super) fn magnitude(node: &Node, ins: &[&Tensor], out: &Tensor) -> Option<Vec<f64>> {
    match node.op {
        // x - lse cancels when x is near the log-sum-exp.
        Op::LogSoftmax => {
            let x = ins[0];
            let (rows, cols) = rows_cols(x);
            let mut m = vec![0.0; x.len()];
            for r in 0..rows {
                let lse = log_sum_exp(&x.data[r * cols..(r + 1) * cols]);
                for c in 0..cols {
                    m[r * cols + c] = x.data[r * cols + c].abs() + lse.abs();
                }
            }
            Some(m)
        }
        // The loss sums |labels · log_softmax| terms.
        Op::CrossEntropyLoss => {
            let (logits, labels) = (ins[0], ins[1]);
            let (rows, cols) = rows_cols(logits);
            let mut total = 0.0;
            for r in 0..rows {
                let row = &logits.data[r * cols..(r + 1) * cols];
                let lse = log_sum_exp(row);
                for c in 0..cols {
                    total += (labels.data[r * cols + c] * (row[c] - lse)).abs()
                        + (labels.data[r * cols + c] * lse).abs();
                }
            }
            Some(vec![total / rows as f64])
        }
        // softmax · S − labels cancels for one-hot rows.
        Op::CrossEntropyLogitsGrad => {
            let (logits, labels) = (ins[0], ins[1]);
            let (rows, cols) = rows_cols(logits);
            let softmax = softmax_rows(logits);
            let mut m = vec![0.0; logits.len()];
            for r in 0..rows {
                let s: f64 = labels.data[r * cols..(r + 1) * cols]
                    .iter()
                    .map(|v| v.abs())
                    .sum();
                for c in 0..cols {
                    let i = r * cols + c;
                    m[i] = (softmax[i] * s + labels.data[i].abs()) / rows as f64;
                }
            }
            Some(m)
        }
        Op::BceLoss => {
            let pred = ins[0];
            Some(vec![out.data[0].abs() + 1.0 / pred.len() as f64])
        }
        // Row-sum normalisation cancels in the subtracted term.
        Op::NormalizeInnerSumGrad { inner, floor } => {
            let (g, x) = (ins[0], ins[1]);
            let (inner, floor) = (inner as usize, f64::from(floor));
            let mut m = vec![0.0; x.len()];
            for r in 0..x.len() / inner {
                let range = r * inner..(r + 1) * inner;
                let sum: f64 = x.data[range.clone()].iter().sum();
                let d = (sum - floor).max(0.0) + floor;
                let dot: f64 = range.clone().map(|i| (g.data[i] * x.data[i]).abs()).sum();
                for i in range {
                    m[i] = g.data[i].abs() / d + dot / (d * d);
                }
            }
            Some(m)
        }
        // log(x) near x = 1 has an absolute, not a relative, error.
        Op::Log => Some(
            ins[0]
                .data
                .iter()
                .zip(&out.data)
                .map(|(_, y)| y.abs() + 1.0)
                .collect(),
        ),
        _ => None,
    }
}
