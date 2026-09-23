//! Attention: every query row takes a softmax-weighted average of the value
//! rows in one contiguous range of keys.
//!
//! All ops share one layout. Q is `[q_rows, heads * head_dim]`, K and V are
//! `[kv_rows, kv_heads * head_dim]`, and query head `h` reads key/value head
//! `h / (heads / kv_heads)` (grouped-query attention). Unless stated
//! otherwise the logit is `q·k / sqrt(head_dim)`. The ops differ only in
//! which keys a query row sees:
//!
//! * causal (`CausalAttention`, `CausalAttentionRoPE`): keys `0..=i`;
//! * sliding window `W`: the last `W` keys up to and including `i`;
//! * full, cross and `MultiHeadAttn` (either `is_cross`): every key;
//! * cached: every query sees cache rows `0..=kv_pos`;
//! * cached block: row `i` sees absolute position `p = kv_pos + i` causally,
//!   limited to the last `window_size` keys when that is nonzero. Rows at or
//!   past `valid_len` are unspecified (NaN);
//! * chunked relative: keys at distance `< left_context - 1`, with
//!   Transformer-XL relative keys and a soft-capped, unscaled logit.
//!
//! The backward ops take `[dO, q, k, v]` and differentiate the forward op
//! named by `fwd_node` at their own `q, k, v`. The mask comes from that
//! forward op; for `CausalAttentionRoPE` autodiff feeds them the rotated
//! `q, k`, so they apply no rotation themselves.

use super::{Error, Tensor};
use crate::graph::{Graph, Node, Op};
use std::borrow::Cow;
use std::ops::Range;

fn invalid(node: &Node, reason: impl Into<String>) -> Error {
    Error::Invalid {
        node: node.id,
        reason: reason.into(),
    }
}

#[derive(Clone, Copy, Debug)]
struct Heads {
    heads: usize,
    kv_heads: usize,
    dim: usize,
}

impl Heads {
    fn new(node: &Node, heads: u32, kv_heads: u32, dim: u32) -> Result<Self, Error> {
        if heads == 0 || kv_heads == 0 || dim == 0 || !heads.is_multiple_of(kv_heads) {
            return Err(invalid(
                node,
                format!("{heads} heads cannot share {kv_heads} KV heads of width {dim}"),
            ));
        }
        Ok(Self {
            heads: heads as usize,
            kv_heads: kv_heads as usize,
            dim: dim as usize,
        })
    }

    fn kv_head(&self, h: usize) -> usize {
        h / (self.heads / self.kv_heads)
    }
}

/// Rows of a 2-D tensor with `cols` columns.
fn rows(node: &Node, t: &Tensor, cols: usize, what: &str) -> Result<usize, Error> {
    if t.shape.len() != 2 || t.shape[1] != cols {
        return Err(invalid(
            node,
            format!("{what} has shape {:?}, expected [_, {cols}]", t.shape),
        ));
    }
    Ok(t.shape[0])
}

/// Rotary embedding with position = row index, rotating the pairs
/// `(d, d + head_dim/2)` of every head by `row · theta^(-2d/head_dim)`.
fn rope(x: &[f64], width: usize, head_dim: usize, theta: f64) -> Vec<f64> {
    let half = head_dim / 2;
    let mut out = x.to_vec();
    for row in 0..x.len() / width {
        for head in 0..width / head_dim {
            for p in 0..half {
                let inv_freq = theta.powf(-2.0 * p as f64 / head_dim as f64);
                let (sin, cos) = (row as f64 * inv_freq).sin_cos();
                let i0 = row * width + head * head_dim + p;
                let i1 = i0 + half;
                out[i0] = x[i0] * cos - x[i1] * sin;
                out[i1] = x[i0] * sin + x[i1] * cos;
            }
        }
    }
    out
}

enum Logit<'a> {
    /// `q·k / sqrt(head_dim)`.
    Scaled,
    /// `cap · tanh(q·(k + r) / cap)` where `r` is row `left - 1 - (i - j)`
    /// of the relative keys.
    Relative {
        rel: &'a [f64],
        left: usize,
        cap: f64,
    },
}

struct Problem<'a> {
    heads: Heads,
    q: Cow<'a, [f64]>,
    k: Cow<'a, [f64]>,
    v: &'a [f64],
    logit: Logit<'a>,
    /// Keys each query row attends, or `None` for an unspecified row.
    keys: Vec<Option<Range<usize>>>,
    /// Extra error scale of a logit between rows `i` and `j` relative to
    /// its absolute terms: RoPE angles computed in `f32` lose accuracy in
    /// proportion to the position (`pos · (ln θ + 2)` roundings).
    rope_ln_theta: Option<f64>,
}

/// Softmax weights of one (query row, head) over its key range.
struct Weights {
    keys: Range<usize>,
    p: Vec<f64>,
    /// Error scale of each exponent `s_j - max`.
    err: Vec<f64>,
}

enum Grad {
    Q,
    K,
    V,
}

/// Which gradient a backward op computes, and from which `dO`.
type Upstream<'a> = (Grad, &'a [f64]);

impl Problem<'_> {
    fn q_width(&self) -> usize {
        self.heads.heads * self.heads.dim
    }

    fn kv_width(&self) -> usize {
        self.heads.kv_heads * self.heads.dim
    }

    fn q_row(&self, i: usize, h: usize) -> &[f64] {
        let base = i * self.q_width() + h * self.heads.dim;
        &self.q[base..base + self.heads.dim]
    }

    fn kv_base(&self, j: usize, h: usize) -> usize {
        j * self.kv_width() + self.heads.kv_head(h) * self.heads.dim
    }

    /// The logit and the size of its summed terms.
    fn logit(&self, i: usize, h: usize, j: usize) -> (f64, f64) {
        let q = self.q_row(i, h);
        let kb = self.kv_base(j, h);
        let k = &self.k[kb..kb + self.heads.dim];
        match self.logit {
            Logit::Scaled => {
                let scale = 1.0 / (self.heads.dim as f64).sqrt();
                let dot: f64 = q.iter().zip(k).map(|(a, b)| a * b).sum();
                let abs: f64 = q.iter().zip(k).map(|(a, b)| (a * b).abs()).sum();
                let rope = self
                    .rope_ln_theta
                    .map_or(1.0, |ln| 1.0 + (i + j) as f64 * (ln + 2.0));
                (dot * scale, abs * scale * rope)
            }
            Logit::Relative { rel, left, cap } => {
                let rb = (left - 1 - (i - j)) * self.q_width() + h * self.heads.dim;
                let r = &rel[rb..rb + self.heads.dim];
                let mut dot = 0.0;
                let mut abs = 0.0;
                for d in 0..self.heads.dim {
                    dot += q[d] * (k[d] + r[d]);
                    abs += q[d].abs() * (k[d].abs() + r[d].abs());
                }
                (cap * (dot / cap).tanh(), abs)
            }
        }
    }

    fn weights(&self, i: usize, h: usize) -> Option<Weights> {
        let keys = self.keys[i].clone()?;
        let logits: Vec<(f64, f64)> = keys.clone().map(|j| self.logit(i, h, j)).collect();
        let max = logits.iter().map(|l| l.0).fold(f64::NEG_INFINITY, f64::max);
        let e: Vec<f64> = logits.iter().map(|l| (l.0 - max).exp()).collect();
        let sum: f64 = e.iter().sum();
        Some(Weights {
            keys,
            p: e.iter().map(|x| x / sum).collect(),
            err: logits.iter().map(|l| l.1 + l.0.abs() + max.abs()).collect(),
        })
    }

    /// `O = P·V` and its error scale `Σ p_j (|v_j| + |v_j − O| · err_j)`:
    /// an error `δ` in logit `j` moves the output by `p_j (v_j − O) δ`.
    fn forward(&self) -> (Vec<f64>, Vec<f64>) {
        let (hd, width) = (self.heads.dim, self.q_width());
        let n = self.keys.len() * width;
        let mut out = vec![f64::NAN; n];
        let mut mag = vec![f64::NAN; n];
        for i in 0..self.keys.len() {
            for h in 0..self.heads.heads {
                let Some(w) = self.weights(i, h) else {
                    continue;
                };
                let base = i * width + h * hd;
                for d in 0..hd {
                    let o: f64 = w
                        .keys
                        .clone()
                        .zip(&w.p)
                        .map(|(j, p)| p * self.v[self.kv_base(j, h) + d])
                        .sum();
                    let m: f64 = w
                        .keys
                        .clone()
                        .enumerate()
                        .map(|(n, j)| {
                            let v = self.v[self.kv_base(j, h) + d];
                            w.p[n] * (v.abs() + (v - o).abs() * w.err[n])
                        })
                        .sum();
                    out[base + d] = o;
                    mag[base + d] = m;
                }
            }
        }
        (out, mag)
    }

    /// One gradient of the loss with respect to `q`, `k` or `v`, given
    /// `dO`, and its error scale.
    ///
    /// With `dP_ij = dO_i·v_j`, `D_i = dO_i·O_i` and
    /// `dS_ij = p_ij (dP_ij − D_i)`: `dQ_i = s Σ_j dS_ij k_j`,
    /// `dK_j = s Σ_i dS_ij q_i`, `dV_j = Σ_i p_ij dO_i`, summed over the
    /// query heads that share a KV head.
    fn backward(&self, d_out: &[f64], which: &Grad) -> (Vec<f64>, Vec<f64>) {
        let (hd, qw) = (self.heads.dim, self.q_width());
        let scale = 1.0 / (hd as f64).sqrt();
        let (o, o_mag) = self.forward();
        let n = match *which {
            Grad::Q => self.q.len(),
            Grad::K | Grad::V => self.k.len(),
        };
        let mut grad = vec![0.0; n];
        let mut mag = vec![0.0; n];
        for i in 0..self.keys.len() {
            for h in 0..self.heads.heads {
                let Some(w) = self.weights(i, h) else {
                    continue;
                };
                let base = i * qw + h * hd;
                let d_o = &d_out[base..base + hd];
                let q = self.q_row(i, h);
                let mut dd = 0.0;
                let mut dd_abs = 0.0;
                for d in 0..hd {
                    dd += d_o[d] * o[base + d];
                    dd_abs += d_o[d].abs() * o_mag[base + d];
                }
                for (n, j) in w.keys.clone().enumerate() {
                    let kb = self.kv_base(j, h);
                    let v = &self.v[kb..kb + hd];
                    let k = &self.k[kb..kb + hd];
                    let p = w.p[n];
                    let mut dp = 0.0;
                    let mut dp_abs = 0.0;
                    for d in 0..hd {
                        dp += d_o[d] * v[d];
                        dp_abs += (d_o[d] * v[d]).abs();
                    }
                    let ds = p * (dp - dd);
                    let t = p * (dp_abs + dd_abs + (dp - dd).abs() * w.err[n]);
                    match *which {
                        Grad::Q => {
                            for d in 0..hd {
                                grad[base + d] += scale * ds * k[d];
                                mag[base + d] += scale * t * k[d].abs();
                            }
                        }
                        Grad::K => {
                            for d in 0..hd {
                                grad[kb + d] += scale * ds * q[d];
                                mag[kb + d] += scale * t * q[d].abs();
                            }
                        }
                        Grad::V => {
                            for d in 0..hd {
                                grad[kb + d] += p * d_o[d];
                                mag[kb + d] += p * d_o[d].abs() * (1.0 + w.err[n]);
                            }
                        }
                    }
                }
            }
        }
        (grad, mag)
    }
}

/// Keys `0..=i` for each of `n` rows.
fn causal(n: usize, window: usize) -> Vec<Option<Range<usize>>> {
    (0..n)
        .map(|i| {
            let start = if window > 0 {
                (i + 1).saturating_sub(window)
            } else {
                0
            };
            Some(start..i + 1)
        })
        .collect()
}

/// The mask a backward op inherits from its forward op.
enum Mask {
    Causal { window: usize },
    Full,
}

fn forward_mask(graph: &Graph, node: &Node, fwd: u32) -> Result<Mask, Error> {
    match graph.node(fwd).op {
        Op::CausalAttention { .. } | Op::CausalAttentionRoPE { .. } => {
            Ok(Mask::Causal { window: 0 })
        }
        Op::SlidingWindowAttention { window_size, .. } => Ok(Mask::Causal {
            window: window_size as usize,
        }),
        Op::FullAttention { .. } | Op::CrossAttention { .. } | Op::MultiHeadAttn { .. } => {
            Ok(Mask::Full)
        }
        ref other => Err(invalid(
            node,
            format!("attention gradient refers to {other:?}, not an attention op"),
        )),
    }
}

/// Build the problem a node poses, and for a backward op, which gradient
/// it computes from which `dO`.
fn problem<'a>(
    graph: &Graph,
    node: &Node,
    ins: &[&'a Tensor],
) -> Result<(Problem<'a>, Option<Upstream<'a>>), Error> {
    let arg = |i: usize| {
        ins.get(i)
            .copied()
            .ok_or_else(|| invalid(node, format!("missing input {i}")))
    };
    let (qkv, grad) = match node.op {
        Op::MultiHeadAttnGradQ { .. } => (1, Some(Grad::Q)),
        Op::MultiHeadAttnGradK { .. } => (1, Some(Grad::K)),
        Op::MultiHeadAttnGradV { .. } => (1, Some(Grad::V)),
        _ => (0, None),
    };
    let (q, k, v) = (arg(qkv)?, arg(qkv + 1)?, arg(qkv + 2)?);
    let (heads, kv_heads, dim) = match node.op {
        Op::CausalAttention {
            num_heads,
            num_kv_heads,
            head_dim,
        }
        | Op::CausalAttentionRoPE {
            num_heads,
            num_kv_heads,
            head_dim,
            ..
        }
        | Op::FullAttention {
            num_heads,
            num_kv_heads,
            head_dim,
        }
        | Op::CrossAttention {
            num_heads,
            num_kv_heads,
            head_dim,
        }
        | Op::MultiHeadAttn {
            num_heads,
            num_kv_heads,
            head_dim,
            ..
        }
        | Op::MultiHeadAttnGradQ {
            num_heads,
            num_kv_heads,
            head_dim,
            ..
        }
        | Op::MultiHeadAttnGradK {
            num_heads,
            num_kv_heads,
            head_dim,
            ..
        }
        | Op::MultiHeadAttnGradV {
            num_heads,
            num_kv_heads,
            head_dim,
            ..
        }
        | Op::SlidingWindowAttention {
            num_heads,
            num_kv_heads,
            head_dim,
            ..
        }
        | Op::CachedAttention {
            num_heads,
            num_kv_heads,
            head_dim,
        }
        | Op::CachedBlockAttention {
            num_heads,
            num_kv_heads,
            head_dim,
            ..
        } => (num_heads, num_kv_heads, head_dim),
        Op::ChunkedRelativeAttention {
            num_heads,
            head_dim,
            ..
        } => (num_heads, num_heads, head_dim),
        _ => return Err(invalid(node, "not an attention op")),
    };
    let heads = Heads::new(node, heads, kv_heads, dim)?;
    let q_rows = rows(node, q, heads.heads * heads.dim, "Q")?;
    let kv_rows = rows(node, k, heads.kv_heads * heads.dim, "K")?;
    if v.shape != k.shape {
        return Err(invalid(node, "V shape differs from K"));
    }
    let same_length = || {
        if q_rows == kv_rows {
            Ok(())
        } else {
            Err(invalid(
                node,
                "self-attention needs as many keys as queries",
            ))
        }
    };

    let mut logit = Logit::Scaled;
    let mut rope_ln_theta = None;
    let (mut q_data, mut k_data) = (Cow::Borrowed(&q.data[..]), Cow::Borrowed(&k.data[..]));
    let keys = match node.op {
        Op::CausalAttention { .. } => {
            same_length()?;
            causal(q_rows, 0)
        }
        Op::CausalAttentionRoPE { rope_theta, .. } => {
            same_length()?;
            if heads.dim % 2 != 0 {
                return Err(invalid(node, "RoPE needs an even head dimension"));
            }
            let theta = f64::from(rope_theta);
            q_data = Cow::Owned(rope(&q.data, heads.heads * heads.dim, heads.dim, theta));
            k_data = Cow::Owned(rope(&k.data, heads.kv_heads * heads.dim, heads.dim, theta));
            rope_ln_theta = Some(theta.ln().abs());
            causal(q_rows, 0)
        }
        Op::SlidingWindowAttention { window_size, .. } => {
            same_length()?;
            causal(q_rows, window_size as usize)
        }
        Op::FullAttention { .. } | Op::CrossAttention { .. } | Op::MultiHeadAttn { .. } => {
            vec![Some(0..kv_rows); q_rows]
        }
        Op::MultiHeadAttnGradQ { fwd_node, .. }
        | Op::MultiHeadAttnGradK { fwd_node, .. }
        | Op::MultiHeadAttnGradV { fwd_node, .. } => match forward_mask(graph, node, fwd_node)? {
            Mask::Causal { window } => {
                same_length()?;
                causal(q_rows, window)
            }
            Mask::Full => vec![Some(0..kv_rows); q_rows],
        },
        Op::CachedAttention { .. } => {
            let pos = arg(3)?.index(0);
            if pos >= kv_rows {
                return Err(invalid(node, format!("kv_pos {pos} outside the cache")));
            }
            vec![Some(0..pos + 1); q_rows]
        }
        Op::CachedBlockAttention { window_size, .. } => {
            let pos = arg(3)?.index(0);
            let valid = arg(4)?.index(0).min(q_rows);
            if pos + valid > kv_rows {
                return Err(invalid(
                    node,
                    format!("block at {pos} with {valid} rows runs past the cache"),
                ));
            }
            let window = window_size as usize;
            (0..q_rows)
                .map(|i| {
                    let end = pos + i + 1;
                    let start = if window > 0 {
                        end.saturating_sub(window)
                    } else {
                        0
                    };
                    (i < valid).then_some(start..end)
                })
                .collect()
        }
        Op::ChunkedRelativeAttention {
            left_context,
            softcap_bits,
            ..
        } => {
            same_length()?;
            let left = left_context as usize;
            let rel = arg(3)?;
            if left < 2 || rel.shape != [left, heads.heads * heads.dim] {
                return Err(invalid(node, "relative keys need [left_context, width]"));
            }
            logit = Logit::Relative {
                rel: &rel.data,
                left,
                cap: f64::from(f32::from_bits(softcap_bits)),
            };
            (0..q_rows)
                .map(|i| Some(i - i.min(left - 2)..i + 1))
                .collect()
        }
        _ => unreachable!(),
    };
    let grad = match grad {
        Some(which) => {
            let d_out = arg(0)?;
            if d_out.shape != q.shape {
                return Err(invalid(node, "dO shape differs from Q"));
            }
            Some((which, &d_out.data[..]))
        }
        None => None,
    };
    Ok((
        Problem {
            heads,
            q: q_data,
            k: k_data,
            v: &v.data,
            logit,
            keys,
            rope_ln_theta,
        },
        grad,
    ))
}

fn solve(graph: &Graph, node: &Node, ins: &[&Tensor]) -> Result<(Vec<f64>, Vec<f64>), Error> {
    let (problem, grad) = problem(graph, node, ins)?;
    Ok(match grad {
        None => problem.forward(),
        Some((which, d_out)) => problem.backward(d_out, &which),
    })
}

pub(super) fn eval(graph: &Graph, node: &Node, ins: &[&Tensor]) -> Result<Vec<f64>, Error> {
    solve(graph, node, ins).map(|r| r.0)
}

pub(super) fn magnitude(
    graph: &Graph,
    node: &Node,
    ins: &[&Tensor],
    _out: &Tensor,
) -> Option<Vec<f64>> {
    match node.op {
        Op::CausalAttention { .. }
        | Op::CausalAttentionRoPE { .. }
        | Op::FullAttention { .. }
        | Op::CrossAttention { .. }
        | Op::MultiHeadAttn { .. }
        | Op::MultiHeadAttnGradQ { .. }
        | Op::MultiHeadAttnGradK { .. }
        | Op::MultiHeadAttnGradV { .. }
        | Op::SlidingWindowAttention { .. }
        | Op::CachedAttention { .. }
        | Op::CachedBlockAttention { .. }
        | Op::ChunkedRelativeAttention { .. } => solve(graph, node, ins).ok().map(|r| r.1),
        _ => None,
    }
}
