//! Convolution, pooling and group normalization over flat NCHW tensors.
//!
//! Every op here reads its batch size from the length of its first input.
//! Convolutions are cross-correlations (no kernel flip) with zero padding,
//! as in PyTorch; max pooling ignores padded positions.

use super::{Error, Tensor};
use crate::graph::{Node, Op};

fn invalid(node: &Node, reason: impl Into<String>) -> Error {
    Error::Invalid {
        node: node.id,
        reason: reason.into(),
    }
}

/// Geometry of a convolution or pooling window.
#[derive(Clone, Copy, Debug)]
struct Window {
    batch: usize,
    /// Input channels (the channel count for depthwise ops and pooling).
    ci: usize,
    h: usize,
    w: usize,
    /// Output channels (equal to `ci` for depthwise ops and pooling).
    co: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    ph: usize,
    pw: usize,
    oh: usize,
    ow: usize,
    /// Output channel `c` reads input channel `c` only, with filter `[c, kh, kw]`.
    depthwise: bool,
}

impl Window {
    #[allow(clippy::too_many_arguments)]
    fn new(
        node: &Node,
        ci: u32,
        h: u32,
        w: u32,
        co: u32,
        kh: u32,
        kw: u32,
        stride: u32,
        ph: u32,
        pw: u32,
        depthwise: bool,
    ) -> Result<Self, Error> {
        let [ci, h, w, co, kh, kw, stride, ph, pw] =
            [ci, h, w, co, kh, kw, stride, ph, pw].map(|v| v as usize);
        if stride == 0 || kh == 0 || kw == 0 || kh > h + 2 * ph || kw > w + 2 * pw {
            return Err(invalid(node, "window does not fit the padded input"));
        }
        Ok(Self {
            batch: 0,
            ci,
            h,
            w,
            co,
            kh,
            kw,
            stride,
            ph,
            pw,
            oh: (h + 2 * ph - kh) / stride + 1,
            ow: (w + 2 * pw - kw) / stride + 1,
            depthwise,
        })
    }

    fn input_len(&self) -> usize {
        self.batch * self.ci * self.h * self.w
    }

    fn output_len(&self) -> usize {
        self.batch * self.co * self.oh * self.ow
    }

    fn kernel_len(&self) -> usize {
        let ci = if self.depthwise { 1 } else { self.ci };
        self.co * ci * self.kh * self.kw
    }

    /// Set the batch from the length of an input-shaped tensor.
    fn batch_from_input(mut self, node: &Node, len: usize) -> Result<Self, Error> {
        let plane = self.ci * self.h * self.w;
        if plane == 0 || !len.is_multiple_of(plane) {
            return Err(invalid(
                node,
                format!("{len} values are not whole NCHW images"),
            ));
        }
        self.batch = len / plane;
        Ok(self)
    }

    /// Set the batch from the length of an output-shaped tensor.
    fn batch_from_output(mut self, node: &Node, len: usize) -> Result<Self, Error> {
        let plane = self.co * self.oh * self.ow;
        if plane == 0 || !len.is_multiple_of(plane) {
            return Err(invalid(
                node,
                format!("{len} values are not whole NCHW outputs"),
            ));
        }
        self.batch = len / plane;
        Ok(self)
    }

    /// Input row/column read by output position `o` at tap `k`, if not padding.
    fn source(o: usize, k: usize, stride: usize, pad: usize, len: usize) -> Option<usize> {
        (o * stride + k).checked_sub(pad).filter(|&i| i < len)
    }

    /// Call `f(output, input, kernel)` with the flat indices of every term
    /// `y[output] += x[input] · w[kernel]` of the convolution.
    fn for_each_tap(&self, mut f: impl FnMut(usize, usize, usize)) {
        for n in 0..self.batch {
            for co in 0..self.co {
                let channels = if self.depthwise {
                    co..co + 1
                } else {
                    0..self.ci
                };
                for oy in 0..self.oh {
                    for ox in 0..self.ow {
                        let y = ((n * self.co + co) * self.oh + oy) * self.ow + ox;
                        for ci in channels.clone() {
                            for ky in 0..self.kh {
                                let Some(iy) = Self::source(oy, ky, self.stride, self.ph, self.h)
                                else {
                                    continue;
                                };
                                for kx in 0..self.kw {
                                    let Some(ix) =
                                        Self::source(ox, kx, self.stride, self.pw, self.w)
                                    else {
                                        continue;
                                    };
                                    let x = ((n * self.ci + ci) * self.h + iy) * self.w + ix;
                                    let filter = if self.depthwise {
                                        co
                                    } else {
                                        co * self.ci + ci
                                    };
                                    let k = (filter * self.kh + ky) * self.kw + kx;
                                    f(y, x, k);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Input index of the first maximum of each pooling window, in
    /// row-major `(ky, kx)` order. Padded positions never win.
    fn winners(&self, node: &Node, x: &Tensor) -> Result<Vec<usize>, Error> {
        let mut winners = Vec::with_capacity(self.output_len());
        for plane in 0..self.batch * self.ci {
            for oy in 0..self.oh {
                for ox in 0..self.ow {
                    let mut best: Option<usize> = None;
                    for ky in 0..self.kh {
                        let Some(iy) = Self::source(oy, ky, self.stride, self.ph, self.h) else {
                            continue;
                        };
                        for kx in 0..self.kw {
                            let Some(ix) = Self::source(ox, kx, self.stride, self.pw, self.w)
                            else {
                                continue;
                            };
                            let i = (plane * self.h + iy) * self.w + ix;
                            if best.is_none_or(|b| x.data[i] > x.data[b]) {
                                best = Some(i);
                            }
                        }
                    }
                    let best = best.ok_or_else(|| {
                        invalid(
                            node,
                            format!("pooling window ({oy}, {ox}) covers only padding"),
                        )
                    })?;
                    winners.push(best);
                }
            }
        }
        Ok(winners)
    }
}

fn conv_window(node: &Node) -> Result<Window, Error> {
    match node.op {
        Op::Conv2d {
            in_channels,
            in_h,
            in_w,
            out_channels,
            kernel_h,
            kernel_w,
            stride,
            padding_h,
            padding_w,
        }
        | Op::Conv2dGradInput {
            in_channels,
            in_h,
            in_w,
            out_channels,
            kernel_h,
            kernel_w,
            stride,
            padding_h,
            padding_w,
        }
        | Op::Conv2dGradWeight {
            in_channels,
            in_h,
            in_w,
            out_channels,
            kernel_h,
            kernel_w,
            stride,
            padding_h,
            padding_w,
        } => Window::new(
            node,
            in_channels,
            in_h,
            in_w,
            out_channels,
            kernel_h,
            kernel_w,
            stride,
            padding_h,
            padding_w,
            false,
        ),
        // F(2,3) is an evaluation strategy for the 3×3 stride-1 convolution.
        Op::WinogradConv2d {
            in_channels,
            in_h,
            in_w,
            out_channels,
            padding,
            ..
        } => Window::new(
            node,
            in_channels,
            in_h,
            in_w,
            out_channels,
            3,
            3,
            1,
            padding,
            padding,
            false,
        ),
        Op::Conv2dDw {
            channels,
            in_h,
            in_w,
            kernel_h,
            kernel_w,
            stride,
            padding_h,
            padding_w,
        } => Window::new(
            node, channels, in_h, in_w, channels, kernel_h, kernel_w, stride, padding_h, padding_w,
            true,
        ),
        Op::MaxPool2d {
            channels,
            in_h,
            in_w,
            kernel_h,
            kernel_w,
            stride,
            padding,
        }
        | Op::MaxPool2dGrad {
            channels,
            in_h,
            in_w,
            kernel_h,
            kernel_w,
            stride,
            padding,
        } => Window::new(
            node, channels, in_h, in_w, channels, kernel_h, kernel_w, stride, padding, padding,
            true,
        ),
        _ => unreachable!(),
    }
}

fn expect_len(node: &Node, what: &str, t: &Tensor, len: usize) -> Result<(), Error> {
    if t.len() != len {
        return Err(invalid(
            node,
            format!("{what} has {} values, expected {len}", t.len()),
        ));
    }
    Ok(())
}

/// `y = x ⋆ w` for `Conv2d`, `Conv2dDw` and `WinogradConv2d`.
fn convolve(node: &Node, g: Window, x: &Tensor, w: &Tensor) -> Result<Vec<f64>, Error> {
    let g = g.batch_from_input(node, x.len())?;
    expect_len(node, "kernel", w, g.kernel_len())?;
    let mut y = vec![0.0; g.output_len()];
    g.for_each_tap(|yi, xi, ki| y[yi] += x.data[xi] * w.data[ki]);
    Ok(y)
}

/// Max pooling backward: `dx[winner(o)] += dy[o]` for every output `o`.
fn max_pool_grad(node: &Node, g: Window, dy: &[f64], x: &Tensor) -> Result<Vec<f64>, Error> {
    let winners = g.winners(node, x)?;
    let mut dx = vec![0.0; x.len()];
    for (o, &i) in winners.iter().enumerate() {
        dx[i] += dy[o];
    }
    Ok(dx)
}

/// Group-normalization layout: `x[n, c, s]` in groups of `C / G` channels.
#[derive(Clone, Copy, Debug)]
struct Groups {
    batch: usize,
    channels: usize,
    spatial: usize,
    groups: usize,
    eps: f64,
}

impl Groups {
    fn new(node: &Node, len: usize) -> Result<Self, Error> {
        let (num_groups, eps, channels, spatial) = match node.op {
            Op::GroupNorm {
                num_groups,
                eps,
                channels,
                spatial,
            }
            | Op::GroupNormSilu {
                num_groups,
                eps,
                channels,
                spatial,
            }
            | Op::GroupNormGradInput {
                num_groups,
                eps,
                channels,
                spatial,
            }
            | Op::GroupNormGradWeightBias {
                num_groups,
                eps,
                channels,
                spatial,
            } => (num_groups, eps, channels, spatial),
            _ => unreachable!(),
        };
        let (channels, spatial, groups) =
            (channels as usize, spatial as usize, num_groups as usize);
        if groups == 0 || channels % groups != 0 {
            return Err(invalid(node, "channels are not divisible into groups"));
        }
        let plane = channels * spatial;
        if plane == 0 || !len.is_multiple_of(plane) {
            return Err(invalid(
                node,
                format!("{len} values are not whole NCHW images"),
            ));
        }
        Ok(Self {
            batch: len / plane,
            channels,
            spatial,
            groups,
            eps: f64::from(eps),
        })
    }

    fn len(&self) -> usize {
        self.batch * self.channels * self.spatial
    }

    /// Size of one `(n, group)` block, contiguous in NCHW.
    fn group_len(&self) -> usize {
        self.channels / self.groups * self.spatial
    }

    fn channel(&self, i: usize) -> usize {
        i / self.spatial % self.channels
    }

    /// `(mean, 1 / sqrt(var + eps))` per `(n, group)`, with the biased
    /// (divide by N) variance.
    fn stats(&self, x: &[f64]) -> Vec<(f64, f64)> {
        x.chunks(self.group_len())
            .map(|block| {
                let n = block.len() as f64;
                let mean = block.iter().sum::<f64>() / n;
                let var = block.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
                (mean, 1.0 / (var + self.eps).sqrt())
            })
            .collect()
    }
}

fn group_norm(g: &Groups, x: &[f64], w: &[f64], b: &[f64]) -> Vec<f64> {
    let stats = g.stats(x);
    (0..x.len())
        .map(|i| {
            let (mean, inv_std) = stats[i / g.group_len()];
            let c = g.channel(i);
            (x[i] - mean) * inv_std * w[c] + b[c]
        })
        .collect()
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// `dx = inv_std · (w·dy − mean(w·dy) − x̂ · mean(w·dy·x̂))`, the exact
/// derivative of `GroupNorm` given the upstream gradient `dy`.
fn group_norm_grad_input(g: &Groups, dy: &[f64], x: &[f64], w: &[f64]) -> Vec<f64> {
    let stats = g.stats(x);
    let size = g.group_len();
    let mut dx = vec![0.0; x.len()];
    for (block, &(mean, inv_std)) in stats.iter().enumerate() {
        let range = block * size..(block + 1) * size;
        let (mut s1, mut s2) = (0.0, 0.0);
        for i in range.clone() {
            let wdy = w[g.channel(i)] * dy[i];
            s1 += wdy;
            s2 += wdy * (x[i] - mean) * inv_std;
        }
        let (m1, m2) = (s1 / size as f64, s2 / size as f64);
        for i in range {
            let xhat = (x[i] - mean) * inv_std;
            dx[i] = inv_std * (w[g.channel(i)] * dy[i] - m1 - xhat * m2);
        }
    }
    dx
}

/// `[Σ dy·x̂ per channel, Σ dy per channel]`.
fn group_norm_grad_weight_bias(g: &Groups, dy: &[f64], x: &[f64]) -> Vec<f64> {
    let stats = g.stats(x);
    let mut out = vec![0.0; 2 * g.channels];
    for i in 0..x.len() {
        let (mean, inv_std) = stats[i / g.group_len()];
        let c = g.channel(i);
        out[c] += dy[i] * (x[i] - mean) * inv_std;
        out[g.channels + c] += dy[i];
    }
    out
}

pub(super) fn eval(node: &Node, ins: &[&Tensor]) -> Result<Vec<f64>, Error> {
    let arg = |i: usize| -> Result<&Tensor, Error> {
        ins.get(i)
            .copied()
            .ok_or_else(|| invalid(node, format!("missing input {i}")))
    };
    let out = match node.op {
        Op::Conv2d { .. } | Op::Conv2dDw { .. } => {
            convolve(node, conv_window(node)?, arg(0)?, arg(1)?)?
        }
        // Inputs are `[input, kernel]`; F(2,3) is an evaluation strategy.
        // The adjoint form reads the kernel rotated and channel-transposed.
        Op::WinogradConv2d {
            in_channels,
            out_channels,
            adjoint,
            ..
        } => {
            let (x, w) = (arg(0)?, arg(1)?);
            if adjoint {
                let (ci, co) = (in_channels as usize, out_channels as usize);
                expect_len(node, "kernel", w, 9 * ci * co)?;
                let mut rotated = vec![0.0; w.len()];
                for o in 0..co {
                    for i in 0..ci {
                        for tap in 0..9 {
                            rotated[(o * ci + i) * 9 + tap] = w.data[(i * co + o) * 9 + 8 - tap];
                        }
                    }
                }
                let rotated = Tensor::new(vec![rotated.len()], rotated);
                convolve(node, conv_window(node)?, x, &rotated)?
            } else {
                convolve(node, conv_window(node)?, x, w)?
            }
        }
        // The adjoint of Conv2d in its input: inputs `[dy, kernel]`.
        Op::Conv2dGradInput { .. } => {
            let (dy, w) = (arg(0)?, arg(1)?);
            let g = conv_window(node)?.batch_from_output(node, dy.len())?;
            expect_len(node, "kernel", w, g.kernel_len())?;
            let mut dx = vec![0.0; g.input_len()];
            g.for_each_tap(|yi, xi, ki| dx[xi] += dy.data[yi] * w.data[ki]);
            dx
        }
        // The adjoint of Conv2d in its kernel: inputs `[dy, input]`.
        Op::Conv2dGradWeight { .. } => {
            let (dy, x) = (arg(0)?, arg(1)?);
            let g = conv_window(node)?.batch_from_output(node, dy.len())?;
            expect_len(node, "input", x, g.input_len())?;
            let mut dw = vec![0.0; g.kernel_len()];
            g.for_each_tap(|yi, xi, ki| dw[ki] += dy.data[yi] * x.data[xi]);
            dw
        }
        Op::MaxPool2d { .. } => {
            let x = arg(0)?;
            let g = conv_window(node)?.batch_from_input(node, x.len())?;
            g.winners(node, x)?.iter().map(|&i| x.data[i]).collect()
        }
        // Inputs `[dy, input]`.
        Op::MaxPool2dGrad { .. } => {
            let (dy, x) = (arg(0)?, arg(1)?);
            let g = conv_window(node)?.batch_from_input(node, x.len())?;
            expect_len(node, "gradient", dy, g.output_len())?;
            max_pool_grad(node, g, &dy.data, x)?
        }
        Op::GroupNorm { .. } | Op::GroupNormSilu { .. } => {
            let (x, w, b) = (arg(0)?, arg(1)?, arg(2)?);
            let g = Groups::new(node, x.len())?;
            expect_len(node, "weight", w, g.channels)?;
            expect_len(node, "bias", b, g.channels)?;
            let y = group_norm(&g, &x.data, &w.data, &b.data);
            if matches!(node.op, Op::GroupNormSilu { .. }) {
                y.into_iter().map(|v| v * sigmoid(v)).collect()
            } else {
                y
            }
        }
        // Inputs `[dy, input, weight]`.
        Op::GroupNormGradInput { .. } => {
            let (dy, x, w) = (arg(0)?, arg(1)?, arg(2)?);
            let g = Groups::new(node, x.len())?;
            expect_len(node, "gradient", dy, g.len())?;
            expect_len(node, "weight", w, g.channels)?;
            group_norm_grad_input(&g, &dy.data, &x.data, &w.data)
        }
        // Inputs `[dy, input]`; output `[grad_weight; C] ++ [grad_bias; C]`.
        Op::GroupNormGradWeightBias { .. } => {
            let (dy, x) = (arg(0)?, arg(1)?);
            let g = Groups::new(node, x.len())?;
            expect_len(node, "gradient", dy, g.len())?;
            group_norm_grad_weight_bias(&g, &dy.data, &x.data)
        }
        _ => unreachable!("{:?} is not a vision op", node.op),
    };
    Ok(out)
}

/// Per-element error scale of `x̂ = (x − mean) · inv_std`: subtracting the
/// mean rounds at the scale of `|x| + |mean|`, not of the difference.
fn centred_scale(g: &Groups, x: &[f64], stats: &[(f64, f64)]) -> Vec<f64> {
    (0..x.len())
        .map(|i| {
            let (mean, inv_std) = stats[i / g.group_len()];
            (x[i].abs() + mean.abs()) * inv_std
        })
        .collect()
}

pub(super) fn magnitude(node: &Node, ins: &[&Tensor], out: &Tensor) -> Option<Vec<f64>> {
    match node.op {
        // A sum of the routed gradients.
        Op::MaxPool2dGrad { .. } => {
            let g = conv_window(node)
                .ok()?
                .batch_from_input(node, ins[1].len())
                .ok()?;
            let abs: Vec<f64> = ins[0].data.iter().map(|v| v.abs()).collect();
            max_pool_grad(node, g, &abs, ins[1]).ok()
        }
        Op::GroupNorm { .. } | Op::GroupNormSilu { .. } => {
            let (x, w, b) = (&ins[0].data, &ins[1].data, &ins[2].data);
            let g = Groups::new(node, x.len()).ok()?;
            let stats = g.stats(x);
            let scale = centred_scale(&g, x, &stats);
            let pre = group_norm(&g, x, w, b);
            Some(
                (0..x.len())
                    .map(|i| {
                        let c = g.channel(i);
                        let m = scale[i] * w[c].abs() + b[c].abs();
                        let slope = if matches!(node.op, Op::GroupNormSilu { .. }) {
                            let s = sigmoid(pre[i]);
                            (s + pre[i] * s * (1.0 - s)).abs()
                        } else {
                            1.0
                        };
                        m * slope + out.data[i].abs()
                    })
                    .collect(),
            )
        }
        // inv_std · (|w·dy| + mean|w·dy| + a · mean(|w·dy| · a)) with `a`
        // the scale of x̂: the three terms cancel.
        Op::GroupNormGradInput { .. } => {
            let (dy, x, w) = (&ins[0].data, &ins[1].data, &ins[2].data);
            let g = Groups::new(node, x.len()).ok()?;
            let stats = g.stats(x);
            let a = centred_scale(&g, x, &stats);
            let size = g.group_len();
            let mut m = vec![0.0; x.len()];
            for (block, &(_, inv_std)) in stats.iter().enumerate() {
                let range = block * size..(block + 1) * size;
                let (mut s1, mut s2) = (0.0, 0.0);
                for i in range.clone() {
                    let wdy = (w[g.channel(i)] * dy[i]).abs();
                    s1 += wdy;
                    s2 += wdy * a[i];
                }
                let (m1, m2) = (s1 / size as f64, s2 / size as f64);
                for i in range {
                    m[i] = inv_std * ((w[g.channel(i)] * dy[i]).abs() + m1 + a[i] * m2);
                }
            }
            Some(m)
        }
        // Σ |dy| · a and Σ |dy| per channel.
        Op::GroupNormGradWeightBias { .. } => {
            let (dy, x) = (&ins[0].data, &ins[1].data);
            let g = Groups::new(node, x.len()).ok()?;
            let stats = g.stats(x);
            let a = centred_scale(&g, x, &stats);
            let mut m = vec![0.0; 2 * g.channels];
            for i in 0..x.len() {
                let c = g.channel(i);
                m[c] += dy[i].abs() * a[i];
                m[g.channels + c] += dy[i].abs();
            }
            Some(m)
        }
        _ => None,
    }
}
