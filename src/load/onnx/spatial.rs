//! Convolutions, transposed convolutions, `Resize` and `GridSample`: the
//! importer's spatial operators, lowered onto the graph's NCHW builders.

use std::collections::HashMap;

use crate::{
    graph::{Graph, Nchw, NodeId},
    load::onnx::{
        OnnxError, constant_values, get_shape, narrow, proto, register_output, resolve_input, view,
        window,
    },
};

/// Per-axis values of a 1D or 2D operator as the 2D ones the kernels take:
/// a 1D one is a row high.
fn plane<T: Copy>(spatial: &[T], fill: T) -> [T; 2] {
    match *spatial {
        [w] => [fill, w],
        [h, w] => [h, w],
        _ => [fill; 2],
    }
}

fn unsupported(node: &proto::Node, what: impl std::fmt::Display) -> OnnxError {
    OnnxError::UnsupportedOp(format!("{} '{}': {what}", node.op_type, node.name))
}

/// The `group` attribute, checked against the channels it divides.
fn groups(node: &proto::Node, channels: &[usize]) -> Result<usize, OnnxError> {
    let groups = node.attrs.i("group", 1);
    if groups < 1 || channels.iter().any(|&c| c % groups as usize != 0) {
        return Err(OnnxError::ShapeError(format!(
            "node '{}': {groups} groups of {channels:?} channels",
            node.name
        )));
    }
    Ok(groups as usize)
}

/// Apply `conv` to each of `groups` channel ranges of a `[batch, channels,
/// spatial]` input, and join the results, `out_channels` of `out_spatial`
/// elements in all, along channels. `conv` gets the group's input and index.
fn per_group(
    graph: &mut Graph,
    x: NodeId,
    [batch, channels, spatial]: [usize; 3],
    groups: usize,
    [out_channels, out_spatial]: [usize; 2],
    mut conv: impl FnMut(&mut Graph, NodeId, usize) -> NodeId,
) -> NodeId {
    let (step, out_step) = (channels / groups, out_channels / groups);
    let mut parts: Vec<(NodeId, usize)> = (0..groups)
        .map(|g| {
            let part = narrow(
                graph,
                x,
                &[batch, channels, spatial],
                1,
                g * step..(g + 1) * step,
            );
            (conv(graph, part, g), out_step)
        })
        .collect();
    // Pairwise, so each level copies the output once.
    while parts.len() > 1 {
        parts = parts
            .chunks(2)
            .map(|pair| match *pair {
                [(a, a_len), (b, b_len)] => {
                    let joined = graph.concat(
                        a,
                        b,
                        batch as u32,
                        a_len as u32,
                        b_len as u32,
                        out_spatial as u32,
                    );
                    (joined, a_len + b_len)
                }
                _ => pair[0],
            })
            .collect();
    }
    parts[0].0
}

/// `Conv` in 1D or 2D. A group of one input channel runs as a depthwise
/// convolution; other groups run one convolution each.
pub(super) fn conv(
    graph: &mut Graph,
    node: &proto::Node,
    name_to_id: &mut HashMap<String, NodeId>,
    shapes: &mut HashMap<String, Vec<usize>>,
) -> Result<(), OnnxError> {
    let input = resolve_input(&node.inputs[0], name_to_id, &node.name)?;
    let kernel = resolve_input(&node.inputs[1], name_to_id, &node.name)?;
    let input_shape = get_shape(&node.inputs[0], shapes);
    let kernel_shape = get_shape(&node.inputs[1], shapes);
    let rank = input_shape.len();
    if !(3..=4).contains(&rank) || kernel_shape.len() != rank {
        return Err(unsupported(
            node,
            format!("{input_shape:?} by {kernel_shape:?} (supported: 1D and 2D)"),
        ));
    }
    let (batch, in_channels, out_channels) = (input_shape[0], input_shape[1], kernel_shape[0]);
    let groups = groups(node, &[in_channels, out_channels])?;
    let group_channels = in_channels / groups;
    if kernel_shape[1] != group_channels {
        return Err(OnnxError::ShapeError(format!(
            "node '{}': kernel {kernel_shape:?} for {in_channels} channels in {groups} groups",
            node.name
        )));
    }
    let (stride, pads) = window(node, &input_shape[2..], &kernel_shape[2..])?;
    let [in_h, in_w] = plane(&input_shape[2..], 1);
    let [kernel_h, kernel_w] = plane(&kernel_shape[2..], 1);
    let [padding_h, padding_w] = plane(&pads, 0);
    let out_h = (in_h + 2 * padding_h - kernel_h) / stride + 1;
    let out_w = (in_w + 2 * padding_w - kernel_w) / stride + 1;
    let taps = kernel_h * kernel_w;
    // The kernels read both operands as flat NCHW.
    let input = view(graph, input, &[batch * in_channels * in_h * in_w])?;
    let kernel = view(graph, kernel, &[out_channels * group_channels * taps])?;
    let convolve = |graph: &mut Graph, x, k, channels, out_channels| {
        graph.conv2d_hw(
            x,
            k,
            batch as u32,
            channels as u32,
            in_h as u32,
            in_w as u32,
            out_channels as u32,
            kernel_h as u32,
            kernel_w as u32,
            stride as u32,
            padding_h as u32,
            padding_w as u32,
        )
    };
    // The depthwise kernel stages up to 7x7 taps and puts every plane on the
    // grid's Z axis.
    let depthwise = groups == in_channels && taps <= 49 && batch * out_channels <= 65535;
    let out = if groups == 1 {
        convolve(graph, input, kernel, in_channels, out_channels)
    } else if depthwise {
        // Output channel `o` reads input channel `o / multiplier`: repeat
        // each input plane that many times.
        let multiplier = out_channels / in_channels;
        let input = if multiplier == 1 {
            input
        } else {
            let planes = view(graph, input, &[batch * in_channels, 1, in_h * in_w])?;
            let repeated =
                graph.broadcast_to(planes, &[batch * in_channels, multiplier, in_h * in_w]);
            view(graph, repeated, &[batch * out_channels * in_h * in_w])?
        };
        graph.conv2d_dw(
            input,
            kernel,
            batch as u32,
            out_channels as u32,
            in_h as u32,
            in_w as u32,
            kernel_h as u32,
            kernel_w as u32,
            stride as u32,
            padding_h as u32,
            padding_w as u32,
        )
    } else {
        let out_step = out_channels / groups;
        let kernel_rows = [out_channels, group_channels * taps];
        per_group(
            graph,
            input,
            [batch, in_channels, in_h * in_w],
            groups,
            [out_channels, out_h * out_w],
            |graph, x, g| {
                let k = narrow(
                    graph,
                    kernel,
                    &kernel_rows,
                    0,
                    g * out_step..(g + 1) * out_step,
                );
                convolve(graph, x, k, group_channels, out_step)
            },
        )
    };
    let out = add_bias(graph, node, out, out_channels, out_h * out_w, name_to_id)?;
    let mut out_shape = vec![batch, out_channels];
    out_shape.extend_from_slice(&[out_h, out_w][4 - rank..]);
    register_output(node, 0, out, &out_shape, name_to_id, shapes);
    Ok(())
}

/// The optional third input, a per-channel bias, added to `out`.
fn add_bias(
    graph: &mut Graph,
    node: &proto::Node,
    out: NodeId,
    channels: usize,
    spatial: usize,
    name_to_id: &HashMap<String, NodeId>,
) -> Result<NodeId, OnnxError> {
    let Some(name) = node.inputs.get(2).filter(|name| !name.is_empty()) else {
        return Ok(out);
    };
    let bias = resolve_input(name, name_to_id, &node.name)?;
    let bias = view(graph, bias, &[channels])?;
    Ok(graph.add_per_channel(out, bias, channels as u32, spatial as u32))
}

/// `ConvTranspose` in 1D or 2D, with per-axis strides, padding that may
/// differ between the sides, `output_padding`, `output_shape`, `auto_pad`
/// and groups. Each group runs the transposed convolution with as much of
/// the padding as both sides share; the rest is cropped, or past the end of
/// the full output, filled with zeros.
pub(super) fn conv_transpose(
    graph: &mut Graph,
    node: &proto::Node,
    name_to_id: &mut HashMap<String, NodeId>,
    shapes: &mut HashMap<String, Vec<usize>>,
) -> Result<(), OnnxError> {
    let attrs = &node.attrs;
    let input = resolve_input(&node.inputs[0], name_to_id, &node.name)?;
    let kernel = resolve_input(&node.inputs[1], name_to_id, &node.name)?;
    let input_shape = get_shape(&node.inputs[0], shapes);
    let kernel_shape = get_shape(&node.inputs[1], shapes);
    let rank = input_shape.len();
    if !(3..=4).contains(&rank) || kernel_shape.len() != rank {
        return Err(unsupported(
            node,
            format!("{input_shape:?} by {kernel_shape:?} (supported: 1D and 2D)"),
        ));
    }
    let (batch, in_channels) = (input_shape[0], input_shape[1]);
    if kernel_shape[0] != in_channels {
        return Err(OnnxError::ShapeError(format!(
            "node '{}': kernel {kernel_shape:?} for {in_channels} channels",
            node.name
        )));
    }
    let groups = groups(node, &[in_channels])?;
    let group_channels = in_channels / groups;
    let out_step = kernel_shape[1];
    let out_channels = out_step * groups;
    let spatial = rank - 2;
    let per_axis = |name: &str, default: i64| -> Result<Vec<i64>, OnnxError> {
        match *attrs.ints(name) {
            [] => Ok(vec![default; spatial]),
            ref values if values.len() == spatial => Ok(values.to_vec()),
            ref values => Err(unsupported(node, format!("{name} {values:?}"))),
        }
    };
    if per_axis("dilations", 1)?.iter().any(|&d| d != 1) {
        return Err(unsupported(node, "dilation (supported: none)"));
    }
    let strides = per_axis("strides", 1)?;
    let extra = per_axis("output_padding", 0)?;
    let extents = &input_shape[2..];
    let kernels = &kernel_shape[2..];
    // Every extent the kernels see fits a `u32`, which keeps the arithmetic
    // below in range.
    let limit = i64::from(u32::MAX);
    let fits = |value: i64, low: i64| (low..=limit).contains(&value);
    if extents.iter().chain(kernels).any(|&e| !fits(e as i64, 1))
        || strides.iter().any(|&s| !fits(s, 1))
        || extra.iter().zip(&strides).any(|(&e, &s)| e < 0 || e >= s)
    {
        return Err(unsupported(
            node,
            format!(
                "{input_shape:?} by {kernel_shape:?}, strides {strides:?} with output_padding {extra:?}"
            ),
        ));
    }
    // The transposed convolution's whole reach, before any padding.
    let full: Vec<i64> = (0..spatial)
        .map(|d| {
            strides[d]
                .checked_mul(extents[d] as i64 - 1)
                .and_then(|reach| reach.checked_add(kernels[d] as i64))
                .filter(|&reach| reach <= limit)
        })
        .collect::<Option<_>>()
        .ok_or_else(|| unsupported(node, format!("{input_shape:?} at strides {strides:?}")))?;
    let auto_pad = attrs.s("auto_pad").unwrap_or("NOTSET");
    let output_shape = attrs.ints("output_shape");
    let (begin, end): (Vec<i64>, Vec<i64>) =
        if output_shape.is_empty() && matches!(auto_pad, "NOTSET" | "VALID") {
            let pads = match *attrs.ints("pads") {
                [] => vec![0; 2 * spatial],
                ref pads
                    if pads.len() == 2 * spatial
                        && auto_pad == "NOTSET"
                        && pads.iter().all(|&p| fits(p, 0)) =>
                {
                    pads.to_vec()
                }
                ref pads => return Err(unsupported(node, format!("pads {pads:?}"))),
            };
            (pads[..spatial].to_vec(), pads[spatial..].to_vec())
        } else {
            // An output shape, given or `SAME`'s input times stride, sets the
            // padding; an odd total puts the extra one first, except for
            // `SAME_UPPER`.
            let target: Vec<i64> = match *output_shape {
                [] => (0..spatial)
                    .map(|d| (extents[d] as i64).saturating_mul(strides[d]))
                    .collect(),
                ref shape if shape.len() >= spatial => shape[shape.len() - spatial..].to_vec(),
                ref shape => return Err(unsupported(node, format!("output_shape {shape:?}"))),
            };
            if target.iter().any(|&t| !fits(t, 1)) {
                return Err(unsupported(node, format!("output shape {target:?}")));
            }
            (0..spatial)
                .map(|d| {
                    let total = full[d] + extra[d] - target[d];
                    match auto_pad {
                        "SAME_UPPER" => (total / 2, total - total / 2),
                        _ => (total - total / 2, total / 2),
                    }
                })
                .unzip()
        };
    let out_extents: Vec<i64> = (0..spatial)
        .map(|d| full[d] + extra[d] - begin[d] - end[d])
        .collect();
    if begin.iter().chain(&end).any(|&p| p < 0) || out_extents.iter().any(|&e| e < 1) {
        return Err(OnnxError::ShapeError(format!(
            "node '{}': padding {begin:?} and {end:?} of {full:?}",
            node.name
        )));
    }
    // Run with the padding both ends share, net of `output_padding` at the
    // end; then crop what remains, or extend past the full output.
    let shared: Vec<i64> = (0..spatial)
        .map(|d| begin[d].min((end[d] - extra[d]).max(0)))
        .collect();
    let [in_h, in_w] = plane(extents, 1);
    let [kernel_h, kernel_w] = plane(kernels, 1);
    let [stride_h, stride_w] = plane(&strides, 1);
    let [shared_h, shared_w] = plane(&shared, 0);
    let [begin_h, begin_w] = plane(&begin, 0);
    let [end_h, end_w] = plane(&end, 0);
    let [extra_h, extra_w] = plane(&extra, 0);
    let [full_h, full_w] = plane(&full, 1);
    let [out_h, out_w] = plane(&out_extents, 1);
    let (run_h, run_w) = (full_h - 2 * shared_h, full_w - 2 * shared_w);
    let input = view(graph, input, &[batch * in_channels * in_h * in_w])?;
    let kernel = view(
        graph,
        kernel,
        &[in_channels * out_step * kernel_h * kernel_w],
    )?;
    let kernel_rows = [in_channels, out_step * kernel_h * kernel_w];
    let out = per_group(
        graph,
        input,
        [batch, in_channels, in_h * in_w],
        groups,
        [out_channels, (run_h * run_w) as usize],
        |graph, x, g| {
            let k = if groups == 1 {
                kernel
            } else {
                narrow(
                    graph,
                    kernel,
                    &kernel_rows,
                    0,
                    g * group_channels..(g + 1) * group_channels,
                )
            };
            let dims = Nchw {
                batch: batch as u32,
                channels: group_channels as u32,
                h: in_h as u32,
                w: in_w as u32,
            };
            graph.conv_transpose_2d(
                x,
                k,
                dims,
                out_step as u32,
                [kernel_h as u32, kernel_w as u32],
                [stride_h as u32, stride_w as u32],
                [shared_h as u32, shared_w as u32],
            )
        },
    );
    let dims = Nchw {
        batch: batch as u32,
        channels: out_channels as u32,
        h: run_h as u32,
        w: run_w as u32,
    };
    let out = trim(
        graph,
        out,
        dims,
        [begin_h - shared_h, end_h - extra_h - shared_h],
        [begin_w - shared_w, end_w - extra_w - shared_w],
    );
    let out = add_bias(
        graph,
        node,
        out,
        out_channels,
        (out_h * out_w) as usize,
        name_to_id,
    )?;
    let mut out_shape = vec![batch, out_channels];
    out_shape.extend(out_extents.iter().map(|&e| e as usize));
    register_output(node, 0, out, &out_shape, name_to_id, shapes);
    Ok(())
}

/// Crop `[top, bottom]` rows and `[left, right]` columns from every plane;
/// a negative bottom or right count appends that many zeros instead.
fn trim(graph: &mut Graph, x: NodeId, dims: Nchw, rows: [i64; 2], columns: [i64; 2]) -> NodeId {
    let crop = |n: i64| n.max(0) as u32;
    let (top, bottom, left, right) = (
        crop(rows[0]),
        crop(rows[1]),
        crop(columns[0]),
        crop(columns[1]),
    );
    let mut out = x;
    let mut dims = dims;
    if top + bottom + left + right > 0 {
        out = graph.crop_2d(out, dims, [top, bottom], [left, right]);
        dims.h -= top + bottom;
        dims.w -= left + right;
    }
    let grow = |n: i64| (-n).max(0) as u32;
    let planes = dims.batch * dims.channels;
    let (below, beside) = (grow(rows[1]), grow(columns[1]));
    if below > 0 {
        let zeros = zeros(graph, planes * below * dims.w);
        out = graph.concat(out, zeros, planes, dims.h * dims.w, below * dims.w, 1);
        dims.h += below;
    }
    if beside > 0 {
        let lines = planes * dims.h;
        let zeros = zeros(graph, lines * beside);
        out = graph.concat(out, zeros, lines, dims.w, beside, 1);
    }
    out
}

fn zeros(graph: &mut Graph, len: u32) -> NodeId {
    let zeros = graph.zeros(1, len as usize);
    graph.reshape(zeros, &[len as usize])
}

/// How `Resize` weighs the input elements around each output position.
struct Interpolation<'a> {
    mode: &'a str,
    transform: &'a str,
    nearest: &'a str,
    cubic_a: f64,
    exclude_outside: bool,
    antialias: bool,
}

impl Interpolation<'_> {
    /// The weights of consecutive input elements for a position `ratio`
    /// past the one before it, in `(0, 1]`, as ONNX's reference computes them.
    fn coefficients(&self, ratio: f64, scale: f64) -> Vec<f64> {
        let a = self.cubic_a;
        let cubic = |x: f64| {
            let x = x.abs();
            if x <= 1.0 {
                ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
            } else if x < 2.0 {
                ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a
            } else {
                0.0
            }
        };
        let normalize = |c: Vec<f64>| {
            let sum: f64 = c.iter().sum();
            c.into_iter().map(|v| v / sum).collect()
        };
        match self.mode {
            "nearest" => {
                // A whole `ratio` is 1: the right element is the position.
                let right = ratio == 1.0
                    || match self.nearest {
                        "round_prefer_floor" => ratio > 0.5,
                        "round_prefer_ceil" => ratio >= 0.5,
                        "floor" => false,
                        _ => true,
                    };
                if right {
                    vec![0.0, 1.0]
                } else {
                    vec![1.0, 0.0]
                }
            }
            "linear" if self.antialias => {
                let scale = scale.min(1.0);
                let start = (-1.0 / scale).floor() as i64 + 1;
                let taps = (start..start + 2 - 2 * start)
                    .map(|i| (1.0 - ((i as f64 - ratio) * scale).abs()).clamp(0.0, 1.0))
                    .collect();
                normalize(taps)
            }
            "linear" => vec![1.0 - ratio, ratio],
            _ if self.antialias => {
                let scale = scale.min(1.0);
                let start = (-2.0 / scale).floor() as i64 + 1;
                let taps = (start..2 - start)
                    .map(|i| cubic(scale * (i as f64 - ratio)))
                    .collect();
                normalize(taps)
            }
            _ => [ratio + 1.0, ratio, 1.0 - ratio, 2.0 - ratio]
                .into_iter()
                .map(cubic)
                .collect(),
        }
    }

    /// The row-major `[input, out]` matrix that resizes one axis by a matrix
    /// product: each output element is a fixed combination of input
    /// elements, whatever their values. Edge elements repeat past the ends.
    fn matrix(&self, input: usize, out: usize, scale: f64) -> Vec<f32> {
        let (width, output_width) = (input as f64, scale * input as f64);
        let mut matrix = vec![0.0f64; out * input];
        for y in 0..out {
            let y_f = y as f64;
            let x = match self.transform {
                "align_corners" if output_width == 1.0 => 0.0,
                "align_corners" => y_f * (width - 1.0) / (output_width - 1.0),
                "asymmetric" => y_f / scale,
                "pytorch_half_pixel" if output_width == 1.0 => -0.5,
                "half_pixel_symmetric" => {
                    let offset = width / 2.0 * (1.0 - out as f64 / output_width);
                    offset + (y_f + 0.5) / scale - 0.5
                }
                _ => (y_f + 0.5) / scale - 0.5,
            };
            let ratio = if x == x.floor() { 1.0 } else { x - x.floor() };
            let mut coefficients = self.coefficients(ratio, scale);
            let taps = coefficients.len() as i64;
            // The window ONNX's reference picks: the `taps` nearest elements,
            // the lower on a tie.
            let pad = (taps + 1) / 2;
            let padded = x + pad as f64;
            let frac = padded - padded.floor();
            let offset = if taps % 2 == 0 {
                -(taps / 2) + i64::from(frac != 0.0)
            } else {
                -((taps - 1) / 2) + i64::from(frac > 0.5)
            };
            let first = padded.floor() as i64 + offset - pad;
            if self.exclude_outside {
                for (t, c) in coefficients.iter_mut().enumerate() {
                    if !(0..input as i64).contains(&(first + t as i64)) {
                        *c = 0.0;
                    }
                }
                let sum: f64 = coefficients.iter().sum();
                if sum != 0.0 {
                    coefficients.iter_mut().for_each(|c| *c /= sum);
                }
            }
            for (t, c) in coefficients.into_iter().enumerate() {
                let index = (first + t as i64).clamp(0, input as i64 - 1) as usize;
                matrix[index * out + y] += c;
            }
        }
        matrix.into_iter().map(|v| v as f32).collect()
    }
}

/// The whole factor by which an `[input, out]` resize matrix repeats each
/// input element, if it does nothing else.
fn repeat_factor(matrix: &[f32], input: usize, out: usize) -> Option<usize> {
    if !out.is_multiple_of(input) {
        return None;
    }
    let factor = out / input;
    let weight = |j: usize, y: usize| if j == y / factor { 1.0 } else { 0.0 };
    (0..input)
        .all(|j| (0..out).all(|y| matrix[j * out + y] == weight(j, y)))
        .then_some(factor)
}

/// `Resize` over the spatial axes of NCHW or NCW, from opset 11. Each
/// resized axis is a constant interpolation matrix applied by a matrix
/// product; one that only repeats elements whole times repeats them.
pub(super) fn resize(
    graph: &mut Graph,
    node: &proto::Node,
    name_to_id: &mut HashMap<String, NodeId>,
    shapes: &mut HashMap<String, Vec<usize>>,
    weights: &HashMap<String, Vec<f32>>,
) -> Result<(), OnnxError> {
    let attrs = &node.attrs;
    let x = resolve_input(&node.inputs[0], name_to_id, &node.name)?;
    let x_shape = get_shape(&node.inputs[0], shapes);
    let rank = x_shape.len();
    if !(3..=4).contains(&rank) || x_shape.contains(&0) {
        return Err(unsupported(
            node,
            format!("{x_shape:?} (supported: NCW and NCHW)"),
        ));
    }
    if node.inputs.len() < 3 {
        return Err(unsupported(node, "scales without roi (opset 10)"));
    }
    let constant = |index: usize| -> Result<Vec<f32>, OnnxError> {
        let Some(name) = node.inputs.get(index).filter(|name| !name.is_empty()) else {
            return Ok(Vec::new());
        };
        let id = resolve_input(name, name_to_id, &node.name)?;
        constant_values(graph, id, weights)
            .map(<[f32]>::to_vec)
            .ok_or_else(|| unsupported(node, "scales and sizes must be constants"))
    };
    let (scales, sizes) = (constant(2)?, constant(3)?);
    let axes: Vec<usize> = match *attrs.ints("axes") {
        [] => (0..rank).collect(),
        ref axes => axes
            .iter()
            .map(|&a| if a < 0 { a + rank as i64 } else { a } as usize)
            .collect(),
    };
    let given = if sizes.is_empty() { &scales } else { &sizes };
    if given.len() != axes.len() || axes.iter().any(|&a| a >= rank) {
        return Err(OnnxError::ShapeError(format!(
            "node '{}': {} values for axes {axes:?} of {x_shape:?}",
            node.name,
            given.len()
        )));
    }
    // Per axis: the scale the coordinates use and the output extent.
    let mut scale: Vec<f64> = vec![1.0; rank];
    let mut out_shape = x_shape.clone();
    if sizes.is_empty() {
        for (&axis, &s) in axes.iter().zip(&scales) {
            scale[axis] = f64::from(s);
            out_shape[axis] = (f64::from(s) * x_shape[axis] as f64) as usize;
        }
    } else {
        let ratios: Vec<f64> = axes
            .iter()
            .zip(&sizes)
            .map(|(&axis, &size)| f64::from(size) / x_shape[axis] as f64)
            .collect();
        let policy = attrs.s("keep_aspect_ratio_policy").unwrap_or("stretch");
        let fixed = match policy {
            "stretch" => None,
            "not_larger" => Some(ratios.iter().copied().fold(f64::INFINITY, f64::min)),
            "not_smaller" => Some(ratios.iter().copied().fold(0.0, f64::max)),
            other => {
                return Err(unsupported(
                    node,
                    format!("keep_aspect_ratio_policy {other}"),
                ));
            }
        };
        for (i, &axis) in axes.iter().enumerate() {
            scale[axis] = fixed.unwrap_or(ratios[i]);
            out_shape[axis] = match fixed {
                Some(s) => (s * x_shape[axis] as f64 + 0.5) as usize,
                None => sizes[i] as usize,
            };
        }
    }
    let identity = |axis: usize| {
        (scale[axis] - 1.0).abs() <= 1e-9 * scale[axis].abs().max(1.0)
            && out_shape[axis] == x_shape[axis]
    };
    // An axis's matrix is `out × in` constants.
    let oversized = (2..rank).any(|d| out_shape[d].saturating_mul(x_shape[d]) > 1 << 24);
    if !identity(0) || !identity(1) || out_shape.contains(&0) || oversized {
        return Err(unsupported(
            node,
            format!("{x_shape:?} to {out_shape:?} (supported: spatial axes)"),
        ));
    }
    let how = Interpolation {
        mode: attrs.s("mode").unwrap_or("nearest"),
        transform: attrs
            .s("coordinate_transformation_mode")
            .unwrap_or("half_pixel"),
        nearest: attrs.s("nearest_mode").unwrap_or("round_prefer_floor"),
        cubic_a: f64::from(attrs.f("cubic_coeff_a", -0.75)),
        exclude_outside: attrs.i("exclude_outside", 0) != 0,
        antialias: attrs.i("antialias", 0) != 0,
    };
    let known = matches!(how.mode, "nearest" | "linear" | "cubic")
        && matches!(
            how.transform,
            "half_pixel"
                | "half_pixel_symmetric"
                | "pytorch_half_pixel"
                | "align_corners"
                | "asymmetric"
        )
        && matches!(
            how.nearest,
            "round_prefer_floor" | "round_prefer_ceil" | "floor" | "ceil"
        )
        && !(how.antialias && how.mode == "nearest");
    if !known {
        return Err(unsupported(
            node,
            format!(
                "mode {}, {} coordinates, nearest_mode {}, antialias {}",
                how.mode, how.transform, how.nearest, how.antialias
            ),
        ));
    }
    let planes = x_shape[0] * x_shape[1];
    let [in_h, in_w] = plane(&x_shape[2..], 1);
    let [out_h, out_w] = plane(&out_shape[2..], 1);
    let resized =
        |axis: usize, input, out| (!identity(axis)).then(|| how.matrix(input, out, scale[axis]));
    let rows = if rank == 4 {
        resized(2, in_h, out_h)
    } else {
        None
    };
    let columns = resized(rank - 1, in_w, out_w);
    let factor = |matrix: &Option<Vec<f32>>, input, out| match *matrix {
        None => Some(1),
        Some(ref m) => repeat_factor(m, input, out),
    };
    let mut out = x;
    if let (Some(fh), Some(fw)) = (factor(&rows, in_h, out_h), factor(&columns, in_w, out_w)) {
        if fh * fw > 1 {
            let dims = Nchw {
                batch: x_shape[0] as u32,
                channels: x_shape[1] as u32,
                h: in_h as u32,
                w: in_w as u32,
            };
            out = graph.upsample_nearest(out, dims, fh as u32, fw as u32);
        }
    } else {
        if let Some(m) = columns {
            let x = view(graph, out, &[planes * in_h, in_w])?;
            let m = graph.constant(m, &[in_w, out_w]);
            out = graph.matmul(x, m);
        }
        if let Some(m) = rows {
            // Bring the rows to the inner axis, resize them, and turn back.
            let x = view(graph, out, &[planes, in_h, out_w])?;
            let x = graph.permute(x, &[0, 2, 1]);
            let x = view(graph, x, &[planes * out_w, in_h])?;
            let m = graph.constant(m, &[in_h, out_h]);
            let x = graph.matmul(x, m);
            let x = view(graph, x, &[planes, out_w, out_h])?;
            out = graph.permute(x, &[0, 2, 1]);
        }
    }
    register_output(node, 0, out, &out_shape, name_to_id, shapes);
    Ok(())
}

/// `floor(x)` for `x >= -shift`, an integer, by the conversion to `U32`,
/// which truncates.
fn floor(graph: &mut Graph, x: NodeId, shift: f32) -> NodeId {
    let shifted = graph.add_scalar(x, shift);
    let whole = graph.to_u32(shifted);
    let whole = graph.to_f32(whole);
    graph.add_scalar(whole, -shift)
}

/// `1 - x`.
fn complement(graph: &mut Graph, x: NodeId) -> NodeId {
    let negated = graph.neg(x);
    graph.add_scalar(negated, 1.0)
}

/// `x` rounded to the nearest integer, a half to the even one as `rint`
/// does, for `x + 0.5 >= -shift`.
fn round_half_even(graph: &mut Graph, x: NodeId, shift: f32) -> NodeId {
    let up = graph.add_scalar(x, 0.5);
    let rounded = floor(graph, up, shift);
    // A half lands exactly on `rounded`; there an odd one steps down. Any
    // other fraction, scaled up, clamps to 1.
    let below = graph.neg(rounded);
    let frac = graph.add(up, below);
    let frac = graph.scale(frac, (1u32 << 25) as f32);
    let frac = graph.clamp(frac, 0.0, 1.0);
    let tie = complement(graph, frac);
    let half = graph.scale(rounded, 0.5);
    let half = floor(graph, half, shift);
    let even = graph.scale(half, -2.0);
    let odd = graph.add(rounded, even);
    let step = graph.mul(tie, odd);
    let step = graph.neg(step);
    graph.add(rounded, step)
}

/// 1 where an integer-valued `x` lies in `[0, size)`, else 0.
fn inside(graph: &mut Graph, x: NodeId, size: usize) -> NodeId {
    let above = graph.add_scalar(x, 1.0);
    let above = graph.clamp(above, 0.0, 1.0);
    let below = graph.neg(x);
    let below = graph.add_scalar(below, size as f32);
    let below = graph.clamp(below, 0.0, 1.0);
    graph.mul(above, below)
}

/// `GridSample` over NCHW, bilinear or nearest, with zero or border
/// padding. Each of the up to four taps of every sample point gathers a row
/// of channels from the image laid out channels-last.
pub(super) fn grid_sample(
    graph: &mut Graph,
    node: &proto::Node,
    name_to_id: &mut HashMap<String, NodeId>,
    shapes: &mut HashMap<String, Vec<usize>>,
) -> Result<(), OnnxError> {
    let attrs = &node.attrs;
    let x = resolve_input(&node.inputs[0], name_to_id, &node.name)?;
    let grid = resolve_input(&node.inputs[1], name_to_id, &node.name)?;
    let x_shape = get_shape(&node.inputs[0], shapes);
    let grid_shape = get_shape(&node.inputs[1], shapes);
    let (&[n, c, h, w], &[grid_n, out_h, out_w, 2]) = (&x_shape[..], &grid_shape[..]) else {
        return Err(unsupported(
            node,
            format!("{x_shape:?} at {grid_shape:?} (supported: 2D)"),
        ));
    };
    let nearest = match attrs.s("mode").unwrap_or("linear") {
        "linear" | "bilinear" => false,
        "nearest" => true,
        other => return Err(unsupported(node, format!("mode {other}"))),
    };
    let border = match attrs.s("padding_mode").unwrap_or("zeros") {
        "zeros" => false,
        "border" => true,
        other => return Err(unsupported(node, format!("padding_mode {other}"))),
    };
    let align_corners = attrs.i("align_corners", 0) != 0;
    // Flat pixel indices are exact in F32 below 2^24; so are sample points.
    let sizes = [n, c, h, w, out_h, out_w];
    let product = |extents: [usize; 3]| extents.iter().fold(1usize, |p, &e| p.saturating_mul(e));
    if grid_n != n
        || sizes.contains(&0)
        || product([n, h, w]) >= 1 << 24
        || product([n, out_h, out_w]) >= 1 << 24
    {
        return Err(OnnxError::ShapeError(format!(
            "node '{}': GridSample of {x_shape:?} at {grid_shape:?}",
            node.name
        )));
    }
    let points = n * out_h * out_w;
    let image = view(graph, x, &[n, c, h * w])?;
    let image = graph.permute(image, &[0, 2, 1]);
    let image = view(graph, image, &[n * h * w, c])?;
    let grid = view(graph, grid, &[points * 2])?;
    // Pixel coordinates of each point along an axis of `size` pixels.
    // Clamping to two past either edge changes no sample: every tap there
    // is outside the image.
    let mut coordinate = |axis: usize, size: usize| -> Result<NodeId, OnnxError> {
        let g = narrow(graph, grid, &[points, 2], 1, axis..axis + 1);
        let g = view(graph, g, &[points, 1])?;
        let (gain, offset) = if align_corners {
            ((size as f32 - 1.0) / 2.0, (size as f32 - 1.0) / 2.0)
        } else {
            (size as f32 / 2.0, (size as f32 - 1.0) / 2.0)
        };
        let g = graph.scale(g, gain);
        let g = graph.add_scalar(g, offset);
        Ok(if border {
            graph.clamp(g, 0.0, size as f32 - 1.0)
        } else {
            graph.clamp(g, -2.0, size as f32 + 1.0)
        })
    };
    let (gx, gy) = (coordinate(0, w)?, coordinate(1, h)?);
    // Taps as (column, row, weight).
    let taps: Vec<(NodeId, NodeId, Option<NodeId>)> = if nearest {
        let shift = 3.0;
        let tap_x = round_half_even(graph, gx, shift);
        let tap_y = round_half_even(graph, gy, shift);
        vec![(tap_x, tap_y, None)]
    } else {
        let mut axis = |g: NodeId| {
            let low = floor(graph, g, 2.0);
            let below = graph.neg(low);
            let high_weight = graph.add(g, below);
            let low_weight = complement(graph, high_weight);
            let high = graph.add_scalar(low, 1.0);
            [(low, low_weight), (high, high_weight)]
        };
        let (columns, rows) = (axis(gx), axis(gy));
        let mut taps = Vec::with_capacity(4);
        for &(row, row_weight) in &rows {
            for &(column, column_weight) in &columns {
                let weight = graph.mul(column_weight, row_weight);
                taps.push((column, row, Some(weight)));
            }
        }
        taps
    };
    // The flat index of the first pixel of each point's image.
    let image_start = (n > 1).then(|| {
        let offsets: Vec<f32> = (0..points)
            .map(|p| (p / (out_h * out_w) * h * w) as f32)
            .collect();
        graph.constant(offsets, &[points, 1])
    });
    let mut out: Option<NodeId> = None;
    for (column, row, weight) in taps {
        let weight = if border {
            weight
        } else {
            let column_inside = inside(graph, column, w);
            let row_inside = inside(graph, row, h);
            let valid = graph.mul(column_inside, row_inside);
            Some(match weight {
                Some(weight) => graph.mul(weight, valid),
                None => valid,
            })
        };
        let column = graph.clamp(column, 0.0, w as f32 - 1.0);
        let row = graph.clamp(row, 0.0, h as f32 - 1.0);
        let row = graph.scale(row, w as f32);
        let mut index = graph.add(row, column);
        if let Some(start) = image_start {
            index = graph.add(index, start);
        }
        let index = graph.to_u32(index);
        let index = view(graph, index, &[points])?;
        let mut sample = graph.embedding(index, image);
        if let Some(weight) = weight {
            let weight = graph.broadcast_inner(weight, c);
            sample = graph.mul(sample, weight);
        }
        out = Some(match out {
            Some(sum) => graph.add(sum, sample),
            None => sample,
        });
    }
    let out = out.expect("a tap");
    let out = view(graph, out, &[n, out_h * out_w, c])?;
    let out = graph.permute(out, &[0, 2, 1]);
    register_output(node, 0, out, &[n, c, out_h, out_w], name_to_id, shapes);
    Ok(())
}
