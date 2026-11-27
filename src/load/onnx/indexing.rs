//! Operators that select, repeat or rearrange elements by position: `Pad`,
//! `Tile`, `GatherElements`, `TopK`, and the gather behind strided `Slice`.

use std::collections::HashMap;

use crate::{
    DType,
    graph::{Graph, NodeId},
    load::onnx::{
        OnnxError, constant_values, get_shape, narrow, proto, register_output, resolve_input,
        scalar_constant, view,
    },
};

/// Element indices are exact in F32 below 2^24.
const EXACT_INDICES: usize = 1 << 24;

fn unsupported(node: &proto::Node, what: impl std::fmt::Display) -> OnnxError {
    OnnxError::UnsupportedOp(format!("{} '{}': {what}", node.op_type, node.name))
}

/// The integer values of the constant input `slot`, if the node has it.
fn constant_ints(
    graph: &Graph,
    node: &proto::Node,
    slot: usize,
    name_to_id: &HashMap<String, NodeId>,
    weights: &HashMap<String, Vec<f32>>,
) -> Result<Option<Vec<i64>>, OnnxError> {
    let Some(name) = node.inputs.get(slot).filter(|name| !name.is_empty()) else {
        return Ok(None);
    };
    let id = resolve_input(name, name_to_id, &node.name)?;
    let values = constant_values(graph, id, weights)
        .ok_or_else(|| unsupported(node, format!("input {slot} must be a constant")))?;
    Ok(Some(values.iter().map(|&v| v as i64).collect()))
}

/// `x` of `shape` with `axis` replaced by the elements at `positions` along
/// it, in that order. The slices of the axis are rows of everything inside
/// it, gathered by index.
pub(super) fn gather_axis(
    graph: &mut Graph,
    node: &proto::Node,
    x: NodeId,
    shape: &[usize],
    axis: usize,
    positions: &[usize],
) -> Result<NodeId, OnnxError> {
    let outer = shape[..axis].iter().product::<usize>();
    let dim = shape[axis];
    let inner = shape[axis + 1..].iter().product::<usize>();
    let rows = outer.saturating_mul(dim);
    if rows >= EXACT_INDICES {
        return Err(unsupported(
            node,
            format!("gathers along axis {axis} of {shape:?} (supported: below 2^24 rows)"),
        ));
    }
    let indices: Vec<u32> = (0..outer)
        .flat_map(|o| positions.iter().map(move |&p| (o * dim + p) as u32))
        .collect();
    let table = view(graph, x, &[rows, inner])?;
    let indices = graph.constant_u32(&indices, &[outer * positions.len()]);
    Ok(graph.embedding(indices, table))
}

/// `Pad` in its four modes, from the attributes before opset 11 and the
/// inputs after, `axes` included. Negative padding crops the axis first.
/// Constant padding then joins filled blocks; the other modes gather the
/// positions they repeat.
pub(super) fn pad(
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
    let mode = attrs.s("mode").unwrap_or("constant");
    let pads = match constant_ints(graph, node, 1, name_to_id, weights)? {
        Some(pads) => pads,
        None => attrs.ints("pads").to_vec(),
    };
    let value = match node.inputs.get(2).filter(|name| !name.is_empty()) {
        Some(name) => {
            let id = resolve_input(name, name_to_id, &node.name)?;
            scalar_constant(graph, id, weights)
                .ok_or_else(|| unsupported(node, "the padding value must be a constant"))?
        }
        None => attrs.f("value", 0.0),
    };
    let axes: Vec<usize> = match constant_ints(graph, node, 3, name_to_id, weights)? {
        Some(axes) => axes
            .iter()
            .map(|&a| if a < 0 { a + rank as i64 } else { a } as usize)
            .collect(),
        None => (0..rank).collect(),
    };
    if pads.len() != 2 * axes.len() || axes.iter().any(|&a| a >= rank) {
        return Err(OnnxError::ShapeError(format!(
            "node '{}': pads {pads:?} for axes {axes:?} of {x_shape:?}",
            node.name
        )));
    }
    if !matches!(mode, "constant" | "edge" | "reflect" | "wrap") {
        return Err(unsupported(node, format!("mode {mode}")));
    }
    let mut out = x;
    let mut shape = x_shape;
    for (i, &axis) in axes.iter().enumerate() {
        let (before, after) = (pads[i], pads[axes.len() + i]);
        let dim = shape[axis] as i64;
        let padded = dim.saturating_add(before).saturating_add(after);
        let (first, last) = ((-before).max(0), (-after).max(0));
        if padded < 1 || padded >= EXACT_INDICES as i64 || first + last >= dim {
            return Err(OnnxError::ShapeError(format!(
                "node '{}': padding axis {axis} of {shape:?} by {before} and {after}",
                node.name
            )));
        }
        // Negative padding crops first; every mode then pads what is left.
        if first + last > 0 {
            out = narrow(
                graph,
                out,
                &shape,
                axis,
                first as usize..(dim - last) as usize,
            );
            shape[axis] = (dim - first - last) as usize;
        }
        let (before, after) = (before.max(0), after.max(0));
        if before + after == 0 {
            continue;
        }
        let dim = shape[axis] as i64;
        out = if mode == "constant" {
            pad_constant(graph, out, &shape, axis, [before, after], value)?
        } else {
            // Where each output element comes from: the nearest edge, the
            // mirror image without repeating the edge, or the other end.
            let period = 2 * (dim - 1);
            let source = |o: i64| match mode {
                "edge" => o.clamp(0, dim - 1),
                "wrap" => o.rem_euclid(dim),
                _ if period == 0 => 0,
                _ => {
                    let m = o.rem_euclid(period);
                    if m < dim { m } else { period - m }
                }
            };
            let positions: Vec<usize> =
                (-before..dim + after).map(|o| source(o) as usize).collect();
            gather_axis(graph, node, out, &shape, axis, &positions)?
        };
        shape[axis] = padded as usize;
    }
    register_output(node, 0, out, &shape, name_to_id, shapes);
    Ok(())
}

/// Blocks of `value`, `before` and `after` slices long, on either side of
/// `axis`.
pub(super) fn pad_constant(
    graph: &mut Graph,
    x: NodeId,
    shape: &[usize],
    axis: usize,
    [before, after]: [i64; 2],
    value: f32,
) -> Result<NodeId, OnnxError> {
    let outer = shape[..axis].iter().product::<usize>();
    let inner = shape[axis + 1..].iter().product::<usize>();
    let mut kept = shape[axis];
    let mut out = view(graph, x, &[outer * kept * inner])?;
    let filled = |graph: &mut Graph, rows: i64| graph.fill(value, &[outer * rows as usize * inner]);
    if before > 0 {
        let block = filled(graph, before);
        out = graph.concat(
            block,
            out,
            outer as u32,
            before as u32,
            kept as u32,
            inner as u32,
        );
        kept += before as usize;
    }
    if after > 0 {
        let block = filled(graph, after);
        out = graph.concat(
            out,
            block,
            outer as u32,
            kept as u32,
            after as u32,
            inner as u32,
        );
    }
    Ok(out)
}

/// `Tile`: each repeated axis becomes that many copies of itself, a
/// broadcast of the whole block from the axis inward.
pub(super) fn tile(
    graph: &mut Graph,
    node: &proto::Node,
    name_to_id: &mut HashMap<String, NodeId>,
    shapes: &mut HashMap<String, Vec<usize>>,
    weights: &HashMap<String, Vec<f32>>,
) -> Result<(), OnnxError> {
    let x = resolve_input(&node.inputs[0], name_to_id, &node.name)?;
    let x_shape = get_shape(&node.inputs[0], shapes);
    let repeats = constant_ints(graph, node, 1, name_to_id, weights)?
        .ok_or_else(|| unsupported(node, "no repeats"))?;
    if repeats.len() != x_shape.len() || repeats.iter().any(|&r| r < 1) {
        return Err(OnnxError::ShapeError(format!(
            "node '{}': Tile of {x_shape:?} by {repeats:?}",
            node.name
        )));
    }
    let mut out = x;
    let mut shape = x_shape;
    for (axis, &times) in repeats.iter().enumerate() {
        let times = times as usize;
        if times == 1 {
            continue;
        }
        let outer = shape[..axis].iter().product::<usize>();
        let block = shape[axis..].iter().product::<usize>();
        let once = view(graph, out, &[outer, 1, block])?;
        out = graph.broadcast_to(once, &[outer, times, block]);
        shape[axis] *= times;
    }
    register_output(node, 0, out, &shape, name_to_id, shapes);
    Ok(())
}

/// `GatherElements`: each output element reads `data` where its own
/// position has the index in place of its coordinate along `axis`. Indices
/// are clamped into the axis, as the gather has no bounds of its own, and
/// truncated, since integer division of index tensors imports as F32.
pub(super) fn gather_elements(
    graph: &mut Graph,
    node: &proto::Node,
    name_to_id: &mut HashMap<String, NodeId>,
    shapes: &mut HashMap<String, Vec<usize>>,
) -> Result<(), OnnxError> {
    let data = resolve_input(&node.inputs[0], name_to_id, &node.name)?;
    let indices = resolve_input(&node.inputs[1], name_to_id, &node.name)?;
    let data_shape = get_shape(&node.inputs[0], shapes);
    let index_shape = get_shape(&node.inputs[1], shapes);
    let rank = data_shape.len() as i64;
    let axis = node.attrs.i("axis", 0);
    let axis = if axis < 0 { axis + rank } else { axis };
    let fits = index_shape.len() == data_shape.len()
        && (0..rank).contains(&axis)
        && (0..data_shape.len()).all(|d| d == axis as usize || index_shape[d] <= data_shape[d]);
    let (len, count) = (
        data_shape.iter().product::<usize>(),
        index_shape.iter().product::<usize>(),
    );
    if !fits || len >= EXACT_INDICES || count == 0 {
        return Err(OnnxError::ShapeError(format!(
            "node '{}': GatherElements of {data_shape:?} at {index_shape:?} on axis {axis}",
            node.name
        )));
    }
    let axis = axis as usize;
    let mut strides = vec![1usize; data_shape.len()];
    for d in (0..data_shape.len() - 1).rev() {
        strides[d] = strides[d + 1] * data_shape[d + 1];
    }
    // Every coordinate but the one along the axis, as an offset into `data`.
    let base: Vec<f32> = (0..count)
        .map(|flat| {
            let mut rest = flat;
            let mut offset = 0;
            for d in (0..index_shape.len()).rev() {
                let coordinate = rest % index_shape[d];
                rest /= index_shape[d];
                if d != axis {
                    offset += coordinate * strides[d];
                }
            }
            offset as f32
        })
        .collect();
    let mut index = view(graph, indices, &[count])?;
    if graph.node(index).ty.dtype == DType::U32 {
        index = graph.to_f32(index);
    }
    // Negative indices count from the end.
    let dim = data_shape[axis] as f32;
    let negative = graph.neg(index);
    let negative = graph.clamp(negative, 0.0, 1.0);
    let wrap = graph.scale(negative, dim);
    let index = graph.add(index, wrap);
    let index = graph.clamp(index, 0.0, dim - 1.0);
    let whole = graph.to_u32(index);
    let whole = graph.to_f32(whole);
    let along = graph.scale(whole, strides[axis] as f32);
    let base = graph.constant(base, &[count]);
    let flat = graph.add(along, base);
    let flat = graph.to_u32(flat);
    let table = view(graph, data, &[len, 1])?;
    let out = graph.embedding(flat, table);
    register_output(node, 0, out, &index_shape, name_to_id, shapes);
    Ok(())
}

/// `TopK` along any axis of up to four: sorted values and their indices,
/// ties to the lower index. [`Graph::top_k_inner`] selects and sorts each
/// row's columns in one dispatch; the values are gathered at them.
pub(super) fn top_k(
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
    // `k` is an input from opset 10 and an attribute before.
    let k = match node.inputs.get(1).filter(|name| !name.is_empty()) {
        Some(name) => {
            let id = resolve_input(name, name_to_id, &node.name)?;
            scalar_constant(graph, id, weights).ok_or_else(|| {
                OnnxError::UnsupportedOp(format!("TopK '{}': k must be a constant", node.name))
            })? as i64
        }
        None => attrs.i("k", 0),
    };
    let axis = attrs.i("axis", -1);
    let axis = if axis < 0 { axis + rank as i64 } else { axis };
    if !(1..=4).contains(&rank) || !(0..rank as i64).contains(&axis) {
        return Err(OnnxError::UnsupportedOp(format!(
            "TopK '{}': axis {axis} of {x_shape:?} (supported: up to 4 axes)",
            node.name
        )));
    }
    let axis = axis as usize;
    let len = x_shape[axis];
    let max_k = crate::graph::MAX_TOP_K_INNER;
    if !(1..=len as i64).contains(&k) || k as usize > max_k {
        return Err(OnnxError::UnsupportedOp(format!(
            "TopK '{}': k = {k} of {len} (supported: 1 to {max_k})",
            node.name
        )));
    }
    let k = k as usize;
    let largest = attrs.i("largest", 1) != 0;
    // Bring the axis innermost; swapping it with the last is its own inverse.
    let mut perm: Vec<usize> = (0..rank).collect();
    perm.swap(axis, rank - 1);
    let rows = x_shape.iter().product::<usize>() / len;
    let mut moved = view(graph, x, &x_shape)?;
    if axis != rank - 1 {
        moved = graph.permute(moved, &perm);
    }
    let mut remaining = view(graph, moved, &[rows, len])?;
    if !largest {
        remaining = graph.neg(remaining);
    }
    // The gather addresses each pick as `row · len + column` in F32.
    let elements = rows * len;
    if elements > 1 << 24 || rows > 65535 {
        return Err(OnnxError::UnsupportedOp(format!(
            "TopK '{}': {rows} rows of {len} (supported: up to 65535 rows and 2^24 elements)",
            node.name
        )));
    }
    let indices = graph.top_k_inner(remaining, k);
    let row_starts = graph.constant(
        (0..rows)
            .flat_map(|row| std::iter::repeat_n((row * len) as f32, k))
            .collect(),
        &[rows * k],
    );
    let flat = view(graph, indices, &[rows * k])?;
    let flat = graph.add(flat, row_starts);
    let flat = graph.to_u32(flat);
    let table = view(graph, remaining, &[elements, 1])?;
    let mut values = graph.embedding(flat, table);
    if !largest {
        values = graph.neg(values);
    }
    let mut moved_shape: Vec<usize> = perm.iter().map(|&d| x_shape[d]).collect();
    moved_shape[rank - 1] = k;
    let mut out_shape = x_shape.clone();
    out_shape[axis] = k;
    for (slot, out) in [values, indices].into_iter().enumerate() {
        let mut out = view(graph, out, &moved_shape)?;
        if axis != rank - 1 {
            out = graph.permute(out, &perm);
        }
        register_output(node, slot, out, &out_shape, name_to_id, shapes);
    }
    Ok(())
}
