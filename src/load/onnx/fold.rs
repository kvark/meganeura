//! Import-time evaluation of nodes whose every input is a constant: shape
//! arithmetic, index ranges, masks and the anchor and position tables an
//! exporter computes from them. The result is one constant, so nothing on
//! the GPU recomputes it, and ops that only ever see such values, `Range`
//! or `ScatterND` among them, need no kernel.
//!
//! Values are F32, as the importer holds every tensor: integer arithmetic
//! is exact below 2^24, and integer division is not truncated.

use std::collections::HashMap;

use crate::{
    graph::{Graph, NodeId},
    load::onnx::{OnnxError, broadcast_shape, constant_values, proto, register_output},
};

/// Folded results larger than this many elements are left to the GPU.
const MAX_FOLDED: usize = 1 << 22;

struct Value {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl Value {
    fn scalar(&self) -> Option<f32> {
        match *self.data {
            [v] => Some(v),
            _ => None,
        }
    }

    fn ints(&self) -> Vec<i64> {
        self.data.iter().map(|&v| v as i64).collect()
    }
}

/// Row-major strides of `shape`.
fn strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }
    strides
}

/// `value` read at every position of `out_shape`, by NumPy broadcasting.
fn broadcast(value: &Value, out_shape: &[usize]) -> Vec<f32> {
    let rank = out_shape.len();
    let mut shape = vec![1; rank - value.shape.len()];
    shape.extend_from_slice(&value.shape);
    let source = strides(&shape);
    let count = out_shape.iter().product::<usize>();
    (0..count)
        .map(|flat| {
            let (mut rest, mut index) = (flat, 0);
            for d in (0..rank).rev() {
                let coordinate = rest % out_shape[d];
                rest /= out_shape[d];
                if shape[d] != 1 {
                    index += coordinate * source[d];
                }
            }
            value.data[index]
        })
        .collect()
}

/// `f` over the broadcast of `values`.
fn zip(values: &[&Value], f: impl Fn(&[f32]) -> f32) -> Option<Value> {
    let shape = values
        .iter()
        .try_fold(Vec::new(), |shape, v| broadcast_shape(&shape, &v.shape))?;
    let spread: Vec<Vec<f32>> = values.iter().map(|v| broadcast(v, &shape)).collect();
    let data = (0..shape.iter().product::<usize>())
        .map(|i| f(&spread.iter().map(|s| s[i]).collect::<Vec<_>>()))
        .collect();
    Some(Value { data, shape })
}

fn truth(v: f32) -> bool {
    v != 0.0
}

fn flag(b: bool) -> f32 {
    f32::from(u8::from(b))
}

fn axis(node: &proto::Node, rank: usize, default: i64) -> Option<usize> {
    let axis = node.attrs.i("axis", default);
    let axis = if axis < 0 { axis + rank as i64 } else { axis };
    (0..rank as i64).contains(&axis).then_some(axis as usize)
}

/// The node evaluated on `inputs` (an absent optional input is `None`), or
/// `None` when this form is not evaluated here.
fn evaluate(node: &proto::Node, inputs: &[Option<Value>]) -> Option<Value> {
    let attrs = &node.attrs;
    let arg = |i: usize| inputs.get(i).and_then(Option::as_ref);
    let unary = |f: fn(f32) -> f32| {
        let x = arg(0)?;
        Some(Value {
            data: x.data.iter().map(|&v| f(v)).collect(),
            shape: x.shape.clone(),
        })
    };
    let binary = |f: fn(f32, f32) -> f32| zip(&[arg(0)?, arg(1)?], |v| f(v[0], v[1]));
    match node.op_type.as_str() {
        "Add" => binary(|a, b| a + b),
        "Sub" => binary(|a, b| a - b),
        "Mul" => binary(|a, b| a * b),
        "Div" => binary(|a, b| a / b),
        "Equal" => binary(|a, b| flag(a == b)),
        "Greater" => binary(|a, b| flag(a > b)),
        "Less" => binary(|a, b| flag(a < b)),
        "And" => binary(|a, b| flag(truth(a) && truth(b))),
        "Or" => binary(|a, b| flag(truth(a) || truth(b))),
        "Not" => unary(|v| flag(!truth(v))),
        "Sin" => unary(f32::sin),
        "Cos" => unary(f32::cos),
        "Log" => unary(f32::ln),
        "Neg" => unary(|v| -v),
        "Cast" => {
            // Integer and boolean targets drop the fraction; the rest keep F32.
            let to = attrs.i("to", 1);
            match to {
                9 => unary(|v| flag(truth(v))),
                2..=7 | 12 | 13 => unary(f32::trunc),
                _ => unary(|v| v),
            }
        }
        "Where" => zip(&[arg(0)?, arg(1)?, arg(2)?], |v| {
            if truth(v[0]) { v[1] } else { v[2] }
        }),
        "Range" => {
            let (start, limit, delta) = (arg(0)?.scalar()?, arg(1)?.scalar()?, arg(2)?.scalar()?);
            let count = ((limit - start) / delta).ceil().max(0.0) as usize;
            Some(Value {
                data: (0..count).map(|i| start + i as f32 * delta).collect(),
                shape: vec![count],
            })
        }
        "ConstantOfShape" => {
            let shape: Vec<usize> = arg(0)?.ints().iter().map(|&d| d.max(0) as usize).collect();
            let value = attrs.tensor("value").map_or(0.0, |t| t.data[0]);
            Some(Value {
                data: vec![value; shape.iter().product()],
                shape,
            })
        }
        "Expand" => {
            let x = arg(0)?;
            let target: Vec<usize> = arg(1)?.ints().iter().map(|&d| d.max(0) as usize).collect();
            let shape = broadcast_shape(&x.shape, &target)?;
            Some(Value {
                data: broadcast(x, &shape),
                shape,
            })
        }
        "Concat" => {
            let parts: Vec<&Value> = inputs.iter().flatten().collect();
            let first = parts.first()?;
            let axis = axis(node, first.shape.len(), 0)?;
            let mut shape = first.shape.clone();
            shape[axis] = parts
                .iter()
                .map(|p| p.shape.get(axis).copied().unwrap_or(0))
                .sum();
            let outer = shape[..axis].iter().product::<usize>();
            let mut data = Vec::with_capacity(shape.iter().product());
            for o in 0..outer {
                for p in &parts {
                    let block = p.shape[axis..].iter().product::<usize>();
                    data.extend_from_slice(&p.data[o * block..(o + 1) * block]);
                }
            }
            Some(Value { data, shape })
        }
        "Gather" => {
            let (data, indices) = (arg(0)?, arg(1)?);
            let axis = axis(node, data.shape.len(), 0)?;
            let dim = data.shape[axis] as i64;
            let outer = data.shape[..axis].iter().product::<usize>();
            let inner = data.shape[axis + 1..].iter().product::<usize>();
            let mut values = Vec::with_capacity(outer * indices.data.len() * inner);
            for o in 0..outer {
                for &i in &indices.data {
                    let i = i as i64;
                    let i = if i < 0 { i + dim } else { i };
                    let row = (o * dim as usize + usize::try_from(i).ok()?) * inner;
                    values.extend_from_slice(data.data.get(row..row + inner)?);
                }
            }
            let mut shape = data.shape[..axis].to_vec();
            shape.extend_from_slice(&indices.shape);
            shape.extend_from_slice(&data.shape[axis + 1..]);
            Some(Value {
                data: values,
                shape,
            })
        }
        "Slice" => {
            let x = arg(0)?;
            let rank = x.shape.len();
            let (starts, ends) = (arg(1)?.ints(), arg(2)?.ints());
            let axes = arg(3).map_or_else(|| (0..starts.len() as i64).collect(), Value::ints);
            let steps = arg(4).map_or_else(|| vec![1; starts.len()], Value::ints);
            // Per axis, the positions it keeps.
            let mut keep: Vec<Vec<usize>> = x.shape.iter().map(|&d| (0..d).collect()).collect();
            for (((&a, &start), &end), &step) in axes.iter().zip(&starts).zip(&ends).zip(&steps) {
                let a = if a < 0 { a + rank as i64 } else { a } as usize;
                let dim = *x.shape.get(a)? as i64;
                if step == 0 {
                    return None;
                }
                let wrap = |v: i64| if v < 0 { v.saturating_add(dim) } else { v };
                let (start, end) = if step > 0 {
                    (wrap(start).clamp(0, dim), wrap(end).clamp(0, dim))
                } else {
                    (wrap(start).clamp(0, dim - 1), wrap(end).clamp(-1, dim - 1))
                };
                let mut positions = Vec::new();
                let mut p = start;
                while (step > 0 && p < end) || (step < 0 && p > end) {
                    positions.push(p as usize);
                    p += step;
                }
                keep[a] = positions;
            }
            let shape: Vec<usize> = keep.iter().map(Vec::len).collect();
            let source = strides(&x.shape);
            let data = (0..shape.iter().product::<usize>())
                .map(|flat| {
                    let (mut rest, mut index) = (flat, 0);
                    for d in (0..rank).rev() {
                        index += keep[d][rest % shape[d]] * source[d];
                        rest /= shape[d];
                    }
                    x.data[index]
                })
                .collect();
            Some(Value { data, shape })
        }
        "ScatterND" => {
            let (data, indices, updates) = (arg(0)?, arg(1)?, arg(2)?);
            let depth = *indices.shape.last()?;
            let block = data.shape.get(depth..)?.iter().product::<usize>();
            let source = strides(&data.shape);
            let mut out = data.data.clone();
            for (n, index) in indices.data.chunks(depth.max(1)).enumerate() {
                let start = index
                    .iter()
                    .zip(&source)
                    .try_fold(0usize, |offset, (&i, &s)| {
                        Some(offset + usize::try_from(i as i64).ok()? * s)
                    })?;
                let update = updates.data.get(n * block..(n + 1) * block)?;
                let target = out.get_mut(start..start + block)?;
                match attrs.s("reduction").unwrap_or("none") {
                    "none" => target.copy_from_slice(update),
                    "add" => target.iter_mut().zip(update).for_each(|(t, &u)| *t += u),
                    "mul" => target.iter_mut().zip(update).for_each(|(t, &u)| *t *= u),
                    _ => return None,
                }
            }
            Some(Value {
                data: out,
                shape: data.shape.clone(),
            })
        }
        "ReduceSum" => {
            let x = arg(0)?;
            let rank = x.shape.len();
            let axes: Vec<usize> = match arg(1).map(Value::ints).filter(|a| !a.is_empty()) {
                Some(axes) => axes
                    .iter()
                    .map(|&a| if a < 0 { a + rank as i64 } else { a } as usize)
                    .collect(),
                None if attrs.i("noop_with_empty_axes", 0) != 0 => {
                    return Some(Value {
                        data: x.data.clone(),
                        shape: x.shape.clone(),
                    });
                }
                None => (0..rank).collect(),
            };
            let kept: Vec<usize> = (0..rank)
                .map(|d| if axes.contains(&d) { 1 } else { x.shape[d] })
                .collect();
            let target = strides(&kept);
            let mut data = vec![0.0; kept.iter().product()];
            for (flat, &v) in x.data.iter().enumerate() {
                let (mut rest, mut index) = (flat, 0);
                for d in (0..rank).rev() {
                    let coordinate = rest % x.shape[d];
                    rest /= x.shape[d];
                    if !axes.contains(&d) {
                        index += coordinate * target[d];
                    }
                }
                data[index] += v;
            }
            let shape = if attrs.i("keepdims", 1) != 0 {
                kept
            } else {
                (0..rank)
                    .filter(|d| !axes.contains(d))
                    .map(|d| x.shape[d])
                    .collect()
            };
            Some(Value { data, shape })
        }
        "MatMul" => {
            let (a, b) = (arg(0)?, arg(1)?);
            let ([m, k], [k2, n]) = (
                <[usize; 2]>::try_from(a.shape.get(a.shape.len().checked_sub(2)?..)?).ok()?,
                <[usize; 2]>::try_from(b.shape.get(b.shape.len().checked_sub(2)?..)?).ok()?,
            );
            if k != k2 {
                return None;
            }
            let lead =
                broadcast_shape(&a.shape[..a.shape.len() - 2], &b.shape[..b.shape.len() - 2])?;
            let batches = lead.iter().product::<usize>();
            let spread = |v: &Value, rows: usize, cols: usize| {
                let mut shape = lead.clone();
                shape.extend([rows, cols]);
                let mut lead_only = v.shape.clone();
                lead_only.truncate(v.shape.len() - 2);
                // Broadcast whole matrices: treat each as one element.
                let matrices = Value {
                    data: (0..lead_only.iter().product::<usize>())
                        .map(|i| i as f32)
                        .collect(),
                    shape: lead_only,
                };
                let order = broadcast(&matrices, &lead);
                order
                    .iter()
                    .flat_map(|&i| {
                        v.data[i as usize * rows * cols..(i as usize + 1) * rows * cols].to_vec()
                    })
                    .collect::<Vec<f32>>()
            };
            let (a_data, b_data) = (spread(a, m, k), spread(b, k, n));
            let mut data = vec![0.0f32; batches * m * n];
            for batch in 0..batches {
                for i in 0..m {
                    for j in 0..n {
                        data[(batch * m + i) * n + j] = (0..k)
                            .map(|p| {
                                a_data[(batch * m + i) * k + p] * b_data[(batch * k + p) * n + j]
                            })
                            .sum();
                    }
                }
            }
            let mut shape = lead;
            shape.extend([m, n]);
            Some(Value { data, shape })
        }
        _ => None,
    }
}

/// Evaluate `node` here if every input is a constant and its form is one
/// [`evaluate`] covers, and register the result as a constant. Returns
/// whether it did.
pub(super) fn fold(
    graph: &mut Graph,
    node: &proto::Node,
    name_to_id: &mut HashMap<String, NodeId>,
    shapes: &mut HashMap<String, Vec<usize>>,
    weights: &HashMap<String, Vec<f32>>,
) -> Result<bool, OnnxError> {
    let mut inputs = Vec::with_capacity(node.inputs.len());
    for name in &node.inputs {
        if name.is_empty() {
            inputs.push(None);
            continue;
        }
        let Some(&id) = name_to_id.get(name) else {
            return Ok(false);
        };
        let Some(data) = constant_values(graph, id, weights) else {
            return Ok(false);
        };
        let Some(shape) = shapes.get(name) else {
            return Ok(false);
        };
        if shape.iter().product::<usize>() != data.len() {
            return Ok(false);
        }
        inputs.push(Some(Value {
            data: data.to_vec(),
            shape: shape.clone(),
        }));
    }
    let Some(value) = evaluate(node, &inputs) else {
        return Ok(false);
    };
    if value.data.is_empty() || value.data.len() > MAX_FOLDED {
        return Ok(false);
    }
    let len = value.data.len();
    let id = graph.constant(value.data, &[len]);
    register_output(node, 0, id, &value.shape, name_to_id, shapes);
    Ok(true)
}
