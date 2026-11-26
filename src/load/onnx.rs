//! ONNX model import: load standard ONNX files into Meganeura's `Graph` IR.
//!
//! Translates an ONNX computation graph into our `Graph` at runtime, then the
//! normal pipeline (optimize -> compile -> Session) handles execution. No Rust
//! codegen needed.

use std::collections::HashMap;
use std::path::Path;

use crate::graph::{Graph, NodeId, Op};

mod fold;
mod indexing;
mod proto;
mod spatial;

/// Result of loading an ONNX model.
pub struct OnnxModel {
    /// The computation graph, ready for optimize() and compile().
    pub graph: Graph,
    /// Named weight tensors extracted from ONNX initializers.
    /// Call `session.set_parameter(name, &data)` for each entry.
    pub weights: HashMap<String, Vec<f32>>,
}

/// Errors that can occur during ONNX import.
#[derive(Debug)]
pub enum OnnxError {
    /// Failed to parse the ONNX protobuf.
    ParseError(String),
    /// An ONNX operator has no equivalent in Meganeura.
    UnsupportedOp(String),
    /// Shape inference failed or produced an invalid shape.
    ShapeError(String),
}

impl std::fmt::Display for OnnxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::ParseError(ref e) => write!(f, "ONNX parse error: {e}"),
            Self::UnsupportedOp(ref e) => write!(f, "unsupported ONNX op: {e}"),
            Self::ShapeError(ref e) => write!(f, "ONNX shape error: {e}"),
        }
    }
}

impl std::error::Error for OnnxError {}

/// Load an ONNX model from a file path.
pub fn load_onnx(path: &Path) -> Result<OnnxModel, OnnxError> {
    let bytes = std::fs::read(path).map_err(|e| OnnxError::ParseError(e.to_string()))?;
    load_onnx_bytes(&bytes, Some(path))
}

/// Load an ONNX model from raw bytes.
/// If `path` is provided, external data files are resolved relative to its parent directory.
pub fn load_onnx_bytes(bytes: &[u8], path: Option<&Path>) -> Result<OnnxModel, OnnxError> {
    let dir = path.map(|p| p.parent().unwrap_or(Path::new("")));
    let mut onnx = proto::parse_model(bytes, dir).map_err(OnnxError::ParseError)?;
    let weights: HashMap<String, Vec<f32>> = onnx
        .initializers
        .iter_mut()
        .map(|t| (t.name.clone(), std::mem::take(&mut t.data)))
        .collect();
    let graph = translate_graph(&onnx, &weights)?;
    Ok(OnnxModel { graph, weights })
}

/// Translate a decoded ONNX graph into a Meganeura Graph.
fn translate_graph(
    onnx: &proto::Graph,
    weights: &HashMap<String, Vec<f32>>,
) -> Result<Graph, OnnxError> {
    let mut graph = Graph::new();
    // Map ONNX tensor names -> Meganeura NodeId
    let mut name_to_id: HashMap<String, NodeId> = HashMap::new();
    // Track output shapes by name for shape inference
    let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();

    // 1. Create parameter nodes for initializers (weights)
    for init in &onnx.initializers {
        let id = graph.parameter(&init.name, &init.shape);
        name_to_id.insert(init.name.clone(), id);
        shapes.insert(init.name.clone(), init.shape.clone());
    }

    // 2. Create input nodes for graph inputs that aren't initializers
    for input in &onnx.inputs {
        let shape = input.shape.clone().unwrap_or_else(|| {
            log::warn!("ONNX input '{}': shape unknown, using [1]", input.name);
            vec![1]
        });
        // Flatten to 2D for our IR: [batch, ..., features] -> [batch*..., features]
        let flat_shape = flatten_to_2d(&shape);
        let id = graph.input(&input.name, &flat_shape);
        name_to_id.insert(input.name.clone(), id);
        shapes.insert(input.name.clone(), shape);
    }

    // 3. Topological sort for correct processing order
    let known_names: Vec<String> = name_to_id.keys().cloned().collect();
    let topo_order = onnx.topological_sort(&known_names);

    // 4. Translate each ONNX node
    for &node_idx in &topo_order {
        let node = &onnx.nodes[node_idx];
        translate_node(&mut graph, node, &mut name_to_id, &mut shapes, weights)?;
    }

    // 5. Set outputs
    let output_ids: Vec<NodeId> = onnx
        .outputs
        .iter()
        .filter_map(|output| name_to_id.get(&output.name).copied())
        .collect();
    graph.set_outputs(output_ids);

    Ok(graph)
}

/// Look up a required input by ONNX name.
fn resolve_input(
    name: &str,
    name_to_id: &HashMap<String, NodeId>,
    node_name: &str,
) -> Result<NodeId, OnnxError> {
    name_to_id.get(name).copied().ok_or_else(|| {
        OnnxError::ShapeError(format!(
            "node '{}': input '{}' not found in graph",
            node_name, name
        ))
    })
}

/// Get the shape of an ONNX tensor by name.
fn get_shape(name: &str, shapes: &HashMap<String, Vec<usize>>) -> Vec<usize> {
    shapes.get(name).cloned().unwrap_or_default()
}

/// Flatten a multi-dimensional shape to 2D [batch, features] for our IR.
/// Collapses all leading dims into the first axis.
fn flatten_to_2d(shape: &[usize]) -> Vec<usize> {
    if shape.len() <= 2 {
        return shape.to_vec();
    }
    let last = *shape.last().unwrap_or(&1);
    let batch: usize = shape[..shape.len() - 1].iter().product();
    vec![batch, last]
}

/// Translate a single ONNX node into Meganeura graph nodes.
fn translate_node(
    graph: &mut Graph,
    node: &proto::Node,
    name_to_id: &mut HashMap<String, NodeId>,
    shapes: &mut HashMap<String, Vec<usize>>,
    weights: &HashMap<String, Vec<f32>>,
) -> Result<(), OnnxError> {
    let attrs = &node.attrs;
    if fold::fold(graph, node, name_to_id, shapes, weights)? {
        return Ok(());
    }

    match node.op_type.as_str() {
        // --- Element-wise unary ---
        "Relu" => unary_op(graph, node, name_to_id, shapes, Op::Relu)?,
        "Sigmoid" => unary_op(graph, node, name_to_id, shapes, Op::Sigmoid)?,
        "Neg" => unary_op(graph, node, name_to_id, shapes, Op::Neg)?,
        "Abs" => unary_op(graph, node, name_to_id, shapes, Op::Abs)?,
        "Log" => unary_op(graph, node, name_to_id, shapes, Op::Log)?,
        "Reciprocal" => unary_op(graph, node, name_to_id, shapes, Op::Recip)?,
        // Exact unless the node asks for the tanh approximation.
        "Gelu" => {
            let op = match attrs.s("approximate") {
                None | Some("none") => Op::GeluErf,
                Some("tanh") => Op::Gelu,
                Some(other) => {
                    return Err(OnnxError::UnsupportedOp(format!(
                        "Gelu '{}': approximate = {other:?}",
                        node.name
                    )));
                }
            };
            unary_op(graph, node, name_to_id, shapes, op)?
        }
        "Silu" | "SiLU" => unary_op(graph, node, name_to_id, shapes, Op::Silu)?,

        // Elementary math, as exporters write decomposed norms and
        // activations. These map onto primitives, and the optimizer folds
        // recognized decompositions back into fused kernels.
        "Sqrt" => unary_op(graph, node, name_to_id, shapes, Op::Sqrt)?,
        "Exp" => unary_op(graph, node, name_to_id, shapes, Op::Exp)?,
        "Tanh" => unary_op(graph, node, name_to_id, shapes, Op::Tanh)?,
        "Pow" => {
            let x = resolve_input(&node.inputs[0], name_to_id, &node.name)?;
            let exponent = node
                .inputs
                .get(1)
                .and_then(|name| name_to_id.get(name))
                .and_then(|&id| scalar_constant(graph, id, weights));
            let out = match exponent {
                Some(1.0) => x,
                Some(2.0) => graph.mul(x, x),
                Some(3.0) => {
                    let square = graph.mul(x, x);
                    graph.mul(square, x)
                }
                Some(0.5) => graph.sqrt(x),
                Some(-0.5) => graph.rsqrt(x),
                Some(-1.0) => graph.recip(x),
                other => {
                    return Err(OnnxError::UnsupportedOp(format!(
                        "Pow: exponent {other:?} (supported: constant 1, 2, 3, 0.5, -0.5, -1)"
                    )));
                }
            };
            let x_shape = get_shape(&node.inputs[0], shapes);
            register_output(node, 0, out, &x_shape, name_to_id, shapes);
        }
        "ReduceMean" | "ReduceSum" | "ReduceMax" => {
            reduce_op(graph, node, name_to_id, shapes, weights)?;
        }
        "Erf" => unary_op(graph, node, name_to_id, shapes, Op::Erf)?,

        // Bounds are inputs from opset 11 and attributes before. A missing
        // one is the largest finite value, so it saturates infinities.
        "Clip" => {
            let x = resolve_input(&node.inputs[0], name_to_id, &node.name)?;
            let x_shape = get_shape(&node.inputs[0], shapes);
            let bound = |slot: usize, attr: &str, default: f32| -> Result<f32, OnnxError> {
                let Some(name) = node.inputs.get(slot).filter(|name| !name.is_empty()) else {
                    return Ok(attrs.f(attr, default));
                };
                let id = resolve_input(name, name_to_id, &node.name)?;
                scalar_constant(graph, id, weights).ok_or_else(|| {
                    OnnxError::UnsupportedOp(format!(
                        "Clip '{}': bounds must be constants",
                        node.name
                    ))
                })
            };
            let min = bound(1, "min", f32::MIN)?.max(f32::MIN);
            let max = bound(2, "max", f32::MAX)?.min(f32::MAX);
            // A lower bound above the upper one leaves the upper one, as ONNX's does.
            let out = graph.clamp(x, min.min(max), max);
            register_output(node, 0, out, &x_shape, name_to_id, shapes);
        }

        // Cast: passthrough (we only support f32)
        "Cast" => {
            let x = resolve_input(&node.inputs[0], name_to_id, &node.name)?;
            let x_shape = get_shape(&node.inputs[0], shapes);
            if !node.outputs.is_empty() {
                name_to_id.insert(node.outputs[0].clone(), x);
                shapes.insert(node.outputs[0].clone(), x_shape);
            }
        }

        // Shape: produces a 1D constant of the input's static shape
        "Shape" => {
            let x_shape = get_shape(&node.inputs[0], shapes);
            let data: Vec<f32> = x_shape.iter().map(|&d| d as f32).collect();
            let len = data.len();
            let id = graph.constant(data, &[len]);
            if !node.outputs.is_empty() {
                name_to_id.insert(node.outputs[0].clone(), id);
                shapes.insert(node.outputs[0].clone(), vec![len]);
            }
        }

        // --- Element-wise binary ---
        "Add" => binary_op(graph, node, name_to_id, shapes, weights, BinaryKind::Add)?,
        "Sub" => binary_op(graph, node, name_to_id, shapes, weights, BinaryKind::Sub)?,
        "Mul" => binary_op(graph, node, name_to_id, shapes, weights, BinaryKind::Mul)?,
        "Div" => binary_op(graph, node, name_to_id, shapes, weights, BinaryKind::Div)?,

        // --- MatMul ---
        "MatMul" => {
            let a = resolve_input(&node.inputs[0], name_to_id, &node.name)?;
            let b = resolve_input(&node.inputs[1], name_to_id, &node.name)?;
            let a_shape = get_shape(&node.inputs[0], shapes);
            let b_shape = get_shape(&node.inputs[1], shapes);
            let (out, out_shape) =
                matmul_nd(graph, a, &a_shape, b, &b_shape)?.ok_or_else(|| {
                    OnnxError::ShapeError(format!(
                        "node '{}': MatMul of {a_shape:?} and {b_shape:?}",
                        node.name
                    ))
                })?;
            register_output(node, 0, out, &out_shape, name_to_id, shapes);
        }

        // Gemm: C = alpha * A' @ B' + beta * C_bias
        // Where A' = transpose(A) if transA, B' = transpose(B) if transB
        "Gemm" => {
            let a = resolve_input(&node.inputs[0], name_to_id, &node.name)?;
            let b = resolve_input(&node.inputs[1], name_to_id, &node.name)?;
            let trans_a = attrs.i("transA", 0) != 0;
            let trans_b = attrs.i("transB", 0) != 0;
            let a_shape = get_shape(&node.inputs[0], shapes);
            let b_shape = get_shape(&node.inputs[1], shapes);
            let (alpha, beta) = (attrs.f("alpha", 1.0), attrs.f("beta", 1.0));
            if a_shape.len() != 2 || b_shape.len() != 2 || alpha != 1.0 || beta != 1.0 {
                return Err(OnnxError::UnsupportedOp(format!(
                    "Gemm of {a_shape:?} and {b_shape:?} scaled by {alpha} and {beta} \
                     (supported: matrices, unscaled)"
                )));
            }
            let a = view(graph, a, &a_shape)?;
            let b = view(graph, b, &b_shape)?;

            let mm = match (trans_a, trans_b) {
                (false, false) => graph.matmul(a, b),
                (true, false) => graph.matmul_at(a, b),
                (false, true) => graph.matmul_bt(a, b),
                (true, true) => {
                    // A^T @ B^T = (B @ A)^T — decompose
                    let ba = graph.matmul(b, a);
                    graph.transpose(ba)
                }
            };

            // Output shape
            let m = if trans_a {
                a_shape.get(1).copied().unwrap_or(1)
            } else {
                a_shape.first().copied().unwrap_or(1)
            };
            let n = if trans_b {
                b_shape.first().copied().unwrap_or(1)
            } else {
                b_shape.get(1).copied().unwrap_or(1)
            };
            let out_shape = vec![m, n];

            // Add bias if present
            let out = if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
                let c = resolve_input(&node.inputs[2], name_to_id, &node.name)?;
                let c = view(graph, c, &[n])?;
                graph.bias_add(mm, c)
            } else {
                mm
            };

            register_output(node, 0, out, &out_shape, name_to_id, shapes);
        }

        // --- Softmax ---
        "Softmax" | "LogSoftmax" => {
            let x = resolve_input(&node.inputs[0], name_to_id, &node.name)?;
            let x_shape = get_shape(&node.inputs[0], shapes);
            let rank = x_shape.len() as i64;
            // Opset 13 normalizes along `axis`, default -1; earlier opsets
            // flatten at `axis`, which is the same when it is the last. An
            // absent axis takes the modern default.
            let axis = attrs.i("axis", -1);
            if axis != -1 && axis != rank - 1 {
                return Err(OnnxError::UnsupportedOp(format!(
                    "{} along axis {axis} of {x_shape:?} (supported: the last axis)",
                    node.op_type
                )));
            }
            let x = as_matrix(graph, x, &x_shape)?;
            let out = if node.op_type == "Softmax" {
                graph.softmax(x)
            } else {
                graph.log_softmax(x)
            };
            register_output(node, 0, out, &x_shape, name_to_id, shapes);
        }

        // --- Normalization ---
        "LayerNormalization" | "LayerNorm" => {
            let x = resolve_input(&node.inputs[0], name_to_id, &node.name)?;
            let scale = resolve_input(&node.inputs[1], name_to_id, &node.name)?;
            let bias = if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
                resolve_input(&node.inputs[2], name_to_id, &node.name)?
            } else {
                // Create zero bias
                let scale_shape = get_shape(&node.inputs[1], shapes);
                let n = scale_shape.iter().product::<usize>().max(1);
                graph.constant(vec![0.0; n], &scale_shape)
            };
            let eps = attrs.f("epsilon", 1e-5);
            let x_shape = get_shape(&node.inputs[0], shapes);
            let x = as_matrix(graph, x, &x_shape)?;
            let cols = x_shape.last().copied().unwrap_or(1);
            let scale = view(graph, scale, &[cols])?;
            let bias = view(graph, bias, &[cols])?;
            let out = graph.layer_norm(x, scale, bias, eps);
            register_output(node, 0, out, &x_shape, name_to_id, shapes);
        }

        "RMSNorm" | "SimplifiedLayerNormalization" => {
            let x = resolve_input(&node.inputs[0], name_to_id, &node.name)?;
            let scale = resolve_input(&node.inputs[1], name_to_id, &node.name)?;
            let eps = attrs.f("epsilon", 1e-5);
            let x_shape = get_shape(&node.inputs[0], shapes);
            let x = as_matrix(graph, x, &x_shape)?;
            let scale = view(graph, scale, &[x_shape.last().copied().unwrap_or(1)])?;
            let out = graph.rms_norm(x, scale, eps);
            register_output(node, 0, out, &x_shape, name_to_id, shapes);
        }

        // --- Embedding (Gather with axis=0) ---
        "Gather" => {
            let data = resolve_input(&node.inputs[0], name_to_id, &node.name)?;
            let indices = resolve_input(&node.inputs[1], name_to_id, &node.name)?;
            let data_shape = get_shape(&node.inputs[0], shapes);
            let indices_shape = get_shape(&node.inputs[1], shapes);
            let rank = data_shape.len() as i64;
            let axis = attrs.i("axis", 0);
            let axis = if axis < 0 { axis + rank } else { axis };
            if !(0..rank).contains(&axis) {
                return Err(OnnxError::ShapeError(format!(
                    "node '{}': Gather axis {axis} of {data_shape:?}",
                    node.name
                )));
            }
            let axis = axis as usize;
            // Constant indices pick along any axis. (Constant data with them
            // folded before translation.)
            if let Some(picks) = constant_values(graph, indices, weights) {
                let dim = data_shape[axis] as i64;
                let positions = picks
                    .iter()
                    .map(|&pick| {
                        let pick = pick as i64;
                        let pick = if pick < 0 { pick + dim } else { pick };
                        usize::try_from(pick).ok().filter(|&p| p < dim as usize)
                    })
                    .collect::<Option<Vec<usize>>>()
                    .ok_or_else(|| {
                        OnnxError::ShapeError(format!(
                            "node '{}': Gather indices {picks:?} out of {dim}",
                            node.name
                        ))
                    })?;
                let out = indexing::gather_axis(graph, node, data, &data_shape, axis, &positions)?;
                let mut out_shape = data_shape[..axis].to_vec();
                out_shape.extend_from_slice(&indices_shape);
                out_shape.extend_from_slice(&data_shape[axis + 1..]);
                register_output(node, 0, out, &out_shape, name_to_id, shapes);
                return Ok(());
            }
            // Otherwise an embedding lookup: rows of the table by index.
            if axis != 0 || graph.node(indices).ty.dtype != crate::graph::DType::U32 {
                return Err(OnnxError::UnsupportedOp(format!(
                    "Gather on axis {axis} with {:?} indices (supported: constant folding, \
                     or rows of a table by U32 indices)",
                    graph.node(indices).ty.dtype
                )));
            }
            let out = graph.embedding(indices, data);
            let hidden = data_shape.get(1).copied().unwrap_or(1);
            let seq_len = indices_shape.iter().product::<usize>().max(1);
            register_output(node, 0, out, &[seq_len, hidden], name_to_id, shapes);
        }

        // --- Transpose ---
        "Transpose" => {
            let x = resolve_input(&node.inputs[0], name_to_id, &node.name)?;
            let x_shape = get_shape(&node.inputs[0], shapes);
            let rank = x_shape.len();
            let perm: Vec<usize> = match *attrs.ints("perm") {
                [] => (0..rank).rev().collect(),
                ref perm => perm.iter().map(|&axis| axis as usize).collect(),
            };
            let mut sorted = perm.clone();
            sorted.sort_unstable();
            if sorted != (0..rank).collect::<Vec<_>>() {
                return Err(OnnxError::ShapeError(format!(
                    "node '{}': Transpose of {x_shape:?} by {perm:?}",
                    node.name
                )));
            }
            let out_shape: Vec<usize> = perm.iter().map(|&axis| x_shape[axis]).collect();
            let (x_shape, perm) = merged_permutation(&x_shape, &perm);
            let rank = x_shape.len();
            if rank > 4 {
                return Err(OnnxError::UnsupportedOp(format!(
                    "Transpose of {x_shape:?} by {perm:?} (supported: up to 4 axes, \
                     after merging unit axes and axes that move together)"
                )));
            }
            let x = view(graph, x, &x_shape)?;
            // Swapping the last two axes is the batched transpose kernel.
            let swaps_last_two = rank >= 2
                && perm[..rank - 2]
                    .iter()
                    .enumerate()
                    .all(|(d, &axis)| d == axis)
                && perm[rank - 2] == rank - 1;
            let out = if perm.iter().enumerate().all(|(d, &axis)| d == axis) {
                x
            } else if swaps_last_two {
                graph.transpose(x)
            } else {
                graph.permute(x, &perm)
            };
            register_output(node, 0, out, &out_shape, name_to_id, shapes);
        }

        "Reshape" => {
            let x = resolve_input(&node.inputs[0], name_to_id, &node.name)?;
            let x_shape = get_shape(&node.inputs[0], shapes);
            let total = x_shape.iter().product::<usize>().max(1);

            // Get target shape from the second input (should be a constant)
            let target = node
                .inputs
                .get(1)
                .and_then(|name| name_to_id.get(name))
                .and_then(|&id| constant_values(graph, id, weights));
            let new_shape = if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
                if let Some(shape_data) = target {
                    resolve_reshape_dims(shape_data, &x_shape, total)
                } else {
                    // Shape input might be produced by a Shape/Constant node
                    // For now, pass through with same shape
                    x_shape.clone()
                }
            } else {
                x_shape.clone()
            };

            // Identity — just register the mapping with the new shape
            if !node.outputs.is_empty() {
                name_to_id.insert(node.outputs[0].clone(), x);
                shapes.insert(node.outputs[0].clone(), new_shape);
            }
        }

        "Flatten" => {
            let x = resolve_input(&node.inputs[0], name_to_id, &node.name)?;
            let x_shape = get_shape(&node.inputs[0], shapes);
            let axis = attrs.i("axis", 1) as usize;
            let dim0: usize = x_shape[..axis].iter().product::<usize>().max(1);
            let dim1: usize = x_shape[axis..].iter().product::<usize>().max(1);
            if !node.outputs.is_empty() {
                name_to_id.insert(node.outputs[0].clone(), x);
                shapes.insert(node.outputs[0].clone(), vec![dim0, dim1]);
            }
        }

        "Squeeze" | "Unsqueeze" => {
            // Shape-only: the tensor keeps its data under the new shape.
            let x = resolve_input(&node.inputs[0], name_to_id, &node.name)?;
            let x_shape = get_shape(&node.inputs[0], shapes);
            // Axes are an attribute before opset 13 and an input after.
            let mut axes: Vec<i64> = attrs.ints("axes").to_vec();
            if axes.is_empty()
                && let Some(&id) = node.inputs.get(1).and_then(|name| name_to_id.get(name))
                && let Some(values) = constant_values(graph, id, weights)
            {
                axes = values.iter().map(|&v| v as i64).collect();
            }
            let new_shape = if node.op_type == "Squeeze" {
                let rank = x_shape.len() as i64;
                let squeezed: Vec<usize> = axes
                    .iter()
                    .map(|&a| (if a < 0 { a + rank } else { a }) as usize)
                    .collect();
                x_shape
                    .iter()
                    .enumerate()
                    .filter(|&(d, &dim)| {
                        dim != 1 || !(squeezed.is_empty() || squeezed.contains(&d))
                    })
                    .map(|(_, &dim)| dim)
                    .collect()
            } else {
                // Axes index the output, so insert in ascending order.
                let rank = (x_shape.len() + axes.len()) as i64;
                let mut positions: Vec<usize> = axes
                    .iter()
                    .map(|&a| (if a < 0 { a + rank } else { a }).clamp(0, rank) as usize)
                    .collect();
                positions.sort_unstable();
                let mut s = x_shape.clone();
                for pos in positions {
                    s.insert(pos.min(s.len()), 1);
                }
                s
            };
            if !node.outputs.is_empty() {
                name_to_id.insert(node.outputs[0].clone(), x);
                shapes.insert(node.outputs[0].clone(), new_shape);
            }
        }

        "Expand" => {
            let x = resolve_input(&node.inputs[0], name_to_id, &node.name)?;
            let x_shape = get_shape(&node.inputs[0], shapes);
            let target: Option<Vec<usize>> = node
                .inputs
                .get(1)
                .and_then(|name| name_to_id.get(name))
                .and_then(|&id| constant_values(graph, id, weights))
                .map(|values| values.iter().map(|&v| v.max(0.0) as usize).collect());
            let Some(target) = target else {
                return Err(OnnxError::UnsupportedOp(
                    "Expand: the shape must be a constant".into(),
                ));
            };
            let Some(out_shape) = broadcast_shape(&x_shape, &target) else {
                return Err(OnnxError::ShapeError(format!(
                    "node '{}': cannot expand {x_shape:?} to {target:?}",
                    node.name
                )));
            };
            let mut shape = vec![1; out_shape.len().saturating_sub(x_shape.len())];
            shape.extend_from_slice(&x_shape);
            if shape.len() != out_shape.len()
                || shape
                    .iter()
                    .zip(&out_shape)
                    .any(|(&s, &o)| s != o && s != 1)
            {
                return Err(OnnxError::ShapeError(format!(
                    "node '{}': cannot expand {x_shape:?} to {target:?}",
                    node.name
                )));
            }
            let out = expand_to(graph, x, &shape, &out_shape).ok_or_else(|| {
                OnnxError::UnsupportedOp(format!(
                    "node '{}': Expand of {x_shape:?} to {out_shape:?} needs more than 4 axes",
                    node.name
                ))
            })?;
            register_output(node, 0, out, &out_shape, name_to_id, shapes);
        }

        // --- Identity / Dropout (inference mode) ---
        "Identity" | "Dropout" => {
            let x = resolve_input(&node.inputs[0], name_to_id, &node.name)?;
            let x_shape = get_shape(&node.inputs[0], shapes);
            if !node.outputs.is_empty() {
                name_to_id.insert(node.outputs[0].clone(), x);
                shapes.insert(node.outputs[0].clone(), x_shape);
            }
        }

        // --- Constant ---
        "Constant" => {
            if let Some(tensor) = attrs.tensor("value") {
                let data = tensor.data.clone();
                let shape = tensor.shape.clone();
                let id = graph.constant(data, &shape);
                if !node.outputs.is_empty() {
                    name_to_id.insert(node.outputs[0].clone(), id);
                    shapes.insert(node.outputs[0].clone(), shape);
                }
            }
        }

        // --- Convolutions, resampling ---
        "Conv" => spatial::conv(graph, node, name_to_id, shapes)?,
        "ConvTranspose" => spatial::conv_transpose(graph, node, name_to_id, shapes)?,
        "Resize" => spatial::resize(graph, node, name_to_id, shapes, weights)?,
        "GridSample" => spatial::grid_sample(graph, node, name_to_id, shapes)?,

        // --- TopK ---
        "TopK" => indexing::top_k(graph, node, name_to_id, shapes, weights)?,

        // --- Padding, repetition and element gathers ---
        "Pad" => indexing::pad(graph, node, name_to_id, shapes, weights)?,
        "Tile" => indexing::tile(graph, node, name_to_id, shapes, weights)?,
        "GatherElements" => indexing::gather_elements(graph, node, name_to_id, shapes)?,

        // --- Concat ---
        "Concat" => {
            let mut inputs = node.inputs.iter().filter(|name| !name.is_empty());
            let first = inputs.next().ok_or_else(|| {
                OnnxError::ShapeError(format!("node '{}': empty Concat", node.name))
            })?;
            let mut out = resolve_input(first, name_to_id, &node.name)?;
            let mut out_shape = get_shape(first, shapes);
            let rank = out_shape.len() as i64;
            let axis = attrs.i("axis", 0);
            let axis = if axis < 0 { axis + rank } else { axis };
            if !(0..rank).contains(&axis) {
                return Err(OnnxError::ShapeError(format!(
                    "node '{}': Concat axis {axis} of {out_shape:?}",
                    node.name
                )));
            }
            let axis = axis as usize;
            // Concatenated shape vectors fold at import.
            let names: Vec<&String> = node.inputs.iter().filter(|name| !name.is_empty()).collect();
            if axis == 0 && out_shape.len() == 1 {
                let parts: Option<Vec<Vec<f32>>> = names
                    .iter()
                    .map(|name| {
                        let id = *name_to_id.get(*name)?;
                        constant_values(graph, id, weights).map(<[f32]>::to_vec)
                    })
                    .collect();
                if let Some(parts) = parts {
                    let data = parts.concat();
                    let len = data.len();
                    let id = graph.constant(data, &[len]);
                    register_output(node, 0, id, &[len], name_to_id, shapes);
                    return Ok(());
                }
            }
            for name in inputs {
                let b = resolve_input(name, name_to_id, &node.name)?;
                let b_shape = get_shape(name, shapes);
                let matches = b_shape.len() == out_shape.len()
                    && (0..out_shape.len()).all(|d| d == axis || b_shape[d] == out_shape[d]);
                if !matches {
                    return Err(OnnxError::ShapeError(format!(
                        "node '{}': Concat of {out_shape:?} and {b_shape:?} on axis {axis}",
                        node.name
                    )));
                }
                // The channel-concat kernel is axis-generic: everything
                // before the axis is its batch, everything after its spatial.
                let outer = out_shape[..axis].iter().product::<usize>() as u32;
                let inner = out_shape[axis + 1..].iter().product::<usize>() as u32;
                out = graph.concat(
                    out,
                    b,
                    outer,
                    out_shape[axis] as u32,
                    b_shape[axis] as u32,
                    inner,
                );
                out_shape[axis] += b_shape[axis];
            }
            register_output(node, 0, out, &out_shape, name_to_id, shapes);
        }

        "Slice" => {
            let x = resolve_input(&node.inputs[0], name_to_id, &node.name)?;
            let x_shape = get_shape(&node.inputs[0], shapes);
            let ints = |slot: usize| -> Option<Vec<i64>> {
                let id = *name_to_id.get(node.inputs.get(slot)?)?;
                let values = constant_values(graph, id, weights)?;
                Some(values.iter().map(|&v| v as i64).collect())
            };
            let (Some(starts), Some(ends)) = (ints(1), ints(2)) else {
                return Err(OnnxError::UnsupportedOp(
                    "Slice: starts and ends must be constants".into(),
                ));
            };
            let axes = ints(3).unwrap_or_else(|| (0..starts.len() as i64).collect());
            let steps = ints(4).unwrap_or_else(|| vec![1; starts.len()]);
            if axes.len() != starts.len()
                || ends.len() != starts.len()
                || steps.len() != starts.len()
                || steps.contains(&0)
            {
                return Err(OnnxError::ShapeError(format!(
                    "node '{}': Slice of {starts:?} to {ends:?} on axes {axes:?} in steps {steps:?}",
                    node.name
                )));
            }
            let mut out = x;
            let mut out_shape = x_shape;
            for (((&axis, &start), &end), &step) in axes.iter().zip(&starts).zip(&ends).zip(&steps)
            {
                let rank = out_shape.len() as i64;
                let axis = if axis < 0 { axis + rank } else { axis };
                if !(0..rank).contains(&axis) {
                    return Err(OnnxError::ShapeError(format!(
                        "node '{}': Slice axis {axis} of {out_shape:?}",
                        node.name
                    )));
                }
                let axis = axis as usize;
                let dim = out_shape[axis] as i64;
                // Negative bounds count from the end. A forward slice clamps
                // both to the axis; a backward one may end just before it.
                let wrap = |v: i64| if v < 0 { v.saturating_add(dim) } else { v };
                let (start, end) = if step > 0 {
                    (wrap(start).clamp(0, dim), wrap(end).clamp(0, dim))
                } else {
                    (wrap(start).clamp(0, dim - 1), wrap(end).clamp(-1, dim - 1))
                };
                // A step longer than the axis takes one element, as the axis does.
                let stride = step.unsigned_abs().min(dim as u64) as i64;
                let span = if step > 0 { end - start } else { start - end };
                let count = (span.max(0) + stride - 1) / stride;
                if count == 0 {
                    return Err(OnnxError::UnsupportedOp(format!(
                        "Slice '{}' leaves nothing of axis {axis} of {out_shape:?}",
                        node.name
                    )));
                }
                out = if step == 1 {
                    narrow(graph, out, &out_shape, axis, start as usize..end as usize)
                } else {
                    let step = if step > 0 { stride } else { -stride };
                    let positions: Vec<usize> =
                        (0..count).map(|j| (start + j * step) as usize).collect();
                    indexing::gather_axis(graph, node, out, &out_shape, axis, &positions)?
                };
                out_shape[axis] = count as usize;
            }
            register_output(node, 0, out, &out_shape, name_to_id, shapes);
        }

        "Split" => {
            let x = resolve_input(&node.inputs[0], name_to_id, &node.name)?;
            let x_shape = get_shape(&node.inputs[0], shapes);
            let rank = x_shape.len() as i64;
            let axis = attrs.i("axis", 0);
            let axis = if axis < 0 { axis + rank } else { axis };
            if !(0..rank).contains(&axis) {
                return Err(OnnxError::ShapeError(format!(
                    "node '{}': Split axis {axis} of {x_shape:?}",
                    node.name
                )));
            }
            let axis = axis as usize;
            let dim = x_shape[axis];
            // Sizes are an attribute before opset 13 and an input after.
            let mut sizes: Vec<i64> = attrs.ints("split").to_vec();
            if let Some(&id) = node.inputs.get(1).and_then(|name| name_to_id.get(name)) {
                let Some(values) = constant_values(graph, id, weights) else {
                    return Err(OnnxError::UnsupportedOp(
                        "Split: the sizes must be constants".into(),
                    ));
                };
                sizes = values.iter().map(|&v| v as i64).collect();
            }
            // Without sizes the parts are even, and the last takes what is left.
            if sizes.is_empty() {
                let parts = attrs.i("num_outputs", node.outputs.len() as i64).max(1) as usize;
                let chunk = dim.div_ceil(parts) as i64;
                sizes = vec![chunk; parts];
                if let Some(last) = sizes.last_mut() {
                    *last = dim as i64 - chunk * (parts as i64 - 1);
                }
            }
            if sizes.len() != node.outputs.len()
                || sizes.iter().any(|&size| !(1..=dim as i64).contains(&size))
                || sizes.iter().sum::<i64>() != dim as i64
            {
                return Err(OnnxError::ShapeError(format!(
                    "node '{}': Split of {x_shape:?} on axis {axis} into {sizes:?}",
                    node.name
                )));
            }
            let mut start = 0;
            for (slot, &size) in sizes.iter().enumerate() {
                let end = start + size as usize;
                let out = narrow(graph, x, &x_shape, axis, start..end);
                let mut out_shape = x_shape.clone();
                out_shape[axis] = size as usize;
                register_output(node, slot, out, &out_shape, name_to_id, shapes);
                start = end;
            }
        }

        // --- GroupNorm ---
        "GroupNormalization" | "GroupNorm" => {
            let x = resolve_input(&node.inputs[0], name_to_id, &node.name)?;
            let scale = resolve_input(&node.inputs[1], name_to_id, &node.name)?;
            let bias = resolve_input(&node.inputs[2], name_to_id, &node.name)?;
            let x_shape = get_shape(&node.inputs[0], shapes);
            let num_groups = attrs.i("num_groups", 32) as u32;
            let eps = attrs.f("epsilon", 1e-5);

            if x_shape.len() == 4 {
                let batch = x_shape[0] as u32;
                let channels = x_shape[1] as u32;
                let spatial = (x_shape[2] * x_shape[3]) as u32;
                let out =
                    graph.group_norm(x, scale, bias, batch, channels, spatial, num_groups, eps);
                register_output(node, 0, out, &x_shape, name_to_id, shapes);
            } else {
                return Err(OnnxError::UnsupportedOp(
                    "GroupNorm: only 4D (NCHW) input supported".into(),
                ));
            }
        }

        // --- BatchNormalization (inference mode) ---
        // With running statistics it is `x * w + b` per channel, where
        // w = scale / sqrt(var + eps) and b = bias - mean * w.
        "BatchNormalization" => {
            let x = resolve_input(&node.inputs[0], name_to_id, &node.name)?;
            let x_shape = get_shape(&node.inputs[0], shapes);
            let eps = attrs.f("epsilon", 1e-5);
            if x_shape.len() < 2 || attrs.i("training_mode", 0) != 0 {
                return Err(OnnxError::UnsupportedOp(format!(
                    "BatchNormalization of {x_shape:?} (supported: inference over [N, C, ...])"
                )));
            }
            let (batch, channels) = (x_shape[0], x_shape[1]);
            let spatial = x_shape[2..].iter().product::<usize>();
            let stat = |slot: usize| {
                let id = *name_to_id.get(node.inputs.get(slot)?)?;
                constant_values(graph, id, weights).filter(|values| values.len() == channels)
            };
            let (Some(scale), Some(bias), Some(mean), Some(var)) =
                (stat(1), stat(2), stat(3), stat(4))
            else {
                return Err(OnnxError::UnsupportedOp(format!(
                    "BatchNormalization '{}': scale, bias, mean and variance must be \
                     constants of {channels} values",
                    node.name
                )));
            };
            let w: Vec<f32> = (0..channels)
                .map(|c| scale[c] / (var[c] + eps).sqrt())
                .collect();
            let b: Vec<f32> = (0..channels).map(|c| bias[c] - mean[c] * w[c]).collect();
            // The scale kernel takes a value per image and channel.
            let w = graph.constant(w.repeat(batch), &[batch * channels]);
            let b = graph.constant(b, &[channels]);
            let x = view(graph, x, &[batch * channels * spatial])?;
            let scaled = graph.mul_per_channel(x, w, channels as u32, spatial as u32);
            let out = graph.add_per_channel(scaled, b, channels as u32, spatial as u32);
            register_output(node, 0, out, &x_shape, name_to_id, shapes);
        }

        // --- MaxPool ---
        "MaxPool" => {
            let input = resolve_input(&node.inputs[0], name_to_id, &node.name)?;
            let input_shape = get_shape(&node.inputs[0], shapes);
            let kernel: Vec<usize> = attrs
                .ints("kernel_shape")
                .iter()
                .map(|&k| k.max(0) as usize)
                .collect();
            if input_shape.len() != 4 || kernel.len() != 2 {
                return Err(OnnxError::UnsupportedOp(format!(
                    "MaxPool of {input_shape:?} by {kernel:?} (supported: 2D)"
                )));
            }
            let (stride, pads) = window(node, &input_shape[2..], &kernel)?;
            if pads[0] != pads[1] {
                return Err(OnnxError::UnsupportedOp(format!(
                    "MaxPool '{}' padded by {pads:?} (supported: the same on every axis)",
                    node.name
                )));
            }
            let [batch, channels, in_h, in_w] = input_shape[..] else {
                unreachable!()
            };
            // Rounding up adds a last window that overhangs the end, unless
            // it would start past the padding, as PyTorch counts it.
            let ceil_mode = attrs.i("ceil_mode", 0) != 0;
            let windows = |extent: usize, k: usize| {
                let span = extent + 2 * pads[0] - k;
                let mut out = span / stride + 1;
                if ceil_mode
                    && !span.is_multiple_of(stride)
                    && span / stride * stride + stride < extent + pads[0]
                {
                    out += 1;
                }
                out
            };
            let (out_h, out_w) = (windows(in_h, kernel[0]), windows(in_w, kernel[1]));
            // Windows past the end read -MAX, which never wins a maximum.
            let overhang = |extent: usize, k: usize, out: usize| {
                ((out - 1) * stride + k).saturating_sub(extent + 2 * pads[0]) as i64
            };
            let mut input = input;
            let mut padded = input_shape.clone();
            for (axis, extra) in [
                (2, overhang(in_h, kernel[0], out_h)),
                (3, overhang(in_w, kernel[1], out_w)),
            ] {
                if extra > 0 {
                    input =
                        indexing::pad_constant(graph, input, &padded, axis, [0, extra], -f32::MAX)?;
                    padded[axis] += extra as usize;
                }
            }
            let input = view(graph, input, &[batch * channels * padded[2] * padded[3]])?;
            let out = graph.max_pool_2d(
                input,
                batch as u32,
                channels as u32,
                padded[2] as u32,
                padded[3] as u32,
                kernel[0] as u32,
                kernel[1] as u32,
                stride as u32,
                pads[0] as u32,
            );
            let out_shape = vec![batch, channels, out_h, out_w];
            register_output(node, 0, out, &out_shape, name_to_id, shapes);
        }

        // --- GlobalAveragePool ---
        "GlobalAveragePool" => {
            let input = resolve_input(&node.inputs[0], name_to_id, &node.name)?;
            let input_shape = get_shape(&node.inputs[0], shapes);
            if input_shape.len() != 4 {
                return Err(OnnxError::UnsupportedOp(
                    "GlobalAveragePool: only 4D (NCHW) supported".into(),
                ));
            }

            let batch = input_shape[0] as u32;
            let channels = input_shape[1] as u32;
            let spatial = (input_shape[2] * input_shape[3]) as u32;

            let input = view(graph, input, &[input_shape.iter().product()])?;
            let out = graph.global_avg_pool(input, batch, channels, spatial);
            let out_shape = vec![input_shape[0], input_shape[1], 1, 1];
            register_output(node, 0, out, &out_shape, name_to_id, shapes);
        }

        // --- Unsupported ops produce a clear error ---
        other => {
            return Err(OnnxError::UnsupportedOp(other.to_string()));
        }
    }

    Ok(())
}

// --- Helpers ---

#[derive(Clone, Copy)]
enum BinaryKind {
    Add,
    Sub,
    Mul,
    Div,
}

fn binary_op(
    graph: &mut Graph,
    node: &proto::Node,
    name_to_id: &mut HashMap<String, NodeId>,
    shapes: &mut HashMap<String, Vec<usize>>,
    weights: &HashMap<String, Vec<f32>>,
    kind: BinaryKind,
) -> Result<(), OnnxError> {
    let a = resolve_input(&node.inputs[0], name_to_id, &node.name)?;
    let b = resolve_input(&node.inputs[1], name_to_id, &node.name)?;
    let a_shape = get_shape(&node.inputs[0], shapes);
    let b_shape = get_shape(&node.inputs[1], shapes);
    let (out, out_shape) = broadcast_binary(graph, (a, &a_shape), (b, &b_shape), kind, weights)?
        .ok_or_else(|| {
            OnnxError::ShapeError(format!(
                "node '{}': cannot broadcast {a_shape:?} with {b_shape:?}",
                node.name
            ))
        })?;
    register_output(node, 0, out, &out_shape, name_to_id, shapes);
    Ok(())
}

/// `id` with `shape`, through a zero-cost reshape when its graph shape
/// differs. The importer tracks ONNX shapes and lets graph nodes carry any
/// view with the same element count.
fn view(graph: &mut Graph, id: NodeId, shape: &[usize]) -> Result<NodeId, OnnxError> {
    let current = &graph.node(id).ty;
    if current.shape == shape {
        return Ok(id);
    }
    if current.num_elements() != shape.iter().product::<usize>() {
        return Err(OnnxError::ShapeError(format!(
            "cannot view {:?} as {shape:?}",
            current.shape
        )));
    }
    // One view per (tensor, shape): rewrite rules see a tensor used twice
    // through views as the same value only when both uses share the node.
    let existing = graph.nodes()[id as usize..].iter().find(|node| {
        matches!(node.op, Op::Identity) && node.inputs == [id] && node.ty.shape == shape
    });
    if let Some(node) = existing {
        return Ok(node.id);
    }
    Ok(graph.reshape(id, shape))
}

/// `id` as a matrix of rows along its last axis.
fn as_matrix(graph: &mut Graph, id: NodeId, shape: &[usize]) -> Result<NodeId, OnnxError> {
    let cols = shape.last().copied().unwrap_or(1).max(1);
    let rows = shape.iter().product::<usize>() / cols;
    view(graph, id, &[rows, cols])
}

/// NumPy-style matrix product: a shared `[K, N]` right side multiplies
/// every row of `a`, and equal leading batch axes multiply per batch. A
/// one-dimensional side is a row or column vector whose axis the result
/// drops.
fn matmul_nd(
    graph: &mut Graph,
    a: NodeId,
    a_shape: &[usize],
    b: NodeId,
    b_shape: &[usize],
) -> Result<Option<(NodeId, Vec<usize>)>, OnnxError> {
    if let [k] = *b_shape {
        let Some((out, mut shape)) = matmul_nd(graph, a, a_shape, b, &[k, 1])? else {
            return Ok(None);
        };
        shape.pop();
        return Ok(Some((out, shape)));
    }
    if let [k] = *a_shape {
        let Some((out, mut shape)) = matmul_nd(graph, a, &[1, k], b, b_shape)? else {
            return Ok(None);
        };
        shape.remove(shape.len() - 2);
        return Ok(Some((out, shape)));
    }
    let (ra, rb) = (a_shape.len(), b_shape.len());
    if ra < 2 || rb < 2 || a_shape[ra - 1] != b_shape[rb - 2] {
        return Ok(None);
    }
    let (m, k, n) = (a_shape[ra - 2], a_shape[ra - 1], b_shape[rb - 1]);
    let b_batch: usize = b_shape[..rb - 2].iter().product();
    if b_batch == 1 {
        let rows = a_shape[..ra - 1].iter().product();
        let a = view(graph, a, &[rows, k])?;
        let b = view(graph, b, &[k, n])?;
        let mut shape = a_shape[..ra - 1].to_vec();
        shape.push(n);
        return Ok(Some((graph.matmul(a, b), shape)));
    }
    let strip = |s: &[usize]| {
        s.iter()
            .skip_while(|&&d| d == 1)
            .copied()
            .collect::<Vec<_>>()
    };
    if strip(&a_shape[..ra - 2]) != strip(&b_shape[..rb - 2]) {
        return Ok(None);
    }
    let lead = if ra >= rb {
        &a_shape[..ra - 2]
    } else {
        &b_shape[..rb - 2]
    };
    let a = view(graph, a, &[b_batch, m, k])?;
    let b = view(graph, b, &[b_batch, k, n])?;
    let mut shape = lead.to_vec();
    shape.extend([m, n]);
    Ok(Some((graph.batch_matmul(a, b), shape)))
}

/// NumPy-broadcast `a ∘ b`, or `None` when the shapes are incompatible or
/// need more than four alternating broadcast axes. A folded scalar becomes a
/// kernel constant, equal shapes run elementwise, and a narrow operand that
/// is a row per matrix row or a trailing block uses the dedicated broadcast
/// ops; anything else broadcasts each operand to the result shape first.
/// Returns the result and its shape.
fn broadcast_binary(
    graph: &mut Graph,
    (a, a_shape): (NodeId, &[usize]),
    (b, b_shape): (NodeId, &[usize]),
    kind: BinaryKind,
    weights: &HashMap<String, Vec<f32>>,
) -> Result<Option<(NodeId, Vec<usize>)>, OnnxError> {
    let Some(out_shape) = broadcast_shape(a_shape, b_shape) else {
        return Ok(None);
    };
    let rank = out_shape.len();
    let padded = |shape: &[usize]| {
        let mut p = vec![1; rank - shape.len()];
        p.extend_from_slice(shape);
        p
    };
    let (pa, pb) = (padded(a_shape), padded(b_shape));
    let (a_full, b_full) = (pa == out_shape, pb == out_shape);
    let elementwise = |graph: &mut Graph, a: NodeId, b: NodeId| match kind {
        BinaryKind::Add => graph.add(a, b),
        BinaryKind::Sub => graph.sub(a, b),
        BinaryKind::Mul => graph.mul(a, b),
        BinaryKind::Div => graph.div(a, b),
    };
    // Folded scalars become kernel constants, which must be finite.
    let foldable =
        |v: f32| v.is_finite() && (!matches!(kind, BinaryKind::Div) || (1.0 / v).is_finite());
    if a_full
        && pb.iter().all(|&d| d == 1)
        && let Some(v) = scalar_constant(graph, b, weights)
        && foldable(v)
    {
        let a = as_matrix(graph, a, &out_shape)?;
        let out = match kind {
            BinaryKind::Add => graph.add_scalar(a, v),
            BinaryKind::Sub => graph.add_scalar(a, -v),
            BinaryKind::Mul => graph.scale(a, v),
            BinaryKind::Div => graph.scale(a, 1.0 / v),
        };
        return Ok(Some((out, out_shape)));
    }
    if b_full
        && pa.iter().all(|&d| d == 1)
        && let Some(v) = scalar_constant(graph, a, weights)
        && v.is_finite()
    {
        let b = as_matrix(graph, b, &out_shape)?;
        let out = match kind {
            BinaryKind::Add => graph.add_scalar(b, v),
            BinaryKind::Sub => {
                let negated = graph.neg(b);
                graph.add_scalar(negated, v)
            }
            BinaryKind::Mul => graph.scale(b, v),
            BinaryKind::Div => {
                let inverse = graph.recip(b);
                graph.scale(inverse, v)
            }
        };
        return Ok(Some((out, out_shape)));
    }
    if a_full && b_full {
        // Both on the canonical matrix view, so every elementwise use of a
        // tensor reads the same node.
        let a = as_matrix(graph, a, &out_shape)?;
        let b = as_matrix(graph, b, &out_shape)?;
        return Ok(Some((elementwise(graph, a, b), out_shape)));
    }
    let commutes = matches!(kind, BinaryKind::Add | BinaryKind::Mul);
    if b_full && !a_full && commutes {
        return broadcast_binary(graph, (b, b_shape), (a, a_shape), kind, weights);
    }
    let total = out_shape.iter().product::<usize>();
    if a_full && rank > 0 {
        let cols = out_shape[rank - 1];
        let b_len = pb.iter().product::<usize>();
        // One value per row: the result's shape with the last axis reduced.
        if pb[..rank - 1] == out_shape[..rank - 1] && pb[rank - 1] == 1 && cols > 0 {
            let a = view(graph, a, &[total / cols, cols])?;
            let b = view(graph, b, &[total / cols, 1])?;
            // Divide by the reciprocal of the narrow side before broadcasting.
            let b = match kind {
                BinaryKind::Div => graph.recip(b),
                _ => b,
            };
            let b = graph.broadcast_inner(b, cols);
            let out = match kind {
                BinaryKind::Add => graph.add(a, b),
                BinaryKind::Sub => graph.sub(a, b),
                BinaryKind::Mul | BinaryKind::Div => graph.mul(a, b),
            };
            return Ok(Some((out, out_shape)));
        }
        // Trailing axes: `b` repeats along the leading ones.
        let leading = pb.iter().take_while(|&&d| d == 1).count();
        if pb[leading..] == out_shape[leading..] && b_len > 0 {
            let a = view(graph, a, &[total / b_len, b_len])?;
            let b = view(graph, b, &[b_len])?;
            let b = match kind {
                BinaryKind::Sub => graph.neg(b),
                BinaryKind::Div => graph.recip(b),
                _ => b,
            };
            let out = match kind {
                BinaryKind::Add | BinaryKind::Sub => graph.bias_add(a, b),
                BinaryKind::Mul | BinaryKind::Div => graph.bias_mul(a, b),
            };
            return Ok(Some((out, out_shape)));
        }
    }
    // Anything else: each operand broadcast to the result's shape.
    let (Some(a), Some(b)) = (
        expand_to(graph, a, &pa, &out_shape),
        expand_to(graph, b, &pb, &out_shape),
    ) else {
        return Ok(None);
    };
    let a = as_matrix(graph, a, &out_shape)?;
    let b = as_matrix(graph, b, &out_shape)?;
    Ok(Some((elementwise(graph, a, b), out_shape)))
}

/// The values of a constant: an initializer or a `Constant` node.
fn constant_values<'a>(
    graph: &'a Graph,
    id: NodeId,
    weights: &'a HashMap<String, Vec<f32>>,
) -> Option<&'a [f32]> {
    match graph.node(id).op {
        Op::Constant { ref data } => Some(data),
        Op::Parameter { ref name } => weights.get(name).map(Vec::as_slice),
        // A reshape is a view of the same values.
        Op::Identity => constant_values(graph, graph.node(id).inputs[0], weights),
        _ => None,
    }
}

/// The value of a one-element constant.
fn scalar_constant(graph: &Graph, id: NodeId, weights: &HashMap<String, Vec<f32>>) -> Option<f32> {
    match *constant_values(graph, id, weights)? {
        [v] => Some(v),
        _ => None,
    }
}

/// `ReduceMean`, `ReduceSum` or `ReduceMax` over the last axis, kept or
/// dropped: the per-row reductions norms, softmax and query selection are
/// written with.
fn reduce_op(
    graph: &mut Graph,
    node: &proto::Node,
    name_to_id: &mut HashMap<String, NodeId>,
    shapes: &mut HashMap<String, Vec<usize>>,
    weights: &HashMap<String, Vec<f32>>,
) -> Result<(), OnnxError> {
    let x = resolve_input(&node.inputs[0], name_to_id, &node.name)?;
    let x_shape = get_shape(&node.inputs[0], shapes);
    let rank = x_shape.len() as i64;
    // Axes are an attribute before opset 18 (13 for ReduceSum), an input after.
    let mut axes: Vec<i64> = node.attrs.ints("axes").to_vec();
    if axes.is_empty()
        && let Some(&id) = node.inputs.get(1).and_then(|name| name_to_id.get(name))
        && let Some(values) = constant_values(graph, id, weights)
    {
        axes = values.iter().map(|&v| v as i64).collect();
    }
    let last_axis = matches!(*axes.as_slice(), [axis] if axis == -1 || axis == rank - 1);
    if !last_axis {
        return Err(OnnxError::UnsupportedOp(format!(
            "{}: axes {axes:?} of {x_shape:?} (supported: the last axis)",
            node.op_type
        )));
    }
    let x = as_matrix(graph, x, &x_shape)?;
    let out = match node.op_type.as_str() {
        "ReduceMean" => graph.mean_inner(x),
        "ReduceSum" => graph.sum_inner(x),
        _ => graph.max_inner(x),
    };
    let mut out_shape = x_shape;
    if node.attrs.i("keepdims", 1) != 0 {
        if let Some(last) = out_shape.last_mut() {
            *last = 1;
        }
    } else {
        out_shape.pop();
    }
    register_output(node, 0, out, &out_shape, name_to_id, shapes);
    Ok(())
}

fn unary_op(
    graph: &mut Graph,
    node: &proto::Node,
    name_to_id: &mut HashMap<String, NodeId>,
    shapes: &mut HashMap<String, Vec<usize>>,
    op: Op,
) -> Result<(), OnnxError> {
    let x = resolve_input(&node.inputs[0], name_to_id, &node.name)?;
    let x_shape = get_shape(&node.inputs[0], shapes);
    // Elementwise: keep whatever view the input has, so no reshape lands
    // inside a pattern the optimizer would fold.
    let ty = graph.node(x).ty.clone();
    let out = graph.add_raw_node(op, vec![x], ty);
    register_output(node, 0, out, &x_shape, name_to_id, shapes);
    Ok(())
}

fn register_output(
    node: &proto::Node,
    output_idx: usize,
    id: NodeId,
    shape: &[usize],
    name_to_id: &mut HashMap<String, NodeId>,
    shapes: &mut HashMap<String, Vec<usize>>,
) {
    if let Some(name) = node.outputs.get(output_idx) {
        if !name.is_empty() {
            name_to_id.insert(name.clone(), id);
            shapes.insert(name.clone(), shape.to_vec());
        }
    }
}

/// The stride and per-axis padding of a convolution or pooling `kernel`
/// over `extents`, in the form the GPU kernels take: one stride for every
/// axis, no dilation, and the same padding before and after each axis.
fn window(
    node: &proto::Node,
    extents: &[usize],
    kernel: &[usize],
) -> Result<(usize, Vec<usize>), OnnxError> {
    let attrs = &node.attrs;
    let rank = extents.len();
    let unsupported = |what: String| {
        OnnxError::UnsupportedOp(format!("{} '{}': {what}", node.op_type, node.name))
    };
    let strides = match *attrs.ints("strides") {
        [] => vec![1; rank],
        ref strides => strides.to_vec(),
    };
    let stride = strides.first().copied().unwrap_or(1);
    if strides.len() != rank
        || !(1..=i64::from(u32::MAX)).contains(&stride)
        || strides.iter().any(|&s| s != stride)
    {
        return Err(unsupported(format!(
            "strides {strides:?} (supported: the same on every axis)"
        )));
    }
    let stride = stride as usize;
    if attrs.ints("dilations").iter().any(|&d| d != 1) {
        return Err(unsupported("dilation (supported: none)".into()));
    }
    let pads: Vec<i64> = match attrs.s("auto_pad").unwrap_or("NOTSET") {
        "NOTSET" => match *attrs.ints("pads") {
            [] => vec![0; 2 * rank],
            ref pads => pads.to_vec(),
        },
        "VALID" => vec![0; 2 * rank],
        // Enough padding for `ceil(extent / stride)` outputs, split between
        // the sides; an odd total puts the extra one on one side.
        "SAME_UPPER" | "SAME_LOWER" => {
            let total: Vec<i64> = extents
                .iter()
                .zip(kernel)
                .map(|(&e, &k)| {
                    let reach = ((e.div_ceil(stride).max(1) - 1) * stride).saturating_add(k);
                    reach.saturating_sub(e) as i64
                })
                .collect();
            let mut pads: Vec<i64> = total.iter().map(|&t| t / 2).collect();
            pads.extend(total.iter().map(|&t| t - t / 2));
            pads
        }
        other => return Err(unsupported(format!("auto_pad {other}"))),
    };
    if pads.len() != 2 * rank || (0..rank).any(|d| pads[d] < 0 || pads[d] != pads[rank + d]) {
        return Err(unsupported(format!(
            "pads {pads:?} (supported: the same on both sides of an axis)"
        )));
    }
    let pads: Vec<usize> = pads[..rank].iter().map(|&p| p as usize).collect();
    for ((&extent, &k), &pad) in extents.iter().zip(kernel).zip(&pads) {
        let padded = extent.saturating_add(pad.saturating_mul(2));
        if k == 0 || padded < k || padded > u32::MAX as usize {
            return Err(OnnxError::ShapeError(format!(
                "node '{}': window {kernel:?} over {extents:?} padded by {pads:?}",
                node.name
            )));
        }
    }
    Ok((stride, pads))
}

/// The same permutation of `shape` with fewer axes: unit axes dropped, and
/// neighbouring axes that stay neighbours in order merged into one.
fn merged_permutation(shape: &[usize], perm: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let kept: Vec<usize> = (0..shape.len()).filter(|&a| shape[a] > 1).collect();
    let position = |axis: usize| kept.iter().position(|&a| a == axis);
    // Runs of kept input axes, in output order.
    let mut runs: Vec<std::ops::Range<usize>> = Vec::new();
    for p in perm.iter().filter_map(|&axis| position(axis)) {
        match runs.last_mut() {
            Some(run) if run.end == p => run.end += 1,
            _ => runs.push(p..p + 1),
        }
    }
    let mut by_input: Vec<usize> = (0..runs.len()).collect();
    by_input.sort_by_key(|&r| runs[r].start);
    let merged_shape = by_input
        .iter()
        .map(|&r| runs[r].clone().map(|p| shape[kept[p]]).product())
        .collect();
    let merged_perm = (0..runs.len())
        .map(|r| by_input.iter().position(|&i| i == r).unwrap())
        .collect();
    (merged_shape, merged_perm)
}

/// The `range` of `x` along `axis` of its `shape`.
fn narrow(
    graph: &mut Graph,
    x: NodeId,
    shape: &[usize],
    axis: usize,
    range: std::ops::Range<usize>,
) -> NodeId {
    let outer = shape[..axis].iter().product::<usize>() as u32;
    let inner = shape[axis + 1..].iter().product::<usize>() as u32;
    let (dim, start, end) = (shape[axis] as u32, range.start as u32, range.end as u32);
    // Drop the leading `start`, then keep the first `end - start`.
    let mut out = x;
    if start > 0 {
        out = graph.split_b(out, outer, start, dim - start, inner);
    }
    if end < dim {
        out = graph.split_a(out, outer, end - start, dim - end, inner);
    }
    out
}

/// `x` of `shape` broadcast to `out_shape` by one broadcast; the ranks
/// match and each axis of `shape` is equal or one. Neighbouring axes that
/// are both repeated, or both kept, merge, so up to four alternating groups
/// fit; past that, `None`.
fn expand_to(graph: &mut Graph, x: NodeId, shape: &[usize], out_shape: &[usize]) -> Option<NodeId> {
    if shape == out_shape {
        return Some(x);
    }
    let (mut from, mut to): (Vec<usize>, Vec<usize>) = (Vec::new(), Vec::new());
    let mut last = None;
    for (&s, &o) in shape.iter().zip(out_shape) {
        if o == 1 {
            continue;
        }
        let repeated = s != o;
        if last == Some(repeated) {
            *from.last_mut()? *= s;
            *to.last_mut()? *= o;
        } else {
            from.push(s);
            to.push(o);
            last = Some(repeated);
        }
    }
    if to.len() > 4 {
        return None;
    }
    let x = view(graph, x, &from).ok()?;
    Some(graph.broadcast_to(x, &to))
}

/// Compute the broadcast output shape (NumPy-style).
fn broadcast_shape(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let len = a.len().max(b.len());
    let mut result = vec![1; len];
    for i in 0..len {
        let da = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let db = if i < b.len() { b[b.len() - 1 - i] } else { 1 };
        if da != db && da != 1 && db != 1 {
            return None;
        }
        result[len - 1 - i] = if da == 1 { db } else { da };
    }
    Some(result)
}

/// Resolve ONNX reshape target dims (handling -1 and 0).
fn resolve_reshape_dims(shape_data: &[f32], input: &[usize], total_elements: usize) -> Vec<usize> {
    // `0` copies the input's dimension at that index; `-1` takes the rest.
    let mut result: Vec<usize> = shape_data
        .iter()
        .enumerate()
        .map(|(i, &d)| match d as i64 {
            0 => input.get(i).copied().unwrap_or(1),
            d if d < 0 => 1,
            d => d as usize,
        })
        .collect();
    if let Some(idx) = shape_data.iter().position(|&d| d as i64 == -1) {
        let known = result.iter().product::<usize>().max(1);
        result[idx] = total_elements / known;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_shape() {
        assert_eq!(broadcast_shape(&[3, 4], &[4]), Some(vec![3, 4]));
        assert_eq!(broadcast_shape(&[1, 4], &[3, 4]), Some(vec![3, 4]));
        assert_eq!(broadcast_shape(&[2, 1], &[1, 3]), Some(vec![2, 3]));
        assert_eq!(broadcast_shape(&[0, 1], &[1, 3]), Some(vec![0, 3]));
        assert_eq!(broadcast_shape(&[2, 3], &[3, 2]), None);
    }

    #[test]
    fn test_resolve_reshape_dims() {
        assert_eq!(resolve_reshape_dims(&[2.0, -1.0], &[6], 6), vec![2, 3]);
        assert_eq!(resolve_reshape_dims(&[3.0, 4.0], &[12], 12), vec![3, 4]);
        assert_eq!(
            resolve_reshape_dims(&[0.0, -1.0, 2.0], &[3, 8], 24),
            vec![3, 4, 2]
        );
    }

    #[test]
    fn test_flatten_to_2d() {
        assert_eq!(flatten_to_2d(&[2, 3, 4]), vec![6, 4]);
        assert_eq!(flatten_to_2d(&[5, 10]), vec![5, 10]);
        assert_eq!(flatten_to_2d(&[10]), vec![10]);
    }

    // ─── Protobuf encoding helpers for building test ONNX models ───

    fn pb_varint(mut val: u64) -> Vec<u8> {
        let mut buf = Vec::new();
        loop {
            let byte = (val & 0x7F) as u8;
            val >>= 7;
            if val == 0 {
                buf.push(byte);
                break;
            }
            buf.push(byte | 0x80);
        }
        buf
    }

    fn pb_field_varint(field: u32, val: u64) -> Vec<u8> {
        let mut buf = pb_varint((field as u64) << 3);
        buf.extend(pb_varint(val));
        buf
    }

    fn pb_field_bytes(field: u32, data: &[u8]) -> Vec<u8> {
        let mut buf = pb_varint(((field as u64) << 3) | 2);
        buf.extend(pb_varint(data.len() as u64));
        buf.extend(data);
        buf
    }

    fn pb_field_f32(field: u32, val: f32) -> Vec<u8> {
        let mut buf = pb_varint(((field as u64) << 3) | 5);
        buf.extend(val.to_le_bytes());
        buf
    }

    /// Build a TensorProto with inline float data.
    fn build_tensor_proto(name: &str, dims: &[i64], data: &[f32]) -> Vec<u8> {
        let mut t = Vec::new();
        // dims (field 1, packed)
        let mut dims_packed = Vec::new();
        for &d in dims {
            dims_packed.extend(pb_varint(d as u64));
        }
        t.extend(pb_field_bytes(1, &dims_packed));
        // data_type = 1 (float32)
        t.extend(pb_field_varint(2, 1));
        // name (field 8)
        t.extend(pb_field_bytes(8, name.as_bytes()));
        // raw_data (field 9): float32 LE bytes
        let raw: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        t.extend(pb_field_bytes(9, &raw));
        t
    }

    /// Build a ValueInfoProto with tensor type and shape.
    fn build_value_info(name: &str, dims: &[i64]) -> Vec<u8> {
        // Dimension messages
        let mut shape_proto = Vec::new();
        for &d in dims {
            let dim_msg = pb_field_varint(1, d as u64);
            shape_proto.extend(pb_field_bytes(1, &dim_msg));
        }
        // TensorTypeProto: field 1 = elem_type (1=float), field 2 = shape
        let mut tensor_type = pb_field_varint(1, 1);
        tensor_type.extend(pb_field_bytes(2, &shape_proto));
        // TypeProto: field 1 = tensor_type
        let type_proto = pb_field_bytes(1, &tensor_type);
        // ValueInfoProto: field 1 = name, field 2 = type
        let mut vi = pb_field_bytes(1, name.as_bytes());
        vi.extend(pb_field_bytes(2, &type_proto));
        vi
    }

    /// Build a NodeProto.
    fn build_node_proto(
        op_type: &str,
        inputs: &[&str],
        outputs: &[&str],
        attrs: &[(&str, i64)], // int attributes only for simplicity
        float_attrs: &[(&str, f32)],
    ) -> Vec<u8> {
        let mut n = Vec::new();
        for inp in inputs {
            n.extend(pb_field_bytes(1, inp.as_bytes()));
        }
        for out in outputs {
            n.extend(pb_field_bytes(2, out.as_bytes()));
        }
        n.extend(pb_field_bytes(4, op_type.as_bytes()));
        for &(name, val) in attrs {
            let mut attr = pb_field_bytes(1, name.as_bytes());
            attr.extend(pb_field_varint(3, val as u64));
            attr.extend(pb_field_varint(20, 2)); // attr_type = INT
            n.extend(pb_field_bytes(5, &attr));
        }
        for &(name, val) in float_attrs {
            let mut attr = pb_field_bytes(1, name.as_bytes());
            attr.extend(pb_field_f32(2, val));
            attr.extend(pb_field_varint(20, 1)); // attr_type = FLOAT
            n.extend(pb_field_bytes(5, &attr));
        }
        n
    }

    /// Build a complete ONNX ModelProto from graph components.
    fn build_onnx_model(
        nodes: &[Vec<u8>],
        initializers: &[Vec<u8>],
        inputs: &[Vec<u8>],
        outputs: &[Vec<u8>],
    ) -> Vec<u8> {
        let mut graph = Vec::new();
        for node in nodes {
            graph.extend(pb_field_bytes(1, node));
        }
        for init in initializers {
            graph.extend(pb_field_bytes(5, init));
        }
        for inp in inputs {
            graph.extend(pb_field_bytes(11, inp));
        }
        for out in outputs {
            graph.extend(pb_field_bytes(12, out));
        }

        let mut model = pb_field_varint(1, 8); // ir_version
        // opset: version=17 (default domain)
        let opset = pb_field_varint(2, 17);
        model.extend(pb_field_bytes(8, &opset));
        model.extend(pb_field_bytes(7, &graph));
        model
    }

    #[test]
    fn test_load_simple_gemm_relu() {
        // Model: Gemm(x, weight, bias, transB=1) -> Relu -> output
        // x: [1, 4], weight: [3, 4], bias: [3] -> output: [1, 3]
        let weight_data: Vec<f32> = (0..12).map(|i| i as f32 * 0.1).collect();
        let bias_data = vec![0.1, 0.2, 0.3];

        let weight_init = build_tensor_proto("weight", &[3, 4], &weight_data);
        let bias_init = build_tensor_proto("bias", &[3], &bias_data);

        let gemm_node = build_node_proto(
            "Gemm",
            &["x", "weight", "bias"],
            &["gemm_out"],
            &[("transB", 1)],
            &[],
        );
        let relu_node = build_node_proto("Relu", &["gemm_out"], &["output"], &[], &[]);

        let x_vi = build_value_info("x", &[1, 4]);
        let weight_vi = build_value_info("weight", &[3, 4]);
        let bias_vi = build_value_info("bias", &[3]);
        let output_vi = build_value_info("output", &[1, 3]);

        let model_bytes = build_onnx_model(
            &[gemm_node, relu_node],
            &[weight_init, bias_init],
            &[x_vi, weight_vi, bias_vi],
            &[output_vi],
        );

        let result = load_onnx_bytes(&model_bytes, None);
        assert!(result.is_ok(), "load failed: {:?}", result.err());

        let onnx_model = result.unwrap();
        assert_eq!(onnx_model.weights.len(), 2);
        assert!(onnx_model.weights.contains_key("weight"));
        assert!(onnx_model.weights.contains_key("bias"));
        assert_eq!(onnx_model.weights["weight"].len(), 12);
        assert_eq!(onnx_model.weights["bias"].len(), 3);

        // Graph should have: 2 params + 1 input + matmul_bt + bias_add + relu = 6 nodes
        let nodes = onnx_model.graph.nodes();
        assert!(nodes.len() >= 5, "expected >= 5 nodes, got {}", nodes.len());

        // Should have exactly 1 output
        assert_eq!(onnx_model.graph.outputs().len(), 1);
    }

    #[test]
    fn gelu_follows_its_approximation() {
        for (approximate, exact) in [
            (None, Some(true)),
            (Some("none"), Some(true)),
            (Some("tanh"), Some(false)),
            (Some("fast"), None),
        ] {
            let mut node = build_node_proto("Gelu", &["x"], &["y"], &[], &[]);
            if let Some(value) = approximate {
                let mut attr = pb_field_bytes(1, b"approximate");
                attr.extend(pb_field_bytes(4, value.as_bytes()));
                attr.extend(pb_field_varint(20, 3)); // attr_type = STRING
                node.extend(pb_field_bytes(5, &attr));
            }
            let bytes = build_onnx_model(
                &[node],
                &[],
                &[build_value_info("x", &[2, 4])],
                &[build_value_info("y", &[2, 4])],
            );
            let Some(exact) = exact else {
                // An approximation this importer does not know is refused.
                assert!(matches!(
                    load_onnx_bytes(&bytes, None),
                    Err(OnnxError::UnsupportedOp(_))
                ));
                continue;
            };
            let graph = load_onnx_bytes(&bytes, None).unwrap().graph;
            let op = &graph.node(graph.outputs()[0]).op;
            assert_eq!(matches!(*op, Op::GeluErf), exact, "{approximate:?}: {op:?}");
            assert_eq!(matches!(*op, Op::Gelu), !exact, "{approximate:?}: {op:?}");
        }
    }

    #[test]
    fn test_parse_input_shapes() {
        // Build a model with a known input shape and verify we parse it
        let weight_init = build_tensor_proto("w", &[10, 5], &[0.0; 50]);
        let matmul_node = build_node_proto("MatMul", &["x", "w"], &["y"], &[], &[]);

        let x_vi = build_value_info("x", &[2, 10]);
        let w_vi = build_value_info("w", &[10, 5]);
        let y_vi = build_value_info("y", &[2, 5]);

        let model_bytes = build_onnx_model(&[matmul_node], &[weight_init], &[x_vi, w_vi], &[y_vi]);

        let graph = proto::parse_model(&model_bytes, None).unwrap();
        // `w` is an initializer, so it is not a graph input.
        assert_eq!(graph.inputs.len(), 1);
        assert_eq!(graph.inputs[0].name, "x");
        assert_eq!(graph.inputs[0].shape, Some(vec![2, 10]));
        assert_eq!(graph.initializers[0].shape, vec![10, 5]);
        assert_eq!(graph.outputs[0].shape, Some(vec![2, 5]));
    }

    #[test]
    fn test_load_matmul_add() {
        // Model: MatMul(x, w) + b -> output
        // x: [1, 4], w: [4, 3], b: [3]
        let w_data: Vec<f32> = (0..12).map(|i| (i as f32) * 0.1).collect();
        let b_data = vec![0.1, 0.2, 0.3];

        let w_init = build_tensor_proto("w", &[4, 3], &w_data);
        let b_init = build_tensor_proto("b", &[3], &b_data);

        let mm_node = build_node_proto("MatMul", &["x", "w"], &["mm_out"], &[], &[]);
        let add_node = build_node_proto("Add", &["mm_out", "b"], &["output"], &[], &[]);

        let x_vi = build_value_info("x", &[1, 4]);
        let w_vi = build_value_info("w", &[4, 3]);
        let b_vi = build_value_info("b", &[3]);
        let out_vi = build_value_info("output", &[1, 3]);

        let model_bytes = build_onnx_model(
            &[mm_node, add_node],
            &[w_init, b_init],
            &[x_vi, w_vi, b_vi],
            &[out_vi],
        );

        let result = load_onnx_bytes(&model_bytes, None);
        assert!(result.is_ok(), "load failed: {:?}", result.err());

        let model = result.unwrap();
        assert_eq!(model.graph.outputs().len(), 1);
        assert_eq!(model.weights.len(), 2);
    }

    /// A decomposed export, built from `(op, inputs, outputs)` triples over
    /// input `x: [4, 16]`, scalar initializers and per-column parameters.
    fn decomposed_model(
        nodes: &[(&str, &[&str], &str)],
        scalars: &[(&str, f32)],
        columns: &[&str],
    ) -> OnnxModel {
        let nodes: Vec<_> = nodes
            .iter()
            .map(|&(op, inputs, output)| build_node_proto(op, inputs, &[output], &[], &[]))
            .collect();
        let mut inits = Vec::new();
        let mut inputs = vec![build_value_info("x", &[4, 16])];
        for &(name, value) in scalars {
            inits.push(build_tensor_proto(name, &[1], &[value]));
            inputs.push(build_value_info(name, &[1]));
        }
        for (i, &name) in columns.iter().enumerate() {
            let data: Vec<f32> = (0..16).map(|j| 0.5 + 0.05 * (i * 16 + j) as f32).collect();
            inits.push(build_tensor_proto(name, &[16], &data));
            inputs.push(build_value_info(name, &[16]));
        }
        let bytes = build_onnx_model(&nodes, &inits, &inputs, &[build_value_info("y", &[4, 16])]);
        load_onnx_bytes(&bytes, None).expect("decomposed export loads")
    }

    /// The loaded graph computes `expected`'s output, and the optimizer
    /// folds it into the single fused op `fused` reports.
    fn assert_folds(
        model: &OnnxModel,
        expected: impl FnOnce(&mut Graph, NodeId, &[NodeId]) -> NodeId,
        columns: &[&str],
        fused: impl Fn(&Op) -> bool,
    ) {
        use crate::reference::{Feeds, evaluate_outputs};
        let mut feeds = Feeds::new();
        let x: Vec<f32> = (0..64).map(|i| (i as f32 * 0.37).sin() * 2.0).collect();
        feeds.set("x", &x);
        for (name, data) in &model.weights {
            feeds.set(name, data);
        }
        let got = evaluate_outputs(&model.graph, &feeds).unwrap();

        let mut reference = Graph::new();
        let rx = reference.input("x", &[4, 16]);
        let params: Vec<_> = columns
            .iter()
            .map(|name| reference.parameter(name, &[16]))
            .collect();
        let out = expected(&mut reference, rx, &params);
        reference.set_outputs(vec![out]);
        let want = evaluate_outputs(&reference, &feeds).unwrap();
        for (g, w) in got[0].data.iter().zip(&want[0].data) {
            assert!((g - w).abs() < 1e-6, "{g} vs {w}");
        }

        let optimized = crate::optimize::optimize(&model.graph.recompose());
        let output = optimized.node(optimized.outputs()[0]);
        assert!(fused(&output.op), "got {:?}", output.op);
        assert!(
            optimized
                .nodes()
                .iter()
                .all(|node| !matches!(node.op, Op::Sqrt | Op::Rsqrt | Op::Exp | Op::MaxInner)),
            "primitives left after folding: {:?}",
            optimized.nodes().iter().map(|n| &n.op).collect::<Vec<_>>()
        );
    }

    #[test]
    fn decomposed_rms_norm_export_folds() {
        let model = decomposed_model(
            &[
                ("Pow", &["x", "two"], "square"),
                ("ReduceMean", &["square", "axes"], "mean"),
                ("Add", &["mean", "eps"], "shifted"),
                ("Sqrt", &["shifted"], "root"),
                ("Div", &["x", "root"], "normalized"),
                ("Mul", &["normalized", "w"], "y"),
            ],
            &[("two", 2.0), ("axes", -1.0), ("eps", 1e-6)],
            &["w"],
        );
        assert_folds(
            &model,
            |g, x, p| g.rms_norm(x, p[0], 1e-6),
            &["w"],
            |op| matches!(*op, Op::RmsNorm { eps } if eps == 1e-6),
        );
    }

    #[test]
    fn decomposed_layer_norm_export_folds() {
        let model = decomposed_model(
            &[
                ("ReduceMean", &["x", "axes"], "mean"),
                ("Sub", &["x", "mean"], "centered"),
                ("Pow", &["centered", "two"], "square"),
                ("ReduceMean", &["square", "axes"], "variance"),
                ("Add", &["variance", "eps"], "shifted"),
                ("Sqrt", &["shifted"], "root"),
                ("Div", &["centered", "root"], "normalized"),
                ("Mul", &["normalized", "w"], "scaled"),
                ("Add", &["scaled", "b"], "y"),
            ],
            &[("two", 2.0), ("axes", -1.0), ("eps", 1e-5)],
            &["w", "b"],
        );
        assert_folds(
            &model,
            |g, x, p| g.layer_norm(x, p[0], p[1], 1e-5),
            &["w", "b"],
            |op| matches!(*op, Op::LayerNorm { eps } if eps == 1e-5),
        );
    }

    #[test]
    fn decomposed_softmax_export_folds() {
        let model = decomposed_model(
            &[
                ("ReduceMax", &["x", "axes"], "max"),
                ("Sub", &["x", "max"], "shifted"),
                ("Exp", &["shifted"], "e"),
                ("ReduceSum", &["e", "axes"], "total"),
                ("Div", &["e", "total"], "y"),
            ],
            &[("axes", -1.0)],
            &[],
        );
        assert_folds(
            &model,
            |g, x, _| g.softmax(x),
            &[],
            |op| matches!(*op, Op::Softmax),
        );
    }

    /// A one-node model over inputs `a` and `b` (with `b` an initializer
    /// when `b_data` is given), evaluated on the reference interpreter.
    fn one_node(
        op: &str,
        (a_shape, a_data): (&[i64], &[f32]),
        (b_shape, b_data): (&[i64], &[f32]),
        b_constant: bool,
    ) -> Result<(OnnxModel, Vec<f32>), OnnxError> {
        let node = build_node_proto(op, &["a", "b"], &["y"], &[], &[]);
        let mut inits = Vec::new();
        if b_constant {
            inits.push(build_tensor_proto("b", b_shape, b_data));
        }
        let bytes = build_onnx_model(
            &[node],
            &inits,
            &[
                build_value_info("a", a_shape),
                build_value_info("b", b_shape),
            ],
            &[build_value_info("y", &[])],
        );
        let model = load_onnx_bytes(&bytes, None)?;
        let mut feeds = crate::reference::Feeds::new();
        feeds.set("a", a_data);
        if !b_constant {
            feeds.set("b", b_data);
        }
        for (name, data) in &model.weights {
            feeds.set(name, data);
        }
        let out = crate::reference::evaluate_outputs(&model.graph, &feeds).unwrap();
        Ok((model, out[0].data.iter().map(|&v| v as f32).collect()))
    }

    fn has_concat(model: &OnnxModel) -> bool {
        model
            .graph
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::Concat { .. }))
    }

    /// `Expand` along leading, interior, trailing and several axes is one
    /// broadcast, with no copy tree of concatenations.
    #[test]
    fn expand_broadcasts_each_singleton_axis() {
        let cases: [(&[i64], &[i64]); 5] = [
            (&[1, 3], &[4, 3]),
            (&[3, 1], &[3, 5]),
            (&[2, 1, 3], &[2, 4, 3]),
            (&[3], &[2, 2, 3]),
            (&[2, 1, 1, 3, 1], &[2, 4, 5, 3, 1]),
        ];
        for (from, to) in cases {
            let n: i64 = from.iter().product();
            let data: Vec<f32> = (0..n).map(|i| i as f32 + 1.0).collect();
            let target: Vec<f32> = to.iter().map(|&d| d as f32).collect();
            let (model, got) =
                one_node("Expand", (from, &data), (&[to.len() as i64], &target), true).unwrap();
            // NumPy: index each output position back through unit axes.
            let to: Vec<usize> = to.iter().map(|&d| d as usize).collect();
            let mut from_p = vec![1usize; to.len() - from.len()];
            from_p.extend(from.iter().map(|&d| d as usize));
            let total: usize = to.iter().product();
            let want: Vec<f32> = (0..total)
                .map(|mut i| {
                    let mut src = 0;
                    let mut stride = 1;
                    for axis in (0..to.len()).rev() {
                        let coord = i % to[axis];
                        i /= to[axis];
                        if from_p[axis] != 1 {
                            src += coord * stride;
                        }
                        stride *= from_p[axis];
                    }
                    data[src]
                })
                .collect();
            assert_eq!(got, want, "{from:?} -> {to:?}");
            assert!(!has_concat(&model), "{from:?} -> {to:?} copies by concat");
        }
    }

    /// Operands broadcast along different axes: `[2, 1] ∘ [1, 2]` is a
    /// `[2, 2]` outer combination for every operator.
    #[test]
    fn binary_ops_broadcast_along_their_own_axes() {
        let (a, b) = ([1.0f32, 2.0], [4.0f32, 8.0]);
        type Apply = fn(f32, f32) -> f32;
        let cases: [(&str, Apply); 4] = [
            ("Add", |x, y| x + y),
            ("Sub", |x, y| x - y),
            ("Mul", |x, y| x * y),
            ("Div", |x, y| x / y),
        ];
        for (op, f) in cases {
            for b_constant in [false, true] {
                let (_, got) = one_node(op, (&[2, 1], &a), (&[1, 2], &b), b_constant).unwrap();
                let want = [f(a[0], b[0]), f(a[0], b[1]), f(a[1], b[0]), f(a[1], b[1])];
                assert_eq!(got, want, "{op}");
                // And the other way around.
                let (_, got) = one_node(op, (&[1, 2], &b), (&[2, 1], &a), b_constant).unwrap();
                let want = [f(b[0], a[0]), f(b[1], a[0]), f(b[0], a[1]), f(b[1], a[1])];
                assert_eq!(got, want, "{op} swapped");
            }
        }
        // A narrow left side of a non-commuting op.
        let (_, got) = one_node("Sub", (&[2], &a), (&[3, 2], &[1.0; 6]), false).unwrap();
        assert_eq!(got, [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn incompatible_broadcasts_are_refused() {
        let six = [1.0; 6];
        for op in ["Add", "Sub", "Mul", "Div"] {
            assert!(
                matches!(
                    one_node(op, (&[2, 3], &six), (&[3, 2], &six), false),
                    Err(OnnxError::ShapeError(_))
                ),
                "{op}"
            );
        }
    }

    #[test]
    fn reductions_over_other_axes_are_refused() {
        let nodes = [build_node_proto(
            "ReduceMean",
            &["x", "axes"],
            &["y"],
            &[],
            &[],
        )];
        let bytes = build_onnx_model(
            &nodes,
            &[build_tensor_proto("axes", &[1], &[0.0])],
            &[
                build_value_info("x", &[4, 16]),
                build_value_info("axes", &[1]),
            ],
            &[build_value_info("y", &[1, 16])],
        );
        assert!(matches!(
            load_onnx_bytes(&bytes, None),
            Err(OnnxError::UnsupportedOp(_))
        ));
    }

    /// Each output of `Split` is its run of columns, the middle one too.
    #[test]
    fn split_outputs_are_column_ranges() {
        let nodes = [build_node_proto(
            "Split",
            &["x", "sizes"],
            &["a", "b", "c"],
            &[("axis", -1)],
            &[],
        )];
        let bytes = build_onnx_model(
            &nodes,
            &[build_tensor_proto("sizes", &[3], &[1.0, 3.0, 2.0])],
            &[
                build_value_info("x", &[2, 6]),
                build_value_info("sizes", &[3]),
            ],
            &["a", "b", "c"].map(|name| build_value_info(name, &[])),
        );
        let model = load_onnx_bytes(&bytes, None).unwrap();
        let x: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let mut feeds = crate::reference::Feeds::new();
        feeds.set("x", &x);
        feeds.set("sizes", &model.weights["sizes"]);
        let got = crate::reference::evaluate_outputs(&model.graph, &feeds).unwrap();
        let got: Vec<Vec<f64>> = got.into_iter().map(|t| t.data).collect();
        let want = [
            vec![0.0, 6.0],
            vec![1.0, 2.0, 3.0, 7.0, 8.0, 9.0],
            vec![4.0, 5.0, 10.0, 11.0],
        ];
        assert_eq!(got, want);
    }
}
