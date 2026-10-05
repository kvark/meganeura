//! ONNX model import: load standard ONNX files into Meganeura's `Graph` IR.
//!
//! Translates an ONNX computation graph into our `Graph` at runtime, then the
//! normal pipeline (optimize -> compile -> Session) handles execution. No Rust
//! codegen needed.

use std::collections::HashMap;
use std::path::Path;

use oxionnx_core::{Graph as OnnxGraph, Node as OnnxNode, OpKind};
use oxionnx_proto::model;

use crate::graph::{Graph, NodeId, Op};

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
    // oxionnx-proto can panic on overflowing lengths; contain dependency panics.
    let (onnx_graph, onnx_weights) = std::panic::catch_unwind(|| {
        if let Some(p) = path.and_then(|p| p.parent()) {
            model::load_with_path(bytes, p)
        } else {
            model::load(bytes)
        }
    })
    .map_err(|_| OnnxError::ParseError("malformed model: protobuf reader panicked".into()))?
    .map_err(OnnxError::ParseError)?;

    // Convert oxionnx Tensor weights to Vec<f32>
    let weights: HashMap<String, Vec<f32>> = onnx_weights
        .into_iter()
        .map(|(name, tensor)| (name, tensor.data))
        .collect();

    // Extract shapes from the raw protobuf (initializer dims + input ValueInfoProto shapes)
    let all_shapes = extract_shapes_from_proto(bytes)?;

    let graph = translate_graph(&onnx_graph, &weights, &all_shapes)?;

    Ok(OnnxModel { graph, weights })
}

/// Extract tensor shapes from the ONNX protobuf: both initializer dims and
/// input/output ValueInfoProto type shapes.
///
/// oxionnx-proto only extracts names from ValueInfoProto, not shapes.
/// We parse the raw protobuf to recover them.
fn extract_shapes_from_proto(bytes: &[u8]) -> Result<HashMap<String, Vec<usize>>, OnnxError> {
    let proto_model = oxionnx_proto::parser::parse_model(bytes).map_err(OnnxError::ParseError)?;
    let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();

    // 1. Initializer shapes (from TensorProto.dims)
    for init in &proto_model.graph.initializers {
        let shape: Vec<usize> = init.dims.iter().map(|&d| d as usize).collect();
        shapes.insert(init.name.clone(), shape);
    }

    // 2. Input shapes from ValueInfoProto (re-parse the graph to get type info)
    //    We need to parse the raw graph protobuf to extract shapes that oxionnx-proto discards.
    let graph_bytes = extract_graph_bytes(bytes)?;
    let input_shapes = parse_value_info_shapes(&graph_bytes, 11)?; // field 11 = input
    let output_shapes = parse_value_info_shapes(&graph_bytes, 12)?; // field 12 = output
    for (name, shape) in input_shapes.into_iter().chain(output_shapes) {
        shapes.entry(name).or_insert(shape);
    }

    Ok(shapes)
}

/// Extract the raw bytes of the GraphProto (field 7) from a ModelProto.
fn extract_graph_bytes(model_bytes: &[u8]) -> Result<Vec<u8>, OnnxError> {
    let mut pos = 0;
    while pos < model_bytes.len() {
        let (tag, next_pos) = read_proto_varint(model_bytes, pos).map_err(OnnxError::ParseError)?;
        let field_no = (tag >> 3) as u32;
        let wire_type = (tag & 0x7) as u8;
        pos = next_pos;

        match wire_type {
            0 => {
                // varint — skip
                let (_, p) = read_proto_varint(model_bytes, pos).map_err(OnnxError::ParseError)?;
                pos = p;
            }
            1 => pos += 8, // fixed64
            5 => pos += 4, // fixed32
            2 => {
                let (len, p) =
                    read_proto_varint(model_bytes, pos).map_err(OnnxError::ParseError)?;
                let (field, end) = delimited(model_bytes, len, p).ok_or_else(truncated)?;
                if field_no == 7 {
                    return Ok(field.to_vec());
                }
                pos = end;
            }
            _ => {
                return Err(OnnxError::ParseError(format!(
                    "unknown wire type {wire_type}"
                )));
            }
        }
    }
    Ok(Vec::new())
}

/// Parse ValueInfoProto entries at a given field number within a GraphProto,
/// extracting (name, shape) pairs.
fn parse_value_info_shapes(
    graph_bytes: &[u8],
    target_field: u32,
) -> Result<Vec<(String, Vec<usize>)>, OnnxError> {
    let mut results = Vec::new();
    let mut pos = 0;

    while pos < graph_bytes.len() {
        let (tag, next_pos) = read_proto_varint(graph_bytes, pos).map_err(OnnxError::ParseError)?;
        let field_no = (tag >> 3) as u32;
        let wire_type = (tag & 0x7) as u8;
        pos = next_pos;

        match wire_type {
            0 => {
                let (_, p) = read_proto_varint(graph_bytes, pos).map_err(OnnxError::ParseError)?;
                pos = p;
            }
            1 => pos = pos.checked_add(8).ok_or_else(truncated)?,
            5 => pos = pos.checked_add(4).ok_or_else(truncated)?,
            2 => {
                let (len, p) =
                    read_proto_varint(graph_bytes, pos).map_err(OnnxError::ParseError)?;
                let (field, end) = delimited(graph_bytes, len, p).ok_or_else(truncated)?;
                if field_no == target_field {
                    // This is a ValueInfoProto — parse name and shape from it
                    if let Some((name, shape)) = parse_single_value_info(field) {
                        results.push((name, shape));
                    }
                }
                pos = end;
            }
            _ => {
                return Err(OnnxError::ParseError(format!(
                    "unknown wire type {wire_type}"
                )));
            }
        }
    }

    Ok(results)
}

/// Parse a single ValueInfoProto message to extract (name, shape).
/// ValueInfoProto: field 1 = name, field 2 = TypeProto
/// TypeProto: field 1 = tensor_type (TypeProto.Tensor)
/// TypeProto.Tensor: field 2 = shape (TensorShapeProto)
/// TensorShapeProto: field 1 = dim (Dimension, repeated)
/// Dimension: field 1 = dim_value (int64)
fn parse_single_value_info(buf: &[u8]) -> Option<(String, Vec<usize>)> {
    let mut name = String::new();
    let mut type_bytes = None;
    let mut pos = 0;

    while pos < buf.len() {
        let (tag, next_pos) = read_proto_varint(buf, pos).ok()?;
        let field_no = (tag >> 3) as u32;
        let wire_type = (tag & 0x7) as u8;
        pos = next_pos;

        match wire_type {
            0 => {
                let (_, p) = read_proto_varint(buf, pos).ok()?;
                pos = p;
            }
            1 => match pos.checked_add(8) {
                Some(next) => pos = next,
                None => break,
            },
            5 => match pos.checked_add(4) {
                Some(next) => pos = next,
                None => break,
            },
            2 => {
                let (len, p) = read_proto_varint(buf, pos).ok()?;
                let (field, end) = delimited(buf, len, p)?;
                match field_no {
                    1 => name = String::from_utf8_lossy(field).into_owned(),
                    2 => type_bytes = Some(field),
                    _ => {}
                }
                pos = end;
            }
            _ => return None,
        }
    }

    let shape = type_bytes.and_then(parse_type_proto_shape)?;
    Some((name, shape))
}

/// Extract shape dims from a TypeProto message.
fn parse_type_proto_shape(buf: &[u8]) -> Option<Vec<usize>> {
    // TypeProto: field 1 = tensor_type (TypeProto.Tensor)
    let mut pos = 0;
    while pos < buf.len() {
        let (tag, next_pos) = read_proto_varint(buf, pos).ok()?;
        let field_no = (tag >> 3) as u32;
        let wire_type = (tag & 0x7) as u8;
        pos = next_pos;

        match wire_type {
            0 => {
                let (_, p) = read_proto_varint(buf, pos).ok()?;
                pos = p;
            }
            1 => match pos.checked_add(8) {
                Some(next) => pos = next,
                None => break,
            },
            5 => match pos.checked_add(4) {
                Some(next) => pos = next,
                None => break,
            },
            2 => {
                let (len, p) = read_proto_varint(buf, pos).ok()?;
                let (field, end) = delimited(buf, len, p)?;
                if field_no == 1 {
                    // tensor_type = TypeProto.Tensor
                    return parse_tensor_type_shape(field);
                }
                pos = end;
            }
            _ => return None,
        }
    }
    None
}

/// Extract shape dims from TypeProto.Tensor: field 2 = shape (TensorShapeProto).
fn parse_tensor_type_shape(buf: &[u8]) -> Option<Vec<usize>> {
    let mut pos = 0;
    while pos < buf.len() {
        let (tag, next_pos) = read_proto_varint(buf, pos).ok()?;
        let field_no = (tag >> 3) as u32;
        let wire_type = (tag & 0x7) as u8;
        pos = next_pos;

        match wire_type {
            0 => {
                let (_, p) = read_proto_varint(buf, pos).ok()?;
                pos = p;
            }
            1 => match pos.checked_add(8) {
                Some(next) => pos = next,
                None => break,
            },
            5 => match pos.checked_add(4) {
                Some(next) => pos = next,
                None => break,
            },
            2 => {
                let (len, p) = read_proto_varint(buf, pos).ok()?;
                let (field, end) = delimited(buf, len, p)?;
                if field_no == 2 {
                    // shape = TensorShapeProto
                    return Some(parse_tensor_shape_dims(field));
                }
                pos = end;
            }
            _ => return None,
        }
    }
    None
}

/// Parse TensorShapeProto: field 1 = dim (repeated Dimension).
/// Dimension: field 1 = dim_value (int64), field 2 = dim_param (string).
fn parse_tensor_shape_dims(buf: &[u8]) -> Vec<usize> {
    let mut dims = Vec::new();
    let mut pos = 0;

    while pos < buf.len() {
        let Ok((tag, next_pos)) = read_proto_varint(buf, pos) else {
            break;
        };
        let field_no = (tag >> 3) as u32;
        let wire_type = (tag & 0x7) as u8;
        pos = next_pos;

        match wire_type {
            0 => {
                let Ok((_, p)) = read_proto_varint(buf, pos) else {
                    break;
                };
                pos = p;
            }
            1 => match pos.checked_add(8) {
                Some(next) => pos = next,
                None => break,
            },
            5 => match pos.checked_add(4) {
                Some(next) => pos = next,
                None => break,
            },
            2 => {
                let Ok((len, p)) = read_proto_varint(buf, pos) else {
                    break;
                };
                let Some((field, end)) = delimited(buf, len, p) else {
                    break;
                };
                if field_no == 1 {
                    // Dimension message
                    dims.push(parse_dimension(field));
                }
                pos = end;
            }
            _ => break,
        }
    }

    dims
}

/// Parse a Dimension message: field 1 = dim_value (int64).
/// Dynamic dims (dim_param) are treated as 0 (unknown).
fn parse_dimension(buf: &[u8]) -> usize {
    let mut pos = 0;
    while pos < buf.len() {
        let Ok((tag, next_pos)) = read_proto_varint(buf, pos) else {
            break;
        };
        let field_no = (tag >> 3) as u32;
        let wire_type = (tag & 0x7) as u8;
        pos = next_pos;

        match wire_type {
            0 => {
                let Ok((val, p)) = read_proto_varint(buf, pos) else {
                    break;
                };
                pos = p;
                if field_no == 1 {
                    return val as usize;
                }
            }
            1 => match pos.checked_add(8) {
                Some(next) => pos = next,
                None => break,
            },
            5 => match pos.checked_add(4) {
                Some(next) => pos = next,
                None => break,
            },
            2 => {
                let Ok((len, p)) = read_proto_varint(buf, pos) else {
                    break;
                };
                let Some((_, end)) = delimited(buf, len, p) else {
                    break;
                };
                pos = end;
            }
            _ => break,
        }
    }
    0 // dynamic/unknown dimension
}

/// Read a protobuf varint from a byte slice at the given position.
fn read_proto_varint(buf: &[u8], mut pos: usize) -> Result<(u64, usize), String> {
    let mut result = 0u64;
    let mut shift = 0u32;
    loop {
        if pos >= buf.len() {
            return Err("varint: unexpected EOF".into());
        }
        let byte = buf[pos];
        pos += 1;
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
        if shift >= 64 {
            return Err("varint: overflow".into());
        }
    }
    Ok((result, pos))
}

/// A field claimed more bytes than the buffer holds.
fn truncated() -> OnnxError {
    OnnxError::ParseError("a field length runs past the end of the message".into())
}

/// Resolve a length-delimited payload and its end offset without wrapping.
fn delimited(buf: &[u8], len: u64, start: usize) -> Option<(&[u8], usize)> {
    let len = usize::try_from(len).ok()?;
    let end = start.checked_add(len)?;
    Some((buf.get(start..end)?, end))
}

/// Translate an oxionnx Graph into a Meganeura Graph.
fn translate_graph(
    onnx: &OnnxGraph,
    weights: &HashMap<String, Vec<f32>>,
    proto_shapes: &HashMap<String, Vec<usize>>,
) -> Result<Graph, OnnxError> {
    let mut graph = Graph::new();
    // Map ONNX tensor names -> Meganeura NodeId
    let mut name_to_id: HashMap<String, NodeId> = HashMap::new();
    // Track output shapes by name for shape inference
    let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();

    // 1. Create parameter nodes for initializers (weights)
    for (name, data) in weights {
        let shape = proto_shapes
            .get(name.as_str())
            .cloned()
            .unwrap_or_else(|| vec![data.len()]);
        let id = graph.parameter(name, &shape);
        name_to_id.insert(name.clone(), id);
        shapes.insert(name.clone(), shape);
    }

    // 2. Create input nodes for graph inputs that aren't initializers
    for input_name in &onnx.input_names {
        if !weights.contains_key(input_name) {
            // Get shape from ValueInfoProto (parsed from raw protobuf)
            let shape = proto_shapes
                .get(input_name.as_str())
                .cloned()
                .unwrap_or_else(|| {
                    log::warn!("ONNX input '{}': shape unknown, using [1]", input_name);
                    vec![1]
                });
            // Flatten to 2D for our IR: [batch, ..., features] -> [batch*..., features]
            let flat_shape = flatten_to_2d(&shape);
            let id = graph.input(input_name, &flat_shape);
            name_to_id.insert(input_name.clone(), id);
            shapes.insert(input_name.clone(), shape);
        }
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
        .output_names
        .iter()
        .filter_map(|name| name_to_id.get(name).copied())
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
    node: &OnnxNode,
    name_to_id: &mut HashMap<String, NodeId>,
    shapes: &mut HashMap<String, Vec<usize>>,
    weights: &HashMap<String, Vec<f32>>,
) -> Result<(), OnnxError> {
    let attrs = &node.attrs;
    let op = &node.op;

    match *op {
        // --- Element-wise unary ---
        OpKind::Relu => unary_op(graph, node, name_to_id, shapes, Op::Relu)?,
        OpKind::Sigmoid => unary_op(graph, node, name_to_id, shapes, Op::Sigmoid)?,
        OpKind::Neg => unary_op(graph, node, name_to_id, shapes, Op::Neg)?,
        OpKind::Abs => unary_op(graph, node, name_to_id, shapes, Op::Abs)?,
        OpKind::Log => unary_op(graph, node, name_to_id, shapes, Op::Log)?,
        OpKind::Reciprocal => unary_op(graph, node, name_to_id, shapes, Op::Recip)?,
        OpKind::Gelu => unary_op(graph, node, name_to_id, shapes, Op::Gelu)?,
        OpKind::SiLU => unary_op(graph, node, name_to_id, shapes, Op::Silu)?,

        // Elementary math, as exporters write decomposed norms and
        // activations. These map onto primitives, and the optimizer folds
        // recognized decompositions back into fused kernels.
        OpKind::Sqrt => unary_op(graph, node, name_to_id, shapes, Op::Sqrt)?,
        OpKind::Exp => unary_op(graph, node, name_to_id, shapes, Op::Exp)?,
        OpKind::Tanh => unary_op(graph, node, name_to_id, shapes, Op::Tanh)?,
        OpKind::Pow => {
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
        OpKind::ReduceMean | OpKind::ReduceSum | OpKind::ReduceMax => {
            reduce_op(graph, node, name_to_id, shapes, weights)?;
        }
        OpKind::Erf => unary_op(graph, node, name_to_id, shapes, Op::Erf)?,

        // Cast: passthrough (we only support f32)
        OpKind::Cast => {
            let x = resolve_input(&node.inputs[0], name_to_id, &node.name)?;
            let x_shape = get_shape(&node.inputs[0], shapes);
            if !node.outputs.is_empty() {
                name_to_id.insert(node.outputs[0].clone(), x);
                shapes.insert(node.outputs[0].clone(), x_shape);
            }
        }

        // Shape: produces a 1D constant of the input's static shape
        OpKind::Shape => {
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
        OpKind::Add => binary_op(graph, node, name_to_id, shapes, weights, BinaryKind::Add)?,
        OpKind::Sub => binary_op(graph, node, name_to_id, shapes, weights, BinaryKind::Sub)?,
        OpKind::Mul => binary_op(graph, node, name_to_id, shapes, weights, BinaryKind::Mul)?,
        OpKind::Div => binary_op(graph, node, name_to_id, shapes, weights, BinaryKind::Div)?,

        // --- MatMul ---
        OpKind::MatMul => {
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
        OpKind::Gemm => {
            let a = resolve_input(&node.inputs[0], name_to_id, &node.name)?;
            let b = resolve_input(&node.inputs[1], name_to_id, &node.name)?;
            let trans_a = attrs.i("transA", 0) != 0;
            let trans_b = attrs.i("transB", 0) != 0;
            let a_shape = get_shape(&node.inputs[0], shapes);
            let b_shape = get_shape(&node.inputs[1], shapes);

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
                graph.bias_add(mm, c)
            } else {
                mm
            };

            register_output(node, 0, out, &out_shape, name_to_id, shapes);
        }

        // --- Softmax ---
        OpKind::Softmax | OpKind::LogSoftmax => {
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
                    node.op.as_str()
                )));
            }
            let x = as_matrix(graph, x, &x_shape)?;
            let out = if matches!(node.op, OpKind::Softmax) {
                graph.softmax(x)
            } else {
                graph.log_softmax(x)
            };
            register_output(node, 0, out, &x_shape, name_to_id, shapes);
        }

        // --- Normalization ---
        OpKind::LayerNorm => {
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

        OpKind::RMSNorm => {
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
        OpKind::Gather => {
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
            // Shape arithmetic (`Shape` → `Gather`) folds at import.
            if let (Some(values), Some(picks)) = (
                constant_values(graph, data, weights),
                constant_values(graph, indices, weights),
            ) {
                let (values, picks) = (values.to_vec(), picks.to_vec());
                let dim = data_shape[axis];
                let inner: usize = data_shape[axis + 1..].iter().product();
                let outer: usize = data_shape[..axis].iter().product();
                if values.len() != outer * dim * inner {
                    return Err(OnnxError::ShapeError(format!(
                        "node '{}': Gather data does not match {data_shape:?}",
                        node.name
                    )));
                }
                let mut out = Vec::with_capacity(outer * picks.len() * inner);
                for o in 0..outer {
                    for &pick in &picks {
                        let pick = pick as i64;
                        let pick = if pick < 0 { pick + dim as i64 } else { pick };
                        if !(0..dim as i64).contains(&pick) {
                            return Err(OnnxError::ShapeError(format!(
                                "node '{}': Gather index {pick} out of {dim}",
                                node.name
                            )));
                        }
                        let start = (o * dim + pick as usize) * inner;
                        out.extend_from_slice(&values[start..start + inner]);
                    }
                }
                let mut out_shape = data_shape[..axis].to_vec();
                out_shape.extend_from_slice(&indices_shape);
                out_shape.extend_from_slice(&data_shape[axis + 1..]);
                let id = graph.constant(out, &out_shape);
                register_output(node, 0, id, &out_shape, name_to_id, shapes);
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
        OpKind::Transpose => {
            let x = resolve_input(&node.inputs[0], name_to_id, &node.name)?;
            let x_shape = get_shape(&node.inputs[0], shapes);
            let rank = x_shape.len();
            let perm: Vec<usize> = match *attrs.ints("perm") {
                [] => (0..rank).rev().collect(),
                ref perm => perm.iter().map(|&axis| axis as usize).collect(),
            };
            let mut sorted = perm.clone();
            sorted.sort_unstable();
            if sorted != (0..rank).collect::<Vec<_>>() || rank > 4 {
                return Err(OnnxError::UnsupportedOp(format!(
                    "Transpose of {x_shape:?} by {perm:?} (supported: rank at most 4)"
                )));
            }
            let out_shape: Vec<usize> = perm.iter().map(|&axis| x_shape[axis]).collect();
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

        OpKind::Reshape => {
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

        OpKind::Flatten => {
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

        OpKind::Squeeze | OpKind::Unsqueeze => {
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
            let new_shape = if matches!(node.op, OpKind::Squeeze) {
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

        OpKind::Expand => {
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
            let out_shape = broadcast_shape(&x_shape, &target);
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
            let mut out = x;
            // Repeat along each broadcast axis by doubling concatenations.
            for axis in 0..shape.len() {
                let copies = out_shape[axis];
                if shape[axis] != 1 || copies == 1 {
                    continue;
                }
                let outer = shape[..axis].iter().product::<usize>() as u32;
                let inner = shape[axis + 1..].iter().product::<usize>() as u32;
                let (mut acc, mut acc_n) = (None, 0u32);
                let (mut piece, mut piece_n) = (out, 1u32);
                let mut remaining = copies;
                while remaining > 0 {
                    if remaining & 1 == 1 {
                        acc = Some(match acc {
                            None => piece,
                            Some(a) => graph.concat(a, piece, outer, acc_n, piece_n, inner),
                        });
                        acc_n += piece_n;
                    }
                    remaining >>= 1;
                    if remaining > 0 {
                        piece = graph.concat(piece, piece, outer, piece_n, piece_n, inner);
                        piece_n *= 2;
                    }
                }
                out = acc.unwrap_or(out);
                shape[axis] = copies;
            }
            register_output(node, 0, out, &out_shape, name_to_id, shapes);
        }

        // --- Identity / Dropout (inference mode) ---
        OpKind::Identity | OpKind::Dropout => {
            let x = resolve_input(&node.inputs[0], name_to_id, &node.name)?;
            let x_shape = get_shape(&node.inputs[0], shapes);
            if !node.outputs.is_empty() {
                name_to_id.insert(node.outputs[0].clone(), x);
                shapes.insert(node.outputs[0].clone(), x_shape);
            }
        }

        // --- Constant ---
        OpKind::Constant => {
            if let Some(tensor) = attrs.tensors.get("value") {
                let data = tensor.data.clone();
                let shape = tensor.shape.clone();
                let id = graph.constant(data, &shape);
                if !node.outputs.is_empty() {
                    name_to_id.insert(node.outputs[0].clone(), id);
                    shapes.insert(node.outputs[0].clone(), shape);
                }
            }
        }

        // --- Conv (1D or 2D) ---
        // Conv1d [N,C,L] is treated as Conv2d with H=1: [N,C,1,L]
        OpKind::Conv => {
            let input = resolve_input(&node.inputs[0], name_to_id, &node.name)?;
            let kernel = resolve_input(&node.inputs[1], name_to_id, &node.name)?;
            let input_shape = get_shape(&node.inputs[0], shapes);
            let kernel_shape = get_shape(&node.inputs[1], shapes);

            let (batch, in_channels, in_h, in_w, out_channels, kernel_h, kernel_w) =
                if input_shape.len() == 4 && kernel_shape.len() == 4 {
                    // Standard Conv2d
                    (
                        input_shape[0] as u32,
                        input_shape[1] as u32,
                        input_shape[2] as u32,
                        input_shape[3] as u32,
                        kernel_shape[0] as u32,
                        kernel_shape[2] as u32,
                        kernel_shape[3] as u32,
                    )
                } else if input_shape.len() == 3 && kernel_shape.len() == 3 {
                    // Conv1d: [N,C,L] → treat as [N,C,1,L]
                    (
                        input_shape[0] as u32,
                        input_shape[1] as u32,
                        1u32,
                        input_shape[2] as u32,
                        kernel_shape[0] as u32,
                        1u32,
                        kernel_shape[2] as u32,
                    )
                } else {
                    return Err(OnnxError::UnsupportedOp(format!(
                        "Conv: expected 3D or 4D input/kernel, got {}D and {}D",
                        input_shape.len(),
                        kernel_shape.len()
                    )));
                };

            let strides = attrs.ints("strides");
            let pads = attrs.ints("pads");
            let stride = strides.first().copied().unwrap_or(1) as u32;
            // For Conv2d pads=[pH_begin, pW_begin, pH_end, pW_end],
            // for Conv1d pads=[p_begin, p_end] → padding_h=0, padding_w=p.
            let (padding_h, padding_w) = if input_shape.len() == 3 {
                // Conv1d: no height padding, width padding only
                (0u32, pads.first().copied().unwrap_or(0) as u32)
            } else {
                let ph = pads.first().copied().unwrap_or(0) as u32;
                let pw = if pads.len() >= 2 { pads[1] as u32 } else { ph };
                (ph, pw)
            };

            let out = graph.conv2d_hw(
                input,
                kernel,
                batch,
                in_channels,
                in_h,
                in_w,
                out_channels,
                kernel_h,
                kernel_w,
                stride,
                padding_h,
                padding_w,
            );

            let out_h = (in_h + 2 * padding_h - kernel_h) / stride + 1;
            let out_w = (in_w + 2 * padding_w - kernel_w) / stride + 1;
            let out_shape = if input_shape.len() == 3 {
                // Conv1d output: [N, C_out, L_out]
                vec![batch as usize, out_channels as usize, out_w as usize]
            } else {
                vec![
                    batch as usize,
                    out_channels as usize,
                    out_h as usize,
                    out_w as usize,
                ]
            };

            // Add bias if present
            let out = if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
                let bias = resolve_input(&node.inputs[2], name_to_id, &node.name)?;
                graph.bias_add(out, bias)
            } else {
                out
            };

            register_output(node, 0, out, &out_shape, name_to_id, shapes);
        }

        // --- Concat ---
        OpKind::Concat => {
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

        OpKind::Slice => {
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
                || steps.iter().any(|&step| step != 1)
            {
                return Err(OnnxError::UnsupportedOp(format!(
                    "Slice with axes {axes:?} and steps {steps:?} (supported: unit steps)"
                )));
            }
            let mut out = x;
            let mut out_shape = x_shape;
            for ((&axis, &start), &end) in axes.iter().zip(&starts).zip(&ends) {
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
                // Negative bounds count from the end; both clamp to the axis.
                let clamp = |v: i64| (if v < 0 { v + dim } else { v }).clamp(0, dim) as u32;
                let (start, end) = (clamp(start), clamp(end).max(clamp(start)));
                let outer = out_shape[..axis].iter().product::<usize>() as u32;
                let inner = out_shape[axis + 1..].iter().product::<usize>() as u32;
                let dim = dim as u32;
                // Drop the leading `start`, then keep the first `end - start`.
                if start > 0 {
                    out = graph.split_b(out, outer, start, dim - start, inner);
                }
                if end < dim {
                    out = graph.split_a(out, outer, end - start, dim - end, inner);
                }
                out_shape[axis] = (end - start) as usize;
            }
            register_output(node, 0, out, &out_shape, name_to_id, shapes);
        }

        // --- GroupNorm ---
        OpKind::GroupNorm => {
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
        // Decompose: output = scale * (x - mean) / sqrt(var + eps) + bias
        // In inference mode, mean and var are running statistics (constants).
        // We precompute: w = scale / sqrt(var + eps), b = bias - mean * w
        // Then: output = x * w + b  (per-channel, broadcast over spatial dims)
        OpKind::BatchNorm => {
            let x = resolve_input(&node.inputs[0], name_to_id, &node.name)?;
            let x_shape = get_shape(&node.inputs[0], shapes);
            let eps = attrs.f("epsilon", 1e-5);

            if x_shape.len() != 4 {
                return Err(OnnxError::UnsupportedOp(
                    "BatchNormalization: only 4D (NCHW) supported".into(),
                ));
            }

            // Get scale, bias, mean, var from weights (they're initializers)
            let scale_data = weights
                .get(&node.inputs[1])
                .ok_or_else(|| OnnxError::ShapeError("BatchNorm: missing scale".into()))?;
            let bias_data = weights
                .get(&node.inputs[2])
                .ok_or_else(|| OnnxError::ShapeError("BatchNorm: missing bias".into()))?;
            let mean_data = weights
                .get(&node.inputs[3])
                .ok_or_else(|| OnnxError::ShapeError("BatchNorm: missing mean".into()))?;
            let var_data = weights
                .get(&node.inputs[4])
                .ok_or_else(|| OnnxError::ShapeError("BatchNorm: missing var".into()))?;

            let c = scale_data.len();
            // Precompute fused weight and bias per channel
            let mut fused_w = vec![0.0f32; c];
            let mut fused_b = vec![0.0f32; c];
            for i in 0..c {
                let inv_std = 1.0 / (var_data[i] + eps).sqrt();
                fused_w[i] = scale_data[i] * inv_std;
                fused_b[i] = bias_data[i] - mean_data[i] * fused_w[i];
            }

            // x * fused_w + fused_b (broadcast over spatial dims)
            // For NCHW: fused_w/fused_b are [C], need to broadcast over [N,C,H,W]
            // Expand to full spatial: tile [C] → [N*C*H*W]
            let n = x_shape[0];
            let h = x_shape[2];
            let w = x_shape[3];
            let spatial = h * w;
            let full_size = n * c * spatial;
            let mut w_expanded = vec![0.0f32; full_size];
            let mut b_expanded = vec![0.0f32; full_size];
            for batch in 0..n {
                for ch in 0..c {
                    for s in 0..spatial {
                        let idx = (batch * c + ch) * spatial + s;
                        w_expanded[idx] = fused_w[ch];
                        b_expanded[idx] = fused_b[ch];
                    }
                }
            }

            let w_node = graph.constant(w_expanded, &[full_size]);
            let b_node = graph.constant(b_expanded, &[full_size]);
            let scaled = graph.mul(x, w_node);
            let out = graph.add(scaled, b_node);
            register_output(node, 0, out, &x_shape, name_to_id, shapes);
        }

        // --- MaxPool ---
        OpKind::MaxPool => {
            let input = resolve_input(&node.inputs[0], name_to_id, &node.name)?;
            let input_shape = get_shape(&node.inputs[0], shapes);
            if input_shape.len() != 4 {
                return Err(OnnxError::UnsupportedOp(
                    "MaxPool: only 4D (NCHW) supported".into(),
                ));
            }

            let channels = input_shape[1] as u32;
            let in_h = input_shape[2] as u32;
            let in_w = input_shape[3] as u32;
            let batch = input_shape[0] as u32;

            let kernel_shape = attrs.ints("kernel_shape");
            let strides = attrs.ints("strides");
            let pads = attrs.ints("pads");
            let kh = kernel_shape.first().copied().unwrap_or(2) as u32;
            let kw = kernel_shape.get(1).copied().unwrap_or(kh as i64) as u32;
            let stride = strides.first().copied().unwrap_or(kh as i64) as u32;
            let padding = pads.first().copied().unwrap_or(0) as u32;

            let out =
                graph.max_pool_2d(input, batch, channels, in_h, in_w, kh, kw, stride, padding);

            let out_h = (in_h + 2 * padding - kh) / stride + 1;
            let out_w = (in_w + 2 * padding - kw) / stride + 1;
            let out_shape = vec![
                batch as usize,
                channels as usize,
                out_h as usize,
                out_w as usize,
            ];
            register_output(node, 0, out, &out_shape, name_to_id, shapes);
        }

        // --- GlobalAveragePool ---
        OpKind::GlobalAveragePool => {
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

            let out = graph.global_avg_pool(input, batch, channels, spatial);
            let out_shape = vec![input_shape[0], input_shape[1], 1, 1];
            register_output(node, 0, out, &out_shape, name_to_id, shapes);
        }

        // --- Unsupported ops produce a clear error ---
        ref other => {
            return Err(OnnxError::UnsupportedOp(other.as_str().to_string()));
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
    node: &OnnxNode,
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
/// every row of `a`, and equal leading batch axes multiply per batch.
fn matmul_nd(
    graph: &mut Graph,
    a: NodeId,
    a_shape: &[usize],
    b: NodeId,
    b_shape: &[usize],
) -> Result<Option<(NodeId, Vec<usize>)>, OnnxError> {
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

/// `a op b` for the broadcasts exported graphs use: a scalar constant
/// (folded into the op as an attribute, even against a one-element
/// tensor), equal element counts, a smaller operand matching the trailing
/// axes (a bias, a mask, a rotary table), or one value per row along the
/// last axis. Returns the result and its shape.
fn broadcast_binary(
    graph: &mut Graph,
    (a, a_shape): (NodeId, &[usize]),
    (b, b_shape): (NodeId, &[usize]),
    kind: BinaryKind,
    weights: &HashMap<String, Vec<f32>>,
) -> Result<Option<(NodeId, Vec<usize>)>, OnnxError> {
    let (a_len, b_len) = (
        a_shape.iter().product::<usize>(),
        b_shape.iter().product::<usize>(),
    );
    // Folded scalars become kernel constants, which must be finite.
    let foldable = |v: f32, kind: BinaryKind| {
        v.is_finite() && (!matches!(kind, BinaryKind::Div) || (1.0 / v).is_finite())
    };
    if b_len == 1
        && let Some(v) = scalar_constant(graph, b, weights)
        && foldable(v, kind)
    {
        let a = as_matrix(graph, a, a_shape)?;
        let out = match kind {
            BinaryKind::Add => graph.add_scalar(a, v),
            BinaryKind::Sub => graph.add_scalar(a, -v),
            BinaryKind::Mul => graph.scale(a, v),
            BinaryKind::Div => graph.scale(a, 1.0 / v),
        };
        return Ok(Some((out, broadcast_shape(a_shape, b_shape))));
    }
    if a_len == 1
        && let Some(v) = scalar_constant(graph, a, weights)
        && v.is_finite()
    {
        let b = as_matrix(graph, b, b_shape)?;
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
        return Ok(Some((out, broadcast_shape(a_shape, b_shape))));
    }
    if a_len == b_len {
        let shape = if a_shape.len() >= b_shape.len() {
            a_shape
        } else {
            b_shape
        };
        // Both on the canonical matrix view, so every elementwise use of a
        // tensor reads the same node.
        let a = as_matrix(graph, a, shape)?;
        let b = as_matrix(graph, b, shape)?;
        let out = match kind {
            BinaryKind::Add => graph.add(a, b),
            BinaryKind::Sub => graph.sub(a, b),
            BinaryKind::Mul => graph.mul(a, b),
            BinaryKind::Div => graph.div(a, b),
        };
        return Ok(Some((out, shape.to_vec())));
    }
    if a_len < b_len {
        return match kind {
            BinaryKind::Add | BinaryKind::Mul => {
                broadcast_binary(graph, (b, b_shape), (a, a_shape), kind, weights)
            }
            BinaryKind::Sub | BinaryKind::Div => Ok(None),
        };
    }
    let rank = a_shape.len();
    if b_shape.len() > rank || rank == 0 {
        return Ok(None);
    }
    // `b` padded to `a`'s rank with leading unit axes.
    let mut padded = vec![1; rank - b_shape.len()];
    padded.extend_from_slice(b_shape);
    let cols = a_shape[rank - 1];
    // One value per row: `a`'s shape with the last axis reduced.
    if padded[..rank - 1] == a_shape[..rank - 1] && padded[rank - 1] == 1 {
        let rows = a_len / cols.max(1);
        let a = view(graph, a, &[rows, cols])?;
        let b = view(graph, b, &[rows, 1])?;
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
        return Ok(Some((out, a_shape.to_vec())));
    }
    // Trailing axes: `b` repeats along `a`'s leading ones.
    let leading = padded.iter().take_while(|&&d| d == 1).count();
    if padded[leading..] == a_shape[leading..] && b_len > 0 {
        let a = view(graph, a, &[a_len / b_len, b_len])?;
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
        return Ok(Some((out, a_shape.to_vec())));
    }
    Ok(None)
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

/// `ReduceMean`, `ReduceSum` or `ReduceMax` over the last axis, keeping
/// it: the per-row reductions norms and softmax are written with.
fn reduce_op(
    graph: &mut Graph,
    node: &OnnxNode,
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
    let keepdims = node.attrs.i("keepdims", 1) != 0;
    if !last_axis || !keepdims {
        return Err(OnnxError::UnsupportedOp(format!(
            "{}: axes {axes:?} of {x_shape:?} with keepdims={keepdims} \
             (supported: the last axis, kept)",
            node.op.as_str()
        )));
    }
    let x = as_matrix(graph, x, &x_shape)?;
    let out = match node.op {
        OpKind::ReduceMean => graph.mean_inner(x),
        OpKind::ReduceSum => graph.sum_inner(x),
        _ => graph.max_inner(x),
    };
    let mut out_shape = x_shape;
    if let Some(last) = out_shape.last_mut() {
        *last = 1;
    }
    register_output(node, 0, out, &out_shape, name_to_id, shapes);
    Ok(())
}

fn unary_op(
    graph: &mut Graph,
    node: &OnnxNode,
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
    node: &OnnxNode,
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

/// Compute the broadcast output shape (NumPy-style).
fn broadcast_shape(a: &[usize], b: &[usize]) -> Vec<usize> {
    let len = a.len().max(b.len());
    let mut result = vec![1; len];
    for i in 0..len {
        let da = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let db = if i < b.len() { b[b.len() - 1 - i] } else { 1 };
        result[len - 1 - i] = da.max(db);
    }
    result
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
        assert_eq!(broadcast_shape(&[3, 4], &[4]), vec![3, 4]);
        assert_eq!(broadcast_shape(&[1, 4], &[3, 4]), vec![3, 4]);
        assert_eq!(broadcast_shape(&[2, 1], &[1, 3]), vec![2, 3]);
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
    fn test_parse_input_shapes() {
        // Build a model with a known input shape and verify we parse it
        let weight_init = build_tensor_proto("w", &[10, 5], &[0.0; 50]);
        let matmul_node = build_node_proto("MatMul", &["x", "w"], &["y"], &[], &[]);

        let x_vi = build_value_info("x", &[2, 10]);
        let w_vi = build_value_info("w", &[10, 5]);
        let y_vi = build_value_info("y", &[2, 5]);

        let model_bytes = build_onnx_model(&[matmul_node], &[weight_init], &[x_vi, w_vi], &[y_vi]);

        // Test shape extraction
        let shapes = extract_shapes_from_proto(&model_bytes).unwrap();
        assert_eq!(shapes.get("x"), Some(&vec![2, 10]));
        assert_eq!(shapes.get("w"), Some(&vec![10, 5]));
        assert_eq!(shapes.get("y"), Some(&vec![2, 5]));
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

        let optimized = crate::optimize::optimize(&model.graph);
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
}

#[cfg(test)]
mod length_tests {
    use super::{delimited, parse_single_value_info, parse_tensor_shape_dims};

    #[test]
    fn a_length_past_the_end_is_rejected_not_sliced() {
        let buf = [0u8; 4];
        assert!(delimited(&buf, 1000, 0).is_none());
        assert_eq!(
            delimited(&buf, 4, 0).map(|(f, e)| (f.len(), e)),
            Some((4, 4))
        );
        assert!(delimited(&buf, 0, 5).is_none());
        assert!(delimited(&buf, usize::MAX as u64, 1).is_none());
    }

    #[test]
    fn a_value_info_with_an_oversized_field_does_not_panic() {
        let buf = [0x12, 0x7E, 0x02, 0x03];
        assert!(parse_single_value_info(&buf).is_none());
    }

    #[test]
    fn a_truncated_tensor_shape_does_not_panic() {
        let buf = [0x12, 0xC8, 0x01, 0x08, 0x01];
        let dims = parse_tensor_shape_dims(&buf);
        assert!(
            dims.is_empty(),
            "a truncated field must not yield dimensions, got {dims:?}"
        );
    }

    #[test]
    fn no_length_panics() {
        let buf: Vec<u8> = (0..64u8).collect();
        for start in 0..buf.len() + 2 {
            for len in [0u64, 1, 7, 63, 64, 65, 1 << 20, u64::MAX, u64::MAX - 1] {
                let _ = delimited(&buf, len, start);
            }
        }
    }
}
