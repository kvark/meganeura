//! ONNX model import: load standard ONNX files into Meganeura's `Graph` IR.
//!
//! Translates an ONNX computation graph into our `Graph` at runtime, then the
//! normal pipeline (optimize -> compile -> Session) handles execution. No Rust
//! codegen needed.

use std::collections::HashMap;
use std::path::Path;

use oxionnx_core::{Graph as OnnxGraph, Node as OnnxNode, OpKind};
use oxionnx_proto::model;

use crate::graph::{Graph, NodeId, Op, TensorType};

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
    let (onnx_graph, onnx_weights) = if let Some(p) = path.and_then(|p| p.parent()) {
        model::load_with_path(bytes, p).map_err(OnnxError::ParseError)?
    } else {
        model::load(bytes).map_err(OnnxError::ParseError)?
    };

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
                let len = len as usize;
                if field_no == 7 {
                    return Ok(model_bytes[p..p + len].to_vec());
                }
                pos = p + len;
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
            1 => pos += 8,
            5 => pos += 4,
            2 => {
                let (len, p) =
                    read_proto_varint(graph_bytes, pos).map_err(OnnxError::ParseError)?;
                let len = len as usize;
                if field_no == target_field {
                    // This is a ValueInfoProto — parse name and shape from it
                    if let Some((name, shape)) = parse_single_value_info(&graph_bytes[p..p + len]) {
                        results.push((name, shape));
                    }
                }
                pos = p + len;
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
            1 => pos += 8,
            5 => pos += 4,
            2 => {
                let (len, p) = read_proto_varint(buf, pos).ok()?;
                let len = len as usize;
                match field_no {
                    1 => name = String::from_utf8_lossy(&buf[p..p + len]).into_owned(),
                    2 => type_bytes = Some(&buf[p..p + len]),
                    _ => {}
                }
                pos = p + len;
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
            1 => pos += 8,
            5 => pos += 4,
            2 => {
                let (len, p) = read_proto_varint(buf, pos).ok()?;
                let len = len as usize;
                if field_no == 1 {
                    // tensor_type = TypeProto.Tensor
                    return parse_tensor_type_shape(&buf[p..p + len]);
                }
                pos = p + len;
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
            1 => pos += 8,
            5 => pos += 4,
            2 => {
                let (len, p) = read_proto_varint(buf, pos).ok()?;
                let len = len as usize;
                if field_no == 2 {
                    // shape = TensorShapeProto
                    return Some(parse_tensor_shape_dims(&buf[p..p + len]));
                }
                pos = p + len;
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
            1 => pos += 8,
            5 => pos += 4,
            2 => {
                let Ok((len, p)) = read_proto_varint(buf, pos) else {
                    break;
                };
                let len = len as usize;
                if field_no == 1 {
                    // Dimension message
                    dims.push(parse_dimension(&buf[p..p + len]));
                }
                pos = p + len;
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
            1 => pos += 8,
            5 => pos += 4,
            2 => {
                let Ok((len, p)) = read_proto_varint(buf, pos) else {
                    break;
                };
                pos = p + len as usize;
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

        // Elementary math ops — these should not appear in well-exported ONNX models.
        // They are decomposition artifacts from PyTorch's torch.onnx.export.
        // Use opset >= 17 with SimplifiedLayerNormalization, or export with
        // `optimum-cli` which preserves compound ops.
        OpKind::Sqrt
        | OpKind::Exp
        | OpKind::Tanh
        | OpKind::Erf
        | OpKind::Pow
        | OpKind::ReduceMean
        | OpKind::ReduceSum => {
            return Err(OnnxError::UnsupportedOp(format!(
                "{}: this is likely a decomposed subgraph from torch.onnx.export. \
                 Use `optimum-cli export onnx` or opset extensions \
                 (SimplifiedLayerNormalization, Gelu, etc.) to export compound ops \
                 instead of their decomposed forms",
                node.op.as_str()
            )));
        }

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
        OpKind::Add => binary_op(graph, node, name_to_id, shapes, BinaryKind::Add)?,
        OpKind::Mul => binary_op(graph, node, name_to_id, shapes, BinaryKind::Mul)?,

        // Sub: a - b = a + neg(b)
        OpKind::Sub => {
            let a = resolve_input(&node.inputs[0], name_to_id, &node.name)?;
            let b = resolve_input(&node.inputs[1], name_to_id, &node.name)?;
            let neg_b = graph.neg(b);
            let a_shape = get_shape(&node.inputs[0], shapes);
            let b_shape = get_shape(&node.inputs[1], shapes);
            let out = if a_shape == b_shape {
                graph.add(a, neg_b)
            } else {
                // Broadcast: assume bias-like pattern
                graph.bias_add(a, neg_b)
            };
            register_output(node, 0, out, &a_shape, name_to_id, shapes);
        }

        // Div: a / b = a * recip(b)
        OpKind::Div => {
            let a = resolve_input(&node.inputs[0], name_to_id, &node.name)?;
            let b = resolve_input(&node.inputs[1], name_to_id, &node.name)?;
            let out = graph.div(a, b);
            let a_shape = get_shape(&node.inputs[0], shapes);
            register_output(node, 0, out, &a_shape, name_to_id, shapes);
        }

        // --- MatMul ---
        OpKind::MatMul => {
            let a = resolve_input(&node.inputs[0], name_to_id, &node.name)?;
            let b = resolve_input(&node.inputs[1], name_to_id, &node.name)?;
            let a_shape = get_shape(&node.inputs[0], shapes);
            let b_shape = get_shape(&node.inputs[1], shapes);
            // Flatten to 2D if needed
            let a_2d = flatten_to_2d(&a_shape);
            let b_2d = flatten_to_2d(&b_shape);
            let out = graph.matmul(a, b);
            let out_shape = if a_2d.len() == 2 && b_2d.len() == 2 {
                vec![a_2d[0], b_2d[1]]
            } else {
                vec![
                    a_shape.first().copied().unwrap_or(1),
                    b_shape.last().copied().unwrap_or(1),
                ]
            };
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
        OpKind::Softmax => unary_op(graph, node, name_to_id, shapes, Op::Softmax)?,
        OpKind::LogSoftmax => unary_op(graph, node, name_to_id, shapes, Op::LogSoftmax)?,

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
            let out = graph.layer_norm(x, scale, bias, eps);
            let x_shape = get_shape(&node.inputs[0], shapes);
            register_output(node, 0, out, &x_shape, name_to_id, shapes);
        }

        OpKind::RMSNorm => {
            let x = resolve_input(&node.inputs[0], name_to_id, &node.name)?;
            let scale = resolve_input(&node.inputs[1], name_to_id, &node.name)?;
            let eps = attrs.f("epsilon", 1e-5);
            let out = graph.rms_norm(x, scale, eps);
            let x_shape = get_shape(&node.inputs[0], shapes);
            register_output(node, 0, out, &x_shape, name_to_id, shapes);
        }

        // --- Embedding (Gather with axis=0) ---
        OpKind::Gather => {
            let axis = attrs.i("axis", 0);
            if axis != 0 {
                return Err(OnnxError::UnsupportedOp(format!(
                    "Gather with axis={axis} (only axis=0 supported)"
                )));
            }
            // ONNX Gather: data[indices] where data is the table
            // Meganeura embedding: (indices, table) -> output
            // Note: ONNX input order is (data, indices), we need (indices, data)
            let table = resolve_input(&node.inputs[0], name_to_id, &node.name)?;
            let indices = resolve_input(&node.inputs[1], name_to_id, &node.name)?;
            let out = graph.embedding(indices, table);
            let table_shape = get_shape(&node.inputs[0], shapes);
            let indices_shape = get_shape(&node.inputs[1], shapes);
            let hidden = table_shape.get(1).copied().unwrap_or(1);
            let seq_len = indices_shape.iter().product::<usize>().max(1);
            register_output(node, 0, out, &[seq_len, hidden], name_to_id, shapes);
        }

        // --- Transpose ---
        OpKind::Transpose => {
            let x = resolve_input(&node.inputs[0], name_to_id, &node.name)?;
            let x_shape = get_shape(&node.inputs[0], shapes);
            let perm = attrs.ints("perm");

            if x_shape.len() == 2 && (perm.is_empty() || perm == [1, 0]) {
                let out = graph.transpose(x);
                let out_shape = vec![x_shape[1], x_shape[0]];
                register_output(node, 0, out, &out_shape, name_to_id, shapes);
            } else if perm.is_empty() {
                // Default: reverse all dims. Only support 2D.
                if x_shape.len() == 2 {
                    let out = graph.transpose(x);
                    let out_shape = vec![x_shape[1], x_shape[0]];
                    register_output(node, 0, out, &out_shape, name_to_id, shapes);
                } else {
                    return Err(OnnxError::UnsupportedOp(format!(
                        "Transpose with {}D (only 2D supported)",
                        x_shape.len()
                    )));
                }
            } else {
                return Err(OnnxError::UnsupportedOp(format!(
                    "Transpose with perm={perm:?} (only [1,0] or default supported)"
                )));
            }
        }

        // --- Reshape / Flatten / Squeeze / Unsqueeze ---
        // These are shape-only ops. In our flat IR, they are identity if the total
        // element count doesn't change. We track the new shape for downstream ops.
        OpKind::Reshape => {
            let x = resolve_input(&node.inputs[0], name_to_id, &node.name)?;
            let x_shape = get_shape(&node.inputs[0], shapes);
            let total = x_shape.iter().product::<usize>().max(1);

            // Get target shape from the second input (should be a constant)
            let new_shape = if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
                if let Some(shape_data) = weights.get(&node.inputs[1]) {
                    resolve_reshape_dims(shape_data, total)
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
            // Shape-only: just propagate with adjusted shape
            let x = resolve_input(&node.inputs[0], name_to_id, &node.name)?;
            let x_shape = get_shape(&node.inputs[0], shapes);
            // Compute new shape (simplified)
            let new_shape = match node.op {
                OpKind::Squeeze => x_shape.iter().copied().filter(|&d| d != 1).collect(),
                OpKind::Unsqueeze => {
                    let axes = attrs.ints("axes");
                    let mut s = x_shape.clone();
                    for &ax in axes.iter().rev() {
                        let pos = if ax < 0 {
                            (s.len() as i64 + ax + 1) as usize
                        } else {
                            ax as usize
                        };
                        s.insert(pos.min(s.len()), 1);
                    }
                    s
                }
                _ => unreachable!(),
            };
            if !node.outputs.is_empty() {
                name_to_id.insert(node.outputs[0].clone(), x);
                shapes.insert(node.outputs[0].clone(), new_shape);
            }
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
            let axis = attrs.i("axis", 0);
            if node.inputs.len() != 2 {
                return Err(OnnxError::UnsupportedOp(format!(
                    "Concat with {} inputs (only 2 supported)",
                    node.inputs.len()
                )));
            }
            let a = resolve_input(&node.inputs[0], name_to_id, &node.name)?;
            let b = resolve_input(&node.inputs[1], name_to_id, &node.name)?;
            let a_shape = get_shape(&node.inputs[0], shapes);
            let b_shape = get_shape(&node.inputs[1], shapes);

            // Only channel-dim concat for 4D (NCHW)
            if a_shape.len() == 4 && axis == 1 {
                let batch = a_shape[0] as u32;
                let ca = a_shape[1] as u32;
                let cb = b_shape[1] as u32;
                let spatial = (a_shape[2] * a_shape[3]) as u32;
                let out = graph.concat(a, b, batch, ca, cb, spatial);
                let out_shape = vec![a_shape[0], a_shape[1] + b_shape[1], a_shape[2], a_shape[3]];
                register_output(node, 0, out, &out_shape, name_to_id, shapes);
            } else {
                return Err(OnnxError::UnsupportedOp(format!(
                    "Concat on axis={axis} with {}D tensors (only NCHW channel concat supported)",
                    a_shape.len()
                )));
            }
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

enum BinaryKind {
    Add,
    Mul,
}

fn binary_op(
    graph: &mut Graph,
    node: &OnnxNode,
    name_to_id: &mut HashMap<String, NodeId>,
    shapes: &mut HashMap<String, Vec<usize>>,
    kind: BinaryKind,
) -> Result<(), OnnxError> {
    let a = resolve_input(&node.inputs[0], name_to_id, &node.name)?;
    let b = resolve_input(&node.inputs[1], name_to_id, &node.name)?;
    let a_shape = get_shape(&node.inputs[0], shapes);
    let b_shape = get_shape(&node.inputs[1], shapes);

    let out = if a_shape == b_shape {
        match kind {
            BinaryKind::Add => graph.add(a, b),
            BinaryKind::Mul => graph.mul(a, b),
        }
    } else {
        // Broadcast: smaller tensor is the bias/scalar
        match kind {
            BinaryKind::Add => graph.bias_add(a, b),
            BinaryKind::Mul => {
                // Element-wise mul doesn't have a broadcast variant in our IR,
                // but for same-total-elements it works as-is
                graph.mul(a, b)
            }
        }
    };

    let out_shape = broadcast_shape(&a_shape, &b_shape);
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
    let ty = TensorType::f32(x_shape.clone());
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
fn resolve_reshape_dims(shape_data: &[f32], total_elements: usize) -> Vec<usize> {
    let raw: Vec<i64> = shape_data.iter().map(|&v| v as i64).collect();
    let mut neg_idx = None;
    let mut known_product = 1usize;

    for (i, &d) in raw.iter().enumerate() {
        if d == -1 {
            neg_idx = Some(i);
        } else if d == 0 {
            // 0 means "keep original dim" — we don't track original per-dim, use 1
        } else {
            known_product *= d as usize;
        }
    }

    let mut result: Vec<usize> = raw
        .iter()
        .map(|&d| if d == -1 || d == 0 { 1 } else { d as usize })
        .collect();

    if let Some(idx) = neg_idx {
        result[idx] = total_elements / known_product.max(1);
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
        assert_eq!(resolve_reshape_dims(&[2.0, -1.0], 6), vec![2, 3]);
        assert_eq!(resolve_reshape_dims(&[3.0, 4.0], 12), vec![3, 4]);
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
}
