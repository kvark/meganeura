//! NNEF model import: load Khronos NNEF models into Meganeura's `Graph` IR.
//!
//! NNEF (Neural Network Exchange Format) models are directories containing:
//! - `graph.nnef` — human-readable text describing the computation graph
//! - `*.dat` — binary tensor files for weights
//!
//! The text format is simple and declarative:
//! ```text
//! version 1.0;
//! graph G( input ) -> ( output )
//! {
//!     input = external(shape = [1, 3, 224, 224]);
//!     filter = variable(shape = [64, 3, 3, 3], label = 'conv1/filter');
//!     conv1 = conv(input, filter, padding = [(1,1),(1,1)]);
//!     output = relu(conv1);
//! }
//! ```

use std::collections::HashMap;
use std::path::Path;

use crate::graph::{Graph, NodeId, Op, TensorType};

/// Result of loading an NNEF model.
pub struct NnefModel {
    /// The computation graph, ready for optimize() and compile().
    pub graph: Graph,
    /// Named weight tensors extracted from .dat files.
    pub weights: HashMap<String, Vec<f32>>,
}

/// Errors during NNEF import.
#[derive(Debug)]
pub enum NnefError {
    Io(std::io::Error),
    ParseError(String),
    UnsupportedOp(String),
    ShapeError(String),
}

impl std::fmt::Display for NnefError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::Io(ref e) => write!(f, "NNEF I/O error: {e}"),
            Self::ParseError(ref e) => write!(f, "NNEF parse error: {e}"),
            Self::UnsupportedOp(ref e) => write!(f, "unsupported NNEF op: {e}"),
            Self::ShapeError(ref e) => write!(f, "NNEF shape error: {e}"),
        }
    }
}

impl std::error::Error for NnefError {}

impl From<std::io::Error> for NnefError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

// ─── Binary tensor format ───────────────────────────────────────

const NNEF_MAGIC: [u8; 2] = [0x4E, 0xEF];

/// Read an NNEF binary tensor (.dat) file.
fn read_tensor_dat(path: &Path) -> Result<(Vec<usize>, Vec<f32>), NnefError> {
    let data = std::fs::read(path)?;
    if data.len() < 128 {
        return Err(NnefError::ParseError(format!(
            "{}: file too small for NNEF header ({}B)",
            path.display(),
            data.len()
        )));
    }
    if data[0..2] != NNEF_MAGIC {
        return Err(NnefError::ParseError(format!(
            "{}: bad magic (expected 0x4EEF)",
            path.display()
        )));
    }
    // Header layout (128 bytes):
    //   [0..2]   magic
    //   [2..4]   version [major, minor]
    //   [4..8]   data_length: u32 LE
    //   [8..12]  rank: u32 LE
    //   [12..44] extents: [u32; 8] LE
    //   [44..48] bits_per_item: u32 LE
    //   [48..52] item_type: u32 LE (0=Float)
    //   [52..128] reserved
    let data_length = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
    let rank = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
    let mut extents = Vec::with_capacity(rank);
    for i in 0..rank.min(8) {
        let off = 12 + i * 4;
        extents.push(
            u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]) as usize,
        );
    }
    let bits_per_item = u32::from_le_bytes([data[44], data[45], data[46], data[47]]);
    let item_type = u32::from_le_bytes([data[48], data[49], data[50], data[51]]);

    let tensor_data = &data[128..128 + data_length.min(data.len() - 128)];

    let floats = match (item_type, bits_per_item) {
        (0, 32) => {
            // Float32
            tensor_data
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        }
        (0, 16) => {
            // Float16 → f32
            tensor_data
                .chunks_exact(2)
                .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect()
        }
        _ => {
            return Err(NnefError::ParseError(format!(
                "{}: unsupported tensor type={item_type} bits={bits_per_item}",
                path.display()
            )));
        }
    };

    Ok((extents, floats))
}

// ─── Text parser for graph.nnef ─────────────────────────────────

/// A parsed NNEF value (argument to an operation).
#[derive(Debug, Clone)]
enum Value {
    Ident(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
}

impl Value {
    fn as_int(&self) -> Option<i64> {
        match *self {
            Value::Int(v) => Some(v),
            Value::Float(v) => Some(v as i64),
            _ => None,
        }
    }
    fn as_float(&self) -> Option<f64> {
        match *self {
            Value::Float(v) => Some(v),
            Value::Int(v) => Some(v as f64),
            _ => None,
        }
    }
    fn as_int_array(&self) -> Option<Vec<i64>> {
        match *self {
            Value::Array(ref arr) => arr.iter().map(|v| v.as_int()).collect(),
            _ => None,
        }
    }
}

/// A parsed NNEF assignment: `name = op(args..., key=val, ...);`
#[derive(Debug)]
struct Assignment {
    name: String,
    op: String,
    positional: Vec<Value>,
    named: HashMap<String, Value>,
}

/// Parse a complete graph.nnef file.
type ParseResult = Result<(Vec<String>, Vec<String>, Vec<Assignment>), NnefError>;

fn parse_graph_nnef(text: &str) -> ParseResult {
    let mut assignments = Vec::new();
    let mut graph_inputs = Vec::new();
    let mut graph_outputs = Vec::new();

    // Skip to graph body
    // Expected: version 1.0; graph NAME( inputs ) -> ( outputs ) { body }
    let full = text.to_string();

    // Find the graph body between { ... }
    let body_start = full
        .find('{')
        .ok_or_else(|| NnefError::ParseError("no '{' found".into()))?;
    let body_end = full
        .rfind('}')
        .ok_or_else(|| NnefError::ParseError("no '}' found".into()))?;

    // Parse graph header for input/output names
    let header = &full[..body_start];
    if let Some(graph_pos) = header.find("graph") {
        let after_graph = &header[graph_pos + 5..];
        // Find ( inputs ) -> ( outputs )
        if let Some(paren1) = after_graph.find('(') {
            let paren1_end = after_graph[paren1..]
                .find(')')
                .map(|p| paren1 + p)
                .unwrap_or(paren1);
            let inputs_str = &after_graph[paren1 + 1..paren1_end];
            graph_inputs = inputs_str
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();

            let rest = &after_graph[paren1_end + 1..];
            if let Some(arrow) = rest.find("->") {
                let after_arrow = &rest[arrow + 2..];
                if let Some(p2) = after_arrow.find('(') {
                    let p2_end = after_arrow[p2..].find(')').map(|p| p2 + p).unwrap_or(p2);
                    let outputs_str = &after_arrow[p2 + 1..p2_end];
                    graph_outputs = outputs_str
                        .split(',')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect();
                }
            }
        }
    }

    // Parse body: sequence of "name = op(...);" assignments
    let body = &full[body_start + 1..body_end];

    // Simple statement-level parser
    let mut pos = 0;
    let body_bytes = body.as_bytes();
    while pos < body_bytes.len() {
        // Skip whitespace and comments
        while pos < body_bytes.len()
            && (body_bytes[pos].is_ascii_whitespace() || body_bytes[pos] == b'#')
        {
            if body_bytes[pos] == b'#' {
                while pos < body_bytes.len() && body_bytes[pos] != b'\n' {
                    pos += 1;
                }
            } else {
                pos += 1;
            }
        }
        if pos >= body_bytes.len() {
            break;
        }

        // Read identifier (assignment target)
        let name_start = pos;
        while pos < body_bytes.len()
            && (body_bytes[pos].is_ascii_alphanumeric() || body_bytes[pos] == b'_')
        {
            pos += 1;
        }
        let name = std::str::from_utf8(&body_bytes[name_start..pos])
            .unwrap()
            .to_string();
        if name.is_empty() {
            break;
        }

        // Skip whitespace, expect '='
        while pos < body_bytes.len() && body_bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if pos >= body_bytes.len() || body_bytes[pos] != b'=' {
            break;
        }
        pos += 1;
        while pos < body_bytes.len() && body_bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }

        // Read op name
        let op_start = pos;
        while pos < body_bytes.len()
            && (body_bytes[pos].is_ascii_alphanumeric() || body_bytes[pos] == b'_')
        {
            pos += 1;
        }
        let op = std::str::from_utf8(&body_bytes[op_start..pos])
            .unwrap()
            .to_string();

        // Parse arguments in parentheses
        while pos < body_bytes.len() && body_bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        let mut positional = Vec::new();
        let mut named = HashMap::new();

        if pos < body_bytes.len() && body_bytes[pos] == b'(' {
            pos += 1; // skip '('
            loop {
                while pos < body_bytes.len() && body_bytes[pos].is_ascii_whitespace() {
                    pos += 1;
                }
                if pos >= body_bytes.len() || body_bytes[pos] == b')' {
                    pos += 1;
                    break;
                }

                // Parse a value or key=value
                let (val, new_pos) = parse_value(body, pos)?;
                pos = new_pos;
                while pos < body_bytes.len() && body_bytes[pos].is_ascii_whitespace() {
                    pos += 1;
                }

                // Check if this is key=value
                if pos < body_bytes.len() && body_bytes[pos] == b'=' {
                    pos += 1;
                    while pos < body_bytes.len() && body_bytes[pos].is_ascii_whitespace() {
                        pos += 1;
                    }
                    let key = match val {
                        Value::Ident(k) => k,
                        _ => {
                            return Err(NnefError::ParseError(
                                "expected identifier before '='".into(),
                            ));
                        }
                    };
                    let (val2, new_pos2) = parse_value(body, pos)?;
                    pos = new_pos2;
                    named.insert(key, val2);
                } else {
                    positional.push(val);
                }

                while pos < body_bytes.len() && body_bytes[pos].is_ascii_whitespace() {
                    pos += 1;
                }
                if pos < body_bytes.len() && body_bytes[pos] == b',' {
                    pos += 1;
                }
            }
        }

        // Skip semicolon
        while pos < body_bytes.len() && body_bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if pos < body_bytes.len() && body_bytes[pos] == b';' {
            pos += 1;
        }

        assignments.push(Assignment {
            name,
            op,
            positional,
            named,
        });
    }

    Ok((graph_inputs, graph_outputs, assignments))
}

/// Parse a single value starting at `pos` in `text`.
fn parse_value(text: &str, mut pos: usize) -> Result<(Value, usize), NnefError> {
    let bytes = text.as_bytes();
    while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
        pos += 1;
    }

    if pos >= bytes.len() {
        return Err(NnefError::ParseError("unexpected end of input".into()));
    }

    match bytes[pos] {
        // String literal
        b'\'' | b'"' => {
            let quote = bytes[pos];
            pos += 1;
            let start = pos;
            while pos < bytes.len() && bytes[pos] != quote {
                pos += 1;
            }
            let s = std::str::from_utf8(&bytes[start..pos]).unwrap().to_string();
            pos += 1; // skip closing quote
            Ok((Value::String(s), pos))
        }
        // Array
        b'[' => {
            pos += 1;
            let mut items = Vec::new();
            loop {
                while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
                    pos += 1;
                }
                if pos >= bytes.len() || bytes[pos] == b']' {
                    pos += 1;
                    break;
                }
                let (val, new_pos) = parse_value(text, pos)?;
                items.push(val);
                pos = new_pos;
                while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
                    pos += 1;
                }
                if pos < bytes.len() && bytes[pos] == b',' {
                    pos += 1;
                }
            }
            Ok((Value::Array(items), pos))
        }
        // Tuple
        b'(' => {
            pos += 1;
            let mut items = Vec::new();
            loop {
                while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
                    pos += 1;
                }
                if pos >= bytes.len() || bytes[pos] == b')' {
                    pos += 1;
                    break;
                }
                let (val, new_pos) = parse_value(text, pos)?;
                items.push(val);
                pos = new_pos;
                while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
                    pos += 1;
                }
                if pos < bytes.len() && bytes[pos] == b',' {
                    pos += 1;
                }
            }
            Ok((Value::Array(items), pos))
        }
        // Number or identifier
        _ => {
            let start = pos;
            // Check for negative number
            if bytes[pos] == b'-' {
                pos += 1;
            }

            if pos < bytes.len() && bytes[pos].is_ascii_digit() {
                // Number
                while pos < bytes.len() && (bytes[pos].is_ascii_digit() || bytes[pos] == b'.') {
                    pos += 1;
                }
                // Check for exponent
                if pos < bytes.len() && (bytes[pos] == b'e' || bytes[pos] == b'E') {
                    pos += 1;
                    if pos < bytes.len() && (bytes[pos] == b'+' || bytes[pos] == b'-') {
                        pos += 1;
                    }
                    while pos < bytes.len() && bytes[pos].is_ascii_digit() {
                        pos += 1;
                    }
                }
                let s = &text[start..pos];
                if s.contains('.') || s.contains('e') || s.contains('E') {
                    Ok((
                        Value::Float(
                            s.parse()
                                .map_err(|e| NnefError::ParseError(format!("bad float: {e}")))?,
                        ),
                        pos,
                    ))
                } else {
                    Ok((
                        Value::Int(
                            s.parse()
                                .map_err(|e| NnefError::ParseError(format!("bad int: {e}")))?,
                        ),
                        pos,
                    ))
                }
            } else {
                // Identifier or keyword
                while pos < bytes.len()
                    && (bytes[pos].is_ascii_alphanumeric()
                        || bytes[pos] == b'_'
                        || bytes[pos] == b'.')
                {
                    pos += 1;
                }
                let s = &text[start..pos];
                match s {
                    "true" => Ok((Value::Bool(true), pos)),
                    "false" => Ok((Value::Bool(false), pos)),
                    _ => Ok((Value::Ident(s.to_string()), pos)),
                }
            }
        }
    }
}

// ─── Graph translation ──────────────────────────────────────────

/// Load an NNEF model from a directory path.
///
/// Expects `path/graph.nnef` and tensor `.dat` files.
pub fn load_nnef(path: &Path) -> Result<NnefModel, NnefError> {
    let graph_path = if path.is_dir() {
        path.join("graph.nnef")
    } else {
        path.to_path_buf()
    };
    let base_dir = graph_path.parent().unwrap_or(Path::new("."));

    let text = std::fs::read_to_string(&graph_path)?;
    let (_input_names, output_names, assignments) = parse_graph_nnef(&text)?;

    let mut graph = Graph::new();
    let mut name_to_id: HashMap<String, NodeId> = HashMap::new();
    let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();
    let mut weights: HashMap<String, Vec<f32>> = HashMap::new();

    for stmt in &assignments {
        match stmt.op.as_str() {
            "external" => {
                let shape = stmt
                    .named
                    .get("shape")
                    .and_then(|v| v.as_int_array())
                    .map(|v| v.into_iter().map(|d| d as usize).collect::<Vec<_>>())
                    .unwrap_or_else(|| vec![1]);
                let flat = flatten_to_2d(&shape);
                let id = graph.input(&stmt.name, &flat);
                name_to_id.insert(stmt.name.clone(), id);
                shapes.insert(stmt.name.clone(), shape);
            }

            "variable" => {
                let shape = stmt
                    .named
                    .get("shape")
                    .and_then(|v| v.as_int_array())
                    .map(|v| v.into_iter().map(|d| d as usize).collect::<Vec<_>>())
                    .unwrap_or_else(|| vec![1]);
                let id = graph.parameter(&stmt.name, &shape);
                name_to_id.insert(stmt.name.clone(), id);
                shapes.insert(stmt.name.clone(), shape.clone());

                // Load tensor data from .dat file
                let label = stmt
                    .named
                    .get("label")
                    .and_then(|v| match *v {
                        Value::String(ref s) => Some(s.as_str()),
                        _ => None,
                    })
                    .unwrap_or(&stmt.name);
                let dat_path = base_dir.join(format!(
                    "{}.dat",
                    label.replace('/', std::path::MAIN_SEPARATOR_STR)
                ));
                if dat_path.exists() {
                    let (_tensor_shape, data) = read_tensor_dat(&dat_path)?;
                    weights.insert(stmt.name.clone(), data);
                }
            }

            "constant" => {
                let shape = stmt
                    .named
                    .get("shape")
                    .and_then(|v| v.as_int_array())
                    .map(|v| v.into_iter().map(|d| d as usize).collect::<Vec<_>>())
                    .unwrap_or_else(|| vec![1]);
                let value = stmt
                    .named
                    .get("value")
                    .and_then(|v| v.as_float())
                    .unwrap_or(0.0) as f32;
                let n: usize = shape.iter().product();
                let id = graph.constant(vec![value; n], &shape);
                name_to_id.insert(stmt.name.clone(), id);
                shapes.insert(stmt.name.clone(), shape);
            }

            // --- Arithmetic ---
            "add" | "sub" | "mul" | "div" => {
                let a = resolve(&stmt.positional[0], &name_to_id, &stmt.name)?;
                let b = resolve(&stmt.positional[1], &name_to_id, &stmt.name)?;
                let a_shape = get_shape_v(&stmt.positional[0], &shapes);
                let out = match stmt.op.as_str() {
                    "add" => graph.add(a, b),
                    "sub" => {
                        let nb = graph.neg(b);
                        graph.add(a, nb)
                    }
                    "mul" => graph.mul(a, b),
                    "div" => graph.div(a, b),
                    _ => unreachable!(),
                };
                name_to_id.insert(stmt.name.clone(), out);
                shapes.insert(stmt.name.clone(), a_shape);
            }

            // --- Unary ---
            "relu" => unary(&mut graph, stmt, &mut name_to_id, &mut shapes, Op::Relu)?,
            "sigmoid" => unary(&mut graph, stmt, &mut name_to_id, &mut shapes, Op::Sigmoid)?,
            "neg" => unary(&mut graph, stmt, &mut name_to_id, &mut shapes, Op::Neg)?,
            "abs" => unary(&mut graph, stmt, &mut name_to_id, &mut shapes, Op::Abs)?,
            "log" => unary(&mut graph, stmt, &mut name_to_id, &mut shapes, Op::Log)?,
            "rcp" => unary(&mut graph, stmt, &mut name_to_id, &mut shapes, Op::Recip)?,
            "silu" => unary(&mut graph, stmt, &mut name_to_id, &mut shapes, Op::Silu)?,
            "gelu" => unary(&mut graph, stmt, &mut name_to_id, &mut shapes, Op::Gelu)?,
            "softmax" => unary(&mut graph, stmt, &mut name_to_id, &mut shapes, Op::Softmax)?,

            // --- MatMul ---
            "matmul" => {
                let a = resolve(&stmt.positional[0], &name_to_id, &stmt.name)?;
                let b = resolve(&stmt.positional[1], &name_to_id, &stmt.name)?;
                let trans_a = stmt
                    .named
                    .get("transposeA")
                    .and_then(|v| match *v {
                        Value::Bool(b) => Some(b),
                        _ => None,
                    })
                    .unwrap_or(false);
                let trans_b = stmt
                    .named
                    .get("transposeB")
                    .and_then(|v| match *v {
                        Value::Bool(b) => Some(b),
                        _ => None,
                    })
                    .unwrap_or(false);
                let a_shape = get_shape_v(&stmt.positional[0], &shapes);
                let b_shape = get_shape_v(&stmt.positional[1], &shapes);

                let out = match (trans_a, trans_b) {
                    (false, false) => graph.matmul(a, b),
                    (true, false) => graph.matmul_at(a, b),
                    (false, true) => graph.matmul_bt(a, b),
                    (true, true) => {
                        let ba = graph.matmul(b, a);
                        graph.transpose(ba)
                    }
                };

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
                name_to_id.insert(stmt.name.clone(), out);
                shapes.insert(stmt.name.clone(), vec![m, n]);
            }

            // --- Linear: matmul(input, filter, transposeB=true) + bias ---
            "linear" => {
                let input = resolve(&stmt.positional[0], &name_to_id, &stmt.name)?;
                let filter = resolve(&stmt.positional[1], &name_to_id, &stmt.name)?;
                let mm = graph.matmul_bt(input, filter);
                let out = if stmt.positional.len() > 2 {
                    let bias = resolve(&stmt.positional[2], &name_to_id, &stmt.name)?;
                    graph.bias_add(mm, bias)
                } else {
                    mm
                };
                let in_shape = get_shape_v(&stmt.positional[0], &shapes);
                let filter_shape = get_shape_v(&stmt.positional[1], &shapes);
                let out_dim = filter_shape.first().copied().unwrap_or(1);
                name_to_id.insert(stmt.name.clone(), out);
                shapes.insert(
                    stmt.name.clone(),
                    vec![in_shape.first().copied().unwrap_or(1), out_dim],
                );
            }

            // --- Reshape ---
            "reshape" => {
                let x = resolve(&stmt.positional[0], &name_to_id, &stmt.name)?;
                let new_shape = stmt
                    .named
                    .get("shape")
                    .and_then(|v| v.as_int_array())
                    .map(|v| v.into_iter().map(|d| d as usize).collect::<Vec<_>>())
                    .unwrap_or_default();
                name_to_id.insert(stmt.name.clone(), x);
                shapes.insert(stmt.name.clone(), new_shape);
            }

            // --- Transpose ---
            "transpose" => {
                let x = resolve(&stmt.positional[0], &name_to_id, &stmt.name)?;
                let x_shape = get_shape_v(&stmt.positional[0], &shapes);
                if x_shape.len() == 2 {
                    let out = graph.transpose(x);
                    name_to_id.insert(stmt.name.clone(), out);
                    shapes.insert(stmt.name.clone(), vec![x_shape[1], x_shape[0]]);
                } else {
                    // For non-2D, treat as identity with reordered shape
                    name_to_id.insert(stmt.name.clone(), x);
                    let mut rev = x_shape;
                    rev.reverse();
                    shapes.insert(stmt.name.clone(), rev);
                }
            }

            // --- Concat ---
            "concat" => {
                let values = &stmt.positional[0];
                let axis = stmt.named.get("axis").and_then(|v| v.as_int()).unwrap_or(0) as usize;
                if let Value::Array(ref items) = *values {
                    if items.len() == 2 {
                        let a = resolve(&items[0], &name_to_id, &stmt.name)?;
                        let b = resolve(&items[1], &name_to_id, &stmt.name)?;
                        let a_shape = get_shape_v(&items[0], &shapes);
                        let b_shape = get_shape_v(&items[1], &shapes);
                        if a_shape.len() == 4 && axis == 1 {
                            let batch = a_shape[0] as u32;
                            let ca = a_shape[1] as u32;
                            let cb = b_shape[1] as u32;
                            let spatial = (a_shape[2] * a_shape[3]) as u32;
                            let out = graph.concat(a, b, batch, ca, cb, spatial);
                            let mut out_shape = a_shape.clone();
                            out_shape[1] += b_shape[1];
                            name_to_id.insert(stmt.name.clone(), out);
                            shapes.insert(stmt.name.clone(), out_shape);
                        } else {
                            return Err(NnefError::UnsupportedOp(format!(
                                "concat on axis={axis} with {}D tensors",
                                a_shape.len()
                            )));
                        }
                    } else {
                        return Err(NnefError::UnsupportedOp(format!(
                            "concat with {} inputs (only 2 supported)",
                            items.len()
                        )));
                    }
                } else {
                    return Err(NnefError::ParseError(
                        "concat expects array argument".into(),
                    ));
                }
            }

            // --- Copy (identity) ---
            "copy" => {
                let x = resolve(&stmt.positional[0], &name_to_id, &stmt.name)?;
                let x_shape = get_shape_v(&stmt.positional[0], &shapes);
                name_to_id.insert(stmt.name.clone(), x);
                shapes.insert(stmt.name.clone(), x_shape);
            }

            // --- Unsupported ---
            other => {
                return Err(NnefError::UnsupportedOp(other.to_string()));
            }
        }
    }

    // Set graph outputs
    let output_ids: Vec<NodeId> = output_names
        .iter()
        .filter_map(|name| name_to_id.get(name).copied())
        .collect();
    graph.set_outputs(output_ids);

    Ok(NnefModel { graph, weights })
}

// ─── Helpers ────────────────────────────────────────────────────

fn flatten_to_2d(shape: &[usize]) -> Vec<usize> {
    if shape.len() <= 2 {
        return shape.to_vec();
    }
    let last = *shape.last().unwrap_or(&1);
    let batch: usize = shape[..shape.len() - 1].iter().product();
    vec![batch, last]
}

fn resolve(
    val: &Value,
    name_to_id: &HashMap<String, NodeId>,
    ctx: &str,
) -> Result<NodeId, NnefError> {
    match *val {
        Value::Ident(ref name) => name_to_id
            .get(name)
            .copied()
            .ok_or_else(|| NnefError::ShapeError(format!("{ctx}: '{name}' not found"))),
        _ => Err(NnefError::ParseError(format!(
            "{ctx}: expected identifier, got {val:?}"
        ))),
    }
}

fn get_shape_v(val: &Value, shapes: &HashMap<String, Vec<usize>>) -> Vec<usize> {
    match *val {
        Value::Ident(ref name) => shapes.get(name).cloned().unwrap_or_default(),
        _ => Vec::new(),
    }
}

fn unary(
    graph: &mut Graph,
    stmt: &Assignment,
    name_to_id: &mut HashMap<String, NodeId>,
    shapes: &mut HashMap<String, Vec<usize>>,
    op: Op,
) -> Result<(), NnefError> {
    let x = resolve(&stmt.positional[0], name_to_id, &stmt.name)?;
    let x_shape = get_shape_v(&stmt.positional[0], shapes);
    let ty = TensorType::f32(x_shape.clone());
    let out = graph.add_raw_node(op, vec![x], ty);
    name_to_id.insert(stmt.name.clone(), out);
    shapes.insert(stmt.name.clone(), x_shape);
    Ok(())
}

// ─── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_graph() {
        let text = r#"
version 1.0;
graph G( input ) -> ( output )
{
    input = external(shape = [1, 784]);
    w = variable(shape = [256, 784], label = 'layer1/weight');
    output = matmul(input, w, transposeB = true);
}
"#;
        let (inputs, outputs, stmts) = parse_graph_nnef(text).unwrap();
        assert_eq!(inputs, vec!["input"]);
        assert_eq!(outputs, vec!["output"]);
        assert_eq!(stmts.len(), 3);
        assert_eq!(stmts[0].op, "external");
        assert_eq!(stmts[1].op, "variable");
        assert_eq!(stmts[2].op, "matmul");
    }

    #[test]
    fn test_parse_mlp_graph() {
        let text = r#"
version 1.0;
graph MLP( x ) -> ( out )
{
    x = external(shape = [1, 4]);
    w1 = variable(shape = [8, 4], label = 'w1');
    b1 = variable(shape = [1, 8], label = 'b1');
    h1 = linear(x, w1, b1);
    h1_relu = relu(h1);
    w2 = variable(shape = [3, 8], label = 'w2');
    out = linear(h1_relu, w2);
}
"#;
        let (inputs, outputs, stmts) = parse_graph_nnef(text).unwrap();
        assert_eq!(inputs, vec!["x"]);
        assert_eq!(outputs, vec!["out"]);
        assert_eq!(stmts.len(), 7);
        assert_eq!(stmts[3].op, "linear");
        assert_eq!(stmts[4].op, "relu");
    }

    #[test]
    fn test_translate_simple_graph() {
        // Build an in-memory NNEF model (no .dat files)
        let text = r#"
version 1.0;
graph G( x ) -> ( out )
{
    x = external(shape = [1, 4]);
    w = variable(shape = [3, 4], label = 'w');
    out = matmul(x, w, transposeB = true);
}
"#;
        // Create a temp dir with graph.nnef
        let tmp = std::env::temp_dir().join("nnef_test_simple");
        let _ = std::fs::create_dir_all(&tmp);
        std::fs::write(tmp.join("graph.nnef"), text).unwrap();

        let result = load_nnef(&tmp);
        assert!(result.is_ok(), "load failed: {:?}", result.err());

        let model = result.unwrap();
        assert_eq!(model.graph.outputs().len(), 1);
        // input + parameter + matmul_bt = 3 nodes
        assert!(model.graph.nodes().len() >= 3);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_binary_tensor_roundtrip() {
        // Create a minimal .dat file and read it back
        let tmp = std::env::temp_dir().join("nnef_test_tensor");
        let _ = std::fs::create_dir_all(&tmp);
        let dat_path = tmp.join("test.dat");

        let shape = [2u32, 3u32];
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        // Build header
        let mut header = vec![0u8; 128];
        header[0..2].copy_from_slice(&NNEF_MAGIC);
        header[2] = 1;
        header[3] = 0; // version 1.0
        let data_len = (data.len() * 4) as u32;
        header[4..8].copy_from_slice(&data_len.to_le_bytes());
        header[8..12].copy_from_slice(&2u32.to_le_bytes()); // rank=2
        header[12..16].copy_from_slice(&shape[0].to_le_bytes());
        header[16..20].copy_from_slice(&shape[1].to_le_bytes());
        header[44..48].copy_from_slice(&32u32.to_le_bytes()); // bits=32
        header[48..52].copy_from_slice(&0u32.to_le_bytes()); // type=Float

        let raw: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let mut file_data = header;
        file_data.extend(&raw);
        std::fs::write(&dat_path, &file_data).unwrap();

        let (read_shape, read_data) = read_tensor_dat(&dat_path).unwrap();
        assert_eq!(read_shape, vec![2, 3]);
        assert_eq!(read_data, data);

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
