//! The part of the ONNX protobuf schema the importer reads, decoded straight
//! from the wire format. Field numbers are those of `onnx.proto`.
//!
//! Every length is checked against the bytes that hold it, so a malformed
//! file is an error, never a panic or an allocation larger than the input.

use std::{
    collections::{HashMap, HashSet, VecDeque},
    io::{Read, Seek},
    path::{Component, Path},
};

/// A graph as the importer needs it.
pub(super) struct Graph {
    pub nodes: Vec<Node>,
    pub initializers: Vec<Tensor>,
    /// Graph inputs that are not initializers, in file order.
    pub inputs: Vec<ValueInfo>,
    pub outputs: Vec<ValueInfo>,
}

/// A named value and its static shape, when the file declares one. A
/// dimension given only by name counts as zero.
pub(super) struct ValueInfo {
    pub name: String,
    pub shape: Option<Vec<usize>>,
}

pub(super) struct Node {
    pub op_type: String,
    pub name: String,
    /// An empty name marks an omitted optional input.
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attrs: Attributes,
}

/// A tensor's values widened to `f32`.
pub(super) struct Tensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

#[derive(Default)]
pub(super) struct Attributes(Vec<Attribute>);

struct Attribute {
    name: String,
    kind: u64,
    f: Option<f32>,
    i: Option<i64>,
    s: Option<String>,
    ints: Vec<i64>,
    t: Option<Tensor>,
}

/// `AttributeProto.AttributeType` values the importer reads.
const FLOAT: u64 = 1;
const INT: u64 = 2;
const STRING: u64 = 3;
const TENSOR: u64 = 4;
const INTS: u64 = 7;

impl Attributes {
    /// The last attribute called `name` of `kind`, as protobuf's last value wins.
    fn get(&self, name: &str, kind: u64) -> Option<&Attribute> {
        self.0
            .iter()
            .rev()
            .find(|a| a.name == name && a.kind == kind)
    }

    pub fn f(&self, name: &str, default: f32) -> f32 {
        self.get(name, FLOAT)
            .map_or(default, |a| a.f.unwrap_or(0.0))
    }

    pub fn i(&self, name: &str, default: i64) -> i64 {
        self.get(name, INT).map_or(default, |a| a.i.unwrap_or(0))
    }

    pub fn ints(&self, name: &str) -> &[i64] {
        self.get(name, INTS).map_or(&[], |a| &a.ints)
    }

    pub fn tensor(&self, name: &str) -> Option<&Tensor> {
        self.get(name, TENSOR).and_then(|a| a.t.as_ref())
    }
}

impl Graph {
    /// Node indices in an order that computes every input before its use,
    /// given the names already available. Nodes left on a cycle, or reading
    /// a name nothing produces, follow in file order.
    pub fn topological_sort(&self, known: &[String]) -> Vec<usize> {
        let known: HashSet<&str> = known.iter().map(String::as_str).collect();
        let mut producer = HashMap::new();
        for (index, node) in self.nodes.iter().enumerate() {
            for output in &node.outputs {
                producer.insert(output.as_str(), index);
            }
        }
        let mut waiting = vec![0usize; self.nodes.len()];
        let mut consumers = vec![Vec::new(); self.nodes.len()];
        for (index, node) in self.nodes.iter().enumerate() {
            for input in &node.inputs {
                if input.is_empty() || known.contains(input.as_str()) {
                    continue;
                }
                if let Some(&source) = producer.get(input.as_str()) {
                    waiting[index] += 1;
                    consumers[source].push(index);
                }
            }
        }
        let mut ready: VecDeque<usize> = (0..self.nodes.len())
            .filter(|&index| waiting[index] == 0)
            .collect();
        let mut order = Vec::with_capacity(self.nodes.len());
        let mut placed = vec![false; self.nodes.len()];
        while let Some(index) = ready.pop_front() {
            order.push(index);
            placed[index] = true;
            for &consumer in &consumers[index] {
                waiting[consumer] -= 1;
                if waiting[consumer] == 0 {
                    ready.push_back(consumer);
                }
            }
        }
        order.extend((0..self.nodes.len()).filter(|&index| !placed[index]));
        order
    }
}

/// Decode a `ModelProto`. `dir` holds the model's external data files; a
/// model read from memory has none.
pub(super) fn parse_model(bytes: &[u8], dir: Option<&Path>) -> Result<Graph, String> {
    let mut graph = None;
    for field in Fields(bytes) {
        let (number, value) = field?;
        if number == 7 {
            graph = Some(parse_graph(value.bytes(number)?, dir)?);
        }
    }
    graph.ok_or_else(|| "the model has no graph".to_owned())
}

fn parse_graph(bytes: &[u8], dir: Option<&Path>) -> Result<Graph, String> {
    let mut nodes = Vec::new();
    let mut initializers = Vec::new();
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    for field in Fields(bytes) {
        let (number, value) = field?;
        match number {
            1 => nodes.push(parse_node(value.bytes(number)?, dir)?),
            5 => initializers.push(parse_tensor(value.bytes(number)?, dir)?),
            11 => inputs.push(parse_value_info(value.bytes(number)?)?),
            12 => outputs.push(parse_value_info(value.bytes(number)?)?),
            _ => {}
        }
    }
    let mut names = HashSet::new();
    if let Some(tensor) = initializers.iter().find(|t| !names.insert(t.name.as_str())) {
        return Err(format!("initializer '{}' is defined twice", tensor.name));
    }
    // Before IR version 4, every initializer was also listed as an input.
    inputs.retain(|input: &ValueInfo| !names.contains(input.name.as_str()));
    Ok(Graph {
        nodes,
        initializers,
        inputs,
        outputs,
    })
}

fn parse_node(bytes: &[u8], dir: Option<&Path>) -> Result<Node, String> {
    let mut node = Node {
        op_type: String::new(),
        name: String::new(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        attrs: Attributes::default(),
    };
    for field in Fields(bytes) {
        let (number, value) = field?;
        match number {
            1 => node.inputs.push(value.string(number)?),
            2 => node.outputs.push(value.string(number)?),
            3 => node.name = value.string(number)?,
            4 => node.op_type = value.string(number)?,
            5 => node
                .attrs
                .0
                .push(parse_attribute(value.bytes(number)?, dir)?),
            _ => {}
        }
    }
    Ok(node)
}

fn parse_attribute(bytes: &[u8], dir: Option<&Path>) -> Result<Attribute, String> {
    let mut a = Attribute {
        name: String::new(),
        kind: 0,
        f: None,
        i: None,
        s: None,
        ints: Vec::new(),
        t: None,
    };
    for field in Fields(bytes) {
        let (number, value) = field?;
        match number {
            1 => a.name = value.string(number)?,
            2 => a.f = Some(f32::from_bits(value.fixed32(number)?)),
            3 => a.i = Some(value.varint(number)? as i64),
            4 => a.s = Some(value.string(number)?),
            5 => a.t = Some(parse_tensor(value.bytes(number)?, dir)?),
            8 => a.ints.extend(value.varints(number)?.map(|v| v as i64)),
            20 => a.kind = value.varint(number)?,
            _ => {}
        }
    }
    // Early exporters left the type unset; infer it from the value given.
    if a.kind == 0 {
        a.kind = if a.t.is_some() {
            TENSOR
        } else if !a.ints.is_empty() {
            INTS
        } else if a.s.is_some() {
            STRING
        } else if a.i.is_some() {
            INT
        } else if a.f.is_some() {
            FLOAT
        } else {
            0
        };
    }
    Ok(a)
}

fn parse_value_info(bytes: &[u8]) -> Result<ValueInfo, String> {
    let mut info = ValueInfo {
        name: String::new(),
        shape: None,
    };
    for field in Fields(bytes) {
        let (number, value) = field?;
        match number {
            1 => info.name = value.string(number)?,
            // TypeProto.tensor_type, then TypeProto.Tensor.shape.
            2 => {
                for field in Fields(value.bytes(number)?) {
                    let (number, value) = field?;
                    if number == 1 {
                        for field in Fields(value.bytes(number)?) {
                            let (number, value) = field?;
                            if number == 2 {
                                info.shape = Some(parse_shape(value.bytes(number)?)?);
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    Ok(info)
}

/// A `TensorShapeProto`: each `dim` holds a `dim_value` or a `dim_param`.
fn parse_shape(bytes: &[u8]) -> Result<Vec<usize>, String> {
    let mut dims = Vec::new();
    for field in Fields(bytes) {
        let (number, value) = field?;
        if number == 1 {
            let mut dim = 0;
            for field in Fields(value.bytes(number)?) {
                let (number, value) = field?;
                if number == 1 {
                    dim = usize::try_from(value.varint(number)? as i64).unwrap_or(0);
                }
            }
            dims.push(dim);
        }
    }
    Ok(dims)
}

/// `TensorProto.DataType` values with a widening to `f32`, and their
/// width in `raw_data`.
fn element_width(data_type: i32) -> Option<usize> {
    Some(match data_type {
        2 | 3 | 9 => 1,       // UINT8, INT8, BOOL
        4 | 5 | 10 | 16 => 2, // UINT16, INT16, FLOAT16, BFLOAT16
        1 | 6 | 12 => 4,      // FLOAT, INT32, UINT32
        7 | 11 | 13 => 8,     // INT64, DOUBLE, UINT64
        _ => return None,
    })
}

fn parse_tensor(bytes: &[u8], dir: Option<&Path>) -> Result<Tensor, String> {
    let mut name = String::new();
    let mut dims = Vec::new();
    let mut data_type = 0;
    let mut raw: &[u8] = &[];
    let mut floats = Vec::new();
    let mut int32s = Vec::new();
    let mut int64s = Vec::new();
    let mut doubles = Vec::new();
    let mut uint64s = Vec::new();
    let mut external = Vec::new();
    let mut external_location = false;
    for field in Fields(bytes) {
        let (number, value) = field?;
        match number {
            1 => dims.extend(value.varints(number)?.map(|v| v as i64)),
            2 => data_type = value.varint(number)? as i32,
            4 => floats.extend(value.fixed32s(number)?.map(f32::from_bits)),
            5 => int32s.extend(value.varints(number)?.map(|v| v as i32)),
            7 => int64s.extend(value.varints(number)?.map(|v| v as i64)),
            8 => name = value.string(number)?,
            9 => raw = value.bytes(number)?,
            10 => doubles.extend(value.fixed64s(number)?.map(f64::from_bits)),
            11 => uint64s.extend(value.varints(number)?),
            13 => external.push(parse_entry(value.bytes(number)?)?),
            14 => external_location = value.varint(number)? == 1,
            _ => {}
        }
    }

    let mut shape = Vec::with_capacity(dims.len());
    let mut count = 1usize;
    for &dim in &dims {
        let dim = usize::try_from(dim)
            .map_err(|_| format!("tensor '{name}' has a negative dimension {dim}"))?;
        count = count
            .checked_mul(dim)
            .ok_or_else(|| format!("tensor '{name}' has more elements than fit in memory"))?;
        shape.push(dim);
    }

    let data = if external_location || !external.is_empty() {
        raw_values(&external_bytes(&name, &external, dir)?, data_type, &name)?
    } else if !raw.is_empty() {
        raw_values(raw, data_type, &name)?
    } else {
        // Each type has one typed field; files without a type use whichever holds data.
        let int32 =
            |data_type| -> Vec<f32> { int32s.iter().map(|&v| int32_value(v, data_type)).collect() };
        let wide = |values: &[i64]| -> Vec<f32> { values.iter().map(|&v| v as f32).collect() };
        let unsigned = || -> Vec<f32> { uint64s.iter().map(|&v| v as f32).collect() };
        let double = || -> Vec<f32> { doubles.iter().map(|&v| v as f32).collect() };
        match data_type {
            1 => floats,
            7 => wide(&int64s),
            11 => double(),
            12 | 13 => unsigned(),
            2..=6 | 9 | 10 | 16 => int32(data_type),
            0 if !floats.is_empty() => floats,
            0 if !int64s.is_empty() => wide(&int64s),
            0 if !int32s.is_empty() => int32(6),
            0 if !doubles.is_empty() => double(),
            0 => unsigned(),
            _ => return Err(format!("tensor '{name}' has unsupported type {data_type}")),
        }
    };
    if data.len() != count {
        return Err(format!(
            "tensor '{name}' holds {} values for shape {shape:?}",
            data.len()
        ));
    }
    Ok(Tensor { name, shape, data })
}

/// One `int32_data` entry, which also carries the narrower integers and
/// the bits of 16-bit floats.
fn int32_value(v: i32, data_type: i32) -> f32 {
    match data_type {
        2 => f32::from(v as u8),
        3 => f32::from(v as i8),
        4 => f32::from(v as u16),
        5 => f32::from(v as i16),
        9 => f32::from(u8::from(v != 0)),
        10 => half::f16::from_bits(v as u16).to_f32(),
        16 => half::bf16::from_bits(v as u16).to_f32(),
        _ => v as f32,
    }
}

/// Little-endian `raw_data` of `data_type`, widened to `f32`.
fn raw_values(bytes: &[u8], data_type: i32, name: &str) -> Result<Vec<f32>, String> {
    let width = element_width(data_type)
        .ok_or_else(|| format!("tensor '{name}' has unsupported type {data_type}"))?;
    if !bytes.len().is_multiple_of(width) {
        return Err(format!(
            "tensor '{name}' has {} raw bytes, not a whole number of {width}-byte values",
            bytes.len()
        ));
    }
    let word = |b: &[u8]| {
        let mut w = [0u8; 8];
        w[..b.len()].copy_from_slice(b);
        u64::from_le_bytes(w)
    };
    Ok(bytes
        .chunks_exact(width)
        .map(|b| {
            let w = word(b);
            match data_type {
                1 => f32::from_bits(w as u32),
                2 => f32::from(w as u8),
                3 => f32::from(w as u8 as i8),
                4 => f32::from(w as u16),
                5 => f32::from(w as u16 as i16),
                6 => w as u32 as i32 as f32,
                7 => w as i64 as f32,
                9 => f32::from(u8::from(w != 0)),
                10 => half::f16::from_bits(w as u16).to_f32(),
                11 => f64::from_bits(w) as f32,
                12 => w as u32 as f32,
                13 => w as f32,
                _ => half::bf16::from_bits(w as u16).to_f32(),
            }
        })
        .collect())
}

/// A `StringStringEntryProto`.
fn parse_entry(bytes: &[u8]) -> Result<(String, String), String> {
    let mut entry = (String::new(), String::new());
    for field in Fields(bytes) {
        let (number, value) = field?;
        match number {
            1 => entry.0 = value.string(number)?,
            2 => entry.1 = value.string(number)?,
            _ => {}
        }
    }
    Ok(entry)
}

/// The bytes an external tensor names: `location` inside the model's
/// directory, from `offset`, for `length` bytes or to the end of the file.
fn external_bytes(
    name: &str,
    entries: &[(String, String)],
    dir: Option<&Path>,
) -> Result<Vec<u8>, String> {
    let dir = dir.ok_or_else(|| {
        format!("tensor '{name}' keeps its data in a separate file; load the model from its path")
    })?;
    let entry = |key: &str| {
        entries
            .iter()
            .rev()
            .find(|entry| entry.0 == key)
            .map(|entry| entry.1.as_str())
    };
    let number = |key: &str| {
        entry(key)
            .map(|v| v.parse::<u64>())
            .transpose()
            .map_err(|e| format!("tensor '{name}': external data {key}: {e}"))
    };
    let location = entry("location")
        .filter(|location| !location.is_empty())
        .ok_or_else(|| format!("tensor '{name}' has no external data location"))?;
    // The location comes from the file: keep it inside the model's directory,
    // checking the text before touching the filesystem and the resolved path
    // after following links.
    let outside = || {
        format!(
            "tensor '{name}': external data location '{location}' is outside the model's directory"
        )
    };
    if location.starts_with(['/', '\\'])
        || location.split(['/', '\\']).any(|part| part == "..")
        || Path::new(location)
            .components()
            .any(|c| !matches!(c, Component::Normal(_) | Component::CurDir))
    {
        return Err(outside());
    }
    let base = if dir.as_os_str().is_empty() {
        Path::new(".")
    } else {
        dir
    };
    let base = base
        .canonicalize()
        .map_err(|e| format!("{}: {e}", base.display()))?;
    let path = base.join(location);
    let path = path
        .canonicalize()
        .map_err(|e| format!("tensor '{name}': {}: {e}", path.display()))?;
    if !path.starts_with(&base) {
        return Err(outside());
    }

    let mut file = std::fs::File::open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let size = file
        .metadata()
        .map_err(|e| format!("{}: {e}", path.display()))?
        .len();
    let offset = number("offset")?.unwrap_or(0);
    let end = match number("length")? {
        Some(length) => offset.checked_add(length),
        None => Some(size),
    };
    let len = end
        .filter(|&end| offset <= end && end <= size)
        .and_then(|end| usize::try_from(end - offset).ok())
        .ok_or_else(|| {
            format!("tensor '{name}': external data range is outside the {size}-byte file")
        })?;
    let mut bytes = vec![0; len];
    file.seek(std::io::SeekFrom::Start(offset))
        .and_then(|_| file.read_exact(&mut bytes))
        .map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(bytes)
}

/// One field's payload, by wire type.
#[derive(Clone, Copy)]
enum Value<'a> {
    Varint(u64),
    Fixed64(u64),
    Bytes(&'a [u8]),
    Fixed32(u32),
}

impl<'a> Value<'a> {
    fn mismatch(number: u32, expected: &str) -> String {
        format!("field {number} is not {expected}")
    }

    fn varint(self, number: u32) -> Result<u64, String> {
        match self {
            Value::Varint(v) => Ok(v),
            Value::Fixed64(_) | Value::Bytes(_) | Value::Fixed32(_) => {
                Err(Self::mismatch(number, "a varint"))
            }
        }
    }

    fn fixed32(self, number: u32) -> Result<u32, String> {
        match self {
            Value::Fixed32(v) => Ok(v),
            Value::Varint(_) | Value::Fixed64(_) | Value::Bytes(_) => {
                Err(Self::mismatch(number, "a 32-bit value"))
            }
        }
    }

    fn bytes(self, number: u32) -> Result<&'a [u8], String> {
        match self {
            Value::Bytes(b) => Ok(b),
            Value::Varint(_) | Value::Fixed64(_) | Value::Fixed32(_) => {
                Err(Self::mismatch(number, "length-delimited"))
            }
        }
    }

    fn string(self, number: u32) -> Result<String, String> {
        Ok(String::from_utf8_lossy(self.bytes(number)?).into_owned())
    }

    /// A repeated varint field, either one element or a packed run.
    fn varints(self, number: u32) -> Result<impl Iterator<Item = u64> + 'a, String> {
        let (single, packed) = match self {
            Value::Varint(v) => (Some(v), &[][..]),
            Value::Bytes(b) => (None, b),
            Value::Fixed64(_) | Value::Fixed32(_) => {
                return Err(Self::mismatch(number, "a varint"));
            }
        };
        let mut pos = 0;
        let mut run = Vec::new();
        while pos < packed.len() {
            let (v, next) = varint(packed, pos)?;
            run.push(v);
            pos = next;
        }
        Ok(single.into_iter().chain(run))
    }

    /// A repeated fixed-width field, either one element or a packed run.
    fn fixed<const N: usize>(self, number: u32) -> Result<impl Iterator<Item = u64> + 'a, String> {
        let (single, packed) = match (N, self) {
            (4, Value::Fixed32(v)) => (Some(u64::from(v)), &[][..]),
            (8, Value::Fixed64(v)) => (Some(v), &[][..]),
            (_, Value::Bytes(b)) if b.len().is_multiple_of(N) => (None, b),
            (_, Value::Varint(_) | Value::Fixed64(_) | Value::Bytes(_) | Value::Fixed32(_)) => {
                return Err(Self::mismatch(number, &format!("{N}-byte values")));
            }
        };
        let run = packed.as_chunks::<N>().0.iter().map(|b| {
            let mut w = [0u8; 8];
            w[..N].copy_from_slice(b);
            u64::from_le_bytes(w)
        });
        Ok(single.into_iter().chain(run))
    }

    fn fixed32s(self, number: u32) -> Result<impl Iterator<Item = u32> + 'a, String> {
        Ok(self.fixed::<4>(number)?.map(|v| v as u32))
    }

    fn fixed64s(self, number: u32) -> Result<impl Iterator<Item = u64> + 'a, String> {
        self.fixed::<8>(number)
    }
}

/// The fields of one message, in wire order.
struct Fields<'a>(&'a [u8]);

impl<'a> Iterator for Fields<'a> {
    type Item = Result<(u32, Value<'a>), String>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.0.is_empty() {
            return None;
        }
        let field = read_field(self.0);
        // Stop after an error; otherwise continue past this field.
        self.0 = match field {
            Ok((_, _, rest)) => rest,
            Err(_) => &[],
        };
        Some(field.map(|(number, value, _)| (number, value)))
    }
}

/// The field at the start of `buf`, and what follows it.
fn read_field(buf: &[u8]) -> Result<(u32, Value<'_>, &[u8]), String> {
    let (key, pos) = varint(buf, 0)?;
    let number =
        u32::try_from(key >> 3).map_err(|_| format!("field number {} is too large", key >> 3))?;
    let rest = &buf[pos..];
    let fixed = |n: usize| {
        rest.get(..n)
            .map(|b| {
                let mut w = [0u8; 8];
                w[..n].copy_from_slice(b);
                (u64::from_le_bytes(w), &rest[n..])
            })
            .ok_or_else(|| format!("field {number} runs past the end of its message"))
    };
    match key & 7 {
        0 => {
            let (v, end) = varint(rest, 0)?;
            Ok((number, Value::Varint(v), &rest[end..]))
        }
        1 => fixed(8).map(|(v, rest)| (number, Value::Fixed64(v), rest)),
        2 => {
            let (len, start) = varint(rest, 0)?;
            let payload = usize::try_from(len)
                .ok()
                .and_then(|len| rest.get(start..start.checked_add(len)?))
                .ok_or_else(|| format!("field {number} runs past the end of its message"))?;
            Ok((
                number,
                Value::Bytes(payload),
                &rest[start + payload.len()..],
            ))
        }
        5 => fixed(4).map(|(v, rest)| (number, Value::Fixed32(v as u32), rest)),
        wire => Err(format!("field {number} has unsupported wire type {wire}")),
    }
}

/// The varint at `pos`, and the position after it.
fn varint(buf: &[u8], mut pos: usize) -> Result<(u64, usize), String> {
    let mut value = 0u64;
    for shift in (0..64).step_by(7) {
        let &byte = buf
            .get(pos)
            .ok_or_else(|| "a varint runs past the end of its message".to_owned())?;
        pos += 1;
        value |= u64::from(byte & 0x7F) << shift;
        if byte & 0x80 == 0 {
            return Ok((value, pos));
        }
    }
    Err("a varint is longer than ten bytes".to_owned())
}

#[cfg(test)]
mod tests {
    use super::{
        Attributes, Fields, Graph, Node, parse_model, parse_shape, parse_tensor, parse_value_info,
    };

    fn varint(mut v: u64) -> Vec<u8> {
        let mut out = Vec::new();
        while v >= 0x80 {
            out.push(v as u8 | 0x80);
            v >>= 7;
        }
        out.push(v as u8);
        out
    }

    fn key(number: u32, wire: u64) -> Vec<u8> {
        varint(u64::from(number) << 3 | wire)
    }

    fn int(number: u32, v: u64) -> Vec<u8> {
        [key(number, 0), varint(v)].concat()
    }

    fn bytes(number: u32, payload: &[u8]) -> Vec<u8> {
        [
            key(number, 2),
            varint(payload.len() as u64),
            payload.to_vec(),
        ]
        .concat()
    }

    fn packed(values: &[u64]) -> Vec<u8> {
        values.iter().flat_map(|&v| varint(v)).collect()
    }

    fn le<T: Copy, const N: usize>(values: &[T], to: fn(T) -> [u8; N]) -> Vec<u8> {
        values.iter().flat_map(|&v| to(v)).collect()
    }

    #[test]
    fn malformed_lengths_are_errors() {
        // Past the end of the buffer, and longer than any buffer.
        let saturated = [&[0x0A][..], &[0xFF; 9], &[0x01, 0x00]].concat();
        for buf in [&[0x0A, 0x05, 1, 2][..], &saturated, &[0x80; 11]] {
            assert!(Fields(buf).any(|field| field.is_err()), "{buf:?}");
        }
        assert!(parse_value_info(&[0x12, 0x7E, 0x02, 0x03]).is_err());
        assert!(parse_shape(&[0x12, 0xC8, 0x01, 0x08, 0x01]).is_err());

        // Any truncation of a model is an error or a smaller model, never a panic.
        let tensor = [bytes(1, &packed(&[2])), int(2, 1), bytes(9, &[0; 8])].concat();
        let graph = [bytes(5, &tensor), bytes(1, &bytes(4, b"Relu"))].concat();
        let model = [int(1, 8), bytes(7, &graph)].concat();
        assert!(parse_model(&model, None).is_ok());
        for end in 0..model.len() {
            let _ = parse_model(&model[..end], None);
        }
    }

    #[test]
    fn repeated_fields_read_packed_and_unpacked() {
        let values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let packed_tensor = [
            bytes(1, &packed(&[2, 3])),
            int(2, 1),
            bytes(4, &le(&values, f32::to_le_bytes)),
        ]
        .concat();
        let mut unpacked = [int(1, 2), int(1, 3), int(2, 1)].concat();
        for v in values {
            unpacked.extend([key(4, 5), v.to_le_bytes().to_vec()].concat());
        }
        for tensor in [packed_tensor, unpacked] {
            let tensor = parse_tensor(&tensor, None).unwrap();
            assert_eq!(tensor.shape, [2, 3]);
            assert_eq!(tensor.data, values);
        }
    }

    #[test]
    fn tensor_types_widen_to_f32() {
        let raw = |data_type: u64, count: u64, payload: &[u8]| {
            [
                bytes(1, &packed(&[count])),
                int(2, data_type),
                bytes(9, payload),
            ]
            .concat()
        };
        let decode = |tensor: &[u8]| parse_tensor(tensor, None).map(|t| t.data);
        let halves = [half::f16::from_f32(1.5), half::f16::from_f32(-2.0)];
        assert_eq!(
            decode(&raw(10, 2, &le(&halves, half::f16::to_le_bytes))),
            Ok(vec![1.5, -2.0])
        );
        assert_eq!(
            decode(&raw(11, 2, &le(&[0.25f64, -8.0], f64::to_le_bytes))),
            Ok(vec![0.25, -8.0])
        );
        assert_eq!(
            decode(&raw(7, 2, &le(&[-3i64, 5], i64::to_le_bytes))),
            Ok(vec![-3.0, 5.0])
        );
        assert_eq!(decode(&raw(9, 2, &[0, 7])), Ok(vec![0.0, 1.0]));
        assert_eq!(decode(&raw(3, 1, &[0xFE])), Ok(vec![-2.0]));

        // Typed fields: 16-bit float bits in int32_data, negative int64_data.
        let bits = [3.0f32, -0.5].map(|v| u64::from(half::bf16::from_f32(v).to_bits()));
        let bf16 = [
            bytes(1, &packed(&[2])),
            int(2, 16),
            bytes(5, &packed(&bits)),
        ]
        .concat();
        assert_eq!(decode(&bf16), Ok(vec![3.0, -0.5]));
        let negative = [
            bytes(1, &packed(&[1])),
            int(2, 7),
            bytes(7, &packed(&[-4i64 as u64])),
        ]
        .concat();
        assert_eq!(decode(&negative), Ok(vec![-4.0]));

        // A ragged payload, a count that disagrees with the shape, a negative
        // dimension and a string tensor are errors.
        assert!(decode(&raw(1, 2, &[0; 6])).is_err());
        assert!(decode(&raw(1, 2, &[0; 4])).is_err());
        assert!(decode(&raw(1, -1i64 as u64, &[0; 4])).is_err());
        assert!(decode(&raw(8, 1, b"ab")).is_err());
    }

    #[test]
    fn external_data_stays_in_the_model_directory() {
        let root = std::env::temp_dir().join(format!("meganeura-onnx-{}", std::process::id()));
        let dir = root.join("model");
        std::fs::create_dir_all(&dir).unwrap();
        let values = [1.0f32, 2.0, 3.0];
        let file = [vec![0; 8], le(&values, f32::to_le_bytes)].concat();
        std::fs::write(dir.join("weights.bin"), &file).unwrap();
        std::fs::write(root.join("outside.bin"), &file).unwrap();
        let entry = |k: &str, v: &str| {
            bytes(
                13,
                &[bytes(1, k.as_bytes()), bytes(2, v.as_bytes())].concat(),
            )
        };
        let tensor = |location: &str, count: u64| {
            [
                bytes(1, &packed(&[count])),
                int(2, 1),
                entry("location", location),
                entry("offset", "8"),
                entry("length", &(4 * count).to_string()),
                int(14, 1),
            ]
            .concat()
        };
        let decode = |tensor: &[u8], dir| parse_tensor(tensor, dir).map(|t| t.data);
        assert_eq!(
            decode(&tensor("weights.bin", 3), Some(&dir)),
            Ok(values.to_vec())
        );
        // Outside the directory, past the end of the file, or with no directory.
        for location in [
            "../outside.bin",
            "/etc/hostname",
            "sub/../../outside.bin",
            "",
        ] {
            assert!(
                decode(&tensor(location, 3), Some(&dir)).is_err(),
                "{location}"
            );
        }
        assert!(decode(&tensor("weights.bin", 4), Some(&dir)).is_err());
        assert!(decode(&tensor("weights.bin", 3), None).is_err());
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(root.join("outside.bin"), dir.join("link.bin")).unwrap();
            assert!(decode(&tensor("link.bin", 3), Some(&dir)).is_err());
        }
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn nodes_run_after_their_inputs() {
        let node = |inputs: &[&str], output: &str| Node {
            op_type: "Relu".to_owned(),
            name: output.to_owned(),
            inputs: inputs.iter().map(|&name| name.to_owned()).collect(),
            outputs: vec![output.to_owned()],
            attrs: Attributes::default(),
        };
        let graph = |nodes| Graph {
            nodes,
            initializers: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        };
        let known = ["a".to_owned()];
        // An empty name is an omitted optional input.
        let chain = graph(vec![
            node(&["b"], "c"),
            node(&["a"], "b"),
            node(&["c", ""], "d"),
        ]);
        assert_eq!(chain.topological_sort(&known), [1, 0, 2]);
        // A cycle follows the nodes that can run, in file order.
        let cycle = graph(vec![
            node(&["y"], "x"),
            node(&["x"], "y"),
            node(&["a"], "z"),
        ]);
        assert_eq!(cycle.topological_sort(&known), [2, 0, 1]);
    }
}
