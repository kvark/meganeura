//! GGUF weight import: read GGML's container format into Meganeura's
//! parameter buffers.
//!
//! GGUF is a weight-and-metadata container, not a graph format. There is no
//! computation graph to import — a GGUF file carries named tensors plus
//! key/value metadata describing the architecture (`llama.block_count`,
//! `llama.attention.head_count`, …). So this module pairs with a model
//! builder in [`crate::models`] the way SafeTensors loading does, rather
//! than producing a [`crate::Graph`] the way [`super::onnx`] does.
//!
//! # Layouts
//!
//! GGUF names dimensions fastest-varying first: a 2-D tensor with
//! `dimensions = [ne0, ne1]` stores element `(i0, i1)` at `i1 * ne0 + i0`.
//! For a projection weight that means `ne0` is the reduction extent K and
//! `ne1` the output extent N, stored as `[N, K]` row-major.
//!
//! Meganeura declares the same weight as `[K, N]` row-major, so the two
//! *unquantized* layouts are transposes of each other and [`GgufTensor::to_f32`]
//! transposes on the way out.
//!
//! The *packed* layouts, on the other hand, already agree. Meganeura's
//! quantizers walk columns of `[K, N]`, emitting block `n * (K/32) + k/32`;
//! GGUF blocks run along `ne0 = K` within each row `n`, giving the same index.
//! Packing already performs the transpose, so [`GgufTensor::to_packed`] never
//! has to. What differs is the arrangement *within* a block, which is what
//! the repack functions below fix up.
//!
//! # What is resolved here
//!
//! Differences between GGML's block encodings and Meganeura's are resolved at
//! load time, so no shader knows GGUF exists:
//!
//! | GGML type | Handling |
//! |-----------|----------|
//! | `F32`, `F16` | transpose, no value change |
//! | `Q4_0` | repack to Meganeura Q4, `m = -8d` |
//! | `Q4_1` | repack to Meganeura Q4 |
//! | `Q8_0` | repack to Meganeura Q8 |
//! | `Q4_K`, `Q6_K` | none — stored in GGML's own layout |
//!
//! Every type is lossless. The K-quants need no repack at all: Meganeura
//! stores them byte-for-byte as GGML does, so [`GgufTensor::to_packed`] hands
//! back the file's bytes and the shaders read them directly. (Q6_K's 210-byte
//! superblocks are not a whole number of words, so its buffer gets a zero-
//! padded tail; the superblocks themselves are untouched.)
//!
//! [`GgufTensor::to_f32`] remains available for every type, and is what the
//! reference dequantizers below implement — but going through it for a
//! quantized weight requantizes on the way back in, roughly doubling the
//! quantization error, so prefer `to_packed`.

use std::collections::HashMap;
use std::path::Path;

use crate::graph::DType;

/// Errors that can occur during GGUF import.
#[derive(Debug)]
pub enum GgufError {
    /// The file is not GGUF, or its header is malformed.
    BadHeader(String),
    /// The file ended before a structure was fully read.
    Truncated {
        /// What was being read when the data ran out.
        what: &'static str,
        /// Byte offset at which the read was attempted.
        offset: usize,
    },
    /// A `ggml_type` or metadata value type outside the known set.
    UnknownType(u32),
    /// A tensor whose element count does not fill whole blocks, or whose
    /// shape cannot be expressed as a Meganeura parameter.
    BadShape(String),
    /// A type that has no lossless Meganeura packed equivalent.
    UnsupportedPack(GgmlType),
    /// Underlying I/O failure.
    Io(std::io::Error),
}

impl std::fmt::Display for GgufError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::BadHeader(ref e) => write!(f, "GGUF header error: {e}"),
            Self::Truncated { what, offset } => {
                write!(f, "GGUF truncated while reading {what} at byte {offset}")
            }
            Self::UnknownType(t) => write!(f, "unknown GGUF type tag {t}"),
            Self::BadShape(ref e) => write!(f, "GGUF shape error: {e}"),
            Self::UnsupportedPack(t) => write!(
                f,
                "{t:?} has no lossless Meganeura packed form; use to_f32() and \
                 Session::set_parameter, which requantizes"
            ),
            Self::Io(ref e) => write!(f, "GGUF I/O error: {e}"),
        }
    }
}

impl std::error::Error for GgufError {}

impl From<std::io::Error> for GgufError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// The `ggml_type` tags this loader understands.
///
/// Numeric values are GGML's and are part of the file format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q8_0 = 8,
    Q4K = 12,
    Q6K = 14,
}

impl GgmlType {
    fn from_tag(tag: u32) -> Result<Self, GgufError> {
        Ok(match tag {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            8 => Self::Q8_0,
            12 => Self::Q4K,
            14 => Self::Q6K,
            other => return Err(GgufError::UnknownType(other)),
        })
    }

    /// Elements per stored block. Non-block types report 1.
    pub fn block_elements(self) -> usize {
        match self {
            Self::F32 | Self::F16 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q8_0 => 32,
            Self::Q4K | Self::Q6K => 256,
        }
    }

    /// Bytes per stored block, straight from `ggml-common.h`.
    pub fn block_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            // f16 d + 16 packed nibble bytes
            Self::Q4_0 => 18,
            // f16 d + f16 m + 16 packed nibble bytes
            Self::Q4_1 => 20,
            // f16 d + 32 int8
            Self::Q8_0 => 34,
            // f16 d + f16 dmin + 12 scale bytes + 128 nibble bytes
            Self::Q4K => 144,
            // 128 low-nibble + 64 high-bit + 16 int8 scales + f16 d
            Self::Q6K => 210,
        }
    }

    /// Byte length of `count` elements stored in this type.
    fn stored_bytes(self, count: usize) -> Result<usize, GgufError> {
        let per = self.block_elements();
        if !count.is_multiple_of(per) {
            return Err(GgufError::BadShape(format!(
                "{count} elements is not a whole number of {per}-element {self:?} blocks"
            )));
        }
        Ok(count / per * self.block_bytes())
    }
}

/// A metadata value. Arrays keep their element values in declaration order.
#[derive(Clone, Debug, PartialEq)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
}

impl GgufValue {
    /// Read as an unsigned integer, whatever width it was stored at.
    ///
    /// Architecture metadata is not written at a consistent width across
    /// producers, so callers that want `llama.block_count` should not have to
    /// care whether it arrived as `U32` or `U64`.
    pub fn as_u64(&self) -> Option<u64> {
        Some(match *self {
            Self::U8(v) => u64::from(v),
            Self::U16(v) => u64::from(v),
            Self::U32(v) => u64::from(v),
            Self::U64(v) => v,
            Self::I8(v) if v >= 0 => v as u64,
            Self::I16(v) if v >= 0 => v as u64,
            Self::I32(v) if v >= 0 => v as u64,
            Self::I64(v) if v >= 0 => v as u64,
            _ => return None,
        })
    }

    /// Read as a float, widening integers where that is exact enough to be
    /// useful (RoPE bases and epsilons are written both ways).
    pub fn as_f64(&self) -> Option<f64> {
        Some(match *self {
            Self::F32(v) => f64::from(v),
            Self::F64(v) => v,
            _ => return self.as_u64().map(|v| v as f64),
        })
    }

    /// Borrow as a string, if it is one.
    pub fn as_str(&self) -> Option<&str> {
        match *self {
            Self::String(ref s) => Some(s),
            _ => None,
        }
    }

    /// Borrow as an array, if it is one.
    pub fn as_array(&self) -> Option<&[GgufValue]> {
        match *self {
            Self::Array(ref a) => Some(a),
            _ => None,
        }
    }
}

/// One tensor: its logical shape, its GGML type, and its bytes exactly as
/// they appear in the file.
#[derive(Clone, Debug)]
pub struct GgufTensor {
    /// Dimensions in GGUF order, fastest-varying first. For a 2-D
    /// projection weight this is `[K, N]`.
    pub dims: Vec<usize>,
    /// How the bytes are encoded.
    pub ggml_type: GgmlType,
    /// Raw file bytes for this tensor, unmodified.
    pub data: Vec<u8>,
}

impl GgufTensor {
    /// Total element count.
    pub fn num_elements(&self) -> usize {
        self.dims.iter().product()
    }

    /// `(K, N)` for a 2-D tensor; `(n, 1)` for a 1-D one.
    ///
    /// Tensors with more than two dimensions have no Meganeura parameter
    /// equivalent and are rejected.
    fn matrix_dims(&self) -> Result<(usize, usize), GgufError> {
        match self.dims.len() {
            1 => Ok((self.dims[0], 1)),
            2 => Ok((self.dims[0], self.dims[1])),
            n => Err(GgufError::BadShape(format!(
                "{n}-dimensional tensors have no Meganeura parameter form"
            ))),
        }
    }

    /// Dequantize to f32 in Meganeura's `[K, N]` row-major order, ready for
    /// `Session::set_parameter`.
    ///
    /// This transposes: GGUF stores `ne0` fastest, Meganeura stores it
    /// slowest. Every GGML type this loader knows is supported here.
    pub fn to_f32(&self) -> Result<Vec<f32>, GgufError> {
        let (k, n) = self.matrix_dims()?;
        // Values in GGUF order first: element (i0, i1) at i1 * ne0 + i0.
        let flat = self.dequantize_flat()?;
        if n == 1 {
            return Ok(flat);
        }
        let mut out = vec![0.0f32; k * n];
        for col in 0..n {
            for row in 0..k {
                out[row * n + col] = flat[col * k + row];
            }
        }
        Ok(out)
    }

    /// Repack into Meganeura's packed layout for `Session::set_parameter_packed`,
    /// returning the [`DType`] the parameter must be declared with.
    ///
    /// Lossless: the quantized values themselves are carried across unchanged
    /// and only their arrangement and scale encoding are adjusted. Rejects
    /// `Q6_K`, which has no Meganeura equivalent.
    ///
    /// `Q4_K` is the one type that needs no work at all — Meganeura stores it
    /// in GGML's own layout, so this hands back the file's bytes verbatim.
    pub fn to_packed(&self) -> Result<(DType, Vec<u8>), GgufError> {
        let (k, _n) = self.matrix_dims()?;
        match self.ggml_type {
            GgmlType::Q4_0 | GgmlType::Q4_1 => {
                require_block_aligned(k, 32, "Q4")?;
                Ok((DType::Q4_0, self.repack_q4()?))
            }
            GgmlType::Q8_0 => {
                require_block_aligned(k, 32, "Q8")?;
                Ok((DType::Q8_0, self.repack_q8()))
            }
            GgmlType::Q4K => {
                require_block_aligned(k, 256, "Q4_K")?;
                Ok((DType::Q4K, self.data.clone()))
            }
            GgmlType::Q6K => {
                require_block_aligned(k, 256, "Q6_K")?;
                // 210-byte superblocks are not a whole number of words, so
                // an odd count leaves the buffer two bytes short of the
                // `array<u32>` binding. Pad the tail; the superblocks
                // themselves stay byte-for-byte.
                let mut bytes = self.data.clone();
                bytes.resize(bytes.len().next_multiple_of(4), 0);
                Ok((DType::Q6K, bytes))
            }
            other => Err(GgufError::UnsupportedPack(other)),
        }
    }

    /// GGUF-order values, one f32 per element.
    fn dequantize_flat(&self) -> Result<Vec<f32>, GgufError> {
        let count = self.num_elements();
        let expect = self.ggml_type.stored_bytes(count)?;
        if self.data.len() != expect {
            return Err(GgufError::BadShape(format!(
                "{:?} tensor of {count} elements needs {expect} bytes, has {}",
                self.ggml_type,
                self.data.len()
            )));
        }
        Ok(match self.ggml_type {
            GgmlType::F32 => self
                .data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&c| f32::from_le_bytes(c))
                .collect(),
            GgmlType::F16 => self
                .data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&c| f16_from_bits(u16::from_le_bytes(c)))
                .collect(),
            GgmlType::Q4_0 => dequant_q4_0(&self.data, count),
            GgmlType::Q4_1 => dequant_q4_1(&self.data, count),
            GgmlType::Q8_0 => dequant_q8_0(&self.data, count),
            GgmlType::Q4K => dequant_q4_k(&self.data, count),
            GgmlType::Q6K => dequant_q6_k(&self.data, count),
        })
    }

    /// GGML Q4_0/Q4_1 → Meganeura Q4.
    ///
    /// Three things change and nothing else does:
    ///
    /// * Q4_0 is symmetric (`value = (q - 8) * d`); Meganeura's form is
    ///   always `q * d + m`, so `m = -8d`. Scaling by 8 only shifts an
    ///   exponent, so this survives the f16 round trip exactly.
    /// * GGML splits a block's nibbles across halves — `qs[j]` holds element
    ///   `j` low and element `j + 16` high — while Meganeura pairs adjacent
    ///   elements, `qs[e / 2]` holding `e` and `e + 1`.
    /// * GGML interleaves each block's header with its payload; Meganeura
    ///   keeps all `(d, m)` words first and all nibble words after.
    fn repack_q4(&self) -> Result<Vec<u8>, GgufError> {
        let count = self.num_elements();
        let blocks = count / 32;
        let stride = self.ggml_type.block_bytes();
        let symmetric = self.ggml_type == GgmlType::Q4_0;
        // Header region then payload region, mirroring `quantize_q4_0`.
        let mut out = vec![0u8; blocks * 4 + blocks * 16];
        let payload = blocks * 4;
        for b in 0..blocks {
            let src = b * stride;
            let d_bits = u16::from_le_bytes([self.data[src], self.data[src + 1]]);
            let (m_bits, qs) = if symmetric {
                let d = f16_from_bits(d_bits);
                (f16_to_bits(-8.0 * d), src + 2)
            } else {
                (
                    u16::from_le_bytes([self.data[src + 2], self.data[src + 3]]),
                    src + 4,
                )
            };
            let dm = u32::from(d_bits) | (u32::from(m_bits) << 16);
            out[b * 4..b * 4 + 4].copy_from_slice(&dm.to_le_bytes());

            let dst = payload + b * 16;
            for e in 0..32u32 {
                // Where GGML put element e.
                let byte = (e % 16) as usize;
                let nibble = if e < 16 {
                    self.data[qs + byte] & 0x0F
                } else {
                    self.data[qs + byte] >> 4
                };
                // Where Meganeura wants it.
                let slot = dst + (e / 2) as usize;
                if e % 2 == 0 {
                    out[slot] |= nibble;
                } else {
                    out[slot] |= nibble << 4;
                }
            }
        }
        Ok(out)
    }

    /// GGML Q8_0 → Meganeura Q8.
    ///
    /// Same block order, same element order, same `value = q * d`. The only
    /// difference is that Meganeura pads the f16 scale out to a full word so
    /// the quants start word-aligned, making each block 36 bytes to GGML's 34.
    fn repack_q8(&self) -> Vec<u8> {
        let blocks = self.num_elements() / 32;
        let mut out = vec![0u8; blocks * 36];
        for b in 0..blocks {
            let src = b * 34;
            let dst = b * 36;
            out[dst..dst + 2].copy_from_slice(&self.data[src..src + 2]);
            // dst + 2 .. dst + 4 stays zero padding.
            out[dst + 4..dst + 36].copy_from_slice(&self.data[src + 2..src + 34]);
        }
        out
    }
}

/// A parsed GGUF file.
#[derive(Clone, Debug)]
pub struct GgufModel {
    /// Metadata key/value pairs, keyed as they appear in the file
    /// (`general.architecture`, `llama.block_count`, …).
    pub metadata: HashMap<String, GgufValue>,
    /// Tensors by name.
    pub tensors: HashMap<String, GgufTensor>,
}

impl GgufModel {
    /// `general.architecture`, if present.
    pub fn architecture(&self) -> Option<&str> {
        self.metadata.get("general.architecture")?.as_str()
    }

    /// Look up a metadata key, trying the architecture-qualified form first.
    ///
    /// GGUF namespaces most hyperparameters under the architecture name, so a
    /// llama file stores `llama.block_count` rather than `block_count`. This
    /// resolves `block_count` against whichever architecture the file
    /// declares, so callers do not have to build the key themselves.
    pub fn arch_key(&self, suffix: &str) -> Option<&GgufValue> {
        if let Some(arch) = self.architecture() {
            if let Some(v) = self.metadata.get(&format!("{arch}.{suffix}")) {
                return Some(v);
            }
        }
        self.metadata.get(suffix)
    }
}

fn require_block_aligned(k: usize, block: usize, label: &str) -> Result<(), GgufError> {
    if k.is_multiple_of(block) {
        return Ok(());
    }
    Err(GgufError::BadShape(format!(
        "{label} needs the reduction extent to be a multiple of {block}, got {k}"
    )))
}

// ---------------------------------------------------------------------------
// Container parsing
// ---------------------------------------------------------------------------

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn take(&mut self, n: usize, what: &'static str) -> Result<&'a [u8], GgufError> {
        let end = self.pos.checked_add(n).ok_or(GgufError::Truncated {
            what,
            offset: self.pos,
        })?;
        if end > self.bytes.len() {
            return Err(GgufError::Truncated {
                what,
                offset: self.pos,
            });
        }
        let out = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn u8(&mut self) -> Result<u8, GgufError> {
        Ok(self.take(1, "u8")?[0])
    }

    fn u16(&mut self) -> Result<u16, GgufError> {
        let b = self.take(2, "u16")?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn u32(&mut self) -> Result<u32, GgufError> {
        let b = self.take(4, "u32")?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Result<u64, GgufError> {
        let b = self.take(8, "u64")?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    /// A length-prefixed, non-null-terminated UTF-8 string.
    fn string(&mut self) -> Result<String, GgufError> {
        let len = usize::try_from(self.u64()?).map_err(|_| GgufError::Truncated {
            what: "string length",
            offset: self.pos,
        })?;
        let bytes = self.take(len, "string")?;
        String::from_utf8(bytes.to_vec())
            .map_err(|e| GgufError::BadHeader(format!("non-UTF-8 string: {e}")))
    }

    fn value(&mut self, tag: u32) -> Result<GgufValue, GgufError> {
        Ok(match tag {
            0 => GgufValue::U8(self.u8()?),
            1 => GgufValue::I8(self.u8()? as i8),
            2 => GgufValue::U16(self.u16()?),
            3 => GgufValue::I16(self.u16()? as i16),
            4 => GgufValue::U32(self.u32()?),
            5 => GgufValue::I32(self.u32()? as i32),
            6 => GgufValue::F32(f32::from_bits(self.u32()?)),
            7 => GgufValue::Bool(self.u8()? != 0),
            8 => GgufValue::String(self.string()?),
            9 => {
                let elem = self.u32()?;
                let len = usize::try_from(self.u64()?).map_err(|_| GgufError::Truncated {
                    what: "array length",
                    offset: self.pos,
                })?;
                // An array of a bogus element type would otherwise be
                // discovered one element at a time.
                if !matches!(elem, 0..=8 | 10..=12) {
                    return Err(GgufError::UnknownType(elem));
                }
                let mut out = Vec::with_capacity(len.min(1024));
                for _ in 0..len {
                    out.push(self.value(elem)?);
                }
                GgufValue::Array(out)
            }
            10 => GgufValue::U64(self.u64()?),
            11 => GgufValue::I64(self.u64()? as i64),
            12 => GgufValue::F64(f64::from_bits(self.u64()?)),
            other => return Err(GgufError::UnknownType(other)),
        })
    }
}

/// Load a GGUF file from disk.
pub fn load_gguf(path: &Path) -> Result<GgufModel, GgufError> {
    let bytes = std::fs::read(path)?;
    load_gguf_bytes(&bytes)
}

/// Parse a GGUF file already in memory.
pub fn load_gguf_bytes(bytes: &[u8]) -> Result<GgufModel, GgufError> {
    let mut r = Reader::new(bytes);
    let magic = r.take(4, "magic")?;
    if magic != b"GGUF" {
        return Err(GgufError::BadHeader(format!(
            "expected magic \"GGUF\", got {magic:?}"
        )));
    }
    let version = r.u32()?;
    if !(2..=3).contains(&version) {
        return Err(GgufError::BadHeader(format!(
            "unsupported GGUF version {version}"
        )));
    }
    let tensor_count = usize::try_from(r.u64()?).map_err(|_| GgufError::Truncated {
        what: "tensor count",
        offset: 8,
    })?;
    let kv_count = usize::try_from(r.u64()?).map_err(|_| GgufError::Truncated {
        what: "metadata count",
        offset: 16,
    })?;

    let mut metadata = HashMap::with_capacity(kv_count);
    for _ in 0..kv_count {
        let key = r.string()?;
        let tag = r.u32()?;
        metadata.insert(key, r.value(tag)?);
    }

    // Tensor infos carry offsets into the data section, so collect them all
    // before resolving any of them.
    struct Info {
        name: String,
        dims: Vec<usize>,
        ggml_type: GgmlType,
        offset: usize,
    }
    let mut infos = Vec::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let name = r.string()?;
        let n_dims = r.u32()? as usize;
        if n_dims > 4 {
            return Err(GgufError::BadShape(format!(
                "tensor `{name}` claims {n_dims} dimensions, GGUF allows at most 4"
            )));
        }
        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dims.push(usize::try_from(r.u64()?).map_err(|_| {
                GgufError::BadShape(format!(
                    "tensor `{name}` has a dimension too large for this platform"
                ))
            })?);
        }
        let ggml_type = GgmlType::from_tag(r.u32()?)?;
        let offset = usize::try_from(r.u64()?).map_err(|_| {
            GgufError::BadShape(format!(
                "tensor `{name}` has an offset too large for this platform"
            ))
        })?;
        infos.push(Info {
            name,
            dims,
            ggml_type,
            offset,
        });
    }

    // Tensor data begins at the next alignment boundary after the infos, and
    // each offset is relative to that point rather than to the file start.
    let alignment = metadata
        .get("general.alignment")
        .and_then(GgufValue::as_u64)
        .unwrap_or(32) as usize;
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(GgufError::BadHeader(format!(
            "general.alignment must be a power of two, got {alignment}"
        )));
    }
    let data_start = r.pos.next_multiple_of(alignment);

    let mut tensors = HashMap::with_capacity(infos.len());
    for info in infos {
        let count: usize = info.dims.iter().product();
        let len = info.ggml_type.stored_bytes(count)?;
        let start = data_start
            .checked_add(info.offset)
            .ok_or(GgufError::Truncated {
                what: "tensor offset",
                offset: data_start,
            })?;
        let end = start.checked_add(len).ok_or(GgufError::Truncated {
            what: "tensor data",
            offset: start,
        })?;
        if end > bytes.len() {
            return Err(GgufError::Truncated {
                what: "tensor data",
                offset: start,
            });
        }
        tensors.insert(
            info.name,
            GgufTensor {
                dims: info.dims,
                ggml_type: info.ggml_type,
                data: bytes[start..end].to_vec(),
            },
        );
    }

    Ok(GgufModel { metadata, tensors })
}

// ---------------------------------------------------------------------------
// Reference dequantization
//
// These mirror `dequantize_row_*` in ggml-quants.c and exist so that every
// GGML type can reach f32 even when it has no packed Meganeura form. They are
// load-time code, not a hot path.
// ---------------------------------------------------------------------------

fn f16_from_bits(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

fn f16_to_bits(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}

fn dequant_q4_0(data: &[u8], count: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; count];
    for (b, chunk) in data.as_chunks::<18>().0.iter().enumerate() {
        let d = f16_from_bits(u16::from_le_bytes([chunk[0], chunk[1]]));
        for j in 0..16 {
            let byte = chunk[2 + j];
            out[b * 32 + j] = ((byte & 0x0F) as f32 - 8.0) * d;
            out[b * 32 + j + 16] = ((byte >> 4) as f32 - 8.0) * d;
        }
    }
    out
}

fn dequant_q4_1(data: &[u8], count: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; count];
    for (b, chunk) in data.as_chunks::<20>().0.iter().enumerate() {
        let d = f16_from_bits(u16::from_le_bytes([chunk[0], chunk[1]]));
        let m = f16_from_bits(u16::from_le_bytes([chunk[2], chunk[3]]));
        for j in 0..16 {
            let byte = chunk[4 + j];
            out[b * 32 + j] = (byte & 0x0F) as f32 * d + m;
            out[b * 32 + j + 16] = (byte >> 4) as f32 * d + m;
        }
    }
    out
}

fn dequant_q8_0(data: &[u8], count: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; count];
    for (b, chunk) in data.as_chunks::<34>().0.iter().enumerate() {
        let d = f16_from_bits(u16::from_le_bytes([chunk[0], chunk[1]]));
        for j in 0..32 {
            out[b * 32 + j] = f32::from(chunk[2 + j] as i8) * d;
        }
    }
    out
}

/// Unpack one of the eight 6-bit (scale, min) pairs GGML packs into the
/// 12-byte `scales` array of a Q4K superblock. Mirrors `get_scale_min_k4`.
fn q4k_scale_min(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        (
            (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    }
}

fn dequant_q4_k(data: &[u8], count: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; count];
    for (b, chunk) in data.as_chunks::<144>().0.iter().enumerate() {
        let d = f16_from_bits(u16::from_le_bytes([chunk[0], chunk[1]]));
        let dmin = f16_from_bits(u16::from_le_bytes([chunk[2], chunk[3]]));
        let scales = &chunk[4..16];
        let qs = &chunk[16..144];
        let base = b * 256;
        // Two 32-element sub-blocks per 32 nibble bytes: low nibbles feed the
        // even sub-block, high nibbles the odd one.
        for pair in 0..4 {
            let (sc_lo, m_lo) = q4k_scale_min(pair * 2, scales);
            let (sc_hi, m_hi) = q4k_scale_min(pair * 2 + 1, scales);
            let d1 = d * f32::from(sc_lo);
            let m1 = dmin * f32::from(m_lo);
            let d2 = d * f32::from(sc_hi);
            let m2 = dmin * f32::from(m_hi);
            for j in 0..32 {
                let byte = qs[pair * 32 + j];
                out[base + pair * 64 + j] = d1 * f32::from(byte & 0x0F) - m1;
                out[base + pair * 64 + 32 + j] = d2 * f32::from(byte >> 4) - m2;
            }
        }
    }
    out
}

fn dequant_q6_k(data: &[u8], count: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; count];
    for (b, chunk) in data.as_chunks::<210>().0.iter().enumerate() {
        let ql = &chunk[0..128];
        let qh = &chunk[128..192];
        let scales = &chunk[192..208];
        let d = f16_from_bits(u16::from_le_bytes([chunk[208], chunk[209]]));
        let base = b * 256;
        // Each 128-element half is walked in 32-element strides, with the two
        // high bits for each quant drawn from a shared `qh` byte.
        for half in 0..2 {
            let ql_half = &ql[half * 64..half * 64 + 64];
            let qh_half = &qh[half * 32..half * 32 + 32];
            // `block_q6_K::scales` is int8_t, and these do go negative.
            let sc = &scales[half * 8..half * 8 + 8];
            let scale = |i: usize| f32::from(sc[i] as i8);
            for j in 0..32 {
                let h = qh_half[j];
                let q1 = ((ql_half[j] & 0x0F) | ((h & 0x03) << 4)) as i32 - 32;
                let q2 = ((ql_half[j + 32] & 0x0F) | (((h >> 2) & 0x03) << 4)) as i32 - 32;
                let q3 = ((ql_half[j] >> 4) | (((h >> 4) & 0x03) << 4)) as i32 - 32;
                let q4 = ((ql_half[j + 32] >> 4) | (((h >> 6) & 0x03) << 4)) as i32 - 32;
                let o = base + half * 128;
                out[o + j] = d * scale(j / 16) * q1 as f32;
                out[o + j + 32] = d * scale(2 + j / 16) * q2 as f32;
                out[o + j + 64] = d * scale(4 + j / 16) * q3 as f32;
                out[o + j + 96] = d * scale(6 + j / 16) * q4 as f32;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal well-formed GGUF builder, so the parser is tested against
    /// bytes laid out per the spec rather than against itself.
    struct Builder {
        kv: Vec<u8>,
        kv_count: u64,
        infos: Vec<u8>,
        info_count: u64,
        data: Vec<u8>,
    }

    impl Builder {
        fn new() -> Self {
            Self {
                kv: Vec::new(),
                kv_count: 0,
                infos: Vec::new(),
                info_count: 0,
                data: Vec::new(),
            }
        }

        fn str_bytes(out: &mut Vec<u8>, s: &str) {
            out.extend_from_slice(&(s.len() as u64).to_le_bytes());
            out.extend_from_slice(s.as_bytes());
        }

        fn kv_u32(mut self, key: &str, v: u32) -> Self {
            Self::str_bytes(&mut self.kv, key);
            self.kv.extend_from_slice(&4u32.to_le_bytes());
            self.kv.extend_from_slice(&v.to_le_bytes());
            self.kv_count += 1;
            self
        }

        fn kv_string(mut self, key: &str, v: &str) -> Self {
            Self::str_bytes(&mut self.kv, key);
            self.kv.extend_from_slice(&8u32.to_le_bytes());
            Self::str_bytes(&mut self.kv, v);
            self.kv_count += 1;
            self
        }

        fn tensor(mut self, name: &str, dims: &[usize], ty: GgmlType, bytes: &[u8]) -> Self {
            Self::str_bytes(&mut self.infos, name);
            self.infos
                .extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for &d in dims {
                self.infos.extend_from_slice(&(d as u64).to_le_bytes());
            }
            self.infos.extend_from_slice(&(ty as u32).to_le_bytes());
            self.infos
                .extend_from_slice(&(self.data.len() as u64).to_le_bytes());
            self.data.extend_from_slice(bytes);
            // Keep every tensor offset alignment-legal.
            while !self.data.len().is_multiple_of(32) {
                self.data.push(0);
            }
            self.info_count += 1;
            self
        }

        fn build(self) -> Vec<u8> {
            let mut out = Vec::new();
            out.extend_from_slice(b"GGUF");
            out.extend_from_slice(&3u32.to_le_bytes());
            out.extend_from_slice(&self.info_count.to_le_bytes());
            out.extend_from_slice(&self.kv_count.to_le_bytes());
            out.extend_from_slice(&self.kv);
            out.extend_from_slice(&self.infos);
            while !out.len().is_multiple_of(32) {
                out.push(0);
            }
            out.extend_from_slice(&self.data);
            out
        }
    }

    fn q4_0_block(d: f32, nibbles: [u8; 32]) -> Vec<u8> {
        let mut b = Vec::with_capacity(18);
        b.extend_from_slice(&f16_to_bits(d).to_le_bytes());
        for j in 0..16 {
            b.push(nibbles[j] | (nibbles[j + 16] << 4));
        }
        b
    }

    fn q8_0_block(d: f32, quants: [i8; 32]) -> Vec<u8> {
        let mut b = Vec::with_capacity(34);
        b.extend_from_slice(&f16_to_bits(d).to_le_bytes());
        b.extend(quants.iter().map(|&q| q as u8));
        b
    }

    #[test]
    fn parses_header_metadata_and_tensor_bytes() {
        let payload: Vec<u8> = (0..8u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let bytes = Builder::new()
            .kv_string("general.architecture", "llama")
            .kv_u32("llama.block_count", 7)
            .tensor("blk.0.weight", &[4, 2], GgmlType::F32, &payload)
            .build();

        let m = load_gguf_bytes(&bytes).unwrap();
        assert_eq!(m.architecture(), Some("llama"));
        assert_eq!(m.arch_key("block_count").unwrap().as_u64(), Some(7));
        let t = &m.tensors["blk.0.weight"];
        assert_eq!(t.dims, vec![4, 2]);
        assert_eq!(t.ggml_type, GgmlType::F32);
        assert_eq!(t.num_elements(), 8);
    }

    #[test]
    fn f32_tensor_transposes_into_meganeura_order() {
        // GGUF [ne0=K=4, ne1=N=2]: element (k, n) sits at n * 4 + k.
        let payload: Vec<u8> = (0..8u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let bytes = Builder::new()
            .tensor("w", &[4, 2], GgmlType::F32, &payload)
            .build();
        let m = load_gguf_bytes(&bytes).unwrap();
        let got = m.tensors["w"].to_f32().unwrap();
        // Meganeura [K=4, N=2] wants (k, n) at k * 2 + n.
        let want: Vec<f32> = (0..4)
            .flat_map(|k| (0..2).map(move |n| (n * 4 + k) as f32))
            .collect();
        assert_eq!(got, want);
    }

    #[test]
    fn q4_0_repack_round_trips_through_meganeura_dequant() {
        // One 32-element column: K = 32, N = 1.
        let nibbles: [u8; 32] = std::array::from_fn(|i| (i % 16) as u8);
        let d = 0.25f32;
        let bytes = Builder::new()
            .tensor("w", &[32, 1], GgmlType::Q4_0, &q4_0_block(d, nibbles))
            .build();
        let m = load_gguf_bytes(&bytes).unwrap();
        let t = &m.tensors["w"];

        let (dtype, packed) = t.to_packed().unwrap();
        assert_eq!(dtype, DType::Q4_0);
        // Meganeura's own reader must see exactly what GGML's semantics say.
        let via_meganeura = crate::runtime::dequantize_q4_0(&packed, 32, 1);
        let want = t.to_f32().unwrap();
        for (i, (&got, &exp)) in via_meganeura.iter().zip(&want).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "element {i}: meganeura {got} vs ggml {exp}"
            );
        }
    }

    #[test]
    fn q4_0_repack_preserves_the_split_half_nibble_order() {
        // Distinct value per element, so any reordering shows up.
        let nibbles: [u8; 32] = std::array::from_fn(|i| (i / 2) as u8);
        let bytes = Builder::new()
            .tensor("w", &[32, 1], GgmlType::Q4_0, &q4_0_block(1.0, nibbles))
            .build();
        let m = load_gguf_bytes(&bytes).unwrap();
        let t = &m.tensors["w"];
        let (_, packed) = t.to_packed().unwrap();
        let got = crate::runtime::dequantize_q4_0(&packed, 32, 1);
        for (e, &n) in nibbles.iter().enumerate() {
            // Q4_0 is symmetric about 8.
            let want = f32::from(n) - 8.0;
            assert!(
                (got[e] - want).abs() < 1e-3,
                "element {e}: got {}, want {want}",
                got[e]
            );
        }
    }

    #[test]
    fn q8_0_repack_is_byte_identical_modulo_padding() {
        let quants: [i8; 32] = std::array::from_fn(|i| (i as i32 - 16) as i8);
        let d = 0.5f32;
        let bytes = Builder::new()
            .tensor("w", &[32, 1], GgmlType::Q8_0, &q8_0_block(d, quants))
            .build();
        let m = load_gguf_bytes(&bytes).unwrap();
        let t = &m.tensors["w"];
        let (dtype, packed) = t.to_packed().unwrap();
        assert_eq!(dtype, DType::Q8_0);
        assert_eq!(packed.len(), 36);
        let got = crate::runtime::dequantize_q8_0(&packed, 32, 1);
        for (i, &q) in quants.iter().enumerate() {
            let want = f32::from(q) * d;
            assert!((got[i] - want).abs() < 1e-3, "element {i}");
        }
    }

    /// The CPU reference the Q4_K shader is tested against, on a superblock
    /// whose answer is worked out by hand.
    #[test]
    fn q4_k_dequantizes_against_a_known_value() {
        let mut block = vec![0u8; 144];
        block[0..2].copy_from_slice(&f16_to_bits(1.0).to_le_bytes());
        block[2..4].copy_from_slice(&f16_to_bits(0.0).to_le_bytes());
        // Sub-block 0 gets scale 1; every quant nibble is 1.
        block[4] = 1;
        for b in block.iter_mut().skip(16) {
            *b = 0x11;
        }
        let bytes = Builder::new()
            .tensor("w", &[256, 1], GgmlType::Q4K, &block)
            .build();
        let m = load_gguf_bytes(&bytes).unwrap();
        let f = m.tensors["w"].to_f32().unwrap();
        assert_eq!(f.len(), 256);
        // d = 1, sc_0 = 1, q = 1, dmin = 0 => 1.
        assert!((f[0] - 1.0).abs() < 1e-3, "got {}", f[0]);
    }

    /// `block_q6_K::scales` is int8_t. Reading it unsigned turns a -1 scale
    /// into 255, which is a 255x error in the wrong direction and silently
    /// plausible on any tensor whose scales happen to be positive.
    #[test]
    fn q6_k_honours_signed_scales() {
        let mut block = vec![0u8; 210];
        // ql/qh zero => every 6-bit quant is 0 - 32 = -32.
        // Scale group 0 negative, group 1 positive.
        block[192] = (-1i8) as u8;
        block[193] = 2i8 as u8;
        block[208..210].copy_from_slice(&f16_to_bits(1.0).to_le_bytes());
        let bytes = Builder::new()
            .tensor("w", &[256, 1], GgmlType::Q6K, &block)
            .build();
        let m = load_gguf_bytes(&bytes).unwrap();
        let f = m.tensors["w"].to_f32().unwrap();
        // is = l/16, so elements 0..16 use scales[0] and 16..32 use scales[1].
        assert!(
            (f[0] - 32.0).abs() < 1e-3,
            "scales[0]=-1 => +32, got {}",
            f[0]
        );
        assert!(
            (f[16] + 64.0).abs() < 1e-3,
            "scales[1]=2 => -64, got {}",
            f[16]
        );
    }

    /// Q4_K is stored in GGML's own layout, so packing is a copy. Anything
    /// that rearranged bytes here would have to be mirrored in the shader.
    #[test]
    fn q4_k_packs_without_touching_the_bytes() {
        let mut block = vec![0u8; 144];
        block[0..2].copy_from_slice(&f16_to_bits(0.0035).to_le_bytes());
        block[2..4].copy_from_slice(&f16_to_bits(0.0021).to_le_bytes());
        for (i, b) in block.iter_mut().enumerate().skip(4) {
            *b = (i * 7 % 251) as u8;
        }
        let bytes = Builder::new()
            .tensor("w", &[256, 1], GgmlType::Q4K, &block)
            .build();
        let m = load_gguf_bytes(&bytes).unwrap();
        let (dtype, packed) = m.tensors["w"].to_packed().unwrap();
        assert_eq!(dtype, DType::Q4K);
        assert_eq!(packed, block, "Q4_K must reach the GPU byte-for-byte");
        // And the declared parameter size has to agree with the file.
        let ty = crate::graph::TensorType::new(vec![256, 1], DType::Q4K);
        assert_eq!(ty.size_bytes(), packed.len());
    }

    /// Q6_K superblocks are 210 bytes, so an odd count leaves the buffer two
    /// bytes short of a word. The tail is padded for the `array<u32>`
    /// binding; the superblocks themselves must survive untouched.
    #[test]
    fn q6_k_packs_verbatim_with_a_padded_tail() {
        let mut block = vec![0u8; 210];
        for (i, b) in block.iter_mut().enumerate() {
            *b = (i * 5 % 253) as u8;
        }
        let bytes = Builder::new()
            .tensor("w", &[256, 1], GgmlType::Q6K, &block)
            .build();
        let m = load_gguf_bytes(&bytes).unwrap();
        let (dtype, packed) = m.tensors["w"].to_packed().unwrap();
        assert_eq!(dtype, DType::Q6K);
        assert_eq!(packed.len(), 212, "one superblock rounds up to 53 words");
        assert_eq!(&packed[..210], &block[..], "superblock must be verbatim");
        assert_eq!(&packed[210..], &[0, 0], "tail is zero padding");
        // The declared parameter size has to agree with what packing emits.
        let ty = crate::graph::TensorType::new(vec![256, 1], DType::Q6K);
        assert_eq!(ty.size_bytes(), packed.len());
    }

    #[test]
    fn rejects_bad_magic_and_truncation() {
        assert!(matches!(
            load_gguf_bytes(b"NOPE____________"),
            Err(GgufError::BadHeader(_))
        ));
        let good = Builder::new()
            .tensor("w", &[32, 1], GgmlType::Q8_0, &q8_0_block(1.0, [0; 32]))
            .build();
        // The data section is the 34-byte block plus alignment padding, so
        // cut past the padding to actually shorten the tensor.
        assert!(matches!(
            load_gguf_bytes(&good[..good.len() - 40]),
            Err(GgufError::Truncated { .. })
        ));
    }

    #[test]
    fn rejects_misaligned_reduction_extent() {
        // K = 16 is not a whole Q8 block.
        let bytes = Builder::new()
            .tensor("w", &[16, 1], GgmlType::F32, &[0u8; 64])
            .build();
        let m = load_gguf_bytes(&bytes).unwrap();
        // F32 has no block constraint, but asking for a packed form does.
        assert!(matches!(
            m.tensors["w"].to_packed(),
            Err(GgufError::UnsupportedPack(GgmlType::F32))
        ));
    }
}
