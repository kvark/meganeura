//! GGUF import: read GGML's container format into a runnable model.
//!
//! GGUF is a weight-and-metadata container rather than a graph format: it
//! carries named tensors plus key/value metadata describing the architecture
//! (`llama.block_count`, `llama.attention.head_count`, …), and leaves the
//! graph implied. [`arch`] reads that description into a [`ModelConfig`] and
//! [`graph`] builds the graph it implies, so the file is enough on its own —
//! nothing here is compiled against a particular model's dimensions.
//!
//! The whole path, from a path on disk to generated text, is
//! [`GgufModel::generator`]:
//!
//! ```no_run
//! use meganeura::load::gguf::{load_gguf, GenerationOptions};
//!
//! let model = load_gguf(std::path::Path::new("model.gguf"))?;
//! let mut gen = model.generator(256)?;
//! println!("{}", gen.generate("The meaning of life is", &GenerationOptions::default())?);
//! # Ok::<(), meganeura::load::gguf::GgufError>(())
//! ```
//!
//! The pieces are separable for callers who want fewer of them: [`arch`] for
//! the description, [`graph`] for the graph, [`weights`] to fill a session
//! from the file, [`vocab`] for the tokenizer the file embeds, and
//! [`generate`] for the loop over the two sessions.
//!
//! The rest of this module is the container itself — parsing, and the
//! block-format conversions that let a GGML tensor land in a Meganeura
//! parameter without losing a bit.
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
//! The *packed* layouts agree. Meganeura's quantizers block along the
//! parameter's **first** dimension, emitting block `n * (K/32) + k/32` for a
//! `[K, N]` weight; GGUF blocks run along `ne0 = K` within each row `n`,
//! which is the same index. Packing performs the transpose itself, so
//! [`GgufTensor::to_packed`] never has to. What differs is the arrangement
//! *within* a block, which is what the repack functions below fix up.
//!
//! That agreement is specific to the forward orientation. Blocking follows
//! the first dimension while every packed decoder indexes along `params.k`;
//! for a transposed `[N, K]` weight those are different axes, so block
//! formats have no correct reading on `MatMulBT` and `compile.rs` refuses
//! them there.
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

pub mod arch;
#[cfg(test)]
pub(crate) mod fixture;
pub mod generate;
pub mod graph;
pub mod vocab;
pub mod weights;

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::graph::DType;

pub use arch::{Architecture, ModelConfig};
pub use generate::{GenerationOptions, Generator, GeneratorOptions};
pub use graph::ModelGraph;
pub use vocab::{TokenizerKind, Vocab};
pub use weights::LoadReport;

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
    /// A type that has no packed Meganeura form. `F32` and `F16` are
    /// stored unpacked — read them with [`GgufTensor::to_f32`].
    UnsupportedPack(GgmlType),
    /// A `ggml_type` this loader does not implement. Such tensors are
    /// still listed in the inventory, but their bytes are not read, so
    /// neither [`GgufTensor::to_packed`] nor [`GgufTensor::to_f32`] can
    /// produce values for them.
    UnsupportedType(u32),
    /// A metadata key the graph builder needs is absent. Holds the key as
    /// the file would have written it, architecture prefix and all.
    MissingKey(String),
    /// A metadata key is present but holds something unusable — the wrong
    /// value type, or a number outside the range it must lie in.
    BadMetadata(String),
    /// `general.architecture` names a family this loader cannot build a
    /// graph for. The weights are still readable; only
    /// [`ModelConfig::from_gguf`] and the builders above it refuse.
    UnsupportedArchitecture(String),
    /// A tensor the graph declares as a parameter is absent from the file,
    /// or is present with the wrong shape.
    MissingTensor(String),
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
                "{t:?} has no packed Meganeura form; read it with to_f32()"
            ),
            Self::UnsupportedType(tag) => {
                write!(f, "ggml_type {tag} is not implemented by this loader")
            }
            Self::MissingKey(ref k) => write!(f, "GGUF metadata has no `{k}`"),
            Self::BadMetadata(ref e) => write!(f, "GGUF metadata error: {e}"),
            Self::UnsupportedArchitecture(ref a) => write!(
                f,
                "no graph builder for architecture `{a}`; its weights can still be \
                 read tensor by tensor"
            ),
            Self::MissingTensor(ref e) => write!(f, "GGUF tensor error: {e}"),
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

/// The `ggml_type` tags this loader understands, plus a catch-all.
///
/// Tags are GGML's and are part of the file format; [`GgmlType::tag`]
/// maps back to them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q8_0,
    Q4K,
    Q6K,
    Q5K,
    Q3K,
    /// GGML BF16, two bytes per element. Unpacked to f32 on read.
    BF16,
    /// A tag outside the set above. Carried so that a file holding one
    /// unimplemented tensor still yields an inventory for the rest; the
    /// tensor's bytes are not read, because its length depends on a block
    /// size this loader does not know.
    Other(u32),
}

impl GgmlType {
    fn from_tag(tag: u32) -> Self {
        match tag {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            8 => Self::Q8_0,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            30 => Self::BF16,
            other => Self::Other(other),
        }
    }

    /// Whether this loader can read values of this type.
    pub fn is_supported(self) -> bool {
        !matches!(self, Self::Other(_))
    }

    /// Elements per stored block, or `None` for an unimplemented type.
    /// Non-block types report 1.
    pub fn block_elements(self) -> Option<usize> {
        Some(match self {
            Self::F32 | Self::F16 | Self::BF16 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q8_0 => 32,
            Self::Q4K | Self::Q6K | Self::Q5K | Self::Q3K => 256,
            Self::Other(_) => return None,
        })
    }

    /// Bytes per stored block, straight from `ggml-common.h`, or `None`
    /// for an unimplemented type.
    pub fn block_bytes(self) -> Option<usize> {
        Some(match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
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
            // f16 d + f16 dmin + 12 scale bytes + 32 high-bit + 128 nibble
            Self::Q5K => 176,
            // 32 hmask + 64 two-bit quants + 12 scale bytes + f16 d
            Self::Q3K => 110,
            Self::Other(_) => return None,
        })
    }

    /// Byte length of `count` elements stored in this type.
    fn stored_bytes(self, count: usize) -> Result<usize, GgufError> {
        let (Some(per), Some(bytes)) = (self.block_elements(), self.block_bytes()) else {
            return Err(GgufError::UnsupportedType(self.tag()));
        };
        if !count.is_multiple_of(per) {
            return Err(GgufError::BadShape(format!(
                "{count} elements is not a whole number of {per}-element {self:?} blocks"
            )));
        }
        (count / per)
            .checked_mul(bytes)
            .ok_or_else(|| GgufError::BadShape(format!("{count} {self:?} elements overflow")))
    }

    /// The numeric tag as it appears in the file.
    pub fn tag(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::BF16 => 30,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q8_0 => 8,
            Self::Q4K => 12,
            Self::Q6K => 14,
            Self::Q5K => 13,
            Self::Q3K => 11,
            Self::Other(tag) => tag,
        }
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
            Self::U8(v) => f64::from(v),
            Self::I8(v) => f64::from(v),
            Self::U16(v) => f64::from(v),
            Self::I16(v) => f64::from(v),
            Self::U32(v) => f64::from(v),
            Self::I32(v) => f64::from(v),
            Self::U64(v) => v as f64,
            Self::I64(v) => v as f64,
            _ => return None,
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
#[derive(Clone)]
pub struct GgufTensor {
    /// Dimensions in GGUF order, fastest-varying first. For a 2-D
    /// projection weight this is `[K, N]`.
    pub dims: Vec<usize>,
    /// How the bytes are encoded.
    pub ggml_type: GgmlType,
    /// The whole file, shared by every tensor in the model.
    file: Arc<[u8]>,
    /// This tensor's slice of it. Empty for a type whose block size is
    /// unknown, since there is no way to say where its bytes end.
    range: std::ops::Range<usize>,
}

impl std::fmt::Debug for GgufTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgufTensor")
            .field("dims", &self.dims)
            .field("ggml_type", &self.ggml_type)
            .field("data_range", &self.range)
            .finish()
    }
}

impl GgufTensor {
    /// Build a tensor from its own bytes, for callers synthesizing one
    /// rather than reading a file.
    pub fn new(dims: Vec<usize>, ggml_type: GgmlType, data: Vec<u8>) -> Self {
        let range = 0..data.len();
        Self {
            dims,
            ggml_type,
            file: Arc::from(data),
            range,
        }
    }

    /// This tensor's raw bytes, exactly as they appear in the file.
    ///
    /// Borrowed from the model's single backing buffer — reading a tensor
    /// costs nothing beyond the file itself.
    pub fn data(&self) -> &[u8] {
        &self.file[self.range.clone()]
    }
    /// Total element count.
    pub fn num_elements(&self) -> usize {
        self.dims.iter().product()
    }

    fn element_count(&self) -> Result<usize, GgufError> {
        self.dims
            .iter()
            .try_fold(1usize, |n, &dim| n.checked_mul(dim))
            .ok_or_else(|| {
                GgufError::BadShape(format!(
                    "tensor dimensions {:?} overflow this platform",
                    self.dims
                ))
            })
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

    /// Dequantize to f32 in GGUF's own order: `[ne1, ne0]` row-major, the
    /// transpose of what [`GgufTensor::to_f32`] returns.
    ///
    /// This is the orientation an *embedding table* wants. A projection
    /// weight is a matrix whose reduction extent Meganeura stores slowest,
    /// so it needs the transpose; a lookup table is a list of rows, and
    /// GGUF already stores `token_embd.weight` as `n_vocab` rows of
    /// `n_embd`, which is exactly `Graph::embedding`'s `[vocab, hidden]`.
    /// Transposing it would be a second, wrong conversion.
    pub fn to_f32_rows(&self) -> Result<Vec<f32>, GgufError> {
        // Element (i0, i1) already lies at i1 * ne0 + i0, which reads as
        // row i1 of length ne0.
        self.dequantize_flat()
    }

    /// One column of a 2-D tensor (or the whole vector, for 1-D), dequantized
    /// to f32 in Meganeura's K-major order.
    ///
    /// Used for embedding tables that have no GPU gather: decode looks up a
    /// single token, which is one GGUF column, without dequantizing the rest
    /// of a multi-gigabyte PLE table.
    pub fn column_f32(&self, col: usize) -> Result<Vec<f32>, GgufError> {
        let (k, n) = self.matrix_dims()?;
        if col >= n {
            return Err(GgufError::BadShape(format!(
                "column {col} is out of range for shape {:?}",
                self.dims
            )));
        }
        let bytes = self
            .ggml_type
            .stored_bytes(k)
            .map_err(|_| GgufError::UnsupportedType(self.ggml_type.tag()))?;
        let start = col * bytes;
        let end = start + bytes;
        let slice = self.data().get(start..end).ok_or(GgufError::Truncated {
            what: "tensor column",
            offset: start,
        })?;
        let column = GgufTensor::new(vec![k], self.ggml_type, slice.to_vec());
        column.dequantize_flat()
    }

    /// The [`DType`] this tensor would occupy, without converting it.
    ///
    /// [`GgufTensor::to_packed`] allocates a whole tensor; callers that only
    /// need the destination — an inventory listing, a shape check — should
    /// ask here instead.
    pub fn packed_dtype(&self) -> Result<DType, GgufError> {
        let (k, _n) = self.matrix_dims()?;
        match self.ggml_type {
            // GGML's Q4_0 is read natively; Q4_1 keeps the repack, because
            // Meganeura's own Q4 is the Q4_1 shape and the conversion is
            // lossless.
            GgmlType::Q4_0 => {
                require_block_aligned(k, 32, "Q4_0")?;
                Ok(DType::Q40)
            }
            GgmlType::Q4_1 => {
                require_block_aligned(k, 32, "Q4_1")?;
                Ok(DType::Q4_0)
            }
            GgmlType::Q8_0 => {
                require_block_aligned(k, 32, "Q8")?;
                Ok(DType::Q8_0)
            }
            GgmlType::Q4K => {
                require_block_aligned(k, 256, "Q4_K")?;
                Ok(DType::Q4K)
            }
            GgmlType::Q6K => {
                require_block_aligned(k, 256, "Q6_K")?;
                Ok(DType::Q6K)
            }
            GgmlType::Q5K => {
                require_block_aligned(k, 256, "Q5_K")?;
                Ok(DType::Q5K)
            }
            GgmlType::Q3K => {
                require_block_aligned(k, 256, "Q3_K")?;
                Ok(DType::Q3K)
            }
            GgmlType::Other(tag) => Err(GgufError::UnsupportedType(tag)),
            other => Err(GgufError::UnsupportedPack(other)),
        }
    }

    /// Repack into Meganeura's packed layout for `Session::set_parameter_packed`,
    /// returning the [`DType`] the parameter must be declared with.
    ///
    /// Lossless for every quantized type. `Q4_0`, `Q4_1` and `Q8_0` are
    /// rearranged into Meganeura's block layout without touching a value;
    /// the K-quants need no work at all, since Meganeura stores them in
    /// GGML's own layout and this hands back the file's bytes (Q6_K with a
    /// zero-padded tail, its 210-byte superblocks not being a whole number
    /// of words).
    ///
    /// `F32` and `F16` are not packed formats and are rejected — read them
    /// with [`GgufTensor::to_f32`]. So is any `ggml_type` this loader does
    /// not implement.
    pub fn to_packed(&self) -> Result<(DType, Cow<'_, [u8]>), GgufError> {
        let dtype = self.packed_dtype()?;
        let count = self.element_count()?;
        let expect = self.ggml_type.stored_bytes(count)?;
        if self.data().len() != expect {
            return Err(GgufError::BadShape(format!(
                "{:?} tensor of {count} elements needs {expect} bytes, has {}",
                self.ggml_type,
                self.data().len()
            )));
        }
        let bytes = match self.ggml_type {
            GgmlType::Q4_1 => Cow::Owned(self.repack_q4(count)?),
            GgmlType::Q8_0 => Cow::Owned(self.repack_q8(count)),
            // Q4_0 and the K-quants are already in GGML's layout, so these
            // borrow the file rather than copying it.
            // 144 and 176 are whole numbers of words, so these need no tail.
            GgmlType::Q4K | GgmlType::Q5K => Cow::Borrowed(self.data()),
            GgmlType::Q4_0 | GgmlType::Q6K | GgmlType::Q3K => {
                // 18-, 210- and 110-byte blocks are not whole words, so an
                // odd count leaves the buffer short of the `array<u32>`
                // binding. Only then is a copy needed; the blocks
                // themselves are never touched.
                let data = self.data();
                let padded = data.len().next_multiple_of(4);
                if padded == data.len() {
                    Cow::Borrowed(data)
                } else {
                    let mut bytes = data.to_vec();
                    bytes.resize(padded, 0);
                    Cow::Owned(bytes)
                }
            }
            // `packed_dtype` has already rejected everything else.
            _ => unreachable!("packed_dtype accepted an unpackable type"),
        };
        Ok((dtype, bytes))
    }

    /// GGUF-order values, one f32 per element.
    fn dequantize_flat(&self) -> Result<Vec<f32>, GgufError> {
        let count = self.element_count()?;
        let expect = self.ggml_type.stored_bytes(count)?;
        if self.data().len() != expect {
            return Err(GgufError::BadShape(format!(
                "{:?} tensor of {count} elements needs {expect} bytes, has {}",
                self.ggml_type,
                self.data().len()
            )));
        }
        Ok(match self.ggml_type {
            GgmlType::F32 => self
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&c| f32::from_le_bytes(c))
                .collect(),
            GgmlType::F16 => self
                .data()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&c| f16_from_bits(u16::from_le_bytes(c)))
                .collect(),
            GgmlType::BF16 => self
                .data()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&c| bf16_from_bits(u16::from_le_bytes(c)))
                .collect(),
            GgmlType::Q4_0 => dequant_q4_0(self.data(), count),
            GgmlType::Q4_1 => dequant_q4_1(self.data(), count),
            GgmlType::Q8_0 => dequant_q8_0(self.data(), count),
            GgmlType::Q4K => dequant_q4_k(self.data(), count),
            GgmlType::Q6K => dequant_q6_k(self.data(), count),
            GgmlType::Q5K => dequant_q5_k(self.data(), count),
            GgmlType::Q3K => dequant_q3_k(self.data(), count),
            GgmlType::Other(tag) => return Err(GgufError::UnsupportedType(tag)),
        })
    }

    /// GGML Q4_1 → Meganeura Q4.
    ///
    /// Meganeura's own Q4 *is* the Q4_1 shape — `value = q * d + m` with a
    /// per-block minimum — so this is a relayout, not a requantize. Two
    /// things change and nothing else does:
    ///
    /// * GGML splits a block's nibbles across halves — `qs[j]` holds element
    ///   `j` low and element `j + 16` high — while Meganeura pairs adjacent
    ///   elements, `qs[e / 2]` holding `e` and `e + 1`.
    /// * GGML interleaves each block's header with its payload; Meganeura
    ///   keeps all `(d, m)` words first and all nibble words after.
    ///
    /// GGML's symmetric Q4_0 does not come through here: it has its own
    /// storage now, so nothing is rebuilt for it. See [`DType::Q40`].
    fn repack_q4(&self, count: usize) -> Result<Vec<u8>, GgufError> {
        debug_assert_eq!(self.ggml_type, GgmlType::Q4_1);
        let src_bytes = self.data();
        let blocks = count / 32;
        let stride = self
            .ggml_type
            .block_bytes()
            .ok_or(GgufError::UnsupportedType(self.ggml_type.tag()))?;
        // Header region then payload region, mirroring `quantize_q4_0`.
        let mut out = vec![0u8; blocks * 4 + blocks * 16];
        let payload = blocks * 4;
        for b in 0..blocks {
            let src = b * stride;
            let d_bits = u16::from_le_bytes([src_bytes[src], src_bytes[src + 1]]);
            let m_bits = u16::from_le_bytes([src_bytes[src + 2], src_bytes[src + 3]]);
            let qs = src + 4;
            let dm = u32::from(d_bits) | (u32::from(m_bits) << 16);
            out[b * 4..b * 4 + 4].copy_from_slice(&dm.to_le_bytes());

            let dst = payload + b * 16;
            for e in 0..32u32 {
                // Where GGML put element e.
                let byte = (e % 16) as usize;
                let nibble = if e < 16 {
                    src_bytes[qs + byte] & 0x0F
                } else {
                    src_bytes[qs + byte] >> 4
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
    fn repack_q8(&self, count: usize) -> Vec<u8> {
        let src_bytes = self.data();
        let blocks = count / 32;
        let mut out = vec![0u8; blocks * 36];
        for b in 0..blocks {
            let src = b * 34;
            let dst = b * 36;
            out[dst..dst + 2].copy_from_slice(&src_bytes[src..src + 2]);
            // dst + 2 .. dst + 4 stays zero padding.
            out[dst + 4..dst + 36].copy_from_slice(&src_bytes[src + 2..src + 34]);
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
///
/// The file is retained once and shared by every tensor rather than copied
/// into one allocation per payload.
pub fn load_gguf(path: &Path) -> Result<GgufModel, GgufError> {
    load_gguf_shared(Arc::from(std::fs::read(path)?))
}

/// Parse a GGUF file already in memory.
///
/// Copies `bytes` into the shared buffer. A caller that can hand over
/// ownership should use [`load_gguf_shared`] and avoid the copy.
pub fn load_gguf_bytes(bytes: &[u8]) -> Result<GgufModel, GgufError> {
    load_gguf_shared(Arc::from(bytes.to_vec()))
}

/// Parse a GGUF file from a buffer the model can take a share of.
///
/// Tensors borrow ranges of this buffer; none of them copies its payload.
pub fn load_gguf_shared(file: Arc<[u8]>) -> Result<GgufModel, GgufError> {
    let bytes: &[u8] = &file;
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
    // These counts are file-controlled. Reserving on them directly lets a
    // 24-byte header ask for a `usize::MAX` allocation, which aborts on
    // capacity overflow instead of returning an error. The smallest record
    // either can produce is a few bytes, so anything beyond the remaining
    // input is a malformed header; reject that and cap eager reservations.
    let remaining = bytes.len() - r.pos;
    if tensor_count > remaining || kv_count > remaining {
        return Err(GgufError::BadHeader(format!(
            "header declares {tensor_count} tensors and {kv_count} metadata entries, \
             more than the {remaining} bytes that follow it"
        )));
    }

    let mut metadata = HashMap::with_capacity(kv_count.min(1024));
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
    let mut infos = Vec::with_capacity(tensor_count.min(1024));
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
        let ggml_type = GgmlType::from_tag(r.u32()?);
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
    let alignment = match metadata.get("general.alignment") {
        Some(value) => value.as_u64().ok_or_else(|| {
            GgufError::BadHeader("general.alignment must be an unsigned integer".into())
        })?,
        None => 32,
    };
    let alignment = usize::try_from(alignment)
        .map_err(|_| GgufError::BadHeader("general.alignment is too large".into()))?;
    if alignment == 0 || !alignment.is_multiple_of(8) {
        return Err(GgufError::BadHeader(format!(
            "general.alignment must be a non-zero multiple of 8, got {alignment}"
        )));
    }
    let data_start = r
        .pos
        .checked_add(alignment - 1)
        .map(|end| end / alignment * alignment)
        .ok_or_else(|| GgufError::BadHeader("tensor-data alignment overflows".into()))?;

    let mut tensors: HashMap<String, GgufTensor> = HashMap::with_capacity(infos.len());
    for info in infos {
        if !info.offset.is_multiple_of(alignment) {
            return Err(GgufError::BadHeader(format!(
                "tensor `{}` offset {} is not aligned to {alignment} bytes",
                info.name, info.offset
            )));
        }
        let count = info
            .dims
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| {
                GgufError::BadShape(format!(
                    "tensor `{}` has dimensions {:?} whose product overflows this platform",
                    info.name, info.dims
                ))
            })?;
        // A type this loader does not implement has an unknown block size,
        // so there is no way to say where its bytes end. List it with no
        // data rather than failing the whole file: a mix that is mostly
        // readable should still yield an inventory, and `to_packed` /
        // `to_f32` report the unimplemented tag when someone asks for
        // values.
        let range = if info.ggml_type.is_supported() {
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
            start..end
        } else {
            0..0
        };
        if tensors.contains_key(&info.name) {
            return Err(GgufError::BadShape(format!(
                "duplicate tensor name `{}`",
                info.name
            )));
        }
        tensors.insert(
            info.name,
            GgufTensor {
                dims: info.dims,
                ggml_type: info.ggml_type,
                file: Arc::clone(&file),
                range,
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

fn bf16_from_bits(bits: u16) -> f32 {
    half::bf16::from_bits(bits).to_f32()
}

/// Only the fixtures need this direction: nothing in the load path writes
/// f16 any more, now that Q4_0 is stored as GGML wrote it.
#[cfg(test)]
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

/// Mirrors `dequantize_row_q5_K`. Q4_K's nibble plus one bit from `qh`,
/// whose bit index is the sub-block number — `qh` is indexed by position
/// within the 32-element stride and shared across all four spans.
fn dequant_q5_k(data: &[u8], count: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; count];
    for (b, chunk) in data.as_chunks::<176>().0.iter().enumerate() {
        let d = f16_from_bits(u16::from_le_bytes([chunk[0], chunk[1]]));
        let dmin = f16_from_bits(u16::from_le_bytes([chunk[2], chunk[3]]));
        let scales = &chunk[4..16];
        let qh = &chunk[16..48];
        let qs = &chunk[48..176];
        let base = b * 256;
        for pair in 0..4 {
            let (sc_lo, m_lo) = q4k_scale_min(pair * 2, scales);
            let (sc_hi, m_hi) = q4k_scale_min(pair * 2 + 1, scales);
            let d1 = d * f32::from(sc_lo);
            let m1 = dmin * f32::from(m_lo);
            let d2 = d * f32::from(sc_hi);
            let m2 = dmin * f32::from(m_hi);
            for l in 0..32 {
                let byte = qs[pair * 32 + l];
                let bits = qh[l];
                let lo =
                    u32::from(byte & 0x0F) + if bits & (1 << (pair * 2)) != 0 { 16 } else { 0 };
                let hi = u32::from(byte >> 4)
                    + if bits & (1 << (pair * 2 + 1)) != 0 {
                        16
                    } else {
                        0
                    };
                out[base + pair * 64 + l] = d1 * lo as f32 - m1;
                out[base + pair * 64 + 32 + l] = d2 * hi as f32 - m2;
            }
        }
    }
    out
}

/// Scale `i` of sixteen for a Q3_K superblock, before the -32 bias.
///
/// The twelve stored bytes expand to sixteen 6-bit values: groups 0 and 1
/// take the low nibbles of `scales[0..8]`, groups 2 and 3 the high
/// nibbles, and each borrows two more bits from `scales[8..12]`. Written
/// from the `kmask1`/`kmask2` shuffle in `dequantize_row_q3_K` rather than
/// from the shader.
fn q3k_scale_6bit(i: usize, scales: &[u8]) -> u8 {
    let b = i % 4;
    let g = i / 4;
    let raw = scales[if g.is_multiple_of(2) { b } else { 4 + b }];
    let nib = if g < 2 { raw & 0x0F } else { raw >> 4 };
    let hi = (scales[8 + b] >> (g * 2)) & 0x03;
    nib | (hi << 4)
}

/// Mirrors `dequantize_row_q3_K`. Note the inverted high bit: a *clear*
/// `hmask` bit subtracts 4 from the 2-bit quant.
fn dequant_q3_k(data: &[u8], count: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; count];
    for (b, chunk) in data.as_chunks::<110>().0.iter().enumerate() {
        let hmask = &chunk[0..32];
        let qs = &chunk[32..96];
        let scales = &chunk[96..108];
        let d = f16_from_bits(u16::from_le_bytes([chunk[108], chunk[109]]));
        let base = b * 256;
        for half in 0..2 {
            for j in 0..4 {
                for within in 0..32 {
                    let sub = within / 16;
                    let l = sub * 16 + within % 16;
                    let q = (qs[half * 32 + l] >> (j * 2)) & 0x03;
                    let set = hmask[l] & (1 << (half * 4 + j)) != 0;
                    let v = i32::from(q) - if set { 0 } else { 4 };
                    let sc = f32::from(q3k_scale_6bit(half * 8 + j * 2 + sub, scales)) - 32.0;
                    out[base + half * 128 + j * 32 + within] = d * sc * v as f32;
                }
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
        alignment: usize,
    }

    impl Builder {
        fn new() -> Self {
            Self {
                kv: Vec::new(),
                kv_count: 0,
                infos: Vec::new(),
                info_count: 0,
                data: Vec::new(),
                alignment: 32,
            }
        }

        pub(super) fn str_bytes(out: &mut Vec<u8>, s: &str) {
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

        fn alignment(self, alignment: usize) -> Self {
            let mut this = self.kv_u32("general.alignment", alignment as u32);
            this.alignment = alignment;
            this
        }

        fn tensor(mut self, name: &str, dims: &[usize], ty: GgmlType, bytes: &[u8]) -> Self {
            let offset = self.data.len();
            self = self.tensor_at_offset(name, dims, ty, offset, bytes);
            self
        }

        fn tensor_at_offset(
            mut self,
            name: &str,
            dims: &[usize],
            ty: GgmlType,
            offset: usize,
            bytes: &[u8],
        ) -> Self {
            Self::str_bytes(&mut self.infos, name);
            self.infos
                .extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for &d in dims {
                self.infos.extend_from_slice(&(d as u64).to_le_bytes());
            }
            self.infos.extend_from_slice(&ty.tag().to_le_bytes());
            self.infos.extend_from_slice(&(offset as u64).to_le_bytes());
            self.data.resize(offset, 0);
            self.data.extend_from_slice(bytes);
            // Keep every tensor offset alignment-legal.
            while !self.data.len().is_multiple_of(self.alignment) {
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
            while !out.len().is_multiple_of(self.alignment) {
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

    fn q4_1_block(d: f32, m: f32, nibbles: [u8; 32]) -> Vec<u8> {
        let mut b = Vec::with_capacity(20);
        b.extend_from_slice(&f16_to_bits(d).to_le_bytes());
        b.extend_from_slice(&f16_to_bits(m).to_le_bytes());
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
        assert_eq!(GgufValue::I32(-7).as_f64(), Some(-7.0));
        assert_eq!(GgufValue::I32(-7).as_u64(), None);
    }

    #[test]
    fn validates_spec_alignment_without_assuming_a_power_of_two() {
        let valid = Builder::new()
            .alignment(24)
            .tensor("w", &[1], GgmlType::F32, &1.0f32.to_le_bytes())
            .build();
        assert!(
            load_gguf_bytes(&valid).is_ok(),
            "24-byte alignment is legal"
        );

        let too_small = Builder::new().kv_u32("general.alignment", 4).build();
        assert!(matches!(
            load_gguf_bytes(&too_small),
            Err(GgufError::BadHeader(_))
        ));

        let bad_offset = Builder::new()
            .tensor_at_offset("w", &[1], GgmlType::F32, 1, &1.0f32.to_le_bytes())
            .build();
        let err = load_gguf_bytes(&bad_offset).unwrap_err();
        assert!(
            matches!(err, GgufError::BadHeader(ref e) if e.contains("offset 1")),
            "misaligned tensor offset produced {err}"
        );
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
    fn quantized_repack_round_trips_multiple_columns() {
        // N = 1 cannot catch a column-order mistake in block layout. Exercise
        // every transcoded format through Meganeura's own decoder.
        let left: [u8; 32] = std::array::from_fn(|i| (i % 16) as u8);
        let right: [u8; 32] = std::array::from_fn(|i| ((i * 3) % 16) as u8);
        // Q4_0 is absent because it no longer transcodes: it is read
        // natively, and its column order is checked against GGML's own
        // dequantizer on the GPU instead.
        let cases = [
            (GgmlType::Q4_1, 1e-2, {
                let mut p = q4_1_block(0.5, -3.0, left);
                p.extend(q4_1_block(0.25, -1.0, right));
                p
            }),
            (GgmlType::Q8_0, 1e-3, {
                let l = std::array::from_fn(|i| (i as i32 - 16) as i8);
                let r = std::array::from_fn(|i| (8 - i as i32) as i8);
                let mut p = q8_0_block(0.5, l);
                p.extend(q8_0_block(0.25, r));
                p
            }),
        ];

        for (ty, tolerance, payload) in cases {
            let bytes = Builder::new().tensor("w", &[32, 2], ty, &payload).build();
            let model = load_gguf_bytes(&bytes).unwrap();
            let tensor = &model.tensors["w"];
            let (dtype, packed) = tensor.to_packed().unwrap();
            let got = if ty == GgmlType::Q8_0 {
                crate::runtime::dequantize_q8_0(&packed, 32, 2)
            } else {
                crate::runtime::dequantize_q4_0(&packed, 32, 2)
            };
            assert_eq!(
                dtype,
                if ty == GgmlType::Q8_0 {
                    DType::Q8_0
                } else {
                    DType::Q4_0
                },
                "{ty:?}"
            );
            for (i, (&actual, expected)) in got.iter().zip(tensor.to_f32().unwrap()).enumerate() {
                assert!(
                    (actual - expected).abs() < tolerance,
                    "{ty:?} element {i}: meganeura {actual} vs GGML {expected}"
                );
            }
        }
    }

    #[test]
    fn to_packed_rejects_a_short_payload() {
        for tensor in [
            GgufTensor::new(vec![256, 1], GgmlType::Q4K, vec![0u8; 10]),
            GgufTensor::new(vec![32, 1], GgmlType::Q4_0, vec![0u8; 4]),
        ] {
            assert!(
                matches!(tensor.to_packed(), Err(GgufError::BadShape(_))),
                "short {:?} payload was accepted",
                tensor.ggml_type
            );
        }

        let overflow = GgufTensor::new(vec![usize::MAX - 31, 2], GgmlType::Q4_0, Vec::new());
        assert!(matches!(
            overflow.to_packed(),
            Err(GgufError::BadShape(ref e)) if e.contains("overflow")
        ));
    }

    /// GGML splits a Q4_0 block's nibbles across halves: byte `j` holds
    /// element `j` and element `j + 16`, never `j` and `j + 1`.
    ///
    /// This pins the reference dequantizer rather than a repack, because
    /// Q4_0 is now read natively and it is this reference the shader is
    /// compared against — so a mistake here would move the target instead
    /// of failing the comparison.
    #[test]
    fn q4_0_reference_dequant_preserves_the_split_half_nibble_order() {
        // Distinct value per element pair, so any reordering shows up.
        let nibbles: [u8; 32] = std::array::from_fn(|i| (i / 2) as u8);
        let bytes = Builder::new()
            .tensor("w", &[32, 1], GgmlType::Q4_0, &q4_0_block(1.0, nibbles))
            .build();
        let m = load_gguf_bytes(&bytes).unwrap();
        let got = m.tensors["w"].to_f32().unwrap();
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
    /// whose values are worked out by hand from `ggml-quants.c` rather than
    /// from this crate's own reader.
    ///
    /// Covers the parts the one-element version missed: a `j >= 4` sub-block,
    /// whose 6-bit scale and min are split across two bytes with their high
    /// two bits borrowed from the `j < 4` entries, and both nibble halves of
    /// a 64-element span.
    #[test]
    fn q4_k_dequantizes_against_hand_computed_values() {
        // Per-sub-block (scale, min), 6-bit each. Sub-block 5 deliberately
        // uses a scale above 15 so the borrowed high bits matter.
        let sc = [1u8, 2, 0, 0, 5, 35, 0, 0];
        let mn = [0u8, 1, 0, 0, 4, 20, 0, 0];

        let mut block = vec![0u8; 144];
        block[0..2].copy_from_slice(&f16_to_bits(1.0).to_le_bytes()); // d
        block[2..4].copy_from_slice(&f16_to_bits(1.0).to_le_bytes()); // dmin

        // Inverse of `get_scale_min_k4`, written from the C rather than
        // from `q4k_scale_min`.
        let scales = &mut block[4..16];
        for j in 0..4 {
            scales[j] = sc[j] & 63;
            scales[j + 4] = mn[j] & 63;
        }
        for j in 4..8 {
            scales[j + 4] = (sc[j] & 0x0F) | ((mn[j] & 0x0F) << 4);
            scales[j - 4] |= (sc[j] >> 4) << 6;
            scales[j] |= (mn[j] >> 4) << 6;
        }

        // qs[0] feeds elements 0 (low) and 32 (high) of the first span;
        // qs[64] feeds elements 128 (low) and 160 (high) of the third.
        block[16] = 5 | (6 << 4);
        block[16 + 64] = 7 << 4;

        let bytes = Builder::new()
            .tensor("w", &[256, 1], GgmlType::Q4K, &block)
            .build();
        let m = load_gguf_bytes(&bytes).unwrap();
        let f = m.tensors["w"].to_f32().unwrap();

        // value = d * sc_j * q - dmin * m_j, with d = dmin = 1.
        for (idx, want) in [
            (0usize, 1.0 * 5.0 - 0.0), // sub-block 0, low nibble
            (32, 2.0 * 6.0 - 1.0),     // sub-block 1, high nibble
            (128, 5.0 * 0.0 - 4.0),    // sub-block 4, low nibble
            (160, 35.0 * 7.0 - 20.0),  // sub-block 5, high nibble
        ] {
            assert!(
                (f[idx] - want).abs() < 1e-3,
                "element {idx}: got {}, want {want}",
                f[idx]
            );
        }
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

    /// A Q4_1 block differs from Q4_0 only by carrying its own `m`, so the
    /// repack keeps both halves of the header rather than deriving one.
    #[test]
    fn q4_1_repack_keeps_the_stored_minimum() {
        let (d, m) = (0.5f32, -3.0f32);
        let nibbles: [u8; 32] = std::array::from_fn(|i| (i % 16) as u8);
        let mut block = Vec::with_capacity(20);
        block.extend_from_slice(&f16_to_bits(d).to_le_bytes());
        block.extend_from_slice(&f16_to_bits(m).to_le_bytes());
        for j in 0..16 {
            block.push(nibbles[j] | (nibbles[j + 16] << 4));
        }
        let bytes = Builder::new()
            .tensor("w", &[32, 1], GgmlType::Q4_1, &block)
            .build();
        let mm = load_gguf_bytes(&bytes).unwrap();
        let (dtype, packed) = mm.tensors["w"].to_packed().unwrap();
        assert_eq!(dtype, DType::Q4_0);
        let got = crate::runtime::dequantize_q4_0(&packed, 32, 1);
        for (e, &n) in nibbles.iter().enumerate() {
            let want = f32::from(n) * d + m;
            assert!(
                (got[e] - want).abs() < 1e-2,
                "element {e}: got {}, want {want}",
                got[e]
            );
        }
    }

    /// A file holding one type this loader does not implement must still
    /// yield an inventory for the rest of it — a Q2_K mix should list even
    /// though its Q2_K tensors cannot be read.
    #[test]
    fn unknown_tensor_types_are_listed_not_fatal() {
        // 10 is Q2_K, which this loader does not implement.
        let bytes = Builder::new()
            .tensor("known", &[32, 1], GgmlType::Q8_0, &q8_0_block(1.0, [0; 32]))
            .tensor("unknown", &[256, 1], GgmlType::Other(10), &[])
            .build();
        let m = load_gguf_bytes(&bytes).unwrap();
        assert_eq!(m.tensors.len(), 2, "both tensors must be listed");

        let known = &m.tensors["known"];
        assert!(known.to_packed().is_ok(), "the readable one still reads");

        let unknown = &m.tensors["unknown"];
        assert_eq!(unknown.dims, vec![256, 1], "shape survives");
        assert_eq!(unknown.ggml_type, GgmlType::Other(10));
        assert!(!unknown.ggml_type.is_supported());
        // Its length depends on a block size we do not know, so no bytes.
        assert!(unknown.data().is_empty());
        assert!(matches!(
            unknown.to_packed(),
            Err(GgufError::UnsupportedType(10))
        ));
        assert!(matches!(
            unknown.to_f32(),
            Err(GgufError::UnsupportedType(10))
        ));
    }

    /// Two tensors under one name would silently shadow each other in the
    /// map, and the loser would never be uploaded.
    #[test]
    fn duplicate_tensor_names_are_rejected() {
        let bytes = Builder::new()
            .tensor("w", &[32, 1], GgmlType::Q8_0, &q8_0_block(1.0, [0; 32]))
            .tensor("w", &[32, 1], GgmlType::Q8_0, &q8_0_block(2.0, [1; 32]))
            .build();
        assert!(matches!(
            load_gguf_bytes(&bytes),
            Err(GgufError::BadShape(_))
        ));
    }

    /// Counts in the header are file-controlled. Reserving on them
    /// directly lets a 24-byte file ask for a `usize::MAX` allocation,
    /// which aborts on capacity overflow instead of returning an error.
    #[test]
    fn rejects_absurd_declared_counts() {
        let mut header = Vec::new();
        header.extend_from_slice(b"GGUF");
        header.extend_from_slice(&3u32.to_le_bytes());
        header.extend_from_slice(&u64::MAX.to_le_bytes()); // tensor_count
        header.extend_from_slice(&0u64.to_le_bytes()); // kv_count
        assert!(matches!(
            load_gguf_bytes(&header),
            Err(GgufError::BadHeader(_))
        ));

        let mut kv = Vec::new();
        kv.extend_from_slice(b"GGUF");
        kv.extend_from_slice(&3u32.to_le_bytes());
        kv.extend_from_slice(&0u64.to_le_bytes());
        kv.extend_from_slice(&u64::MAX.to_le_bytes()); // kv_count
        assert!(matches!(load_gguf_bytes(&kv), Err(GgufError::BadHeader(_))));
    }

    /// Individually representable dimensions can still overflow their
    /// product, which would panic or wrap rather than report a bad shape.
    #[test]
    fn rejects_overflowing_shape_product() {
        let huge = 1u64 << 40;
        let mut infos = Vec::new();
        Builder::str_bytes(&mut infos, "w");
        infos.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        infos.extend_from_slice(&huge.to_le_bytes());
        infos.extend_from_slice(&huge.to_le_bytes());
        infos.extend_from_slice(&GgmlType::F32.tag().to_le_bytes());
        infos.extend_from_slice(&0u64.to_le_bytes()); // offset

        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&1u64.to_le_bytes()); // one tensor
        out.extend_from_slice(&0u64.to_le_bytes()); // no metadata
        out.extend_from_slice(&infos);
        while !out.len().is_multiple_of(32) {
            out.push(0);
        }
        assert!(matches!(load_gguf_bytes(&out), Err(GgufError::BadShape(_))));
    }

    /// Tensors index one shared buffer, and the K-quants hand it straight
    /// back. Loading a model should not cost a second copy of every
    /// payload on top of the file.
    #[test]
    fn k_quant_packing_borrows_the_file() {
        let block = vec![7u8; 144];
        let bytes = Builder::new()
            .tensor("w", &[256, 1], GgmlType::Q4K, &block)
            .build();
        let file: Arc<[u8]> = Arc::from(bytes);
        let m = load_gguf_shared(Arc::clone(&file)).unwrap();
        let t = &m.tensors["w"];

        // The tensor's bytes are a slice of the file, not a copy.
        let (_, packed) = t.to_packed().unwrap();
        assert!(
            matches!(packed, Cow::Borrowed(_)),
            "Q4_K should hand back the file's bytes, not clone them"
        );
        assert!(
            std::ptr::eq(packed.as_ptr(), t.data().as_ptr()),
            "the borrow should point into the shared buffer"
        );
        assert_eq!(&*packed, &block[..]);
    }

    /// GGML Q4_0 is read natively, not rebuilt into Meganeura's wider Q4.
    ///
    /// This is the load path a `Q4_0` GGUF actually takes, and it used to
    /// allocate and rewrite every block; 18 bytes in, 18 bytes out, borrowed
    /// from the file when the block count leaves a whole number of words.
    /// Q4_1 still repacks, because Meganeura's Q4 *is* the Q4_1 shape.
    #[test]
    fn ggml_q4_0_is_read_natively_and_q4_1_still_repacks() {
        // Two blocks: 36 bytes, a whole number of words.
        let block: Vec<u8> = (0..18u8).collect();
        let mut payload = block.clone();
        payload.extend_from_slice(&block);
        let bytes = Builder::new()
            .tensor("w", &[32, 2], GgmlType::Q4_0, &payload)
            .build();
        let file: Arc<[u8]> = Arc::from(bytes);
        let m = load_gguf_shared(Arc::clone(&file)).unwrap();
        let t = &m.tensors["w"];
        let (dtype, packed) = t.to_packed().unwrap();
        assert_eq!(dtype, DType::Q40);
        assert_eq!(packed.len(), 36, "native Q4_0 is 18 bytes a block");
        assert!(
            matches!(packed, Cow::Borrowed(_)),
            "an aligned Q4_0 buffer needs no copy"
        );
        assert!(
            std::ptr::eq(packed.as_ptr(), t.data().as_ptr()),
            "the borrow should point into the shared buffer"
        );

        // One block: 18 bytes, two short of a word. Only the tail is added.
        let odd = Builder::new()
            .tensor("w", &[32, 1], GgmlType::Q4_0, &block)
            .build();
        let odd_model = load_gguf_bytes(&odd).unwrap();
        let (_, packed) = odd_model.tensors["w"].to_packed().unwrap();
        assert_eq!(packed.len(), 20);
        assert_eq!(&packed[..18], &block[..], "the block itself is untouched");
        assert_eq!(&packed[18..], &[0, 0]);

        // Q4_1 carries a per-block minimum, so it becomes Meganeura's Q4:
        // 20 bytes a block, rebuilt rather than borrowed.
        let q41 = Builder::new()
            .tensor("w", &[32, 1], GgmlType::Q4_1, &[0u8; 20])
            .build();
        let q41_model = load_gguf_bytes(&q41).unwrap();
        let (dtype, packed) = q41_model.tensors["w"].to_packed().unwrap();
        assert_eq!(dtype, DType::Q4_0);
        assert_eq!(packed.len(), 20);
        assert!(matches!(packed, Cow::Owned(_)), "Q4_1 repacks");
    }

    /// Q6_K only copies when its superblock count leaves the buffer short
    /// of a word, and even then the superblocks are untouched.
    #[test]
    fn q6_k_borrows_when_already_word_aligned() {
        // Two superblocks: 420 bytes, a whole number of words.
        let block = vec![3u8; 210];
        let mut payload = block.clone();
        payload.extend_from_slice(&block);
        let bytes = Builder::new()
            .tensor("w", &[256, 2], GgmlType::Q6K, &payload)
            .build();
        let m = load_gguf_bytes(&bytes).unwrap();
        let (_, packed) = m.tensors["w"].to_packed().unwrap();
        assert_eq!(packed.len(), 420);
        assert!(
            matches!(packed, Cow::Borrowed(_)),
            "an aligned Q6_K buffer needs no copy"
        );

        // One superblock: 210 bytes, two short of a word.
        let odd = Builder::new()
            .tensor("w", &[256, 1], GgmlType::Q6K, &block)
            .build();
        let m = load_gguf_bytes(&odd).unwrap();
        let (_, packed) = m.tensors["w"].to_packed().unwrap();
        assert!(matches!(packed, Cow::Owned(_)), "padding needs a copy");
        assert_eq!(packed.len(), 212);
        assert_eq!(&packed[..210], &block[..]);
    }

    /// Q5_K is Q4_K's nibble plus a `qh` bit worth 16, and the bit index is
    /// the sub-block number rather than a position within `qh`.
    ///
    /// Every asserted element has a nonzero scale, so a decoder that read
    /// the wrong `qh` bit cannot hide behind a zero. Sub-block 7 also uses
    /// a scale above 15, which exercises the `j >= 4` packing that borrows
    /// its high two bits from the `j < 4` entries.
    #[test]
    fn q5_k_dequantizes_against_hand_computed_values() {
        let mut sc = [0u8; 8];
        let mut mn = [0u8; 8];
        sc[0] = 2;
        mn[0] = 1;
        sc[1] = 2;
        mn[1] = 1;
        sc[7] = 35;
        mn[7] = 20;

        let mut block = vec![0u8; 176];
        block[0..2].copy_from_slice(&f16_to_bits(1.0).to_le_bytes()); // d
        block[2..4].copy_from_slice(&f16_to_bits(1.0).to_le_bytes()); // dmin

        // Inverse of `get_scale_min_k4`, written from the C.
        let scales = &mut block[4..16];
        for j in 0..4 {
            scales[j] = sc[j] & 63;
            scales[j + 4] = mn[j] & 63;
        }
        for j in 4..8 {
            scales[j + 4] = (sc[j] & 0x0F) | ((mn[j] & 0x0F) << 4);
            scales[j - 4] |= (sc[j] >> 4) << 6;
            scales[j] |= (mn[j] >> 4) << 6;
        }

        // qh[0] carries one bit per sub-block: bit 0 set (element 0), bit 1
        // clear (element 32), bit 7 set (element 224).
        block[16] = 0b1000_0001;
        // qs[0] feeds elements 0 (low) and 32 (high) of the first span.
        block[48] = 3 | (5 << 4);
        // qs[96] feeds elements 192 (low) and 224 (high) of the fourth.
        block[48 + 96] = 6 << 4;

        let bytes = Builder::new()
            .tensor("w", &[256, 1], GgmlType::Q5K, &block)
            .build();
        let m = load_gguf_bytes(&bytes).unwrap();
        let f = m.tensors["w"].to_f32().unwrap();

        // d = dmin = 1, so value = sc_j * (nibble + 16*bit) - m_j.
        for (idx, want) in [
            // sub-block 0: nibble 3, bit set   -> 2*19 - 1
            (0usize, 37.0f32),
            // sub-block 1: nibble 5, bit clear -> 2*5 - 1. A decoder stuck
            // on bit 0 would read 21 here and give 41.
            (32, 9.0),
            // sub-block 7: nibble 6, bit set   -> 35*22 - 20
            (224, 750.0),
        ] {
            assert!(
                (f[idx] - want).abs() < 1e-3,
                "element {idx}: got {}, want {want}",
                f[idx]
            );
        }
    }

    /// Q3_K's high bit is inverted — a *clear* `hmask` bit subtracts 4 —
    /// and its scales use their own shuffle rather than
    /// `get_scale_min_k4`. Both are pinned here with hand-computed values,
    /// including a `g >= 2` scale group whose nibble comes from the high
    /// half of a stored byte.
    #[test]
    fn q3_k_dequantizes_against_hand_computed_values() {
        let mut block = vec![0u8; 110];
        block[108..110].copy_from_slice(&f16_to_bits(1.0).to_le_bytes());

        // scales[] holds sixteen 6-bit values. Index 0 takes the low
        // nibble of byte 0 plus bits 0-1 of byte 8; index 8 takes the
        // *high* nibble of byte 0 plus bits 4-5 of byte 8.
        let scales = &mut block[96..108];
        scales[0] = 0x04 | (0x03 << 4); // scale 0 nibble 4, scale 8 nibble 3
        scales[8] = 0b0001_0001; // scale 0 hi bits = 1, scale 8 hi bits = 1
        // scale 0  = 4 | (1 << 4) = 20 -> 20 - 32 = -12
        // scale 8  = 3 | (1 << 4) = 19 -> 19 - 32 = -13

        // Element 0: half 0, j 0, sub 0, l 0 -> qs[0] bits 0-1, hmask[0] bit 0.
        block[32] = 0b11; // q = 3
        block[0] = 0b0000_0001; // hmask bit 0 set -> no -4
        // Element 128: half 1, j 0, sub 0, l 0 -> qs[32] bits 0-1,
        // hmask[0] bit 4. Leave that bit clear so the -4 applies.
        block[64] = 0b10; // q = 2, bit 4 of hmask[0] is clear -> 2 - 4 = -2

        let bytes = Builder::new()
            .tensor("w", &[256, 1], GgmlType::Q3K, &block)
            .build();
        let m = load_gguf_bytes(&bytes).unwrap();
        let f = m.tensors["w"].to_f32().unwrap();

        // element 0:   d=1, scale 0 = -12, q = 3 (bit set) -> -36
        assert!((f[0] + 36.0).abs() < 1e-3, "element 0: got {}", f[0]);
        // element 128: d=1, scale 8 = -13, q = 2 - 4 = -2 -> 26
        assert!((f[128] - 26.0).abs() < 1e-3, "element 128: got {}", f[128]);
    }

    /// Q5_K needs no tail; Q3_K's 110-byte superblocks do.
    #[test]
    fn new_k_quants_pack_verbatim() {
        let q5: Vec<u8> = (0..176).map(|i| (i * 7 % 251) as u8).collect();
        let bytes = Builder::new()
            .tensor("w", &[256, 1], GgmlType::Q5K, &q5)
            .build();
        let m = load_gguf_bytes(&bytes).unwrap();
        let (dtype, packed) = m.tensors["w"].to_packed().unwrap();
        assert_eq!(dtype, DType::Q5K);
        assert!(
            matches!(packed, Cow::Borrowed(_)),
            "176 is a whole word count"
        );
        assert_eq!(&*packed, &q5[..]);
        assert_eq!(
            crate::graph::TensorType::new(vec![256, 1], DType::Q5K).size_bytes(),
            packed.len()
        );

        let q3: Vec<u8> = (0..110).map(|i| (i * 5 % 253) as u8).collect();
        let bytes = Builder::new()
            .tensor("w", &[256, 1], GgmlType::Q3K, &q3)
            .build();
        let m = load_gguf_bytes(&bytes).unwrap();
        let (dtype, packed) = m.tensors["w"].to_packed().unwrap();
        assert_eq!(dtype, DType::Q3K);
        assert_eq!(packed.len(), 112, "110 rounds up to 28 words");
        assert_eq!(&packed[..110], &q3[..], "superblock must be verbatim");
        assert_eq!(&packed[110..], &[0, 0]);
        assert_eq!(
            crate::graph::TensorType::new(vec![256, 1], DType::Q3K).size_bytes(),
            packed.len()
        );
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

    /// Blocks run along K, so a tensor can hold whole blocks overall while
    /// still splitting one across columns. `to_packed` has to catch that.
    #[test]
    fn rejects_misaligned_reduction_extent() {
        // K = 16 is not a whole 32-block, though 16 x 2 elements is.
        let block = q8_0_block(1.0, [0; 32]);
        let bytes = Builder::new()
            .tensor("w", &[16, 2], GgmlType::Q8_0, &block)
            .build();
        let m = load_gguf_bytes(&bytes).unwrap();
        let err = m.tensors["w"].to_packed().unwrap_err();
        assert!(
            matches!(err, GgufError::BadShape(ref e) if e.contains("multiple of 32")),
            "expected a block-alignment error, got {err}"
        );

        // F32 is not a packed format at all, which is a different refusal.
        let f32_bytes = Builder::new()
            .tensor("w", &[16, 1], GgmlType::F32, &[0u8; 64])
            .build();
        let m = load_gguf_bytes(&f32_bytes).unwrap();
        assert!(matches!(
            m.tensors["w"].to_packed(),
            Err(GgufError::UnsupportedPack(GgmlType::F32))
        ));
    }
}
