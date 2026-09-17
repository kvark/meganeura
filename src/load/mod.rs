//! Model loading from standard interchange formats.
//!
//! [`nnef`] and [`onnx`] import a computation graph into Meganeura's
//! `Graph` IR, which then flows through the normal pipeline:
//! `Graph -> optimize -> compile -> ExecutionPlan -> Session`.
//!
//! `gguf` is also different in kind: GGUF is a weight-and-metadata
//! container with no graph in it, so that module reads the architecture
//! description, builds the graph it implies, fills it from the file's own
//! tensors, and exposes a `Generator` that owns the sessions and the
//! tokenizer. It is behind the `gguf` cargo feature.

#[cfg(feature = "gguf")]
pub mod gguf;
pub mod nnef;
pub mod onnx;
