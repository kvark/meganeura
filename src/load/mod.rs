//! Model loading from standard interchange formats.
//!
//! [`nnef`] and [`onnx`] import a computation graph into Meganeura's
//! `Graph` IR, which then flows through the normal pipeline:
//! `Graph -> optimize -> compile -> ExecutionPlan -> Session`.
//!
//! [`gguf`] is different in kind: GGUF is a weight-and-metadata container
//! with no graph in it, so that module yields named tensors and
//! architecture metadata to pair with a builder in [`crate::models`], the
//! way SafeTensors loading does.

pub mod gguf;
pub mod nnef;
pub mod onnx;
