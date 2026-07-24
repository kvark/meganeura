//! Model loading from standard interchange formats.
//!
//! Each submodule imports a model from a specific format into Meganeura's
//! `Graph` IR, which then flows through the normal pipeline:
//! `Graph -> optimize -> compile -> ExecutionPlan -> Session`.

pub mod nnef;
pub mod onnx;
