#![allow(
    clippy::match_like_matches_macro,
    clippy::redundant_pattern_matching,
    clippy::needless_lifetimes,
    clippy::new_without_default,
    clippy::single_match,
    clippy::too_many_arguments,
    clippy::collapsible_if
)]
#![warn(
    trivial_numeric_casts,
    unused_extern_crates,
    clippy::pattern_type_mismatch
)]

//! Meganeura: E-graph optimized neural network framework on blade-graphics.
//!
//! Models are defined as declarative computation graphs, optimized via
//! equality saturation (egglog), and compiled to static GPU dispatch
//! sequences — no manual CUDA-graphing needed.

pub mod autodiff;
pub mod cache;
pub mod codegen;
pub mod compile;
pub mod data;
pub mod graph;
pub mod load;
pub mod models;
pub mod nn;
pub mod optimize;
pub mod profiler;
pub mod runtime;
pub mod schedule;
pub mod train;

pub use data::{DataLoader, MnistDataset};
pub use graph::{DType, Graph, NodeId, TensorType};
pub use load::nnef::{NnefError, NnefModel, load_nnef};
pub use load::onnx::{OnnxError, OnnxModel, load_onnx, load_onnx_bytes};
pub use optimize::OptimizeReport;
pub use runtime::{MemorySummary, Session};
pub use train::{
    EpochStats, LossHistory, MetricCallback, Optimizer, StepMetrics, TrainConfig, TrainHistory,
    Trainer, build_inference_session, build_session, build_session_cached,
    build_session_unoptimized, build_session_with_report, compile_training_graph,
};
