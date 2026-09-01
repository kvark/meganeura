#![allow(
    clippy::match_like_matches_macro,
    clippy::redundant_pattern_matching,
    clippy::needless_lifetimes,
    clippy::new_without_default,
    clippy::single_match,
    clippy::too_many_arguments,
    clippy::collapsible_if,
    clippy::needless_range_loop
)]
#![warn(
    trivial_numeric_casts,
    unused_extern_crates,
    clippy::pattern_type_mismatch
)]

//! Meganeura: graph-optimized neural network framework on blade-graphics.
//!
//! Models are defined as declarative computation graphs, optimized with
//! greedy rewrites by default (or optional equality saturation via egglog),
//! and compiled to static GPU dispatch sequences — no manual CUDA-graphing
//! needed.

#[doc(hidden)]
pub mod autodiff;
#[doc(hidden)]
pub mod cache;
#[doc(hidden)]
pub mod codegen;
#[doc(hidden)]
pub mod compile;
pub mod config;
pub mod data;
pub mod eager;
pub mod graph;
pub mod load;
#[doc(hidden)]
pub mod memplan;
pub mod models;
pub mod nn;
pub mod optimize;
#[doc(hidden)]
pub mod outline;
pub mod profiler;
pub mod runtime;
#[doc(hidden)]
pub mod schedule;
pub mod train;

pub use codegen::{CoopCaps, coop_caps, set_coop_caps};
pub use compile::CompileOptions;
pub use compile::TuningKnobs;
pub use data::{DataLoader, MnistDataset};
pub use graph::{DType, Graph, NodeId, TensorType};
pub use load::nnef::{NnefError, NnefModel, load_nnef};
pub use load::onnx::{OnnxError, OnnxModel, load_onnx, load_onnx_bytes};
pub use optimize::{ExtractionCost, OptimizeConfig, OptimizeMode, OptimizeReport};
pub use runtime::{
    CoopPolicy, DebugStepReport, DeviceMemoryStats, DispatchAnomaly, ExternalBindError,
    ExternalSlot, GpuOptions, MemorySummary, ReadNodeError, RuntimePrefixInfo, Session,
    SessionOptions, TuneOutcome, init_gpu_context, init_gpu_context_with,
};
pub use train::{
    EpochStats, LossHistory, MetricCallback, Mode, Optimizer, SessionConfig, StepMetrics,
    TrainConfig, TrainHistory, Trainer, build, build_inference_session, build_session,
    build_session_unoptimized, compile_training_graph,
};
