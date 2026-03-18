#![allow(
    clippy::match_like_matches_macro,
    clippy::redundant_pattern_matching,
    clippy::needless_lifetimes,
    clippy::new_without_default,
    clippy::single_match,
)]
#![warn(
    trivial_numeric_casts,
    unused_extern_crates,
    clippy::pattern_type_mismatch,
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
pub mod graph;
pub mod optimize;
pub mod runtime;
pub mod train;

pub use graph::{DType, Graph, NodeId, TensorType};
pub use optimize::OptimizeReport;
pub use runtime::Session;
pub use train::{build_session, build_session_cached, build_session_with_report, compile_training_graph, TrainConfig};
