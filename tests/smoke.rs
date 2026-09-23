//! Broad stack checks: compilation, execution, training state and diagnostics.
//! `training_correctness` is behind the `models` feature; the rest run featureless.
mod cache_inference;
mod checkpoint_validation;
mod debug_session;
mod gpu_smoke;
mod optimizer_memory;
mod provenance;
mod shared_winograd_checkpoint;
#[cfg(feature = "models")]
mod training_correctness;
