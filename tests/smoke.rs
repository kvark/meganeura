//! Broad stack checks: compilation, execution, training state and diagnostics.
mod cache_inference;
mod checkpoint_validation;
mod debug_session;
mod gpu_smoke;
mod optimizer_memory;
mod provenance;
mod shared_winograd_checkpoint;
mod training_correctness;
mod vision_ops_smoke;
