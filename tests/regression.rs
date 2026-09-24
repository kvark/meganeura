//! Focused regressions share one executable instead of linking the stack per file.
//! Model-graph children are behind the `models` feature; run the rest featureless.
mod adam_state_rw;
mod back_to_back_step;
mod block_matmul;
mod cached_query_attention;
mod constant_dedup;
mod conv_derivatives;
mod coop_conv_unaligned_k;
mod coop_matmul_skinny;
mod device_local;
mod eager;
#[cfg(feature = "models")]
mod efficientnet_smoke;
mod flash_grad_kv_short;
mod gemma_inference_ops;
mod gemv_parity;
mod grad_accumulate;
mod grad_clip;
mod horizontal_matmul;
mod int_dot_gemv;
mod laprop;
mod multi_input_trainer;
mod rmsnorm_matmul_prologue;
mod schedule_pointwise;
mod schedule_reduction;
#[cfg(feature = "models")]
mod sd_unet_coop_parity;
mod shader_audit;
mod shared_parameters;
#[cfg(all(feature = "hf-hub", feature = "models"))]
mod smollm2_correctness;
mod submission_chunks;
mod tune;
mod whisper_correctness;
