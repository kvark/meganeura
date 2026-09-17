//! Focused regressions share one executable instead of linking the stack per file.
//! Model-graph children are behind the `models` feature; run the rest featureless.
mod adam_state_rw;
mod back_to_back_step;
mod cached_query_attention;
mod constant_dedup;
mod conv1x1_grad_residual;
mod conv_derivatives;
mod coop_conv_unaligned_k;
mod coop_matmul_skinny;
mod eager;
#[cfg(feature = "models")]
mod efficientnet_smoke;
mod exclusive_cumsum;
mod f16_embedding;
mod fusion_preserves_outputs;
mod gemv_parity;
mod grad_accumulate;
mod grad_clip;
mod grad_debug;
mod gradcheck;
mod gradcheck_vision;
mod gradcheck_vision_large;
mod horizontal_matmul;
mod inference_parity_large;
mod int_dot_gemv;
mod intermediate_output_aliasing;
mod laprop;
mod materialize;
mod matmul_param_order;
mod mha_head_dim32;
mod mixed_attention_widths;
mod multi_input_trainer;
mod scatter_add_atomic;
mod schedule_pointwise;
mod schedule_reduction;
#[cfg(all(feature = "hf-hub", feature = "models"))]
mod smollm2_correctness;
mod softplus_tail;
mod submission_chunks;
mod tune;
mod whisper_correctness;
