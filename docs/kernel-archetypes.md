# Kernel Archetypes

Instead of carrying hundreds of hand-written GPU shaders, meganeura generates specialized WGSL kernels from a small set of composable **schedule templates**.

## Pointwise

Arbitrary elementwise DAGs (`PointwiseDAG`) fused into a single dispatch. The e-graph optimizer chains ops like `relu -> neg -> silu` and collapses them automatically. Replaces what would otherwise be `unary.wgsl`, `binary.wgsl`, and dozens of one-off elementwise shaders.

Each DAG node is an arithmetic op or activation; the schedule template walks the DAG and emits inline WGSL for the fused body. One thread per element, one dispatch for the entire chain.

## Reduction

Per-row tree reduction with optional **prologue** (transform before reducing, e.g. `v*v` for sum-of-squares) and **epilogue** (per-element post-processing using the reduced scalar, e.g. `x * rsqrt(mean + eps) * weight` for RMSNorm).

This single template replaces hand-written `rms_norm.wgsl`, `softmax.wgsl`, `layer_norm.wgsl`, and similar kernels. Two-pass reductions (like softmax = max-reduce then sum-exp-reduce) compose naturally.

## Matmul

Tiled matrix multiplication with:
- **Epilogue fusion** (`MatMulEpilogue`) — fuse BiasAdd, ReLU, Silu, etc. into the store phase, eliminating a separate dispatch and an intermediate buffer write.
- **Prologue support** (`MatMulPrologue`) — fuse normalization (e.g. RMSNorm scale) into the A-tile staging phase.
- **K-split GEMV** — for batch-1 LM decode (M=1), switch to a GEMV kernel that splits the K dimension across threads with vec4-coalesced reads. ~2.5x faster than the tiled path for single-token generation.
- **Cooperative matrix** — when `VK_KHR_cooperative_matrix` is available, stages tiles through shared memory in f16 and dispatches hardware tensor-core multiply-accumulate. Includes vec4 staging for both normal and transposed layouts.

## Attention

Unified forward attention with online softmax, BKV=8 KV-tiling, and runtime mask selection:
- **Causal** — `kv_seq == 0` triggers `kv_len = pos + 1` at runtime.
- **Full** — `kv_seq == q_seq` for bidirectional attention (Whisper encoder, ViT).
- **Cross** — `kv_seq` set to actual KV sequence length.
- **Sliding window** — `window_size > 0` limits how far back each position attends.

Parameterized by `head_dim` (determines workgroup size). A single generated shader handles all 6 forward attention variants (CausalAttention, CausalAttentionRoPE, FullAttention, CrossAttention, MultiHeadAttn, SlidingWindowAttention). Backward pass uses separate hand-written shaders with fused GradKV dispatch.

## Cost Model

The e-graph optimizer uses a `FusionCostModel` to pick the cheapest equivalent expression after equality saturation, preferring fused kernels that eliminate intermediate buffer writes. The cost of a kernel is proportional to its HBM traffic — fusing two ops that share an intermediate tensor saves one full read+write of that tensor's size.
