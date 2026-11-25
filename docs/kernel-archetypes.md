# Kernel Archetypes

Instead of carrying hundreds of hand-written GPU shaders, meganeura generates specialized WGSL kernels from a small set of composable **schedule templates**.

## Pointwise

Arbitrary elementwise DAGs (`PointwiseDAG`) fused into a single dispatch. The e-graph optimizer chains ops like `relu -> neg -> silu` and collapses them automatically. Every unary and binary elementwise op lowers to it, including the tanh-form GELU and the gated units.

Each DAG node is an arithmetic op or activation; the schedule template walks the DAG and emits inline WGSL for the fused body. One thread per element, one dispatch for the entire chain.

Inputs are read at the current element, or through a **broadcast load** at `(i / divisor) % modulus` of the flat index. Per-channel NCHW biases (`AddPerChannel`) and per-column biases and scales (`BiasAdd`, `BiasMul`) lower this way, so they fuse with the activations around them. Broadcast DAGs stay out of reduction prologues and matmul epilogues, whose element positions are not flat indices. Before chain fusion, reads of uniform constant tensors (the `zeros[n]` and `scale[n]` autodiff builds for ReLU masks, mean gradients and row broadcasts) become literals, which frees their input slots under the three-input binding cap.

## Reduction

Per-row tree reduction with optional **prologue** (transform before reducing, e.g. `v*v` for sum-of-squares), **extra accumulators** (`extra_prologues`: additional integrands reduced over the same inputs in the same pass), and **epilogue** (per-element post-processing using the reduced scalars, e.g. `x * rsqrt(mean + eps) * weight` for RMSNorm). Per-row broadcast inputs, per-column inputs (weights/biases), and indexed gather streams are supported.

RMSNorm and Softmax compile to this template. LayerNorm does not: it needs the mean before the squared deviations, and the template performs one reduction per row. Two-pass reductions (like softmax = max-reduce then sum-exp-reduce) compose naturally.

Current boundary, still hand-written by design: GroupNorm (its epilogue needs per-channel loads indexed by a function of both row and column — a layout-aware input category), the norm backwards that reduce along the *column* axis (GradW/GradWB), and LogSoftmax.

## Matmul

Tiled matrix multiplication (`src/codegen/`, `src/kernels/matmul.rs`) with:
- **Epilogue fusion** (`MatMulEpilogue`) — fuse BiasAdd, ReLU, Silu, etc. into the store phase, eliminating a separate dispatch and an intermediate buffer write.
- **Prologue support** (`MatMulPrologue`) — fuse normalization (e.g. RMSNorm scale) into the A-tile staging phase.
- **K-split GEMV** — for batch-1 LM decode (M=1), switch to a GEMV kernel that splits the K dimension across threads with vec4-coalesced reads, quantized weight decoding (`Q4_0`, `Q8`, `Q3_K`..`Q6_K`), optional integer dot-product activations (`dot4I8Packed`), and optional folded input RMSNorm.
- **Cooperative matrix** — when `VK_KHR_cooperative_matrix` or Metal simdgroup matrices are available, dispatches hardware tensor-core multiply-accumulate across typed F16/BF16/F32 matrix shapes, subgroup-tiled and workgroup-tiled schedules, split-K, and direct/staged loads.

## Convolution

Implicit-GEMM 2D convolution (`src/kernels/conv.rs`) lowered across scalar tiles, subgroup-tiled cooperative kernels (`src/codegen/workgroup.rs`), legacy cooperative convolutions, BF16 paths, depthwise conv (`conv2d_dw`), and Winograd 3×3 transform + batched matmul (`src/shaders/winograd_*.wgsl`).

## Attention

Generated forward and backward FlashAttention kernels (`src/codegen/attention.rs`, `src/kernels/attention_grad.rs`) with online softmax, configurable KV-tiling, register-blocked and cooperative/product-matrix variants, and runtime mask selection:
- **Causal** — `kv_seq == 0` triggers `kv_len = pos + 1` at runtime.
- **Full** — `kv_seq == q_seq` for bidirectional attention (Whisper encoder, ViT).
- **Cross** — `kv_seq` set to actual KV sequence length.
- **Sliding window** — `window_size > 0` limits how far back each position attends.

Decode and specialized attention paths (cached KV attention, split-K cached block attention, chunked relative attention, biased attention) live in dedicated WGSL shaders in `src/shaders/` alongside the remaining hand-written layout/norm/optimizer kernels.

## Cost Model

The e-graph optimizer uses a `FusionCostModel` to pick the cheapest equivalent expression after equality saturation, preferring fused kernels that eliminate intermediate buffer writes. The cost of a kernel is proportional to its HBM traffic — fusing two ops that share an intermediate tensor saves one full read+write of that tensor's size.
