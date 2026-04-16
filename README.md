# meganeura

[![CI](https://github.com/kvark/meganeura/actions/workflows/ci.yml/badge.svg)](https://github.com/kvark/meganeura/actions/workflows/ci.yml)
[![Docs](https://docs.rs/meganeura/badge.svg)](https://docs.rs/meganeura)
[![Crates.io](https://img.shields.io/crates/v/meganeura.svg?label=meganeura)](https://crates.io/crates/meganeura)

Meganeura - a cross-platform Neural Network training and inference library in Rust.

![logo](etc/logo.png)

## Why Meganeura?

- **Portable**. It's powered by [blade-graphics](https://github.com/kvark/blade/tree/main/blade-graphics) for accessing GPUs across the board: Linux, Windows, MacOS, even edge devices on iOS or Android. Not toasters though.
- **Fast**. Within 2× of ROCm, and 2–5× of optimized CUDA/MLX depending on model (see [Inferena](https://kvark.github.io/inferena/) benchmarks). LM decode on NVIDIA is ~2.5× faster than the unoptimized baseline thanks to K-split GEMV kernels.
- **Lean**. It packs a handful of kernel archetypes (pointwise, reduction, matmul, attention) that compose into specialized GPU shaders at compile time. [E-graph](https://egraphs-good.github.io/) equality saturation discovers optimal kernel fusions — similar to [Luminal](https://github.com/luminal-ai/luminal) but with a cost-model-driven extractor.

## Benchmarks

See [Inferena](https://kvark.github.io/inferena/) for a comprehensive comparison between different frameworks.

SmolVLA action expert training (chunk_size=50, vlm_seq_len=16, float32, random weights).

| GPU | Framework | Compile | Forward | Backward |
|-----|-----------|---------|---------|----------|
| Radeon 890M (RADV) | Meganeura 3d34aad29c5c9151dfb59b2a3be073ac203c30af | 0 s | 14.2 ms | 36.4 ms |
| Radeon 890M (RADV) | PyTorch 2.10.0 ROCm | 7.30 s | 20.9 ms | 48.0 ms |
| GeForce RTX 5080 (590/Linux) | Meganeura 550bb6caf09c819f199084d2263794e14f683463 | 0 s | 6.1 ms | 35.1 ms |
| GeForce RTX 5080 (590/Linux) | PyTorch 2.11.0+cu128 | 3.41 s | 1.57 ms | 4.68 ms |
| Apple M3 | Meganeura 5ddf5e5c9b7b99ebb8d9d21e5c47110297ffeaa5 | 0s | 48.8 ms | 89.7 ms |
| Apple M3 | PyTorch 2.11.0 | 6.0s | 10.2 ms | 31.6 ms |

Full automatic differentiation through all ops including fused CausalAttention (with LSE-based backward), RoPE, Softmax, RmsNorm, and SwiGLU.
Gradient norms verified against PyTorch via [Inferena](https://kvark.github.io/inferena/) correctness checking.

Run `bash bench/compare.sh` to reproduce.

## Profiling

Examples accept `MEGANEURA_TRACE=<filename>` environment for saming binary Perfetto traces.
You can open them with [Perfetto Trace Viewer](https://ui.perfetto.dev/#!/viewer):
![perfetto trace](etc/example-trace.png)

## Kernel Archetypes

Instead of carrying hundreds of hand-written GPU shaders, meganeura generates specialized WGSL kernels from a small set of composable **schedule templates**:

- **Pointwise** — arbitrary elementwise DAGs (`PointwiseDAG`) fused into a single dispatch. Chains like `relu → neg → silu` collapse automatically. Replaces `unary.wgsl`, `binary.wgsl`.
- **Reduction** — per-row tree reduction with optional prologue (e.g., `v*v` for sum-of-squares) and epilogue (e.g., `x * rsqrt(mean+eps) * weight` for RMSNorm). Replaces `rms_norm.wgsl`, `softmax.wgsl`.
- **Matmul** — tiled matmul with epilogue fusion (`MatMulEpilogue`) and prologue support (`MatMulPrologue`) for fusing norm-scale into A-tile staging. Includes K-split GEMV variants for batch-1 LM decode.
- **Attention** — FlashAttention-1 forward with online softmax, parameterized by mask type (none/causal), head dimension, and GQA grouping. Replaces `mha_forward.wgsl`.

The e-graph optimizer uses a `FusionCostModel` to pick the cheapest equivalent expression after equality saturation, preferring fused kernels that eliminate intermediate buffer writes.

## System Requirements

It works on on anything with Vulkan, including LavaPipe, or MacOS devices.
Runs best when _cooperative matrix operations_ are hardware-accelerated:
- **Vulkan**: GPU and driver supporting [VK_KHR_cooperative_matrix](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_cooperative_matrix.html) (NVIDIA Volta+, AMD RDNA3+, Intel Arc)
- **Metal**: Apple GPU with simdgroup matrix support (Apple M1+)

Happy to rely on Mesa's [Lavapipe](https://www.phoronix.com/news/Lavapipe-CPU-Vulkan-Windows) for CI or local compatibility tests.

## Standard Model Loading

Meganeura can load models from standard interchange formats and run them through the normal pipeline (e-graph optimization → compile → GPU execution) without any Rust codegen:

- **ONNX** — via `load_onnx("model.onnx")`. Uses [oxionnx-proto](https://crates.io/crates/oxionnx-proto) for lightweight protobuf parsing. Supports Gemm, MatMul, activations, normalization, attention, convolution, and shape ops. Decomposed subgraphs (from `torch.onnx.export`) should be re-exported with compound ops preserved via `optimum-cli`.
- **NNEF** — via `load_nnef("model_dir/")`. Hand-rolled parser for the Khronos [NNEF](https://www.khronos.org/nnef/) text format and binary tensor files. Supports matmul, linear, convolution, activations, normalization, and reshape ops.

Both loaders translate the foreign graph into Meganeura's IR, where the e-graph optimizer discovers kernel fusions (e.g. `x * sigmoid(x)` → Silu, `Silu(gate) * up` → SwiGLU) before compilation.
