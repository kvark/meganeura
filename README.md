# meganeura

[![CI](https://github.com/kvark/meganeura/actions/workflows/ci.yml/badge.svg)](https://github.com/kvark/meganeura/actions/workflows/ci.yml)
[![Docs](https://docs.rs/meganeura/badge.svg)](https://docs.rs/meganeura)
[![Crates.io](https://img.shields.io/crates/v/meganeura.svg?label=meganeura)](https://crates.io/crates/meganeura)

Meganeura - a cross-platform Neural Network training and inference library in Rust.

![logo](etc/logo.png)

For GPU access, we use [blade-graphics](https://github.com/kvark/blade/tree/main/blade-graphics), which opens the doors to Linux, Windows, and MacOS systems. No vendor locking, althought expect lower performance than anything that targets NVidia/CUDA directly.

Instead of including the "batteries" - kernels for all kind of cases and hardware - we are going to explore the search space using [e-graph](https://egraphs-good.github.io/), similar to [Luminal](https://github.com/luminal-ai/luminal). To translate a given graph into hardware instructions, we are generating [Naga IR](https://docs.rs/naga/latest/naga/) directly, skipping the intermediate text.

## Benchmarks

SmolVLA action expert on AMD RDNA3 (chunk_size=50, vlm_seq_len=16, 10 denoise steps, float32):

| Metric | meganeura | PyTorch ROCm |
|---|---|---|
| Steps/second | 40.9 | 41.6 |
| ms / step | 24.5 | 24.1 |
| ms / full chunk | 245 | 241 |

SmolVLA action expert training (chunk_size=50, vlm_seq_len=16, 16 layers, float32, random weights, AMD RDNA3).
Single-head attention (head_dim=64) — GQA reshape not yet implemented in meganeura:

| Metric | meganeura | PyTorch ROCm |
|---|---|---|
| Forward avg | 22.8 ms | 18.8 ms |
| Forward median | 22.5 ms | 18.6 ms |
| Train step avg | 77.9 ms | 68.2 ms |
| Train step median | 78.2 ms | 68.2 ms |
| Approx backward | 55.1 ms | 49.4 ms |

Run `bash bench/compare.sh` to reproduce, or `bash bench/compare.sh --model smolvla_train` for training only.

## System Requirements

Matrix multiplication uses [cooperative matrix operations](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_cooperative_matrix.html) for hardware-accelerated 8x8 tile math. This requires one of:

- **Vulkan**: GPU and driver supporting `VK_KHR_cooperative_matrix` (NVIDIA Volta+, AMD RDNA3+, Intel Arc)
- **Metal**: Apple GPU with simdgroup matrix support (Apple M1+)
