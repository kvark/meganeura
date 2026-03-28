# meganeura

[![CI](https://github.com/kvark/meganeura/actions/workflows/ci.yml/badge.svg)](https://github.com/kvark/meganeura/actions/workflows/ci.yml)
[![Docs](https://docs.rs/meganeura/badge.svg)](https://docs.rs/meganeura)
[![Crates.io](https://img.shields.io/crates/v/meganeura.svg?label=meganeura)](https://crates.io/crates/meganeura)

Meganeura - a cross-platform Neural Network training and inference library in Rust.

![logo](etc/logo.png)

For GPU access, we use [blade-graphics](https://github.com/kvark/blade/tree/main/blade-graphics), which opens the doors to Linux, Windows, and MacOS systems. No vendor locking, althought expect lower performance than anything that targets NVidia/CUDA directly.

Instead of including the "batteries" - kernels for all kind of cases and hardware - we are going to explore the search space using [e-graph](https://egraphs-good.github.io/), similar to [Luminal](https://github.com/luminal-ai/luminal). To translate a given graph into hardware instructions, we are generating [Naga IR](https://docs.rs/naga/latest/naga/) directly, skipping the intermediate text.

## Benchmarks

SmolVLA action expert inference on AMD RDNA3 (chunk_size=50, vlm_seq_len=16, 10 denoise steps, float32):

| Metric | meganeura | PyTorch ROCm |
|---|---|---|
| Avg latency (ms) | **181.4** | 277.5 |
| ms / step | **18.1** | 27.8 |
| Steps/second | **55.1** | 36.0 |

SmolVLA action expert training on AMD RDNA3 (chunk_size=50, vlm_seq_len=16, float32, random weights).
Full GQA (15/5 heads, head_dim=64), exact backward through all ops including fused MHA and RmsNorm:

| Metric | meganeura | PyTorch ROCm |
|---|---|---|
| Forward avg | **17.0 ms** | 28.4 ms |
| Train step avg | 180.1 ms | 97.3 ms |
| Approx backward | 163.1 ms | 68.8 ms |

Gradients verified against PyTorch: 152/152 parameters pass (cos_sim > 0.99, norm_err < 5%) on the full production config.

Run `bash bench/compare.sh` to reproduce (runs inference + training + SmolLM2 benchmarks by default).

## System Requirements

Matrix multiplication uses [cooperative matrix operations](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_cooperative_matrix.html) for hardware-accelerated 8x8 tile math. This requires one of:

- **Vulkan**: GPU and driver supporting `VK_KHR_cooperative_matrix` (NVIDIA Volta+, AMD RDNA3+, Intel Arc)
- **Metal**: Apple GPU with simdgroup matrix support (Apple M1+)
