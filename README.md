# meganeura

[![CI](https://github.com/kvark/meganeura/actions/workflows/ci.yml/badge.svg)](https://github.com/kvark/meganeura/actions/workflows/ci.yml)
[![Docs](https://docs.rs/meganeura/badge.svg)](https://docs.rs/meganeura)
[![Crates.io](https://img.shields.io/crates/v/meganeura.svg?label=meganeura)](https://crates.io/crates/meganeura)

Meganeura - a cross-platform Neural Network training and inference library in Rust.

![logo](etc/logo.png)

For GPU access, we use [blade-graphics](https://github.com/kvark/blade/tree/main/blade-graphics), which opens the doors to Linux, Windows, and MacOS systems. No vendor locking, althought expect lower performance than anything that targets NVidia/CUDA directly.

Instead of including the "batteries" - kernels for all kind of cases and hardware - we are going to explore the search space using [e-graph](https://egraphs-good.github.io/), similar to [Luminal](https://github.com/luminal-ai/luminal).

## Benchmarks

SmolVLA action expert training (chunk_size=50, vlm_seq_len=16, float32, random weights).
Full GQA (15/5 heads, head_dim=64), exact backward through all ops including fused MHA and RmsNorm:

| GPU | Framework | Rev | Forward | Backward | Train step |
|---|---|---|---|---|---|
| Radeon 890M | meganeura | `af2b956` | **17.0 ms** | 163.1 ms | 180.1 ms |
| Radeon 890M | PyTorch 2.5 ROCm | — | 28.4 ms | 68.8 ms | 97.3 ms |
| Radeon 780M | meganeura | `637c4b1` | **14.5 ms** | 82.7 ms | **97.2 ms** |
| Radeon 780M | PyTorch ROCm | — | ✗ | ✗ | ✗ |

SmolVLA action expert inference (chunk_size=50, vlm_seq_len=16, 10 denoise steps, float32):

| GPU | Framework | Rev | ms / step | Steps/s |
|---|---|---|---|---|
| Radeon 890M | meganeura | `af2b956` | **18.1** | **55.1** |
| Radeon 890M | PyTorch 2.5 ROCm | — | 27.8 | 36.0 |
| Radeon 780M | meganeura | `637c4b1` | **14.5** | **69.0** |
| Radeon 780M | PyTorch ROCm | — | ✗ | ✗ |

PyTorch ROCm does not ship kernels for gfx1103 (780M). The 890M was tested with `HSA_OVERRIDE_GFX_VERSION`.

Gradients verified against PyTorch: 152/152 parameters pass (cos_sim > 0.99, norm_err < 5%) on the full production config.

Run `bash bench/compare.sh` to reproduce.

## System Requirements

It works on on anything with Vulkan, including LavaPipe, or MacOS devices.
Runs best when [cooperative matrix operations](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_cooperative_matrix.html) is hardware-accelerated for 8x8 tile math:
- **Vulkan**: GPU and driver supporting `VK_KHR_cooperative_matrix` (NVIDIA Volta+, AMD RDNA3+, Intel Arc)
- **Metal**: Apple GPU with simdgroup matrix support (Apple M1+)
