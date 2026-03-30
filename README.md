# meganeura

[![CI](https://github.com/kvark/meganeura/actions/workflows/ci.yml/badge.svg)](https://github.com/kvark/meganeura/actions/workflows/ci.yml)
[![Docs](https://docs.rs/meganeura/badge.svg)](https://docs.rs/meganeura)
[![Crates.io](https://img.shields.io/crates/v/meganeura.svg?label=meganeura)](https://crates.io/crates/meganeura)

Meganeura - a cross-platform Neural Network training and inference library in Rust.

![logo](etc/logo.png)

## Why Meganeura?

- **Portable**. It's powered by [blade-graphics](https://github.com/kvark/blade/tree/main/blade-graphics) for accessing GPUs across the board: Linux, Windows, MacOS, even edge devices on iOS or Android. Not toasters though.
- **Fast**. More of a promise than reality at this point. It doesn't beat production-optimized CUDA or MLX stacks yet, but it is faster than ROCm on laptop APUs.
- **Lean**. It packs a bunch of kernels, but the real power comes from their auto-discovery. During the optimization pre-process, it explores the search space using [e-graph](https://egraphs-good.github.io/), similar to [Luminal](https://github.com/luminal-ai/luminal).

## Benchmarks

See [Infermark](https://kvark.github.io/infermark/) for a comprehensive comparison between different frameworks.

SmolVLA action expert training (chunk_size=50, vlm_seq_len=16, float32, random weights).
Full GQA (15/5 heads, head_dim=64), exact backward through all ops including fused MHA and RmsNorm:

| GPU | Framework | Compile | Forward | Backward |
|-----|-----------|---------|---------|----------|
| Radeon 890M (RADV) | Meganeura 3d34aad29c5c9151dfb59b2a3be073ac203c30af | 0 s | 14.2 ms | 36.4 ms |
| Radeon 890M (RADV) | PyTorch 2.10.0 ROCm | 7.30 s | 20.9 ms | 48.0 ms |
| GeForce RTX 5080 (590/Linux) | Meganeura 550bb6caf09c819f199084d2263794e14f683463 | 0 s | 6.1 ms | 35.1 ms |
| GeForce RTX 5080 (590/Linux) | PyTorch 2.11.0+cu128 | 3.41 s | 1.57 ms | 4.68 ms |
| GeForce RTX 3050 (566.36/Windows) | Meganeura 550bb6caf09c819f199084d2263794e14f683463 | 0 s | 11.2 ms | 53.3 ms |
| GeForce RTX 3050 (566.36/Windows) | PyTorch 2.11.0+cu128 | 0 s (unsupported) | 12.3 ms | 33.8 ms |
| Apple M3 | Meganeura db9ae11f28189307c4384d75b0030336828aece8 | 0s | 45.5 ms | 87.0 ms |
| Apple M3 | PyTorch 2.11.0 | 5.92s | 17.7 ms | 78.1 ms |

Gradients verified against PyTorch (CPU): 88/136 parameters pass strict threshold (cos_sim > 0.99, norm_err < 5%). Failures are in attention and layernorm weights of deeper layers where f32 precision differences compound; gradient magnitudes (norm_err) remain < 2% for all parameters.

Run `bash bench/compare.sh` to reproduce.

## System Requirements

It works on on anything with Vulkan, including LavaPipe, or MacOS devices.
Runs best when [cooperative matrix operations](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_cooperative_matrix.html) is hardware-accelerated for 8x8 tile math:
- **Vulkan**: GPU and driver supporting `VK_KHR_cooperative_matrix` (NVIDIA Volta+, AMD RDNA3+, Intel Arc)
- **Metal**: Apple GPU with simdgroup matrix support (Apple M1+)
