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

| GPU | Framework | Compile | Forward | Backward |
|-----|-----------|---------|---------|----------|
| Radeon 890M (RADV) | Meganeura 550bb6caf09c819f199084d2263794e14f683463 | 0 s | 19.4 ms | 85.6 ms |
| Radeon 890M (RADV) | PyTorch 2.10.0 ROCm | 6.79 s | 21.14 ms | 54.3 ms |
| Radeon 780M (RADV) | Meganeura 550bb6caf09c819f199084d2263794e14f683463 | 0 s | 14.5 ms | 82.7 ms |
| Radeon 780M (RADV) | PyTorch 2.9.1 ROCm (eager) | :x: | :x: | :x: |
| GeForce RTX 5080 (590/Linux) | Meganeura 550bb6caf09c819f199084d2263794e14f683463 | 0 s | 6.1 ms | 35.1 ms |
| GeForce RTX 5080 (590/Linux) | PyTorch 2.11.0+cu128 | 3.41 s | 1.57 ms | 4.68 ms |
| GeForce RTX 3050 (566.36/Windows) | Meganeura 550bb6caf09c819f199084d2263794e14f683463 | 0 s | 11.2 ms | 53.3 ms |
| GeForce RTX 3050 (566.36/Windows) | PyTorch 2.11.0+cu128 | 0 s (unsupported) | 12.3 ms | 33.8 ms |
| Apple M3 | Meganeura 2bc4de61517375bfff01db859da586ece4da8124 | 0s | 27.6 ms | 176.9 ms |
| Apple M3 | PyTorch 2.11.0 | 7.54s | 10.0 ms | 30.2 ms |

PyTorch ROCm does not ship kernels for gfx1103 (780M). The 890M was tested with `HSA_OVERRIDE_GFX_VERSION`.

Gradients verified against PyTorch (CPU): 88/136 parameters pass strict threshold (cos_sim > 0.99, norm_err < 5%). Failures are in attention and layernorm weights of deeper layers where f32 precision differences compound; gradient magnitudes (norm_err) remain < 2% for all parameters.

Run `bash bench/compare.sh` to reproduce.

## System Requirements

It works on on anything with Vulkan, including LavaPipe, or MacOS devices.
Runs best when [cooperative matrix operations](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_cooperative_matrix.html) is hardware-accelerated for 8x8 tile math:
- **Vulkan**: GPU and driver supporting `VK_KHR_cooperative_matrix` (NVIDIA Volta+, AMD RDNA3+, Intel Arc)
- **Metal**: Apple GPU with simdgroup matrix support (Apple M1+)
