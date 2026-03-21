# meganeura

Meganeura - a cross-platform Neural Network training and inference library in Rust.

![logo](etc/logo.png)

For GPU access, we use [blade-graphics](https://github.com/kvark/blade/tree/main/blade-graphics), which opens the doors to Linux, Windows, and MacOS systems. No vendor locking, althought expect lower performance than anything that targets NVidia/CUDA directly.

Instead of including the "batteries" - kernels for all kind of cases and hardware - we are going to explore the search space using [e-graph](https://egraphs-good.github.io/), similar to [Luminal](https://github.com/luminal-ai/luminal). To translate a given graph into hardware instructions, we are generating [Naga IR](https://docs.rs/naga/latest/naga/) directly, skipping the intermediate text.

## System Requirements

Matrix multiplication uses [cooperative matrix operations](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_cooperative_matrix.html) for hardware-accelerated 8x8 tile math. This requires one of:

- **Vulkan**: GPU and driver supporting `VK_KHR_cooperative_matrix` (NVIDIA Volta+, AMD RDNA3+, Intel Arc)
- **Metal**: Apple GPU with simdgroup matrix support (Apple M1+)
