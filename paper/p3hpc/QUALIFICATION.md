# Final qualification coverage

All eleven GPUs pass all five core models, both profiles, and three phases:
330/330 cells. Seven GPUs provide performance samples; the other four are
correctness-only and have no published timing or CPU/GPU ratio.

| Qualification device | Native path | Independent oracle | Cells |
|---|---|---|---:|
| Radeon 780M | Vulkan | CPU PyTorch f32 | 30/30 |
| Ryzen 9600X iGPU | Vulkan | CPU PyTorch f32 | 30/30 |
| Intel RPL-U | Vulkan | CPU PyTorch f32 | 30/30 |
| Quest 3S / Adreno 740 | Vulkan | CPU f64, full tensors | 30/30 |

Each uses one process per model/profile, not three performance replicates.
Android captures are represented by hashed manifests and full comparison
receipts; raw NPZ tensors are absent from the archives. The offline audit
checks those receipts, not the original tensor arithmetic.

Earlier ROCm failures on the AMD iGPUs and the unavailable XPU device on RPL-U
are configuration-specific bring-up observations. They do not turn CPU-oracle
qualifications into fresh paired GPU measurements.

All 246 completed measured native cells (core plus large models) qualify;
226 have a complete qualified PyTorch comparison. The B570 360M campaign is
incomplete and contributes no independently qualified cells. Its native
internal checks pass, but the reference fails during XPU graph capture and
the harness aborts. See [RESULTS.md](RESULTS.md) for failure attribution and
[main.tex](main.tex) for the distinct numerical gates used by the engines.
