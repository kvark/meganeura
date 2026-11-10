# Final qualification coverage

All eleven GPUs pass five core models, two profiles and three phases:
**330/330 cells**. Seven GPUs provide performance samples; four are
correctness-only, with no published CPU/GPU timing ratio.

| Qualification device | Native path | Independent oracle | Cells |
|---|---|---|---:|
| Radeon 780M | Vulkan | CPU PyTorch f32 | 30/30 |
| Ryzen 9600X iGPU | Vulkan | CPU PyTorch f32 | 30/30 |
| Intel RPL-U | Vulkan | CPU PyTorch f32 | 30/30 |
| Quest 3S / Adreno 740 | Vulkan | CPU f64, full tensors | 30/30 |

These use one process per model/profile, not three performance replicates.
Android archives retain hashed manifests and full comparison receipts but
omit raw NPZ tensors; replay audits the receipts, not original arithmetic.
Configuration-specific GPU-reference availability does not turn CPU-oracle
qualification into a paired GPU performance measurement.

The RTX 3050 (Windows 11, driver 610.88) moved to the timed set. Its
November 10 archive repeats the collection at the same revisions with
compiled CUDA PyTorch as the reference: 30/30 native cells and 30 ratios.
The earlier CPU-reference archive is `nvidia-3050-orig.tgz`, outside the
manifest.

All 252 measured native cells, including 42 larger-model cells, qualify;
231 have complete qualified PyTorch comparisons. B570 360M now completes:
all six native cells and four ratios, including every accelerated phase.
One strict reference process fails gradient repeatability; strict training
and its later one-token phase have no aggregated ratio. RTX 5070 accelerated
360M passes all phases. No old result fills a missing current cell.

See [RESULTS.md](RESULTS.md) for the audit and retained failures, and
[main.tex](main.tex) for the distinct numerical gates used by the engines.
