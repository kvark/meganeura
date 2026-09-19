# RMSNorm subgroup experiment

Based on `84da75c5f07421ca6aa6eabb919c2f718a868d74`. Reuse the existing GEMV
subgroup reduction for its fused sum-of-squares prologue as well as its matrix
sum. The existing tuner and qualification thresholds are unchanged.

Use `gguf_latency.md`'s matched official F16 model, settings and llama.cpp pin.
Three fresh-process pairs per GPU, reversing order; milliseconds, median of
process medians:

| GPU | Phase | Control | Both subgroup reductions |
| --- | --- | ---: | ---: |
| RTX 5070 | Prefill | 7.365 | 7.219 |
| RTX 5070 | Decode | 1.814 | 1.804 |
| Arc B570 | Prefill | 21.200 | 21.206 |
| Arc B570 | Decode | 4.541 | 4.519 |

All 33 next-token choices agree with the independent CPU reference in every
run. Maximum per-row relative L2 errors are below 9.5e-6 and 1.77e-4 respectively.
The eight existing GEMV unit/code-generation tests pass.

Decode gains are below 1% and smaller than process-to-process variation;
NVIDIA experimental medians span 1.68--1.83 ms and Intel 4.49--4.58 ms.
This does not justify replacing the existing candidate in production.
Raw results and binaries are not tracked.
