# Qualified GEMV row grouping

Source: `a5a47c6e9abbdd2e8272d2ae3f15f5f9de1360c5`. The row-only
intermediate is `192eed7755c94f421fafbcebf60b56d2a379aed5`; control is
`69bbb48450fee8f9957b682ffec95733ca50c1f4`, the [native-row fusion work](gguf-row-weights.md).
Model, llama.cpp revision, hardware and commands follow the
[matched GGUF diagnostic](gguf_latency.md). All arms use ordinary command
recording. The submitted paper and frozen Inferena cohort are unchanged.

Dense transposed-B GEMV now measures 1/2/4 contiguous rows per workgroup,
crossed with 32/64/128/256 threads and tree/subgroup reduction. Vector
accumulators share input loads and folded RMSNorm work; plain and residual-add
forms use the same generator. Installing a selection also updates the dispatch
grid, including partial row groups and wide two-dimensional grids. Other
layouts keep their existing geometry. Old serialized shapes default to one row.

Subgroup reduction uses actual subgroup IDs/counts, not a lane-to-subgroup
mapping or an atomic slot allocator. Its single workgroup barrier is conditional
on having multiple subgroups. This does not remove the separate RMSNorm
prologue's barriers. The count is workgroup-uniform in the
[WGSL specification](https://www.w3.org/TR/2026/CRD-WGSL-20260915/#uniformity-inputs);
the pinned Naga already supports these built-ins for Vulkan and Metal.

## Repeated comparison

Milliseconds; median of three fresh-process medians with rotated engine order.
One sequence, 128-token prefill, 32 cached decode steps and full-logit host copies;
three warmups and seven measured sequences per process. Engines and GPUs ran
sequentially, without competing builds or profilers. CPU/GPU clocks were not fixed.

| GPU | Phase | Control | Row grouping | + subgroup IDs | llama.cpp |
| --- | --- | ---: | ---: | ---: | ---: |
| RTX 5070 | Prefill | 8.475 | 8.435 | 8.395 | 7.118 |
| RTX 5070 | Decode/token | 2.513 | 2.362 | 2.399 | 1.377 |
| Arc B570 | Prefill | 21.267 | 21.202 | 21.170 | 15.609 |
| Arc B570 | Decode/token | 5.195 | 4.770 | 4.708 | 3.437 |

The combined change improves decode in all three control pairs: 4.5%/9.4%
by the NVIDIA/Intel medians. Its process medians span 2.355–2.428 ms and
4.650–4.724 ms, versus 2.506–2.513 ms and 5.145–5.227 ms for the controls.
Row grouping accounts for most of the improvement. A separate NVIDIA gain
from the subgroup simplification is not established: one row-only run reached
1.635 ms, with a 0.370 ms recording stage versus about 1.0–1.1 ms otherwise,
while its wait stage remained near 1.2 ms. Do not attribute that outlier to
faster kernels or claim an unmeasured clock explanation.

Preparation across both sessions rises from 3.67 to 7.60 seconds on NVIDIA
and 10.70 to 25.44 seconds on Intel. These are process starts with existing
driver caches, not cache-cold compilation. The diagnostic retains its
30-second/session, 64-class, 256-MiB limits. All sessions visit all seven
eligible classes and qualify every comparison without exhausting the budget.
The larger search does not increase the library's default two-second limit.

All 24 saved logit sets are finite and retain all 33 CPU-reference next-token
choices. Maximum per-row relative L2 on the combined path is 0.00000954 on
NVIDIA and 0.000176 on Intel. This is a fixed-token numerical check, not a
language-quality comparison or a guarantee of identical intermediate arithmetic.

Decode remains 1.74x/1.37x llama.cpp; prefill is 1.18x/1.36x. Recording and
submission still take roughly 1.0/1.35 ms per decode on these CPUs. That is
comparable to the remaining whole-call gap and motivates CPU-side attribution.
These stages are not a GPU barrier-cost budget. Dispatch counts are unchanged.

452 CPU tests, the 17-test GEMV group and the 22-test GGUF suite pass, with GPU
checks on both devices. Existing tests cover every geometry, partial groups,
normalized tuning, packed Q4_0 reduction and a 262,145-output grid. All six CI
jobs pass, including Metal, Windows and Rust host coverage. No new test binary
or relaxed tolerance; the known Naga Workgroup ArrayStride warning remains.
Raw timings, logits and binaries stay outside Git.
