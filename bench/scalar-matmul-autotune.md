# Scalar matmul autotuning (2026-09-20)

Implementation: `1a872343cec90b3dfc75d64cdaef72e26b013f25`. Control:
`741b35aaa36aa5029b24c36779b84e3dd280f469`, including automatic submission
selection. Model, llama.cpp revision, hardware and reproduction command follow
the [matched GGUF diagnostic](gguf_latency.md). Neither the submitted paper nor
the frozen Inferena cohort changes.

The existing scalar generator now exposes its K-stage controls for F16-stored
weights as well as F32. The qualified per-class tuner searches tile sizes 32/64,
K stages 8/16/32 and contiguous/interleaved columns. F32 activations and
accumulation are retained; block-quantized layouts are unchanged. Native F32
cooperative candidates remain eligible where supported. No device/model table
or reduced-precision cooperative implementation is added.

Selections persist in the dispatch and pipeline key, including profiling and
state-preserving tuning swaps. Equivalent initial layouts are not probed twice.
The existing private-input qualification, scratch budget, paired noise guard
and soft deadline apply. Disabling tuning preserves the original defaults.
Every step still records fresh commands.

## Repeated whole-model comparison

Three fresh processes per arm and GPU, rotated engine order. One sequence,
128-token prefill, 32 cached decode steps, F32 K/V caches and full-logit host
copies. Three warmup sequences precede seven saved sequences per process.
CPU affinity is `0,2,4,6,8,10`; clocks are not fixed. Builds finish before
measurement, and GPUs run sequentially. Driver caches are not cleared.

Milliseconds, median of three process medians, measured after tuning:

| GPU / phase | Previous tuner | Layout search | llama.cpp |
| --- | ---: | ---: | ---: |
| RTX 5070 prefill | 7.280 | 7.270 | 7.116 |
| RTX 5070 decode/token | 1.417 | 1.439 | 1.371 |
| Arc B570 prefill | 20.024 | 14.178 | 15.651 |
| Arc B570 decode/token | 3.746 | 3.785 | 3.442 |

Intel prefill improves by 29.2%, with process medians 14.176-14.219 ms versus
19.892-20.055 ms. It is 9.4% faster than llama.cpp here. Every process selects
K16/tile64 for the wide projection and K8/tile32 for the four narrower classes,
all with contiguous columns. NVIDIA retains K32, using tile64 for the wide
projection and tile32 elsewhere. Its prefill medians span 7.231-7.294 ms versus
7.270-7.288 ms: no material change.

Decode search is unchanged. Automatic decode medians span 1.394-1.542 ms on
NVIDIA and 3.757-4.031 ms on Intel. Intel's first scheduling search keeps two
chunks rather than four after rejecting noisy comparisons; this run is retained.
The median decode gap to llama.cpp remains about 5%/10%. NVIDIA prefill remains
about 2% behind. This is interactive latency, not batched-serving throughput.

Every kernel search visits all seven eligible classes within the unchanged
30-second/session, 256 MiB budget. Prefill compares 78 candidates rather than
28, and all comparisons qualify. Median prefill-search time rises from
3.09 to 3.98 seconds on NVIDIA and 11.10 to 14.28 seconds on Intel. Median
whole preparation is 9.52/31.54 seconds versus 8.63/28.54 seconds; llama.cpp
takes 0.24/0.50 seconds. These are not cold-cache compiler measurements.
The separate submission search keeps its two-second/session budget.

All 18 saved logit sets are finite and preserve all 33 independent CPU-reference
next-token choices. Maximum per-row relative L2 for the new Meganeura runs is
0.00000954/0.000177; llama.cpp is 0.01121/0.01314. This is fixed-token numerical
evidence, not language-quality equivalence. No qualification tolerance changes.

All 449 active library tests, all-target/all-feature Clippy, the existing GPU
training-state swap check and six existing GPU tuning regressions pass on both
GPUs. The native-F32-only GPU test requires different hardware and is not run
on these cards. No new test executable. Known Naga Workgroup ArrayStride
validation errors remain.

The source-only pilot is preserved at `experiment/f16-scalar-knobs-2026-09-20`
(`4f2b4ff`). Its manual K32/K16/K8 sweep motivated the search: Intel preferred
smaller stages while NVIDIA preferred K32. The repeated automatic results above
are separate from that exploratory sweep. Raw results and binaries stay outside
Git; the normal tuned helper command reproduces the search without extra flags.
