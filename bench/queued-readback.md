# Queued output readback (2026-09-20)

Implementation: `7c95707b9a69478c1bd498f940b7dde6e851a7e1`, on
`999bfadff710c67e0c200ca47474551c01dfc4cb`. Model, llama.cpp revision,
hardware and commands follow the [matched GGUF diagnostic](gguf_latency.md).
No change to the submitted paper or frozen Inferena cohort.

`Session::wait_read_output` queues a known staged download before the CPU waits
for graph completion. Both GPUs already select staging for the 196608-byte
logits. Unknown allocations and mapped reads still wait first. The existing
readback probe, bit checks and allocation policy are unchanged. All commands
are freshly recorded; this is not command-buffer replay.

## Isolating the ordering change

The source-only experiment `experiment/overlap-output-readback-2026-09-20`
(`9f20b35c9805788d9bc35b78c33eae696b5fc405`) runs both readback orders in the
same tuned sessions. Three fresh processes per GPU, reversing the arm order in
the middle process; three warmup and seven saved sequences per arm. Builds and
GPUs run sequentially, without a profiler. CPU affinity is `0,2,4,6,8,10`, with
unfixed clocks and existing driver caches.

| GPU | Serial decode/token | Queued decode/token | Paired gains |
| --- | ---: | ---: | --- |
| RTX 5070 | 1.402 ms | 1.386 ms | 1.37%, 1.17%, 1.65% |
| Arc B570 | 3.784 ms | 3.653 ms | 2.75%, 4.01%, 3.60% |

Values are medians of process medians. All twelve saved logit sets match bit
for bit between arms and retain all 33 CPU-reference token choices. Reproduce
on the experiment branch with `MEGANEURA_COMPARE_READBACK=1`; add
`MEGANEURA_OVERLAP_READBACK=1` to run queued first. These control switches are
not in the production helper.

Separate Nsight Systems 2025.5.2 Vulkan/OS-runtime captures use one submission
per graph and no kernel tuning to check ordering. In the final 32 decode steps,
the serial path has a host `poll` between graph and readback submissions in
32/32 cases; the queued path has none. Nsight does not expose the timeline
semaphore wait itself in these traces. Profiled times are not latency evidence
and do not measure barrier cost.

## Fresh default-invocation comparison

Three fresh processes per engine and GPU, with rotated engine order. Kernel
and submission tuning retain their 30-second and two-second per-session
budgets. Every kernel search visits all eligible classes and all comparisons
qualify; no kernel search exhausts its budget. Milliseconds, median of process
medians, including CPU copying of all logits:

| GPU / phase | Meganeura | llama.cpp |
| --- | ---: | ---: |
| RTX 5070 prefill | 7.210 | 7.126 |
| RTX 5070 decode/token | 1.389 | 1.361 |
| Arc B570 prefill | 14.048 | 15.650 |
| Arc B570 decode/token | 3.666 | 3.431 |

NVIDIA is about 1.2%/2.0% behind; Intel prefill is 10.2% faster and decode 6.8%
slower. Decode process medians span 1.346-1.503 ms and 3.656-3.957 ms. The slower
NVIDIA run keeps eight attention splits instead of sixteen, despite identical
GEMV choices; Intel's slower run selects two submission chunks instead of four.
This is evidence to investigate tuning stability, not proof that those choices
explain every timing difference. The conservative noise guard is unchanged.

All twelve fresh logit sets are finite and retain all 33 CPU-reference choices.
Maximum per-row relative L2 errors are 0.00000954/0.000178 for Meganeura and
0.01121/0.01314 for llama.cpp. Median whole preparation is 9.61/31.45 seconds
for Meganeura and 0.25/0.51 seconds for llama.cpp, with warm driver caches.
This is one model and batch-one latency, not batched throughput or language
quality. Intel's secondary PCIe x1 installation limits readback generalization.

All 449 active library tests, all-target/all-feature Clippy and both existing
device-local regressions pass on both GPUs. The readback regression checks
fresh GPU writes, empty reads and a partial 16 MiB staging tail. No new test
executable. Implementation CI passes all six jobs, including host coverage;
known Naga Workgroup ArrayStride diagnostics remain. Raw data and captures
stay outside Git.
