# Queue output readback before the CPU wait (2026-09-20)

Source-only experiment on `999bfadff710c67e0c200ca47474551c01dfc4cb`.
Uses the [matched GGUF diagnostic](gguf_latency.md), unchanged tensors,
precision, kernel tuning and fresh command recording. No Inferena changes.

Both GPUs already select staged rather than mapped reads for the 196608-byte
logits. The new `wait_read_output` queues a known staged copy before the CPU
wait. Unknown allocations and mapped reads wait first; the existing readback
probe still chooses the path and checks every bit. No new threshold.

Three fresh processes per GPU, both readback orders in each process using
the exact same tuned sessions. Reverse order in the middle process. Each arm
has three warmup and seven saved sequences. CPU affinity `0,2,4,6,8,10`,
unfixed clocks, sequential GPUs, no competing build or profiler.

| GPU | Serial decode/token | Queued decode/token | Paired gains |
| --- | ---: | ---: | --- |
| RTX 5070 | 1.402 ms | 1.386 ms | 1.37%, 1.17%, 1.65% |
| Arc B570 | 3.784 ms | 3.653 ms | 2.75%, 4.01%, 3.60% |

Median of process medians. All twelve saved logit sets are finite, agree on
the 33 CPU-reference token choices, and match bit for bit between arms.
Maximum per-row relative L2 against CPU is 0.00000954/0.000177. An earlier
independently retuned Intel pair selected different chunk counts and did not
isolate readback ordering; it is not used in this table.

Reproduce with `MEGANEURA_COMPARE_READBACK=1` and the ordinary tuned helper
command. Add `MEGANEURA_OVERLAP_READBACK=1` to run queued first. Output suffixes
are `-serial` and `-queued`. In queued results, `combined_wait_readback=true`
means the second CPU interval includes waiting and copying; the third is zero.
These intervals do not separate kernel or barrier cost. Raw data stays outside
Git. The existing readback regression now queues fresh producer work for every
read size, including the partial 16 MiB staging tail; it passes on both GPUs.
