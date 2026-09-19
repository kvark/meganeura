# Ordinary command recording (2026-09-19)

Follow-up to [GEMV row grouping](gemv-row-grouping.md), using the same
[matched GGUF diagnostic](gguf_latency.md): SmolLM2-135M, one sequence,
128-token prefill, 32 cached decode steps and full-logit host copies.
Commands are freshly recorded. The submitted paper and Inferena cohort are
unchanged; this is not batched-throughput evidence.

## CPU attribution

Source-only instrumentation is preserved at
`experiment/recording-cost-2026-09-19`, commit
`27a2236` (based on the pre-squash head of #195). It places CPU clocks around
ordinary recording stages, accumulates them locally, then prints after submit.
One process per GPU; medians of the 224 measured decode calls, in microseconds:

| Recording stage | RTX 5070 | Arc B570 |
| --- | ---: | ---: |
| Wait/start encoder | 11 | 67 |
| Barrier calls | 164 | 56 |
| Pipeline lookup | 104 | 183 |
| Pipeline binding | 62 | 62 |
| Resource binding | 194 | 650 |
| Dispatch calls | 396 | 421 |
| Final submission | 26 | 17 |

These are instrumented CPU observations, including clock overhead, not GPU
kernel or barrier durations. Clocks were not fixed. The stage medians need not
sum to a whole-call median, and they are not a decomposition of the gap to
llama.cpp. Nsight's Vulkan API trace omitted most dispatch/barrier API calls,
so summing the captured API rows would undercount recording work.

## Resolve pipeline variants outside recording

Implementation `1e13217c8f87aab6e6c3d34073268fe02f45be4e` resolves variants
after compilation, tuning, attention-sequence replacement and tuning swaps.
Recording still binds current buffers. It no longer allocates candidate lists
or rehashes generated kernel definitions each step; no descriptor or command
recording is retained.

Three fresh processes per arm and GPU, rotated engine order, with existing
driver caches. Median of process medians, milliseconds:

| GPU / phase | #195 control | Resolved variants | llama.cpp |
| --- | ---: | ---: | ---: |
| RTX 5070 prefill | 8.397 | 8.320 | 7.120 |
| RTX 5070 decode/token | 2.398 | 2.415 | 1.365 |
| Arc B570 prefill | 21.195 | 20.919 | 15.627 |
| Arc B570 decode/token | 4.696 | 4.640 | 3.440 |

This is a small change, not the answer to the gap. Intel recording time falls
from 1.347 to 1.283 ms; its decode improves in all three pairs (about 1.2% by
the medians). NVIDIA decode is mixed, with overlapping process ranges:
2.370-2.445 ms before, 2.377-2.431 ms after. No NVIDIA decode gain is established.

All 18 saved logit sets are finite and preserve all 33 independent CPU-reference
next-token choices. Maximum per-row relative L2 for the new Meganeura runs is
0.00000953/0.000176. Kernel searches visit all seven classes per session, qualify
every comparison and finish within the unchanged 30-second/session limit.
This does not establish language-quality equivalence or identical intermediate
arithmetic between engines.

452 existing library tests and all four opt-in GPU transition/staging tests
pass, the latter on both GPUs. No tests or test executables were added.
[CI for the implementation](https://github.com/kvark/meganeura/actions/runs/35473055263)
passes all six jobs, including Metal, Windows and host coverage. The known
Naga Workgroup ArrayStride validation warning is unchanged. Rust host line
coverage is 83.7% overall and 81.1% in `runtime.rs`; WGSL is not instrumented.

## Overlap recording with GPU execution

llama.cpp's pinned Vulkan implementation submits graph sections while recording
the rest (`ggml_vk_build_graph` and its caller). The Meganeura diagnostic had
explicitly selected one submission, serializing its recording and execution.
The existing `set_submission_chunks` path lets earlier sections execute while
later ones are recorded. It preserves barrier-group order and records every
command again; it is not the discarded command-replay prototype.

Source-only sweep:
`experiment/submission-overlap-2026-09-19`, commit
`7cbccd290531384032f79992ff0a0e83b3a53f95`. After the ordinary kernel search,
it changes only submission count. Each process sweeps
`1,2,4,8,16,32,64,32,16,8,4,2,1`, with three warmup and seven measured
sequences at every setting. The two sessions share weights/caches; kernel
choices stay fixed within a process. Three fresh processes per GPU, with a
fresh llama.cpp run before or after each sweep in alternating order.

Median of the three process medians, taking the median of the two sweep
positions first (64 has one position). Milliseconds; each cell is
**prefill / decode per token**:

| Submissions | RTX 5070 | Arc B570 |
| --- | ---: | ---: |
| 1 | 8.342 / 2.416 | 20.905 / 4.608 |
| 2 | 7.570 / 1.701 | 20.317 / 4.040 |
| 4 | 7.359 / 1.495 | 20.017 / 3.757 |
| 8 | 7.279 / 1.415 | 19.903 / 3.648 |
| 16 | 7.279 / 1.444 | 19.912 / 3.649 |
| 32 | 7.364 / 1.525 | 20.025 / 3.786 |
| 64 | 7.506 / 1.660 | 21.201 / 3.879 |
| llama.cpp | 7.120 / 1.366 | 15.639 / 3.439 |

Eight submissions reduce decode by about 41% on NVIDIA and 21% on Intel.
At that point Meganeura is 1.04x/1.06x llama.cpp decode latency and
1.02x/1.27x prefill latency. Eight-submission process medians span
1.398-1.421 ms on NVIDIA and 3.647-3.650 ms on Intel. Intel's apparent further
gain at sixteen in the initial exploratory run did not survive these repeats.
More submissions eventually lose performance; these measurements are not a
reason to hard-code eight for every workload or platform.

All 78 Meganeura logit sets are bit-identical across the thirteen settings
within each process. All 84 sets including llama.cpp are finite and retain
33/33 independent CPU-reference next-token choices. Model, batch, inputs,
readback, precision and validation are unchanged. Kernel search qualifies all
comparisons within the existing budgets. Encoder resizing is outside timing.
This is an exploratory scheduling sweep, not an automatically selected,
held-out result or a new paper cohort.

A separate Nsight Systems 2025.5.2 capture of `1,8,1` confirms overlap. Matching
the eight queue submissions for each of 320 captured decode steps shows GPU
work starting before the last submission returned in all 320; the median lead
is 0.745 ms. Profiling used Vulkan/OS-runtime tracing with CPU sampling and
context-switch tracing disabled. Profiled times are not used in the table.
Recording time also changes with chunking; clocks were not fixed, so this is
not a complete attribution of the wall-time gain. Neither waiting time nor
wall time minus kernel time measures GPU barrier cost.

Reproduce on that source branch after building `gguf_latency`:

```sh
GGUF_SUBMISSION_CHUNKS=1,2,4,8,16,32,64,32,16,8,4,2,1 \
  target/release/examples/gguf_latency model.gguf output-prefix 30
```

Use the model revision, single-device selection and CPU affinity from the
matched diagnostic. Each setting writes a separate suffix; keep raw outputs
and native captures outside Git. The next engineering step is a bounded
scheduling search on representative work, respecting caller scheduling policy
and live KV/training state. The production default remains one submission;
this PR does not silently install the best point from the sweep.
