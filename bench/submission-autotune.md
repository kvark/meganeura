# Automatic submission search (2026-09-19)

Implementation: `7951b44bf63c76838200d8bd631afc47c024faba`, following the
[ordinary-recording sweep](command-recording.md). Single-submit control:
`1e13217c8f87aab6e6c3d34073268fe02f45be4e`. Model, llama.cpp revision and
sampling follow the [matched GGUF diagnostic](gguf_latency.md). This does
not alter the submitted paper or frozen Inferena cohort.

`Session::tune_submissions` measures the initialized graph on private writable
allocations, borrowing readonly weights. Aliasing and requested memory
placement are preserved. All writable words must match the original schedule,
including during timed probes. Live caches, parameters and optimizer state
are not advanced. The same ordinary recorder serves probes and `step()`;
every command is recorded again. No model/device table selects chunk counts.

The helper primes a real 128-token prefix and 32 decode positions before
searching both sessions. Each search has two seconds, six alternating sample
pairs, one warmup per side and a 1% minimum gain plus the existing noise guard.
The library's default minimum gain is 5%. Candidates are powers of two through
64, plus a shorter graph's upper bound. Probe reset/check work is outside the
graph samples but inside the search budget. The deadline is checked between
complete probes; setup and GPU work cannot be preempted. Incomplete comparisons
cannot win. Runtime-appended optimizer work and host output readback are not
sampled by this scheduling API.

## Post-search measurements

Three fresh processes per arm and GPU, with rotated engine order and existing
driver caches. CPU affinity is the six physical cores (`0,2,4,6,8,10`);
CPU/GPU clocks are not fixed. All builds/profiling finish before measurement,
and GPUs run sequentially. Kernel tuning is unchanged: 30 seconds per session,
64 classes and 256 MiB scratch. Every session visits all seven eligible classes
and qualifies every comparison within its kernel-search budget.

The table uses the seven measured sequences after three warmups, not the
search samples. Input updates, recording, execution, waiting and full-logit
host copies are included. Milliseconds; median of three process medians:

| GPU / phase | Single submit | Automatic choice | llama.cpp |
| --- | ---: | ---: | ---: |
| RTX 5070 prefill | 8.334 | 7.268 | 7.131 |
| RTX 5070 decode/token | 2.431 | 1.423 | 1.385 |
| Arc B570 prefill | 20.913 | 20.008 | 15.644 |
| Arc B570 decode/token | 4.596 | 3.779 | 3.438 |

Decode improves in all three pairs: about 41% on NVIDIA and 18% on Intel by
the medians. NVIDIA selects eight chunks for both phases in every process;
decode medians span 1.416-1.423 ms. Intel decode selects `4,2,4`, with medians
3.776-4.018 ms. Its noisier second run rejects faster candidate medians and
keeps two chunks. The automatic search does **not** recover the full Intel
manual-sweep gain; retaining this run matters. Intel prefill selects `8,4,4`.

Relative to llama.cpp, decode is 1.03x/1.10x and prefill 1.02x/1.28x. These
are batch-one interactive latencies, not peak-throughput or batched-serving
results. A different model, request batch or cache length needs its own search.
No claim of a GPU barrier-cost decomposition follows from these wall times.

NVIDIA's scheduling searches take 0.17/0.78 seconds for decode/prefill. Intel
takes 0.82/about 2.05 seconds; its final 64-chunk prefill comparison exhausts
the soft budget and cannot replace the fully qualified incumbent. Every reached
scheduling comparison is bitwise-qualified. Private buffer requests are about
23/31 MiB, released after each search; these are not peak VRAM measurements and
exclude driver command/descriptor pools. Median whole preparation, including
input priming and both searches, is 8.65/28.66 seconds versus 7.63/25.49 for the
single-submit control and 0.25/0.49 for llama.cpp. These are not cache-cold
compiler measurements.

All 18 saved logit sets are finite and retain all 33 independent CPU-reference
next-token choices. Maximum per-row relative L2 in the automatic Meganeura runs
is 0.00000953/0.000176. This is fixed-token numerical evidence, not identical
intermediate arithmetic or language-quality equivalence between engines.

453 library tests pass. The four existing opt-in GPU transition/staging tests
and both chunk regressions pass on both GPUs. The added broad GPU check detects
changed tail words and covers early bounds; the training test also checks
unchanged state and subsequent optimizer updates. No new test executable or
tolerance relaxation. The known Naga Workgroup ArrayStride warning remains.
[Implementation CI](https://github.com/kvark/meganeura/actions/runs/35475657564)
passes all six jobs, including Metal, Windows and coverage. Rust host line
coverage is 83.7% overall and 86.3% in the new scheduling module (all 16 of its
functions are covered). WGSL execution is not instrumented.

Reproduce with the normal tuned `gguf_latency ... 30` invocation in the matched
diagnostic; `submission_tuning` records options, decisions and samples. Keep
raw outputs, logits and binaries outside Git. Intel prefill remains the largest
gap. More stable scheduling decisions and native/rectangular cooperative-matrix
coverage are separate follow-ups, not measured benefits of this change.

## Check timestamp attribution before the next kernel change

A separate B570 capture (`MEGANEURA_GPU_TIMING=1`, same source/helper) has
suspicious attribution: a `[128,3072,576]` matmul gets about 6.5 microseconds,
while the following SwiGLU interval gets 0.20-0.26 milliseconds. This is not
evidence that SwiGLU itself dominates. The pinned Blade writes pass markers at
`TOP_OF_PIPE`, after barriers targeting compute/indirect stages. A
[Vulkan timestamp's dependency is limited to its requested stage](https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdWriteTimestamp.html);
these markers need not wait for preceding compute to finish. Calibration to
the CPU clock does not fix attribution to individual dispatches.

Validate stage-aligned boundaries against isolated work or a native capture
before ranking Intel kernels by these intervals. This concern does not affect
the host-wall scheduling probes or the uninstrumented comparison above. Do
not shift labels by hand or infer a barrier cost by subtracting timestamps.
