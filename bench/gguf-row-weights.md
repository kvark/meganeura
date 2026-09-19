# Dense GGUF rows with general fusions

Source: `69bbb48450fee8f9957b682ffec95733ca50c1f4`. Control:
`861d87b9a85f666a5a131f82d07c0b66d9131e16`, including attention search.
Both use ordinary command recording. Model, llama.cpp revision, hardware,
commands and sampling follow [the matched GGUF diagnostic](gguf_latency.md).
The submitted paper and frozen Inferena cohort are unchanged.

Dense F32/F16 projections retain the file's contiguous `[N,K]` rows and use
transposed-B matmul. Block formats retain their existing layout. A fused GEMV
add reuses the existing transposed shader. Gate/up packing now works in both
orientations, sharing its implementation across SwiGLU/GeGLU and greedy/egglog
rewrites. RMSNorm folding also retains the transposed layout. Both fused GEMV
forms enter the existing qualified width/reduction search with unchanged
budgets and precision. No device names or model-specific thresholds are added.

Dense row updates preserve the current destination bytes, not a cached host
image that can go stale after checkpoint restores or shared-session updates.
Large output counts span a two-dimensional dispatch grid: B570 reports an
X limit of 65,535 workgroups, while the 5070 reports 2,147,483,647. This is an
indexing/legality fix, not a performance threshold. The common GEMV source
handles both grids, including a partial final row.

## Repeated comparison

Milliseconds; median of three fresh-process medians, with rotated engine order.
One sequence, 128-token prefill, 32 cached decode steps, f32 K/V and full logits
copied to the host. Each process has three warmups and seven measured sequences.
Both Meganeura sessions use 30-second/64-class/256-MiB tuning limits. Engines
and GPUs run sequentially without competing builds or profilers.

| GPU | Phase | Control | Native rows + fusions | llama.cpp |
| --- | --- | ---: | ---: | ---: |
| RTX 5070 | Prefill | 8.777 | 8.479 | 7.087 |
| RTX 5070 | Decode/token | 3.002 | 2.514 | 1.385 |
| Arc B570 | Prefill | 21.337 | 21.251 | 15.610 |
| Arc B570 | Decode/token | 5.561 | 5.129 | 3.425 |

Decode improves in all three trials, about 16%/8% by the NVIDIA/Intel medians.
NVIDIA's process medians range from 2.475 to 2.748 ms versus 2.934 to 3.087 ms
for the control. Intel ranges from 5.107 to 5.133 ms versus 5.541 to 5.585 ms.
CPU/GPU clocks were not fixed. Earlier pilots had larger NVIDIA variation in
recording/readback stages even with identical matrix choices; report the ranges,
not the fastest process. These overlapping stages are not a barrier-cost budget.

All sessions visit all seven eligible classes, qualify every comparison and
finish within their budget. Median preparation is 3.73/10.76 seconds versus
3.97/11.22 for the control and 0.28/0.52 for llama.cpp. These are process starts
with existing driver caches, not cache-cold compilation measurements.

All 18 saved logit sets are finite and retain all 33 CPU-reference next-token
choices. Maximum per-row relative L2 for the new path is 0.00000976 on NVIDIA
and 0.000177 on Intel. This checks numerical agreement on fixed tokens, not
language quality or identical intermediate arithmetic across engines.

The gap remains: decode is 1.82x/1.50x llama.cpp; prefill is 1.20x/1.36x.
Next candidates are measured output-row grouping in GEMV and cooperative-matrix
coverage. Fewer dispatches alone did not predict the pilot's winners.
The [row-grouping follow-up](gemv-row-grouping.md) measures the first of these
without changing the precision policy or tuning limits.

452 CPU tests, the 22-test GGUF suite and the 17-test GEMV group pass, with GPU
checks on both devices. The broad regression includes source restaging, odd
f16 dimensions, both activations, normalized tuning and wide vocabulary grids.
No new test executable or tolerance relaxation. The known Naga Workgroup
ArrayStride validation warning remains. Raw timings, logits and binaries stay
outside Git. The earlier replay-only experiment is retained at `0558f63` on
`perf/gguf-native-rows-2026-09-19`, not treated as production performance.
