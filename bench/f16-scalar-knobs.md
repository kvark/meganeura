# F16-storage scalar layouts (2026-09-20)

Pilot on top of `741b35aaa36aa5029b24c36779b84e3dd280f469`, using the
[matched GGUF diagnostic](gguf_latency.md). This enables the existing K-stage
and column-layout controls for plain F16 weights, without rounding F32
activations or accumulators. Defaults and block-quantized layouts are unchanged.

One fresh process per setting and GPU, 30-second kernel-search budget, automatic
submission search, three warmup sequences and seven saved sequences. CPU affinity
`0,2,4,6,8,10`; builds and the two GPUs run sequentially. Clocks are not fixed.
Median prefill milliseconds, with contiguous columns:

| GPU | K32 (default) | K16 | K8 |
| --- | ---: | ---: | ---: |
| RTX 5070 | 7.247 | 7.802 | 8.324 |
| Arc B570 | 20.081 | 15.630 | 14.013 |

Interleaving columns did not improve either GPU's best setting. All twelve
saved logit sets are finite and preserve the 33 independent CPU-reference
next-token choices. Maximum per-row relative L2 is 0.00000954 on NVIDIA and
0.000176 on Intel; the existing private-input tuner rejects no comparison for
invalid output. These are exploratory results, not replicated causal estimates:
tile, attention, GEMV and submission choices are retuned in every process.
Different winners motivate per-class measured selection, not a global default
change or device table. No claim about native F16 cooperative arithmetic follows.

Reproduce with `MEGANEURA_MATMUL_K_STAGE=8` (also 16 and 32) and
`MEGANEURA_INTERLEAVE_COLUMNS=0` (also 1), using the normal tuned helper command.
Raw results remain outside Git.
