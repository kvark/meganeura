# Native dense GGUF rows: fused GEMV add

Experimental follow-up to `621de44ed57ff7aa54dc0151417f05997962cb73`,
based on `6672ef385d79e73accae5cc7c7d2eefca8ff50fd`. Dense F32/F16
projections retain `[N,K]` rows. A transposed GEMV with an addend now uses
one dispatch, sharing the existing shader source and qualified GEMV search.
No precision policy, tolerance, or tuning budget changes.

Same model, llama.cpp revision and diagnostic as `gguf_latency.md`. Three
fresh-process trials with rotated order; median of process medians in ms:

| GPU | Phase | Two-dispatch add | Fused add | llama.cpp |
| --- | --- | ---: | ---: | ---: |
| RTX 5070 | Prefill | 7.25 | 7.25 | 7.09 |
| RTX 5070 | Decode/token | 1.611 | 1.537 | 1.368 |
| Arc B570 | Prefill | 21.24 | 21.24 | 15.64 |
| Arc B570 | Decode/token | 4.281 | 4.287 | 3.445 |

**Both Meganeura arms use the rejected serialized-replay experiment.** These
are kernel-layout comparisons, not production latency claims. Decode dispatches
fall from 544 to 484. NVIDIA improves by about 4.6%; Intel does not improve.
All 33 saved next-token choices match the independent CPU reference in every
run. Maximum relative L2 errors are 0.00000787 (NVIDIA), 0.000177 (Intel).

The existing every-shape GEMV regression passes on both GPUs, the eight GEMV
unit tests pass, and all 149 GGUF unit tests pass. Full model-feature regression
coverage is still pending. Native rows still lose gate/up packing and norm
fusion; restore those before considering production. Raw artifacts stay outside
Git. This experiment does not change the paper or its frozen cohort.
