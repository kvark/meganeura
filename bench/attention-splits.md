# Attention split-count pilot

Based on `84da75c5f07421ca6aa6eabb919c2f718a868d74`, with only a diagnostic
`MEGANEURA_ATTENTION_SPLITS=1..16` override. One means the existing unsplit
kernel. Do not adopt the override as a production default.

Use `gguf_latency.md`'s official F16 model and settings, with the override
applied to both sessions. One process per condition, in the order 8, 1, 2, 4,
16; medians in milliseconds. This pilot is not a replicated comparison.

| Splits | 5070 prefill | 5070 decode | B570 prefill | B570 decode |
| ---: | ---: | ---: | ---: | ---: |
| 8, current | 7.48 | 1.87 | 21.33 | 4.43 |
| 1 | 7.51 | 2.92 | 19.96 | 7.50 |
| 2 | 7.54 | 2.74 | 21.37 | 6.91 |
| 4 | 7.29 | 2.15 | 20.97 | 4.98 |
| 16 | 7.58 | 1.74 | 21.99 | 4.25 |

The preferred geometry differs between prefill and decode. All 33 next-token
predictions match the independent CPU reference in every condition on both
GPUs. A production search would need to qualify and time the whole split/combine
sequence, including its scratch layout, against the unsplit alternative.
Raw results and binaries are not tracked.
