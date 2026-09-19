# GGUF dense-weight layout experiment

Preserve GGUF's `[N,K]` row order for dense F16/F32 projections and use
`MatMulBT`. Packed weights retain their existing layout. RoPE unpermutation,
row slicing and the embedding table are unchanged.

The prototype also lowers single-row `FusedMatMulBTAdd` to the existing
GEMV-BT followed by an add. Its tiled lowering cost roughly 4.6 ms per decode
on the 5070; the two-dispatch form reduced that to 1.61 ms in a pilot.
This is not the desired final implementation: a fused GEMV-BT add should
avoid the extra dispatch and temporary.

The experiment exposed another missing search case: fused transposed matmuls
were excluded from tuning. Extending the existing search recovered Intel
prefill from 27.3 to 21.1 ms in consecutive pilots. This small general fix is
separate from the layout experiment.

Use `gguf_latency.md`'s official F16 model, reference and settings. Control:
`84da75c5f07421ca6aa6eabb919c2f718a868d74`. Three fresh-process pairs per GPU,
reversing order; milliseconds, median of process medians:

| GPU | Phase | Control | Native rows with tuning |
| --- | --- | ---: | ---: |
| RTX 5070 | Prefill | 7.270 | 7.265 |
| RTX 5070 | Decode | 1.814 | 1.609 |
| Arc B570 | Prefill | 21.318 | 21.227 |
| Arc B570 | Decode | 4.549 | 4.284 |

Experimental decode medians span 1.605--1.638 ms and 4.280--4.284 ms.
All 33 predictions agree with the independent CPU reference in every run.
Maximum relative L2 logit errors are 7.87e-6 and 1.76e-4. No reduced-precision
activation conversion or qualification relaxation is added.

The layout currently loses RMSNorm/GEMV and gate/up packing fusions. Dispatch
counts increase from 454/394 (prefill/decode) to 484/544. Restore the applicable
general fusions and update GGUF layout tests/documentation before production;
the change is only qualified on the matched SmolLM2 model so far.
Raw results and binaries are not tracked. The paper cohort is unchanged.
