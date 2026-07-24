# SmolLM performance regression — 2026-07-23

## Conclusion

The remembered sub-2× NVIDIA inference gap was real. Under an adapted current
accelerated protocol, an April Meganeura checkpoint ran SmolLM2-135M
inference in about 5.39 ms versus a current PyTorch value near 2.8 ms. The
current local Meganeura tree takes about 8.69 ms.

The largest discontinuity begins at `09d14ca` (2026-05-17),
`runtime+compile: don't coop-promote matmul-with-epilogue`. This was a
necessary correctness fix, not an accidental removal: the old path used
cooperative workgroup geometry with a scalar epilogue pipeline and silently
left output rows unwritten.

We implemented and validated the smallest safe cooperative-matrix epilogue
variant. It stages the cooperative accumulator through workgroup memory and
evaluates the existing unary pointwise DAG with guarded scalar lanes. The
current packed SwiGLU graph, however, contains no epilogue-bearing matmuls at
the relevant sites, so the new capability does not recover the old SmolLM
result.

## Checkpoint measurements

The historical checkouts were run through the current Inferena workload as far
as their APIs allowed, with the corrected RoPE theta applied where necessary.
They are diagnostic bisection evidence, not final paper cells.

| Checkpoint | Approx. date/state | Inference (ms) | Training (ms) |
|---|---|---:|---:|
| `8042e00` | April historical checkpoint | 5.394 | 22.329 |
| `7a90119` | immediately before the discontinuity | 5.452 | 22.324 |
| `09d14ca` | correctness fix | 9.373 | 58.028 |
| `8e31f2f` | later runtime work | 9.439 | 57.963 |
| `321219e` | Naga/Blade update | 9.406 | 58.795 |
| `3d24850` | e-graph extraction ownership | 9.997 | 35.849 |
| current dirty tree | repaired accelerated 5×20 protocol | 8.692 | 31.113 |

The current PyTorch accelerated result is 2.809 ms inference and 8.885 ms
training. Thus the diagnostic April gap is about 1.9× for inference, while the
current gap is about 3.1×.

## Root cause

Before `09d14ca`:

1. `fuse_epilogues` could fold a unary operation into a matmul and remap its
   output.
2. Runtime cooperative-matrix promotion later changed the dispatch's
   workgroup geometry and marked it cooperative.
3. Pipeline lookup prioritized the scalar matmul-with-epilogue pipeline
   because no cooperative+epilogue variant existed.
4. The scalar shader then ran with cooperative geometry. In the motivating
   case only the first output-row tile was covered; later rows remained zero,
   and downstream operations could produce infinities.

The fix correctly:

- prevents cooperative promotion when a matmul already carries an epilogue;
- prevents epilogue fusion from remapping protected named buffers;
- adds skinny-matmul and full-model finite-output regression tests.

The performance cost is that common transformer matmul+epilogue shapes now use
the scalar path even when the device supports fast cooperative matrices.

## Attempts that did not recover the result

- Lowering the cooperative-workgroup threshold from 128 to 16 worsened the
  current result to roughly 9.42 ms inference / 34.81 ms training.
- Disabling epilogue fusion globally did not restore the April performance.
- Later memory, reduction-fusion, and e-graph work recovered a substantial
  part of training time but did not recreate a correct cooperative epilogue.

These checks argue against simply reverting the guard or tuning the promotion
threshold.

## Implemented experiment

The implemented path covers:

1. cooperative matmul with the existing single-output epilogue DAG;
2. one geometry/pipeline key selected atomically at compile time;
3. numerical tests over edge tiles, skinny/tall matrices, named outputs, and
   every supported epilogue;
4. strict and accelerated output/gradient checks;
5. per-dispatch and end-to-end comparison on RTX 5080.

Observed outcome:

- no regression in strict-f32 correctness;
- a 1024×16 by 16×128 GPU regression test selects the cooperative path and
  validates the fused epilogue;
- accelerated SmolLM does not materially improve because the current graph
  has no eligible epilogue-bearing matmul;
- gradients remain within the paper gate;
- no reliance on a workgroup geometry rewrite after pipeline selection.

The remaining experiment is to repeat the regression and end-to-end run on
AMD. Recovering the historical SmolLM result now requires profiling the
current packed SwiGLU graph rather than further relaxing this promotion
guard.

## Paper use

This is a strong gap-analysis case study because it shows why performance
numbers changed and why “just turn tensor cores back on” is not acceptable.
It also motivates a general architectural lesson: kernel variant, workgroup
geometry, padding, and epilogue support should be one compiler decision rather
than independently mutated by compiler and runtime stages.

Raw diagnostic logs are under:

- `results/smollm-bisect-7a90119.*`
- `results/smollm-bisect-09d14ca.*`
- `results/smollm-bisect-8e31f2f.*`
- `results/smollm-bisect-321219e.*`
- `results/smollm-bisect-3d24850.*`
