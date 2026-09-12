# CUDA Graph methodology and final outcomes

The original paper-v1 path called default `torch.compile` but bypassed the
legacy explicit CUDA Graph helpers. The reviewer concern was valid:
compiler graph capture is not CUDA command-graph replay. The camera-ready
manuscript now replaces those timings with the final Inferena `17d13a3`
cohort. [Complete findings](RESULTS.md).

## What the frozen collector does

Arithmetic (strict/accelerated) and preparation (light/searched) are
independent axes. On CUDA it collects:

| Condition | PyTorch | Meganeura |
|---|---|---|
| Light, no replay | Default compilation, no CUDA Graph | Greedy graph compilation, kernel search off |
| Light, replay | Default compilation, whole-phase CUDA Graph | Same light policy |
| Searched, replay | Max-autotune compilation, whole-phase CUDA Graph | Bounded measured kernel selection during build |

Inductor's internal `triton.cudagraphs` option is off in every condition.
The explicit capture switch owns replay; other compiler-mode options remain.
The record retains resolved options and per-phase execution/qualification.
Preparation, compilation, capture, warmup, and execution share one dedicated
CUDA stream.

Capture separately covers full inference, the minimal shape, and
forward/loss/backward without optimizer update. Qualification compares every
participating output and gradient element against uncaptured PyTorch.
Each phase has two uncaptured repeats and two consecutive replays;
accelerated training instead has eight uncaptured repeats. Fixed absolute,
RMS, and maximum-error bounds are not fitted to observed repeatability.
The full-gradient summaries distinguish ordinary nondeterminism from replay
error. Cross-engine validation remains sampled outputs and parameter norms,
not the same full-element test.

Five warmups precede twenty synchronized host-wall samples per phase.
Input loading, readback, capture, and qualification are outside these
performance samples. Compiler preparation, graph preparation, and full-tensor
qualification have separate fields. No eager result substitutes for a failed
compiler or capture condition.

Every condition has a fresh process and empty private TorchInductor/Triton
caches. Persistent driver and vendor-library state remains as found.
Engine order alternates and condition order rotates. This mitigates ordering
effects without creating independent first-ever cold starts; light can still
benefit from lower-level history. The final protocol was not modified to
flush caches.

## Final cohort outcome

Both RTX 5070 and H100 complete all 90 pairs. Default replay reduces H100
135M stateless-token latency 4.244 → 1.276 ms and diffusion F+L+B
12.035 → 4.035 ms. It changes the scientific conclusion and is included
in both primary light and searched CUDA comparisons.

Windows 3050 completes 31 pairs, then fails on the second strict searched
135M training capture. The same condition succeeded in repetition one.
The H100 extension completes five strict pairs (360M all three conditions;
1.7B both default conditions), then fails on searched 1.7B training capture.
The first reported error in both is `CUBLAS_STATUS_EXECUTION_FAILED` from
`cublasSgemm` inside the training forward call; ending capture then reports
`cudaErrorStreamCaptureInvalidated`.

These are observed capture-path failures, not proven OOM or a diagnosis of
which library, driver, or harness interaction caused them. Meganeura completes
its side, but neither failed pair contributes a timing ratio. Partial records
retain their actual replicate count and stay outside the complete-cohort
aggregates. Unreached conditions are unmeasured.

ROCm/XPU final campaigns explicitly omit searched compilation after bring-up
failures/timeouts; the default condition remains useful availability evidence.
ROCm replay is unqualified in this protocol, not claimed impossible.
MPS and CPU are explicitly eager. There is no claim that every backend
executes an identical compiled or captured implementation.
