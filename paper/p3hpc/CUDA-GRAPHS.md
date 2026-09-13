# Replay methodology and final outcomes

The final Inferena revision is `efb1e520`; [RESULTS.md](RESULTS.md)
identifies all observations. Compiler graph capture is not command-graph
replay. The manuscript measures the latter explicitly on NVIDIA.

## Frozen conditions

| Condition | PyTorch | Meganeura |
|---|---|---|
| Light, no replay | Default compilation, uncaptured | Greedy compilation, empirical search off |
| Light, replay | Default compilation, whole-phase CUDA Graph | Same light policy |
| Searched, replay | Max-autotune compilation, whole-phase CUDA Graph | Bounded measured selection during build |

RTX 5070 and H100 complete all three conditions. Windows RTX 3050 completes
the first two and explicitly omits searched compilation.

Inductor's internal `triton.cudagraphs` option is disabled in every condition;
the explicit switch owns replay. Compilation, preparation, capture, warmup,
and execution share one dedicated CUDA stream.

Full inference, minimal forward, and F+loss+backward are captured separately.
Replay qualification compares all participating outputs and gradient elements
against uncaptured PyTorch. Each phase has two uncaptured repeats and two
consecutive replays; accelerated training uses eight uncaptured repeats.
Fixed RMS/maximum bounds are not fitted to the observed errors. This is
distinct from the cross-engine sampled-output and gradient-norm gates.

Five warmups precede twenty synchronized samples. Loading, readback, capture,
and validation are outside steady-state samples. Compiler preparation, capture,
and research qualification have separate recorded costs. A failed requested
compiler/capture condition never receives an eager substitute.

Every condition has a fresh process and empty private Inductor/Triton caches.
Persistent driver and vendor-library state remains as found. Rotated condition
and engine order mitigates history dependence without guaranteeing a cold start;
light may reuse lower-level state from earlier searched work.

## Outcome

RTX 5070 and H100 each complete 90 pairs; Windows completes 60. Default replay
reduces H100 135M stateless-token latency 3.496 → 1.275 ms and diffusion
F+L+B 14.322 → 4.031 ms. Windows has no failed pair in this final collection.

The H100 extension completes five strict pairs: all three 360M conditions and
both default 1.7B conditions. Searched 1.7B fails during training forward capture,
with a generated Triton launch reporting an earlier capture error and graph
finalization reporting `cudaErrorStreamCaptureInvalidated`. The initiating
error is not identified. This is not a proven OOM or cuBLAS diagnosis.
Meganeura completes, but that unpaired result supplies no validated speed ratio.
Later conditions are unmeasured.

## Important limits

The collector still restricts explicit replay to NVIDIA. ROCm exposes graphs
through PyTorch's CUDA API and XPU has its own graph API, but neither was
qualified here. This is a protocol omission, **not** evidence of unavailable
backend functionality. Their final default/no-replay subsets follow separately
documented search failures or timeouts.

MPS and CPU are eager. The pinned MPS API has no CUDA-style replay entry point,
but an MPS Inductor backend exists and was not exercised. The paper does not
claim these are the strongest automatic policies on every device.
See [deferred methodology work](NEXT-COHORT.md); the frozen cohort is unchanged.
