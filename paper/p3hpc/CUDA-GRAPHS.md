# CUDA Graph baseline: what to explain at P3HPC

The concern is substantive. Inspecting Inferena's `paper-arxiv-1` source shows
that the paper's `bench_v2` path called `torch.compile(model)` in default mode.
Explicit CUDA Graph helpers existed, but only the separate legacy runner used
them (as a fallback when compilation was unavailable). They did not make the
paper-v1 benchmark a graphed benchmark. The reduced Meganeura merge did not
change that frozen Inferena source.

Three different graphs matter:

| Mechanism | What it removes or controls | What it does not establish |
|---|---|---|
| PyTorch compiler graph | Captures tensor computation; enables fusion and kernel selection | Default compilation alone is not proof of CUDA Graph replay |
| CUDA Graph | Replays a prepared device-work sequence with stable storage, reducing host launch overhead | Does not improve every kernel or support arbitrary dynamic Python behavior |
| Meganeura execution plan | Fixes kernels, storage, dependencies and dispatch groups | Current `step` still encodes/submits commands; it is not native CUDA Graph replay |

PyTorch's `reduce-overhead` and `max-autotune` can enable automatic CUDA Graph
use. Eligibility, skips and capture boundaries still matter; naming a mode
is not sufficient evidence. These are general automatic optimizations, fully
consistent with our preference against workload-specific hand tuning.
[Compiler modes](https://docs.pytorch.org/docs/stable/generated/torch.compile)

## Clean comparison branch

The follow-up is in Inferena's
[`experiment/p3hpc-cuda-graphs`](https://github.com/kvark/inferena/tree/experiment/p3hpc-cuda-graphs),
based on its merged main. Its `EXPERIMENT.md` owns the reproduction commands
and collection plan. `scripts/p3hpc.py` qualifies every selected engine pair
before a replicated campaign, checks declared versions/backends/modes and
validity gates, rotates execution order and retains failures outside Git.
`--profile` separately exports PyTorch host/device timelines alongside the
Meganeura dispatch sidecars; it is not a completed overhead analysis.
The branch retains the same five workloads, precision classes
and cross-engine validity gates, and adds explicit **whole-phase** capture:

- Full forward, minimal-shape forward, and forward/loss/backward each have a
  separate graph. Loss and backward cannot quietly remain outside capture.
- Inputs, parameters, outputs and gradients retain stable addresses. The
  captured backward overwrites gradients; they are not reset to `None` between
  replays. Qualification checks consecutive replays to catch accumulation.
- Every output and participating gradient element is checked against the
  uncaptured implementation before timing. This tests replay integrity, not a
  stronger cross-engine claim than the unchanged sampled/norm validation.
- Compilation, capture and CPU/readback qualification are outside steady-state
  timing and reported separately. Graph pool memory is not hidden.
- Timings are synchronized host wall time per call with resident inputs and
  no readback. There is still **no optimizer update**, data loading, H2D input
  transfer, KV cache or distributed communication in these workloads.
- Requested compile/capture failures stop that runner. Records disclose actual
  options and per-phase capture/validation; no eager timing is substituted.

This follows the lifetime and gradient rules in
[PyTorch's CUDA Graph documentation](https://docs.pytorch.org/docs/main/notes/cuda.html#cuda-graphs).
It deliberately excludes Inductor's inner graph trees when capturing the full
phase, while retaining automatic kernel search in the max-autotune condition.

## What new evidence is required

The September 8 strict ResNet-50 pilot qualified all three captured phases,
including every element of 108 gradient tensors. Its two process-level records
remain outside Git, with source frozen at Inferena's
`experiment/p3hpc-cuda-graphs-pilot-2026-09-08`. This checks the harness; one
process per condition is not a replicated performance claim or a new engine
comparison. See that branch's `EXPERIMENT.md` for the small result table and
the substantial graph-pool residency cost observed in the pilot.

First qualify all workloads. Then freeze an idle-device campaign with default
compiled/no-graph, default compiled/graph, and max-autotune/graph, rotating
configuration order across at least three fresh processes. Keep the same
precision settings and timing boundaries. Recollect Meganeura in the same
campaign, preserve failures, and report per-process dispersion, preparation
cost and memory. Store records outside main; retain source refs and concise
conclusions. NVIDIA results cannot establish ROCm/Metal behavior.

Do not replace individual favorable cells in the old table or compare new
PyTorch measurements to old Meganeura timings. A new qualified cohort gets its
own table and provenance. Until then, the submitted table remains a comparison
against **default-compiled PyTorch without verified CUDA Graph replay**, not
the strongest automatic CUDA baseline. Small-shape launch-sensitive claims
need particular care.

All three reviews were read on September 8. Review 1 explicitly challenges
the missing CUDA Graph baseline; Reviews 1 and 2 also question mixed reference
versions and execution modes. Review 2 requests a host/runtime-versus-GPU
breakdown. Capture qualification is therefore necessary but not sufficient:
the controlled comparison and overhead analysis remain open in the
[reviewer-response matrix](REVISION.md#reviewer-response-matrix).
