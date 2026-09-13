# Author review: corrections and collection gate

September 13, 2026. The manuscript still reports Inferena `17d13a3` /
Meganeura `fcdd76d1`. No measurements have been relabeled or replaced.
This review supersedes the earlier recommendation to submit without recollection.

## Decision

One replacement primary cohort is warranted: the harness unnecessarily
excluded native-f32 cooperative tiles and omitted available non-NVIDIA
replay paths. Correct those before freezing another revision. Do not repeat
the full three-condition NVIDIA campaign or require max-autotune everywhere.

Recommended primary policy:

- Enable Meganeura's bounded, numerically qualified kernel search on every
  device, independently of PyTorch's mode. Keep its incumbent on no improvement.
- Use PyTorch default compilation and qualified whole-phase replay wherever
  the pinned backend supports it. Default is not literally “untuned”: library
  heuristics and persistent lower-level caches remain possible.
- Keep strict/accelerated arithmetic, three fresh process pairs, five warmups,
  twenty samples, alternating engine order, and all numerical gates unchanged.
- Attempt compiler/replay qualification explicitly. Record failures, unsupported
  APIs, and deliberately selected fallback conditions separately; never
  silently substitute eager or uncaptured timings.
- Retain the expensive existing search/replay ablations as a separately pinned
  study. Add a small native tuning-off/on ablation after candidate integration,
  with preparation time and amortization, rather than repeating expensive
  PyTorch search across every machine.

This is a **bounded-search startup policy**, not an equal-deadline experiment.
Meganeura's current two-second search limit is soft, per session, and cannot
preempt an in-flight driver compile or GPU qualification. Three benchmark
sessions and all ordinary compilation also cost time. A genuine equal-budget
study needs a common preparation boundary and a supervised deadline, with
timeout outcomes retained. Do not claim it from the current mode flags.
Use a process-level safety deadline to stop runaway preparation in the next
collector; keep research validation separately timed and do not relax it.

NVIDIA collection drops from 90 to 30 paired processes. On H100 the removed
arms account for 209.1 minutes of recorded PyTorch compilation alone:
194.1 searched plus 15.1 default/no-replay. The retained default/replay arm
has 15.1 minutes of PyTorch compilation plus 4.9 minutes of graph validation,
before loading, native preparation, and other overhead. These are recorded
components, not a promise of a twenty-minute complete campaign.

## Engineering gates before freezing

| Gate | Finding and bounded action |
|---|---|
| Full-width arithmetic | Add a native-f32-only cooperative policy. `Auto` is insufficient because it permits f16 forward on f16-only devices; `Disabled` incorrectly excludes legal native-f32 tiles. Audit attention independently and keep the existing numerical qualification. |
| ROCm replay | `bench.py` checks `torch.version.cuda` and rejects HIP before qualification. ROCm deliberately uses `torch.cuda` APIs. Reuse the same full-output/full-gradient replay gate and dedicated stream rather than adding a weaker HIP test. |
| XPU replay | Pinned `torch.xpu.XPUGraph` and `torch.xpu.graph` exist. Parameterize the existing capture implementation over the device API; qualify actual models, including the retained embedding-backward workaround. API presence is not proof of model success. |
| Apple | Pinned MPS has no CUDA-style replay API. MPSGraph is not equivalent. However, `bench.py` also skips `torch.compile` on MPS despite an existing Inductor backend. Remove that assumption and qualify compilation on M3 before choosing its primary condition. |
| Convolution specialization | The ~1.325× ResNet F+L+B prototype is promising, but rejects production tuning and compiles both fallback and specialized pipelines. Integrate immutable-parameter specialization into ordinary candidate identity/code generation, charge compilation, and retain scalar/fallback candidates. Do not merge its experimental string-rewrite/environment-switch interface as production design. |
| Candidate coverage | First add legal convolution staging and residual-add GEMV workgroup widths to the existing bounded search. K=32 regressed in the prototype, so it is a candidate, not a new default. Keep scratch/time caps and correctness gates. |
| Representation scope | Packed/unpacked weights alter allocation and graph representation, not just a workgroup constant. Keep this a separate ablation unless a small, qualified integration is ready; it must not become an open-ended prerequisite for recollection. |
| Timing and memory | Pick up merged #178 for development traces. Preserve profile revision and timestamp granularity. Memory already records plan bytes and allocator peaks, but not comparable whole-process VRAM high-water marks; do not rename snapshots to peaks. |
| Freeze | Land accepted runtime changes, update the dependency pin, run broad correctness and one local end-to-end qualification, then freeze the collection branch. No hardware-specific winning constants or new broad test matrix. |

The existing scalar-GEMM staging prototype also hit an independent tight f64
oracle failure shared by its incumbent. It is not qualified for promotion;
do not relax that oracle merely to add more search candidates.

## Manuscript changes made now

- Treat RPL-U GPU availability and intermittent Windows PyTorch failure as
  positive deployment evidence for Meganeura. Keep unproved crash root causes
  distinct from observed execution failures. The RPL-U numbers are explicitly
  GPU-versus-CPU; the former phrase denied GPU-versus-GPU superiority but was
  unnecessarily confusing.
- Describe MI300X as a compute-focused accelerator without a usable Vulkan
  driver stack, not proof of a fundamental tensor-compute limitation. RADV
  support would be engineering work; neither a short fix nor its cost is proven.
- Remove obsolete 780M/Whisper and preceding-installation/version narrative.
  Keep relevant separately pinned experiments self-describing.
- Show cumulative preparation cost, including 3.23 hours of H100 searched
  PyTorch compilation versus 125 seconds for Meganeura, with capture and
  research qualification separate. A failed searched condition is an
  availability result, not merely a threat to a speed ratio.
- Clarify the H100 transfer-without-prior-device-optimization interpretation,
  hybrid graphics/ML contexts, Inferena's cross-framework purpose, and normal
  CPU power management with a measured latency effect but no energy claim.
- Fix strict/accelerated table subheaders; add generated SmolLM2 absolute-time
  bars. Use paired bars because prefill and F+L+B are independent measurements:
  stacking them, or wall/CPU/GPU intervals that overlap, would invent a partition.
- Add memory accounting with explicitly different scopes, not a false peak-VRAM
  comparison. SGD already expands to stochastic gradient descent in Section II.

## Source checks

- [Pinned XPU graph implementation](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/torch/xpu/graphs.py).
- [ROCm/PyTorch API semantics](https://docs.pytorch.org/docs/main/notes/hip.html).
- [Pinned MPS API](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/torch/mps/__init__.py)
  and [Inductor backend](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/torch/_inductor/codegen/mps.py).
- [Mesa's supported RADV hardware](https://docs.mesa3d.org/drivers/radv.html#supported-hardware).
- [Reproducible optimization studies](../../docs/experiments.md).

Runtime/collector changes and on-device qualification are **not completed**
by this manuscript revision. Do not distribute a new collection command yet.
