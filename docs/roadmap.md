# Roadmap

Strategic plan for making meganeura faster while preserving its core
property: **no hand-written kernel per model pattern** — everything
composes from the four archetypes (pointwise, reduction, matmul,
attention) plus e-graph fusion.

## Positioning

The benchmark picture (see README / [inferena.tech](https://inferena.tech)):

- **Apple / AMD / Intel**: leading PyTorch on transformer inference and
  training. Hold the lead while closing the remaining losses.
- **NVIDIA**: close to parity on inference; training still trails CUDA.
  NVIDIA is a first-class target — the goal is parity-to-lead there,
  not just on the platforms PyTorch neglects.
- **Known losses**: ResNet-50 (conv-heavy), Whisper-tiny (small
  dispatches, CPU-overhead-bound), NVIDIA training (tensor cores
  largely unusable for f32 gradients).

Everything below is organized so each track attacks one of those
losses or removes a structural ceiling. Items marked **[research]**
have open questions; the rest are engineering with known shape.

Ground rules accumulated from past work (do not re-litigate without
new hardware or new data — see `rejected-optimizations.md` and
`perf-pipeline-stats-retrospective.md`):

- CPU command encoding is **not** a bottleneck — measured negligible.
  Command-buffer reuse is a non-goal (blade doesn't expose it, and
  there's nothing to win).
- Cooperative matrix on Vulkan **and** Metal already ships through
  Naga (`coop_mat`, subgroups, f16) and is load-bearing in matmul,
  flash attention forward, and both flash backward kernels. It is not
  a TODO; the only missing *type* is bf16 (see C2).
- Dynamic shapes need no new GPU machinery (no indirect dispatch):
  the command stream is re-encoded every iteration and shapes flow
  through uniform params, so runtime shapes already have everywhere
  they need to go. The only cost of a shape change is re-running plan
  emission, which is cheap; pipelines are shape-agnostic and already
  carry over.
- GPU-time wins on dispatches under ~50 µs routinely lose on
  wall-clock. Always judge by `step()` wall-clock.
- Register-count-driven cost models don't track wall-clock. Bytes
  moved does.

---

## Track A — Compiler: make the e-graph earn its keep

### A1. Repeated-block outlining (hierarchical IR)  ← *first stage landed*

*Landed:* signature-periodicity detection + edge-isomorphism
verification (`src/outline.rs`), per-block saturation with extraction
on block outputs, `OptimizeReport.outlined_regions`. *Still open:*
applying rewrites by stamping extracted terms per instance (today the
global pattern appliers do the rewriting, so per-block extractor
decisions can't yet disable a fusion for out-of-region patterns), and
cross-block boundary fusions.

**Problem.** Graphs over 300 nodes skip egglog entirely
(`optimize.rs:186`) and fall back to direct pattern matching — every
real training graph (SmolVLA ≈ 750 nodes) never sees equality
saturation. Saturation cost is superlinear in node count, but the
graphs are *repetitive*: N identical transformer blocks differing only
in parameter bindings.

**Plan.**
1. Detect repeated subgraphs structurally (hash each node's op +
   input-structure; blocks with identical hashes and identical
   internal topology are instances of one template). Model builders in
   `src/models/` can also annotate block boundaries explicitly to
   bootstrap, but detection must work on imported ONNX/NNEF graphs to
   preserve generality.
2. Outline one block instance into a sub-graph, run the full egglog
   pipeline on it (well under the 300-node cutoff), then stamp the
   optimized body back into every instance with per-instance parameter
   substitution. The existing `DerivedParam` machinery already handles
   per-instance derived weights (e.g. SwiGLUConcat creates one derived
   buffer per layer).
3. Run a final cheap pass over the stitched graph for cross-block
   fusions (block-boundary pointwise chains).

**Touchpoints.** `optimize.rs` (new outlining pass before
`graph_to_egglog`), `graph.rs` (subgraph extraction/substitution).

**Validation.** `fusion_preserves_outputs` test extended to a 2-block
and 16-block synthetic model; assert the 16-block graph reports
egglog_time > 0 in `OptimizeReport` (i.e. saturation actually ran) and
identical fusion counts per block.

**Wins.** E-graph coverage on flagship workloads; compile time drops
superlinearly; smaller pipeline population (identical blocks share
pipelines — they mostly already do via shape-keyed caches).

### A2. Single owner for kernel-variant selection

**Problem.** Geometry is chosen in `compile.rs` (dispatch emission),
then *rewritten* at session build (`runtime.rs:1796-1906` coop and
small-tile selection), while pipeline lookup (`Pipelines::get`,
`runtime.rs:1130`) makes its own independent decision. The
disagreements are documented hazards: epilogue+coop is disabled
because the lookup returns the scalar epilogue pipeline before
checking `use_coop` while the geometry was already rewritten for coop
(`runtime.rs:1734-1742`) — so every fused-epilogue matmul is stuck on
scalar geometry.

**Plan.** Make session build produce, per dispatch, an atomic
`(pipeline key, geometry, buffer padding requirement)` in one pass
that has access to GPU caps. `Pipelines::get` becomes a pure map
lookup with no fallback hierarchy. Then enable the blocked variant:
coop matmul with PointwiseDAG store-phase epilogue (the codegen
already supports emitting the DAG body; it's the selection layering
that blocks it).

**Touchpoints.** `runtime.rs` (variant selection pass, `Pipelines`),
`codegen.rs` (coop store epilogue emission).

**Validation.** Existing correctness suite; new test asserting
epilogue dispatches take the coop path when caps allow; benchmark
SmolVLA / SmolLM2 train (every transformer MLP has a fused bias/act
epilogue).

### A3. Bytes-moved cost model for extraction  ← *landed*

*Landed:* e-class sizes are mapped after saturation (every node
binding's value → tensor bytes) and `FusionCostModel` charges each
e-node its read+write traffic; on under-cutoff graphs the extracted
terms gate the appliers. `MEGANEURA_NO_TRAFFIC_COST=1` reverts.

**Problem.** `FusionCostModel` is constant (fused = 9, everything
else = 10) — `kernel-archetypes.md` *describes* an HBM-traffic cost
model that the code doesn't implement. The extractor can't distinguish
a fusion that saves 100 MB of traffic from one that saves 1 KB, and
per-pattern judgment calls live in env vars instead of the model.

**Plan.** Shapes are static at extraction time. Thread a
`NodeId → TensorType` map through `graph_to_egglog` (encode the
output shape as a term argument or side table keyed by the leaf/op
identity), and make the fold compute
`cost = bytes_read + bytes_written` per op. Fusion then wins by
exactly the intermediate traffic it eliminates. Calibration constant
per archetype only if measurements force it; **no register counts**
(see ground rules).

**Touchpoints.** `optimize.rs` (cost model, egglog encoding).

**Validation.** Unit tests: extractor prefers SwiGLUConcat on a large
MLP but not on a degenerate 1-element one; fusion decisions on the
bench models unchanged or justified by measurement.

**Unlocks.** A4-style future rewrites (layout, Winograd-vs-GEMM,
remat) all need a real cost signal to be selected per-shape.

---

## Track B — Memory: plan it, then spend it on speed

### B1. Buffer lifetime analysis + aliasing  ← *landed*

**Problem.** Every logical buffer — activations, gradients, LSE
buffers — gets a dedicated allocation that lives for the whole
session (`runtime.rs:1945`). Training memory is O(all activations +
all gradients) forever, which caps model size on the 8–16 GB GPUs the
project targets, and is the prerequisite blocker for rematerialization
(B3).

**Plan.** The dispatch sequence is static and already partitioned into
barrier groups, so this is linear-scan allocation:

1. At session build (after coop padding upsizes and dispatch
   reordering, before buffer creation), compute per-buffer live
   intervals **at barrier-group granularity** — dispatches within a
   group run concurrently, so intervals must be disjoint with strict
   inequality across groups (a barrier must separate the old tenant's
   last use from the new tenant's first write).
2. Pin (never alias): params, inputs, constants, outputs, loss,
   gradients (read by runtime-encoded optimizer/clip passes after the
   plan's dispatches), derived params, quantized weight buffers,
   anything externally bindable, cross-step state (KV caches =
   `CacheWrite` outputs), read-modify-write outputs (`ScatterAdd`),
   and any buffer whose first use is a read (live-in from a previous
   step).
3. Greedy best-fit assignment of remaining intermediates onto a pool
   of physical allocations; allocation size = max over tenants.
4. Kill switch `MEGANEURA_NO_ALIAS=1`; savings reported in
   `MemorySummary` and the build log.

**Coop-padding safety.** Only matmul **output** and **addend** buffers
are padded for full-tile coop stores (`runtime.rs:1857-1873`); A/B
staging is bounds-guarded. Coop stores write the *full* padded region
every step, so a reused allocation's stale bytes are overwritten
before any read; addend padding garbage lands only in discarded output
padding. The zero-fill-at-alloc guarantee therefore still holds per
*physical* allocation. This reasoning must live next to the code and
be re-checked if a kernel ever gains unguarded reads.

**Validation.** Pure unit tests on synthetic plans (disjointness,
pinning, max-sizing); full GPU suite (gradcheck, training_correctness,
smollm2/whisper/resnet correctness) with aliasing on vs off; memory
summary deltas on the bench models.

### B2. Device-local memory for intermediates  ← *implemented, opt-in pending bench*

*Landed:* `MEGANEURA_DEVICE_LOCAL=1` places allocations whose tenants
are all step-local intermediates (the non-pinned set from B1's
analysis) in `Memory::Device`, zeroed by one GPU transfer pass at
session build. Default-off until the dGPU bench numbers are in; flips
to default (or is removed) based on RTX 5080 / Radeon measurements.

**Problem.** Every buffer is `Memory::Shared` (host-visible,
host-coherent — `runtime.rs:1954`). Free on UMA (Apple, iGPUs). On
discrete boards this places all traffic in the ReBAR heap, whose
GPU-side caching behavior varies by vendor/driver — a plausible
contributor to the remaining NVIDIA training gap, and cheap to test.

**Plan.** Only inputs, outputs, loss, and params need host visibility
(params could even go device-local with a one-time staging upload).
Add an allocation-class decision per buffer (host-visible vs
`Memory::Device`), keyed off the same pinning analysis as B1.
Readback paths (`read_buffer`, grad clipping) need staging for
device-local buffers — keep grads host-visible initially to avoid
touching the clip path.

**Validation.** Measure `step()` on RTX 5080 and Radeon (dGPU) for
SmolLM2/SmolVLA train, before any code is final — if the win isn't
real, close the item with a note in `rejected-optimizations.md`.

### B3. [research] E-graph rematerialization (activation checkpointing)

**Problem.** No checkpointing exists; trainable model size is bounded
by storing every activation. Classic checkpointing APIs (manual block
annotations) are against the project's grain.

**Idea.** Store-vs-recompute is an extraction problem: extend the
e-graph with explicit `Remat(x)` equivalences and let a bytes-moved ⊕
memory-pressure cost model (from A3, with the live-range data from B1)
choose. Nobody ships equality-saturation-driven remat for GPU
training; this is the project's most distinctive research angle, and
it differentiates from Luminal (inference-only e-graphs).

**Open questions.** Cost model needs a memory *budget* term, not just
traffic (extraction under constraint — possibly ILP extraction on the
small outlined block from A1 rather than the whole graph). Interaction
with barrier-group concurrency. Start after A1 + A3 + B1 land.

---

## Track C — Precision: tensor cores for training

### C1. Error-compensated f16 coop matmul **(highest expected win)**

**Problem.** The biggest measured-and-blocked win in the repo:
lowering the coop threshold took SmolLM2 train 44 ms → 29 ms but
failed gradcheck — Ampere's Vulkan path exposes only f16 coop tiles,
and f16 staging loses too much precision across 30+ layers
(`rejected-optimizations.md`, "Lower coop matmul threshold").

**Plan.** Split-precision GEMM (Ootomo & Yokota 2022): stage A as
`a_hi: f16` plus residual `a_lo: f16` (same for B), accumulate
`hi·hi + hi·lo + lo·hi` in f32 — ~3× the MMA work for near-f32
accuracy, still far ahead of scalar ALUs. Implement as a third matmul
variant in the coop template family (`matmul_coop.wgsl` ancestry);
gate on a precision-sensitive context flag (training graphs) while
inference keeps plain f16 staging. Start with the K-heavy backward
matmuls (`MatMulAT`) where the threshold experiment showed the
largest win.

**Validation.** `grad_check` end-to-end finite differences (the test
that killed the naive version) across ≥ 30-layer configs; loss-curve
parity on SmolLM2 train; wall-clock on RTX 5080 *and* RDNA3.

**Risk.** 3× MMA cost eats the margin on staging-bound shapes — but
the same experiments showed staging dominates ~99% of time, so MMA
headroom exists. If hi·lo terms still lose gradcheck, fall back to
compensated accumulation only for K > threshold.

### C2. bf16 cooperative matrix

Coop matrix and subgroups already ship through Naga on Vulkan and
Metal — the gap is purely the **bf16 type**: `VK_KHR_shader_bfloat16`
+ the bf16 cooperative-matrix formats in Vulkan, `bfloat` in MSL 3.1.
bf16's dynamic range removes the loss-scaling problem that f16
staging has, potentially making C1's residual trick unnecessary on
hardware that exposes bf16 tiles. Requires plumbing through naga and
blade first (upstream work we're well-positioned to do), then it's a
`CoopConfig` variant here. Sequence after C1 — C1 works on today's
naga.

### C3. Mixed-precision training recipe

Framework-level complement to C1/C2: f16 (later bf16) activations +
f32 master weights (the `ToF16` straight-through op exists), dynamic
loss scaling, optional stochastic rounding on the weight update.
Halves activation traffic — and staging/memory is ~99% of kernel time
by our own measurements, so this pays even where tensor cores don't.
Design the policy so the *graph* stays precision-annotated (dtype on
nodes) rather than a global flag, so the e-graph can see and rewrite
casts. Validation: gradcheck with tolerance bands, loss-curve overlay
vs f32 baseline on SmolLM2 + SmolVLA.

---

## Track D — Small-dispatch latency (Whisper-tiny, decode)

Per the retrospective: below ~50 µs/dispatch, submission and
synchronization dominate. Encoding is *not* the cost (ground rules) —
the dispatch *count* and the barriers between groups are.

### D1. Horizontal fusion (dispatch batching)

Independent same-shape dispatches inside one barrier group (Q/K/V
projections; per-head pointwise) still bind and launch separately.
Batch them: one dispatch with a workgroup-ID-routed switch over 2–4
op bodies, or for matmuls a batched variant (already have `batch` in
workgroups[2] — extend to heterogeneous B operands). Target: cut
Whisper-tiny's dispatch count per step by ≥ 30%. This is a compile.rs
pass over existing barrier groups + one codegen template — fully
general, no per-model work.

### D2. [research] Persistent decode megakernel

For M=1 autoregressive decode the whole transformer step could run as
one (or a few) persistent dispatches with a device-side work queue —
the structural end-state for dispatch overhead, and where recent LLM
serving work is converging. Hard under WGSL/Vulkan forward-progress
rules; the tractable first cut is a **persistent layer-loop kernel**:
one dispatch iterates layers for the GEMV-shaped decode path, with
workgroup-scope barriers (all weights pre-bound via one bind group;
shapes uniform across layers). Prototype on SmolLM2 decode where the
GEMV path already exists. High risk, high distinctiveness; measure
against D1 first to know how much is left on the table.

---

## Track E — Conv/vision gap (ResNet-50)

### E1. Layout as an e-graph decision (NHWC)

Conv kernels are NCHW-flat only. Implicit-GEMM conv wants
channels-innermost coalescing; most frameworks moved to NHWC for
exactly this reason. Rather than hand-tuning NCHW kernels, introduce
layout-tagged tensor types and `ToNHWC/ToNCHW` rewrites in the
e-graph with transform costs from A3, letting extraction place layout
conversions optimally (typically: once at input, once at output).
Archetype templates get a layout parameter — still zero hand-written
per-model kernels. Validate on ResNet-50 fwd+train and EfficientNet
inference vs the NCHW baseline.

### E2. Re-measure the parked conv/coop items on current hardware

`Conv2dGradInputGemmCoopV2` lost on wall-clock from 14× workgroup
count on a 20-SM RTX 3050; split-K and double-buffered staging were
rejected with explicit "would help on bigger GPUs" notes. We now have
RTX 5080 (84 SMs) and Radeon 890M in the bench fleet. One day of
re-measurement, three potential reversals — do this before investing
in new conv kernels. Whatever flips gets promoted from env-var opt-in
to cap-keyed default; whatever doesn't gets its rejection note
updated with the new data.

---

## Track F — Session-build autotuning (in-memory only)

**Problem.** Tile sizes (64/32), `flash_ept_cap` (32), coop workgroup
thresholds (16/32), GEMV switchover — all fixed heuristics, tuned
mostly on one GPU, applied to Apple/RDNA/Intel/NVIDIA alike.

**Plan.** At session build, for the handful of distinct
(archetype, shape-class) pairs in the plan, run a micro-search over
the small knob set (2–4 values each), measuring real `step()`-context
wall-clock, and pick winners. **No on-disk cache** — the tune budget
must simply be small relative to a training run (target: < 2 s for a
SmolLM2-class plan; the archetype design keeps the space tiny).
In-memory only, per session; inference sessions can opt out
(`SessionConfig`) to protect TTFT.

**Lessons honored.** This is wall-clock tuning, not register-stats
tuning (that was tried and removed); per-kernel × per-cap
combinatorial tuning is what made the old `auto_tune_flash_ept` take
30 s — bounding the space per shape-class is the fix.

---

## Track G — Quantized fine-tuning (QLoRA-style)

Q4/Q8 weight formats, derived-param plumbing, and the training stack
already coexist. Frozen quantized base + trainable f16/f32 LoRA
adapters = fine-tuning SmolLM2/Gemma-class models on Radeon 780M /
Apple M-series — a capability no other framework offers, and squarely
on the wedge. The quantized-weights-disable-coop restriction is
acceptable here (LoRA matmuls are small; base-model matmuls are
inference-shaped). Needs: LoRA graph helper in `nn`, gradient flow
around frozen quantized params (StopGradient exists), an example +
bench. Mostly assembly of existing parts; good first "product"
milestone after B1/C1.

---

## Sequencing

```
Now        B1 aliasing ──→ B2 device-local ──→ B3 remat [research]
           A1 outlining ──→ A3 bytes-moved cost ─┘
Next       A2 variant-selection unification (unlocks coop+epilogue)
           C1 split-f16 coop ──→ C3 MP recipe ──→ C2 bf16 (after naga/blade)
Parallel   E2 re-measures (1 day, hardware exists)
           D1 horizontal fusion
Later      F autotuning · E1 layouts · G QLoRA · D2 megakernel [research]
```

Success metrics (tracked via the existing bench harness + CI latency
baseline): SmolLM2/SmolVLA train ms/step on RTX 5080 vs PyTorch CUDA
(target: parity), ResNet-50 and Whisper-tiny flipped to wins, max
trainable model size at fixed VRAM (B-track), and zero regressions on
`bench_ci_latency` against `bench/ci_latency_baseline.json`.
