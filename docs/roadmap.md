# Roadmap

Strategic plan for making Meganeura faster while preserving reusable,
operator/shape-driven specialization rather than per-model kernels. The
organizing families are pointwise, reduction, matmul, convolution and attention.
Handwritten templates still exist; consolidating them into generators is
ongoing.

## September 2026 priorities

The [study guide](study/README.md), [audit](audit-2026-09.md) and
[performance plan](study/performance-plan.md) are the current decision record.
The detailed tracks below retain earlier motivation and experiment notes;
historical timings and rejected ideas are not current universal guarantees.

The frozen GPU-reference matrix has 12/20 strict minimal-latency wins but a
1.78× median valid training-time ratio. Discrete AMD is strong; Apple training
still loses to eager MPS. Intel is compared with a labeled CPU fallback.
See [results](study/results.md) for denominators and numerical gates. Do not
mix exploratory Inferena history with publication-grade results.

State-isolated exact-class search is implemented for the existing 32/64 scalar
and legal native-f32 cooperative matmul tiles. It is default-off. Scalar GPU
qualification passed on RTX 5070; native-f32 hardware and fleet qualification
remain ahead. A separate harness checks transfer to whole-step time;
automatic confirmation, f16-input/complex-fusion search and persistence remain open.
See the [implementation contract](study/performance-plan.md).
The [five-process synthetic transfer pilot](experiments/tuning-2026-09-05/README.md)
shows repeatable 1.15×/1.13× whole-step gains on two larger chains, with no
selection change on two smaller ones. It is not model/fleet qualification.

The subsequent [six-case inference/training holdouts](experiments/holdouts-2026-09-06/README.md)
retain five processes, full control-session gradients/moments and matched Adam
updates through step 78. All numerical checks pass, but no whole-step gain or
regression clears the predeclared guard. The unchanged ResNet control exposes
timing drift, and only 1/512 of its plan dispatches is eligible for matmul search.
The [controlled six-process crossover](experiments/crossover-2026-09-06/README.md)
now confirms the dense chain in both session orientations (median 1.177×),
but not MLP+Adam; two MLP A/A controls fail the noise screen. ResNet remains
unchanged, with 98.26–98.65% of search time spent in qualification. A checked
selection-only swap preserves tensor/optimizer state through role reversal.
The [readback follow-up](experiments/readback-2026-09-06/README.md) now separates
qualification's CPU copies/checks from transfer/dispatch/wait, using published
Blade 0.9/Naga 30 (Rust 1.92). Across six paired processes, read-optimized
private staging lowers median ResNet search from 606 to 39 ms: CPU readback
allocation/copy falls from 582 to 2 ms while unchanged validation stays about
6 ms. All 108 class comparisons qualify and full state is bit-exact through
Adam step 178. Download becomes the private staging default; Shared remains
available. Preparation and transfers cost more, and one dense search is slower
overall; every attempt is retained. This is not a whole-step speed result.
The [allocation/reuse follow-up](experiments/staging-reuse-2026-09-06/README.md)
locates that preparation cost in staging allocation and tests one exact-size
slot retained only within a tuning call. Six process pairs lower median dense
search 44.01→31.84 ms and MLP 64.27→45.46 ms; ResNet has no reuse opportunity
and no guarded change. SameSize becomes the option default, with identical
scratch requests, zero retention at return and all validation intact.
The [whole-step profiles](experiments/training-profile-2026-09-06/README.md)
now consistently put 60.66–60.77% of ResNet's instrumented dispatch time in
backward convolution, and 36.25–40.58% of SmolLM2's in backward attention.
All 45 profiled full states match, but short-case timing drift is substantial;
these optimizer-free profiles are not whole-step speedup evidence. The follow-up
also fixes dX indexing outside same padding and adds full f64 derivative oracles.
Exact-class scalar convolution-derivative tile selection is now implemented,
with full NCHW keys, physical scratch sizes, batch-aware references and unchanged
precision/budgets. The [corrected six-process crossover](experiments/conv-tiles-corrected-2026-09-06/README.md)
observes ResNet 17.5808→16.7293 ms (median ratio 1.05056×), but all six decisions
remain inconclusive under the unchanged 5%+noise guard. Four dX classes change
eight dispatches; the small Adam/SGD chains keep their tiles and make real updates.
Original malformed flat-layout controls are retained/disqualified; public
operand checks, full forward oracles and nonzero training preflights prevent
their zero-data success from becoming evidence. Sampling now dominates search
cost, and the eight-class structural prior visits only 8/45 ResNet classes.
The [indexing repair](experiments/conv-indexing-2026-09-06/README.md) replaces
unsafe f32 reciprocal decomposition with integer division in the shared scalar
and generated cooperative kernels. Adversarial full forward/dX/dW GPU oracles
reproduce the old width-41 error and pass at widths/kernel sizes 41/47/55 after
repair. Four uniforms and the obsolete tuner-only filter are removed, without
weakening numerical or overflow checks. Its cost check is separate from tuning.
Next add bounded split-K weight gradients
with charged partial storage and whole-step confirmation;
attention schedules remain a separate device-qualified family.
Keep tuning opt-in; do not fit a smaller margin or model rule.

The [checkpoint/memory follow-up](study/checkpoints-and-memory.md) implements
logical format-3 saves and preflighted restores, lazy Adam/LaProp state, and
resident tensor-buffer accounting. RTX tests cover malformed late fields,
different allocation padding, the next optimizer update, and an 8 MiB unused
moment allocation avoided for 1,048,576 F32 elements. Peak driver memory and
cross-backend restore behavior still need qualification.

The next sequence is:

1. State-safe tuning, exact variant contracts and stronger numerical evidence.
2. Bounded bidirectional shape-level search; thresholds become search priors.
3. Reusable convolution-derivative and Metal attention schedules.
4. Persistent winners with device/driver/compiler provenance; fleet/peak-memory
   qualification of the implemented lazy optimizer and logical checkpoints.
5. Layout/rematerialization and new precision formats only with an observed
   need, capability support, correctness evidence and an amortization budget.

Existing profiles prioritize work but do not establish that command encoding,
API overhead or register pressure can never matter. Use whole-step time for
acceptance and profiles for localization. Cooperative backward is experimental
and default-off; compensated f16 is not a safe automatic derivative path.
General dynamic shape replanning is not yet exposed by the session API.

The initial audit performed no GPU benchmarking. The user subsequently
released the device for qualification and experiments. Timings in the older
tracks below remain historical, not rerun results.

---

## Track A — Compiler: make the e-graph earn its keep

### A1. Repeated-block outlining (hierarchical IR)  ← *landed*

*Landed:* signature-periodicity detection + edge-isomorphism
verification (`src/outline.rs`), per-block saturation with extraction
on block outputs, `OptimizeReport.outlined_regions`, and extracted
terms stamped per instance — the e-graph's per-site decision is what
runs; the global pattern appliers are gone. Nodes outside regions are
chunked into under-cutoff windows and saturated too, so every node
passes through the e-graph. *Still open:* cross-block boundary
fusions (segment edges are opaque leaves).

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

### A2. Single owner for kernel-variant selection  ← *partially landed*

**Original problem.** Geometry was chosen in `compile.rs`, rewritten
again during session construction, and independently interpreted by
pipeline lookup. The historical epilogue+coop path could therefore
run a scalar epilogue pipeline with cooperative geometry and leave
output rows unwritten.

**Landed.** Cooperative matmul now has a guarded PointwiseDAG
store-phase epilogue. Pipeline maps are keyed by the actual DAG, and
runtime promotion requires an epilogue form the cooperative generator
supports. Accumulator tiles are staged through workgroup memory before
guarded scalar lanes evaluate the DAG. A 1024×16 by 16×128 GPU
regression test verifies both cooperative selection and output parity.

*Also landed:* runtime variant decisions have a single owner —
`runtime::select_variants` performs cooperative promotion (including
generated conv kernels and buffer padding), the RmsNorm→matmul
prologue fusion, and small-tile demotion in one pass, records a
`scalar_fallback` per promoted dispatch, and the flash EPT knobs
travel as plan data (`TuningKnobs`) instead of ambient globals — the
generated WGSL and dispatch geometry can no longer disagree.

**Remaining plan.** `Pipelines::get` becomes a pure map lookup with no
fallback hierarchy. Extend cooperative epilogues beyond the current
unary/no-extra-binding subset only when an active graph and profile
justify each form.

**Touchpoints.** `runtime.rs` (variant selection pass, `Pipelines`),
`codegen.rs` (coop store epilogue emission).

**Validation.** The existing correctness suite and the new cooperative
epilogue test pass. Current SmolLM uses packed SwiGLU and has no
epilogue-bearing matmul at the relevant sites, so the specialization
does not recover the historical result. Profile the current graph
before broadening the path.

### A3. Bytes-moved cost model for extraction  ← *landed*

*Landed:* e-class sizes are mapped after saturation (every node
binding's value → tensor bytes) and `FusionCostModel` charges each
e-node its read+write traffic; extraction decides every rewrite and
its terms are stamped into the graph (the constant-cost fallback and
`MEGANEURA_NO_TRAFFIC_COST` are gone).

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

### B2. Device-local memory for intermediates  ← *landed, default-on*

Parameter gradients, Adam moment buffers, the clip accumulator, and
temporal grad-accumulators also use `Memory::Device` by default
(host diagnostics go through the existing staging readback/upload).

*Landed:* allocations whose tenants are all step-local intermediates
(the non-pinned set from B1's analysis) use `Memory::Device`, zeroed by
one GPU transfer pass at session build. This is default-on;
`MEGANEURA_NO_DEVICE_LOCAL=1` restores host-visible allocations for
diagnosis. Continue tracking the effect on both RTX and Radeon release
benchmarks because ReBAR and cache behavior remain driver-dependent.

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

### B2a. Lazy optimizer state and logical persistence  ← *implemented, RTX qualified*

Keep logical parameter types separate from physical capacities. Format-3
checkpoints omit padding and validate every record before writes or moment
allocation. Cache format 5 invalidates plans lacking the metadata. Legacy
files keep physical/partial-load compatibility with preflighted writes.

Adam/LaProp moments are allocated on first configuration, write, explicit
step or applicable restore. SGD/F+L+B sessions no longer reserve them; reads
of uninitialized state return zeros without GPU allocation. Clearing the
optimizer retains existing state. Memory reports separately count graph,
moments, accumulators and auxiliary buffers; the previous accumulator total
incorrectly multiplied by the parameter count. Update, clip and accumulation
loops use logical lengths; poisoned tails cannot affect optimizer results or
overrun logical-sized grouped diagnostics. Raw F32 reads are capacity-checked.

**Still open:** driver-peak sampling on larger workloads, Metal↔Vulkan restore
qualification, structured allocation failures, crash-atomic checkpoint file
replacement, and a separately designed complete training-loop snapshot.
See the [contract and tests](study/checkpoints-and-memory.md). This does not
implement activation rematerialization or reduce Adam's required moment size.

### B3. [research] E-graph rematerialization (activation checkpointing)

**Problem.** No activation checkpointing exists; trainable model size is bounded
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

### C1. Error-compensated f16 coop matmul  ← *automatic use rolled back*

The generator validates, but `3326f39` removed automatic compensated
derivative selection and `f2a0108` pins tiny-gradient correctness. Both hi and
lo can underflow: mantissa compensation does not preserve exponent range.
`Auto` uses native-f32 tiles or scalar f32 for protected derivatives.
`AllowF16` explicitly relaxes that contract. The proposal below is historical,
not a description of current defaults or established near-f32 accuracy.

### C1 (historical)

The original write-up:

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

Cooperative matrices and subgroups already have a Naga/Blade path. bf16 needs
end-to-end type, advertised tile, feature-bit and backend support, followed
by numerical qualification. Its f32-like exponent range helps the underflow
problem but its shorter mantissa is not f32 accuracy. This is research, not
a one-enum implementation task or a prerequisite blocked on completing C1.
See the [precision plan](study/performance-plan.md).

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

### D1. Horizontal fusion (dispatch batching)  ← *landed (matmul)*

*Landed:* same-A `MatMul` / `MatMulAT` / `MatMulBT` siblings in one
barrier group (up to 3, typically Q/K/V) become one dispatch.
Pointwise/conv packing is still open.

### D1 (historical)

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

## Track F — The representation menu: empirical kernel selection

**End state.** No hand-maintained performance heuristics for kernel
variants. Instead, a per-device **menu of legal representations** and a
measured choice among them:

1. **Legality** — the menu is derived from capabilities: the tile
   shapes/dtypes the driver advertises, subgroup support, f16. Facts,
   registered once per device, never guessed.
2. **Validity** — declarative numeric constraints filter the menu per
   value: `requires_full_precision` (autodiff-marked) × the format each
   representation stages through. This stage can never be empirical —
   the canonical cautionary tale is the coop-threshold experiment that
   went 44 ms → 29 ms *and failed gradcheck*
   (`rejected-optimizations.md`): a purely measured selector picks the
   fast wrong kernel every time. Validity is checked by parity/gradcheck
   tests, not by timing.
3. **Speed** — whatever survives 1–2 is chosen by measured
   `step()`-context wall-clock, per (archetype, shape-class), applied to
   every instance of that class. The remaining hard-coded thresholds
   (coop workgroup minimums, the Apple >1024 veto, small-tile cutoff,
   GEMV switchover) demote from *deciders* to *search priors* — the
   starting configuration when tuning is off or budget-limited.

**Landed.** Variant decisions have one owner (`select_variants`) and codegen
knobs travel as `TuningKnobs` plan data. The September first cut of
`Session::tune` / `tune_with` measures scalar and native-f32 cooperative tiles per exact eligible
class, with private scratch, numerical qualification, interleaved samples and
explicit budgets/reports. It replaces the earlier family-wide cooperative
demotion tuner, which ran live steps and mutated training/cache state.
Default-off. Scalar correctness/state GPU tests passed on RTX 5070. Native-f32
shaders pass offline validation; that device cannot qualify their execution.

**Remaining plan.**
1. *Fleet qualification and whole-step confirmation* before default-on;
   preserve the implemented scratch-state isolation contract.
2. *Broader exact-class candidates:* f16-input cooperative, packed/complex-fused
   and GEMV implementations need complete precision/padding/binding contracts.
   Native-f32 matmul already challenges profitability thresholds where its
   complete candidate fits existing allocations.
3. *Recompiling knobs:* EPT caps / tile sizes / generated-kernel
   workgroup sizes are plan data now, so a candidate is a plan rebuild
   (~seconds). Bound the space per shape-class (2–4 values per knob).
4. *Persist measured winners* with a stronger performance identity: device,
   driver, backend/compiler/generator revision, numerical policy and validation
   provenance. Semantic plan caching is not already a tuning database;
   invalidation is not free or solved by capability bits alone.

**Non-goals.** Representation choice stays *out of the e-graph*: the
e-graph is semantic and device-free, and `coop(x) ≡ scalar(x)` is not a
true congruence when staging dtypes differ. The e-graph picks *what* to
compute (extraction by bytes-moved, validated offline by benches); the
plan/runtime layer picks *how* (measured per device). Coupling them is
how the register-cost experiment failed — `optimize` runs before a GPU
exists in many paths.

**Lessons honored.** Batched scratch wall-clock in the first slice, not an
individual timestamp or register-count prediction. It still needs whole-step
confirmation because batching/barriers/cache reuse change context. The old
`auto_tune_flash_ept` took 30 s because it tuned per-kernel × per-cap —
shape-class bounding is the fix.

---

## Track G — Quantized fine-tuning (QLoRA-style)

Q4/Q8 weight formats, derived-param plumbing, and the training stack
already coexist. Frozen quantized base + trainable f16/f32 LoRA
adapters = fine-tuning SmolLM2/Gemma-class models on Radeon 780M /
Apple M-series. MLX-LM already offers LoRA/QLoRA on Apple; the opportunity is
a common Vulkan/Metal path, not uniqueness. See the sourced
[alternatives comparison](study/alternatives.md). The quantized-weights-disable-coop restriction is
acceptable here (LoRA matmuls are small; base-model matmuls are
inference-shaped). Needs: LoRA graph helper in `nn`, gradient flow
around frozen quantized params (StopGradient exists), an example +
bench. Mostly assembly of existing parts; good first "product"
milestone after memory and numerical-contract work.

---

## Historical sequencing

This earlier sequence is retained to explain the track dependencies, not as
the current priority order. In particular, C1's automatic use was rolled back
and safe autotuning is now an early priority rather than a final add-on.

```
Now        B1 aliasing ──→ B2 device-local ──→ B3 remat [research]
           A1 outlining ──→ A3 bytes-moved cost ─┘
Next       A2 remaining variant-selection unification
              (safe unary coop+epilogue landed)
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
