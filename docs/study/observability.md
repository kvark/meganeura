# Observability and debugging: the honest eager-PyTorch comparison

The short workshop answer: **eager PyTorch has the more convenient interactive
debugging model and the richer ecosystem.** Meganeura trades immediate operator
execution for a compiled plan, then restores useful visibility through named
graph values, dispatch provenance, materialized debug sessions, shader/plan
dumps, numerical probes and structured profiles. These are real facilities,
but they are not a Python debugger, an arbitrary backward-hook system, or a
complete GPU anomaly detector.

This chapter describes current source, including the September f32-matmul
tuner. It is not a result from the frozen paper. Run GPU examples only on an
available device; instrumented/tuning executions can disturb other timings.

## Compare the debugging models, not just the feature names

| Question | Eager PyTorch | Meganeura today |
|---|---|---|
| Where does execution happen? | Tensor operations execute as Python runs; autograd records the backward graph along the way. A Python breakpoint exposes the current program and tensor objects. | Graph builders describe work. `build` compiles it; `step` submits it. Rust breakpoints inspect builders/compiler/runtime, not a suspended WGSL invocation. |
| Can I inspect an intermediate? | Keep its tensor reference and inspect it; reading a device value can synchronize. | Name the node and use `read_node_by_name` after `step` and `wait`. A fused-away value or reused allocation may be unavailable. Debug options and explicit graph outputs help. |
| Can I inspect gradients? | `.grad` for eligible tensors, `retain_grad()` for non-leaves, and tensor backward hooks. | `read_param_grad` and bulk norm reads expose parameter gradients; moments and Adam's counter have read APIs. No equivalent general runtime hook on every differentiable intermediate. |
| Where did a bad value originate? | Autograd anomaly detection can associate a failing backward function with its forward traceback and reject NaN backward results. | `step_debug` reports named nonfinite output prefixes, with dispatch indices and originating node IDs. It examines final buffer contents, not every write as it happens. |
| What was compiled? | `torch.compile` adds graph capture, breaks, guards and recompiles; its logging/tracing tools help explain them. | One explicit graph and execution plan; inspect rewrites, groups, bindings, geometry, alias map and generated shaders. No automatic fallback executing arbitrary unsupported Python. |
| What explains time and memory? | Profiler activities, operator shapes/stacks, allocation tracking and traces, subject to backend support and instrumentation cost. | CPU spans/Perfetto, GPU pass timestamps, structured dispatch profiles, logical/physical memory summaries and device memory statistics. Narrower tooling and continuous device coverage. |

PyTorch's execution/autograd model is documented in
[Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html).
The gradient facilities are distinct APIs:
[retain_grad](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.retain_grad.html)
and [register_hook](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.register_hook.html).
The anomaly detector's documented NaN/backward contract should not be enlarged
into “finds every nonfinite value anywhere.”
[Autograd anomaly detection](https://docs.pytorch.org/docs/stable/autograd.html#debugging-and-anomaly-detection)

“Static means impossible to debug” is wrong. “Our debug mode makes eager
PyTorch's developer experience irrelevant” is also wrong. The practical
advantage of an explicit plan is that allocation and scheduling decisions are
inspectable without first understanding a dynamic capture system. The cost is
that source-level values need deliberate preservation and reconstruction.

## Follow a value through the system

```text
Graph node: operation, tensor type, name, creation location
  → rewrites and autodiff: values may change or disappear
  → dispatch: label + origin node IDs + buffer bindings
  → runtime: chosen pipeline + workgroups + barrier group
  → allocation: logical buffer → physical allocation
  → observation: readable tensor, numeric report, or timing row
```

`Graph::named` and the `nn` helpers supply semantic labels; `#[track_caller]`
and creation-location data improve graph-construction errors. Dispatch
`origin` can contain several IDs after fusion. It is provenance, not a claim
that each original intermediate still occupies storage. Generated derivative
nodes and graph renumbering also mean that an original numeric ID is not a
universal stable identifier across builds. Prefer distinct semantic names and
inspect the resulting plan. When names repeat, `read_node_by_name` uses the
last recorded match.
[Graph representation](../../src/graph.rs),
[Dispatch representation](../../src/compile.rs),
[runtime reads and reports](../../src/runtime.rs)

There are two independent reasons a tensor may be unavailable:

- **Fusion/elimination:** no dispatch materializes the value at all. More
  memory cannot recover an unwritten intermediate.
- **Lifetime aliasing:** it was materialized, but a later value owns those
  bytes after the step. Reading the allocation is not reading the old value.

`ReadNodeError` distinguishes `UnknownNode`, `UnknownName`, `FusedAway` and
`Aliased`. Identical immutable constants can share an allocation and remain
readable. Do not work around an `Aliased` error with a raw buffer read and
interpret the result as the requested node.
[Memory planner](../../src/memplan.rs)

## A practical inspection loop

First reduce the graph, fix inputs and parameters, name the suspect boundary,
and make the numerical expectation explicit. Small independent references are
more useful than a large tensor dump. The following uses existing APIs and
executes GPU work when run:

```rust
use meganeura::{Graph, Mode, OptimizeMode, SessionConfig, build};

let mut graph = Graph::new();
let x = graph.input("x", &[2, 4]);
let w = graph.parameter("w", &[4, 3]);
let h = graph.matmul(x, w);
let h = graph.named(h, "projection");
let y = graph.relu(h);
graph.set_outputs(vec![y]);

let mut config = SessionConfig::debug();
config.mode = Mode::Inference; // debug() otherwise defaults to training
config.optimize.mode = OptimizeMode::Off;
let (mut session, _) = build(&graph, config);
session.set_input("x", &[1.0; 8]);
session.set_parameter("w", &[0.25; 12]);
session.step();
session.wait();
let projection = session.read_node_by_name("projection").unwrap();
assert_eq!(&projection[..6], &[1.0; 6]);
```

`SessionConfig::debug()` disables dispatch-level fusion through `build` and
asks the runtime to disable aliases, device-local placement and its late
prologue/horizontal fusions. It does **not** set graph rewrites to `Off` or
disable cooperative precision choices. Set those independently when isolating
them. `SessionOptions { debug: true }` on a plan that is already fused cannot
undo the earlier compiler pass. Keeping a suspect value as an additional
graph output can preserve that boundary without materializing every node.
[Build configuration](../../src/train.rs),
[debug-session regression examples](../../tests/debug_session.rs)

`read_node` currently returns an allocation-sized `Vec<f32>`, not a typed
logical tensor. Cooperative padding can add trailing elements; retain the
graph shape and slice the logical extent. Do not reinterpret u32 indices or
packed/f16 storage as meaningful f32 values. Use the slot's typed input/output
APIs and known logical size. Reads of host-visible buffers require a completed
step; explicit `wait()` is the safe habit. Device-local reads incur staging and
synchronization, so inspection is not a free operation.

For inspect-as-you-build work, `eager::Eager` binds host data and offers
`eval(&graph, node) -> Tensor`. It builds an unoptimized, materialized session
and rebuilds when the graph grows; it can run other built nodes as well. It
shares Meganeura's GPU kernels, rather than supplying an independent CPU
oracle or PyTorch's immediate operator dispatch. Its cached execution should
not be mistaken for support for arbitrary in-place graph edits.
[Eager helper](../../src/eager.rs)

## What the numerical probes actually promise

`step_debug()` runs a normal full step, waits, then traverses dispatches in
plan order. For each readable **primary output**, it scans at most the first
65,536 f32 elements and reports NaN, Inf and maximum absolute magnitude.
Non-debug aliased outputs are skipped and counted. `first_bad()` is the first
reported anomaly, not necessarily the first arithmetic operation that went
wrong.

Important limitations to remember in questions:

- It does not scan the entire extent of a large output, all extra outputs,
  every input, or all optimizer/accumulator state. No anomaly is not a
  certificate of finite execution.
- Post-step scanning can miss overwritten values, even without lifetime
  aliasing when multiple operations intentionally write the same buffer.
  A bad input may make the first consumer look responsible.
- It does not detect finite but wrong answers, silent underflow to zero,
  incorrect derivatives or a bad loss scale. Those require numerical
  comparisons, tiny-gradient tests and training trajectories.
- `step_debug` still calls `step`: an active optimizer, accumulation or KV
  update remains active. Use a disposable/restored session for repeatable
  diagnosis. It is not a snapshot or a pause-before-update facility.

`trace_dispatches(threshold)` is a lower-level logging aid with a similar
prefix cap; it does not protect against aliased final contents the way
`read_node` does. Treat it accordingly.
[Probe implementation](../../src/runtime.rs)

Parameter inspection goes further than just loss logging: `read_param`,
`read_param_grad`, `read_all_param_grad_norms`, `read_adam_m`, `read_adam_v` and
`adam_step_count` expose useful training state. Check the precise API and
norm population when comparing to PyTorch. A parameter's norm cannot reveal
an incorrect gradient direction. The paper's norm-vector validation is
therefore not a substitute for debugging elementwise gradients.
[Evidence limits](results.md)

The high-level trainer also exposes step-level metrics and `MetricCallback`;
that is different from an arbitrary per-operation backward hook.

## A bisection ladder for wrong results

Change one layer at a time, using identical saved inputs and parameters. Keep
the failing configuration as well as the simplified one; instrumentation can
hide races or change precision/rounding.

| Hypothesis | Controlled change | Evidence and interpretation |
|---|---|---|
| Wrong model or shapes | Smaller named graph, unoptimized reference, shape/location errors | Check layouts, reduction axes and loss normalization before blaming GPU scheduling. |
| Rewrite or fusion problem | `OptimizeMode::Off`, then independently `options.fuse_dispatches = false` | A passing unoptimized graph narrows the compiler transformation, not necessarily a shader bug. |
| Reduced-precision problem | `runtime.coop = CoopPolicy::Disabled`; verify flash/weight options too | Check tiny/large operands and elementwise derivatives. f32 storage alone does not establish f32 arithmetic everywhere. |
| Lifetime reuse problem | `runtime.no_alias = true`, then selected `pin_buffers` | A passing no-alias run points toward lifetimes/access declarations. Pinning changes memory pressure, so confirm a minimal repro. |
| Device-local staging problem | `runtime.no_device_local = true` | Inspect upload/readback, placement and completion. On Metal, private buffers cannot be read by a host pointer. |
| Group hazard or synchronization | `runtime.serial_dispatch = true`; inspect dump read/write sets | One pass per dispatch adds barriers; passing here narrows but does not prove a missing dependency. |
| Lowering/driver problem | Preserve generated WGSL, exact variant, capability data and Blade/Naga revisions | Validate offline; then reproduce on real hardware. API validation is not an ML numerical oracle. |

Equivalent `MEGANEURA_*` switches are in the central
[configuration registry](../../src/config.rs). They affect clients that
explicitly call `from_env`; the core is environment-free. `MEGANEURA_OPTIMIZER`
means the **graph rewrite optimizer**, not Adam/SGD. Log resolved options so
an environment override or misspelled variable does not invalidate the test.
`MEGANEURA_DUMP_PLAN` and `MEGANEURA_DUMP_WGSL=<directory>` help preserve the
failing plan and shader sources. GPU context creation enables API validation
in debug-assertion builds; that is separate from `SessionConfig::debug()`.

Safe checks while the GPU is occupied include `cargo test --lib`, compiler
checks, offline shader validation, artifact replay and the CPU-only
`diagnose_fusion` example. Building a `Session` or an `Eager` evaluator is
**not** CPU-only: even session construction can run a cooperative smoke test.

## Explain performance without confusing the instruments

Meganeura has three complementary views:

1. **Build/host spans:** graph rewrites, autodiff, compilation, GPU setup and
   runtime work. The `profiler` feature installs CPU tracing support;
   `profiler::init/save` can emit a Perfetto trace. An embedding application's
   existing global tracing subscriber requires deliberate integration.
2. **Structured GPU profile:** `capture_session_profile` retains raw samples,
   selected pipeline keys, workgroups, provenance, family/phase aggregates,
   memory/device information and instrumentation overhead. Timestamp pools
   must be enabled before context creation. Captures serialize dispatches;
   two additional normal executions per sample advance the command-buffer
   ring. The preparation callback must reset changing state for **all** runs.
3. **Normal end-to-end timing:** grouped execution with appropriate warmups,
   waits, transfer boundaries and repeated trials on an idle device. This is
   the test of a speed claim, not the sum of instrumented dispatch medians.

The structured collector rejects missing timestamps, too many passes and
extra runtime-appended optimizer/accumulation/clipping timestamps that cannot
be assigned plan metadata. Capture without those appended passes; separately
measure a complete optimizer-backed workload. Register statistics, where the
backend exposes them, are supporting diagnostics, not a timing model.
[Profiler implementation](../../src/profiler.rs),
[profiling protocol](../performance-profiling.md),
[GPU example](../../examples/profile_session.rs)

`MemorySummary` separates plan capacities (including padding), graph
allocations after aliasing, and actually allocated moments, accumulators and
auxiliary buffers. `total_allocated_bytes()` sums those retained buffer
requests; structured profiles expose the same total as `resident_buffer_bytes`.
Lazy moment reads return zeros without creating GPU state, so inspection
does not turn an SGD/F+L+B session into an Adam-sized allocation.
F32 parameter reads reject reduced/integer storage instead of reinterpreting
it, and raw F32 buffer reads check capacity before copying. Bulk parameter
norm inspection remains F32-only; logical sizes do not imply automatic
dequantization in every diagnostic API.
These fields are not a driver allocation or peak measurement. See the
[checkpoint/memory chapter](checkpoints-and-memory.md) for the field map,
restore-error guarantees and qualification tests.

`device_memory_stats` reports a broader API-level process
view; it does not establish system-wide pressure, fragmentation or peak
allocation history. Compare the same quantity and measurement boundary across
engines. PyTorch's profiler can record shapes, stacks and memory activity,
and documents overhead from such instrumentation too.
[PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler.html)

Compiled PyTorch also has useful observability: graph-break/recompile/guard
logs and `tlparse` traces. Reverting to eager or bisecting compiler backends
can distinguish capture, autodiff and code-generation failures. A fair
comparison includes these facilities instead of calling compilation a black
box. Trace bundles can contain model source: review them before sharing.
[Compiler troubleshooting](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_troubleshooting.html)

## Make autotuning explainable from its first version

Convolution-derivative searches add `TuneOptions.scope` and the optional
`TuneClass.conv2d` shape. Inspect all twelve NCHW convolution parameters, not
just M/N/K: dX also has batch in dispatch Z. Scalar convolution tiles appear
as distinct Small/large shader entries, unlike dense `use_small_tiles`.
The [convolution crossover](../experiments/conv-tiles-2026-09-06/README.md)
archives complete before/after dispatches and declared buffer sizes so a
selection can be traced back to its exact class. Missing historical scope is
Dense; missing convolution shape means a dense class, not an unknown batch.

Those full dispatch records exposed a vacuous benchmark: the first small-case
builder supplied nonflat operands to the flat convolution API, producing zero
forward workgroups. Matching zero gradients/moments had passed the original
runner. The [corrected cohort](../experiments/conv-tiles-corrected-2026-09-06/README.md)
rejects zero-workgroup plans, records `prefix_training_signal` and
`parameters_updated`, and validates nonzero gradients/moments and actual
parameter changes. Public convolution helpers reject malformed operand shapes
early; independent tests read full forward outputs as well as derivatives.
The original twelve controls are retained/disqualified, not overwritten.
This is an example of observability finding invalid work, not merely locating
slow work. Full state agreement alone cannot establish workload validity.

`Session::tune_with(TuneOptions)` returns a serializable `TuneReport` rather
than only logging a winner. Inspect exact
`(shader,M,N,K,precision,binding capacity,placement)` classes, eligible/visited
counts, incumbent/challenger/selected implementation, qualification status, raw paired
samples, medians, the noise guard and budget/validation rejection decisions.
Reports retain resolved options, total/comparison elapsed time and pipeline setup
time, so a winner's search cost remains visible.
`serde_json::to_string_pretty(&report)` can preserve the evidence. The selected
pipeline keys remain visible in `dispatch_pipeline_keys()`.

The follow-up prompted by the holdouts adds `TuneOutcome.phase_times`:
preparation (including pipeline setup and scratch allocation), qualification
(data/uploads/dispatches/readbacks/CPU checks), warmup (including input reset),
and the paired sampling loop. These are non-overlapping host wall times, not
GPU timestamps. `compile_time` is a subset of preparation; do not add it again.
The original four-phase sum excluded final decision bookkeeping and scratch
destruction. The allocation follow-up below measures cleanup separately; always
compare phase sums with total `elapsed` rather than forcing equality.
An early exit retains partial time for the active phase; an unreached phase
is `None`. Entire `phase_times` is `None` for historical reports written before
instrumentation, including the first pilot and six-case holdouts—not zero
measured cost. The later crossover records contain measured phases.

The readback follow-up adds `qualification_breakdown` within those phases:
input preparation, CPU upload copy, upload transfer/wait, dispatch/wait,
readback transfer/wait, CPU allocation/copy from mapped staging, and CPU
validation. Repeated operations accumulate, early exits retain partial times,
and absent historical/unreached measurements remain `None`. The enclosing
qualification timer already contains these costs. Transfer/dispatch fields
include host encoding/submission and waiting; they are not GPU timestamps.
CPU validation still scans complete outputs and compares sampled f64 dots.
`TuneOptions::staging` controls only the one private copy buffer: Shared and
Download keep its capacity and all candidate bindings/validation unchanged.
New options default to Download after the measured promotion gates passed;
missing settings in historical reports still deserialize as Shared. See the
[separate staging protocol and results](../experiments/readback-2026-09-06/README.md).

There may be two comparisons per class; the second challenges the last accepted
winner, not necessarily the original heuristic choice. `failure` identifies
shader rejection or the implementation/input pattern that failed qualification.
It is not a full numerical discrepancy trace. Native cooperative candidates
include complete geometry/padding constraints, and unsupported or policy-disabled
matrix types are never silently substituted by f16 implementations.

This f32 search never runs a live step or binds live tensors. Its
scratch data exercises ordinary and tiny full-mantissa operands, both variants are
checked against sampled f64 dots and each other, and the soft deadline includes
qualification/compilation. These checks do not prove every input or guarantee
a whole-model win. Choices are session-local, and scratch timing omits normal
cross-kernel overlap, cache history and surrounding work. See the
[implementation contract and hardware qualification](performance-plan.md).

The [retained transfer experiment](../experiments/tuning-2026-09-05/README.md)
is a concrete report-reading exercise: compare the three placement-sensitive
classes, per-class decisions, final pipeline keys, search cost and independent
whole-step samples. On this RTX 5070 the native-f32 coverage flag is false;
positive scalar timings cannot be used as evidence for that unexecuted path.

The [optimizer-backed holdout records](../experiments/holdouts-2026-09-06/README.md)
add full in-memory parameter/gradient/moment comparisons, expected Adam counts,
loss trajectories, memory-accounting stages and every timing pair. Read their
ResNet run 3 as a debugging exercise: identical pipeline keys, a 1.071× ratio
of medians, but only 0.01470 ms median paired gain. The guard rejects the apparent
gain. This separates “the kernels changed,” “the numerical state matches,”
and “whole-step time improved.” The retained summaries are not tensor dumps;
CPU replay checks their consistency, not the absent full vectors. Validation
readbacks are outside timers, followed by settling steps, and process memory
samples with both sessions resident are not peak VRAM.

The [predeclared crossover experiment](../experiments/crossover-2026-09-06/README.md)
adds an untuned/untuned control, balanced session roles and continuous optional
GPU telemetry. `Session::swap_tuning_with` exchanges complete legal f32 tile
choices between compatible sessions sharing one GPU context; it does not
exchange tensor allocations, weights, moments or training age. It preflights
the entire layout and prepares missing pipelines before changing choices.
Preparation and waits stay outside timing. On error, choices and tensor state
are unchanged, although newly prepared pipelines may remain cached. Callers
must still match inputs, optimizer settings and age. This is an experimental
control primitive, not a snapshot, persistent tuning profile, or automatic
whole-step confirmation/rollback feature.

The retained crossover is another report-reading exercise. Dense inference
passes both role orientations in all six processes (median 1.177×); MLP+Adam
has four inconclusive outcomes and two noisy A/A controls; ResNet changes no
selection. Bit-exact state checks through Adam step 178 do not imply a training
speedup. That cohort localized ResNet's search to qualification (median 538 of
546 ms), not sampling (7 ms). Its timers still combined CPU checking,
transfers and GPU work within that phase; they did not prove a readback cause.
The 250 ms device telemetry is too coarse to describe every short timing pair.

The subsequent Shared/Download cohort provides the narrower diagnosis. On
Blade 0.9 and the same RTX 5070, median ResNet qualification falls from 598
to 21 ms. Its CPU mapped-memory-to-vector allocation/copy falls from 582 to
2 ms, while complete finite/parity scans and sampled f64 checks remain about
6 ms. Readback transfer/encode/wait actually rises from 0.54 to 2.33 ms, and
preparation rises from 0.65 to 10.41 ms; total search still falls from 606 to
39 ms. The raw records retain all six processes, both search orders, every
check and the first slower dense run. This supports changing the private
staging policy, not shortening validation, claiming pure GPU transfer timings,
or inferring physical heap/cache properties. The Metal backend maps Shared
and Download to the same storage mode; no Metal performance gain is measured.

The [allocation/reuse follow-up](../experiments/staging-reuse-2026-09-06/README.md)
adds `preparation_breakdown`: checks, pipeline setup, candidate buffer allocation,
staging management, encoder creation and binding/geometry work. The pipeline
field is exactly `compile_time`, already within preparation. `phase_times.cleanup`
measures comparison resource destruction, including early exits. Reused staging's
last release is `TuneReport::final_cleanup`, inside total search but outside
comparison times. Release of a previous size belongs to preparation's staging
management. Historical missing fields remain `None`; never add nested times twice.

`TuneOutcome.scratch` records actual binding/staging requests and reuse status.
`TuneReport.scratch` counts staging allocations/reuses/releases, peak simultaneous
scratch requests and bytes retained at return (zero). These are requested bytes,
not physical heap sizes or driver peak VRAM. One slot is reused only at the same
exact size within one call. A size change releases it before new scratch
allocation; candidate bindings/encoders and every validation operation remain
fresh. `staging_reuse: Fresh` disables reuse, and missing historical options
still mean Fresh even though new options now default to SameSize.

Six matched processes reduce dense search from median 44.01 to 31.84 ms and
MLP+Adam from 64.27 to 45.46 ms. Staging allocations fall 3→1 and 5→2; ResNet
stays at one. All 108 comparisons qualify with bit-exact state checks through
Adam step 178 and equal scratch requests. No validation or qualification gain
guard passes: the improvement is allocation/cleanup, not fewer numerical checks.
Read the retained first dense slowdown and the ResNet search-order reversal
before interpreting medians; ResNet passes no gain/regression guard, which
does not establish zero harm. No whole-step or fleet speedup was measured.

## Profiled-state parity is not stationary timing or an independent oracle

The [whole-step localization runner](../experiments/training-profile-2026-09-06/README.md)
profiles synthetic ResNet, SmolLM2 and Whisper F+L+B without optimizer/clip
passes. It compares every retained profiled full state before the collector's
two ordinary ring-advance steps overwrite it; readbacks use separate encoders,
outside the retained wall timer. The GPU regression checks this callback order
and exact timestamp counts. All 45 captured full states match bitwise.

Normal timing blocks still drift: Whisper's first after-block median is nearly
twice its before-block median. Profiled/normal wall ratios are retained, not
silently treated as normal step times. Readback interruptions and timestamp
passes can perturb execution; coarse telemetry does not identify the cause.
The phase label is the scheduled prefix/suffix at the last loss dispatch, so
independent gradient seeds may appear in the prefix. Use shader direction and
origins as well as the phase label.

Source inspection then found a non-same-padding convolution dX bug that both
control and profiled paths shared. A direct f64 scatter oracle exposed it;
the scalar and cooperative generators are now fixed. Read the
[design lesson](design-decisions.md#9-a-shared-baseline-is-not-an-independent-oracle):
observability helps find where to look, but neither exact same-engine parity
nor a complete timing trace proves the underlying derivative is correct.

The [indexing cost check](../experiments/conv-indexing-2026-09-06/README.md)
adds optional full-state SHA-256 digests to the existing profiling runner. Sorted
tensor names, lengths and f32 bits, plus optimizer counter/allocation, make
source-to-source parity checks possible without archiving model tensors. Hashing
and all readbacks stay outside step timers. A digest checks identity, not
correctness: both implementations could still share an address error. Here the
independent f64 oracle first reproduced a width-41 dW error; the shared integer
indexing repair then passes all full-output regressions. Raw digests cannot
replace missing vectors for numerical reanalysis.

The [split-K plan prototype](../experiments/split-k-2026-09-06/README.md) makes
another distinction visible: one logical dW now has a partial producer and a
SumRows consumer. Both retain its origin; labels identify the two passes. The
logical node still reads the **final** gradient. To inspect partials, retain a
diagnostic `no_alias` plan and read its producer output buffer after completion;
ordinary alias reuse can overwrite that temporary later in the step. Do not
treat its buffer index as a persistent graph-node identity. CPU/GPU profiling
must include the final reduction even though its family is not convolution.
Independent f64 checks exposed a long tiny-gradient partial error that final
gradient parity alone would miss. That partition count remains unqualified;
observing a plausible final tensor does not make every intermediate correct.

`Session::measure_conv_weight_splits` now returns ordinary TuneReports for an
explicit class without installing a choice. Read `candidate_split_k` as well as
the tile: baseline/challenger tile fields are identical, but the challenger has
two passes. `FasterCandidate` describes that isolated comparison, not a live-plan
change. Scratch bindings are upstream/input/**final output**/partials, with the
largest binding charged again for staging; the last buffer is no longer implicitly
the final result. Timed repetitions include both passes and their barrier.

The [retained cohort](../experiments/split-k-sequence-2026-09-06/README.md) contains
32 rejections with no warmup/sampling observations, not zero-cost measurements.
Full f64 scans expose two control errors that sampled dots missed. A subsequent
CPU FMA diagnostic reproduces those GPU bits, separating accumulation error from
addressing error. Future generic failure messages also identify the variant;
the four older pointwise two-way records lack that attribution and stay unchanged.
The pointwise probe spends about three seconds in CPU validation before rejecting:
the phase breakdown prevents mistaking this for a return of the old readback-copy
bottleneck. Full-vector checks are executed but vectors are not archived; replay
cannot retrospectively change their tolerance or inspect an unrecorded element.

The [bounded accumulation reporting test](../experiments/compensated-dw-2026-09-06/README.md)
illustrates another distinction: a successful test process means all declared
rows were recorded, not that the candidate qualified. Read JSON `status`, the
`qualified` count and each row's errors. Its 230/240 pass count still means
rejection, and no performance samples are collected. The failed candidate's
source tag remains separate from the active, unchanged arithmetic.

## What we should improve next

The highest-value debugging improvements are typed logical-tensor reads;
explicit probe coverage/truncation in reports; optional complete scans of all
outputs; first-write-time probes or selected dispatch checkpoints; and stable
source-to-optimized-node mappings across transformations. A minimal reproducible
bundle should include graph/options, revision/device/driver identity, seeds,
input hashes, plan/shader dumps and validation results, with tensor contents
opt-in because weights and inputs may be private. None is claimed as shipped
by this chapter.

For workshop rehearsal, answer: “Can you print an activation?”, “Why did that
activation disappear?”, “What exactly does first_bad mean?”, and “Would you
trust timings collected with this probe enabled?” without calling all four
questions the same debugging problem.
