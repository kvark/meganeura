# Megakernel integration and device scheduling plan

Agreed direction, October 24, 2026. Implementation starts with **Milestone 1**,
then proceeds to the **Naga investigation in Milestone 2**. The existing
[prototype and measurements](megakernel.md) are the starting evidence.
Milestone 1 is complete with explicit fallback and measured regressions.
Milestone 2 qualifies the existing fence interface on Vulkan. Session device
scheduling now covers compatible generated regions; broader coverage remains.

Implementation is delivered in separate testable commits. The standalone
prototype is the baseline; subsequent commits update the milestone checklist
only when the corresponding implementation and checks are present.

Implementation evidence:

- `d94fce7`: committed the standalone prototype and this integration plan.
- `9a341e5`: ordinary pointwise and reduction entry points
  now call reusable bodies with explicit logical workgroup IDs. Persistent
  entry points follow in the Session integration. The kernel memo revision
  was bumped. Validation: 27 schedule unit tests and all 167 oracle tests
  passed on NVIDIA with Vulkan synchronization validation; library Clippy
  passed with all features.
- `e1e4594`: eligible operations now use persistent tile queues
  with ordinary operation boundaries, direct Session bindings, GPU queue
  reset, explicit coverage/fallback reports, and strict rejection. This step
  retains one launch per operation; combining operations into regions remains
  below. Both ordinary and persistent paths passed all 167 oracle tests on
  NVIDIA with Vulkan synchronization validation. Focused Session checks cover
  worker pools 1/3/64/256, ragged pointwise/reduction tiles, changed inputs,
  repeated steps, chunking, profiling execution, and repeated external records.
  With the environment selector set to three workers, 68 smoke tests passed
  (two ignored). The regression run passed 181 tests and identified one
  profiling expectation needing the reset's timestamp slot; all six profiling
  tests passed after correcting it (21 regression tests were ignored).
  The focused Session tests also passed on lavapipe. All 505 unit tests and
  all-target/all-feature Clippy passed.
- `dc2e107`: replaced default-only test session construction with shared
  environment configuration; the allocation-reuse control remains explicitly
  ordinary. Oracle reports now carry execution coverage, and an eligible
  oracle case asserts strict persistent coverage as well as numerical results.
  Ordinary and persistent-with-fallback runs both passed 168 oracle,
  182 regression, 68 smoke, and 22 GGUF tests on NVIDIA (21 regression and two
  smoke tests ignored). Lavapipe verified the unsupported F16/BF16 capability
  errors. Strict checks reject runtime optimizer fallback before any update.
  The CI latency example ran in both modes and records raw samples, coverage,
  memory, and construction times. It retains its existing forward/backward
  metric and adds a separate SGD metric with a parameter-update assertion.
  These short, unpaired CI runs do not satisfy the production measurement gate.
- Handler composition preparation: generated tile templates now have explicit
  namespace slots for bindings, parameter types, scratch and callable bodies.
  Empty namespaces preserve the existing entry-point contracts. The 27 schedule
  tests include Naga validation of two independently namespaced bodies in one
  module; focused strict Session checks still pass. Region planning and launch
  reduction remain the next step.
- Access scheduling now orders RAW, WAR and WAW hazards, including fused
  operands, additional outputs and preserved partial outputs. Lifetime pinning
  uses the same access metadata. A deterministic randomized schedule check
  compares reordered execution with sequential execution over mutable buffers;
  the GPU check covers explicit in-place pointwise storage with 1/64 workers.
  All 506 unit tests passed (seven ignored). Ordinary and persistent runs each
  passed 168 oracle, 182 regression, 68 smoke and 22 GGUF tests on NVIDIA with
  Vulkan synchronization validation; all-target/all-feature Clippy passed.
- `0bc2a28`: shared generated bindings now accept a shader data group index,
  preserving the ordinary operand and uniform contracts for region slots.
  Both focused Session checks and all-target/all-feature Clippy passed.
- Recording and profiling now use execution-region indices independently of
  compiled-operation indices. Profile schema 4 retains the untimed source
  operations for each physical launch and supports mixed-phase/family rows.
  This preparation still uses single-operation regions. All six profiling
  tests passed in ordinary and persistent configurations, alongside the
  focused Session checks and all-target/all-feature Clippy.
- Blade was first pinned to `6af8c5783becd798ceaad25f3101935040241f31`
  ([dependency branch](https://github.com/kvark/blade/tree/meganeura-compute-limits)).
  Its Vulkan capabilities now report storage-buffer counts, bound group counts
  and storage binding ranges for resource-aware region composition. Other
  backends explicitly report that these limits are unavailable. Blade's 20
  unit tests passed (one ignored), Clippy passed, and enumeration was checked
  on NVIDIA and lavapipe. The updated Meganeura pin passed all-target checking
  and both focused Session tests. Naga remains at the original revision.
- Independent compatible generated operations now share persistent regions
  of up to three handlers. Planning follows physical allocation, checks
  resource limits and overlapping tensor ranges, and rebuilds after sharing or
  external rebinding. Imported storage conservatively keeps singleton regions;
  BF16 module declarations also remain singleton-only. Every region boundary
  has a hardware barrier. Strict checks cover mixed/ragged pointwise and
  reduction tiles, empty work, fan-out/fan-in, pools 1/3/256, repeated records,
  and physical dependencies introduced by rebinding. A ten-operation fixture
  executes in four launches, and a five-operation profile records two timings
  with five untimed source members. Worker-owned dependent chains remain next.
  Validation: 506 unit tests (seven ignored); NVIDIA persistent runs passed
  168 oracle, 182 regression, 69 smoke and 22 GGUF tests (21 regression and two
  smoke tests ignored). Extended profiling checks passed in both selections;
  focused region/rebinding checks also passed on lavapipe. Clippy, documentation
  and formatting checks passed. These checks establish correctness and launch
  reduction, not a production performance improvement.
- Worker-owned pointwise chains now cross consecutive singleton dependency
  groups when their grids/extents match, reads are per-element, and aliases
  preserve the same logical index mapping. Workgroup storage barriers separate
  bodies; device-wide dependencies keep dispatch boundaries. Serial selection
  disables chaining. An eight-operation strict fixture runs in three launches
  with pools 1/3/256; its serial control retains eight. A same-grid broadcast
  fixture correctly retains the necessary boundary. NVIDIA passed 168 oracle,
  182 regression, 70 smoke and 22 GGUF tests (21 regression/two smoke ignored);
  all four focused Session checks also passed on lavapipe. All-target/all-feature
  Clippy passed. Paired production benchmarking is the remaining Milestone 1 gate.
- Blade now uses `7a20f25eb0d63591611c9622cb8391b5f88f88a4`. Metal reports
  its buffer capacity with a slot reserved for Blade's size table, including
  the combined storage/plain-data budget now checked by region planning.
  Blade's unit tests/Clippy and Apple-target compilation passed; Meganeura's
  core/models/GGUF/profiler library also passed Apple-target compilation.
  Full all-target/all-feature cross-compilation hit C dependencies requiring
  an Apple toolchain/SDK. Metal GPU execution is not qualified by these checks.
  All four focused Session tests passed on NVIDIA with the new pin, alongside
  all-target/all-feature Clippy. Naga is unchanged.
- Optional `PersistentOptions::track_workers` counters now support diagnostic
  utilization measurements. They are off by default, use the existing GPU
  reset, and are included in scheduler residency/range checks. The readback API
  waits for tracked work and returns one task-count row per execution region.
  Repeated/chunked/external, empty, fallback, independent, chain and rebinding
  checks verify histogram totals against actual logical tasks. All four
  focused Session tests passed on NVIDIA and lavapipe; Clippy and docs passed.
  Primary benchmark runs keep this instrumentation off.
- Existing CI latency and SmolVLA training benchmarks now share alternating
  paired measurement, exact compiled-plan matching, fixed-state restoration,
  full output/gradient/parameter comparison, raw samples, coverage and memory
  reports. Diagnostic runs separately collect GPU intervals, pipeline
  statistics and worker task histograms. Short ordinary/paired/diagnostic
  runs passed, including full SmolVLA state parity; all-target/all-feature
  Clippy and formatting passed. Longer production measurements follow.
- `d3b167b`: production qualification completed with 50 alternating pairs
  per case and separate ten-pair diagnostic runs. All full output/gradient/
  parameter comparisons were exact. Persistent execution regressed on the
  production MLP and SmolVLA workloads and remains opt-in. SmolVLA training
  reduced 523 graph launches to 522; 435 launches and optimizer work remain
  ordinary fallback. [Results, raw-record identities and reproduction](megakernel_session_results.md)
  record timings, variation, construction, memory, worker activity and scope.
  This completes Milestone 1 and permits the Naga investigation to start.
- Milestone 2 selects the existing native fence extension and coherent aliases;
  ordinary atomics remain relaxed. Naga `f09d62e54d743f89879e4e8dfb32e717146f2670`
  adds IR capability enforcement, explicit unsupported Metal rejection and
  rejection of SPIR-V access operands that import cannot preserve. Its compiler
  suite passed 144 unit, 267 integration and six documentation tests (24 ignored),
  strict Clippy, formatting and SPIR-V validation. The full wgpu workspace has
  exactly the same 244 failures as its unchanged baseline; the
  [memory contract](megakernel_memory_contract.md) records their classes and logs.
  Blade `63fb651a3d0c7bad6964dba9b2585f794641862f` exposes the capability only for
  its Vulkan memory-model path and uses that Naga revision. Blade passed 20 unit
  tests (one ignored), strict Clippy and Metal-target compilation (warnings;
  no Metal hardware run). Both dependency branches are pushed. Bounded protocol
  checks passed on NVIDIA and lavapipe across four DAG shapes, 1/3/64/4096 workers
  and 16 reset epochs, verifying all-lane data, exactly-once tasks and completion.
  With the published pins, Meganeura passed 506 unit tests (seven ignored),
  71 smoke tests in each ordinary/persistent selection (two ignored), all-target
  strict Clippy, formatting and the lavapipe protocol check. Cargo confirms one
  shared Naga dependency identity.
  This qualifies the synchronization contract; it does not yet enable Session
  device scheduling or establish a performance improvement.

The objective is to run Meganeura graphs through persistent GPU workers using
the ordinary `Graph` / `build` / `Session` interface, and qualify them with the
same correctness and performance machinery as ordinary execution. Coverage
includes inference, backward graphs, optimizer updates, and stateful decoding.
Portable regions with explicit fallback are the first milestone; complete
device scheduling remains a subsequent deliverable, not a claim made by that
first milestone.

## Architecture and invariants

Keep graph optimization, autodiff, numerical policy, and semantic lowering
shared. Lower the existing `ExecutionPlan` into persistent regions and ordinary
dispatch spans; later, use the same work representation for a device-scheduled
dependency graph. A work descriptor identifies a generated kernel handler,
logical workgroup/tile range, parameters, and session buffer references.

- A worker executes one tile cooperatively. Handler selection and barriers
  must be uniform within its workgroup. Logical workgroup IDs are distinct
  from physical worker IDs. Scratch must be initialized for each task that
  needs it, including after switching handlers.
- Generate only the handlers used by a plan. Share their arithmetic and
  addressing code with ordinary kernels, preserving precision, padding,
  fusion, and storage-format contracts.
- Preserve `Session` upload/readback, external encoder, checkpoint, parameter
  sharing, cache, and profiling behavior. Scheduler reset must not reset
  parameters, optimizer moments, accumulators, or KV caches.
- Order read-after-write, write-after-read, write-after-write, and physical
  allocation reuse hazards. Include extra outputs, fused operands, and
  mutations of input buffers in access metadata.
- Keep ordinary execution the default during qualification. A typed config
  selects persistent execution, worker geometry, and fallback policy. Any
  environment selector belongs in `config.rs`, not in core execution.
- Record fallback and coverage explicitly. Strict qualification rejects work
  outside its declared persistent scope; it must never silently run an
  ordinary kernel and report persistent success.
- Workers are logical workgroups; neither phase promises CU/SM pinning or
  assumes that every launched workgroup is simultaneously resident.

## Milestone 1 — persistent regions inside Session

### 1.1 Execution selection and reporting

- [x] Add typed execution options through `SessionConfig.runtime` /
  `SessionOptions`; keep the exact API small and decide names during implementation.
- [x] Add a construction report with selected execution mode, persistent
  regions, logical tasks, physical dispatches, worker configuration, and
  fallback spans/reasons. Preserve origin node IDs for diagnostics.
- [x] Integrate both `step()` and externally recorded execution, including
  waiting, destruction, repeated steps, and submission chunk boundaries.
- [x] Specify profiling and tuning behavior for regions. Report region-level
  timings honestly; do not invent individual kernel timings inside one launch.
  Rebuild or invalidate region plans when tuning changes their kernels.
- [x] Include execution mode, handler/configuration identity, and relevant
  compiler/backend capabilities in affected cache keys.

### 1.2 Region planning and session storage

- [x] Partition the compiled dispatch plan using existing access and barrier
  information. Begin with conservative boundaries and compatible kernels.
- [x] Admit independent tasks and chains whose dependencies are wholly owned
  by one worker. Preserve hardware dispatch barriers wherever a result must
  become visible to another worker. Multi-stage/global reductions retain
  those boundaries.
- [x] Resolve descriptors against session allocations and offsets after
  storage layout is known. Respect binding-count/range limits, mixed storage
  types, shared parameters, and external buffer rebinding. Split a region or
  report fallback when its binding contract cannot be represented.
- [x] Start without intermediate lifetime aliasing for persistent sessions.
  Preserve user-visible sharing and state. Restore lifetime reuse only with
  a proof that region scheduling and physical reuse are ordered consistently.
- [x] Allocate descriptors, queue state, and scratch once. Per-step recording
  resets scheduler state on the GPU; it does not rebuild tasks or copy live
  tensor arenas through the host. Include these allocations in memory reports.

### 1.3 Reusable pointwise and reduction handlers

- [x] Refactor generated pointwise and reduction kernels into callable tile
  bodies shared by ordinary and persistent entry points. Preserve the
  existing schedule representation instead of adding a second operator list.
- [x] Pass logical dispatch geometry explicitly. Audit early returns, uniform
  control flow, subgroup assumptions, barrier placement, and reused scratch.
- [x] Group handlers by compatible workgroup/resource requirements. Do not
  force existing 32/128/256-thread or cooperative kernels into the prototype's
  64-thread geometry. Account for combined register/shared-memory occupancy.
- [x] Keep unsupported operators, variants, and dtype combinations as
  explicit ordinary spans. Matrix/attention/convolution kernels can be added
  after this first generated-handler path is qualified.

### 1.4 Existing harness integration

- [x] Thread execution selection through `reference::gpu::Options`, its
  lowering variants, shared test helpers, and benchmark session construction.
  Audit tests that construct default sessions directly so the requested
  backend is actually exercised.
- [x] Run existing oracle, smoke, regression, and available model tests under
  ordinary and persistent-with-fallback configurations. Keep existing
  tolerances, precision policies, poisoning, and hardware capability gates.
- [x] Run eligible pointwise/reduction cases in strict mode and assert both
  numerical results and actual persistent coverage. Cover partial tiles,
  mixed task lengths, fan-out/fan-in boundaries, empty work, in-place accesses,
  repeated resets, and pools smaller/larger than the task count.
- [x] Check repeated inference, training state updates, gradient accumulation,
  clipping, KV-cache mutation, checkpoint restore, sharing/rebinding, and
  external submission behavior through their existing tests. In this phase,
  optimizer/clipping/accumulation may remain ordinary spans and must be reported.
- [x] Run the existing latency and model benchmarks with identical inputs,
  parameters, precision, tuning budgets, and measurement boundaries. Retain
  the standalone prototype as a scheduling diagnostic.

**Exit evidence:** existing graphs execute through `Session` without manual
`Job` construction; eligible generated pointwise/reduction regions execute
persistently and pass strict checks; unsupported work is reported and runs
correctly through ordinary spans; shared harness results and paired production
baseline measurements are recorded. A graph with no eligible region is valid
in fallback mode but contributes zero persistent coverage. Performance results
may include regressions; this milestone does not enable automatic adoption.

## Milestone 2 — minimally intrusive atomic semantics in our Naga fork

Start this investigation after Milestone 1. Its output is an evidenced compiler
design and the smallest justified implementation, not a predetermined syntax
or a blanket strengthening of all atomics.

### Baseline to inspect first

At the start of this investigation, Meganeura and Blade both used Naga
`a489554c96f87cbc5252ee00350730f6e93869f9`; Blade was
`7a20f25eb0d63591611c9622cb8391b5f88f88a4`. The initial source inspection
used Blade `eae0c6f7aa3167c79d1135050cf8e17c17fabd0a`; the intervening changes
only expose compute resource limits. Source inspection already shows:

- The WGSL frontend accepts `enable wgpu_memory_fence;` and lowers
  `storageFence()` / `workgroupFence()` to `Statement::MemoryBarrier`.
- The SPIR-V writer gives a storage memory fence Device scope, or QueueFamily
  scope with the Vulkan memory model. Its Vulkan path adds availability and
  visibility semantics. Ordinary control barriers remain separate.
- Atomic loads/stores use `Expression::Load` / `Statement::Store`; atomic
  read-modify-write operations use `Statement::Atomic`. Any explicit ordering
  proposal must account for all three paths, including compare-exchange.
- Blade already detects/enables Vulkan memory-model support and passes that
  selection to Naga. Availability of this path does not prove the proposed
  queue protocol correct or establish equivalent semantics on other backends.

Inspect [WGSL lowering](https://github.com/kvark/wgpu/blob/a489554c96f87cbc5252ee00350730f6e93869f9/naga/src/front/wgsl/lower/mod.rs),
[Naga IR](https://github.com/kvark/wgpu/blob/a489554c96f87cbc5252ee00350730f6e93869f9/naga/src/ir/mod.rs),
and [SPIR-V memory emission](https://github.com/kvark/wgpu/blob/a489554c96f87cbc5252ee00350730f6e93869f9/naga/src/back/spv/writer.rs).
Recheck these against the fork revision used when the investigation starts.

### 2.1 Define the required protocol before choosing an API

The [qualified memory contract](megakernel_memory_contract.md) spells out
all-lane publication/acquisition, RMW fan-in and a bounded bitmap ready queue.
It reuses the existing fence extension and avoids both global waiting and
Vulkan availability/visibility chains. The document records the compiler,
backend and device evidence, including the remaining qualification limits.

- [x] Write the happens-before argument for tensor publication and queue-slot
  publication, consumer acquisition, dependency-counter fan-in, and slot reuse.
  Cover writes from **every producer lane** and reads by **every consumer lane**,
  not just the leaders performing queue atomics.
- [x] Identify required memory scope, storage classes, availability/visibility,
  fence ordering, and any per-operation ordering. Specify compare-exchange
  success/failure semantics and legal combinations if the protocol needs them.
- [x] Separate memory ordering from execution progress. Specify bounded queue
  capacity, publication without blocking all producers, exactly-once release
  of ready tasks, counter/sequence wrap, and completion with work in flight.
  An empty ready queue alone is not completion. Do not introduce a global
  spinning barrier or rely on a guessed resident-workgroup count.

Standard [WGSL atomics](https://www.w3.org/TR/WGSL/#atomic-builtin-functions)
have relaxed ordering and its synchronization builtins have workgroup scope.
The fork's fence extension is a separate candidate contract. Evaluate it
against the [Vulkan memory model](https://docs.vulkan.org/spec/latest/appendices/memorymodel.html),
including the enabled features and emitted SPIR-V, before deciding it is sufficient.

### 2.2 Compare the smallest consistent exposures

| Candidate | Question to resolve |
|---|---|
| Existing fence extension plus ordinary atomics | Can it express and preserve the full protocol with no new language/IR surface, and at acceptable cost? |
| Explicit ordering/scope on memory fences | If the existing fences are too broad or incomplete, can a small extension to their representation cover the requirements? |
| Explicit ordering/scope on atomic operations | Is per-operation control necessary or materially simpler/faster, and how should loads, stores, RMW, and CAS represent it consistently? |

- [x] Compare changes to parsing, IR, validation, transformations, reflection,
  serialization, and each writer/reader. Account for optimizer movement or
  elimination of accesses across fences and ordered atomic operations.
- [x] Preserve standard WGSL behavior. Gate native extensions explicitly;
  follow the fork's existing extension conventions and return clear errors
  on unsupported targets. Do not use backend-wide switches to silently alter
  every ordinary atomic operation.
- [x] Audit Vulkan/SPIR-V first and Metal separately. A workgroup rendezvous
  is not an equivalent implementation of a device memory fence. Document
  every backend's supported semantics or rejection path.
- [x] Record a short decision with competing designs, protocol proof,
  compatibility costs, emitted code, and measurements. Prefer reuse of the
  existing path if it satisfies the contract; add only demonstrated gaps.

### 2.3 Compiler and device qualification

- [x] Add focused parser/validator and backend-output coverage in the Naga
  fork, following its existing test layout. Check unsupported order/scope
  combinations and that ordinary atomic output/behavior stays unchanged.
- [x] Inspect/disassemble generated SPIR-V and validate its scopes, memory
  model, semantics, storage classes, and capability declarations with
  `spirv-val`. Exercise enabled and unsupported feature configurations.
- [x] Run repeated GPU litmus/protocol tests for message passing, all-lane
  payloads, multi-producer fan-in, queue publication/reuse, and termination.
  Vary scheduling pressure, worker counts, and graph shape. Use bounded
  observations/watchdogs so a progress failure is diagnosable.
- [x] Combine specification reasoning, compiler inspection, and hardware
  evidence. A passing stress test or validation layer alone is not a memory
  ordering/progress proof. Do not claim Metal qualification from a Vulkan run.
- [x] Update Blade/Naga pins together when changes are needed; retain one
  Naga type/version identity across Blade and Meganeura's developer tools.

**Exit evidence:** a documented choice establishes the least intrusive
sufficient interface, all-lane publication/acquisition is justified, queue
progress is specified, compiler/protocol checks pass on declared targets, and
unsupported targets fail clearly. Syntax and IR changes are conditional on
the investigation; identifying and qualifying sufficient existing semantics
is also a valid outcome.

## Follow-through — device-scheduled graphs

The first implementation applies the qualified protocol to compatible generated
pointwise/reduction regions of up to three handlers. Typed configuration and the
shared environment selector expose it; the existing paired benchmark accepts
`--device-scheduling`. Physical RAW/WAR/WAW dependencies, all-lane fences,
coherent aliases, bounded scans, immutable reset images and completion readback
are integrated. Profiling, external records, chunking, serial selection and
rebinding retain their Session contracts. Initial lifetime aliasing remains off.

NVIDIA device-mode qualification passed 168 oracle, 182 regression, 72 smoke and
22 GGUF tests (21 regression and two smoke tests ignored). The oracle harness
also checks queue completion. Focused checks cover fan-in, cross-tile reductions,
empty work among live tasks, 1/3/64/4096 workers, serial boundaries, physical
RAW/WAR/WAW aliases, parameter rebinding and repeated external reset/recording.
All 506 unit checks pass after documenting the new environment selector; strict
Clippy, formatting and documentation pass. All five focused Session checks pass
on NVIDIA and lavapipe; ordinary-mode smoke also passes 72 tests (two ignored).
At `275456d`, [paired production results](megakernel_device_results.md) show exact
state/output comparisons but regressions of 28–65%. Model graph launches fall
from 154 to 151 forward and 523 to 518 training; coverage is unchanged and
optimizer work remains ordinary. The first implementation also uses device
queues for singleton regions. Measurements justify retaining portable queues
where no inter-operation dependency needs the native protocol.

The measured follow-up retains portable queues for independent and singleton
regions, allocating a reset image only when the Session contains a device queue.
Coverage and physical launch boundaries are unchanged. Mixed portable/device
queues pass the focused tests on NVIDIA and lavapipe, including repeated resets
and worker totals. The full NVIDIA device selection again passes 168 oracle,
182 regression, 72 smoke and 22 GGUF tests; strict Clippy, formatting and
core/models/GGUF/profiler Metal-target compilation pass. Metal device scheduling
remains unsupported. Updated paired measurements follow this change.
At clean `b8a1905`, the repeated 50-pair measurements reduce median paired
ratios to 1.034/1.244/1.232 for CI inference/backward/SGD and 1.214/1.170 for
model forward/training. All comparisons remain exact. Native model regions
still use only one to three workers in the diagnostic samples; see the retained
[selective-queue evidence](megakernel_device_results.md#selective-device-queues).

Device workers now retry at most 64 unsuccessful ready scans before retiring,
and exit earlier when the unfinished-task count reaches zero. This gives short
dependent handoffs an opportunity to retain workers while preserving bounded
execution without a residency assumption. All five focused Session checks pass
on NVIDIA and lavapipe, including 4,096 workers. NVIDIA again passes 168 oracle,
182 regression, 72 smoke and 22 GGUF tests (21 regression and two smoke tests
ignored); strict all-target Clippy and formatting pass. At clean `a2dc351`, 50
model pairs passed exact full-state comparisons. Median paired ratios were
1.091 forward and 1.139 training, with substantial training outliers retained.
Separate diagnostics show the 194-task native region using 60 workers instead
of three. Ordinary remains faster; [results](megakernel_device_results.md#bounded-idle-retries)
record the distributions. CI has no native regions and its unchanged execution
path was not remeasured.

Handler-coverage preparation separates the existing implementation generator
from pipeline creation and routes ordinary and persistent bindings through the
same function with an explicit group index. Selection, arithmetic and coverage
are unchanged. NVIDIA passes 168 oracle and 72 smoke tests (two ignored), plus
the five focused Session checks after routing persistent bindings through the
shared function. Strict all-target Clippy and formatting pass.

The device queue uses the same Session work descriptors, tracking ready tasks,
predecessor counts, unfinished work and successor edges on the GPU. It releases
consumers after whole-operation completion; measure whether tile-level
dependencies are useful before complicating that protocol.
Never occupy workers waiting on a particular unready graph node when ready
work can be executed instead.

Broaden handlers to matrix products, attention, convolution, data movement,
quantized/reduced storage, and their derivatives. Include optimizer, clipping,
and accumulation work currently recorded outside `ExecutionPlan`. Reintroduce
aliasing with explicit physical reuse dependencies. Reuse the Milestone 1
backend matrix and reports rather than building another application harness.

Full graph coverage requires strict execution of all currently supported
operator/dtype/state combinations on each claimed target, with remaining
exceptions enumerated. Correct fallback execution is useful throughout
development but does not prove full device scheduling. Record actual launch
counts and boundaries; graphs split for resource or backend constraints must
not be reported as single-dispatch graphs.

## Shared verification and benchmark gates

The following commands already exist. Run the relevant suites in each backend
configuration after the selector/harness wiring is implemented; these commands
alone currently select the ordinary backend for normal graph sessions.

```sh
cargo test --lib
cargo test --test oracle --all-features -- --test-threads=1
cargo test --test smoke --all-features -- --test-threads=1
cargo test --test regression --all-features -- --test-threads=1
cargo test --test gguf_model --all-features -- --test-threads=1
cargo fmt --all -- --check
cargo check --all-targets
cargo clippy --all-targets --all-features -- -D warnings
cargo doc --no-deps --all-features
```

Use [testing.md](testing.md) for applicable opt-in tests and device setup.
Extend existing suites with scheduler-specific invariants rather than
duplicating each operator's reference test. Keep reports of unsupported
hardware, skipped cases, and persistent coverage alongside pass counts.

Use [`bench_ci_latency`](../bench/bench_ci_latency.rs) for completed inference
and training latency, existing model benchmarks for representative whole
graphs, and [`build_measured`](../src/train/search.rs) for eventual measured
adoption. Compare the production ordinary backend with persistent execution;
keep the prototype's same-body control as a separate diagnostic.

For each comparison, preserve adapter/driver/compiler revisions, inputs,
precision, tuning policy, and training/cache state. Use warmup and interleaved
paired samples. Measure completed work, keep upload/readback boundaries
identical, and restore matched state between training comparisons. Report
whole-step wall time, available GPU time, construction/compilation time,
memory, persistent/fallback coverage, dispatch count, and worker utilization.
Retain raw samples and regressions. Reduced launch count or an isolated kernel
win does not bypass numerical qualification and the whole-program timing gate.
