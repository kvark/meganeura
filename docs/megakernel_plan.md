# Megakernel integration and device scheduling plan

Agreed direction, October 24, 2026. Implementation starts with **Milestone 1**,
then proceeds to the **Naga investigation in Milestone 2**. The existing
[prototype and measurements](megakernel.md) are the starting evidence.
Session integration is in progress; the compiler investigation follows it.

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
- Initial Session path: eligible operations now use persistent tile queues
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
  all-target/all-feature Clippy passed. Tests with explicit default sessions
  still need the harness audit below before claiming complete backend coverage.

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
- [ ] Add a construction report with selected execution mode, persistent
  regions, logical tasks, physical dispatches, worker configuration, and
  fallback spans/reasons. Preserve origin node IDs for diagnostics.
- [ ] Integrate both `step()` and externally recorded execution, including
  waiting, destruction, repeated steps, and submission chunk boundaries.
- [ ] Specify profiling and tuning behavior for regions. Report region-level
  timings honestly; do not invent individual kernel timings inside one launch.
  Rebuild or invalidate region plans when tuning changes their kernels.
- [ ] Include execution mode, handler/configuration identity, and relevant
  compiler/backend capabilities in affected cache keys.

### 1.2 Region planning and session storage

- [ ] Partition the compiled dispatch plan using existing access and barrier
  information. Begin with conservative boundaries and compatible kernels.
- [ ] Admit independent tasks and chains whose dependencies are wholly owned
  by one worker. Preserve hardware dispatch barriers wherever a result must
  become visible to another worker. Multi-stage/global reductions retain
  those boundaries.
- [ ] Resolve descriptors against session allocations and offsets after
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
- [ ] Pass logical dispatch geometry explicitly. Audit early returns, uniform
  control flow, subgroup assumptions, barrier placement, and reused scratch.
- [ ] Group handlers by compatible workgroup/resource requirements. Do not
  force existing 32/128/256-thread or cooperative kernels into the prototype's
  64-thread geometry. Account for combined register/shared-memory occupancy.
- [ ] Keep unsupported operators, variants, and dtype combinations as
  explicit ordinary spans. Matrix/attention/convolution kernels can be added
  after this first generated-handler path is qualified.

### 1.4 Existing harness integration

- [ ] Thread execution selection through `reference::gpu::Options`, its
  lowering variants, shared test helpers, and benchmark session construction.
  Audit tests that construct default sessions directly so the requested
  backend is actually exercised.
- [ ] Run existing oracle, smoke, regression, and available model tests under
  ordinary and persistent-with-fallback configurations. Keep existing
  tolerances, precision policies, poisoning, and hardware capability gates.
- [ ] Run eligible pointwise/reduction cases in strict mode and assert both
  numerical results and actual persistent coverage. Cover partial tiles,
  mixed task lengths, fan-out/fan-in boundaries, empty work, in-place accesses,
  repeated resets, and pools smaller/larger than the task count.
- [ ] Check repeated inference, training state updates, gradient accumulation,
  clipping, KV-cache mutation, checkpoint restore, sharing/rebinding, and
  external submission behavior through their existing tests. In this phase,
  optimizer/clipping/accumulation may remain ordinary spans and must be reported.
- [ ] Run the existing latency and model benchmarks with identical inputs,
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

At the current pins, Meganeura and Blade both use Naga
`a489554c96f87cbc5252ee00350730f6e93869f9`; Blade is
`eae0c6f7aa3167c79d1135050cf8e17c17fabd0a`. Source inspection already shows:

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

- [ ] Write the happens-before argument for tensor publication and queue-slot
  publication, consumer acquisition, dependency-counter fan-in, and slot reuse.
  Cover writes from **every producer lane** and reads by **every consumer lane**,
  not just the leaders performing queue atomics.
- [ ] Identify required memory scope, storage classes, availability/visibility,
  fence ordering, and any per-operation ordering. Specify compare-exchange
  success/failure semantics and legal combinations if the protocol needs them.
- [ ] Separate memory ordering from execution progress. Specify bounded queue
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

- [ ] Compare changes to parsing, IR, validation, transformations, reflection,
  serialization, and each writer/reader. Account for optimizer movement or
  elimination of accesses across fences and ordered atomic operations.
- [ ] Preserve standard WGSL behavior. Gate native extensions explicitly;
  follow the fork's existing extension conventions and return clear errors
  on unsupported targets. Do not use backend-wide switches to silently alter
  every ordinary atomic operation.
- [ ] Audit Vulkan/SPIR-V first and Metal separately. A workgroup rendezvous
  is not an equivalent implementation of a device memory fence. Document
  every backend's supported semantics or rejection path.
- [ ] Record a short decision with competing designs, protocol proof,
  compatibility costs, emitted code, and measurements. Prefer reuse of the
  existing path if it satisfies the contract; add only demonstrated gaps.

### 2.3 Compiler and device qualification

- [ ] Add focused parser/validator and backend-output coverage in the Naga
  fork, following its existing test layout. Check unsupported order/scope
  combinations and that ordinary atomic output/behavior stays unchanged.
- [ ] Inspect/disassemble generated SPIR-V and validate its scopes, memory
  model, semantics, storage classes, and capability declarations with
  `spirv-val`. Exercise enabled and unsupported feature configurations.
- [ ] Run repeated GPU litmus/protocol tests for message passing, all-lane
  payloads, multi-producer fan-in, queue publication/reuse, and termination.
  Vary scheduling pressure, worker counts, and graph shape. Use bounded
  observations/watchdogs so a progress failure is diagnosable.
- [ ] Combine specification reasoning, compiler inspection, and hardware
  evidence. A passing stress test or validation layer alone is not a memory
  ordering/progress proof. Do not claim Metal qualification from a Vulkan run.
- [ ] Update Blade/Naga pins together when changes are needed; retain one
  Naga type/version identity across Blade and Meganeura's developer tools.

**Exit evidence:** a documented choice establishes the least intrusive
sufficient interface, all-lane publication/acquisition is justified, queue
progress is specified, compiler/protocol checks pass on declared targets, and
unsupported targets fail clearly. Syntax and IR changes are conditional on
the investigation; identifying and qualifying sufficient existing semantics
is also a valid outcome.

## Follow-through — device-scheduled graphs

After Milestone 2, apply its synchronization contract to the same Session work
descriptors. Maintain ready tasks, predecessor counts, in-flight work, and
successor edges on the GPU. Start with whole-operation completion before
releasing consumers, then measure whether tile-level dependencies are useful.
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
