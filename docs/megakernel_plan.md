# Megakernel integration and device scheduling plan

Agreed direction, October 24, 2026. Implementation starts with **Milestone 1**,
then proceeds to the **Naga investigation in Milestone 2**. The existing
[prototype and measurements](megakernel.md) are the starting evidence.
Milestone 1 is complete with explicit fallback and measured regressions.
Milestone 2's descriptor-reference audit has been repaired for whole-graph
execution; regional publication still needs review, as documented in the
[memory contract](megakernel_memory_contract.md). The regional
backend remains available for comparison. The next milestone now implements
**one persistent worker pool for the entire graph**, including optimizer,
clipping and accumulation nodes, in a single compute dispatch. This is selected
with `PersistentOptions::whole_graph` or `MEGANEURA_WHOLE_GRAPH=1` together with
device scheduling and a worker count. Ordinary execution remains the default.

The [NVIDIA counter investigation](megakernel_performance_analysis.md) identifies
fixed worker sizing and portable queue overhead as the leading SmolVLA costs.
Those measurements describe the regional implementation. Whole-graph execution
has a different resource and scheduling cost and needs its own measurements;
the descriptor-reference repair requires fresh measurements.

## Whole-graph milestone

- [x] Canonicalize backing-buffer descriptors across typed views and offsets;
  add explicit native binding aliases and alias-aware SPIR-V access analysis.
  Recheck cross-type publication, slices and optimizer chunks before accepting
  more performance changes.
- [ ] Audit and repair descriptor identity in the regional device-scheduled path.

- [x] Reuse and deduplicate the selected ordinary tile implementations, preserving
  their arithmetic and logical geometry. Adapt smaller workgroups without
  putting workgroup barriers behind lane-dependent control flow.
- [x] Replace per-operation bind groups with typed buffer descriptor arrays and
  immutable node parameters. Lift uniform parameters into function arguments so
  Naga can still prove barrier uniformity. Pool shared scratch across handlers.
- [x] Build a device queue over every graph node. Add physical RAW/WAR/WAW edges,
  including lifetime reuse, and publish successors only after every producer
  tile completes. Workers remain live until the graph completes.
- [x] Capture the existing SGD, Adam/LaProp, clipping and accumulation bindings
  into that same queue. `step()` performs a transfer reset and one compute
  dispatch in one submission, with no intermediate accumulation submission.
- [x] Preserve immutable parameters across external recordings; reuse metadata
  only after the corresponding submission completes. Report total physical
  compute dispatches separately from logical optimizer phases.
- [x] Pass all 168 oracle tests with strict whole-graph selection, the six focused
  queue/replay/optimizer tests, and broader ordinary-path regression checks.
- [x] Record 350 paired comparisons and qualified NVIDIA SDK captures. Every
  selected case uses one dispatch; SmolVLA at 256 workers regresses by
  2.300× forward and 1.886× with SGD. Counter evidence identifies fourfold
  duplicated tensor calculations in the initial lane adapter.
- [x] Skip inactive subgroups' cooperative products using scope-aware Naga
  validation. SDK estimates fall from 4× to 1.012× ordinary tensor activity per
  step. Cache descriptor indices, use fixed uniform offsets, omit transitive
  edges and retain only necessary per-access memory propagation; repeat oracle
  and focused graph qualification.
- [x] Repeat clean full-step measurements after those optimizations: 550 paired
  comparisons pass at `58458d2`; SmolVLA ratios improve to 1.897× forward and
  1.534× with SGD. Baseline parity remains unmet.
- [x] Extend SDK captures to instruction and memory counters. Trace excess
  local-memory traffic to cooperative attention's output array and replace it
  with scalar accumulators while preserving arithmetic in both paths.
  Clean `36304a0` passes 250 new paired comparisons; SmolVLA ratios are now
  1.816× forward and 1.510× with SGD.
- [x] Collect sequential single-pass SDK stall components with stable-clock,
  instruction-rate and complete-state checks. Workgroup-barrier waiting rises
  from 3.6% to 50.4% of active warp cycles; memory-barrier waiting is only 0.5%.
  Separate scheduler waiting and inactive-lane costs before changing semantics.
- [x] Test skipping inactive lanes' synchronization-free loops. The candidate
  passes 664 library tests, 168 strict oracle tests and 250 paired comparisons;
  SDK instruction and load counts fall, but full-step timing does not show a
  useful improvement. Revert it and retain the evidence.
- [x] Pack two or four independent logical tiles into spare worker subgroups,
  with isolated scratch and admission checks for uniform barrier control.
  Keep query-dependent causal attention unpacked. Pass 665 library tests,
  all 168 strict oracle tests and seven focused whole-graph tests, including
  repeated packed matrix execution with 1/3/64/4096 workers.
- [x] Measure concurrent packing at `88d5e51`: 250 isolated full-step paired
  comparisons pass. SmolVLA forward improves to 1.468× ordinary, while training
  measures 1.426× with a wider interval. Separate SDK captures preserve exact
  forward state and estimate 1.87× warp instructions and 1.11× global-load
  sectors. Every selected case still uses one dispatch; parity remains unmet.
- [x] Probe 64/128/512 workers, retest inactive-loop suppression, and qualify
  head-axis packing with 100 comparisons. None establishes a useful improvement
  over the published implementation. Barrier removal and direct-successor
  reservation also pass focused checks without a timing gain; retain the
  evidence and revert these candidates.
- [x] Put Linux builds, tests and captures in separate bounded cgroups using the
  checked-in `scripts/run_bounded_job.py`; verify membership, memory limits and
  exit propagation. Fix the shared-DAG debug-log expansion that caused host OOM.
- [x] Measure the combined register/shared footprint. Probe parameter layout,
  metadata lifetimes, range-aware dependencies, wider attention workgroups and
  leader-computed coordinates. Correctness passes but timing and follow-up SDK
  instruction counts show no useful gain; retain the evidence and revert them.
- [x] Add temporary device-clock instrumentation and compare worker lifetimes,
  handler intervals and ordinary dispatch profiles. Scalar matrix handlers and
  small SGD claims remain expensive; narrow nodes leave many workers waiting.
  Reject parameter hints and read-only/visibility prototypes that fail to
  improve both phases. Preserve the instrumentation and raw evidence locally.
- [x] Batch four independent SGD tiles per claim for large grids while retaining
  the selected element arithmetic. Clamp final-batch completion counts and
  retain the straight-line handler for small initial grids. Pass 665 library,
  168 strict oracle and seven focused tests, including a partial 1,025-tile
  optimizer chunk followed by small chunks sharing the handler.
- [x] Qualify large-grid batching with 250 final paired comparisons and a fresh
  100-comparison run of the prior runtime. SmolVLA training medians are 20.965 ms
  and 21.552 ms; normalized intervals overlap. Preserve the uncertainty and
  leave baseline parity open. Small CI training avoids the unrestricted
  batching prototype's regression.
- [ ] Isolate remaining queue/indexing/resource costs and reach baseline
  performance under the same full-step measurement and correctness gates. Keep this opt-in until performance justifies
  a default change.
- [x] Requalify canonical descriptors at `ed81459`: 665 library, 168 strict
  oracle, eight focused GPU tests and 250 paired comparisons pass. Fresh SDK
  captures preserve exact state and show 1.924× warp instructions, 2.552× integer
  instructions and 62.28% barrier waiting among active warp cycles. SmolVLA
  ratios remain 1.489× forward and 1.375× training; parity is still unmet.
- [x] Audit complete-step immutability, including optimizer writes and backing
  allocation reuse. Test separate read-only tables with canonical references.
  Handler variants improve forward but regress training; conservative shared
  bindings show no useful gain. Preserve 200 exploratory paired comparisons,
  register diagnostics and worker-count controls, and revert the probes.
- [x] Measure worker placement with a temporary NVIDIA SM-ID probe. Narrow
  matrix nodes cluster, but electing one worker per SM does not improve the
  matching instrumented control. Check compiler inlining, descriptor uniformity
  and smaller CI worker pools; 760 comparisons pass without closing the gap.
  Restore the qualified runtime and preserve the failed/preliminary attempts.
- [x] Time queue reset separately with worker histograms disabled. Its median
  interval is 2.4 microseconds for CI and 3.9–5.1 microseconds for SmolVLA, too
  small to explain the gap. Do not add a reset protocol on that hypothesis.
- [x] Probe a single-worker executor with workgroup synchronization and cached
  node parameters. It improves the one-worker controls but remains slower than
  parallel execution. Test parameter reuse and aggregated publication across
  multiple claims at 256 workers; neither unrestricted nor large-grid-only reuse
  shows a useful full-step gain. Preserve 430 paired comparisons and revert all
  three prototypes.
- [x] Test an append-only ready-node queue and completion-gated leader polling,
  preserving publication proofs. Also test 512-lane workers and constant active
  masks. All 400 exploratory comparisons pass, but no runtime change wins.
  SDK captures show polling can lose substantial warp instructions without
  reducing latency. Revert the probes and preserve their evidence.
- [x] Require matching benchmark/SDK adapters and collected shader activity.
  Exclude a RADV-target/NVIDIA-monitor capture and audit the primary NVIDIA
  captures; record device identity and explicit capture validity in receipts.
- [x] Compare an isolated matrix handler with unused forward/training handlers
  compiled into the same shader, keeping actual graph work fixed. Sweep worker
  counts and measure failed claims. Larger resource footprints can improve this
  narrow workload; register count alone does not explain its slowdown.
- [x] Test a bounded claim-priority window based on worker start order. Failed
  claims decrease, but full training regresses against fresh controls. Preserve
  560 normal comparisons and twelve diagnostics across these experiments;
  revert the temporary runtime and benchmark API changes.
- [x] Isolate constant descriptors, dimensions and grid coordinates in the
  fixed-work matrix fixture. Check generated substitutions and collect four
  matched SDK capture pairs. Integer activity falls without a useful combined
  shader speedup. Explicit inner-loop expansion and a 46-worker control also
  fail to close the gap. Preserve 660 normal comparisons and twelve diagnostics;
  revert every runtime probe.
- [x] Compare serial and independent-node dependencies with identical shader
  code and resource layouts. At 256 workers, independent matrix nodes take
  about 0.80 ms in the combined shader versus 2.97 ms with an artificial chain;
  the ordinary independent control takes about 0.71 ms. Preserve 240 normal
  comparisons and six metadata/resource diagnostics. This isolates a large
  scheduling cost but does not establish a full-model optimization.
- [x] Probe terminal-only publication elision, preserving every input acquire,
  nonterminal publication and final workgroup rendezvous. Compare 100 candidate
  pairs with 100 fresh-control pairs; no convincing full-step gain is established.
  Revert the candidate and all temporary dependency-fixture APIs.
- [x] Measure widths 1/2/4/8/16/32 with fixed code and resources. At 256
  workers, combined matrix shaders approach their independent-node timing at
  eight chains; 64 workers plateau near 1.4 ms. Real row partitioning does not
  necessarily supply that many ready tiles. Preserve 720 comparisons and
  36 diagnostics, including 18 audited dependency receipts.
- [x] Test saturating compare-exchange claims without changing publication.
  All 120 matrix and 100 full-step comparisons pass, but both model phases
  regress substantially. Revert the scheduler change.
- [x] Compile each forward implementation family separately into the matrix
  shader. Eighteen diagnostics expose a horizontal-matmul geometry mismatch:
  SmallTile grids used the default 64-wide scalar body. Correct the common
  ordinary/graph generator and distinguish scalar geometry in pipeline keys.
- [x] Qualify the geometry fix at `12f7c5a`: 665 library tests, 168 strict
  whole-graph oracle tests, nine focused GPU tests, all-target Clippy and 250
  primary pairs pass. Fresh old-binary controls show about 5% lower persistent
  forward latency and effectively unchanged training. Six SDK groups per mode
  retain the previous exact forward state and report zero local load/store
  traffic; the primary paired ratios remain 1.427 forward and 1.380 training.
- [x] Capture the corrected backward graph with the NVIDIA SDK. Preserve the
  failed cross-backend exact-state gate and audit the known gradient difference
  separately: maximum error 4.66e-9, with exact stability within each backend.
  The graph-only capture exposes substantial persistent local-memory traffic.
- [x] Probe cooperative output-address lifetimes. Saving one or two registers
  gives no useful timing improvement; revert that change and preserve its data.
- [x] Replace scalar matmul output arrays with direct stores in both paths at
  `5dbabed`. Preserve K arithmetic, geometry, bounds and epilogues. Pass 665
  library tests, 168 strict oracle cases, 57 focused GPU tests and 250 primary
  pairs. Backward SDK captures retain exact same-backend pre-change state and
  eliminate measured local traffic. A fresh 100-pair old-binary control gives
  20.725 ms training versus 19.403 ms after the change; forward is unchanged.
- [x] Separate CI host recording from completion waits. Recording is already
  faster with persistent workers; the remaining gap occurs after submission.
  Retest immutable descriptor variants after scalar stores; training still
  regresses, so revert the probe.
- [x] Extend SDK capture to complete forward/backward/SGD replay with zero
  learning rate. Keep exact within-backend replay and default exact cross-backend
  checks; add an explicit comparison using the existing benchmark tolerances.
  Qualify six counter groups and eighteen stall groups, preserving all state,
  selected graph/optimizer arithmetic and one persistent compute dispatch.
  Local traffic remains zero; barrier waiting is 51.77% of active warp cycles,
  while no-instruction waiting is below 1%.
- [x] Test flat concurrent packing across grid-axis boundaries with partial
  final claims and unchanged barrier admission. Two parser and seven GPU tests
  pass, including ragged matrix grids and 1/3/64/4096 workers. Eighty candidate
  and fresh-control comparisons show no timing improvement. Metadata disproves
  the proposed utilization benefit for SmolVLA: all sixteen eligible cooperative
  nodes already pack four tiles along their 128-wide axis. Revert the probe.
- [x] Test alternating scratch buffers for scalar Small tiles, preserving K
  arithmetic and halving their source-level loop barriers. Dynamic offsets slow
  the ordinary path; explicit constant-offset stages regress persistent timing.
  Preserve the compiler/oracle/replay checks and 120 comparisons; revert both.
- [x] Combine node and parameter broadcasts while retaining the ticket broadcast,
  empty-queue path and every payload fence. All 200 candidate/control comparisons
  pass without a useful timing gain. Matched two-group complete-step SDK captures
  show nearly unchanged instruction activity and barrier waiting. Revert the
  scheduler change and preserve the six resource diagnostics across these probes.
- [x] Test handler `DontInline` controls without changing instruction operands.
  Applying the hint to every handler fails three exact smoke checks. Restricting
  it to scalar matrix handlers passes all seven checks and eighty model
  comparisons, but raises forward/training latency to 15.518/60.716 ms versus
  6.082/19.514 ms control. Matched full-step SDK captures preserve exact
  same-backend state but expose substantial local traffic and more instructions.
  Revert both backend probes and the local dependency override; retain the
  original tests, production pins and rebuilt binaries.
- [x] Share scalar bodies across horizontal slices by selecting buffer views
  uniformly before the K loop. Preserve the selected geometry and arithmetic;
  two focused GPU tests and eighty comparisons pass. Smaller compiled shaders
  do not improve completed-step timing, so revert the probe.
- [x] Compare 16×32, 16×16 and 32×16 scalar tiles for narrow fused products.
  Reject 16×16 on timing and 16×32's ordinary forward regression. Integrate
  32×16 selection after split-K planning, preserving tuning knobs and explicit
  schedules. At `bed8c91`, 666 library, 168 strict oracle, 50 matrix regression
  and eight whole-graph tests pass, followed by 250 primary comparisons.
  Training improves from 19.459 to 19.132 ms against a fresh 100-comparison
  old-binary control; the normalized interval still overlaps no change.
  Six SDK groups per mode retain exact same-backend state and zero local traffic.
  Preserve all 950 comparisons, ten resource diagnostics and rejected variants.
- [x] Omit redundant scratch clearing for proven F32 scalar matmul bodies only.
  Preserve staging, tails, padding, K=0 behavior, lane participation, reuse
  ordering and every payload fence. Eight whole-graph and 168 strict oracle
  checks pass, followed by 400 candidate/control comparisons. Code shrinks and
  SDK estimates fall slightly, but model timing intervals include no gain and
  normalized CI SGD regresses. Revert the probe and preserve the evidence.
- [x] Test independent pointwise batches using the existing optimizer batching
  machinery. Nine focused checks pass, including partial 1,025-tile and small
  three-tile grids. Neither four tiles at a 1,024-tile threshold nor two at 512
  selects model work: its largest generated pointwise grid has only 141 tiles.
  Preserve eighty control comparisons, two resource diagnostics and the
  construction audit; generated model shaders are exactly unchanged. Revert
  both prototypes without treating these timings as a batching improvement.
- [x] Compare the existing contiguous and interleaved scalar column mappings
  with identical shapes, K stages, precision and per-output accumulation order.
  Eight whole-graph checks and 200 completed-step comparisons pass. Interleaving
  reduces estimated shared wavefronts but increases instructions and SDK replay
  time; normal training shows no useful gain. Restore the default and preserve
  two resource captures, exact same-backend replay state and 168 artifact hashes.
- [x] Restrict native cooperative products to their output-owning subgroup,
  preserving staging and all workgroup barriers. Eight whole-graph checks and
  200 comparisons pass. Four SDK groups per mode confirm roughly half the
  tensor-active work, exact prior state and zero local traffic, but completed
  steps show no useful gain. Revert the probe; preserve two resource diagnostics
  and 185 independently verified artifact hashes.
- [x] Distribute native cooperative output rows across subgroups while retaining
  the original K loop and unconditional workgroup barriers. Reject the first
  variant because its outer loop prevents four-tile packing admission; retain
  the fixed-barrier variant at `762faf2` after 666 library, 168 strict oracle,
  50 matrix and eight whole-graph checks. Final 50-pair model comparisons show persistent
  forward/training medians 2.7%/0.6% below fresh controls; normalized forward
  improvement remains inconclusive. Preserve CI inference's between-run
  variation, including the A/B/B/A follow-up, rather than claim a CI speedup.
  Keep all 1,880 normal comparisons, six resource diagnostics and both-phase
  SDK captures. Alternate subgroup widths have ownership and validation
  coverage; GPU execution remains qualified only on RTX 5080.
- [ ] Investigate the remaining scheduling/integer cost and barrier waiting.
  Sweep explicit worker counts on the current geometry, including the small CI
  graphs; preserve the requested counts and report actual execution geometry.
  Separately assess deeper scalar K staging to reduce barrier frequency,
  preserving configured choices, output arithmetic (including padded tails),
  resource admission and the existing correctness gates. Neither change is
  implemented yet. Require completed-step gains against fresh controls.
  The measured no-instruction waiting does not establish a cache bottleneck.
  Finer dependencies require proofs for payload ranges,
  allocation reuse, original tile coordinates and completion aggregation.
  Retain both model phases and all CI gates; baseline parity remains open.

The Blade extension exposes binding inspection and runtime buffer-array binding
in [commit 1bb5aef](https://github.com/kvark/blade/commit/1bb5aef0fea5fbc4249fdc40a7f1398f5c3ac012).
No additional Naga atomic syntax or memory-semantics changes are required.
Whole-graph selection rejects unsupported work instead of silently splitting
the graph. See the [whole-graph contract and results](megakernel_whole_graph.md).

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

Coverage now extends through the selected ordinary implementation instead of a
second operator registry. Naga converts its compute entry into a namespaced
callable tile, maps logical IDs and grid dimensions, and resets shared scratch
on every invocation. Exact three-dimensional workgroup geometry and subgroup
inputs are preserved. Pointwise/reduction handlers retain their existing shared
bodies. Unsupported conversion contracts produce explicit fallback. Physical
pipeline keys identify composed regions; separate implementation keys preserve
each operation's arithmetic selection and are checked by the paired harness.

Cooperative handlers require 16-byte shared allocation multiples, including a
padded scheduler ticket: a four-byte ticket disturbed cooperative scratch
placement and failed attention/convolution checks on NVIDIA. Region resource
checks include this padding and the device's reservation for workgroup-scope
matrices, and split incompatible handlers. Boolean scratch is rejected until
backend-specific layout accounting is shared. Blade
`5011dea27d0df487196ab279ef53a93aaef2607b` corrects its unused eight-binding ceiling
to the actual 64-bit Vulkan mask capacity and asserts that capacity before
creating a descriptor layout. Device buffer limits still constrain composition.
This admits nine-binding attention-gradient layouts already supported by
ordinary execution. Naga remains `f09d62e`; there is no new atomic interface.
Blade passes 20 unit tests (one ignored), strict Clippy, formatting and
Metal-target compilation. Its dependency branch is pushed.

The expanded path passes all 168 oracle and 22 GGUF tests on NVIDIA with strict
device selection, synchronization validation and no graph fallback. All 182
regression tests pass with appended optimizer fallback allowed (21 ignored).
The portable strict selection also passes all 168 oracle tests.
Device selection passes 73 smoke tests (two ignored); a strict attention
backward case also passes on lavapipe.
All six focused Session checks pass on NVIDIA and lavapipe. The new matrix and
attention fixture checks exact ordinary outputs across 1/3/64/4096 workers,
repeated changed inputs, preserved two-dimensional local geometry and scratch
reuse. Both rectangular-matrix tuning checks pass after adding the workgroup
matrix reservation to the resource bound.
All 663 library tests pass (eight ignored); strict all-target/all-feature Clippy,
formatting, documentation and core/models/GGUF/profiler Metal-target compilation
pass. Metal GPU execution remains unqualified.

At clean `55fb598`, 50-pair CI and SmolVLA measurements plus separate ten-pair
diagnostics pass exact full-state comparisons. These workloads have no ordinary
graph fallback: 5/5 and 18/18 CI operations, 154/154 and 523/523 model operations
use persistent tiles. Graph launches are 5/13 for CI and 136/363 for model
forward/training. Appended SGD remains ordinary. Median paired ratios are
1.108/1.283/1.274 for CI inference/backward/SGD and 1.930/1.643 for model
forward/training, with large training tails retained. Construction also costs
more. Ordinary remains the default; [complete records](megakernel_device_results.md#selected-kernel-tiles)
include distributions, memory, coverage, worker activity and raw-file identities.

Lifetime reuse now follows the ordinary allocation plan. Regions are planned
after allocation; pairwise physical RAW/WAR/WAW edges order old-tenant accesses
before new-tenant writes inside native regions. Dispatch barriers preserve that
order between regions. Portable independent regions reject those overlaps, and
worker-owned chains reject overlapping distinct logical references. This
preserves the original liveness proof when hardware barrier groups are merged.
Debug and explicit `no_alias` controls remain available. The convolution
optimizer reuse control now runs on the selected backend. A focused fixture
checks reduced allocations and a native WAR edge absent from the logical graph,
with 1/3/64/4096 workers, changed inputs, chunking and repeated external records.
Validation passes all 168 oracle tests in each strict portable/device selection
and 22 GGUF tests in strict device selection on NVIDIA with synchronization
validation. Device selection also passes 182 regression and 74 smoke tests
(21 regression and two smoke tests ignored). All seven focused Session checks
pass on NVIDIA and lavapipe. Strict all-target/all-feature Clippy, formatting,
documentation and core/models/GGUF/profiler Metal-target compilation pass.
At clean `23642ba`, repeated 50-pair primary and ten-pair diagnostic comparisons
remain exact. Selected residency equals ordinary residency plus scheduler bytes:
563.491 MiB model forward and 982.258 MiB training, reductions of 36.38/45.40 MiB
from the preceding implementation. Coverage and launch counts are unchanged;
one additional model training region uses native ordering. Median paired ratios
are 1.094/1.289/1.275 for CI and 1.899/1.665 for model forward/training. Ordinary
remains the default; [measurements](megakernel_device_results.md#restored-lifetime-reuse)
retain construction, distributions, diagnostics and raw identities.

Appended-pass preparation extracts the existing region compiler and handler
resource checks from graph planning. Graph-specific alias decoration and queue
dependencies remain in the planner. All seven focused checks pass on NVIDIA
and lavapipe; strict all-target/all-feature Clippy and formatting pass. Short
CI and SmolVLA paired checks have exact numerical results and identical physical
pipeline keys, implementation keys and execution reports to `23642ba`. This
preparation does not yet change appended optimizer coverage.

Appended runtime work now uses that shared compiler and the exact ordinary
SGD, Adam/LaProp, global/adaptive clipping and accumulation implementations.
Each existing dispatch becomes a portable singleton queue; phase barriers,
parameter-arena chunks, bindings and the accumulation submit/wait boundary are
preserved. Up to five queue slots per parameter pair retain separate completion
counters and optional worker histograms for every dispatch in the last recorded
step. Reset touches scheduler storage only. Explicit GPU optimizer calls reset
their queues without erasing graph completion state. Conversion/resource
failures are reported, and strict execution rejects them before graph/update
work. The explicitly requested CPU SGD helper remains outside strict scope.

`execution_report().appended` records actual runtime labels, implementation and
physical pipeline keys, logical grids and worker geometry, separately from
graph regions and configured fallback. The paired harness checks matching
ordinary/selected appended arithmetic and geometry, completion and worker
totals outside the measured interval. Appended phases report no native device
scheduling and do not reduce physical launch counts.

On NVIDIA with synchronization validation, strict device selection passes all
168 oracle, 182 regression, 75 smoke and 22 GGUF tests (21 regression and two
smoke tests ignored). Ordinary smoke also passes 75 tests. All 663 library
checks pass (eight ignored), as do strict all-target/all-feature Clippy,
formatting, documentation and core/models/GGUF/profiler Metal-target compilation.
All eight focused Session checks pass on NVIDIA and lavapipe. The new training
fixture compares exact parameters, gradients, Adam/LaProp moments, step counters
and grouped norms across 1/3/64/4096 workers with both portable and native graph
selection. It covers arena chunks, repeated external records, global/adaptive
clipping, accumulation-only steps, accumulation followed by an update, and a
separate GPU optimizer call. General model-training checks now exercise GPU
SGD; only the allocation-padding fixture's explicit CPU-update case selects an
ordinary control. Metal GPU execution remains unqualified.

At clean `d2c504d`, 50-pair CI and SmolVLA runs and separate ten-pair diagnostics
remain exact, now with no graph or appended SGD fallback. Total compute launches
including updates are 19→14 for CI training and 527→367 for model training;
graph boundaries remain unchanged. Appended queue reservations add 6,400 and
174,080 bytes respectively, and every recorded queue completes. Median paired
ratios are 1.095/1.289/1.301 for CI and 1.904/1.615 for model forward/training.
Ordinary remains the default. [Complete measurements](megakernel_device_results.md#persistent-appended-training-phases)
retain distributions, construction costs, memory, actual appended coverage,
worker diagnostics and raw-file identities. This completes the planned
integration and qualification on the declared Vulkan targets; it does not claim
single-launch execution of every graph, Metal GPU qualification or a speedup.

The device queue uses the same Session work descriptors, tracking ready tasks,
predecessor counts, unfinished work and successor edges on the GPU. It releases
consumers after whole-operation completion; measure whether tile-level
dependencies are useful before complicating that protocol.
Never occupy workers waiting on a particular unready graph node when ready
work can be executed instead.

Selected-kernel conversion now covers the tested matrix products, attention,
convolution, data movement, quantized/reduced storage and derivative paths.
Optimizer, clipping and accumulation work recorded outside `ExecutionPlan` now
uses the same conversion, resource checks and pipeline compiler. Qualification
continues through the Milestone 1 backend matrix and reports.

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
