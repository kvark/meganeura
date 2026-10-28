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
- [x] Sweep explicit worker counts on the current geometry with 1,140 paired
  comparisons and bracketing 256-worker controls. CI backward/SGD improve about
  4–5% at 64 workers, but 128 workers slows model training and other model
  counts show no reliable normalized gain. Preserve all requested/reported
  counts; do not introduce an automatic policy or change defaults.
- [x] Probe 64-deep scalar K staging through the shared benchmark configuration,
  preserving explicit schedules and the original 32-rounded arithmetic tail.
  Both variable-bound and fixed-inner-loop variants pass eight expanded smoke
  checks and 200 comparisons each. Neither improves both model phases; revert
  both. Retain eight resource captures and 463 independently verified hashes
  across this investigation and the worker sweep. Rebuilt binaries match the
  qualified binaries exactly, and all eight restored smoke checks pass.
- [x] Test a per-worker ready-word cursor with full wraparound scans and all
  original fences. Eight expanded smoke checks and 200 comparisons pass, but
  model performance does not improve; revert the cursor. Preserve the
  99-operation independent-root fixture and generated shaders with the probe.
- [x] Test direct handoff of an entire single-claim successor, withholding its
  ready bit while the finishing worker owns it. Shared-state variants fail a
  packed-matrix case even without executing a handoff; padding does not fix it.
  A private reservation passes eight expanded smoke checks and 200 comparisons,
  but establishes no model gain. Revert the prototype and preserve failures,
  four resource diagnostics per experiment, and the exact qualified binaries.
- [x] Reduce the extra-workgroup-state failure to a shared worker-ID scalar,
  without handoff logic. Preserve five passing and nine failing focused runs,
  one explicitly cancelled stalled diagnostic, five passing standalone matrix
  cases and 17 validated SPIR-V modules. Four/eight live words pass where
  one/two/three fail, but alignment operands do not fix it and a standalone
  shader does not reproduce it. Keep the cause unresolved and all diagnostic
  changes reverted; the qualified binaries and eight smoke checks are restored.
- [x] Audit and probe moving ordinary workgroup availability/visibility from
  individual accesses to existing barriers in Naga. Preserve NonPrivatePointer,
  decorations, atomics and synchronization placement, with a module-wide
  fallback for subgroup barriers. Five focused compiler tests, eight smoke
  checks, publication stress and 200 comparisons pass. No reliable model gain
  appears; four resource captures are unchanged. Revert the prototype and
  preserve 225 independently verified artifact hashes, including the confirmed
  pre-existing subgroup-only SPIR-V validation failure and harness failures.
  Qualified binaries and all eight restored smoke checks are unchanged.
- [x] Probe subgroup-local cooperative row staging with the original output
  geometry and arithmetic. Duplicate B staging per subgroup, mask inactive
  cooperative loads, and test subgroup-barrier lowering and compute-uniform
  subgroup-size operands in Naga. Fourteen focused compiler tests, 48 shader
  tests, eight expanded smoke tests, publication stress and 200 comparisons
  pass. Persistent forward slows 5.7%; training establishes no gain. Revert the
  probe, preserving four resource captures and 282 independently verified
  artifact hashes. Restore the exact qualified binaries and eight smoke checks.
- [x] Audit real model dependencies before changing scheduling. Independently
  reconstruct both captured hazard graphs. Whole-buffer slices offer no forward
  overlap; hypothetical training intervals remove some gradient-arena ordering.
  Identify 16 matrix/SwiGLU/matrix chains with two compatible row bands, covering
  60.7% of ordinary forward dispatch intervals and 16.9% of training-graph
  intervals. Audit original tile coordinates, packing, padded stores, allocation
  reuse and masked element-zero reads. Preserve the failed timing setup and the
  successful state-exact profiles; restore the qualified runtime and binaries.
- [x] Prototype two-band release for the audited matrix/SwiGLU/matrix pattern.
  Preserve arithmetic, grids, coordinates, packing, padded stores and masked
  element-zero reads; rebuild physical hazards and aggregate completion. Check
  five row/column shapes across four worker counts, then compare against both
  the qualified runtime and identical partitioning with whole-node dependencies.
  All 300 comparisons pass, but no useful gain is established. Preserve four
  resource records and 503 independently verified hashes; revert the prototype
  and restore exact qualified binaries and all eight smoke checks.
- [x] Retain proved scalar gradient-output ranges with whole-operation completion
  and unchanged node counts. Keep all reads and unsupported writes conservative;
  reconstruct physical hazards and test parameter-sharing rebinding. A same-binary
  control improves CI backward/SGD by 4.0/4.4%; final binaries improve them by
  5.6/5.4% against fresh qualified controls. Neither comparison establishes a
  model gain. Preserve 1,200 normal comparisons, 19 resource records, the fixture
  setup failure and seven strict matrix admission failures reproduced on the
  preceding runtime. All 666 library, 168 strict oracle, 50 original-mode matrix
  and nine whole-graph smoke tests pass, as do formatting and strict Clippy.
- [x] Compare four-float scalar matrix input staging for A+B, B only and A only,
  preserving shared layouts and arithmetic in both backends. A same-binary
  A-only control improves model forward by 4.8% but regresses CI inference by
  8.4%. Restricting staging to K >= 512 restores identical CI shaders, but the
  forward raw interval includes no gain and training improves only 0.5%.
  Preserve 1,600 normal comparisons, 32 resource records, indexing audits and
  focused tail/transpose/worker checks. Revert all variants and the temporary
  benchmark switch; restore exact qualified binaries and nine smoke checks.
- [x] Skip scalar matrix reduction work only when a whole subgroup has no valid
  output. Reject a per-thread guard: it removes FP32 work but slows persistent
  replay. The subgroup-uniform form passes 667 library, 168 strict oracle,
  50 original-mode matrix and nine smoke tests. Query compute-vote support in
  Blade and retain the original shader when it is unreported. Final CI backward
  and SGD improve 4.4/3.0%; completed-step model intervals remain inconclusive.
  SDK resident forward/training improve 3.8/3.7% with exact same-backend state.
  Preserve 1,500 normal comparisons, 30 resource records and 16 SDK captures.
- [x] Qualify uniform interior staging for F32 transposed-B matrices at
  `540eccf`, preserving original checked edge loads and selected arithmetic.
  Final completed-step model training improves 5.6% raw / 2.9% normalized;
  SDK resident training improves 2.0%. Recheck 667 library, 168 strict oracle,
  50 original-mode matrix and nine smoke tests. Preserve 2,000 comparisons
  across the rejected broad/transposed-A probes and qualified transposed-B form.
- [x] Test conditional scalar A loads with a merged value and branch-local
  stores. Both pass focused correctness checks but regress model timing.
  Preserve 1,000 comparisons and 20 resource records, then restore the runtime.
- [x] Test an explicit subgroup broadcast of the already workgroup-uniform
  claim ticket. Validate and independently invert 35 SPIR-V transformations;
  preserve 500 comparisons and ten resource records. No model-training gain
  appears and CI SGD regresses 6.6%; restore the backend and dependency lock.
  No new Naga semantics or uniformity-validation exception is needed.
- [x] Permute scalar B's shared columns without changing output ownership.
  Reject broad and transposed-B-only forms after 1,000 comparisons and 20
  resource records. Broad SDK training reduces estimated shared activity
  11.7% but increases instructions and slows replay 4.5%; preserve eight
  captures / 32 groups with exact same-backend state. Both forms slow ordinary
  training more than persistent training, so improved normalized ratios do not
  count toward parity. Restore the runtime and all nine smoke checks.
- [x] Test uniform interior A staging across F32 scalar layouts, then restrict
  it separately to transposed-A and transposed-B products. Preserve coordinate
  and source proofs, 1,500 comparisons and 30 resource records. Broad/AT-only
  forms regress CI SGD 5.4%/4.6%; no form establishes a model-training speedup.
  Reject all three, restore exact sources/locks and pass nine smoke checks.
- [x] Initialize invalid A rows once and stage only valid rows, using variable
  or fixed loop bounds. Both pass value/source proofs and focused correctness
  gates, but persistent forward/training regress 16.4%/9.7% and 12.4%/10.5%.
  Preserve 1,000 comparisons and 20 resource records; reject both, restore
  qualified sources/locks and pass nine smoke checks in separate bounded cgroups.
- [x] Bypass coordinate division for one-dimensional grids, broadly and only
  within repeated optimizer batches. Both pass 49 codegen checks, nine smoke
  tests and 168 strict oracle cases. Broad timing is inconclusive; the subset
  slows training 2.0%. Preserve 1,000 comparisons and 20 resource records.
  Broad training SDK captures reduce integer-pipe activity 6.7% with essentially
  flat replay time and exact same-backend state. Revert both; reduced instruction
  or register counts alone do not establish progress toward parity.
- [x] Hoist scalar staging bases/strides for normal A and transposed B separately.
  Prove the original checked and direct addresses, including unsigned wrapping;
  pass 49 codegen checks, nine smoke tests and eight strict matrix oracle cases
  per form. Preserve 1,000 comparisons and 20 resource records. A slows forward
  1.8%; transposed B slows CI SGD 5.7%; neither establishes a training gain.
  Revert both and recheck all nine restored smoke tests.
- [x] Refresh device-clock traces on qualified runtime `540eccf`, with 200
  diagnostic comparisons, 20 resource records and current ordinary dispatch
  profiles. Validate all 30 transformed SPIR-V modules and invert the changes
  byte-for-byte. The probe raises model registers from 102/103 to 112/111;
  CI GPU intervals grow about 10–12%. Model traces show 168 long-lived workers
  and 88 entering near completion. Handler intervals occupy about 34% of
  forward worker time and 46–50% of training worker time; these are not elapsed
  savings. Restore the runtime/fork/lock exactly and pass nine restored smoke
  tests. Preserve the trace, instrumentation costs and measurement limits.
- [x] Stage four adjacent K values in the native cooperative prologue, keeping
  scalar bindings, factor order, conversion, shared cells and synchronization.
  Audit 3.28 million cells and expand the packed fixture for K tails. Two
  independent sequences preserve 1,000 comparisons and improve persistent
  forward 3.6%/4.7%; CI and training shaders remain byte-identical. SDK forward
  replay improves 6.6%, with 18.3% less estimated integer-pipe activity despite
  more global-load sectors. Retain `9ff1b50` after full qualification: 667
  library, 168 strict oracle, 50 matrix and nine smoke tests, formatting,
  strict Clippy and Metal compilation. The repeat ratios remain 1.293× forward
  and 1.220× complete training; parity is unmet.
- [x] Probe group-zero ownership of A scratch across column-packed native
  cooperative prologues, including initialization and unchanged barriers.
  Preserve 500 completed-step comparisons and four SDK captures. Forward's
  −0.9% median change is inconclusive; SDK replay worsens 0.8% despite 7.5%
  less estimated integer activity. Global loads remain unchanged, and emitted
  A/factor loads remain outside the ownership guard. Restore `9ff1b50`, pass
  nine restored smoke tests and rebuild byte-identical qualified benchmarks.
  Preserve the rejected patch, source inverse audit and bounded-cgroup evidence.
- [x] Guard both complete A staging regions after checking local-state escape
  and side effects. Preserve another 500 comparisons and four SDK captures.
  Estimated global-load sectors fall 35.6%, integer activity 17.7% and warp
  instructions 9.3%, but forward's completed-step change is inconclusive and
  SDK replay stays flat. Restore `9ff1b50`, pass nine restored smoke tests and
  rebuild byte-identical controls. The result weakens duplicate A staging as
  an explanation for the remaining latency gap.
- [x] Reduce the wide cooperative prologue from four to two tiles per claim,
  preserving axis selection and scratch capacity, then guard all four inactive
  A/B staging regions. Preserve 1,250 normal comparisons and ten SDK backend
  captures across the two studies. The guards remove the extra staging loads,
  but normalized completed-step intervals include no change; SDK replay gains
  only about 1%. Retain neither prototype. Restore `9ff1b50`, pass nine smoke
  tests and rebuild byte-identical controls. Record source inverses, exact
  forward state and separate bounded-cgroup receipts.
- [x] Narrow eligible native cooperative prologue output tiles from 32×32 to
  32×16 through the common selector, keeping four active tiles per worker.
  Refresh grids after fusion, retain portable wide fallbacks and public API
  geometry, and distinguish eligibility in pipeline identities. Preserve 2,000
  normal comparisons and eight SDK captures. Repeat/final forward improves
  5.9%/6.4%; two other intervals include no change. SDK replay improves 6.0%
  twice. Ordinary also improves; no relative-gap reduction is established.
  Retain `d01534a` after 668 library, 168 strict oracle, 50 matrix and nine smoke
  tests, exact shader/resource bridges, Clippy, formatting and Metal compilation.
  The final paired ratios are 1.322× forward and 1.217× complete training.
- [x] Restore 2 KiB of existing f16 scratch capacity while retaining narrow
  geometry and all handler bodies. Prove the exact source inverse and
  unchanged code size/registers; preserve 500 comparisons and four SDK captures.
  Forward completed-step timing is unchanged within uncertainty; SDK replay
  rises only 0.4% and active warps stay near 32.5%. This weakens capacity as the
  explanation for the preceding 6% gain. Remove the diagnostic, pass nine
  restored smoke tests and rebuild byte-identical `d01534a` controls.
- [x] Specialize rowwise dKV's component count through the existing head-width
  pipeline key. Verify 32,896 ordered lane components, 669 library tests,
  nine smoke tests and 22 attention reference tests in both execution modes.
  Preserve 500 normal comparisons and four complete-step SDK captures.
  Ordinary registers fall 35→24 and combined code shrinks, but normal training
  timing is inconclusive and SDK replay improves only 0.65%. Reject the probe,
  restore source/tests and pass smoke/format plus byte-identical rebuilt controls.
- [x] Pack independent scalar attention heads without changing cooperative
  handlers or scratch capacity. Expose coordinate dependencies to admission;
  remove redundant exact-grid head guards while rejecting causal row packing.
  Preserve 2,700 normal comparisons, twenty diagnostics and four SDK captures.
  A larger repeat establishes a 4.68% absolute training gain; normalized gap
  improvement remains inconclusive. SDK replay improves 4.71% with exact
  same-backend state. Retain `8a63a0a` after 669 library, 168 strict oracle,
  50 matrix and ten smoke tests, Clippy, formatting and Metal compilation.
  Preserve CI's bidirectional timing movement and the test-only binary bridge.
- [x] Extend scalar packing to partial head batches while retaining subgroup
  and cooperative divisor packing. Separate padded claim limits from logical
  completion, clamp inactive coordinates/scratch and reject overflow. Emit
  the completion variable only for graphs that need partial scalar batches.
  Preserve 2,000 normal comparisons, twenty diagnostics and four SDK captures.
  The refined primary/repeat improve absolute training by 4.08%/2.02%; the
  repeat also improves its normalized comparison. CI/forward shader functions
  and resources match the control and their timing intervals include no change.
  Retain `2997a85` after 669 library, 168 strict oracle, 50 matrix and ten smoke
  tests, attention references, Clippy, formatting and Metal compilation.
  Preserve the initial CI SGD regression, exact training-shader SDK bridge,
  recovered audit exit and corrected unordered-statistics audit.
- [x] Repair inactive scratch reads before further barrier optimization. The
  previous lane adapter masked writes but let inactive groups read an active
  group's scratch during in-place updates. Numerical and coordinate tests did
  not establish race freedom. Redirect ordinary shared loads to a reserved,
  initialized read-only element and skip cooperative loads in inactive
  subgroups. Check the generated addresses and subgroup control directly;
  rerun full correctness, completed-step timing and SDK state qualification.
  Treat `2997a85` as a historical performance control with this known defect.
  Retain `137bfdd` after 670 library, 168 strict oracle, 50 matrix and ten smoke
  tests, attention references, Clippy, formatting and Metal compilation.
  Preserve 500 normal comparisons, fifteen diagnostics, four SDK backend
  captures and exact within-backend state. Model timing intervals include no
  change; CI SGD has a small 0.36% absolute regression. The final namespace
  correction has byte-identical benchmark shaders and resources.
- [x] Audit and guard inactive storage reads, including in-place writes and
  read-only aliases. A structural regression fails on the prior source; the
  emitted-shader audit verifies all 102 storage-read sites in adapted model
  handlers. Retain `e4e70c5` after 671 library, 168 strict oracle, 50 matrix and
  ten smoke tests, attention references, Clippy, formatting and Metal compile.
  Preserve 500 primary comparisons and exact SDK state. Training registers
  fall 91→80 and participating workers rise 168→252, but absolute training
  time regresses 5.58%; fewer instructions/registers do not establish a win.
- [x] Revisit leader-only polling with the corrected 80-register training
  shader. All 200 exploratory comparisons pass without a useful model gain;
  revert the probe. A separate 200-comparison same-binary worker control finds
  6.43% lower absolute training time at 168 rather than 256 workers; normalized
  uncertainty includes no change. Keep explicit counts and the 256-worker
  primary gate, and preserve all 900 normal comparisons and 24 diagnostics.
- [x] Test a serial tail for the common dQ/dKV reduction, keeping the exact
  addition tree while removing two barriers. A symbolic source audit and all
  correctness gates pass; emitted access guards and head packing are preserved.
  Preserve 500 completed-step comparisons, ten diagnostics and four SDK backend
  captures with bitwise within-backend state. Training measures 17.728→17.878 ms,
  with uncertainty including no change. Warp instructions and shared wavefronts
  fall only 1.05%, while barrier waiting stays near 64% of active warp cycles.
  Revert the patch and rebuild the retained `e4e70c5` runtime.
- [x] Test exact-order subgroup shuffles with an explicit Blade capability,
  runtime geometry check, original-tree fallback and preserved partial head
  packing. Correct Naga's compute subgroup-geometry uniformity classification
  for the experiment. All graph correctness gates pass, including a regression
  for the actual rowwise generation route. In 500 comparisons, persistent
  training improves 1.66%, but ordinary training regresses 8.12%; SDK replay
  confirms the tradeoff. Reject the common change, preserve the fork patches
  and all evidence, and restore byte-identical retained benchmark binaries.
- [x] Unroll the exact-order shuffle helper and compare retained tree, looped
  shuffles and unrolled shuffles in 750 completed-step comparisons. Unrolling
  removes the ordinary shader's extra shared allocation and improves ordinary
  training versus the looped control. A fixed 500-comparison confirmation
  improves ordinary training 3.65% versus the retained tree but establishes no
  persistent gain; the relative training gap grows 3.73%. SDK replay improves
  5.13% ordinary and 1.78% persistent. Archive the ordinary optimization
  candidate, revert the experiment and restore exact retained binaries.
- [x] Prove fully occupied packed claims across all shared-handler grids and
  restore dynamic activity before reused handlers receive partial grids.
  Preserve unchanged helper guards and ordinary shader bytes. Qualify 1,000
  completed-step comparisons, ten diagnostics and four SDK backend captures;
  confirmation establishes no absolute model gain. Archive the candidate and
  restore byte-identical retained binaries. Investigate same-binary timing
  variation before the next small optimization.
- [x] Trace same-binary variation with cgroup resources, host phases, NVML and
  calibrated GPU timestamps. Preserve 600 diagnostic comparisons and exact
  saved states. Long samples are GPU-side; untimed prewarming suppresses large
  spikes, but a substantial persistent GPU gap remains. Retain an opt-in shared
  `--prewarm-ms` control, default zero, after 50 five-case state comparisons.
  Record post-job cgroup pressure while preserving exit status, termination
  forwarding and GPU serialization. Keep the original primary gate unchanged.
- [x] Refresh device-clock attribution on retained runtime `e4e70c5`. Preserve
  600 diagnostic comparisons, four preliminary comparisons and 64 independently
  validated/reversible SPIR-V transformations. The full probe changes training
  registers and worker participation; a sparse probe preserves 80 registers
  and 252 participating workers with about 2% model GPU overhead. Narrow matrix
  stages remain expensive, while SGD overlaps backward work and leaves only
  0.64 ms after the last graph handler. Restore exact runtime/forks and binaries,
  pass all ten restored smoke tests, and archive both probes locally.
- [x] Isolate a resource/concurrency change without clock bookkeeping: append
  12 KiB of unused scratch to the training shader, preserving requested workers,
  arithmetic, indices, guards, registers and code size. Pass 1,000 completed-step
  comparisons plus ten diagnostics. Fresh SDK replay improves 7.6% with fewer
  average active warps and essentially unchanged FP32 work. Preserve 441
  verified artifacts; revert the diagnostic allocation rather than adopt a
  model-specific threshold. The remaining baseline gap stays open.
- [x] Test one complete claim broadcast with leader-loaded parameters. Correct
  the subgroup-prelude reference exposed by the unchanged packing test, then
  pass 52 codegen checks, ten GPU smoke tests and 50 diagnostic comparisons.
  Model resources and participation stay unchanged, with no material GPU gain
  in preflight. Archive the prototype and restore exact runtime/binaries;
  ten restored smoke tests pass. Preserve 319 verified artifacts and both
  initial failed checks. No new Naga API or atomic semantics are needed.
- [x] Revisit completion-gated idle polling on the repaired resource footprint.
  Retain it only for multiword bitmaps, preserving the one-word shader exactly.
  Pass 1,000 completed-step comparisons and fresh SDK captures: training improves
  7.0% under the original protocol, 5.5% with prewarming and 6.9% in SDK replay.
  Record the 1.05% original forward regression and changed register allocation;
  neither all-case speedup nor isolated polling causality is claimed. Qualify
  671 library, 168 strict oracle, 50 matrix and eleven whole-graph tests, including
  bitmap boundaries, repeated execution and 4,096 workers. Preserve 150 diagnostic
  comparisons across the unrestricted, multiword and rejected constant-width
  variants, with 1,089 verified artifacts. Keep the requested pool unchanged.
- [ ] Isolate the remaining model scheduling/instruction cost with controls that
  preserve arithmetic and completed-step boundaries. The retained dependency
  ranges and subgroup guard improve CI training without establishing model
  completed-step parity. Transposed-B staging improved training; the latest
  storage-repair comparison measures a 21.9% training gap at 256 workers. The rejected per-thread guard
  shows that fewer FP32 operations alone do not establish a speedup.
  Recheck both model phases and all
  CI cases before retaining further changes; do not infer elapsed cost from
  barrier-wait counters alone.
  Use the refreshed trace to target the forward `50×4096×720` matrix product,
  transposed training products and attention gradients. Compare each selected
  handler's ordinary resource footprint with its cost inside the combined
  worker shader before choosing a change. Preserve arithmetic and test changes
  individually; exclude clock instrumentation from normal performance gates.
  Both store-only and complete-region A sharing fail to establish a speedup.
  Two-tile packing doubles the wide prologue's claims from 64 to 128 but leaves
  half the physical groups inactive; guarding their staging does not establish
  a completed-step gain. The retained 32×16 prologue also supplies 128 claims,
  keeps four physical groups active and reduces the compiled scratch footprint.
  Restoring its previous scratch capacity does not reproduce the earlier
  latency difference, and average active warps remain near 32.5% of peak.
  Deprioritize small capacity-only reductions; per-node utilization and the
  separate costs of the changed cooperative body remain unmeasured.
  Component-count specialization does not establish a training gain. Scalar
  head packing first improved absolute training by 4.68%. Partial scalar batches
  now pack dQ and dKV in groups of four, including smaller tails, and improve
  training by another 2.02% in the larger repeat. Padded claims preserve valid
  coordinates, uniform loops and exact logical completion; the subsequent
  scratch-read audit found a race that the numerical qualification missed.
  Keep cooperative packing unchanged and preserve explicit requested worker
  counts. Continue investigating the forward gap; the latest repaired runtime's
  paired ratios are 1.355/1.219 at the unchanged 256-worker primary gate.
  The storage guard's training regression and the worker-count diagnostic
  support investigating concurrency costs, without attributing all delay to
  polling or residency. Simple leader-only polling does not resolve the gap.
  Finishing the last three attention-reduction levels in lane zero removes two
  barriers without a useful timing gain. Fresh retained control ratios are
  1.357/1.231 forward/training. Preserve that rejected probe's exact-order and
  generated-source audits; its small counter changes do not establish that
  other synchronization is free or removable.
  A subgroup-shuffle reduction removes five helper barriers, but makes ordinary
  training materially slower while improving persistent training only 1.66%.
  Its smaller persistent/ordinary ratio largely reflects the slower baseline.
  Static unrolling fixes that ordinary regression but does not establish a
  persistent completed-step gain in confirmation. Archive it as an ordinary
  optimization candidate; both shuffle variants and their dependency changes
  remain reverted. The confirmation's retained-control paired ratios are
  1.424/1.209 forward/training, with substantial forward timing variation.
  The packed-activity follow-up proves full physical-group coverage across known
  grids and demotes reused handlers before compiling a new partial grid. Only
  one matrix handler in each model phase specializes; all other shader bytes,
  including inactive access guards, remain identical. Preserve 1,000 normal
  comparisons, ten diagnostics and four SDK backend captures. Confirmation
  establishes no absolute model gain, despite 2.01% lower estimated integer
  activity; revert the patch and restore byte-identical retained binaries.
  The stability investigation localizes the large timing spikes to GPU execution
  and suppresses them with untimed prewarming. The warmed diagnostic still
  measures 15.584 ms persistent versus 11.034 ms ordinary GPU training time;
  worker counters are enabled in that diagnostic. Fresh sparse clocks preserve
  training's register count and worker participation and identify narrow matrix
  stages as the next targets. `MatMulBT[50x720x4096]` spans about 208 µs against
  127 µs in ordinary profiling; overlapping node spans are not additive costs.
  The full probe's 6.9% apparent training improvement is confounded by changed
  allocation, bookkeeping and work distribution. The independent capacity
  control now reproduces a 7.6% SDK training gain with the same 80 registers and
  code size: shared bytes grow 29,344→41,632 and participating workers change
  252→168, while all 256 workers are still requested. Active compute warps fall
  49.19%→32.76% of peak. This supports excess concurrency as a contributor;
  driver allocation, bounds, initialization and cache behavior remain possible
  contributors. The hard-coded padding is not a portable resource policy.
  Combining a claim into one broadcast leaves model participation and timings
  essentially unchanged in preflight, so do not assume broadcast count is the
  limiting cost. Completion-gated retries now provide a general multiword-bitmap
  policy without padding or worker retirement. Its training registers rise to
  82 and participation falls to 168; fresh SDK replay improves 6.9%, so allocation
  and reduced polling remain joint contributors. The complete-step training gain
  is 7.0% without prewarming and 5.5% with it, at a 1.05% original forward cost.
  Constant bitmap-width specialization does not recover that cost and is removed.
  Packing the two private epoch words into one also leaves the compiled register
  counts unchanged and shows no useful timing improvement. A validated, exactly reversible SPIR-V probe of
  directional epoch fences provides no useful GPU-time gain with the same
  allocation. Both follow-ups are archived after 125 diagnostic comparisons;
  retain the existing Naga API and qualified runtime. See the
  [epoch state and fence controls](megakernel_whole_graph.md#epoch-state-and-directional-fence-controls).
  The shared matrix selector now exposes 128 rather than 32 tiles for sixteen
  narrow backward products, reusing the existing 32×32 kernel. Another 1,000
  completed-step comparisons improve training 3.7% under the original protocol
  and 1.9% with prewarming; SDK replay improves 3.6%. All other measured shaders
  remain unchanged, and both forks keep their existing interfaces. See the
  [narrow transposed-B qualification](megakernel_whole_graph.md#narrow-transposed-b-tile-selection).
  The follow-up reuses 32×16 tiles for sixteen `50x720x4096` backward stages,
  increasing each from 46 to 90 tiles. Another 1,000 completed-step comparisons
  improve persistent training 3.6% original / 1.9% prewarmed; SDK replay improves
  2.7%. Retain the ordinary-execution tradeoff explicitly: prewarmed training
  regresses 0.6%, and SDK ordinary replay regresses 0.4%. Both backends still
  select the same implementations. See the
  [column-refinement qualification](megakernel_whole_graph.md#narrow-transposed-b-columns).
  The next qualification adds 32×32 routing for fused transposed-B products on
  fewer than sixteen original workgroups and extends fused-plain 32×16 geometry
  to K≥256. It also makes `schedule_locked` authoritative during selection.
  Another 1,000 paired comparisons improve original forward/training 4.8%/4.1%;
  SDK replay improves 3.5%/4.4%. Ordinary training improves about 1.8%.
  Prewarmed forward's larger 9.1% wall shift accompanies an 8.3% ordinary shift
  and an inconclusive normalized interval; do not attribute all of it to the
  selector. See the [fused-matrix qualification](megakernel_whole_graph.md#fused-matrix-tile-selection).
  Audit exhausted-tile claims in the single-word small-graph scheduler with a
  counter precheck, a private exhaustion bitmap and a delayed completion load.
  All three pass 52 codegen checks, fourteen smoke tests and fifty diagnostic
  comparisons each. The delayed load's short timestamp gain does not survive
  600 completed-step CI comparisons: all raw persistent intervals include one,
  and prewarmed backward's normalized ratio regresses 1.45%. Revert all probes;
  restored binaries match the qualified control and fourteen smoke tests pass.
  Preserve 932 verified artifact hashes and 44 completed cgroup records. See the
  [single-word claim controls](megakernel_whole_graph.md#single-word-claim-controls).
  Test scalar B register lookahead and current-stage B-first staging without
  increasing scratch or removing barriers. Each passes 52 codegen checks,
  fourteen smoke tests and fifty diagnostic comparisons, but persistent model
  timing regresses. B-first forward SDK replay is 7.8% slower with nearly
  unchanged estimated instruction/load/shared work and zero local traffic.
  Preserve four exact-state captures / sixteen groups, the initial corrected
  build failure, 693 verified artifact hashes and 29 completed cgroup records.
  Revert both probes and restore the exact qualified binaries and smoke checks.
  See the [scalar B staging evidence](megakernel_whole_graph.md#scalar-b-register-staging).
  Cache each lane's two per-row prologue factors across the cooperative K loop,
  preserving the original shared-cache initialization/barrier, factor order and
  inactive guards. Exactly one ordinary module and one forward graph handler
  change; all training and CI shaders remain identical. Another 1,000 completed-step
  comparisons improve persistent forward 5.8% original / 6.6% prewarmed, with
  a 5.1% improvement in the prewarmed normalized ratio. Fresh forward SDK replay
  improves 6.9% persistent / 2.3% ordinary, with 14.6% fewer estimated warp
  instructions and zero persistent local traffic. Retain the full qualification,
  four exact-state SDK captures, 616 verified artifact hashes and 49 completed
  cgroups. See the [prologue-factor qualification](megakernel_whole_graph.md#cooperative-prologue-factor-caching).
  Audit per-K-column factor reuse with per-component guards, a complete-vector
  setup block and loads inside the existing first-row vector branch. Preserve
  factor order and row/K/activity bounds; inverse audits recover the original
  ordinary shaders and isolate one changed graph handler in each probe.
  Each passes 52 codegen checks, two prologue regressions, fourteen smoke tests
  and fifty diagnostic comparisons, but persistent forward GPU time regresses
  5.8%/3.0%/2.9%. Complete-vector SDK replay regresses 2.9% despite 3.3% fewer
  global-load sectors and 5.6% fewer shared wavefronts; estimated warp/integer
  work increases. Preserve four exact-state captures, 1,017 verified artifact
  hashes and 48 completed cgroups. Revert all three probes and restore exact
  qualified binaries and smoke checks. See the
  [column-factor evidence](megakernel_whole_graph.md#cooperative-column-factor-reuse).
  Refresh sparse per-handler timings after the matrix-selector and row-factor
  changes. Eight preflight and 200 paired diagnostics pass; instrumentation adds
  1.9%/1.3% forward/training GPU time, with unchanged model participation and
  training register/shared counts. Reversal recovers all 37 original SPIR-V
  modules, and all callable handlers and ordinary shaders remain identical.
  Preserve both final training steps (14.90/13.14 ms), exact ordinary profiles,
  431 verified artifact hashes and 26 completed cgroups. Restore the runtime,
  compiler fork, lock and exact benchmark binaries; fourteen smoke tests pass.
  See the [refreshed attribution](megakernel_whole_graph.md#refreshed-handler-clock-attribution).
  Test short-K transposed-A weight-gradient geometry using existing 32×32,
  32×64 and 64×32 scalar tiles. Exactly 66 training nodes change in each probe;
  arithmetic order, storage and locked schedules remain intact in both backends.
  Each passes 52 codegen checks, nine selector tests, 51 matrix regressions,
  fourteen smoke tests and fifty diagnostic comparisons. Persistent training
  GPU time regresses 9.5%/2.5%/1.3%, despite lower code size and, for 64×32,
  fewer registers. Preserve exact per-node task audits, 923 verified artifact
  hashes and 43 completed cgroups, including two corrected non-GPU audit errors.
  Revert all three probes and restore exact qualified binaries and smoke checks.
  See the [short-K geometry evidence](megakernel_whole_graph.md#short-k-transposed-a-tile-geometry).
  Test optimizer overlap using global deferral, a recording-order chain and
  an independent-command chain ordered by prerequisite count. Preserve every
  original dependency, all 61 shader sources, pipeline resources, logical work,
  workers, fences and one dispatch/submission. Each passes 52 codegen checks,
  fourteen smoke tests and fifty full-state diagnostics. Global deferral and
  recording-order chaining regress training about 5%; the latter also postpones
  three updates until the full graph completes. Prerequisite-count chaining
  preserves earlier eligibility but establishes no gain. Four global-deferral
  training SDK captures confirm a 5.03% regression with nearly unchanged
  estimated instruction/traffic totals and higher barrier waiting. Preserve
  exact per-backend exported states, fifteen dependency audits, 1,023 verified
  artifact hashes and 42 completed cgroups. Correct the missing restoration
  receipts, preserving the failed manifest; no GPU rerun. Revert all probes,
  restore exact qualified binaries and pass fourteen final smoke tests. See
  the [optimizer-overlap evidence](megakernel_whole_graph.md#optimizer-overlap-controls).
  Inspect forward cooperative-attention staging and synchronization, where
  refreshed spans are about 20.5 µs per node versus 9.2 µs in ordinary profiling.
  Restrict score computation to its publishing subgroup, then independently
  share each row's softmax through existing score storage and two new arrays.
  Each probe passes 52 codegen checks, 22 attention references in each execution
  mode, fourteen smoke tests and fifty full-state diagnostics. Score ownership
  leaves model timings within 0.2%; shared softmax adds a barrier, 160 graph
  scratch bytes and registers without a speedup. Four forward SDK captures show
  0.57% slower persistent replay despite small estimated instruction savings.
  Preserve exact exported states, corrected source/scratch audits, 723 verified
  artifact hashes and 34 completed cgroups. Revert both probes, restore exact
  qualified binaries and pass fourteen restored smoke tests. See the
  [attention ownership evidence](megakernel_whole_graph.md#cooperative-attention-ownership-and-row-softmax).
  Audit subgroup-local sharing without additional workgroup barriers or scratch.
  Check active source lanes and matching row IDs at runtime, gated by reported
  compute shuffle/ballot support. Test one row producer and distributed keys;
  both preserve the original max/sum/PV sequence. Each passes 53 codegen checks,
  22 attention references in each mode, fourteen smoke tests and fifty full-state
  diagnostics, but forward/training GPU time regresses 0.50%/0.39% and 0.85%/0.29%.
  Preserve 6,624/6,630 symbolic ownership/order cases, exact source audits, 675
  verified hashes and 30 completed cgroups. Revert both shaders, the temporary
  Blade capability and dependency override; restore exact qualified binaries and
  pass fourteen final smoke tests. See the
  [subgroup-sharing evidence](megakernel_whole_graph.md#cooperative-attention-subgroup-sharing).
  Audit repeated forward-attention scratch initialization across six head widths
  and 3,913 sequence/mask/window cases. Test omitting converter clears, then also
  their entry barrier, preserving all shader/worker barriers and inactive guards.
  Each passes 52 codegen checks, 22 attention references in each mode, fourteen
  smoke tests and fifty full-state diagnostics. Model GPU timings stay within
  0.2% of control; no useful gain. All ordinary shaders remain exact; each graph
  changes only one attention handler. Preserve corrected temporary-name auditing,
  659 verified hashes and 32 completed bounded cgroups. Revert both probes and
  restore nineteen source/dependency hashes, exact qualified binaries and fourteen
  smoke checks. See the
  [scratch-initialization evidence](megakernel_whole_graph.md#cooperative-attention-scratch-initialization).
  Audit complete initialization in selected 16×16 f16-input cooperative matrix
  handlers. Check 3,564 parameter/tile cases, including zero K, padded staging,
  prologue factors and result storage. Omit converter clears, then their entry
  barrier; both leave model GPU diagnostics within 0.2%. Each passes 52 codegen,
  51 ordinary matrix, fourteen smoke and fifty full-state diagnostic checks.
  Expanded strict matrix sweeps pass 44 cases and reject the same seven typed/
  workgroup cases on both candidates and the unmodified baseline. Preserve these
  admission failures, exact source inverses, 660 verified hashes and 37 completed
  cgroups. Revert both probes and restore exact qualified binaries and smoke
  tests. See the [matrix reset evidence](megakernel_whole_graph.md#cooperative-matrix-scratch-initialization).
  Specialize the immutable ready-bitmap width, first for all graphs and then
  only multiword graphs, preserving exact scan order, claims, fences and workers.
  Audit frame/cache width agreement, reverse exactly two source substitutions,
  and preserve all ordinary shaders; the refinement also preserves all CI shaders.
  Each passes 676 library tests, 168 strict oracle cases, 51 matrix regressions,
  fourteen smoke and 52 codegen checks, plus 1,000 completed-step comparisons.
  Refined SDK training improves 1.02% GPU
  time, but its absolute completed-step interval includes no change. Run one
  predeclared reversed-order, 1,000-comparison confirmation: relative prewarmed
  training improves again, while every absolute model interval still includes
  one. Revert both because the absolute gain remains unestablished. Preserve all
  3,000 comparisons, 100 diagnostics, sixteen SDK groups, 908 verified artifact
  hashes and 98 completed bounded cgroups; restore exact sources/binaries and
  fourteen smoke tests. See the [ready-width evidence](megakernel_whole_graph.md#immutable-ready-bitmap-width).
  Complete the deferred direct forward comparison by selecting continuous
  bitmap scanning on the latest handler footprint. Full qualification and 1,000
  completed-step comparisons pass. Registers fall by two in each model phase;
  participation stays 168, unlike the earlier epoch qualification. Fifty GPU
  diagnostics improve 1.18% forward / 0.81% training, but every absolute model
  completed-step interval includes no change. Revert the control and retain the
  epoch policy. Preserve the corrected brace-balanced source audit, 396 verified
  hashes and 38 completed cgroups; restore exact sources/binaries and fourteen
  smoke checks. No confirmation or fresh SDK capture is claimed. See the
  [current epoch control](megakernel_whole_graph.md#completion-epoch-control-on-the-current-runtime).
  Add an advisory ready-word summary with the original full scan before epoch
  waiting. Permit stale hints without letting them authorize ownership or idle
  epochs. Check 30,948 ready/hint/worker states, 7,295 scan boundary cases and
  twenty publication/clear interleavings; expand GPU tests to 1,024/1,025 nodes.
  Full qualification and 1,000 completed-step comparisons pass. Retain `c3ad104`:
  persistent training improves 1.39% original / 1.61% prewarmed, with both pooled
  absolute intervals below one; matched SDK training improves 2.24%. Forward
  completed-step gain remains unestablished. Every handler and ordinary/CI shader
  stays exact, resources/participation remain unchanged, and each model phase
  adds eight queue bytes. Preserve all eight SDK states, 32 metric groups,
  542 verified artifact hashes and 43 completed cgroups. See the
  [summary evidence](megakernel_whole_graph.md#advisory-ready-word-summary).
  Test reducing summary-maintenance RMWs and adjusting hint publication relative
  to the existing release fence. Early publication without empty-word clears
  leaves GPU diagnostics flat; guarded RMWs regress forward 0.90% and training
  0.17%. Four training SDK backend captures show 5.56% fewer global atomic
  sectors, nearly unchanged L2 atomic traffic and no useful replay gain. Revert
  both probes after 100 diagnostics, exact shader audits, codegen/smoke checks
  and state validation. Preserve 761 verified artifact hashes and 31 completed
  cgroups; restore twenty source/dependency hashes, both qualified binaries and
  fourteen smoke tests. No normal completed-step series or full qualification
  is claimed for these rejected probes. See the
  [maintenance evidence](megakernel_whole_graph.md#advisory-summary-maintenance-controls).
  Simplify hint traversal when the summary fits one word (33–1,024 nodes),
  preserving the general path, cyclic preference, full fallback and all workers.
  Full qualification, 645,696 rotation checks, exact source audits, fifty
  diagnostics and 1,000 completed-step comparisons pass. Original training
  improves 0.71%, but prewarmed training remains inconclusive, failing the fixed
  two-protocol retention criterion. Revert without timing repeats or fresh SDK
  capture. Preserve 414 verified artifact hashes and 39 completed cgroups;
  restore twenty source/dependency hashes, both qualified binaries and fourteen
  smoke tests. See the [single-word scan evidence](megakernel_whole_graph.md#rotated-single-word-summary-scan).
  Test two-tile sequential claims for large short-K transposed-A grids. The
  initial guard excludes subgroup votes and changes no model shader; preserve
  it as an admission control. Revised looped and fixed two-call forms admit 33
  nodes but regress training GPU diagnostics 1.04%/1.15%, with registers rising
  84→103/96. Preserve the scratch-order proof, focused 255/256/289-tile and K=65
  exclusion test, 150 diagnostics and exact source inverses. Revert all three;
  verify 1,020 artifact hashes, 42 completed cgroups, twenty-one restored
  source/dependency hashes, both qualified binaries and fourteen smoke tests.
  No normal timing, full qualification or fresh SDK capture is claimed. See the
  [two-tile evidence](megakernel_whole_graph.md#targeted-transposed-a-two-tile-claims).
  Test private second-tile reservation through the existing outer loop, keeping
  every handler, case, scratch barrier and per-tile completion unchanged. Prove
  reserved-work liveness in 5,862 finite states; extend the focused smoke case
  to both eight- and forty-one-node graphs. The initial form regresses training
  GPU diagnostics 1.67%; SDK replay stays flat despite 3.51%/3.73% fewer estimated
  global/L2 atomic sectors. Compare disabled metadata with all 61 shader sources
  identical, then move reservation checks outside idle polling; the latter
  regresses 1.82%. Revert all three. Preserve 175 diagnostics, sixteen SDK groups,
  four actual full-state exports and exact source audits. Independently verify
  1,228 artifacts, 46 completed bounded cgroups, twenty-one restored source hashes,
  both qualified binaries and fourteen smoke tests. No normal completed-step
  series or full qualification is claimed. See the
  [private-reservation evidence](megakernel_whole_graph.md#private-second-tile-reservation).
  Test omission of the final scalar AT K-stage reuse barrier in both backends,
  retaining intermediate-stage and outer reuse synchronization. Verify 7,878
  scratch-order states, 804 bound cases, 52 codegen checks, fifteen smoke tests,
  exact source inverses and fifty diagnostics. Training GPU time changes −0.11%;
  sixteen SDK groups show replay +0.15% persistent / +0.04% ordinary and nearly
  unchanged persistent barrier share. Revert the probe and preserve four actual
  state exports, 448 verified artifacts, 26 completed cgroups and the corrected
  source-auditor failure. Restore source/binary hashes and fourteen smoke tests.
  See the [final-stage barrier evidence](megakernel_whole_graph.md#final-scalar-at-stage-barrier).
  Sample individual AT tile entry/exit times with immediate timestamp stores,
  keeping clocks out of handler live ranges. Preserve 200 calibrated diagnostics
  and 10,932 intervals from two final training traces covering twenty nodes.
  Training GPU time rises 1.25% with unchanged registers/shared memory. The two
  largest shapes average only 53–54 overlapping tile intervals; between-tile
  gaps can include other useful work. Exact source/SPIR-V inverses, full-state
  checks and ordinary profiles pass. Remove the probe and verify 371 artifacts,
  27 completed cgroups, restored source/binary hashes and fourteen smoke tests.
  See the [individual tile evidence](megakernel_whole_graph.md#individual-at-tile-clocks).
  Record optimizer claims alongside the sampled AT tiles, preserving existing
  four-tile claims and comparing identical shaders/storage with recording
  disabled. All 300 diagnostics pass; combined recording adds 0.13% GPU time
  over AT-only and 1.07% over qualified. Across two combined final steps, no
  optimizer body overlaps either largest sampled AT shape or occupies any
  same-worker gap between sampled AT claims. Preserve 70,628 intervals across
  both modes, exact source/SPIR-V audits and the corrected debug-label auditor.
  Remove the probe; verify 504 artifacts, 35 completed cgroups, restored source/
  binary hashes and fourteen smoke tests. See the
  [optimizer-claim attribution](megakernel_whole_graph.md#optimizer-claims-alongside-at-tiles).
  Record all graph claims, including packed axes and partial completion. Check
  3,648 geometry cases and preserve 300 calibrated diagnostics plus 248,172
  final-step intervals. The largest AT shape runs alongside about ninety BT
  bodies and finishes before BT in all 32 matched observations. Its pair
  envelope is about 159 µs versus 186 µs for ordinary BT+AT intervals; differing
  measurement contracts prevent an exact speedup claim. Full recording costs
  0.20% over sparse in pooled GPU medians, with a longer final trace preserved.
  Restore source/binary hashes and fourteen smoke tests; verify 515 artifacts
  and 37 completed bounded cgroups. See the
  [all-claim attribution](megakernel_whole_graph.md#all-graph-claim-attribution).
  Test two-tile claims for explicit F32 SwiGLUConcat/Grad grids of at least 256
  tiles. Verify exact arithmetic, mixed small/large cached handlers, odd tails,
  clamped completion and unchanged SGD batching. Pass 677 library tests,
  168 strict oracle cases, 51 matrix regressions, 53 codegen checks, fifteen
  smoke tests, Clippy and Metal compilation. Fifty diagnostics reduce model
  GPU time about 0.65%, but all four absolute model intervals include no change
  in 1,000 fixed completed-step comparisons. Revert without extra repeats or
  SDK captures; restore 22 source hashes, both qualified binaries and fourteen
  smoke tests. Preserve the corrected plan-parser and reference-pattern lint
  failures. See [two-tile SwiGLU claims](megakernel_whole_graph.md#two-tile-swiglu-claims).
  Test four-tile metadata with the same 61 shader sources and native resources
  as the two-tile candidate. All 150 three-way diagnostics pass; forward GPU
  time improves 1.68% versus qualified and 1.10% versus two tiles, while training
  is +0.34% versus qualified. Pass full qualification and 1,000 normal pairs:
  original forward improves 1.62% and prewarmed training 1.41%, but neither phase
  passes both protocols. Revert without extra repeats or SDK captures; verify
  optimizer accounting across all 1,150 comparisons, restore 22 source hashes,
  both qualified binaries and fourteen smoke tests. See the
  [four-tile metadata control](megakernel_whole_graph.md#four-tile-swiglu-metadata-control).
  Isolate forward and gradient widths through metadata in one frozen binary.
  All 61 shaders, native resources, non-width node words and dependency edges
  remain exact across four modes. Preserve 250 diagnostics; forward-only is
  the sole policy passing all predeclared advancement guards. Remove the probe
  and unused gradient repeat wrapper. The production candidate passes another
  50 diagnostics, full qualification and 1,000 normal comparisons, but neither
  model phase passes both protocols. Original forward's upper interval is
  1.0001700494; keep the criterion and revert without extra repeats or SDK
  captures. Restore 22 source hashes, both qualified binaries and fourteen
  smoke tests. See [family isolation and forward-only batching](megakernel_whole_graph.md#swiglu-family-isolation-and-forward-only-batching).
  Compare three versus four tiles for forward-only claims: all 61 shaders and
  native statistics match, and all 150 diagnostic comparisons pass correctness
  and work accounting. Three tiles offer 134 claims per 400-tile node versus
  100, but forward/training GPU medians are 0.24%/0.20% slower than four tiles.
  The predeclared advancement gate fails. Revert without additional timing or
  SDK captures; restore 22 source hashes, both binaries and fourteen smoke tests.
  See the [three-tile forward control](megakernel_whole_graph.md#three-tile-forward-claim-control).
  Stop the claim-width sweep and extend complete-claim timing to forward.
  Preserve 300 calibrated diagnostics, identical recording-mode shaders/native
  statistics/storage, all 15,898 claims per final trace and 322 physical-edge
  ordering checks. The 137-node dependency chain has 477–484 µs of predecessor-
  to-entry gaps, only about 21 µs overlapping other recorded handler bodies.
  Full recording adds 0.98% forward GPU time. Restore 21 main/three Blade source
  hashes, both qualified binaries and fourteen smoke tests; retain no probe.
  See [complete forward claim attribution](megakernel_whole_graph.md#complete-forward-claim-attribution).
  Bracket completion, ready publication, the zero-ticket claim and handler entry
  with identical disabled controls. Preserve 300 diagnostics, 63,592 final claim
  intervals, 644 edge-order checks and 1,500 unique stage timestamps. Recording
  adds 3.87% forward GPU time versus qualified and 0.47% versus disabled stage
  recording. On 136 dependent chain nodes, completion-to-publication totals
  about 142 µs, publication-to-claim is bounded by 153–184 µs, and ticket return
  to entry takes about 124 µs. The probe also enlarges gaps; no interval is
  established recoverable time. Restore all sources/binaries and fourteen
  smoke tests. See [forward publication and entry stages](megakernel_whole_graph.md#forward-publication-and-entry-stages).
  Audit whether the last-tile acquire/release fence can also supply the release
  before every successor-counter decrement. Preserve the source-level fan-in/
  fan-out argument and ten validated SPIR-V modules: each of five candidate
  graphs removes exactly one barrier, with the retained fence dominating the
  loop and no intervening payload writes. Pass 52 codegen/fourteen smoke tests,
  twenty audit comparisons and fifty production diagnostics. Forward improves
  0.60%, but training regresses 0.67%, failing the predeclared 0.5% guardrail.
  Restore all 22 source hashes, the Blade hook, both qualified binaries and
  fourteen smoke tests without extra timing or SDK captures. The memory contract
  and Naga API remain unchanged. See the
  [successor release-fence control](megakernel_whole_graph.md#successor-release-fence-control).
  Audit immutable parameter reuse on consecutive claims. Copy caching leaves
  model diagnostics flat; complete value caching increases forward registers
  from 86 to 100 and GPU time by 2.01%. Both pass 52 codegen/fourteen smoke tests
  but fail their fixed diagnostic gates. Preserve the direct shared-read
  control's two packing admission failures and stop it before GPU execution.
  Audit exact emitted post-read barrier elision with a matched loader control:
  44 validated SPIR-V pairs, fourteen smoke tests and 75 comparisons preserve
  all source, state and work checks. Training improves 0.46% against qualified
  and 0.33% against the loader control, below the 0.5% gate against both.
  Restore every probe, both compiler forks, 22 source hashes, both qualified
  binaries and restored smoke tests. Preserve all 175 diagnostics and failures
  without extra repeats, normal timing or fresh SDK captures. Close parameter
  caching/broadcast/barrier controls for this runtime; no Naga API change is
  justified. See [parameter staging controls](megakernel_whole_graph.md#parameter-staging-controls).
  Audit native cooperative B prefetch after mapping staging names to operands.
  Preserve tail bounds, alias safety, inactive reads, arithmetic order, selected
  geometry, every barrier and packing admission. Pass 52 codegen/fourteen smoke
  tests and fifty diagnostics, with exact inverses of both changed ordinary
  sources and unchanged graph schedulers. Forward/training GPU diagnostics
  regress 1.40%/1.50%, failing the fixed gate; ordinary execution stays roughly
  flat. Registers stay 86 forward and fall 84→83 in training, so register count
  alone does not explain the result. Restore all 22 source hashes, both qualified
  binaries and fourteen smoke tests without extra timing or SDK captures. See
  [native cooperative B prefetch](megakernel_whole_graph.md#native-cooperative-b-prefetch).
  Audit guarding lane-level payload fences with the existing activity mask.
  Preserve every active reader/writer's direct availability/visibility and all
  leader fences and workgroup barriers. Exact source inverses, 102 guarded
  storage-read sites and five validated SPIR-V modules establish the intended
  transformation; all 71 handler cases retain matching acquire/release guards
  around their calls. Pass full library/oracle/matrix/Clippy/Metal qualification,
  fifty diagnostics and five audit comparisons. Forward GPU diagnostics improve
  1.56%, but 1,000 completed-step comparisons improve only original forward:
  prewarmed forward and both training intervals remain inconclusive. Restore
  all 23 source hashes, the temporary Blade hook, both qualified binaries and
  fourteen smoke tests without extra runs or SDK captures. Keep the production
  memory contract unchanged. See
  [active-lane payload fences](megakernel_whole_graph.md#active-lane-payload-fences).
  Audit skipping fallback words already examined by the advisory pass in the
  same attempt, preserving an authoritative visit to every word before waiting.
  Pass 94,454 static host cases, thirty publication schedules, 52 codegen checks,
  fourteen smoke tests and fifty diagnostics. Exact inverses recover both model
  graph sources; other shaders, arithmetic, fences and allocations stay exact.
  Forward/training GPU diagnostics change +0.21%/−0.30%, failing the fixed 0.5%
  improvement gate. Restore 23 source hashes, both qualified binaries and
  fourteen smoke tests; preserve 363 verified artifacts and seventeen completed
  bounded scopes. Do not add repeats, normal timing or fresh SDK captures. See
  [disjoint ready-word scans](megakernel_whole_graph.md#disjoint-ready-word-scans).
  Audit storing forward tree GEMV scratch as adjacent scalar components so the
  graph can reuse its existing f32 bank. Preserve the exact reduction tree,
  products, guards, barriers, geometry and scheduler. Pass 52 codegen checks,
  fourteen smoke tests, four GEMV parity tests in each ordinary/strict mode and
  fifty diagnostics. Shared memory falls 4,112 bytes in each model shader;
  forward registers fall 86→77, but participation rises 168→252 and GPU time
  regresses 21.38%. Training improves 0.51%, failing the combined guardrail.
  Restore 24 source hashes, both qualified binaries and fourteen smoke tests;
  preserve 367 verified artifacts and nineteen completed bounded scopes without
  extra timing or SDK captures. Do not treat reduced resources or participation
  counts as proof of a cause or gain. See
  [scalar GEMV scratch pooling](megakernel_whole_graph.md#scalar-gemv-scratch-pooling).
  Test bounded 64/256-tick idle-poll delays against qualified execution and an
  identical zero-delay probe. Pass 52 codegen checks, fourteen smoke tests per
  probe mode, 100 diagnostics and exact source/SPIR-V inverse audits. Neither
  delay improves a model phase by 0.5% against both controls; 64 ticks also
  regresses training 1.01% against zero delay. Restore 24 main hashes, three
  Blade sources, both binaries and fourteen smoke tests; remove the temporary
  clock helper. Preserve 670 verified artifacts and 24 completed bounded scopes
  without extra timing or SDK captures. No Naga API change follows. Clock ticks
  do not guarantee elapsed delay, and this study does not measure the achieved
  polling interval or traffic. See
  [bounded idle-poll backoff](megakernel_whole_graph.md#bounded-idle-poll-backoff).
  Test spacing node counters 128 bytes apart against qualified and a dense
  control with identical shaders, capacity and reset size. Preserve every atomic
  and fence; validate all padding. Pass 52 codegen checks, fourteen smoke tests
  per layout, 75 diagnostics and exact source/layout/resource audits. Forward/
  training regress 0.82%/1.73% against dense and 1.12%/1.68% against qualified.
  Restore all 24 source hashes, both binaries and fourteen smoke tests. Preserve
  479 verified artifacts and seventeen completed bounded scopes; do not add
  spacing sweeps or SDK captures after the failed gate. No hardware contention
  cause is established. See
  [node-counter spacing](megakernel_whole_graph.md#node-counter-spacing).
  Test fetching complete node metadata only after successful tile allocation,
  retaining two scalar reads before the RMW and every original atomic/fence.
  A CPU-only export control preserves all GPU sources, resources and native
  statistics. Pass 52 codegen checks, fourteen smoke tests per build, 75
  diagnostics and exact source inverses. No model gains against both controls;
  CI SGD regresses 4.47% against control. Final-step counters show 12,902/26,133
  unsuccessful model attempts, but do not establish their elapsed-time cost.
  Restore 24 source hashes, both binaries and fourteen smoke tests; remove the
  export. Preserve 531 verified artifacts and 23 completed bounded scopes with
  no extra timing or SDK captures. See
  [late claim-metadata fetch](megakernel_whole_graph.md#late-claim-metadata-fetch).
  Specialize disabled worker histograms out of generated graph shaders, with
  matched timing-only controls and a separate fully tracked correctness path.
  Pass 52 codegen tests, fourteen smoke tests and 100 fixed diagnostics; recover
  all five original graph shaders by exact inverse and preserve all 61 tracked
  sources/statistics. Timing-only forward/training regress 5.37%/18.27% despite
  fewer registers and smaller code. Restore all 25 source hashes, both qualified
  binaries and fourteen smoke tests; remove the temporary harness selector.
  Preserve 648 verified artifacts and 24 completed study scopes, including the
  initial analyzer's corrected null-list error, without repeating GPU work.
  See [disabled worker-tracking specialization](megakernel_whole_graph.md#disabled-worker-tracking-specialization).
  Test replacing dependency RMWs with atomic stores only for single-predecessor
  destinations, retaining every fence and the original multi-predecessor path.
  Audit 33,867 finite DAGs, 707,952 transitions and 209 metadata records per mode.
  The fixed 75 diagnostics improve forward 0.59%/0.64% against qualified and an
  identical-shader zero-flag control. Remove host hooks, pass full library,
  strict oracle, matrix, smoke, Clippy and Metal compilation qualification,
  and validate five emitted SPIR-V modules plus five audit comparisons.
  The fixed 1,000 normal comparisons establish no absolute model gain under
  both protocols; normalized forward gains coincide with slower ordinary
  controls. Restore all 25 source hashes, both binaries and fourteen smoke
  tests without extra timing or SDK captures. Preserve 1,175 verified artifacts
  and 54 completed bounded scopes. See
  [single-predecessor publication](megakernel_whole_graph.md#single-predecessor-publication).
  Vary worker node preference within each ready word, preserving word traversal,
  complete fallback, every atomic/fence and all requested workers. Check finite
  bit permutations and 209 metadata records per mode; preserve an identical
  zero-rotation shader control and qualified third control. Pass 52 codegen
  tests, fourteen smoke tests per mode and 75 fixed diagnostics. Neither model
  improves 0.5% against both controls; training regresses 0.69% against zero
  rotation. Restore 25 source hashes, both binaries and fourteen smoke tests
  without extra timing or SDK captures. Preserve 909 verified artifacts and
  eighteen completed study scopes. See
  [node-bit rotation](megakernel_whole_graph.md#node-bit-rotation).
  Test vector representation of immutable parameter storage, shared staging
  and uniform values, retaining the host byte layout and every synchronization
  operation. Verify 3,072 layout/lane states, 1,579,008 word mappings and exact
  inverses of five graph shaders. Pass 52 codegen checks, fourteen smoke tests
  and fifty fixed diagnostics. Forward changes −0.08%; training regresses
  11.10% while registers fall 84→80 and participation rises 168→252. Restore
  all 25 source hashes, both binaries and fourteen smoke tests. Preserve 355
  verified artifacts and fifteen completed study scopes without extra timing
  or SDK captures. See
  [vector parameter staging](megakernel_whole_graph.md#vector-parameter-staging).
  Test strips of eight X tiles for unpacked, nonsequential claims against an
  identical shader retaining original order. Prove exact coordinate coverage
  across 17,958 grid/mode states and audit 209 metadata records per mode.
  Pass 52 codegen checks, fourteen smoke tests per mode and 75 diagnostics.
  Forward/training regress 0.47%/0.33% against the probe control; neither model
  improves against both controls. Restore 25 source hashes, both binaries and
  fourteen smoke tests without a width sweep or extra timing/SDK captures.
  Preserve 907 verified artifacts and eighteen completed study scopes. See
  [tile strip ordering](megakernel_whole_graph.md#tile-strip-ordering).
  Collect four Nsight Graphics traces with full-state checks, source correlation
  and bounded sampling bandwidth. Export and account for all 7.08M/7.18M
  persistent source samples. Ticket-broadcast waits contribute 48.20%/36.70%
  of samples; these are waiting-warp counts, not removable elapsed time.
  SGD segment search has 4.59× more samples per replay inside the persistent
  shader, while the entire SGD handler differs only 3.65%. Preserve this as
  a distinct lookup hypothesis and retain the unsupported driver-version
  caveat. Restore 26 source hashes, both qualified binaries and fourteen smoke
  tests. Archive the canceled default-bandwidth probe's pressure events as well
  as the four successful captures' zero-event receipts. See
  [Nsight Graphics source sampling](megakernel_whole_graph.md#nsight-graphics-source-sampling).
  Test SGD predecessor buckets in existing segment padding against an identical
  original-search shader control. Check 87,636 states, 1,283,708 group queries
  and every actual benchmark segment selection, preserving allocations,
  arithmetic, eligibility, batches, synchronization and workers. Pass 52 codegen
  checks, fourteen smoke tests per mode and 75 fixed diagnostics. Training
  improves 0.68% against the probe control but regresses 3.15% against qualified
  execution, failing the fixed gate. Restore 29 source hashes, both qualified
  binaries and fourteen smoke tests without further timing or SDK captures.
  Preserve 508 verified artifacts and nineteen completed study scopes. See
  [SGD segment buckets](megakernel_whole_graph.md#sgd-segment-buckets).
  Do not repeat native B lookahead, rejected scalar prefetch or subgroup-local
  staging layouts, or repeat guarded-fence or disjoint-scan timing after their
  failed fixed gates. The scalar GEMV scratch-pooling candidate is also closed
  on this runtime; do not mask its regression by silently capping workers.
  The tested idle-backoff delays are closed under the fixed gate; do not scan
  further tick values without new attribution and a distinct hypothesis.
  The tested counter spacing is also closed; preserve its dense storage control
  and regression rather than treating separated addresses as a speedup proxy.
  Late full-node fetching is closed under the fixed gate. Use the archived
  failed-claim counts only to motivate distinct policies with a progress proof;
  do not reopen rejected prechecks, exhaustion caches or priority windows.
  Disabled worker-tracking specialization is closed under its fixed gate.
  Do not infer untracked participation or occupancy from tracked histograms,
  or mask its regression with a worker cap or shared-memory padding.
  Single-predecessor atomic-store publication is closed under the fixed
  completed-step gate. Do not substitute normalized forward ratios or CI gains
  for the required absolute model improvement under both protocols.
  The tested node-bit rotation is closed under its fixed diagnostic gate.
  Structural same-word independence alone does not establish simultaneous
  readiness or recoverable time; do not sweep rotations after this failure.
  Vector parameter staging is closed under its failed diagnostic gate. Fewer
  registers and source-level vector loads do not establish faster execution;
  do not hide the training regression with padding or a worker cap.
  The tested tile strip ordering is closed under its fixed gate. Exact tile
  permutations do not establish useful locality or justify a strip-width sweep.
  The tested SGD segment buckets are closed under the failed fixed gate.
  Fewer modeled search reads and the source-sample ratio do not establish a
  speedup or justify a bucket-density sweep. A further lookup candidate needs
  a distinct mechanism and exact group-to-segment selection, including partial
  tiles and learning-rate scales. Preserve arithmetic, optimizer eligibility,
  workers and the existing diagnostic/completed-step/SDK acceptance gates.
  Further overlap or scheduling candidates need a distinct mechanism supported
  by attribution; code size, inactive-lane counts or barrier-wait percentages
  alone do not establish a cause. Require unchanged correctness, resource,
  completed-step and matched SDK gates for any candidate that advances beyond
  diagnostics.
  Neither empty recorded intervals nor participating workers establish idle
  hardware, recoverable time or occupancy. Do not repeat rejected attention
  ownership, softmax-sharing or scratch-clear probes from the span comparison.
  Preserve partial completion, small grids, SGD batching, all requested
  workers and the fixed absolute criterion under both protocols.
  Keep every completed-step gate; lower claim traffic alone is insufficient.
  Keep the short-K AT gap as an unresolved target without repeating rejected
  tile selectors or delaying optimizer eligibility.
  Do not sum overlapping spans or infer a memory bottleneck from traffic alone.
  Keep the five-case primary gate and
  add matched `--prewarm-ms 100` controls. Preserve all samples, state checks,
  requested workers and acceptance budgets; changed cache/power state is not a
  runtime optimization.
  Use qualified `c3ad104` and its archived `target/ready-word-summary/final-*`
  binaries and receipt as the next control, and retain all completed-step gates;
  instruction or traffic savings alone are insufficient retention evidence.
  Preserve initialization, writer ownership, publication, independent B storage
  and every logical tile's arithmetic and completion accounting.
  Keep the shared-state failure as a separate unresolved diagnostic.
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
