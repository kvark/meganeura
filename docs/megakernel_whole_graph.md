# One dispatch for the entire Session graph

`PersistentOptions { whole_graph: true, device_scheduling: true, ... }` runs a
nonempty graph in one compute dispatch. A fixed physical worker pool repeatedly
claims logical workgroup tiles and schedules dependent operations on the GPU.
The step's optimizer, global/adaptive clipping and temporal accumulation are
nodes in the same dispatch. The transfer reset and compute dispatch share one
queue submission. Initial parameter/moment allocation and explicit optimizer
calls remain separate operations outside that step boundary.

For the application benchmarks, use `--whole-graph --workers 256`. For a regular
Session configured from the environment:

```sh
MEGANEURA_PERSISTENT_WORKERS=256 MEGANEURA_DEVICE_SCHEDULING=1 \
MEGANEURA_WHOLE_GRAPH=1 your_application
```

Ordinary execution remains the default. The older region implementation remains
available by leaving `whole_graph` off. Its earlier counter results do not
describe the costs of this combined shader.

## Resource and arithmetic contract

The same implementation selector and binding function supply ordinary and
whole-graph execution. A Blade visitor captures buffers and plain parameter
bytes, including the optimizer's existing segmented arena bindings. Kernels are
deduplicated by source, call and layout. Per-type descriptor arrays replace a
bind group for every operation. Node parameter records select descriptors and
carry the original uniform bytes. Uniform word offsets are fixed by each
handler's layout; descriptor indices are passed by value through its helpers.

One physical workgroup is at least 256 invocations and must be a multiple of
each selected logical workgroup. Eligible smaller tiles share the worker in
groups of two or four, each with its original logical coordinates and separate
scratch. The chosen grid axis minimizes queue claims and each claim width
divides that axis, so claims cannot straddle its boundary. Inactive groups
participate in every workgroup barrier but do not write shared/storage memory
or execute cooperative products assigned to local accumulators. Other extra-lane
calculations are still repeated. Subgroup handlers use virtual subgroup coordinates. A shader guard
checks dense, linear subgroup mapping, full groups and compatible widths before
executing graph work. This mapping is a qualification condition, not a portable
Vulkan assumption: [subgroup and local invocation indices are not generally
related](https://docs.vulkan.org/refpages/latest/refpages/source/SubgroupLocalInvocationId.html).
Unsupported mapping sets a completion diagnostic and executes no tile in that
worker. Call `validate_persistent_completion()` during qualification.

Shared arrays reuse a pool by element type and slot; capacity is the maximum
across handlers, including each handler's packing width. Original per-tile
initialization remains in each callable body. Packing retries with smaller
widths if the combined scratch exceeds the device limit.
Uniform parameters are loaded through `workgroupUniformLoad` and passed by value
through tile helpers, preserving Naga's uniform-control-flow validation.
No validation flags related to barrier uniformity are disabled.

Packing requires uniform barrier control across the physical workgroup. An
admission-only module inserts workgroup-scope cooperative operations at barriers
to apply the fork's scope-aware dependency analysis through helper calls and
locals. These probes never appear in the runtime shader. Query-dependent causal
attention loops fail this check and retain one tile per claim. Native workgroup
broadcasts and switch-containing bodies also stay unpacked; the latter avoids
relying on incomplete post-switch exit analysis. Failure to qualify packing
retains the same handler inside the single graph dispatch.

Admission checks descriptor counts, storage ranges, shared memory (including
reserved workgroup cooperative storage), and counter sizes. Imported allocations
are rejected because distinct handles may alias without a provable relationship.
Workgroup cooperative tiles must already match the physical workgroup size;
atomic side effects currently require matching geometry as well. Unsupported
resources reject whole-graph selection even when regional fallback is enabled.
This is an opt-in Vulkan path, qualified on the RTX 5080; it is not unrestricted
support for arbitrary shaders or hardware.

## Dependencies and completion

The ready bitmap contains nodes, not individual tiles. Each node has a next-tile
counter, remaining-tile counter and predecessor count. A worker claims one or
more concurrently executed tiles from a ready node; the final claim clears its
ready bit. Completion subtracts that claim's logical tile count, and the final
completion releases every successor whose predecessor count reaches zero. Zero-tile
nodes use one control task to propagate dependencies and perform no arithmetic.

Edges point forward in recording order. Conflicting reads/writes of the same
backing allocation create RAW/WAR/WAW edges, covering arena slices and lifetime
reuse. The initial implementation conservatively orders the entire allocation,
which can serialize independent slices. Transitive dependency edges are omitted
without changing reachability. Producer and consumer lanes use the existing
[native fence protocol](megakernel_memory_contract.md). Written descriptor tables
use NonPrivatePointer accesses and tile-boundary fences; read-only typed aliases
retain their explicit coherent annotation.
The ready load, tile-completion RMWs and predecessor RMWs carry the ordering
chain; every consumer lane performs its own storage visibility operation.

Idle workers keep looking for ready work until the global remaining-node count
is zero. No workgroup waits at a global barrier or requires an unscheduled
workgroup to become resident. Admission constructs a DAG, and finite executing
tiles publish their successors. Completion validation checks remaining nodes,
ready bits, remaining tiles, predecessor counters, diagnostics and optional
per-node worker claim totals.

`Session::record` creates separate immutable scheduler parameters and queue
storage for each outstanding recording. Recordings are associated with the
submission reported through `track_submission`; report submissions in recording
order before recording later unsubmitted work. Waiting for an older submission
does not permit a newer unsubmitted recording's metadata to be overwritten.
After completion, unchanged topology reuses descriptors, dependencies, queue
storage and pipeline; only uniform values are uploaded. The existing external
recording API still rejects temporal accumulation, while `step()` supports it.

`ExecutionReport::compute_dispatches` reports physical launches including the
optimizer. `appended` rows describe logical optimizer phases within the same
dispatch when `device_scheduled` is true. One queue reset is a transfer operation,
not a compute launch. A profiling interval covers the combined graph; it cannot
attribute time to its internal nodes.

## Qualification and measurements

Initial checks cover 1, 3, 64 and 4096 workers, repeated resets, empty nodes,
fan-in/fan-out, physical allocation reuse, external repeated recording, selected
matrix/attention kernels, mixed 128/256-lane reductions, and exact small-graph
parameters, gradients and optimizer state. Vulkan validation is enabled in the
GPU tests. All seven focused whole-graph tests pass. Packed matrix coverage
includes two- and four-tile claims, partial rows and repeated scratch reuse
with 1, 3, 64 and 4096 workers. The external-recording test
also waits for a submitted step while a later recording with a different
learning rate remains unsubmitted, then verifies all three parameter updates.

On RTX 5080, strict whole-graph selection passes all **168 oracle tests**,
including attention, convolution, reduced-precision storage, scatter, fusion and
backward cases. Queue completion is checked by the shared reference harness.
The earlier broad ordinary-path integration run passed **664 library, 168 oracle, 182
regression, 81 smoke and 22 GGUF tests** (8 library, 21 regression and 2 smoke
tests remain ignored).
Formatting, all-target Clippy, bare/all-feature checks and documentation pass.
The Metal library cross-check passes with `models,gguf`; all-feature cross-checking
is blocked by the Linux C compiler's unsupported Apple flags in `ring`.
No Metal GPU execution is claimed.

The concurrent-packing change passes **665 library tests** (eight ignored),
all **168 strict whole-graph oracle tests**, the seven focused graph tests,
formatting and strict all-target/all-feature Clippy. The Naga and Blade
dependency revisions are unchanged.

```sh
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
MEGANEURA_PERSISTENT_WORKERS=64 MEGANEURA_DEVICE_SCHEDULING=1 \
MEGANEURA_WHOLE_GRAPH=1 MEGANEURA_PERSISTENT_STRICT=1 \
cargo test --test oracle --all-features -- --test-threads=1
```

Logs remain on local storage under `target/whole-graph-*.log`.

## Completed-step benchmark

Clean implementation `96e57613f5e2feaba79755f8b30a9197c592d52e`, RTX 5080
(GB203), NVIDIA 595.91.07. Each row uses 50 alternating ordinary/selected pairs
after at least ten warmup pairs and one second of warmup. Parameters are restored
outside timing; each sample waits for completion and checks outputs, gradients
and updated parameters. SDK sampling, GPU timestamps, worker counters, builds
and artifact copies are disabled during primary measurements.
SmolVLA times `step()` plus completion wait; the CI boundary also includes input
upload and output/loss readback. Combined pipelines compile lazily on first use
and are warmed before timing. The harness's `construction_ms` field measures
Session construction and therefore excludes that first-use compilation; it is
not a complete cold-start metric for this backend.

| Workload | Workers | Ordinary median, ms | Whole-graph median, ms | Median paired ratio | 95% bootstrap interval |
|---|---:|---:|---:|---:|---:|
| SmolVLA forward | 64 | 4.436 | 13.241 | 2.989 | 2.972–3.016 |
| SmolVLA forward/backward/SGD | 64 | 12.302 | 45.257 | 3.495 | 3.000–3.509 |
| SmolVLA forward | 256 | 4.398 | 10.103 | 2.300 | 2.294–2.313 |
| SmolVLA forward/backward/SGD | 256 | 15.134 | 28.536 | 1.886 | 1.851–1.913 |
| CI inference | 256 | 0.0573 | 0.0757 | 1.323 | 1.318–1.332 |
| CI forward/backward | 256 | 0.0655 | 0.0994 | 1.523 | 1.508–1.533 |
| CI forward/backward/SGD | 256 | 0.0677 | 0.1054 | 1.570 | 1.551–1.582 |

All **350 comparisons** pass. SmolVLA forward and all CI cases are bitwise
equal; SmolVLA training's maximum absolute difference is `4.6566128730773926e-9`,
within the existing comparator tolerance. Both paths have identical compiled
plans and selected implementation keys. Every whole-graph case reports exactly
**one physical compute dispatch**, with no fallback. SmolVLA's ordinary path
has 154 forward dispatches and 527 forward/backward/SGD dispatches. The four SGD
rows in the whole-graph report are logical nodes in its single dispatch.

The uninstrumented scheduler allocations are 51,904 bytes for SmolVLA forward
and 98,668 bytes for training. These include metadata, immutable parameters and
both queue/reset buffers. Separate outstanding external recordings retain
additional snapshots until their submissions complete.

Intervals bootstrap each row's paired ratios 10,000 times using seed `20261025`
in table order. They describe within-run sampling uncertainty. The 64-worker
training distribution is broad, and ordinary training medians differ between
runs; a median paired ratio need not equal a ratio of marginal medians. These
data establish correctness and a regression, not a portable optimum or speedup.

## NVIDIA counter diagnosis

The Perf SDK 2026.3 forward captures use the same ten-second resident replay,
three-second monitor and 20 ms sampling interval as the
[regional investigation](megakernel_performance_analysis.md). Each capture has
146 samples after discarding the first five, successful target/monitor exits and
no SDK warnings. Ordinary, 64-worker and 256-worker captures have identical
implementation keys and bitwise-identical state, SHA-256
`4e5d66e6ce5e95142b848651d316b4df64855a9d85669499cabc3a81089be629`.

| Forward metric | Ordinary | Whole graph, 64 | Whole graph, 256 |
|---|---:|---:|---:|
| Diagnostic replay median, ms | 3.789 | 12.633 | 9.857 |
| SM active, % | 61.36 | 75.38 | 98.60 |
| Active compute warp slots, % of peak | 9.55 | 12.56 | 32.87 |
| Tensor pipe activity, % of peak | 2.66 | 3.19 | 4.14 |
| DRAM read throughput, % of peak | 11.12 | 3.35 | 4.30 |
| L2 texture-sector hit rate, % | 69.97 | 93.18 | 93.17 |
| Launched workgroups per second | 4,404,327 | 5,129 | 26,301 |
| GPC clock, GHz | 2.842 | 2.845 | 2.834 |

The lane adapter has a concrete avoidable cost: a logical 64-invocation tensor
tile executes in a physical 256-invocation workgroup. Extra subgroup lanes
repeat the tile's calculations and cooperative matrix instructions, while their
stores are masked. That preserves barrier participation and arithmetic but
performs four copies of this calculation. The product of tensor activity and
replay duration is **4.00× / 4.05×** ordinary for 64/256 workers. This is an
aggregate activity-time estimate, not an exact per-kernel instruction count;
its agreement with the explicit fourfold duplication is strong evidence for
this cost. More active warp slots do not imply more useful graph work.

Multiplying workgroup launch rate by replay duration estimates about 65 and 259
workgroups per step, close to the selected fixed pools of 64 and 256. This
independently supports the change in launch behavior. The ordinary estimate is
16,689, close to its 16,538 logical workgroups. Clocks are nearly unchanged and
DRAM throughput is far below peak, so clock throttling and aggregate bandwidth
saturation do not explain this regression.

Separate five-pair timestamp/statistics runs report **98 registers/thread and
26,112 shared bytes/workgroup** for the combined forward shader; training uses
**91 registers/thread and 26,128 shared bytes/workgroup**. Every operation pays
the combined pipeline's resource footprint. Ready-queue polling, dynamic
descriptor/parameter indexing, conservative allocation dependencies and coherent
accesses/fences add costs, but these captures do not isolate their individual
latencies. The driver's implausible approximately 64 GiB local-memory statistic
is excluded from spill claims. The timestamp training run contains a 90 ms
outlier; the table above uses the separate primary measurements.

## Optimization in progress

[Naga f4fe7e8](https://github.com/kvark/wgpu/commit/f4fe7e85790d44f46c7107675ca075a3471d6163)
checks cooperative control flow at the matrix's scope, including helper calls,
locals and loop-carried dependencies. Subgroup-uniform activity arguments can
therefore guard the extra subgroups' products. Workgroup operations retain their
existing validation; no workgroup barrier is placed inside the new condition.
The ordinary arithmetic and logical geometry are unchanged. The corresponding
[Blade pin](https://github.com/kvark/blade/commit/fd04c27dff43ab02894186d26078afeb833c9631)
uses the same Naga identity. Atomic syntax and fence semantics are unchanged.

A separate SDK capture after masking those products estimates **1.012× ordinary
tensor activity per step**, down from approximately 4×. This is the ratio of
tensor activity × replay duration, not an instruction count. Ordinary and
whole-graph replay medians are 3.793 ms and 9.277 ms; both captures have 146
counter samples and identical complete forward state. Removing duplicate tensor
work alone therefore does not explain or remove the whole slowdown. Captures are
in `target/megakernel-results/masked-products-sdk/` with commands, binary hashes
and source-status receipts. They precede the later indexing/fence changes.

Further changes cache descriptor indices as function arguments, reconstruct
uniforms with constant offsets, remove transitive dependency edges, and omit
per-access availability/visibility for descriptor globals that Naga proves are
written by this dispatch. Those globals retain NonPrivatePointer on every
access, so the existing tile-boundary fences cover communication. Read-only
aliases keep `@coherent` to prevent private-access elision. A generated-SPIR-V
check verifies both classes of table accesses, including mixed geometry and a
cooperative handler.

Qualification passes 664 library tests, all 168 strict whole-graph oracle tests,
and the six focused graph tests. Naga passes 144 unit, 268 integration and six
documentation tests (24 ignored), plus strict all-target/all-feature Clippy;
Blade passes 20 unit tests (one ignored). Meganeura's strict Clippy also passes.
Clean commit `58458d2bcd1fb4c518be714e2e11e27130b95fdd` was then measured with
the same completed-step boundaries, 50 alternating pairs per row and at least
ten warmup pairs / one second of warmup. No diagnostics or concurrent builds
were active during these measurements.

| Workload | Workers | Ordinary median, ms | Whole-graph median, ms | Median paired ratio | 95% bootstrap interval |
|---|---:|---:|---:|---:|---:|
| SmolVLA forward | 256 | 4.448 | 8.426 | 1.897 | 1.890–1.910 |
| SmolVLA forward/backward/SGD | 256 | 15.123 | 23.134 | 1.534 | 1.518–1.549 |
| CI inference | 256 | 0.0517 | 0.0665 | 1.278 | 1.252–1.423 |
| CI forward/backward | 256 | 0.0663 | 0.1003 | 1.513 | 1.487–1.522 |
| CI forward/backward/SGD | 256 | 0.0683 | 0.1037 | 1.515 | 1.477–1.537 |
| CI inference | 64 | 0.0522 | 0.0687 | 1.317 | 1.310–1.326 |
| CI forward/backward | 64 | 0.0665 | 0.0975 | 1.459 | 1.419–1.472 |
| CI forward/backward/SGD | 64 | 0.0686 | 0.1008 | 1.474 | 1.459–1.485 |
| CI inference | 16 | 0.0522 | 0.0689 | 1.316 | 1.305–1.328 |
| CI forward/backward | 16 | 0.0670 | 0.0992 | 1.475 | 1.432–1.493 |
| CI forward/backward/SGD | 16 | 0.0682 | 0.1020 | 1.495 | 1.485–1.505 |

All **550 comparisons** pass with identical plans and arithmetic selection,
one persistent dispatch and no fallback. Forward and CI state are exact;
SmolVLA training's maximum absolute difference remains
`4.6566128730773926e-9`. Bootstrap settings match the earlier table, with samples
drawn in sorted artifact-filename order and then case order. Results and SHA-256
identities are in `target/megakernel-results/whole-graph-opt-58458d2/summary.json`.
**Baseline parity has not been reached.** The SmolVLA paired ratios improve
from 2.300 / 1.886 to 1.897 / 1.534; the small CI graph still has substantial
GPU scheduling overhead. Separate diagnostic timestamps place CI inference at
about 13 / 31 microseconds of GPU time and training at 28 / 72 microseconds for
ordinary / persistent execution. These timestamps are not primary timings.

The same commit's separate SDK forward capture estimates **1.028× ordinary
tensor activity per step**, with replay medians 3.775 / 7.995 ms. Both captures
have 146 samples, exact state and successful target/monitor exits without SDK
warnings. Receipts are under the results directory's `sdk-forward/` subdirectory.
Thus the duplicate matrix work is removed, while the remaining slowdown needs
resource and scheduling improvements.

## Instruction and local-memory costs

A second SDK metric set measures instructions and memory transactions during
the same state-checked forward replay. The script accepts `--metrics` for
reproducible single-pass sets; exact metric names and commands are saved in
each receipt. The SDK's derived stall percentages require multiple passes on
this device and cannot be collected by its standalone live sampler. Individual
raw stall components can be sampled in sequential single-pass groups, as
described below; failed configuration probes are excluded from attribution.

At the `58458d2` implementation, ordinary / persistent replay medians are
3.785 / 7.988 ms. Counter rates multiplied by replay duration estimate the
following per-step work. These are whole-workload activity estimates, not
per-kernel counts or isolated timings.

| Metric | Ordinary | Persistent | Ratio |
|---|---:|---:|---:|
| Warp instructions, millions | 389.8 | 1137.6 | 2.919 |
| Predicated-on integer FMA-pipe thread instructions, millions | 701.0 | 2910.6 | 4.152 |
| Heavy FP32-pipe thread instructions, millions | 1770.1 | 1718.6 | 0.971 |
| Light FP32-pipe thread instructions, millions | 2649.1 | 2890.7 | 1.091 |
| Global-load sectors, millions | 51.11 | 136.78 | 2.676 |
| Global-store sectors, millions | 2.813 | 2.843 | 1.011 |
| Local-load sectors, millions | 0 | 14.139 | — |
| Local-store sectors, millions | 0 | 15.695 | — |
| Shared-memory wavefronts, millions | 128.96 | 208.88 | 1.620 |

The local-memory traffic is measured directly, independent of the driver's
unusable local-memory-size statistic. Explicit scalar output accumulators in
cooperative attention replace its dynamically indexed local array, using the
existing template expansion in both ordinary and persistent paths. The order
of each accumulator's operations, precision and selected kernel remain the same.
The follow-up SDK capture estimates **0.041 million local-load sectors and
0.690 million local-store sectors** per persistent step: reductions of 99.7%
and 95.6%. Replay medians are 3.797 / 7.553 ms. Both captures have 146 samples
and the same bitwise-exact state hash as the original forward capture. These
receipts are under `target/megakernel-results/whole-graph-memory-sdk/` and
`scalar-attention-sdk/`; the latter records the uncommitted candidate source.

The scalar change substantially reduces this memory cost but leaves roughly
2.88× the warp instructions, 4.29× integer-pipe instructions and 1.62× shared
wavefronts. These aggregates do not isolate the cost of queue polling,
descriptor addressing, extra lanes or scratch initialization.

The scalar-accumulator change passes 664 library tests (eight ignored), all
26 ordinary attention oracle tests, all 168 strict whole-graph oracle tests,
the six focused graph tests and strict all-target/all-feature Clippy. SDK
capture qualification also compares the complete forward state with the
unchanged arithmetic reference.

Experiments with directional payload fences, scalar parameter broadcasts,
unchanged-parameter upload elision and merged metadata broadcasts gave no
material improvement and were reverted. Per-node descriptor specialization
also gave no useful speedup and made training compilation exceed four minutes.
Claiming four tiles per queue operation regressed CI and SmolVLA and was
reverted. Saved patches and exploratory paired records remain under `target/`;
none of those experiments changes the published fence contract.

## Warp waiting and synchronization

The capture script also accepts repeated `--metric-group` arguments. Each group
must fit one hardware pass; groups run sequentially during one longer resident
replay. Every group retains its command, timestamps, raw CSV, 146 post-warmup
samples and individual counter means. The target must remain resident through
the last group and pass the same complete-state and arithmetic checks. These
groups are not simultaneous measurements.

On the `5fff909` implementation, 18 groups collect the raw active-warp rate and
17 issue states. The L1TEX component of long-scoreboard waiting and the non-RT
MIO component each fit one pass; their combined SDK metrics do not. Normalizing
each raw rate by the first group's `smsp__warps_active.avg.per_second` gives:

| Active warp-cycle state | Ordinary, % | Whole graph, 256 workers, % |
|---|---:|---:|
| Workgroup barrier | 3.623 | 50.447 |
| Long scoreboard, L1TEX | 42.303 | 21.740 |
| Fixed-latency wait | 12.913 | 9.749 |
| MIO throttle, non-RT | 17.380 | 4.707 |
| Short scoreboard | 4.032 | 4.460 |
| Selected to issue | 9.402 | 3.929 |
| No instruction | 0.588 | 0.950 |
| Memory barrier | 0.000 | 0.493 |

Clock and instruction-rate counters accompany every group. Their means vary by
at most 0.67% from the across-group mean, and the 17 states sum to 100.001% /
99.920%, supporting cross-window stability. Replay medians are 3.777 / 7.398 ms;
both modes have the same forward-state hash listed above. Target and monitor
exits succeed without SDK warnings. Receipts and the derived summary are in
`target/megakernel-results/whole-graph-stalls-sdk/`. The executable is from the
published implementation; source-status receipts disclose the modified capture
script and an unrelated untracked configuration directory.

These are **shares of active warp cycles, not shares of elapsed time**. NVIDIA
defines barrier waiting as a warp waiting for sibling warps to reach the
[workgroup barrier](https://docs.nvidia.com/nsight-graphics/UserGuide/shader-profiler.html).
Here that includes both handler barriers and inactive workers/extra subgroups
waiting for their leader or useful lanes. The data does not establish that
removing barriers would recover half the runtime. The low memory-barrier and
instruction-fetch shares argue against prioritizing another fence API change
or instruction-cache optimization. The smaller long-scoreboard percentage also
does not imply less absolute memory waiting: persistent execution has more
active warp cycles per step. A leader's atomic or fence latency can still delay
many waiting warps; a small aggregate memory-barrier share does not exclude
that critical-path cost.

A compact reproduction of the active, barrier and memory-barrier groups is:

```sh
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json \
python3 scripts/profile_smolvla_sdk.py --sdk /x/Tools/nsight-perf-sdk-2026.3 \
  --output target/new-stall-capture --phase forward --workers 256 --whole-graph \
  --metric-group smsp__warps_active.avg.per_second \
    gpc__cycles_elapsed.avg.per_second sm__inst_executed_realtime.sum.per_second \
  --metric-group smsp__warps_issue_stalled_barrier.avg.per_second \
    gpc__cycles_elapsed.avg.per_second sm__inst_executed_realtime.sum.per_second \
  --metric-group smsp__warps_issue_stalled_membar.avg.per_second \
    gpc__cycles_elapsed.avg.per_second sm__inst_executed_realtime.sum.per_second
```

Further exploratory changes moved parameters into a uniform buffer, removed the
native scratch-clear prefix, and resumed bitmap searches at the last successful
word. Each passed short paired state comparisons but gave no useful full-step
improvement and was reverted. The uniform-buffer SDK capture likewise showed no
replay improvement and increased estimated integer-pipe instructions. These
negative results narrow the remaining investigation without weakening the
memory contract or changing ordinary arithmetic selection.

An additional candidate guarded synchronization-free loops with the existing
lane-activity predicate. It preserved calls containing barriers and loops that
could return, then passed full Naga validation, 664 library tests, all 168 strict
whole-graph oracle tests and 250 completed-step paired comparisons. SDK captures
estimated 2.13× ordinary warp instructions and 1.81× global-load sectors, down
from 2.88× / 2.69×. Complete forward state remained bitwise identical. However,
the uninstrumented SmolVLA paired ratios were 1.798× forward (95% interval
1.778–1.845) and 1.509× training (1.472–1.618), overlapping the published results
below. CI ratios were 1.340× / 1.500× / 1.519×. The transformation was reverted
because it did not establish a useful timing improvement. Its patch, binary
hashes, raw samples and bootstrap summary are retained in
`target/megakernel-results/masked-loops-paired/`; separate counters are in
`target/megakernel-results/masked-loops-sdk/`.

This also bounds the conclusion from excess instructions: many can be removed
without recovering the elapsed-time gap. It motivated concurrent tile packing,
measured below, which puts spare subgroups to work on independent logical tiles
while preserving scratch isolation and the completion/fence contract. The
earlier rejected sequential four-tile batching did not provide this behavior.

## Scalar-accumulator completed-step measurement

Clean commit `36304a0` includes the scalar attention accumulators. The same
uninstrumented 50-pair protocol passes all **250 comparisons**, with exact
forward/CI state and maximum training difference `4.6566128730773926e-9`.
Every persistent case reports one dispatch and identical arithmetic selection.
All rows use 256 workers.

| Workload | Ordinary median, ms | Whole-graph median, ms | Median paired ratio | 95% bootstrap interval |
|---|---:|---:|---:|---:|
| SmolVLA forward | 4.439 | 8.047 | 1.816 | 1.804–1.858 |
| SmolVLA forward/backward/SGD | 15.133 | 22.821 | 1.510 | 1.485–1.529 |
| CI inference | 0.0520 | 0.0691 | 1.324 | 1.316–1.332 |
| CI forward/backward | 0.0665 | 0.1002 | 1.496 | 1.485–1.513 |
| CI forward/backward/SGD | 0.0677 | 0.1032 | 1.505 | 1.485–1.524 |

These results improve SmolVLA over the initial implementation but remain below
ordinary performance. CI does not use the changed attention kernel; its gap
remains. Records and hashes are under
`target/megakernel-results/whole-graph-scalar-36304a0/summary.json`, with the
same bootstrap procedure in filename/case order. The isolated final run is
the only SmolVLA record used here. An invalid ICD-path attempt and two cancelled
overlapping runs are retained and explicitly excluded in that summary.

## Concurrent tile packing: latest measurements

Commit `88d5e51` packs eligible logical tiles concurrently into the existing
worker workgroup. The matrix tiles use separate scratch ranges and preserve
their original subgroup coordinates and arithmetic. The grid axis is selected
from actual dispatch dimensions to reduce claims. Causal attention remains
unpacked because its query-dependent barrier loop fails the admission check.
The focused packing fixture covers two- and four-tile claims, partial rows,
repeated scratch reuse and worker pools from 1 to 4096. The regional selected
geometry/scratch test also passes after the shared subgroup-detection change.

All **250 completed-step comparisons** pass at 256 workers, with identical
plans and implementation keys, one physical dispatch and no fallback. Each row
uses 50 alternating pairs and at least ten warmup pairs / one second of warmup.
Forward and CI state are exact; the maximum training difference remains
`4.6566128730773926e-9`. Tracked source was clean. Receipts disclose the unrelated
untracked `configs/` directory, binary hashes, commands and start/end times;
SDK, CI and SmolVLA runs were sequential, with no profiling, builds or analysis
during the primary measurements.

| Workload | Ordinary median, ms | Whole-graph median, ms | Median paired ratio | 95% bootstrap interval |
|---|---:|---:|---:|---:|
| SmolVLA forward | 4.421 | 6.483 | 1.468 | 1.461–1.514 |
| SmolVLA forward/backward/SGD | 15.149 | 21.620 | 1.426 | 1.410–1.553 |
| CI inference | 0.05054 | 0.06830 | 1.337 | 1.250–1.440 |
| CI forward/backward | 0.06647 | 0.10046 | 1.522 | 1.499–1.529 |
| CI forward/backward/SGD | 0.06810 | 0.10078 | 1.486 | 1.469–1.505 |

The forward median falls from 8.047 to 6.483 ms compared with the preceding
scalar-accumulator measurement, while the ordinary median is nearly unchanged.
The training interval is broader and overlaps the preceding result, so it does
not establish a training speedup. CI uses no packed matrix tiles and retains
its scheduling overhead. **Baseline parity remains unmet.** Raw records and the
10,000-resample bootstrap summary, using the same seed and filename/case order,
are in `target/megakernel-results/packed-tiles-88d5e51/`.

Separate SDK captures collect six sequential single-pass groups per mode, each
with 146 samples, successful exits and no warnings. Complete forward state
retains SHA-256 `4e5d66e6ce5e95142b848651d316b4df64855a9d85669499cabc3a81089be629`.
Ordinary / persistent replay medians are 3.775 / 6.013 ms. Estimated per-step
activity, calculated as counter rate times replay duration, is:

| Metric | Persistent / ordinary |
|---|---:|
| Warp instructions | 1.869× |
| Predicated-on integer FMA-pipe thread instructions | 2.537× |
| Global-load sectors | 1.115× |
| Shared-memory wavefronts | 1.185× |
| Tensor activity × duration | 1.016× |

The corresponding pre-packing estimates were 2.88× warp instructions, 4.29×
integer-pipe instructions, 2.69× global-load sectors and 1.62× shared wavefronts.
Packing substantially reduces redundant work while maintaining the tensor work
of the ordinary implementation. These remain aggregate activity estimates.

Barrier waiting is now 62.50% of persistent active warp cycles, versus 3.73%
ordinary; memory-barrier waiting is 0.49%. Clock/instruction-rate means vary by
at most 0.33% across their sampled groups. The increased barrier share alongside
lower runtime reinforces that it is not an elapsed-time fraction. The capture
does not separate scheduler waiting from handler barriers. Receipts, raw CSVs
and the derived summary are in `target/megakernel-results/packed-tiles-sdk/`.

## Reproduction and artifact identities

On this Linux host, run each build, test and benchmark in a separate user
cgroup. The parent session must remain outside it. For example:

```sh
systemd-run --user --scope --slice=ml.slice \
  -p MemoryHigh=16G -p MemoryMax=20G -p MemorySwapMax=1G \
  cargo build --release --features models --example bench_smolvla_train --example bench_ci_latency
```

The checked-in [launcher](../scripts/run_bounded_job.py) creates a unique scope,
verifies its membership and limits from inside the child, then executes the
command. For example:

```sh
python3 scripts/run_bounded_job.py release-build -- \
  cargo build --release --features models --example bench_smolvla_train --example bench_ci_latency
```

It requires a systemd user manager and cgroup v2. Verification fails closed if
membership or limits differ. The child inherits the current directory and
environment, and its exit code is propagated. Captures keep their target and
SDK monitor in that scope. GPU jobs remain serial: wait for the previous job's
exit before launching another. Primary measurements exclude builds and analysis.
Launcher checks verified scope membership, limits and propagation of exit code 7.
The earlier local launcher remains at `/x/Tools/meganeura-run-job.py`.

A diagnostic run exposed an unrelated host-memory failure before GPU execution:
the optimizer eagerly expanded a shared expression DAG for a debug-log argument,
even when that logging target was filtered out. The kernel killed the benchmark
at approximately 29 GiB resident memory. Commit `cbf5167` logs the root term and
DAG size instead. The same debug-enabled SmolVLA workload then completed all 20
paired comparisons with 3.54 GiB peak RSS; ordinary training construction took
88 ms. The failed run contributes no timing data. Logs are
`target/head-packing-oom.log`, `target/head-packing-bounded.log` (a bounded
reproduction stopped before its memory limit), and `target/head-packing-fixed.log`.

## Subsequent experiments

Worker counts 64, 128 and 512 each passed 20 short SmolVLA comparisons. None
clearly improved both phases over 256: 64 workers measured 5.94 ms forward but
33.89 ms training; 128 measured 6.21 / 22.80 ms and 512 measured 6.14 / 21.64 ms.
These short probes are not primary performance claims. Records are
`target/packed-workers-{64,128,512}.{json,log}`.

Retesting inactive-loop suppression after packing passed 100 comparisons with
the full 50-pair protocol. Forward measured 1.448× ordinary (95% interval
1.441–1.519), training 1.418× (1.392–1.453). Both overlap the published results,
so the transformation remains reverted. The patch, binary hash and summary are
under `target/megakernel-results/packed-masked-paired/`.

Another prototype separated workgroup coordinates into scalar arguments, so
barrier analysis could admit independent attention heads while still rejecting
query-axis packing with unequal loop bounds. Three-tile claims packed SmolVLA's
15 heads. Compiler checks and the focused selected-geometry GPU test passed;
100 full-step comparisons also passed with unchanged arithmetic selection.
However, shared storage grew from 29,184 to 41,216 bytes for forward execution.
Forward measured 6.534 ms / 1.475× ordinary (1.464–1.483), training 21.645 ms /
1.431× (1.416–1.610), giving no useful improvement. The prototype was reverted;
records and its patch are under `target/megakernel-results/head-packing-paired/`.
Sorting scratch slots by size did not reduce this allocation and was also
reverted; its short probe and patch are `target/head-ranked-smolvla.*` and
`target/head-packing-ranked.patch`.

Removing the final worker-loop barrier, relying on the next queue broadcast's
rendezvous, passed all seven focused graph tests but gave no clear CI or
SmolVLA improvement. A direct-successor experiment let the final producer reserve
the successor's first tile, fencing that reservation before publishing remaining
work. It also passed the seven focused tests, 150 CI comparisons and 20 short
SmolVLA comparisons without improving either workload. Both were reverted.
Patches and records are `target/joined-claim*` and
`target/successor-reservation*`. No rejected experiment changes the published
queue or memory contract.

Further resource diagnostics at `05ed893` report **113 / 111 registers per
thread**, **29,184 / 29,200 shared bytes**, and **113,280 / 273,664 executable
bytes** for forward / training combined shaders. These are separate instrumented
runs, not primary timing measurements. Ordinary forward pipelines use 16–120
registers and at most 16,768 shared bytes. The common footprint is a plausible
occupancy cost, but these statistics do not isolate its runtime contribution.
The driver's implausible local-memory statistic remains excluded from spill
claims. Pipeline statistics require `MEGANEURA_GPU_CAPTURE=1` as well as
`--diagnostics`; records are `target/resource-stats.{json,log}`.

Five follow-ups were reverted:

| Experiment | Finding |
|---|---|
| Struct-valued node parameters | Two codegen tests and both diagnostic comparisons pass; registers and shared storage remain unchanged. |
| Reload completion metadata after the handler | Both diagnostic comparisons pass; forward registers stay 113 and training falls only from 111 to 110. |
| Physical tensor-range dependencies | Seven focused tests and 20 short comparisons pass. Forward retains 161 edges; training falls only from 706 to 698. Persistent medians are 6.312 / 21.483 ms. |
| 128-lane cooperative forward attention in both paths | The selected-geometry test and 20 comparisons pass. The combined register/shared footprint is unchanged; persistent medians are 6.211 / 21.513 ms. |
| Leader-computed tile coordinates | Seven focused tests and 20 comparisons pass. Persistent medians are 6.177 / 21.581 ms. |

The three timing probes use ten pairs per phase and at least one second of
warmup. They do not establish an improvement over the published measurements.
Their forward comparisons are exact and maximum training error is 4.66e-9.
Patches and records use `target/parameter-struct*`, `metadata-lifetime*`,
`range-dependency*`, `wide-attention*` and `leader-coordinates*`; the summary and
patch hashes are in `target/megakernel-results/resource-followups/summary.json`.

A separate SDK capture of leader-computed coordinates collects two sequential
single-pass groups per mode, each with 146 samples, successful exits and no
warnings. Forward state retains the same hash as the published capture.
Ordinary / persistent replay medians are 3.774 / 6.044 ms. Activity-time estimates
are 1.889× ordinary warp instructions, 2.552× integer-pipe instructions, 1.132×
global-load sectors and 1.204× shared wavefronts, close to the published packing
results. Thus this experiment did not remove a substantial part of the measured
excess instruction work. Receipts are in
`target/megakernel-results/leader-coordinates-sdk/`. All follow-up jobs used
separate bounded cgroups. The published runtime and performance claims remain
unchanged; parity is still unmet.

## Earlier artifact identities

All records remain on local storage at
`target/megakernel-results/whole-graph-20261025/`. The measured SmolVLA executable
has SHA-256 `fd52b1400a974c8c7a326afcc40b071310aecb61aa4cba9c5a19a64de6398637`.

```sh
cargo build --release --features models --example bench_smolvla_train --example bench_ci_latency
target/release/examples/bench_smolvla_train --whole-graph --workers 256 --warmup 10 --runs 50 --warmup-ms 1000
target/release/examples/bench_ci_latency --whole-graph --workers 256 --warmup 10 --runs 50 --warmup-ms 1000
python3 scripts/profile_smolvla_sdk.py --sdk /x/Tools/nsight-perf-sdk-2026.3 \
  --output target/new-whole-graph-capture --phase forward --workers 64 256 --whole-graph
```

| Record | SHA-256 |
|---|---|
| `smolvla-64.json` | `24a0aaa2065745d13eeeb6fba2b8f05e9ff32884c791b2dac8513b47146a9d0c` |
| `smolvla-256.json` | `69944be71d04a4c609a844c6538c2463d11b88eff3882b4d6f8799d5c53eaf3d` |
| `ci-256.json` | `e8c5e06ca134e7b6cd3a1ba157fe2fbd32c425fde3e7eef084c294257bab35c1` |
| `smolvla-256-diagnostics.json` | `0d650e390fc5843e818bef26d185e75f3cb8bdc0f703d282bb0e4c1616adeeaa` |
| `sdk-forward/ordinary-receipt.json` | `e209faec2241740d5c13306595bab3fb8defb3407b83d5563e61a05ed2ec28a1` |
| `sdk-forward/whole-64-receipt.json` | `d280ce9059afbe6f73229c35f6fb6605b5e7906f9af305593850effcf359eb35` |
| `sdk-forward/whole-256-receipt.json` | `a996d1fc0fc624c1502cdbc7e14074d6a0a028cffa3fadb3f9fb8c33870ca618` |
