# One dispatch for the entire Session graph

`PersistentOptions { whole_graph: true, device_scheduling: true, ... }` runs a
nonempty graph in one compute dispatch. A fixed physical worker pool repeatedly
claims logical workgroup tiles and schedules dependent operations on the GPU.
The step's optimizer, global/adaptive clipping and temporal accumulation are
nodes in the same dispatch. The transfer reset and compute dispatch share one
queue submission. Initial parameter/moment allocation and explicit optimizer
calls remain separate operations outside that step boundary.

The latest qualified runtime is `12f7c5a`; its
[horizontal geometry fix and fresh measurements](#correct-horizontal-matmul-geometry)
are recorded below. Performance parity remains unmet.

For the application benchmarks, use `--whole-graph --workers 256`. For a regular
Session configured from the environment:

```sh
MEGANEURA_PERSISTENT_WORKERS=256 MEGANEURA_DEVICE_SCHEDULING=1 \
MEGANEURA_WHOLE_GRAPH=1 your_application
```

Ordinary execution remains the default. The older region implementation remains
available by leaving `whole_graph` off. Its earlier counter results do not
describe the costs of this combined shader.

On this Linux profiling host, launch GPU tests and measurements with
`python3 scripts/run_bounded_job.py LABEL --gpu -- COMMAND ...`. Each job gets a
separate verified cgroup (16 GiB soft memory limit, 20 GiB hard limit, 1 GiB swap).
The GPU flag holds a shared file lock across command execution, so marked jobs
cannot overlap. Library tests also use the GPU and need this flag. Builds use
the same launcher without the GPU flag; finish them before primary measurements.

## Resource and arithmetic contract

The same implementation selector and binding function supply ordinary and
whole-graph execution. A Blade visitor captures buffers and plain parameter
bytes, including the optimizer's existing segmented arena bindings. Kernels are
deduplicated by source, call and layout. Typed declarations alias one descriptor
array, with exactly one descriptor at offset zero per backing buffer. Node
parameters select its canonical array index and carry the original binding's
element offset and uniform bytes. This preserves descriptor identity across
types, slices and optimizer chunks, as required by shader-only synchronization.
Uniform word offsets are fixed by each handler's layout; descriptor indices and
element offsets are passed by value through its helpers.

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

SGD handlers first built for grids of at least 1,024 tiles also batch four
adjacent tiles sequentially when logical and physical workgroups match. This
amortizes queue claims, parameter broadcasts and publication across independent
updates, without changing the selected SGD body or element arithmetic. The
handler must have no workgroup scratch. A final partial batch executes and
counts only the remaining logical tiles; the cached handler can therefore also
serve later small chunks. Other optimizer handlers retain their existing claims.

Admission checks descriptor counts, storage ranges, shared memory (including
reserved workgroup cooperative storage), and counter sizes. Imported allocations
are rejected because distinct handles may alias without a provable relationship.
Workgroup cooperative tiles must already match the physical workgroup size;
atomic side effects currently require matching geometry as well. Unsupported
resources reject whole-graph selection even when regional fallback is enabled.
Storage arrays must be directly indexed and binding offsets must be divisible
by their element stride. This is an opt-in Vulkan path tested on the RTX 5080;
the [descriptor-reference audit](megakernel_memory_contract.md) supersedes earlier
qualification claims. It is not unrestricted support for arbitrary shaders or hardware.

## Dependencies and completion

The ready bitmap contains nodes, not individual tiles. Each node has a next-tile
counter, remaining-tile counter and predecessor count. A worker claims one or
more tiles from a ready node; the final claim clears its
ready bit. Completion subtracts that claim's logical tile count, and the final
completion releases every successor whose predecessor count reaches zero. Zero-tile
nodes use one control task to propagate dependencies and perform no arithmetic.

Edges point forward in recording order. Conflicting reads/writes of the same
backing allocation create RAW/WAR/WAW edges, covering arena slices and lifetime
reuse. The initial implementation conservatively orders the entire allocation,
which can serialize independent slices. Transitive dependency edges are omitted
without changing reachability. Producer and consumer lanes use the existing
[native fence protocol](megakernel_memory_contract.md). Naga combines writes
across all declarations mapped to the same descriptor binding, preserving
NonPrivatePointer accesses through typed aliases. Tile-boundary fences provide
availability and visibility, with explicit SPIR-V `Aliased` decorations.
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

The [descriptor repair measurements](#descriptor-identity-repair-current-measurements)
are the current qualification record. Measurements before `ed81459` below are
historical hardware observations: they predate the descriptor-reference repair
and do not establish its memory-model contract.

Initial checks cover 1, 3, 64 and 4096 workers, repeated resets, empty nodes,
fan-in/fan-out, physical allocation reuse, external repeated recording, selected
matrix/attention kernels, mixed 128/256-lane reductions, and exact small-graph
parameters, gradients and optimizer state. Vulkan validation is enabled in the
GPU tests. All seven focused whole-graph tests pass. Packed matrix coverage
includes two- and four-tile claims, partial rows and repeated scratch reuse
with 1, 3, 64 and 4096 workers. The external-recording test
also waits for a submitted step while a later recording with a different
learning rate remains unsubmitted, then verifies all three parameter updates.
The optimizer fixture includes a 1,025-tile SGD chunk with a partial final
batch, followed by small chunks reusing the handler, across all four worker
counts and repeated optimizer/reset stages.

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

## Concurrent tile packing measurements

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

## Worker clock diagnostics

A temporary Blade instrumentation patch inserts device-scope
`OpReadClockKHR` instructions at explicit marker calls, without changing Naga's
language or validation rules. Each transformed module passes `spirv-val` before
pipeline creation. The device clock and Vulkan memory-model device-scope
features are queried and enabled. These patches are diagnostic artifacts, not
part of the runtime dependency. Source and raw records are under
`target/clock-profile/`; the first capture is preserved in its `first/` directory.

In that capture, worker entry-to-exit envelopes are 6.421 ms forward and
21.035 ms training, within 0.04% of the corresponding dispatch timestamps.
Task counts agree with the logical grids and all recorded intervals are
contained in worker lifetimes. There are 168 long-lived workers; the other 88
enter near completion. Summed handler intervals occupy 27.47% / 45.39% of
summed worker lifetimes. These are worker-time fractions, **not elapsed-time
savings**: many workers wait while a narrow node has too few tiles to occupy
them. Handler intervals include the publication fence and workgroup rendezvous.
Instrumentation also raises register counts to 126 / 123, and clock reads do
not guarantee compiler code-motion barriers, so this is diagnostic attribution.

Scalar matrix handlers dominate the forward trace. Its 50 `MatMul` nodes have
3.671 ms of summed node spans and 32 `FusedMatMulAdd` nodes have 2.625 ms;
cooperative attention contributes 0.307 ms. Spans can overlap and must not be
added as elapsed time. Separate ordinary dispatch profiles retain bitwise exact
complete state and show materially shorter matrix intervals. For example,
`FusedMatMulAdd[50x720x2048]` is about 63 microseconds ordinary versus
115–118 microseconds between the first and last persistent worker timestamps.
The instrumentation differs, so this does not establish an exact kernel ratio.
Training also exposes roughly 95,000 small SGD tiles as a queue-batching target.
Ordinary profiles and comparisons are `target/clock-profile/ordinary.*` and
`compare-{forward,backward}.json`.

Three compiler-oriented experiments followed:

| Experiment | Finding |
|---|---|
| Subgroup-uniform parameter hints | Ten paired comparisons pass, but forward registers remain 126 and timing does not show a useful gain. No Naga uniformity change is retained. |
| Split read-only descriptor views and rely on existing tile fences for read visibility | All 20 uninstrumented comparisons pass. Forward improves to 5.228 ms, but training regresses to 27.519 ms. A separate diagnostic run reports 140 / 139 registers, up from 113 / 111. |
| Fence-based visibility with the original descriptor views | All 20 comparisons pass; forward / training medians are 6.317 / 21.649 ms, with large training outliers. No improvement is established. |

Read-only splitting therefore has a substantial resource tradeoff. None of
these experiments is retained, and no memory-semantics change is proposed from
the short correctness probes. Patches and receipts are
`target/clock-profile/{parameter-hint,readonly-clock,fence-visibility}*` and
`readonly-primary-meganeura.patch`; timing records are
`target/{parameter-hint-probe,readonly-primary-smolvla,fence-visibility-smolvla}.{json,log}`.
All jobs used separate bounded cgroups and local storage.

## Large SGD claim batches

The first four-tile SGD prototype batches every grid. All seven focused tests
and 250 paired comparisons pass, but small CI training regresses to 0.1108 ms
from the published 0.1008 ms. Restricting the transformation to large initial
grids removes that regression. Sixteen-tile claims offer little further benefit
in short SmolVLA probes, so the retained width is four. The selected SGD body,
element arithmetic and graph submission count remain unchanged.

At **`b23d422`**, all **250 final comparisons** pass with the same 50-pair,
256-worker protocol and isolated measurements. Forward and CI comparisons are
exact; maximum SmolVLA training error is 4.66e-9. Every selected case reports one
compute dispatch and no fallback, with identical plans and arithmetic selection.
All 665 library tests (eight ignored), 168 strict whole-graph oracle tests,
seven focused tests, formatting and strict all-target/all-feature Clippy pass.

| Workload | Ordinary median, ms | Persistent median, ms | Paired ratio | 95% bootstrap interval |
|---|---:|---:|---:|---:|
| SmolVLA forward | 4.441 | 6.510 | 1.465 | 1.453–1.474 |
| SmolVLA forward/backward/SGD | 15.129 | 20.965 | 1.390 | 1.378–1.608 |
| CI inference | 0.05213 | 0.07008 | 1.343 | 1.338–1.351 |
| CI forward/backward | 0.06603 | 0.10120 | 1.533 | 1.519–1.538 |
| CI forward/backward/SGD | 0.06737 | 0.10095 | 1.498 | 1.483–1.519 |

A fresh, separate checkout of the prior runtime (`3628005`) passes another
100 SmolVLA comparisons under the same protocol. Its forward median is 6.497 ms
(1.475× ordinary, interval 1.463–1.507), and training is 21.552 ms
(1.424×, 1.400–1.453). Large-grid batching lowers the observed training median
by 2.7%; the paired-ratio intervals overlap, and these separate runs do not
establish a statistically isolated speedup. The small-grid penalty is avoided,
and the transformation reduces large SGD queues to roughly one claim per four
logical tiles. **Baseline parity remains unmet.**

The qualified executables, source/binary hashes, commands, cgroup identities,
successful exits, raw samples and summaries are under
`target/megakernel-results/sgd-large-paired/`; the contemporary comparison is in
`sgd-contemporary-baseline/`. Both use clean tracked source. Prototype records
are `sgd-batch4-paired/` and `target/sgd-batch{,16}-smolvla.{json,log}`. All
measurements exclude profiling, builds and concurrent analysis.

## Descriptor identity repair: current measurements

At **`ed81459`**, Naga `bc7b70b` and Blade `c09c89f` preserve one descriptor
reference per backing buffer across typed views and nonzero offsets. No
publication fence or selected arithmetic is removed. Isolated reruns against
the published pins pass **665 library tests** (eight ignored), **168 strict
whole-graph oracle tests**, **eight focused GPU tests**, and strict Clippy.
The [memory contract](megakernel_memory_contract.md) records compiler validation
and the eight workspace failures already present in the saved baseline.

All **250 primary comparisons** pass: 50 alternating AB/BA pairs for each case,
256 workers, at least ten warmup pairs and one second of warmup. Tracked source
is clean; receipts disclose unrelated untracked `configs/`. Every selected case
uses one compute dispatch and the same arithmetic, without fallback. Forward
and CI state are exact; maximum SmolVLA training error is **4.66e-9**. Identical
state is restored outside timing, and qualification readbacks are excluded.

| Workload | Ordinary median, ms | Persistent median, ms | Paired ratio | 95% bootstrap interval |
|---|---:|---:|---:|---:|
| SmolVLA forward | 4.448 | 6.608 | 1.489 | 1.484–1.494 |
| SmolVLA forward/backward/SGD | 15.154 | 20.884 | 1.375 | 1.358–1.396 |
| CI inference | 0.05220 | 0.07068 | 1.353 | 1.345–1.374 |
| CI forward/backward | 0.06598 | 0.10280 | 1.557 | 1.547–1.565 |
| CI forward/backward/SGD | 0.06745 | 0.10130 | 1.510 | 1.489–1.523 |

Intervals bootstrap paired-ratio medians 10,000 times with seed 20261025 and
describe within-run uncertainty. The repair is retained for correctness;
these measurements do not show performance parity or an isolated speedup.

A separate NVIDIA Perf SDK forward capture collects six sequential groups per
mode, each with 146 post-warmup samples. All monitor/target processes exit zero,
without SDK warnings, and both modes produce identical state (SHA-256
`4e5d66e6ce5e95142b848651d316b4df64855a9d85669499cabc3a81089be629`). Diagnostic
replay medians are **3.772 ms ordinary / 6.103 ms persistent**. Cross-group clock
means stay within 0.25% and warp-instruction rates within 0.70% of their means.

| Estimated activity per forward replay | Persistent / ordinary |
|---|---:|
| Warp instructions | 1.924× |
| Integer-pipe thread instructions | 2.552× |
| Global-load sectors | 1.117× |
| Shared-memory wavefronts | 1.193× |
| Tensor activity | 1.017× |
| DRAM read activity | 1.000× |

Persistent execution also has about 39,854 local-load and 678,795 local-store
sectors per replay, versus zero ordinary. Workgroup-barrier waiting is **62.28%**
of active warp cycles, versus **3.75%** ordinary; memory-barrier waiting is
**0.47%** versus zero. These are active-warp shares, not elapsed-time fractions
or recoverable-runtime estimates. Counter rates times replay duration estimate
activity; the peak-normalized tensor and DRAM metrics support relative comparison.

A separate pipeline-statistics capture reports **112 registers** in both
combined shaders, with **29,200 / 29,232 bytes** of shared storage and
**114,560 / 272,000 bytes** of executable code for forward/training. The driver's
implausible approximately 64 GiB per-thread Local Memory Size field is excluded;
local traffic above comes from the SDK counters.

The measured gap still includes substantial extra integer/addressing and worker
coordination work, with waiting workers contributing to barrier stalls. These
counters do not isolate each handler's elapsed cost. The follow-ups below do
not improve both phases. Worker placement is now measured, while spill
attribution remains open. Retained changes must pass both phases' existing gates.

Primary binaries, commands, hashes, cgroup identities, receipts and raw samples
are in `target/megakernel-results/canonical-descriptors-paired/`; SDK records are
in `canonical-descriptors-sdk/`. All GPU jobs use separate bounded scopes and
the shared GPU lock. No builds, profiling or analysis overlap primary timing.
Library tests also execute GPU work: an earlier overlapping test attempt was
followed by isolated library, focused and oracle reruns before these measurements.

## Immutable allocations and occupancy follow-ups

An audit at `4469a03` includes every appended optimizer command and every slice
of a backing allocation. Forward has **147 immutable allocations out of 170**,
totaling **399,576,768 bytes**. Training has only **13 out of 205**, totaling
**187,056 bytes**, because optimizer updates make the weights mutable.

A temporary implementation separated immutable and mutable descriptor tables,
retaining one offset-zero descriptor per physical buffer and aliasing typed
views within each table. Distinct immutable-binding masks required 18 forward
handlers instead of 15, and 39 training handlers instead of 33. It preserved
the mutable publication protocol and allowed private reads only for storage
never written anywhere in the step.

A second version conservatively demoted all allocations sharing a handler
binding with mutable storage, propagating until no further demotions occurred.
That retained one handler per kernel and one descriptor per buffer. It left
137 immutable forward buffers (399,400,768 bytes) and three immutable training
buffers (11,056 bytes).

These are **exploratory** runs: 20 alternating pairs per phase, at least ten
warmup pairs and one second of warmup, with the same completed-step boundary
and full state comparison. They do not replace the 50-pair primary results.

| Experiment | Forward persistent ms / paired ratio | Training persistent ms / paired ratio | Diagnostic registers, forward / training |
|---|---:|---:|---:|
| Immutable handler variants | 5.941 / 1.334 | 25.163 / 1.625 | 143 / 140 |
| Immutable bindings shared by every invocation | 6.522 / 1.509 | 20.830 / 1.357 | 112 / 105 |
| Original shader, 84 workers | 5.564 / 1.410 | 24.633 / 1.663 | Not recaptured |
| Original shader, 256 workers, fresh control | 6.583 / 1.568 | 21.110 / 1.373 | Not recaptured |
| Uniform broadcast inside each handler | 6.512 / 1.488 | 20.800 / 1.388 | 123 / 118 |

All **200 comparisons** in these five completed trials pass; forward is exact
and maximum training error is 4.66e-9. Each persistent step uses one dispatch
and unchanged selected arithmetic. The variant trial's forward gain coincides
with higher register use and a training regression. The 84-worker control
has the same direction of tradeoff, making occupancy a plausible contributor;
these runs do not establish SM placement or isolate a read-only-load benefit.
Moving the uniform broadcast inside each selected handler increases registers
and shows no useful gain. None of these implementation changes is retained.

Two incomplete attempts are also recorded. The initial variant generator
renamed a storage struct still referenced by its typed table; shader validation
rejected it, and shared type declarations were preserved for the completed
trial. A separate 64 KiB shared-memory reservation intended to constrain
residency was rejected by the runtime's 48 KiB workgroup limit before dispatch.
No admission or validation check was disabled.

Patches, preserved binaries, samples, separate pipeline captures, cgroup logs
and a hashed summary are in `target/megakernel-results/immutable-variants/`.
GPU jobs were serialized in bounded scopes and their exits confirmed before
the next GPU job. Builds and analysis did not overlap measurements. The
published runtime and compiler pins remain the qualified canonical-descriptor
implementation above; the rejected probes did not proceed to its full
library/oracle/CI acceptance gate.

## Worker placement and compiler follow-ups

Temporary instrumentation based on
[`VK_NV_shader_sm_builtins`](https://docs.vulkan.org/refpages/latest/refpages/source/VK_NV_shader_sm_builtins.html)
records the leader's SM ID at entry and exit alongside per-node worker counts.
Both instrumented SPIR-V modules pass Vulkan 1.3 validation before pipeline
creation. In the 256-worker capture, **168 workers perform tasks on 84 SMs,
exactly two per SM**. Every leader's entry and exit IDs match. The first
46-tile matrix node uses only 23 SMs; sixteen `FusedMatMulAdd[50x720x2048]`
nodes use a median of 33 SMs. With 84 workers, every active worker has its own
SM. These are observations on this device/driver, not a portable placement rule.

A probe elects one worker per SM for nodes with at most 84 claims, allowing all
workers to handle larger nodes. It raises the narrow fused-matrix family's
median SM count from 33 to 46, but does not improve the matching instrumented
control. Both use 116 forward / 115 training registers. With 20 alternating
pairs per phase, control persistent medians are **7.118 / 22.511 ms** and owner
medians are **7.214 / 23.014 ms**. The SM instrumentation and capture settings
affect compilation; these timings do not replace the uninstrumented primary
results. The owner policy is reverted. Clustering alone does not establish the
cause of the elapsed-time gap.

The driver's pipeline-executable query returns **zero internal representations**
for both combined shaders, so this route supplies no ISA or spill locations.
Further compiler probes retain the selected arithmetic and memory protocol:

- An SPIR-V `Inline` hint leaves 112 registers and executable sizes unchanged.
- Actual inlining, after dead-branch elimination and return merging, produces
  one function with no calls in each module. Validation and state comparisons
  pass, but registers remain 112 and code grows to 115,968 / 274,688 bytes.
  Normal 20-pair medians are 6.571 / 21.069 ms, with ratios 1.492 / 1.411.
- A descriptor-uniformity probe removes 102 / 200 `NonUniform` decorations,
  restricted to accesses rooted at the canonical payload binding. The builder
  supplies every descriptor index through `workgroupUniformLoad`, and Vulkan
  permits [dynamically uniform descriptor indexing within a workgroup](https://docs.vulkan.org/guide/latest/extensions/VK_EXT_descriptor_indexing.html).
  The corresponding core feature/capability is enabled; alias decorations and
  memory-access semantics remain intact. Validation and state comparisons pass,
  but registers and executable sizes are unchanged. Normal 20-pair medians are
  6.564 / 20.948 ms, with ratios 1.490 / 1.413.

Neither compiler probe shows a useful gain, and neither is retained. The initial
inliner warned that early returns prevented complete inlining; return merging
then rejected unreachable blocks. Removing dead branches first resolved that
precondition. Failed/preliminary outputs are preserved separately.

A 50-pair sweep with the preserved canonical CI executable also fails to close
the gap, despite these graphs having only 1–16 logical tiles per node:

| Workers | Inference persistent ms / ratio | Backward persistent ms / ratio | SGD persistent ms / ratio |
|---:|---:|---:|---:|
| 1 | 0.09433 / 1.810 | 0.27334 / 4.197 | 0.32322 / 4.808 |
| 4 | 0.07043 / 1.352 | 0.11261 / 1.725 | 0.13058 / 1.942 |
| 16 | 0.07112 / 1.339 | 0.09831 / 1.496 | 0.09868 / 1.461 |
| 64 | 0.07419 / 1.292 | 0.09658 / 1.482 | 0.09623 / 1.444 |

All **760 comparisons** across the ownership/control, compiler and CI pool
trials pass. CI and forward state are exact; maximum training error is 4.66e-9.
Every persistent step has one compute dispatch and matched selected arithmetic.
All runs use separate bounded cgroups, serialize GPU work and keep builds and
analysis outside measurements. No runtime change from these trials is retained.

Finally, a separate timestamp capture disables worker histograms while retaining
GPU timing, so the reset image has its normal size. Twenty pairs per workload
pass. Median pass intervals for `reset whole-graph queue` are **2.432 µs** for
each CI case, **3.920 µs** for SmolVLA forward and **5.136 µs** for training.
These include the transfer pass boundary, not just DMA time. They are too small
to explain the gap, and no more complex reset protocol is added. The following
probes compare intra-workgroup scheduling and cross-workgroup coordination;
larger-model performance and all shared acceptance gates remain in scope.

Raw records are in `target/{sm-placement,shader-isa,inline-probe,spirv-opt-probe,
uniform-descriptor-probe,ci-worker-sweep,queue-reset-timing}/`. The combined
manifest, hashes and placement checks are in
`target/megakernel-results/placement-compiler-followups/summary.json`.

## Coordination follow-ups

A temporary specialization for an explicitly selected **one worker** walks
nodes in recording order, caches each node's parameters and executes its tiles
with workgroup storage/shared barriers. Every dependency points from an earlier
recorded node to a later one, so this order respects RAW/WAR/WAW dependencies.
All payload accesses are confined to one physical workgroup; the existing
canonical descriptors remain intact. Selected handler bodies, tile packing and
arithmetic are unchanged. Actual tile completion still updates diagnostics.

| One-worker workload | Original persistent ms | Specialized persistent ms | Specialized / ordinary paired ratio |
|---|---:|---:|---:|
| CI inference | 0.09433 | 0.06661 | 1.300 |
| CI forward/backward | 0.27334 | 0.15581 | 2.382 |
| CI forward/backward/SGD | 0.32322 | 0.17628 | 2.635 |
| SmolVLA forward | 263.379 | 234.231 | 60.787 |
| SmolVLA forward/backward/SGD | 1561.585 | 983.762 | 81.875 |

CI uses 50 pairs per case and compares with the earlier one-worker sweep.
SmolVLA uses a fresh original control followed by the specialization, each with
20 pairs per phase. All runs have at least ten warmups and one second of warmup.
These are separate runs, not directly interleaved original/specialized pairs;
each reported ratio compares its persistent candidate with its own interleaved
ordinary control. Ordinary forward latency also moves between the model runs.
Training improves substantially within the one-worker configuration, but
remains far slower than the qualified 256-worker path. Several coordination
operations change together, so this does not assign a precise cost to fences
alone. The specialization is reverted.

Two 256-worker probes keep node parameters while claiming further tiles from
the same node. They synchronize scratch between claims and defer payload
publication and completion accounting until the worker finishes that node's
claims. One applies this to all nodes; the other only repeats when the node has
more tiles than the pool can claim at once. Both retain cross-workgroup fences
at publication/acquisition and the original selected arithmetic.

| Probe | CI inference / backward / SGD paired ratios | SmolVLA forward persistent ms / ratio | SmolVLA training persistent ms / ratio |
|---|---|---:|---:|
| Reuse across all nodes | 1.377 / 1.575 / 1.638 | 6.281 / 1.586 | 21.042 / 1.397 |
| Reuse only on larger grids | 1.434 / 1.532 / 1.525 | 6.527 / 1.515 | 21.167 / 1.391 |

Each probe runs 20 pairs for all five cases. Neither demonstrates a useful
full-step gain, and both are reverted. Across these probes and the one-worker
controls, **430 normal comparisons** pass, plus three one-worker diagnostic
comparisons checking task counts. CI and model forward are exact; maximum
training error is 4.66e-9. Every candidate uses one dispatch and the same
selected implementation keys as its ordinary control. The initial reuse
attempt was rejected before execution because WGSL does not permit taking a
vector component's address; the completed probe broadcasts the whole vector.
An initial build omitted the model feature and was rerun with it enabled.

All jobs run in separate verified bounded cgroups, GPU jobs are serialized,
and builds/analysis do not overlap measurements. Patches, binaries, raw samples,
logs and hashes are local under `target/{single-worker-probe,node-cache-probe}/`
and `target/megakernel-results/coordination-followups/summary.json`. The runtime
and compiler pins are restored and release examples rebuilt. These diagnostic
experiments do not change the primary qualification or establish parity. The
following investigation tests ready-bitmap polling and contention, with local
publication proofs and the same completed-step comparisons.

## Ready-queue and polling results

An append-only queue reserves one slot per graph node. It preserves the existing
payload fences, marks a slot consumed after its final tile claim, and advances
a monotonic search head. Consumers scan past unpublished reservations so they
can execute other ready nodes. Slots are not recycled within a dispatch.
Completion checks retain exact node/tile counts and additionally require a
permutation of all nodes in consumed slots. The proof sketch is preserved with
the prototype; it does not replace full acceptance qualification.

Another probe keeps the bitmap and lets idle leaders wait for a change in the
remaining-node count before rescanning. A new release fence after publishing
successors and before decrementing that count pairs with an acquire before
the next scan. Without that ordering, a worker could cache the new count while
missing published bits. Sibling lanes wait at the uniform broadcast. Successful
claims can continue consuming already-ready tiles within the same completion
epoch. No new Naga interface is involved.

Two additional probes raise the physical worker width to 512 lanes, or make
the active-lane mask constant only when the original logical tile fills the
physical worker. Each preserves the selected arithmetic and reports its actual
worker geometry. Every probe runs 20 pairs in all five cases, with ten warmups
and at least one second of warmup.

| Probe | CI inference / backward / SGD paired ratios | SmolVLA forward ms / paired ratio | SmolVLA training ms / paired ratio |
|---|---|---:|---:|
| Ready-node queue | 1.385 / 1.564 / 1.569 | 6.298 / 1.631 | 21.364 / 1.411 |
| 512-lane workers | 1.486 / 1.725 / 1.810 | 8.598 / 1.951 | 33.801 / 2.232 |
| Constant full-tile active mask | 1.335 / 1.563 / 1.510 | 6.581 / 1.570 | 20.812 / 1.376 |
| Completion-gated leader polling | 1.349 / 1.631 / 1.692 | 6.613 / 1.529 | 21.064 / 1.384 |

All **400 comparisons** pass with one dispatch and matching implementation
keys. CI and forward are exact; maximum training error is 4.66e-9. None shows
a useful full-step gain, so all four runtime changes are reverted. Ordinary
control latency varies across runs; these exploratory ratios are not substitutes
for the primary 50-pair results.

Fresh Perf SDK captures compare graph-only forward replay, collecting six
sequential groups for the queue and three for completion-gated polling. Every
group has 146 post-warmup samples, successful target/monitor exits and no SDK
warnings. Device identity and complete state agree between modes. Rate times
replay duration gives the following activity estimates relative to ordinary:

| Implementation | Persistent replay ms | Warp instructions | Integer instructions | Global-load sectors | Shared wavefronts | Barrier share of active warp cycles |
|---|---:|---:|---:|---:|---:|---:|
| Qualified bitmap, earlier capture | 6.103 | 1.924× | 2.552× | 1.117× | 1.193× | 62.28% |
| Ready-node queue | 6.128 | 2.153× | 2.528× | 1.158× | 1.358× | 54.58% |
| Completion-gated leader polling | 6.151 | 1.603× | 2.544× | 1.140× | 1.097× | 65.60% |

Ordinary replay medians are 3.772, 3.765 and 3.774 ms respectively. Reducing
polling removes substantial warp instructions and shared traffic without
reducing replay latency; the measured integer activity remains almost unchanged.
The barrier percentages are not elapsed-time fractions. These results shift the
next experiment toward matrix addressing and the resource cost of combining
handlers, rather than claiming that idle instruction removal recovers the gap.

One initial queue capture selected RADV for the benchmark while NvPerf monitored
the RTX 5080. Both processes completed and numerical state matched, but NVIDIA
shader counters were zero. That capture is explicitly **invalid and excluded**.
The benchmark now emits adapter metadata before resident capture and includes
it in the JSON result. The helper requires the NVIDIA driver, one matching
NvPerf device, and nonzero shader activity when those metrics are collected.
Receipts include adapter identity, `counter_capture_valid` and an error on
failure. Matching real SDK logs pass; missing, non-NVIDIA, mismatched and
ambiguous adapter fixtures are rejected. Corrected captures pass these checks.
The earlier primary capture's NVIDIA ICD, target log and monitor identity were
also audited successfully; its measurements remain valid.

The capture guard is retained. Release examples rebuild, strict benchmark
Clippy and formatting pass. The runtime and dependency pins match the qualified
implementation. Raw samples, binaries, proofs, patches and SDK records are in
`target/{ready-queue-probe,wide-worker-probe,active-lane-probe,completion-poll-probe}/`;
the combined manifest is `target/megakernel-results/queue-followups/summary.json`.
All workloads used verified bounded cgroups, serialized GPU jobs and local
storage. Builds and analysis did not overlap timing. Parity remains unmet.

## Matrix resource isolation and claim priority

A temporary fixture repeats the selected `FusedMatMulAdd[50x720x2048]`
SmallTile operation 32 times, with its original `[23, 2, 1]` logical grid.
Every operation writes the same output, creating a strict WAW chain with
1,472 logical tasks. Both modes use identical operands, model-buffer sizes and
selected arithmetic; each comparison checks the full 36,000-element output.
The operands remain hot across repeated operations. This fixture diagnoses
matrix execution and does not replace the complete-model acceptance gate.

Three persistent variants compile only the active handler, or also compile
the original SmolVLA forward/training handlers without scheduling additional
nodes. The extra handlers affect code, shared scratch and parameter capacity.
Separate pipeline captures report the following resources:

| Shader | Registers | Shared bytes |
|---|---:|---:|
| Ordinary matrix | 48 | 8,448 |
| Isolated persistent matrix | 48 | 9,584 |
| With unused forward handlers | 112 | 29,200 |
| With unused training handlers | 105 | 29,232 |

Twenty paired completed-step samples per case give these persistent medians.
Ordinary medians range from 1.673 to 1.729 ms across the same runs.

| Workers | Isolated ms | With forward handlers ms | With training handlers ms |
|---|---:|---:|---:|
| 32 | 4.125 | 3.725 | 3.712 |
| 64 | 2.109 | 1.942 | 1.944 |
| 84 | 2.123 | 1.934 | 1.936 |
| 128 | 2.772 | 2.946 | 2.951 |
| 256 | 4.179 | 2.948 | 2.931 |

Adding unused handlers increases resource use yet improves the 256-worker
case. The best sampled pools are 64–84 workers; padded variants still take
about 1.15 times ordinary latency. Thus register count alone does not explain
the slowdown. The results are consistent with worker residency and contention
affecting this narrow workload; they do not isolate either as the sole cause.
The driver's implausible Local Memory Size statistic remains excluded.

Reading existing tile counters after separate diagnostic steps requires no
new shader instrumentation. At 256 workers, unsuccessful claims total 6,701
for the isolated shader and 3,885/3,884 for the padded shaders. At 64 workers,
all three total 576. Each count is one final-step diagnostic sample.

A temporary priority window obtains each worker's rank from an atomic start
counter. During the first four unsuccessful scans of a completion epoch,
workers whose rank exceeds a node's claim count skip that node. All workers
become eligible after those bounded scans; wide nodes remain available
throughout. The hint does not assume workgroup launch order or simultaneous
residency. Existing publication fences, failed-claim helping and ready-bit
clearing remain intact, and completion verifies every requested worker started.

The window reduces the 256-worker diagnostic failed-claim totals to
3,307/1,622/1,653. Normal isolated-matrix latency falls to 3.657 ms, but the
padded cases remain 2.954/2.935 ms. Registers are 48/116/108 respectively,
with unchanged shared storage. Full-step comparisons against fresh controls
show no useful overall gain:

| Case | Existing scheduler ms | Priority window ms | Existing / priority paired ratios |
|---|---:|---:|---:|
| CI inference | 0.07071 | 0.07469 | 1.350 / 1.344 |
| CI forward/backward | 0.10293 | 0.10341 | 1.545 / 1.597 |
| CI forward/backward/SGD | 0.10246 | 0.10989 | 1.507 / 1.574 |
| SmolVLA forward | 6.408 | 6.281 | 1.513 / 1.621 |
| SmolVLA forward/backward/SGD | 20.953 | 21.644 | 1.391 / 1.432 |

Each ratio compares persistent execution with its own paired ordinary control;
ordinary forward latency varies between runs. Reducing failed atomic claims
alone does not recover the performance gap. The next controlled comparison
holds the worker pool fixed while isolating descriptor/address calculations.

All **560 normal comparisons and twelve diagnostics** pass with one persistent
dispatch, identical selected arithmetic and no fallback. Matrix, CI and model
forward outputs are exact; maximum model-training error is 4.66e-9. The
temporary handler-padding API, claim dump and priority scheduler are reverted.
Runtime source matches `ed81459`, and the ordinary release benchmark examples
are rebuilt. No new Naga or Blade change is retained. All jobs use verified
bounded cgroups, serialized GPU work and local `/x` storage; builds and
analysis do not overlap measurements. Raw samples, source patches, binaries
and the priority proof remain in `target/{matrix-handler-probe,claim-priority-probe}/`.
The checked manifest and hashes are in
`target/megakernel-results/matrix-followups/summary.json`. Parity remains unmet.

## Matrix constants and inner-loop expansion

Further controls keep the same 32-node matrix chain, selected arithmetic,
operands and complete-output checks. A descriptor probe substitutes a constant
descriptor index and element offset only when every node using that handler
agrees. A dimensions probe substitutes the invariant integer uniform words while
leaving descriptor lookup dynamic. A combined probe also substitutes invariant
grid coordinates. Constants replace helper-argument uses directly, avoiding a
dependence on driver propagation through calls. Uniform-specialized diagnostics
assert that reused recordings do not change the embedded plain values.
Generated shader dumps verify every intended substitution.

Twenty normal pairs per case give these completed-step persistent medians at
64 workers. Ordinary medians range from 1.669 to 1.703 ms in these runs.

| Probe | Isolated ms | With unused forward handlers ms | With unused training handlers ms |
|---|---:|---:|---:|
| Fresh control | 2.104 | 1.936 | 1.940 |
| Constant descriptors and offsets | 2.049 | 1.937 | 1.943 |
| Constant dimensions | 1.843 | 1.929 | 1.929 |
| Constants plus grid coordinates | 1.998 | 1.928 | 1.917 |
| Explicit inner-loop expansion | 2.162 | 1.939 | 1.946 |

The isolated dimensions result improves, but the combined shaders retain their
gap. At 256 workers, the combined variants stay between 2.924 and 2.955 ms;
the isolated variants range from 4.084 to 4.342 ms. A fresh 46-worker control,
matching each node's logical tile count, takes 2.107/1.938/1.946 ms and does not
improve on the 64-worker control.

Four separate SDK comparisons collect three sequential single-pass groups per
mode at 64 workers. Every group has 146 samples, successful target/monitor exits,
matching NVIDIA adapter identity and no SDK warning. All eight replays preserve
exact output and matching arithmetic (state SHA-256
`cbf2e7ebdfb60979fce668c7879a168a1923982a6d37441a12c5399d0a1752fe`).
Repeated instruction rates deviate by at most 0.92% from their group mean;
clock rates deviate by at most 0.28%. Rate times replay duration gives:

| Capture | Ordinary / persistent replay ms | Warp activity ratio | Integer activity ratio | Global-load ratio | Shared-wavefront ratio | Persistent barrier share |
|---|---:|---:|---:|---:|---:|---:|
| Isolated control | 1.728 / 2.136 | 1.312 | 1.863 | 1.050 | 1.028 | 27.49% |
| Forward-handler control | 1.704 / 1.969 | 1.202 | 3.380 | 1.054 | 1.029 | 27.02% |
| Isolated, constant dimensions | 1.710 / 1.873 | 1.290 | 1.252 | 1.059 | 1.033 | 27.71% |
| Forward handlers, all constants | 1.710 / 1.954 | 1.148 | 2.856 | 1.052 | 1.026 | 27.22% |

The integer column uses the SDK's FMA-heavy integer-thread metric; it does not
enumerate every integer operation. Ratios use each capture's ordinary control.
These are activity estimates, not simultaneous exact instruction counts.
Ordinary barrier shares are about 2.6%; all barrier percentages describe active
warp cycles, not elapsed-time fractions.

Constant dimensions reduce the isolated shader's estimated integer activity
by about one-third and shorten replay. In the combined shader, fixing all three
sources of dynamic addresses removes about 16% of that activity but improves
replay by less than 1%. Thus descriptor indirection and these address calculations
alone do not recover the combined shader's gap. Compiling unused handlers still
changes instruction behavior even when the executed matrix work is fixed.

The inner-loop probe separately expands the selected scalar matrix template's
32 arithmetic iterations using the existing unroll generator. It preserves
every accumulator's operation order, all stage loads, scratch, stores and
barriers. Dumps contain 128 explicit accumulator updates and no `kk` loop in
the active handler. The isolated executable remains 17,408 bytes; forward and
training combinations grow to 121,216 and 303,104 bytes without a timing gain.
All probes retain 48/112/105 registers for isolated/forward/training variants
and the same shared-memory capacities. No missing-unroll diagnosis is established.

All **660 normal comparisons and twelve diagnostics** pass with exact output,
one persistent dispatch, identical selected arithmetic and no fallback. These
are matrix diagnostics; no whole-model speedup is claimed. All runtime and
temporary benchmark API changes are reverted, production release examples are
rebuilt and formatting passes. Runtime source and dependency pins remain at the
qualified implementation. Every job used a verified bounded cgroup, GPU jobs
were serialized, and builds/analysis stayed outside measurement intervals.
Source patches, proofs, binaries, raw samples and SDK receipts remain in
`target/{matrix-address-probe,matrix-unroll-probe}/`; their checked manifest is
`target/megakernel-results/matrix-address/summary.json`.

The following experiment varies serial versus independent-node dependencies
while holding shader code and resource layouts fixed. Full-model baseline
parity remains unmet.

## Independent nodes and terminal publication

A new matrix fixture gives each of the 32 selected matrix nodes its own output
allocation. The nodes share the same three input buffers and have no native
data hazards; lifetime aliasing is disabled. Every node retains the original
46 logical tiles, parameters and arithmetic. Every paired sample reads all
32 complete outputs outside timing, verifies that they agree exactly and are
nonzero, and compares them with ordinary execution.

The parallel mode uses the empty dependency graph. A serial control adds an
artificial edge from each node to the next in the persistent path only. Both
modes have identical physical pipeline keys, descriptor layouts, buffer sizes,
parameter words, metadata sizes and queue sizes. Separate sessions own distinct
allocations. The edge buffers both have 32 words; unused entries are unreachable.
Diagnostic receipts verify that only dependency metadata and initial readiness
differ. Neither the scheduler shader nor its publication protocol changes.

Twenty completed-step pairs per case give these medians:

| Workers | Compiled handlers | Ordinary independent ms, parallel / chain runs | Persistent parallel ms | Persistent artificial chain ms |
|---|---|---:|---:|---:|
| 64 | Matrix only | 0.733 / 0.710 | 1.565 | 2.135 |
| 64 | With unused forward handlers | 0.708 / 0.707 | 1.405 | 1.963 |
| 64 | With unused training handlers | 0.708 / 0.739 | 1.410 | 1.990 |
| 256 | Matrix only | 0.733 / 0.728 | 0.788 | 4.144 |
| 256 | With unused forward handlers | 0.709 / 0.726 | 0.798 | 2.970 |
| 256 | With unused training handlers | 0.708 / 0.732 | 0.799 | 2.972 |

Within each handler variant, driver executable statistics also match between
the dependency modes: 48/112/105 registers and 9,584/29,200/29,232 shared bytes
for isolated/forward/training variants. The implausible driver Local Memory Size
statistic remains excluded. The independent combined shader comes much closer
to ordinary execution despite retaining its resource footprint and dynamic
addresses. Adding a chain exposes a large scheduling cost in this narrow
workload. This does not establish how much overlap a real model permits, nor
does it prove that finer dependencies will recover the full-model gap.

The ordinary control executes the same independent plan in both modes. Thus
the artificial-chain ratios measure an imposed delay, not ordinary execution
of a naturally dependent graph. These distinct-output results also are not
directly comparable with earlier fixtures that repeatedly wrote one output.

A separate prototype removes publication fences and the first post-body
workgroup rendezvous only for nodes with no successors. All input acquisition,
nonterminal publication and the final workgroup barrier remain. Conservative
RAW/WAR/WAW reachability means terminal payloads have no later conflicting
access in the dispatch. Their external visibility still depends on real dispatch
completion **and existing API memory dependencies**, not the scheduler counter
or host fence alone. The proof and source patch are preserved with the probe.

Twenty paired comparisons per full-step case, followed by fresh controls, give:

| Case | Existing scheduler ms | Terminal elision ms | Existing / elision paired ratios |
|---|---:|---:|---:|
| CI inference | 0.07007 | 0.07082 | 1.359 / 1.358 |
| CI forward/backward | 0.10347 | 0.10166 | 1.547 / 1.544 |
| CI forward/backward/SGD | 0.10132 | 0.10162 | 1.502 / 1.510 |
| SmolVLA forward | 6.549 | 6.557 | 1.570 / 1.519 |
| SmolVLA forward/backward/SGD | 21.284 | 20.888 | 1.406 / 1.383 |

Ratios compare each persistent sample with its own ordinary control. Forward
latency is essentially unchanged. The small training difference does not
establish a benefit given substantial variation, including long outliers in
both modes. The terminal candidate is reverted without claiming full memory
protocol qualification or a speedup.

All **440 normal comparisons and six diagnostics** pass with one persistent
dispatch, identical selected arithmetic and no fallback. Matrix, CI and model
forward outputs are exact; maximum model-training error is 4.66e-9. All runtime
and temporary benchmark APIs are reverted, production release examples rebuilt,
and formatting checked. Runtime source still matches `ed81459`; Naga and Blade
pins are unchanged. No broad GPU suite is repeated for these discarded probes.

Every build, test and analysis job uses a verified separate bounded cgroup.
GPU jobs are serialized, builds/analysis stay outside timing intervals, and all
artifacts use local `/x` storage. The recovered terminal-probe logs confirm the
20 GiB memory caps and GPU lock; neither job remained running after recovery.
Raw results, patches, proofs and binaries are in
`target/{matrix-dependency-probe,terminal-publication-probe}/`; the checked
manifest and artifact hashes are in
`target/megakernel-results/matrix-dependencies/summary.json`.

The following controls measure intermediate dependency widths before attempting
finer dependencies in a real graph. Such a change needs explicit proofs for
access ranges, lifetime reuse, original tile coordinates and completion counts.

## Dependency width and handler isolation

The distinct-output fixture adds an edge `n-W -> n` for each node `n >= W`.
Widths 1/2/4/8/16 create that many independent chains; width 32 leaves all
nodes independent. The ordinary plan remains independent at every width.
All cases retain the same shader keys, resource layouts, parameter words and
buffer sizes within each compiled-handler variant. Eighteen metadata receipts
check the exact edges, ready bits, predecessor counters and unchanged fields.

Twenty normal pairs per case give these persistent medians at 256 workers:

| Ready chains | Matrix only ms | With unused forward handlers ms | With unused training handlers ms |
|---|---:|---:|---:|
| 1 | 4.141 | 2.972 | 2.979 |
| 2 | 2.168 | 1.518 | 1.534 |
| 4 | 1.230 | 0.983 | 1.019 |
| 8 | 0.911 | 0.801 | 0.840 |
| 16 | 0.767 | 0.802 | 0.829 |
| 32 | 0.796 | 0.801 | 0.798 |

Ordinary independent controls range from 0.708 to 0.760 ms. At 64 workers,
the combined variants plateau near 1.4 ms even with more independent chains.
The 256-worker combined shader needs about eight chains to approach its
independent-node timing. This is an imposed scheduling constraint, not a model
speedup. Splitting a real matrix by rows may enable earlier consumers without
increasing the total ready tile count enough to reproduce this benefit.

A separate scheduler probe replaces fetch-add claims with a load and retrying
compare-exchange, advancing the tile counter only while work remains. It retains
last-claim/helper ready-bit clearing, partial-batch clamping and every publication
fence. The forward-handler matrix chain slows from 2.972 to 3.106 ms. Full model
forward/training medians become 7.841/24.871 ms, substantially worse than the
qualified scheduler. All 120 matrix and 100 full-step pairs pass, but the probe
is reverted. Final counter overshoot is not a count of failed CAS attempts.

Eighteen further diagnostics compile each unique forward implementation family
into the independent matrix shader without scheduling its work:

| Compiled alongside the matrix | Registers | Shared bytes |
|---|---:|---:|
| Nothing | 48 | 9,584 |
| Horizontal scalar matmul | 128 | 16,880 |
| Cooperative matmul with prologue | 108 | 16,768 |
| Cooperative attention | 80 | 12,688 |
| GEMV | 80 | 15,728 |
| Full forward handler set | 112 | 29,200 |

The other individual forward families retain 48 registers. Composition remains
nonadditive: the horizontal-only combination has more registers than the full
shader. Instrumented timings from these diagnostics are not speedup evidence.
Inspecting that horizontal family reveals the geometry bug fixed below.

All **940 normal comparisons and 54 diagnostics** across the width, claim and
family probes pass with unchanged selected arithmetic, one persistent dispatch
and no fallback. The scheduler and temporary benchmark APIs are reverted.
One resource-only capture omitted metadata dumps; a separate complete capture
provides all 18 audited width receipts. Local proofs, patches, binaries and raw
results are indexed in `target/megakernel-results/dependency-width/summary.json`.

## Correct horizontal matmul geometry

Horizontal fusion admits `Kernel::SmallTile` and retains its 32×32 output-tile
grid. Its scalar runtime key previously omitted tile geometry, and the generator
always emitted the default 64×64 body. The small grid therefore overscheduled
the larger body and included its larger register tile in the combined shader.

Commit `12f7c5a` carries scalar geometry in the horizontal implementation key
and passes it, with the session's existing staging knobs, to the common
generator. Ordinary and persistent execution now both honor the selected
32×32 or 64×64 tile. The plan's logical grids, bindings and output buffers remain
the same. Queue scheduling, publication, descriptors and worker counts do not
change. In the default model, K-stage depth and column layout already matched
the previous defaults; the corrected body visits K in the same order.

The expanded GPU regression checks all three packed outputs against a CPU
reference with non-multiple M/N/K dimensions and repeated input updates. Both
geometries remain alive on one context to exercise separate cache entries, in
ordinary and whole-graph modes. The fixture asserts actual selection; an initial
too-small large-tile control was automatically reselected and is preserved as a
failed fixture attempt. Module tests check generated tile-column spacing as
well as validation, since overscheduling can still produce correct output.

Qualification passes **665 library tests (eight ignored), 168 strict whole-graph
oracle tests and nine focused GPU tests**: two horizontal regressions and seven
whole-graph geometry, scratch, replay, lifetime and optimizer tests. Formatting
and strict all-feature/all-target Clippy pass. Naga and Blade pins are unchanged;
their previously recorded checks are not rerun for this shared generator fix.

Fresh production binaries from clean tracked source `12f7c5a` run 50 alternating
AB/BA pairs per workload, with ten warmups and at least one second of warmup.
The existing completed-step boundaries, state restoration and full-state checks
remain unchanged. Every case reports one persistent dispatch, matching corrected
arithmetic and no fallback. CI and forward state are exact; maximum training
error is 4.66e-9.

| Workload, 256 workers | Ordinary ms | Persistent ms | Paired ratio | 95% interval |
|---|---:|---:|---:|---:|
| CI inference | 0.05527 | 0.07423 | 1.358 | 1.230–1.430 |
| CI forward/backward | 0.06613 | 0.10271 | 1.551 | 1.541–1.569 |
| CI forward/backward/SGD | 0.06743 | 0.10201 | 1.522 | 1.510–1.539 |
| SmolVLA forward | 4.235 | 6.061 | 1.427 | 1.413–1.439 |
| SmolVLA forward/backward/SGD | 14.947 | 20.716 | 1.380 | 1.360–1.401 |

Intervals bootstrap paired-ratio medians 10,000 times and describe within-run
uncertainty, not between-run variation. A separate 20-pair-per-case control uses
the preserved pre-fix binaries: persistent forward/training medians are
6.396/20.779 ms. The new forward median is about 5.2% lower; training is effectively
unchanged and CI shows no established benefit. Old-binary controls are historical
timing comparisons, not matching-arithmetic pairs against the corrected binary.
Their recorded worktree revision is not their binary build identity. An earlier
40-pair model exploration is also preserved. Baseline parity remains unmet.

Separate driver diagnostics report 116/117 registers for corrected forward/
training shaders, versus 112/112 before the fix. Forward shared storage falls
from 29,200 to 20,880 bytes; training remains 29,232 bytes. Executable sizes fall
from 114,560/272,000 to 101,248/259,968 bytes. The driver's implausible Local
Memory Size field remains excluded.

Fresh NVIDIA SDK captures use the previous six sequential single-pass metric
groups per mode, with 146 samples in every group, no SDK warnings, matching
NVIDIA adapter identity and successful target/monitor exits. Both corrected
paths retain the exact previous forward-state SHA-256
`4e5d66e6ce5e95142b848651d316b4df64855a9d85669499cabc3a81089be629`.
Repeated instruction rates deviate by at most 0.50% from their group mean;
clock rates deviate by at most 0.23%. Rate times replay duration gives:

| Metric | Corrected ordinary | Corrected persistent |
|---|---:|---:|
| Graph replay ms | 3.579 | 5.582 |
| Warp instructions, relative to ordinary | 1.000 | 1.934 |
| FMA-heavy integer-thread activity, relative | 1.000 | 2.625 |
| Global-load sectors, relative | 1.000 | 1.097 |
| Shared-memory wavefronts, relative | 1.000 | 1.155 |
| Barrier waiting, share of active warp cycles | 3.70% | 61.57% |
| Local-load/store sectors | 0 / 0 | 0 / 0 |

The earlier forward capture estimated about 39,854 local-load and 678,795
local-store sectors for persistent execution. The corrected capture reports
zero for both metrics. This is evidence for the measured forward replay, not
proof that every supported graph is spill-free. The integer metric does not
count all integer operations; rate-times-duration values are activity estimates,
and barrier share is not an elapsed-time fraction. Substantial extra instructions
and coordination costs remain after local traffic disappears.

All jobs use verified separate bounded cgroups; GPU jobs are serialized, and
builds/analysis/profiling do not overlap primary timing. Artifacts stay on `/x`.
Tests, source patch, pre-fix controls, production binaries and SDK receipts are
indexed in `target/megakernel-results/horizontal-geometry/summary.json`. The next
investigation targets the remaining cost on this corrected runtime, including
training counters and the cooperative-prologue handler's resource footprint.

## Scalar matmul output stores (October 26, 2026)

The corrected runtime's first backward SDK capture completes successfully on
both backends, but the wrapper's cross-backend bitwise check fails. That failure
is preserved. A separate audit compares all 246,883,105 state floats: loss and
parameters are exact; 54,475,347 gradient values differ, with maximum error
4.6566128730773926e-9. The largest fraction of the existing benchmark tolerance
is 0.000460. Each backend remains bitwise stable across its entire capture.
The SDK script's exact-state gate is not weakened.

This capture measures **forward plus backward with the optimizer disabled**,
not the complete SGD step. Ordinary/persistent replay takes 10.611/17.175 ms.
Rate times duration estimates about 12.08 million local-load and 206.12 million
local-store sectors for persistent execution, versus zero ordinary. The earlier
forward-only zero-traffic result therefore did not extend to backward work.

A cooperative-matmul probe moves output-address calculations after accumulation.
It saves one forward and two training registers, but 40 normal comparisons show
no useful timing gain. The probe is reverted; its source, binary and two resource
diagnostics remain in `target/coop-output-address-probe/`.

Commit `5dbabed` changes the common scalar matmul generator. It previously built
a nested array from completed scalar accumulators and dynamically indexed that
array in output loops. It now emits a bounds-checked store for each accumulator,
in the same row/column order. Both ordinary and persistent execution use this
change. K staging and accumulation, selected geometry, column mapping, split
offsets, fused addends and epilogues remain unchanged. The WGSL store fragment
stays in `matmul.wgsl`. Queue scheduling, fences, scratch isolation and submission
are unchanged; the shared convolution generator retains its existing output path.

Qualification passes **665 library tests (eight ignored), 168 strict whole-graph
oracle cases, 50 matrix regressions and seven whole-graph tests**, plus formatting
and strict all-feature/all-target Clippy. Matrix regressions cover transposes,
partial dimensions, packed outputs, rectangular candidates and split reductions.
Naga and Blade pins are unchanged. Clean tracked source `5dbabed` produces the
final binaries; the unrelated untracked `configs/` directory is disclosed.

All 250 primary comparisons pass with 256 workers, 50 alternating AB/BA pairs
per workload, ten warmups and at least one second of warmup. Every persistent
case has one dispatch, identical selected arithmetic and no fallback. CI and
forward are exact; maximum training error remains 4.66e-9.

| Workload | Ordinary ms | Persistent ms | Paired ratio | 95% interval |
|---|---:|---:|---:|---:|
| CI inference | 0.05728 | 0.07564 | 1.319 | 1.310–1.327 |
| CI forward/backward | 0.06613 | 0.10239 | 1.553 | 1.543–1.567 |
| CI forward/backward/SGD | 0.06714 | 0.10096 | 1.508 | 1.484–1.524 |
| SmolVLA forward | 4.207 | 6.056 | 1.436 | 1.427–1.482 |
| SmolVLA forward/backward/SGD | 14.945 | 19.403 | 1.310 | 1.276–1.471 |

Intervals bootstrap paired-ratio medians 10,000 times and describe within-run
uncertainty. Long outliers on both paths remain in every calculation and widen
the training interval; no samples are excluded. A subsequent 50-pair-per-phase
control uses the preserved geometry-corrected binary: forward/training medians
are 6.097/20.725 ms persistent and 4.217/14.940 ms ordinary. The observed training
median falls about **6.4%**, with forward effectively unchanged. The normalized
training intervals overlap, so the median improvement is not a guarantee across
runs. Independently bootstrapping the two runs' persistent latency medians gives
a new/old ratio of 0.936 (95% interval 0.924–0.952); ordinary is 1.000
(0.978–1.021). These intervals also exclude between-run drift. The source
revision observed by an old binary is not its build identity.
Earlier 20-pair-per-phase candidate/control runs are also preserved: there are
470 normal comparisons and four resource diagnostics across this investigation.
**Baseline parity remains unmet.**

The new backward SDK capture uses six sequential single-pass groups per mode,
146 samples each, matching NVIDIA adapters, successful process exits and no
warnings. Both modes preserve their exact respective pre-change state hashes;
the known cross-backend gradient difference remains. Repeated instruction rates
deviate by at most 0.22% from their group means and clocks by at most 0.25%.

| Backward graph metric | Ordinary | Persistent |
|---|---:|---:|
| Replay ms | 10.526 | 15.926 |
| Warp instructions, relative to ordinary | 1.000 | 1.833 |
| FMA-heavy integer-thread activity, relative | 1.000 | 2.923 |
| Global-load sectors, relative | 1.000 | 1.364 |
| Shared-memory wavefronts, relative | 1.000 | 1.186 |
| Barrier waiting, share of active warp cycles | 5.74% | 57.40% |
| Local-load/store sectors | 0 / 0 | 0 / 0 |

Removing the output arrays eliminates the measured local traffic and shortens
persistent backward replay by about 7.3%. This supports array materialization or
its resulting register pressure as a cause; the aggregate counters do not map
traffic to individual instructions. Counts estimate rate times duration, the
integer metric covers one pipe class, and barrier share is not elapsed-time share.
The capture excludes optimizer work and is not a full-step timing substitute.
It uses the preserved exploratory binary; the only subsequent shader change
before the final production build corrects a source comment.

Separate full-step resource diagnostics report unchanged forward resources:
116 registers, 20,880 shared bytes and 101,248 executable bytes. Training falls
from 117 to 114 registers and from 259,968 to 249,344 executable bytes, with
29,232 shared bytes unchanged. The driver's implausible Local Memory Size field
is excluded. Diagnostic instrumentation timings are not performance results.

All jobs use verified separate cgroups with 16 GiB MemoryHigh, 20 GiB MemoryMax
and 1 GiB MemorySwapMax. GPU jobs are serialized; builds, copies and heavy
analysis do not overlap primary timing. The agent remains outside those scopes
and all artifacts stay on `/x`. Captures, patches, binaries, test logs, raw
samples, uncertainty calculations and artifact hashes are indexed by
`target/megakernel-results/scalar-output-stores/summary.json`; the preserved
initial backward failure and numerical audit are in
`target/corrected-training-sdk/summary.json`. Remaining work is to isolate the
extra instruction and scheduling cost after local traffic disappears.

## Earlier artifact identities

All records remain on local storage at
`target/megakernel-results/whole-graph-20261025/`. The measured SmolVLA executable
has SHA-256 `fd52b1400a974c8c7a326afcc40b071310aecb61aa4cba9c5a19a64de6398637`.

```sh
cargo build --release --features models --example bench_smolvla_train --example bench_ci_latency
target/release/examples/bench_smolvla_train --whole-graph --workers 256 --warmup 10 --runs 50 --warmup-ms 1000
target/release/examples/bench_ci_latency --whole-graph --workers 256 --warmup 10 --runs 50 --warmup-ms 1000
python3 scripts/run_bounded_job.py sdk-capture --gpu -- \
  env VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
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
