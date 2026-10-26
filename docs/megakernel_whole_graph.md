# One dispatch for the entire Session graph

`PersistentOptions { whole_graph: true, device_scheduling: true, ... }` runs a
nonempty graph in one compute dispatch. A fixed physical worker pool repeatedly
claims logical workgroup tiles and schedules dependent operations on the GPU.
The step's optimizer, global/adaptive clipping and temporal accumulation are
nodes in the same dispatch. The transfer reset and compute dispatch share one
queue submission. Initial parameter/moment allocation and explicit optimizer
calls remain separate operations outside that step boundary.

The current runtime adds [bounded gradient writes](#bounded-gradient-write-dependencies)
to the qualified cooperative row distribution at `762faf2`. Removing false
ordering between independent bias and weight gradients improves CI training;
the model gap remains. The intervening scheduler, workgroup-propagation,
subgroup-staging and two-band release probes remain reverted. Performance parity
remains unmet.

For model benchmarks, use `--whole-graph --workers 256`. The
[current worker sweep](#worker-counts-and-deeper-scalar-staging) finds lower CI
backward/SGD latency with an explicit 64-worker pool; no universal replacement
count is established. For a regular Session configured from the environment:

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
reuse. Reads and unrecognized writes cover the entire allocation. Selected scalar
`MatMulAT` and `SumRows` outputs use proved byte ranges, allowing disjoint writes
without weakening physical read/write hazards. Rebinding rebuilds these ranges
from the actual buffer pieces. Transitive dependency edges are omitted without
changing reachability. Producer and consumer lanes use the existing
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

The [narrow scalar tile measurements](#narrow-scalar-tiles)
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

## Complete-step SDK replay and host attribution

The retained runtime remains `5dbabed`; these changes extend capture tooling.
`--capture-phase step` includes forward, backward and SGD, using a zero learning
rate so repeated captures preserve state. It selects the same graph and optimizer
pipeline keys and task counts as the normal nonzero-learning-rate benchmark:
ordinary execution has 527 compute dispatches, while persistent execution has
one. Four SGD nodes contribute 97,522 logical tiles. Zero-rate replay is a
diagnostic and does not replace the completed-step measurements above.

The SDK wrapper keeps exact cross-backend comparison as its default. Explicit
`--state-comparison benchmark` compares every captured float using the shared
benchmark budget `1e-5 + 1e-4 * max(abs(a), abs(b))`, records the policy and error,
and requires NumPy for chunked memory-mapped comparison. Both policies still
require bitwise-stable replay within each backend. The previously failed exact
backward capture remains preserved. Run a new complete-step capture with:

```sh
python3 scripts/run_bounded_job.py sdk-step --gpu -- \
  env VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  python3 scripts/profile_smolvla_sdk.py --sdk /x/Tools/nsight-perf-sdk-2026.3 \
  --output target/new-step-capture --phase step --workers 256 --whole-graph \
  --state-comparison benchmark
```

Separate six-group counter and eighteen-group stall captures each collect 146
samples per group and mode. All target and monitor exits succeed, adapters match,
and no SDK warnings occur. Captured state hashes match each backend's previous
backward capture exactly. Comparing all 246,883,105 floats across backends gives
the existing maximum error of 4.66e-9; 54,475,347 float representations differ.
The maximum error/budget fraction is 0.0004594. Cross-backend state is numerically
qualified, not bitwise equal.

The six-group ordinary/persistent replay medians are 11.941/17.245 ms. Both modes
have zero local-load and local-store sectors. Rate-times-duration estimates are
1.844× warp instructions, 2.659× FMA-heavy integer-thread activity, 1.297× global
load sectors and 1.216× shared-memory wavefronts. The integer metric covers that
execution pipe rather than all integer instructions. Repeated instruction rates
vary by at most 0.33% from their within-mode group mean; clocks by at most 0.25%.

The eighteen-group replay medians are 11.982/17.287 ms. Its leading issue states
are below, expressed as percentages of active warp cycles, **not elapsed-time
percentages**. The SDK defines no-instruction waiting to include instruction
selection/fetch as well as cache misses; these measurements do not establish
instruction-cache thrashing.

| Issue state | Ordinary % | Persistent % |
|---|---:|---:|
| L1TEX memory dependency | 60.049 | 19.109 |
| Workgroup barrier | 3.035 | 51.770 |
| Memory barrier | 0.000 | 1.428 |
| No instruction | 0.340 | 0.939 |
| MIO dependency | 4.072 | 3.028 |
| MIO queue throttle | 10.270 | 6.660 |
| Fixed-latency dependency | 8.518 | 8.568 |

Across those eighteen groups, repeated instruction-rate deviation is at most
0.91% and clock deviation at most 0.38%. Counters describe the combined shader;
they do not attribute barrier cycles to individual scheduler or tile barriers.

A separate 300-comparison CI host probe splits input upload, recording/submission,
completion wait and readback. For forward/backward/SGD, median recording/submission
falls from 18.475 to 7.360 microseconds, while completion wait rises from 48.055
to 92.805 microseconds. Input and readback are unchanged at 0.22 and 0.54
microseconds. The other two CI cases show the same direction. These instrumented
host intervals include driver and wakeup costs and are not pure GPU timestamps
or primary timing results. The temporary instrumentation is reverted.

Retesting immutable descriptor variants after scalar output stores gives
6.114 ms forward and 23.354 ms training, with paired ratios 1.551 and 1.567 across
twenty pairs per case. All 100 CI/model comparisons preserve selected arithmetic,
state budgets and one dispatch, but training regresses against the retained
19.403 ms primary measurement. Training resources rise to 142 registers and
337,920 bytes of code, from 114 and 249,344. The variant is reverted.

Capture-tool qualification passes three execution admissions, seven rejection
fixtures and six state-comparison fixtures, plus formatting and strict all-feature,
all-target Clippy. Runtime sources are unchanged; the existing runtime test
qualification still applies. All jobs use the verified bounded cgroups and GPU
serialization described above. Captures, tooling source patches, binaries,
host/variant probes and 236 independently verified artifact hashes are indexed by
`target/megakernel-results/step-sdk/summary.json`. The independent hash-audit log is
`target/step-sdk-artifact-audit.log`.

## Flat packing probe

A subsequent prototype packs consecutive linear tile indices across grid-axis
boundaries, allowing a partial final claim. The existing lane masks suppress
inactive groups' writes/products, scratch remains separate, and inactive groups
clamp reads to the last valid tile. The generic queue already clamps final-batch
completion. Barrier admission still applies the scope-aware checker to every
original barrier, rejecting divergent control before trying existing axis packing.
No queue, fence or submission protocol changes are involved.

Both parser checks and all seven whole-graph GPU tests pass. The existing packed
matrix fixture includes 130- and 258-tile grids with partial four-tile batches,
1/3/64/4096 workers, changed inputs, exact outputs and checked completion counts.
Twenty alternating pairs per model phase, followed by twenty pairs of the
preserved current binary, give:

| Case | Flat ordinary / persistent ms | Current ordinary / persistent ms | Flat / current paired ratios |
|---|---:|---:|---:|
| SmolVLA forward | 4.248 / 6.378 | 3.980 / 6.023 | 1.542 / 1.448 |
| SmolVLA forward/backward/SGD | 14.948 / 19.544 | 14.893 / 19.367 | 1.317 / 1.302 |

All eighty normal comparisons preserve selected arithmetic, one persistent
dispatch, no fallback and the previous state-error bound. Two separate metadata
diagnostics reveal that all sixteen eligible cooperative nodes have grid
`[2, 128, 1]`. They already pack four tiles along the 128-wide axis in the retained
shader. Flat packing changes tile grouping without increasing the number of
useful subgroups per worker; the suspected divisibility limitation is absent in
this model. The experiment does not establish a timing improvement and is fully
reverted, including its temporary metadata dump. The restored benchmark binaries
are rebuilt from the retained source.

These are exploratory comparisons with all outliers retained; diagnostic timings
are excluded, and the rejected candidate does not receive the full oracle gate.
Source patch, proof, binaries, test logs, raw samples and metadata are preserved
in `target/flat-packing-probe/`, indexed by
`target/megakernel-results/flat-packing/summary.json`. Every build, test, timing and
analysis job uses a verified separate bounded cgroup, and GPU jobs run serially.
The runtime remains `5dbabed`, with the primary qualification and parity gap above.

## Scalar staging and combined broadcast probes

Three further prototypes are evaluated on source base `f3b38b6` and fully
reverted. The retained runtime and its primary qualification remain `5dbabed`.
Every candidate preserves selected arithmetic and one persistent compute
dispatch, and all normal state comparisons satisfy the existing budget.

The first two change the common scalar Small-tile generator used by both paths.
Alternating two scratch stages lets iteration i+1 load a different buffer while
slower lanes finish reading iteration i. The staging barrier in i+1 then orders
all those reads before iteration i+2 reuses the first buffer. K arithmetic,
geometry, tails, split offsets and output stores are unchanged. Larger tiles keep
their original scratch and reuse barrier. The first version uses dynamic stage
bases; the second emits two explicit stages with constant offsets and checks the
original end-K condition before each stage.

Twenty alternating pairs per model phase produce:

| Variant | Forward ordinary / persistent ms | Training ordinary / persistent ms |
|---|---:|---:|
| Dynamic scratch offsets | 4.679 / 6.032 | 15.999 / 19.130 |
| Subsequent retained-runtime control | 3.834 / 5.712 | 14.926 / 19.487 |
| Constant scratch offsets | 4.102 / 6.625 | 15.376 / 20.222 |

The dynamic variant's paired ratios fall to 1.289 forward and 1.194 training,
partly because ordinary execution becomes slower. This is not evidence of
approaching the original baseline. The explicit-stage variant also fails to
improve persistent timing. Dynamic/static forward resources are both 112
registers and 29,328 shared bytes, with 102,400/122,880 code bytes. Training uses
114/103 registers, 29,360 shared bytes and 250,880/283,264 code bytes. Reduced
register counts alone do not establish a speedup.

The dynamic candidate's library run passes 664 tests and fails the assertion
that expected one Small-tile scratch allocation. Updating that expectation for
two stages makes the focused check pass; the original failure is preserved.
The static candidate passes all 48 code-generation tests. Each version passes
eight strict matrix oracle cases and all seven whole-graph GPU tests. Both the
runtime changes and the temporary test-expectation change are reverted.

The third prototype groups shared node metadata and parameter words into one
struct. The claiming leader publishes the node before the unchanged ticket
broadcast; all lanes then copy parameters using that ticket, followed by one
uniform struct load. Successful claims use two uniform loads instead of four.
The empty/done paths, shared-memory bound, payload fences, claims, dependencies
and completion protocol are unchanged. Two compiler checks and all seven
whole-graph GPU tests pass.

Twenty pairs per case, followed by twenty fresh retained-runtime pairs, give:

| Case | Combined broadcast persistent ms | Control persistent ms | Candidate / control paired ratios |
|---|---:|---:|---:|
| CI inference | 0.07047 | 0.07008 | 1.357 / 1.358 |
| CI forward/backward | 0.10160 | 0.10226 | 1.532 / 1.551 |
| CI forward/backward/SGD | 0.10153 | 0.10070 | 1.507 / 1.499 |
| SmolVLA forward | 6.044 | 6.009 | 1.513 / 1.549 |
| SmolVLA forward/backward/SGD | 19.472 | 19.348 | 1.279 / 1.283 |

The combined broadcast retains 116 forward registers and uses 113 training
registers, versus 114 retained. Shared storage is unchanged. Code sizes are
101,120/249,344 bytes. There is no useful completed-step gain.

A matched SDK check collects two sequential single-pass groups per mode and
variant, with 146 samples per group. Both captures include zero-rate SGD,
matching adapters, successful target/monitor exits and no warnings. Each
backend preserves its exact previous state hash; all 246,883,105 cross-backend
floats satisfy the existing tolerance, with maximum error 4.66e-9. Persistent
candidate/control replay medians are 17.232/17.262 ms, with approximately
2.791/2.797 billion warp instructions per replay. Barrier waiting is
51.92%/51.81% of active warp cycles. Ordinary replay is 11.953/11.959 ms.
Repeated instruction-rate deviation within each two-group capture is at most
0.46%, and clock deviation at most 0.0022%. These counters show little change
from reducing the WGSL broadcasts. They do not establish the driver's exact
barrier instruction counts; active-warp shares are not elapsed-time shares.

All **320 normal comparisons and six resource diagnostics** retain their raw
samples and outliers. No full oracle qualification is claimed for the rejected
candidates. Timings from resource captures and zero-rate resident replay do not
replace the normal completed-step gate. The restored benchmark binaries are
rebuilt, and all runtime source changes are absent. Builds, tests, timing,
captures and analysis use verified bounded cgroups; GPU jobs remain serialized
and every artifact stays on `/x`.

Patches, proofs, binaries, test logs, source dumps, SDK records and 226 artifact
hashes are indexed by
`target/megakernel-results/synchronization-probes/summary.json`. The original
library assertion failure remains in `target/scalar-double-buffer-probe/lib.log`.
The following investigation tests handler function boundaries and register
lifetimes, with local-traffic and completed-step checks.

## Handler function-control probes

Two temporary Blade backend probes apply SPIR-V `DontInline` controls after
normal validated generation, on Meganeura source base `bca1536`. The production
Naga and Blade pins, application shaders, arithmetic, fences, descriptors,
scratch and scheduler protocol remain unchanged. Both probes are fully reverted;
the retained runtime remains `5dbabed` and its primary qualification above applies.

The first probe changes the control words of all direct handler callees of
`graph_main`, excluding `graph_claim`. A binary audit verifies that only those
words change in fifteen modules, all validated for Vulkan 1.3. Four exact smoke
tests pass and three fail: mixed-width reductions, optimizer updates, and the
attention branch of the selected-geometry test. Maximum differences are
1.19e-7, 2.33e-10 and 1.19e-7 respectively. **These exact-test failures reject
the candidate**, regardless of their numerical magnitude. No performance run
qualifies it. Removing the hint restores all seven passes.

The second probe selects only scalar matrix handlers, including their horizontal
wrappers. A shadow emission with debug names must become byte-for-byte identical
to the original module after removing debug instructions, including the header
and IDs. This identifies the intended handler functions without changing the
driver's input debug settings. Independent audits verify all twenty-three
original/modified module pairs: only twenty-five selected function-control words
change in total, and both versions pass Vulkan 1.3 validation. The classification
uses matrix-dimension arguments and scalar accumulators as an experimental
filter, not a proposed compiler API. `DontInline` is advisory; the SPIR-V hint
does not prove the driver's machine-code call structure.

All seven unchanged whole-graph smoke tests pass. Twenty alternating pairs per
model phase, followed by a fresh retained-runtime control, give:

| Case | Hint ordinary / persistent ms | Control ordinary / persistent ms | Hint / control paired ratios |
|---|---:|---:|---:|
| SmolVLA forward | 3.646 / 15.518 | 4.224 / 6.082 | 4.258 / 1.442 |
| SmolVLA forward/backward/SGD | 11.824 / 60.716 | 14.897 / 19.514 | 5.049 / 1.312 |

All eighty comparisons retain identical implementation keys, one persistent
compute dispatch, no fallback and the existing maximum training error of
4.66e-9. All samples and outliers remain. These are exploratory twenty-pair
runs; ordinary execution also varies between runs. The much slower persistent
candidate is rejected without claiming full oracle qualification.

Four separate resource diagnostics show forward registers rising from 116 to
127 and training from 114 to 128. Code sizes rise from 101,248/249,344 to
116,352/285,184 bytes; shared storage remains 20,880/29,232 bytes. Those diagnostic
timings and the driver's implausible Local Memory Size field are excluded from
performance conclusions.

Matched complete-step SDK captures include zero-rate SGD and one single-pass
metric group per mode/variant, with 146 samples each, matching adapters,
successful exits and no SDK warnings. Both variants preserve the exact prior
state hash within each backend. All 246,883,105 cross-backend floats pass the
existing comparison budget with maximum error 4.66e-9; they are not bitwise equal
across backends. Selected graph/optimizer pipelines and task counts match the
normal nonzero-rate training benchmark.

Persistent replay rises from **17.402 to 55.590 ms**, while ordinary replay stays
11.944/11.952 ms. Estimated persistent warp instructions rise from **2.830 to
8.595 billion per replay**. Local traffic changes from zero to approximately
**427.10 million load and 3.301 billion store sectors per replay**; ordinary
traffic remains zero. These rate-times-duration estimates support increased
local-memory and instruction costs from the compiler hint. They do not identify
particular stack/spill instructions or prove how much time each cost contributes.
The single-pass capture has no cross-group stability estimate, and zero-rate
replay does not replace the completed-step gate.

Both backend patches and the Cargo override are reverted. The retained examples
are rebuilt from the git-pinned dependencies, and all seven restored whole-graph
smoke tests pass. Every build, test, copy, capture and heavy analysis uses a
verified bounded cgroup; GPU work is serialized and artifacts stay on `/x`.
Proofs, patches, failed and successful logs, binaries, SPIR-V/source dumps,
measurements and **324 independently verified artifact hashes** are indexed by
`target/megakernel-results/noinline-probes/summary.json`. The independent audit
log is `target/noinline-probes-artifact-audit.log`.

The next source-level experiment tested a shared scalar body for horizontal
matrix slices, selecting buffer views uniformly before entering the K loop.
Its result and the subsequent tile selection are recorded below. Baseline
parity remains open.

## Narrow scalar tiles

Runtime `bed8c9196c7821aa2989199f71db5473901dac62` selects existing 32×16 scalar
tiles for unpinned F32 fused matmul/add products with M≤64, N≤1024 and K≥1024,
when the remaining choice is SmallTile without an epilogue. Selection runs
**after split-K planning**, so the existing split kernels and reduction nodes
remain authoritative. It preserves the configured K stage and column mapping;
an illegal shape retains SmallTile. Explicit schedules are unchanged. Both
ordinary and persistent execution use this common selection and retain each
output's original K accumulation order and addend.

Exactly sixteen nodes in each model phase change from 46 to 90 logical tiles,
adding 704 graph tiles. Other implementation keys and appended optimizer tasks
are unchanged. Worker count, queue protocol, canonical descriptors, payload
fences and the single dispatch/submission boundary are unchanged. The production
Naga and Blade pins are unchanged; there are no local dependency overrides.

### Candidate comparison

The preceding horizontal-body experiment uniformly selected B/C buffer views
and offsets before calling one shared scalar body. Source audits confirm one
such body in forward and two in training. Two focused GPU tests cover three
siblings, both 32/64 geometries, tails and changed inputs; all seven whole-graph
smoke tests pass. Eighty candidate/control comparisons show no timing benefit:
persistent forward/training medians are 6.072/19.457 ms versus 6.088/19.245 ms
control. Code size falls to 96,768/233,728 bytes, but that does not justify
retaining the rewrite. It is reverted; no full oracle or SDK qualification is
claimed for it.

Three scalar shapes were then tested with the same eligible fused products:

| Rows × columns | Tiles per changed node | Persistent forward / training ms | Observation |
|---|---:|---:|---|
| 16×32 | 92 | 5.904 / 19.257 | Repeat improves persistent timing, but ordinary forward is 2.3% slower than its fresh control |
| 16×16 | 180 | 6.010 / 19.511 | Twenty-pair exploration does not justify a repeat |
| 32×16 | 90 | 5.943 / 18.915 | Fifty-pair repeat supports lower training latency without an established ordinary regression |

The 32×16 repeat's fresh persistent control is 5.982/19.546 ms. Its candidate /
control median ratio is 0.993 forward (95% bootstrap interval 0.969–1.019) and
0.968 training (0.936–0.987). These are exploratory results, separate from the
clean-runtime primary measurements below. Temporary selectors hard-coded the
default tuning knobs; integration fixes that limitation and preserves the
original split-K policy. Every temporary source rewrite is reverted.

All tile candidates pass their focused exact-output fixture and the existing
whole-graph checks. Each has separate resource diagnostics. The first 16×16
diagnostic failed before execution because its ICD path was mistyped; that log
and empty result remain alongside the successful corrected capture. Separate
single-group complete-step SDK checks for 16×32 and 32×16 preserve the prior
state hash within each backend and measure zero local traffic. Instrumented
timings are excluded from the normal timing gate.

### Qualification and primary measurements

The integrated source passes **666 library tests** (eight ignored), **168 strict
whole-graph oracle tests**, **50 matrix regression tests**, and **eight whole-graph
tests**, plus formatting and strict all-target/all-feature Clippy. The new
selection test covers K-stage 8 fallback, stage 16 with interleaving, stage 32,
an explicit SmallTile schedule and K=8192's original split-K/reduction sequence.
The new GPU fixture compares exact signed dyadic CPU results for 35×65×1031
with an addend, changed inputs, ordinary execution and 1/3/256 workers. It checks
selected geometry, complete worker counts and one persistent compute dispatch.

Clean-runtime binaries then run fifty alternating pairs per case with 256
workers, ten warmups and one second of additional warmup. All **250 primary
comparisons** pass the unchanged full-state budget, matching ordinary/persistent
implementation keys and one persistent compute dispatch without fallback.
The maximum training difference remains 4.66e-9. All samples and outliers are
retained. Times are completed-step wall medians; ratios are medians of paired
ratios and therefore need not equal the ratio of the time columns.

| Case | Ordinary ms | Persistent ms | Paired ratio | 95% bootstrap interval |
|---|---:|---:|---:|---:|
| CI inference | 0.052125 | 0.071101 | 1.363 | 1.355–1.379 |
| CI forward/backward | 0.066411 | 0.102471 | 1.537 | 1.526–1.557 |
| CI forward/backward/SGD | 0.067991 | 0.100801 | 1.490 | 1.478–1.508 |
| SmolVLA forward | 4.134971 | 5.853895 | 1.471 | 1.395–1.475 |
| SmolVLA forward/backward/SGD | 15.088333 | 19.131577 | 1.276 | 1.254–1.388 |

A fresh old-runtime binary control adds fifty pairs per model phase. Persistent
control medians are 6.096263/19.459284 ms, giving candidate / control ratios of
0.960 forward (0.930–1.002) and **0.983 training (0.971–0.993)**. Ordinary control
medians are 4.256071/15.085087 ms; corresponding ratios are 0.972 (0.911–0.985)
and 1.000 (0.971–1.096). Thus training's absolute improvement is about 1.7%,
while forward and normalized paired comparisons remain inconclusive. The
candidate / control ratios of paired medians are 1.030 (0.974–1.045) forward
and 0.995 (0.929–1.078) training. Bootstrap intervals use 10,000 resamples and
describe within-run sampling uncertainty, not between-run drift or other GPUs.
**Baseline parity remains unmet; ordinary execution remains the default.**

### Complete-step counter evidence

Final resource diagnostics report 116/114 registers, 20,880/29,232 bytes of
shared storage and 108,544/256,512 bytes of compiled code for forward/training.
The six-group NVIDIA SDK capture uses complete graph plus zero-rate SGD replay,
146 samples per group per backend, the matching RTX 5080 adapter, successful
target/monitor exits and no SDK warnings. Receipts identify the clean runtime
and preserved binaries. Repeated instruction rates vary by at most 0.245%
across the five groups collecting that metric; elapsed-cycle rates vary by at
most 0.357% across all six groups.

The actual state files match the prior qualified SHA-256 within each backend:
`f67022044ae8966ac48c5d62cd0907411fb42f4d1864c1ba0b3d555b17ce7be0`
for ordinary and
`1e359875fa92965d4b0aa9026c050702ae3ac165aa59c0a4cc74a6381589c3b7`
for persistent. Within-backend replay is exact. All 246,883,105 floats pass the
explicit benchmark comparison across backends with maximum error 4.66e-9;
cross-backend state is not bitwise equal. Selected graph and optimizer work
matches the normal nonzero-rate training case.

Diagnostic replay medians are 12.003 ms ordinary and 17.058 ms persistent.
Rate-times-duration estimates give **1.546 versus 2.825 billion warp
instructions**, **2.346 versus 6.342 billion integer thread instructions** and
**zero local load/store traffic** in both modes. Barrier waiting represents
3.121% versus 50.165% of active warp cycles; memory-barrier waiting is 0% versus
1.461%. These are counter shares, not elapsed-time attribution. They support
continued investigation of synchronization and execution overhead but do not
prove a specific stall's contribution to latency. Zero-rate SDK replay does
not replace the normal completed-step performance gate.

Every build, test, copy, capture and heavy analysis runs in a verified bounded
cgroup; GPU jobs are serialized and all artifacts remain on local `/x` storage.
The complete exploration and final qualification retain **950 normal paired
comparisons**, **ten resource diagnostics**, proofs, patches, binaries, source/
SPIR-V dumps, SDK receipts and state files. The manifest
`target/megakernel-results/narrow-scalar-tiles/summary.json` indexes **637
independently verified artifact hashes**; the independent audit log is
`target/narrow-scalar-artifact-audit.log`.

The subsequent scratch-clearing and pointwise-batching experiments are recorded
below. Neither is retained; the qualified runtime remains `bed8c91`.

## Scratch clearing and pointwise batching probes

The scratch probe omits the callable converter's explicit zeroing loops and
initial workgroup barrier for F32 scalar matmul bodies only. Each K stage already
overwrites all shared A/B cells it reads, including zero values for edge inputs,
then synchronizes before computing. Padding is never read; K=0 and empty splits
perform no shared reads. The original end-of-stage and worker-loop barriers
order subsequent reuse. Admission covers plain/AT/BT products, addends, scalar
horizontal slices and split-K bodies, preserving their original geometry and
arithmetic. Other families, regional execution and all payload fences retain
their existing behavior. Shared capacity and pooling are unchanged.

Source audits confirm removal from four forward and ten training handlers.
All eight whole-graph checks and all 168 strict whole-graph oracle tests pass.
Normal measurements retain **400 comparisons**: twenty-pair CI/model trials
and fresh controls, then fifty-pair model repeats with another fresh control.
Every comparison preserves implementation keys, logical work, one compute
dispatch and no fallback. Maximum training error remains 4.66e-9. The first
repeat-control launch used a mistyped ICD path and failed before GPU execution;
its log and empty result are preserved beside the successful corrected run.

| Case | Candidate ordinary / persistent ms | Control ordinary / persistent ms | Persistent candidate / control | 95% bootstrap interval |
|---|---:|---:|---:|---:|
| SmolVLA forward, fifty pairs | 4.154 / 5.791 | 4.209 / 5.811 | 0.997 | 0.955–1.012 |
| SmolVLA training, fifty pairs | 15.173 / 18.953 | 14.848 / 19.224 | 0.986 | 0.944–1.001 |
| CI inference, twenty pairs | 0.049471 / 0.069051 | 0.057451 / 0.075850 | 0.910 | 0.870–0.931 |
| CI backward, twenty pairs | 0.066565 / 0.099536 | 0.066685 / 0.102351 | 0.973 | 0.962–0.991 |
| CI SGD, twenty pairs | 0.068116 / 0.102576 | 0.068500 / 0.101465 | 1.011 | 0.994–1.054 |

All samples and outliers remain. The large ordinary CI inference movement
illustrates why the absolute changes alone cannot establish a runtime benefit.
Candidate/control ratios of paired medians are 1.001 (0.970–1.060) forward and
0.977 (0.895–1.001) training. CI SGD's paired ratio regresses from 1.483 to 1.517,
a relative ratio of 1.023 (1.008–1.052). Intervals use 10,000 bootstrap resamples
and describe sampling within these runs, not between-run drift. The probe does
not establish a useful completed-step gain and is reverted.

Separate resource diagnostics report unchanged 116/114 registers and
20,880/29,232 bytes of shared storage. Code decreases from 108,544/256,512 to
104,576/246,656 bytes for forward/training. Candidate and fresh-control SDK
captures each collect three groups per backend, with 146 samples per group,
matching adapters, successful exits and no warnings. The actual state files
match the prior qualified hash within each backend; cross-backend state passes
the explicit benchmark comparison with maximum error 4.66e-9. Replay is exact
within each backend, and graph/optimizer arithmetic remains matched.

| Complete-step SDK diagnostic | Candidate | Fresh control |
|---|---:|---:|
| Ordinary replay ms | 12.002 | 12.009 |
| Persistent replay ms | 17.017 | 17.064 |
| Persistent warp instructions per replay | 2.808 billion | 2.829 billion |
| Persistent shared wavefronts per replay | 537.518 million | 541.199 million |
| Persistent barrier share of active warp cycles | 50.027% | 50.299% |
| Local load/store traffic, both backends | zero | zero |

Repeated instruction and clock rates vary by at most 0.327% and 0.267%
respectively across these captures' three groups. The small reductions in
estimated work and diagnostic replay time do not override the normal timing
gate. Counter counts are rates multiplied by replay duration, and barrier
shares are not elapsed-time attribution. The proof, patches, rejected source,
binaries, dumps, measurements and SDK states are indexed by
`target/megakernel-results/scalar-scratch/summary.json`, with **177 independently
verified hashes** and audit log `target/scalar-scratch-artifact-audit.log`.

The following pointwise experiment reuses the existing scratch-free sequential
batch loop, original logical coordinates and clamped completion RMW. Admission
requires generated F32 pointwise bodies without broadcasts and matching logical/
physical workgroup sizes. The first version requests four tiles per claim at
the existing 1,024-tile threshold. A second version requests two at 512, leaving
SGD's four/1,024 policy unchanged. Both pass nine whole-graph tests, including a
new exact-output fixture with changed inputs, 1/3/256 workers, a partial
1,025-tile grid and a three-tile grid using the same handler. Its initial failure
was a mistaken assertion about compilation order, before execution; the corrected
fixture requires the exact multiset of two large and two small grids.

Neither policy selects model work. A construction audit finds five generated
pointwise nodes totaling 161 tiles in forward and forty totaling 4,690 tiles
in training. The largest grid has **141 tiles**. After excluding broadcasts,
eligible work totals only seven forward tiles and 4,536 training tiles; none
meets either per-node threshold. There is no reason to relax broadcast admission
to reach these thresholds. The two variants' eighty normal comparisons are
therefore controls, not evidence of a batching speedup. Two resource diagnostics
match the retained pipeline reports, and both complete generated model shader
files are byte-for-byte identical to the retained runtime. Full oracle and SDK
qualification are not claimed for these prototypes.

Both pointwise policies, temporary diagnostics and the temporary fixture are
reverted. The evidence is indexed by
`target/megakernel-results/pointwise-batches/summary.json`, with **122 independently
verified hashes** and audit log `target/pointwise-batches-artifact-audit.log`.
The retained examples are rebuilt byte-for-byte identically to the preserved
qualified binaries, and all eight restored whole-graph checks pass. Every
build, test, copy, capture and heavy analysis uses a verified bounded
cgroup; GPU jobs are serialized and all artifacts stay on local `/x` storage.

The following experiment compares the existing contiguous and interleaved
scalar column mappings while preserving shapes, K staging and each output's
accumulation order. Baseline parity remains open.

## Interleaved scalar columns

An exploratory default-policy probe enables the existing interleaved scalar
column mapping in both backends. The same mapping selects B shared-memory
reads, fused addends and output columns. It preserves tile dimensions, K
staging, split count, precision and every output's accumulation order; explicit
scalar-shape choices remain authoritative. Exactly sixteen model shape keys per
phase change their interleave flag, while logical work and optimizer tasks stay
unchanged. Generated model sources confirm the alternate second-column loads.

Eight whole-graph smoke checks and **200 normal paired comparisons** pass.
Each case has twenty AB/BA pairs after ten warmups and at least one second of
warmup. All samples are retained. Full-state budgets, matched backend selection,
one persistent dispatch and no fallback remain intact; maximum training error
is 4.66e-9. These are exploratory runs, with no full oracle or matrix-regression
qualification claimed for the rejected probe.

| Case | Candidate ordinary / persistent ms | Fresh control ordinary / persistent ms | Candidate / control persistent median, bootstrap 95% interval |
|---|---:|---:|---:|
| Model forward | 3.844 / 5.677 | 4.185 / 5.802 | 0.978 [0.948, 1.027] |
| Model training + SGD | 15.189 / 19.731 | 15.128 / 18.930 | 1.042 [0.991, 1.055] |
| CI inference | 0.05876 / 0.07716 | 0.05219 / 0.07045 | 1.095 [1.061, 1.110] |
| CI backward | 0.06715 / 0.10169 | 0.06696 / 0.10278 | 0.989 [0.980, 1.016] |
| CI SGD | 0.06890 / 0.10325 | 0.06792 / 0.10185 | 1.014 [0.986, 1.039] |

The intervals describe within-run sampling uncertainty, not between-run drift.
The paired persistent/ordinary medians are 1.485 / 1.310 for the two model
phases, versus 1.403 / 1.271 in the fresh control. This does not establish a
useful completed-step benefit.

Two separate resource diagnostics retain 116 / 114 registers and 20,880 /
29,232 shared bytes for forward / training. Code grows from 108,544 / 256,512
to 109,440 / 258,560 bytes. Three supported SDK groups per backend each retain
146 post-warmup samples, no monitor warnings and the exact prior same-backend
state hashes. Cross-backend replay passes the existing benchmark budget.

| Training replay | Candidate | Fresh control |
|---|---:|---:|
| Ordinary / persistent ms | 12.287 / 17.676 | 12.008 / 17.083 |
| Ordinary / persistent estimated warp instructions | 1.590 / 2.877 billion | 1.542 / 2.831 billion |
| Ordinary / persistent estimated shared wavefronts | 407.615 / 476.807 million | 447.523 / 543.752 million |
| Persistent barrier share of active warp cycles | 51.07% | 50.09% |

Both modes retain zero measured local load/store traffic. The smaller shared
wavefront estimate does not produce lower instruction count or replay time.
These counters do not establish a bank-conflict or vectorization explanation.
Counts are rate-times-duration estimates; barrier shares are not elapsed-time
attribution. SDK replay uses zero-rate SGD and does not replace normal timing.

The probe is reverted exactly. Its proof, patch, binaries, samples and captures
remain at `target/scalar-interleave-probe/`. The independently checked manifest
is `target/megakernel-results/scalar-interleave/summary.json` with **168 artifact
hashes**; the restore audit is `target/scalar-interleave-artifact-audit.log`.
All jobs used verified bounded cgroups and local storage, with serialized GPU
work and no builds or analysis during normal timing or SDK capture.

## Cooperative subgroup ownership

The native cooperative matmul template stages inputs with all 64 invocations,
then each subgroup computes the same four accumulators. Only logical subgroup
zero stores their results, including the shared result used by scalar epilogues.
A one-template probe wraps cooperative operand loads and products in the same
subgroup-zero condition. It includes compensated products, preserves accumulator
initialization and every K iteration, and leaves both surrounding workgroup
barriers outside the condition. The separate 8×8 F32 and workgroup-scoped
templates are unchanged. Ordinary and persistent execution share this change;
packed persistent tiles retain independently virtualized subgroup indices.

Eight whole-graph smoke checks and **200 normal paired comparisons** pass with
the unchanged state budgets and maximum training error of 4.66e-9. Selected
implementation keys, logical work, optimizer tasks, one persistent dispatch and
absence of fallback match the retained runtime. Generated graph sources confirm
the four operand loads and products are inside the owner guard, between the
original barriers. Each case uses twenty AB/BA pairs, ten warmups and at least
one warmup second. All samples remain; no full oracle or matrix-regression
qualification is claimed for this rejected probe.

| Case | Candidate ordinary / persistent ms | Fresh control ordinary / persistent ms | Candidate / control persistent median, bootstrap 95% interval |
|---|---:|---:|---:|
| Model forward | 4.478 / 5.764 | 4.089 / 5.762 | 1.000 [0.976, 1.052] |
| Model training + SGD | 14.860 / 18.795 | 15.064 / 19.046 | 0.987 [0.917, 1.076] |
| CI inference | 0.05213 / 0.07190 | 0.05284 / 0.07241 | 0.993 [0.948, 1.074] |
| CI backward | 0.06595 / 0.10292 | 0.06670 / 0.10281 | 1.001 [0.993, 1.009] |
| CI SGD | 0.06799 / 0.10224 | 0.06794 / 0.10148 | 1.007 [0.991, 1.023] |

Ordinary forward is slower in this short run: candidate/control ratio 1.095
[1.043, 1.216]. Consequently its lower paired persistent/ordinary ratio, 1.280
versus 1.465, is not a persistent speedup. Training's paired ratio is essentially
unchanged, 1.270 versus 1.269. Bootstrap intervals describe within-run sampling
uncertainty, not between-run drift; these results do not establish a useful gain.

Two separate resource captures show forward registers falling from 116 to 100,
while training increases from 114 to 115. Shared memory stays at 20,880 / 29,232
bytes; code changes from 108,544 / 256,512 to 110,080 / 256,640 bytes.
Four supported SDK groups per backend collect 146 samples each, with matching
adapters, successful exits, no warnings and exact prior same-backend state
hashes. Cross-backend state passes the existing benchmark budget.

| Training replay | Candidate | Fresh control |
|---|---:|---:|
| Ordinary / persistent ms | 12.004 / 17.016 | 12.007 / 17.064 |
| Ordinary / persistent estimated warp instructions | 1.567 / 2.814 billion | 1.547 / 2.830 billion |
| Ordinary / persistent estimated shared wavefronts | 444.537 / 533.853 million | 449.838 / 539.899 million |
| Ordinary / persistent tensor-active percentage | 0.4314% / 0.3013% | 0.8393% / 0.5993% |
| Persistent barrier share of active warp cycles | 50.23% | 50.29% |

Both modes retain zero measured local load/store traffic. Multiplying the tensor
percentage by replay duration yields approximately 51.4% / 50.1% of the control's
ordinary / persistent tensor-active work. This supports actual duplication in
the original shader, but removing it changes persistent replay by only 0.28%
and does not improve the normal completed-step gate. Tensor-active percentage
is not elapsed-time attribution. Instruction and wavefront totals are
rate-times-duration estimates; SDK zero-rate SGD replay remains diagnostic.

The probe is reverted exactly. Its proof, source patch, measured models-feature
binaries, samples and captures are in `target/coop-owner-subgroup-probe/`.
The first build omitted the required models feature and failed before compiling;
its log is preserved. An additional all-feature build was not used for timing
or SDK capture. The independently checked manifest is
`target/megakernel-results/coop-owner-subgroup/summary.json`, with **185 artifact
hashes**; the restore audit is `target/coop-owner-artifact-audit.log`.
Both rebuilt benchmark binaries match the retained binaries byte-for-byte, and
all eight restored whole-graph checks pass. The final records are
`target/coop-owner-restored-{build,smoke}.log` and
`target/coop-owner-final-audit.log`.
All builds, tests, copies, captures and analysis use verified bounded cgroups,
serialized GPU jobs and local storage. Builds and analysis do not overlap
normal timing or SDK capture. The retained runtime and primary measurements
remain unchanged; baseline parity remains unmet.

The following investigation distributes distinct cooperative output rows across
subgroups while preserving output geometry and arithmetic. The owner-only result
does not by itself establish that a different distribution will be faster.

## Cooperative row distribution

Runtime `762faf20b03a0d36e0a56a9b5180aa486aafd0bd` distributes the native
cooperative matmul's two output row halves across logical subgroups zero and
one. Each computes two accumulators. At subgroup width 64, subgroup zero also
computes the second row's two accumulators; additional subgroups at narrower
widths never store. The original 64-invocation geometry, four operand tiles,
single K loop and unconditional staging/barriers remain. Packed graph tiles
retain separate virtual subgroup coordinates and four-tile packing.

Each output preserves its K order, conversion, fused-add initialization and
compensated main/high-low/low-high product order. Typed splits, output casts,
post-product addends, horizontal copies, prologue caching and shared epilogues
retain their behavior. The separate 8×8 F32 and workgroup-scoped templates are
unchanged. New fragments reside in the existing WGSL template. No scheduler,
descriptor, fence, Naga or validation-flag change is needed. Both ordinary and
persistent execution use the same updated template.

The initial variant put K loops inside subgroup-width-dependent rounds. It
produced correct results but failed the existing packing assertion: Naga's
scope-aware admission analysis could not prove the outer loop workgroup-uniform,
so the native handler packed one tile instead of four. Its 80 comparisons,
two resource captures and failed assertion remain in the record. The retained
variant keeps barriers outside all subgroup-dependent control. Ownership proofs
and shader validation cover alternate widths; hardware execution is qualified
only on RTX 5080.

Qualification passes **666 library tests** (eight existing ignores), **168 strict
whole-graph oracle tests**, **50 matrix regressions**, **eight whole-graph smoke
tests**, formatting and strict all-target/all-feature Clippy. One existing typed
source assertion referred to the old accumulator name; it now checks both row
paths and all four static multiply-adds. No numerical tolerance was changed.
Although this edit only affected a test, rebuilding changed the example binary
hashes. The failed equivalence audit is preserved, and the final binaries
received fresh primary measurements, resources and both-phase SDK captures.
Their generated model shaders match the exploratory fixed-barrier shaders
byte-for-byte; those exploratory timings remain separately identified.

The final primary dataset contains **250 comparisons**, fifty AB/BA pairs for
each case, plus **250 fresh old-binary control comparisons**. Every case passes
the existing state budgets and completion checks, with matching implementation
keys, one persistent compute dispatch and no fallback. Maximum training error
remains 4.66e-9. Paired ratios are medians of per-pair ratios, not ratios of the
separate latency medians.

| Case | Ordinary ms | Persistent ms | Paired persistent / ordinary, bootstrap 95% interval |
|---|---:|---:|---:|
| CI inference | 0.05225 | 0.07096 | 1.363 [1.352, 1.371] |
| CI backward | 0.06629 | 0.10208 | 1.545 [1.535, 1.555] |
| CI SGD | 0.06768 | 0.10187 | 1.508 [1.493, 1.531] |
| Model forward | 4.270 | 5.654 | 1.382 [1.328, 1.388] |
| Model training + SGD | 15.177 | 19.106 | 1.256 [1.235, 1.268] |

Against the fresh controls, persistent forward changes from 5.814 to 5.654 ms:
ratio **0.973 [0.913, 0.982]**, about 2.7% lower. Training changes from 19.216 to
19.106 ms: **0.994 [0.979, 0.997]**, about 0.6% lower. Normalizing by each run's
ordinary timings gives forward **0.996 [0.916, 1.016]**, an inconclusive change,
and training **0.969 [0.947, 0.991]**. These are modest gains; parity remains unmet.

CI inference varies between runs despite identical ordinary and persistent GPU
pipeline identities. The primary persistent median is 6.5% lower than control,
but its normalized ratio is 2.9% higher. An additional **600-comparison A/B/B/A
sequence** gives candidate/control persistent ratios 1.113 then 0.966, with
normalized ratios 0.965 then 0.988; the second comparison's intervals include
no change. CI backward and SGD intervals include no change. No stable CI
inference speedup or regression is established, nor a specific host-side cause.
All samples remain. Bootstrap intervals use 10,000 resamples and describe
within-run uncertainty, not between-run variation.

Two final resource captures show persistent forward/training register counts
falling from 116/114 to **100/103**. Shared memory remains **20,880/29,232 bytes**;
code changes from 108,544/256,512 to **106,752/256,512 bytes**.

Four SDK groups per backend and phase collect 146 post-warmup samples each.
All final captures have matching adapters, successful exits, no monitor warnings,
exact prior same-backend state hashes and zero measured local load/store traffic.
Cross-backend state passes the existing benchmark budget. Training SDK replay
uses zero-rate SGD and is diagnostic, separate from normal completed-step timing.

| Final SDK replay | Candidate ordinary / persistent | Fresh control ordinary / persistent |
|---|---:|---:|
| Forward ms | 3.852 / 5.248 | 3.677 / 5.324 |
| Training ms | 12.001 / 16.909 | 11.997 / 17.074 |
| Forward estimated warp instructions, billion | 0.382 / 0.679 | 0.382 / 0.696 |
| Training estimated warp instructions, billion | 1.549 / 2.803 | 1.553 / 2.824 |
| Forward estimated shared wavefronts, million | 126.057 / 141.433 | 129.848 / 144.561 |
| Training estimated shared wavefronts, million | 448.750 / 536.862 | 451.353 / 539.028 |
| Forward barrier share of active warp cycles | 4.10% / 57.17% | 3.97% / 57.42% |
| Training barrier share of active warp cycles | 3.12% / 50.35% | 3.13% / 50.21% |

Duration-weighted tensor activity is approximately half the control in both
phases and backends. Persistent instruction estimates fall about 2.5% forward
and 0.8% training, but training still executes about **1.81× ordinary warp
instructions**. Barrier waiting remains substantial. Ordinary forward SDK replay
is slower in this capture, consistent with treating normalized forward benefit
as unresolved. Rates multiplied by replay duration are estimates, and active
warp-cycle shares are not elapsed-time attribution. These counters do not prove
that a particular queue instruction or cache effect causes the remaining gap.

All **1,880 normal comparisons**, **six resource diagnostics** and **seven SDK
capture directories** (three exploratory, four final) are preserved under
`target/coop-row-subgroups-{probe,packed,qualified}/`. The manifest is
`target/megakernel-results/coop-row-subgroups/summary.json`, with **607 independently
verified artifact hashes**. It also checks source patches against base `70c75f4`,
generated packing, pipeline identities, exact SDK state files and restored final
binaries. Verification logs are `target/coop-rows-manifest.log` and
`target/coop-rows-artifact-audit.log`.

The final CI binary SHA-256 is
`9164358b0a3baf3e3c3a491ebbb4199736fb05fd81289c98edff18fa88af5fce`;
the model binary is
`f4c43ae7e5a36fbe571cb51496a0f7237982401cc1d01dfed76dbac70fe7448a`.
Builds, tests, copies and analyses use verified bounded cgroups on local `/x`
storage. GPU jobs are serialized; builds and heavy analysis do not overlap
normal timing or SDK capture. The following investigation measures explicit
worker counts on this geometry and deeper scalar K staging under the same
arithmetic, completion and completed-step gates.

## Worker counts and deeper scalar staging

On source base `e9a32a9`, the qualified row-distribution binaries are swept at
explicit worker counts, without changing runtime code or defaults. The earlier
CI sweep used different persistent pipelines; the earlier model sweep preceded
the current geometry and register reductions. Fresh 256-worker controls bracket
each sweep. Every run retains matching arithmetic, complete state/completion
checks, one persistent dispatch, no fallback and exact requested/reported
worker counts. All GPU pipeline identities match the qualified runtime for
each workload; only the runtime pool count changes.

CI uses fifty AB/BA pairs per case, **900 comparisons** total:

| Workers, execution order | Inference persistent ms | Backward persistent ms | SGD persistent ms |
|---|---:|---:|---:|
| 256, before | 0.07159 | 0.10218 | 0.10155 |
| 32 | 0.06816 | 0.09689 | 0.09803 |
| 128 | 0.07440 | 0.09794 | 0.09813 |
| 64 | 0.07335 | 0.09672 | 0.09720 |
| 16 | 0.07487 | 0.09836 | 0.09859 |
| 256, after | 0.07081 | 0.10235 | 0.10157 |

At 64 workers, backward is about 5.3–5.5% below both controls and SGD about
4.3% below them; normalized comparisons also improve. Inference comparisons
remain inconclusive because ordinary latency varies between runs. The 64-worker
paired persistent/ordinary ratios are still 1.349/1.460/1.435, so this does not
close the baseline gap. It is evidence for an explicit CI choice, not for
silently capping requested workers or a graph-independent automatic policy.

The model uses twenty pairs per phase, **240 comparisons** total. Counts 168
and 336 also sample two/four workers per SM on this 84-SM device; that is not
a portable workgroup-placement assumption.

| Workers, execution order | Forward persistent ms / paired ratio | Training persistent ms / paired ratio |
|---|---:|---:|
| 256, before | 5.714 / 1.366 | 19.076 / 1.273 |
| 128 | 5.446 / 1.394 | 20.467 / 1.373 |
| 336 | 5.415 / 1.354 | 18.933 / 1.264 |
| 168 | 5.379 / 1.386 | 18.900 / 1.289 |
| 512 | 5.725 / 1.370 | 19.080 / 1.281 |
| 256, after | 5.580 / 1.374 | 18.991 / 1.266 |

128 workers slows training substantially. Neither 168 nor 336 establishes a
normalized improvement against both controls; forward ordinary latency ranges
from 3.861 to 4.248 ms. Keep the existing model setting and preserve all samples.
The raw records, binaries, receipts and comparisons are in
`target/worker-count-current/`.

Two subsequent prototypes add an explicit 64-deep setting to the existing
scalar staging knob and pass it through the shared paired benchmark config to
both backends. The default remains 32 throughout. Explicit schedules, including
the existing 16-deep split-long products, stay pinned; quantized decoders retain
their fixed 32-deep layout. Each output retains ascending-K arithmetic and the
old 32-rounded tail, rather than adding another 32 zero FMAs. Neither prototype
changes the queue, descriptors, fences or submission boundary.

The first stages 64 values and uses a variable arithmetic-loop bound for the
final tail. The second executes two fixed 32-step inner loops and guards only
the second loop. Counted and unrolled 64-deep ragged products are added to the
existing smoke fixture and compared with an independent CPU result under
ordinary execution and 1/3/256 persistent workers. Both variants pass all eight
expanded whole-graph checks and **200 normal comparisons each**, twenty pairs
for all five cases with fresh 32-stage controls from the same binaries.

| Variant | Forward ordinary / persistent ms | Training ordinary / persistent ms |
|---|---:|---:|
| Variable bound, 32-stage control | 3.904 / 5.406 | 14.947 / 18.880 |
| Variable bound, 64-stage candidate | 4.267 / 6.680 | 14.887 / 21.193 |
| Fixed loops, 32-stage control | 4.219 / 5.404 | 14.818 / 18.991 |
| Fixed loops, 64-stage candidate | 3.918 / 5.750 | 14.464 / 19.088 |

The variable-bound candidate regresses persistent forward/training by about
24%/12%. Fixed loops reduce that loss, but forward still regresses about 6.4%
and training is effectively unchanged. Neither improves both model phases.
CI backward/SGD show small gains, insufficient to justify the model regressions.
The short samples include ordinary-training outliers; nothing is removed.
Bootstrap intervals describe within-run uncertainty, not drift or causal
attribution. No full oracle or matrix-regression qualification is claimed for
these rejected candidates.

Eight separate resource diagnostics show:

| Combined shader | Registers forward / training | Shared bytes forward / training | Code bytes forward / training |
|---|---:|---:|---:|
| 32-stage controls, both variants | 100 / 103 | 20,880 / 29,232 | 106,752 / 256,512 |
| 64-stage variable bound | 102 / 112 | 29,200 / 45,744 | 108,928 / 292,608 |
| 64-stage fixed loops | 100 / 114 | 29,200 / 45,744 | 133,376 / 342,656 |

Deeper staging trades barrier frequency for greater scratch and code size;
these resource observations do not assign the latency regression to one cause.
Generated graph sources verify the variable bound and the fixed-loop tail guard
with second-half shared indices. The first source-audit assertion expected the
pre-emission variable spelling; it was corrected to check the emitted loop and
index structure. Its failed log remains `target/workers-stage64-manifest.log`.
No new SDK counter capture is claimed for these probes.

Both prototypes, the temporary benchmark option and the expanded test are
reverted exactly. Rebuilt examples match the qualified binaries byte-for-byte,
and all eight restored whole-graph checks pass. The patches, proofs, measured
binaries and generated sources remain in `target/scalar-stage64-probe/` and
`target/scalar-stage64-fixed-loops/`.
The combined manifest is `target/megakernel-results/workers-stage64/summary.json`:
**1,540 normal comparisons, eight resource diagnostics and 463 independently
verified artifact hashes**. Successful audit logs are
`target/workers-stage64-manifest-fixed.log` and
`target/workers-stage64-artifact-audit.log`; restoration logs are
`target/workers-stage64-restored-{build,smoke}.log`.

All work uses verified bounded cgroups and local `/x` storage. GPU jobs are
serialized, and no builds, copies or heavy analysis overlap normal timing.
The qualified runtime remains `762faf2`, and baseline parity remains unmet.
The subsequent scheduler locality experiments are recorded below.

## Scheduler locality probes

Two experiments start from `a611752`, with the qualified runtime and binary
controls from `762faf2`. Neither changes arithmetic selection, worker counts,
graph geometry, memory fences or the complete-step submission boundary. Each
uses twenty AB/BA pairs per case, ten warmups and at least one second of warmup,
with a fresh control process before the candidate. All 400 normal comparisons
pass the existing state and completion gates, matching implementation keys,
one persistent compute dispatch and no fallback. Maximum training error remains
4.66e-9. Bootstrap intervals use 10,000 resamples with seed 20261026 and describe
within-run uncertainty, not drift or causal attribution.

The first probe starts each worker's full ready-bitmap scan at its last
successful ready word. Every word remains reachable, including wraparound.
The expanded fixture contains two independent chains and a join across 99
operations, with allocation reuse disabled to preserve their independence.
It passes all eight whole-graph smoke checks with poison, repeated steps,
external recording, serial mode and worker counts 1, 3, 64 and 4096.

The second probe lets the final producer reserve one newly ready successor
whose entire logical workload fits in one existing claim. Its ready bit stays
unpublished, so the already-running worker has exclusive ownership. Other
successors publish normally. The successor remains in the unfinished-node
count and follows the original claim, payload and completion fences. A private
reservation passes all eight smoke checks, including both multi-claim and
single-claim versions of the 99-operation fixture.

| Probe / case | Control ordinary / persistent ms | Candidate ordinary / persistent ms | Persistent candidate / control [95% interval] |
|---|---:|---:|---:|
| Cursor, CI inference | 0.05253 / 0.07089 | 0.05165 / 0.06989 | 0.986 [0.951, 1.057] |
| Cursor, CI backward | 0.06520 / 0.10265 | 0.06626 / 0.10173 | 0.991 [0.981, 1.034] |
| Cursor, CI SGD | 0.06827 / 0.10286 | 0.06825 / 0.10092 | 0.981 [0.973, 0.996] |
| Cursor, model forward | 4.221 / 5.642 | 4.309 / 5.718 | 1.014 [0.993, 1.064] |
| Cursor, model training | 14.779 / 19.006 | 14.768 / 19.553 | 1.029 [0.971, 1.045] |
| Handoff, CI inference | 0.05327 / 0.07132 | 0.05696 / 0.07378 | 1.035 [0.947, 1.113] |
| Handoff, CI backward | 0.06646 / 0.10286 | 0.06650 / 0.10156 | 0.987 [0.980, 0.996] |
| Handoff, CI SGD | 0.06855 / 0.10278 | 0.06834 / 0.10271 | 0.999 [0.973, 1.018] |
| Handoff, model forward | 4.245 / 5.764 | 3.947 / 5.742 | 0.996 [0.944, 1.070] |
| Handoff, model training | 14.767 / 19.073 | 14.954 / 18.894 | 0.991 [0.950, 1.008] |

The cursor's training ratio normalized to ordinary execution worsens by 3.2%
[1.015, 1.157]. The handoff's normalized candidate/control ratios are 1.012
[0.991, 1.056] forward and 0.992 [0.835, 1.109] training. Neither establishes a
useful model improvement, so both are reverted. The isolated CI reductions
do not justify retaining either change.

Eight separate resource diagnostics pass the same correctness gates. Forward /
training registers remain 100 / 103 and shared storage 20,880 / 29,232 bytes for
both candidates and controls. Generated executable sizes are:

| Variant | Forward bytes | Training bytes |
|---|---:|---:|
| Qualified control | 106,752 | 256,512 |
| Ready-word cursor | 106,880 | 256,640 |
| Private successor handoff | 107,776 | 257,536 |

No new SDK capture or counter-based causal claim accompanies these rejected
probes. The qualified SDK results above remain the current counter evidence.

The handoff investigation also preserves a correctness failure. An initial
workgroup scalar reservation produces nonfinite output in the packed-matrix
scratch test at 2080 columns, one worker and the first replay. Instrumenting
the reservation branch confirms that this graph never takes that branch;
neither of its two nodes fits one claim. Deferring metadata writes until the
next claim still fails, as does a vec4-padded shared reservation. Using a private
reservation passes the focused case and complete eight-test suite. This shows
sensitivity to the added shared state, but does not establish an alignment,
Naga or driver cause. Five failing test logs and their source variants remain
in `target/single-claim-handoff/`; no failed variant is retained in the runtime.
The next correctness investigation will minimize that reproducer and inspect
its emitted WGSL, SPIR-V and shared layout; the follow-up is recorded below.

Raw measurements, patches, proofs, binary receipts and generated shaders remain
under `target/ready-word-cursor/` and `target/single-claim-handoff/`. The combined
manifest is `target/megakernel-results/scheduler-locality/summary.json`.
Its 435 artifact hashes are independently verified; the manifest SHA-256 is
`b711bf95d0043d92cf49bd9a662a1a3aea9e33d95c13359c2986102c519aef4d`.
Both prototypes and their temporary test extensions are reverted. Rebuilt
examples match the qualified binaries byte-for-byte, and all eight restored
smoke tests pass. Restoration and audit logs use the
`target/scheduler-locality-*` prefix. Every heavy job uses a verified separate
cgroup with the existing 16/20/1 GiB limits; GPU jobs are serialized and artifacts
remain on local `/x`. The qualified runtime, Naga/Blade pins and primary
measurements remain unchanged. Baseline parity is still unmet.

## Shared-state failure reduction

At `f78a9f4`, the failing graph is reduced to 53×720 input, 2080 output columns,
one worker and one replay of the existing packed-matrix scratch test. The
smallest worker change reproduced here adds a live shared `u32`, stores the
worker ID in it and uses it for the initial ready-word index. There is no
successor handoff. The captured payload WGSL is identical to the passing
control; only this shared declaration, store and scan-start read differ.

| Diagnostic | Result |
|---|---|
| Original shared handoff | Nonfinite output at index zero |
| Unused extra shared scalar | Pass |
| Live shared worker ID, before or after scratch declarations | Nonfinite output at index zero |
| Live scalar with an initialization barrier | Incorrect finite output |
| Two or three live shared words | Incorrect finite output |
| Four or eight live shared words | Pass |
| Explicit `Aligned 16` operands on all five cooperative loads | Still fails |
| Standalone 16×16 cooperative matrix with 1/2/3/4/8 live words | All five cases pass |
| Packing disabled in graph admission | Stalls; explicitly cancelled |

The four/eight-word probes initialize and consume every word, unlike the
earlier vec4 variant that used only its first component. The pattern is
consistent with a layout sensitivity but does not establish its cause.
Vulkan requires these cooperative matrix pointers and strides to meet the
minimum of 16 bytes and the natural row/column alignment; static SPIR-V
validation does not prove the runtime address alignment.
See [VUID-RuntimeSpirv-OpCooperativeMatrixLoadKHR-08986](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html#VUID-RuntimeSpirv-OpCooperativeMatrixLoadKHR-08986).
All 17 captured SPIR-V modules validate for Vulkan 1.3. Requesting driver internal
representations returns an empty list, so no native instruction or actual
shared-address inspection is claimed. The aligned-load diagnostic changes
only the five cooperative-load memory operands and does not fix the output.

The first attempt to disable packing changed a builder default that graph
admission overrides; its generated shader is unchanged and its failed run is
retained as an ineffective probe. The corrected admission change reduces each
half-precision scratch array from 512 to 256 elements, but stalls after the
shader dump. Repeated checks find the process still live, waiting without
new output or sustained CPU/GPU activity. Its exact cgroup is explicitly
stopped after more than two minutes, and the driver records exit -15. This is
a cancelled diagnostic, not a proven GPU deadlock or a passing check. The
standalone harness also retains its initial compile failure from an invalid
cleanup-method call; correcting that harness allows all five matrix cases
to execute successfully.

The record contains 15 focused runs: five passes, nine failures and one
cancellation, plus the five standalone matrix cases. These are correctness
diagnostics, with no normal benchmark or new SDK performance claim. All
temporary worker, test, packing, example and Blade instrumentation changes
are reverted. Rebuilt benchmark binaries exactly match the qualified hashes,
and all eight restored whole-graph smoke tests pass. Work remains inside
verified bounded cgroups, with serialized GPU jobs and local `/x` artifacts.

Sources, patches, outputs and process observations are in
`target/shared-state-repro/`. The manifest
`target/megakernel-results/shared-state-repro/summary.json` has 206 independently
verified artifact hashes and SHA-256
`18701f31cac9804d26dbeb34515543dc91090e4b2587643bcb13672cb077b487`.
The cause remains open; no alignment workaround or dependency change is retained.

The inspection identifies a separate performance hypothesis. Current Naga
emits per-access availability/visibility for ordinary workgroup memory accesses,
while workgroup control barriers provide ordering without the corresponding
propagation bits. Before changing this, audit whether existing barriers can
provide that propagation for every supported synchronization path, preserving
NonPrivatePointer and explicit decorations. The
[Vulkan memory model](https://docs.vulkan.org/spec/latest/appendices/memorymodel.html#availability-visibility-semantics)
defines the required barrier propagation; it does not by itself prove the
proposed compiler transformation. The subsequent prototype and its limits are
recorded below. It is not an explanation for the shared-state failure.

## Workgroup barrier propagation

At `e246ee6`, a temporary Naga change moves ordinary workgroup-memory
availability/visibility from individual accesses to the existing workgroup
control barriers. It retains NonPrivatePointer, barrier placement and scopes,
relaxed atomics, explicit storage decorations and fence-only propagation.
This includes barriers generated for workgroupUniformLoad and initialization.
The ordering argument relies on valid cross-invocation non-atomic communication
being synchronized with WorkgroupMemory semantics; it does not admit data races
or change the WGSL memory contract. Pointer helpers using workgroup globals
are covered; this validator rejects workgroup-pointer function parameters.

Any subgroup control or memory barrier, including nested or unused functions
and entries, disables the transformation for the entire module. Reusing a
Writer recomputes this decision. A separate pinned-compiler reproduction
confirms that subgroup-only barriers already fail Vulkan SPIR-V validation with
`VUID-StandaloneSpirv-MemorySemantics-10870`: the emitter selects SubgroupMemory
without a Vulkan-supported storage-class semantics bit. The test explicitly
preserves this rejection in fallback cases. No valid subgroup GPU execution
or repair of that separate issue is claimed.

Five focused Naga tests pass, checking operands, scopes, decorated accesses,
relaxed atomics, writer reuse and the fallback. All eight whole-graph smoke
tests and the publication/reuse stress test pass. Four initial test-development
failures are preserved: an unsupported subgroup enable directive, an invalid
workgroup-pointer parameter, the existing subgroup validation error, and the
test's initial SPIR-V 1.0 target. The final fixture uses supported syntax and
SPIR-V 1.3, and checks the existing subgroup rejection explicitly.

The normal comparison uses 20 AB/BA pairs per case for both fresh control and
candidate binaries: **200 comparisons** across three CI cases and both model
phases. All preserve selection, state, completion, tolerance and one-dispatch
gates. Ordinary and persistent paths use the same compiler in each binary.

| Model case | Control ordinary / persistent ms | Candidate ordinary / persistent ms | Candidate / control paired ratio, 95% interval |
|---|---:|---:|---:|
| Forward | 3.921 / 5.652 | 3.912 / 5.405 | 0.998 [0.971, 1.021] |
| Training + SGD | 14.860 / 18.863 | 14.842 / 19.059 | 1.005 [0.908, 1.142] |

The last column compares median paired persistent/ordinary ratios. Although
the raw forward persistent median falls, the normalized result establishes no
gain. Training has large outliers in both binaries, reaching 78–86 ms, which
are retained without filtering. Its raw persistent candidate/control interval
is especially wide, [0.991, 2.355]. The CI normalized ratios are 0.980, 0.996
and 1.016; all intervals include one. These data do not establish equivalence
or a small improvement, and the prototype fails the retention gate.

Four separate resource captures report unchanged forward/training resources:
100/103 registers, 20,880/29,232 shared bytes and 106,752/256,512 executable bytes.
All 43 dumped shader sources, including both whole-graph shaders, match between
control and candidate. Equal executable sizes do not prove identical native
instructions. No new SDK capture or full oracle qualification is claimed for
this rejected compiler prototype.

The temporary Cargo override and Naga implementation/tests are reverted after
preserving their patches and fixture. Both forks are clean, rebuilt examples
match the qualified binary hashes exactly, and all eight restored smoke tests
pass. Each heavy job uses a verified separate cgroup with 16/20/1 GiB limits;
GPU jobs serialize and artifacts remain on `/x`. Captures and raw samples are
under `target/workgroup-barrier-propagation/`. The manifest
`target/megakernel-results/workgroup-barrier-propagation/summary.json` has
**225 independently verified artifact hashes** and SHA-256
`1072455373873c0be6bf09093fe85a3617ad7703b7acb67776ce705d81210e7c`.
The qualified runtime, dependency pins and primary measurements remain unchanged.
Baseline parity is still unmet. The following probe supplies a scratch-ownership
argument and temporary subgroup-barrier lowering before measuring that direction.

## Subgroup-local cooperative row staging

At `5356830`, a temporary native cooperative matmul change keeps 64 logical
invocations, the original output tile, per-output accumulation order, K loop,
logical grid and four-tile packing. On 32-lane subgroups, each subgroup stages
its own A row tile and copies both B column tiles into disjoint shared slices.
Two staging passes cover the original 64 staging coordinates. The two K-loop
rendezvous use subgroup barriers. Other subgroup widths retain the original
staging and workgroup synchronization path. Prologue initialization, epilogue
scatter, graph publication/completion and scratch-reuse barriers remain.

The ownership audit finds a necessary adapter change: previously, inactive
packed groups masked writes and products but could still load a clamped active
scratch slice. The old workgroup barriers ordered those reads; subgroup barriers
would not. The probe also masks cooperative loads assigned to local matrices,
and rejects conversion if any cooperative load in such a handler is not
maskable. Generated model sources confirm all five loads in the changed handler
have activity guards. The two attention loads retain their original workgroup
synchronization. Partial packing and scratch reuse remain explicitly tested.

The temporary Naga patch maps subgroup barriers to AcquireRelease plus
WorkgroupMemory, retaining subgroup execution and memory scopes. Vulkan ignores
SubgroupMemory; it does not identify the shared storage class. See the
[Vulkan memory semantics](https://docs.vulkan.org/spec/latest/appendices/memorymodel.html#memory-semantics).
The cooperative operand checker also recognizes subgroup_size as workgroup-uniform:
[WGSL guarantees dispatch uniformity in compute](https://gpuweb.github.io/gpuweb/wgsl/#subgroups),
and Naga restricts cooperative loads/products to compute. Subgroup and invocation
IDs keep their previous classifications. No general relaxation of barrier
control validation or cross-backend repair is made.

Five focused fence tests and nine cooperative-matrix tests pass, including
SPIR-V validation, both operand scopes and negative divergent cases. All 48
shader-generation tests pass, retaining the four-tile packing assertion. All
eight expanded whole-graph smoke tests and publication/reuse stress pass. The
packed-matrix test covers plain and fused-prologue paths, short K tails,
single-tile claims, inactive groups, 1/3/64/4096 workers and three replays.

Development failures remain in the record: reserved names in the first WGSL
fixtures; missing subgroup-barrier capabilities in positive shader tests;
packing rejection from the conservative subgroup-size classification; a helper
fixture hitting existing workgroup control-flow rejection; and two fixture
variants retaining an unsupported 64×64 subgroup matrix type. The final operand
test uses valid shapes and retains the negative control tests. The first artifact
audit also mistakenly expected guards on the unchanged attention loads; the
corrected audit checks every load in the subgroup-synchronized handler.

Fresh control and candidate binaries each run 20 AB/BA pairs for all three CI
cases and both model phases: **200 normal comparisons**. All retain matching
selection, state/completion/tolerance checks, one persistent dispatch and no
fallback. Both execution modes share the candidate compiler and arithmetic.

| Model case | Control ordinary / persistent ms | Candidate ordinary / persistent ms | Candidate / control persistent median, 95% interval |
|---|---:|---:|---:|
| Forward | 4.242 / 5.703 | 4.613 / 6.028 | 1.057 [1.051, 1.116] |
| Training + SGD | 14.944 / 18.708 | 13.970 / 19.662 | 1.051 [0.997, 1.150] |

Forward's normalized paired ratio is 0.967 [0.909, 1.011], but ordinary execution
slows more, so this is not a persistent speedup. Training's normalized ratio is
1.068 [0.978, 1.920], with substantial timing variation retained. CI normalized
ratios are 1.018, 1.001 and 1.017; their intervals all include one. The probe
establishes no useful completed-step improvement and is rejected.

Four separate resource captures report forward/training shared storage growing
from 20,880/29,232 to 24,976/33,328 bytes. Registers change from 100/103 to 100/102;
executable sizes change from 106,752/256,512 to 107,776/254,848 bytes. Extra staging
and storage are potential costs, not an isolated causal attribution. No new SDK
capture or full oracle qualification is claimed for this rejected prototype.

All temporary runtime, shader, test, Cargo and Naga changes are reverted.
Rebuilt examples match the qualified hashes byte-for-byte, both forks are clean,
and all eight restored smoke tests pass. Heavy jobs use verified separate
16/20/1 GiB cgroups, GPU jobs serialize, and all artifacts stay on local `/x`.
Sources, patches, samples and failures are under `target/subgroup-row-staging/`.
The manifest `target/megakernel-results/subgroup-row-staging/summary.json` contains
**282 independently verified artifact hashes**, with SHA-256
`48d1e6d3edb41ca67d2d3539b91e3121b66148e3fd32b9388745910d498f022b`.
The qualified runtime and dependency pins remain unchanged. Next, audit which
real model dependencies permit safe release of row or element ranges; the prior
independent-node diagnostics motivate measuring that opportunity before changing
the scheduler. Baseline parity remains unmet.

## Real model dependency and row-range audit

At `c47b40a`, temporary diagnostics under the existing shader-dump option record
the selected dispatches, physical allocations, logical extents, binding offsets,
packing, parameters and dependency metadata. Scheduler code, shaders, arithmetic
and compiler pins are unchanged. Both 256-worker diagnostic comparisons pass
state, selection and completion checks with one compute dispatch and one submit.
An independent pairwise hazard reconstruction matches the captured transitive
closure, reduced edges and initial predecessor counts: 154 nodes / 161 edges
for forward and 527 nodes / 706 edges for forward/backward/SGD.

Forward has no sliced allocations, so replacing whole-allocation conflicts with
logical intervals provides no additional ordering freedom. Training packs
parameters and gradients into arenas. A hypothetical interval analysis removes
6,597 ordered node pairs and changes the reduced graph from 706 to 698 edges;
the longest unweighted path changes from 349 to 336 nodes. This uses the largest
logical extent at each physical offset, rounded to 16 bytes, and keeps appended
bindings conservative. It is an opportunity estimate, not a proof that every
selected kernel stays in those intervals or a performance prediction. The four
SGD chunks already have independent readiness: their immediate predecessors are
`[522]`, `[494,495]`, `[406,407]` and `[164,165]`. They are not chained together by
the optimizer's shared read-only segment table.

The narrower row-release candidate is the 16 repeated feed-forward chains:
`[50,720] × [720,4096] → SwiGLU [50,2048] → matrix/add [50,720]`.
Each chain has three selected dispatches and two compatible row bands, rows
0–31 and 32–49. The cooperative output includes padding through row 63.
Separate ordinary dispatch profiles locate the work:

| Graph | Eligible dispatches | Their median interval sum | All dispatch median sum | Share |
|---|---:|---:|---:|---:|
| Forward | 48 / 154 | 2.430144 ms | 4.003744 ms | 60.7% |
| Forward/backward, optimizer disabled | 48 / 523 | 2.108224 ms | 12.482208 ms | 16.9% |

Each profile contains five samples per dispatch, with exact state unchanged by
profiling. Attributions match the captured labels, grids and logical outputs.
The profiled/unprofiled wall ratios are 1.202 and 1.276: these are instrumented
ordinary intervals, not persistent per-node timings or projected speedups.
The initial profile attempt fails with `MissingGpuTimings`; enabling the existing
`MEGANEURA_GPU_TIMING` setting fixes the setup. Both logs are retained.

The static coordinate audit preserves the original grids and packing. The
cooperative matrix's two absolute ticket intervals are `[0,128)` and `[128,256)`,
each with 32 four-tile claims. SwiGLU uses `[0,256)` and `[256,400)`, with 256 and
144 claims. The final scalar matrix uses `[0,45)` and `[45,90)`, with 45 claims
each. All original coordinates occur exactly once, boundaries align with claims,
and same-operation partitions have no overlapping writes or read/write hazards.
Splitting these operations adds 48 scheduler nodes without adding payload tiles.
Only two bands are available; the earlier synthetic eight-chain result does not
establish a benefit for this model.

Two physical hazards prevent a simple row-for-row dependency rule:

- Every inference chain reuses the cooperative output allocation for the final
  matrix output, whose narrower rows overlap the first producer band. Both final
  bands must respect those WAW and SwiGLU-read WAR dependencies. The captured
  training chains use separate allocations at this boundary.
- Scalar matrix staging evaluates an element-zero load for masked rows or K
  lanes before selecting zero. The second final-matrix band therefore also reads
  the first SwiGLU band, even though that value is discarded. Its footprint must
  include that load; matching only the mathematical output rows is insufficient.

The next prototype should preserve these ranges, padded stores and all unrelated
physical hazards, rebuilding the full hazard relation before reducing edges.
Removing only old direct edges is unsafe: whole-node chains previously carried
transitive reuse ordering that finer completion can break. Completion and worker
counts must aggregate back to the original dispatches. Compare earlier release
against a control with identical partitioning, shader and whole-node barriers,
then apply the normal state/selection/latency gates. No partitioned runtime or
new SDK performance result is claimed by this audit.

All temporary diagnostics are reverted. Rebuilt examples match the qualified
hashes byte-for-byte, both forks are clean, and eight restored whole-graph smoke
tests pass. Builds, GPU jobs, copies and analyses use verified separate cgroups
with 16/20/1 GiB limits; GPU work is serialized and artifacts remain on local
`/x`. Captures, source snapshots, diagnostic binary, scripts, profiles and the
failed setup are under `target/dependency-range-audit/`. The manifest is
`target/megakernel-results/dependency-range-audit/summary.json`, with **86
independently verified artifact hashes** and SHA-256
`5b1e3bf5c20f79f260ef4044a073cdc01733218d863f5ff3682e36be5bb47391`.
The qualified runtime and primary performance results are unchanged. Baseline
parity remains unmet.

## Two-band release of feed-forward chains

At `e8ffb25`, a temporary host-side prototype splits each of the 16 audited
matrix/SwiGLU/matrix chains into two row bands. It changes no worker or payload
shader. Each part retains the original grid, parameter block and absolute ticket
coordinates; its initial claim counter starts at the band's first tile and its
remaining counter counts only that band. Completion checks use the absolute end,
while worker totals aggregate the parts. Cached parameter uploads and appended
optimizer records keep their original operation indexing.

The prototype rebuilds all physical hazards before reducing edges. Only accesses
inside an admitted chain use the proved row footprints; other relationships stay
whole-allocation. Cooperative padding, inference output reuse and the scalar
element-zero read remain in those footprints. A separate control has identical
parts and payloads but keeps whole-operation dependency edges between them.
The two source versions differ only in the host dependency-selection constant.

| Model graph | Expanded nodes | Whole-operation edges | Early-release edges |
|---|---:|---:|---:|
| Forward | 202 | 289 | 241 |
| Forward/backward/SGD | 575 | 834 | 786 |

Both modes execute the original 17,242 / 177,346 payload tiles. Independent
audits match every coordinate, parameter word, descriptor allocation, initial
counter and ready bit, full hazard closure and reduced edge set. Both contain
all 16 partitioned chains. Their 43 generated shader files are identical to each
other, and the model pipeline identities match the qualified runtime.

All eight expanded whole-graph smoke tests pass. The packed-matrix fixture adds
complete feed-forward chains at 33/50/53/64 rows with 4096 intermediate columns,
and 53 rows with 4224 columns. Each runs three exact-output replays at
1/3/64/4096 workers. Twenty captured partitioned graphs confirm that the intended
cases actually enter the new path. Existing tests also retain optimizer,
allocation-reuse, scratch, geometry and external-recording coverage.

The normal benchmark runs qualified / whole-operation / early / early /
whole-operation / qualified, with ten interleaved ordinary/persistent pairs per
case per visit: **300 comparisons** across all five cases. Selection, state
budgets and completion pass throughout, with one persistent compute dispatch and
no fallback. Maximum training error remains 4.66e-9.

| Graph | Qualified ordinary / persistent ms | Partitioned whole-operation ordinary / persistent ms | Early-release ordinary / persistent ms |
|---|---:|---:|---:|
| Forward | 3.901 / 5.381 | 3.993 / 5.400 | 4.266 / 5.691 |
| Forward/backward/SGD | 14.828 / 18.921 | 14.791 / 18.816 | 14.812 / 18.873 |

Forward early/whole persistent latency is **1.054 [0.992, 1.065]**; normalized
by each run's ordinary timing, **0.981 [0.918, 1.029]**. Against the qualified
runtime, raw latency is **1.058 [1.004, 1.080]**, while the normalized comparison
is inconclusive at **0.963 [0.907, 1.015]**. Training early/whole is
**1.003 [0.927, 1.045]** and early/qualified **0.998 [0.924, 1.051]**;
both normalized intervals also include no change. All CI comparisons are
inconclusive. These 10,000-resample intervals do not capture every source of
between-run variation: qualified forward medians change from 5.268 to 5.596 ms,
and the partitioned control changes from 5.733 to 5.369 ms. No useful gain is
established, so neither partitioning nor earlier release is retained.

Four separate native resource records confirm equal executable statistics in both modes:
forward/training use 100/103 registers, 20,880/29,232 shared bytes and
106,752/256,512 code bytes. With worker histograms enabled, early-release
scheduler storage is 438,364 / 1,272,128 bytes; the whole-operation control needs
192 more bytes for its extra edges. Initial diagnostics did not enable native
capture and contain no executable statistics; the separate capture-enabled runs
supply this check. The first resource audit incorrectly compared hash-map-derived
pipeline lists positionally. Comparing multisets, preserving duplicates, passes.
That failed audit and the initial Rust import-path build failure remain in the
record. No full oracle qualification or new SDK counter capture is claimed.

All temporary runtime and test changes are reverted. Rebuilt examples match the
qualified hashes byte-for-byte, both forks remain clean, and eight restored
whole-graph smoke tests pass. Every heavy job uses a verified separate 16/20/1 GiB
cgroup; GPU jobs serialize, and artifacts stay on local `/x`. Sources, binaries,
scripts, samples and failures are under `target/row-range-release/`. The manifest
`target/megakernel-results/row-range-release/summary.json` has **503 independently
verified artifact hashes** and SHA-256
`0e3943af3d131346c18fe4829ff89fbb681efa01de402af903e7e8c09f675c5b`.

Two-band release does not reproduce the earlier synthetic wide-graph benefit.
Next, test proven disjoint gradient-arena writes without increasing node counts
or changing whole-operation completion. That is a separate training hypothesis;
the qualified runtime and primary results remain unchanged, and baseline parity
is still unmet.

## Bounded gradient-write dependencies

The retained change narrows only proved write-only outputs when constructing
physical dependency edges. Scalar float32 `MatMulAT` outputs cover `M*N*4` bytes,
including each output of a horizontally packed dispatch. `SumRows` covers `N*4`
bytes for its serial layout and `N*workgroups.y*4` for partial-row outputs. These
templates guard their stores and never read the destination. Cooperative and
split-K matrix kernels, fused prologues/epilogues and reduced-storage outputs are
excluded. Checked arithmetic and allocation bounds fall back to the whole
allocation when the extent cannot be proved.

All reads, unrecognized writes and optimizer bindings remain whole-allocation.
Consequently only disjoint WAW relationships disappear; physical RAW/WAR and
overlapping WAW hazards retain their ordering. The planner reconstructs the full
hazard relation before reducing edges. It changes no node count, original tile,
parameter block, arithmetic, descriptor identity, shader, fence or completion
protocol. Parameter sharing invalidates cached metadata and recomputes ranges
from the rebound buffer pieces. Imported allocations remain rejected.

Independent audits reconstruct exact output bounds, full hazard closure and its
transitive reduction, initial claims, remaining counts, predecessor counts and
ready bits. The CI training graph removes two ordered pairs: bias-gradient
`SumRows` nodes 12 and 16 no longer precede independent weight-gradient
`MatMulAT` nodes 14 and 17. Removing transitive ordering can increase the number
of direct edges; edge count alone is not the scheduling opportunity.

| Graph | Nodes | Whole-allocation edges | Bounded-write edges | Ordered pairs removed |
|---|---:|---:|---:|---:|
| CI inference | 5 | 4 | 4 | 0 |
| CI backward | 18 | 19 | 20 | 2 |
| CI backward/SGD | 19 | 20 | 22 | 2 |
| Model forward | 154 | 161 | 161 | 0 |
| Model backward/SGD | 527 | 706 | 698 | 6,597 |

The model refines 99 matrix outputs and 37 row-sum outputs; CI refines three
matrix outputs and two row sums. No forward access changes. All 19 CI and 43
model generated shader files match their controls. Executable resource multisets
also match, retaining 100/103 registers, 20,880/29,232 shared bytes and
106,752/256,512 code bytes for model forward/training.

An initial 200-comparison screen suggested a CI benefit but no model gain. To
separate the dependency change from binary-layout effects, a temporary config
switch then selects whole-allocation or bounded-write hazards in one executable.
Whole / bounded / bounded / whole visits each contain 25 interleaved
ordinary/persistent pairs per case: **500 comparisons**, or 50 pairs per policy
and case. CI backward and SGD persistent latency improve **4.0% and 4.4%**.
Their bounded/whole ratios are 0.960 [0.948, 0.963] and 0.956 [0.943, 0.969];
normalizing by ordinary timing gives 0.947 [0.931, 0.963] and
0.941 [0.928, 0.955]. Model forward/training show no clear change. The switch,
diagnostic metadata dumps and temporary public configuration field are removed.

The same-binary inference control has identical graph metadata, yet its raw
latency ratio is 1.046 [1.011, 1.076]; normalization is inconclusive at
1.036 [0.936, 1.121]. This variation is preserved rather than attributed to a
dependency change that inference does not contain.

Fresh final binaries then run against saved `762faf2` binaries in the same
mirrored order, again with **500 comparisons** and 50 pairs per implementation
and case. Every comparison passes selected-implementation matching, full state
budgets and completion checks, with one persistent dispatch and no fallback.
Maximum model training error remains 4.66e-9.

| Case | Previous ordinary / persistent ms | Final ordinary / persistent ms | Final median paired persistent / ordinary |
|---|---:|---:|---:|
| CI inference | 0.05242 / 0.06786 | 0.05173 / 0.06963 | 1.371 |
| CI backward | 0.06630 / 0.10196 | 0.06635 / 0.09627 | 1.443 |
| CI backward/SGD | 0.06763 / 0.10226 | 0.06835 / 0.09675 | 1.420 |
| Model forward | 4.292 / 5.726 | 4.324 / 5.732 | 1.319 |
| Model backward/SGD | 14.736 / 18.894 | 14.749 / 18.887 | 1.358 |

Final CI backward and SGD persistent latency improve **5.6% and 5.4%**:
ratios 0.944 [0.928, 0.957] and 0.946 [0.940, 0.960]. The normalized ratios are
0.934 [0.923, 0.957] and 0.942 [0.928, 0.951]. Inference is inconclusive at
1.026 [0.976, 1.046]. Model forward/training remain unchanged within the
measurement intervals: 1.001 [0.987, 1.012] and 1.000 [0.985, 1.017].
Intervals use 10,000 bootstrap resamples and do not cover all between-run
variation. Paired ratios are medians of individual sample ratios, not ratios of
the aggregate medians; ordinary model-training samples have substantial spread.

Qualification passes 666 library tests (eight existing ignores), 168 strict
whole-graph oracle tests, 50 matrix regressions in their original mode and nine
whole-graph smoke tests, plus formatting and strict all-target/all-feature Clippy.
The added fixture compares exact parameters, gradients and optimizer state for
three gradient branches at 1/3/64/4096 workers, including SGD, Adam, clipping,
poisoned storage and repeated external records. A second part redirects two
matrix outputs and then shares their parameter allocation, checking overlapping
WAW/RAW ordering and cached replays against exact CPU values. The first version
of that fixture kept independent reads scheduled before its redirected writes;
the failed run is preserved, and the fixture now explicitly orders writes first.

An additional strict whole-graph run of all 50 matrix regressions passes 43 and
rejects seven at admission for unsupported scalar types or cooperative physical
workgroup sizes. All seven failures reproduce on the preceding qualified runtime.
No admission check or numerical tolerance is relaxed. These cases limit coverage;
the passing original-mode suite does not qualify them for whole-graph execution.

This investigation retains **1,200 normal comparisons** and **19 separate native
resource records**. No new Perf SDK counter capture is claimed. All heavy jobs
run in verified separate cgroups with 16/20/1 GiB limits, GPU jobs serialize, and
no build or heavy analysis overlaps normal measurement. Artifacts stay on local
`/x` under `target/gradient-write-ranges/`. The manifest
`target/megakernel-results/gradient-write-ranges/summary.json` contains **979
independently verified artifact hashes**, with SHA-256
`ad81207646b3ecd9b5fbf1c5e5acf68486a40106b4a657047f914376ffff7093`.
The measured final CI/model binaries have SHA-256
`c7529ae3eb8a374ec1d7a8086ff8aac7313a43c75c0f62750661c0596e19355c` and
`2ca00e91956a3eb426493746e7b9e8660321de3aab41156a71ebfb5d816266c7`.
Both dependency forks remain unchanged and clean. This is a CI training gain;
ordinary-baseline parity and the model performance gap remain open.

## Scalar matrix vector staging

Four-float input staging tests whether scalar matrix loads contribute useful
optimization headroom. The prototype reads aligned `vec4<f32>` values and
scatters their unchanged components into the existing padded shared arrays.
Accumulator arithmetic, reduction order, output stores, grids, scratch capacity,
dependencies and fences remain unchanged. Ordinary and persistent execution use
the same implementation selection. Eligibility excludes horizontal packs,
cooperative/split kernels, reduced storage, prologues/epilogues and locked
schedules; alignment checks admit only complete in-bounds vectors. Masked reads
still access element zero, covered by the unchanged whole-allocation read edges.

Three initial variants stage both A and B, only B, or only A. Each changes 50
forward and 200 training nodes. An independent indexing audit checks exact
shared-cell coverage and vector alignment at first/last row, column and K-stage
boundaries for 22 actual model shapes: **93,184 vector checks**. Each variant
passes a focused exact-reference fixture covering normal, A-transposed and
B-transposed products with partial tiles, two input seeds and ordinary execution
plus 1/3/64/4096 workers. A fourth fixture uses reductions of 516/517 elements
for the later long-K variant.

Each initial variant runs 200 normal comparisons in mirrored control / candidate
/ candidate / control visits, with ten interleaved ordinary/persistent pairs per
case per visit. These screens do not establish a useful model improvement:

| Staged input | Forward persistent candidate/control, 95% interval | Training persistent candidate/control, 95% interval |
|---|---:|---:|
| A+B | 0.972 [0.952, 1.054] | 0.991 [0.981, 1.086] |
| B only | 1.085 [1.071, 1.132] | 0.985 [0.915, 1.021] |
| A only | 0.970 [0.955, 1.067] | 0.988 [0.911, 1.040] |

The A-only forward signal warrants a same-binary control. A temporary benchmark
flag reconstructs the same selected plan with schedules locked to suppress only
the vector-staging wrapper. Both modes perform the same untimed reconstruction
and assert unchanged dispatches and buffer sizes. Audits reproduce the qualified
control's selected keys, executable resources, scheduler bytes and graph shader
hashes. Another **500 comparisons** use mirrored visits with 25 pairs per case,
giving 50 pairs per mode and case.

| Case | Scalar / vector persistent ms | Vector/scalar, 95% interval | Ratio normalized by paired ordinary timing, 95% interval |
|---|---:|---:|---:|
| CI inference | 0.06677 / 0.07241 | 1.084 [1.059, 1.111] | 0.965 [0.936, 0.990] |
| CI backward | 0.09615 / 0.09734 | 1.012 [0.991, 1.025] | 0.982 [0.960, 0.997] |
| CI backward/SGD | 0.09691 / 0.09742 | 1.005 [0.970, 1.030] | 0.989 [0.969, 1.004] |
| Model forward | 6.003 / 5.717 | 0.952 [0.942, 0.999] | 0.983 [0.967, 0.998] |
| Model backward/SGD | 18.799 / 18.739 | 0.997 [0.980, 1.013] | 0.972 [0.904, 1.076] |

The **4.8% forward gain** comes with an **8.4% CI inference regression**.
Ordinary CI inference also slows by 13.5%, so its favorable normalized ratio
does not justify retaining the change. Model training shows no clear benefit.
A-only staging keeps model shared memory unchanged and uses 100/101 registers
and 107,264/255,488 code bytes, versus the control's 100/103 registers and
106,752/256,512 code bytes. CI inference code grows from 17,536 to 26,240 bytes;
backward and SGD code also grow. These are native resource statistics, not SASS
instruction counts or an attribution of elapsed time.

A narrower A-only variant requires **K >= 512**. It changes 49 forward and 113
training nodes; the excluded forward node has K=32. CI selected keys, shader
files and executable resources now match the scalar mode exactly. Mixing scalar
and vector handlers increases model code to 116,224/290,560 bytes, with 100/100
registers and unchanged shared capacity. The same focused correctness checks
pass, followed by another **500 same-binary comparisons**:

| Case | Scalar / vector persistent ms | Vector/scalar, 95% interval | Ratio normalized by paired ordinary timing, 95% interval |
|---|---:|---:|---:|
| CI inference | 0.07596 / 0.07187 | 0.946 [0.930, 0.982] | 1.011 [0.977, 1.031] |
| CI backward | 0.09662 / 0.09802 | 1.014 [0.994, 1.032] | 1.009 [0.993, 1.042] |
| CI backward/SGD | 0.09681 / 0.09851 | 1.018 [0.987, 1.031] | 1.012 [0.990, 1.038] |
| Model forward | 5.972 / 5.783 | 0.968 [0.958, 1.012] | 0.973 [0.956, 0.995] |
| Model backward/SGD | 18.794 / 18.692 | 0.995 [0.989, 0.999] | 0.948 [0.898, 1.043] |

Identical CI inference shaders still show raw timing movement, while their
normalized interval includes no change. The model forward raw interval also
includes no change. Training improves only about 0.5%, and its normalized
interval is inconclusive. Ordinary model-training timing has substantial spread.
All intervals use 10,000 bootstrap resamples; they do not cover every source of
between-visit variation or establish performance on additional graphs.

All **1,600 normal comparisons** pass matching selection within each backend
pair, full state budgets and worker completion, with one persistent dispatch and
no fallback. Maximum model error remains 4.66e-9. There are **32 separate native
resource records** and no new Perf SDK captures. The rejected prototypes do not
receive full oracle/library/matrix qualification. The initial audit-script
failure on an omitted optional JSON field is preserved alongside its corrected
run; no GPU correctness failure is hidden.

All prototype code, the focused fixture and temporary benchmark switch are
removed. The rebuilt CI/model binaries exactly reproduce the qualified
`3cb7d57` hashes recorded above; formatting and all nine whole-graph smoke checks
pass. Both dependency forks remain unchanged. Every heavy job uses a verified
separate cgroup with 16/20/1 GiB memory limits; GPU jobs serialize and no build or
heavy analysis overlaps measurements. Local artifacts are under
`target/scalar-vector-staging{,-a-only,-b-only,-a-long}/`, with the manifest at
`target/megakernel-results/scalar-vector-staging/summary.json`. It contains
**1,405 independently verified artifact hashes**, with SHA-256
`4b5381e8ea6a78d60a445b8f5e73c8da0f0ad060743a0fd3a541d7ad78ca40a8`.

Vector staging remains evidence of possible forward headroom, not a retained
optimization. Further work must distinguish payload instruction cost from
combined-handler code/resource effects while keeping all CI and training gates.
The qualified runtime and baseline performance gap remain unchanged.

## Subgroup-uniform scalar output guards

Runtime `ac8ce60fbb2bd195358fc9b45eb555a1996d4ae1` skips a scalar matrix
subgroup's K reduction only when none of its threads owns a valid output.
`subgroupAny` votes on each thread's first output row and column, accounting for
interleaved columns. A thread's other outputs have greater coordinates, so a
false per-thread predicate implies no valid store. The vote can only enlarge
that active set: partial subgroups keep their original arithmetic. Every thread
still stages operands and reaches both workgroup barriers. K order, padding
arithmetic for active subgroups, output stores, tile geometry, dependencies,
descriptors, fences and whole-step boundaries remain unchanged. Both backends
use the same selected matrix template.

An independent geometry audit checks 1,009,152 thread/tile ownership cases over
square and rectangular tiles, both column layouts and boundary dimensions. It
finds wholly inactive warps account for 18.75% of forward and 16.3% of training
scalar warp K iterations in the audited model geometry. These are potential
reductions, not measured native instructions or a predicted elapsed-time gain.

The initial per-thread guard passes nine smoke tests, eight strict matrix
oracle tests and **500 paired comparisons**, but slows persistent model forward
by 8.0% and training by 19.2%. The candidate/control intervals are
1.080 [1.055, 1.088] and 1.192 [1.050, 1.279]. SDK captures confirm that persistent
FP32 thread activity falls 20.6/13.7%, yet resident replay grows from
5.246/16.912 ms to 5.471/18.988 ms. Persistent local-memory traffic stays zero.
This rejects padded arithmetic count alone as a sufficient explanation of the
performance gap. The per-thread source, binaries, timings and counters remain
preserved; this form is not retained.

Replacing its predicate with a subgroup vote avoids that regression. Another
**500 comparisons** find 3.9% lower persistent forward latency and 3.4/3.1%
lower CI backward/SGD latency; training-model improvement is inconclusive.
The final implementation then adds capability selection. Blade graphics
[`fde3a9f`](https://github.com/kvark/blade/commit/fde3a9fe609af7f31d50c53fc0e1469fc0e17cfa)
reports compute-stage subgroup voting from Vulkan's stage and `VOTE` property
bits. Subgroup width alone is not sufficient, as specified by
[Vulkan's subgroup properties](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceSubgroupProperties.html).
Metal/GLES currently do not report this capability and retain the
original shader. Codegen defaults to that fallback. Blade macros remain pinned
at `c09c89f`; Naga remains unchanged at `bc7b70b`. No local dependency override,
new Naga syntax or memory-semantics change remains.

Qualification passes **667 library tests** (eight existing ignores), **168
strict whole-graph oracle tests**, **50 matrix regressions in their original
mode**, **nine whole-graph smoke tests**, formatting, strict all-target/all-feature
Clippy and Metal-target compilation. The new shader test covers normal,
transposed and fused-add forms, four tile geometries, both column layouts and
validation with/without subgroup capability. Blade passes 21 library tests
(one existing ignore), plus Metal and WebAssembly compilation. GPU execution
qualification remains RTX 5080 Vulkan; the original-mode matrix run does not
qualify previously rejected strict whole-graph matrix configurations.

All 19 CI and 43 model shader bodies match the successful subgroup prototype
after ignoring comments/whitespace. Its five complete graph shaders match
byte-for-byte; executable resource multisets, selected keys and scheduler bytes
also match. Model forward/training use 102/103 registers, 20,880/29,232 shared
bytes and 107,904/263,296 code bytes, versus the old 100/103 registers and
106,752/256,512 code bytes. CI retains 48 registers and adds 16 reported shared
bytes as the existing subgroup geometry checks become active.

Fresh final binaries run another **500 comparisons** against saved qualified
`3cb7d57` binaries: control / final / final / control, with 25 interleaved
ordinary/persistent pairs per case and visit. No performance override remains.

| Case | Final ordinary / persistent ms | Final median paired persistent / ordinary | Persistent final/control, bootstrap 95% interval |
|---|---:|---:|---:|
| CI inference | 0.05207 / 0.06912 | 1.329 | 0.981 [0.956, 0.990] |
| CI backward | 0.06551 / 0.09345 | 1.420 | 0.956 [0.946, 0.973] |
| CI backward/SGD | 0.06684 / 0.09417 | 1.406 | 0.970 [0.950, 0.978] |
| Model forward | 4.198 / 5.596 | 1.332 | 0.980 [0.974, 1.015] |
| Model backward/SGD | 14.754 / 18.749 | 1.282 | 0.999 [0.943, 1.010] |

CI backward/SGD improve **4.4% / 3.0%**. Normalizing by paired ordinary timing
also shows gains: 0.970 [0.959, 0.986] and 0.985 [0.970, 0.998]. The inference
normalized interval is inconclusive. Neither model phase establishes a
completed-step gain: normalized ratios are 0.979 [0.958, 1.003] forward and
1.011 [0.948, 1.047] training. Intervals use 10,000 bootstrap resamples and do
not cover every source of between-visit variation. Paired ratios are medians
of individual ratios, not ratios of aggregate medians.

Fresh final SDK captures collect four sequential metric groups per resident
replay, with matched selections and exact same-backend state against the saved
runtime. Training includes SGD at zero learning rate for stable replay. Group
clock and instruction-rate means vary by less than 1% within each capture.

| Phase | Ordinary control / final replay ms | Persistent control / final replay ms | Persistent warp instructions, billions, control / final | Persistent FP32 thread instructions, billions, control / final |
|---|---:|---:|---:|---:|
| Forward | 3.852 / 3.744 | 5.253 / 5.051 | 0.686 / 0.650 | 3.597 / 2.962 |
| Training + SGD | 12.005 / 11.990 | 16.892 / 16.266 | 2.807 / 2.739 | 18.975 / 16.937 |

Persistent resident replay improves **3.8% / 3.7%**. Estimated FP32 thread
activity falls 17.6/10.7%, and shared wavefronts fall 12.6/9.4%. Persistent local
loads/stores remain zero. Ordinary execution now has small local traffic:
about 79,780 load/store sectors forward and 323,180 training. These counters do
not identify its source, and no SASS attribution is claimed. Barrier waiting
increases from 57.07/50.09% to 59.19/50.87% of active warp cycles despite the
faster replay. Those percentages are not elapsed-time fractions. Resident replay
does not replace the completed-step model gate above.

All **1,500 normal comparisons** preserve matching selections, full state
budgets and completion, with one persistent dispatch and no fallback. Maximum
model error remains 4.66e-9. The record contains **30 native resource records**
and **16 SDK backend captures / 64 metric groups**, including the rejected
per-thread variant. The initial source-audit filename assumption failed because
shader filenames hash their contents; its log is preserved with the corrected
body comparison. No numerical tolerance or validation rule is relaxed.

Heavy jobs run in verified separate cgroups with 16/20/1 GiB limits; GPU jobs
serialize and no build or heavy analysis overlaps measurements. All artifacts
remain on local `/x` under `target/scalar-output-guard{,-subgroup,-qualified}/`.
The manifest `target/megakernel-results/scalar-output-guard/summary.json` has
**1,269 independently verified artifact hashes**, SHA-256
`7064d00abe13aee7aafa88b9d2516300a64046b690ba416b502766c0a0b0d483`.
Its frozen source patch matches the retained runtime commit. Measured final
CI/model binary SHA-256 values are
`8a0ecaf9ebc0c5ad6f4fedccd6957e2373d96773cd85e8db8be826d06c4175f6` and
`de9aea135071b6b0b8ef9f08bdd5db6aca253ccbc252073ee2b70e85b781bfcf`.
The retained change improves CI training and resident GPU replay; model
completed-step improvement and ordinary-baseline parity remain unproven.

## Interior staging for transposed scalar matrices

Runtime `540eccfd312b7e6ae372e434d36c3a10d145c684` adds a uniform direct-load
path for complete B stages in F32 transposed-B scalar products. These stages
are contiguous along K. The predicate proves that all tile columns and all K
positions are inside the matrix; the original per-element checked path handles
every edge stage. The loop already guarantees `t < end_k <= K`, and subtracting
the tile width from N avoids overflowing an upper-bound addition. Each direct
load uses the same original address and writes the same shared cell. A staging,
K order, zero padding, accumulator updates, stores, subgroup output guards,
barriers, descriptors, dependencies and completion remain unchanged. Both
backends use the same template. Other layouts and weight formats keep their
existing staging. No configuration hook or Naga/Blade change is added.

A coordinate audit checks **3,133,440 staged elements** over both B mappings,
legal tile/stage sizes, boundary dimensions and unsigned-32-bit limits. Every
admitted interior rectangle contains exactly the original coordinates. This
does not authorize allocations beyond the existing runtime limits or predict
native instruction counts.

The broad prototype enables the branch for every F32 scalar B layout. Its
500 paired comparisons pass, but persistent forward regresses **3.5%**
(candidate/control 1.035 [1.025, 1.046]); ordinary forward regresses 4.5%.
Training's raw ratio is 0.963 [0.897, 1.017], despite a favorable normalized
ratio. The forward regression rejects this form. Restricting the branch to
transposed-A products adds another 500 passing comparisons, but CI SGD
regresses **5.2%** and model training has no established gain. Its training
register count drops from 103 to 100, showing that this resource reduction
alone does not establish a speedup. Both prototypes remain rejected.

The transposed-B prototype instead improves persistent model training by
**5.7%** in 500 comparisons: 17.871 versus 18.944 ms, ratio
0.943 [0.937, 0.958]. Its normalized ratio is 0.950 [0.899, 0.963]. Both
transposed-only variants leave the complete forward shader byte-for-byte
unchanged; their differing forward timings are not evidence of a shader gain.

Final qualification passes **667 library tests** (eight existing ignores),
**168 strict whole-graph oracle tests**, **50 matrix regressions in their
original mode**, **nine whole-graph smoke tests**, formatting, strict
all-target/all-feature Clippy and Metal-target compilation. The existing
scalar edge fixture now also checks the transposed-B 64×64 kernel on a
35×65×1031 product with an addend, signed dyadic CPU results, changed inputs,
poisoned storage and ordinary/1/3/256-worker execution. Interior columns and K
stages use the direct path, while the final column tile and K tail exercise
the original path. Its first run failed a new geometry assertion expecting
32×32 SmallTile; the existing selection is 64×64 Default. Only that test
expectation was corrected, and the initial failure remains in the artifacts.
No tolerance, validation or admission rule is relaxed. These original-mode
matrix results do not qualify the previously rejected strict configurations.

All 19 CI and 43 model shader files match the successful transposed-B prototype
byte-for-byte. Pipeline resource multisets, selections and scheduler sizes also
match. Training retains **103 registers and 29,232 shared bytes**; code grows
from 263,296 to 267,776 bytes. Forward remains 102 registers, 20,880 shared
bytes and 107,904 code bytes. CI inference is unchanged; backward/SGD retain
48 registers and 9,600 shared bytes, with 61,312/65,408 code bytes.

Fresh final binaries run another **500 paired comparisons**, using
control/final/final/control visits and 25 interleaved ordinary/persistent pairs
per case and visit. The complete nonzero-rate training step remains the primary
boundary, with 256 workers, ten warmups and at least one second of warmup.

| Case | Final ordinary / persistent ms | Final median paired persistent / ordinary | Persistent final/control, bootstrap 95% interval |
|---|---:|---:|---:|
| CI inference | 0.05193 / 0.06961 | 1.342 | 1.015 [1.001, 1.066] |
| CI backward | 0.06494 / 0.09343 | 1.432 | 0.998 [0.986, 1.007] |
| CI backward/SGD | 0.06682 / 0.09455 | 1.413 | 1.005 [1.0003, 1.0095] |
| Model forward | 4.127 / 5.541 | 1.335 | 1.004 [0.988, 1.074] |
| Model backward/SGD | 14.613 / 17.892 | 1.225 | 0.944 [0.938, 0.966] |

Persistent model training falls from **18.953 to 17.892 ms**, a **5.6%** gain.
Normalizing by paired ordinary timing gives 0.971 [0.951, 0.991], a **2.9%**
improvement. Ordinary training's ratio is 0.984 [0.965, 1.002], so its apparent
gain remains inconclusive. CI SGD's raw median rises about **0.5% / 0.47 µs**;
its normalized interval, 0.999 [0.990, 1.008], is inconclusive. No CI improvement
is claimed. CI inference and model forward shader bytes are unchanged, so their
timing movements illustrate variability. All samples and outliers remain.
The 10,000-resample intervals describe these runs and do not cover every source
of between-visit variation. Paired-ratio medians need not equal ratios of time
medians. **Training remains about 22.5% slower than ordinary execution.**

Fresh NVIDIA SDK captures compare complete training plus zero-rate SGD replay,
with four sequential metric groups per backend and variant. All four backend
captures have matching adapters, successful exits, at least 100 post-warmup
samples per group and valid receipts. Actual state-file hashes match exactly
within each backend and retain the prior qualified state hashes. All complete
cross-backend state comparisons pass with maximum error 4.66e-9. Clock-rate
spread is at most 0.213%, and instruction-rate spread at most 1.066%, across
groups within a capture. No new forward counter claim is made.

| Complete-step SDK diagnostic | Control | Final |
|---|---:|---:|
| Ordinary resident replay ms | 11.9949 | 11.9427 |
| Persistent resident replay ms | 16.2214 | 15.8999 |
| Persistent warp instructions per replay, billions | 2.73978 | 2.72635 |
| Persistent integer-pipe thread instructions, billions | 6.57675 | 6.58486 |
| Persistent shared wavefronts, millions | 487.995 | 489.197 |
| Persistent barrier share of active warp cycles | 51.01% | 50.76% |
| Persistent local load/store sectors | zero | zero |

Persistent resident replay improves **2.0%**, less than the primary completed-step
gain. Aggregate integer and shared activity stays approximately flat; these
counters do not establish reduced integer instruction count as the cause.
Ordinary local load/store sectors rise from about 323,000 to 407,000 per replay.
No SASS or source-level spill attribution is available. Rate-times-duration
values estimate activity, barrier shares are not elapsed-time fractions, and
zero-rate resident replay does not replace the nonzero-rate primary gate.

The record preserves **2,000 normal comparisons, 40 native resource records,
four SDK backend captures / sixteen metric groups**, all variants, the fixture
failure and final qualification. Every comparison retains matching selected
arithmetic, full state/completion budgets, one persistent dispatch and no
fallback. All heavy jobs use verified separate cgroups with 16/20/1 GiB memory
limits; GPU jobs serialize and artifacts remain on `/x`.

Artifacts are under `target/scalar-interior-staging{,-at,-bt,-qualified}/`.
The manifest `target/megakernel-results/scalar-interior-staging/summary.json`
indexes **1,361 independently verified hashes**, SHA-256
`a2bb70df4e99c89c030e7359939c4e12668c7ffad212f79c128f7b1f71f87678`.
The frozen final patch matches the retained runtime commit. Final measured
CI/model binary SHA-256 values are
`b364fe6014d166ec297e37219d3b85cac1c99b267bd70b5f98348c23d2b5d8db` and
`3beabdb3af3e97ee825f25ae696fa25f4cb90d18914b1cc2e019be7739c4b1c7`.
The independent verification log is `target/interior-staging-verify.log`.
Baseline parity remains unmet; ordinary execution remains the default.

## Conditional scalar A staging

Two rejected prototypes on `6e8013f` test whether avoiding discarded A reads
helps the current scalar matrix path. The original checked load reads element
zero for an invalid row or K position, then selects zero for the shared cell.
Both prototypes instead read A only inside `if in_bounds`. Valid coordinates
use the identical original address; invalid coordinates still write positive
zero. Every staging cell is written before the unchanged barrier. All B formats
and normal/transposed/batched A mappings use this common template; arithmetic,
geometry, B staging, dependencies, descriptors and completion are unchanged.

The first form initializes a temporary float to zero, conditionally loads it,
then writes shared memory after the branch. The second puts the shared store
inside both branches, avoiding the temporary value across the merge. Neither
adds a barrier inside conditional control. Generated-source audits restore the
original masked expression and compare ordinary shader bodies after removing
only whitespace/comments; all other payload code matches.

Each prototype passes **40 code-generation checks, nine whole-graph smoke tests
and eight strict matrix oracle cases**. Each then runs control/candidate/
candidate/control visits with 25 interleaved ordinary/persistent pairs per case
and visit: **1,000 normal comparisons** total. All state, selection, completion,
one-dispatch and no-fallback gates pass, with maximum model error 4.66e-9.
The controls are immutable binaries of qualified runtime `540eccf`, including
its retained transposed-B interior staging. Worker count remains 256, with ten
warmup pairs and at least one second of warmup. No SDK sampling or build overlaps
normal measurements, and all samples and outliers remain.

| Form / model case | Control ordinary / persistent ms | Candidate ordinary / persistent ms | Persistent candidate/control, bootstrap 95% interval |
|---|---:|---:|---:|
| Merged value, forward | 4.114 / 5.553 | 4.346 / 5.744 | 1.034 [1.022, 1.101] |
| Merged value, training + SGD | 14.636 / 17.758 | 14.774 / 19.002 | 1.070 [1.004, 1.160] |
| Branch-local stores, forward | 4.134 / 5.530 | 4.347 / 6.748 | 1.220 [1.183, 1.232] |
| Branch-local stores, training + SGD | 14.648 / 17.747 | 14.827 / 20.409 | 1.150 [1.145, 1.240] |

The first form regresses persistent forward/training by **3.4% / 7.0%**. Its
normalized training ratio is 1.064 [1.035, 1.086]. Moving the stores into the
branches regresses forward/training by **22.0% / 15.0%**; normalized ratios are
1.146 [1.130, 1.169] and 1.131 [1.108, 1.158]. Thus removing the temporary value
does not provide a remedy. CI also fails to establish a useful overall gain:
the first form slows backward 3.2%, while the second slows SGD 2.4% with a
normalized ratio of 1.024 [1.012, 1.039]. These 10,000-resample intervals describe
within-run uncertainty, not every source of between-visit variation.

Twenty separate native resource records preserve **102/103 registers** and
**20,880/29,232 shared bytes** for forward/training in both forms. The control's
107,904/267,776 code bytes grow to 109,568/272,768 with the merged value and
111,232/278,144 with branch-local stores. Resource totals do not identify native
predication, spills or the cause of the elapsed-time regressions. No new SDK
capture or full library/oracle/matrix-regression qualification is claimed for
either rejected candidate.

Removing the discarded element-zero read changes the shader's read footprint,
but the current whole-allocation read dependencies remain conservative. This
does not qualify a narrower hazard analysis or a row-partitioned scheduler.
The earlier two-band result remains rejected; any future revisit would need
its own physical hazard proof, matched controls and completed-step gain large
enough to cover the staging cost. Fewer source-level reads alone are not a
sufficient performance argument.

Both prototypes are reverted exactly. Release examples are rebuilt, all nine
restored whole-graph smoke tests pass, and runtime/dependency sources match
`540eccf`. Saved lockfiles also match. The first manifest attempt incorrectly
assumed ignored `Cargo.lock` was tracked; its failed log remains beside the
corrected audit. GPU work stays serialized in verified separate 16/20/1 GiB
cgroups, and every artifact remains on local `/x`.

Sources, binaries, samples, proofs and captures are under
`target/scalar-conditional-a{,-store}/`. The manifest
`target/megakernel-results/scalar-conditional-a/summary.json` contains **635
independently verified artifact hashes**, SHA-256
`03b93d9d7daad5afaca53ed00c0f823a7acf9801c51d42bce4bcdb546e1d8fa6`.
The independent audit log is `target/conditional-a-verify.log`.
The retained runtime and its performance claims remain unchanged; baseline
parity is still unmet.

## Subgroup broadcast of the claim ticket

A temporary Blade backend probe on `7a8804e` tests whether explicitly exposing
the claim ticket's subgroup uniformity helps driver control flow. It inserts
one `OpGroupNonUniformBroadcastFirst` of the `uvec4` result immediately after
the single `graph_claim` call. The function already returns
`workgroupUniformLoad(&graph_ticket)`, so every active invocation has the same
value; selecting the first active value in each subgroup preserves all four
components, including the sentinel values. Original workgroup barriers remain.
The RTX 5080 reports compute-stage ballot support for this instruction. See the
[SPIR-V instruction specification](https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpGroupNonUniformBroadcastFirst).

The temporary postprocessor uses a separate DEBUG module to identify the call,
requiring that stripping its debug instructions reproduces the original
module. It validates the input, named and transformed modules with `spirv-val`
before pipeline creation. An independent inverse audit of **35 modules** removes
only the broadcast and any added capability/scope declaration, restores the
call result ID and bound, and reproduces every original byte. Memory operations,
scopes and semantics are unchanged. No Naga validator relaxation is involved.
This is a device-specific diagnostic, not a portable API change.

Nine whole-graph smoke tests pass. Control/candidate/candidate/control visits
then preserve **500 normal comparisons**, with 25 interleaved ordinary/
persistent pairs per case and visit, ten warmup pairs, at least one second of
warmup and 256 workers. Controls are the frozen qualified `540eccf` binaries.
Every selection, complete-state, completion, one-dispatch and no-fallback gate
passes; maximum model error is 4.66e-9. No build, heavy analysis or SDK capture
overlaps normal measurements.

| Case | Control ordinary / persistent ms | Candidate ordinary / persistent ms | Persistent candidate/control, bootstrap 95% interval |
|---|---:|---:|---:|
| CI inference | 0.05170 / 0.06587 | 0.05728 / 0.07466 | 1.133 [1.074, 1.145] |
| CI backward | 0.06453 / 0.09327 | 0.06513 / 0.09433 | 1.011 [1.001, 1.035] |
| CI SGD | 0.06630 / 0.09445 | 0.06661 / 0.10066 | 1.066 [1.049, 1.073] |
| Model forward | 4.120 / 5.517 | 4.183 / 5.591 | 1.013 [1.005, 1.024] |
| Model training + SGD | 14.586 / 17.819 | 14.631 / 17.855 | 1.002 [0.986, 1.023] |

There is no useful model-training gain. The normalized forward/training ratios
are 1.003 [0.976, 1.030] and 0.982 [0.951, 1.020], both inconclusive. CI SGD
regresses **6.6% raw / 4.2% normalized**, with normalized interval
[1.021, 1.064]. CI inference's ordinary timing also rises 10.8%; its normalized
interval includes no change. All samples and outliers remain. These
10,000-resample intervals describe within-run uncertainty, not every source of
between-visit variation.

Ten separate resource records retain 48 CI registers, 102/103 model registers
and all original shared-memory sizes. CI backward/SGD code grows from
61,312/65,408 to 61,696/67,584 bytes; model forward grows from 107,904 to 108,032
bytes. Training remains 267,776 bytes. All **19 CI and 43 model WGSL files** are
byte-identical between control and candidate, as are ordinary pipeline resource
multisets. These counts do not establish native control-flow uniformity or
explain the timing changes. No new SDK capture or full-suite qualification is
claimed for this rejected probe.

The backend patch and ignored lockfile are restored exactly. Both forks are
clean, release examples are rebuilt from the original pins, and all nine
restored whole-graph smoke tests pass. The recovered initial smoke log contains
its passing result and separate cgroup receipt; no stale GPU job remained and
the recovered run was not duplicated. Every heavy job uses a verified separate
16/20/1 GiB cgroup, GPU work is serialized, and all artifacts remain on `/x`.

Sources, binaries, module proofs, samples and captures are under
`target/ticket-subgroup-hint/`. The manifest
`target/megakernel-results/ticket-subgroup-hint/summary.json` contains **435
independently verified artifact hashes**, SHA-256
`01a671f874b16eb8934d954ab9568fb02a3ffab667a4bdf37ddf3bf21cb34386`.
The independent verification log is `target/ticket-subgroup-hint/verify.log`.
Runtime `540eccf` remains qualified; baseline parity remains unmet.

## Scalar B shared-memory column permutation

Two rejected prototypes on `f466007` permute the physical shared-memory columns
of F32 scalar B tiles while retaining each thread's original output columns.
For `BN = 16 * TN`, the mapping is
`P(c) = c / TN + (c % TN) * 16`. It is bijective, and the original output column
`c = tx * TN + j` maps to `tx + j * 16`. Both staging and reduction use this
mapping. Unlike the earlier interleaved-output experiment, global storage
addresses and output ownership remain unchanged. So do masked element-zero
reads, padding, per-output K order, guards, barriers, shared capacity and hazards.

The broad form applies to non-interleaved F32 tiles with `TN > 1`; the second
restricts it to transposed B. Other formats and explicit interleaved output
layouts remain unchanged. NVIDIA documents successive 32-bit shared words in
successive banks, motivating a test of consecutive reduction loads across
`tx`; this source-level observation does not establish native vectorization or
a speedup. See the [CUDA programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html).
An enumerated index proof checks unique staging writes and exact load mapping.
Generated-source audits invert only the changed shared B indices and reproduce
the original ordinary shader bodies after whitespace/comment normalization.

Each variant passes **40 code-generation checks, nine whole-graph smoke tests
and eight strict matrix oracle cases**. Each preserves 500 normal comparisons
in control/candidate/candidate/control order: 25 interleaved ordinary/persistent
pairs per case and visit, ten warmup pairs, at least one second of warmup and
256 workers. Controls are immutable qualified `540eccf` binaries. All full-state,
selection, completion, one-dispatch and no-fallback gates pass, with maximum
model error 4.66e-9. No build, heavy analysis or SDK sampling overlaps timings.

| Form / case | Control ordinary / persistent ms | Candidate ordinary / persistent ms | Persistent candidate/control, bootstrap 95% interval |
|---|---:|---:|---:|
| Broad, forward | 4.094 / 5.398 | 4.177 / 5.684 | 1.053 [0.982, 1.098] |
| Broad, training + SGD | 14.600 / 17.741 | 15.889 / 18.320 | 1.033 [1.028, 1.074] |
| Transposed B, forward | 3.777 / 5.164 | 4.203 / 5.609 | 1.086 [1.022, 1.095] |
| Transposed B, training + SGD | 14.649 / 18.060 | 16.170 / 18.497 | 1.024 [1.009, 1.040] |

Broad persistent training regresses **3.3%**, while ordinary training regresses
**8.8%**. The transposed-B subset regresses **2.4% / 10.4%**, respectively.
Their normalized training ratios, 0.947 [0.930, 0.961] and 0.930 [0.856, 0.950],
improve by degrading the ordinary baseline more. That does not satisfy the
performance objective. Broad CI SGD also regresses 4.2% raw / 3.6% normalized;
the subset's CI SGD comparison is inconclusive.

The subset's complete forward and CI-inference shaders are **byte-identical**
to their controls, with matching compiled pipeline identities and resource
multisets in both backends. Their timing shifts, including an apparent
normalized forward improvement, therefore cannot be attributed to this code
change. The 10,000-resample intervals describe within-run uncertainty, not all
between-visit variation. All samples and outliers remain in the record.

Twenty resource records retain 48 CI registers, 102/103 model registers and
the original shared capacities. Broad model code grows from 107,904/267,776 to
109,952/273,280 bytes; the subset leaves forward unchanged and grows training
to 270,080 bytes. Ordinary-source audits cover all 19 CI and 43 model files per
variant. Broad/subset changes affect six/three CI files and eleven/four model
files, including the combined graph shaders.

Fresh SDK captures cover **the broad variant only**, with both model phases,
both backends and four single-pass metric groups per capture: eight backend
captures / 32 groups. Actual saved states match exactly within each backend;
cross-backend state passes the existing budget. Training replay includes SGD
at zero learning rate and does not replace the nonzero-rate completed-step gate.

| Phase / version | Ordinary / persistent replay ms | Persistent warp instructions, billions | Persistent integer-pipe instructions, billions | Persistent shared wavefronts, millions |
|---|---:|---:|---:|---:|
| Forward, control | 3.744 / 5.037 | 0.656 | 1.999 | 124.595 |
| Forward, broad | 3.812 / 5.231 | 0.668 | 2.125 | 124.758 |
| Training, control | 11.910 / 15.901 | 2.722 | 6.575 | 488.368 |
| Training, broad | 13.004 / 16.610 | 2.774 | 6.757 | 431.394 |

Persistent training's estimated shared activity falls **11.7%**, but integer
activity rises 2.8%, warp instructions rise 1.9% and replay slows **4.5%**.
Forward shared activity is approximately flat, integer activity rises 6.3%
and replay slows 3.9%. Persistent local load/store traffic remains zero.
Ordinary local load/store estimates rise from about 79,000 to 121,000 sectors
in forward and fall from about 405,000 to 200,000 in training. No native
instruction or spill attribution is established from these totals.

Clock spread across groups is at most 0.406%, and instruction-rate spread is
at most 1.705%. Persistent workgroup-barrier waiting rises from 58.94% to 59.61%
of active warp cycles in forward and 50.81% to 52.00% in training. Counter counts
are rate-times-duration estimates; these waiting shares are not elapsed-time
attribution. The evidence rejects reduced shared activity as a sufficient
argument for this layout and directs further work toward instruction cost.

Both prototypes are restored exactly to runtime `540eccf`; dependency locks
match the frozen qualified copy, both forks are clean, release examples are
rebuilt, and all nine restored smoke checks pass. No full library/oracle/matrix
qualification or subset SDK capture is claimed for the rejected prototypes.
Every heavy job uses a verified separate 16/20/1 GiB cgroup; GPU jobs remain
serialized and all artifacts are on local `/x`.

The records are under `target/scalar-shared-b-permutation{,-bt}/`. The manifest
`target/megakernel-results/scalar-shared-b-permutation/summary.json` preserves
**1,000 comparisons, 20 resource records, eight SDK captures / 32 metric groups,
and 814 independently verified artifact hashes**, SHA-256
`014007c7b3c923ca3c954fec292bfce5e1c61bc63886b8caba9efd2f9a1bde17`.
Its independent audit log is `target/scalar-shared-b-permutation/verify.log`.
Baseline parity remains unmet.

## Uniform interior A staging

Three rejected prototypes on `5095e6c` test a uniform direct-load path for
complete scalar A stages. The predicate checks `M >= BM`,
`tile_row <= M - BM` and `K - t >= KTILE`; the existing loop guarantees
`t < end_k <= K`. Every admitted A coordinate is therefore in bounds, and
the direct path uses its original storage index and shared cell. The original
per-element checked stage handles every edge, including its element-zero reads
and positive-zero padding. B staging, K order, stores, subgroup guards,
barriers, shared capacity, descriptors and dependencies remain unchanged.

The broad form enables this path for all F32 scalar layouts. Two subsets
restrict it to transposed-A or transposed-B products. Both backends use the
same template; other weight formats retain the original checked A stage.
An independent coordinate proof covers **3,133,440 staged elements**, including
both A mappings, legal tile/stage sizes, edge dimensions and unsigned-32-bit
limits. This proves the coordinate predicate without authorizing allocations
beyond the runtime's existing limits. Generated-source audits replace the new
branch with its unchanged edge body and reproduce the original ordinary shader
bodies after whitespace/comment normalization.

Each form passes **40 code-generation checks, nine whole-graph smoke tests and
eight strict matrix oracle cases**, followed by 500 normal comparisons. The
control/candidate/candidate/control protocol uses 25 interleaved ordinary/
persistent pairs per case and visit, ten warmup pairs, at least one second of
warmup and 256 workers. Controls are frozen qualified `540eccf` binaries.
All 1,500 comparisons preserve full-state budgets, matching selections,
completion, one persistent dispatch and no fallback. Maximum model error stays
4.66e-9. Builds, heavy analyses and profiling do not overlap normal timings.

| Form / case | Control ordinary / persistent ms | Candidate ordinary / persistent ms | Persistent candidate/control, bootstrap 95% interval |
|---|---:|---:|---:|
| Broad, forward | 4.169 / 5.524 | 4.654 / 5.583 | 1.011 [0.997, 1.072] |
| Broad, training + SGD | 14.563 / 17.802 | 15.829 / 18.259 | 1.026 [1.004, 1.035] |
| Transposed A, training + SGD | 14.626 / 17.757 | 14.570 / 17.835 | 1.004 [0.987, 1.084] |
| Transposed B, training + SGD | 14.618 / 17.818 | 15.226 / 18.118 | 1.017 [0.996, 1.025] |

The broad form slows persistent training **2.6%** and CI SGD **5.4%**; ordinary
forward/training also regress **11.6% / 8.7%**. Transposed-A restriction gives
no established model-training gain and slows CI SGD **4.6%**, or 4.9% normalized.
The transposed-B subset gives no established persistent training gain and slows
ordinary training **4.2%**. Its small 0.8% raw CI SGD reduction accompanies a
2.0% ordinary slowdown and does not remedy the model result.

Normalized training ratios are 0.921 [0.873, 0.955], 1.012 [0.988, 1.108] and
0.973 [0.951, 0.987] for broad, transposed-A and transposed-B forms. The favorable
broad/transposed-B ratios reflect the larger ordinary slowdown and fail to
establish faster persistent steps. All samples and outliers remain. The
10,000-resample intervals describe within-run uncertainty, not all between-visit
variation.
Both restricted variants leave the complete forward and CI-inference shaders
byte-identical, with matching compiled pipeline identities and resource
multisets in both backends; their timing shifts are not attributed to this
change.

Thirty resource records retain 48 CI registers, 102/103 model registers and
all original shared capacities. Broad model code grows from 107,904/267,776 to
112,896/282,368 bytes. Restricted forward code stays unchanged; training grows
to 272,896 bytes for transposed A and 272,768 for transposed B. Transposed-A
CI SGD code shrinks from 65,408 to 65,024 bytes despite its slowdown. These
totals and source-level check removal do not establish native instruction
counts or explain the elapsed-time changes. No new SDK capture or full-suite
qualification is claimed for these rejected candidates.

All three probes are restored exactly to runtime `540eccf`, including the
retained B-stage optimization. Dependency locks match the qualified copy, both
forks are clean, release examples are rebuilt, and all nine restored smoke
tests pass. Heavy jobs use verified separate 16/20/1 GiB cgroups, GPU work is
serialized, and every artifact stays on local `/x`.

Records are under `target/scalar-interior-a{,-at,-bt}/`. The manifest
`target/megakernel-results/scalar-interior-a/summary.json` preserves **1,500
comparisons, 30 resource records and 991 independently verified artifact
hashes**, SHA-256
`21f0cafb3925e8503b5078ac6ac13e99a9490c22a5fc70997c3e151c1b252d28`.
The independent audit log is `target/scalar-interior-a/verify.log`.
Baseline parity remains unmet.

## Staging only valid A rows

Two rejected prototypes on `f758c61` initialize invalid A rows to positive zero
once, before the K loop, then stage only valid rows on each iteration. They
apply to F32 scalar products with normal A indexing, including transposed B;
transposed A and other weight formats retain the original stage. The first
form bounds the staging loop by the valid-row count. The second keeps the
original constant iteration count and guards each iteration by that count.
Both backends use the same template.

The valid prefix and explicitly cleared suffix partition the used shared A
cells. Every valid cell retains its original address and checked K-tail load;
invalid cells remain zero across stages. The first existing staging barrier
orders both initialization and valid writes. No barrier, reduction, store,
shared capacity, subgroup guard or dependency changes. K=0 uses none of the
initialized cells. Removing invalid-row element-zero reads does not qualify
narrower read hazards: dependencies still cover the original allocations.

Each form independently compares **3,674,112 shared-cell values across 4,784
stages and 184 layouts**, starting from poisoned scratch and using opaque
input values. The proof covers zero dimensions, edges and unsigned-32-bit
limits without authorizing larger allocations. Inverse source audits reproduce
the original ordinary shader bodies after restoring only A initialization and
staging; all 19 CI and 43 model files are accounted for.

Each variant passes **40 code-generation checks, nine whole-graph smoke tests
and eight strict matrix oracle cases**, then 500 normal comparisons against
frozen qualified `540eccf` binaries. The control/candidate/candidate/control
protocol uses 25 interleaved pairs per case and visit, ten warmup pairs, at
least one second of warmup and 256 workers. All 1,000 comparisons preserve
matching selections, full-state budgets, completion, one persistent dispatch
and no fallback. Maximum model error remains 4.66e-9.

| Loop form / case | Control ordinary / persistent ms | Candidate ordinary / persistent ms | Persistent candidate/control, bootstrap 95% interval |
|---|---:|---:|---:|
| Variable bound, forward | 4.187 / 5.586 | 4.839 / 6.504 | 1.164 [1.129, 1.181] |
| Variable bound, training + SGD | 14.610 / 17.756 | 16.294 / 19.481 | 1.097 [1.013, 1.179] |
| Fixed bound, forward | 4.118 / 5.487 | 4.392 / 6.168 | 1.124 [1.114, 1.198] |
| Fixed bound, training + SGD | 14.603 / 17.785 | 14.935 / 19.650 | 1.105 [1.090, 1.110] |

The variable-bound form slows persistent forward/training **16.4% / 9.7%**;
ordinary execution also slows **15.6% / 11.5%**. Its favorable normalized
training ratio, 0.967 [0.948, 0.998], reflects the larger ordinary regression.
The fixed-bound form slows persistent forward/training **12.4% / 10.5%** and
CI SGD **6.2%**; normalized training and CI SGD regress **7.6% / 3.3%**. Neither
form advances parity. All samples remain; the 10,000-resample intervals
describe within-run uncertainty, not every source of between-visit variation.

Twenty resource records preserve 48 CI registers, 102/103 model registers and
the original shared capacities. Model code grows from 107,904/267,776 bytes
to 124,800/293,888 with variable bounds and 118,656/286,464 with fixed bounds.
The variable-bound CI code shrinks despite slower medians. Source operations
and these resource totals do not establish native instruction counts, loop
unrolling, spills or the cause of the timing changes. No fresh SDK capture or
full-suite qualification is claimed for the rejected prototypes.

Both probes are restored exactly to runtime `540eccf`. Dependency locks match
the qualified copy, both forks are clean, release examples are rebuilt, and
all nine restored smoke tests pass. Heavy jobs use verified separate cgroups
with 16 GiB memory high, 20 GiB maximum and 1 GiB swap; GPU work is serialized,
and builds/heavy analyses do not overlap measurements. Artifacts stay on `/x`.

Records are under `target/scalar-bounded-a{,-fixed}/`. The manifest
`target/megakernel-results/scalar-bounded-a/summary.json` preserves **1,000
comparisons, 20 resource records, 24 verified measurement scopes and 665
independently verified artifact hashes**, SHA-256
`6f35ed2ef68dec4f732ec441268732f939ac9d1cc19a12a6feb7722cc23d025e`.
The independent audit log is `target/scalar-bounded-a/verify.log`.
Baseline parity remains unmet.

## Direct coordinates for one-dimensional grids

Two rejected prototypes on `a6a7629` use the current node's uniform
`grid.y == 1 && grid.z == 1` predicate to construct logical coordinates
directly. Other grids retain the original quotient/remainder projection.
The broad form applies to every handler, including packed tiles. The subset
applies only inside the existing repeated optimizer batches. Neither changes
batch admission, selected arithmetic, shared state, barriers, fences or queue
operations. Runtime inspection preserves reuse of a handler with different
grids; no dimensions are assumed from its first recording.

For an entered claim, the tile index is below the logical count. Sequential
repeats retain their bounded final batch. Packed widths divide their selected
extent, and inactive groups clamp to the last active coordinate. For a
one-dimensional grid, x-packed coordinates therefore equal the tile plus its
clamped offset; packing on y or z has width one. The resulting y/z coordinates
are zero.
Zero-work nodes retain their outer skip, and overflow admission is unchanged.
Each independent proof checks **338,998 coordinates across 602 grids**, including
partial batches, inactive groups and unsigned-32-bit boundaries. This proves
the mapping without authorizing allocations beyond existing limits.

Both variants pass **49 code-generation checks, nine whole-graph smoke tests
and all 168 strict oracle tests**. Each then receives 500 normal comparisons
against frozen qualified `540eccf` binaries, using the established
control/candidate/candidate/control protocol, 25 interleaved pairs per case
and visit, ten warmup pairs, at least one second of warmup and 256 workers.
All 1,000 comparisons preserve matching selections, full-state budgets,
completion, one persistent dispatch and no fallback; maximum model error is
4.66e-9.

| Form / case | Control ordinary / persistent ms | Candidate ordinary / persistent ms | Persistent candidate/control, bootstrap 95% interval |
|---|---:|---:|---:|
| Broad, forward | 4.118 / 5.493 | 4.118 / 5.517 | 1.004 [0.943, 1.065] |
| Broad, training + SGD | 14.622 / 17.815 | 14.639 / 17.873 | 1.003 [0.961, 1.087] |
| Repeated batches, training + SGD | 14.791 / 17.749 | 14.583 / 18.113 | 1.020 [1.006, 1.106] |

The broad form establishes no speedup in either model phase or CI. The subset
slows persistent training **2.0% raw / 2.6% normalized**. All samples and outliers
remain; bootstrap intervals describe within-run uncertainty, not all
between-visit variation. The subset's complete forward and CI shaders and
compiled pipeline/resource multisets are identical, so their timing shifts
are not attributed to this change.

Source audits account for 19 CI and 43 model modules per variant. Every ordinary
module is byte-identical, with matching selected pipeline keys and resource
multisets. Reversing only the coordinate transformation reproduces every whole
graph module after whitespace/comment normalization. Initial order-sensitive
resource assertions and their multiset corrections remain in the audit records.
Twenty resource records preserve shared capacities. Training registers fall
from 103 to **100 / 101** for broad/subset forms; code changes from 267,776 to
268,928 / 267,648 bytes. These reductions do not establish faster execution.

Fresh SDK captures cover **broad training only**: four backend captures and
sixteen sequential single-pass metric groups. Actual saved states match exactly
within each backend, and the cross-backend comparison satisfies the existing
budget. Zero-rate SGD keeps resident replay stable; it does not replace the
nonzero-rate completed-step gate.

| Version | Ordinary / persistent replay ms | Persistent warp instructions, billions | Persistent integer-pipe instructions, billions | Persistent shared wavefronts, millions |
|---|---:|---:|---:|---:|
| Control | 11.911 / 15.883 | 2.721 | 6.573 | 488.285 |
| Broad | 11.941 / 15.873 | 2.708 | 6.132 | 488.137 |

Estimated persistent integer-pipe work falls **6.7%**, but replay changes only
**-0.06%**. Warp instructions fall 0.48%, shared activity remains approximately
flat, and persistent local load/store traffic stays zero. Barrier waiting is
50.83% / 51.04% of active warp cycles. Across metric groups, clock spread is at
most 0.412% and instruction-rate spread 1.016%. Rate-times-duration counts are
estimates, and barrier shares do not attribute elapsed time. This transformation
reduces measured integer activity without advancing the completed-step gate;
it does not isolate the remaining cause of the performance gap.

Both probes are restored exactly to runtime `540eccf`, locks match the qualified
copy, both forks are clean, release examples are rebuilt, and all nine restored
smoke tests pass. No full retained-change qualification or subset/forward SDK
capture is claimed. Heavy jobs use verified separate 16/20/1 GiB cgroups,
GPU jobs are serialized, measurements exclude builds/heavy analyses, and all
artifacts stay on local `/x`.

Records are under `target/linear-grid{,-repeat}/`. The manifest
`target/megakernel-results/linear-grid/summary.json` preserves **1,000 normal
comparisons, 20 resource records, four SDK captures / sixteen metric groups,
26 verified measurement scopes and 762 independently verified artifact hashes**,
SHA-256 `92ec945f6265c987570a9fb524863c408aabafc73f5cb805c3cefcf17965c07f`.
The independent audit log is `target/linear-grid/verify.log`.
Baseline parity remains unmet.

## Scalar staging address hoisting

Two rejected prototypes on `8554bd0` precompute a lane's matrix base and stage
stride before the scalar K loop. The first applies to F32 products with normal
A storage and staging maps. The second restores A and applies the same
transformation only to F32 transposed B, preserving both its retained interior
branch and its original checked edge path. Both backends use the same template.

For the admitted K tiles, `KT` divides 256. Therefore
`floor((tid + e*256)/KT) = floor(tid/KT) + e*(256/KT)` and the remainder is
`tid % KT`. The original matrix index equals `base + e*stride + t`, including
unsigned wrapping. The base uses `tile_row` for A and `tile_col` for transposed
B. Original bounds checks, discarded element-zero reads, shared addresses,
K order, splits, barriers, subgroup guards and output arithmetic remain.
Precomputation reads only immutable parameters when K=0; no matrix read is
introduced. Dependencies and selected geometry remain unchanged.

Each proof checks **3,354,624 addresses across eleven layouts**, including
2,325,808 wrapping cases and original masked reads. These arithmetic boundary
checks do not authorize allocations beyond existing limits. Source audits
remove only the base/stride declarations and restore the original index
expressions, reproducing every ordinary shader body after whitespace/comment
normalization. All 19 CI and 43 model modules per variant are accounted for.

Each form passes **49 codegen checks, nine whole-graph smoke tests and eight
strict matmul oracle cases**, followed by 500 normal comparisons against frozen
qualified `540eccf` binaries. The established control/candidate/candidate/control
protocol uses 25 interleaved pairs per case and visit, ten warmup pairs, at least
one second of warmup and 256 workers. All 1,000 comparisons preserve matching
selections, full-state budgets, completion, one persistent dispatch and no
fallback; maximum model error remains 4.66e-9.

| Operand / case | Control ordinary / persistent ms | Candidate ordinary / persistent ms | Persistent candidate/control, bootstrap 95% interval |
|---|---:|---:|---:|
| A, forward | 4.124 / 5.531 | 4.232 / 5.629 | 1.018 [1.010, 1.076] |
| A, training + SGD | 14.582 / 17.853 | 14.772 / 18.112 | 1.015 [0.993, 1.024] |
| Transposed B, training + SGD | 14.630 / 17.779 | 14.592 / 17.996 | 1.012 [0.997, 1.086] |
| Transposed B, CI SGD | 0.06661 / 0.09468 | 0.06637 / 0.10003 | 1.057 [1.023, 1.065] |

A hoisting slows persistent/ordinary forward **1.8% / 2.6%** and establishes no
training improvement. Transposed-B hoisting establishes no model-training gain
and slows CI SGD **5.7% raw / 5.2% normalized**. Its complete model-forward and
CI-inference shaders and compiled pipeline/resource multisets are identical
in both backends despite timing shifts, including 7.0% higher persistent CI
inference. Those shifts are not attributed to the shader change. All samples
remain; bootstrap intervals do not cover all between-visit variation.

Twenty resource records preserve 48 CI registers, 102/103 model registers and
all shared capacities. A hoisting grows model code from 107,904/267,776 to
108,288/268,416 bytes. Transposed-B model code retains the original sizes;
its CI SGD code grows from 65,408 to 66,560 bytes. These totals do not establish
native instruction identities, load-address latency or the cause of the timing
changes. No fresh SDK capture or full retained-change qualification is claimed.

Both probes are restored exactly to runtime `540eccf`, locks match the qualified
copy, both forks are clean, release examples are rebuilt, and all nine restored
smoke tests pass. Heavy jobs use separate verified 16/20/1 GiB cgroups, GPU work
is serialized, builds and heavy analyses do not overlap measurements, and all
artifacts stay on local `/x`.

Records are under `target/scalar-{a,bt}-address/`. The manifest
`target/megakernel-results/scalar-stage-address/summary.json` preserves **1,000
comparisons, 20 resource records, 24 verified measurement scopes and 667
independently verified artifact hashes**, SHA-256
`9aa3c53d539a513b4ecc9bb7b68635b6f6aa574ff50686defa64b80908fdd4a3`.
The independent audit log is `target/scalar-a-address/verify.log`.
Baseline parity remains unmet.

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
