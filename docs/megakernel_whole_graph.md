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
each selected logical workgroup. Extra lanes participate in every workgroup
barrier; only the original lanes write shared/storage memory. Inactive subgroups
skip cooperative multiply-adds assigned to local accumulators. Other extra-lane
calculations are still repeated. Subgroup handlers use virtual subgroup coordinates. A shader guard
checks dense, linear subgroup mapping, full groups and compatible widths before
executing graph work. This mapping is a qualification condition, not a portable
Vulkan assumption: [subgroup and local invocation indices are not generally
related](https://docs.vulkan.org/refpages/latest/refpages/source/SubgroupLocalInvocationId.html).
Unsupported mapping sets a completion diagnostic and executes no tile in that
worker. Call `validate_persistent_completion()` during qualification.

Shared arrays reuse a pool by element type and slot; capacity is the maximum
across handlers. Original per-tile initialization remains in each callable body.
Uniform parameters are loaded through `workgroupUniformLoad` and passed by value
through tile helpers, preserving Naga's uniform-control-flow validation.
No validation flags related to barrier uniformity are disabled.

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
counter, remaining-tile counter and predecessor count. A worker claims a tile
from a ready node; the final claim clears its ready bit. The final completing
tile releases every successor whose predecessor count reaches zero. Zero-tile
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
GPU tests. All six focused whole-graph tests pass. The external-recording test
also waits for a submitted step while a later recording with a different
learning rate remains unsubmitted, then verifies all three parameter updates.

On RTX 5080, strict whole-graph selection passes all **168 oracle tests**,
including attention, convolution, reduced-precision storage, scatter, fusion and
backward cases. Queue completion is checked by the shared reference harness.
The ordinary-path integration run passes **664 library, 168 oracle, 182
regression, 81 smoke and 22 GGUF tests** (8 library, 21 regression and 2 smoke
tests remain ignored).
Formatting, all-target Clippy, bare/all-feature checks and documentation pass.
The Metal library cross-check passes with `models,gguf`; all-feature cross-checking
is blocked by the Linux C compiler's unsupported Apple flags in `ring`.
No Metal GPU execution is claimed.

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
without recovering the elapsed-time gap. The next resource investigation is
whether a physical worker can use spare subgroups for additional independent
logical tiles. Such a change must preserve scratch isolation, arithmetic,
workgroup-barrier uniformity and the existing completion/fence contract; the
earlier rejected sequential four-tile batching does not provide this behavior.

## Latest completed-step measurement

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

## Reproduction and artifact identities

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
