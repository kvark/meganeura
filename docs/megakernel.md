# Persistent megakernel prototype

The [integration and Naga plan](megakernel_plan.md) records the completed
portable-region and compiler milestones, followed by device-scheduled Session
regions and the remaining operator coverage work.

## Session integration

Generated pointwise and reduction operations can now execute through persistent
tile queues using the ordinary `Graph` / `build` / `Session` path:

```rust,no_run
# let graph = meganeura::Graph::new();
let mut config = meganeura::SessionConfig::from_env();
config.runtime.persistent = Some(meganeura::PersistentOptions {
    workers: 64,
    allow_fallback: true,
    ..Default::default()
});
let (mut session, _) = meganeura::build(&graph, config);
let coverage = session.execution_report();
session.step();
session.wait();
```

Compatible independent operations now share a persistent launch, with up to
three generated handlers per region. Each launch drains logical tiles using
the handlers' existing workgroup size. Region boundaries retain hardware
barriers. Matching pointwise tiles can also form a worker-owned chain; global
reductions and dependencies crossing workers retain dispatch boundaries.
The ordinary backend remains the default. BF16 handlers currently stay in
single-operation regions because their module-wide declarations are shared.

A chain contains only flat pointwise bodies with identical extent and logical
grid, no broadcast reads, and consistent logical references wherever physical
storage overlaps. Each queue claim owns the same index range through every
body. All lanes finish a body and execute `storageBarrier()` before advancing
to the next one; no other worker reads or writes that index range inside the
region. External consumers run after the region's dispatch barrier. Chains
only cross consecutive singleton dependency groups, and explicit serial
dispatch selection disables chaining. This needs no device-wide fence,
spinning wait, or assumption about simultaneous worker residency.

Regions are planned after storage allocation, with checks for device binding
limits, binding ranges, shared-memory use, and physical overlap. Vulkan and Metal
report these limits; unreported backend limits produce explicit fallback or strict
rejection. Session buffers are bound directly, including parameter sharing and
external rebindings. Rebinding rebuilds the regions. Sessions with imported
storage retain one operation per launch because separate imported handles may
alias an allocation. Intermediate lifetime aliasing is disabled in this mode.
Queue counters are allocated once, included in `MemorySummary.scheduler_bytes`,
and reset by a GPU transfer at every step, including externally recorded steps.
Tensor data and training/cache state are not reset. Profiling reserves a slot
for queue reset and reports one timing per physical launch, with untimed source
operations attached. Tuning rebuilds the persistent pipelines after selection.

`execution_report()` lists operation provenance, logical queue tasks, worker geometry,
physical graph launches, fallback reasons, and configured appended optimizer,
clipping, or accumulation work. `allow_fallback: false` rejects unsupported
graph operations and execution of ordinary optimizer/accumulation passes.
A queue task represents one tile or one matching tile through a chain;
`worker_owned_chain` identifies the latter and the region retains all source
operation indices.
For a separate diagnostic run, `PersistentOptions::track_workers` enables
per-worker task counters. After a completed step,
`Session::read_persistent_worker_tasks()` downloads one row per region, with
empty rows for ordinary fallback. The counts reset each step and sum to the
region's logical tasks. This adds an atomic per claimed task and increases
scheduler storage, so keep it disabled in primary latency comparisons.
A passthrough graph can have zero tasks; qualification must assert nonzero
persistent coverage for cases intended to exercise this path.

Harnesses using environment configuration can select the path with
`MEGANEURA_PERSISTENT_WORKERS=64`, and reject fallbacks with
`MEGANEURA_PERSISTENT_STRICT=1`. `reference::gpu::Options.persistent` provides
the same explicit selection for oracle checks.

### Device dependencies inside Session regions

`PersistentOptions::device_scheduling = true` (or
`MEGANEURA_DEVICE_SCHEDULING=1` alongside the worker selector) enables the
[qualified Vulkan fence contract](megakernel_memory_contract.md). The same
generated pointwise/reduction bodies use a bounded ready bitmap and
whole-operation completion counters when a region contains dependencies.
Independent and singleton regions retain the portable queue, with no native
fences or bitmap scans. Compatible dependent operations can share
one launch even when consumers read tiles produced by other workers. The planner
orders physical RAW/WAR/WAW hazards, including rebinding, and annotates aliased
read-only inputs as coherent. Immutable inputs keep their ordinary accesses.

Regions still contain at most three compatible handlers. Different workgroup
geometry, resource limits, imported storage and unsupported kernels retain
physical dispatch boundaries or explicit fallback. Serial selection retains one
operation per launch. BF16 handlers remain singleton-only. This is device
scheduling within reported regions, not full graph coverage or a promise of one
launch for every graph.

The queues and any immutable reset image are allocated once and included in
scheduler residency. Sessions with device regions use a GPU copy to restore
counters and root readiness at each step; other sessions retain zero-fill reset.
Tensor,
optimizer and cache data are untouched. Every worker scan and task loop is
bounded. `ExecutionRegion::device_scheduled` identifies the selected protocol.
After a completed step, `Session::validate_persistent_completion()` checks that
tasks, dependency/tile counters, ready bits and error flags have drained. This
downloads queue state and belongs outside timing; oracle checks and the paired
benchmark call it automatically. Worker-counter readback checks completion too.

The existing paired benchmarks select this path with `--device-scheduling`.
The ordinary backend remains the default while correctness and completed-step
measurements qualify each expansion in scope.

## Standalone prototype

`meganeura::megakernel` experiments with a fixed pool of 64-thread workgroups
that stay alive and dispatch work inside a single GPU launch. Each worker
atomically claims a job, interprets its operations, and returns to the queue
until the finite batch is exhausted. The worker count is configurable and
independent of the number of jobs. This remains a separate experimental API
for scheduling diagnostics.

```sh
# Arguments: workers, jobs, vector width, residual MLP layers, timing pairs.
cargo run --release --example megakernel -- 64 1024 64 2 20
cargo test --test smoke megakernel -- --nocapture
```

The example checks complete final arenas (inputs, retained intermediates, and
outputs) against independent CPU f64 calculations before and after timing. The workload is
`x + W2 * relu(W1 * rms_norm(x))`, repeated for the requested layer count.
Weights are shared and read-only; jobs represent independent vectors. JSON
output includes raw alternating-order paired samples, median GPU compute and
host wall times, dispatch counts, numerical error, and jobs executed by each
worker. A 500 ms alternating warmup precedes the samples. GPU timestamps must
be available for the example. The executor also works without timestamps
(`Options::gpu_timing = false`).

## Work representation and dispatch

- A `Job` contains initial f32 data and an instruction sequence. Offsets refer
  to its own private arena. Different jobs may have different dimensions,
  operation sequences, and lengths, including no operations.
- `Instruction` supports affine, add, ReLU, unweighted RMS normalization, and
  a row-major matrix-vector product. The matrix reads the batch's shared weight
  buffer. Each instruction cooperatively runs on all 64 worker invocations.
- `Executor::new` validates element ranges, integer arithmetic, and aliases,
  then uploads immutable descriptors and data into device-local storage.
  Exact in-place elementwise and RMS operations are allowed; overlapping
  matrix input/output and partial overlaps are rejected.
- `Executor::run(Execution::Persistent)` restores initial data, clears queue
  and counters, launches the configured workers once, copies results, and
  waits. Queue reset is a transfer command, not another compute dispatch.
  `read_job` returns a complete arena and `statistics` reports completion
  counts and per-worker job/instruction totals after the wait.
- `Execution::Separate` executes the same instruction bodies and buffer
  layout, using one dispatch per instruction depth with a GPU memory barrier
  between dispatches. Both paths reset inputs and copy results identically.
  This isolates scheduling differences; it is not a comparison with optimized
  matrix kernels or the production graph compiler.

## Synchronization contract

Only the workgroup leader increments the queue head. `workgroupUniformLoad`
broadcasts job and instruction descriptors so loop termination, opcode
selection, and barriers are uniform within each workgroup. A storage barrier
orders every instruction's writes before its successor reads them. RMS
normalization also uses workgroup barriers for its reduction.

Every writable arena belongs to exactly one job. A claimed job executes to
completion on the same worker, so there are no tensor dependencies between
workers. The global atomic queue allocates immutable descriptors; it does
not publish computed tensors. Completion counters are diagnostics read by
the host after the submission fence, not synchronization flags for consumers.

This distinction follows the [WGSL memory model](https://www.w3.org/TR/WGSL/#memory-model):
atomics have relaxed ordering and shader barriers have workgroup scope.
A storage atomic flag by itself cannot safely publish ordinary tensor writes
to another workgroup. The prototype therefore has no cross-workgroup spin
waits or global barrier, and does not require all workers to be resident
simultaneously. Oversubscribing workers affects performance, not progress.

## Scope and next experiments

Persistence lasts for one batch. Workers are logical workgroups, not physical
CU/SM identifiers; Vulkan/Metal control their placement. The API does not pin
one worker to each CU, accept host submissions while running, or implement
arbitrary graph DAGs with cross-worker dependencies. Such dependencies need
dispatch boundaries or a backend-specific publication/scheduling protocol.

This prototype uses f32 scalar matrix-vector bodies, one central queue, and
global-memory intermediates. Interpreter branching, barriers, queue contention,
and limited parallelism within one job can outweigh launch savings. A worker
sweep and varying job size/depth are necessary before selecting a pool size.
Tiled matrix work, chunked queue claims, generated instruction specialization,
and lowering independent graph regions are possible follow-up experiments.

The smoke checks cover ragged vectors around the 64-thread boundary, mixed
program lengths, all operation kinds, in-place writes, NaN-poisoned scratch,
repeated resets, one worker executing many jobs, more workers than jobs,
empty queues, exact completion counts, and invalid descriptors. The example
adds a multi-layer residual MLP and paired GPU timings.

## Initial measurements (October 24, 2026)

Linux, RTX 5080, NVIDIA 595.91.07, release build, 500 ms warmup and 30
alternating timing pairs per configuration. All retained intermediate/output values
passed the CPU f64 check; maximum absolute error was below `2.4e-6`.
Times below are GPU compute microseconds (reset and readback are outside
this pass). Clocks were not fixed; these are exploratory measurements,
not a production performance claim.

| Jobs | Width | Layers | Workers | Separate | Persistent | Separate / persistent |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 64 | 2 | 1 | 105.1 | 12104.0 | 0.009× |
| 1024 | 64 | 2 | 16 | 104.9 | 772.4 | 0.14× |
| 1024 | 64 | 2 | 64 | 105.2 | 194.7 | 0.54× |
| 1024 | 64 | 2 | 128 | 105.7 | 125.2 | 0.84× |
| 1024 | 64 | 2 | 256 | 106.4 | 102.7 | 1.04× |
| 64 | 16 | 16 | 64 | 144.9 | 43.6 | 3.33× |
| 137 | 65 | 3 | 64 | 58.6 | 80.0 | 0.73× |

The deep, small-vector case replaces 80 dispatches with one; its median host
wall time also fell from 261.8 to 91.1 us. For 1024 wider jobs, limiting the
pool to 64 workers loses substantial parallelism; 256 workers were roughly
even with the ten-dispatch control. The odd-width case loses at 64 workers.
Removing dispatches alone does not establish a speedup.

Select the adapter explicitly on a machine with multiple GPUs. The example
honors `MEGANEURA_DEVICE_ID`; on Linux the Vulkan loader can also be restricted:

```sh
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  cargo run --release --example megakernel -- 64 64 16 16 30
```

The prototype's correctness tests also passed on the integrated AMD
RAPHAEL_MENDOCINO GPU and Lavapipe. The complete 48-test smoke suite passed on RTX 5080
with Vulkan synchronization validation enabled. Metal has not been tested.
