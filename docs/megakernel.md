# Persistent megakernel prototype

The [integration and Naga plan](megakernel_plan.md) records the next milestones:
persistent regions through `Session`, followed by investigation of the fork's
memory/atomic semantics for device-scheduled graphs.

`meganeura::megakernel` experiments with a fixed pool of 64-thread workgroups
that stay alive and dispatch work inside a single GPU launch. Each worker
atomically claims a job, interprets its operations, and returns to the queue
until the finite batch is exhausted. The worker count is configurable and
independent of the number of jobs. The normal graph compiler/runtime is
unchanged; this is an explicit experimental API.

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
