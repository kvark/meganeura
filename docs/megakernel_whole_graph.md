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
carry the original uniform bytes.

One physical workgroup is at least 256 invocations and must be a multiple of
each selected logical workgroup. Extra lanes repeat logical calculations and
participate in every barrier; only the original lanes write shared/storage
memory. Subgroup handlers use virtual subgroup coordinates. A shader guard
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
which can serialize independent slices. Descriptors are coherent and producer
and consumer lanes use the existing [native fence protocol](megakernel_memory_contract.md).
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
regression and 22 GGUF tests** (8 library and 21 regression tests remain ignored).
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

Logs remain on local storage under `target/whole-graph-*.log`. Fresh completed-step
measurements and the SDK capture follow this implementation checkpoint.
