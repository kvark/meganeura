# Checkpoints, optimizer lifecycle and memory

This chapter describes the September 6 follow-up after rebasing onto `8069cf3`.
It is current runtime behavior, not a change to the frozen paper experiments.
The two changes share one principle: a tensor's logical identity must not be
confused with the buffers currently allocated to execute it.

## What a checkpoint promises

New saves use SafeTensors checkpoint format **3**. A logical manifest records
each compiled parameter's name, shape and storage type. The payload contains
only that tensor's data, not backend-dependent allocation padding. Adam and
LaProp use the same `adam_m.*` / `adam_v.*` tensor names, accompanied by the
step counter and an explicit list of parameters with saved moments.

F32, F16 and U32 use their actual dtype and logical shape in the SafeTensors
header. Packed Q4/Q8 remain U8 byte payloads; the manifest distinguishes the
quantization format and original logical shape. Loading does not silently
convert storage types or change the numerical contract.

The compiler now retains `ExecutionPlan::param_types` independently of buffer
capacities. Build-cache format **5** invalidates older cached plans without
that metadata. Directly constructed plans must supply it to use checkpoints.
`param_size` reports logical elements, and batched F32 `read_params` omits
padding. `set_parameter` accepts a logical-sized payload and zero-fills any
larger destination slot; existing full-slot uploads still work.

Optimizer updates, clipping, accumulation and grouped diagnostics now use
logical element counts too. Allocation padding is not a gradient: even finite
nonzero tail data must not affect norms or updates. Tests poison parameter
and gradient tails independently and compare SGD, Adam, LaProp and CPU
fallback results with unpadded controls. F32 parameter reads reject other
storage types, and raw F32 buffer reads reject oversized views before copying.

[Compiler metadata](../../src/compile.rs), [cache](../../src/cache.rs),
[checkpoint implementation](../../src/runtime/checkpoint.rs)

## Why validate the whole file first?

Previously a valid first tensor could be copied before a malformed later
moment or step counter was discovered. The caller received an error but had
already lost the old parameter state. The new boundary is:

```text
read + parse
    → validate version, complete logical layout, tensors and step counter
    → prepare derived-weight CPU staging and check destination compatibility
    → wait, allocate required moments, reset absent moments
    → apply tensors with zero padding, restore counter and staging
```

All validation happens before writes or optimizer allocation. A malformed or
unreadable file leaves parameters, gradients, moments, counters and moment
allocation unchanged. This is **preflight failure atomicity**, not rollback
after a GPU allocation failure or device loss. Backend allocation still has
panic/failure paths; the memory-budget guard is not a reservation.

Format 3 rejects mismatched parameter sets, same-byte-count wrong shapes,
wrong dtypes, missing tensors, unexpected tensors, duplicate moment
declarations, name collisions, bad metadata and unsupported format versions.
Every declared moment is validated even when the destination is inference
and will not allocate or use it. A parameter without saved moments starts
with zero moments: a reused training session does not retain stale values.
If the destination has no moments and the file supplies none, it stays lazy.

Derived reduced-storage weights have a CPU staging cache used when updating
one constituent parameter. Restore now refreshes that cache too, preventing
a later source update from resurrecting another source's pre-load values.
The cache is reconstructed from the saved reduced-storage values; it cannot
recover precision discarded before saving.

## Portability and resume limits

Different destination padding is allowed; different named logical parameter
shapes or storage types are not. Training-to-inference restore works when
both compiled plans have the same parameter set. This includes materialized
derived parameters, so it is not arbitrary migration between graphs or
optimization policies.

Formats 1/2 remain readable with their legacy same-physical-size, partial-load
contract. All applicable legacy writes are preflighted, but missing parameters
still warn and remain unchanged. Re-save as format 3 to obtain the new strict
logical contract. Older Meganeura versions are not forward-compatible format-3
readers; use the updated loader.

A checkpoint is not a complete training-loop snapshot. It does not save:

- optimizer choice, learning rate, beta/epsilon, decay or per-parameter rates;
- gradient accumulation contents/window, clipping cadence or diagnostics;
- inputs, KV state, data-loader position, or application RNG state.

Use a fresh session, configure the same optimizer and training-loop settings,
and load at an update boundary for a controlled resume. Successful load into
a reused session leaves accumulation/clip/diagnostic configuration and state
alone; manage those explicitly. Saving is not yet a crash-atomic file replace.
Do not confuse persistence checkpoints with activation checkpointing: the
latter trades backward recomputation for activation memory and remains open.

The RTX qualification deliberately changes allocation capacities while
retaining logical types, then checks the next Adam update against a control.
This establishes a cross-padding invariant on one backend, not completed
Metal↔Vulkan or fleet qualification. Odd F16 tails and Q4/Q8 storage round
trips are covered; arbitrary import conversion is a separate concern.

## Lazy optimizer state

A trainable graph needs gradients even for forward+loss+backward inspection
or SGD. It does not need Adam's two moment buffers until an adaptive optimizer
or explicit state restore requests them.

`set_adam`, `set_laprop`, explicit `adam_step`, moment writes, or a checkpoint
with applicable moments initialize and zero those buffers once. Initialization
preserves device-local placement unless debug/no-device-local configuration
requests shared storage. Runtime optimizer/clip/accumulation kernels require
F32 trainable parameter storage.
Read-only `read_adam_m/v` and `read_adam_states` return correctly sized zeros
before initialization without allocating GPU moments. Saving also stays lazy.

`clear_optimizer` and switching to SGD retain previously allocated moments
and counters, so switching back does not silently reset training. This is
allocation on first use, not an allocator that continually shrinks. Gradient
accumulators are independently lazy and remain allocated when disabled. The
four-byte clipping scalar still exists for a trainable session; optional
grouped-gradient diagnostics have their own shared allocation.

For 1,048,576 unpadded F32 trainable elements, two moments require
`1,048,576 × 4 × 2 = 8,388,608` bytes, or **8 MiB**. A GPU regression verifies
that this allocation is absent initially and appears exactly once when Adam
is selected. This saves unused storage in SGD/F+L+B sessions; it does not
reduce the moments required by an actual Adam run or establish a speedup.

## Read memory reports correctly

| Quantity | Meaning |
|---|---|
| `total_buffer_bytes` | Sum of plan buffer capacities before aliasing; includes runtime padding, despite often being labeled “logical.” |
| `allocated_buffer_bytes` | Graph buffers after aliasing, with minimum allocation sizes; excludes optimizer storage. |
| `adam_state_bytes` | Actually allocated moment buffer requests; zero before initialization. |
| `grad_accumulator_bytes` | Each allocated gradient accumulator counted once, not multiplied by the parameter count. |
| `optimizer_aux_bytes` | Clip scalar and optional grouped-gradient diagnostic buffer. |
| `total_allocated_bytes()` | Graph + moments + accumulators + auxiliary buffer requests currently held by this session. |
| `device_local_bytes` | Device-local subset, including resident optimizer buffers but excluding the shared grouped diagnostic buffer. |
| `device_memory_stats()` | Optional API-reported process usage/budget, broader than the session's tensor buffers. |

The structured profile adds `resident_buffer_bytes`, moments, accumulators
and auxiliary bytes alongside its existing graph-memory fields. “Resident”
here describes retained buffer allocation requests, not physical-page
residency or a driver peak measurement. Staging, pipelines, command buffers,
driver allocation granularity, other sessions and CPU memory are outside this
sum. Query API-level statistics separately and use a defined peak-sampling
protocol before comparing VRAM use with another engine.

[Runtime accounting and lifecycle](../../src/runtime.rs),
[structured profile](../../src/profiler.rs),
[observability comparison](observability.md)

## Qualification and rehearsal

The follow-up passed 241 CPU library tests and 85 selected GPU tests on RTX
5070 / NVIDIA 595.71.05, including eight checkpoint regressions, five
memory/bounds regressions, four scalar tuning tests and eight training-correctness tests.
The focused checkpoint/optimizer suites also pass in debug builds with GPU API validation.
The native-f32 tuner test remains excluded on this f16-tile-only device.
The train/deploy example decreases loss from 0.693147 to 0.026063 and reloads
into fresh inference with 8/8 accuracy. These are correctness/accounting
checks, not a new latency benchmark or large-model convergence study.

CPU-only preflight and cache checks:

```sh
cargo test --lib
```

Focused real-device checks (require an available GPU):

```sh
cargo test --release --test checkpoint_validation --test optimizer_memory -- --test-threads=1
cargo run --release --example train_deploy
```

Practice answering: “If the last tensor is wrong, what changed?”, “Can I
resume halfway through an accumulation window?”, “Why does this file load
into differently padded inference?”, and “Does the 8 MiB figure mean the
driver's peak fell by 8 MiB?” The answers are respectively: no checkpoint
state mutation; not from this snapshot alone; matching logical identity;
and no, the checked quantity is retained buffer requests.
