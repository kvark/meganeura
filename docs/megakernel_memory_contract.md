# Device graph memory contract — investigation

Milestone 1 is complete. This is the candidate protocol for Milestone 2;
compiler-output checks and bounded device tests must qualify it before Session
uses it. The initial target is Vulkan with its memory model enabled.

## Smallest candidate interface

Retain the fork's existing `enable wgpu_memory_fence;`, `storageFence()` and
`workgroupFence()`. Leave ordinary WGSL atomics relaxed. No new ordering enum,
scope argument, atomic expression or compare-exchange semantics is needed by
the protocol below. Directional fences or ordered atomics remain alternatives
only if measurements demonstrate a useful improvement.

At Naga `a489554`, `storageFence()` emits a QueueFamily-scope acquire/release
memory barrier with UniformMemory, MakeAvailable and MakeVisible under the
Vulkan memory model. Storage accesses carry NonPrivatePointer. The barrier
does not wait for other workgroups. This scope covers a single dispatch.
Standard WGSL synchronization remains workgroup-scoped and its atomics remain
relaxed ([WGSL memory model](https://gpuweb.github.io/gpuweb/wgsl/#memory-model)).

The argument uses fence-to-fence synchronization through relaxed atomic RMW
release sequences, and fence/control-barrier/fence synchronization within a
workgroup. These are rules in the
[Vulkan memory model](https://docs.vulkan.org/spec/latest/appendices/memorymodel.html#memory-model-synchronizes-with).
Availability and visibility are handled explicitly by **every payload lane**;
ordering a leader's atomics alone is insufficient.

## Publication by all lanes

After a cooperative tile body finishes:

1. Every lane executes `storageFence()`, publishing its own storage writes
   directly to the queue-family domain.
2. All lanes execute `workgroupBarrier()`.
3. The leader executes `storageFence()` before decrementing the operation's
   remaining-tile counter with a relaxed atomic RMW.

For any writing lane, its first fence is a release before the workgroup
control barrier; the leader's fence is an acquire after that barrier. Both
fences cover storage. This orders the lane's writes before the leader's
completion signal. Each writing lane already made its writes available at the
final domain; the proof does not ask the leader to publish other lanes' writes.

The last tile leader, identified by a decrement returning one, executes
`storageFence()` after its RMW. Every earlier tile completion is a contiguous
RMW on the same counter, so the last read acquires all their release sequences.
It then decrements each successor's predecessor counter, fencing before each
signal and after a decrement returning one. The sole winner publishes that
successor's tiles. Counters are separate from tensor storage and cannot be
aliased with it. Multiple edges from the same predecessor are either deduplicated
or counted and decremented with identical multiplicity.

## Bounded ready queue and acquisition

The first protocol uses an indexed ready bitmap over immutable task slots.
This avoids a partially published FIFO reservation and a full queue that blocks
all producers. Capacity equals the statically known tile count; each slot is
published and claimed at most once during a dispatch. A slot identifies the
handler, tile, buffer references and successor metadata. Root bits are set by
the ordered reset/initialization before the worker launch.

To publish a ready task, its leader executes `storageFence()` and then relaxed
`atomicOr` on the task's bit. A consumer leader loads a bitmap word, chooses a
set bit and uses relaxed `atomicAnd` to clear it. The returned old value must
contain that bit; otherwise it lost the claim and continues searching. Only
RMW operations modify the bitmap during a dispatch. The winning RMW reads
from the hypothetical release sequence headed by the publication RMW, even
when other bits in the word were modified in between.

After winning, the consumer leader executes `storageFence()` and broadcasts
the selected immutable task through workgroup storage and a uniform workgroup
barrier (`workgroupUniformLoad`). **Every consumer lane then executes
`storageFence()` before reading tensor data.** The leader's fence, workgroup
control barrier and each lane's fence extend storage ordering to every reader.
Each reader performs its own visibility operation directly from the queue-family
domain. No availability/visibility chain through a leader is required.

Blade currently leaves `vulkanMemoryModelAvailabilityVisibilityChains`
disabled. This protocol intentionally does not depend on multi-element
availability or visibility chains; those are a separate
[Vulkan feature](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceVulkanMemoryModelFeatures.html).
The ordering chain and the availability/visibility operations serve different
purposes and must not be conflated.

## Progress, completion and reuse

A worker searches the finite bitmap for any ready task. It never waits for a
particular predecessor or slot. A worker that finds no claimable work may retire;
an empty observation is **not** global completion. A completing worker publishes
successors before it searches again. If another worker takes its newly published
work, that worker continues the chain. Otherwise the publisher can claim it
itself. Thus currently executing producers sustain progress as idle workers
retire; unscheduled workers are not needed to satisfy a global rendezvous.
This assumes normal execution progress of scheduled workgroups and finite tile
bodies, not simultaneous residency of all launched workgroups.

Every successful claim executes exactly one task and decrements a total remaining
count after publishing its consequences. Host validation after GPU completion
requires zero remaining work, expected operation counters, and no diagnostic
error. Underfilled completion is an error, never successful graph execution.
Diagnostic tests impose explicit task/scan bounds and report the first failed
invariant instead of allowing an unbounded spin.

The graph must be acyclic after adding RAW/WAR/WAW and physical-reuse edges.
Task counts, tile counts, predecessor counts, bitmap sizes and buffer ranges
are checked in wider host arithmetic before allocation. Counts fit u32 and
cannot underflow; a bitmap has no monotonic ticket or sequence counter to wrap.
Empty graphs need no worker launch. Unsupported capacity or binding requirements
produce an explicit region split or rejection.

Slots are reused only between completed dispatches, after the caller's tracked
submission has finished and GPU reset precedes the next launch. Reset clears
bitmap/counters/diagnostics, not tensors, optimizer state or caches. There is no
concurrent host enqueue and no within-dispatch slot recycling. A general reusable
ring would require another protocol and is outside this initial contract.

## Compiler and backend audit

- Parsing already requires the native extension and can reject it using frontend
  capabilities. IR validation currently does not enforce MEMORY_FENCE for
  `Statement::MemoryBarrier`; clients that parse with permissive frontend options
  can miss the backend restriction. Add the corresponding validator check.
- SPIR-V has distinct memory/control barriers and relaxed load/store/RMW/CAS
  paths. Check generated scopes, semantics and NonPrivatePointer operands with
  SPIRV-Tools, including the original ordinary-atomic fixture. Preserve these
  operations through WGSL round trips, SPIR-V import and transformations.
- MSL currently routes memory and control barriers through the same writer,
  despite not advertising MEMORY_FENCE. Reject unsupported memory fences
  explicitly instead of silently inserting a workgroup rendezvous.
- The June 2026 [Metal language specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf),
  section 6.16.3, describes `atomic_thread_fence` on Apple silicon with MSL 3.2+.
  That is a plausible later implementation of the existing fence API, requiring
  target gating and Metal execution qualification. It does not require new
  atomic syntax. Older targets need a clear rejection.
- GLSL/HLSL retain their existing fence writers; their cross-workgroup resource
  coherency requirements and execution behavior are not qualified here. Blade
  must expose this device-graph contract only for a supported Vulkan memory-model
  path, with unsupported backends rejected before graph execution.

The existing interface requires no parser/IR serialization expansion and no
blanket change to atomic ordering. The next evidence is compiler tests followed
by all-lane message passing, counter fan-in, bitmap publication/claim, repeated
resets and bounded termination under scheduling pressure. Stress success alone
will not replace this ordering and progress argument.
