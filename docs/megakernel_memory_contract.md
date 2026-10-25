# Device graph memory contract

Milestone 1 is complete. Milestone 2 selects the existing native fence extension
for the bounded protocol below. The whole-graph path now preserves descriptor
identity across types and slices; the regional path still needs that repair.
Passing device tests alone does not establish the memory-model contract: the
publication argument also requires the shared-reference condition below.

## Descriptor-reference audit and repair

The Vulkan [reference definition](https://docs.vulkan.org/spec/latest/appendices/memorymodel.html#memory-model-reference)
identifies a storage-buffer reference with its descriptor, including the array
element. Shader-only write/read and write/write
[location ordering](https://docs.vulkan.org/spec/latest/appendices/memorymodel.html#memory-model-location-ordered)
requires the same reference. Distinct descriptors pointing into the same physical
buffer do not satisfy that condition. Coherent accesses and shader fences alone
do not repair it; ordering through the device domain requires API synchronization.

At `c4a9cc0`, whole-graph tables deduplicate `(buffer, offset)` independently for
each storage type. A SmolVLA audit observes one buffer shared across typed tables
in forward execution; training has 37 cross-table occurrences and 282 occurrences
of a buffer already present at a different offset. These counts identify possible
reference aliases, not exact overlapping access ranges. Regional scheduling also
needs review because handlers can have distinct descriptors for shared storage.

The repair uses one descriptor at offset zero per backing buffer. Node offsets
become shader parameters, and typed declarations map to the same descriptor
binding and array index. Vulkan explicitly permits these
[typed binding aliases](https://docs.vulkan.org/spec/latest/chapters/interfaces.html#interfaces-resources-setandbinding).
Their SPIR-V declarations need `Aliased`; Naga's access analysis must account for
writes through every declaration mapped to the same descriptor. Ordinary WGSL
validation remains intact: aliasing is an explicit native backend mapping.

Historical measurements remain hardware observations, not portable correctness
qualification, even after the repaired implementation passes new tests. In
particular, read-only and visibility-removal prototypes are not eligible for
acceptance based on their timings. Their coherent cross-binding treatment is
superseded by the shared-reference requirement.

## Smallest candidate interface

Retain the fork's existing `enable wgpu_memory_fence;`, `storageFence()`,
`workgroupFence()` and `@coherent` storage decoration. Leave ordinary WGSL
atomics relaxed. No new ordering enum,
scope argument, atomic expression or compare-exchange semantics is needed by
the protocol below. Directional fences or ordered atomics remain alternatives
only if measurements demonstrate a useful improvement.

At Naga `a489554`, `storageFence()` emits a QueueFamily-scope acquire/release
memory barrier with UniformMemory, MakeAvailable and MakeVisible under the
Vulkan memory model. Participating storage accesses must carry NonPrivatePointer. The barrier
does not wait for other workgroups. This scope covers a single dispatch.
Standard WGSL synchronization remains workgroup-scoped and its atomics remain
relaxed ([WGSL memory model](https://gpuweb.github.io/gpuweb/wgsl/#memory-model)).

The argument uses fence-to-fence synchronization through relaxed atomic RMW
release sequences, and fence/control-barrier/fence synchronization within a
workgroup. These are rules in the
[Vulkan memory model](https://docs.vulkan.org/spec/latest/appendices/memorymodel.html#memory-model-synchronizes-with).
Availability and visibility are handled explicitly by **every payload lane**;
ordering a leader's atomics alone is insufficient.

The repaired SPIR-V writer groups storage globals by their final mapped
descriptor set and binding. Multiple declarations in a group receive `Aliased`.
Declared access flags and actual write/atomic uses are combined across the
reachable declarations. This prevents a read-only declaration from acquiring
`NonWritable` or private-access assumptions when another declaration writes the
same descriptor. An entry point that has no such writes retains the immutable
optimization. Ordinary IR binding validation is unchanged; native remapping is
explicit in the existing backend binding map.

Whole-graph typed tables map to one Blade buffer-array binding. Every backing
buffer appears exactly once at offset zero; all views use its canonical array
index. Original binding offsets are converted to element offsets and passed with
that index through uniformly loaded node parameters. The converter requires
directly indexed storage arrays and checks offset alignment and representability.
Tensor accesses retain `NonPrivatePointer` when any reachable alias is written,
and use the lane-level fences below for availability and visibility. They do
not need redundant per-access propagation. Storage-pointer helpers remain
rejected because pointer provenance is not preserved by the current converter.

The regional backend still uses distinct handler bindings and coherent accesses.
Its cross-descriptor publication remains unqualified; those annotations do not
satisfy the same-reference requirement. The repaired whole-graph contract must
not be attributed to that older path.

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

The original regional queue below permits idle workers to retire. The optional
[whole-graph pool](megakernel_whole_graph.md) instead keeps workers polling until
the remaining-node count reaches zero; its bitmap contains ready nodes and each
node has a tile-claim counter. It uses the same lane-level publication/visibility
contract and orders physical allocation conflicts across every node, including
optimizer phases. It does not add a global workgroup barrier.

A worker searches the finite bitmap for any ready task. It never waits for a
particular predecessor or slot. Session workers make at most 64 unsuccessful
scan rounds before retiring, allowing short producer/consumer handoffs to retain
workers. A zero total-remaining count permits an earlier exit. Each round uses a
uniform broadcast, so every lane agrees on the claim or retirement. These fixed
retries extend the initial protocol's single scan; they do not wait indefinitely
for another workgroup or require simultaneous residency.
A worker that finds no claimable work may retire;
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

- Parsing requires the native extension and can reject it using frontend
  capabilities. IR validation now also enforces MEMORY_FENCE for
  `Statement::MemoryBarrier`, including clients that parse with permissive
  frontend options.
- SPIR-V has distinct memory/control barriers and relaxed load/store/RMW/CAS
  paths. Check generated scopes, semantics and NonPrivatePointer operands with
  SPIRV-Tools, including the original ordinary-atomic fixture. Preserve these
  operations through WGSL round trips and transformations. The SPIR-V importer
  cannot preserve NonPrivatePointer/MakePointerAvailable/MakePointerVisible
  operands; it now rejects those accesses explicitly. Basic Vulkan-model
  atomic-only inputs remain accepted. Import of arbitrary scope/ordered-atomic
  semantics is not qualified; the scheduler uses WGSL-to-SPIR-V export.
- MSL previously routed memory and control barriers through the same writer,
  despite not advertising MEMORY_FENCE. It now rejects memory fences explicitly
  instead of silently inserting a workgroup rendezvous.
- The June 2026 [Metal language specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf),
  section 6.16.3, describes `atomic_thread_fence` on Apple silicon with MSL 3.2+.
  That is a plausible later implementation of the existing fence API, requiring
  target gating and Metal execution qualification. It does not require new
  atomic syntax. Older targets need a clear rejection.
- GLSL/HLSL retain their existing fence writers; their cross-workgroup resource
  coherency requirements and execution behavior are not qualified here. Blade
  exposes `device_memory_fence` only for its Vulkan memory-model path. This gates
  native fence/coherent validation capabilities; other backends reject shaders
  requiring them before execution.

## Interface decision and compatibility

| Interface | Required changes | Decision |
|---|---|---|
| Existing fences and relaxed atomics | Canonical descriptor references, alias-aware access flags, capability validation and protocol checks | Selected for the repaired whole-graph path |
| Directional/scoped fences | New WGSL arguments or builtins, IR representation, validation of order/scope/storage combinations, updates to serialization and every reader/writer | Potentially cheaper than acquire/release at every fence, but no demonstrated requirement yet |
| Ordered/scoped atomics | Changes across atomic loads, stores, RMW and compare-exchange, including separate CAS failure constraints; equivalent reader/writer and serialization work | Does not remove the need to publish/acquire every lane's non-atomic payload; defer until a measured benefit justifies it |

There is no new parser grammar, IR variant, reflection field or serialized field.
Ordinary atomic load/store/RMW/CAS semantics remain relaxed, including both CAS
outcomes. The queue uses OR/AND and decrements, so it needs no CAS ordering API.
Naga compaction retains fence statements and their order; constant evaluation
does not evaluate atomic operations. Tests cover compaction followed by a WGSL
round trip and inspect the resulting SPIR-V. No new optimization pass moves
loads/stores across fences. Typed aliases retain descriptor identity and their
shared access analysis through shader composition; storage-pointer helpers
remain outside the accepted path.

The compatibility changes are deliberate errors for previously unqualified
paths: IR without the fence capability, Metal fence emission, and SPIR-V import
that would discard the required access operands. Vulkan WGSL export keeps the
existing emitted fence and atomic instructions. Unsupported targets can still
run ordinary Session execution or the portable Milestone 1 regions.

The existing fences are broad acquire/release operations. This investigation
establishes sufficiency, not an optimal fence count or a speedup. Milestone 1's
[paired production measurements](megakernel_session_results.md) already show
regressions; ordinary execution stays the default. The same paired harness must
measure actual Session device scheduling before any adoption decision. The
protocol stress runs below are correctness checks, not comparative benchmarks.

## Descriptor repair validation, October 25

The repair uses Naga
[`bc7b70b`](https://github.com/kvark/wgpu/commit/bc7b70bc5d283202e1ec904dd244489f2fe0366a)
and Blade
[`c09c89f`](https://github.com/kvark/blade/commit/c09c89f79681f879ef37b28125b914be45fc1726).
Blade's `BufferAlias` explicitly names an earlier buffer or buffer-array binding;
aliases share its descriptor count and update storage. Invalid targets and alias
chains are rejected. Other backends do not implement this native layout option.

- Naga passes 144 unit, 269 integration and six documentation tests (24 ignored),
  plus strict Clippy. The alias fixture checks combined and selected entry points,
  private-access preservation for immutable inputs, `Aliased`/`NonWritable`, and
  NonPrivatePointer. All six generated variants pass Vulkan 1.3 `spirv-val`.
- Blade passes 21 unit tests (one ignored) and strict Clippy. Meganeura passes
  665 library tests (eight ignored), all 168 strict whole-graph oracle tests,
  eight focused GPU tests, and strict all-feature/all-target Clippy.
- The GPU protocol fixture now reads `vec4<u32>` through a typed alias of its
  scalar payload binding. Its 256 dispatches cover every lane, fan-in/fan-out,
  ragged tiles, reset epochs and 1/3/64/4096 workers. The whole-graph optimizer
  fixture covers nonzero arena offsets, large and small chunks sharing handlers,
  and repeated recorded steps.
- The required full Vulkan workspace run executes 1,152 tests: 1,144 pass,
  eight fail, and six are skipped. All eight failure identities match the saved
  pre-change baseline: three passthrough tests need `dxc`, and five depth/stencil
  initialization tests fail. No new failure identities appear. This run selects
  the NVIDIA Vulkan ICD and serializes tests.

The initial Naga suite invocation lacked validation tools on `PATH`; rerunning
with the installed tools passes. Its SPIRV-Cross version changes five generated
GLSL snapshots, including integer-dot translation errors; the corresponding
SPIR-V assembly is unchanged and those generated GLSL changes are discarded.
The initial new fixture omitted `enable wgpu_binding_array`; that error and
compile errors in tests referring to the removed view-write field were fixed
before the successful runs above.

Logs and the comparison of workspace failures are local under `target/alias-*`.
Every build, test and analysis job uses the verified bounded cgroup launcher.
These checks qualify the repaired whole-graph implementation on the tested
device; they do not qualify the regional backend or the old timing prototypes.

## Historical qualification evidence (descriptor audit supersedes the proof)

The original tested pins were Naga
[`f09d62e`](https://github.com/kvark/wgpu/commit/f09d62e54d743f89879e4e8dfb32e717146f2670)
and Blade
[`63fb651`](https://github.com/kvark/blade/commit/63fb651a3d0c7bad6964dba9b2585f794641862f).
Meganeura's runtime tile conversion, developer tools and Blade share that single
Naga identity.
The subsequent Blade pin `5011dea27d0df487196ab279ef53a93aaef2607b` corrects the
binding-group ceiling for broader handler coverage; its memory-fence capability
and Naga dependency are unchanged.
The published pins pass 506 Meganeura unit tests (seven ignored), 71 smoke tests
in each ordinary/persistent selection (two ignored), strict all-target Clippy,
formatting and the lavapipe protocol check. Blade passes 20 unit tests (one
ignored), strict Clippy and Metal-target compilation, with existing target
warnings and no Metal GPU qualification.

The compiler fixture emits SPIR-V 1.3 with the Vulkan memory model. `spirv-as`
and `spirv-val --target-env vulkan1.2` accept it. The inspected output has:

- QueueFamily scope (5), acquire/release + UniformMemory + MakeAvailable +
  MakeVisible semantics (24648) for storage fences;
- relaxed atomic load/store/RMW and CAS success/failure semantics (0), with
  QueueFamily storage scope and Workgroup workgroup scope;
- NonPrivatePointer on mutable storage accesses and queue-family
  MakePointerVisible on the coherent read-only alias;
- VulkanMemoryModel capability, without VulkanMemoryModelDeviceScope.

The GPU check is [device_protocol.rs](../tests/device_protocol.rs), with its
[WGSL protocol](../tests/shaders/device_protocol.wgsl). Both NVIDIA RTX 5080
(595.91.07) and lavapipe passed 256 bounded dispatches: empty work, a 64-node
chain with ragged tiles, 65-way fan-in followed by fan-out, and eight fully
connected layers. Each uses 1/3/64/4096 workers and 16 reset epochs with changing
inputs. The test compares every lane's payload against an independent integer
reference, checks every task executes once, all readiness/dependency/tile
counters drain, and worker totals match. The empty dispatch intentionally
exercises worker retirement; production empty graphs need no launch. Each wait
has a 30-second timeout and the shader's scan/task loops have finite bounds.
Vulkan synchronization validation was enabled. No Metal execution was performed.

The checks supplement the ordering and progress argument above; successful
stress runs or validation layers alone do not establish that argument.

Naga's focused compiler suite passed 144 unit tests, 267 integration tests and
six documentation tests (24 ignored). Pinned-toolchain Clippy and formatting
passed. The required `cargo xtask test` workspace run executed 1,568 tests:
1,324 passed, 244 failed and 12 were skipped. All 244 failed test identities
reproduced on the unchanged `a489554` baseline, with no additional failures:

- 235 OpenGL tests encounter GLSL 330 capability errors;
- three Vulkan passthrough tests require the unavailable `dxc` executable;
- five Vulkan depth/stencil initialization tests fail;
- `player::player tests::test_api` fails.

The complete failure identities and logs are retained locally under
`target/megakernel-results/`: `wgpu-failed-tests.json`,
`naga-fences-baseline-failures.log`, and `naga-fences-workspace-qualified.log`.
The five depth/stencil failures are
`discarding_both_depth_and_stencil_with_diverging_load_ops`,
`discarding_depth_target_at_nonzero_mip_and_layer_depth32float_stencil8`,
`discarding_either_depth_or_stencil_aspect_at_nonzero_mip_and_layer`,
`discarding_depth_target_at_nonzero_mip_and_layer`, and
`render_pass_store_inits_only_target_depth_stencil_subresource`.
No CTS run was requested or performed. SPIRV-Cross tool-version differences
regenerated five unrelated GLSL snapshots; these were restored, and only the new
publication fixture's output was retained.

GPU protocol logs are `device-protocol-nvidia.log` and
`device-protocol-lavapipe.log` in the same ignored results directory. Reproduce
the protocol with the normal pinned dependencies:

```sh
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json \
VK_LAYER_VALIDATE_SYNC=1 \
systemd-run --user --scope --slice=ml.slice \
  cargo test --test smoke device_protocol --all-features -- --nocapture --test-threads=1
```

Use `lvp_icd.json` for lavapipe. An unsupported capability is reported explicitly
by this test, so its exit status alone is not evidence of backend qualification.
