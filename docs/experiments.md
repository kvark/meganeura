# Experiments: source refs and conclusions

Keep experiment implementations on an `experiment/*` branch or immutable tag.
Git records their base and changes; main needs the question, result, limitation
and reproduction command, not another source snapshot or artifact archive.
This follows Blade's rendering branches such as
[`experiment/merge-gbuffer-pass`](https://github.com/kvark/blade/tree/experiment/merge-gbuffer-pass)
and [`experiment/unfused-restir`](https://github.com/kvark/blade/tree/experiment/unfused-restir).

For a new experiment, commit the runner, configuration and dependency lock on
its branch before collecting data. Tag the measured source. Write outputs to
ignored `results/` or outside the checkout. Do not commit binaries, traces,
raw dumps, compressed dumps or evidence-replay tests to main. If exact observed
samples are needed for publication review, retain them outside Git as a
separate research artifact; rerunning a revision reproduces the procedure, not
the original timing noise. Record the environment and failure counts with the
conclusion. Adopt production changes independently of experiment scaffolding.

For the later NVIDIA/Intel attention, matrix and e-graph investigation, see
[GPU gap, September 21](gpu-gap-2026-09.md). Reproducers remain on its experiment
branch; the proposed production changes contain no raw sweep or trace archives.

The [October 5 Intel follow-up](gpu-gap-2026-10.md) profiles the current code,
batches scalar attention-gradient reductions, and retains repeated comparisons
on Arc B570 and RTX 5070. Its results are separate from the paper cohort.
The October 6 continuation adds egglog backward-layout exploration and checks
its coverage within the existing 60-second training search budget.

## Workgroup cooperative matrices — October 15

The historical prototype measured RTX 5070 gains from
[`VK_NV_cooperative_matrix2`](https://docs.vulkan.org/features/latest/features/proposals/VK_NV_cooperative_matrix2.html).
Workgroup scope lets several subgroups participate in one matrix operation;
flexible dimensions allow larger logical tiles than the existing subgroup
shapes. The local B570 driver does not expose this extension. This is a candidate
for measured selection in the accelerated path, not a universal replacement.

These are pre-#257 measurements, not performance claims for the current branch.
In particular, the BF16 baseline used scalar checked staging. #257 replaced it
with the shared vectorized kernel on native 16×16×16 devices. The historical
2.7×/3.9× BT/AT ratios below must not be quoted as gains over current main;
the October 16 measurements below use the new incumbent and normal race search.

All three repositories publish the prototype on `experiment/workgroup-matrix`:

| Repository | Published branch | Base |
|---|---|---|
| Meganeura | [experiment/workgroup-matrix](https://github.com/kvark/meganeura/tree/experiment/workgroup-matrix) | `f648a4534a87e29ac29810a88dfa19117f70df9b` |
| Blade | [experiment/workgroup-matrix](https://github.com/kvark/blade/tree/experiment/workgroup-matrix) | `96c1dc731b1d5e133a6c0a8a020014619ff2e434` |
| Naga/WGPU | [experiment/workgroup-matrix](https://github.com/kvark/wgpu/tree/experiment/workgroup-matrix) | `a97559b6318dc0a857ace502ffc7f3e442951f39` |

The experiment checkpoint pinned [Blade 457e685](https://github.com/kvark/blade/commit/457e68505bfc47fbd56533b011dd69be21784136)
and [Naga c32067f](https://github.com/kvark/wgpu/commit/c32067f08c27ba70d6da40a08049ba4268052a6f).
The experiment branches include dependency locks and build without sibling
checkouts using `--locked`. The production integration retains main's ignored
`Cargo.lock` policy; those locks belong to the historical experiment only.

The measurements below were collected before these commits. Their source is
frozen as patches, new files, dependency locks and binary hashes in
`/x/Code/inferena-results/workgroup-matrix-20261015/final-manifest.json`;
there is no measured Git tag. Publication replaces relative path dependencies
with exact Git revisions and updates these notes. Kernel code matches the
frozen prototype; dependency versions are preserved. Raw results remain outside
Git.

Naga adds an optional scope argument, for example
`coop_mat64x64<f32, C, workgroup>`, carrying it through validation, IR and
SPIR-V. Existing syntax retains subgroup scope. Workgroup shaders require
SPIR-V 1.6 and `SPV_NV_cooperative_matrix2`; unsupported Metal emission is
rejected. The prototype also fixes expression emission before `coopStore`,
which previously failed on a dynamic pointer/stride expression.

Blade queries/enables the workgroup-scope and flexible-dimension features,
records operand types, dimension granularities, required invocation counts
and reserved shared memory, and selects SPIR-V 1.6 only for workgroup shaders.
The RTX advertises up to 256 invocations, dimensions up to 1024, and 8192 bytes
of reserved shared memory. Other matrix2 features, including tensor addressing
and reductions, are not enabled or implemented here.

Meganeura adds an opt-in kernel variant to the existing MatMul/MatMulBT/MatMulAT
shader groups. F32 storage is staged through F16 workgroup arrays; BF16 storage
loads directly; accumulation and output stay F32. Selection requires explicit
`CoopPolicy::AllowF16` plus `SessionOptions.workgroup_matmul`. Device support,
shared-memory capacity and complete aligned tiles are checked. Fused, batched,
split-K, reduced-weight and BF16-output products remain on their existing paths.
Saved workgroup choices are reset before device selection; the prototype uses
cache format 32. Default and Auto selection are unchanged.

Measurements use RTX 5070 / NVIDIA 595.91.07, Arc B570 / Mesa
26.0.3-1ubuntu1, Linux 7.0.0-38-generic and Rust 1.98.0 on the same host.
A 120-case exploratory sweep tried ten tiles on 512³ and 1024³ products.
The confirmation uses one fixed tile, M=N=64, K=32, 64 invocations, across
five shapes, three layouts and two storage modes. Three fresh processes give
90 records, including 18 records where ragged dimensions retain the baseline.
The two rectangular shapes and ragged shape were not used to select the tile.

Each process compares sessions with the option disabled/enabled in the same
binary, with tuning disabled. It warms each variant for 500 ms, then alternates
ABBA/BAAB to collect 20 samples per variant; the second process reverses the
starting order. Each sample records 64 session replays, submits and waits,
then divides wall time by 64. BF16 timings include both explicit input-conversion
dispatches. Compilation, allocation and correctness readback are outside the
samples. Ordinary driver caches are retained; clocks are not locked. A 500 ms
NVIDIA telemetry trace is retained, and no builds or other benchmark jobs ran
concurrently. Each process has a 300-second watchdog. These are warm execution
microbenchmarks, not cold-start or whole-model results.

Times are medians of three process medians; speedups are medians of the three
paired ratios. NN is A×B; BT and AT transpose B and A respectively.

| M × N × K | Operands | Layout | Historical baseline → workgroup, ms | Historical paired speedup (range) |
|---|---|---|---:|---:|
| 1024 × 1024 × 1024 | F32 storage, F16 multiply | NN | 0.14783 → 0.08379 | 1.760× (1.757–1.764) |
| 1024 × 1024 × 1024 | F32 storage, F16 multiply | BT | 0.16477 → 0.10732 | 1.536× (1.535–1.537) |
| 1024 × 1024 × 1024 | F32 storage, F16 multiply | AT | 0.16599 → 0.14031 | 1.183× (1.182–1.183) |
| 1024 × 1024 × 1024 | BF16 | NN | 0.16281 → 0.12646 | 1.288× (1.287–1.289) |
| 1024 × 1024 × 1024 | BF16 | BT | 0.38346 → 0.14210 | 2.698× (2.695–2.700) |
| 1024 × 1024 × 1024 | BF16 | AT | 0.37981 → 0.09658 | 3.932× (3.929–3.942) |
| 128 × 1536 × 576 | BF16 | AT | 0.06436 → 0.03980 | 1.617× (1.616–1.628) |
| 128 × 576 × 576 | BF16 | NN | 0.03502 → 0.03924 | 0.897× (0.892–0.908) |
| 512 × 512 × 512 | F32 storage, F16 multiply | AT | 0.04381 → 0.04591 | 0.957× (0.954–0.958) |

The full 30-case table, pipeline keys and ranges are in the artifact's
`analysis.json`. Gains depend on shape/layout; the two negative rows above
correspond to roughly 11.4% and 4.5% slower paired execution. The 128×576×576
F32-storage baseline selects scalar F32, unlike the other F32-storage rows;
its gains must not be presented as a comparison at identical multiply precision.
Ragged cases use identical pipelines, with median ratios 0.999–1.006.

All 120 exploratory and 90 confirmation records pass full-output F64 checks
with exactly representable inputs; observed error is zero. A separate frozen
quality binary passes 96 cases per GPU, checking 3,492,096 output values each,
with Vulkan validation enabled and poisoned allocations. It covers NN/AT/BT,
aligned/ragged shapes, two feeds, nonrepresentable F32 inputs and BF16 values
spanning 10⁻⁹ to 10⁹. Its oracle explicitly rounds operands to the selected
precision, then accumulates in F64. Maximum relative L2 errors against that
oracle are 7.21e-7 on RTX and 4.33e-7 on B570. Thirty-six RTX records select
workgroup matrices; none on B570 or under NativeF32 do. This establishes kernel
arithmetic, not model acceptance against unrounded F32 or training convergence.

Naga scope/version/roundtrip/rejection tests, Blade capability tests, Meganeura
selection/cache tests and 39 codegen tests pass. Clippy and formatting pass for
the affected crates. Full `cargo xtask test` runs 1,139 tests: 1,131 pass and
eight fail, with six skipped. All eight reproduce on the unmodified pinned
Naga/WGPU fork: the three shader-passthrough tests and five depth/stencil
zero-initialization tests. Exact names and logs are retained in
`wgpu-failed-tests.json`, `wgpu-full-test2.log` and `wgpu-control-failures.log`.
Initial missing nextest/SPIR-V tools were supplied locally; tool-version-only
SPIRV-Cross snapshot changes were archived and removed from the source diff.

Before publication, rebuilding with the exact Git dependencies and locked
versions passed the Naga and Blade capability tests, five Meganeura workgroup
tests and 42 codegen tests. The rebuilt quality example repeated all 96 cases
on each GPU, with 36 workgroup selections on RTX and zero on B570, the same
maximum errors, and no Vulkan validation output.

The PR #3 review follow-up registers the missing SPIR-V capabilities and adds
WGSL/SPIR-V snapshots, including BF16 128×256 tiles. Matrix dimensions now use
positive `u32` values instead of a fixed enum ending at 128. Naga rejects
workgroup load/store addresses and strides that depend on individual lanes,
tracking scalar locals and function arguments; subgroup operand uniformity
remains a separate limitation. Store-expression availability has its own
regression test and follow-up commit. Scope mismatch diagnostics identify the
actual operand.

Blade compacts each selected entry point before choosing its SPIR-V version.
It validates dimensions, invocation counts, a common device configuration for
each product, and user plus reserved shared memory after resolving overrides.
The memory bound includes padding between differently aligned variables and
32-bit boolean storage, following
[Vulkan's workgroup storage rules](https://docs.vulkan.org/spec/latest/chapters/shaders.html).
These device checks live in Blade, which has the queried Vulkan properties.
Meganeura accepts larger eligible tiles and checks shared-memory arithmetic for
overflow. Default/Auto selection remains unchanged.

Review validation passes all 411 Naga tests/doctests, 15 Blade tests, 43
Meganeura codegen tests and six workgroup-related tests; Clippy and formatting
also pass. `cargo xtask validate spv` passes, as do CLI-generated workgroup,
BF16 and compacted vertex/compute binaries under `spirv-val --target-env vulkan1.3`.
The full WGPU run across all local adapters has 2,145 passing and 253 failing
tests (24 skipped). Its failure set exactly matches the previously tested,
unchanged `2f6069d` base: 236 OpenGL/player initialization failures, six Intel
Vulkan, eight NVIDIA Vulkan and three llvmpipe Vulkan failures.

The final quality checks again pass 96 cases on each GPU, with the same maximum
errors and no Vulkan validation output. Another 96 RTX cases exercise 128×256
tiles: K=32 for staged F16, K=64 for direct BF16, with 256 invocations. Twenty-four
records select those large tiles; the rest check strict or ragged fallbacks.
All 5,514,240 output values pass the rounded-operand F64 oracle, with maximum
relative L2 error 2.04e-7. This is kernel correctness evidence, not a new speed
measurement or a model acceptance gate. Logs, exact failed test names, binary
hashes and summaries are in
`/x/Code/inferena-results/workgroup-review-20261015/`.

Reproduce from a checkout of the Meganeura experiment branch, one GPU job at a time:

```sh
cargo build --locked --release --example workgroup_probe --example workgroup_quality
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json TILE=64,64,32,64 \
  timeout 300s target/release/examples/workgroup_probe > /tmp/workgroup-pair-1.jsonl
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json TILE=64,64,32,64 REVERSE=1 \
  timeout 300s target/release/examples/workgroup_probe > /tmp/workgroup-pair-2.jsonl
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json TILE=64,64,32,64 \
  timeout 300s target/release/examples/workgroup_probe > /tmp/workgroup-pair-3.jsonl
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
  timeout 300s target/release/examples/workgroup_quality > /tmp/workgroup-quality-rtx.jsonl
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/intel_icd.json \
  VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
  timeout 300s target/release/examples/workgroup_quality > /tmp/workgroup-quality-b570.jsonl
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
  timeout 300s target/release/examples/workgroup_quality --large-tile > /tmp/workgroup-quality-large.jsonl
```

At this prototype checkpoint, selection still forced an explicitly requested
tile. The measured integration below removes that restriction from normal search.

### Measured rectangular integration — October 16

`feat/rectangular-cooperative-search` combines the typed-matmul branch with the
workgroup prototype. Workgroup products now take `MatrixTypes`, share the ordinary
NN/AT/BT entries (including addends), and carry their shape and partition count
through pipeline identity, dispatch geometry, private-scratch qualification,
timing, memo persistence and rollback. The bounded space includes 32×64, 64×32,
64×64, 64×128, 128×64, 128×128, 128×256 and 256×128 with K=32/64. Device
granularity, invocation counts and user plus reserved shared memory filter it.

Matching F16/BF16 operands load directly. F32 operands stage in the requested
multiplication type; F32 accumulation precedes the addend and any output rounding.
Candidates keep the incumbent's arithmetic: an automatic F16 incumbent can vary
its tile, but an F32 incumbent is not silently narrowed. Split products are
qualified and timed with their SumRows reduction. Ragged tiles, incompatible
prologues/epilogues, horizontal packs and unsupported devices keep existing paths.
This does not add a benchmark profile or relax model acceptance thresholds.

Kernel memo revision 7 and plan format 33 invalidate the earlier experimental
encodings and main's revision-6 decisions. Cached automatic choices fall back
under the new session's precision policy; explicit typed graphs retain their
types. `SessionOptions::workgroup_matmul` remains a diagnostic forced-tile
control; leave it unset for measured selection.

Integration found a Blade entry-point regression: Naga revalidated unbound
resources before Blade had assigned their pipeline bindings. The separate
`fix/entry-point-unbound-resources` commit `b6af80c` supplies temporary bindings
during that validation and removes them afterwards, preserving real bindings.
The integration now pins the merged [Blade 4845d65](https://github.com/kvark/blade/commit/4845d6529c38b07718a624bf3c4f2ab4317c9bf4)
and matching [Naga 13f582d](https://github.com/kvark/wgpu/commit/13f582ddd2792befb46952498010248bd65818e6),
including the workgroup, entry-point and unified BF16-capability changes.

Before the #257 rebase, validation included 2,304 generated SPIR-V variants,
356 qualified workgroup comparisons on RTX 5070, exact full-output checks before
and after tuning, and the B570 unsupported-extension fallback. The ordinary
regression suite passed on both devices (152 tests each, with 18 existing skips). These runs include
typed F16/BF16, NN/AT/BT, addends, rounded BF16 output and complete split-K results.
Logs and measurement receipts are under
`/x/Code/inferena-results/rectangular-search-20261016/`.

Six Inferena diagnostics also pass: strict and accelerated SmolLM2-135M,
and accelerated Whisper-tiny, on both B570 and RTX 5070. Each session starts
with an empty kernel memo and a 30-second search budget; phase warmup is two
seconds with eight timing samples, so these are not replacement cohort records.
They retain Inferena's existing full-tensor output/gradient gates. RTX SmolLM2
actually measures 52 workgroup comparisons across its searches, all qualified,
and selects rectangles including 64×128, 128×64 and 32×64. A later kernel winner
is rejected when its complete program does not improve. B570 never selects the
unsupported extension and retains its existing subgroup/scalar paths.

These device runs predate the shared typed-staging baseline from #257. They
remain historical correctness evidence. The following rerun tests the rebased
integration against the current baseline.

#### After #257 and #262

The integration is rebased onto `41a15d3`, including #257's vectorized typed
staging and #262's race search. Workgroup tiles participate in race family and
partition handling. Both race and sequential qualification use the effective
matrix types, including implicit F32-storage/F16 multiplication; the small-input
probe must remain representable in the selected multiplication type. A race
qualifies its finalists, and the complete program still needs output/gradient
qualification and paired timing before replacing the incumbent. The final
integration preserves main's search driver, including its whole-program timing
reservation, without adding a separate kernel-budget limit.

The first measured candidate is `fee1c1e`; the baseline is `41a15d3`. The final
candidate `26c1378` removes a budget cap inherited from the sequential prototype,
restoring `train/search.rs` and `train/search/measure.rs` exactly to main. Its
matrix kernels, qualification and race integration are unchanged. The artifact at
`/x/Code/inferena-results/pr261-race-20261016/` retains source patches, exact
dependency locks, binary hashes, protocol receipts, samples and telemetry.
Both builds use the same crates.io versions; the PR updates the Blade/Naga
pins listed above. Devices and drivers are RTX 5070 / NVIDIA 595.91.07 and
B570 / Mesa 26.0.3-1ubuntu1, Linux 7.0.0-38-generic. GPU jobs run serially,
without concurrent builds; driver caches are retained and clocks are not locked.

The kernel experiment compares the ordinary untuned path with automatic race
selection in the same binary, without forcing a tile. It uses five shapes,
three layouts and two storage modes, three fresh RTX processes and one B570
process. Each variant warms for 500 ms, followed by 20 balanced ABBA/BAAB samples
of 64 replays; the middle RTX process reverses the starting order. The table
shows medians of process medians and medians of paired ratios. BF16 includes
both explicit input conversions.

| M × N × K | Operands | Layout | Untuned → searched, ms | Paired speedup |
|---|---|---|---:|---:|
| 1024 × 1024 × 1024 | F32 storage, F16 multiply | NN | 0.14792 → 0.08141 | 1.819× |
| 1024 × 1024 × 1024 | F32 storage, F16 multiply | BT | 0.16590 → 0.09423 | 1.760× |
| 1024 × 1024 × 1024 | F32 storage, F16 multiply | AT | 0.16627 → 0.11550 | 1.441× |
| 1024 × 1024 × 1024 | BF16 | NN | 0.14221 → 0.11545 | 1.233× |
| 1024 × 1024 × 1024 | BF16 | BT | 0.15747 → 0.14237 | 1.104× |
| 1024 × 1024 × 1024 | BF16 | AT | 0.16257 → 0.09792 | 1.661× |

All six select workgroup kernels. The 128×1536×576 products improve by
1.21–1.29× for F32-storage/F16 multiplication and 1.18–1.32× for BF16.
The ragged 257×512×65 products retain existing paths. Across the 30-case
matrix, the worst median searched/untuned slowdown is 1.0% on RTX and 2.5%
on B570. B570 offers no workgroup candidates; its scalar tuning gains are
preexisting behavior, not gains from this PR. The 1024³ searches take
0.15–0.23 seconds after earlier cases warm the context and pipelines; these
are not cold compilation measurements.

The fixed historical 64×64, K=32, 64-invocation tile was also rerun in three
fresh RTX processes. Its BF16 NN/BT/AT speedups are now 1.122×/1.105×/1.649×.
This confirms the review's concern: the old 2.7×/3.9× ratios depended on the
slower pre-#257 baseline. Automatic search still finds useful gains against
the improved kernel.

The whole-program gate still matters. In RTX Stable Diffusion accelerated,
three private kernel races find workgroup winners, but the final programs use
none. Two initially accepted inference programs are superseded by better graph
choices. The training program's 8.856 → 8.800 ms comparison does not clear the
improvement/noise guard, so its isolated kernel win is not installed. Kernel
speedups alone therefore do not establish a model speedup.

The first candidate passes formatting, Clippy for all targets/features with
warnings denied, documentation with warnings denied, and a featureless build.
Each GPU passes 639 library tests (8 ignored), 178 regression tests (21 ignored),
and all four calibration tests including the normally ignored cases. Tests
retain existing unsupported-feature early returns. Codegen covers 2,304 Naga
validation/SPIR-V emission variants. Independent full-output kernel checks pass
288 cases and 12,498,432 values against an F64 oracle that rounds operands to the
selected arithmetic, including poisoned allocations and large 128×256 tiles.
Maximum relative L2 error is 7.21e-7. All 288 cases also pass with
`VK_LAYER_KHRONOS_validation` explicitly enabled. Loader traces confirm that the
layer was inserted on both devices; there are no validation errors. This
establishes arithmetic for the selected precision, not acceptance against
unrounded F32 or training convergence.

#### Full-model comparisons — October 16–17

Inferena `7cecdc6` compares main and the PR on five models, strict/accelerated
profiles and both GPUs: 40 primary processes on `fee1c1e`. Every build search starts with a
fresh in-process kernel memo, shared across its program alternatives; persisted
memo files are disabled. The budget is 30 seconds per search, with 10 seconds
of phase warmup and 20 timing samples. Process order is balanced across profiles;
processes have 600-second watchdogs. The harness records pipeline keys in both
builds. Training timing means forward, loss and backward, without an optimizer
update. These runs retain the existing full-tensor output/gradient gates against
the first checked program; they are not a new independent PyTorch cohort.

All 112 selected programs pass qualification. Seven workgroup candidates win
private races and 87 are eliminated; none report shader or correctness failures.
Only RTX SmolLM2 accelerated retains workgroup kernels in its final programs:
one inference dispatch and 31 training dispatches. Its inference/backward times
improve by 1.6%/0.7% in this pair. It uses 64×128 in inference and 32×64/128×64 in
training. Kernel probing takes 0.87 seconds for inference and 1.73 seconds for
training, within the existing budget. Both builds reject the same 55 inference
and 12 training graph alternatives under the output bounds. The other workloads
and strict profiles select no workgroup kernels; their small differences must
not be attributed to workgroup acceleration.

The table retains every primary result, including outliers investigated below.
Positive deltas mean slower. Each row is one fresh process per arm, so small
differences are not evidence of a repeatable model speedup.

| GPU | Model / profile | Preparation Δ | Inference Δ | Latency Δ | Backward Δ |
|---|---|---:|---:|---:|---:|
| B570 | StableDiffusion / strict | +0.4% | -0.1% | +0.4% | -4.4% |
| B570 | StableDiffusion / accelerated | +0.2% | +25.3% | -3.6% | +2.6% |
| B570 | ResNet-50 / strict | -0.5% | +0.8% | +3.2% | +1.3% |
| B570 | ResNet-50 / accelerated | -1.2% | -0.0% | -2.6% | -3.1% |
| B570 | SmolLM2-135M / strict | +0.2% | +0.6% | -3.1% | +0.1% |
| B570 | SmolLM2-135M / accelerated | +0.1% | -0.2% | +2.0% | +0.1% |
| B570 | Whisper-tiny / strict | -0.8% | -3.1% | -3.4% | -0.9% |
| B570 | Whisper-tiny / accelerated | +0.4% | -1.7% | -0.2% | +0.2% |
| B570 | SmolVLA / strict | -2.3% | -5.1% | -2.6% | -0.7% |
| B570 | SmolVLA / accelerated | +0.2% | +0.2% | +3.2% | +1.4% |
| RTX 5070 | StableDiffusion / strict | -2.9% | +1.9% | -0.5% | -0.6% |
| RTX 5070 | StableDiffusion / accelerated | +0.5% | +0.8% | -0.4% | -0.3% |
| RTX 5070 | ResNet-50 / strict | +3.7% | -1.9% | +0.6% | +1.3% |
| RTX 5070 | ResNet-50 / accelerated | +3.1% | -0.4% | -1.2% | +2.5% |
| RTX 5070 | SmolLM2-135M / strict | +0.2% | +1.3% | -4.8% | +2.3% |
| RTX 5070 | SmolLM2-135M / accelerated | +0.5% | -1.6% | -1.0% | -0.7% |
| RTX 5070 | Whisper-tiny / strict | +0.2% | +0.6% | +1.9% | -1.4% |
| RTX 5070 | Whisper-tiny / accelerated | +0.0% | -0.2% | -0.1% | -0.4% |
| RTX 5070 | SmolVLA / strict | +0.0% | -0.1% | -0.1% | -0.5% |
| RTX 5070 | SmolVLA / accelerated | -0.0% | -0.6% | +8.6% | -1.9% |

B570 Stable Diffusion accelerated inference's initial +25.3% changes to −6.1%
with reversed process order and −0.4% with 30-second warmup/100 samples. Main
itself varies from 3.892 to 4.712 ms in those pairs. Those two followups do not
reproduce the large inference loss, but the final-candidate pair below does.
Backward is 2.5–3.1% slower across the three initial pairs. The longer
pair selects the same graph and differs in one of 886 pipeline keys. A separate
untuned control with 30-second warmup/100 samples uses identical pipeline keys
in both builds and gives +0.9% inference, −1.3% latency and −0.5% backward.
That control does not reproduce a general ordinary-path slowdown, but it does
not erase the searched backward gap. No workgroup candidate is offered on B570,
and its kernel probing stays below the reserved budget.

RTX SmolLM2 strict backward's +2.3% accompanies a different search choice:
main selects 16 submission chunks, while the PR keeps eight. The 16-chunk
comparison and a later attention alternative miss the PR run's existing
improvement/noise guard. The gate is retained, rather than weakened to force a
particular result.

RTX SmolVLA accelerated latency's +8.6% primary result uses identical final
pipeline keys, graph and submission count, and the same sampled GPU clocks.
A reversed process-order repeat with 30-second warmup/100 samples gives +4.5%
inference, +1.9% latency and +0.9% backward. Main's latency changes from 1.011
to 1.085 ms while the PR changes from 1.098 to 1.106 ms. The large primary
latency loss does not repeat; these results do not establish zero regression.

The initial candidate's inherited half-budget cap binds only in RTX ResNet
training: 15.04/15.00 seconds of kernel probing defer 12/9 later trials in
strict/accelerated mode. Neither case offers workgroup candidates. The final
candidate removes that unrelated policy change instead of changing the timing
guard or numerical thresholds. Main's existing reservation still leaves time
to qualify and time a kernel-tuned program. Fresh comparisons below cover the
affected ResNet cases, workgroup-selecting SmolLM2, and the B570 backward gap;
the other primary searches never reach the removed allowance.

The final candidate passes the same static checks and, on each GPU, 637 library
tests (8 ignored) and 178 regression tests (21 ignored). The two removed library
tests exercised the deleted budget cap. A fresh eight-process comparison
qualifies all 24 selected programs:

| GPU | Model / profile | Preparation Δ | Inference Δ | Latency Δ | Backward Δ |
|---|---|---:|---:|---:|---:|
| B570 | StableDiffusion / accelerated | -0.1% | +18.6% | +4.6% | +5.6% |
| RTX 5070 | ResNet-50 / strict | +6.5% | -0.6% | +0.4% | -1.1% |
| RTX 5070 | ResNet-50 / accelerated | +3.7% | +1.1% | +0.1% | +2.0% |
| RTX 5070 | SmolLM2-135M / accelerated | +1.2% | -1.7% | +4.9% | -2.4% |

These runs use the same protocol, except B570 Stable Diffusion uses 30-second
warmup and 100 samples. SmolLM2's modest inference gain repeats; individual
model phases still have losses. ResNet's execution differences remain small,
but its 3.7–6.5% preparation increase remains visible after removing the cap.

The final B570 pair is 4.293 → 5.090 ms for inference and 10.765 → 11.365 ms
for backward. Both builds select graph 1 for inference and graph 13 for
training. Only one of 300 inference pipeline keys and two of 886 training
keys differ. The inference samples have somewhat lower GPU clocks in the PR
run; backward clocks overlap at 2250–2300 MHz. This does not establish a cause.
The budget cap never bound in this workload, so removing it does not explain
or fix the loss. These results do **not** establish regression-free behavior.

A dependency-only control uses main plus the same Blade/Naga pins, with no
rectangular-matrix or search changes (`ccf65e0`). Two fresh pairs, ordered
updated-backend/main/main/updated-backend, use the same 30-second warmup and
100 samples. Inference changes by −15.5% and +13.4%; backward by +0.4% and +1.2%.
All 12 selected programs qualify. The large inference difference changes sign,
and the smaller backward difference does not reproduce the full PR gap. This
control does not establish a dependency regression or clear the PR's slowdown.

A final diagnostic pair enables context timestamps and retains ten per-dispatch
GPU profiles after each phase, again with 30-second benchmark warmup and 100
samples. The normal grouped-step timings in this mode are −6.0% inference,
+7.4% latency and −2.7% backward. The profiled backward GPU sum is
13.416 → 13.434 ms (+0.13%); changed pipeline keys account for +0.016 ms of
summed dispatch medians. There is no dominant changed kernel in that profile.
All six selected programs qualify. Profiling expands backward wall time to
about 22.44 ms, and enabling context timestamps changes the execution mode;
this diagnostic cannot replace the uninstrumented comparisons or clear them.
The cause of the earlier loss remains unresolved. A clean no-regression claim
needs fixed-program replay under controlled operating conditions; weakening
the acceptance gate or changing unrelated kernel heuristics is not justified
by these measurements.

#### Matched-input and CPU-affinity controls — October 17

The follow-up keeps main `41a15d3` and runtime candidate `26c1378` (documentation
head `de90b90`) fixed. It corrects one Inferena protocol mismatch: search used
the initialization callback once, whereas benchmark execution uploaded inputs
before every step. Both arms now use `build_measured_with_inputs`, refreshing
inputs before qualification, warmup and each sample. Uploads remain outside the
execution timer but consume the same 30-second search budget. The production
fix is [Inferena #67](https://github.com/kvark/inferena/pull/67); these diagnostic
runs retain the earlier harness/model definitions for a matched comparison.

Each paired run holds two processes resident and executes them serially. After
20 seconds of balanced phase warmup, 20 ABBA/BAAB rounds collect 32 executions
per block: 1,280 samples per arm and phase. Inputs are uploaded before each
execution; ordinary grouped submission and host wall timing stay enabled.
Context timestamps and per-dispatch profiling stay disabled. Compilation,
search and GPU work are serial between processes. No builds run alongside
measurements. The watchdog is 1,800 seconds per pair. Ordinary driver caches
are retained, GPU clocks are unlocked, and the CPU governor is `powersave`.
These are diagnostic comparisons, not standalone cohort records.

The B570 control replays frozen selected plans and requires exact execution-plan,
pipeline-key, barrier-group and full-output/gradient hashes before timing.
Positive deltas mean slower; ratios are paired medians across rounds.

| B570 Stable Diffusion accelerated comparison | Inference Δ | Latency Δ | Backward Δ |
|---|---:|---:|---:|
| Fresh independent selections, unrestricted CPU | +4.87% | +5.18% | -0.01% |
| Same main-selected plan on main/PR, unrestricted CPU | -1.62% | +0.18% | -0.99% |
| Same main binary and plan twice, unrestricted CPU | +0.19% | +3.36% | +0.21% |
| Same main-selected plan on main/PR, CPU 2 | -0.11% | +0.07% | -0.47% |
| Each build's own selected plan, CPU 2 | +0.04% | +0.13% | -0.01% |
| Same main binary and plan twice, CPU 2 | +0.03% | +0.14% | -0.15% |

An approximately 0.16 ms CPU-encoding gap reproduces between two copies of main
and disappears when both run on the same core. Each build's own selected-plan
differences then lie within the identical-binary control's variation. This
isolates the earlier B570 loss to measurement/operating conditions rather than
a changed kernel. It does not assert that a fresh noisy search always selects
identical programs.

RTX SmolLM2 accelerated with matched inputs improves inference/latency/backward
by 2.37%/2.02%/2.57% in one fresh paired search. The earlier latency loss does
not reproduce. Unrestricted RTX ResNet pairs still show backward differences
of +1.81% accelerated and +2.57% strict. Both head searches reject the
`conv_dw_splits=8` challenger under the unchanged noise gate, where main accepts
it. RTX SmolVLA backward is +1.64%, accompanied by higher CPU encoding time.
Fresh searches with CPU affinity applied only during final timing investigate
those differences without reducing parallel shader compilation.

| RTX 5070 fresh search, CPU 2 during timing | Inference Δ | Latency Δ | Backward Δ |
|---|---:|---:|---:|
| ResNet-50 / accelerated | +0.20% | +0.13% | -2.59% |
| ResNet-50 / strict | -0.71% | +0.33% | -1.06% |
| SmolVLA / accelerated | +0.00% | -0.10% | -0.09% |

All 18 selected programs qualify. In the accelerated ResNet repeat, main now
rejects `conv_dw_splits=8` while the PR accepts it, reversing the earlier choice
and training difference. SmolVLA's encoding times become 3.4021/3.4024 ms and its
training gap disappears. These controls do not reproduce a consistent runtime
regression: the largest remaining positive delta is 0.33%. They retain the
original negative results above and do not guarantee identical outcomes for
every future 30-second search. A bounded resampling prototype is retained on
local branch `experiment/whole-program-refinement`; it is not adopted, since
the current evidence does not justify changing the search policy for this PR.

A separate construction diagnostic records the generated SPIR-V and time in
`vkCreateComputePipelines`, using one CPU core to make compiler time attributable.
All 122 untuned ResNet shader binaries match byte for byte between main and PR.
The searched control has 794 common source/entry-point combinations, also all
byte-identical. Aggregate driver compiler time is 31.772 seconds for main and
28.834 seconds for PR; process wall time is 65.75/63.34 seconds. These are
instrumented, single-worker diagnostics, not normal preparation benchmarks.
Repeated untuned construction also varies within one binary. The earlier
preparation increase is concentrated in driver compilation, and this control
does not reproduce a systematic cost increase from the new Blade/Naga code.

A final uninstrumented preparation control uses separate, initially empty
NVIDIA shader-cache directories for each arm, with a 1 GiB limit. Both runs
create cache files there, confirming the override is effective. Run order is
main cold, PR cold, PR warm, main warm; only those private directories are
reused. Kernel-decision memos remain fresh. Normal parallel compilation is
retained. Each phase executes one diagnostic step after search, so its
execution samples are not used as performance evidence.

| ResNet strict preparation | Main | PR | Δ |
|---|---:|---:|---:|
| Cold shader cache, reported preparation | 53.698 s | 53.220 s | -0.89% |
| Warm shader cache, reported preparation | 49.832 s | 50.013 s | +0.36% |
| Cold cache, total trial construction | 2.257 s | 2.229 s | -1.22% |
| Warm cache, total trial construction | 1.162 s | 1.114 s | -4.12% |

All 12 selected programs qualify. Cold/warm shader-cache state accounts for
several seconds of preparation in either binary; controlling it removes the
earlier multi-percent setup penalty. Across all eight fresh paired comparisons
and these four preparation processes, all 60 selected programs qualify. The
largest remaining positive execution delta in the final targeted controls is
0.33%, and preparation is within 0.4% with equal cache warmth. These bounds
describe the observed controls, not a guarantee that every independent search
returns the same kernel choices. The unchanged correctness and whole-program
performance gates remain the acceptance policy.

The paired controller, exact-plan replay and cache protocol are published on
[Inferena's measurement-control branch](https://github.com/kvark/inferena/tree/experiment/pr261-measurement-controls)
at `0bb365c`. Raw records, frozen plans, exact shader binaries, original patches,
dependency locks, binary hashes, CPU profiles and telemetry are retained
outside Git under `/x/Code/inferena-results/pr261-noloss-20261017/`. The source
checkpoint postdates these runs; their original source/lock/hash snapshots
remain in that artifact. No numerical or performance acceptance threshold is
loosened for these controls.

To measure normal search against the untuned session, without forcing a tile:

```sh
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json WORKGROUP_SEARCH=1 CASE=512 \
  timeout 300s target/release/examples/workgroup_probe > /tmp/rectangular-search.jsonl
```

## Compiler stages — September 10

Source: `experiment/compiler-stages-2026-09-10` in
[Inferena](https://github.com/kvark/inferena/tree/experiment/compiler-stages-2026-09-10),
[Meganeura](https://github.com/kvark/meganeura/tree/experiment/compiler-stages-2026-09-10)
and [Blade](https://github.com/kvark/blade/tree/experiment/compiler-stages-2026-09-10).
The latter two add CPU spans only. Inferena pins both revisions and instruments
the installed Triton compiler. RTX 5070 / 595.71.05, i5-12400F, strict f32,
Python 3.13.13, PyTorch 2.13.0+cu130 at `cf30153`, Triton 3.7.1.

Three fresh processes per engine/model/cache state; private initially empty
compiler/driver disk caches, then a new process reusing each cache. GPU/OS
caches are not flushed. Engine order alternates. SmolLM2 runs 128-token prefill
and one stateless token; the other two models include F+loss+backward.
PyTorch uses default compilation with validated explicit CUDA Graphs.
Compilation is serialized (`TORCHINDUCTOR_COMPILE_THREADS=1`) to observe its
stages, so these are preparation diagnostics, not publication speed samples.

| Fresh-cache process | Naga parse, median µs/call | All Naga stages, ms | Vulkan pipeline creation, ms | Cold Triton compilations, ms |
|---|---:|---:|---:|---:|
| SmolLM2-135M | 141 | 7.24 | 386.8 | 2235.0 |
| ResNet-50 | 147 | 14.75 | 632.2 | 6725.5 |
| Whisper-tiny | 172 | 20.20 | 851.1 | 6598.8 |

Each cell is a median across three process records. Naga's total sums disjoint
parse, validation, specialization and SPIR-V emission spans; it excludes WGSL
generation and native driver work. Native pipeline calls number 21/66/56,
including reuse within a process. Cold Triton compilations number 37/181/96;
their per-call medians are 60.22/29.83/65.75 ms, including lowering to cubin.
The engines generate different kernel catalogues: these are **not paired
identical kernels**, nor a 10,000× compiler-speed result.

| Reported preparation, seconds | Meganeura fresh / reused | PyTorch fresh / reused |
|---|---:|---:|
| SmolLM2-135M | 1.175 / 0.779 | 23.990 / 6.066 |
| ResNet-50 | 1.385 / 0.713 | 19.521 / 2.225 |
| Whisper-tiny | 1.420 / 0.555 | 11.256 / 1.189 |

Preparation retains each runner's declared boundary: native graph/session
construction, versus Torch specialization/first execution; not just shader
compilation. Reused-cache native pipeline totals fall to 1.67/5.50/3.90 ms.
A separate 36-process tracing-disabled control changes preparation medians
by at most 2.1%; this sequential control is not a paired overhead confidence
interval. All 36 traced and 36 control processes completed. Cross-engine
forward/gradient-norm gates passed, CUDA replay validation stayed enabled,
and native output records repeated exactly. An earlier series overlapped an
automatic OS update and a user-space crash-handler loop; it is excluded.

Reproduce with `scripts/compile_study.py --output <new-outside-checkout-dir>`
in the pinned Inferena environment; repeat with `--untraced`. The later
experiment-branch `scripts/study_results.py` checks numerical pairs and exports
stage counts/times. Traces, caches and raw records stay outside Git. Cheap
Naga processing is measured and supports an on-device search budget; driver
compilation, qualification and whole-step amortization still must be charged.

The narrower `experiment/compiler-gemm-2026-09-10` Inferena tag compares f32
GEMMs with M=128, N=576, K=576/1536, output tiles 32/64, K tile 32 and 256
threads. Both retain runtime dimensions and IEEE f32 multiplication; generated
instructions/layouts need not be identical. Run `scripts/gemm_compile_study.py
--output <new-dir>`, then a separate cohort with `--warm-compiler`. The latter
compiles/loads the opposite tile first and reports that warmup separately.

Across three processes per cell, a new candidate after compiler warmup takes
**30.8–31.2 ms native versus 100.4–166.6 ms Triton**, including native pipeline
creation or cubin compilation/launcher loading. Native Naga stages total about
0.35–0.41 ms; the Vulkan driver accounts for about 30.4–30.7 ms. Reused private
disk caches give 0.43–0.52 versus 0.72–0.87 ms. First compiler use is separately
38.1–38.4 versus 375–442 ms; Triton's first-use initialization also appears on
cache hits. Allocation, execution and validation are outside these preparation
intervals. All 96 primary processes pass full-output f64 checks on ordinary and
tiny inputs, using the tuner's unchanged bound. This matched-domain microstudy
supports a roughly 3.2–5.4× cold-candidate preparation advantage here, not
10,000× or a kernel-performance claim.

## Full-model tuning — September 10

Sources: Inferena `experiment/tune-full-models-2026-09-10` and
`experiment/tune-smollm-confirm-2026-09-10`, pinning the compiler-stage revisions
above. Strict f32 on the same RTX 5070, real Inferena shapes/weights, grouped
unprofiled execution, five warmups and twenty samples per fresh process.
Run `scripts/tune_study.py --output <new-dir>`; confirmation adds
`--models SmolLM2-135M SmolLM2-360M --variants untuned default --replicates 6`.
All kernel qualification and whole-model numerical gates remain unchanged.

The six-pair AB/BA confirmation finds **1.092× SmolLM2-135M prefill speedup**:
12.699→11.628 ms medians, with median paired gain 1.068 ms versus twice its MAD
0.011 ms. Three exact classes switch 64→32 tiles, affecting 90 dispatches.
Median extra preparation is 85 ms: about **80 prefills to amortize**. Outputs
repeat exactly across all twelve runs. Stateless-token dispatches have no
eligible classes and do not benefit. SmolLM2-360M retains every tile: no guarded
gain, with 224 ms extra preparation. Its outputs also repeat exactly.

A separate three-process, three-model cohort widens the search to 128 classes
and 60 seconds, retaining the 64 MiB scratch cap. ResNet training changes
44.049→42.202 ms medians, below the 5% whole-step guard, while preparation grows
0.712→6.629 s. Two convolution classes reject output qualification. Whisper
shows negligible whole-step benefit. No tolerance is relaxed or default changed.
An interrupted 512 MiB exploratory cohort lacks its final manifest and is not
the confirmation evidence above. These results supersede an overly broad
negative reading of the earlier synthetic holdouts, not the frozen paper matrix.

## Profile-guided convolution specialization — September 10

Sources: `experiment/conv-specialization-2026-09-10` and
`experiment/conv-native-division-2026-09-10` in Inferena and Meganeura. Qualified
short Nsight Graphics traces concentrate ResNet training samples in scalar
convolution indexing/staging. The experiment binds immutable u32 parameters
as WGSL constants; a second arm uses native division with constant divisors.
No model/card rule, changed summation order, relaxed gate or default promotion.

Six fresh strict-f32 process pairs per arm give ResNet F+loss+backward medians
44.153→34.444→33.329 ms (untuned / constants / constants plus native division).
The roughly 1.28× / 1.325× gains clear the 5% plus paired-noise guard against
untuned. All recorded output fields repeat exactly; the existing full-f64
convolution regression oracles pass separately for both variants. Whisper's
roughly 1% gain does not clear the guard. A separate 135M control is unchanged.

For constants alone, six fresh-driver-cache pairs confirm the gain but add
3.067 s preparation: about 319 training steps to amortize, versus about twelve
with warm driver caches. The prototype compiles fallback and exact pipelines;
these are implementation ablations, not automated per-class selection. Run
`scripts/tune_study.py --models ResNet-50 Whisper-tiny --variants untuned
fixed-params fixed-native-div --replicates 6 --output <new-dir>` at the later
Inferena tag; the earlier tag supports `untuned fixed-params` and
`--fresh-driver-cache`. Native-tool methodology and attribution limits are in
Inferena's [analysis](https://github.com/kvark/inferena/blob/experiment/p3hpc-gap-2026-09-10/ANALYSIS.md).
The GPU interval also shrinks in a qualified Graphics capture, but PC sample
shares are not executed-instruction counts or wall-time barrier costs.

The later `experiment/conv-k-stage-2026-09-10` tags retain a negative result:
doubling K staging to 32 passes the full oracles but regresses ResNet training
33.340→36.834 ms across six paired processes, despite fewer barrier rounds.
No global K change is justified. A separate cold-cache confirmation of constant
native division amortizes its 2.757 s extra preparation after about 256 steps.

## Production convolution search — September 13

Measured code: `3996155`; source and the exact dependency lock are retained on
[`experiment/conv-production-2026-09-13`](https://github.com/kvark/meganeura/tree/experiment/conv-production-2026-09-13).
The lock-only child does not change the measured implementation. No raw records,
caches or binaries are committed. NVIDIA RTX 5070 / 595.91.07 and Intel Arc B570 /
Mesa 26.0.3-1ubuntu1 run sequentially on the same i5-12400F host. B570 occupies the
secondary PCIe 3.0 ×1 slot; Linux 7.0.0-31-generic, Rust 1.98.0. Both advertise
no native-f32 cooperative tile here; this study uses scalar f32 throughout.

The existing `tune_crossover` runner adds `--convolution`: forward/dX/dW search
with the production defaults (2 s soft deadline, eight classes, 64 MiB scratch).
The existing generator's uniform 32/64 tiles compete with constant-parameter,
native-division 32/64 tiles at K=16/32. No hardware/model rule, precision change
or relaxed qualification. Each comparison must pass the full-output parity and
sampled f64 checks on ordinary/tiny inputs before paired kernel measurements.
Only fully qualified, guarded winners are installed; rejected and unused
convolution pipelines are released at the end of search.

Six fresh processes per GPU, ordinary driver caches (not flushed), include a
40-pair untuned/untuned control and four 20-pair role-reversed blocks. Every
sample times `step+wait`, including the optimizer when present. Search, swaps,
settling and diagnostic readbacks are outside whole-step samples. Search itself
includes compilation, uploads/readbacks, CPU validation, sampling and cleanup.
Full control-session comparisons check outputs, gradients, parameters and
optimizer state; search and swaps must preserve their own session bit-for-bit.
The 250 ms NVIDIA telemetry process remains enabled on both devices; it does
not measure Intel clocks. Timings below are medians across process reports.

| GPU / workload | Baseline → selected, ms | Speedup | Search, s |
|---|---:|---:|---:|
| RTX 5070 / ResNet-50 F+loss+backward | 21.133 → 17.729 | 1.192× | 2.005 |
| RTX 5070 / convolution chain + Adam | 0.1281 → 0.1023 | 1.246× | 0.158 |
| RTX 5070 / convolution chain + SGD | 0.1277 → 0.1022 | 1.250× | 0.159 |
| Arc B570 / ResNet-50 F+loss+backward | 35.880 → 31.651 | 1.134× | 2.009 |
| Arc B570 / convolution chain + Adam | 0.2478 → 0.1863 | 1.330× | 0.388 |
| Arc B570 / convolution chain + SGD | 0.2469 → 0.1854 | 1.331× | 0.381 |

All 36 case runs complete, pass numerical/state checks and clear the unchanged
5% plus paired-noise whole-step guard in both session roles. ResNet visits 7/68
eligible classes on NVIDIA and 4/68 on Intel, changing 19 and 15–16 dispatches.
NVIDIA selects both K sizes; every selected Intel ResNet specialization keeps
K=16. This supports per-shape measurement, not a global K change. ResNet search
amortizes after roughly 589/475 steps on NVIDIA/Intel; the short optimizer chains
need about 6,100–6,300 steps. Session tensor allocation requests are unchanged;
peak search scratch is 18.48 MiB for ResNet and 0.122 MiB for the chains. These
are requested bytes, not measured peak VRAM including driver/compiler heaps.

A separate three-process check per GPU starts with an empty private driver cache
on the local SSD. All 18 case runs qualify and clear the whole-step guard.
ResNet median speedups are 1.175× NVIDIA and 1.107× Intel, with 2.050/2.004 s
search. The cold NVIDIA search reaches five classes instead of seven. The
deadline is soft: the largest observed ResNet overrun across these cohorts is
53 ms. Adam precedes SGD and can warm shared kernels even in the fresh-cache
processes. Neither cohort resets OS caches, GPU clocks or driver process state.

Reproduce at the retained branch, on one idle GPU at a time:

```sh
cargo build --locked --release --example tune_crossover
MEGANEURA_DEVICE_ID=0x2f04 target/release/examples/tune_crossover /tmp/conv-r1.json 1 --convolution
```

Use six new output paths and seeds 1–6; `0xe20c` selects B570. The runner prints
and records the actual adapter plus requested GPU options. For the cold check,
use seeds 1–3 and a new **local** directory per process, setting
`__GL_SHADER_DISK_CACHE=1`, `__GL_SHADER_DISK_CACHE_PATH=<dir>` and
`MESA_SHADER_CACHE_DIR=<dir>`. Earlier implicit-adapter and network-cache pilots
are not these confirmation cohorts.

The existing full-f64 convolution oracle regressions also pass on both GPUs,
covering forward/dX/dW, both spatial tiles, rectangular/odd/strided/padded and
reciprocal-boundary shapes, ordinary/tiny operands, and zero budgets. Existing
state-swap tests cover optimizer updates, pipeline reuse and unused-pipeline
release. The known Naga Workgroup-layout validation warning remains; no new
validation error is waived. This is synthetic ResNet batch 1 at 224² plus small
optimizer chains, not the paper's batch-4 Inferena cohort, pretrained accuracy,
convergence, Metal qualification or a PyTorch comparison. It supports opt-in
production search; it does not silently change the collection protocol.

Validation reuses existing tests: 275 library and 181 regression tests pass,
plus five scalar tuning regressions and the convolution/state-swap checks on
both GPUs. An isolated `cargo llvm-cov` run over these checks reports Rust host
line coverage of 99.4% in `tune.rs`, 82.3% in `runtime/tuning.rs`, 90.8% in
`codegen.rs` and 77.3% overall. This includes inline unit-test code but excludes
`tests/`, `examples/` and `bench/`; it does not instrument WGSL or exercise
unsupported native-f32 cooperative hardware. CI retains its ordinary suite's
separate coverage artifact by revision. No new regression executable is added.

## Parameter placement and host RAM — September 10

Source-only `experiment/parameter-allocation-2026-09-10` tags in Inferena,
Meganeura and Blade separate allocator rounding from actual memory placement.
On this RTX 5070, the 1.7B strict-f32 plan requests about 9.4 GiB, but ordinary
`Shared` allocation puts about 4.9 GiB of its bindings on the host heap.
It is not an all-VRAM scaling point. The general device-parameter prototype
retains named/original/packed weights and uses bounded 16 MiB upload staging.

A four-arm diagnostic pilot and **one completed untraced pair** preserve the
full prefill hash. The latter changes prefill 951.234→54.887 ms and stateless
token 1941.822→16.282 ms with the same plan and kernels. Do not report this as
a replicated speedup: subsequent confirmation was stopped as a precaution
after NVIDIA mapping-allocation errors in the Shared control. The manifest
remains incomplete. Further experiments use resident-only controls; no GPU
reset, reboot or host OOM occurred in these bounded runs.

The 360M control separates another issue: free-list allocation reduces buddy
rounding but does not improve step time. Actual heap bindings, plan requests,
allocator blocks and process-driver accounting answer different questions.
Fresh qualified 135M/1.7B CUDA-Graph/Vulkan Systems pairs show substantial
resident GPU-execution gaps; the later sustained-token control below also
exposes CPU power-policy sensitivity. These measurements do not isolate
barrier cost. Procedures, limitations and source pins are in Inferena's
[analysis](https://github.com/kvark/inferena/blob/experiment/p3hpc-gap-2026-09-10/ANALYSIS.md#placement-and-allocation-ablation--september-10).
The collection tag and paper tables remain unchanged.

Resident-only preparation follow-ups use the source-only Inferena tag
`experiment/parameter-preparation-2026-09-10`, pinning Meganeura `854b5b6`
and Blade `7b6d97a`. Six fresh-process replicates per model visit all six
orders of three arms: fresh upload allocations, reused bounded staging,
then reused staging plus a cache-blocked CPU transpose. No profiler runs
during this confirmation; all 54 recorded output sets repeat exactly.

| Complete process, median seconds | Fresh staging | Reused staging | Reuse + CPU transpose |
|---|---:|---:|---:|
| SmolLM2-135M | 10.537 | 2.939 | 2.625 |
| SmolLM2-360M | 14.164 | 5.718 | 4.521 |
| SmolLM2-1.7B | 43.163 | 36.179 | 28.167 |

These are startup/workflow savings, not shader-compilation or steady-state
speedups: the process includes checkpoint loading, both sessions, warmup,
measurement, validation and cleanup. No step gain clears the guard. Separate
Systems captures attribute the first saving to hundreds of avoided Vulkan
allocations; the next largest preparation span is CPU tensor conversion.
Run `scripts/tune_study.py --stream-weights --baseline device-params-buddy
--variants device-params-buddy device-params-reuse device-params-tiled
--models SmolLM2-135M SmolLM2-360M SmolLM2-1.7B --replicates 6 --output <new-dir>`
with the documented memory guard. No new main-branch fixture or binary is needed.

The `experiment/packing-layout-2026-09-10` and
`experiment/packing-warmup-2026-09-10` Inferena tags test the graph's packed
SwiGLU copy separately. Keeping the original unpacked weights saves about
3 GiB on 1.7B. Six AB/BA process pairs with 100 warmups and 100 samples confirm
16.316→14.558 ms stateless-token latency (1.121×), with identical full prefill
and token hashes. Prefill is unchanged; smaller models have no guarded step
gain. This is not yet an automatic representation search or default change.
An alternative scalar-matmul column layout is slower in the five-model pilot.

Longer warmup also reveals that 135M's short token window is not sustained
steady state: about 2.56 versus 3.58 ms on this configuration, with stable
prefill. The `experiment/host-latency-2026-09-10` Inferena tag localizes this:
GPU token time stays about 1.88 ms, but the CPU downclocks toward 800 MHz and
command-recording thread CPU time grows from about 0.59 to 1.44–1.48 ms.
Small Shared-allocation controls reproduce it; fixed-core and per-task
utilization-hint controls do not prevent it. No system power setting changed.
Preserve this sensitivity, not just the fastest samples. The collection tag is
unchanged; these diagnostic controls are not replacement paper timings.

The source-only `experiment/native-token-graphics-2026-09-10` tag qualifies
SDK-triggered capture after warmup, with matching full output hashes and no
reported hardware-event overflow. Token PC samples concentrate in residual-add
GEMV (48%), RMSNorm-fused GEMV (41%) and transposed GEMV (11%). Memory-dependency
stalls dominate the sampled warp states; these are not wall-time barrier costs.
The first plain/RMSNorm GEMV-width pilot finds no broad win and does not vary
the residual-add hotspot. Detailed controls and limits stay in Inferena's
[analysis](https://github.com/kvark/inferena/blob/experiment/p3hpc-gap-2026-09-10/ANALYSIS.md#host-side-latency-transient).

## Residual-add GEMV width — September 10

The source-only `experiment/gemv-add-width-2026-09-10` tags in Inferena and
Meganeura vary the residual-add hotspot separately, using the same general
32/64/128/256-thread generator. A 24-process pilot selects 128 for confirmation;
it is not a model/card rule or automatic production selection. All other
kernels and the grouped schedule stay fixed. The subsequent 54-process cohort
uses six replicates, all six three-arm orders, resident parameters, streamed
weights, 100 warmups and 100 retained samples, without profiling.

| Stateless token, median ms | Original 32 threads | 128 threads | 128 + unpacked weights |
|---|---:|---:|---:|
| SmolLM2-135M | 3.586 | 2.984 | 2.912 |
| SmolLM2-360M | 5.368 | 4.714 | 4.828 |
| SmolLM2-1.7B | 16.314 | 15.520 | 13.738 |

Width alone clears the 5% plus paired-noise guard on 135M/360M (1.20×/1.14×),
but not 1.7B. The combined arm clears it on all three; its 1.7B gain is 1.19×.
Prefill does not materially change. Some token windows retain CPU-related
drift; no samples are removed. These are this host's process-level observations,
not a cross-platform policy or replacement paper data.

All prefill output records match exactly. Changed reduction order changes token
hashes between widths, but each variant repeats exactly across processes and
full token/prefill-prefix relative L2 stays below 1.64e-5. Existing shader,
GEMV parity and broad smoke tests pass, as do 217,792 full-f64 ordinary/tiny
output checks. No validation bound is weakened or new regression file added.

Reproduce with `scripts/tune_study.py --models SmolLM2-135M SmolLM2-360M
SmolLM2-1.7B --variants untuned gemv-add128 unpacked-gemv-add128 --replicates 6
--stream-weights --resident --warmup-runs 100 --measurement-runs 100
--output <new-dir>` under the documented memory guard. Incorporating these
legal choices into bounded per-class selection is separate engineering work;
the current production tuner excludes GEMV.

Qualified Systems and short Graphics controls independently localize the 135M
improvement on the GPU. Systems' grouped token interval falls 1.877→1.389 ms;
the Graphics residual-add pipeline's reported resident-warp limit rises 24→48.
The 1.7B combined arm has a 12.163 ms grouped GPU interval versus Torch's
11.506 ms first-to-last-kernel span, but still pays substantial CPU command
recording. Prefill retains a large GPU gap. These instrumented intervals are
diagnostics, not replacements for the unprofiled table or a barrier-cost metric.

The bounded `experiment/matmul-k-stage-2026-09-10` follow-up stops **before
performance measurement**. K=8/16/32 passes existing shader/matmul/edge checks,
but full-f64 screening at M=128, N=2048, K=2048 rejects all depths, including
the original K=32: 2 ordinary and 10 tiny outputs out of 262144 exceed the
unchanged bound. Complete output hashes agree across depths and output tiles;
the pre-change binary has the same failures. This is a limit of the existing
scalar reduction under the tight oracle, not a new staging regression or a
failed whole-model gate. A matched-input IEEE-f32 Triton control has the same
failure counts at both tiles. The experiment runner reports output identity and
the first failure without retaining full arrays. No bound or default is changed.

## September tuning foundation

These are development observations on RTX 5070 / driver 595.71.05, not updates
to the frozen paper matrix. Existing `evidence/*` tags retain measured code.
The old milestone and audit branch remain available; they are not to be merged
back to restore the discarded archives. Historical protocols and full run
details are available in the
[archived experiment tree](https://github.com/kvark/meganeura/tree/bc04aa31e33b62f79445ca6d9519209ddcf3e756/docs/experiments).

### tuning-2026-09-05

Source: `evidence/tuning-2026-09-05`. Runner: `tune_session`.
Five fresh processes established a synthetic f32 tile-search transfer pilot;
local isolated wins did not establish a general model-level improvement. This
device did not expose native-f32 cooperative tiles.

### holdouts-2026-09-06

Source: `evidence/holdouts-2026-09-06`. Runner: `tune_holdouts`.
Six inference/training holdouts showed that kernel winners can fail to improve
whole-step time. Keep search cost, amortization and whole-step acceptance
separate from isolated timings.

### crossover-2026-09-06

Source: `evidence/crossover-2026-09-06`. Runner: `tune_crossover`.
Controlled six-process AB/BA confirmation accepted a roughly 1.177× dense-chain
whole-step improvement. None of the wider holdouts passed the whole-step guard.
No automatic default promotion followed.

### readback-2026-09-06

Source: `evidence/readback-2026-09-06`. Runner: `tune_readback`.
Separating GPU completion, staging copy and CPU validation localized a search
cost, not a kernel cost. Read-optimized staging reduced ResNet search from
about 606 to 39 ms (copy about 582 to 2 ms), preserving validation. Blade 0.9.

### staging-reuse-2026-09-06

Sources: `evidence/allocation-profile-2026-09-06`,
`evidence/staging-reuse-2026-09-06`. Runner: `tune_staging_reuse`.
Call-local exact-size reuse reduced dense/MLP search from about 44/64 to 32/45
ms. Validation, state isolation and memory accounting stayed unchanged.

### training-profile-2026-09-06

Source: `evidence/training-profile-2026-09-06`. Runner: `profile_training`.
Convolution derivatives consumed about 61% of ResNet F+loss+backward; attention
backward about 36–41% of SmolLM2. These profiles exclude optimizer updates and
localize work; they do not establish an end-to-end improvement.

### conv-tiles-2026-09-06

Source: `evidence/conv-tiles-2026-09-06`. Runner: `tune_crossover --conv-derivatives`.
The first convolution cohort used an incorrectly initialized input and is not
performance evidence. The qualification and runner were repaired.

### conv-tiles-corrected-2026-09-06

Source: `evidence/conv-tiles-corrected-2026-09-06`.
Corrected `tune_crossover --conv-derivatives` runs found no whole-step guarded win for
ResNet training. Full nonzero-gradient qualification prevents zero-data success
from admitting a candidate. Tile search remains opt-in.

### conv-indexing-2026-09-06

Sources: `evidence/conv-indexing-baseline-2026-09-06`,
`evidence/conv-indexing-exact-2026-09-06`.
Runner: `profile_training --conv-indexing`.
Floating reciprocal indexing was wrong at width 41. Exact integer indexing
fixed the defect but cost about 23% on the measured ResNet F+loss+backward.
Correctness took priority; this was not a performance win.

### conv-divisor-2026-09-06

Sources: `evidence/conv-divisor-baseline-2026-09-06`,
`evidence/conv-divisor-reciprocal-2026-09-06`.
Runner: `profile_training --conv-divisor`.
A shared all-integer invariant-divisor implementation recovered about 2% while
preserving exact addressing. It did not erase the indexing repair's full cost.

### split-k-2026-09-06

The split-K prototype is included in `evidence/split-k-sequence-2026-09-06`.
Runner: `measure_split_k`.
Plans charge partial storage and expose legal split counts; this is an explicit
probe API, not a promoted production selection policy.

### split-k-sequence-2026-09-06

Source: `evidence/split-k-sequence-2026-09-06`. Runner: `measure_split_k`.
A synthetic long reduction improved from 3.084 to 0.445 ms at eight splits
(6.93× for the complete isolated sequence). Both profiled large controls failed
accuracy qualification. No training-speedup claim or automatic installation.

### compensated-dw-2026-09-06

Source: `evidence/compensated-dw-accuracy-2026-09-06`.
Reproduce at that tag with `cargo test --release --test conv_derivatives
report_bounded_weight_accumulation_qualification -- --ignored --nocapture
--test-threads=1`. The shared compensated candidate passed 230/240 accuracy rows;
10 tiny structured-cancellation rows failed. Arithmetic was reverted, split-K
promotion deferred, and the bounded milestone closed. There were no performance
measurements after this rejection.

## Kernel search races — October 16

The [search design](search-convergence.md) describes the current policy. The
measurements below retain their tested revisions and limitations; they do not
imply that later revisions have repeated the whole-model campaign. The
reproduction and scaling harness is `examples/search_race.rs`.

Historical pre-review lavapipe comparison at `0d02d3e` (CPU emulation, so only
the search behaviour transfers), 30-second budget, the example's
SmolLM2-shaped block. This predates synchronized repetitions and balanced
rounds and is not validation of the revised policy:

| Strategy | Search | Classes | Comparisons | Qualified | Winners |
|---|---:|---:|---:|---:|---:|
| Sequential | 30.1 s, budget hit | 2 of 4 | 12 | 12 | 0 |
| Race, calibrated | 15.2 s | 4 of 4 | 42 | 1 | 1 |

### Review smoke check — RX 7900 XT, October 16

The revised code was checked with the process-local Mesa 26.2.4 build used
for BF16 experiments, a warm driver cache, and the example's 2-second budget.
Each policy ran twice in reversed policy order on fresh sessions/contexts;
the table gives the observed ranges, not confidence intervals. Other users'
background GPU clients were not stopped. No other review test/build was run
concurrently with these final measurements. This is a small block, not an
Inferena cohort or a controlled whole-model performance claim.

| Context timing | Policy | Search (s) | Selected step (ms) |
|---|---|---:|---:|
| Host | Sequential | 0.279–0.290 | 0.1442–0.1444 |
| Host | Race + profile | 0.216–0.222 | 0.1492–0.1525 |
| Host | Race, no profile | 0.146–0.147 | 0.1490–0.1491 |
| GPU timestamps | Sequential (host-timed duels) | 0.270–0.271 | 0.1412–0.1413 |
| GPU timestamps | Race + profile | 0.066–0.069 | 0.1469–0.1474 |
| GPU timestamps | Race, no profile | 0.120–0.122 | 0.1464–0.1502 |

All runs covered 3/3 eligible classes and 31 challengers. Sequential qualified
31 candidates, host races 2, and timestamp races 3. Profiling cost 67–70 ms in
host mode and 30–33 ms with timestamps. It reduced timestamp repetition counts
enough to repay calibration here, but added cost in host mode. More importantly,
the raced selections had slightly **worse** post-search step times than sequential
search. Lower search latency and fewer qualifications do not establish better
convergence. These observations motivated the investigation below; they are
not the results of the final default policy.

Reproduce the policy orders with `cargo run --release --all-features --example
search_race -- 2 sequential race race-uncalibrated race-uncalibrated race
sequential`; with `MEGANEURA_GPU_TIMING=1`, use `2 race-uncalibrated race
sequential sequential race race-uncalibrated`. Use the same ICD/device for both.

### AMD regression diagnosis and correction

Two distinct problems explained the regression, without changing warmup or
numerical qualification:

1. **Host descriptor-cache churn hid a GEMV win.** The same 32-lane GEMV with
   four column groups measured about 8–10 us in sequential duels but 12–18 us
   in a shared-encoder host race. Instrumenting encoding and submission/wait
   exposed the extra cost; Blade's `reset_descriptor_pool` discards layouts
   unused in the previous recording. Separate reusable host encoders reduced
   steady-state encoding from roughly 45–55 us to 2–3 us per 16-dispatch batch,
   and the race found the GEMV improvement. Buffers remain shared, and GPU
   timestamp races still use one encoder/submission per round.
2. **Finalists were ranked below the configured resolution.** After fixing
   host timing, both timing paths preferred eight GEMV column groups over four
   for only a roughly 1–2% isolated gain. The shared-context experiment swapped
   just that choice between retained sessions: the approximately 5 us
   whole-program penalty followed the eight-group choice. With serial dispatch
   the penalty disappeared (174.8 versus 174.9 us after the swap); per-dispatch
   profiling also showed near-equal GEMV times. This implicates interaction with
   overlapping dispatches rather than a slower isolated kernel. The original
   5% plus noise guard applied only against the original incumbent, not between
   finalists. Applying it there as well retains the four-group candidate here.

Release recheck on the same RX 7900 XT/private Mesa, 2-second budgets, warm
driver cache, two visits per policy in reversed order, without concurrent
review builds/tests (other users' clients remain untouched):

| Context timing | Policy | Search (s) | Post-search step (ms) |
|---|---|---:|---:|
| Host | Sequential | 0.275–0.280 | 0.1377–0.1403 |
| Host | Race + profile | 0.201 | 0.1370–0.1389 |
| Host | Race, no profile | 0.131–0.134 | 0.1386–0.1393 |
| GPU timestamps | Sequential (host-timed duels) | 0.270–0.273 | 0.1428 |
| GPU timestamps | Race + profile | 0.067 | 0.1426–0.1434 |
| GPU timestamps | Race, no profile | 0.123–0.127 | 0.1443–0.1477 |

The default race qualifies three winners rather than all 31 challengers and
selects the same dense tiles and four-group GEMV as sequential in these runs.
The example now also retains all programs and times interleaved held-out steps
in balanced order, independently of which search last warmed the device.
Those medians were 144.6–145.9 us sequential versus 146.4–146.9 us calibrated
race in host mode, and 163.1–165.3 versus 162.6–162.8 us with timestamps. A
shared-context host pair had identical selections (zero choices to swap) and
139.6 versus 139.5 us, then 139.8 versus 139.8 us after the swap. This removes
the reproducible regression above; small differences between fresh contexts
are not evidence of better kernels. It remains a small-block test, not a claim
of optimality or a whole-model cohort result.

Set `MEGANEURA_SEARCH_VERBOSE=1` to print every candidate's raw samples and
decision. `MEGANEURA_SEARCH_SHARED_CONTEXT=1` shares the context/profile across
fresh sessions; with two strategies it also swaps their selected kernels and
repeats the held-out comparison. `MEGANEURA_SERIAL_DISPATCH=1` separates the
effect of overlapping dispatches. Never infer whole-program improvements from
isolated scratch wins alone: `build_measured` retains its whole-step guard.

### B570 and RTX 5070 validation — October 16

Tested `180946f` against main `12df95c`, with the same Blade `96c1dc7` and
Naga `a97559b` pins. The host ran Linux `7.0.0-38-generic`, Mesa
`26.0.3-1ubuntu1` on B570 and NVIDIA `595.91.07` on RTX 5070. GPU jobs ran
serially with process-tree watchdogs and no concurrent builds. Clocks were
recorded but not fixed. Kernel memos were private to each search; primary
driver caches were warm, with separate fresh-cache experiments.

The Inferena runner at `7cecdc6` used 30 seconds of search per session,
10 seconds of warmup and 20 held-out samples. Five models (SmolLM2-135M,
Whisper-tiny, SmolVLA, ResNet-50 and the small Stable Diffusion U-Net) ran
in strict and accelerated profiles on both GPUs: 20 PR configurations.
Six configurations had fresh main controls; the other 14 used earlier main
measurements under the same protocol. Reversed-order repeats covered B570
SD strict and RTX SD accelerated. Forward/backward below excludes optimizer
updates, and all four rows use fresh main controls.

| GPU / model / profile | Main forward/backward (ms) | Race forward/backward (ms) |
|---|---:|---:|
| B570 / SmolLM2-135M / strict | 55.388 | 53.654 |
| B570 / ResNet-50 / strict | 53.352 | 50.255 |
| RTX 5070 / ResNet-50 / strict | 27.721 | 25.135 |
| RTX 5070 / SD / accelerated | 4.960 | 4.594 |

The reversed RTX SD pair was 4.935 versus 4.593 ms. These are useful fast
paths, not a claim that every selected program improved. In particular,
RTX ResNet strict latency rose from 2.415 to 2.508 ms in its fresh pair.
B570 SD strict needs further selection/operating-state investigation:

| B570 SD protocol | Main inference / latency / forward+backward (ms) | Race inference / latency / forward+backward (ms) |
|---|---:|---:|
| Normal pair 1 | 4.055 / 4.199 / 11.379 | 4.266 / 4.267 / 11.412 |
| Reversed normal pair 2 | 4.220 / 4.195 / 11.369 | 4.277 / 5.020 / 11.231 |
| Separate 30s warmup, 100 samples | 4.931 / 4.327 / 10.914 | 4.989 / 4.439 / 11.061 |

Pair 2 selected a flash-attention latency variant and reached lower clocks
near the held-out measurement. The 19.7% outlier did not recur in the longer
warmup pair, but that pair was still 1–3% slower. Plan choice and clock behavior
are not isolated; do not discard the outlier, average different protocols, or
claim no performance regressions from these runs.

Shorter search budgets also miss fast paths. Each cell below is one fresh
search followed by the normal held-out forward/backward measurement:

| GPU / model / profile | 5s search | 10s | 20s | 30s |
|---|---:|---:|---:|---:|
| B570 / SmolLM2-135M / strict | 85.823 ms | 85.818 ms | 54.881 ms | 53.654 ms |
| RTX 5070 / ResNet-50 / strict | 28.626 ms | 28.670 ms | 27.571 ms | 25.135 ms |
| RTX 5070 / SD / accelerated | 8.804 ms | 8.797 ms | 4.983 ms | 4.594 ms |

Keep 30 seconds per session for this cohort. Faster kernel races do not
remove construction, qualification and whole-program comparison costs.

Kernel-only scaling used 1, 4, 16 and 64 independent matrix classes, budgets
of 2 and 10 seconds, and sequential, calibrated-race and uncalibrated-race
policies. The 64-class results at the 10-second limit were:

| GPU / timing / driver cache | Sequential search / classes visited | Calibrated race / classes visited |
|---|---:|---:|
| B570 / host / warm | 10.001s / 53 | 3.608s / 64 |
| RTX 5070 / host / warm | 6.737s / 64 | 2.703s / 64 |
| B570 / host / fresh | 10.001s / 55 | 3.672s / 64 |
| RTX 5070 / host / fresh | 6.745s / 64 | 2.722s / 64 |
| B570 / timestamps / warm | 10.001s / 51 | 2.380s / 64 |
| RTX 5070 / timestamps / warm | 6.912s / 64 | 1.680s / 64 |

Construction compiles incumbent pipelines before this tuning timer. A visited
class can be partial at a deadline; the races above completed all 64. Profile
startup can outweigh the race gain for a single class. Timestamp experiments
do not measure whole-model overhead of enabling timestamps on the context.
The campaign exercises existing candidate lists, up to 24 candidates per
race, without #261's rectangular candidates or the cumulative budget cap
from its earlier prototype.

**Retain the whole-program gate.** A separate control held two initially
identical sessions in one context, tuned only one, then warmed them in
alternating order for 10 seconds and collected 30 balanced pairs. Both
outputs passed an exact constant-product oracle. On the 64-class graph:

| GPU / policy | Untuned control (ms) | Tuned (ms) | Full-graph change |
|---|---:|---:|---:|
| B570 / race | 1.6941 | 1.8016 | +6.34% |
| B570 / sequential | 1.7003 | 1.7878 | +5.15% |
| RTX 5070 / race | 0.7023 | 0.7957 | +13.29% |
| RTX 5070 / sequential | 0.7014 | 0.7948 | +13.32% |

Private kernel wins can lose when composed, with either strategy. These
percentages compare each tuned session to its untuned control, not race
against sequential. `Session::tune_with` intentionally does not execute the
live graph. `build_measured` preserves the ordinary fallback, qualifies each
challenger and requires paired whole-program timing before replacement.
Cached choices must pass that same gate. Neither class coverage nor a memo
hit is a reason to bypass it, and short search-time comparisons do not prove
performance under every sustained operating state.

The consolidated example reproduces the control and scaling workload:

```sh
MEGANEURA_SEARCH_CLASSES=64 MEGANEURA_SEARCH_SHARED_CONTEXT=1 \
  MEGANEURA_SEARCH_WARMUP_SECONDS=10 \
  cargo run --release --all-features --example search_race -- 10 untuned race
```

Use `untuned sequential` for the other control and `sequential race
race-uncalibrated` for coverage comparisons; vary the class count through
1, 4, 16 and 64. Pick the same ICD/device for each comparison. The example
also swaps choices between a shared-context pair and repeats the timing.
Its immediate before/after samples are labelled diagnostic; use the balanced
held-out samples for comparisons. Search-time correctness gates remain active,
and every scaling output is checked before and after tuning and after swaps.
These example controls observe whole-program effects; they do not replace
`build_measured`'s acceptance decision.

Across the hardware campaign, 119/119 selected-session native qualification
checks passed in 41 model processes, including controls and budget sweeps.
All 92 scaling processes completed; 80 independent-matrix cases matched an
exact output oracle, while 12 SmolLM2-block cases had no extra external
oracle. Both public-API memo retry probes passed. Library tests (628 passed,
8 ignored), six race/state tests on each GPU, 20 RTX search tests, library
Clippy and formatting also passed. No independent PyTorch whole-model oracle
was rerun. The hardware results above describe `180946f`; later production review fixes
require separate validation.

### Earlier review correctness checks

Validation of the reviewed tree: 628 library tests and 174 regressions passed
on lavapipe (8 and 21 ignored); unsupported native-matrix cases skip there.
On RX 7900 XT, six race/state tests (including native F16 split-K) and the
timestamped calibration test passed with Vulkan synchronization validation
enabled, without synchronization hazards. Formatting, all-target/all-feature
Clippy with warnings denied, the featureless check, and documentation build
also passed. These are correctness checks, separate from the smoke timings.

### Encoder budgets and partial-profile upgrades

The review follow-up after `56ab8c1` counts encoder uniform scratch, skips
host encoders for rejected shaders, and lets a later resource allowance fill
missing profile measurements. An expired calibration slice leaves the
submission round trip unmeasured, so it can be retried. The whole-program
acceptance gate and sequential comparison control remain in place.

On B570 and RTX 5070, the final tree passed 630 library tests (8 ignored) and
176 regression tests (21 ignored) per GPU. The affected ignored staging-release
and split-sequence state/budget tests also passed on both devices, as did
explicit host and timestamp calibration checks. Formatting, all-target,
all-feature Clippy with warnings denied, the featureless check and documentation
with warnings denied passed.

A fresh 64-class process at `56ab8c1` and a final-tree process on each GPU used
the same lockfile, drivers and kernel as the B570/RTX campaign above, with warm
driver caches, a 10-second search budget, 10-second balanced warmups and 30
held-out pairs. GPU jobs ran serially under watchdogs. Every output passed the
exact constant-product oracle before/after tuning and after swapping choices.

| GPU | Previous head search | Updated search | Classes completed | Changed kernel choices |
|---|---:|---:|---:|---:|
| B570 | 3.465 s | 3.439 s | 64/64 | 0 |
| RTX 5070 | 2.507 s | 2.460 s | 64/64 | 0 |

The selected-program medians after swapping choices were 1.7963/1.7955 ms
(previous/updated) on B570 and 0.8154/0.8118 ms on RTX. These small differences
are not evidence of a speedup: the choices are identical, and operating state
still varies. This checks search coverage and timing after the review fixes;
it does not repeat the full Inferena cohort. The isolated choices still lose
to the untuned full graph here, reinforcing the whole-program gate.
