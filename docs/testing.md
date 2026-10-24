# Regression checks and diagnosis

Iteration should exercise a few broad stack contracts, then use the same
diagnostic tools on a failing application. A test per symptom is not the goal.

```sh
# CPU compiler/graph checks; no GPU required.
cargo test --lib

# Broad GPU coverage: optimizer/checkpoint state, cache, model training,
# quantized weights, named reads and nonfinite attribution.
cargo test --test smoke -- --test-threads=1

# Every op, derivative and compiler transform against the f64 reference.
cargo test --test oracle -- --test-threads=1

# CI/full regression check, including the narrower historical cases.
cargo test --tests -- --test-threads=1
# ...to also cover the first-party models (`models`) and the GGUF loader
# (`gguf`), which nothing enables by default:
cargo test --tests --all-features -- --test-threads=1
```

## Persistent Session execution

The same GPU suites accept the persistent tile selector. Keep adapter,
precision, poisoning, and feature settings identical to the ordinary run:

```sh
MEGANEURA_PERSISTENT_WORKERS=64 cargo test --test oracle --test smoke \
  --test regression --test gguf_model --all-features -- --test-threads=1
# Qualified Vulkan device dependencies within compatible regions:
MEGANEURA_PERSISTENT_WORKERS=64 MEGANEURA_DEVICE_SCHEDULING=1 \
  cargo test --test oracle --test smoke --test regression --test gguf_model \
  --all-features -- --test-threads=1
# This targeted oracle case explicitly checks ordinary and strict persistent
# execution, numerical results, and nonzero persistent coverage.
cargo test --test oracle smoke::persistent_execution_reports_actual_coverage
```

`reference::Report.execution` and `Session::execution_report()` expose actual
coverage and fallback reasons. The full suite allows ordinary fallback for
operators outside the generated pointwise/reduction scope; passing it does not
mean all graph operations are persistent. Session tests cover queue reset,
chunking and external records, while the existing tests retain their training,
checkpoint, shared storage, cache, and precision assertions. Test constructors
use the shared environment configuration even when they override another
runtime option.
`PersistentOptions::device_scheduling` selects the native-fence protocol.
`ExecutionRegion::device_scheduled` identifies those launches. Regions currently
hold up to three compatible generated handlers and release successors only after
all predecessor tiles finish. Resource, geometry and unsupported-handler boundaries
remain physical dispatches. `validate_persistent_completion()` waits and checks
unfinished tasks, counters and diagnostics after a step; it performs a readback,
so correctness checks call it outside timed intervals. The reference harness and
paired benchmark do this automatically. Ordinary execution remains the default.
The convolution optimizer test keeps its allocation-reuse control explicitly
ordinary: persistent sessions currently disable lifetime aliasing. Its other
sessions exercise the selected backend, and the memory-reduction assertion
remains active for the ordinary control.
Profile schema 4 includes that execution report and records one timing per
physical launch. Profiling windows use execution-region indices. Each timed
row lists its untimed source operations in `members`; `logical_workgroups`
is only populated for a single-operation launch. Physical worker grids and
selected-pipeline statistics describe the actual launch.

`bench_ci_latency` accepts the same selector and emits execution coverage,
resident bytes, construction times, adapter/driver information, and raw samples.
The existing `train_step_*` metrics measure forward/backward, preserving the
CI baseline. The additional `train_update_*` metrics include SGD and assert
that parameters changed; the earlier benchmark's comment claimed an update
without configuring one. Update timings need their own baseline.
The benchmark's ordinary single-mode run retains its short CI warmup.
For a production comparison, both existing benchmarks accept `--paired`:

```sh
cargo run --release --example bench_ci_latency -- --paired
cargo run --release --example bench_smolvla_train --features models -- --paired
# Same paired harness, selecting device dependencies for the persistent side:
cargo run --release --example bench_ci_latency -- --device-scheduling
# Separate instrumented run: GPU pass intervals, worker task histograms.
MEGANEURA_GPU_CAPTURE=1 cargo run --release --example bench_smolvla_train --features models -- \
  --paired --diagnostics --warmup 5 --warmup-ms 500 --runs 10
```

Select the same adapter and configuration for every run. Defaults are 64
persistent workers, at least 10 warmup pairs and one second of warmup, then
50 alternating ordinary/persistent sample pairs. Override these with
`--workers`, `--warmup`, `--warmup-ms`, and `--runs`. Both sessions explicitly
disable kernel search and must have identical compiled execution plans.
The ordinary/persistent selection overrides the environment selector.

Each execution restores identical parameters outside timing; SGD uses a
fixed learning rate and must change parameters. The CI boundary includes
input upload, completed execution and output/loss readback. SmolVLA retains
its existing completed-step boundary, with upload/readback outside timing.
Every sample compares full logits or loss; training also compares every
available parameter gradient and updated parameter. SmolVLA stages those
reads together and excludes allocation padding from comparison. These checks
supplement the f64 oracle suite without changing its tolerances.

JSON reports retain raw times, quantiles, paired ratios, source/toolchain and
device identity, coverage, selected pipelines, residency and observed build
times. Construction shares one GPU context and builds ordinary sessions
first, so these are not matched cold-cache construction measurements.
Primary runs have no timing/counter instrumentation. `--diagnostics` enables
GPU pass intervals and per-worker task counts; `MEGANEURA_GPU_CAPTURE=1` also
requests native pipeline statistics. Keep those results separate
from primary wall times. GPU intervals cover pass starts through submission
completion, excluding separate upload/readback submissions. Generated JSON
and logs belong under `target/`, with conclusions and reproduction commands
recorded in [the integration plan](megakernel_plan.md).

## Android / Quest 3S

Quest 3S qualification uses Vulkan on Adreno 740, Android 14 (API 34).
The tested driver is `810c66ceb7, Idb4ff819d7, 1782305085` (2026-06-24).
It exposes packed integer dot products and 64-lane subgroups, but neither
native shader f16 nor cooperative matrices. Half storage is packed into
32-bit words and decoded into f32; conversion uses explicit nearest-even
rounding. Subgroup GEMV derives its subgroup count from the workgroup width
and subgroup size. Packed Q4 reads and convolution index remainders use
forms verified on this driver.

The 2026-10-10 qualification passed 163 oracle, 146 regular regression,
63 smoke, 22 GGUF, and 571 unit tests on the headset, including the opt-in
smoke and unit checks. Another 12 opt-in regression checks passed, covering
tuning, training state, and PyTorch ResNet/Whisper references. Six opt-in
cooperative-matrix checks were inapplicable; regular tests also retain their
hardware capability gates. The three host-only manifest checks and the full
regular suite passed separately on Windows/RTX 3050.

Strict-f32, untuned Inferena runs also passed forward and training comparisons
for StableDiffusion, ResNet-50, SmolVLA, SmolLM2-135M, and Whisper-tiny. These
compare 256 sampled output values, shapes, losses, and parameter-gradient norms
against preserved PyTorch CUDA records. The separate oracle suite checks full
elementwise results. Five timed samples follow warmup; this is not a paired
performance comparison with the reference machine.

For full model qualification, `scripts/qualification/prepare.py` instruments an
isolated copy of Inferena's Android runner. It reads every source parameter and
input before warmup, then every inference/latency output, the actual GPU loss,
and every canonical parameter gradient after the measured training step. The
benchmark graphs and timed regions are preserved. Content-addressed blobs and
SHA-256 manifests retain the exact device values, including integer token IDs.

`scripts/qualification/compare.py` loads those parameters and inputs into the
matching PyTorch CPU architectures. It checks the complete parameter/gradient
inventory and layouts, handles tied weights, and compares every element. The
acceptance limits are `1e-6 + 1e-4 * max(abs(reference))` for maximum error and
`1e-6 + 1e-4 * rms(reference)` for RMS error, per tensor and for all gradients
together. Elementwise `allclose` failures are reported separately; that is not
the acceptance policy. CPU float64 arithmetic is the default oracle, using the
exact f32 device values promoted to float64; the GPU remains strict f32.

The complete five-model run passes this policy on Quest 3S. All inference and
latency outputs, training losses, and the following gradients were compared:

| Benchmark workload | Gradient tensors | Gradient elements |
|---|---:|---:|
| StableDiffusion conditioned U-Net | 181 | 10,928,768 |
| ResNet-50, batch 4 | 108 | 25,530,472 |
| SmolVLA action expert, 16 layers / 50 actions | 152 | 99,848,592 |
| SmolLM2-135M, 128 tokens | 272 | 134,515,008 |
| Whisper-tiny encoder, 1,500 positions | 66 | 7,632,384 |

These checks found two issues missed by the sampled comparisons: SmolVLA's
training builder used unmasked self-attention instead of inference's causal
attention, and SmolLM2's vocabulary-projection backward accumulated 49,152
terms sequentially. Training now shares SmolVLA's forward builder. Long scalar
matrix products use bounded partial sums when eligible, reducing f32 rounding
error; cooperative and explicitly selected schedules retain their kernels.
The final SmolLM2 gradients pass against CPU float64. Three tensors still
exceed this strict gate against CPU float32, which itself exceeds the same
gate against CPU float64 for those tensors. The tolerance was not widened.
After these fixes, 957 regular and 21 applicable opt-in checks passed on Quest;
the full regular Windows/RTX 3050 suite passed 960 checks. The same six opt-in
cooperative-matrix checks remain inapplicable to this headset.

Prepare with an archived Android runner lockfile, build with the NDK linker,
then use the PyTorch environment to run the capture and comparison:

```sh
python scripts/qualification/prepare.py --inferena ../inferena \
  --output target/full-model-qualification/runner --lock /path/to/Cargo.lock
cargo build --release --locked --target aarch64-linux-android \
  --manifest-path target/full-model-qualification/runner/Cargo.toml
python scripts/qualification/run.py --serial QUEST_ADB_SERIAL \
  --binary target/full-model-qualification/runner/target/aarch64-linux-android/release/inferena-meganeura \
  --runner target/full-model-qualification/runner --inferena ../inferena \
  --output results/quest3s/model-qualification/new-run
```

The output directory must be new. Supply the SmolLM2-135M checkpoint under
`/data/local/tmp/meganeura/models/SmolLM2-135M` first. Each run freezes the CPU
architecture, exporter, dependency lockfile, source patch, and binary hash with
its evidence. This qualifies the benchmark workloads' forward/loss/backward
computations; it does not cover optimizer updates or every possible input.

Build with the Android NDK linker configured for `aarch64-linux-android`
(the qualification used NDK r27 and `aarch64-linux-android29-clang`):

```sh
cargo test --release --target aarch64-linux-android \
  --features models,gguf,profiler --lib --tests --no-run
```

Push each executable printed by Cargo to `/data/local/tmp/meganeura`,
mark it executable, and run with `TMPDIR=/data/local/tmp/meganeura` and
`--test-threads=1`. Run the smoke suite's three `harness_manifest::` checks
on the host checkout; exclude that module from the device run. The ONNX
fixtures are embedded. For the opt-in PyTorch comparisons, generate
`scripts/gen_reference.py` on the host and copy its two JSON files into
`bench/results` relative to the device test directory. Run the applicable
`--ignored` tuning, staging, and gradient checks separately on an idle device.
Native cooperative-matrix checks require different hardware. The optional
`hf-hub` downloader is not needed: Android loads supplied model assets.

For unattended Quest runs, save `settings get global stay_on_while_plugged_in`,
then temporarily keep the plugged-in headset awake:

```sh
adb shell svc power stayon true
adb shell am broadcast -a com.oculus.vrpowermanager.prox_close
```

After testing, broadcast
`com.oculus.vrpowermanager.automation_disable` and restore the saved global
setting. Use `adb -s <serial-or-ip:port>` when multiple transports are present.

## Check every op against its definition

[`meganeura::reference`](../src/reference/mod.rs) is the written definition
of every graph op: a naive `f64` interpreter that shares no code with the
lowering. Its dispatch is an exhaustive `match`, so an op without reference
semantics does not compile. The `oracle` suite checks everything else against
it:

```sh
cargo test --test oracle -- --test-threads=1
# Search further for pipeline bugs, or rerun one failing graph:
ORACLE_FUZZ_COUNT=400 cargo test --test oracle fuzz:: -- --test-threads=1
ORACLE_FUZZ_SEED=23 cargo test --test oracle fuzz:: -- --test-threads=1
```

Three questions are kept apart, because one gradient check on the GPU
conflates them:

1. **Is the differentiation rule right?** `reference::gradients::check`
   evaluates `autodiff::differentiate` with the reference interpreter and
   compares every parameter gradient with central finite differences, in
   `f64` on the CPU. This excludes GPU execution as the cause, not mistakes
   in the reference or finite-difference error. Ops sit mid-graph under
   `gradients::weighted_loss` (`coef · Σ w ⊙ y` with fixed random `w`), so a
   backward that assumes `dL/dy = 1` fails. Random directional probes cover
   every element at once, including ones with small gradients that
   norm-based comparisons miss.
2. **Does each kernel compute its op?** `reference::gpu::check_inference`
   runs a graph on the device and compares every output element with the
   reference. The per-family suites build single-op graphs, including
   backward ops, over shapes chosen to reach each lowering the compiler can
   select: GEMV and tiled products, row-split reductions, flash-attention
   variants, convolution tiles, and the boundaries between them.
3. **Does the compiler preserve meaning?** `fuzz` builds random graphs and
   runs them through the whole pipeline (e-graph rewrites, dispatch fusion,
   memory planning, and the training step) against the unoptimized
   reference.

Sessions under test fill every buffer that holds no parameter with NaN
(`SessionOptions::poison`, or `MEGANEURA_POISON=1` for any run). A kernel that
reads memory nothing wrote then shows up as NaN instead of a plausible zero.

Tolerances follow the error analysis instead of a fixed threshold. For sums
of products the rounding error scales with `Σ|aᵢbᵢ|`, not with the result,
so `reference::error_scales` evaluates such ops on the magnitudes of their
inputs and carries these scales through every linear op in the graph. Element
`i` passes when `|got − want| ≤ rtol · (sᵢ + floor · max s)`, with
`rtol = 2e-4` by default. This accommodates cancellation without scaling the
bound to observed device errors. It is a practical error model, not a proof
that every wrong index or missing term will be detected.

Family sweeps can run each alternative lowering as well as the default:
no dispatch fusion (`gpu::Options::lowerings`). Training checks
compare every user output and every parameter gradient, including a
parameter's slice of a packed parameter the optimizer replaced it with.

The rest of the suite covers what a reference cannot: optimizer and
checkpoint state, caches, tuning and profiling, loaders, parity with
external models, dispatch geometry at extreme sizes, plan shape (fusion
happened, a pack formed), bit-exactness between fused and expanded forms,
packed quantized formats, and cooperative-matrix paths on hardware that has
them. A numerical check of an op belongs in `oracle`, not in a new focused
test.

Adding an op means giving it reference semantics (the compiler insists),
adding its shapes to its family's sweep, and adding a mid-graph gradient case
if it is differentiable. The checks then apply automatically.

Limits: lavapipe has no cooperative-matrix support, so those kernels are only
checked on hardware that has it; run the suite there too. Packed quantized
weights have no reference yet: the interpreter reports them as unsupported.

## The oracle does not cover quantized activations

`CompileOptions::quantized_activations` defaults to **true**, and for a
quantized weight format it quantizes the GEMV activation row to Q8_1 and runs
the inner product on integer dot products. This is the one switch in the
compiler that changes the numbers rather than the route to them, so it is
never selected by measurement — but it means the default decode path is not
the path the f64 reference describes, and the oracle cannot check it,
because packed weights have no reference.

What covers it instead: `regression::int_dot_gemv` pins the exact integer
arithmetic against a host model of `vec_dot_*_q8_1` and separately bounds the
error against the f32 path, so a matching-but-wrong reference cannot pass
vacuously. `gguf_model::a_quantized_model_agrees_with_its_own_dequantization`
does the same end to end, guarded by
`gguf_model::quantizing_actually_changes_the_weights`.

The gap is coverage, not correctness: a regression that made this fire on an
*additional* weight format or shader group would be invisible to every check
above, because they all construct the graphs that select it explicitly. Set
`quantized_activations: false` in a test that means to measure the f32 path —
`gpu_smoke` does this for exactly that reason.

## Suite layout

`smoke`, `regression` and `oracle` compile their modules into three
executables (`oracle` from `tests/oracle/`). `gguf_model` is a fourth,
behind the `gguf` feature. Add a case to an existing module;
`autotests = false` deliberately stops new files becoming new link jobs.
For example, the old `--test checkpoint_validation` selection becomes
`--test smoke checkpoint_validation::`. `--test tune` becomes
`--test regression tune::`.

The cost of that choice is that a new file under `tests/` is dead code until
one of those roots declares it as a `mod` child, and nothing in the build
notices when that is missing. `smoke::harness_manifest` walks the same graph
the compiler does — from the `[[test]]` targets, through their `mod`
declarations — and fails if a file is unreachable, or if a `mod` names a
file that is not there. It also asserts `autotests = false` is still set, so
it cannot pass vacuously if autodiscovery comes back. Three files
(`outline_optimize`, `profile_windows`, `resnet_correctness`) sat uncompiled
until that check existed.

A test whose fixture is not in the repository must be `#[ignore]`d, not
return early on a missing file. `resnet_mini_matches_pytorch` and
`whisper_conv_stem_ffn_matches_pytorch` compare against PyTorch output that
`scripts/gen_reference.py` writes into the gitignored `bench/results/`. An
early `return` made them report green while checking nothing, which is how
they went unnoticed for as long as they were uncompiled.

Tests set `SessionConfig` fields and do not write environment variables.
The repository defaults `RUST_TEST_THREADS` to 1 for a shared workstation
GPU; an explicit environment setting or `--test-threads` still overrides
that default. CI passes `--test-threads=4`: most of a test is graph setup
and queue waits, and a hosted runner keeps four sessions in flight.

## Hardware checks

Pin the adapter with `MEGANEURA_DEVICE_ID` and reuse a GPU context across
sessions. The test helpers do this automatically; an explicitly requested
device that cannot be opened is an error. Repeated context creation exhausted
the NVIDIA driver on the RTX 5070 and previously caused silent fallback to
another adapter.

Cooperative attention now keeps scalar-consumed staging in f32 and selects
forward, dQ and dK/dV independently according to available workgroup memory.
The attention oracle and wide-head parity cases cover precision and fallback;
CPU shader checks cover staging types and storage requirements even when the
CI adapter lacks cooperative matrices.

**Padding and `f32` equality.** A padded allocation must not change a
result — every optimizer, clip and accumulation pass bounds its loops by
`s.len` from the segment table, so the poisoned tail never enters the
arithmetic. That much is exact. What padding can legitimately perturb is the
*order* in which f32 values are summed, and that moves results by a unit in
the last place. The LaProp plus adaptive-clip path reduces a workgroup-sized
tree whose lane occupancy follows the tile layout, so reassociating a few
squares in a different order is not exact.

Asserting bit equality across paddings therefore fails on rounding rather
than on a defect. `optimizer_memory::optimizer_clipping_and_diagnostics_ignore_poisoned_allocation_padding`
now compares with a `1e-5` relative tolerance, the same `close` form
`gguf_model` uses. Both observed values sit within one ULP of the exact f64
result, and the padded run is the *closer* of the two. The tolerance is
loose enough for a few ULP and far tighter than the failure it guards
against: a single leaked `1000.0` tail element inflates the adaptive-clip
norm by roughly 2400x, so the check still separates noise from a real leak by
about five orders of magnitude.

A test that compares `f32` results should say which it means. Bit equality is
the right contract for a value that must be reproduced exactly (an inference
logit, a checkpoint byte-for-byte) and the wrong one for an accumulation
whose summation order is an implementation detail.

## Track coverage before pruning

Linux CI instruments its existing full test run with
[cargo-llvm-cov](https://github.com/taiki-e/cargo-llvm-cov), publishes line,
function and region coverage in the job summary, and retains per-file JSON
and browsable HTML as a revision-labelled CI artifact for 30 days. No coverage
files, binaries, or generated experiment records belong in Git.

Locally, install `cargo-llvm-cov` and `rustup component add llvm-tools-preview`:

```sh
cargo llvm-cov clean --workspace
cargo llvm-cov --lib --test smoke --no-report -- --test-threads=1
cargo llvm-cov report --ignore-filename-regex '(^|/)(tests|examples|bench)/' --html
# Open target/llvm-cov/html/index.html, then compare with the full suite:
cargo llvm-cov --tests --no-report -- --test-threads=1
cargo llvm-cov report --ignore-filename-regex '(^|/)(tests|examples|bench)/' --html
```

These are **Rust host** measurements, not WGSL execution coverage. Generating a
shader does not prove its arithmetic correct. Pair coverage with broad GPU
output/gradient parity and state round-trips. Record the adapter, feature
configuration and skips: lavapipe/Metal CI currently skip known unstable
backprop and bit-exact cases; hardware runs must not silently inherit those
skips. Coverage does not establish numerical accuracy or all-platform coverage.

September 8 baseline on RTX 5070, Rust 1.98.0 / cargo-llvm-cov 0.9.1, default
features, no backprop/bit-exact skips (ordinary ignored tests remain ignored):

| Suite | Rust lines | Functions | Regions |
|---|---:|---:|---:|
| Library + broad stack | 77.61% | 76.76% | 77.46% |
| Plus focused/isolated regressions | 84.16% | 80.69% | 83.51% |

The additional tests cover 1,937 host lines, notably compilation (662), runtime
(355), autodiff (257), EfficientNet (239) and eager evaluation (78). They are
not all redundant; removing them wholesale would lose real coverage. The
experiment-only 240-row compensated-dW reporter and its two CPU reference
self-tests were removed from main after this measurement. They do not execute
production Rust; the rejected implementation and reproduction remain at
`evidence/compensated-dw-accuracy-2026-09-06`.

Before retiring a focused test, check its incremental coverage and the
contract it asserts. Preserve a compact case when it catches an otherwise
uncovered behavior (bounds, aliasing, cancellation or state mutation); group
shape variants into a small table inside a broad test. A higher percentage
alone is not a reason to grow the suite. Expensive sweeps and rejected-kernel
qualification belong on experiment refs, not the default regression path.

## Diagnose through the stack

On failure, keep the source revision, configuration and deterministic inputs,
then use the [debugging tools](../README.md#debugging): name values,
materialize the suspect intermediates, disable aliasing/fusion as controls,
inspect `step_debug`, parameter gradients and dispatch provenance, and dump
the implicated shader. For time regressions use uninstrumented timing first,
then a separate dispatch profile. Improve these general tools when diagnosis
is painful instead of encoding each intermediate as a permanent test.

The broad suite itself exercises named intermediate reads, nonfinite
attribution, provenance, and checkpoint restoration. Debuggability is a tested
stack capability, not just a promise to inspect a failed numerical assertion.
