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

## Known failures on real hardware

These reproduce on a clean checkout with all features, and are recorded here
because the CI adapter (lavapipe) does not reach them. Confirm before
attributing one to your change.

**Context churn exhausts the NVIDIA driver.** `NoSupportedDeviceFound` after
roughly ten create/drop cycles in one process; lavapipe and the Intel driver
tolerate it. It fails `vision::conv2d_grad_weight_split_k` and
`vision::conv2d_tuned_kernels` on the RTX 5070, each of which builds nine
sessions. Sharing one context across the whole oracle fixes those two, but it
also changes the attention comparisons' results, so it is not a free win —
the trade needs the attention sensitivity understood first. Two tests must
not be read as evidence that a plan change broke convolution: the failure is
in the harness, not the kernels.

**The attention oracle is order-dependent, and the plan is not why.**
`attention::causal_forward` and its siblings fail under a filter that selects
a subset of the `attention` module and pass when the whole module runs. The
whole `attention` module creates 422 GPU contexts; one test creates 17.

Ruled out by measurement, in order:

- *Context count.* It is not the driver limit — that produces
  `NoSupportedDeviceFound`, and these cases run fine at 422 contexts. It is
  not "warm-up" either: `attention_autodiff` runs before `causal_forward`
  alphabetically and creates no context at all.
- *The plan.* Hashing every dispatch's `shader`, `params`, `workgroups` and
  buffer bindings gives `0x384f3e69d4f3f584` for the failing `q=31, dim=256`
  case in both runs — byte-identical. `MEGANEURA_DUMP_PLAN` agrees, and the
  sweep's own kernel list agrees (`MultiHeadAttn`, `coop=false`).
- *The operands.* `Feeds::fill_random` is seeded from a constant
  (`100 + shape_index`), `Feeds::set` widens f32 to f64, and the graph is the
  same object either way.
- *The tolerance.* `Tolerance::default()` is a constant (`rtol` 2e-4,
  `floor` 1e-3) and `Options::default()` re-reads it per call.

What is left is the GPU returning a different result for the same plan and the
same inputs: element 0 is `8.7646484e-1` alone and `8.7633395e-1` in the
module, a relative difference of `1.5e-4` — about 2500 ULP at that magnitude,
so a genuine reduction-order difference rather than rounding. Something
process-wide changes how the driver executes the reduction, and it is not any
of the above. Candidates not yet eliminated: driver-side shader cache state,
and an interaction with the `profiler`'s process-global armed buffer, which
`init_gpu_context_with` arms on every call. Both are outside this crate's
control until someone can hold a fixed plan and vary only the process history.

Until that is understood, do not "fix" these by loosening the tolerance — the
whole module passing is evidence the kernels are correct, so the subset
failures are a harness artefact, not a kernel defect. And do not read a
failing attention case as evidence that a scheduling change broke something.

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

## Claims that were checked and did not hold

Recorded because an audit that only keeps its hits teaches the wrong lesson,
and because each of these looked like a defect on inspection.

**Adding a `ShaderEntry` variant already breaks the build.** There are ~105
variants and five exhaustive matches over them — `profile_family`,
`shader_group`, `entry_point` in `compile.rs`, and `shader_data_layout` and
`bind_dispatch` in `runtime.rs`. A new variant produces five `E0004`s naming
the function and line. So the "15 coordinated edits" cost is not what the
type system sees; what it does not check is the WGSL and the pipeline `key()`,
and `key()` is a cascade over *dispatch shape* (kernel, epilogue, weight
format, coop), not over entries, so it has no per-entry arm to miss.

**RMSNorm's `(2..=32)` rows-per-workgroup bound is not an occupancy bug.** At
`cols == 64` the bound gives `rows_per_workgroup == 1`, so a 256-thread
workgroup covers one row: 64 lanes compute, 192 exit early, and the reduction
tree still runs over all 256. The obvious fix — extending the bound to
`(2..=64)` so eight rows share a workgroup — measures no better. On an RTX
5070, 65536 rows, 200 runs: `cols=64` at 6.7 GB/s against `cols=32` at
6.6 GB/s, a ratio of 0.98 and 1.00 across two runs. The kernel runs at ~6.6
GB/s, which is memory-bound, so idle lanes cost nothing. Widening the bound
would only churn the reduction order for nothing. Do not "fix" this without a
new measurement showing the kernel has become compute-bound.

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
