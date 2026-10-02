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

**The attention oracle is order-dependent.** `attention::causal_forward` and
its siblings fail when run under a filter that selects a subset of the
`attention` module, and pass when the whole module runs. Plans, coop policy
and workgroup geometry are identical either way, so the difference is in
state the process carries between cases. Tolerance misses are small and
consistent with f16 rounding on the large-head-dim cases (`dim` 128 and 256),
so they are not obviously the same problem as the context churn above.

**Padding changes optimizer results in one configuration.** In
`optimizer_memory::optimizer_clipping_and_diagnostics_ignore_poisoned_allocation_padding`,
LaProp plus *global* gradient clipping plus a padded gradient allocation
yields a last-ULP difference against the same run with no padding. Adam is
unaffected, and adaptive clipping is unaffected; the gradients themselves
are bit-identical in every combination, so the divergence is in the clip
pass, not in backward. `optimizer_len` (`runtime.rs`) falls back to
`plan.buffers[param] / 4` when a buffer has no `param_types` entry, and a
gradient never has one — so the fallback returns the *padded* element count
for a gradient-sized allocation. That path is the thing to check first.

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
