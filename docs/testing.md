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

## Suite layout

`smoke`, `regression` and `oracle` compile their modules into three
executables (`oracle` from `tests/oracle/`). Four environment-mutating test
targets remain process-isolated, for seven integration executables. Add a
case to an existing module, or register a module in one of these suites;
`autotests = false` deliberately stops new files becoming new link jobs.
The repository defaults `RUST_TEST_THREADS` to 1 for its shared GPU; an explicit
environment setting or `--test-threads` still overrides that default.
For example, the old `--test checkpoint_validation` selection becomes
`--test smoke checkpoint_validation::`. `--test tune` becomes
`--test regression tune::`.

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
