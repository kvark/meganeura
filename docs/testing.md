# Regression checks and diagnosis

Iteration should exercise a few broad stack contracts, then use the same
diagnostic tools on a failing application. A test per symptom is not the goal.

```sh
# CPU compiler/graph checks; no GPU required.
cargo test --lib

# Broad GPU coverage: forward/backward, optimizer/checkpoint state, cache,
# model training, vision operators, named reads and nonfinite attribution.
cargo test --test smoke -- --test-threads=1

# CI/full regression check, including the narrower historical cases.
cargo test --tests -- --test-threads=1
```

`smoke` and `regression` compile their modules into two executables. Four
environment-mutating test targets remain process-isolated. This reduces
integration executables from 50 to 6. Add a
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
then follow [the observability guide](study/observability.md): name values,
materialize the suspect intermediates, disable aliasing/fusion as controls,
inspect `step_debug`, parameter gradients and dispatch provenance, and dump
the implicated shader. For time regressions use uninstrumented timing first,
then a separate dispatch profile. Improve these general tools when diagnosis
is painful instead of encoding each intermediate as a permanent test.

The broad suite itself exercises named intermediate reads, nonfinite
attribution, provenance, and checkpoint restoration. Debuggability is a tested
stack capability, not just a promise to inspect a failed numerical assertion.
