# F32 tile search: whole-step transfer experiment

This is a synthetic engineering experiment, outside the frozen P3HPC result
set. It compares Meganeura's deterministic initial selection with its opt-in
scratch tuner, not with PyTorch or any other engine. No result should be
generalized to model training, serving, or a different device.

## Protocol

[Rust harness](../../../examples/tune_session.rs): four dense inference chains,
deterministic signed full-mantissa input and weight data, default graph
rewrites/scheduling/allocation. No activation or model-specific kernel rule.
Two matched sessions share a context; only the second is tuned. F32 storage
and operands: native-f32 cooperative candidates are allowed if advertised;
otherwise cooperative matrices are disabled rather than substituting f16.

Search uses the default six paired samples × 16 isolated dispatches, 5% plus
paired-MAD guard, 64 MiB scratch cap, and an explicit 10-second soft deadline.
The harness then warms both sessions for 30 steps each and records 40 whole-step
pairs, alternating baseline/tuned and tuned/baseline. Samples include normal
encoding, submission and wait, but exclude uploads, compilation, search and
output readback. All sample arrays retain acquisition order. Whole-graph
output parity uses finite values and relative L2 ≤ 2e-4 against the untuned
output; the tuner's independent f64-dot qualification is a separate check.

Repeat in five independent processes; each rebuilds sessions and reruns search.
Do not select the fastest process or discard a process with no accepted winner.
The per-process 5% + paired-noise guard is descriptive, not a confidence interval
or an automatic runtime rollback. Search amortization must use whole-step
time saved, not isolated-kernel speedup.

## Reproduction

Run on an idle GPU. The harness refuses a dirty tracked checkout or an existing
output path. Each JSON records its actual source revision, compiler,
lockfile/executable hashes, device/driver, complete search report, pipeline keys,
output checks and raw whole-step samples. `Cargo.lock` here preserves the exact
dependency resolution; the root lockfile is intentionally not tracked by the
library. Use this resolution when rebuilding the measured source revision.

```sh
cargo build --release --locked --example tune_session
target/release/examples/tune_session --device
for run in 01 02 03 04 05; do
    target/release/examples/tune_session "new-run-${run}.json"
done
```

The available RTX 5070 / NVIDIA 595.71.05 reports `f32_tile=0`, `f16_tile=16`.
It can qualify and measure the scalar search, not the native-f32 extension.
The scalar GPU test suite already passed four tuning and 19 related regression
tests; native-f32 shaders have offline Naga/SPIR-V coverage only on this host.

## Results

Raw runs and interpretation will be recorded here after the committed harness
has run. Frozen `paper/results` and `paper/tables` are not experiment outputs.
