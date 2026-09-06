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

Measured source: `0e27b68e92bfb72745eb2f582107236cd541c409`, Rust 1.98.0,
Linux x86_64, Intel Core i5-12400F host, RTX 5070 / NVIDIA 595.71.05.
The original source is retained by Git tag `evidence/tuning-2026-09-05`.
The development branch was subsequently rebased onto `8069cf3` and extended
with checkpoint/memory work. Those changes do not alter these raw records or
turn them into measurements of the newer source; reproduce this table from
the tagged revision and retained lockfile.

All five processes used the same lockfile and executable hashes. The GPU
reported 0% utilization before the serial experiment; no other qualification
or benchmark was run concurrently. Clocks/power were not locked and driver
shader caches were not cleared between runs; observed search costs are not a
clean-install cold-start estimate. Sequential processes on one host are
repeatability evidence, not independent fleet samples.

Raw evidence: [run 1](run-01.json), [run 2](run-02.json), [run 3](run-03.json),
[run 4](run-04.json), [run 5](run-05.json). No runs were discarded.

| Dense chain (rows × input → width; layers) | Whole-step ms, baseline → tuned¹ | Median speedup² (process range) | Runs above guard | Changed dispatches | Median search cost |
|---|---:|---:|---:|---:|---:|
| 33 × 17 → 65; 4 | 0.0562 → 0.0562 | 1.000× (1.000–1.019×) | 0/5 | 0 | 11.5 ms |
| 32 × 256 → 256; 8 | 0.1118 → 0.1117 | 1.001× (0.990–1.002×) | 0/5 | 0 | 20.2 ms |
| 128 × 512 → 512; 8 | 0.3078 → 0.2678 | 1.151× (1.127–1.155×) | 5/5 | 8 | 65.7 ms |
| 64 × 1024 → 1024; 4 | 0.3007 → 0.2678 | 1.127× (1.118–1.132×) | 5/5 | 4 | 98.3 ms |

¹ Each entry is the median of five per-process medians.
² Median of five within-process ratios, not the ratio of the aggregated times.

All 60 class comparisons qualified; 30 retained the incumbent and 30 chose the
challenger. Both larger chains consistently switched every dispatch from
64×64 to 32×32 scalar tiles; both smaller chains retained 32×32. Every graph's
final output matched its untuned reference exactly in this experiment.
This does not promise bitwise equality for all f32 implementations or inputs.

The larger cases clear the descriptive whole-step guard in all five processes.
Median per-process amortization estimates are **1,617 steps** for the 512-wide
chain and **2,850 steps** for the 1024-wide chain, using search cost divided by
measured whole-step time saved. The smaller cases pay search cost with no
demonstrated benefit. These costs exclude ordinary session construction and
do not count this experiment's extra control-session/confirmation overhead.

The two larger shapes have exactly 16 initial 64-tile output workgroups, where
the static `<16` small-tile rule retains Tile64. The measured search challenges
that boundary without changing the cutoff, adding a device-name exception, or
recognizing a model. Three classes per chain arise from shape and binding
placement; the final host-visible output is not pooled with device-local
intermediates even at equal M/N/K.

CPU record replay: `cargo test --test tuning_evidence -- --nocapture` recomputes
the raw-sample medians, paired guards, kernel decisions, pipeline-change counts
and process speedups. This checks evidence consistency, not GPU correctness.

Interpretation: this is positive evidence that isolated scalar choices can
transfer to whole-step gains at this heuristic boundary. It does **not** show
a PyTorch win, a model-training gain, native-f32 cooperative performance,
transfer across devices, or enough holdout coverage to enable tuning by default.
No hardware-specific threshold was changed in response to these four shapes.
Next: broader held-out shapes/models, native-f32 hardware, and state-safe
whole-step confirmation before persistence/default-on. Frozen `paper/results`
and `paper/tables` remain unchanged.
