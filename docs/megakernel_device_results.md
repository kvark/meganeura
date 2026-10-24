# Initial device-region qualification

Measured October 24, 2026, at clean commit `275456d0f975d73f4eb5803a8f1b2cdd28c21788`,
with Blade `63fb651` and Naga `f09d62e`, Rust 1.98, NVIDIA RTX 5080 and driver
595.91.07. This is the first Session integration of the qualified fence protocol.
It applies the device queue to every eligible region, including singleton
operations. Ordinary execution remains the default.

All comparisons were exact: full logits or loss, every parameter gradient and
every updated parameter, plus drained device queues. Correctness suites passed
168 oracle, 182 regression, 72 smoke and 22 GGUF tests under device selection;
ordinary smoke and the five focused Session checks on NVIDIA/lavapipe also
passed. Full operator coverage is still limited by ordinary fallback.

## Paired completed-step measurements

The [existing paired harness](testing.md) used 64 workers, at least ten warmup
pairs and one second of warmup, then 50 alternating ordinary/device pairs.
Kernel search was disabled, compiled plans matched exactly, and fixed state was
restored outside timing. Primary runs disabled timestamps and worker counters.
Queue completion validation and numerical comparison ran outside timing.
The CI boundary includes upload, completed execution and output/loss readback;
SmolVLA measures completed execution with uploads and readback outside timing.
The model is the same deterministic random-weight, 16-layer base action expert
used in [Milestone 1](megakernel_session_results.md), with action length 50 and
VLM sequence length 16. Training includes SGD at learning rate 0.001.

| Case | Ordinary median ms | Device median ms | Median paired ratio | Ordinary/device p95 ms | Graph launches |
|---|---:|---:|---:|---:|---:|
| CI inference | 0.05126 | 0.06567 | 1.282 | 0.05529 / 0.06977 | 5 → 5 |
| CI forward/backward | 0.06598 | 0.10847 | 1.653 | 0.07005 / 0.11286 | 18 → 18 |
| CI forward/backward/SGD | 0.06710 | 0.11001 | 1.638 | 0.07275 / 0.11659 | 18 → 18 |
| SmolVLA forward | 4.38559 | 5.88388 | 1.344 | 4.39977 / 5.90875 | 154 → 151 |
| SmolVLA forward/backward/SGD | 15.46200 | 19.80327 | 1.285 | 79.69025 / 86.91250 | 523 → 518 |

The forward model IQRs were 3.98564–4.39014 ms ordinary and
5.43118–5.89511 ms device; training IQRs were 15.06149–15.52737 and
19.45489–19.91248 ms. Training retains substantial tail outliers on both paths.
Ratios are medians of paired samples, not ratios of the two medians.

Coverage remains 2/5 compiled operations in CI inference, 5/18 in CI training,
21/154 in SmolVLA forward and 88/523 in model training. The model uses 18 and
83 device regions respectively, with 133 and 435 ordinary graph launches.
Optimizer updates remain ordinary and are reported separately from graph
launches. The CI cases have no multi-operation region, so the device protocol
adds overhead without reducing any launch count.

Resident buffer MiB were 0.070 → 0.084 for CI inference, 0.132 → 0.174 for CI
training, 563.425 → 599.878 for model forward and 982.081 → 1027.737 for model
training. Device queue/reset-image bytes were 2,560, 9,216, 77,312 and 265,216.
Most of the model residency increase comes from keeping intermediate lifetime
aliasing disabled, as in Milestone 1.

Observed ordinary/device construction times were 0.87/29.46 ms for CI inference,
10.31/42.84 ms for CI training, 56.66/159.98 ms for model forward and
93.29/164.59 ms for model training. These use a shared GPU context, ordinary
construction first and driver cache reuse; they are not matched cold-start
comparisons.

## Separate diagnostic runs

Ten alternating pairs followed at least five warmup pairs and 500 ms of warmup,
with timestamps, worker counters and native pipeline statistics enabled.
Median GPU pass intervals, ordinary/device in ms, were:

| Case | Ordinary | Device |
|---|---:|---:|
| CI inference | 0.01317 | 0.02947 |
| CI forward/backward | 0.02642 | 0.06824 |
| CI forward/backward/SGD | 0.03134 | 0.07219 |
| SmolVLA forward | 3.46250 | 4.77677 |
| SmolVLA forward/backward/SGD | 12.83194 | 18.58598 |

Worker totals matched all logical tasks. In the final recorded diagnostic
steps, CI regions used one active worker each, model forward regions used one
to four and model training regions one to 44 out of 64 launched workers.
These are per-region task histograms, not hardware occupancy measurements.
The driver again reports an implausible 64 GiB local-memory statistic; it is
retained raw and excluded from resource conclusions.

## Decision and retained evidence

The first implementation improves launch counts modestly and regresses whole
program performance. Retain the result. The next planner change should use the
portable queue when a region has no inter-operation dependencies, reserving the
device protocol for regions that need it. Further operator coverage and binding
composition are still needed before claiming full device-scheduled graphs.

Run sequentially on an otherwise idle GPU, with
`VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json` and
`systemd-run --user --scope --slice=ml.slice` before each Cargo command:

```sh
cargo run --release --example bench_ci_latency --offline -- --device-scheduling
cargo run --release --example bench_smolvla_train --features models --offline -- --device-scheduling --force
MEGANEURA_GPU_CAPTURE=1 cargo run --release --example bench_ci_latency --offline -- \
  --device-scheduling --diagnostics --warmup 5 --warmup-ms 500 --runs 10
MEGANEURA_GPU_CAPTURE=1 cargo run --release --example bench_smolvla_train --features models --offline -- \
  --device-scheduling --diagnostics --warmup 5 --warmup-ms 500 --runs 10 --force
```

Raw JSON and corresponding logs remain in ignored `target/megakernel-results/`:

| Record | SHA-256 |
|---|---|
| `device-ci-paired.json` | `b6086e3dc1ace4d96faa141b36702993fb8a831d8631b6f5ff820c69b4be2b19` |
| `device-smolvla-paired.json` | `e5adc7f74649df7456d4627497cc80ddfd267290e6fec1c65297eb817aac1385` |
| `device-ci-diagnostic.json` | `5cd098f41b976129627761c5cde135a6e81b29658a3332357cf0247e50aab85e` |
| `device-smolvla-diagnostic.json` | `36430c131d9ec4693aef33492bcb6bad311e7dc73810e09c2d29b776353767bb` |
