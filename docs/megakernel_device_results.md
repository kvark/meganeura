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
VLM sequence length 16. SGD learning rates are 0.001 for CI and 1e-5 for SmolVLA.

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

## Selective device queues

Clean commit `b8a1905` retains portable queues for independent and singleton
regions. The same 50-pair primary and ten-pair diagnostic procedure was repeated,
with unchanged graph launch counts and operator coverage. Every numerical and
queue-completion comparison still passed exactly.

| Case | Ordinary median ms | Selected-path median ms | Median paired ratio | Ordinary/selected p95 ms |
|---|---:|---:|---:|---:|
| CI inference | 0.05212 | 0.05369 | 1.034 | 0.05545 / 0.05944 |
| CI forward/backward | 0.06544 | 0.08108 | 1.244 | 0.06974 / 0.08535 |
| CI forward/backward/SGD | 0.06728 | 0.08235 | 1.232 | 0.07166 / 0.08740 |
| SmolVLA forward | 4.43738 | 5.39863 | 1.214 | 4.85093 / 5.78077 |
| SmolVLA forward/backward/SGD | 15.42334 | 18.03511 | 1.170 | 79.22423 / 85.18203 |

The CI cases now use zero native device regions. Model forward uses two native
regions and 16 portable regions; training uses three native and 80 portable
regions. Ordinary graph fallback remains 133 and 435 launches respectively.
Forward IQRs were 4.16525–4.71502 ms ordinary and 5.03162–5.74027 ms selected;
training IQRs were 15.06895–15.51092 and 14.90072–18.10892 ms. Retain the training
variation and tail outliers when interpreting its median ratio.

CI queue residency falls to 1,280 bytes for inference and 4,608 for training
because those sessions need no reset image. Model residency remains 599.878 MiB
forward and 1027.737 MiB training, including 77,312 and 265,216 scheduler bytes.
Observed ordinary/selected construction ms were 0.85/13.54, 11.45/17.11,
55.71/78.42 and 75.17/92.43 for CI inference/training and model forward/training.
The same shared-context/cache caveat applies.

Diagnostic median GPU pass intervals, ordinary/selected in ms, were
0.01346/0.01638, 0.02656/0.03960 and 0.03066/0.04354 for the three CI cases;
3.46357/4.25846 for model forward and 14.37938/16.89542 for model training.
All worker totals matched. Native model regions used only one to three active
workers in the final samples, processing 208 forward and 221 training tasks.
Portable regions used up to 50 and 64 workers, processing 753 and 9,077 tasks.
This is consistent with idle workers retiring before downstream fan-out becomes
ready; bounded idle retries are a possible next scheduler improvement, without
requiring a global rendezvous or a residency assumption.

The narrower selection reduces overhead but still loses to ordinary execution.
It remains opt-in. Raw records retain the same arguments, boundaries, source
identity, full distributions, coverage and diagnostics:

| Record | SHA-256 |
|---|---|
| `device-selective-ci-paired.json` | `45a3dfb4808225935e3d041165ee24df0b94823d6ce6f3de3f1d4952ff136f14` |
| `device-selective-smolvla-paired.json` | `0a05cb542b296bd03e260d2c487ae46e8bb5a0685b024febc49859eabf5dd139` |
| `device-selective-ci-diagnostic.json` | `5914316fff5c6fca71a6ed3532ef1e123f8cf8239189b56bc2eadd88308fe33c` |
| `device-selective-smolvla-diagnostic.json` | `f58229e054a3f0a33d4c6d57ab9057e4e00e060e1dba672be649f79a3f3cf30e` |

## Bounded idle retries

Clean commit `a2dc351` permits 64 unsuccessful ready scans before retirement,
with earlier exit when no tasks remain. The same 50-pair model procedure and
separate ten-pair diagnostics were repeated. CI was not repeated because it
contains no native regions and its execution path is unchanged. All full
output, gradient, updated-parameter and queue-completion checks passed exactly.
Coverage, launch boundaries and primary residency are unchanged.

| Case | Ordinary median ms | Selected-path median ms | Median paired ratio | Ordinary/selected p95 ms |
|---|---:|---:|---:|---:|
| SmolVLA forward | 4.40998 | 4.85037 | 1.091 | 4.85238 / 5.27048 |
| SmolVLA forward/backward/SGD | 15.40930 | 17.55240 | 1.139 | 80.09333 / 85.00842 |

Forward IQRs were 3.74203–4.50464 ms ordinary and 4.08145–4.99782 ms selected.
Training IQRs were 12.98529–74.05009 and 17.17648–81.35342 ms; its paired-ratio
IQR was 1.063–1.365, with p95 6.968. These large timing outliers limit conclusions
about the training improvement relative to the previous run. Construction was
55.60/112.76 ms forward and 75.84/123.46 ms training, subject to the same
shared-context and cache caveat.

Diagnostic median GPU intervals were 3.46770/3.79272 ms forward and
14.36165/16.42907 ms training. In each final diagnostic step the 194-task
bias/bias/RMSNorm region used 60 workers, compared with three before retries.
The smaller native regions used 12 forward workers and five/16 training workers;
every histogram total matched its task count. This supports retaining bounded
retries to improve task distribution, while ordinary execution remains faster
and remains the default. Broader operator coverage is the next step.

| Record | SHA-256 |
|---|---|
| `device-retry-smolvla-paired.json` | `7803fd294005d70718012097be4c975d37103e89e45d786708f9ca7f49d47ddc` |
| `device-retry-smolvla-diagnostic.json` | `feab04019b2df8068751ae9279a2f68f3b99408f84cb8a6cbc8342f229eac7c1` |

## Selected kernel tiles

Clean commit `55fb598aac3d9ad7baac1bae51e9f54fc3468169` converts the selected
ordinary compute implementations into callable tiles. Blade is `5011dea` and
Naga remains `f09d62e`. Measurements on October 24 use the same RTX 5080,
driver 595.91.07, Rust 1.98 and benchmark boundaries described above. The GPU
was idle before the sequential runs; no other test or build ran during timed
measurement. Clocks were not locked. Each primary case has 50 alternating pairs
after at least ten pairs and one second of warmup; separate diagnostics use ten
pairs after at least five pairs and 500 ms. Both compiled plans and selected
arithmetic implementation keys match. All outputs, gradients, updated parameters
and queue-completion checks are exact in both primary and diagnostic runs.

| Case | Ordinary median ms | Selected median ms | Median paired ratio | Ordinary/selected p95 ms | Graph launches |
|---|---:|---:|---:|---:|---:|
| CI inference | 0.04920 | 0.05468 | 1.108 | 0.05549 / 0.06006 | 5 → 5 |
| CI forward/backward | 0.06550 | 0.08386 | 1.283 | 0.06968 / 0.08834 | 18 → 13 |
| CI forward/backward/SGD | 0.06814 | 0.08682 | 1.274 | 0.07251 / 0.09226 | 18 → 13 |
| SmolVLA forward | 4.46189 | 8.63320 | 1.930 | 4.88305 / 8.97097 | 154 → 136 |
| SmolVLA forward/backward/SGD | 15.10691 | 24.75937 | 1.643 | 77.49349 / 103.67405 | 523 → 363 |

All compiled operations are covered in these cases: 5/5 and 18/18 for CI,
154/154 and 523/523 for SmolVLA. There are no ordinary graph fallbacks. CI has
five portable regions forward and 12 portable plus one native region training.
SmolVLA has 132 portable plus four native regions forward, and 349 portable plus
14 native regions training. **Appended SGD remains ordinary fallback** and is
included in completed training latency but excluded from graph launch counts.
This is broader graph coverage, not complete persistent training-state coverage.

Forward model IQRs are 4.43168–4.79281 ms ordinary and 8.59069–8.91568 ms selected;
the paired-ratio IQR is 1.884–1.961. Training IQRs are 12.23420–15.47590 and
24.39593–86.99833 ms, with paired-ratio IQR 1.600–5.532 and p95 8.627. The large
training outliers remain in the raw records. Reduced launches do not compensate
for tile execution overhead here; ordinary execution remains the default.

| Case | Resident MiB ordinary → selected | Scheduler bytes | Construction ms ordinary / selected |
|---|---:|---:|---:|
| CI inference | 0.06995 → 0.08289 | 1,280 | 0.86 / 50.13 |
| CI training | 0.13229 → 0.17171 | 6,656 | 10.26 / 183.32 |
| SmolVLA forward | 563.42498 → 599.87085 | 69,632 | 62.96 / 852.25 |
| SmolVLA training | 982.08075 → 1027.66156 | 185,856 | 80.38 / 1872.68 |

Lifetime aliasing remains disabled for persistent sessions. Construction uses a
shared context, ordinary sessions first and shader-cache reuse, so these are
observed costs rather than matched cold-start measurements. Runtime shader
conversion/composition introduces substantial construction cost in this version.

Diagnostic median GPU pass intervals, ordinary/selected in ms, are
0.01330/0.01920, 0.02792/0.04781 and 0.03061/0.05171 for the three CI cases;
4.08541/8.20472 for model forward and 14.35851/21.86987 for model training.
These are instrumented completed-pass intervals, not primary wall timings or
individual handler timings. Every worker histogram sums to its logical task
count. Native model regions use 9–64 active workers forward and 12–64 training;
portable model regions use 40–64 and 1–64 respectively. This does not measure
hardware occupancy. Model persistent executable register counts range from
16–125 forward and 16–172 training; shared bytes range from 16–33,552 and
16–42,000. The implausible 64 GiB local-memory statistic is retained raw and
excluded from resource conclusions.

The reproduction commands above are unchanged. All four records are retained
under `target/megakernel-results/`, outside Git, with matching `.log` files:

| Record | SHA-256 |
|---|---|
| `generic-tile-ci-paired.json` | `ac0161bb402996a35a14565ac23dc7bda08eea6610e4e5575bc4d986b49a0779` |
| `generic-tile-smolvla-paired.json` | `7eeb8085b29b3fda08a8f8f076a419772a58825484e5f18b6a583552882200b9` |
| `generic-tile-ci-diagnostic.json` | `548ed54f817b3b068cda9ea5dc6f862546eb8d069d1410230029c8aad871f47c` |
| `generic-tile-smolvla-diagnostic.json` | `e814251cf7517bdc95a6021c9d3b91a7940a764cf702564eeaad7f2bec2275d3` |
