# P3HPC cohort: September 21, 2026

The current paper uses the v10 collection. It keeps partial campaigns,
graphics-only qualification, and the earlier H100 size study separate.
No benchmark, retry, protocol change, or outlier removal was performed
during this paper update.

## Identity and health

All nine current archives use:

- Inferena `06f2f800a4a57254d25b3331f1876631305a15ab`
- Meganeura `dbb43648b31237409075bbd3fa077ce06ee510bc`
- Blade `eaff5092096aab136f11fa728b81c1bed3c0dcd4`
- Python 3.13.13 and PyTorch 2.13.0, source
  `cf30153c4c131c8164ee7798e5022d810682e2cb`

The native optimizer is calibrated egglog construction. All GPU references
use default compilation; CUDA/HIP/XPU replay complete phases after numerical
checks. MPS compiles without an equivalent public whole-phase replay API.
All shared checkpoint hashes and the within-cohort Cargo.lock agree.
[cohort.sha256](artifact/cohort.sha256) identifies current and earlier
archives separately. The existing `paper-p3hpc-2026` tag still identifies
the earlier engine, `428fc2d2`; it was not moved.

| GPU comparison | Valid / planned pairs | Status |
|---|---:|---|
| RTX 5070, Linux | 30 / 30 | Complete |
| H100 80GB, Linux | 30 / 30 | Complete |
| RTX 3050, Windows | 30 / 30 | Complete |
| RX 7900 XT | 18 / 30 | Interrupted during r2 accelerated ResNet |
| Radeon 780M | 25 / 30 | PyTorch/HIP failure during r3 accelerated SmolLM2 |
| Arc B570 | 30 / 30 | Complete |
| Apple M3 | 30 / 30 | Complete |

Total: **193 valid pairs, one failed pair, one interrupted pair, and 15
unreached pairs**. The 780M has all 15 strict pairs. The 7900 XT has two
strict replicates per workload; accelerated ResNet and Whisper have one,
and its other accelerated conditions have two. All five accelerated 780M
conditions have two valid pairs.

The 780M log reports an unspecified HIP launch failure during training
preparation, after compilation completed in 67.584 seconds. Meganeura's
record for that failed pair is successful, but it is not a paired timing.
The 7900 XT log records Ctrl-C/KeyboardInterrupt. Its runner stops after
announcing PyTorch, with a still-running compilation receipt and no result
JSON. That does not establish a compiler timeout or framework crash.

| Graphics-only qualification | Native identity | Passed conditions |
|---|---|---:|
| Intel RPL-U | Intel Graphics (RPL-U), Vulkan | 10 / 10 |
| Ryzen 5 9600X integrated Radeon | RADV RAPHAEL_MENDOCINO, device 5056 | 10 / 10 |

These are one-process qualifications against eager CPU PyTorch, not timing
campaigns. The Ryzen run uses a ROCm wheel but explicitly selects CPU for
the oracle. Neither CPU reference contributes to performance, preparation,
or search aggregates. The AMD name is the observed 9600X adapter, not a
separate Mendocino APU. See [qualification instructions](QUALIFICATION.md).

The separately supplied Ryzen GPU bring-up report records SIGSEGV without
an architecture override and incorrect eager/compiled output agreement with
one. The wheel reportedly lacks gfx1036 Tensile coverage. These are reported
GPU availability findings; the current archive independently verifies the
Vulkan qualification, not the diagnosis of those earlier GPU failures.

Every completed pair passes the offline audit: raw/joined identity, source
and checkpoint identity, timing medians, outer numerical gates, full replay
statistics, and native construction/qualification receipts. Maxima across
193 GPU and 20 CPU-oracle pairs are 0.6281% sampled-output L2, 0.07954% loss,
2.7218% total-gradient-norm error, and 3.0470% parameter-norm-vector L2.
Every pair individually meets the 5% gradient limits.

The five complete GPU campaigns' stored replication reports match the
recomputed values. The auditor also recomputes the three-process rule for
strict 780M, whose interrupted campaign has no final report. Incomplete
groups are never described as having completed that rule.

## Performance and the comparison with the earlier cohort

Ratios are Meganeura/PyTorch synchronized wall time. A ratio below one
favors Meganeura. Tables show every completed condition with its actual
strict/accelerated replicate counts. Aggregates require all five workloads
to have three valid process pairs for that device and arithmetic contract.

| Contract / phase | Median ratio | Nominal native wins | Platform count |
|---|---:|---:|---:|
| Strict inference | 1.988 | 5 / 30 | 6 |
| Strict minimal shape | 1.575 | 10 / 30 | 6 |
| Strict F+L+B | 2.294 | 2 / 30 | 6 |
| Accelerated inference | 2.746 | 1 / 25 | 5 |
| Accelerated minimal shape | 1.746 | 4 / 25 | 5 |
| Accelerated F+L+B | 3.263 | 0 / 25 | 5 |

Strict includes 780M but excludes 7900 XT. Accelerated excludes both AMD
devices. Different platform populations prevent using these aggregate
differences as a precision ablation. Strict Pennycook workload means are
0.47/0.93 inference, 0.58/0.91 minimal, and 0.38/0.99 training
(Meganeura/PyTorch), over the same six systems for every workload.

For the version comparison, the earlier v9 data use Inferena `fa5a04e1`
and Meganeura `428fc2d2`. Both cohorts already enable native tuning,
strict native-f32 cooperative tiles, and the same PyTorch compilation/replay
policies. The new cohort changes the native optimizer, its search scope,
construction/validation costs, and the parameter-norm representation.
This is not an isolated egglog or tuning ablation.

Using exactly the current aggregate's devices and workloads on both sides:

| Contract / phase | Earlier ratio | Current ratio |
|---|---:|---:|
| Strict inference | 1.938 | 1.988 |
| Strict minimal | 1.510 | 1.575 |
| Strict training | 2.580 | 2.294 |
| Accelerated inference | 2.623 | 2.746 |
| Accelerated minimal | 1.751 | 1.746 |
| Accelerated training | 3.513 | 3.263 |

Useful native changes include 1.59x faster strict one-token SmolLM2 and
1.46x faster diffusion training on RTX 5070, and 1.99x faster minimal
SmolVLA on B570. Regressions include H100 strict SmolLM2/SmolVLA minimal
forwards taking 1.75x/2.62x as long, and B570 strict ResNet training taking
1.29x as long. The corresponding PyTorch medians change by less than 1%.

The partial 7900 XT data remain encouraging: strict SmolLM2 and SmolVLA
minimal ratios are 0.19 and 0.22, with SmolVLA training at 0.61.
They are not a substitute for its missing third replicate.

M3 strict diffusion inference spans 6.384–36.048 ms across process medians;
RTX 5070 strict minimal SmolVLA spans 1.116–2.029 ms. All observations remain
in the figure/ranges. The data do not isolate tuning choices, clock policy,
thermals, or co-tenancy as the cause of the variation.

## Search and preparation

The current 60-second soft session deadline covers measured construction,
including initialization and full-model qualification. Up to four graph
forms and 64 programs explore dispatch fusion and fresh submission chunking.
Each program receives up to two seconds of private kernel search, with no
class cap and 1 GiB scratch. Plans/snapshots are bounded separately by 75%
of reported available memory. Large graphs explore one verified repeated
region; this is not exhaustive graph scheduling.

The 193 completed GPU pairs contain 541 sessions and 8,832 program trials.
83 sessions select a non-ordinary graph; 167 report unfinished program
search, and 234 report bounded/truncated alternative extraction. The longest
session search is 66.121 seconds. 117 sessions report no suitable bounded
repeated region and retain the ordinary graph while exploring physical choices.

Private probes visit 122,352 / 171,875 kernel-class instances across programs.
Outcomes include 12,082 FasterCandidate, 76,238 KeepBaseline, 1,240 InvalidOutput,
and 2,566 TimeBudget entries. Whole-program qualification rejects another
86 trials. Rejected challengers cannot win. Repeated/reused comparisons are
not distinct algorithms, and replacement counts are not graph-level speedups.

M3 uses 2,443 cooperative dispatch instances in strict mode. No measured
Vulkan device exposes native-f32 tiles through this stack. B570 exposes
rectangular floating-point cooperative tiles, but the pinned Blade/Naga
path supports only square shapes; it records zero cooperative dispatches
in both contracts. Its poor matrix performance is not evidence that the
hardware lacks matrix acceleration.

Across 30 fully replicated strict groups, native preparation medians span
19.378–188.445 seconds (median 105.325), versus 1.876–84.175 PyTorch
(median 31.239). H100 totals 47.34 minutes native preparation versus 14.12
minutes default compilation. Native search is not generally cheaper here.

Across completed GPU pairs, private kernel search totals 95.43 minutes,
candidate initialization 57.40, and full-program qualification callbacks
64.35. Trial elapsed time includes these components and must not be added
to them as a separate partition. The broader construction boundary also
prevents treating an old/new `compile_s` ratio as shader-compiler slowdown.

## Earlier size study and other evidence

The earlier H100 135M/360M/1.7B series remains intact under its own pins.
All twelve extended-model pairs passed; its 135M baseline is also from v9.
It never fills current coverage or enters current aggregates.

Strict training ratios narrow 3.91 → 3.51 → 2.89 with size; prefill remains
near three. Accelerated training widens 5.29 → 5.64 → 7.10. This establishes
larger-model execution at the earlier revision, not the new optimizer's
scaling. Batch one, 128-token prefill, stateless minimal forward, and no
optimizer/distributed execution limit both studies.

The earlier 1.7B memory row records 16.72 GiB of native plan allocation,
12.86 GiB PyTorch allocator peak, and 13.54 GiB reserved. These different
accounting scopes do not establish a comparative resident-VRAM peak.

The separate v7 H100 pilot retains its actual search-off/on control and
170.3-minute PyTorch max-autotune cost. It is not the current preparation
policy. Existing Nsight/compiler diagnostics keep their own source refs;
no new profile or removable-barrier percentage is claimed.

MI300X's original report remains separate (SHA-256
`e9be97e695140e8a36e09da0f0dd850113b0b22359182921dbc34b689d9776e8`).
[Mesa issue 13399](https://gitlab.freedesktop.org/mesa/mesa/-/work_items/13399)
tracks experimental RADV/CDNA work. It is a plausible route to MI300X Vulkan,
not a driver we qualified or a new benchmark result.

## Reproduction

From the repository root, using standard-library Python 3.11+:

```sh
python3 paper/p3hpc/artifact/cohort.py "$HOME/Downloads/p3hpc" \
  --previous "$HOME/Downloads/p3hpc-v4" \
  --check paper/p3hpc/tables --output target/p3hpc-20260921
```

The supplementary `records.jsonl.xz` supplies all current and selected earlier
JSON values, streamed one file at a time. It can replace both directories:

```sh
python cohort.py records.jsonl.xz --check tables --output regenerated
```

The audit regenerates nine LaTeX fragments and a 70-row current GPU-condition
CSV, plus separately labeled earlier conditions, failures, and search summaries.
No CPU-reference timings appear in that CSV. Raw artifacts and publication
binaries remain outside Git. The two supplied AMD logs and their final runner
logs are included in the supplement, not just summarized in the prose.
