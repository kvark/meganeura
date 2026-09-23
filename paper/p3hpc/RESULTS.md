# P3HPC cohort: September 21, 2026

The current paper uses the v10 collection. It keeps partial campaigns,
graphics-only qualification, and the earlier H100 size study separate.
The replacement RX 7900 XT archive is complete; it supersedes the accidentally
interrupted upload. No old and new replicates are spliced together.

The next candidate is **Inferena protocol v13**, revision
[`757f6a8`](https://github.com/kvark/inferena/blob/757f6a80b3dee2d6e9392fbbfc28e64febc9fe28/EXPERIMENT.md)
on `experiment/p3hpc-cuda-graphs`, pinned to Meganeura
`dd9bf8acfd69918c4e4feb1db9c40a3b8f44aa36` and Blade
`fbb4f28c4869e81ae15de58925945b423b9c1ac5`. It retains sixteen graph/schedule
forms, interleaves graph and physical-plan choices, and warms paired
comparisons for two pairs and 250 ms within the shared 60-second session
deadline. Both engines still warm held-out timing for five calls and two
seconds. Checkpoints, compilation deadlines, replay requirements and numerical
gates are unchanged. ROCm Whisper now uses eager efficient SDPA inside the
otherwise compiled encoder, following the separate failure investigation below.
That policy also failed a later AMD run. The affected path remains unqualified;
an isolated passing retry does not establish a fix.
Use the instructions in Inferena's `EXPERIMENT.md` to qualify each backend
before collection. Do not relabel the paper's v10 records as v13.

The v11 B570 StableDiffusion gradient failure was investigated separately.
The sinusoidal fixture was nearly rank two and also exceeded the gradient
bounds in PyTorch f32; a GELU derivative bug was fixed independently.
V12 introduced matched uniform parameters and pinned the fix. V13 retains
both, rather than relaxing validation or hiding the failed qualification.

## ROCm Whisper repeatability: separate qualification finding

The RX 7900 XT campaign `rubik-20260923T152328187857Z-2a8cbf52` uses Inferena
`2a8cbf52b8131223d207c054d749c715e625afec`, Meganeura `dd9bf8ac`, and PyTorch
2.13.0+rocm7.2 / ROCm 7.2.53211. Its first four strict pairs pass. Whisper's
PyTorch inference capture passes, but training fails at `uncaptured repeat 1
output 0`, before training HIP capture or cross-engine validation. Meganeura
finishes its run, but that does not establish a valid paired comparison.

The failed encoder output has 576,000 elements. Maximum error is
0.00196732 against a 9.10140e-6 bound; RMS error is 9.50484e-5 against
3.07932e-6. There are 161,419 pointwise mismatches (28.0%); that count is
diagnostic, not an additional acceptance gate.

The AMD-side investigation identifies the switch from sinusoidal weights to
`name-index-uniform-v1` as the point where the repeatability check began to
fail. It reports elevated drift with the earlier weights as well. The new
fixture exposed the failure under the unchanged tolerance rule; it does not
locate the underlying compiler or kernel defect. Do not describe the uniform
initializer as injecting randomness between calls: its values are fixed by
the canonical parameter name and element index.

Initial same-GPU controls found math SDPA inside a compiled encoder failed
even when SDPA itself was eager, while eager efficient SDPA gave bit-exact
training outputs. A fully eager encoder with math SDPA also repeated exactly.
Automatic selection passes some isolated runs but fails in the campaign;
the enabled-backend list alone does not identify the selected kernel.
Inferena `757f6a8` therefore records eager efficient SDPA for ROCm Whisper
only. The encoder remains compiled, HIP replay remains enabled, and no
tolerance is relaxed. This is a numerical portability finding in the AMD
vendor stack, not an installation or collection setup failure.

A later campaign, `rubik-20260923T163520494362Z-757f6a80`, confirms failure with
the eager efficient policy correctly applied at `757f6a80`. The first four strict
pairs pass, with bit-exact ordinary output repeats in all three phases. Whisper
again fails at `uncaptured repeat 1 output 0`, with maximum
error 0.00235241 against 9.10142e-6 and RMS error 9.13354e-5 against 3.07931e-6.
There are 157,226 pointwise mismatches out of 576,000. Compilation completed in
7.90584 seconds and inference capture passed; no accelerated pair was reached.
This counterexample invalidates the policy as an established workaround and
prevents attributing the fault specifically to math SDPA. The underlying defect
remains unresolved. Retrying until a process passes would select successful
runs, not demonstrate repeatability. No tolerance change, automatic retry, or
eager timing substitution is justified by these controls.

Evidence stays outside Git. The earlier `2a8cbf52` upload, `amd-dgpu-temp.tgz`,
had SHA-256
`366a8d035e5a86ed7aa3cf4b37bb2183c7f0a8ca7f2f1f20d182c5eed59792fb`.
That archive establishes the failed uniform-fixture campaign. The initializer
bisection and backend controls were reported separately by the AMD-side
investigation; no quantitative old/new-weight comparison is inferred from
this archive. The replacement upload with that filename contains the
`757f6a80` campaign, SHA-256
`46960bab05d86559153ce5d2f74bdf99d36f1a4a93dd41bc029078db4176da39`;
its manifest, PyTorch error record, preparation receipt and runner log agree.
Neither these failures nor the passing controls enter the v10
tables, portability scores, or supplementary measurement stream.

## Optimizer studies outside the cohort

Do not label the current schedule study as greedy versus egglog: **both arms
already use egglog**. The historical rewrite-only ablation, first recorded in
`22594fe`, found essentially the same active graphs and no GPU improvement
distinguishable from run-to-run variation. Its SmolLM CPU rewrite times were
0.089 ms (greedy), 2.94 ms (outlined egglog), 32.6 ms (windowed), and 56.2 ms
(whole graph); whole differentiated-graph saturation took 7.43 s. Those numbers
describe the earlier rule set, not the current scheduler or an execution
speedup. There is no supported positive GPU effect size for that comparison.

The September 22 schedule diagnostic used
[`2153aeb`](https://github.com/kvark/meganeura/tree/2153aeba92469a494de4a08202faf94e597e3d26),
whose tree is identical to merged `e8d7d9e`. It compares one ordinary extraction
plus private kernel probes with joint graph/schedule search. The exact
[`search_study` example](https://github.com/kvark/meganeura/blob/e8d7d9e3b2192671c0a9fb1aab2e3fc505153260/examples/search_study.rs)
checks every output against an f64 oracle with absolute error bound
`2e-5 + 2e-4 * abs(reference)`. All 24 candidates per case passed on both GPUs.
Cooperative matrices were disabled; clocks were not forced. GPUs ran serially.
Each number is the upper median of eight sums of per-dispatch GPU timestamps
after twelve warmup steps, one process per device. It excludes CPU submission
and is not a multi-process inference benchmark.

| M × N × K | RTX 5070: one graph → joint (µs) | Arc B570: one graph → joint (µs) |
|---|---:|---:|
| 50 × 4096 × 720 | 57.95 → 45.57 | 135.42 → 110.73 |
| 50 × 960 × 720 | 31.36 → 19.01 | 53.85 → 39.12 |
| 50 × 720 × 960 | 25.50 → 17.15 | 63.12 → 38.70 |
| 50 × 960 × 720 + add | 31.42 → 18.94 | 54.17 → 34.17 |

Speedups are 1.27–1.66× on NVIDIA and 1.22–1.63× on Intel. Private probes use
the same defaults, but the total budgets differ: two-second ordinary kernel
probes versus up to forty seconds, sixteen forms and 24 programs for joint
search. Observed per-product construction, qualification and brief readout
increase from 0.4–0.7 to 6.3–7.1 seconds on NVIDIA and 0.8–1.7 to 7.4–12.5
seconds on Intel. This is not a fixed-budget or pure shader-compilation study.

The preceding explicit split-K enumerator at `da30316` already gave
45.3 / 19.5 / 19.0 / 21.2 µs on those NVIDIA shapes, versus one-graph
57.8 / 31.7 / 25.2 / 31.8 µs. Finding split-K explains the gain; the data do
not show that an equality representation beats an equally capable enumerator.
Egglog's benefit is retaining these choices together with graph rewrites in
one transformation system.

To reproduce separately from collection, at the pinned source:

```sh
cargo build --release --example search_study
MEGANEURA_DEVICE_ID=12036 MEGANEURA_GPU_TIMING=1 target/release/examples/search_study
MEGANEURA_DEVICE_ID=57868 MEGANEURA_GPU_TIMING=1 target/release/examples/search_study
```

Those device IDs belong to this workstation; use the intended adapter's ID
elsewhere. The example disables cooperative matrices; the environment setting
enables timestamp readout. Do not run it alongside a collection or include these
times in the publication aggregates. The full-model SmolVLA checkpoint in
[compiler-search.md](../../docs/compiler-search.md#api-and-current-boundary)
also compares two already-egraph revisions and changes multiple policies;
it is not a causal egglog-versus-greedy result.

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
| RX 7900 XT | 30 / 30 | Complete |
| Radeon 780M | 25 / 30 | PyTorch/HIP failure during r3 accelerated SmolLM2 |
| Arc B570 | 30 / 30 | Complete |
| Apple M3 | 30 / 30 | Complete |

Total: **205 valid pairs, one failed pair, and four unreached pairs**.
The 780M has all 15 strict pairs. All five accelerated 780M
conditions have two valid pairs.

The 780M log reports an unspecified HIP launch failure during training
preparation, after compilation completed in 67.584 seconds. Meganeura's
record for that failed pair is successful, but it is not a paired timing.
The replacement 7900 XT campaign has no failed condition. Its original
Ctrl-C log is not a framework crash and no longer describes the selected data.

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
205 GPU and 20 CPU-oracle pairs are 0.6281% sampled-output L2, 0.07954% loss,
2.7218% total-gradient-norm error, and 3.0470% parameter-norm-vector L2.
Every pair individually meets the 5% gradient limits.

The six complete GPU campaigns' stored replication reports match the
recomputed values. The auditor also recomputes the three-process rule for
strict 780M, whose failed campaign has no final report. Incomplete
groups are never described as having completed that rule.

## Performance and the comparison with the earlier cohort

Ratios are Meganeura/PyTorch synchronized wall time. A ratio below one
favors Meganeura. Tables show every completed condition with its actual
strict/accelerated replicate counts. Aggregates require all five workloads
to have three valid process pairs for that device and arithmetic contract.

| Contract / phase | Median ratio | Nominal native wins | Platform count |
|---|---:|---:|---:|
| Strict inference | 1.851 | 9 / 35 | 7 |
| Strict minimal shape | 1.286 | 13 / 35 | 7 |
| Strict F+L+B | 2.154 | 5 / 35 | 7 |
| Accelerated inference | 2.322 | 5 / 30 | 6 |
| Accelerated minimal shape | 1.544 | 8 / 30 | 6 |
| Accelerated F+L+B | 3.051 | 3 / 30 | 6 |

Strict includes all seven GPU-reference platforms. Accelerated excludes the
partial 780M sweep. Different platform populations prevent using these aggregate
differences as a precision ablation. Strict Pennycook workload means are
0.51/0.92 inference, 0.62/0.79 minimal, and 0.41/0.97 training
(Meganeura/PyTorch), over the same seven systems for every workload.

For the version comparison, the earlier v9 data use Inferena `fa5a04e1`
and Meganeura `428fc2d2`. Both cohorts already enable native tuning,
strict native-f32 cooperative tiles, and the same PyTorch compilation/replay
policies. The new cohort changes the native optimizer, its search scope,
construction/validation costs, and the parameter-norm representation.
This is not an isolated egglog or tuning ablation.

Using exactly the current aggregate's devices and workloads on both sides:

| Contract / phase | Earlier ratio | Current ratio |
|---|---:|---:|
| Strict inference | 1.830 | 1.851 |
| Strict minimal | 1.284 | 1.286 |
| Strict training | 2.467 | 2.154 |
| Accelerated inference | 2.462 | 2.322 |
| Accelerated minimal | 1.615 | 1.544 |
| Accelerated training | 3.079 | 3.051 |

Useful native changes include 1.59x faster strict one-token SmolLM2 and
1.46x faster diffusion training on RTX 5070, and 1.99x faster minimal
SmolVLA on B570. Regressions include H100 strict SmolLM2/SmolVLA minimal
forwards taking 1.75x/2.62x as long, and B570 strict ResNet training taking
1.29x as long. The corresponding PyTorch medians change by less than 1%.

The complete 7900 XT data remain encouraging: strict SmolLM2 and SmolVLA
minimal ratios are 0.19 and 0.22, with SmolVLA training at 0.61.

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

The 205 completed GPU pairs contain 574 sessions and 9,462 program trials.
88 sessions select a non-ordinary graph; 171 report unfinished program
search, and 246 report bounded/truncated alternative extraction. The longest
session search is 66.121 seconds. 123 sessions report no suitable bounded
repeated region and retain the ordinary graph while exploring physical choices.

Private probes visit 132,616 / 183,671 kernel-class instances across programs.
Outcomes include 13,098 FasterCandidate, 81,114 KeepBaseline, 1,481 InvalidOutput,
and 2,639 TimeBudget entries. Whole-program qualification rejects another
108 trials. Rejected challengers cannot win. Repeated/reused comparisons are
not distinct algorithms, and replacement counts are not graph-level speedups.

M3 uses 2,443 cooperative dispatch instances in strict mode. No measured
Vulkan device exposes native-f32 tiles through this stack. B570 exposes
rectangular floating-point cooperative tiles, but the pinned Blade/Naga
path supports only square shapes; it records zero cooperative dispatches
in both contracts. Its poor matrix performance is not evidence that the
hardware lacks matrix acceleration.

Across 35 fully replicated strict groups, native preparation medians span
13.759–188.445 seconds (median 92.294), versus 1.876–84.175 PyTorch
(median 30.327). H100 totals 47.34 minutes native preparation versus 14.12
minutes default compilation. Native search is not generally cheaper here.

Across completed GPU pairs, private kernel search totals 98.42 minutes,
candidate initialization 59.80, and full-program qualification callbacks
67.21. Trial elapsed time includes these components and must not be added
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
binaries remain outside Git. The selected 780M failure log and its final runner
log belong in the supplement. The obsolete 7900 XT interrupt log does not
describe the replacement archive. Existing PDF/ZIP packages still need
regeneration; they do not change when this source tree is edited.
