# P3HPC cohort: September 21, 2026

The current paper uses the v10 collection. It keeps partial campaigns,
graphics-only qualification, and the earlier H100 size study separate.
The replacement RX 7900 XT archive is complete; it supersedes the accidentally
interrupted upload. No old and new replicates are spliced together.

The next candidate is **Inferena protocol v14**, revision
[`adfa3e9`](https://github.com/kvark/inferena/blob/adfa3e93ae564e41ddd7a2210591b550a8175aa6/EXPERIMENT.md)
on `experiment/p3hpc-cuda-graphs`, pinned to Meganeura
`2f018f9722dd403f14c33e4361bbff60f3dc2083` and Blade
`fbb4f28c4869e81ae15de58925945b423b9c1ac5`. It retains sixteen graph/schedule
forms, interleaves graph and physical-plan choices, and warms paired
comparisons for two pairs and 250 ms within the shared 60-second session
deadline. Both engines still warm held-out timing for five calls and two
seconds. Checkpoints, compilation deadlines, replay requirements and numerical
gates are unchanged. The pin includes #211's reference suite and runtime
correctness fixes, #212's reference-check guards, and #213's simplified
lowerings, training Winograd and attention changes. It also includes #214's
compact search receipts and reference-scale correction. Uncaptured PyTorch calls
now receive the same full-tensor repeatability checks as captured calls.
Failed compiled phases retain their failure, without discarding previously
qualified inference. One separate
eager/math diagnostic may validate native results, but never supplies compiled
timings. The remaining planned conditions continue after classified numerical
failures; unknown execution faults still stop collection. Incomplete phase
replicate groups are excluded from performance aggregates.
ROCm Whisper returns to ordinary compiled automatic SDPA: the eager-efficient
workaround did not restore repeatability. The failure remains unresolved;
an isolated passing retry does not establish a fix.
Use the instructions in Inferena's `EXPERIMENT.md` to qualify each backend
before collection. Do not relabel the paper's v10 records as v14.

The v11 B570 StableDiffusion gradient failure was investigated separately.
The sinusoidal fixture was nearly rank two and also exceeded the gradient
bounds in PyTorch f32; a GELU derivative bug was fixed independently.
V12 introduced matched uniform parameters and pinned the fix. Later candidates
retain both, rather than relaxing validation or hiding the failed qualification.

## Reference-suite review and timing checks: September 23

Merged [#211](https://github.com/kvark/meganeura/pull/211) adds an independent
f64 graph interpreter, finite-difference checks of symbolic derivatives,
GPU comparisons of complete outputs and gradients, and composed/random graph
checks through the compiler. It also fixes production code: gradient-observable
fusion, scalar loss reductions, normalization cancellation, GELU tails, RoPE
indexing, and cooperative-convolution bounds. It is not just a test-only change.
The new paper section describes these checks separately from the benchmark's
sampled cross-engine comparison and full-tensor repeatability gates.

Review found two ways the comparison helper could accept unchecked values:
zipping a short error-scale array silently omitted elements, and an infinite
error bound accepted arbitrary finite errors. Follow-up
[#212](https://github.com/kvark/meganeura/pull/212), revision
`4c6925bd0d8f1ce07a0f9ae7809e84dffeed366e`, checks those contracts and gradient
inventory lengths. It also avoids reading a nonexistent gradient for packed
parameters that are observed outputs but do not reach the loss. This follow-up
changes reference checks and tests, not production kernels. It merged as
`c475fd0e05f72929eca3bf994584805596b80b8c` and was the previous Inferena pin;
the before/after runtime study below concerns #211, not this test-only follow-up.

Focused operator, block and regression checks passed on RTX 5070 and Arc B570.
The four follow-up guard/smoke cases passed on both. A 24-case NVIDIA
normalization sweep also passed, with a driver-lifetime diagnostic workaround:
repeated context creation in the test executable can fail to load
`libnvidia-tls.so.595.91.07` because its static TLS block cannot be allocated.
Preloading `libGLX_nvidia.so.0` kept the driver loaded and avoided that failure.
The benchmark runs did **not** use this preload. The existing Naga workgroup
layout validation warnings remain. These are targeted checks, not a full-suite
or all-platform correctness certificate.

Strict-f32 timing checks used the unchanged native Inferena runner with
Meganeura [before #211](https://github.com/kvark/meganeura/tree/dd9bf8acfd69918c4e4feb1db9c40a3b8f44aa36)
and [after #211](https://github.com/kvark/meganeura/tree/2fc49b4a6968f2a6f5ced491346110b6718c88dd).
Both binaries matched explicit clean builds of their declared dependency pins.
Measured construction stayed enabled with the usual 60-second session search,
at least five calls and two seconds of held-out warmup, and 20 timed samples.
GPUs and revisions ran serially, one process per condition.

| GPU / model | Inference before → after (ms) | Training before → after (ms) |
|---|---:|---:|
| RTX 5070 / Whisper-tiny | 5.804 → 6.193 | 28.599 → 28.522 |
| RTX 5070 / StableDiffusion | 1.618 → 1.734 | 7.066 → 7.136 |
| Arc B570 / Whisper-tiny | 13.879 → 13.902 | 95.179 → 95.322 |
| Arc B570 / StableDiffusion | 2.999 → 2.996 | 12.097 → 12.252 |

Intel inference is essentially unchanged in these samples; NVIDIA inference
is 6.7–7.2% slower. This warrants replication, not a confirmed regression claim:
there is only one fresh process per arm, and NVIDIA Whisper's same-shape latency
series changes from 6.271 to 6.173 ms. Training changes by -0.3% to +1.3%.
Preparation changes by at most 1.5%, but a brief CPU build overlapped the first
NVIDIA Whisper calibration, so these are not clean preparation-cost ablations.
None of these spot timings enter the paper's cohort tables.
The repeated NVIDIA checks below supersede the single-process performance
interpretation, without discarding these original observations.

Fresh strict PyTorch qualification passed all phases for Whisper on both GPUs
and StableDiffusion on NVIDIA. Intel StableDiffusion exhausted the 120-second
compile limit with `TORCHINDUCTOR_COMPILE_THREADS=1`, used here to limit host RAM.
That failed result was retained. A separate eager/math, uncaptured Intel process
passed full-tensor repeatability, and served only as a correctness reference.
All four native comparisons passed the existing cross-engine gates: sampled
output relative-L2 errors were 3.3e-6–1.5e-5 and parameter-gradient-norm vector
errors were 3.3e-7–1.6e-6. This does not claim full-vector agreement across engines,
nor a passing compiled Intel StableDiffusion condition. No AMD, Apple, Windows,
accelerated-mode or larger-model qualification was run here.

Inferena's nine Python tests, including actual CUDA replay and a deliberately
drifting CPU training model, and its nine Rust harness tests passed. The drift
test preserves qualified inference and rejects training; receipt tests prevent
eager timing substitution and reuse of incomplete replicate groups. Clippy
passed for the harness and the reference-test target. Raw diagnostics remain
outside Git under `/var/tmp/meganeura-pr211.viemZ0` on zork; only methods,
revisions, findings and summary values are retained here.

### NVIDIA repetition and kernel-cost check

Two additional fresh processes per revision and model used the exact same
binaries and settings as above. The second pair ran after/before and the third
before/after, with no concurrent builds or GPU work. Each process still includes
full search, qualification, five-call/two-second warmup, and 20 held-out samples.
The original process remains included; no run was discarded or retried.

| Model | Before: process inference medians (ms) | After: process inference medians (ms) | Change of medians |
|---|---|---|---:|
| Whisper-tiny | 5.804, 6.246, 6.238 | 6.193, 6.168, 6.206 | -0.7% |
| StableDiffusion | 1.618, 1.607, 1.618 | 1.734, 1.673, 1.731 | +7.0% |

Whisper's apparent regression does not reproduce. Its first process selected an
additional private matrix-kernel change and returned a faster first inference
series; the same session's later latency series was already 6.271 ms. The two
new before/after pairs select the same kernel changes and are within 1.3%.
Median training is 28.563 → 28.499 ms for Whisper and 7.105 → 7.136 ms for
StableDiffusion. All new post-fix runs retain full-tensor candidate qualification
and pass the existing sampled-output/gradient-norm comparison against the
preserved strict CUDA PyTorch reference.

StableDiffusion is slower at the default tuned policy in this small repeat.
All three pre-fix inference sessions select graph 1 with eight submission chunks.
Post-fix sessions select four, eight and four chunks respectively. In the first
post-fix search, eight chunks measured 1.68765 ms against four chunks at 1.74201 ms,
but was rejected by the unchanged 5% minimum-improvement floor plus measured
noise margin. The first pre-fix search instead rejected the intermediate
four-chunk candidate and later accepted eight chunks against a slower incumbent.
This illustrates sensitivity to earlier search decisions; it does not isolate
the cost of more accurate arithmetic. The floor and all collection settings
remain unchanged in the new pin.

A separate diagnostic disables measured construction in both arms and collects
eight per-pass timestamp samples from each ordinary plan, one process per arm.
It is not a replacement benchmark or a matched-plan measurement of the tuned
winner. Both revisions have 55 dispatches for Whisper and 229 for StableDiffusion.
Whisper's timestamped GPU total is 5.809 → 5.804 ms; its normalization family is
0.129 → 0.111 ms. StableDiffusion's GPU total is 2.272 → 2.308 ms (+1.6%), with
normalization at 0.233 → 0.252 ms, about 19 µs more. This supports a modest
normalization cost on Diffusion, alongside the separate tuning-selection effect.
Do not subtract these instrumented, untuned totals from tuned wall times to
attribute the remainder to barriers or CPU overhead.

The repeat and profiling artifacts remain outside Git at
`/var/tmp/meganeura-pr211-repeat.KAcYQO` on zork. No new cohort was collected.
The newly merged #212 changes only the reference validator/tests, not the
runtime paths measured in this #211 comparison.

## Simplified lowerings and training Winograd: September 24

Merged [#213](https://github.com/kvark/meganeura/pull/213), `a63bb726`, removes
duplicate hand-written elementwise/reduction paths, carries mixed-radix
convolution indices between tile loads, and extends Winograd to training
forward and input gradients. Weight gradients remain direct. Logical weights
stay authoritative; transformed weights are execution scratch. Attention now
handles more head widths and cached decode splits use the live KV range.
The stateless paper workload does not measure that last improvement.

The comparison uses the unchanged Inferena workloads with Meganeura
`c475fd0e05f72929eca3bf994584805596b80b8c` and
`a63bb726b48492283b71deb282083c6c07bf1577`. Both GPUs run serially with measured
construction enabled, a 60-second soft session budget, five calls and two
seconds of held-out warmup, and 20 samples. Convolution models receive two
fresh processes per arm, in before/after then after/before order; Whisper
receives one. These diagnostics do not enter the frozen cohort tables.
Large-record analysis overlapped the first Intel diffusion calibration; the
reverse-order pair ran without it. Both observations are retained below.

Values are medians of the process medians, in milliseconds:

| GPU / model | Inference before → after | Minimal forward before → after | Training before → after |
|---|---:|---:|---:|
| RTX 5070 / StableDiffusion | 1.761 → 1.734 | 1.716 → 1.693 | 7.123 → 6.535 |
| RTX 5070 / ResNet-50 | 5.229 → 5.226 | 2.705 → 2.697 | 30.598 → 27.465 |
| RTX 5070 / Whisper-tiny | 6.226 → 6.248 | 6.248 → 6.254 | 28.530 → 28.538 |
| Arc B570 / StableDiffusion | 2.992 → 2.974 | 3.145 → 2.985 | 12.597 → 11.438 |
| Arc B570 / ResNet-50 | 12.811 → 12.928 | 5.072 → 5.355 | 57.737 → 54.535 |
| Arc B570 / Whisper-tiny | 13.875 → 13.839 | 13.824 → 13.837 | 95.480 → 95.039 |

Convolution training improves 5.5–10.2%; ordinary inference stays within 1.6%.
Intel ResNet minimal forward is an exception: 4.925 → 5.188 ms in the first
pair and 5.218 → 5.521 ms in the second. Intel diffusion's earlier minimal
forward varies with the selected two- versus eight-chunk plan, 3.308 versus
2.981 ms. These are complete tuned-pipeline comparisons, not isolated kernel
speedups or statistical confidence intervals.

Preparation also falls for the convolution models. The clean Intel diffusion
pair takes 181.439 → 133.946 seconds. Its old inference searches reach the
60-second deadline with 62 trials; the new searches finish 64 trials in about
37 seconds. These include search, compilation, initialization and qualification,
not just shader compilation. Whisper preparation is essentially unchanged.

Training uses more execution-plan storage: ResNet rises from 485 to 606 MB on
both GPUs, and NVIDIA diffusion from 116 to 198 MB. Intel diffusion's selected
plans span 116–160 MB before and 215–239 MB after. These are allocated plan
bytes after aliasing, not comparative resident-VRAM peaks.

All before/after sampled outputs and per-parameter gradient-norm comparisons
agree to rounding scale. ResNet output samples are identical, with gradient-norm
relative-L2 changes below 1.7e-9; Whisper samples and norms are identical.
Both strict diffusion repetitions and Whisper pass comparison against the
preserved independent PyTorch references described above. Intel diffusion uses
the separate eager reference, not a substituted compiled timing. One new
accelerated diffusion process per GPU also passes against those f32 references;
output relative-L2 error is at most 4.9e-6 and gradient-norm vector error at most
8.1e-7. This is a numerical cross-check, not an accelerated paired timing study.
The construction gate still checks full tensors against each revision's
ordinary plan. No AMD, Apple, Windows or full five-model campaign was run here.

Sixteen focused checks per GPU pass on the merged revision: elementwise and
reduction families, normalization, convolution tiles and split-K gradients,
Winograd forward/backward, non-power-of-two attention, cached attention, and
checkpoint state. One additional attention-training test fails on both GPUs
with identical failures before and after #213. The reference checker resets
the incoming error scale after RoPE, even when a zero-angle rotation leaves a
cancelled gradient unchanged. Follow-up
[#214](https://github.com/kvark/meganeura/pull/214) propagates that scale through
the absolute rotation matrix. With this checker fix, attention training and
all eight rotary checks pass on both GPUs. No shader or tolerance changes.

The same follow-up uses egglog's let-binding printer for diagnostic expressions.
Expanding shared residual nodes into trees produced roughly 650 MiB
ResNet JSON records. This changes receipt formatting, not extracted graphs or
candidate ordering. Four search tests, the new CPU rotation check, clippy, and
Inferena's nine harness tests pass. These fixes merged as `2f018f9` and are in
the new Inferena pin, but not the before/after timings above. NVIDIA oracle
checks use the driver-lifetime
preload described above; benchmark processes do not.
The follow-up's CI also passes, including Linux host coverage and macOS tests.
A final native ResNet check on B570 with the merged pin and normal 60-second
budget completes all three phases, retaining 16 graph forms per phase and
identical recorded outputs, loss and gradient norms. Its JSON is 5,729,552 bytes,
versus 680,306,712 bytes for the earlier #213 record. Trial counts differ, so
this is a receipt-size check, not a preparation-speed ablation. An earlier
one-second smoke budget was too short to qualify training and produced no
result; that diagnostic does not change the collection budget.

### Forward tile-loader control

The Intel single-image regression also appears with measured construction off.
Both ordinary plans have 149 dispatches. Eight per-pass samples give GPU totals
of 5.453 → 5.604 ms; most of the increase is in convolutions. These instrumented
totals are not subtracted from the tuned wall times to infer barrier or CPU cost.

Branch `experiment/pr213-conv-loader-control` preserves two controls based on
`a63bb726`: `e0f9c4c` restores only the preceding forward-convolution loader;
`938acc0` instead resets its mixed-radix index at each K tile and carries it
only through that tile's loads. The latter is one moved declaration with the
K-stage offset added. It avoids carrying index state through the multiply loop;
there is no new model/device rule or tuning option.

The staging-local version reduces timestamped convolution time on both GPUs.
Intel single-image wall time is 5.736 → 5.369 ms in the timestamp-enabled
control. An initial NVIDIA pair has variable wall times, 3.457 ms original
versus 3.651 ms changed, despite lower convolution GPU time. A separate plain
repeat brackets the changed process with two original processes: batch-four
inference is 5.928 / 5.907 versus 5.561 ms, and single-image is 3.726 / 3.721
versus 3.644 ms. Diffusion is essentially unchanged on both GPUs. All outputs,
losses and gradient norms recorded by these controls are identical across the
shader change. These untuned controls are not replacements for the table above
or a clock-controlled register-pressure study.

Follow-up [#215](https://github.com/kvark/meganeura/pull/215) contains this
two-line shader diff. The existing tiled-convolution and Winograd-training
oracle checks pass on both GPUs; shader-generation validation and clippy pass.
This shader change is not in the `2f018f9` benchmark pin. Review it before
launching another full cohort; the original #213 latency regression remains
in that pin.

Raw records and commands remain outside Git under
`/var/tmp/meganeura-pr213.6TmK2x` on zork. No publication artifacts or collected
cohort files were replaced.

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
