# Experiments: source refs and conclusions

Keep experiment implementations on an `experiment/*` branch or immutable tag.
Git records their base and changes; main needs the question, result, limitation
and reproduction command, not another source snapshot or artifact archive.
This follows Blade's rendering branches such as
[`experiment/merge-gbuffer-pass`](https://github.com/kvark/blade/tree/experiment/merge-gbuffer-pass)
and [`experiment/unfused-restir`](https://github.com/kvark/blade/tree/experiment/unfused-restir).

For a new experiment, commit the runner, configuration and dependency lock on
its branch before collecting data. Tag the measured source. Write outputs to
ignored `results/` or outside the checkout. Do not commit binaries, traces,
raw dumps, compressed dumps or evidence-replay tests to main. If exact observed
samples are needed for publication review, retain them outside Git as a
separate research artifact; rerunning a revision reproduces the procedure, not
the original timing noise. Record the environment and failure counts with the
conclusion. Adopt production changes independently of experiment scaffolding.

## Compiler stages — September 10

Source: `experiment/compiler-stages-2026-09-10` in
[Inferena](https://github.com/kvark/inferena/tree/experiment/compiler-stages-2026-09-10),
[Meganeura](https://github.com/kvark/meganeura/tree/experiment/compiler-stages-2026-09-10)
and [Blade](https://github.com/kvark/blade/tree/experiment/compiler-stages-2026-09-10).
The latter two add CPU spans only. Inferena pins both revisions and instruments
the installed Triton compiler. RTX 5070 / 595.71.05, i5-12400F, strict f32,
Python 3.13.13, PyTorch 2.13.0+cu130 at `cf30153`, Triton 3.7.1.

Three fresh processes per engine/model/cache state; private initially empty
compiler/driver disk caches, then a new process reusing each cache. GPU/OS
caches are not flushed. Engine order alternates. SmolLM2 runs 128-token prefill
and one stateless token; the other two models include F+loss+backward.
PyTorch uses default compilation with validated explicit CUDA Graphs.
Compilation is serialized (`TORCHINDUCTOR_COMPILE_THREADS=1`) to observe its
stages, so these are preparation diagnostics, not publication speed samples.

| Fresh-cache process | Naga parse, median µs/call | All Naga stages, ms | Vulkan pipeline creation, ms | Cold Triton compilations, ms |
|---|---:|---:|---:|---:|
| SmolLM2-135M | 141 | 7.24 | 386.8 | 2235.0 |
| ResNet-50 | 147 | 14.75 | 632.2 | 6725.5 |
| Whisper-tiny | 172 | 20.20 | 851.1 | 6598.8 |

Each cell is a median across three process records. Naga's total sums disjoint
parse, validation, specialization and SPIR-V emission spans; it excludes WGSL
generation and native driver work. Native pipeline calls number 21/66/56,
including reuse within a process. Cold Triton compilations number 37/181/96;
their per-call medians are 60.22/29.83/65.75 ms, including lowering to cubin.
The engines generate different kernel catalogues: these are **not paired
identical kernels**, nor a 10,000× compiler-speed result.

| Reported preparation, seconds | Meganeura fresh / reused | PyTorch fresh / reused |
|---|---:|---:|
| SmolLM2-135M | 1.175 / 0.779 | 23.990 / 6.066 |
| ResNet-50 | 1.385 / 0.713 | 19.521 / 2.225 |
| Whisper-tiny | 1.420 / 0.555 | 11.256 / 1.189 |

Preparation retains each runner's declared boundary: native graph/session
construction, versus Torch specialization/first execution; not just shader
compilation. Reused-cache native pipeline totals fall to 1.67/5.50/3.90 ms.
A separate 36-process tracing-disabled control changes preparation medians
by at most 2.1%; this sequential control is not a paired overhead confidence
interval. All 36 traced and 36 control processes completed. Cross-engine
forward/gradient-norm gates passed, CUDA replay validation stayed enabled,
and native output records repeated exactly. An earlier series overlapped an
automatic OS update and a user-space crash-handler loop; it is excluded.

Reproduce with `scripts/compile_study.py --output <new-outside-checkout-dir>`
in the pinned Inferena environment; repeat with `--untraced`. The later
experiment-branch `scripts/study_results.py` checks numerical pairs and exports
stage counts/times. Traces, caches and raw records stay outside Git. Cheap
Naga processing is measured and supports an on-device search budget; driver
compilation, qualification and whole-step amortization still must be charged.

The narrower `experiment/compiler-gemm-2026-09-10` Inferena tag compares f32
GEMMs with M=128, N=576, K=576/1536, output tiles 32/64, K tile 32 and 256
threads. Both retain runtime dimensions and IEEE f32 multiplication; generated
instructions/layouts need not be identical. Run `scripts/gemm_compile_study.py
--output <new-dir>`, then a separate cohort with `--warm-compiler`. The latter
compiles/loads the opposite tile first and reports that warmup separately.

Across three processes per cell, a new candidate after compiler warmup takes
**30.8–31.2 ms native versus 100.4–166.6 ms Triton**, including native pipeline
creation or cubin compilation/launcher loading. Native Naga stages total about
0.35–0.41 ms; the Vulkan driver accounts for about 30.4–30.7 ms. Reused private
disk caches give 0.43–0.52 versus 0.72–0.87 ms. First compiler use is separately
38.1–38.4 versus 375–442 ms; Triton's first-use initialization also appears on
cache hits. Allocation, execution and validation are outside these preparation
intervals. All 96 primary processes pass full-output f64 checks on ordinary and
tiny inputs, using the tuner's unchanged bound. This matched-domain microstudy
supports a roughly 3.2–5.4× cold-candidate preparation advantage here, not
10,000× or a kernel-performance claim.

## Full-model tuning — September 10

Sources: Inferena `experiment/tune-full-models-2026-09-10` and
`experiment/tune-smollm-confirm-2026-09-10`, pinning the compiler-stage revisions
above. Strict f32 on the same RTX 5070, real Inferena shapes/weights, grouped
unprofiled execution, five warmups and twenty samples per fresh process.
Run `scripts/tune_study.py --output <new-dir>`; confirmation adds
`--models SmolLM2-135M SmolLM2-360M --variants untuned default --replicates 6`.
All kernel qualification and whole-model numerical gates remain unchanged.

The six-pair AB/BA confirmation finds **1.092× SmolLM2-135M prefill speedup**:
12.699→11.628 ms medians, with median paired gain 1.068 ms versus twice its MAD
0.011 ms. Three exact classes switch 64→32 tiles, affecting 90 dispatches.
Median extra preparation is 85 ms: about **80 prefills to amortize**. Outputs
repeat exactly across all twelve runs. Stateless-token dispatches have no
eligible classes and do not benefit. SmolLM2-360M retains every tile: no guarded
gain, with 224 ms extra preparation. Its outputs also repeat exactly.

A separate three-process, three-model cohort widens the search to 128 classes
and 60 seconds, retaining the 64 MiB scratch cap. ResNet training changes
44.049→42.202 ms medians, below the 5% whole-step guard, while preparation grows
0.712→6.629 s. Two convolution classes reject output qualification. Whisper
shows negligible whole-step benefit. No tolerance is relaxed or default changed.
An interrupted 512 MiB exploratory cohort lacks its final manifest and is not
the confirmation evidence above. These results supersede an overly broad
negative reading of the earlier synthetic holdouts, not the frozen paper matrix.

## Profile-guided convolution specialization — September 10

Sources: `experiment/conv-specialization-2026-09-10` and
`experiment/conv-native-division-2026-09-10` in Inferena and Meganeura. Qualified
short Nsight Graphics traces concentrate ResNet training samples in scalar
convolution indexing/staging. The experiment binds immutable u32 parameters
as WGSL constants; a second arm uses native division with constant divisors.
No model/card rule, changed summation order, relaxed gate or default promotion.

Six fresh strict-f32 process pairs per arm give ResNet F+loss+backward medians
44.153→34.444→33.329 ms (untuned / constants / constants plus native division).
The roughly 1.28× / 1.325× gains clear the 5% plus paired-noise guard against
untuned. All recorded output fields repeat exactly; the existing full-f64
convolution regression oracles pass separately for both variants. Whisper's
roughly 1% gain does not clear the guard. A separate 135M control is unchanged.

For constants alone, six fresh-driver-cache pairs confirm the gain but add
3.067 s preparation: about 319 training steps to amortize, versus about twelve
with warm driver caches. The prototype compiles fallback and exact pipelines;
these are implementation ablations, not automated per-class selection. Run
`scripts/tune_study.py --models ResNet-50 Whisper-tiny --variants untuned
fixed-params fixed-native-div --replicates 6 --output <new-dir>` at the later
Inferena tag; the earlier tag supports `untuned fixed-params` and
`--fresh-driver-cache`. Native-tool methodology and attribution limits are in
Inferena's [analysis](https://github.com/kvark/inferena/blob/experiment/p3hpc-gap-2026-09-10/ANALYSIS.md).
The GPU interval also shrinks in a qualified Graphics capture, but PC sample
shares are not executed-instruction counts or wall-time barrier costs.

The later `experiment/conv-k-stage-2026-09-10` tags retain a negative result:
doubling K staging to 32 passes the full oracles but regresses ResNet training
33.340→36.834 ms across six paired processes, despite fewer barrier rounds.
No global K change is justified. A separate cold-cache confirmation of constant
native division amortizes its 2.757 s extra preparation after about 256 steps.

## Parameter placement and host RAM — September 10

Source-only `experiment/parameter-allocation-2026-09-10` tags in Inferena,
Meganeura and Blade separate allocator rounding from actual memory placement.
On this RTX 5070, the 1.7B strict-f32 plan requests about 9.4 GiB, but ordinary
`Shared` allocation puts about 4.9 GiB of its bindings on the host heap.
It is not an all-VRAM scaling point. The general device-parameter prototype
retains named/original/packed weights and uses bounded 16 MiB upload staging.

A four-arm diagnostic pilot and **one completed untraced pair** preserve the
full prefill hash. The latter changes prefill 951.234→54.887 ms and stateless
token 1941.822→16.282 ms with the same plan and kernels. Do not report this as
a replicated speedup: subsequent confirmation was stopped as a precaution
after NVIDIA mapping-allocation errors in the Shared control. The manifest
remains incomplete. Further experiments use resident-only controls; no GPU
reset, reboot or host OOM occurred in these bounded runs.

The 360M control separates another issue: free-list allocation reduces buddy
rounding but does not improve step time. Actual heap bindings, plan requests,
allocator blocks and process-driver accounting answer different questions.
Fresh qualified 135M/1.7B CUDA-Graph/Vulkan Systems pairs show that the remaining
resident gap is chiefly GPU execution, not host encoding. They do not isolate
barrier cost. Procedures, limitations and source pins are in Inferena's
[analysis](https://github.com/kvark/inferena/blob/experiment/p3hpc-gap-2026-09-10/ANALYSIS.md#placement-and-allocation-ablation--september-10).
The collection tag and paper tables remain unchanged.

Resident-only preparation follow-ups use the source-only Inferena tag
`experiment/parameter-preparation-2026-09-10`, pinning Meganeura `854b5b6`
and Blade `7b6d97a`. Six fresh-process replicates per model visit all six
orders of three arms: fresh upload allocations, reused bounded staging,
then reused staging plus a cache-blocked CPU transpose. No profiler runs
during this confirmation; all 54 recorded output sets repeat exactly.

| Complete process, median seconds | Fresh staging | Reused staging | Reuse + CPU transpose |
|---|---:|---:|---:|
| SmolLM2-135M | 10.537 | 2.939 | 2.625 |
| SmolLM2-360M | 14.164 | 5.718 | 4.521 |
| SmolLM2-1.7B | 43.163 | 36.179 | 28.167 |

These are startup/workflow savings, not shader-compilation or steady-state
speedups: the process includes checkpoint loading, both sessions, warmup,
measurement, validation and cleanup. No step gain clears the guard. Separate
Systems captures attribute the first saving to hundreds of avoided Vulkan
allocations; the next largest preparation span is CPU tensor conversion.
Run `scripts/tune_study.py --stream-weights --baseline device-params-buddy
--variants device-params-buddy device-params-reuse device-params-tiled
--models SmolLM2-135M SmolLM2-360M SmolLM2-1.7B --replicates 6 --output <new-dir>`
with the documented memory guard. No new main-branch fixture or binary is needed.

The `experiment/packing-layout-2026-09-10` and
`experiment/packing-warmup-2026-09-10` Inferena tags test the graph's packed
SwiGLU copy separately. Keeping the original unpacked weights saves about
3 GiB on 1.7B. Six AB/BA process pairs with 100 warmups and 100 samples confirm
16.316→14.558 ms stateless-token latency (1.121×), with identical full prefill
and token hashes. Prefill is unchanged; smaller models have no guarded step
gain. This is not yet an automatic representation search or default change.
An alternative scalar-matmul column layout is slower in the five-model pilot.

Longer warmup also reveals that 135M's short token window is not sustained
steady state: about 2.56 versus 3.58 ms on this configuration, with stable
prefill. Preserve this sensitivity, not just the fastest samples. Diagnostic
clock/host/GPU correlation must precede attribution. The collection tag is
unchanged; these new source-only controls are not replacement paper timings.

## September tuning foundation

These are development observations on RTX 5070 / driver 595.71.05, not updates
to the frozen paper matrix. Existing `evidence/*` tags retain measured code.
The old milestone and audit branch remain available; they are not to be merged
back to restore the discarded archives. Historical protocols and full run
details are available in the
[archived experiment tree](https://github.com/kvark/meganeura/tree/bc04aa31e33b62f79445ca6d9519209ddcf3e756/docs/experiments).

### tuning-2026-09-05

Source: `evidence/tuning-2026-09-05`. Runner: `tune_session`.
Five fresh processes established a synthetic f32 tile-search transfer pilot;
local isolated wins did not establish a general model-level improvement. This
device did not expose native-f32 cooperative tiles.

### holdouts-2026-09-06

Source: `evidence/holdouts-2026-09-06`. Runner: `tune_holdouts`.
Six inference/training holdouts showed that kernel winners can fail to improve
whole-step time. Keep search cost, amortization and whole-step acceptance
separate from isolated timings.

### crossover-2026-09-06

Source: `evidence/crossover-2026-09-06`. Runner: `tune_crossover`.
Controlled six-process AB/BA confirmation accepted a roughly 1.177× dense-chain
whole-step improvement. None of the wider holdouts passed the whole-step guard.
No automatic default promotion followed.

### readback-2026-09-06

Source: `evidence/readback-2026-09-06`. Runner: `tune_readback`.
Separating GPU completion, staging copy and CPU validation localized a search
cost, not a kernel cost. Read-optimized staging reduced ResNet search from
about 606 to 39 ms (copy about 582 to 2 ms), preserving validation. Blade 0.9.

### staging-reuse-2026-09-06

Sources: `evidence/allocation-profile-2026-09-06`,
`evidence/staging-reuse-2026-09-06`. Runner: `tune_staging_reuse`.
Call-local exact-size reuse reduced dense/MLP search from about 44/64 to 32/45
ms. Validation, state isolation and memory accounting stayed unchanged.

### training-profile-2026-09-06

Source: `evidence/training-profile-2026-09-06`. Runner: `profile_training`.
Convolution derivatives consumed about 61% of ResNet F+loss+backward; attention
backward about 36–41% of SmolLM2. These profiles exclude optimizer updates and
localize work; they do not establish an end-to-end improvement.

### conv-tiles-2026-09-06

Source: `evidence/conv-tiles-2026-09-06`. Runner: `tune_crossover --conv-derivatives`.
The first convolution cohort used an incorrectly initialized input and is not
performance evidence. The qualification and runner were repaired.

### conv-tiles-corrected-2026-09-06

Source: `evidence/conv-tiles-corrected-2026-09-06`.
Corrected `tune_crossover --conv-derivatives` runs found no whole-step guarded win for
ResNet training. Full nonzero-gradient qualification prevents zero-data success
from admitting a candidate. Tile search remains opt-in.

### conv-indexing-2026-09-06

Sources: `evidence/conv-indexing-baseline-2026-09-06`,
`evidence/conv-indexing-exact-2026-09-06`.
Runner: `profile_training --conv-indexing`.
Floating reciprocal indexing was wrong at width 41. Exact integer indexing
fixed the defect but cost about 23% on the measured ResNet F+loss+backward.
Correctness took priority; this was not a performance win.

### conv-divisor-2026-09-06

Sources: `evidence/conv-divisor-baseline-2026-09-06`,
`evidence/conv-divisor-reciprocal-2026-09-06`.
Runner: `profile_training --conv-divisor`.
A shared all-integer invariant-divisor implementation recovered about 2% while
preserving exact addressing. It did not erase the indexing repair's full cost.

### split-k-2026-09-06

The split-K prototype is included in `evidence/split-k-sequence-2026-09-06`.
Runner: `measure_split_k`.
Plans charge partial storage and expose legal split counts; this is an explicit
probe API, not a promoted production selection policy.

### split-k-sequence-2026-09-06

Source: `evidence/split-k-sequence-2026-09-06`. Runner: `measure_split_k`.
A synthetic long reduction improved from 3.084 to 0.445 ms at eight splits
(6.93× for the complete isolated sequence). Both profiled large controls failed
accuracy qualification. No training-speedup claim or automatic installation.

### compensated-dw-2026-09-06

Source: `evidence/compensated-dw-accuracy-2026-09-06`.
Reproduce at that tag with `cargo test --release --test conv_derivatives
report_bounded_weight_accumulation_qualification -- --ignored --nocapture
--test-threads=1`. The shared compensated candidate passed 230/240 accuracy rows;
10 tiny structured-cancellation rows failed. Arithmetic was reverted, split-K
promotion deferred, and the bounded milestone closed. There were no performance
measurements after this rejection.
