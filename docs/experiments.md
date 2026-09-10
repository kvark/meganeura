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
