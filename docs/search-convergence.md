# Converging measured search

Design on `claude/search-race`, branched from `main` at `12df95c`.
New tuning options default to racing and measured calibration; explicit
sequential/uncalibrated options retain the comparison path. This distinguishes
implemented heuristics from guarantees; the cost model is not an acceptance oracle.
It changes how candidates are measured and ordered, not what is legal or how
results are qualified. Kernel families that arrive later, such as #261's
rectangular workgroup tiles, become additional candidates in the same
machinery.

## Problem

Measured search has two nested loops. `build_measured` compares complete
programs: each trial builds a session, initializes it, qualifies the whole
model, runs private kernel probes, warms up both programs and times paired
whole steps. Inside each trial, `Session::tune_with` compares implementations
for each class of identical dispatches.

Both loops pay a large fixed cost for every candidate they consider, take
candidates in a fixed coverage order, and accept changes one at a time. Adding
a kernel family therefore adds search cost roughly in proportion to its size,
without making the good candidates arrive sooner.

- The P3HPC cohort's per-trial receipts
  ([results](../paper/p3hpc/RESULTS.md#where-the-search-time-goes)) split
  426.7 search-minutes into kernel probes 29.1%, session construction 19.3%,
  whole-program comparison 27.9%, qualification 14.1% and initialization 9.6%.
  The 250 ms warmup floor alone took 75.6 minutes (17.7%). Only 561 of 18,152
  comparisons (3.1%) replaced the incumbent.
- Budgets cover little. Sixty-second SmolVLA training searches visited 6
  programs on Arc B570 and 21–24 on RTX 5070, out of 64
  ([October](gpu-gap-2026-10.md#backward-layout-search--october-6),
  [compiler search](compiler-search.md#pr-206-model-check)). Production
  convolution tuning visited 7 of 68 eligible classes on NVIDIA and 4 of 68 on
  Intel ([experiments](experiments.md#production-convolution-search--september-13)).
- Results depend on order and warmup. Putting backward layouts first regressed
  held-out training from 19.5 to 23.2 ms
  ([October](gpu-gap-2026-10.md#backward-layout-search--october-6)). Warming
  up for 64 pairs instead of 2 changed the selected SmolVLA inference program,
  3.64 against 2.60 ms ([compiler search](compiler-search.md#cost-and-enumeration)).

## Where kernel probing spends time

`tune_with` ranks classes by `members·m·n·k`, a structural prior, and visits at
most `max_classes` (8) of them. Memoized classes still use those slots, so a
class ranked ninth is never tuned in any trial. Each challenger is a separate
duel with the current incumbent:

1. Compile the pair's pipelines on the calling thread, one at a time.
2. Allocate private scratch and a new command encoder.
3. Qualify both variants on two input patterns. Each pattern and variant
   needs a NaN fill, a dispatch and a readback, then 32 f64 reference dots and
   a full-output comparison.
4. Warm up each variant once, then time 6 ABBA pairs of
   `encode+submit+wait` on the CPU clock.
5. Accept if the paired gain exceeds 5% plus twice the median absolute
   deviation.

That is 22–29 blocking GPU round trips per challenger. Every challenger is
qualified, including the large majority that lose. Comparisons are sequential
and greedy, so the selected tile depends on challenger order. A challenger that
fails to compile or qualify is never recorded in the memo: later searches probe
it again, and `kernel_probes_pending` stays true, which costs `build_measured`
an extra whole-program qualification per trial.

#261 multiplies this cost. A typed or automatic-F16 class gains up to 16
workgroup tiles on top of its partition counts, 16–22 challengers in total, so
one class can use the whole default 2-second budget.

## Design

Measure many candidates at once and cheaply. Qualify only what could be
installed. Spend the budget where predicted and measured time say it matters.
Legality checks, numerical qualification before installation, private scratch
and whole-step acceptance in `build_measured` do not change.

### Machine profile

The first part of the budget measures general properties of the machine. It
does not look anything up by device name.

| Measurement | Method | Used for |
|---|---|---|
| Submission round trip | Empty submit and wait | Host calibration subtraction and target duration |
| Pass and dispatch overhead | Passes of trivial dispatches, GPU timestamps | Dispatches per timed pass |
| Memory bandwidth | Streaming kernel, grown to a target duration | Cost model |
| Matrix throughput | Meganeura's own matmul kernels, grown to a target duration | Cost model |
| Parallel width | Fixed work at doubling workgroup counts, until time grows | Wave quantization |
| Timer noise | Repeated identical measurements | Diagnostic; adaptive rounds remain future work |

`tune::MachineProfile` is plain serializable data. Calibration is bounded by a
fraction of the tuning budget (one tenth, at most 250 ms) and runs once per
context/timing mode in a process, only when there is new work to probe. It is
charged to the tuning call, never to a `kernel_probes_pending` query, and
respects the scratch limit and device headroom. Sub-budgets leave time for
bandwidth and matmul after overhead/width discovery. Deadlines are soft: an
in-flight compilation or measurement cannot be preempted, but an expired
stage does not start more probes.

Partial profiles are cached too, so a short allowance does not repeatedly
spend every trial's budget without getting to the kernel search. Missing
measurements leave structural ordering in place. The cache's global lock is
not held while measuring. A caller can pass a validated, finite profile
instead; nothing is persisted. `TuneReport::calibration_time` reports cost in
this call, while `MachineProfile::elapsed` describes the original measurement.

### Cost model

A dispatch's predicted time is a roofline using the profile's numbers: the
larger of arithmetic time (scaled by wave utilization) and `bytes / bandwidth`,
plus dispatch overhead. The initial throughput is the measured ordinary F32
matmul rate, **not** a calibrated rate for every dtype or kernel family. Dispatches
without a matrix shape use bytes over bandwidth. After each class race, the
measured incumbent time corrects that family's predictions for the classes not
yet visited. Predictions only order and bound work; they never select a winner.
Parallel width is an approximate occupancy prior from one particular kernel,
not a device limit. Binding sizes approximate traffic, not cache behavior;
the bounded streaming working set can measure cache-resident bandwidth rather
than sustained off-chip bandwidth.

### Class races

A class's incumbent and all of its legal challengers are measured together.

1. Compile every candidate pipeline in one parallel batch, using the session
   builder's `compile_all` pool. Workers stop starting new jobs at the search
   deadline; skipped jobs are retryable, not shader rejections. Up to eight
   already-running driver calls may overrun it.
2. Allocate one scratch for the class, sized for its largest candidate, and
   upload ordinary-magnitude inputs once.
3. Run rounds. With GPU timestamps, a round is one submission holding one timed
   pass per candidate. Every rotated round is followed by its reverse, with
   the same candidates, to balance position-dependent drift. Every dispatch
   inside a pass has a barrier before the next: both repeated output writes
   and split-product/SumRows dependencies require ordering. Repetition counts
   target 2% pass overhead when calibrated, capped by `dispatches_per_sample`;
   the target is not a guarantee. Split sequences normally use one repetition.
   With no warmup or `target_sample_time: None`, the fixed count is retained. Without
   timestamps, each candidate is a CPU-timed submission, as today, but owns a
   reusable command encoder. Blade evicts a layout's descriptor cache after
   an unused recording; cycling many candidates through one host encoder
   otherwise measures allocation/destruction on every round. Timestamp rounds
   already use all live layouts in one recording. Both paths
   honor `warmup_runs` and the search deadline.
4. After four rounds (two forward/reverse pairs), drop unpromising challengers:
   their fastest observed sample is slower than the incumbent's median less
   `min_improvement`. Keep the best four by median. Eliminate only after a
   complete forward/reverse pair. Finalists run until they have `sample_pairs`
   rounds paired with the incumbent.
5. Among finalists that beat the incumbent, prefer the earliest candidate that
   the fastest cannot beat by the same `min_improvement` plus paired-noise
   guard. This anchors ties to the fastest, not a chain of marginal wins, and
   does not chase insignificant differences in isolated-kernel timing. Qualify
   that choice with today's checks. If it fails, apply the preference rule to
   the remaining finalists. If no finalist beats the incumbent, nothing is
   qualified. Timing ties retain the declared candidate priority, so this is
   deliberately not an order-independence guarantee.

Shared rounds remove greedy incumbent replacement and reduce timing overhead;
they do not prove order independence or convergence. Early elimination and the
four-finalist cap are heuristics: warmup, noise, or incorrect fast candidates
can hide a useful alternative. They need held-out comparison against sequential
search across budgets and devices. Losers are not qualified, but no winner is
installed before qualification finishes within the deadline. With timestamps,
the roughly 14 timed submissions per challenger become shared submissions.
`build_measured` still accepts kernel choices only through whole-step timing;
direct `tune_with` provides isolated-kernel evidence, not that guarantee.

### Class order and memo

Classes are visited in order of predicted time: members times the predicted
time per dispatch. When the budget is short, the deadline therefore cuts the
smallest classes. Otherwise, visiting stops only once the classes left are
predicted to total less than 0.1% of the plan. That bar is far below
`min_improvement` for three reasons:

- Kernel winners are accepted together, so wins in small classes add up.
- A race's cost scales with its class's kernel time, so small classes are cheap.
- Predictions can be several times off. On lavapipe, a GEMV class predicted at
  0.4% of the plan took about 1.1% and held the only win. A bar of
  `min_improvement`, and then of 1%, skipped it.

Memoized classes restore their winners before new probing or the share cutoff,
without taking a `max_classes` slot. A completed race records its class as
decided, including candidate shader rejections and failed candidate outputs.
Scratch/device-memory limits, invalid timing, an invalid incumbent control,
and budget interruptions are not permanent decisions. They must be retryable
with better conditions or larger budgets. Memo scope distinguishes strategy
and host/GPU timing; revision 6 follows #261's independent revision 5.

The 0.1% cutoff uses **uncorrected** predictions on both sides; measured family
corrections reorder work only. It is a time-saving heuristic, not a
bound on achievable speedup. Validate it against uncalibrated runs rather than
using increased class coverage as a substitute for selected-program quality.

### Whole-program search (next phase)

The same ideas apply to `build_measured`:

- Calibrate once per search and share the profile with every trial's tuning.
- Skip plans already measured before building a session for them, and run
  submission-chunk probes on the existing session instead of rebuilding it.
- Keep one warmed incumbent across trials instead of warming it up again in
  every comparison.
- Order the frontier by predicted gain: the incumbent's per-dispatch profile
  plus deltas predicted by the cost model and earlier races, as
  [compiler search](compiler-search.md#cost-and-enumeration) proposes.
- Accept jointly. Collect the changes that win their own measurements, confirm
  them together in one whole-step comparison, and bisect if the combination
  fails, so wins below the 5% bar can add up.

## Implementation status

`TuneOptions::strategy` selects default `Race` or explicit `Sequential`, the
previous per-challenger duel, so the two can be compared on fresh sessions.
`TuneOptions::calibration` selects default `Measure`, `Off`, or a supplied
`Profile`. Missing historical settings still deserialize to Sequential/Off.
Suites that use the tuner to qualify every candidate request the
sequential, uncalibrated search explicitly. This section records what has
landed on the branch.

- [x] Machine profile and calibration (`runtime/calibration.rs`), cached per
  context in a process.
- [x] Class races with parallel compilation and finalist-only qualification
  (`runtime/tuning/race.rs`), timed by GPU timestamps when the session's
  context records them.
- [x] Predicted class order with per-family correction, the 0.1% cutoff, and
  memo changes: decided classes take no slot, and a completed race decides
  every challenger with a reusable decision. Kernel memo revision 6 and the
  memo scope record the strategy and timing mode.
- [x] `examples/search_race.rs` compares strategies on fresh sessions.
- [x] Shared calibration in `build_measured`: measured once with the search's
  own budget, before the first trial that probes kernels, reported in
  `BuildSearchReport::machine_profile` and supplied to every later trial.
- [ ] Whole-program changes from the next phase.

Historical pre-review lavapipe comparison at `0d02d3e` (CPU emulation, so only
the search behaviour transfers), 30-second budget, the example's
SmolLM2-shaped block. This predates synchronized repetitions and balanced
rounds and is not validation of the revised policy:

| Strategy | Search | Classes | Comparisons | Qualified | Winners |
|---|---:|---:|---:|---:|---:|
| Sequential | 30.1 s, budget hit | 2 of 4 | 12 | 12 | 0 |
| Race, calibrated | 15.2 s | 4 of 4 | 42 | 1 | 1 |

### Review smoke check — RX 7900 XT, October 16

The revised code was checked with the process-local Mesa 26.2.4 build used
for BF16 experiments, a warm driver cache, and the example's 2-second budget.
Each policy ran twice in reversed policy order on fresh sessions/contexts;
the table gives the observed ranges, not confidence intervals. Other users'
background GPU clients were not stopped. No other review test/build was run
concurrently with these final measurements. This is a small block, not an
Inferena cohort or a controlled whole-model performance claim.

| Context timing | Policy | Search (s) | Selected step (ms) |
|---|---|---:|---:|
| Host | Sequential | 0.279–0.290 | 0.1442–0.1444 |
| Host | Race + profile | 0.216–0.222 | 0.1492–0.1525 |
| Host | Race, no profile | 0.146–0.147 | 0.1490–0.1491 |
| GPU timestamps | Sequential (host-timed duels) | 0.270–0.271 | 0.1412–0.1413 |
| GPU timestamps | Race + profile | 0.066–0.069 | 0.1469–0.1474 |
| GPU timestamps | Race, no profile | 0.120–0.122 | 0.1464–0.1502 |

All runs covered 3/3 eligible classes and 31 challengers. Sequential qualified
31 candidates, host races 2, and timestamp races 3. Profiling cost 67–70 ms in
host mode and 30–33 ms with timestamps. It reduced timestamp repetition counts
enough to repay calibration here, but added cost in host mode. More importantly,
the raced selections had slightly **worse** post-search step times than sequential
search. Lower search latency and fewer qualifications do not establish better
convergence. These observations motivated the investigation below; they are
not the results of the final default policy.

Reproduce the policy orders with `cargo run --release --all-features --example
search_race -- 2 sequential race race-uncalibrated race-uncalibrated race
sequential`; with `MEGANEURA_GPU_TIMING=1`, use `2 race-uncalibrated race
sequential sequential race race-uncalibrated`. Use the same ICD/device for both.

### AMD regression diagnosis and correction

Two distinct problems explained the regression, without changing warmup or
numerical qualification:

1. **Host descriptor-cache churn hid a GEMV win.** The same 32-lane GEMV with
   four column groups measured about 8–10 us in sequential duels but 12–18 us
   in a shared-encoder host race. Instrumenting encoding and submission/wait
   exposed the extra cost; Blade's `reset_descriptor_pool` discards layouts
   unused in the previous recording. Separate reusable host encoders reduced
   steady-state encoding from roughly 45–55 us to 2–3 us per 16-dispatch batch,
   and the race found the GEMV improvement. Buffers remain shared, and GPU
   timestamp races still use one encoder/submission per round.
2. **Finalists were ranked below the configured resolution.** After fixing
   host timing, both timing paths preferred eight GEMV column groups over four
   for only a roughly 1–2% isolated gain. The shared-context experiment swapped
   just that choice between retained sessions: the approximately 5 us
   whole-program penalty followed the eight-group choice. With serial dispatch
   the penalty disappeared (174.8 versus 174.9 us after the swap); per-dispatch
   profiling also showed near-equal GEMV times. This implicates interaction with
   overlapping dispatches rather than a slower isolated kernel. The original
   5% plus noise guard applied only against the original incumbent, not between
   finalists. Applying it there as well retains the four-group candidate here.

Release recheck on the same RX 7900 XT/private Mesa, 2-second budgets, warm
driver cache, two visits per policy in reversed order, without concurrent
review builds/tests (other users' clients remain untouched):

| Context timing | Policy | Search (s) | Post-search step (ms) |
|---|---|---:|---:|
| Host | Sequential | 0.275–0.280 | 0.1377–0.1403 |
| Host | Race + profile | 0.201 | 0.1370–0.1389 |
| Host | Race, no profile | 0.131–0.134 | 0.1386–0.1393 |
| GPU timestamps | Sequential (host-timed duels) | 0.270–0.273 | 0.1428 |
| GPU timestamps | Race + profile | 0.067 | 0.1426–0.1434 |
| GPU timestamps | Race, no profile | 0.123–0.127 | 0.1443–0.1477 |

The default race qualifies three winners rather than all 31 challengers and
selects the same dense tiles and four-group GEMV as sequential in these runs.
The example now also retains all programs and times interleaved held-out steps
in balanced order, independently of which search last warmed the device.
Those medians were 144.6–145.9 us sequential versus 146.4–146.9 us calibrated
race in host mode, and 163.1–165.3 versus 162.6–162.8 us with timestamps. A
shared-context host pair had identical selections (zero choices to swap) and
139.6 versus 139.5 us, then 139.8 versus 139.8 us after the swap. This removes
the reproducible regression above; small differences between fresh contexts
are not evidence of better kernels. It remains a small-block test, not a claim
of optimality or a whole-model cohort result.

Set `MEGANEURA_SEARCH_VERBOSE=1` to print every candidate's raw samples and
decision. `MEGANEURA_SEARCH_SHARED_CONTEXT=1` shares the context/profile across
fresh sessions; with two strategies it also swaps their selected kernels and
repeats the held-out comparison. `MEGANEURA_SERIAL_DISPATCH=1` separates the
effect of overlapping dispatches. Never infer whole-program improvements from
isolated scratch wins alone: `build_measured` retains its whole-step guard.

## Follow-ups outside the main plan

- **Blade pipeline cache.** Blade creates pipelines with a null
  `VkPipelineCache`. Parallel bulk compilation covers the in-process cost, and
  driver disk caches already bring a repeated pipeline down to about 0.5 ms
  ([experiments](experiments.md#compiler-stages--september-10)). A persisted
  cache would mainly help drivers without one.
- **Per-encoder timestamps in Blade.** Timing is a context setting: every
  encoder on a timing context records pass timestamps, and timing-enabled
  contexts do not reproduce production behaviour
  ([profiling](performance-profiling.md#state-checked-training-localization)).
  Letting a private search encoder request timestamps would give races GPU
  timing on production contexts. Until then, races use timestamps when the
  context has them and the CPU clock otherwise.
- **Pre-baked machine profiles.** A caller can already pass a profile.
  Shipping or caching profiles is a separate decision.
- **Overlapped compilation.** Compile the next class's candidates while the
  current class races.

## Validation

Lavapipe supports Blade's calibrated timestamps, so CI can exercise both
timing paths. On hardware, run `tune_with` and `build_measured` with both
strategies on the same sessions, and compare classes visited, decisions,
elapsed time and the selected program's held-out time.

Review regression coverage includes transient failures remaining retryable,
cached tails restored with no probe slots, non-mutating pending-work checks,
partial-profile reuse, expired compilation budgets, balanced round order,
invalid supplied profiles, host/timestamped F16 split sequences, finalist
margin/noise guards, non-chaining ties, and preservation
of live training/optimizer state. Broader performance validation still needs repeated
paired whole-model runs, cold/warm compilation, host/timestamp modes, short
and long budgets, and shared memo reuse. One small block or CPU-emulated run
does not establish that the search has converged.

Validation of the reviewed tree: 628 library tests and 174 regressions passed
on lavapipe (8 and 21 ignored); unsupported native-matrix cases skip there.
On RX 7900 XT, six race/state tests (including native F16 split-K) and the
timestamped calibration test passed with Vulkan synchronization validation
enabled, without synchronization hazards. Formatting, all-target/all-feature
Clippy with warnings denied, the featureless check, and documentation build
also passed. These are correctness checks, separate from the smoke timings.
