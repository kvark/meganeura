# Converging measured search

Prototype design on `claude/search-race`, branched from `main` at `12df95c`.
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
| Submission round trip | Empty submit and wait | CPU-timed sample sizes |
| Pass and dispatch overhead | Passes of trivial dispatches, GPU timestamps | Dispatches per timed pass |
| Memory bandwidth | Streaming kernel, grown to a target duration | Cost model |
| Matrix throughput | Meganeura's own matmul kernels, grown to a target duration | Cost model |
| Parallel width | Fixed work at doubling workgroup counts, until time grows | Wave quantization |
| Timer noise | Repeated identical measurements | Rounds per decision |

`tune::MachineProfile` is plain serializable data. Calibration is bounded by a
fraction of the tuning budget and runs once per context in a process. A caller
can pass a profile instead, which skips calibration. Nothing is persisted, so
pre-baked profiles remain a later, separate decision.

### Cost model

A dispatch's predicted time is a roofline using the profile's numbers: the
larger of `flops / family throughput` and `bytes / bandwidth`, divided by the
wave utilization of its workgroup count, plus dispatch overhead. Dispatches
without a matrix shape use bytes over bandwidth. After each class race, the
measured incumbent time corrects that family's predictions for the classes not
yet visited. Predictions only order and bound work; they never select a winner.

### Class races

A class's incumbent and all of its legal challengers are measured together.

1. Compile every candidate pipeline in one parallel batch, using the session
   builder's `compile_all` pool.
2. Allocate one scratch for the class, sized for its largest candidate, and
   upload ordinary-magnitude inputs once.
3. Run rounds. With GPU timestamps, a round is one submission holding one timed
   pass per candidate, in rotated order. A pass repeats its candidate's
   dispatch until the measured pass overhead is under 2% of the pass, and the
   pass time is divided by the repetitions. Split sequences keep one pass per
   dispatch, so dependent dispatches stay ordered. Without timestamps, each
   candidate is a CPU-timed submission, as today.
4. After a few rounds, drop challengers that cannot meet the acceptance bar:
   their fastest sample is already slower than the incumbent's median less
   `min_improvement`. Keep only the best few (successive halving). Finalists run
   until they have `sample_pairs` rounds paired with the incumbent.
5. Qualify the fastest finalist that passes the existing acceptance rule with
   today's checks. If it fails, try the next. If no finalist beats the
   incumbent, nothing is qualified.

Every candidate shares rounds, clocks and inputs, so the result no longer
depends on challenger order. Losers are never qualified. With timestamps, the
roughly 14 timed submissions per challenger become one pass per candidate per
round.

Repeated dispatches inside one pass can overlap, which flatters kernels that
leave the GPU partly idle. The 2% rule therefore keeps one dispatch per pass
unless the kernel is short compared with pass overhead, and `build_measured`
still accepts kernel choices only through whole-step timing.

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

Memoized classes install their winners without taking a `max_classes` slot. A completed race records its class as
decided, including challengers that failed to compile or qualify, so later
searches do not repeat them.

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

## Prototype status

`TuneOptions::strategy` selects `Race`, the default, or `Sequential`, the
previous per-challenger duel, so the two can be compared on the same sessions.
`TuneOptions::calibration` selects `Measure`, the default, `Off`, or a supplied
`Profile`. Suites that use the tuner to qualify every candidate request the
sequential, uncalibrated search explicitly. This section records what has
landed on the branch.

- [x] Machine profile and calibration (`runtime/calibration.rs`), cached per
  context in a process.
- [x] Class races with parallel compilation and finalist-only qualification
  (`runtime/tuning/race.rs`), timed by GPU timestamps when the session's
  context records them.
- [x] Predicted class order with per-family correction, the 0.1% cutoff, and
  memo changes: decided classes take no slot, and a completed race decides
  every challenger. Kernel memo revision 5 and the memo scope record the strategy.
- [x] `examples/search_race.rs` compares strategies on fresh sessions.
- [ ] Shared calibration in `build_measured`: nested tuning already reuses
  the per-context profile; the search does not yet measure it up front.
- [ ] Whole-program changes from the next phase.

First lavapipe comparison (CPU emulation, so only the search behaviour
transfers), 30-second budget, the example's SmolLM2-shaped block:

| Strategy | Search | Classes | Comparisons | Qualified | Winners |
|---|---:|---:|---:|---:|---:|
| Sequential | 30.1 s, budget hit | 2 of 4 | 12 | 12 | 0 |
| Race, calibrated | 15.2 s | 4 of 4 | 42 | 1 | 1 |

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
