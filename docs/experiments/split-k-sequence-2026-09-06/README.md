# Isolated split-K sequence measurements

## Predeclared protocol

Extend the [plan prototype](../split-k-2026-09-06/README.md) through the existing
scratch runner. This experiment compares the compiler's current scalar dW tile
against the **same tile** with 2, 3, 4 and 8 reduction partitions plus SumRows.
It does not install a choice in a live session or measure a whole-step gain.
No workload rule, precision change or new decision threshold follows from it.

`Session::measure_conv_weight_splits` measures one explicit exact class with at
most four legal counts, reusing TuneOptions, phase timers, scratch staging,
paired samples and the 5%+2MAD decision guard. Every count compares against the
original unsplit control, not the previous winner. Report `candidate_split_k`
alongside the existing tile fields: identical tiles do not mean identical plans.
Historical reports omit this field. The ordinary live tile tuner and fixed-layout
swap still never install split-K.

Scratch now names the final output slot explicitly. Append one f32 partial buffer
after it, placed on device like a temporary. Charge **both inputs + final output
+ partials + one staging buffer as large as the largest binding** before pipeline
or buffer allocation. Baseline and challenger reuse those same buffers sequentially;
there is no second hidden output, upload or readback allocation. The staging slot
may be reused only at exactly the same size within the call and is released at
return. These are GPU buffer requests, not CPU oracle storage, pipelines, driver
heap allocation or whole-process peak VRAM. Live-session residency is reported
separately and remains unchanged.

Execute the existing ordinary/tiny synthetic patterns, NaN sentinels, full finite
and elementwise cross-variant scans and sampled f64 dots. For split comparisons,
add full f64 checks of **both final outputs and every partial**, with the original
`scale*1e-5 + abs(reference)*2e-4` elementwise bound and `2e-4` relative-L2 gate,
nonzero reference norm and no norm floor. A failure, including a bad baseline,
rejects the comparison before warmup/sampling. Record the failing variant/pattern
or partial/element where available; never lower accuracy to obtain a timing.
The CPU partial reference covers contiguous reduction ranges and is checked
against independent forward scatter. The existing full convolution, optimizer,
staging, skip and state-isolation tests must pass before retained measurements.

The shared scratch runner repeats the **entire sequence**, creating a compute
pass for each dispatch so the partial write, reduction read, and next repetition
are barrier-delimited. Timers include host encoding, submission and wait for all
passes. Normalize by sequence repetitions, not number of constituent dispatches.
Ordinary one-dispatch tuning uses the same runner. Readbacks/validation remain
outside samples, inside reported search cost. Restore ordinary inputs and use
one warmup per arm, six alternating A/B pairs, 16 sequences per sample. Keep the
existing 64 MiB scratch bound; allow a 120-second soft deadline per class to
expose full-oracle cost. An incomplete pair cannot win.

Freeze the qualified source and runner as
`evidence/split-k-sequence-2026-09-06`, build in release with the
[archived Blade 0.9 lock](../readback-2026-09-06/Cargo.lock.gz), SHA-256
`72cecddd0f090e82d5db46bc6f7f80429a058b2afa0eaec84a9b27e7340a4a80`.
Run exactly four fresh `measure_split_k` processes, seeds 1..4, retaining all
attempts, including accuracy failures, timeouts and noisy timings. Do not build,
run other GPU tests or perform heavy CPU analysis concurrently with this cohort.

Use these fixed tuples `(batch,Ci,H,W,Co,Kh,Kw,stride,padH,padW)`:

| Case | Tuple | Motivation |
|---|---|---|
| spatial-7x7 | `(1,3,224,224,64,7,7,2,3,3)` | Profiled small-output long dW reduction. |
| pointwise | `(1,256,56,56,64,1,1,1,0,0)` | Repeated long 1x1 contraction shape. |
| rectangular-tail | `(3,5,7,9,7,2,3,1,1,0)` | Small padded/batched uneven-tile control. |
| long-tail | `(2,3,1,32771,5,1,1,1,0,0)` | Long reduction and partial numerical sensitivity. |

Both case order and count order rotate by `seed-1`. This balances positions
across four processes; it is not a crossover of session allocation roles. The
synthetic scratch qualification/timing inputs are identical across processes.
Live preflight inputs vary by seed, execute one unoptimized scalar F+L+B step
without optimizer, and must be finite/nonzero. Compare all input, weight,
gradient and loss/partial tensors bitwise before/after measuring; also require
identical live plan, pipeline keys, tensor residency and zero Adam state/count.
These checks establish isolation, not mathematical correctness of the live step.

Retain source/executable/lock hashes, device/driver, process windows, host samples,
250 ms optional GPU telemetry, every report/sample/decision and complete state
comparison summaries. Numerical rejection is a completed experimental result, not
an excuse to omit the case or restart it. Raw vectors are not retained; replay
can check recorded arithmetic and scope but cannot rerun their absent oracles.

Report all per-process sequence medians and guards, rejection counts, scratch
requests and qualification/search costs. Do not pool shapes, treat 2MAD as a
confidence interval, present producer-only time as a sequence gain, or call an
isolated result a PyTorch/whole-step speedup. Whole-step experiments across rebuilt
plans need their own protocol only if these results warrant proceeding.

## Retained results

All four processes completed September 6, 22:09:46.219–22:10:10.134 UTC on
RTX 5070 / driver 595.71.05, i5-12400F, Rust 1.98.0. Measured source is
`8d77f90d4565aa150b8b5f70a50a8d72331cabb4` at the tag above; executable SHA-256 is
`34f94f2fcdd3ba90ab7dc5f2de81416f5e6720520f9d45ee41d676133119d301`.
Every predeclared attempt is retained: [1](run-01.json.gz), [2](run-02.json.gz),
[3](run-03.json.gz), [4](run-04.json.gz), with [digests](SHA256SUMS) and a
[per-process summary](summary.json). There were no retries, discarded runs,
concurrent builds, other GPU tests or heavy host analysis during the cohort.

All 16 live plan/state comparisons remain bit-identical, including all six
retained tensor roles, allocation requests and zero optimizer state. Of 64
isolated comparisons, **32 qualify and 32 reject before timing**. The qualified
comparisons supply 192 complete pairs; 27 pass the unchanged gain guard and
five keep the baseline. These decisions are per comparison, not a family-wide
selection or automatic installation.

Each time below is the median of four process medians, in milliseconds. Ratios
are medians of the four process ratios, not ratios of the displayed medians.

| Shape | Splits | Unsplit | Complete sequence | Ratio | Gain guards passed |
|---|---:|---:|---:|---:|---:|
| rectangular-tail | 2 | 0.015384 | 0.016306 | 0.9411× | 0/4 |
| rectangular-tail | 3 | 0.015373 | 0.014268 | 1.0765× | 4/4 |
| rectangular-tail | 4 | 0.015370 | 0.014230 | 1.0801× | 3/4 |
| rectangular-tail | 8 | 0.015415 | 0.012328 | 1.2492× | 4/4 |
| long-tail | 2 | 3.084264 | 1.616171 | 1.9087× | 4/4 |
| long-tail | 3 | 3.083817 | 1.091155 | 2.8263× | 4/4 |
| long-tail | 4 | 3.083641 | 0.829207 | 3.7192× | 4/4 |
| long-tail | 8 | 3.083991 | 0.444891 | 6.9317× | 4/4 |

This demonstrates an isolated scheduling opportunity, including the final
reduction cost. It does not establish ResNet acceleration. Two-way splitting
slows the small case's medians, and four-way splitting misses the guard in
process 3. Small-case absolute timings also vary by order: process 4's first,
eight-way comparison is 0.01738→0.01421 ms rather than roughly 0.0154→0.0123.
Keep that variation rather than claiming every observation is stationary.

### Full scans expose control errors

Neither profiled large shape reaches timing:

- `spatial-7x7`: all 16 comparisons reject the unsplit control on ordinary
  inputs. Element 177 is `-7.426374e-3`, versus f64 `-7.406972790242605e-3`.
  Its approximately `1.94e-5` error exceeds the approximately `1.15e-5` bound.
- `pointwise`: twelve comparisons (counts 3, 4, 8) reject the unsplit control
  on tiny inputs. Element 6391 is `1.4238133e-14`, versus f64
  `1.4255220975027645e-14`; approximately `1.71e-17` error exceeds the
  approximately `1.29e-17` bound. The four two-way comparisons exit earlier
  at the ordinary reference/parity gate. Their original message does not
  identify the variant or element; do not invent that attribution.

A subsequent **CPU-only diagnostic**, now an ordinary library regression,
scatters from input coordinates independently of the GPU K-gather. For the two
reported control elements, sequential `f32::mul_add` reproduces the observed
GPU bits exactly, and independent f64 scatter agrees with the runtime oracle.
These particular discrepancies are accumulation rounding, not an indexing
discrepancy. This does not prove every element or backend behaves identically.
The follow-up also adds the variant number to future generic failure messages;
it does not relabel or rerun the frozen cohort above.

The ordinary tuner's 32 sampled f64 dots missed these two control elements;
full scans find them. Strict-f32 storage/arithmetic policy is not a promise that
every long sum meets a fixed error bound. Do not weaken the bound, time an
unqualified control, or treat agreement between two scalar tiles with the same
accumulation order as independent correctness evidence.

Conversely, this cohort's three-way long-tail comparison passes its hashed
synthetic patterns, while the [earlier LCG fixture](../split-k-2026-09-06/README.md)
still fails a tiny partial on the same shape. No shader arithmetic changed
between those tests. Qualification is evidence about the executed inputs, not
a guarantee over all values for that shape. The earlier rejection remains
active and prevents calling three-way splitting universally qualified.

### Search and memory costs

| Shape | Median search | Median qualification | Median CPU validation | Peak scratch requests |
|---|---:|---:|---:|---:|
| spatial-7x7 | 72.18 ms | 67.09 ms | 55.23 ms | 7,363,328 B |
| pointwise | 3,038.27 ms | 3,033.41 ms | 3,015.87 ms | 7,815,168 B |
| rectangular-tail | 31.96 ms | 9.78 ms | 5.02 ms | 22,764 B |
| long-tail | 2,000.68 ms | 390.07 ms | 339.79 ms | 3,408,724 B |

The pointwise row pays for full f64 checks repeatedly and then rejects; this is
CPU numerical work, not the old mapped-readback problem. The long-tail row
spends about 1.57 seconds in sampling. The larger 120-second ceiling was not
exhausted; the default two-second ceiling remains a soft bound and can stop
this exhaustive probe before all counts finish. No claim is made that this
full-oracle search is production-cheap.

Live tensor requests are separately 23,156,236 / 11,370,508 / 38,400 /
9,962,516 bytes in the table's order. Staging allocations equal releases in
every call; retained bytes are zero. The small case needs two or three staging
allocations depending on count order because eight-way partials become the
largest binding. Other cases allocate staging once and reuse it three times.
All 98 telemetry samples remain: utilization 0–99%, graphics clock 202–2902 MHz,
reported device memory 271–348 MiB, power 7.06–47.35 W and temperature 44–48°C.
Those coarse observations neither diagnose each short timing transition nor
measure peak VRAM.

Before measurement, the ordinary release suite, all 267 library tests including
ignored state/staging checks, all eleven convolution suites, five scalar tuner
tests, Clippy, MSRV, strict docs, package verification and frozen-paper replay
passed. Native-f32 cooperative hardware remains outside this device's coverage.
The [CPU evidence replay](../../../tests/split_k_evidence.rs) checks the full
roster, source identities, windows/telemetry, tensor summaries, complete sequence
sizes, staging lifetimes, phase arithmetic, raw pairs, guards and summary.
Mutation tests reject changed counts, bytes, timings, rejected-work promotion and
state. This replays recorded observations, not absent full oracle vectors.

The next general engineering task is to improve long-reduction accuracy and
qualification coverage, including the discovered control fixtures, before
claiming improvement on the profiled shapes. Favor a small shared accumulation
strategy, not a shape blacklist or tolerance exception. Whole-step experiments
with qualified rebuilt plans remain useful, but the synthetic 6.93× result is
not sufficient to promote split-K for training. Defaults and paper tables stay
unchanged.
