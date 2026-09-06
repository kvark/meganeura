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
[archived Blade 0.9 lock](../readback-2026-09-06/Cargo.lock), SHA-256
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
