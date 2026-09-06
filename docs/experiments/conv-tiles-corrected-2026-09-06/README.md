# Corrected convolution tile crossover

## Predeclared correction, before measurement

This is a new cohort, not a replacement for the [original six attempts](../conv-tiles-2026-09-06/README.md).
Archive review found that the new small-case builder violated `Graph::conv2d`'s
documented flat NCHW operand contract: it supplied 4-D inputs and weights.
Forward compilation inferred batch from `shape[0]`, yielding zero workgroups
for the first convolution. Both sessions therefore had zero loss, gradients
and moments; advancing Adam's counter did not establish actual training.
Their original `complete` status reflects insufficient runner checks. Those
cases are disqualified as training/performance evidence. The ResNet case used
the proper flat layout and nonzero gradients; its six inconclusive records
remain separately interpretable. No raw files or evidence tags are rewritten.

Correct only the small-case operand shapes to flat arrays of identical declared
element counts. The public convolution helper now rejects nonflat/mismatched
operands early instead of silently accepting them; it does not add a new layout.
The full f64 oracle now checks every forward output as well as both derivatives.
The distinct-state convolution swap test uses proper flat operands and requires
a real parameter update. No tuning eligibility, shader arithmetic, candidates,
numerical tolerances, priorities, budgets or performance gates change.

Strengthen the runner: reject any zero-workgroup convolution cohort plan before
the prefix; require nonzero loss and gradient norms after the matched prefix;
require actual parameter changes in both optimizer-backed sessions by the end.
Keep full tensor/moment/counter comparisons, exact search/swap isolation and
all timed loss checks. Full source and this correction are committed before
measurement, with a new immutable evidence tag and executable identity.

Retain six fresh serial processes, seeds 1..6, running the same ordered three
cases (ResNet F+L+B, two-layer convolution Adam, same graph SGD), initializers,
strict-f32 policies, eight-class/64 MiB/ten-second search and role-reversed
whole-step protocol documented in the original predeclared protocol. The
same 5% plus twice-paired-MAD guard applies. No further reruns, case changes,
budget expansion or threshold fitting after these results. If a new validity
issue arises, retain/disqualify it rather than silently repairing its records.

Use the same [Blade 0.9 lock](../readback-2026-09-06/Cargo.lock), SHA-256
`72cecddd0f090e82d5db46bc6f7f80429a058b2afa0eaec84a9b27e7340a4a80`.
Build before measurements; no other build, qualification or heavy host analysis
runs during the cohort. Telemetry, create-new incremental records, full raw
dispatch contracts and all evidence limitations remain unchanged.

```sh
cargo build --release --locked --example tune_crossover
for seed in 1 2 3 4 5 6; do
    target/release/examples/tune_crossover "new-corrected-conv-${seed}.json" "$seed" --conv-derivatives
done
```

## Retained results

Measured source `e2def382b4971cd800d93bf408f0dd575025a3a7`, immutable tag
`evidence/conv-tiles-corrected-2026-09-06`; executable SHA-256
`c747cba9644622477f33326bbf79956f833011c331f049ed3c3dc122b478378d`.
All six serial processes ran September 6, 18:11–18:15 UTC on the RTX 5070 /
NVIDIA 595.71.05, Intel i5-12400F, Rust 1.98.0. No attempts were replaced:
[run 1](run-01.json), [run 2](run-02.json), [run 3](run-03.json),
[run 4](run-04.json), [run 5](run-05.json), [run 6](run-06.json).
[Machine-readable summary](summary.json).

| Case | Whole-step ms, baseline → selected¹ | Median process ratio (range) | Changed dispatches | Decision in all six processes |
|---|---:|---:|---:|---|
| ResNet-50 F+L+B | 17.5808 → 16.7293 | 1.05056× (1.04823–1.05100×) | 8 | Inconclusive |
| Convolution + Adam | 0.134955 → 0.134767 | 1.00027× (0.99667–1.00250×) | 0 | Unchanged selection |
| Convolution + SGD | 0.140568 → 0.140513 | 0.99955× (0.99853–1.00077×) | 0 | Unchanged selection |

¹ Medians of six per-process pooled medians; ratios are aggregated separately,
not calculated from those aggregate times. No cross-workload pooling.

All 18 A/A controls pass the stability screen. ResNet's observed per-process
time reductions are 4.60–4.85%, below 5% even before adding the noise margin.
Neither orientation nor pooled timing passes the unchanged gain guard. This
is a repeatable positive observation, **not a confirmed whole-step gain under
this protocol**. Do not lower the margin, advertise a PyTorch/training win,
calculate accepted amortization or enable tuning by default. Stable initial
A/A does not eliminate later period effects: unchanged SGD has a 0.95957×
right-selected orientation in process 3 despite an approximately 1.00035×
pooled ratio. Retain both statistics; identical choices did not cause a new
kernel regression, and pooled near-unity is not proof of no timing drift.

ResNet has 105 eligible derivative dispatches in 45 exact classes; the unchanged
eight-class structural budget visits seven dX classes and one dW class. Four
dX classes choose 32 instead of 64 in every process:

| Input channels / spatial | Output channels / kernel / stride | Repeated dispatches | Isolated median ms, 64 → 32² |
|---|---|---:|---:|
| 256 / 14×14 | 256 / 3×3 / 1 | 5 | 0.21802 → 0.12787 |
| 512 / 14×14 | 512 / 3×3 / 2 | 1 | 0.62700 → 0.42907 |
| 256 / 28×28 | 256 / 3×3 / 2 | 1 | 0.39401 → 0.33249 |
| 1024 / 14×14 | 2048 / 1×1 / 2 | 1 | 0.38929 → 0.30889 |

² Medians across six process-specific isolated medians. These changes pass
the isolated guard; they are not additional whole-step measurements. The
other four visited classes retain 64; all three small-case classes retain 32.
The search does not visit all 45 classes or the expensive stem dW. The
repetition×contraction-size ordering is only a structural prior, not a ranking
of measured time or available parallelism. Its coverage must remain visible.

All 84 private class comparisons qualify. All declared full-state comparisons
are bit-exact; loss/gradient prefix signals and Adam moments are nonzero, and
both optimizer sessions actually change parameters. Adam reaches step 178.
All 720 A/A and 1,440 crossover loss pairs are finite, nonzero and bit-exact.
Search and swaps preserve state exactly. Resident requested tensor bytes stay
311,439,916 / 317,660 / 286,652 for ResNet / Adam / SGD respectively. These are
request-accounting and control-session observations, not independent model
oracles, allocator/driver peaks or convergence evidence.

| Case | Median search ms | Preparation | Qualification | Warmup | Sampling | Peak requested scratch bytes |
|---|---:|---:|---:|---:|---:|---:|
| ResNet | 640.39 | 137.39 | 73.15 | 12.49 | 412.13 | 19,376,128 |
| Adam chain | 36.93 | 12.23 | 3.20 | 0.70 | 19.39 | 127,872 |
| SGD chain | 37.74 | 12.20 | 3.21 | 0.74 | 19.88 | 127,872 |

Independent medians need not sum; total includes cleanup/bookkeeping. ResNet
requests seven staging allocations and one exact-size reuse; small cases one
allocation and two reuses. All are released before return. Search now spends
most time sampling the larger derivative kernels, not CPU readback copying.
This is not comparable to the old one-class dense-only ResNet search cost.

All 862 telemetry samples are retained: GPU utilization 0–93%, graphics clocks
217–2,902 MHz, memory clocks 405–14,001 MHz, temperature 42–64°C. Clocks were
not locked; graphics processes remained resident and no compute process was
listed before the cohort. No build or other measurement overlapped it.

CPU [replay and mutation tests](../../../tests/conv_tuning_evidence.rs) check
source/settings, exact class-to-dispatch changes, physical buffers, batch-aware
geometry, scratch lifetimes/timers, complete state rosters, training signals,
loss pairs, swaps, telemetry and all decision arithmetic. Development builds
enable accurate f64 JSON round-trip parsing: default parsing introduced a few
ulps of cancellation error in a tiny paired timing statistic. The recorded
data and strict comparison/decision tolerances were not relaxed. The dependency
lock is unchanged. Full vectors remain unarchived; replay does not recreate
them. Earlier cohorts and frozen paper tables remain separate.

Next: bounded split-K dW with charged partial storage and final reduction,
an explicit direction/search-coverage policy, independent oracles and another
predeclared whole-step test. Neither broader coverage nor a new algorithm is
obtained by fitting the existing performance threshold.
