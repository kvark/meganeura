# P3HPC cohort: September 25, 2026

The final v14 collection is usable with the failures and timing variation
described below. No GPU measurements were rerun during this audit.
All framework timing tables and the figure use only this cohort. Earlier
size-study rows, optimizer timing ablations, compiler comparisons and profile
timing tables have been removed. The separate DinoVision application and
MI300X driver findings supply no framework comparison times.

## Frozen inputs

- Inferena: `7b8fcb72e55410d8e33bbe948f89be50c691fcf7`.
- Meganeura: `0dbfcc0029bf98b33a03fc792e1f4e90ead17f23`.
- Blade: `fbb4f28c4869e81ae15de58925945b423b9c1ac5`.
- PyTorch: 2.13.0, source `cf30153c4c131c8164ee7798e5022d810682e2cb`,
  with cu130, ROCm 7.2, XPU, MPS or CPU wheels as recorded.
- Python: 3.13.13.
- Both measured repositories are publicly tagged `paper-p3hpc-2026-final`:
  [Inferena](https://github.com/kvark/inferena/tree/paper-p3hpc-2026-final) and
  [Meganeura](https://github.com/kvark/meganeura/tree/paper-p3hpc-2026-final).
  The tags identify benchmark sources, not this paper revision.
- Synthetic parameters: `name-index-uniform-v1`; SmolLM2 uses the pinned
  public checkpoints.

All fourteen campaign manifests agree on source, lockfile hash, and hashes
of shared checkpoints. Each native session uses calibrated egglog search:
sixteen graph/schedule forms, up to 64 programs, and a shared soft 60-second
deadline. Strict permits native-f32 cooperative matrices. PyTorch uses default
compilation with a supervised 120-second deadline; no max-autotune.
CUDA/HIP/XPU replay is enabled. MPS compiles and qualifies ordinary calls.
Both engines warm each measured phase for at least five calls and two seconds,
then retain twenty host-wall samples. There are three process pairs per
measured condition; CPU-oracle qualifications use one.

The input directory is `~/Downloads/p3hpc`. The tracked
[SHA-256 manifest](artifact/cohort.sha256) identifies fourteen archives and
two standalone crash logs. Raw data stay outside Git.

## Coverage and failures

| Evidence | Outcome | Use |
|---|---|---|
| RTX 5070, H100, Windows RTX 3050, Arc B570, Apple M3 main campaigns | 30/30 full pairs each | Main timing table |
| RX 7900 XT main campaign | 24 full pairs; six qualified inference-only pairs | Native times retained; compiled PyTorch Whisper training fails and its minimal timing is unreached |
| 360M on RTX 5070 and RX 7900 XT | 6/6 full pairs each | Same-revision size study |
| 360M and 1.7B on H100 | 12/12 full pairs | Same-revision size study |
| Radeon 780M, Ryzen 9600X iGPU, Intel RPL-U | 10/10 qualifications each | Vulkan correctness against eager CPU; no CPU/GPU speed ratios |
| Radeon 780M GPU-reference attempt | First strict 135M pair fails in PyTorch/HIP | Availability failure; 29 planned pairs unreached |
| Arc B570 360M attempt | First strict pair fails in PyTorch/XPU capture | Availability failure; five planned pairs unreached |

The main matrix has **180 valid inference phase pairs and 174 valid pairs in
each other phase**. The extension adds 24 complete pairs. Thus there are
198 full GPU pairs and six inference-only pairs, plus 30 CPU-oracle
qualifications. Do not count the six eager diagnostics as compiled timing
pairs, or the 34 unreached pairs as execution failures.

RX 7900 XT Whisper fails the first ordinary training repeat in all six
processes, before HIP capture. Maximum output error is 0.001838–0.002452
against a bound of about 0.000009101. RMS error is 0.00007956–0.00009781
against about 0.000003079. Between 133,282 and 170,203 of 576,000 elements
also exceed the diagnostic pointwise bounds. All six inference captures pass.
Every separate eager/math diagnostic passes repeatability and the native
cross-engine gates; its largest parameter-norm-vector error is 0.000712%.
This is a PyTorch/ROCm numerical portability failure, not a disqualification
of RX 7900 XT or Meganeura. The table retains the native measurements from
all three processes, independently qualified against the eager diagnostics:

| Native Whisper time (ms) | Inference | Minimal shape | F+L+B |
|---|---:|---:|---:|
| Strict | 4.47 | 4.45 | 21.55 |
| Accelerated | 3.92 | 3.92 | 20.48 |

Only inference has a compiled PyTorch ratio. No eager timing enters a
compiled ratio. The top-level
`complete-with-failures` and failed replicated-gradient report accurately
describe the two missing Whisper training groups; they are not archive damage.

The B570 360M failure occurs in `aten.embedding_dense_backward` during
training graph capture: “wait method cannot be used for an event associated
with a command graph.” Compilation finished in 89.62 seconds. The working
135M embedding probe/workaround does not establish capture support at 360M.
The native 360M run completes its own full-tensor construction checks, but
has no independent reference, so its timings are not admitted.

The archived 780M attempt reports an unspecified HIP launch failure on its
first strict 135M condition. Its preparation sidecar still says `running`;
we cannot claim compilation completed. The separate text traceback has
different addresses and a different stack, so it is supplementary failure
evidence, not a second identified campaign. No successful retry replaces it.

The new RPL-U and 9600X archives establish native qualification, not a fresh
GPU-reference availability probe. The paper identifies RPL-U's unavailable
XPU path and the reported 9600X ROCm failures from the separate bring-up
evidence. A CPU wheel alone would not establish either finding. The shared
RADV name `RAPHAEL_MENDOCINO` is not a separate Mendocino machine.

## Audit

The auditor reads all JSON records and 244 text logs, without extracting or
executing archive contents. It checks:

- Archive hashes; shared source, lockfile and checkpoint identity.
- Raw/joined record equality, sample medians, arithmetic permissions,
  execution backend, warmups, search limits and compilation receipts.
- Full-tensor ordinary/replay reports, including MPS and CPU, and recorded
  native construction qualification.
- Cross-engine sampled outputs, scalar loss, total gradient norm and
  per-parameter norm vectors; replicated gradient rules for complete groups.
- Preservation of partial phases, absence of timings for failed/unreached
  reference phases, and the separation of eager diagnostic data.

All admitted results pass. Across the complete GPU and CPU-oracle pairs,
maximum parameter-norm-vector error is **0.296%**, sampled-output relative L2
error **0.241%**, scalar-loss error **0.0140%**, and total-gradient-norm error
**0.258%**. The six eager diagnostics are tighter. The full tensors were
not archived, so this verifies retained evidence, not unrecorded elements.

The logs contain no additional unexplained campaign failures. Printed
`NaN`/`inf` gradient comparisons in partial Whisper runner tables denote
missing reference gradients; the JSON stores absent comparison metrics,
not accepted nonfinite tensors.

## Performance

All ratios are Meganeura/PyTorch; lower is better.

| Contract | Inference | Minimal shape | F+L+B |
|---|---:|---:|---:|
| Strict, median over qualified groups | 1.27 (30 groups) | 1.09 (29) | 1.92 (29) |
| Strict, nominal native wins | 7/30 | 12/29 | 4/29 |
| Accelerated, median over qualified groups | 2.00 (30) | 1.27 (29) | 2.61 (29) |
| Accelerated, nominal native wins | 6/30 | 10/29 | 4/29 |

The same phase groups are present under both contracts. RX 7900 XT Whisper
compiled PyTorch minimal/training is absent; native timings and valid
inference ratios remain. A failed reference is an availability result,
not an infinite speedup.

The Pennycook table keeps **all six main platforms**, including RX 7900 XT.
For each engine/workload/phase, a missing qualified three-process timing gives
a zero score. On a platform with only one qualified engine, that engine is
the best available and has efficiency one; this is not a speedup over the
failed reference. Strict Meganeura/PyTorch workload-mean scores are 0.64/0.90
for inference, 0.76/0.56 for minimal shapes, and 0.48/0.74 for training.
PyTorch's Whisper minimal and training scores are zero because its campaign
did not supply them. The minimal phase was unreached, not independently shown
to be unsupported. Native Whisper timings qualify through the separate
eager diagnostics. CPU-oracle qualification campaigns and larger-model runs
do not enter this declared six-platform set.

RX 7900 XT wins strict SmolLM2, SmolVLA and diffusion in all phases.
RTX 5070 strict inference reaches 1.20 for SmolLM2 and 1.09 for SmolVLA,
and beats PyTorch for diffusion (0.91) and ResNet (0.74).
B570 still loses every main timing comparison: strict inference ratios
1.76–2.62, training 2.50–5.07. It records no cooperative dispatches;
the current graphics path does not support its rectangular tiles.
M3 records 2,364 strict cooperative dispatch instances across sessions.

**M3 is numerically sound but noisy.** Native strict one-token SmolLM2
process medians span 6.88–43.11 ms; PyTorch strict SmolVLA inference spans
8.95–40.35 ms. Both engines vary, so no single-cause explanation follows
from the receipts. The operator reports possible sleep interruptions. A
post-hoc sensitivity check removes samples above three times their own
process median, identically for both engines: just five of 3,600 samples.
It leaves all inference aggregates and 59/60 engine/condition/phase medians
unchanged. PyTorch strict SmolLM2 training changes from 101.96 to 99.40 ms
(2.51%); no ranking changes. This does not identify which calls slept or
remove sustained machine-state variation. The analyzer prints this check
without altering the primary results. Keep the original medians and ranges;
M3 supports broad observations, not strong claims about near-ties.

## Scaling and preparation

The current H100 series is complete; no earlier data fill gaps:

The 135M baseline and extension use different H100 allocations. Their GPU
UUIDs and host kernels differ (6.8.0-106 versus 6.8.0-90), while GPU model,
132-SM configuration, memory capacity, driver 580.126.09 and software pins
match. The 360M and 1.7B runs share the second allocation. This is not a
controlled same-host size ablation; the paper discloses that limitation.

| H100 M/P ratio | 135M | 360M | 1.7B |
|---|---:|---:|---:|
| Strict prefill | 1.84 | 1.93 | 2.52 |
| Strict one token | 1.01 | 1.15 | 2.20 |
| Strict F+L+B | 3.25 | 2.96 | 2.78 |
| Accelerated prefill | 3.73 | 5.00 | 12.26 |
| Accelerated F+L+B | 4.36 | 4.84 | 6.58 |

Larger weights narrow only the strict training gap. They do not amortize
the inference gap uniformly. At 360M, RTX 5070 strict prefill/one-token
ratios are 1.69/1.40, while RX 7900 XT remains ahead at 0.70/0.26;
its strict training ratio is 0.86. These are batch-one, 128-token prefill
and stateless one-token forwards, not KV-cached serving or production training.

Native accelerated 1.7B prefill is slower than strict (53.58 vs 29.90 ms).
The main cohort is not a tuning-off/on or cooperative-off/on ablation.
The final cohort cannot attribute a causal speedup to egglog: it contains no
same-revision greedy or search-off arm. The paper describes the alternatives
it retains and reports the current search decisions, without importing older
ablation numbers.

Main-matrix search tries 19,696 programs in 504 sessions; 169 select a
different graph/schedule from ordinary extraction. There are 459 unfinished
searches and 396 extraction truncations; no skipped regions. Kernel probes
visit or reuse 291,436 of 293,310 class instances, accepting 7,133 replacements,
rejecting 3,214 kernel outputs and 725 whole programs. Repeated/reused classes
are not independent algorithms. The longest session search is 64.39 seconds.

Strict process-median preparation across thirty workload/platform groups
is 127.90 seconds native versus 26.73 seconds PyTorch. H100's main matrix
totals 72.81 versus 16.86 minutes. The native policy is now much more
expensive than default compilation. Max-autotune failures and long waits in
bring-up explain the bounded reference policy, but their older timings are
not reused. Do not infer whole-model startup from Naga parse time.
H100 strict native plan allocations grow 1.58 → 3.90 → 17.04 GiB; these are
not physical VRAM peaks or comparable in scope to PyTorch allocator peaks.

## Reproduction and paper status

```sh
python3 paper/p3hpc/artifact/cohort.py "$HOME/Downloads/p3hpc" \
  --check paper/p3hpc/tables --output target/p3hpc-20260925
python3 -m unittest discover -s paper/p3hpc/artifact -p 'test_*.py'
```

The audit regenerates nine LaTeX fragments, the SmolLM2 figure among them,
plus per-condition timings/ranges, failures, and compact search summaries.
Generated data remain under ignored `target/`. Only analysis code, hashes,
compact findings, and paper sources are tracked.

The artifact description now follows the SC26 AD template at
[`sc26-repro/b5195e6`](https://github.com/jennfshr/sc26-repro/tree/b5195e67d9ad0b5d07e8b6840558c7251c73b3c0/for-paper-authors),
using its vendored `sc26repro.sty`, contribution/artifact mapping and six
required artifact subsections. It is appended after the bibliography; there
is no AE/badge claim. The paper retains IEEE proceedings formatting.

This update does not regenerate the PDF or either submission ZIP. Existing
submission files still describe the earlier cohort and must be rebuilt at
final packaging. The supplement must include its README and the new checker,
hash manifest, tables, and record stream together. Earlier audit and diagnostic
notes remain in Git history at `cc49b11`.
