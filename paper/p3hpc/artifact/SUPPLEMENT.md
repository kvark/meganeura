# Meganeura P3HPC final supplement

Final October 31–November 1, 2026 framework evidence for the P3HPC paper, with the
RTX 3050 recollected on November 10 against compiled CUDA PyTorch. Frozen sources:
Meganeura 567028b805425375148e88a7e1d464fafdd99777 and
Inferena a0799e4210eafa25161ec50d475b186c4b19d3da. The old
paper-p3hpc-2026-final tags point to earlier measurements.

## Contents

- `archives/`: sixteen original campaign archives; `final-cohort.sha256`
  identifies them exactly. No old run fills a missing result.
- `final_cohort.py`: frozen-validator audit and final table generation.
- `cohort.py`: historical code used only for the TikZ plot layout by this analyzer.
- `tables/`: eight cohort TeX fragments, derived `gap.tex` / `gap.json`,
  `summary.json`, `results.csv`, and
  `preparation.csv` (native setup/search and reference compilation by process).
- `op-bench/`: repository-shaped overlay with the separate strict-f32
  single-operation summary, shapes and runners.
- `warmup/`: summary and provenance of the separate 5/15/30/60-second diagnostic.
- `RESULTS.md`, `QUALIFICATION.md`: interpretation and coverage.
- `MANIFEST.sha256`: hashes of the packaged files.

## Replay

Python 3.12+, a CPU, several GiB of temporary disk, and a clean Inferena
checkout at the exact revision above. No GPU or PyTorch install is required.
Once the checkout is present, replay does not access the network:

```sh
sha256sum -c MANIFEST.sha256
python3 final_cohort.py archives --inferena /path/to/frozen/inferena \
  --check tables --output regenerated
```

The command checks all archive digests, 282 desktop native receipts, 795
paired-phase receipts and ten Android full-tensor receipts, then reproduces
the supplied tables and unrounded summaries. Android raw NPZ tensors are
absent, so the audit verifies hashed manifests and complete comparison
receipts, not original elementwise arithmetic. Rejected references remain
missing. B570 360M completes all native cells and all accelerated reference phases;
one strict reference process fails gradient repeatability.

## Read the numbers

All published ratios are Meganeura / PyTorch time: below one favors native.
A cell is a model/profile/phase, with three processes when measured. All 330
core native cells qualify across eleven GPUs. Seven timed GPUs plus completed
larger models contribute 252 native performance cells and 231 paired ratios.
The other four devices are correctness-only. No outliers are removed; CSV
process medians and paired ratios retain ranges, especially important on M3.
Numerical gates affect both eligibility and search selection. Native
single-step full-gradient fidelity is not the same as PyTorch's AMP training
trajectory gate. Numerical rejection alone establishes neither a vendor bug
nor a native speed advantage. Performance-portability scores inherit those
coverage and policy choices.

Fresh collection is not needed to regenerate the paper. To reproduce the
procedure on hardware, use the frozen Inferena checkout's EXPERIMENT.md,
including its driver prerequisites, two arithmetic profiles, ten-second
desktop warmup, thirty-second session search and recorded kernel memo state.
The warmup summary is a separate small diagnostic, not a new cohort arm;
its full raw records are outside this supplement. DinoVision is separately
versioned and publicly archived at the links in the paper.

The gap accounting is a derived analysis of the same frozen result CSV. Run
`python3 gap_analysis.py --input tables/results.csv --check tables` to verify
its per-condition identity and geometric means. `gap.json` is the full matched
condition inventory.

## Separate single-operation study

`op-bench/` is a small overlay for a Meganeura checkout, preserving the
`examples/` and `paper/p3hpc/presentation/data/` paths expected by the driver.
Copy its contents into a checkout that contains this runner and the tuner
scratch fix (single-op rows were collected on main `d32a495`), then see
`run_all.sh` for GPU-specific arguments.
Offline reading of `op-bench.tsv` needs no GPU. Re-running it needs both
engines and hardware; the framework-cohort analyzer does not execute it.

This uses three native processes and one reference process per shape, with
thirty device samples per process. Operands differ between engines. Both
engines time 30 serial operations and divide by 30, including the gaps
between them. The measurements support kernel comparisons, not an exact share
of the model-level gap or an independent cross-engine accuracy test. Only the
median native rows are retained; per-process runs remain outside this
supplement. The source runner and recorded
protocol are supplied, not all executed binaries or GPU traces.
