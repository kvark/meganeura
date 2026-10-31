# Meganeura P3HPC final supplement

Final October 2026 framework evidence for the P3HPC paper. Frozen sources:
Meganeura c19cb1caedd90be7abc61de61f6197018711398e and
Inferena add19182e5479e3f7549894c524ac6691d3dc363. The old
paper-p3hpc-2026-final tags point to earlier measurements.

## Contents

- `archives/`: sixteen original campaign archives; `final-cohort.sha256`
  identifies them exactly. No old run fills a missing result.
- `final_cohort.py`: frozen-validator audit and final table generation.
- `cohort.py`: historical code used only for the TikZ plot layout by this analyzer.
- `tables/`: eight TeX fragments, `summary.json`, `results.csv`, and
  `preparation.csv` (native setup/search and reference compilation by process).
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

The command checks all archive digests, 277 desktop native receipts, 769
paired-phase receipts and ten Android full-tensor receipts, then reproduces
the supplied tables and unrounded summaries. Android raw NPZ tensors are
absent, so the audit verifies hashed manifests and complete comparison
receipts, not original elementwise arithmetic. Rejected references remain
missing. The incomplete B570 360M campaign is excluded from qualified cells.

## Read the numbers

All published ratios are Meganeura / PyTorch time: below one favors native.
A cell is a model/profile/phase, with three processes when measured. All 330
core native cells qualify across eleven GPUs. Seven timed GPUs plus completed
larger models contribute 246 native performance cells and 226 paired ratios.
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
condition inventory. `attention-followup.json` is separate October 24 local
evidence, excluded from all primary cohort ratios and coverage; it includes
the retained 100 samples per arm, protocol, source hashes and reported
qualification status for eight fixed-plan native shader comparisons. This
command replays sample medians and tables; it neither retimes models nor
independently reproduces their numerical qualification. No new PyTorch timing
or learning-convergence result is supplied by that local study.
