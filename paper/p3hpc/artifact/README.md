# Final P3HPC evidence

`final_cohort.py` audits the sixteen archives identified by
[final-cohort.sha256](final-cohort.sha256). The measured revisions are Inferena
`a0799e4210eafa25161ec50d475b186c4b19d3da` and Meganeura
`567028b805425375148e88a7e1d464fafdd99777`. The paired protocol is v33;
Android is v15. Older paper tags do not identify this collection.

## Replay

Use Python 3.12+ and a clean Inferena checkout at the measured revision:

```sh
python3 paper/p3hpc/artifact/final_cohort.py "$HOME/Downloads/p3hpc" \
  --inferena /path/to/frozen/inferena \
  --check paper/p3hpc/tables --output target/p3hpc-final
```

No GPU, PyTorch install, network or model downloads are required once the
archives and checkout are present. Extraction needs several GiB of temporary
disk. The analyzer imports the frozen collector's validators, verifies archive
hashes, input identity, binary checks and qualification receipts, then uses
its phase-level reporting rules. It does not execute archive contents.
The historical `cohort.py` supplies only the TikZ plot layout, not qualification
or results for the final cohort.

`--check` compares eight TeX fragments, the unrounded summary JSON and CSV.
Additional `results.json`, `search-receipts.json` and `failures.json` go to the
output directory for inspection; they are not needed as duplicate Git data.
CSV time ratios are **Meganeura / PyTorch**, and the median-of-process-medians
statistic is unchanged. Per-process medians and paired ratios preserve ranges;
these are not confidence intervals. No outliers are dropped.

## Interpretation

- Eleven GPUs pass 330/330 core native cells. Six measured GPUs plus completed
  larger models supply 222 native performance cells and 201 paired ratios.
- B570 360M completes all six native cells; four have ratios. One strict
  reference process fails gradient repeatability, leaving no aggregated
  strict training or subsequent latency ratio. All accelerated phases pass.
- Numerical rejection, execution failure and later phases not attempted are
  different outcomes. Missing references are neither zero nor infinite speedups.
- Native full-gradient single-step checks and PyTorch AMP trajectory gates have
  different scopes. A wider gate could change both engines' eligibility and
  native kernel selection; no tolerance sweep was measured.
- M3 has substantial process variation despite sleep prevention. Small timing
  differences should not be treated as robust wins.
- Kernel-decision reuse is recorded. First-session misses are not controlled
  driver/page-cache cold starts; subsequent sessions may already reuse decisions.
- Android archives omit raw NPZ tensors. The audit checks hashes and complete
  full-tensor comparison receipts, not the original tensor arithmetic.

The [result ledger](../RESULTS.md) and paper give failure details and policy.
The five correctness-only GPUs never contribute CPU/GPU timing ratios.

## Separate warmup diagnostic

`warmup/summary.csv` and `protocol.json` preserve a two-process-per-GPU
SmolLM2-135M accelerated study at the older Meganeura `c19cb1c` and a distinct
Inferena study revision; it is not a final-cohort condition. It keeps each selected program through cumulative
5/15/30/60-second checkpoints, after search has exercised the GPU. It is not a
cold-start study, does not directly sample ten seconds, and does not establish
a universal warmup duration. Full raw study records remain in the separate
study archive; this supplement carries the summary and provenance only.

## Packaging

After building the paper, `submission.py` requires `--inferena` and validates
the data against the checked-in tables before writing a source ZIP and a
supplement. The supplement retains the original sixteen archives (about
147 MiB compressed), the analyzers and compact generated inputs. It does not
include the old search ablation. It also carries the separate single-op
summary, available process rows and runners as a repository-shaped overlay.
See [SUPPLEMENT.md](SUPPLEMENT.md).

`cohort.py`, `cohort.sha256`, `search_ablation.py`, `search-ablation.sha256`,
and `ablation.tex` describe historical campaigns. `footprint.py` also measures
source and binary size for the new pinned revision. Do not run their older
protocol checks against the new collection. Legacy verification of the
original arXiv evidence remains separate in `verify.py` and `test_verify.py`.

### Derived gap accounting

`gap_analysis.py` uses the frozen result CSV on identical qualified model sets
and generates `gap.json`/`gap.tex` without GPU execution.

```sh
python3 paper/p3hpc/artifact/gap_analysis.py --check paper/p3hpc/tables
```
