# Final P3HPC evidence

`final_cohort.py` audits the sixteen archives identified by
[final-cohort.sha256](final-cohort.sha256). The measured revisions are Inferena
`add19182e5479e3f7549894c524ac6691d3dc363` and Meganeura
`c19cb1caedd90be7abc61de61f6197018711398e`. The paired protocol is v33;
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

- Eleven GPUs pass 330/330 core native cells. Seven measured GPUs plus completed
  larger models supply 246 native performance cells and 226 paired ratios.
- The B570 360M campaign is incomplete; no independent reference or replicated
  timing exists. Internal native checks alone do not qualify a published cell.
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
The four correctness-only GPUs never contribute CPU/GPU timing ratios.

## Separate warmup diagnostic

`warmup/summary.csv` and `protocol.json` preserve a two-process-per-GPU
SmolLM2-135M accelerated study at the same Meganeura revision, but a distinct
Inferena study revision. It keeps each selected program through cumulative
5/15/30/60-second checkpoints, after search has exercised the GPU. It is not a
cold-start study, does not directly sample ten seconds, and does not establish
a universal warmup duration. Full raw study records remain in the separate
study archive; this supplement carries the summary and provenance only.

## Packaging

After building the paper, `submission.py` requires `--inferena` and validates
the data against the checked-in tables before writing a source ZIP and a
supplement. The supplement retains the original sixteen archives (about
151 MiB compressed), the analyzers and compact generated inputs. It does not
include the old search ablation. See [SUPPLEMENT.md](SUPPLEMENT.md).

`cohort.py`, `cohort.sha256`, `search_ablation.py`, `search-ablation.sha256`,
`footprint.py`, and `ablation.tex` are historical. Do not run their older
protocol checks against the new collection. Legacy verification of the
original arXiv evidence remains separate in `verify.py` and `test_verify.py`.
