# Meganeura P3HPC evidence

This directory contains the CPU-only auditor for the September 25 v14 cohort.
The paper uses Inferena `7b8fcb72`, Meganeura `0dbfcc00`, and PyTorch 2.13.0
at source `cf30153c`. Full revisions and findings are in
[RESULTS.md](../RESULTS.md). Raw measurements, PDFs and ZIPs stay outside Git.

The fourteen campaign archives, plus the later search-ablation archive,
contain:

- Six main five-model campaigns: 180 qualified inference phase pairs and
  174 qualified pairs in each other phase.
- Three completed larger-model campaigns: 24 full pairs, including 360M on
  RTX 5070, RX 7900 XT and H100, and 1.7B on H100.
- Two hard-failed GPU-reference attempts: 780M 135M and B570 360M.
- Three Vulkan qualifications against eager CPU PyTorch: ten conditions each
  on 780M, Ryzen 9600X's iGPU, and Intel RPL-U. No CPU/GPU timing comparison.
- `search-ablation.tgz`: 60 native-only processes on the RTX 5070 / B570 host
  without search and with the library's two-second tuner, plus the runner
  patch, driver script and kernel-level tile study. Its SHA-256 is in
  [search-ablation.sha256](search-ablation.sha256).

RX 7900 XT's main campaign also contains six separate eager/math diagnostics
after compiled Whisper training failed repeatability. Qualified inference is
retained. Failed or unreached phases have no accepted paired timing.
The native times remain in the table, qualified independently by those
diagnostics. This is a failure of PyTorch's ROCm path, not of Meganeura or
the GPU. Eager diagnostic timings never replace compiled timings.

Two standalone crash logs accompany the archives. The 780M text traceback
differs from the archived process; it is not counted as another measurement.

## Audit and regenerate

From the repository root, using standard-library Python 3.11+:

```sh
python3 paper/p3hpc/artifact/cohort.py "$HOME/Downloads/p3hpc" \
  --check paper/p3hpc/tables --output target/p3hpc-20260925
python3 paper/p3hpc/artifact/search_ablation.py "$HOME/Downloads/p3hpc" \
  "$HOME/Downloads/p3hpc/search-ablation.tgz" \
  --check paper/p3hpc/tables --output target/p3hpc-20260925
python3 -m unittest discover -s paper/p3hpc/artifact -p 'test_*.py'
```

The auditor checks archive hashes, source/checkpoint identity, raw/joined
agreement, sample medians, cross-engine numerical gates, full-tensor
repeatability/replay receipts, construction limits, workload warmups,
compilation deadlines and phase-level replication. It does not execute archive
contents, use a GPU, or access the network. Reading one JSON file at a time
keeps the detailed search receipts manageable.

`cohort.py` writes eight LaTeX fragments (including the strict ratio figure,
`ratio-plot.tex`), `conditions.csv`, `failures.json`, and
`search-summary.json`. `search_ablation.py` checks every ablation record's
identity and policy, validates its outputs against the cohort's PyTorch
outputs for the same GPU, workload and replicate, and writes `ablation.tex`
and `ablation.csv`. The 60-second search arm comes from the cohort; the
ablation never supplies a PyTorch or search timing.
[footprint.py](footprint.py) reproduces the source-line, stripped-runner and
PyTorch install-closure figures.
Each CSV row includes per-phase paired counts, independently qualified counts
for each engine, medians, process ranges and ratios. An empty ratio means there is no qualified paired timing, not zero.

Headline medians use every main-matrix phase group with three qualified pairs.
The Pennycook table keeps all six main platforms for every workload/phase.
An engine without a qualified three-process timing on any of those platforms
gets zero for that workload/phase. Where only one engine qualifies, its
efficiency is one as the best available result. RX 7900 XT stays included:
Meganeura's Whisper timings qualify through the separate eager diagnostics;
PyTorch's training failed and its minimal timing was unreached. No speed
ratio is invented for either missing phase.
The size study and CPU-oracle qualifications never enter these aggregates.

Apple M3 passes validation, but has large process-to-process timing variation
in both engines, with possible sleep interruptions reported by the operator.
The auditor prints a post-hoc check dropping samples above three times their
own process median, using the same rule for both engines. Five of 3,600
samples are flagged; no inference aggregate changes, and only PyTorch strict
SmolLM2 training changes (2.51%). This cannot remove variation spanning whole
timing windows. The primary results keep all samples and process ranges.
Near-ties are not statistically established wins.

## Supplementary record stream

At final packaging, preserve every original JSON value and every text log:

```sh
python3 paper/p3hpc/artifact/cohort.py "$HOME/Downloads/p3hpc" \
  --bundle /absolute/path/to/records.jsonl.xz
```

This XZ stream is not an engine cache or a new measurement protocol.
Its first line is an object with `format: p3hpc-files-v3` and the input digest
map. Each input then has a filename line, one `[path, value]` line per retained
file, and a `null` terminator. JSON files retain parsed values; text logs
retain strings. The order follows [cohort.sha256](cohort.sha256).
Ordinary SVG presentations, model weights, executables, caches and traces are
not needed to replay the analysis.

From an extracted supplement containing the analyzers, manifests, stream,
ablation archive and tables:

```sh
python3 cohort.py records.jsonl.xz --check tables --output regenerated
python3 search_ablation.py records.jsonl.xz search-ablation.tgz --check tables --output regenerated
```

The original archive hashes establish input identity; a separate
`MANIFEST.sha256` must protect the actual packaged files. Full-element
qualification reports do not include every tensor element. The checker
cannot reconstruct values the runners never retained.

The source ZIP and supplementary ZIP must each have clear top-level
instructions. The supplementary ZIP must contain a **README**. Do not include
earlier benchmark cohorts, H100 tuning pilots, or development timing profiles.
Separate availability notes for MI300X and the qualification-only GPUs are
supporting evidence, not substitutes for current results.

The source ZIP must include `artifact-description.tex` and `sc26repro.sty` as
well as `main.tex`, both bibliography files and the required figures/tables.
The AD uses the SC26 template; the optional AE section is omitted. Both
measured repositories carry the public tag `paper-p3hpc-2026-final`.

The PDF, source ZIP and supplementary ZIP are rebuilt together under the
ignored `paper/p3hpc/submission/` (see [paper/README.md](../../README.md)).
Do not combine the checkers with old tables or a v2 record stream.

## Legacy original-submission audit

`verify.py`, `verify.sh`, `package.py`, and most cases in `test_verify.py`
still cover the original five-machine data under `paper/results/`.
They do not package or verify the current cohort. The earlier detailed
artifact instructions and measurements remain in Git history; `cohort.py`
is the entry point for the current paper.
