# Meganeura P3HPC supplement

Evidence and CPU-only analyzers for "Meganeura: Vulkan and Metal as a
Performance-Portability Layer for GPU Training and Inference" (P3HPC, SC26).
The measured sources are public and tagged `paper-p3hpc-2026-final`:
Meganeura `0dbfcc0029bf98b33a03fc792e1f4e90ead17f23` and Inferena
`7b8fcb72e55410d8e33bbe948f89be50c691fcf7`.

## Contents

| File | Purpose |
|---|---|
| `records.jsonl.xz` | Every JSON value and text log from the fourteen campaign archives and two crash logs listed in `cohort.sha256` |
| `search-ablation.tgz` | Native-only search ablation: 60 runner records and logs, driver `ablation.py`, `ablation-runner.patch`, kernel-level tile study |
| `cohort.py` | Audits the campaigns and regenerates eight LaTeX fragments, including the ratio figure |
| `search_ablation.py` | Audits the ablation, validates it against the cohort's PyTorch outputs, and writes `ablation.tex` |
| `footprint.py` | Reproduces the source-line, stripped-runner and PyTorch install-size figures |
| `cohort.sha256`, `search-ablation.sha256` | Digests of the original archives |
| `tables/` | The nine LaTeX fragments used by the paper |
| `MANIFEST.sha256` | Digests of every packaged file |

## Replay the analysis

Standard-library Python 3.11 or later; no GPU or network is needed.

```sh
sha256sum -c MANIFEST.sha256
python3 cohort.py records.jsonl.xz --check tables --output regenerated
python3 search_ablation.py records.jsonl.xz search-ablation.tgz --check tables --output regenerated
```

Both commands must finish without an error; `--check` requires every
regenerated fragment to match `tables/` exactly. `regenerated/` also receives
per-condition medians and ranges (`conditions.csv`), failure records
(`failures.json`), search summaries (`search-summary.json`) and the ablation
CSV. The first command takes a few minutes.

`cohort.py` checks source and checkpoint identity, raw/joined record
agreement, sample medians, the cross-engine numerical gates, PyTorch's
full-tensor repeatability and replay receipts, native construction limits,
warmups, compilation deadlines, and phase-level replication. The retained
reports do not include every tensor element, so values the runners never kept
cannot be rechecked.

## Reading the results

- Headline medians use every main-matrix phase group with three qualified
  process pairs. Failed or unreached PyTorch phases have no timing; a failed
  reference is never counted as a speedup.
- On RX 7900 XT, compiled PyTorch fails Whisper training repeatability in all
  six processes. A separate eager/math diagnostic validates Meganeura's
  outputs and gradients; its timings never replace compiled timings.
- The Pennycook table keeps all six main platforms. An engine without a
  qualified three-process timing on any platform scores zero for that
  workload and phase; where only one engine qualifies, it has efficiency one.
- Apple M3 passes validation but varies widely between processes in both
  engines. `cohort.py` prints a post-hoc check that drops samples above three
  times their process median (5 of 3,600); the tables keep all samples.
- The two hard-failed campaigns (780M and B570 360M) are retained as
  availability evidence, not replaced by retries.
- The search ablation reruns only the no-search and two-second-tuner policies;
  its 60-second search arm is the cohort itself.

## Footprint figures

```sh
python3 footprint.py source <Meganeura checkout at 0dbfcc00>
python3 footprint.py binary <stripped or unstripped inferena-meganeura runner>
<measured venv>/bin/python footprint.py closure <measured venv site-packages>
```

## Fresh collection

Collection needs the GPUs and PyTorch backends listed in the paper. Follow
Inferena's `EXPERIMENT.md` at the tag above and run `scripts/p3hpc.py`. For
the search ablation, apply `ablation-runner.patch` to Inferena and run
`ablation.py` from `search-ablation.tgz` on the GPU host.
