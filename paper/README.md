# Meganeura papers

## P3HPC (SC26 workshop)

The current paper is [p3hpc/main.tex](p3hpc/main.tex). Its framework comparisons
use only the September 25 v14 cohort: Inferena `7b8fcb72`, Meganeura
`0dbfcc00`, and PyTorch 2.13.0 at source `cf30153c`. Both measured repositories
are tagged `paper-p3hpc-2026-final`. No earlier benchmark fills a missing cell.

[RESULTS.md](p3hpc/RESULTS.md) records the audit, failures, timing caveats,
the same-revision search ablation, footprint figures and conclusions. [QUALIFICATION.md](p3hpc/QUALIFICATION.md) describes the native-only
GPU qualifications, which do not supply CPU-versus-GPU speed comparisons.
Raw data, PDFs and ZIPs stay outside Git.

Replay the evidence without a GPU, from the repository root (Python 3.11+):

```sh
python3 paper/p3hpc/artifact/cohort.py "$HOME/Downloads/p3hpc" \
  --check paper/p3hpc/tables --output target/p3hpc-20260925
python3 paper/p3hpc/artifact/search_ablation.py "$HOME/Downloads/p3hpc" \
  "$HOME/Downloads/p3hpc/search-ablation.tgz" \
  --check paper/p3hpc/tables --output target/p3hpc-20260925
python3 -m unittest discover -s paper/p3hpc/artifact -p 'test_*.py'
```

`cohort.py` validates the fourteen archives and two crash logs identified by
[cohort.sha256](p3hpc/artifact/cohort.sha256) and regenerates eight LaTeX
fragments, including the strict ratio figure. `search_ablation.py` validates
the native-only search ablation against the cohort's PyTorch outputs and
writes the ninth, `ablation.tex`. Both also emit CSV/JSON summaries. See the
[artifact README](p3hpc/artifact/README.md) for details.

### Template and build

The [workshop instructions](https://p3hpc.org/workshop/2026/submissions/)
require IEEE proceedings format, at most 12 content pages and 16 pages
including references and appendices. The paper uses the vendored IEEEtran
conference class, version 1.8b, and does not change its margins or font sizes.

The artifact description follows the
[SC26 author template at b5195e6](https://github.com/jennfshr/sc26-repro/tree/b5195e67d9ad0b5d07e8b6840558c7251c73b3c0/for-paper-authors).
[p3hpc/sc26repro.sty](p3hpc/sc26repro.sty) is copied from that revision;
its IEEEtran class matches ours apart from whitespace.
[p3hpc/artifact-description.tex](p3hpc/artifact-description.tex) uses the
template's contribution/artifact map and six subsections per artifact.
It is appended after the bibliography, with no authors added to its heading,
no example/explanation text, and no optional AE or badge claim.

Build from `paper/p3hpc`:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

IEEEtran needs the PostScript base fonts (Times, Courier and Helvetica).
For the small TeX Live container, install `collection-fontsrecommended`.
Do not substitute `lmodern`, which changes the IEEE font.

The bibliography combines shared `paper/references.bib` and workshop additions
in `paper/p3hpc/references.bib`; their citation keys are distinct.
A source ZIP must retain both files and their relative paths, as well as
`artifact-description.tex`, `sc26repro.sty`, the class/style and figures/tables.

After building the PDF, package the source and supplementary ZIPs together:

```sh
python3 paper/p3hpc/artifact/submission.py "$HOME/Downloads/p3hpc" \
  --pdf paper/p3hpc/main.pdf --output paper/p3hpc/submission \
  --submission-id <SC submission number>
```

The source ZIP carries both bibliography files, `main.bbl`, the class/style,
SC's `fancyhdr.sty`, the AD, figures and tables with their relative paths.
SC26 wants auxiliary materials as one `<SC submission number>aux.zip` with a
short `readme.txt`; ours is [artifact/SUPPLEMENT.md](p3hpc/artifact/SUPPLEMENT.md)
and sits at the archive root with a regenerated `records.jsonl.xz`, the
ablation archive, the analyzers and a `MANIFEST.sha256`.

The first-page copyright block follows the
[SC26 Instructions for Workshop Authors](https://submissions.supercomputing.org/static_resources/SC26_Instructions_for_Workshop_Authors.pdf)
(Step 3a) with SC's `fancyhdr.sty`. The camera-ready PDF must also pass IEEE
PDF eXpress (conference ID 72768X) and stay under 4 MB. Do not use the legacy packager below for
the current cohort.

## Original arXiv report

`main.tex`, `tables/` and `results/` at this directory level belong to
[arXiv:2608.01563](https://arxiv.org/abs/2608.01563).
Its frozen data use Meganeura `7561a64` and Inferena `7ca9c5c7`, both tagged
`paper-arxiv-1`. They are separate from the current P3HPC evidence.

From `paper/`, `python3 mktables.py` regenerates those historical tables.
From the repository root, the legacy checker remains available:

```sh
python3 paper/p3hpc/artifact/verify.py --repository --show-facts
```

The original report's 780M Whisper backward pair is excluded from training
ratios by its oracle-consistency audit. That finding does not describe the
new P3HPC cohort. `artifact/package.py` packages only this original evidence.
The separately versioned `dinovision-section.tex` comes from
`kvark/dinovision/experiments`; update it from there, not in place.
