# Meganeura paper

This directory contains the source of
[arXiv:2608.01563](https://arxiv.org/abs/2608.01563). The numeric tables in
`tables/` are generated from the frozen benchmark artifacts in `results/`
(five devices, five workloads, strict and accelerated modes, all at
Meganeura `7561a64` / Inferena `7ca9c5c7`; the gap-profile sidecars in
`results/*/profiles/` record their own revision):

```sh
python3 mktables.py   # regenerates tables/*.tex, the ratio figure, and facts
```

Both revisions are preserved under the public tag `paper-arxiv-1`.
`dinovision-section.tex` is the frozen fragment from
`kvark/dinovision/experiments`; update it from there, not in place.

Rerun it after refreshing `results/`; every table in `main.tex` updates in
place. The script also prints the aggregate numbers cited in prose so a text
sweep can be checked against the artifacts.

The Radeon 780M--Whisper backward pair is retained in the raw results but
excluded symmetrically from training ratios and aggregation because the
PyTorch/ROCm record fails the cross-backend oracle-consistency audit. The
generator prints that audit and marks the affected table entry with
`\ddagger`; all forward measurements remain included.

`p3hpc/` holds the P3HPC (SC26 workshop) submission: IEEE format
(vendored `IEEEtran.cls`/`.bst`), single-blind, reuses `../tables/` and
`../references.bib`. Build the same way from inside that directory, but
note that IEEEtran needs the PostScript base fonts (Times/Courier/
Helvetica), which the small TeX Live image lacks --- run
`tlmgr install collection-fontsrecommended` in the container first (or
use the full `texlive/texlive:latest` image). Do not substitute
`lmodern`: it silently replaces the IEEE Times font.
The P3HPC paper has been accepted. Camera-ready is due **September 25, 2026**;
the working target is a reviewed draft by September 19. See the
[official submission page](https://p3hpc.org/workshop/2026/submissions/),
[revision plan](p3hpc/REVISION.md), and [study guide](../docs/study/README.md).
Reviewer-specific revisions still need to be incorporated when available.

Replay the frozen evidence without a GPU, from the repository root:

```sh
python3 paper/p3hpc/artifact/verify.py --repository --show-facts
python3 -m unittest discover -s paper/p3hpc/artifact -p 'test_*.py'
```

This checks records, sampled-output/gradient-norm gates, medians and table
regeneration. It does not run benchmarks or automatically validate all prose.
Keep post-freeze development results separate from `paper/results/`.

Build locally with:

```sh
latexmk -pdf main.tex
```

If `latexmk` is unavailable:

```sh
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

The repository was also checked with the current small TeX Live container:

```sh
podman run --rm -v "$PWD:/paper:Z" -w /paper \
  docker.io/texlive/texlive:latest-small \
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Before producing a revised arXiv version:

1. ~~populate the strict and practical-default result tables from the frozen
   device matrix~~ (done — `mktables.py`);
2. ~~add the Radeon 780M machine~~ (done — full member of the matrix);
3. ~~add per-dispatch profiles for the largest frozen gaps~~ (done —
   integrated in the gap-analysis section; optionally recapture the M3
   Whisper profile at the frozen revision to replace the pre-optimization
   one);
4. verify every bibliography entry against its primary source;
5. update the AI-assistance disclosure to match the final workflow;
6. run arXiv's TeX source checker and inspect the rendered PDF.

The public v1 is a technical preprint, not a peer-reviewed conference paper.
A later systems-conference version can use the same technical core after
adapting format, anonymity, artifact, and venue-policy requirements.
