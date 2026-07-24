# Meganeura paper

This directory contains the working arXiv manuscript. The benchmark numbers
in `main.tex` remain placeholders until Meganeura and Inferena are committed,
pinned to one another, and rerun on the complete device matrix.

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

Before submission:

1. populate the strict and practical-default result tables from the frozen
   device matrix;
2. generate figures from immutable Inferena JSON artifacts;
3. add per-dispatch analysis for the two largest frozen gaps;
4. record clean Meganeura and Inferena revisions in the artifact appendix;
5. verify every bibliography entry against its primary source;
6. update the AI-assistance disclosure to match the final workflow;
7. run arXiv's TeX source checker and inspect the rendered PDF.

The intended initial submission is a technical preprint, not an anonymous
conference manuscript. A later systems-conference version can use the same
technical core after adapting format, anonymity, artifact, and venue-policy
requirements.
