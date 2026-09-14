# Camera-ready handoff

Submission **ws_p3hpc104**. Prepared September 14, 2026; nothing uploaded.
The author must approve the manuscript and any publication agreement.

The manuscript and packages use the completed v9 cohort at Inferena
`fa5a04e1` / Meganeura `428fc2d2`: all 252 pairs validate. Always-on native
tuning, strict native-f32 cooperative tiles, CUDA/HIP/XPU replay, compiled
MPS and bounded default PyTorch compilation are checked in executed records.
[RESULTS.md](RESULTS.md) gives the audit and cross-cohort interpretation;
[REVISION.md](REVISION.md) maps reviewer and author feedback to the revision.
The September 13 and first September 14 packages are superseded; no further
cohort is needed for the claims in this manuscript.

## Files

The local handoff directory is `~/Documents/P3HPC/camera-ready-20260914-r2/`:

- `meganeura-p3hpc-camera-ready.pdf`: manuscript for final author review.
- `meganeura-p3hpc-sources.zip`: TeX, bibliography, templates, figure and
  generated tables; build independently without a GPU or measurement archive.
- `meganeura-p3hpc-supplementary.zip`: every JSON record from the nine
  supplied campaigns, losslessly compressed, the offline analyzer, expected
  tables and per-condition CSV. The separately labeled H100 tuning pilot has
  its original analyzer; the MI300X report is included. No runner text logs,
  model weights, executables, driver caches or trace binaries.
- `BUILD-INFO.txt` and `SHA256SUMS`: source/build provenance and package hashes.

These publication files remain outside Git. The paper PR contains source,
small generated LaTeX fragments, archive identities and study documentation.
The main benchmark and separate profiling studies keep their own revisions.
The annotated tag `paper-p3hpc-2026` identifies measured Meganeura commit
`428fc2d2322229e5338f5d80a10d700340d593cd`, not the manuscript source.
No new benchmark was run during either paper update.

This author-review revision explains max-autotune's omission where the
reference policy is introduced, adds a sourced discussion of Hopper WGMMA
and the measured stack's subgroup-only matrix interface, and pins paper
links to Meganeura/Inferena code and experiment reports. It also removes
authorial em-dashes and repetitive rhetorical framing without changing data,
validation or the AI-assistance disclosure.

The supplied upload requirements prefer a PDF below 2 MB and reject files
above 4 MB, with separate PDF, LaTeX source ZIP and supplementary ZIP fields.
The PDF is about 320 KB, the source ZIP below 200 KB and the self-contained
supplementary ZIP below 3 MB. Exact bytes and SHA-256 digests accompany the
packages. No external data download is needed to reproduce the eight
generated table/figure fragments and all 84 condition groups.
No copyright/ISBN line or PDF eXpress conference ID was supplied or invented.

## Venue requirements

P3HPC explicitly requires the **IEEE proceedings template**, at most 12
main-text pages excluding references/appendices, and 5–16 pages overall.
Its camera-ready deadline is **September 25, 2026**. The author's supplied
portal screenshot further specifies 11:59 p.m. AoE for Stage 3.
[Official P3HPC instructions](https://p3hpc.org/workshop/2026/submissions/).

The manuscript uses the vendored `IEEEtran` class in conference mode, 10-point
type on US Letter, with no custom margin/line-spacing reductions. The packaged
v9 build has 13 pages; the main argument, acknowledgments and artifact
description end on page 12, with references continuing on page 13. Tables and all pages
were visually inspected; fonts are embedded Type 1 and no overflowing boxes,
undefined citations or references remain. PDF title/author metadata is set.
These local checks are **not** an IEEE PDF eXpress certification.

The screenshot labels Stage 3 **SC Workshop: P3HPC: Program Material**,
separately from **Workshop Camera-Ready Upload**. Treat them as distinct
program-information and proceedings-paper tasks, not a promised additional
review round. The exact authenticated form fields have not been inspected.
The title, author details, and short program abstract below are ready to paste;
add a biography or other material if the form requests it.

IEEE describes PDF eXpress as a conference-configured compatibility checker,
not an automatic substitute for final submission. Use the conference ID and
copyright/ISBN line supplied by the actual upload instructions, if required;
none has been invented or copied from another SC year.
[IEEE author guidance](https://events.ieee.org/planning-basics/ieee-conference-publications/publishing-information-for-ieee-conference-authors/).

SC26 requires AI-generated text disclosure and tool citations identifying
affected sections. The acknowledgment names the tools and scopes assistance
to all numbered sections and the artifact description. Confirm the final
disclosure accurately describes the author's review and satisfies any
acceptance-specific placement requirements.
[SC26 workshop policy](https://sc26.supercomputing.org/program/workshops/).

## Program material

Title: **Meganeura: Vulkan and Metal as a Performance-Portability Layer for
GPU Training and Inference**

Author/presenter: **Dzmitry Malyshau**, Independent Researcher.
Email: kvark@fastmail.com. ORCID: 0009-0005-6410-4276.

Short abstract (under 150 words; no provisional performance numbers):

Meganeura explores Vulkan and Metal as a common foundation for GPU inference
and training. Implemented in Rust, it combines computational graph compilation
with bounded, automatic kernel tuning, aiming to keep the engine small and
general while adapting to different hardware. We examine this approach across
NVIDIA, AMD, Intel, and Apple GPUs, using PyTorch as a reference. Our evaluation
considers not only steady-state performance, but also preparation time,
numerical validation, memory use, and practical deployment barriers. We discuss
how graphics APIs can support machine learning alongside rendering, where their
execution model helps or limits performance, and what observability is needed
to understand those trade-offs. The work asks how much performance portability
a compact engine can achieve without maintaining separate implementations for
each vendor.

## Final author actions

- Review the final PDF and the evidence/reviewer guides. In particular,
  native tuning improves several workloads but is not always cheaper than
  default PyTorch compilation; M3 variation and the accelerated 1.7B regression
  remain disclosed. The final cohort is not a controlled tuning/f32 ablation.
- A permanent URL/DOI for the supplement would help discovery, but is not
  required to replay this self-contained submission. Git has no raw data.
- Confirm author/title/ORCID, disclosure, rights and any required first-page
  notice. The supplied portal screenshot already marks copyright submitted;
  verify that it still applies to this final title/version.
- Run the venue's PDF checker if requested, upload to the camera-ready form,
  and separately complete Stage 3 program material. Verify both receipts.
  Registration and the talk date/length remain author/organizer matters.

## Rebuilding

In the extracted source bundle, from `paper/p3hpc` with an IEEE-compatible
TeX installation:

```sh
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The supplementary ZIP's README gives the independent CPU-only analysis command.
It checks all 252 final pairs and regenerates eight table/figure fragments.
The separate H100 pilot command checks its 90 earlier pairs and reproduces
the explicitly labeled preparation/search example, outside primary aggregates.
The separate legacy verifier still targets the original-submission dataset;
it is not the verifier for the final cohort. Independent extraction/rebuild,
ZIP integrity and lossless JSON-content checks are recorded in BUILD-INFO.txt.
