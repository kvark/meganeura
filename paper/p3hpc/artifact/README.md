# Meganeura P3HPC artifact

This research object supports the paper *Vulkan and Metal as a
Performance-Portability Layer for GPU Training and Inference*. It contains
the frozen measurements, the exact analysis used to generate the paper's
tables and aggregate values, and source snapshots for the two measured
systems. The analysis replay needs only Python 3 and does not need a GPU or
network access.

The extended companion report is
[arXiv:2608.01563](https://arxiv.org/abs/2608.01563).

The package has not yet been independently evaluated. Its permanent DOI is
pending the workshop artifact-freeze process.

## Quick verification

From the extracted top-level directory, run:

```sh
./verify.sh
```

or, on a system that does not preserve executable bits:

```sh
python3 verify.py
```

This command takes seconds. It checks every packaged file against
`MANIFEST.sha256`, parses and audits the raw records, regenerates all six
LaTeX table/figure fragments in a temporary directory, and compares them
byte-for-byte with `expected/tables/`. Add `--show-facts` to print the full
aggregate fact block used for the prose audit.

This is the recommended first evaluation tier. It verifies the paper from
the retained observations without pretending that hardware-dependent timing
can be recreated on an arbitrary machine.

## Contributions and supporting artifacts

The paper makes four principal contributions:

- **C1: graphics APIs as an ML performance-portability layer.** Supported by
  **A1**, the complete strict and accelerated five-machine result matrix;
  **A2**, the deterministic analysis; and **A3**, the frozen source.
- **C2: mechanism-resolved performance results.** Supported by the ordinary
  timings in **A1** and the per-dispatch timestamp sidecars in **A4**.
- **C3: independent correctness gates.** Supported by the output, loss,
  total-gradient, and per-parameter-gradient evidence in every **A1** record.
- **C4: validity-disciplined Pennycook aggregation.** Supported by **A1** and
  the aggregation implementation and expected output in **A2**, including the
  symmetric oracle-dispute exclusion and its cross-backend audit.

The computational artifacts are:

- **A1 — `results/`:** 50 device/workload/arithmetic cells. Each cell has a
  Meganeura record, a PyTorch record, and a joined summary: 150 JSON files in
  total. Ten SVGs retain the harness's per-device presentation output.
- **A2 — `mktables.py`, `verify.py`, and `expected/`:** standard-library-only
  analysis, structural checks, generated tables, and the fact block.
- **A3 — `source/`:** Git-archive snapshots of Meganeura and Inferena at the
  measured revisions. A separate Meganeura snapshot retains the revision used
  for the gap profiles.
- **A4 — `results/*/profiles/`:** five structured hardware-timestamp
  sidecars for NVIDIA ResNet-50 and Apple Whisper, including normal benchmark
  samples, instrumentation overhead, dispatch geometry, and family totals.
- **A5 — DinoVision deployment evidence:** the physical Quest 3S case study
  is larger and independently versioned. Source is frozen at
  [kvark/dinovision commit dc35cdf1](https://github.com/kvark/dinovision/tree/dc35cdf1c7c910cdd93c5b5362846842ae469a21),
  and weights, raw records, manifests, and media are frozen at
  [Hugging Face revision 2c3f9017](https://huggingface.co/mad-bot/dinovision/tree/2c3f9017fe74c41482b165890c14737a2ccd4b6a).
  A5 is not part of the performance-portability aggregate.

`BUILD_INFO.json` records the full revisions and package provenance.
`ENVIRONMENTS.json` is a mechanically extracted summary; the unabridged
metadata remains in every raw JSON file.

## Frozen revisions

- Meganeura matrix: `7561a64ec5a7e4bcdcd2c719aaaffe5912ed5e85`
  (`paper-arxiv-1`)
- Inferena harness: `7ca9c5c7b2cd614343a3de3dcc86999ced66e8c0`
  (`paper-arxiv-1`)
- Meganeura gap profiles: `b1405a3a52fabf9858aca5cbd80e246811cb6a58`

The public repositories are
[Meganeura](https://github.com/kvark/meganeura) and
[Inferena](https://github.com/kvark/inferena). The source snapshots make
inspection possible without the network; Git clones are preferable for a
full rerun because they preserve revision metadata.

## Machine and software matrix

| Artifact directory | Measured path | Python and reference | Meganeura driver | Warmups / samples |
|---|---|---|---|---:|
| `nvidia` | RTX 5070, CUDA | Python 3.14.4; PyTorch 2.13.0+cu130, CUDA 13.0 | Vulkan, NVIDIA 595.71.05 | 5 / 20 |
| `amd-d` | RX 7900 XT, ROCm | Python 3.12.13; PyTorch 2.10.0+rocm7.1, ROCm 7.1.25424 | Vulkan, RADV Mesa 26.0.3 | 5 / 20 |
| `amd-i` | Radeon 780M, ROCm | Python 3.12.3; PyTorch 2.12.0+rocm7.14.0, ROCm 7.14.60850 | Vulkan, RADV Mesa 25.2.8 | 5 / 20 |
| `intel` | RPL-U iGPU versus CPU fallback | Python 3.13.7; PyTorch 2.11.0+xpu executing on CPU | Vulkan, ANV Mesa 26.0.3 | 4 / 20 |
| `mac` | Apple M3, MPS | Python 3.13.3; PyTorch 2.11.0 on MPS | Metal, macOS 15.7.3 | 5 / 20 |

The Intel collection records four untimed warmups rather than five. All
three timed series still retain 20 samples. The deviation is disclosed here
and in the workshop manuscript; it is not normalized after collection.

The reference package channels differ because the newest usable vendor wheel
differed by machine. Most importantly, the discrete and integrated AMD
machines intentionally use different ROCm channels. The tagged Inferena
requirements files are installation starting points, not complete lockfiles;
the original experiment did not retain a full transitive `pip freeze` for
every host. Exact Python, PyTorch, GPU runtime, driver, arithmetic, and
protocol metadata are retained. This limitation prevents a claim of
bit-identical environment reconstruction.

## Protocol

There are five workloads, five machines, and two arithmetic contracts:

- `paper-v1-strict`: persistent data, accumulation, and output are f32;
  PyTorch TF32 and Meganeura f16 cooperative-input paths are disabled.
- `paper-v1-accelerated`: PyTorch may use TF32 and Meganeura may use eligible
  f16-input/f32-accumulate forward paths. These permissions are not treated
  as numerically equivalent to each other or merged with strict results.

Each retained cell contains full/prefill inference, a minimal-shape latency
series, and forward-loss-backward timing without an optimizer update. The
LLM minimal shape is a stateless one-token forward without a KV cache and is
not decode latency. Inputs are deterministic. The analyzer reports medians
and applies the paper's independent forward and backward validity gates
before performance-portability aggregation.

## Full experiment rerun

A rerun requires one of the measured GPU/driver classes or a new machine on
which the comparison will constitute an extension rather than an exact
reproduction. Start from the public frozen tags:

```sh
git clone --branch paper-arxiv-1 https://github.com/kvark/meganeura.git
git clone --branch paper-arxiv-1 https://github.com/kvark/inferena.git
cd inferena
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-<nvidia|amd|intel|apple>.txt
./run.sh --check
```

Before timing, compare `python --version`, `torch.__version__`, the reported
GPU runtime, and the graphics driver with the matrix above and
`ENVIRONMENTS.json`. Package indexes move, so explicitly install the recorded
PyTorch build if the requirements file resolves another version. For the RX
7900 XT, use the recorded ROCm 7.1 PyTorch channel rather than the newer 7.14
pin needed by the 780M.

Run only the two evaluated engines, first under the practical permission and
then under the strict control:

```sh
./run.sh -f pytorch,meganeura \
  --warmup-runs 5 --measurement-runs 20 \
  --results-dir results/paper-v1-accelerated

./run.sh -f pytorch,meganeura --strict \
  --warmup-runs 5 --measurement-runs 20 \
  --results-dir results/paper-v1-strict
```

Use `--warmup-runs 4` only when recreating the recorded Intel protocol.
`--profile --profile-samples 5` requests structured per-dispatch timestamps;
the supplied gap profiles were captured at the separately disclosed
Meganeura profile revision. No RenderDoc capture is required.

One strict-plus-accelerated sweep took less than an hour per original
machine, dominated by `torch.compile`. Hardware access and moving vendor
package channels are the expensive parts; analysis replay is intentionally
separate and fast.

## Known limitations

- One physical machine represents each vendor/class except AMD; the artifact
  does not establish vendor-wide performance.
- Timings are 20 samples from one process-level collection per cell, not
  independent process replicates.
- The Apple and Intel references are eager, and Intel falls back to CPU.
- No manual CUDA Graph capture was added to PyTorch.
- The PyTorch/ROCm Whisper backward record on the 780M fails the
  cross-backend oracle-consistency check in both arithmetic contracts. Raw
  timings and diagnostics are retained, but that paired training comparison
  is excluded symmetrically from ratios and aggregation; its valid forward
  timings remain included. The records identify an unusable local oracle but
  do not establish whether the root cause lies in PyTorch, ROCm, or the driver.
- The gap profiles predate the matrix freeze by one Meganeura revision. The
  NVIDIA control median matches the frozen result; the Apple profile is
  explicitly a pre-optimization diagnostic.
- Full transitive Python environment freezes were not retained.

## License, citation, and contact

The packaged code, measurements, and documentation are distributed under
the MIT license in `LICENSE`. Citation metadata in `CITATION.cff` includes the
public companion-report identifier.

Questions: Dzmitry Malyshau, `kvark@fastmail.com`, ORCID
`0009-0005-6410-4276`.
