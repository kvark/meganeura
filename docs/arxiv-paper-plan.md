# Meganeura arXiv paper plan

Status: working plan for a strong technical preprint. Benchmark numbers in
this document are iteration results, not the frozen publication artifact.

## Working title

**Meganeura: How Close Can Portable Graphics APIs Get to Vendor-Native
Machine Learning?**

A more conventional alternative is **Meganeura: Portable GPU Training and
Inference through Graphics APIs**. Keep the question title if the evaluation
remains the center of the paper; use the declarative title if the system
architecture becomes the center.

## Verdict and positioning

This is a worthwhile systems paper even without claiming a universally faster
compiler or a novel ML algorithm. The research question is useful:

> How much of a mature vendor-native ML stack's performance can one recover
> with a single training-and-inference system built on portable graphics APIs,
> and what prevents closing the remaining gap?

The strongest version is an empirical systems paper with three contributions:

1. **A unified portable system.** One typed graph, autodiff implementation,
   compiler, kernel set, and runtime serves training and inference through
   Vulkan and Metal. This addresses the common split between training in a
   framework such as PyTorch and deployment through unrelated,
   platform-specific inference engines.
2. **A controlled performance-portability study.** Matched workloads,
   deterministic parameters, explicit arithmetic contracts, complete
   forward/loss/backward timing, distributed output samples, gradient checks,
   raw timing distributions, and several GPU classes make the comparison
   useful beyond this project.
3. **Engineering evidence about what worked and what did not.** Kernel
   archetypes, specialization, memory planning, and cooperative matrices are
   positive results. Direct Naga-IR authoring, equality saturation for the
   current small rewrite set, and the cooperative-matmul epilogue regression
   and repair are unusually informative negative or mixed results.

Equality saturation should not be sold as the main source of speed. Current
ablations show that the greedy implementation reaches essentially the same
graphs much faster. The e-graph remains useful research infrastructure and an
honest ablation.

## Draft abstract

Machine-learning training is concentrated in feature-rich vendor-native
stacks, while deployed inference is commonly rebuilt for a collection of
platform-specific runtimes. We ask how close a single portable implementation
can come to the performance of those stacks while supporting both training
and inference. We present Meganeura, a Rust system that lowers a typed static
graph and its automatically differentiated training graph to specialized GPU
programs executed through Vulkan and Metal. Its compiler combines graph
rewrites, kernel archetypes for pointwise, reduction, matrix, convolution, and
attention operations, target-aware specialization, and lifetime-based memory
planning. We evaluate five matched transformer, robotics, diffusion-style,
vision, and speech workloads against PyTorch on [devices to be frozen]. Our
primary results use a strict-f32 arithmetic contract; a separate accelerated
class permits documented reduced-input/f32-accumulate hardware paths. The
study reports both cases where portable execution approaches or exceeds the
reference and cases where training remains substantially behind, then
attributes the gaps to kernel coverage, launch structure, and portable API
constraints. We also report negative results from direct Naga-IR generation
and equality-saturation-based rewriting. The results show that a uniform
portable training-and-inference stack is practical and competitive on
selected workloads, while identifying the work required to make that claim
general.

Replace the bracketed device phrase and add two or three exact headline
numbers only after the revision and hardware matrix are frozen.

## Research questions

- **RQ1 — performance:** What fraction of PyTorch/CUDA performance does
  Meganeura achieve for full inference, minimal-shape latency, and
  forward-plus-loss-plus-backward?
- **RQ2 — portability:** How does the same Meganeura source revision behave on
  NVIDIA, AMD discrete, integrated/edge-class, and Apple GPUs?
- **RQ3 — precision:** How do strict f32 and documented accelerated arithmetic
  change speed and numerical agreement?
- **RQ4 — compiler contribution:** Which gains come from graph rewrites,
  kernel specialization, cooperative matrices, memory planning, and other
  compiler/runtime choices?
- **RQ5 — remaining gaps:** Which dispatches, missing specializations, API
  constraints, or launch patterns explain the distance to the native stack?
- **RQ6 — implementation complexity:** What did direct shader-IR generation
  cost, why was WGSL a better authoring/debugging boundary, and how much
  shader-family duplication remains?

## Suggested paper structure

1. **Introduction**
   - The training/deployment stack split.
   - Why Vulkan/Metal are attractive but difficult ML targets.
   - The research question, headline results, and contributions.
2. **Background and goals**
   - Portable graphics APIs versus CUDA/ROCm-style compute stacks.
   - Static graphs, autodiff, shader compilation, and arithmetic formats.
   - Explicit non-goals: universal framework coverage and universal wins.
3. **System design**
   - Graph and import path.
   - Autodiff and the shared inference/training pipeline.
   - Rewrites and scheduling.
   - Kernel archetypes and target specialization.
   - Runtime, memory planning, and Vulkan/Metal execution.
4. **Methodology**
   - Matched workloads and objectives.
   - Strict and accelerated precision contracts.
   - Timing boundaries, warmups, raw samples, and device controls.
   - Forward and backward correctness gates.
5. **Evaluation**
   - Main multi-device strict-f32 results.
   - Accelerated results, with invalid training cells omitted.
   - Portability and edge-class results.
   - Compile time, memory, and deployment footprint.
6. **Ablations and gap analysis**
   - Rewrites off versus greedy versus equality saturation.
   - Cooperative formats and kernel families.
   - The May cooperative-matmul-epilogue correctness regression.
   - Per-dispatch profiles for the largest remaining gaps.
7. **Lessons from failed approaches**
   - Direct Naga IR and the `Expression is not cached` invariant.
   - Why a WGSL parse/validation step was a debugging and normalization
     advantage, not an architectural requirement.
   - Other measured rejected optimizations.
8. **Related work**
   - ML compilers and portable runtimes.
   - LlamaWeb/WebGPU inference as the closest recent graphics-API study;
     distinguish Meganeura's shared training-and-inference stack.
   - Performance portability.
   - Equality-saturation-based tensor compilation.
   - Keep unpublished projects peripheral.
9. **Limitations and threats to validity**
10. **Conclusion**

## Figures and tables

- System pipeline: model/IR → rewrite → autodiff (training) → schedule/kernel
  archetype → target specialization → memory plan → Vulkan/Metal.
- Arithmetic-contract table for strict versus accelerated runs.
- Main result table with median and IQR; inference and training validity shown
  separately.
- Cross-device performance-portability plot, normalized to the native engine
  available on each device.
- One per-dispatch profile explaining the dominant NVIDIA gap.
- Optimizer ablation table showing graph size, optimizer time, and end-to-end
  GPU time.
- Deployment-footprint table comparing the stripped Meganeura runner with the
  explicitly defined PyTorch runtime dependency closure.
- Timeline/case study of the cooperative-matmul epilogue regression.
- Direct-Naga versus WGSL implementation-size table.

## Claims to make

- A single portable graphics-API implementation can be competitive with a
  vendor-native ML stack on selected, clearly identified workloads.
- The same compiler/runtime architecture supports both inference and
  backpropagation across multiple GPU vendors and device classes.
- Precision policy materially affects both performance and validity and must
  be reported as part of a benchmark.
- Portable execution still has large, explainable gaps; the paper identifies
  them rather than hiding them.
- For the current rewrite set, greedy rewriting is the practical default;
  equality saturation did not improve the selected runtime graph enough to
  justify its compile cost.

## Claims not to make

- Meganeura generally beats PyTorch, CUDA, cuDNN, or every platform runtime.
- Vulkan and Metal provide identical features or performance.
- `accelerated-f32` uses the same input format in both engines.
- The scaled conditioned diffusion U-Net is checkpoint-compatible Stable
  Diffusion 1.5 or a task-quality evaluation.
- The Whisper workload is a complete encoder-decoder transcription system.
- Historical Inferena rows are directly comparable to the repaired protocol.
- The current five workloads establish complete operator or model coverage.
- Edge deployment is proven until at least one credible edge-class target is
  rerun from the frozen revision.
- That portable graphics-API ML inference is itself novel; recent WebGPU work
  already establishes that space. Meganeura's distinction is the uniform
  training/inference system, broader workload mix, and backward evaluation.

## Evaluation matrix

The minimum convincing matrix is:

| Class | Suggested device | Why |
|---|---|---|
| NVIDIA desktop | RTX 5080 | Strong vendor-native baseline and current optimization target |
| AMD desktop | RX 7900 XT | Same Vulkan implementation on a different vendor |
| Integrated/edge-class | Radeon 890M or similar | Memory/launch behavior unlike a desktop GPU |
| Apple | M3 or newer | Same system architecture through Metal |
| Older/consumer | RTX 3050, if convenient | Shows behavior away from the flagship target |

An Android device would strengthen the word “edge.” If one is unavailable,
say “edge-class” and treat Android support as system capability rather than
evaluation evidence.

For every primary device:

- use the same Meganeura and Inferena commits;
- run strict f32 first;
- run accelerated mode as a separate table;
- retain 5 warmups and at least 20 measured samples;
- record raw samples, median, IQR, driver, OS, API, clocks/power policy, and
  device identifiers;
- validate forward and training independently;
- do not carry historical table cells into the paper.

## Current iteration evidence

An archived 2026-07-23 strict-f32 RTX 5080 run passed all five forward and
backward gates, but used the former convolution-only U-Net. A development run
of the replacement 10.93M-parameter conditioned diffusion U-Net passes both
strict and accelerated forward/backward gates. In the full pre-freeze
5-warmup/20-sample sweep, its strict inference is 2.152 ms versus PyTorch's
1.941 ms and its strict training is 8.569 ms versus 4.493 ms. Accelerated
Whisper now passes the repaired gradient gate at 3.207 ms inference and
20.292 ms training versus 1.847 ms and 6.918 ms. These dirty-tree results
establish viability; clean pinned reruns still determine the paper cells.

The historical SmolLM regression is also paper-worthy. On the current
protocol checkpoint, Meganeura moved from roughly 5.45/22.3 ms
inference/training to 9.37/58.0 ms at commit `09d14ca`, which disabled
cooperative promotion for matmuls with epilogues to fix invalid workgroup
geometry. Later work recovered part of the training loss, but not the safe
cooperative epilogue path. This pass added one: it stages accumulator tiles
through workgroup memory and evaluates the pointwise DAG with guarded scalar
lanes. It does not recover this historical gap, because the current packed
SwiGLU graph has no epilogue-bearing matmuls at the relevant sites. This is a
useful result in its own right—the missing historical performance cannot be
assigned to one still-active kernel site.

## Publication and arXiv logistics

- Recommended primary category: **cs.LG (Machine Learning)**. The paper is an
  ML-systems methodology and evaluation contribution, and this category gives
  it the most relevant readership. The 2026 WebGPU dispatch study uses the
  same primary category for a closely related systems/performance paper.
- Cross-list to **cs.PF (Performance)** because performance measurement and
  evaluation are the central empirical method.
- Cross-list to **cs.DC (Distributed, Parallel, and Cluster Computing)**
  because its scope explicitly includes parallel computation and it is the
  primary category of LlamaWeb. Do not add further cross-lists unless the
  manuscript changes materially; arXiv's
  [cross-list guidance](https://info.arxiv.org/help/cross.html) advises that
  more than one or two is rarely appropriate. Category descriptions are in
  the [current taxonomy](https://arxiv.org/category_taxonomy).
- Check endorsement before the final week.
- Submit TeX sources, figures, bibliography, and a reproducible public
  artifact link. arXiv is moderated but not peer reviewed.
- The same technical manuscript can later become a conference submission,
  subject to that venue's formatting, anonymity, prior-publication, and AI
  policies. arXiv does not need to wait for that process.

arXiv's current [submission guidelines](https://info.arxiv.org/help/submit/)
and [submittal agreement](https://info.arxiv.org/help/policies/submission_agreement.html)
do not state a conference-style prohibition on generative-AI assistance or a
separate mandatory AI-disclosure rule. They do require the submitter to
represent that the listed people are the original authors and that the work
meets accepted standards of scholarly communication. Keep a voluntary
disclosure for transparency, re-check the live policy during submission, and
treat any later venue's rules as a separate requirement. A suitable
disclosure is:

> Generative AI tools were used for code review, benchmark-harness
> development, and editorial assistance. The author designed the study,
> reviewed generated changes, executed the experiments, verified the reported
> claims and references, and accepts responsibility for the manuscript.

Adjust that sentence so it precisely describes the final workflow.

### Endorsement outreach

Start a draft submission with `cs.LG` selected before contacting anyone.
arXiv will then say whether endorsement is required and provide the request
link; follow its [endorsement instructions](https://info.arxiv.org/help/endorsement.html).
Prefer, in order:

1. someone the author already knows who publishes ML-systems work on arXiv;
2. an eligible author of a directly related paper, verified through arXiv's
   “Which authors of this paper are endorsers?” link;
3. a single carefully chosen cold contact, not a batch mailing.

Relevant people to consider, subject to the eligibility check, are Tyler
Sorensen or Reese Levine (LlamaWeb), Charlie Ruan or Ruihang Lai (WebLLM), and
Jędrzej Maczan (the 2026 WebGPU dispatch study, which uses
`cs.LG`/`cs.DC`/`cs.PF`). Tyler Sorensen is the strongest topical fit;
Jędrzej Maczan is the closest methodological fit but may not yet meet arXiv's
paper-count threshold. Authors of TinyIREE or the e-graph papers would be
valuable reviewers, but may be eligible in compiler categories rather than
the requested ML endorsement domain.

A concise request should include the generated endorsement link, the draft,
and three sentences explaining the question, artifact, and result. Ask the
recipient to endorse only if, after a skim, they consider the paper topical
and refereeable for `cs.LG`.

### Papers to use as storytelling precedents

- **[LlamaWeb](https://arxiv.org/abs/2605.20706)**: the closest portability
  narrative. Follow its pattern of
  naming three concrete system constraints, connecting each to one design
  response, and evaluating across a broad device matrix. Its early summary
  table is a good model for making coverage, memory, and performance legible.
- **[TVM](https://arxiv.org/abs/1802.04799)**: the clearest compiler-paper
  arc—hardware diversity, an end-to-end
  pipeline figure, graph- and operator-level design, then operator and
  end-to-end evaluation. Meganeura should emulate the structure without
  implying TVM's breadth.
- **[WebLLM](https://arxiv.org/abs/2412.15803)**: a compact seven-page
  example with the project link visible,
  one graphical architecture figure, and a narrow “fraction of native
  performance retained” message.
- **[TinyIREE](https://arxiv.org/abs/2205.14479)**: a strong precedent for
  treating binary/runtime size as a first-class result, with exact artifact
  definitions and explicit caveats.
- **[Characterizing WebGPU Dispatch Overhead](https://arxiv.org/abs/2604.02344)**:
  a model for separating API, framework, and kernel costs; reporting
  failed/inconclusive experiments; and treating dtype mismatches as threats
  rather than burying them.

## Freeze checklist

- [x] Diagnose the recent performance regression and implement the safe
      cooperative epilogue; record that the current graph does not exercise it.
- [x] Choose greedy rewriting as the production default from the ablation.
- [x] Replace the convolution-only diffusion proxy with a conditioned U-Net.
- [x] Repair accelerated Whisper backward precision and pass a development
      correctness run.
- [ ] Commit Meganeura and update Inferena to that exact revision.
- [ ] Tag the protocol and record both repository hashes in every artifact.
- [ ] Rerun strict and accelerated correctness after the final code change.
- [ ] Rerun the complete device matrix without changing source.
- [ ] Recheck Vulkan validation on the frozen driver/layer versions. The
      pinned Naga SPIR-V backend currently triggers
      `VUID-StandaloneSpirv-None-10684`, an open upstream wgpu/Naga issue that
      wgpu itself suppresses but Blade reports. Either consume a verified
      upstream fix or disclose the exact warning; do not suppress unrelated
      validation messages.
- [ ] Capture per-dispatch profiles for the largest two gaps.
- [ ] Add peak memory and reproduce the deployment-footprint measurement from
      the clean release artifacts.
- [ ] Archive raw JSON, environment manifests, plots, and analysis scripts.
- [ ] Make the artifact repository publicly resolvable before submission.
- [ ] Verify every citation and every numerical claim manually.
- [ ] Add the AI-assistance disclosure.
- [ ] Build the arXiv source bundle in a clean environment.

## Productization signals

The paper should make possible product directions legible without becoming a
product pitch:

- a Rust-native portable training/inference SDK for robotics and embedded
  applications;
- local fine-tuning plus inference without shipping separate desktop and edge
  runtimes;
- a compiler/runtime component for applications that already depend on
  Vulkan or Metal;
- engineering services or hosted tooling for profiling and specializing
  portable GPU graphs.

The most valuable commercial claim is reduced stack fragmentation, supported
by measured portability and artifact quality—not merely another inference
latency leaderboard.
