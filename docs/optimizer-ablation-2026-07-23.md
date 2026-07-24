# Graph-rewrite ablation — 2026-07-23

## Question

Does equality saturation select better executable graphs than a deterministic
greedy application of the same current rewrite set, and does that difference
improve end-to-end GPU performance?

## Modes

- `off`: no forward rewrite pass;
- `greedy`: deterministic repeated application of the same fusion rules;
- `egglog-windowed`: equality saturation in fixed-size graph segments;
- `egglog-outlined`: repeated-layer outlining plus equality saturation;
- `egglog-whole`: one whole-graph equality-saturation problem.

The e-graph modes use a tensor-traffic-aware extraction cost unless noted.
AST-size and tensor-traffic extraction were also compared.

## CPU optimizer results

Medians below are from three repetitions. “Nodes” is
source/final-active-node count; training includes autodiff and full-graph
optimization.

| Workload/phase | Off (ms; nodes) | Greedy (ms; nodes) | Windowed (ms; nodes) | Outlined (ms; nodes) |
|---|---:|---:|---:|---:|
| SmolLM inference | 0.073; 726/726 | 0.089; 726/666 | 32.604; 726/666 | 2.944; 726/666 |
| SmolLM training | 25.763; 728/1649 | 25.559; 728/1468 | 1241.976; 728/1474 | 1099.779; 728/1469 |
| SmolVLA inference | 0.044; 381/381 | 0.055; 381/349 | 17.637; 381/349 | 4.685; 381/349 |
| SmolVLA training | 0.223; 385/823 | 0.236; 385/743 | 460.019; 385/762 | 580.132; 385/742 |
| Former convolution-only U-Net inference | 0.023; 119/119 | 0.016; 119/119 | 4.834; 119/119 | 3.994; 119/119 |
| Former convolution-only U-Net training | 0.085; 124/272 | 0.109; 124/272 | 47.573; 124/271 | 46.535; 124/271 |
| ResNet inference | 0.049; 284/284 | 0.034; 284/284 | 11.236; 284/284 | 12.329; 284/284 |
| ResNet training | 58.757; 286/839 | 61.634; 286/839 | 290.878; 286/839 | 475.344; 286/839 |
| Whisper inference | 0.023; 147/147 | 0.023; 147/147 | 5.791; 147/147 | 5.885; 147/147 |
| Whisper training | 68.905; 149/364 | 67.489; 149/356 | 153.696; 149/355 | 228.899; 149/355 |

Windowed extraction reported three failed segments for SmolLM training and ten
for SmolVLA training. In the standalone benchmark those failures fall back to
the original segment; in an abort-on-panic build the same mode is not robust
enough for production.

Whole-graph medians for representative cases were:

- SmolLM inference: 56.24 ms, 666 active nodes;
- SmolLM training: 7431.81 ms, 1486 active nodes;
- SmolVLA inference: 20.19 ms, 349 active nodes;
- SmolVLA training: 1092.89 ms, 742 active nodes.

AST-size and tensor-traffic extraction selected effectively the same graphs
for the current rule set. This does not show that traffic-aware costs are
useless; it shows that the available alternatives do not yet force meaningful
cost tradeoffs.

The diffusion rows above describe the workload that existed when this
ablation was recorded. They are retained as rewrite-engine evidence, not as
results for the newer conditioned diffusion U-Net.

## End-to-end RTX 5080 results

Each entry uses five warmups and 20 retained strict-f32 samples.

### SmolLM2-135M

| Mode | Compile (s) | Inference median (ms) | Inference IQR | Latency (ms) | Training (ms) |
|---|---:|---:|---:|---:|---:|
| Off | 0.864 | 11.208 | 11.178–11.249 | 2.481 | 38.733 |
| Greedy | 0.945 | 11.033 | 11.014–11.083 | 2.308 | 41.126 |
| Egglog outlined | 2.028 | 11.016 | 10.965–11.072 | 2.332 | 41.027 |
| Egglog whole | 8.479 | 10.990 | 10.768–11.044 | 2.265 | 41.021 |

Relative to off, greedy improves inference by about 1.6% and minimal latency
by about 7.0%, but makes training about 6.2% slower. Equality saturation is
within run-to-run variation of greedy while increasing compile time.

### SmolVLA

| Mode | Compile (s) | Inference median (ms) | Inference IQR | Latency (ms) | Training (ms) |
|---|---:|---:|---:|---:|---:|
| Off | 0.697 | 3.521 | 3.416–3.625 | 1.566 | 10.567 |
| Greedy | 0.777 | 3.465 | 3.273–3.774 | 1.478 | 10.626 |
| Egglog outlined | 1.373 | 3.593 | 3.441–3.768 | 1.477 | 10.981 |

Greedy improves inference by about 1.6% and latency by about 5.6% versus off,
with essentially unchanged training. Outlined egglog has no demonstrated
runtime advantage over greedy.

## Interpretation

The current evidence rejects a strong claim that equality saturation drives
Meganeura's performance. The more precise result is:

- the rewrite rules themselves provide small inference/latency wins on the
  transformer workloads;
- a simple greedy pass finds the same useful forms at negligible CPU cost;
- equality saturation adds substantial compile cost and no material GPU
  benefit for this rule set;
- some training graphs become slower after the selected fusions;
- outlining makes equality saturation much more tractable than whole-graph
  extraction, but not yet preferable to greedy.

This is still a useful paper result. It constrains when e-graphs are warranted
in tensor compilers and prevents attributing kernel/runtime performance to the
wrong subsystem.

## Recommended action

Use greedy rewriting for the paper's production baseline unless a new rewrite
introduces a real global choice that changes the ablation. Keep the e-graph
modes as experimental infrastructure and report them in the ablation section.

Before final publication:

- rerun the GPU comparison after the optimizer default is frozen;
- add a no-rewrite/greedy breakdown for at least one convolutional workload;
- report dispatch counts and chosen rule counts next to runtime;
- if retaining traffic-aware extraction as a contribution, add a constructed
  or real case where AST and traffic costs select different executable graphs.

Raw data:

- `results/optimizer-ablation-2026-07-23/core-v2.jsonl`
- `results/optimizer-ablation-2026-07-23/cost-v2.jsonl`
- `results/optimizer-ablation-2026-07-23/whole.jsonl`
- `results/optimizer-e2e-stable-2026-07-23/`
