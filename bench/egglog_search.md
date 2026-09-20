# Measured extraction experiment

This branch keeps graph alternatives until after measuring their kernel
configurations. It reuses Meganeura's egglog rules, extractor and graph stamper.
It is not a production replacement for the post-plan tuner yet.

## What changed

Egglog's `extract_variants` varies only the root constructor and chooses the
cheapest children. The experiment asks the existing extractor for more
representatives by excluding selected e-nodes. This exposes nested choices
without implementing another rewriting engine. A bounded queue controls the
work; `truncated` means alternatives remain unexplored. This is neither a
globally optimal search nor a k-best guarantee.

Several escaping roots are extracted together, so a candidate contains their
shared computation. Small regions can have opaque dependencies outside them;
the existing outliner identifies repeated model regions. Estimated traffic
only orders exploration. It is not a substitute for timing the complete
lowered candidate, including its barriers, temporary storage and kernels.

## Joint graph and kernel probe

```sh
cargo test --lib optimize::search::
cargo build --release --example egglog_search
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/intel_icd.json \
  target/release/examples/egglog_search 128 576 1536 forward
# Repeat with reverse, and with nvidia_icd.json.
```

The graph is `Add(MatMul(A,B), C)`. Both fused/unfused representations cross
K stages 8/16/32 and both column layouts: 12 complete candidates, all compiled
before timing. Later graph optimization and dispatch fusion are disabled so
they cannot collapse the alternatives. Precision is native f32; neither test
GPU exposes a native-f32 cooperative tile through this stack.

Each candidate is checked against an independent f64 matrix product over all
output elements. Inputs are bounded, nonzero synthetic dyadic values, not
trained model weights. This example is not comprehensive numerical
qualification. Timing uses fresh ordinary step/submit/wait execution, 20
repetitions per sample, three discarded and nine retained samples. Candidate
order rotates and reverses within a run. No command recording is replayed.

At implementation revision `d508b1a`, B570 results for M=128, K=576, N=1536
(milliseconds, two fresh processes with opposite initial order):

| Representation | Default K=32, plain columns | Best measured configuration |
| --- | ---: | ---: |
| Unfused, forward order | 0.21529 | 0.20415 |
| Fused, forward order | 0.22038 | 0.20041 |
| Unfused, reverse order | 0.21414 | 0.20366 |
| Fused, reverse order | 0.22047 | 0.19999 |

The unfused default wins, but the tuned fused form wins in both orders. Its
final margin is small, about 2%, not evidence of a large whole-model gain.
Both winners use K=16; the best fused column layout differs between runs.
Choosing a graph family from its default timing would discard that winner.
Extraction took 2.1–2.2 ms; all 12 builds together took 135–137 ms with warm
driver caches. These are not cold-start compiler comparisons.

The same probe also ran 50×512×512, 50×720×960 and 1×576×1536 on B570 and RTX
5070. The one-row case lowers to GEMV, so changing the matrix K-stage does not
create distinct kernels there and timing differences are noise, not a win.
Hardware: i5-12400F, affinity 0,2,4,6,8,10, unlocked clocks; NVIDIA 595.91.07,
Intel Mesa 26.0.3 with B570 on the secondary x1 link. Raw artifacts stay outside
Git; the branch retains the code and commands needed to repeat the experiment.

## Model-region survey and remaining work

```sh
cargo build --release --features models --example egglog_model_search
target/release/examples/egglog_model_search SmolVLA
target/release/examples/egglog_model_search Whisper-tiny
```

The CPU-only survey found eight bounded alternatives in a 45-node SmolVLA
region (seven repeated instances) in 6.9 ms. A 32-node Whisper region (three
instances) produced one representative in 4.0 ms. The latter is important:
many useful choices, including generated epilogues, are currently introduced
by lowering and are absent from the logical rewrite rules.

Before production integration:

- Retain legal lowering/fusion families as well as logical rewrites. Tune each
  complete candidate before selecting a representation.
- Preserve the region boundary's actual placement, precision and observable
  outputs. Turning every cut dependency into a host-visible input would time
  a different memory contract.
- Select before final allocation and scheduling. This avoids growing live
  buffers and repairing barrier groups for each operation-specific tuner.
- Reuse qualification, scratch limits, paired sampling and noise guards;
  record exclusions and unfinished searches. Do not replace them with a sum
  of independently measured kernel times.

No whole-model timing improvement or automatic production search is claimed.

## Split-K as a structural candidate

`egglog_search 50 720 960 forward split` also crosses 32/64 output tiles with
1/2/4/8 K partitions, before final scheduling and allocation. Only the plain
matrix product currently supports partitioning. Its candidate includes SumRows
and the original add, so a fused one-dispatch form competes with complete
multi-dispatch forms, not with a partial kernel time. The implementation reuses
the scalar generator and reduction; it does not duplicate a matmul shader.

This is motivated by a validated Nsight Systems capture of the actual SmolVLA
action-expert workload: PyTorch uses many split-K GEMMs and combine kernels.
The prototype changes neither the benchmark protocol nor production defaults.
The source branch includes an opt-in broad test of normal/AT/BT products,
uneven K partitions, ragged output edges and rejection without plan mutation.
It passes on both RTX 5070 and B570 against full independent f64 products.
