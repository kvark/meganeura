# Measured graph and dispatch search

Source-only experiment, 20 September 2026. Production PR #200 is separate.
This prototype uses the existing egglog rules, extractor and graph stamper.
It does not replace the production tuner or change the submitted paper cohort.

## Architecture

Keep equivalent graph families until their implementations have been tuned:

1. Saturate a bounded region of the original graph. Extract its escaping roots
   together, preserving shared work and the actual external bindings.
2. Lower several representatives with their legal physical alternatives.
   Split-K includes partial storage, the final reduction and surrounding work.
3. Initialize, qualify and kernel-tune each complete candidate before comparing
   whole-program execution against the incumbent.
4. Retain the measured winner, then confirm it using independent samples.

Egglog's root-only `extract_variants` chooses cheapest children and misses nested
alternatives. Excluding selected e-nodes exposes those alternatives through the
existing extractor, without implementing another rewrite engine. Whole-constructor
exclusions precede individual sites so a small bound reaches different families.
Estimated traffic orders exploration; it does not choose the measured winner.

The helper explores eight representatives of one verified repeated region.
Dispatch fusion and scalar split-K produce 18 distinct plans; adding three
serial-reduction widths produces 45. Model split-K currently uses eight
partitions with a 64-wide output tile and K stage eight. The small matrix probe
also explores other partition counts and tiles. These are bounded search spaces,
not exhaustive optimization. Other physical choices may not be represented.

Alternatives are lowered before allocation and scheduling. The current public
`select` prototype accepts complete plans and caller initialization/qualification
callbacks. The follow-up below adds pre-allocation cached-attention choices and
private persistent-state restoration. The public live-session tuner remains
unchanged; migrating that API is still required before removing its installer.

## Isolation, reuse and validation

Only an incumbent and one challenger session coexist. Inputs, outputs and
intermediates remain private. Matching immutable weights can share allocations
through the existing `Session::share_parameter_from` API. Derived weights are
shared only when the complete parameter interface agrees, including tensor type,
storage format, source names and transform. Otherwise their dependent sources
take the normal initialization path. The caller must not mutate the incumbent.

Completed kernel searches are reused within one selector, keyed by geometry,
placement, precision, code-generation knobs and the initial/challenger sequence.
There is no persistent/global tuning cache. Failed, incomplete or unqualified
searches do not populate the memo; reports distinguish reuse from fresh samples.
A private-kernel winner is still an approximation: whole-program interactions
can change which local choice would be best.

Every complete model is checked before tuning, after tuning and after sampling.
All outputs must be finite and meet Inferena's unchanged forward gates:
relative L2 and relative squared-norm error below 1%. Independent full CPU
references are used in the model study, not a sample of tensor elements. Those
measurements used the immutable-only selector at the revision below. No numerical
threshold was relaxed.

Whole-step samples include fresh recording, submission and waiting, but exclude
output readback. Existing alternating pairs, a 2% improvement threshold and
the paired noise guard decide selections. Confirmation uses 30 warmup pairs and
40 held-out pairs. Search results are not themselves unbiased benchmark samples.

The deadline is soft: in-flight driver work and validation cannot be preempted.
There is a 64-program limit and a conservative 3-GiB sum of declared logical
plan bytes for the two sessions, not a driver-heap bound. Runs additionally use
a 4-GiB host-memory cgroup, no swap and a 240-second external timeout.
Truncation and phase costs are recorded; `extraction_truncated` is separate
from finishing all enumerated plans.

## Workload and reproduction

Measured implementation:
`95d55651f411ae68e2b19a59d1082562d5c4a2bf`.
Blade:
`2b328f8b643798813d8c9319b807030215d33b98`.
Inferena model/reference code:
`fa5a04e1c1b38405cfa371a27c5dcef1319835d5`.
CPU PyTorch 2.13.0:
`cf30153c4c131c8164ee7798e5022d810682e2cb`.

This is Inferena's deterministic SmolVLA action expert (50 action tokens,
16 context tokens), not a pretrained full vision-language policy. Whisper
studies use its encoder with 3000 mel frames, not full transcription.
CPU reference SHA-256:

- SmolVLA: `4e830f6106c65b9150e4c873b37de81b57fa3ec469e0ea866200ab0cfdc16295`.
- Whisper: `c85fa1f231c4cb966cd364a45bd1b2754b8c8641fea5fad22b3201554fe93d04`.

Use the pinned Inferena Python environment for the reference helper. Use a
separate Cargo target directory for each worktree; sharing own-crate build
artifacts between checkouts can produce a stale experimental binary.

```sh
python bench/egglog_reference.py ../inferena SmolVLA /tmp/smolvla.f32
CARGO_BUILD_JOBS=1 cargo build --release --features models --example egglog_model_search
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  taskset -c 0,2,4,6,8,10 target/release/examples/egglog_model_search \
  SmolVLA /tmp/smolvla.f32 fast --serial-sums \
  --warmup=32 --seconds=120 --confirm
# Repeat with --reverse; only challengers are reversed.
# For B570: intel_icd.json and strict instead of fast.
# For a shorter search: --seconds=20. For warmup ablation: --warmup=8.
```

The original graph is the default; `--optimized-graph` retains the earlier
diagnostic. Before `52a9f25`, the helper required `--original-graph` instead.
`--static` disables inner kernel tuning only. `--profile` adds separate
instrumented pass samples, not headline latency. Later `--program=N` selects
one fully expanded plan for attribution; older revisions applied it before
serial-reduction expansion.

Machine: i5-12400F, six physical cores, unlocked CPU/GPU clocks; RTX 5070,
NVIDIA 595.91.07; Arc B570, Mesa 26.0.3 on the secondary PCIe x1 link.
GPUs and builds run sequentially. NVIDIA uses the accelerated policy and
Intel native F32; each comparison uses the same policy for both arms.
These are warm-cache process starts, not cold driver-compilation measurements.

## SmolVLA results

Default 32 warmup pairs, complete 45-plan search. Both control and candidates
include the single-subgroup-writer correction described in the kernel report.
Times are held-out medians in milliseconds; reduction is the median paired
reduction.

| GPU / search order | Greedy, freshly tuned | Selected | Reduction | Search cost |
| --- | ---: | ---: | ---: | ---: |
| RTX 5070 / forward | 4.475 | 3.633 | 18.9% | 33.2 s |
| RTX 5070 / reverse | 4.501 | 3.654 | 18.8% | 32.3 s |
| B570 / forward | 9.800 | 6.133 | 37.7% | 42.6 s |
| B570 / reverse | 9.822 | 6.125 | 37.6% | 47.9 s |

All four selected plans win all 40 held-out pairs. Full CPU relative L2 stays
below 5.6e-6. Selection reaches 99 split products plus serial row reductions.
Forward order chooses width 64, reverse width 256; the noise guard does not
establish a universal winning width.

### Budget and warmup ablations

The following cost ablations use `3d65bb05c8e497a527f6d77cf9592f68f39065b4`,
before the store-only correction. That revision's full searches give roughly
18%/37.5% reductions, with the same selected graph families and numerical errors.
The table above repeats the main comparison after the correction; these
historical ablations are not presented as additional final-revision samples.

With a 20-second bound and the same warmup policy, NVIDIA visits 27/26 plans
and retains 18.2%/13.7% reductions; Intel visits 13/12 and retains 32.1%/23.8%.
Each wins all 40 confirmation pairs. Order sensitivity remains a limitation,
even though bounded search still improves the incumbent.

Eight warmup pairs reduce full-search cost to 23.4/23.9 seconds on NVIDIA
and 27.2/32.5 seconds on Intel, retaining roughly 18%/37.5% improvements.
This is a policy ablation, not a new default. One short NVIDIA run changes
speed substantially within confirmation despite identical selected kernels;
its unusually large 28.5% median reduction is not used as the headline result.
Unlocked clocks and within-run drift matter; longer warmup does not constitute
a clock-control guarantee.

Four further fresh-process NVIDIA runs of the short eight-warmup search,
two in each order, give 17.95–18.25% reductions and win all 40 held-out pairs
each. Their control medians are 4.437–4.467 ms and selected medians
3.629–3.653 ms. These repeat the ordinary result, not the drifting outlier.

## Search cost and a greedy counterexample

The initial 45-plan implementation at `469071b` takes 36–38 seconds on NVIDIA
and 84–87 seconds on Intel. Separating kernel warmup from whole-program warmup
at `c2c79cc` barely changes those costs. Intel's dominant phase is candidate
initialization, about 43 seconds, not kernel warmup. Complete-interface sharing
cuts that phase to 2–6 seconds; NVIDIA initialization falls from about five
seconds to 1–2 seconds. No input/output validation is removed.

Kernel memoization is independently useful. On the earlier 18-plan search,
`43deac7` versus `df3bce4` reduces complete search from 35–36 to 12–14 seconds
on NVIDIA and 57 to 32–33 seconds on Intel. It removes repeated private-scratch
comparisons, not whole-program qualification or timing. Those earlier costs
use a different warmup policy and must not be compared directly with 45 plans.

The small `egglog_search` probe at `d508b1a` demonstrates the original concern:
for B570 M=128, K=576, N=1536, the unfused default takes 0.215 ms versus
0.220 ms fused, but tuning produces 0.204 ms unfused versus 0.200 ms fused.
Both initial candidate orders agree. The ~2% final margin is small; the point
is that selecting a graph family before tuning can discard the winner.
The full-F64 probe also covers other shapes, transpose directions, ragged
edges and uneven K partitions.

## What to carry forward

Keep the existing egglog equivalence engine and common measurement machinery.
Make graph/lowering alternatives a pre-allocation build-stage choice, rather
than adding another operation-specific live-plan patcher. Retain the greedy
incumbent, a bounded budget, legal numerical policies and explicit truncation.

Production integration still needs a representative-data/qualification contract
at the build API, runtime optimizer support, and removal of the old attention
installer after its callers migrate. Explicit persistent writes are now supported
by the experimental selector, as described below.
Broader physical choices, better bounded exploration order and reuse of compiled
pipelines remain opportunities. Do not infer global optimality or PyTorch
parity from these results.

See [subgroup-attention.md](subgroup-attention.md) for the Nsight attribution,
Whisper attention result and negative cooperative-kernel trials. Raw results,
traces and binaries remain outside Git; source revisions preserve reproduction.

## Stateful attention follow-up

Implementation and measurement helper: `84f65f2` (same Blade revision as above).
This is a cached-attention region with input/output scaling and KV writes, not
a whole-model or paper-cohort result. It does not change the model timings above.

`ExecutionPlan::set_attention_splits` lowers one or two dispatches before a
session exists. The ordinary scheduler and allocator then handle barriers and
scratch lifetimes. Legality checks are shared with the existing tuner. The inner
kernel search cannot change the structural alternative being compared.

The common selector snapshots inputs, parameters and constants that the plan
writes, then restores them before each validation, warmup and timed trial. It
returns the selected session with its initial persistent state. Immutable weights
can still be shared; sharing writable parameters is rejected. Runtime-added SGD,
Adam and accumulation must be configured after search: their additional buffers
and counters are not part of this snapshot contract. Updates explicitly encoded
in the execution plan are supported. A failed incumbent invalidates the search;
it cannot silently survive as the fallback.

The API refinement at `a8edb56` makes qualification read-only: the selector owns the
step and wait, then gives the checker an immutable session. A checker cannot
accidentally advance the cache again, re-upload weights or change the optimizer
through the ordinary session API. Broad workflow checks cover cache mutation,
compiled forward/backward/SGD, rejected sharing and invalid-incumbent handling
inside the existing library test binary; no additional test executables are used.
All 468 all-feature library tests, including the opt-in GPU checks, pass in both
the NVIDIA-selected and Intel-selected runs. All-target/all-feature Clippy passes.
This is local prototype verification, not additional coverage claimed for PR #200.

Snapshots count towards the logical memory bound. Reports separate snapshot
bytes and capture/restore time from qualification and execution. Restoration is
outside the execution timer but affects cache warmth. The following held-out
comparison instead repeats an idempotent prefix write without restoration in
either arm, to check ordinary steady-state execution.

All five split choices are checked against every element of an independent f64
output, plus exact complete K/V contents. Geometry: 12 query heads, four KV heads,
head dimension 64, capacity 2048, F32, full-cache position. Search uses 16 warmup
pairs and 12 selection pairs; confirmation uses 30 warmup pairs and 40 new pairs.
The control is the existing position-balanced attention tuner, not an untuned
kernel. Both GPUs and both challenger orders complete all five plans.

| GPU / queries | Existing tuner, ms | Pre-allocated search, ms | Selected splits |
| --- | ---: | ---: | --- |
| RTX 5070 / 1 | 0.077 / 0.077 | 0.077 / 0.076 | 16 / 16 |
| RTX 5070 / 16 | 0.266 / 0.265 | 0.265 / 0.265 | 16 / 16 |
| B570 / 1 | 0.172 / 0.194 | 0.172 / 0.193 | 16 / 16 |
| B570 / 16 | 0.884 / 0.885 | 0.884 / 0.869 | 4 / 16 |

Pairs are forward/reverse challenger order. This supports performance parity,
not a general speedup. Intel's reverse-order 16-query result is about 1.9% faster,
near the 2% selection threshold; the control picks four splits in both orders.
Unlocked-clock drift also changes absolute Intel decode times across processes.

Search takes 1.42–1.83 seconds on NVIDIA and 0.44–0.67 on Intel; total state copying
takes 9–11 and 38–46 ms respectively, with a 4-MiB snapshot per session. These are
not comparable cold-start compilation costs: external CPU-reference construction
is excluded, driver caches are warm, and the legacy tuner excludes session build.

```sh
CARGO_BUILD_JOBS=1 cargo build --release --example egglog_search
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  taskset -c 0,2,4,6,8,10 target/release/examples/egglog_search attention 16 2048 forward
# Repeat with reverse; replace 16 with 1 for decode, and nvidia with intel.
```

This removes the need for a live patcher on the experimental path. It does not
yet remove the production patcher or automatically choose representative inputs.

A forward-order SmolVLA regression check at `a8edb56`, using the same 45 plans
and validation gates, takes 33.3 / 42.9 seconds on NVIDIA / Intel. Held-out
medians are 4.465→3.627 ms and 9.679→5.941 ms (18.8% / 38.5% paired reductions,
40/40 wins each). All candidates qualify; relative L2 remains 5.55e-6 / 5.42e-6.
These single repeats confirm the earlier result, not an additional optimization
gain from state handling or a replacement for the two-order table above.
