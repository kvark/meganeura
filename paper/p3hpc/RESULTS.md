# Final P3HPC cohort: evidence guide

Updated September 14, 2026. The manuscript now uses the completed v9 cohort,
not the superseded v7 tables. No new timing was collected, condition changed,
or outlier removed during analysis.

## Identity and health

All nine archives share Inferena fa5a04e1c1b38405cfa371a27c5dcef1319835d5,
Meganeura 428fc2d2322229e5338f5d80a10d700340d593cd, Blade
f6f2729e850cc0aefdc0bb18523da58a72765169, Python 3.13.13 and
PyTorch 2.13.0 at cf30153c4c131c8164ee7798e5022d810682e2cb.
Vendor wheels/libraries differ. Cargo.lock and shared checkpoint hashes agree
after normalizing Windows separators.
The annotated Meganeura tag
[paper-p3hpc-2026](https://github.com/kvark/meganeura/tree/paper-p3hpc-2026)
preserves the measured engine revision, separately from the manuscript branch.
[cohort.sha256](artifact/cohort.sha256) identifies the original archives.

| Configuration | Valid / selected pairs | Reference |
|---|---:|---|
| RTX 5070, Linux | 30 / 30 | Default compilation + CUDA Graph |
| H100 80GB, Linux | 30 / 30 | Default compilation + CUDA Graph |
| RTX 3050, Windows | 30 / 30 | Default compilation + CUDA Graph |
| RX 7900 XT | 30 / 30 | Default compilation + HIP graph |
| Radeon 780M | 30 / 30 | Same, with recorded ROCm overrides |
| Arc B570 | 30 / 30 | Default compilation + XPU graph; math SDPA and embedding workaround |
| Apple M3, macOS | 30 / 30 | Compiled MPS; no equivalent public whole-phase replay API |
| Intel RPL-U | 30 / 30 | Vulkan versus explicitly selected compiled CPU |
| H100 360M/1.7B extension | 12 / 12 | Both contracts, three processes each, CUDA Graph |

Eight main campaigns contribute 240 pairs; the extension adds 12. There are
no interrupted campaigns or failed pairs. Seven configurations have GPU
references; RPL-U stays separate. RTX 5070 and B570 share a host and were
measured sequentially; B570 uses the secondary PCIe 3.0 x1 link.

Every pair passes the frozen Inferena checker and an independent audit of
raw/joined agreement, timing medians, numerical errors, executed policies,
full replay statistics and replication. All nine replication reports agree
with the retained errors. Every pair individually meets the 5% gradient
bounds: maxima are 0.6281% sampled-output L2, 0.0861% scalar loss, 2.9419%
total-gradient norm, and 3.2983% parameter-norm-vector L2.

Health does not imply low timing variance. M3 strict native SmolVLA training
ranges 52.837–109.035 ms around a 70.404 ms median (79.8% range/median).
H100 native strict ResNet inference ranges 4.259–6.554 ms around 4.581 ms
(50.1%). These observations remain in the tables and range whiskers.
The data do not identify whether tuning decisions, clocks, thermals or other
machine activity caused the variation.

H100 records the intended GPU, CUDA 13.0, the correct Triton backend, qualified
CUDA Graph execution and allocator peaks in every record. Its optional
NVML/nvidia-smi process-memory measurement is missing, not zero. There is
no CPU fallback. Do not infer continuous environment stability or complete
VRAM telemetry solely from successful execution.

## Final performance

Ratios are Meganeura/PyTorch synchronized wall time; lower favors Meganeura.
All native runs tune. All references compile, with applicable replay.

| Contract / phase | Median ratio over 35 comparisons | Nominal native wins |
|---|---:|---:|
| Strict inference | 1.830 | 7 / 35 |
| Strict minimal shape | 1.284 | 10 / 35 |
| Strict F+L+B | 2.467 | 3 / 35 |
| Accelerated inference | 2.119 | 6 / 35 |
| Accelerated minimal shape | 1.611 | 9 / 35 |
| Accelerated F+L+B | 2.908 | 4 / 35 |

Strict Pennycook workload means are 0.51/0.93 inference, 0.60/0.85 minimal,
and 0.39/0.98 training (Meganeura/PyTorch). These conditional GPU scores
exclude RPL-U, the extra H100 model sizes and the separate MI300X attempt.
Support across every attempted GPU would give both stacks a zero score:
PyTorch lacks a usable RPL-U GPU path; Meganeura lacks a validated MI300X
Vulkan driver path.

## What changed from the preceding cohort?

The comparison below uses the previous v7 primary condition at Inferena
efb1e520 / Meganeura 75dfe901 (archives now in ~/Downloads/p3hpc-v3).
That condition disabled native tuning and strict cooperative matrices.
CUDA already replayed, but ROCm/XPU did not; MPS and CPU were eager.
Consequently this is a cross-cohort comparison, not a controlled ablation.

| Device | Median native strict gain: inference / minimal / training |
|---|---:|
| RTX 5070 | 1.10 / 1.01 / 1.10 |
| H100 | 1.09 / 1.11 / 1.29 |
| RTX 3050 | 1.08 / 1.01 / 1.02 |
| RX 7900 XT | 1.10 / 1.00 / 1.24 |
| Radeon 780M | 1.01 / 1.00 / 1.02 |
| Arc B570 | 1.27 / 1.03 / 1.17 |
| Apple M3 | 1.09 / 1.00 / 0.92 |
| Intel RPL-U | 1.11 / 1.03 / 1.19 |

Each entry is the median of five old/new native time ratios, not a ratio of
aggregate times. Particularly useful improvements are strict ResNet training:

- RTX 5070: 44.240 → 31.739 ms, 1.39x faster.
- H100: 51.185 → 32.372 ms, 1.58x faster.
- H100 135M training: 52.405 → 38.652 ms, 1.36x faster.

The overall strict training ratio nevertheless changes 2.433 → 2.467;
minimal latency improves 1.430 → 1.284; inference is nearly unchanged
(1.833 → 1.830). The stronger reference matters. B570's median strict
reference gain is 1.59x inference and 2.08x minimal latency; native
improvements alone cannot predict the new paired ratio. M3 has substantial
variation and mixed native changes. No causal cooperative-f32 speedup is
established by these non-ablation data.

The main paper presents v9 results standalone. It retains a separately
identified H100 pilot for the actual search-off/on control and preparation
costs, not a mixed-revision main table. The original v7 analyzer is available
at paper commit 249464b; its load_campaign and aggregate functions reproduce
the old side of this comparison.

## Search and preparation

Every native session reports measured search, All scope, no class cap,
a 60-second soft deadline and a 1 GiB scratch ceiling with a device-memory
guard. Strict uses NativeF32; accelerated uses Auto with full-width
derivative protection. Apple M3 exposes native-f32 tiles and actually uses
them in all five strict workloads (2,150 dispatch instances over 15 processes).
No measured Vulkan device exposes native-f32 tiles through this stack;
NVIDIA gains therefore cannot be attributed to strict cooperative f32.

Of 708 sessions, 684 visit every eligible class (including zero-class
sessions); 13,322 / 14,085 class instances are visited. The 24 truncated
sessions are ResNet training on 780M, B570, M3 and RPL-U. The longest search
is 60.441 seconds. Candidate comparisons record 13,417 FasterCandidate,
31,456 KeepBaseline, 330 InvalidOutput and 24 TimeBudget decisions.
Numerically rejected candidates are not installed. Counts include repeated
processes and candidate comparisons, not distinct kernels or graph speedups.
GEMV, reduced-input cooperative variants and arbitrary graph representations
remain outside the implemented search domain.

Across 35 strict GPU groups, native compile+tune medians span 0.532–95.996 s
(median 7.614), versus 1.914–84.402 s PyTorch (median 29.983).
Native search can exceed reference compilation: RTX 5070 ResNet
33.813/13.262 s and M3 ResNet 95.996/5.724 s. Do not claim uniformly cheaper
startup or equal end-to-end budgets.

H100's main campaign totals 7.10 minutes native compilation/search,
13.73 PyTorch compilation and 4.66 research qualification. Its extension
totals 1.91, 9.88 and 27.32 minutes respectively. Large-model qualification
cost is CPU/readback validation, not kernel tuning. All reference
compilations finish below the enforced 120-second limit.

The separate v7 H100 pilot has three-process search-off/on controls:
native strict 135M prefill 13.265 → 9.744 ms, training 52.405 → 42.721 ms.
Across 30 searched pairs, PyTorch compilation totals 170.3 minutes versus
114.7 seconds native; default/replay PyTorch totals 14.0 minutes.
Those pilot numbers are not the current 60-second search policy.

## Scaling and capacity

Both 360M and 1.7B complete three processes under both arithmetic contracts.
Strict training ratios narrow 3.91 → 3.51 → 2.89 with model size, but prefill
stays near three and one-token ratio widens to 2.50 at 1.7B.
Accelerated training instead widens 5.29 → 5.64 → 7.10. Accelerated 1.7B
prefill is 12.22x PyTorch and is slower natively than strict (53.663 versus
34.198 ms). Permissions do not guarantee a faster selected implementation.

H100 strict 1.7B records 16.72 GiB of native execution-plan allocation versus
12.86 GiB PyTorch allocator peak / 13.54 GiB reserved. These scopes differ;
they do not establish a comparative peak-VRAM result. Small batch/sequence,
stateless token forward and no optimizer/distributed execution still limit
the scaling claim.

## Other evidence and reproduction

The RTX profiling study uses Meganeura 75dfe901 and two-second search,
not the primary cohort's revision/policy. Its grouped GPU span localizes a
convolution-gradient bottleneck, but its 5/71-class coverage is obsolete for
the final NVIDIA runs. No new Nsight trace or removable-barrier fraction is
claimed. [Diagnostic source and analysis](https://github.com/kvark/inferena/blob/3f8f994ce02aaf89ec5167d601403dc314b08e93/ANALYSIS-2026-09-13.md).

The MI300X report remains separately identified (SHA-256
e9be97e695140e8a36e09da0f0dd850113b0b22359182921dbc34b689d9776e8).
Its driver experiments are not timing cells. RPL-U CPU, XPU workarounds,
and the 780M ROCm overrides remain explicit portability evidence; old
Windows/H100 failures are not attributed to the complete final cohort.

From the repository root, using Python 3.11+:

    python3 paper/p3hpc/artifact/cohort.py "$HOME/Downloads/p3hpc" \
      --check paper/p3hpc/tables --output target/p3hpc-final-data

The supplementary ZIP contains all original JSON content, losslessly
recompressed as records.jsonl.xz; the same checker accepts that file instead
of the archive directory. It checks eight generated table/figure fragments
and exports all 84 condition groups to CSV, without a GPU or network.
Original text logs are not included; no raw measurement value is discarded.
No additional cohort or Intel server rental is required for this scoped paper.
