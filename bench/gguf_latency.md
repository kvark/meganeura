# Matched Vulkan GGUF diagnostic

`gguf_latency.rs` and `gguf_latency.cpp` compare identical GGUF tensors and
fixed token IDs. Both use one sequence, 128-token prefill, 32 cached decode steps,
context capacity 256, F32 K/V caches, three warmup sequences and seven measured
sequences. Prefill returns only the last row of logits. Whole-call times include
recording, submission, waiting and copying the full vocabulary to CPU memory.
The first measured sequence saves 33 logit vectors.

This is interactive batch-one latency, not batched throughput or language
quality. A 128-token prompt is not 128 concurrent requests. llama.cpp uses
Vulkan with all layers offloaded and flash attention enabled. Meganeura records
fresh commands on every step. This diagnostic does not change the paper cohort.

## Reproduction

The measured model is `HuggingFaceTB/SmolLM2-135M` at
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, converted to F16 with the official
converter from llama.cpp `05f2dcfdba3879c55f735efa0f124b1a56f7ed11`.
Use that checkout for both conversion and the comparison binary. The resulting
GGUF SHA-256 is
`56df427e4aa9a57d67b207d45e1a2b51e80b956b11cd9fbf0c162429f95168c7`.
The converter's Q/K permutation matters; an earlier hand-converted file was
incorrect and its timings were discarded.

```sh
hf download HuggingFaceTB/SmolLM2-135M \
  --revision 93efa2f097d58c2a74874c7e644dbc9b0cee75a2 --local-dir hf-smollm2
# Install the pinned llama.cpp converter's requirements in a separate venv.
python ../llama.cpp/convert_hf_to_gguf.py hf-smollm2 \
  --outtype f16 --outfile model.gguf
cargo build --release --features gguf --example gguf_latency
cmake -S ../llama.cpp -B ../llama.cpp/build -DGGML_VULKAN=ON
cmake --build ../llama.cpp/build -j2
c++ -O2 -std=c++17 bench/gguf_latency.cpp \
  -I ../llama.cpp/include -I ../llama.cpp/ggml/include \
  -L ../llama.cpp/build/bin -Wl,-rpath,"$PWD/../llama.cpp/build/bin" \
  -lllama -lggml -lggml-base -o /tmp/llama-latency

# Use intel_icd.json for Intel; select exactly one GPU.
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export GGML_VK_VISIBLE_DEVICES=0
MEGANEURA_COOP_F16=1 target/release/examples/gguf_latency model.gguf /tmp/meg 30
/tmp/llama-latency model.gguf /tmp/llama
```

Run engines and GPUs sequentially, without competing builds or profilers.
Repeat in fresh processes and reverse engine order. The optional budget is a soft
construction-time limit in seconds per session; `0` disables measured search.
It includes compilation, initialization, qualification and measurements. In-flight
driver calls cannot be preempted. Private kernel probes retain their full
qualification and the diagnostic's original 256 MiB scratch limit.

`build_measured` compares graph forms, kernel choices and submission chunks in one
complete-plan search. Each candidate checks every logit and cache element against
an untuned reference (`atol=1e-5`, `rtol=1e-4`), with full staged readback.
Decode calibration primes that cache through the middle of the decode range
(position 144). Trials reset private mutable state;
selected prefill/decode sessions share caches only after search. Reference
preparation and priming are outside the search budget but inside `prepare_ms`.
The old scope argument and separate live submission tuner are removed. Historical
measurements below retain their original protocol and revision.

Known staged output downloads are queued before the CPU wait. Initial probes
and mapped reads still wait first. `decode_record_finish_ms` reports CPU
record/submit and combined wait/copy intervals, not separate GPU costs.
For separate captures, use `MEGANEURA_GPU_TIMING=1` or llama.cpp's
`GGML_VK_PERF_LOGGER=1`. Instrumentation changes execution; wall time minus
summed pass intervals is not a measurement of barrier cost.

## Measured checkpoint and experiment history

The source-only branch `experiment/llama-catchup-pre-cleanup-2026-09-20` at
`137f93364afe06cfff89c888efddb6bb05c54d33` preserves the implementations,
intermediate reports and links to isolated experiments. It contains no raw
data or binaries. In particular, [queued readback](https://github.com/kvark/meganeura/blob/137f93364afe06cfff89c888efddb6bb05c54d33/bench/queued-readback.md)
records the final pre-cleanup checkpoint and [scalar-layout tuning](https://github.com/kvark/meganeura/blob/137f93364afe06cfff89c888efddb6bb05c54d33/bench/scalar-matmul-autotune.md)
isolates the 29% Intel prefill improvement.

Milliseconds, median of three fresh-process medians, rotated engine order:

| GPU / phase | Meganeura | llama.cpp |
| --- | ---: | ---: |
| RTX 5070 prefill | 7.210 | 7.126 |
| RTX 5070 decode/token | 1.389 | 1.361 |
| Arc B570 prefill | 14.048 | 15.650 |
| Arc B570 decode/token | 3.666 | 3.431 |

These are measurements of `7c95707b9a69478c1bd498f940b7dde6e851a7e1`, not a
new cohort on the cleanup branch. All saved logits are finite and retain the
33 independent CPU-reference token choices. Maximum per-row relative L2:
0.00000954/0.000178 for Meganeura, 0.01121/0.01314 for llama.cpp. Preparation
takes 9.61/31.45 seconds versus 0.25/0.51 seconds, with existing driver caches.

The i5-12400F used physical cores `0,2,4,6,8,10`, unfixed clocks, NVIDIA
595.91.07 and Intel Mesa 26.0.3. B570 was on a secondary PCIe x1 link, limiting
readback generalization. Decode process medians vary with measured split/chunk
choices; this is close performance, not stable parity. Reusable command
recording remains an attribution experiment, not a production path. Known
Naga Workgroup ArrayStride diagnostics remain unresolved.
