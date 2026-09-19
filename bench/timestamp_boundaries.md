# GPU timestamp attribution diagnostic

This source-only experiment tests Blade's pass-start-to-next-start intervals,
not engine speed. `timestamp_boundaries` runs 65,536 independent integer
recurrences. A heavy pass performs 4,096 iterations; a light pass performs one.
Isolated and neighboring passes use the same pipeline and live buffer. All
output words are checked exactly against a CPU-computed affine recurrence
after every submission. GPU timestamps must also fall inside the CPU envelope.
Three warmups and ten saved samples per case, one process per GPU/configuration.

On Blade `eaff5092096aab136f11fa728b81c1bed3c0dcd4`, Intel B570 attributes an
isolated heavy pass only 0.104 microseconds. In a heavy/light pair, it assigns
0.104/645.286 microseconds instead: almost all work is under the wrong label.
NVIDIA 5070 already attributes the pair approximately correctly (15.104/4.064).

Blade correction: `e20b44849ccc33871bee5b0464bfe4cb8de737cf`.
Changing both pass and final markers from `TOP_OF_PIPE` to `BOTTOM_OF_PIPE`
makes the query wait for preceding work, rather than relying on barriers for
other pipeline stages. See the [timestamp synchronization scope](https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdWriteTimestamp.html).
The resulting median intervals, microseconds:

| Case | Arc B570 | RTX 5070 |
| --- | ---: | ---: |
| Heavy alone | 645.807 | 15.360 |
| Light alone | 10.521 | 2.976 |
| Heavy / light | 645.313 / 14.166 | 15.392 / 4.064 |
| Light / heavy | 11.224 / 649.115 | 3.248 / 16.320 |
| Heavy / light / heavy | 645.625 / 13.412 / 648.594 | 15.520 / 4.080 / 14.416 |

This is instrumentation: completion-stage markers can change overlap. No
ordinary untimed execution or numerical tolerance is changed. AMD and Metal
were not measured here; the implementation change is Vulkan-only.

Reproduce after checking out this source branch and the Blade fix:

```sh
cargo build --release --example timestamp_boundaries \
  --config 'patch."https://github.com/kvark/blade".blade-graphics.path="../blade/blade-graphics"'
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/intel_icd.json \
  target/release/examples/timestamp_boundaries > /tmp/timestamps.json
```

Use the NVIDIA ICD for the other GPU. Build without the patch for the original
behavior. Keep JSON, binaries and profiler captures outside Git.

The same correction changes the B570 SmolLM2-135M prefill profile: dense F16
matmul/add intervals total 17.375 ms (77% of the sum of dispatch medians),
cached attention 3.330 ms (15%), RMSNorm 0.533 ms and SwiGLU 0.266 ms. This
identifies useful kernel targets; it is not a decomposition of the 28% gap to
llama.cpp, and includes profiling/barrier effects. Earlier Intel rankings that
charged large matmuls to following normalizations/activations are not reliable.
The uninstrumented host-wall benchmark and scheduling search are unaffected.
