# Deployment-footprint checkpoint (2026-07-24)

This is a pre-freeze Linux x86-64 measurement for the paper. It supports the
deployment-cost discussion; it is not a claim that Meganeura and PyTorch have
equivalent framework coverage.

## Results

| Artifact | Bytes | Binary units |
|---|---:|---:|
| Stripped `inferena-meganeura` release executable | 13,452,560 | 12.83 MiB |
| PyTorch and Triton distribution files | 1,876,675,267 | 1.75 GiB |
| Active PyTorch runtime dependency closure | 5,003,725,538 | 4.66 GiB |

The Meganeura executable includes the graph compiler, autodiff, runtime, model
builders, and embedded shader sources. It was built from the sibling
Meganeura tree through Inferena and copied through GNU `strip -o`, so the
unstripped benchmark artifact was preserved. `ldd` reports OpenSSL, zlib,
zstd, `libgcc_s`, `libm`, and glibc in addition to the loader. The GPU driver
is also a system prerequisite. “Self-contained” in the paper therefore means
one application executable with no Python or vendor userspace ML runtime, not
a statically linked binary that has no OS dependencies.

The PyTorch environment contained PyTorch `2.13.0+cu130` and Triton `3.7.1`.
The complete virtual environment was not measured because it also contained
unrelated tools and old CUDA-12 packages. Instead, the dependency closure was
computed from installed Python distribution metadata:

1. Start with the `torch` distribution.
2. Recursively follow requirements whose environment markers are active,
   including the CUDA extras requested from `cuda-toolkit`.
3. Sum each unique installed file once.

The resulting closure contains 29 distributions, including PyTorch, Triton,
the CUDA 13 libraries requested by PyTorch, cuDNN, cuSparseLt, NCCL, NVSHMEM,
and ordinary Python dependencies. Python itself, OS libraries, the NVIDIA
driver, model weights, downloaded data, caches, and development tools are
excluded. Model weights and system GPU drivers are likewise excluded from the
Meganeura row.

## Freeze requirements

- Rebuild and strip the runner from the clean, pinned Meganeura and Inferena
  revisions.
- Archive the executable hash, byte count, `file`, and `ldd` output.
- Archive the Python dependency list, versions, file-size report, and
  measurement script.
- Repeat on Apple if the paper makes a cross-platform footprint claim.
- Present this as deployment footprint, not feature parity: PyTorch provides
  a much broader API, dynamic execution environment, and operator ecosystem.
