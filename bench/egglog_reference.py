"""Full CPU reference for the immutable model-search experiment, not timings.

Usage: python egglog_reference.py INFERENA_CHECKOUT MODEL OUTPUT.f32
Use the pinned Inferena venv and source revision recorded in egglog_search.md.
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(sys.argv[1]) / "frameworks/pytorch"))
import bench
import torch

torch.set_num_threads(6)
name, destination = sys.argv[2:]
assert name in ("SmolVLA", "Whisper-tiny")
spec = bench.MODEL_REGISTRY[name]
bench._configure_benchmark_precision("cpu", strict=True)
model = bench.load_model(name, spec, "cpu").eval().requires_grad_(False)
inputs = bench.prepare_inputs(spec["type"], model, "cpu")
with torch.no_grad():
    output = bench._benchmark_logits(
        spec["type"], bench._benchmark_forward(spec["type"], model, inputs)
    ).contiguous()
assert torch.isfinite(output).all()
output.numpy().astype("<f4").tofile(destination)
print(json.dumps({"model": name, "shape": list(output.shape),
                  "torch": torch.__version__, "torch_revision": torch.version.git_version,
                  "dtype": str(output.dtype), "sha256": bench.sha256_f32_tensor(output)}))
