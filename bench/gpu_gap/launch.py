"""Targeted current-protocol pair with separate profiling, never a publication campaign."""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).parent / "inferena"
label, model, backend, precision = sys.argv[1:5]
destination = Path(__file__).parent / label
destination.mkdir()
env = dict(os.environ, PYTHON=sys.executable, CARGO_BUILD_JOBS="1",
           INFERENA_TORCH_BACKEND=backend, INFERENA_TORCH_MODE="default",
           INFERENA_GRAPH_REPLAY="1", MEGANEURA_TUNE="1",
           MEGANEURA_DEVICE_ID="12036" if backend == "cuda" else "57868",
           INFERENA_TUNE_SECONDS="60", INFERENA_COMPILE_SECONDS="120",
           INFERENA_PREPARATION_REPORT=str(destination / "torch-preparation.json"),
           INFERENA_REQUIRE_LOCAL_WEIGHTS="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
env.pop("VIRTUAL_ENV", None)
if "--local" in sys.argv:
    env["INFERENA_MEGANEURA_PATH"] = str(Path(__file__).parent / "source")
if "--inference-only" in sys.argv:
    env["INFERENA_INFERENCE_ONLY"] = "1"
if "--nsys" in sys.argv:
    env.update(INFERENA_NSYS=str(Path(__file__).with_name("nsys-window.py")),
               INFERENA_NSYS_DIR=str(destination))
command = ["bash", str(root / "run.sh"), "-f", "pytorch,meganeura", "-m", model,
           "--measurement-runs", "12",
           "--results-dir", str(destination)]
if "--inference-only" in sys.argv:
    command.append("--inference-only")
if "--native-only" in sys.argv:
    command[3] = "meganeura"
if "--nsys" not in sys.argv:
    command += ["--profile", "--profile-samples", "3"]
if precision == "strict":
    command.append("--strict")
manifest = {"inferena": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "purpose": "diagnostic; NVTX-windowed inference capture plus separate pass profiles",
            "command": command, "backend": backend, "precision": precision}
manifest["blade"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent / "blade", text=True).strip()
manifest["local_engine"] = env.get("INFERENA_MEGANEURA_PATH")
manifest["engine_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent / "source", text=True).strip()
manifest["flash_ept_cap"] = env.get("MEGANEURA_FLASH_EPT_CAP")
manifest["inference_only"] = env.get("INFERENA_INFERENCE_ONLY") == "1"
(destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
with (destination / "runner.log").open("w") as log:
    result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
print(destination, "exit", result.returncode, flush=True)
sys.exit(result.returncode)
