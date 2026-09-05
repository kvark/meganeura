#!/usr/bin/env python3
"""Verify the frozen P3HPC research object without GPU access."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
EXPECTED = ROOT / "expected"
INFERENA_REV = "7ca9c5c7b2cd"
MEGANEURA_REV = "7561a64"
PROFILE_REV = "b1405a3"
PLATFORMS = {"nvidia", "amd-d", "amd-i", "intel", "mac"}
MODELS = {"SmolLM2-135M", "SmolVLA", "StableDiffusion", "ResNet-50", "Whisper-tiny"}
MODES = {"paper-v1-strict", "paper-v1-accelerated"}
EXPECTED_TORCH = {
    "nvidia": "2.13.0+cu130",
    "amd-d": "2.10.0+rocm7.1",
    "amd-i": "2.12.0+rocm7.14.0",
    "intel": "2.11.0+xpu",
    "mac": "2.11.0",
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest() -> int:
    manifest = ROOT / "MANIFEST.sha256"
    if not manifest.is_file():
        fail("MANIFEST.sha256 is missing")
    checked = 0
    for line_number, line in enumerate(manifest.read_text().splitlines(), 1):
        if not line:
            continue
        try:
            expected_hash, relative = line.split("  ", 1)
        except ValueError as error:
            raise RuntimeError(f"bad manifest line {line_number}") from error
        path = (ROOT / relative).resolve()
        try:
            path.relative_to(ROOT.resolve())
        except ValueError as error:
            raise RuntimeError(f"manifest path escapes bundle: {relative}") from error
        if not path.is_file():
            fail(f"manifest file is missing: {relative}")
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            fail(f"hash mismatch for {relative}: {actual_hash} != {expected_hash}")
        checked += 1
    return checked


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot parse {path.relative_to(ROOT)}") from error


def audit_results() -> dict[str, int]:
    all_files = [path for path in RESULTS.rglob("*") if path.is_file()]
    summaries = list(RESULTS.glob("*/paper-v1-*/*_summary.json"))
    meganeura = list(RESULTS.glob("*/paper-v1-*/*_meganeura.json"))
    pytorch = list(RESULTS.glob("*/paper-v1-*/*_pytorch.json"))
    svgs = list(RESULTS.glob("*/paper-v1-*/*.svg"))
    profiles = list(RESULTS.glob("*/profiles/*.json"))
    counts = {
        "all": len(all_files),
        "summaries": len(summaries),
        "meganeura": len(meganeura),
        "pytorch": len(pytorch),
        "svgs": len(svgs),
        "profiles": len(profiles),
    }
    expected_counts = {
        "all": 165,
        "summaries": 50,
        "meganeura": 50,
        "pytorch": 50,
        "svgs": 10,
        "profiles": 5,
    }
    if counts != expected_counts:
        fail(f"unexpected result inventory: {counts} != {expected_counts}")

    for path in meganeura + pytorch:
        data = load_json(path)
        platform = path.relative_to(RESULTS).parts[0]
        mode = path.parent.name
        framework = "meganeura" if path in meganeura else "pytorch"
        model = path.name.removesuffix(f"_{framework}.json")
        if platform not in PLATFORMS:
            fail(f"unknown platform in {path}")
        if mode not in MODES or model not in MODELS:
            fail(f"unknown model or mode in {path}")
        if data.get("model") != model or data.get("framework") != framework:
            fail(f"record identity does not match filename in {path}")
        if data.get("benchmark_rev") != INFERENA_REV:
            fail(f"wrong Inferena revision in {path}")
        if data.get("protocol", {}).get("name") != "inferena-paper-v1":
            fail(f"wrong protocol in {path}")
        expected_warmups = 4 if platform == "intel" else 5
        protocol = data.get("protocol", {})
        if protocol.get("warmup_runs") != expected_warmups:
            fail(f"wrong warmup count in {path}")
        if protocol.get("measurement_runs") != 20:
            fail(f"wrong retained-sample count in {path}")
        samples = data.get("timing_samples_ms", {})
        for series in ("inference", "latency", "training"):
            if len(samples.get(series, [])) != 20:
                fail(f"{series} does not retain 20 samples in {path}")
            if any(not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0
                   for x in samples[series]):
                fail(f"{series} has nonpositive or nonfinite samples in {path}")
            median = statistics.median(samples[series])
            summary_median = data["timing_summary_ms"][series]["median"]
            if not math.isclose(median, summary_median, rel_tol=0, abs_tol=1e-6):
                fail(f"{series} summary median disagrees with samples in {path}")
            # Public timing values were rounded to milliseconds with three decimals.
            if not math.isclose(median, data["timings"][f"{series}_ms"],
                                rel_tol=0, abs_tol=0.000501):
                fail(f"{series} table timing disagrees with samples in {path}")
        expected_class = (
            "strict-f32" if "paper-v1-strict" in path.parts
            else "reduced-input-f32-accumulate"
        )
        if data.get("precision", {}).get("comparison_class") != expected_class:
            fail(f"wrong arithmetic contract in {path}")

        if data.get("framework") == "meganeura":
            if data.get("framework_rev") != MEGANEURA_REV:
                fail(f"wrong Meganeura revision in {path}")
        elif data.get("framework") == "pytorch":
            if data.get("framework_rev") != EXPECTED_TORCH[platform]:
                fail(f"wrong PyTorch version in {path}")
        else:
            fail(f"unknown framework in {path}")

    for path in summaries:
        data = load_json(path)
        if not isinstance(data, list) or len(data) != 2 or {item.get("framework") for item in data} != {
            "meganeura", "pytorch"
        }:
            fail(f"malformed joined summary in {path}")
        for item in data:
            raw_path = path.with_name(path.name.replace("_summary.json", f"_{item['framework']}.json"))
            if item != load_json(raw_path):
                fail(f"joined summary disagrees with raw record {raw_path}")
        mg = next(item for item in data if item["framework"] == "meganeura")
        pt = next(item for item in data if item["framework"] == "pytorch")
        validate_pair(mg, pt, path)

    for path in profiles:
        data = load_json(path)
        if data.get("framework_rev") != PROFILE_REV:
            fail(f"wrong profile revision in {path}")
        if len(data.get("normal_benchmark", {}).get("samples_ms", [])) != 20:
            fail(f"profile does not retain its 20 control samples in {path}")
        if data.get("profile", {}).get("measurement", {}).get("sample_count") != 5:
            fail(f"profile does not retain five timestamp samples in {path}")

    return counts


def validate_pair(mg: dict, pt: dict, path: Path) -> None:
    """Recompute the frozen harness gates from retained samples and norms.

    These records do not contain full tensors or elementwise gradients.
    The replay deliberately preserves that limitation.
    """
    a, b = mg["outputs"], pt["outputs"]

    def relative_l2(values, reference):
        delta = sum((x - y) ** 2 for x, y in zip(values, reference))
        norm = sum(x ** 2 for x in reference)
        return math.sqrt(delta / norm) if norm else (math.inf if delta else 0.0)

    sample_match = len(a["logits_sample"]) == len(b["logits_sample"]) == 256
    names_match = a["gradient_norms"].keys() == b["gradient_norms"].keys()
    if not sample_match or not names_match:
        fail(f"output sample or parameter inventory differs in {path}")
    values = [a["loss"], b["loss"], a["grad_norm"], b["grad_norm"],
              *a["logits_sample"], *b["logits_sample"],
              *a["gradient_norms"].values(), *b["gradient_norms"].values()]
    if not all(math.isfinite(value) for value in values):
        fail(f"nonfinite validation evidence in {path}")
    loss_error = abs(a["loss"] - b["loss"]) / max(abs(a["loss"]), abs(b["loss"]), 1e-12)
    output_error = relative_l2(a["logits_sample"], b["logits_sample"])
    grad_error = abs(a["grad_norm"] - b["grad_norm"]) / max(abs(b["grad_norm"]), 1e-12)
    names = sorted(b["gradient_norms"])
    norm_error = relative_l2([a["gradient_norms"][name] for name in names],
                            [b["gradient_norms"][name] for name in names])
    validation = mg["validation"]
    for key, value in (("loss_relative_error", loss_error),
                       ("output_relative_l2_error", output_error),
                       ("total_gradient_relative_error", grad_error),
                       ("parameter_gradient_relative_l2_error", norm_error)):
        if not math.isclose(value, validation[key], rel_tol=1e-9, abs_tol=1e-12):
            fail(f"{key} disagrees with retained evidence in {path}")
    forward = a["output_shape"] == b["output_shape"] and loss_error < 0.01 and output_error < 0.01
    training = forward and grad_error < 0.05 and norm_error < 0.05
    if validation["forward_valid"] != forward or validation["training_valid"] != training:
        fail(f"stored validity gates disagree with evidence in {path}")
    disputed = path.relative_to(RESULTS).parts[0] == "amd-i" and mg["model"] == "Whisper-tiny"
    expected_training = not disputed
    if not forward or training != expected_training:
        fail(f"validity pattern differs from the frozen 48/50 comparison in {path}")


def normalized_facts(stdout: str) -> str:
    lines = [
        line.rstrip()
        for line in stdout.splitlines()
        if not line.startswith("tables written to ")
    ]
    return "\n".join(lines).rstrip() + "\n"


def replay_analysis(repository: bool = False) -> str:
    with tempfile.TemporaryDirectory(prefix="meganeura-p3hpc-verify-") as tmp:
        work = Path(tmp)
        shutil.copy2(ROOT / "mktables.py", work / "mktables.py")
        shutil.copytree(RESULTS, work / "results")
        process = subprocess.run(
            [sys.executable, "mktables.py"],
            cwd=work,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        expected_tables = ROOT / "tables" if repository else EXPECTED / "tables"
        generated_tables = work / "tables"
        expected_names = sorted(path.name for path in expected_tables.glob("*.tex"))
        generated_names = sorted(path.name for path in generated_tables.glob("*.tex"))
        if generated_names != expected_names:
            fail(
                f"generated table inventory differs: {generated_names} != {expected_names}"
            )
        for name in expected_names:
            actual = (generated_tables / name).read_bytes()
            expected = (expected_tables / name).read_bytes()
            if actual != expected:
                fail(f"regenerated table differs from expected/tables/{name}")

        facts = normalized_facts(process.stdout)
        if not repository and facts != (EXPECTED / "facts.txt").read_text():
            fail("regenerated aggregate fact block differs from expected/facts.txt")
        return facts


def main() -> int:
    global ROOT, RESULTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show-facts", action="store_true", help="print the verified aggregate fact block"
    )
    parser.add_argument(
        "--repository", action="store_true",
        help="audit paper/results and replay tracked tables directly; no bundle manifest required",
    )
    args = parser.parse_args()
    if args.repository:
        ROOT = Path(__file__).resolve().parents[2]
        RESULTS = ROOT / "results"

    try:
        manifest_count = None if args.repository else verify_manifest()
        counts = audit_results()
        facts = replay_analysis(args.repository)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1

    if manifest_count is not None:
        print(f"OK: {manifest_count} packaged files match MANIFEST.sha256")
    print(
        "OK: result inventory is "
        f"{counts['summaries']} cells, {counts['profiles']} profiles, "
        f"{counts['all']} files"
    )
    print("OK: revisions, protocols, arithmetic contracts, samples and medians match")
    print("OK: summaries match raw records; sampled-output and gradient-norm gates replay")
    print("OK: all generated tables match expected output")
    if not args.repository:
        print("OK: aggregate facts match the packaged expected output")
    if args.show_facts:
        print()
        print(facts, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
