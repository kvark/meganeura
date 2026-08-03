#!/usr/bin/env python3
"""Verify the frozen P3HPC research object without GPU access."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
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
        if platform not in PLATFORMS:
            fail(f"unknown platform in {path}")
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
        if not isinstance(data, list) or {item.get("framework") for item in data} != {
            "meganeura", "pytorch"
        }:
            fail(f"malformed joined summary in {path}")

    for path in profiles:
        data = load_json(path)
        if data.get("framework_rev") != PROFILE_REV:
            fail(f"wrong profile revision in {path}")
        if len(data.get("normal_benchmark", {}).get("samples_ms", [])) != 20:
            fail(f"profile does not retain its 20 control samples in {path}")
        if data.get("profile", {}).get("measurement", {}).get("sample_count") != 5:
            fail(f"profile does not retain five timestamp samples in {path}")

    return counts


def normalized_facts(stdout: str) -> str:
    lines = [
        line.rstrip()
        for line in stdout.splitlines()
        if not line.startswith("tables written to ")
    ]
    return "\n".join(lines).rstrip() + "\n"


def replay_analysis() -> str:
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
        expected_tables = EXPECTED / "tables"
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
        expected_facts = (EXPECTED / "facts.txt").read_text()
        if facts != expected_facts:
            fail("regenerated aggregate fact block differs from expected/facts.txt")
        return facts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show-facts", action="store_true", help="print the verified aggregate fact block"
    )
    args = parser.parse_args()

    try:
        manifest_count = verify_manifest()
        counts = audit_results()
        facts = replay_analysis()
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1

    print(f"OK: {manifest_count} packaged files match MANIFEST.sha256")
    print(
        "OK: result inventory is "
        f"{counts['summaries']} cells, {counts['profiles']} profiles, "
        f"{counts['all']} files"
    )
    print("OK: revisions, protocols, arithmetic contracts, and sample counts match")
    print("OK: all generated tables and aggregate facts match expected output")
    if args.show_facts:
        print()
        print(facts, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
