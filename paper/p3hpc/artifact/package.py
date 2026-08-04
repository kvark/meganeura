#!/usr/bin/env python3
"""Build and smoke-test the deterministic P3HPC supplementary archive."""

from __future__ import annotations

import argparse
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile


HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]
REPO = PAPER.parent
BUNDLE_NAME = "meganeura-p3hpc-artifact-v1"
MEGANEURA_REV = "7561a64ec5a7e4bcdcd2c719aaaffe5912ed5e85"
INFERENA_REV = "7ca9c5c7b2cd614343a3de3dcc86999ced66e8c0"
PROFILE_REV = "b1405a3a52fabf9858aca5cbd80e246811cb6a58"
DINOVISION_SOURCE_REV = "dc35cdf1c7c910cdd93c5b5362846842ae469a21"
DINOVISION_EVIDENCE_REV = "2c3f9017fe74c41482b165890c14737a2ccd4b6a"
COMPANION_REPORT = "https://arxiv.org/abs/2608.01563"
COMPANION_REPORT_DOI = "10.48550/arXiv.2608.01563"
ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def run(command: list[str], *, cwd: Path, text: bool = True):
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
    ).stdout


def git_revision(repo: Path, revision: str) -> str:
    return run(["git", "rev-parse", f"{revision}^{{commit}}"], cwd=repo).strip()


def ensure_clean(repo: Path) -> str:
    status = run(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=repo)
    if status:
        raise RuntimeError("commit or remove worktree changes before packaging:\n" + status)
    return git_revision(repo, "HEAD")


def safe_extract_git_archive(repo: Path, revision: str, destination: Path) -> None:
    archive = run(
        ["git", "archive", "--format=tar", revision], cwd=repo, text=False
    )
    destination.mkdir(parents=True)
    root = destination.resolve()
    with tarfile.open(fileobj=BytesIO(archive), mode="r:") as source:
        for member in source.getmembers():
            target = (destination / member.name).resolve()
            try:
                target.relative_to(root)
            except ValueError as error:
                raise RuntimeError(f"unsafe archive path: {member.name}") from error
        source.extractall(destination, filter="data")


def normalized_facts(stdout: str) -> str:
    lines = [
        line.rstrip()
        for line in stdout.splitlines()
        if not line.startswith("tables written to ")
    ]
    return "\n".join(lines).rstrip() + "\n"


def representative_environments(results: Path) -> dict:
    output = {"schema_version": 1, "note": "Representative strict SmolLM records; every raw record retains its own metadata.", "platforms": {}}
    for platform in ("nvidia", "amd-d", "amd-i", "intel", "mac"):
        base = results / platform / "paper-v1-strict"
        meganeura = json.loads((base / "SmolLM2-135M_meganeura.json").read_text())
        pytorch = json.loads((base / "SmolLM2-135M_pytorch.json").read_text())
        output["platforms"][platform] = {
            "meganeura": {
                "framework_rev": meganeura["framework_rev"],
                "benchmark_rev": meganeura["benchmark_rev"],
                "backend": meganeura["backend"],
                "environment": meganeura["environment"],
                "protocol": meganeura["protocol"],
            },
            "pytorch": {
                "framework_rev": pytorch["framework_rev"],
                "benchmark_rev": pytorch["benchmark_rev"],
                "backend": pytorch["backend"],
                "environment": pytorch["environment"],
                "protocol": pytorch["protocol"],
            },
        }
    return output


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(stage: Path) -> None:
    files = sorted(
        path for path in stage.rglob("*")
        if path.is_file() and path.name != "MANIFEST.sha256"
    )
    lines = [f"{sha256(path)}  {path.relative_to(stage).as_posix()}" for path in files]
    (stage / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")


def write_deterministic_zip(stage: Path, output: Path) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in sorted(item for item in stage.rglob("*") if item.is_file()):
            relative = Path(BUNDLE_NAME) / path.relative_to(stage)
            info = zipfile.ZipInfo(relative.as_posix(), ZIP_TIME)
            executable = path.name in {"verify.py", "verify.sh"}
            info.external_attr = ((0o755 if executable else 0o644) & 0xFFFF) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes(), compresslevel=9)
    os.replace(temporary, output)
    digest = sha256(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n"
    )
    return digest


def smoke_test_archive(archive: Path) -> str:
    with tempfile.TemporaryDirectory(prefix="meganeura-p3hpc-smoke-") as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(archive) as source:
            source.extractall(root)
        bundle = root / BUNDLE_NAME
        return run([sys.executable, "verify.py"], cwd=bundle)


def assemble(inferena_repo: Path, output: Path) -> tuple[str, str]:
    paper_revision = ensure_clean(REPO)
    if git_revision(REPO, MEGANEURA_REV) != MEGANEURA_REV:
        raise RuntimeError("Meganeura benchmark revision is unavailable")
    if git_revision(REPO, PROFILE_REV) != PROFILE_REV:
        raise RuntimeError("Meganeura profile revision is unavailable")
    if git_revision(inferena_repo, INFERENA_REV) != INFERENA_REV:
        raise RuntimeError("Inferena benchmark revision is unavailable")

    with tempfile.TemporaryDirectory(prefix="meganeura-p3hpc-package-") as tmp:
        stage = Path(tmp) / BUNDLE_NAME
        stage.mkdir()
        for source, target in (
            (REPO / "LICENSE", stage / "LICENSE"),
            (REPO / "CITATION.cff", stage / "CITATION.cff"),
            (HERE / "README.md", stage / "README.md"),
            (HERE / "verify.py", stage / "verify.py"),
            (HERE / "verify.sh", stage / "verify.sh"),
            (PAPER / "mktables.py", stage / "mktables.py"),
        ):
            shutil.copy2(source, target)
        shutil.copytree(PAPER / "results", stage / "results")

        environments = representative_environments(stage / "results")
        (stage / "ENVIRONMENTS.json").write_text(
            json.dumps(environments, indent=2, sort_keys=True) + "\n"
        )
        build_info = {
            "schema_version": 1,
            "bundle": BUNDLE_NAME,
            "paper_repository_revision": paper_revision,
            "meganeura_matrix_revision": MEGANEURA_REV,
            "inferena_revision": INFERENA_REV,
            "meganeura_profile_revision": PROFILE_REV,
            "dinovision_source_revision": DINOVISION_SOURCE_REV,
            "dinovision_evidence_revision": DINOVISION_EVIDENCE_REV,
            "companion_report": COMPANION_REPORT,
            "companion_report_doi": COMPANION_REPORT_DOI,
        }
        (stage / "BUILD_INFO.json").write_text(
            json.dumps(build_info, indent=2, sort_keys=True) + "\n"
        )

        analysis = run([sys.executable, "mktables.py"], cwd=stage)
        expected = stage / "expected"
        expected.mkdir()
        shutil.move(stage / "tables", expected / "tables")
        (expected / "facts.txt").write_text(normalized_facts(analysis))

        repository_tables = PAPER / "tables"
        for generated in sorted((expected / "tables").glob("*.tex")):
            tracked = repository_tables / generated.name
            if not tracked.is_file() or generated.read_bytes() != tracked.read_bytes():
                raise RuntimeError(f"generated {generated.name} differs from paper/tables")

        source_root = stage / "source"
        safe_extract_git_archive(REPO, MEGANEURA_REV, source_root / "meganeura")
        safe_extract_git_archive(
            REPO, PROFILE_REV, source_root / "meganeura-profile-revision"
        )
        safe_extract_git_archive(inferena_repo, INFERENA_REV, source_root / "inferena")

        write_manifest(stage)
        digest = write_deterministic_zip(stage, output)

    verification = smoke_test_archive(output)
    return digest, verification


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inferena-repo",
        type=Path,
        default=REPO.parent / "inferena",
        help="Inferena Git checkout containing the frozen revision",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE.parent / "submission" / f"{BUNDLE_NAME}.zip",
        help="output ZIP path",
    )
    args = parser.parse_args()
    try:
        digest, verification = assemble(
            args.inferena_repo.resolve(), args.output.resolve()
        )
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print(verification, end="")
    print(f"archive: {args.output.resolve()}")
    print(f"sha256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
