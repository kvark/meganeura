#!/usr/bin/env python3
"""Reproduce the paper's size figures (no GPU).

  footprint.py source CHECKOUT      Rust and WGSL lines under CHECKOUT/src at 0dbfcc00
  footprint.py binary RUNNER        stripped size and linked libraries of a runner copy
  footprint.py closure SITE         installed size of torch's dependency closure; run it
                                    with the measured environment's Python

Source lines exclude blank lines, comments and `#[cfg(test)]` modules. The
closure follows active Requires-Dist edges and requested extras from torch and
counts each installed file once.
"""

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


def rust_lines(text):
    count, block, skip, pending = 0, False, None, False
    for raw in text.split("\n"):
        line = raw.strip()
        if block:
            block = "*/" not in line
            continue
        if line.startswith("/*"):
            block = "*/" not in line
            continue
        if not line or line.startswith("//"):
            continue
        if skip is not None:
            skip += line.count("{") - line.count("}")
            if skip <= 0:
                skip = None
            continue
        if line.startswith("#[cfg(test)]"):
            pending = True
            continue
        if pending:
            pending = False
            if re.match(r"(pub(\(crate\))? )?mod \w+\s*\{", line):
                skip = line.count("{") - line.count("}") or None
                continue
        count += 1
    return count


def source(checkout):
    rust = wgsl = shaders = 0
    for path in sorted((checkout / "src").rglob("*")):
        if path.suffix == ".rs":
            rust += rust_lines(path.read_text(encoding="utf-8"))
        elif path.suffix == ".wgsl":
            shaders += 1
            wgsl += sum(1 for line in path.read_text(encoding="utf-8").split("\n")
                        if line.strip() and not line.strip().startswith("//"))
    print(f"Rust {rust} lines; WGSL {wgsl} lines in {shaders} files")


def binary(runner):
    with tempfile.TemporaryDirectory() as directory:
        copy = Path(directory) / runner.name
        shutil.copy2(runner, copy)
        subprocess.run(["strip", str(copy)], check=True)
        size = copy.stat().st_size
    print(f"stripped {size} bytes = {size / 2**20:.1f} MiB")
    print(subprocess.run(["ldd", str(runner)], capture_output=True, text=True).stdout.strip())


def closure(site):
    import importlib.metadata as metadata
    from packaging.markers import default_environment
    from packaging.requirements import Requirement

    def norm(name):
        return re.sub(r"[-_.]+", "-", name).lower()

    dists = {norm(dist.metadata["Name"]): dist for dist in metadata.distributions(path=[str(site)])}
    files, sizes, visited, stack = set(), {}, set(), [("torch", frozenset())]
    while stack:
        name, extras = stack.pop()
        if (name, extras) in visited or name not in dists:
            continue
        visited.add((name, extras))
        dist = dists[name]
        if name not in sizes:
            total = 0
            for file in dist.files or []:
                path = os.path.realpath(dist.locate_file(file))
                if path not in files and os.path.isfile(path):
                    files.add(path)
                    total += os.path.getsize(path)
            sizes[name] = total
        for text in dist.requires or []:
            requirement = Requirement(text)
            environment = default_environment()
            if not requirement.marker or any(requirement.marker.evaluate(dict(environment, extra=extra))
                                             for extra in {"", *extras}):
                stack.append((norm(requirement.name), frozenset(requirement.extras)))
    total = sum(sizes.values())
    print(f"{len(sizes)} distributions, {total / 2**30:.2f} GiB; torch + triton "
          f"{(sizes.get('torch', 0) + sizes.get('triton', 0)) / 2**30:.2f} GiB")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kind", choices=("source", "binary", "closure"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    {"source": source, "binary": binary, "closure": closure}[args.kind](args.path)


if __name__ == "__main__":
    main()
