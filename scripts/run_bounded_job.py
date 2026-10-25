#!/usr/bin/env python3
"""Run one Linux build, test, or benchmark in a separate bounded user cgroup.

Usage: python3 scripts/run_bounded_job.py LABEL -- COMMAND ...
Requires systemd's user manager and the unified cgroup hierarchy.
"""
import json
import os
from pathlib import Path
import re
import sys
import uuid


if len(sys.argv) > 1 and sys.argv[1] == "--inside":
    unit, *command = sys.argv[2:]
    relative = next(
        (line[3:] for line in Path("/proc/self/cgroup").read_text().splitlines()
         if line.startswith("0::")),
        "",
    )
    if not relative.endswith("/ml.slice/" + unit + ".scope"):
        raise SystemExit(f"Unexpected child cgroup: {relative!r}")
    cgroup = Path("/sys/fs/cgroup") / relative.lstrip("/")
    limits = {key: (cgroup / key).read_text().strip()
              for key in ("memory.high", "memory.max", "memory.swap.max")}
    expected = {"memory.high": str(16 * 1024**3),
                "memory.max": str(20 * 1024**3),
                "memory.swap.max": str(1024**3)}
    if limits != expected:
        raise SystemExit(f"Unexpected child memory limits: {limits!r}")
    print(json.dumps({"cgroup": relative, "limits": limits}), file=sys.stderr, flush=True)
    os.execvp(command[0], command)
else:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    label, *command = sys.argv[1:]
    if command and command[0] == "--":
        command.pop(0)
    if not re.fullmatch(r"[a-z0-9-]+", label) or not command:
        raise SystemExit("Use a lowercase letter/digit/hyphen label and a command.")
    unit = "meganeura-" + label + "-" + uuid.uuid4().hex[:8]
    os.execvp("systemd-run", [
        "systemd-run", "--user", "--scope", "--slice=ml.slice", "--unit=" + unit,
        "-p", "MemoryHigh=16G", "-p", "MemoryMax=20G", "-p", "MemorySwapMax=1G",
        sys.executable, str(Path(__file__).resolve()), "--inside", unit, *command,
    ])
