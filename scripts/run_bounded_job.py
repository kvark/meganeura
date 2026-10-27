#!/usr/bin/env python3
"""Run one Linux build, test, or benchmark in a separate bounded user cgroup.

Usage: python3 scripts/run_bounded_job.py LABEL [--gpu] -- COMMAND ...
Requires systemd's user manager and the unified cgroup hierarchy.
Use --gpu for GPU tests, benchmarks and profiling to serialize their processes.
"""
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import uuid


if len(sys.argv) > 1 and sys.argv[1] == "--inside":
    unit, *command = sys.argv[2:]
    gpu = command[0] == "--gpu"
    inherited = ()
    if gpu:
        command.pop(0)
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
    if gpu:
        # Share the lock across checkouts, while keeping it on local runtime storage.
        path = Path("/run/user") / str(os.getuid()) / "meganeura-gpu-job.lock"
        lock = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        os.set_inheritable(lock, True)
        inherited = (lock,)
        print("Acquired GPU job lock", file=sys.stderr, flush=True)
    child = subprocess.Popen(command, pass_fds=inherited)

    def forward_signal(sig, _frame):
        try:
            child.send_signal(sig)
        except ProcessLookupError:
            pass

    previous = {sig: signal.signal(sig, forward_signal)
                for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        returncode = child.wait()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    # Read before this scope disappears. Limits alone do not show whether a
    # benchmark actually encountered reclaim, throttling, or memory pressure.
    resources = {
        key: (cgroup / key).read_text().strip()
        for key in ("memory.peak", "memory.events", "memory.swap.peak",
                    "memory.pressure", "cpu.stat", "cpu.pressure", "io.pressure")
        if (cgroup / key).exists()
    }
    print(json.dumps({"job_resources": {"cgroup": relative,
                                      "exit_code": returncode,
                                      **resources}}), file=sys.stderr, flush=True)
    if returncode < 0:
        sig = -returncode
        if sig not in (signal.SIGKILL, signal.SIGSTOP):
            signal.signal(sig, signal.SIG_DFL)
        os.kill(os.getpid(), sig)
    raise SystemExit(returncode)
else:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    label, *command = sys.argv[1:]
    gpu = bool(command and command[0] == "--gpu")
    if gpu:
        command.pop(0)
    if command and command[0] == "--":
        command.pop(0)
    if not re.fullmatch(r"[a-z0-9-]+", label) or not command:
        raise SystemExit("Use a lowercase letter/digit/hyphen label and a command.")
    unit = "meganeura-" + label + "-" + uuid.uuid4().hex[:8]
    os.execvp("systemd-run", [
        "systemd-run", "--user", "--scope", "--slice=ml.slice", "--unit=" + unit,
        "-p", "MemoryHigh=16G", "-p", "MemoryMax=20G", "-p", "MemorySwapMax=1G",
        sys.executable, str(Path(__file__).resolve()), "--inside", unit,
        *(["--gpu"] if gpu else []), *command,
    ])
