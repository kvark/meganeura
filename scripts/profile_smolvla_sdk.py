#!/usr/bin/env python3
"""Collect Nsight Perf SDK counters for identical, state-checked SmolVLA graphs."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time


METRICS = [
    "gr__cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__cycles_active.avg.pct_of_peak_sustained_elapsed",
    "tpc__warps_active_shader_cs_realtime.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_tensor_cycles_active_realtime.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed_realtime.avg.pct_of_peak_sustained_elapsed",
    "dram__read_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__write_throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__average_t_sector_hit_rate_srcunit_tex_realtime.pct",
    "gr__ctas_launched_queue_sync_realtime.sum.per_second",
    "gpc__cycles_elapsed.avg.per_second",
]


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def capture(repo, sdk, output, phase, workers):
    mode = "ordinary" if workers is None else str(workers)
    prefix = output / mode
    binary = repo / "target/release/examples/bench_smolvla_train"
    env = {k: v for k, v in os.environ.items() if not k.startswith("MEGANEURA_")}
    if workers is not None:
        env.update(MEGANEURA_PERSISTENT_WORKERS=str(workers),
                   MEGANEURA_DEVICE_SCHEDULING="1", MEGANEURA_PERSISTENT_STRICT="1")
    command = [str(binary), "--force", "--warmup", "10", "--runs", "20",
               "--profile-json", str(prefix), "--capture-phase", phase,
               "--capture-seconds", "10"]
    monitor_command = [str(sdk / "NvPerf/bin/x64/Monitoring"),
                       "--metrics", ",".join(METRICS),
                       "--output-yaml", str(prefix) + ".yaml", "--csv",
                       "--output-csv", str(prefix) + ".csv", "--csv-duration", "3",
                       "-n", "0.02"]
    receipt = {
        "target_command": command, "monitor_command": monitor_command,
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "source_status": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, text=True).strip(),
        "binary_sha256": sha256(binary),
        "benchmark_source_sha256": sha256(repo / "bench/bench_smolvla_train.rs"),
        "environment": {k: v for k, v in env.items()
                        if k.startswith("MEGANEURA_") or k == "VK_DRIVER_FILES"},
        "started_unix": time.time(),
    }
    log_path = Path(str(prefix) + ".log")
    with log_path.open("x") as log:
        target = subprocess.Popen(command, env=env, cwd=repo,
                                  stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 180
            while f"CAPTURE_READY {phase}" not in log_path.read_text(errors="replace"):
                if target.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError(f"Target did not become ready: {log_path}")
                time.sleep(0.05)
            receipt["ready_unix"] = time.time()
            capture_deadline = time.monotonic() + 9.5
            time.sleep(1)
            receipt["monitor_started_unix"] = time.time()
            monitor_log = Path(str(prefix) + "-monitor.log")
            with monitor_log.open("x") as log:
                monitor = subprocess.run(monitor_command, env=env, cwd=output,
                                         stdout=log, stderr=subprocess.STDOUT, timeout=20)
            receipt["monitor_exit"] = monitor.returncode
            receipt["monitor_finished_unix"] = time.time()
            if (monitor.returncode or target.poll() is not None
                    or time.monotonic() >= capture_deadline):
                raise RuntimeError("SDK failed or collection exceeded the resident capture window")
            if any(marker in monitor_log.read_text() for marker in ("WRN", "ERR")):
                raise RuntimeError(f"SDK reported a warning/error: {monitor_log}")
            receipt["target_exit"] = target.wait(timeout=30)
            if receipt["target_exit"]:
                raise RuntimeError(f"Target failed state qualification: {log_path}")
            record = json.loads(Path(str(prefix) + f".{phase}.json").read_text())
            assert record["profile_state_exact"] and record["profile"] is None
            execution = record["execution"]
            assert not execution["appended"] and not execution["appended_fallback"]
            if workers is not None:
                assert all(r["workers"] and not r["fallback_reason"]
                           for r in execution["regions"])
            receipt["state_sha256"] = sha256(Path(str(prefix) + f".{phase}.state.bin"))
            with Path(str(prefix) + ".csv").open() as source:
                rows = list(csv.DictReader(source))[5:]
            assert len(rows) >= 100, "Incomplete counter collection"
            means = {}
            for metric in METRICS:
                column, = [key for key in rows[0] if key.split(" (")[0] == metric]
                values = [float(row[column]) for row in rows]
                assert all(math.isfinite(value) for value in values)
                means[metric] = statistics.mean(values)
            receipt["counter_sample_count_after_warmup"] = len(rows)
            receipt["counter_means"] = means
            receipt["diagnostic_capture_median_ms"] = statistics.median(record["capture_samples_ms"])
            return receipt, record
        finally:
            if target.poll() is None:
                target.terminate()
                target.wait(timeout=10)
            receipt["finished_unix"] = time.time()
            Path(str(prefix) + "-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, required=True, help="extracted Perf SDK root")
    parser.add_argument("--output", type=Path, required=True, help="new local artifact directory")
    parser.add_argument("--phase", choices=("forward", "backward"), default="forward")
    parser.add_argument("--workers", type=int, nargs="+", default=[64, 256])
    args = parser.parse_args()
    assert all(1 <= workers <= 65535 for workers in args.workers)
    assert len(set(args.workers)) == len(args.workers)
    repo = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=False)
    reference, reference_record = capture(repo, args.sdk.resolve(), args.output.resolve(), args.phase, None)
    for workers in args.workers:
        receipt, record = capture(repo, args.sdk.resolve(), args.output.resolve(), args.phase, workers)
        assert record["implementation_keys"] == reference_record["implementation_keys"]
        assert record["state_layout"] == reference_record["state_layout"]
        assert receipt["state_sha256"] == reference["state_sha256"], "Backend state differs"
        print(f"{workers} workers: identical state and arithmetic; capture qualified", flush=True)


if __name__ == "__main__":
    main()
