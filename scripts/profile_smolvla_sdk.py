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


def capture(repo, sdk, output, phase, workers, whole_graph=False, metric_groups=None):
    metric_groups = metric_groups or [METRICS]
    capture_seconds = max(10, 4 * len(metric_groups) + 4)
    mode = "ordinary" if workers is None else ("whole-" if whole_graph else "") + str(workers)
    prefix = output / mode
    binary = repo / "target/release/examples/bench_smolvla_train"
    env = {k: v for k, v in os.environ.items() if not k.startswith("MEGANEURA_")}
    if workers is not None:
        env.update(MEGANEURA_PERSISTENT_WORKERS=str(workers),
                   MEGANEURA_DEVICE_SCHEDULING="1", MEGANEURA_PERSISTENT_STRICT="1")
        if whole_graph:
            env["MEGANEURA_WHOLE_GRAPH"] = "1"
    command = [str(binary), "--force", "--warmup", "10", "--runs", "20",
               "--profile-json", str(prefix), "--capture-phase", phase,
               "--capture-seconds", str(capture_seconds)]
    monitor_groups = []
    for index, metrics in enumerate(metric_groups):
        group_prefix = str(prefix) + (f"-group-{index:02}" if len(metric_groups) > 1 else "")
        monitor_groups.append({
            "metrics": metrics, "prefix": group_prefix,
            "command": [str(sdk / "NvPerf/bin/x64/Monitoring"),
                        "--metrics", ",".join(metrics),
                        "--output-yaml", group_prefix + ".yaml", "--csv",
                        "--output-csv", group_prefix + ".csv", "--csv-duration", "3",
                        "-n", "0.02"],
        })
    receipt = {
        "target_command": command, "monitor_groups": monitor_groups,
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "source_status": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, text=True).strip(),
        "binary_sha256": sha256(binary),
        "benchmark_source_sha256": sha256(repo / "bench/bench_smolvla_train.rs"),
        "capture_script_sha256": sha256(Path(__file__)),
        "environment": {k: v for k, v in env.items()
                        if k.startswith("MEGANEURA_")
                        or k in ("VK_DRIVER_FILES", "VK_ICD_FILENAMES")},
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
            capture_deadline = time.monotonic() + capture_seconds - 0.5
            time.sleep(1)
            values_by_metric = {}
            for index, group in enumerate(monitor_groups):
                group["started_unix"] = time.time()
                monitor_log = Path(group["prefix"] + "-monitor.log")
                with monitor_log.open("x") as log:
                    monitor = subprocess.run(group["command"], env=env, cwd=output,
                                             stdout=log, stderr=subprocess.STDOUT, timeout=20)
                group["exit"] = monitor.returncode
                group["finished_unix"] = time.time()
                if (monitor.returncode or target.poll() is not None
                        or time.monotonic() >= capture_deadline):
                    raise RuntimeError("SDK failed or collection exceeded the resident capture window")
                if any(marker in monitor_log.read_text() for marker in ("WRN", "ERR")):
                    raise RuntimeError(f"SDK reported a warning/error: {monitor_log}")
                with Path(group["prefix"] + ".csv").open() as source:
                    rows = list(csv.DictReader(source))[5:]
                assert len(rows) >= 100, "Incomplete counter collection"
                group["sample_count_after_warmup"] = len(rows)
                group["counter_means"] = {}
                for metric in group["metrics"]:
                    column, = [key for key in rows[0] if key.split(" (")[0] == metric]
                    values = [float(row[column]) for row in rows]
                    assert all(math.isfinite(value) for value in values)
                    group["counter_means"][metric] = statistics.mean(values)
                    values_by_metric.setdefault(metric, []).extend(values)
                print(f"{mode}: counter group {index + 1}/{len(monitor_groups)} collected", flush=True)
            receipt["counter_means"] = {name: statistics.mean(values)
                                         for name, values in values_by_metric.items()}
            if len(monitor_groups) == 1:
                group = monitor_groups[0]
                receipt.update(monitor_command=group["command"], monitor_exit=group["exit"],
                               monitor_started_unix=group["started_unix"],
                               monitor_finished_unix=group["finished_unix"],
                               counter_sample_count_after_warmup=group["sample_count_after_warmup"])
            receipt["target_exit"] = target.wait(
                timeout=max(30, capture_deadline - time.monotonic() + 30))
            if receipt["target_exit"]:
                raise RuntimeError(f"Target failed state qualification: {log_path}")
            record = json.loads(Path(str(prefix) + f".{phase}.json").read_text())
            assert record["profile_state_exact"] and record["profile"] is None
            execution = record["execution"]
            assert not execution["appended"] and not execution["appended_fallback"]
            if workers is not None:
                assert all(r["workers"] and not r["fallback_reason"]
                           for r in execution["regions"])
                if whole_graph:
                    assert execution["compute_dispatches"] == 1
            receipt["state_sha256"] = sha256(Path(str(prefix) + f".{phase}.state.bin"))
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
    parser.add_argument("--whole-graph", action="store_true")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--metrics", nargs="+",
                           help="SDK metric names that can be collected in one pass")
    selection.add_argument("--metric-group", nargs="+", action="append",
                           help="repeat for sequential single-pass sets during one resident replay")
    args = parser.parse_args()
    metric_groups = args.metric_group or [args.metrics or METRICS]
    assert all(1 <= workers <= 65535 for workers in args.workers)
    assert len(set(args.workers)) == len(args.workers)
    repo = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=False)
    reference, reference_record = capture(repo, args.sdk.resolve(), args.output.resolve(), args.phase, None,
                                          metric_groups=metric_groups)
    for workers in args.workers:
        receipt, record = capture(repo, args.sdk.resolve(), args.output.resolve(), args.phase, workers,
                                  args.whole_graph, metric_groups)
        assert record["implementation_keys"] == reference_record["implementation_keys"]
        assert record["state_layout"] == reference_record["state_layout"]
        assert receipt["state_sha256"] == reference["state_sha256"], "Backend state differs"
        print(f"{workers} workers: identical state and arithmetic; capture qualified", flush=True)


if __name__ == "__main__":
    main()
