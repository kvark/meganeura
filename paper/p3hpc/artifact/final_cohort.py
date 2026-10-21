#!/usr/bin/env python3
"""Audit the final v33/v15 archives and regenerate publication inputs, without a GPU.

Use the frozen Inferena checkout for qualification: do not reimplement or relax
its gates in the paper analyzer. All performance ratios here are Meganeura / PyTorch.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics as stats
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
INFERENA = "add19182e5479e3f7549894c524ac6691d3dc363"
MEGANEURA = "c19cb1caedd90be7abc61de61f6197018711398e"
DEVICES = {"nvidia-5070": "RTX 5070", "nvidia-h100": "H100", "nvidia-3050": "RTX 3050",
           "amd-7900xt": "RX 7900 XT", "amd-mi300": "MI300X", "intel-b570": "Arc B570", "apple-m3": "Apple M3"}
QUALIFIERS = {"amd-780m": "Radeon 780M", "amd-9600x": "Ryzen 9600X iGPU",
              "intel-rplu": "Intel RPL-U", "qualcomm-adreno": "Quest 3S / Adreno 740"}
MODELS = ("SmolLM2-135M", "SmolVLA", "StableDiffusion", "ResNet-50", "Whisper-tiny")
PHASES = ("inference", "latency", "training")


def seconds(value):
    return value["secs"] + value["nanos"] * 1e-9


def table(columns, heading, lines):
    return "\n".join([r"\begin{tabular}{" + columns + "}", r"\toprule", heading + r" \\",
                      r"\midrule", *[" & ".join(row) + r" \\" for row in lines], r"\bottomrule", r"\end{tabular}", ""])


def ratio(value):
    if value is None:
        return "--"
    return (r"\textbf{" + f"{value:.2f}" + "}") if value < 1 else f"{value:.2f}"


def label(model):
    return "SD U-Net" if model == "StableDiffusion" else model


def audit(directories):
    from p3hpc import check_native_record, check_pair
    from p3hpc_android import check_native as check_android
    from p3hpc_report import cohort_rows, construction_rows, read_record, archive_path

    rows = cohort_rows(directories)
    campaigns, setups, searches, processes, failures = {}, [], [], [], []
    checks, hashes = Counter(), defaultdict(set)
    for directory in directories:
        name = directory.parent.name
        c = json.loads((directory / "campaign.json").read_text())
        campaigns[name] = c
        args = SimpleNamespace(**c["args"])
        android = c["protocol"] == "p3hpc-android-campaign-v15"
        assert android or c["protocol"] == "p3hpc-paired-campaign-v33"
        assert c["source"] == INFERENA and c["meganeura"]["rev"] == MEGANEURA
        assert c["warmup"]["minimum_seconds"] == (2 if android else 10)
        if not android:
            assert c["native_policy"]["search_seconds_per_session"] == 30
        if not android and c["status"] != "incomplete":
            assert c["final_binary_check"]["status"] == "pass"
        for key, value in c["sha256"].items():
            hashes[key.replace("\\", "/")].add(value)
        local_setups = {(s["model"], s["precision"], s["replicate"]): dict(archive=name, **s)
                        for s in construction_rows(directory)}
        setups.extend(local_setups.values())
        count = 20 if args.collect else 1
        for run in c["runs"]:
            identity = dict(archive=name, model=run["model"], precision=run["precision"], replicate=run["replicate"])
            folder = archive_path(directory, run["path"])
            m, p = (read_record(directory, run, engine) for engine in ("meganeura", "pytorch"))
            setup = local_setups[run["model"], run["precision"], run["replicate"]]
            setup["reference_compile_seconds"] = p.get("timings", {}).get("compile_s")
            setup["reference_complete"] = all(run.get("phases", {}).get(phase, {}).get("status") == "valid"
                                              for phase in PHASES)
            phase_setup = p.get("execution", {}).get("graph_replay", {}).get("phases", {})
            setup["reference_phase_preparation_seconds"] = (
                sum(v["preparation_s"] for v in phase_setup.values())
                if setup["reference_complete"] and all("preparation_s" in v for v in phase_setup.values()) else None)
            if android:
                check_android(m, args, run["model"], run["precision"], count, c["meganeura"], c["native_device"])
                oracle = json.loads((folder / "comparison-float64-eager.json").read_text())
                assert oracle["passed"] is True and set(oracle["phases"]) == set(PHASES)
                for phase, details in oracle["phases"].items():
                    path = folder / f"{phase}.json"
                    capture = json.loads(path.read_text())
                    assert details["manifest_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
                    expected = {"output:0"} if phase != "training" else {"loss", *("gradient:" + k for k in capture["gradients"])}
                    assert set(details["checks"]) == expected and details["passed"] is True
                    assert all(v["passed"] and v["finite"] for v in details["checks"].values())
                    if phase == "training":
                        assert details["all_gradients"]["passed"] is True
                checks["android_full_tensor_receipts"] += 1
            else:
                check_native_record(m, args, count, c["source"], run["precision"])
                assert len(m["framework_rev"]) >= 7 and MEGANEURA.startswith(m["framework_rev"])
                for stage in ("before", "after"):
                    assert run["binary_checks"][stage]["status"] == "pass"
                assert run["binary_checks"]["before"]["sha256"] == run["binary_checks"]["after"]["sha256"]
                checks["native_receipts"] += 1
                for phase, value in run.get("phases", {}).items():
                    if value["status"] == "valid":
                        check_pair([m, p], args, run["mode"], run["graphs"], count, c["source"],
                                   replicated=args.collect, precision=run["precision"], phases=(phase,))
                        checks["paired_phase_receipts"] += 1
            if run["status"] != "valid":
                failures.append(dict(**identity, status=run["status"],
                                     phases={k: v["status"] for k, v in run.get("phases", {}).items()},
                                     error=p.get("error", "")[-500:]))
            for engine, record in (("native", m), ("reference", p)):
                for phase, samples in record.get("timing_samples_ms", {}).items():
                    if not samples:
                        continue
                    assert len(samples) == count and all(math.isfinite(v) and v > 0 for v in samples)
                    processes.append(dict(**identity, engine=engine, phase=phase, median_ms=stats.median(samples), collect=args.collect))
            for session in m["optimizer"]["sessions"]:
                search = session["search"]
                selected = search["trials"][search["selected"]]
                reused = (selected.get("kernel_tuning") or {}).get("reused_classes", [])
                searches.append(dict(**identity, seconds=seconds(search["elapsed"]), trials=len(search["trials"]),
                    truncated=search["truncated"], selected=selected["description"],
                    numerical_rejections=sum(t["outcome"]["decision"] == "InvalidOutput" for t in search["trials"]),
                    scratch_limits=sum(t["outcome"]["decision"] == "ScratchLimit" for t in search["trials"]),
                    cooperative_dispatches=session["cooperative_dispatches"],
                    bf16_dispatches=session["bf16_matrix_dispatches"],
                    bf16_shape=session["bf16_subgroup_shape"], bf16_workgroup_tiles=session["bf16_workgroup_tiles"],
                    selected_reused_workgroup_classes=sum("CooperativeWorkgroup" in choice for _, choice in reused),
                    unsupported=search["unsupported_matrix_inputs"]))
    assert all(len(v) == 1 for v in hashes.values()), "input hashes differ"
    assert set(campaigns) == {*DEVICES, *QUALIFIERS, *(d + "-large" for d in ("nvidia-5070", "nvidia-h100", "amd-7900xt", "amd-mi300")), "intel-b570-large-fail"}
    for row in rows:
        row["archive"] = Path(row.pop("campaign")).parent.name
        ps = [p for p in processes if all(p[k] == row[k] for k in ("archive", "model", "precision", "phase"))]
        row["ratio"] = 1 / row["speedup"] if row["speedup"] is not None else None
        for engine in ("native", "reference"):
            values = [p["median_ms"] for p in ps if p["engine"] == engine]
            row[engine + "_process_medians_ms"] = values if row[engine + "_ms"] is not None else []
        ratios = []
        if row["ratio"] is not None:
            for rep in (1, 2, 3):
                values = {p["engine"]: p["median_ms"] for p in ps if p["replicate"] == rep}
                ratios.append(values["native"] / values["reference"])
        row["paired_ratios"] = ratios
    return rows, campaigns, setups, searches, processes, failures, dict(checks)


def publication(rows, campaigns, setups, searches, processes):
    lookup = {(r["archive"], r["model"], r["precision"], r["phase"]): r for r in rows}
    def get(device, model, precision, phase):
        return lookup[device, model, precision, phase]
    output, summary = {}, {"inferena": INFERENA, "meganeura": MEGANEURA}
    lines = []
    for device, name in DEVICES.items():
        for model in MODELS:
            values = []
            for precision in ("strict", "accelerated"):
                for phase in PHASES:
                    r = get(device, model, precision, phase)
                    values.append(ratio(r["ratio"]) if r["ratio"] is not None else rf"{r['native_ms']:.2f}\,ms$^\dagger$")
            lines.append([name if model == MODELS[0] else "", label(model), *values])
    heading = (r"GPU & Workload & \multicolumn{3}{c}{Strict} & \multicolumn{3}{c}{Accelerated} \\"
               r"\cmidrule(lr){3-5}\cmidrule(lr){6-8} & & Inf. & Min. & F+L+B & Inf. & Min. & F+L+B")
    output["ratios.tex"] = table("llrrrrrr", heading, lines)
    for precision in ("strict", "accelerated"):
        summary[precision] = {}
        for phase in PHASES:
            values = [get(d, m, precision, phase)["ratio"] for d in DEVICES for m in MODELS]
            values = [v for v in values if v is not None]
            summary[precision][phase] = dict(median=stats.median(values), pairs=len(values), nominal_wins=sum(v < 1 for v in values))
    lines = []
    for d, name in DEVICES.items():
        c = campaigns[d]
        backend = c["args"]["backend"]
        driver = c["native_device"]["driver_info"].split("-")[0] or "built-in"
        if d == "amd-mi300":
            driver = "patched RADV 8c7f5fdd"
        coverage = [sum(get(d, m, p, phase)["ratio"] is not None for m in MODELS for p in ("strict", "accelerated")) for phase in PHASES]
        lines.append([name, {"apple-m3": "macOS", "nvidia-3050": "Windows"}.get(d, "Linux"),
                      "Metal" if backend == "mps" else "Vulkan", driver, backend.upper(), "/".join(map(str, coverage))])
    output["devices.tex"] = table("lllllr", "GPU & OS & API & Driver & PyTorch & Paired cells (inf./min./train)", lines)
    output["qualification.tex"] = table("lll", "GPU & Native cells & Reference", [
        [name, "30/30", "CPU f64" if d == "qualcomm-adreno" else "CPU f32"] for d, name in QUALIFIERS.items()])
    lines = []
    for d in ("nvidia-5070", "amd-7900xt", "nvidia-h100", "amd-mi300"):
        first = True
        for model in ("SmolLM2-135M", "SmolLM2-360M", "SmolLM2-1.7B"):
            source = d if model.endswith("135M") else d + "-large"
            if (source, model, "strict", "inference") not in lookup:
                continue
            for precision in ("strict", "accelerated"):
                rs = [get(source, model, precision, phase) for phase in PHASES]
                lines.append([DEVICES[d] if first else "", model.removeprefix("SmolLM2-") if precision == "strict" else "",
                              "S" if precision == "strict" else "A", *[f"{r['native_ms']:.2f} ({ratio(r['ratio'])})" for r in rs]])
                first = False
    output["scaling.tex"] = table("lllrrr", "GPU & Size & & Prefill & One token & F+L+B", lines)
    lines, pp = [], []
    for model in MODELS:
        scores = []
        for phase in PHASES:
            rs = [get(d, model, "strict", phase) for d in DEVICES]
            for engine in ("native", "reference"):
                score = 0 if any(r[engine + "_ms"] is None for r in rs) else len(rs) / sum(
                    r[engine + "_ms"] / min(r[e + "_ms"] for e in ("native", "reference") if r[e + "_ms"] is not None) for r in rs)
                scores.append(score)
        pp.append(scores)
        lines.append([label(model), *[f"{v:.2f}" for v in scores]])
    summary["portability_means"] = [stats.mean(v) for v in zip(*pp)]
    lines.append(["Workload mean", *[f"{v:.2f}" for v in summary["portability_means"]]])
    output["portability.tex"] = table("lrrrrrr", r"Workload & \multicolumn{2}{c}{Inference} & \multicolumn{2}{c}{Minimal} & \multicolumn{2}{c}{F+L+B} \\ & M & P & M & P & M & P", lines)
    lines = []
    for d, name in DEVICES.items():
        ss = [s for s in searches if s["archive"] == d]
        lines.append([name, str(len(ss)), str(sum(s["trials"] for s in ss)), str(sum(s["truncated"] for s in ss)),
                      str(sum(not s["selected"].startswith("graph=0,") for s in ss)),
                      str(sum(s["numerical_rejections"] for s in ss)), str(sum(s["scratch_limits"] for s in ss))])
    output["search.tex"] = table("lrrrrrr", r"GPU & Sessions & Trials & Trunc. & Graph $\ne0$ & Invalid & Scratch", lines)
    lines = []
    for d, name in DEVICES.items():
        ss = [s for s in setups if s["archive"] == d]
        cold = [s["compile_seconds"] for s in ss if s["cache_state"] == "cold-first-session"]
        warm = [s["compile_seconds"] for s in ss if s["cache_state"] == "warm-decisions"]
        lines.append([name, str(len(cold)), f"{stats.median(cold):.2f}", str(len(warm)), f"{stats.median(warm):.2f}"])
    output["preparation.tex"] = table("lrrrr", "GPU & Cold n & Median s & Warm n & Median s", lines)
    stream = io.StringIO(newline="")
    fields = ["archive", "model", "precision", "replicate", "cache_state", "native_preparation_seconds",
              "native_search_seconds", "reference_compile_seconds", "reference_phase_preparation_seconds",
              "reference_complete"]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for s in setups:
        if s["archive"] not in DEVICES:
            continue
        writer.writerow({**{key: s[key] for key in fields if key in s},
                         "native_preparation_seconds": s["compile_seconds"],
                         "native_search_seconds": sum(session["search_seconds"] for session in s["sessions"])})
    output["preparation.csv"] = stream.getvalue()
    # Reuse only the historical plot layout; every plotted value comes from this audit.
    import cohort
    legacy_rows = {d: {(p, m, "default-graph1"): {"ratio_" + phase: get(d, m, p, phase)["ratio"] for phase in PHASES}
                       for p in ("strict", "accelerated") for m in MODELS} for d in DEVICES}
    plot_campaigns = {d: {"args": {"backend": "cuda", "eager": False}} for d in DEVICES}
    output["ratio-plot.tex"] = cohort.ratio_plot(plot_campaigns, legacy_rows)
    ranges = []
    for panel, phase in enumerate(PHASES):
        origin = 21 + panel * 57
        for i, d in enumerate(DEVICES):
            for j, m in enumerate(MODELS):
                values = get(d, m, "strict", phase)["paired_ratios"]
                if not values:
                    continue
                assert 0.15 < min(values) <= max(values) < 6
                lo, hi = [origin + math.log(v / 0.15) / math.log(6 / 0.15) * 50 for v in (min(values), max(values))]
                y = -i * 6 + (2 - j) * 0.95
                ranges.append(rf"\draw[black!35,line width=0.25mm] ({lo:.3f},{y:.3f}) -- ({hi:.3f},{y:.3f});")
    output["ratio-plot.tex"] = output["ratio-plot.tex"].replace(r"\end{tikzpicture}", "\n".join(ranges) + "\n" + r"\end{tikzpicture}")
    summary["cells"] = dict(core_native=sum(r["native_status"] == "valid" for r in rows if r["model"] in MODELS),
                            measured_native=sum(r["native_ms"] is not None for r in rows), paired=sum(r["ratio"] is not None for r in rows))
    summary["cache_states"] = dict(Counter(s["cache_state"] for s in setups))
    summary["search"] = dict(sessions=sum(s["archive"] in DEVICES for s in searches),
        trials=sum(s["trials"] for s in searches if s["archive"] in DEVICES),
        numerical_rejections=sum(s["numerical_rejections"] for s in searches if s["archive"] in DEVICES),
        scratch_limits=sum(s["scratch_limits"] for s in searches if s["archive"] in DEVICES),
        minutes=sum(s["seconds"] for s in searches if s["archive"] in DEVICES) / 60)
    summary["accelerated_selection"] = {d: dict(
        sessions=len(ss := [s for s in searches if s["archive"] == d and s["precision"] == "accelerated"]),
        cooperative_sessions=sum(s["cooperative_dispatches"] > 0 for s in ss),
        bf16_sessions=sum(s["bf16_dispatches"] > 0 for s in ss),
        selected_reused_workgroup_sessions=sum(s["selected_reused_workgroup_classes"] > 0 for s in ss)) for d in DEVICES}
    summary["elapsed_device_hours"] = sum(c["elapsed_seconds"] for c in campaigns.values()) / 3600
    summary["spread"] = {}
    for d in DEVICES:
        groups = defaultdict(list)
        for p in processes:
            if p["archive"] == d:
                groups[p["model"], p["precision"], p["phase"], p["engine"]].append(p["median_ms"])
        values = [100 * (max(v) - min(v)) / stats.median(v) for v in groups.values() if len(v) == 3]
        summary["spread"][d] = dict(median_percent=stats.median(values), above_10_percent=sum(v > 10 for v in values), groups=len(values))
    return output, summary


def main():
    if not __debug__:
        raise SystemExit("Run without -O: assertions are part of the artifact audit.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", type=Path)
    parser.add_argument("--inferena", required=True, type=Path, help="clean checkout of the measured Inferena revision")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--check", type=Path, help="compare generated TeX and data with this directory")
    args = parser.parse_args()
    assert subprocess.check_output(["git", "-C", str(args.inferena), "rev-parse", "HEAD"], text=True).strip() == INFERENA
    assert not subprocess.check_output(["git", "-C", str(args.inferena), "diff", "HEAD", "--", "scripts", "models", "frameworks/pytorch/PROTOCOL", ".python-version"]), "qualification sources are modified"
    sys.path.insert(0, str(args.inferena.resolve() / "scripts"))
    with tempfile.TemporaryDirectory(prefix="p3hpc-final-") as temp:
        directories = []
        for line in (HERE / "final-cohort.sha256").read_text().splitlines():
            digest, name = line.split("  ", 1)
            path = args.archives / name
            with path.open("rb") as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == digest, name
            destination = Path(temp) / path.stem
            with tarfile.open(path) as archive:
                archive.extractall(destination, filter="data")
            manifests = list(destination.rglob("campaign.json"))
            assert len(manifests) == 1
            directories.append(manifests[0].parent)
        rows, campaigns, setups, searches, processes, failures, checks = audit(directories)
        output, summary = publication(rows, campaigns, setups, searches, processes)
    summary["receipt_checks"] = checks
    for name, data in (("summary", summary), ("results", rows), ("search-receipts", searches), ("failures", failures)):
        output[name + ".json"] = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output.mkdir(parents=True, exist_ok=True)
    for name, content in output.items():
        if args.check and (name.endswith(".tex") or name == "summary.json" or name == "preparation.csv"):
            assert (args.check / name).read_bytes() == content.encode(), "generated data differ: " + name
        (args.output / name).write_text(content)
    with (args.output / "results.csv").open("w") as stream:
        fields = ["archive", "model", "precision", "phase", "native_status", "reference_status", "native_ms", "reference_ms", "ratio", "native_process_medians_ms", "reference_process_medians_ms", "paired_ratios"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    if args.check:
        assert (args.check / "results.csv").read_bytes() == (args.output / "results.csv").read_bytes(), "generated CSV differs"
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
