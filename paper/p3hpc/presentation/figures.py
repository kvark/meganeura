# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib==3.11.2"]
# ///
"""Figures from the final cohort CSV and the separately versioned warmup study."""
import ast
import csv
import math
from pathlib import Path
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch, Rectangle

HERE = Path(__file__).resolve().parent
INPUT = HERE / "data"
OUTPUT = HERE / "figures"
INK, TEAL, ORANGE, CANVAS, MUTED = "#23343b", "#006b70", "#a44616", "#faf9f6", "#637679"
DEVICES = {"nvidia-5070": "RTX 5070", "nvidia-h100": "H100", "nvidia-3050": "RTX 3050",
           "amd-7900xt": "RX 7900 XT", "amd-mi300": "MI300X", "intel-b570": "Arc B570", "apple-m3": "Apple M3"}
MODELS = ["SmolLM2-135M", "SmolVLA", "StableDiffusion", "ResNet-50", "Whisper-tiny"]
SHORT = ["SmolLM2", "SmolVLA", "U-Net", "ResNet", "Whisper"]
PHASES = ["inference", "latency", "training"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "figure.facecolor": CANVAS, "axes.facecolor": CANVAS,
    "svg.fonttype": "none", "svg.hashsalt": "meganeura-p3hpc-final"})
ROWS = list(csv.DictReader((INPUT / "results.csv").open()))
DATA = {(r["archive"], r["model"], r["precision"], r["phase"]): r for r in ROWS}
assert len(DATA) == 372
assert sum(bool(r["ratio"]) for r in ROWS) == 226


def save(fig, name):
    OUTPUT.mkdir(exist_ok=True)
    fig.savefig(OUTPUT / name, metadata={"Date": None})
    plt.close(fig)


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#d1dad8", linewidth=.7)
    ax.set_axisbelow(True)


def ratio(device, model, profile, phase):
    raw = DATA[device, model, profile, phase]["ratio"]
    return float(raw) if raw else None


def shade(value):
    """Teal below parity, brown above, deeper with distance from 1 (capped at 4×)."""
    if value is None:
        return to_rgb("#e9ebe8"), MUTED
    amount = .12 + .6 * min(abs(math.log2(value)) / 2, 1)
    base = to_rgb(TEAL if value < 1 else ORANGE)
    fill = tuple(1 - amount + amount * c for c in base)
    return fill, "#ffffff" if amount > .5 else INK


def ratio_panels(profile):
    """Every core comparison of one profile: one panel per phase, one cell per GPU and model."""
    fig = plt.figure(figsize=(11.56, 4.38))
    left, width, gap = .095, .285, .012
    for panel, (phase, title) in enumerate(zip(PHASES, ["Inference", "Minimal shape", "Training"])):
        ax = fig.add_axes((left + panel * (width + gap), .17, width, .64))
        for row, device in enumerate(DEVICES):
            for col, model in enumerate(MODELS):
                value = ratio(device, model, profile, phase)
                fill, ink = shade(value)
                ax.add_patch(Rectangle((col + .04, row + .06), .92, .88, facecolor=fill, linewidth=0))
                ax.text(col + .5, row + .5, "—" if value is None else f"{value:.2f}",
                        ha="center", va="center", fontsize=12.5, color=ink)
        ax.set(xlim=(0, 5), ylim=(7, 0))
        ax.set_xticks([c + .5 for c in range(5)], SHORT, fontsize=9.5)
        ax.xaxis.tick_top()
        ax.set_yticks([r + .5 for r in range(7)], list(DEVICES.values()) if panel == 0 else [""] * 7, fontsize=12)
        ax.tick_params(length=0, pad=4)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, loc="left", fontsize=14, weight="bold", pad=24)
    fig.text(left, .085, "Meganeura time ÷ PyTorch time.  Teal: Meganeura faster.  —: PyTorch did not pass its checks.",
             fontsize=11.5, weight="bold")
    fig.text(left, .03, "Minimal shape: one token, action or image. Training: forward, loss and backward. "
             "Apple M3 times vary by a median 34% between processes.", fontsize=10, color=MUTED)
    save(fig, f"{profile}-ratios.svg")


def precision_gain():
    """How much faster each engine gets when reduced precision is allowed."""
    fig, axes = plt.subplots(1, 2, figsize=(11.56, 4.38), sharey=True)
    fig.subplots_adjust(left=.105, right=.985, top=.84, bottom=.2, wspace=.06)
    for ax, (phase, title) in zip(axes, [("inference", "Inference"), ("training", "Training")]):
        for i, device in enumerate(DEVICES):
            # Same workloads for both engines: both profiles passed for each.
            models = [m for m in MODELS if all(DATA[device, m, p, phase][k]
                      for p in ("strict", "accelerated") for k in ("native_ms", "reference_ms"))]
            for offset, key, color in ((-.2, "reference_ms", ORANGE), (.2, "native_ms", TEAL)):
                value = statistics.median(float(DATA[device, m, "strict", phase][key])
                                          / float(DATA[device, m, "accelerated", phase][key]) for m in models)
                ax.barh(i + offset, value, height=.38, color=color)
                ax.text(value + .04, i + offset, f"{value:.2f}×", va="center", fontsize=10.5, color=color)
        ax.axvline(1, color=MUTED, linestyle="--", linewidth=1)
        ax.set_xlim(0, 3.75)
        ax.set_xticks([0, 1, 2, 3], ["0", "1×", "2×", "3×"], fontsize=11)
        ax.set_title(title, loc="left", weight="bold", fontsize=14)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(range(7), list(DEVICES.values()), fontsize=12)
    axes[0].invert_yaxis()
    axes[0].tick_params(axis="y", length=0)
    fig.legend(handles=[Patch(color=ORANGE, label="PyTorch"), Patch(color=TEAL, label="Meganeura")],
               loc="upper left", bbox_to_anchor=(.098, 1), ncol=2, frameon=False, fontsize=12.5)
    fig.text(.45, .95, "Speedup when reduced precision is allowed. Right of the dashed line: faster.", fontsize=12, va="center")
    fig.text(.105, .075, "Strict time ÷ accelerated time, median over the workloads where both engines passed in both profiles.",
             fontsize=10.5)
    fig.text(.105, .03, "Five workloads per GPU, except H100 (four), RX 7900 XT training (four) and MI300X (three; two for training).",
             fontsize=10, color=MUTED)
    save(fig, "precision-gain.svg")


def scaling():
    fig, axes = plt.subplots(1, 2, figsize=(11.56, 4.38))
    fig.subplots_adjust(left=.08, right=.98, top=.83, bottom=.25, wspace=.28)
    for ax, d in zip(axes, ["nvidia-h100", "amd-mi300"]):
        for phase, title, color, lift in [("inference", "Prefill", TEAL, -17), ("training", "Training", ORANGE, 8)]:
            rs = [DATA[d if size == "135M" else d + "-large", "SmolLM2-" + size, "strict", phase] for size in ["135M", "360M", "1.7B"]]
            vals = [float(r["ratio"]) for r in rs]
            ax.plot(range(3), vals, "o-", color=color, label=title, linewidth=2)
            for i, (r, v) in enumerate(zip(rs, vals)):
                spread = ast.literal_eval(r["paired_ratios"])
                ax.vlines(i, min(spread), max(spread), color=color, linewidth=3)
                ax.annotate(f"{v:.2f}", (i, v), xytext=(0, lift), textcoords="offset points", ha="center", color=color, fontsize=12)
        ax.set_title(DEVICES[d], loc="left", weight="bold")
        ax.set_xticks(range(3), ["135M", "360M", "1.7B"])
        ax.set_ylim(.9, 4.3)
        ax.axhline(1, color=MUTED, linestyle="--", linewidth=1)
        clean(ax)
    axes[0].set_ylabel("Strict time ratio · lower is better")
    axes[1].legend(frameon=False, loc="upper left")
    fig.text(.08, .07, "Dots: Meganeura time ÷ PyTorch time. Bars: range over the three process pairs.", fontsize=11)
    save(fig, "scaling.svg")


def preparation():
    rows = list(csv.DictReader((INPUT / "preparation-cache.csv").open()))
    assert len(rows) == 7
    fig, ax = plt.subplots(figsize=(11.56, 4.38))
    fig.subplots_adjust(left=.16, right=.93, top=.94, bottom=.19)
    for shift, index, color, name in [(-.18, "first_seconds", TEAL, "First-session memo miss (10 processes/GPU)"), (.18, "reused_seconds", ORANGE, "Reused decisions (20 processes/GPU)")]:
        values = [float(r[index]) for r in rows]
        ax.barh([i+shift for i in range(7)], values, height=.32, color=color, label=name)
        for i, v in enumerate(values): ax.text(v+.5, i+shift, f"{v:.1f}", va="center", fontsize=10, color=color)
    ax.set_yticks(range(7), [r["device"] for r in rows]);ax.invert_yaxis()
    ax.set_xlim(0, 85);ax.set_xlabel("Median native preparation, seconds per workload process")
    ax.legend(loc="lower left", bbox_to_anchor=(0, -.30), ncol=2, frameon=False, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "preparation.svg")


def preparation_models():
    rows = list(csv.DictReader((INPUT / "preparation.csv").open()))
    fig, ax = plt.subplots(figsize=(11.56, 4.38))
    fig.subplots_adjust(left=.20, right=.96, top=.88, bottom=.23)
    for i, model in enumerate(MODELS):
        devices = defaultdict(list)
        for row in rows:
            if row["model"] == model and row["precision"] == "strict" and row["reference_complete"] == "True":
                devices[row["archive"]].append(row)
        assert len(devices) == (6 if model in ("ResNet-50", "Whisper-tiny") else 7)
        assert all(len(group) == 3 for group in devices.values())
        values = [statistics.median(statistics.median(float(r[key]) for r in group)
                                   for group in devices.values())
                  for key in ("native_preparation_seconds", "reference_compile_seconds")]
        for offset, value, color in zip((-.18, .18), values, (TEAL, ORANGE)):
            ax.barh(i+offset, value, height=.31, color=color)
            ax.text(value+1, i+offset, f"{value:.1f} s", va="center", fontsize=12, color=color)
    ax.set_yticks(range(5), ["SmolLM2-135M", "SmolVLA", "SD U-Net", "ResNet-50", "Whisper-tiny"])
    ax.invert_yaxis(); ax.set_xlim(0, 80)
    ax.set_xlabel("Seconds per workload process · strict f32")
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(.20, .95, "Meganeura: all preparation, almost all of it search", color=TEAL, fontsize=13, weight="bold")
    fig.text(.20, .89, "PyTorch: compilation and first calls", color=ORANGE, fontsize=13, weight="bold")
    fig.text(.20, .065, "Median across GPUs of three-process medians. ResNet and Whisper: 6 GPUs with complete PyTorch results.", fontsize=10)
    fig.text(.20, .018, "The columns cover different work under different limits.", fontsize=10)
    save(fig, "preparation-models.svg")


def compiler_map():
    """The compiler's route, top row left to right, then the bottom row back."""
    fig, ax = plt.subplots(figsize=(11.56, 4.38))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set(xlim=(0, 1156), ylim=(438, 0)); ax.axis("off")
    from matplotlib.patches import FancyBboxPatch
    stops = [
        ("Describe", "Rust / ONNX / NNEF / GGUF", "Typed ops: primitive,", "composite, private"),
        ("Rewrite", "egglog e-graph", "Forward alternatives", "+ matrix schedules"),
        ("Differentiate", "One backward per forward", "Then explore backward layouts", "Inference skips this stop"),
        ("Plan", "Lower + fuse operators", "Order dispatches", "Reuse buffers by lifetime"),
        ("Specialize", "Shape + device + precision", "Scalar, subgroup or", "workgroup matrix kernels"),
        ("Compile", "WGSL → Naga → Blade", "Vulkan / Metal pipelines", "on the target device"),
        ("Check + time", "Initialize → check → tune", "Race complete programs", "Keep the fastest that passes"),
        ("Execute", "Static plan, resident tensors", "Record → submit → wait", "Share the application's queue"),
    ]
    points = [(20+285*i, 25) for i in range(4)] + [(20+285*i, 219) for i in reversed(range(4))]
    for i, ((x, y), stop) in enumerate(zip(points, stops)):
        ax.add_patch(FancyBboxPatch((x, y), 261, 145, boxstyle="round,pad=0,rounding_size=8",
                                   facecolor="#e7eeeb", edgecolor="#b6cdca", linewidth=1))
        ax.text(x+14, y+26, f"{i+1:02d}", fontsize=13, weight="bold", color=ORANGE, va="center")
        ax.text(x+51, y+26, stop[0], fontsize=15, weight="bold", color=TEAL, va="center")
        for j, line in enumerate(stop[1:]):
            ax.text(x+14, y+63+26*j, line, fontsize=11, va="center")
        if i < 3:
            a, b = (x+264, y+75), (x+281, y+75)
        elif i == 3:
            a, b = (x+130, y+149), (x+130, y+190)
        elif i < 7:
            a, b = (x-3, y+75), (x-21, y+75)
        else:
            continue
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color=TEAL, lw=2))
    ax.text(578, 194, "Search repeats 02–07: layouts, kernels, fusion and submission count", ha="center", fontsize=11, color=ORANGE)
    ax.text(20, 396, "Compiler tests: CPU f64 interpreter + finite differences", fontsize=12, weight="bold")
    ax.text(20, 426, "Benchmark harness: Inferena checks both engines, then warms up and times the selected programs", fontsize=11)
    save(fig, "compiler-map.svg")


def timing_spread():
    fig, ax = plt.subplots(figsize=(11.56, 4.38))
    fig.subplots_adjust(left=.09, right=.98, top=.94, bottom=.23)
    for engine, color, offset in [("native", TEAL, -.14), ("reference", ORANGE, .14)]:
        for i, d in enumerate(DEVICES):
            values = []
            for r in ROWS:
                if r["archive"] != d: continue
                ps = ast.literal_eval(r[engine + "_process_medians_ms"])
                if len(ps) == 3: values.append(100*(max(ps)-min(ps))/statistics.median(ps))
            ax.scatter([i+offset]*len(values), values, color=color, alpha=.65, s=18, label=engine if i == 0 else None)
    ax.set_xticks(range(7), list(DEVICES.values()), rotation=10, ha="right", fontsize=11)
    ax.set_yscale("log");ax.set_ylim(.02, 250)
    ax.set_ylabel("Process spread, % (log scale)")
    ax.axhline(10, color="#637679", linestyle="--", linewidth=1)
    ax.legend(frameon=False, loc="upper left")
    clean(ax)
    fig.text(.09, .035, "Spread = (max − min) / median of 3 process medians. Qualified timed cells; all repeats retained.", fontsize=11)
    save(fig, "timing-spread.svg")


def warmup():
    rows = list(csv.DictReader((INPUT / "warmup.csv").open()))
    fig, axes = plt.subplots(1, 2, figsize=(11.56, 4.38))
    fig.subplots_adjust(left=.08, right=.98, top=.85, bottom=.24, wspace=.25)
    for ax, gpu, title in zip(axes, ["rtx5070", "b570"], ["RTX 5070", "Arc B570"]):
        for phase, color in zip(PHASES, [TEAL, "#6a5b98", ORANGE]):
            for rep, style in [("1", "o-"), ("2", "s--")]:
                rs = [r for r in rows if r["gpu"] == gpu and r["phase"] == phase and r["replicate"] == rep]
                ax.plot([float(r["checkpoint_seconds"]) for r in rs], [float(r["relative_to_60s_percent"]) for r in rs], style,
                        color=color, label=phase if rep == "1" else None, linewidth=1.7, markersize=5)
        ax.set_title(title, loc="left", weight="bold");ax.set_xticks([5, 15, 30, 60]);ax.set_ylim(-3.4, 1.4)
        ax.set_xlabel("Cumulative post-search warmup, seconds");clean(ax)
    axes[0].set_ylabel("Time relative to own 60 s result, %")
    axes[1].legend(frameon=False, fontsize=10, loc="lower right")
    fig.text(.08, .035, "Two processes/GPU (solid / dashed). Same selected program; earlier timing windows add elapsed heat.", fontsize=11)
    save(fig, "warmup.svg")


if __name__ == "__main__":
    ratio_panels("strict")
    ratio_panels("accelerated")
    precision_gain()
    scaling();preparation();preparation_models();compiler_map();timing_spread();warmup()
