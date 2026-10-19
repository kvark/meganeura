# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib==3.11.2"]
# ///
"""Figures from the final cohort CSV and the separately versioned warmup study."""
import ast
import csv
from pathlib import Path
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

HERE = Path(__file__).resolve().parent
INPUT = HERE / "data"
OUTPUT = HERE / "figures"
INK, TEAL, ORANGE, CANVAS = "#23343b", "#006b70", "#a44616", "#faf9f6"
DEVICES = {"nvidia-5070": "RTX 5070", "nvidia-h100": "H100", "nvidia-3050": "RTX 3050",
           "amd-7900xt": "RX 7900 XT", "amd-mi300": "MI300X", "intel-b570": "Arc B570", "apple-m3": "Apple M3"}
MODELS = ["SmolLM2-135M", "SmolVLA", "StableDiffusion", "ResNet-50", "Whisper-tiny"]
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


def heatmap(phase, profile, filename):
    fig = plt.figure(figsize=(11.56, 4.38))
    ax = fig.add_axes((.16, .16, .835, .68))
    values = [[float(r["ratio"]) if (r := DATA[d, m, profile, phase])["ratio"] else None for m in MODELS] for d in DEVICES]
    colors = [[to_rgb("#e2e6e3" if v is None else "#dcece6" if v < 1 else "#f3e2d6") for v in row] for row in values]
    ax.imshow(colors, aspect="auto")
    for i, row in enumerate(values):
        for j, v in enumerate(row):
            ax.text(j, i, "—" if v is None else f"{v:.2f}", ha="center", va="center", fontsize=17, weight="bold",
                    color=INK if v is None else TEAL if v < 1 else ORANGE)
    ax.set_xticks(range(5), ["SmolLM2\n135M", "SmolVLA\nexpert", "SD U-Net", "ResNet-50", "Whisper\nencoder"])
    ax.set_yticks(range(7), list(DEVICES.values()))
    ax.xaxis.tick_top()
    ax.tick_params(axis="both", length=0, pad=8, labelsize=13)
    ax.set_xticks([x - .5 for x in range(6)], minor=True)
    ax.set_yticks([y - .5 for y in range(8)], minor=True)
    ax.grid(which="minor", color=CANVAS, linewidth=4)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.text(.16, .085, "Meganeura / PyTorch time · lower is better", fontsize=13, weight="bold")
    fig.text(.16, .025, "— No qualified paired ratio. M3: large process variation; near-ties are inconclusive.", fontsize=11, color="#637679")
    save(fig, filename)


def scaling():
    fig, axes = plt.subplots(1, 2, figsize=(11.56, 4.38))
    fig.subplots_adjust(left=.08, right=.98, top=.83, bottom=.25, wspace=.28)
    for ax, d in zip(axes, ["nvidia-h100", "amd-mi300"]):
        for phase, title, color in [("inference", "Prefill", TEAL), ("training", "F+L+B", ORANGE)]:
            rs = [DATA[d if size == "135M" else d + "-large", "SmolLM2-" + size, "strict", phase] for size in ["135M", "360M", "1.7B"]]
            vals = [float(r["ratio"]) for r in rs]
            ax.plot(range(3), vals, "o-", color=color, label=title, linewidth=2)
            for i, (r, v) in enumerate(zip(rs, vals)):
                spread = ast.literal_eval(r["paired_ratios"])
                ax.vlines(i, min(spread), max(spread), color=color, linewidth=3)
                ax.annotate(f"{v:.2f}", (i, v), xytext=(0, 8), textcoords="offset points", ha="center", color=color, fontsize=12)
        ax.set_title(DEVICES[d], loc="left", weight="bold")
        ax.set_xticks(range(3), ["135M", "360M", "1.7B"])
        ax.set_ylim(.9, 4.3)
        ax.axhline(1, color="#637679", linestyle="--", linewidth=1)
        clean(ax)
    axes[0].set_ylabel("Strict time ratio · lower is better")
    axes[1].legend(frameon=False, loc="upper left")
    fig.text(.08, .07, "Dots: ratio of median process times. Lines at each dot: paired-process range, not a confidence interval.", fontsize=11)
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
    fig.text(.20, .95, "Meganeura: preparation, including search + checks", color=TEAL, fontsize=13, weight="bold")
    fig.text(.20, .89, "PyTorch: compilation + first calls; checks/capture extra", color=ORANGE, fontsize=13, weight="bold")
    fig.text(.20, .065, "Median across GPUs of 3-process medians. Same complete pairs: 7 GPUs; ResNet / Whisper: 6.", fontsize=10)
    fig.text(.20, .018, "Recorded cache states retained. These columns include different work; budgets are not matched.", fontsize=10)
    save(fig, "preparation-models.svg")


def compiler_map():
    """One numbered route, including the design checkpoints from the story draft."""
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
        ("Qualify + time", "Initialize → check → tune", "Race complete programs", "Keep a qualified winner"),
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
    ax.text(20, 396, "Compiler tests: CPU f64 oracle + finite differences", fontsize=12, weight="bold")
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
    heatmap("inference", "strict", "strict-inference.svg")
    heatmap("training", "strict", "strict-training.svg")
    heatmap("training", "accelerated", "accelerated-training.svg")
    scaling();preparation();preparation_models();compiler_map();timing_spread();warmup()
