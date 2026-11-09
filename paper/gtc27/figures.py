# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib==3.11.2"]
# ///
"""GTC poster figures: NVIDIA rows of the P3HPC cohort and the RTX 5070 single-op study."""
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
P3HPC = HERE.parent / "p3hpc"
OUTPUT = HERE / "figures"
INK, TEAL, ORANGE, CANVAS, MUTED, GRID = "#23343b", "#006b70", "#a44616", "#ffffff", "#637679", "#d1dad8"
DEVICES = {"nvidia-5070": "RTX 5070", "nvidia-h100": "H100 SXM"}
MODELS = ["SmolLM2-135M", "SmolVLA", "StableDiffusion", "ResNet-50", "Whisper-tiny"]
SHORT = ["SmolLM2-135M", "SmolVLA expert", "SD U-Net", "ResNet-50", "Whisper enc."]
PHASES = [("inference", "Prefill /\nbatch"), ("latency", "Minimal\nshape"), ("training", "Training\nstep")]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 22, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "figure.facecolor": CANVAS, "axes.facecolor": CANVAS,
    "svg.fonttype": "none", "svg.hashsalt": "meganeura-gtc27"})
ROWS = list(csv.DictReader((P3HPC / "tables" / "results.csv").open()))
DATA = {(r["archive"], r["model"], r["precision"], r["phase"]): r for r in ROWS}


def save(fig, name):
    OUTPUT.mkdir(exist_ok=True)
    fig.savefig(OUTPUT / name, metadata={"Date": None})
    plt.close(fig)


def clean(ax, axis="x"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=axis, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def times(device, model, profile, phase):
    r = DATA[device, model, profile, phase]
    if not r["ratio"]:
        return None
    return float(r["native_ms"]), float(r["reference_ms"])


def shade(value):
    """Teal below parity, brown above, deeper with distance from 1 (capped at 4×)."""
    if value is None:
        return "#eceeec", MUTED
    depth = min(abs(math.log(value)) / math.log(4), 1)
    base = to_rgb(TEAL if value < 1 else ORANGE)
    fill = tuple(1 - (1 - c) * (.12 + .78 * depth) for c in base)
    return fill, ("white" if depth > .55 else INK)


def ratio_grid():
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.4), gridspec_kw={"wspace": .06})
    for ax, (device, label) in zip(axes, DEVICES.items()):
        for row, model in enumerate(MODELS):
            for col, (profile, phase) in enumerate((p, ph) for p in ("strict", "accelerated") for ph, _ in PHASES):
                pair = times(device, model, profile, phase)
                value = pair[0] / pair[1] if pair else None
                fill, ink = shade(value)
                x = col + (.25 if col >= 3 else 0)
                ax.add_patch(Rectangle((x + .04, row + .05), .92, .9, facecolor=fill, linewidth=0))
                ax.text(x + .5, row + .5, f"{value:.2f}" if value else "n/q", ha="center", va="center",
                        fontsize=21, color=ink, weight="bold" if value and value < 1 else "normal")
        ax.set_xlim(0, 6.25)
        ax.set_ylim(len(MODELS), -1.15)
        ax.set_xticks([c + .5 + (.25 if c >= 3 else 0) for c in range(6)],
                      [name for _ in range(2) for _, name in PHASES], fontsize=15)
        ax.set_yticks([r + .5 for r in range(len(MODELS))], SHORT if ax is axes[0] else [])
        ax.tick_params(length=0)
        ax.spines[:].set_visible(False)
        ax.text(3.125, -1.0, label, ha="center", va="center", fontsize=25, weight="bold", color=INK)
        ax.text(1.5, -.45, "strict f32", ha="center", va="center", fontsize=19, color=MUTED)
        ax.text(4.75, -.45, "reduced precision", ha="center", va="center", fontsize=19, color=MUTED)
    fig.subplots_adjust(left=.19, right=.995, top=.99, bottom=.13)
    save(fig, "ratios.svg")


def geomean(values):
    return math.exp(sum(map(math.log, values)) / len(values))


def precision_gains():
    """Speedup each engine gets from its own reduced-precision path over strict f32, same conditions."""
    groups = []
    for device, label in DEVICES.items():
        for phase, name in PHASES:
            torch, native = [], []
            for model in MODELS:
                strict, accel = times(device, model, "strict", phase), times(device, model, "accelerated", phase)
                if strict and accel:
                    native.append(strict[0] / accel[0])
                    torch.append(strict[1] / accel[1])
            groups.append((f"{label}\n{name.replace(chr(10), ' ')}", geomean(torch), geomean(native), len(torch)))
    fig, ax = plt.subplots(figsize=(13.2, 6.6))
    height = .36
    for i, (name, torch, native, n) in enumerate(groups):
        y = i + (.5 if i >= 3 else 0)
        ax.barh(y - height / 2 - .02, torch, height, color=MUTED, left=0)
        ax.barh(y + height / 2 + .02, native, height, color=TEAL, left=0)
        ax.text(torch + .04, y - height / 2 - .02, f"{torch:.2f}×", va="center", fontsize=19, color=INK)
        ax.text(native + .04, y + height / 2 + .02, f"{native:.2f}×", va="center", fontsize=19, color=INK)
    ax.set_yticks([i + (.5 if i >= 3 else 0) for i in range(len(groups))], [g[0] for g in groups], fontsize=17)
    ax.invert_yaxis()
    ax.axvline(1, color=INK, linewidth=1.5)
    ax.set_xlim(0, 3.4)
    ax.set_xlabel("Speedup from reduced precision (geometric mean, same qualified models)", fontsize=18)
    clean(ax)
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[Rectangle((0, 0), 1, 1, color=MUTED), Rectangle((0, 0), 1, 1, color=TEAL)],
              labels=["PyTorch: autocast BF16", "Meganeura: searched precision policy"],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=17)
    fig.subplots_adjust(left=.2, right=.98, top=.93, bottom=.13)
    save(fig, "precision-gains.svg")


# RTX 5070 ratios at Meganeura 6b1e784, before split-K, cached Winograd weight
# transforms and the 128x64 tile (op-bench.tsv as of commit d30ca390).
BEFORE = {"square-1024": 2.03, "square-2048": 2.02, "square-4096": 2.06, "unet-ff-down": 4.40,
          "unet-ctx-kv": 3.41, "unet-3x3-256@8": .70}
OPS = [
    ("Square GEMM", [("square-1024", "1024³"), ("square-2048", "2048³"), ("square-4096", "4096³")]),
    ("Transformer GEMM", [("whisper-qkv", "Whisper QKV 1500×384→1152"), ("whisper-ff-up", "Whisper FF up"),
                          ("whisper-ff-down", "Whisper FF down"), ("llm-ff-up", "SmolLM2 FF up 512×576→1536")]),
    ("Small GEMM (U-Net)", [("unet-ff-up", "64×256→1024"), ("unet-ff-down", "64×1024→256 (long K)"),
                            ("unet-ctx-kv", "77×768→256 (long K)"), ("unet-proj-16", "256×128→128")]),
    ("3×3 conv (Winograd)", [("unet-3x3-64@32", "64 ch @ 32²"), ("unet-3x3-128@16", "128 ch @ 16²"),
                             ("unet-3x3-256@8", "256 ch @ 8²"), ("resnet-3x3-64@56", "ResNet 64 ch @ 56²"),
                             ("resnet-3x3-512@7", "ResNet 512 ch @ 7²")]),
    ("Other conv", [("unet-stem-4@32", "U-Net stem 4→320"), ("resnet-1x1-256@56", "ResNet 1×1 256 ch"),
                    ("resnet-stem-7x7", "ResNet stem 7×7")]),
    ("Dispatch", [("add-256", "Dependent 256-add, per op"), ("add-256-single", "Submit + wait, one add")]),
]


def single_ops():
    rows = {}
    for line in (P3HPC / "presentation" / "data" / "op-bench.tsv").open():
        if line.startswith(("#", "gpu\t")):
            continue
        gpu, engine, _, shape, gpu_ms, wall_ms, *_ = line.rstrip("\n").split("\t")
        if gpu == "rtx5070":
            # The single-submit row has no device time; it compares host round trips.
            rows[shape, engine] = float(wall_ms if shape == "add-256-single" else gpu_ms)
    fig, ax = plt.subplots(figsize=(13.2, 12.8))
    y, ticks, labels = 0, [], []
    for group, shapes in OPS:
        ax.text(-.02, y + .1, group, fontsize=19, weight="bold", color=INK, va="center", ha="right",
                transform=ax.get_yaxis_transform())
        y += .8
        for shape, label in shapes:
            value = rows[shape, "meganeura"] / rows[shape, "pytorch"]
            color = TEAL if value < 1 else ORANGE
            if shape in BEFORE:
                ax.annotate("", xy=(value, y), xytext=(BEFORE[shape], y),
                            arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=2, shrinkA=7, shrinkB=9, mutation_scale=22))
                ax.plot(BEFORE[shape], y, "o", ms=13, mfc=CANVAS, mec=MUTED, mew=2.2)
            ax.plot(value, y, "o", ms=15, color=color, mec=CANVAS, mew=2)
            # Label on the side away from the earlier value's arrow.
            left = shape in BEFORE and BEFORE[shape] > value
            ax.text(value * (.9 if left else 1.11), y, f"{value:.2f}", va="center",
                    ha="right" if left else "left", fontsize=18, color=INK)
            ticks.append(y)
            labels.append(label)
            y += .9
        y += .3
    ax.set_xscale("log")
    ax.set_xlim(.115, 6)
    ax.set_xticks([.125, .25, .5, 1, 2, 4], ["⅛", "¼", "½", "1", "2", "4"])
    ax.minorticks_off()
    ax.axvline(1, color=INK, linewidth=1.5)
    ax.set_yticks(ticks, labels, fontsize=17)
    ax.set_ylim(y - .2, -.6)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Meganeura time / PyTorch time  (log scale; left of 1 is faster)", fontsize=18)
    clean(ax)
    ax.plot([], [], "o", ms=13, mfc=CANVAS, mec=MUTED, mew=2.2, label="before split-K, cached Winograd weights, 128×64 tile")
    ax.plot([], [], "o", ms=15, color=INK, label="now")
    ax.legend(loc="lower left", bbox_to_anchor=(-.02, 1.0), ncol=2, frameon=False, fontsize=15.5,
              handletextpad=.2, columnspacing=1)
    fig.subplots_adjust(left=.36, right=.98, top=.955, bottom=.065)
    save(fig, "single-ops.svg")


ratio_grid()
precision_gains()
single_ops()
