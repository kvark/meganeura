"""One GEMM or convolution pass per shape, strict f32, through PyTorch.

Usage: torch_ops.py <shapes.txt> <cuda|xpu> [shape] [--describe]   (ROCm builds use cuda)

Kinds: gemm, conv (forward), dgrad (input gradient) and wgrad (weight
gradient). The gradients call aten.convolution_backward with only that output
enabled, the library call autograd makes for it.

gpu_ms: device events around one replay of a graph of 30 serial operations,
divided by 30; median of 30 replays, no profiler. The interval includes the
device's dispatch gaps between the 30 kernels but no host enqueue gaps.
Graph outputs must match eager execution within 1e-6 of the output's largest
magnitude (last-bit differences occur on XPU), 1e-5 for gradients, whose
reductions a library may split with atomics.
wall_ms: median of 50 ordinary eager calls, each synchronized.
Column 6 lists the vendor kernels from one untimed profiled call, truncated.
Column 7 is a JSON list of the same call's device activities with full names,
which carry the vendor's layout and tile (cuDNN's `nhwckrsc`, `tilesize...`,
Tensile's `MT128x64x8`), the launch geometry the profiler records, and each
one's share of the profiled time, so layout conversions show up separately.

--describe runs the shape a few times without timing or output, for the
vendor logs run_all.sh enables around it, which name the chosen algorithm and
memory formats where the kernel names do not (oneDNN's `gen_conv`).
"""
import json
import os
import statistics
import sys
import tempfile
import time

import torch
from torch.profiler import ProfilerActivity, profile

SERIAL = 30
REPLAYS = 30
ACTIVITIES = ("kernel", "gpu_memcpy", "gpu_memset")
GEOMETRY = ("grid", "block", "register", "shared", "size", "bytes", "occupancy")

dev = sys.argv[2]
describe = "--describe" in sys.argv[3:]
only = next((a for a in sys.argv[3:] if a != "--describe"), None)
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = True
mod = torch.cuda if dev == "cuda" else torch.xpu
activity = ProfilerActivity.CUDA if dev == "cuda" else ProfilerActivity.XPU
Graph = torch.cuda.CUDAGraph if dev == "cuda" else torch.xpu.XPUGraph


def kernels(op):
    # Separate, untimed call: profiling perturbs ROCm execution speed.
    with profile(activities=[activity]) as prof:
        op()
        mod.synchronize()
    names = {e.name[:40] for e in prof.events() if e.device_type.name in ("CUDA", "XPU")}
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "trace.json")
        prof.export_chrome_trace(path)
        with open(path) as f:
            events = json.load(f)["traceEvents"]
    found = {}
    for e in events:
        if e.get("cat", "").lower() not in ACTIVITIES:
            continue
        k = found.setdefault(e["name"], {"name": e["name"], "n": 0, "us": 0.0})
        k["n"] += 1
        k["us"] += float(e.get("dur", 0))
        for key, value in e.get("args", {}).items():
            if any(g in key.lower() for g in GEOMETRY):
                k.setdefault(key, value)
    total = sum(k["us"] for k in found.values()) or 1.0
    detail = sorted(found.values(), key=lambda k: -k["us"])
    for k in detail:
        k["share"] = round(k["us"] / total, 3)
        k["us"] = round(k["us"], 2)
    return "+".join(sorted(names)), json.dumps(detail, separators=(",", ":"))


for line in open(sys.argv[1]):
    if line.startswith("#") or not line.strip():
        continue
    fields = line.split()
    kind, name, d = fields[0], fields[1], list(map(int, fields[2:]))
    if only is not None and name != only:
        continue
    g = torch.Generator().manual_seed(1)
    tolerance = 1e-6
    if kind == "gemm":
        m, k, n = d
        a = torch.randn(m, k, generator=g).to(dev)
        b = torch.randn(k, n, generator=g).to(dev)
        op = lambda: torch.mm(a, b)
        flops = 2 * m * k * n
    else:
        ci, h, w, co, k, s, p = d[:7]
        batch = d[7] if len(d) > 7 else 1
        oh = (h + 2 * p - k) // s + 1
        ow = (w + 2 * p - k) // s + 1
        x = torch.randn(batch, ci, h, w, generator=g).to(dev)
        wt = torch.randn(co, ci, k, k, generator=g).to(dev)
        if kind == "conv":
            op = lambda: torch.nn.functional.conv2d(x, wt, stride=s, padding=p)
        else:
            dy = torch.randn(batch, co, oh, ow, generator=g).to(dev)
            mask = [kind == "dgrad", kind == "wgrad", False]
            index = mask.index(True)
            op = lambda: torch.ops.aten.convolution_backward(
                dy, x, wt, None, [s, s], [p, p], [1, 1], False, [0, 0], 1, mask
            )[index]
            tolerance = 1e-5
        flops = 2 * batch * co * oh * ow * ci * k * k
    with torch.no_grad():
        if describe:
            for _ in range(3):
                op()
            mod.synchronize()
            print(f"{kind}\t{name}\tdescribed", flush=True)
            continue
        start = time.time()
        while time.time() - start < 1.0:
            op()
            mod.synchronize()
        wall = []
        for _ in range(50):
            t = time.perf_counter()
            op()
            mod.synchronize()
            wall.append((time.perf_counter() - t) * 1e3)
        eager = op()
        mod.synchronize()

        # Capture on a side stream after warming it, as graph capture requires.
        side = mod.Stream()
        side.wait_stream(mod.current_stream())
        with mod.stream(side):
            for _ in range(3):
                op()
        mod.current_stream().wait_stream(side)
        graph = Graph()
        with (torch.cuda.graph(graph) if dev == "cuda" else torch.xpu.graph(graph)):
            outs = [op() for _ in range(SERIAL)]
        graph.replay()
        mod.synchronize()
        # oneDNN's accumulation order can change the last bit between calls.
        scale = eager.abs().max().item()
        worst = max((o - eager).abs().max().item() for o in outs)
        if worst > tolerance * scale:
            raise SystemExit(f"{name}: graph output differs from eager by {worst}")
        for _ in range(5):
            graph.replay()
        mod.synchronize()
        times = []
        for _ in range(REPLAYS):
            begin = mod.Event(enable_timing=True)
            end = mod.Event(enable_timing=True)
            begin.record()
            graph.replay()
            end.record()
            end.synchronize()
            times.append(begin.elapsed_time(end) / SERIAL)
    gpu = statistics.median(times)
    names, detail = kernels(op)
    print(
        f"{kind}\t{name}\t{gpu:.4f}\t{statistics.median(wall):.4f}\t"
        f"{flops / gpu / 1e9:.3f}\tgraph-interval:{names}\t{detail}",
        flush=True,
    )
