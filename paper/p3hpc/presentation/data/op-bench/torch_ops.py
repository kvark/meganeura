"""One GEMM or convolution per shape, strict f32, through PyTorch.

Usage: torch_ops.py <shapes.txt> <cuda|xpu> [shape]   (ROCm builds use cuda)

gpu_ms: device events around one replay of a graph of 30 serial operations,
divided by 30; median of 30 replays, no profiler. The interval includes the
device's dispatch gaps between the 30 kernels but no host enqueue gaps.
Graph outputs must match eager execution within 1e-6 of the output's largest
magnitude (last-bit differences occur on XPU).
wall_ms: median of 50 ordinary eager calls, each synchronized.
The last column lists the vendor kernels from one untimed profiled call.
"""
import statistics
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile

SERIAL = 30
REPLAYS = 30

dev = sys.argv[2]
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = True
mod = torch.cuda if dev == "cuda" else torch.xpu
activity = ProfilerActivity.CUDA if dev == "cuda" else ProfilerActivity.XPU
Graph = torch.cuda.CUDAGraph if dev == "cuda" else torch.xpu.XPUGraph


def kernel_names(op):
    # Separate, untimed call: profiling perturbs ROCm execution speed.
    with profile(activities=[activity]) as prof:
        op()
        mod.synchronize()
    names = {e.name[:40] for e in prof.events() if e.device_type.name in ("CUDA", "XPU")}
    return "+".join(sorted(names))


for line in open(sys.argv[1]):
    if line.startswith("#") or not line.strip():
        continue
    fields = line.split()
    kind, name, d = fields[0], fields[1], list(map(int, fields[2:]))
    if len(sys.argv) > 3 and name != sys.argv[3]:
        continue
    g = torch.Generator().manual_seed(1)
    if kind == "gemm":
        m, k, n = d
        a = torch.randn(m, k, generator=g).to(dev)
        b = torch.randn(k, n, generator=g).to(dev)
        op = lambda: torch.mm(a, b)
        flops = 2 * m * k * n
    else:
        ci, h, w, co, k, s, p = d
        x = torch.randn(1, ci, h, w, generator=g).to(dev)
        wt = torch.randn(co, ci, k, k, generator=g).to(dev)
        op = lambda: torch.nn.functional.conv2d(x, wt, stride=s, padding=p)
        oh = (h + 2 * p - k) // s + 1
        ow = (w + 2 * p - k) // s + 1
        flops = 2 * co * oh * ow * ci * k * k
    with torch.no_grad():
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
        if worst > 1e-6 * scale:
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
    print(
        f"{kind}\t{name}\t{gpu:.4f}\t{statistics.median(wall):.4f}\t"
        f"{flops / gpu / 1e9:.3f}\tgraph-interval:{kernel_names(op)}",
        flush=True,
    )
