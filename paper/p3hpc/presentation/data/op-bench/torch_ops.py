"""One GEMM or convolution per shape, strict f32. Prints the median summed
device-kernel time per op (torch.profiler) and the median event-timed wall
time of a single op."""
import sys, time, statistics, torch
from torch.profiler import profile, ProfilerActivity

dev = sys.argv[2]
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = True
mod = torch.cuda if dev == "cuda" else torch.xpu
act = ProfilerActivity.CUDA if dev == "cuda" else ProfilerActivity.XPU

def sync(): mod.synchronize()

for line in open(sys.argv[1]):
    if line.startswith("#") or not line.strip():
        continue
    f = line.split(); kind, name, d = f[0], f[1], list(map(int, f[2:]))
    if len(sys.argv) > 3 and name != sys.argv[3]:
        continue
    g = torch.Generator().manual_seed(1)
    if kind == "gemm":
        m, k, n = d
        a = torch.randn(m, k, generator=g).to(dev); b = torch.randn(k, n, generator=g).to(dev)
        op = lambda: torch.mm(a, b)
        flops = 2 * m * k * n
    else:
        ci, h, w, co, k, s, p = d
        x = torch.randn(1, ci, h, w, generator=g).to(dev); wt = torch.randn(co, ci, k, k, generator=g).to(dev)
        op = lambda: torch.nn.functional.conv2d(x, wt, stride=s, padding=p)
        oh = (h + 2 * p - k) // s + 1; ow = (w + 2 * p - k) // s + 1
        flops = 2 * co * oh * ow * ci * k * k
    with torch.no_grad():
        t0 = time.time()
        while time.time() - t0 < 1.0:
            op(); sync()
        wall = []
        for _ in range(50):
            t = time.perf_counter(); op(); sync(); wall.append((time.perf_counter() - t) * 1e3)
        iters = 30
        with profile(activities=[act]) as prof:
            for _ in range(iters):
                op()
            sync()
    kern = [e for e in prof.events() if e.device_type.name in ("CUDA", "XPU") and e.time_range.elapsed_us() > 0]
    # Group kernels into ops by order: every op launches the same kernel list.
    per_op = len(kern) // iters
    sums = [sum(e.time_range.elapsed_us() for e in kern[i * per_op:(i + 1) * per_op]) / 1e3 for i in range(iters)]
    names = "+".join(sorted({e.name[:40] for e in kern}))
    gpu = statistics.median(sums)
    print(f"{kind}\t{name}\t{gpu:.4f}\t{statistics.median(wall):.4f}\t{flops / gpu / 1e9:.3f}\t{per_op}:{names}", flush=True)
