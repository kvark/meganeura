#!/usr/bin/env python3
import os
import shutil
import sys

nsys = os.environ.get("GAP_NSYS", shutil.which("nsys"))
assert nsys, "Set GAP_NSYS to the Nsight Systems executable"
os.environ["NSYS_NVTX_PROFILER_REGISTER_ONLY"] = "0"
args = sys.argv[1:]
if args[0] == "profile":
    engine = "meganeura" if any("--trace=vulkan" in arg for arg in args) else "pytorch"
    destination = next(arg.split("=", 1)[1] for arg in args if arg.startswith("--output="))
    descriptor = os.open(destination + "-nsys.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.dup2(descriptor, 2)
    os.close(descriptor)
    args[1:1] = ["--capture-range=nvtx", "--capture-range-end=stop",
                 f"--nvtx-capture={engine}/inference/measure@*"]
os.execv(nsys, [nsys, *args])
