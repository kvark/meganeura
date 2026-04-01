#!/usr/bin/env python3
"""PyTorch reference for the CI latency benchmark.

Same 3-layer MLP (128->64->10, batch=16) as bench_ci_latency.rs.
Measures upload + forward + backward + readback latency.

Usage:
  python3 bench/bench_ci_latency_pytorch.py [--device cpu|cuda|mps]
"""

import argparse
import json
import time

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 10, bias=False)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = MLP().to(device).float()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    batch = 16
    x_data = torch.tensor(
        [((i * 7 + 13) % 256) / 255.0 for i in range(batch * 128)],
        dtype=torch.float32,
    ).reshape(batch, 128)
    labels = torch.tensor([i % 10 for i in range(batch)], dtype=torch.long)

    # --- Warmup (forward) ---
    model.eval()
    with torch.no_grad():
        for _ in range(args.warmup):
            x = x_data.to(device)
            out = model(x)
            _ = out.cpu()

    # --- Warmup (training) ---
    model.train()
    for _ in range(args.warmup):
        x = x_data.to(device)
        lab = labels.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, lab)
        loss.backward()
        optimizer.step()
        _ = loss.item()

    if device.type == "cuda":
        torch.cuda.synchronize()

    # --- Benchmark forward ---
    model.eval()
    fwd_times = []
    with torch.no_grad():
        for _ in range(args.runs):
            t0 = time.perf_counter()
            x = x_data.to(device)
            out = model(x)
            _ = out.cpu()
            if device.type == "cuda":
                torch.cuda.synchronize()
            fwd_times.append((time.perf_counter() - t0) * 1000)

    # --- Benchmark training ---
    model.train()
    train_times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        x = x_data.to(device)
        lab = labels.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, lab)
        loss.backward()
        optimizer.step()
        loss_val = loss.item()
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_times.append((time.perf_counter() - t0) * 1000)

    fwd_times.sort()
    train_times.sort()
    result = {
        "benchmark": "ci_latency",
        "framework": "pytorch",
        "device": args.device,
        "forward_avg_ms": round(sum(fwd_times) / len(fwd_times), 2),
        "forward_median_ms": round(fwd_times[len(fwd_times) // 2], 2),
        "train_step_avg_ms": round(sum(train_times) / len(train_times), 2),
        "train_step_median_ms": round(train_times[len(train_times) // 2], 2),
        "loss": round(loss_val, 6),
        "runs": args.runs,
        "warmup": args.warmup,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
