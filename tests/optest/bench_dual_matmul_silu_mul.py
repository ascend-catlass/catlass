#!/usr/bin/env python3
"""Benchmark DualMatmul against the task CSV on Ascend950.

Run with the ATK/TorchNPU Python environment after building torch-catlass:
    python bench_dual_matmul_silu_mul.py --csv .../05_DualMatmul_测试集.csv
"""

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401

import torch_catlass


def measure(fn, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / repeat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("dual_matmul_perf.json"))
    args = parser.parse_args()

    rows = list(csv.DictReader(args.csv.open(encoding="gb18030")))
    results = []
    for row in rows:
        m, k, n = (int(row[key]) for key in ("M", "K", "N"))
        baseline_us = float(row["小算子标杆耗时(us)"].strip())
        x = torch.randn((m, k), dtype=torch.float16, device="npu")
        b0 = torch.randn((n, k), dtype=torch.float16, device="npu")
        b1 = torch.randn((n, k), dtype=torch.float16, device="npu")

        def fused():
            return torch_catlass.ascend950_dual_matmul_silu_mul(x, b0, b1, torch.float16)

        # First call also performs JIT compilation; exclude it from timing.
        fused()
        torch.npu.synchronize()
        fused_us = measure(fused, args.warmup, args.repeat)
        results.append({
            "idx": int(row["idx"]),
            "m": m,
            "k": k,
            "n": n,
            "baseline_us": baseline_us,
            "fused_us": fused_us,
            "speedup": baseline_us / fused_us,
        })
        print(f"{m:5d} {k:5d} {n:5d}: {fused_us:9.3f} us, {baseline_us / fused_us:6.3f}x")

    baseline_total = sum(item["baseline_us"] for item in results)
    fused_total = sum(item["fused_us"] for item in results)
    summary = {
        "cases": len(results),
        "baseline_total_us": baseline_total,
        "fused_total_us": fused_total,
        "average_speedup_ratio": baseline_total / fused_total,
        "mean_case_speedup": sum(item["speedup"] for item in results) / len(results),
        "results": results,
    }
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in summary if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
