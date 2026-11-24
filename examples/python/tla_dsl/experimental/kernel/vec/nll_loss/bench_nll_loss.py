#!/usr/bin/env python3
"""Paired CrossEntropy benchmark for the NLLLoss example."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu


def _load_run():
    path = Path(__file__).with_name("nll_loss.py")
    spec = importlib.util.spec_from_file_location("nll_loss_example", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run


run = _load_run()


def timed(function, call_args):
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    function(*call_args)
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--n", type=int, default=257)
    parser.add_argument("--c", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    torch.manual_seed(args.seed)
    device = f"npu:{args.device}"
    x = torch.randn(args.n, args.c, dtype=torch.float32, device=device)
    target = torch.randint(0, args.c, (args.n,), device=device)
    call_args = (x, target)

    def candidate_fn(x, target):
        return run(x, target, from_logits=True)

    candidate = candidate_fn(*call_args)
    reference = F.cross_entropy(*call_args)
    torch.npu.synchronize()
    error = float((candidate - reference).abs())
    if error > 2e-5:
        raise RuntimeError(f"accuracy check failed: {error}")

    if args.warmup:
        for _ in range(args.warmup):
            candidate_warmup = candidate_fn(*call_args)
            reference_warmup = F.cross_entropy(*call_args)
        torch.npu.synchronize()
        if not torch.isfinite(candidate_warmup) or not torch.isfinite(reference_warmup):
            raise RuntimeError("non-finite warmup result")

    gc.collect()
    gc.disable()
    candidate_ms = []
    reference_ms = []
    try:
        for trial in range(args.trials):
            order = (
                (("candidate", candidate_fn), ("reference", F.cross_entropy))
                if trial % 2 == 0
                else (("reference", F.cross_entropy), ("candidate", candidate_fn))
            )
            for name, function in order:
                if name == "candidate":
                    elapsed = timed(function, call_args)
                    candidate_ms.append(elapsed)
                else:
                    reference_ms.append(timed(function, call_args))
    finally:
        gc.enable()

    candidate_mean = statistics.mean(candidate_ms)
    reference_mean = statistics.mean(reference_ms)
    result = {
        "n": args.n,
        "c": args.c,
        "warmup": args.warmup,
        "trials": args.trials,
        "accuracy_error": error,
        "candidate_mean_ms": candidate_mean,
        "reference_mean_ms": reference_mean,
        "speedup": reference_mean / candidate_mean,
        "candidate_median_ms": statistics.median(candidate_ms),
        "reference_median_ms": statistics.median(reference_ms),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
