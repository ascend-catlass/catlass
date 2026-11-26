#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Paired benchmark for constant padding against the Torch reference."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path
import statistics

import torch
import torch.nn.functional as F
import torch_npu


def _load_pad():
    path = Path(__file__).with_name("pad.py")
    spec = importlib.util.spec_from_file_location("pad_example", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.pad


pad = _load_pad()


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
    parser.add_argument("--m", type=int, default=16384)
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--left", type=int, default=3)
    parser.add_argument("--right", type=int, default=5)
    parser.add_argument("--value", type=float, default=-2.5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    torch.manual_seed(0)
    x = torch.randn(
        args.m,
        args.n,
        dtype=torch.float32,
        device=f"npu:{args.device}",
    )
    candidate_args = (x, args.left, args.right, args.value)
    reference_args = (x, (args.left, args.right), "constant", args.value)

    candidate = pad(*candidate_args)
    reference = F.pad(*reference_args)
    torch.npu.synchronize()
    if not torch.equal(candidate, reference):
        raise RuntimeError("accuracy check failed")

    for _ in range(args.warmup):
        pad(*candidate_args)
        F.pad(*reference_args)
    torch.npu.synchronize()

    candidate_ms = []
    reference_ms = []
    gc.collect()
    gc.disable()
    try:
        for trial in range(args.trials):
            order = (
                (("candidate", pad, candidate_args),)
                + (("reference", F.pad, reference_args),)
                if trial % 2 == 0
                else (("reference", F.pad, reference_args),)
                + (("candidate", pad, candidate_args),)
            )
            for name, function, call_args in order:
                elapsed = timed(function, call_args)
                (candidate_ms if name == "candidate" else reference_ms).append(elapsed)
    finally:
        gc.enable()

    candidate_mean = statistics.mean(candidate_ms)
    reference_mean = statistics.mean(reference_ms)
    print(
        json.dumps(
            {
                "m": args.m,
                "n": args.n,
                "left": args.left,
                "right": args.right,
                "warmup": args.warmup,
                "trials": args.trials,
                "candidate_mean_ms": candidate_mean,
                "reference_mean_ms": reference_mean,
                "speedup": reference_mean / candidate_mean,
                "candidate_median_ms": statistics.median(candidate_ms),
                "reference_median_ms": statistics.median(reference_ms),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
