# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""E2E: the operations that run on each core's own scalar unit.

Both are valid in a `tla.cube` region, in a `tla.vec.func`, and in the plain
kernel body, so each is exercised in more than one of those.

`tla.sqrt` / `Float.sqrt()` on a float scalar lowers to `math.sqrt` outside a
SIMT region and to `tla.simt_sqrt` inside one.

`x.to(dtype, round_mode=...)` selects one of the five conversion instructions
the scalar unit has, through a meta-op bitcode helper:

    NEAREST_EVEN -> conv_f322s32r   f32 -> i32, ties to even
    NEAREST_AWAY -> conv_f322s32a   f32 -> i32, ties away from zero
    FLOOR        -> conv_f322s32f   f32 -> i32, toward -inf
    CEIL         -> conv_f322s32c   f32 -> i32, toward +inf
    TRUNC        -> arith.fptosi    f32 -> i32, toward zero (no instruction exists)
    ODD          -> conv_f322f16o   f32 -> f16, round to odd

The inputs include exact ties (+/-0.5, +/-1.5, +/-2.5) and negatives, so every
rounding mode is distinguishable from the others.
"""

from __future__ import annotations

import argparse
import math
from typing import TYPE_CHECKING

import numpy as np

import catlass.tla as tla
from catlass.params import ScalarRoundMode

if TYPE_CHECKING:
    import torch

INPUTS = [0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 2.3, -2.3, 7.8, -7.8]
LENGTH = len(INPUTS)
CONST_INPUT = 6.25


# --- scalar sqrt -----------------------------------------------------------


@tla.kernel
def sqrt_kernel_body(src: tla.Tensor, out: tla.Tensor) -> None:
    """Plain kernel body: no cube or vector region."""
    for i in tla.range(0, LENGTH, 1):
        out[i] = tla.sqrt(src[i])


@tla.kernel
def sqrt_cube_region(src: tla.Tensor, out: tla.Tensor) -> None:
    """The AIC scalar unit."""
    with tla.cube():
        for i in tla.range(0, LENGTH, 1):
            out[i] = tla.sqrt(src[i])


@tla.kernel
def sqrt_vector_region(src: tla.Tensor, out: tla.Tensor) -> None:
    """The AIV scalar unit, inside a SIMD vector function."""
    with tla.vector():
        with tla.vec.func(mode="simd"):
            for i in tla.range(0, LENGTH, 1):
                out[i] = tla.sqrt(src[i])


@tla.kernel
def sqrt_method_form(src: tla.Tensor, out: tla.Tensor) -> None:
    """``x.sqrt()`` is the same operation as ``tla.sqrt(x)``."""
    for i in tla.range(0, LENGTH, 1):
        out[i] = src[i].sqrt()


@tla.kernel
def sqrt_constant_fold(out: tla.Tensor) -> None:
    """A trace-time literal folds in Python; no op reaches the device."""
    out[0] = tla.sqrt(tla.Float32(CONST_INPUT))


# --- scalar f32 conversions ------------------------------------------------


@tla.kernel
def round_nearest_even(src: tla.Tensor, out: tla.Tensor) -> None:
    for i in tla.range(0, LENGTH, 1):
        out[i] = src[i].to(tla.Int32, round_mode=ScalarRoundMode.NEAREST_EVEN)


@tla.kernel
def round_nearest_away(src: tla.Tensor, out: tla.Tensor) -> None:
    for i in tla.range(0, LENGTH, 1):
        out[i] = src[i].to(tla.Int32, round_mode=ScalarRoundMode.NEAREST_AWAY)


@tla.kernel
def round_floor_in_cube(src: tla.Tensor, out: tla.Tensor) -> None:
    """The helper has to be reachable from the AIC."""
    with tla.cube():
        for i in tla.range(0, LENGTH, 1):
            out[i] = src[i].to(tla.Int32, round_mode=ScalarRoundMode.FLOOR)


@tla.kernel
def round_ceil_in_vector(src: tla.Tensor, out: tla.Tensor) -> None:
    """...and from the AIV."""
    with tla.vector():
        with tla.vec.func(mode="simd"):
            for i in tla.range(0, LENGTH, 1):
                out[i] = src[i].to(tla.Int32, round_mode=ScalarRoundMode.CEIL)


@tla.kernel
def round_trunc(src: tla.Tensor, out: tla.Tensor) -> None:
    for i in tla.range(0, LENGTH, 1):
        out[i] = src[i].to(tla.Int32, round_mode=ScalarRoundMode.TRUNC)


@tla.kernel
def narrow_odd(src: tla.Tensor, out: tla.Tensor) -> None:
    for i in tla.range(0, LENGTH, 1):
        out[i] = src[i].to(tla.Float16, round_mode=ScalarRoundMode.ODD)


@tla.kernel
def narrow_default(src: tla.Tensor, out: tla.Tensor) -> None:
    """No round_mode: arith.truncf, which is round-to-nearest-even."""
    for i in tla.range(0, LENGTH, 1):
        out[i] = src[i].to(tla.Float16)


# --- golden references -----------------------------------------------------


def _nearest_away(v: float) -> int:
    return math.floor(v + 0.5) if v >= 0 else math.ceil(v - 0.5)


def _f32_to_f16_odd(value: float) -> float:
    """Reference for conv_f322f16o: keep exact values, else pick the odd neighbour."""
    v32 = np.float32(value)
    nearest = np.float16(v32)
    if np.float32(nearest) == v32:
        return float(nearest)
    toward_zero = nearest
    if abs(np.float32(nearest)) > abs(v32):
        toward_zero = np.nextafter(nearest, np.float16(0.0))
    if int(toward_zero.view(np.uint16)) & 1:
        return float(toward_zero)
    away = np.float16(np.inf) if v32 > 0 else np.float16(-np.inf)
    return float(np.nextafter(toward_zero, away))


SQRT_CASES = (
    ("kernel_body", sqrt_kernel_body),
    ("cube_region", sqrt_cube_region),
    ("vector_region", sqrt_vector_region),
    ("method_form", sqrt_method_form),
)

ROUND_CASES = (
    ("nearest_even", round_nearest_even, round),  # Python round() is ties-to-even
    ("nearest_away", round_nearest_away, _nearest_away),
    ("floor_cube", round_floor_in_cube, math.floor),
    ("ceil_vector", round_ceil_in_vector, math.ceil),
    ("trunc", round_trunc, int),
)


# --- harness ---------------------------------------------------------------


def _require_torch_npu(device: int):
    import torch

    try:
        import torch_npu
    except ImportError as exc:
        raise SystemExit("torch_npu is required for this example") from exc
    torch_npu.npu.set_device(device)
    return torch


def _gm(tensor_1d) -> tla.Tensor:
    return tla.from_dlpack(tensor_1d.contiguous(), layout_tag=tla.arch.RowMajor)


def _run_sqrt_case(args, torch, device: str, name: str, kernel) -> int:
    src = torch.rand(LENGTH, dtype=torch.float32, device=device) * 9.0 + 0.5
    out = torch.full((LENGTH,), -1.0, dtype=torch.float32, device=device)
    src_t, out_t = _gm(src), _gm(out)

    tla.compile(kernel, src_t, out_t)(src_t, out_t, block_num=args.block_num)
    torch.npu.synchronize()

    if not torch.allclose(out, torch.sqrt(src), rtol=1e-5, atol=1e-5):
        print(
            f"sqrt_{name}_failed expected={torch.sqrt(src).tolist()} actual={out.tolist()}"
        )
        return 1
    print(f"sqrt_{name}_ok=True")
    return 0


def _run_sqrt_constant_fold(args, torch, device: str) -> int:
    out = torch.full((1,), -1.0, dtype=torch.float32, device=device)
    out_t = _gm(out)

    tla.compile(sqrt_constant_fold, out_t)(out_t, block_num=args.block_num)
    torch.npu.synchronize()

    actual = float(out[0].item())
    if abs(actual - math.sqrt(CONST_INPUT)) > 1e-5:
        print(
            f"sqrt_constant_fold_failed expected={math.sqrt(CONST_INPUT)} actual={actual}"
        )
        return 1
    print(f"sqrt_constant_fold_ok=True value={actual}")
    return 0


def _run_round_case(args, torch, device: str, name: str, kernel, golden) -> int:
    src = torch.tensor(INPUTS, dtype=torch.float32, device=device)
    out = torch.full((LENGTH,), -99, dtype=torch.int32, device=device)
    src_t, out_t = _gm(src), _gm(out)

    tla.compile(kernel, src_t, out_t)(src_t, out_t, block_num=args.block_num)
    torch.npu.synchronize()

    expected = [golden(v) for v in INPUTS]
    actual = [int(out[i].item()) for i in range(LENGTH)]
    if actual != expected:
        print(
            f"round_{name}_failed inputs={INPUTS} expected={expected} actual={actual}"
        )
        return 1
    print(f"round_{name}_ok=True values={actual}")
    return 0


def _run_narrow_case(args, torch, device: str) -> int:
    """f32 -> f16 round-to-odd, against a host reference and against truncf."""
    src = torch.tensor(INPUTS, dtype=torch.float32, device=device)
    src_t = _gm(src)

    results = {}
    for name, kernel in (("odd", narrow_odd), ("default", narrow_default)):
        out = torch.zeros(LENGTH, dtype=torch.float16, device=device)
        out_t = _gm(out)
        tla.compile(kernel, src_t, out_t)(src_t, out_t, block_num=args.block_num)
        torch.npu.synchronize()
        results[name] = [float(out[i].item()) for i in range(LENGTH)]

    expected_odd = [_f32_to_f16_odd(v) for v in INPUTS]
    if results["odd"] != expected_odd:
        print(f"narrow_odd_failed expected={expected_odd} actual={results['odd']}")
        return 1
    print(f"narrow_odd_ok=True values={results['odd']}")

    # torch narrows with round-to-nearest-even, which is what the default path
    # must match -- and what ODD must NOT match wherever the value is inexact.
    expected_rne = [float(v) for v in src.to(torch.float16).tolist()]
    if results["default"] != expected_rne:
        print(
            f"narrow_default_failed expected={expected_rne} actual={results['default']}"
        )
        return 1
    print("narrow_default_ok=True (matches round-to-nearest-even)")

    inexact = [i for i, v in enumerate(INPUTS) if float(np.float16(np.float32(v))) != v]
    differing = [i for i in inexact if results["odd"][i] != expected_rne[i]]
    if inexact and not differing:
        print(f"odd_indistinguishable_from_rne inexact_indices={inexact}")
        return 1
    print(
        f"odd_differs_from_rne_ok=True on {len(differing)}/{len(inexact)} inexact inputs"
    )
    return 0


def run(args: argparse.Namespace) -> int:
    torch = _require_torch_npu(args.device)
    device = f"npu:{args.device}"
    torch.manual_seed(0)

    for name, kernel in SQRT_CASES:
        if _run_sqrt_case(args, torch, device, name, kernel) != 0:
            return 1
    if _run_sqrt_constant_fold(args, torch, device) != 0:
        return 1
    for name, kernel, golden in ROUND_CASES:
        if _run_round_case(args, torch, device, name, kernel, golden) != 0:
            return 1
    if _run_narrow_case(args, torch, device) != 0:
        return 1

    print("verification_ok=True")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scalar-unit sqrt and f32 conversions."
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--block-num", type=int, default=1)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
