# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end checks for ``tla.log`` subnormal handling (``LogAlgo``).

``FLUSH_TO_ZERO`` keeps the existing AVE ``vln`` lowering. On Ascend 950 the
f32 ``vln`` flushes positive subnormal inputs (``0 < x < 2^-126``) to zero
(``-inf``); negative inputs stay NaN and ``+/-0.0`` is ``-inf``. The f16 ``vln``
path already computes positive subnormal inputs. ``SUBNORMAL`` lowers to the
software implementation and must match the fp64 reference for both dtypes.
Normal inputs must agree in both modes.
"""

from __future__ import annotations

import argparse
from typing import Any

import catlass.tla as tla

_VECTOR_ELE = 64
_DTYPE: Any = tla.Float32
_ALGO: Any = tla.params.LogAlgo.FLUSH_TO_ZERO


@tla.kernel
def log_algo_op(mem_x: tla.Tensor, mem_y: tla.Tensor) -> None:
    loaded = tla.flag("loaded", tla.arch.MTE2, tla.arch.VECTOR)
    done = tla.flag("done", tla.arch.VECTOR, tla.arch.MTE3)
    x_gm = tla.tile_view(mem_x, tla.make_shape(_VECTOR_ELE), tla.make_coord(0))
    y_gm = tla.tile_view(mem_y, tla.make_shape(_VECTOR_ELE), tla.make_coord(0))
    x_ptr = tla.allocate(_VECTOR_ELE, _DTYPE, tla.AddressSpace.ub, 256)
    y_ptr = tla.allocate(_VECTOR_ELE, _DTYPE, tla.AddressSpace.ub, 256)
    x_ub = tla.make_tensor_like(x_ptr, x_gm, tla.arch.RowMajor)
    y_ub = tla.make_tensor_like(y_ptr, y_gm, tla.arch.RowMajor)

    with tla.vector():
        tla.copy(x_ub, x_gm)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        with tla.vec.func(mode="simd"):
            x_tile = tla.tile_view(x_ub, tla.make_shape(_VECTOR_ELE), tla.make_coord(0))
            y_tile = tla.tile_view(y_ub, tla.make_shape(_VECTOR_ELE), tla.make_coord(0))
            y_tile.store(tla.log(x_tile.load(), mode=_ALGO))
        tla.set_flag(done)
        tla.wait_flag(done)
        tla.copy(y_gm, y_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def _set_config(dtype: str, algo: str) -> None:
    global _VECTOR_ELE, _DTYPE, _ALGO
    _DTYPE = {"f16": tla.Float16, "f32": tla.Float32}[dtype]
    _VECTOR_ELE = 128 if dtype == "f16" else 64
    _ALGO = {
        "ftz": tla.params.LogAlgo.FLUSH_TO_ZERO,
        "subnormal": tla.params.LogAlgo.SUBNORMAL,
    }[algo]


def _log_values(count: int, dtype: str) -> list[float]:
    """Normal and subnormal magnitudes, plus sign and zero edge cases."""
    normal = [2.0, 8.0, 1.0, 0.5, 0.25, 0.125, 1024.0, 0.001953125]
    if dtype == "f16":
        # f16 normals start at 2^-14, subnormals run 2^-24..2^-15.
        subnormal = [2.0**-15, 2.0**-16, 2.0**-18, 2.0**-20, 2.0**-22, 2.0**-24]
    else:
        # f32 normals start at 2^-126, subnormals run 2^-149..2^-127.
        subnormal = [
            2.0**-127,
            2.0**-130,
            2.0**-135,
            2.0**-140,
            2.0**-145,
            2.0**-149,
        ]
    # Negative subnormals pin the FTZ range boundary (only 0 < x is flushed),
    # and signed zeros pin log(+/-0) = -inf.
    edge = [-subnormal[0], -subnormal[-1], -0.0, 0.0]
    pattern = normal + subnormal + edge
    return [pattern[index % len(pattern)] for index in range(count)]


def _reference(values: Any, algo: str) -> Any:
    """fp64 ln reference.

    ``FLUSH_TO_ZERO`` reuses the existing ``vln`` path. On Ascend 950 the f32
    ``vln`` flushes positive subnormal inputs (``0 < x < 2^-126``) to zero, so
    they become ``-inf``; negative inputs keep the plain logarithm's NaN and
    ``+/-0.0`` yields ``-inf``. The f16 ``vln`` path already computes positive
    subnormal inputs, so its FTZ reference is the plain logarithm.
    """
    import torch

    wide = values.to(torch.float64)
    expected = torch.log(wide)
    if algo == "ftz" and values.dtype == torch.float32:
        flushed = (wide > 0) & (wide < torch.tensor(2.0**-126, dtype=torch.float64))
        expected = torch.where(
            flushed,
            torch.full_like(expected, float("-inf")),
            expected,
        )
    return expected


def _run_case(dtype: str, algo: str) -> int:
    import torch

    _set_config(dtype, algo)
    torch_dtype = {"f16": torch.float16, "f32": torch.float32}[dtype]
    values = torch.tensor(_log_values(_VECTOR_ELE, dtype), dtype=torch_dtype)
    x = values.to("npu")
    y = torch.full_like(x, 123.0)
    tla_x = tla.from_dlpack(x, layout_tag=tla.arch.RowMajor)
    tla_y = tla.from_dlpack(y, layout_tag=tla.arch.RowMajor)
    artifact = tla.compile(log_algo_op, tla_x, tla_y, options="--npu-arch 3510")
    artifact(tla_x, tla_y, block_num=1)
    torch.npu.synchronize()

    actual = y.cpu()
    expected = _reference(values, algo).to(torch_dtype)
    finite = torch.isfinite(expected)
    nan = torch.isnan(expected)
    neg_inf = torch.isneginf(expected)
    pos_inf = torch.isposinf(expected)
    # Check NaN and both infinities on their own lanes: a flush-to-zero bug that
    # turns a negative subnormal into -inf must not pass as a finite lane.
    ok = bool(
        torch.equal(torch.isnan(actual), nan)
        and torch.equal(torch.isinf(actual) & (actual > 0), pos_inf)
        and torch.equal(torch.isinf(actual) & (actual < 0), neg_inf)
        and torch.allclose(
            actual[finite].to(torch.float64),
            expected[finite].to(torch.float64),
            rtol=2e-3 if dtype == "f16" else 1e-5,
            atol=0.0,
        )
    )
    print(f"op=log_algo dtype={dtype} mode={algo} passed={ok}")
    if not ok:
        for index, (value, got, want) in enumerate(
            zip(values.tolist(), actual.tolist(), expected.tolist())
        ):
            print(f"  lane={index} x={value!r} got={got!r} expected={want!r}")
    print(f"kernel.o path={artifact.kernel_binary_path}")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true", help="run one dtype/mode case")
    mode.add_argument(
        "--batch-run",
        action="store_true",
        help="run both modes for both dtypes",
    )
    parser.add_argument("--dtype", choices=("f16", "f32"))
    parser.add_argument("--mode", choices=("ftz", "subnormal"))
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if args.batch_run and (args.mode is not None or args.dtype is not None):
        parser.error("--mode/--dtype cannot be combined with --batch-run")

    import torch
    import torch_npu  # noqa: F401 -- registers the NPU backend

    torch.npu.set_device(args.device)
    if args.batch_run:
        failed = 0
        total = 0
        for dtype in ("f32", "f16"):
            for algo in ("ftz", "subnormal"):
                failed += _run_case(dtype, algo)
                total += 1
        print(f"batch summary: passed={total - failed} failed={failed} total={total}")
        return 0 if failed == 0 else 1
    return _run_case(args.dtype or "f32", args.mode or "ftz")


if __name__ == "__main__":
    raise SystemExit(main())
