# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""E2E guard for floating-point ``tla.arange`` across the helper integer boundary.

The float lowering builds the lane sequence as an integer ``vci`` and adds the
base in the float domain. This sample pins bases that cross the intermediate
integer type (i16 for f16, i32 for f32) and checks the result stays positive
instead of wrapping to the negative side.

Battery case: ``vector-arange-float-edge``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DSL_EXAMPLE_PATH = str(_REPO_ROOT / "examples" / "python" / "tla_dsl")

if _DSL_EXAMPLE_PATH not in sys.path:
    # append (not insert(0)) to avoid shadowing installed packages (G.PSL.03).
    sys.path.append(_DSL_EXAMPLE_PATH)

import catlass.tla as tla  # noqa: E402
from catlass.tla.runtime import from_dlpack  # noqa: E402

F16_ELE = 128
F32_ELE = 64
# One past the i16 maximum; the float sequence must stay positive.
F16_BASE = 32767
# Crosses the i32 maximum inside a 64-lane f32 sequence.
F32_BASE = 2147483615


@tla.kernel
def arange_f16_increase(dst: tla.Tensor) -> None:
    out = tla.tile_view(dst, tla.make_shape(F16_ELE), tla.make_coord(0))
    ub_ptr = tla.allocate(F16_ELE, tla.Float16, tla.AddressSpace.ub, 256)
    ub = tla.make_tensor_like(ub_ptr, out, tla.arch.RowMajor)
    vec_done = tla.flag("vec_done", tla.pipes.VECTOR, tla.pipes.MTE3)
    with tla.vector():
        with tla.vec.func(mode="simd"):
            tla.tile_view(ub, tla.make_shape(F16_ELE), tla.make_coord(0)).store(
                tla.arange(F16_BASE, dtype=tla.Float16)
            )
        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)
        tla.copy(out, ub)


@tla.kernel
def arange_f16_decrease(dst: tla.Tensor) -> None:
    out = tla.tile_view(dst, tla.make_shape(F16_ELE), tla.make_coord(0))
    ub_ptr = tla.allocate(F16_ELE, tla.Float16, tla.AddressSpace.ub, 256)
    ub = tla.make_tensor_like(ub_ptr, out, tla.arch.RowMajor)
    vec_done = tla.flag("vec_done", tla.pipes.VECTOR, tla.pipes.MTE3)
    with tla.vector():
        with tla.vec.func(mode="simd"):
            tla.tile_view(ub, tla.make_shape(F16_ELE), tla.make_coord(0)).store(
                tla.arange(F16_BASE, order="decrease", dtype=tla.Float16)
            )
        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)
        tla.copy(out, ub)


@tla.kernel
def arange_f32_increase(dst: tla.Tensor) -> None:
    out = tla.tile_view(dst, tla.make_shape(F32_ELE), tla.make_coord(0))
    ub_ptr = tla.allocate(F32_ELE, tla.Float32, tla.AddressSpace.ub, 256)
    ub = tla.make_tensor_like(ub_ptr, out, tla.arch.RowMajor)
    vec_done = tla.flag("vec_done", tla.pipes.VECTOR, tla.pipes.MTE3)
    with tla.vector():
        with tla.vec.func(mode="simd"):
            tla.tile_view(ub, tla.make_shape(F32_ELE), tla.make_coord(0)).store(
                tla.arange(F32_BASE, dtype=tla.Float32)
            )
        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)
        tla.copy(out, ub)


@tla.kernel
def arange_f32_decrease(dst: tla.Tensor) -> None:
    out = tla.tile_view(dst, tla.make_shape(F32_ELE), tla.make_coord(0))
    ub_ptr = tla.allocate(F32_ELE, tla.Float32, tla.AddressSpace.ub, 256)
    ub = tla.make_tensor_like(ub_ptr, out, tla.arch.RowMajor)
    vec_done = tla.flag("vec_done", tla.pipes.VECTOR, tla.pipes.MTE3)
    with tla.vector():
        with tla.vec.func(mode="simd"):
            tla.tile_view(ub, tla.make_shape(F32_ELE), tla.make_coord(0)).store(
                tla.arange(F32_BASE, order="decrease", dtype=tla.Float32)
            )
        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)
        tla.copy(out, ub)


CASES = (
    ("f16-increase", arange_f16_increase, tla.Float16, F16_ELE, F16_BASE, "increase"),
    ("f16-decrease", arange_f16_decrease, tla.Float16, F16_ELE, F16_BASE, "decrease"),
    ("f32-increase", arange_f32_increase, tla.Float32, F32_ELE, F32_BASE, "increase"),
    ("f32-decrease", arange_f32_decrease, tla.Float32, F32_ELE, F32_BASE, "decrease"),
)


def _reference(numpy, dtype_name: str, base: int, n_ele: int, order: str):
    """Model the float path: convert base first, then add the lane index."""
    if dtype_name == "f16":
        base_f = numpy.float16(base)
        lanes = numpy.arange(n_ele, dtype=numpy.float16)
        ref = (base_f + lanes).astype(numpy.float16)
    else:
        base_f = numpy.float32(base)
        lanes = numpy.arange(n_ele, dtype=numpy.float32)
        ref = (base_f + lanes).astype(numpy.float32)
    return ref if order == "increase" else ref[::-1].copy()


def run(args: argparse.Namespace) -> int:
    import numpy as np
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    passed = 0
    failed = 0
    for name, kernel, tla_dtype, n_ele, base, order in CASES:
        torch_dtype = torch.float16 if tla_dtype is tla.Float16 else torch.float32
        host = torch.empty(n_ele, dtype=torch_dtype, device="npu")
        tla_out = from_dlpack(
            host.contiguous(), layout_tag=tla.arch.RowMajor, origin_shape=(n_ele,)
        )
        artifact = tla.compile(kernel, tla_out, options="--npu-arch 3510")
        artifact(tla_out, block_num=args.block_num)
        torch.npu.synchronize()

        actual = host.cpu()
        expected = torch.from_numpy(
            _reference(np, name.split("-")[0], base, n_ele, order)
        )
        matched = torch.equal(actual, expected)
        positive = bool((actual > 0).all())
        ok = matched and positive
        passed += int(ok)
        failed += int(not ok)
        print(
            f"case={name} dtype={name.split('-')[0]} order={order} base={base} "
            f"matched={matched} all_positive={positive} "
            f"first={actual[0].item():.1f} kernel.o={artifact.kernel_binary_path}"
        )
    print(f"arange float edge: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile and run floating-point arange across integer boundaries."
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--block-num",
        type=int,
        default=1,
        help="Launch block count; must be 1 (kernel is not block-partitioned)",
    )
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
