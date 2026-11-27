# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Any

import catlass.tla as tla
from catlass.params import LoadDist, NormalLoadParams

from vector_op_harness import (
    DirectVectorOpConfig,
    DirectVectorOpHarness,
    vector_kernel_config,
)

# i8 VL = 256. Each loop reads VL/2 = 128 and writes VL = 256.
VECTOR_ELE = 512
VL_ELE = 256
HALF_VL_ELE = VL_ELE // 2
LOOPS = VECTOR_ELE // VL_ELE
SRC_ELE = LOOPS * HALF_VL_ELE
OUT_VALID_ELE = LOOPS * VL_ELE
ALL_DTYPES = ("i8",)

_KERNEL_DTYPE = tla.Int8
_KERNEL_ELEMENT_BYTES = 1
_KERNEL_SHAPE = (VECTOR_ELE,)
_KERNEL_SENTINEL: int = -101
_US_B8_LOAD = NormalLoadParams(load_dist=LoadDist.DIST_US_B8)


@tla.kernel
def load_us_b8_op(mem_src: tla.Tensor, mem_dst: tla.Tensor) -> None:
    ub_loaded = tla.flag("ub_loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    vec_done = tla.flag("vec_done", tla.pipes.VECTOR, tla.pipes.MTE3)

    src_gm = tla.tile_view(mem_src, tla.make_shape(SRC_ELE), tla.make_coord(0))
    # Copy back only the lanes the kernel writes; host-initialized sentinel
    # remains in the unused tail of mem_dst.
    dst_gm = tla.tile_view(mem_dst, tla.make_shape(OUT_VALID_ELE), tla.make_coord(0))

    src_ub = _make_ub_tensor(src_gm, SRC_ELE)
    dst_ub = _make_ub_tensor(dst_gm, OUT_VALID_ELE)

    with tla.vector():
        tla.copy(src_ub, src_gm)

        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)

        with tla.vec.func(mode="simd"):
            for i in tla.range(LOOPS):
                # VL/2-wide view at block i: US_B8 reads VL/2 b8 elements from
                # this base and upsamples to a VL register.
                src_tile = tla.tile_view(
                    src_ub, tla.make_shape(HALF_VL_ELE), tla.make_coord(i)
                )
                dst_tile = tla.tile_view(
                    dst_ub, tla.make_shape(VL_ELE), tla.make_coord(i)
                )
                up_reg = src_tile.load(_US_B8_LOAD)
                dst_tile.store(up_reg)

        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)

        tla.copy(dst_gm, dst_ub)

        tla.pipe_barrier(tla.pipes.ALL)


def _make_ub_tensor(like_tensor: Any, num_ele: int) -> Any:
    ptr = tla.allocate(num_ele, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    return tla.make_tensor_like(ptr, like_tensor, tla.arch.RowMajor)


def _operator_specs() -> dict[str, dict[str, Any]]:
    return {
        "us_b8": {
            "default_atol": 0.0,
            "dtypes": ALL_DTYPES,
        },
    }


def _is_unsupported_case(op_name: str, dtype_name: str) -> bool:
    del op_name
    return dtype_name not in ALL_DTYPES


def _print_skip(op_name: str, dtype_name: str, shape: tuple[int, ...]) -> None:
    del shape
    print(f"skip op={op_name} dtype={dtype_name}: DIST_US_B8 currently requires i8")


def _set_kernel_config(
    op_name: str, dtype_name: str, shape: tuple[int, ...] | None = None
) -> tuple[type[Any], Any, float | int]:
    global VL_ELE, HALF_VL_ELE, LOOPS, SRC_ELE, OUT_VALID_ELE
    global _KERNEL_DTYPE, _KERNEL_ELEMENT_BYTES, _KERNEL_SHAPE, _KERNEL_SENTINEL
    if op_name not in _operator_specs():
        raise SystemExit(f"unknown load_us_b8 operator {op_name!r}")

    del shape
    if dtype_name != "i8":
        raise SystemExit(f"DIST_US_B8 currently requires i8, got {dtype_name}")
    config = vector_kernel_config(dtype_name, (VECTOR_ELE,), ALL_DTYPES)
    VL_ELE = config.lanes
    HALF_VL_ELE = VL_ELE // 2
    LOOPS = VECTOR_ELE // VL_ELE
    SRC_ELE = LOOPS * HALF_VL_ELE
    OUT_VALID_ELE = LOOPS * VL_ELE
    _KERNEL_SHAPE = (VECTOR_ELE,)
    _KERNEL_DTYPE = config.tla_dtype
    _KERNEL_ELEMENT_BYTES = config.element_bytes
    _KERNEL_SENTINEL = config.default_sentinel
    return config.tla_dtype, config.torch_dtype, config.default_sentinel


def _make_inputs(args: Any, dtype_name: str, torch: Any) -> tuple[Any, ...]:
    _, dtype, _ = _set_kernel_config(args.op, dtype_name, args.shape)
    device = "npu"
    # Distinct small values so up-sampling is easy to verify visually.
    src = torch.arange(SRC_ELE, dtype=torch.float32, device=device).to(dtype)
    return (src,)


def _expected(op_name: str, inputs: tuple[Any, ...]) -> Any:
    del op_name
    import torch

    (src,) = inputs
    # Kernel writes LOOPS*VL lanes and leaves the host sentinel in the rest.
    dst = torch.full(
        (VECTOR_ELE,), _KERNEL_SENTINEL, dtype=src.dtype, device=src.device
    )
    upsampled = torch.repeat_interleave(src, 2)
    dst[:OUT_VALID_ELE] = upsampled[:OUT_VALID_ELE]
    return (dst,)


HARNESS = DirectVectorOpHarness(
    DirectVectorOpConfig(
        description=(
            "Compile and run DIST_US_B8 single-destination up-sample load (i8)."
        ),
        kernel=load_us_b8_op,
        all_dtypes=ALL_DTYPES,
        operator_specs=_operator_specs,
        set_kernel_config=_set_kernel_config,
        get_vector_elements=lambda: VECTOR_ELE,
        get_kernel_shape=lambda: _KERNEL_SHAPE,
        make_inputs=_make_inputs,
        expected=_expected,
        unsupported_case=_is_unsupported_case,
        print_skip=_print_skip,
        script_path=Path(__file__).resolve(),
        float_dtypes=frozenset(),
        input_count=1,
        output_count=1,
    )
)


def main() -> int:
    return HARNESS.main()


if __name__ == "__main__":
    raise SystemExit(main())
