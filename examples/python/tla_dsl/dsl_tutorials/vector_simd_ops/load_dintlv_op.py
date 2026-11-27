# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""E2E: DIST_DINTLV_B32 dual-destination interleaved UB load (f32 only).

Each iteration points a VL-wide tile at offset ``2*i*VL`` (start of a ``2*VL``
span), loads with ``DIST_DINTLV_B32``, and stores the even/odd VL registers.

i32/u32 are rejected: AscendNPU-IR lowers DINTLV_B32 only to ``vldsx2.v64f32``.
"""

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

# Multiple of 2*VL for f32 (VL=64).
VECTOR_ELE = 512
VL_ELE = 64
LOOPS = VECTOR_ELE // (2 * VL_ELE)
OUT_VALID_ELE = LOOPS * VL_ELE
ALL_DTYPES = ("f32",)

_KERNEL_DTYPE = tla.Float32
_KERNEL_ELEMENT_BYTES = 4
_KERNEL_SHAPE = (VECTOR_ELE,)
_KERNEL_SENTINEL: float | int = -7.0
_DINTLV_LOAD = NormalLoadParams(load_dist=LoadDist.DIST_DINTLV_B32)


@tla.kernel
def load_dintlv_op(
    mem_src: tla.Tensor,
    mem_even: tla.Tensor,
    mem_odd: tla.Tensor,
) -> None:
    ub_loaded = tla.flag("ub_loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    vec_done = tla.flag("vec_done", tla.pipes.VECTOR, tla.pipes.MTE3)

    src_gm = tla.tile_view(mem_src, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    # Copy back only the lanes DINTLV writes; host-initialized sentinel remains
    # in the unused half of mem_even / mem_odd.
    even_gm = tla.tile_view(mem_even, tla.make_shape(OUT_VALID_ELE), tla.make_coord(0))
    odd_gm = tla.tile_view(mem_odd, tla.make_shape(OUT_VALID_ELE), tla.make_coord(0))

    src_ub = _make_ub_tensor(src_gm, VECTOR_ELE)
    even_ub = _make_ub_tensor(even_gm, OUT_VALID_ELE)
    odd_ub = _make_ub_tensor(odd_gm, OUT_VALID_ELE)

    with tla.vector():
        tla.copy(src_ub, src_gm)

        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)

        with tla.vec.func(mode="simd"):
            for i in tla.range(LOOPS):
                # VL-wide view at element offset 2*i*VL: AVE DINTLV reads 2*VL
                # from this base (same convention as HIVMAVE ProcessVsstb).
                src_tile = tla.tile_view(
                    src_ub, tla.make_shape(VL_ELE), tla.make_coord(i * 2)
                )
                even_tile = tla.tile_view(
                    even_ub, tla.make_shape(VL_ELE), tla.make_coord(i)
                )
                odd_tile = tla.tile_view(
                    odd_ub, tla.make_shape(VL_ELE), tla.make_coord(i)
                )
                even_reg, odd_reg = src_tile.load(_DINTLV_LOAD)
                even_tile.store(even_reg)
                odd_tile.store(odd_reg)

        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)

        tla.copy(even_gm, even_ub)
        tla.copy(odd_gm, odd_ub)

        tla.pipe_barrier(tla.pipes.ALL)


def _make_ub_tensor(like_tensor: Any, num_ele: int) -> Any:
    ptr = tla.allocate(num_ele, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    return tla.make_tensor_like(ptr, like_tensor, tla.arch.RowMajor)


def _operator_specs() -> dict[str, dict[str, Any]]:
    return {
        "dintlv_b32": {
            "default_atol": 0.0,
            "dtypes": ALL_DTYPES,
        },
    }


def _is_unsupported_case(op_name: str, dtype_name: str) -> bool:
    del op_name
    return dtype_name not in ALL_DTYPES


def _print_skip(op_name: str, dtype_name: str, shape: tuple[int, ...]) -> None:
    del shape
    print(
        f"skip op={op_name} dtype={dtype_name}: DIST_DINTLV_B32 currently requires f32"
    )


def _set_kernel_config(
    op_name: str, dtype_name: str, shape: tuple[int, ...] | None = None
) -> tuple[type[Any], Any, float | int]:
    global VL_ELE, LOOPS, OUT_VALID_ELE
    global _KERNEL_DTYPE, _KERNEL_ELEMENT_BYTES, _KERNEL_SHAPE, _KERNEL_SENTINEL
    if op_name not in _operator_specs():
        raise SystemExit(f"unknown load_dintlv operator {op_name!r}")

    del shape
    if dtype_name != "f32":
        raise SystemExit(f"DIST_DINTLV_B32 currently requires f32, got {dtype_name}")
    config = vector_kernel_config(dtype_name, (VECTOR_ELE,), ALL_DTYPES)
    if VECTOR_ELE % (2 * config.lanes) != 0:
        raise SystemExit(
            f"VECTOR_ELE={VECTOR_ELE} must be a multiple of 2*VL={2 * config.lanes}"
        )

    _KERNEL_SHAPE = (VECTOR_ELE,)
    VL_ELE = config.lanes
    LOOPS = VECTOR_ELE // (2 * VL_ELE)
    OUT_VALID_ELE = LOOPS * VL_ELE
    _KERNEL_DTYPE = config.tla_dtype
    _KERNEL_ELEMENT_BYTES = config.element_bytes
    _KERNEL_SENTINEL = config.default_sentinel
    return config.tla_dtype, config.torch_dtype, config.default_sentinel


def _make_inputs(args: Any, dtype_name: str, torch: Any) -> tuple[Any, ...]:
    _, dtype, _ = _set_kernel_config(args.op, dtype_name, args.shape)
    device = "npu"
    idx = torch.arange(VECTOR_ELE, dtype=torch.float32, device=device)
    src = ((idx % 37.0) * 0.25 + 1.0).to(dtype)
    return (src,)


def _expected(op_name: str, inputs: tuple[Any, ...]) -> tuple[Any, Any]:
    del op_name
    import torch

    (src,) = inputs
    # Kernel writes LOOPS*VL lanes and leaves the host sentinel in the rest.
    even = torch.full_like(src, _KERNEL_SENTINEL)
    odd = torch.full_like(src, _KERNEL_SENTINEL)
    even[:OUT_VALID_ELE] = src[0::2][:OUT_VALID_ELE]
    odd[:OUT_VALID_ELE] = src[1::2][:OUT_VALID_ELE]
    return even, odd


HARNESS = DirectVectorOpHarness(
    DirectVectorOpConfig(
        description=(
            "Compile and run DIST_DINTLV_B32 dual-destination interleaved load (f32)."
        ),
        kernel=load_dintlv_op,
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
        float_dtypes=frozenset({"f32"}),
        output_count=2,
    )
)


def main() -> int:
    return HARNESS.main()


if __name__ == "__main__":
    raise SystemExit(main())
