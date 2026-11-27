# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Any

import catlass.tla as tla
from catlass.params import NormalStoreParams, StoreDist

from vector_op_harness import (
    DirectVectorOpConfig,
    DirectVectorOpHarness,
    vector_kernel_config,
)

VECTOR_ELE = 256
VL_ELE = 64
LOOPS = (VECTOR_ELE + VL_ELE - 1) // VL_ELE
ALL_DTYPES = ("i32", "i16", "i8", "f32", "f16")

# first_element (AscendC DIST_FIRST_ELEMENT_* / asc_storealign_1st, HIVMAVE
# ONEPT_*): the store ignores the mask and writes only lane 0 (the first
# element) of the source register to the dst base. The _b8/_b16/_b32 suffix
# fixes the stored element width, so it must match the kernel dtype.
FIRST_ELEMENT_CONFIGS = {
    "i32": StoreDist.DIST_FIRST_ELEMENT_B32,
    "f32": StoreDist.DIST_FIRST_ELEMENT_B32,
    "i16": StoreDist.DIST_FIRST_ELEMENT_B16,
    "f16": StoreDist.DIST_FIRST_ELEMENT_B16,
    "i8": StoreDist.DIST_FIRST_ELEMENT_B8,
}

_KERNEL_DTYPE = tla.Int32
_KERNEL_TORCH_DTYPE = None
_KERNEL_ELEMENT_BYTES = 4
_KERNEL_SENTINEL = -7
_STORE_DIST = StoreDist.DIST_FIRST_ELEMENT_B32
_KERNEL_SHAPE = (VECTOR_ELE,)


def _make_ub_tensor(
    like_tensor: Any,
    dtype: type[Any],
    element_bytes: int,
) -> Any:
    alignment = 512 if element_bytes == 8 else 256
    ptr = tla.allocate(VECTOR_ELE, dtype, tla.AddressSpace.ub, alignment)
    return tla.make_tensor_like(ptr, like_tensor, tla.arch.RowMajor)


@tla.kernel
def store_first_element(
    mem_a: tla.Tensor,
    mem_out: tla.Tensor,
) -> None:
    ub_loaded = tla.flag("ub_loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    vec_done = tla.flag("vec_done", tla.pipes.VECTOR, tla.pipes.MTE3)

    a_gm = tla.tile_view(mem_a, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    out_gm = tla.tile_view(mem_out, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    a_ub = _make_ub_tensor(a_gm, _KERNEL_DTYPE, _KERNEL_ELEMENT_BYTES)
    out_ub = _make_ub_tensor(out_gm, _KERNEL_DTYPE, _KERNEL_ELEMENT_BYTES)

    with tla.vector():
        tla.copy(a_ub, a_gm)
        # Seed the output UB buffer with the GM sentinel baseline: ONEPT writes
        # only lane 0 of each tile, so the untouched lanes must keep the
        # sentinel to prove the register tail did not leak into UB.
        tla.copy(out_ub, out_gm)

        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)
        with tla.vec.func(mode="simd"):
            for i in tla.range(LOOPS):
                a_tile = tla.tile_view(a_ub, tla.make_shape(VL_ELE), tla.make_coord(i))
                # Each first_element store writes only the first element of the
                # loaded register, i.e. a[i * VL_ELE]; place the dst window at
                # the same offset so the two sides stay comparable. The tile
                # base stays 32B-aligned because VL_ELE * element_bytes >= 64.
                out_tile = tla.tile_view(
                    out_ub, tla.make_shape(VL_ELE), tla.make_coord(i)
                )

                a_reg = a_tile.load()
                out_tile.store(a_reg, NormalStoreParams(store_dist=_STORE_DIST))

        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)
        tla.copy(out_gm, out_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def _operator_specs() -> dict[str, dict[str, Any]]:
    return {
        "store_first_element": {
            "default_atol": 0,
        }
    }


def _set_kernel_config(
    op_name: str,
    dtype_name: str,
    shape: tuple[int, ...] | None = None,
) -> tuple[type[Any], Any, float | int]:
    global \
        VL_ELE, \
        LOOPS, \
        VECTOR_ELE, \
        _KERNEL_DTYPE, \
        _KERNEL_TORCH_DTYPE, \
        _KERNEL_ELEMENT_BYTES
    global _KERNEL_SENTINEL
    global _STORE_DIST
    global _KERNEL_SHAPE
    specs = _operator_specs()
    if op_name not in specs:
        choices = ", ".join(sorted(specs))
        raise SystemExit(f"unknown op {op_name!r}; expected one of: {choices}")
    config = vector_kernel_config(dtype_name, shape, ALL_DTYPES)
    VECTOR_ELE = config.vector_elements
    _KERNEL_SHAPE = shape if shape is not None else (VECTOR_ELE,)
    VL_ELE = config.lanes
    LOOPS = config.loops
    _KERNEL_DTYPE = config.tla_dtype
    _KERNEL_TORCH_DTYPE = config.torch_dtype
    _KERNEL_ELEMENT_BYTES = config.element_bytes
    _KERNEL_SENTINEL = config.default_sentinel
    _STORE_DIST = FIRST_ELEMENT_CONFIGS[dtype_name]
    return config.tla_dtype, config.torch_dtype, config.default_sentinel


def _make_inputs(args: Any, dtype_name: str, torch: Any) -> tuple[Any, ...]:
    """Fill input data with common and corner testcases"""
    _, _, _ = _set_kernel_config(args.op, dtype_name, args.shape)
    if dtype_name in ("i16", "f16"):
        values = (
            0,
            1,
            -1,
            33,
            67,
            127,
            -129,
            0x1234,
            -0x1234,
            32767,
            -32768,
        )
    else:
        values = (
            0,
            1,
            -1,
            33,
            67,
            127,
            -128,
            0x12,
            -0x12,
            0x7F,
            -0x80,
        )
    pattern = torch.tensor(values, dtype=_KERNEL_TORCH_DTYPE, device="npu")
    repeats = (VECTOR_ELE + len(values) - 1) // len(values)
    a = pattern.repeat(repeats)[:VECTOR_ELE].contiguous()
    return (a,)


def _expected(_op_name: str, inputs: tuple[Any, ...]) -> Any:
    """Compute the first_element store expectation.

    Each of the LOOPS iterations loads a VL-wide tile and the ONEPT store
    writes only lane 0 to the tile base, so exactly one element per loop
    survives:

        a:   [t0 .................. | t1 .................. | ...]
                                           t0[0] survives
        out: [t0[0] sentinel ...   | t1[0] sentinel ...    | ...]

    Everything that was not written must keep the harness sentinel, which also
    proves the remaining lanes of the register did not leak to UB.
    """
    import torch

    a = inputs[0]
    result = torch.full_like(a, _KERNEL_SENTINEL)
    for i in range(LOOPS):
        base = i * VL_ELE
        result[base] = a[base]
    return result


HARNESS = DirectVectorOpHarness(
    DirectVectorOpConfig(
        description=(
            "Compile and run first-element stores: i32/f32 -> "
            "DIST_FIRST_ELEMENT_B32, i16/f16 -> DIST_FIRST_ELEMENT_B16, "
            "i8 -> DIST_FIRST_ELEMENT_B8. Each store writes only lane 0 of "
            "the source register to the dst tile base."
        ),
        kernel=store_first_element,
        all_dtypes=ALL_DTYPES,
        operator_specs=_operator_specs,
        set_kernel_config=_set_kernel_config,
        get_vector_elements=lambda: VECTOR_ELE,
        get_kernel_shape=lambda: _KERNEL_SHAPE,
        make_inputs=_make_inputs,
        expected=_expected,
        unsupported_case=lambda _op, _dtype: False,
        print_skip=lambda _op, _dtype, _shape: None,
        script_path=Path(__file__).resolve(),
        float_dtypes=frozenset({"f32", "f16"}),
        input_count=1,
        output_count=1,
    )
)


def main() -> int:
    return HARNESS.main()


if __name__ == "__main__":
    raise SystemExit(main())
