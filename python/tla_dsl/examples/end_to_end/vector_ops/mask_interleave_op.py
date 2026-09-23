# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Run MaskSSA interleave/deinterleave with the shared vector op harness.

Example: python mask_interleave_op.py mask_interleave --all-dtypes --device 0
Example: python mask_interleave_op.py mask_deinterleave --all-dtypes --device 0
Both output masks contain true and false lanes, observed through vector select.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import catlass.tla as tla

from vector_op_harness import (
    DirectVectorOpConfig,
    DirectVectorOpHarness,
    vector_kernel_config,
)

VECTOR_ELE = 512
VL_ELE = 64
LOOPS = 8
ALL_DTYPES = ("i8", "f16", "f32")  # Covers b8, b16, b32

_KERNEL_DTYPE = tla.Float32
_KERNEL_SHAPE = (VECTOR_ELE,)
_KERNEL_OP = "mask_interleave"


@tla.kernel
def mask_interleave_op(
    mem_data_a: tla.Tensor,
    mem_data_b: tla.Tensor,
    mem_out0: tla.Tensor,
    mem_out1: tla.Tensor,
) -> None:
    ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    vec_done = tla.flag("vec_done", tla.arch.VECTOR, tla.arch.MTE3)

    data_a_gm = tla.tile_view(mem_data_a, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    data_b_gm = tla.tile_view(mem_data_b, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    out0_gm = tla.tile_view(mem_out0, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    out1_gm = tla.tile_view(mem_out1, tla.make_shape(VECTOR_ELE), tla.make_coord(0))

    data_a_ub = _make_ub_tensor(data_a_gm)
    data_b_ub = _make_ub_tensor(data_b_gm)
    out0_ub = _make_ub_tensor(out0_gm)
    out1_ub = _make_ub_tensor(out1_gm)

    with tla.vector():
        tla.copy(data_a_ub, data_a_gm)
        tla.copy(data_b_ub, data_b_gm)

        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)

        with tla.vec.func(mode="simd"):
            for i in tla.range(LOOPS):
                data_a_t = _chunk(data_a_ub, i)
                data_b_t = _chunk(data_b_ub, i)
                out0_t = _chunk(out0_ub, i)
                out1_t = _chunk(out1_ub, i)

                if tla.const_expr(_KERNEL_OP == "mask_interleave"):
                    # H / NOT Q make both interleaved halves nonconstant.
                    mask0 = tla.create_mask(pattern=tla.mask.H, dtype=_KERNEL_DTYPE)
                    mask1 = tla.bitwise_not(
                        tla.create_mask(pattern=tla.mask.Q, dtype=_KERNEL_DTYPE)
                    )
                    mask_out0, mask_out1 = tla.interleave(mask0, mask1)
                else:
                    # Odd boundaries distinguish even/odd outputs, detecting
                    # swapped results or extracting the same struct field twice.
                    mask0, _ = tla.update_mask(VL_ELE // 2 - 1, dtype=_KERNEL_DTYPE)
                    tail, _ = tla.update_mask(VL_ELE // 4 + 1, dtype=_KERNEL_DTYPE)
                    mask1 = tla.bitwise_not(tail)
                    mask_out0, mask_out1 = tla.deinterleave(mask0, mask1)

                # Apply masks to data
                vec_a = data_a_t.load()
                vec_b = data_b_t.load()
                result0 = tla.where(mask_out0, vec_a, vec_b)
                result1 = tla.where(mask_out1, vec_a, vec_b)

                out0_t.store(result0)
                out1_t.store(result1)

        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)

        tla.copy(out0_gm, out0_ub)
        tla.copy(out1_gm, out1_ub)

        tla.pipe_barrier(tla.pipes.ALL)


def _make_ub_tensor(like_tensor: Any) -> Any:
    ptr = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    return tla.make_tensor_like(ptr, like_tensor, tla.arch.RowMajor)


def _chunk(tensor: Any, chunk_idx: Any) -> Any:
    return tla.tile_view(
        tensor,
        tla.make_shape(VL_ELE),
        tla.make_coord(chunk_idx),
    )


def _operator_specs() -> dict[str, dict[str, Any]]:
    return {
        "mask_interleave": {
            "default_atol": 0.0,
        },
        "mask_deinterleave": {
            "default_atol": 0.0,
        },
    }


def _is_unsupported_case(op_name: str, dtype_name: str) -> bool:
    del op_name, dtype_name
    return False


def _print_skip(op_name: str, dtype_name: str, shape: tuple[int, ...]) -> None:
    del shape
    print(f"skip op={op_name} dtype={dtype_name}: unsupported case")


def _set_kernel_config(
    op_name: str, dtype_name: str, shape: tuple[int, ...] | None = None
) -> tuple[type[Any], Any, float | int]:
    global VECTOR_ELE, VL_ELE, LOOPS, _KERNEL_DTYPE
    global _KERNEL_SHAPE, _KERNEL_OP

    if op_name not in _operator_specs():
        raise SystemExit("unknown mask interleave/deinterleave operator")

    config = vector_kernel_config(dtype_name, shape or (VECTOR_ELE,), ALL_DTYPES)
    if config.vector_elements % config.lanes:
        raise ValueError("mask interleave example requires complete vector registers")
    VECTOR_ELE = config.vector_elements
    _KERNEL_SHAPE = (VECTOR_ELE,)
    VL_ELE = config.lanes
    LOOPS = config.loops
    _KERNEL_DTYPE = config.tla_dtype
    _KERNEL_OP = op_name
    return config.tla_dtype, config.torch_dtype, config.default_sentinel


def _make_inputs(args: Any, dtype_name: str, torch: Any) -> tuple[Any, Any]:
    _, dtype, _ = _set_kernel_config(args.op, dtype_name, args.shape)
    device = "npu"

    # Disjoint positive/negative values make every mask bit observable.
    idx = torch.arange(VECTOR_ELE, dtype=torch.int32, device=device)
    return ((idx % 31) + 1).to(dtype), (-(idx % 29) - 1).to(dtype)


def _expected(op_name: str, inputs: tuple[Any, ...]) -> tuple[Any, Any]:
    import torch

    # Compute the reference on CPU, independently of device execution.
    data_a, data_b = (value.cpu() for value in inputs)
    lane = torch.arange(VL_ELE)
    if op_name == "mask_interleave":
        mask0 = lane < VL_ELE // 2
        mask1 = lane >= VL_ELE // 4
        stream = torch.stack((mask0, mask1), dim=1).flatten()
        masks = (stream[:VL_ELE], stream[VL_ELE:])
    elif op_name == "mask_deinterleave":
        mask0 = lane < VL_ELE // 2 - 1
        mask1 = lane >= VL_ELE // 4 + 1
        stream = torch.cat((mask0, mask1))
        masks = (stream[0::2], stream[1::2])
    else:
        raise AssertionError(op_name)
    # Each vector chunk is rearranged independently.
    return tuple(
        torch.where(mask.repeat(LOOPS), data_a, data_b).to(inputs[0].device)
        for mask in masks
    )


HARNESS = DirectVectorOpHarness(
    DirectVectorOpConfig(
        description="Compile and run mask interleave/deinterleave operations on NPU.",
        kernel=mask_interleave_op,
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
        float_dtypes=frozenset({"f32", "f16"}),
        input_count=2,
        output_count=2,
    )
)


def main() -> int:
    return HARNESS.main()


if __name__ == "__main__":
    raise SystemExit(main())
