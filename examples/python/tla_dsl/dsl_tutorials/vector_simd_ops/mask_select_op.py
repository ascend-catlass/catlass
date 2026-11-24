# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Run MaskSSA select and observe its predicate result through vector select."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vector_op_harness import (
    DirectVectorOpConfig,
    DirectVectorOpHarness,
    vector_kernel_config,
)

from catlass import tla

VECTOR_ELE = 512
VL_ELE = 64
LOOPS = 8
ALL_DTYPES = (
    "bool",
    "i8",
    "u8",
    "i16",
    "u16",
    "i32",
    "u32",
    "f16",
    "bf16",
    "f32",
    "i64",
    "u64",
    "f8e4m3fn",
    "f8e5m2",
    "hif8",
    "f4e2m1x2",
    "f4e1m2x2",
)

_KERNEL_DTYPE = tla.Float32
_KERNEL_SHAPE = (VECTOR_ELE,)


@tla.kernel
def mask_select_op(
    mem_data_a: tla.Tensor,
    mem_data_b: tla.Tensor,
    mem_out: tla.Tensor,
) -> None:
    ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    vec_done = tla.flag("vec_done", tla.arch.VECTOR, tla.arch.MTE3)

    data_a_gm = tla.tile_view(mem_data_a, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    data_b_gm = tla.tile_view(mem_data_b, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    out_gm = tla.tile_view(mem_out, tla.make_shape(VECTOR_ELE), tla.make_coord(0))

    data_a_ub = _make_ub_tensor(data_a_gm)
    data_b_ub = _make_ub_tensor(data_b_gm)
    out_ub = _make_ub_tensor(out_gm)

    with tla.vector():
        tla.copy(data_a_ub, data_a_gm)
        tla.copy(data_b_ub, data_b_gm)
        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)

        with tla.vec.func(mode="simd"):
            for i in tla.range(LOOPS):
                data_a_t = _chunk(data_a_ub, i)
                data_b_t = _chunk(data_b_ub, i)
                out_t = _chunk(out_ub, i)

                selector = tla.create_mask(pattern=tla.mask.H, dtype=_KERNEL_DTYPE)
                src0 = tla.create_mask(pattern=tla.mask.Q, dtype=_KERNEL_DTYPE)
                src1 = tla.create_mask(pattern=tla.mask.M4, dtype=_KERNEL_DTYPE)
                selected = tla.where(selector, src0, src1)

                data_a = data_a_t.load()
                data_b = data_b_t.load()
                out_t.store(tla.where(selected, data_a, data_b))

        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)
        tla.copy(out_gm, out_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def _make_ub_tensor(like_tensor: Any) -> Any:
    ptr = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    return tla.make_tensor_like(ptr, like_tensor, tla.arch.RowMajor)


def _chunk(tensor: Any, chunk_idx: Any) -> Any:
    return tla.tile_view(tensor, tla.make_shape(VL_ELE), tla.make_coord(chunk_idx))


def _operator_specs() -> dict[str, dict[str, Any]]:
    return {"mask_select": {"default_atol": 0.0}}


def _is_unsupported_case(op_name: str, dtype_name: str) -> bool:
    del op_name, dtype_name
    return False


def _print_skip(op_name: str, dtype_name: str, shape: tuple[int, ...]) -> None:
    del shape
    print(f"skip op={op_name} dtype={dtype_name}: unsupported case")


def _set_kernel_config(
    op_name: str, dtype_name: str, shape: tuple[int, ...] | None = None
) -> tuple[type[Any], Any, float | int]:
    global VECTOR_ELE, VL_ELE, LOOPS, _KERNEL_DTYPE, _KERNEL_SHAPE

    if op_name not in _operator_specs():
        raise SystemExit("unknown mask select operator")
    config = vector_kernel_config(dtype_name, shape or (VECTOR_ELE,), ALL_DTYPES)
    if config.vector_elements % config.lanes:
        raise ValueError("mask select example requires complete vector registers")
    VECTOR_ELE = config.vector_elements
    VL_ELE = config.lanes
    LOOPS = config.loops
    _KERNEL_DTYPE = config.tla_dtype
    _KERNEL_SHAPE = (VECTOR_ELE,)
    return config.tla_dtype, config.torch_dtype, config.default_sentinel


def _make_inputs(args: Any, dtype_name: str, torch: Any) -> tuple[Any, Any]:
    _, dtype, _ = _set_kernel_config(args.op, dtype_name, args.shape)
    index = torch.arange(VECTOR_ELE, dtype=torch.int64)
    if dtype_name == "bool":
        data_a = ((index % 3) == 0).to(torch.bool)
        data_b = ((index % 5) == 0).to(torch.bool)
    else:
        data_a = ((index % 31) + 1).to(dtype)
        data_b = (-(index % 29) - 1).to(dtype)
    return data_a.to("npu"), data_b.to("npu")


def _expected(op_name: str, inputs: tuple[Any, ...]) -> Any:
    if op_name != "mask_select":
        raise AssertionError(op_name)

    import torch

    data_a, data_b = (value.cpu() for value in inputs)
    lane = torch.arange(VL_ELE)
    selector = lane < VL_ELE // 2
    src0 = lane < VL_ELE // 4
    src1 = (lane % 4) == 0
    selected = torch.where(selector, src0, src1).repeat(LOOPS)
    if data_a.dtype in (torch.uint8, torch.uint16, torch.uint32, torch.uint64):
        # torch.where has no unsigned-integer kernels; select in int64 and cast
        # back. Values stay exact for these inputs.
        result = torch.where(
            selected, data_a.to(torch.int64), data_b.to(torch.int64)
        ).to(data_a.dtype)
    else:
        result = torch.where(selected, data_a, data_b)
    return result.to(inputs[0].device)


HARNESS = DirectVectorOpHarness(
    DirectVectorOpConfig(
        description="Compile and run MaskSSA select on NPU.",
        kernel=mask_select_op,
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
        float_dtypes=frozenset({"f16", "bf16", "f32"}),
        input_count=2,
    )
)


def main() -> int:
    return HARNESS.main()


if __name__ == "__main__":
    raise SystemExit(main())
