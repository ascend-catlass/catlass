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

from vector_op_harness import (
    DirectVectorOpConfig,
    DirectVectorOpHarness,
    vector_kernel_config,
)

VECTOR_ELE = 64
VL_ELE = 64
LOOPS = 1
ALL_DTYPES = ("f32", "f16", "i8", "i16", "i32", "u8", "u16", "u32")
_INT_DTYPES = frozenset({"i8", "i16", "i32", "u8", "u16", "u32"})
_UNSIGNED = frozenset({"u8", "u16", "u32"})
# Unsigned inputs are shifted above the signed boundary so the full unsigned
# range is exercised (u8 > 127, u16 > 32767, u32 > 2**31 - 1).
_UNSIGNED_BIAS = {"u8": 128, "u16": 32768, "u32": 0x80000000}

_KERNEL_DTYPE = tla.Float32
_KERNEL_ELEMENT_BYTES = 4
_KERNEL_SHAPE = (VECTOR_ELE,)
_MASK_PATTERN = tla.mask.M4


@tla.kernel
def squeeze_op(mem_src: tla.Tensor, mem_dst: tla.Tensor) -> None:
    ub_loaded = tla.flag("ub_loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    vec_done = tla.flag("vec_done", tla.pipes.VECTOR, tla.pipes.MTE3)

    src_gm = tla.tile_view(mem_src, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    dst_gm = tla.tile_view(mem_dst, tla.make_shape(VECTOR_ELE), tla.make_coord(0))

    src_ub_ptr = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    dst_ub_ptr = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    src_ub = tla.make_tensor_like(src_ub_ptr, src_gm, tla.arch.RowMajor)
    dst_ub = tla.make_tensor_like(dst_ub_ptr, dst_gm, tla.arch.RowMajor)

    with tla.vector():
        tla.copy(src_ub, src_gm)

        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)
        with tla.vec.func(mode="simd"):
            remaining = VECTOR_ELE
            for i in tla.range(LOOPS):
                src_tile = tla.tile_view(
                    src_ub, tla.make_shape(VL_ELE), tla.make_coord(i)
                )
                dst_tile = tla.tile_view(
                    dst_ub, tla.make_shape(VL_ELE), tla.make_coord(i)
                )
                src_vec = src_tile.load()
                tail, remaining = tla.update_mask(remaining, dtype=_KERNEL_DTYPE)
                m4 = tla.create_mask(pattern=_MASK_PATTERN, dtype=_KERNEL_DTYPE)
                active = tla.bitwise_and(m4, tail)
                dst_tile.store(tla.squeeze(src_vec, active), mask=tail)

        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)

        tla.copy(dst_gm, dst_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def _operator_specs() -> dict[str, dict[str, Any]]:
    return {
        "squeeze": {
            "default_atol": 1e-4,
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
    global \
        VL_ELE, \
        LOOPS, \
        VECTOR_ELE, \
        _KERNEL_DTYPE, \
        _KERNEL_ELEMENT_BYTES, \
        _KERNEL_SHAPE
    if op_name not in _operator_specs():
        raise SystemExit("unknown squeeze operator")
    config = vector_kernel_config(dtype_name, shape, ALL_DTYPES)
    VECTOR_ELE = config.vector_elements
    _KERNEL_SHAPE = shape if shape is not None else (VECTOR_ELE,)
    VL_ELE = config.lanes
    LOOPS = max(1, config.loops)
    _KERNEL_DTYPE = config.tla_dtype
    _KERNEL_ELEMENT_BYTES = config.element_bytes
    return config.tla_dtype, config.torch_dtype, config.default_sentinel


def _make_inputs(args: Any, dtype_name: str, torch: Any) -> tuple[Any, ...]:
    _, dtype, _ = _set_kernel_config(args.op, dtype_name, args.shape)
    device = "npu"
    if dtype_name in _INT_DTYPES:
        # NPU arange lacks most integer dtypes; build on CPU then move to NPU.
        idx = torch.arange(VECTOR_ELE, dtype=torch.int64)
        if dtype_name in _UNSIGNED:
            vals = (idx % 50) + (_UNSIGNED_BIAS[dtype_name] - 10)
        else:
            vals = (idx % 50) - 20
        return (vals.to(dtype).to(device),)
    idx = torch.arange(VECTOR_ELE, dtype=torch.float32, device=device)
    return ((idx * 0.5 + 1.0).to(dtype),)


def _expected(op_name: str, inputs: tuple[Any, ...]) -> Any:
    del op_name
    import torch

    # Compute on CPU (the harness compares in CPU space). Use a strided basic
    # slice rather than boolean-mask indexing: some CPU torch builds do not
    # implement advanced "index" for UInt16/UInt32 (e.g. index_cpu for UInt16),
    # while chunk[::4] is a plain view supported for every dtype.
    src = inputs[0].detach().cpu()
    out = torch.zeros_like(src)
    for i in range(LOOPS):
        start = i * VL_ELE
        if start >= src.numel():
            break
        end = min(start + VL_ELE, src.numel())
        chunk = src[start:end]
        # M4 keeps lanes 0, 4, 8, ... == every 4th element.
        packed = chunk[::4]
        out[start : start + packed.numel()] = packed
    return out


HARNESS = DirectVectorOpHarness(
    DirectVectorOpConfig(
        description="Compile and run a vector squeeze kernel (M4 mask).",
        kernel=squeeze_op,
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
    )
)


def main() -> int:
    return HARNESS.main()


if __name__ == "__main__":
    raise SystemExit(main())
