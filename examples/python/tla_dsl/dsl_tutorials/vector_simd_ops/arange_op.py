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

VECTOR_ELE = 400
VL_ELE = 64
LOOPS = (VECTOR_ELE + VL_ELE - 1) // VL_ELE
ALL_DTYPES = ("i8", "i16", "i32", "f16", "f32")

_KERNEL_DTYPE = tla.Int32
_IS_FLOAT_DTYPE = False
_KERNEL_TORCH_DTYPE = None
_KERNEL_ELEMENT_BYTES = 4
_KERNEL_SHAPE = (VECTOR_ELE,)
_ARANGE_ORDER = "increase"
_BATCH_OPS = ("increase",) * 4


@tla.kernel
def arange_op(mem_out: tla.Tensor) -> None:
    ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    vec_done = tla.flag("vec_done", tla.arch.VECTOR, tla.arch.MTE3)

    out_gm = tla.tile_view(mem_out, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    out_ub_ptr = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    out_ub = tla.make_tensor_like(out_ub_ptr, out_gm, tla.arch.RowMajor)

    with tla.vector():
        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)
        with tla.vec.func(mode="simd"):
            # Floating-point arange requires a compile-time constant base, so
            # unroll its chunk loop; integer dtypes keep the dynamic loop.
            if tla.const_expr(_IS_FLOAT_DTYPE):
                for i in tla.range_constexpr(LOOPS):
                    out_tile = tla.tile_view(
                        out_ub, tla.make_shape(VL_ELE), tla.make_coord(i)
                    )
                    out_tile.store(
                        tla.arange(i * VL_ELE, order=_ARANGE_ORDER, dtype=_KERNEL_DTYPE)
                    )
            else:
                for i in tla.range(LOOPS):
                    out_tile = tla.tile_view(
                        out_ub, tla.make_shape(VL_ELE), tla.make_coord(i)
                    )
                    out_tile.store(
                        tla.arange(i * VL_ELE, order=_ARANGE_ORDER, dtype=_KERNEL_DTYPE)
                    )

        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)
        tla.copy(out_gm, out_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def _batch_arange_store(out_tile: Any, chunk_start: Any, op_name: str) -> None:
    out_tile.store(tla.arange(chunk_start, order=op_name, dtype=_KERNEL_DTYPE))


@tla.kernel
def arange_op_batch(mem_out: tla.Tensor) -> None:
    ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    vec_done = tla.flag("vec_done", tla.arch.VECTOR, tla.arch.MTE3)
    block_idx = tla.arch.block_idx()
    out_gm = tla.tile_view(
        mem_out, tla.make_shape(VECTOR_ELE), tla.make_coord(block_idx)
    )
    out_ub_ptr = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    out_ub = tla.make_tensor_like(out_ub_ptr, out_gm, tla.arch.RowMajor)

    with tla.vector():
        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)
        with tla.vec.func(mode="simd"):
            # Floating-point arange requires a compile-time constant base, so
            # unroll its chunk loop; integer dtypes keep the dynamic loop.
            if tla.const_expr(_IS_FLOAT_DTYPE):
                for i in tla.range_constexpr(LOOPS):
                    out_tile = tla.tile_view(
                        out_ub, tla.make_shape(VL_ELE), tla.make_coord(i)
                    )
                    chunk_start = i * VL_ELE
                    if block_idx == 0:
                        _batch_arange_store(out_tile, chunk_start, _BATCH_OPS[0])
                    elif block_idx == 1:
                        _batch_arange_store(out_tile, chunk_start, _BATCH_OPS[1])
                    elif block_idx == 2:
                        _batch_arange_store(out_tile, chunk_start, _BATCH_OPS[2])
                    else:
                        _batch_arange_store(out_tile, chunk_start, _BATCH_OPS[3])
            else:
                for i in tla.range(LOOPS):
                    out_tile = tla.tile_view(
                        out_ub, tla.make_shape(VL_ELE), tla.make_coord(i)
                    )
                    chunk_start = i * VL_ELE
                    if block_idx == 0:
                        _batch_arange_store(out_tile, chunk_start, _BATCH_OPS[0])
                    elif block_idx == 1:
                        _batch_arange_store(out_tile, chunk_start, _BATCH_OPS[1])
                    elif block_idx == 2:
                        _batch_arange_store(out_tile, chunk_start, _BATCH_OPS[2])
                    else:
                        _batch_arange_store(out_tile, chunk_start, _BATCH_OPS[3])
        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)
        tla.copy(out_gm, out_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def _operator_specs() -> dict[str, dict[str, Any]]:
    return {
        "increase": {
            "default_atol": 0,
        },
        "decrease": {
            "default_atol": 0,
        },
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
        _IS_FLOAT_DTYPE, \
        _KERNEL_TORCH_DTYPE, \
        _KERNEL_ELEMENT_BYTES
    global _KERNEL_SHAPE, _ARANGE_ORDER
    specs = _operator_specs()
    if op_name not in specs:
        choices = ", ".join(sorted(specs))
        raise SystemExit(
            f"unknown arange variant {op_name!r}; expected one of: {choices}"
        )
    _ARANGE_ORDER = op_name
    config = vector_kernel_config(dtype_name, shape, ALL_DTYPES)
    VECTOR_ELE = config.vector_elements
    _KERNEL_SHAPE = shape if shape is not None else (VECTOR_ELE,)
    VL_ELE = config.lanes
    LOOPS = config.loops
    _KERNEL_DTYPE = config.tla_dtype
    _IS_FLOAT_DTYPE = config.tla_dtype in (tla.Float16, tla.Float32)
    _KERNEL_TORCH_DTYPE = config.torch_dtype
    _KERNEL_ELEMENT_BYTES = config.element_bytes
    return config.tla_dtype, config.torch_dtype, config.default_sentinel


def _configure_batch(
    ops: tuple[str, ...], dtype_name: str, shape: tuple[int, ...]
) -> tuple[type[Any], Any, float | int]:
    global VL_ELE, LOOPS, VECTOR_ELE, _KERNEL_DTYPE, _IS_FLOAT_DTYPE
    global _KERNEL_TORCH_DTYPE
    global _KERNEL_ELEMENT_BYTES, _KERNEL_SHAPE, _BATCH_OPS
    if not 1 <= len(ops) <= 4:
        raise SystemExit("arange batch requires one to four operations")
    config = vector_kernel_config(dtype_name, shape, ALL_DTYPES)
    VECTOR_ELE = config.vector_elements
    _KERNEL_SHAPE = shape
    VL_ELE = config.lanes
    LOOPS = config.loops
    _KERNEL_DTYPE = config.tla_dtype
    _IS_FLOAT_DTYPE = config.tla_dtype in (tla.Float16, tla.Float32)
    _KERNEL_TORCH_DTYPE = config.torch_dtype
    _KERNEL_ELEMENT_BYTES = config.element_bytes
    _BATCH_OPS = ops + (ops[-1],) * (4 - len(ops))
    return config.tla_dtype, config.torch_dtype, config.default_sentinel


def _make_inputs(args: Any, dtype_name: str, _torch: Any) -> tuple[Any, ...]:
    """Arange is output-only; sync module globals and return no GM input tensors."""
    _, _, _ = _set_kernel_config(args.op, dtype_name, args.shape)
    return tuple()


def _expected(op_name: str, _inputs: tuple[Any, ...]) -> Any:
    import torch

    if op_name == "decrease":
        result = torch.empty(VECTOR_ELE, dtype=torch.int64)
        for i in range(LOOPS):
            start = i * VL_ELE
            end = min((i + 1) * VL_ELE, VECTOR_ELE)
            block_len = end - start
            result[start:end] = torch.arange(
                start + VL_ELE - 1, start + VL_ELE - 1 - block_len, -1
            )
        idx = result
    elif op_name == "increase":
        idx = torch.arange(VECTOR_ELE, dtype=torch.int64, device="cpu")
    else:
        raise ValueError("mode can only be 'increase' or 'decrease' for tla.arange")
    return idx.to(dtype=_KERNEL_TORCH_DTYPE, device="npu")


HARNESS = DirectVectorOpHarness(
    DirectVectorOpConfig(
        description="Compile and run vector arange kernels.",
        kernel=arange_op,
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
        float_dtypes=frozenset({"f16", "f32"}),
        input_count=0,
        output_count=1,
        batch_kernel=arange_op_batch,
        configure_batch=_configure_batch,
    )
)


def main() -> int:
    return HARNESS.main()


if __name__ == "__main__":
    raise SystemExit(main())
