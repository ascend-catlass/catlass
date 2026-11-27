# -------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -------------------------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Any

import catlass.tla as tla
from catlass.params import BlockLoadParams

from vector_op_harness import (
    DirectVectorOpConfig,
    DirectVectorOpHarness,
    vector_kernel_config,
)

# Strided block load (AscendC vsldb): one instruction gathers 8 DataBlocks
# whose heads are block_stride DataBlocks (32B) apart. Three modes are covered:
#   stride: block_stride=4 (sparse gather)
#   repeat: block_stride=0 (first DataBlock replicated 8x)
#   post:   block_stride=1 + repeat_stride=2 (POST_MODE_NORMAL pre-offset,
#           equal to a contiguous load starting 2 DataBlocks ahead)
VECTOR_ELE = 512
ALL_DTYPES = ("f32", "i32", "f16")

VL_ELE = 64
LOOPS = VECTOR_ELE // VL_ELE
DB_ELE = VL_ELE // 8  # lanes per 32-byte DataBlock
BLOCK_STRIDE = 4
POST_STRIDE = 0
# Per-loop source advance in DataBlocks, rounded up to a whole number of VL
# tiles: a tla.tile_view coord counts tiles (one VL lane group), not elements,
# so the advance must be a multiple of 8 DataBlocks (256B).
ADV_DB = 32
SRC_ELE = LOOPS * ADV_DB * DB_ELE
OUT_VALID_ELE = VECTOR_ELE

_KERNEL_DTYPE = tla.Float32
_KERNEL_SHAPE = (VECTOR_ELE,)
_KERNEL_SENTINEL: int = -7


def _load_params() -> BlockLoadParams:
    return BlockLoadParams(block_stride=BLOCK_STRIDE, repeat_stride=POST_STRIDE)


@tla.kernel
def load_stride_op(mem_src: tla.Tensor, mem_dst: tla.Tensor) -> None:
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
                # VL-wide tile view only supplies the gather base address; the
                # coord counts VL tiles, so one loop advances ADV_DB//8 tiles
                # (the 8 gathered DataBlocks may reach beyond the view width).
                src_tile = tla.tile_view(
                    src_ub, tla.make_shape(VL_ELE), tla.make_coord(i * (ADV_DB // 8))
                )
                dst_tile = tla.tile_view(
                    dst_ub, tla.make_shape(VL_ELE), tla.make_coord(i)
                )
                stride_reg = src_tile.load(_load_params())
                dst_tile.store(stride_reg)

        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)

        tla.copy(dst_gm, dst_ub)

        tla.pipe_barrier(tla.pipes.ALL)


def _make_ub_tensor(like_tensor: Any, num_ele: int) -> Any:
    ptr = tla.allocate(num_ele, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    return tla.make_tensor_like(ptr, like_tensor, tla.arch.RowMajor)


def _operator_specs() -> dict[str, dict[str, Any]]:
    return {
        "stride": {"default_atol": 0.0, "dtypes": ALL_DTYPES},
        "repeat": {"default_atol": 0.0, "dtypes": ALL_DTYPES},
        "post": {"default_atol": 0.0, "dtypes": ALL_DTYPES},
    }


def _is_unsupported_case(op_name: str, dtype_name: str) -> bool:
    return op_name not in _operator_specs() or dtype_name not in ALL_DTYPES


def _print_skip(op_name: str, dtype_name: str, shape: tuple[int, ...]) -> None:
    del shape
    print(f"skip op={op_name} dtype={dtype_name}: stride load covers {ALL_DTYPES}")


def _set_kernel_config(
    op_name: str, dtype_name: str, shape: tuple[int, ...] | None = None
) -> tuple[type[Any], Any, float | int]:
    global VL_ELE, LOOPS, DB_ELE, BLOCK_STRIDE, POST_STRIDE
    global ADV_DB, SRC_ELE, OUT_VALID_ELE
    global _KERNEL_DTYPE, _KERNEL_SHAPE, _KERNEL_SENTINEL
    if op_name not in _operator_specs():
        raise SystemExit(f"unknown load_stride operator {op_name!r}")

    del shape
    config = vector_kernel_config(dtype_name, (VECTOR_ELE,), ALL_DTYPES)
    VL_ELE = config.lanes
    LOOPS = VECTOR_ELE // VL_ELE
    DB_ELE = VL_ELE // 8
    if op_name == "stride":
        # Sparse gather: DataBlock j head is 4 DataBlocks after DataBlock j-1.
        BLOCK_STRIDE, POST_STRIDE = 4, 0
    elif op_name == "repeat":
        # block_stride == 0 repeats the first DataBlock into all 8 slots.
        BLOCK_STRIDE, POST_STRIDE = 0, 0
    else:
        # POST_MODE_NORMAL pre-offset of 2 DataBlocks, then contiguous read.
        BLOCK_STRIDE, POST_STRIDE = 1, 2
    # Round the touched span (pre-offset + 8 gathered DataBlocks) up to whole
    # VL tiles: tile_view coords count tiles, not elements.
    span_db = POST_STRIDE + 7 * BLOCK_STRIDE + 1
    ADV_DB = ((span_db + 7) // 8) * 8
    SRC_ELE = LOOPS * ADV_DB * DB_ELE
    OUT_VALID_ELE = LOOPS * VL_ELE
    _KERNEL_SHAPE = (VECTOR_ELE,)
    _KERNEL_DTYPE = config.tla_dtype
    _KERNEL_SENTINEL = config.default_sentinel
    return config.tla_dtype, config.torch_dtype, config.default_sentinel


def _make_inputs(args: Any, dtype_name: str, torch: Any) -> tuple[Any, ...]:
    _, dtype, _ = _set_kernel_config(args.op, dtype_name, args.shape)
    device = "npu"
    # Distinct values so every gathered DataBlock is uniquely identifiable.
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
    for i in range(LOOPS):
        base = i * ADV_DB * DB_ELE
        # vsldb gathers 8 DataBlocks: DataBlock j head sits at
        # base + post_stride + j * block_stride (all in DataBlock units).
        heads = [base + (POST_STRIDE + j * BLOCK_STRIDE) * DB_ELE for j in range(8)]
        gathered = torch.cat([src[head : head + DB_ELE] for head in heads], dim=0)
        out_lo = i * VL_ELE
        dst[out_lo : out_lo + VL_ELE] = gathered
    return (dst,)


HARNESS = DirectVectorOpHarness(
    DirectVectorOpConfig(
        description=("Compile and run vsldb strided block load (stride/repeat/post)."),
        kernel=load_stride_op,
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
        input_count=1,
        output_count=1,
    )
)


def main() -> int:
    return HARNESS.main()


if __name__ == "__main__":
    raise SystemExit(main())
