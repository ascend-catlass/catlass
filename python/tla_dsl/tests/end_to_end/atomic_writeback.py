# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Check two producers writing MAX, MIN, and ordinary GM tiles on Ascend 950."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from catlass import tla

example_root = str(
    Path(__file__).resolve().parents[4] / "examples" / "python" / "tla_dsl"
)
if example_root not in sys.path:
    sys.path.append(example_root)

from helpers import create_tla_tensor
from tla.kernel.matmul.basic_mmad.basic_matmul_fixpipe_per_channel import (
    encode_per_channel_quant_tensor,
)

SIDE = 32


@tla.kernel
def vector_writeback(src: tla.Tensor, out: tla.Tensor) -> None:
    ptr = tla.allocate(SIDE * SIDE, src.ptr.dtype, tla.AddressSpace.ub, 256)
    loaded = tla.flag("loaded", tla.pipes.MTE2, tla.pipes.MTE3)
    with tla.vector():
        block = tla.arch.block_idx()
        shape = tla.make_shape(SIDE, SIDE)
        tile = tla.tile_view(src, shape, tla.make_coord(block, 0))
        ub = tla.make_tensor_like(ptr, tile, tla.arch.RowMajor)
        tla.copy(ub, tile)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        maximum = tla.tile_view(out, shape, tla.make_coord(1, 0))
        minimum = tla.tile_view(out, shape, tla.make_coord(2, 0))
        plain = tla.tile_view(out, shape, tla.make_coord(3 + block, 0))
        tla.copy(
            maximum,
            ub,
            tla.params.CopyUbToGmParams(atomic_mode=tla.params.AtomicMode.MAX),
        )
        tla.copy(
            minimum,
            ub,
            tla.params.CopyUbToGmParams(atomic_mode=tla.params.AtomicMode.MIN),
        )
        tla.copy(plain, ub)


@tla.jit
def store_acc(
    dst: tla.Tensor,
    acc: tla.Tensor,
    scales: tla.Tensor,
    quant: tla.Constexpr[tla.params.QuantMode],
    relu: tla.Constexpr[bool],
    mode: tla.Constexpr[tla.params.AtomicMode],
) -> None:
    if tla.const_expr(quant == tla.params.QuantMode.PER_CHANNEL):
        tla.copy(
            dst,
            acc,
            tla.params.CopyL0C2DstParams(
                atomic_mode=mode,
                quant_mode=quant,
                quant_tensor=scales,
                relu_enable=relu,
            ),
        )
    elif tla.const_expr(quant == tla.params.QuantMode.PER_TENSOR):
        tla.copy(
            dst,
            acc,
            tla.params.CopyL0C2DstParams(
                atomic_mode=mode,
                quant_mode=quant,
                quant_scale=2.0,
                relu_enable=relu,
            ),
        )
    else:
        tla.copy(
            dst, acc, tla.params.CopyL0C2DstParams(atomic_mode=mode, relu_enable=relu)
        )


@tla.kernel
def cube_writeback(
    lhs: tla.Tensor,
    rhs: tla.Tensor,
    out: tla.Tensor,
    gm_scales: tla.Tensor,
    integer_input: tla.Constexpr[bool],
    quant: tla.Constexpr[tla.params.QuantMode],
    relu: tla.Constexpr[bool],
) -> None:
    size = SIDE * SIDE
    dtype = lhs.ptr.dtype
    acc_dtype = tla.Int32 if integer_input else tla.Float32
    a1 = tla.allocate(size, dtype, tla.AddressSpace.l1, 512)
    b1 = tla.allocate(size, dtype, tla.AddressSpace.l1, 512)
    a0 = tla.allocate(size, dtype, tla.AddressSpace.l0a, 512)
    b0 = tla.allocate(size, dtype, tla.AddressSpace.l0b, 512)
    c0 = tla.allocate(size, acc_dtype, tla.AddressSpace.l0c, 1024)
    q1 = tla.allocate(SIDE, tla.UInt64, tla.AddressSpace.l1, 512)
    loaded = tla.flag("loaded", tla.pipes.MTE2, tla.pipes.MTE1)
    staged = tla.flag("staged", tla.pipes.MTE1, tla.pipes.CUBE)
    ready = tla.flag("ready", tla.pipes.CUBE, tla.pipes.FIX)
    with tla.cube():
        block = tla.arch.block_idx()
        shape = tla.make_shape(SIDE, SIDE)
        tile = tla.tile_view(lhs, shape, tla.make_coord(block, 0))
        right = tla.tile_view(rhs, shape, tla.make_coord(0, 0))
        maximum = tla.tile_view(out, shape, tla.make_coord(1, 0))
        minimum = tla.tile_view(out, shape, tla.make_coord(2, 0))
        plain = tla.tile_view(out, shape, tla.make_coord(3 + block, 0))
        l1_a = tla.make_tensor_like(a1, tile)
        l1_b = tla.make_tensor_like(b1, right)
        l0_a = tla.make_tensor_like(a0, l1_a)
        l0_b = tla.make_tensor_like(b0, l1_b)
        acc = tla.make_tensor_like(c0, maximum)
        scales = tla.make_tensor(
            q1,
            tla.make_layout(
                tla.make_shape(SIDE),
                tla.make_stride(1),
                layoutTag=tla.arch.VectorLayout,
            ),
        )
        if tla.const_expr(quant == tla.params.QuantMode.PER_CHANNEL):
            tla.copy(
                scales,
                tla.tile_view(gm_scales, tla.make_shape(SIDE), tla.make_coord(0)),
            )
        tla.copy(l1_a, tile)
        tla.copy(l1_b, right)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        tla.copy(l0_a, l1_a)
        tla.copy(l0_b, l1_b)
        tla.set_flag(staged)
        tla.wait_flag(staged)
        tla.mmad(acc, l0_a, l0_b, init_c=True, unit_flag=0)
        tla.set_flag(ready)
        tla.wait_flag(ready)
        store_acc(maximum, acc, scales, quant, relu, tla.params.AtomicMode.MAX)
        store_acc(minimum, acc, scales, quant, relu, tla.params.AtomicMode.MIN)
        store_acc(plain, acc, scales, quant, relu, tla.params.AtomicMode.NONE)


def make_repeated_row_tile(values, dtype):
    row = torch.tensor(values, dtype=dtype).repeat(SIDE)[:SIDE]
    return row.repeat(SIDE, 1)


def check_case(src_dtype, dst_dtype, quant=None, relu=False):
    # Each producer owns a plain output tile. Both reduce into tiles 1 and 2.
    src = torch.cat(
        (
            make_repeated_row_tile([-7, 4, 8, 4, -4, 6, 14], src_dtype),
            make_repeated_row_tile([-2, 9, -3, 7, -7, 14, 20], src_dtype),
        )
    )
    old = make_repeated_row_tile([-4, 7, 2, 12, -12, 10, 10], dst_dtype)
    initial = torch.full((6 * SIDE, SIDE), 10, dtype=dst_dtype)
    initial[SIDE : 2 * SIDE] = old
    initial[2 * SIDE : 3 * SIDE] = old
    expected_plain = src.float()
    if quant is None:
        kernel = vector_writeback
        extra = ()
    else:
        kernel = cube_writeback
        # A non-unit diagonal makes floating MMAD results require rounding on store.
        factor = 1 if src_dtype == torch.int8 else 1.0009765625
        rhs = torch.eye(SIDE).mul(factor).to(src_dtype)
        expected_plain = expected_plain * factor
        scales = (torch.arange(SIDE) % 3 + 1).float()
        packed = encode_per_channel_quant_tensor(scales, torch.int8)
        if quant == tla.params.QuantMode.PER_CHANNEL:
            expected_plain *= scales
        elif quant == tla.params.QuantMode.PER_TENSOR:
            expected_plain *= 2
        if relu:
            expected_plain = expected_plain.relu()
        extra = (src_dtype == torch.int8, quant, relu)

    out = initial.npu()
    tensors = [create_tla_tensor(src.npu(), "row")]
    if quant is not None:
        tensors.append(create_tla_tensor(rhs.npu(), "row"))
    tensors.append(create_tla_tensor(out, "row"))
    if quant is not None:
        tensors.append(
            create_tla_tensor(packed.npu(), "vector", element_type=tla.UInt64)
        )
    artifact = tla.compile(kernel, *tensors, *extra, options="--npu-arch 3510")
    artifact(*tensors, block_num=2)
    torch.npu.synchronize()

    result = out.cpu()
    plain = result[3 * SIDE : 5 * SIDE]
    torch.testing.assert_close(plain, expected_plain.to(dst_dtype), rtol=0, atol=0)
    a, b = plain.chunk(2)
    expected = initial.clone()
    expected[SIDE : 2 * SIDE] = old.maximum(a).maximum(b)
    expected[2 * SIDE : 3 * SIDE] = old.minimum(a).minimum(b)
    expected[3 * SIDE : 5 * SIDE] = plain
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    for dtype in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        check_case(dtype, dtype)
    quant = tla.params.QuantMode
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        check_case(torch.float16, dtype, quant.NO_QUANT)
    check_case(torch.int8, torch.int32, quant.NO_QUANT)
    check_case(torch.int8, torch.int8, quant.PER_TENSOR)
    check_case(torch.int8, torch.int8, quant.PER_CHANNEL, relu=True)
    print("Atomic MAX/MIN writeback passed on Vector and Cube.")


if __name__ == "__main__":
    main()
