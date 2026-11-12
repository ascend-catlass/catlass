# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Basic MMAD with bias (flag sync): Kernel + Host in one file.

Dynamic GM; mnk/dtype from CLI, while A/B/C stay RowMajor. ``gm_bias`` is a
required operand: the bias (1 x N) travels GM -> L1 -> biasbuf and is consumed
by the first MMAD of every block through ``tla.mmad(..., bias=...)``.

Operand dtypes follow ``basic_matmul.py``: f16, bf16, f32, i8 and the two fp8
formats, chosen independently per side. The output is i32 for the integer route
(i8 x i8) and f32 for every other one, so the accumulator is always either i32
or f32 and the bias dtype follows it -- fp32 alongside fp32 accumulation, i32
alongside the integer MMAD route.
"""

from __future__ import annotations

import sys
from pathlib import Path

_DSL_EXAMPLE_PATH = str(
    next(p for p in Path(__file__).resolve().parents if (p / "helpers").is_dir())
)

if _DSL_EXAMPLE_PATH not in sys.path:
    sys.path.append(_DSL_EXAMPLE_PATH)

import argparse

import catlass.tla as tla
import torch
import torch_npu  # noqa: F401

from helpers import TilingParams


@tla.kernel
def basic_mmad_bias_kernel(
    gm_a: tla.Tensor,
    gm_b: tla.Tensor,
    gm_bias: tla.Tensor,
    gm_c: tla.Tensor,
    _tiling: tla.Constexpr[TilingParams],
    acc_is_int: tla.Constexpr[bool],
) -> None:
    c0 = 0
    c1 = 1

    dtype_a = gm_a.ptr.dtype
    dtype_b = gm_b.ptr.dtype

    m = gm_a.origin_shape[0]
    n = gm_b.origin_shape[1]
    k = gm_a.origin_shape[1]

    l1a0_data_ready = tla.flag("l1a0_data_ready", tla.arch.MTE2, tla.arch.MTE1)
    l1a1_data_ready = tla.flag("l1a1_data_ready", tla.arch.MTE2, tla.arch.MTE1)
    l1b0_data_ready = tla.flag("l1b0_data_ready", tla.arch.MTE2, tla.arch.MTE1)
    l1b1_data_ready = tla.flag("l1b1_data_ready", tla.arch.MTE2, tla.arch.MTE1)
    l1a0_available = tla.flag("l1a0_available", tla.arch.MTE1, tla.arch.MTE2)
    l1a1_available = tla.flag("l1a1_available", tla.arch.MTE1, tla.arch.MTE2)
    l1b0_available = tla.flag("l1b0_available", tla.arch.MTE1, tla.arch.MTE2)
    l1b1_available = tla.flag("l1b1_available", tla.arch.MTE1, tla.arch.MTE2)
    l0a0_available = tla.flag("l0a0_available", tla.arch.CUBE, tla.arch.MTE1)
    l0a1_available = tla.flag("l0a1_available", tla.arch.CUBE, tla.arch.MTE1)
    l0b0_available = tla.flag("l0b0_available", tla.arch.CUBE, tla.arch.MTE1)
    l0b1_available = tla.flag("l0b1_available", tla.arch.CUBE, tla.arch.MTE1)
    l0_ab_data_ready = tla.flag("l0_ab_data_ready", tla.arch.MTE1, tla.arch.CUBE)
    l0c_available = tla.flag("l0c_available", tla.arch.FIX, tla.arch.CUBE)
    # Bias pipeline: GM -> L1 (MTE2), L1 -> biasbuf (MTE1), biasbuf consumed by CUBE.
    l1_bias_data_ready = tla.flag("l1_bias_data_ready", tla.arch.MTE2, tla.arch.MTE1)
    l1_bias_available = tla.flag("l1_bias_available", tla.arch.MTE1, tla.arch.MTE2)
    l0_bias_available = tla.flag("l0_bias_available", tla.arch.CUBE, tla.arch.MTE1)

    l1a0_ptr = tla.allocate(
        _tiling.l1_tm * _tiling.l1_tk, dtype_a, tla.AddressSpace.l1, 512
    )
    l1a1_ptr = tla.allocate(
        _tiling.l1_tm * _tiling.l1_tk, dtype_a, tla.AddressSpace.l1, 512
    )
    l1b0_ptr = tla.allocate(
        _tiling.l1_tk * _tiling.l1_tn, dtype_b, tla.AddressSpace.l1, 512
    )
    l1b1_ptr = tla.allocate(
        _tiling.l1_tk * _tiling.l1_tn, dtype_b, tla.AddressSpace.l1, 512
    )

    l0a0_ptr = tla.allocate(
        _tiling.l0_tm * _tiling.l0_tk, dtype_a, tla.AddressSpace.l0a, 512
    )
    l0a1_ptr = tla.allocate(
        _tiling.l0_tm * _tiling.l0_tk, dtype_a, tla.AddressSpace.l0a, 512
    )
    l0b0_ptr = tla.allocate(
        _tiling.l0_tk * _tiling.l0_tn, dtype_b, tla.AddressSpace.l0b, 512
    )
    l0b1_ptr = tla.allocate(
        _tiling.l0_tk * _tiling.l0_tn, dtype_b, tla.AddressSpace.l0b, 512
    )

    # Integer route: an i8 x i8 MMAD accumulates into an i32 L0C, not fp32.
    acc_dtype = tla.Int32 if acc_is_int else tla.Float32
    # The bias buffer mirrors the accumulator: f32 alongside fp32 accumulation,
    # i32 alongside the integer MMAD route.
    bias_dtype = tla.Int32 if acc_is_int else tla.Float32
    l1_bias_ptr = tla.allocate(_tiling.l1_tn, bias_dtype, tla.AddressSpace.l1, 512)
    l0_bias_ptr = tla.allocate(_tiling.l0_tn, bias_dtype, tla.AddressSpace.biasbuf, 512)
    l0c_ptr = tla.allocate(
        _tiling.l0_tm * _tiling.l0_tn, acc_dtype, tla.AddressSpace.l0c, 1024
    )

    grid_m = (m + _tiling.l1_tm - 1) // _tiling.l1_tm
    grid_n = (n + _tiling.l1_tn - 1) // _tiling.l1_tn
    total_blocks = grid_m * grid_n

    with tla.cube():
        tla.set_flag(l1a0_available)
        tla.set_flag(l1a1_available)
        tla.set_flag(l1b0_available)
        tla.set_flag(l1b1_available)
        tla.set_flag(l0a0_available)
        tla.set_flag(l0a1_available)
        tla.set_flag(l0b0_available)
        tla.set_flag(l0b1_available)
        tla.set_flag(l0c_available)
        tla.set_flag(l1_bias_available)
        tla.set_flag(l0_bias_available)

        runtime_zero = tla.as_numeric(0)
        l1_buf_idx = runtime_zero
        l0_buf_idx = runtime_zero

        block_range = tla.range(
            tla.arch.block_idx(), total_blocks, tla.arch.block_num()
        )
        for block_linear in block_range:
            block_row = block_linear // grid_n
            block_col = block_linear % grid_n
            gm_a_by_core = tla.tile_view(
                gm_a, tla.make_shape(_tiling.l1_tm, k), tla.make_coord(block_row, c0)
            )
            gm_b_by_core = tla.tile_view(
                gm_b, tla.make_shape(k, _tiling.l1_tn), tla.make_coord(c0, block_col)
            )
            gm_c_by_core = tla.tile_view(
                gm_c,
                tla.make_shape(_tiling.l1_tm, _tiling.l1_tn),
                tla.make_coord(block_row, block_col),
            )
            # Bias is 1 x N; take the N-column slice owned by this MN block.
            gm_bias_by_core = tla.tile_view(
                gm_bias, tla.make_shape(_tiling.l1_tn), tla.make_coord(block_col)
            )

            k_block = gm_a_by_core.origin_shape[1]
            k_l1_count = (k_block + _tiling.l1_tk - 1) // _tiling.l1_tk
            k_l1_range = tla.range(c0, k_l1_count, c1)

            l0_c = tla.make_tensor_like(l0c_ptr, gm_c_by_core)
            # The bias tile is one continuous 1-D run, so it is tagged VectorLayout --
            # the tag that matches `tla::detail::isVector` on the device side.
            l1_bias = tla.make_tensor_like(
                l1_bias_ptr, gm_bias_by_core, tla.arch.VectorLayout
            )
            l0_bias = tla.make_tensor_like(l0_bias_ptr, l1_bias, tla.arch.VectorLayout)

            # Bias is block-resident: reload the slice for this block's columns
            # before the K loop. The single L1 slot is reused across blocks, so
            # wait until the previous block's L1 -> biasbuf moved off it.
            tla.wait_flag(l1_bias_available)
            tla.copy(l1_bias, gm_bias_by_core)
            tla.set_flag(l1_bias_data_ready)

            for k_l1 in k_l1_range:
                gm_a_by_l1 = tla.tile_view(
                    gm_a_by_core,
                    tla.make_shape(_tiling.l1_tm, _tiling.l1_tk),
                    tla.make_coord(c0, k_l1),
                )
                gm_b_by_l1 = tla.tile_view(
                    gm_b_by_core,
                    tla.make_shape(_tiling.l1_tk, _tiling.l1_tn),
                    tla.make_coord(k_l1, c0),
                )

                l1_a = tla.make_tensor_like(
                    l1a0_ptr if (l1_buf_idx == c0) else l1a1_ptr, gm_a_by_l1
                )
                l1_b = tla.make_tensor_like(
                    l1b0_ptr if (l1_buf_idx == c0) else l1b1_ptr, gm_b_by_l1
                )
                if l1_buf_idx == c0:
                    tla.wait_flag(l1a0_available)
                else:
                    tla.wait_flag(l1a1_available)
                tla.copy(l1_a, gm_a_by_l1)
                if l1_buf_idx == c0:
                    tla.set_flag(l1a0_data_ready)
                else:
                    tla.set_flag(l1a1_data_ready)

                if l1_buf_idx == c0:
                    tla.wait_flag(l1b0_available)
                else:
                    tla.wait_flag(l1b1_available)
                tla.copy(l1_b, gm_b_by_l1)
                if l1_buf_idx == c0:
                    tla.set_flag(l1b0_data_ready)
                else:
                    tla.set_flag(l1b1_data_ready)

                k_l0_count = (l1_a.origin_shape[1] + _tiling.l0_tk - 1) // _tiling.l0_tk
                k_l0_range = tla.range(c0, k_l0_count, c1)

                for k_l0 in k_l0_range:
                    l1_a_by_l0 = tla.tile_view(
                        l1_a,
                        tla.make_shape(_tiling.l0_tm, _tiling.l0_tk),
                        tla.make_coord(c0, k_l0),
                    )
                    l1_b_by_l0 = tla.tile_view(
                        l1_b,
                        tla.make_shape(_tiling.l0_tk, _tiling.l0_tn),
                        tla.make_coord(k_l0, c0),
                    )

                    l0_a = tla.make_tensor_like(
                        l0a0_ptr if (l0_buf_idx == c0) else l0a1_ptr, l1_a_by_l0
                    )
                    l0_b = tla.make_tensor_like(
                        l0b0_ptr if (l0_buf_idx == c0) else l0b1_ptr, l1_b_by_l0
                    )
                    if k_l0 == 0:
                        if l1_buf_idx == c0:
                            tla.wait_flag(l1a0_data_ready)
                        else:
                            tla.wait_flag(l1a1_data_ready)

                    if l0_buf_idx == c0:
                        tla.wait_flag(l0a0_available)
                    else:
                        tla.wait_flag(l0a1_available)
                    tla.copy(l0_a, l1_a_by_l0)
                    if k_l0 == k_l0_count - 1:
                        if l1_buf_idx == c0:
                            tla.set_flag(l1a0_available)
                        else:
                            tla.set_flag(l1a1_available)

                    if k_l0 == 0:
                        if l1_buf_idx == c0:
                            tla.wait_flag(l1b0_data_ready)
                        else:
                            tla.wait_flag(l1b1_data_ready)
                    if l0_buf_idx == c0:
                        tla.wait_flag(l0b0_available)
                    else:
                        tla.wait_flag(l0b1_available)
                    tla.copy(l0_b, l1_b_by_l0)
                    if k_l0 == k_l0_count - 1:
                        if l1_buf_idx == c0:
                            tla.set_flag(l1b0_available)
                        else:
                            tla.set_flag(l1b1_available)

                    init_c = True if k_l1 == 0 and k_l0 == 0 else False

                    # L1 -> biasbuf once per block, right before the first MMAD
                    # that consumes it. The MTE1_MTE2 release lets the next
                    # block reload L1 bias as soon as this copy is done; the
                    # M_MTE1 release comes after the MMAD below.
                    if init_c:
                        tla.wait_flag(l1_bias_data_ready)
                        tla.wait_flag(l0_bias_available)
                        tla.copy(l0_bias, l1_bias)
                        tla.set_flag(l1_bias_available)

                    tla.set_flag(l0_ab_data_ready)
                    tla.wait_flag(l0_ab_data_ready)

                    unit_flag = (
                        0b11
                        if (k_l1 == k_l1_count - 1) and (k_l0 == k_l0_count - 1)
                        else 0b10
                    )
                    if init_c:
                        tla.mmad(
                            l0_c,
                            l0_a,
                            l0_b,
                            init_c=init_c,
                            unit_flag=unit_flag,
                            bias=l0_bias,
                        )
                        tla.set_flag(l0_bias_available)
                    else:
                        tla.mmad(l0_c, l0_a, l0_b, init_c=init_c, unit_flag=unit_flag)
                    if l0_buf_idx == c0:
                        tla.set_flag(l0a0_available)
                        tla.set_flag(l0b0_available)
                    else:
                        tla.set_flag(l0a1_available)
                        tla.set_flag(l0b1_available)
                    l0_buf_idx = c1 - l0_buf_idx
                l1_buf_idx = c1 - l1_buf_idx

            tla.copy(
                gm_c_by_core,
                l0_c,
                tla.params.CopyL0C2DstParams(unit_flag=0b11),
            )

        tla.wait_flag(l1a0_available)
        tla.wait_flag(l1a1_available)
        tla.wait_flag(l1b0_available)
        tla.wait_flag(l1b1_available)
        tla.wait_flag(l0a0_available)
        tla.wait_flag(l0a1_available)
        tla.wait_flag(l0b0_available)
        tla.wait_flag(l0b1_available)
        tla.wait_flag(l0c_available)
        tla.wait_flag(l1_bias_available)
        tla.wait_flag(l0_bias_available)


# torch cannot export fp8 over DLPack, so fp8 buffers go through
# create_tla_tensor's element_type override.
_FP8_TLA_TYPES = {
    "f8e4m3fn": tla.Float8E4M3FN,
    "f8e5m2": tla.Float8E5M2,
}

# Operand dtypes, the same set basic_matmul.py accepts.
_TORCH_DTYPES = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
    "i8": torch.int8,
    "f8e4m3fn": torch.float8_e4m3fn,
    "f8e5m2": torch.float8_e5m2,
}
_DTYPE_CHOICES = ("f16", "bf16", "f32", "i8", "f8e4m3fn", "f8e5m2")


def run(args: argparse.Namespace) -> int:
    from helpers import (
        get_block_num,
        create_tla_tensor,
        compare,
    )

    torch.npu.set_device(args.device)
    print(
        f"--- mnk=({args.m},{args.n},{args.k}) dtype={args.dtype_a}/{args.dtype_b} ---"
    )
    torch.manual_seed(0)
    dtype_a = _TORCH_DTYPES[args.dtype_a]
    dtype_b = _TORCH_DTYPES[args.dtype_b]

    is_int_route = args.dtype_a == "i8"
    is_fp8_route = args.dtype_a in _FP8_TLA_TYPES
    # i8 x i8 is the integer MMAD route and lands in i32; every other route
    # keeps the fp32 accumulator and writes fp32.
    out_dtype = torch.int32 if is_int_route else torch.float32
    if is_int_route:
        if args.dtype_b != "i8":
            raise SystemExit(
                "the integer mmad route requires --dtype-a i8 --dtype-b i8"
            )
        a = torch.randint(-8, 8, (args.m, args.k), dtype=dtype_a, device="cpu")
        b = torch.randint(-8, 8, (args.k, args.n), dtype=dtype_b, device="cpu")
        # The bias follows the accumulator: i32 alongside the integer route.
        bias = torch.randint(-8, 8, (args.n,), dtype=torch.int32, device="cpu")
        # Small magnitudes keep the exact int32 product inside float64.
        ref = a.to(torch.int32) @ b.to(torch.int32) + bias
    elif is_fp8_route:
        if args.dtype_b not in _FP8_TLA_TYPES:
            raise SystemExit(
                "the fp8 mmad route requires f8e4m3fn/f8e5m2 on both sides"
            )
        # Round the operands through the fp8 format on the host, so the
        # reference multiplies exactly the values the device sees. Only the
        # accumulation order then differs, which the tolerance covers.
        a = (torch.rand(args.m, args.k, device="cpu") * 4.0 - 2.0).to(dtype_a)
        b = (torch.rand(args.k, args.n, device="cpu") * 4.0 - 2.0).to(dtype_b)
        bias = torch.rand(args.n, dtype=torch.float32, device="cpu") * 10.0 - 5.0
        ref = a.float() @ b.float() + bias
    else:
        a = torch.rand(args.m, args.k, dtype=dtype_a, device="cpu") * 10.0 - 5.0
        b = torch.rand(args.k, args.n, dtype=dtype_b, device="cpu") * 10.0 - 5.0
        bias = torch.rand(args.n, dtype=torch.float32, device="cpu") * 10.0 - 5.0
        ref = a.float() @ b.float() + bias

    # The accumulator starts non-zero on purpose: the first MMAD of each block
    # takes init_c=True, which is what folds the bias in and discards this
    # value, so a wrong init shows up as a wrong result.
    if is_int_route:
        c = torch.randint(-8, 8, (args.m, args.n), dtype=out_dtype, device="cpu")
    else:
        c = torch.rand(args.m, args.n, dtype=out_dtype, device="cpu")

    # A/B/C are always RowMajor in this example.
    a = a.contiguous().npu()
    b = b.contiguous().npu()
    c = c.contiguous().npu()
    # Bias is a 1 x N row vector; it stays Vector whatever A/B/C do.
    bias = bias.contiguous().npu()
    # fp8 cannot cross DLPack as itself, so those buffers go over as bytes with
    # create_tla_tensor's element_type telling the kernel what they hold.
    a_tensor = create_tla_tensor(a, "row", _FP8_TLA_TYPES.get(args.dtype_a))
    b_tensor = create_tla_tensor(b, "row", _FP8_TLA_TYPES.get(args.dtype_b))
    bias_tensor = create_tla_tensor(bias, "vector")
    c_tensor = create_tla_tensor(c, "row")

    artifact = tla.compile(
        basic_mmad_bias_kernel,
        a_tensor,
        b_tensor,
        bias_tensor,
        c_tensor,
        TilingParams(l1_tm=128, l0_tm=128),
        is_int_route,
        options="--npu-arch 3510",
    )
    block_num = get_block_num(args.block_num, args.device, kind="cube")
    artifact(a_tensor, b_tensor, bias_tensor, c_tensor, block_num=block_num)
    torch.npu.synchronize()

    result = c.detach().cpu()
    if is_int_route:
        # Integer MMAD is exact, so no tolerance: compare() falls back to
        # element-wise equality for integer dtypes.
        passed = compare(result, ref)
    else:
        passed = compare(result, ref, args.k)
    print(f"passed={passed} cache_key={artifact.cache_key}")
    print(f"kernel.o={artifact.kernel_binary_path}")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Basic MMAD with bias, dynamic GM.")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--k", type=int, default=1024)
    # A/B/C are fixed to RowMajor in this example, so there is no layout knob.
    # The two sides pick their dtype independently, as in basic_matmul.py. The
    # output follows from them: i8 x i8 writes i32, everything else writes f32.
    parser.add_argument("--dtype-a", choices=_DTYPE_CHOICES, default="f16")
    parser.add_argument("--dtype-b", choices=_DTYPE_CHOICES, default="f16")
    parser.add_argument("--block-num", type=int, default=-1)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
