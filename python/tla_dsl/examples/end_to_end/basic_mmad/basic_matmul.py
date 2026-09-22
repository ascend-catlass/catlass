# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Basic MMAD (flag sync): Kernel + Host in one file.

Dynamic GM; mnk/dtype/layout from CLI.
"""

from __future__ import annotations

import sys
from pathlib import Path

_DSL_EXAMPLE_PATH = str((Path(__file__).resolve().parent / "..").resolve())

if _DSL_EXAMPLE_PATH not in sys.path:
    sys.path.insert(0, _DSL_EXAMPLE_PATH)

import argparse

import catlass.tla as tla
import torch
import torch_npu  # noqa: F401

from common import TilingParams


@tla.kernel
def basic_mmad_kernel(
    gm_a: tla.Tensor,
    gm_b: tla.Tensor,
    gm_c: tla.Tensor,
    _tiling: tla.Constexpr[TilingParams],
    hf32_mode: tla.Constexpr[tla.params.HF32Mode],
    acc_is_int: tla.Constexpr[bool],
    relu_enable: tla.Constexpr[bool],
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
    l0c_ptr = tla.allocate(
        _tiling.l0_tm * _tiling.l0_tn, acc_dtype, tla.AddressSpace.l0c, 512
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

            k_block = gm_a_by_core.origin_shape[1]
            k_l1_count = (k_block + _tiling.l1_tk - 1) // _tiling.l1_tk
            k_l1_range = tla.range(c0, k_l1_count, c1)

            l0_c = tla.make_tensor_like(l0c_ptr, gm_c_by_core)

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

                    tla.set_flag(l0_ab_data_ready)
                    tla.wait_flag(l0_ab_data_ready)

                    unit_flag = (
                        0b11
                        if (k_l1 == k_l1_count - 1) and (k_l0 == k_l0_count - 1)
                        else 0b10
                    )
                    init_c = True if k_l1 == 0 and k_l0 == 0 else False
                    if tla.const_expr(
                        hf32_mode != tla.params.HF32Mode.HF32_DISABLE
                        and dtype_a == tla.Float32
                        and dtype_b == tla.Float32
                    ):
                        tla.mmad(
                            l0_c,
                            l0_a,
                            l0_b,
                            init_c=init_c,
                            unit_flag=unit_flag,
                            hf32_mode=hf32_mode,
                        )
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
                tla.params.CopyL0C2DstParams(unit_flag=0b11, relu_enable=relu_enable),
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


# torch cannot export fp8 over DLPack, so fp8 buffers go through
# create_tla_tensor's element_type override.
_FP8_TLA_TYPES = {
    "f8e4m3fn": tla.Float8E4M3FN,
    "f8e5m2": tla.Float8E5M2,
}


def run(args: argparse.Namespace) -> int:
    from common import (
        get_block_num,
        create_tla_tensor,
        compare,
        to_hf32,
    )

    torch.npu.set_device(args.device)
    print(
        f"--- mnk=({args.m},{args.n},{args.k}) "
        f"layout={args.layout_a}/{args.layout_b}/{args.layout_c} "
        f"dtype={args.dtype_a}/{args.dtype_b}/{args.dtype_c} ---"
    )
    torch.manual_seed(0)
    dtypes = {
        "f16": torch.float16,
        "bf16": torch.bfloat16,
        "f32": torch.float32,
        "i8": torch.int8,
        "i32": torch.int32,
        "f8e4m3fn": torch.float8_e4m3fn,
        "f8e5m2": torch.float8_e5m2,
    }
    dtype_a = dtypes[args.dtype_a]
    dtype_b = dtypes[args.dtype_b]
    dtype_c = dtypes[args.dtype_c]

    hf32_mode = tla.params.HF32Mode.HF32_NEAREST_EVEN

    # i8,i8 -> i32 is the integer MMAD route; the accumulator in L0C is i32.
    # It is exact and never takes the hf32 path, which is fp32-only.
    is_int_route = args.dtype_a == "i8"
    # hf32 is an fp32-only rounding mode; every other route leaves it off.
    enable_hf32 = False
    is_fp8_route = args.dtype_a in ("f8e4m3fn", "f8e5m2")
    if is_int_route:
        if args.dtype_b != "i8" or args.dtype_c != "i32":
            raise SystemExit(
                "the integer mmad route requires --dtype-a i8 --dtype-b i8 --dtype-c i32"
            )
        a = torch.randint(-8, 8, (args.m, args.k), dtype=dtype_a, device="cpu")
        b = torch.randint(-8, 8, (args.k, args.n), dtype=dtype_b, device="cpu")
        c = torch.zeros(args.m, args.n, dtype=dtype_c, device="cpu")
        # Small magnitudes keep the exact int32 product inside float64.
        ref = (a.double() @ b.double()).to(torch.int32)
    elif is_fp8_route:
        if args.dtype_b not in ("f8e4m3fn", "f8e5m2") or args.dtype_c != "f32":
            raise SystemExit(
                "the fp8 mmad route requires f8e4m3fn/f8e5m2 operands and --dtype-c f32"
            )
        # Round the operands through the fp8 format on the host, so the reference
        # multiplies exactly the values the device sees. Only the accumulation
        # order then differs, which is what the tolerance below covers.
        a = (torch.rand(args.m, args.k, device="cpu") * 4.0 - 2.0).to(dtype_a)
        b = (torch.rand(args.k, args.n, device="cpu") * 4.0 - 2.0).to(dtype_b)
        c = torch.zeros(args.m, args.n, dtype=dtype_c, device="cpu")
        ref = a.float() @ b.float()
    else:
        a = torch.rand(args.m, args.k, dtype=dtype_a, device="cpu") * 10.0 - 5.0
        b = torch.rand(args.k, args.n, dtype=dtype_b, device="cpu") * 10.0 - 5.0
        c = torch.rand(args.m, args.n, dtype=dtype_c, device="cpu") * 10.0 - 5.0

        enable_hf32 = (
            hf32_mode != tla.params.HF32Mode.HF32_DISABLE
            and dtype_a == torch.float32
            and dtype_b == torch.float32
        )
        if enable_hf32:
            ref = to_hf32(a, hf32_mode) @ to_hf32(b, hf32_mode)
        else:
            ref = a.float() @ b.float()
            if dtype_c in (torch.float16, torch.bfloat16):
                ref = ref.to(dtype_c).float()

    a = (
        a.contiguous() if args.layout_a == "row" else a.permute(1, 0).contiguous()
    ).npu()
    b = (
        b.contiguous() if args.layout_b == "row" else b.permute(1, 0).contiguous()
    ).npu()
    # A ColumnMajor (m, n) result is stored as an (n, m) buffer; the fixpipe
    # writes it transposed on its way out of L0C (nz2dn), so the kernel body is
    # unchanged -- the route follows the destination tensor's layout tag.
    if args.layout_c == "col":
        c = c.permute(1, 0)
    c = c.contiguous().npu()
    a_tensor = create_tla_tensor(a, args.layout_a, _FP8_TLA_TYPES.get(args.dtype_a))
    b_tensor = create_tla_tensor(b, args.layout_b, _FP8_TLA_TYPES.get(args.dtype_b))
    c_tensor = create_tla_tensor(c, args.layout_c, _FP8_TLA_TYPES.get(args.dtype_c))

    artifact = tla.compile(
        basic_mmad_kernel,
        a_tensor,
        b_tensor,
        c_tensor,
        TilingParams(),  # default tiling: L1: (256, 256, 128); L0: (256, 256, 32)
        hf32_mode,
        is_int_route,
        bool(args.relu_enable),
        options="--npu-arch 3510",
    )
    block_num = get_block_num(args.block_num, args.device, kind="cube")
    artifact(a_tensor, b_tensor, c_tensor, block_num=block_num)
    torch.npu.synchronize()

    result = c.detach().cpu()
    if args.layout_c == "col":
        result = result.transpose(0, 1).contiguous()
    if args.relu_enable:
        torch.nn.functional.relu_(ref)
    if is_int_route:
        # Integer MMAD is exact, so no tolerance: compare() falls back to
        # element-wise equality for integer dtypes.
        passed = compare(result, ref)
    elif enable_hf32:
        passed = compare(result, ref, enable_hf32=True)
    else:
        passed = compare(result, ref, args.k)
    print(f"passed={passed} cache_key={artifact.cache_key}")
    print(f"kernel.o={artifact.kernel_binary_path}")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--layout-a", choices=("row", "col"), default="row")
    parser.add_argument("--layout-b", choices=("row", "col"), default="row")
    # `col` sends the L0C -> GM store through the transposing (nz2dn) fixpipe.
    parser.add_argument("--layout-c", choices=("row", "col"), default="row")
    dtypes_ab = ("f16", "bf16", "f32", "i8", "f8e4m3fn", "f8e5m2")
    parser.add_argument("--dtype-a", choices=dtypes_ab, default="f16")
    parser.add_argument("--dtype-b", choices=dtypes_ab, default="f16")
    parser.add_argument(
        "--dtype-c", choices=("f16", "bf16", "f32", "i32"), default="f32"
    )
    parser.add_argument("--relu-enable", type=int, choices=(0, 1), default=0)
    parser.add_argument("--block-num", type=int, default=-1)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
