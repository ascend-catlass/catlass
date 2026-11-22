# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""GEMV (m == 1): A(row of K) x B(K x N) -> C(1 x N), one block per N tile.

Kernel + Host in one file.

The mmad mode is not a separate op: ``tla.mmad`` reads the lhs layout. A zN lhs
is the matrix form; a rank-1 ``VectorLayout`` lhs carrying the K elements of a
single row is the gemv form, with rhs still nZ and the accumulator still
``L0Clayout`` written out as a row. That makes the staging path the interesting
part: A travels GM->L1->L0A as a flat ``VectorLayout`` run, and B keeps the
usual GM->L1->L0B zN->nZ route.

``--dtype-ab`` picks the operand type, shared by A and B; the output type,
which is also the L0C accumulator type, follows from it: the mmad routes take
f16 / f32 / f8e5m2 operands into an f32 accumulator and i8 operands into an
i32 one. The host hands an f8e5m2 buffer over as a same-width integer view
plus an explicit element type, because torch cannot export fp8 over DLPack.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
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


_DTYPES = {
    "f32": torch.float32,
    "f16": torch.float16,
    "i8": torch.int8,
    "i32": torch.int32,
    "f8e5m2": torch.float8_e5m2,
}

# torch cannot export fp8 over DLPack at all, so an fp8 buffer is handed to
# ``create_tla_tensor`` as a same-width integer view with the real type here.
_FP8_TLA_TYPES = {"f8e5m2": tla.Float8E5M2}


@dataclass(frozen=True)
class GemvTilingParams:
    l1_tk: tla.Constexpr[int] = 128
    l1_tn: tla.Constexpr[int] = 128
    l0_tk: tla.Constexpr[int] = 64


@tla.kernel
def basic_matmul_gemv_kernel(
    gm_a: tla.Tensor,
    gm_b: tla.Tensor,
    gm_c: tla.Tensor,
    _tiling: tla.Constexpr[GemvTilingParams],
) -> None:
    c0 = 0
    c1 = 1

    dtype_a = gm_a.ptr.dtype
    dtype_b = gm_b.ptr.dtype
    # L0C accumulates in the element type C is written out as: f32 on the
    # floating-point routes, i32 on the integer one.
    dtype_c = gm_c.ptr.dtype

    # m == 1 is implied by the gemv form: the lhs is one flat K row.
    k = gm_a.origin_shape[0]
    n = gm_b.origin_shape[1]
    l0_tn = _tiling.l1_tn

    l1a_data_ready = tla.flag("l1a_data_ready", tla.arch.MTE2, tla.arch.MTE1)
    l1b_data_ready = tla.flag("l1b_data_ready", tla.arch.MTE2, tla.arch.MTE1)
    l1a_available = tla.flag("l1a_available", tla.arch.MTE1, tla.arch.MTE2)
    l1b_available = tla.flag("l1b_available", tla.arch.MTE1, tla.arch.MTE2)
    l0a_available = tla.flag("l0a_available", tla.arch.CUBE, tla.arch.MTE1)
    l0b_available = tla.flag("l0b_available", tla.arch.CUBE, tla.arch.MTE1)
    l0_ab_data_ready = tla.flag("l0_ab_data_ready", tla.arch.MTE1, tla.arch.CUBE)

    l1a_ptr = tla.allocate(_tiling.l1_tk, dtype_a, tla.AddressSpace.l1, 512)
    l1b_ptr = tla.allocate(_tiling.l1_tk * l0_tn, dtype_b, tla.AddressSpace.l1, 512)
    l0a_ptr = tla.allocate(_tiling.l0_tk, dtype_a, tla.AddressSpace.l0a, 512)
    l0b_ptr = tla.allocate(_tiling.l0_tk * l0_tn, dtype_b, tla.AddressSpace.l0b, 512)
    # The M == 1 accumulator still occupies whole 16-row fractals: one 16x16
    # fractal per 16 columns of the row.
    l0c_ptr = tla.allocate(
        16 * _tiling.l1_tn,
        dtype_c,
        tla.AddressSpace.l0c,
        1024,
    )

    grid_n = (n + _tiling.l1_tn - 1) // _tiling.l1_tn

    with tla.cube():
        tla.set_flag(l1a_available)
        tla.set_flag(l1b_available)
        tla.set_flag(l0a_available)
        tla.set_flag(l0b_available)

        # One block owns one N tile: every block walks the full K of its columns.
        block_range = tla.range(tla.arch.block_idx(), grid_n, tla.arch.block_num())
        for block_col in block_range:
            gm_b_by_core = tla.tile_view(
                gm_b,
                tla.make_shape(k, _tiling.l1_tn),
                tla.make_coord(c0, block_col),
            )
            gm_c_by_core = tla.tile_view(
                gm_c,
                tla.make_shape(1, _tiling.l1_tn),
                tla.make_coord(c0, block_col),
            )

            l0_c = tla.make_tensor_like(l0c_ptr, gm_c_by_core)

            k_l1_count = (
                gm_b_by_core.origin_shape[0] + _tiling.l1_tk - 1
            ) // _tiling.l1_tk
            k_l1_range = tla.range(c0, k_l1_count, c1)

            for k_l1 in k_l1_range:
                # A: a flat K run on both sides, so the tile is 1-D.
                gm_a_by_l1 = tla.tile_view(
                    gm_a,
                    tla.make_shape(_tiling.l1_tk),
                    tla.make_coord(k_l1),
                )
                gm_b_by_l1 = tla.tile_view(
                    gm_b_by_core,
                    tla.make_shape(_tiling.l1_tk, _tiling.l1_tn),
                    tla.make_coord(k_l1, c0),
                )

                # No layout tag: L1 keeps the VectorLayout of the flat GM run.
                l1_a = tla.make_tensor_like(l1a_ptr, gm_a_by_l1)
                l1_b = tla.make_tensor_like(l1b_ptr, gm_b_by_l1)

                tla.wait_flag(l1a_available)
                tla.copy(l1_a, gm_a_by_l1)
                tla.set_flag(l1a_data_ready)

                tla.wait_flag(l1b_available)
                tla.copy(l1_b, gm_b_by_l1)
                tla.set_flag(l1b_data_ready)

                # Read the K extent off the GM-rooted tile: a derived L1
                # tensor's origin_shape is dynamic and cannot be used in
                # arithmetic.
                k_l0_count = (
                    gm_a_by_l1.origin_shape[0] + _tiling.l0_tk - 1
                ) // _tiling.l0_tk
                k_l0_range = tla.range(c0, k_l0_count, c1)

                for k_l0 in k_l0_range:
                    l1_a_by_l0 = tla.tile_view(
                        l1_a,
                        tla.make_shape(_tiling.l0_tk),
                        tla.make_coord(k_l0),
                    )
                    l1_b_by_l0 = tla.tile_view(
                        l1_b,
                        tla.make_shape(_tiling.l0_tk, l0_tn),
                        tla.make_coord(k_l0, c0),
                    )

                    # L0A keeps the flat VectorLayout of its L1 source, which is
                    # what selects the gemv form; L0B is always nZ here.
                    l0_a = tla.make_tensor_like(l0a_ptr, l1_a_by_l0)
                    l0_b = tla.make_tensor_like(l0b_ptr, l1_b_by_l0)

                    if k_l0 == 0:
                        tla.wait_flag(l1a_data_ready)
                        tla.wait_flag(l1b_data_ready)

                    tla.wait_flag(l0a_available)
                    tla.copy(l0_a, l1_a_by_l0)
                    tla.wait_flag(l0b_available)
                    tla.copy(l0_b, l1_b_by_l0)

                    tla.set_flag(l0_ab_data_ready)
                    tla.wait_flag(l0_ab_data_ready)

                    unit_flag = (
                        0b11
                        if (k_l1 == k_l1_count - 1) and (k_l0 == k_l0_count - 1)
                        else 0b10
                    )
                    init_c = k_l1 == 0 and k_l0 == 0
                    tla.mmad(
                        l0_c,
                        l0_a,
                        l0_b,
                        init_c=init_c,
                        unit_flag=unit_flag,
                        compute_order=tla.params.ComputeOrder.M_FIRST,
                    )

                    tla.set_flag(l0a_available)
                    tla.set_flag(l0b_available)

                # Every MTE1 read of this L1 slice is done; the next K tile
                # may overwrite it.
                tla.set_flag(l1a_available)
                tla.set_flag(l1b_available)

            tla.copy(
                gm_c_by_core,
                l0_c,
                tla.params.CopyL0C2DstParams(unit_flag=0b11, relu_enable=False),
            )

        tla.wait_flag(l1a_available)
        tla.wait_flag(l1b_available)
        tla.wait_flag(l0a_available)
        tla.wait_flag(l0b_available)


def run(args: argparse.Namespace) -> int:
    from helpers import compare, create_tla_tensor, get_block_num

    tiling = GemvTilingParams()
    if args.k <= 0 or args.n <= 0:
        raise SystemExit(f"n={args.n} and k={args.k} must be positive")

    # i8 x i8 accumulates into i32; every other operand pair accumulates into
    # f32, which is also the only type fp8 can be written out as.
    dtype_a = dtype_b = _DTYPES[args.dtype_ab]
    dtype_c_name = "i32" if args.dtype_ab == "i8" else "f32"
    dtype_c = _DTYPES[dtype_c_name]

    torch.npu.set_device(args.device)
    print(
        f"--- gemv m=1 n={args.n} k={args.k} "
        f"dtype={args.dtype_ab}/{args.dtype_ab}/{dtype_c_name} ---"
    )
    torch.manual_seed(0)

    if args.dtype_ab == "i8":
        # Small magnitudes keep the exact int32 product inside float64.
        a = torch.randint(-8, 8, (args.k,), dtype=dtype_a, device="cpu")
        b = torch.randint(-8, 8, (args.k, args.n), dtype=dtype_b, device="cpu")
        c = torch.zeros(1, args.n, dtype=dtype_c, device="cpu")
        ref = (a.double() @ b.double()).to(torch.int32).reshape(1, args.n)
    elif args.dtype_ab == "f8e5m2":
        # Round the operands through e5m2 on the host, so the reference
        # multiplies exactly the values the device sees; only the accumulation
        # order then differs, which is what the tolerance below covers.
        a = (torch.rand(args.k, device="cpu") * 4.0 - 2.0).to(dtype_a)
        b = (torch.rand(args.k, args.n, device="cpu") * 4.0 - 2.0).to(dtype_b)
        c = torch.zeros(1, args.n, dtype=dtype_c, device="cpu")
        ref = (a.float() @ b.float()).reshape(1, args.n)
    else:
        a = torch.rand(args.k, dtype=dtype_a, device="cpu") * 10.0 - 5.0
        b = torch.rand(args.k, args.n, dtype=dtype_b, device="cpu") * 10.0 - 5.0
        c = torch.zeros(1, args.n, dtype=dtype_c, device="cpu")
        # The device multiplies the low-precision operands and accumulates in
        # fp32, so the reference widens first and never rounds the product to
        # the operand type.
        ref = (a.float() @ b.float()).reshape(1, args.n)

    # A is bound as the flat 1-D row the gemv lhs has to be; B and C keep the
    # row-major layout the matrix form uses. The output buffer is the one that
    # has to be rebound in place: the kernel writes through the npu storage, so
    # the cpu tensor handed to ``compare`` must alias it.
    element_ab = _FP8_TLA_TYPES.get(args.dtype_ab)
    a_tensor = create_tla_tensor(a.contiguous().npu(), "vector", element_ab)
    b_tensor = create_tla_tensor(b.contiguous().npu(), "row", element_ab)
    c = c.contiguous().npu()
    c_tensor = create_tla_tensor(c, "row")

    artifact = tla.compile(
        basic_matmul_gemv_kernel,
        a_tensor,
        b_tensor,
        c_tensor,
        tiling,
        options="--npu-arch 3510",
    )
    total_blocks = (args.n + tiling.l1_tn - 1) // tiling.l1_tn
    block_num = min(
        get_block_num(args.block_num, args.device, kind="cube"), total_blocks
    )
    artifact(a_tensor, b_tensor, c_tensor, block_num=block_num)
    torch.npu.synchronize()

    result = c.detach().cpu()
    if args.dtype_ab == "i8":
        # Integer MMAD is exact, so no tolerance: compare() falls back to
        # element-wise equality for integer dtypes.
        passed = compare(result, ref)
    else:
        passed = compare(result, ref, args.k)
    print(f"passed={passed} cache_key={artifact.cache_key}")
    print(f"kernel.o={artifact.kernel_binary_path}")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument(
        "--dtype-ab",
        choices=("f32", "f16", "i8", "f8e5m2"),
        default="f32",
        help="operand type for A and B; C follows as f32 (i8 accumulates to i32)",
    )
    parser.add_argument("--block-num", type=int, default=-1)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
