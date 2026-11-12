# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Basic MMAD (explicit mutex): Kernel + Host in one file.

Dynamic GM; mnk/dtype/layout from CLI.
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
def basic_mmad_mutex_kernel(
    gm_a: tla.Tensor,
    gm_b: tla.Tensor,
    gm_c: tla.Tensor,
    _tiling: tla.Constexpr[TilingParams],
) -> None:
    c0 = 0
    c1 = 1

    dtype_a = gm_a.ptr.dtype
    dtype_b = gm_b.ptr.dtype

    m = gm_a.origin_shape[0]
    n = gm_b.origin_shape[1]
    k = gm_a.origin_shape[1]

    mutex_l1a0 = tla.mutex(resource="l1a0", id=0)
    mutex_l1a1 = tla.mutex(resource="l1a1", id=1)
    mutex_l1b0 = tla.mutex(resource="l1b0", id=2)
    mutex_l1b1 = tla.mutex(resource="l1b1", id=3)

    mutex_l0a0 = tla.mutex(resource="l0a0", id=4)
    mutex_l0a1 = tla.mutex(resource="l0a1", id=5)
    mutex_l0b0 = tla.mutex(resource="l0b0", id=6)
    mutex_l0b1 = tla.mutex(resource="l0b1", id=7)

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

    l0c_ptr = tla.allocate(
        _tiling.l0_tm * _tiling.l0_tn, tla.Float32, tla.AddressSpace.l0c, 1024
    )

    grid_m = (m + _tiling.l1_tm - 1) // _tiling.l1_tm
    grid_n = (n + _tiling.l1_tn - 1) // _tiling.l1_tn
    total_blocks = grid_m * grid_n

    with tla.cube():
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

                mutex_l1a = mutex_l1a0 if (l1_buf_idx == c0) else mutex_l1a1
                mutex_l1a.lock(pipe=tla.arch.MTE2)
                tla.copy(l1_a, gm_a_by_l1)
                mutex_l1a.unlock(pipe=tla.arch.MTE2)

                mutex_l1b = mutex_l1b0 if (l1_buf_idx == c0) else mutex_l1b1
                mutex_l1b.lock(pipe=tla.arch.MTE2)
                tla.copy(l1_b, gm_b_by_l1)
                mutex_l1b.unlock(pipe=tla.arch.MTE2)

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

                    mutex_l0a = mutex_l0a0 if (l0_buf_idx == c0) else mutex_l0a1
                    mutex_l1a.lock(pipe=tla.arch.MTE1)
                    mutex_l0a.lock(pipe=tla.arch.MTE1)
                    tla.copy(l0_a, l1_a_by_l0)
                    mutex_l0a.unlock(pipe=tla.arch.MTE1)
                    mutex_l1a.unlock(pipe=tla.arch.MTE1)

                    mutex_l0b = mutex_l0b0 if (l0_buf_idx == c0) else mutex_l0b1
                    mutex_l1b.lock(pipe=tla.arch.MTE1)
                    mutex_l0b.lock(pipe=tla.arch.MTE1)
                    tla.copy(l0_b, l1_b_by_l0)
                    mutex_l0b.unlock(pipe=tla.arch.MTE1)
                    mutex_l1b.unlock(pipe=tla.arch.MTE1)

                    mutex_l0a.lock(pipe=tla.arch.CUBE)
                    mutex_l0b.lock(pipe=tla.arch.CUBE)

                    init_c = True if k_l1 == 0 and k_l0 == 0 else False
                    unit_flag = (
                        0b11
                        if (k_l1 == k_l1_count - 1) and (k_l0 == k_l0_count - 1)
                        else 0b10
                    )

                    tla.mmad(l0_c, l0_a, l0_b, init_c=init_c, unit_flag=unit_flag)

                    mutex_l0b.unlock(pipe=tla.arch.CUBE)
                    mutex_l0a.unlock(pipe=tla.arch.CUBE)

                    l0_buf_idx = c1 - l0_buf_idx
                l1_buf_idx = c1 - l1_buf_idx

            tla.copy(
                gm_c_by_core,
                l0_c,
                tla.params.CopyL0C2DstParams(unit_flag=0b11),
            )


def run(args: argparse.Namespace) -> int:
    from helpers import (
        get_block_num,
        create_tla_tensor,
        compare,
    )

    torch.npu.set_device(args.device)
    print(
        f"--- mnk=({args.m},{args.n},{args.k}) "
        f"layout={args.layout_a}/{args.layout_b} "
        f"dtype={args.dtype_a}/{args.dtype_b}/{args.dtype_c} ---"
    )
    torch.manual_seed(0)
    dtypes = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}
    dtype_a = dtypes[args.dtype_a]
    dtype_b = dtypes[args.dtype_b]
    dtype_c = dtypes[args.dtype_c]
    a = torch.rand(args.m, args.k, dtype=dtype_a, device="cpu") * 10.0 - 5.0
    b = torch.rand(args.k, args.n, dtype=dtype_b, device="cpu") * 10.0 - 5.0
    c = torch.rand(args.m, args.n, dtype=dtype_c, device="cpu") * 10.0 - 5.0
    ref = a.float() @ b.float()
    if dtype_c in (torch.float16, torch.bfloat16):
        ref = ref.to(dtype_c).float()

    a = (
        a.contiguous() if args.layout_a == "row" else a.permute(1, 0).contiguous()
    ).npu()
    b = (
        b.contiguous() if args.layout_b == "row" else b.permute(1, 0).contiguous()
    ).npu()
    c = c.contiguous().npu()
    a_tensor = create_tla_tensor(a, args.layout_a)
    b_tensor = create_tla_tensor(b, args.layout_b)
    c_tensor = create_tla_tensor(c, "row")

    artifact = tla.compile(
        basic_mmad_mutex_kernel,
        a_tensor,
        b_tensor,
        c_tensor,
        TilingParams(),
        options="--npu-arch 3510",
    )
    block_num = get_block_num(args.block_num, args.device, kind="cube")
    artifact(a_tensor, b_tensor, c_tensor, block_num=block_num)
    torch.npu.synchronize()

    passed = compare(c.detach().cpu(), ref, args.k)
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
    parser.add_argument("--dtype-a", choices=("f16", "bf16", "f32"), default="f16")
    parser.add_argument("--dtype-b", choices=("f16", "bf16", "f32"), default="f16")
    parser.add_argument("--dtype-c", choices=("f16", "bf16", "f32"), default="f32")
    parser.add_argument("--block-num", type=int, default=-1)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
