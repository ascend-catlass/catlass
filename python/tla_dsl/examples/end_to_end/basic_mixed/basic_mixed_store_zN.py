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

import sys
from pathlib import Path

_DSL_EXAMPLE_PATH = str((Path(__file__).resolve().parent / "..").resolve())

if _DSL_EXAMPLE_PATH not in sys.path:
    sys.path.insert(0, _DSL_EXAMPLE_PATH)

import argparse

import catlass.tla as tla
from catlass.params import BlockStoreParams


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


M_DIM = 60
N_DIM = 64
K_DIM = 256

M_L1 = 64
N_L1 = 64
K_L1 = 256

ELEM = None  # ElemSpec, bound by run()
UB_A_SIZE = M_L1 // 2 * K_L1
UB_A_ZN_SIZE = M_L1 // 2 * K_L1


L1A_SIZE = M_L1 * K_L1
L1B_SIZE = K_L1 * N_L1

L0A_SIZE = M_L1 * K_L1
L0B_SIZE = K_L1 * N_L1
L0C_SIZE = M_L1 * N_L1

DESCRIPTION = "Basic Mixed UB RowMajor→zN Store + BlockStore, over every element type the route supports."


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


@tla.kernel
def basic_mixed_store_zN(
    lhs: tla.Tensor,
    rhs: tla.Tensor,
    out: tla.Tensor,
) -> None:
    mmad_done = tla.flag("mmad_done", tla.arch.CUBE, tla.arch.FIX)
    l1b_done = tla.flag("l1b_done", tla.arch.MTE2, tla.arch.MTE1)
    l0_loaded = tla.flag("l0_loaded", tla.arch.MTE1, tla.arch.CUBE)

    ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)

    store_done = tla.flag("store_done", tla.arch.VECTOR, tla.arch.MTE3)

    ub2l1_done = tla.cross_flag("ub2l1_done")

    l1a_ptr = tla.allocate(L1A_SIZE, ELEM.tla(), tla.AddressSpace.l1, 512)
    l1b_ptr = tla.allocate(L1B_SIZE, ELEM.tla(), tla.AddressSpace.l1, 512)
    l0a_ptr = tla.allocate(L0A_SIZE, ELEM.tla(), tla.AddressSpace.l0a, 512)
    l0b_ptr = tla.allocate(L0B_SIZE, ELEM.tla(), tla.AddressSpace.l0b, 512)
    l0c_ptr = tla.allocate(L0C_SIZE, ELEM.tla_acc(), tla.AddressSpace.l0c, 512)

    ub_a_ptr = tla.allocate(UB_A_SIZE, ELEM.tla(), tla.AddressSpace.ub, 256)
    ub_a_zN_ptr = tla.allocate(UB_A_ZN_SIZE, ELEM.tla(), tla.AddressSpace.ub, 256)

    with tla.cube():
        gm_a = tla.tile_view(lhs, tla.make_shape(M_L1, K_L1), tla.make_coord(0, 0))
        gm_b = tla.tile_view(rhs, tla.make_shape(K_L1, N_L1), tla.make_coord(0, 0))
        l1_a = tla.make_tensor_like(l1a_ptr, gm_a, tla.arch.zN)
        l1_b = tla.make_tensor_like(l1b_ptr, gm_b, tla.arch.zN)

        tla.copy(l1_b, gm_b)
        tla.set_flag(l1b_done)

        l1_b_l0 = tla.tile_view(l1_b, tla.make_shape(K_L1, N_L1), tla.make_coord(0, 0))
        l0_b = tla.make_tensor_like(l0b_ptr, l1_b_l0, tla.arch.nZ)
        tla.wait_flag(l1b_done)
        tla.copy(l0_b, l1_b_l0)

        l1_a_l0 = tla.tile_view(l1_a, tla.make_shape(M_L1, K_L1), tla.make_coord(0, 0))
        l0_a = tla.make_tensor_like(l0a_ptr, l1_a_l0, tla.arch.zN)

        tla.cross_core_wait_flag(ub2l1_done, tla.arch.MTE1)
        tla.copy(l0_a, l1_a_l0)

        tla.set_flag(l0_loaded)
        tla.wait_flag(l0_loaded)

        gm_c = tla.tile_view(out, tla.make_shape(M_L1, N_L1), tla.make_coord(0, 0))
        l0_c = tla.make_tensor_like(l0c_ptr, gm_c, tla.arch.L0Clayout)
        tla.mmad(l0_c, l0_a, l0_b, init_c=True)

        tla.set_flag(mmad_done)
        tla.wait_flag(mmad_done)
        tla.copy(gm_c, l0_c)

        tla.pipe_barrier(tla.pipes.ALL)

    with tla.vector():
        vec_idx = tla.arch.sub_block_idx()

        gm_a_tile = tla.tile_view(lhs, tla.make_shape(M_L1, K_L1), tla.make_coord(0, 0))
        gm_a = tla.tile_view(
            gm_a_tile,
            tla.make_shape(gm_a_tile.origin_shape[0] // 2, K_L1),
            tla.make_coord(vec_idx, 0),
        )

        ub_a_tile = tla.make_tensor(
            ub_a_ptr,
            tla.make_layout(tla.make_shape(M_L1 // 2, K_L1), tla.make_stride(K_L1, 1)),
            tla.make_coord(0, 0),
        )
        ub_a = tla.tile_view(
            ub_a_tile,
            tla.make_shape(gm_a.origin_shape[0], gm_a.origin_shape[1]),
            tla.make_coord(0, 0),
        )

        tla.copy(ub_a, gm_a)

        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)

        ub_a_zN = tla.make_tensor_like(ub_a_zN_ptr, ub_a, tla.arch.zN)

        vf_row_loops = ub_a.origin_shape[0]
        vf_len = ELEM.vf_len
        vf_col_loops = ceil_div(K_L1, vf_len)
        block_stride = (
            ub_a_zN.stride[1][1] // ub_a_zN.shape[1][0]
        )  # data blocks in zN rows
        for row_tile_idx in tla.range(vf_row_loops):
            for col_tile_idx in tla.range(vf_col_loops):
                with tla.vec.func(mode="simd"):
                    a_chunk = tla.tile_view(
                        ub_a,
                        tla.make_shape(1, vf_len),
                        tla.make_coord(row_tile_idx, col_tile_idx),
                    )
                    a_zN_chunk = tla.tile_view(
                        ub_a_zN,
                        tla.make_shape(1, vf_len),
                        tla.make_coord(row_tile_idx, col_tile_idx),
                    )
                    a_zN_chunk.store(
                        a_chunk.load(),
                        params=BlockStoreParams(block_stride=block_stride),
                    )

        tla.set_flag(store_done)
        tla.wait_flag(store_done)

        l1_a_tile = tla.make_tensor_like(l1a_ptr, gm_a_tile, tla.arch.zN)
        l1_a = tla.tile_view(
            l1_a_tile,
            tla.make_shape(l1_a_tile.origin_shape[0] // 2, K_L1),
            tla.make_coord(vec_idx, 0),
        )

        tla.copy(l1_a, ub_a_zN)
        tla.cross_core_set_flag(ub2l1_done, tla.arch.MTE3)  # wait uba -> l1a

        tla.pipe_barrier(tla.pipes.ALL)


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    import torch
    import torch_npu  # noqa: F401

    from common import elem_spec, make_operand_tensors

    global ELEM
    ELEM = elem_spec(args.dtype)
    mi, ni, ki = int(args.m), int(args.n), int(args.k)

    torch.npu.set_device(args.device)
    torch.manual_seed(0)
    print(f"--- basic_mixed_store_zN dtype={args.dtype} mnk=({mi},{ni},{ki}) ---")

    # The GM shape is the CLI's, not the L1 tile's: the destination row count is
    # a runtime value, so an M that is not a multiple of the fractal height is
    # exactly what this route has to get right.
    a, b, ref = ELEM.operands(mi, ni, ki)
    out = torch.full(
        (mi, ni), ELEM.sentinel(args.sentinel), dtype=ELEM.torch_acc(), device="cpu"
    ).npu()
    a_tensor, b_tensor, c_tensor = make_operand_tensors(
        ELEM, a, b, out, args.layout_a, args.layout_b
    )

    artifact = tla.compile(
        basic_mixed_store_zN, a_tensor, b_tensor, c_tensor, options="--npu-arch 3510"
    )
    artifact(a_tensor, b_tensor, c_tensor, block_num=args.block_num)
    torch.npu.synchronize()

    passed = ELEM.check(out.detach().cpu(), ref, ki)
    print(f"passed={passed} cache_key={artifact.cache_key}")
    return 0 if passed else 1


def main() -> int:
    from common import ELEM_CHOICES

    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--m", type=int, default=M_DIM)
    parser.add_argument("--n", type=int, default=N_DIM)
    parser.add_argument("--k", type=int, default=K_DIM)
    parser.add_argument("--dtype", choices=ELEM_CHOICES, default="f32")
    parser.add_argument("--layout-a", choices=("row", "col"), default="row")
    parser.add_argument("--layout-b", choices=("row", "col"), default="row")
    parser.add_argument("--block-num", type=int, default=1)
    parser.add_argument("--sentinel", type=float, default=-9.0)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
