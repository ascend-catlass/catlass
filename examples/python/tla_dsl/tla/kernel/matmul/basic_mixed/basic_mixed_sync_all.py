# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""MIX sample: ``tla.sync_all()`` between the cube and vector halves of a kernel.

The AICs compute ``lhs @ rhs`` into the GM workspace ``wk`` (L0C -> GM). A
``tla.sync_all()`` written once on the cube side and once on the vector side
then barriers all cores of the launch: every AIV of a block arrives, every AIC
waits for them and barriers across cores, and only then are the AIVs released.
After the barrier each AIV moves its share of ``wk`` into ``out`` through UB.

The sample launches two blocks. AIC ``i`` owns rows
``[i*CUBE_TILE_M, (i+1)*CUBE_TILE_M)``; AIV ``j`` of block ``i`` owns the
global AIV index ``i*sub_block_num + j`` and walks the ``AIV_TILE_M``-row tiles
of ``wk`` with stride ``block_num * sub_block_num``. That covers all
``M_DIM / AIV_TILE_M`` tiles with both 1:1 (2 AIVs) and 1:2 (4 AIVs).

Both kernel entries share ``_sync_all_impl`` and differ only in ``core_ratio``.
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

M_DIM = 32
N_DIM = 32
K_DIM = 32
CUBE_TILE_M = 16
AIV_TILE_M = 8
AIV_TILES = M_DIM // AIV_TILE_M
REQUIRED_BLOCK_NUM = M_DIM // CUBE_TILE_M
L1_STAGE_BYTES = 256 * 1024
L0A_BYTES = CUBE_TILE_M * K_DIM * 4
L0B_BYTES = K_DIM * N_DIM * 4
L0C_BYTES = CUBE_TILE_M * N_DIM * 4
UB_TILE_BYTES = AIV_TILE_M * N_DIM * 4

DESCRIPTION = (
    "Basic Mixed matmul with a full-core tla.sync_all() barrier: cube writes "
    "the GM workspace, the AIVs then move it into out through UB."
)

# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


@tla.jit
def _sync_all_impl(
    lhs: tla.Tensor,
    rhs: tla.Tensor,
    wk_c: tla.Tensor,
    out_c: tla.Tensor,
) -> None:
    k = lhs.origin_shape[1]
    n = rhs.origin_shape[1]

    mmad_done = tla.flag("mmad_done", tla.pipes.CUBE, tla.pipes.FIX)
    l1_loaded = tla.flag("l1_loaded", tla.pipes.MTE2, tla.pipes.MTE1)
    l0_loaded = tla.flag("l0_loaded", tla.pipes.MTE1, tla.pipes.CUBE)
    wk_in_ub = tla.flag("wk_in_ub", tla.pipes.MTE2, tla.pipes.MTE3)
    out_stored = tla.flag("out_stored", tla.pipes.MTE3, tla.pipes.MTE2)

    l1a_ptr = tla.allocate(L1_STAGE_BYTES // 4, tla.Float32, tla.AddressSpace.l1, 512)
    l1b_ptr = tla.allocate(L1_STAGE_BYTES // 4, tla.Float32, tla.AddressSpace.l1, 512)
    l0a_ptr = tla.allocate(L0A_BYTES // 4, tla.Float32, tla.AddressSpace.l0a, 512)
    l0b_ptr = tla.allocate(L0B_BYTES // 4, tla.Float32, tla.AddressSpace.l0b, 512)
    l0c_ptr = tla.allocate(L0C_BYTES // 4, tla.Float32, tla.AddressSpace.l0c, 1024)
    ub_ptr = tla.allocate(UB_TILE_BYTES // 4, tla.Float32, tla.AddressSpace.ub, 256)

    with tla.cube():
        aic_idx = tla.arch.block_idx()
        gm_a = tla.tile_view(
            lhs, tla.make_shape(CUBE_TILE_M, k), tla.make_coord(aic_idx, 0)
        )
        gm_b = tla.tile_view(rhs, tla.make_shape(k, n), tla.make_coord(0, 0))
        gm_wk = tla.tile_view(
            wk_c, tla.make_shape(CUBE_TILE_M, n), tla.make_coord(aic_idx, 0)
        )

        l1_a = tla.make_tensor_like(l1a_ptr, gm_a, tla.arch.zN)
        l1_b = tla.make_tensor_like(l1b_ptr, gm_b, tla.arch.zN)
        tla.copy(l1_a, gm_a)
        tla.copy(l1_b, gm_b)

        tla.set_flag(l1_loaded)
        tla.wait_flag(l1_loaded)

        l1_a_l0 = tla.tile_view(
            l1_a, tla.make_shape(CUBE_TILE_M, k), tla.make_coord(0, 0)
        )
        l1_b_l0 = tla.tile_view(l1_b, tla.make_shape(k, n), tla.make_coord(0, 0))
        l0_a = tla.make_tensor_like(l0a_ptr, l1_a_l0, tla.arch.zN)
        l0_b = tla.make_tensor_like(l0b_ptr, l1_b_l0, tla.arch.nZ)
        l0_c = tla.make_tensor_like(l0c_ptr, gm_wk, tla.arch.L0Clayout)
        tla.copy(l0_a, l1_a_l0)
        tla.copy(l0_b, l1_b_l0)

        tla.set_flag(l0_loaded)
        tla.wait_flag(l0_loaded)

        tla.mmad(l0_c, l0_a, l0_b, init_c=True)

        tla.set_flag(mmad_done)
        tla.wait_flag(mmad_done)

        tla.copy(gm_wk, l0_c)

        # The cube call waits for every AIV of this block, barriers with the
        # other AICs, and only then releases the AIVs.
        tla.sync_all()

    with tla.vector():
        # The vector call signals this AIV's arrival and blocks until every AIC
        # of the launch has finished writing its workspace slice.
        tla.sync_all()

        sub_idx = tla.arch.sub_block_idx()
        sub_num = tla.arch.sub_block_num().to(tla.Int32)
        aiv = tla.arch.block_idx() * sub_num + sub_idx
        aiv_total = tla.arch.block_num() * sub_num

        ub_wk = tla.make_tensor_like(
            ub_ptr,
            tla.tile_view(wk_c, tla.make_shape(AIV_TILE_M, n), tla.make_coord(0, 0)),
            tla.arch.RowMajor,
        )

        for tile in tla.range(aiv, AIV_TILES, aiv_total):
            gm_wk = tla.tile_view(
                wk_c, tla.make_shape(AIV_TILE_M, n), tla.make_coord(tile, 0)
            )
            gm_out = tla.tile_view(
                out_c, tla.make_shape(AIV_TILE_M, n), tla.make_coord(tile, 0)
            )
            tla.copy(ub_wk, gm_wk)
            tla.set_flag(wk_in_ub)
            tla.wait_flag(wk_in_ub)
            tla.copy(gm_out, ub_wk)
            tla.set_flag(out_stored)
            tla.wait_flag(out_stored)

        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel(core_ratio=tla.arch.C1V1)
def basic_mixed_sync_all_c1v1(
    lhs: tla.Tensor,
    rhs: tla.Tensor,
    wk_c: tla.Tensor,
    out_c: tla.Tensor,
) -> None:
    """Two AICs, one AIV each; the two AIVs split the AIV tiles."""
    _sync_all_impl(lhs, rhs, wk_c, out_c)


@tla.kernel(core_ratio=tla.arch.C1V2)
def basic_mixed_sync_all_c1v2(
    lhs: tla.Tensor,
    rhs: tla.Tensor,
    wk_c: tla.Tensor,
    out_c: tla.Tensor,
) -> None:
    """Two AICs, two AIVs each; the four AIVs split the AIV tiles."""
    _sync_all_impl(lhs, rhs, wk_c, out_c)


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------


def golden(lhs, rhs):
    import torch

    return lhs.to(torch.float32) @ rhs.to(torch.float32)


def run(args: argparse.Namespace) -> int:
    import torch
    import torch_npu  # noqa: F401

    from helpers import compare, create_tla_tensor

    torch.npu.set_device(args.device)
    torch.manual_seed(0)
    print(
        f"--- sync_all core_ratio={args.core_ratio} "
        f"mnk=({M_DIM},{N_DIM},{K_DIM}) block_num={REQUIRED_BLOCK_NUM} ---"
    )

    lhs = torch.rand(M_DIM, K_DIM, dtype=torch.float32) * 10.0 - 5.0
    rhs = torch.rand(K_DIM, N_DIM, dtype=torch.float32) * 10.0 - 5.0
    ref = golden(lhs, rhs)
    wk = torch.full((M_DIM, N_DIM), args.sentinel, dtype=torch.float32).npu()
    out = torch.full((M_DIM, N_DIM), args.sentinel, dtype=torch.float32).npu()

    a_tensor = create_tla_tensor(lhs.npu(), "row")
    b_tensor = create_tla_tensor(rhs.npu(), "row")
    wk_tensor = create_tla_tensor(wk, "row")
    out_tensor = create_tla_tensor(out, "row")

    kernel = (
        basic_mixed_sync_all_c1v1
        if args.core_ratio == "c1v1"
        else basic_mixed_sync_all_c1v2
    )
    artifact = tla.compile(
        kernel, a_tensor, b_tensor, wk_tensor, out_tensor, options="--npu-arch 3510"
    )
    artifact(a_tensor, b_tensor, wk_tensor, out_tensor, block_num=REQUIRED_BLOCK_NUM)
    torch.npu.synchronize()

    wk_passed = compare(wk.detach().cpu(), ref, K_DIM)
    out_passed = compare(out.detach().cpu(), ref, K_DIM)
    passed = bool(wk_passed and out_passed)
    print(
        f"wk_passed={wk_passed} out_passed={out_passed} passed={passed} "
        f"cache_key={artifact.cache_key}"
    )
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--core-ratio", choices=("c1v1", "c1v2"), default="c1v2")
    parser.add_argument("--sentinel", type=float, default=-9.0)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
