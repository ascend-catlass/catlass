# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""MIX AIC:AIV = 1:1 end-to-end sample (vs default 1:2 ``basic_mixed.py``).

Differences from ``basic_mixed.py``:
- ``@tla.kernel(core_ratio=tla.arch.C1V1)``
- one AIV owns the full MxN epilogue (no ``sub_block_idx`` half-split)
- cross-core sync uses ``mode=4`` with ``aiv_id=0`` only; wait uses
  ``tla.arch.SCALAR`` (CV 1:1 FFTS / SyncAll contract; lowers to
  ``SET.CROSS.CORE`` + ``WAIT.FLAG.DEV.S``, not INTRA)
- cube result stages L0C→GM (CV 1:1 forbids tightly-coupled L0C→UB)
"""

from __future__ import annotations

import sys
from pathlib import Path

# On-device sample under python/tla_dsl/tests/end_to_end; battery case
# mixed-core-ratio-11. Reuse example helpers.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_DSL_EXAMPLE_PATH = str(_REPO_ROOT / "examples" / "python" / "tla_dsl")

if _DSL_EXAMPLE_PATH not in sys.path:
    # append (not insert(0)) to avoid shadowing installed packages (G.PSL.03).
    sys.path.append(_DSL_EXAMPLE_PATH)

import argparse

import catlass.tla as tla

M_DIM = 32
N_DIM = 32
K_DIM = 32
VECTOR_TILE_M = 32
VECTOR_TILE_N = 32
VECTOR_REG_TILE_M = 2
UB_TILE_BYTES = VECTOR_TILE_M * VECTOR_TILE_N * 4
L1_STAGE_BYTES = 256 * 1024
L0A_BYTES = 32 * 32 * 4
L0B_BYTES = 32 * 32 * 4
L0C_BYTES = 32 * 32 * 4

DESCRIPTION = "Basic Mixed Cube+Vector add with core_ratio=tla.arch.C1V1."


@tla.kernel(core_ratio=tla.arch.C1V1)
def basic_mixed_core_ratio_11(
    lhs: tla.Tensor,
    rhs: tla.Tensor,
    out: tla.Tensor,
    addend: tla.Tensor,
) -> None:
    m = lhs.origin_shape[0]
    k = lhs.origin_shape[1]
    n = rhs.origin_shape[1]

    mmad_done = tla.flag("mmad_done", tla.arch.CUBE, tla.arch.FIX)
    l1_loaded = tla.flag("l1_loaded", tla.arch.MTE2, tla.arch.MTE1)
    l0_loaded = tla.flag("l0_loaded", tla.arch.MTE1, tla.arch.CUBE)

    ub_load_ready = tla.flag("ub_load_ready", tla.arch.VECTOR, tla.arch.MTE2)
    ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    vec_done = tla.flag("vec_done", tla.arch.VECTOR, tla.arch.MTE3)

    fix_done = tla.cross_flag("fix_done", mode=4)

    l1a_ptr = tla.allocate(L1_STAGE_BYTES // 4, tla.Float32, tla.AddressSpace.l1, 512)
    l1b_ptr = tla.allocate(L1_STAGE_BYTES // 4, tla.Float32, tla.AddressSpace.l1, 512)
    l0a_ptr = tla.allocate(L0A_BYTES // 4, tla.Float32, tla.AddressSpace.l0a, 512)
    l0b_ptr = tla.allocate(L0B_BYTES // 4, tla.Float32, tla.AddressSpace.l0b, 512)
    l0c_ptr = tla.allocate(L0C_BYTES // 4, tla.Float32, tla.AddressSpace.l0c, 1024)

    c_ub_ptr = tla.allocate(UB_TILE_BYTES // 4, tla.Float32, tla.AddressSpace.ub, 256)
    addend_ub_ptr = tla.allocate(
        UB_TILE_BYTES // 4, tla.Float32, tla.AddressSpace.ub, 256
    )
    result_ub_ptr = tla.allocate(
        UB_TILE_BYTES // 4, tla.Float32, tla.AddressSpace.ub, 256
    )

    with tla.cube():
        gm_a = tla.tile_view(lhs, tla.make_shape(m, k), tla.make_coord(0, 0))
        gm_b = tla.tile_view(rhs, tla.make_shape(k, n), tla.make_coord(0, 0))
        gm_c = tla.tile_view(out, tla.make_shape(m, n), tla.make_coord(0, 0))

        l1_a = tla.make_tensor_like(l1a_ptr, gm_a, tla.arch.zN)
        l1_b = tla.make_tensor_like(l1b_ptr, gm_b, tla.arch.zN)
        tla.copy(l1_a, gm_a)
        tla.copy(l1_b, gm_b)

        tla.set_flag(l1_loaded)
        tla.wait_flag(l1_loaded)

        l1_a_l0 = tla.tile_view(l1_a, tla.make_shape(m, k), tla.make_coord(0, 0))
        l1_b_l0 = tla.tile_view(l1_b, tla.make_shape(k, n), tla.make_coord(0, 0))
        l0_a = tla.make_tensor_like(l0a_ptr, l1_a_l0, tla.arch.zN)
        l0_b = tla.make_tensor_like(l0b_ptr, l1_b_l0, tla.arch.nZ)
        l0_c = tla.make_tensor_like(l0c_ptr, gm_c, tla.arch.L0Clayout)
        tla.copy(l0_a, l1_a_l0)
        tla.copy(l0_b, l1_b_l0)

        tla.set_flag(l0_loaded)
        tla.wait_flag(l0_loaded)

        tla.mmad(l0_c, l0_a, l0_b, init_c=True)

        tla.set_flag(mmad_done)
        tla.wait_flag(mmad_done)

        # Ascend 950 CV 1:1 禁止紧耦合 L0C→UB；结果经 GM 交给单 AIV（对齐
        # AscendNPU-IR InsertLoadStoreForMixCV / AscendC SyncAll 数据通路）。
        tla.copy(gm_c, l0_c)

        tla.cross_core_set_flag(fix_done, tla.arch.FIX, aiv_id=0)
        tla.pipe_barrier(tla.pipes.ALL)

    with tla.vector():
        gm_result = tla.tile_view(
            out,
            tla.make_shape(VECTOR_TILE_M, VECTOR_TILE_N),
            tla.make_coord(0, 0),
        )
        gm_addend = tla.tile_view(
            addend,
            tla.make_shape(VECTOR_TILE_M, VECTOR_TILE_N),
            tla.make_coord(0, 0),
        )
        ub_result = tla.make_tensor_like(result_ub_ptr, gm_result, tla.arch.RowMajor)
        ub_addend = tla.make_tensor_like(addend_ub_ptr, gm_addend, tla.arch.RowMajor)
        ub_c = tla.make_tensor_like(c_ub_ptr, gm_result, tla.arch.RowMajor)

        # CV 1:1 FFTS / SyncAll：wait 固定 PIPE_S（AscendC wait_flag_dev(PIPE_S)）。
        # mode=4 仅 aiv_id=0；pass 按 core_ratio 落 SET.CROSS.CORE + WAIT.FLAG.DEV。
        tla.cross_core_wait_flag(fix_done, tla.arch.SCALAR, aiv_id=0)

        tla.set_flag(ub_load_ready)
        tla.wait_flag(ub_load_ready)
        tla.copy(ub_c, gm_result)
        tla.copy(ub_addend, gm_addend)
        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)

        for row_tile_idx in tla.range(0, VECTOR_TILE_M // VECTOR_REG_TILE_M, 1):
            with tla.vec.func(mode="simd"):
                c_chunk = tla.tile_view(
                    ub_c,
                    tla.make_shape(VECTOR_REG_TILE_M, VECTOR_TILE_N),
                    tla.make_coord(row_tile_idx, 0),
                )
                addend_chunk = tla.tile_view(
                    ub_addend,
                    tla.make_shape(VECTOR_REG_TILE_M, VECTOR_TILE_N),
                    tla.make_coord(row_tile_idx, 0),
                )
                result_chunk = tla.tile_view(
                    ub_result,
                    tla.make_shape(VECTOR_REG_TILE_M, VECTOR_TILE_N),
                    tla.make_coord(row_tile_idx, 0),
                )
                result_chunk.store(c_chunk.load() + addend_chunk.load())

        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)
        tla.copy(gm_result, ub_result)

        tla.pipe_barrier(tla.pipes.ALL)


def golden(lhs, rhs, addend):
    import torch

    return lhs.to(torch.float32) @ rhs.to(torch.float32) + addend.to(torch.float32)


def prepare_npu(buf, layout: str):
    storage = buf.contiguous() if layout == "row" else buf.permute(1, 0).contiguous()
    return storage.npu()


def run(args: argparse.Namespace) -> int:
    import torch
    import torch_npu

    from helpers import (
        get_block_num,
        create_tla_tensor,
        compare,
    )

    mi, ni, ki = int(args.m), int(args.n), int(args.k)
    if (mi, ni) != (M_DIM, N_DIM):
        raise SystemExit(
            f"this sample fixes M=N={M_DIM} for the 1:1 single-AIV epilogue; got {(mi, ni)}"
        )

    torch.npu.set_device(args.device)
    print(f"--- core_ratio=tla.arch.C1V1 mnk=({mi},{ni},{ki}) ---")
    print(f"kernel.options={basic_mixed_core_ratio_11.options}")

    lhs = torch.rand(mi, ki, dtype=torch.float32, device="cpu") * 10.0 - 5.0
    rhs = torch.rand(ki, ni, dtype=torch.float32, device="cpu") * 10.0 - 5.0
    addend = torch.full((mi, ni), 3.0, dtype=torch.float32, device="cpu")
    out = torch.full((mi, ni), args.sentinel, dtype=torch.float32, device="cpu")
    ref = golden(lhs, rhs, addend)

    lhs = prepare_npu(lhs, args.layout_a)
    rhs = prepare_npu(rhs, args.layout_b)
    out = prepare_npu(out, "row")
    addend = prepare_npu(addend, "row")
    a_tensor = create_tla_tensor(lhs, args.layout_a)
    b_tensor = create_tla_tensor(rhs, args.layout_b)
    c_tensor = create_tla_tensor(out, "row")
    d_tensor = create_tla_tensor(addend, "row")

    artifact = tla.compile(
        basic_mixed_core_ratio_11,
        a_tensor,
        b_tensor,
        c_tensor,
        d_tensor,
        options="--npu-arch 3510",
    )
    block_num = get_block_num(args.block_num, args.device, kind="mix")
    # 本样例固定处理 (0,0) 上的整块 MxN，未按 block_idx 分片；多 block 会竞态写同一 GM。
    if args.block_num < 0:
        block_num = 1
    artifact(a_tensor, b_tensor, c_tensor, d_tensor, block_num=block_num)
    torch.npu.synchronize()

    passed = compare(out.detach().cpu(), ref, ki)
    print(f"passed={passed} cache_key={artifact.cache_key}")
    print(f"kernel.o={artifact.kernel_binary_path}")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--m", type=int, default=M_DIM)
    parser.add_argument("--n", type=int, default=N_DIM)
    parser.add_argument("--k", type=int, default=K_DIM)
    parser.add_argument("--layout-a", choices=("row", "col"), default="row")
    parser.add_argument("--layout-b", choices=("row", "col"), default="row")
    parser.add_argument("--block-num", type=int, default=-1)
    parser.add_argument("--sentinel", type=float, default=-9.0)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
