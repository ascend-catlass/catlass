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

_DSL_EXAMPLE_PATH = str(
    next(p for p in Path(__file__).resolve().parents if (p / "helpers").is_dir())
)

if _DSL_EXAMPLE_PATH not in sys.path:
    sys.path.append(_DSL_EXAMPLE_PATH)

import argparse

import catlass.tla as tla
from catlass.params import CopyL0C2DstParams, L0C2UBMode

M_DIM = 60
N_DIM = 100
K_DIM = 64

M_L1 = 64
N_L1 = 128
K_L1 = 64

L1A_SIZE = M_L1 * K_L1
L1B_SIZE = K_L1 * N_L1

L0A_SIZE = M_L1 * K_L1
L0B_SIZE = K_L1 * N_L1
L0C_SIZE = M_L1 * N_L1

UB_C_SIZE = L0C_SIZE

DESCRIPTION = "Basic Mixed FIXPIPE L0C→UB(NO_SPLIT_VEC_0)→GM(ColumnMajor); f32 only."


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


@tla.kernel
def basic_mixed_fixpipe_nz2dn(
    lhs: tla.Tensor,
    rhs: tla.Tensor,
    out: tla.Tensor,
) -> None:
    mmad_done = tla.flag("mmad_done", tla.arch.CUBE, tla.arch.FIX)
    l1b_done = tla.flag("l1b_done", tla.arch.MTE2, tla.arch.MTE1)
    l0_loaded = tla.flag("l0_loaded", tla.arch.MTE1, tla.arch.CUBE)

    fix_done = tla.cross_flag("fix_done", mode=4)

    l1a_ptr = tla.allocate(L1A_SIZE, tla.Float32, tla.AddressSpace.l1, 512)
    l1b_ptr = tla.allocate(L1B_SIZE, tla.Float32, tla.AddressSpace.l1, 512)
    l0a_ptr = tla.allocate(L0A_SIZE, tla.Float32, tla.AddressSpace.l0a, 512)
    l0b_ptr = tla.allocate(L0B_SIZE, tla.Float32, tla.AddressSpace.l0b, 512)
    l0c_ptr = tla.allocate(L0C_SIZE, tla.Float32, tla.AddressSpace.l0c, 1024)

    ub_c_ptr = tla.allocate(UB_C_SIZE, tla.Float32, tla.AddressSpace.ub, 256)

    with tla.cube():
        gm_a = tla.tile_view(lhs, tla.make_shape(M_L1, K_L1), tla.make_coord(0, 0))
        gm_b = tla.tile_view(rhs, tla.make_shape(K_L1, N_L1), tla.make_coord(0, 0))
        l1_a = tla.make_tensor_like(l1a_ptr, gm_a, tla.arch.zN)
        l1_b = tla.make_tensor_like(l1b_ptr, gm_b, tla.arch.zN)

        tla.copy(l1_a, gm_a)
        tla.copy(l1_b, gm_b)

        tla.set_flag(l1b_done)
        tla.wait_flag(l1b_done)

        l1_a_l0 = tla.tile_view(l1_a, tla.make_shape(M_L1, K_L1), tla.make_coord(0, 0))
        l0_a = tla.make_tensor_like(l0a_ptr, l1_a_l0, tla.arch.zN)
        l1_b_l0 = tla.tile_view(l1_b, tla.make_shape(K_L1, N_L1), tla.make_coord(0, 0))
        l0_b = tla.make_tensor_like(l0b_ptr, l1_b_l0, tla.arch.nZ)

        tla.copy(l0_a, l1_a_l0)
        tla.copy(l0_b, l1_b_l0)

        tla.set_flag(l0_loaded)
        tla.wait_flag(l0_loaded)

        out_col = tla.make_tensor(
            out.ptr,
            tla.make_layout(
                tla.make_shape(M_DIM, N_DIM),
                tla.make_stride(1, M_DIM),
                layoutTag=tla.arch.ColumnMajor,
            ),
            coord=tla.make_coord(0, 0),
        )
        gm_c = tla.tile_view(out_col, tla.make_shape(M_L1, N_L1), tla.make_coord(0, 0))

        l0_c = tla.make_tensor_like(l0c_ptr, gm_c, tla.arch.L0Clayout)
        tla.mmad(l0_c, l0_a, l0_b, init_c=True)

        tla.set_flag(mmad_done)
        tla.wait_flag(mmad_done)

        ub_c = tla.make_tensor_like(ub_c_ptr, l0_c, tla.arch.ColumnMajor)
        tla.copy(
            ub_c, l0_c, params=CopyL0C2DstParams(l0c2ub_mode=L0C2UBMode.NO_SPLIT_VEC_0)
        )

        tla.cross_core_set_flag(fix_done, tla.arch.FIX, aiv_id=0)
        tla.pipe_barrier(tla.pipes.ALL)

    with tla.vector():
        vec_idx = tla.arch.sub_block_idx()

        gm_c = tla.tile_view(out, tla.make_shape(N_L1, M_L1), tla.make_coord(0, 0))
        ub_c = tla.make_tensor_like(ub_c_ptr, gm_c, tla.arch.RowMajor)

        if vec_idx == 0:
            tla.cross_core_wait_flag(fix_done, tla.arch.MTE3, aiv_id=0)
            tla.copy(gm_c, ub_c)  # copy to gm row->row

        tla.pipe_barrier(tla.pipes.ALL)


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------


def golden(lhs, rhs):
    import torch

    ref = lhs.to(torch.float32) @ rhs.to(torch.float32)
    return ref.transpose(1, 0).contiguous()


def prepare_npu(buf, layout: str):
    storage = buf.contiguous() if layout == "row" else buf.permute(1, 0).contiguous()
    return storage.npu()


def run(args: argparse.Namespace) -> int:
    import torch
    import torch_npu

    from helpers import create_tla_tensor

    mi, ni, ki = int(args.m), int(args.n), int(args.k)

    torch.npu.set_device(args.device)
    print(f"--- mnk=({mi},{ni},{ki}) ---")
    lhs = torch.rand(mi, ki, dtype=torch.float32, device="cpu") * 10.0 - 5.0
    rhs = torch.rand(ki, ni, dtype=torch.float32, device="cpu") * 10.0 - 5.0
    out = torch.full((ni, mi), args.sentinel, dtype=torch.float32, device="cpu")
    ref = golden(lhs, rhs)

    lhs = prepare_npu(lhs, args.layout_a)
    rhs = prepare_npu(rhs, args.layout_b)
    out = prepare_npu(out, "row")
    a_tensor = create_tla_tensor(lhs, args.layout_a)
    b_tensor = create_tla_tensor(rhs, args.layout_b)
    c_tensor = create_tla_tensor(out, "row")

    artifact = tla.compile(
        basic_mixed_fixpipe_nz2dn,
        a_tensor,
        b_tensor,
        c_tensor,
        options="--npu-arch 3510",
    )
    artifact(a_tensor, b_tensor, c_tensor, block_num=args.block_num)
    torch.npu.synchronize()

    result = out.detach().cpu().float()
    passed = bool(torch.isclose(result, ref, rtol=0.0, atol=1e-4).all())
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
    parser.add_argument("--block-num", type=int, default=1)
    parser.add_argument("--sentinel", type=float, default=-9.0)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
