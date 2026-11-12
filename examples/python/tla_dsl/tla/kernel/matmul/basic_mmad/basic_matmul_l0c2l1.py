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

M_DIM = 50
N_DIM = 60
K_DIM = 64

M_L1 = 64
N_L1 = 64
K_L1 = 64

# First mmad: C = A(M,K) @ B(K,N)
L1A_SIZE = M_L1 * K_L1
L1B_SIZE = K_L1 * N_L1
L0A1_SIZE = M_L1 * K_L1
L0B1_SIZE = K_L1 * N_L1

# L0C accumulator, reused by both mmads (holds C, then E)
L0C_SIZE = M_L1 * N_L1

# Second mmad: E = C(M,N) @ D(N,N)
L1C_SIZE = M_L1 * N_L1
L1D_SIZE = N_L1 * N_L1
L0A2_SIZE = M_L1 * N_L1
L0B2_SIZE = N_L1 * N_L1

L0A_SIZE = max(L0A1_SIZE, L0A2_SIZE)
L0B_SIZE = max(L0B1_SIZE, L0B2_SIZE)

DESCRIPTION = "Basic MMAD L0C->L1 pipeline: E = (A@B)@D; f32, cube-only."


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


@tla.kernel
def basic_matmul_l0c2l1(
    mem_a: tla.Tensor,
    mem_b: tla.Tensor,
    mem_d: tla.Tensor,
    mem_e: tla.Tensor,
) -> None:
    l1ab_done = tla.flag("l1ab_done", tla.arch.MTE2, tla.arch.MTE1)
    l0_loaded = tla.flag("l0_loaded", tla.arch.MTE1, tla.arch.CUBE)
    mmad_done = tla.flag("mmad_done", tla.arch.CUBE, tla.arch.FIX)

    fixl1_done = tla.flag("fixl1_done", tla.arch.FIX, tla.arch.MTE1)

    l1d_done = tla.flag("l1d_done", tla.arch.MTE2, tla.arch.MTE1)

    l1a_ptr = tla.allocate(L1A_SIZE, tla.Float32, tla.AddressSpace.l1, 512)
    l1b_ptr = tla.allocate(L1B_SIZE, tla.Float32, tla.AddressSpace.l1, 512)
    l1c_ptr = tla.allocate(L1C_SIZE, tla.Float32, tla.AddressSpace.l1, 512)
    l1d_ptr = tla.allocate(L1D_SIZE, tla.Float32, tla.AddressSpace.l1, 512)

    # the two-mmads shares the same l0a/l0b/l0c space
    l0a_ptr = tla.allocate(L0A_SIZE, tla.Float32, tla.AddressSpace.l0a, 512)
    l0b_ptr = tla.allocate(L0B_SIZE, tla.Float32, tla.AddressSpace.l0b, 512)
    l0c_ptr = tla.allocate(L0C_SIZE, tla.Float32, tla.AddressSpace.l0c, 1024)

    with tla.cube():
        # ---- C = A @ B -------------------------------------------------
        gm_a = tla.tile_view(mem_a, tla.make_shape(M_L1, K_L1), tla.make_coord(0, 0))
        gm_b = tla.tile_view(mem_b, tla.make_shape(K_L1, N_L1), tla.make_coord(0, 0))
        l1_a = tla.make_tensor_like(l1a_ptr, gm_a, tla.arch.zN)
        l1_b = tla.make_tensor_like(l1b_ptr, gm_b, tla.arch.zN)

        tla.copy(l1_a, gm_a)
        tla.copy(l1_b, gm_b)
        tla.set_flag(l1ab_done)
        tla.wait_flag(l1ab_done)

        l1_a_l0 = tla.tile_view(l1_a, tla.make_shape(M_L1, K_L1), tla.make_coord(0, 0))
        l1_b_l0 = tla.tile_view(l1_b, tla.make_shape(K_L1, N_L1), tla.make_coord(0, 0))
        l0a_a = tla.make_tensor_like(l0a_ptr, l1_a_l0, tla.arch.zN)
        l0b_b = tla.make_tensor_like(l0b_ptr, l1_b_l0, tla.arch.nZ)

        tla.copy(l0a_a, l1_a_l0)
        tla.copy(l0b_b, l1_b_l0)
        tla.set_flag(l0_loaded)
        tla.wait_flag(l0_loaded)

        gm_e = tla.tile_view(mem_e, tla.make_shape(M_L1, M_L1), tla.make_coord(0, 0))
        l0c_c = tla.make_tensor_like(l0c_ptr, gm_e, tla.arch.L0Clayout)
        tla.mmad(l0c_c, l0a_a, l0b_b, init_c=True)
        tla.set_flag(mmad_done)
        tla.wait_flag(mmad_done)

        # copy l0c->l1c, L0Clayout->zN
        l1_c = tla.make_tensor_like(l1c_ptr, l0c_c, tla.arch.zN)
        tla.copy(l1_c, l0c_c)
        tla.set_flag(fixl1_done)
        tla.wait_flag(fixl1_done)

        # ---- E = C @ D -------------------------------------------------
        gm_d = tla.tile_view(mem_d, tla.make_shape(N_L1, N_L1), tla.make_coord(0, 0))
        l1_d = tla.make_tensor_like(l1d_ptr, gm_d, tla.arch.zN)
        tla.copy(l1_d, gm_d)
        tla.set_flag(l1d_done)
        tla.wait_flag(l1d_done)

        l1_c_l0 = tla.tile_view(l1_c, tla.make_shape(M_L1, N_L1), tla.make_coord(0, 0))
        l1_d_l0 = tla.tile_view(l1_d, tla.make_shape(N_L1, N_L1), tla.make_coord(0, 0))
        l0a_c = tla.make_tensor_like(l0a_ptr, l1_c_l0, tla.arch.zN)
        l0b_d = tla.make_tensor_like(l0b_ptr, l1_d_l0, tla.arch.nZ)
        tla.copy(l0a_c, l1_c)
        tla.copy(l0b_d, l1_d)
        tla.set_flag(l0_loaded)
        tla.wait_flag(l0_loaded)

        l0c_e = tla.make_tensor_like(
            l0c_ptr, l0c_c, tla.arch.L0Clayout
        )  # two-mmad out shapes are same
        tla.mmad(l0c_e, l0a_c, l0b_d, init_c=True)
        tla.set_flag(mmad_done)
        tla.wait_flag(mmad_done)

        tla.copy(gm_e, l0c_e)

        tla.pipe_barrier(tla.pipes.ALL)


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------


def golden(a, b, d):
    import torch

    a = a.to(torch.float32)
    b = b.to(torch.float32)
    d = d.to(torch.float32)
    return a @ b @ d


def prepare_npu(buf, layout: str):
    storage = buf.contiguous() if layout == "row" else buf.permute(1, 0).contiguous()
    return storage.npu()


def run(args: argparse.Namespace) -> int:
    import torch
    import torch_npu

    from helpers import (
        create_tla_tensor,
        compare,
    )

    torch.npu.set_device(args.device)
    # e = a @ b @ d
    a = torch.rand((M_DIM, K_DIM), dtype=torch.float32, device="cpu") * 10.0 - 5.0
    b = torch.rand((K_DIM, N_DIM), dtype=torch.float32, device="cpu") * 10.0 - 5.0
    d = torch.rand((N_DIM, N_DIM), dtype=torch.float32, device="cpu") * 10.0 - 5.0
    e = torch.full((M_DIM, N_DIM), args.sentinel, dtype=torch.float32, device="cpu")
    ref = golden(a, b, d)

    a = prepare_npu(a, "row")
    b = prepare_npu(b, "row")
    d = prepare_npu(d, "row")
    e = prepare_npu(e, "row")
    a_tensor = create_tla_tensor(a, "row")
    b_tensor = create_tla_tensor(b, "row")
    d_tensor = create_tla_tensor(d, "row")
    e_tensor = create_tla_tensor(e, "row")

    artifact = tla.compile(
        basic_matmul_l0c2l1,
        a_tensor,
        b_tensor,
        d_tensor,
        e_tensor,
        options="--npu-arch 3510",
    )
    artifact(a_tensor, b_tensor, d_tensor, e_tensor, block_num=args.block_num)
    torch.npu.synchronize()

    passed = compare(e.detach().cpu(), ref, max(K_DIM, N_DIM))
    print(f"passed={passed} cache_key={artifact.cache_key}")
    print(f"kernel.o={artifact.kernel_binary_path}")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--block-num", type=int, default=1)
    parser.add_argument("--sentinel", type=float, default=-9.0)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
