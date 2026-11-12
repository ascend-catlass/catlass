# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Minimal ptr/make_tensor MMAD: Kernel + Host in one file.

Static problem sizes. Host binds buffers via ``from_dlpack`` Prefer ``basic_matmul.py`` for dynamic-GM e2e.
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

M_DIM = 64
N_DIM = 64
K_DIM = 64
L1_M_DIM = 32
L1_N_DIM = 32
L1_K_DIM = 32
L0A_BYTES = 32 * 32 * 4
L0B_BYTES = 32 * 32 * 4
L0C_BYTES = 32 * 32 * 4


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


@tla.kernel
def basic_mmad_ptr(lhs: tla.Tensor, rhs: tla.Tensor, out: tla.Tensor) -> None:
    l0c_data_ready = tla.flag("l0c_data_ready", tla.arch.CUBE, tla.arch.FIX)
    l1_loaded = tla.flag("l1_loaded", tla.arch.MTE2, tla.arch.MTE1)
    l0_loaded = tla.flag("l0_loaded", tla.arch.MTE1, tla.arch.CUBE)

    l1a_ptr = tla.allocate(L1_M_DIM * L1_K_DIM, tla.Float32, tla.AddressSpace.l1, 512)
    l1b_ptr = tla.allocate(L1_K_DIM * L1_N_DIM, tla.Float32, tla.AddressSpace.l1, 512)
    l0a_ptr = tla.allocate(L0A_BYTES // 4, tla.Float32, tla.AddressSpace.l0a, 512)
    l0b_ptr = tla.allocate(L0B_BYTES // 4, tla.Float32, tla.AddressSpace.l0b, 512)
    l0c_ptr = tla.allocate(L0C_BYTES // 4, tla.Float32, tla.AddressSpace.l0c, 1024)

    with tla.cube():
        gm_a = tla.make_tensor(
            32 + lhs.ptr + 32 * K_DIM,
            tla.make_layout(
                tla.make_shape(L1_M_DIM, L1_K_DIM), tla.make_stride(K_DIM, 1)
            ),
        )
        gm_b = tla.tile_view(
            rhs, tla.make_shape(L1_K_DIM, L1_N_DIM), tla.make_coord(1, 0)
        )
        gm_c = tla.make_tensor(
            out.ptr,
            tla.make_layout(
                tla.make_shape(L1_K_DIM, L1_N_DIM), tla.make_stride(L1_N_DIM, 1)
            ),
        )
        l1_a = tla.make_tensor_like(l1a_ptr, gm_a, tla.arch.zN)
        l1_b = tla.make_tensor_like(l1b_ptr, gm_b, tla.arch.zN)
        tla.copy(l1_a, gm_a)
        tla.copy(l1_b, gm_b)

        tla.set_flag(l1_loaded)
        tla.wait_flag(l1_loaded)

        l1_a_by_l0 = tla.tile_view(
            l1_a, tla.make_shape(L1_M_DIM, L1_K_DIM), tla.make_coord(0, 0)
        )
        l1_b_by_l0 = tla.tile_view(
            l1_b, tla.make_shape(L1_K_DIM, L1_N_DIM), tla.make_coord(0, 0)
        )
        l0_a = tla.make_tensor_like(l0a_ptr, l1_a_by_l0, tla.arch.zN)
        l0_b = tla.make_tensor_like(l0b_ptr, l1_b_by_l0, tla.arch.nZ)
        l0_c = tla.make_tensor_like(l0c_ptr, gm_c, tla.arch.L0Clayout)
        tla.copy(l0_a, l1_a_by_l0)
        tla.copy(l0_b, l1_b_by_l0)

        tla.set_flag(l0_loaded)
        tla.wait_flag(l0_loaded)

        tla.mmad(l0_c, l0_a, l0_b, init_c=True)

        tla.set_flag(l0c_data_ready)
        tla.wait_flag(l0c_data_ready)
        tla.copy(gm_c, l0_c)


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    from helpers import compare
    from catlass.tla.runtime import from_dlpack

    torch.npu.set_device(args.device)
    torch.manual_seed(0)
    a = torch.rand(M_DIM, K_DIM, dtype=torch.float32, device="cpu") * 10.0 - 5.0
    b = torch.rand(K_DIM, N_DIM, dtype=torch.float32, device="cpu") * 10.0 - 5.0
    c = torch.rand(L1_M_DIM, L1_N_DIM, dtype=torch.float32, device="cpu") * 10.0 - 5.0
    # Kernel uses a ptr+offset and b tile at (1,0); match that window.
    ref = a[32:, 32:] @ b[32:, :32]

    # Contiguous on CPU, then upload so contiguous does not run on NPU.
    a = a.contiguous().npu()
    b = b.contiguous().npu()
    c = c.contiguous().npu()
    a_tensor = from_dlpack(a, layout_tag=tla.arch.RowMajor, origin_shape=(M_DIM, K_DIM))
    b_tensor = from_dlpack(b, layout_tag=tla.arch.RowMajor, origin_shape=(K_DIM, N_DIM))
    c_tensor = from_dlpack(
        c, layout_tag=tla.arch.RowMajor, origin_shape=(L1_M_DIM, L1_N_DIM)
    )

    artifact = tla.compile(
        basic_mmad_ptr,
        a_tensor,
        b_tensor,
        c_tensor,
        options="--npu-arch 3510",
    )
    artifact(a_tensor, b_tensor, c_tensor, block_num=args.block_num)
    torch.npu.synchronize()

    passed = compare(c.detach().cpu(), ref, K_DIM)
    print(f"passed={passed} kernel.o={artifact.kernel_binary_path}")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal ptr/make_tensor MMAD.")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--block-num", type=int, default=1)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
