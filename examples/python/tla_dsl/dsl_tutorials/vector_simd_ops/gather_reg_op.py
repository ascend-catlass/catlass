# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end on-device check for ``tla.gather`` Reg→Reg path (vselr).

f16 VL (128 lanes): ``dst[i] = src[index[i] % 128]``. Index is u16 bit pattern
carried as i16 for GM↔UB copy support.

``--mixed`` adds a small cube GM→L1 copy so the kernel is AIC/AIV mix
(regression for issue #822: ``gather_reg_*`` must carry AIV core type).
"""

from __future__ import annotations

import argparse
from typing import Any

import catlass.tla as tla

VL_ELE = 128  # full VL for f16 / u16
CUBE_M = 16
CUBE_N = 16


@tla.kernel
def gather_reg_op(
    mem_src: tla.Tensor, mem_idx: tla.Tensor, mem_dst: tla.Tensor
) -> None:
    ub_loaded = tla.flag("ub_loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    vec_done = tla.flag("vec_done", tla.pipes.VECTOR, tla.pipes.MTE3)

    src_gm = tla.tile_view(mem_src, tla.make_shape(VL_ELE), tla.make_coord(0))
    idx_gm = tla.tile_view(mem_idx, tla.make_shape(VL_ELE), tla.make_coord(0))
    dst_gm = tla.tile_view(mem_dst, tla.make_shape(VL_ELE), tla.make_coord(0))

    src_ub = tla.make_tensor_like(
        tla.allocate(VL_ELE, tla.Float16, tla.AddressSpace.ub, 256),
        src_gm,
        tla.arch.RowMajor,
    )
    # Host uses int16; Pass casts to ui16 for the bc stub.
    idx_ub = tla.make_tensor_like(
        tla.allocate(VL_ELE, tla.Int16, tla.AddressSpace.ub, 256),
        idx_gm,
        tla.arch.RowMajor,
    )
    dst_ub = tla.make_tensor_like(
        tla.allocate(VL_ELE, tla.Float16, tla.AddressSpace.ub, 256),
        dst_gm,
        tla.arch.RowMajor,
    )

    with tla.vector():
        tla.copy(src_ub, src_gm)
        tla.copy(idx_ub, idx_gm)

        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)

        with tla.vec.func(mode="simd"):
            src_tile = tla.tile_view(src_ub, tla.make_shape(VL_ELE), tla.make_coord(0))
            idx_tile = tla.tile_view(idx_ub, tla.make_shape(VL_ELE), tla.make_coord(0))
            dst_tile = tla.tile_view(dst_ub, tla.make_shape(VL_ELE), tla.make_coord(0))
            src_reg = src_tile.load()
            idx_reg = idx_tile.load()
            dst_tile.store(tla.gather(src_reg, idx_reg))

        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)

        tla.copy(dst_gm, dst_ub)
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def gather_reg_mixed_op(
    mem_src: tla.Tensor,
    mem_idx: tla.Tensor,
    mem_dst: tla.Tensor,
    mem_cube: tla.Tensor,
) -> None:
    ub_loaded = tla.flag("ub_loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    vec_done = tla.flag("vec_done", tla.pipes.VECTOR, tla.pipes.MTE3)

    src_gm = tla.tile_view(mem_src, tla.make_shape(VL_ELE), tla.make_coord(0))
    idx_gm = tla.tile_view(mem_idx, tla.make_shape(VL_ELE), tla.make_coord(0))
    dst_gm = tla.tile_view(mem_dst, tla.make_shape(VL_ELE), tla.make_coord(0))
    cube_gm = tla.tile_view(
        mem_cube, tla.make_shape(CUBE_M, CUBE_N), tla.make_coord(0, 0)
    )

    src_ub = tla.make_tensor_like(
        tla.allocate(VL_ELE, tla.Float16, tla.AddressSpace.ub, 256),
        src_gm,
        tla.arch.RowMajor,
    )
    idx_ub = tla.make_tensor_like(
        tla.allocate(VL_ELE, tla.Int16, tla.AddressSpace.ub, 256),
        idx_gm,
        tla.arch.RowMajor,
    )
    dst_ub = tla.make_tensor_like(
        tla.allocate(VL_ELE, tla.Float16, tla.AddressSpace.ub, 256),
        dst_gm,
        tla.arch.RowMajor,
    )
    l1 = tla.make_tensor_like(
        tla.allocate(CUBE_M * CUBE_N, tla.Float16, tla.AddressSpace.l1, 512),
        cube_gm,
        tla.arch.zN,
    )

    with tla.cube():
        tla.copy(l1, cube_gm)

    with tla.vector():
        tla.copy(src_ub, src_gm)
        tla.copy(idx_ub, idx_gm)

        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)

        with tla.vec.func(mode="simd"):
            src_tile = tla.tile_view(src_ub, tla.make_shape(VL_ELE), tla.make_coord(0))
            idx_tile = tla.tile_view(idx_ub, tla.make_shape(VL_ELE), tla.make_coord(0))
            dst_tile = tla.tile_view(dst_ub, tla.make_shape(VL_ELE), tla.make_coord(0))
            dst_tile.store(tla.gather(src_tile.load(), idx_tile.load()))

        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)

        tla.copy(dst_gm, dst_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def _runtime_tensor(dev_buf: Any) -> Any:
    return tla.from_dlpack(
        dev_buf.contiguous(),
        layout_tag=tla.arch.RowMajor,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="On-device e2e for tla.gather Reg→Reg (vselr)."
    )
    parser.add_argument("--run", action="store_true")
    parser.add_argument(
        "--mixed",
        action="store_true",
        help="Compile/run AIC/AIV mix kernel (cube GM→L1 + vector gather).",
    )
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("pass --run to compile and execute on NPU")

    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)

    src = torch.linspace(-17.0, 46.0, VL_ELE, dtype=torch.float16, device="npu")
    # Reverse permutation + wrap: index[i] = (VL_ELE - 1 - i) % VL_ELE
    idx_u = (VL_ELE - 1 - torch.arange(VL_ELE, dtype=torch.int32)) % VL_ELE
    idx = idx_u.to(dtype=torch.int16).to(device="npu")
    dst = torch.full((VL_ELE,), -999.0, dtype=torch.float16, device="npu")

    tla_src = _runtime_tensor(src)
    tla_idx = _runtime_tensor(idx)
    tla_dst = _runtime_tensor(dst)

    if args.mixed:
        cube = torch.ones((CUBE_M, CUBE_N), dtype=torch.float16, device="npu")
        tla_cube = _runtime_tensor(cube)
        artifact = tla.compile(
            gather_reg_mixed_op,
            tla_src,
            tla_idx,
            tla_dst,
            tla_cube,
            options="--npu-arch 3510",
        )
        artifact(tla_src, tla_idx, tla_dst, tla_cube, block_num=1)
        mode = "mixed"
    else:
        artifact = tla.compile(
            gather_reg_op,
            tla_src,
            tla_idx,
            tla_dst,
            options="--npu-arch 3510",
        )
        artifact(tla_src, tla_idx, tla_dst, block_num=1)
        mode = "aiv"

    torch.npu.synchronize()

    expected = src[idx_u.to(torch.long).to(device="npu")]
    ok = bool(torch.isclose(dst, expected, rtol=0.0, atol=1e-3).all())
    print(
        f"compile_ok=True host=torch_npu op=gather dtype=f16 layout=row "
        f"path=reg mode={mode}"
    )
    print(f"kernel.o path={artifact.kernel_binary_path}")
    print("launch_ok=True")
    print(f"output equals expected gather (reg)? {ok}")
    if not ok:
        mism = (
            (~torch.isclose(dst, expected, rtol=0.0, atol=1e-3))
            .nonzero(as_tuple=False)
            .flatten()[:8]
        )
        print(f"first mismatch indices={mism.tolist()}")
        if mism.numel():
            i = int(mism[0].item())
            print(f"dst[{i}] actual={float(dst[i])} expected={float(expected[i])}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
