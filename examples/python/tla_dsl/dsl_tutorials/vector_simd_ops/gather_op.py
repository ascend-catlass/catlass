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

import argparse
from typing import Any

import catlass.tla as tla

VECTOR_ELE = 64


@tla.kernel
def gather_op(mem_src: tla.Tensor, mem_idx: tla.Tensor, mem_dst: tla.Tensor) -> None:
    x_loaded = tla.flag("x_loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    idx_loaded = tla.flag("idx_loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    done = tla.flag("done", tla.pipes.VECTOR, tla.pipes.MTE3)
    src_gm = tla.tile_view(mem_src, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    idx_gm = tla.tile_view(mem_idx, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    dst_gm = tla.tile_view(mem_dst, tla.make_shape(VECTOR_ELE), tla.make_coord(0))

    src_ptr = tla.allocate(VECTOR_ELE, tla.Float32, tla.AddressSpace.ub, 256)
    idx_ptr = tla.allocate(VECTOR_ELE, tla.Int32, tla.AddressSpace.ub, 256)
    dst_ptr = tla.allocate(VECTOR_ELE, tla.Float32, tla.AddressSpace.ub, 256)

    src_ub = tla.make_tensor_like(src_ptr, src_gm, tla.arch.RowMajor)
    idx_ub = tla.make_tensor_like(idx_ptr, idx_gm, tla.arch.RowMajor)
    dst_ub = tla.make_tensor_like(dst_ptr, dst_gm, tla.arch.RowMajor)

    with tla.vector():
        tla.copy(src_ub, src_gm)
        tla.copy(idx_ub, idx_gm)

        tla.set_flag(x_loaded)
        tla.wait_flag(x_loaded)

        with tla.vec.func(mode="simd"):
            x_tile = tla.tile_view(
                src_ub, tla.make_shape(VECTOR_ELE), tla.make_coord(0)
            )
            idx_tile = tla.tile_view(
                idx_ub, tla.make_shape(VECTOR_ELE), tla.make_coord(0)
            )
            dst_tile = tla.tile_view(
                dst_ub, tla.make_shape(VECTOR_ELE), tla.make_coord(0)
            )

            indices = idx_tile.load()
            gathered = tla.gather(x_tile, indices)
            dst_tile.store(gathered)

        tla.set_flag(done)
        tla.wait_flag(done)

        tla.copy(dst_gm, dst_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def _runtime_tensor(dev_buf: Any) -> Any:
    return tla.from_dlpack(
        dev_buf.contiguous(),
        layout_tag=tla.arch.RowMajor,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--dtype", choices=("f32",), default="f32")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("pass --run")

    import torch
    import torch_npu

    torch.npu.set_device(args.device)
    src = torch.linspace(-17.0, 46.0, VECTOR_ELE, dtype=torch.float32, device="npu")
    idx = torch.arange(VECTOR_ELE - 1, -1, -1, dtype=torch.int32, device="npu")
    dst = torch.full((VECTOR_ELE,), -999.0, dtype=torch.float32, device="npu")

    tla_src = _runtime_tensor(src)
    tla_idx = _runtime_tensor(idx)
    tla_dst = _runtime_tensor(dst)

    artifact = tla.compile(
        gather_op,
        tla_src,
        tla_idx,
        tla_dst,
        options="--npu-arch 3510",
    )
    artifact(tla_src, tla_idx, tla_dst, block_num=1)
    torch.npu.synchronize()

    expected = src[idx.to(torch.long)]
    ok = bool(torch.isclose(dst, expected, rtol=0.0, atol=1e-4).all())
    print("compile_ok=True host=torch_npu op=gather dtype=f32 layout=row")
    print(f"kernel.o path={artifact.kernel_binary_path}")
    print("launch_ok=True")
    print(f"output equals expected gather? {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
