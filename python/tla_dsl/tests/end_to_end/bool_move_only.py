# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""On-device Bool VectorSSA move-only: GM↔UB copy + load/store (mask ALL / H)."""

from __future__ import annotations

import argparse

import catlass.tla as tla
import torch
from catlass.tla.runtime import from_dlpack

VECTOR_ELE = 256


@tla.kernel
def bool_move_all(mem_in: tla.Tensor, mem_out: tla.Tensor) -> None:
    ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    vec_done = tla.flag("vec_done", tla.arch.VECTOR, tla.arch.MTE3)
    gm_in = tla.tile_view(mem_in, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    gm_out = tla.tile_view(mem_out, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    ub_in = tla.make_tensor_like(
        tla.allocate(VECTOR_ELE, tla.Bool, tla.AddressSpace.ub, 256),
        gm_in,
        tla.arch.RowMajor,
    )
    ub_out = tla.make_tensor_like(
        tla.allocate(VECTOR_ELE, tla.Bool, tla.AddressSpace.ub, 256),
        gm_out,
        tla.arch.RowMajor,
    )
    with tla.vector():
        tla.copy(ub_in, gm_in)
        tla.copy(ub_out, gm_out)
        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)
        with tla.vec.func(mode="simd"):
            ub_out.store(
                ub_in.load(),
                mask=tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Bool),
            )
        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)
        tla.copy(gm_out, ub_out)
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def bool_move_half(mem_in: tla.Tensor, mem_out: tla.Tensor) -> None:
    ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    vec_done = tla.flag("vec_done", tla.arch.VECTOR, tla.arch.MTE3)
    gm_in = tla.tile_view(mem_in, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    gm_out = tla.tile_view(mem_out, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    ub_in = tla.make_tensor_like(
        tla.allocate(VECTOR_ELE, tla.Bool, tla.AddressSpace.ub, 256),
        gm_in,
        tla.arch.RowMajor,
    )
    ub_out = tla.make_tensor_like(
        tla.allocate(VECTOR_ELE, tla.Bool, tla.AddressSpace.ub, 256),
        gm_out,
        tla.arch.RowMajor,
    )
    with tla.vector():
        tla.copy(ub_in, gm_in)
        tla.copy(ub_out, gm_out)
        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)
        with tla.vec.func(mode="simd"):
            ub_out.store(
                ub_in.load(),
                mask=tla.create_mask(pattern=tla.mask.H, dtype=tla.Bool),
            )
        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)
        tla.copy(gm_out, ub_out)
        tla.pipe_barrier(tla.pipes.ALL)


def _host_pattern() -> torch.Tensor:
    """Distinct 32B DataBlocks so a first-block broadcast bug fails the compare."""
    idx = torch.arange(VECTOR_ELE, dtype=torch.int64)
    block, lane = idx // 32, idx % 32
    return ((lane == block) | ((lane & 7) == (block & 7))).to(torch.bool)


def _run(kernel, device: int, half: bool) -> None:
    torch.npu.set_device(device)
    host = _host_pattern()
    inp = host.to("npu")
    out = torch.zeros(VECTOR_ELE, dtype=torch.bool, device="npu")
    tin = from_dlpack(inp.contiguous(), layout_tag=tla.arch.RowMajor)
    tout = from_dlpack(out.contiguous(), layout_tag=tla.arch.RowMajor)
    tla.compile(kernel, tin, tout, options="--npu-arch 3510")(tin, tout)
    torch.npu.synchronize()
    got = out.cpu()
    expected = (
        host
        if not half
        else torch.cat(
            [host[: VECTOR_ELE // 2], torch.zeros(VECTOR_ELE // 2, dtype=torch.bool)]
        )
    )
    if not torch.equal(got, expected):
        bad = (got != expected).nonzero(as_tuple=False).flatten()[0].item()
        raise SystemExit(
            f"bool move-only mismatch (half={half}): index={bad} "
            f"got={bool(got[bad])} expected={bool(expected[bad])}"
        )
    print(f"ok bool_move_{'half' if half else 'all'} device={device} n={VECTOR_ELE}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", nargs="?", default="all", choices=("all", "half", "both")
    )
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if args.mode in ("all", "both"):
        _run(bool_move_all, args.device, half=False)
    if args.mode in ("half", "both"):
        _run(bool_move_half, args.device, half=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
