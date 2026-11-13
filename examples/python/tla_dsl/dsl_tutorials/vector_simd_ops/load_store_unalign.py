# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end unaligned ``tile.load`` / ``tile.store`` with auto Pre/Post.

Modes (positional ``op``):

* ``add`` — unaligned UPDATE load/store (f32 / f16 / bf16).
  Python emits UPDATE only; ``tla-unalign-ureg`` inserts
  ``load_unalign_pre`` / ``create_ureg`` / ``store_unalign_post`` (clean
  ``tla.range``: Pre/create before for, Post after; ``for``⊃``if`` or a
  same-window load after store keeps per-trip dual-instr). Lowers to
  ``vldas``+``vldus.post`` / empty ureg+``vstus.post``+``vstas``.
  ``--offset``: element offset of the first window (``1`` is not 32-byte aligned).
  Default HIVM count/advance = ``tile_extent`` (``origin_shape``); an explicit
  ``post_update_stride`` must be in ``(0, VL]`` and is the caller's responsibility.
  ``--tile-extent``: window length in elements (default: dtype full-register
  width; ``1`` = one element per UPDATE).
  ``--mode once``: e2e-only; set loop trips to 1 (not a DSL API).
  ``--mode update`` (default): ``LOOPS`` trips.
  ``--inplace``: ``z = z + x`` with UnAlign load+store UPDATE on the same ``z``
  tile (shared storage identity).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import catlass.tla as tla
from catlass.params import PostMode, UnalignLoadParams, UnalignStoreParams
from catlass.tla.runtime import from_dlpack

from vector_op_harness import require_torch_npu, runtime_kwargs

# Compile-time specialization. ``run()`` overwrites these before ``tla.compile``.
_KERNEL_DTYPE = tla.Float32
TILE_EXTENT = 64
OFFSET = 1
LOOPS = 3
PAD = 64
_MODE = "update"  # "update" | "once"
_INPLACE = False
N_ELE = OFFSET + (LOOPS - 1) * TILE_EXTENT + TILE_EXTENT + PAD

_LOAD_UPDATE = UnalignLoadParams(post_mode=PostMode.POST_MODE_UPDATE)
_STORE_UPDATE = UnalignStoreParams(post_mode=PostMode.POST_MODE_UPDATE)


def _window(ub, offset: int):
    """Rank-1 window of ``TILE_EXTENT`` at ``offset`` elements (coord=0).

    A ``tile_view`` still carries its coord beside the base pointer, so the
    window is rebuilt with ``coord<0>`` the same way UPDATE does. Default
    layouts set ``origin_shape == shape``, so HIVM count/advance = TILE_EXTENT.
    """
    base = tla.tile_view(ub, tla.make_shape(TILE_EXTENT), tla.make_coord(0))
    layout = tla.make_layout(
        tla.make_shape(TILE_EXTENT),
        tla.make_stride(1),
        layoutTag=tla.arch.RowMajor,
    )
    return tla.make_tensor(base.ptr + offset, layout, tla.make_coord(0))


@tla.kernel
def load_store_unalign_op(
    mem_x: tla.Tensor, mem_y: tla.Tensor, mem_z: tla.Tensor
) -> None:
    ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    vec_done = tla.flag("vec_done", tla.arch.VECTOR, tla.arch.MTE3)

    x_gm = tla.tile_view(mem_x, tla.make_shape(N_ELE), tla.make_coord(0))
    y_gm = tla.tile_view(mem_y, tla.make_shape(N_ELE), tla.make_coord(0))
    z_gm = tla.tile_view(mem_z, tla.make_shape(N_ELE), tla.make_coord(0))

    x_ptr = tla.allocate(N_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    y_ptr = tla.allocate(N_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    z_ptr = tla.allocate(N_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    x_ub = tla.make_tensor_like(x_ptr, x_gm, tla.arch.RowMajor)
    y_ub = tla.make_tensor_like(y_ptr, y_gm, tla.arch.RowMajor)
    z_ub = tla.make_tensor_like(z_ptr, z_gm, tla.arch.RowMajor)

    with tla.vector():
        tla.copy(x_ub, x_gm)
        tla.copy(y_ub, y_gm)
        tla.copy(z_ub, z_gm)

        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)
        # Pointer arithmetic must dominate the vector helper call. Building the
        # window inside ``vec.func`` leaves ``ptr_add`` in that region, and the
        # SIMD outline pass cannot erase the region while the call still uses it.
        x_tile = _window(x_ub, OFFSET)
        y_tile = _window(y_ub, OFFSET)
        z_tile = _window(z_ub, OFFSET)
        with tla.vec.func(mode="simd"):
            # Clean ``tla.range``: ``tla-unalign-ureg`` puts Pre/Post outside.
            trips = 1 if _MODE == "once" else LOOPS
            for _ in tla.range(trips):
                x_reg, x_tile = x_tile.load(_LOAD_UPDATE)
                y_reg, y_tile = y_tile.load(_LOAD_UPDATE)
                z_tile = z_tile.store(tla.add(x_reg, y_reg), _STORE_UPDATE)

        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)
        tla.copy(z_gm, z_ub)
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def load_store_unalign_inplace_op(mem_x: tla.Tensor, mem_z: tla.Tensor) -> None:
    """In-place ``z += x``: UnAlign load+store UPDATE share ``z`` storage identity."""
    ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    vec_done = tla.flag("vec_done", tla.arch.VECTOR, tla.arch.MTE3)

    x_gm = tla.tile_view(mem_x, tla.make_shape(N_ELE), tla.make_coord(0))
    z_gm = tla.tile_view(mem_z, tla.make_shape(N_ELE), tla.make_coord(0))

    x_ptr = tla.allocate(N_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    z_ptr = tla.allocate(N_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    x_ub = tla.make_tensor_like(x_ptr, x_gm, tla.arch.RowMajor)
    z_ub = tla.make_tensor_like(z_ptr, z_gm, tla.arch.RowMajor)

    with tla.vector():
        tla.copy(x_ub, x_gm)
        tla.copy(z_ub, z_gm)

        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)
        x_tile = _window(x_ub, OFFSET)
        z_tile = _window(z_ub, OFFSET)
        with tla.vec.func(mode="simd"):
            trips = 1 if _MODE == "once" else LOOPS
            for _ in tla.range(trips):
                x_reg, x_tile = x_tile.load(_LOAD_UPDATE)
                z_cur = z_tile
                z_reg, _ = z_cur.load(_LOAD_UPDATE)
                z_tile = z_cur.store(tla.add(x_reg, z_reg), _STORE_UPDATE)

        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)
        tla.copy(z_gm, z_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def _trip_count() -> int:
    return 1 if _MODE == "once" else LOOPS


def _default_tile_extent(dtype_name: str) -> int:
    return {"f32": 64, "f16": 128, "bf16": 128}[dtype_name]


def _configure(
    dtype_name: str,
    offset: int,
    mode: str,
    tile_extent: int | None,
    *,
    inplace: bool = False,
) -> None:
    global TILE_EXTENT, OFFSET, LOOPS, N_ELE, _KERNEL_DTYPE, _MODE, _INPLACE
    global _LOAD_UPDATE, _STORE_UPDATE
    dtype = {
        "f32": tla.Float32,
        "f16": tla.Float16,
        "bf16": tla.BFloat16,
    }[dtype_name]
    _KERNEL_DTYPE = dtype
    TILE_EXTENT = (
        _default_tile_extent(dtype_name) if tile_extent is None else tile_extent
    )
    if TILE_EXTENT < 1:
        raise SystemExit("--tile-extent must be >= 1")
    OFFSET = offset
    _MODE = mode
    _INPLACE = inplace
    LOOPS = 3
    N_ELE = OFFSET + (_trip_count() - 1) * TILE_EXTENT + TILE_EXTENT + PAD
    _LOAD_UPDATE = UnalignLoadParams(post_mode=PostMode.POST_MODE_UPDATE)
    _STORE_UPDATE = UnalignStoreParams(post_mode=PostMode.POST_MODE_UPDATE)


def _wrap(buf: Any) -> Any:
    return from_dlpack(
        buf.contiguous(), layout_tag=tla.arch.RowMajor
    ).mark_compact_shape_dynamic(0)


def _expected_add(x: Any, y: Any, z_sentinel: Any) -> Any:
    """Dense windows: each trip covers ``TILE_EXTENT`` elements (advance == extent)."""
    z = z_sentinel.clone()
    for i in range(_trip_count()):
        start = OFFSET + i * TILE_EXTENT
        end = start + TILE_EXTENT
        z[start:end] = x[start:end] + y[start:end]
    return z


def _expected_inplace_add(x: Any, z0: Any) -> Any:
    """In-place windows: ``z[w] = z0[w] + x[w]`` per trip."""
    z = z0.clone()
    for i in range(_trip_count()):
        start = OFFSET + i * TILE_EXTENT
        end = start + TILE_EXTENT
        z[start:end] = z0[start:end] + x[start:end]
    return z


def run(args: argparse.Namespace) -> int:
    torch = require_torch_npu(args.device, Path(__file__).stem)
    torch.npu.set_device(args.device)

    _configure(
        args.dtype,
        args.offset,
        args.mode,
        args.tile_extent,
        inplace=args.inplace,
    )
    torch_of = {
        "f32": torch.float32,
        "f16": torch.float16,
        "bf16": torch.bfloat16,
    }
    torch_dtype = torch_of[args.dtype]
    sentinel = -7.0

    print(
        f"--- op={'inplace_add' if _INPLACE else 'add'} dtype={args.dtype} "
        f"mode={_MODE} offset={OFFSET} tile_extent={TILE_EXTENT} "
        f"trips={_trip_count()} n={N_ELE} ---"
    )
    torch.npu.manual_seed(0)
    x = torch.rand(N_ELE, dtype=torch_dtype, device="npu") * 10.0 - 5.0
    if _INPLACE:
        z = torch.rand(N_ELE, dtype=torch_dtype, device="npu") * 10.0 - 5.0
        expected = _expected_inplace_add(x, z)
        tla_x, tla_z = _wrap(x), _wrap(z)
        artifact = tla.compile(
            load_store_unalign_inplace_op,
            tla_x,
            tla_z,
            **runtime_kwargs(args),
        )
        artifact(tla_x, tla_z, block_num=args.block_num)
    else:
        y = torch.rand(N_ELE, dtype=torch_dtype, device="npu") * 10.0 - 5.0
        z = torch.full((N_ELE,), sentinel, dtype=torch_dtype, device="npu")
        expected = _expected_add(x, y, z)
        tla_x, tla_y, tla_z = _wrap(x), _wrap(y), _wrap(z)
        artifact = tla.compile(
            load_store_unalign_op,
            tla_x,
            tla_y,
            tla_z,
            **runtime_kwargs(args),
        )
        artifact(tla_x, tla_y, tla_z, block_num=args.block_num)
    torch.npu.synchronize()

    ok = bool(torch.equal(z, expected))
    mismatch = (z != expected).nonzero(as_tuple=False)
    first = None
    if mismatch.numel():
        i = int(mismatch[0].item())
        first = {"index": i, "actual": z[i].item(), "expected": expected[i].item()}
    print(f"compile_ok=True host=torch_npu op=add dtype={args.dtype}")
    print(f"kernel.o path={artifact.kernel_binary_path}")
    print("launch_ok=True")
    print(f"outputs equal expected add? {ok}")
    print(f"first mismatch={first}")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unaligned PostMode UPDATE load/store with auto Pre/Post"
    )
    parser.add_argument("op", nargs="?", default="add", choices=("add",))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--block-num", type=int, default=1)
    parser.add_argument("--dtype", choices=("f32", "f16", "bf16"), default="f32")
    parser.add_argument(
        "--offset",
        type=int,
        default=1,
        help="Element offset of the first window. 1 is not 32-byte aligned.",
    )
    parser.add_argument(
        "--tile-extent",
        type=int,
        default=None,
        help=(
            "UPDATE window length in elements (origin_shape / HIVM count). "
            "Default: dtype full-register width (f32=64, f16/bf16=128). "
            "Use 1 for one-element-per-trip."
        ),
    )
    parser.add_argument("--mode", choices=("update", "once"), default="update")
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="In-place z+=x with UnAlign load+store UPDATE on the same z tile",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Run f32/f16 × update/once × offset{0,1} × tile_extent{default,1} "
            "(dense); also one inplace case."
        ),
    )
    args = parser.parse_args()
    if args.offset < 0:
        raise SystemExit("--offset must be >= 0")
    if args.tile_extent is not None and args.tile_extent < 1:
        raise SystemExit("--tile-extent must be >= 1")

    if not args.all:
        return run(args)

    rc = 0
    for dtype in ("f32", "f16"):
        for mode in ("update", "once"):
            for offset in (0, 1):
                for tile_extent in (None, 1):
                    args.dtype = dtype
                    args.mode = mode
                    args.offset = offset
                    args.tile_extent = tile_extent
                    args.inplace = False
                    rc |= run(args)
    args.dtype = "f32"
    args.mode = "update"
    args.offset = 1
    args.tile_extent = None
    args.inplace = True
    rc |= run(args)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
