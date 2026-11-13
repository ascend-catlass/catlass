# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Device regression for ``tla.copy`` on packed fp4 tiles.

Two fp4 elements share a byte, so the copy routes stage fp4 as ``int8`` and
halve the extent -- which is exact only on whole bytes. fp4 carries MX
payloads, and the routes require a whole number of bytes: an odd row ends
mid-byte, and the other half of that byte is an element outside the tile,
which the copy can neither carry nor disturb.

This checks both halves of that. Even extents must copy byte for byte, in both
directions and both encodings, leaving the storage past the tile alone -- 34
and 40 are included because they are even without being a multiple of 32, so
they pin the rule to whole bytes rather than to MX scale groups. Odd extents
33 and 35 must be refused rather than silently mangled.

``0xAB`` in the source and ``0x55`` in the destination make a dropped or
clobbered nibble legible: a dropped tail reads as ``0x55`` and an overwritten
neighbour as ``0xAB``. ``--dynamic`` marks the layouts dynamic, so the extent
reaches the route as a runtime value -- which cannot be shown to be whole
groups and so must be refused too, the case a static check misses entirely.

A load is block-granular: it rounds the row up to 32 bytes and writes UB to
that boundary, for every element type and extent. The run prints how far it
reached rather than failing on it.

Needs an Ascend 950PR device, so pytest does not collect this directory --
see tests/conftest.py. Run it directly on a box that has one, passing a free
card: ``python mx_fp4_packed_extent_copy_op.py --device <id>``, adding
``--dynamic`` for the runtime-extent half and ``--format`` or ``--extent`` to
narrow. Exit status is the failure count. The refusals this file checks are
also covered without a device by tests/test_mx_fp4_copy_extents.py, which the
normal pytest run does collect; what needs the hardware is the byte-level
check that a legal extent copies exactly and leaves the storage past the tile
alone.
"""

from __future__ import annotations

import argparse
import sys

from catlass import tla
from catlass.tla.runtime import from_dlpack

FP4_PER_BYTE = 2  # two fp4 elements share a byte, hence the copy's unit
ROWS = 2  # more than one row, so each row is checked separately
PITCH = 128  # fp4 elements per row, i.e. 64 bytes -- an even byte pitch
PITCH_B = PITCH // 2
FILL_SRC = 0xAB
FILL_DST = 0x55

_FP4_DTYPE = {"e2m1": tla.Float4E2M1, "e1m2": tla.Float4E1M2}


def _int8_view(ptr, rows: int, cols: int):
    return tla.make_tensor(
        ptr,
        tla.make_layout(
            tla.make_shape(rows, cols),
            tla.make_stride(PITCH_B, 1),
            origin_shape=tla.make_shape(rows, cols),
            layoutTag=tla.arch.RowMajor,
        ),
    )


def _fp4_view(ptr, fp4_dtype, extent):
    return tla.make_tensor(
        tla.recast_ptr(ptr, dtype=fp4_dtype),
        tla.make_layout(
            tla.make_shape(ROWS, extent),
            tla.make_stride(PITCH, 1),
            origin_shape=tla.make_shape(ROWS, extent),
            layoutTag=tla.arch.RowMajor,
        ),
    )


def store_kernel(fp4_name: str):
    """UB -> GM: stage 0xAB in UB, then store an (ROWS, extent) fp4 tile."""
    fp4_dtype = _FP4_DTYPE[fp4_name]

    @tla.kernel
    def k(gm_fill, gm_out, gm_len):
        ub = tla.allocate(ROWS * PITCH_B, tla.Int8, tla.AddressSpace.ub, 512)
        # Three pipes again: the fill lands on MTE2, nothing computes, and the
        # store runs on MTE3, so the two hand-offs each need their own flag.
        filled = tla.flag("filled", tla.arch.MTE2, tla.arch.MTE3)
        with tla.vector():
            tla.copy(_int8_view(ub, ROWS, PITCH_B), gm_fill)
            tla.set_flag(filled)
            tla.wait_flag(filled)
            extent = gm_len.origin_shape[1]
            tla.copy(
                tla.tile_view(
                    gm_out, tla.make_shape(ROWS, extent), tla.make_coord(0, 0)
                ),
                _fp4_view(ub, fp4_dtype, extent),
            )

    return k


def load_kernel(fp4_name: str):
    """GM -> UB: prefill UB with 0x55, load the fp4 tile, dump UB back out."""
    fp4_dtype = _FP4_DTYPE[fp4_name]

    @tla.kernel
    def k(gm_fill, gm_src, gm_out, gm_len):
        ub = tla.allocate(ROWS * PITCH_B, tla.Int8, tla.AddressSpace.ub, 512)
        filled = tla.flag("filled", tla.arch.MTE2, tla.arch.MTE3)
        loaded = tla.flag("loaded", tla.arch.MTE2, tla.arch.MTE3)
        with tla.vector():
            tla.copy(_int8_view(ub, ROWS, PITCH_B), gm_fill)
            tla.set_flag(filled)
            tla.wait_flag(filled)
            # Both the prefill above and the fp4 load below write this buffer
            # on MTE2, and ``filled`` is an MTE2 -> MTE3 hand-off: waiting on
            # it holds MTE3, not the next MTE2 write. Without this barrier the
            # prefill can land on top of the bytes the copy under test just
            # brought in, and the script reports a stale 0x55 as a dropped
            # element -- a synchronisation bug wearing the costume of the fp4
            # bug this file exists to catch.
            tla.pipe_barrier(tla.arch.MTE2)
            extent = gm_len.origin_shape[1]
            tla.copy(
                _fp4_view(ub, fp4_dtype, extent),
                tla.tile_view(
                    gm_src, tla.make_shape(ROWS, extent), tla.make_coord(0, 0)
                ),
            )
            tla.set_flag(loaded)
            tla.wait_flag(loaded)
            tla.copy(gm_out, _int8_view(ub, ROWS, PITCH_B))

    return k


def _expected(extent: int) -> list[int]:
    """The destination bytes of one row after a copy of ``extent`` elements."""
    whole = extent // 2
    row = [FILL_SRC] * whole + [FILL_DST] * (PITCH_B - whole)
    if extent % 2:
        # The tail byte takes the source's low nibble and keeps its own high
        # one: the element past the tile must survive the copy.
        row[whole] = (FILL_SRC & 0x0F) | (FILL_DST & 0xF0)
    return row


def _bytes_of(t) -> list[list[int]]:
    flat = [b & 0xFF for b in t.detach().cpu().flatten().tolist()]
    return [flat[r * PITCH_B : (r + 1) * PITCH_B] for r in range(ROWS)]


def _tensor(torch, value: int):
    return torch.full(
        (ROWS, PITCH_B),
        value - 256 if value > 127 else value,
        dtype=torch.int8,
        device="npu",
    )


def _gm_fp4(t, fp4_dtype, dynamic: bool):
    """The fp4 parent: a full (ROWS, PITCH) row-major buffer.

    The tile under test is cut out of it with ``tile_view``, so the row pitch
    stays the parent's 128 elements -- declaring the tile extent here instead
    would make the pitch the extent and address the wrong bytes entirely.
    """
    view = from_dlpack(
        t,
        layout_tag=tla.arch.RowMajor,
        origin_shape=(ROWS, PITCH),
        element_type=fp4_dtype,
    )
    return view.mark_layout_dynamic() if dynamic else view


def _extent_tensor(torch, extent: int):
    """The buffer behind the extent argument, allocated by the caller.

    It is split from ``_extent_arg`` so that the caller holds the only
    reference that matters: the view handed to the kernel does not own the
    device memory, so the tensor has to outlive the launch. Returning both from
    one helper would leave the caller discarding one of them, which is the
    shape ``G.FNM.07`` reports.
    """
    return torch.zeros((1, extent), dtype=torch.int8, device="npu")


def _extent_arg(t, dynamic: bool):
    """A tensor passed only so the kernel can read the extent off its shape.

    Marked dynamic it reaches the kernel as a runtime value, which is how the
    extent gets past any frontend check -- exactly the case this regression is
    here for.
    """
    view = from_dlpack(
        t, layout_tag=tla.arch.RowMajor, origin_shape=(1, t.shape[1])
    )
    return view.mark_layout_dynamic() if dynamic else view


def _int8_arg(t, dynamic: bool):
    view = from_dlpack(t, layout_tag=tla.arch.RowMajor, origin_shape=(ROWS, PITCH_B))
    return view.mark_layout_dynamic() if dynamic else view


def check_store(fp4_name: str, extent: int, dynamic: bool, torch) -> list[str]:
    """A whole-group extent must copy exactly; anything else must be refused."""
    if extent % FP4_PER_BYTE or dynamic:
        return _expect_refusal("ub->gm", store_kernel, fp4_name, extent, dynamic, torch)
    fill = _tensor(torch, FILL_SRC)
    out = _tensor(torch, FILL_DST)
    extent_src = _extent_tensor(torch, extent)
    length = _extent_arg(extent_src, dynamic)
    tensors = (
        _int8_arg(fill, dynamic),
        _gm_fp4(out, _FP4_DTYPE[fp4_name], dynamic),
        length,
    )
    artifact = tla.compile(store_kernel(fp4_name), *tensors, options="--npu-arch 3510")
    artifact(*tensors, block_num=1)
    torch.npu.synchronize()
    return _diff(
        f"ub->gm {fp4_name} extent={extent}", _bytes_of(out), extent, strict=True
    )


def _expect_refusal(label, build, fp4_name, extent, dynamic, torch) -> list[str]:
    """Compiling this copy must fail, and fail on the extent."""
    fill = _tensor(torch, FILL_SRC)
    buf = _tensor(torch, FILL_DST)
    extent_src = _extent_tensor(torch, extent)
    length = _extent_arg(extent_src, dynamic)
    fp4 = _gm_fp4(buf, _FP4_DTYPE[fp4_name], dynamic)
    tensors = (
        (_int8_arg(fill, dynamic), fp4, length)
        if label == "ub->gm"
        else (_int8_arg(fill, dynamic), fp4, _int8_arg(buf, dynamic), length)
    )
    what = f"{label} {fp4_name} extent={extent}{' dynamic' if dynamic else ''}"
    try:
        tla.compile(build(fp4_name), *tensors, options="--npu-arch 3510")
    except Exception as exc:  # noqa: BLE001 -- any refusal is the point
        # Match the reason, not just the failure: a refusal for an unrelated
        # route or dtype problem would otherwise read as a pass.
        if "must be even" in str(exc):
            return []
        return [f"{what}: refused, but not for the extent: {str(exc).splitlines()[-1]}"]
    return [
        f"{what}: compiled, but an odd extent cannot be copied without "
        "dropping the last element or overwriting the first one past the tile"
    ]


def check_load(fp4_name: str, extent: int, dynamic: bool, torch) -> list[str]:
    """As check_store, in the other direction."""
    if extent % FP4_PER_BYTE or dynamic:
        return _expect_refusal("gm->ub", load_kernel, fp4_name, extent, dynamic, torch)
    fill = _tensor(torch, FILL_DST)  # UB starts at 0x55
    src = _tensor(torch, FILL_SRC)
    out = _tensor(torch, 0)
    extent_src = _extent_tensor(torch, extent)
    length = _extent_arg(extent_src, dynamic)
    tensors = (
        _int8_arg(fill, dynamic),
        _gm_fp4(src, _FP4_DTYPE[fp4_name], dynamic),
        _int8_arg(out, dynamic),
        length,
    )
    artifact = tla.compile(load_kernel(fp4_name), *tensors, options="--npu-arch 3510")
    artifact(*tensors, block_num=1)
    torch.npu.synchronize()
    return _diff(
        f"gm->ub {fp4_name} extent={extent}", _bytes_of(out), extent, strict=False
    )


def _diff(label: str, rows: list[list[int]], extent: int, strict: bool) -> list[str]:
    """Compare one destination row against what the copy owed it.

    ``strict`` asks for the bytes past the tile to be untouched as well. That
    holds for a GM destination, but not for a UB one: a load is block-granular
    -- the row is rounded up to 32 bytes and UB is written to that boundary,
    whatever the element type and whatever the extent -- so beyond the tile
    this reports how far the write reached instead of failing on it. Inside the
    tile both directions owe every byte, the tail nibble included.
    """
    want = _expected(extent)
    whole, tail = extent // 2, extent % 2
    fails = []
    for r, row in enumerate(rows):
        for i in range(whole + tail):
            got, exp = row[i], want[i]
            if got == exp:
                continue
            note = ""
            if tail and i == whole:
                note = (
                    "  <- the tail byte: 0x55 means the final element was "
                    "dropped, 0xab means the element past the tile was "
                    "overwritten, 0x50 means it arrived as a zero nibble"
                )
            fails.append(
                f"{label}: row {r} byte {i} is 0x{got:02x}, expected 0x{exp:02x}{note}"
            )
            break  # one report per row is enough to identify the failure
        past = [i for i in range(whole + tail, PITCH_B) if row[i] != FILL_DST]
        if not past:
            continue
        if strict:
            fails.append(
                f"{label}: row {r} byte {past[0]} is 0x{row[past[0]]:02x}, "
                f"expected 0x{FILL_DST:02x} -- the copy wrote "
                f"{len(past)} byte(s) past the tile"
            )
        elif r == 0:
            print(
                f"    (note: the load reached {len(past)} byte(s) past the "
                f"tile, to byte {past[-1]} -- block granularity)"
            )
    return fails


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--format", choices=sorted(_FP4_DTYPE), action="append")
    ap.add_argument("--extent", type=int, action="append")
    ap.add_argument(
        "--dynamic",
        action="store_true",
        help="mark the layouts dynamic, so the extent is a runtime value",
    )
    ap.add_argument("--device", type=int, default=0, help="NPU card to run on")
    args = ap.parse_args()
    formats = args.format or sorted(_FP4_DTYPE)
    extents = args.extent or [32, 34, 40, 64, 33, 35]

    import torch
    import torch_npu  # noqa: F401 -- registers the backend in the device entry

    torch.npu.set_device(args.device)

    fails: list[str] = []
    for fp4_name in formats:
        for extent in extents:
            for name, fn in (("ub->gm", check_store), ("gm->ub", check_load)):
                got = fn(fp4_name, extent, args.dynamic, torch)
                tag = "dynamic" if args.dynamic else "static "
                print(
                    f"{name} {fp4_name} {tag} extent={extent:3d} "
                    f"{'FAIL' if got else 'ok'}"
                )
                fails += got

    for f in fails:
        print(f"  FAIL: {f}", file=sys.stderr)
    print(f"\n{len(fails)} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
