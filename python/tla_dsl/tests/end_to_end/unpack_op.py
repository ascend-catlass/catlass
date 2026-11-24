# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end ``tla.unpack`` sign/zero extension on Ascend 950PR.

Each case loads one full 256-byte register of narrow integers, unpacks the
selected half with ``tla.unpack`` and stores the widened result. The reference
widens the same host values element-wise (signed cast for the signed paths,
masked zero extension for u32->u64), so boundary and signedness-critical
patterns compare bit-exactly.

The two register halves hold mirrored boundary patterns (forward order in the
low half, reverse order in the high half) so that swapping the LOW/HIGH
argument cannot go unnoticed.

Covered paths: i8->i16, i16->i32, i32->i64, u32->u64, each with LOW and HIGH.

Battery case: ``unpack``.
"""

from __future__ import annotations

import argparse

from catlass import tla


def _flat(ptr, n):
    return tla.make_tensor(ptr, tla.make_layout(tla.make_shape(n), tla.make_stride(1)))


def _cycle(values, lanes):
    """Repeat ``values`` in order until ``lanes`` elements are filled."""
    return (values * ((lanes + len(values) - 1) // len(values)))[:lanes]


# Per-path narrow values cycled to fill one register half. Each list holds the
# boundary / signedness-critical patterns from the design doc: max, zero,
# all-ones (=-1) and min.
_VALUE_LISTS = {
    "i8->i16": [127, 0, -1, -128],
    "i16->i32": [32767, 0, -1, -32768],
    "i32->i64": [2147483647, 0, -1, -2147483648],
    # u32->u64 rides on int32 carriers, so its all-ones and top-bit patterns
    # are spelled with their signed bit patterns (-1 and INT32_MIN).
    "u32->u64": [0x7FFFFFFF, 0, -1, -2147483648],
}


def make_unpack_kernel(src_storage, src_dtype, dst_storage, dst_dtype, src_lanes, dst_lanes, part):
    """Unpack ``part`` half of a full source register into a full wide register."""

    @tla.kernel
    def unpack_kernel(gm_in: tla.Tensor, gm_out: tla.Tensor) -> None:
        loaded = tla.flag("unpack_loaded", tla.arch.MTE2, tla.arch.VECTOR)
        computed = tla.flag("unpack_computed", tla.arch.VECTOR, tla.arch.MTE3)
        p_in = tla.allocate(src_lanes, src_storage, tla.AddressSpace.ub, 256)
        p_out = tla.allocate(dst_lanes, dst_storage, tla.AddressSpace.ub, 256)
        ub_in = _flat(p_in, src_lanes)
        ub_out = _flat(p_out, dst_lanes)
        with tla.vector():
            tla.copy(ub_in, _flat(gm_in.ptr, src_lanes))
            tla.set_flag(loaded)
            tla.wait_flag(loaded)
            with tla.vec.func(mode="simd"):
                src_reg = tla.tile_view(ub_in, tla.make_shape(src_lanes), tla.make_coord(0)).load()
                if src_dtype is not src_storage:
                    src_reg = src_reg.bitcast(src_dtype)
                wide = tla.unpack(src_reg, part=part)
                if dst_dtype is not dst_storage:
                    wide = wide.bitcast(dst_storage)
                tla.tile_view(ub_out, tla.make_shape(dst_lanes), tla.make_coord(0)).store(wide)
            tla.set_flag(computed)
            tla.wait_flag(computed)
            tla.copy(_flat(gm_out.ptr, dst_lanes), ub_out)
            tla.pipe_barrier(tla.pipes.ALL)

    return unpack_kernel


def run(args: argparse.Namespace) -> int:
    import torch
    import torch_npu  # noqa: F401

    from catlass.tla.runtime import from_dlpack

    torch.npu.set_device(args.device)

    def bind(buf):
        return from_dlpack(buf.contiguous(), layout_tag=tla.arch.RowMajor)

    _TORCH_DST = {tla.Int16: torch.int16, tla.Int32: torch.int32, tla.Int64: torch.int64}

    # (name, src storage dsl dtype, src dsl dtype, dst dsl dtype, dst storage
    #  dsl dtype, torch src dtype, value list). Unsigned register types ride on
    #  signed GM/UB carriers (torch has no uint32/uint64) via same-width
    #  bitcasts, mirroring the frontend compile test.
    cases = (
        ("unpack-i8-i16", tla.Int8, tla.Int8, tla.Int16, tla.Int16, torch.int8, _VALUE_LISTS["i8->i16"]),
        ("unpack-i16-i32", tla.Int16, tla.Int16, tla.Int32, tla.Int32, torch.int16, _VALUE_LISTS["i16->i32"]),
        ("unpack-i32-i64", tla.Int32, tla.Int32, tla.Int64, tla.Int64, torch.int32, _VALUE_LISTS["i32->i64"]),
        ("unpack-u32-u64", tla.Int32, tla.UInt32, tla.UInt64, tla.Int64, torch.int32, _VALUE_LISTS["u32->u64"]),
    )

    passed = 0
    for name, src_storage, src_dtype, dst_dtype, dst_storage, torch_src_dtype, values in cases:
        src_lanes = 2048 // src_dtype.width
        dst_lanes = 2048 // dst_dtype.width
        half_lanes = src_lanes // 2
        # Mirrored halves: a swapped LOW/HIGH selection reads a different
        # value sequence and fails the comparison below.
        src = torch.tensor(
            _cycle(values, half_lanes) + _cycle(values[::-1], half_lanes),
            dtype=torch_src_dtype,
        )
        for part in (tla.RegPart.LOW, tla.RegPart.HIGH):
            kernel = make_unpack_kernel(
                src_storage, src_dtype, dst_storage, dst_dtype, src_lanes, dst_lanes, part
            )
            in_buf = torch.zeros(src_lanes, dtype=torch_src_dtype).npu()
            in_buf.copy_(src)
            out = torch.zeros(dst_lanes, dtype=_TORCH_DST[dst_storage]).npu()
            to_in = bind(in_buf)
            to_out = bind(out)
            torch.npu.synchronize()
            artifact = tla.compile(kernel, to_in, to_out, options="--npu-arch 3510")
            artifact(to_in, to_out, block_num=args.block_num)
            torch.npu.synchronize()
            half = src[:half_lanes] if part is tla.RegPart.LOW else src[half_lanes:]
            if src_dtype is tla.UInt32:
                expected = (half.to(torch.int64) & 0xFFFFFFFF).to(_TORCH_DST[dst_storage])
            else:
                expected = half.to(_TORCH_DST[dst_storage])
            got = out.cpu()
            if not torch.equal(got, expected):
                raise AssertionError(
                    f"{name}/{part.value}: expected {expected[:8].tolist()}..., "
                    f"got {got[:8].tolist()}..."
                )
            passed += 1
            print(
                f"PASS case={name}/{part.value} lanes={dst_lanes} "
                f"head={got[:8].tolist()}"
            )

    print(f"tla.unpack sign/zero extension: {passed} passed, 0 failed, 0 skipped")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--block-num", type=int, default=1)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
