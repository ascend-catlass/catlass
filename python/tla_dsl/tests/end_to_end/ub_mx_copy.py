# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Device regressions for the MX element types on the vector copy routes.

Each case goes GM -> UB, UB -> UB (row to row), UB -> GM, then compares the whole
destination buffer -- the untouched bytes included -- against a numpy reference.

The point of the fp4 cases is the element width. Packed fp4 is 64 elements per
32-byte block, not 32: a copy that stages fp4 as int8 moves twice the bytes it
should, and the slack left past the tile in both buffers turns that into a
detected byte mismatch instead of a silent overrun. The one-byte types are 32
per block, and the ragged N cases keep the tile a non-multiple of the block so
the round-up is exercised on both sides.

numpy buffers are counted in bytes, so every pitch the kernel states in elements
is converted here; for the one-byte types the two coincide.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from catlass import tla
from catlass.tla.runtime import from_dlpack, make_fake_tensor

# name -> (numpy dtype, element type, element bits, elements per 32-byte block)
DTYPES = {
    "fp8": (np.uint8, tla.Float8E4M3FN, 8, 32),
    "e8m0": (np.uint8, tla.Float8E8M0, 8, 32),
    "fp4": (np.uint8, tla.Float4E2M1, 4, 64),
}

ROWS = 32
SOURCE_SENTINEL = 0x5A
DESTINATION_SENTINEL = 0x3C
VALUE_MODULUS = 251

# (dtype, cols, padded): the tile is always ROWS x cols and its pitch rounds up
# to whole blocks. "padded" adds one more block to the destination pitch, which
# is the only way to leave a row gap and take the per-row copy path instead of
# the single contiguous run.
CASES = (
    ("fp8", 64, False),
    ("fp8", 40, False),
    ("fp8", 64, True),
    ("e8m0", 64, False),
    ("e8m0", 40, False),
    ("e8m0", 64, True),
    ("fp4", 128, False),
    ("fp4", 40, False),
    ("fp4", 128, True),
)


def round_up(value, multiple):
    return (value + multiple - 1) // multiple * multiple


def geometry(dtype, cols, padded):
    """Guard prefix and tile pitches, counted in elements of the copy's own type.

    The guard is one 32-byte block, so a tile start stays on a block boundary;
    for fp4 that is 64 elements, not 32.
    """
    _, _, bits, per_block = DTYPES[dtype]
    guard = 32 * 8 // bits
    src_pitch = round_up(cols, per_block)
    dst_pitch = src_pitch + (per_block if padded else 0)
    return guard, src_pitch, dst_pitch


def offset_ptr(base, elements, element_type, bits):
    """Advance a UB pointer by ``elements`` of the tile's own type.

    ``tla.ptr_add`` scales the offset by the pointee's byte width, which a
    sub-byte type has no integral value for, so fp4 steps in bytes and recasts
    back to the element type.
    """
    if bits < 8:
        as_bytes = tla.recast_ptr(base, dtype=tla.UInt8)
        return tla.recast_ptr(as_bytes + elements * bits // 8, dtype=element_type)
    return base + elements


def make_kernel(bits, element_type, guard, rows, cols, src_pitch, dst_pitch, storage):
    @tla.kernel
    def mx_copy(source: tla.Tensor, destination: tla.Tensor) -> None:
        source_ptr = tla.allocate(storage, source.element_type, tla.AddressSpace.ub, 256)
        destination_ptr = tla.allocate(
            storage, destination.element_type, tla.AddressSpace.ub, 256
        )
        source_buffer = tla.make_tensor_like(source_ptr, source, tla.arch.RowMajor)
        destination_buffer = tla.make_tensor_like(
            destination_ptr, destination, tla.arch.RowMajor
        )
        source_tile = tla.make_tensor(
            offset_ptr(source_ptr, guard, element_type, bits),
            tla.make_layout(
                tla.make_shape(rows, cols),
                tla.make_stride(src_pitch, 1),
                layoutTag=tla.arch.RowMajor,
            ),
        )
        destination_tile = tla.make_tensor(
            offset_ptr(destination_ptr, guard, element_type, bits),
            tla.make_layout(
                tla.make_shape(rows, cols),
                tla.make_stride(dst_pitch, 1),
                layoutTag=tla.arch.RowMajor,
            ),
        )
        ready = tla.flag("ready", tla.pipes.MTE2, tla.pipes.VECTOR)
        done = tla.flag("done", tla.pipes.VECTOR, tla.pipes.MTE3)
        with tla.vector():
            tla.copy(source_buffer, source)
            tla.copy(destination_buffer, destination)
            tla.set_flag(ready)
            tla.wait_flag(ready)
            tla.copy(destination_tile, source_tile)
            tla.set_flag(done)
            tla.wait_flag(done)
            tla.copy(destination, destination_buffer)
            tla.pipe_barrier(tla.pipes.ALL)

    return mx_copy


def build_case(np_dtype, bits, guard, rows, cols, src_pitch, dst_pitch, storage_bytes):
    """Source buffer, destination-before buffer and the copied expectation."""
    guard_b = guard * bits // 8
    cols_b = cols * bits // 8
    src_pitch_b = src_pitch * bits // 8
    dst_pitch_b = dst_pitch * bits // 8

    values = ((np.arange(rows * cols_b) * 7 + 3) % VALUE_MODULUS).astype(np.uint8)
    matrix = values.reshape(rows, cols_b)

    source = np.full(storage_bytes, SOURCE_SENTINEL, np_dtype)
    for row in range(rows):
        start = guard_b + row * src_pitch_b
        source[start : start + cols_b] = matrix[row]

    before = np.full(storage_bytes, DESTINATION_SENTINEL, np_dtype)
    # One row at a time: the copy moves whole blocks, so the tile's row padding
    # travels with it and the destination's own padding stays untouched.
    expected = before.copy()
    for row in range(rows):
        src_start = guard_b + row * src_pitch_b
        dst_start = guard_b + row * dst_pitch_b
        expected[dst_start : dst_start + src_pitch_b] = source[src_start : src_start + src_pitch_b]
    return source, before, expected


def run(args):
    records = []
    if not args.compile_only:
        import torch
        import torch_npu  # noqa: F401 -- registers the backend before set_device

        torch.npu.set_device(args.device)

    def fake(element_type, storage):
        return make_fake_tensor(
            element_type,
            (storage,),
            (1,),
            addrspace=tla.AddressSpace.gm,
            origin_shape=(storage,),
            layout_tag=tla.arch.RowMajor,
        )

    def check(dtype, cols, padded):
        np_dtype, element_type, bits, _ = DTYPES[dtype]
        guard, src_pitch, dst_pitch = geometry(dtype, cols, padded)
        span = ROWS * max(src_pitch, dst_pitch)
        # Twice the span: a copy that stages fp4 two elements per byte moves
        # twice the bytes, and it has to land inside the buffer to be caught as
        # a byte mismatch rather than run off the end.
        storage = guard + 2 * span + guard
        source, before, expected = build_case(
            np_dtype, bits, guard, ROWS, cols, src_pitch, dst_pitch, storage * bits // 8
        )
        case_name = (
            f"{dtype}-{ROWS}x{cols}-pitch{src_pitch}x{dst_pitch}"
            f"-{'padded' if padded else 'tight'}"
        )
        print(f"case={case_name}", flush=True)
        compiled = make_kernel(
            bits, element_type, guard, ROWS, cols, src_pitch, dst_pitch, storage
        ).compile(
            type_args=(fake(element_type, storage), fake(element_type, storage)),
            options="--npu-arch 3510",
        )
        record = dict(
            case=case_name,
            dtype=dtype,
            rows=ROWS,
            cols=cols,
            src_pitch=src_pitch,
            dst_pitch=dst_pitch,
            padded=padded,
            kernel=str(compiled.kernel_binary_path),
            launches=0,
        )
        if not args.compile_only:
            source_device = torch.from_numpy(source.copy()).to(f"npu:{args.device}")
            destination_device = torch.from_numpy(before.copy()).to(
                f"npu:{args.device}"
            )
            # origin_shape states the element count: a packed fp4 buffer can only
            # be exported as bytes, so the byte extent would otherwise stand in
            # for it.
            source_view = from_dlpack(
                source_device,
                element_type=element_type,
                layout_tag=tla.arch.RowMajor,
                origin_shape=(storage,),
            )
            destination_view = from_dlpack(
                destination_device,
                element_type=element_type,
                layout_tag=tla.arch.RowMajor,
                origin_shape=(storage,),
            )
            compiled(source_view, destination_view, block_num=1)
            torch.npu.synchronize(args.device)
            np.testing.assert_array_equal(
                destination_device.cpu().numpy().view(np_dtype), expected
            )
            np.testing.assert_array_equal(
                source_device.cpu().numpy().view(np_dtype), source
            )
            record["launches"] = 1
        records.append(record)

    selected = [case for case in CASES if args.case in ("all", case[0])]
    if not selected:
        raise SystemExit(f"no case for {args.case}")
    for dtype, cols, padded in selected:
        check(dtype, cols, padded)

    result = dict(
        case=args.case,
        status="compiled" if args.compile_only else "passed",
        records=records,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("all", "fp8", "e8m0", "fp4"), default="all")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--output", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
