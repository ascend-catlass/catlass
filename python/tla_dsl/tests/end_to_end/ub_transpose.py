# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Device regressions for the transposing UB -> UB copy on a single tile.

Each case moves one tile through GM -> UB, then UB -> UB with the destination
tagged ColumnMajor (VNCHWCONV), then UB -> GM, and compares the whole GM buffer
-- the untouched bytes included -- against a numpy reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from catlass import tla
from catlass.tla.runtime import from_dlpack, make_fake_tensor

GUARD = 32

# name -> (storage dtype, element type, element bytes, elements per 32-byte block, rows per tile)
DTYPES = {
    "u32": (np.uint32, tla.UInt32, 4, 8, 16),
    "u16": (np.uint16, tla.UInt16, 2, 16, 16),
    "u8": (np.uint8, tla.UInt8, 1, 32, 32),
    "fp8": (np.uint8, tla.Float8E4M3FN, 1, 32, 32),
}

# torch has no unsigned 32/16-bit dtype, so those buffers travel as the signed
# dtype of the same width and are viewed back before comparing.
FEED_DTYPE = {np.dtype(np.uint32): np.int32, np.dtype(np.uint16): np.int16}

SOURCE_SENTINEL = 0x5A
DESTINATION_SENTINEL = 0x3C
VALUE_MODULUS = 251

# (dtype, rows, cols, padded): "padded" widens both pitches by one extra block,
# which is the only non-tight footprint the block multiple leaves reachable.
CASES = (
    # u32: 8 elements per block, 16 rows per tile.
    ("u32", 16, 16, False),
    ("u32", 16, 24, False),
    ("u32", 48, 8, False),
    ("u32", 16, 16, True),
    # u16: 16 elements per block, 16 rows per tile.
    ("u16", 16, 16, False),
    ("u16", 32, 64, False),
    ("u16", 16, 16, True),
    # 1-byte elements: 32 elements per block, 32 rows per tile.
    ("u8", 32, 32, False),
    ("u8", 64, 64, False),
    ("u8", 32, 96, False),
    ("u8", 32, 32, True),
    ("fp8", 32, 32, False),
    ("fp8", 64, 64, False),
    ("fp8", 32, 96, False),
    ("fp8", 32, 32, True),
)

# The block multiple is a hard requirement of the copy, not of this script: a
# tile shorter than one block in either dimension is rejected by the frontend.
REJECTED = (
    ("u32", 24, 8),
    ("u16", 16, 24),
    ("u8", 48, 32),
)


def round_up(value, multiple):
    return (value + multiple - 1) // multiple * multiple


def geometry(dtype, rows, cols, padded):
    """Source pitch, destination pitch and the element span they need.

    The pitches round up to whole 32-byte blocks because a tight pitch would
    start every column on an unaligned address, and a padded case doubles them
    so both buffers stay sparse.
    """
    _, _, _, per_block, tile_rows = DTYPES[dtype]
    src_pitch = round_up(cols, per_block)
    dst_pitch = round_up(rows, tile_rows)
    if padded:
        src_pitch = round_up(src_pitch, 2 * per_block)
        dst_pitch = round_up(dst_pitch, 2 * tile_rows)
    src_span = (rows - 1) * src_pitch + cols
    dst_span = (cols - 1) * dst_pitch + rows
    return src_pitch, dst_pitch, max(src_span, dst_span)


def sentinel(dtype, byte_value):
    itemsize = np.dtype(dtype).itemsize
    return int.from_bytes(bytes([byte_value]) * itemsize, "little")


def make_kernel(dtype, rows, cols, src_pitch, dst_pitch, storage):
    @tla.kernel
    def transpose(source: tla.Tensor, destination: tla.Tensor) -> None:
        source_ptr = tla.allocate(storage, source.element_type, tla.AddressSpace.ub, 256)
        destination_ptr = tla.allocate(
            storage, destination.element_type, tla.AddressSpace.ub, 256
        )
        source_buffer = tla.make_tensor_like(source_ptr, source, tla.arch.RowMajor)
        destination_buffer = tla.make_tensor_like(
            destination_ptr, destination, tla.arch.RowMajor
        )
        source_tile = tla.make_tensor(
            source_ptr + GUARD,
            tla.make_layout(
                tla.make_shape(rows, cols),
                tla.make_stride(src_pitch, 1),
                layoutTag=tla.arch.RowMajor,
            ),
        )
        destination_tile = tla.make_tensor(
            destination_ptr + GUARD,
            tla.make_layout(
                tla.make_shape(rows, cols),
                tla.make_stride(1, dst_pitch),
                layoutTag=tla.arch.ColumnMajor,
            ),
        )
        ready = tla.flag("ready", tla.arch.MTE2, tla.arch.VECTOR)
        done = tla.flag("done", tla.arch.VECTOR, tla.arch.MTE3)
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

    return transpose


def build_case(dtype, rows, cols, src_pitch, dst_pitch, storage):
    """Source buffer, destination-before buffer and the transposed expectation."""
    np_dtype = DTYPES[dtype][0]
    values = ((np.arange(rows * cols) * 7 + 3) % VALUE_MODULUS).astype(np.int64)
    matrix = values.reshape(rows, cols)

    source = np.full(storage, sentinel(np_dtype, SOURCE_SENTINEL), np_dtype)
    for row in range(rows):
        start = GUARD + row * src_pitch
        source[start : start + cols] = matrix[row]

    before = np.full(storage, sentinel(np_dtype, DESTINATION_SENTINEL), np_dtype)
    expected = before.copy()
    for column in range(cols):
        start = GUARD + column * dst_pitch
        expected[start : start + rows] = matrix[:, column]
    return source, before, expected, np_dtype


def run(args):
    records = []
    torch = None
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

    def check(dtype, rows, cols, padded):
        _, element_type, _, _, _ = DTYPES[dtype]
        src_pitch, dst_pitch, span = geometry(dtype, rows, cols, padded)
        storage = GUARD + span + GUARD
        source, before, expected, np_dtype = build_case(
            dtype, rows, cols, src_pitch, dst_pitch, storage
        )
        case_name = (
            f"{dtype}-{rows}x{cols}-pitch{src_pitch}x{dst_pitch}"
            f"-{'padded' if padded else 'tight'}"
        )
        print(f"case={case_name}", flush=True)
        compiled = make_kernel(dtype, rows, cols, src_pitch, dst_pitch, storage).compile(
            type_args=(fake(element_type, storage), fake(element_type, storage)),
            options="--npu-arch 3510",
        )
        record = dict(
            case=case_name,
            dtype=dtype,
            rows=rows,
            cols=cols,
            src_pitch=src_pitch,
            dst_pitch=dst_pitch,
            padded=padded,
            kernel=str(compiled.kernel_binary_path),
            launches=0,
        )
        if not args.compile_only:
            feed = FEED_DTYPE.get(np.dtype(np_dtype), np_dtype)
            source_device = torch.from_numpy(source.view(feed).copy()).to(
                f"npu:{args.device}"
            )
            destination_device = torch.from_numpy(before.view(feed).copy()).to(
                f"npu:{args.device}"
            )
            source_view = from_dlpack(
                source_device, element_type=element_type, layout_tag=tla.arch.RowMajor
            )
            destination_view = from_dlpack(
                destination_device,
                element_type=element_type,
                layout_tag=tla.arch.RowMajor,
            )
            compiled(source_view, destination_view, block_num=1)
            torch.npu.synchronize(args.device)
            actual = destination_device.cpu().numpy().view(np_dtype)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(
                source_device.cpu().numpy().view(np_dtype), source
            )
            record["launches"] = 1
        records.append(record)

    def check_rejected(dtype, rows, cols):
        _, element_type, _, _, _ = DTYPES[dtype]
        src_pitch, dst_pitch, span = geometry(dtype, rows, cols, False)
        storage = GUARD + span + GUARD
        case_name = f"{dtype}-{rows}x{cols}-rejected"
        print(f"case={case_name}", flush=True)
        try:
            make_kernel(dtype, rows, cols, src_pitch, dst_pitch, storage).compile(
                type_args=(fake(element_type, storage), fake(element_type, storage)),
                options="--npu-arch 3510",
            )
        except Exception as error:  # noqa: BLE001 -- the frontend raises its own op error
            message = str(error)
            if "must be a multiple of" not in message:
                raise AssertionError(
                    f"{case_name}: rejected for the wrong reason: {message}"
                ) from error
        else:
            raise AssertionError(
                f"{case_name}: accepted a tile that is not a whole 32-byte block"
            )
        records.append(dict(case=case_name, rejected=True, launches=0))

    # --case rejected runs only the rejection cases, so it must not be held to
    # the positive-case selection below.
    run_cases = args.case != "rejected"
    selected = [case for case in CASES if args.case in ("all", case[0])]
    if run_cases and not selected:
        raise SystemExit(f"no case for {args.case}")
    for dtype, rows, cols, padded in selected:
        check(dtype, rows, cols, padded)
    if args.case in ("all", "rejected"):
        for dtype, rows, cols in REJECTED:
            check_rejected(dtype, rows, cols)

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
    parser.add_argument(
        "--case",
        choices=("all", "u32", "u16", "u8", "fp8", "rejected"),
        default="all",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--output", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
