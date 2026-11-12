# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Device regressions for PACK4/UNPACK4 and existing FP8 SIMD casts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from catlass import tla
from catlass.params import (
    CastParams,
    LoadDist,
    NormalLoadParams,
    NormalStoreParams,
    RoundMode,
    SatMode,
    StoreDist,
)
from catlass.tla.runtime import from_dlpack, make_fake_tensor

GUARD = 32
RAW_CAPACITY = 320


def make_layout_kernel(
    operation, bits, count, *, explicit=False, sparse=False, registers=False
):
    source_lanes = (
        (256 if registers else count) if operation == "unpack" else 2048 // bits
    )

    @tla.kernel
    def transfer(source: tla.Tensor, destination: tla.Tensor):
        source_ptr = tla.allocate(
            RAW_CAPACITY, source.element_type, tla.AddressSpace.ub, 256
        )
        destination_ptr = tla.allocate(
            RAW_CAPACITY, tla.UInt8, tla.AddressSpace.ub, 256
        )
        source_buffer = tla.make_tensor_like(source_ptr, source, tla.arch.RowMajor)
        destination_buffer = tla.make_tensor_like(
            destination_ptr, destination, tla.arch.RowMajor
        )
        source_tile = tla.make_tensor(
            source_ptr + GUARD,
            tla.make_layout(tla.make_shape(source_lanes), tla.make_stride(1)),
        )
        destination_tile = tla.make_tensor(
            destination_ptr + GUARD,
            tla.make_layout(
                tla.make_shape(256 if operation == "unpack" else count),
                tla.make_stride(1),
            ),
        )
        word_tile = tla.make_tensor(
            tla.recast_ptr(destination_ptr + GUARD, dtype=tla.UInt32),
            tla.make_layout(tla.make_shape(64), tla.make_stride(1)),
        )
        ready = tla.flag("ready", tla.arch.MTE2, tla.arch.VECTOR)
        done = tla.flag("done", tla.arch.VECTOR, tla.arch.MTE3)
        with tla.vector():
            tla.copy(source_buffer, source)
            tla.copy(destination_buffer, destination)
            tla.set_flag(ready)
            tla.wait_flag(ready)
            with tla.vec.func(mode="simd"):
                if tla.const_expr(operation == "unpack"):
                    if tla.const_expr(registers):
                        words = tla.unpack(
                            tla.unpack(source_tile.load(), part=tla.RegPart.LOW),
                            part=tla.RegPart.LOW,
                        )
                        word_tile.store(words)
                    else:
                        destination_tile.store(
                            source_tile.load(
                                NormalLoadParams(load_dist=LoadDist.DIST_UNPACK4_B8)
                            )
                        )
                else:
                    values = source_tile.load()
                    if tla.const_expr(registers):
                        values = tla.pack(
                            tla.pack(values, part=tla.RegPart.LOW), part=tla.RegPart.LOW
                        )
                        destination_tile.store(values)
                    elif tla.const_expr(explicit):
                        selected, _ = tla.update_mask(
                            count * (32 // bits), dtype=source.element_type
                        )
                        if tla.const_expr(sparse):
                            selected = tla.bitwise_and(
                                selected,
                                tla.create_mask(dtype=tla.UInt8, pattern=tla.mask.M3),
                            )
                        destination_tile.store(
                            values,
                            NormalStoreParams(store_dist=StoreDist.DIST_PACK4_B32),
                            mask=selected,
                        )
                    else:
                        destination_tile.store(
                            values,
                            NormalStoreParams(store_dist=StoreDist.DIST_PACK4_B32),
                        )
            tla.set_flag(done)
            tla.wait_flag(done)
            tla.copy(destination, destination_buffer)
            tla.pipe_barrier(tla.pipes.ALL)

    return transfer


def make_fp8_kernel(
    dtype, count, *, narrow, saturate=False, stride=1, cast_params=None
):
    padded = (count + 63) // 64 * 64
    source_capacity = GUARD + padded * (stride if narrow else 1) + 64
    destination_capacity = GUARD + padded + 64
    cast = cast_params or CastParams(
        round_mode=RoundMode.CAST_RINT,
        sat_mode=SatMode.SAT if saturate else SatMode.NOSAT,
    )

    @tla.kernel
    def convert(source: tla.Tensor, destination: tla.Tensor):
        source_ptr = tla.allocate(
            source_capacity, source.element_type, tla.AddressSpace.ub, 256
        )
        destination_ptr = tla.allocate(
            destination_capacity, destination.element_type, tla.AddressSpace.ub, 256
        )
        source_buffer = tla.make_tensor_like(source_ptr, source, tla.arch.RowMajor)
        destination_buffer = tla.make_tensor_like(
            destination_ptr, destination, tla.arch.RowMajor
        )
        source_values = tla.make_tensor(
            source_ptr + GUARD,
            tla.make_layout(
                tla.make_shape(source_capacity - GUARD), tla.make_stride(1)
            ),
        )
        destination_values = tla.make_tensor(
            destination_ptr + GUARD,
            tla.make_layout(tla.make_shape(count), tla.make_stride(1)),
        )
        ready = tla.flag("ready", tla.arch.MTE2, tla.arch.VECTOR)
        done = tla.flag("done", tla.arch.VECTOR, tla.arch.MTE3)
        with tla.vector():
            tla.copy(source_buffer, source)
            tla.copy(destination_buffer, destination)
            tla.set_flag(ready)
            tla.wait_flag(ready)
            with tla.vec.func(mode="simd"):
                for packet in tla.range((count + 63) // 64):
                    output_tile = tla.tile_view(
                        destination_values, tla.make_shape(64), tla.make_coord(packet)
                    )
                    if tla.const_expr(narrow):
                        input_tile = tla.tile_view(
                            source_values,
                            tla.make_shape(64 * stride),
                            tla.make_coord(packet),
                        )
                        if tla.const_expr(stride == 2):
                            values, unused = input_tile.load(
                                NormalLoadParams(load_dist=LoadDist.DIST_DINTLV_B32)
                            )
                        else:
                            values = input_tile.load()
                        output_tile.store(
                            values.to(dtype, cast),
                            NormalStoreParams(store_dist=StoreDist.DIST_PACK4_B32),
                        )
                    else:
                        input_tile = tla.tile_view(
                            source_values, tla.make_shape(64), tla.make_coord(packet)
                        )
                        codes = input_tile.load(
                            NormalLoadParams(load_dist=LoadDist.DIST_UNPACK4_B8)
                        )
                        output_tile.store(codes.to(tla.Float32, cast))
            tla.set_flag(done)
            tla.wait_flag(done)
            tla.copy(destination, destination_buffer)
            tla.pipe_barrier(tla.pipes.ALL)

    return convert


def decode_fp8(code, exponent_bits):
    """Independent scalar codebook, including signed zero and nonfinite codes."""
    mantissa_bits = 7 - exponent_bits
    exponent, mantissa = divmod(code & 127, 1 << mantissa_bits)
    sign = -1 if code & 128 else 1
    if exponent == (1 << exponent_bits) - 1:
        if exponent_bits == 5:
            return sign * math.inf if mantissa == 0 else math.nan
        if mantissa == 7:
            return math.nan
    bias = (1 << (exponent_bits - 1)) - 1
    return (
        sign
        * ((1 if exponent else 0) + mantissa / (1 << mantissa_bits))
        * 2.0 ** (max(exponent, 1) - bias)
    )


def encode_fp8(value, exponent_bits, saturate):
    """Nearest-even codebook lookup; no device conversion algorithm is reused."""
    if math.isnan(value):
        return 0 if saturate else 127
    sign = 128 if math.copysign(1, value) < 0 else 0
    last = 126 if exponent_bits == 4 else 123
    maximum = decode_fp8(last, exponent_bits)
    magnitude = abs(value)
    if magnitude > maximum:
        if saturate:
            return sign | last
        midpoint = maximum + (maximum - decode_fp8(last - 1, exponent_bits)) / 2
        if magnitude > midpoint or (magnitude == midpoint and last & 1):
            return 127 if exponent_bits == 4 else sign | 124
    code = min(
        range(last + 1),
        key=lambda item: (abs(magnitude - decode_fp8(item, exponent_bits)), item & 1),
    )
    return sign | code


def boundary_values(exponent_bits):
    last = 126 if exponent_bits == 4 else 123
    finite = [decode_fp8(code, exponent_bits) for code in range(last + 1)]
    midpoints = [(a + b) / 2 for a, b in zip(finite, finite[1:])]
    midpoints.append(finite[-1] + (finite[-1] - finite[-2]) / 2)
    values = [0.0, -0.0, math.nan, math.inf, -math.inf, 2**-149, -(2**-149)]
    for midpoint in midpoints:
        middle = np.float32(midpoint)
        for value in (
            np.nextafter(middle, np.float32(-np.inf)),
            middle,
            np.nextafter(middle, np.float32(np.inf)),
        ):
            values.extend((value, -value))
    return np.asarray(values, np.float32)


def _assert_values(actual, expected):
    np.testing.assert_equal(actual, expected)
    if np.issubdtype(expected.dtype, np.floating):
        zeros = expected == 0
        np.testing.assert_equal(np.signbit(actual[zeros]), np.signbit(expected[zeros]))


def run(args):
    records = []
    if not args.compile_only:
        import torch
        import torch_npu  # noqa: F401 -- register the backend in the device entry

        torch.npu.set_device(args.device)

    def check(
        case_name,
        function,
        input_cases,
        input_type,
        output_type,
        *,
        decode=None,
        format_exponent_bits=4,
        decode_count=0,
    ):
        def fake(element_type, storage_size):
            return make_fake_tensor(
                element_type,
                (storage_size,),
                (1,),
                addrspace=tla.AddressSpace.gm,
                origin_shape=(storage_size,),
                layout_tag=tla.arch.RowMajor,
            )

        first_source, first_destination, _ = input_cases[0]
        type_args = (
            fake(input_type, first_source.size),
            fake(output_type, first_destination.size),
        )
        print(f"case={case_name}", flush=True)
        compiled = function.compile(type_args=type_args, options="--npu-arch 3510")
        record = dict(
            case=case_name, kernel=str(compiled.kernel_binary_path), launches=0
        )
        if decode is not None:
            decoder = decode.compile(
                type_args=(type_args[1], fake(tla.Float32, first_destination.size)),
                options="--npu-arch 3510",
            )
            record["decoder"] = str(decoder.kernel_binary_path)
        if not args.compile_only:
            source_device = torch.from_numpy(first_source.copy()).to(
                f"npu:{args.device}"
            )
            destination_device = torch.from_numpy(first_destination.copy()).to(
                f"npu:{args.device}"
            )
            source_view = from_dlpack(
                source_device, element_type=input_type, layout_tag=tla.arch.RowMajor
            )
            destination_view = from_dlpack(
                destination_device,
                element_type=output_type,
                layout_tag=tla.arch.RowMajor,
            )
            if decode is not None:
                restored_before = np.full(first_destination.size, -7.125, np.float32)
                restored_device = torch.from_numpy(restored_before.copy()).to(
                    f"npu:{args.device}"
                )
                restored_view = from_dlpack(
                    restored_device, layout_tag=tla.arch.RowMajor
                )
            for source_data, destination_before, expected_values in input_cases:
                source_device.copy_(torch.from_numpy(source_data))
                destination_device.copy_(torch.from_numpy(destination_before))
                if decode is not None:
                    restored_device.copy_(torch.from_numpy(restored_before))
                compiled(source_view, destination_view, block_num=1)
                if decode is not None:
                    decoder(destination_view, restored_view, block_num=1)
                torch.npu.synchronize(args.device)
                _assert_values(source_device.cpu().numpy(), source_data)
                _assert_values(destination_device.cpu().numpy(), expected_values)
                if decode is not None:
                    restored_expected = restored_before.copy()
                    restored_expected[GUARD : GUARD + decode_count] = [
                        decode_fp8(int(code), format_exponent_bits)
                        for code in expected_values[GUARD : GUARD + decode_count]
                    ]
                    _assert_values(restored_device.cpu().numpy(), restored_expected)
                record["launches"] += 2 if decode is not None else 1
        records.append(record)

    if args.case == "layout":
        settings = (
            ("pack", 8, 64, False, False, False),
            ("pack", 8, 13, True, True, False),
            ("pack", 16, 13, False, False, False),
            ("pack", 16, 13, True, False, False),
            ("pack", 32, 13, False, False, False),
            ("pack", 32, 13, False, False, True),
            ("unpack", 8, 13, False, False, False),
            ("unpack", 8, 13, False, False, True),
        )
        for operation, bits, count, explicit, sparse, registers in settings:
            numpy_dtype, source_dtype = {
                8: (np.uint8, tla.UInt8),
                16: (np.int16, tla.Int16),
                32: (np.int32, tla.Int32),
            }[bits]
            examples = []
            for repeat in range(2):
                source = (
                    (
                        (np.arange(RAW_CAPACITY * (bits // 8)) * 37 + 91 + repeat * 17)
                        % 256
                    )
                    .astype(np.uint8)
                    .view(numpy_dtype)
                )
                before = np.full(RAW_CAPACITY, 0xA5, np.uint8)
                expected = before.copy()
                if operation == "unpack":
                    expected[GUARD : GUARD + 256] = 0
                    expected[GUARD : GUARD + 256 : 4] = source.view(np.uint8)[
                        GUARD : GUARD + 64
                    ]
                else:
                    selected = np.arange(count)[::3] if sparse else np.arange(count)
                    expected[GUARD + selected] = source.view(np.uint8)[
                        GUARD * (bits // 8) + selected * 4
                    ]
                examples.append((source, before, expected))
            label = f"{operation}-b{bits}-n{count}-explicit{int(explicit)}-sparse{int(sparse)}-registers{int(registers)}"
            check(
                label,
                make_layout_kernel(
                    operation,
                    bits,
                    count,
                    explicit=explicit,
                    sparse=sparse,
                    registers=registers,
                ),
                examples,
                source_dtype,
                tla.UInt8,
            )
    else:
        dtype, exponent_bits = (
            (tla.Float8E4M3FN, 4) if args.case == "e4m3" else (tla.Float8E5M2, 5)
        )
        values = boundary_values(exponent_bits)
        count = values.size
        capacity = GUARD + ((count + 63) // 64) * 64 + 64
        for saturate in (False, True):
            examples = []
            for repeat in range(2):
                source = np.full(capacity, 37.0, np.float32)
                source[GUARD : GUARD + count] = np.roll(values, repeat)
                before = np.full(capacity, 0xA5, np.uint8)
                expected = before.copy()
                expected[GUARD : GUARD + count] = [
                    encode_fp8(float(value), exponent_bits, saturate)
                    for value in source[GUARD : GUARD + count]
                ]
                examples.append((source, before, expected))
            check(
                f"{args.case}-encode-sat{int(saturate)}",
                make_fp8_kernel(dtype, count, narrow=True, saturate=saturate),
                examples,
                tla.Float32,
                dtype,
                decode=make_fp8_kernel(dtype, count, narrow=False),
                format_exponent_bits=exponent_bits,
                decode_count=count,
            )
        for count in (256, 255, 1):
            capacity = GUARD + ((count + 63) // 64) * 64 + 64
            examples = []
            for repeat in range(2):
                source = np.full(capacity, 0xA5, np.uint8)
                source[GUARD : GUARD + count] = (
                    np.roll(np.arange(count, dtype=np.uint8), repeat)
                    if count > 1
                    else np.array([128], np.uint8)
                )
                before = np.full(capacity, -7.125, np.float32)
                expected = before.copy()
                expected[GUARD : GUARD + count] = [
                    decode_fp8(int(code), exponent_bits)
                    for code in source[GUARD : GUARD + count]
                ]
                examples.append((source, before, expected))
            check(
                f"{args.case}-decode-n{count}",
                make_fp8_kernel(dtype, count, narrow=False),
                examples,
                dtype,
                tla.Float32,
            )
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
    parser.add_argument("--case", choices=("layout", "e4m3", "e5m2"), required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--output", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
