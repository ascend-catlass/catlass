# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Test-only native and composed FP4 conversion kernels."""

import math
from itertools import pairwise

from catlass import tla
from catlass.params import (
    CastParams,
    LoadDist,
    NormalLoadParams,
    NormalStoreParams,
    SatMode,
    StoreDist,
)
from _fp4_reference import decode_fp4, nearest_code
from _fp4_compositions import fp32_to_fp4, fp4_to_fp32, merge_owned_nibbles

GUARD = 32


def make_native_kernel(
    fp4_dtype,
    direction,
    element_count,
    compute_capacity,
    byte_capacity,
    rounding,
):
    compute_type = tla.BFloat16
    values_per_register = 128
    byte_count = element_count // 2
    cast_params = CastParams(sat_mode=SatMode.SAT, round_mode=rounding)
    load_params = NormalLoadParams(load_dist=LoadDist.DIST_UNPACK4_B8)
    store_params = NormalStoreParams(store_dist=StoreDist.DIST_PACK4_B32)

    @tla.kernel
    def convert(source: tla.Tensor, dest: tla.Tensor):
        loaded = tla.flag("loaded", tla.arch.MTE2, tla.arch.VECTOR)
        done = tla.flag("done", tla.arch.VECTOR, tla.arch.MTE3)
        if tla.const_expr(direction == "narrow"):
            source_ub_ptr = tla.allocate(
                compute_capacity, compute_type, tla.AddressSpace.ub, 256
            )
            destination_ub_ptr = tla.allocate(
                byte_capacity, fp4_dtype, tla.AddressSpace.ub, 256
            )
            compute_ub_ptr = source_ub_ptr
            packed_ub_ptr = destination_ub_ptr
        else:
            source_ub_ptr = tla.allocate(
                byte_capacity, fp4_dtype, tla.AddressSpace.ub, 256
            )
            destination_ub_ptr = tla.allocate(
                compute_capacity, compute_type, tla.AddressSpace.ub, 256
            )
            compute_ub_ptr = destination_ub_ptr
            packed_ub_ptr = source_ub_ptr
        source_ub = tla.make_tensor_like(source_ub_ptr, source, tla.arch.RowMajor)
        destination_ub = tla.make_tensor_like(
            destination_ub_ptr, dest, tla.arch.RowMajor
        )
        computation = tla.make_tensor(
            compute_ub_ptr + GUARD,
            tla.make_layout(tla.make_shape(element_count), tla.make_stride(1)),
        )
        storage = tla.make_tensor(
            packed_ub_ptr + GUARD,
            tla.make_layout(tla.make_shape(byte_count), tla.make_stride(1)),
        )
        with tla.vector():
            tla.copy(source_ub, source)
            tla.copy(destination_ub, dest)
            tla.set_flag(loaded)
            tla.wait_flag(loaded)
            with tla.vec.func(mode="simd"):
                full_bytes = tla.create_mask(dtype=tla.UInt8, pattern=tla.mask.ALL)
                full_values = tla.create_mask(dtype=compute_type, pattern=tla.mask.ALL)
                for packet in tla.range(
                    (element_count + values_per_register - 1) // values_per_register
                ):
                    values = tla.tile_view(
                        computation,
                        tla.make_shape(values_per_register),
                        tla.make_coord(packet),
                    )
                    packed_tile = tla.tile_view(
                        storage, tla.make_shape(64), tla.make_coord(packet)
                    )
                    if tla.const_expr(direction == "narrow"):
                        encoded = (
                            values.load().to(fp4_dtype, cast_params).bitcast(tla.UInt8)
                        )
                        if tla.const_expr(element_count % values_per_register == 0):
                            packed_tile.store(encoded, store_params, mask=full_bytes)
                        else:
                            packed_tile.store(encoded, store_params)
                    else:
                        encoded = packed_tile.load(load_params).bitcast(fp4_dtype)
                        decoded = encoded.to(compute_type, cast_params)
                        if tla.const_expr(element_count % values_per_register == 0):
                            values.store(decoded, mask=full_values)
                        else:
                            values.store(decoded)
            tla.set_flag(done)
            tla.wait_flag(done)
            tla.copy(dest, destination_ub)
            tla.pipe_barrier(tla.pipes.ALL)

    return convert


def quantize(value, codebook, rounding):
    if math.isnan(value):
        return 0
    finite = [(i, x) for i, x in enumerate(codebook) if math.isfinite(x)]
    limit = max((x for _, x in finite))
    value = max(-limit, min(limit, value))
    if rounding in ("nearest", "even"):
        return nearest_code(value, codebook, ties_away=rounding == "nearest")
    lower = rounding == "floor" or (rounding == "trunc" and value >= 0)
    candidates = (
        [(i, x) for i, x in finite if x <= value]
        if lower
        else [(i, x) for i, x in finite if x >= value]
    )
    boundary = (max if lower else min)((x for _, x in candidates))
    candidates = [i for i, x in candidates if x == boundary]
    return next(
        (
            i
            for i in candidates
            if math.copysign(1, codebook[i]) == math.copysign(1, value)
        ),
        candidates[0],
    )


def make_reference_data(
    fp4_format,
    direction,
    element_count,
    compute_capacity,
    byte_capacity,
    rounding,
):
    import numpy as np

    codebook = [decode_fp4(i, 2 if fp4_format == "e2m1" else 1, 1) for i in range(16)]
    positions = GUARD + np.arange(element_count)
    byte_count = element_count // 2
    if direction == "widen":
        source = np.full(byte_capacity, 165, np.uint8)
        codes = (np.arange(byte_count) % 256).astype(np.uint8)
        source[GUARD : GUARD + byte_count] = codes
        values = np.empty(element_count, np.float32)
        values[::2] = np.asarray(codebook, np.float32)[codes & 15]
        values[1::2] = np.asarray(codebook, np.float32)[codes >> 4]
        before = np.full(compute_capacity, 42405, np.uint16)
        expected = before.copy()
        expected[positions] = (values.view(np.uint32) >> 16).astype(np.uint16)
        return (source, before, expected)
    positive = sorted({v for v in codebook if math.isfinite(v) and v >= 0})
    samples = [
        0.0,
        -0.0,
        math.nan,
        math.inf,
        -math.inf,
        positive[-1] * 2,
        -positive[-1] * 2,
    ]
    for low, high in pairwise(positive):
        midpoint = np.float32((low + high) / 2)
        bits = int(np.asarray([midpoint], np.float32).view(np.uint32)[0] >> 16)
        neighbors = (np.asarray([bits - 1, bits, bits + 1], np.uint32) << 16).view(
            np.float32
        )
        for value in neighbors:
            samples.extend((value, -value))
    values = np.resize(np.asarray(samples, np.float32), element_count)
    bits = (values.view(np.uint32) >> 16).astype(np.uint16)
    values = (bits.astype(np.uint32) << 16).view(np.float32)
    source = np.full(compute_capacity, 16916, np.uint16)
    source[positions] = bits
    codes = np.asarray(
        [quantize(float(v), codebook, rounding) for v in values], np.uint8
    )
    codes = codes[::2] | codes[1::2] << 4
    before = np.full(byte_capacity, 165, np.uint8)
    expected = before.copy()
    expected[GUARD : GUARD + byte_count] = codes
    return (source, before, expected)


def make_composed_kernel(
    format_name,
    direction,
    element_count,
    row_count,
    input_stride,
    nibble_offset,
    scale,
    rounding_mode,
):
    dtype = tla.Float4E2M1x2 if format_name == "e2m1" else tla.Float4E1M2x2
    packets = (element_count + nibble_offset + 127) // 128
    float_pitch = packets * 128 * input_stride + 64
    byte_pitch = packets * 64 + 64
    float_capacity = GUARD + row_count * float_pitch + 64
    byte_capacity = GUARD + row_count * byte_pitch + 64
    inv_scale = 1.0 / scale
    params = CastParams(sat_mode=SatMode.SAT, round_mode=rounding_mode)
    unpack = NormalLoadParams(load_dist=LoadDist.DIST_UNPACK4_B8)
    pack = NormalStoreParams(store_dist=StoreDist.DIST_PACK4_B32)

    @tla.kernel
    def composed_fp4(source: tla.Tensor, destination: tla.Tensor):
        float_ub_ptr = tla.allocate(
            float_capacity, tla.Float32, tla.AddressSpace.ub, 256
        )
        packed_ub_ptr = tla.allocate(byte_capacity, tla.UInt8, tla.AddressSpace.ub, 256)
        if tla.const_expr(direction == "narrow"):
            source_ub = tla.make_tensor_like(float_ub_ptr, source, tla.arch.RowMajor)
            destination_ub = tla.make_tensor_like(
                packed_ub_ptr, destination, tla.arch.RowMajor
            )
        else:
            source_ub = tla.make_tensor_like(packed_ub_ptr, source, tla.arch.RowMajor)
            destination_ub = tla.make_tensor_like(
                float_ub_ptr, destination, tla.arch.RowMajor
            )
        floats = tla.make_tensor(
            float_ub_ptr + GUARD,
            tla.make_layout(
                tla.make_shape(row_count * float_pitch), tla.make_stride(1)
            ),
        )
        payloads = tla.make_tensor(
            packed_ub_ptr + GUARD,
            tla.make_layout(tla.make_shape(row_count * byte_pitch), tla.make_stride(1)),
        )
        loaded = tla.flag("loaded", tla.arch.MTE2, tla.arch.VECTOR)
        done = tla.flag("done", tla.arch.VECTOR, tla.arch.MTE3)
        with tla.vector():
            tla.copy(source_ub, source)
            tla.copy(destination_ub, destination)
            tla.set_flag(loaded)
            tla.wait_flag(loaded)
            with tla.vec.func(mode="simd"):
                lanes = tla.arange(dtype=tla.Int32)
                all_bytes = tla.create_mask(dtype=tla.UInt8, pattern=tla.mask.ALL)
                all_values = tla.create_mask(dtype=tla.Float32, pattern=tla.mask.ALL)
                for row in tla.range(row_count):
                    values = tla.tile_view(
                        floats, tla.make_shape(float_pitch), tla.make_coord(row)
                    )
                    bytes_row = tla.tile_view(
                        payloads, tla.make_shape(byte_pitch), tla.make_coord(row)
                    )
                    for packet in tla.range(packets):
                        payload_tile = tla.tile_view(
                            bytes_row, tla.make_shape(64), tla.make_coord(packet)
                        )
                        if tla.const_expr(direction == "narrow"):
                            first_tile = tla.tile_view(
                                values,
                                tla.make_shape(64 * input_stride),
                                tla.make_coord(packet * 2),
                            )
                            second_tile = tla.tile_view(
                                values,
                                tla.make_shape(64 * input_stride),
                                tla.make_coord(packet * 2 + 1),
                            )
                            if tla.const_expr(input_stride == 2):
                                first_register, unused_first = first_tile.load(
                                    NormalLoadParams(load_dist=LoadDist.DIST_DINTLV_B32)
                                )
                                second_register, unused_second = second_tile.load(
                                    NormalLoadParams(load_dist=LoadDist.DIST_DINTLV_B32)
                                )
                            else:
                                first_register = first_tile.load()
                                second_register = second_tile.load()
                            if tla.const_expr(scale != 1.0):
                                first_register = first_register * inv_scale
                                second_register = second_register * inv_scale
                            encoded = fp32_to_fp4(
                                first_register, second_register, dtype, params
                            ).bitcast(tla.UInt8)
                            if tla.const_expr(
                                nibble_offset == 0 and element_count % 128 == 0
                            ):
                                payload_tile.store(encoded, pack, mask=all_bytes)
                            elif (
                                packet * 128 >= nibble_offset
                                and (packet + 1) * 128 <= element_count + nibble_offset
                            ):
                                payload_tile.store(encoded, pack, mask=all_bytes)
                            else:
                                first_index = lanes * 2 + packet * 128 - nibble_offset
                                merged = merge_owned_nibbles(
                                    encoded,
                                    payload_tile.load(unpack),
                                    first_index,
                                    element_count,
                                )
                                payload_tile.store(merged, pack, mask=all_bytes)
                        else:
                            first_register, second_register = fp4_to_fp32(
                                payload_tile.load(unpack).bitcast(dtype)
                            )
                            if tla.const_expr(scale != 1.0):
                                first_register = first_register * scale
                                second_register = second_register * scale
                            y0 = tla.tile_view(
                                values, tla.make_shape(64), tla.make_coord(packet * 2)
                            )
                            y1 = tla.tile_view(
                                values,
                                tla.make_shape(64),
                                tla.make_coord(packet * 2 + 1),
                            )
                            if tla.const_expr(element_count % 128 == 0):
                                y0.store(first_register, mask=all_values)
                                y1.store(second_register, mask=all_values)
                            else:
                                y0.store(
                                    first_register,
                                    mask=tla.cmp(
                                        lanes + packet * 128, element_count, "lt"
                                    ),
                                )
                                y1.store(
                                    second_register,
                                    mask=tla.cmp(
                                        lanes + packet * 128 + 64, element_count, "lt"
                                    ),
                                )
            tla.set_flag(done)
            tla.wait_flag(done)
            tla.copy(destination, destination_ub)
            tla.pipe_barrier(tla.pipes.ALL)

    return (composed_fp4, float_pitch, byte_pitch, float_capacity, byte_capacity)


def payload(
    format_name,
    direction,
    element_count,
    row_count,
    input_stride,
    nibble_offset,
    scale,
    rounding,
    float_pitch,
    byte_pitch,
    float_capacity,
    byte_capacity,
):
    import numpy as np

    book = [decode_fp4(i, 2 if format_name == "e2m1" else 1, 1) for i in range(16)]
    if direction == "widen":
        source = np.full(byte_capacity, 165, np.uint8)
        before = np.full(float_capacity, -7.125, np.float32)
        expected = before.copy()
        for row in range(row_count):
            codes = ((np.arange((element_count + 1) // 2) + row * 31) % 256).astype(
                np.uint8
            )
            source[GUARD + row * byte_pitch : GUARD + row * byte_pitch + len(codes)] = (
                codes
            )
            unpacked = np.empty(len(codes) * 2, np.float32)
            unpacked[::2] = np.array(book, np.float32)[codes & 15]
            unpacked[1::2] = np.array(book, np.float32)[codes >> 4]
            expected[
                GUARD + row * float_pitch : GUARD + row * float_pitch + element_count
            ] = unpacked[:element_count] * np.float32(scale)
        return (source, before, expected)
    positive = sorted({sample_value for sample_value in book if sample_value >= 0})
    boundaries = positive + [(lower + upper) / 2 for lower, upper in pairwise(positive)]
    samples = [0.0, -0.0, math.inf, -math.inf, math.nan, 10000000000.0, -10000000000.0]
    for value in boundaries:
        value = np.float32(value)
        for sample_value in (
            np.nextafter(value, np.float32(-np.inf)),
            value,
            np.nextafter(value, np.float32(np.inf)),
        ):
            samples.extend((sample_value, -sample_value))
    source = np.full(float_capacity, 37.0, np.float32)
    before = np.full(byte_capacity, 165, np.uint8)
    expected = before.copy()
    for row in range(row_count):
        values = np.roll(
            np.resize(np.asarray(samples, np.float32), element_count), row * 13
        )
        source[
            GUARD
            + row * float_pitch
            + (np.arange(element_count) + nibble_offset) * input_stride
        ] = values
        with np.errstate(over="ignore", invalid="ignore"):
            scaled = values * np.float32(1.0 / scale) if scale != 1.0 else values
        for element_index, sample_value in enumerate(scaled):
            code = quantize(float(sample_value), book, rounding)
            byte_index = GUARD + row * byte_pitch + (element_index + nibble_offset) // 2
            shift = 4 * ((element_index + nibble_offset) % 2)
            expected[byte_index] = (
                int(expected[byte_index]) & ~(15 << shift) | code << shift
            )
    return (source, before, expected)


def make_slot_kernel(
    slot,
    *,
    narrow,
    active,
    pair_mask=False,
):
    @tla.kernel
    def convert(source: tla.Tensor, destination: tla.Tensor):
        source_ptr = tla.allocate(352, source.element_type, tla.AddressSpace.ub, 256)
        destination_ptr = tla.allocate(
            352, destination.element_type, tla.AddressSpace.ub, 256
        )
        source_buffer = tla.make_tensor_like(source_ptr, source, tla.arch.RowMajor)
        destination_buffer = tla.make_tensor_like(
            destination_ptr, destination, tla.arch.RowMajor
        )
        source_tile = tla.make_tensor(
            source_ptr + GUARD,
            tla.make_layout(
                tla.make_shape(128 if narrow else 256),
                tla.make_stride(1),
            ),
        )
        destination_tile = tla.make_tensor(
            destination_ptr + GUARD,
            tla.make_layout(
                tla.make_shape(active * 4 if narrow else active * 2),
                tla.make_stride(1),
            ),
        )
        ready = tla.flag("ready", tla.arch.MTE2, tla.arch.VECTOR)
        done = tla.flag("done", tla.arch.VECTOR, tla.arch.MTE3)
        with tla.vector():
            tla.copy(source_buffer, source)
            tla.copy(destination_buffer, destination)
            tla.set_flag(ready)
            tla.wait_flag(ready)
            with tla.vec.func(mode="simd"):
                if tla.const_expr(narrow and pair_mask):
                    predicate, _ = tla.update_mask(active, dtype=tla.UInt32)
                else:
                    predicate, _ = tla.update_mask(
                        active * 2 if narrow else active * 4,
                        dtype=source.element_type,
                    )
                values = source_tile.load().to(
                    destination.element_type,
                    CastParams(
                        reg_slot=slot,
                        sat_mode=SatMode.SAT,
                    ),
                    predicate,
                )
                if tla.const_expr(narrow):
                    values = values.bitcast(tla.UInt8).bitcast(destination.element_type)
                destination_tile.store(values)
            tla.set_flag(done)
            tla.wait_flag(done)
            tla.copy(destination, destination_buffer)
            tla.pipe_barrier(tla.pipes.ALL)

    return convert
