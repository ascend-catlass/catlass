# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Native and FP32-composed FP4 primitive device regressions.

BF16 -> FP4 -> BF16 example (run this file with an installed DSL):
    python fp4_op.py --case roundtrip --format e1m2 --run --device 0
    python fp4_op.py --case roundtrip --format e2m1 --run --device 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from catlass import tla
from catlass.params import (
    CastParams,
    LoadDist,
    NormalLoadParams,
    NormalStoreParams,
    RegSlot,
    RoundMode,
    SatMode,
    StoreDist,
)
from catlass.tla.runtime import from_dlpack, make_fake_tensor
from _fp4_kernels import (
    make_native_kernel,
    make_slot_kernel,
    make_reference_data,
    make_composed_kernel,
    payload,
    quantize,
)
from _fp4_reference import decode_fp4
from fp8_pack4_op import GUARD, _assert_values

ROUNDING = {
    "nearest": RoundMode.CAST_ROUND,
    "even": RoundMode.CAST_RINT,
    "floor": RoundMode.CAST_FLOOR,
    "ceil": RoundMode.CAST_CEIL,
    "trunc": RoundMode.CAST_TRUNC,
}


def make_roundtrip_kernel(fp4_dtype):
    """One BF16 register -> packed E1M2/E2M1 -> one BF16 register."""

    @tla.kernel
    def bf16_fp4_roundtrip(
        source: tla.Tensor, packed: tla.Tensor, restored: tla.Tensor
    ):
        source_ptr = tla.allocate(192, tla.BFloat16, tla.AddressSpace.ub, 256)
        packed_ptr = tla.allocate(128, fp4_dtype, tla.AddressSpace.ub, 256)
        restored_ptr = tla.allocate(192, tla.BFloat16, tla.AddressSpace.ub, 256)
        source_buffer = tla.make_tensor_like(source_ptr, source, tla.arch.RowMajor)
        packed_buffer = tla.make_tensor_like(packed_ptr, packed, tla.arch.RowMajor)
        restored_buffer = tla.make_tensor_like(
            restored_ptr, restored, tla.arch.RowMajor
        )
        source_tile = tla.make_tensor(
            source_ptr + GUARD,
            tla.make_layout(tla.make_shape(128), tla.make_stride(1)),
        )
        packed_tile = tla.make_tensor(
            packed_ptr + GUARD,
            tla.make_layout(tla.make_shape(64), tla.make_stride(1)),
        )
        restored_tile = tla.make_tensor(
            restored_ptr + GUARD,
            tla.make_layout(tla.make_shape(128), tla.make_stride(1)),
        )
        loaded = tla.flag("loaded", tla.arch.MTE2, tla.arch.VECTOR)
        done = tla.flag("done", tla.arch.VECTOR, tla.arch.MTE3)
        with tla.vector():
            tla.copy(source_buffer, source)
            tla.copy(packed_buffer, packed)
            tla.copy(restored_buffer, restored)
            tla.set_flag(loaded)
            tla.wait_flag(loaded)
            with tla.vec.func(mode="simd"):
                bf16_values = source_tile.load()
                fp4_pairs = bf16_values.to(
                    fp4_dtype,
                    CastParams(
                        round_mode=RoundMode.CAST_RINT,
                        sat_mode=SatMode.SAT,
                        reg_slot=RegSlot.ZERO,
                    ),
                )
                bf16_restored = fp4_pairs.to(
                    tla.BFloat16, CastParams(reg_slot=RegSlot.ZERO)
                )
                # Slot zero has one pair byte in every four register bytes.
                packed_tile.store(
                    fp4_pairs,
                    NormalStoreParams(store_dist=StoreDist.DIST_PACK4_B32),
                )
                restored_tile.store(bf16_restored)
            tla.set_flag(done)
            tla.wait_flag(done)
            tla.copy(packed, packed_buffer)
            tla.copy(restored, restored_buffer)
            tla.pipe_barrier(tla.pipes.ALL)

    return bf16_fp4_roundtrip


def make_dintlv_kernel(dtype):
    @tla.kernel
    def split(source: tla.Tensor, first: tla.Tensor, second: tla.Tensor):
        source_ptr = tla.allocate(320, dtype, tla.AddressSpace.ub, 256)
        first_ptr = tla.allocate(224, dtype, tla.AddressSpace.ub, 256)
        second_ptr = tla.allocate(224, dtype, tla.AddressSpace.ub, 256)
        source_ub = tla.make_tensor_like(source_ptr, source, tla.arch.RowMajor)
        first_ub = tla.make_tensor_like(first_ptr, first, tla.arch.RowMajor)
        second_ub = tla.make_tensor_like(second_ptr, second, tla.arch.RowMajor)
        source_tile = tla.make_tensor(
            source_ptr + GUARD, tla.make_layout(tla.make_shape(256), tla.make_stride(1))
        )
        first_tile = tla.make_tensor(
            first_ptr + GUARD, tla.make_layout(tla.make_shape(127), tla.make_stride(1))
        )
        second_tile = tla.make_tensor(
            second_ptr + GUARD, tla.make_layout(tla.make_shape(127), tla.make_stride(1))
        )
        ready = tla.flag("ready", tla.arch.MTE2, tla.arch.VECTOR)
        done = tla.flag("done", tla.arch.VECTOR, tla.arch.MTE3)
        with tla.vector():
            tla.copy(source_ub, source)
            tla.copy(first_ub, first)
            tla.copy(second_ub, second)
            tla.set_flag(ready)
            tla.wait_flag(ready)
            with tla.vec.func(mode="simd"):
                even, odd = source_tile.load(
                    NormalLoadParams(load_dist=LoadDist.DIST_DINTLV_B16)
                )
                first_tile.store(even)
                second_tile.store(odd)
            tla.set_flag(done)
            tla.wait_flag(done)
            tla.copy(first, first_ub)
            tla.copy(second, second_ub)
            tla.pipe_barrier(tla.pipes.ALL)

    return split


def run(args):
    if args.run:
        import torch
        import torch_npu  # noqa: F401 -- register the backend

        torch.npu.set_device(args.device)
    records = []

    def check(label, kernels, dtypes, cases):
        def fake(dtype, size):
            return make_fake_tensor(
                dtype,
                (size,),
                (1,),
                addrspace=tla.AddressSpace.gm,
                origin_shape=(size,),
                layout_tag=tla.arch.RowMajor,
            )

        compiled = []
        for kernel, indices in kernels:
            input_types = []
            for index in indices:
                input_types.append(fake(dtypes[index], cases[0][0][index].size))
            compiled.append(
                (
                    kernel.compile(
                        type_args=tuple(input_types),
                        options="--npu-arch 3510",
                    ),
                    indices,
                )
            )
        record = dict(
            case=label,
            kernels=[str(k.kernel_binary_path) for k, _ in compiled],
            launches=0,
        )
        print(f"case={label}", flush=True)
        if args.run:

            def physical(array):
                return array.view(np.int16) if array.dtype == np.uint16 else array

            tensors = [
                torch.from_numpy(physical(a.copy())).to(f"npu:{args.device}")
                for a in cases[0][0]
            ]
            views = [
                from_dlpack(tensor, element_type=dtype, layout_tag=tla.arch.RowMajor)
                for tensor, dtype in zip(tensors, dtypes)
            ]
            for initial, expected in cases:
                for tensor, value in zip(tensors, initial):
                    tensor.copy_(torch.from_numpy(physical(value.copy())))
                for kernel, indices in compiled:
                    kernel(*(views[i] for i in indices), block_num=1)
                    record["launches"] += 1
                torch.npu.synchronize(args.device)
                for tensor, wanted in zip(tensors, expected):
                    actual = tensor.cpu().numpy().view(wanted.dtype)
                    _assert_values(actual, wanted)
        records.append(record)

    dtype = tla.Float4E2M1x2 if args.format == "e2m1" else tla.Float4E1M2x2
    book = [decode_fp4(i, 2 if args.format == "e2m1" else 1, 1) for i in range(16)]
    if args.case == "native":
        for direction in ("narrow", "widen"):
            for rounding in tuple(ROUNDING) if direction == "narrow" else ("nearest",):
                kernel = make_native_kernel(
                    dtype,
                    direction,
                    512,
                    608,
                    352,
                    ROUNDING[rounding],
                )
                source, initial, expected = make_reference_data(
                    args.format, direction, 512, 608, 352, rounding
                )
                cases = [
                    ([source.copy(), initial.copy()], [source.copy(), expected.copy()])
                ]
                changed = source.copy()
                wanted = expected.copy()
                if direction == "narrow":
                    changed[GUARD : GUARD + 512] = np.roll(
                        changed[GUARD : GUARD + 512], 1
                    )
                    values = (
                        changed[GUARD : GUARD + 512].astype(np.uint32) << 16
                    ).view(np.float32)
                    codes = np.asarray(
                        [quantize(float(x), book, rounding) for x in values], np.uint8
                    )
                    wanted[GUARD : GUARD + 256] = codes[::2] | codes[1::2] << 4
                    types = [tla.BFloat16, dtype]
                else:
                    changed[GUARD : GUARD + 256] = np.roll(
                        changed[GUARD : GUARD + 256], 1
                    )
                    codes = changed[GUARD : GUARD + 256]
                    values = np.empty(512, np.float32)
                    values[::2] = np.asarray(book, np.float32)[codes & 15]
                    values[1::2] = np.asarray(book, np.float32)[codes >> 4]
                    wanted[GUARD : GUARD + 512] = (values.view(np.uint32) >> 16).astype(
                        np.uint16
                    )
                    types = [dtype, tla.BFloat16]
                cases.append(
                    ([changed.copy(), initial.copy()], [changed.copy(), wanted])
                )
                check(
                    f"{args.format}-{direction}-{rounding}",
                    [(kernel, (0, 1))],
                    types,
                    cases,
                )
    elif args.case == "roundtrip":
        source, packed_initial, packed_expected = make_reference_data(
            args.format, "narrow", 128, 192, 128, "even"
        )
        restored_initial = np.full(192, 0xA5A5, np.uint16)
        cases = []
        for repeat in range(2):
            source_case = source.copy()
            source_case[GUARD : GUARD + 128] = np.roll(
                source[GUARD : GUARD + 128], repeat * 2
            )
            expected_pairs = packed_expected.copy()
            expected_pairs[GUARD : GUARD + 64] = np.roll(
                packed_expected[GUARD : GUARD + 64], repeat
            )
            pair_codes = expected_pairs[GUARD : GUARD + 64]
            decoded = np.empty(128, np.float32)
            decoded[::2] = np.asarray(book, np.float32)[pair_codes & 15]
            decoded[1::2] = np.asarray(book, np.float32)[pair_codes >> 4]
            expected_restored = restored_initial.copy()
            expected_restored[GUARD : GUARD + 128] = (
                decoded.view(np.uint32) >> 16
            ).astype(np.uint16)
            cases.append(
                (
                    [source_case, packed_initial.copy(), restored_initial.copy()],
                    [source_case.copy(), expected_pairs, expected_restored],
                )
            )
        check(
            f"{args.format}-bf16-fp4-roundtrip",
            [(make_roundtrip_kernel(dtype), (0, 1, 2))],
            [tla.BFloat16, dtype, tla.BFloat16],
            cases,
        )
    elif args.case == "composed":
        for direction in ("narrow", "widen"):
            for rounding in tuple(ROUNDING) if direction == "narrow" else ("even",):
                kernel, float_pitch, byte_pitch, float_capacity, byte_capacity = (
                    make_composed_kernel(
                        args.format, direction, 255, 2, 2, 1, 2.0, ROUNDING[rounding]
                    )
                )
                source, initial, expected = payload(
                    args.format,
                    direction,
                    255,
                    2,
                    2,
                    1,
                    2.0,
                    rounding,
                    float_pitch,
                    byte_pitch,
                    float_capacity,
                    byte_capacity,
                )
                types = (
                    [tla.Float32, tla.UInt8]
                    if direction == "narrow"
                    else [tla.UInt8, tla.Float32]
                )
                check(
                    f"{args.format}-fp32-{direction}-{rounding}",
                    [(kernel, (0, 1))],
                    types,
                    [
                        (
                            [source.copy(), initial.copy()],
                            [source.copy(), expected.copy()],
                        )
                        for _ in range(2)
                    ],
                )
    elif args.case == "slots":
        for slot in RegSlot:
            position = list(RegSlot).index(slot)
            for narrow in (True, False):
                active = 61 if position % 2 else 64
                cases = []
                for repeat in range(2):
                    if narrow:
                        values = np.roll(
                            np.resize(np.asarray(book, np.float32), 128), repeat
                        )
                        source = np.full(352, 0xA5A5, np.uint16)
                        source[GUARD : GUARD + 128] = (
                            values.view(np.uint32) >> 16
                        ).astype(np.uint16)
                        initial = np.full(352, 0xA5, np.uint8)
                        wanted = initial.copy()
                        codes = np.asarray(
                            [quantize(float(x), book, "nearest") for x in values],
                            np.uint8,
                        )
                        wanted[GUARD : GUARD + active * 4] = 0
                        wanted[GUARD + 4 * np.arange(active) + position] = (
                            codes[::2] | codes[1::2] << 4
                        )[:active]
                        types = [tla.BFloat16, dtype]
                    else:
                        source = np.full(352, 0xA5, np.uint8)
                        codes = np.roll(np.arange(64, dtype=np.uint8), repeat)
                        source[GUARD + 4 * np.arange(64) + position] = codes
                        initial = np.full(352, 0xA5A5, np.uint16)
                        wanted = initial.copy()
                        values = np.empty(128, np.float32)
                        values[::2] = np.asarray(book, np.float32)[codes & 15]
                        values[1::2] = np.asarray(book, np.float32)[codes >> 4]
                        wanted[GUARD : GUARD + active * 2] = (
                            values[: active * 2].view(np.uint32) >> 16
                        ).astype(np.uint16)
                        types = [dtype, tla.BFloat16]
                    cases.append(
                        ([source.copy(), initial.copy()], [source.copy(), wanted])
                    )
                check(
                    f"{args.format}-slot{position}-{'narrow' if narrow else 'widen'}",
                    [
                        (
                            make_slot_kernel(
                                slot,
                                narrow=narrow,
                                active=active,
                                pair_mask=narrow and position % 2 == 1,
                            ),
                            (0, 1),
                        )
                    ],
                    types,
                    cases,
                )
    elif args.case == "dintlv":
        for item_type in (tla.Float16, tla.BFloat16, tla.Int16, tla.UInt16):
            cases = []
            for repeat in range(2):
                source = np.arange(320, dtype=np.uint16) + repeat * 97
                first = np.full(224, 0xA5A5, np.uint16)
                second = first.copy()
                expected_first, expected_second = first.copy(), second.copy()
                expected_first[GUARD : GUARD + 127] = source[GUARD : GUARD + 254 : 2]
                expected_second[GUARD : GUARD + 127] = source[
                    GUARD + 1 : GUARD + 255 : 2
                ]
                cases.append(
                    (
                        [source.copy(), first, second],
                        [source.copy(), expected_first, expected_second],
                    )
                )
            check(
                f"dintlv-{item_type.__name__}",
                [(make_dintlv_kernel(item_type), (0, 1, 2))],
                [item_type] * 3,
                cases,
            )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=("native", "roundtrip", "composed", "slots", "dintlv"),
        required=True,
    )
    parser.add_argument("--format", choices=("e2m1", "e1m2"), default="e2m1")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    records = run(args)
    if args.report:
        args.report.write_text(json.dumps(records, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                passed=True,
                cases=len(records),
                launches=sum(x["launches"] for x in records),
            )
        )
    )


if __name__ == "__main__":
    main()
