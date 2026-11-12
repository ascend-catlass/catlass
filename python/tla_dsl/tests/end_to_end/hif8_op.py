# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""HiF8 SIMD conversion and typed storage device regressions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from catlass import tla
from catlass.params import CastParams, RegSlot, RoundMode, SatMode
from catlass.tla.runtime import from_dlpack, make_fake_tensor
from _lowp_reference import (
    boundary_samples,
    decode_hif8,
    encode_hif8_hybrid,
    encode_hif8_rna,
    hif8_hybrid_samples,
)
from fp8_pack4_op import GUARD, _assert_values, make_fp8_kernel


def make_slot_kernel(
    slot,
    *,
    narrow,
    active,
    values_per_byte=1,
    round_mode=RoundMode.CAST_ROUND,
    saturate=False,
    sparse=False,
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
                tla.make_shape(64 * values_per_byte if narrow else 256),
                tla.make_stride(1),
            ),
        )
        destination_tile = tla.make_tensor(
            destination_ptr + GUARD,
            tla.make_layout(
                tla.make_shape(
                    256
                    if sparse
                    else active * 4
                    if narrow
                    else active * values_per_byte
                ),
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
                predicate, _ = tla.update_mask(
                    active * values_per_byte if narrow else active * 4,
                    dtype=source.element_type,
                )
                if tla.const_expr(sparse):
                    predicate = tla.bitwise_and(
                        predicate,
                        tla.create_mask(dtype=source.element_type, pattern=tla.mask.M3),
                    )
                values = source_tile.load().to(
                    destination.element_type,
                    CastParams(
                        reg_slot=slot,
                        sat_mode=SatMode.SAT
                        if saturate or values_per_byte == 2
                        else SatMode.NOSAT,
                        round_mode=round_mode,
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


def run(args):
    if args.run:
        import torch
        import torch_npu  # noqa: F401 -- register the device backend

        torch.npu.set_device(args.device)
    records = []

    def check(label, kernel, cases, source_dtype, destination_dtype):
        source_data, initial, _, _ = cases[0]

        def fake(dtype, size):
            return make_fake_tensor(
                dtype,
                (size,),
                (1,),
                addrspace=tla.AddressSpace.gm,
                origin_shape=(size,),
                layout_tag=tla.arch.RowMajor,
            )

        compiled = kernel.compile(
            type_args=(
                fake(source_dtype, source_data.size),
                fake(destination_dtype, initial.size),
            ),
            options="--npu-arch 3510",
        )
        entry = dict(case=label, kernel=str(compiled.kernel_binary_path), launches=0)
        print(f"case={label}", flush=True)
        if args.run:
            source_device = torch.from_numpy(source_data.copy()).to(
                f"npu:{args.device}"
            )
            destination_device = torch.from_numpy(initial.copy()).to(
                f"npu:{args.device}"
            )
            source_view = from_dlpack(
                source_device, element_type=source_dtype, layout_tag=tla.arch.RowMajor
            )
            destination_view = from_dlpack(
                destination_device,
                element_type=destination_dtype,
                layout_tag=tla.arch.RowMajor,
            )
            for source_data, initial, expected, checked in cases:
                source_device.copy_(torch.from_numpy(source_data))
                destination_device.copy_(torch.from_numpy(initial))
                compiled(source_view, destination_view, block_num=1)
                torch.npu.synchronize(args.device)
                _assert_values(source_device.cpu().numpy(), source_data)
                _assert_values(
                    destination_device.cpu().numpy()[checked], expected[checked]
                )
                entry["launches"] += 1
        records.append(entry)

    hybrid = args.round_mode == "hybrid"
    round_mode = RoundMode.CAST_HYBRID if hybrid else RoundMode.CAST_ROUND
    encode = encode_hif8_hybrid if hybrid else encode_hif8_rna
    samples = (
        hif8_hybrid_samples()
        if hybrid
        else boundary_samples([decode_hif8(code) for code in range(256)])
    )
    if args.case == "core":
        count = samples.size
        capacity = GUARD + (count + 63) // 64 * 64 + 64
        for saturate in (False, True):
            cases = []
            for repeat in range(2):
                source = np.full(capacity, 37, np.float32)
                source[GUARD : GUARD + count] = np.roll(samples, repeat)
                initial = np.full(capacity, 0xA5, np.uint8)
                expected = initial.copy()
                expected[GUARD : GUARD + count] = [
                    encode(float(x), saturate=saturate)
                    for x in source[GUARD : GUARD + count]
                ]
                cases.append((source, initial, expected, np.ones(capacity, bool)))
            check(
                f"encode-sat{int(saturate)}",
                make_fp8_kernel(
                    tla.HiFloat8,
                    count,
                    narrow=True,
                    cast_params=CastParams(
                        sat_mode=SatMode.SAT if saturate else SatMode.NOSAT,
                        round_mode=round_mode,
                    ),
                ),
                cases,
                tla.Float32,
                tla.HiFloat8,
            )
        cases = []
        for repeat in range(2):
            source = np.full(352, 0xA5, np.uint8)
            source[GUARD : GUARD + 256] = np.roll(
                np.arange(256, dtype=np.uint8), repeat
            )
            initial = np.full(352, -7.125, np.float32)
            expected = initial.copy()
            decoded = [decode_hif8(int(x)) for x in source[GUARD : GUARD + 256]]
            expected[GUARD : GUARD + 256] = decoded
            cases.append((source, initial, expected, np.ones(352, bool)))
        check(
            "decode-all-codes",
            make_fp8_kernel(tla.HiFloat8, 256, narrow=False, cast_params=CastParams()),
            cases,
            tla.HiFloat8,
            tla.Float32,
        )
        cases = []
        for repeat in range(2):
            source = np.arange(608, dtype=np.float32) / 17 + repeat
            initial = np.full(352, 0xA5, np.uint8)
            expected = initial.copy()
            expected[GUARD : GUARD + 255] = [
                encode(float(x), saturate=False)
                for x in source[GUARD : GUARD + 510 : 2]
            ]
            cases.append((source, initial, expected, np.ones(initial.size, bool)))
        check(
            "stride2-tail255",
            make_fp8_kernel(
                tla.HiFloat8,
                255,
                narrow=True,
                stride=2,
                cast_params=CastParams(round_mode=round_mode),
            ),
            cases,
            tla.Float32,
            tla.HiFloat8,
        )
    else:
        for slot in RegSlot:
            position = list(RegSlot).index(slot)
            configurations = (
                [(True, False)] if hybrid else [(True, False), (False, False)]
            )
            if args.case == "masks":
                configurations = [(True, False), (True, True)]
            for narrow, saturate in configurations:
                cases = []
                sparse = args.case == "masks"
                active = 61 if position % 2 else 64
                for repeat in range(2):
                    if narrow:
                        source = np.full(352, 37, np.float32)
                        selected_samples = (
                            samples[np.linspace(0, samples.size - 1, 64, dtype=int)]
                            if hybrid
                            else np.resize(samples, 64)
                        )
                        source[GUARD : GUARD + 64] = np.roll(selected_samples, repeat)
                        initial = np.full(352, 0xA5, np.uint8)
                        expected = initial.copy()
                        lanes = np.arange(64 if sparse else active)
                        indices = GUARD + 4 * lanes + position
                        expected[indices] = [
                            encode(float(x), saturate=saturate)
                            if not sparse or (lane < active and lane % 3 == 0)
                            else 0
                            for lane, x in enumerate(source[GUARD : GUARD + lanes.size])
                        ]
                        checked = np.ones(352, bool)
                        checked[GUARD : GUARD + (256 if sparse else active * 4)] = False
                        checked[indices] = (
                            True  # Other cast bytes are not part of the slot contract.
                        )
                        source_dtype, destination_dtype = tla.Float32, tla.HiFloat8
                    else:
                        source = np.full(352, 0xA5, np.uint8)
                        codes = np.roll(np.arange(64, dtype=np.uint8), repeat)
                        source[GUARD + 4 * np.arange(64) + position] = codes
                        initial = np.full(352, -7.125, np.float32)
                        expected = initial.copy()
                        decoded = [decode_hif8(int(x)) for x in codes[:active]]
                        expected[GUARD : GUARD + active] = decoded
                        checked = np.ones(352, bool)
                        source_dtype, destination_dtype = tla.HiFloat8, tla.Float32
                    cases.append((source, initial, expected, checked))
                check(
                    f"slot{position}-{'narrow' if narrow else 'widen'}"
                    + (f"-sparse-sat{int(saturate)}" if sparse else ""),
                    make_slot_kernel(
                        slot,
                        narrow=narrow,
                        active=active,
                        round_mode=round_mode,
                        saturate=saturate,
                        sparse=sparse,
                    ),
                    cases,
                    source_dtype,
                    destination_dtype,
                )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("core", "slots", "masks"), default="core")
    parser.add_argument("--round-mode", choices=("round", "hybrid"), default="round")
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
