# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Ascend 950 LoadAlign modes compared with an independent CPU byte reference."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from catlass import tla
from catlass.params import LoadDist, NormalLoadParams
from catlass.tla.runtime import from_dlpack, make_fake_tensor


@dataclass(frozen=True)
class LoadAlignCase:
    mode: int
    token: str
    type_name: str
    element_bytes: int

    @property
    def name(self):
        return f"{self.token}-{self.type_name}"

    @property
    def source_stride(self):
        return 32 if self.mode == 0 else (128 if self.mode in (3, 4) else 512)

    @property
    def output_stride(self):
        return 256


_B8 = (("Int8", 1), ("UInt8", 1), ("Float8E4M3FN", 1), ("Float8E5M2", 1))
_B16 = (("Int16", 2), ("UInt16", 2), ("Float16", 2), ("BFloat16", 2))
_B32 = (("Int32", 4), ("UInt32", 4), ("Float32", 4))
CASES = tuple(
    LoadAlignCase(mode, token, type_name, width)
    for mode, token, dtypes in (
        (0, "DIST_BRC_B8", _B8),
        (1, "DIST_DS_B8", _B8),
        (2, "DIST_DS_B16", _B16),
        (3, "DIST_UNPACK_B32", _B32),
        (4, "DIST_UNPACK_B8", _B8),
    )
    for type_name, width in dtypes
)


def input_bytes(case, iterations, edge_patterns=False):
    indices = np.arange(iterations * case.source_stride + 128, dtype=np.uint32)
    if edge_patterns:
        words = np.array(
            [
                0,
                0x80000000,
                0x7F800000,
                0xFF800000,
                0x7FC12345,
                0xFFFFFFFF,
                1,
                0x807F0080,
            ],
            dtype=np.uint32,
        )
        return np.resize(words.view(np.uint8), indices.size)
    return ((indices * 37 + 11) ^ (indices >> 3) ^ (indices >> 8)).astype(np.uint8)


def golden_bytes(case, source, iterations, offset):
    expected = np.zeros(iterations * case.output_stride, dtype=np.uint8)
    for iteration in range(iterations):
        base = iteration * case.source_stride + offset
        output = expected[
            iteration * case.output_stride : (iteration + 1) * case.output_stride
        ]
        if case.mode == 0:
            output[:] = source[base]
        elif case.mode == 1:
            output[:] = source[base : base + 512 : 2]
        elif case.mode == 2:
            output[:] = source[base : base + 512].reshape(-1, 2)[::2].reshape(-1)
        elif case.mode == 3:
            output.reshape(-1, 4)[::2] = source[base : base + 128].reshape(-1, 4)
        elif case.mode == 4:
            output[::2] = source[base : base + 128]
    return expected


def make_kernel(case, iterations):
    dtype = getattr(tla, case.type_name)
    distribution = getattr(LoadDist, case.token)
    source_stride = case.source_stride
    output_stride = case.output_stride
    source_capacity = iterations * source_stride + 128
    output_capacity = iterations * output_stride
    element_bytes = case.element_bytes
    source_stride_elements = source_stride // element_bytes
    output_stride_elements = output_stride // element_bytes

    @tla.kernel
    def transfer(
        source: tla.Tensor,
        output: tla.Tensor,
        iteration_count: tla.Int16,
        source_offset: tla.Int32,
    ):
        source_ptr = tla.allocate(source_capacity, tla.UInt8, tla.AddressSpace.ub, 256)
        output_ptr = tla.allocate(output_capacity, tla.UInt8, tla.AddressSpace.ub, 256)
        source_buffer = tla.make_tensor_like(source_ptr, source, tla.arch.RowMajor)
        output_buffer = tla.make_tensor_like(output_ptr, output, tla.arch.RowMajor)
        # Fixed VF bases; make_tensor coords are elements, tile_view coords are tiles.
        typed_source_ptr = tla.recast_ptr(source_ptr, dtype=dtype)
        typed_output_ptr = tla.recast_ptr(output_ptr, dtype=dtype)
        source_layout = tla.make_layout(
            tla.make_shape(source_capacity // element_bytes), tla.make_stride(1)
        )
        output_layout = tla.make_layout(
            tla.make_shape(output_capacity // element_bytes), tla.make_stride(1)
        )
        # Convert units once; signed division in the VF loop adds address work.
        source_offset_elements = source_offset // element_bytes
        ready = tla.flag("ready", tla.pipes.MTE2, tla.pipes.VECTOR)
        done = tla.flag("done", tla.pipes.VECTOR, tla.pipes.MTE3)
        with tla.vector():
            tla.copy(source_buffer, source)
            tla.set_flag(ready)
            tla.wait_flag(ready)
            with tla.vec.func(mode="simd"):
                for iteration in tla.range(iteration_count):
                    source_tile = tla.make_tensor(
                        typed_source_ptr,
                        source_layout,
                        tla.make_coord(
                            iteration * source_stride_elements + source_offset_elements
                        ),
                    )
                    first_tile = tla.make_tensor(
                        typed_output_ptr,
                        output_layout,
                        tla.make_coord(iteration * output_stride_elements),
                    )
                    first_tile.store(
                        source_tile.load(NormalLoadParams(load_dist=distribution))
                    )
            tla.set_flag(done)
            tla.wait_flag(done)
            tla.copy(output, output_buffer)
            tla.pipe_barrier(tla.pipes.ALL)

    return transfer


def run_case(case, iterations, device, *, compile_only=False):
    source_bytes = iterations * case.source_stride + 128
    output_bytes = iterations * case.output_stride

    def fake(size):
        return make_fake_tensor(
            tla.UInt8,
            (size,),
            (1,),
            addrspace=tla.AddressSpace.gm,
            origin_shape=(size,),
            layout_tag=tla.arch.RowMajor,
        )

    compiled = make_kernel(case, iterations).compile(
        type_args=(
            fake(source_bytes),
            fake(output_bytes),
            tla.Int16(iterations),
            tla.Int32(0),
        ),
        options="--npu-arch 3510",
    )
    print(
        f"case={case.name} iterations={iterations} kernel={compiled.kernel_binary_path}",
        flush=True,
    )
    if compile_only:
        return 0

    import torch

    settings = [(offset, False) for offset in (0, 32, 64)]
    if iterations == 3:
        if case.mode == 0:
            settings.extend((offset, False) for offset in (1, 31, 63))
        settings.append((32, True))
    for offset, edge_patterns in settings:
        source_host = input_bytes(case, iterations, edge_patterns)
        expected = golden_bytes(case, source_host, iterations, offset)
        source_device = torch.from_numpy(source_host.copy()).to(f"npu:{device}")
        output_device = torch.from_numpy(np.full(output_bytes, 0xA5, np.uint8)).to(
            f"npu:{device}"
        )
        source_view = from_dlpack(source_device, layout_tag=tla.arch.RowMajor)
        output_view = from_dlpack(output_device, layout_tag=tla.arch.RowMajor)
        compiled(
            source_view,
            output_view,
            tla.Int16(iterations),
            tla.Int32(offset),
            block_num=1,
        )
        torch.npu.synchronize(device)
        np.testing.assert_array_equal(output_device.cpu().numpy(), expected)
        np.testing.assert_array_equal(source_device.cpu().numpy(), source_host)
        print(
            f"passed: {case.name} iterations={iterations} offset={offset} edge_patterns={edge_patterns}",
            flush=True,
        )
    return len(settings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("all", *(dict.fromkeys(case.token for case in CASES))),
        default="all",
    )
    parser.add_argument(
        "--dtype",
        choices=("all", *(dict.fromkeys(case.type_name for case in CASES))),
        default="all",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    cases = tuple(
        case
        for case in CASES
        if (args.mode == "all" or args.mode == case.token)
        and (args.dtype == "all" or args.dtype == case.type_name)
    )
    if not cases:
        parser.error("the selected dtype is not supported by this LoadAlign mode")
    if args.device < 0:
        parser.error("device must be nonnegative")
    if not args.compile_only:
        import torch
        import torch_npu  # noqa: F401 -- register the backend only for device execution

        torch.npu.set_device(args.device)
    checks = 0
    for case in cases:
        for iterations in (1, 3):
            checks += run_case(
                case, iterations, args.device, compile_only=args.compile_only
            )
    print(
        f"status={'compiled' if args.compile_only else 'passed'} kernels={len(cases) * 2} checks={checks}"
    )


if __name__ == "__main__":
    main()
