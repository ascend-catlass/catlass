# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Ascend 950 DINTLV_B8 compared with an independent CPU byte reference."""

from __future__ import annotations

import argparse

import numpy as np
from catlass import tla
from catlass.params import LoadDist, NormalLoadParams
from catlass.tla.runtime import from_dlpack, make_fake_tensor


_FP4_DTYPES = ("Float4E2M1x2", "Float4E1M2x2")
_DTYPES = (
    "Int8",
    "UInt8",
    "Float8E4M3FN",
    "Float8E5M2",
    "HiFloat8",
    "Float8E8M0",
    *_FP4_DTYPES,
)


def input_bytes(iterations, edge_patterns=False, packed_patterns=False):
    indices = np.arange(iterations * 512 + 128, dtype=np.uint32)
    if packed_patterns:
        # Both outputs see all 256 codes, including E8M0 NaN and FP4x2 nibble pairs.
        return np.resize(np.repeat(np.arange(256, dtype=np.uint8), 2), indices.size)
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


def golden_bytes(source, iterations, offset):
    expected = np.zeros(iterations * 512, dtype=np.uint8)
    for iteration in range(iterations):
        base = iteration * 512 + offset
        output = expected[iteration * 512 : (iteration + 1) * 512]
        output[:256] = source[base : base + 512 : 2]
        output[256:] = source[base + 1 : base + 512 : 2]
    return expected


def make_kernel(type_name, iterations):
    dtype = getattr(tla, type_name)
    distribution = LoadDist.DIST_DINTLV_B8
    source_stride = 512
    output_stride = 512
    source_capacity = iterations * source_stride + 128
    output_capacity = iterations * output_stride
    source_stride_elements = source_stride
    output_stride_elements = output_stride
    register_elements = 256

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
            tla.make_shape(source_capacity), tla.make_stride(1)
        )
        output_layout = tla.make_layout(
            tla.make_shape(output_capacity), tla.make_stride(1)
        )
        ready = tla.flag("ready", tla.arch.MTE2, tla.arch.VECTOR)
        done = tla.flag("done", tla.arch.VECTOR, tla.arch.MTE3)
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
                            iteration * source_stride_elements + source_offset
                        ),
                    )
                    first_tile = tla.make_tensor(
                        typed_output_ptr,
                        output_layout,
                        tla.make_coord(iteration * output_stride_elements),
                    )
                    first, second = source_tile.load(
                        NormalLoadParams(load_dist=distribution)
                    )
                    first_tile.store(first)
                    second_tile = tla.make_tensor(
                        typed_output_ptr,
                        output_layout,
                        tla.make_coord(
                            iteration * output_stride_elements + register_elements
                        ),
                    )
                    second_tile.store(second)
            tla.set_flag(done)
            tla.wait_flag(done)
            tla.copy(output, output_buffer)
            tla.pipe_barrier(tla.pipes.ALL)

    return transfer


def run_case(type_name, iterations, device, *, compile_only=False):
    source_bytes = iterations * 512 + 128
    output_bytes = iterations * 512

    def fake(size):
        return make_fake_tensor(
            tla.UInt8,
            (size,),
            (1,),
            addrspace=tla.AddressSpace.gm,
            origin_shape=(size,),
            layout_tag=tla.arch.RowMajor,
        )

    compiled = make_kernel(type_name, iterations).compile(
        type_args=(
            fake(source_bytes),
            fake(output_bytes),
            tla.Int16(iterations),
            tla.Int32(0),
        ),
        options="--npu-arch 3510",
    )
    print(
        f"case=DIST_DINTLV_B8-{type_name} iterations={iterations} kernel={compiled.kernel_binary_path}",
        flush=True,
    )
    if compile_only:
        return 0

    import torch

    settings = [(offset, False, False) for offset in (0, 32, 64)]
    if iterations == 3:
        settings.append((32, True, False))
    if type_name in (*_FP4_DTYPES, "Float8E8M0"):
        settings.append((32, False, True))
    for offset, edge_patterns, packed_patterns in settings:
        source_host = input_bytes(iterations, edge_patterns, packed_patterns)
        expected = golden_bytes(source_host, iterations, offset)
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
            f"passed: DIST_DINTLV_B8-{type_name} iterations={iterations} offset={offset} edge_patterns={edge_patterns} packed_patterns={packed_patterns}",
            flush=True,
        )
    return len(settings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("all", *_DTYPES), default="all")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    dtypes = _DTYPES if args.dtype == "all" else (args.dtype,)
    if args.device < 0:
        parser.error("device must be nonnegative")
    if not args.compile_only:
        import torch
        import torch_npu  # noqa: F401 -- register the backend only for device execution

        torch.npu.set_device(args.device)
    checks = 0
    for type_name in dtypes:
        for iterations in (1, 3):
            checks += run_case(
                type_name, iterations, args.device, compile_only=args.compile_only
            )
    print(
        f"status={'compiled' if args.compile_only else 'passed'} kernels={len(dtypes) * 2} checks={checks}"
    )


if __name__ == "__main__":
    main()
