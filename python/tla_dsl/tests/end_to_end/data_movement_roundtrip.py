# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Check 64-bit GM → UB → register → UB → GM movement on device.

Exercise both RowMajor and VectorLayout GM → UB copies, with a RowMajor
writeback. Check both registers and the GM destination guards byte-for-byte.
"""

from __future__ import annotations

import argparse

import catlass.tla as tla

_REGISTER_BYTES = 256
_GUARD_BYTE = 0xA5
_DTYPES = {"i64": "int64", "u64": "uint64", "f64": "float64"}
_LAYOUTS = {"row": tla.arch.RowMajor, "vector": tla.arch.VectorLayout}


@tla.kernel
def data_movement_roundtrip(
    src: tla.Tensor, gm_dst: tla.Tensor, src_layout: tla.Constexpr
) -> None:
    elements = src.origin_shape[0]
    lanes = elements // 2
    gm_src = tla.make_tensor(
        src.ptr,
        tla.make_layout(
            tla.make_shape(elements), tla.make_stride(1), layoutTag=src_layout
        ),
    )
    ub_src = tla.make_tensor_like(
        tla.allocate(
            elements, gm_src.element_type, tla.AddressSpace.ub, _REGISTER_BYTES
        ),
        gm_src,
        src_layout,
    )
    ub_dst = tla.make_tensor_like(
        tla.allocate(
            elements, gm_dst.element_type, tla.AddressSpace.ub, _REGISTER_BYTES
        ),
        gm_dst,
        tla.arch.RowMajor,
    )
    loaded = tla.flag("loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    stored = tla.flag("stored", tla.pipes.VECTOR, tla.pipes.MTE3)

    with tla.vector():
        tla.copy(ub_src, gm_src)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)

        with tla.vec.func(mode="simd"):
            for tile in tla.range(2):
                src_tile = tla.tile_view(
                    ub_src, tla.make_shape(lanes), tla.make_coord(tile)
                )
                dst_tile = tla.tile_view(
                    ub_dst, tla.make_shape(lanes), tla.make_coord(tile)
                )
                dst_tile.store(src_tile.load())

        tla.set_flag(stored)
        tla.wait_flag(stored)
        tla.copy(gm_dst, ub_dst)
        tla.pipe_barrier(tla.pipes.ALL)


def run(args: argparse.Namespace) -> int:
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    dtype = getattr(torch, _DTYPES[args.dtype])
    # Match the existing f64 GM → L1 sample's same-width DLPack override.
    storage_dtype = torch.int64 if args.dtype == "f64" else dtype
    element_type = tla.Float64 if args.dtype == "f64" else None
    lanes = _REGISTER_BYTES // 8
    elements = 2 * lanes
    payload_bytes = 2 * _REGISTER_BYTES

    # Distinct register tiles and varied high and low words.
    offsets = torch.arange(payload_bytes)
    payload = ((71 * offsets + offsets // _REGISTER_BYTES) % 256).to(torch.uint8)
    if args.dtype == "f64":
        # Preserve signed zero, infinities, NaN payloads, subnormals and extremes.
        bits = torch.tensor(
            [
                0,
                -(1 << 63),
                0x3FF0000000000000,
                -0x4010000000000000,
                0x7FF0000000000000,
                -0x0010000000000000,
                0x7FF8000000001234,
                0x7FF0000000000001,
                1,
                -(1 << 63) + 1,
                0x0010000000000000,
                0x7FEFFFFFFFFFFFFF,
            ],
            dtype=torch.int64,
        ).view(torch.uint8)
        payload[: bits.numel()] = bits
    src = payload.view(storage_dtype).to("npu")

    host_backing = torch.full(
        (payload_bytes + 2 * _REGISTER_BYTES,), _GUARD_BYTE, dtype=torch.uint8
    )
    backing = host_backing.to("npu")
    dst = backing.view(storage_dtype)[lanes : lanes + elements]
    host_backing[_REGISTER_BYTES : _REGISTER_BYTES + payload_bytes] = payload

    tla_src = tla.from_dlpack(
        src,
        layout_tag=tla.arch.RowMajor,
        origin_shape=(elements,),
        element_type=element_type,
    )
    tla_dst = tla.from_dlpack(
        dst,
        layout_tag=tla.arch.RowMajor,
        origin_shape=(elements,),
        element_type=element_type,
    )
    artifact = tla.compile(
        data_movement_roundtrip,
        tla_src,
        tla_dst,
        _LAYOUTS[args.layout],
        options="--npu-arch 3510",
    )
    # The kernel handles the whole payload on one vector core.
    artifact(tla_src, tla_dst, block_num=1)
    torch.npu.synchronize()

    passed = torch.equal(backing.cpu(), host_backing)
    print(
        f"{args.dtype} {args.layout}: {elements} elements, including both guard checks, passed={passed}"
    )
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check GM and register data movement on NPU."
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dtype", choices=tuple(_DTYPES), required=True)
    parser.add_argument("--layout", choices=tuple(_LAYOUTS), required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
