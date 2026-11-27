# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""On-device HiFloat8 GM ↔ UB ↔ GM round-trip.

Proves typed ``hifloat8_t`` BC symbols (tile uint8 DataCopyPad) move every byte
correctly. Host storage is uint8; ``from_dlpack(..., element_type=HiFloat8)``
selects the typed route. Register pass-through uses UInt8 bitcast because
HiFloat8 has no arithmetic.
"""

from __future__ import annotations

import argparse

import catlass.tla as tla

_REGISTER_BYTES = 256
_GUARD_BYTE = 0xA5
_ELE = _REGISTER_BYTES  # 1 byte / HiFloat8 lane


@tla.kernel
def hifp8_gm_ub_copy(src: tla.Tensor, dst: tla.Tensor) -> None:
    elements = src.origin_shape[0]
    ub_src = tla.make_tensor_like(
        tla.allocate(elements, tla.HiFloat8, tla.AddressSpace.ub, _REGISTER_BYTES),
        src,
        tla.arch.RowMajor,
    )
    ub_dst = tla.make_tensor_like(
        tla.allocate(elements, tla.HiFloat8, tla.AddressSpace.ub, _REGISTER_BYTES),
        dst,
        tla.arch.RowMajor,
    )
    loaded = tla.flag("loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    stored = tla.flag("stored", tla.pipes.VECTOR, tla.pipes.MTE3)

    with tla.vector():
        tla.copy(ub_src, src)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)

        with tla.vec.func(mode="simd"):
            # Identity through UInt8: exercises load/store of typed HiFloat8
            # tiles without numeric ops.
            tile_src = tla.tile_view(ub_src, tla.make_shape(_ELE), tla.make_coord(0))
            tile_dst = tla.tile_view(ub_dst, tla.make_shape(_ELE), tla.make_coord(0))
            tile_dst.store(tile_src.load().bitcast(tla.UInt8).bitcast(tla.HiFloat8))

        tla.set_flag(stored)
        tla.wait_flag(stored)
        tla.copy(dst, ub_dst)
        tla.pipe_barrier(tla.pipes.ALL)


def run(args: argparse.Namespace) -> int:
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)

    # Full 0..255 pattern so the move cannot hide behind a signed twin.
    payload = torch.arange(_ELE, dtype=torch.uint8)
    src = payload.to("npu")

    host_backing = torch.full(
        (_ELE + 2 * _REGISTER_BYTES,), _GUARD_BYTE, dtype=torch.uint8
    )
    backing = host_backing.to("npu")
    dst = backing[_REGISTER_BYTES : _REGISTER_BYTES + _ELE]
    host_backing[_REGISTER_BYTES : _REGISTER_BYTES + _ELE] = payload

    tla_src = tla.from_dlpack(
        src,
        layout_tag=tla.arch.RowMajor,
        origin_shape=(_ELE,),
        element_type=tla.HiFloat8,
    )
    tla_dst = tla.from_dlpack(
        dst,
        layout_tag=tla.arch.RowMajor,
        origin_shape=(_ELE,),
        element_type=tla.HiFloat8,
    )
    artifact = tla.compile(
        hifp8_gm_ub_copy,
        tla_src,
        tla_dst,
        options="--npu-arch 3510",
    )
    artifact(tla_src, tla_dst, block_num=1)
    torch.npu.synchronize()

    passed = torch.equal(backing.cpu(), host_backing)
    print(
        f"hif8 RowMajor GM↔UB: {_ELE} elements + guards, "
        f"passed={passed} cache_key={getattr(artifact, 'cache_key', None)}"
    )
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="On-device HiFloat8 GM ↔ UB ↔ GM round-trip."
    )
    parser.add_argument("--device", type=int, default=0)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
