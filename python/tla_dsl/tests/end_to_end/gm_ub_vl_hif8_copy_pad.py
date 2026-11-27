# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""On-device GM→UB VectorLayout HiFloat8 CopyPadParams non-zero fill.

Stages 7 hif8 bytes from GM into an 8-element UB VL tile with
``right_padding=1`` and ``padding_value=0xA5``, then writes the UB tile back
to GM. Host checks payload bytes and the pad slot (regression: must not be 0).
"""

from __future__ import annotations

import argparse

import catlass.tla as tla

_SRC_ELEMS = 7
_DST_ELEMS = 8
_PAD_BITS = 0xA5
_ALIGN = 256


@tla.kernel
def gm_ub_vl_hif8_pad_then_writeback(gm_src: tla.Tensor, gm_dst: tla.Tensor) -> None:
    ub_ptr = tla.allocate(_DST_ELEMS, tla.HiFloat8, tla.AddressSpace.ub, _ALIGN)
    loaded = tla.flag("loaded", tla.pipes.MTE2, tla.pipes.MTE3)
    with tla.vector():
        ub = tla.make_tensor(
            ub_ptr,
            tla.make_layout(
                tla.make_shape(_DST_ELEMS),
                tla.make_stride(1),
                origin_shape=tla.make_shape(_DST_ELEMS),
                layoutTag=tla.arch.VectorLayout,
            ),
            tla.make_coord(0),
        )
        tla.copy(
            ub,
            gm_src,
            params=tla.params.CopyPadParams(
                right_padding=1,
                padding_value=_PAD_BITS,
            ),
        )
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        tla.copy(gm_dst, ub)
        tla.pipe_barrier(tla.pipes.ALL)


def run(device: int) -> int:
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(device)

    # Distinct payload bytes so a silent zero-fill cannot pass.
    src_host = torch.arange(1, _SRC_ELEMS + 1, dtype=torch.uint8)
    dst_host = torch.full((_DST_ELEMS,), 0xFF, dtype=torch.uint8)

    src_dev = src_host.to(f"npu:{device}")
    dst_dev = dst_host.to(f"npu:{device}")
    gm_src = tla.from_dlpack(
        src_dev,
        layout_tag=tla.arch.VectorLayout,
        origin_shape=(_SRC_ELEMS,),
        element_type=tla.HiFloat8,
    )
    gm_dst = tla.from_dlpack(
        dst_dev,
        layout_tag=tla.arch.VectorLayout,
        origin_shape=(_DST_ELEMS,),
        element_type=tla.HiFloat8,
    )
    artifact = tla.compile(
        gm_ub_vl_hif8_pad_then_writeback,
        gm_src,
        gm_dst,
        options="--npu-arch 3510",
    )
    artifact(gm_src, gm_dst, block_num=1)
    torch.npu.synchronize()

    got = dst_dev.cpu()
    payload_ok = torch.equal(got[:_SRC_ELEMS], src_host)
    pad_ok = int(got[_SRC_ELEMS].item()) == _PAD_BITS
    passed = bool(payload_ok and pad_ok)
    print(
        f"GM→UB VL hif8 pad right=1 value=0x{_PAD_BITS:02x}: "
        f"payload_ok={payload_ok} pad_ok={pad_ok} "
        f"pad_byte=0x{int(got[_SRC_ELEMS].item()):02x} passed={passed}"
    )
    if not passed:
        print(f"  got={[hex(int(x)) for x in got.tolist()]}")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    return run(parser.parse_args().device)


if __name__ == "__main__":
    raise SystemExit(main())
