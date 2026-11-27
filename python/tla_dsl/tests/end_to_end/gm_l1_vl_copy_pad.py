# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""On-device GM→L1 VectorLayout CopyPadParams non-zero fill (dav_3510 OUTTOL1).

Stages 7 f32 elements from GM into an 8-element L1 VL tile with
``right_padding=1`` and ``padding_value=1.5``, then prints the L1 buffer as
one u32 C0. Host checks that payload words match the source and the pad
slot holds IEEE bits of 1.5 (not zero — the P1 regression).
"""

from __future__ import annotations

import argparse
import re
import struct
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DSL_EXAMPLE_PATH = str(_REPO_ROOT / "examples" / "python" / "tla_dsl")
if _DSL_EXAMPLE_PATH not in sys.path:
    sys.path.append(_DSL_EXAMPLE_PATH)

from catlass import tla

_SRC_ELEMS = 7
_DST_ELEMS = 8  # src + right_padding=1
_PAD_VALUE = 1.5
_C0_U32 = 8
_ALIGN = 512


@tla.kernel
def gm_l1_vl_pad_then_print(gm_src: tla.Tensor) -> None:
    l1_ptr = tla.allocate(_DST_ELEMS, tla.Float32, tla.AddressSpace.l1, _ALIGN)
    copied = tla.flag("vl_pad_copied", tla.pipes.MTE2, tla.pipes.MTE1)
    with tla.cube():
        l1 = tla.make_tensor(
            l1_ptr,
            tla.make_layout(
                tla.make_shape(_DST_ELEMS),
                tla.make_stride(1),
                origin_shape=tla.make_shape(_DST_ELEMS),
                layoutTag=tla.arch.VectorLayout,
            ),
            tla.make_coord(0),
        )
        tla.copy(
            l1,
            gm_src,
            params=tla.params.CopyPadParams(
                right_padding=1,
                padding_value=_PAD_VALUE,
            ),
        )
        tla.set_flag(copied)
        tla.wait_flag(copied)
        # Print needs zN on L1; flat 32B run is one C0 of eight u32 (7 f32 + pad).
        print_ptr = tla.recast_ptr(l1_ptr, dtype=tla.UInt32)
        printed = tla.make_tensor(
            print_ptr,
            tla.make_layout_with_tag((1, _C0_U32), tla.UInt32, tla.arch.zN),
        )
        tla.print(printed, _C0_U32)


def _parse_u32_c0(stdout: str) -> list[int]:
    # Prefer ``values=[...]`` from tla.print; fall back to trailing int list.
    m = re.search(r"values=\[([^\]]+)\]", stdout)
    if m is None:
        return []
    return [int(tok.strip(), 0) for tok in m.group(1).split(",") if tok.strip()]


def run(device: int) -> int:
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(device)

    src_host = torch.arange(_SRC_ELEMS, dtype=torch.float32)
    src_dev = src_host.to(f"npu:{device}")
    gm_src = tla.from_dlpack(
        src_dev,
        layout_tag=tla.arch.VectorLayout,
        origin_shape=(_SRC_ELEMS,),
    )
    artifact = tla.compile(
        gm_l1_vl_pad_then_print,
        gm_src,
        options="--npu-arch 3510",
    )

    buf = StringIO()
    with redirect_stdout(buf):
        artifact(gm_src, block_num=1)
        torch.npu.synchronize()
    out = buf.getvalue()
    words = _parse_u32_c0(out)
    if len(words) < _DST_ELEMS:
        print(f"GM→L1 VL pad: failed to parse {_DST_ELEMS} u32 words from print:\n{out}")
        return 1

    expected_payload = [
        struct.unpack("<I", struct.pack("<f", float(v)))[0]
        for v in src_host.tolist()
    ]
    pad_bits = struct.unpack("<I", struct.pack("<f", _PAD_VALUE))[0]

    payload_ok = words[:_SRC_ELEMS] == expected_payload
    pad_ok = words[_SRC_ELEMS] == pad_bits
    passed = bool(payload_ok and pad_ok)
    print(
        f"GM→L1 VL pad right=1 value={_PAD_VALUE}: "
        f"payload_ok={payload_ok} pad_ok={pad_ok} "
        f"pad_word=0x{words[_SRC_ELEMS]:08x} expect=0x{pad_bits:08x} passed={passed}"
    )
    if not passed:
        print(f"  words={[hex(w) for w in words[:_DST_ELEMS]]}")
        print(f"  stdout:\n{out}")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    return run(parser.parse_args().device)


if __name__ == "__main__":
    raise SystemExit(main())
