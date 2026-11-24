# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Convolution e2e for img2col pad fill with one height-direction load.

``torch.nn.functional.conv2d`` always pads with zero. To check a non-zero
``pad_value``, the host golden manually constant-pads the feature map, then
runs ``conv2d`` with ``padding=0``. The kernel passes the same fill as the
``Img2colParams.pad_value`` bit pattern and explicitly requests one load
along the destination height. The convolution uses the same kernel as the basic convolution example.

    python tests/end_to_end/img2col_pad_value_conv.py --device 0
    python tests/end_to_end/img2col_pad_value_conv.py --device 0 --control-wrong-pad
"""

from __future__ import annotations

import argparse
import importlib.util
import struct
from pathlib import Path

import catlass.tla as tla
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
from catlass.tla.runtime import from_dlpack

example_path = (
    Path(__file__).resolve().parents[4]
    / "examples/python/tla_dsl/tla/kernel/conv/basic_conv2d/basic_conv2d.py"
)
spec = importlib.util.spec_from_file_location(
    "_catlass_img2col_basic_conv2d", example_path
)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load the basic convolution example from {example_path}")
conv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conv)

OPTIONS = "--npu-arch 3510"

# f16 1.0 bit pattern.
PAD_FILL_F16 = 1.0
PAD_VALUE_BITS = struct.unpack("<H", struct.pack("<e", PAD_FILL_F16))[0]


def golden(x: torch.Tensor, w: torch.Tensor, pad_fill: float) -> torch.Tensor:
    """Constant-pad then conv2d with padding=0, matching non-zero pad_value."""
    x_padded = F.pad(
        x.float(),
        (conv.PAD_L, conv.PAD_R, conv.PAD_T, conv.PAD_B),
        mode="constant",
        value=pad_fill,
    )
    ref = F.conv2d(
        x_padded,
        w.float(),
        stride=(conv.STRIDE_H, conv.STRIDE_W),
        padding=0,
        dilation=(conv.DIL_H, conv.DIL_W),
    )
    return ref[0].permute(1, 2, 0).contiguous().view(conv.M, conv.COUT)


def main() -> int:
    ap = argparse.ArgumentParser(description="img2col pad_value convolution e2e")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--tolerance", type=float, default=5e-3)
    ap.add_argument(
        "--control-wrong-pad",
        action="store_true",
        help="compare against a golden that used a different pad fill; must FAIL",
    )
    args = ap.parse_args()

    torch.npu.set_device(args.device)
    torch.manual_seed(0)

    x = torch.randn(1, conv.CIN, conv.HI, conv.WI, dtype=torch.float16)
    w = torch.randn(conv.COUT, conv.CIN, conv.KH, conv.KW, dtype=torch.float16)

    golden_fill = 0.0 if args.control_wrong_pad else PAD_FILL_F16
    want = golden(x, w, golden_fill)

    a = conv.to_nc1hwc0(x).npu()
    b = conv.to_filter_matrix(w).npu()
    c = torch.zeros(conv.M, conv.N, dtype=torch.float32).npu()

    a_tensor = from_dlpack(
        a,
        layout_tag=tla.arch.RowMajor,
        origin_shape=(conv.FMAP_ROWS * conv.CIN1, conv.C0),
    )
    b_tensor = from_dlpack(
        b, layout_tag=tla.arch.RowMajor, origin_shape=(conv.K, conv.N)
    )
    c_tensor = from_dlpack(
        c, layout_tag=tla.arch.RowMajor, origin_shape=(conv.M, conv.N)
    )

    artifact = tla.compile(
        conv.basic_conv2d, a_tensor, b_tensor, c_tensor, PAD_VALUE_BITS, options=OPTIONS
    )
    artifact(a_tensor, b_tensor, c_tensor, block_num=1)
    torch.npu.synchronize()

    got = c.detach().cpu()
    err = (got - want).abs().max().item()
    rel = err / max(want.abs().max().item(), 1e-9)
    ok = rel < args.tolerance

    if args.control_wrong_pad:
        if not ok:
            print(
                f"PASS (control): wrong pad fill disagrees "
                f"max_abs={err:.4g} rel={rel:.3g}",
                flush=True,
            )
            return 0
        print(
            "FAILED (control): wrong pad fill still matched, comparison insensitive",
            flush=True,
        )
        return 2

    if ok:
        print(
            f"PASS: pad_value={PAD_VALUE_BITS:#x} fill={PAD_FILL_F16} "
            f"max_abs={err:.4g} rel={rel:.3g}",
            flush=True,
        )
        return 0
    print(f"FAILED: max_abs={err:.4g} rel={rel:.3g}", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
