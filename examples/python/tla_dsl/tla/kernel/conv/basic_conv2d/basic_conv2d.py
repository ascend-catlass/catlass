# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The smallest possible convolution: one block, one K-part, one N-part.

The convolution is an ordinary GEMM plus one instruction. ``tla.experimental.img2col`` expands
the NC1HWC0 feature map in L1 into the implicit-GEMM matrix as it loads L0A, and
``tla.mmad`` then does what it always does:

    M = ho * wo      N = cout      K = cin1 * kh * kw * C0

Validated against ``torch.nn.functional.conv2d``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_DSL_EXAMPLE_PATH = str(
    next(p for p in Path(__file__).resolve().parents if (p / "helpers").is_dir())
)
if _DSL_EXAMPLE_PATH not in sys.path:
    sys.path.append(_DSL_EXAMPLE_PATH)

import catlass.tla as tla
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

C0 = 16  # 32 bytes / sizeof(f16)
CIN1 = 2
CIN = CIN1 * C0
HI = WI = 8
KH = KW = 3
COUT = 32
PAD_T = PAD_B = PAD_L = PAD_R = 1
STRIDE_H = STRIDE_W = 1
DIL_H = DIL_W = 1

HO = (HI + PAD_T + PAD_B - DIL_H * (KH - 1) - 1) // STRIDE_H + 1
WO = (WI + PAD_L + PAD_R - DIL_W * (KW - 1) - 1) // STRIDE_W + 1
M = HO * WO
K = CIN1 * KH * KW * C0
N = COUT

# The feature map as a matrix: HI*WI rows of CIN columns, nested NC1HWC0.
FMAP_ROWS = HI * WI


@tla.kernel
def basic_conv2d(fmap: tla.Tensor, filt: tla.Tensor, out: tla.Tensor) -> None:
    l1_loaded = tla.flag("l1_loaded", tla.arch.MTE2, tla.arch.MTE1)
    l0_loaded = tla.flag("l0_loaded", tla.arch.MTE1, tla.arch.CUBE)
    l0c_ready = tla.flag("l0c_ready", tla.arch.CUBE, tla.arch.FIX)

    l1a_ptr = tla.allocate(CIN1 * HI * WI * C0, tla.Float16, tla.AddressSpace.l1, 512)
    l1b_ptr = tla.allocate(K * N, tla.Float16, tla.AddressSpace.l1, 512)
    l0a_ptr = tla.allocate(M * K, tla.Float16, tla.AddressSpace.l0a, 512)
    l0b_ptr = tla.allocate(K * N, tla.Float16, tla.AddressSpace.l0b, 512)
    l0c_ptr = tla.allocate(M * N, tla.Float32, tla.AddressSpace.l0c, 1024)

    with tla.cube():
        # --- GM -> L1. Two ordinary copies; neither knows about convolution. ---
        # The feature map reaches the kernel as a flat RowMajor buffer -- that is
        # all the argument ABI accepts -- and is re-viewed here as NC1HWC0:
        # ((h*w, 1), (C0, c1)), with the c1 leaf at the feature map's h*w*C0
        # pitch. Same bytes, and it is the view img2col needs.
        gm_a = tla.make_tensor(
            fmap.ptr,
            tla.make_layout(
                tla.make_shape((FMAP_ROWS, 1), (C0, CIN1)),
                tla.make_stride((C0, FMAP_ROWS * C0), (1, FMAP_ROWS * C0)),
                origin_shape=tla.make_shape(FMAP_ROWS, CIN),
                layoutTag=tla.arch.zNUnAlign,
            ),
        )
        gm_b = tla.tile_view(filt, tla.make_shape(K, N), tla.make_coord(0, 0))
        l1_a = tla.make_tensor_like(l1a_ptr, gm_a, tla.arch.zNUnAlign)
        l1_b = tla.make_tensor_like(l1b_ptr, gm_b, tla.arch.zN)
        tla.copy(l1_a, gm_a)
        tla.copy(l1_b, gm_b)
        tla.set_flag(l1_loaded)
        tla.wait_flag(l1_loaded)

        # --- L1 -> L0. The one convolution-specific instruction. ---
        # The img2col operand's logical shape is (M, K) -- which is neither the
        # feature map's shape nor the filter's, so it needs its own reference
        # tile for make_tensor_like to derive the zN tree from. The reference is
        # metadata only: nothing is ever read through this pointer.
        img2col_shape = tla.make_tensor(
            fmap.ptr,
            tla.make_layout(
                tla.make_shape(M, K),
                tla.make_stride(K, 1),
                layoutTag=tla.arch.RowMajor,
            ),
        )
        l0_a = tla.make_tensor_like(l0a_ptr, img2col_shape, tla.arch.zN)
        tla.experimental.img2col(
            l0_a,
            l1_a,
            tla.experimental.Img2colParams(
                fmap_hw=(HI, WI),
                filter_hw=(KH, KW),
                stride_hw=(STRIDE_H, STRIDE_W),
                dilation_hw=(DIL_H, DIL_W),
                pad=(PAD_L, PAD_R, PAD_T, PAD_B),
            ),
        )
        l0_b = tla.make_tensor_like(l0b_ptr, gm_b, tla.arch.nZ)
        tla.copy(l0_b, l1_b)
        tla.set_flag(l0_loaded)
        tla.wait_flag(l0_loaded)

        # --- Compute and epilogue: plain GEMM, untouched. ---
        gm_c = tla.tile_view(out, tla.make_shape(M, N), tla.make_coord(0, 0))
        l0_c = tla.make_tensor_like(l0c_ptr, gm_c, tla.arch.L0Clayout)
        tla.mmad(l0_c, l0_a, l0_b, init_c=True)
        tla.set_flag(l0c_ready)
        tla.wait_flag(l0c_ready)
        tla.copy(gm_c, l0_c)


def to_nc1hwc0(x: torch.Tensor) -> torch.Tensor:
    """(1, CIN, HI, WI) -> the NC1HWC0 byte order, flattened."""
    return (
        x.view(1, CIN1, C0, HI, WI)
        .permute(0, 1, 3, 4, 2)  # (1, cin1, hi, wi, c0)
        .contiguous()
        .view(FMAP_ROWS * CIN1, C0)
    )


def to_filter_matrix(w: torch.Tensor) -> torch.Tensor:
    """(COUT, CIN, KH, KW) -> the (K, N) matrix, K ordered (cin1, kh, kw, c0).

    That is the K order Catlass's conv filter layout uses, and it has to agree
    with the order img2col emits K in -- which is the one thing here that cannot
    be settled by reading the headers.
    """
    return (
        w.view(COUT, CIN1, C0, KH, KW)
        .permute(1, 3, 4, 2, 0)  # (cin1, kh, kw, c0, cout)
        .contiguous()
        .view(K, COUT)
    )


def run(args: argparse.Namespace) -> int:
    from catlass.tla.runtime import from_dlpack
    from helpers import compare

    torch.npu.set_device(args.device)
    torch.manual_seed(0)

    x = torch.randn(1, CIN, HI, WI, dtype=torch.float16)
    w = torch.randn(COUT, CIN, KH, KW, dtype=torch.float16)

    ref = F.conv2d(
        x.float(),
        w.float(),
        stride=(STRIDE_H, STRIDE_W),
        padding=(PAD_T, PAD_L),
        dilation=(DIL_H, DIL_W),
    )
    # (1, cout, ho, wo) -> (M = ho*wo, cout), matching the kernel's L0C.
    ref = ref[0].permute(1, 2, 0).contiguous().view(M, COUT)

    a = to_nc1hwc0(x).npu()
    b = to_filter_matrix(w).npu()
    c = torch.zeros(M, N, dtype=torch.float32).npu()

    a_tensor = from_dlpack(
        a, layout_tag=tla.arch.RowMajor, origin_shape=(FMAP_ROWS * CIN1, C0)
    )
    b_tensor = from_dlpack(b, layout_tag=tla.arch.RowMajor, origin_shape=(K, N))
    c_tensor = from_dlpack(c, layout_tag=tla.arch.RowMajor, origin_shape=(M, N))

    artifact = tla.compile(
        basic_conv2d, a_tensor, b_tensor, c_tensor, options="--npu-arch 3510"
    )
    artifact(a_tensor, b_tensor, c_tensor, block_num=1)
    torch.npu.synchronize()

    passed = compare(c.detach().cpu(), ref, K)
    print(f"passed={passed} kernel.o={artifact.kernel_binary_path}")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Minimal conv2d via tla.experimental.img2col."
    )
    parser.add_argument("--device", type=int, default=0)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
