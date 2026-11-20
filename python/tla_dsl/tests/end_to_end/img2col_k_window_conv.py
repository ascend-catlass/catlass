# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Convolution through tla.experimental.img2col with the K axis split several ways.

A destination K window only has to be C0-aligned: Load3Dv2 takes kStartPt and
kExtension verbatim from the destination tile and reads channelSize from the
source feature map, so a window may lie inside one C1 slice
(``kh * kw * C0`` = 144 elements here) or straddle the boundary between two.

This runs the same 3x3 convolution once per K-window size, accumulating every
window into one L0C, and checks each result against
``torch.nn.functional.conv2d``. The window sizes are the ones that matter:

    144  one whole C1 slice -- the only granularity the old verifier allowed
     48  wholly inside a C1 slice
     96  STRADDLES a C1 boundary (0..96, 96..192, ...) -- the risky case, and
         the one aclnn's own L0 tiling produces for kAL1 = 864
    288  two whole slices
     16  the C0 minimum

The filter is re-sliced to match each window, which is a plain row slice of the
packed (K, cout) weight matrix because K is ordered ``(c1, kh, kw, c0)``.

    python tests/end_to_end/img2col_k_window_conv.py --device 3
"""

from __future__ import annotations

import argparse

import catlass.tla as tla
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401  (registers the npu device)
from catlass.tla.runtime import from_dlpack

C0 = 16
KH = KW = 3
K_SLICE = KH * KW * C0  # 144: one C1 slice of the implicit-GEMM K axis
CIN = 96  # c1 = 6, so the full K axis is 864
COUT = 32
HI, WI = 10, 16  # interior-only output: 8 x 14 = 112 positions
OPTIONS = "--npu-arch 3510"

C1 = CIN // C0
PLANE = HI * WI
M = (HI - KH + 1) * (WI - KW + 1)
K_FULL = C1 * K_SLICE
# 144 is the extent the old rule allowed; the rest are what this test adds.
WINDOWS = (144, 48, 96, 288, 16)


@tla.kernel
def conv_k_windows(
    fmap: tla.Tensor,
    filt: tla.Tensor,
    out: tla.Tensor,
    k_window: tla.Constexpr[int],
) -> None:
    """One 3x3 convolution, accumulated over K windows of ``k_window`` elements."""
    windows = K_FULL // k_window
    m_fractals = M // 16
    k_fractals = k_window // 16

    band_ready = tla.flag("band_ready", tla.arch.MTE2, tla.arch.MTE1)
    filt_ready = tla.flag("filt_ready", tla.arch.MTE2, tla.arch.MTE1)
    filt_free = tla.flag("filt_free", tla.arch.MTE1, tla.arch.MTE2)
    l0a_ready = tla.flag("l0a_ready", tla.arch.MTE1, tla.arch.CUBE)
    l0a_free = tla.flag("l0a_free", tla.arch.CUBE, tla.arch.MTE1)
    l0b_ready = tla.flag("l0b_ready", tla.arch.MTE1, tla.arch.CUBE)
    l0b_free = tla.flag("l0b_free", tla.arch.CUBE, tla.arch.MTE1)
    l0c_ready = tla.flag("l0c_ready", tla.arch.CUBE, tla.arch.FIX)

    l1a_ptr = tla.allocate(C1 * PLANE * C0, tla.Float16, tla.AddressSpace.l1, 512)
    l1b_ptr = tla.allocate(k_window * COUT, tla.Float16, tla.AddressSpace.l1, 512)
    l0a_ptr = tla.allocate(M * k_window, tla.Float16, tla.AddressSpace.l0a, 512)
    l0b_ptr = tla.allocate(k_window * COUT, tla.Float16, tla.AddressSpace.l0b, 512)
    l0c_ptr = tla.allocate(M * COUT, tla.Float32, tla.AddressSpace.l0c, 1024)

    with tla.cube():
        tla.set_flag(filt_free)
        tla.set_flag(l0a_free)
        tla.set_flag(l0b_free)

        # The feature map arrives NC1HWC0: C1 planes of PLANE positions, each a
        # C0-wide channel block. That is the zNUnAlign nesting img2col reads,
        # and the whole feature map goes to L1 in one copy.
        gm_fmap = tla.make_tensor(
            fmap.ptr,
            tla.make_layout(
                tla.make_shape((PLANE, 1), (C0, C1)),
                tla.make_stride((C0, PLANE * C0), (1, PLANE * C0)),
                origin_shape=tla.make_shape(PLANE, CIN),
                layoutTag=tla.arch.zNUnAlign,
            ),
        )
        l1_fmap = tla.make_tensor_like(l1a_ptr, gm_fmap, tla.arch.zNUnAlign)
        tla.copy(l1_fmap, gm_fmap)
        tla.set_flag(band_ready)
        tla.wait_flag(band_ready)

        for kw in tla.range_constexpr(windows):
            k_start = kw * k_window
            # Filter rows [k_start, k_start + k_window) of the packed (K, cout)
            # matrix: the same window of the K axis the img2col below loads.
            gm_filt = tla.tile_view(
                filt,
                tla.make_shape(k_window, COUT),
                tla.make_coord(kw, 0),
            )
            tla.wait_flag(filt_free)
            l1_filt = tla.make_tensor_like(l1b_ptr, gm_filt, tla.arch.zN)
            tla.copy(l1_filt, gm_filt)
            tla.set_flag(filt_ready)
            tla.wait_flag(filt_ready)

            tla.wait_flag(l0b_free)
            l0_b = tla.make_tensor_like(l0b_ptr, l1_filt, tla.arch.nZ)
            tla.copy(l0_b, l1_filt)
            tla.set_flag(filt_free)
            tla.set_flag(l0b_ready)
            tla.wait_flag(l0b_ready)

            # img2col reads kStartPt from the destination coord, and the tile's
            # address already carries that offset, so the base is pulled back by
            # the same amount to keep every window in the one L0A buffer.
            # `_Pointer` has no __sub__, hence the negative constant.
            k_back = -(k_start // 16) * M * 16
            l0_a = tla.make_tensor(
                l0a_ptr + k_back,
                tla.make_layout(
                    tla.make_shape((16, m_fractals), (16, k_fractals)),
                    tla.make_stride((16, 256), (1, M * 16)),
                    origin_shape=tla.make_shape(M, k_window),
                    layoutTag=tla.arch.zN,
                ),
                tla.make_coord(0, k_start),
            )
            tla.wait_flag(l0a_free)
            tla.experimental.img2col(
                l0_a,
                l1_fmap,
                tla.experimental.Img2colParams(
                    fmap_hw=(HI, WI), filter_hw=(KH, KW), pad=(0, 0, 0, 0)
                ),
            )
            tla.set_flag(l0a_ready)
            tla.wait_flag(l0a_ready)

            l0_a_mmad = tla.make_tensor(
                l0a_ptr,
                tla.make_layout(
                    tla.make_shape((16, m_fractals), (16, k_fractals)),
                    tla.make_stride((16, 256), (1, M * 16)),
                    origin_shape=tla.make_shape(M, k_window),
                    layoutTag=tla.arch.zN,
                ),
            )
            gm_out = tla.tile_view(
                out, tla.make_shape(M, COUT), tla.make_coord(0, 0)
            )
            l0_c = tla.make_tensor_like(l0c_ptr, gm_out, tla.arch.L0Clayout)
            tla.mmad(l0_c, l0_a_mmad, l0_b, init_c=(kw == 0))
            tla.set_flag(l0a_free)
            tla.set_flag(l0b_free)

        tla.set_flag(l0c_ready)
        tla.wait_flag(l0c_ready)
        gm_out_final = tla.tile_view(
            out, tla.make_shape(M, COUT), tla.make_coord(0, 0)
        )
        l0_c_final = tla.make_tensor_like(l0c_ptr, gm_out_final, tla.arch.L0Clayout)
        tla.copy(gm_out_final, l0_c_final)

        tla.wait_flag(filt_free)
        tla.wait_flag(l0a_free)
        tla.wait_flag(l0b_free)


def pack_fmap(x: torch.Tensor) -> torch.Tensor:
    """(cin, hi, wi) -> (c1 * hi * wi, C0), i.e. NC1HWC0."""
    cin, hi, wi = x.shape
    return (
        x.reshape(cin // C0, C0, hi * wi)
        .permute(0, 2, 1)
        .contiguous()
        .reshape(cin // C0 * hi * wi, C0)
    )


def pack_filter(w: torch.Tensor) -> torch.Tensor:
    """(cout, cin, KH, KW) -> (c1*KH*KW*C0, cout), K ordered (c1, kh, kw, c0)."""
    cout, cin, kh, kw = w.shape
    c1 = cin // C0
    return (
        w.reshape(cout, c1, C0, kh, kw)
        .permute(1, 3, 4, 2, 0)
        .contiguous()
        .reshape(c1 * kh * kw * C0, cout)
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--tolerance", type=float, default=5e-3)
    ap.add_argument(
        "--control-shift",
        action="store_true",
        help="self-check: rotate the packed filter by one C0 block, which must "
             "make every window FAIL. A comparison that passes here is not "
             "testing anything.",
    )
    ap.add_argument(
        "--windows",
        type=int,
        nargs="*",
        default=list(WINDOWS),
        help="K window sizes to run; each must divide %d and be C0-aligned"
        % K_FULL,
    )
    args = ap.parse_args()

    torch.npu.set_device(args.device)
    torch.manual_seed(0)
    x = torch.randn(CIN, HI, WI, dtype=torch.float16)
    weight = torch.randn(COUT, CIN, KH, KW, dtype=torch.float16)

    want = (
        F.conv2d(x.float().unsqueeze(0), weight.float())[0]
        .permute(1, 2, 0)
        .reshape(M, COUT)
    )

    fmap = from_dlpack(
        pack_fmap(x).npu(),
        layout_tag=tla.arch.RowMajor,
        origin_shape=(C1 * PLANE, C0),
    )
    packed_filter = pack_filter(weight)
    if args.control_shift:
        packed_filter = packed_filter.roll(C0, dims=0).contiguous()
    filt = from_dlpack(
        packed_filter.npu(),
        layout_tag=tla.arch.RowMajor,
        origin_shape=(K_FULL, COUT),
    )

    failures = []
    for k_window in args.windows:
        if k_window % C0 or K_FULL % k_window:
            raise ValueError(
                f"K window {k_window} must be C0-aligned and divide {K_FULL}"
            )
        dev_out = torch.zeros(M, COUT, dtype=torch.float32).npu()
        out = from_dlpack(
            dev_out, layout_tag=tla.arch.RowMajor, origin_shape=(M, COUT)
        )
        artifact = tla.compile(conv_k_windows, fmap, filt, out, k_window,
                               options=OPTIONS)
        artifact(fmap, filt, out, block_num=1)
        torch.npu.synchronize()

        got = dev_out.cpu()
        err = (got - want).abs().max().item()
        rel = err / max(want.abs().max().item(), 1e-9)
        ok = rel < args.tolerance
        straddles = K_SLICE % k_window != 0 and k_window % K_SLICE != 0
        note = "  (straddles C1 boundaries)" if straddles else ""
        print(
            f"  K window {k_window:4d} x {K_FULL // k_window:3d}: "
            f"{'PASS' if ok else 'FAIL'} max_abs={err:.4g} rel={rel:.3g}{note}",
            flush=True,
        )
        if not ok:
            failures.append(k_window)

    if args.control_shift:
        # Inverted: the rotated filter must disagree everywhere.
        if len(failures) == len(args.windows):
            print("PASS (control): the shifted filter fails every window",
                  flush=True)
            return 0
        print("FAILED (control): a shifted filter still matched conv2d, so the "
              "comparison is not sensitive", flush=True)
        return 2
    if failures:
        print(f"FAILED for K windows {failures}", flush=True)
        return 2
    print("PASS: every C0-aligned K window matches conv2d", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
