# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end on-device check for ``tla.histograms``.

Covers every supported ``HistogramsParams`` combination:

* FREQUENCY × BIN0 / BIN1 (per-value half-range counts via ``dhistv2``)
* ACCUMULATE × BIN0 / BIN1 (CDF thresholds via ``chistv2``; low-half inputs
  contribute to BIN1)
* FREQUENCY into an existing ``dst`` (cross-call add-on)
* ACCUMULATE all-zero src (BIN1 lanes must be 256, not 0)
"""

from __future__ import annotations

import argparse
from typing import Any

import catlass.tla as tla

SRC_ELE = 256  # full VL for i8/u8
DST_ELE = 128  # full VL for i16 (one bin half)


@tla.kernel
def histograms_frequency_bins_op(
    mem_src: tla.Tensor, mem_dst_lo: tla.Tensor, mem_dst_hi: tla.Tensor
) -> None:
    ub_loaded = tla.flag("ub_loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    vec_done = tla.flag("vec_done", tla.pipes.VECTOR, tla.pipes.MTE3)

    src_gm = tla.tile_view(mem_src, tla.make_shape(SRC_ELE), tla.make_coord(0))
    dst_lo_gm = tla.tile_view(mem_dst_lo, tla.make_shape(DST_ELE), tla.make_coord(0))
    dst_hi_gm = tla.tile_view(mem_dst_hi, tla.make_shape(DST_ELE), tla.make_coord(0))

    src_ub = tla.make_tensor_like(
        tla.allocate(SRC_ELE, tla.Int8, tla.AddressSpace.ub, 256),
        src_gm,
        tla.arch.RowMajor,
    )
    # Host/torch side uses int16; Pass casts to ui16 for the bc stub.
    dst_lo_ub = tla.make_tensor_like(
        tla.allocate(DST_ELE, tla.Int16, tla.AddressSpace.ub, 256),
        dst_lo_gm,
        tla.arch.RowMajor,
    )
    dst_hi_ub = tla.make_tensor_like(
        tla.allocate(DST_ELE, tla.Int16, tla.AddressSpace.ub, 256),
        dst_hi_gm,
        tla.arch.RowMajor,
    )

    with tla.vector():
        tla.copy(src_ub, src_gm)

        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)

        with tla.vec.func(mode="simd"):
            src_tile = tla.tile_view(src_ub, tla.make_shape(SRC_ELE), tla.make_coord(0))
            dst_lo_tile = tla.tile_view(
                dst_lo_ub, tla.make_shape(DST_ELE), tla.make_coord(0)
            )
            dst_hi_tile = tla.tile_view(
                dst_hi_ub, tla.make_shape(DST_ELE), tla.make_coord(0)
            )

            keys = src_tile.load()
            mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Int8)
            hist_lo = tla.full(0, dtype=tla.Int16)
            hist_hi = tla.full(0, dtype=tla.Int16)
            hist_lo = tla.histograms(
                keys,
                hist_lo,
                tla.params.HistogramsParams(
                    mode=tla.params.HistogramsBinType.BIN0,
                    type=tla.params.HistogramsType.FREQUENCY,
                ),
                mask=mask,
            )
            hist_hi = tla.histograms(
                keys,
                hist_hi,
                tla.params.HistogramsParams(
                    mode=tla.params.HistogramsBinType.BIN1,
                    type=tla.params.HistogramsType.FREQUENCY,
                ),
                mask=mask,
            )
            dst_lo_tile.store(hist_lo)
            dst_hi_tile.store(hist_hi)

        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)

        tla.copy(dst_lo_gm, dst_lo_ub)
        tla.copy(dst_hi_gm, dst_hi_ub)
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def histograms_accumulate_bins_op(
    mem_src: tla.Tensor, mem_dst_lo: tla.Tensor, mem_dst_hi: tla.Tensor
) -> None:
    ub_loaded = tla.flag("ub_loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    vec_done = tla.flag("vec_done", tla.pipes.VECTOR, tla.pipes.MTE3)

    src_gm = tla.tile_view(mem_src, tla.make_shape(SRC_ELE), tla.make_coord(0))
    dst_lo_gm = tla.tile_view(mem_dst_lo, tla.make_shape(DST_ELE), tla.make_coord(0))
    dst_hi_gm = tla.tile_view(mem_dst_hi, tla.make_shape(DST_ELE), tla.make_coord(0))

    src_ub = tla.make_tensor_like(
        tla.allocate(SRC_ELE, tla.Int8, tla.AddressSpace.ub, 256),
        src_gm,
        tla.arch.RowMajor,
    )
    dst_lo_ub = tla.make_tensor_like(
        tla.allocate(DST_ELE, tla.Int16, tla.AddressSpace.ub, 256),
        dst_lo_gm,
        tla.arch.RowMajor,
    )
    dst_hi_ub = tla.make_tensor_like(
        tla.allocate(DST_ELE, tla.Int16, tla.AddressSpace.ub, 256),
        dst_hi_gm,
        tla.arch.RowMajor,
    )

    with tla.vector():
        tla.copy(src_ub, src_gm)

        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)

        with tla.vec.func(mode="simd"):
            src_tile = tla.tile_view(src_ub, tla.make_shape(SRC_ELE), tla.make_coord(0))
            dst_lo_tile = tla.tile_view(
                dst_lo_ub, tla.make_shape(DST_ELE), tla.make_coord(0)
            )
            dst_hi_tile = tla.tile_view(
                dst_hi_ub, tla.make_shape(DST_ELE), tla.make_coord(0)
            )

            keys = src_tile.load()
            mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Int8)
            hist_lo = tla.full(0, dtype=tla.Int16)
            hist_hi = tla.full(0, dtype=tla.Int16)
            hist_lo = tla.histograms(
                keys,
                hist_lo,
                tla.params.HistogramsParams(
                    mode=tla.params.HistogramsBinType.BIN0,
                    type=tla.params.HistogramsType.ACCUMULATE,
                ),
                mask=mask,
            )
            hist_hi = tla.histograms(
                keys,
                hist_hi,
                tla.params.HistogramsParams(
                    mode=tla.params.HistogramsBinType.BIN1,
                    type=tla.params.HistogramsType.ACCUMULATE,
                ),
                mask=mask,
            )
            dst_lo_tile.store(hist_lo)
            dst_hi_tile.store(hist_hi)

        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)

        tla.copy(dst_lo_gm, dst_lo_ub)
        tla.copy(dst_hi_gm, dst_hi_ub)
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def histograms_frequency_into_op(mem_src: tla.Tensor, mem_dst: tla.Tensor) -> None:
    """FREQUENCY into an existing dst (cross-call accumulate; required ``dst``)."""
    ub_loaded = tla.flag("ub_loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    vec_done = tla.flag("vec_done", tla.pipes.VECTOR, tla.pipes.MTE3)

    src_gm = tla.tile_view(mem_src, tla.make_shape(SRC_ELE), tla.make_coord(0))
    dst_gm = tla.tile_view(mem_dst, tla.make_shape(DST_ELE), tla.make_coord(0))

    src_ub = tla.make_tensor_like(
        tla.allocate(SRC_ELE, tla.Int8, tla.AddressSpace.ub, 256),
        src_gm,
        tla.arch.RowMajor,
    )
    dst_ub = tla.make_tensor_like(
        tla.allocate(DST_ELE, tla.Int16, tla.AddressSpace.ub, 256),
        dst_gm,
        tla.arch.RowMajor,
    )

    with tla.vector():
        tla.copy(src_ub, src_gm)
        tla.copy(dst_ub, dst_gm)

        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)

        with tla.vec.func(mode="simd"):
            src_tile = tla.tile_view(src_ub, tla.make_shape(SRC_ELE), tla.make_coord(0))
            dst_tile = tla.tile_view(dst_ub, tla.make_shape(DST_ELE), tla.make_coord(0))
            keys = src_tile.load()
            hist = dst_tile.load()
            hist = tla.histograms(
                keys,
                hist,
                tla.params.HistogramsParams(
                    mode=tla.params.HistogramsBinType.BIN0,
                    type=tla.params.HistogramsType.FREQUENCY,
                ),
            )
            dst_tile.store(hist)

        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)

        tla.copy(dst_gm, dst_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def _runtime_tensor(dev_buf: Any) -> Any:
    return tla.from_dlpack(
        dev_buf.contiguous(),
        layout_tag=tla.arch.RowMajor,
    )


def _u8_keys_host(src_bytes: Any) -> Any:
    import torch

    return src_bytes.detach().cpu().view(torch.uint8).to(torch.int32)


def _golden_frequency_half(src_bytes: Any, *, bin1: bool) -> Any:
    """CPU reference for FREQUENCY BIN0/BIN1 per-value counts."""
    import torch

    hist = torch.zeros(DST_ELE, dtype=torch.int32)
    vals = _u8_keys_host(src_bytes)
    for v in vals.tolist():
        if bin1:
            if 128 <= v <= 255:
                hist[v - 128] += 1
        elif 0 <= v <= 127:
            hist[v] += 1
    return hist.to(torch.int16)


def _golden_accumulate_half(src_bytes: Any, *, bin1: bool) -> Any:
    """CPU reference for ACCUMULATE CDF halves (``src <= n`` / ``src <= 128+n``)."""
    import torch

    vals = _u8_keys_host(src_bytes)
    hist = torch.zeros(DST_ELE, dtype=torch.int32)
    for n in range(DST_ELE):
        thr = 128 + n if bin1 else n
        hist[n] = int((vals <= thr).sum().item())
    return hist.to(torch.int16)


def _print_half_mismatch(
    label: str, actual: Any, expected: Any, *, ok_lo: bool, ok_hi: bool
) -> None:
    if ok_lo and ok_hi:
        return
    mism_lo = actual[0].ne(expected[0]).nonzero(as_tuple=False).flatten()[:8]
    mism_hi = actual[1].ne(expected[1]).nonzero(as_tuple=False).flatten()[:8]
    print(f"{label} first BIN0 mismatch indices={mism_lo.tolist()}")
    print(f"{label} first BIN1 mismatch indices={mism_hi.tolist()}")
    if mism_lo.numel():
        i = int(mism_lo[0].item())
        print(
            f"{label} BIN0[{i}] actual={int(actual[0][i])} expected={int(expected[0][i])}"
        )
    if mism_hi.numel():
        i = int(mism_hi[0].item())
        print(
            f"{label} BIN1[{i}] actual={int(actual[1][i])} expected={int(expected[1][i])}"
        )


def _run_dual_bins(
    device: int,
    *,
    kernel: Any,
    op_name: str,
    keys_u: Any,
    golden_fn: Any,
) -> int:
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(device)
    src = keys_u.to(dtype=torch.uint8).view(torch.int8).to(device="npu")
    dst_lo = torch.zeros(DST_ELE, dtype=torch.int16, device="npu")
    dst_hi = torch.zeros(DST_ELE, dtype=torch.int16, device="npu")

    tla_src = _runtime_tensor(src)
    tla_lo = _runtime_tensor(dst_lo)
    tla_hi = _runtime_tensor(dst_hi)

    artifact = tla.compile(
        kernel,
        tla_src,
        tla_lo,
        tla_hi,
        options="--npu-arch 3510",
    )
    artifact(tla_src, tla_lo, tla_hi, block_num=1)
    torch.npu.synchronize()

    exp_lo = golden_fn(src, bin1=False).to(device="npu")
    exp_hi = golden_fn(src, bin1=True).to(device="npu")
    ok_lo = bool(dst_lo.eq(exp_lo).all())
    ok_hi = bool(dst_hi.eq(exp_hi).all())
    ok = ok_lo and ok_hi

    print(f"compile_ok=True host=torch_npu op={op_name} dtype=i8->i16 layout=row")
    print(f"kernel.o path={artifact.kernel_binary_path}")
    print("launch_ok=True")
    print(f"BIN0 equals expected? {ok_lo}")
    print(f"BIN1 equals expected? {ok_hi}")
    _print_half_mismatch(
        op_name, (dst_lo, dst_hi), (exp_lo, exp_hi), ok_lo=ok_lo, ok_hi=ok_hi
    )
    return 0 if ok else 1


def _run_frequency_bins(device: int) -> int:
    import torch

    keys_u = (torch.arange(SRC_ELE, dtype=torch.int32) * 3 + 7) % 256
    return _run_dual_bins(
        device,
        kernel=histograms_frequency_bins_op,
        op_name="histograms_frequency_bins",
        keys_u=keys_u,
        golden_fn=_golden_frequency_half,
    )


def _run_accumulate_bins(device: int) -> int:
    import torch

    # Mixed keys + AscendC-style identity pattern [0..255].
    keys_mixed = (torch.arange(SRC_ELE, dtype=torch.int32) * 3 + 7) % 256
    keys_identity = torch.arange(SRC_ELE, dtype=torch.int32)
    rc = _run_dual_bins(
        device,
        kernel=histograms_accumulate_bins_op,
        op_name="histograms_accumulate_bins_mixed",
        keys_u=keys_mixed,
        golden_fn=_golden_accumulate_half,
    )
    rc |= _run_dual_bins(
        device,
        kernel=histograms_accumulate_bins_op,
        op_name="histograms_accumulate_bins_identity",
        keys_u=keys_identity,
        golden_fn=_golden_accumulate_half,
    )
    # Reviewer counterexample: all zeros → ACCUMULATE BIN1 lanes are 256.
    rc |= _run_dual_bins(
        device,
        kernel=histograms_accumulate_bins_op,
        op_name="histograms_accumulate_bins_all_zero",
        keys_u=torch.zeros(SRC_ELE, dtype=torch.int32),
        golden_fn=_golden_accumulate_half,
    )
    return rc


def _run_frequency_into(device: int) -> int:
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(device)
    keys_u = (torch.arange(SRC_ELE, dtype=torch.int32) * 5 + 11) % 128
    src = keys_u.to(dtype=torch.uint8).view(torch.int8).to(device="npu")
    # Seed accumulator with 1 in every bin; FREQUENCY should add this round's counts.
    dst = torch.ones(DST_ELE, dtype=torch.int16, device="npu")

    tla_src = _runtime_tensor(src)
    tla_dst = _runtime_tensor(dst)

    artifact = tla.compile(
        histograms_frequency_into_op,
        tla_src,
        tla_dst,
        options="--npu-arch 3510",
    )
    artifact(tla_src, tla_dst, block_num=1)
    torch.npu.synchronize()

    exp = _golden_frequency_half(src, bin1=False).to(device="npu") + 1
    ok = bool(dst.eq(exp).all())
    print(
        "compile_ok=True host=torch_npu op=histograms_frequency_into "
        "dtype=i8->i16 layout=row"
    )
    print(f"kernel.o path={artifact.kernel_binary_path}")
    print("launch_ok=True")
    print(f"output equals expected frequency+seed? {ok}")
    if not ok:
        mism = dst.ne(exp).nonzero(as_tuple=False).flatten()[:8]
        print(f"first mismatch indices={mism.tolist()}")
        if mism.numel():
            i = int(mism[0].item())
            print(f"dst[{i}] actual={int(dst[i])} expected={int(exp[i])}")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="On-device e2e for tla.histograms (all FREQUENCY/ACCUMULATE modes)."
    )
    parser.add_argument("--run", action="store_true", help="Compile and launch on NPU")
    parser.add_argument(
        "--mode",
        choices=("frequency", "accumulate", "frequency_into", "all"),
        default="all",
        help="Which kernel path to exercise (default: all)",
    )
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("pass --run to compile and execute on NPU")

    rc = 0
    if args.mode in ("frequency", "all"):
        rc |= _run_frequency_bins(args.device)
    if args.mode in ("accumulate", "all"):
        rc |= _run_accumulate_bins(args.device)
    if args.mode in ("frequency_into", "all"):
        rc |= _run_frequency_into(args.device)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
