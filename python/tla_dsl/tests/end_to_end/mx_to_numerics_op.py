# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Numerical device regressions for ``tla.experimental.mx_to`` quantization.

These check the two properties that an IR test cannot see, because both are
decided by the arithmetic the meta-op emits rather than by which meta-op is
selected:

NaN preservation
    The OCP MX scale is E8M0 and reserves ``0xFF`` for NaN; a group holding a
    NaN must carry that scale, so the whole group decodes as NaN. The block
    scale is otherwise derived from the exponent of the group's ``|max|``, and
    a NaN's biased exponent is 255 -- which, left alone, yields a perfectly
    finite scale (``255 - 8 = 247`` for E4M3, ``255 - 15 = 240`` for E5M2) and
    a payload that casts to zero, so the NaN reads back as ``0.0``.

Midpoint rounding
    The hardware cast rounds ties to even. ``CAST_ROUND`` asks for ties away
    from zero, so quantization corrects them -- and that correction must not
    leak into ``CAST_RINT`` or ``CAST_NONE``, which want the native result.
    1.0625 is the exact midpoint between the E4M3 codes 1.0 and 1.125, so it
    separates the two: ties-to-even keeps 1.0, ties-away gives 1.125.

Run with ``--format`` / ``--mode`` to narrow, otherwise every case runs.
"""

from __future__ import annotations

import argparse
import sys

from catlass import tla
from catlass.params import RoundMode
from catlass.tla.runtime import from_dlpack

GROUP = 32
COLS = 64  # two MX groups, so a NaN in the first must not reach the second

_MX_DTYPE = {"e4m3": tla.Float8E4M3FN, "e5m2": tla.Float8E5M2}
# The finite scale the old, NaN-blind code produced: 255 - emax.
_NAN_BLIND_SCALE = {"e4m3": 247, "e5m2": 240}


def quantize_kernel(mx_name: str, round_mode: RoundMode):
    """f32 (1, COLS) in GM -> MX payload + E8M0 scale back out to GM."""
    mx_dtype = _MX_DTYPE[mx_name]

    @tla.kernel
    def k(gm_src, gm_payload, gm_scale):
        ub_src = tla.allocate(COLS, tla.Float32, tla.AddressSpace.ub, 512)
        ub_pay = tla.allocate(COLS, tla.Int8, tla.AddressSpace.ub, 512)
        ub_scl = tla.allocate(COLS // GROUP, tla.Int8, tla.AddressSpace.ub, 512)

        dense = tla.make_layout(
            tla.make_shape(1, COLS),
            tla.make_stride(COLS, 1),
            origin_shape=tla.make_shape(1, COLS),
            layoutTag=tla.arch.RowMajor,
        )
        scale_l = tla.make_layout(
            tla.make_shape(1, COLS // GROUP),
            tla.make_stride(COLS // GROUP, 1),
            origin_shape=tla.make_shape(1, COLS // GROUP),
            layoutTag=tla.arch.RowMajor,
        )

        # The load, the conversion and the store sit on three different pipes
        # (MTE2, VECTOR, MTE3), so each hand-off needs its own flag or the
        # store races the conversion and reads whatever UB held before.
        loaded = tla.flag("loaded", tla.arch.MTE2, tla.arch.VECTOR)
        converted = tla.flag("converted", tla.arch.VECTOR, tla.arch.MTE3)

        with tla.vector():
            t_src = tla.make_tensor(ub_src, dense)
            tla.copy(t_src, gm_src)
            tla.set_flag(loaded)
            tla.wait_flag(loaded)

            payload = tla.make_tensor(
                tla.recast_ptr(ub_pay, dtype=mx_dtype),
                tla.make_layout(tla.make_shape(1, COLS), tla.make_stride(COLS, 1)),
            )
            scale = tla.make_tensor(
                tla.recast_ptr(ub_scl, dtype=tla.Float8E8M0),
                tla.make_layout(
                    tla.make_shape(1, COLS // GROUP),
                    tla.make_stride(COLS // GROUP, 1),
                ),
            )
            tla.experimental.mx_to(t_src, (payload, scale), round_mode=round_mode)
            tla.set_flag(converted)
            tla.wait_flag(converted)

            # Hand both buffers back as bytes; the host decodes them.
            tla.copy(gm_payload, tla.make_tensor(ub_pay, dense))
            tla.copy(gm_scale, tla.make_tensor(ub_scl, scale_l))

    return k


def _decode_e8m0(byte: int) -> float:
    """E8M0 shared exponent -> float multiplier. 0xFF is NaN, 0x00 is 2**-127."""
    if byte == 0xFF:
        return float("nan")
    return 2.0 ** (byte - 127)


def _decode_fp8(byte: int, mx_name: str) -> float:
    """Decode one E4M3 (OCP fp8e4m3fn) or E5M2 byte."""
    sign = -1.0 if byte & 0x80 else 1.0
    if mx_name == "e4m3":
        exp, man, bias, mbits = (byte >> 3) & 0x0F, byte & 0x07, 7, 3
        if exp == 0x0F and man == 0x07:
            return float("nan")  # e4m3fn: the single NaN code
    else:
        exp, man, bias, mbits = (byte >> 2) & 0x1F, byte & 0x03, 15, 2
        if exp == 0x1F:
            return float("nan") if man else sign * float("inf")
    if exp == 0:
        return sign * man / (1 << mbits) * 2.0 ** (1 - bias)
    return sign * (1 + man / (1 << mbits)) * 2.0 ** (exp - bias)


def _run(mx_name: str, round_mode: RoundMode, values: list[float], torch):
    src = torch.tensor([values], dtype=torch.float32, device="npu")
    payload = torch.zeros((1, COLS), dtype=torch.int8, device="npu")
    scale = torch.zeros((1, COLS // GROUP), dtype=torch.int8, device="npu")
    tensors = (
        from_dlpack(
            src, layout_tag=tla.arch.RowMajor, origin_shape=(1, COLS)
        ).mark_layout_dynamic(),
        from_dlpack(
            payload, layout_tag=tla.arch.RowMajor, origin_shape=(1, COLS)
        ).mark_layout_dynamic(),
        from_dlpack(
            scale, layout_tag=tla.arch.RowMajor, origin_shape=(1, COLS // GROUP)
        ).mark_layout_dynamic(),
    )
    artifact = tla.compile(
        quantize_kernel(mx_name, round_mode), *tensors, options="--npu-arch 3510"
    )
    artifact(*tensors, block_num=1)
    torch.npu.synchronize()
    p = [b & 0xFF for b in payload.detach().cpu().flatten().tolist()]
    s = [b & 0xFF for b in scale.detach().cpu().flatten().tolist()]
    return p, s


def check_nan(mx_name: str, torch) -> list[str]:
    """A NaN anywhere in group 0 must make group 0 NaN and leave group 1 alone."""
    fails = []
    values = [float("nan")] + [1.0] * (COLS - 1)
    payload, scale = _run(mx_name, RoundMode.CAST_ROUND, values, torch)

    if scale[0] != 0xFF:
        fails.append(
            f"{mx_name}: group 0 holds a NaN but its scale byte is {scale[0]} "
            f"(0x{scale[0]:02x}), not the E8M0 NaN code 255"
            + (
                "  <- exactly the NaN-blind value 255-emax"
                if scale[0] == _NAN_BLIND_SCALE[mx_name]
                else ""
            )
        )
    # Decoding element 0 must give NaN, not 0.0.
    first = _decode_e8m0(scale[0]) * _decode_fp8(payload[0], mx_name)
    if first == first:  # NaN is the only value unequal to itself
        fails.append(f"{mx_name}: element 0 decodes to {first}, expected NaN")
    # The second group shares no scale with the first, so it must be untouched.
    if scale[1] == 0xFF:
        fails.append(f"{mx_name}: the NaN in group 0 leaked into group 1's scale")
    else:
        second = _decode_e8m0(scale[1]) * _decode_fp8(payload[GROUP], mx_name)
        if abs(second - 1.0) > 1e-6:
            fails.append(
                f"{mx_name}: group 1 element decodes to {second}, expected 1.0"
            )
    return fails


# 1.0625 is the midpoint between the E4M3 codes 0x78 (1.0) and 0x79 (1.125).
# Every group member shares a scale, so keeping the rest at 1.0 leaves the
# block scale at 2**0 and the payload directly comparable to those codes.
_MIDPOINT_EXPECT = {
    RoundMode.CAST_ROUND: (0x79, 1.125),  # ties away from zero
    RoundMode.CAST_RINT: (0x78, 1.0),  # ties to even
    RoundMode.CAST_NONE: (0x78, 1.0),  # documented as RINT when lossy
}


def check_midpoint(mx_name: str, round_mode: RoundMode, torch) -> list[str]:
    if mx_name != "e4m3":
        return []  # the code pair above is E4M3-specific
    want_code, want_value = _MIDPOINT_EXPECT[round_mode]
    values = [1.0625] + [1.0] * (COLS - 1)
    payload, scale = _run(mx_name, round_mode, values, torch)
    got_value = _decode_e8m0(scale[0]) * _decode_fp8(payload[0], mx_name)
    if payload[0] != want_code:
        return [
            f"{round_mode}: 1.0625 -> E4M3 payload 0x{payload[0]:02x} "
            f"(decodes {got_value}), expected 0x{want_code:02x} ({want_value})"
        ]
    return []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--format", choices=sorted(_MX_DTYPE), action="append")
    ap.add_argument(
        "--mode",
        choices=[m.name for m in _MIDPOINT_EXPECT],
        action="append",
        help="restrict the midpoint cases to these round modes",
    )
    ap.add_argument("--device", type=int, default=0, help="NPU card to run on")
    args = ap.parse_args()
    formats = args.format or sorted(_MX_DTYPE)
    modes = [RoundMode[m] for m in (args.mode or [m.name for m in _MIDPOINT_EXPECT])]

    import torch
    import torch_npu  # noqa: F401 -- register the backend in the device entry

    torch.npu.set_device(args.device)

    fails: list[str] = []
    for mx_name in formats:
        got = check_nan(mx_name, torch)
        print(f"nan      {mx_name:5s} {'FAIL' if got else 'ok'}")
        fails += got
    for mx_name in formats:
        for mode in modes:
            got = check_midpoint(mx_name, mode, torch)
            if mx_name == "e4m3":
                print(
                    f"midpoint {mx_name:5s} {mode.name:11s} {'FAIL' if got else 'ok'}"
                )
            fails += got

    for f in fails:
        print(f"  FAIL: {f}", file=sys.stderr)
    print(f"\n{len(fails)} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
