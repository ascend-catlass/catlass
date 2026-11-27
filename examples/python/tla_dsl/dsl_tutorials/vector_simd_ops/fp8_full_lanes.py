# -------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -------------------------------------------------------------------------------------------
"""Check FP8-to-Float16 widening for every byte encoding and both slots."""

from __future__ import annotations

import argparse

import catlass.tla as tla
from catlass.tla.runtime import from_dlpack

N = 128
_FP8_FORMATS = {"e4m3": tla.Float8E4M3FN, "e5m2": tla.Float8E5M2}


def make_widen(fp8_type, slot, masked):
    params = tla.params.CastParams(reg_slot=slot)

    @tla.kernel
    def fp8_widen(gm_in: tla.Tensor, gm_out: tla.Tensor) -> None:
        loaded = tla.flag("loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
        done = tla.flag("done", tla.pipes.VECTOR, tla.pipes.MTE3)
        ub_in = tla.make_tensor_like(
            tla.allocate(2 * N, fp8_type, tla.AddressSpace.ub, 256),
            gm_in,
            tla.arch.RowMajor,
        )
        ub_out = tla.make_tensor_like(
            tla.allocate(N, tla.Float16, tla.AddressSpace.ub, 256),
            gm_out,
            tla.arch.RowMajor,
        )
        with tla.vector():
            tla.arch.set_ctrl(
                tla.params.CTRLValue(
                    saturation_strategy=tla.params.OverrideStrategy.USE_API
                )
            )
            tla.copy(ub_in, gm_in)
            tla.set_flag(loaded)
            tla.wait_flag(loaded)
            with tla.vec.func(mode="simd"):
                values = tla.tile_view(
                    ub_in, tla.make_shape(2 * N), tla.make_coord(0)
                ).load()
                mask = None
                if tla.const_expr(masked):
                    mask = tla.create_mask(pattern=tla.mask.M3, dtype=fp8_type)
                result = values.to(tla.Float16, params, mask)
                tla.tile_view(ub_out, tla.make_shape(N), tla.make_coord(0)).store(
                    result
                )
            tla.set_flag(done)
            tla.wait_flag(done)
            tla.copy(gm_out, ub_out)
            tla.pipe_barrier(tla.pipes.ALL)

    return fp8_widen


def run(args: argparse.Namespace) -> int:
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    fp8_type = _FP8_FORMATS[args.fmt]
    raw = torch.arange(2 * N, dtype=torch.uint8)
    if fp8_type is tla.Float8E5M2:
        # Check every bit, including the NaN payload.
        expected = raw.to(torch.int16) << 8
    else:
        expected = raw.view(torch.float8_e4m3fn).to(torch.float16).view(torch.int16)

    dense = raw.view(torch.int8).npu()
    wide = torch.full((N,), -99.0, dtype=torch.float16, device="npu")
    t_in = from_dlpack(dense, layout_tag=tla.arch.RowMajor, element_type=fp8_type)
    t_out = from_dlpack(wide, layout_tag=tla.arch.RowMajor)

    for offset, slot in enumerate((tla.params.RegSlot.ZERO, tla.params.RegSlot.ONE)):
        indices = torch.arange(offset, 2 * N, 2)
        selected = expected[indices]
        for masked in (False, True):
            wide.fill_(-99.0)
            artifact = tla.compile(
                make_widen(fp8_type, slot, masked),
                t_in,
                t_out,
                options="--npu-arch 3510",
            )
            artifact(t_in, t_out, block_num=1)
            torch.npu.synchronize()
            actual = wide.cpu().view(torch.int16)
            reference = (
                torch.where(indices % 3 == 0, selected, 0) if masked else selected
            )
            if fp8_type is tla.Float8E4M3FN:
                # Compare NaNs by classification after conversion through Float32.
                both_nan = (
                    actual.view(torch.float16).isnan()
                    & reference.view(torch.float16).isnan()
                )
                actual = torch.where(both_nan, reference, actual)
            torch.testing.assert_close(actual, reference)
            print(
                f"{fp8_type.__name__} to Float16: slot={slot.name}, M3 mask={masked} passed"
            )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fmt", choices=tuple(_FP8_FORMATS), default="e4m3")
    parser.add_argument("--device", type=int, default=0)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
