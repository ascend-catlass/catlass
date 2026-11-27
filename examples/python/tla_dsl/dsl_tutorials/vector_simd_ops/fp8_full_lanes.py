# -------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -------------------------------------------------------------------------------------------
"""Numerically check every lane of the fp8 vector casts.

Two cases, because they fail independently:

``roundtrip`` sends a full f16 register through fp8 and back. It is the
obvious test and the weaker one: both legs are composed through f32 with
the same packing selector, so a wrong selector cancels itself and the
round trip is the identity for *any* input. It cannot see a packing bug.

``masked`` repeats ``widen`` under ``tla.mask.M3``, a strided pattern, at BOTH
reg_slots. A contiguous mask survives a change of lane domain and an all-true one
is blind to it, so only a strided mask can tell whether the predicate reached
every instruction of a multi-instruction cast. Both slots are run because the
mask and the slot interact: e5m2 -> f16 takes a different lowering from the other
fp8 pairs, and it failed at slot ONE while passing at slot ZERO.

``widen`` reads a dense fp8 buffer -- every one of the 256 bit patterns,
not values synthesised from f16 -- and widens it to f16 in one ``.to()``.
Nothing cancels here, so the destination lanes pin the selector: with
reg_slot ZERO the cast must return ``src[2j]``, the even elements. A leg
built from the 2x even/odd part instead of the 4x pack pattern returns a
strided half of the source and is caught on almost every lane.
"""

from __future__ import annotations

import argparse

import catlass.tla as tla
from catlass.tla.runtime import from_dlpack

N = 128
_FP8 = tla.Float8E4M3FN
_FP8_OF = {"e4m3": tla.Float8E4M3FN, "e5m2": tla.Float8E5M2}


def _cast_params(slot):
    return tla.params.CastParams(
        reg_slot=slot,
        sat_mode=tla.params.SatMode.NOSAT,
        round_mode=tla.params.RoundMode.CAST_ROUND,
    )


_CAST = tla.params.CastParams(
    # reg_slot selects which elements a width-changing cast keeps, exactly as
    # it does for every other 2x widen/narrow: ZERO takes the even ones, so
    # fp8 -> f16 returns src[2j] and f16 -> fp8 writes dst[2j].
    reg_slot=tla.params.RegSlot.ZERO,
    sat_mode=tla.params.SatMode.NOSAT,
    round_mode=tla.params.RoundMode.CAST_ROUND,
)


@tla.kernel
def fp8_full_lanes(gm_in: tla.Tensor, gm_out: tla.Tensor) -> None:
    loaded = tla.flag("loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    done = tla.flag("done", tla.pipes.VECTOR, tla.pipes.MTE3)
    ub_in = tla.make_tensor_like(
        tla.allocate(N, tla.Float16, tla.AddressSpace.ub, 256), gm_in, tla.arch.RowMajor
    )
    ub_out = tla.make_tensor_like(
        tla.allocate(N, tla.Float16, tla.AddressSpace.ub, 256),
        gm_out,
        tla.arch.RowMajor,
    )
    with tla.vector():
        tla.copy(ub_in, gm_in)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        with tla.vec.func(mode="simd"):
            values = tla.tile_view(ub_in, tla.make_shape(N), tla.make_coord(0)).load()
            result = values.to(_FP8, _CAST).to(tla.Float16, _CAST)
            tla.tile_view(ub_out, tla.make_shape(N), tla.make_coord(0)).store(result)
        tla.set_flag(done)
        tla.wait_flag(done)
        tla.copy(gm_out, ub_out)
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def fp8_dense_widen(gm_in: tla.Tensor, gm_out: tla.Tensor) -> None:
    """Widen a dense fp8 register to f16 -- one cast, nothing to cancel."""
    loaded = tla.flag("loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    done = tla.flag("done", tla.pipes.VECTOR, tla.pipes.MTE3)
    ub_in = tla.make_tensor_like(
        tla.allocate(2 * N, _FP8, tla.AddressSpace.ub, 256), gm_in, tla.arch.RowMajor
    )
    ub_out = tla.make_tensor_like(
        tla.allocate(N, tla.Float16, tla.AddressSpace.ub, 256),
        gm_out,
        tla.arch.RowMajor,
    )
    with tla.vector():
        tla.copy(ub_in, gm_in)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        with tla.vec.func(mode="simd"):
            values = tla.tile_view(
                ub_in, tla.make_shape(2 * N), tla.make_coord(0)
            ).load()
            result = values.to(tla.Float16, _CAST)
            tla.tile_view(ub_out, tla.make_shape(N), tla.make_coord(0)).store(result)
        tla.set_flag(done)
        tla.wait_flag(done)
        tla.copy(gm_out, ub_out)
        tla.pipe_barrier(tla.pipes.ALL)


def make_masked_widen(slot):
    """Widen a dense fp8 register to f16 under a strided (M3) mask."""
    params = _cast_params(slot)

    @tla.kernel
    def fp8_masked_widen(gm_in: tla.Tensor, gm_out: tla.Tensor) -> None:
        loaded = tla.flag("loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
        done = tla.flag("done", tla.pipes.VECTOR, tla.pipes.MTE3)
        ub_in = tla.make_tensor_like(
            tla.allocate(2 * N, _FP8, tla.AddressSpace.ub, 256),
            gm_in,
            tla.arch.RowMajor,
        )
        ub_out = tla.make_tensor_like(
            tla.allocate(N, tla.Float16, tla.AddressSpace.ub, 256),
            gm_out,
            tla.arch.RowMajor,
        )
        with tla.vector():
            tla.copy(ub_in, gm_in)
            tla.set_flag(loaded)
            tla.wait_flag(loaded)
            with tla.vec.func(mode="simd"):
                values = tla.tile_view(
                    ub_in, tla.make_shape(2 * N), tla.make_coord(0)
                ).load()
                m = tla.create_mask(pattern=tla.mask.M3, dtype=_FP8)
                result = values.to(tla.Float16, params, m)
                tla.tile_view(ub_out, tla.make_shape(N), tla.make_coord(0)).store(
                    result
                )
            tla.set_flag(done)
            tla.wait_flag(done)
            tla.copy(gm_out, ub_out)
            tla.pipe_barrier(tla.pipes.ALL)

    return fp8_masked_widen


def _report(name: str, fmt: str, actual, expected) -> bool:
    import torch

    a = actual.view(torch.int16)
    e = expected.view(torch.int16)
    matches = torch.equal(a, e)
    print(f"fmt={fmt} case={name} lanes={a.numel()} passed={matches}")
    mismatch = (a != e).nonzero().flatten()
    if mismatch.numel():
        print(f"mismatch_count={mismatch.numel()} indices={mismatch[:16].tolist()}")
        for index in mismatch[:8].tolist():
            print(
                f"  lane={index} actual=0x{int(a[index]) & 0xFFFF:04x} "
                f"expected=0x{int(e[index]) & 0xFFFF:04x}"
            )
    return matches


def run(args: argparse.Namespace) -> int:
    import torch
    import torch_npu
    from catlass.bc_compile import init

    global _FP8
    _FP8 = _FP8_OF[args.fmt]
    torch_fp8 = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}[args.fmt]
    torch.npu.set_device(args.device)
    init()

    # Pick finite values already representable by the selected fp8 format.
    # They are distinct across both halves, so a one-part conversion fails
    # loudly, while avoiding host/device round-tie policy differences.
    raw = torch.arange(256, dtype=torch.uint8)
    candidates = raw.view(torch_fp8).to(torch.float16)
    src = (
        candidates[torch.isfinite(candidates.to(torch.float32))][:N].contiguous().npu()
    )
    out = torch.full((N,), -99.0, dtype=torch.float16, device="npu")
    t_in = from_dlpack(src, layout_tag=tla.arch.RowMajor).mark_compact_shape_dynamic(0)
    t_out = from_dlpack(out, layout_tag=tla.arch.RowMajor).mark_compact_shape_dynamic(0)
    artifact = tla.compile(fp8_full_lanes, t_in, t_out, options="--npu-arch 3510")
    artifact(t_in, t_out, block_num=1)
    torch.npu.synchronize()

    ok = _report(
        "roundtrip", args.fmt, out.cpu(), src.cpu().to(torch_fp8).to(torch.float16)
    )

    # Case 2: a dense fp8 source. Every bit pattern the format defines, with
    # the non-finite ones zeroed so the comparison stays exact.
    dense = raw.view(torch_fp8)
    finite = torch.isfinite(dense.to(torch.float32))
    dense = torch.where(finite, dense.to(torch.float32), torch.zeros(())).to(torch_fp8)
    wide_out = torch.full((N,), -99.0, dtype=torch.float16, device="npu")
    t_in = from_dlpack(
        dense.npu().view(torch.int8),
        layout_tag=tla.arch.RowMajor,
        origin_shape=(2 * N,),
        element_type=_FP8,
    ).mark_compact_shape_dynamic(0)
    t_out = from_dlpack(
        wide_out, layout_tag=tla.arch.RowMajor
    ).mark_compact_shape_dynamic(0)
    tla.compile(fp8_dense_widen, t_in, t_out, options="--npu-arch 3510")(
        t_in, t_out, block_num=1
    )
    torch.npu.synchronize()

    # reg_slot ZERO keeps the even elements of the source register.
    expected_wide = dense.to(torch.float16)[0::2].contiguous()
    ok = _report("widen", args.fmt, wide_out.cpu(), expected_wide) and ok

    # Case 3: the same widen under tla.mask.M3. reg_slot ZERO reads source
    # element 2j for destination lane j, so lane j survives iff 2j is a
    # multiple of 3; every other lane must come back zero.
    dense_f16 = dense.to(torch.float16)
    for slot_name, slot in (
        ("ZERO", tla.params.RegSlot.ZERO),
        ("ONE", tla.params.RegSlot.ONE),
    ):
        si = 0 if slot_name == "ZERO" else 1
        masked_out = torch.full((N,), -99.0, dtype=torch.float16, device="npu")
        t_out = from_dlpack(
            masked_out, layout_tag=tla.arch.RowMajor
        ).mark_compact_shape_dynamic(0)
        tla.compile(make_masked_widen(slot), t_in, t_out, options="--npu-arch 3510")(
            t_in, t_out, block_num=1
        )
        torch.npu.synchronize()
        # slot si reads source element 2j+si for destination lane j; M3 keeps it
        # only when that index is a multiple of 3.
        expected_masked = torch.tensor(
            [
                float(dense_f16[2 * j + si]) if (2 * j + si) % 3 == 0 else 0.0
                for j in range(N)
            ],
            dtype=torch.float16,
        )
        ok = (
            _report(
                f"masked slot {slot_name}", args.fmt, masked_out.cpu(), expected_masked
            )
            and ok
        )

    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fmt", choices=tuple(_FP8_OF), default="e4m3")
    parser.add_argument("--device", type=int, default=0)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
