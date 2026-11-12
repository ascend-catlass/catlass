# -------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and
# conditions of CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the
# License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
# PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -------------------------------------------------------------------------------------------------

"""Frontend surface for CAST_NONE / CAST_RINT / CAST_ODD on VectorSSA.to."""

from __future__ import annotations

import pytest

import catlass.tla as tla
from catlass.tla.runtime import make_fake_tensor

_LANES = 64


def _ub(dtype: type[tla.Numeric]) -> tla.Tensor:
    return make_fake_tensor(
        dtype,
        (_LANES,),
        (1,),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(_LANES,),
        layout_tag=tla.arch.RowMajor,
    )


def _cast_kernel(dst_dtype: type[tla.Numeric], params: tla.params.CastParams):
    @tla.kernel
    def k(src: tla.Tensor, dst: tla.Tensor) -> None:
        src_t = tla.tile_view(src, tla.make_shape(_LANES), tla.make_coord(0))
        dst_t = tla.tile_view(dst, tla.make_shape(_LANES), tla.make_coord(0))
        with tla.vector():
            with tla.vec.func(mode="simd"):
                dst_t.store(src_t.load().to(dst_dtype, params))

    return k


@pytest.mark.parametrize(
    "mode, code",
    [
        (tla.params.RoundMode.CAST_ROUND, 0),
        (tla.params.RoundMode.CAST_FLOOR, 1),
        (tla.params.RoundMode.CAST_CEIL, 2),
        (tla.params.RoundMode.CAST_TRUNC, 3),
        (tla.params.RoundMode.CAST_NONE, 4),
        (tla.params.RoundMode.CAST_RINT, 5),
        (tla.params.RoundMode.CAST_ODD, 6),
        (tla.params.RoundMode.CAST_HYBRID, 7),
    ],
)
def test_round_mode_trait_codes(mode, code) -> None:
    assert tla.params.CastParams(round_mode=mode).codes()[2] == code


@pytest.mark.parametrize(
    "mode",
    (
        tla.params.RoundMode.CAST_NONE,
        tla.params.RoundMode.CAST_RINT,
        tla.params.RoundMode.CAST_ODD,
    ),
)
def test_new_round_modes_emit_on_same_type_f32(mode) -> None:
    params = tla.params.CastParams(
        reg_slot=tla.params.RegSlot.ZERO,
        sat_mode=tla.params.SatMode.UNKNOWN,
        round_mode=mode,
    )
    mlir = _cast_kernel(tla.Float32, params).dump_mlir(
        type_args=(_ub(tla.Float32), _ub(tla.Float32))
    )
    assert "tla.cast" in mlir


def test_cast_none_must_be_explicit_in_params() -> None:
    # AscendC-compatible: NONE is never the silent default.
    assert tla.params.CastParams().round_mode is tla.params.RoundMode.CAST_ROUND
    assert tla.params.CastParams().round_mode is not tla.params.RoundMode.CAST_NONE
