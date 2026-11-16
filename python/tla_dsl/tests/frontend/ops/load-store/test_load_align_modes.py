# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Region staging requires nested, single-item with statements.
# ruff: noqa: SIM117

from __future__ import annotations

from typing import Any

import pytest
from catlass import tla
from catlass.core_api import MaskSSA, VectorSSA
from catlass.execution_lowering import TlaLoweringError, UnsupportedExecutionLowering
from catlass.params import (
    LoadDist,
    MaskLoadDist,
    MaskLoadParams,
    NormalLoadParams,
    PostMode,
)
from catlass.tla.runtime import make_fake_tensor

_BYTE_TYPES = (
    (tla.Int8, "i8"),
    (tla.UInt8, "ui8"),
    (tla.Float8E4M3FN, "f8E4M3FN"),
    (tla.Float8E5M2, "f8E5M2"),
)
_HALFWORD_TYPES = (
    (tla.Int16, "i16"),
    (tla.UInt16, "ui16"),
    (tla.Float16, "f16"),
    (tla.BFloat16, "bf16"),
)
_WORD_TYPES = (
    (tla.Int32, "i32"),
    (tla.UInt32, "ui32"),
    (tla.Float32, "f32"),
)
_MODES = (
    (LoadDist.DIST_BRC_B8, _BYTE_TYPES, 1, 256),
    (LoadDist.DIST_DS_B8, _BYTE_TYPES, 512, 256),
    (LoadDist.DIST_DS_B16, _HALFWORD_TYPES, 256, 128),
    (LoadDist.DIST_UNPACK_B32, _WORD_TYPES, 32, 64),
    (LoadDist.DIST_UNPACK_B8, _BYTE_TYPES, 128, 256),
)


def _tensor(dtype: type[tla.Numeric], extent: int, *, addrspace=tla.AddressSpace.ub):
    return make_fake_tensor(
        dtype,
        (extent,),
        (1,),
        addrspace=addrspace,
        origin_shape=(extent,),
        layout_tag=tla.arch.RowMajor,
    )


def _load_kernel(mode: str, source_view: int):
    @tla.kernel
    def single(source: tla.Tensor, destination: tla.Tensor):
        source_tile = tla.tile_view(
            source, tla.make_shape(source_view), tla.make_coord(0)
        )
        with tla.vector():
            with tla.vec.func(mode="simd"):
                value = source_tile.load(NormalLoadParams(load_dist=mode))
                assert isinstance(value, VectorSSA)
                destination.store(value)

    return single


@pytest.mark.parametrize(
    "mode,dtype,token,source_extent,lanes",
    [
        pytest.param(mode, dtype, token, count, lanes, id=f"{mode}-{token}")
        for mode, types, count, lanes in _MODES
        for dtype, token in types
    ],
)
@pytest.mark.parametrize("use_single_element_view", (False, True))
def test_load_align_result_is_full_register(
    compiler_tlair: Any,
    mode: str,
    dtype: type[tla.Numeric],
    token: str,
    source_extent: int,
    lanes: int,
    use_single_element_view: bool,
) -> None:
    # The tile supplies an address; distribution controls the actual read span.
    source_view = 1 if use_single_element_view else source_extent
    args = (_tensor(dtype, source_extent), _tensor(dtype, lanes))
    ir = compiler_tlair(_load_kernel(mode, source_view), type_args=args)
    loads = [line for line in ir.splitlines() if '"tla.load"' in line]
    assert len(loads) == 1
    assert f"#tla.load_dist<{mode}>" in loads[0]
    result_types = loads[0].split("->", 1)[1]
    assert result_types.count(f"!tla.vector<{lanes}x{token}>") == 1
    assert "!tla.mask" not in result_types


@pytest.mark.parametrize(
    "mode,dtype",
    [
        pytest.param(mode, dtype, id=f"{mode}-{dtype.__name__}")
        for mode, types, _, _ in _MODES
        for dtype in (tla.Bool, tla.Float8E8M0, tla.Int8, tla.Int16, tla.Float32)
        if dtype not in tuple(candidate for candidate, _ in types)
    ],
)
def test_load_align_rejects_unsupported_element_type(
    compiler_tlair: Any, mode: str, dtype: type[tla.Numeric]
) -> None:
    args = (_tensor(dtype, 512), _tensor(dtype, 256))
    with pytest.raises(TlaLoweringError, match=f"DIST_{mode.upper()} requires"):
        compiler_tlair(_load_kernel(mode, 1), type_args=args)


@pytest.mark.parametrize("mode,types,source_extent,lanes", _MODES)
def test_load_align_requires_ub(
    compiler_tlair: Any, mode: str, types, source_extent: int, lanes: int
) -> None:
    dtype = types[0][0]
    args = (
        _tensor(dtype, source_extent, addrspace=tla.AddressSpace.gm),
        _tensor(dtype, lanes),
    )
    with pytest.raises(tla.TlaCoreAPIError, match="expected addrspace ub, got gm"):
        compiler_tlair(_load_kernel(mode, source_extent), type_args=args)


@pytest.mark.parametrize("mode,types,source_extent,lanes", _MODES)
def test_load_align_rejects_post_update(
    compiler_tlair: Any, mode: str, types, source_extent: int, lanes: int
) -> None:
    @tla.kernel
    def update(source: tla.Tensor):
        with tla.vector():
            with tla.vec.func(mode="simd"):
                source.load(
                    NormalLoadParams(
                        load_dist=mode,
                        post_mode=PostMode.POST_MODE_UPDATE,
                    )
                )

    with pytest.raises(
        UnsupportedExecutionLowering,
        match=f"post_mode=update with load_dist='{mode}' is not supported yet",
    ):
        compiler_tlair(update, type_args=(_tensor(types[0][0], source_extent),))


def test_mask_downsample_remains_a_mask_load(compiler_tlair: Any) -> None:
    @tla.kernel
    def mask_load(source: tla.Tensor):
        with tla.vector():
            with tla.vec.func(mode="simd"):
                mask = source.load(MaskLoadParams(load_dist=MaskLoadDist.DIST_DS))
                assert isinstance(mask, MaskSSA)

    ir = compiler_tlair(mask_load, type_args=(_tensor(tla.Int8, 64),))
    assert "#tla.load_dist<ds>" in ir
    assert "-> !tla.mask<64>" in ir


@pytest.mark.parametrize("mode", [mode for mode, _, _, _ in _MODES])
def test_vector_distributions_are_not_mask_distributions(
    compiler_tlair: Any, mode: str
) -> None:
    @tla.kernel
    def mask_load(source: tla.Tensor):
        with tla.vector():
            with tla.vec.func(mode="simd"):
                source.load(MaskLoadParams(load_dist=mode))

    with pytest.raises(UnsupportedExecutionLowering, match="unsupported MaskLoadDist"):
        compiler_tlair(mask_load, type_args=(_tensor(tla.Int8, 256),))
