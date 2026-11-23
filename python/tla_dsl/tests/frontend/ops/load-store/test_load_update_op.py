# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Frontend TLAIR coverage for aligned ``post_mode=update`` load/store."""

from __future__ import annotations

from typing import Any

import pytest

import catlass.tla as tla
from catlass.execution_lowering import TlaLoweringError, UnsupportedExecutionLowering
from catlass.params import (
    NormalLoadParams,
    NormalStoreParams,
    PostMode,
    UnalignLoadParams,
    UnalignStoreParams,
)
from catlass.runtime import TlaCoreAPIError
from catlass.tla.runtime import make_fake_tensor

_LOAD_UPD = NormalLoadParams(post_mode=PostMode.POST_MODE_UPDATE)
_STORE_UPD = NormalStoreParams(post_mode=PostMode.POST_MODE_UPDATE)


def _ub_tensor(
    dtype: type[tla.Numeric] = tla.Float32,
    extent: int = 64,
) -> tla.Tensor:
    return make_fake_tensor(
        dtype,
        (extent,),
        (1,),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(extent,),
        layout_tag=tla.arch.RowMajor,
    )


@tla.kernel
def load_post_mode_update_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    """Align UPDATE with default stride (= origin_shape); unpack ``(vec, next)``."""
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            x_reg, _src_tile = src_tile.load(
                NormalLoadParams(post_mode=PostMode.POST_MODE_UPDATE)
            )
            dst_tile.store(x_reg)


@tla.kernel
def load_post_update_stride_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst_tile.store(src_tile.load(NormalLoadParams(post_update_stride=4)))


@tla.kernel
def load_post_mode_update_outside_loop_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            x_reg, _src_tile = src_tile.load(_LOAD_UPD)
            dst_tile.store(x_reg)


@tla.kernel
def load_post_mode_update_ok_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
            dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
            for _ in tla.range(2):
                x_reg, src_tile = src_tile.load(_LOAD_UPD)
                dst_tile = dst_tile.store(x_reg, _STORE_UPD)


@tla.kernel
def load_post_mode_update_local_params_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
            dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
            load_params = NormalLoadParams(post_mode=PostMode.POST_MODE_UPDATE)
            store_params = NormalStoreParams(post_mode=PostMode.POST_MODE_UPDATE)
            for _ in tla.range(2):
                x_reg, src_tile = src_tile.load(load_params)
                dst_tile = dst_tile.store(x_reg, store_params)


@tla.kernel
def load_post_mode_update_nonzero_coord_in_loop_kernel(
    src: tla.Tensor, dst: tla.Tensor
) -> None:
    """Illegal: UPDATE rejects nonzero coord (must prefold to coord=0)."""
    with tla.vector():
        with tla.vec.func(mode="simd"):
            src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(1))
            dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(1))
            for _ in tla.range(2):
                x_reg, src_tile = src_tile.load(_LOAD_UPD)
                dst_tile = dst_tile.store(x_reg, _STORE_UPD)


@tla.kernel
def load_post_mode_update_nonzero_coord_outside_loop_kernel(
    src: tla.Tensor, dst: tla.Tensor
) -> None:
    """Illegal: UPDATE rejects nonzero coord even outside a loop."""
    with tla.vector():
        with tla.vec.func(mode="simd"):
            src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(1))
            dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
            x_reg, _src_tile = src_tile.load(_LOAD_UPD)
            dst_tile.store(x_reg)


@tla.kernel
def load_update_then_ordinary_load_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
            out = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
            _x, tile = tile.load(_LOAD_UPD)
            y = tile.load()
            out.store(y)


@tla.kernel
def ordinary_loop_carries_nonzero_coord_tile_kernel(
    src: tla.Tensor, dst: tla.Tensor
) -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(1))
            out = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
            for _ in tla.range(2):
                x = tile.load()
                out.store(x)
                tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(1))


@tla.kernel
def load_update_propagates_origin_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            win = tla.make_layout(
                tla.make_shape(64),
                tla.make_stride(1),
                origin_shape=tla.make_shape(40),
                layoutTag=tla.arch.RowMajor,
            )
            src_tile = tla.make_tensor(src.ptr, win, tla.make_coord(0))
            dst_tile = tla.make_tensor(dst.ptr, win, tla.make_coord(0))
            for _ in tla.range(2):
                x_reg, src_tile = src_tile.load(_LOAD_UPD)
                dst_tile = dst_tile.store(x_reg, _STORE_UPD)


@tla.kernel
def load_update_nonunit_stride_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    """Illegal: UPDATE rebuilds with make_stride(1); non-unit layout rejected."""
    with tla.vector():
        with tla.vec.func(mode="simd"):
            # ``src`` is typed with stride=(2,); do not retile away the layout.
            x_reg, _src_tile = src.load(_LOAD_UPD)
            dst.store(x_reg)


@tla.kernel
def load_update_tail_mask_epilogue_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
            dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
            x_full, src_tile = src_tile.load(_LOAD_UPD)
            dst_tile = dst_tile.store(x_full, _STORE_UPD)
            x_tail, src_tile = src_tile.load(_LOAD_UPD)
            tail, _ = tla.update_mask(16, dtype=tla.Float32)
            dst_tile = dst_tile.store(x_tail, _STORE_UPD, mask=tail)


@tla.kernel
def load_post_mode_update_dynamic_stride_kernel(
    src: tla.Tensor,
    dst: tla.Tensor,
    stride: "index",  # noqa: F821
) -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            step = tla.as_numeric(stride)
            src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
            dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
            for _ in tla.range(1):
                x_reg, src_tile = src_tile.load(
                    NormalLoadParams(
                        post_mode=PostMode.POST_MODE_UPDATE,
                        post_update_stride=step,
                    )
                )
                dst_tile = dst_tile.store(
                    x_reg,
                    NormalStoreParams(
                        post_mode=PostMode.POST_MODE_UPDATE,
                        post_update_stride=step,
                    ),
                )


def test_load_update_default_stride_uses_origin_shape(
    compiler_tlair: Any,
) -> None:
    """Align UPDATE with default post_update_stride=0 advances by origin_shape."""
    mlir = compiler_tlair(
        load_post_mode_update_kernel,
        type_args=(_ub_tensor(), _ub_tensor()),
    )
    assert "tla.load_update" in mlir
    assert "64 : i32" in mlir or "%c64" in mlir or "64 : index" in mlir


def test_load_update_rejects_stride_without_update_mode(
    compiler_tlair: Any,
) -> None:
    with pytest.raises(
        (TlaLoweringError, UnsupportedExecutionLowering, NotImplementedError),
        match="post_update_stride",
    ):
        compiler_tlair(
            load_post_update_stride_kernel,
            type_args=(_ub_tensor(), _ub_tensor()),
        )


def test_load_post_mode_update_outside_loop_emits_tlair(compiler_tlair: Any) -> None:
    mlir = compiler_tlair(
        load_post_mode_update_outside_loop_kernel,
        type_args=(_ub_tensor(), _ub_tensor()),
    )
    assert "tla.load_update" in mlir
    assert "scf.for" not in mlir


@pytest.mark.parametrize(
    "kernel",
    [
        load_post_mode_update_ok_kernel,
        load_post_mode_update_local_params_kernel,
    ],
)
def test_load_store_post_mode_update_emits_tlair(
    compiler_tlair: Any, kernel: Any
) -> None:
    mlir = compiler_tlair(
        kernel, type_args=(_ub_tensor(extent=128), _ub_tensor(extent=128))
    )
    assert "tla.load_update" in mlir
    assert "tla.store_update" in mlir
    assert "scf.for" in mlir
    assert "!tla.ptr" in mlir
    if kernel is load_post_mode_update_ok_kernel:
        assert "64 : i32" in mlir
        assert "!tla.coord<0>" in mlir
        assert "tla.tile_view" in mlir


@pytest.mark.parametrize(
    "kernel",
    [
        load_post_mode_update_nonzero_coord_in_loop_kernel,
        load_post_mode_update_nonzero_coord_outside_loop_kernel,
    ],
)
def test_load_store_post_mode_update_nonzero_coord_rejected(
    compiler_tlair: Any, kernel: Any
) -> None:
    with pytest.raises(TlaLoweringError, match="origin window"):
        compiler_tlair(
            kernel,
            type_args=(_ub_tensor(extent=256), _ub_tensor(extent=256)),
        )


def test_load_store_post_mode_update_nonunit_stride_rejected(
    compiler_tlair: Any,
) -> None:
    strided = make_fake_tensor(
        tla.Float32,
        (64,),
        (2,),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(64,),
        layout_tag=tla.arch.RowMajor,
    )
    with pytest.raises(TlaLoweringError, match="all strides == 1"):
        compiler_tlair(
            load_update_nonunit_stride_kernel,
            type_args=(strided, _ub_tensor(extent=64)),
        )


@pytest.mark.parametrize(
    "dtype",
    [
        tla.Int8,
        tla.UInt8,
        tla.Float8E4M3FN,
        tla.Float8E5M2,
        tla.HiFloat8,
        tla.Float8E8M0,
        tla.Float4E2M1x2,
        tla.Float4E1M2x2,
        tla.Int16,
        tla.UInt16,
        tla.Float16,
        tla.BFloat16,
        tla.Int32,
        tla.UInt32,
        tla.Float32,
    ],
)
def test_load_store_post_mode_update_supports_b8_b16_b32(
    compiler_tlair: Any, dtype: Any
) -> None:
    """Align update accepts b8/b16/b32 widths (existing kernel tile=64)."""
    mlir = compiler_tlair(
        load_post_mode_update_ok_kernel,
        type_args=(
            _ub_tensor(dtype=dtype, extent=128),
            _ub_tensor(dtype=dtype, extent=128),
        ),
    )
    assert "tla.load_update" in mlir
    assert "tla.store_update" in mlir


@tla.kernel
def load_post_mode_update_b64_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    """b64 VL is 32 elements; keep the update tile within one register."""
    with tla.vector():
        with tla.vec.func(mode="simd"):
            src_tile = tla.tile_view(src, tla.make_shape(32), tla.make_coord(0))
            dst_tile = tla.tile_view(dst, tla.make_shape(32), tla.make_coord(0))
            for _ in tla.range(2):
                x_reg, src_tile = src_tile.load(_LOAD_UPD)
                dst_tile = dst_tile.store(x_reg, _STORE_UPD)


@pytest.mark.parametrize("dtype", [tla.Int64, tla.UInt64, tla.Float64])
def test_load_store_post_mode_update_supports_b64(
    compiler_tlair: Any, dtype: Any
) -> None:
    """b64 UPDATE: i64/u64/f64 (AscendC TypeGet<double>→vector_u64, u32 path)."""
    mlir = compiler_tlair(
        load_post_mode_update_b64_kernel,
        type_args=(
            _ub_tensor(dtype=dtype, extent=64),
            _ub_tensor(dtype=dtype, extent=64),
        ),
    )
    assert "tla.load_update" in mlir
    assert "tla.store_update" in mlir


def test_load_store_post_mode_update_rejects_unsupported_dtype(
    compiler_tlair: Any,
) -> None:
    # Unpacked fp4 (not fp4x2) is outside UPDATE's AscendC TypeGet set for VF move.
    with pytest.raises(TlaLoweringError, match="b8/b16/b32/b64"):
        compiler_tlair(
            load_post_mode_update_ok_kernel,
            type_args=(
                _ub_tensor(dtype=tla.Float4E2M1, extent=128),
                _ub_tensor(dtype=tla.Float4E2M1, extent=128),
            ),
        )


@pytest.mark.parametrize(
    "params",
    [
        (
            NormalLoadParams(post_mode=PostMode.POST_MODE_UPDATE),
            NormalStoreParams(post_mode=PostMode.POST_MODE_UPDATE),
        ),
        (
            UnalignLoadParams(post_mode=PostMode.POST_MODE_UPDATE),
            UnalignStoreParams(post_mode=PostMode.POST_MODE_UPDATE),
        ),
    ],
    ids=["align", "unalign"],
)
def test_load_store_post_mode_update_rejects_mismatched_elem_width(
    compiler_tlair: Any, params: tuple[Any, Any]
) -> None:
    """UPDATE store must not accept i32 register into a u8 window (stride/instr width)."""
    load_p, store_p = params

    @tla.kernel
    def mismatched(src: tla.Tensor, dst: tla.Tensor) -> None:
        with tla.vector():
            with tla.vec.func(mode="simd"):
                src_tile = tla.tile_view(src, tla.make_shape(32), tla.make_coord(0))
                dst_tile = tla.tile_view(dst, tla.make_shape(32), tla.make_coord(0))
                x_reg, src_tile = src_tile.load(load_p)
                dst_tile = dst_tile.store(x_reg, store_p)

    with pytest.raises((TlaLoweringError, TlaCoreAPIError), match="element types must match"):
        compiler_tlair(
            mismatched,
            type_args=(
                _ub_tensor(dtype=tla.Int32, extent=64),
                _ub_tensor(dtype=tla.UInt8, extent=32),
            ),
        )


@pytest.mark.parametrize(
    "params,load_op,store_op",
    [
        (
            (
                NormalLoadParams(post_mode=PostMode.POST_MODE_UPDATE),
                NormalStoreParams(post_mode=PostMode.POST_MODE_UPDATE),
            ),
            "tla.load_update",
            "tla.store_update",
        ),
        (
            (
                UnalignLoadParams(post_mode=PostMode.POST_MODE_UPDATE),
                UnalignStoreParams(post_mode=PostMode.POST_MODE_UPDATE),
            ),
            "tla.load_unalign_update",
            "tla.store_unalign_update",
        ),
    ],
    ids=["align", "unalign"],
)
def test_load_store_post_mode_update_supports_bool(
    compiler_tlair: Any,
    params: tuple[Any, Any],
    load_op: str,
    store_op: str,
) -> None:
    """Bool is AscendC b8 move-only UPDATE (Align + UnAlign → int8_t/uint8_t)."""
    load_p, store_p = params

    @tla.kernel
    def bool_update(src: tla.Tensor, dst: tla.Tensor) -> None:
        with tla.vector():
            with tla.vec.func(mode="simd"):
                src_tile = tla.tile_view(src, tla.make_shape(256), tla.make_coord(0))
                dst_tile = tla.tile_view(dst, tla.make_shape(256), tla.make_coord(0))
                for _ in tla.range(2):
                    x_reg, src_tile = src_tile.load(load_p)
                    dst_tile = dst_tile.store(x_reg, store_p)

    mlir = compiler_tlair(
        bool_update,
        type_args=(
            _ub_tensor(dtype=tla.Bool, extent=512),
            _ub_tensor(dtype=tla.Bool, extent=512),
        ),
    )
    assert load_op in mlir
    assert store_op in mlir


def test_load_update_then_ordinary_load_uses_next_window_tile(
    compiler_tlair: Any,
) -> None:
    mlir = compiler_tlair(
        load_update_then_ordinary_load_kernel,
        type_args=(_ub_tensor(extent=256), _ub_tensor(extent=256)),
    )
    assert "tla.load_update" in mlir
    assert '"tla.load"' in mlir
    assert "tla.make_tensor" in mlir
    load_update_i = mlir.index("tla.load_update")
    make_tensor_i = mlir.index("tla.make_tensor", load_update_i)
    ordinary_i = mlir.index('"tla.load"', make_tensor_i)
    assert load_update_i < make_tensor_i < ordinary_i


def test_ordinary_loop_carries_nonzero_coord_tile_without_update(
    compiler_tlair: Any,
) -> None:
    mlir = compiler_tlair(
        ordinary_loop_carries_nonzero_coord_tile_kernel,
        type_args=(_ub_tensor(extent=256), _ub_tensor(extent=256)),
    )
    assert "scf.for" in mlir
    assert "tla.load_update" not in mlir
    assert "tla.ptr_add" not in mlir
    assert "!tla.coord<64>" in mlir


def test_load_update_propagates_origin_shape(compiler_tlair: Any) -> None:
    mlir = compiler_tlair(
        load_update_propagates_origin_kernel,
        type_args=(_ub_tensor(extent=256), _ub_tensor(extent=256)),
    )
    assert "tla.load_update" in mlir
    assert "tla.store_update" in mlir
    assert "tla.make_tensor" in mlir
    assert mlir.count("!tla.shape<40>") >= 2
    assert (
        "!tla.layout<!tla.shape<64>, !tla.stride<1>, !tla.shape<40>, RowMajor>" in mlir
    )


def test_load_update_tail_mask_epilogue_emits_update_mask(
    compiler_tlair: Any,
) -> None:
    mlir = compiler_tlair(
        load_update_tail_mask_epilogue_kernel,
        type_args=(_ub_tensor(extent=80), _ub_tensor(extent=80)),
    )
    assert "tla.load_update" in mlir
    assert "tla.store_update" in mlir
    assert "tla.update_mask" in mlir
    assert mlir.count("tla.update_mask") == 1


def test_load_store_post_mode_update_dynamic_stride_emits_tlair(
    compiler_tlair: Any,
) -> None:
    mlir = compiler_tlair(
        load_post_mode_update_dynamic_stride_kernel,
        type_args=(_ub_tensor(), _ub_tensor(), 64),
    )
    assert "tla.load_update" in mlir
    assert "tla.store_update" in mlir
    assert "%arg2" in mlir
