from __future__ import annotations

from typing import Any

import pytest

import catlass.tla as tla
import catlass.runtime as runtime_mod  # noqa: F401
from catlass.execution_lowering import TlaLoweringError
from catlass.params import NormalStoreParams, StoreDist
from catlass.tla.runtime import make_fake_tensor


def _ub_tensor(
    dtype: type[tla.Numeric],
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
def store_dist(src: tla.Tensor, dst: tla.Tensor, dist: tla.Constexpr[str]) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            src_reg = src_tile.load()
            dst_tile.store(src_reg, NormalStoreParams(store_dist=dist))


@tla.kernel
def store_pack_b64(src: tla.Tensor, dst: tla.Tensor) -> None:
    # AscendC same-type b64 pack: full register is 32 lanes.
    src_tile = tla.tile_view(src, tla.make_shape(32), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(32), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            src_reg = src_tile.load()
            dst_tile.store(
                src_reg, NormalStoreParams(store_dist=StoreDist.DIST_PACK_B64)
            )


@pytest.mark.parametrize(
    ("dist", "source_dtype", "dest_dtype", "extent"),
    (
        (StoreDist.DIST_NORM, tla.Float32, tla.Float32, 64),
        (StoreDist.DIST_NORM_B8, tla.UInt8, tla.UInt8, 256),
        (StoreDist.DIST_NORM_B8, tla.UInt16, tla.UInt16, 128),
        (StoreDist.DIST_NORM_B16, tla.UInt16, tla.UInt16, 128),
        (StoreDist.DIST_NORM_B16, tla.UInt32, tla.UInt32, 64),
        (StoreDist.DIST_NORM_B32, tla.UInt32, tla.UInt32, 64),
        (StoreDist.DIST_NORM_B32, tla.UInt64, tla.UInt64, 32),
        # AscendC same-type natural packs (geometry 1).
        (StoreDist.DIST_PACK_B32, tla.Float32, tla.Float32, 64),
        (StoreDist.DIST_PACK_B32, tla.Int32, tla.Int32, 64),
        (StoreDist.DIST_PACK_B16, tla.Int16, tla.Int16, 128),
        (StoreDist.DIST_PACK_B16, tla.Float16, tla.Float16, 128),
    ),
)
def test_store_dist_emits_tlair(
    compiler_tlair: Any,
    dist: StoreDist,
    source_dtype: type[tla.Numeric],
    dest_dtype: type[tla.Numeric],
    extent: int,
) -> None:
    @tla.kernel
    def kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
        src_tile = tla.tile_view(src, tla.make_shape(extent), tla.make_coord(0))
        dst_tile = tla.tile_view(dst, tla.make_shape(extent), tla.make_coord(0))
        with tla.vector():
            with tla.vec.func(mode="simd"):
                src_reg = src_tile.load()
                dst_tile.store(src_reg, NormalStoreParams(store_dist=dist))

    mlir = compiler_tlair(
        kernel,
        type_args=(
            _ub_tensor(source_dtype, extent=extent),
            _ub_tensor(dest_dtype, extent=extent),
        ),
    )

    assert "tla.store" in mlir
    if dist != StoreDist.DIST_NORM:
        assert f"#tla.store_dist<{dist}>" in mlir


@pytest.mark.parametrize(
    ("dist", "source_dtype", "dest_dtype"),
    (
        (StoreDist.DIST_NORM_B16, tla.UInt8, tla.UInt8),
        (StoreDist.DIST_NORM_B32, tla.UInt16, tla.UInt16),
        (StoreDist.DIST_NORM_B8, tla.UInt32, tla.Int32),
    ),
)
def test_explicit_norm_requires_compatible_types(
    compiler_tlair: Any, dist: str, source_dtype: type[tla.Numeric], dest_dtype: type[tla.Numeric]
) -> None:
    with pytest.raises(TlaLoweringError, match="requires matching source and destination types"):
        compiler_tlair(
            store_dist,
            type_args=(
                _ub_tensor(source_dtype),
                _ub_tensor(dest_dtype),
                dist,
            ),
        )


def test_store_pack_b32_f16_post_cast_layout(compiler_tlair: Any) -> None:
    """AscendC MulCast/GenStoreL2: f16 reg + PACK_B32 into compacted f16 tile."""

    @tla.kernel
    def kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
        src_tile = tla.tile_view(src, tla.make_shape(128), tla.make_coord(0))
        dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
        with tla.vector():
            with tla.vec.func(mode="simd"):
                src_reg = src_tile.load()
                dst_tile.store(
                    src_reg, NormalStoreParams(store_dist=StoreDist.DIST_PACK_B32)
                )

    mlir = compiler_tlair(
        kernel,
        type_args=(
            _ub_tensor(tla.Float16, extent=128),
            _ub_tensor(tla.Float16, extent=64),
        ),
    )
    assert "#tla.store_dist<pack_b32>" in mlir


@pytest.mark.parametrize(
    ("source_dtype", "dest_dtype"),
    (
        (tla.Int64, tla.Int64),
        (tla.UInt64, tla.UInt64),
        (tla.Float64, tla.Float64),
    ),
)
def test_store_pack_b64_emits_tlair(
    compiler_tlair: Any,
    source_dtype: type[tla.Numeric],
    dest_dtype: type[tla.Numeric],
) -> None:
    mlir = compiler_tlair(
        store_pack_b64,
        type_args=(
            _ub_tensor(source_dtype, extent=32),
            _ub_tensor(dest_dtype, extent=32),
        ),
    )
    assert "tla.store" in mlir
    assert "#tla.store_dist<pack_b64>" in mlir


def test_store_pack_b64_rejects_non_b64_source(compiler_tlair: Any) -> None:
    with pytest.raises(TlaLoweringError, match="DIST_PACK_B64 requires matching"):
        compiler_tlair(
            store_pack_b64,
            type_args=(
                _ub_tensor(tla.Float32, extent=32),
                _ub_tensor(tla.Int64, extent=32),
            ),
        )


def test_store_pack_b64_rejects_different_b64_types(compiler_tlair: Any) -> None:
    with pytest.raises(TlaLoweringError, match="DIST_PACK_B64 requires matching"):
        compiler_tlair(
            store_pack_b64,
            type_args=(
                _ub_tensor(tla.Int64, extent=32),
                _ub_tensor(tla.UInt64, extent=32),
            ),
        )


def test_store_pack_b64_rejects_non_b64_dest(compiler_tlair: Any) -> None:
    with pytest.raises(TlaLoweringError, match="DIST_PACK_B64 requires matching"):
        compiler_tlair(
            store_pack_b64,
            type_args=(
                _ub_tensor(tla.Int64, extent=32),
                _ub_tensor(tla.Int32, extent=32),
            ),
        )


def test_store_pack_b32_rejects_cross_width(compiler_tlair: Any) -> None:
    with pytest.raises(TlaLoweringError, match="DIST_PACK_B32 requires matching"):
        compiler_tlair(
            store_dist,
            type_args=(
                _ub_tensor(tla.Float32),
                _ub_tensor(tla.Float16),
                StoreDist.DIST_PACK_B32,
            ),
        )


def test_store_pack_b16_rejects_cross_width(compiler_tlair: Any) -> None:
    with pytest.raises(TlaLoweringError, match="DIST_PACK_B16 requires matching"):
        compiler_tlair(
            store_dist,
            type_args=(
                _ub_tensor(tla.Int16, extent=128),
                _ub_tensor(tla.Int8, extent=128),
                StoreDist.DIST_PACK_B16,
            ),
        )
