from __future__ import annotations

from typing import Any

import pytest

import catlass.tla as tla
from catlass.execution_lowering import TlaLoweringError
from catlass.params import NormalStoreParams, StoreDist
from catlass.tla.runtime import make_fake_tensor


def _ub_tensor(dtype: type[tla.Numeric], extent: int) -> tla.Tensor:
    return make_fake_tensor(
        dtype,
        (extent,),
        (1,),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(extent,),
        layout_tag=tla.arch.RowMajor,
    )


def _store_intlv(dist: str, even_extent: int, odd_extent: int, dest_extent: int):
    @tla.kernel
    def kernel(even: tla.Tensor, odd: tla.Tensor, dest: tla.Tensor) -> None:
        even_tile = tla.tile_view(even, tla.make_shape(even_extent), tla.make_coord(0))
        odd_tile = tla.tile_view(odd, tla.make_shape(odd_extent), tla.make_coord(0))
        dest_tile = tla.tile_view(dest, tla.make_shape(dest_extent), tla.make_coord(0))
        with tla.vector():
            with tla.vec.func(mode="simd"):
                v0 = even_tile.load()
                v1 = odd_tile.load()
                dest_tile.store((v0, v1), NormalStoreParams(store_dist=dist))

    return kernel


@pytest.mark.parametrize(
    ("dtype", "element", "lanes", "dist"),
    [
        (tla.Bool, "i1", 256, StoreDist.DIST_INTLV_B8),
        (tla.Int8, "i8", 256, StoreDist.DIST_INTLV_B8),
        (tla.UInt8, "ui8", 256, StoreDist.DIST_INTLV_B8),
        (tla.Float8E8M0, "!tla.f8e8m0", 256, StoreDist.DIST_INTLV_B8),
        (tla.Float16, "f16", 128, StoreDist.DIST_INTLV_B16),
        (tla.BFloat16, "bf16", 128, StoreDist.DIST_INTLV_B16),
        (tla.Int16, "i16", 128, StoreDist.DIST_INTLV_B16),
        (tla.Float32, "f32", 64, StoreDist.DIST_INTLV_B32),
        (tla.Int32, "i32", 64, StoreDist.DIST_INTLV_B32),
        (tla.UInt32, "ui32", 64, StoreDist.DIST_INTLV_B32),
    ],
)
def test_store_intlv_emits_dual_source_tlair(
    compiler_tlair: Any,
    dtype: type[tla.Numeric],
    element: str,
    lanes: int,
    dist: str,
) -> None:
    mlir = compiler_tlair(
        _store_intlv(dist, lanes, lanes, lanes * 2),
        type_args=(
            _ub_tensor(dtype, lanes),
            _ub_tensor(dtype, lanes),
            _ub_tensor(dtype, lanes * 2),
        ),
    )
    (store_line,) = [line for line in mlir.splitlines() if "tla.store" in line]
    assert f"#tla.store_dist<{dist}>" in store_line
    assert store_line.count(f"!tla.vector<{lanes}x{element}>") == 2


@pytest.mark.parametrize(
    ("dtype", "lanes", "dist", "match"),
    [
        # Use VL appropriate to the dtype so load succeeds; store_dist rejects.
        (tla.Float32, 64, StoreDist.DIST_INTLV_B8, "DIST_INTLV_B8 requires"),
        (tla.Int8, 256, StoreDist.DIST_INTLV_B16, "DIST_INTLV_B16 requires"),
        (tla.Float16, 128, StoreDist.DIST_INTLV_B32, "DIST_INTLV_B32 requires"),
    ],
)
def test_store_intlv_rejects_wrong_element_width(
    compiler_tlair: Any,
    dtype: type[tla.Numeric],
    lanes: int,
    dist: str,
    match: str,
) -> None:
    with pytest.raises(TlaLoweringError, match=match):
        compiler_tlair(
            _store_intlv(dist, lanes, lanes, lanes * 2),
            type_args=(
                _ub_tensor(dtype, lanes),
                _ub_tensor(dtype, lanes),
                _ub_tensor(dtype, lanes * 2),
            ),
        )


def test_store_intlv_requires_tuple(compiler_tlair: Any) -> None:
    @tla.kernel
    def single(src: tla.Tensor, dest: tla.Tensor) -> None:
        src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
        dest_tile = tla.tile_view(dest, tla.make_shape(128), tla.make_coord(0))
        with tla.vector():
            with tla.vec.func(mode="simd"):
                v = src_tile.load()
                dest_tile.store(
                    v, NormalStoreParams(store_dist=StoreDist.DIST_INTLV_B32)
                )

    with pytest.raises(TlaLoweringError, match="requires a 2-tuple"):
        compiler_tlair(
            single,
            type_args=(_ub_tensor(tla.Float32, 64), _ub_tensor(tla.Float32, 128)),
        )


def test_store_intlv_rejects_mask(compiler_tlair: Any) -> None:
    @tla.kernel
    def masked(even: tla.Tensor, odd: tla.Tensor, dest: tla.Tensor) -> None:
        even_tile = tla.tile_view(even, tla.make_shape(64), tla.make_coord(0))
        odd_tile = tla.tile_view(odd, tla.make_shape(64), tla.make_coord(0))
        dest_tile = tla.tile_view(dest, tla.make_shape(128), tla.make_coord(0))
        with tla.vector():
            with tla.vec.func(mode="simd"):
                v0 = even_tile.load()
                v1 = odd_tile.load()
                preg = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                dest_tile.store(
                    (v0, v1),
                    NormalStoreParams(store_dist=StoreDist.DIST_INTLV_B32),
                    mask=preg,
                )

    with pytest.raises(TlaLoweringError, match="does not accept a predicate mask"):
        compiler_tlair(
            masked,
            type_args=(
                _ub_tensor(tla.Float32, 64),
                _ub_tensor(tla.Float32, 64),
                _ub_tensor(tla.Float32, 128),
            ),
        )
