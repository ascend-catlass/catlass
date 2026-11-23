from __future__ import annotations

from catlass.tla.runtime import make_fake_tensor


from typing import Any

import pytest

import catlass.tla as tla
from catlass.core_api import MaskSSA, TlaCoreAPIError
from catlass.params import MaskLoadParams, MaskStoreParams
import catlass.runtime as runtime_mod  # noqa: F401


def _ub_tensor(
    dtype: type[tla.Numeric],
    extent: int,
) -> tla.Tensor:
    return make_fake_tensor(
        dtype,
        (extent,),
        (1,),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(extent,),
        layout_tag=tla.arch.RowMajor,
    )


def _ub_tensor_2d(
    dtype: type[tla.Numeric],
    rows: int,
    cols: int,
) -> tla.Tensor:
    return make_fake_tensor(
        dtype,
        (rows, cols),
        (cols, 1),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(rows, cols),
        layout_tag=tla.arch.RowMajor,
    )


def _gm_tensor(
    dtype: type[tla.Numeric],
    extent: int,
) -> tla.Tensor:
    return make_fake_tensor(
        dtype,
        (extent,),
        (1,),
        origin_shape=(extent,),
        layout_tag=tla.arch.RowMajor,
    )


@tla.kernel
def load_store_mask_roundtrip_f32(
    mask_ub: tla.Tensor, data: tla.Tensor, dst: tla.Tensor
) -> None:
    # f32 UB → N=64 (same dtype as companion).
    mask_tile = tla.tile_view(mask_ub, tla.make_shape(2), tla.make_coord(0))
    data_tile = tla.tile_view(data, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            pattern = tla.create_mask(pattern=tla.mask.H, dtype=tla.Float32)
            mask_tile.store(pattern, MaskStoreParams())
            loaded = mask_tile.load(MaskLoadParams())
            assert isinstance(loaded, MaskSSA)
            v = data_tile.load()
            dst_tile.store(tla.add(v, v, mask=loaded), mask=loaded)


@tla.kernel
def load_store_mask_roundtrip_f16(
    mask_ub: tla.Tensor, data: tla.Tensor, dst: tla.Tensor
) -> None:
    mask_tile = tla.tile_view(mask_ub, tla.make_shape(8), tla.make_coord(0))
    data_tile = tla.tile_view(data, tla.make_shape(128), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(128), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            pattern = tla.create_mask(pattern=tla.mask.H, dtype=tla.Float16)
            mask_tile.store(pattern, MaskStoreParams())
            loaded = mask_tile.load(MaskLoadParams())
            assert isinstance(loaded, MaskSSA)
            v = data_tile.load()
            dst_tile.store(tla.add(v, v, mask=loaded), mask=loaded)


@tla.kernel
def load_store_mask_roundtrip_i8(
    mask_ub: tla.Tensor, data: tla.Tensor, dst: tla.Tensor
) -> None:
    mask_tile = tla.tile_view(mask_ub, tla.make_shape(32), tla.make_coord(0))
    data_tile = tla.tile_view(data, tla.make_shape(256), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(256), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            pattern = tla.create_mask(pattern=tla.mask.H, dtype=tla.Int8)
            mask_tile.store(pattern, MaskStoreParams())
            loaded = mask_tile.load(MaskLoadParams())
            assert isinstance(loaded, MaskSSA)
            v = data_tile.load()
            dst_tile.store(tla.add(v, v, mask=loaded), mask=loaded)


@tla.kernel
def load_store_mask_roundtrip_i32_ub(
    mask_ub: tla.Tensor, data: tla.Tensor, dst: tla.Tensor
) -> None:
    # Same-width int UB still valid with f32 companion.
    mask_tile = tla.tile_view(mask_ub, tla.make_shape(2), tla.make_coord(0))
    data_tile = tla.tile_view(data, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            pattern = tla.create_mask(pattern=tla.mask.H, dtype=tla.Float32)
            mask_tile.store(pattern, MaskStoreParams())
            loaded = mask_tile.load(MaskLoadParams())
            assert isinstance(loaded, MaskSSA)
            v = data_tile.load()
            dst_tile.store(tla.add(v, v, mask=loaded), mask=loaded)


@tla.kernel
def load_store_mask_roundtrip_i64_ub(
    mask_ub: tla.Tensor, data: tla.Tensor, dst: tla.Tensor
) -> None:
    # AscendC Mask NORM: b64 UB → mask<32> (address typing only).
    mask_tile = tla.tile_view(mask_ub, tla.make_shape(1), tla.make_coord(0))
    data_tile = tla.tile_view(data, tla.make_shape(32), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(32), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            pattern = tla.create_mask(pattern=tla.mask.H, dtype=tla.Int64)
            mask_tile.store(pattern, MaskStoreParams())
            loaded = mask_tile.load(MaskLoadParams())
            assert isinstance(loaded, MaskSSA)
            v = data_tile.load()
            dst_tile.store(v, mask=loaded)


@tla.kernel
def load_store_mask_fa_like_oversized_tile(
    mask_ub: tla.Tensor, data: tla.Tensor, dst: tla.Tensor
) -> None:
    """fa_mask-style: oversized view is only an address; N from UB dtype."""
    mask_tile = tla.tile_view(mask_ub, tla.make_shape(1, 64), tla.make_coord(0, 0))
    data_tile = tla.tile_view(data, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            pattern = tla.create_mask(pattern=tla.mask.H, dtype=tla.Float32)
            mask_tile.store(pattern, MaskStoreParams())
            loaded = mask_tile.load(MaskLoadParams())
            assert isinstance(loaded, MaskSSA)
            v = data_tile.load()
            dst_tile.store(tla.add(v, v, mask=loaded), mask=loaded)


@tla.kernel
def store_mask_mismatched_ub(mask_ub: tla.Tensor) -> None:
    mask_tile = tla.tile_view(mask_ub, tla.make_shape(8), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            pattern = tla.create_mask(pattern=tla.mask.H, dtype=tla.Float32)
            mask_tile.store(pattern, MaskStoreParams())


@tla.kernel
def load_mask_wrong_elem(mask_ub: tla.Tensor) -> None:
    mask_tile = tla.tile_view(mask_ub, tla.make_shape(8), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            _ = mask_tile.load(MaskLoadParams())


@tla.kernel
def load_mask_from_gm(mask_gm: tla.Tensor) -> None:
    mask_tile = tla.tile_view(mask_gm, tla.make_shape(2), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            _ = mask_tile.load(MaskLoadParams())


@tla.kernel
def load_mask_outside_vec_func(mask_ub: tla.Tensor) -> None:
    mask_tile = tla.tile_view(mask_ub, tla.make_shape(2), tla.make_coord(0))
    with tla.vector():
        _ = mask_tile.load(MaskLoadParams())


@pytest.mark.parametrize(
    ("kernel", "ub_dtype", "data_dtype", "mask_extent", "data_extent", "mask_n"),
    (
        (load_store_mask_roundtrip_f32, tla.Float32, tla.Float32, 2, 64, 64),
        (load_store_mask_roundtrip_f16, tla.Float16, tla.Float16, 8, 128, 128),
        (load_store_mask_roundtrip_i8, tla.Int8, tla.Int8, 32, 256, 256),
        (load_store_mask_roundtrip_i32_ub, tla.Int32, tla.Float32, 2, 64, 64),
        (load_store_mask_roundtrip_i32_ub, tla.UInt32, tla.Float32, 2, 64, 64),
        (load_store_mask_roundtrip_i64_ub, tla.Int64, tla.Int64, 1, 32, 32),
        (load_store_mask_roundtrip_i64_ub, tla.UInt64, tla.UInt64, 1, 32, 32),
    ),
)
def test_load_store_mask_emits_tlair(
    compiler_tlair: Any,
    kernel: Any,
    ub_dtype: type[tla.Numeric],
    data_dtype: type[tla.Numeric],
    mask_extent: int,
    data_extent: int,
    mask_n: int,
) -> None:
    mlir = compiler_tlair(
        kernel,
        type_args=(
            _ub_tensor(ub_dtype, mask_extent),
            _ub_tensor(data_dtype, data_extent),
            _ub_tensor(data_dtype, data_extent),
        ),
    )
    assert "tla.load" in mlir
    assert "tla.store" in mlir
    assert f"!tla.mask<{mask_n}>" in mlir
    assert "tla.load_mask" not in mlir
    assert "tla.store_mask" not in mlir
    assert f"-> !tla.mask<{mask_n}>" in mlir


def test_load_store_mask_fa_like_oversized_tile_emits_tlair(
    compiler_tlair: Any,
) -> None:
    mlir = compiler_tlair(
        load_store_mask_fa_like_oversized_tile,
        type_args=(
            _ub_tensor_2d(tla.Float32, 1, 128),
            _ub_tensor(tla.Float32, 64),
            _ub_tensor(tla.Float32, 64),
        ),
    )
    assert "!tla.mask<64>" in mlir
    assert "-> !tla.mask<64>" in mlir


def test_store_mask_rejects_mismatched_ub_width() -> None:
    # i8 UB ⇒ mask<256>, but create_mask(f32) is mask<64>.
    mask_ub = _ub_tensor(tla.Int8, 8)
    with pytest.raises(TlaCoreAPIError, match="implies !tla.mask"):
        store_mask_mismatched_ub.dump_mlir(type_args=(mask_ub,))


def test_load_mask_accepts_b64_ub(compiler_tlair: Any) -> None:
    # AscendC Mask NORM: SupportBytes includes 8 (i64/u64 address typing).
    mask_ub = _ub_tensor(tla.Int64, 1)
    mlir = compiler_tlair(load_mask_wrong_elem, type_args=(mask_ub,))
    assert "tla.load" in mlir
    assert "!tla.mask<32>" in mlir


def test_load_mask_rejects_gm() -> None:
    mask_gm = _gm_tensor(tla.Float32, 2)
    with pytest.raises(TlaCoreAPIError, match="addrspace ub"):
        load_mask_from_gm.dump_mlir(type_args=(mask_gm,))


def test_load_mask_requires_vec_func() -> None:
    mask_ub = _ub_tensor(tla.Float32, 2)
    with pytest.raises(Exception, match="vec.func"):
        load_mask_outside_vec_func.dump_mlir(type_args=(mask_ub,))
