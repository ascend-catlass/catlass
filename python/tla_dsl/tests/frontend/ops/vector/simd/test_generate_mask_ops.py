from __future__ import annotations

import pytest

import catlass.tla as tla
from catlass.core_api import MaskSSA
from catlass.tla.runtime import make_fake_tensor


def _vector_tensor(
    dtype: type[tla.Numeric] = tla.Float32,
    size: int = 64,
) -> tla.Tensor:
    return make_fake_tensor(
        dtype,
        (size,),
        (1,),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(size,),
        layout_tag=tla.arch.RowMajor,
    )


@tla.kernel
def generate_mask_kernel(
    src: tla.Tensor,
    dst: tla.Tensor,
    lanes: tla.Constexpr[int],
    offset: tla.Constexpr[int],
) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(lanes), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(lanes), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            value = src_tile.load()
            mask = value.generate_mask(offset)
            assert isinstance(mask, MaskSSA)
            dst_tile.store(tla.where(mask, value, value))


@pytest.mark.parametrize(
    ("dtype", "lanes", "offset"),
    [(tla.Float32, 64, 0), (tla.Float32, 64, 31), (tla.Float16, 128, 15),
     (tla.UInt32, 64, 7), (tla.Int16, 128, 0)],
    ids=["f32-low", "f32-high", "f16-high", "u32", "i16"],
)
def test_generate_mask_emits_tla_op(dtype, lanes, offset) -> None:
    tensor = _vector_tensor(dtype, lanes)
    mlir = generate_mask_kernel.dump_mlir(
        type_args=(tensor, tensor, lanes, offset)
    )
    assert "tla.generate_mask" in mlir
    assert f"offset {offset} :" in mlir
    assert f"-> !tla.mask<{lanes}>" in mlir
    assert mlir.index("tla.vec.func") < mlir.index("tla.generate_mask")


@tla.kernel
def generate_mask_bad_offset_kernel(
    src: tla.Tensor, dst: tla.Tensor, lanes: tla.Constexpr[int], offset: tla.Constexpr[int]
) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(lanes), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(lanes), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            value = src_tile.load()
            mask = value.generate_mask(offset)
            dst_tile.store(tla.where(mask, value, value))


@pytest.mark.parametrize(
    ("dtype", "lanes", "offset", "message"),
    [
        (tla.Float32, 64, 32, r"offset must be in 0\.\.31 for a f32 source"),
        (tla.Float32, 64, -1, r"offset must be in 0\.\.31 for a f32 source"),
        (tla.Float16, 128, 16, r"offset must be in 0\.\.15 for a f16 source"),
    ],
    ids=["f32-over", "f32-negative", "f16-over"],
)
def test_generate_mask_rejects_out_of_range_offset(dtype, lanes, offset, message) -> None:
    tensor = _vector_tensor(dtype, lanes)
    with pytest.raises(tla.TlaCoreAPIError, match=message):
        generate_mask_bad_offset_kernel.dump_mlir(type_args=(tensor, tensor, lanes, offset))


@tla.kernel
def generate_mask_b8_kernel(
    src: tla.Tensor, dst: tla.Tensor, lanes: tla.Constexpr[int]
) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(lanes), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(lanes), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            value = src_tile.load()
            mask = value.generate_mask(0)
            dst_tile.store(tla.where(mask, value, value))


def test_generate_mask_rejects_b8_source() -> None:
    tensor = _vector_tensor(tla.Int8, 256)
    with pytest.raises(tla.TlaCoreAPIError, match="only supports 16- and 32-bit elements"):
        generate_mask_b8_kernel.dump_mlir(type_args=(tensor, tensor, 256))


@tla.kernel
def generate_mask_non_int_offset_kernel(
    src: tla.Tensor, dst: tla.Tensor, lanes: tla.Constexpr[int], offset: tla.Constexpr[float]
) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(lanes), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(lanes), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            value = src_tile.load()
            mask = value.generate_mask(offset)
            dst_tile.store(tla.where(mask, value, value))


def test_generate_mask_rejects_non_int_offset() -> None:
    tensor = _vector_tensor(tla.Float32, 64)
    with pytest.raises(tla.TlaCoreAPIError, match="offset must be a compile-time Python int"):
        generate_mask_non_int_offset_kernel.dump_mlir(type_args=(tensor, tensor, 64, 1.0))


def test_generate_mask_rejects_simt_mode() -> None:
    @tla.kernel
    def kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
        src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
        dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
        with tla.vector():
            with tla.vec.func(mode="simt"):
                value = src_tile.load()
                mask = value.generate_mask(0)
                dst_tile.store(tla.where(mask, value, value))

    tensor = _vector_tensor()
    with pytest.raises(tla.TlaCoreAPIError, match="tla.generate_mask.*mode='simd'"):
        kernel.dump_mlir(type_args=(tensor, tensor))


def test_generate_mask_is_method_only() -> None:
    assert callable(tla.VectorSSA.generate_mask)
    assert "generate_mask" not in tla.core.__all__
