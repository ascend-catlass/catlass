"""NumOne Bool VectorSSA (AscendC RegTensor<bool>, b8).

Arithmetic / gather / scatter / non-NORM dist stay move-only.
DATA_BLOCK_COPY (BlockLoad/BlockStore) matches AscendC bool→b8.

Type capacity / extern / bitcast are covered in test_tensor_types,
test_extern_reg, and test_vector_bitcast_ops. MaskLoad US/DS ABI is in
vector-mask-load-dist-us-ds-lowering.mlir.
"""

from __future__ import annotations

import pytest

import catlass.tla as tla
from catlass.core_api import TlaCoreAPIError
from catlass.tla.runtime import make_fake_tensor


def _ub(dtype, lanes: int = 256):
    return make_fake_tensor(
        dtype,
        (lanes,),
        (1,),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(lanes,),
        layout_tag=tla.arch.RowMajor,
    )


@tla.kernel
def _create_mask_bool_kernel() -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            _ = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Bool)


def test_create_mask_bool_emits_mask256() -> None:
    mlir = _create_mask_bool_kernel.dump_mlir()
    assert "tla.create_mask" in mlir
    assert "!tla.mask<256>" in mlir
    assert "dtype = i1" in mlir


@tla.kernel
def _bool_add_kernel(src: tla.Tensor) -> None:
    tile = tla.tile_view(src, tla.make_shape(256), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            v = tile.load()
            _ = tla.add(v, v)


@tla.kernel
def _bool_where_kernel(src_a: tla.Tensor, src_b: tla.Tensor) -> None:
    a = tla.tile_view(src_a, tla.make_shape(256), tla.make_coord(0))
    b = tla.tile_view(src_b, tla.make_shape(256), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Bool)
            _ = tla.where(mask, a.load(), b.load())


@tla.kernel
def _bool_gather_kernel(src: tla.Tensor, idx: tla.Tensor) -> None:
    tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    idx_tile = tla.tile_view(idx, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            _ = tla.gather(tile, idx_tile.load())


@tla.kernel
def _bool_scatter_kernel(dst: tla.Tensor, src: tla.Tensor, idx: tla.Tensor) -> None:
    dst_tile = tla.tile_view(dst, tla.make_shape(256), tla.make_coord(0))
    src_tile = tla.tile_view(src, tla.make_shape(256), tla.make_coord(0))
    idx_tile = tla.tile_view(idx, tla.make_shape(128), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            tla.scatter(dst_tile, src_tile.load(), idx_tile.load())


@tla.kernel
def _bool_block_store_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(256), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(256), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst_tile.store(src_tile.load(), tla.params.BlockStoreParams(block_stride=1))


@tla.kernel
def _bool_block_load_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(256), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(256), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst_tile.store(
                src_tile.load(tla.params.BlockLoadParams(block_stride=1))
            )


@tla.kernel
def _bool_block_load_update_kernel(src: tla.Tensor) -> None:
    tile = tla.tile_view(src, tla.make_shape(256), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            _ = tile.load(
                tla.params.BlockLoadParams(
                    block_stride=1,
                    repeat_stride=1,
                    post_mode=tla.params.PostMode.POST_MODE_UPDATE,
                )
            )


def test_bool_add_rejected() -> None:
    with pytest.raises(TlaCoreAPIError, match="move-only"):
        _bool_add_kernel.dump_mlir(type_args=(_ub(tla.Bool),))


def test_bool_where_rejected() -> None:
    with pytest.raises(TlaCoreAPIError, match="move-only|where"):
        _bool_where_kernel.dump_mlir(type_args=(_ub(tla.Bool), _ub(tla.Bool)))


def test_bool_gather_rejected() -> None:
    with pytest.raises(TlaCoreAPIError, match="move-only|gather"):
        _bool_gather_kernel.dump_mlir(
            type_args=(_ub(tla.Bool, 64), _ub(tla.Int32, 64))
        )


def test_bool_scatter_rejected() -> None:
    with pytest.raises(TlaCoreAPIError, match="move-only|scatter"):
        _bool_scatter_kernel.dump_mlir(
            type_args=(_ub(tla.Bool), _ub(tla.Bool), _ub(tla.UInt16, 128))
        )


def test_bool_block_store_emits() -> None:
    mlir = _bool_block_store_kernel.dump_mlir(
        type_args=(_ub(tla.Bool), _ub(tla.Bool))
    )
    assert "block_stride" in mlir
    assert "tla.store" in mlir


def test_bool_block_load_emits() -> None:
    mlir = _bool_block_load_kernel.dump_mlir(
        type_args=(_ub(tla.Bool), _ub(tla.Bool))
    )
    assert "block_stride = 1" in mlir
    assert "tla.load" in mlir


def test_bool_block_load_update_emits() -> None:
    mlir = _bool_block_load_update_kernel.dump_mlir(type_args=(_ub(tla.Bool),))
    # Block UPDATE: block_stride is an SSA operand (source, repeat, block).
    lines = [ln for ln in mlir.splitlines() if "tla.load_update" in ln]
    assert len(lines) == 1
    assert lines[0].count(",") >= 2


@tla.kernel
def _bool_mask_load_kernel(src: tla.Tensor) -> None:
    tile = tla.tile_view(src, tla.make_shape(256), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            _ = tile.load(tla.params.MaskLoadParams())


@tla.kernel
def _bool_mask_store_kernel(dst: tla.Tensor) -> None:
    tile = tla.tile_view(dst, tla.make_shape(256), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Bool)
            tile.store(mask, tla.params.MaskStoreParams())


def test_bool_mask_load_store_emits_mask_ssa() -> None:
    """Bool UB is valid MaskReg address typing (AscendC b8 / sizeof(bool)==1)."""
    load_mlir = _bool_mask_load_kernel.dump_mlir(type_args=(_ub(tla.Bool),))
    assert "tla.load" in load_mlir and "!tla.mask<256>" in load_mlir
    store_mlir = _bool_mask_store_kernel.dump_mlir(type_args=(_ub(tla.Bool),))
    assert "tla.store" in store_mlir and "!tla.mask<256>" in store_mlir


@tla.kernel
def _int8_to_bool_store_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(256), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(256), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst_tile.store(src_tile.load())


def test_int8_to_bool_store_rejected() -> None:
    with pytest.raises(TlaCoreAPIError, match="element types must match"):
        _int8_to_bool_store_kernel.dump_mlir(type_args=(_ub(tla.Int8), _ub(tla.Bool)))
