"""Full-pipeline compile repros for UPDATE review findings."""

from __future__ import annotations

import pytest

import catlass.tla as tla
from catlass.params import (
    NormalLoadParams,
    NormalStoreParams,
    PostMode,
    UnalignLoadParams,
    UnalignStoreParams,
)
from catlass.tla.runtime import make_fake_tensor

LOAD = NormalLoadParams(post_mode=PostMode.POST_MODE_UPDATE)
STORE = NormalStoreParams(post_mode=PostMode.POST_MODE_UPDATE)
UNALIGN_LOAD = UnalignLoadParams(post_mode=PostMode.POST_MODE_UPDATE)
UNALIGN_STORE = UnalignStoreParams(post_mode=PostMode.POST_MODE_UPDATE)


def _gm(n: int = 256) -> tla.Tensor:
    return make_fake_tensor(
        tla.Float32,
        (n,),
        (1,),
        origin_shape=(n,),
        layout_tag=tla.arch.RowMajor,
    )


@tla.kernel
def outside_chained(mem_x: tla.Tensor, mem_z: tla.Tensor) -> None:
    x_gm = tla.tile_view(mem_x, tla.make_shape(256), tla.make_coord(0))
    z_gm = tla.tile_view(mem_z, tla.make_shape(256), tla.make_coord(0))
    x_ub = tla.make_tensor_like(
        tla.allocate(256, tla.Float32, tla.AddressSpace.ub, 256),
        x_gm,
        tla.arch.RowMajor,
    )
    z_ub = tla.make_tensor_like(
        tla.allocate(256, tla.Float32, tla.AddressSpace.ub, 256),
        z_gm,
        tla.arch.RowMajor,
    )
    with tla.vector():
        tla.copy(x_ub, x_gm)
        with tla.vec.func(mode="simd"):
            tile = tla.tile_view(x_ub, tla.make_shape(64), tla.make_coord(0))
            out = tla.tile_view(z_ub, tla.make_shape(64), tla.make_coord(0))
            x0, tile = tile.load(LOAD)
            out = out.store(x0, STORE)
            x1, tile = tile.load(LOAD)
            out = out.store(x1, STORE)
        tla.copy(z_gm, z_ub)


@tla.kernel
def loop_update(mem_x: tla.Tensor, mem_z: tla.Tensor) -> None:
    x_gm = tla.tile_view(mem_x, tla.make_shape(256), tla.make_coord(0))
    z_gm = tla.tile_view(mem_z, tla.make_shape(256), tla.make_coord(0))
    x_ub = tla.make_tensor_like(
        tla.allocate(256, tla.Float32, tla.AddressSpace.ub, 256),
        x_gm,
        tla.arch.RowMajor,
    )
    z_ub = tla.make_tensor_like(
        tla.allocate(256, tla.Float32, tla.AddressSpace.ub, 256),
        z_gm,
        tla.arch.RowMajor,
    )
    with tla.vector():
        tla.copy(x_ub, x_gm)
        with tla.vec.func(mode="simd"):
            # Cut origin windows with tile_view, then UPDATE in a loop.
            tile = tla.tile_view(x_ub, tla.make_shape(64), tla.make_coord(0))
            out = tla.tile_view(z_ub, tla.make_shape(64), tla.make_coord(0))
            for _ in tla.range(2):
                x, tile = tile.load(LOAD)
                out = out.store(x, STORE)
        tla.copy(z_gm, z_ub)


def test_outside_chained_full_compile(isolated_compile_cache) -> None:
    del isolated_compile_cache
    art = tla.compile(outside_chained, _gm(), _gm())
    assert art is not None


@tla.kernel
def loop_update_dynamic_extent(
    mem_x: tla.Tensor,
    mem_z: tla.Tensor,
    vl: "index",  # noqa: F821
) -> None:
    x_gm = tla.tile_view(mem_x, tla.make_shape(256), tla.make_coord(0))
    z_gm = tla.tile_view(mem_z, tla.make_shape(256), tla.make_coord(0))
    x_ub = tla.make_tensor_like(
        tla.allocate(256, tla.Float32, tla.AddressSpace.ub, 256),
        x_gm,
        tla.arch.RowMajor,
    )
    z_ub = tla.make_tensor_like(
        tla.allocate(256, tla.Float32, tla.AddressSpace.ub, 256),
        z_gm,
        tla.arch.RowMajor,
    )
    with tla.vector():
        tla.copy(x_ub, x_gm)
        with tla.vec.func(mode="simd"):
            # Dynamic extent + static origin must keep ``coord<0>`` for UPDATE.
            tile = tla.tile_view(x_ub, tla.make_shape(vl), tla.make_coord(0))
            out = tla.tile_view(z_ub, tla.make_shape(vl), tla.make_coord(0))
            for _ in tla.range(2):
                x, tile = tile.load(LOAD)
                out = out.store(x, STORE)
        tla.copy(z_gm, z_ub)


def test_loop_update_full_compile(isolated_compile_cache) -> None:
    del isolated_compile_cache
    art = tla.compile(loop_update, _gm(), _gm())
    assert art is not None


@tla.kernel
def loop_unalign_update_with_sidecar(mem_x: tla.Tensor, mem_z: tla.Tensor) -> None:
    """UnAlign UPDATE in a for, plus a loop-carried ordinary (non-UPDATE) tile.

    Covers VectorRegion ``scf.for`` init rematerialization when a sidecar /
    TensorDesc base may still be abstract ``!tla.ptr`` in ``valueMap``.
    """
    x_gm = tla.tile_view(mem_x, tla.make_shape(256), tla.make_coord(0))
    z_gm = tla.tile_view(mem_z, tla.make_shape(256), tla.make_coord(0))
    x_ub = tla.make_tensor_like(
        tla.allocate(256, tla.Float32, tla.AddressSpace.ub, 256),
        x_gm,
        tla.arch.RowMajor,
    )
    z_ub = tla.make_tensor_like(
        tla.allocate(256, tla.Float32, tla.AddressSpace.ub, 256),
        z_gm,
        tla.arch.RowMajor,
    )
    side_ub = tla.make_tensor_like(
        tla.allocate(256, tla.Float32, tla.AddressSpace.ub, 256),
        z_gm,
        tla.arch.RowMajor,
    )
    with tla.vector():
        tla.copy(x_ub, x_gm)
        with tla.vec.func(mode="simd"):
            tile = tla.tile_view(x_ub, tla.make_shape(64), tla.make_coord(0))
            out = tla.tile_view(z_ub, tla.make_shape(64), tla.make_coord(0))
            side = tla.tile_view(side_ub, tla.make_shape(64), tla.make_coord(0))
            # Natural for: UnAlign UPDATE inside the loop; sidecar ordinary store
            # uses a fresh tile_view each trip (not UPDATE-carried).
            for _ in tla.range(2):
                x, tile = tile.load(UNALIGN_LOAD)
                out = out.store(x, UNALIGN_STORE)
                side.store(x)
                side = tla.tile_view(side_ub, tla.make_shape(64), tla.make_coord(0))
        tla.copy(z_gm, z_ub)


def test_loop_unalign_update_with_sidecar_full_compile(isolated_compile_cache) -> None:
    del isolated_compile_cache
    art = tla.compile(loop_unalign_update_with_sidecar, _gm(), _gm())
    assert art is not None


def test_loop_update_dynamic_extent_full_compile(isolated_compile_cache) -> None:
    del isolated_compile_cache
    art = tla.compile(loop_update_dynamic_extent, _gm(), _gm(), tla.Int32(64))
    assert art is not None


def _gm_dtype(dtype: type[tla.Numeric], n: int = 256) -> tla.Tensor:
    return make_fake_tensor(
        dtype,
        (n,),
        (1,),
        origin_shape=(n,),
        layout_tag=tla.arch.RowMajor,
    )


@tla.kernel
def unalign_update_byte_encoding_roundtrip(mem_x: tla.Tensor, mem_z: tla.Tensor) -> None:
    """UnAlign UPDATE memcpy on UB for move-only byte encodings (no GM copy)."""
    x_ub = tla.make_tensor_like(
        tla.allocate(256, mem_x.element_type, tla.AddressSpace.ub, 256),
        mem_x,
        tla.arch.RowMajor,
    )
    z_ub = tla.make_tensor_like(
        tla.allocate(256, mem_z.element_type, tla.AddressSpace.ub, 256),
        mem_z,
        tla.arch.RowMajor,
    )
    with tla.vector():
        with tla.vec.func(mode="simd"):
            src_tile = tla.tile_view(x_ub, tla.make_shape(256), tla.make_coord(0))
            dst_tile = tla.tile_view(z_ub, tla.make_shape(256), tla.make_coord(0))
            x, src_tile = src_tile.load(UNALIGN_LOAD)
            dst_tile = dst_tile.store(x, UNALIGN_STORE)


@pytest.mark.parametrize(
    "dtype",
    [tla.HiFloat8, tla.Float8E8M0, tla.Float4E2M1x2, tla.Float4E1M2x2],
)
def test_unalign_update_move_only_encodings_full_compile(
    isolated_compile_cache, dtype: type[tla.Numeric]
) -> None:
    """UnAlign UPDATE must not be rejected as arithmetic for byte encodings."""
    del isolated_compile_cache
    art = tla.compile(
        unalign_update_byte_encoding_roundtrip,
        _gm_dtype(dtype),
        _gm_dtype(dtype),
    )
    assert art is not None


@tla.kernel
def align_update_e8m0_tail_mask(mem_x: tla.Tensor, mem_z: tla.Tensor) -> None:
    """Align UPDATE copy with a non-zero tail so update_mask(dtype=e8m0) lowers."""
    x_ub = tla.make_tensor_like(
        tla.allocate(400, tla.Float8E8M0, tla.AddressSpace.ub, 256),
        mem_x,
        tla.arch.RowMajor,
    )
    z_ub = tla.make_tensor_like(
        tla.allocate(400, tla.Float8E8M0, tla.AddressSpace.ub, 256),
        mem_z,
        tla.arch.RowMajor,
    )
    with tla.vector():
        with tla.vec.func(mode="simd"):
            x_tile = tla.tile_view(x_ub, tla.make_shape(256), tla.make_coord(0))
            z_tile = tla.tile_view(z_ub, tla.make_shape(256), tla.make_coord(0))
            x_reg, x_tile = x_tile.load(LOAD)
            z_tile = z_tile.store(x_reg, STORE)
            x_tail, x_tile = x_tile.load(LOAD)
            tail, _ = tla.update_mask(144, dtype=tla.Float8E8M0)
            z_tile = z_tile.store(x_tail, STORE, mask=tail)


def test_align_update_e8m0_tail_mask_full_compile(isolated_compile_cache) -> None:
    del isolated_compile_cache
    art = tla.compile(
        align_update_e8m0_tail_mask,
        _gm_dtype(tla.Float8E8M0, 400),
        _gm_dtype(tla.Float8E8M0, 400),
    )
    assert art is not None


@tla.kernel
def align_update_b64_roundtrip(mem_x: tla.Tensor, mem_z: tla.Tensor) -> None:
    """Align UPDATE on b64: HIVM post uses signless i32 carrier (not ui32 LLVM struct)."""
    x_ub = tla.make_tensor_like(
        tla.allocate(64, mem_x.element_type, tla.AddressSpace.ub, 256),
        mem_x,
        tla.arch.RowMajor,
    )
    z_ub = tla.make_tensor_like(
        tla.allocate(64, mem_z.element_type, tla.AddressSpace.ub, 256),
        mem_z,
        tla.arch.RowMajor,
    )
    with tla.vector():
        with tla.vec.func(mode="simd"):
            src = tla.tile_view(x_ub, tla.make_shape(32), tla.make_coord(0))
            dst = tla.tile_view(z_ub, tla.make_shape(32), tla.make_coord(0))
            for _ in tla.range(2):
                x, src = src.load(LOAD)
                dst = dst.store(x, STORE)


@pytest.mark.parametrize("dtype", [tla.Int64, tla.UInt64, tla.Float64])
def test_align_update_b64_full_compile(
    isolated_compile_cache, dtype: type[tla.Numeric]
) -> None:
    del isolated_compile_cache
    art = tla.compile(
        align_update_b64_roundtrip,
        _gm_dtype(dtype, 64),
        _gm_dtype(dtype, 64),
    )
    assert art is not None
