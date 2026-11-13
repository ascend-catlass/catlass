"""Full-pipeline compile repros for UPDATE review findings."""

from __future__ import annotations

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
