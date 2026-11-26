from __future__ import annotations

from catlass.tla.runtime import make_fake_tensor


import pytest

import catlass.tla as tla
import catlass.runtime as runtime_mod  # noqa: F401
from catlass import core_api as core


# --- Frontend IR emission -----------------------------------------------------


@tla.kernel
def ptr_extract_kernel(mem_in: tla.Tensor) -> None:
    tile = tla.tile_view(mem_in, tla.make_shape(16, 16), tla.make_coord(0, 0))
    ptr = tla.allocate((16, 16), tla.Float32, tla.AddressSpace.l1, 256)
    # Extract a pointer from a GM kernel-arg tile, offset by 4 elements, and build a
    # row-major GM tensor from it via make_tensor.
    gm_ptr = tile.ptr
    gm_off = gm_ptr + 4
    gm_src = tla.make_tensor(
        gm_off, tla.make_layout(tla.make_shape(8, 8), tla.make_stride(8, 1))
    )
    # Offset an allocator-backed L1 pointer by 8 elements and build a zN L1 tensor via
    # make_tensor_like (clones gm_src's shape), consuming the offset pointer.
    l1_off = ptr + 8
    l1_dst = tla.make_tensor_like(l1_off, gm_src, tla.arch.zN)
    with tla.cube():
        tla.copy(l1_dst, gm_src)


def _host_mem():
    return make_fake_tensor(
        tla.Float32,
        (16, 16),
        (16, 1),
        origin_shape=(16, 16),
        layout_tag=tla.arch.RowMajor,
    )


def test_ptr_emits_tensor_ptr_and_ptr_add() -> None:
    mlir = ptr_extract_kernel.dump_mlir(type_args=(_host_mem(),))
    assert "tla.tensor_ptr" in mlir
    assert "tla.ptr_add" in mlir
    # tensor_ptr result type matches the tensor's embedded ptr type (GM f32, align 4).
    assert "tla.tensor_ptr" in mlir and "-> !tla.ptr<f32, gm, 4>" in mlir
    # ptr_add preserves the source pointer's alignment (256, not refined).
    assert "!tla.ptr<f32, l1, 256>" in mlir
    # Both offset pointers feed make_tensor / make_tensor_like.
    assert "tla.make_tensor" in mlir
    assert "tla.make_tensor_like" in mlir


def test_ptr_add_rejects_float_offset() -> None:
    @tla.kernel
    def bad_kernel(mem_in: tla.Tensor) -> None:
        ptr = tla.allocate(16, tla.Float32, tla.AddressSpace.l1, 256)
        _ = tla.make_tensor(
            ptr + 1.5,  # type: ignore[operator]
            tla.make_layout(tla.make_shape(4, 4), tla.make_stride(4, 1)),
        )

    with pytest.raises(Exception):
        bad_kernel.dump_mlir(type_args=(_host_mem(),))


# --- ptr_with_coord -----------------------------------------------------------


@tla.kernel
def ptr_with_coord_kernel(mem_in: tla.Tensor) -> None:
    # tile_view coordinates are tile granularity: coord (1, 0) over an (8, 16)
    # shape becomes element coord (8, 0), and the parent's RowMajor stride (16,
    # 1) puts the view start 8*16 = 128 elements into the storage.
    tail = tla.tile_view(mem_in, tla.make_shape(8, 16), tla.make_coord(1, 0))
    _ = tail.ptr_with_coord


def test_ptr_with_coord_emits_tensor_ptr_with_coord() -> None:
    mlir = ptr_with_coord_kernel.dump_mlir(type_args=(_host_mem(),))
    assert "tla.tensor_ptr_with_coord" in mlir
    # Result type matches the tensor's embedded ptr type, like tla.tensor_ptr.
    assert "-> !tla.ptr<f32, gm, 4>" in mlir
    # The emitted view carries the element coordinate (8, 0), not the tile
    # coordinate (1, 0): the tile-to-element scaling is part of the offset.
    assert "!tla.coord<8,0>" in mlir


def test_ptr_with_coord_keeps_tensor_ptr_untouched() -> None:
    @tla.kernel
    def both_kernel(mem_in: tla.Tensor) -> None:
        tail = tla.tile_view(mem_in, tla.make_shape(8, 16), tla.make_coord(1, 0))
        _ = tail.ptr
        _ = tail.ptr_with_coord

    mlir = both_kernel.dump_mlir(type_args=(_host_mem(),))
    # Both ops emitted; the raw one stays the storage base (no coord folding).
    assert "tla.tensor_ptr" in mlir
    assert "tla.tensor_ptr_with_coord" in mlir


@tla.kernel
def ptr_with_coord_runtime_kernel(mem_in: tla.Tensor) -> None:
    # The row comes from the launch geometry, so the view coordinate is a
    # runtime value. The tensor type only records `?` for it; the lowering has
    # to take the offset from the make_coord operand.
    block = tla.arch.block_idx()
    tail = tla.tile_view(mem_in, tla.make_shape(8, 16), tla.make_coord(block, 0))
    _ = tail.ptr_with_coord


def test_ptr_with_coord_emits_runtime_coord() -> None:
    mlir = ptr_with_coord_runtime_kernel.dump_mlir(type_args=(_host_mem(),))
    assert "tla.tensor_ptr_with_coord" in mlir
    # A dynamic coordinate leaf; there is no static offset to read from the type.
    assert "!tla.coord<?,0>" in mlir


_alias_metadata: dict[str, tuple] = {}


@tla.kernel
def alias_metadata_kernel() -> None:
    buf_ptr = tla.allocate(8 * 16, tla.Int32, tla.AddressSpace.ub, 256)
    whole = tla.make_tensor(
        buf_ptr,
        tla.make_layout(tla.make_shape(8, 16), tla.make_stride(16, 1)),
        tla.make_coord(0, 0),
    )
    tile = tla.tile_view(whole, tla.make_shape(2, 4), tla.make_coord(1, 1))
    alias = tla.make_tensor_like(tile.ptr, tile, tla.arch.RowMajor)
    fresh_ptr = tla.allocate(8 * 16, tla.Int32, tla.AddressSpace.ub, 256)
    fresh = tla.make_tensor_like(fresh_ptr, tile, tla.arch.RowMajor)
    fields = ("shape", "stride", "coord", "origin_shape")
    for name, tensor in (("tile", tile), ("alias", alias), ("fresh", fresh)):
        _alias_metadata[name] = tuple(
            core._tensor_metadata_field(core._as_value(tensor), field)
            for field in fields
        )


_layout_alias_metadata: dict[str, tuple] = {}


@tla.kernel
def layout_change_alias_kernel() -> None:
    buf_ptr = tla.allocate(8 * 16, tla.Int32, tla.AddressSpace.ub, 256)
    whole = tla.make_tensor(
        buf_ptr,
        tla.make_layout(tla.make_shape(8, 16), tla.make_stride(16, 1)),
        tla.make_coord(0, 0),
    )
    tile = tla.tile_view(whole, tla.make_shape(2, 4), tla.make_coord(1, 1))
    # A new layout must re-derive shape/stride instead of borrowing the
    # reference's RowMajor trees.
    column = tla.make_tensor_like(tile.ptr, tile, tla.arch.ColumnMajor)
    znalign = tla.make_tensor_like(tile.ptr, tile, tla.arch.zNUnAlign)
    # On L1 the layout is inferred (RowMajor -> zN), so the same pointer is no
    # longer a same-layout alias.
    l1_ptr = tla.allocate(8 * 16, tla.Int32, tla.AddressSpace.l1, 256)
    l1_whole = tla.make_tensor(
        l1_ptr,
        tla.make_layout(tla.make_shape(8, 16), tla.make_stride(16, 1)),
        tla.make_coord(0, 0),
    )
    l1_tile = tla.tile_view(l1_whole, tla.make_shape(2, 4), tla.make_coord(1, 1))
    l1_alias = tla.make_tensor_like(l1_tile.ptr, l1_tile)
    fields = ("shape", "stride", "coord", "origin_shape")
    for name, tensor in (
        ("column", column),
        ("znalign", znalign),
        ("l1_tile", l1_tile),
        ("l1_alias", l1_alias),
    ):
        _layout_alias_metadata[name] = tuple(
            core._tensor_metadata_field(core._as_value(tensor), field)
            for field in fields
        )


def test_make_tensor_like_from_tile_ptr_keeps_reference_metadata() -> None:
    _alias_metadata.clear()
    alias_metadata_kernel.dump_mlir()
    tile_meta = _alias_metadata["tile"]
    alias_meta = _alias_metadata["alias"]
    fresh_meta = _alias_metadata["fresh"]
    # ``tile.ptr`` names the tile's own storage, so the alias keeps the tile's
    # geometry -- including the view start. An origin-derived remap would reset
    # the coord to (0, 0) and a later tile_view/get_tile would start at element
    # 0 instead of the tile's element (2, 4).
    assert alias_meta == tile_meta
    assert alias_meta[2] == (2, 4)
    # A pointer that is not the reference's storage still re-derives the block.
    assert fresh_meta != tile_meta
    assert fresh_meta[2] == (0, 0)


def test_make_tensor_like_alias_with_new_layout_remaps_geometry() -> None:
    _layout_alias_metadata.clear()
    layout_change_alias_kernel.dump_mlir()
    column_meta = _layout_alias_metadata["column"]
    znalign_meta = _layout_alias_metadata["znalign"]
    l1_tile_meta = _layout_alias_metadata["l1_tile"]
    l1_alias_meta = _layout_alias_metadata["l1_alias"]

    # RowMajor -> ColumnMajor re-derives the strides for the new layout instead
    # of keeping the reference's (16, 1).
    assert column_meta == ((2, 4), (1, 8), (0, 0), (2, 4))

    # NZ-family layouts need four leaves; borrowing the reference's rank-2
    # RowMajor trees would leave the descriptor with the new tag and the old
    # geometry, which the descriptor verifier rejects.
    assert znalign_meta[0] == ((2, 1), (8, 1))
    assert znalign_meta[1] == ((8, 16), (1, 16))
    assert znalign_meta[2] == (0, 0)

    # L1 inference turns RowMajor into zN, so this alias is remapped as well.
    assert l1_alias_meta != l1_tile_meta
    assert l1_alias_meta[0] == ((16, 1), (8, 1))
    assert l1_alias_meta[1] == ((8, 128), (1, 128))
    assert l1_alias_meta[2] == (0, 0)
