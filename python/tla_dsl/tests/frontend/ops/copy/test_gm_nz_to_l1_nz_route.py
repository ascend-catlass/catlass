"""Frontend guard for the GM nZ -> L1 nZ copy route.

A GM tile whose host-side buffer is already packed as ``nZ`` -- C0-row by
16-column fractals -- is copied straight into an ``L1`` ``nZ`` tile. Both
endpoints then carry the same packing, so the copy is a plain block move (no
ND -> nZ reorder) and the route is keyed on the mirrored tags rather than on a
ColumnMajor source.

This is the transposed twin of the ``zN`` -> ``zN`` route. There is no
``nZUnAlign`` counterpart: the block move steps whole C0-row groups and whole
logical columns, so both axes must be padded and a tight packing cannot be
spelled at all. That is why this file has no unaligned case while
``test_gm_zn_to_l1_zn_route.py`` does. Because the padding is part of the
layout, every buffer here is described at its padded extent
(``ceil(M/C0) * C0`` by ``ceil(N/16) * 16``), which is also how the copy
addresses it.

Note on the spelling these tests use: they hand a fractal-tagged GM tensor
straight to ``dump_mlir`` as a kernel parameter, which is a *route-table probe*,
not a form that survives a full compile. A kernel entry parameter must be a
linear GM layout (``RowMajor`` / ``ColumnMajor`` / ``VectorLayout``); the
supported way to reach this route is ``src.ptr`` + ``tla.make_tensor`` with the
fractal layout, as in ``tests/test_copy_lowering.py`` and the lit test.
"""

from __future__ import annotations

import pytest

import catlass.tla as tla
from catlass.tla.runtime import make_fake_tensor


def _nz_tensor(dtype, addrspace, rows: int = 32, cols: int = 32):
    """A padded nZ tile of ``rows`` x ``cols`` logical elements.

    nZ packs C0-row by 16-column fractals, so the row groups run
    ``ceil(rows / C0)`` and the column groups ``ceil(cols / 16)``; the row-group
    pitch is the column count rounded up to 16 times C0.
    """
    c0 = 32 // (dtype.width // 8)
    cols_ru = -(-cols // 16) * 16
    shape = (
        (c0, -(-rows // c0)),
        (16, -(-cols // 16)),
    )
    stride = (
        (1, cols_ru * c0),
        (c0, 16 * c0),
    )
    return make_fake_tensor(
        dtype,
        shape,
        stride,
        addrspace=addrspace,
        origin_shape=(rows, cols),
        layout_tag=tla.arch.nZ,
    )


def _gm(dtype, rows: int = 32, cols: int = 32):
    return _nz_tensor(dtype, tla.AddressSpace.gm, rows, cols)


def _l1(dtype, rows: int = 32, cols: int = 32):
    return _nz_tensor(dtype, tla.AddressSpace.l1, rows, cols)


@tla.kernel
def gm_nz_to_l1_nz_kernel(dst: tla.Tensor, src: tla.Tensor) -> None:
    with tla.cube():
        tla.copy(dst, src)


@tla.kernel
def gm_nz_to_l1_zn_kernel(dst: tla.Tensor, src: tla.Tensor) -> None:
    with tla.cube():
        tla.copy(dst, src)


@tla.kernel
def l1_nz_to_gm_nz_kernel(dst: tla.Tensor, src: tla.Tensor) -> None:
    with tla.cube():
        tla.copy(dst, src)


@tla.kernel
def gm_nz_tile_at_coord_kernel(dst: tla.Tensor, src: tla.Tensor) -> None:
    """The tiled-weight case: copy one nZ block out of a larger GM buffer."""
    tile = tla.tile_view(src, tla.make_shape(32, 32), tla.make_coord(1, 0))
    with tla.cube():
        tla.copy(dst, tile)


def test_gm_nz_to_l1_nz_is_accepted() -> None:
    mlir = gm_nz_to_l1_nz_kernel.dump_mlir(
        type_args=(_l1(tla.Float32), _gm(tla.Float32))
    )
    assert "tla.copy" in mlir
    assert "nZ" in mlir


@pytest.mark.parametrize(
    "dtype",
    (
        tla.Float32,
        tla.Float16,
        tla.BFloat16,
        tla.Int8,
        tla.UInt16,
        tla.UInt32,
        tla.Float8E4M3FN,
        tla.Float8E5M2,
    ),
)
def test_gm_nz_to_l1_nz_accepts_every_registered_dtype(dtype) -> None:
    mlir = gm_nz_to_l1_nz_kernel.dump_mlir(type_args=(_l1(dtype), _gm(dtype)))
    assert "tla.copy" in mlir


def test_gm_nz_to_l1_nz_refuses_an_unregistered_dtype() -> None:
    """i32 has no GM -> L1 nZ block move registered, so the frontend says so."""
    with pytest.raises(Exception, match="unsupported copy route"):
        gm_nz_to_l1_nz_kernel.dump_mlir(type_args=(_l1(tla.Int32), _gm(tla.Int32)))


def test_gm_nz_to_l1_zn_is_refused() -> None:
    """The mirrored packing is the whole route: nZ in, nZ out."""
    dst = make_fake_tensor(
        tla.Float32,
        ((16, 2), (8, 4)),
        ((8, 128), (1, 256)),
        addrspace=tla.AddressSpace.l1,
        origin_shape=(32, 32),
        layout_tag=tla.arch.zN,
    )
    with pytest.raises(Exception, match="unsupported copy route"):
        gm_nz_to_l1_zn_kernel.dump_mlir(type_args=(dst, _gm(tla.Float32)))


def test_l1_nz_to_gm_nz_is_refused() -> None:
    """The copy unit has no L1 -> GM direction at all."""
    with pytest.raises(Exception, match="Hardware unsupported copy route"):
        l1_nz_to_gm_nz_kernel.dump_mlir(type_args=(_gm(tla.Float32), _l1(tla.Float32)))


def test_gm_nz_tile_view_carries_its_coord() -> None:
    """A block of a prepacked buffer copies by carrying the tile coordinate.

    The payload the stub decodes has no element offset of its own, so a sliced
    GM nZ tile only lands in the right place if the coordinate survives into the
    descriptor.
    """
    src = make_fake_tensor(
        tla.Float32,
        ((8, 8), (16, 2)),
        ((1, 256), (8, 128)),
        addrspace=tla.AddressSpace.gm,
        origin_shape=(64, 32),
        layout_tag=tla.arch.nZ,
    )
    dst = make_fake_tensor(
        tla.Float32,
        ((8, 4), (16, 2)),
        ((1, 256), (8, 128)),
        addrspace=tla.AddressSpace.l1,
        origin_shape=(32, 32),
        layout_tag=tla.arch.nZ,
    )
    mlir = gm_nz_tile_at_coord_kernel.dump_mlir(type_args=(dst, src))
    assert "tla.tile_view" in mlir
    # The tile starts at row 32 of a 64-row buffer.
    assert "!tla.coord<32,0>" in mlir
    assert "tla.copy" in mlir


def test_gm_nz_pads_a_row_count_that_is_not_a_multiple_of_c0() -> None:
    """nZ rounds M up to C0 rows, so 20 rows occupy 3 row groups.

    Unlike the zN family there is no tight-pitch tag to reach for: the padding
    is part of the layout, and both sides describe the same buffer.
    """
    mlir = gm_nz_to_l1_nz_kernel.dump_mlir(
        type_args=(
            _l1(tla.Float32, rows=20, cols=20),
            _gm(tla.Float32, rows=20, cols=20),
        )
    )
    assert "!tla.shape<(8,3),(16,2)>" in mlir
    assert "!tla.shape<20,20>" in mlir
