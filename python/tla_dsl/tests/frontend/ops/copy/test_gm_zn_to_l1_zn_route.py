"""Frontend guard for the GM zN -> L1 zN copy route.

A GM tile whose host-side buffer is already packed in fractal order is tagged
``zN``. Both endpoints then carry the same packing, so the copy is a plain block
move (no ND -> zN reorder) and the route is keyed on the mirrored tag rather
than on a RowMajor source.

Only the aligned ``zN`` is a GM source tag. ``zNUnAlign`` -- the tight M-axis
packing -- belongs to the ``UB`` -> ``L1`` route, where the tile is built in UB
and its pitch is the vector unit's business. A GM buffer read as ``zN`` has to
be padded to whole 16-row fractals instead, which this file pins as a negative
case.

The dtype set matches the RowMajor -> zN route: the element width only selects
C0, and no element-format conversion happens on the way in.

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


def _zn_tensor(dtype, addrspace, rows: int = 32, cols: int = 32):
    """A tightly packed zN tile of ``rows`` x ``cols`` logical elements."""
    c0 = 32 // (dtype.width // 8)
    c0_per_fractal = 16
    shape = (
        (c0_per_fractal, rows // c0_per_fractal),
        (c0, cols // c0),
    )
    stride = (
        (c0, c0_per_fractal * c0),
        (1, rows * c0),
    )
    return make_fake_tensor(
        dtype,
        shape,
        stride,
        addrspace=addrspace,
        origin_shape=(rows, cols),
        layout_tag=tla.arch.zN,
    )


def _gm_unaligned(dtype, rows: int, cols: int = 32):
    """A tight zNUnAlign GM tile: row pitch ``rows * C0``, no fractal padding."""
    c0 = 32 // (dtype.width // 8)
    return make_fake_tensor(
        dtype,
        ((rows, 1), (c0, cols // c0)),
        ((c0, rows * c0), (1, rows * c0)),
        addrspace=tla.AddressSpace.gm,
        origin_shape=(rows, cols),
        layout_tag=tla.arch.zNUnAlign,
    )


def _gm(dtype, rows: int = 32, cols: int = 32):
    return _zn_tensor(dtype, tla.AddressSpace.gm, rows, cols)


def _l1(dtype, rows: int = 32, cols: int = 32):
    return _zn_tensor(dtype, tla.AddressSpace.l1, rows, cols)


@tla.kernel
def gm_zn_to_l1_zn_kernel(dst: tla.Tensor, src: tla.Tensor) -> None:
    with tla.cube():
        tla.copy(dst, src)


@tla.kernel
def gm_zn_to_l1_nz_kernel(dst: tla.Tensor, src: tla.Tensor) -> None:
    with tla.cube():
        tla.copy(dst, src)


@tla.kernel
def l1_zn_to_gm_zn_kernel(dst: tla.Tensor, src: tla.Tensor) -> None:
    with tla.cube():
        tla.copy(dst, src)


@tla.kernel
def gm_zn_tile_at_coord_kernel(dst: tla.Tensor, src: tla.Tensor) -> None:
    """The tiled-weight case: copy one zN block out of a larger GM buffer."""
    tile = tla.tile_view(src, tla.make_shape(32, 32), tla.make_coord(1, 0))
    with tla.cube():
        tla.copy(dst, tile)


def test_gm_zn_to_l1_zn_is_accepted() -> None:
    mlir = gm_zn_to_l1_zn_kernel.dump_mlir(
        type_args=(_l1(tla.Float32), _gm(tla.Float32))
    )
    assert "tla.copy" in mlir
    assert "zN" in mlir


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
def test_gm_zn_to_l1_zn_accepts_every_registered_dtype(dtype) -> None:
    mlir = gm_zn_to_l1_zn_kernel.dump_mlir(type_args=(_l1(dtype), _gm(dtype)))
    assert "tla.copy" in mlir


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
def test_gm_unaligned_zn_is_refused_for_every_dtype(dtype) -> None:
    """A tight GM source is not a registered route, whatever its element width.

    The tag is legitimate on UB -> L1, where the source tile is built in UB; a GM
    buffer must be padded to whole 16-row fractals and tagged zN instead. Refusing
    it here rather than lowering it is what keeps the block move's row-group
    arithmetic sound at the GM end.
    """
    with pytest.raises(Exception, match="unsupported copy route"):
        gm_zn_to_l1_zn_kernel.dump_mlir(
            type_args=(_l1(dtype, rows=20), _gm_unaligned(dtype, 20))
        )


def test_gm_zn_to_l1_zn_refuses_an_unregistered_dtype() -> None:
    """i32 has no GM -> L1 zN block move registered, so the frontend says so."""
    with pytest.raises(Exception, match="unsupported copy route"):
        gm_zn_to_l1_zn_kernel.dump_mlir(type_args=(_l1(tla.Int32), _gm(tla.Int32)))


def test_gm_unaligned_zn_to_l1_zn_refuses_an_unregistered_dtype() -> None:
    """Refused for the tag first, so the dtype never gets a say."""
    with pytest.raises(Exception, match="unsupported copy route"):
        gm_zn_to_l1_zn_kernel.dump_mlir(
            type_args=(_l1(tla.Int32, rows=20), _gm_unaligned(tla.Int32, 20))
        )


def test_gm_zn_to_l1_nz_is_refused() -> None:
    """The mirrored packing is the whole route: zN in, zN out."""
    dst = make_fake_tensor(
        tla.Float32,
        ((8, 4), (16, 2)),
        ((1, 256), (8, 128)),
        addrspace=tla.AddressSpace.l1,
        origin_shape=(32, 32),
        layout_tag=tla.arch.nZ,
    )
    with pytest.raises(Exception, match="unsupported copy route"):
        gm_zn_to_l1_nz_kernel.dump_mlir(type_args=(dst, _gm(tla.Float32)))


def test_l1_zn_to_gm_zn_is_refused() -> None:
    """The copy unit has no L1 -> GM direction at all."""
    with pytest.raises(Exception, match="Hardware unsupported copy route"):
        l1_zn_to_gm_zn_kernel.dump_mlir(type_args=(_gm(tla.Float32), _l1(tla.Float32)))


def test_gm_zn_tile_view_carries_its_coord() -> None:
    """A block of a prepacked buffer copies by carrying the tile coordinate.

    The payload the stub decodes has no element offset of its own, so a sliced
    GM zN tile only lands in the right place if the coordinate survives into the
    descriptor.
    """
    src = make_fake_tensor(
        tla.Float32,
        ((16, 4), (8, 4)),
        ((8, 128), (1, 512)),
        addrspace=tla.AddressSpace.gm,
        origin_shape=(64, 32),
        layout_tag=tla.arch.zN,
    )
    dst = make_fake_tensor(
        tla.Float32,
        ((16, 2), (8, 4)),
        ((8, 128), (1, 256)),
        addrspace=tla.AddressSpace.l1,
        origin_shape=(32, 32),
        layout_tag=tla.arch.zN,
    )
    mlir = gm_zn_tile_at_coord_kernel.dump_mlir(type_args=(dst, src))
    assert "tla.tile_view" in mlir
    # The tile starts at row 32 of a 64-row buffer.
    assert "!tla.coord<32,0>" in mlir
    assert "tla.copy" in mlir
