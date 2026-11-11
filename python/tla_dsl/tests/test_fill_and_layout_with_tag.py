"""Frontend tests for ``tla.make_layout_with_tag``, ``tla.get_tile`` and ``Tensor.fill``.

``make_layout_with_tag`` composes an ``!tla.layout`` from a flat 2-D logical
shape + dtype + tag without a reference tile; the layout trees come from the
same materializer ``make_tensor_like`` uses, so for equal extents/dtype/tag the
two APIs must produce identical layout types (parity guard).

``tla.get_tile`` extracts a sub-tile via ``tla.make_tensor``: the tile's coord
is ``coord + source.coord``, its physical shape is regenerated from the cropped
extent (strides pass through unchanged), and its ``origin_shape`` is the
requested extent cropped against the source's ``origin_shape`` (per dim
``min(shape[d], origin[d] - coord[d])``).

``Tensor.fill`` fills the tile itself: the emitted ``tla.fill`` carries the
tensor's own coord and ``origin_shape`` -- no caller-side region, no cropping.
"""

from __future__ import annotations

import re

import pytest

import catlass.tla as tla
from catlass.execution_lowering import TlaLoweringError
from catlass.tla.runtime import make_fake_tensor


def _l1_tensor(shape2d, dtype, tag):
    """An L1 zN/nZ tensor built via make_layout_with_tag + make_tensor."""
    layout = tla.make_layout_with_tag(shape2d, dtype, tag)
    ptr = tla.allocate(shape2d, dtype, tla.AddressSpace.l1, 512)
    return tla.make_tensor(ptr, layout)


# --- make_layout_with_tag ------------------------------------------------------


def test_make_layout_with_tag_parity_with_make_tensor_like() -> None:
    """Same extent/dtype/tag: make_layout_with_tag + make_tensor must emit the
    identical !tla.layout type as make_tensor_like(alloc, like_ref, tag)."""

    @tla.kernel
    def _kernel(mem_ref: tla.Tensor) -> None:
        ptr = tla.allocate((32, 64), tla.Float16, tla.AddressSpace.l1, 512)
        via_like = tla.make_tensor_like(ptr, mem_ref, tla.arch.zN)
        via_tag = tla.make_tensor(
            ptr, tla.make_layout_with_tag((32, 64), tla.Float16, tla.arch.zN)
        )
        _ = via_like, via_tag

    mem_ref = make_fake_tensor(
        tla.Float16,
        ((16, 2), (16, 4)),
        ((16, 256), (1, 256)),
        layout_tag=tla.arch.zN,
        origin_shape=(32, 64),
    )
    mlir = _kernel.dump_mlir(type_args=(mem_ref,))

    layout_re = re.compile(
        r"!tla\.layout<!tla\.shape<\(16,2\),\(16,4\)>, !tla\.stride<\(16,256\),\(1,256\)>, "
        r"!tla\.shape<32,64>, zN>"
    )
    layouts = layout_re.findall(mlir)
    assert len(layouts) == 2, f"expected two identical layout emissions, got {layouts}"
    assert layouts[0] == layouts[1], f"parity broken:\n{layouts[0]}\nvs\n{layouts[1]}"


def test_make_layout_with_tag_all_tags() -> None:
    """Every supported tag composes a layout whose origin equals the requested
    logical extent."""

    @tla.kernel
    def _kernel() -> None:
        ptr = tla.allocate((32, 64), tla.Float16, tla.AddressSpace.l1, 512)
        for tag in (
            tla.arch.RowMajor,
            tla.arch.ColumnMajor,
            tla.arch.zN,
            tla.arch.nZ,
            tla.arch.L0Clayout,
            tla.arch.zNUnAlign,
        ):
            _ = tla.make_tensor(ptr, tla.make_layout_with_tag((32, 64), tla.Float16, tag))

    mlir = _kernel.dump_mlir()
    assert "RowMajor" in mlir
    assert "ColumnMajor" in mlir
    assert ", zN>" in mlir
    assert ", nZ>" in mlir
    assert "L0Clayout" in mlir
    assert "zNUnAlign" in mlir


def test_make_layout_with_tag_rejects_nested_shape() -> None:
    @tla.kernel
    def _kernel() -> None:
        _ = tla.make_layout_with_tag(((16, 2), (16, 4)), tla.Float16, tla.arch.zN)

    with pytest.raises((TlaLoweringError, tla.TlaCoreAPIError), match="flat 2D logical shape"):
        _kernel.dump_mlir()


def test_make_layout_with_tag_rejects_bad_tag() -> None:
    @tla.kernel
    def _kernel() -> None:
        _ = tla.make_layout_with_tag((32, 64), tla.Float16, "zN")

    from catlass.execution_lowering import UnsupportedExecutionLowering
    with pytest.raises((TypeError, UnsupportedExecutionLowering), match="layout_tag must be a tla.arch layout sentinel"):
        _kernel.dump_mlir()


def test_make_layout_with_tag_accepts_mx_scale_tags() -> None:
    """The on-chip MxScale tags materialize (2-element / 32-byte C0) and the
    kernel compiles through the full pipeline."""

    @tla.kernel
    def _kernel() -> None:
        ptr = tla.allocate((32, 64), tla.Float8E8M0, tla.AddressSpace.l1, 512)
        a = tla.make_tensor(ptr, tla.make_layout_with_tag((32, 64), tla.Float8E8M0, tla.arch.zZMxScale))
        b = tla.make_tensor(ptr, tla.make_layout_with_tag((32, 64), tla.Float8E8M0, tla.arch.nNMxScale))
        _ = a, b

    mlir = _kernel.dump_mlir()
    assert ", zZMxScale>" in mlir
    assert ", nNMxScale>" in mlir


def test_make_layout_with_tag_mx_scale_requires_e8m0() -> None:
    """The MxScale tags fix C0 at 2 elements / 32 bytes -- an e8m0-only
    property -- so another element type is rejected up front instead of
    compiling into a silently wrong layout."""

    @tla.kernel
    def _kernel() -> None:
        ptr = tla.allocate((32, 64), tla.Float16, tla.AddressSpace.l1, 512)
        _ = tla.make_tensor(ptr, tla.make_layout_with_tag((32, 64), tla.Float16, tla.arch.zZMxScale))

    with pytest.raises((TlaLoweringError, tla.TlaCoreAPIError), match="requires element type"):
        _kernel.dump_mlir()


def test_make_layout_with_tag_rejects_gm_mx_scale_tags() -> None:
    """The GM-side MxScale tags describe global-memory storage (their NZFamily
    trees carry stride-0 leaves the on-chip materializer cannot express), so
    they are refused with a clear message instead of an opaque make_stride
    failure."""

    for tag in (
        tla.arch.RowMajorMxScaleA,
        tla.arch.ColMajorMxScaleA,
        tla.arch.RowMajorMxScaleB,
        tla.arch.ColMajorMxScaleB,
    ):

        def _build(tag=tag):
            @tla.kernel
            def _kernel() -> None:
                ptr = tla.allocate((32, 64), tla.Float8E8M0, tla.AddressSpace.l1, 512)
                _ = tla.make_tensor(ptr, tla.make_layout_with_tag((32, 64), tla.Float8E8M0, tag))

            return _kernel

        with pytest.raises(
            (TlaLoweringError, tla.TlaCoreAPIError), match="unsupported layout tag"
        ):
            _build().dump_mlir()


def test_make_layout_with_tag_rejects_outside_kernel() -> None:
    # Outside a @tla.kernel the frontend state check fires before anything else.
    with pytest.raises(tla.TlaIRNotExecutableError, match="make_layout_with_tag"):
        tla.make_layout_with_tag((32, 64), tla.Float16, tla.arch.zN)


# --- get_tile -------------------------------------------------------------------


def test_get_tile_offsets_coord_and_regenerates_shape() -> None:
    """coord adds to the source's own coord; the physical shape is regenerated
    from the cropped extent (strides keep the source buffer's pitch) and the
    origin becomes the requested (uncropped) extent."""

    @tla.kernel
    def _kernel() -> None:
        t = _l1_tensor((32, 64), tla.Float16, tla.arch.zN)
        tile = tla.get_tile(t, tla.make_coord(0, 32), tla.make_shape(32, 32))
        _ = t, tile

    mlir = _kernel.dump_mlir()
    # Physical shape regenerated for the (32, 32) tile; strides stay the
    # source's; origin is the tile extent.
    assert re.search(
        r"!tla\.layout<!tla\.shape<\(16,2\),\(16,2\)>, !tla\.stride<\(16,256\),\(1,512\)>, "
        r"!tla\.shape<32,32>, zN>",
        mlir,
    )
    assert "!tla.coord<0,32>" in mlir


def test_get_tile_crops_shape_against_origin() -> None:
    """The extent is cropped per dim min(shape[d], origin[d] - coord[d]); it
    never runs past the source's logical edge."""

    @tla.kernel
    def _kernel() -> None:
        t = _l1_tensor((32, 64), tla.Float16, tla.arch.zN)
        # Requested (32, 32) from (16, 48): dim0 -> 32-16=16, dim1 -> 64-48=16.
        tile = tla.get_tile(t, tla.make_coord(16, 48), tla.make_shape(32, 32))
        _ = t, tile

    mlir = _kernel.dump_mlir()
    assert "!tla.coord<16,48>" in mlir
    assert re.search(r"!tla\.shape<16,16>, zN>", mlir)


def test_get_tile_on_tile_accumulates_coord() -> None:
    """get_tile of a get_tile: coords add up and the crop runs against the
    intermediate tile's own origin."""

    @tla.kernel
    def _kernel() -> None:
        t = _l1_tensor((32, 64), tla.Float16, tla.arch.zN)
        tile1 = tla.get_tile(t, tla.make_coord(16, 32), tla.make_shape(16, 32))
        # (0, 8) within tile1 (origin (16, 32)) -> absolute (16, 40);
        # extent (16, 8) fits.
        tile2 = tla.get_tile(tile1, tla.make_coord(0, 8), tla.make_shape(16, 8))
        _ = t, tile1, tile2

    mlir = _kernel.dump_mlir()
    assert "!tla.coord<16,40>" in mlir
    assert re.search(r"!tla\.shape<16,8>, zN>", mlir)


def test_get_tile_then_fill_covers_the_tile() -> None:
    """The canonical pad-zeroing chain: tile the pad region, then fill it --
    the emitted tla.fill carries the tile's coord and origin verbatim."""

    @tla.kernel
    def _kernel() -> None:
        t = _l1_tensor((32, 64), tla.Float8E4M3FN, tla.arch.zN)
        pad = tla.get_tile(t, tla.make_coord(0, 32), tla.make_shape(32, 32))
        with tla.cube():
            pad.fill(0)
        _ = t, pad

    mlir = _kernel.dump_mlir()
    assert "!tla.coord<0,32>" in mlir
    assert "!tla.shape<32,32>" in mlir
    assert "tla.fill" in mlir


def test_get_tile_rejects_plain_tuple_args() -> None:
    @tla.kernel
    def _kernel() -> None:
        t = _l1_tensor((32, 64), tla.Float16, tla.arch.zN)
        _ = tla.get_tile(t, (0, 32), tla.make_shape(32, 32))

    with pytest.raises(Exception, match="tla.make_coord"):  # noqa: B017
        _kernel.dump_mlir()


# --- Tensor.fill ----------------------------------------------------------------


def test_fill_emits_whole_tile_region() -> None:
    """fill() covers the tile itself: coord = the tensor's own coord, extent =
    origin_shape -- no caller-side region, no cropping."""

    @tla.kernel
    def _kernel() -> None:
        t = _l1_tensor((32, 64), tla.Float8E4M3FN, tla.arch.zN)
        with tla.cube():
            t.fill(0)
        _ = t

    mlir = _kernel.dump_mlir()
    assert "!tla.coord<0,0>" in mlir
    assert "!tla.shape<32,64>" in mlir


def test_fill_on_view_covers_the_view() -> None:
    """fill() on a tile_view emits the view's own (global) coord and its
    (already cropped) origin."""

    @tla.kernel
    def _kernel() -> None:
        t = _l1_tensor((32, 64), tla.Float8E4M3FN, tla.arch.zN)
        # Tile coord (1, 1) on a (16, 32) tile = element offset (16, 32).
        v = tla.tile_view(t, tla.make_shape(16, 32), tla.make_coord(1, 1))
        with tla.cube():
            v.fill(0)
        _ = t, v

    mlir = _kernel.dump_mlir()
    assert "!tla.coord<16,32>" in mlir
    assert "!tla.shape<16,32>" in mlir


def test_fill_zero_value_bit_pattern_fp8() -> None:
    @tla.kernel
    def _kernel() -> None:
        t = _l1_tensor((32, 64), tla.Float8E4M3FN, tla.arch.zN)
        with tla.cube():
            t.fill(0)
        _ = t

    mlir = _kernel.dump_mlir()
    assert "tla.fill" in mlir


def test_fill_rejects_nonzero_for_packed_types() -> None:
    """Packed fp4/fp8 retain the zero-only contract."""

    for dtype in (tla.Float8E4M3FN, tla.Float8E5M2, tla.Float4E2M1, tla.Float4E1M2):

        def _build(dtype=dtype):
            @tla.kernel
            def _kernel() -> None:
                t = _l1_tensor((32, 64), dtype, tla.arch.zN)
                with tla.cube():
                    t.fill(1)
                _ = t

            return _kernel

        with pytest.raises((TlaLoweringError, tla.TlaCoreAPIError), match="supports only value 0"):
            _build().dump_mlir()


def test_fill_fp8_zero_only() -> None:
    @tla.kernel
    def _kernel_zero() -> None:
        t = _l1_tensor((32, 64), tla.Float8E4M3FN, tla.arch.zN)
        with tla.cube():
            t.fill(0)
        _ = t

    mlir = _kernel_zero.dump_mlir()
    assert "tla.fill" in mlir

    @tla.kernel
    def _kernel_nonzero() -> None:
        t = _l1_tensor((32, 64), tla.Float8E4M3FN, tla.arch.zN)
        with tla.cube():
            t.fill(1)
        _ = t

    with pytest.raises((TlaLoweringError, tla.TlaCoreAPIError), match="supports only value 0"):
        _kernel_nonzero.dump_mlir()


def test_fill_rejects_i8() -> None:
    @tla.kernel
    def _kernel() -> None:
        t = _l1_tensor((32, 64), tla.Int8, tla.arch.zN)
        with tla.cube():
            t.fill(0)
        _ = t

    with pytest.raises((TlaLoweringError, tla.TlaCoreAPIError), match="unsupported element type"):
        _kernel.dump_mlir()


def test_fill_rejects_ub_tensor() -> None:
    @tla.kernel
    def _kernel() -> None:
        layout = tla.make_layout_with_tag((32, 64), tla.Float16, tla.arch.zN)
        ptr = tla.allocate((32, 64), tla.Float16, tla.AddressSpace.ub, 512)
        t = tla.make_tensor(ptr, layout)
        with tla.cube():
            t.fill(0)
        _ = t

    with pytest.raises((TlaLoweringError, tla.TlaCoreAPIError), match="expected addrspace l1"):
        _kernel.dump_mlir()


def test_fill_rejects_rowmajor() -> None:
    @tla.kernel
    def _kernel() -> None:
        t = _l1_tensor((32, 64), tla.Float16, tla.arch.RowMajor)
        with tla.cube():
            t.fill(0)
        _ = t

    with pytest.raises((TlaLoweringError, tla.TlaCoreAPIError), match="expected layout tag zN or nZ"):
        _kernel.dump_mlir()


def test_fill_rejects_outside_cube() -> None:
    @tla.kernel
    def _kernel() -> None:
        t = _l1_tensor((32, 64), tla.Float16, tla.arch.zN)
        # No tla.cube() around the fill.
        t.fill(0)
        _ = t

    with pytest.raises(Exception, match="cube"):  # noqa: B017
        _kernel.dump_mlir()
