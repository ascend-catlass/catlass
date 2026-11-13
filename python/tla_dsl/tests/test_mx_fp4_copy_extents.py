# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import pytest

import catlass.tla as tla
from catlass.tla.runtime import make_fake_tensor

# Frontend coverage for the whole-byte rule on the fp4 copy routes that stage
# through a byte descriptor: GM -> UB, UB -> GM and UB -> L1. Those routes
# convert the descriptor to int8 denomination and halve it, which is exact only
# on whole bytes, so core_api refuses a value it cannot show to be even. Two
# fp4 elements share a byte; an odd row ends mid-byte, and the other half of
# that byte is an element outside the tile.
#
# Three fields are halved and so three are checked: the innermost origin
# extent, the innermost origin coordinate and -- on a RowMajor tile -- the row
# stride. Each has a static-odd case and a runtime case below, because a
# runtime leaf reaches the predicate as None and an ``isinstance(v, int) and
# v % 2`` test waves it straight through.
#
# This is the automated half of the rule: the refusal happens while tracing, so
# it needs no NPU and CI runs it. The byte-level half -- that a legal extent
# really does copy byte for byte and leaves the storage past the tile alone --
# does need a device and lives in
# tests/end_to_end/mx_fp4_packed_extent_copy_op.py, which pytest does not
# collect (conftest.py ignores end_to_end/*).

ROWS = 2
#: Row pitch of the GM parent, in elements. Even, so it is never itself the
#: thing being refused.
PITCH = 128

FP4 = (tla.Float4E2M1, tla.Float4E1M2)
#: Even extents, including ones that are not a multiple of 32: 34 elements are
#: exactly 17 bytes, so the halving is lossless and the copy is legal. Keeping
#: them here is what pins the rule to whole bytes rather than to MX groups.
WHOLE_BYTES = (32, 34, 40, 64)
#: Odd extents end mid-byte and are the ones that cannot be expressed.
PARTIAL_BYTES = (33, 35)


def _gm(dtype, extent):
    """A GM tile of `extent` fp4 elements per row, inside a PITCH-wide parent."""
    return make_fake_tensor(
        dtype,
        (ROWS, extent),
        (PITCH, 1),
        origin_shape=(ROWS, extent),
        layout_tag=tla.arch.RowMajor,
    )


def _kernels(dtype, extent):
    """A store and a load of one `extent`-wide fp4 tile between GM and UB.

    The kernels are built per case because the UB allocation needs the element
    type as a concrete ``tla.Numeric``; a tensor argument's ``.dtype`` is a
    string, so it cannot carry it.
    """

    def ub():
        return tla.make_tensor(
            tla.allocate((ROWS, PITCH), dtype, tla.AddressSpace.ub, 256),
            tla.make_layout(tla.make_shape(ROWS, extent), tla.make_stride(PITCH, 1)),
        )

    @tla.kernel
    def store(gm: tla.Tensor) -> None:
        with tla.vector():
            tla.copy(gm, ub())

    @tla.kernel
    def load(gm: tla.Tensor) -> None:
        with tla.vector():
            tla.copy(ub(), gm)

    return store, load


@pytest.mark.parametrize("dtype", FP4)
@pytest.mark.parametrize("extent", WHOLE_BYTES)
def test_even_extent_is_accepted(dtype, extent) -> None:
    store, load = _kernels(dtype, extent)
    gm = _gm(dtype, extent)
    store.dump_mlir(type_args=(gm,))
    load.dump_mlir(type_args=(gm,))


@pytest.mark.parametrize("dtype", FP4)
@pytest.mark.parametrize("extent", PARTIAL_BYTES)
@pytest.mark.parametrize("which", (0, 1), ids=("store", "load"))
def test_odd_extent_is_refused(dtype, extent, which) -> None:
    kernel = _kernels(dtype, extent)[which]
    with pytest.raises(tla.TlaCoreAPIError, match="must be even"):
        kernel.dump_mlir(type_args=(_gm(dtype, extent),))


@pytest.mark.parametrize("dtype", FP4)
def test_runtime_extent_is_refused(dtype) -> None:
    """A runtime extent cannot be shown to be even, so it is refused.

    A runtime extent cannot be shown to be even, so it is refused. This is the
    case the earlier static-only check waved through: it returned None for
    anything that was not a static int, while the halving happens on device on
    whatever value arrives. The reported reproducer compiled one UB -> GM
    kernel with ``extent: tla.Int32`` and replayed it with 32, 33 and 34; only
    33 was wrong, and nothing in the frontend had objected.
    """

    @tla.kernel
    def store(gm: tla.Tensor, extent: tla.Int32) -> None:
        ub = tla.make_tensor(
            tla.allocate((ROWS, PITCH), dtype, tla.AddressSpace.ub, 256),
            tla.make_layout(tla.make_shape(ROWS, extent), tla.make_stride(PITCH, 1)),
        )
        with tla.vector():
            tla.copy(gm, ub)

    with pytest.raises(tla.TlaCoreAPIError, match="not known until run time"):
        store.dump_mlir(type_args=(_gm(dtype, 32), tla.Int32(32)))


@pytest.mark.parametrize("dtype", FP4)
@pytest.mark.parametrize("which", (0, 1), ids=("store", "load"))
def test_odd_row_stride_is_refused(dtype, which) -> None:
    """An odd row stride puts every row after the first half a byte off.

    The extent here is a whole number of bytes, so only the stride can be the
    thing refused -- which is what makes this a test of the stride branch and
    not a second extent test wearing a different pitch.
    """
    odd_pitch = PITCH + 1
    gm = make_fake_tensor(
        dtype,
        (ROWS, 32),
        (odd_pitch, 1),
        origin_shape=(ROWS, 32),
        layout_tag=tla.arch.RowMajor,
    )

    def ub():
        return tla.make_tensor(
            tla.allocate((ROWS, odd_pitch), dtype, tla.AddressSpace.ub, 256),
            tla.make_layout(tla.make_shape(ROWS, 32), tla.make_stride(odd_pitch, 1)),
        )

    @tla.kernel
    def store(arg: tla.Tensor) -> None:
        with tla.vector():
            tla.copy(arg, ub())

    @tla.kernel
    def load(arg: tla.Tensor) -> None:
        with tla.vector():
            tla.copy(ub(), arg)

    with pytest.raises(tla.TlaCoreAPIError, match="row stride must be even"):
        (store, load)[which].dump_mlir(type_args=(gm,))


@pytest.mark.parametrize("dtype", FP4)
@pytest.mark.parametrize("which", (0, 1), ids=("store", "load"))
def test_odd_origin_coordinate_is_refused(dtype, which) -> None:
    """An odd innermost origin coordinate starts the tile mid-byte.

    The coordinate is set on the tensor directly rather than through
    ``tla.tile_view``, whose coordinate is in tile units: it multiplies by the
    tile extent, and an extent that got this far is already even, so no odd
    element coordinate can come out of it. A tile that arrives already offset
    can still carry one, which is the case this pins.
    """
    gm = make_fake_tensor(
        dtype,
        (ROWS, 32),
        (PITCH, 1),
        origin_shape=(ROWS, 32),
        coord=(0, 1),
        layout_tag=tla.arch.RowMajor,
    )
    store, load = _kernels(dtype, 32)
    with pytest.raises(tla.TlaCoreAPIError, match="origin coordinate must be even"):
        (store, load)[which].dump_mlir(type_args=(gm,))


@pytest.mark.parametrize("dtype", FP4)
def test_runtime_origin_coordinate_is_refused(dtype) -> None:
    """A runtime coordinate cannot be shown to be even, so it is refused.

    This is the extent hole again, one field over: the descriptor carries a
    runtime leaf as None, so a static-only ``% 2`` accepted it and the device
    then truncated ``coord1 / 2``. A tile asked for at element 1 was copied
    from element 0, silently and with no frontend objection.
    """

    @tla.kernel
    def store(arg: tla.Tensor, off: tla.Int32) -> None:
        tile = tla.tile_view(arg, tla.make_shape(ROWS, 32), tla.make_coord(0, off))
        ub = tla.make_tensor(
            tla.allocate((ROWS, PITCH), dtype, tla.AddressSpace.ub, 256),
            tla.make_layout(tla.make_shape(ROWS, 32), tla.make_stride(PITCH, 1)),
        )
        with tla.vector():
            tla.copy(tile, ub)

    parent = make_fake_tensor(
        dtype,
        (ROWS, PITCH),
        (PITCH, 1),
        origin_shape=(ROWS, PITCH),
        layout_tag=tla.arch.RowMajor,
    )
    with pytest.raises(tla.TlaCoreAPIError, match="not known until run time"):
        store.dump_mlir(type_args=(parent, tla.Int32(0)))


@pytest.mark.parametrize("dtype", FP4)
@pytest.mark.parametrize("extent", PARTIAL_BYTES)
def test_odd_extent_is_refused_on_ub_to_l1(dtype, extent) -> None:
    """UB -> L1 halves the descriptor too, so it carries the same rule.

    ``copyUBToL1Fp4`` converts a RowMajor source through the same
    ``fp4RowMajorDescToBytes`` the GM routes use, and the zN destination
    through ``fp4zNDescToBytes``; both halve the packed extent. Bounding the
    rule to GM <-> UB alone let this route back into the silent half-byte
    error the rule exists to stop, which is what this case pins down.
    """

    @tla.kernel
    def stage(arg: tla.Tensor) -> None:
        ub = tla.make_tensor(
            tla.allocate((ROWS, PITCH), dtype, tla.AddressSpace.ub, 256),
            tla.make_layout(tla.make_shape(ROWS, extent), tla.make_stride(PITCH, 1)),
        )
        l1 = tla.make_tensor_like(
            tla.allocate((ROWS, PITCH), dtype, tla.AddressSpace.l1, 512),
            arg,
            tla.arch.zN,
        )
        with tla.vector():
            tla.copy(l1, ub)

    with pytest.raises(tla.TlaCoreAPIError, match="must be even"):
        stage.dump_mlir(type_args=(_gm(dtype, extent),))


# The zN C0-column pitch, on UB -> L1. ``copyUBToL1Fp4`` converts the
# destination with ``fp4zNDescToBytes`` on both registered pairings, which
# halves ``stride3`` -- the pitch between C0 columns -- so that pitch carries
# the same even requirement as a RowMajor row stride. The layouts the frontend
# materializes derive it as a whole number of C0s and so are always even; the
# explicit nested layout below is the only way an odd pitch arrives.
#
# The pair matters more than either case alone. The halved stride sits at a
# different depth per layout (``stride0``, the first of a RowMajor tile's two
# leaves, against ``stride3``, the last of a zN tile's four), so a check that
# reuses one layout's position on the other reads a nested tuple where an
# integer belongs and refuses every tile of that layout. A refusal case alone
# passes just as happily under that bug as under a correct rule.

#: A (16, 128) fp4 zN tile: C0 of 64 fp4 elements, 16 rows to a fractal, so a
#: materialized pitch between C0 columns of 16 * 64.
ZN_ROWS, ZN_COLS = 16, 128
ZN_SHAPE = ((16, 1), (64, 2))
ZN_PITCH = 1024


def _zn_stage(dtype, pitch):
    """A UB RowMajor -> L1 zN copy whose destination C0-column pitch is `pitch`."""

    @tla.kernel
    def stage() -> None:
        ub = tla.make_tensor(
            tla.allocate((ZN_ROWS, ZN_COLS), dtype, tla.AddressSpace.ub, 256),
            tla.make_layout(
                tla.make_shape(ZN_ROWS, ZN_COLS), tla.make_stride(ZN_COLS, 1)
            ),
        )
        l1 = tla.make_tensor(
            tla.allocate((ZN_ROWS, ZN_COLS), dtype, tla.AddressSpace.l1, 512),
            tla.make_layout(
                tla.make_shape(ZN_SHAPE[0], ZN_SHAPE[1]),
                tla.make_stride((64, ZN_PITCH), (1, pitch)),
                layoutTag=tla.arch.zN,
            ),
        )
        with tla.vector():
            tla.copy(l1, ub)

    return stage


@pytest.mark.parametrize("dtype", FP4)
def test_even_zn_c0_column_pitch_is_accepted(dtype) -> None:
    """The pitch a materialized zN layout carries copies losslessly."""
    _zn_stage(dtype, ZN_PITCH).dump_mlir(type_args=())


@pytest.mark.parametrize("dtype", FP4)
def test_odd_zn_c0_column_pitch_is_refused(dtype) -> None:
    """An odd pitch starts every C0 column after the first one nibble early.

    ``stride3 / 2`` truncates 1025 to 512 rather than 512.5, so the second C0
    column reads from half a byte before where the tile puts it.
    """
    with pytest.raises(tla.TlaCoreAPIError, match="C0-column pitch must be even"):
        _zn_stage(dtype, ZN_PITCH + 1).dump_mlir(type_args=())

