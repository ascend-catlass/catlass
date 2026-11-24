"""Tensor construction inside a SIMT region is rejected at the call site.

A ``tla.vec.func(mode="simt")`` region is outlined into a per-thread function
whose buffers cross the launch ABI as one bare pointer each -- there is no
descriptor or layout on the other side. A view/tensor built *inside* the region
materializes a ``tla.tensor_desc`` whose base is the enclosing raw memref, so
the capture check used to reject the whole region with "this region indexes 1
raw memref(s) directly". That diagnostic describes the ABI, not the call the
user wrote, and cannot be acted on.

The supported pattern is to build the view/tensor outside the region and read
it inside through element access or ``ptr_with_coord``. These tests pin both
halves: the call-site rejection for every tensor-construction API, and the
outside/inside form that must keep working.
"""

from __future__ import annotations

import pytest

import catlass.tla as tla
from catlass.tla.runtime import make_fake_tensor


def _gm_tensor(shape: tuple[int, int] = (2, 16)) -> tla.Tensor:
    return make_fake_tensor(
        tla.Float32,
        shape,
        (shape[1], 1),
        addrspace=tla.AddressSpace.gm,
        origin_shape=shape,
    )


def _gm_out() -> tla.Tensor:
    return make_fake_tensor(
        tla.Float32,
        (64,),
        (1,),
        addrspace=tla.AddressSpace.gm,
        origin_shape=(64,),
    )


def _ub_tensor(extent: int = 64) -> tla.Tensor:
    return make_fake_tensor(
        tla.Float32,
        (extent,),
        (1,),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(extent,),
        layout_tag=tla.arch.RowMajor,
    )


# --- Rejection inside a SIMT region ------------------------------------------


@tla.kernel
def _tile_view_in_simt(x: tla.Tensor, out: tla.Tensor) -> None:
    # The shape from the report: build the view, then read its start address.
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=1):
            view = tla.tile_view(x, tla.make_shape(1, 16), tla.make_coord(1, 0))
            _ = view.ptr_with_coord
            out[0] = view[0, 0]


@tla.kernel
def _get_tile_in_simt(x: tla.Tensor, out: tla.Tensor) -> None:
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=1):
            view = tla.get_tile(x, tla.make_coord(0, 0), tla.make_shape(1, 16))
            out[0] = view[0, 0]


@tla.kernel
def _make_tensor_in_simt(x: tla.Tensor, out: tla.Tensor) -> None:
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=1):
            local = tla.make_tensor(
                x.ptr,
                tla.make_layout(tla.make_shape(16), tla.make_stride(1)),
                coord=tla.make_coord(0),
            )
            out[0] = local[0]


@tla.kernel
def _make_tensor_like_in_simt(x: tla.Tensor, out: tla.Tensor) -> None:
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=1):
            local = tla.make_tensor_like(x.ptr, x, tla.arch.RowMajor)
            out[0] = local[0]


@pytest.mark.parametrize(
    ("kernel", "op_name", "type_args"),
    [
        (_tile_view_in_simt, "tile_view", lambda: (_gm_tensor(), _gm_out())),
        (_get_tile_in_simt, "get_tile", lambda: (_gm_tensor(), _gm_out())),
        (_make_tensor_in_simt, "make_tensor", lambda: (_gm_tensor(), _gm_out())),
        (
            _make_tensor_like_in_simt,
            "make_tensor_like",
            lambda: (_ub_tensor(), _gm_out()),
        ),
    ],
)
def test_tensor_construction_inside_simt_is_rejected(kernel, op_name, type_args) -> None:
    with pytest.raises(
        Exception,
        match=rf"tla\.{op_name}: SIMT vec\.func 内不支持创建 {op_name}。",
    ):
        kernel.dump_mlir(type_args=type_args())


def test_tile_view_rejection_names_the_supported_pattern() -> None:
    # The diagnostic has to tell the user what to do instead: create the view
    # outside the region and read its start address inside.
    with pytest.raises(
        Exception,
        match=r"请在区域外创建视图，再在区域内读取 view\.ptr_with_coord。",
    ):
        _tile_view_in_simt.dump_mlir(type_args=(_gm_tensor(), _gm_out()))


# --- The supported outside/inside form ---------------------------------------


@tla.kernel
def _view_outside_pointer_inside(x: tla.Tensor, out: tla.Tensor) -> None:
    """Build the view outside the region, read it inside (the issue's positive case)."""
    view = tla.tile_view(x, tla.make_shape(1, 16), tla.make_coord(1, 0))
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=1):
            _ = view.ptr_with_coord
            out[0] = view[0, 0]


def test_view_outside_pointer_inside_simt_is_supported() -> None:
    mlir = _view_outside_pointer_inside.dump_mlir(type_args=(_gm_tensor(), _gm_out()))

    assert "tla.tile_view" in mlir
    assert "tla.tensor_ptr_with_coord" in mlir
    # The view is created before the region opens; the pointer read is inside.
    assert mlir.index("tla.tile_view") < mlir.index('"tla.vec.func"')
    assert mlir.index('"tla.vec.func"') < mlir.index("tla.tensor_ptr_with_coord")


@tla.kernel
def _make_tensor_outside_inside_simt(x: tla.Tensor, out: tla.Tensor) -> None:
    local = tla.make_tensor(
        x.ptr,
        tla.make_layout(tla.make_shape(16), tla.make_stride(1)),
        coord=tla.make_coord(0),
    )
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=1):
            out[0] = local[0]


def test_make_tensor_outside_simt_is_supported() -> None:
    mlir = _make_tensor_outside_inside_simt.dump_mlir(
        type_args=(_gm_tensor(), _gm_out())
    )

    assert "tla.make_tensor" in mlir
    assert mlir.index("tla.make_tensor") < mlir.index('"tla.vec.func"')


@tla.kernel
def _tile_view_inside_simd(x: tla.Tensor) -> None:
    # The restriction is SIMT-only: a SIMD region keeps building views in place.
    with tla.vector():
        with tla.vec.func(mode="simd"):
            _ = tla.tile_view(x, tla.make_shape(1, 16), tla.make_coord(1, 0))


def test_tile_view_inside_simd_is_still_allowed() -> None:
    mlir = _tile_view_inside_simd.dump_mlir(type_args=(_gm_tensor(),))

    assert "tla.tile_view" in mlir
