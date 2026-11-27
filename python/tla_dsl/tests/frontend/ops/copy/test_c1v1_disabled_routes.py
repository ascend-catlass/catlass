"""C1V1 (Ascend 950 CV 1:1) rejects the tightly coupled L0C->UB / L1->UB copies."""

from __future__ import annotations

import pytest

from catlass import tla
from catlass.core_api import _remap_tensor_like_prefix_fields_for_layout_trees
from catlass.runtime import TlaCoreAPIError
from catlass.tla.runtime import make_fake_tensor

_ROWS = 128
_COLS = 128


def _fractal(dtype, addrspace, layout_tag):
    geometry = _remap_tensor_like_prefix_fields_for_layout_trees(
        (_ROWS, _COLS), dtype.dtype, str(layout_tag)
    )
    assert geometry is not None
    return make_fake_tensor(
        dtype,
        geometry.shape,
        geometry.stride,
        addrspace=addrspace,
        origin_shape=(_ROWS, _COLS),
        layout_tag=layout_tag,
    )


def _l0c():
    return _fractal(tla.Float32, tla.AddressSpace.l0c, tla.arch.L0Clayout)


def _l1():
    return _fractal(tla.Float32, tla.AddressSpace.l1, tla.arch.zN)


def _ub():
    return make_fake_tensor(
        tla.Float32,
        (_ROWS, _COLS),
        (_COLS, 1),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(_ROWS, _COLS),
        layout_tag=tla.arch.RowMajor,
    )


@tla.kernel(core_ratio=tla.arch.C1V1)
def c1v1_copy_kernel(dst: tla.Tensor, src: tla.Tensor) -> None:
    with tla.cube():
        tla.copy(dst, src)
    with tla.vector():
        pass


@tla.kernel(core_ratio=tla.arch.C1V2)
def c1v2_copy_kernel(dst: tla.Tensor, src: tla.Tensor) -> None:
    with tla.cube():
        tla.copy(dst, src)
    with tla.vector():
        pass


@pytest.mark.parametrize(
    ("src", "route"),
    [(_l0c, "l0c -> ub"), (_l1, "l1 -> ub")],
    ids=["l0c_to_ub", "l1_to_ub"],
)
def test_c1v1_rejects_tightly_coupled_ub_paths(src, route: str) -> None:
    with pytest.raises(TlaCoreAPIError, match="core_ratio=tla.arch.C1V1") as excinfo:
        c1v1_copy_kernel.dump_mlir(type_args=(_ub(), src()))
    assert route in str(excinfo.value)


def test_c1v2_still_accepts_l0c_to_ub() -> None:
    mlir = c1v2_copy_kernel.dump_mlir(type_args=(_ub(), _l0c()))
    assert "tla.copy" in mlir


def test_c1v2_l1_to_ub_keeps_the_generic_unsupported_route_diagnostic() -> None:
    """Only C1V1 closes L1->UB; C1V2 keeps the pre-existing route diagnostic."""
    with pytest.raises(
        TlaCoreAPIError, match="copy: unsupported copy route"
    ) as excinfo:
        c1v2_copy_kernel.dump_mlir(type_args=(_ub(), _l1()))
    message = str(excinfo.value)
    assert "unsupported copy route l1,zN -> ub,RowMajor" in message
    assert "core_ratio=tla.arch.C1V1" not in message
