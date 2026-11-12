from __future__ import annotations

from catlass.tla.runtime import make_fake_tensor


from typing import Any

import pytest

import catlass.tla as tla

from catlass.execution_lowering import TlaLoweringError
from catlass.runtime import TlaCoreAPIError
import catlass.runtime as runtime_mod  # noqa: F401


def _ub_tensor(
    dtype: type[tla.Numeric] = tla.Int32,
    extent: int = 64,
) -> tla.Tensor:
    return make_fake_tensor(
        dtype,
        (extent,),
        (1,),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(extent,),
        layout_tag=tla.arch.RowMajor,
    )


def _gm_tensor(
    dtype: type[tla.Numeric] = tla.Int32,
    shape: tuple[int, ...] = (64,),
) -> tla.Tensor:
    if len(shape) == 1:
        stride: tuple[int, ...] = (1,)
    elif len(shape) == 2:
        stride = (shape[1], 1)
    else:
        raise ValueError(f"unsupported rank for test helper: {shape!r}")
    return make_fake_tensor(
        dtype,
        shape,
        stride,
        origin_shape=shape,
        layout_tag=tla.arch.RowMajor,
    )


@tla.kernel
def atomic_add_l0c2gm_fake_atomic_mode(
    l0c_tensor: tla.Tensor, gm_tensor: tla.Tensor
) -> None:
    """Test atomic add mode, data path from L0C to GM"""

    _m, _n = 32, 32
    l0c_tile = tla.tile_view(l0c_tensor, tla.make_shape(_m, _n), tla.make_coord(0, 0))
    gm_tile = tla.tile_view(gm_tensor, tla.make_shape(_m, _n), tla.make_coord(0, 0))
    with tla.cube():
        tla.copy(
            gm_tile,
            l0c_tile,
            tla.params.CopyL0C2DstParams(unit_flag=0b11, atomic_mode=2),
        )


@tla.kernel
def atomic_add_ub2gm(ub_tensor: tla.Tensor, gm_tensor: tla.Tensor) -> None:
    """Test atomic add mode, data path from UB to GM"""

    _vec_len = 64
    x_ub = tla.tile_view(ub_tensor, tla.make_shape(_vec_len), tla.make_coord(0))
    x_gm = tla.tile_view(gm_tensor, tla.make_shape(_vec_len), tla.make_coord(0))
    with tla.vector():
        tla.copy(
            x_gm,
            x_ub,
            tla.params.CopyUbToGmParams(atomic_mode=tla.params.AtomicMode.ADD),
        )


@tla.kernel
def atomic_add_ub2gm_direct_arg(ub_tensor: tla.Tensor, gm_tensor: tla.Tensor) -> None:
    """Atomic UB→GM with kernel-arg GM dst (no tile_view), including dynamic if."""

    with tla.vector():
        if tla.arch.block_idx() == 0:
            tla.copy(
                gm_tensor,
                ub_tensor,
                tla.params.CopyUbToGmParams(atomic_mode=tla.params.AtomicMode.ADD),
            )


@tla.kernel
def atomic_add_l0c2gm(gm_tensor: tla.Tensor) -> None:
    """Test atomic add mode, data path from L0C to GM"""

    _m, _n = 32, 32
    l0c_ptr = tla.allocate(_m * _n, tla.Float32, tla.AddressSpace.l0c, 1024)
    l0c_tensor = tla.make_tensor_like(l0c_ptr, gm_tensor, tla.arch.L0Clayout)
    l0c_tile = tla.tile_view(l0c_tensor, tla.make_shape(_m, _n), tla.make_coord(0, 0))
    gm_tile = tla.tile_view(gm_tensor, tla.make_shape(_m, _n), tla.make_coord(0, 0))
    with tla.cube():
        tla.copy(
            gm_tile,
            l0c_tile,
            tla.params.CopyL0C2DstParams(
                unit_flag=0b11, atomic_mode=tla.params.AtomicMode.ADD
            ),
        )


@tla.kernel
def atomic_none_ub2gm(ub_tensor: tla.Tensor, gm_tensor: tla.Tensor) -> None:
    """Test common data copy, without atomic operation"""

    _vec_len = 64
    x_ub = tla.tile_view(ub_tensor, tla.make_shape(_vec_len), tla.make_coord(0))
    x_gm = tla.tile_view(gm_tensor, tla.make_shape(_vec_len), tla.make_coord(0))
    with tla.vector():
        tla.copy(x_gm, x_ub)


@tla.kernel
def gm2ub_atomic_false(gm_tensor: tla.Tensor, ub_tensor: tla.Tensor) -> None:
    _vec_len = 64
    x_gm = tla.tile_view(gm_tensor, tla.make_shape(_vec_len), tla.make_coord(0))
    x_ub = tla.tile_view(ub_tensor, tla.make_shape(_vec_len), tla.make_coord(0))
    with tla.vector():
        tla.copy(
            x_ub,
            x_gm,
            tla.params.CopyL0C2DstParams(atomic_mode=tla.params.AtomicMode.ADD),
        )


def test_atomic_add_ub2gm(compiler_tlair: Any) -> None:
    mlir = compiler_tlair(atomic_add_ub2gm, type_args=(_ub_tensor(), _gm_tensor()))

    assert "tla.copy" in mlir
    assert "atomic_mode = #tla.atomic_mode<add>" in mlir


def test_atomic_add_ub2gm_direct_arg(compiler_tlair: Any) -> None:
    """Kernel-arg GM dst must work without identity tile_view (_ArgProxy has no .dtype)."""
    mlir = compiler_tlair(
        atomic_add_ub2gm_direct_arg, type_args=(_ub_tensor(), _gm_tensor())
    )

    assert "tla.copy" in mlir
    assert "atomic_mode = #tla.atomic_mode<add>" in mlir


def test_atomic_add_l0c2gm(compiler_tlair: Any) -> None:
    mlir = compiler_tlair(
        atomic_add_l0c2gm,
        type_args=(_gm_tensor(dtype=tla.Float32, shape=(32, 32)),),
    )

    assert "tla.copy" in mlir
    assert "atomic_mode = #tla.atomic_mode<add>" in mlir


def test_atomic_none_ub2gm(compiler_tlair: Any) -> None:
    mlir = compiler_tlair(atomic_none_ub2gm, type_args=(_ub_tensor(), _gm_tensor()))

    assert "tla.copy" in mlir
    assert "atomic_mode" not in mlir


def test_atomic_add_gm2ub_rejected() -> None:
    """Atomic add with dst=UB should fail because atomic add requires dst=GM."""

    with pytest.raises(TlaLoweringError, match="the dst location should only be GM"):
        gm2ub_atomic_false.dump_mlir(type_args=(_gm_tensor(), _ub_tensor()))


def test_atomic_illega_f64_dtype() -> None:
    with pytest.raises((TlaLoweringError, TlaCoreAPIError)):
        atomic_add_ub2gm.dump_mlir(
            type_args=(_ub_tensor(dtype=tla.Int64), _gm_tensor(dtype=tla.Int64))
        )
