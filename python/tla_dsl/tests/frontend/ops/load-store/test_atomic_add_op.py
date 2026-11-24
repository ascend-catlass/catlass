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
def atomic_ub2gm(
    ub_tensor: tla.Tensor,
    gm_tensor: tla.Tensor,
    mode: tla.Constexpr[tla.params.AtomicMode],
) -> None:
    _vec_len = 64
    x_ub = tla.tile_view(ub_tensor, tla.make_shape(_vec_len), tla.make_coord(0))
    x_gm = tla.tile_view(gm_tensor, tla.make_shape(_vec_len), tla.make_coord(0))
    with tla.vector():
        tla.copy(
            x_gm,
            x_ub,
            tla.params.CopyUbToGmParams(atomic_mode=mode),
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
def atomic_l0c2gm(
    gm_tensor: tla.Tensor, mode: tla.Constexpr[tla.params.AtomicMode]
) -> None:
    _m, _n = 32, 32
    l0c_ptr = tla.allocate(_m * _n, tla.Float32, tla.AddressSpace.l0c, 1024)
    l0c_tensor = tla.make_tensor_like(l0c_ptr, gm_tensor, tla.arch.L0Clayout)
    l0c_tile = tla.tile_view(l0c_tensor, tla.make_shape(_m, _n), tla.make_coord(0, 0))
    gm_tile = tla.tile_view(gm_tensor, tla.make_shape(_m, _n), tla.make_coord(0, 0))
    with tla.cube():
        tla.copy(
            gm_tile,
            l0c_tile,
            tla.params.CopyL0C2DstParams(unit_flag=0b11, atomic_mode=mode),
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
def ub2gm_with_python_none(ub: tla.Tensor, gm: tla.Tensor) -> None:
    with tla.vector():
        tla.copy(gm, ub, tla.params.CopyUbToGmParams(atomic_mode=None))


@tla.kernel
def ub2gm_with_fixpipe_params(ub: tla.Tensor, gm: tla.Tensor) -> None:
    with tla.vector():
        tla.copy(
            gm,
            ub,
            tla.params.CopyL0C2DstParams(
                atomic_mode=tla.params.AtomicMode.MAX, relu_enable=True
            ),
        )


@tla.kernel
def ub2gm_with_default_params(ub: tla.Tensor, gm: tla.Tensor) -> None:
    with tla.vector():
        tla.copy(gm, ub, tla.params.CopyUbToGmParams())


@tla.kernel
def gm2ub_atomic_false(
    gm_tensor: tla.Tensor,
    ub_tensor: tla.Tensor,
    mode: tla.Constexpr[tla.params.AtomicMode],
) -> None:
    _vec_len = 64
    x_gm = tla.tile_view(gm_tensor, tla.make_shape(_vec_len), tla.make_coord(0))
    x_ub = tla.tile_view(ub_tensor, tla.make_shape(_vec_len), tla.make_coord(0))
    with tla.vector():
        tla.copy(
            x_ub,
            x_gm,
            tla.params.CopyUbToGmParams(atomic_mode=mode),
        )


ATOMIC_MODES = (
    tla.params.AtomicMode.ADD,
    tla.params.AtomicMode.MAX,
    tla.params.AtomicMode.MIN,
)


@pytest.mark.parametrize("mode", ATOMIC_MODES)
@pytest.mark.parametrize(
    "dtype", (tla.Int8, tla.Int16, tla.Int32, tla.Float16, tla.BFloat16, tla.Float32)
)
def test_atomic_ub2gm(compiler_tlair: Any, mode, dtype) -> None:
    mlir = compiler_tlair(
        atomic_ub2gm, type_args=(_ub_tensor(dtype), _gm_tensor(dtype), mode)
    )
    assert f"atomic_mode = #tla.atomic_mode<{mode.value}>" in mlir


def test_atomic_add_ub2gm_direct_arg(compiler_tlair: Any) -> None:
    """Kernel-arg GM dst must work without identity tile_view (_ArgProxy has no .dtype)."""
    mlir = compiler_tlair(
        atomic_add_ub2gm_direct_arg, type_args=(_ub_tensor(), _gm_tensor())
    )

    assert "tla.copy" in mlir
    assert "atomic_mode = #tla.atomic_mode<add>" in mlir


@pytest.mark.parametrize("mode", ATOMIC_MODES)
@pytest.mark.parametrize("dtype", (tla.Float32, tla.Float16, tla.BFloat16))
def test_atomic_l0c2gm(compiler_tlair: Any, mode, dtype) -> None:
    mlir = compiler_tlair(
        atomic_l0c2gm,
        type_args=(_gm_tensor(dtype=dtype, shape=(32, 32)), mode),
    )

    assert "tla.copy" in mlir
    assert f"atomic_mode = #tla.atomic_mode<{mode.value}>" in mlir


@pytest.mark.parametrize("dtype", (tla.Int32, tla.UInt8))
def test_atomic_none_ub2gm(compiler_tlair: Any, dtype) -> None:
    mlir = compiler_tlair(
        atomic_none_ub2gm, type_args=(_ub_tensor(dtype), _gm_tensor(dtype))
    )

    assert "tla.copy" in mlir
    assert "atomic_mode" not in mlir


@pytest.mark.parametrize(
    ("kernel", "message"),
    [
        (ub2gm_with_python_none, r"CopyUbToGmParams\.atomic_mode must be one of"),
        (ub2gm_with_fixpipe_params, "expects CopyUbToGmParams, got CopyL0C2DstParams"),
    ],
)
def test_ub2gm_rejects_invalid_params(compiler_tlair, kernel, message):
    with pytest.raises((TlaLoweringError, TlaCoreAPIError), match=message):
        compiler_tlair(kernel, type_args=(_ub_tensor(), _gm_tensor()))


@pytest.mark.parametrize(
    ("kernel", "extra_args"),
    [(ub2gm_with_default_params, ()), (atomic_ub2gm, (tla.params.AtomicMode.NONE,))],
)
def test_ub2gm_params_without_atomic_mode(compiler_tlair, kernel, extra_args):
    mlir = compiler_tlair(kernel, type_args=(_ub_tensor(), _gm_tensor(), *extra_args))
    assert "tla.copy" in mlir
    assert "atomic_mode" not in mlir


@pytest.mark.parametrize("mode", ATOMIC_MODES)
def test_atomic_gm2ub_rejected(mode) -> None:
    with pytest.raises(TlaLoweringError, match="the dst location should only be GM"):
        gm2ub_atomic_false.dump_mlir(type_args=(_gm_tensor(), _ub_tensor(), mode))


@pytest.mark.parametrize("mode", ATOMIC_MODES)
@pytest.mark.parametrize("dtype", (tla.UInt8, tla.UInt16, tla.UInt32, tla.Int64))
def test_atomic_dst_dtype_rejected(mode, dtype) -> None:
    with pytest.raises(
        (TlaLoweringError, TlaCoreAPIError), match="atomic operation.*data type"
    ):
        atomic_ub2gm.dump_mlir(type_args=(_ub_tensor(dtype), _gm_tensor(dtype), mode))
