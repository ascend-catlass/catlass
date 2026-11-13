from __future__ import annotations

from catlass.tla.runtime import make_fake_tensor


from typing import Any

import pytest

import catlass.tla as tla
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


@tla.kernel
def arange_store(dst: tla.Tensor) -> None:
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst_tile.store(tla.arange(3, dtype=tla.Int32))


@tla.kernel
def arange_decrease_store(dst: tla.Tensor) -> None:
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst_tile.store(tla.arange(128 + 8, order="decrease", dtype=tla.Int32))
            # Note: for decreasing mode, the actual start value is ` start + VL - 1`


@tla.kernel
def arange_f16_is_rejected(dst: tla.Tensor) -> None:
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst_tile.store(tla.arange(dtype=tla.Float16))


@tla.kernel
def arange_f32_is_rejected(dst: tla.Tensor) -> None:
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst_tile.store(tla.arange(dtype=tla.Float32))


@tla.kernel
def arange_outside_vec_func(dst: tla.Tensor) -> None:
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            pass
        dst_tile.store(tla.arange(dtype=tla.Int32))


def test_arange_emits_tlair(compiler_tlair: Any) -> None:
    mlir = compiler_tlair(arange_store, type_args=(_ub_tensor(),))

    assert "tla.arange" in mlir
    assert 'order = "increase"' in mlir
    assert "tla.store" in mlir


def test_arange_decrease_emits_tlair(compiler_tlair: Any) -> None:
    mlir = compiler_tlair(arange_decrease_store, type_args=(_ub_tensor(),))

    assert "tla.arange" in mlir
    assert 'order = "decrease"' in mlir
    assert "tla.store" in mlir


@pytest.mark.parametrize(
    ("kernel", "dtype"),
    [
        (arange_f16_is_rejected, tla.Float16),
        (arange_f32_is_rejected, tla.Float32),
    ],
)
def test_arange_unsupported_dtype_is_rejected(
    kernel: Any, dtype: type[tla.Numeric]
) -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="unsupported vector element dtype"):
        kernel.dump_mlir(type_args=(_ub_tensor(dtype),))


def test_arange_outside_vec_func_is_rejected() -> None:
    with pytest.raises(
        tla.TlaCoreAPIError,
        match=r"tla\.arange must be nested inside tla\.vec\.func\(\)",
    ):
        arange_outside_vec_func.dump_mlir(type_args=(_ub_tensor(),))
