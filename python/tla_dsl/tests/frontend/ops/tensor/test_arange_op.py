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
def arange_f16_store(dst: tla.Tensor) -> None:
    # A 256-byte register holds 128 f16 lanes.
    dst_tile = tla.tile_view(dst, tla.make_shape(128), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst_tile.store(tla.arange(3, dtype=tla.Float16))


@tla.kernel
def arange_f32_store(dst: tla.Tensor) -> None:
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst_tile.store(tla.arange(8, order="decrease", dtype=tla.Float32))


@tla.kernel
def arange_f32_float_base_store(dst: tla.Tensor) -> None:
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst_tile.store(tla.arange(2.0, dtype=tla.Float32))


@tla.kernel
def arange_f32_numeric_int_base_store(dst: tla.Tensor) -> None:
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst_tile.store(tla.arange(tla.Float32(3), dtype=tla.Float32))


@tla.kernel
def arange_f32_fractional_base_store(dst: tla.Tensor) -> None:
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst_tile.store(tla.arange(0.5, dtype=tla.Float32))


@tla.kernel
def arange_f32_index_base_store(dst: tla.Tensor) -> None:
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            for i in tla.range(2):
                dst_tile.store(tla.arange(i * 64, dtype=tla.Float32))


@tla.kernel
def arange_bf16_is_rejected(dst: tla.Tensor) -> None:
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst_tile.store(tla.arange(dtype=tla.BFloat16))


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
    ("kernel", "dtype", "extent", "order", "base", "start_dtype"),
    [
        (arange_f16_store, tla.Float16, 128, "increase", 3, "i16"),
        (arange_f32_store, tla.Float32, 64, "decrease", 8, "i32"),
    ],
)
def test_arange_float_dtype_emits_tlair(
    compiler_tlair: Any,
    kernel: Any,
    dtype: type[tla.Numeric],
    extent: int,
    order: str,
    base: int,
    start_dtype: str,
) -> None:
    mlir = compiler_tlair(kernel, type_args=(_ub_tensor(dtype, extent),))

    assert "tla.arange" in mlir
    assert f'order = "{order}"' in mlir
    assert f"x{dtype.dtype}" in mlir
    # The start operand keeps the integer width of the float element type.
    assert f"value = {base} : {start_dtype}" in mlir


def test_arange_float_base_literal_emits_tlair(compiler_tlair: Any) -> None:
    mlir = compiler_tlair(
        arange_f32_float_base_store, type_args=(_ub_tensor(tla.Float32),)
    )

    assert "tla.arange" in mlir
    assert "arith.constant" in mlir
    assert "f32" in mlir


def test_arange_float_numeric_int_base_emits_i32_start(compiler_tlair: Any) -> None:
    mlir = compiler_tlair(
        arange_f32_numeric_int_base_store, type_args=(_ub_tensor(tla.Float32),)
    )

    assert "tla.arange" in mlir
    assert "value = 3 : i32" in mlir


def test_arange_fractional_base_is_rejected() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="base must be an integer value"):
        arange_f32_fractional_base_store.dump_mlir(type_args=(_ub_tensor(tla.Float32),))


def test_arange_float_runtime_base_is_rejected() -> None:
    # A runtime base would be narrowed to i16/i32 without a range check.
    with pytest.raises(tla.TlaCoreAPIError, match="compile-time integer constant"):
        arange_f32_index_base_store.dump_mlir(type_args=(_ub_tensor(tla.Float32),))


def test_arange_bf16_is_rejected() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="unsupported vector element dtype"):
        arange_bf16_is_rejected.dump_mlir(type_args=(_ub_tensor(tla.BFloat16),))


def test_arange_outside_vec_func_is_rejected() -> None:
    with pytest.raises(
        tla.TlaCoreAPIError,
        match=r"tla\.arange must be nested inside tla\.vec\.func\(\)",
    ):
        arange_outside_vec_func.dump_mlir(type_args=(_ub_tensor(),))
