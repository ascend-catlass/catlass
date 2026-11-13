from __future__ import annotations

from catlass.tla.runtime import make_fake_tensor


from collections.abc import Callable

import pytest

import catlass.tla as tla
from catlass.execution_lowering import UnsupportedExecutionLowering
import catlass.runtime as runtime_mod  # noqa: F401


def _vector_tensor(dtype: type[tla.Numeric] = tla.Float32) -> tla.Tensor:
    return make_fake_tensor(
        dtype,
        (64,),
        (1,),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(64,),
        layout_tag=tla.arch.RowMajor,
    )


@tla.kernel
def vector_float_unary_kernel(src: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            src_reg = src_tile.load()
            lane_mask = tla.create_mask(pattern=tla.mask.H)
            _ = tla.exp(src_reg, mask=lane_mask)
            _ = tla.log(src_reg)
            _ = tla.sqrt(src_reg)
            _ = tla.abs(src_reg)
            _ = tla.neg(src_reg, mask=lane_mask)


@tla.kernel
def vector_int_unary_kernel(src: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            src_reg = src_tile.load()
            _ = tla.abs(src_reg)
            _ = tla.neg(src_reg)


@tla.kernel
def vector_float16_unary_kernel(src: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            src_reg = src_tile.load()
            _ = tla.exp(src_reg)
            _ = tla.abs(src_reg)
            _ = tla.neg(src_reg)


@tla.kernel
def vector_int16_unary_kernel(src: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            src_reg = src_tile.load()
            _ = tla.abs(src_reg)
            _ = tla.neg(src_reg)


@tla.kernel
def unary_float_on_int_kernel(src: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            _ = tla.exp(src_tile.load())


@tla.kernel
def unary_bf16_exp_kernel(src: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            _ = tla.exp(src_tile.load())


@tla.kernel
def unary_bf16_abs_kernel(src: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            _ = tla.abs(src_tile.load())


@tla.kernel
def unary_bf16_neg_kernel(src: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            _ = tla.neg(src_tile.load())


def test_vector_unary_public_exports_exist() -> None:
    for name in ("exp", "log", "sqrt", "abs", "neg"):
        assert callable(getattr(tla, name))


def test_vector_unary_math_namespace_is_not_exposed() -> None:
    assert not hasattr(tla, "math")


def test_vector_unary_ops_emit_tla_ops() -> None:
    float_mlir = vector_float_unary_kernel.dump_mlir(type_args=(_vector_tensor(),))
    float16_mlir = vector_float16_unary_kernel.dump_mlir(
        type_args=(_vector_tensor(tla.Float16),)
    )
    int_mlir = vector_int_unary_kernel.dump_mlir(type_args=(_vector_tensor(tla.Int32),))
    int16_mlir = vector_int16_unary_kernel.dump_mlir(
        type_args=(_vector_tensor(tla.Int16),)
    )

    for op_name in ("exp", "log", "sqrt", "abs", "neg"):
        assert f"tla.{op_name}" in float_mlir
    assert "tla.exp" in float16_mlir
    assert "tla.abs" in float16_mlir
    assert "tla.neg" in float16_mlir
    assert "tla.abs" in int_mlir
    assert "tla.neg" in int_mlir
    assert "tla.abs" in int16_mlir
    assert "tla.neg" in int16_mlir


def test_vector_unary_mask_keyword_is_preserved() -> None:
    mlir = vector_float_unary_kernel.dump_mlir(type_args=(_vector_tensor(),))

    exp_line = next(line for line in mlir.splitlines() if "tla.exp" in line)
    neg_line = next(line for line in mlir.splitlines() if "tla.neg" in line)
    assert "mask" in exp_line
    assert "mask" in neg_line


def test_vector_unary_missing_operand_is_rejected() -> None:
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        tla.exp()


@pytest.mark.parametrize(
    "kernel, dtype, match",
    (
        (
            unary_float_on_int_kernel,
            tla.Int32,
            "tla.exp requires f16 or f32 element type",
        ),
        (
            unary_bf16_exp_kernel,
            tla.BFloat16,
            "tla.exp requires f16 or f32 element type",
        ),
        (
            unary_bf16_abs_kernel,
            tla.BFloat16,
            "requires f16/f32 or i8/i16/i32 element type",
        ),
        (
            unary_bf16_neg_kernel,
            tla.BFloat16,
            "requires f16/f32 or i8/i16/i32 element type",
        ),
    ),
)
def test_invalid_vector_unary_api_is_rejected(
    kernel: Callable[..., object],
    dtype: type[tla.Numeric],
    match: str,
) -> None:
    with pytest.raises(
        (TypeError, tla.TlaCoreAPIError, UnsupportedExecutionLowering), match=match
    ):
        kernel.dump_mlir(type_args=(_vector_tensor(dtype),))
