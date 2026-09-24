# tests/test_numeric_arith.py — Numeric arithmetic operators

from __future__ import annotations

from catlass.tla.runtime import make_fake_tensor


import operator

import pytest

from catlass._mlir import ir as mlir_ir
import catlass.tla as tla
from catlass.params import ScalarRoundMode as _SRM
import catlass.runtime as runtime_mod


def _gm_tensor_1d(length: int, *, dtype: type = tla.Int32) -> tla.Tensor:
    return make_fake_tensor(
               dtype,
               (length,),
               (1,),
               origin_shape=(length,),
               layout_tag=tla.arch.RowMajor,
           )


@tla.kernel
def _kernel_tiling_floordiv(tiling_int: tla.Tensor, out: tla.Tensor) -> None:
    embed = tiling_int[0]
    stages = embed // 128
    out[0] = stages


@tla.kernel
def _kernel_numeric_arith_i32(tiling_int: tla.Tensor, out: tla.Tensor) -> None:
    embed = tiling_int[0]
    a = embed + 1
    b = embed - 2
    c = embed * 3
    d = embed // 64
    e = embed % 64
    f = 256 // embed
    g = -embed
    out[0] = a
    out[1] = b
    out[2] = c
    out[3] = d
    out[4] = e
    out[5] = f
    out[6] = g


@tla.kernel
def _kernel_numeric_arith_f32(meta: tla.Tensor, out: tla.Tensor) -> None:
    x = meta[0]
    y = x + 1.5
    z = x * 2.0
    w = x / 4.0
    out[0] = y
    out[1] = z
    out[2] = w


@tla.kernel
def _kernel_numeric_binary(meta: tla.Tensor, out: tla.Tensor) -> None:
    a = meta[0]
    b = meta[1]
    out[0] = a + b
    out[1] = a * b
    out[2] = a // b


@tla.kernel
def _kernel_float_floordiv_rejected(meta: tla.Tensor, out: tla.Tensor) -> None:
    x = meta[0]
    out[0] = x // 2.0


@tla.kernel
def _kernel_numeric_bitwise(meta: tla.Tensor, out: tla.Tensor) -> None:
    x = meta[0]
    out[0] = x & 7
    out[1] = x | 1
    out[2] = x ^ 3
    out[3] = x << 2
    out[4] = x >> 1
    out[5] = ~x


@tla.kernel
def _kernel_numeric_int_pow_rejected(meta: tla.Tensor, out: tla.Tensor) -> None:
    x = meta[0]
    out[0] = x**2


@tla.kernel
def _kernel_numeric_float_pow(meta: tla.Tensor, out: tla.Tensor) -> None:
    x = meta[0]
    out[0] = x**2.0


def test_tiling_floordiv_returns_int32_and_divsi() -> None:
    tiling = _gm_tensor_1d(8, dtype=tla.Int32)
    out = _gm_tensor_1d(8, dtype=tla.Int32)
    mlir = _kernel_tiling_floordiv.dump_mlir(type_args=(tiling, out))
    assert "tla.scalar_load" in mlir
    assert "arith.divsi" in mlir


def test_numeric_i32_arith_ops() -> None:
    tiling = _gm_tensor_1d(8, dtype=tla.Int32)
    out = _gm_tensor_1d(8, dtype=tla.Int32)
    mlir = _kernel_numeric_arith_i32.dump_mlir(type_args=(tiling, out))
    assert "arith.addi" in mlir
    assert "arith.subi" in mlir
    assert "arith.muli" in mlir
    assert "arith.divsi" in mlir
    assert "arith.remsi" in mlir


def test_numeric_f32_arith_ops() -> None:
    meta = _gm_tensor_1d(8, dtype=tla.Float32)
    out = _gm_tensor_1d(8, dtype=tla.Float32)
    mlir = _kernel_numeric_arith_f32.dump_mlir(type_args=(meta, out))
    assert "arith.addf" in mlir
    assert "arith.mulf" in mlir
    assert "arith.divf" in mlir


def test_numeric_binary_with_numeric() -> None:
    meta = _gm_tensor_1d(8, dtype=tla.Int32)
    out = _gm_tensor_1d(8, dtype=tla.Int32)
    mlir = _kernel_numeric_binary.dump_mlir(type_args=(meta, out))
    assert "arith.addi" in mlir
    assert "arith.muli" in mlir
    assert "arith.divsi" in mlir


def test_numeric_float_floordiv_rejected() -> None:
    meta = _gm_tensor_1d(8, dtype=tla.Float32)
    out = _gm_tensor_1d(8, dtype=tla.Float32)
    with pytest.raises(Exception, match="only supported for integer"):
        _kernel_float_floordiv_rejected.dump_mlir(type_args=(meta, out))


@tla.kernel
def _kernel_numeric_i32_truediv_rejected(meta: tla.Tensor, out: tla.Tensor) -> None:
    a = meta[0]
    out[0] = a / 2


def test_numeric_i32_truediv_rejected() -> None:
    """Integer ``/`` is rejected; use ``//`` or float operands."""
    meta = _gm_tensor_1d(8, dtype=tla.Int32)
    out = _gm_tensor_1d(8, dtype=tla.Int32)
    with pytest.raises(Exception, match=r"Numeric '/' is only supported for float"):
        _kernel_numeric_i32_truediv_rejected.dump_mlir(type_args=(meta, out))


def test_as_numeric_and_host_constructors() -> None:
    assert isinstance(tla.Int32(7), tla.Int32)
    assert isinstance(tla.Float32(1.5), tla.Float32)
    assert isinstance(tla.as_numeric(3), tla.Int32)
    assert isinstance(tla.as_numeric(1.25), tla.Float32)
    numeric = tla.Int32(3)
    assert tla.as_numeric(numeric) is numeric
    assert tla.Int32(2) + tla.Int32(3)  # host python path
    assert int(tla.Int32(2) + tla.Int32(3)) == 5


def test_numeric_generated_methods_preserve_python_contract() -> None:
    assert tla.Numeric.__add__.__name__ == "__add__"
    assert tla.Numeric.__add__.__qualname__ == "Numeric.__add__"
    assert tla.Integer.__rand__.__name__ == "__rand__"
    assert tla.Numeric.__hash__ is None
    with pytest.raises(TypeError, match="unhashable type"):
        hash(tla.Int32(1))

    assert int(2 + tla.Int32(3)) == 5
    assert int(7 - tla.Int32(2)) == 5
    assert int(3 * tla.Int32(4)) == 12
    assert int(8 // tla.Int32(2)) == 4
    assert int(8 % tla.Int32(3)) == 2
    assert int(1 << tla.Int32(3)) == 8
    assert int(16 >> tla.Int32(2)) == 4
    assert int(0b1100 & tla.Int32(0b1010)) == 0b1000
    assert int(0b1100 | tla.Int32(0b0011)) == 0b1111
    assert int(0b1100 ^ tla.Int32(0b1010)) == 0b0110
    assert (2.0 ** tla.Float32(3.0)).value == 8.0


def test_numeric_same_type_required_and_div_rules() -> None:
    """Same-type only; integer ``/`` rejected; float ``/`` and int ``//`` ok."""
    with pytest.raises(TypeError, match=r"Numeric '/' is only supported for float"):
        _ = tla.Int32(4) / tla.Int32(2)
    with pytest.raises(TypeError, match="same type"):
        _ = tla.Int32(1) + tla.Float32(1.0)
    assert isinstance(tla.Float32(4.0) / tla.Float32(2.0), tla.Float32)
    assert (tla.Float32(4.0) / tla.Float32(2.0)).value == 2.0
    assert int(tla.Int32(4) // tla.Int32(2)) == 2


def test_mixed_numeric_comparison_requires_explicit_cast() -> None:
    """Rich comparisons reject mixed Numeric types (including ``==`` / ``!=``)."""
    left = tla.Int32(1)
    right = tla.Float32(1.0)
    for op in (
        operator.eq,
        operator.ne,
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,
    ):
        with pytest.raises(TypeError, match="same type"):
            _ = op(left, right)


def test_numeric_host_bitwise_and_float_pow() -> None:
    a = tla.Int32(0b1100)
    b = tla.Int32(0b1010)
    assert int(a & b) == 0b1000
    assert int(a | b) == 0b1110
    assert int(a ^ b) == 0b0110
    assert int(a << 1) == 0b11000
    assert int(a >> 2) == 0b11
    assert int(~tla.Int32(0)) == -1
    with pytest.raises(TypeError, match=r"Numeric '\*\*' is only supported for float"):
        _ = tla.Int32(3) ** 2
    assert (tla.Float32(3.0) ** tla.Float32(2.0)).value == 9.0
    assert bool(tla.Int32(1)) is True
    assert bool(tla.Int32(0)) is False


def test_numeric_bitwise_mlir() -> None:
    meta = _gm_tensor_1d(8, dtype=tla.Int32)
    out = _gm_tensor_1d(8, dtype=tla.Int32)
    mlir = _kernel_numeric_bitwise.dump_mlir(type_args=(meta, out))
    assert "arith.andi" in mlir
    assert "arith.ori" in mlir
    assert "arith.xori" in mlir
    assert "arith.shli" in mlir
    assert "arith.shrsi" in mlir
    assert "math.ipowi" not in mlir


def test_numeric_int_pow_rejected() -> None:
    meta = _gm_tensor_1d(8, dtype=tla.Int32)
    out = _gm_tensor_1d(8, dtype=tla.Int32)
    with pytest.raises(Exception, match=r"Numeric '\*\*' is only supported for float"):
        _kernel_numeric_int_pow_rejected.dump_mlir(type_args=(meta, out))


def test_numeric_float_pow_mlir() -> None:
    meta = _gm_tensor_1d(8, dtype=tla.Float32)
    out = _gm_tensor_1d(8, dtype=tla.Float32)
    mlir = _kernel_numeric_float_pow.dump_mlir(type_args=(meta, out))
    assert "math.powf" in mlir
    assert "math.ipowi" not in mlir


def test_cast_host_and_ssa_sitofp() -> None:
    """``cast`` / ``.to`` construct the target type; SSA cross-type emits arith ops."""

    # Host: no MLIR context required
    assert isinstance(tla.cast(5, tla.Float32), tla.Float32)
    host_f = tla.cast(tla.Int32(5), tla.Float32)
    assert isinstance(host_f, tla.Float32)
    assert host_f.value == 5.0
    assert tla.Int32(5).to(tla.Float32).value == 5.0

    ctx = mlir_ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx, mlir_ir.Location.unknown(), mlir_ir.InsertionPoint(
        mlir_ir.Module.create().body
    ):
        with runtime_mod._frontend_emission(module=None):
            i32_ty = mlir_ir.IntegerType.get_signless(32)
            i32_val = mlir_ir.Operation.create(
                "arith.constant",
                results=[i32_ty],
                attributes={"value": mlir_ir.IntegerAttr.get(i32_ty, 5)},
            ).results[0]
            result = tla.cast(tla.Int32(i32_val), tla.Float32)
            ir_v = result.ir_value()
            assert isinstance(result, tla.Float32)
            assert str(ir_v.type) == "f32"
            assert ir_v.owner.name == "arith.sitofp"


def test_numeric_add_captures_user_loc() -> None:
    """Numeric arithmetic attaches caller source location (via _capture_user_loc)."""

    ctx = mlir_ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx, mlir_ir.Location.unknown(), mlir_ir.InsertionPoint(
        mlir_ir.Module.create().body
    ):
        with runtime_mod._frontend_emission(module=None):
            i32_ty = mlir_ir.IntegerType.get_signless(32)
            i32_val = mlir_ir.Operation.create(
                "arith.constant",
                results=[i32_ty],
                attributes={"value": mlir_ir.IntegerAttr.get(i32_ty, 3)},
            ).results[0]
            result = tla.Int32(i32_val) + 1
            loc_str = str(result.ir_value().owner.location)
            assert "test_numeric_arith.py" in loc_str
            assert "unknown" not in loc_str.lower()


def test_bool_host_to_signed_and_unsigned() -> None:
    """``Bool`` converts to signed and unsigned integers as 0/1 (host path)."""
    assert tla.Bool(True).to(tla.Int8).value == 1
    assert tla.Bool(False).to(tla.Int8).value == 0
    assert tla.Bool(True).to(tla.Int16).value == 1
    assert tla.Bool(True).to(tla.Int32).value == 1
    assert tla.Bool(True).to(tla.UInt8).value == 1
    assert tla.Bool(True).to(tla.UInt16).value == 1
    assert tla.Bool(False).to(tla.UInt8).value == 0


def test_bool_ssa_to_signed_and_unsigned_zero_extends() -> None:
    """SSA ``Bool`` widens with ``arith.extui``: True == 1 for any wider int."""

    ctx = mlir_ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx, mlir_ir.Location.unknown(), mlir_ir.InsertionPoint(
        mlir_ir.Module.create().body
    ):
        with runtime_mod._frontend_emission(module=None):
            i1 = mlir_ir.IntegerType.get_signless(1)
            one = mlir_ir.Operation.create(
                "arith.constant",
                results=[i1],
                attributes={"value": mlir_ir.IntegerAttr.get(i1, 1)},
            ).results[0]
            b = tla.Bool(one)
            for target in (tla.Int8, tla.Int16, tla.Int32, tla.UInt8, tla.UInt16):
                result = b.to(target)
                assert isinstance(result, target)
                assert result.value.owner.name == "arith.extui", (
                    f"Bool.to({target.__name__}) must zero-extend (arith.extui)"
                )


def test_bool_ssa_to_float_uitofp() -> None:

    ctx = mlir_ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx, mlir_ir.Location.unknown(), mlir_ir.InsertionPoint(
        mlir_ir.Module.create().body
    ):
        with runtime_mod._frontend_emission(module=None):
            i1 = mlir_ir.IntegerType.get_signless(1)
            one = mlir_ir.Operation.create(
                "arith.constant",
                results=[i1],
                attributes={"value": mlir_ir.IntegerAttr.get(i1, 1)},
            ).results[0]
            result = tla.Bool(one).to(tla.Float32)
            assert isinstance(result, tla.Float32)
            assert result.value.owner.name == "arith.uitofp"


@tla.kernel
def _kernel_numeric_sqrt(meta: tla.Tensor, out: tla.Tensor) -> None:
    x = meta[0]
    out[0] = tla.sqrt(x)


@tla.kernel
def _kernel_numeric_sqrt_method(meta: tla.Tensor, out: tla.Tensor) -> None:
    x = meta[0]
    out[0] = x.sqrt()


@tla.kernel
def _kernel_numeric_sqrt_in_cube(meta: tla.Tensor, out: tla.Tensor) -> None:
    with tla.cube():
        out[0] = tla.sqrt(meta[0])


@tla.kernel
def _kernel_numeric_sqrt_in_simt(meta: tla.Tensor, out: tla.Tensor) -> None:
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=64):
            out[0] = tla.sqrt(meta[0])


@tla.kernel
def _kernel_numeric_sqrt_int_rejected(meta: tla.Tensor, out: tla.Tensor) -> None:
    out[0] = tla.sqrt(meta[0])


def test_numeric_sqrt_outside_simt_emits_math_sqrt() -> None:
    """Scalar sqrt outside a SIMT region is math.sqrt on the core scalar unit."""
    meta = _gm_tensor_1d(8, dtype=tla.Float32)
    out = _gm_tensor_1d(8, dtype=tla.Float32)
    mlir = _kernel_numeric_sqrt.dump_mlir(type_args=(meta, out))
    assert "math.sqrt" in mlir
    assert "tla.simt_sqrt" not in mlir


def test_numeric_sqrt_method_matches_free_function() -> None:
    meta = _gm_tensor_1d(8, dtype=tla.Float32)
    out = _gm_tensor_1d(8, dtype=tla.Float32)
    mlir = _kernel_numeric_sqrt_method.dump_mlir(type_args=(meta, out))
    assert "math.sqrt" in mlir


def test_numeric_sqrt_in_cube_region_emits_math_sqrt() -> None:
    meta = _gm_tensor_1d(8, dtype=tla.Float32)
    out = _gm_tensor_1d(8, dtype=tla.Float32)
    mlir = _kernel_numeric_sqrt_in_cube.dump_mlir(type_args=(meta, out))
    assert "math.sqrt" in mlir


def test_numeric_sqrt_in_simt_region_keeps_simt_op() -> None:
    """The SIMT form is untouched: a per-thread scalar still gets tla.simt_sqrt."""
    meta = _gm_tensor_1d(8, dtype=tla.Float32)
    out = _gm_tensor_1d(8, dtype=tla.Float32)
    mlir = _kernel_numeric_sqrt_in_simt.dump_mlir(type_args=(meta, out))
    assert "tla.simt_sqrt" in mlir


def test_numeric_sqrt_rejects_integer() -> None:
    meta = _gm_tensor_1d(8, dtype=tla.Int32)
    out = _gm_tensor_1d(8, dtype=tla.Int32)
    with pytest.raises(Exception, match="float-only"):
        _kernel_numeric_sqrt_int_rejected.dump_mlir(type_args=(meta, out))


def test_numeric_sqrt_constant_folds() -> None:
    assert abs(float(tla.Float32(6.25).sqrt().value) - 2.5) < 1e-6


@tla.kernel
def _kernel_round_cast_floor(meta: tla.Tensor, out: tla.Tensor) -> None:
    out[0] = meta[0].to(tla.Int32, round_mode=_SRM.FLOOR)


@tla.kernel
def _kernel_round_cast_trunc(meta: tla.Tensor, out: tla.Tensor) -> None:
    out[0] = meta[0].to(tla.Int32, round_mode=_SRM.TRUNC)


def test_round_cast_emits_tla_scalar_round_cast() -> None:
    meta = _gm_tensor_1d(8, dtype=tla.Float32)
    out = _gm_tensor_1d(8, dtype=tla.Int32)
    mlir = _kernel_round_cast_floor.dump_mlir(type_args=(meta, out))
    assert "tla.scalar_round_cast" in mlir
    assert '"rd"' in mlir or "rd " in mlir


def test_round_cast_trunc_stays_on_fptosi() -> None:
    """CAST_TRUNC needs no helper: arith.fptosi already truncates toward zero."""
    meta = _gm_tensor_1d(8, dtype=tla.Float32)
    out = _gm_tensor_1d(8, dtype=tla.Int32)
    mlir = _kernel_round_cast_trunc.dump_mlir(type_args=(meta, out))
    assert "arith.fptosi" in mlir
    assert "tla.scalar_round_cast" not in mlir


def test_round_cast_rejects_non_f32_to_i32() -> None:
    with pytest.raises(TypeError, match="only for Float32 -> Int32"):
        tla.Float32(1.5).to(tla.Int64, round_mode=_SRM.FLOOR)


def test_round_cast_rejects_float_target() -> None:
    with pytest.raises(TypeError, match="float -> integer"):
        tla.Float32(1.5).to(tla.Float16, round_mode=_SRM.FLOOR)


def test_round_cast_rejects_non_enum() -> None:
    with pytest.raises(TypeError, match="must be a tla.params.ScalarRoundMode"):
        tla.Float32(1.5).to(tla.Int32, round_mode="floor")


def test_round_cast_rejects_vector_round_mode() -> None:
    """Passing the AVE cast's RoundMode here must name the right enum."""
    from catlass.params import RoundMode

    with pytest.raises(TypeError, match="configures the AVE vector cast"):
        tla.Float32(1.5).to(tla.Int32, round_mode=RoundMode.CAST_FLOOR)


@pytest.mark.parametrize(
    "mode,value,expected",
    [
        (_SRM.NEAREST_EVEN, 2.5, 2),  # ties to even
        (_SRM.NEAREST_AWAY, 2.5, 3),  # ties away from zero
        (_SRM.FLOOR, -2.3, -3),
        (_SRM.CEIL, -2.3, -2),
        (_SRM.TRUNC, -2.3, -2),
    ],
)
def test_round_cast_constant_folds_per_mode(mode, value, expected) -> None:
    """A literal must fold the way the instruction would round it."""
    assert int(tla.Float32(value).to(tla.Int32, round_mode=mode).value) == expected


@tla.kernel
def _kernel_sqrt_f16_scalar_rejected(meta: tla.Tensor, out: tla.Tensor) -> None:
    out[0] = tla.sqrt(meta[0])


@tla.kernel
def _kernel_sqrt_f16_scalar_method_rejected(meta: tla.Tensor, out: tla.Tensor) -> None:
    out[0] = meta[0].sqrt()


def test_scalar_sqrt_rejects_f16_outside_simt() -> None:
    """math.sqrt selects only for f32; f16 would be 'Cannot select' in the backend."""
    meta = _gm_tensor_1d(8, dtype=tla.Float16)
    out = _gm_tensor_1d(8, dtype=tla.Float16)
    with pytest.raises(Exception, match="requires Float32"):
        _kernel_sqrt_f16_scalar_rejected.dump_mlir(type_args=(meta, out))


def test_scalar_sqrt_method_rejects_f16_outside_simt() -> None:
    meta = _gm_tensor_1d(8, dtype=tla.Float16)
    out = _gm_tensor_1d(8, dtype=tla.Float16)
    with pytest.raises(Exception, match="requires Float32"):
        _kernel_sqrt_f16_scalar_method_rejected.dump_mlir(type_args=(meta, out))


@tla.kernel
def _kernel_narrow_odd(meta: tla.Tensor, out: tla.Tensor) -> None:
    out[0] = meta[0].to(tla.Float16, round_mode=_SRM.ODD)


@tla.kernel
def _kernel_narrow_default(meta: tla.Tensor, out: tla.Tensor) -> None:
    out[0] = meta[0].to(tla.Float16)


def test_narrow_odd_emits_scalar_round_cast() -> None:
    meta = _gm_tensor_1d(8, dtype=tla.Float32)
    out = _gm_tensor_1d(8, dtype=tla.Float16)
    mlir = _kernel_narrow_odd.dump_mlir(type_args=(meta, out))
    assert "tla.scalar_round_cast" in mlir
    assert "arith.truncf" not in mlir


def test_narrow_default_stays_on_truncf() -> None:
    """Plain .to(Float16) keeps round-to-nearest-even; ODD must be opt-in."""
    meta = _gm_tensor_1d(8, dtype=tla.Float32)
    out = _gm_tensor_1d(8, dtype=tla.Float16)
    mlir = _kernel_narrow_default.dump_mlir(type_args=(meta, out))
    assert "arith.truncf" in mlir
    assert "tla.scalar_round_cast" not in mlir


def test_odd_rejects_integer_target() -> None:
    with pytest.raises(TypeError, match="only to Float32 -> Float16"):
        tla.Float32(1.5).to(tla.Int32, round_mode=_SRM.ODD)


def test_odd_rejects_f32_target() -> None:
    with pytest.raises(TypeError, match="only to Float32 -> Float16"):
        tla.Float32(1.5).to(tla.Float32, round_mode=_SRM.ODD)


def test_int_modes_reject_float_target() -> None:
    with pytest.raises(TypeError, match="ODD is the one"):
        tla.Float32(1.5).to(tla.Float16, round_mode=_SRM.FLOOR)


@pytest.mark.parametrize(
    "value,expected",
    [
        (2.0, 2.0),  # exactly representable: unchanged, even mantissa is fine
        (1.0 + 2.0**-11, 1.0009765625),  # tie: rne gives 1.0, odd goes up
        (-0.1, -0.10003662109375),
    ],
)
def test_odd_constant_folds(value, expected) -> None:
    """A literal must fold the way conv_f322f16o rounds it."""
    folded = float(tla.Float32(value).to(tla.Float16, round_mode=_SRM.ODD).value)
    assert folded == expected
