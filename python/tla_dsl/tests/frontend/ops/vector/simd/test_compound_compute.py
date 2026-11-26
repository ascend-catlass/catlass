import pytest

import catlass.tla as tla


def test_compound_ops_are_exported_under_arch():
    for name in ("axpy", "sub_abs", "muls_cast"):
        assert callable(getattr(tla.arch, name))
        assert not hasattr(tla, name)
    assert not hasattr(tla.arch, "abs_sub")
    assert not hasattr(tla, "abs_sub")


@tla.kernel
def _compound_pair(dtype: tla.Constexpr[type], alpha: tla.Constexpr[float | int]):
    with tla.vector():
        with tla.vec.func(mode="simd"):
            x = tla.full(2, dtype)
            y = tla.full(3, dtype)
            tla.arch.axpy(x, y, alpha)
            if tla.const_expr(dtype is not tla.UInt64):
                tla.arch.sub_abs(x, y)


@pytest.mark.parametrize(
    "dtype,element,scalar_type,lanes,alpha",
    [
        (tla.Float16, "f16", "f16", 128, 0.5),
        (tla.Float32, "f32", "f32", 64, 0.5),
        (tla.Int64, "i64", "i64", 32, 2),
        (tla.UInt64, "ui64", "i64", 32, (1 << 63) + 1),
    ],
)
def test_compound_pair_uses_same_element_type(
    dtype, element, scalar_type, lanes, alpha
):
    mlir = _compound_pair.dump_mlir(type_args=(dtype, alpha))
    vector = f"!tla.vector<{lanes}x{element}>"
    mask = f"!tla.mask<{lanes}>"
    axpy = next(line for line in mlir.splitlines() if "tla.axpy " in line)
    assert f": {vector}, {vector}, {scalar_type} mask {mask} -> {vector}" in axpy
    if dtype is not tla.UInt64:
        sub_abs = next(line for line in mlir.splitlines() if "tla.sub_abs " in line)
        assert f": {vector}, {vector} mask {mask} -> {vector}" in sub_abs


@tla.kernel
def _muls_cast_slots(scalar: tla.Float32):
    with tla.vector():
        with tla.vec.func(mode="simd"):
            src = tla.full(2, tla.Float32)
            tla.arch.muls_cast(src, scalar)
            mask = tla.create_mask(pattern=tla.mask.VL16, dtype=tla.Float32)
            tla.arch.muls_cast(src, 0.5, reg_slot=tla.params.RegSlot.ONE, mask=mask)


def test_muls_cast_uses_input_mask_domain_and_full_f16_result():
    mlir = _muls_cast_slots.dump_mlir(type_args=(tla.Float32(0.5),))
    lines = [line for line in mlir.splitlines() if "tla.muls_cast " in line]
    assert len(lines) == 2
    for slot, line in enumerate(lines):
        assert f"part {slot}" in line
        assert "!tla.vector<64xf32> mask !tla.mask<64> -> !tla.vector<128xf16>" in line


@tla.kernel
def _invalid_compound(operation: tla.Constexpr[str]):
    with tla.vector():
        with tla.vec.func(mode="simd"):
            f32 = tla.full(1, tla.Float32)
            f16 = tla.full(1, tla.Float16)
            if tla.const_expr(operation == "axpy"):
                tla.arch.axpy(f32, f32, tla.Float16(1.0))
            elif tla.const_expr(operation == "axpy_bool"):
                i64 = tla.full(1, tla.Int64)
                tla.arch.axpy(i64, i64, True)
            elif tla.const_expr(operation == "sub_abs"):
                tla.arch.sub_abs(f32, f16)
            elif tla.const_expr(operation == "muls_cast_input"):
                tla.arch.muls_cast(f16, 1.0)
            elif tla.const_expr(operation == "muls_cast_scalar"):
                tla.arch.muls_cast(f32, tla.Float16(1.0))
            elif tla.const_expr(operation == "muls_cast_slot"):
                tla.arch.muls_cast(f32, 1.0, reg_slot=tla.params.RegSlot.TWO)
            else:
                mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float16)
                tla.arch.muls_cast(f32, 1.0, mask=mask)


@pytest.mark.parametrize(
    "operation,error",
    [
        ("axpy", "scalar must match the input element type f32"),
        ("axpy_bool", "scalar must match the input element type i64"),
        ("sub_abs", "vector operands must have matching elements"),
        ("muls_cast_input", "src must have Float32 elements"),
        ("muls_cast_scalar", "scalar must match the input element type f32"),
        ("muls_cast_slot", "reg_slot must be RegSlot.ZERO or RegSlot.ONE"),
        ("muls_cast_mask", "mask has 128 predicate lanes, expected 64"),
    ],
)
def test_compound_ops_reject_invalid_operands(operation, error):
    with pytest.raises(tla.TlaCoreAPIError, match=error):
        _invalid_compound.dump_mlir(type_args=(operation,))
