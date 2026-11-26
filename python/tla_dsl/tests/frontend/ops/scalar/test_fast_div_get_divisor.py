"""Frontend tests for ``tla.fast_div_get_divisor`` / ``FastDivDivisor``."""

from __future__ import annotations

import pytest

import catlass.tla as tla
from catlass.tla.runtime import make_fake_tensor

_INT32_MAX = (1 << 31) - 1
_INT64_MAX = (1 << 63) - 1


@tla.kernel
def _literal32_kernel(out: tla.Tensor) -> None:
    with tla.vector():
        fd = tla.fast_div_get_divisor(5)
        out[0] = fd.magic
        out[1] = fd.shift


def test_python_int_small_selects_uint32() -> None:
    out = make_fake_tensor(tla.UInt32, (4,), (1,))
    mlir = _literal32_kernel.dump_mlir(type_args=(out,))
    assert "tla.scalar_fast_div_divisor" in mlir
    # One i32 divisor, two i32 results.
    assert "5 : i32" in mlir
    assert ": i32 -> i32, i32" in mlir


@tla.kernel
def _literal64_kernel(out: tla.Tensor) -> None:
    with tla.vector():
        fd = tla.fast_div_get_divisor(_INT32_MAX + 1)
        out[0] = fd.magic
        out[1] = fd.shift


def test_python_int_above_int32_max_selects_uint64() -> None:
    out = make_fake_tensor(tla.UInt64, (4,), (1,))
    mlir = _literal64_kernel.dump_mlir(type_args=(out,))
    assert "tla.scalar_fast_div_divisor" in mlir
    assert f"{_INT32_MAX + 1} : i64" in mlir
    assert ": i64 -> i64, i64" in mlir


@tla.kernel
def _numeric_kernel(
    out32: tla.Tensor, out64: tla.Tensor, d32: tla.UInt32, d64: tla.UInt64
) -> None:
    with tla.vector():
        fd32 = tla.fast_div_get_divisor(d32)
        fd64 = tla.fast_div_get_divisor(d64)
        out32[0] = fd32.magic
        out32[1] = fd32.shift
        out64[0] = fd64.magic
        out64[1] = fd64.shift


def test_numeric_width_selects_variant() -> None:
    out32 = make_fake_tensor(tla.UInt32, (4,), (1,))
    out64 = make_fake_tensor(tla.UInt64, (4,), (1,))
    mlir = _numeric_kernel.dump_mlir(
        type_args=(out32, out64, tla.UInt32(0), tla.UInt64(0))
    )
    assert mlir.count("tla.scalar_fast_div_divisor") == 2
    assert ": i32 -> i32, i32" in mlir
    assert ": i64 -> i64, i64" in mlir


@tla.kernel
def _signed32_kernel(out: tla.Tensor, d: tla.Int32) -> None:
    with tla.vector():
        fd = tla.fast_div_get_divisor(d)
        out[0] = fd.magic


def test_signed32_numeric_converts_to_uint32_variant() -> None:
    out = make_fake_tensor(tla.UInt32, (4,), (1,))
    mlir = _signed32_kernel.dump_mlir(type_args=(out, tla.Int32(0)))
    assert "tla.scalar_fast_div_divisor" in mlir
    assert ": i32 -> i32, i32" in mlir


@tla.kernel
def _narrow_numeric_kernel(
    out32: tla.Tensor, out64: tla.Tensor, d16: tla.UInt16, di64: tla.Int64
) -> None:
    with tla.vector():
        a = tla.fast_div_get_divisor(d16)
        b = tla.fast_div_get_divisor(di64)
        out32[0] = a.magic
        out64[0] = b.magic


def test_narrow_and_wide_numerics_follow_width() -> None:
    out32 = make_fake_tensor(tla.UInt32, (4,), (1,))
    out64 = make_fake_tensor(tla.UInt64, (4,), (1,))
    mlir = _narrow_numeric_kernel.dump_mlir(
        type_args=(out32, out64, tla.UInt16(0), tla.Int64(0))
    )
    assert mlir.count("tla.scalar_fast_div_divisor") == 2
    assert ": i32 -> i32, i32" in mlir
    assert ": i64 -> i64, i64" in mlir


@tla.kernel
def _attributes_kernel(out: tla.Tensor) -> None:
    with tla.vector():
        fd = tla.fast_div_get_divisor(7)
        out[0] = fd.magic
        out[1] = fd.shift


def test_results_are_uint32_numerics() -> None:
    seen: dict[str, object] = {}

    def record(fd: tla.FastDivDivisor) -> None:
        seen["magic"] = fd.magic
        seen["shift"] = fd.shift
        seen["divisor"] = fd.divisor_mlir

    @tla.kernel
    def kernel(out: tla.Tensor) -> None:
        with tla.vector():
            record(tla.fast_div_get_divisor(7))
            out[0] = 0

    out = make_fake_tensor(tla.UInt32, (4,), (1,))
    kernel.dump_mlir(type_args=(out,))
    assert isinstance(seen["magic"], tla.UInt32)
    assert isinstance(seen["shift"], tla.UInt32)
    assert isinstance(seen["divisor"], tla.UInt32)
    assert seen["divisor"].value == 7


@tla.kernel
def _cube_kernel(out: tla.Tensor) -> None:
    with tla.cube():
        fd = tla.fast_div_get_divisor(3)
        out[0] = fd.magic


def test_allowed_in_a_cube_region() -> None:
    out = make_fake_tensor(tla.UInt32, (4,), (1,))
    assert "tla.scalar_fast_div_divisor" in _cube_kernel.dump_mlir(type_args=(out,))


@pytest.mark.parametrize("value", [0, -1, -5, _INT64_MAX + 1])
def test_rejects_out_of_range_python_int(value: int) -> None:
    def make(value: int):
        @tla.kernel
        def kernel(out: tla.Tensor) -> None:
            with tla.vector():
                fd = tla.fast_div_get_divisor(value)
                out[0] = fd.magic

        return kernel

    out = make_fake_tensor(tla.UInt64, (4,), (1,))
    with pytest.raises(tla.TlaCoreAPIError, match="divisor"):
        make(value).dump_mlir(type_args=(out,))


def test_rejects_uint32_numeric_above_int32_max() -> None:
    @tla.kernel
    def kernel(out: tla.Tensor) -> None:
        with tla.vector():
            fd = tla.fast_div_get_divisor(tla.UInt32(_INT32_MAX + 1))
            out[0] = fd.magic

    out = make_fake_tensor(tla.UInt32, (4,), (1,))
    with pytest.raises(tla.TlaCoreAPIError, match="divisor"):
        kernel.dump_mlir(type_args=(out,))


@pytest.mark.parametrize("bad", [True, 1.5, "5"])
def test_rejects_non_integer_divisor(bad: object) -> None:
    def make(bad: object):
        @tla.kernel
        def kernel(out: tla.Tensor) -> None:
            with tla.vector():
                fd = tla.fast_div_get_divisor(bad)
                out[0] = fd.magic

        return kernel

    out = make_fake_tensor(tla.UInt32, (4,), (1,))
    with pytest.raises((tla.TlaCoreAPIError, TypeError)):
        make(bad).dump_mlir(type_args=(out,))


@tla.kernel
def _outside_regions(out: tla.Tensor) -> None:
    fd = tla.fast_div_get_divisor(5)
    out[0] = fd.magic


def test_rejected_outside_core_regions() -> None:
    out = make_fake_tensor(tla.UInt32, (4,), (1,))
    with pytest.raises(
        tla.TlaCoreAPIError,
        match="must be nested inside tla.cube\\(\\) or tla.vector\\(\\)",
    ):
        _outside_regions.dump_mlir(type_args=(out,))


@tla.kernel
def _inside_simd(out: tla.Tensor) -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            fd = tla.fast_div_get_divisor(5)
            out[0] = fd.magic


def test_rejected_inside_vec_func() -> None:
    out = make_fake_tensor(tla.UInt32, (4,), (1,))
    with pytest.raises(tla.TlaCoreAPIError, match="outside tla.vec.func"):
        _inside_simd.dump_mlir(type_args=(out,))


def test_requires_a_kernel() -> None:
    with pytest.raises(tla.TlaIRNotExecutableError):
        tla.fast_div_get_divisor(5)


# --- bare SSA proxy (raw mlir.ir.Value, not a Numeric) ---


@tla.kernel
def _bare_ssa_kernel(out: tla.Tensor, d: tla.UInt32) -> None:
    with tla.vector():
        fd = tla.fast_div_get_divisor(d.ir_value())
        out[0] = fd.magic


def test_accepts_bare_ssa_divisor() -> None:
    out = make_fake_tensor(tla.UInt32, (4,), (1,))
    mlir = _bare_ssa_kernel.dump_mlir(type_args=(out, tla.UInt32(0)))
    assert "tla.scalar_fast_div_divisor" in mlir
    assert ": i32 -> i32, i32" in mlir


@tla.kernel
def _narrow_bare_ssa_kernel(out: tla.Tensor, d: tla.UInt16) -> None:
    with tla.vector():
        fd = tla.fast_div_get_divisor(d.ir_value())
        out[0] = fd.magic


def test_bare_ssa_divisor_widens_by_width() -> None:
    out = make_fake_tensor(tla.UInt32, (4,), (1,))
    mlir = _narrow_bare_ssa_kernel.dump_mlir(type_args=(out, tla.UInt16(0)))
    assert "tla.scalar_fast_div_divisor" in mlir
    assert ": i32 -> i32, i32" in mlir


@tla.kernel
def _signed_narrow_bare_ssa_kernel(out: tla.Tensor, d: tla.Int16) -> None:
    with tla.vector():
        fd = tla.fast_div_get_divisor(d.ir_value())
        out[0] = fd.magic


def test_bare_signless_i16_zero_extends() -> None:
    """A bare value has unknown signedness: its 16-bit pattern is zero-extended."""
    out = make_fake_tensor(tla.UInt32, (4,), (1,))
    mlir = _signed_narrow_bare_ssa_kernel.dump_mlir(type_args=(out, tla.Int16(0)))
    assert "tla.scalar_fast_div_divisor" in mlir
    assert "arith.extui" in mlir
    assert "arith.extsi" not in mlir


@tla.kernel
def _bool_bare_ssa_kernel(out: tla.Tensor, flag: tla.Bool) -> None:
    with tla.vector():
        fd = tla.fast_div_get_divisor(flag.ir_value())
        out[0] = fd.magic


def test_rejects_bool_bare_ssa() -> None:
    out = make_fake_tensor(tla.UInt32, (4,), (1,))
    with pytest.raises(tla.TlaCoreAPIError, match="i1/bool"):
        _bool_bare_ssa_kernel.dump_mlir(type_args=(out, tla.Bool(True)))


# --- no dynamic validation: divisor constraints are the caller's contract ---


@tla.kernel
def _runtime_divisor_kernel(out: tla.Tensor, d: tla.UInt32) -> None:
    with tla.vector():
        fd = tla.fast_div_get_divisor(d)
        out[0] = fd.magic


def test_runtime_divisor_emits_no_guard() -> None:
    out = make_fake_tensor(tla.UInt32, (4,), (1,))
    mlir = _runtime_divisor_kernel.dump_mlir(type_args=(out, tla.UInt32(0)))
    assert "tla.scalar_fast_div_divisor" in mlir
    assert "tla.trap" not in mlir
    assert "scf.if" not in mlir
