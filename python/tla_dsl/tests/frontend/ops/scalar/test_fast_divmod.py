"""Frontend tests for SIMD fast division/remainder via ``FastDivDivisor``."""

from __future__ import annotations

import pytest

import catlass.tla as tla
from catlass.tla.runtime import make_fake_tensor

_INT32_MAX = (1 << 31) - 1
_INT64_MAX = (1 << 63) - 1


@tla.kernel
def _runtime32_kernel(
    out_q: tla.Tensor, out_r: tla.Tensor, x: tla.UInt32, d: tla.UInt32
) -> None:
    with tla.vector():
        fd = tla.fast_div_get_divisor(d)
        out_q[0] = x // fd
        out_r[0] = x % fd


def test_reflected_operators_emit_simd_ops() -> None:
    out_q = make_fake_tensor(tla.UInt32, (4,), (1,))
    out_r = make_fake_tensor(tla.UInt32, (4,), (1,))
    mlir = _runtime32_kernel.dump_mlir(
        type_args=(out_q, out_r, tla.UInt32(0), tla.UInt32(0))
    )
    assert "tla.scalar_simd_fast_div" in mlir
    assert "tla.scalar_simd_fast_mod" in mlir
    assert ": i32 -> i32" in mlir


@tla.kernel
def _runtime64_kernel(
    out_q: tla.Tensor, out_r: tla.Tensor, x: tla.UInt64, d: tla.UInt64
) -> None:
    with tla.vector():
        fd = tla.fast_div_get_divisor(d)
        out_q[0] = x // fd
        out_r[0] = x % fd


def test_uint64_uses_wide_variant() -> None:
    out_q = make_fake_tensor(tla.UInt64, (4,), (1,))
    out_r = make_fake_tensor(tla.UInt64, (4,), (1,))
    mlir = _runtime64_kernel.dump_mlir(
        type_args=(out_q, out_r, tla.UInt64(0), tla.UInt64(0))
    )
    assert "tla.scalar_simd_fast_div" in mlir
    assert "tla.scalar_simd_fast_mod" in mlir
    assert ": i64 -> i64" in mlir


@tla.kernel
def _literal_kernel(out_q: tla.Tensor, out_r: tla.Tensor) -> None:
    with tla.vector():
        fd = tla.fast_div_get_divisor(7)
        out_q[0] = 100 // fd
        out_r[0] = 100 % fd


def test_python_int_dividend_is_accepted() -> None:
    out_q = make_fake_tensor(tla.UInt32, (4,), (1,))
    out_r = make_fake_tensor(tla.UInt32, (4,), (1,))
    mlir = _literal_kernel.dump_mlir(type_args=(out_q, out_r))
    assert "100 : i32" in mlir
    assert "tla.scalar_simd_fast_div" in mlir
    assert "tla.scalar_simd_fast_mod" in mlir


@tla.kernel
def _narrow_dividend_kernel(out_q: tla.Tensor, x: tla.UInt16, d: tla.UInt32) -> None:
    with tla.vector():
        fd = tla.fast_div_get_divisor(d)
        out_q[0] = x // fd


def test_narrower_dividend_widens_to_divisor_width() -> None:
    out_q = make_fake_tensor(tla.UInt32, (4,), (1,))
    mlir = _narrow_dividend_kernel.dump_mlir(
        type_args=(out_q, tla.UInt16(0), tla.UInt32(0))
    )
    assert "tla.scalar_simd_fast_div" in mlir
    assert ": i32 -> i32" in mlir


@pytest.mark.parametrize("value", [_INT32_MAX + 1, _INT64_MAX + 1, -1])
def test_rejects_out_of_range_python_int_dividend(value: int) -> None:
    def make(value: int):
        @tla.kernel
        def kernel(out_q: tla.Tensor) -> None:
            with tla.vector():
                fd = tla.fast_div_get_divisor(7)
                out_q[0] = value // fd

        return kernel

    out_q = make_fake_tensor(tla.UInt32, (4,), (1,))
    with pytest.raises(tla.TlaCoreAPIError, match="dividend"):
        make(value).dump_mlir(type_args=(out_q,))


def test_rejects_uint32_dividend_above_int32_max() -> None:
    @tla.kernel
    def kernel(out_q: tla.Tensor) -> None:
        with tla.vector():
            fd = tla.fast_div_get_divisor(7)
            out_q[0] = tla.UInt32(_INT32_MAX + 1) // fd

    out_q = make_fake_tensor(tla.UInt32, (4,), (1,))
    with pytest.raises(tla.TlaCoreAPIError, match="dividend"):
        kernel.dump_mlir(type_args=(out_q,))


@tla.kernel
def _wider_dividend_kernel(out_q: tla.Tensor, x: tla.UInt64) -> None:
    with tla.vector():
        fd = tla.fast_div_get_divisor(tla.UInt32(7))
        out_q[0] = x // fd


def test_rejects_dividend_wider_than_divisor() -> None:
    out_q = make_fake_tensor(tla.UInt32, (4,), (1,))
    with pytest.raises(tla.TlaCoreAPIError, match="wider"):
        _wider_dividend_kernel.dump_mlir(type_args=(out_q, tla.UInt64(0)))


@pytest.mark.parametrize("bad", [True, 1.5, "5"])
def test_rejects_non_integer_dividend(bad: object) -> None:
    def make(bad: object):
        @tla.kernel
        def kernel(out_q: tla.Tensor) -> None:
            with tla.vector():
                fd = tla.fast_div_get_divisor(7)
                out_q[0] = bad // fd

        return kernel

    out_q = make_fake_tensor(tla.UInt32, (4,), (1,))
    with pytest.raises((tla.TlaCoreAPIError, TypeError)):
        make(bad).dump_mlir(type_args=(out_q,))


@tla.kernel
def _outside_region_kernel(out_q: tla.Tensor, x: tla.UInt32) -> None:
    fd = tla.fast_div_get_divisor(7)
    out_q[0] = x // fd


def test_rejected_outside_core_regions() -> None:
    out_q = make_fake_tensor(tla.UInt32, (4,), (1,))
    with pytest.raises(
        tla.TlaCoreAPIError,
        match="must be nested inside tla.cube\\(\\) or tla.vector\\(\\)",
    ):
        _outside_region_kernel.dump_mlir(type_args=(out_q, tla.UInt32(0)))


@tla.kernel
def _inside_simd_kernel(out_q: tla.Tensor, x: tla.UInt32) -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            fd = tla.fast_div_get_divisor(7)
            out_q[0] = x // fd


def test_rejected_inside_vec_func() -> None:
    out_q = make_fake_tensor(tla.UInt32, (4,), (1,))
    with pytest.raises(tla.TlaCoreAPIError, match="outside tla.vec.func"):
        _inside_simd_kernel.dump_mlir(type_args=(out_q, tla.UInt32(0)))


# --- bare SSA dividend proxy ---


@tla.kernel
def _bare_dividend_kernel(
    out_q: tla.Tensor, x: tla.UInt32, d: tla.UInt32
) -> None:
    with tla.vector():
        fd = tla.fast_div_get_divisor(d)
        out_q[0] = x.ir_value() // fd


def test_accepts_bare_ssa_dividend() -> None:
    out_q = make_fake_tensor(tla.UInt32, (4,), (1,))
    mlir = _bare_dividend_kernel.dump_mlir(
        type_args=(out_q, tla.UInt32(0), tla.UInt32(0))
    )
    assert "tla.scalar_simd_fast_div" in mlir


@tla.kernel
def _wide_bare_dividend_kernel(
    out_q: tla.Tensor, x: tla.UInt64, d: tla.UInt32
) -> None:
    with tla.vector():
        fd = tla.fast_div_get_divisor(d)
        out_q[0] = x.ir_value() // fd


def test_rejects_bare_ssa_dividend_wider_than_divisor() -> None:
    out_q = make_fake_tensor(tla.UInt32, (4,), (1,))
    with pytest.raises(tla.TlaCoreAPIError, match="wider"):
        _wide_bare_dividend_kernel.dump_mlir(
            type_args=(out_q, tla.UInt64(0), tla.UInt32(0))
        )


def test_out_of_range_dividend_reports_the_api_name() -> None:
    @tla.kernel
    def kernel(out_q: tla.Tensor) -> None:
        with tla.vector():
            fd = tla.fast_div_get_divisor(7)
            out_q[0] = (_INT32_MAX + 1) // fd

    out_q = make_fake_tensor(tla.UInt32, (4,), (1,))
    with pytest.raises(tla.TlaCoreAPIError, match="fast_div_get_divisor"):
        kernel.dump_mlir(type_args=(out_q,))
