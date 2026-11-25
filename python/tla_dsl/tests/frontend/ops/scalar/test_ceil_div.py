from __future__ import annotations

import pytest

from catlass import core
import catlass.tla as tla
from catlass.tla import ceil_div


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (128, 64, 2),
        (129, 64, 3),
        (1, 64, 1),
        (100, 1, 100),
        (100, 0, 0),
        (0, 64, 0),
        (0, 0, 0),
        (2147483647, 1, 2147483647),
        (2147483647, 2147483647, 1),
        (2147483647, 2, 1073741824),
    ],
)
def test_constant_ceil_div(a, b, expected):
    assert tla.ceil_div is ceil_div is core.ceil_div
    result = ceil_div(a, b)
    assert isinstance(result, int)
    assert not isinstance(result, bool)
    assert result == expected
    for lhs, rhs in ((tla.Int32(a), b), (a, tla.Int32(b)), (tla.Int32(a), tla.Int32(b))):
        result = ceil_div(lhs, rhs)
        assert isinstance(result, tla.Int32)
        assert result.value == expected


@pytest.mark.parametrize(
    "invalid",
    [True, 1.0, tla.Bool(True), tla.Float32(1), tla.Int64(1), tla.UInt32(1)],
)
def test_ceil_div_rejects_unsupported_types(invalid):
    with pytest.raises(TypeError, match="Python int or Int32"):
        ceil_div(invalid, 1)
    with pytest.raises(TypeError, match="Python int or Int32"):
        ceil_div(1, invalid)


@pytest.mark.parametrize(
    "lhs,rhs,operand",
    [(tla.Int32(1.5), 1, "num1"), (1, tla.Int32(1.5), "num2")],
)
def test_ceil_div_rejects_float_backed_int32(lhs, rhs, operand):
    with pytest.raises(TypeError, match=f"{operand} must be an integer"):
        ceil_div(lhs, rhs)


@pytest.mark.parametrize("invalid", [-(1 << 31) - 1, 1 << 31, tla.Int32(1 << 31)])
def test_ceil_div_rejects_out_of_range_literals(invalid):
    with pytest.raises(ValueError, match="outside the Int32 range"):
        ceil_div(invalid, 0)
    with pytest.raises(ValueError, match="outside the Int32 range"):
        ceil_div(0, invalid)


@tla.kernel
def _constant_kernel(out: tla.Tensor) -> None:
    out[0] = tla.ceil_div(2147483647, 2)
    out[1] = tla.ceil_div(tla.Int32(129), 64)


@tla.kernel
def _dynamic_kernel(out: tla.Tensor, a: tla.Int32, b: tla.Int32) -> None:
    out[0] = tla.ceil_div(a, b)
    out[1] = tla.ceil_div(a, 64)
    out[2] = tla.ceil_div(129, b)
    out[3] = tla.ceil_div(a + 1, b)
    for i in tla.range(4, 6):
        out[i] = tla.ceil_div(i, b)
    chosen = a
    if a < b:
        chosen = b
    out[6] = tla.ceil_div(chosen, b)


def test_ceil_div_folds_constants_during_tracing(compiler_tlair):
    out = tla.make_fake_tensor(tla.Int32, (2,), (1,), layout_tag=tla.arch.RowMajor)
    constant = compiler_tlair(_constant_kernel, type_args=(out,))
    assert "1073741824 : i32" in constant
    assert "3 : i32" in constant
    assert "arith.divsi" not in constant
    assert "arith.remsi" not in constant
    assert "tla.ceil_div" not in constant


def test_ceil_div_traces_dynamic_operands(compiler_tlair):
    out = tla.make_fake_tensor(tla.Int32, (7,), (1,), layout_tag=tla.arch.RowMajor)
    compiler_tlair(_dynamic_kernel, type_args=(out, tla.Int32(0), tla.Int32(0)))
