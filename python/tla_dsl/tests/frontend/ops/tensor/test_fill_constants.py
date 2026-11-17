"""Numeric constants retain their destination format on the L1 fill route."""

import math
import re
from fractions import Fraction

import pytest

from catlass import tla
from catlass.core_api import _fill_value_bits
from catlass.execution_lowering import TlaLoweringError


@pytest.mark.parametrize(
    "token,value,pattern",
    [
        ("f8e4m3fn", 1.0, 0x3838),
        ("f8e4m3fn", -2.0, 0xC0C0),
        ("f8e4m3fn", 0.15625, 0x2222),
        ("f8e4m3fn", -0.0, 0x8080),
        ("f8e4m3fn", 0, 0),
        ("f8e5m2", 1.0, 0x3C3C),
        ("f8e5m2", -1.5, 0xBEBE),
        ("f8e5m2", 57344.0, 0x7B7B),
        ("f8e5m2", 0, 0),
        ("f4e2m1", 1.0, 0x2222),
        ("f4e2m1", -3.0, 0xDDDD),
        ("f4e2m1", 0, 0),
        ("f4e1m2", 1.0, 0x4444),
        ("f4e1m2", -0.75, 0xBBBB),
        ("f4e1m2", 0, 0),
        ("f8e8m0", 1.0, 0x7F7F),
        ("f8e8m0", 2.0, 0x8080),
        ("f8e8m0", 0.5, 0x7E7E),
        ("f8e8m0", math.ldexp(1.0, -127), 0x0000),
        ("f8e8m0", math.ldexp(1.0, 127), 0xFEFE),
        ("f8e8m0", 1, 0x7F7F),
        ("f8e8m0", 2**53, 0xB4B4),
        ("f8e8m0", 2**54, 0xB5B5),
        ("f8e8m0", 2**126, 0xFDFD),
        ("f8e8m0", 2**127, 0xFEFE),
    ],
)
def test_numeric_fill_pattern(token, value, pattern):
    assert _fill_value_bits("fill", token, value) == pattern


_FILL_FORMATS = [
    ("f4e2m1", 4, 1, 1, 0x7),
    ("f4e1m2", 4, 2, 1, 0x7),
    ("f8e4m3fn", 8, 3, 7, 0x7E),
    ("f8e5m2", 8, 2, 15, 0x7B),
]


def _positive_finite_values(mantissa_bits, bias, last_code):
    # Decode the format with exact rational arithmetic, independently of fill.
    values = []
    for code in range(last_code + 1):
        exponent, mantissa = divmod(code, 1 << mantissa_bits)
        significand = Fraction(mantissa, 1 << mantissa_bits)
        power = 1 - bias
        if exponent:
            significand += 1
            power = exponent - bias
        values.append(float(significand * Fraction(2) ** power))
    return values


@pytest.mark.parametrize("token,width,mantissa_bits,bias,last_code", _FILL_FORMATS)
def test_fill_preserves_every_finite_encoding(
    token, width, mantissa_bits, bias, last_code
):
    values = _positive_finite_values(mantissa_bits, bias, last_code)
    repeat_pattern = 0x1111 if width == 4 else 0x0101
    for code, magnitude in enumerate(values):
        for sign in (0, 1):
            value = math.copysign(magnitude, -1 if sign else 1)
            expected = (code | (sign << (width - 1))) * repeat_pattern
            assert _fill_value_bits("fill", token, value) == expected
            if value and value.is_integer():
                assert _fill_value_bits("fill", token, int(value)) == expected


@pytest.mark.parametrize("token,width,mantissa_bits,bias,last_code", _FILL_FORMATS)
def test_fill_rejects_midpoints_and_their_neighbours(
    token, width, mantissa_bits, bias, last_code
):
    values = _positive_finite_values(mantissa_bits, bias, last_code)
    for lower, upper in zip(values, values[1:]):
        midpoint = (lower + upper) / 2
        for magnitude in (
            math.nextafter(midpoint, lower),
            midpoint,
            math.nextafter(midpoint, upper),
        ):
            for value in (magnitude, -magnitude):
                with pytest.raises(
                    tla.TlaCoreAPIError, match="not exactly representable"
                ):
                    _fill_value_bits("fill", token, value)


@pytest.mark.parametrize("token", ["f4e2m1", "f4e1m2", "f8e4m3fn", "f8e5m2"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_low_precision_fill_rejects_nonfinite_constants(token, value):
    with pytest.raises(tla.TlaCoreAPIError, match="finite"):
        _fill_value_bits("fill", token, value)


@pytest.mark.parametrize(
    "dtype,value",
    [
        (tla.Float4E2M1, 1.25),
        (tla.Float4E1M2, 1.125),
        (tla.Float8E4M3FN, 1.0625),
        (tla.Float8E5M2, 1.125),
        (tla.Float4E2M1, 1.1),
        (tla.Float4E1M2, 1.1),
        (tla.Float8E4M3FN, 1.1),
        (tla.Float8E5M2, 1.1),
    ],
)
def test_inexact_fill_fails_during_tracing(dtype, value):
    @tla.kernel
    def kernel():
        ptr = tla.allocate((16, 64), dtype, tla.AddressSpace.l1, 512)
        tensor = tla.make_tensor(
            ptr, tla.make_layout_with_tag((16, 64), dtype, tla.arch.zN)
        )
        with tla.cube():
            tensor.fill(value)

    with pytest.raises(
        (TlaLoweringError, tla.TlaCoreAPIError),
        match="not exactly representable",
    ):
        kernel.dump_mlir()


@pytest.mark.parametrize(
    "dtype,tag,shape,pattern",
    [
        (tla.Float8E4M3FN, tla.arch.zN, (32, 64), 0x3838),
        (tla.Float8E5M2, tla.arch.nZ, (64, 32), 0x3C3C),
        (tla.Float4E2M1, tla.arch.zN, (32, 64), 0x2222),
        (tla.Float4E1M2, tla.arch.nZ, (64, 32), 0x4444),
        (tla.Float8E8M0, tla.arch.zZMxScale, (32, 8), 0x7F7F),
        (tla.Float8E8M0, tla.arch.nNMxScale, (8, 32), 0x7F7F),
    ],
)
def test_nonzero_fill_builds_typed_ir(dtype, tag, shape, pattern):
    @tla.kernel
    def kernel():
        ptr = tla.allocate(shape, dtype, tla.AddressSpace.l1, 512)
        tensor = tla.make_tensor(ptr, tla.make_layout_with_tag(shape, dtype, tag))
        with tla.cube():
            tensor.fill(1.0)

    ir = kernel.dump_mlir()
    assert "tla.fill" in ir
    assert re.search(rf"arith.constant {pattern} : i32", ir)


@pytest.mark.parametrize("value", [0, -1, 1.5, math.inf, math.nan, 2.0**-128, 2.0**128])
def test_e8m0_fill_rejects_unrepresentable_scale(value):
    with pytest.raises(tla.TlaCoreAPIError, match="E8M0|finite"):
        _fill_value_bits("fill", "f8e8m0", value)


@pytest.mark.parametrize(
    "token,value", [("f8e4m3fn", 449), ("f8e5m2", 60000), ("f4e2m1", 7), ("f4e1m2", 2)]
)
def test_fill_rejects_out_of_range_constants(token, value):
    with pytest.raises(tla.TlaCoreAPIError, match="range"):
        _fill_value_bits("fill", token, value)


@pytest.mark.parametrize("value", [True, "1", None])
def test_fill_rejects_non_numeric_values(value):
    with pytest.raises(tla.TlaCoreAPIError, match="trace-time"):
        _fill_value_bits("fill", "f8e4m3fn", value)


def test_fill_rejects_runtime_value():
    @tla.kernel
    def kernel(value: tla.Int32):
        ptr = tla.allocate((32, 64), tla.Float8E4M3FN, tla.AddressSpace.l1, 512)
        tensor = tla.make_tensor(
            ptr, tla.make_layout_with_tag((32, 64), tla.Float8E4M3FN, tla.arch.zN)
        )
        with tla.cube():
            tensor.fill(value)

    with pytest.raises((TlaLoweringError, tla.TlaCoreAPIError), match="trace-time"):
        kernel.dump_mlir(type_args=(tla.Int32(1),))


def test_e8m0_fill_rejects_operand_layout():
    @tla.kernel
    def kernel():
        ptr = tla.allocate((32, 64), tla.Float8E8M0, tla.AddressSpace.l1, 512)
        tensor = tla.make_tensor(
            ptr, tla.make_layout_with_tag((32, 64), tla.Float8E8M0, tla.arch.zN)
        )
        with tla.cube():
            tensor.fill(1.0)

    with pytest.raises(
        (TlaLoweringError, tla.TlaCoreAPIError), match="E8M0 fill requires"
    ):
        kernel.dump_mlir()


@pytest.mark.parametrize(
    "value",
    [
        2**53 - 1,
        2**53 + 1,
        2**54 - 1,
        2**54 + 1,
        2**126 - 1,
        2**126 + 1,
        2**127 - 1,
        2**127 + 1,
        2**128,
        2**1024,
    ],
)
def test_e8m0_fill_rejects_inexact_integer(value):
    with pytest.raises(tla.TlaCoreAPIError, match="E8M0|finite"):
        _fill_value_bits("fill", "f8e8m0", value)


@pytest.mark.parametrize(
    "tag,shape",
    [(tla.arch.zZMxScale, (32, 8)), (tla.arch.nNMxScale, (8, 32))],
)
@pytest.mark.parametrize(
    "value,pattern",
    [
        (2**53, 0xB4B4),
        (2**127, 0xFEFE),
        (2**53 + 1, None),
        (2**54 - 1, None),
        (2**127 - 1, None),
        (2**127 + 1, None),
    ],
)
def test_e8m0_fill_preserves_integer_precision_in_kernel(tag, shape, value, pattern):
    @tla.kernel
    def kernel():
        ptr = tla.allocate(shape, tla.Float8E8M0, tla.AddressSpace.l1, 512)
        tensor = tla.make_tensor(
            ptr, tla.make_layout_with_tag(shape, tla.Float8E8M0, tag)
        )
        with tla.cube():
            tensor.fill(value)

    if pattern is None:
        with pytest.raises(
            (TlaLoweringError, tla.TlaCoreAPIError), match="E8M0 fill requires"
        ):
            kernel.dump_mlir()
    else:
        ir = kernel.dump_mlir()
        assert re.search(rf"arith.constant {pattern} : i32", ir)
