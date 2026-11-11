"""Scalar encoding and trace-time constraints for L1 Tensor.fill."""

import math
import re

import numpy as np
import pytest

from catlass import tla
from catlass.core_api import _fill_value_bits


# Expected constants are independent of the host encoders used by the API.
@pytest.mark.parametrize(
    "token,value,bits",
    [
        ("i16", -1, 65535),
        ("i16", -32768, 32768),
        ("i16", 32767, 32767),
        ("u16", 65535, 65535),
        ("u16", 32768, 32768),
        ("f16", 1.5, 0x3E00),
        ("bf16", 1.5, 0x3FC0),
        ("i32", -1, -1),
        ("i32", -2147483648, -2147483648),
        ("i32", 2147483647, 2147483647),
        ("u32", 0x80000001, -2147483647),
        ("u32", 0xFFFFFFFF, -1),
        ("f32", -1.0, -1082130432),
        ("f16", -0.0, 0x8000),
        ("bf16", -0.0, 0x8000),
        ("f32", -0.0, -2147483648),
        ("f16", math.inf, 0x7C00),
        ("f16", -math.inf, 0xFC00),
        ("bf16", math.inf, 0x7F80),
        ("bf16", -math.inf, 0xFF80),
        ("f32", math.inf, 0x7F800000),
        ("f32", -math.inf, -8388608),
        ("bf16", 1.00390625, 0x3F80),
        ("bf16", 1.01171875, 0x3F82),
        ("f16", 0, 0),
        ("bf16", 0, 0),
        ("f32", 0, 0),
    ],
)
def test_scalar_bits(token, value, bits):
    assert _fill_value_bits("fill", token, value) == bits


@pytest.mark.parametrize(
    "token,exponent,mantissa",
    [
        ("f16", 0x7C00, 0x03FF),
        ("bf16", 0x7F80, 0x007F),
        ("f32", 0x7F800000, 0x007FFFFF),
    ],
)
def test_nan_classification(token, exponent, mantissa):
    bits = _fill_value_bits("fill", token, math.nan)
    assert bits & exponent == exponent
    assert bits & mantissa


@pytest.mark.parametrize(
    "token,lower,upper",
    [
        ("i16", -32768, 32767),
        ("u16", 0, 65535),
        ("i32", -2147483648, 2147483647),
        ("u32", 0, 4294967295),
    ],
)
def test_integer_bounds(token, lower, upper):
    for value in (lower - 1, upper + 1):
        with pytest.raises(tla.TlaCoreAPIError, match="must be in"):
            _fill_value_bits("fill", token, value)
    with pytest.raises(tla.TlaCoreAPIError, match="requires an int"):
        _fill_value_bits("fill", token, 1.0)


@pytest.mark.parametrize(
    "token,value",
    [("f16", 65520.0), ("bf16", 3.4e38), ("f32", 3.5e38), ("f32", 10**400)],
)
def test_float_overflow(token, value):
    with pytest.raises(tla.TlaCoreAPIError, match="overflows|cannot be encoded"):
        _fill_value_bits("fill", token, value)


@pytest.mark.parametrize(
    "value", [True, False, None, "0", np.int32(0), np.float32(0), np.float64(0)]
)
def test_reject_non_python_scalars(value):
    with pytest.raises(tla.TlaCoreAPIError, match="trace-time int or float"):
        _fill_value_bits("fill", "f16", value)


@pytest.mark.parametrize(
    "dtype,value,bits,tag",
    [
        (tla.Int16, -1, 65535, tla.arch.zN),
        (tla.UInt16, 65535, 65535, tla.arch.nZ),
        (tla.Float16, 1.5, 15872, tla.arch.zN),
        (tla.BFloat16, 1.5, 16320, tla.arch.nZ),
        (tla.Int32, -1, -1, tla.arch.zN),
        (tla.UInt32, 0x80000001, -2147483647, tla.arch.nZ),
        (tla.Float32, -1.0, -1082130432, tla.arch.zN),
    ],
)
def test_fill_emits_encoded_i32(dtype, value, bits, tag):
    @tla.kernel
    def kernel():
        ptr = tla.allocate((32, 64), dtype, tla.AddressSpace.l1, 512)
        tile = tla.make_tensor(ptr, tla.make_layout_with_tag((32, 64), dtype, tag))
        with tla.cube():
            tile.fill(value)

    ir = kernel.dump_mlir()
    assert re.search(rf"arith.constant {bits} : i32", ir)
    assert "tla.fill" in ir


@pytest.mark.parametrize("dtype", [tla.UInt8, tla.Float8E8M0])
def test_unsupported_types(dtype):
    @tla.kernel
    def kernel():
        ptr = tla.allocate((32, 64), dtype, tla.AddressSpace.l1, 512)
        tile = tla.make_tensor(
            ptr, tla.make_layout_with_tag((32, 64), dtype, tla.arch.zN)
        )
        with tla.cube():
            tile.fill(0)

    with pytest.raises(Exception, match="supported element type"):
        kernel.dump_mlir()


def test_reject_runtime_scalar():
    @tla.kernel
    def kernel(value: tla.Int32):
        ptr = tla.allocate((32, 64), tla.Int32, tla.AddressSpace.l1, 512)
        tile = tla.make_tensor(
            ptr, tla.make_layout_with_tag((32, 64), tla.Int32, tla.arch.zN)
        )
        with tla.cube():
            tile.fill(value)

    with pytest.raises(Exception, match="trace-time int or float"):
        kernel.dump_mlir(type_args=(tla.Int32(0),))
