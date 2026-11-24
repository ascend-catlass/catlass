"""Host and frontend coverage for scalar dtype limits."""

import math

import pytest

from catlass import tla
from catlass._mlir import ir
from catlass.tla.runtime import make_fake_tensor


@pytest.mark.parametrize(
    ("dtype", "lower", "upper"),
    [
        (tla.Bool, 0, 1),
        (tla.Int8, -128, 127),
        (tla.Int16, -32768, 32767),
        (tla.Int32, -(2**31), 2**31 - 1),
        (tla.Int64, -(2**63), 2**63 - 1),
        (tla.UInt8, 0, 255),
        (tla.UInt16, 0, 65535),
        (tla.UInt32, 0, 2**32 - 1),
        (tla.UInt64, 0, 2**64 - 1),
    ],
)
def test_integer_limits(dtype, lower, upper):
    assert dtype.min == dtype.lowest == lower
    assert dtype.max == upper
    assert isinstance(dtype.max, int)
    assert not hasattr(dtype, "inf")


@pytest.mark.parametrize(
    ("dtype", "max_bits", "min_bits", "inf_bits", "quiet_bits", "signaling_bits"),
    [
        (tla.Float16, 0x7BFF, 0x0400, 0x7C00, 0x7E00, 0x7D00),
        (tla.BFloat16, 0x7F7F, 0x0080, 0x7F80, 0x7FC0, 0x7FA0),
        (tla.Float32, 0x7F7FFFFF, 0x00800000, 0x7F800000, 0x7FC00000, 0x7FA00000),
        (
            tla.Float64,
            0x7FEFFFFFFFFFFFFF,
            0x0010000000000000,
            0x7FF0000000000000,
            0x7FF8000000000000,
            0x7FF4000000000000,
        ),
    ],
)
def test_float_limits(dtype, max_bits, min_bits, inf_bits, quiet_bits, signaling_bits):
    assert 0 < dtype.denorm_min < dtype.min < dtype.max
    assert dtype.lowest == -dtype.max
    assert dtype.inf == dtype.infinity == math.inf
    assert dtype.neg_inf == dtype.negative_infinity == -math.inf
    assert math.isnan(dtype.nan) and math.isnan(dtype.quiet_nan)
    assert math.isnan(dtype.signaling_nan)
    mask = (1 << dtype.width) - 1
    sign = 1 << (dtype.width - 1)
    for value, bits in (
        (dtype.max, max_bits),
        (dtype.lowest, sign | max_bits),
        (dtype.min, min_bits),
        (dtype.denorm_min, 1),
        (dtype.inf, inf_bits),
        (dtype.neg_inf, sign | inf_bits),
        (dtype.quiet_nan, quiet_bits),
        (dtype.signaling_nan, signaling_bits),
    ):
        (host_bits,) = dtype(value).__c_pointers__()
        assert host_bits & mask == bits

    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            literal = dtype(dtype.signaling_nan).ir_value()
            expected = ir.Attribute.parse(f"0x{signaling_bits:X} : {literal.type}")
            assert literal.owner.attributes["value"] == expected
        assert module.operation.verify()


@tla.kernel
def _store_limits(
    int_out: tla.Tensor,
    uint_out: tla.Tensor,
    float_out: tla.Tensor,
    bool_out: tla.Tensor,
):
    int_out[0] = tla.Int32.max
    uint_out[0] = tla.UInt64.max
    uint_out[1] = tla.UInt64(tla.UInt64.max)
    float_out[0] = tla.Float32.signaling_nan
    bool_out[0] = tla.Bool.min
    bool_out[1] = tla.Bool.max


def test_limits_in_kernel(require_bc):
    type_args = tuple(
        make_fake_tensor(dtype, (2,), (1,), layout_tag=tla.arch.RowMajor)
        for dtype in (tla.Int32, tla.UInt64, tla.Float32, tla.Bool)
    )
    mlir = _store_limits.dump_mlir(type_args=type_args)
    assert "2147483647 : i32" in mlir
    assert "-1 : i64" in mlir
    assert "0x7FA00000 : f32" in mlir
    assert "arith.constant false" in mlir
    assert "arith.constant true" in mlir
    _store_limits.compile(type_args=type_args, options="--npu-arch 3510")


@tla.kernel
def _store_float_argument(out: tla.Tensor, value: tla.Float32):
    out[0] = value


def test_signaling_nan_launch_payload(require_bc):
    out = make_fake_tensor(tla.Float32, (1,), (1,), layout_tag=tla.arch.RowMajor)
    out.data_ptr = 0x1000
    out._external_binding = True
    compiled = _store_float_argument.compile(
        type_args=(out, tla.Float32(0.0)), options="--npu-arch 3510"
    )
    for value in (tla.Float32.signaling_nan, tla.Float32(tla.Float32.signaling_nan)):
        payload = compiled.execution_args.generate_launch_payload((out, value))
        assert payload[8:12] == bytes.fromhex("0000a07f")
