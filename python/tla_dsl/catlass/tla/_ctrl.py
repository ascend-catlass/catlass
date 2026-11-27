"""Scalar CTRL register queries and masked updates."""

from catlass._mlir import ir

from .. import params
from ..base_dsl.typing import Bool, Int64, UInt64
from ..core_api import (
    _const_i64,
    _op_error,
    _require_core_region,
    _require_frontend_state,
)

# (first bit, last bit, semantic type, inverted Boolean encoding).
_FIELDS = {
    "atomic_format": (6, 8, params.AtomicFormat, False),
    "atomic_operation": (9, 10, params.AtomicOperation, False),
    "fp8_as_hifloat8": (45, 45, bool, False),
    "float_saturation": (48, 48, bool, True),
    "ub_dcache": (49, 49, bool, False),
    "fp8_nan_saturation": (50, 50, bool, True),
    "integer_saturation": (53, 53, bool, False),
    "conversion_mode": (59, 59, params.ConversionMode, False),
    "saturation_strategy": (60, 60, params.OverrideStrategy, False),
}


def _placement(name):
    _require_frontend_state(name)
    _require_core_region(name)


def _range(name, lo, hi, *, snapshot=False):
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in (lo, hi)):
        _op_error(name, "start_bit and end_bit must be compile-time integers")
    if snapshot and (lo, hi) == (0, 63):
        return
    if not any((lo, hi) == field[:2] for field in _FIELDS.values()):
        _op_error(name, "select one complete documented CTRL field")


def _bits(value, loc):
    if isinstance(value, int):
        return _const_i64(value, loc=loc)
    result = value.ir_value(loc=loc)
    i64 = ir.IntegerType.get_signless(64)
    if result.type == i64:
        return result
    op = (
        "arith.extui"
        if isinstance(value, Bool)
        else "builtin.unrealized_conversion_cast"
    )
    return ir.Operation.create(op, results=[i64], operands=[result], loc=loc).result


def _binary(name, left, right, loc):
    return ir.Operation.create(
        "arith." + name, operands=[left, right], results=[left.type], loc=loc
    ).result


def _encode(name, value, lo, hi, loc):
    if isinstance(value, params.CTRLValue):
        if lo is not None or hi is not None:
            _op_error(name, "CTRLValue cannot be combined with bit endpoints")
        if (
            value.atomic_format is params.AtomicFormat.DISABLED
            and value.atomic_operation is not None
        ):
            _op_error(name, "atomic_operation requires an enabled atomic_format")
        fields = []
        for member, (first, last, dtype, invert) in _FIELDS.items():
            val = getattr(value, member)
            if val is None:
                continue
            if dtype is bool:
                if not isinstance(val, (bool, Bool)):
                    _op_error(name, f"{member} requires bool or tla.Bool")
            elif not isinstance(val, dtype):
                _op_error(name, f"{member} requires {dtype.__name__}")
            fields.append((first, last, val, invert))
        if not fields:
            _op_error(name, "CTRLValue must select at least one field")
    else:
        _range(name, lo, hi)
        if isinstance(value, bool) or not isinstance(value, (int, Int64, UInt64)):
            _op_error(name, "raw value requires an integer, Int64 or UInt64")
        constant = value if isinstance(value, int) else value.value
        limit = (1 << (hi - lo + 1)) - 1
        if lo in (6, 9):
            if not isinstance(constant, int):
                _op_error(name, "raw atomic fields require a compile-time integer")
            limit -= 1
        if isinstance(constant, int) and not 0 <= constant <= limit:
            _op_error(
                name, "raw value exceeds the field range or uses a reserved encoding"
            )
        fields = [(lo, hi, value, False)]
    selected = 0
    literal = 0
    dynamic = []
    for first, last, val, invert in fields:
        selected |= ((1 << (last - first + 1)) - 1) << first
        constant = val if isinstance(val, int) else val.value
        if isinstance(constant, (bool, int)):
            literal |= (int(constant) ^ int(invert)) << first
        else:
            operand = _bits(val, loc)
            if invert:
                operand = _binary("xori", operand, _bits(1, loc), loc)
            dynamic.append(_binary("shli", operand, _bits(first, loc), loc))
    encoded = _bits(literal, loc)
    for operand in dynamic:
        encoded = _binary("ori", encoded, operand, loc)
    return encoded, ir.IntegerAttr.get(ir.IntegerType.get_signless(64), selected)


def get_ctrl(*, start_bit=0, end_bit=63, loc=None):
    """Read the issuer's CTRL register or a right-justified documented field."""
    _placement("arch.get_ctrl")
    _range("arch.get_ctrl", start_bit, end_bit, snapshot=True)
    i32 = ir.IntegerType.get_signless(32)
    op = ir.Operation.create(
        "tla.arch.get_ctrl",
        results=[ir.IntegerType.get_signless(64)],
        attributes={
            "lo": ir.IntegerAttr.get(i32, start_bit),
            "hi": ir.IntegerAttr.get(i32, end_bit),
        },
        loc=loc,
    )
    unsigned = ir.Operation.create(
        "builtin.unrealized_conversion_cast",
        operands=[op.result],
        results=[UInt64.mlir_type()],
        loc=loc,
    )
    return UInt64(unsigned.result)


def set_ctrl(value, *, start_bit=None, end_bit=None, loc=None):
    """Update one raw CTRL field or all fields selected by a CTRLValue patch."""
    _placement("arch.set_ctrl")
    bits, mask = _encode("arch.set_ctrl", value, start_bit, end_bit, loc)
    ir.Operation.create(
        "tla.arch.set_ctrl", operands=[bits], attributes={"mask": mask}, loc=loc
    )
