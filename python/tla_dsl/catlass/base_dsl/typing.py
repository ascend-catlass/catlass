"""Typing protocols, ``Numeric`` value model, and ``Pointer`` ABC."""

from __future__ import annotations

import ctypes
import math
import operator
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    ClassVar,
    Generic,
    Protocol,
    TypeVar,
    cast as typing_cast,
    get_origin,
    runtime_checkable,
)

import numpy as np
from catlass._mlir import ir as mlir_ir  # type: ignore[assignment]

from ..address_space import AddressSpace
from .op import (
    dsl_user_op,
    _bind_frontend_category,
    _bind_frontend_value,
    _current_frontend_state,
)

_T = TypeVar("_T")

_TOKEN_TO_NUMERIC: dict[str, type["Numeric"]] = {}


def _binary_op(
    op: Callable[..., Any],
    *,
    flip: bool = False,
) -> Callable[..., Any]:
    """Binary op wrapper on Numeric (emits ``arith.*``)."""

    _COMPARE_OPS = (
        operator.eq,
        operator.ne,
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,
    )

    def wrapper(
        lhs: "Numeric",
        rhs: Any,
        *,
        loc: mlir_ir.Location | None = None,
    ) -> Any:
        from catlass._mlir.dialects import arith

        if not isinstance(rhs, Numeric):
            if not isinstance(rhs, (int, float, bool, mlir_ir.Value)):
                return NotImplemented
            rhs = as_numeric(rhs)

        in_frontend = _current_frontend_state() is not None
        host_fold = (
            not in_frontend
            and isinstance(lhs.value, (int, float, bool))
            and isinstance(rhs.value, (int, float, bool))
        )

        # No silent promotion: both operands must already share a concrete type.
        if type(lhs) is not type(rhs):
            raise TypeError(
                f"Numeric operands must have the same type, got "
                f"{type(lhs).__name__} and {type(rhs).__name__}; "
                f"cast explicitly with .to(...) before the operation"
            )

        # ``/`` is float-only; integers must use ``//`` (no truncate / promote).
        if op is operator.truediv and type(lhs).is_integer:
            raise TypeError(
                "Numeric '/' is only supported for float types; "
                "use '//' for integer division, or cast to Float32 first"
            )

        # ``**`` is float-only; Ascend/TLA backend does not lower ``math.ipowi``.
        if op is operator.pow and type(lhs).is_integer:
            raise TypeError(
                "Numeric '**' is only supported for float types; "
                "cast to Float32 first (integer pow is not supported by the backend)"
            )

        res_type: type[Numeric] = Bool if op in _COMPARE_OPS else type(lhs)

        if host_fold:
            a, b = lhs.value, rhs.value
            if flip:
                a, b = b, a
            return res_type(op(a, b))

        if not in_frontend:
            raise RuntimeError(
                "Numeric arithmetic on SSA values requires TLA frontend lowering context"
            )

        lhs_ir = lhs.ir_value(loc=loc)
        rhs_ir = rhs.ir_value(loc=loc)
        if flip:
            lhs_ir, rhs_ir = rhs_ir, lhs_ir

        if op in _COMPARE_OPS:
            # Inside a SIMT region a comparison is a TLA op like the arithmetic
            # ones; tla-vector-region lowers it back to the arith op below.
            _SIMT_CMP_MODES = {
                operator.eq: "eq",
                operator.ne: "ne",
                operator.lt: "lt",
                operator.le: "le",
                operator.gt: "gt",
                operator.ge: "ge",
            }
            from ..runtime import _in_simt_vec_func

            if _in_simt_vec_func():
                attributes = {
                    "mode": mlir_ir.StringAttr.get(_SIMT_CMP_MODES[op]),
                }
                if not type(lhs).is_float and not type(lhs).signed:
                    attributes["isUnsigned"] = mlir_ir.UnitAttr.get()
                result = mlir_ir.Operation.create(
                    "tla.simt_cmp",
                    operands=[lhs_ir, rhs_ir],
                    results=[mlir_ir.IntegerType.get_signless(1)],
                    attributes=attributes,
                    loc=loc,
                ).results[0]
                return Bool(result)

            fpred, ipred = {
                operator.eq: ("OEQ", "eq"),
                operator.ne: ("ONE", "ne"),
                operator.lt: ("OLT", "slt"),
                operator.le: ("OLE", "sle"),
                operator.gt: ("OGT", "sgt"),
                operator.ge: ("OGE", "sge"),
            }[op]
            if type(lhs).is_float:
                result = arith.CmpFOp(
                    getattr(arith.CmpFPredicate, fpred), lhs_ir, rhs_ir, loc=loc
                ).result
            else:
                if (
                    op in (operator.lt, operator.le, operator.gt, operator.ge)
                    and not type(lhs).signed
                ):
                    ipred = {"slt": "ult", "sle": "ule", "sgt": "ugt", "sge": "uge"}[
                        ipred
                    ]
                result = arith.CmpIOp(
                    getattr(arith.CmpIPredicate, ipred), lhs_ir, rhs_ir, loc=loc
                ).result
            return Bool(result)

        unsigned = not res_type.signed
        # Inside a SIMT region the scalar arithmetic operators are TLA ops in
        # their own right, so the per-thread body stays recognisable as TLA IR;
        # tla-vector-region lowers them back to the arith ops below. This
        # generalises the base's add-only routing to the whole set.
        _SIMT_OPS = {
            operator.add: "tla.simt_add",
            operator.sub: "tla.simt_sub",
            operator.mul: "tla.simt_mul",
            operator.truediv: "tla.simt_div",
            operator.floordiv: "tla.simt_div",
            operator.pow: "tla.simt_pow",
        }
        if op in _SIMT_OPS:
            from ..runtime import _in_simt_vec_func

            # tla.simt_div lowers integers to arith.divsi, so an unsigned '//'
            # keeps the general path and its arith.divui.
            simt_eligible = not (
                op is operator.floordiv and not res_type.is_float and unsigned
            )
            if _in_simt_vec_func() and simt_eligible:
                if op is operator.truediv and not res_type.is_float:
                    raise TypeError(
                        "Numeric '/' is only supported for float types; use '//' for integers"
                    )
                if op is operator.floordiv and res_type.is_float:
                    raise TypeError(
                        "Numeric '//' is only supported for integer types; use '/' for floats"
                    )
                result = mlir_ir.Operation.create(
                    _SIMT_OPS[op],
                    operands=[lhs_ir, rhs_ir],
                    results=[res_type.mlir_type()],
                    loc=loc,
                ).results[0]
                return res_type(result)
        if op is operator.add:
            name = "arith.addf" if res_type.is_float else "arith.addi"
        elif op is operator.sub:
            name = "arith.subf" if res_type.is_float else "arith.subi"
        elif op is operator.mul:
            name = "arith.mulf" if res_type.is_float else "arith.muli"
        elif op is operator.truediv:
            # Integer ``/`` already rejected above; only float ``divf`` remains.
            name = "arith.divf"
        elif op is operator.floordiv:
            if res_type.is_float:
                raise TypeError(
                    "Numeric '//' is only supported for integer types; use '/' for floats"
                )
            name = "arith.divui" if unsigned else "arith.divsi"
        elif op is operator.mod:
            if res_type.is_float:
                raise TypeError("Numeric '%' is only supported for integer types")
            name = "arith.remui" if unsigned else "arith.remsi"
        elif op is operator.pow:
            # Integer ``**`` already rejected above; only float ``powf`` remains.
            name = "math.powf"
        elif op is operator.and_:
            if res_type.is_float:
                raise TypeError("Numeric '&' is only supported for integer types")
            name = "arith.andi"
        elif op is operator.or_:
            if res_type.is_float:
                raise TypeError("Numeric '|' is only supported for integer types")
            name = "arith.ori"
        elif op is operator.xor:
            if res_type.is_float:
                raise TypeError("Numeric '^' is only supported for integer types")
            name = "arith.xori"
        elif op is operator.lshift:
            if res_type.is_float:
                raise TypeError("Numeric '<<' is only supported for integer types")
            name = "arith.shli"
        elif op is operator.rshift:
            if res_type.is_float:
                raise TypeError("Numeric '>>' is only supported for integer types")
            name = "arith.shrui" if unsigned else "arith.shrsi"
        else:
            raise TypeError(f"unsupported Numeric operator: {op!r}")

        result = mlir_ir.Operation.create(
            name,
            operands=[lhs_ir, rhs_ir],
            results=[res_type.mlir_type()],
            loc=loc,
        ).results[0]
        return res_type(result)

    return wrapper


# ScalarRoundMode -> tla.scalar_round_cast mode attribute. TRUNC is absent
# because there is nothing to map it to: probing ccec for dav-c310 shows the
# scalar unit has conv_f322s32{r,a,f,c} and no toward-zero form, so TRUNC stays
# on arith.fptosi.
_SCALAR_ROUND_CAST_MODE = {}


def _scalar_round_cast_mode_table() -> dict:
    """Built lazily: params imports typing, so the enum cannot be imported here."""
    global _SCALAR_ROUND_CAST_MODE
    if not _SCALAR_ROUND_CAST_MODE:
        from ..params import ScalarRoundMode

        _SCALAR_ROUND_CAST_MODE = {
            ScalarRoundMode.NEAREST_EVEN: "rn",
            ScalarRoundMode.NEAREST_AWAY: "ra",
            ScalarRoundMode.FLOOR: "rd",
            ScalarRoundMode.CEIL: "ru",
            ScalarRoundMode.ODD: "o",
        }
    return _SCALAR_ROUND_CAST_MODE


def _round_f32_to_f16_odd(value: float) -> float:
    """Constant-fold f32 -> f16 round-to-odd, matching ``conv_f322f16o``.

    Round to odd keeps an exactly representable value unchanged; otherwise it
    picks whichever of the two neighbouring f16 values has an odd mantissa,
    which is always the one nearer zero unless that one is even. Implemented on
    the bit pattern rather than with a rounding mode, since Python has none.
    """
    v32 = np.float32(value)
    if not np.isfinite(v32):
        # inf and NaN pass through any rounding unchanged.
        return float(np.float16(v32))
    nearest = np.float16(v32)
    if np.float32(nearest) == v32:
        return float(nearest)  # exactly representable: no rounding to do
    # Step to the neighbour nearer zero, if round-to-nearest went outward.
    toward_zero = nearest
    if abs(np.float32(nearest)) > abs(v32):
        toward_zero = np.nextafter(nearest, np.float16(0.0))
    if int(toward_zero.view(np.uint16)) & 1:
        return float(toward_zero)
    away = np.float16(np.inf) if v32 > 0 else np.float16(-np.inf)
    return float(np.nextafter(toward_zero, away))


def _round_host_float(value: float, round_mode: Any) -> int:
    """Constant-fold a float -> int conversion under an explicit rounding mode."""
    from ..params import ScalarRoundMode

    if round_mode is ScalarRoundMode.TRUNC:
        return int(value)
    if round_mode is ScalarRoundMode.FLOOR:
        return math.floor(value)
    if round_mode is ScalarRoundMode.CEIL:
        return math.ceil(value)
    if round_mode is ScalarRoundMode.NEAREST_EVEN:
        # Python's round() is round-half-to-even, matching conv_f322s32r.
        return round(value)
    if round_mode is ScalarRoundMode.NEAREST_AWAY:
        # Ties away from zero, matching conv_f322s32a; round() would go to even.
        return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)
    raise TypeError(f"unsupported round_mode {round_mode}")


def _validate_scalar_round_mode(value: "Numeric", dtype: Any, round_mode: Any) -> None:
    """Reject round_mode anywhere it would be silently ignored."""
    from ..params import RoundMode, ScalarRoundMode

    if isinstance(round_mode, RoundMode):
        raise TypeError(
            f"round_mode {round_mode} is a RoundMode, which configures the AVE "
            "vector cast (tla.cast). A scalar conversion takes a "
            "tla.params.ScalarRoundMode"
        )
    if not isinstance(round_mode, ScalarRoundMode):
        raise TypeError(
            "round_mode must be a tla.params.ScalarRoundMode, got "
            f"{type(round_mode).__name__}"
        )
    src_ty = type(value)
    if not (isinstance(dtype, NumericMeta) and not dtype.is_abstract):
        raise TypeError(
            "round_mode applies only to a conversion between concrete Numeric "
            f"types, not to {getattr(dtype, '__name__', dtype)}"
        )
    # ODD is the one f32 -> f16 instruction; every other mode targets i32.
    if round_mode is ScalarRoundMode.ODD:
        if src_ty.width != 32 or not src_ty.is_float or dtype is not Float16:
            raise TypeError(
                "round_mode ODD is the f32 -> f16 round-to-odd conversion and "
                "applies only to Float32 -> Float16. Got "
                f"{src_ty.__name__} -> {dtype.__name__}"
            )
        return
    if not (src_ty.is_float and dtype.is_integer):
        raise TypeError(
            "round_mode applies only to a float -> integer conversion, "
            f"not {src_ty.__name__} -> {dtype.__name__} (ODD is the one "
            "float -> float mode)"
        )
    if round_mode is ScalarRoundMode.TRUNC:
        return
    if src_ty.width != 32 or dtype.width != 32 or not dtype.signed:
        raise TypeError(
            "round_mode is supported only for Float32 -> Int32; the scalar unit "
            "has no rounding conversion for other widths. Got "
            f"{src_ty.__name__} -> {dtype.__name__}"
        )
    if round_mode not in _scalar_round_cast_mode_table():
        raise TypeError(f"unsupported round_mode {round_mode}")


def _decorate_generated_method(
    owner: str,
    name: str,
    method: Callable[..., Any],
    *,
    comparison: bool = False,
) -> Callable[..., Any]:
    """Preserve public metadata and decorate a generated numeric method."""
    method.__name__ = name
    method.__qualname__ = f"{owner}.{name}"
    method.__annotations__["return"] = "Any" if comparison else "Numeric"
    return dsl_user_op(method)


def _binary_method(
    owner: str,
    name: str,
    op: Callable[..., Any],
    *,
    flip: bool = False,
    comparison: bool = False,
) -> Callable[..., Any]:
    """Create a user-facing special method backed by ``_binary_op``."""

    def method(
        self,
        other: Any,
        *,
        loc: mlir_ir.Location | None = None,
    ) -> Any:
        return _binary_op(op, flip=flip)(self, other, loc=loc)

    return _decorate_generated_method(owner, name, method, comparison=comparison)


def _reflected_method(
    owner: str,
    name: str,
    forward_name: str,
) -> Callable[..., Any]:
    """Create a reflected method that preserves forward-method overrides."""

    def method(
        self,
        other: Any,
        *,
        loc: mlir_ir.Location | None = None,
    ) -> Any:
        return getattr(self, forward_name)(other, loc=loc)

    return _decorate_generated_method(owner, name, method)


def _reflected_shift_method(
    name: str,
    forward_name: str,
    direction: str,
) -> Callable[..., Any]:
    """Create a reflected integer shift with the public operand diagnostic."""

    def method(
        self,
        other: Any,
        *,
        loc: mlir_ir.Location | None = None,
    ) -> Any:
        other_ = as_numeric(other)
        if not isinstance(other_, Integer):
            raise ValueError(f"Cannot {direction} shift {other_} with {self}")
        return getattr(other_, forward_name)(self, loc=loc)

    return _decorate_generated_method("Integer", name, method)


class DslType(type):
    """Metaclass for DSL types: ``is_abstract`` + type identity."""

    _is_abstract: bool

    def __new__(
        cls,
        name: str,
        bases: tuple,
        attrs: dict,
        is_abstract: bool = False,
        **kwargs: Any,
    ) -> Any:
        new_cls = super().__new__(cls, name, bases, attrs)
        new_cls._is_abstract = is_abstract
        return new_cls

    @property
    def is_abstract(cls) -> bool:
        return cls._is_abstract


def _emit_scalar_cast(cast_name, src, dst_ty, loc):
    """Emit a scalar conversion, as tla.simt_cast inside a SIMT region.

    Outside one this stays a plain arith op, the way it always was.
    """
    from ..runtime import _in_simt_vec_func

    if _in_simt_vec_func():
        return mlir_ir.Operation.create(
            "tla.simt_cast",
            operands=[src],
            results=[dst_ty],
            attributes={"kind": mlir_ir.StringAttr.get(cast_name.split(".")[1])},
            loc=loc,
        ).results[0]
    return mlir_ir.Operation.create(
        cast_name, operands=[src], results=[dst_ty], loc=loc
    ).results[0]


class NumericMeta(DslType):
    """Metaclass for numeric types.

    Class-creation kwargs: ``width``, ``dtype`` (TLA token), ``mlir_type`` factory,
    ``is_abstract``.
    """

    width: int
    dtype: str
    signed: bool
    _mlir_type_factory: Callable[[], mlir_ir.Type] | None

    def __new__(
        cls,
        name: str,
        bases: tuple,
        attrs: dict,
        width: int = 0,
        dtype: str = "",
        mlir_type: Callable[[], mlir_ir.Type] | None = None,
        is_abstract: bool = False,
        signed: bool = True,
        **kwargs: Any,
    ) -> Any:
        del kwargs

        def _extract_mlir_values(self: "Numeric") -> tuple[mlir_ir.Value, ...]:
            return (self.ir_value(),)

        def _new_from_mlir_values(
            self: "Numeric", values: list[Any] | tuple[Any, ...]
        ) -> "Numeric":
            if len(values) != 1:
                raise TypeError(
                    f"Numeric expects 1 MLIR value after dynamic if, got {len(values)}"
                )
            return type(self)(values[0])

        new_attrs = {
            "__extract_mlir_values__": _extract_mlir_values,
            "__new_from_mlir_values__": _new_from_mlir_values,
        }
        # Prefer explicit methods defined on the class body.
        new_attrs.update(attrs)
        new_cls = super().__new__(cls, name, bases, new_attrs, is_abstract=is_abstract)
        new_cls.width = width
        new_cls.dtype = dtype
        new_cls.signed = signed
        new_cls._mlir_type_factory = staticmethod(mlir_type) if mlir_type else None
        if dtype and not is_abstract:
            token = dtype.strip().lower()
            if token not in _TOKEN_TO_NUMERIC:
                _TOKEN_TO_NUMERIC[token] = typing_cast(type[Numeric], new_cls)
        return new_cls

    def mlir_type(cls, ctx: mlir_ir.Context | None = None) -> mlir_ir.Type:
        """Return MLIR type for this Numeric class (optional ``ctx``)."""
        factory = cls._mlir_type_factory
        if factory is None:
            raise TypeError(f"{cls.__name__} has no mlir_type factory")
        if ctx is None:
            return factory()
        with ctx:
            return factory()

    def is_same_kind(cls, other: type) -> bool:
        return (cls.is_integer and getattr(other, "is_integer", False)) or (
            cls.is_float and getattr(other, "is_float", False)
        )

    @property
    def is_integer(cls) -> bool:
        return False

    @property
    def is_float(cls) -> bool:
        return False


class IntegerMeta(NumericMeta):
    """Metaclass for integer numeric types."""

    def __new__(
        cls,
        name: str,
        bases: tuple,
        attrs: dict,
        width: int = 32,
        signed: bool = True,
        dtype: str = "",
        mlir_type: Callable[[], mlir_ir.Type] | None = None,
        is_abstract: bool = False,
        **kwargs: Any,
    ) -> Any:
        # Inject ``__c_pointers__`` on Integer family (host bits as ``list[int]``).
        def _c_pointers(self: "Integer") -> list[int]:
            if isinstance(self.value, mlir_ir.Value):
                raise TypeError(
                    f"{type(self).__name__} SSA value cannot provide host __c_pointers__"
                )
            if width == 1:
                return [1 if bool(self.value) else 0]
            if signed:
                c_value = getattr(ctypes, f"c_int{width}")(int(self.value))
                return [c_value.value & ((1 << width) - 1)]
            return [int(self.value) & ((1 << width) - 1)]

        # Class body wins if it defines ``__c_pointers__``; else metaclass injects.
        injected = {"__c_pointers__": _c_pointers}
        merged = {**injected, **attrs}
        return super().__new__(
            cls,
            name,
            bases,
            merged,
            width=width,
            dtype=dtype,
            mlir_type=mlir_type,
            is_abstract=is_abstract,
            signed=signed,
            **kwargs,
        )

    @property
    def is_integer(cls) -> bool:
        return True

    @property
    def is_float(cls) -> bool:
        return False


class FloatMeta(NumericMeta):
    """Metaclass for floating-point numeric types."""

    def __new__(
        cls,
        name: str,
        bases: tuple,
        attrs: dict,
        width: int = 32,
        dtype: str = "",
        mlir_type: Callable[[], mlir_ir.Type] | None = None,
        is_abstract: bool = False,
        **kwargs: Any,
    ) -> Any:
        return super().__new__(
            cls,
            name,
            bases,
            attrs,
            width=width,
            dtype=dtype,
            mlir_type=mlir_type,
            is_abstract=is_abstract,
            signed=True,
            **kwargs,
        )

    @property
    def is_integer(cls) -> bool:
        return False

    @property
    def is_float(cls) -> bool:
        return True


@dsl_user_op
def cast(
    obj: Any,
    type_: type["Numeric"],
    *,
    loc: mlir_ir.Location | None = None,
) -> "Numeric":
    """Cast ``obj`` to a concrete Numeric type.

    ``cast`` constructs the target type (``type_(obj)``). Cross-type SSA
    conversion is done in ``Numeric.__init__`` / ``.to`` (via arith ops), not by
    rebinding the raw SSA value.
    """
    if not isinstance(type_, NumericMeta):
        raise TypeError(f"cast target must be a Numeric type, got {type_!r}")
    if type_.is_abstract:
        if not isinstance(obj, type_):
            raise TypeError(
                f"can't cast {obj!r} to abstract {type_.__name__}. "
                "Pass a concrete type instead, e.g. Int32, Float32."
            )
        return obj
    ctor = typing_cast(Callable[..., Numeric], type_)
    return ctor(obj, loc=loc)


def _mlir_i(width: int, *, unsigned: bool = False) -> Callable[[], mlir_ir.Type]:
    def factory() -> mlir_ir.Type:
        if unsigned:
            return mlir_ir.IntegerType.get_unsigned(width)
        return mlir_ir.IntegerType.get_signless(width)

    return factory


def _mlir_f16() -> mlir_ir.Type:
    return mlir_ir.F16Type.get()


def _mlir_bf16() -> mlir_ir.Type:
    return mlir_ir.BF16Type.get()


def _mlir_f32() -> mlir_ir.Type:
    return mlir_ir.F32Type.get()


def _mlir_f8e4m3fn() -> mlir_ir.Type:
    # OCP E4M3 with finite-only encoding -- the format CANN calls fp8_e4m3fn_t.
    # Deliberately Float8E4M3FNType, not Float8E4M3Type: MLIR carries both, and
    # they are different formats. The FN ("finite only") variant has no
    # infinities and a single NaN encoding; the plain E4M3 has infinities and
    # several NaN encodings, and is not what the cube implements.
    return mlir_ir.Float8E4M3FNType.get()


def _mlir_f8e5m2() -> mlir_ir.Type:
    return mlir_ir.Float8E5M2Type.get()


# The cube element formats MLIR 19 has no builtin for -- Float4E2M1FN and
# Float8E8M0FNU both land in LLVM 20. Until then the Tla dialect spells them
# itself, so these go through the type bridge rather than mlir_ir.
def _mlir_f4e2m1() -> mlir_ir.Type:
    from .._tla_type_bridge import float4_e2m1_type_get

    return float4_e2m1_type_get(mlir_ir.Context.current)


def _mlir_f4e1m2() -> mlir_ir.Type:
    from .._tla_type_bridge import float4_e1m2_type_get

    return float4_e1m2_type_get(mlir_ir.Context.current)


def _mlir_f8e8m0() -> mlir_ir.Type:
    from .._tla_type_bridge import float8_e8m0_type_get

    return float8_e8m0_type_get(mlir_ir.Context.current)


class Numeric(metaclass=NumericMeta, is_abstract=True):
    """Numeric type and value (``DslType`` / ``NumericMeta`` first-class).

    - Type tag: ``dtype=tla.Float16``; class attr ``dtype`` is the TLA token string.
    - Value: ``tla.Int32(5)`` / ``tensor[i]`` → concrete Numeric; arithmetic via operators.
    """

    dtype: ClassVar[str] = ""
    width: ClassVar[int] = 0
    signed: ClassVar[bool] = True

    def __new__(cls, value: Any = None, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        if cls.is_abstract:
            raise TypeError(
                f"{cls.__name__} is abstract; use a concrete type (e.g. Int32, Float32)"
            )
        if value is None:
            raise TypeError(f"{cls.__name__}() missing required argument: 'value'")
        return object.__new__(cls)

    def __init__(
        self,
        value: Any,
        *,
        loc: mlir_ir.Location | None = None,
    ) -> None:
        # Cross-type conversion: host via Python; SSA via ``.to`` (emits arith ops).
        # Entry is construction / ``cast``, not a bare SSA unwrap.
        cls = type(self)
        if isinstance(value, Numeric):
            if type(value) is cls:
                value = value.value
            elif isinstance(value.value, (bool, int, float)):
                value = float(value.value) if cls.is_float else int(value.value)
            else:
                converted = value.to(cls, loc=loc)
                self.value = converted.value
                self.__tla_category__ = "numeric"
                if isinstance(self.value, mlir_ir.Value):
                    _bind_frontend_value(self, self.value)
                    _bind_frontend_category(self, "numeric")
                    _bind_frontend_category(self.value, "numeric")
                return
        if isinstance(value, mlir_ir.Value):
            expected = cls.mlir_type()
            if str(value.type) != str(expected):
                # MLIR ``index`` is not a user Numeric; cast at construction
                # into the requested signed integer type.
                if isinstance(value.type, mlir_ir.IndexType):
                    if not (cls.is_integer and cls.signed):
                        raise TypeError(
                            f"unsupported cast from MLIR index to {cls.__name__}"
                        )
                    value = mlir_ir.Operation.create(
                        "arith.index_cast",
                        operands=[value],
                        results=[expected],
                        loc=loc,
                    ).results[0]
                else:
                    src_cls = Numeric.from_mlir_type(value.type)
                    value = src_cls(value).to(cls, loc=loc).value
            self.value: Any = value
            self.__tla_category__ = "numeric"
            _bind_frontend_value(self, value)
            _bind_frontend_category(self, "numeric")
            _bind_frontend_category(value, "numeric")
        elif isinstance(value, (bool, int, float)):
            self.value = value
            self.__tla_category__ = "numeric"
        else:
            raise TypeError(
                f"{type(self).__name__} expects bool|int|float|mlir.ir.Value|Numeric, "
                f"got {type(value).__name__}"
            )

    @classmethod
    def from_dtype_token(cls, token: str) -> type[Numeric]:
        """Resolve a TLA element token (e.g. ``\"i32\"``) to a concrete Numeric class."""
        key = token.strip().lower()
        try:
            return _TOKEN_TO_NUMERIC[key]
        except KeyError as exc:
            raise TypeError(f"unsupported Tla element type token: {token!r}") from exc

    @classmethod
    def from_mlir_type(cls, ty: mlir_ir.Type) -> type[Numeric]:
        # User-facing Numerics do not include MLIR ``index``; map to Int32.
        if isinstance(ty, mlir_ir.IndexType):
            token = "i32"
        elif isinstance(ty, mlir_ir.F16Type):
            token = "f16"
        elif isinstance(ty, mlir_ir.BF16Type):
            token = "bf16"
        elif isinstance(ty, mlir_ir.F32Type):
            token = "f32"
        elif isinstance(ty, mlir_ir.Float8E4M3FNType):
            token = "f8e4m3fn"
        elif isinstance(ty, mlir_ir.Float8E5M2Type):
            token = "f8e5m2"
        elif str(ty) in ("!tla.f4e2m1", "!tla.f4e1m2", "!tla.f8e8m0"):
            # Dialect types, so there is no mlir_ir class to isinstance against;
            # the printed mnemonic is the token.
            token = str(ty).removeprefix("!tla.")
        elif isinstance(ty, mlir_ir.F64Type):
            raise TypeError(f"unsupported element type for Numeric: {ty!r}")
        elif isinstance(ty, mlir_ir.IntegerType):
            int_type = mlir_ir.IntegerType(ty)
            prefix = "u" if int_type.is_unsigned else "i"
            token = f"{prefix}{int_type.width}"
        else:
            raise TypeError(f"unsupported element type for Numeric: {ty!r}")
        try:
            return _TOKEN_TO_NUMERIC[token]
        except KeyError as exc:
            raise TypeError(f"unsupported element type for Numeric: {ty!r}") from exc

    def ir_value(self, *, loc: mlir_ir.Location | None = None) -> mlir_ir.Value:
        if isinstance(self.value, mlir_ir.Value):
            return self.value
        cls = type(self)
        element_type = cls.mlir_type()
        if cls.is_float:
            return mlir_ir.Operation.create(
                "arith.constant",
                results=[element_type],
                attributes={
                    "value": mlir_ir.FloatAttr.get(element_type, float(self.value))
                },
                loc=loc,
            ).results[0]
        if isinstance(self.value, float) and not float(self.value).is_integer():
            raise TypeError(
                f"expected integer literal for {cls.__name__}, got {self.value!r}"
            )
        return mlir_ir.Operation.create(
            "arith.constant",
            results=[element_type],
            attributes={
                "value": mlir_ir.IntegerAttr.get(element_type, int(self.value))
            },
            loc=loc,
        ).results[0]

    @dsl_user_op
    def to(
        self,
        dtype: Any,
        *,
        round_mode: Any = None,
        loc: mlir_ir.Location | None = None,
    ) -> Any:
        """Convert to another type.

        ``round_mode`` applies only to a float32 -> int32 conversion and names
        one of the scalar unit's four rounding instructions
        (``ScalarRoundMode.NEAREST_EVEN`` / ``NEAREST_AWAY`` / ``FLOOR`` /
        ``CEIL``). Omitted -- or ``TRUNC`` -- keeps ``arith.fptosi``, which
        truncates toward zero. This is not :class:`RoundMode`, which configures
        the AVE vector cast.
        """
        if round_mode is not None:
            _validate_scalar_round_mode(self, dtype, round_mode)
        if dtype is int:
            if isinstance(self.value, (int, float, bool)):
                return int(self.value)
            raise TypeError(f"{type(self).__name__} SSA cannot convert to Python int")
        if dtype is float:
            if isinstance(self.value, (int, float, bool)):
                return float(self.value)
            raise TypeError(f"{type(self).__name__} SSA cannot convert to Python float")
        if dtype is bool:
            if isinstance(self.value, (int, float, bool)):
                return bool(self.value)
            raise TypeError(f"{type(self).__name__} SSA cannot convert to Python bool")
        if dtype is mlir_ir.Value:
            return self.ir_value(loc=loc)
        if not isinstance(dtype, NumericMeta):
            raise TypeError(f"unsupported to() target: {dtype!r}")
        if dtype.is_abstract:
            raise TypeError(
                f"can't convert to abstract {dtype.__name__}; "
                "use a concrete type (e.g. Int32, Float32)"
            )
        if dtype is type(self):
            return self
        # Host: construct target (Python cast in __init__). SSA: emit arith ops.
        if isinstance(self.value, (bool, int, float)):
            if round_mode is not None and type(self).is_float:
                # Fold with the SAME rounding the instruction would apply, so a
                # literal and a runtime value do not disagree.
                from ..params import ScalarRoundMode

                if round_mode is ScalarRoundMode.ODD:
                    return dtype(_round_f32_to_f16_odd(float(self.value)))
                if dtype.is_integer:
                    return dtype(_round_host_float(float(self.value), round_mode))
            return dtype(self, loc=loc)
        src = self.ir_value(loc=loc)
        dst_ty = dtype.mlir_type()
        src_ty = type(self)
        if str(src.type) == str(dst_ty):
            return dtype(src)
        if src_ty.is_integer and dtype.is_integer:
            if dtype.width == src_ty.width:
                return self.bitcast(dtype, loc=loc)
            cast_name = (
                ("arith.extui" if not src_ty.signed else "arith.extsi")
                if dtype.width > src_ty.width
                else "arith.trunci"
            )
            result = _emit_scalar_cast(cast_name, src, dst_ty, loc)
            return dtype(result)
        if src_ty.is_float and dtype.is_float:
            mode = _scalar_round_cast_mode_table().get(round_mode)
            if mode is not None:
                result = mlir_ir.Operation.create(
                    "tla.scalar_round_cast",
                    operands=[src],
                    results=[dst_ty],
                    attributes={"mode": mlir_ir.StringAttr.get(mode)},
                    loc=loc,
                ).results[0]
                return dtype(result)
            cast_name = "arith.extf" if dtype.width > src_ty.width else "arith.truncf"
            result = _emit_scalar_cast(cast_name, src, dst_ty, loc)
            return dtype(result)
        if src_ty.is_integer and dtype.is_float:
            cast_name = "arith.uitofp" if not src_ty.signed else "arith.sitofp"
            result = _emit_scalar_cast(cast_name, src, dst_ty, loc)
            return dtype(result)
        if src_ty.is_float and dtype.is_integer:
            mode = _scalar_round_cast_mode_table().get(round_mode)
            if mode is not None:
                result = mlir_ir.Operation.create(
                    "tla.scalar_round_cast",
                    operands=[src],
                    results=[dst_ty],
                    attributes={"mode": mlir_ir.StringAttr.get(mode)},
                    loc=loc,
                ).results[0]
                return dtype(result)
            cast_name = "arith.fptoui" if not dtype.signed else "arith.fptosi"
            result = _emit_scalar_cast(cast_name, src, dst_ty, loc)
            return dtype(result)
        raise TypeError(
            f"unsupported Numeric.to({dtype.__name__}) from {type(self).__name__}"
        )

    @dsl_user_op
    def bitcast(
        self,
        dtype: type["Numeric"],
        *,
        loc: mlir_ir.Location | None = None,
    ) -> "Numeric":
        if not isinstance(dtype, NumericMeta) or dtype.is_abstract:
            raise TypeError(
                f"bitcast dtype must be a concrete Numeric type, got {dtype!r}"
            )
        if dtype is type(self):
            return self
        if dtype.width != type(self).width:
            raise ValueError(
                f"bitcast requires same bit width: {type(self).__name__}({type(self).width}) "
                f"vs {dtype.__name__}({dtype.width})"
            )
        src = self.ir_value(loc=loc)
        result = mlir_ir.Operation.create(
            "arith.bitcast",
            operands=[src],
            results=[dtype.mlir_type()],
            loc=loc,
        ).results[0]
        return dtype(result)

    def __repr__(self) -> str:
        if isinstance(self.value, mlir_ir.Value):
            return f"{type(self).__name__}(<{self.value.type}>)"
        return f"{type(self).__name__}({self.value!r})"

    def __int__(self) -> int:
        if isinstance(self.value, (int, float, bool)):
            return int(self.value)
        raise TypeError(f"{type(self).__name__} SSA cannot be converted to Python int")

    def __index__(self) -> int:
        if isinstance(self.value, int) and not isinstance(self.value, bool):
            return int(self.value)
        raise TypeError(f"{type(self).__name__} SSA cannot be used as a Python index")

    def __bool__(self) -> bool:
        # ``__bool__``: static host values only.
        if isinstance(self.value, (int, float, bool)):
            return bool(self.value)
        raise TypeError(
            f"{type(self).__name__} SSA cannot be converted to Python bool; "
            "use explicit comparison in the DSL"
        )

    def __get_mlir_types__(
        self, context: mlir_ir.Context | None = None
    ) -> list[mlir_ir.Type]:
        ctx = context
        if ctx is None:
            ctx = mlir_ir.Context.current
        with ctx:
            return [type(self).mlir_type()]

    def __c_pointers__(self) -> list[int]:
        # Base stub; concrete packing lives on IntegerMeta / Float* subclasses.
        raise ValueError(
            "only support built-in types: bool, (u)int{8, 16, 32, 64}, "
            f"float{{16, 32, 64}}, bf16, index; got {type(self).__name__}"
        )

    __add__ = _binary_method("Numeric", "__add__", operator.add)
    __radd__ = _binary_method("Numeric", "__radd__", operator.add)
    __sub__ = _binary_method("Numeric", "__sub__", operator.sub)
    __rsub__ = _binary_method("Numeric", "__rsub__", operator.sub, flip=True)
    __mul__ = _binary_method("Numeric", "__mul__", operator.mul)
    __rmul__ = _binary_method("Numeric", "__rmul__", operator.mul)
    __truediv__ = _binary_method("Numeric", "__truediv__", operator.truediv)
    __rtruediv__ = _binary_method(
        "Numeric", "__rtruediv__", operator.truediv, flip=True
    )
    __floordiv__ = _binary_method("Numeric", "__floordiv__", operator.floordiv)
    __rfloordiv__ = _binary_method(
        "Numeric", "__rfloordiv__", operator.floordiv, flip=True
    )
    __mod__ = _binary_method("Numeric", "__mod__", operator.mod)
    __rmod__ = _binary_method("Numeric", "__rmod__", operator.mod, flip=True)

    @dsl_user_op
    def __neg__(self, *, loc: mlir_ir.Location | None = None) -> "Numeric":
        if isinstance(self.value, (int, float, bool)):
            return type(self)(-self.value)
        if _current_frontend_state() is None:
            raise RuntimeError("Numeric.__neg__ on SSA requires frontend context")
        v = self.ir_value(loc=loc)
        if type(self).is_float:
            result = mlir_ir.Operation.create(
                "arith.negf", operands=[v], results=[v.type], loc=loc
            ).results[0]
        else:
            zero = type(self)(0).ir_value(loc=loc)
            result = mlir_ir.Operation.create(
                "arith.subi", operands=[zero, v], results=[v.type], loc=loc
            ).results[0]
        return type(self)(result)

    @dsl_user_op
    def __abs__(self, *, loc: mlir_ir.Location | None = None) -> "Numeric":
        if isinstance(self.value, (int, float, bool)):
            return type(self)(abs(self.value))
        if _current_frontend_state() is None:
            raise RuntimeError("Numeric.__abs__ on SSA requires frontend context")
        v = self.ir_value(loc=loc)
        if type(self).is_float:
            from ..runtime import _in_simt_vec_func

            name = "tla.simt_abs" if _in_simt_vec_func() else "math.absf"
            result = mlir_ir.Operation.create(
                name, operands=[v], results=[v.type], loc=loc
            ).results[0]
        else:
            result = mlir_ir.Operation.create(
                "math.absi", operands=[v], results=[v.type], loc=loc
            ).results[0]
        return type(self)(result)

    # Type mismatches must raise here; returning NotImplemented would let Python
    # silently fall back to identity comparison for ``==`` and ``!=``.
    __eq__ = _binary_method("Numeric", "__eq__", operator.eq, comparison=True)
    __ne__ = _binary_method("Numeric", "__ne__", operator.ne, comparison=True)
    __lt__ = _binary_method("Numeric", "__lt__", operator.lt, comparison=True)
    __le__ = _binary_method("Numeric", "__le__", operator.le, comparison=True)
    __gt__ = _binary_method("Numeric", "__gt__", operator.gt, comparison=True)
    __ge__ = _binary_method("Numeric", "__ge__", operator.ge, comparison=True)
    __pow__ = _binary_method("Numeric", "__pow__", operator.pow)
    __rpow__ = _binary_method("Numeric", "__rpow__", operator.pow, flip=True)

    @staticmethod
    def _from_python_value(value: Any) -> "Numeric":
        """Map Python / MLIR scalar to a concrete Numeric."""
        if isinstance(value, Numeric):
            return value
        if isinstance(value, bool):
            return Bool(value)
        if isinstance(value, int):
            if (value <= 2147483647) and (value >= -2147483648):
                return Int32(value)
            return Int64(value)
        if isinstance(value, float):
            return Float32(value)
        if isinstance(value, mlir_ir.Value):
            # Ascend IR may still emit MLIR ``index`` (layout leaves).
            # ``from_mlir_type`` maps index → Int32 and ``Numeric.__init__``
            # emits ``arith.index_cast``.
            return Numeric.from_mlir_type(value.type)(value)
        raise ValueError(f"unable to convert {value} in type {type(value)} to Numeric")


def as_numeric(obj: Any) -> Numeric:
    """Convert a Python primitive or MLIR value to a Numeric."""
    return Numeric._from_python_value(obj)


class Integer(
    Numeric, metaclass=IntegerMeta, width=32, mlir_type=_mlir_i(32), is_abstract=True
):
    """Abstract integer numeric family."""

    @dsl_user_op
    def __invert__(self, *, loc: mlir_ir.Location | None = None) -> "Integer":
        cls = typing_cast(type[Integer], type(self))
        if isinstance(self.value, (int, bool)):
            return cls(~int(self.value))
        if _current_frontend_state() is None:
            raise RuntimeError("Integer.__invert__ on SSA requires frontend context")
        v = self.ir_value(loc=loc)
        ones = cls(-1).ir_value(loc=loc)
        result = mlir_ir.Operation.create(
            "arith.xori", operands=[v, ones], results=[v.type], loc=loc
        ).results[0]
        return cls(result)

    __lshift__ = _binary_method("Integer", "__lshift__", operator.lshift)
    __rlshift__ = _reflected_shift_method("__rlshift__", "__lshift__", "left")
    __rshift__ = _binary_method("Integer", "__rshift__", operator.rshift)
    __rrshift__ = _reflected_shift_method("__rrshift__", "__rshift__", "right")
    __and__ = _binary_method("Integer", "__and__", operator.and_)
    __rand__ = _reflected_method("Integer", "__rand__", "__and__")
    __or__ = _binary_method("Integer", "__or__", operator.or_)
    __ror__ = _reflected_method("Integer", "__ror__", "__or__")
    __xor__ = _binary_method("Integer", "__xor__", operator.xor)
    __rxor__ = _reflected_method("Integer", "__rxor__", "__xor__")


class Float(
    Numeric, metaclass=FloatMeta, width=32, mlir_type=_mlir_f32, is_abstract=True
):
    """Abstract floating-point numeric family."""

    _host_bits: ClassVar[Callable[[float], int]]

    @dsl_user_op
    def sqrt(self, *, loc: mlir_ir.Location | None = None) -> "Float":
        """Square root of a scalar float.

        The scalar counterpart of ``tla.sqrt``, dispatched the same way
        ``__abs__`` is: inside a ``tla.vec.func(mode="simt")`` this is the
        per-thread ``tla.simt_sqrt``, and everywhere else -- a cube region, a
        SIMD vector region, the kernel body -- it is ``math.sqrt`` on the
        core's own scalar unit.
        """
        if isinstance(self.value, (int, float, bool)):
            return type(self)(math.sqrt(float(self.value)))
        if _current_frontend_state() is None:
            raise RuntimeError("Float.sqrt on SSA requires frontend context")
        from ..runtime import _in_simt_vec_func

        in_simt = _in_simt_vec_func()
        # Outside a SIMT region this is math.sqrt, which the backend selects
        # only for f32 -- f16 and bf16 come back as "Cannot select: fsqrt" from
        # inside the compiler. Reject them here so the message names the kernel.
        if not in_simt and type(self).width != 32:
            raise TypeError(
                f"sqrt on a scalar outside a SIMT region requires Float32; got "
                f"{type(self).__name__}. Convert first with .to(tla.Float32), "
                "or use a tla.vec.func(mode='simt') region"
            )
        v = self.ir_value(loc=loc)
        name = "tla.simt_sqrt" if in_simt else "math.sqrt"
        result = mlir_ir.Operation.create(
            name, operands=[v], results=[v.type], loc=loc
        ).results[0]
        return type(self)(result)

    def __c_pointers__(self) -> list[int]:
        cls = type(self)
        if isinstance(self.value, mlir_ir.Value):
            raise TypeError(
                f"{cls.__name__} SSA value cannot provide host __c_pointers__"
            )
        return [cls._host_bits(float(self.value))]


class Bool(
    Integer,
    metaclass=IntegerMeta,
    width=1,
    dtype="i1",
    signed=False,
    mlir_type=_mlir_i(1),
):
    # ``signed=False``: a Bool is a 0/1 truth value. Widening must zero-extend so
    # ``Bool(True).to(Int8)`` == 1 (``arith.extui``), not -1 (``arith.extsi``);
    # unsigned targets like UInt8/UInt16 are also reachable this way.
    pass


class Int8(
    Integer,
    metaclass=IntegerMeta,
    width=8,
    dtype="i8",
    signed=True,
    mlir_type=_mlir_i(8),
):
    pass


class Int16(
    Integer,
    metaclass=IntegerMeta,
    width=16,
    dtype="i16",
    signed=True,
    mlir_type=_mlir_i(16),
):
    pass


class Int32(
    Integer,
    metaclass=IntegerMeta,
    width=32,
    dtype="i32",
    signed=True,
    mlir_type=_mlir_i(32),
):
    pass


class Int64(
    Integer,
    metaclass=IntegerMeta,
    width=64,
    dtype="i64",
    signed=True,
    mlir_type=_mlir_i(64),
):
    pass


class UInt8(
    Integer,
    metaclass=IntegerMeta,
    width=8,
    dtype="u8",
    signed=False,
    mlir_type=_mlir_i(8, unsigned=True),
):
    pass


class UInt16(
    Integer,
    metaclass=IntegerMeta,
    width=16,
    dtype="u16",
    signed=False,
    mlir_type=_mlir_i(16, unsigned=True),
):
    pass


class UInt32(
    Integer,
    metaclass=IntegerMeta,
    width=32,
    dtype="u32",
    signed=False,
    mlir_type=_mlir_i(32, unsigned=True),
):
    pass


class UInt64(
    Integer,
    metaclass=IntegerMeta,
    width=64,
    dtype="u64",
    signed=False,
    mlir_type=_mlir_i(64, unsigned=True),
):
    pass


class Float16(Float, metaclass=FloatMeta, width=16, dtype="f16", mlir_type=_mlir_f16):
    @staticmethod
    def _host_bits(value: float) -> int:
        f16_val = np.float16(value)
        return int(f16_val.view(np.uint16))


class BFloat16(
    Float, metaclass=FloatMeta, width=16, dtype="bf16", mlir_type=_mlir_bf16
):
    @staticmethod
    def _host_bits(value: float) -> int:
        bits = int(np.float32(value).view(np.uint32))
        # Round-to-nearest-even when truncating f32 → bf16 (not plain >> 16).
        if (bits & 0x7F800000) != 0x7F800000:
            bits = (bits + 0x00007FFF + ((bits >> 16) & 1)) & 0xFFFFFFFF
        return int(np.uint16(bits >> 16))


class Float32(Float, metaclass=FloatMeta, width=32, dtype="f32", mlir_type=_mlir_f32):
    @staticmethod
    def _host_bits(value: float) -> int:
        return struct.unpack("I", struct.pack("f", float(value)))[0]


class Float8E4M3FN(
    Float, metaclass=FloatMeta, width=8, dtype="f8e4m3fn", mlir_type=_mlir_f8e4m3fn
):
    """OCP E4M3 with finite-only encoding, the cube's 8-bit float operand format.

    Named for MLIR's ``Float8E4M3FN``, which is the type this maps to. MLIR also
    has a plain ``Float8E4M3`` -- a *different* format, with infinities and
    several NaN encodings -- that the cube does not implement, so the ``FN``
    suffix is load-bearing rather than decoration.
    """


class Float8E5M2(
    Float, metaclass=FloatMeta, width=8, dtype="f8e5m2", mlir_type=_mlir_f8e5m2
):
    """OCP E5M2, the wider-exponent 8-bit float operand format."""


class Float4E2M1(
    Float, metaclass=FloatMeta, width=4, dtype="f4e2m1", mlir_type=_mlir_f4e2m1
):
    """OCP MXFP4: packed 4-bit float, 2-bit exponent / 1-bit mantissa.

    Storage stays byte-addressable -- a tile of these is buffered as i8 with
    half as many elements -- but that is a lowering detail, not something a
    kernel says. What the kernel writes is the format, and the format is what
    selects the cube's mad_mx variant.

    Microscaling-only: the cube has no plain fp4 route, so a tile of these must
    be loaded by ``tla.copy(..., scale=...)`` and matmul'd with ``tla.mmad_mx``.
    """


class Float4E1M2(
    Float, metaclass=FloatMeta, width=4, dtype="f4e1m2", mlir_type=_mlir_f4e1m2
):
    """Packed 4-bit float, 1-bit exponent / 2-bit mantissa.

    The same width as [`Float4E2M1`] and a different format; the two are told
    apart by type, and may be mixed across the two matmul operands.
    """


class Float8E8M0(
    Float, metaclass=FloatMeta, width=8, dtype="f8e8m0", mlir_type=_mlir_f8e8m0
):
    """OCP E8M0 shared exponent -- the element of a microscaling scale block.

    One per 32 elements along K. No arithmetic is ever performed on it here: it
    is handed to the cube, which applies it.
    """


class Constexpr(Generic[_T]):
    """Annotation marker for compile-time-only kernel / dataclass fields.

    Written as ``tla.Constexpr[T]``. Semantics live in the Python syntax guide
    and the ``kernel`` / ``dataclass`` Host API type tables — not a callable
    Host entry.
    """

    @classmethod
    def is_constexpr_annotation(cls, annotation: Any) -> bool:
        if annotation is cls:
            return True
        origin = get_origin(annotation)
        return origin is cls


def is_constexpr_annotation(annotation: Any) -> bool:
    """True for ``Constexpr`` / ``tla.Constexpr`` markers, including string forms.

    String annotations (``from __future__ import annotations``) like
    ``"tla.Constexpr[int]"`` are matched textually since they cannot be resolved
    without module globals here. Any qualifier is accepted — ``Constexpr``,
    ``tla.Constexpr``, ``catlass.tla.Constexpr``, or whatever alias the module
    imported it under — because an unrecognized qualifier would silently demote
    the parameter to a runtime argument.
    """
    if Constexpr.is_constexpr_annotation(annotation):
        return True
    if isinstance(annotation, str):
        compact = annotation.replace(" ", "")
        head = compact.split("[", 1)[0]
        return head == "Constexpr" or head.endswith(".Constexpr")
    return False


@runtime_checkable
class JitArgument(Protocol):
    """Protocol for objects that can be passed to JIT-compiled Tla functions."""

    def __c_pointers__(self) -> list[int]:
        raise NotImplementedError

    def __get_mlir_types__(self, context: Any | None = None) -> list[Any]:
        raise NotImplementedError

    def __extract_mlir_values__(self) -> list[Any]:
        raise NotImplementedError

    def __new_from_mlir_values__(self, values: list[Any]) -> Any:
        raise NotImplementedError


@dataclass(frozen=True)
class TypedPointer:
    """Type descriptor for a pointer element type and memory space.

    ``Pointer[dtype, space]`` returns a ``TypedPointer`` for APIs that need a
    compile-time pointer type, such as an external function ABI. It is not a
    runtime or SSA pointer value.

    Args:
        dtype: Concrete :class:`Numeric` element type.
        space: Target memory space as a
            :class:`~catlass.address_space.AddressSpace` member.
    """

    dtype: type[Numeric]
    space: AddressSpace

    def __post_init__(self) -> None:
        # Validate `dtype` and `space`
        if (
            not isinstance(self.dtype, type)
            or not issubclass(self.dtype, Numeric)
            or not self.dtype.dtype
        ):
            raise TypeError(
                "TypedPointer dtype must be a concrete Numeric type, "
                f"got {self.dtype!r}"
            )
        if not isinstance(self.space, AddressSpace):
            raise TypeError(
                f"TypedPointer memory space must be AddressSpace, got {self.space!r}"
            )

    def __repr__(self) -> str:
        return f"TypedPointer[{self.dtype}, {self.space}]"


class Pointer(ABC):
    """Abstract JIT pointer (typed ``Pointer`` protocol)."""

    def __class_getitem__(cls, args: Any) -> TypedPointer:
        if not isinstance(args, tuple) or len(args) != 2:
            raise TypeError("Pointer[...] expects (dtype, memory_space)")
        return TypedPointer(*args)

    @property
    @abstractmethod
    def dtype(self) -> Any: ...

    @property
    def value_type(self) -> Any:
        return self.dtype

    @abstractmethod
    def __get_mlir_types__(self, context: Any | None = None) -> list[Any]: ...

    @abstractmethod
    def __extract_mlir_values__(self) -> list[Any]: ...

    @abstractmethod
    def __new_from_mlir_values__(self, values: list[Any]) -> Pointer: ...


__all__ = [
    "DslType",
    "NumericMeta",
    "IntegerMeta",
    "FloatMeta",
    "Numeric",
    "Integer",
    "Float",
    "cast",
    "as_numeric",
    "Bool",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
    "Float16",
    "BFloat16",
    "Float32",
    "Pointer",
    "TypedPointer",
    "Constexpr",
    "JitArgument",
]
