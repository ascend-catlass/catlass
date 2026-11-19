from __future__ import annotations

import pytest

import catlass.tla as tla
from catlass.execution_lowering import TlaLoweringError


_DTYPE = tla.Float4E2M1x2
_OTHER_DTYPE = tla.Float4E2M1x2
_LANES = 256
_OTHER_LANES = 256
_DEST_LANES = 256
_OFFSET = 0
_ALIGNMENT = 256
_MODE = "simd"
_OP = "bitwise_or"
_EXPLICIT_MASK = False
_STORE_MASK = False


@tla.jit
def _tile(dtype: object, lanes: int) -> object:
    # Pair counts, not 4-bit elements: one lane is one packed byte.
    ptr = tla.allocate(1024, dtype, tla.AddressSpace.ub, _ALIGNMENT)
    layout = tla.make_layout(
        tla.make_shape(1024),
        tla.make_stride(1),
        origin_shape=tla.make_shape(lanes),
    )
    return tla.make_tensor(ptr, layout, tla.make_coord(_OFFSET))


@tla.kernel
def _kernel() -> None:
    a = _tile(_DTYPE, _LANES)
    b = _tile(_OTHER_DTYPE, _OTHER_LANES)
    dst = _tile(_DTYPE, _DEST_LANES)
    with tla.vector():
        with tla.vec.func(mode=_MODE):
            lhs = a.load()
            rhs = b.load()
            if tla.const_expr(_EXPLICIT_MASK):
                mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Int8)
                if tla.const_expr(_OP == "bitwise_not"):
                    result = tla.bitwise_not(lhs, mask=mask)
                elif tla.const_expr(_OP == "bitwise_and"):
                    result = tla.bitwise_and(lhs, rhs, mask=mask)
                elif tla.const_expr(_OP == "bitwise_xor"):
                    result = tla.bitwise_xor(lhs, rhs, mask=mask)
                else:
                    result = tla.bitwise_or(lhs, rhs, mask=mask)
            elif tla.const_expr(_OP == "bitwise_not"):
                result = tla.bitwise_not(lhs)
            elif tla.const_expr(_OP == "bitwise_and"):
                result = tla.bitwise_and(lhs, rhs)
            elif tla.const_expr(_OP == "bitwise_xor"):
                result = tla.bitwise_xor(lhs, rhs)
            elif tla.const_expr(_OP == "mixed"):
                result = tla.bitwise_not(
                    tla.bitwise_and(tla.bitwise_or(lhs, rhs), tla.bitwise_xor(lhs, rhs))
                )
            else:
                result = tla.bitwise_or(tla.bitwise_or(lhs, rhs), lhs)
            if tla.const_expr(_STORE_MASK):
                mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Int8)
                dst.store(result, mask=mask)
            else:
                dst.store(result)


def _reset(monkeypatch: pytest.MonkeyPatch, **values: object) -> None:
    for name, value in values.items():
        monkeypatch.setattr(f"{__name__}._{name.upper()}", value)


@pytest.mark.parametrize("dtype,spelling", [(tla.Float4E2M1x2, "f4e2m1x2"), (tla.Float4E1M2x2, "f4e1m2x2")])
@pytest.mark.parametrize("lanes", [1, 127, 256])
@pytest.mark.parametrize("offset", [0, 32])
@pytest.mark.parametrize("op", ["bitwise_or", "bitwise_and", "bitwise_xor", "bitwise_not", "mixed"])
def test_fp4x2_bitwise_preserves_pair_lanes(
    monkeypatch: pytest.MonkeyPatch, dtype, spelling, lanes, offset, op
) -> None:
    _reset(
        monkeypatch,
        dtype=dtype,
        other_dtype=dtype,
        lanes=lanes,
        other_lanes=lanes,
        dest_lanes=lanes,
        offset=offset,
        op=op,
    )
    ir = _kernel.dump_mlir(type_args=())
    # One lane is one packed byte, so the vector is N pairs, never 2N nibbles.
    assert f"!tla.vector<{lanes}x!tla.{spelling}>" in ir
    if op == "mixed":
        for name in ("bitwise_or", "bitwise_and", "bitwise_xor", "bitwise_not"):
            assert ir.count(f"tla.{name} ") == 1
    else:
        assert ir.count(f"tla.{op} ") == (2 if op == "bitwise_or" else 1)
    assert "tla.load " in ir
    assert "tla.store " in ir


@pytest.mark.parametrize("op", ["bitwise_or", "bitwise_and", "bitwise_xor", "bitwise_not"])
def test_fp4x2_bitwise_accepts_explicit_mask(monkeypatch: pytest.MonkeyPatch, op) -> None:
    # The predicate is byte-granular, matching the pair lane exactly, so the
    # generic mask path applies and no FP4-specific rejection is left.
    _reset(monkeypatch, op=op, explicit_mask=True, store_mask=True)
    ir = _kernel.dump_mlir(type_args=())
    assert "!tla.mask<256>" in ir
    assert f"tla.{op} " in ir


def test_fp4x2_bitwise_requires_identical_pair_format(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset(monkeypatch, other_dtype=tla.Float4E1M2x2)
    with pytest.raises((tla.TlaCoreAPIError, TlaLoweringError), match="element type"):
        _kernel.dump_mlir(type_args=())


def test_fp4x2_bitwise_not_rejects_non_pair_float(monkeypatch: pytest.MonkeyPatch) -> None:
    # f4e2m1x2 joined the byte-carrier set on purpose; the logical 4-bit formats
    # have no vector form, and f8 stays outside the bitwise whitelist.
    _reset(monkeypatch, dtype=tla.Float8E4M3FN, other_dtype=tla.Float8E4M3FN, op="bitwise_not")
    with pytest.raises((tla.TlaCoreAPIError, TlaLoweringError), match="element type"):
        _kernel.dump_mlir(type_args=())
