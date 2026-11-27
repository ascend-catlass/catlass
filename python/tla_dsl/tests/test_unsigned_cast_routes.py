from __future__ import annotations

import pytest

import catlass.tla as tla
from catlass.tla.runtime import make_fake_tensor

# The vector unit's unsigned cast routes, checked on the host: which pairs the
# frontend admits, which it refuses and why, and that reg_slot stays honest on
# the 4x steps.
#
# The reachable unsigned set is deliberately not the signed width lattice. The
# AVE op definitions advertise more than HIVMAVEToAVEIntrin implements -- vfptoui
# lists i8/i16/i32/i64 results but only builds f16->u8, vuitofp's summary claims
# u64->f32 but only u8->f16 exists, and vextui has no U32->U64 arm although
# vextsi has S32->S64. A route missing there is not a verification failure: it
# reaches llvm_unreachable in the backend. So the frontend gates the routes, and
# these tests pin that gate.

_LANES = {
    tla.UInt8: 256,
    tla.UInt16: 128,
    tla.UInt32: 64,
    tla.UInt64: 32,
    tla.Int8: 256,
    tla.Int16: 128,
    tla.Int32: 64,
    tla.Float16: 128,
    tla.Float32: 64,
    tla.Float8E4M3FN: 256,
}

_INTEGER_NARROWS = (
    (tla.UInt16, tla.UInt8),
    (tla.UInt32, tla.UInt16),
    (tla.UInt32, tla.UInt8),
    (tla.Int16, tla.UInt8),
    (tla.Int32, tla.UInt8),
    (tla.Int32, tla.UInt16),
    (tla.UInt32, tla.Int16),
)

_UNSIGNED_WIDENS = (
    (tla.UInt8, tla.UInt16),
    (tla.UInt16, tla.UInt32),
    (tla.UInt8, tla.UInt32),
    (tla.UInt8, tla.Float16),
)
_UNSIGNED_NARROWS = (*_INTEGER_NARROWS, (tla.Float16, tla.UInt8))

# The 4x integer steps, where reg_slot names a pack quarter rather than an
# even/odd part, so all four slots must be accepted.
_FOUR_X = (
    (tla.UInt8, tla.UInt32),
    (tla.UInt32, tla.UInt8),
    (tla.Int32, tla.UInt8),
)

_SAT_MODES = (
    tla.params.SatMode.NOSAT,
    tla.params.SatMode.SAT,
)
_ROUND_MODES = (
    tla.params.RoundMode.CAST_ROUND,
    tla.params.RoundMode.CAST_FLOOR,
    tla.params.RoundMode.CAST_CEIL,
    tla.params.RoundMode.CAST_TRUNC,
)
_DEFAULT = tla.params.CastParams()
_NOSAT = tla.params.CastParams(sat_mode=tla.params.SatMode.NOSAT)


def _ub(dtype: type[tla.Numeric]) -> tla.Tensor:
    n = _LANES[dtype]
    return make_fake_tensor(
        dtype,
        (n,),
        (1,),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(n,),
        layout_tag=tla.arch.RowMajor,
    )


def _cast_kernel(src_dtype, dst_dtype, params):
    # Each side is tiled at its own lane count: a register is 256 bytes, so a u8
    # tile holds 256 lanes where a u32 one holds 64.
    src_lanes = _LANES[src_dtype]
    dst_lanes = _LANES[dst_dtype]

    @tla.kernel
    def k(src: tla.Tensor, dst: tla.Tensor) -> None:
        src_t = tla.tile_view(src, tla.make_shape(src_lanes), tla.make_coord(0))
        dst_t = tla.tile_view(dst, tla.make_shape(dst_lanes), tla.make_coord(0))
        with tla.vector():
            with tla.vec.func(mode="simd"):
                dst_t.store(src_t.load().to(dst_dtype, params))

    return k


def _compile(src_dtype, dst_dtype, params) -> None:
    _cast_kernel(src_dtype, dst_dtype, params).dump_mlir(
        type_args=(_ub(src_dtype), _ub(dst_dtype))
    )


@pytest.mark.parametrize(("src", "dst"), (*_UNSIGNED_NARROWS, *_UNSIGNED_WIDENS))
def test_every_supported_unsigned_route_compiles(src, dst) -> None:
    params = _NOSAT if (src, dst) in _UNSIGNED_NARROWS else _DEFAULT
    _compile(src, dst, params)


@pytest.mark.parametrize(
    ("src", "dst", "sat", "rnd"),
    tuple(
        (src, dst, sat, rnd)
        for src, dst in _UNSIGNED_NARROWS
        for sat in _SAT_MODES
        for rnd in (
            (tla.params.RoundMode.CAST_NONE,)
            if (src, dst) in _INTEGER_NARROWS
            else _ROUND_MODES
        )
    ),
)
def test_unsigned_route_accepts_supported_cast_modes(src, dst, sat, rnd) -> None:
    _compile(src, dst, tla.params.CastParams(sat_mode=sat, round_mode=rnd))


@pytest.mark.parametrize(("src", "dst"), _FOUR_X)
@pytest.mark.parametrize(
    "slot", (tla.params.RegSlot.ONE, tla.params.RegSlot.TWO, tla.params.RegSlot.THREE)
)
def test_nondefault_pack_quarters_are_valid_on_a_4x_unsigned_cast(
    src, dst, slot
) -> None:
    """A 4x step reads pp0..pp3, so TWO and THREE must survive the frontend."""
    sat = tla.params.SatMode.UNKNOWN if src is tla.UInt8 else tla.params.SatMode.NOSAT
    _compile(src, dst, tla.params.CastParams(reg_slot=slot, sat_mode=sat))


@pytest.mark.parametrize("slot", tuple(tla.params.RegSlot))
def test_uint8_to_float32_accepts_all_pack_quarters(slot) -> None:
    _compile(
        tla.UInt8,
        tla.Float32,
        tla.params.CastParams(reg_slot=slot),
    )


@pytest.mark.parametrize("slot", (tla.params.RegSlot.ZERO, tla.params.RegSlot.ONE))
def test_int16_to_uint32_accepts_even_and_odd_slots(slot) -> None:
    _compile(
        tla.Int16,
        tla.UInt32,
        tla.params.CastParams(reg_slot=slot),
    )


@pytest.mark.parametrize(
    ("src", "dst"),
    (
        (tla.UInt8, tla.UInt16),
        (tla.UInt16, tla.UInt32),
        (tla.UInt16, tla.UInt8),
        (tla.UInt32, tla.UInt16),
        (tla.Int16, tla.UInt8),
        (tla.Int16, tla.UInt32),
        (tla.Int32, tla.UInt16),
        (tla.UInt32, tla.Int16),
        (tla.UInt8, tla.Float16),
        (tla.Float16, tla.UInt8),
    ),
)
@pytest.mark.parametrize("slot", (tla.params.RegSlot.TWO, tla.params.RegSlot.THREE))
def test_pack_quarter_slots_are_refused_on_a_2x_unsigned_cast(src, dst, slot) -> None:
    """A 2x step carries an even/odd part, where TWO and THREE would act as ZERO."""
    sat = (
        tla.params.SatMode.NOSAT
        if (src, dst) in _UNSIGNED_NARROWS
        else tla.params.SatMode.UNKNOWN
    )
    with pytest.raises(tla.TlaCoreAPIError, match="pack quarter"):
        _compile(src, dst, tla.params.CastParams(reg_slot=slot, sat_mode=sat))


@pytest.mark.parametrize(
    ("src", "dst"),
    (
        # u64 takes no part in any route, in either direction. A u64 vector
        # cannot be produced: there is no 64-bit tile element type to load one
        # from, and no widen yields one, since vextui has no U32 -> U64 arm
        # (unlike the signed S32 -> S64). vtrunci does have a 64->32 arm, but
        # nothing can feed it an unsigned operand, so u64 -> u32 would advertise
        # a route no caller can invoke.
        (tla.UInt32, tla.UInt64),
        (tla.UInt16, tla.UInt64),
        (tla.UInt8, tla.UInt64),
        (tla.UInt64, tla.UInt32),
        (tla.UInt64, tla.Float32),
        (tla.UInt64, tla.Float16),
        (tla.Float32, tla.UInt64),
        # Only u8 reaches a float. f32 -> u8 is spelled as two casts through f16.
        (tla.UInt16, tla.Float16),
        (tla.UInt32, tla.Float32),
        (tla.UInt16, tla.Float32),
        (tla.Float32, tla.UInt8),
        (tla.Float32, tla.UInt16),
        (tla.Float32, tla.UInt32),
        (tla.Float16, tla.UInt16),
        (tla.Float16, tla.UInt32),
        # No unsigned type pairs with an fp8 format.
        (tla.UInt8, tla.Float8E4M3FN),
        (tla.Float8E4M3FN, tla.UInt8),
        # Mixed signedness routes outside the supported pairs.
        (tla.UInt8, tla.Int32),
        (tla.UInt8, tla.Int16),
        (tla.UInt32, tla.Int8),
        (tla.Int8, tla.UInt32),
        (tla.Int8, tla.UInt16),
        (tla.Int32, tla.UInt32),
        # Unsigned integers have no same-type cast.
        (tla.UInt8, tla.UInt8),
    ),
)
def test_unsupported_unsigned_cast_is_refused(src, dst) -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="complete width lattice"):
        _compile(src, dst, _DEFAULT)


def test_signed_integer_routes_compile() -> None:
    for src, dst in (
        (tla.Int8, tla.Int16),
        (tla.Int16, tla.Int32),
        (tla.Int8, tla.Int32),
        (tla.Int32, tla.Float32),
    ):
        _compile(src, dst, _DEFAULT)
    _compile(tla.Int32, tla.Int8, _NOSAT)
    _compile(tla.Float32, tla.Int32, _NOSAT)


@pytest.mark.parametrize("dtype", (tla.UInt8, tla.UInt16, tla.UInt32))
def test_shift_right_rejects_unsigned_shift_amounts(dtype) -> None:
    lanes = _LANES[dtype]

    @tla.kernel
    def k(src: tla.Tensor, dst: tla.Tensor) -> None:
        src_t = tla.tile_view(src, tla.make_shape(lanes), tla.make_coord(0))
        dst_t = tla.tile_view(dst, tla.make_shape(lanes), tla.make_coord(0))
        with tla.vector():
            with tla.vec.func(mode="simd"):
                reg = src_t.load()
                dst_t.store(tla.shift_right(reg, reg))

    with pytest.raises(tla.TlaCoreAPIError, match="shift vector has element type"):
        k.dump_mlir(type_args=(_ub(dtype), _ub(dtype)))
