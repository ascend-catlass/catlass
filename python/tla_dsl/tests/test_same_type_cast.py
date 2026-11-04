from __future__ import annotations

import pytest

import catlass.tla as tla
from catlass.tla.runtime import make_fake_tensor

# The same-type float casts: f16->f16, bf16->bf16, f32->f32. Each one keeps its
# element type and re-rounds every lane through the AVE vtrc op, so only the
# round mode reaches the instruction. These tests pin the frontend gate: which
# same-type pairs are admitted, that SAT (and any same-type non-float pair) is
# refused with a diagnostic instead of a backend fatal error, and that the
# pack-quarter slots stay honest on an equal-width conversion.
#
# The other same-type pairs are not a matter of taste: the lowering would reach
# HIVMAVEToAVEIntrin with no instruction for them and die with a fatal
# "Unsupported type conversion op." rather than a diagnostic. The frontend
# refuses them first.

_LANES = {
    tla.Float16: 128,
    tla.BFloat16: 128,
    tla.Float32: 64,
    tla.Int32: 64,
    tla.Float8E4M3FN: 256,
}

_SAME_TYPE_FLOATS = (
    tla.Float16,
    tla.BFloat16,
    tla.Float32,
)

_ROUND_MODES = (
    tla.params.RoundMode.CAST_ROUND,
    tla.params.RoundMode.CAST_FLOOR,
    tla.params.RoundMode.CAST_CEIL,
    tla.params.RoundMode.CAST_TRUNC,
    tla.params.RoundMode.CAST_NONE,
    tla.params.RoundMode.CAST_RINT,
    tla.params.RoundMode.CAST_ODD,
)


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


def _cast_kernel(dtype: type[tla.Numeric], params: tla.params.CastParams):
    lanes = _LANES[dtype]

    @tla.kernel
    def k(src: tla.Tensor, dst: tla.Tensor) -> None:
        src_t = tla.tile_view(src, tla.make_shape(lanes), tla.make_coord(0))
        dst_t = tla.tile_view(dst, tla.make_shape(lanes), tla.make_coord(0))
        with tla.vector():
            with tla.vec.func(mode="simd"):
                dst_t.store(src_t.load().to(dtype, params))

    return k


def _compile(dtype: type[tla.Numeric], params: tla.params.CastParams) -> None:
    _cast_kernel(dtype, params).dump_mlir(type_args=(_ub(dtype), _ub(dtype)))


@pytest.mark.parametrize("dtype", _SAME_TYPE_FLOATS)
def test_every_same_type_float_cast_compiles(dtype) -> None:
    _compile(dtype, tla.params.CastParams())


@pytest.mark.parametrize("dtype", _SAME_TYPE_FLOATS)
@pytest.mark.parametrize("rnd", _ROUND_MODES)
def test_every_same_type_float_cast_accepts_every_round_mode(dtype, rnd) -> None:
    """The round mode is the only knob vtrc carries; every mode must be admitted."""
    _compile(dtype, tla.params.CastParams(round_mode=rnd))


@pytest.mark.parametrize("dtype", _SAME_TYPE_FLOATS)
@pytest.mark.parametrize("sat", (tla.params.SatMode.UNKNOWN, tla.params.SatMode.NOSAT))
def test_same_type_float_cast_accepts_unsaturated_sat_modes(dtype, sat) -> None:
    """UNKNOWN and NOSAT both say what vtrc does: no saturation at all."""
    _compile(dtype, tla.params.CastParams(sat_mode=sat))


@pytest.mark.parametrize("dtype", _SAME_TYPE_FLOATS)
@pytest.mark.parametrize("slot", (tla.params.RegSlot.ZERO, tla.params.RegSlot.ONE))
def test_same_type_float_cast_accepts_full_register_slots(dtype, slot) -> None:
    """An equal-width cast reads the whole register, so ZERO and ONE are moot
    but harmless -- same as f16 -> bf16."""
    _compile(dtype, tla.params.CastParams(reg_slot=slot))


@pytest.mark.parametrize("dtype", _SAME_TYPE_FLOATS)
@pytest.mark.parametrize("slot", (tla.params.RegSlot.TWO, tla.params.RegSlot.THREE))
def test_pack_quarter_slots_are_refused_on_a_same_type_cast(dtype, slot) -> None:
    """TWO and THREE name a pack quarter, which an equal-width cast never reads."""
    with pytest.raises(tla.TlaCoreAPIError, match="pack quarter"):
        _compile(dtype, tla.params.CastParams(reg_slot=slot))


@pytest.mark.parametrize("dtype", _SAME_TYPE_FLOATS)
def test_saturating_same_type_cast_is_refused(dtype) -> None:
    """SAT promises clamping that vtrc cannot deliver; refuse it rather than
    silently drop the knob."""
    with pytest.raises(tla.TlaCoreAPIError, match="never saturates"):
        _compile(dtype, tla.params.CastParams(sat_mode=tla.params.SatMode.SAT))


@pytest.mark.parametrize(
    "dtype",
    (
        tla.Int32,
        tla.Float8E4M3FN,
    ),
)
def test_same_type_non_float_cast_is_refused(dtype) -> None:
    """Integers and fp8 have no same-type cast; without this gate they reach
    the backend and die with a fatal error, not a diagnostic."""
    with pytest.raises(tla.TlaCoreAPIError, match="same-type cast is only supported"):
        _compile(dtype, tla.params.CastParams())
