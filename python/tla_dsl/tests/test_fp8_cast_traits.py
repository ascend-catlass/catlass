from __future__ import annotations

import pytest

import catlass.tla as tla
from catlass import execution
from catlass.tla.runtime import make_fake_tensor

# The vector unit's fp8 legs and the CastParams attributes they accept, checked
# on the host: which pairs the frontend admits, which it refuses, and that a
# reg_slot the lowering cannot honour is refused rather than silently demoted.

_LANES = {
    tla.Float8E4M3FN: 256,
    tla.Float8E5M2: 256,
    tla.Float16: 128,
    tla.BFloat16: 128,
    tla.Float32: 64,
    tla.Int32: 64,
    tla.Int8: 256,
}


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


def _cast_kernel(
    src_dtype: type[tla.Numeric],
    dst_dtype: type[tla.Numeric],
    params: tla.params.CastParams,
):
    # Each side is tiled at its own lane count: a register is 256 bytes, so an
    # fp8 tile holds 256 lanes where an f32 one holds 64. Sizing both by the
    # destination silently mismatches the source on every narrowing cast.
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


def _lower(src_dtype, dst_dtype, params) -> None:
    """Compile far enough to run the vector lowering, which is what refuses."""
    _cast_kernel(src_dtype, dst_dtype, params).compile(
        type_args=(_ub(src_dtype), _ub(dst_dtype)), options="--npu-arch 3510"
    )


def _compile(src_dtype, dst_dtype, params) -> None:
    _cast_kernel(src_dtype, dst_dtype, params).dump_mlir(
        type_args=(_ub(src_dtype), _ub(dst_dtype))
    )


_DEFAULT = tla.params.CastParams()
_FP8 = (tla.Float8E4M3FN, tla.Float8E5M2)
_PARTNERS = (tla.Float32, tla.Float16, tla.BFloat16)
_FLOAT_SLOTS = (tla.params.RegSlot.ZERO, tla.params.RegSlot.ONE)
_FP8_F32_SLOTS = tuple(tla.params.RegSlot)
_SAT_MODES = (
    tla.params.SatMode.NOSAT,
    tla.params.SatMode.SAT,
    tla.params.SatMode.UNKNOWN,
)
_ROUND_MODES = (
    tla.params.RoundMode.CAST_ROUND,
    tla.params.RoundMode.CAST_FLOOR,
    tla.params.RoundMode.CAST_CEIL,
    tla.params.RoundMode.CAST_TRUNC,
)


@pytest.mark.parametrize("fp8", _FP8)
@pytest.mark.parametrize("other", _PARTNERS)
def test_fp8_pairs_with_f32_f16_bf16_both_ways(fp8, other) -> None:
    """Only f32 is a hardware pair; f16 and bf16 are composed through it."""
    _compile(fp8, other, _DEFAULT)
    _compile(other, fp8, _DEFAULT)


@pytest.mark.parametrize("fp8", _FP8)
@pytest.mark.parametrize("other", (tla.Float16, tla.BFloat16))
@pytest.mark.parametrize("fp8_is_source", (True, False))
@pytest.mark.parametrize("slot", _FLOAT_SLOTS)
@pytest.mark.parametrize("sat", _SAT_MODES)
@pytest.mark.parametrize("rnd", _ROUND_MODES)
def test_all_valid_fp8_cast_traits(fp8, other, fp8_is_source, slot, sat, rnd) -> None:
    """Compile every valid composed fp8<->f16/bf16 trait combination."""
    src_dtype, dst_dtype = (fp8, other) if fp8_is_source else (other, fp8)
    _compile(
        src_dtype,
        dst_dtype,
        tla.params.CastParams(reg_slot=slot, sat_mode=sat, round_mode=rnd),
    )


@pytest.mark.parametrize("fp8", _FP8)
@pytest.mark.parametrize("fp8_is_source", (True, False))
@pytest.mark.parametrize("slot", _FP8_F32_SLOTS)
@pytest.mark.parametrize("sat", _SAT_MODES)
@pytest.mark.parametrize("rnd", _ROUND_MODES)
def test_all_four_pack_quarters_are_valid_on_direct_fp8_f32(
    fp8, fp8_is_source, slot, sat, rnd
) -> None:
    """Every direct fp8<->f32 cast preserves its pp0..pp3 selector."""
    src_dtype, dst_dtype = (fp8, tla.Float32) if fp8_is_source else (tla.Float32, fp8)
    _compile(
        src_dtype,
        dst_dtype,
        tla.params.CastParams(reg_slot=slot, sat_mode=sat, round_mode=rnd),
    )


@pytest.mark.parametrize(
    "sat",
    (tla.params.SatMode.SAT, tla.params.SatMode.UNKNOWN),
)
def test_fp8_narrow_accepts_default_rounding_with_sat_and_unknown(sat) -> None:
    _compile(
        tla.Float32,
        tla.Float8E4M3FN,
        tla.params.CastParams(sat_mode=sat),
    )


@pytest.mark.parametrize(
    "slot", (tla.params.RegSlot.TWO, tla.params.RegSlot.THREE)
)
@pytest.mark.parametrize("other", (tla.Float16, tla.BFloat16))
def test_pack_quarter_slots_are_refused_on_a_composed_fp8_cast(slot, other) -> None:
    """The composed fp8<->16-bit route has only even/odd placement."""
    with pytest.raises(tla.TlaCoreAPIError, match="pack quarter"):
        _compile(
            tla.Float8E4M3FN,
            other,
            tla.params.CastParams(reg_slot=slot),
        )


@pytest.mark.parametrize(
    "slot", (tla.params.RegSlot.TWO, tla.params.RegSlot.THREE)
)
def test_pack_quarter_slots_stay_legal_on_the_integer_4x_cast(slot) -> None:
    """The one cast that really reads pp0..pp3."""
    _compile(tla.Int32, tla.Int8, tla.params.CastParams(reg_slot=slot))


# The two refusals below are the lowering's, not the front end's. Which AVE
# conversions the backend actually builds is what decides them, and that is
# TlaVectorRegionPass's knowledge -- no other cast pair is gated at the API
# surface either, so fp8 is not special-cased there. They therefore need a real
# compile: dump_mlir stops at frontend TLA IR, which both of these reach
# perfectly well.
def test_fp8_to_fp8_is_refused() -> None:
    """No fp8-to-fp8 re-encode instruction exists.

    The exponent ranges differ, so it is a requantisation rather than a cast.
    """
    with pytest.raises(execution.TlaKernelCompileError):
        _lower(tla.Float8E4M3FN, tla.Float8E5M2, _DEFAULT)


def test_fp8_to_integer_is_refused() -> None:
    """fp8 pairs with f32, f16 and bf16 -- there is no fp8-to-integer path."""
    with pytest.raises(execution.TlaKernelCompileError):
        _lower(tla.Float8E4M3FN, tla.Int32, _DEFAULT)
