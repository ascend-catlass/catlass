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


_NOSAT = tla.params.CastParams(sat_mode=tla.params.SatMode.NOSAT)
_FP8 = (tla.Float8E4M3FN, tla.Float8E5M2)
_FLOAT_SLOTS = (tla.params.RegSlot.ZERO, tla.params.RegSlot.ONE)
_FP8_F32_SLOTS = tuple(tla.params.RegSlot)
_FP16 = (tla.Float16, tla.BFloat16)
_FP8_WIDEN_THROUGH_F32 = (
    (tla.Float8E4M3FN, tla.Float16),
    (tla.Float8E4M3FN, tla.BFloat16),
    (tla.Float8E5M2, tla.BFloat16),
)
_SAT_MODES = (
    tla.params.SatMode.NOSAT,
    tla.params.SatMode.SAT,
)
_FP8_NARROW_ROUND_MODES = (
    tla.params.RoundMode.CAST_RINT,
    tla.params.RoundMode.CAST_NONE,
)


@pytest.mark.parametrize("src,dst", _FP8_WIDEN_THROUGH_F32)
@pytest.mark.parametrize("slot", _FLOAT_SLOTS)
def test_fp8_widen_through_f32_accepts_even_and_odd_slots(src, dst, slot) -> None:
    _compile(
        src,
        dst,
        tla.params.CastParams(reg_slot=slot),
    )


@pytest.mark.parametrize("src", _FP16)
@pytest.mark.parametrize("dst", _FP8)
@pytest.mark.parametrize("slot", _FLOAT_SLOTS)
@pytest.mark.parametrize("sat", _SAT_MODES)
@pytest.mark.parametrize("rnd", _FP8_NARROW_ROUND_MODES)
def test_half_to_fp8_accepts_supported_round_modes(src, dst, slot, sat, rnd) -> None:
    _compile(
        src, dst, tla.params.CastParams(reg_slot=slot, sat_mode=sat, round_mode=rnd)
    )


@pytest.mark.parametrize("src", (*_FP16, tla.Float32))
@pytest.mark.parametrize("dst", _FP8)
@pytest.mark.parametrize("sat", _SAT_MODES)
@pytest.mark.parametrize(
    "rnd",
    tuple(mode for mode in tla.params.RoundMode if mode not in _FP8_NARROW_ROUND_MODES),
)
def test_fp8_narrow_rejects_unsupported_round_modes(src, dst, sat, rnd) -> None:
    params = tla.params.CastParams(sat_mode=sat, round_mode=rnd)
    with pytest.raises(
        tla.TlaCoreAPIError,
        match=f"{src.dtype}->{dst.dtype} requires CAST_RINT or CAST_NONE",
    ):
        _compile(src, dst, params)


@pytest.mark.parametrize("slot", _FLOAT_SLOTS)
def test_e5m2_to_f16_accepts_even_and_odd_slots(slot) -> None:
    _compile(
        tla.Float8E5M2,
        tla.Float16,
        tla.params.CastParams(reg_slot=slot),
    )


@pytest.mark.parametrize("fp8", _FP8)
@pytest.mark.parametrize("slot", _FP8_F32_SLOTS)
def test_fp8_to_f32_accepts_all_pack_quarters(fp8, slot) -> None:
    _compile(
        fp8,
        tla.Float32,
        tla.params.CastParams(reg_slot=slot),
    )


@pytest.mark.parametrize("fp8", _FP8)
@pytest.mark.parametrize("slot", _FP8_F32_SLOTS)
@pytest.mark.parametrize("sat", _SAT_MODES)
@pytest.mark.parametrize("rnd", _FP8_NARROW_ROUND_MODES)
def test_f32_to_fp8_accepts_all_pack_quarters(fp8, slot, sat, rnd) -> None:
    _compile(
        tla.Float32,
        fp8,
        tla.params.CastParams(reg_slot=slot, sat_mode=sat, round_mode=rnd),
    )


@pytest.mark.parametrize("fp8", _FP8)
@pytest.mark.parametrize("half", _FP16)
@pytest.mark.parametrize("fp8_is_source", (True, False))
@pytest.mark.parametrize("slot", (tla.params.RegSlot.TWO, tla.params.RegSlot.THREE))
def test_pack_quarter_slots_are_refused_for_fp8_and_fp16(
    fp8, half, fp8_is_source, slot
) -> None:
    src, dst = (fp8, half) if fp8_is_source else (half, fp8)
    sat = tla.params.SatMode.UNKNOWN if fp8_is_source else tla.params.SatMode.NOSAT
    with pytest.raises(tla.TlaCoreAPIError, match="pack quarter"):
        _compile(src, dst, tla.params.CastParams(reg_slot=slot, sat_mode=sat))


def test_fp8_to_fp8_is_refused() -> None:
    """No fp8-to-fp8 re-encode instruction exists.

    The exponent ranges differ, so it is a requantisation rather than a cast.
    """
    with pytest.raises(execution.TlaKernelCompileError):
        _lower(tla.Float8E4M3FN, tla.Float8E5M2, _NOSAT)


def test_fp8_to_integer_is_refused() -> None:
    """FP8 to integer is rejected during lowering."""
    with pytest.raises(execution.TlaKernelCompileError):
        _lower(tla.Float8E4M3FN, tla.Int32, _NOSAT)
