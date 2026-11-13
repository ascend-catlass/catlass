from __future__ import annotations

import pytest

from catlass import tla
from catlass.core_api import TlaCoreAPIError
from catlass.params import CastParams, RegSlot, RoundMode, SatMode
from catlass.tla.runtime import make_fake_tensor


def _ub(dtype, count):
    return make_fake_tensor(
        dtype,
        (count,),
        (1,),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(count,),
        layout_tag=tla.arch.RowMajor,
    )


def _kernel(dest_type, params, mask_type=None):
    @tla.kernel
    def convert(src: tla.Tensor, dst: tla.Tensor):
        with tla.vector():  # noqa: SIM117 -- separate DSL regions
            with tla.vec.func(mode="simd"):
                if tla.const_expr(mask_type is None):
                    value = src.load().to(dest_type, params)
                else:
                    mask = tla.create_mask(dtype=mask_type, pattern=tla.mask.M3)
                    value = src.load().to(dest_type, params, mask)
                dst.store(value)

    return convert


@pytest.mark.parametrize("dtype", [tla.Float4E2M1x2, tla.Float4E1M2x2])
@pytest.mark.parametrize("slot", list(RegSlot))
@pytest.mark.parametrize(
    "rounding",
    [
        RoundMode.CAST_ROUND,
        RoundMode.CAST_FLOOR,
        RoundMode.CAST_CEIL,
        RoundMode.CAST_TRUNC,
        RoundMode.CAST_RINT,
        RoundMode.CAST_NONE,
    ],
)
def test_fp4_narrow_accepts_native_modes_and_pair_mask(
    compiler_tlair, dtype, slot, rounding
):
    params = CastParams(reg_slot=slot, sat_mode=SatMode.SAT, round_mode=rounding)
    ir = compiler_tlair(
        _kernel(dtype, params, tla.UInt32),
        type_args=(_ub(tla.BFloat16, 128), _ub(dtype, 256)),
    )
    (cast_line,) = [line for line in ir.splitlines() if '"tla.cast"' in line]
    slot_code = list(RegSlot).index(slot)
    round_code = params.codes()[2]
    assert f"trait = array<i32: {slot_code}, 1, {round_code}>" in cast_line
    assert f"!tla.vector<?x!tla.{dtype.dtype}>" in cast_line
    assert "!tla.mask<64>" in ir


@pytest.mark.parametrize("dtype", [tla.Float4E2M1x2, tla.Float4E1M2x2])
@pytest.mark.parametrize("slot", list(RegSlot))
def test_fp4_widen_preserves_pair_count(compiler_tlair, dtype, slot):
    ir = compiler_tlair(
        _kernel(tla.BFloat16, CastParams(reg_slot=slot), dtype),
        type_args=(_ub(dtype, 256), _ub(tla.BFloat16, 128)),
    )
    (cast_line,) = [line for line in ir.splitlines() if '"tla.cast"' in line]
    slot_code = list(RegSlot).index(slot)
    assert f"trait = array<i32: {slot_code}, 2, 0>" in cast_line
    assert "!tla.vector<128xbf16>" in cast_line
    assert "!tla.mask<256>" in ir


def test_fp4_narrow_rejects_a_byte_lane_mask(compiler_tlair):
    with pytest.raises(
        TlaCoreAPIError, match="BF16 source mask or a 64-lane pair mask"
    ):
        compiler_tlair(
            _kernel(tla.Float4E2M1x2, CastParams(sat_mode=SatMode.SAT), tla.UInt8),
            type_args=(_ub(tla.BFloat16, 128), _ub(tla.Float4E2M1x2, 256)),
        )


@pytest.mark.parametrize("dtype", [tla.Float4E2M1x2, tla.Float4E1M2x2])
@pytest.mark.parametrize("mask_type", [tla.BFloat16, tla.UInt32])
def test_fp4_native_saturation_accepts_source_and_pair_masks(
    compiler_tlair, dtype, mask_type
):
    ir = compiler_tlair(
        _kernel(dtype, CastParams(sat_mode=SatMode.UNKNOWN), mask_type),
        type_args=(_ub(tla.BFloat16, 128), _ub(dtype, 256)),
    )
    assert (
        "!tla.mask<128>" in ir if mask_type is tla.BFloat16 else "!tla.mask<64>" in ir
    )
    assert "trait = array<i32: 0, 0, 0>" in ir


def test_fp4_narrow_rejects_nosat(compiler_tlair):
    with pytest.raises(TlaCoreAPIError, match="requires SAT"):
        compiler_tlair(
            _kernel(tla.Float4E2M1x2, CastParams()),
            type_args=(_ub(tla.BFloat16, 128), _ub(tla.Float4E2M1x2, 256)),
        )


def test_fp32_to_fp4_must_be_composed(compiler_tlair):
    with pytest.raises(TlaCoreAPIError, match="compose other routes explicitly"):
        compiler_tlair(
            _kernel(tla.Float4E2M1x2, CastParams(sat_mode=SatMode.SAT)),
            type_args=(_ub(tla.Float32, 64), _ub(tla.Float4E2M1x2, 256)),
        )


@tla.kernel
def _bad_arithmetic(src: tla.Tensor, dst: tla.Tensor):
    with tla.vector():  # noqa: SIM117 -- separate DSL regions
        with tla.vec.func(mode="simd"):
            value = src.load()
            dst.store(value + value)


@pytest.mark.parametrize("dtype", [tla.Float4E2M1x2, tla.Float4E1M2x2])
def test_payloads_do_not_silently_gain_integer_arithmetic(compiler_tlair, dtype):
    with pytest.raises(TlaCoreAPIError, match="FP4x2 arithmetic"):
        compiler_tlair(_bad_arithmetic, type_args=(_ub(dtype, 256), _ub(dtype, 256)))


@pytest.mark.parametrize("dtype", [tla.Float4E2M1x2, tla.Float4E1M2x2])
@pytest.mark.parametrize(
    ("rounding", "diagnostic"),
    [
        (RoundMode.CAST_ODD, "BF16->FP4x2 supports"),
        (RoundMode.CAST_HYBRID, "CAST_HYBRID only supports Float32->HiFloat8"),
    ],
)
def test_fp4_narrow_rejects_unsupported_rounding(
    compiler_tlair, dtype, rounding, diagnostic
):
    with pytest.raises(TlaCoreAPIError, match=diagnostic):
        compiler_tlair(
            _kernel(dtype, CastParams(sat_mode=SatMode.SAT, round_mode=rounding)),
            type_args=(_ub(tla.BFloat16, 128), _ub(dtype, 256)),
        )


@pytest.mark.parametrize("dtype", [tla.Float4E2M1x2, tla.Float4E1M2x2])
def test_packed_pairs_reject_scalar_literals(dtype):
    with pytest.raises(TypeError, match="packed encoding.*scalar literal"):
        _ = dtype(0).ir_value()
