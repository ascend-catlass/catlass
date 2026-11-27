from __future__ import annotations

import pytest

from catlass import tla
from catlass.params import CastParams, RegSlot, RoundMode, SatMode
from catlass.tla.runtime import make_fake_tensor


def _ub(dtype, size):
    return make_fake_tensor(
        dtype,
        (size,),
        (1,),
        addrspace=tla.AddressSpace.ub,
        layout_tag=tla.arch.RowMajor,
    )


def _convert(dst_type, params, mask_type=None):
    @tla.kernel
    def kernel(src: tla.Tensor, dst: tla.Tensor):
        with tla.vector():  # noqa: SIM117 -- nested vector regions
            with tla.vec.func(mode="simd"):
                mask = None
                if tla.const_expr(mask_type is not None):
                    mask = tla.create_mask(pattern=tla.mask.VL3, dtype=mask_type)
                result = src.load().to(dst_type, params, mask)
                dst.store(result)

    return kernel


@pytest.mark.parametrize("slot", (RegSlot.ZERO, RegSlot.ONE))
@pytest.mark.parametrize("mask_type", (None, tla.BFloat16))
def test_bf16_exponent_cast(compiler_tlair, slot, mask_type):
    dst = _ub(tla.Float8E8M0, 256)
    params = CastParams(reg_slot=slot)
    ir = compiler_tlair(
        _convert(tla.Float8E8M0, params, mask_type),
        type_args=(_ub(tla.BFloat16, 128), dst),
    )
    (cast,) = [line for line in ir.splitlines() if '"tla.cast"' in line]
    operands = ["!tla.vector<128xbf16>"]
    if mask_type is not None:
        operands.append("!tla.mask<128>")
    assert f"({', '.join(operands)}) -> !tla.vector<?x!tla.f8e8m0>" in cast
    assert f"trait = array<i32: {', '.join(map(str, params.codes()))}>" in cast


@pytest.mark.parametrize(
    "sat_mode,round_mode",
    [
        *(
            (SatMode.UNKNOWN, mode)
            for mode in RoundMode
            if mode is not RoundMode.CAST_NONE
        ),
        (SatMode.SAT, RoundMode.CAST_NONE),
        (SatMode.NOSAT, RoundMode.CAST_NONE),
    ],
)
def test_bf16_exponent_cast_requires_none_and_unknown(
    compiler_tlair, sat_mode, round_mode
):
    params = CastParams(sat_mode=sat_mode, round_mode=round_mode)
    with pytest.raises(
        tla.TlaCoreAPIError,
        match=r"bf16->f8e8m0 requires CAST_NONE and SatMode\.UNKNOWN",
    ):
        compiler_tlair(
            _convert(tla.Float8E8M0, params),
            type_args=(_ub(tla.BFloat16, 128), _ub(tla.Float8E8M0, 256)),
        )


@pytest.mark.parametrize(
    "settings,message",
    [
        ({"source": tla.Float32}, "pairs with BFloat16"),
        ({"source": tla.Float8E8M0, "dest": tla.Float16}, "pairs with BFloat16"),
        ({"source": tla.Float8E8M0, "dest": tla.BFloat16, "mask": tla.BFloat16}, "expected 256"),
        ({"slot": RegSlot.TWO}, "pack quarter"),
        ({"slot": RegSlot.THREE}, "pack quarter"),
        ({"mask": tla.UInt8}, "expected 128"),
    ],
)
def test_e8m0_rejects_invalid_casts(compiler_tlair, settings, message):
    def lower(source=tla.BFloat16, dest=tla.Float8E8M0, slot=RegSlot.ZERO, mask=None):
        compiler_tlair(
            _convert(dest, CastParams(reg_slot=slot), mask),
            type_args=(_ub(source, 32), _ub(dest, 64)),
        )

    with pytest.raises(tla.TlaCoreAPIError, match=message):
        lower(**settings)


@pytest.mark.parametrize("slot", [RegSlot.ZERO, RegSlot.ONE])
def test_e8m0_widens_with_byte_mask(compiler_tlair, slot):
    src = _ub(tla.Float8E8M0, 256)
    dst = _ub(tla.BFloat16, 128)
    params = CastParams(reg_slot=slot)
    ir = compiler_tlair(
        _convert(tla.BFloat16, params, tla.Float8E8M0),
        type_args=(src, dst),
    )
    (cast,) = [line for line in ir.splitlines() if '"tla.cast"' in line]
    assert "(!tla.vector<256x!tla.f8e8m0>, !tla.mask<256>) -> !tla.vector<128xbf16>" in cast
    assert f"trait = array<i32: {', '.join(map(str, params.codes()))}>" in cast
