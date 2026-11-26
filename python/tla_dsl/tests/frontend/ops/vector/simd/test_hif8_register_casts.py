from __future__ import annotations

import pytest

from catlass import tla
from catlass.core_api import TlaCoreAPIError
from catlass.params import CastParams, RegSlot, RoundMode, SatMode
from catlass.tla.runtime import make_fake_tensor
from catlass.types import dtype_size_bytes


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


_HIF8_FLOAT_SLOTS = [(tla.Float32, slot) for slot in RegSlot] + [
    (tla.Float16, slot) for slot in (RegSlot.ZERO, RegSlot.ONE)
]


@pytest.mark.parametrize(("dtype", "slot"), _HIF8_FLOAT_SLOTS)
@pytest.mark.parametrize("sat", [SatMode.NOSAT, SatMode.SAT])
@pytest.mark.parametrize("round_mode", [RoundMode.CAST_ROUND, RoundMode.CAST_HYBRID])
def test_hif8_narrow_is_a_register_cast(compiler_tlair, dtype, slot, sat, round_mode):
    ir = compiler_tlair(
        _kernel(
            tla.HiFloat8, CastParams(reg_slot=slot, sat_mode=sat, round_mode=round_mode)
        ),
        type_args=(_ub(dtype, 2048 // dtype.width), _ub(tla.HiFloat8, 256)),
    )
    (cast_line,) = [line for line in ir.splitlines() if '"tla.cast"' in line]
    slot_code = list(RegSlot).index(slot)
    sat_code = 1 if sat is SatMode.SAT else 2
    round_code = 7 if round_mode is RoundMode.CAST_HYBRID else 0
    assert f"trait = array<i32: {slot_code}, {sat_code}, {round_code}>" in cast_line
    assert "!tla.vector<?x!tla.hif8>" in cast_line


@pytest.mark.parametrize("src", [tla.Float16, tla.Float32])
@pytest.mark.parametrize(
    "round_mode", [RoundMode.CAST_NONE, RoundMode.CAST_RINT, RoundMode.CAST_CEIL]
)
def test_hif8_requires_explicit_round_or_hybrid(compiler_tlair, src, round_mode):
    with pytest.raises(TlaCoreAPIError, match="requires CAST_ROUND"):
        compiler_tlair(
            _kernel(tla.HiFloat8, CastParams(round_mode=round_mode)),
            type_args=(_ub(src, 2048 // src.width), _ub(tla.HiFloat8, 256)),
        )


@pytest.mark.parametrize(("dtype", "slot"), _HIF8_FLOAT_SLOTS)
def test_hif8_widen_preserves_source_mask_domain(compiler_tlair, dtype, slot):
    lanes = 2048 // dtype.width
    params = CastParams(reg_slot=slot)
    ir = compiler_tlair(
        _kernel(dtype, params, tla.HiFloat8),
        type_args=(_ub(tla.HiFloat8, 256), _ub(dtype, lanes)),
    )
    (cast_line,) = [line for line in ir.splitlines() if '"tla.cast"' in line]
    assert f"trait = array<i32: {', '.join(map(str, params.codes()))}>" in cast_line
    assert "!tla.mask<256>" in ir
    assert f"!tla.vector<{lanes}x{dtype.dtype}>" in cast_line


@pytest.mark.parametrize("src,dst", [(tla.Float16, tla.HiFloat8), (tla.HiFloat8, tla.Float16)])
@pytest.mark.parametrize("slot", [RegSlot.TWO, RegSlot.THREE])
def test_float16_hif8_rejects_quarter_slots(compiler_tlair, src, dst, slot):
    params = CastParams(reg_slot=slot, round_mode=RoundMode.CAST_ROUND if src is tla.Float16 else RoundMode.CAST_NONE)
    with pytest.raises(TlaCoreAPIError, match="pack quarter"):
        compiler_tlair(
            _kernel(dst, params),
            type_args=(_ub(src, 2048 // src.width), _ub(dst, 2048 // dst.width)),
        )


@tla.kernel
def _bad_arithmetic(src: tla.Tensor, dst: tla.Tensor):
    with tla.vector():  # noqa: SIM117 -- separate DSL regions
        with tla.vec.func(mode="simd"):
            value = src.load()
            dst.store(value + value)


@pytest.mark.parametrize("dtype", [tla.HiFloat8])
def test_payloads_do_not_silently_gain_integer_arithmetic(compiler_tlair, dtype):
    with pytest.raises(TlaCoreAPIError, match="HiFloat8 arithmetic"):
        compiler_tlair(_bad_arithmetic, type_args=(_ub(dtype, 256), _ub(dtype, 256)))


@pytest.mark.parametrize(
    ("source_type", "destination_type"),
    [
        (tla.HiFloat8, tla.Float32),
        (tla.Float32, tla.Float32),
        (tla.Float32, tla.Float16),
        (tla.Float32, tla.Float8E4M3FN),
        (tla.Float16, tla.Float32),
    ],
)
def test_hybrid_rejects_other_routes(compiler_tlair, source_type, destination_type):
    with pytest.raises(
        TlaCoreAPIError, match="CAST_HYBRID only supports Float16/Float32->HiFloat8"
    ):
        compiler_tlair(
            _kernel(destination_type, CastParams(round_mode=RoundMode.CAST_HYBRID)),
            type_args=(
                _ub(source_type, 256 // dtype_size_bytes(source_type.dtype)),
                _ub(destination_type, 256 // dtype_size_bytes(destination_type.dtype)),
            ),
        )


@tla.kernel
def _hif8_control_flow(src: tla.Tensor, dst: tla.Tensor, limit: tla.Int32):
    with tla.vector():
        with tla.vec.func(mode="simd"):
            value = src.load()
            for iteration in tla.range(limit):
                if iteration > 0:
                    value = value.bitcast(tla.UInt8).bitcast(tla.HiFloat8)
            dst.store(value)


def test_hif8_registers_pass_through_if_and_for(compiler_tlair):
    ir = compiler_tlair(
        _hif8_control_flow,
        type_args=(_ub(tla.HiFloat8, 256), _ub(tla.HiFloat8, 256), 2),
    )
    assert "scf.for" in ir
    assert "scf.if" in ir
    assert "!tla.hif8" in ir
