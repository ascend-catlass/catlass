from __future__ import annotations

import pytest

import catlass.tla as tla
from catlass.tla.runtime import make_fake_tensor


def _gm(dtype, lanes=128):
    return make_fake_tensor(
        dtype,
        (lanes,),
        (1,),
        addrspace=tla.AddressSpace.gm,
        origin_shape=(lanes,),
        layout_tag=tla.arch.RowMajor,
    )


def _kernel(params, masked=False, lanes=128):
    pattern = tla.mask.M4 if masked else tla.mask.ALL

    @tla.kernel
    def k(src: tla.Tensor, dst: tla.Tensor) -> None:
        src_t = tla.tile_view(src, tla.make_shape(lanes), tla.make_coord(0))
        dst_t = tla.tile_view(dst, tla.make_shape(lanes), tla.make_coord(0))
        src_ub = tla.make_tensor_like(
            tla.allocate((lanes,), tla.Float16, tla.AddressSpace.ub, 256),
            src_t,
            tla.arch.RowMajor,
        )
        dst_ub = tla.make_tensor_like(
            tla.allocate((lanes,), tla.BFloat16, tla.AddressSpace.ub, 256),
            dst_t,
            tla.arch.RowMajor,
        )
        loaded = tla.flag("loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
        converted = tla.flag("converted", tla.pipes.VECTOR, tla.pipes.MTE3)
        with tla.vector():
            tla.copy(src_ub, src_t)
            tla.set_flag(loaded)
            tla.wait_flag(loaded)
            with tla.vec.func(mode="simd"):
                mask = tla.create_mask(pattern=pattern, dtype=tla.Float16)
                dst_ub.store(src_ub.load().to(tla.BFloat16, params, mask))
            tla.set_flag(converted)
            tla.wait_flag(converted)
            tla.copy(dst_t, dst_ub)

    return k


@pytest.mark.parametrize(
    "round_mode",
    [
        mode
        for mode in tla.params.RoundMode
        if mode not in (tla.params.RoundMode.CAST_HYBRID, tla.params.RoundMode.CAST_ODD)
    ],
)
@pytest.mark.parametrize("reg_slot", (tla.params.RegSlot.ZERO, tla.params.RegSlot.ONE))
@pytest.mark.parametrize("masked", (False, True))
def test_half_to_bf16_compiles_with_supported_rounding(
    round_mode, reg_slot, masked, isolated_compile_cache
):
    params = tla.params.CastParams(
        reg_slot=reg_slot, sat_mode=tla.params.SatMode.UNKNOWN, round_mode=round_mode
    )
    _kernel(params, masked).compile(
        type_args=(_gm(tla.Float16), _gm(tla.BFloat16)), options="--npu-arch 3510"
    )


def test_half_to_bf16_rejects_hybrid_rounding():
    params = tla.params.CastParams(
        sat_mode=tla.params.SatMode.UNKNOWN,
        round_mode=tla.params.RoundMode.CAST_HYBRID,
    )
    with pytest.raises(
        tla.TlaCoreAPIError, match="CAST_HYBRID only supports Float16/Float32->HiFloat8"
    ):
        _kernel(params).dump_mlir(type_args=(_gm(tla.Float16), _gm(tla.BFloat16)))


@pytest.mark.parametrize("lanes", (17, 128))
def test_half_to_bf16_preserves_valid_lanes(lanes):
    mlir = _kernel(
        tla.params.CastParams(sat_mode=tla.params.SatMode.UNKNOWN), lanes=lanes
    ).dump_mlir(type_args=(_gm(tla.Float16, lanes), _gm(tla.BFloat16, lanes)))
    assert f"!tla.vector<{lanes}xbf16>" in mlir


@pytest.mark.parametrize("sat_mode", (tla.params.SatMode.NOSAT, tla.params.SatMode.SAT))
def test_half_to_bf16_rejects_explicit_sat_mode(sat_mode):
    with pytest.raises(
        tla.TlaCoreAPIError, match=r"f16->bf16 requires SatMode\.UNKNOWN"
    ):
        _kernel(tla.params.CastParams(sat_mode=sat_mode)).dump_mlir(
            type_args=(_gm(tla.Float16), _gm(tla.BFloat16))
        )


@pytest.mark.parametrize("reg_slot", (tla.params.RegSlot.TWO, tla.params.RegSlot.THREE))
def test_half_to_bf16_rejects_pack_quarters(reg_slot):
    with pytest.raises(tla.TlaCoreAPIError, match="does not select a pack quarter"):
        _kernel(
            tla.params.CastParams(
                reg_slot=reg_slot, sat_mode=tla.params.SatMode.UNKNOWN
            )
        ).dump_mlir(type_args=(_gm(tla.Float16), _gm(tla.BFloat16)))
