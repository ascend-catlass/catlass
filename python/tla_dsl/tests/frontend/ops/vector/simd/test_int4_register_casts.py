from __future__ import annotations

import pytest

from catlass import tla
from catlass._mlir import ir
from catlass.core_api import TlaCoreAPIError
from catlass.params import (
    CastParams,
    LoadDist,
    NormalLoadParams,
    RegSlot,
    RoundMode,
    SatMode,
)
from catlass.tla.runtime import make_fake_tensor
from catlass.types import dtype_size_bits, dtype_size_bytes


def _ub(dtype, count):
    return make_fake_tensor(
        dtype,
        (count,),
        (1,),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(count,),
        layout_tag=tla.arch.RowMajor,
    )


def _kernel(dest_type, load_dist=None):
    @tla.kernel
    def convert(src: tla.Tensor, dst: tla.Tensor):
        with tla.vector():  # noqa: SIM117 -- separate DSL regions
            with tla.vec.func(mode="simd"):
                if tla.const_expr(load_dist is None):
                    value = src.load()
                else:
                    value = src.load(NormalLoadParams(load_dist=load_dist))
                dst.store(
                    value.to(
                        dest_type,
                        CastParams(),
                    )
                )

    return convert


def test_int4_type_is_registered():
    assert tla.Int4x2.dtype == "i4x2"
    assert tla.Int4x2.width == 8
    assert dtype_size_bits("i4x2") == 8
    assert dtype_size_bytes("i4x2") == 1
    with ir.Context() as ctx:
        type_ = tla.Int4x2.mlir_type(ctx)
        assert str(type_) == "!tla.i4x2"
        assert tla.Numeric.from_mlir_type(type_) is tla.Int4x2
    assert tla.Numeric.from_dtype_token("i4x2") is tla.Int4x2


@pytest.mark.parametrize(
    ("dest_type", "lanes"),
    [
        (tla.Int16, 128),
        (tla.Float16, 128),
        (tla.BFloat16, 128),
    ],
)
def test_int4_widens_to_a_byte_carried_type(compiler_tlair, dest_type, lanes):
    mlir = compiler_tlair(
        _kernel(dest_type),
        type_args=(_ub(tla.Int4x2, 256), _ub(dest_type, 256)),
    )
    (cast_line,) = [line for line in mlir.splitlines() if '"tla.cast"' in line]
    assert "!tla.vector<256x!tla.i4x2>" in cast_line
    assert f"!tla.vector<{lanes}x{dest_type.dtype}>" in cast_line


def test_int4_unpack_b8_fills_a_full_register(compiler_tlair):
    mlir = compiler_tlair(
        _kernel(tla.Float16, LoadDist.DIST_UNPACK_B8),
        type_args=(_ub(tla.Int4x2, 128), _ub(tla.Float16, 128)),
    )
    assert "#tla.load_dist<unpack_b8>" in mlir
    (cast_line,) = [line for line in mlir.splitlines() if '"tla.cast"' in line]
    assert "!tla.vector<256x!tla.i4x2>" in cast_line
    assert "!tla.vector<128xf16>" in cast_line


def test_int4_unpack4_b8_fills_a_full_register(compiler_tlair):
    mlir = compiler_tlair(
        _kernel(tla.Float16, LoadDist.DIST_UNPACK4_B8),
        type_args=(_ub(tla.Int4x2, 64), _ub(tla.Float16, 128)),
    )
    assert "#tla.load_dist<unpack4_b8>" in mlir
    (cast_line,) = [line for line in mlir.splitlines() if '"tla.cast"' in line]
    assert "!tla.vector<256x!tla.i4x2>" in cast_line
    assert "!tla.vector<128xf16>" in cast_line


def test_int4_widen_rejects_explicit_rounding(compiler_tlair):
    @tla.kernel
    def convert(src: tla.Tensor, dst: tla.Tensor):
        with tla.vector():  # noqa: SIM117 -- separate DSL regions
            with tla.vec.func(mode="simd"):
                dst.store(
                    src.load().to(
                        tla.Float16,
                        CastParams(round_mode=RoundMode.CAST_RINT),
                    )
                )

    with pytest.raises(TlaCoreAPIError, match="requires CAST_NONE"):
        compiler_tlair(
            convert,
            type_args=(_ub(tla.Int4x2, 256), _ub(tla.Float16, 128)),
        )


@pytest.mark.parametrize("dtype", [tla.Int16, tla.Float16])
@pytest.mark.parametrize("slot", list(RegSlot))
@pytest.mark.parametrize("mask_dtype", [tla.UInt16, tla.UInt32])
def test_int4_narrows_a_16_bit_pair(compiler_tlair, dtype, slot, mask_dtype):
    rounding = RoundMode.CAST_RINT if dtype is tla.Float16 else RoundMode.CAST_NONE
    params = CastParams(reg_slot=slot, sat_mode=SatMode.NOSAT, round_mode=rounding)

    @tla.kernel
    def convert(src: tla.Tensor, dst: tla.Tensor):
        with tla.vector():  # noqa: SIM117 -- keep separate vector and SIMD scopes
            with tla.vec.func(mode="simd"):
                mask = tla.create_mask(dtype=mask_dtype, pattern=tla.mask.M3)
                dst.store(src.load().to(tla.Int4x2, params, mask))

    ir_text = compiler_tlair(convert, type_args=(_ub(dtype, 128), _ub(tla.Int4x2, 256)))
    (cast_line,) = [line for line in ir_text.splitlines() if '"tla.cast"' in line]
    assert f"!tla.mask<{2048 // mask_dtype.width}>" in cast_line
    assert "!tla.vector<?x!tla.i4x2>" in cast_line
    slot_code, sat_code, round_code = params.codes()
    assert f"trait = array<i32: {slot_code}, {sat_code}, {round_code}>" in cast_line


def test_int4_rejects_unsupported_source_type(compiler_tlair):
    with pytest.raises(TlaCoreAPIError, match="Int4x2 narrows from Int16 or Float16"):
        compiler_tlair(
            _kernel(tla.Int4x2),
            type_args=(_ub(tla.Float32, 64), _ub(tla.Int4x2, 256)),
        )


def test_int4_rejects_i8_target(compiler_tlair):
    with pytest.raises(TlaCoreAPIError, match="Int4x2 widens to"):
        compiler_tlair(
            _kernel(tla.Int8),
            type_args=(_ub(tla.Int4x2, 256), _ub(tla.Int8, 128)),
        )


def test_int4_rejects_other_targets(compiler_tlair):
    with pytest.raises(TlaCoreAPIError, match="Int4x2 widens to"):
        compiler_tlair(
            _kernel(tla.Float32),
            type_args=(_ub(tla.Int4x2, 256), _ub(tla.Float32, 64)),
        )
