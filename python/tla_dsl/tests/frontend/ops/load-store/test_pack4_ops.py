from __future__ import annotations

import pytest

from catlass import tla
from catlass.execution_lowering import TlaLoweringError, UnsupportedExecutionLowering
from catlass.params import (
    LoadDist,
    NormalLoadParams,
    NormalStoreParams,
    PostMode,
    StoreDist,
)
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


@tla.kernel
def _unpack(src: tla.Tensor, dst: tla.Tensor):
    with tla.vector():  # noqa: SIM117 -- separate DSL regions
        with tla.vec.func(mode="simd"):
            expanded = src.load(NormalLoadParams(load_dist=LoadDist.DIST_UNPACK4_B8))
            dst.store(expanded)


@tla.kernel
def _pack(src: tla.Tensor, dst: tla.Tensor):
    with tla.vector():  # noqa: SIM117 -- separate DSL regions
        with tla.vec.func(mode="simd"):
            dst.store(
                src.load(), NormalStoreParams(store_dist=StoreDist.DIST_PACK4_B32)
            )


@pytest.mark.parametrize(
    "dtype, token",
    [
        (tla.Int8, "i8"),
        (tla.UInt8, "ui8"),
        (tla.Float8E4M3FN, "f8E4M3FN"),
        (tla.Float8E5M2, "f8E5M2"),
    ],
)
def test_unpack4_result_is_full_register(compiler_tlair, dtype, token):
    ir = compiler_tlair(_unpack, type_args=(_ub(dtype, 64), _ub(dtype, 256)))
    assert "#tla.load_dist<unpack4_b8>" in ir
    assert f"!tla.vector<256x{token}>" in ir


@pytest.mark.parametrize("dtype", [tla.Float16, tla.Float32, tla.Int16, tla.UInt32])
def test_unpack4_rejects_non_byte_storage(compiler_tlair, dtype):
    with pytest.raises(TlaLoweringError, match="DIST_UNPACK4_B8 requires a 1-byte"):
        compiler_tlair(_unpack, type_args=(_ub(dtype, 64), _ub(dtype, 64)))


@pytest.mark.parametrize(
    "source, lanes",
    [
        (tla.Int8, 256),
        (tla.Int16, 128),
        (tla.Int32, 64),
        (tla.Float8E4M3FN, 256),
        (tla.Float8E5M2, 256),
    ],
)
def test_pack4_accepts_byte_halfword_and_word_registers(compiler_tlair, source, lanes):
    ir = compiler_tlair(_pack, type_args=(_ub(source, lanes), _ub(tla.UInt8, 13)))
    assert "#tla.store_dist<pack4_b32>" in ir


@pytest.mark.parametrize("dest", [tla.Int16, tla.Float32])
def test_pack4_rejects_non_byte_destination(compiler_tlair, dest):
    with pytest.raises(TlaLoweringError, match="DIST_PACK4_B32 requires"):
        compiler_tlair(_pack, type_args=(_ub(tla.Int32, 64), _ub(dest, 64)))


@tla.kernel
def _unpack_update(source: tla.Tensor, destination: tla.Tensor):
    with tla.vector():
        with tla.vec.func(mode="simd"):
            values, source = source.load(
                NormalLoadParams(
                    load_dist=LoadDist.DIST_UNPACK4_B8,
                    post_mode=PostMode.POST_MODE_UPDATE,
                    post_update_stride=64,
                )
            )
            destination.store(values)


@tla.kernel
def _pack_update(source: tla.Tensor, destination: tla.Tensor):
    with tla.vector():
        with tla.vec.func(mode="simd"):
            destination = destination.store(
                source.load(),
                NormalStoreParams(
                    store_dist=StoreDist.DIST_PACK4_B32,
                    post_mode=PostMode.POST_MODE_UPDATE,
                    post_update_stride=64,
                ),
            )


@pytest.mark.parametrize(
    "kernel, diagnostic",
    [
        (
            _unpack_update,
            "post_mode=update with load_dist='unpack4_b8' is not supported yet",
        ),
        (
            _pack_update,
            "post_mode=update with store_dist='pack4_b32' is not supported yet",
        ),
    ],
)
def test_pack4_modes_reject_post_update(compiler_tlair, kernel, diagnostic):
    with pytest.raises(UnsupportedExecutionLowering, match=diagnostic):
        compiler_tlair(kernel, type_args=(_ub(tla.UInt8, 256), _ub(tla.UInt8, 256)))
