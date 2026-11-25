"""Tests for ``tla.log`` subnormal handling (``tla.params.LogAlgo``)."""

from __future__ import annotations

from catlass.tla.runtime import make_fake_tensor

import pytest

import catlass.tla as tla
from catlass.execution_lowering import UnsupportedExecutionLowering


def _vector_tensor(dtype: type[tla.Numeric] = tla.Float32) -> tla.Tensor:
    lanes = 64 if dtype is tla.Float32 else 128
    return make_fake_tensor(
        dtype,
        (lanes,),
        (1,),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(lanes,),
        layout_tag=tla.arch.RowMajor,
    )


@tla.kernel
def log_default_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst_tile.store(tla.log(src_tile.load()))


@tla.kernel
def log_subnormal_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            reg = src_tile.load()
            mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            dst_tile.store(tla.log(reg, mask=mask, mode=tla.params.LogAlgo.SUBNORMAL))


@tla.kernel
def log_subnormal_f16_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(128), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(128), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst_tile.store(tla.log(src_tile.load(), mode=tla.params.LogAlgo.SUBNORMAL))


@tla.kernel
def log_bad_mode_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst_tile.store(tla.log(src_tile.load(), mode="subnormal"))


@tla.kernel
def log_scalar_subnormal_kernel(src: tla.Tensor) -> None:
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=256):
            tid, _, _ = tla.arch.thread_idx()
            value = src[tid]
            _ = tla.log(value, mode=tla.params.LogAlgo.SUBNORMAL)


def test_log_algo_is_params_scoped_only() -> None:
    assert tla.params.LogAlgo.FLUSH_TO_ZERO.value == "flush_to_zero"
    assert tla.params.LogAlgo.SUBNORMAL.value == "subnormal"
    assert not hasattr(tla, "LogAlgo")


def test_log_default_keeps_flush_to_zero_path() -> None:
    mlir = log_default_kernel.dump_mlir(type_args=(_vector_tensor(), _vector_tensor()))

    assert "tla.log" in mlir
    assert "#tla.log_algo" not in mlir


def test_log_subnormal_emits_log_algo_attribute() -> None:
    mlir = log_subnormal_kernel.dump_mlir(
        type_args=(_vector_tensor(), _vector_tensor())
    )

    assert "tla.log" in mlir
    assert "#tla.log_algo<subnormal>" in mlir


def test_log_subnormal_f16_emits_log_algo_attribute() -> None:
    tensor = _vector_tensor(tla.Float16)
    mlir = log_subnormal_f16_kernel.dump_mlir(type_args=(tensor, tensor))

    assert "#tla.log_algo<subnormal>" in mlir


def test_log_rejects_non_logalgo_mode() -> None:
    with pytest.raises(
        (TypeError, tla.TlaCoreAPIError, UnsupportedExecutionLowering),
        match="expected LogAlgo",
    ):
        log_bad_mode_kernel.dump_mlir(type_args=(_vector_tensor(), _vector_tensor()))


def test_log_rejects_subnormal_on_scalar_form() -> None:
    with pytest.raises(
        (TypeError, tla.TlaCoreAPIError, UnsupportedExecutionLowering),
        match="only supported for the SIMD vector form",
    ):
        log_scalar_subnormal_kernel.dump_mlir(type_args=(_vector_tensor(),))
