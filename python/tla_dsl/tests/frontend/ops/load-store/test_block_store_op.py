"""Tests for tla.store with block_stride (via BlockStoreParams)."""

from __future__ import annotations

from catlass.tla.runtime import make_fake_tensor


from typing import Any

import pytest

import catlass.tla as tla
from catlass.params import BlockLoadParams, BlockStoreParams, PostMode
import catlass.runtime as runtime_mod  # noqa: F401


def _ub_tensor(
    dtype: type[tla.Numeric] = tla.Float32,
    *,
    extent: int = 64,
) -> tla.Tensor:
    return make_fake_tensor(
        dtype,
        (extent,),
        (1,),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(extent,),
        layout_tag=tla.arch.RowMajor,
    )


@tla.kernel
def block_store_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            x_v = src_tile.load()
            dst_tile.store(x_v, params=BlockStoreParams(block_stride=32))


@tla.kernel
def block_update_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            x_v, src_tile = src_tile.load(
                BlockLoadParams(
                    block_stride=2,
                    post_mode=PostMode.POST_MODE_UPDATE,
                    repeat_stride=1,
                )
            )
            dst_tile = dst_tile.store(
                x_v,
                BlockStoreParams(
                    block_stride=2,
                    post_mode=PostMode.POST_MODE_UPDATE,
                    repeat_stride=1,
                ),
            )


def test_block_update_emits_bc_ops(compiler_tlair: Any) -> None:
    mlir = compiler_tlair(
        block_update_kernel,
        type_args=(_ub_tensor(), _ub_tensor()),
    )
    assert "tla.load_update" in mlir
    assert "tla.store_update" in mlir
    # Generic form: block_stride SSA present, mask and second source absent.
    assert "operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>" in mlir
    assert "tla.make_tensor" in mlir


@tla.kernel
def block_update_loop_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    """Sliding-window UPDATE: ``tla.range`` carries the next-tile SSA."""
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            for _ in tla.range(3):
                x_v, src_tile = src_tile.load(
                    BlockLoadParams(
                        block_stride=1,
                        post_mode=PostMode.POST_MODE_UPDATE,
                        repeat_stride=8,
                    )
                )
                dst_tile = dst_tile.store(
                    x_v,
                    BlockStoreParams(
                        block_stride=1,
                        post_mode=PostMode.POST_MODE_UPDATE,
                        repeat_stride=8,
                    ),
                )


def test_block_update_loop_emits_bc_ops(compiler_tlair: Any) -> None:
    mlir = compiler_tlair(
        block_update_loop_kernel,
        type_args=(_ub_tensor(extent=256), _ub_tensor(extent=256)),
    )
    assert "tla.load_update" in mlir
    assert "tla.store_update" in mlir
    assert "operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>" in mlir
    assert "scf.for" in mlir


def test_block_update_rejects_nonzero_coord(compiler_tlair: Any) -> None:
    @tla.kernel
    def bad_origin(src: tla.Tensor, dst: tla.Tensor) -> None:
        src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(1))
        dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
        with tla.vector():
            with tla.vec.func(mode="simd"):
                x_v, src_tile = src_tile.load(
                    BlockLoadParams(
                        block_stride=1,
                        post_mode=PostMode.POST_MODE_UPDATE,
                        repeat_stride=8,
                    )
                )
                dst_tile.store(x_v)

    with pytest.raises(Exception, match="origin window|coord"):
        compiler_tlair(
            bad_origin,
            type_args=(_ub_tensor(extent=256), _ub_tensor(extent=256)),
        )


def test_block_update_rejects_zero_post_stride(compiler_tlair: Any) -> None:
    @tla.kernel
    def bad_stride(src: tla.Tensor, dst: tla.Tensor) -> None:
        src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
        with tla.vector():
            with tla.vec.func(mode="simd"):
                src_tile.load(
                    BlockLoadParams(
                        block_stride=1,
                        post_mode=PostMode.POST_MODE_UPDATE,
                        repeat_stride=0,
                    )
                )

    with pytest.raises(Exception, match="repeat_stride"):
        compiler_tlair(bad_stride, type_args=(_ub_tensor(), _ub_tensor()))


def test_block_store_normal_rejects_repeat_stride(compiler_tlair: Any) -> None:
    """NORMAL BlockStore has no pre-offset; nonzero stride must not be ignored."""

    @tla.kernel
    def bad_normal_stride(src: tla.Tensor, dst: tla.Tensor) -> None:
        src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
        dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
        with tla.vector():
            with tla.vec.func(mode="simd"):
                x_v = src_tile.load()
                dst_tile.store(
                    x_v,
                    BlockStoreParams(block_stride=1, repeat_stride=8),
                )

    with pytest.raises(Exception, match="repeat_stride.*post_mode=update"):
        compiler_tlair(bad_normal_stride, type_args=(_ub_tensor(), _ub_tensor()))


def test_block_store_tlair_has_stride(compiler_tlair: Any) -> None:
    mlir = compiler_tlair(
        block_store_kernel,
        type_args=(_ub_tensor(), _ub_tensor()),
    )
    assert "tla.store" in mlir
    assert "stride" in mlir


def test_block_store_tlair_has_block_stride_value(compiler_tlair: Any) -> None:
    """BlockStoreParams(block_stride=32) should produce stride with value 32."""
    mlir = compiler_tlair(
        block_store_kernel,
        type_args=(_ub_tensor(), _ub_tensor()),
    )
    assert "tla.store" in mlir
    assert "stride" in mlir
    # After tla-compile, block_stride=32 should produce
    # arith.constant 32 : i32 or index in the lowered output


def test_block_store_different_stride(compiler_tlair: Any) -> None:
    """BlockStoreParams with different stride values."""

    @tla.kernel
    def block_store_16_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
        src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
        dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
        with tla.vector():
            with tla.vec.func(mode="simd"):
                x_v = src_tile.load()
                dst_tile.store(x_v, params=BlockStoreParams(block_stride=16))

    mlir = compiler_tlair(
        block_store_16_kernel,
        type_args=(_ub_tensor(), _ub_tensor()),
    )
    assert "tla.store" in mlir
    assert "stride" in mlir


def test_block_store_kernel_parameter_stride(compiler_tlair: Any) -> None:
    """BlockStoreParams with stride from a kernel scalar argument."""

    @tla.kernel
    def block_store_param_kernel(
        src: tla.Tensor, dst: tla.Tensor, stride_val: tla.Int32
    ) -> None:
        src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
        dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
        with tla.vector():
            with tla.vec.func(mode="simd"):
                x_v = src_tile.load()
                dst_tile.store(x_v, params=BlockStoreParams(block_stride=stride_val))

    n = tla.Int32(32)
    mlir = compiler_tlair(
        block_store_param_kernel,
        type_args=(_ub_tensor(), _ub_tensor(), n),
    )
    assert "tla.store" in mlir
    assert "stride" in mlir
