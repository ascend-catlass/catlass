"""Tests for tla.store with unaligned_ub_access (via params)."""

from __future__ import annotations

from catlass.tla.runtime import make_fake_tensor


from typing import Any

import pytest
import catlass.tla as tla
from catlass.params import NormalStoreParams, PostMode, UnalignStoreParams
from catlass.execution_lowering import TlaLoweringError
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


def _gm_tensor(
    dtype: type[tla.Numeric] = tla.Float32,
    *,
    extent: int = 64,
) -> tla.Tensor:
    return make_fake_tensor(
        dtype,
        (extent,),
        (1,),
        origin_shape=(extent,),
        layout_tag=tla.arch.RowMajor,
    )


@tla.kernel
def store_normal_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            x_v = src_tile.load()
            dst_tile.store(x_v)


@tla.kernel
def store_to_gm_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst_tile.store(src_tile.load())


def test_store_normal_has_no_unalign_attr(compiler_tlair: Any) -> None:
    mlir = compiler_tlair(
        store_normal_kernel,
        type_args=(_ub_tensor(), _ub_tensor()),
    )

    assert "tla.store" in mlir
    assert "unaligned_ub_access" not in mlir


def test_store_unalign_using_params_emits_tlair(compiler_tlair: Any) -> None:
    """store_unalign via store(..., UnalignStoreParams())."""

    @tla.kernel
    def store_unalign_params_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
        src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
        dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
        with tla.vector():
            with tla.vec.func(mode="simd"):
                x_v = src_tile.load()
                dst_tile.store(x_v, params=UnalignStoreParams())

    mlir = compiler_tlair(
        store_unalign_params_kernel,
        type_args=(_ub_tensor(), _ub_tensor()),
    )

    assert "tla.store" in mlir
    assert "unaligned_ub_access" in mlir


@pytest.mark.parametrize(
    "post_mode", [PostMode.POST_MODE_NORMAL, PostMode.POST_MODE_UPDATE]
)
def test_store_unalign_rejects_predicate(compiler_tlair: Any, post_mode: str) -> None:
    @tla.kernel
    def store_unalign_masked(dst: tla.Tensor) -> None:
        dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
        with tla.vector():
            with tla.vec.func(mode="simd"):
                mask, _ = tla.update_mask(1, dtype=tla.Float32)
                dst_tile.store(
                    tla.full(1.0, tla.Float32),
                    UnalignStoreParams(post_mode=post_mode),
                    mask=mask,
                )

    with pytest.raises(TlaLoweringError, match="UnalignStoreParams does not accept mask="):
        compiler_tlair(store_unalign_masked, type_args=(_ub_tensor(),))


def test_store_normal_using_params_emits_tlair(compiler_tlair: Any) -> None:
    """Normal aligned store via store(..., NormalStoreParams())."""

    @tla.kernel
    def store_normal_params_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
        src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
        dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
        with tla.vector():
            with tla.vec.func(mode="simd"):
                x_v = src_tile.load()
                dst_tile.store(x_v, params=NormalStoreParams())

    mlir = compiler_tlair(
        store_normal_params_kernel,
        type_args=(_ub_tensor(), _ub_tensor()),
    )

    assert "tla.store" in mlir
    assert "unaligned_ub_access" not in mlir


def test_store_rejects_gm_tensor(compiler_tlair: Any) -> None:
    with pytest.raises(
        tla.TlaCoreAPIError,
        match="invalid argument 'dest'.*expected addrspace ub, got gm",
    ):
        compiler_tlair(
            store_to_gm_kernel,
            type_args=(_ub_tensor(), _gm_tensor()),
        )
