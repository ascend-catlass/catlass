"""Tests for tla.store with unaligned_ub_access (via params)."""

from __future__ import annotations

from catlass.tla.runtime import make_fake_tensor


from typing import Any

import pytest
import catlass.tla as tla
from catlass.params import (
    NormalStoreParams,
    PostMode,
    UnalignMaskStoreParams,
    UnalignStoreParams,
)
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

    with pytest.raises(
        TlaLoweringError, match="UnalignStoreParams does not accept mask="
    ):
        compiler_tlair(store_unalign_masked, type_args=(_ub_tensor(),))


@pytest.mark.parametrize(
    ("dtype", "lanes", "stride"),
    [
        (tla.Float16, 128, 8),
        (tla.Int32, 64, 2),
    ],
)
@pytest.mark.parametrize("alignment", [32, 64])
def test_store_unaligned_mask_advances_window(
    compiler_tlair: Any,
    dtype: type[tla.Numeric],
    lanes: int,
    alignment: int,
    stride: int,
) -> None:
    @tla.kernel
    def store_mask() -> None:
        ptr = tla.allocate(64, dtype, tla.AddressSpace.ub, alignment)
        layout = tla.make_layout(tla.make_shape(64), tla.make_stride(1))
        dst = tla.make_tensor(ptr, layout)
        with tla.vector():
            with tla.vec.func(mode="simd"):
                mask = tla.create_mask(pattern=tla.mask.VL1, dtype=dtype)
                dst.store(mask, UnalignMaskStoreParams())

    ir = compiler_tlair(store_mask, type_args=())
    assert '"tla.store_unalign_update"' in ir
    assert f"!tla.mask<{lanes}>" in ir
    assert f"value = {stride} : i32" in ir
    assert f", ub, {alignment}>, !tla.ureg)" in ir


@pytest.mark.parametrize(
    ("dtype", "offset", "alignment", "conditional", "valid"),
    [
        (tla.Int32, 0, 8, False, True),
        (tla.Int32, 2, 8, False, True),
        (tla.Int32, 1, 8, True, False),
        (tla.Float16, 0, 16, False, True),
        (tla.Float16, 8, 16, False, True),
        (tla.Float16, 4, 16, True, False),
    ],
)
def test_store_unaligned_mask_carries_window_in_range(
    compiler_tlair: Any,
    dtype: type[tla.Numeric],
    offset: int,
    alignment: int,
    conditional: bool,
    valid: bool,
) -> None:
    @tla.kernel
    def store_mask() -> None:
        ptr = tla.allocate(64, dtype, tla.AddressSpace.ub, 32)
        layout = tla.make_layout(tla.make_shape(32), tla.make_stride(1))
        dst = tla.make_tensor(ptr + offset if offset else ptr, layout)
        with tla.vector():
            with tla.vec.func(mode="simd"):
                for i in tla.range(3):
                    mask = tla.create_mask(pattern=tla.mask.VL1, dtype=dtype)
                    if conditional:
                        if i < 2:
                            dst = dst.store(mask, UnalignMaskStoreParams())
                    else:
                        dst = dst.store(mask, UnalignMaskStoreParams())

    if not valid:
        with pytest.raises(
            TlaLoweringError, match=f"{alignment}-byte-aligned UB address"
        ):
            compiler_tlair(store_mask, type_args=())
        return
    ir = compiler_tlair(store_mask, type_args=())
    assert '"scf.for"' in ir
    assert '"tla.store_unalign_update"' in ir
    assert '"builtin.unrealized_conversion_cast"' not in ir
    assert ", ub, 32>, !tla.ureg)" in ir


@pytest.mark.parametrize(
    ("control_flow", "dtype", "alignment"),
    [
        ("if", tla.Int32, 32),
        ("if-expression", tla.Float16, 64),
        ("if-for", tla.Int32, 64),
        ("for-if", tla.Float16, 32),
    ],
)
def test_store_unaligned_mask_preserves_pointer_type_in_control_flow(
    compiler_tlair: Any, control_flow: str, dtype: type[tla.Numeric], alignment: int
) -> None:
    @tla.kernel
    def store_mask(flag: tla.Int32, trips: tla.Int32) -> None:
        ptr = tla.allocate(64, dtype, tla.AddressSpace.ub, alignment)
        layout = tla.make_layout(tla.make_shape(8), tla.make_stride(1))
        dst = tla.make_tensor(ptr, layout)
        with tla.vector():
            with tla.vec.func(mode="simd"):
                window = dst
                mask = tla.create_mask(pattern=tla.mask.VL1, dtype=dtype)
                params = UnalignMaskStoreParams()
                if control_flow == "if":
                    if flag > 0:
                        window = window.store(mask, params)
                elif control_flow == "if-expression":
                    window = window.store(mask, params) if flag > 0 else window
                elif control_flow == "if-for":
                    if flag > 0:
                        for _ in tla.range(trips):
                            window = window.store(mask, params)
                else:
                    for _ in tla.range(trips):
                        if flag > 0:
                            window = window.store(mask, params)
                window.store(mask, params)
                dst.store(mask, params)

    ir = compiler_tlair(store_mask, type_args=(tla.Int32(1), tla.Int32(3)))
    assert '"scf.if"' in ir
    assert ('"scf.for"' in ir) == ("for" in control_flow)
    assert '"builtin.unrealized_conversion_cast"' not in ir
    stores = [line for line in ir.splitlines() if '"tla.store_unalign_update"' in line]
    assert len(stores) == 3
    assert all(f", ub, {alignment}>, !tla.ureg)" in store for store in stores)


@pytest.mark.parametrize(
    ("dtype", "then_offset", "else_offset", "extra_offset", "known_misaligned"),
    [
        (tla.Int32, 1, 3, 0, True),
        (tla.Int32, 1, 3, 1, False),
        (tla.Float16, 4, 12, 0, True),
        (tla.Float16, 4, 12, 4, False),
        (tla.Int32, 1, 2, 0, False),
        (tla.Int32, 1, None, 0, False),
        (tla.Int32, 1, None, 1, False),
    ],
)
def test_store_unaligned_mask_address_after_pointer_branch(
    compiler_tlair: Any,
    isolated_compile_cache: Any,
    dtype: type[tla.Numeric],
    then_offset: int,
    else_offset: int | None,
    extra_offset: int,
    known_misaligned: bool,
) -> None:
    del isolated_compile_cache

    @tla.kernel
    def store_mask(flag: tla.Int32, offset: tla.Int32) -> None:
        ptr = tla.allocate(64, dtype, tla.AddressSpace.ub, 32)
        layout = tla.make_layout(tla.make_shape(8), tla.make_stride(1))
        selected = ptr
        if flag > 0:
            selected = ptr + then_offset
        else:
            selected = ptr + (offset if else_offset is None else else_offset)
        dst = tla.make_tensor(
            selected + extra_offset if extra_offset else selected, layout
        )
        with tla.vector():
            with tla.vec.func(mode="simd"):
                mask = tla.create_mask(pattern=tla.mask.VL1, dtype=dtype)
                dst.store(mask, UnalignMaskStoreParams())

    args = (tla.Int32(1), tla.Int32(3))
    if known_misaligned:
        with pytest.raises(TlaLoweringError, match="byte-aligned UB address"):
            compiler_tlair(store_mask, type_args=args)
    else:
        assert tla.compile(store_mask, *args, options="--npu-arch 3510") is not None


@pytest.mark.parametrize(
    ("dest_dtype", "mask_dtype", "offset", "error", "message"),
    [
        (tla.Int8, tla.Int8, 0, tla.TlaCoreAPIError, "expected b16/b32"),
        (tla.Int64, tla.Int64, 0, tla.TlaCoreAPIError, "expected b16/b32"),
        (tla.Int32, tla.Float16, 0, tla.TlaCoreAPIError, "implies !tla.mask"),
        (tla.Int32, tla.Int32, 1, TlaLoweringError, "8-byte-aligned UB address"),
        (tla.Float16, tla.Float16, 4, TlaLoweringError, "16-byte-aligned UB address"),
    ],
)
def test_store_unaligned_mask_rejects_invalid_dest(
    compiler_tlair: Any,
    dest_dtype: type[tla.Numeric],
    mask_dtype: type[tla.Numeric],
    offset: int,
    error: type[Exception],
    message: str,
) -> None:
    @tla.kernel
    def store_mask() -> None:
        ptr = tla.allocate(64, dest_dtype, tla.AddressSpace.ub, 32)
        layout = tla.make_layout(tla.make_shape(32), tla.make_stride(1))
        dst = tla.make_tensor(ptr + offset if offset else ptr, layout)
        with tla.vector():
            with tla.vec.func(mode="simd"):
                mask = tla.create_mask(pattern=tla.mask.VL1, dtype=mask_dtype)
                dst.store(mask, UnalignMaskStoreParams())

    with pytest.raises(error, match=message):
        compiler_tlair(store_mask, type_args=())


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
