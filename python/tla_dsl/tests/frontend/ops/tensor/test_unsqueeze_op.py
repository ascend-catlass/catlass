"""Tests for tla.unsqueeze mask-expand vector op."""

from __future__ import annotations

import pytest

from catlass.tla.runtime import make_fake_tensor


import catlass.tla as tla
import catlass.runtime as runtime_mod  # noqa: F401


def _vector_tensor(dtype: object, lanes: int) -> tla.Tensor:
    return make_fake_tensor(
        dtype,
        (lanes,),
        (1,),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(lanes,),
        layout_tag=tla.arch.RowMajor,
    )


def _make_unsqueeze_kernel(lanes: int, dtype: object, mask_dtype: object | None = None):
    mask_dtype = dtype if mask_dtype is None else mask_dtype

    @tla.kernel
    def kernel(dst: tla.Tensor) -> None:
        dst_tile = tla.tile_view(dst, tla.make_shape(lanes), tla.make_coord(0))
        with tla.vector():
            with tla.vec.func(mode="simd"):
                mask = tla.create_mask(pattern=tla.mask.M4, dtype=mask_dtype)
                idx = tla.unsqueeze(mask, dtype)
                dst_tile.store(idx)

    return kernel


def test_unsqueeze_op_emits_mlir() -> None:
    kernel = _make_unsqueeze_kernel(64, tla.Int32)
    mlir = kernel.dump_mlir(type_args=(_vector_tensor(tla.Int32, 64),))
    assert "tla.unsqueeze" in mlir
    assert "tla.create_mask" in mlir
    assert mlir.index("tla.vec.func") < mlir.index("tla.unsqueeze")
    unsqueeze_lines = [line for line in mlir.splitlines() if "tla.unsqueeze" in line]
    assert len(unsqueeze_lines) == 1
    # tla-to-vector lowering (vunsqueeze_* bitcode call) is covered by
    # tests/lit/vector/simd/vector-unsqueeze-lowering.mlir


def test_unsqueeze_op_integer_dtypes_emit_mlir() -> None:
    for dtype, lanes, vector_ty in (
        (tla.Int8, 256, "256xi8"),
        (tla.Int16, 128, "128xi16"),
        (tla.Int32, 64, "64xi32"),
        (tla.UInt8, 256, "256xui8"),
        (tla.UInt16, 128, "128xui16"),
        (tla.UInt32, 64, "64xui32"),
    ):
        kernel = _make_unsqueeze_kernel(lanes, dtype)
        mlir = kernel.dump_mlir(type_args=(_vector_tensor(dtype, lanes),))
        assert "tla.unsqueeze" in mlir
        assert vector_ty in mlir
        idx_lines = [line for line in mlir.splitlines() if "tla.unsqueeze" in line]
        assert len(idx_lines) == 1


def test_unsqueeze_rejects_float_dtype() -> None:
    kernel = _make_unsqueeze_kernel(64, tla.Float32)
    with pytest.raises(tla.TlaCoreAPIError, match="unsupported vector element dtype"):
        kernel.dump_mlir(type_args=(_vector_tensor(tla.Float32, 64),))


def test_unsqueeze_rejects_mask_width_mismatch() -> None:
    # i16 mask is mask<128>; an i32 result needs mask<64>.
    kernel = _make_unsqueeze_kernel(64, tla.Int32, mask_dtype=tla.Int16)
    with pytest.raises(tla.TlaCoreAPIError, match="predicate lanes"):
        kernel.dump_mlir(type_args=(_vector_tensor(tla.Int32, 64),))


def test_unsqueeze_rejects_non_numeric_dtype() -> None:
    kernel = _make_unsqueeze_kernel(64, "i32", mask_dtype=tla.Int32)
    with pytest.raises(tla.TlaCoreAPIError, match="integer Numeric"):
        kernel.dump_mlir(type_args=(_vector_tensor(tla.Int32, 64),))


def test_unsqueeze_rejects_simt_mode() -> None:
    @tla.kernel
    def kernel(dst: tla.Tensor) -> None:
        dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
        with tla.vector():
            with tla.vec.func(mode="simt"):
                mask = tla.create_mask(pattern=tla.mask.M4, dtype=tla.Int32)
                dst_tile.store(tla.unsqueeze(mask, tla.Int32))

    with pytest.raises(
        tla.TlaCoreAPIError, match=r"requires tla\.vec\.func\(mode='simd'\)"
    ):
        kernel.dump_mlir(type_args=(_vector_tensor(tla.Int32, 64),))


def test_unsqueeze_result_is_full_register_for_partial_mask() -> None:
    # A 32-lane tail still yields a 64-lane UInt32 index.
    @tla.kernel
    def kernel(dst: tla.Tensor) -> None:
        dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
        with tla.vector():
            with tla.vec.func(mode="simd"):
                keep, _ = tla.update_mask(32, dtype=tla.Int32)
                dst_tile.store(tla.unsqueeze(keep, tla.UInt32), mask=keep)

    mlir = kernel.dump_mlir(type_args=(_vector_tensor(tla.UInt32, 64),))
    assert "tla.unsqueeze" in mlir
    assert "!tla.vector<64xui32>" in mlir
