"""Tests for tla.squeeze mask-compress vector op."""

from __future__ import annotations

import pytest

from catlass.execution_lowering import TlaLoweringError
from catlass.tla.runtime import make_fake_tensor


import catlass.tla as tla
import catlass.runtime as runtime_mod  # noqa: F401


def _vector_tensor_pair() -> tuple[tla.Tensor, tla.Tensor]:
    return (
        make_fake_tensor(
            tla.Float32,
            (64,),
            (1,),
            addrspace=tla.AddressSpace.ub,
            origin_shape=(64,),
            layout_tag=tla.arch.RowMajor,
        ),
        make_fake_tensor(
            tla.Float32,
            (64,),
            (1,),
            addrspace=tla.AddressSpace.ub,
            origin_shape=(64,),
            layout_tag=tla.arch.RowMajor,
        ),
    )


@tla.kernel
def squeeze_vector_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            x = src_tile.load()
            mask = tla.create_mask(pattern=tla.mask.VL8, dtype=tla.Float32)
            packed = tla.squeeze(x, mask)
            dst_tile.store(packed)


def test_squeeze_op_emits_mlir() -> None:
    src, dst = _vector_tensor_pair()
    mlir = squeeze_vector_kernel.dump_mlir(type_args=(src, dst))
    assert "tla.squeeze" in mlir
    assert "tla.create_mask" in mlir
    assert mlir.index("tla.vec.func") < mlir.index("tla.squeeze")
    squeeze_lines = [line for line in mlir.splitlines() if "tla.squeeze" in line]
    assert len(squeeze_lines) == 1
    # tla-to-vector lowering (vsqueeze_* bitcode call) is covered by
    # tests/lit/vector/simd/vector-squeeze-lowering.mlir


def _vector_tensor_pair_for(dtype: object, lanes: int) -> tuple[tla.Tensor, tla.Tensor]:
    return (
        make_fake_tensor(
            dtype,
            (lanes,),
            (1,),
            addrspace=tla.AddressSpace.ub,
            origin_shape=(lanes,),
            layout_tag=tla.arch.RowMajor,
        ),
        make_fake_tensor(
            dtype,
            (lanes,),
            (1,),
            addrspace=tla.AddressSpace.ub,
            origin_shape=(lanes,),
            layout_tag=tla.arch.RowMajor,
        ),
    )


def _make_squeeze_kernel(lanes: int, dtype: object):
    @tla.kernel
    def kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
        src_tile = tla.tile_view(src, tla.make_shape(lanes), tla.make_coord(0))
        dst_tile = tla.tile_view(dst, tla.make_shape(lanes), tla.make_coord(0))
        with tla.vector():
            with tla.vec.func(mode="simd"):
                x = src_tile.load()
                mask = tla.create_mask(pattern=tla.mask.M4, dtype=dtype)
                packed = tla.squeeze(x, mask)
                dst_tile.store(packed)

    return kernel


def test_squeeze_op_integer_dtypes_emit_mlir() -> None:
    for dtype, lanes, vector_ty in (
        (tla.Int8, 256, "256xi8"),
        (tla.Int16, 128, "128xi16"),
        (tla.Int32, 64, "64xi32"),
        (tla.UInt8, 256, "256xui8"),
        (tla.UInt16, 128, "128xui16"),
        (tla.UInt32, 64, "64xui32"),
    ):
        kernel = _make_squeeze_kernel(lanes, dtype)
        src, dst = _vector_tensor_pair_for(dtype, lanes)
        mlir = kernel.dump_mlir(type_args=(src, dst))
        assert "tla.squeeze" in mlir
        assert vector_ty in mlir
        squeeze_lines = [line for line in mlir.splitlines() if "tla.squeeze" in line]
        assert len(squeeze_lines) == 1


def _integer_kernel(dtype, record, *, append=False, stride=0, extra_mask=False):
    @tla.kernel
    def kernel(src: tla.Tensor, dst: tla.Tensor):
        with tla.vector():
            with tla.vec.func(mode="simd"):
                keep = tla.create_mask(pattern=tla.mask.VL8, dtype=dtype)
                packed = tla.squeeze(src.load(), keep, record_nbytes=record)
                if tla.const_expr(append):
                    dst = dst.store(
                        packed,
                        tla.params.UnalignStoreParams(
                            post_mode=tla.params.PostMode.POST_MODE_UPDATE_NBYTES,
                            post_update_stride=stride,
                        ),
                        mask=keep if extra_mask else None,
                    )
                else:
                    dst.store(packed)

    return kernel


def _integer_pair(dtype):
    return _vector_tensor_pair_for(dtype, 2048 // dtype.width)


def test_squeeze_without_byte_count_recording():
    ir = _integer_kernel(tla.Int16, False).dump_mlir(type_args=_integer_pair(tla.Int16))
    assert "tla.squeeze" in ir
    assert "store_reg = true" not in ir


@pytest.mark.parametrize(
    "dtype", [tla.Int8, tla.UInt8, tla.Int16, tla.UInt16, tla.Int32, tla.UInt32]
)
def test_squeeze_append_uses_update_nbytes(dtype):
    ir = _integer_kernel(dtype, True, append=True).dump_mlir(
        type_args=_integer_pair(dtype)
    )
    assert "tla.squeeze" in ir and f"{2048 // dtype.width}x" in ir
    assert "store_reg = true" in ir
    assert "tla.store_unalign_update" in ir
    assert "compressed = true" in ir
    assert "!tla.ureg" in ir
    assert "tla.clear_squeeze_state" not in ir  # Compiler inserts lifecycle ops.


@pytest.mark.parametrize("record", [None, "true", 1])
def test_squeeze_rejects_non_bool_byte_count_flag(record):
    with pytest.raises(
        tla.TlaCoreAPIError, match="record_nbytes must be a bool"
    ):
        _integer_kernel(tla.Int16, record).dump_mlir(type_args=_integer_pair(tla.Int16))


def test_squeeze_append_rejects_a_second_length_source():
    with pytest.raises(Exception, match="recorded byte count"):
        _integer_kernel(
            tla.Int16,
            True,
            append=True,
            stride=1,
        ).dump_mlir(type_args=_integer_pair(tla.Int16))


def test_squeeze_append_rejects_a_second_predicate():
    with pytest.raises(Exception, match="does not accept mask"):
        _integer_kernel(tla.Int16, True, append=True, extra_mask=True).dump_mlir(
            type_args=_integer_pair(tla.Int16)
        )


@tla.kernel
def squeeze_default_update(src: tla.Tensor, dst: tla.Tensor):
    with tla.vector():
        with tla.vec.func(mode="simd"):
            keep = tla.create_mask(pattern=tla.mask.VL8, dtype=tla.Float32)
            packed = tla.squeeze(src.load(), keep)
            dst = dst.store(
                packed,
                tla.params.UnalignStoreParams(
                    post_mode=tla.params.PostMode.POST_MODE_UPDATE
                ),
            )


def test_squeeze_default_keeps_ordinary_update_length():
    ir = squeeze_default_update.dump_mlir(type_args=_vector_tensor_pair())
    assert "tla.store_unalign_update" in ir
    assert "store_reg = true" not in ir
    assert "arith.constant 64" in ir


def _branch_squeeze_kernel(post_mode, *, add_after_merge=False):
    @tla.kernel
    def kernel(src: tla.Tensor, dst: tla.Tensor, condition: tla.Int32):
        with tla.vector():
            with tla.vec.func(mode="simd"):
                values = src.load()
                keep = tla.create_mask(pattern=tla.mask.VL8, dtype=tla.Float16)
                packed = values
                if condition > 0:
                    packed = tla.squeeze(values, keep, record_nbytes=True)
                else:
                    packed = tla.squeeze(values + values, keep, record_nbytes=True)
                if tla.const_expr(add_after_merge):
                    packed = packed + values
                dst.store(
                    packed,
                    tla.params.UnalignStoreParams(
                        post_mode=post_mode
                    ),
                )

    return kernel


@pytest.mark.parametrize(
    "post_mode",
    [tla.params.PostMode.POST_MODE_UPDATE, tla.params.PostMode.POST_MODE_UPDATE_NBYTES],
)
@pytest.mark.parametrize("add_after_merge", [False, True])
def test_unaligned_store_uses_selected_mode_after_if(post_mode, add_after_merge):
    kernel = _branch_squeeze_kernel(post_mode, add_after_merge=add_after_merge)
    ir = kernel.dump_mlir(type_args=(*_integer_pair(tla.Float16), tla.Int32(0)))
    assert "scf.if" in ir
    assert "tla.store_unalign_update" in ir
    compressed = post_mode == tla.params.PostMode.POST_MODE_UPDATE_NBYTES
    assert ("compressed = true" in ir) is compressed


def test_update_nbytes_requires_unaligned_store():
    @tla.kernel
    def kernel(src: tla.Tensor, dst: tla.Tensor):
        with tla.vector():
            with tla.vec.func(mode="simd"):
                dst.store(
                    src.load(),
                    tla.params.NormalStoreParams(
                        post_mode=tla.params.PostMode.POST_MODE_UPDATE_NBYTES
                    ),
                )

    with pytest.raises(TlaLoweringError, match="requires UnalignStoreParams"):
        kernel.dump_mlir(type_args=_vector_tensor_pair())
