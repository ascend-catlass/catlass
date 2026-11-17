"""Tests for tla.squeeze mask-compress vector op."""

from __future__ import annotations

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
