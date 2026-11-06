from __future__ import annotations

from typing import Any

import pytest

import catlass.tla as tla
from catlass.execution_lowering import UnsupportedExecutionLowering
from catlass.tla.runtime import make_fake_tensor


def _vector_tensor(dtype: type[tla.Numeric] = tla.Float32, shape: int = 64) -> tla.Tensor:
    return make_fake_tensor(
        dtype,
        (shape,),
        (1,),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(shape,),
        layout_tag=tla.arch.RowMajor,
    )


@tla.kernel
def vector_reduce_kernel(src: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            src_reg = src_tile.load()
            reduce_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            _ = src_reg.reduce(tla.ReductionOp.ADD, mask=reduce_mask)
            _ = src_reg.reduce(tla.ReductionOp.MAX, mask=reduce_mask)
            _ = src_reg.reduce(tla.ReductionOp.MIN, mask=reduce_mask)


@tla.kernel
def vector_reduce_i64_kernel(src: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(32), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            src_reg = src_tile.load()
            reduce_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Int64)
            _ = src_reg.reduce(tla.ReductionOp.ADD, mask=reduce_mask)



@tla.kernel
def vector_masked_reduce_kernel(src: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            mask = tla.create_mask(pattern=tla.mask.H)
            src_reg = src_tile.load()
            _ = src_reg.reduce(tla.ReductionOp.ADD, mask=mask)


@tla.kernel
def vector_unsigned_reduce_kernel(src: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            reduce_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.UInt16)
            _ = src_tile.load().reduce(tla.ReductionOp.ADD, mask=reduce_mask)


@tla.kernel
def reduce_non_reduction_op_kernel(src: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            reduce_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            _ = src_tile.load().reduce("add", mask=reduce_mask)


@tla.kernel
def reduce_init_value_keyword_kernel(src: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            reduce_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            _ = src_tile.load().reduce(tla.ReductionOp.ADD, mask=reduce_mask, init_value=0.0)


@tla.kernel
def reduce_profile_keyword_kernel(src: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            reduce_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            _ = src_tile.load().reduce(tla.ReductionOp.ADD, mask=reduce_mask, reduction_profile=0)


@tla.kernel
def reduce_none_keywords_kernel(src: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            reduce_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            _ = src_tile.load().reduce(
                tla.ReductionOp.ADD,
                mask=reduce_mask,
                init_value=None,
                reduction_profile=None,
            )


@tla.kernel
def vector_reduce_no_mask_kernel(src: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            _ = src_tile.load().reduce(tla.ReductionOp.ADD)


def test_vector_reduce_public_export_exists() -> None:
    assert [op.value for op in tla.ReductionOp] == ["add", "max", "min"]
    assert [mode.value for mode in tla.ReduceMode] == ["intra_block", "inter_block"]


def test_vector_reduce_emits_supported_kinds() -> None:
    mlir = vector_reduce_kernel.dump_mlir(type_args=(_vector_tensor(),))

    assert mlir.count("tla.reduce") == 3
    for kind in ("add", "max", "min"):
        assert f'kind = "{kind}"' in mlir


def test_vector_reduce_emits_mask() -> None:
    mlir = vector_masked_reduce_kernel.dump_mlir(type_args=(_vector_tensor(),))

    reduce_line = next(line for line in mlir.splitlines() if "tla.reduce" in line)
    assert "tla.reduce" in reduce_line
    assert "!tla.mask<64>" in reduce_line
    assert 'kind = "add"' in reduce_line


def test_unsigned_reduction_has_no_init_profile_attrs() -> None:
    mlir = vector_unsigned_reduce_kernel.dump_mlir(type_args=(_vector_tensor(tla.UInt16),))

    assert "!tla.ptr<ui16" in mlir
    assert "tla.reduce" in mlir
    assert "init_value" not in mlir
    assert "reduction_profile" not in mlir


def test_reduce_accepts_none_semantic_keywords() -> None:
    mlir = reduce_none_keywords_kernel.dump_mlir(type_args=(_vector_tensor(),))

    assert "tla.reduce" in mlir
    assert "init_value" not in mlir
    assert "reduction_profile" not in mlir


def test_unsupported_reduction_element_type_is_rejected() -> None:
    with pytest.raises(
        tla.TlaCoreAPIError,
        match=r"VectorSSA\.reduce.*unsupported reduction element type i64",
    ):
        vector_reduce_i64_kernel.dump_mlir(type_args=(_vector_tensor(tla.Int64, 32),))


def test_reduce_rejects_non_reduction_op() -> None:
    with pytest.raises(
        (tla.TlaCoreAPIError, UnsupportedExecutionLowering), match="expected ReductionOp"
    ):
        reduce_non_reduction_op_kernel.dump_mlir(type_args=(_vector_tensor(),))


@pytest.mark.parametrize(
    "kernel, match",
    (
        (reduce_init_value_keyword_kernel, "only supports init_value=None"),
        (
            reduce_profile_keyword_kernel,
            "only supports reduction_profile=None",
        ),
    ),
)
def test_reduce_hides_unsupported_semantic_keywords(
    kernel: Any,
    match: str,
) -> None:
    with pytest.raises(UnsupportedExecutionLowering, match=match):
        kernel.dump_mlir(type_args=(_vector_tensor(),))


def test_reduce_requires_explicit_mask() -> None:
    with pytest.raises(
        UnsupportedExecutionLowering,
        match=r"missing.*keyword-only.*argument.*'mask'",
    ):
        vector_reduce_no_mask_kernel.dump_mlir(type_args=(_vector_tensor(),))


# --- unsigned reduce signed-twin guard (u16 65535 vs 1) --------------------
# Device behavior is covered by lit/vector/simd/unsigned-casts.mlir
# (ave.hir.reduction <umax>/<umin>). Keep a card-free frontend check here.


_U16_N = 128


@tla.kernel
def u16_reduce_max_min_kernel(
    src: tla.Tensor, out_max: tla.Tensor, out_min: tla.Tensor
) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(_U16_N), tla.make_coord(0))
    max_tile = tla.tile_view(out_max, tla.make_shape(1), tla.make_coord(0))
    min_tile = tla.tile_view(out_min, tla.make_shape(1), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            va = src_tile.load()
            mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.UInt16)
            max_tile.store(va.reduce(tla.ReductionOp.MAX, mask=mask))
            min_tile.store(va.reduce(tla.ReductionOp.MIN, mask=mask))


def test_u16_reduce_max_min_keeps_unsigned_vector_type() -> None:
    """Signed twin would flip reduce(65535,1); frontend must keep ui16 on reduce."""
    src = make_fake_tensor(
        tla.UInt16,
        (_U16_N,),
        (1,),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(_U16_N,),
        layout_tag=tla.arch.RowMajor,
    )
    out = make_fake_tensor(
        tla.UInt16,
        (1,),
        (1,),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(1,),
        layout_tag=tla.arch.RowMajor,
    )
    mlir = u16_reduce_max_min_kernel.dump_mlir(type_args=(src, out, out))
    assert mlir.count("tla.reduce") == 2
    assert 'kind = "max"' in mlir
    assert 'kind = "min"' in mlir
    assert "!tla.vector<128xui16>" in mlir
    assert "!tla.vector<1xui16>" in mlir


# --- intra-block reduce (mode=ReduceMode.INTRA_BLOCK) ------------------------
# Each 32-byte data block reduces independently; the per-block results land
# packed in the least-significant lanes. The frontend dispatches through
# VectorSSA.reduce(mode=INTRA_BLOCK), which still emits the separate
# tla.reduce_intra_block IR op.


@tla.kernel
def vector_reduce_intra_block_kernel(src: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            src_reg = src_tile.load()
            reduce_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            _ = src_reg.reduce(tla.ReductionOp.ADD, mode=tla.ReduceMode.INTRA_BLOCK, mask=reduce_mask)
            _ = src_reg.reduce(tla.ReductionOp.MAX, mode=tla.ReduceMode.INTRA_BLOCK, mask=reduce_mask)
            _ = src_reg.reduce(tla.ReductionOp.MIN, mode=tla.ReduceMode.INTRA_BLOCK, mask=reduce_mask)


@tla.kernel
def vector_reduce_intra_block_i32_kernel(src: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            reduce_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Int32)
            _ = src_tile.load().reduce(tla.ReductionOp.ADD, mode=tla.ReduceMode.INTRA_BLOCK, mask=reduce_mask)


@tla.kernel
def vector_reduce_intra_block_u16_kernel(src: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(128), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            reduce_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.UInt16)
            _ = src_tile.load().reduce(tla.ReductionOp.ADD, mode=tla.ReduceMode.INTRA_BLOCK, mask=reduce_mask)


@tla.kernel
def vector_reduce_intra_block_i16_single_lane_kernel(src: tla.Tensor) -> None:
    # Tail tile with logical shape 1: the only valid element must still get
    # one widened i32 result lane, not fail lane derivation (floor // 2 -> 0).
    src_tile = tla.tile_view(src, tla.make_shape(1), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            reduce_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Int16)
            _ = src_tile.load().reduce(tla.ReductionOp.ADD, mode=tla.ReduceMode.INTRA_BLOCK, mask=reduce_mask)


@tla.kernel
def vector_reduce_intra_block_i16_odd_lanes_kernel(src: tla.Tensor) -> None:
    # Odd tail: the last valid element's widened lane must survive the
    # rounding (ceil), so 9 valid i16 lanes -> 5 i32 result lanes.
    src_tile = tla.tile_view(src, tla.make_shape(9), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            reduce_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Int16)
            _ = src_tile.load().reduce(tla.ReductionOp.ADD, mode=tla.ReduceMode.INTRA_BLOCK, mask=reduce_mask)


@tla.kernel
def vector_reduce_intra_block_i8_kernel(src: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(256), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            reduce_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Int8)
            _ = src_tile.load().reduce(tla.ReductionOp.ADD, mode=tla.ReduceMode.INTRA_BLOCK, mask=reduce_mask)


@tla.kernel
def vector_reduce_intra_block_i64_kernel(src: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(32), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            reduce_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Int64)
            _ = src_tile.load().reduce(tla.ReductionOp.ADD, mode=tla.ReduceMode.INTRA_BLOCK, mask=reduce_mask)


@tla.kernel
def vector_reduce_intra_block_bf16_kernel(src: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(128), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            reduce_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.BFloat16)
            _ = src_tile.load().reduce(tla.ReductionOp.ADD, mode=tla.ReduceMode.INTRA_BLOCK, mask=reduce_mask)


@tla.kernel
def reduce_intra_block_non_reduction_op_kernel(src: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            reduce_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            _ = src_tile.load().reduce("add", mode=tla.ReduceMode.INTRA_BLOCK, mask=reduce_mask)


@tla.kernel
def reduce_intra_block_no_mask_kernel(src: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            _ = src_tile.load().reduce(tla.ReductionOp.ADD, mode=tla.ReduceMode.INTRA_BLOCK)


@tla.kernel
def _reduce_intra_block_in_simt(src: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simt"):
            src_reg = src_tile.load()
            mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            _ = src_reg.reduce(tla.ReductionOp.ADD, mode=tla.ReduceMode.INTRA_BLOCK, mask=mask)


def test_vector_reduce_intra_block_emits_supported_kinds() -> None:
    mlir = vector_reduce_intra_block_kernel.dump_mlir(type_args=(_vector_tensor(),))

    assert mlir.count("tla.reduce_intra_block") == 3
    for kind in ("add", "max", "min"):
        assert f'kind = "{kind}"' in mlir


def test_vector_reduce_intra_block_i32_and_u16() -> None:
    i32_mlir = vector_reduce_intra_block_i32_kernel.dump_mlir(type_args=(_vector_tensor(tla.Int32),))
    assert "tla.reduce_intra_block" in i32_mlir
    assert 'kind = "add"' in i32_mlir
    assert "!tla.vector<64xi32>" in i32_mlir

    u16_mlir = vector_reduce_intra_block_u16_kernel.dump_mlir(
        type_args=(_vector_tensor(tla.UInt16, 128),)
    )
    assert "tla.reduce_intra_block" in u16_mlir
    assert 'kind = "add"' in u16_mlir
    # ADD on u16 widens: the result is the 64-lane ui32 instruction dst.
    assert "!tla.vector<64xui32>" in u16_mlir


def test_vector_reduce_intra_block_i16_tail_lanes() -> None:
    # Single-lane tail: lane derivation must stay positive (ceil, not floor).
    single_mlir = vector_reduce_intra_block_i16_single_lane_kernel.dump_mlir(
        type_args=(_vector_tensor(tla.Int16, 128),)
    )
    assert "!tla.vector<1xi32>" in single_mlir

    # Odd tail: ceil keeps the last valid element's widened lane.
    odd_mlir = vector_reduce_intra_block_i16_odd_lanes_kernel.dump_mlir(
        type_args=(_vector_tensor(tla.Int16, 128),)
    )
    assert "!tla.vector<9xi16>" in odd_mlir
    assert "!tla.vector<5xi32>" in odd_mlir


def test_reduce_intra_block_unsupported_element_types_rejected() -> None:
    with pytest.raises(
        tla.TlaCoreAPIError,
        match=r"VectorSSA\.reduce: unsupported reduce_intra_block element type i8",
    ):
        vector_reduce_intra_block_i8_kernel.dump_mlir(type_args=(_vector_tensor(tla.Int8, 256),))

    with pytest.raises(
        tla.TlaCoreAPIError,
        match=r"VectorSSA\.reduce: unsupported reduce_intra_block element type i64",
    ):
        vector_reduce_intra_block_i64_kernel.dump_mlir(type_args=(_vector_tensor(tla.Int64, 32),))

    with pytest.raises(
        tla.TlaCoreAPIError,
        match=r"VectorSSA\.reduce: unsupported reduce_intra_block element type bf16",
    ):
        vector_reduce_intra_block_bf16_kernel.dump_mlir(
            type_args=(_vector_tensor(tla.BFloat16, 128),)
        )


def test_reduce_intra_block_rejects_non_reduction_op() -> None:
    with pytest.raises(
        (tla.TlaCoreAPIError, UnsupportedExecutionLowering), match="expected ReductionOp"
    ):
        reduce_intra_block_non_reduction_op_kernel.dump_mlir(type_args=(_vector_tensor(),))


def test_reduce_intra_block_requires_explicit_mask() -> None:
    with pytest.raises(
        UnsupportedExecutionLowering,
        match=r"missing.*keyword-only.*argument.*'mask'",
    ):
        reduce_intra_block_no_mask_kernel.dump_mlir(type_args=(_vector_tensor(),))


def test_reduce_intra_block_rejects_simt_mode() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="only supported.*simd"):
        _reduce_intra_block_in_simt.dump_mlir(type_args=(_vector_tensor(),))
