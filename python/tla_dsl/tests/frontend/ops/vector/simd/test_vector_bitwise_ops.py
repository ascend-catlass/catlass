from __future__ import annotations

from catlass.tla.runtime import make_fake_tensor


import inspect
import re
from pathlib import Path

import pytest

import catlass.tla as tla
import catlass.runtime as runtime_mod  # noqa: F401
from catlass._mlir.dialects import tla as tla_ops_gen

_REG_BITWISE_DTYPE: type[tla.Numeric] = tla.Int32


@tla.kernel
def _mask_bitwise_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            reg = src_tile.load()
            zero = tla.sub(reg, reg)
            all_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            h_mask = tla.create_mask(pattern=tla.mask.H, dtype=tla.Float32)
            q_mask = tla.create_mask(pattern=tla.mask.Q, dtype=tla.Float32)
            m4_mask = tla.create_mask(pattern=tla.mask.M4, dtype=tla.Float32)

            not_mask = tla.bitwise_not(q_mask)
            and_mask = tla.bitwise_and(h_mask, m4_mask)
            or_mask = tla.bitwise_or(q_mask, m4_mask, mask=all_mask)
            xor_mask = tla.bitwise_xor(h_mask, m4_mask, mask=all_mask)

            tmp0 = tla.where(not_mask, reg, zero)
            tmp1 = tla.where(and_mask, tmp0, zero)
            tmp2 = tla.where(or_mask, tmp1, zero)
            dst_tile.store(tla.where(xor_mask, tmp2, zero), mask=all_mask)


_REG_BITWISE_LANES = 64


@tla.kernel
def _vector_ssa_bitwise_unary_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(_REG_BITWISE_LANES), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(_REG_BITWISE_LANES), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            reg = src_tile.load()
            all_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=_REG_BITWISE_DTYPE)
            dst_tile.store(tla.bitwise_not(reg), mask=all_mask)


@tla.kernel
def _vector_ssa_bitwise_binary_kernel(
    src0: tla.Tensor, src1: tla.Tensor, dst: tla.Tensor
) -> None:
    src0_tile = tla.tile_view(src0, tla.make_shape(_REG_BITWISE_LANES), tla.make_coord(0))
    src1_tile = tla.tile_view(src1, tla.make_shape(_REG_BITWISE_LANES), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(_REG_BITWISE_LANES), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            reg0 = src0_tile.load()
            reg1 = src1_tile.load()
            all_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=_REG_BITWISE_DTYPE)
            and_reg = tla.bitwise_and(reg0, reg1)
            or_reg = tla.bitwise_or(and_reg, reg0, mask=all_mask)
            dst_tile.store(tla.bitwise_xor(or_reg, reg1), mask=all_mask)

@tla.kernel
def _unsigned_binary_kernel(src0: tla.Tensor, src1: tla.Tensor, dst: tla.Tensor) -> None:
    src0_tile = tla.tile_view(src0, tla.make_shape(64), tla.make_coord(0))
    src1_tile = tla.tile_view(src1, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            lhs = src0_tile.load()
            rhs = src1_tile.load()
            mask = tla.create_mask(pattern=tla.mask.ALL, dtype=_REG_BITWISE_DTYPE)
            result = _UNSIGNED_BITWISE_OP(lhs, rhs, mask=mask)
            dst_tile.store(result, mask=mask)


_UNSIGNED_BITWISE_OP = tla.bitwise_and


@tla.kernel
def _mixed_signed_unsigned_kernel(src0: tla.Tensor, src1: tla.Tensor, dst: tla.Tensor) -> None:
    src0_tile = tla.tile_view(src0, tla.make_shape(64), tla.make_coord(0))
    src1_tile = tla.tile_view(src1, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            lhs = src0_tile.load()
            rhs = src1_tile.load()
            dst_tile.store(tla.bitwise_and(lhs, rhs))


def _tensor_args(
    dtype: type[tla.Numeric] = tla.Float32, count: int = 2, lanes: int = 64
) -> tuple[tla.Tensor, ...]:
    return tuple(
        make_fake_tensor(
            dtype,
            (lanes,),
            (1,),
            addrspace=tla.AddressSpace.ub,
            origin_shape=(lanes,),
            layout_tag=tla.arch.RowMajor,
        )
        for _ in range(count)
    )


@pytest.mark.parametrize(
    ("binding_name", "op_name"),
    (
        ("bitwise_not", "tla.bitwise_not"),
        ("bitwise_and", "tla.bitwise_and"),
        ("bitwise_or", "tla.bitwise_or"),
        ("bitwise_xor", "tla.bitwise_xor"),
    ),
)
def test_mask_bitwise_bindings_and_public_ops_emit_mlir(
    binding_name: str, op_name: str
) -> None:
    assert hasattr(tla_ops_gen, binding_name)
    assert op_name in _mask_bitwise_kernel.dump_mlir(type_args=_tensor_args())


@pytest.mark.parametrize(
    ("op", "required_operand_count"),
    (
        (tla.bitwise_not, 1),
        (tla.bitwise_and, 2),
        (tla.bitwise_or, 2),
        (tla.bitwise_xor, 2),
    ),
)
def test_bitwise_mask_is_optional_keyword_only(
    op: object, required_operand_count: int
) -> None:
    mask_parameter = inspect.signature(op).parameters["mask"]
    assert mask_parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert mask_parameter.default is None
    positional_args = [object()] * (required_operand_count + 1)
    with pytest.raises(TypeError, match="positional"):
        op(*positional_args)


_REG_BITWISE_UNARY_DTYPES = (
    tla.Float16, tla.Float32, tla.Int8, tla.Int16, tla.Int32, tla.Int64,
    tla.UInt8, tla.UInt16, tla.UInt32, tla.UInt64,
)
_REG_BITWISE_BINARY_DTYPES = (
    tla.Float16,
    tla.BFloat16,
    tla.Bool,
    tla.Int64,
    tla.UInt64,
    tla.Float32,
    tla.Int32,
    tla.Int16,
    tla.Int8,
)
_REGTENSOR_BITWISE_CASES = tuple(
    ("bitwise_not", "tla.bitwise_not", dtype, _vector_ssa_bitwise_unary_kernel, 2)
    for dtype in _REG_BITWISE_UNARY_DTYPES
) + tuple(
    (binding_name, op_name, dtype, _vector_ssa_bitwise_binary_kernel, 3)
    for binding_name, op_name in (
        ("bitwise_and", "tla.bitwise_and"),
        ("bitwise_or", "tla.bitwise_or"),
        ("bitwise_xor", "tla.bitwise_xor"),
    )
    for dtype in _REG_BITWISE_BINARY_DTYPES
)


@pytest.mark.parametrize(
    ("binding_name", "op_name", "dtype", "kernel", "tensor_count"),
    _REGTENSOR_BITWISE_CASES,
)
def test_vector_ssa_bitwise_bindings_and_public_ops_emit_mlir(
    binding_name: str,
    op_name: str,
    dtype: type[tla.Numeric],
    kernel: object,
    tensor_count: int,
) -> None:
    global _REG_BITWISE_DTYPE, _REG_BITWISE_LANES
    _REG_BITWISE_DTYPE = dtype
    _REG_BITWISE_LANES = 32 if dtype.width == 64 else 64
    assert hasattr(tla_ops_gen, binding_name)
    assert op_name in kernel.dump_mlir(type_args=_tensor_args(dtype, tensor_count, _REG_BITWISE_LANES))

@pytest.mark.parametrize(
    ("dtype", "element_type"),
    ((tla.UInt8, "ui8"), (tla.UInt16, "ui16"), (tla.UInt32, "ui32")),
)
@pytest.mark.parametrize(
    ("op", "op_name"),
    ((tla.bitwise_and, "and"), (tla.bitwise_or, "or"), (tla.bitwise_xor, "xor")),
)
def test_unsigned_vector_bitwise_binary_emits_typed_mlir(
    dtype: type[tla.Numeric], element_type: str, op: object, op_name: str
) -> None:
    global _REG_BITWISE_DTYPE, _UNSIGNED_BITWISE_OP
    _REG_BITWISE_DTYPE = dtype
    _UNSIGNED_BITWISE_OP = op
    asm = _unsigned_binary_kernel.dump_mlir(type_args=_tensor_args(dtype, 3))
    vector_type = rf"!tla\.vector<64x{element_type}>"
    lanes = 256 // (int(element_type[2:]) // 8)
    assert re.search(
        rf"tla\.bitwise_{op_name} %\w+, %\w+ mask %\w+ : "
        rf"{vector_type},\s*{vector_type} mask !tla\.mask<{lanes}> -> {vector_type}",
        asm,
    )


def test_bitwise_and_rejects_mixed_signed_unsigned_elements() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="src1_reg has element type.*expected"):
        _mixed_signed_unsigned_kernel.dump_mlir(
            type_args=(
                *_tensor_args(tla.UInt32, 1),
                *_tensor_args(tla.Int32, 1),
                *_tensor_args(tla.UInt32, 1),
            )
        )

def test_bitwise_ops_are_in_vector_lowering_info() -> None:
    # The vector lowering is split across TlaVectorRegion*.cpp (driver, common,
    # SIMD, SIMT), so search all of them rather than naming one file: which file
    # a given op is handled in is not what this test is about.
    pass_dir = Path(__file__).parents[5] / "csrc/mlir/lib/Passes"
    sources = sorted(pass_dir.glob("TlaVectorRegion*.cpp"))
    assert sources, f"no vector-region pass sources found under {pass_dir}"
    source = "\n".join(path.read_text(encoding="utf-8") for path in sources)

    for op_name in ("BitwiseNotOp", "BitwiseAndOp", "BitwiseOrOp", "BitwiseXorOp"):
        assert f"::tla::{op_name}" in source


def test_mask_bitwise_rejects_non_mask_predicate() -> None:
    with pytest.raises(tla.TlaCoreAPIError, match="tla.bitwise_not.*mask"):
        _invalid_mask_bitwise_kernel.dump_mlir(type_args=_tensor_args())


@tla.kernel
def _invalid_mask_bitwise_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            reg = src_tile.load()
            all_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            dst_tile.store(reg, mask=tla.bitwise_not(all_mask, mask=reg))


def test_vector_ssa_bitwise_rejects_mixed_vector_and_mask() -> None:
    with pytest.raises(
        tla.TlaCoreAPIError, match="both be MaskSSA values or both be VectorSSA"
    ):
        _invalid_mixed_bitwise_kernel.dump_mlir(type_args=_tensor_args(tla.Int32, 2))


@tla.kernel
def _invalid_mixed_bitwise_kernel(src: tla.Tensor, dst: tla.Tensor) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            reg = src_tile.load()
            all_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Int32)
            bad = tla.bitwise_and(reg, all_mask, mask=all_mask)
            dst_tile.store(bad, mask=all_mask)
