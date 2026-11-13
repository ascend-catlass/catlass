from __future__ import annotations

from catlass.tla.runtime import make_fake_tensor


import pytest

import catlass.tla as tla
import catlass.runtime as runtime_mod  # noqa: F401
from catlass.core_api import MaskSSA


def _vector_tensor(shape: int = 64, dtype: object = tla.Float32) -> tla.Tensor:
    return make_fake_tensor(
        dtype,
        (shape,),
        (1,),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(shape,),
        layout_tag=tla.arch.RowMajor,
    )


@tla.kernel
def cmp_kernel(lhs: tla.Tensor, rhs: tla.Tensor) -> None:
    lhs_tile = tla.tile_view(lhs, tla.make_shape(64), tla.make_coord(0))
    rhs_tile = tla.tile_view(rhs, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            lhs_reg = lhs_tile.load()
            rhs_reg = rhs_tile.load()
            assert isinstance(tla.cmp(lhs_reg, rhs_reg, "lt"), MaskSSA)
            assert isinstance(tla.cmp(lhs_reg, rhs_reg, "le"), MaskSSA)
            assert isinstance(tla.cmp(lhs_reg, rhs_reg, "gt"), MaskSSA)
            assert isinstance(tla.cmp(lhs_reg, rhs_reg, "ge"), MaskSSA)


@tla.kernel
def scalar_cmp_kernel(lhs: tla.Tensor) -> None:
    lhs_tile = tla.tile_view(lhs, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            lhs_reg = lhs_tile.load()
            assert isinstance(tla.cmp(lhs_reg, 0.0, "gt"), MaskSSA)


@tla.kernel
def scalar_cmp_i64_kernel(lhs: tla.Tensor) -> None:
    lhs_tile = tla.tile_view(lhs, tla.make_shape(32), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            lhs_reg = lhs_tile.load()
            _ = tla.cmp(lhs_reg, 0, "gt")


@tla.kernel
def masked_add_kernel(lhs: tla.Tensor, rhs: tla.Tensor, dst: tla.Tensor) -> None:
    lhs_tile = tla.tile_view(lhs, tla.make_shape(64), tla.make_coord(0))
    rhs_tile = tla.tile_view(rhs, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            lhs_reg = lhs_tile.load()
            rhs_reg = rhs_tile.load()
            mask = tla.cmp(lhs_reg, rhs_reg, "lt")
            dst_tile.store(tla.add(lhs_reg, rhs_reg, mask=mask), mask=mask)


@tla.kernel
def masked_cmp_kernel(lhs: tla.Tensor, rhs: tla.Tensor) -> None:
    lhs_tile = tla.tile_view(lhs, tla.make_shape(64), tla.make_coord(0))
    rhs_tile = tla.tile_view(rhs, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            lhs_reg = lhs_tile.load()
            rhs_reg = rhs_tile.load()
            active = tla.create_mask(pattern=tla.mask.H)
            assert isinstance(tla.cmp(lhs_reg, rhs_reg, "lt", mask=active), MaskSSA)


@tla.kernel
def static_vs_dynamic_tile_cmp_kernel(mem: tla.Tensor) -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            static_tile = tla.tile_view(mem, tla.make_shape(64), tla.make_coord(0))
            ref = static_tile.load()
            for i in tla.range(4):
                dynamic_tile = tla.tile_view(mem, tla.make_shape(64), tla.make_coord(i))
                _ = tla.cmp(ref, dynamic_tile.load(), "lt")


@tla.kernel
def full_tensor_vs_tile_view_cmp_kernel(full: tla.Tensor, other: tla.Tensor) -> None:
    full_tile = tla.tile_view(full, tla.make_shape(64), tla.make_coord(0))
    other_tile = tla.tile_view(other, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            _ = tla.cmp(full_tile.load(), other_tile.load(), "lt")


@tla.kernel
def mask_widths_kernel() -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            _ = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Int64)
            _ = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            _ = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float16)
            _ = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Int8)


@tla.kernel
def mismatched_mask_add_kernel(
    lhs: tla.Tensor, rhs: tla.Tensor, dst: tla.Tensor
) -> None:
    lhs_tile = tla.tile_view(lhs, tla.make_shape(64), tla.make_coord(0))
    rhs_tile = tla.tile_view(rhs, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            lhs_reg = lhs_tile.load()
            rhs_reg = rhs_tile.load()
            mask = tla.create_mask(pattern=tla.mask.H, dtype=tla.Float16)
            dst_tile.store(tla.add(lhs_reg, rhs_reg, mask=mask))


@tla.kernel
def mismatched_mask_bitwise_kernel() -> None:
    with tla.vector():
        with tla.vec.func(mode="simd"):
            mask64 = tla.create_mask(pattern=tla.mask.H, dtype=tla.Float32)
            mask128 = tla.create_mask(pattern=tla.mask.H, dtype=tla.Float16)
            _ = tla.bitwise_or(mask64, mask128)


def test_cmp_symbol_is_exported() -> None:
    assert callable(tla.cmp)


def _assert_cmp_mode(mlir: str, mode: str) -> None:
    assert f'mode = "{mode}"' in mlir or f'tla.cmp "{mode}"' in mlir


def test_cmp_api_emits_cmp_and_returns_mask() -> None:
    tensor = _vector_tensor()
    mlir = cmp_kernel.dump_mlir(type_args=(tensor, tensor))

    assert mlir.count("tla.cmp") == 4
    for mode in ("lt", "le", "gt", "ge"):
        _assert_cmp_mode(mlir, mode)


def test_cmp_accepts_scalar_rhs() -> None:
    tensor = _vector_tensor()
    mlir = scalar_cmp_kernel.dump_mlir(type_args=(tensor,))

    _assert_cmp_mode(mlir, "gt")
    assert "!tla.mask<64>" in mlir


def test_cmp_accepts_input_mask() -> None:
    tensor = _vector_tensor()
    mlir = masked_cmp_kernel.dump_mlir(type_args=(tensor, tensor))

    _assert_cmp_mode(mlir, "lt")
    assert "mask" in mlir


def test_cmp_result_can_mask_vector_op_and_store() -> None:
    tensor = _vector_tensor()
    mlir = masked_add_kernel.dump_mlir(type_args=(tensor, tensor, tensor))

    assert mlir.count("tla.cmp") == 1
    assert "tla.add" in mlir
    assert "tla.store" in mlir


def test_create_mask_encodes_physical_predicate_lanes() -> None:
    mlir = mask_widths_kernel.dump_mlir()

    for lanes in (32, 64, 128, 256):
        assert f"!tla.mask<{lanes}>" in mlir


def test_masked_vector_op_rejects_mismatched_predicate_lanes() -> None:
    tensor = _vector_tensor()
    with pytest.raises(
        Exception,
        match=r"mask has 128 predicate lanes, expected 64 for f32 VectorSSA",
    ):
        mismatched_mask_add_kernel.dump_mlir(type_args=(tensor, tensor, tensor))


def test_mask_bitwise_rejects_mismatched_mask_types() -> None:
    with pytest.raises(
        Exception,
        match=r"src1_reg has type !tla\.mask<128>, expected !tla\.mask<64>",
    ):
        mismatched_mask_bitwise_kernel.dump_mlir()


def test_cmp_rejects_unsupported_dtype() -> None:
    tensor = _vector_tensor(shape=32, dtype=tla.Int64)
    with pytest.raises(Exception, match="unsupported compare element type"):
        scalar_cmp_i64_kernel.dump_mlir(type_args=(tensor,))


@pytest.mark.parametrize(
    "dtype,shape", [(tla.UInt8, 256), (tla.UInt16, 128), (tla.UInt32, 64)]
)
def test_cmp_accepts_unsigned_element_types(dtype, shape: int) -> None:
    tensor = _vector_tensor(shape=shape, dtype=dtype)
    mlir = cmp_kernel.dump_mlir(type_args=(tensor, tensor))
    assert mlir.count("tla.cmp") == 4


def test_cmp_accepts_static_tile_with_dynamic_tile_view() -> None:
    tensor = _vector_tensor(shape=256)
    mlir = static_vs_dynamic_tile_cmp_kernel.dump_mlir(type_args=(tensor,))

    assert mlir.count("tla.cmp") == 1
    _assert_cmp_mode(mlir, "lt")
    assert "scf.for" in mlir


def test_cmp_accepts_two_tensor_tile_views() -> None:
    tensor = _vector_tensor()
    mlir = full_tensor_vs_tile_view_cmp_kernel.dump_mlir(type_args=(tensor, tensor))

    assert mlir.count("tla.cmp") == 1
    _assert_cmp_mode(mlir, "lt")
