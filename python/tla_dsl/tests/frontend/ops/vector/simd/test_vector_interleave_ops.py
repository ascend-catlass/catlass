from __future__ import annotations

from catlass.tla.runtime import make_fake_tensor


import pytest
from catlass._mlir import ir as mlir_ir

import catlass.tla as tla
import catlass.runtime as runtime_mod  # noqa: F401
from catlass.base_dsl import BaseDSL


def _vector_tensor(
    dtype: type[tla.Numeric] = tla.Float32,
    size: int = 64,
) -> tla.Tensor:
    return make_fake_tensor(
        dtype,
        (size,),
        (1,),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(size,),
        layout_tag=tla.arch.RowMajor,
    )


@tla.kernel
def vector_interleave_kernel(
    src0: tla.Tensor,
    src1: tla.Tensor,
    dst0: tla.Tensor,
    dst1: tla.Tensor,
) -> None:
    src0_tile = tla.tile_view(src0, tla.make_shape(64), tla.make_coord(0))
    src1_tile = tla.tile_view(src1, tla.make_shape(64), tla.make_coord(0))
    dst0_tile = tla.tile_view(dst0, tla.make_shape(64), tla.make_coord(0))
    dst1_tile = tla.tile_view(dst1, tla.make_shape(64), tla.make_coord(0))

    with tla.vector():
        with tla.vec.func(mode="simd"):
            out0, out1 = tla.interleave(src0_tile.load(), src1_tile.load())
            dst0_tile.store(out0)
            dst1_tile.store(out1)


@tla.kernel
def vector_deinterleave_kernel(
    src0: tla.Tensor,
    src1: tla.Tensor,
    dst0: tla.Tensor,
    dst1: tla.Tensor,
) -> None:
    src0_tile = tla.tile_view(src0, tla.make_shape(64), tla.make_coord(0))
    src1_tile = tla.tile_view(src1, tla.make_shape(64), tla.make_coord(0))
    dst0_tile = tla.tile_view(dst0, tla.make_shape(64), tla.make_coord(0))
    dst1_tile = tla.tile_view(dst1, tla.make_shape(64), tla.make_coord(0))

    with tla.vector():
        with tla.vec.func(mode="simd"):
            out0, out1 = tla.deinterleave(src0_tile.load(), src1_tile.load())
            dst0_tile.store(out0)
            dst1_tile.store(out1)


@tla.kernel
def vector_interleave_distinct_coords_kernel(
    src: tla.Tensor,
    dst: tla.Tensor,
) -> None:
    src0_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    src1_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(1))
    dst0_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    dst1_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(1))

    with tla.vector():
        with tla.vec.func(mode="simd"):
            src0 = src0_tile.load()
            src1 = src1_tile.load()
            interleaved0, interleaved1 = tla.interleave(src0, src1)
            deinterleaved0, deinterleaved1 = tla.deinterleave(src0, src1)
            dst0_tile.store(interleaved0)
            dst1_tile.store(interleaved1)
            dst0_tile.store(deinterleaved0)
            dst1_tile.store(deinterleaved1)


def test_vector_interleave_public_export_exists() -> None:
    assert callable(tla.interleave)
    assert callable(tla.deinterleave)


def test_vector_interleave_emits_tla_op() -> None:
    tensor = _vector_tensor()
    mlir = vector_interleave_kernel.dump_mlir(
        type_args=(tensor, tensor, tensor, tensor)
    )

    assert "tla.interleave" in mlir
    assert "tla.vector" in mlir
    assert "tla.vec.func" in mlir
    assert mlir.index("tla.vec.func") < mlir.index("tla.interleave")

    interleave_lines = [line for line in mlir.splitlines() if "tla.interleave" in line]
    assert len(interleave_lines) == 1
    assert "->" in interleave_lines[0]


def test_vector_deinterleave_emits_tla_op() -> None:
    tensor = _vector_tensor()
    mlir = vector_deinterleave_kernel.dump_mlir(
        type_args=(tensor, tensor, tensor, tensor)
    )

    assert "tla.deinterleave" in mlir
    assert "tla.vector" in mlir
    assert "tla.vec.func" in mlir
    assert mlir.index("tla.vec.func") < mlir.index("tla.deinterleave")

    deinterleave_lines = [
        line for line in mlir.splitlines() if "tla.deinterleave" in line
    ]
    assert len(deinterleave_lines) == 1
    assert "->" in deinterleave_lines[0]


def test_vector_interleave_ops_accept_distinct_tensor_coordinates() -> None:
    tensor = _vector_tensor(size=128)
    lowered = BaseDSL()._lower(
        vector_interleave_distinct_coords_kernel.fn,
        kind=vector_interleave_distinct_coords_kernel.kind,
        options=dict(vector_interleave_distinct_coords_kernel.options),
        type_args=(tensor, tensor),
        location=vector_interleave_distinct_coords_kernel.decorator_location,
    )

    assert lowered.module.operation.verify()
    mlir = lowered.asm()
    assert mlir.count("tla.interleave") == 1
    assert mlir.count("tla.deinterleave") == 1


def test_vector_interleave_ops_reject_mismatched_element_types() -> None:
    f32_tensor = _vector_tensor(tla.Float32)
    f16_tensor = _vector_tensor(tla.Float16)

    for kernel in (vector_interleave_kernel, vector_deinterleave_kernel):
        lowered = BaseDSL()._lower(
            kernel.fn,
            kind=kernel.kind,
            options=dict(kernel.options),
            type_args=(f32_tensor, f16_tensor, f32_tensor, f32_tensor),
            location=kernel.decorator_location,
        )

        with pytest.raises(mlir_ir.MLIRError, match="same element type"):
            lowered.module.operation.verify()
