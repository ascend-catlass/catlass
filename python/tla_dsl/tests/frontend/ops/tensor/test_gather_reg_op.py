"""Tests for tla.gather Reg→Reg path (vselr)."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from catlass.tla.runtime import make_fake_tensor

import catlass.tla as tla


def _f16_pair() -> tuple[tla.Tensor, tla.Tensor, tla.Tensor]:
    return (
        make_fake_tensor(
            tla.Float16,
            (128,),
            (1,),
            addrspace=tla.AddressSpace.ub,
            origin_shape=(128,),
            layout_tag=tla.arch.RowMajor,
        ),
        make_fake_tensor(
            tla.Int16,
            (128,),
            (1,),
            addrspace=tla.AddressSpace.ub,
            origin_shape=(128,),
            layout_tag=tla.arch.RowMajor,
        ),
        make_fake_tensor(
            tla.Float16,
            (128,),
            (1,),
            addrspace=tla.AddressSpace.ub,
            origin_shape=(128,),
            layout_tag=tla.arch.RowMajor,
        ),
    )


def _gm_f16(shape: tuple[int, ...]) -> tla.Tensor:
    strides = [1]
    for dim in reversed(shape[1:]):
        strides.insert(0, strides[0] * dim)
    return make_fake_tensor(
        tla.Float16,
        shape,
        tuple(strides),
        addrspace=tla.AddressSpace.gm,
        origin_shape=shape,
        layout_tag=tla.arch.RowMajor,
    )


def _gm_i16(n: int) -> tla.Tensor:
    return make_fake_tensor(
        tla.Int16,
        (n,),
        (1,),
        addrspace=tla.AddressSpace.gm,
        origin_shape=(n,),
        layout_tag=tla.arch.RowMajor,
    )


@pytest.fixture
def gather_reg_example(monkeypatch):
    # tests/frontend/ops/tensor -> examples/python/tla_dsl
    root = Path(__file__).resolve().parents[6] / "examples" / "python" / "tla_dsl"
    monkeypatch.syspath_prepend(str(root / "dsl_tutorials/vector_simd_ops"))
    return importlib.import_module("gather_reg_op")


@tla.kernel
def gather_reg_vector_kernel(
    src: tla.Tensor, idx: tla.Tensor, dst: tla.Tensor
) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(128), tla.make_coord(0))
    idx_tile = tla.tile_view(idx, tla.make_shape(128), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(128), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            s = src_tile.load()
            i = idx_tile.load()
            dst_tile.store(tla.gather(s, i))


def test_gather_reg_op_emits_mlir() -> None:
    src, idx, dst = _f16_pair()
    mlir = gather_reg_vector_kernel.dump_mlir(type_args=(src, idx, dst))
    assert "tla.gather" in mlir
    assert "tla.gather_reg" not in mlir
    assert mlir.index("tla.vec.func") < mlir.index("tla.gather")
    lines = [line for line in mlir.splitlines() if "tla.gather" in line]
    assert len(lines) == 1
    # Reg→Reg form: both operands are !tla.vector
    assert "!tla.vector" in lines[0]


def test_gather_reg_mixed_example_emits_mlir(gather_reg_example) -> None:
    example = gather_reg_example
    src = _gm_f16((example.VL_ELE,))
    idx = _gm_i16(example.VL_ELE)
    dst = _gm_f16((example.VL_ELE,))
    cube = _gm_f16((example.CUBE_M, example.CUBE_N))
    mlir = example.gather_reg_mixed_op.dump_mlir(type_args=(src, idx, dst, cube))
    assert "tla.cube" in mlir
    assert "tla.vector" in mlir
    assert "tla.gather" in mlir
    assert "!tla.vector" in next(
        line for line in mlir.splitlines() if "tla.gather" in line
    )


def test_gather_reg_mixed_example_full_compile(
    gather_reg_example, isolated_compile_cache
) -> None:
    """Full HIVMC compile must accept gather_reg_* in a mix AIC/AIV kernel (#822)."""
    del isolated_compile_cache
    example = gather_reg_example
    src = _gm_f16((example.VL_ELE,))
    idx = _gm_i16(example.VL_ELE)
    dst = _gm_f16((example.VL_ELE,))
    cube = _gm_f16((example.CUBE_M, example.CUBE_N))
    artifact = tla.compile(
        example.gather_reg_mixed_op,
        src,
        idx,
        dst,
        cube,
        options="--npu-arch 3510",
    )
    assert artifact is not None
    assert Path(artifact.kernel_binary_path).is_file()


def _i32_tensor(n: int) -> tla.Tensor:
    return make_fake_tensor(
        tla.Int32,
        (n,),
        (1,),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(n,),
        layout_tag=tla.arch.RowMajor,
    )


@tla.kernel
def gather_reg_wide_index_kernel(
    src: tla.Tensor, idx: tla.Tensor, dst: tla.Tensor
) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(32), tla.make_coord(0))
    idx_tile = tla.tile_view(idx, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(32), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst_tile.store(tla.gather(src_tile.load(), idx_tile.load()))


def test_gather_reg_allows_wider_index() -> None:
    src, idx, dst = _i32_tensor(32), _i32_tensor(64), _i32_tensor(32)
    mlir = gather_reg_wide_index_kernel.dump_mlir(type_args=(src, idx, dst))
    gather_line = next(line for line in mlir.splitlines() if "tla.gather" in line)
    assert "32xi32" in gather_line and "64xi32" in gather_line


@tla.kernel
def gather_reg_narrow_index_kernel(
    src: tla.Tensor, idx: tla.Tensor, dst: tla.Tensor
) -> None:
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    idx_tile = tla.tile_view(idx, tla.make_shape(32), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            dst_tile.store(tla.gather(src_tile.load(), idx_tile.load()))


def test_gather_reg_rejects_narrower_index() -> None:
    src, idx, dst = _i32_tensor(64), _i32_tensor(32), _i32_tensor(64)
    with pytest.raises(tla.TlaCoreAPIError, match="fewer lanes"):
        gather_reg_narrow_index_kernel.dump_mlir(type_args=(src, idx, dst))
