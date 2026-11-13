from __future__ import annotations

from catlass.tla.runtime import make_fake_tensor


import pytest

import catlass.tla as tla
import catlass.runtime as runtime_mod  # noqa: F401


def _ub_tensor(dtype: type[tla.Numeric] = tla.Float32) -> tla.Tensor:
    return make_fake_tensor(
        dtype,
        (64,),
        (1,),
        addrspace=tla.AddressSpace.ub,
        origin_shape=(64,),
        layout_tag=tla.arch.RowMajor,
    )


@tla.kernel
def gather_vector_kernel(src: tla.Tensor, idx_mem: tla.Tensor, dst: tla.Tensor) -> None:
    """Kernel exercising tla.gather: per-lane indexed load from a UB tile."""
    src_tile = tla.tile_view(src, tla.make_shape(64), tla.make_coord(0))
    idx_tile = tla.tile_view(idx_mem, tla.make_shape(64), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            indices = idx_tile.load()
            gathered = tla.gather(src_tile, indices)
            dst_tile.store(gathered)


@tla.kernel
def gather_i64_vector_kernel(
    src: tla.Tensor, idx_mem: tla.Tensor, dst: tla.Tensor
) -> None:
    """Kernel used to reach gather validation with the 32-lane i64 capacity."""
    src_tile = tla.tile_view(src, tla.make_shape(32), tla.make_coord(0))
    idx_tile = tla.tile_view(idx_mem, tla.make_shape(32), tla.make_coord(0))
    dst_tile = tla.tile_view(dst, tla.make_shape(32), tla.make_coord(0))
    with tla.vector():
        with tla.vec.func(mode="simd"):
            indices = idx_tile.load()
            dst_tile.store(tla.gather(src_tile, indices))


def test_gather_op_emits_mlir() -> None:
    """tla.gather produces expected MLIR with proper operands and region nesting."""
    src = _ub_tensor()
    idx_mem = _ub_tensor(tla.Int32)
    dst = _ub_tensor()
    mlir = gather_vector_kernel.dump_mlir(type_args=(src, idx_mem, dst))

    assert "tla.gather" in mlir
    assert "tla.store" in mlir
    assert "tla.vector" in mlir
    assert mlir.index("tla.vector") < mlir.index("tla.gather")
    gather_lines = [line for line in mlir.splitlines() if "tla.gather" in line]
    assert len(gather_lines) == 1


@pytest.mark.parametrize("index_dtype", [tla.Int16, tla.Int64])
def test_gather_rejects_non_i32_indices(index_dtype: type[tla.Numeric]) -> None:
    """tla.gather accepts only i32 index vectors."""
    kernel = (
        gather_vector_kernel if index_dtype is tla.Int16 else gather_i64_vector_kernel
    )
    lanes = 64 if index_dtype is tla.Int16 else 32
    with pytest.raises(tla.TlaCoreAPIError, match="gather indices must be i32"):
        kernel.dump_mlir(
            type_args=(
                _ub_tensor(),
                make_fake_tensor(
                    index_dtype,
                    (lanes,),
                    (1,),
                    addrspace=tla.AddressSpace.ub,
                    origin_shape=(lanes,),
                    layout_tag=tla.arch.RowMajor,
                ),
                _ub_tensor(),
            )
        )


def test_gather_rejects_non_ub_src() -> None:
    """tla.gather raises when the source tile is not in ub address space."""

    @tla.kernel
    def bad_src(l0c_mem: tla.Tensor, idx_mem: tla.Tensor, dst: tla.Tensor) -> None:
        src_tile = tla.tile_view(l0c_mem, tla.make_shape(64), tla.make_coord(0))
        idx_tile = tla.tile_view(idx_mem, tla.make_shape(64), tla.make_coord(0))
        dst_tile = tla.tile_view(dst, tla.make_shape(64), tla.make_coord(0))
        with tla.vector():
            with tla.vec.func(mode="simd"):
                indices = idx_tile.load()
                dst_tile.store(tla.gather(src_tile, indices))

    l0c_mem = make_fake_tensor(
        tla.Float32,
        (64,),
        (1,),
        origin_shape=(64,),
        layout_tag=tla.arch.RowMajor,
    )
    idx_mem = _ub_tensor(tla.Int32)
    dst = _ub_tensor()

    with pytest.raises(
        tla.TlaCoreAPIError, match="invalid argument 'x'.*expected addrspace ub"
    ):
        bad_src.dump_mlir(type_args=(l0c_mem, idx_mem, dst))
