from __future__ import annotations

from catlass.tla.runtime import make_fake_tensor


import pytest

import catlass.tla as tla
import catlass.runtime as runtime_mod  # noqa: F401


# --- Frontend IR emission -----------------------------------------------------


@tla.kernel
def ptr_extract_kernel(mem_in: tla.Tensor) -> None:
    tile = tla.tile_view(mem_in, tla.make_shape(16, 16), tla.make_coord(0, 0))
    ptr = tla.allocate((16, 16), tla.Float32, tla.AddressSpace.l1, 256)
    # Extract a pointer from a GM kernel-arg tile, offset by 4 elements, and build a
    # row-major GM tensor from it via make_tensor.
    gm_ptr = tile.ptr
    gm_off = gm_ptr + 4
    gm_src = tla.make_tensor(
        gm_off, tla.make_layout(tla.make_shape(8, 8), tla.make_stride(8, 1))
    )
    # Offset an allocator-backed L1 pointer by 8 elements and build a zN L1 tensor via
    # make_tensor_like (clones gm_src's shape), consuming the offset pointer.
    l1_off = ptr + 8
    l1_dst = tla.make_tensor_like(l1_off, gm_src, tla.arch.zN)
    with tla.cube():
        tla.copy(l1_dst, gm_src)


def _host_mem():
    return make_fake_tensor(
        tla.Float32,
        (16, 16),
        (16, 1),
        origin_shape=(16, 16),
        layout_tag=tla.arch.RowMajor,
    )


def test_ptr_emits_tensor_ptr_and_ptr_add() -> None:
    mlir = ptr_extract_kernel.dump_mlir(type_args=(_host_mem(),))
    assert "tla.tensor_ptr" in mlir
    assert "tla.ptr_add" in mlir
    # tensor_ptr result type matches the tensor's embedded ptr type (GM f32, align 4).
    assert "tla.tensor_ptr" in mlir and "-> !tla.ptr<f32, gm, 4>" in mlir
    # ptr_add preserves the source pointer's alignment (256, not refined).
    assert "!tla.ptr<f32, l1, 256>" in mlir
    # Both offset pointers feed make_tensor / make_tensor_like.
    assert "tla.make_tensor" in mlir
    assert "tla.make_tensor_like" in mlir


def test_ptr_add_rejects_float_offset() -> None:
    @tla.kernel
    def bad_kernel(mem_in: tla.Tensor) -> None:
        ptr = tla.allocate(16, tla.Float32, tla.AddressSpace.l1, 256)
        _ = tla.make_tensor(
            ptr + 1.5,  # type: ignore[operator]
            tla.make_layout(tla.make_shape(4, 4), tla.make_stride(4, 1)),
        )

    with pytest.raises(Exception):
        bad_kernel.dump_mlir(type_args=(_host_mem(),))
