"""Host tensor construction: ``make_tensor`` / ``make_fake_tensor`` / ``from_dlpack``."""

from __future__ import annotations

from typing import Any

import pytest

import catlass.tla as tla
import catlass.runtime as runtime_mod
from catlass.core_api import _category
from catlass.tla.runtime import make_fake_tensor
import catlass.tla.runtime as tla_runtime_mod


def _torch():
    return pytest.importorskip("torch")


def _make_tensor_from_torch(buf: Any, *, layout_tag: Any = None) -> Any:
    """Canonical Host bind path: Torch storage -> dual-mode ``tla.make_tensor``."""
    torch = _torch()
    layout_tag = layout_tag or tla.arch.RowMajor
    contig = buf.contiguous()
    dtype_map = {
        torch.float16: tla.Float16,
        torch.float32: tla.Float32,
        torch.int32: tla.Int32,
    }
    dtype = dtype_map.get(contig.dtype)
    if dtype is None:
        raise AssertionError(f"unsupported torch dtype for Host bind: {contig.dtype}")
    shape = tuple(int(dim) for dim in contig.shape)
    stride = tuple(int(step) for step in contig.stride())
    ptr = tla.make_ptr(dtype, int(contig.data_ptr()), tla.AddressSpace.gm)
    layout = tla.make_layout(
        tla.make_shape(*shape),
        tla.make_stride(*stride),
        layoutTag=layout_tag,
        origin_shape=tla.make_shape(*shape),
    )
    return tla.make_tensor(ptr, layout)


def test_make_tensor_binds_torch_buffer_without_dlpack() -> None:
    # Typical framework usage: reuse Torch allocation, skip DLPack capsule.
    torch = _torch()
    buf = torch.empty((128, 64), dtype=torch.float16)
    tensor = _make_tensor_from_torch(buf)
    assert tensor.data_ptr == int(buf.data_ptr())
    assert tensor.shape == tuple(buf.shape)
    assert tensor.stride == tuple(buf.stride())
    assert tensor._external_binding is True
    desc = tensor.tla_tensor_type_descriptor()
    assert str(desc.element_type).lower() in ("f16", "float16")


def test_make_tensor_rejects_kernel_ssa_and_raw_values() -> None:
    layout = tla.make_layout((4, 8), (8, 1), layoutTag=tla.arch.RowMajor)
    with pytest.raises(TypeError, match="make_ptr"):
        tla.make_tensor(0, layout)  # type: ignore[arg-type]
    ptr = tla.make_ptr(tla.Float16, 0, tla.AddressSpace.gm)
    with pytest.raises(TypeError, match="make_layout"):
        tla.make_tensor(ptr, (4, 8))  # type: ignore[arg-type]


def test_make_tensor_trusts_layout_metadata() -> None:
    # Hot path does not gate RM compact / unit-inner-stride; from_dlpack does.
    ptr = tla.make_ptr(tla.Float16, 0, tla.AddressSpace.gm)
    noncompact = tla.make_layout((4, 8), (1, 4), layoutTag=tla.arch.RowMajor)
    assert tla.make_tensor(ptr, noncompact).stride == (1, 4)
    strided = tla.make_layout(
        (4, 8),
        (8, 2),
        layoutTag=tla.arch.RowMajor,
        origin_shape=(4, 8),
    )
    assert tla.make_tensor(ptr, strided).stride == (8, 2)


def test_make_tensor_rejects_layout_tag_shape_mismatch() -> None:
    ptr = tla.make_ptr(tla.Float16, 0, tla.AddressSpace.gm)
    # zN origin infer still requires nested ((m0,m1),(n0,n1)).
    layout = tla.make_layout(
        (32, 64),
        (64, 1),
        layoutTag=tla.arch.zN,
    )
    with pytest.raises(Exception, match="expects shape"):
        tla.make_tensor(ptr, layout)


def test_make_tensor_mark_layout_dynamic_from_torch() -> None:
    torch = _torch()
    buf = torch.empty((8, 4), dtype=torch.float16)
    tensor = _make_tensor_from_torch(buf)
    updated = tensor.mark_layout_dynamic(leading_dim=1)
    assert updated is tensor
    assert tensor._dynamic_stride_tree == (None, 1)


def test_make_tensor_mark_compact_shape_dynamic_from_torch() -> None:
    torch = _torch()
    buf = torch.empty((16, 8), dtype=torch.float16)
    tensor = _make_tensor_from_torch(buf)
    updated = tensor.mark_compact_shape_dynamic(mode=0)
    assert updated is tensor
    assert tensor._dynamic_shape_tree[0] is None


def test_host_make_tensor_is_dual_mode_entry() -> None:
    # Aligned with make_ptr: public Host bind is tla.make_tensor, not tla.runtime.
    assert not hasattr(tla_runtime_mod, "make_tensor")
    assert "make_tensor" not in getattr(tla_runtime_mod, "__all__", ())
    ptr = tla.make_ptr(tla.Float16, 0, tla.AddressSpace.gm)
    layout = tla.make_layout((4, 8), (8, 1), layoutTag=tla.arch.RowMajor)
    host = tla.make_tensor(ptr, layout)
    assert host.data_ptr == 0
    assert host.shape == (4, 8)
    assert callable(tla.make_ptr)
    assert callable(tla.make_layout)


def test_make_fake_tensor_host_sample() -> None:
    host = make_fake_tensor(
        tla.Float32,
        (4, 8),
        (8, 1),
        origin_shape=(4, 8),
        layout_tag=tla.arch.RowMajor,
    )
    assert isinstance(host, tla.Tensor)
    assert host.data_ptr == 0
    assert host._external_binding is False
    assert host.stride == (8, 1)
    types = host.__get_mlir_types__()
    assert len(types) == 1
    assert "tla.tensor" in str(types[0])


def test_make_fake_tensor_defaults_layout_tag_to_row_major() -> None:
    host = make_fake_tensor(tla.Float32, (4, 8), (8, 1))
    assert host.layout_tag == "RowMajor"
    assert host.origin_shape == (4, 8)


def test_make_fake_tensor_requires_stride() -> None:
    with pytest.raises(TypeError, match="stride"):
        make_fake_tensor(tla.Float32, (4, 8))  # type: ignore[call-arg]


def test_make_fake_tensor_zn_explicit_trees() -> None:
    host = make_fake_tensor(
        tla.Float16,
        ((16, 2), (16, 4)),
        ((16, 256), (1, 512)),
        layout_tag=tla.arch.zN,
        origin_shape=(32, 64),
        coord=(0, 0),
    )
    assert host.origin_shape == (32, 64)
    assert host.coord == (0, 0)
    assert host.shape != (32, 64)
    assert isinstance(host.shape[0], tuple)


def test_host_runtime_does_not_export_tensor_ctor() -> None:
    assert not hasattr(runtime_mod, "_Tensor")
    assert "_Tensor" not in getattr(runtime_mod, "__all__", ())
    assert not hasattr(tla, "_Tensor")


def test_host_make_shape_coord_stride_extend_same_types() -> None:
    shape = tla.make_shape(4, 8)
    stride = tla.make_stride(8, 1)
    coord = tla.make_coord(0, 0)
    layout = tla.make_layout(shape, stride, layoutTag=tla.arch.RowMajor)
    ptr = tla.make_ptr(tla.Float16, 0, tla.AddressSpace.gm)
    host = tla.make_tensor(ptr, layout, coord=coord)
    assert host.shape == (4, 8)
    assert host.stride == (8, 1)
    assert host.coord == (0, 0)


def test_host_tensor_tile_view_still_requires_kernel_context() -> None:
    mem = make_fake_tensor(tla.Float16, (1, 2), (2, 1), origin_shape=(1, 2))
    with pytest.raises(tla.TlaIRNotExecutableError):
        _ = tla.tile_view(mem, tla.make_shape(1, 2), tla.make_coord(0, 0))


def test_region_helpers_keep_region_stub_behavior() -> None:
    stub = tla.cube()
    assert stub.__class__.__name__ == "_RegionStub"
    assert _category(stub) == "region"
