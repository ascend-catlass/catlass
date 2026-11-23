# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# See LICENSE in the root of the software repository for the full text of the License.

"""Byte-exact device regressions for UB register loads and stores."""

from __future__ import annotations

import pytest

from catlass import tla
from catlass.params import (
    NormalLoadParams,
    NormalStoreParams,
    PostMode,
    StoreDist,
    UnalignLoadParams,
    UnalignStoreParams,
)
from catlass.tla.runtime import from_dlpack

pytestmark = pytest.mark.npu

_COPY_BYTES = 512
_ALIGN_LOAD = NormalLoadParams(post_mode=PostMode.POST_MODE_UPDATE)
_ALIGN_LOAD_SHORT = NormalLoadParams(
    post_mode=PostMode.POST_MODE_UPDATE, post_update_stride=8
)
_ALIGN_STORE = NormalStoreParams(post_mode=PostMode.POST_MODE_UPDATE)
_UNALIGN_LOAD = UnalignLoadParams(post_mode=PostMode.POST_MODE_UPDATE)
_UNALIGN_STORE = UnalignStoreParams(post_mode=PostMode.POST_MODE_UPDATE)


@tla.kernel
def _register_move(
    src: tla.Tensor,
    dst: tla.Tensor,
    dtype: tla.Constexpr,
    width: tla.Constexpr,
    mode: tla.Constexpr,
    copy_bytes: tla.Constexpr,
):
    src_ptr = tla.allocate(copy_bytes, tla.UInt8, tla.AddressSpace.ub, 32)
    dst_ptr = tla.allocate(copy_bytes, tla.UInt8, tla.AddressSpace.ub, 32)
    src_bytes = tla.make_tensor_like(src_ptr, src, tla.arch.RowMajor)
    dst_bytes = tla.make_tensor_like(dst_ptr, dst, tla.arch.RowMajor)

    typed_src = tla.recast_ptr(src_ptr, dtype=dtype)
    typed_dst = tla.recast_ptr(dst_ptr, dtype=dtype)
    if tla.const_expr(mode in ("unaligned", "unaligned-update")):
        typed_src = typed_src + 1
        typed_dst = typed_dst + 1

    loaded = tla.flag("loaded", tla.arch.MTE2, tla.arch.VECTOR)
    stored = tla.flag("stored", tla.arch.VECTOR, tla.arch.MTE3)
    with tla.vector():
        tla.copy(src_bytes, src)
        tla.copy(dst_bytes, dst)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)

        with tla.vec.func(mode="simd"):
            if tla.const_expr(mode == "update-load-subview"):
                window = tla.make_layout(tla.make_shape(32), tla.make_stride(1))
                span = tla.make_layout(tla.make_shape(64), tla.make_stride(1))
                x = tla.make_tensor(typed_src, window)
                y = tla.make_tensor(typed_dst, span)
                first, x = x.load(_ALIGN_LOAD_SHORT)
                tla.tile_view(y, tla.make_shape(32), tla.make_coord(0)).store(first)
                x = tla.make_tensor(x.ptr, span)
                part = tla.tile_view(x, tla.make_shape(32), tla.make_coord(1))
                second = part.load(UnalignLoadParams())
                tla.tile_view(y, tla.make_shape(32), tla.make_coord(1)).store(second)
            elif tla.const_expr(mode == "update-store-subview"):
                window = tla.make_layout(tla.make_shape(64), tla.make_stride(1))
                x = tla.make_tensor(typed_src, window)
                y = tla.make_tensor(typed_dst, window)
                value = x.load()
                y = y.store(value, _ALIGN_STORE)
                part = tla.tile_view(y, tla.make_shape(16), tla.make_coord(1))
                part.store(value, UnalignStoreParams())
            elif tla.const_expr(mode == "update"):
                window = tla.make_layout(tla.make_shape(16), tla.make_stride(1))
                x = tla.make_tensor(typed_src, window)
                y = tla.make_tensor(typed_dst, window)
                first, x = x.load(_ALIGN_LOAD)
                y = y.store(first, _ALIGN_STORE)
                second, x = x.load(_ALIGN_LOAD)
                tail, _ = tla.update_mask(5, dtype=dtype)
                y = y.store(second, _ALIGN_STORE, mask=tail)
            elif tla.const_expr(mode == "unaligned-update"):
                window = tla.make_layout(tla.make_shape(16), tla.make_stride(1))
                x = tla.make_tensor(typed_src, window)
                y = tla.make_tensor(typed_dst, window)
                for _ in tla.range(2):
                    value, x = x.load(_UNALIGN_LOAD)
                    y = y.store(value, _UNALIGN_STORE)
            elif tla.const_expr(mode == "unaligned"):
                window = tla.make_layout(
                    tla.make_shape(256 // width), tla.make_stride(1)
                )
                x = tla.make_tensor(typed_src, window)
                y = tla.make_tensor(typed_dst, window)
                y.store(x.load(UnalignLoadParams()), UnalignStoreParams())
            else:
                x = tla.make_tensor(
                    typed_src, tla.make_layout(tla.make_shape(32), tla.make_stride(1))
                )
                if tla.const_expr(mode == "pack"):
                    y = tla.make_tensor(
                        typed_dst,
                        tla.make_layout(tla.make_shape(16), tla.make_stride(1)),
                    )
                    y.store(
                        x.load(), NormalStoreParams(store_dist=StoreDist.DIST_PACK_B64)
                    )
                else:
                    y = tla.make_tensor(
                        typed_dst,
                        tla.make_layout(tla.make_shape(32), tla.make_stride(1)),
                    )
                    tail, _ = tla.update_mask(5, dtype=dtype)
                    y.store(x.load(), mask=tail)

        tla.set_flag(stored)
        tla.wait_flag(stored)
        tla.copy(dst, dst_bytes)
        tla.pipe_barrier(tla.pipes.ALL)


@pytest.mark.parametrize("dtype", (tla.Int64, tla.UInt64, tla.Float64))
@pytest.mark.parametrize(
    "mode", ("update", "unaligned", "unaligned-update", "pack", "masked")
)
def test_b64_register_move(device, dtype, mode):
    _verify_move(device, dtype, 8, mode)


@pytest.mark.parametrize(
    "dtype,width", ((tla.UInt8, 1), (tla.UInt16, 2), (tla.UInt32, 4))
)
def test_unsigned_unaligned_register_move(device, dtype, width):
    _verify_move(device, dtype, width, "unaligned")


def test_unaligned_store_to_subview_of_updated_window(device):
    _verify_move(device, tla.Float32, 4, "update-store-subview")


def test_b64_unaligned_load_from_subview_of_updated_window(device):
    _verify_move(device, tla.Int64, 8, "update-load-subview")


def _verify_move(device, dtype, width, mode):
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(device)
    copy_bytes = 768 if mode == "update-load-subview" else _COPY_BYTES
    indices = torch.arange(copy_bytes)
    source = ((indices * 17 + indices // 256) % 256).to(torch.uint8)
    expected = torch.full_like(source, 0xA5)
    if mode == "update-load-subview":
        expected[:256] = source[:256]
        expected[256:512] = source[320:576]
    elif mode == "update-store-subview":
        expected[:256] = source[:256]
        expected[320:384] = source[:64]
    elif mode == "pack":
        expected[:128] = source[:256].reshape(32, 8)[:, :4].reshape(-1)
    elif mode in ("unaligned", "unaligned-update"):
        expected[width : width + 256] = source[width : width + 256]
    else:
        length = (16 + 5 if mode == "update" else 5) * 8
        expected[:length] = source[:length]

    src_npu, dst_npu = source.npu(), torch.full_like(source, 0xA5).npu()
    src = from_dlpack(src_npu, layout_tag=tla.arch.RowMajor)
    dst = from_dlpack(dst_npu, layout_tag=tla.arch.RowMajor)
    artifact = tla.compile(
        _register_move,
        src,
        dst,
        dtype,
        width,
        mode,
        copy_bytes,
        options="--npu-arch 3510",
    )
    artifact(src, dst, block_num=1)
    torch.npu.synchronize()
    torch.testing.assert_close(dst_npu.cpu(), expected, rtol=0, atol=0)
