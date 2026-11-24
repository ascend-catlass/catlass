# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# See LICENSE in the root of the software repository for the full text of the License.

"""Byte-exact device regressions for UB register loads and stores."""

from __future__ import annotations

import pytest

from catlass import tla
from catlass.params import (
    LoadDist,
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


@tla.kernel
def _dual_update(
    src: tla.Tensor,
    dst: tla.Tensor,
    dtype: tla.Constexpr,
    width: tla.Constexpr,
    load_dist: tla.Constexpr,
    store_dist: tla.Constexpr,
    unaligned: tla.Constexpr,
    transfer_bytes: tla.Constexpr,
    dynamic_stride: tla.Constexpr,
):
    src_ptr = tla.allocate(1280, tla.UInt8, tla.AddressSpace.ub, 32)
    dst_ptr = tla.allocate(2304, tla.UInt8, tla.AddressSpace.ub, 32)
    src_bytes = tla.make_tensor_like(src_ptr, src, tla.arch.RowMajor)
    dst_bytes = tla.make_tensor_like(dst_ptr, dst, tla.arch.RowMajor)
    loaded = tla.flag("dual_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    stored = tla.flag("dual_stored", tla.arch.VECTOR, tla.arch.MTE3)
    layout = tla.make_layout(tla.make_shape(512 // width), tla.make_stride(1))
    stride = transfer_bytes // width
    if tla.const_expr(dynamic_stride):
        stride = tla.Int32(stride) + tla.arch.block_idx()
    if tla.const_expr(unaligned):
        x = tla.make_tensor(tla.recast_ptr(src_ptr + width, dtype=dtype), layout)
        y = tla.make_tensor(tla.recast_ptr(dst_ptr + 3 * width, dtype=dtype), layout)
    else:
        x = tla.make_tensor(tla.recast_ptr(src_ptr + 32, dtype=dtype), layout)
        y = tla.make_tensor(tla.recast_ptr(dst_ptr + 64, dtype=dtype), layout)
    with tla.vector():
        tla.copy(src_bytes, src)
        tla.copy(dst_bytes, dst)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        with tla.vec.func(mode="simd"):
            if tla.const_expr(unaligned):
                load = UnalignLoadParams(
                    load_dist=load_dist,
                    post_mode=PostMode.POST_MODE_UPDATE,
                    post_update_stride=stride,
                )
                store = UnalignStoreParams(
                    store_dist=store_dist,
                    post_mode=PostMode.POST_MODE_UPDATE,
                    post_update_stride=stride,
                )
            else:
                load = NormalLoadParams(
                    load_dist=load_dist,
                    post_mode=PostMode.POST_MODE_UPDATE,
                    post_update_stride=544 // width,
                )
                store = NormalStoreParams(
                    store_dist=store_dist,
                    post_mode=PostMode.POST_MODE_UPDATE,
                    post_update_stride=576 // width,
                )
            register = tla.make_layout(tla.make_shape(256 // width), tla.make_stride(1))
            first = tla.make_tensor(
                tla.recast_ptr(src_ptr + 32, dtype=dtype), register
            ).load()
            second = tla.make_tensor(
                tla.recast_ptr(src_ptr + 288, dtype=dtype), register
            ).load()
            for iteration in tla.range(2):
                pair, x = x.load(load)
                even, odd = pair
                dump = dst_ptr + (1280 + 512 * iteration)
                tla.make_tensor(tla.recast_ptr(dump, dtype=dtype), register).store(even)
                tla.make_tensor(
                    tla.recast_ptr(dump + 256, dtype=dtype), register
                ).store(odd)
                y = y.store((first, second), store)
        tla.set_flag(stored)
        tla.wait_flag(stored)
        tla.copy(dst, dst_bytes)
        tla.pipe_barrier(tla.pipes.ALL)


@pytest.mark.parametrize(
    "dtype,width",
    (
        (tla.Bool, 1),
        (tla.Int8, 1),
        (tla.UInt8, 1),
        (tla.Float8E4M3FN, 1),
        (tla.Float8E5M2, 1),
        (tla.HiFloat8, 1),
        (tla.Float8E8M0, 1),
        (tla.Float4E2M1x2, 1),
        (tla.Float4E1M2x2, 1),
        (tla.Int16, 2),
        (tla.UInt16, 2),
        (tla.Float16, 2),
        (tla.BFloat16, 2),
        (tla.Int32, 4),
        (tla.UInt32, 4),
        (tla.Float32, 4),
        (tla.Int64, 8),
        (tla.UInt64, 8),
        (tla.Float64, 8),
    ),
)
@pytest.mark.parametrize("unaligned", (False, True), ids=("aligned", "unaligned"))
def test_dual_update_register_move(device, dtype, width, unaligned):
    _verify_dual_update(device, dtype, width, unaligned, 512)


@pytest.mark.parametrize(
    "dtype,width,transfer_bytes,dynamic_stride",
    (
        (tla.UInt8, 1, 192, False),
        (tla.UInt8, 1, 352, False),
        (tla.UInt8, 1, 352, True),
        (tla.Int16, 2, 193 * 2, False),
        (tla.Int16, 2, 193 * 2, True),
    ),
)
def test_dual_unaligned_update_partial_stride(
    device, dtype, width, transfer_bytes, dynamic_stride
):
    _verify_dual_update(device, dtype, width, True, transfer_bytes, dynamic_stride)


def _verify_dual_update(
    device, dtype, width, unaligned, transfer_bytes, dynamic_stride=False
):
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(device)
    unit = min(width, 4)
    load_dist, store_dist = {
        1: (LoadDist.DIST_DINTLV_B8, StoreDist.DIST_INTLV_B8),
        2: (LoadDist.DIST_DINTLV_B16, StoreDist.DIST_INTLV_B16),
        4: (LoadDist.DIST_DINTLV_B32, StoreDist.DIST_INTLV_B32),
    }[unit]
    values = torch.randint(
        2 if dtype is tla.Bool else 256,
        (1280,),
        dtype=torch.uint8,
        generator=torch.Generator().manual_seed(5),
    )
    expected = torch.full((2304,), 0xA5, dtype=torch.uint8)
    checked = torch.ones_like(expected, dtype=torch.bool)
    store_bytes = torch.stack(
        (values[32:288].reshape(-1, unit), values[288:544].reshape(-1, unit)), dim=1
    ).flatten()
    for iteration in range(2):
        if unaligned:
            src_offset = width + iteration * transfer_bytes
            dst_offset = 3 * width + iteration * transfer_bytes
        else:
            src_offset = 32 + iteration * 544
            dst_offset = 64 + iteration * 576
        expected[dst_offset : dst_offset + transfer_bytes] = store_bytes[
            :transfer_bytes
        ]
        elements = values[src_offset : src_offset + transfer_bytes].reshape(-1, unit)
        dump = 1280 + iteration * 512
        for lane in range(2):
            valid = elements[lane::2].flatten()
            begin = dump + lane * 256
            end = begin + valid.numel()
            expected[begin:end] = valid
            # A partial load only defines the lanes in its effective prefix.
            checked[end : begin + 256] = False

    src_npu, dst_npu = values.npu(), torch.full_like(expected, 0xA5).npu()
    src = from_dlpack(src_npu, layout_tag=tla.arch.RowMajor)
    dst = from_dlpack(dst_npu, layout_tag=tla.arch.RowMajor)
    artifact = tla.compile(
        _dual_update,
        src,
        dst,
        dtype,
        width,
        load_dist,
        store_dist,
        unaligned,
        transfer_bytes,
        dynamic_stride,
        options="--npu-arch 3510",
    )
    artifact(src, dst, block_num=1)
    torch.npu.synchronize()
    torch.testing.assert_close(
        dst_npu.cpu()[checked], expected[checked], rtol=0, atol=0
    )
