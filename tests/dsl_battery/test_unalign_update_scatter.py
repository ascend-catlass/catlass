# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# See LICENSE in the root of the software repository for the full text of the License.

"""Check scatter ordering against unaligned UPDATE on shared UB storage."""

import pytest

from catlass import tla
from catlass.params import (
    LoadDist,
    MemType,
    PostMode,
    StoreDist,
    UnalignLoadParams,
    UnalignStoreParams,
)
from catlass.tla.runtime import from_dlpack

pytestmark = pytest.mark.npu

_LANES = 64
_ROUNDS = 2
_SCRATCH_ELEMENTS = 288
_INITIAL_VALUE = 0x11111111
_UPDATE_VALUE = 0x22222222
_SCATTER_VALUE = 0x33333333


@tla.kernel
def _unalign_update_scatter(
    initial: tla.Tensor,
    result: tla.Tensor,
    read_after_scatter: tla.Constexpr,
    dual: tla.Constexpr,
):
    step = 2 * _LANES if dual else _LANES
    buffer = tla.allocate(_SCRATCH_ELEMENTS, tla.Int32, tla.AddressSpace.ub, 32)
    scratch = tla.make_tensor_like(buffer, initial, tla.arch.RowMajor)
    window_layout = tla.make_layout(tla.make_shape(step), tla.make_stride(1))
    window = tla.make_tensor(buffer + 1, window_layout)
    if tla.const_expr(read_after_scatter):
        output = tla.allocate(_ROUNDS * _LANES, tla.Int32, tla.AddressSpace.ub, 32)
        result_ub = tla.make_tensor_like(output, result, tla.arch.RowMajor)
        register_layout = tla.make_layout(tla.make_shape(_LANES), tla.make_stride(1))
        load = UnalignLoadParams(
            post_mode=PostMode.POST_MODE_UPDATE,
            load_dist=LoadDist.DIST_DINTLV_B32 if dual else LoadDist.DIST_NORM,
        )
    else:
        result_ub = scratch
        store = UnalignStoreParams(
            post_mode=PostMode.POST_MODE_UPDATE,
            store_dist=StoreDist.DIST_INTLV_B32 if dual else StoreDist.DIST_NORM,
        )

    loaded = tla.flag("scatter_update_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    stored = tla.flag("scatter_update_stored", tla.arch.VECTOR, tla.arch.MTE3)
    with tla.vector():
        tla.copy(scratch, initial)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        with tla.vec.func(mode="simd"):
            values = tla.full(_UPDATE_VALUE, dtype=tla.Int32)
            if tla.const_expr(read_after_scatter):
                lanes = tla.arange(dtype=tla.Int32)
                for iteration in tla.range(_ROUNDS):
                    indices = (lanes + iteration * step).bitcast(tla.UInt32)
                    tla.scatter(scratch, values, indices)
                    tla.local_mem_bar(MemType.VEC_STORE, MemType.VEC_LOAD)
                    payload, window = window.load(load)
                    if tla.const_expr(dual):
                        payload, _ = payload
                    tile = tla.make_tensor(output + iteration * _LANES, register_layout)
                    tile.store(payload)
            else:
                tail = tla.full(_SCATTER_VALUE, dtype=tla.Int32)
                first_lane = tla.create_mask(pattern=tla.mask.VL1, dtype=tla.Int32)
                for iteration in tla.range(_ROUNDS):
                    if tla.const_expr(dual):
                        window = window.store((values, values), store)
                    else:
                        window = window.store(values, store)
                    tla.local_mem_bar(MemType.VEC_STORE, MemType.VEC_STORE)
                    last_element = tla.full(
                        (iteration + 1) * step, dtype=tla.Int32
                    ).bitcast(tla.UInt32)
                    tla.scatter(scratch, tail, last_element, mask=first_lane)
                    # Complete the scatter before a later UPDATE can flush its tail.
                    tla.local_mem_bar(MemType.VEC_STORE, MemType.VEC_STORE)
        tla.set_flag(stored)
        tla.wait_flag(stored)
        tla.copy(result, result_ub)
        tla.pipe_barrier(tla.pipes.ALL)


@pytest.mark.parametrize("read_after_scatter", (True, False), ids=("read", "write"))
@pytest.mark.parametrize("dual", (False, True), ids=("single", "dual"))
def test_unalign_update_scatter_order(device, read_after_scatter, dual):
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(device)
    initial = torch.full((_SCRATCH_ELEMENTS,), _INITIAL_VALUE, dtype=torch.int32).npu()
    step = 2 * _LANES if dual else _LANES
    if read_after_scatter:
        expected = torch.full((_ROUNDS, _LANES), _INITIAL_VALUE, dtype=torch.int32)
        updated_lanes = _LANES // 2 if dual else _LANES - 1
        expected[:, :updated_lanes] = _UPDATE_VALUE
        expected = expected.flatten()
    else:
        expected = torch.full((_SCRATCH_ELEMENTS,), _INITIAL_VALUE, dtype=torch.int32)
        expected[1 : _ROUNDS * step + 1] = _UPDATE_VALUE
        expected[step : _ROUNDS * step + 1 : step] = _SCATTER_VALUE
    output = torch.empty_like(expected).npu()
    src = from_dlpack(initial, layout_tag=tla.arch.RowMajor)
    dst = from_dlpack(output, layout_tag=tla.arch.RowMajor)
    kernel = tla.compile(
        _unalign_update_scatter,
        src,
        dst,
        read_after_scatter,
        dual,
        options="--npu-arch 3510",
    )
    kernel(src, dst, block_num=1)
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
