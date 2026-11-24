# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Per-row symmetric INT8 dynamic quantization with FP32 scale and dequantization."""

from functools import lru_cache

import catlass.tla as tla

VECTOR_LANES = 64
SCALE_EPSILON = 1.0e-12
DEFAULT_Q_MAX = 127.0


@tla.kernel
def dynamic_quant_kernel(
    x_addr: tla.Int64,
    q_addr: tla.Int64,
    scale_addr: tla.Int64,
    deq_addr: tla.Int64,
    m: tla.Int32,
    N: tla.Constexpr[int],
    Q_MAX: tla.Constexpr[float],
) -> None:
    gm_x = tla.make_ptr(tla.Float32, x_addr, tla.AddressSpace.gm)
    gm_q = tla.make_ptr(tla.Int8, q_addr, tla.AddressSpace.gm)
    gm_scale = tla.make_ptr(tla.Float32, scale_addr, tla.AddressSpace.gm)
    gm_deq = tla.make_ptr(tla.Float32, deq_addr, tla.AddressSpace.gm)

    round_to_int = tla.params.CastParams(
        reg_slot=tla.params.RegSlot.ZERO,
        sat_mode=tla.params.SatMode.SAT,
        round_mode=tla.params.RoundMode.CAST_RINT,
    )
    int_to_float = tla.params.CastParams()
    pack_int8 = tla.params.NormalStoreParams(
        store_dist=tla.params.StoreDist.DIST_PACK4_B32,
    )

    # Each mutex protects its UB buffers until the consuming pipe is done.
    input_slot = tla.mutex(resource="dq_input", id=0)
    output_slot = tla.mutex(resource="dq_output", id=1)

    row_layout = tla.make_layout(
        tla.make_shape(1, N),
        tla.make_stride(N, 1),
        origin_shape=tla.make_shape(1, N),
    )
    scale_layout = tla.make_layout(
        tla.make_shape(1, 1),
        tla.make_stride(1, 1),
        origin_shape=tla.make_shape(1, 1),
    )
    ub_x = tla.make_tensor(
        tla.allocate(N, tla.Float32, tla.AddressSpace.ub, 256), row_layout
    )
    ub_q = tla.make_tensor(
        tla.allocate(N, tla.Int8, tla.AddressSpace.ub, 256), row_layout
    )
    ub_deq = tla.make_tensor(
        tla.allocate(N, tla.Float32, tla.AddressSpace.ub, 256), row_layout
    )
    ub_scale = tla.make_tensor(
        tla.allocate(1, tla.Float32, tla.AddressSpace.ub, 256), scale_layout
    )

    tile_shape = tla.make_shape(1, VECTOR_LANES)
    tiles_per_row = N // VECTOR_LANES
    block_idx = tla.arch.block_idx()
    block_num = tla.arch.block_num()

    with tla.vector():
        for row in tla.range(block_idx, m, block_num):
            row_offset = row.to(tla.Int64) * tla.Int64(N)
            gm_row_x = tla.make_tensor(gm_x + row_offset, row_layout)
            gm_row_q = tla.make_tensor(gm_q + row_offset, row_layout)
            gm_row_deq = tla.make_tensor(gm_deq + row_offset, row_layout)
            gm_row_scale = tla.make_tensor(gm_scale + row.to(tla.Int64), scale_layout)

            input_slot.lock(pipe=tla.arch.MTE2)
            tla.copy(ub_x, gm_row_x)
            input_slot.unlock(pipe=tla.arch.MTE2)

            input_slot.lock(pipe=tla.arch.VECTOR)
            output_slot.lock(pipe=tla.arch.VECTOR)
            with tla.vec.func(mode="simd"):
                mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)

                # Accumulate lane-wise maxima before reducing across the row.
                lane_max = tla.full(0.0, dtype=tla.Float32)
                for col_tile in tla.range_constexpr(tiles_per_row):
                    x_tile = tla.tile_view(
                        ub_x, tile_shape, tla.make_coord(0, col_tile)
                    )
                    lane_max = tla.max(
                        lane_max, tla.abs(x_tile.load(), mask=mask), mask=mask
                    )
                amax = lane_max.reduce(tla.ReductionOp.MAX, mask=mask)
                scale = tla.mul(amax, 1.0 / Q_MAX)
                tla.tile_view(
                    ub_scale, tla.make_shape(1, 1), tla.make_coord(0, 0)
                ).store(scale)

                scale_vec = tla.full(scale, dtype=tla.Float32)
                inv_scale = tla.div(
                    tla.full(Q_MAX, dtype=tla.Float32),
                    tla.max(
                        tla.full(amax, dtype=tla.Float32), SCALE_EPSILON, mask=mask
                    ),
                    mask=mask,
                )

                for col_tile in tla.range_constexpr(tiles_per_row):
                    coord = tla.make_coord(0, col_tile)
                    x_tile = tla.tile_view(ub_x, tile_shape, coord)
                    scaled = tla.mul(x_tile.load(), inv_scale, mask=mask)
                    clipped = tla.max(
                        tla.min(scaled, Q_MAX, mask=mask), -Q_MAX, mask=mask
                    )
                    q_int = clipped.to(tla.Int32, round_to_int, mask=mask)
                    q_float = q_int.to(tla.Float32, int_to_float, mask=mask)

                    # Each clipped code fits in INT8, so storing its low byte is exact.
                    tla.tile_view(ub_q, tile_shape, coord).store(
                        q_int, pack_int8, mask=mask
                    )
                    tla.tile_view(ub_deq, tile_shape, coord).store(
                        tla.mul(q_float, scale_vec, mask=mask)
                    )

            output_slot.unlock(pipe=tla.arch.VECTOR)
            input_slot.unlock(pipe=tla.arch.VECTOR)

            output_slot.lock(pipe=tla.arch.MTE3)
            tla.copy(gm_row_q, ub_q)
            tla.copy(gm_row_deq, ub_deq)
            tla.copy(gm_row_scale, ub_scale)
            output_slot.unlock(pipe=tla.arch.MTE3)


# ======================================================================================
# HOST side: input checks, compile and launch.
# ======================================================================================


@lru_cache(maxsize=32)
def _vector_cores(device_index):
    import torch

    return int(torch.npu.get_device_properties(device_index).vector_core_num)


@lru_cache(maxsize=128)
def _compile(n, q_max, device_index):
    # M is a runtime argument. Keep separate executors because each one binds to
    # the device on its first launch.
    return tla.compile(
        dynamic_quant_kernel,
        tla.Int64(0),
        tla.Int64(0),
        tla.Int64(0),
        tla.Int64(0),
        tla.Int32(0),
        n,
        q_max,
        options="--npu-arch 3510",
    )


def run(x, q_max=DEFAULT_Q_MAX):
    """Return INT8 codes, FP32 row scales, and FP32 dequantized values."""
    import torch

    if x.dtype != torch.float32:
        raise ValueError("x must be float32")
    if x.ndim != 2:
        raise ValueError("x must be 2-D [M, N]")
    m, n = x.shape
    if n % VECTOR_LANES:
        raise ValueError(f"N must be a multiple of {VECTOR_LANES}, got {n}")
    q_max = float(q_max)
    if not 0.0 < q_max <= DEFAULT_Q_MAX:
        raise ValueError(f"q_max must be greater than 0 and at most {DEFAULT_Q_MAX:g}")

    device = x.device
    q = torch.empty((m, n), dtype=torch.int8, device=device)
    deq = torch.empty((m, n), dtype=torch.float32, device=device)
    if n == 0:
        return q, torch.zeros((m,), dtype=torch.float32, device=device), deq
    scale = torch.empty((m,), dtype=torch.float32, device=device)
    if m == 0:
        return q, scale, deq

    if device.type != "npu":
        raise ValueError("x must be on an NPU")
    x = x.contiguous()
    device_index = device.index
    artifact = _compile(n, q_max, device_index)
    blocks = _vector_cores(device_index)
    stream = torch.npu.current_stream(device)
    # The kernel ABI carries raw addresses. Keep the input allocation valid
    # until the asynchronous launch has finished using it.
    x.record_stream(stream)
    args = (
        tla.Int64(x.data_ptr()),
        tla.Int64(q.data_ptr()),
        tla.Int64(scale.data_ptr()),
        tla.Int64(deq.data_ptr()),
        m,
    )
    if device_index == torch.npu.current_device():
        artifact(*args, block_num=blocks, stream=stream.npu_stream)
    else:
        with torch.npu.device(device):
            artifact(*args, block_num=blocks, stream=stream.npu_stream)
    return q, scale, deq


if __name__ == "__main__":
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    m, n = 64, 512
    torch.manual_seed(0)
    x = torch.randn(m, n, dtype=torch.float32, device="npu")
    x[3] = 0.0  # all-zero row

    q, scale, deq = run(x)
    xr = x.cpu()
    amax = xr.abs().amax(-1, keepdim=True)
    ref_scale = amax * (1.0 / 127.0)
    inv_scale = torch.full_like(amax, 127.0) / amax.clamp_min(1.0e-12)
    ref_q = (xr * inv_scale).clamp(-127, 127).round().to(torch.int8)
    print("scale err:", (scale.cpu() - ref_scale.squeeze(-1)).abs().max().item())
    print("int8 mismatch:", (q.cpu() != ref_q).sum().item())
    print("deq err:", (deq.cpu() - ref_q.float() * ref_scale).abs().max().item())
