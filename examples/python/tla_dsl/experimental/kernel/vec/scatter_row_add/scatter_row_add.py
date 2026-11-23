# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Row-wise scatter-add: out[m, ids[m, s]] += values[m, s].

One thread per row.  The row tile is staged in UB: MTE2 loads it, the SIMT
stage scatters inside UB, MTE3 writes it back, so the random writes never touch
GM directly.  Repeated indices in a row are accumulated in ascending slot order
by the single owner thread, and out-of-range indices are skipped.
"""

import catlass.tla as tla

ROWS_PER_TILE = 16
VL_ELE = 64


@tla.kernel
def scatter_row_add_kernel(
    gm_ids: tla.Tensor,  # [M, S] int32 RowMajor
    gm_values: tla.Tensor,  # [M, S] fp32 RowMajor
    gm_out: tla.Tensor,  # [M, V] fp32 RowMajor
    S: tla.Constexpr[int],
    V: tla.Constexpr[int],
    ZERO_INIT: tla.Constexpr[int],
) -> None:
    m = gm_out.origin_shape[0]
    row_tiles = (m + ROWS_PER_TILE - 1) // ROWS_PER_TILE
    tile_ele = ROWS_PER_TILE * V

    ub_ptr = tla.allocate(tile_ele, tla.Float32, tla.AddressSpace.ub, 256)
    ub_tile = tla.make_tensor(
        ub_ptr,
        tla.make_layout(
            tla.make_shape(ROWS_PER_TILE, V),
            tla.make_stride(V, 1),
        ),
    )
    ub_flat_layout = tla.make_layout(tla.make_shape(tile_ele), tla.make_stride(1))
    ub_flat = tla.make_tensor(ub_ptr, ub_flat_layout)

    gid = tla.arch.block_idx()
    aiv_num = tla.arch.block_num()

    with tla.vector():
        for ti in tla.range(gid, row_tiles, aiv_num):
            row0 = ti * ROWS_PER_TILE
            rows = ROWS_PER_TILE
            if m - row0 < ROWS_PER_TILE:
                rows = m - row0

            gm_tile = tla.make_tensor(
                gm_out.ptr + row0.to(tla.Int64) * tla.Int64(V),
                tla.make_layout(
                    tla.make_shape(ROWS_PER_TILE, V),
                    tla.make_stride(V, 1),
                    origin_shape=tla.make_shape(rows, V),
                ),
            )
            ub_tile_dyn = tla.make_tensor(
                ub_ptr,
                tla.make_layout(
                    tla.make_shape(ROWS_PER_TILE, V),
                    tla.make_stride(V, 1),
                    origin_shape=tla.make_shape(rows, V),
                ),
            )

            if ZERO_INIT == 0:
                tla.copy(ub_tile_dyn, gm_tile)
            else:
                with tla.vec.func(mode="simd"):
                    zero = tla.full(0, dtype=tla.Float32)
                    for i in tla.range(tile_ele // VL_ELE):
                        tla.tile_view(
                            ub_flat,
                            tla.make_shape(VL_ELE),
                            tla.make_coord(i),
                        ).store(zero)
            tla.pipe_barrier(tla.pipes.ALL)

            with tla.vec.func(mode="simt", thread_block_dim=128):
                tid, _, _ = tla.arch.thread_idx()
                if tid < ROWS_PER_TILE:
                    if row0 + tid < m:
                        for s in tla.range(0, S, 1):
                            c = gm_ids[row0 + tid, s]
                            if c >= 0:
                                if c < V:
                                    pos = tid * V + c
                                    ub_flat[pos] = (
                                        ub_flat[pos] + gm_values[row0 + tid, s]
                                    )

            tla.pipe_barrier(tla.pipes.ALL)
            tla.copy(gm_tile, ub_tile_dyn)
            tla.pipe_barrier(tla.pipes.ALL)


# ======================================================================================
# HOST side: shape checks, compile and launch.
# ======================================================================================

_COMPILED = {}


def _vector_cores(index):
    import torch

    return int(torch.npu.get_device_properties(index).vector_core_num)


def run(global_col_ids, values, num_cols, out=None):
    """Return ``out`` ``[M, num_cols]`` fp32 with ``values`` scattered into it.

    ``out`` may be provided to accumulate on top of existing values; otherwise a
    zero-filled buffer is allocated.  Indices outside ``[0, num_cols)`` are
    skipped, and repeated indices accumulate in ascending slot order.
    """
    import torch
    from catlass.tla.runtime import from_dlpack

    if global_col_ids.dim() != 2 or values.dim() != 2:
        raise ValueError("global_col_ids and values must be 2-D [M, S]")
    if global_col_ids.shape != values.shape:
        raise ValueError(
            "global_col_ids and values must share shape, got {} vs {}".format(
                tuple(global_col_ids.shape), tuple(values.shape)
            )
        )

    m, s = global_col_ids.shape
    v = int(num_cols)
    if v <= 0:
        raise ValueError("num_cols must be positive")
    if v % 8 != 0:
        raise ValueError(
            "num_cols must be a multiple of 8 so each row stays 32-byte aligned"
        )
    if s > 0 and v > 0 and ROWS_PER_TILE * v * 4 > 227328:
        raise ValueError(
            "num_cols={} exceeds the UB tile budget ({} rows x V fp32); "
            "tile the columns before calling".format(v, ROWS_PER_TILE)
        )

    device = values.device
    zero_init = 0
    if out is None:
        out = torch.empty((m, v), dtype=torch.float32, device=device)
        zero_init = 1
    else:
        if out.shape != (m, v):
            raise ValueError(
                "out must be [M, num_cols] = [{}, {}], got {}".format(
                    m, v, tuple(out.shape)
                )
            )
        if out.dtype != torch.float32:
            raise ValueError("out must be float32")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")

    if m == 0 or v == 0 or s == 0:
        if zero_init:
            out.zero_()
        return out

    ids = global_col_ids.to(torch.int32).contiguous()
    vals = values.to(torch.float32).contiguous()

    device_index = int(str(device).split(":")[-1]) if ":" in str(device) else 0
    vector_cores = _vector_cores(device_index)

    ids_t = from_dlpack(ids, layout_tag=tla.arch.RowMajor)
    vals_t = from_dlpack(vals, layout_tag=tla.arch.RowMajor)
    out_t = from_dlpack(out, layout_tag=tla.arch.RowMajor)

    key = (m, s, v, zero_init)
    artifact = _COMPILED.get(key)
    if artifact is None:
        artifact = tla.compile(
            scatter_row_add_kernel,
            ids_t,
            vals_t,
            out_t,
            s,
            v,
            zero_init,
            options="--npu-arch 3510",
        )
        _COMPILED[key] = artifact

    artifact(
        ids_t,
        vals_t,
        out_t,
        block_num=vector_cores,
    )
    return out
