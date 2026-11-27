# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Reduce FP32 rows with sum, max, mean, or logsumexp.

One AIV owns each complete row. Each tile stays in UB for its reduction,
including both passes of logsumexp, before its results are copied to GM.
"""

from threading import Lock

import catlass.tla as tla
import torch
import torch_npu  # noqa: F401
from catlass.tla.runtime import from_dlpack

VL_ELE = 64  # FP32 elements per SIMD vector
MIN_F = -3.4028234663852886e38
UB_BYTES = 248 * 1024  # Usable per-AIV UB after the compiler reservation.
_OUTPUT_TILE_BYTES = VL_ELE * 4
_MAX_COLUMNS = (UB_BYTES - _OUTPUT_TILE_BYTES) // 4
_CAST_NONE = tla.params.CastParams()

OP_SUM = 0
OP_MAX = 1
OP_MEAN = 2
OP_LOGSUMEXP = 3


def _chunk(tensor, row, col):
    """64-lane view of ``tensor[row, col * 64 : (col + 1) * 64]``."""
    return tla.tile_view(tensor, tla.make_shape(1, VL_ELE), tla.make_coord(row, col))


@tla.kernel
def row_reduce_kernel(
    gm_x: tla.Tensor,  # [M, N] fp32
    gm_out: tla.Tensor,  # [M] fp32
    N: tla.Constexpr[int],
    N_PAD: tla.Constexpr[int],
    TAIL: tla.Constexpr[int],
    INV_N: tla.Constexpr[float],
    OP: tla.Constexpr[int],
    TILE_ROWS: tla.Constexpr[int],
    BUFS: tla.Constexpr[int],
) -> None:
    m = gm_x.origin_shape[0]

    x_ptr = [None] * BUFS
    s_ptr = [None] * BUFS
    # A mutex owns both the input tile and packed output of one UB slot.
    tile_mtx = [None] * BUFS
    for bi in tla.range_constexpr(BUFS):
        x_ptr[bi] = tla.allocate(
            TILE_ROWS * N_PAD, tla.Float32, tla.AddressSpace.ub, 256
        )
        # A full vector stages the TILE_ROWS results in consecutive lanes.
        s_ptr[bi] = tla.allocate(VL_ELE, tla.Float32, tla.AddressSpace.ub, 256)
        tile_mtx[bi] = tla.mutex(resource="rr_tile", id=bi)

    # UB rows are padded to a whole 64-lane vector; the GM side stays dense.
    tile_ub = tla.make_layout(
        tla.make_shape(TILE_ROWS, N),
        tla.make_stride(N_PAD, 1),
        origin_shape=tla.make_shape(TILE_ROWS, N),
    )
    tile_gm = tla.make_layout(
        tla.make_shape(TILE_ROWS, N),
        tla.make_stride(N, 1),
        origin_shape=tla.make_shape(TILE_ROWS, N),
    )
    pack_ub = tla.make_layout(
        tla.make_shape(1, TILE_ROWS),
        tla.make_stride(1, 1),
        origin_shape=tla.make_shape(1, TILE_ROWS),
    )
    pack_gm = tla.make_layout(
        tla.make_shape(1, TILE_ROWS),
        tla.make_stride(TILE_ROWS, 1),
        origin_shape=tla.make_shape(1, TILE_ROWS),
    )
    ub_tile = [None] * BUFS
    ub_s = [None] * BUFS
    for bj in tla.range_constexpr(BUFS):
        ub_tile[bj] = tla.make_tensor(x_ptr[bj], tile_ub)
        ub_s[bj] = tla.make_tensor(s_ptr[bj], pack_ub)

    n_chunks = N // VL_ELE
    # Assign disjoint runs of consecutive tiles so every row is reduced on one AIV.
    n_tiles = m // TILE_ROWS
    gid = tla.arch.block_idx()
    aiv_num = tla.arch.block_num()
    tiles_per_block = (n_tiles + aiv_num - 1) // aiv_num
    tile_begin = gid * tiles_per_block
    block_limit = tile_begin + tiles_per_block
    tile_end = block_limit if block_limit < n_tiles else n_tiles

    with tla.vector():
        # Each slot passes from MTE2 to VECTOR to MTE3. Its next load waits
        # for writeback to finish, while the other slots can overlap transfers
        # and compute on different pipes.
        for grp in tla.range((tiles_per_block + BUFS - 1) // BUFS):
            for bi in tla.range_constexpr(BUFS):
                tile = tile_begin + grp * BUFS + bi
                # The final buffer group can contain slots owned by the next block.
                if tile < tile_end:
                    tile_off = tile.to(tla.Int64) * tla.Int64(TILE_ROWS * N)
                    gm_tile_x = tla.make_tensor(gm_x.ptr + tile_off, tile_gm)
                    out_off = tile.to(tla.Int64) * tla.Int64(TILE_ROWS)
                    gm_tile_s = tla.make_tensor(gm_out.ptr + out_off, pack_gm)

                    tile_mtx[bi].lock(pipe=tla.pipes.MTE2)
                    tla.copy(ub_tile[bi], gm_tile_x)
                    tile_mtx[bi].unlock(pipe=tla.pipes.MTE2)

                    tile_mtx[bi].lock(pipe=tla.pipes.VECTOR)
                    with tla.vec.func(mode="simd"):
                        mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                        if tla.const_expr(TAIL != 0):
                            tail_mask, _ = tla.update_mask(TAIL, dtype=tla.Float32)
                        else:
                            tail_mask = mask
                        lane = tla.arange(base=0, order="increase", dtype=tla.Int32).to(
                            tla.Float32, _CAST_NONE, mask
                        )

                        # Reduce full chunks lane-wise before the horizontal reduction.
                        # The tail is reduced separately so padded lanes and masked
                        # operations cannot affect the full-chunk accumulators.
                        if tla.const_expr(TILE_ROWS == 4):
                            # Independent row accumulators expose instruction-level
                            # parallelism. Dynamic chunk loops keep register demand
                            # bounded as the row width increases.
                            if tla.const_expr(OP == OP_LOGSUMEXP):
                                # Pass 1: per-row max, head chunks then tail.
                                mx0 = tla.full(MIN_F, dtype=tla.Float32)
                                mx1 = tla.full(MIN_F, dtype=tla.Float32)
                                mx2 = tla.full(MIN_F, dtype=tla.Float32)
                                mx3 = tla.full(MIN_F, dtype=tla.Float32)
                                for c1 in tla.range(n_chunks):
                                    mx0 = tla.max(
                                        mx0,
                                        _chunk(ub_tile[bi], 0, c1).load(),
                                        mask=mask,
                                    )
                                    mx1 = tla.max(
                                        mx1,
                                        _chunk(ub_tile[bi], 1, c1).load(),
                                        mask=mask,
                                    )
                                    mx2 = tla.max(
                                        mx2,
                                        _chunk(ub_tile[bi], 2, c1).load(),
                                        mask=mask,
                                    )
                                    mx3 = tla.max(
                                        mx3,
                                        _chunk(ub_tile[bi], 3, c1).load(),
                                        mask=mask,
                                    )
                                if tla.const_expr(TAIL != 0):
                                    tm0 = tla.max(
                                        tla.full(MIN_F, dtype=tla.Float32),
                                        _chunk(ub_tile[bi], 0, n_chunks).load(),
                                        mask=tail_mask,
                                    )
                                    tm1 = tla.max(
                                        tla.full(MIN_F, dtype=tla.Float32),
                                        _chunk(ub_tile[bi], 1, n_chunks).load(),
                                        mask=tail_mask,
                                    )
                                    tm2 = tla.max(
                                        tla.full(MIN_F, dtype=tla.Float32),
                                        _chunk(ub_tile[bi], 2, n_chunks).load(),
                                        mask=tail_mask,
                                    )
                                    tm3 = tla.max(
                                        tla.full(MIN_F, dtype=tla.Float32),
                                        _chunk(ub_tile[bi], 3, n_chunks).load(),
                                        mask=tail_mask,
                                    )
                                    b0 = tla.max(
                                        tla.full(
                                            mx0.reduce(tla.ReductionOp.MAX, mask=mask),
                                            dtype=tla.Float32,
                                        ),
                                        tla.full(
                                            tm0.reduce(
                                                tla.ReductionOp.MAX, mask=tail_mask
                                            ),
                                            dtype=tla.Float32,
                                        ),
                                        mask=mask,
                                    )
                                    b1 = tla.max(
                                        tla.full(
                                            mx1.reduce(tla.ReductionOp.MAX, mask=mask),
                                            dtype=tla.Float32,
                                        ),
                                        tla.full(
                                            tm1.reduce(
                                                tla.ReductionOp.MAX, mask=tail_mask
                                            ),
                                            dtype=tla.Float32,
                                        ),
                                        mask=mask,
                                    )
                                    b2 = tla.max(
                                        tla.full(
                                            mx2.reduce(tla.ReductionOp.MAX, mask=mask),
                                            dtype=tla.Float32,
                                        ),
                                        tla.full(
                                            tm2.reduce(
                                                tla.ReductionOp.MAX, mask=tail_mask
                                            ),
                                            dtype=tla.Float32,
                                        ),
                                        mask=mask,
                                    )
                                    b3 = tla.max(
                                        tla.full(
                                            mx3.reduce(tla.ReductionOp.MAX, mask=mask),
                                            dtype=tla.Float32,
                                        ),
                                        tla.full(
                                            tm3.reduce(
                                                tla.ReductionOp.MAX, mask=tail_mask
                                            ),
                                            dtype=tla.Float32,
                                        ),
                                        mask=mask,
                                    )
                                else:
                                    b0 = tla.full(
                                        mx0.reduce(tla.ReductionOp.MAX, mask=mask),
                                        dtype=tla.Float32,
                                    )
                                    b1 = tla.full(
                                        mx1.reduce(tla.ReductionOp.MAX, mask=mask),
                                        dtype=tla.Float32,
                                    )
                                    b2 = tla.full(
                                        mx2.reduce(tla.ReductionOp.MAX, mask=mask),
                                        dtype=tla.Float32,
                                    )
                                    b3 = tla.full(
                                        mx3.reduce(tla.ReductionOp.MAX, mask=mask),
                                        dtype=tla.Float32,
                                    )

                                # Reuse the UB tile for sum(exp(x - max)); both
                                # passes share the same GM input transfer.
                                a0 = tla.full(0.0, dtype=tla.Float32)
                                a1 = tla.full(0.0, dtype=tla.Float32)
                                a2 = tla.full(0.0, dtype=tla.Float32)
                                a3 = tla.full(0.0, dtype=tla.Float32)
                                for c2 in tla.range(n_chunks):
                                    a0 = tla.add(
                                        a0,
                                        tla.exp(
                                            tla.sub(
                                                _chunk(ub_tile[bi], 0, c2).load(),
                                                b0,
                                                mask=mask,
                                            ),
                                            mask=mask,
                                        ),
                                        mask=mask,
                                    )
                                    a1 = tla.add(
                                        a1,
                                        tla.exp(
                                            tla.sub(
                                                _chunk(ub_tile[bi], 1, c2).load(),
                                                b1,
                                                mask=mask,
                                            ),
                                            mask=mask,
                                        ),
                                        mask=mask,
                                    )
                                    a2 = tla.add(
                                        a2,
                                        tla.exp(
                                            tla.sub(
                                                _chunk(ub_tile[bi], 2, c2).load(),
                                                b2,
                                                mask=mask,
                                            ),
                                            mask=mask,
                                        ),
                                        mask=mask,
                                    )
                                    a3 = tla.add(
                                        a3,
                                        tla.exp(
                                            tla.sub(
                                                _chunk(ub_tile[bi], 3, c2).load(),
                                                b3,
                                                mask=mask,
                                            ),
                                            mask=mask,
                                        ),
                                        mask=mask,
                                    )
                                if tla.const_expr(TAIL != 0):
                                    ta0 = tla.add(
                                        tla.full(0.0, dtype=tla.Float32),
                                        tla.exp(
                                            tla.sub(
                                                _chunk(ub_tile[bi], 0, n_chunks).load(),
                                                b0,
                                                mask=tail_mask,
                                            ),
                                            mask=tail_mask,
                                        ),
                                        mask=tail_mask,
                                    )
                                    ta1 = tla.add(
                                        tla.full(0.0, dtype=tla.Float32),
                                        tla.exp(
                                            tla.sub(
                                                _chunk(ub_tile[bi], 1, n_chunks).load(),
                                                b1,
                                                mask=tail_mask,
                                            ),
                                            mask=tail_mask,
                                        ),
                                        mask=tail_mask,
                                    )
                                    ta2 = tla.add(
                                        tla.full(0.0, dtype=tla.Float32),
                                        tla.exp(
                                            tla.sub(
                                                _chunk(ub_tile[bi], 2, n_chunks).load(),
                                                b2,
                                                mask=tail_mask,
                                            ),
                                            mask=tail_mask,
                                        ),
                                        mask=tail_mask,
                                    )
                                    ta3 = tla.add(
                                        tla.full(0.0, dtype=tla.Float32),
                                        tla.exp(
                                            tla.sub(
                                                _chunk(ub_tile[bi], 3, n_chunks).load(),
                                                b3,
                                                mask=tail_mask,
                                            ),
                                            mask=tail_mask,
                                        ),
                                        mask=tail_mask,
                                    )
                                    l0 = tla.add(
                                        b0,
                                        tla.log(
                                            tla.add(
                                                tla.full(
                                                    a0.reduce(
                                                        tla.ReductionOp.ADD, mask=mask
                                                    ),
                                                    dtype=tla.Float32,
                                                ),
                                                tla.full(
                                                    ta0.reduce(
                                                        tla.ReductionOp.ADD,
                                                        mask=tail_mask,
                                                    ),
                                                    dtype=tla.Float32,
                                                ),
                                                mask=mask,
                                            ),
                                            mask=mask,
                                        ),
                                        mask=mask,
                                    )
                                    l1 = tla.add(
                                        b1,
                                        tla.log(
                                            tla.add(
                                                tla.full(
                                                    a1.reduce(
                                                        tla.ReductionOp.ADD, mask=mask
                                                    ),
                                                    dtype=tla.Float32,
                                                ),
                                                tla.full(
                                                    ta1.reduce(
                                                        tla.ReductionOp.ADD,
                                                        mask=tail_mask,
                                                    ),
                                                    dtype=tla.Float32,
                                                ),
                                                mask=mask,
                                            ),
                                            mask=mask,
                                        ),
                                        mask=mask,
                                    )
                                    l2 = tla.add(
                                        b2,
                                        tla.log(
                                            tla.add(
                                                tla.full(
                                                    a2.reduce(
                                                        tla.ReductionOp.ADD, mask=mask
                                                    ),
                                                    dtype=tla.Float32,
                                                ),
                                                tla.full(
                                                    ta2.reduce(
                                                        tla.ReductionOp.ADD,
                                                        mask=tail_mask,
                                                    ),
                                                    dtype=tla.Float32,
                                                ),
                                                mask=mask,
                                            ),
                                            mask=mask,
                                        ),
                                        mask=mask,
                                    )
                                    l3 = tla.add(
                                        b3,
                                        tla.log(
                                            tla.add(
                                                tla.full(
                                                    a3.reduce(
                                                        tla.ReductionOp.ADD, mask=mask
                                                    ),
                                                    dtype=tla.Float32,
                                                ),
                                                tla.full(
                                                    ta3.reduce(
                                                        tla.ReductionOp.ADD,
                                                        mask=tail_mask,
                                                    ),
                                                    dtype=tla.Float32,
                                                ),
                                                mask=mask,
                                            ),
                                            mask=mask,
                                        ),
                                        mask=mask,
                                    )
                                else:
                                    l0 = tla.add(
                                        b0,
                                        tla.log(
                                            tla.full(
                                                a0.reduce(
                                                    tla.ReductionOp.ADD, mask=mask
                                                ),
                                                dtype=tla.Float32,
                                            ),
                                            mask=mask,
                                        ),
                                        mask=mask,
                                    )
                                    l1 = tla.add(
                                        b1,
                                        tla.log(
                                            tla.full(
                                                a1.reduce(
                                                    tla.ReductionOp.ADD, mask=mask
                                                ),
                                                dtype=tla.Float32,
                                            ),
                                            mask=mask,
                                        ),
                                        mask=mask,
                                    )
                                    l2 = tla.add(
                                        b2,
                                        tla.log(
                                            tla.full(
                                                a2.reduce(
                                                    tla.ReductionOp.ADD, mask=mask
                                                ),
                                                dtype=tla.Float32,
                                            ),
                                            mask=mask,
                                        ),
                                        mask=mask,
                                    )
                                    l3 = tla.add(
                                        b3,
                                        tla.log(
                                            tla.full(
                                                a3.reduce(
                                                    tla.ReductionOp.ADD, mask=mask
                                                ),
                                                dtype=tla.Float32,
                                            ),
                                            mask=mask,
                                        ),
                                        mask=mask,
                                    )
                            else:
                                init = 0.0 if OP != OP_MAX else MIN_F
                                s0 = tla.full(init, dtype=tla.Float32)
                                s1 = tla.full(init, dtype=tla.Float32)
                                s2 = tla.full(init, dtype=tla.Float32)
                                s3 = tla.full(init, dtype=tla.Float32)
                                for c3 in tla.range(n_chunks):
                                    if tla.const_expr(OP == OP_MAX):
                                        s0 = tla.max(
                                            s0,
                                            _chunk(ub_tile[bi], 0, c3).load(),
                                            mask=mask,
                                        )
                                        s1 = tla.max(
                                            s1,
                                            _chunk(ub_tile[bi], 1, c3).load(),
                                            mask=mask,
                                        )
                                        s2 = tla.max(
                                            s2,
                                            _chunk(ub_tile[bi], 2, c3).load(),
                                            mask=mask,
                                        )
                                        s3 = tla.max(
                                            s3,
                                            _chunk(ub_tile[bi], 3, c3).load(),
                                            mask=mask,
                                        )
                                    else:
                                        s0 = tla.add(
                                            s0,
                                            _chunk(ub_tile[bi], 0, c3).load(),
                                            mask=mask,
                                        )
                                        s1 = tla.add(
                                            s1,
                                            _chunk(ub_tile[bi], 1, c3).load(),
                                            mask=mask,
                                        )
                                        s2 = tla.add(
                                            s2,
                                            _chunk(ub_tile[bi], 2, c3).load(),
                                            mask=mask,
                                        )
                                        s3 = tla.add(
                                            s3,
                                            _chunk(ub_tile[bi], 3, c3).load(),
                                            mask=mask,
                                        )
                                red = (
                                    tla.ReductionOp.MAX
                                    if OP == OP_MAX
                                    else tla.ReductionOp.ADD
                                )
                                if tla.const_expr(TAIL != 0):
                                    if tla.const_expr(OP == OP_MAX):
                                        ts0 = tla.max(
                                            tla.full(init, dtype=tla.Float32),
                                            _chunk(ub_tile[bi], 0, n_chunks).load(),
                                            mask=tail_mask,
                                        )
                                        ts1 = tla.max(
                                            tla.full(init, dtype=tla.Float32),
                                            _chunk(ub_tile[bi], 1, n_chunks).load(),
                                            mask=tail_mask,
                                        )
                                        ts2 = tla.max(
                                            tla.full(init, dtype=tla.Float32),
                                            _chunk(ub_tile[bi], 2, n_chunks).load(),
                                            mask=tail_mask,
                                        )
                                        ts3 = tla.max(
                                            tla.full(init, dtype=tla.Float32),
                                            _chunk(ub_tile[bi], 3, n_chunks).load(),
                                            mask=tail_mask,
                                        )
                                    else:
                                        ts0 = tla.add(
                                            tla.full(init, dtype=tla.Float32),
                                            _chunk(ub_tile[bi], 0, n_chunks).load(),
                                            mask=tail_mask,
                                        )
                                        ts1 = tla.add(
                                            tla.full(init, dtype=tla.Float32),
                                            _chunk(ub_tile[bi], 1, n_chunks).load(),
                                            mask=tail_mask,
                                        )
                                        ts2 = tla.add(
                                            tla.full(init, dtype=tla.Float32),
                                            _chunk(ub_tile[bi], 2, n_chunks).load(),
                                            mask=tail_mask,
                                        )
                                        ts3 = tla.add(
                                            tla.full(init, dtype=tla.Float32),
                                            _chunk(ub_tile[bi], 3, n_chunks).load(),
                                            mask=tail_mask,
                                        )
                                    h0 = tla.full(
                                        s0.reduce(red, mask=mask), dtype=tla.Float32
                                    )
                                    h1 = tla.full(
                                        s1.reduce(red, mask=mask), dtype=tla.Float32
                                    )
                                    h2 = tla.full(
                                        s2.reduce(red, mask=mask), dtype=tla.Float32
                                    )
                                    h3 = tla.full(
                                        s3.reduce(red, mask=mask), dtype=tla.Float32
                                    )
                                    q0 = tla.full(
                                        ts0.reduce(red, mask=tail_mask),
                                        dtype=tla.Float32,
                                    )
                                    q1 = tla.full(
                                        ts1.reduce(red, mask=tail_mask),
                                        dtype=tla.Float32,
                                    )
                                    q2 = tla.full(
                                        ts2.reduce(red, mask=tail_mask),
                                        dtype=tla.Float32,
                                    )
                                    q3 = tla.full(
                                        ts3.reduce(red, mask=tail_mask),
                                        dtype=tla.Float32,
                                    )
                                    if tla.const_expr(OP == OP_MAX):
                                        l0 = tla.max(h0, q0, mask=mask)
                                        l1 = tla.max(h1, q1, mask=mask)
                                        l2 = tla.max(h2, q2, mask=mask)
                                        l3 = tla.max(h3, q3, mask=mask)
                                    else:
                                        l0 = tla.add(h0, q0, mask=mask)
                                        l1 = tla.add(h1, q1, mask=mask)
                                        l2 = tla.add(h2, q2, mask=mask)
                                        l3 = tla.add(h3, q3, mask=mask)
                                else:
                                    l0 = tla.full(
                                        s0.reduce(red, mask=mask), dtype=tla.Float32
                                    )
                                    l1 = tla.full(
                                        s1.reduce(red, mask=mask), dtype=tla.Float32
                                    )
                                    l2 = tla.full(
                                        s2.reduce(red, mask=mask), dtype=tla.Float32
                                    )
                                    l3 = tla.full(
                                        s3.reduce(red, mask=mask), dtype=tla.Float32
                                    )
                                if tla.const_expr(OP == OP_MEAN):
                                    l0 = tla.mul(l0, INV_N, mask=mask)
                                    l1 = tla.mul(l1, INV_N, mask=mask)
                                    l2 = tla.mul(l2, INV_N, mask=mask)
                                    l3 = tla.mul(l3, INV_N, mask=mask)

                            # Pack four row results for one MTE3 transfer per tile.
                            packed = tla.full(0.0, dtype=tla.Float32)
                            packed = tla.where(tla.cmp(lane, 0, "eq"), l0, packed)
                            packed = tla.where(tla.cmp(lane, 1, "eq"), l1, packed)
                            packed = tla.where(tla.cmp(lane, 2, "eq"), l2, packed)
                            packed = tla.where(tla.cmp(lane, 3, "eq"), l3, packed)
                            tla.tile_view(
                                ub_s[bi],
                                tla.make_shape(1, TILE_ROWS),
                                tla.make_coord(0, 0),
                            ).store(packed)
                        else:
                            # A single-row tile keeps the same reduction and writes lane 0.
                            if tla.const_expr(OP == OP_LOGSUMEXP):
                                mx = tla.full(MIN_F, dtype=tla.Float32)
                                for d1 in tla.range(n_chunks):
                                    mx = tla.max(
                                        mx,
                                        _chunk(ub_tile[bi], 0, d1).load(),
                                        mask=mask,
                                    )
                                if tla.const_expr(TAIL != 0):
                                    tmx = tla.max(
                                        tla.full(MIN_F, dtype=tla.Float32),
                                        _chunk(ub_tile[bi], 0, n_chunks).load(),
                                        mask=tail_mask,
                                    )
                                    mb = tla.max(
                                        tla.full(
                                            mx.reduce(tla.ReductionOp.MAX, mask=mask),
                                            dtype=tla.Float32,
                                        ),
                                        tla.full(
                                            tmx.reduce(
                                                tla.ReductionOp.MAX, mask=tail_mask
                                            ),
                                            dtype=tla.Float32,
                                        ),
                                        mask=mask,
                                    )
                                else:
                                    mb = tla.full(
                                        mx.reduce(tla.ReductionOp.MAX, mask=mask),
                                        dtype=tla.Float32,
                                    )
                                ac = tla.full(0.0, dtype=tla.Float32)
                                for d2 in tla.range(n_chunks):
                                    ac = tla.add(
                                        ac,
                                        tla.exp(
                                            tla.sub(
                                                _chunk(ub_tile[bi], 0, d2).load(),
                                                mb,
                                                mask=mask,
                                            ),
                                            mask=mask,
                                        ),
                                        mask=mask,
                                    )
                                if tla.const_expr(TAIL != 0):
                                    tac = tla.add(
                                        tla.full(0.0, dtype=tla.Float32),
                                        tla.exp(
                                            tla.sub(
                                                _chunk(ub_tile[bi], 0, n_chunks).load(),
                                                mb,
                                                mask=tail_mask,
                                            ),
                                            mask=tail_mask,
                                        ),
                                        mask=tail_mask,
                                    )
                                    l0 = tla.add(
                                        mb,
                                        tla.log(
                                            tla.add(
                                                tla.full(
                                                    ac.reduce(
                                                        tla.ReductionOp.ADD, mask=mask
                                                    ),
                                                    dtype=tla.Float32,
                                                ),
                                                tla.full(
                                                    tac.reduce(
                                                        tla.ReductionOp.ADD,
                                                        mask=tail_mask,
                                                    ),
                                                    dtype=tla.Float32,
                                                ),
                                                mask=mask,
                                            ),
                                            mask=mask,
                                        ),
                                        mask=mask,
                                    )
                                else:
                                    l0 = tla.add(
                                        mb,
                                        tla.log(
                                            tla.full(
                                                ac.reduce(
                                                    tla.ReductionOp.ADD, mask=mask
                                                ),
                                                dtype=tla.Float32,
                                            ),
                                            mask=mask,
                                        ),
                                        mask=mask,
                                    )
                            else:
                                init = 0.0 if OP != OP_MAX else MIN_F
                                ac = tla.full(init, dtype=tla.Float32)
                                for d3 in tla.range(n_chunks):
                                    if tla.const_expr(OP == OP_MAX):
                                        ac = tla.max(
                                            ac,
                                            _chunk(ub_tile[bi], 0, d3).load(),
                                            mask=mask,
                                        )
                                    else:
                                        ac = tla.add(
                                            ac,
                                            _chunk(ub_tile[bi], 0, d3).load(),
                                            mask=mask,
                                        )
                                red = (
                                    tla.ReductionOp.MAX
                                    if OP == OP_MAX
                                    else tla.ReductionOp.ADD
                                )
                                if tla.const_expr(TAIL != 0):
                                    if tla.const_expr(OP == OP_MAX):
                                        tac = tla.max(
                                            tla.full(init, dtype=tla.Float32),
                                            _chunk(ub_tile[bi], 0, n_chunks).load(),
                                            mask=tail_mask,
                                        )
                                    else:
                                        tac = tla.add(
                                            tla.full(init, dtype=tla.Float32),
                                            _chunk(ub_tile[bi], 0, n_chunks).load(),
                                            mask=tail_mask,
                                        )
                                    if tla.const_expr(OP == OP_MAX):
                                        l0 = tla.max(
                                            tla.full(
                                                ac.reduce(red, mask=mask),
                                                dtype=tla.Float32,
                                            ),
                                            tla.full(
                                                tac.reduce(red, mask=tail_mask),
                                                dtype=tla.Float32,
                                            ),
                                            mask=mask,
                                        )
                                    else:
                                        l0 = tla.add(
                                            tla.full(
                                                ac.reduce(red, mask=mask),
                                                dtype=tla.Float32,
                                            ),
                                            tla.full(
                                                tac.reduce(red, mask=tail_mask),
                                                dtype=tla.Float32,
                                            ),
                                            mask=mask,
                                        )
                                else:
                                    l0 = tla.full(
                                        ac.reduce(red, mask=mask), dtype=tla.Float32
                                    )
                                if tla.const_expr(OP == OP_MEAN):
                                    l0 = tla.mul(l0, INV_N, mask=mask)
                            tla.tile_view(
                                ub_s[bi],
                                tla.make_shape(1, 1),
                                tla.make_coord(0, 0),
                            ).store(l0)
                    tile_mtx[bi].unlock(pipe=tla.pipes.VECTOR)

                    tile_mtx[bi].lock(pipe=tla.pipes.MTE3)
                    tla.copy(gm_tile_s, ub_s[bi])
                    tile_mtx[bi].unlock(pipe=tla.pipes.MTE3)


# ======================================================================================
# HOST side: shape checks, compile and launch.
# ======================================================================================

_COMPILED = {}
_COMPILE_LOCK = Lock()

_OP_NAMES = {"sum": OP_SUM, "max": OP_MAX, "mean": OP_MEAN, "logsumexp": OP_LOGSUMEXP}

_CORES = {}


def reset_stream_cache():
    """Compatibility entry point. CATLASS resolves the active stream per launch."""


def _pipeline(m, n_pad):
    """Choose the tile height and number of slots from the padded UB footprint.

    Group four rows when two complete slots fit and the row count is divisible
    by four. Each slot stores a padded input tile and an output staging vector.
    """
    row_bytes = n_pad * 4
    four_row_slot_bytes = 4 * row_bytes + _OUTPUT_TILE_BYTES
    tile_rows = 4 if m % 4 == 0 and 2 * four_row_slot_bytes <= UB_BYTES else 1
    slot_bytes = tile_rows * row_bytes + _OUTPUT_TILE_BYTES
    bufs = min(4, UB_BYTES // slot_bytes)
    return tile_rows, bufs


def _compile_plan(x, out, m, n, op_id, device_index):
    if device_index not in _CORES:
        props = torch.npu.get_device_properties(device_index)
        _CORES[device_index] = int(props.vector_core_num)

    n_pad = (n + VL_ELE - 1) // VL_ELE * VL_ELE
    tile_rows, bufs = _pipeline(m, n_pad)
    x_t = from_dlpack(x, layout_tag=tla.arch.RowMajor)
    out_t = from_dlpack(out, layout_tag=tla.arch.RowMajor)
    artifact = tla.compile(
        row_reduce_kernel,
        x_t,
        out_t,
        n,
        n_pad,
        n % VL_ELE,
        1.0 / n,
        op_id,
        tile_rows,
        bufs,
        options="--npu-arch 3510",
    )
    return artifact, x_t, out_t, _CORES[device_index], Lock()


def run(x, op="sum"):
    """Return the per-row reduction of ``x`` over the last axis as fp32 ``[M]``."""
    if x.dtype != torch.float32:
        raise ValueError("x must be float32")
    if x.dim() != 2:
        raise ValueError("x must be 2-D [M, N]")
    if not isinstance(op, str) or op not in _OP_NAMES:
        raise ValueError("op must be one of {}".format(sorted(_OP_NAMES)))
    m, n = x.shape
    if n < VL_ELE:
        raise ValueError("N must be >= {}, got {}".format(VL_ELE, n))
    if m == 0:
        return x.new_empty((0,))
    if n > _MAX_COLUMNS:
        raise ValueError(
            f"N must be <= {_MAX_COLUMNS} to fit one row and its output in UB, got {n}"
        )
    if x.device.type != "npu":
        raise ValueError("x must be an NPU tensor")

    device_index = x.device.index
    op_id = _OP_NAMES[op]
    key = (device_index, m, n, op_id)
    with torch.npu.device(device_index):
        if not x.is_contiguous():
            x = x.contiguous()
        out = torch.empty((m,), dtype=torch.float32, device=x.device)
        plan = _COMPILED.get(key)
        if plan is None:
            with _COMPILE_LOCK:
                plan = _COMPILED.get(key)
                if plan is None:
                    plan = _compile_plan(x, out, m, n, op_id, device_index)
                    _COMPILED[key] = plan

        artifact, x_t, out_t, cores, launch_lock = plan
        # Reuse the DLPack descriptors for this shape. Protect both address
        # updates until the launch has packed the arguments.
        with launch_lock:
            x_t.data_ptr = x.data_ptr()
            out_t.data_ptr = out.data_ptr()
            artifact(x_t, out_t, block_num=cores)
    return out


if __name__ == "__main__":
    torch.npu.set_device(0)
    torch.manual_seed(0)

    for m, n in ((64, 512), (37, 1000), (5, 65), (4096, 4096), (4096, 1003)):
        x = torch.randn(m, n, dtype=torch.float32, device="npu")
        xc = x.cpu()
        ref = {
            "sum": torch.sum(xc, dim=-1),
            "max": torch.amax(xc, dim=-1),
            "mean": torch.mean(xc, dim=-1),
            "logsumexp": torch.logsumexp(xc, dim=-1),
        }
        for name in ("sum", "max", "mean", "logsumexp"):
            got = run(x, name).cpu()
            print(
                "shape",
                (m, n),
                name,
                "max abs err",
                (got - ref[name]).abs().max().item(),
            )
