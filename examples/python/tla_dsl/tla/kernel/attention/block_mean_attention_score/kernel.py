# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Single-pass K @ Q.T with per-query deferred normalization."""

from __future__ import annotations

import catlass.tla as tla
from helpers import TilingParams
from .constants import BLOCK_TOKENS, align_up
from .stage_qk import (
    QUERY_ALIGNMENT_TOKENS,
    C_SLICE_ROWS,
    UB_SCORE_STRIDE,
    allocate_qk,
    prepare_query,
    produce_qk,
    copy_scores,
)
from .stage_vector import (
    COUNTS_CACHE_ELEMS,
    COUNTS_LOAD_ELEMS,
    allocate_vector,
    initialize_counts,
    prepare_vector,
    phase_one,
    finalize,
)


@tla.kernel
def block_mean_attention_score_kernel(
    q: tla.Tensor,
    k: tla.Tensor,
    counts: tla.Tensor,
    out: tla.Tensor,
    ws: tla.Tensor,
    _h: tla.Constexpr[int],
    _d: tla.Constexpr[int],
    _tiling: tla.Constexpr[TilingParams],
) -> None:
    qbufs = allocate_qk()
    blocks = counts.shape[0]
    _s = blocks * BLOCK_TOKENS
    # A full vector load near the end needs guard elements beyond valid counts.
    counts_guard = COUNTS_LOAD_ELEMS - 1
    counts_cache_elems = align_up(blocks + counts_guard, COUNTS_LOAD_ELEMS)
    cache_counts = counts_cache_elems <= COUNTS_CACHE_ELEMS
    vbufs = allocate_vector(COUNTS_CACHE_ELEMS, _tiling.l1_tm)
    workspace_mutex = vbufs.mw
    ready0 = tla.cross_flag("block_mean_attention_score_score_ready_0", mode=4)
    free0 = tla.cross_flag("block_mean_attention_score_score_free_0", mode=4)
    ready1 = tla.cross_flag("block_mean_attention_score_score_ready_1", mode=4)
    free1 = tla.cross_flag("block_mean_attention_score_score_free_1", mode=4)
    query_tiles = (_s + _tiling.l1_tn - 1) // _tiling.l1_tn
    key_tiles = (_s + _tiling.l1_tm - 1) // _tiling.l1_tm
    scale = _d**-0.5

    with tla.vector():
        initialize_counts(counts, vbufs, blocks, counts_cache_elems, cache_counts)
        tla.cross_core_set_flag(free0, tla.arch.VECTOR, aiv_id=0)
        tla.cross_core_set_flag(free0, tla.arch.VECTOR, aiv_id=1)
        tla.cross_core_set_flag(free1, tla.arch.VECTOR, aiv_id=0)
        tla.cross_core_set_flag(free1, tla.arch.VECTOR, aiv_id=1)

    for task in tla.range(tla.arch.block_idx(), _h * query_tiles, tla.arch.block_num()):
        # Keep consecutive Query tiles of one head together for K-cache reuse.
        head = task // query_tiles
        query_start = (task % query_tiles) * _tiling.l1_tn
        nq = _s - query_start if _s - query_start < _tiling.l1_tn else _tiling.l1_tn
        # SPLIT_N gives each of the two AIVs a 16-aligned Query half.
        nq_pad = align_up(nq, QUERY_ALIGNMENT_TOKENS)
        with tla.cube():
            prepare_query(q, qbufs, head, query_start, nq, nq_pad, _s, _d)
        with tla.vector():
            prepare_vector(counts, vbufs, query_start, nq, nq_pad, cache_counts)
            # One GM statistics region is shared by all tasks of this group.
            # Hold its write ownership until the last phase-one DMA is issued.
            workspace_mutex.lock(pipe=tla.arch.MTE3)
        for key_tile in tla.range(key_tiles):
            key_start = key_tile * _tiling.l1_tm
            mk = _s - key_start if _s - key_start < _tiling.l1_tm else _tiling.l1_tm
            serial = task * key_tiles + key_tile
            score_ptr = vbufs.score0 if serial % 2 == 0 else vbufs.score1
            ready = ready0 if serial % 2 == 0 else ready1
            free = free0 if serial % 2 == 0 else free1
            with tla.cube():
                for cpart in tla.range(
                    (_tiling.l1_tm + C_SLICE_ROWS - 1) // C_SLICE_ROWS
                ):
                    slice_start = key_start + cpart * C_SLICE_ROWS
                    slice_rows = mk - cpart * C_SLICE_ROWS
                    slice_rows = (
                        slice_rows if slice_rows < C_SLICE_ROWS else C_SLICE_ROWS
                    )
                    if slice_rows > 0:
                        slice_serial = serial * 2 + cpart
                        produce_qk(
                            q,
                            k,
                            qbufs,
                            head,
                            query_start,
                            slice_start,
                            nq,
                            nq_pad,
                            slice_rows,
                            _s,
                            _d,
                            slice_serial,
                        )
                        if cpart == 0:
                            tla.cross_core_wait_flag(free, tla.arch.FIX, aiv_id=0)
                            tla.cross_core_wait_flag(free, tla.arch.FIX, aiv_id=1)
                        copy_scores(
                            qbufs,
                            score_ptr + cpart * C_SLICE_ROWS * UB_SCORE_STRIDE,
                            slice_rows,
                            nq_pad,
                            slice_serial,
                        )
                tla.cross_core_set_flag(ready, tla.arch.FIX, aiv_id=0)
                tla.cross_core_set_flag(ready, tla.arch.FIX, aiv_id=1)
            with tla.vector():
                tla.cross_core_wait_flag(ready, tla.arch.VECTOR, aiv_id=0)
                tla.cross_core_wait_flag(ready, tla.arch.VECTOR, aiv_id=1)
                phase_one(
                    counts,
                    ws,
                    vbufs,
                    score_ptr,
                    key_start,
                    mk,
                    scale,
                    key_tile,
                    serial,
                    blocks,
                    key_tiles,
                    cache_counts,
                    _tiling.l1_tm,
                )
                tla.cross_core_set_flag(free, tla.arch.VECTOR, aiv_id=0)
                tla.cross_core_set_flag(free, tla.arch.VECTOR, aiv_id=1)
        with tla.vector():
            workspace_mutex.unlock(pipe=tla.arch.MTE3)
            finalize(
                counts,
                ws,
                out,
                vbufs,
                head,
                query_start,
                nq,
                nq_pad,
                blocks,
                key_tiles,
                cache_counts,
                _tiling.l1_tm,
            )

    with tla.cube():
        tla.cross_core_wait_flag(free0, tla.arch.FIX, aiv_id=0)
        tla.cross_core_wait_flag(free0, tla.arch.FIX, aiv_id=1)
        tla.cross_core_wait_flag(free1, tla.arch.FIX, aiv_id=0)
        tla.cross_core_wait_flag(free1, tla.arch.FIX, aiv_id=1)
        # All scores have been consumed. Join the two L0C owners explicitly.
        for mutex in (qbufs.mc0, qbufs.mc1):
            mutex.lock(pipe=tla.arch.SCALAR)
            mutex.unlock(pipe=tla.arch.SCALAR)
    with tla.vector():
        # Resource-specific joins finish staging, output DMA, workspace reads,
        # and running statistics without draining unrelated pipelines.
        for mutex in (vbufs.mo0, vbufs.mo1, vbufs.ms0, vbufs.ms1, vbufs.mw, vbufs.mr):
            mutex.lock(pipe=tla.arch.SCALAR)
            mutex.unlock(pipe=tla.arch.SCALAR)
