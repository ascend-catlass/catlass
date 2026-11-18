# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Standalone operator: matmul + row-wise exp + selected-column scatter-add.

For ``x [M, H]``, ``weight [V, H]``, ``row_logsumexp [M]``,
``global_col_ids [M, S]`` and ``values [M, S]``, compute:

    C = bf16_rne(x @ weight.T)                  # fp32 accumulation
    E = exp(float32(C) * inv_temperature - row_logsumexp[:, None])
    R = -E * values.sum(dim=-1, keepdim=True)
    R[m, global_col_ids[m, s] - col_start] += values[m, s]
    Y = bf16_rne(R * inv_temperature)           # [M, V] bf16

Here ``S = num_selected``, ``inv_temperature = 1 / temperature`` and
``col_start = shard_index * V``.  The scatter-add applies only to global column
ids in ``[col_start, col_start + V)``; repeated ids accumulate in slot order.
All ``values`` participate in the row sum, including entries whose column ids
are outside this shard.  The operator accepts arbitrary selected columns,
including ``S = 0``, and does not compute or require a top-k ordering.

``row_logsumexp`` is supplied by the caller; the operator does not compute it.
It is a per-row log-normalization offset for the temperature-scaled matmul
result.  When it covers the full column range, ``E`` is the corresponding
column shard of that normalized exponential; its local row sum need not be 1.

Public API
----------
``run(x, weight, global_col_ids, row_logsumexp, values, temperature,
shard_index=0, out=None)`` -> ``Y [M, V] bf16``.

Both matrix operands are bf16, column ids are int32, and ``row_logsumexp`` and
``values`` are fp32.  ``weight`` stores one column shard as ``[V, H]``; the
matmul uses its transpose.  ``run`` returns a view into a padded output buffer,
which may be noncontiguous.  Optional ``out`` supplies that full
``[M_pad, V_pad]`` bf16 buffer.

The host layer only allocates (once, zero-initialised) staging buffers,
compiles (cached per shape/device) and launches.  Padding, global-to-local
column id conversion and the negative row sum run on device in the V0
prologue of the same kernel.  ``prepare``/``launch`` expose the same work
split for benchmarking the fused device kernel without preparation overhead.

The matmul accumulates in fp32 and is rounded to bf16 (ties to even) before
the fp32 epilogue.  The output is rounded to bf16 again after the final scale.

Kernel structure
----------------
One launch, one kernel, three stages:

    V0 prologue  (all 56 AIVs, vector side only)
                    x[M,H]      -> x_pad[M_pad,H_pad]
                    weight[V,H] -> w_pad[V_pad,H_pad]
                    lse[M]      -> lse_pad[M_pad]
                    ids[M,S]    -> local ids, out-of-shard/pad rows -> -1
                    values[M,S] -> values_pad[M_pad,S]
                    neg_row_sum -> -sum_s values, via the vector reduce
                 then a mode-4 AIV->AIC arrive, a mode-0 all-AIC barrier and
                 the existing tile_ready handshake release the cube half.

The V0 prologue is the in-file reuse section: ``pad_rowsum_prepass`` runs in
the vector half and ``wait_prepass`` gates the cube half.  When the inputs are already tile-aligned and ``S <= 64``, the host
aliases x/weight/lse/values as their own staging buffers and the ``alias``
compile-time switch turns the four copies off; V0 then only localises ids and
reduces the row sums.  Non-aligned shapes keep the full prologue.

    cube half    for each (m, v) tile:  K-blocked matmul into L0C
                                        -> fixpipe copy L0C -> UB (fp32)
    vector half  for each (m, v) tile:  epilogue in place in UB
                                        -> pack bf16 in UB -> DMA Y to GM

The padded buffers are allocated once on the host and zero-initialized; V0
rewrites only the valid region, so the column padding of ``x_pad``/``w_pad``
stays zero and is never re-cleared.  The cube/vector halves walk the same task
list, so they stay in lock step.  Every stage holds exactly one tile (one L0C
accumulator, one UB staging tile), which gives a one-matmul lookahead: the
vector epilogue of tile ``t-1`` runs while the cube executes the MMAD chain of
tile ``t``, and the fixpipe copy of tile ``t`` then waits for that epilogue to
release UB.
"""

import torch
import torch_npu  # noqa: F401

import catlass.tla as tla
from catlass.tla.runtime import from_dlpack

# --------------------------------------------------------------------------- #
# Tile configuration
# --------------------------------------------------------------------------- #
# Matmul dimensions are M x N x K, with N = V and K = H.  The selected-column
# count S is separate and is passed as num_selected.
TILE_M = 256  # C tile rows; also the M padding granularity
TILE_N = 256  # C tile columns; also the V padding granularity
TILE_K = 256  # H padding granularity; the L1 K block is selected per shape
K_STEP = 64  # depth of one L0A/L0B fragment inside a K block


# Schedule constants depend on both grid dimensions.  Keep padding independent
# of the L1 K block so callers can reuse the same padded output contract.
def _select_schedule(m, v):
    """Return (swizzle_m, swizzle_n, tile_k) at kernel specialisation time."""
    m_tiles = (m + TILE_M - 1) // TILE_M
    v_tiles = (v + TILE_N - 1) // TILE_N
    # Empirical thresholds on Ascend 950PR (28 Cube Cores), H=2560.
    # Large M alone does not imply that the 32x2 mapping improves L2 reuse:
    # V=16384 favours 2x8, V=32768 favours 8x4, and V>=65536 favours 32x2.
    if m_tiles >= 32 and v_tiles >= 256:
        return 32, 2, 128
    if m_tiles >= 8 and v_tiles >= 128:
        return 8, 4, 128
    tile_k = 128 if m_tiles >= 32 else 256
    # Avoid unused super-tile slots when either grid dimension is tiny.
    return max(1, min(m_tiles, 2)), max(1, min(v_tiles, 8)), tile_k


_COMPILE_OPTS = (
    "--npu-arch 3510"
    " --cce-disable-asc-reserved-ubuf"
    " --cce-disable-vf-stack-reserved-ubuf"
)


def _pad_to(value, multiple):
    return ((value + multiple - 1) // multiple) * multiple


# =============================================================================
# 复用区：前段模板（pad 搬运 + ids 本地化 + 可选 row_sum 归约）
#
# 调用顺序：
#   1) vector 段先调用 pad_rowsum_prepass(...)：完成 x/weight/lse/values 的
#      padding（含对齐短路、H 非 16 倍数时的 SIMT 尾部）、ids 本地化
#      （减 col_start，片外与片尾行置 -1），可选地用硬件 64-lane reduce 求
#      -sum(values)，最后发 mode-4 到达标志；
#   2) cube 段开头调用 wait_prepass(...)：等本核两个 AIV 到达（mode 4），
#      再做全 AIC 的 mode-0 栅栏，通过后发 mode-2 放行/复位。
#   两段之间以及各自前后均保留 pipe_barrier(ALL)。
#
# 对外参数（其余 M/H/V/S 一律从 tensor shape 推导，不暴露）：
#   col_start            : 本分片的全局列起点，本地 id = id - col_start
#   tile_m/tile_k/tile_n : 256 对齐粒度，必须与后续 GEMM 的 tile 一致
#   reduce               : 是否计算 -sum(values) 并写入 gm_neg_sum_pad
#   alias                : 是否启用对齐短路。仅当 M/H/V 都是 tile 的整数倍、
#                          S <= 64 且输入连续、base 指针满足 DMA 对齐时置 True；
#                          此时四个输入直接别名，V0 只做 ids 与 row_sum。
#                          False 时按维度零拷贝（shape 已对齐则不搬），否则
#                          落到 host 分配的 staging 缓冲搬运。
#
# 复用方式：整文件复制后替换 cube 段 GEMM 与 vector 段 epilogue，
# 保留本复用区、kernel 骨架与 host 侧 staging/编译/launch。
# =============================================================================


# Internal staging sizes.  They only affect performance, not the result.
_PREPASS_TILE_R = 16  # rows per MTE staging tile
_PREPASS_COL_TILE_MAX = 256  # columns per MTE staging tile
_PREPASS_LSE_CHUNK = 1024  # fp32 elements per lse chunk
_PREPASS_SIMT_THREADS = 128  # threads per SIMT block


@tla.jit
def pad_rowsum_prepass(
    gm_x_raw: tla.Tensor,  # [M, H] bf16
    gm_weight_raw: tla.Tensor,  # [V, H] bf16
    gm_lse_raw: tla.Tensor,  # [M] fp32
    gm_x_pad: tla.Tensor,  # [M_pad, H_pad] bf16
    gm_w_pad: tla.Tensor,  # [V_pad, H_pad] bf16, row-major
    gm_lse_pad: tla.Tensor,  # [M_pad, 1] fp32 (pointer is enough)
    gm_neg_sum_pad: tla.Tensor,  # [M_pad, 1] fp32
    col_start: int,
    tile_m: int,
    tile_k: int,
    tile_n: int,
    reduce: bool,
    alias: bool,
    pad_arrive,
    gm_ids_raw: tla.Tensor = None,  # [M, S] int32, None when S == 0
    gm_values_raw: tla.Tensor = None,  # [M, S] fp32, None when S == 0
    gm_ids_pad: tla.Tensor = None,  # [M_pad, S] int32
    gm_values_pad: tla.Tensor = None,  # [M_pad, S] fp32
) -> None:
    zero, one = 0, 1
    m_raw = gm_x_raw.origin_shape[0]
    h_raw = gm_x_raw.origin_shape[1]
    v_raw = gm_weight_raw.origin_shape[0]
    m_pad = gm_x_pad.origin_shape[0]
    h_pad = gm_x_pad.origin_shape[1]
    v_pad = gm_w_pad.origin_shape[0]
    num_selected = gm_ids_raw.origin_shape[1] if gm_ids_raw is not None else 0

    # ---- staging constants (all compile-time).
    h16 = (h_raw // 16) * 16  # BF16 32-byte granularity: 16 elements
    col_tile = min(_PREPASS_COL_TILE_MAX, h16) if h16 > 0 else 0
    val_pad = ((num_selected + 63) // 64) * 64
    val_chunks = val_pad // 64
    val_rows = 16 if val_pad > 64 else 32
    # ids staging: at most 4 KiB of int32 per chunk.
    id_rows = min(128, max(1, 1024 // num_selected)) if num_selected > 0 else 1

    gid = tla.arch.block_idx() * 2 + tla.arch.sub_block_idx()
    aiv_num = tla.arch.block_num() * 2

    # When alias is set the caller guarantees all four staging buffers *are*
    # the raw inputs; otherwise a buffer is copied exactly when its shape is
    # not already padded (the host aliases the same tensors under the same
    # condition).
    copy_x = (not alias) and (m_raw != m_pad or h_raw != h_pad)
    copy_w = (not alias) and (v_raw != v_pad or h_raw != h_pad)
    copy_lse = (not alias) and (m_raw != m_pad)
    copy_values = (not alias) and (m_raw != m_pad)

    # ---- UB scratch.
    ub_pad_ptr = None
    if tla.const_expr(h16 > 0):
        ub_pad_ptr = tla.allocate(
            _PREPASS_TILE_R * col_tile, gm_x_raw.ptr.dtype, tla.AddressSpace.ub, 256
        )
    ub_lse_ptr = tla.allocate(_PREPASS_LSE_CHUNK, tla.Float32, tla.AddressSpace.ub, 256)
    ub_vals_ptr = None
    ub_red_ptr = None
    ub_ids_ptr = None
    if tla.const_expr(num_selected > 0):
        ub_vals_ptr = tla.allocate(
            val_rows * val_pad, tla.Float32, tla.AddressSpace.ub, 256
        )
        ub_red_ptr = tla.allocate(val_rows * 8, tla.Float32, tla.AddressSpace.ub, 256)
        if tla.const_expr(num_selected % 8 == 0):
            ub_ids_ptr = tla.allocate(
                id_rows * num_selected, tla.Int32, tla.AddressSpace.ub, 256
            )

    # ---- x[M, H] -> x_pad[M_pad, H_pad].
    if tla.const_expr(h16 > 0 and copy_x):
        gm_x_rows = (m_raw + _PREPASS_TILE_R - 1) // _PREPASS_TILE_R
        col_tiles = (h16 + col_tile - 1) // col_tile
        x_tiles = gm_x_rows * col_tiles
        for xt in tla.range(gid, x_tiles, aiv_num):
            row_tile = xt // col_tiles
            col_t = xt % col_tiles
            row0 = row_tile * _PREPASS_TILE_R
            col0 = col_t * col_tile
            rows = _PREPASS_TILE_R
            if m_raw - row0 < _PREPASS_TILE_R:
                rows = m_raw - row0
            cols = col_tile
            if h16 - col0 < col_tile:
                cols = h16 - col0
            gm_src = tla.make_tensor(
                gm_x_raw.ptr
                + row0.to(tla.Int64) * tla.Int64(h_raw)
                + col0.to(tla.Int64),
                tla.make_layout(
                    tla.make_shape(_PREPASS_TILE_R, col_tile),
                    tla.make_stride(h_raw, 1),
                    origin_shape=tla.make_shape(rows, cols),
                ),
            )
            gm_dst = tla.make_tensor(
                gm_x_pad.ptr
                + row0.to(tla.Int64) * tla.Int64(h_pad)
                + col0.to(tla.Int64),
                tla.make_layout(
                    tla.make_shape(_PREPASS_TILE_R, col_tile),
                    tla.make_stride(h_pad, 1),
                    origin_shape=tla.make_shape(rows, cols),
                ),
            )
            ub_tile = tla.make_tensor(
                ub_pad_ptr,
                tla.make_layout(
                    tla.make_shape(_PREPASS_TILE_R, col_tile),
                    tla.make_stride(col_tile, 1),
                    origin_shape=tla.make_shape(rows, cols),
                ),
            )
            tla.copy(ub_tile, gm_src)
            tla.pipe_barrier(tla.pipes.ALL)
            tla.copy(gm_dst, ub_tile)
            tla.pipe_barrier(tla.pipes.ALL)

    # ---- weight[V, H] -> w_pad[V_pad, H_pad], same schedule.
    if tla.const_expr(h16 > 0 and copy_w):
        w_col_tiles = (h16 + col_tile - 1) // col_tile
        gm_w_rows = (v_raw + _PREPASS_TILE_R - 1) // _PREPASS_TILE_R
        w_tiles = gm_w_rows * w_col_tiles
        for wt in tla.range(gid, w_tiles, aiv_num):
            w_row_tile = wt // w_col_tiles
            w_col_t = wt % w_col_tiles
            w_row0 = w_row_tile * _PREPASS_TILE_R
            w_col0 = w_col_t * col_tile
            w_rows = _PREPASS_TILE_R
            if v_raw - w_row0 < _PREPASS_TILE_R:
                w_rows = v_raw - w_row0
            w_cols = col_tile
            if h16 - w_col0 < col_tile:
                w_cols = h16 - w_col0
            gm_w_src = tla.make_tensor(
                gm_weight_raw.ptr
                + w_row0.to(tla.Int64) * tla.Int64(h_raw)
                + w_col0.to(tla.Int64),
                tla.make_layout(
                    tla.make_shape(_PREPASS_TILE_R, col_tile),
                    tla.make_stride(h_raw, 1),
                    origin_shape=tla.make_shape(w_rows, w_cols),
                ),
            )
            gm_w_dst = tla.make_tensor(
                gm_w_pad.ptr
                + w_row0.to(tla.Int64) * tla.Int64(h_pad)
                + w_col0.to(tla.Int64),
                tla.make_layout(
                    tla.make_shape(_PREPASS_TILE_R, col_tile),
                    tla.make_stride(h_pad, 1),
                    origin_shape=tla.make_shape(w_rows, w_cols),
                ),
            )
            ub_w_tile = tla.make_tensor(
                ub_pad_ptr,
                tla.make_layout(
                    tla.make_shape(_PREPASS_TILE_R, col_tile),
                    tla.make_stride(col_tile, 1),
                    origin_shape=tla.make_shape(w_rows, w_cols),
                ),
            )
            tla.copy(ub_w_tile, gm_w_src)
            tla.pipe_barrier(tla.pipes.ALL)
            tla.copy(gm_w_dst, ub_w_tile)
            tla.pipe_barrier(tla.pipes.ALL)

    # ---- x tail columns (H % 16): SIMT element-wise fallback.
    if tla.const_expr(h16 < h_raw):
        with tla.vec.func(mode="simt", thread_block_dim=_PREPASS_SIMT_THREADS):
            tid, _, _ = tla.arch.thread_idx()
            gtid = gid * _PREPASS_SIMT_THREADS + tid
            gthreads = aiv_num * _PREPASS_SIMT_THREADS
            for xtr in tla.range(gtid, m_raw, gthreads):
                for xtc in tla.range(h16, h_raw, one):
                    gm_x_pad[xtr, xtc] = gm_x_raw[xtr, xtc]
        tla.pipe_barrier(tla.pipes.ALL)

    # ---- weight tail columns (H % 16): SIMT element-wise fallback.
    if tla.const_expr(h16 < h_raw):
        with tla.vec.func(mode="simt", thread_block_dim=_PREPASS_SIMT_THREADS):
            tid, _, _ = tla.arch.thread_idx()
            gtid = gid * _PREPASS_SIMT_THREADS + tid
            gthreads = aiv_num * _PREPASS_SIMT_THREADS
            for wtr in tla.range(gtid, v_raw, gthreads):
                for wtc in tla.range(h16, h_raw, one):
                    gm_w_pad[wtr, wtc] = gm_weight_raw[wtr, wtc]
        tla.pipe_barrier(tla.pipes.ALL)

    # ---- lse [M] -> lse_pad [M_pad].
    if tla.const_expr(copy_lse):
        lse_chunks = (m_raw + _PREPASS_LSE_CHUNK - 1) // _PREPASS_LSE_CHUNK
        for lt in tla.range(gid, lse_chunks, aiv_num):
            lse_base = lt * _PREPASS_LSE_CHUNK
            lse_n = _PREPASS_LSE_CHUNK
            if m_raw - lse_base < _PREPASS_LSE_CHUNK:
                lse_n = m_raw - lse_base
            gm_lse_src = tla.make_tensor(
                gm_lse_raw.ptr + lse_base,
                tla.make_layout(
                    tla.make_shape(_PREPASS_LSE_CHUNK),
                    tla.make_stride(1),
                    origin_shape=tla.make_shape(lse_n),
                ),
            )
            gm_lse_dst = tla.make_tensor(
                gm_lse_pad.ptr + lse_base,
                tla.make_layout(
                    tla.make_shape(_PREPASS_LSE_CHUNK),
                    tla.make_stride(1),
                    origin_shape=tla.make_shape(lse_n),
                ),
            )
            ub_lse = tla.make_tensor(
                ub_lse_ptr,
                tla.make_layout(
                    tla.make_shape(_PREPASS_LSE_CHUNK),
                    tla.make_stride(1),
                    origin_shape=tla.make_shape(lse_n),
                ),
            )
            tla.copy(ub_lse, gm_lse_src)
            tla.pipe_barrier(tla.pipes.ALL)
            tla.copy(gm_lse_dst, ub_lse)
            tla.pipe_barrier(tla.pipes.ALL)

    # ---- ids: global -> shard-local, out-of-shard and pad rows -> -1.
    if tla.const_expr(num_selected > 0):
        if tla.const_expr(num_selected % 8 == 0):
            # Rows are 32-byte aligned: MTE the chunk into UB, rewrite the
            # ids in place with scalar UB accesses, then MTE it back.
            ub_ids = tla.make_tensor(
                ub_ids_ptr,
                tla.make_layout(
                    tla.make_shape(id_rows, num_selected),
                    tla.make_stride(num_selected, 1),
                ),
            )
            id_chunk_count = (m_raw + id_rows - 1) // id_rows
            for ic in tla.range(gid, id_chunk_count, aiv_num):
                ir0 = ic * id_rows
                inn = id_rows
                if m_raw - ir0 < id_rows:
                    inn = m_raw - ir0
                gm_id_src = tla.make_tensor(
                    gm_ids_raw.ptr + ir0.to(tla.Int64) * tla.Int64(num_selected),
                    tla.make_layout(
                        tla.make_shape(id_rows, num_selected),
                        tla.make_stride(num_selected, 1),
                        origin_shape=tla.make_shape(inn, num_selected),
                    ),
                )
                ub_id_chunk = tla.make_tensor(
                    ub_ids_ptr,
                    tla.make_layout(
                        tla.make_shape(id_rows, num_selected),
                        tla.make_stride(num_selected, 1),
                        origin_shape=tla.make_shape(inn, num_selected),
                    ),
                )
                tla.copy(ub_id_chunk, gm_id_src)
                tla.pipe_barrier(tla.pipes.ALL)
                with tla.vec.func(mode="simt", thread_block_dim=_PREPASS_SIMT_THREADS):
                    tid, _, _ = tla.arch.thread_idx()
                    # Flatten (row, slot): with a row-per-thread loop only
                    # ``id_rows`` lanes work (8 for S=128), which is far too
                    # serial for the rewrite.
                    id_elems = inn * num_selected
                    for ii in tla.range(tid, id_elems, _PREPASS_SIMT_THREADS):
                        id_row = ii // num_selected
                        id_sel = ii % num_selected
                        id_val = ub_ids[id_row, id_sel]
                        loc = id_val - col_start
                        if loc < 0:
                            loc = -1
                        if loc >= v_raw:
                            loc = -1
                        ub_ids[id_row, id_sel] = loc
                tla.pipe_barrier(tla.pipes.ALL)
                gm_id_dst = tla.make_tensor(
                    gm_ids_pad.ptr + ir0.to(tla.Int64) * tla.Int64(num_selected),
                    tla.make_layout(
                        tla.make_shape(id_rows, num_selected),
                        tla.make_stride(num_selected, 1),
                        origin_shape=tla.make_shape(inn, num_selected),
                    ),
                )
                tla.copy(gm_id_dst, ub_id_chunk)
                tla.pipe_barrier(tla.pipes.ALL)
        else:
            with tla.vec.func(mode="simt", thread_block_dim=_PREPASS_SIMT_THREADS):
                tid, _, _ = tla.arch.thread_idx()
                gtid = gid * _PREPASS_SIMT_THREADS + tid
                gthreads = aiv_num * _PREPASS_SIMT_THREADS
                # Flatten (row, slot) so adjacent threads touch adjacent
                # elements; the per-row loop is latency bound on GM.
                id_total = m_raw * num_selected
                for ii in tla.range(gtid, id_total, gthreads):
                    id_row = ii // num_selected
                    id_sel = ii % num_selected
                    id_val = gm_ids_raw[id_row, id_sel]
                    loc = id_val - col_start
                    if loc < 0:
                        loc = -1
                    if loc >= v_raw:
                        loc = -1
                    gm_ids_pad[id_row, id_sel] = loc
            tla.pipe_barrier(tla.pipes.ALL)
        # Pad rows are processed by the full CV segment, so their ids must
        # never hit a column.
        with tla.vec.func(mode="simt", thread_block_dim=_PREPASS_SIMT_THREADS):
            tid, _, _ = tla.arch.thread_idx()
            gtid = gid * _PREPASS_SIMT_THREADS + tid
            gthreads = aiv_num * _PREPASS_SIMT_THREADS
            for pr in tla.range(m_raw + gtid, m_pad, gthreads):
                for psel in tla.range(0, num_selected, one):
                    gm_ids_pad[pr, psel] = -1
        tla.pipe_barrier(tla.pipes.ALL)

        # ---- values: copy to the padded buffer and optionally reduce.
        ub_vals = tla.make_tensor(
            ub_vals_ptr,
            tla.make_layout(
                tla.make_shape(val_rows, val_pad),
                tla.make_stride(val_pad, 1),
            ),
        )
        ub_red = tla.make_tensor(
            ub_red_ptr,
            tla.make_layout(tla.make_shape(val_rows * 8), tla.make_stride(1)),
        )
        val_chunk_count = (m_raw + val_rows - 1) // val_rows
        for vc in tla.range(gid, val_chunk_count, aiv_num):
            vr0 = vc * val_rows
            vn = val_rows
            if m_raw - vr0 < val_rows:
                vn = m_raw - vr0
            if tla.const_expr(num_selected % 8 == 0):
                # A row of 32-bit elements is 32-byte aligned, so the whole
                # chunk moves with MTE.  Only the lane padding (when S is not
                # a multiple of 64) needs SIMT zeroing.
                gm_val_src = tla.make_tensor(
                    gm_values_raw.ptr + vr0.to(tla.Int64) * tla.Int64(num_selected),
                    tla.make_layout(
                        tla.make_shape(val_rows, val_pad),
                        tla.make_stride(num_selected, 1),
                        origin_shape=tla.make_shape(vn, num_selected),
                    ),
                )
                ub_val_dst = tla.make_tensor(
                    ub_vals_ptr,
                    tla.make_layout(
                        tla.make_shape(val_rows, val_pad),
                        tla.make_stride(val_pad, 1),
                        origin_shape=tla.make_shape(vn, num_selected),
                    ),
                )
                tla.copy(ub_val_dst, gm_val_src)
                tla.pipe_barrier(tla.pipes.ALL)
                if tla.const_expr(num_selected < val_pad):
                    with tla.vec.func(
                        mode="simt", thread_block_dim=_PREPASS_SIMT_THREADS
                    ):
                        tid, _, _ = tla.arch.thread_idx()
                        for lr in tla.range(tid, vn, _PREPASS_SIMT_THREADS):
                            for vz in tla.range(num_selected, val_pad, one):
                                ub_vals[lr, vz] = 0.0
                    tla.pipe_barrier(tla.pipes.ALL)
                if tla.const_expr(copy_values):
                    gm_val_dst = tla.make_tensor(
                        gm_values_pad.ptr + vr0.to(tla.Int64) * tla.Int64(num_selected),
                        tla.make_layout(
                            tla.make_shape(val_rows, val_pad),
                            tla.make_stride(num_selected, 1),
                            origin_shape=tla.make_shape(vn, num_selected),
                        ),
                    )
                    ub_val_src = tla.make_tensor(
                        ub_vals_ptr,
                        tla.make_layout(
                            tla.make_shape(val_rows, val_pad),
                            tla.make_stride(val_pad, 1),
                            origin_shape=tla.make_shape(vn, num_selected),
                        ),
                    )
                    tla.copy(gm_val_dst, ub_val_src)
                    tla.pipe_barrier(tla.pipes.ALL)
            else:
                with tla.vec.func(mode="simt", thread_block_dim=_PREPASS_SIMT_THREADS):
                    tid, _, _ = tla.arch.thread_idx()
                    for lr in tla.range(tid, vn, _PREPASS_SIMT_THREADS):
                        for vs in tla.range(0, num_selected, one):
                            val = gm_values_raw[vr0 + lr, vs]
                            ub_vals[lr, vs] = val
                            if tla.const_expr(copy_values):
                                gm_values_pad[vr0 + lr, vs] = val
                        for vz in tla.range(num_selected, val_pad, one):
                            ub_vals[lr, vz] = 0.0
                tla.pipe_barrier(tla.pipes.ALL)

            if tla.const_expr(reduce):
                # Hardware reduce over 64 fp32 lanes; chunk sums are combined
                # lane-wise first, mirroring the NPU torch.sum(-1) order.
                with tla.vec.func(mode="simd"):
                    all_lanes = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                    for sr in tla.range(vn):
                        acc = tla.tile_view(
                            ub_vals,
                            tla.make_shape(1, 64),
                            tla.make_coord(sr, 0),
                        ).load()
                        for ci in range(1, val_chunks):
                            nxt = tla.tile_view(
                                ub_vals,
                                tla.make_shape(1, 64),
                                tla.make_coord(sr, ci),
                            ).load()
                            acc = tla.add(acc, nxt, mask=all_lanes)
                        red = acc.reduce(tla.ReductionOp.ADD, mask=all_lanes)
                        tla.tile_view(
                            ub_red,
                            tla.make_shape(1),
                            tla.make_coord(sr * 8),
                        ).store(red)
                tla.pipe_barrier(tla.pipes.ALL)

                with tla.vec.func(mode="simt", thread_block_dim=_PREPASS_SIMT_THREADS):
                    tid, _, _ = tla.arch.thread_idx()
                    for nr in tla.range(tid, vn, _PREPASS_SIMT_THREADS):
                        gm_neg_sum_pad[vr0 + nr, 0] = -ub_red[nr * 8]
                tla.pipe_barrier(tla.pipes.ALL)
    elif tla.const_expr(reduce):
        # No selected columns: the row sum is the empty sum, negated.
        with tla.vec.func(mode="simt", thread_block_dim=_PREPASS_SIMT_THREADS):
            tid, _, _ = tla.arch.thread_idx()
            gtid = gid * _PREPASS_SIMT_THREADS + tid
            gthreads = aiv_num * _PREPASS_SIMT_THREADS
            zero_f = tla.as_numeric(0.0)
            for zr in tla.range(gtid, m_raw, gthreads):
                gm_neg_sum_pad[zr, 0] = -zero_f
        tla.pipe_barrier(tla.pipes.ALL)

    # Every AIV has finished its share of the prologue.
    tla.pipe_barrier(tla.pipes.ALL)
    if tla.arch.sub_block_idx() == 0:
        tla.cross_core_set_flag(pad_arrive, tla.arch.MTE3, aiv_id=0)
    else:
        tla.cross_core_set_flag(pad_arrive, tla.arch.MTE3, aiv_id=1)


@tla.jit
def wait_prepass(pad_arrive, pad_ffts, pad_release) -> None:
    """Wait for the prepass on every core, then publish the release.

    Call at the start of ``with tla.cube():``.  ``pad_arrive`` is mode 4 (one
    token per AIV), ``pad_ffts`` is the mode-0 all-AIC barrier and
    ``pad_release`` is the mode-2 release/reset flag.
    """
    tla.pipe_barrier(tla.pipes.ALL)
    tla.cross_core_wait_flag(pad_arrive, tla.arch.SCALAR, aiv_id=0)
    tla.cross_core_wait_flag(pad_arrive, tla.arch.SCALAR, aiv_id=1)
    tla.cross_core_set_flag(pad_ffts, tla.arch.FIX)
    tla.cross_core_wait_flag(pad_ffts, tla.arch.SCALAR)
    # Release/reset: both AIVs of this core may proceed.
    tla.cross_core_set_flag(pad_release, tla.arch.SCALAR)


@tla.kernel
def matmul_row_exp_scatter_add_kernel(
    gm_x: tla.Tensor,  # [M_pad, H_pad] bf16, row-major; X
    gm_weight: tla.Tensor,  # logical [H_pad, V_pad] column-major; W^T
    gm_row_logsumexp: tla.Tensor,  # [M_pad, 1] fp32; row_logsumexp[m]
    gm_neg_row_sum: tla.Tensor,  # [M_pad, 1] fp32; -sum_s values[m, s]
    gm_local_col_ids: tla.Tensor,  # [M_pad, max(S, 1)] int32; local column ids
    gm_values: tla.Tensor,  # [M_pad, max(S, 1)] fp32; values[m, s]
    gm_out: tla.Tensor,  # [M_pad, V_pad] bf16, row-major; Y
    gm_x_raw: tla.Tensor,  # [M, H] bf16, row-major; unpadded X
    gm_weight_raw: tla.Tensor,  # [V, H] bf16, row-major; unpadded W
    gm_lse_raw: tla.Tensor,  # [M] fp32; unpadded row_logsumexp
    gm_col_ids_raw: tla.Tensor,  # [M, max(S, 1)] int32; global column ids
    gm_values_raw: tla.Tensor,  # [M, max(S, 1)] fp32; unpadded values
    inv_temperature: float,
    col_start: int,
    alias: tla.Constexpr[bool],
    num_selected: tla.Constexpr[int],
) -> None:
    zero, one = 0, 1
    tile_m, tile_n, k_step = TILE_M, TILE_N, K_STEP
    # The fixpipe splits the C tile over the two AIVs of the cube core, so each
    # AIV owns half of the rows.
    rows_per_aiv = tile_m // 2
    simd_lanes = 64  # fp32 lanes per SIMD fragment
    m = gm_x.origin_shape[0]  # padded M
    v = gm_out.origin_shape[1]  # padded V
    swizzle_m, swizzle_n, tile_k = _select_schedule(m, v)
    m_tiles = (m + tile_m - 1) // tile_m
    v_tiles = (v + tile_n - 1) // tile_n
    k_blocks = (gm_x.origin_shape[1] + tile_k - 1) // tile_k
    k_chunks = tile_k // k_step
    m_supers = (m_tiles + swizzle_m - 1) // swizzle_m
    v_supers = (v_tiles + swizzle_n - 1) // swizzle_n
    task_count = m_supers * v_supers * swizzle_m * swizzle_n
    in_dtype = gm_x.ptr.dtype

    # ------------------------------------------------------------------ sync
    # tile_ready is the only cube <-> vector hand-off: the cube half sets it
    # once UB holds a fresh fp32 C tile.  The vector half sets it back once
    # the output has been read out of UB, ping-ponging at tile granularity.
    tile_ready = tla.cross_flag("tile_ready")
    # V0 handshake: each AIV reports to its own AIC (mode 4), the AICs then
    # rendezvous on an all-core barrier (mode 0) and publish the mode-2
    # release before the matmul touches the padded buffers.
    pad_arrive = tla.cross_flag("v0_pad_arrive", mode=4)
    pad_ffts = tla.cross_flag("v0_pad_ffts", mode=0)
    pad_release = tla.cross_flag("v0_pad_release", mode=2)
    # Load/compute mutexes, one per staging slot.  Ids are fixed literals; they
    # are hardware resources, not handles.
    mutex_l1_a0 = tla.mutex("l1_a0", id=0)
    mutex_l1_b0 = tla.mutex("l1_b0", id=1)
    mutex_l1_a1 = tla.mutex("l1_a1", id=2)
    mutex_l1_b1 = tla.mutex("l1_b1", id=3)
    mutex_l0_a0 = tla.mutex("l0_a0", id=4)
    mutex_l0_b0 = tla.mutex("l0_b0", id=5)
    mutex_l0_a1 = tla.mutex("l0_a1", id=6)
    mutex_l0_b1 = tla.mutex("l0_b1", id=7)
    mutex_l0_c = tla.mutex("l0_c", id=8)
    # Protect packed output between the VECTOR producer and the MTE3 consumer.
    mutex_output = tla.mutex("output", id=10)

    # -------------------------------------------------------------- staging
    # L1 holds two ping-pong slots of the A and B operands, one K block each.
    l1_a0_ptr = tla.allocate(tile_m * tile_k, in_dtype, tla.AddressSpace.l1, 512)
    l1_a1_ptr = tla.allocate(tile_m * tile_k, in_dtype, tla.AddressSpace.l1, 512)
    l1_b0_ptr = tla.allocate(tile_k * tile_n, in_dtype, tla.AddressSpace.l1, 512)
    l1_b1_ptr = tla.allocate(tile_k * tile_n, in_dtype, tla.AddressSpace.l1, 512)
    # L0A/L0B hold one K_STEP fragment, ping-ponged inside a K block.
    l0_a0_ptr = tla.allocate(tile_m * k_step, in_dtype, tla.AddressSpace.l0a, 512)
    l0_a1_ptr = tla.allocate(tile_m * k_step, in_dtype, tla.AddressSpace.l0a, 512)
    l0_b0_ptr = tla.allocate(k_step * tile_n, in_dtype, tla.AddressSpace.l0b, 512)
    l0_b1_ptr = tla.allocate(k_step * tile_n, in_dtype, tla.AddressSpace.l0b, 512)
    # One fp32 accumulator; the fixpipe copies it out whole, so it cannot be
    # double buffered (L0C on this part is exactly one TILE_M x TILE_N tile).
    l0_c_ptr = tla.allocate(tile_m * tile_n, tla.Float32, tla.AddressSpace.l0c, 1024)
    # UB: one fp32 C half-tile, one packed bf16 output half-tile and row scalars.
    ub_c_ptr = tla.allocate(
        rows_per_aiv * tile_n, tla.Float32, tla.AddressSpace.ub, 256
    )
    ub_out_ptr = tla.allocate(
        rows_per_aiv * tile_n, tla.BFloat16, tla.AddressSpace.ub, 256
    )
    ub_row_logsumexp_ptr = tla.allocate(
        rows_per_aiv, tla.Float32, tla.AddressSpace.ub, 256
    )
    ub_neg_row_sum_ptr = tla.allocate(
        rows_per_aiv, tla.Float32, tla.AddressSpace.ub, 256
    )
    # ======================================================================= #
    # Cube half: matmul, then hand the tile over to the vector half.
    # ======================================================================= #
    with tla.cube():
        # Wait for the prepass on every core (mode 4 + mode 0), then publish
        # the mode-2 release.  Only then may the matmul read the pads.
        wait_prepass(pad_arrive, pad_ffts, pad_release)

        # Logical views used as layout templates; the fixpipe writes the AIV
        # half of ``ub_c_full`` at offset 0 of its own UB.
        ub_c_full = tla.make_tensor(
            ub_c_ptr,
            tla.make_layout(tla.make_shape(tile_m, tile_n), tla.make_stride(tile_n, 1)),
        )
        l0_c = tla.make_tensor_like(l0_c_ptr, ub_c_full, tla.arch.L0Clayout)

        for task in tla.range(tla.arch.block_idx(), task_count, tla.arch.block_num()):
            # Task -> (m tile, v tile) through the super-tile swizzle.
            super_tile = task // (swizzle_m * swizzle_n)
            slot = task % (swizzle_m * swizzle_n)
            m_tile = (super_tile // v_supers) * swizzle_m + slot // swizzle_n
            v_tile = (super_tile % v_supers) * swizzle_n + slot % swizzle_n
            if m_tile < m_tiles and v_tile < v_tiles:
                # Every tile's first block is loaded into slot 0.  Reset the
                # phase here; an odd block count leaves the previous tile at 1.
                l1_phase = tla.as_numeric(zero)
                # Stage the first K block before entering the pipeline.
                gm_a_head = tla.tile_view(
                    gm_x,
                    tla.make_shape(tile_m, tile_k),
                    tla.make_coord(m_tile, zero),
                )
                gm_b_head = tla.tile_view(
                    gm_weight,
                    tla.make_shape(tile_k, tile_n),
                    tla.make_coord(zero, v_tile),
                )
                l1_a_head = tla.make_tensor_like(l1_a0_ptr, gm_a_head, tla.arch.zN)
                l1_b_head = tla.make_tensor_like(l1_b0_ptr, gm_b_head, tla.arch.nZ)
                mutex_l1_a0.lock(pipe=tla.arch.MTE2)
                tla.copy(l1_a_head, gm_a_head)
                mutex_l1_a0.unlock(pipe=tla.arch.MTE2)
                mutex_l1_b0.lock(pipe=tla.arch.MTE2)
                tla.copy(l1_b_head, gm_b_head)
                mutex_l1_b0.unlock(pipe=tla.arch.MTE2)

                mutex_l0_c.lock(pipe=tla.arch.CUBE)

                for k_block in tla.range(zero, k_blocks, one):
                    if l1_phase == zero:
                        mutex_l1_a0.lock(pipe=tla.arch.MTE1)
                        mutex_l1_b0.lock(pipe=tla.arch.MTE1)
                    else:
                        mutex_l1_a1.lock(pipe=tla.arch.MTE1)
                        mutex_l1_b1.lock(pipe=tla.arch.MTE1)

                    # Prefetch the next K block into the other L1 slot.
                    if k_block + one < k_blocks:
                        gm_a_next = tla.tile_view(
                            gm_x,
                            tla.make_shape(tile_m, tile_k),
                            tla.make_coord(m_tile, k_block + one),
                        )
                        gm_b_next = tla.tile_view(
                            gm_weight,
                            tla.make_shape(tile_k, tile_n),
                            tla.make_coord(k_block + one, v_tile),
                        )
                        next_a_ptr = l1_a1_ptr if l1_phase == zero else l1_a0_ptr
                        next_b_ptr = l1_b1_ptr if l1_phase == zero else l1_b0_ptr
                        l1_a_next = tla.make_tensor_like(
                            next_a_ptr, gm_a_next, tla.arch.zN
                        )
                        l1_b_next = tla.make_tensor_like(
                            next_b_ptr, gm_b_next, tla.arch.nZ
                        )
                        if l1_phase == zero:
                            mutex_l1_a1.lock(pipe=tla.arch.MTE2)
                            mutex_l1_b1.lock(pipe=tla.arch.MTE2)
                        else:
                            mutex_l1_a0.lock(pipe=tla.arch.MTE2)
                            mutex_l1_b0.lock(pipe=tla.arch.MTE2)
                        tla.copy(l1_a_next, gm_a_next)
                        tla.copy(l1_b_next, gm_b_next)
                        if l1_phase == zero:
                            mutex_l1_a1.unlock(pipe=tla.arch.MTE2)
                            mutex_l1_b1.unlock(pipe=tla.arch.MTE2)
                        else:
                            mutex_l1_a0.unlock(pipe=tla.arch.MTE2)
                            mutex_l1_b0.unlock(pipe=tla.arch.MTE2)

                    cur_a_ptr = l1_a0_ptr if l1_phase == zero else l1_a1_ptr
                    cur_b_ptr = l1_b0_ptr if l1_phase == zero else l1_b1_ptr
                    gm_a_cur = tla.tile_view(
                        gm_x,
                        tla.make_shape(tile_m, tile_k),
                        tla.make_coord(m_tile, k_block),
                    )
                    gm_b_cur = tla.tile_view(
                        gm_weight,
                        tla.make_shape(tile_k, tile_n),
                        tla.make_coord(k_block, v_tile),
                    )
                    l1_a = tla.make_tensor_like(cur_a_ptr, gm_a_cur, tla.arch.zN)
                    l1_b = tla.make_tensor_like(cur_b_ptr, gm_b_cur, tla.arch.nZ)

                    # Prime the L0 fragments with the first K_STEP of the block.
                    l1_a_first = tla.tile_view(
                        l1_a,
                        tla.make_shape(tile_m, k_step),
                        tla.make_coord(zero, zero),
                    )
                    l1_b_first = tla.tile_view(
                        l1_b,
                        tla.make_shape(k_step, tile_n),
                        tla.make_coord(zero, zero),
                    )
                    l0_a_first = tla.make_tensor_like(
                        l0_a0_ptr, l1_a_first, tla.arch.zN
                    )
                    l0_b_first = tla.make_tensor_like(
                        l0_b0_ptr, l1_b_first, tla.arch.nZ
                    )
                    mutex_l0_a0.lock(pipe=tla.arch.MTE1)
                    mutex_l0_b0.lock(pipe=tla.arch.MTE1)
                    tla.copy(l0_a_first, l1_a_first)
                    tla.copy(l0_b_first, l1_b_first)
                    mutex_l0_a0.unlock(pipe=tla.arch.MTE1)
                    mutex_l0_b0.unlock(pipe=tla.arch.MTE1)

                    l0_phase = tla.as_numeric(zero)
                    for k_chunk in tla.range(zero, k_chunks, one):
                        if l0_phase == zero:
                            mutex_l0_a0.lock(pipe=tla.arch.CUBE)
                            mutex_l0_b0.lock(pipe=tla.arch.CUBE)
                        else:
                            mutex_l0_a1.lock(pipe=tla.arch.CUBE)
                            mutex_l0_b1.lock(pipe=tla.arch.CUBE)

                        # Load the next L0 fragment into the other slot.
                        l0_src_a = tla.tile_view(
                            l1_a,
                            tla.make_shape(tile_m, k_step),
                            tla.make_coord(zero, k_chunk + one),
                        )
                        l0_src_b = tla.tile_view(
                            l1_b,
                            tla.make_shape(k_step, tile_n),
                            tla.make_coord(k_chunk + one, zero),
                        )
                        next_l0_a_ptr = l0_a1_ptr if l0_phase == zero else l0_a0_ptr
                        next_l0_b_ptr = l0_b1_ptr if l0_phase == zero else l0_b0_ptr
                        l0_a_next = tla.make_tensor_like(
                            next_l0_a_ptr, l0_src_a, tla.arch.zN
                        )
                        l0_b_next = tla.make_tensor_like(
                            next_l0_b_ptr, l0_src_b, tla.arch.nZ
                        )
                        if k_chunk + one < k_chunks:
                            if l0_phase == zero:
                                mutex_l0_a1.lock(pipe=tla.arch.MTE1)
                                mutex_l0_b1.lock(pipe=tla.arch.MTE1)
                            else:
                                mutex_l0_a0.lock(pipe=tla.arch.MTE1)
                                mutex_l0_b0.lock(pipe=tla.arch.MTE1)
                            tla.copy(l0_a_next, l0_src_a)
                            tla.copy(l0_b_next, l0_src_b)
                            if l0_phase == zero:
                                mutex_l0_a1.unlock(pipe=tla.arch.MTE1)
                                mutex_l0_b1.unlock(pipe=tla.arch.MTE1)
                            else:
                                mutex_l0_a0.unlock(pipe=tla.arch.MTE1)
                                mutex_l0_b0.unlock(pipe=tla.arch.MTE1)

                        cur_l0_a_ptr = l0_a0_ptr if l0_phase == zero else l0_a1_ptr
                        cur_l0_b_ptr = l0_b0_ptr if l0_phase == zero else l0_b1_ptr
                        l1_a_cur = tla.tile_view(
                            l1_a,
                            tla.make_shape(tile_m, k_step),
                            tla.make_coord(zero, k_chunk),
                        )
                        l1_b_cur = tla.tile_view(
                            l1_b,
                            tla.make_shape(k_step, tile_n),
                            tla.make_coord(k_chunk, zero),
                        )
                        l0_a = tla.make_tensor_like(cur_l0_a_ptr, l1_a_cur, tla.arch.zN)
                        l0_b = tla.make_tensor_like(cur_l0_b_ptr, l1_b_cur, tla.arch.nZ)

                        # First fragment of the tile initialises the accumulator;
                        # the last one releases L0C to the fixpipe.
                        init_c = True if k_block == zero and k_chunk == zero else False
                        unit_flag = (
                            0b11
                            if (k_block + one >= k_blocks)
                            and (k_chunk + one >= k_chunks)
                            else 0b10
                        )
                        tla.mmad(l0_c, l0_a, l0_b, init_c=init_c, unit_flag=unit_flag)
                        if l0_phase == zero:
                            mutex_l0_a0.unlock(pipe=tla.arch.CUBE)
                            mutex_l0_b0.unlock(pipe=tla.arch.CUBE)
                        else:
                            mutex_l0_a1.unlock(pipe=tla.arch.CUBE)
                            mutex_l0_b1.unlock(pipe=tla.arch.CUBE)
                        l0_phase = one - l0_phase

                    if l1_phase == zero:
                        mutex_l1_a0.unlock(pipe=tla.arch.MTE1)
                        mutex_l1_b0.unlock(pipe=tla.arch.MTE1)
                    else:
                        mutex_l1_a1.unlock(pipe=tla.arch.MTE1)
                        mutex_l1_b1.unlock(pipe=tla.arch.MTE1)
                    l1_phase = one - l1_phase

                # Accumulator is complete: hand the fp32 tile to the vector half.
                # The fixpipe waits for the last MMAD (unit_flag bit 0) and splits
                # the tile over the two AIVs.
                mutex_l0_c.unlock(pipe=tla.arch.CUBE)
                mutex_l0_c.lock(pipe=tla.arch.FIX)
                tla.cross_core_wait_flag(tile_ready, tla.arch.FIX)
                tla.copy(
                    ub_c_full,
                    l0_c,
                    tla.params.CopyL0C2DstParams(
                        unit_flag=3,
                        l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M,
                    ),
                )
                tla.cross_core_set_flag(tile_ready, tla.arch.FIX)
                mutex_l0_c.unlock(pipe=tla.arch.FIX)

    # ======================================================================= #
    # Vector half: epilogue in place, then narrow and store the output.
    # ======================================================================= #
    with tla.vector():
        # V0 comes from the in-file reuse section (see pad_rowsum_prepass).
        gm_w_pad_rm = tla.make_tensor(
            gm_weight.ptr,
            tla.make_layout(
                tla.make_shape(gm_weight.origin_shape[1], gm_weight.origin_shape[0]),
                tla.make_stride(gm_weight.origin_shape[0], 1),
            ),
        )
        pad_rowsum_prepass(
            gm_x_raw,
            gm_weight_raw,
            gm_lse_raw,
            gm_x,
            gm_w_pad_rm,
            gm_row_logsumexp,
            gm_neg_row_sum,
            col_start,
            TILE_M,
            TILE_K,
            TILE_N,
            True,
            alias,
            pad_arrive,
            gm_col_ids_raw if num_selected > 0 else None,
            gm_values_raw if num_selected > 0 else None,
            gm_local_col_ids if num_selected > 0 else None,
            gm_values if num_selected > 0 else None,
        )

        aiv = tla.arch.sub_block_idx()
        ub_c = tla.make_tensor(
            ub_c_ptr,
            tla.make_layout(
                tla.make_shape(rows_per_aiv, tile_n), tla.make_stride(tile_n, 1)
            ),
        )
        ub_out = tla.make_tensor(
            ub_out_ptr,
            tla.make_layout(
                tla.make_shape(rows_per_aiv, tile_n), tla.make_stride(tile_n, 1)
            ),
        )
        out_cast = tla.params.CastParams(
            reg_slot=tla.params.RegSlot.ZERO,
            sat_mode=tla.params.SatMode.NOSAT,
            round_mode=tla.params.RoundMode.CAST_TRUNC,
        )
        pack_store = tla.params.NormalStoreParams(
            store_dist=tla.params.StoreDist.DIST_PACK_B32
        )
        ub_row_logsumexp = tla.make_tensor(
            ub_row_logsumexp_ptr,
            tla.make_layout(tla.make_shape(rows_per_aiv), tla.make_stride(1)),
        )
        ub_neg_row_sum = tla.make_tensor(
            ub_neg_row_sum_ptr,
            tla.make_layout(tla.make_shape(rows_per_aiv), tla.make_stride(1)),
        )
        # Per-row scalar loads are broadcast to every SIMD lane.
        broadcast_load = tla.params.NormalLoadParams(
            load_dist=tla.params.LoadDist.DIST_BRC_B32
        )
        # 2.2.0 has no CAST_RINT (only CEIL/FLOOR/ROUND/TRUNC), so both
        # ties-to-even bf16 roundings below are emulated on the f32 bit pattern:
        # (x + 0x7FFF + ((x >> 16) & 1)) & ~0xFFFF.

        tla.cross_core_set_flag(tile_ready, tla.arch.VECTOR)

        for task in tla.range(tla.arch.block_idx(), task_count, tla.arch.block_num()):
            super_tile = task // (swizzle_m * swizzle_n)
            slot = task % (swizzle_m * swizzle_n)
            m_tile = (super_tile // v_supers) * swizzle_m + slot // swizzle_n
            v_tile = (super_tile % v_supers) * swizzle_n + slot % swizzle_n
            if m_tile < m_tiles and v_tile < v_tiles:
                row_base = m_tile * tile_m + aiv * rows_per_aiv
                col_base = v_tile * tile_n
                tla.cross_core_wait_flag(tile_ready, tla.arch.VECTOR)

                # Step 1 - per-row lse and negative row sum.  A 4-byte pitch
                # is not a legal DMA tile, so use one scalar GM read per row.
                with tla.vec.func(mode="simt", thread_block_dim=rows_per_aiv):
                    row_thread, _, _ = tla.arch.thread_idx()
                    ub_row_logsumexp[row_thread] = gm_row_logsumexp[
                        row_base + row_thread, zero
                    ]
                    ub_neg_row_sum[row_thread] = gm_neg_row_sum[
                        row_base + row_thread, zero
                    ]
                tla.pipe_barrier(tla.pipes.ALL)

                # Step 2 - matmul result -> bf16 -> exp, then scale by
                # the negative row sum:
                #   R = exp(bf16_rne(C) * invT - lse) * (-sum_s values)
                # Preparation computes the negative row sum once, avoiding
                # a sign flip in the inner loop.
                with tla.vec.func(mode="simd"):
                    all_lanes = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                    lsb_one = tla.full(1, dtype=tla.Int32)
                    round_bias = tla.full(32767, dtype=tla.Int32)
                    bf16_keep = tla.full(-65536, dtype=tla.Int32)
                    for row in tla.range(rows_per_aiv):
                        neg_row_sum = tla.tile_view(
                            ub_neg_row_sum,
                            tla.make_shape(1),
                            tla.make_coord(row),
                        ).load(params=broadcast_load)
                        logsumexp_row = tla.tile_view(
                            ub_row_logsumexp,
                            tla.make_shape(1),
                            tla.make_coord(row),
                        ).load(params=broadcast_load)
                        for col_block in tla.range(zero, tile_n // simd_lanes, one):
                            c_frag = tla.tile_view(
                                ub_c,
                                tla.make_shape(1, simd_lanes),
                                tla.make_coord(row, col_block),
                            ).load()
                            # Round the matmul result to bf16 before the fp32
                            # epilogue (ties to even).
                            c_frag = c_frag.bitcast(tla.Int32)
                            lsb = tla.bitwise_and(
                                tla.shift_right(c_frag, 16),
                                lsb_one,
                                mask=all_lanes,
                            )
                            c_frag = tla.add(
                                tla.add(c_frag, round_bias, mask=all_lanes),
                                lsb,
                                mask=all_lanes,
                            )
                            c_frag = tla.bitwise_and(c_frag, bf16_keep, mask=all_lanes)
                            c_frag = c_frag.bitcast(tla.Float32)
                            c_frag = tla.mul(c_frag, inv_temperature, mask=all_lanes)
                            c_frag = tla.sub(c_frag, logsumexp_row, mask=all_lanes)
                            c_frag = tla.exp(c_frag, mask=all_lanes)
                            c_frag = tla.mul(c_frag, neg_row_sum, mask=all_lanes)
                            tla.tile_view(
                                ub_c,
                                tla.make_shape(1, simd_lanes),
                                tla.make_coord(row, col_block),
                            ).store(c_frag, mask=all_lanes)

                tla.pipe_barrier(tla.pipes.ALL)

                # Step 3 - selected-column scatter-add: one thread per row,
                # one read-modify-write per selected slot.  Repeated ids add
                # sequentially.  ``gm_local_col_ids`` holds shard-local column
                # ids; preparation maps ids outside this shard to -1.
                with tla.vec.func(mode="simt", thread_block_dim=rows_per_aiv):
                    row_thread, _, _ = tla.arch.thread_idx()
                    global_row = row_base + row_thread
                    for selected_idx in tla.range(zero, num_selected, one):
                        local_col = gm_local_col_ids[global_row, selected_idx]
                        if local_col >= zero:
                            if local_col >= col_base:
                                if local_col < col_base + tile_n:
                                    ub_c[row_thread, local_col - col_base] = (
                                        ub_c[row_thread, local_col - col_base]
                                        + gm_values[global_row, selected_idx]
                                    )

                tla.pipe_barrier(tla.pipes.ALL)

                # Step 4 - Y = bf16_rne(R * invT).  Round in fp32 first, so the
                # subsequent TRUNC cast is exact, including halfway inputs.
                # PACK_B32 compacts the low bf16 half of each register lane.
                mutex_output.lock(pipe=tla.arch.VECTOR)
                with tla.vec.func(mode="simd"):
                    out_lanes = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                    store_lanes = tla.create_mask(
                        pattern=tla.mask.ALL, dtype=tla.BFloat16
                    )
                    out_lsb_one = tla.full(1, dtype=tla.Int32)
                    out_round_bias = tla.full(32767, dtype=tla.Int32)
                    out_bf16_mask = tla.full(-65536, dtype=tla.Int32)
                    for out_round_row in tla.range(rows_per_aiv):
                        for out_col_block in tla.range(zero, tile_n // simd_lanes, one):
                            out_frag = tla.tile_view(
                                ub_c,
                                tla.make_shape(1, simd_lanes),
                                tla.make_coord(out_round_row, out_col_block),
                            ).load()
                            out_frag = tla.mul(
                                out_frag, inv_temperature, mask=out_lanes
                            )
                            out_frag = out_frag.bitcast(tla.Int32)
                            out_lsb = tla.bitwise_and(
                                tla.shift_right(out_frag, 16),
                                out_lsb_one,
                                mask=out_lanes,
                            )
                            out_frag = tla.add(
                                tla.add(out_frag, out_round_bias, mask=out_lanes),
                                out_lsb,
                                mask=out_lanes,
                            )
                            out_frag = tla.bitwise_and(
                                out_frag, out_bf16_mask, mask=out_lanes
                            )
                            out_frag = out_frag.bitcast(tla.Float32)
                            tla.tile_view(
                                ub_out,
                                tla.make_shape(1, simd_lanes),
                                tla.make_coord(out_round_row, out_col_block),
                            ).store(
                                out_frag.to(tla.BFloat16, out_cast, out_lanes),
                                pack_store,
                                mask=store_lanes,
                            )
                mutex_output.unlock(pipe=tla.arch.VECTOR)

                tla.pipe_barrier(tla.pipes.ALL)

                # Step 5 - packed bf16 output tile -> GM without conversion.
                # The output can exceed 2**31 elements; widen before multiplying.
                gm_out_tile = tla.make_tensor(
                    gm_out.ptr
                    + row_base.to(tla.Int64) * tla.Int64(v)
                    + col_base.to(tla.Int64),
                    tla.make_layout(
                        tla.make_shape(rows_per_aiv, tile_n), tla.make_stride(v, 1)
                    ),
                )
                mutex_output.lock(pipe=tla.arch.MTE3)
                tla.copy(gm_out_tile, ub_out)
                mutex_output.unlock(pipe=tla.arch.MTE3)
                # Hand the fp32 staging tile back; mutex_output separately
                # keeps the next pack from overwriting an in-flight DMA source.
                tla.pipe_barrier(tla.pipes.ALL)
                tla.cross_core_set_flag(tile_ready, tla.arch.VECTOR)

        tla.pipe_barrier(tla.pipes.ALL)


# ----------------------------------------------------------------------------------------------
# Host layer
# ----------------------------------------------------------------------------------------------
def _check_2d(name, t, dtype, rows=None, cols=None):
    if t.dim() != 2:
        raise ValueError(f"{name} must be 2-D, got shape {tuple(t.shape)}")
    if t.dtype != dtype:
        raise ValueError(f"{name} must be {dtype}, got {t.dtype}")
    if rows is not None and t.shape[0] != rows:
        raise ValueError(f"{name} must have {rows} rows, got {t.shape[0]}")
    if cols is not None and t.shape[1] != cols:
        raise ValueError(f"{name} must have {cols} columns, got {t.shape[1]}")


class _Prepared:
    __slots__ = (
        "compiled",
        "launch",
        "blocks",
        "m_pad",
        "h_pad",
        "v_pad",
        "m",
        "v",
        "x_pad",
        "weight_pad",
        "lse_pad",
        "neg_row_sum_pad",
        "col_ids_pad",
        "values_pad",
        "out_buf",
        "x_raw",
        "weight_raw",
        "lse_raw",
        "col_ids_raw",
        "values_raw",
        "col_start",
        "v0_mode",
    )


_COMPILE_CACHE = {}
_STAGE_CACHE = {}

# GM addresses consumed by MTE should start on a DMA block boundary.
_DMA_ALIGN_BYTES = 512


def _dma_aligned(t):
    """True when the tensor base address meets the DMA alignment."""
    return t.numel() == 0 or int(t.data_ptr()) % _DMA_ALIGN_BYTES == 0


def _stage(device, name, shape, dtype):
    """Cached staging buffer, zero-initialized on its first allocation."""
    key = (device, name, shape, dtype)
    buf = _STAGE_CACHE.get(key)
    if buf is None:
        buf = torch.zeros(shape, dtype=dtype, device=f"npu:{device}")
        _STAGE_CACHE[key] = buf
    return buf


def _compiled_entry(device, m_pad, h_pad, v_pad, m, h, v, alias, num_selected, sample):
    # The kernel is specialised on the padded *and* the raw shapes: the V0
    # tile loops walk M/H/V directly, and the V0 mode is selected from them.
    # Two raw shapes can share one padded shape, so both must be part of the
    # key.
    key = (device, m_pad, h_pad, v_pad, m, h, v, alias, num_selected)
    entry = _COMPILE_CACHE.get(key)
    if entry is None:
        (
            x_t,
            weight_t,
            lse_t,
            neg_row_sum_t,
            col_ids_t,
            values_t,
            out_t,
            x_raw_t,
            weight_raw_t,
            lse_raw_t,
            col_ids_raw_t,
            values_raw_t,
        ) = sample
        # Only shapes and dtypes are specialised; the scalars below are runtime
        # arguments, so any valid ones compile the same kernel.
        entry = tla.compile(
            matmul_row_exp_scatter_add_kernel,
            x_t,
            weight_t,
            lse_t,
            neg_row_sum_t,
            col_ids_t,
            values_t,
            out_t,
            x_raw_t,
            weight_raw_t,
            lse_raw_t,
            col_ids_raw_t,
            values_raw_t,
            1.0,
            0,
            alias,
            num_selected,
            options=_COMPILE_OPTS,
        )
        _COMPILE_CACHE[key] = entry
    return entry


def _to_tla(
    x_pad,
    weight_pad,
    lse_pad,
    neg_row_sum_pad,
    col_ids_pad,
    values_pad,
    out_buf,
    x_raw,
    weight_raw,
    lse_raw,
    col_ids_raw,
    values_raw,
):
    return (
        from_dlpack(x_pad, layout_tag=tla.arch.RowMajor),
        from_dlpack(weight_pad, layout_tag=tla.arch.ColumnMajor),
        from_dlpack(lse_pad.reshape(-1, 1), layout_tag=tla.arch.RowMajor),
        from_dlpack(neg_row_sum_pad.reshape(-1, 1), layout_tag=tla.arch.RowMajor),
        from_dlpack(col_ids_pad, layout_tag=tla.arch.RowMajor),
        from_dlpack(values_pad, layout_tag=tla.arch.RowMajor),
        from_dlpack(out_buf, layout_tag=tla.arch.RowMajor),
        from_dlpack(x_raw, layout_tag=tla.arch.RowMajor),
        from_dlpack(weight_raw, layout_tag=tla.arch.RowMajor),
        from_dlpack(lse_raw, layout_tag=tla.arch.RowMajor),
        from_dlpack(col_ids_raw, layout_tag=tla.arch.RowMajor),
        from_dlpack(values_raw, layout_tag=tla.arch.RowMajor),
    )


def prepare(
    x,
    weight,
    global_col_ids,
    row_logsumexp,
    values,
    temperature,
    shard_index=0,
    out=None,
):
    """Allocate staging buffers and compile the single fused kernel.

    Uses the same input contract as ``run``.  All padding, id localisation and
    the negative row sum are done on device by the kernel's V0 prologue; the
    host only allocates (once, zero-initialised) staging buffers, compiles and
    launches.  Returns an object whose ``launch()`` runs the device kernel
    only, so preparation cost can be excluded from timing.  Optional ``out``
    supplies a preallocated ``[M_pad, V_pad]`` bf16 buffer.

    Staging buffers are shared by device, role and padded shape.  A later
    preparation can overwrite inputs held by an earlier prepared object.

    When ``M``/``H``/``V`` are exact tile multiples, ``S <= 64`` and every
    input base address is DMA-aligned, all four large inputs are read in
    place; V0 then only localises ids and reduces the row sums.  Otherwise
    the per-dimension fast path applies.  Inputs whose base address is not
    DMA-aligned are rejected instead of being copied on the host.
    """
    m, h = x.shape
    v = weight.shape[0]
    num_selected = global_col_ids.shape[1]
    device = (
        x.device.index if x.device.index is not None else torch.npu.current_device()
    )

    col_start = int(shard_index) * v
    m_pad = _pad_to(m, TILE_M)
    h_pad = _pad_to(h, TILE_K)
    v_pad = _pad_to(v, TILE_N)

    # V0 reads the inputs through plain pointers, so they must be contiguous.
    for _name, _t in (
        ("x", x),
        ("weight", weight),
        ("global_col_ids", global_col_ids),
        ("row_logsumexp", row_logsumexp),
        ("values", values),
    ):
        if not _t.is_contiguous():
            raise ValueError(f"{_name} must be contiguous")

    # The runtime rejects zero-width RowMajor views.  Keep one dummy column
    # for S = 0; num_selected stays zero, so the kernel never reads it.
    selected_cols_pad = max(1, num_selected)
    if num_selected > 0:
        ids_raw = global_col_ids
        values_raw = values
    else:
        ids_raw = _stage(device, "col_ids_raw_dummy", (m, 1), torch.int32)
        values_raw = _stage(device, "values_raw_dummy", (m, 1), torch.float32)

    # The prepass reads inputs through plain pointers, so every base address
    # it may consume directly must meet the DMA alignment.
    if not (
        _dma_aligned(x)
        and _dma_aligned(weight)
        and _dma_aligned(global_col_ids)
        and _dma_aligned(row_logsumexp)
        and (num_selected == 0 or _dma_aligned(values))
    ):
        raise ValueError("input base addresses must be DMA-aligned")

    # Aligned short path: M/H/V are exact tile multiples, S is small and every
    # pointer is usable, so the four large inputs are aliased (no staging).
    alias = (
        m % TILE_M == 0 and h % TILE_K == 0 and v % TILE_N == 0 and num_selected <= 64
    )

    # Padded staging buffers are allocated once, zero-initialized and cached by
    # shape.  V0 overwrites only the valid region, so the column padding of
    # x_pad / w_pad stays zero without any per-call clearing.
    if alias:
        x_pad = x
        weight_pad = weight
        lse_pad = row_logsumexp
        values_pad = values_raw
    else:
        if m_pad == m and h_pad == h:
            x_pad = x
        else:
            x_pad = _stage(device, "x", (m_pad, h_pad), torch.bfloat16)
        if v_pad == v and h_pad == h:
            weight_pad = weight
        else:
            weight_pad = _stage(device, "w", (v_pad, h_pad), torch.bfloat16)
        if m_pad == m:
            lse_pad = row_logsumexp
        else:
            lse_pad = _stage(device, "lse", (m_pad,), torch.float32)
        if num_selected > 0 and m_pad == m:
            values_pad = values
        else:
            values_pad = _stage(
                device, "values", (m_pad, selected_cols_pad), torch.float32
            )
    col_ids_pad = _stage(device, "col_ids", (m_pad, selected_cols_pad), torch.int32)
    neg_row_sum_pad = _stage(device, "neg_row_sum", (m_pad,), torch.float32)
    if out is None:
        out_buf = torch.empty(
            (m_pad, v_pad), dtype=torch.bfloat16, device=f"npu:{device}"
        )
    else:
        if tuple(out.shape) != (m_pad, v_pad) or out.dtype != torch.bfloat16:
            raise ValueError(
                f"out must be bf16 [{m_pad}, {v_pad}], got {out.dtype} "
                f"{tuple(out.shape)}"
            )
        if out.device.type != "npu" or not out.is_contiguous():
            raise ValueError("out must be a contiguous NPU tensor")
        out_buf = out

    inv_temperature = 1.0 / float(temperature)
    sample = _to_tla(
        x_pad,
        weight_pad,
        lse_pad,
        neg_row_sum_pad,
        col_ids_pad,
        values_pad,
        out_buf,
        x,
        weight,
        row_logsumexp,
        ids_raw,
        values_raw,
    )
    entry = _compiled_entry(
        device, m_pad, h_pad, v_pad, m, h, v, alias, num_selected, sample
    )
    blocks = max(1, int(torch.npu.get_device_properties(device).cube_core_num))

    p = _Prepared()
    p.compiled = (entry, sample, inv_temperature, blocks)
    p.blocks = blocks
    p.m_pad, p.h_pad, p.v_pad, p.m, p.v = m_pad, h_pad, v_pad, m, v
    p.x_pad, p.weight_pad, p.lse_pad, p.neg_row_sum_pad = (
        x_pad,
        weight_pad,
        lse_pad,
        neg_row_sum_pad,
    )
    p.col_ids_pad, p.values_pad, p.out_buf = col_ids_pad, values_pad, out_buf
    p.x_raw, p.weight_raw, p.lse_raw = x, weight, row_logsumexp
    p.col_ids_raw, p.values_raw = ids_raw, values_raw
    p.col_start = col_start
    p.v0_mode = 1 if alias else 0

    def launch():
        entry, sample, inv_t, blocks_ = p.compiled
        entry(*sample, inv_t, p.col_start, block_num=blocks_)

    p.launch = launch
    return p


def run(
    x,
    weight,
    global_col_ids,
    row_logsumexp,
    values,
    temperature,
    shard_index=0,
    out=None,
):
    """Compute matmul, row-wise exp/scaling and selected-column scatter-add.

    Inputs are ``x [M, H]`` and ``weight [V, H]`` in bf16,
    ``global_col_ids [M, S]`` in int32, and ``row_logsumexp [M]`` and
    ``values [M, S]`` in fp32.  Columns in this shard cover
    ``[shard_index*V, (shard_index+1)*V)``.  Out-of-shard ids skip scatter-add;
    all values still contribute to their row sums.  Repeated ids accumulate.

    Returns ``Y [M, V] bf16`` as a possibly noncontiguous view.  Optional
    ``out`` is the full contiguous ``[M_pad, V_pad]`` bf16 output buffer;
    reusing one avoids a per-call allocation.
    """
    if x.device.type != "npu":
        raise ValueError("x must be on an NPU device")
    _check_2d("x", x, torch.bfloat16)
    _check_2d("weight", weight, torch.bfloat16, cols=x.shape[1])
    _check_2d("global_col_ids", global_col_ids, torch.int32, rows=x.shape[0])
    if row_logsumexp.dim() != 1 or row_logsumexp.dtype != torch.float32:
        raise ValueError("row_logsumexp must be 1-D float32")
    if row_logsumexp.shape[0] != x.shape[0]:
        raise ValueError("row_logsumexp must have M entries")
    _check_2d(
        "values", values, torch.float32, rows=x.shape[0], cols=global_col_ids.shape[1]
    )
    if not float(temperature) > 0:
        raise ValueError("temperature must be positive")
    if int(shard_index) < 0:
        raise ValueError("shard_index must be non-negative")

    m, v = x.shape[0], weight.shape[0]
    if m == 0 or v == 0:
        return torch.empty((m, v), dtype=torch.bfloat16, device=x.device)

    p = prepare(
        x,
        weight,
        global_col_ids,
        row_logsumexp,
        values,
        temperature,
        shard_index,
        out=out,
    )
    p.launch()
    torch.npu.synchronize()
    return p.out_buf[:m, :v]
