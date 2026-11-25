# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""GroupNorm forward (optionally fused with Swish), fp32 / fp16 / bf16.

``x`` is ``[N, C, S...]`` with ``C % G == 0``; a group's channels are
contiguous, so the tensor is viewed as ``[N * G, L]`` with ``L = (C / G) * S``
and every row is normalized on its own:

    mean = sum(x) / L
    var  = sum((x - mean)^2) / L
    y    = (x - mean) * rsqrt(var + eps) * w[c] + b[c]      (+ silu(y))

性能设计要点（对应 README 的「功能说明 - 高性能设计」）：

1. **按 channel 切 chunk，天然覆盖非对齐 `S`**：一个组内每个 channel 的
   ``S`` 个元素单独成段，UB 里每段按 ``S_PAD = ceil(S/64)*64`` 补齐。``S % 64
   == 0`` 时 ``S_PAD == S``，整行在 UB 里仍然是紧密的 ``L`` 个元素，行 DMA 退化
   为**一条连续描述符**（按 ``(CPG, S)`` 分块下发会变成 ``CPG`` 条小块 DMA，小
   channel 时吞吐掉一半以上）；``S % 64 != 0`` 时每段末尾是一条完整 64-lane
   向量、只放行 ``S % 64`` 个 lane。统计量只累加有效 lane，落盘侧 GM 保持稠密
   ``S``（DMA 只搬 ``S`` 个元素），因此 padding lane 既不参与统计也不出 UB，
   不需要任何 ``UnalignStoreParams``。
2. **每 buffer 一个 mutex 的三段流水**：``BUFS`` 个行缓冲各挂一个 mutex，同一
   时刻「MTE2 搬第 r+1 行 / VECTOR 算第 r 行 / MTE3 写第 r-1 行」三段重叠。缓冲
   复用由 mutex 而不是程序序决定，``BUFS`` 按 UB 预算取，保证至少两块在飞。
3. **一条平坦的动态 chunk 循环 + 有上限的 trace 期展开**：不需要 channel 下标
   的遍历（两个统计 pass 的对齐路径、无仿射的缩放 pass）都写成一条 trip 为
   ``L / 64`` 的平坦循环。按 ``CPG`` 个「每 channel 一条 ``S / 64`` 次的动态
   循环」写时，小 channel（如 ``CPG = 16, S = 128``，每 channel 只有 2 个
   chunk）会退化成几十条两迭代的短循环，每条都要付一次循环的固定开销，实测慢
   2.5 倍。必须按 channel 走的路径（``TAIL != 0`` 的统计、仿射缩放）把 channel
   内的 chunk 在 trace 期展开，但**展开深度上限 ``K_UNROLL = 16``**：一个基本块
   里塞 32 个 chunk 会让 256 B 向量寄存器溢出，``S = 2048`` 时实测慢 3 倍；更长
   的 channel 因此降级为 ``n_grp`` 次动态迭代 × ``k_unroll`` 个展开 chunk（外加
   余数），短 channel 则完全展开、无动态循环。
4. **每行只做两次归约**：lane 内先按 64-lane 向量累加，循环结束后只做一次
   ``reduce`` 得到 mean，再一次得到 var，归约的固定开销被 ``L/64`` 次累加摊薄。
5. **方差用「先去均值再平方和」**：精度优于单趟写法；不用
   ``E[x^2] - mean^2``（均值远大于方差时 fp32 相消会失真）。
6. **仿射参数按 channel 折算成 scale/shift**：pass 3 的每个 channel 只取一次
   ``w[c]`` / ``b[c]``（``DIST_BRC_B32`` 直接把 1 个 f32 元素广播成 64-lane
   寄存器，是向量 load 而不是标量 load，不会在 chunk 前插入依赖停顿），并预先
   折成 ``scale = rstd * w[c]``，于是每个 chunk 只剩
   ``(x - mean) * scale + b[c]``，比非仿射路径只多一条 FMA：既没有逐 chunk 的
   ``k // spad_chunks`` 整数除法，也没有逐 chunk 的表查找（这两项在长行上是仿射
   路径的主要开销）。先减均值再缩放（而不是折成 ``x * A + B``）是为了保住精度：
   行均值远大于方差时 ``x * A + B`` 会在大数相减处丢掉有效位。一个 chunk 不会
   跨 channel，所以该方案天然支持 ``S`` 不对齐。``(row % G) * (C/G)`` 由 host
   预计算成 ``gm_cb`` 标量表（避免 kernel 内的整数取模），每个块只多一次标量读。
7. **全程 FP32 计算**：fp16/bf16 的 UB 缓冲按原 dtype 存，``DIST_UNPACK_B16``
   载入即转 fp32，``DIST_PACK_B32`` 落盘时转回，UB 占用减半、向量计算不降精度。
8. **Swish 在归一再仿射之后原地融合**：``y * sigmoid(y)`` 复用同一份寄存器，
   不额外读写 UB，``SWISH == 0`` 时整段在编译期消除。
"""

import functools
import inspect
import threading
import weakref

import torch

import catlass.tla as tla
from catlass.tla.runtime import _Tensor

VL_ELE = 64  # fp32 vector length (256 B / 4 B)
UB_BYTES = 262144  # per-AIV UB capacity, used to size the row pipeline

# 16-bit -> fp32 unpack on load, fp32 -> 16-bit pack on store (values are
# already in range after 64-lane rounding, so no saturation).
_CAST_NONE = tla.params.CastParams()
_CAST_EVEN = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ZERO,
    sat_mode=tla.params.SatMode.NOSAT,
    round_mode=tla.params.RoundMode.CAST_ROUND,
)
_LOAD_UNPACK16 = tla.params.NormalLoadParams(
    load_dist=tla.params.LoadDist.DIST_UNPACK_B16
)
# One f32 scalar from UB broadcast into a full 64-lane register. Used for the
# per-channel affine weight/bias: it is a *vector* load (pipelined with the
# rest of the chunk loop) instead of a scalar load, which would otherwise put
# a dependent ~20-cycle stall in front of every chunk.
_LOAD_BRC32 = tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_BRC_B32)
_STORE_PACK32 = tla.params.NormalStoreParams(
    store_dist=tla.params.StoreDist.DIST_PACK_B32
)


class _RowView:
    """A row buffer's flat ``(1, CPG * S_PAD)`` view plus its UB element type.

    The element type rides on the buffer instead of living in a module global:
    the kernel body is traced once per compile, so a global would make a compile
    observe a dtype set by a concurrent ``run`` of another dtype. Reading it per
    chunk costs nothing -- these helpers only ever run at trace time.
    """

    __slots__ = ("tensor", "dtype")

    def __init__(self, tensor, dtype):
        self.tensor = tensor
        self.dtype = dtype


def _chunk(row, k):
    """64-lane view of chunk ``k`` of a row buffer.

    ``row.tensor`` is the flat ``(1, CPG * S_PAD)`` addressing view of the buffer,
    so chunk ``k`` is simply elements ``[64 * k, 64 * k + 64)``: the channel a
    chunk belongs to is ``k * 64 // S_PAD`` only when a per-channel scalar (the
    affine weight) is needed, and the passes that do not need it just walk ``k``.
    """
    return tla.tile_view(row.tensor, tla.make_shape(1, VL_ELE), tla.make_coord(0, k))


def _load(row, k, mask=None):
    """Load one 64-lane chunk as fp32 (unpack when the buffer is 16-bit)."""
    tile = _chunk(row, k)
    if row.dtype is tla.Float32:
        value = tile.load()
    else:
        value = tile.load(_LOAD_UNPACK16).to(tla.Float32, _CAST_NONE)
    if mask is not None:
        value = tla.add(value, tla.full(0.0, dtype=tla.Float32), mask=mask)
    return value


def _store(row, k, value):
    tile = _chunk(row, k)
    if row.dtype is tla.Float32:
        tile.store(value)
    else:
        tile.store(value.to(row.dtype, _CAST_EVEN), _STORE_PACK32)


def _reduce(total, mask, tail_total=None, tail_mask=None):
    """Broadcast the lane-wise sum to a scalar.

    A masked vector op zeroes the lanes it does not touch, so the tail partial
    sums live in their own accumulator: the head accumulator is reduced with the
    full mask first, then the (masked) tail reduction is added.
    """
    value = tla.full(total.reduce(tla.ReductionOp.ADD, mask=mask), dtype=tla.Float32)
    if tail_total is not None:
        value = tla.add(
            value,
            tla.full(
                tail_total.reduce(tla.ReductionOp.ADD, mask=tail_mask),
                dtype=tla.Float32,
            ),
            mask=mask,
        )
    return value


def _sigmoid(value, mask):
    one = tla.full(1.0, dtype=tla.Float32)
    return tla.div(
        one, tla.add(one, tla.exp(tla.neg(value), mask=mask), mask=mask), mask=mask
    )


def _accum_sum(acc, tensor, k, mask):
    """Add chunk ``k`` into the lane-wise sum accumulator."""
    return tla.add(acc, _load(tensor, k, mask), mask=mask)


def _accum_sq(acc, tensor, k, mean, mask):
    """Add ``(x - mean)^2`` of chunk ``k`` into the accumulator.

    ``d`` is local to this call so it does not have to stay live across the
    (possibly unrolled) loops that walk a channel.
    """
    d = tla.sub(_load(tensor, k, mask), mean, mask=mask)
    return tla.add(acc, tla.mul(d, d, mask=mask), mask=mask)


def _normalize(tensor, k, mean, rstd, mask, swish):
    """``(x - mean) * rstd`` (+ swish) in place, one 64-lane chunk (no affine)."""
    value = tla.mul(tla.sub(_load(tensor, k, mask), mean, mask=mask), rstd, mask=mask)
    if swish:
        value = tla.mul(value, _sigmoid(value, mask), mask=mask)
    _store(tensor, k, value)


def _affine_chunk(tensor, k, mean, scale, shift, mask, swish):
    """``(x - mean) * scale + shift`` (+ swish) in place, one 64-lane chunk.

    ``scale`` / ``shift`` are the per-channel ``rstd * w[c]`` and ``b[c]``
    broadcast registers, computed once per channel by the caller. Folding
    ``rstd`` into ``w[c]`` this way leaves the chunk body at a single extra FMA
    over the non-affine path: no per-chunk table lookup, no per-chunk
    ``k // spad_chunks`` divide, and the mean is still subtracted *before*
    scaling (keeping the accuracy of the direct form).
    """
    value = tla.mul(tla.sub(_load(tensor, k, mask), mean, mask=mask), scale, mask=mask)
    value = tla.add(value, shift, mask=mask)
    if swish:
        value = tla.mul(value, _sigmoid(value, mask), mask=mask)
    _store(tensor, k, value)


@tla.kernel
def group_norm_kernel(
    gm_x: tla.Tensor,  # [rows, L]
    gm_w: tla.Tensor,  # [C] fp32 (unused when HAS_AFFINE == 0)
    gm_b: tla.Tensor,  # [C] fp32 (unused when HAS_AFFINE == 0)
    gm_cb: tla.Tensor,  # [rows] int32 = (row % G) * (C / G)
    gm_y: tla.Tensor,  # [rows, L]
    CPG: tla.Constexpr[int],
    S: tla.Constexpr[int],
    S_PAD: tla.Constexpr[int],
    C: tla.Constexpr[int],
    C_PAD: tla.Constexpr[int],
    TAIL: tla.Constexpr[int],
    EPS: tla.Constexpr[float],
    HAS_AFFINE: tla.Constexpr[int],
    SWISH: tla.Constexpr[int],
    BUFS: tla.Constexpr[int],
    ELEM: tla.Constexpr[int],  # 0 fp32 / 1 fp16 / 2 bf16
) -> None:
    rows = gm_x.origin_shape[0]
    inv_l = 1.0 / float(CPG * S)
    n_full = S // VL_ELE
    spad_chunks = S_PAD // VL_ELE
    n_chunk = CPG * n_full  # dense 64-lane chunks in one row
    n_chunk_pad = CPG * spad_chunks  # same, counting the padded tail chunk
    swish_on = tla.const_expr(SWISH == 1)
    # UB element type, folded at trace time. Keeping it a kernel argument rather
    # than a module global makes the traced kernel a pure function of its
    # arguments; the two 16-bit branches are compile-time distinct.
    elem_dtype = tla.Float32
    if tla.const_expr(ELEM == 1):
        elem_dtype = tla.Float16
    if tla.const_expr(ELEM == 2):
        elem_dtype = tla.BFloat16

    # Unroll cap for the walks that must step channel by channel (``TAIL != 0``
    # in passes 1/2, and the affine pass). Such a walk is emitted as a dynamic
    # loop of ``CPG`` trips with the channel's chunks unrolled at trace time --
    # that is what avoids a dynamic loop per short channel. But the unroll does
    # not scale: from ~16 chunks per basic block on, the scheduler runs out of
    # 256 B vector registers, the block spills, and the affine pass was measured
    # 3x slower at ``S = 2048`` (32 chunks) than at ``S = 1024`` (16). So a long
    # channel is strip-mined: ``n_grp`` dynamic iterations of ``k_unroll``
    # unrolled chunks, plus ``n_rem`` leftover chunks; a short channel stays
    # fully unrolled with no dynamic loop at all.
    K_UNROLL = 16
    k_unroll = K_UNROLL if n_full > K_UNROLL else max(n_full, 1)
    n_grp = n_full // k_unroll
    n_rem = n_full % k_unroll

    # One mutex per row buffer: MTE2 unlocks it when the GM->UB copy retires,
    # VECTOR owns the buffer next, and MTE3 takes it over once the store lands.
    x_ptr = [None] * BUFS
    row_mutex = [None] * BUFS
    for bi in tla.range_constexpr(BUFS):
        x_ptr[bi] = tla.allocate(CPG * S_PAD, elem_dtype, tla.AddressSpace.ub, 256)
        row_mutex[bi] = tla.mutex(resource="gn_row", id=bi)

    # Two views of every row buffer:
    #
    # * ``ub_row`` is the DMA view. With ``TAIL == 0`` (``S % 64 == 0``) the row
    #   is one dense run of ``L`` elements on both sides, so it is described as
    #   ``(1, L)`` and the copy lowers to a SINGLE contiguous descriptor.
    #   Describing it as ``(CPG, S)`` instead makes ``tla.copy`` emit one
    #   descriptor per channel; for many small channels (``CPG = 16, S = 128``)
    #   those 256-byte transfers cap the kernel at ~0.5 TB/s, while the same
    #   bytes as one 4 KB transfer reach ~1.3 TB/s.
    #   With ``TAIL != 0`` UB keeps ``S_PAD`` per channel so the partial chunk is
    #   still a full 64-lane vector, and the copy is 2-D strided.
    # * ``row_map`` is a flat ``(1, CPG * S_PAD)`` view that the vector passes
    #   address channels/chunks through; it is the same memory either way.
    seg_layout = tla.make_layout(
        tla.make_shape(CPG, S),
        tla.make_stride(S_PAD, 1),
        origin_shape=tla.make_shape(CPG, S),
    )
    flat_layout = tla.make_layout(
        tla.make_shape(1, CPG * S),
        tla.make_stride(CPG * S, 1),
        origin_shape=tla.make_shape(1, CPG * S),
    )
    gm_layout = tla.make_layout(
        tla.make_shape(CPG, S),
        tla.make_stride(S, 1),
        origin_shape=tla.make_shape(CPG, S),
    )
    map_layout = tla.make_layout(
        tla.make_shape(1, CPG * S_PAD),
        tla.make_stride(CPG * S_PAD, 1),
        origin_shape=tla.make_shape(1, CPG * S_PAD),
    )
    ub_row = [None] * BUFS
    row_map = [None] * BUFS
    for bj in tla.range_constexpr(BUFS):
        if tla.const_expr(TAIL == 0):
            ub_row[bj] = tla.make_tensor(x_ptr[bj], flat_layout)
        else:
            ub_row[bj] = tla.make_tensor(x_ptr[bj], seg_layout)
        row_map[bj] = _RowView(tla.make_tensor(x_ptr[bj], map_layout), elem_dtype)

    # ``w`` / ``b`` are resident in UB for the whole block: both are only ``C``
    # floats, and pass 3 reads one scalar per channel per row. Reading those
    # scalars straight from GM costs a full memory round trip each (``2 * CPG``
    # per row, all dependent on the row loop); from UB they are register loads.
    # This is what keeps many-channel groups (large ``CPG``, small ``S``) from
    # being scalar-load bound.
    wb_mutex = tla.mutex(resource="gn_wb", id=31)
    # ``HAS_AFFINE == 0`` folds the whole affine path away, so the two tables are
    # only allocated -- and ``_pipeline`` only reserves UB for them -- when they
    # are actually copied in and read back.
    wb_views = None
    if tla.const_expr(HAS_AFFINE == 1):
        w_ptr = tla.allocate(C_PAD, tla.Float32, tla.AddressSpace.ub, 256)
        b_ptr = tla.allocate(C_PAD, tla.Float32, tla.AddressSpace.ub, 256)
        affine_layout = tla.make_layout(
            tla.make_shape(1, C),
            tla.make_stride(C, 1),
            origin_shape=tla.make_shape(1, C),
        )
        # Rank-1 view of the same buffers: the broadcast load needs a scalar tile
        # (``make_shape(1)``), which cannot be indexed through the 2-D copy view.
        affine_flat = tla.make_layout(
            tla.make_shape(C),
            tla.make_stride(1),
            origin_shape=tla.make_shape(C),
        )
        ub_w = tla.make_tensor(w_ptr, affine_layout)
        ub_b = tla.make_tensor(b_ptr, affine_layout)
        wb_views = [
            tla.make_tensor(w_ptr, affine_flat),
            tla.make_tensor(b_ptr, affine_flat),
        ]

    gid = tla.arch.block_idx()
    aiv_num = tla.arch.block_num()
    # Rows are dealt out *strided*: block ``gid`` takes ``gid, gid + aiv, ...``.
    # One row is one contiguous GM range (``L`` elements), so a row still costs
    # exactly one DMA in and one DMA out, and ``r < rows`` is the only guard the
    # tail needs. A contiguous per-block span would need a second bound
    # (``r < row_begin + rows_per_block``): with ``rows < aiv_num`` every block
    # owns a single row, and without that bound each block would also recompute
    # the next ``BUFS - 1`` rows, duplicating work up to ``BUFS`` times -- which
    # is exactly what makes tiny batches slow.
    row_stride = aiv_num
    rows_per_block = (rows + aiv_num - 1) // aiv_num

    with tla.vector():
        # One MTE2 fetch per block for the whole affine table; the matching
        # VECTOR lock is taken inside the row loop, so the first row waits for
        # the copy and every later one finds the mutex already open.
        if tla.const_expr(HAS_AFFINE == 1):
            wb_mutex.lock(pipe=tla.arch.MTE2)
            tla.copy(ub_w, gm_w)
            tla.copy(ub_b, gm_b)
            wb_mutex.unlock(pipe=tla.arch.MTE2)

        # Software pipeline: the three stages are issued back to back for every
        # buffer and the mutexes (not program order) decide when a stage runs, so
        # MTE2 for buffer bi+1 overlaps VECTOR for bi and MTE3 for bi-1.
        for grp in tla.range((rows_per_block + BUFS - 1) // BUFS):
            for bi in tla.range_constexpr(BUFS):
                r = gid + (grp * BUFS + bi) * row_stride
                if r < rows:
                    off = r.to(tla.Int64) * tla.Int64(CPG * S)
                    if tla.const_expr(TAIL == 0):
                        gm_row_x = tla.make_tensor(gm_x.ptr + off, flat_layout)
                        gm_row_y = tla.make_tensor(gm_y.ptr + off, flat_layout)
                    else:
                        gm_row_x = tla.make_tensor(gm_x.ptr + off, gm_layout)
                        gm_row_y = tla.make_tensor(gm_y.ptr + off, gm_layout)

                    row_mutex[bi].lock(pipe=tla.arch.MTE2)
                    tla.copy(ub_row[bi], gm_row_x)
                    row_mutex[bi].unlock(pipe=tla.arch.MTE2)

                    row_mutex[bi].lock(pipe=tla.arch.VECTOR)
                    if tla.const_expr(HAS_AFFINE == 1):
                        wb_mutex.lock(pipe=tla.arch.VECTOR)
                    with tla.vec.func(mode="simd"):
                        mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                        tail_mask = None
                        if tla.const_expr(TAIL != 0):
                            tail_mask, _ = tla.update_mask(TAIL, dtype=tla.Float32)

                        # (1) mean: one lane-wise accumulator, then a single
                        # reduce.
                        #
                        # ``TAIL == 0``: the row is a dense run of ``n_chunk``
                        # chunks, so one flat dynamic loop walks it. This is why
                        # the row is *not* walked as ``CPG`` nested loops of
                        # ``S / 64``: with many short channels (``CPG = 16,
                        # S = 128`` -> 2 chunks per channel) the nested form
                        # compiles to dozens of two-iteration dynamic loops and
                        # loses ~2.5x, while the flat loop keeps one long,
                        # pipelinable trip count.
                        #
                        # ``TAIL != 0``: every channel is followed by its padded
                        # partial chunk, so the head chunks are *not* contiguous
                        # in the flat UB view -- the row must be walked channel
                        # by channel (dynamic loop of ``CPG`` trips, ``n_full``
                        # head chunks unrolled), and each channel's partial
                        # chunk goes to its own ``tail_mask`` accumulator (see
                        # ``_reduce`` for why the tail needs a separate one).
                        acc = tla.full(0.0, dtype=tla.Float32)
                        tacc = None
                        if tla.const_expr(TAIL == 0):
                            for k1 in tla.range(n_chunk):
                                acc = _accum_sum(acc, row_map[bi], k1, mask)
                        else:
                            tacc = tla.full(0.0, dtype=tla.Float32)
                            for cc1 in tla.range(CPG):
                                if tla.const_expr(n_full <= K_UNROLL):
                                    for j1 in tla.range_constexpr(n_full):
                                        acc = _accum_sum(
                                            acc,
                                            row_map[bi],
                                            cc1 * spad_chunks + j1,
                                            mask,
                                        )
                                else:
                                    for g1 in tla.range(n_grp):
                                        for jj1 in tla.range_constexpr(k_unroll):
                                            acc = _accum_sum(
                                                acc,
                                                row_map[bi],
                                                cc1 * spad_chunks + g1 * k_unroll + jj1,
                                                mask,
                                            )
                                    for rr1 in tla.range_constexpr(n_rem):
                                        acc = _accum_sum(
                                            acc,
                                            row_map[bi],
                                            cc1 * spad_chunks + n_grp * k_unroll + rr1,
                                            mask,
                                        )
                                tacc = _accum_sum(
                                    tacc,
                                    row_map[bi],
                                    cc1 * spad_chunks + n_full,
                                    tail_mask,
                                )
                        mean = tla.mul(
                            _reduce(acc, mask, tacc, tail_mask), inv_l, mask=mask
                        )

                        # (2) var: sum of centred squares, same walk as pass 1
                        # (flat dense loop, or channel-wise when the row is
                        # padded). This second read of the row is what keeps fp32
                        # accuracy: E[x^2] - mean^2 in a single pass cancels
                        # catastrophically once the mean is much larger than the
                        # variance.
                        sq = tla.full(0.0, dtype=tla.Float32)
                        tsq = None
                        if tla.const_expr(TAIL == 0):
                            for k2 in tla.range(n_chunk):
                                sq = _accum_sq(sq, row_map[bi], k2, mean, mask)
                        else:
                            tsq = tla.full(0.0, dtype=tla.Float32)
                            for cc2 in tla.range(CPG):
                                if tla.const_expr(n_full <= K_UNROLL):
                                    for j2 in tla.range_constexpr(n_full):
                                        sq = _accum_sq(
                                            sq,
                                            row_map[bi],
                                            cc2 * spad_chunks + j2,
                                            mean,
                                            mask,
                                        )
                                else:
                                    for g2 in tla.range(n_grp):
                                        for jj2 in tla.range_constexpr(k_unroll):
                                            sq = _accum_sq(
                                                sq,
                                                row_map[bi],
                                                cc2 * spad_chunks + g2 * k_unroll + jj2,
                                                mean,
                                                mask,
                                            )
                                    for rr2 in tla.range_constexpr(n_rem):
                                        sq = _accum_sq(
                                            sq,
                                            row_map[bi],
                                            cc2 * spad_chunks + n_grp * k_unroll + rr2,
                                            mean,
                                            mask,
                                        )
                                tsq = _accum_sq(
                                    tsq,
                                    row_map[bi],
                                    cc2 * spad_chunks + n_full,
                                    mean,
                                    tail_mask,
                                )
                        var = tla.mul(
                            _reduce(sq, mask, tsq, tail_mask), inv_l, mask=mask
                        )
                        rstd = tla.div(
                            tla.full(1.0, dtype=tla.Float32),
                            tla.sqrt(tla.add(var, EPS, mask=mask), mask=mask),
                            mask=mask,
                        )

                        # (3) normalise + optional affine (+ swish) in place.
                        #
                        # Without affine the row is one flat chunk loop whose
                        # body is load / sub / mul / store, the same shape as
                        # passes 1 and 2.
                        #
                        # With affine the row is walked channel by channel: a
                        # dynamic loop of ``CPG`` trips, with the channel's chunks
                        # unrolled at trace time (strip-mined with ``k_unroll``
                        # when the channel is long, see ``K_UNROLL``). Per channel
                        # the two affine scalars are fetched *once* by broadcast
                        # load (``_LOAD_BRC32``) and folded into a single
                        # ``scale = rstd * w[c]`` register, so a chunk costs one
                        # extra FMA over the non-affine path and there is no
                        # per-chunk ``k // spad_chunks`` divide and no per-chunk
                        # table lookup -- both of those dominated the pass once
                        # the rows got long. Padding lanes of a partial channel
                        # are only ever written inside UB (the DMA out moves
                        # ``S`` elements per channel), so head chunks need no
                        # mask and the trailing partial one uses ``tail_mask``.
                        if tla.const_expr(HAS_AFFINE == 1):
                            ch_base = gm_cb[r]
                            for cc3 in tla.range(CPG):
                                wv = tla.tile_view(
                                    wb_views[0],
                                    tla.make_shape(1),
                                    tla.make_coord(ch_base + cc3),
                                ).load(_LOAD_BRC32)
                                bv = tla.tile_view(
                                    wb_views[1],
                                    tla.make_shape(1),
                                    tla.make_coord(ch_base + cc3),
                                ).load(_LOAD_BRC32)
                                scale = tla.mul(rstd, wv, mask=mask)
                                if tla.const_expr(n_full <= K_UNROLL):
                                    for j3 in tla.range_constexpr(n_full):
                                        _affine_chunk(
                                            row_map[bi],
                                            cc3 * spad_chunks + j3,
                                            mean,
                                            scale,
                                            bv,
                                            mask,
                                            swish_on,
                                        )
                                else:
                                    for g3 in tla.range(n_grp):
                                        for jj3 in tla.range_constexpr(k_unroll):
                                            _affine_chunk(
                                                row_map[bi],
                                                cc3 * spad_chunks + g3 * k_unroll + jj3,
                                                mean,
                                                scale,
                                                bv,
                                                mask,
                                                swish_on,
                                            )
                                    for rr3 in tla.range_constexpr(n_rem):
                                        _affine_chunk(
                                            row_map[bi],
                                            cc3 * spad_chunks + n_grp * k_unroll + rr3,
                                            mean,
                                            scale,
                                            bv,
                                            mask,
                                            swish_on,
                                        )
                                if tla.const_expr(TAIL != 0):
                                    _affine_chunk(
                                        row_map[bi],
                                        cc3 * spad_chunks + n_full,
                                        mean,
                                        scale,
                                        bv,
                                        tail_mask,
                                        swish_on,
                                    )
                        else:
                            for k3 in tla.range(n_chunk_pad):
                                _normalize(row_map[bi], k3, mean, rstd, mask, swish_on)
                    if tla.const_expr(HAS_AFFINE == 1):
                        wb_mutex.unlock(pipe=tla.arch.VECTOR)
                    row_mutex[bi].unlock(pipe=tla.arch.VECTOR)

                    row_mutex[bi].lock(pipe=tla.arch.MTE3)
                    tla.copy(gm_row_y, ub_row[bi])
                    row_mutex[bi].unlock(pipe=tla.arch.MTE3)

        tla.pipe_barrier(tla.pipes.ALL)


# ======================================================================================
# HOST side: shape checks, compile and launch.
# ======================================================================================
# 性能设计要点：小 shape 的耗时由 host 侧启动开销决定（from_dlpack / stream 解析 /
# payload 打包各占几微秒到几十微秒），所以这里做三级缓存：
#
#   _COMPILED  按 (rows, CPG, S, eps, 仿射, swish) 缓存编译产物与 BUFS；
#   _BINDINGS  按 (data_ptr, rows, cols, dtype) 直接构造 host 张量，绕开 from_dlpack；
#   _FAST      按 (张量地址, 形状参数) 缓存「绑定张量 + 预打包 payload + 启动器」。
#
# 设备索引、stream、vector core 数同样缓存；切换 stream 后需要调用
# reset_stream_cache()。

_COMPILED = {}
_VECTOR_CORES = {}
_BINDINGS = {}
_DUMMY = {}
_CBASE = {}
_FAST = {}
_MAX_BINDINGS = 4096
_MAX_FAST = 256

_INPUT_DTYPE = {
    torch.float32: (0, tla.Float32),
    torch.float16: (1, tla.Float16),
    torch.bfloat16: (2, tla.BFloat16),
}


def _vector_cores(index):
    cores = _VECTOR_CORES.get(index)
    if cores is None:
        cores = int(torch.npu.get_device_properties(index).vector_core_num)
        _VECTOR_CORES[index] = cores
    return cores


def _select_dtype(dtype):
    """``(ELEM constexpr code, TLA element dtype)`` for an input dtype."""
    entry = _INPUT_DTYPE.get(dtype)
    if entry is None:
        raise ValueError("dtype must be float32/float16/bfloat16, got {}".format(dtype))
    return entry


def _bind(buf, rows, cols, elem):
    """Bind a contiguous device buffer as a TLA host tensor, straight from its address.

    ``cols is None`` binds the buffer as a flat rank-1 run (used for the flat
    ``w`` / ``b`` / channel-base vectors, which the kernel indexes with a single
    scalar subscript).

    ``from_dlpack`` consumes a DLPack capsule that *keeps the producer tensor
    alive* (``_dlpack_source``), so a bound output buffer can never be released
    and the caching allocator can never hand the same address back -- every call
    would then miss the launch-plan cache and re-bind, which at these sizes costs
    more than the kernel itself. Building ``_Tensor`` from ``data_ptr`` makes it
    an external binding with no ownership, exactly like the fp32 path, so the
    plan is reused across calls. The caller keeps ``buf`` alive across the launch
    and the binding is memoized by ``(address, rows, cols, dtype)``.
    """
    sig = (buf.data_ptr(), rows, cols, elem)
    view = _BINDINGS.get(sig)
    if view is not None:
        return view
    if cols is None:
        view = _Tensor(
            (rows,),
            elem,
            origin_shape=(rows,),
            coord=(0,),
            stride=(1,),
            layout_tag=tla.arch.RowMajor,
            data_ptr=buf.data_ptr(),
        )
    else:
        view = _Tensor(
            (rows, cols),
            elem,
            origin_shape=(rows, cols),
            coord=(0, 0),
            stride=(cols, 1),
            layout_tag=tla.arch.RowMajor,
            data_ptr=buf.data_ptr(),
        )
    if len(_BINDINGS) < _MAX_BINDINGS:
        _BINDINGS[sig] = view
    return view


def _dummy(device):
    key = device.index
    buf = _DUMMY.get(key)
    if buf is None:
        buf = torch.ones((1, 1), dtype=torch.float32, device=device)
        _DUMMY[key] = buf
    return buf


# Casting a non-fp32 ``weight`` to fp32 launches an extra device kernel, i.e. one
# extra device task on *every* launch; at these sizes two casts cost more than the
# GroupNorm itself. The fp32 copy is therefore cached, keyed by
# ``(address, _version, C)`` and validated against the *identity* of the source
# tensor: an entry is reused only when its weakref still resolves to the same
# object. A freed source drops the weakref, so a recycled ``data_ptr`` can never
# alias a live entry, and a regular in-place update bumps ``_version``. Writes
# that bypass the version counter (``weight.data.copy_()``, a raw kernel writing
# the buffer) are *not* detected -- the caller must pass a fresh tensor then.
# fp32 contiguous weights skip the copy entirely.
_AFFINE = {}
_MAX_AFFINE = 64


def _affine_fp32(t, n):
    if t.dtype is torch.float32 and t.is_contiguous():
        return t
    key = (t.data_ptr(), int(t._version), n)
    hit = _AFFINE.get(key)
    if hit is not None and hit[0]() is t:
        return hit[1]
    copy = t.detach().to(torch.float32).contiguous()
    if len(_AFFINE) >= _MAX_AFFINE:
        _AFFINE.clear()
    _AFFINE[key] = (weakref.ref(t), copy)
    return copy


def _channel_base(rows, groups, cpg, device):
    """``(row % G) * (C / G)`` table; shape-only, so it is cached per device."""
    key = (rows, groups, cpg, device.index)
    table = _CBASE.get(key)
    if table is None:
        table = ((torch.arange(rows) % groups) * cpg).to(
            dtype=torch.int32, device=device
        )
        if len(_CBASE) >= 64:
            _CBASE.clear()
        _CBASE[key] = table
    return table


# Resolving the current stream costs ~11 us per launch and packing
# the launch payload another ~10 us; both are cached here. A launch issued under
# a different stream context keeps the stream cached on the first
# call, so call ``reset_stream_cache()`` after switching streams -- it drops both
# the resolved streams and the launch plans that captured them.
_STREAMS = {}
_LAUNCH_ACCEPTS_PRINT = None


def reset_stream_cache():
    """Forget the cached streams and the launch plans bound to them."""
    _STREAMS.clear()
    _FAST.clear()


def _launch_stream(index):
    key = (threading.get_ident(), index)
    stream = _STREAMS.get(key)
    if stream is None:
        stream = int(torch.npu.current_stream(index).npu_stream)
        _STREAMS[key] = stream
    return stream


def _resolve_launcher(artifact):
    try:
        from catlass import execution as execution_mod
        from catlass.base_dsl.jit_executor import _resolved_dyn_ubuf_bytes

        executor = artifact._get_executor()
        module = executor.jit_module
        if getattr(module, "uses_scalar_print", False) or getattr(
            module, "uses_tensor_print", False
        ):
            return None
        return (
            execution_mod,
            module.execution_args,
            _resolved_dyn_ubuf_bytes(module, 0),
            executor.function_handle,
            module,
        )
    except Exception:
        return None


def _make_launcher(artifact, tensors, block_num, stream):
    """Return a zero-argument launcher that reuses the packed payload.

    ``None`` means the runtime plumbing was not reusable and the caller has to
    fall back to the plain artifact call.
    """
    plumbing = _resolve_launcher(artifact)
    if plumbing is None:
        return None
    execution_mod, execution_args, dyn_ub, function_handle, module = plumbing
    try:
        payload = execution_args.generate_launch_payload(tensors)
    except Exception:
        return None
    if payload is None:
        return None

    global _LAUNCH_ACCEPTS_PRINT
    if _LAUNCH_ACCEPTS_PRINT is None:
        parameters = inspect.signature(execution_mod.launch_kernel).parameters
        _LAUNCH_ACCEPTS_PRINT = "uses_scalar_print" in parameters
    kwargs = {
        "function_handle": function_handle,
        "stream": stream,
        "block_num": block_num,
        "payload": payload,
        "dyn_ubuf_bytes": dyn_ub,
    }
    if _LAUNCH_ACCEPTS_PRINT:
        kwargs["uses_scalar_print"] = getattr(module, "uses_scalar_print", False)
    return functools.partial(execution_mod.launch_kernel, **kwargs)


def _pipeline(cpg, s_pad, elem_bytes, c_pad, has_affine):
    """Pick the number of in-flight row buffers from the UB budget.

    ``BUFS`` row buffers of ``CPG * S_PAD`` elements must coexist with the two
    resident affine tables (``2 * C_PAD`` fp32 elements, only allocated when
    there is affine). Two buffers are the minimum for any load/compute/store
    overlap and the cap of 6 keeps the prologue short. Wide rows fall back to a
    single buffer.
    """
    row_bytes = cpg * s_pad * elem_bytes
    slack = 8192 + (2 * c_pad * 4 if has_affine else 0)
    bufs = (UB_BYTES - slack) // row_bytes if row_bytes else 1
    return max(1, min(6, int(bufs)))


def run(x, num_groups, weight=None, bias=None, eps=1e-5, swish=False):
    """GroupNorm over ``x`` ``[N, C, S...]``; ``swish=True`` fuses ``silu``."""
    device = x.device
    if x.dim() < 3:
        raise ValueError("x must be at least 3-D [N, C, S...]")
    if type(num_groups) is not int:
        raise ValueError("num_groups must be an int")
    n, c = x.shape[0], x.shape[1]
    if not 1 <= num_groups <= c:
        raise ValueError("num_groups must be in [1, C], got {}".format(num_groups))
    if c % num_groups != 0:
        raise ValueError("C must be divisible by num_groups")
    spatial = 1
    for d in x.shape[2:]:
        spatial *= d
    cpg = c // num_groups
    row_len = cpg * spatial
    elem_code, elem = _select_dtype(x.dtype)

    rows = n * num_groups
    # The kernel indexes a row as one dense run of ``L`` elements, so a
    # contiguous ``x`` is bound straight from its address; only a non-contiguous
    # input pays for one copy. ``out`` is allocated with the *input* shape and
    # returned as is: no reshape/view is needed on either side of the launch.
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty(x.shape, dtype=x.dtype, device=device)

    if weight is None:
        # A lone ``bias`` is rejected instead of being silently ignored: the
        # kernel has an all-or-nothing affine path, so dropping it would return a
        # plausible but wrong tensor.
        if bias is not None:
            raise ValueError("bias requires weight; pass both or neither")
        has_affine = 0
        w = _dummy(device)
        b = _dummy(device)
        cb = _channel_base(1, 1, 0, device)
    else:
        has_affine = 1
        if weight.numel() != c or bias is None or bias.numel() != c:
            raise ValueError(
                "weight/bias must have C elements, and bias is required with weight"
            )
        w = _affine_fp32(weight, c)
        b = _affine_fp32(bias, c)
        cb = _channel_base(rows, num_groups, cpg, device)

    s_pad = (spatial + VL_ELE - 1) // VL_ELE * VL_ELE
    c_pad = (c + VL_ELE - 1) // VL_ELE * VL_ELE
    tail = spatial - (spatial // VL_ELE) * VL_ELE

    # One dictionary lookup for the whole hot path: the key pins the buffers
    # (which also keeps temporaries alive) plus every compile-time parameter.
    # ``C`` is listed explicitly even though ``num_groups * cpg`` already implies
    # it, so the key matches the compile tuple and the README one for one. The
    # device index is part of the key because addresses are only unique per
    # device.
    key = (
        x.data_ptr(),
        w.data_ptr(),
        b.data_ptr(),
        cb.data_ptr(),
        out.data_ptr(),
        device.index,
        rows,
        c,
        num_groups,
        cpg,
        spatial,
        has_affine,
        int(swish),
        float(eps),
        x.dtype,
    )
    plan = _FAST.get(key)
    if plan is not None:
        launch = plan[0]
        if launch is not None:
            launch()
        else:
            plan[1](*plan[2], block_num=plan[3], stream=plan[4])
        return out

    # Building the plan is the only place that turns these tensors into raw GM
    # addresses, so the device checks live here rather than on every call: a
    # ``_FAST`` hit replays a plan that keeps the very tensors its key was
    # computed from alive, hence a hit cannot carry a tensor of another device.
    if device.type != "npu":
        raise ValueError("x must be an Ascend (npu) tensor, got {}".format(device))
    if has_affine and (weight.device != device or bias.device != device):
        raise ValueError("weight/bias must be on the same device as x")

    # ``C`` must be part of the key: it is folded into the kernel as the ``C`` /
    # ``C_PAD`` constants (affine table size and copy length) and into ``BUFS``
    # through ``c_pad``. E.g. ``(N=2, C=100, G=1)`` and ``(N=1, C=200, G=2)``
    # share ``rows = 2`` and ``cpg = 100`` but need different affine buffers.
    meta = (
        rows,
        cpg,
        c,
        spatial,
        has_affine,
        int(swish),
        float(eps),
        str(x.dtype),
    )
    entry = _COMPILED.get(meta)
    if entry is None:
        bufs = _pipeline(cpg, s_pad, x.element_size(), c_pad, has_affine)
        artifact = tla.compile(
            group_norm_kernel,
            _bind(x, rows, row_len, elem),
            _bind(w, 1, c if has_affine else 1, tla.Float32),
            _bind(b, 1, c if has_affine else 1, tla.Float32),
            _bind(cb, rows if has_affine else 1, None, tla.Int32),
            _bind(out, rows, row_len, elem),
            cpg,
            spatial,
            s_pad,
            c,
            c_pad,
            tail,
            float(eps),
            has_affine,
            int(swish),
            bufs,
            elem_code,
            options="--npu-arch 3510",
        )
        entry = (artifact, bufs)
        _COMPILED[meta] = entry
    artifact = entry[0]

    device_index = device.index
    cores = _vector_cores(device_index)
    stream = _launch_stream(device_index)
    tensors = (
        _bind(x, rows, row_len, elem),
        _bind(w, c if has_affine else 1, None, tla.Float32),
        _bind(b, c if has_affine else 1, None, tla.Float32),
        _bind(cb, rows if has_affine else 1, None, tla.Int32),
        _bind(out, rows, row_len, elem),
    )
    plan = (
        _make_launcher(artifact, tensors, cores, stream),
        artifact,
        tensors,
        cores,
        stream,
    )
    if len(_FAST) >= _MAX_FAST:
        _FAST.clear()
    _FAST[key] = plan

    launch = plan[0]
    if launch is not None:
        launch()
    else:
        artifact(*tensors, block_num=cores, stream=stream)
    return out


if __name__ == "__main__":
    import torch_npu
    import torch.nn.functional as F

    torch.npu.set_device(0)
    torch.manual_seed(0)

    tol = {torch.float32: 2e-5, torch.float16: 3e-3, torch.bfloat16: 2e-2}
    for dt in (torch.float32, torch.float16, torch.bfloat16):
        n, c, s, g = 3, 32, 128, 8
        x = torch.randn(n, c, s, dtype=dt, device="npu")
        w = torch.randn(c, dtype=dt, device="npu")
        b = torch.randn(c, dtype=dt, device="npu")
        ref = F.group_norm(x.cpu().float(), g, w.cpu().float(), b.cpu().float(), 1e-5)
        y = run(x, g, w, b)
        err = (y.cpu().float() - ref).abs().max().item()
        print("group_norm", str(dt).split(".")[-1], "err", err)
        if err > tol[dt]:
            raise AssertionError((dt, err))

        ref_sw = F.silu(ref)
        y_sw = run(x, g, w, b, swish=True)
        err = (y_sw.cpu().float() - ref_sw).abs().max().item()
        print("group_norm_swish", str(dt).split(".")[-1], "err", err)
        if err > tol[dt] * 2:
            raise AssertionError((dt, err))

    # no affine
    x = torch.randn(2, 16, 2, 64, dtype=torch.float32, device="npu")
    ref = F.group_norm(x.cpu(), 4)
    y = run(x, 4)
    err = (y.cpu() - ref).abs().max().item()
    print("group_norm no-affine 4d err", err)
    if err > 2e-5:
        raise AssertionError(err)
