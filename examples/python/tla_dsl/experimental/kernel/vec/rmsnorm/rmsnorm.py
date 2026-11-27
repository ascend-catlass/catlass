# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Row-wise RMSNorm (fp32, optional weight/bias), one AIV block per row tile.

    y[r] = x[r] / sqrt(mean(x[r]^2) + eps) * w + b

性能设计要点（与 ``layernorm`` 同构，只是没有去均值这一趟）：

1. **tile 粒度连续 DMA + BUFS 三段流水**：连续 ``TILE_ROWS`` 行一个 tile，
   每个块认领一段连续 tile，每个 tile 一次 MTE2 搬入、一次 MTE3 搬出；``BUFS``
   个缓冲用 per-buffer mutex 串起来，让搬入 / 计算 / 搬出三段重叠。缓冲数按 UB
   预算取，至少保证两个 tile 在飞。
2. **行摊薄（TILE_ROWS=4 时最明显）**：四行的平方和合并在同一个动态 chunk 循环，
   四条独立累加链提供 ILP；缩放趟的 ``w``/``b`` lane 每 tile 只 load 一次。
   ``TILE_ROWS`` 按 UB 预算在 4 / 2 / 1 间选择（见 host 侧 ``_pipeline``）。
3. **先向量累加、每行只 reduce 一次**：reduce 的固定开销摊到 ``n_chunks`` 次迭代。
4. **``w``/``b`` 常驻 UB**：每块开头搬一次，用 ``wb_mutex``（MTE2 -> VECTOR）
   保护；无仿射时该同步在编译期消除。
5. **非对齐 N 的尾块路径**：UB 行按 ``N_PAD = ceil(N/64)*64`` 补齐（GM 侧稠密
   ``N``），末块按完整 64-lane 向量读写，``update_mask`` 只放行前 ``TAIL`` 个
   lane；对齐时整条路径被 ``const_expr`` 消除。
"""

import functools
import inspect
import threading

import torch

import catlass.tla as tla
from catlass.tla.runtime import _Tensor, from_dlpack

VL_ELE = 64  # fp32 vector length (256 B / 4 B)
UB_BYTES = 262144  # per-AIV UB capacity, used to size the tile pipeline


def _chunk(tensor, row, col):
    """64-lane view of ``tensor[row, col * 64 : (col + 1) * 64]``."""
    return tla.tile_view(tensor, tla.make_shape(1, VL_ELE), tla.make_coord(row, col))


def _reduce_total(sq_sum, mask, tail_acc=None, tail_mask=None):
    """Broadcast the row sum of ``sq_sum`` (plus ``tail_acc``) as a scalar.

    A masked vector op zeroes the lanes it does not touch, so the tail
    accumulator only carries data in its first ``TAIL`` lanes and the head
    accumulator must be reduced *before* the two are combined.
    """
    total = tla.full(sq_sum.reduce(tla.ReductionOp.ADD, mask=mask), dtype=tla.Float32)
    if tail_acc is not None:
        total = tla.add(
            total,
            tla.full(
                tail_acc.reduce(tla.ReductionOp.ADD, mask=tail_mask), dtype=tla.Float32
            ),
            mask=mask,
        )
    return total


def _rstd_of(sq_sum, INV_N, EPS, mask, tail_acc=None, tail_mask=None):
    """Broadcast ``1 / sqrt(mean(x^2) + eps)`` for one row."""
    ms = tla.mul(_reduce_total(sq_sum, mask, tail_acc, tail_mask), INV_N, mask=mask)
    return tla.div(
        tla.full(1.0, dtype=tla.Float32),
        tla.sqrt(tla.add(ms, EPS, mask=mask), mask=mask),
        mask=mask,
    )


# --- partial-chunk (N % 64 != 0) helpers ------------------------------------
# Same scheme as ``layernorm``: the tail chunk is one full 64-lane vector inside
# the padded UB row, only its leading ``TAIL`` lanes are valid.


def _tail_sq_sum(acc, tensor, row, col, tail_mask):
    x = _chunk(tensor, row, col).load()
    return tla.add(acc, tla.mul(x, x, mask=tail_mask), mask=tail_mask)


def _tail_scale(tensor, row, col, rstd, wv, bv, tail_mask):
    x = _chunk(tensor, row, col)
    out = tla.mul(x.load(), rstd, mask=tail_mask)
    if wv is not None:
        out = tla.add(tla.mul(out, wv, mask=tail_mask), bv, mask=tail_mask)
    x.store(out)


@tla.kernel
def rmsnorm_kernel(
    gm_x: tla.Tensor,  # [M, N] fp32
    gm_w: tla.Tensor,  # [N] fp32 (unused when HAS_AFFINE == 0)
    gm_b: tla.Tensor,  # [N] fp32 (unused when HAS_AFFINE == 0)
    gm_y: tla.Tensor,  # [M, N] fp32
    N: tla.Constexpr[int],
    N_PAD: tla.Constexpr[int],
    TAIL: tla.Constexpr[int],
    INV_N: tla.Constexpr[float],
    EPS: tla.Constexpr[float],
    HAS_AFFINE: tla.Constexpr[int],
    TILE_ROWS: tla.Constexpr[int],
    BUFS: tla.Constexpr[int],
) -> None:
    m = gm_x.origin_shape[0]

    # One mutex per pipeline buffer guards a whole (load -> compute -> store)
    # round trip: MTE2 unlocks it when the GM->UB copy retires, VECTOR then owns
    # the buffer, and MTE3 takes it over once the stores retire. Ids 0..BUFS-1
    # are the tile buffers; 31 is the resident w/b pair.
    x_ptr = [None] * BUFS
    tile_mutex = [None] * BUFS
    for bi in tla.range_constexpr(BUFS):
        x_ptr[bi] = tla.allocate(
            TILE_ROWS * N_PAD, tla.Float32, tla.AddressSpace.ub, 256
        )
        tile_mutex[bi] = tla.mutex(resource="rms_tile", id=bi)
    wb_mutex = tla.mutex(resource="rms_wb", id=31)

    w_ptr = tla.allocate(N_PAD, tla.Float32, tla.AddressSpace.ub, 256)
    b_ptr = tla.allocate(N_PAD, tla.Float32, tla.AddressSpace.ub, 256)

    # UB rows are padded to a whole 64-lane vector (N_PAD) so the tail chunk can
    # be read as one full vector; the global-memory side stays dense (N).
    row_layout = tla.make_layout(
        tla.make_shape(1, N),
        tla.make_stride(N_PAD, 1),
        origin_shape=tla.make_shape(1, N),
    )
    tile_layout = tla.make_layout(
        tla.make_shape(TILE_ROWS, N),
        tla.make_stride(N_PAD, 1),
        origin_shape=tla.make_shape(TILE_ROWS, N),
    )
    gm_layout = tla.make_layout(
        tla.make_shape(TILE_ROWS, N),
        tla.make_stride(N, 1),
        origin_shape=tla.make_shape(TILE_ROWS, N),
    )
    ub_tile = [None] * BUFS
    for bj in tla.range_constexpr(BUFS):
        ub_tile[bj] = tla.make_tensor(x_ptr[bj], tile_layout)
    ub_w = tla.make_tensor(w_ptr, row_layout)
    ub_b = tla.make_tensor(b_ptr, row_layout)

    n_chunks = N // VL_ELE
    gid = tla.arch.block_idx()
    aiv_num = tla.arch.block_num()
    n_tiles = m // TILE_ROWS
    # Each block owns a *contiguous* span of tiles, so one tile is one contiguous
    # GM range (TILE_ROWS * N elements) and costs exactly one DMA in and out.
    tiles_per_block = (n_tiles + aiv_num - 1) // aiv_num
    tile_begin = gid * tiles_per_block

    with tla.vector():
        # w/b are resident for the whole block, so they are fetched once and the
        # mutex only has to cover the first (and only) MTE2 -> VECTOR hand-off.
        if tla.const_expr(HAS_AFFINE == 1):
            wb_mutex.lock(pipe=tla.pipes.MTE2)
            tla.copy(ub_w, gm_w)
            tla.copy(ub_b, gm_b)
            wb_mutex.unlock(pipe=tla.pipes.MTE2)

        # Software pipeline: the three stages below are issued back to back for
        # every buffer, and the mutexes (not program order) decide when a stage
        # actually runs, so MTE2 for buffer bi+1 overlaps VECTOR for bi and MTE3
        # for bi-1. ``tiles_per_block`` is the only dynamic trip count.
        for grp in tla.range((tiles_per_block + BUFS - 1) // BUFS):
            for bi in tla.range_constexpr(BUFS):
                tile = tile_begin + grp * BUFS + bi
                if tile < n_tiles:
                    tile_off = tile.to(tla.Int64) * tla.Int64(TILE_ROWS * N)
                    gm_tile_x = tla.make_tensor(gm_x.ptr + tile_off, gm_layout)
                    gm_tile_y = tla.make_tensor(gm_y.ptr + tile_off, gm_layout)

                    tile_mutex[bi].lock(pipe=tla.pipes.MTE2)
                    tla.copy(ub_tile[bi], gm_tile_x)
                    tile_mutex[bi].unlock(pipe=tla.pipes.MTE2)

                    tile_mutex[bi].lock(pipe=tla.pipes.VECTOR)
                    if tla.const_expr(HAS_AFFINE == 1):
                        wb_mutex.lock(pipe=tla.pipes.VECTOR)
                    with tla.vec.func(mode="simd"):
                        mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                        tail_mask = None
                        if tla.const_expr(TAIL != 0):
                            tail_mask, _ = tla.update_mask(TAIL, dtype=tla.Float32)

                        # Two passes over the tile: (1) lane-wise sum of squares,
                        # (2) scale + optional affine, written back in place.
                        # The sum stays in 64 lanes and is reduced once per row.
                        if tla.const_expr(TILE_ROWS == 4):
                            # Four rows share one dynamic chunk loop: independent
                            # square-sum chains (ILP) and one w/b load per chunk
                            # instead of one per row.
                            q0 = tla.full(0.0, dtype=tla.Float32)
                            q1 = tla.full(0.0, dtype=tla.Float32)
                            q2 = tla.full(0.0, dtype=tla.Float32)
                            q3 = tla.full(0.0, dtype=tla.Float32)
                            for c1 in tla.range(n_chunks):
                                t0 = _chunk(ub_tile[bi], 0, c1).load()
                                q0 = tla.add(q0, tla.mul(t0, t0, mask=mask), mask=mask)
                                t1 = _chunk(ub_tile[bi], 1, c1).load()
                                q1 = tla.add(q1, tla.mul(t1, t1, mask=mask), mask=mask)
                                t2 = _chunk(ub_tile[bi], 2, c1).load()
                                q2 = tla.add(q2, tla.mul(t2, t2, mask=mask), mask=mask)
                                t3 = _chunk(ub_tile[bi], 3, c1).load()
                                q3 = tla.add(q3, tla.mul(t3, t3, mask=mask), mask=mask)
                            if tla.const_expr(TAIL != 0):
                                tq0 = tla.full(0.0, dtype=tla.Float32)
                                tq1 = tla.full(0.0, dtype=tla.Float32)
                                tq2 = tla.full(0.0, dtype=tla.Float32)
                                tq3 = tla.full(0.0, dtype=tla.Float32)
                                tq0 = _tail_sq_sum(
                                    tq0, ub_tile[bi], 0, n_chunks, tail_mask
                                )
                                tq1 = _tail_sq_sum(
                                    tq1, ub_tile[bi], 1, n_chunks, tail_mask
                                )
                                tq2 = _tail_sq_sum(
                                    tq2, ub_tile[bi], 2, n_chunks, tail_mask
                                )
                                tq3 = _tail_sq_sum(
                                    tq3, ub_tile[bi], 3, n_chunks, tail_mask
                                )
                                r0 = _rstd_of(q0, INV_N, EPS, mask, tq0, tail_mask)
                                r1 = _rstd_of(q1, INV_N, EPS, mask, tq1, tail_mask)
                                r2 = _rstd_of(q2, INV_N, EPS, mask, tq2, tail_mask)
                                r3 = _rstd_of(q3, INV_N, EPS, mask, tq3, tail_mask)
                            else:
                                r0 = _rstd_of(q0, INV_N, EPS, mask)
                                r1 = _rstd_of(q1, INV_N, EPS, mask)
                                r2 = _rstd_of(q2, INV_N, EPS, mask)
                                r3 = _rstd_of(q3, INV_N, EPS, mask)

                            # (2) scale in place; the w/b lanes are loaded once
                            # per chunk and reused by all four rows.
                            for c2 in tla.range(n_chunks):
                                wv = None
                                bv = None
                                if tla.const_expr(HAS_AFFINE == 1):
                                    wv = _chunk(ub_w, 0, c2).load()
                                    bv = _chunk(ub_b, 0, c2).load()

                                x0 = _chunk(ub_tile[bi], 0, c2)
                                v0 = tla.mul(x0.load(), r0, mask=mask)
                                if tla.const_expr(HAS_AFFINE == 1):
                                    v0 = tla.add(
                                        tla.mul(v0, wv, mask=mask), bv, mask=mask
                                    )
                                x0.store(v0)

                                x1 = _chunk(ub_tile[bi], 1, c2)
                                v1 = tla.mul(x1.load(), r1, mask=mask)
                                if tla.const_expr(HAS_AFFINE == 1):
                                    v1 = tla.add(
                                        tla.mul(v1, wv, mask=mask), bv, mask=mask
                                    )
                                x1.store(v1)

                                x2 = _chunk(ub_tile[bi], 2, c2)
                                v2 = tla.mul(x2.load(), r2, mask=mask)
                                if tla.const_expr(HAS_AFFINE == 1):
                                    v2 = tla.add(
                                        tla.mul(v2, wv, mask=mask), bv, mask=mask
                                    )
                                x2.store(v2)

                                x3 = _chunk(ub_tile[bi], 3, c2)
                                v3 = tla.mul(x3.load(), r3, mask=mask)
                                if tla.const_expr(HAS_AFFINE == 1):
                                    v3 = tla.add(
                                        tla.mul(v3, wv, mask=mask), bv, mask=mask
                                    )
                                x3.store(v3)

                            if tla.const_expr(TAIL != 0):
                                tw = None
                                tb = None
                                if tla.const_expr(HAS_AFFINE == 1):
                                    tw = _chunk(ub_w, 0, n_chunks).load()
                                    tb = _chunk(ub_b, 0, n_chunks).load()
                                _tail_scale(
                                    ub_tile[bi], 0, n_chunks, r0, tw, tb, tail_mask
                                )
                                _tail_scale(
                                    ub_tile[bi], 1, n_chunks, r1, tw, tb, tail_mask
                                )
                                _tail_scale(
                                    ub_tile[bi], 2, n_chunks, r2, tw, tb, tail_mask
                                )
                                _tail_scale(
                                    ub_tile[bi], 3, n_chunks, r3, tw, tb, tail_mask
                                )
                        elif tla.const_expr(TILE_ROWS == 2):
                            # Same two passes with two rows per tile: still two
                            # independent chains and one w/b load per chunk.
                            q0 = tla.full(0.0, dtype=tla.Float32)
                            q1 = tla.full(0.0, dtype=tla.Float32)
                            for c1 in tla.range(n_chunks):
                                t0 = _chunk(ub_tile[bi], 0, c1).load()
                                q0 = tla.add(q0, tla.mul(t0, t0, mask=mask), mask=mask)
                                t1 = _chunk(ub_tile[bi], 1, c1).load()
                                q1 = tla.add(q1, tla.mul(t1, t1, mask=mask), mask=mask)
                            if tla.const_expr(TAIL != 0):
                                tq0 = tla.full(0.0, dtype=tla.Float32)
                                tq1 = tla.full(0.0, dtype=tla.Float32)
                                tq0 = _tail_sq_sum(
                                    tq0, ub_tile[bi], 0, n_chunks, tail_mask
                                )
                                tq1 = _tail_sq_sum(
                                    tq1, ub_tile[bi], 1, n_chunks, tail_mask
                                )
                                r0 = _rstd_of(q0, INV_N, EPS, mask, tq0, tail_mask)
                                r1 = _rstd_of(q1, INV_N, EPS, mask, tq1, tail_mask)
                            else:
                                r0 = _rstd_of(q0, INV_N, EPS, mask)
                                r1 = _rstd_of(q1, INV_N, EPS, mask)

                            for c2 in tla.range(n_chunks):
                                wv = None
                                bv = None
                                if tla.const_expr(HAS_AFFINE == 1):
                                    wv = _chunk(ub_w, 0, c2).load()
                                    bv = _chunk(ub_b, 0, c2).load()

                                x0 = _chunk(ub_tile[bi], 0, c2)
                                v0 = tla.mul(x0.load(), r0, mask=mask)
                                if tla.const_expr(HAS_AFFINE == 1):
                                    v0 = tla.add(
                                        tla.mul(v0, wv, mask=mask), bv, mask=mask
                                    )
                                x0.store(v0)

                                x1 = _chunk(ub_tile[bi], 1, c2)
                                v1 = tla.mul(x1.load(), r1, mask=mask)
                                if tla.const_expr(HAS_AFFINE == 1):
                                    v1 = tla.add(
                                        tla.mul(v1, wv, mask=mask), bv, mask=mask
                                    )
                                x1.store(v1)

                            if tla.const_expr(TAIL != 0):
                                tw = None
                                tb = None
                                if tla.const_expr(HAS_AFFINE == 1):
                                    tw = _chunk(ub_w, 0, n_chunks).load()
                                    tb = _chunk(ub_b, 0, n_chunks).load()
                                _tail_scale(
                                    ub_tile[bi], 0, n_chunks, r0, tw, tb, tail_mask
                                )
                                _tail_scale(
                                    ub_tile[bi], 1, n_chunks, r1, tw, tb, tail_mask
                                )
                        else:
                            # Single-row tiles (wide N): no row to share w/b loads with.
                            sq = tla.full(0.0, dtype=tla.Float32)
                            for c1 in tla.range(n_chunks):
                                t0 = _chunk(ub_tile[bi], 0, c1).load()
                                sq = tla.add(sq, tla.mul(t0, t0, mask=mask), mask=mask)
                            if tla.const_expr(TAIL != 0):
                                tq = tla.full(0.0, dtype=tla.Float32)
                                tq = _tail_sq_sum(
                                    tq, ub_tile[bi], 0, n_chunks, tail_mask
                                )
                                rstd = _rstd_of(sq, INV_N, EPS, mask, tq, tail_mask)
                            else:
                                rstd = _rstd_of(sq, INV_N, EPS, mask)

                            for c2 in tla.range(n_chunks):
                                xo = _chunk(ub_tile[bi], 0, c2)
                                out = tla.mul(xo.load(), rstd, mask=mask)
                                if tla.const_expr(HAS_AFFINE == 1):
                                    out = tla.add(
                                        tla.mul(
                                            out, _chunk(ub_w, 0, c2).load(), mask=mask
                                        ),
                                        _chunk(ub_b, 0, c2).load(),
                                        mask=mask,
                                    )
                                xo.store(out)

                            if tla.const_expr(TAIL != 0):
                                tw = None
                                tb = None
                                if tla.const_expr(HAS_AFFINE == 1):
                                    tw = _chunk(ub_w, 0, n_chunks).load()
                                    tb = _chunk(ub_b, 0, n_chunks).load()
                                _tail_scale(
                                    ub_tile[bi], 0, n_chunks, rstd, tw, tb, tail_mask
                                )
                    if tla.const_expr(HAS_AFFINE == 1):
                        wb_mutex.unlock(pipe=tla.pipes.VECTOR)
                    tile_mutex[bi].unlock(pipe=tla.pipes.VECTOR)

                    tile_mutex[bi].lock(pipe=tla.pipes.MTE3)
                    tla.copy(gm_tile_y, ub_tile[bi])
                    tile_mutex[bi].unlock(pipe=tla.pipes.MTE3)

        tla.pipe_barrier(tla.pipes.ALL)


# ======================================================================================
# HOST side: shape checks, compile and launch.
# ======================================================================================
# 性能设计要点：小 shape（M <= 1024）的耗时完全由 host 侧启动开销决定，实测单次
# 开销里 from_dlpack ~47 us/张量、get_device_properties ~285 us、current_stream
# ~11 us、payload 打包 ~10 us，而 kernel 本身只有几微秒。因此这里做三级缓存：
#
#   _COMPILED  按 (M, N, eps, 是否仿射) 缓存编译产物；
#   _BINDINGS  按 (data_ptr, rows, cols) 直接构造 host 张量，绕开 from_dlpack；
#   _FAST      按 (四个张量地址, M, N, eps, 仿射) 缓存「绑定张量 + 预打包 payload
#              + functools.partial 启动器」，命中时只剩一次字典查找 + 一次函数调用。
#
# 设备索引、stream、vector core 数同样缓存（键用 int，不用 str(device)）。
# 换了 torch.npu.stream 需要调用 reset_stream_cache()。

_COMPILED = {}
_VECTOR_CORES = {}
_BINDINGS = {}
_DUMMY = {}
_ZERO_BIAS = {}
_FAST = {}
_MAX_BINDINGS = 4096
_MAX_FAST = 256


def _vector_cores(index):
    cores = _VECTOR_CORES.get(index)
    if cores is None:
        cores = int(torch.npu.get_device_properties(index).vector_core_num)
        _VECTOR_CORES[index] = cores
    return cores


def _bind(buf, rows, cols):
    """Bind a contiguous fp32 torch buffer as a 2-D TLA host tensor.

    ``from_dlpack`` costs tens of microseconds per tensor and dominates the
    launch at these sizes, so the host tensor is built straight from the buffer
    address. The caller keeps ``buf`` alive across the launch, and the binding
    is memoized by ``(address, rows, cols)``.
    """
    sig = (buf.data_ptr(), rows, cols)
    view = _BINDINGS.get(sig)
    if view is not None:
        return view
    if buf.dtype is not torch.float32:
        return from_dlpack(buf, layout_tag=tla.arch.RowMajor)
    view = _Tensor(
        (rows, cols),
        tla.Float32,
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
    # Keying on the integer index avoids the ~0.4 us ``str(torch.device)`` cost
    # on every call.
    key = device.index
    buf = _DUMMY.get(key)
    if buf is None:
        buf = torch.ones((1, 1), dtype=torch.float32, device=device)
        _DUMMY[key] = buf
    return buf


def _zero_bias(device, n):
    key = (device.index, n)
    buf = _ZERO_BIAS.get(key)
    if buf is None:
        buf = torch.zeros((n,), dtype=torch.float32, device=device)
        _ZERO_BIAS[key] = buf
    return buf


# Resolving the current stream through torch costs ~11 us per launch and packing
# the launch payload another ~10 us; both are cached here. A launch issued inside
# a different ``torch.npu.stream`` context keeps the stream cached on the first
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


def _pipeline(m, n):
    """Pick the tile shape and prefetch depth from the UB budget.

    ``TILE_ROWS`` is the widest tile that still leaves room for two in-flight
    tiles plus the resident w/b pair (``8 * n * rows`` for two tiles,
    ``8 * n`` for w/b, 8 KiB of slack) -- wider is strictly better because the
    w/b load is shared by more rows. ``BUFS`` then fills whatever UB is left,
    capped at 4, so a shallow pipeline is only used for very wide ``N``.
    """
    tile_rows = 1
    for candidate in (4, 2):
        if m % candidate == 0 and 8 * n * candidate + 8 * n + 8192 <= UB_BYTES:
            tile_rows = candidate
            break
    budget = UB_BYTES - 8 * n - 8192
    bufs = max(1, min(4, budget // (4 * n * tile_rows)))
    return tile_rows, bufs


def run(x, weight=None, bias=None, eps=1e-6):
    """Return RMSNorm(x) over the last axis (fp32 in / fp32 out)."""
    if x.dtype is not torch.float32:
        raise ValueError("x must be float32")
    if x.dim() != 2:
        raise ValueError("x must be 2-D [M, N]")
    m, n = x.shape
    n_pad = (n + VL_ELE - 1) // VL_ELE * VL_ELE
    tail = n - (n // VL_ELE) * VL_ELE

    has_affine = 1 if weight is not None else 0
    if not x.is_contiguous():
        # The kernel binds the buffer as a dense [M, N] host tensor, so a
        # strided input has to be materialised first.
        x = x.contiguous()
    device = x.device
    if weight is None:
        w = _dummy(device)
    elif weight.dtype is torch.float32 and weight.is_contiguous():
        w = weight
    else:
        w = weight.to(torch.float32).contiguous()
    if bias is None:
        b = _zero_bias(device, n) if has_affine else _dummy(device)
    elif bias.dtype is torch.float32 and bias.is_contiguous():
        b = bias
    else:
        b = bias.to(torch.float32).contiguous()
    if has_affine:
        if w.numel() != n:
            raise ValueError("weight must have N elements")
        if bias is not None and b.numel() != n:
            raise ValueError("bias must have N elements")

    out = torch.empty((m, n), dtype=torch.float32, device=device)
    if m == 0:
        return out

    # One dictionary lookup for the whole hot path: the key pins the buffers
    # (which also keeps temporaries alive) plus every compile-time parameter.
    key = (
        x.data_ptr(),
        w.data_ptr(),
        b.data_ptr(),
        out.data_ptr(),
        m,
        n,
        has_affine,
        float(eps),
    )
    plan = _FAST.get(key)
    if plan is not None:
        launch = plan[0]
        if launch is not None:
            launch()
        else:
            plan[1](*plan[2], block_num=plan[3], stream=plan[4])
        return out

    tile_rows, bufs = _pipeline(m, n_pad)
    compile_key = (m, n, has_affine, float(eps))
    entry = _COMPILED.get(compile_key)
    if entry is None:
        artifact = tla.compile(
            rmsnorm_kernel,
            _bind(x, m, n),
            _bind(w, 1, n),
            _bind(b, 1, n),
            _bind(out, m, n),
            n,
            n_pad,
            tail,
            1.0 / float(n),
            float(eps),
            has_affine,
            tile_rows,
            bufs,
            options="--npu-arch 3510",
        )
        entry = (artifact, tile_rows, bufs)
        _COMPILED[compile_key] = entry
    artifact = entry[0]

    device_index = device.index
    cores = _vector_cores(device_index)
    stream = _launch_stream(device_index)
    tensors = (_bind(x, m, n), _bind(w, 1, n), _bind(b, 1, n), _bind(out, m, n))
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

    torch.npu.set_device(0)
    m, n = 128, 256
    torch.manual_seed(0)
    x = torch.randn(m, n, dtype=torch.float32, device="npu")
    w = torch.randn(n, dtype=torch.float32, device="npu")

    y = run(x, w)
    xr = x.cpu()
    ref = xr / torch.sqrt(xr.pow(2).mean(-1, keepdim=True) + 1e-6) * w.cpu()
    print("max abs err:", (y.cpu() - ref).abs().max().item())
