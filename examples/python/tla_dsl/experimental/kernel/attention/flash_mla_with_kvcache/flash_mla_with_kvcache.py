# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Flash MLA decoding with a paged KV cache: end-to-end example.

Multi-head latent attention decodes against a paged cache that stores 576
latent features per token; the first 512 of them are the absorbed values, so the
same page feeds both the QK product and the PV product:

.. math::

    S_{h,q,k} = \\frac{Q_{h,q} \\cdot K_k}{\\sqrt{576}}, \\qquad
    O_{h,q} = \\sum_k \\operatorname{softmax}(S)_{h,q,k} \\cdot V_k

``cache_seqlens`` makes the cache per-request variable length, and
``mask_mode=3`` applies the right-down causal mask over those lengths. Only the
query side is variable length (``cu_seqlens_q`` batches the requests), which is
the decode shape: ``total_q`` query rows in total, one or a few per request.

The operator is a single mixed Cube/Vector kernel with a one-tile QK lookahead;
when neither the query rows nor the heads fill the device, the host splits the
query rows over head groups and, for short inputs, the KV range over chunks
(``kvsplit``) and reduces the f32 partials in a second kernel.

Supported product: Ascend 950PR/950DT.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass

import catlass.tla as tla
import torch
import torch_npu  # noqa: F401  -- registers torch.npu
from catlass.tla.runtime import from_dlpack


# =============================================================================
# Fixed shapes and selector values.
# =============================================================================
#
# The kernel is tiled for one latent-attention head: the cache carries 576
# features per token, of which the first 512 are the absorbed values. Nothing here
# depends on the DSL.

PAGE_SIZE = 128  # tokens per KV page, fixed by the paged cache layout
HEAD_DIM_QK = 576  # latent query/key features
HEAD_DIM_V = 512  # absorbed value features (the leading 512 of the 576)
KV_HEAD_NUM = 1  # MLA: every query head reads the same latent KV head

MASK_NONE = 0
MASK_RIGHT_DOWN_CAUSAL = 3
MASK_MODES = (MASK_NONE, MASK_RIGHT_DOWN_CAUSAL)

CACHE_LAYOUT_BBND = "PA_BBND"  # [pages, 128, 1, 576]
CACHE_LAYOUT_NZ = "PA_NZ"  # [pages, 1, 36, 128, 16]
CACHE_LAYOUTS = (CACHE_LAYOUT_BBND, CACHE_LAYOUT_NZ)

HEAD_CHOICES = (64, 96)  # query heads a single (token, group) task can hold
Q_BLOCK = 64  # unpaired tile of query heads per task
KV_BLOCK = 128  # logical KV tokens per pipeline tile
QUERY_CHUNK = 32  # query rows per golden chunk, bounds the golden's P buffer
DTYPE_NAMES = ("float16", "bfloat16")

NPU_ARCH = "--npu-arch 3510"  # Ascend 950PR/DT

D_QK = HEAD_DIM_QK
D_V = HEAD_DIM_V


# =============================================================================
# Device kernels.
# =============================================================================
#
# ``task = (query row, head group)``, ``tile = KV_BLOCK=128 logical KV tokens``.
#
# * AIC runs the Cube pipeline: one-tile QK lookahead (gather K(t+1) and issue
# QK(t+1) while the vector core consumes S(t)), paged K/V gather, QK over the
# 576-wide cache (6 x 96 MMAD chunks, L0A/L0B ping-pong) and PV over the 512
# absorbed value features (four 128-wide halves, L0B ping-pong).
# * AIV runs the online softmax over the valid columns, writes P back to L1 as
# fp16/bf16, rescales the running O and produces LSE.
#
# The KV cache is either ``PA_BBND`` ``[pages, 128, 1, 576]`` or ``PA_NZ``
# ``[pages, 1, 36, 128, 16]``; ``mask_mode`` 0 is dense and 3 is right-down causal
# over ``cache_seqlens[b]``.
#
# ``flash_mla_kvsplit_reduce_kernel`` is the combine pass of the KV-split variant:
# it merges the f32 partial outputs and partial LSEs of ``kvsplit`` KV chunks into
# the final fp16/bf16 output and the global LSE.
@tla.kernel
def flash_mla_with_kvcache_kernel(
    mem_q: tla.Tensor,
    mem_k: tla.Tensor,
    mem_o: tla.Tensor,
    mem_lse: tla.Tensor,
    mem_po: tla.Tensor,
    mem_pl: tla.Tensor,
    mem_bt: tla.Tensor,
    mem_cu: tla.Tensor,
    mem_kvlen: tla.Tensor,
    softmax_scale: tla.Float32,
    total_q: tla.Int32,
    batch: tla.Int32,
    heads: tla.Int32,
    num_pages: tla.Int32,
    kv_blk_stride: tla.Int32,
    mask_mode: tla.Int32,
    d_qk: tla.Constexpr[int],
    d_v: tla.Constexpr[int],
    q_block: tla.Constexpr[int],
    # 低并行度（total_q 极少）时把一个 query 的 head 维切成 hgroups 组，
    # 由多个核并行处理同一 query 的不同 head 段；hgroups=1 时行为不变。
    hgroups: tla.Constexpr[int],
    # KV-split：把每个 task 的 tile 序列切成 kvsplit 段，由多个核并行计算，
    # f32 partial 写入 mem_po，partial LSE 写入 mem_pl；kvsplit=1 时行为不变。
    kvsplit: tla.Constexpr[int],
    kv_block: tla.Constexpr[int],
    pv_width: tla.Constexpr[int],
    nz_layout: tla.Constexpr[int],
) -> None:
    c0 = 0
    c1 = 1
    c2 = 2
    alignment = 512
    l0c_alignment = 1024  # 2.2.1 起 L0C 要求 1024 字节对齐
    min_value = -3.0e38
    max_floor = -1.0e30
    vl_f32 = 64
    c8 = 8
    q_sub = q_block // 2
    num_dim_chunks = d_v // vl_f32
    # N=64 用两 token 一 task（M=128），L0A 只能放 6 段 Q(64x96)；N=96 用 6x96。
    qk_chunk = 64 if q_block == 128 else 96
    qk_full_chunks = d_qk // qk_chunk
    # K 分半的维度必须同时是 qk_chunk 与 nZ 16 维分组的整数倍（chunk 严格归属
    # 低/高半页，否则 MTE2 覆盖与 MTE1 读取的 mutex 交接会错半页）。
    qk_half_chunks = qk_full_chunks // 2
    k_split = qk_chunk * qk_half_chunks
    # pairing（M=128）时每个 AIV 子核负责一个 token 的 64 个 head。
    head_scale = c0 if q_block == 128 else c1
    pv_splits = d_v // pv_width
    pv_chunks = pv_width // vl_f32
    out_chunk = 128
    out_chunks = d_v // out_chunk
    io_dtype = mem_q.ptr.dtype

    # task = 一个 query token（N=96）或一对相邻 query token（N=64）。
    num_tasks = total_q * hgroups * kvsplit
    if tla.const_expr(q_block == 128):
        num_tasks = tla.as_numeric(c0)
    for count_b in tla.range(batch):
        if tla.const_expr(q_block == 128):
            count_qlen = mem_cu[count_b + c1] - mem_cu[count_b]
            num_tasks = num_tasks + (count_qlen + c1) // c2

    # ---------------------------------------------------------------- buffers
    l1_q_ptr = tla.allocate(q_block * d_qk, io_dtype, tla.AddressSpace.l1, alignment)
    l1_k_ptr = tla.allocate(d_qk * kv_block, io_dtype, tla.AddressSpace.l1, alignment)
    l1_v_ptr = tla.allocate(kv_block * d_v, io_dtype, tla.AddressSpace.l1, alignment)
    l1_p_ptr = tla.allocate(
        q_block * kv_block, io_dtype, tla.AddressSpace.l1, alignment
    )

    # L0A/L0B 双槽：QK 的 K 分段与 PV 的 V 分段按 slot 轮转，使 MTE1 的下一段
    # 搬运能与 Cube 的当前段 MMAD 重叠。
    l0a_q0_ptr = tla.allocate(
        q_block * qk_chunk, io_dtype, tla.AddressSpace.l0a, alignment
    )
    l0a_q1_ptr = tla.allocate(
        q_block * qk_chunk, io_dtype, tla.AddressSpace.l0a, alignment
    )
    l0a_p_ptr = tla.allocate(
        q_block * kv_block, io_dtype, tla.AddressSpace.l0a, alignment
    )
    l0b_elems = qk_chunk if qk_chunk > pv_width else pv_width
    l0b0_ptr = tla.allocate(
        kv_block * l0b_elems, io_dtype, tla.AddressSpace.l0b, alignment
    )
    l0b1_ptr = tla.allocate(
        kv_block * l0b_elems, io_dtype, tla.AddressSpace.l0b, alignment
    )
    l0a_q_ptrs = (l0a_q0_ptr, l0a_q1_ptr)
    l0b_ptrs = (l0b0_ptr, l0b1_ptr)

    l0c_s_ptr = tla.allocate(
        q_block * kv_block, tla.Float32, tla.AddressSpace.l0c, l0c_alignment
    )
    # L0C O 双槽：FIX(h) 读槽 h%2 时，PV(h+1) 可以写另一槽，MMAD 不再等 FIX。
    # UB otmp 双槽只在非双 token 变体启用（M=128 时双槽会超出 UB）。
    l0c_o_ptr = tla.allocate(
        q_block * pv_width, tla.Float32, tla.AddressSpace.l0c, l0c_alignment
    )
    l0c_o1_ptr = tla.allocate(
        q_block * pv_width, tla.Float32, tla.AddressSpace.l0c, l0c_alignment
    )
    l0c_o_ptrs = (l0c_o_ptr, l0c_o1_ptr)
    # N=96：每个 PV half 独占一个 otmp 槽（SPLIT_M 下每核只有 q_sub 行），
    # 因此 4 个 half 不再互相覆盖，跨核 O 握手可以降到每 tile 一次。
    # M=128：UB 放不下 4 槽，保持 2 槽 + 每 half 一次握手。
    is_pair = tla.const_expr(q_block == 128)
    otmp_n = c1 if is_pair else 4
    otmp_stride = q_block * pv_width if is_pair else q_sub * pv_width

    ub_s_ptr = tla.allocate(
        q_sub * kv_block, tla.Float32, tla.AddressSpace.ub, alignment
    )
    ub_p_ptr = tla.allocate(
        (q_sub + c1) * kv_block, io_dtype, tla.AddressSpace.ub, alignment
    )
    ub_o_ptr = tla.allocate(q_sub * d_v, tla.Float32, tla.AddressSpace.ub, alignment)
    ub_otmp_ptr = tla.allocate(
        otmp_n * otmp_stride, tla.Float32, tla.AddressSpace.ub, alignment
    )
    # 输出 staging 复用 UB P 区（epilogue 时 P 已搬到 L1，且不与 AIC 的下一
    # task FIX 冲突）；不能复用 otmp，因为融合后要在同一个 pass 里读 otmp。
    ub_out_ptr = tla.recast_ptr(ub_p_ptr, dtype=io_dtype)
    ub_now_max_ptr = tla.allocate(q_sub, tla.Float32, tla.AddressSpace.ub, alignment)
    ub_last_max_ptr = tla.allocate(q_sub, tla.Float32, tla.AddressSpace.ub, alignment)
    ub_sum_ptr = tla.allocate(q_sub, tla.Float32, tla.AddressSpace.ub, alignment)
    ub_dm_ptr = tla.allocate(q_sub, tla.Float32, tla.AddressSpace.ub, alignment)
    ub_dm_prev_ptr = tla.allocate(q_sub, tla.Float32, tla.AddressSpace.ub, alignment)
    ub_upd_sum_ptr = tla.allocate(q_sub, tla.Float32, tla.AddressSpace.ub, alignment)
    ub_blk_sum_ptr = tla.allocate(q_sub, tla.Float32, tla.AddressSpace.ub, alignment)
    ub_lse_ptr = tla.allocate(q_sub, tla.Float32, tla.AddressSpace.ub, alignment)
    # 1/sum 在 task 收尾只算一次，供融合后的输出 pass 复用。
    ub_inv_ptr = tla.allocate(q_sub, tla.Float32, tla.AddressSpace.ub, alignment)

    # ------------------------------------------------------------ cross flags
    s_ready = tla.cross_flag("fmla_s_ready")
    s_free = tla.cross_flag("fmla_s_free")
    p_ready = tla.cross_flag("fmla_p_ready")
    o_ready = tla.cross_flag("fmla_o_ready")
    o_free = tla.cross_flag("fmla_o_free")

    # Intra-core producer/consumer mutexes. Cross-core handoff above stays on
    # tla.cross_flag; tla.mutex is only valid inside one AIC/AIV.
    mtx_q = tla.mutex("fmla_q_l1", id=0)
    mtx_k = tla.mutex("fmla_k_l1", id=1)
    # K 高半页（dims 288~575）的独立 mutex：K DMA 拆成两半后，QK 的前 3 个
    # 96 维 chunk 只等低半页，高半页 DMA 与低半页的 MTE1/MMAD 重叠。
    mtx_kh = tla.mutex("fmla_kh_l1", id=15)
    mtx_v = tla.mutex("fmla_v_l1", id=2)
    # V 高半页（cols 256~511，PV 的 half 2/3）的独立 mutex：V gather 拆半后
    # PV 前两个 half 只等低半页（与 K 分半同构）。
    mtx_vh = tla.mutex("fmla_vh_l1", id=12)
    mtx_l0s0 = tla.mutex("fmla_l0s0", id=9)
    mtx_l0s1 = tla.mutex("fmla_l0s1", id=10)
    mtx_l0p = tla.mutex("fmla_l0p", id=13)
    mtx_l0s = (mtx_l0s0, mtx_l0s1)
    mtx_l0c_s = tla.mutex("fmla_l0c_s", id=4)
    mtx_l0c_o = tla.mutex("fmla_l0c_o", id=5)
    mtx_l0c_o1 = tla.mutex("fmla_l0c_o1", id=14)
    mtx_l0c_o_s = (mtx_l0c_o, mtx_l0c_o1)
    mtx_ub_p = tla.mutex("fmla_ub_p", id=6)
    mtx_ub_out = tla.mutex("fmla_ub_out", id=7)
    mtx_ub_lse = tla.mutex("fmla_ub_lse", id=8)

    cast_zero = tla.params.CastParams(
        reg_slot=tla.params.RegSlot.ZERO,
        sat_mode=tla.params.SatMode.SAT,
        round_mode=tla.params.RoundMode.CAST_ROUND,
    )
    cast_one = tla.params.CastParams(
        reg_slot=tla.params.RegSlot.ONE,
        sat_mode=tla.params.SatMode.SAT,
        round_mode=tla.params.RoundMode.CAST_ROUND,
    )

    with tla.vector():
        # ub_o 只在核内清零一次：每个 task 的第一次累加都会被
        # dm = exp(floor - tile_max) = 0 覆盖，但**核内第一个 task** 若某些行
        # 在该 tile 全被 mask（此时 dm=1），acc_old*1 会读到 UB 未初始化值
        # （可能是 NaN，NaN*0 仍然是 NaN）。因此保留一次核级清零，去掉每 task
        # 的 98 KiB 向量 store，把 task 边界的 AIV 串行尾部缩短约 0.5 us/task。
        ub_o_seed = tla.make_tensor(
            ub_o_ptr,
            tla.make_layout(
                tla.make_shape(q_sub, d_v),
                tla.make_stride(d_v, c1),
            ),
        )
        with tla.vec.func(mode="simd"):
            preg_seed = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            zero_seed = tla.full(0.0, dtype=tla.Float32)
            for i_seed in tla.range_constexpr(q_sub):
                for j_seed in tla.range_constexpr(num_dim_chunks):
                    tile_seed = tla.tile_view(
                        ub_o_seed,
                        tla.make_shape(c1, vl_f32),
                        tla.make_coord(i_seed, j_seed),
                    )
                    tile_seed.store(zero_seed, mask=preg_seed)
        tla.cross_core_set_flag(s_free, tla.arch.VECTOR)
        tla.cross_core_set_flag(o_free, tla.arch.VECTOR)

    task_range = tla.range(
        tla.arch.block_idx(),
        num_tasks,
        tla.arch.block_num(),
    )
    for task in task_range:
        head_base = c0
        vec_idx = tla.arch.sub_block_idx()
        # task 的高位是 head 组、低位是 query 行（hgroups=1 时退化为原语义）。
        chunk = task % kvsplit
        task_rest = task // kvsplit
        hg = task_rest % hgroups
        task_row = task_rest // hgroups
        head_off = hg * q_block * head_scale + vec_idx * q_sub * head_scale

        # ---- task -> (request, 起始 query token) ----
        req = tla.as_numeric(0)
        row_a = task_row
        pair_acc = tla.as_numeric(c0)
        for scan_batch in tla.range(batch):
            scan_lo = mem_cu[scan_batch]
            scan_hi = mem_cu[scan_batch + 1]
            if tla.const_expr(q_block == 128):
                # 两 token 一 task：按每个请求的 pair 数扫描。
                pair_n = (scan_hi - scan_lo + c1) // c2
                if task_row >= pair_acc:
                    if task_row < pair_acc + pair_n:
                        req = scan_batch
                        row_a = scan_lo + (task_row - pair_acc) * c2
                pair_acc = pair_acc + pair_n
            else:
                if task_row >= scan_lo:
                    if task_row < scan_hi:
                        req = scan_batch

        q_end = mem_cu[req + 1]
        kv_len = mem_kvlen[req]

        # ---- 本 task 覆盖的 token 数与 M 行数 ----
        tokens_here = tla.as_numeric(c1)
        if tla.const_expr(q_block == 128):
            tokens_here = q_end - row_a
            if tokens_here > c2:
                tokens_here = tla.as_numeric(c2)
        q_tile_size = heads
        if tla.const_expr(q_block == 128):
            q_tile_size = tokens_here * heads
        else:
            q_tile_size = heads - hg * q_block
            if q_tile_size > q_block:
                q_tile_size = tla.as_numeric(q_block)

        # ---- 每个 AIV 子核负责的行（pairing 时一个子核一个 token）----
        row_here = row_a
        rows_here = q_tile_size - vec_idx * q_sub
        if rows_here > q_sub:
            rows_here = tla.as_numeric(q_sub)
        if rows_here < c0:
            rows_here = tla.as_numeric(0)
        if tla.const_expr(q_block == 128):
            row_here = row_a + vec_idx
            rows_here = c0
            if row_here < q_end:
                rows_here = tla.as_numeric(q_sub)

        # ---- 可见列：AIC 用最晚一行（决定 tile 数），AIV 用自己的行 ----
        row_last = row_a
        if tla.const_expr(q_block == 128):
            row_last = row_a + tokens_here - c1
        allowed = kv_len
        if mask_mode == 3:
            # Number of visible KV columns for this query token: j <= kv_len - q_len + local_i.
            allowed = kv_len - (q_end - row_last) + c1
            if allowed > kv_len:
                allowed = kv_len
            if allowed < c0:
                allowed = tla.as_numeric(0)
        allowed_vec = kv_len
        if mask_mode == 3:
            allowed_vec = kv_len - (q_end - row_here) + c1
            if allowed_vec > kv_len:
                allowed_vec = kv_len
            if allowed_vec < c0:
                allowed_vec = tla.as_numeric(0)
        num_tiles = (kv_len + kv_block - c1) // kv_block
        if mask_mode == 3:
            # tile_base >= allowed 的 tile 全列被 mask，直接不进循环。
            num_tiles = (allowed + kv_block - c1) // kv_block
        # V 分半只对多 tile task 启用：单 tile task（ctx<=128）没有跨 tile
        # 重叠窗口，额外的 DMA 描述符与 mutex 交接是纯开销（R4 实测 005/047
        # +2.1%/+3.2% 回退），整页一次搬完反而更快。
        # KV-split：本 task 只处理全局 tile 序列的第 chunk 段。
        chunk_start = tla.as_numeric(0)
        if tla.const_expr(kvsplit > c1):
            tiles_pc = (num_tiles + kvsplit - c1) // kvsplit
            chunk_start = chunk * tiles_pc
            num_tiles = num_tiles - chunk_start
            if num_tiles > tiles_pc:
                num_tiles = tiles_pc
            if num_tiles < c0:
                num_tiles = tla.as_numeric(0)
        split_v = num_tiles > c1

        # ---- 跨 task K 预取：下一个 task 的首 tile 页 + 本 task 是否要自己搬 ----
        # PA_BBND 的 K 要按列主序转置进 nZ，DMA 明显更贵，每个 task 开头的
        # K(tile 0) 因此和 QK(0) 串在关键路径上。在本 task 最后一轮（QK 已全部
        # 读完、没人再读 l1_k）提前搬下一个 task 的 K(0)，这段 DMA 就能盖在
        # 最后一个 tile 的 PV 上。PA_NZ 是整页扁平搬运，本来就已被盖住，不启用。
        # self_k=1 表示上一个 task 没替本 task 预取，prologue 仍需自己搬。
        self_k = tla.as_numeric(0)
        nx_page = tla.as_numeric(c0)
        nx_tiles = tla.as_numeric(c0)
        if tla.const_expr(q_block == 96):
            if nz_layout == 0:
                # 只有在「一核 >= 2 个 task」时跨 task 预取才有意义（否则每核只有一个
                # task，预取纯属多余，且小 shape 会白付扫描成本）。
                if num_tasks <= tla.arch.block_num():
                    self_k = tla.as_numeric(0)
                # 循环内的跨 task 预取带 kvsplit == 1 守卫，KV 分片路径没有预取，
                # 因此它必须自己搬 K(tile 0)，不能沿用上一个 task 的结论。
                if tla.const_expr(kvsplit == 1):
                    if task >= tla.arch.block_idx() + c1:
                        self_k = tla.as_numeric(1)
                nx_task = task + tla.arch.block_num()
                if nx_task < num_tasks:
                    nx_req = tla.as_numeric(c0)
                    for scan_next in tla.range(batch):
                        if nx_task >= mem_cu[scan_next]:
                            if nx_task < mem_cu[scan_next + 1]:
                                nx_req = scan_next
                    nx_kv_len = mem_kvlen[nx_req]
                    nx_allowed = nx_kv_len
                    if mask_mode == 3:
                        nx_allowed = nx_kv_len - (mem_cu[nx_req + 1] - nx_task) + c1
                        if nx_allowed > nx_kv_len:
                            nx_allowed = nx_kv_len
                        if nx_allowed < c0:
                            nx_allowed = tla.as_numeric(0)
                    nx_tiles = (nx_kv_len + kv_block - c1) // kv_block
                    if mask_mode == 3:
                        nx_tiles = (nx_allowed + kv_block - c1) // kv_block
                    if nx_tiles > c0:
                        nx_page = mem_bt[
                            nx_req * num_pages + tla.arch.block_idx() % nx_tiles
                        ]

        # ------------------------------------------------- 每个 task 的 UB 初值
        # ub_o 不再每 task 清零：每个 task 的第一个「可见」tile 进来时 running
        # max 仍是 floor(max_floor=-1e30)，重缩放因子 dm = exp(floor - tile_max)
        # 在 fp32 下下溢为精确 0，于是 acc_new = acc_old*dm + otmp 里的
        # acc_old（上一 task 的残留）被乘 0 覆盖；单 tile task 的收尾 epilogue
        # 用的也是同一个 dm。因此 48x512 f32（98 KiB、384 条向量 store）的
        # 零初始化是纯冗余，删掉它直接缩短 task 边界的 AIV 串行尾部。
        with tla.vector():
            ub_now_max_i8 = tla.make_tensor(
                ub_now_max_ptr,
                tla.make_layout(tla.make_shape(q_sub), tla.make_stride(c1)),
            )
            ub_sum_i8 = tla.make_tensor(
                ub_sum_ptr,
                tla.make_layout(tla.make_shape(q_sub), tla.make_stride(c1)),
            )
            with tla.vec.func(mode="simd"):
                preg_full = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                zero8 = tla.full(0.0, dtype=tla.Float32)
                floor8 = tla.full(max_floor, dtype=tla.Float32)
                ub_now_max_i8.store(floor8, mask=preg_full)
                ub_sum_i8.store(zero8, mask=preg_full)

        # ---------------------------------------------------------- task 级视图
        l1_q = tla.make_tensor(
            l1_q_ptr,
            tla.make_layout_with_tag((q_block, d_qk), io_dtype, tla.arch.zN),
        )
        l1_k = tla.make_tensor(
            l1_k_ptr,
            tla.make_layout_with_tag((d_qk, kv_block), io_dtype, tla.arch.nZ),
        )
        # K 的维度对半视图：nZ 的 16 维分组物理连续，前/后段各自是一段
        # 合法的 nZ，供 DMA 分半与细粒度 mutex 交接。k_split 与 chunk 对齐。
        l1_k_lo = tla.make_tensor(
            l1_k_ptr,
            tla.make_layout_with_tag((k_split, kv_block), io_dtype, tla.arch.nZ),
        )
        l1_k_hi = tla.make_tensor(
            l1_k_ptr + k_split * kv_block,
            tla.make_layout_with_tag((d_qk - k_split, kv_block), io_dtype, tla.arch.nZ),
        )
        l1_v = tla.make_tensor(
            l1_v_ptr,
            tla.make_layout_with_tag((kv_block, d_v), io_dtype, tla.arch.zN),
        )
        # V 的列对半视图：zN 的 16 列分组物理连续，cols 0~255 / 256~511 各自
        # 是一段合法的 (kv_block, 256) zN，供 gather 分半与细粒度 mutex 交接。
        l1_v_lo = tla.make_tensor(
            l1_v_ptr,
            tla.make_layout_with_tag((kv_block, d_v // 2), io_dtype, tla.arch.zN),
        )
        l1_v_hi = tla.make_tensor(
            l1_v_ptr + (d_v // 2) * kv_block,
            tla.make_layout_with_tag((kv_block, d_v // 2), io_dtype, tla.arch.zN),
        )
        l1_p = tla.make_tensor(
            l1_p_ptr,
            tla.make_layout_with_tag((q_block, kv_block), io_dtype, tla.arch.zN),
        )
        ub_s = tla.make_tensor(
            ub_s_ptr,
            tla.make_layout(
                tla.make_shape(rows_here, kv_block),
                tla.make_stride(kv_block, c1),
            ),
        )
        ub_now_max = tla.make_tensor(
            ub_now_max_ptr,
            tla.make_layout(tla.make_shape(q_sub), tla.make_stride(c1)),
        )
        ub_last_max = tla.make_tensor(
            ub_last_max_ptr,
            tla.make_layout(tla.make_shape(q_sub), tla.make_stride(c1)),
        )
        ub_sum = tla.make_tensor(
            ub_sum_ptr,
            tla.make_layout(tla.make_shape(q_sub), tla.make_stride(c1)),
        )
        ub_dm = tla.make_tensor(
            ub_dm_ptr,
            tla.make_layout(tla.make_shape(q_sub), tla.make_stride(c1)),
        )
        ub_blk_sum = tla.make_tensor(
            ub_blk_sum_ptr,
            tla.make_layout(tla.make_shape(q_sub), tla.make_stride(c1)),
        )
        ub_p_row = tla.make_tensor(
            ub_p_ptr,
            tla.make_layout(
                tla.make_shape(q_sub + c1, kv_block),
                tla.make_stride(kv_block, c1),
            ),
        )
        ub_p_zN_full = tla.make_tensor_like(ub_p_ptr, ub_p_row, tla.arch.zNUnAlign)
        ub_p_zN = tla.tile_view(
            ub_p_zN_full,
            tla.make_shape(q_sub, kv_block),
            tla.make_coord(c0, c0),
        )
        ub_p_copy = tla.tile_view(
            ub_p_zN_full,
            tla.make_shape(q_sub, kv_block),
            tla.make_coord(c0, c0),
        )
        l1_p_sub = tla.tile_view(
            l1_p,
            tla.make_shape(q_sub, kv_block),
            tla.make_coord(vec_idx, c0),
        )
        ub_o_re = tla.make_tensor(
            ub_o_ptr,
            tla.make_layout(
                tla.make_shape(rows_here, d_v),
                tla.make_stride(d_v, c1),
            ),
        )
        ub_otmp_re = tla.make_tensor(
            ub_otmp_ptr,
            tla.make_layout(
                tla.make_shape(rows_here, pv_width),
                tla.make_stride(pv_width, c1),
            ),
        )
        ub_otmp_re1 = tla.make_tensor(
            ub_otmp_ptr + otmp_stride,
            tla.make_layout(
                tla.make_shape(rows_here, pv_width),
                tla.make_stride(pv_width, c1),
            ),
        )
        otmp_slots = (ub_otmp_re,) if otmp_n == c1 else (ub_otmp_re, ub_otmp_re1)
        if tla.const_expr(otmp_n > c2):
            otmp_slots = otmp_slots + (
                tla.make_tensor(
                    ub_otmp_ptr + 2 * otmp_stride,
                    tla.make_layout(
                        tla.make_shape(rows_here, pv_width),
                        tla.make_stride(pv_width, c1),
                    ),
                ),
                tla.make_tensor(
                    ub_otmp_ptr + 3 * otmp_stride,
                    tla.make_layout(
                        tla.make_shape(rows_here, pv_width),
                        tla.make_stride(pv_width, c1),
                    ),
                ),
            )

        # Q 常驻 L1（zN）
        with tla.cube():
            q_gm = tla.make_tensor(
                mem_q.ptr + (row_a * heads + head_base + hg * q_block) * d_qk,
                tla.make_layout(
                    tla.make_shape(q_tile_size, d_qk),
                    tla.make_stride(d_qk, c1),
                ),
            )
            l1_q_valid = tla.tile_view(
                l1_q,
                tla.make_shape(q_tile_size, d_qk),
                tla.make_coord(c0, c0),
            )
            mtx_q.lock(pipe=tla.arch.MTE2)
            tla.copy(l1_q_valid, q_gm)
            mtx_q.unlock(pipe=tla.arch.MTE2)

        # S 的 L0C/UB 视图在 prologue 与循环体间复用。
        l0c_s_tile = tla.make_tensor(
            l0c_s_ptr,
            tla.make_layout_with_tag(
                (q_block, kv_block), tla.Float32, tla.arch.L0Clayout
            ),
        )
        ub_s_fix_tile = tla.make_tensor(
            ub_s_ptr,
            tla.make_layout(
                tla.make_shape(q_block, kv_block),
                tla.make_stride(kv_block, c1),
                layoutTag=tla.arch.RowMajor,
            ),
        )

        # 每核错开 tile 起点：单请求大 shape 里 32 个核本来会同时读同一个物理页，
        # 把 L2 打成热点；tile 顺序对 online softmax 无关，错开后各核读不同页。
        # 只在 N=96（一个 token 一 task）路径启用：paired 路径的 task 覆盖两个
        # token，页面节奏本就不同，实测收益在噪声内（011 +2%）。
        # tile 顺序错开：单请求大 ctx 时全部核在同一 wave 读同一批物理页，会把
        # L2 打成热点。online softmax 与 O 累加对 tile 顺序无关，故按核环形错开。
        # q96（单 token/task）与 paired（两 token/task）都适用。
        stagger = tla.as_numeric(0)
        if num_tiles > c0:
            stagger = tla.arch.block_idx() % num_tiles

        # ------------------------------------------------- QK 软件流水 prologue
        # 循环体提前一个 tile 做 gather+QK：AIV 处理 tile t 的 softmax 时，
        # Cube 已经在算 QK(t+1)，避免 Cube 在 softmax 期间空转。K 仍是单缓冲，
        # QK(t) 的 L1->L0 搬运完成后才允许覆盖。
        with tla.cube():
            if num_tiles > c0:
                phys0 = mem_bt[req * num_pages + chunk_start + stagger]
                page_base0 = phys0 * kv_blk_stride
                if self_k == c0:
                    mtx_k.lock(pipe=tla.arch.MTE2)
                    if nz_layout == 0:
                        k_src0 = tla.make_tensor(
                            mem_k.ptr + page_base0,
                            tla.make_layout(
                                tla.make_shape(k_split, kv_block),
                                tla.make_stride(c1, d_qk),
                                layoutTag=tla.arch.ColumnMajor,
                            ),
                        )
                        tla.copy(l1_k_lo, k_src0)
                    else:
                        k_words0 = (k_split * kv_block) // 2
                        k_src_ptr0 = tla.recast_ptr(
                            mem_k.ptr + page_base0, dtype=tla.Int32
                        )
                        k_dst_ptr0 = tla.recast_ptr(l1_k_ptr, dtype=tla.Int32)
                        k_flat_layout0 = tla.make_layout(
                            tla.make_shape(k_words0),
                            tla.make_stride(c1),
                            layoutTag=tla.arch.VectorLayout,
                        )
                        tla.copy(
                            tla.make_tensor(k_dst_ptr0, k_flat_layout0),
                            tla.make_tensor(k_src_ptr0, k_flat_layout0),
                        )
                    mtx_k.unlock(pipe=tla.arch.MTE2)
                    mtx_kh.lock(pipe=tla.arch.MTE2)
                    if nz_layout == 0:
                        k_src0h = tla.make_tensor(
                            mem_k.ptr + page_base0 + k_split,
                            tla.make_layout(
                                tla.make_shape(d_qk - k_split, kv_block),
                                tla.make_stride(c1, d_qk),
                                layoutTag=tla.arch.ColumnMajor,
                            ),
                        )
                        tla.copy(l1_k_hi, k_src0h)
                    else:
                        k_words0h = ((d_qk - k_split) * kv_block) // 2
                        k_src_ptr0h = tla.recast_ptr(
                            mem_k.ptr + page_base0 + k_split * kv_block,
                            dtype=tla.Int32,
                        )
                        k_dst_ptr0h = tla.recast_ptr(
                            l1_k_ptr + k_split * kv_block, dtype=tla.Int32
                        )
                        k_flat_layout0h = tla.make_layout(
                            tla.make_shape(k_words0h),
                            tla.make_stride(c1),
                            layoutTag=tla.arch.VectorLayout,
                        )
                        tla.copy(
                            tla.make_tensor(k_dst_ptr0h, k_flat_layout0h),
                            tla.make_tensor(k_src_ptr0h, k_flat_layout0h),
                        )
                    mtx_kh.unlock(pipe=tla.arch.MTE2)

                l0c_s0 = l0c_s_tile
                mtx_l0c_s.lock(pipe=tla.arch.CUBE)
                mtx_q.lock(pipe=tla.arch.MTE1)
                mtx_k.lock(pipe=tla.arch.MTE1)
                for ci in tla.range_constexpr(qk_full_chunks):
                    if ci == qk_half_chunks:
                        mtx_k.unlock(pipe=tla.arch.MTE1)
                        mtx_kh.lock(pipe=tla.arch.MTE1)
                    l1_q_c = tla.tile_view(
                        l1_q,
                        tla.make_shape(q_block, qk_chunk),
                        tla.make_coord(c0, ci),
                    )
                    l0_q = tla.make_tensor_like(l0a_q_ptrs[ci % 2], l1_q_c)
                    l1_k_c = tla.tile_view(
                        l1_k,
                        tla.make_shape(qk_chunk, kv_block),
                        tla.make_coord(ci, c0),
                    )
                    l0_k = tla.make_tensor_like(l0b_ptrs[ci % 2], l1_k_c)
                    mtx_l0s[ci % 2].lock(pipe=tla.arch.MTE1)
                    tla.copy(l0_q, l1_q_c)
                    tla.copy(l0_k, l1_k_c)
                    mtx_l0s[ci % 2].unlock(pipe=tla.arch.MTE1)
                    mtx_l0s[ci % 2].lock(pipe=tla.arch.CUBE)
                    tla.mmad(l0c_s0, l0_q, l0_k, init_c=(ci == 0))
                    mtx_l0s[ci % 2].unlock(pipe=tla.arch.CUBE)
                mtx_kh.unlock(pipe=tla.arch.MTE1)
                mtx_q.unlock(pipe=tla.arch.MTE1)
                mtx_l0c_s.unlock(pipe=tla.arch.CUBE)

                mtx_l0c_s.lock(pipe=tla.arch.FIX)
                tla.cross_core_wait_flag(s_free, tla.arch.FIX)
                tla.copy(
                    ub_s_fix_tile,
                    l0c_s_tile,
                    tla.params.CopyL0C2DstParams(
                        l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M
                    ),
                )
                mtx_l0c_s.unlock(pipe=tla.arch.FIX)
                tla.cross_core_set_flag(s_ready, tla.arch.FIX)

        for it in tla.range(num_tiles):
            tile = chunk_start + (it + stagger) % num_tiles
            tile_base = tile * kv_block
            kv_tile_size = kv_len - tile_base
            if kv_tile_size > kv_block:
                kv_tile_size = tla.as_numeric(kv_block)

            col_limit = kv_tile_size
            col_limit_vec = kv_tile_size
            if mask_mode == 3:
                causal_limit = allowed - tile_base
                if causal_limit < c0:
                    causal_limit = tla.as_numeric(0)
                if causal_limit < col_limit:
                    col_limit = causal_limit
                causal_vec = allowed_vec - tile_base
                if causal_vec < c0:
                    causal_vec = tla.as_numeric(0)
                if causal_vec < col_limit_vec:
                    col_limit_vec = causal_vec

            if col_limit > c0:
                if tla.const_expr(q_block == 96 and kvsplit == 1):
                    if nz_layout == 0:
                        with tla.cube():
                            # ---- 跨 task 预取下一个 task 的 K(tile 0) ----
                            # 整页写需同时持有低/高半页 mutex（QK 分半消费）。
                            if it + c1 == num_tiles:
                                if nx_tiles > c0:
                                    page_base_n = nx_page * kv_blk_stride
                                    mtx_k.lock(pipe=tla.arch.MTE2)
                                    mtx_kh.lock(pipe=tla.arch.MTE2)
                                    k_src_n = tla.make_tensor(
                                        mem_k.ptr + page_base_n,
                                        tla.make_layout(
                                            tla.make_shape(d_qk, kv_block),
                                            tla.make_stride(c1, d_qk),
                                            layoutTag=tla.arch.ColumnMajor,
                                        ),
                                    )
                                    tla.copy(l1_k, k_src_n)
                                    mtx_kh.unlock(pipe=tla.arch.MTE2)
                                    mtx_k.unlock(pipe=tla.arch.MTE2)
                with tla.cube():
                    # ---- 预取 K(tile+1) 并计算 QK(tile+1) ----
                    # tile 的 QK 已在上一次迭代（或 prologue）算完，这里把下一
                    # 个 tile 的 QK 填进 AIV 处理 softmax 的空档。
                    if it + c1 < num_tiles:
                        phys = mem_bt[
                            req * num_pages
                            + chunk_start
                            + (it + stagger + c1) % num_tiles
                        ]
                        page_base = phys * kv_blk_stride
                        # K 分半搬运：低半（dims 0~287）先行，QK 的前 3 个
                        # chunk 只等它；高半的 MTE2 与前三 chunk 的 MTE1/MMAD
                        # 重叠，消掉整页 DMA 的串行等待。
                        mtx_k.lock(pipe=tla.arch.MTE2)
                        if nz_layout == 0:
                            # PA_BBND 页内是 [token, 576]；按 (d, token) 列主序读入 nZ。
                            k_src = tla.make_tensor(
                                mem_k.ptr + page_base,
                                tla.make_layout(
                                    tla.make_shape(k_split, kv_block),
                                    tla.make_stride(c1, d_qk),
                                    layoutTag=tla.arch.ColumnMajor,
                                ),
                            )
                            tla.copy(l1_k_lo, k_src)
                        else:
                            # PA_NZ 页内物理布局恰好就是 (d_qk, kv_block) 的 nZ：
                            # 以 u64 扁平搬运半页，避免 18 次小组 DMA。
                            k_words = (k_split * kv_block) // 2
                            k_src_ptr = tla.recast_ptr(
                                mem_k.ptr + page_base, dtype=tla.Int32
                            )
                            k_dst_ptr = tla.recast_ptr(l1_k_ptr, dtype=tla.Int32)
                            k_flat_layout = tla.make_layout(
                                tla.make_shape(k_words),
                                tla.make_stride(c1),
                                layoutTag=tla.arch.VectorLayout,
                            )
                            k_src = tla.make_tensor(k_src_ptr, k_flat_layout)
                            k_dst = tla.make_tensor(k_dst_ptr, k_flat_layout)
                            tla.copy(k_dst, k_src)
                        mtx_k.unlock(pipe=tla.arch.MTE2)
                        mtx_kh.lock(pipe=tla.arch.MTE2)
                        if nz_layout == 0:
                            k_srch = tla.make_tensor(
                                mem_k.ptr + page_base + k_split,
                                tla.make_layout(
                                    tla.make_shape(d_qk - k_split, kv_block),
                                    tla.make_stride(c1, d_qk),
                                    layoutTag=tla.arch.ColumnMajor,
                                ),
                            )
                            tla.copy(l1_k_hi, k_srch)
                        else:
                            k_wordsh = ((d_qk - k_split) * kv_block) // 2
                            k_src_ptrh = tla.recast_ptr(
                                mem_k.ptr + page_base + k_split * kv_block,
                                dtype=tla.Int32,
                            )
                            k_dst_ptrh = tla.recast_ptr(
                                l1_k_ptr + k_split * kv_block,
                                dtype=tla.Int32,
                            )
                            k_flat_layouth = tla.make_layout(
                                tla.make_shape(k_wordsh),
                                tla.make_stride(c1),
                                layoutTag=tla.arch.VectorLayout,
                            )
                            k_srch = tla.make_tensor(k_src_ptrh, k_flat_layouth)
                            k_dsth = tla.make_tensor(k_dst_ptrh, k_flat_layouth)
                            tla.copy(k_dsth, k_srch)
                        mtx_kh.unlock(pipe=tla.arch.MTE2)

                        # ---- QK：576 维拆成 6 x 96，L0A/L0B 双槽 ping-pong ----
                        mtx_l0c_s.lock(pipe=tla.arch.CUBE)
                        mtx_q.lock(pipe=tla.arch.MTE1)
                        mtx_k.lock(pipe=tla.arch.MTE1)
                        for ci in tla.range_constexpr(qk_full_chunks):
                            if ci == qk_half_chunks:
                                mtx_k.unlock(pipe=tla.arch.MTE1)
                                mtx_kh.lock(pipe=tla.arch.MTE1)
                            l1_q_c = tla.tile_view(
                                l1_q,
                                tla.make_shape(q_block, qk_chunk),
                                tla.make_coord(c0, ci),
                            )
                            l0_q = tla.make_tensor_like(l0a_q_ptrs[ci % 2], l1_q_c)
                            l1_k_c = tla.tile_view(
                                l1_k,
                                tla.make_shape(qk_chunk, kv_block),
                                tla.make_coord(ci, c0),
                            )
                            l0_k = tla.make_tensor_like(l0b_ptrs[ci % 2], l1_k_c)
                            mtx_l0s[ci % 2].lock(pipe=tla.arch.MTE1)
                            tla.copy(l0_q, l1_q_c)
                            tla.copy(l0_k, l1_k_c)
                            mtx_l0s[ci % 2].unlock(pipe=tla.arch.MTE1)
                            mtx_l0s[ci % 2].lock(pipe=tla.arch.CUBE)
                            tla.mmad(l0c_s_tile, l0_q, l0_k, init_c=(ci == 0))
                            mtx_l0s[ci % 2].unlock(pipe=tla.arch.CUBE)

                        mtx_kh.unlock(pipe=tla.arch.MTE1)
                        mtx_q.unlock(pipe=tla.arch.MTE1)
                        mtx_l0c_s.unlock(pipe=tla.arch.CUBE)

                with tla.cube():
                    # ---- 整页 gather V（cache 前 512 维），列对半分两次 ----
                    # 低半（cols 0~255，PV half 0/1）先行；高半的 MTE2 与前两个
                    # half 的 MTE1/MMAD 重叠，消掉 128KB 整页等待。
                    phys_v = mem_bt[req * num_pages + tile]
                    page_base_v = phys_v * kv_blk_stride
                    if split_v:
                        mtx_v.lock(pipe=tla.arch.MTE2)
                        if nz_layout == 0:
                            v_src = tla.make_tensor(
                                mem_k.ptr + page_base_v,
                                tla.make_layout(
                                    tla.make_shape(kv_block, d_v // 2),
                                    tla.make_stride(d_qk, c1),
                                    layoutTag=tla.arch.RowMajor,
                                ),
                            )
                            tla.copy(l1_v_lo, v_src)
                        else:
                            # V 是页内前 32 个 16 特征组，物理上连续；l1_v 的 zN
                            # 物理布局与它一致，同样用扁平搬运（拆成两个 64KB）。
                            v_words = (d_v * kv_block) // 4
                            v_src_ptr = tla.recast_ptr(
                                mem_k.ptr + page_base_v, dtype=tla.Int32
                            )
                            v_dst_ptr = tla.recast_ptr(l1_v_ptr, dtype=tla.Int32)
                            v_flat_layout = tla.make_layout(
                                tla.make_shape(v_words),
                                tla.make_stride(c1),
                                layoutTag=tla.arch.VectorLayout,
                            )
                            v_src = tla.make_tensor(v_src_ptr, v_flat_layout)
                            v_dst = tla.make_tensor(v_dst_ptr, v_flat_layout)
                            tla.copy(v_dst, v_src)
                        mtx_v.unlock(pipe=tla.arch.MTE2)
                        mtx_vh.lock(pipe=tla.arch.MTE2)
                        if nz_layout == 0:
                            v_srch = tla.make_tensor(
                                mem_k.ptr + page_base_v + d_v // 2,
                                tla.make_layout(
                                    tla.make_shape(kv_block, d_v // 2),
                                    tla.make_stride(d_qk, c1),
                                    layoutTag=tla.arch.RowMajor,
                                ),
                            )
                            tla.copy(l1_v_hi, v_srch)
                        else:
                            v_wordsh = (d_v * kv_block) // 4
                            v_src_ptrh = tla.recast_ptr(
                                mem_k.ptr + page_base_v + (d_v // 2) * kv_block,
                                dtype=tla.Int32,
                            )
                            v_dst_ptrh = tla.recast_ptr(
                                l1_v_ptr + (d_v // 2) * kv_block, dtype=tla.Int32
                            )
                            v_flat_layouth = tla.make_layout(
                                tla.make_shape(v_wordsh),
                                tla.make_stride(c1),
                                layoutTag=tla.arch.VectorLayout,
                            )
                            v_srch = tla.make_tensor(v_src_ptrh, v_flat_layouth)
                            v_dsth = tla.make_tensor(v_dst_ptrh, v_flat_layouth)
                            tla.copy(v_dsth, v_srch)
                        mtx_vh.unlock(pipe=tla.arch.MTE2)
                    else:
                        # 单 tile task：无重叠窗口，整页一次搬完（与 R3 相同）。
                        mtx_v.lock(pipe=tla.arch.MTE2)
                        if nz_layout == 0:
                            v_src = tla.make_tensor(
                                mem_k.ptr + page_base_v,
                                tla.make_layout(
                                    tla.make_shape(kv_block, d_v),
                                    tla.make_stride(d_qk, c1),
                                    layoutTag=tla.arch.RowMajor,
                                ),
                            )
                            tla.copy(l1_v, v_src)
                        else:
                            v_words = (d_v * kv_block) // 2
                            v_src_ptr = tla.recast_ptr(
                                mem_k.ptr + page_base_v, dtype=tla.Int32
                            )
                            v_dst_ptr = tla.recast_ptr(l1_v_ptr, dtype=tla.Int32)
                            v_flat_layout = tla.make_layout(
                                tla.make_shape(v_words),
                                tla.make_stride(c1),
                                layoutTag=tla.arch.VectorLayout,
                            )
                            v_src = tla.make_tensor(v_src_ptr, v_flat_layout)
                            v_dst = tla.make_tensor(v_dst_ptr, v_flat_layout)
                            tla.copy(v_dst, v_src)
                        mtx_v.unlock(pipe=tla.arch.MTE2)

            with tla.vector():
                if col_limit > c0:
                    tla.cross_core_wait_flag(s_ready, tla.arch.VECTOR)

                    n0 = col_limit_vec
                    if n0 > vl_f32:
                        n0 = tla.as_numeric(vl_f32)
                    n1 = col_limit_vec - vl_f32
                    if n1 < c0:
                        n1 = tla.as_numeric(0)
                    if n1 > vl_f32:
                        n1 = tla.as_numeric(vl_f32)

                    # ---- pass1: scale + 有效列掩码 + 行 max ----
                    with tla.vec.func(mode="simd"):
                        preg_full = tla.create_mask(
                            pattern=tla.mask.ALL, dtype=tla.Float32
                        )
                        one_mask = tla.update_mask(1, dtype=tla.Float32)[0]
                        mask_n0 = tla.update_mask(n0, dtype=tla.Float32)[0]
                        mask_n1 = tla.update_mask(n1, dtype=tla.Float32)[0]
                        min_reg = tla.full(min_value, dtype=tla.Float32)
                        for i0 in tla.range(rows_here):
                            ub_s_i0 = tla.tile_view(
                                ub_s,
                                tla.make_shape(c1, vl_f32),
                                tla.make_coord(i0, c0),
                            )
                            ub_s_i1 = tla.tile_view(
                                ub_s,
                                tla.make_shape(c1, vl_f32),
                                tla.make_coord(i0, c1),
                            )
                            s_reg0 = ub_s_i0.load()
                            s_reg1 = ub_s_i1.load()
                            s_reg0 = tla.mul(s_reg0, softmax_scale, mask=preg_full)
                            s_reg1 = tla.mul(s_reg1, softmax_scale, mask=preg_full)
                            s_reg0 = tla.where(mask_n0, s_reg0, min_reg)
                            s_reg1 = tla.where(mask_n1, s_reg1, min_reg)
                            ub_s_i0.store(s_reg0, mask=preg_full)
                            ub_s_i1.store(s_reg1, mask=preg_full)
                            tile_max_reg = tla.max(s_reg0, s_reg1, mask=preg_full)
                            row_max_reg = tile_max_reg.reduce(
                                tla.ReductionOp.MAX, mask=preg_full
                            )
                            ub_last_max_i = tla.tile_view(
                                ub_last_max, tla.make_shape(1), tla.make_coord(i0)
                            )
                            ub_last_max_i.store(
                                row_max_reg,
                                params=tla.params.UnalignStoreParams(),
                                mask=one_mask,
                            )
                        tla.local_mem_bar(
                            tla.params.MemType.VEC_STORE,
                            tla.params.MemType.VEC_LOAD,
                        )

                    # ---- 合并 running max / dm / running sum ----
                    with tla.vec.func(mode="simd"):
                        preg_full2 = tla.create_mask(
                            pattern=tla.mask.ALL, dtype=tla.Float32
                        )
                        ub_now_max_v = tla.make_tensor(
                            ub_now_max_ptr,
                            tla.make_layout(
                                tla.make_shape(vl_f32), tla.make_stride(c1)
                            ),
                        )
                        ub_last_max_v = tla.make_tensor(
                            ub_last_max_ptr,
                            tla.make_layout(
                                tla.make_shape(vl_f32), tla.make_stride(c1)
                            ),
                        )
                        ub_sum_v = tla.make_tensor(
                            ub_sum_ptr,
                            tla.make_layout(
                                tla.make_shape(vl_f32), tla.make_stride(c1)
                            ),
                        )
                        ub_dm_v = tla.make_tensor(
                            ub_dm_ptr,
                            tla.make_layout(
                                tla.make_shape(vl_f32), tla.make_stride(c1)
                            ),
                        )
                        old_max_v = ub_now_max_v.load()
                        blk_max_v = ub_last_max_v.load()
                        glob_max_v = tla.max(old_max_v, blk_max_v, mask=preg_full2)
                        dm_v = tla.exp(
                            tla.sub(old_max_v, glob_max_v, mask=preg_full2),
                            mask=preg_full2,
                        )
                        sum_old_v = ub_sum_v.load()
                        upd_sum_v = tla.mul(sum_old_v, dm_v, mask=preg_full2)
                        ub_now_max_v.store(glob_max_v, mask=preg_full2)
                        # 延迟重缩放：保留 dm(tile-1) 供下一次迭代的尾部使用。
                        ub_dm_prev_v = tla.make_tensor(
                            ub_dm_prev_ptr,
                            tla.make_layout(
                                tla.make_shape(vl_f32), tla.make_stride(c1)
                            ),
                        )
                        ub_dm_prev_v.store(ub_dm_v.load(), mask=preg_full2)
                        ub_dm_v.store(dm_v, mask=preg_full2)
                        ub_upd_sum_v = tla.make_tensor(
                            ub_upd_sum_ptr,
                            tla.make_layout(
                                tla.make_shape(vl_f32), tla.make_stride(c1)
                            ),
                        )
                        ub_upd_sum_v.store(upd_sum_v, mask=preg_full2)
                        tla.local_mem_bar(
                            tla.params.MemType.VEC_STORE,
                            tla.params.MemType.VEC_LOAD,
                        )

                    # ---- pass2: exp + P(f16) 落盘 + 块内 sum ----
                    mtx_ub_p.lock(pipe=tla.arch.VECTOR)
                    with tla.vec.func(mode="simd"):
                        preg_full3 = tla.create_mask(
                            pattern=tla.mask.ALL, dtype=tla.Float32
                        )
                        preg_all_b16 = tla.create_mask(
                            pattern=tla.mask.ALL, dtype=io_dtype
                        )
                        one_mask3 = tla.update_mask(1, dtype=tla.Float32)[0]
                        for i2 in tla.range(rows_here):
                            ub_now_max_i = tla.tile_view(
                                ub_now_max, tla.make_shape(1), tla.make_coord(i2)
                            )
                            max_reg = ub_now_max_i.load(
                                params=tla.params.NormalLoadParams(
                                    load_dist=tla.params.LoadDist.DIST_BRC_B32
                                )
                            )
                            ub_s_row = tla.tile_view(
                                ub_s,
                                tla.make_shape(c1, 2 * vl_f32),
                                tla.make_coord(i2, c0),
                            )
                            s_odd, s_even = ub_s_row.load(
                                params=tla.params.NormalLoadParams(
                                    load_dist=tla.params.LoadDist.DIST_DINTLV_B32
                                )
                            )
                            exp_odd = tla.exp(
                                tla.sub(s_odd, max_reg, mask=preg_full3),
                                mask=preg_full3,
                            )
                            exp_even = tla.exp(
                                tla.sub(s_even, max_reg, mask=preg_full3),
                                mask=preg_full3,
                            )
                            sum_reg = tla.add(exp_odd, exp_even, mask=preg_full3)
                            block_sum = sum_reg.reduce(
                                tla.ReductionOp.ADD, mask=preg_full3
                            )
                            ub_blk_sum_i = tla.tile_view(
                                ub_blk_sum, tla.make_shape(1), tla.make_coord(i2)
                            )
                            ub_blk_sum_i.store(
                                block_sum,
                                params=tla.params.UnalignStoreParams(),
                                mask=one_mask3,
                            )
                            p_even = exp_even.to(io_dtype, cast_one, mask=preg_full3)
                            p_odd = exp_odd.to(io_dtype, cast_zero, mask=preg_full3)
                            p_zn = tla.bitwise_or(p_even, p_odd, mask=preg_all_b16)
                            ub_p_zn_i = tla.tile_view(
                                ub_p_zN,
                                tla.make_shape(c1, 2 * vl_f32),
                                tla.make_coord(i2, c0),
                            )
                            ub_p_zn_i.store(
                                p_zn,
                                params=tla.params.BlockStoreParams(
                                    block_stride=q_sub + c1
                                ),
                                mask=preg_all_b16,
                            )
                        tla.local_mem_bar(
                            tla.params.MemType.VEC_STORE,
                            tla.params.MemType.VEC_LOAD,
                        )
                    mtx_ub_p.unlock(pipe=tla.arch.VECTOR)

                    # ---- 合并 running sum ----
                    with tla.vec.func(mode="simd"):
                        preg_full4 = tla.create_mask(
                            pattern=tla.mask.ALL, dtype=tla.Float32
                        )
                        ub_blk_sum_v = tla.make_tensor(
                            ub_blk_sum_ptr,
                            tla.make_layout(
                                tla.make_shape(vl_f32), tla.make_stride(c1)
                            ),
                        )
                        ub_sum_v2 = tla.make_tensor(
                            ub_sum_ptr,
                            tla.make_layout(
                                tla.make_shape(vl_f32), tla.make_stride(c1)
                            ),
                        )
                        ub_upd_sum_v2 = tla.make_tensor(
                            ub_upd_sum_ptr,
                            tla.make_layout(
                                tla.make_shape(vl_f32), tla.make_stride(c1)
                            ),
                        )
                        blk_sum_v = ub_blk_sum_v.load()
                        new_sum_v = tla.add(
                            ub_upd_sum_v2.load(), blk_sum_v, mask=preg_full4
                        )
                        ub_sum_v2.store(new_sum_v, mask=preg_full4)

                    # S 已消费完，释放给下一个 tile 的 FIX
                    tla.cross_core_set_flag(s_free, tla.arch.VECTOR)

                    # ---- UB P -> L1 P (zN) ----
                    mtx_ub_p.lock(pipe=tla.arch.MTE3)
                    tla.copy(l1_p_sub, ub_p_copy)
                    mtx_ub_p.unlock(pipe=tla.arch.MTE3)
                    tla.cross_core_set_flag(p_ready, tla.arch.MTE3)

            with tla.cube():
                if col_limit > c0:
                    if it + c1 < num_tiles:
                        # FIX S(下一个迭代的 tile)：AIV 已在 softmax(tile) 中释放 UB S。
                        mtx_l0c_s.lock(pipe=tla.arch.FIX)
                        tla.cross_core_wait_flag(s_free, tla.arch.FIX)
                        tla.copy(
                            ub_s_fix_tile,
                            l0c_s_tile,
                            tla.params.CopyL0C2DstParams(
                                l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M
                            ),
                        )
                        mtx_l0c_s.unlock(pipe=tla.arch.FIX)
                        tla.cross_core_set_flag(s_ready, tla.arch.FIX)
                    tla.cross_core_wait_flag(p_ready, tla.arch.MTE1)
                    # 每个 half 独占 otmp 槽时，整 tile 只需一次 o_free 握手。
                    if tla.const_expr(otmp_n >= pv_splits):
                        tla.cross_core_wait_flag(o_free, tla.arch.FIX)
                    l0_p = tla.make_tensor_like(l0a_p_ptr, l1_p)
                    mtx_l0p.lock(pipe=tla.arch.MTE1)
                    tla.copy(l0_p, l1_p)
                    mtx_l0p.unlock(pipe=tla.arch.MTE1)
                    mtx_v.lock(pipe=tla.arch.MTE1)
                    for half in tla.range_constexpr(pv_splits):
                        if half == pv_splits // 2:
                            if split_v:
                                mtx_v.unlock(pipe=tla.arch.MTE1)
                                mtx_vh.lock(pipe=tla.arch.MTE1)
                        l1_v_half = tla.get_tile(
                            l1_v,
                            tla.make_coord(c0, half * pv_width),
                            tla.make_shape(kv_block, pv_width),
                        )
                        l0_v = tla.make_tensor_like(l0b_ptrs[half % 2], l1_v_half)
                        mtx_l0s[half % 2].lock(pipe=tla.arch.MTE1)
                        tla.copy(l0_v, l1_v_half)
                        mtx_l0s[half % 2].unlock(pipe=tla.arch.MTE1)
                        o_slot = half % c2
                        otmp_slot = half if otmp_n > c1 else c0
                        l0c_o = tla.make_tensor(
                            l0c_o_ptrs[o_slot],
                            tla.make_layout_with_tag(
                                (q_block, pv_width),
                                tla.Float32,
                                tla.arch.L0Clayout,
                            ),
                        )
                        mtx_l0c_o_cur = mtx_l0c_o_s[o_slot]
                        mtx_l0s[half % 2].lock(pipe=tla.arch.CUBE)
                        mtx_l0p.lock(pipe=tla.arch.CUBE)
                        mtx_l0c_o_cur.lock(pipe=tla.arch.CUBE)
                        tla.mmad(l0c_o, l0_p, l0_v, init_c=True)
                        mtx_l0c_o_cur.unlock(pipe=tla.arch.CUBE)
                        mtx_l0p.unlock(pipe=tla.arch.CUBE)
                        mtx_l0s[half % 2].unlock(pipe=tla.arch.CUBE)

                        mtx_l0c_o_cur.lock(pipe=tla.arch.FIX)
                        if tla.const_expr(otmp_n < pv_splits):
                            tla.cross_core_wait_flag(o_free, tla.arch.FIX)
                        ub_otmp_fix = tla.make_tensor(
                            ub_otmp_ptr
                            + (otmp_slot if otmp_n > c1 else c0) * otmp_stride,
                            tla.make_layout(
                                tla.make_shape(q_block, pv_width),
                                tla.make_stride(pv_width, c1),
                                layoutTag=tla.arch.RowMajor,
                            ),
                        )
                        tla.copy(
                            ub_otmp_fix,
                            l0c_o,
                            tla.params.CopyL0C2DstParams(
                                l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M
                            ),
                        )
                        mtx_l0c_o_cur.unlock(pipe=tla.arch.FIX)
                        if tla.const_expr(otmp_n < pv_splits):
                            tla.cross_core_set_flag(o_ready, tla.arch.FIX)
                    if split_v:
                        mtx_vh.unlock(pipe=tla.arch.MTE1)
                    else:
                        mtx_v.unlock(pipe=tla.arch.MTE1)
                    if tla.const_expr(otmp_n >= pv_splits):
                        tla.cross_core_set_flag(o_ready, tla.arch.FIX)

            with tla.vector():
                if col_limit > c0:
                    if it > c0:
                        # 延迟重缩放：用迭代序判断“第一个 tile”，与错开后的
                        # tile 号无关（tile 顺序对 online softmax 无影响）。
                        # tile-1 的 O 重缩放推迟到 softmax(tile) 之后：p_ready(tile)
                        # 不再排在 PV(tile-1) 之后，Cube 不会在 tile 边界空等。
                        ub_dm_prev_re = tla.make_tensor(
                            ub_dm_prev_ptr,
                            tla.make_layout(tla.make_shape(q_sub), tla.make_stride(c1)),
                        )
                        if tla.const_expr(otmp_n >= pv_splits):
                            tla.cross_core_wait_flag(o_ready, tla.arch.VECTOR)
                        for half in tla.range_constexpr(pv_splits):
                            if tla.const_expr(otmp_n < pv_splits):
                                tla.cross_core_wait_flag(o_ready, tla.arch.VECTOR)
                            ub_otmp_cur = otmp_slots[half % len(otmp_slots)]
                            with tla.vec.func(mode="simd"):
                                preg_full5 = tla.create_mask(
                                    pattern=tla.mask.ALL, dtype=tla.Float32
                                )
                                for i3 in tla.range(rows_here):
                                    ub_dm_i = tla.tile_view(
                                        ub_dm_prev_re,
                                        tla.make_shape(1),
                                        tla.make_coord(i3),
                                    )
                                    dm_reg = ub_dm_i.load(
                                        params=tla.params.NormalLoadParams(
                                            load_dist=tla.params.LoadDist.DIST_BRC_B32
                                        )
                                    )
                                    for j3 in tla.range(pv_chunks):
                                        acc_tile = tla.tile_view(
                                            ub_o_re,
                                            tla.make_shape(c1, vl_f32),
                                            tla.make_coord(i3, half * pv_chunks + j3),
                                        )
                                        otmp_tile = tla.tile_view(
                                            ub_otmp_cur,
                                            tla.make_shape(c1, vl_f32),
                                            tla.make_coord(i3, j3),
                                        )
                                        acc_reg = acc_tile.load()
                                        otmp_reg = otmp_tile.load()
                                        scaled = tla.mul(
                                            acc_reg, dm_reg, mask=preg_full5
                                        )
                                        merged = tla.add(
                                            scaled, otmp_reg, mask=preg_full5
                                        )
                                        acc_tile.store(merged, mask=preg_full5)
                            if tla.const_expr(otmp_n < pv_splits):
                                tla.cross_core_set_flag(o_free, tla.arch.VECTOR)
                        if tla.const_expr(otmp_n >= pv_splits):
                            tla.cross_core_set_flag(o_free, tla.arch.VECTOR)

        if tla.const_expr(q_block == 96 and kvsplit == 1):
            if nz_layout == 0:
                # 本 task 没有有效 tile 时循环体没执行，K 预取在循环外补一次，
                # 保证下一个 task 的 self_k 判定始终成立。整页写需同时持有
                # 低/高半页 mutex（QK 分半消费）。
                if num_tiles == c0:
                    if nx_tiles > c0:
                        with tla.cube():
                            page_base_z = nx_page * kv_blk_stride
                            mtx_k.lock(pipe=tla.arch.MTE2)
                            mtx_kh.lock(pipe=tla.arch.MTE2)
                            k_src_z = tla.make_tensor(
                                mem_k.ptr + page_base_z,
                                tla.make_layout(
                                    tla.make_shape(d_qk, kv_block),
                                    tla.make_stride(c1, d_qk),
                                    layoutTag=tla.arch.ColumnMajor,
                                ),
                            )
                            tla.copy(l1_k, k_src_z)
                            mtx_kh.unlock(pipe=tla.arch.MTE2)
                            mtx_k.unlock(pipe=tla.arch.MTE2)

        # ------------------------------------------------------ 输出 + LSE（每 task 一次）
        with tla.vector():
            ub_sum_out = tla.make_tensor(
                ub_sum_ptr,
                tla.make_layout(tla.make_shape(q_sub), tla.make_stride(c1)),
            )
            ub_now_max_out = tla.make_tensor(
                ub_now_max_ptr,
                tla.make_layout(tla.make_shape(q_sub), tla.make_stride(c1)),
            )
            ub_out_t = tla.make_tensor(
                ub_out_ptr,
                tla.make_layout(
                    tla.make_shape(rows_here, out_chunk),
                    tla.make_stride(out_chunk, c1),
                ),
            )
            ub_lse_t = tla.make_tensor(
                ub_lse_ptr,
                tla.make_layout(tla.make_shape(q_sub), tla.make_stride(c1)),
            )
            ub_dm_last = tla.make_tensor(
                ub_dm_ptr,
                tla.make_layout(tla.make_shape(q_sub), tla.make_stride(c1)),
            )
            ub_inv = tla.make_tensor(
                ub_inv_ptr,
                tla.make_layout(tla.make_shape(q_sub), tla.make_stride(c1)),
            )
            mtx_ub_lse.lock(pipe=tla.arch.VECTOR)
            with tla.vec.func(mode="simd"):
                preg_full6 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                preg_all_b16_6 = tla.create_mask(pattern=tla.mask.ALL, dtype=io_dtype)
                one_mask6 = tla.update_mask(1, dtype=tla.Float32)[0]
                one_reg = tla.full(1.0, dtype=tla.Float32)
                zero_reg6 = tla.full(0.0, dtype=tla.Float32)
                neg_inf_reg = tla.full(float("-inf"), dtype=tla.Float32)
                for i6 in tla.range(rows_here):
                    ub_sum_i6 = tla.tile_view(
                        ub_sum_out, tla.make_shape(1), tla.make_coord(i6)
                    )
                    sum_reg6 = ub_sum_i6.load(
                        params=tla.params.NormalLoadParams(
                            load_dist=tla.params.LoadDist.DIST_BRC_B32
                        )
                    )
                    pos_mask = tla.cmp(sum_reg6, 0.0, mode="gt", mask=preg_full6)
                    inv_raw = tla.div(one_reg, sum_reg6, mask=preg_full6)
                    inv_reg = tla.where(pos_mask, inv_raw, zero_reg6)
                    ub_inv_i6 = tla.tile_view(
                        ub_inv, tla.make_shape(1), tla.make_coord(i6)
                    )
                    ub_inv_i6.store(
                        inv_reg,
                        params=tla.params.UnalignStoreParams(),
                        mask=one_mask6,
                    )

                    ub_max_i6 = tla.tile_view(
                        ub_now_max_out, tla.make_shape(1), tla.make_coord(i6)
                    )
                    max_reg6 = ub_max_i6.load(
                        params=tla.params.NormalLoadParams(
                            load_dist=tla.params.LoadDist.DIST_BRC_B32
                        )
                    )
                    lse_raw = tla.add(
                        max_reg6,
                        tla.log(sum_reg6, mask=preg_full6),
                        mask=preg_full6,
                    )
                    lse_reg6 = tla.where(pos_mask, lse_raw, neg_inf_reg)
                    ub_lse_t_i6 = tla.tile_view(
                        ub_lse_t, tla.make_shape(1), tla.make_coord(i6)
                    )
                    ub_lse_t_i6.store(
                        lse_reg6,
                        params=tla.params.UnalignStoreParams(),
                        mask=one_mask6,
                    )

                tla.local_mem_bar(
                    tla.params.MemType.VEC_STORE,
                    tla.params.MemType.VEC_LOAD,
                )
            mtx_ub_lse.unlock(pipe=tla.arch.VECTOR)
            # 128 列一块输出：最后 tile 的 O 重缩放、归一化、cast 与 DMA 合成
            # 一个 pass，ub_o 在 task 收尾只读写一遍。
            if num_tiles > c0:
                if tla.const_expr(otmp_n >= pv_splits):
                    tla.cross_core_wait_flag(o_ready, tla.arch.VECTOR)
                for oc in tla.range_constexpr(out_chunks):
                    if tla.const_expr(otmp_n < pv_splits):
                        tla.cross_core_wait_flag(o_ready, tla.arch.VECTOR)
                    ub_otmp_ep = otmp_slots[oc % len(otmp_slots)]
                    o_gm = tla.make_tensor(
                        mem_o.ptr
                        + (head_base + head_off) * total_q * d_v
                        + row_here * d_v
                        + oc * out_chunk,
                        tla.make_layout(
                            tla.make_shape(rows_here, out_chunk),
                            tla.make_stride(total_q * d_v, c1),
                        ),
                    )
                    mtx_ub_out.lock(pipe=tla.arch.VECTOR)
                    with tla.vec.func(mode="simd"):
                        preg_full7 = tla.create_mask(
                            pattern=tla.mask.ALL, dtype=tla.Float32
                        )
                        preg_all_b16_7 = tla.create_mask(
                            pattern=tla.mask.ALL, dtype=io_dtype
                        )
                        for i7 in tla.range(rows_here):
                            ub_inv_i7 = tla.tile_view(
                                ub_inv, tla.make_shape(1), tla.make_coord(i7)
                            )
                            inv_reg7 = ub_inv_i7.load(
                                params=tla.params.NormalLoadParams(
                                    load_dist=tla.params.LoadDist.DIST_BRC_B32
                                )
                            )
                            ub_dm_i7 = tla.tile_view(
                                ub_dm_last, tla.make_shape(1), tla.make_coord(i7)
                            )
                            dm_reg7 = ub_dm_i7.load(
                                params=tla.params.NormalLoadParams(
                                    load_dist=tla.params.LoadDist.DIST_BRC_B32
                                )
                            )
                            acc_tile7_0 = tla.tile_view(
                                ub_o_re,
                                tla.make_shape(c1, vl_f32),
                                tla.make_coord(i7, 2 * oc),
                            )
                            acc_tile7_1 = tla.tile_view(
                                ub_o_re,
                                tla.make_shape(c1, vl_f32),
                                tla.make_coord(i7, 2 * oc + c1),
                            )
                            otmp_tile7_0 = tla.tile_view(
                                ub_otmp_ep,
                                tla.make_shape(c1, vl_f32),
                                tla.make_coord(i7, c0),
                            )
                            otmp_tile7_1 = tla.tile_view(
                                ub_otmp_ep,
                                tla.make_shape(c1, vl_f32),
                                tla.make_coord(i7, c1),
                            )
                            out_tile7 = tla.tile_view(
                                ub_out_t,
                                tla.make_shape(c1, 2 * vl_f32),
                                tla.make_coord(i7, c0),
                            )
                            acc_reg7_0 = tla.add(
                                tla.mul(acc_tile7_0.load(), dm_reg7, mask=preg_full7),
                                otmp_tile7_0.load(),
                                mask=preg_full7,
                            )
                            acc_reg7_1 = tla.add(
                                tla.mul(acc_tile7_1.load(), dm_reg7, mask=preg_full7),
                                otmp_tile7_1.load(),
                                mask=preg_full7,
                            )
                            acc_reg7_0 = tla.mul(acc_reg7_0, inv_reg7, mask=preg_full7)
                            acc_reg7_1 = tla.mul(acc_reg7_1, inv_reg7, mask=preg_full7)
                            if tla.const_expr(kvsplit > c1):
                                acc_tile7_0.store(acc_reg7_0, mask=preg_full7)
                                acc_tile7_1.store(acc_reg7_1, mask=preg_full7)
                            out_reg7_0 = acc_reg7_0.to(
                                io_dtype, cast_zero, mask=preg_full7
                            )
                            out_reg7_1 = acc_reg7_1.to(
                                io_dtype, cast_zero, mask=preg_full7
                            )
                            if tla.const_expr(kvsplit == 1):
                                packed7, _ = tla.deinterleave(out_reg7_0, out_reg7_1)
                                out_tile7.store(packed7, mask=preg_all_b16_7)
                    mtx_ub_out.unlock(pipe=tla.arch.VECTOR)

                    # kvsplit > 1 时本 task 的输出是 FP32 partial（写 mem_po），
                    # ub_out_t 里此刻装的是 P，不能当作 O 写到 mem_o。
                    if tla.const_expr(kvsplit == 1):
                        if rows_here > c0:
                            mtx_ub_out.lock(pipe=tla.arch.MTE3)
                            tla.copy(o_gm, ub_out_t)
                            mtx_ub_out.unlock(pipe=tla.arch.MTE3)
                    if tla.const_expr(otmp_n < pv_splits):
                        tla.cross_core_set_flag(o_free, tla.arch.VECTOR)
                if tla.const_expr(kvsplit > c1):
                    po_gm = tla.make_tensor(
                        mem_po.ptr
                        + (head_base + head_off) * total_q * kvsplit * d_v
                        + row_here * kvsplit * d_v
                        + chunk * d_v,
                        tla.make_layout(
                            tla.make_shape(rows_here, d_v),
                            tla.make_stride(total_q * kvsplit * d_v, c1),
                        ),
                    )
                    if rows_here > c0:
                        mtx_ub_out.lock(pipe=tla.arch.MTE3)
                        tla.copy(po_gm, ub_o_re)
                        mtx_ub_out.unlock(pipe=tla.arch.MTE3)
                # 最后一个 tile 的 otmp 已全部消费完，释放给 AIC 的下一 FIX。
                if tla.const_expr(otmp_n >= pv_splits):
                    tla.cross_core_set_flag(o_free, tla.arch.VECTOR)
            else:
                # 没有有效 KV tile：正常输出全 0，不必等 otmp。
                for oc in tla.range_constexpr(out_chunks):
                    o_gm = tla.make_tensor(
                        mem_o.ptr
                        + (head_base + head_off) * total_q * d_v
                        + row_here * d_v
                        + oc * out_chunk,
                        tla.make_layout(
                            tla.make_shape(rows_here, out_chunk),
                            tla.make_stride(total_q * d_v, c1),
                        ),
                    )
                    mtx_ub_out.lock(pipe=tla.arch.VECTOR)
                    with tla.vec.func(mode="simd"):
                        preg_all_b16_8 = tla.create_mask(
                            pattern=tla.mask.ALL, dtype=io_dtype
                        )
                        zero_b16 = tla.full(0.0, dtype=io_dtype)
                        for i8 in tla.range(rows_here):
                            out_tile8 = tla.tile_view(
                                ub_out_t,
                                tla.make_shape(c1, 2 * vl_f32),
                                tla.make_coord(i8, c0),
                            )
                            out_tile8.store(zero_b16, mask=preg_all_b16_8)
                    mtx_ub_out.unlock(pipe=tla.arch.VECTOR)
                    if tla.const_expr(kvsplit > c1):
                        mtx_ub_out.lock(pipe=tla.arch.VECTOR)
                        with tla.vec.func(mode="simd"):
                            preg_z9 = tla.create_mask(
                                pattern=tla.mask.ALL, dtype=tla.Float32
                            )
                            zero_z9 = tla.full(0.0, dtype=tla.Float32)
                            for i9 in tla.range(rows_here * num_dim_chunks):
                                acc_z9 = tla.tile_view(
                                    ub_o_re,
                                    tla.make_shape(c1, vl_f32),
                                    tla.make_coord(
                                        i9 // num_dim_chunks, i9 % num_dim_chunks
                                    ),
                                )
                                acc_z9.store(zero_z9, mask=preg_z9)
                        mtx_ub_out.unlock(pipe=tla.arch.VECTOR)

                    if rows_here > c0:
                        mtx_ub_out.lock(pipe=tla.arch.MTE3)
                        if tla.const_expr(kvsplit == 1):
                            tla.copy(o_gm, ub_out_t)
                        else:
                            po_gm_z = tla.make_tensor(
                                mem_po.ptr
                                + (head_base + head_off) * total_q * kvsplit * d_v
                                + row_here * kvsplit * d_v
                                + chunk * d_v,
                                tla.make_layout(
                                    tla.make_shape(rows_here, d_v),
                                    tla.make_stride(total_q * kvsplit * d_v, c1),
                                ),
                            )
                            tla.copy(po_gm_z, ub_o_re)
                        mtx_ub_out.unlock(pipe=tla.arch.MTE3)

            # LSE is written once per task (TND layout); the host converts it
            # to the official NTD (heads, total_q) layout. Keep the same
            # rows_here guard as the O write.
            if rows_here > c0:
                if tla.const_expr(kvsplit == 1):
                    lse_gm = tla.make_tensor(
                        mem_lse.ptr + row_here * heads + head_base + head_off,
                        tla.make_layout(
                            tla.make_shape(rows_here),
                            tla.make_stride(c1),
                        ),
                    )
                lse_ub = tla.make_tensor(
                    ub_lse_ptr,
                    tla.make_layout(
                        tla.make_shape(rows_here),
                        tla.make_stride(c1),
                    ),
                )
                mtx_ub_lse.lock(pipe=tla.arch.MTE3)
                if tla.const_expr(kvsplit == 1):
                    lse_gm = tla.make_tensor(
                        mem_lse.ptr + row_here * heads + head_base + head_off,
                        tla.make_layout(
                            tla.make_shape(rows_here),
                            tla.make_stride(c1),
                        ),
                    )
                else:
                    # partial LSE 布局：[(chunk*total_q + row)*heads + head]，
                    # 与官方 LSE 完全同形（连续 head 维，32B 对齐），一次 DMA 写完。
                    lse_gm = tla.make_tensor(
                        mem_pl.ptr
                        + (chunk * total_q + row_here) * heads
                        + head_base
                        + head_off,
                        tla.make_layout(
                            tla.make_shape(rows_here),
                            tla.make_stride(c1),
                        ),
                    )
                tla.copy(lse_gm, lse_ub)
                mtx_ub_lse.unlock(pipe=tla.arch.MTE3)

    with tla.cube():
        tla.cross_core_wait_flag(s_free, tla.arch.FIX)
        tla.cross_core_wait_flag(o_free, tla.arch.FIX)


# =========================================================================
# KERNEL B：KV-split 归并（纯 DSL）
#   task = (query row, head group of lse_group heads)
#   输入：f32 partial（与 kernel A 的 ub_o_re 同布局）+ partial LSE（8 lane，尾部 -inf）
#   输出：官方 f16 O（NTD）+ 官方 LSE（TND）
# =========================================================================
@tla.kernel
def flash_mla_kvsplit_reduce_kernel(
    mem_po: tla.Tensor,
    mem_pl: tla.Tensor,
    mem_o: tla.Tensor,
    mem_lse: tla.Tensor,
    total_q: tla.Int32,
    heads: tla.Int32,
    d_v: tla.Constexpr[int],
    kvsplit: tla.Constexpr[int],
    lse_group: tla.Constexpr[int],
) -> None:
    c0 = 0
    c1 = 1
    c8 = 8
    alignment = 512
    vl_f32 = 64
    out_chunk = 128
    out_chunks = d_v // out_chunk
    ndc = d_v // vl_f32
    io_dtype = mem_o.ptr.dtype

    ub_p = tla.allocate(kvsplit * d_v, tla.Float32, tla.AddressSpace.ub, alignment)
    ub_acc = tla.allocate(d_v, tla.Float32, tla.AddressSpace.ub, alignment)
    kb_alloc = max(kvsplit * lse_group, vl_f32)
    ub_l = tla.allocate(kb_alloc, tla.Float32, tla.AddressSpace.ub, 32)
    ub_w = tla.allocate(kvsplit * vl_f32, tla.Float32, tla.AddressSpace.ub, 32)
    ub_lg = tla.allocate(lse_group * c8, tla.Float32, tla.AddressSpace.ub, 32)
    ub_o = tla.allocate(
        out_chunks * out_chunk, io_dtype, tla.AddressSpace.ub, alignment
    )

    # 权重区：(chunk, head) 两维，head 维连续（lane = head，向量化到 head 上）
    lay_ls = tla.make_layout(
        tla.make_shape(kvsplit, lse_group), tla.make_stride(lse_group, c1)
    )
    lay_wp = tla.make_layout(
        tla.make_shape(kvsplit, vl_f32), tla.make_stride(vl_f32, c1)
    )
    ubl = tla.make_tensor(ub_l, lay_ls)
    ubw = tla.make_tensor(ub_w, lay_wp)
    ublg = tla.make_tensor(
        ub_lg,
        tla.make_layout(tla.make_shape(c1, lse_group), tla.make_stride(lse_group, c1)),
    )
    # 块行布局：一"块" = 64 个 f32 lane，列坐标恒 0、块序号走行坐标
    ubp_flat = tla.make_tensor(
        ub_p, tla.make_layout(tla.make_shape(kvsplit, d_v), tla.make_stride(d_v, c1))
    )
    ubp_blk = tla.make_tensor(
        ub_p,
        tla.make_layout(
            tla.make_shape(kvsplit * ndc, vl_f32), tla.make_stride(vl_f32, c1)
        ),
    )
    ubacc_blk = tla.make_tensor(
        ub_acc,
        tla.make_layout(tla.make_shape(ndc, vl_f32), tla.make_stride(vl_f32, c1)),
    )
    ubacc_128 = tla.make_tensor(
        ub_acc,
        tla.make_layout(
            tla.make_shape(out_chunks, out_chunk), tla.make_stride(out_chunk, c1)
        ),
    )
    ubo = tla.make_tensor(
        ub_o,
        tla.make_layout(
            tla.make_shape(out_chunks, out_chunk), tla.make_stride(out_chunk, c1)
        ),
    )
    ubo_flat = tla.make_tensor(
        ub_o, tla.make_layout(tla.make_shape(c1, d_v), tla.make_stride(d_v, c1))
    )

    mtx_l = tla.mutex("kvb_lse", id=21)
    mtx_p = tla.mutex("kvb_par", id=22)
    mtx_o = tla.mutex("kvb_out", id=23)

    cast_zero = tla.params.CastParams(
        reg_slot=tla.params.RegSlot.ZERO,
        sat_mode=tla.params.SatMode.SAT,
        round_mode=tla.params.RoundMode.CAST_ROUND,
    )
    cast_one = tla.params.CastParams(
        reg_slot=tla.params.RegSlot.ONE,
        sat_mode=tla.params.SatMode.SAT,
        round_mode=tla.params.RoundMode.CAST_ROUND,
    )
    dintlv = tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_DINTLV_B32)

    num_tasks = total_q * (heads // lse_group)
    for task in tla.range(tla.arch.block_idx(), num_tasks, tla.arch.block_num()):
        row = task % total_q
        head0 = (task // total_q) * lse_group
        gm_lg = tla.make_tensor(
            mem_lse.ptr + row * heads + head0,
            tla.make_layout(
                tla.make_shape(c1, lse_group), tla.make_stride(lse_group, c1)
            ),
        )
        with tla.vector():
            # ---------------- 权重（每 task 一次，lane 维 = head）
            mtx_l.lock(pipe=tla.arch.MTE2)
            for ic in tla.range_constexpr(kvsplit):
                gm_lc = tla.make_tensor(
                    mem_pl.ptr + (ic * total_q + row) * heads + head0,
                    tla.make_layout(
                        tla.make_shape(c1, lse_group), tla.make_stride(lse_group, c1)
                    ),
                )
                l_ic_t = tla.tile_view(
                    ubl, tla.make_shape(c1, lse_group), tla.make_coord(ic, c0)
                )
                tla.copy(l_ic_t, gm_lc)
            mtx_l.unlock(pipe=tla.arch.MTE2)
            mtx_l.lock(pipe=tla.arch.VECTOR)
            with tla.vec.func(mode="simd"):
                pm = tla.update_mask(lse_group, dtype=tla.Float32)[0]
                one_reg = tla.full(1.0, dtype=tla.Float32)
                zero_reg = tla.full(0.0, dtype=tla.Float32)
                neg_inf = tla.full(float("-inf"), dtype=tla.Float32)
                floor_reg = tla.full(-1.0e29, dtype=tla.Float32)
                mx = neg_inf
                for ia in tla.range_constexpr(kvsplit):
                    la_t = tla.tile_view(
                        ubl, tla.make_shape(c1, lse_group), tla.make_coord(ia, c0)
                    )
                    la_v = la_t.load()
                    mx = tla.max(
                        mx,
                        tla.where(
                            tla.cmp(la_v, floor_reg, mode="gt", mask=pm),
                            la_v,
                            floor_reg,
                        ),
                        mask=pm,
                    )
                sm = zero_reg
                for ib in tla.range_constexpr(kvsplit):
                    lb_t = tla.tile_view(
                        ubl, tla.make_shape(c1, lse_group), tla.make_coord(ib, c0)
                    )
                    e_v = tla.exp(tla.sub(lb_t.load(), mx, mask=pm), mask=pm)
                    sm = tla.add(sm, e_v, mask=pm)
                    wb_t = tla.tile_view(
                        ubw, tla.make_shape(c1, vl_f32), tla.make_coord(ib, c0)
                    )
                    wb_t.store(e_v, mask=pm)
                pos = tla.cmp(sm, 0.0, mode="gt", mask=pm)
                invv = tla.where(pos, tla.div(one_reg, sm, mask=pm), zero_reg)
                tla.local_mem_bar(
                    tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                )
                for ic2 in tla.range_constexpr(kvsplit):
                    wc_t = tla.tile_view(
                        ubw, tla.make_shape(c1, vl_f32), tla.make_coord(ic2, c0)
                    )
                    wc_t.store(tla.mul(wc_t.load(), invv, mask=pm), mask=pm)
                lg_t = tla.tile_view(
                    ublg, tla.make_shape(c1, lse_group), tla.make_coord(c0, c0)
                )
                lg_t.store(
                    tla.where(pos, tla.add(mx, tla.log(sm, mask=pm), mask=pm), neg_inf),
                    mask=pm,
                )
                tla.local_mem_bar(
                    tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                )
            mtx_l.unlock(pipe=tla.arch.VECTOR)
            mtx_l.lock(pipe=tla.arch.MTE3)
            tla.copy(gm_lg, ublg)
            mtx_l.unlock(pipe=tla.arch.MTE3)

            # ---------------- 逐 head：累加（持 mtx_p）→ 打包（持 mtx_o）→ 落盘
            for ih in tla.range(lse_group):
                head = head0 + ih
                gm_p = tla.make_tensor(
                    mem_po.ptr + (head * total_q + row) * kvsplit * d_v,
                    tla.make_layout(
                        tla.make_shape(kvsplit, d_v), tla.make_stride(d_v, c1)
                    ),
                )
                gm_o = tla.make_tensor(
                    mem_o.ptr + (head * total_q + row) * d_v,
                    tla.make_layout(tla.make_shape(c1, d_v), tla.make_stride(d_v, c1)),
                )
                mtx_p.lock(pipe=tla.arch.MTE2)
                tla.copy(ubp_flat, gm_p)
                mtx_p.unlock(pipe=tla.arch.MTE2)
                mtx_p.lock(pipe=tla.arch.VECTOR)
                with tla.vec.func(mode="simd"):
                    pf = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                    idx_ih = tla.full(ih, dtype=tla.Int32)
                    w_row0 = tla.tile_view(
                        ubw, tla.make_shape(c1, vl_f32), tla.make_coord(c0, c0)
                    ).load()
                    w_seed = tla.gather(w_row0, idx_ih)
                    for jz in tla.range_constexpr(ndc):
                        p_z = tla.tile_view(
                            ubp_blk, tla.make_shape(c1, vl_f32), tla.make_coord(jz, c0)
                        )
                        a_z = tla.tile_view(
                            ubacc_blk,
                            tla.make_shape(c1, vl_f32),
                            tla.make_coord(jz, c0),
                        )
                        a_z.store(tla.mul(p_z.load(), w_seed, mask=pf), mask=pf)
                    tla.local_mem_bar(
                        tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                    )
                    for ic in tla.range_constexpr(1, kvsplit):
                        w_c = tla.gather(
                            tla.tile_view(
                                ubw, tla.make_shape(c1, vl_f32), tla.make_coord(ic, c0)
                            ).load(),
                            idx_ih,
                        )
                        for jc in tla.range_constexpr(ndc):
                            p_t = tla.tile_view(
                                ubp_blk,
                                tla.make_shape(c1, vl_f32),
                                tla.make_coord(ic * ndc + jc, c0),
                            )
                            a_t = tla.tile_view(
                                ubacc_blk,
                                tla.make_shape(c1, vl_f32),
                                tla.make_coord(jc, c0),
                            )
                            a_t.store(
                                tla.add(
                                    a_t.load(),
                                    tla.mul(p_t.load(), w_c, mask=pf),
                                    mask=pf,
                                ),
                                mask=pf,
                            )
                    tla.local_mem_bar(
                        tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                    )
                mtx_p.unlock(pipe=tla.arch.VECTOR)
                mtx_o.lock(pipe=tla.arch.VECTOR)
                with tla.vec.func(mode="simd"):
                    pf2 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                    p16 = tla.create_mask(pattern=tla.mask.ALL, dtype=io_dtype)
                    for oc in tla.range_constexpr(out_chunks):
                        a_tile = tla.tile_view(
                            ubacc_128,
                            tla.make_shape(c1, out_chunk),
                            tla.make_coord(oc, c0),
                        )
                        o_t = tla.tile_view(
                            ubo, tla.make_shape(c1, out_chunk), tla.make_coord(oc, c0)
                        )
                        so, se = a_tile.load(params=dintlv)
                        o_16 = so.to(io_dtype, cast_zero, mask=pf2)
                        e_16 = se.to(io_dtype, cast_one, mask=pf2)
                        o_t.store(tla.bitwise_or(e_16, o_16, mask=p16), mask=p16)
                mtx_o.unlock(pipe=tla.arch.VECTOR)
                mtx_o.lock(pipe=tla.arch.MTE3)
                tla.copy(gm_o, ubo_flat)
                mtx_o.unlock(pipe=tla.arch.MTE3)


_EXECUTOR_CACHE = {}
# Inputs are usually re-used across decode steps/benchmark trials. Wrapping a
# tensor through DLPack dominates the host path, so keep the wrapper keyed by
# the exact tensor object and its storage pointer.
_TENSOR_CACHE = {}
_TENSOR_CACHE_LIMIT = 32
_PLAN_CACHE = {}
_PLAN_CACHE_LIMIT = 32


def _tla_tensor(value):
    return from_dlpack(
        value, layout_tag=arch_row_major(), origin_shape=(value.numel(),)
    ).mark_layout_dynamic()


def _tla_tensor_cached(value):
    key = (value.data_ptr(), value.dtype, value.numel(), str(value.device))
    cached = _TENSOR_CACHE.get(key)
    if cached is not None and cached[0] is value:
        return cached[1]
    wrapped = _tla_tensor(value)
    if len(_TENSOR_CACHE) >= _TENSOR_CACHE_LIMIT:
        _TENSOR_CACHE.clear()
    _TENSOR_CACHE[key] = (value, wrapped)
    return wrapped


def arch_row_major():
    return tla.arch.RowMajor


def _check_cache_shape(k_cache, cache_layout):
    if cache_layout == "PA_BBND":
        if tuple(k_cache.shape[1:]) != (PAGE_SIZE, 1, D_QK):
            raise ValueError("PA_BBND k_cache must have shape [pages, 128, 1, 576]")
        return int(k_cache.shape[1]), int(k_cache.stride(0))
    if cache_layout == "PA_NZ":
        if tuple(k_cache.shape[1:]) != (1, D_QK // 16, PAGE_SIZE, 16):
            raise ValueError("PA_NZ k_cache must have shape [pages, 1, 36, 128, 16]")
        return int(k_cache.shape[3]), int(k_cache.stride(0))
    raise ValueError("cache_layout must be PA_BBND or PA_NZ")


def _check_capacity_shapes(q, block_table, cu_seqlens_q, cache_seqlens):
    """Validate the length-tensor shapes; these costs are host-only.

    The plan cache key only captures shapes, strides, dtypes and scalars, so a
    cache hit must still reject mismatched length tensors instead of indexing
    them blindly.
    """
    if q.ndim != 3:
        raise ValueError("q must have shape [total_q, num_heads, 576]")
    if block_table.ndim != 2:
        raise ValueError("block_table must be two-dimensional")
    batch = int(block_table.shape[0])
    if cu_seqlens_q.shape != (batch + 1,):
        raise ValueError("cu_seqlens_q must contain batch + 1 entries")
    if cache_seqlens.shape != (batch,):
        raise ValueError("cache_seqlens must contain one entry per request")


def _capacity_signature(cu_seqlens_q, cache_seqlens):
    """Identify the current contents of the two length tensors cheaply.

    Reading their values costs one device reduction plus a host sync, so the
    value checks are skipped while both the storage and the in-place update
    counter are unchanged.
    """
    return (
        cu_seqlens_q.data_ptr(),
        cu_seqlens_q._version,
        cache_seqlens.data_ptr(),
        cache_seqlens._version,
    )


def _check_capacity_values(q, block_table, cu_seqlens_q, cache_seqlens):
    """Validate the two capacity invariants that read tensor values.

    Without them a too large cache_seqlens or a mismatched cu_seqlens_q would
    drive the device kernel past the block_table and output bounds.
    """
    if int(cu_seqlens_q[-1]) != int(q.shape[0]):
        raise ValueError("cu_seqlens_q[-1] must equal q.shape[0]")
    if int(cache_seqlens.max()) > int(block_table.shape[1]) * PAGE_SIZE:
        raise ValueError("cache_seqlens must not exceed block_table pages * page_size")


def _plan_key(
    q, k_cache, block_table, cu_seqlens_q, cache_seqlens, scale, mask_mode, cache_layout
):
    return (
        tuple(q.shape),
        q.dtype,
        q.stride(),
        tuple(k_cache.shape),
        k_cache.dtype,
        k_cache.stride(),
        tuple(block_table.shape),
        block_table.dtype,
        block_table.stride(),
        tuple(cu_seqlens_q.shape),
        cu_seqlens_q.dtype,
        tuple(cache_seqlens.shape),
        cache_seqlens.dtype,
        str(q.device),
        str(k_cache.device),
        str(block_table.device),
        str(cu_seqlens_q.device),
        str(cache_seqlens.device),
        cache_layout,
        int(mask_mode),
        float(scale),
    )


def flash_mla_with_kvcache(
    q,
    k_cache,
    block_table,
    cu_seqlens_q,
    cache_seqlens,
    softmax_scale=None,
    mask_mode=MASK_NONE,
    cache_layout="PA_BBND",
):
    scale_value = 1.0 if softmax_scale is None else float(softmax_scale)
    # Shape-level checks are host-only; the two invariants that read tensor
    # values cost one device reduction, so they run on a plan miss and whenever
    # the length tensors changed.
    _check_capacity_shapes(q, block_table, cu_seqlens_q, cache_seqlens)
    capacity_sig = _capacity_signature(cu_seqlens_q, cache_seqlens)
    plan_key = _plan_key(
        q,
        k_cache,
        block_table,
        cu_seqlens_q,
        cache_seqlens,
        scale_value,
        mask_mode,
        cache_layout,
    )
    plan = _PLAN_CACHE.get(plan_key)

    if plan is None:
        if q.shape[2] != D_QK:
            raise ValueError("q head dimension must be 576")
        if q.shape[1] not in (64, 96):
            raise ValueError("q num_heads must be 64 or 96")
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("q dtype must be float16 or bfloat16")
        if not q.is_contiguous():
            raise ValueError("q must be contiguous")
        if k_cache.dtype != q.dtype:
            raise ValueError("k_cache dtype must match q dtype")
        if block_table.dtype != torch.int32:
            raise ValueError("block_table dtype must be int32")
        if cu_seqlens_q.dtype != torch.int32 or cache_seqlens.dtype != torch.int32:
            raise ValueError("sequence-length tensors must be int32")
        if not block_table.is_contiguous():
            raise ValueError("block_table must be contiguous")
        if not cu_seqlens_q.is_contiguous() or not cache_seqlens.is_contiguous():
            raise ValueError("sequence-length tensors must be contiguous")
        if mask_mode not in (MASK_NONE, MASK_RIGHT_DOWN_CAUSAL):
            raise ValueError("mask_mode must be 0 or 3")
        _, kv_blk_stride = _check_cache_shape(k_cache, cache_layout)

        total_q, heads, _ = q.shape
        batch = int(block_table.shape[0])
        num_pages = int(block_table.shape[1])
        _check_capacity_values(q, block_table, cu_seqlens_q, cache_seqlens)
        device_properties = torch_npu.npu.get_device_properties(q.device)
        cube_core_num = int(device_properties.cube_core_num)
        if cube_core_num < 1:
            raise RuntimeError("NPU device reports no available Cube cores")
        # N=64：两个相邻 query token 合并成一个 task（M=128），K/V 每 token
        # 只加载一次；只有 task 数仍能铺满 Cube 核（total_q >= 2*cores）时才启用，
        # 避免 T=1/8 这类小 batch 把并行度砍半。N=96：一个 token 一个 task（M=96）。
        # 合并后 task 数 = total_q/2；要求至少 4 个完整的 task 波次避免尾部失衡。
        paired = heads <= Q_BLOCK and total_q >= 8 * cube_core_num
        # 低并行度补核（纯 DSL 调度层）：total_q 极少时（decode 形态）任务数远小于
        # 核数，把一个 query 的 head 维切成 hgroups 组，多核并行处理同一 query 的
        # 不同 head 段。每组至少 16 个 head（MMA 原生 M 粒度），heads 需整除。
        hgroups = 1
        if not paired and total_q <= 4:
            budget = max(1, cube_core_num // max(1, total_q))
            for g in range(min(budget, heads // 16), 1, -1):
                if heads % g == 0 and heads // g >= 16:
                    hgroups = g
                    break
        # KV-split：低并行度（task 数远不够铺满核）时把每个 task 的 KV tile
        # 序列切成 kvsplit 段并行算，再由 kernel B 归并。要求每个 chunk 至少
        # 2 个 tile，否则并行收益被归并开销吃掉。
        ksplit = 1
        if not paired:
            tiles_max = (int(cache_seqlens.max()) + KV_BLOCK - 1) // KV_BLOCK
            base_tasks = max(1, total_q * hgroups)
            want = max(1, cube_core_num // base_tasks)
            for k in (8, 4, 2):
                if k <= want and 2 * k <= tiles_max:
                    ksplit = k
                    break
        lse_group = 1
        if ksplit > 1:
            # kernel B 的 partial-LSE 一次读 lse_group 个 head（32B 对齐），
            # 故 lane 数下限是 8；再取「2 的幂且整除 heads」。
            lg = max(8, (total_q * heads) // 64)
            lg = min(lg, 16, heads)
            while lg > 8 and (heads % lg != 0 or (lg & (lg - 1)) != 0):
                lg -= 1
            if heads % lg != 0:
                lg = 8
            lse_group = lg
        q_block = (
            2 * heads
            if paired
            else (
                heads // hgroups
                if hgroups > 1
                else (Q_BLOCK if heads <= Q_BLOCK else 96)
            )
        )
        # PV 宽度固定 128：L0B 用两个 32 KiB slot 做 V 分段 ping-pong，
        # 与 QK 的 K 分段共用同一组 slot。
        pv_width = 128
        executor_key = (
            str(q.dtype),
            str(q.device),
            cache_layout,
            heads,
            q_block,
            hgroups,
            ksplit,
            lse_group,
        )
        executor = _EXECUTOR_CACHE.get(executor_key)
        if executor is None:
            probe_out = torch.empty(
                (heads, total_q, D_V), dtype=q.dtype, device=q.device
            )
            probe_lse = torch.empty(
                (total_q, heads), dtype=torch.float32, device=q.device
            )
            executor = tla_compile(
                q,
                k_cache,
                probe_out,
                probe_lse,
                block_table,
                cu_seqlens_q,
                cache_seqlens,
                scale_value,
                total_q,
                batch,
                heads,
                num_pages,
                kv_blk_stride,
                mask_mode,
                q_block,
                hgroups,
                ksplit,
                lse_group,
                pv_width,
                cache_layout,
            )
            _EXECUTOR_CACHE[executor_key] = executor

        plan = {
            "batch": batch,
            "total_q": total_q,
            "heads": heads,
            "kv_blk_stride": kv_blk_stride,
            "num_pages": num_pages,
            "executor": executor,
            "kvsplit": ksplit,
            "lse_group": lse_group,
            "launch_blocks": min(total_q * hgroups * ksplit, cube_core_num),
            "launch_blocks_b": min(
                max(1, total_q * (heads // lse_group)), cube_core_num
            ),
            "dummy_po": torch.empty((8,), dtype=torch.float32, device=q.device),
            "dummy_pl": torch.empty((8,), dtype=torch.float32, device=q.device),
            "scale_t": tla_float32(scale_value),
            "total_q_t": tla_int32(total_q),
            "batch_t": tla_int32(batch),
            "heads_t": tla_int32(heads),
            "pages_t": tla_int32(num_pages),
            "blk_stride_t": tla_int32(kv_blk_stride),
            "mask_t": tla_int32(int(mask_mode)),
            "capacity_sig": capacity_sig,
        }
        if len(_PLAN_CACHE) >= _PLAN_CACHE_LIMIT:
            _PLAN_CACHE.clear()
        _PLAN_CACHE[plan_key] = plan
    elif plan["capacity_sig"] != capacity_sig:
        _check_capacity_values(q, block_table, cu_seqlens_q, cache_seqlens)
        plan["capacity_sig"] = capacity_sig

    total_q = plan["total_q"]
    heads = plan["heads"]
    # Official layout_out=NTD: (heads, total_q, D_V); LSE: (heads, total_q).
    output = torch.empty((heads, total_q, D_V), dtype=q.dtype, device=q.device)
    lse_tnd = torch.empty((total_q, heads), dtype=torch.float32, device=q.device)
    if total_q == 0 or plan["batch"] == 0:
        return output, lse_tnd.transpose(0, 1)

    q_t = _tla_tensor_cached(q)
    k_t = _tla_tensor_cached(k_cache)
    o_t = _tla_tensor(output)
    lse_t = _tla_tensor(lse_tnd)
    bt_t = _tla_tensor_cached(block_table)
    cu_t = _tla_tensor_cached(cu_seqlens_q)
    kvlen_t = _tla_tensor_cached(cache_seqlens)

    ksplit = plan["kvsplit"]
    if ksplit > 1:
        po = torch.empty(
            (ksplit * heads * total_q * D_V,),
            dtype=torch.float32,
            device=q.device,
        )
        pl = torch.empty(
            (ksplit * total_q * heads,),
            dtype=torch.float32,
            device=q.device,
        )
        po_t = _tla_tensor(po)
        pl_t = _tla_tensor(pl)
    else:
        po_t = _tla_tensor(plan["dummy_po"])
        pl_t = _tla_tensor(plan["dummy_pl"])
    ex_a = plan["executor"]
    if ksplit > 1:
        ex_a, ex_b = ex_a
    ex_a(
        q_t,
        k_t,
        o_t,
        lse_t,
        po_t,
        pl_t,
        bt_t,
        cu_t,
        kvlen_t,
        plan["scale_t"],
        plan["total_q_t"],
        plan["batch_t"],
        plan["heads_t"],
        plan["pages_t"],
        plan["blk_stride_t"],
        plan["mask_t"],
        block_num=plan["launch_blocks"],
    )
    if ksplit > 1:
        ex_b(
            po_t,
            pl_t,
            o_t,
            lse_t,
            plan["total_q_t"],
            plan["heads_t"],
            block_num=plan["launch_blocks_b"],
        )
    # Official contract: attn_out is NTD (heads, total_q, DV) and
    # softmax_lse is (heads, total_q). The kernel writes LSE in TND; returning
    # the transposed view avoids a per-call copy. Callers that need a packed
    # buffer can request `.contiguous()`.
    return output, lse_tnd.transpose(0, 1)


def tla_compile(
    q,
    k_cache,
    output,
    lse_tnd,
    block_table,
    cu_seqlens_q,
    cache_seqlens,
    scale_value,
    total_q,
    batch,
    heads,
    num_pages,
    kv_blk_stride,
    mask_mode,
    q_block,
    hgroups,
    kvsplit,
    lse_group,
    pv_width,
    cache_layout,
):
    dummy = torch.empty((8,), dtype=torch.float32, device=q.device)
    probe_po = (
        torch.empty(
            (kvsplit * heads * total_q * D_V,), dtype=torch.float32, device=q.device
        )
        if kvsplit > 1
        else dummy
    )
    probe_pl = (
        torch.empty((kvsplit * heads * total_q,), dtype=torch.float32, device=q.device)
        if kvsplit > 1
        else dummy
    )
    ex_a = tla.compile(
        flash_mla_with_kvcache_kernel,
        _tla_tensor_cached(q),
        _tla_tensor_cached(k_cache),
        _tla_tensor(output),
        _tla_tensor(lse_tnd),
        _tla_tensor(probe_po),
        _tla_tensor(probe_pl),
        _tla_tensor_cached(block_table),
        _tla_tensor_cached(cu_seqlens_q),
        _tla_tensor_cached(cache_seqlens),
        tla_float32(scale_value),
        tla_int32(total_q),
        tla_int32(batch),
        tla_int32(heads),
        tla_int32(num_pages),
        tla_int32(kv_blk_stride),
        tla_int32(int(mask_mode)),
        D_QK,
        D_V,
        q_block,
        hgroups,
        kvsplit,
        KV_BLOCK,
        pv_width,
        0 if cache_layout == "PA_BBND" else 1,
        options=NPU_ARCH,
    )
    if kvsplit == 1:
        return ex_a
    ex_b = tla.compile(
        flash_mla_kvsplit_reduce_kernel,
        _tla_tensor(probe_po),
        _tla_tensor(probe_pl),
        _tla_tensor(output),
        _tla_tensor(lse_tnd),
        tla_int32(total_q),
        tla_int32(heads),
        D_V,
        kvsplit,
        lse_group,
        options=NPU_ARCH,
    )
    return (ex_a, ex_b)


def tla_int32(value):
    return tla.Int32(value)


def tla_float32(value):
    return tla.Float32(value)


# =============================================================================
# Torch golden.
# =============================================================================
#
# A straightforward chunked implementation of the same contract: gather the
# valid KV pages through ``block_table``, run the QK product, apply the mask,
# take the softmax in fp32 and reduce PV. It runs on whatever device the inputs
# carry, so the example can check the kernel against it on the device (fast) or
# on the CPU (a fully independent path, for cross-checking).


def _check_golden_cache_shape(cache_shape, cache_layout):
    if cache_layout == CACHE_LAYOUT_BBND:
        expected = (PAGE_SIZE, KV_HEAD_NUM, HEAD_DIM_QK)
        if tuple(cache_shape[1:]) != expected:
            raise ValueError(
                "PA_BBND cache must have shape "
                "[num_blocks, 128, 1, 576], got {}".format(tuple(cache_shape))
            )
    elif cache_layout == CACHE_LAYOUT_NZ:
        expected = (KV_HEAD_NUM, HEAD_DIM_QK // 16, PAGE_SIZE, 16)
        if tuple(cache_shape[1:]) != expected:
            raise ValueError(
                "PA_NZ cache must have shape "
                "[num_blocks, 1, 36, 128, 16], got {}".format(tuple(cache_shape))
            )
    else:
        raise ValueError("cache_layout must be one of {}".format(CACHE_LAYOUTS))


def _gather_kv(k_cache, block_table_row, kv_len, cache_layout):
    """Collect the first kv_len tokens following the paged block table."""
    num_pages = (kv_len + PAGE_SIZE - 1) // PAGE_SIZE
    physical_pages = block_table_row[:num_pages].long()
    if cache_layout == CACHE_LAYOUT_BBND:
        blocks = k_cache.index_select(0, physical_pages).squeeze(2)
    else:
        blocks = k_cache.index_select(0, physical_pages).squeeze(1)
        blocks = blocks.permute(0, 2, 1, 3).reshape(num_pages, PAGE_SIZE, HEAD_DIM_QK)
    return blocks.reshape(-1, HEAD_DIM_QK)[:kv_len]


def flash_mla_with_kvcache_golden(
    q,
    k_cache,
    block_table,
    cu_seqlens_q,
    cache_seqlens,
    softmax_scale=None,
    mask_mode=MASK_NONE,
    cache_layout=CACHE_LAYOUT_BBND,
):
    if q.ndim != 3:
        raise ValueError("q must have shape [total_q, num_heads, 576]")
    if q.shape[2] != HEAD_DIM_QK:
        raise ValueError("q head dimension must be 576")
    if q.shape[1] not in HEAD_CHOICES:
        raise ValueError("q num_heads must be 64 or 96")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("q dtype must be float16 or bfloat16")
    if k_cache.dtype != q.dtype:
        raise ValueError("k_cache dtype must match q dtype")
    if block_table.ndim != 2:
        raise ValueError("block_table must be two-dimensional")
    batch = block_table.shape[0]
    if cu_seqlens_q.shape != (batch + 1,):
        raise ValueError("cu_seqlens_q must contain batch + 1 entries")
    if cache_seqlens.shape != (batch,):
        raise ValueError("cache_seqlens must contain one entry per request")
    _check_golden_cache_shape(k_cache.shape, cache_layout)
    if mask_mode not in MASK_MODES:
        raise ValueError("mask_mode must be 0 or 3")
    if softmax_scale is None:
        softmax_scale = 1.0
    if int(cu_seqlens_q[0].item()) != 0:
        raise ValueError("cu_seqlens_q must start at zero")
    if int(cu_seqlens_q[-1].item()) != q.shape[0]:
        raise ValueError("cu_seqlens_q must end at the total query length")
    if torch.any(cu_seqlens_q[1:] < cu_seqlens_q[:-1]):
        raise ValueError("cu_seqlens_q must be non-decreasing")

    total_q, num_heads, _ = q.shape
    output = torch.zeros(
        (total_q, num_heads, HEAD_DIM_V), dtype=q.dtype, device=q.device
    )
    lse = torch.full(
        (total_q, num_heads), float("-inf"), dtype=torch.float32, device=q.device
    )

    for batch_idx in range(batch):
        q_start = int(cu_seqlens_q[batch_idx].item())
        q_end = int(cu_seqlens_q[batch_idx + 1].item())
        q_len = q_end - q_start
        kv_len = int(cache_seqlens[batch_idx].item())
        if q_len <= 0 or kv_len <= 0:
            continue
        if mask_mode == MASK_RIGHT_DOWN_CAUSAL and kv_len < q_len:
            raise ValueError(
                "right-down causal mask requires cache length >= query length"
            )

        kv = _gather_kv(k_cache, block_table[batch_idx], kv_len, cache_layout).float()
        keys = kv
        values = kv[:, :HEAD_DIM_V]
        queries = q[q_start:q_end].float()

        for chunk_start in range(0, q_len, QUERY_CHUNK):
            chunk_end = min(chunk_start + QUERY_CHUNK, q_len)
            scores = torch.matmul(queries[chunk_start:chunk_end], keys.transpose(0, 1))
            scores = scores * softmax_scale
            if mask_mode == MASK_RIGHT_DOWN_CAUSAL:
                query_positions = (kv_len - q_len) + torch.arange(
                    chunk_start, chunk_end, device=q.device
                )
                key_positions = torch.arange(kv_len, device=q.device)
                visible = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
                scores = scores.masked_fill(~visible.unsqueeze(1), float("-inf"))

            row_max = scores.amax(dim=-1)
            finite = torch.isfinite(row_max)
            safe_max = torch.where(finite, row_max, torch.zeros_like(row_max))
            weights = torch.exp(scores - safe_max.unsqueeze(-1))
            row_sum = weights.sum(dim=-1)
            valid = finite & (row_sum > 0)
            lse_chunk = torch.where(
                valid,
                safe_max + torch.log(row_sum),
                torch.full_like(row_sum, float("-inf")),
            )
            weights = torch.where(
                valid.unsqueeze(-1),
                weights / row_sum.clamp_min(1).unsqueeze(-1),
                torch.zeros_like(weights),
            )
            out_chunk = torch.matmul(weights, values)
            output[q_start + chunk_start : q_start + chunk_end] = out_chunk.to(q.dtype)
            lse[q_start + chunk_start : q_start + chunk_end] = lse_chunk

    # Official layout_out=NTD: (num_heads, total_q, head_dim_v).
    output = output.transpose(0, 1).contiguous()
    lse = lse.transpose(0, 1).contiguous()
    return output, lse


# =============================================================================
# Example driver: build inputs, launch, check against the golden.
# =============================================================================


@dataclass
class _Case:
    """One example case: a batch of requests and the operator selectors."""

    batch: int
    total_q: int
    kv_len: int
    heads: int
    dtype: str
    layout: str
    mask: int


#: The suite replayed by ``--cases``. Between them the cases cover the decode
#: shape (one query row), the paired two-token path, the 96-head path, both
#: cache layouts, both mask modes, both dtypes, a partial last page and a
#: KV-split case.
_CASES = (
    _Case(1, 1, 512, 64, "float16", CACHE_LAYOUT_BBND, MASK_NONE),
    _Case(1, 8, 128, 64, "float16", CACHE_LAYOUT_BBND, MASK_NONE),
    _Case(1, 64, 512, 64, "float16", CACHE_LAYOUT_BBND, MASK_NONE),
    _Case(2, 512, 1024, 64, "float16", CACHE_LAYOUT_BBND, MASK_NONE),
    _Case(1, 64, 512, 96, "float16", CACHE_LAYOUT_NZ, MASK_NONE),
    _Case(1, 64, 512, 64, "float16", CACHE_LAYOUT_BBND, MASK_RIGHT_DOWN_CAUSAL),
    _Case(2, 130, 383, 96, "bfloat16", CACHE_LAYOUT_NZ, MASK_RIGHT_DOWN_CAUSAL),
    _Case(2, 512, 2048, 64, "bfloat16", CACHE_LAYOUT_BBND, MASK_NONE),
)

#: Share of the elements that must land inside ``atol``, the same rule the
#: operator's frozen workload applies.
MATCHED_RATIO = 0.999


@dataclass
class _Inputs:
    """The operator inputs plus the per-request lengths they were built from."""

    q: torch.Tensor  # [total_q, heads, 576]
    k_cache: torch.Tensor  # [pages, ...] in the requested layout
    block_table: torch.Tensor  # [batch, pages_per_request] int32
    cu_seqlens_q: torch.Tensor  # [batch + 1] int32
    cache_seqlens: torch.Tensor  # [batch] int32
    softmax_scale: float


def _torch_dtype(name):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError("dtype must be float16 or bfloat16")


def _build_inputs(case, args):
    """Allocate the case's inputs.

    Every request gets its own physical pages, then the page order is shuffled:
    a kernel that ignored ``block_table`` and walked the cache linearly would
    still read valid memory, but the wrong tokens.
    """

    dtype = _torch_dtype(case.dtype)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    q_lens = [case.total_q // case.batch] * case.batch
    for index in range(case.total_q % case.batch):
        q_lens[index] += 1
    if case.mask == MASK_RIGHT_DOWN_CAUSAL and any(
        case.kv_len < q_len for q_len in q_lens
    ):
        raise ValueError("mask_mode=3 needs kv_len >= every request's query length")

    pages = max(1, -(-case.kv_len // PAGE_SIZE))
    num_blocks = case.batch * pages
    if case.layout == CACHE_LAYOUT_BBND:
        cache_shape = (num_blocks, PAGE_SIZE, KV_HEAD_NUM, HEAD_DIM_QK)
    else:
        cache_shape = (num_blocks, KV_HEAD_NUM, HEAD_DIM_QK // 16, PAGE_SIZE, 16)

    q = torch.randn(case.total_q, case.heads, HEAD_DIM_QK, generator=generator)
    k_cache = torch.randn(cache_shape, generator=generator)

    order = torch.randperm(num_blocks, generator=generator).tolist()
    block_table = [order[row * pages : (row + 1) * pages] for row in range(case.batch)]

    cu = [0]
    for q_len in q_lens:
        cu.append(cu[-1] + q_len)
    if cu[-1] != case.total_q:
        raise AssertionError("query split does not add up")

    device = torch.device("npu:%d" % args.device)
    return _Inputs(
        q=q.to(dtype).npu(device),
        k_cache=k_cache.to(dtype).npu(device),
        block_table=torch.tensor(block_table, dtype=torch.int32).npu(device),
        cu_seqlens_q=torch.tensor(cu, dtype=torch.int32).npu(device),
        cache_seqlens=torch.tensor([case.kv_len] * case.batch, dtype=torch.int32).npu(
            device
        ),
        softmax_scale=args.scale if args.scale > 0 else 1.0 / math.sqrt(HEAD_DIM_QK),
    )


def _check_output(result, expected, atol):
    """Return (matched_ratio, max_abs_error) of ``result`` against ``expected``."""

    diff = (result.float() - expected.float()).abs()
    if diff.numel() == 0:
        return 1.0, 0.0
    matched = float((diff <= atol).float().mean())
    return matched, float(diff.max())


def _check_lse(result, expected, atol):
    """Return (max_abs_error, same_invalid_mask) over the rows both call finite.

    A row with no visible token is ``-inf`` on both sides, so comparing the two
    infinities would only produce NaNs; the two masks are compared instead.
    """

    both_finite = torch.isfinite(result) & torch.isfinite(expected)
    same_mask = bool((torch.isfinite(result) == torch.isfinite(expected)).all())
    if not bool(both_finite.any()):
        return 0.0, same_mask
    error = (result[both_finite].float() - expected[both_finite].float()).abs()
    return float(error.max()), same_mask


def _launch(inputs, case):
    return flash_mla_with_kvcache(
        inputs.q,
        inputs.k_cache,
        inputs.block_table,
        inputs.cu_seqlens_q,
        inputs.cache_seqlens,
        inputs.softmax_scale,
        case.mask,
        case.layout,
    )


def _min_bytes(case):
    """Bytes one launch must move at least: latent pages, q, out and lse.

    V reuses the 576-wide latent page that K already needs, so a page is counted
    once. With ``mask_mode=3`` some rows see fewer than ``kv_len`` tokens, so
    this is an upper bound there.
    """

    element = 2  # both fp16 and bf16 are two bytes wide
    kv = case.batch * case.kv_len * HEAD_DIM_QK * element
    q = case.total_q * case.heads * HEAD_DIM_QK * element
    out = case.total_q * case.heads * HEAD_DIM_V * element
    lse = case.total_q * case.heads * 4
    return kv + q + out + lse


def _measure_latency(case, args, inputs):
    """Time ``--perf-iters`` consecutive launches, one event pair per launch."""

    output, lse = _launch(inputs, case)
    for _ in range(max(0, args.perf_warmup - 1)):
        output, lse = _launch(inputs, case)
    torch.npu.synchronize(args.device)

    starts = [torch.npu.Event(enable_timing=True) for _ in range(args.perf_iters)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(args.perf_iters)]
    for start, end in zip(starts, ends):
        start.record()
        output, lse = _launch(inputs, case)
        end.record()
    torch.npu.synchronize(args.device)

    # The timed launches must still produce the declared output shapes: a silent
    # shape change would make the latency and bandwidth numbers meaningless.
    if tuple(output.shape) != (case.heads, case.total_q, HEAD_DIM_V):
        raise RuntimeError("timed launch returned an unexpected output shape")
    if tuple(lse.shape) != (case.heads, case.total_q):
        raise RuntimeError("timed launch returned an unexpected lse shape")

    samples = sorted(start.elapsed_time(end) for start, end in zip(starts, ends))
    mean_ms = sum(samples) / len(samples)
    total_bytes = _min_bytes(case)
    return {
        "latency_scope": "eager_end_to_end_including_host",
        "latency_us_mean": mean_ms * 1000.0,
        "latency_us_min": samples[0] * 1000.0,
        "latency_us_max": samples[-1] * 1000.0,
        "bytes_moved": total_bytes,
        "effective_gbps": total_bytes / mean_ms / 1e6,
        "tokens_per_s": case.total_q / (mean_ms * 1e-3),
        "perf_warmup": args.perf_warmup,
        "perf_iters": len(samples),
    }


def _run_case(case, args):
    inputs = _build_inputs(case, args)
    result, lse = _launch(inputs, case)
    torch.npu.synchronize(args.device)

    record = {
        "name": "B%d_T%d_KV%d_H%d_%s_%s_M%d"
        % (
            case.batch,
            case.total_q,
            case.kv_len,
            case.heads,
            case.dtype,
            case.layout,
            case.mask,
        ),
        "batch": case.batch,
        "total_q": case.total_q,
        "kv_len": case.kv_len,
        "heads": case.heads,
        "dtype": case.dtype,
        "cache_layout": case.layout,
        "mask_mode": case.mask,
    }
    if args.perf or args.perf_only:
        record.update(_measure_latency(case, args, inputs))
    if args.perf_only:
        record["passed"] = None
        return record

    golden = (
        (
            inputs.q.cpu(),
            inputs.k_cache.cpu(),
            inputs.block_table.cpu(),
            inputs.cu_seqlens_q.cpu(),
            inputs.cache_seqlens.cpu(),
        )
        if args.golden_device == "cpu"
        else (
            inputs.q,
            inputs.k_cache,
            inputs.block_table,
            inputs.cu_seqlens_q,
            inputs.cache_seqlens,
        )
    )
    expected_out, expected_lse = flash_mla_with_kvcache_golden(
        golden[0],
        golden[1],
        golden[2],
        golden[3],
        golden[4],
        inputs.softmax_scale,
        case.mask,
        case.layout,
    )
    expected_out = expected_out.to(result.device)
    expected_lse = expected_lse.to(lse.device)

    atol = (
        args.atol if args.atol > 0 else (0.01 if case.dtype == "bfloat16" else 0.0025)
    )
    matched, max_abs = _check_output(result, expected_out, atol)
    lse_error, lse_mask_ok = _check_lse(lse, expected_lse, atol)
    lse_ok = lse_mask_ok and lse_error <= max(atol * 10.0, 0.02)

    record.update(
        {
            "atol": atol,
            "matched_ratio": matched,
            "out_max_abs_error": max_abs,
            "lse_max_abs_error": lse_error,
            "passed": bool(matched >= MATCHED_RATIO and lse_ok),
        }
    )
    return record


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch", "--b", type=int, default=1)
    parser.add_argument("--total-q", "--t", type=int, default=512)
    parser.add_argument("--kv-len", "--k", type=int, default=1024)
    parser.add_argument("--heads", "--h", type=int, default=64, choices=HEAD_CHOICES)
    parser.add_argument("--dtype", choices=DTYPE_NAMES, default="float16")
    parser.add_argument("--layout", choices=CACHE_LAYOUTS, default=CACHE_LAYOUT_BBND)
    parser.add_argument(
        "--mask",
        type=int,
        choices=MASK_MODES,
        default=MASK_NONE,
        help="0: dense, 3: right-down causal over cache_seqlens",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.0,
        help="softmax scale; 0 uses 1/sqrt(576)",
    )
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument(
        "--golden-device",
        choices=("npu", "cpu"),
        default="npu",
        help="where the golden runs; cpu is the slow independent cross-check",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=0.0,
        help="absolute tolerance; 0 uses 0.0025 (fp16) or 0.01 (bf16)",
    )
    parser.add_argument(
        "--perf", action="store_true", help="also report device latency"
    )
    parser.add_argument(
        "--perf-only",
        "--perf_only",
        dest="perf_only",
        action="store_true",
        help="only measure latency, skip the golden check",
    )
    parser.add_argument("--perf-iters", type=int, default=20)
    parser.add_argument(
        "--perf-warmup", type=int, default=5, help="launches discarded before timing"
    )
    parser.add_argument(
        "--cases", action="store_true", help="validate the boundary-case suite"
    )
    args = parser.parse_args(argv)
    if args.device < 0:
        parser.error("--device must be nonnegative")
    if args.batch < 1 or args.total_q < 1 or args.kv_len < 1:
        parser.error("--batch, --total-q and --kv-len must be positive")
    if args.batch > args.total_q:
        parser.error("--batch must not exceed --total-q")
    if not math.isfinite(args.scale) or args.scale < 0:
        parser.error("--scale must be finite and nonnegative")
    if not math.isfinite(args.atol) or args.atol < 0:
        parser.error("--atol must be finite and nonnegative")
    if args.perf_iters < 1:
        parser.error("--perf-iters must be positive")
    if args.perf_warmup < 0:
        parser.error("--perf-warmup must be nonnegative")
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.npu.set_device(args.device)
    cases = (
        _CASES
        if args.cases
        else (
            _Case(
                args.batch,
                args.total_q,
                args.kv_len,
                args.heads,
                args.dtype,
                args.layout,
                args.mask,
            ),
        )
    )
    failed = 0
    for case in cases:
        result = _run_case(case, args)
        failed += 1 if result["passed"] is False else 0
        print(json.dumps(result), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
