# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Fused MoE gate + combine in a single kernel::

    p[b, :]          = softmax(logits[b, :])
    w[b, j], e[b, j] = j-th largest p[b, :] and its expert id
    w[b, :]         /= sum_j w[b, j]            (when RENORM == 1)
    y[b, :]          = sum_{j < K} w[b, j] * experts[e[b, j], b, :]

一个 AIV 块处理一段连续 token，一个 token 的处理链是
``MTE2(logits) -> VECTOR(softmax+topk) -> 标量(读回 w/idx) -> MTE2(K 条专家行)
-> VECTOR(K 次乘加) -> MTE3(w/idx/y)``。

高性能设计要点：

1. **一个 64-lane 向量装下一个 token 的全部 logits**：行按 ``E_PAD = ceil(E/64)*64``
   补齐留在 UB，E 不是 64 的倍数时只多一条尾块路径，GM 侧始终是稠密的 ``[B, E]``，
   行首字节偏移不要求 32 B 对齐。
2. **两遍定 softmax，省掉一趟写回**：第一遍求行 max，第二遍把 ``exp(x - max)``
   直接原地覆盖 logits，同时累加分母；因为 exp 单调，后面的 top-k 选择直接在
   exp 值上做，不必再单独存一份概率。
3. **选择 = max + lane 位置扫描**：每选出一个 top 值，只做一次全行 max 和一次
   "等于最大值的最小 lane 号"扫描（``min`` 归约），K 个结果打包进同一个 64-lane
   向量的 lane 0..K-1，最后一次 DMA 写回，权重和 expert id 各只写一次。
4. **gate 与 combine 融合，中间结果不出 UB**：top-k 的权重 / id 打包在 UB 的
   64-lane 缓冲里，``local_mem_bar(VEC_STORE, SCALAR_LOAD)`` 之后由标量单元直接
   读回 lane 0..K-1，省掉一次 kernel launch、一次概率写回和一次 host 侧 gather
   索引的 materialize。id 在 kernel 内一次转 int32 再落盘。
5. **数据相关 gather 交给 MTE2**：一个 token 的 K 条专家行在 GM 上互不相邻，但每条
   行本身连续。用标量读回的 ``idx`` 直接算出 ``idx * B + b`` 行的 GM 地址，K 条
   整行 DMA 连续下发；vector 核不做逐元素 4 B GM 读，因此吞吐由 DMA 而不是
   标量 load 决定。
6. **K 条行乘加全程在寄存器里完成**：每个 chunk 读入 K 条行的同一段，累加器留在
   向量寄存器中，只把最终结果写回行 0 的同一段，K 条行只占 K 个 UB 行缓冲、只做
   一次 UB 写回，没有 K 次 UB 往返。
7. **per-token mutex 的多段流水**：``BUFS`` 个 token 槽位各挂一个 mutex，MTE2 搬
   第 t+1 个 token、VECTOR 算第 t 个、MTE3 写第 t-1 个重叠执行；槽位复用由 mutex
   而不是程序序控制，``BUFS`` 按 ``E_PAD + K * D_PAD`` 的 UB 预算自动选 1..4。
8. **尾块只收窄 mask**：E / D 不是 64 的倍数时，最后一段按整 64 lane 计算，只在
   store 时用 ``update_mask`` 收窄；UB 行宽固定 ``E_PAD`` / ``D_PAD``，GM 行宽是
   稠密的 ``E`` / ``D``，所以 E / D 只要求是正整数，不要求 32 B 对齐。
9. **无 padding 干扰**：padding lane 在求 max / 求分母 / 选择前都显式置为 ``-inf``，
   取 exp 后恒为 0，而真实 lane 的 ``exp(x - max)`` 最大值恒为 1，所以 padding
   永远不会被选中（要求 ``K <= E``）。
10. **``experts`` 以 ``[H * B, D]`` 的 2-D 视图传入**（``from_dlpack`` 不支持 3-D
    layout），连续输入下就是零拷贝视图，行号在 kernel 内算出。
"""

import functools
import inspect
import threading

import catlass.tla as tla
from catlass.tla.runtime import _Tensor, from_dlpack

VL_ELE = 64  # fp32 vector length (256 B / 4 B)
MIN_F = -3.4028234663852886e38
POS_INF = 3.0e38
_CAST_NONE = tla.params.CastParams()
UB_BYTES = 262144
MAX_K = VL_ELE


@tla.kernel
def moe_route_kernel(
    gm_logits: tla.Tensor,  # [B, E] fp32
    gm_w: tla.Tensor,  # [B, K] fp32 top-k weights
    gm_idx: tla.Tensor,  # [B, K] int32 expert ids
    gm_e: tla.Tensor,  # [H * B, D] fp32 (experts in token order)
    gm_y: tla.Tensor,  # [B, D] fp32
    B: tla.Constexpr[int],
    E: tla.Constexpr[int],
    E_PAD: tla.Constexpr[int],
    TAIL: tla.Constexpr[int],
    D: tla.Constexpr[int],
    D_PAD: tla.Constexpr[int],
    D_TAIL: tla.Constexpr[int],
    K: tla.Constexpr[int],
    RENORM: tla.Constexpr[int],
    BUFS: tla.Constexpr[int],
) -> None:
    # 每个 in-flight token 一个槽位 + 一个 mutex：槽位里是 logits 行、打包后的
    # top-k 权重 / id、以及 K 条专家输出行；mutex 在 MTE2 / VECTOR / 标量 / MTE2 /
    # VECTOR / MTE3 之间传递所有权，相邻 token 的各段因此可以重叠。
    x_ptr = [None] * BUFS
    v_ptr = [None] * BUFS
    i_ptr = [None] * BUFS
    r_ptr = [None] * BUFS
    tok_mutex = [None] * BUFS
    for bi in tla.range_constexpr(BUFS):
        x_ptr[bi] = tla.allocate(E_PAD, tla.Float32, tla.AddressSpace.ub, 256)
        v_ptr[bi] = tla.allocate(VL_ELE, tla.Float32, tla.AddressSpace.ub, 256)
        i_ptr[bi] = tla.allocate(VL_ELE, tla.Int32, tla.AddressSpace.ub, 256)
        rows = [None] * K
        for j in tla.range_constexpr(K):
            rows[j] = tla.allocate(D_PAD, tla.Float32, tla.AddressSpace.ub, 256)
        r_ptr[bi] = rows
        tok_mutex[bi] = tla.mutex(resource="moe_tok", id=bi)

    row_ub = tla.make_layout(
        tla.make_shape(1, E),
        tla.make_stride(E_PAD, 1),
        origin_shape=tla.make_shape(1, E),
    )
    row_gm = tla.make_layout(
        tla.make_shape(1, E),
        tla.make_stride(E, 1),
        origin_shape=tla.make_shape(1, E),
    )
    pack_ub = tla.make_layout(
        tla.make_shape(1, K),
        tla.make_stride(1, 1),
        origin_shape=tla.make_shape(1, K),
    )
    pack_gm = tla.make_layout(
        tla.make_shape(1, K),
        tla.make_stride(K, 1),
        origin_shape=tla.make_shape(1, K),
    )
    # 专家输出行：UB 侧按 D_PAD（64 lane 对齐）留宽，GM 侧是稠密的 D。
    erow_ub = tla.make_layout(
        tla.make_shape(1, D),
        tla.make_stride(D_PAD, 1),
        origin_shape=tla.make_shape(1, D),
    )
    erow_gm = tla.make_layout(
        tla.make_shape(1, D),
        tla.make_stride(D, 1),
        origin_shape=tla.make_shape(1, D),
    )
    # 打包缓冲的全宽（64 lane）视图：gather 按 lane 号广播 top-k 权重。
    packfull_ub = tla.make_layout(
        tla.make_shape(1, VL_ELE),
        tla.make_stride(VL_ELE, 1),
        origin_shape=tla.make_shape(1, VL_ELE),
    )
    ub_x = [None] * BUFS
    ub_pv = [None] * BUFS
    ub_pi = [None] * BUFS
    ub_pvf = [None] * BUFS
    ub_pif = [None] * BUFS
    ub_r = [None] * BUFS
    for bj in tla.range_constexpr(BUFS):
        ub_x[bj] = tla.make_tensor(x_ptr[bj], row_ub)
        ub_pv[bj] = tla.make_tensor(v_ptr[bj], pack_ub)
        ub_pi[bj] = tla.make_tensor(i_ptr[bj], pack_ub)
        ub_pvf[bj] = tla.make_tensor(v_ptr[bj], packfull_ub)
        ub_pif[bj] = tla.make_tensor(i_ptr[bj], packfull_ub)
        rows = [None] * K
        for j in tla.range_constexpr(K):
            rows[j] = tla.make_tensor(r_ptr[bj][j], erow_ub)
        ub_r[bj] = rows

    n_chunks = E // VL_ELE
    d_chunks = D // VL_ELE
    gid = tla.arch.block_idx()
    aiv_num = tla.arch.block_num()
    # 每个块认领一段连续 token，DMA 走的是相邻 GM 行。
    toks_per_block = (B + aiv_num - 1) // aiv_num
    tok_begin = gid * toks_per_block

    with tla.vector():
        for grp in tla.range((toks_per_block + BUFS - 1) // BUFS):
            for bi in tla.range_constexpr(BUFS):
                r = tok_begin + grp * BUFS + bi
                if r < B:
                    row_off = r.to(tla.Int64) * tla.Int64(E)
                    out_off = r.to(tla.Int64) * tla.Int64(K)
                    gm_row_x = tla.make_tensor(gm_logits.ptr + row_off, row_gm)
                    gm_row_w = tla.make_tensor(gm_w.ptr + out_off, pack_gm)
                    gm_row_i = tla.make_tensor(gm_idx.ptr + out_off, pack_gm)

                    tok_mutex[bi].lock(pipe=tla.pipes.MTE2)
                    tla.copy(ub_x[bi], gm_row_x)
                    tok_mutex[bi].unlock(pipe=tla.pipes.MTE2)

                    tok_mutex[bi].lock(pipe=tla.pipes.VECTOR)
                    with tla.vec.func(mode="simd"):
                        mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                        if tla.const_expr(TAIL != 0):
                            tail_mask, _ = tla.update_mask(TAIL, dtype=tla.Float32)
                        else:
                            tail_mask = mask

                        lane_i = tla.arange(base=0, order="increase", dtype=tla.Int32)
                        lane_f = lane_i.to(tla.Float32, _CAST_NONE, mask)
                        pad_lane = tla.cmp(lane_f, float(TAIL), "ge")

                        # ---- softmax 分母：行 max，再累加 exp ----
                        mx = tla.full(MIN_F, dtype=tla.Float32)
                        for c1 in tla.range(n_chunks):
                            v1 = tla.tile_view(
                                ub_x[bi],
                                tla.make_shape(1, VL_ELE),
                                tla.make_coord(0, c1),
                            ).load()
                            mx = tla.max(mx, v1, mask=mask)
                        tmx = tla.full(MIN_F, dtype=tla.Float32)
                        if tla.const_expr(TAIL != 0):
                            t1 = tla.tile_view(
                                ub_x[bi],
                                tla.make_shape(1, VL_ELE),
                                tla.make_coord(0, n_chunks),
                            ).load()
                            t1 = tla.where(
                                pad_lane, tla.full(MIN_F, dtype=tla.Float32), t1
                            )
                            tmx = tla.max(tmx, t1, mask=mask)
                        max_b = tla.max(
                            tla.full(
                                mx.reduce(tla.ReductionOp.MAX, mask=mask),
                                dtype=tla.Float32,
                            ),
                            tla.full(
                                tmx.reduce(tla.ReductionOp.MAX, mask=mask),
                                dtype=tla.Float32,
                            ),
                            mask=mask,
                        )

                        # exp(x - max) 原地覆盖 logits：后面直接在 exp 值上选 top-k。
                        acc = tla.full(0.0, dtype=tla.Float32)
                        for c2 in tla.range(n_chunks):
                            xc2 = tla.tile_view(
                                ub_x[bi],
                                tla.make_shape(1, VL_ELE),
                                tla.make_coord(0, c2),
                            )
                            e2 = tla.exp(
                                tla.sub(xc2.load(), max_b, mask=mask), mask=mask
                            )
                            xc2.store(e2)
                            acc = tla.add(acc, e2, mask=mask)
                        tacc = tla.full(0.0, dtype=tla.Float32)
                        if tla.const_expr(TAIL != 0):
                            xt2 = tla.tile_view(
                                ub_x[bi],
                                tla.make_shape(1, VL_ELE),
                                tla.make_coord(0, n_chunks),
                            )
                            e2t = tla.exp(tla.sub(t1, max_b, mask=mask), mask=mask)
                            # t1 的 padding lane 已是 -inf（exp 后为 0），这里再显式
                            # 清零，避免 ``-inf - max`` 在极端数值下产生非零值污染分母。
                            e2t = tla.where(
                                pad_lane, tla.full(0.0, dtype=tla.Float32), e2t
                            )
                            xt2.store(e2t)
                            tacc = tla.add(tacc, e2t, mask=mask)
                        den = tla.add(
                            tla.full(
                                acc.reduce(tla.ReductionOp.ADD, mask=mask),
                                dtype=tla.Float32,
                            ),
                            tla.full(
                                tacc.reduce(tla.ReductionOp.ADD, mask=mask),
                                dtype=tla.Float32,
                            ),
                            mask=mask,
                        )
                        inv = tla.div(tla.full(1.0, dtype=tla.Float32), den, mask=mask)
                        tla.local_mem_bar(
                            tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                        )

                        # ---- K 次选择：一次全行 max + 一次 lane 位置扫描 ----
                        packed_v = tla.full(0.0, dtype=tla.Float32)
                        packed_i = tla.full(0.0, dtype=tla.Float32)
                        for kk in tla.range_constexpr(K):
                            pmx = tla.full(MIN_F, dtype=tla.Float32)
                            for c3 in tla.range(n_chunks):
                                v3 = tla.tile_view(
                                    ub_x[bi],
                                    tla.make_shape(1, VL_ELE),
                                    tla.make_coord(0, c3),
                                ).load()
                                pmx = tla.max(pmx, v3, mask=mask)
                            t3 = tla.full(MIN_F, dtype=tla.Float32)
                            if tla.const_expr(TAIL != 0):
                                tv3 = tla.tile_view(
                                    ub_x[bi],
                                    tla.make_shape(1, VL_ELE),
                                    tla.make_coord(0, n_chunks),
                                ).load()
                                # 尾块的 padding lane 是占位的 0（不是真实概率），显式置
                                # 成 -inf，保证它既不会被选成 max，也不会被位置扫描命中。
                                tv3 = tla.where(
                                    pad_lane, tla.full(MIN_F, dtype=tla.Float32), tv3
                                )
                                t3 = tla.max(t3, tv3, mask=mask)
                            top_b = tla.max(
                                tla.full(
                                    pmx.reduce(tla.ReductionOp.MAX, mask=mask),
                                    dtype=tla.Float32,
                                ),
                                tla.full(
                                    t3.reduce(tla.ReductionOp.MAX, mask=mask),
                                    dtype=tla.Float32,
                                ),
                                mask=mask,
                            )

                            pos = tla.full(POS_INF, dtype=tla.Float32)
                            for c4 in tla.range(n_chunks):
                                v4 = tla.tile_view(
                                    ub_x[bi],
                                    tla.make_shape(1, VL_ELE),
                                    tla.make_coord(0, c4),
                                ).load()
                                lc4 = tla.arange(
                                    base=c4 * VL_ELE, order="increase", dtype=tla.Int32
                                ).to(tla.Float32, _CAST_NONE, mask)
                                pos = tla.min(
                                    pos,
                                    tla.where(
                                        tla.cmp(v4, top_b, "eq"),
                                        lc4,
                                        tla.full(POS_INF, dtype=tla.Float32),
                                    ),
                                    mask=mask,
                                )
                            tpos = tla.full(POS_INF, dtype=tla.Float32)
                            if tla.const_expr(TAIL != 0):
                                tv4 = tla.tile_view(
                                    ub_x[bi],
                                    tla.make_shape(1, VL_ELE),
                                    tla.make_coord(0, n_chunks),
                                ).load()
                                tv4 = tla.where(
                                    pad_lane, tla.full(MIN_F, dtype=tla.Float32), tv4
                                )
                                lc4t = tla.arange(
                                    base=n_chunks * VL_ELE,
                                    order="increase",
                                    dtype=tla.Int32,
                                ).to(tla.Float32, _CAST_NONE, mask)
                                tpos = tla.min(
                                    tpos,
                                    tla.where(
                                        tla.cmp(tv4, top_b, "eq"),
                                        lc4t,
                                        tla.full(POS_INF, dtype=tla.Float32),
                                    ),
                                    mask=mask,
                                )
                            idx_b = tla.min(
                                tla.full(
                                    pos.reduce(tla.ReductionOp.MIN, mask=mask),
                                    dtype=tla.Float32,
                                ),
                                tla.full(
                                    tpos.reduce(tla.ReductionOp.MIN, mask=mask),
                                    dtype=tla.Float32,
                                ),
                                mask=mask,
                            )

                            sel = tla.cmp(lane_f, float(kk), "eq")
                            packed_v = tla.where(
                                sel,
                                tla.full(
                                    top_b.reduce(tla.ReductionOp.MAX, mask=mask),
                                    dtype=tla.Float32,
                                ),
                                packed_v,
                            )
                            packed_i = tla.where(
                                sel,
                                tla.full(
                                    idx_b.reduce(tla.ReductionOp.MIN, mask=mask),
                                    dtype=tla.Float32,
                                ),
                                packed_i,
                            )

                            # 抹掉本轮选中的那一个 lane（按 lane 号 idx_b，而不是按取值），
                            # 下一轮选次大。按取值抹会把所有并列的最大值一起抹掉，出
                            # 现重复概率时后续几轮就会漏掉重复项、top-k 集合因此改变；
                            # 按 lane 号精确抹一个，并列时下一轮自然选中同值的另一个 lane
                            # （lane 号最小的优先），与稳定排序语义一致。
                            for c5 in tla.range(n_chunks):
                                xc5 = tla.tile_view(
                                    ub_x[bi],
                                    tla.make_shape(1, VL_ELE),
                                    tla.make_coord(0, c5),
                                )
                                v5 = xc5.load()
                                lc5 = tla.arange(
                                    base=c5 * VL_ELE, order="increase", dtype=tla.Int32
                                ).to(tla.Float32, _CAST_NONE, mask)
                                xc5.store(
                                    tla.where(
                                        tla.cmp(lc5, idx_b, "eq"),
                                        tla.full(MIN_F, dtype=tla.Float32),
                                        v5,
                                    ),
                                    mask=mask,
                                )
                            if tla.const_expr(TAIL != 0):
                                xt5 = tla.tile_view(
                                    ub_x[bi],
                                    tla.make_shape(1, VL_ELE),
                                    tla.make_coord(0, n_chunks),
                                )
                                v5t = xt5.load()
                                lc5t = tla.arange(
                                    base=n_chunks * VL_ELE,
                                    order="increase",
                                    dtype=tla.Int32,
                                ).to(tla.Float32, _CAST_NONE, mask)
                                xt5.store(
                                    tla.where(
                                        tla.cmp(lc5t, idx_b, "eq"),
                                        tla.full(MIN_F, dtype=tla.Float32),
                                        v5t,
                                    ),
                                    mask=tail_mask,
                                )
                            tla.local_mem_bar(
                                tla.params.MemType.VEC_STORE,
                                tla.params.MemType.VEC_LOAD,
                            )

                        # 只归一化打包后的 K 个权重：默认全 softmax，或 RENORM 时把
                        # top-k 权重重新归一到 1。
                        if tla.const_expr(RENORM == 1):
                            ksel = tla.where(
                                tla.cmp(lane_f, float(K), "lt"),
                                packed_v,
                                tla.full(0.0, dtype=tla.Float32),
                            )
                            ksum = tla.full(
                                ksel.reduce(tla.ReductionOp.ADD, mask=mask),
                                dtype=tla.Float32,
                            )
                            inv = tla.div(
                                tla.full(1.0, dtype=tla.Float32), ksum, mask=mask
                            )
                        packed_v = tla.mul(packed_v, inv, mask=mask)

                        tla.tile_view(
                            ub_pv[bi], tla.make_shape(1, VL_ELE), tla.make_coord(0, 0)
                        ).store(packed_v)
                        # id 在 kernel 内直接转 int32（一次转换代替 K 次），
                        # 省掉 host 侧的一次 device cast。
                        tla.tile_view(
                            ub_pif[bi], tla.make_shape(1, VL_ELE), tla.make_coord(0, 0)
                        ).store(packed_i.to(tla.Int32, _CAST_NONE, mask))
                        # 打包结果下一步要被标量单元读出来算专家行地址，先保证
                        # vector store 对后续标量读可见。
                        tla.local_mem_bar(
                            tla.params.MemType.VEC_STORE, tla.params.MemType.SCALAR_LOAD
                        )
                    tok_mutex[bi].unlock(pipe=tla.pipes.VECTOR)

                    # ---- 标量段：读回 top-k id，用 id 直接算专家行的 GM 地址。
                    # 数据相关的 gather 全部交给 MTE2 的整行 DMA，vector 核不做
                    # 逐元素 4 B GM 读。锁在 SCALAR 上，保证标量读等的是上一段
                    # vector store 完成；K 条 DMA 依赖标量读结果，解锁放在 MTE2
                    # 上，下一段 VECTOR 因此等的是 DMA 落地。
                    tok_mutex[bi].lock(pipe=tla.pipes.SCALAR)
                    for j in tla.range_constexpr(K):
                        src = ub_pi[bi][0, j].to(tla.Int64) * tla.Int64(B) + r.to(
                            tla.Int64
                        )
                        gm_row_e = tla.make_tensor(
                            gm_e.ptr + src * tla.Int64(D), erow_gm
                        )
                        tla.copy(ub_r[bi][j], gm_row_e)
                    tok_mutex[bi].unlock(pipe=tla.pipes.MTE2)

                    # ---- 加权累加：每个 chunk 的累加器留在寄存器里，K 条行读进来
                    # 乘加后只写回行 0，避免 K 次 UB 往返；权重用 gather 广播成
                    # 全 lane 向量（lane j 即 w[b, j]）。
                    tok_mutex[bi].lock(pipe=tla.pipes.VECTOR)
                    with tla.vec.func(mode="simd"):
                        # 上一段的标量读必须早于本段（以及后面 token）的 vector store。
                        tla.local_mem_bar(
                            tla.params.MemType.SCALAR_LOAD, tla.params.MemType.VEC_STORE
                        )
                        dmask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                        if tla.const_expr(D_TAIL != 0):
                            dtail_mask, _ = tla.update_mask(D_TAIL, dtype=tla.Float32)
                        else:
                            dtail_mask = dmask
                        pvv = ub_pvf[bi]
                        # 全 0 的 lane 索引：gather 把 lane 0 广播到全部 lane。
                        lane_i = tla.arange(base=0, order="increase", dtype=tla.Int32)
                        zero_i = tla.sub(lane_i, lane_i)
                        wbv = tla.gather(pvv, zero_i, mask=dmask)

                        for c in tla.range_constexpr(d_chunks):
                            bt0 = tla.tile_view(
                                ub_r[bi][0],
                                tla.make_shape(1, VL_ELE),
                                tla.make_coord(0, c),
                            )
                            bacc = tla.mul(bt0.load(), wbv, mask=dmask)
                            for j in tla.range_constexpr(K - 1):
                                btj = tla.tile_view(
                                    ub_r[bi][j + 1],
                                    tla.make_shape(1, VL_ELE),
                                    tla.make_coord(0, c),
                                )
                                bacc = tla.add(
                                    bacc,
                                    tla.mul(
                                        btj.load(),
                                        tla.gather(
                                            pvv, tla.add(zero_i, j + 1), mask=dmask
                                        ),
                                        mask=dmask,
                                    ),
                                    mask=dmask,
                                )
                            bt0.store(bacc, mask=dmask)

                        # 尾块整 64 lane 计算、只落盘前 D_TAIL 个 lane。
                        if tla.const_expr(D_TAIL != 0):
                            tt0 = tla.tile_view(
                                ub_r[bi][0],
                                tla.make_shape(1, VL_ELE),
                                tla.make_coord(0, d_chunks),
                            )
                            tacc = tla.mul(tt0.load(), wbv, mask=dmask)
                            for j in tla.range_constexpr(K - 1):
                                ttj = tla.tile_view(
                                    ub_r[bi][j + 1],
                                    tla.make_shape(1, VL_ELE),
                                    tla.make_coord(0, d_chunks),
                                )
                                tacc = tla.add(
                                    tacc,
                                    tla.mul(
                                        ttj.load(),
                                        tla.gather(
                                            pvv, tla.add(zero_i, j + 1), mask=dmask
                                        ),
                                        mask=dmask,
                                    ),
                                    mask=dmask,
                                )
                            tt0.store(tacc, mask=dtail_mask)
                    tok_mutex[bi].unlock(pipe=tla.pipes.VECTOR)

                    tok_mutex[bi].lock(pipe=tla.pipes.MTE3)
                    gm_row_y = tla.make_tensor(
                        gm_y.ptr + r.to(tla.Int64) * tla.Int64(D), erow_gm
                    )
                    tla.copy(gm_row_y, ub_r[bi][0])
                    tla.copy(gm_row_w, ub_pv[bi])
                    tla.copy(gm_row_i, ub_pi[bi])
                    tok_mutex[bi].unlock(pipe=tla.pipes.MTE3)

        tla.pipe_barrier(tla.pipes.ALL)


# ======================================================================================
# HOST side: shape checks, compile and launch.
# ======================================================================================
# 与 scan 相同的三级缓存（编译产物 / host 张量绑定 / 预打包 payload），小 shape 的
# 端到端耗时因此由一次字典查找 + 一次 launch 组成，而不是每次重新 from_dlpack 和
# 打包参数。

_COMPILED = {}
_VECTOR_CORES = {}
_BINDINGS = {}
_FAST = {}
_MAX_BINDINGS = 4096
_MAX_FAST = 512

_TORCH_DTYPES = {}


def _vector_cores(index):
    cores = _VECTOR_CORES.get(index)
    if cores is None:
        import torch

        cores = int(torch.npu.get_device_properties(index).vector_core_num)
        _VECTOR_CORES[index] = cores
    return cores


def _bind(buf, rows, cols, dtype):
    key = (buf.data_ptr(), rows, cols, dtype)
    view = _BINDINGS.get(key)
    if view is not None:
        return view
    tla_dtype = tla.Float32 if dtype == "f32" else tla.Int32
    view = _Tensor(
        (rows, cols),
        tla_dtype,
        origin_shape=(rows, cols),
        coord=(0, 0),
        stride=(cols, 1),
        layout_tag=tla.arch.RowMajor,
        data_ptr=buf.data_ptr(),
    )
    if len(_BINDINGS) < _MAX_BINDINGS:
        _BINDINGS[key] = view
    return view


_STREAMS = {}
_LAUNCH_ACCEPTS_PRINT = None


def reset_stream_cache():
    _STREAMS.clear()
    _FAST.clear()


def _launch_stream(index):
    key = (threading.get_ident(), index)
    stream = _STREAMS.get(key)
    if stream is None:
        import torch

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


def _pipeline(row_elems, extra_elems=0):
    per_buf = 4 * row_elems + 4 * extra_elems
    return max(1, min(4, (UB_BYTES - 8192) // per_buf))


def _lazy_fast(key, launch, artifact, tensors, cores, stream):
    """Cache ``(payload launcher, artifact, host tensors, cores, stream)``."""
    plan = (launch, artifact, tensors, cores, stream)
    if len(_FAST) >= _MAX_FAST:
        _FAST.clear()
    _FAST[key] = plan
    return plan


def _call_plan(plan):
    launch = plan[0]
    if launch is not None:
        launch()
    else:
        plan[1](*plan[2], block_num=plan[3], stream=plan[4])


def run(logits, experts, k=2, renorm=True):
    """Fused gate + combine, one kernel launch: returns ``(y, w, idx)``.

    ``w / idx`` are the top-``k`` softmax weights (fp32) and expert ids (int32) of
    every token, and ``y[b, :] = sum_j w[b, j] * experts[idx[b, j], b, :]``.
    """
    import torch

    if logits.dtype is not torch.float32 or logits.dim() != 2:
        raise ValueError("logits must be a 2-D float32 tensor")
    if experts.dtype is not torch.float32 or experts.dim() != 3:
        raise ValueError("experts must be a 3-D [H, B, D] float32 tensor")
    if not 1 <= k <= MAX_K:
        raise ValueError("k must be in [1, {}]".format(MAX_K))
    if k > logits.shape[1]:
        raise ValueError("k must be <= E (= logits.shape[1])")
    b, e = logits.shape
    h, eb, d = experts.shape
    if eb != b:
        raise ValueError("experts must be [H, B, D] with B == logits.shape[0]")
    if h < e:
        raise ValueError(
            "experts needs H >= E (= logits.shape[1]); top-k ids index [0, E)"
        )
    renorm_flag = 1 if renorm else 0

    if not logits.is_contiguous():
        logits = logits.contiguous()
    # [H, B, D] -> [H * B, D]：连续输入下是零拷贝视图（from_dlpack 不支持 3-D）。
    e2 = experts.reshape(h * b, d)
    if not e2.is_contiguous():
        e2 = e2.contiguous()

    device = logits.device
    y = torch.empty((b, d), dtype=torch.float32, device=device)
    w = torch.empty((b, k), dtype=torch.float32, device=device)
    idx = torch.empty((b, k), dtype=torch.int32, device=device)
    if b == 0:
        return y, w, idx

    e_pad = (e + VL_ELE - 1) // VL_ELE * VL_ELE
    e_tail = e - (e // VL_ELE) * VL_ELE
    d_pad = (d + VL_ELE - 1) // VL_ELE * VL_ELE
    d_tail = d - (d // VL_ELE) * VL_ELE
    # 每个 token 槽位 = logits 行 + K 条专家行 + 打包权重 / id。
    bufs = _pipeline(e_pad, k * d_pad + 2 * VL_ELE)

    key = (
        "moe",
        logits.data_ptr(),
        e2.data_ptr(),
        y.data_ptr(),
        w.data_ptr(),
        idx.data_ptr(),
        b,
        e,
        k,
        d,
        renorm_flag,
    )
    plan = _FAST.get(key)
    if plan is not None:
        _call_plan(plan)
        return y, w, idx

    compile_key = (b, e, k, d, renorm_flag)
    artifact = _COMPILED.get(compile_key)
    if artifact is None:
        artifact = tla.compile(
            moe_route_kernel,
            _bind(logits, b, e, "f32"),
            _bind(w, b, k, "f32"),
            _bind(idx, b, k, "i32"),
            _bind(e2, h * b, d, "f32"),
            _bind(y, b, d, "f32"),
            b,
            e,
            e_pad,
            e_tail,
            d,
            d_pad,
            d_tail,
            k,
            renorm_flag,
            bufs,
            options="--npu-arch 3510",
        )
        _COMPILED[compile_key] = artifact

    index = device.index
    cores = _vector_cores(index)
    stream = _launch_stream(index)
    tensors = (
        _bind(logits, b, e, "f32"),
        _bind(w, b, k, "f32"),
        _bind(idx, b, k, "i32"),
        _bind(e2, h * b, d, "f32"),
        _bind(y, b, d, "f32"),
    )
    plan = _lazy_fast(
        key,
        _make_launcher(artifact, tensors, cores, stream),
        artifact,
        tensors,
        cores,
        stream,
    )
    _call_plan(plan)
    return y, w, idx


if __name__ == "__main__":
    import torch_npu  # noqa: F401
    import torch

    torch.npu.set_device(0)
    torch.manual_seed(0)

    # E / D 覆盖 64 整除、非整除、小于一个向量；B 覆盖核数非整除。
    for b, e, k, d, h in (
        (64, 256, 2, 128, 256),
        (37, 1000, 4, 100, 1000),
        (16, 2048, 8, 256, 2048),
        (5, 63, 1, 96, 63),
        (9, 65, 2, 1, 65),
        (129, 100, 3, 63, 100),
    ):
        logits = torch.randn(b, e, dtype=torch.float32, device="npu")
        experts = torch.randn(h, b, d, dtype=torch.float32, device="npu")
        for renorm in (True, False):
            y, w, idx = run(logits, experts, k, renorm)
            p = torch.softmax(logits.cpu(), dim=-1)
            ref_w, ref_i = torch.topk(p, k, dim=-1)
            if renorm:
                ref_w = ref_w / ref_w.sum(dim=-1, keepdim=True)
            sel = experts.cpu().permute(1, 0, 2)
            rows = torch.arange(b).unsqueeze(1)
            ref_y = (ref_w.unsqueeze(-1) * sel[rows, ref_i]).sum(dim=1)
            err_w = (w.cpu() - ref_w).abs().max().item()
            err_i = (idx.cpu() - ref_i.to(torch.int32)).abs().max().item()
            err_y = (y.cpu() - ref_y).abs().max().item()
            print(
                "moe",
                (b, e, k, d, h),
                "renorm",
                renorm,
                "w err",
                err_w,
                "idx err",
                err_i,
                "y err",
                err_y,
            )
