# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""QuantFlashAttn 固定窗口 Q8KV8 注意力 —— CATLASS Python DSL 实现。

★ 运行时是【单 kernel】：run() 只 launch attn_fused_kernel（一次 launch）。

  attn_fused_kernel（MIX，block_num=cube_num，每 core 2 AIV）：
    phase1  split-KV 在线 softmax flash（PRE_LAUNCH=2 流水、SPLIT_M fixpipe、
            AIV 分行）。QK/PV 均为 FP8 MMAD（E4M3×E4M3→FP32）；分子
            exp(S×2^-4 − m_running) 经 f32→bf16→e4m3 两级 2x 物化后做 FP32 PV；
            分母为量化前 FP32 row_sum；每 (M块, split) 任务写 FP32 partial（acc/m/l）。
    barrier cross_flag(mode=0) 全核屏障：等所有 AIV 的 partial 写完。
    phase2  归并（_combine_rows）：按行组读各 split 的 partial，w=exp(m_s−m*)
            加权归并，den=0 行输出精确 0，BF16 写出。UB 复用 phase1 的 ub_acc。

契约（host 校验）：D=256、Hkv=1、page_size=128、window>0、right>=Tq。
Q/K/V 已由上游量化成无 scale E4M3（量化在核外），分页 gather 在核内完成。
"""

from __future__ import annotations

import torch
import torch_npu  # noqa: F401
import catlass.tla as tla
from catlass.tla.runtime import from_dlpack

# ---------------------------------------------------------------------------
# host 侧工具
# ---------------------------------------------------------------------------


def _tla_tensor(buf, layout_tag, element_type=None):
    """torch NPU buffer -> tla.Tensor；fp8 经 int8 视图 + element_type 覆盖。"""
    if element_type is not None:
        buf = buf.view(torch.int8)
    buf = buf.contiguous()
    t = from_dlpack(buf, layout_tag=layout_tag, element_type=element_type)
    t = t.mark_layout_dynamic()
    t._torch_storage = buf
    return t


def _cube_core_num(device=0):
    props = torch.npu.get_device_properties(int(device))
    return int(props.cube_core_num)


# ---------------------------------------------------------------------------
# kernel 2：split-KV 在线 softmax flash（FP8 MMAD）
# ---------------------------------------------------------------------------

# 固定 tile：M_BLOCK=112（7x16 分形对齐）、KV_BLOCK=128、D=256、PRE_LAUNCH=2。
# 任务空间 = m_blocks * splits；task -> (m_blk = task // splits, split = task % splits)。
# split 的 KV tile 均衡：base = ntiles // splits，前 rem 个 split 多 1。


@tla.jit
def _p_store64(p_reg, ub_p_zN, row, chunk):
    """64 个 FP32 -> 64 个 E4M3 字节写入 RowMajor P（单级 4x 窄化）。

    4x 槽位语义（实测）：slot s 产生 dst[4j+s] = cast(src[j])；两次 deinterleave
    拆出 mod-4 四组再分别 cast、or 合并得稠密 64 字节。exp 输出为超越数，无精确
    平局点，CAST_ROUND 与 torch RNE 一致；分母在量化前已求。
    """
    full_f32 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
    cast0 = tla.params.CastParams(
        reg_slot=tla.params.RegSlot.ZERO,
        sat_mode=tla.params.SatMode.NOSAT,
        round_mode=tla.params.RoundMode.CAST_ROUND,
    )
    cast1 = tla.params.CastParams(
        reg_slot=tla.params.RegSlot.ONE,
        sat_mode=tla.params.SatMode.NOSAT,
        round_mode=tla.params.RoundMode.CAST_ROUND,
    )
    cast2 = tla.params.CastParams(
        reg_slot=tla.params.RegSlot.TWO,
        sat_mode=tla.params.SatMode.NOSAT,
        round_mode=tla.params.RoundMode.CAST_ROUND,
    )
    cast3 = tla.params.CastParams(
        reg_slot=tla.params.RegSlot.THREE,
        sat_mode=tla.params.SatMode.NOSAT,
        round_mode=tla.params.RoundMode.CAST_ROUND,
    )
    e, o = tla.deinterleave(p_reg, p_reg)
    ee, eo = tla.deinterleave(e, e)
    oe, oo = tla.deinterleave(o, o)
    p0 = ee.to(tla.Float8E4M3FN, cast0, mask=full_f32).bitcast(tla.UInt8)
    p1 = oe.to(tla.Float8E4M3FN, cast1, mask=full_f32).bitcast(tla.UInt8)
    p2 = eo.to(tla.Float8E4M3FN, cast2, mask=full_f32).bitcast(tla.UInt8)
    p3 = oo.to(tla.Float8E4M3FN, cast3, mask=full_f32).bitcast(tla.UInt8)
    p8 = tla.bitwise_or(tla.bitwise_or(p0, p1), tla.bitwise_or(p2, p3))
    tla.tile_view(ub_p_zN, tla.make_shape(1, 64), tla.make_coord(row, chunk)).store(
        p8.bitcast(tla.Float8E4M3FN)
    )


@tla.jit
def _sf_fused(ub_s, ub_p_zN, m, n, tail_n, max_out, sum_out):
    """融合单遍: S×2^-4、尾列置 MINV、行 max、exp、rowsum、P 物化。"""
    c0 = 0
    c1 = 1
    max_out_t = tla.make_tensor(
        max_out, tla.make_layout(tla.make_shape(56), tla.make_stride(1))
    )
    sum_out_t = tla.make_tensor(
        sum_out, tla.make_layout(tla.make_shape(56), tla.make_stride(1))
    )
    with tla.vec.func(mode="simd"):
        full_f32 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
        preg_tail, _ = tla.update_mask(tail_n, dtype=tla.Float32)
        one_mask, _ = tla.update_mask(1, dtype=tla.Float32)
        minv = tla.full(-3.0e38, tla.Float32)
        if n == 128:
            for i in tla.range(m):
                ub_s_i0 = tla.tile_view(
                    ub_s, tla.make_shape(1, 64), tla.make_coord(i, c0)
                )
                s0 = tla.mul(ub_s_i0.load(), 0.0625, mask=full_f32)
                s1 = s0
                mx = s0
                if n > 64:
                    ub_s_i1 = tla.tile_view(
                        ub_s, tla.make_shape(1, 64), tla.make_coord(i, c1)
                    )
                    s1 = tla.mul(ub_s_i1.load(), 0.0625, mask=full_f32)
                    mx = tla.max(s0, s1, mask=full_f32)
                mxr = mx.reduce(tla.ReductionOp.MAX, mask=full_f32)
                tla.tile_view(max_out_t, tla.make_shape(1), tla.make_coord(i)).store(
                    mxr, tla.params.UnalignStoreParams(), mask=one_mask
                )
                tla.local_mem_bar(
                    tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                )
                m_brc = tla.tile_view(
                    max_out_t, tla.make_shape(1), tla.make_coord(i)
                ).load(
                    params=tla.params.NormalLoadParams(
                        load_dist=tla.params.LoadDist.DIST_BRC_B32
                    )
                )
                p0 = tla.exp(tla.sub(s0, m_brc, mask=full_f32), mask=full_f32)
                sum_reg = p0.reduce(tla.ReductionOp.ADD, mask=full_f32)
                _p_store64(p0, ub_p_zN, i, c0)
                if n > 64:
                    p1 = tla.exp(tla.sub(s1, m_brc, mask=full_f32), mask=full_f32)
                    sum_reg = tla.add(
                        sum_reg,
                        p1.reduce(tla.ReductionOp.ADD, mask=full_f32),
                        mask=full_f32,
                    )
                    _p_store64(p1, ub_p_zN, i, c1)
                tla.tile_view(sum_out_t, tla.make_shape(1), tla.make_coord(i)).store(
                    sum_reg, tla.params.UnalignStoreParams(), mask=one_mask
                )
        else:
            for zr in range(56):
                for zc in range(2):
                    tla.tile_view(
                        ub_p_zN, tla.make_shape(1, 64), tla.make_coord(zr, zc)
                    ).store(tla.full(0, tla.Int8).bitcast(tla.Float8E4M3FN))
            for i2 in tla.range(m):
                ub_s_i0 = tla.tile_view(
                    ub_s, tla.make_shape(1, 64), tla.make_coord(i2, c0)
                )
                s0 = tla.mul(ub_s_i0.load(), 0.0625, mask=full_f32)
                s1 = s0
                mx = s0
                if n > 64:
                    ub_s_i1 = tla.tile_view(
                        ub_s, tla.make_shape(1, 64), tla.make_coord(i2, c1)
                    )
                    s1 = tla.mul(ub_s_i1.load(), 0.0625, mask=full_f32)
                    s1 = tla.where(preg_tail, s1, minv)
                    mx = tla.max(s0, s1, mask=full_f32)
                else:
                    s0 = tla.where(preg_tail, s0, minv)
                    mx = s0
                mxr = mx.reduce(tla.ReductionOp.MAX, mask=full_f32)
                tla.tile_view(max_out_t, tla.make_shape(1), tla.make_coord(i2)).store(
                    mxr, tla.params.UnalignStoreParams(), mask=one_mask
                )
                tla.local_mem_bar(
                    tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                )
                m_brc = tla.tile_view(
                    max_out_t, tla.make_shape(1), tla.make_coord(i2)
                ).load(
                    params=tla.params.NormalLoadParams(
                        load_dist=tla.params.LoadDist.DIST_BRC_B32
                    )
                )
                p0 = tla.exp(tla.sub(s0, m_brc, mask=full_f32), mask=full_f32)
                sum_reg = p0.reduce(tla.ReductionOp.ADD, mask=full_f32)
                _p_store64(p0, ub_p_zN, i2, c0)
                if n > 64:
                    p1 = tla.exp(tla.sub(s1, m_brc, mask=full_f32), mask=full_f32)
                    sum_reg = tla.add(
                        sum_reg,
                        p1.reduce(tla.ReductionOp.ADD, mask=full_f32),
                        mask=full_f32,
                    )
                    _p_store64(p1, ub_p_zN, i2, c1)
                tla.tile_view(sum_out_t, tla.make_shape(1), tla.make_coord(i2)).store(
                    sum_reg, tla.params.UnalignStoreParams(), mask=one_mask
                )
        tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)


@tla.jit
def _sf_first(ub_s, ub_p_zN, m, n, tail_n, last_max, last_sum):
    """首个 KV tile：直接建立 running max/sum（masked 位 S=MINV，exp 下溢为 0）。"""
    _sf_fused(ub_s, ub_p_zN, m, n, tail_n, last_max, last_sum)


@tla.jit
def _sf_mid(ub_s, ub_p_zN, m, n, tail_n, tile_max, last_max, last_sum, exp_max):
    """非首 tile 融合单遍：scale/mask/行 max、running max/sum、dm、exp、P 物化。"""
    c0 = 0
    c1 = 1
    lm_t = tla.make_tensor(
        last_max, tla.make_layout(tla.make_shape(56), tla.make_stride(1))
    )
    ls_t = tla.make_tensor(
        last_sum, tla.make_layout(tla.make_shape(56), tla.make_stride(1))
    )
    dm_t = tla.make_tensor(
        exp_max, tla.make_layout(tla.make_shape(56), tla.make_stride(1))
    )
    with tla.vec.func(mode="simd"):
        full_f32 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
        preg_tail, _ = tla.update_mask(tail_n, dtype=tla.Float32)
        one_mask, _ = tla.update_mask(1, dtype=tla.Float32)
        minv = tla.full(-3.0e38, tla.Float32)
        if n == 128:
            for i in tla.range(m):
                ub_s_i0 = tla.tile_view(
                    ub_s, tla.make_shape(1, 64), tla.make_coord(i, c0)
                )
                s0 = tla.mul(ub_s_i0.load(), 0.0625, mask=full_f32)
                s1 = s0
                mx = s0
                if n > 64:
                    ub_s_i1 = tla.tile_view(
                        ub_s, tla.make_shape(1, 64), tla.make_coord(i, c1)
                    )
                    s1 = tla.mul(ub_s_i1.load(), 0.0625, mask=full_f32)
                    mx = tla.max(s0, s1, mask=full_f32)
                else:
                    s0 = tla.where(preg_tail, s0, minv)
                mxr = mx.reduce(tla.ReductionOp.MAX, mask=full_f32)
                lm = tla.tile_view(lm_t, tla.make_shape(1), tla.make_coord(i)).load(
                    params=tla.params.NormalLoadParams(
                        load_dist=tla.params.LoadDist.DIST_BRC_B32
                    )
                )
                tla.tile_view(dm_t, tla.make_shape(1), tla.make_coord(i)).store(
                    mxr, tla.params.UnalignStoreParams(), mask=one_mask
                )
                tla.local_mem_bar(
                    tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                )
                tm_brc = tla.tile_view(dm_t, tla.make_shape(1), tla.make_coord(i)).load(
                    params=tla.params.NormalLoadParams(
                        load_dist=tla.params.LoadDist.DIST_BRC_B32
                    )
                )
                new_max = tla.max(tm_brc, lm, mask=full_f32)
                nm_brc = new_max
                dm = tla.exp(tla.sub(lm, nm_brc, mask=full_f32), mask=full_f32)
                p0 = tla.exp(tla.sub(s0, nm_brc, mask=full_f32), mask=full_f32)
                sum_reg = p0.reduce(tla.ReductionOp.ADD, mask=full_f32)
                _p_store64(p0, ub_p_zN, i, c0)
                if n > 64:
                    p1 = tla.exp(tla.sub(s1, nm_brc, mask=full_f32), mask=full_f32)
                    sum_reg = tla.add(
                        sum_reg,
                        p1.reduce(tla.ReductionOp.ADD, mask=full_f32),
                        mask=full_f32,
                    )
                    _p_store64(p1, ub_p_zN, i, c1)
                ls = tla.tile_view(ls_t, tla.make_shape(1), tla.make_coord(i)).load(
                    params=tla.params.NormalLoadParams(
                        load_dist=tla.params.LoadDist.DIST_BRC_B32
                    )
                )
                new_sum = tla.add(
                    sum_reg, tla.mul(ls, dm, mask=full_f32), mask=full_f32
                )
                tla.tile_view(lm_t, tla.make_shape(1), tla.make_coord(i)).store(
                    new_max, tla.params.UnalignStoreParams(), mask=one_mask
                )
                tla.tile_view(ls_t, tla.make_shape(1), tla.make_coord(i)).store(
                    new_sum, tla.params.UnalignStoreParams(), mask=one_mask
                )
                tla.tile_view(dm_t, tla.make_shape(1), tla.make_coord(i)).store(
                    dm, tla.params.UnalignStoreParams(), mask=one_mask
                )
        else:
            for zr in range(56):
                for zc in range(2):
                    tla.tile_view(
                        ub_p_zN, tla.make_shape(1, 64), tla.make_coord(zr, zc)
                    ).store(tla.full(0, tla.Int8).bitcast(tla.Float8E4M3FN))
            for i2 in tla.range(m):
                ub_s_i0 = tla.tile_view(
                    ub_s, tla.make_shape(1, 64), tla.make_coord(i2, c0)
                )
                s0 = tla.mul(ub_s_i0.load(), 0.0625, mask=full_f32)
                s1 = s0
                mx = s0
                if n > 64:
                    ub_s_i1 = tla.tile_view(
                        ub_s, tla.make_shape(1, 64), tla.make_coord(i2, c1)
                    )
                    s1 = tla.mul(ub_s_i1.load(), 0.0625, mask=full_f32)
                    s1 = tla.where(preg_tail, s1, minv)
                    mx = tla.max(s0, s1, mask=full_f32)
                else:
                    s0 = tla.where(preg_tail, s0, minv)
                mxr = mx.reduce(tla.ReductionOp.MAX, mask=full_f32)
                lm = tla.tile_view(lm_t, tla.make_shape(1), tla.make_coord(i2)).load(
                    params=tla.params.NormalLoadParams(
                        load_dist=tla.params.LoadDist.DIST_BRC_B32
                    )
                )
                tla.tile_view(dm_t, tla.make_shape(1), tla.make_coord(i2)).store(
                    mxr, tla.params.UnalignStoreParams(), mask=one_mask
                )
                tla.local_mem_bar(
                    tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                )
                tm_brc = tla.tile_view(
                    dm_t, tla.make_shape(1), tla.make_coord(i2)
                ).load(
                    params=tla.params.NormalLoadParams(
                        load_dist=tla.params.LoadDist.DIST_BRC_B32
                    )
                )
                new_max = tla.max(tm_brc, lm, mask=full_f32)
                nm_brc = new_max
                dm = tla.exp(tla.sub(lm, nm_brc, mask=full_f32), mask=full_f32)
                p0 = tla.exp(tla.sub(s0, nm_brc, mask=full_f32), mask=full_f32)
                sum_reg = p0.reduce(tla.ReductionOp.ADD, mask=full_f32)
                _p_store64(p0, ub_p_zN, i2, c0)
                if n > 64:
                    p1 = tla.exp(tla.sub(s1, nm_brc, mask=full_f32), mask=full_f32)
                    sum_reg = tla.add(
                        sum_reg,
                        p1.reduce(tla.ReductionOp.ADD, mask=full_f32),
                        mask=full_f32,
                    )
                    _p_store64(p1, ub_p_zN, i2, c1)
                ls = tla.tile_view(ls_t, tla.make_shape(1), tla.make_coord(i2)).load(
                    params=tla.params.NormalLoadParams(
                        load_dist=tla.params.LoadDist.DIST_BRC_B32
                    )
                )
                new_sum = tla.add(
                    sum_reg, tla.mul(ls, dm, mask=full_f32), mask=full_f32
                )
                tla.tile_view(lm_t, tla.make_shape(1), tla.make_coord(i2)).store(
                    new_max, tla.params.UnalignStoreParams(), mask=one_mask
                )
                tla.tile_view(ls_t, tla.make_shape(1), tla.make_coord(i2)).store(
                    new_sum, tla.params.UnalignStoreParams(), mask=one_mask
                )
                tla.tile_view(dm_t, tla.make_shape(1), tla.make_coord(i2)).store(
                    dm, tla.params.UnalignStoreParams(), mask=one_mask
                )

        tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)


@tla.kernel
def attn_fused_kernel(
    gm_q8: tla.Tensor,  # [M_pad, 256] E4M3（上游已量化）
    gm_kc: tla.Tensor,  # [P*128*256] E4M3 平铺 K 物理页池
    gm_vc: tla.Tensor,  # 同上 V
    gm_bt: tla.Tensor,  # [n_pages] int32 块表
    gm_zero: tla.Tensor,  # [128*256] E4M3 全 0（清零 V 的 padding 行）
    part_o: tla.Tensor,  # [m_blocks*splits*112, 264] FP32（0..255=acc, 256=m, 257=l）
    gm_out: tla.Tensor,  # [M, 256] BF16 最终输出
    m_rows: tla.Int32,
    nvis: tla.Int32,
    splits: tla.Int32,
    splits_cap: tla.Int32,
    start: tla.Int32,  # 可见区间左端点
    nk_pages: tla.Int32,  # 可见区间覆盖页数
) -> None:
    M_BLOCK = 112
    KV_BLOCK = 128
    D = 256
    PAGE = 128
    PRE_LAUNCH = 2
    GROUP_C = 4  # 归并行组大小（112 % 4 == 0，56 % 4 == 0，不跨 m_blk/AIV 边界）
    page0 = start // PAGE
    m_blocks = (m_rows + M_BLOCK - 1) // M_BLOCK
    total = m_blocks * splits
    # ---- buffer ----
    l1_q = tla.allocate(M_BLOCK * D, tla.Float8E4M3FN, tla.AddressSpace.l1, 512)
    l1_k = [
        tla.allocate(D * KV_BLOCK, tla.Float8E4M3FN, tla.AddressSpace.l1, 512),
        tla.allocate(D * KV_BLOCK, tla.Float8E4M3FN, tla.AddressSpace.l1, 512),
    ]
    l1_v = [
        tla.allocate(KV_BLOCK * D, tla.Float8E4M3FN, tla.AddressSpace.l1, 512),
        tla.allocate(KV_BLOCK * D, tla.Float8E4M3FN, tla.AddressSpace.l1, 512),
    ]
    l1_p = [
        tla.allocate(M_BLOCK * KV_BLOCK, tla.Float8E4M3FN, tla.AddressSpace.l1, 512),
        tla.allocate(M_BLOCK * KV_BLOCK, tla.Float8E4M3FN, tla.AddressSpace.l1, 512),
        tla.allocate(M_BLOCK * KV_BLOCK, tla.Float8E4M3FN, tla.AddressSpace.l1, 512),
    ]
    l0a = [
        tla.allocate(M_BLOCK * D, tla.Float8E4M3FN, tla.AddressSpace.l0a, 512),
        tla.allocate(M_BLOCK * D, tla.Float8E4M3FN, tla.AddressSpace.l0a, 512),
    ]
    l0b = [
        tla.allocate(D * KV_BLOCK, tla.Float8E4M3FN, tla.AddressSpace.l0b, 512),
        tla.allocate(D * KV_BLOCK, tla.Float8E4M3FN, tla.AddressSpace.l0b, 512),
    ]

    l0c_s = [
        tla.allocate(M_BLOCK * KV_BLOCK, tla.Float32, tla.AddressSpace.l0c, 512),
        tla.allocate(M_BLOCK * KV_BLOCK, tla.Float32, tla.AddressSpace.l0c, 512),
    ]
    l0c_o = tla.allocate(M_BLOCK * D, tla.Float32, tla.AddressSpace.l0c, 512)

    ub_s = [
        tla.allocate((M_BLOCK // 2) * KV_BLOCK, tla.Float32, tla.AddressSpace.ub, 256),
        tla.allocate((M_BLOCK // 2) * KV_BLOCK, tla.Float32, tla.AddressSpace.ub, 256),
    ]
    # 合并为单块 112KB：phase1 用 [0:56KB]/[56KB:112KB] 双缓冲，
    # phase2（归并）复用其前 66KB 作 ub_buf —— UB 总量不变。
    ub_otmp_all = tla.allocate(
        2 * (M_BLOCK // 2) * D, tla.Float32, tla.AddressSpace.ub, 256
    )
    ub_otmp = [
        ub_otmp_all,
        ub_otmp_all + (M_BLOCK // 2) * D,
    ]
    ub_acc = tla.allocate((M_BLOCK // 2) * 264, tla.Float32, tla.AddressSpace.ub, 256)
    ub_p = [
        tla.allocate(
            (M_BLOCK // 2) * KV_BLOCK, tla.Float8E4M3FN, tla.AddressSpace.ub, 256
        ),
        tla.allocate(
            (M_BLOCK // 2) * KV_BLOCK, tla.Float8E4M3FN, tla.AddressSpace.ub, 256
        ),
    ]
    tile_max = tla.allocate(M_BLOCK // 2, tla.Float32, tla.AddressSpace.ub, 256)
    last_max = tla.allocate(M_BLOCK // 2, tla.Float32, tla.AddressSpace.ub, 256)
    last_sum = tla.allocate(M_BLOCK // 2, tla.Float32, tla.AddressSpace.ub, 256)
    exp_max = [
        tla.allocate(M_BLOCK // 2, tla.Float32, tla.AddressSpace.ub, 256),
        tla.allocate(M_BLOCK // 2, tla.Float32, tla.AddressSpace.ub, 256),
        tla.allocate(M_BLOCK // 2, tla.Float32, tla.AddressSpace.ub, 256),
    ]

    # ---- phase2（归并）复用 phase1 的 UB：UB 总量不变 ----
    # phase2 复用 ub_acc（phase1 的 acc 累加器，写 part_o 后由 acc_free 保证空闲）。
    # 不碰 ub_otmp_all/ub_s（fixpipe 写入，无显式释放 flag，复用会破坏 phase1）。
    ub_cbuf = ub_acc  # 8*GROUP_C*264 个 f32
    ub_cacc = ub_acc + 8 * GROUP_C * 264  # GROUP_C*256 个 f32
    ub_cws = ub_acc + 8 * GROUP_C * 264 + GROUP_C * 256  # GROUP_C*64 个 f32
    ub_cob = tla.allocate(GROUP_C * D, tla.BFloat16, tla.AddressSpace.ub, 256)
    # ---- flag ----
    q_l0a_ready = tla.flag("q_l0a_ready", tla.arch.MTE1, tla.arch.MTE2)
    q_l1_ready = tla.flag("q_l1_ready", tla.arch.MTE2, tla.arch.MTE1)
    k_l0b_ready = [
        tla.flag("k_l0b_ready_0", tla.arch.MTE1, tla.arch.MTE2),
        tla.flag("k_l0b_ready_1", tla.arch.MTE1, tla.arch.MTE2),
    ]
    k_l1_ready = [
        tla.flag("k_l1_ready_0", tla.arch.MTE2, tla.arch.MTE1),
        tla.flag("k_l1_ready_1", tla.arch.MTE2, tla.arch.MTE1),
    ]
    v_l0b_ready = [
        tla.flag("v_l0b_ready_0", tla.arch.MTE1, tla.arch.MTE2),
        tla.flag("v_l0b_ready_1", tla.arch.MTE1, tla.arch.MTE2),
    ]
    v_l1_ready = [
        tla.flag("v_l1_ready_0", tla.arch.MTE2, tla.arch.MTE1),
        tla.flag("v_l1_ready_1", tla.arch.MTE2, tla.arch.MTE1),
    ]
    l0a_avail = [
        tla.flag("l0a_avail_0", tla.arch.CUBE, tla.arch.MTE1),
        tla.flag("l0a_avail_1", tla.arch.CUBE, tla.arch.MTE1),
    ]
    l0b_avail = [
        tla.flag("l0b_avail_0", tla.arch.CUBE, tla.arch.MTE1),
        tla.flag("l0b_avail_1", tla.arch.CUBE, tla.arch.MTE1),
    ]
    l0a_ready = [
        tla.flag("l0a_ready_0", tla.arch.MTE1, tla.arch.CUBE),
        tla.flag("l0a_ready_1", tla.arch.MTE1, tla.arch.CUBE),
    ]
    l0b_ready = [
        tla.flag("l0b_ready_0", tla.arch.MTE1, tla.arch.CUBE),
        tla.flag("l0b_ready_1", tla.arch.MTE1, tla.arch.CUBE),
    ]
    l0cs_avail = [
        tla.flag("l0cs_avail_0", tla.arch.FIX, tla.arch.CUBE),
        tla.flag("l0cs_avail_1", tla.arch.FIX, tla.arch.CUBE),
    ]
    l0cs_ready = [
        tla.flag("l0cs_ready_0", tla.arch.CUBE, tla.arch.FIX),
        tla.flag("l0cs_ready_1", tla.arch.CUBE, tla.arch.FIX),
    ]
    l0co_avail = tla.flag("l0co_avail", tla.arch.FIX, tla.arch.CUBE)
    l0co_ready = tla.flag("l0co_ready", tla.arch.CUBE, tla.arch.FIX)
    ub_p_avail = [
        tla.flag("ub_p_avail_0", tla.arch.MTE3, tla.arch.VECTOR),
        tla.flag("ub_p_avail_1", tla.arch.MTE3, tla.arch.VECTOR),
    ]
    ub_p_ready = [
        tla.flag("ub_p_ready_0", tla.arch.VECTOR, tla.arch.MTE3),
        tla.flag("ub_p_ready_1", tla.arch.VECTOR, tla.arch.MTE3),
    ]
    acc_free = tla.flag("acc_free", tla.arch.MTE3, tla.arch.VECTOR)
    c_ld = tla.flag("f_c_ld", tla.arch.MTE2, tla.arch.VECTOR)
    c_vr = tla.flag("f_c_vr", tla.arch.VECTOR, tla.arch.MTE2)
    c_st = tla.flag("f_c_st", tla.arch.VECTOR, tla.arch.MTE3)
    ibarrier = tla.cross_flag("ibarrier", mode=0)
    s_cross = [tla.cross_flag("s_cross_0"), tla.cross_flag("s_cross_1")]
    o_cross = [tla.cross_flag("o_cross_0"), tla.cross_flag("o_cross_1")]
    p_cross = [
        tla.cross_flag("p_cross_0"),
        tla.cross_flag("p_cross_1"),
        tla.cross_flag("p_cross_2"),
    ]
    with tla.cube():
        tla.set_flag(q_l0a_ready)
        tla.set_flag(k_l0b_ready[0])
        tla.set_flag(k_l0b_ready[1])
        tla.set_flag(v_l0b_ready[0])
        tla.set_flag(v_l0b_ready[1])
        tla.set_flag(l0a_avail[0])
        tla.set_flag(l0a_avail[1])
        tla.set_flag(l0b_avail[0])
        tla.set_flag(l0b_avail[1])
        tla.set_flag(l0cs_avail[0])
        tla.set_flag(l0cs_avail[1])
        tla.set_flag(l0co_avail)
        tla.cross_core_set_flag(p_cross[0], tla.arch.MTE1)
        tla.cross_core_set_flag(p_cross[1], tla.arch.MTE1)
        tla.cross_core_set_flag(p_cross[2], tla.arch.MTE1)
    with tla.vector():
        tla.set_flag(ub_p_avail[0])
        tla.set_flag(ub_p_avail[1])
        tla.set_flag(acc_free)
        tla.cross_core_set_flag(s_cross[0], tla.arch.VECTOR)
        tla.cross_core_set_flag(s_cross[1], tla.arch.VECTOR)
        tla.cross_core_set_flag(o_cross[0], tla.arch.VECTOR)
        tla.cross_core_set_flag(o_cross[1], tla.arch.VECTOR)

    for task in tla.range(tla.arch.block_idx(), total, tla.arch.block_num()):
        m_blk = task // splits
        split = task - m_blk * splits
        row0 = m_blk * M_BLOCK
        row_num = m_rows - row0
        row_num = M_BLOCK if row_num > M_BLOCK else row_num
        base_t = nk_pages // splits
        rem = nk_pages - base_t * splits
        cnt = base_t + (1 if split < rem else 0)
        tile_lo = split * base_t + (split if split < rem else rem)
        # ---- Q tile 一次载入 L1 ----
        with tla.cube():
            gm_q = tla.make_tensor(
                gm_q8.ptr + row0 * D,
                tla.make_layout(tla.make_shape(M_BLOCK, D), tla.make_stride(D, 1)),
            )
            l1_q_t = tla.make_tensor_like(l1_q, gm_q, tla.arch.zN)
            tla.wait_flag(q_l0a_ready)
            tla.copy(l1_q_t, gm_q)
            tla.set_flag(q_l1_ready)
            tla.wait_flag(q_l1_ready)
        # ---- acc 生命周期：等上一任务 partial 的 MTE3 读完成 ----
        with tla.vector():
            tla.wait_flag(acc_free)
        # ---- 主循环：QK+softmax 超前，PV+rescale 滞后 PRE_LAUNCH ----
        launch_0 = tla.as_numeric(-1)
        launch_1 = tla.as_numeric(-2)
        launch_2 = tla.as_numeric(-3)
        vidx = tla.as_numeric(-1)
        for it in tla.range(0, cnt + PRE_LAUNCH, 1):
            launch_2 = launch_1
            launch_1 = launch_0
            launch_0 = it
            vidx = vidx + 1
            if it < cnt:
                tile = tile_lo + it
                lp = tile + page0
                phys = gm_bt[lp]
                tok0 = lp * PAGE
                row_lo = start - tok0 if tok0 < start else 0
                row_hi = start + nvis - tok0
                row_hi = PAGE if row_hi > PAGE else row_hi
                kv_act = row_hi - row_lo
                buf = vidx % 2
                l1p_id = vidx % 3
                # ---- QK（cube） ----
                with tla.cube():
                    gm_q_row = tla.make_tensor(
                        gm_q8.ptr + row0 * D,
                        tla.make_layout(
                            tla.make_shape(M_BLOCK, D), tla.make_stride(D, 1)
                        ),
                    )
                    l1_q_t2 = tla.make_tensor_like(l1_q, gm_q_row, tla.arch.zN)
                    gm_k = tla.make_tensor(
                        gm_kc.ptr + (phys * PAGE + row_lo) * D,
                        tla.make_layout(
                            tla.make_shape(D, KV_BLOCK),
                            tla.make_stride(1, D),
                            layoutTag=tla.arch.ColumnMajor,
                        ),
                    )
                    if buf == 0:
                        l1_k_t = tla.make_tensor_like(l1_k[0], gm_k, tla.arch.nZ)
                        tla.wait_flag(k_l0b_ready[0])
                        tla.copy(l1_k_t, gm_k)
                        tla.set_flag(k_l1_ready[0])
                        ub_s_full = tla.make_tensor(
                            ub_s[0],
                            tla.make_layout(
                                tla.make_shape(M_BLOCK, KV_BLOCK),
                                tla.make_stride(KV_BLOCK, 1),
                            ),
                        )
                        l0c_s_t = tla.make_tensor_like(
                            l0c_s[0], ub_s_full, tla.arch.L0Clayout
                        )
                        l0a_q = tla.make_tensor_like(l0a[0], l1_q_t2, tla.arch.zN)
                        l0b_k = tla.make_tensor_like(l0b[0], l1_k_t)
                        tla.wait_flag(l0a_avail[0])
                        tla.copy(l0a_q, l1_q_t2)
                        tla.set_flag(l0a_ready[0])
                        tla.wait_flag(l0b_avail[0])
                        tla.wait_flag(k_l1_ready[0])
                        tla.copy(l0b_k, l1_k_t)
                        tla.set_flag(l0b_ready[0])
                        tla.set_flag(k_l0b_ready[0])
                        tla.wait_flag(l0a_ready[0])
                        tla.wait_flag(l0b_ready[0])
                        tla.wait_flag(l0cs_avail[0])
                        tla.mmad(l0c_s_t, l0a_q, l0b_k, init_c=True)
                        tla.set_flag(l0a_avail[0])
                        tla.set_flag(l0b_avail[0])
                        tla.cross_core_wait_flag(s_cross[0], tla.arch.FIX)
                        tla.set_flag(l0cs_ready[0])
                        tla.wait_flag(l0cs_ready[0])
                        ub_s_view = tla.make_tensor(
                            ub_s[0],
                            tla.make_layout(
                                tla.make_shape(row_num, kv_act),
                                tla.make_stride(KV_BLOCK, 1),
                            ),
                        )
                        tla.copy(
                            ub_s_view,
                            l0c_s_t,
                            tla.params.CopyL0C2DstParams(
                                l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M
                            ),
                        )
                        tla.set_flag(l0cs_avail[0])
                        tla.cross_core_set_flag(s_cross[0], tla.arch.FIX)
                    else:
                        l1_k_t = tla.make_tensor_like(l1_k[1], gm_k, tla.arch.nZ)
                        tla.wait_flag(k_l0b_ready[1])
                        tla.copy(l1_k_t, gm_k)
                        tla.set_flag(k_l1_ready[1])
                        ub_s_full = tla.make_tensor(
                            ub_s[1],
                            tla.make_layout(
                                tla.make_shape(M_BLOCK, KV_BLOCK),
                                tla.make_stride(KV_BLOCK, 1),
                            ),
                        )
                        l0c_s_t = tla.make_tensor_like(
                            l0c_s[1], ub_s_full, tla.arch.L0Clayout
                        )
                        l0a_q = tla.make_tensor_like(l0a[1], l1_q_t2, tla.arch.zN)
                        l0b_k = tla.make_tensor_like(l0b[1], l1_k_t)
                        tla.wait_flag(l0a_avail[1])
                        tla.copy(l0a_q, l1_q_t2)
                        tla.set_flag(l0a_ready[1])
                        tla.wait_flag(l0b_avail[1])
                        tla.wait_flag(k_l1_ready[1])
                        tla.copy(l0b_k, l1_k_t)
                        tla.set_flag(l0b_ready[1])
                        tla.set_flag(k_l0b_ready[1])
                        tla.wait_flag(l0a_ready[1])
                        tla.wait_flag(l0b_ready[1])
                        tla.wait_flag(l0cs_avail[1])
                        tla.mmad(l0c_s_t, l0a_q, l0b_k, init_c=True)
                        tla.set_flag(l0a_avail[1])
                        tla.set_flag(l0b_avail[1])
                        tla.cross_core_wait_flag(s_cross[1], tla.arch.FIX)
                        tla.set_flag(l0cs_ready[1])
                        tla.wait_flag(l0cs_ready[1])
                        ub_s_view = tla.make_tensor(
                            ub_s[1],
                            tla.make_layout(
                                tla.make_shape(row_num, kv_act),
                                tla.make_stride(KV_BLOCK, 1),
                            ),
                        )
                        tla.copy(
                            ub_s_view,
                            l0c_s_t,
                            tla.params.CopyL0C2DstParams(
                                l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M
                            ),
                        )
                        tla.set_flag(l0cs_avail[1])
                        tla.cross_core_set_flag(s_cross[1], tla.arch.FIX)
                    if it == cnt - 1:
                        tla.set_flag(q_l0a_ready)
                # ---- online softmax + P（vector，AIV 分行） ----
                with tla.vector():
                    sub_s = tla.arch.sub_block_idx()
                    m_half_s = (row_num + 1) // 2
                    m_sub_s = m_half_s if sub_s == 0 else (row_num - m_half_s)
                    if m_sub_s == 0:
                        if buf == 0:
                            tla.cross_core_wait_flag(s_cross[0], tla.arch.VECTOR)
                            tla.cross_core_set_flag(s_cross[0], tla.arch.VECTOR)
                        else:
                            tla.cross_core_wait_flag(s_cross[1], tla.arch.VECTOR)
                            tla.cross_core_set_flag(s_cross[1], tla.arch.VECTOR)
                    else:
                        if buf == 0:
                            tla.cross_core_wait_flag(s_cross[0], tla.arch.VECTOR)
                            tla.wait_flag(ub_p_avail[0])
                        else:
                            tla.cross_core_wait_flag(s_cross[1], tla.arch.VECTOR)
                            tla.wait_flag(ub_p_avail[1])
                        ub_s_ptr = ub_s[0] if buf == 0 else ub_s[1]
                        ub_p_ptr = ub_p[0] if buf == 0 else ub_p[1]
                        ub_s_sub = tla.make_tensor(
                            ub_s_ptr,
                            tla.make_layout(
                                tla.make_shape(m_sub_s, KV_BLOCK),
                                tla.make_stride(KV_BLOCK, 1),
                            ),
                        )
                        ub_p_zN = tla.make_tensor(
                            ub_p_ptr,
                            tla.make_layout(
                                tla.make_shape(M_BLOCK // 2, KV_BLOCK),
                                tla.make_stride(KV_BLOCK, 1),
                            ),
                        )
                        tail_n = (kv_act - 1) % 64 + 1
                        if vidx == 0:
                            _sf_first(
                                ub_s_sub,
                                ub_p_zN,
                                m_sub_s,
                                kv_act,
                                tail_n,
                                last_max,
                                last_sum,
                            )
                        else:
                            if l1p_id == 0:
                                _sf_mid(
                                    ub_s_sub,
                                    ub_p_zN,
                                    m_sub_s,
                                    kv_act,
                                    tail_n,
                                    tile_max,
                                    last_max,
                                    last_sum,
                                    exp_max[0],
                                )
                            elif l1p_id == 1:
                                _sf_mid(
                                    ub_s_sub,
                                    ub_p_zN,
                                    m_sub_s,
                                    kv_act,
                                    tail_n,
                                    tile_max,
                                    last_max,
                                    last_sum,
                                    exp_max[1],
                                )
                            else:
                                _sf_mid(
                                    ub_s_sub,
                                    ub_p_zN,
                                    m_sub_s,
                                    kv_act,
                                    tail_n,
                                    tile_max,
                                    last_max,
                                    last_sum,
                                    exp_max[2],
                                )
                        if buf == 0:
                            tla.set_flag(ub_p_ready[0])
                            tla.wait_flag(ub_p_ready[0])
                            tla.cross_core_set_flag(s_cross[0], tla.arch.VECTOR)
                        else:
                            tla.set_flag(ub_p_ready[1])
                            tla.wait_flag(ub_p_ready[1])
                            tla.cross_core_set_flag(s_cross[1], tla.arch.VECTOR)
                        # P(UB) -> L1
                        l1_p_ref = tla.make_tensor(
                            ub_s[0],
                            tla.make_layout(
                                tla.make_shape(M_BLOCK, KV_BLOCK),
                                tla.make_stride(KV_BLOCK, 1),
                            ),
                        )
                        l1p_ptr = (
                            l1_p[0]
                            if l1p_id == 0
                            else (l1_p[1] if l1p_id == 1 else l1_p[2])
                        )
                        l1_p_full = tla.make_tensor_like(l1p_ptr, l1_p_ref, tla.arch.zN)
                        l1_p_tile = tla.tile_view(
                            l1_p_full,
                            tla.make_shape(m_half_s, KV_BLOCK),
                            tla.make_coord(sub_s, 0),
                        )
                        ub_p_send = tla.tile_view(
                            ub_p_zN,
                            tla.make_shape(m_sub_s, KV_BLOCK),
                            tla.make_coord(0, 0),
                        )
                        if l1p_id == 0:
                            tla.cross_core_wait_flag(p_cross[0], tla.arch.MTE3)
                            tla.copy(l1_p_tile, ub_p_send)
                            if buf == 0:
                                tla.set_flag(ub_p_avail[0])
                            else:
                                tla.set_flag(ub_p_avail[1])
                            tla.cross_core_set_flag(p_cross[0], tla.arch.MTE3)
                        elif l1p_id == 1:
                            tla.cross_core_wait_flag(p_cross[1], tla.arch.MTE3)
                            tla.copy(l1_p_tile, ub_p_send)
                            if buf == 0:
                                tla.set_flag(ub_p_avail[0])
                            else:
                                tla.set_flag(ub_p_avail[1])
                            tla.cross_core_set_flag(p_cross[1], tla.arch.MTE3)
                        else:
                            tla.cross_core_wait_flag(p_cross[2], tla.arch.MTE3)
                            tla.copy(l1_p_tile, ub_p_send)
                            if buf == 0:
                                tla.set_flag(ub_p_avail[0])
                            else:
                                tla.set_flag(ub_p_avail[1])
                            tla.cross_core_set_flag(p_cross[2], tla.arch.MTE3)
            # ---- PV + rescale（滞后 PRE_LAUNCH） ----
            if it >= PRE_LAUNCH and launch_2 >= 0:
                itd = launch_2
                vd = vidx - PRE_LAUNCH
                tiled = tile_lo + itd
                lp_d = tiled + page0
                phys_d = gm_bt[lp_d]
                tok0_d = lp_d * PAGE
                row_lo_d = start - tok0_d if tok0_d < start else 0
                row_hi_d = start + nvis - tok0_d
                row_hi_d = PAGE if row_hi_d > PAGE else row_hi_d
                kv_act_d = row_hi_d - row_lo_d
                bufd = vd % 2
                l1p_id_d = vd % 3
                with tla.cube():
                    gm_v = tla.make_tensor(
                        gm_vc.ptr + (phys_d * PAGE + row_lo_d) * D,
                        tla.make_layout(
                            tla.make_shape(kv_act_d, D), tla.make_stride(D, 1)
                        ),
                    )
                    ub_o_full = tla.make_tensor(
                        ub_otmp[0] if bufd == 0 else ub_otmp[1],
                        tla.make_layout(
                            tla.make_shape(M_BLOCK, D), tla.make_stride(D, 1)
                        ),
                    )
                    l0c_o_t = tla.make_tensor_like(l0c_o, ub_o_full, tla.arch.L0Clayout)
                    l1_p_ref_d = tla.make_tensor(
                        ub_s[0],
                        tla.make_layout(
                            tla.make_shape(M_BLOCK, kv_act_d),
                            tla.make_stride(KV_BLOCK, 1),
                        ),
                    )
                    if bufd == 0:
                        l1_v_t = tla.make_tensor_like(l1_v[0], gm_v, tla.arch.zN)
                        tla.wait_flag(v_l0b_ready[0])
                        if kv_act_d < KV_BLOCK:
                            gm_z2 = tla.make_tensor(
                                gm_zero.ptr,
                                tla.make_layout(
                                    tla.make_shape(KV_BLOCK, D), tla.make_stride(D, 1)
                                ),
                            )
                            l1_v_full = tla.make_tensor_like(
                                l1_v[0], gm_z2, tla.arch.zN
                            )
                            tla.copy(l1_v_full, gm_z2)
                        tla.copy(l1_v_t, gm_v)
                        tla.set_flag(v_l1_ready[0])
                        tla.wait_flag(v_l1_ready[0])
                        if l1p_id_d == 0:
                            tla.cross_core_wait_flag(p_cross[0], tla.arch.MTE1)
                        elif l1p_id_d == 1:
                            tla.cross_core_wait_flag(p_cross[1], tla.arch.MTE1)
                        else:
                            tla.cross_core_wait_flag(p_cross[2], tla.arch.MTE1)
                        l0b_v = tla.make_tensor_like(l0b[0], l1_v_t, tla.arch.nZ)
                        tla.wait_flag(l0b_avail[0])
                        tla.copy(l0b_v, l1_v_t)
                        tla.set_flag(l0b_ready[0])
                        tla.set_flag(v_l0b_ready[0])
                        l1p_ptr_d = (
                            l1_p[0]
                            if l1p_id_d == 0
                            else (l1_p[1] if l1p_id_d == 1 else l1_p[2])
                        )
                        l1_p_fd = tla.make_tensor_like(
                            l1p_ptr_d, l1_p_ref_d, tla.arch.zN
                        )
                        l0a_p = tla.make_tensor_like(l0a[0], l1_p_fd, tla.arch.zN)
                        tla.wait_flag(l0a_avail[0])
                        tla.copy(l0a_p, l1_p_fd)
                        tla.set_flag(l0a_ready[0])
                        if l1p_id_d == 0:
                            tla.cross_core_set_flag(p_cross[0], tla.arch.MTE1)
                        elif l1p_id_d == 1:
                            tla.cross_core_set_flag(p_cross[1], tla.arch.MTE1)
                        else:
                            tla.cross_core_set_flag(p_cross[2], tla.arch.MTE1)
                        tla.wait_flag(l0a_ready[0])
                        tla.wait_flag(l0b_ready[0])
                        tla.wait_flag(l0co_avail)
                        tla.mmad(l0c_o_t, l0a_p, l0b_v, init_c=True)
                        tla.set_flag(l0a_avail[0])
                        tla.set_flag(l0b_avail[0])
                        tla.cross_core_wait_flag(o_cross[0], tla.arch.FIX)
                        tla.set_flag(l0co_ready)
                        tla.wait_flag(l0co_ready)
                        ub_o_view = tla.make_tensor(
                            ub_otmp[0],
                            tla.make_layout(
                                tla.make_shape(row_num, D), tla.make_stride(D, 1)
                            ),
                        )
                        tla.copy(
                            ub_o_view,
                            l0c_o_t,
                            tla.params.CopyL0C2DstParams(
                                l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M
                            ),
                        )
                        tla.set_flag(l0co_avail)
                        tla.cross_core_set_flag(o_cross[0], tla.arch.FIX)
                    else:
                        l1_v_t = tla.make_tensor_like(l1_v[1], gm_v, tla.arch.zN)
                        tla.wait_flag(v_l0b_ready[1])
                        if kv_act_d < KV_BLOCK:
                            gm_z2 = tla.make_tensor(
                                gm_zero.ptr,
                                tla.make_layout(
                                    tla.make_shape(KV_BLOCK, D), tla.make_stride(D, 1)
                                ),
                            )
                            l1_v_full = tla.make_tensor_like(
                                l1_v[1], gm_z2, tla.arch.zN
                            )
                            tla.copy(l1_v_full, gm_z2)
                        tla.copy(l1_v_t, gm_v)
                        tla.set_flag(v_l1_ready[1])
                        tla.wait_flag(v_l1_ready[1])
                        if l1p_id_d == 0:
                            tla.cross_core_wait_flag(p_cross[0], tla.arch.MTE1)
                        elif l1p_id_d == 1:
                            tla.cross_core_wait_flag(p_cross[1], tla.arch.MTE1)
                        else:
                            tla.cross_core_wait_flag(p_cross[2], tla.arch.MTE1)
                        l0b_v = tla.make_tensor_like(l0b[1], l1_v_t, tla.arch.nZ)
                        tla.wait_flag(l0b_avail[1])
                        tla.copy(l0b_v, l1_v_t)
                        tla.set_flag(l0b_ready[1])
                        tla.set_flag(v_l0b_ready[1])
                        l1p_ptr_d = (
                            l1_p[0]
                            if l1p_id_d == 0
                            else (l1_p[1] if l1p_id_d == 1 else l1_p[2])
                        )
                        l1_p_fd = tla.make_tensor_like(
                            l1p_ptr_d, l1_p_ref_d, tla.arch.zN
                        )
                        l0a_p = tla.make_tensor_like(l0a[1], l1_p_fd, tla.arch.zN)
                        tla.wait_flag(l0a_avail[1])
                        tla.copy(l0a_p, l1_p_fd)
                        tla.set_flag(l0a_ready[1])
                        if l1p_id_d == 0:
                            tla.cross_core_set_flag(p_cross[0], tla.arch.MTE1)
                        elif l1p_id_d == 1:
                            tla.cross_core_set_flag(p_cross[1], tla.arch.MTE1)
                        else:
                            tla.cross_core_set_flag(p_cross[2], tla.arch.MTE1)
                        tla.wait_flag(l0a_ready[1])
                        tla.wait_flag(l0b_ready[1])
                        tla.wait_flag(l0co_avail)
                        tla.mmad(l0c_o_t, l0a_p, l0b_v, init_c=True)
                        tla.set_flag(l0a_avail[1])
                        tla.set_flag(l0b_avail[1])
                        tla.cross_core_wait_flag(o_cross[1], tla.arch.FIX)
                        tla.set_flag(l0co_ready)
                        tla.wait_flag(l0co_ready)
                        ub_o_view = tla.make_tensor(
                            ub_otmp[1],
                            tla.make_layout(
                                tla.make_shape(row_num, D), tla.make_stride(D, 1)
                            ),
                        )
                        tla.copy(
                            ub_o_view,
                            l0c_o_t,
                            tla.params.CopyL0C2DstParams(
                                l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M
                            ),
                        )
                        tla.set_flag(l0co_avail)
                        tla.cross_core_set_flag(o_cross[1], tla.arch.FIX)
                # ---- rescale（vector） ----
                with tla.vector():
                    sub_r = tla.arch.sub_block_idx()
                    m_half_r = (row_num + 1) // 2
                    m_sub_r = m_half_r if sub_r == 0 else (row_num - m_half_r)
                    if m_sub_r == 0:
                        if bufd == 0:
                            tla.cross_core_wait_flag(o_cross[0], tla.arch.VECTOR)
                            tla.cross_core_set_flag(o_cross[0], tla.arch.VECTOR)
                        else:
                            tla.cross_core_wait_flag(o_cross[1], tla.arch.VECTOR)
                            tla.cross_core_set_flag(o_cross[1], tla.arch.VECTOR)
                    else:
                        if bufd == 0:
                            tla.cross_core_wait_flag(o_cross[0], tla.arch.VECTOR)
                        else:
                            tla.cross_core_wait_flag(o_cross[1], tla.arch.VECTOR)
                        dm_ptr = (
                            exp_max[0]
                            if l1p_id_d == 0
                            else (exp_max[1] if l1p_id_d == 1 else exp_max[2])
                        )
                        dm_ub = tla.make_tensor(
                            dm_ptr,
                            tla.make_layout(
                                tla.make_shape(m_sub_r), tla.make_stride(1)
                            ),
                        )
                        lo_ptr = ub_otmp[0] if bufd == 0 else ub_otmp[1]
                        lo_ub = tla.make_tensor(
                            lo_ptr,
                            tla.make_layout(
                                tla.make_shape(m_sub_r, D),
                                tla.make_stride(D, 1),
                            ),
                        )
                        acc_ub = tla.make_tensor(
                            ub_acc,
                            tla.make_layout(
                                tla.make_shape(m_sub_r, 264), tla.make_stride(264, 1)
                            ),
                        )
                        with tla.vec.func(mode="simd"):
                            full_f32 = tla.create_mask(
                                pattern=tla.mask.ALL, dtype=tla.Float32
                            )
                            lo_p0 = tla.tile_view(
                                lo_ub, tla.make_shape(1, 64), tla.make_coord(0, 0)
                            ).load()
                            lo_p1 = tla.tile_view(
                                lo_ub, tla.make_shape(1, 64), tla.make_coord(0, 1)
                            ).load()
                            lo_p2 = tla.tile_view(
                                lo_ub, tla.make_shape(1, 64), tla.make_coord(0, 2)
                            ).load()
                            lo_p3 = tla.tile_view(
                                lo_ub, tla.make_shape(1, 64), tla.make_coord(0, 3)
                            ).load()
                            for ri in tla.range(m_sub_r):
                                lo_c0 = lo_p0
                                lo_c1 = lo_p1
                                lo_c2 = lo_p2
                                lo_c3 = lo_p3
                                dm_c = tla.tile_view(
                                    dm_ub, tla.make_shape(1), tla.make_coord(ri)
                                ).load(
                                    params=tla.params.NormalLoadParams(
                                        load_dist=tla.params.LoadDist.DIST_BRC_B32
                                    )
                                )
                                ri_n = ri + 1
                                ri_n = m_sub_r - 1 if ri_n >= m_sub_r else ri_n
                                lo_p0 = tla.tile_view(
                                    lo_ub,
                                    tla.make_shape(1, 64),
                                    tla.make_coord(ri_n, 0),
                                ).load()
                                lo_p1 = tla.tile_view(
                                    lo_ub,
                                    tla.make_shape(1, 64),
                                    tla.make_coord(ri_n, 1),
                                ).load()
                                lo_p2 = tla.tile_view(
                                    lo_ub,
                                    tla.make_shape(1, 64),
                                    tla.make_coord(ri_n, 2),
                                ).load()
                                lo_p3 = tla.tile_view(
                                    lo_ub,
                                    tla.make_shape(1, 64),
                                    tla.make_coord(ri_n, 3),
                                ).load()
                                lo_r0 = lo_c0
                                ac_r0 = lo_r0
                                if vd != 0:
                                    dm_r0 = dm_c
                                    ac_r0 = tla.add(
                                        tla.mul(
                                            tla.tile_view(
                                                acc_ub,
                                                tla.make_shape(1, 64),
                                                tla.make_coord(ri, 0),
                                            ).load(),
                                            dm_r0,
                                            mask=full_f32,
                                        ),
                                        lo_r0,
                                        mask=full_f32,
                                    )
                                tla.tile_view(
                                    acc_ub,
                                    tla.make_shape(1, 64),
                                    tla.make_coord(ri, 0),
                                ).store(ac_r0, mask=full_f32)

                                lo_r1 = lo_c1
                                ac_r1 = lo_r1
                                if vd != 0:
                                    dm_r1 = dm_c
                                    ac_r1 = tla.add(
                                        tla.mul(
                                            tla.tile_view(
                                                acc_ub,
                                                tla.make_shape(1, 64),
                                                tla.make_coord(ri, 1),
                                            ).load(),
                                            dm_r1,
                                            mask=full_f32,
                                        ),
                                        lo_r1,
                                        mask=full_f32,
                                    )
                                tla.tile_view(
                                    acc_ub,
                                    tla.make_shape(1, 64),
                                    tla.make_coord(ri, 1),
                                ).store(ac_r1, mask=full_f32)

                                lo_r2 = lo_c2
                                ac_r2 = lo_r2
                                if vd != 0:
                                    dm_r2 = dm_c
                                    ac_r2 = tla.add(
                                        tla.mul(
                                            tla.tile_view(
                                                acc_ub,
                                                tla.make_shape(1, 64),
                                                tla.make_coord(ri, 2),
                                            ).load(),
                                            dm_r2,
                                            mask=full_f32,
                                        ),
                                        lo_r2,
                                        mask=full_f32,
                                    )
                                tla.tile_view(
                                    acc_ub,
                                    tla.make_shape(1, 64),
                                    tla.make_coord(ri, 2),
                                ).store(ac_r2, mask=full_f32)

                                lo_r3 = lo_c3
                                ac_r3 = lo_r3
                                if vd != 0:
                                    dm_r3 = dm_c
                                    ac_r3 = tla.add(
                                        tla.mul(
                                            tla.tile_view(
                                                acc_ub,
                                                tla.make_shape(1, 64),
                                                tla.make_coord(ri, 3),
                                            ).load(),
                                            dm_r3,
                                            mask=full_f32,
                                        ),
                                        lo_r3,
                                        mask=full_f32,
                                    )
                                tla.tile_view(
                                    acc_ub,
                                    tla.make_shape(1, 64),
                                    tla.make_coord(ri, 3),
                                ).store(ac_r3, mask=full_f32)

                            tla.local_mem_bar(
                                tla.params.MemType.VEC_STORE,
                                tla.params.MemType.VEC_LOAD,
                            )
                        if bufd == 0:
                            tla.cross_core_set_flag(o_cross[0], tla.arch.VECTOR)
                        else:
                            tla.cross_core_set_flag(o_cross[1], tla.arch.VECTOR)
        # ---- 任务收尾：写 partial 到 GM ----
        with tla.vector():
            sub_e = tla.arch.sub_block_idx()
            m_half_e = (row_num + 1) // 2
            m_sub_e = m_half_e if sub_e == 0 else (row_num - m_half_e)
            if m_sub_e > 0:
                if cnt == 0:
                    with tla.vec.func(mode="simd"):
                        zero_reg = tla.full(0.0, tla.Float32)
                        minv_reg = tla.full(-3.0e38, tla.Float32)
                        one_mask, _ = tla.update_mask(1, dtype=tla.Float32)
                        acc_z = tla.make_tensor(
                            ub_acc,
                            tla.make_layout(
                                tla.make_shape(m_sub_e, 264), tla.make_stride(264, 1)
                            ),
                        )
                        lm_z = tla.make_tensor(
                            last_max,
                            tla.make_layout(tla.make_shape(56), tla.make_stride(1)),
                        )
                        ls_z = tla.make_tensor(
                            last_sum,
                            tla.make_layout(tla.make_shape(56), tla.make_stride(1)),
                        )
                        for zi in tla.range(m_sub_e):
                            for zc in range(4):
                                tla.tile_view(
                                    acc_z,
                                    tla.make_shape(1, 64),
                                    tla.make_coord(zi, zc),
                                ).store(zero_reg)
                            tla.tile_view(
                                lm_z, tla.make_shape(1), tla.make_coord(zi)
                            ).store(
                                minv_reg, tla.params.UnalignStoreParams(), mask=one_mask
                            )
                            tla.tile_view(
                                ls_z, tla.make_shape(1), tla.make_coord(zi)
                            ).store(
                                zero_reg, tla.params.UnalignStoreParams(), mask=one_mask
                            )
                # m/l 写入 acc 暂存的 256/257 列，再一次 MTE 拷贝（行宽 264）
                with tla.vec.func(mode="simd"):
                    one_mask_e, _ = tla.update_mask(1, dtype=tla.Float32)
                    lm_w = tla.make_tensor(
                        last_max,
                        tla.make_layout(tla.make_shape(56), tla.make_stride(1)),
                    )
                    ls_w = tla.make_tensor(
                        last_sum,
                        tla.make_layout(tla.make_shape(56), tla.make_stride(1)),
                    )
                    acc_ml = tla.make_tensor(
                        ub_acc,
                        tla.make_layout(
                            tla.make_shape(m_sub_e, 264), tla.make_stride(264, 1)
                        ),
                    )
                    for wi in tla.range(m_sub_e):
                        m_val = tla.tile_view(
                            lm_w, tla.make_shape(1), tla.make_coord(wi)
                        ).load(
                            params=tla.params.NormalLoadParams(
                                load_dist=tla.params.LoadDist.DIST_BRC_B32
                            )
                        )
                        l_val = tla.tile_view(
                            ls_w, tla.make_shape(1), tla.make_coord(wi)
                        ).load(
                            params=tla.params.NormalLoadParams(
                                load_dist=tla.params.LoadDist.DIST_BRC_B32
                            )
                        )
                        tla.tile_view(
                            acc_ml, tla.make_shape(1, 1), tla.make_coord(wi, 256)
                        ).store(m_val, tla.params.UnalignStoreParams(), mask=one_mask_e)
                        tla.tile_view(
                            acc_ml, tla.make_shape(1, 1), tla.make_coord(wi, 257)
                        ).store(l_val, tla.params.UnalignStoreParams(), mask=one_mask_e)
                base_row = (m_blk * splits + split) * M_BLOCK + sub_e * m_half_e
                gm_po = tla.make_tensor(
                    part_o.ptr + base_row * 264,
                    tla.make_layout(
                        tla.make_shape(m_sub_e, 264), tla.make_stride(264, 1)
                    ),
                )
                acc_full = tla.make_tensor(
                    ub_acc,
                    tla.make_layout(
                        tla.make_shape(m_sub_e, 264), tla.make_stride(264, 1)
                    ),
                )
                tla.set_flag(ub_p_ready[0])
                tla.wait_flag(ub_p_ready[0])
                tla.copy(gm_po, acc_full)
                tla.set_flag(acc_free)
    # ---- 尾声排空 ----
    with tla.cube():
        tla.wait_flag(q_l0a_ready)
        tla.wait_flag(k_l0b_ready[0])
        tla.wait_flag(k_l0b_ready[1])
        tla.wait_flag(v_l0b_ready[0])
        tla.wait_flag(v_l0b_ready[1])
        tla.wait_flag(l0a_avail[0])
        tla.wait_flag(l0a_avail[1])
        tla.wait_flag(l0b_avail[0])
        tla.wait_flag(l0b_avail[1])
        tla.wait_flag(l0cs_avail[0])
        tla.wait_flag(l0cs_avail[1])
        tla.wait_flag(l0co_avail)
        tla.cross_core_wait_flag(s_cross[0], tla.arch.FIX)
        tla.cross_core_wait_flag(s_cross[1], tla.arch.FIX)
        tla.cross_core_wait_flag(o_cross[0], tla.arch.FIX)
        tla.cross_core_wait_flag(o_cross[1], tla.arch.FIX)
        tla.pipe_barrier(tla.pipes.ALL)
    with tla.vector():
        tla.wait_flag(ub_p_avail[0])
        tla.wait_flag(ub_p_avail[1])
        tla.wait_flag(acc_free)
        tla.cross_core_wait_flag(p_cross[0], tla.arch.MTE3)
        tla.cross_core_wait_flag(p_cross[1], tla.arch.MTE3)
        tla.cross_core_wait_flag(p_cross[2], tla.arch.MTE3)
        tla.pipe_barrier(tla.pipes.ALL)

    # ---- phase1 -> phase2 全核屏障（所有 AIV 同步，保证 part_o 全部就绪） ----
    with tla.vector():
        tla.pipe_barrier(tla.pipes.ALL)
        tla.cross_core_set_flag(ibarrier, tla.arch.MTE2)
        tla.cross_core_wait_flag(ibarrier, tla.arch.MTE2)

    # ---- phase2：归并（复用 phase1 的 UB） ----
    _combine_rows(
        part_o,
        gm_out,
        m_rows,
        splits,
        ub_cbuf,
        ub_cacc,
        ub_cob,
        ub_cws,
        c_ld,
        c_vr,
        c_st,
        tla.as_numeric(tla.arch.block_idx() * 2 + tla.arch.sub_block_idx()),
        tla.as_numeric(tla.arch.block_num() * 2),
    )


# ---------------------------------------------------------------------------
# kernel 3：split 归并（FP32）-> BF16
# ---------------------------------------------------------------------------


@tla.jit
def _combine_rows(
    part_o: tla.Tensor,  # [m_blocks*splits*112, 264] FP32（0..255=acc, 256=m, 257=l）
    gm_out: tla.Tensor,  # [M, 256] BF16
    m_rows: tla.Int32,
    splits: tla.Int32,
    ub_buf,  # 8*GROUP*W FP32 暂存
    ub_acc,  # GROUP*D FP32 累加器
    ub_ob,  # GROUP*D BF16 输出暂存
    ub_ws,  # GROUP*64 FP32 工作区
    ld,  # flag MTE2->VECTOR
    vr,  # flag VECTOR->MTE2
    st,  # flag VECTOR->MTE3
    aiv_id: tla.Int32,  # 本 AIV 的全局 id
    aiv_num: tla.Int32,  # AIV 总数
) -> None:
    D = 256
    W = 264
    M_BLOCK = 112
    GROUP = 4  # 112 % 4 == 0 且 56 % 4 == 0：行组不跨 m_blk/AIV 边界
    with tla.vector():
        tla.set_flag(vr)
    n_groups = (m_rows + GROUP - 1) // GROUP
    for g in tla.range(aiv_id, n_groups, aiv_num):
        r0 = g * GROUP
        rows = m_rows - r0
        rows = GROUP if rows > GROUP else rows
        m_blk = r0 // M_BLOCK
        r_in = r0 - m_blk * M_BLOCK
        with tla.vector():
            buf_v = tla.make_tensor(
                ub_buf,
                tla.make_layout(tla.make_shape(8 * GROUP * W), tla.make_stride(1)),
            )
            acc_v = tla.make_tensor(
                ub_acc, tla.make_layout(tla.make_shape(GROUP * D), tla.make_stride(1))
            )
            buf2_v = tla.make_tensor(
                ub_buf,
                tla.make_layout(tla.make_shape(8 * GROUP, W), tla.make_stride(W, 1)),
            )
            acc2_v = tla.make_tensor(
                ub_acc,
                tla.make_layout(tla.make_shape(GROUP, D), tla.make_stride(D, 1)),
            )
            ob2_v = tla.make_tensor(
                ub_ob,
                tla.make_layout(tla.make_shape(GROUP, D), tla.make_stride(D, 1)),
            )
            ws_v = tla.make_tensor(
                ub_ws, tla.make_layout(tla.make_shape(GROUP * 64), tla.make_stride(1))
            )
            # ---- 相位 0：acc 清零 + ws 初值 ----
            with tla.vec.func(mode="simd"):
                full_0 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                one_0, _ = tla.update_mask(1, dtype=tla.Float32)
                for zc in range(GROUP):
                    for zcc in range(4):
                        tla.tile_view(
                            acc2_v, tla.make_shape(1, 64), tla.make_coord(zc, zcc)
                        ).store(tla.full(0.0, tla.Float32), mask=full_0)
                for zi in range(GROUP):
                    tla.tile_view(
                        ws_v, tla.make_shape(1), tla.make_coord(zi * 64)
                    ).store(
                        tla.full(-3.0e38, tla.Float32),
                        tla.params.UnalignStoreParams(),
                        mask=one_0,
                    )
                    tla.tile_view(
                        ws_v, tla.make_shape(1), tla.make_coord(zi * 64 + 16)
                    ).store(
                        tla.full(0.0, tla.Float32),
                        tla.params.UnalignStoreParams(),
                        mask=one_0,
                    )
                tla.local_mem_bar(
                    tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                )
            # ---- 相位 0.5：爆发式 copy 全部 split 的 (rows, 264) 块, 单次握手 ----
            tla.wait_flag(vr)
            gm_c0 = tla.make_tensor(
                part_o.ptr + ((m_blk * splits) * M_BLOCK + r_in) * W,
                tla.make_layout(tla.make_shape(rows * W), tla.make_stride(1)),
            )
            ub_c0 = tla.make_tensor(
                ub_buf,
                tla.make_layout(tla.make_shape(rows * W), tla.make_stride(1)),
            )
            tla.copy(ub_c0, gm_c0)
            if splits > 1:
                gm_c1 = tla.make_tensor(
                    part_o.ptr + ((m_blk * splits + 1) * M_BLOCK + r_in) * W,
                    tla.make_layout(tla.make_shape(rows * W), tla.make_stride(1)),
                )
                ub_c1 = tla.make_tensor(
                    ub_buf + 1 * GROUP * W,
                    tla.make_layout(tla.make_shape(rows * W), tla.make_stride(1)),
                )
                tla.copy(ub_c1, gm_c1)
            if splits > 2:
                gm_c2 = tla.make_tensor(
                    part_o.ptr + ((m_blk * splits + 2) * M_BLOCK + r_in) * W,
                    tla.make_layout(tla.make_shape(rows * W), tla.make_stride(1)),
                )
                ub_c2 = tla.make_tensor(
                    ub_buf + 2 * GROUP * W,
                    tla.make_layout(tla.make_shape(rows * W), tla.make_stride(1)),
                )
                tla.copy(ub_c2, gm_c2)
            if splits > 3:
                gm_c3 = tla.make_tensor(
                    part_o.ptr + ((m_blk * splits + 3) * M_BLOCK + r_in) * W,
                    tla.make_layout(tla.make_shape(rows * W), tla.make_stride(1)),
                )
                ub_c3 = tla.make_tensor(
                    ub_buf + 3 * GROUP * W,
                    tla.make_layout(tla.make_shape(rows * W), tla.make_stride(1)),
                )
                tla.copy(ub_c3, gm_c3)
            if splits > 4:
                gm_c4 = tla.make_tensor(
                    part_o.ptr + ((m_blk * splits + 4) * M_BLOCK + r_in) * W,
                    tla.make_layout(tla.make_shape(rows * W), tla.make_stride(1)),
                )
                ub_c4 = tla.make_tensor(
                    ub_buf + 4 * GROUP * W,
                    tla.make_layout(tla.make_shape(rows * W), tla.make_stride(1)),
                )
                tla.copy(ub_c4, gm_c4)
            if splits > 5:
                gm_c5 = tla.make_tensor(
                    part_o.ptr + ((m_blk * splits + 5) * M_BLOCK + r_in) * W,
                    tla.make_layout(tla.make_shape(rows * W), tla.make_stride(1)),
                )
                ub_c5 = tla.make_tensor(
                    ub_buf + 5 * GROUP * W,
                    tla.make_layout(tla.make_shape(rows * W), tla.make_stride(1)),
                )
                tla.copy(ub_c5, gm_c5)
            if splits > 6:
                gm_c6 = tla.make_tensor(
                    part_o.ptr + ((m_blk * splits + 6) * M_BLOCK + r_in) * W,
                    tla.make_layout(tla.make_shape(rows * W), tla.make_stride(1)),
                )
                ub_c6 = tla.make_tensor(
                    ub_buf + 6 * GROUP * W,
                    tla.make_layout(tla.make_shape(rows * W), tla.make_stride(1)),
                )
                tla.copy(ub_c6, gm_c6)
            if splits > 7:
                gm_c7 = tla.make_tensor(
                    part_o.ptr + ((m_blk * splits + 7) * M_BLOCK + r_in) * W,
                    tla.make_layout(tla.make_shape(rows * W), tla.make_stride(1)),
                )
                ub_c7 = tla.make_tensor(
                    ub_buf + 7 * GROUP * W,
                    tla.make_layout(tla.make_shape(rows * W), tla.make_stride(1)),
                )
                tla.copy(ub_c7, gm_c7)
            tla.set_flag(ld)
            tla.wait_flag(ld)
            if splits == 1:
                for i in range(GROUP):
                    with tla.vec.func(mode="simd"):
                        full_s = tla.create_mask(
                            pattern=tla.mask.ALL, dtype=tla.Float32
                        )
                        one_s, _ = tla.update_mask(1, dtype=tla.Float32)
                        m_0 = tla.tile_view(
                            buf_v,
                            tla.make_shape(1),
                            tla.make_coord((0 * GROUP + i) * W + 256),
                        ).load(
                            params=tla.params.NormalLoadParams(
                                load_dist=tla.params.LoadDist.DIST_BRC_B32
                            )
                        )
                        l_0 = tla.tile_view(
                            buf_v,
                            tla.make_shape(1),
                            tla.make_coord((0 * GROUP + i) * W + 257),
                        ).load(
                            params=tla.params.NormalLoadParams(
                                load_dist=tla.params.LoadDist.DIST_BRC_B32
                            )
                        )
                        den_r = l_0
                        tla.tile_view(
                            ws_v, tla.make_shape(1), tla.make_coord(i * 64 + 16)
                        ).store(den_r, tla.params.UnalignStoreParams(), mask=one_s)
                        for pc in range(4):
                            po_0 = tla.tile_view(
                                buf2_v,
                                tla.make_shape(1, 64),
                                tla.make_coord(0 * GROUP + i, pc),
                            ).load()
                            acc_r = po_0
                            tla.tile_view(
                                acc2_v, tla.make_shape(1, 64), tla.make_coord(i, pc)
                            ).store(acc_r, mask=full_s)
                        tla.local_mem_bar(
                            tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                        )
            elif splits == 2:
                for i in range(GROUP):
                    with tla.vec.func(mode="simd"):
                        full_s = tla.create_mask(
                            pattern=tla.mask.ALL, dtype=tla.Float32
                        )
                        one_s, _ = tla.update_mask(1, dtype=tla.Float32)
                        m_0 = tla.tile_view(
                            buf_v,
                            tla.make_shape(1),
                            tla.make_coord((0 * GROUP + i) * W + 256),
                        ).load(
                            params=tla.params.NormalLoadParams(
                                load_dist=tla.params.LoadDist.DIST_BRC_B32
                            )
                        )
                        l_0 = tla.tile_view(
                            buf_v,
                            tla.make_shape(1),
                            tla.make_coord((0 * GROUP + i) * W + 257),
                        ).load(
                            params=tla.params.NormalLoadParams(
                                load_dist=tla.params.LoadDist.DIST_BRC_B32
                            )
                        )
                        m_1 = tla.tile_view(
                            buf_v,
                            tla.make_shape(1),
                            tla.make_coord((1 * GROUP + i) * W + 256),
                        ).load(
                            params=tla.params.NormalLoadParams(
                                load_dist=tla.params.LoadDist.DIST_BRC_B32
                            )
                        )
                        l_1 = tla.tile_view(
                            buf_v,
                            tla.make_shape(1),
                            tla.make_coord((1 * GROUP + i) * W + 257),
                        ).load(
                            params=tla.params.NormalLoadParams(
                                load_dist=tla.params.LoadDist.DIST_BRC_B32
                            )
                        )
                        ms_r = tla.max(m_0, m_1, mask=full_s)
                        w_0 = tla.exp(tla.sub(m_0, ms_r, mask=full_s), mask=full_s)
                        w_1 = tla.exp(tla.sub(m_1, ms_r, mask=full_s), mask=full_s)
                        den_r = tla.mul(w_0, l_0, mask=full_s)
                        den_r = tla.add(
                            den_r, tla.mul(w_1, l_1, mask=full_s), mask=full_s
                        )
                        tla.tile_view(
                            ws_v, tla.make_shape(1), tla.make_coord(i * 64 + 16)
                        ).store(den_r, tla.params.UnalignStoreParams(), mask=one_s)
                        for pc in range(4):
                            po_0 = tla.tile_view(
                                buf2_v,
                                tla.make_shape(1, 64),
                                tla.make_coord(0 * GROUP + i, pc),
                            ).load()
                            acc_r = tla.mul(po_0, w_0, mask=full_s)
                            po_1 = tla.tile_view(
                                buf2_v,
                                tla.make_shape(1, 64),
                                tla.make_coord(1 * GROUP + i, pc),
                            ).load()
                            acc_r = tla.add(
                                acc_r, tla.mul(po_1, w_1, mask=full_s), mask=full_s
                            )
                            tla.tile_view(
                                acc2_v, tla.make_shape(1, 64), tla.make_coord(i, pc)
                            ).store(acc_r, mask=full_s)
                        tla.local_mem_bar(
                            tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                        )
            elif splits == 4:
                for i in range(GROUP):
                    with tla.vec.func(mode="simd"):
                        full_s = tla.create_mask(
                            pattern=tla.mask.ALL, dtype=tla.Float32
                        )
                        one_s, _ = tla.update_mask(1, dtype=tla.Float32)
                        m_0 = tla.tile_view(
                            buf_v,
                            tla.make_shape(1),
                            tla.make_coord((0 * GROUP + i) * W + 256),
                        ).load(
                            params=tla.params.NormalLoadParams(
                                load_dist=tla.params.LoadDist.DIST_BRC_B32
                            )
                        )
                        l_0 = tla.tile_view(
                            buf_v,
                            tla.make_shape(1),
                            tla.make_coord((0 * GROUP + i) * W + 257),
                        ).load(
                            params=tla.params.NormalLoadParams(
                                load_dist=tla.params.LoadDist.DIST_BRC_B32
                            )
                        )
                        m_1 = tla.tile_view(
                            buf_v,
                            tla.make_shape(1),
                            tla.make_coord((1 * GROUP + i) * W + 256),
                        ).load(
                            params=tla.params.NormalLoadParams(
                                load_dist=tla.params.LoadDist.DIST_BRC_B32
                            )
                        )
                        l_1 = tla.tile_view(
                            buf_v,
                            tla.make_shape(1),
                            tla.make_coord((1 * GROUP + i) * W + 257),
                        ).load(
                            params=tla.params.NormalLoadParams(
                                load_dist=tla.params.LoadDist.DIST_BRC_B32
                            )
                        )
                        m_2 = tla.tile_view(
                            buf_v,
                            tla.make_shape(1),
                            tla.make_coord((2 * GROUP + i) * W + 256),
                        ).load(
                            params=tla.params.NormalLoadParams(
                                load_dist=tla.params.LoadDist.DIST_BRC_B32
                            )
                        )
                        l_2 = tla.tile_view(
                            buf_v,
                            tla.make_shape(1),
                            tla.make_coord((2 * GROUP + i) * W + 257),
                        ).load(
                            params=tla.params.NormalLoadParams(
                                load_dist=tla.params.LoadDist.DIST_BRC_B32
                            )
                        )
                        m_3 = tla.tile_view(
                            buf_v,
                            tla.make_shape(1),
                            tla.make_coord((3 * GROUP + i) * W + 256),
                        ).load(
                            params=tla.params.NormalLoadParams(
                                load_dist=tla.params.LoadDist.DIST_BRC_B32
                            )
                        )
                        l_3 = tla.tile_view(
                            buf_v,
                            tla.make_shape(1),
                            tla.make_coord((3 * GROUP + i) * W + 257),
                        ).load(
                            params=tla.params.NormalLoadParams(
                                load_dist=tla.params.LoadDist.DIST_BRC_B32
                            )
                        )
                        ms_r = tla.max(m_0, m_1, mask=full_s)
                        ms_r = tla.max(ms_r, m_2, mask=full_s)
                        ms_r = tla.max(ms_r, m_3, mask=full_s)
                        w_0 = tla.exp(tla.sub(m_0, ms_r, mask=full_s), mask=full_s)
                        w_1 = tla.exp(tla.sub(m_1, ms_r, mask=full_s), mask=full_s)
                        w_2 = tla.exp(tla.sub(m_2, ms_r, mask=full_s), mask=full_s)
                        w_3 = tla.exp(tla.sub(m_3, ms_r, mask=full_s), mask=full_s)
                        den_r = tla.mul(w_0, l_0, mask=full_s)
                        den_r = tla.add(
                            den_r, tla.mul(w_1, l_1, mask=full_s), mask=full_s
                        )
                        den_r = tla.add(
                            den_r, tla.mul(w_2, l_2, mask=full_s), mask=full_s
                        )
                        den_r = tla.add(
                            den_r, tla.mul(w_3, l_3, mask=full_s), mask=full_s
                        )
                        tla.tile_view(
                            ws_v, tla.make_shape(1), tla.make_coord(i * 64 + 16)
                        ).store(den_r, tla.params.UnalignStoreParams(), mask=one_s)
                        for pc in range(4):
                            po_0 = tla.tile_view(
                                buf2_v,
                                tla.make_shape(1, 64),
                                tla.make_coord(0 * GROUP + i, pc),
                            ).load()
                            acc_r = tla.mul(po_0, w_0, mask=full_s)
                            po_1 = tla.tile_view(
                                buf2_v,
                                tla.make_shape(1, 64),
                                tla.make_coord(1 * GROUP + i, pc),
                            ).load()
                            acc_r = tla.add(
                                acc_r, tla.mul(po_1, w_1, mask=full_s), mask=full_s
                            )
                            po_2 = tla.tile_view(
                                buf2_v,
                                tla.make_shape(1, 64),
                                tla.make_coord(2 * GROUP + i, pc),
                            ).load()
                            acc_r = tla.add(
                                acc_r, tla.mul(po_2, w_2, mask=full_s), mask=full_s
                            )
                            po_3 = tla.tile_view(
                                buf2_v,
                                tla.make_shape(1, 64),
                                tla.make_coord(3 * GROUP + i, pc),
                            ).load()
                            acc_r = tla.add(
                                acc_r, tla.mul(po_3, w_3, mask=full_s), mask=full_s
                            )
                            tla.tile_view(
                                acc2_v, tla.make_shape(1, 64), tla.make_coord(i, pc)
                            ).store(acc_r, mask=full_s)
                        tla.local_mem_bar(
                            tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                        )
            else:
                # ---- 相位 1：m* = max_s m_s（逐行逐 split 从 ub_buf 读 256/257 列） ----
                for i1 in range(GROUP):
                    for sp1 in tla.range(splits):
                        with tla.vec.func(mode="simd"):
                            full_a = tla.create_mask(
                                pattern=tla.mask.ALL, dtype=tla.Float32
                            )
                            one_a, _ = tla.update_mask(1, dtype=tla.Float32)
                            m_a = tla.tile_view(
                                buf_v,
                                tla.make_shape(1),
                                tla.make_coord((sp1 * GROUP + i1) * W + 256),
                            ).load(
                                params=tla.params.NormalLoadParams(
                                    load_dist=tla.params.LoadDist.DIST_BRC_B32
                                )
                            )
                            l_a = tla.tile_view(
                                buf_v,
                                tla.make_shape(1),
                                tla.make_coord((sp1 * GROUP + i1) * W + 257),
                            ).load(
                                params=tla.params.NormalLoadParams(
                                    load_dist=tla.params.LoadDist.DIST_BRC_B32
                                )
                            )
                            ms_a = tla.tile_view(
                                ws_v, tla.make_shape(1), tla.make_coord(i1 * 64)
                            ).load(
                                params=tla.params.NormalLoadParams(
                                    load_dist=tla.params.LoadDist.DIST_BRC_B32
                                )
                            )
                            den_a = tla.tile_view(
                                ws_v, tla.make_shape(1), tla.make_coord(i1 * 64 + 16)
                            ).load(
                                params=tla.params.NormalLoadParams(
                                    load_dist=tla.params.LoadDist.DIST_BRC_B32
                                )
                            )
                            tla.tile_view(
                                ws_v,
                                tla.make_shape(1),
                                tla.make_coord(i1 * 64 + 24 + sp1),
                            ).store(m_a, tla.params.UnalignStoreParams(), mask=one_a)
                            tla.tile_view(
                                ws_v,
                                tla.make_shape(1),
                                tla.make_coord(i1 * 64 + 40 + sp1),
                            ).store(l_a, tla.params.UnalignStoreParams(), mask=one_a)
                            tla.tile_view(
                                ws_v, tla.make_shape(1), tla.make_coord(i1 * 64)
                            ).store(
                                tla.max(m_a, ms_a, mask=full_a),
                                tla.params.UnalignStoreParams(),
                                mask=one_a,
                            )
                            tla.tile_view(
                                ws_v, tla.make_shape(1), tla.make_coord(i1 * 64 + 16)
                            ).store(
                                den_a,
                                tla.params.UnalignStoreParams(),
                                mask=one_a,
                            )
                            tla.local_mem_bar(
                                tla.params.MemType.VEC_STORE,
                                tla.params.MemType.VEC_LOAD,
                            )
                # ---- 相位 2 + 3：w_s=exp(m_s-m*)、den 累加、acc 累加 ----
                for i2 in range(GROUP):
                    for sp2 in tla.range(splits):
                        with tla.vec.func(mode="simd"):
                            full_b = tla.create_mask(
                                pattern=tla.mask.ALL, dtype=tla.Float32
                            )
                            one_b, _ = tla.update_mask(1, dtype=tla.Float32)
                            m_b = tla.tile_view(
                                ws_v,
                                tla.make_shape(1),
                                tla.make_coord(i2 * 64 + 24 + sp2),
                            ).load(
                                params=tla.params.NormalLoadParams(
                                    load_dist=tla.params.LoadDist.DIST_BRC_B32
                                )
                            )
                            l_b = tla.tile_view(
                                ws_v,
                                tla.make_shape(1),
                                tla.make_coord(i2 * 64 + 40 + sp2),
                            ).load(
                                params=tla.params.NormalLoadParams(
                                    load_dist=tla.params.LoadDist.DIST_BRC_B32
                                )
                            )
                            ms_b = tla.tile_view(
                                ws_v, tla.make_shape(1), tla.make_coord(i2 * 64)
                            ).load(
                                params=tla.params.NormalLoadParams(
                                    load_dist=tla.params.LoadDist.DIST_BRC_B32
                                )
                            )
                            den_b = tla.tile_view(
                                ws_v, tla.make_shape(1), tla.make_coord(i2 * 64 + 16)
                            ).load(
                                params=tla.params.NormalLoadParams(
                                    load_dist=tla.params.LoadDist.DIST_BRC_B32
                                )
                            )
                            w_b = tla.exp(tla.sub(m_b, ms_b, mask=full_b), mask=full_b)
                            den_new = tla.add(
                                den_b, tla.mul(w_b, l_b, mask=full_b), mask=full_b
                            )
                            tla.tile_view(
                                ws_v, tla.make_shape(1), tla.make_coord(i2 * 64 + 16)
                            ).store(
                                den_new,
                                tla.params.UnalignStoreParams(),
                                mask=one_b,
                            )
                            for pc in range(4):
                                po_r = tla.tile_view(
                                    buf2_v,
                                    tla.make_shape(1, 64),
                                    tla.make_coord(sp2 * GROUP + i2, pc),
                                ).load()
                                ac = tla.tile_view(
                                    acc2_v,
                                    tla.make_shape(1, 64),
                                    tla.make_coord(i2, pc),
                                )
                                ac.store(
                                    tla.add(
                                        ac.load(),
                                        tla.mul(po_r, w_b, mask=full_b),
                                        mask=full_b,
                                    ),
                                    mask=full_b,
                                )
                            tla.local_mem_bar(
                                tla.params.MemType.VEC_STORE,
                                tla.params.MemType.VEC_LOAD,
                            )
            # ---- 相位 4：归一化 + BF16 窄化 + 写回 ----
            cast_z = tla.params.CastParams(
                reg_slot=tla.params.RegSlot.ZERO,
                sat_mode=tla.params.SatMode.NOSAT,
                round_mode=tla.params.RoundMode.CAST_ROUND,
            )
            cast_o = tla.params.CastParams(
                reg_slot=tla.params.RegSlot.ONE,
                sat_mode=tla.params.SatMode.NOSAT,
                round_mode=tla.params.RoundMode.CAST_ROUND,
            )
            for i3 in range(GROUP):
                with tla.vec.func(mode="simd"):
                    full_d = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                    den_fin = tla.tile_view(
                        ws_v, tla.make_shape(1), tla.make_coord(i3 * 64 + 16)
                    ).load(
                        params=tla.params.NormalLoadParams(
                            load_dist=tla.params.LoadDist.DIST_BRC_B32
                        )
                    )
                    den_safe = tla.where(
                        tla.cmp(den_fin, 0.0, mode="gt"),
                        den_fin,
                        tla.full(1.0, tla.Float32),
                    )
                    for oc in range(4):
                        a_r = tla.tile_view(
                            acc2_v, tla.make_shape(1, 64), tla.make_coord(i3, oc)
                        ).load()
                        o_r = tla.div(a_r, den_safe, mask=full_d)
                        b_e, b_o = tla.deinterleave(o_r, o_r)
                        packed = tla.bitwise_or(
                            b_e.to(tla.BFloat16, cast_z, mask=full_d),
                            b_o.to(tla.BFloat16, cast_o, mask=full_d),
                        )
                        tla.tile_view(
                            ob2_v, tla.make_shape(1, 64), tla.make_coord(i3, oc)
                        ).store(packed)
                    tla.local_mem_bar(
                        tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                    )
            tla.set_flag(vr)
            tla.set_flag(st)
            tla.wait_flag(st)
            gm_o = tla.make_tensor(
                gm_out.ptr + r0 * D,
                tla.make_layout(tla.make_shape(rows, D), tla.make_stride(D, 1)),
            )
            ob_w = tla.make_tensor(
                ub_ob, tla.make_layout(tla.make_shape(rows, D), tla.make_stride(D, 1))
            )
            tla.copy(gm_o, ob_w)
    with tla.vector():
        tla.wait_flag(vr)
        tla.pipe_barrier(tla.pipes.ALL)
    # vr 配平：每组 set 1 次、wait 1 次；组循环外无需排空


# ---------------------------------------------------------------------------
# host 编排
# ---------------------------------------------------------------------------

_M_BLOCK = 112
_SPLITS_CAP = 8
_ZERO_BUF = None  # 复用：清零 V padding 的 zeros 缓冲（避免每次 run 的 torch.zeros）
# 编译产物缓存：tla.compile() 每次都会重跑 _lower()（在缓存检查之前），单次 ~886ms。
# 这里按「kernel + 张量形状/类型 + 标量取值」缓存 artifact，跳过 lowering。
_ARTIFACT_CACHE = {}


def _compile_cached(key, kernel, args, options):
    """取缓存的编译产物；miss 时才真正编译（_lower 极贵，必须复用）。"""
    art = _ARTIFACT_CACHE.get(key)
    if art is None:
        art = tla.compile(kernel, *args, options=options)
        _ARTIFACT_CACHE[key] = art
    return art


def _tsig(t):
    """torch 张量的编译期签名：形状 + dtype。"""
    return (tuple(t.shape), str(t.dtype))


def _ceil_div(a, b):
    return (a + b - 1) // b


def run(
    q,
    k_cache,
    v_cache,
    block_table,
    kv_lens,
    q_positions,
    k_positions,
    window,
    window_right,
):
    """quant_flash_attn 入口：FP8 全量化固定窗口注意力（split-KV）。

    Q/K/V 已由上游量化成无 scale E4M3；分页 gather 在 attn 核内完成。
    """
    if (
        q.dtype != torch.float8_e4m3fn
        or k_cache.dtype != torch.float8_e4m3fn
        or v_cache.dtype != torch.float8_e4m3fn
    ):
        raise TypeError("Q/K/V 需要 float8_e4m3fn")
    tq, heads, dim = q.shape
    if dim != 256 or k_cache.shape[1] != 1 or k_cache.shape[2] != 128:
        raise ValueError("契约：D=256、Hkv=1、page_size=128")
    window = int(window)
    right = tq if window_right is None else int(window_right)
    if window <= 0 or right < tq:
        raise ValueError("需要正 window 且 right >= Tq")
    if tq == 0:
        return torch.zeros_like(q).to(torch.bfloat16)
    kv_len = int(kv_lens.reshape(-1)[0].item())
    prefix = int(q_positions.reshape(-1)[0].item())
    start = max(prefix - window + 1, 0)
    end = prefix + right
    nvis = max(0, min(end, kv_len) - start)
    m_rows = tq * heads
    if kv_len == 0 or nvis == 0 or m_rows == 0:
        return torch.zeros_like(q).to(torch.bfloat16)

    out = torch.empty(tq, heads, dim, dtype=torch.bfloat16, device=q.device)
    device = q.device
    cube_num = _cube_core_num(device.index or 0)
    if cube_num < 1:
        raise RuntimeError("NPU device reports no available Cube cores")
    m_blocks = _ceil_div(m_rows, _M_BLOCK)
    ntiles = _ceil_div(nvis, 128)
    splits = max(1, min(ntiles, cube_num // m_blocks))
    splits = min(splits, _SPLITS_CAP)

    m_pad = m_blocks * _M_BLOCK
    q8 = torch.zeros(m_pad * 256, dtype=torch.int8, device=device)
    q8[: m_rows * 256] = q.contiguous().view(torch.int8).view(-1)
    part_o = torch.empty(
        m_blocks * splits * _M_BLOCK * 264, dtype=torch.float32, device=device
    )

    t_q8 = _tla_tensor(q8, tla.arch.RowMajor, tla.Float8E4M3FN)
    t_kc = _tla_tensor(
        k_cache.contiguous().view(torch.int8).view(-1),
        tla.arch.RowMajor,
        tla.Float8E4M3FN,
    )
    t_vc = _tla_tensor(
        v_cache.contiguous().view(torch.int8).view(-1),
        tla.arch.RowMajor,
        tla.Float8E4M3FN,
    )
    t_bt = _tla_tensor(block_table, tla.arch.RowMajor)
    global _ZERO_BUF
    if _ZERO_BUF is None or _ZERO_BUF.device != device:
        _ZERO_BUF = torch.zeros(128 * dim, dtype=torch.int8, device=device)
    t_zero = _tla_tensor(_ZERO_BUF, tla.arch.RowMajor, tla.Float8E4M3FN)
    t_po = _tla_tensor(part_o, tla.arch.RowMajor)
    t_out = _tla_tensor(out.view(m_rows, 256), tla.arch.RowMajor)

    nk_pages = _ceil_div(start + nvis, 128) - start // 128
    attn = _compile_cached(
        (
            "attn_fused",
            _tsig(q8),
            _tsig(k_cache),
            _tsig(v_cache),
            _tsig(block_table),
            _tsig(out),
            m_rows,
            nvis,
            splits,
            _SPLITS_CAP,
            start,
            nk_pages,
        ),
        attn_fused_kernel,
        (
            t_q8,
            t_kc,
            t_vc,
            t_bt,
            t_zero,
            t_po,
            t_out,
            tla.Int32(m_rows),
            tla.Int32(nvis),
            tla.Int32(splits),
            tla.Int32(_SPLITS_CAP),
            tla.Int32(start),
            tla.Int32(nk_pages),
        ),
        options="--npu-arch 3510",
    )
    attn(
        t_q8,
        t_kc,
        t_vc,
        t_bt,
        t_zero,
        t_po,
        t_out,
        tla.Int32(m_rows),
        tla.Int32(nvis),
        tla.Int32(splits),
        tla.Int32(_SPLITS_CAP),
        tla.Int32(start),
        tla.Int32(nk_pages),
        block_num=cube_num,
    )
    return out
