# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""histc：CATLASS Python 模板实现（纯 AIV 多核）。

结构（两段式：并行统计 + 单核归并）：

  histc_range（仅 autorange，每核 min/max）
    → histc_hist（主 kernel：每核一段连续区间，窗口内 dhistv2 累加后写 partials 行）
    → histc_tail（全局尾块，仅 bins > 255 或 N < 256 时才单独起核）
    → histc_reduce（单核整块 2-D 搬运，按列求和 + 一次 cast；尾块可并入）

高性能设计点（对应关系见各 kernel docstring 与 README「实现说明」）：

1. 桶索引用 i8 承载、`DIST_PACK4_B32` 一次落 256B，再用 `tla.histograms`（dhistv2，
   BIN0/BIN1 各 128 桶）在寄存器内直方图化，避免逐元素标量累加。
2. 每趟窗口固定 255 列，key `255` 是哨兵「丢弃」：越界值与非对齐尾块的残留 lane 都只
   需生成哨兵键，不必在 SIMD region 内用动态 mask。
3. 不做跨核 GM 原子累加（实测 ≈15 ns/元素），改为每核写自己的非原子 partials 行，
   由单核整块搬运后按列求和：归并只有一次 2-D 读 + 256 次向量加。
4. 尾块优先并入 histc_reduce（核内读 x 的尾组 + `arange` 掩码），省掉一次 launch、
   一次宿主 staging 拷贝与 partials 的尾行。
5. MTE2（搬入）与 VECTOR（计算）之间用 `tla.mutex` 表达生产者/消费者；
   region 之间的 UB 顺序由 `pipe_barrier` 保证（同一 region 内 UB store→load 无顺序保证）。
6. 宿主侧压缩发射开销：缓存 launch payload / 流句柄 / `from_dlpack` 绑定 / 输出缓冲 /
   几何参数，热路径只做一次 ACL launch（详见 `_Lean`）。
"""

from __future__ import annotations

import catlass.tla as tla

TILE_ELE = 16384  # 每个 MTE2 tile 的元素数（UB 64 KB，减少每 tile 的 barrier 次数）
GROUP_ELE = 256  # dhistv2 一次处理的 key 数
WINDOW = 255  # 每趟窗口宽度（256 留一个哨兵键）
TRASH = 255  # 哨兵键
MAX_BINS = 4096

_CAST_FLOOR = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ZERO,
    sat_mode=tla.params.SatMode.NOSAT,
    round_mode=tla.params.RoundMode.CAST_FLOOR,
)
_CAST_TRUNC = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ZERO,
    sat_mode=tla.params.SatMode.NOSAT,
    round_mode=tla.params.RoundMode.CAST_TRUNC,
)
_CAST_RINT = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ZERO,
    sat_mode=tla.params.SatMode.NOSAT,
    round_mode=tla.params.RoundMode.CAST_RINT,
)
_CAST_F16 = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ZERO,
    sat_mode=tla.params.SatMode.NOSAT,
    round_mode=tla.params.RoundMode.CAST_NONE,
)
_PACK4 = tla.params.NormalStoreParams(store_dist=tla.params.StoreDist.DIST_PACK4_B32)
_HIST_BIN0 = tla.params.HistogramsParams(
    mode=tla.params.HistogramsBinType.BIN0,
    type=tla.params.HistogramsType.FREQUENCY,
)
_HIST_BIN1 = tla.params.HistogramsParams(
    mode=tla.params.HistogramsBinType.BIN1,
    type=tla.params.HistogramsType.FREQUENCY,
)
_UNPACK16 = tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_UNPACK_B16)
_ATOMIC_ADD = tla.params.CopyUbToGmParams(atomic_mode=tla.params.AtomicMode.ADD)


def _ub(ptr, count):
    return tla.make_tensor(
        ptr, tla.make_layout(tla.make_shape(count), tla.make_stride(1))
    )


def _row_tensor(base_tensor, offset, length):
    """在 base_tensor 的 ptr 上偏移 offset 个元素，逻辑形状 [length]。"""
    off = tla.Int64(offset) if isinstance(offset, int) else offset.to(tla.Int64)
    return tla.make_tensor(
        base_tensor.ptr + off,
        tla.make_layout(tla.make_shape(length), tla.make_stride(1)),
    )


def _dyn_tensor(base_tensor, length, origin):
    """UB/GM tile：物理容量 length，有效区 origin（运行时）。"""
    return tla.make_tensor(
        base_tensor.ptr,
        tla.make_layout(
            tla.make_shape(length),
            tla.make_stride(1),
            origin_shape=tla.make_shape(origin),
        ),
    )


def _to_f32_scalar(v, is_f16):
    """标量 f16 → f32；f32 输入原样返回。"""
    if is_f16:
        return v.to(tla.Float32)
    return v


def _load_f32_only(x_tile, sub):
    return tla.tile_view(x_tile, tla.make_shape(64), tla.make_coord(sub)).load()


def _load_f32_f16(x_tile, sub):
    reg = tla.tile_view(x_tile, tla.make_shape(128), tla.make_coord(sub // 2)).load()
    slot = tla.params.RegSlot.ZERO if sub % 2 == 0 else tla.params.RegSlot.ONE
    return reg.to(
        tla.Float32,
        tla.params.CastParams(
            reg_slot=slot,
            sat_mode=tla.params.SatMode.NOSAT,
            round_mode=tla.params.RoundMode.CAST_NONE,
        ),
    )


def _load_f32(x_tile, sub, is_f16):
    """取第 sub 个 64-lane f32 子寄存器（f16 输入按 2x 展宽，组内顺序无关）。"""
    if is_f16:
        reg = tla.tile_view(
            x_tile, tla.make_shape(128), tla.make_coord(sub // 2)
        ).load()
        slot = tla.params.RegSlot.ZERO if sub % 2 == 0 else tla.params.RegSlot.ONE
        return reg.to(
            tla.Float32,
            tla.params.CastParams(
                reg_slot=slot,
                sat_mode=tla.params.SatMode.NOSAT,
                round_mode=tla.params.RoundMode.CAST_NONE,
            ),
        )
    return tla.tile_view(x_tile, tla.make_shape(64), tla.make_coord(sub)).load()


def _emit_keys(
    x_tile,
    group_index,
    ub_keys,
    lo_v,
    hi_v,
    bins_v,
    delta_v,
    bins,
    base,
    single,
    is_f16,
    valid_v=None,
):
    """把一个 256 元素组的桶索引打包成 256 个 i8 key（4×64 → 256B）。

    快路径：`t = (x - lo) * bins / delta` 一次算出浮点桶位，`floor` 取整后
    `min(k, bins-1)` 把 `t == bins`（即 `x == max`）并入最后一桶；`t < 0` 或
    `t > bins` 直接置哨兵键 `TRASH`，因此不需要在 SIMD region 内做动态 mask。
    i8 用 `CAST_TRUNC` 截断（值域 0..255），最后 `DIST_PACK4_B32` 一次落 256B。

    多趟窗口（bins > 255）时 `single=False`：先把桶号减去本趟 `base`，落在
    `[0, 255)` 之外的（`hi_c != raw`）置哨兵，保证每趟只统计自己那 255 列。

    `valid_v` 给出该组内有效 lane 数（运行时）：只有 lane < valid_v 计入，其余
    置哨兵键。尾块用它把「向后对齐读到的重复前缀」丢掉。
    """
    trash = tla.full(TRASH, dtype=tla.Int32)
    k_max = tla.Int32(bins - 1)
    k_neg = tla.Int32(-1)
    w_max = tla.Int32(WINDOW - 1)
    for sub in range(4):
        xr = _load_f32(x_tile, group_index * 4 + sub, is_f16)
        t = tla.sub(xr, lo_v)
        t = tla.mul(t, bins_v)
        t = tla.div(t, delta_v)
        m_ge = tla.cmp(t, 0.0, mode="ge")
        m_le = tla.cmp(t, float(bins), mode="le")
        k = t.to(tla.Int32, _CAST_FLOOR)
        if single:
            k = tla.min(k, k_max)
            k = tla.where(m_ge, k, trash)
        else:
            raw = tla.sub(k, base)
            lo_c = tla.max(raw, k_neg)
            hi_c = tla.min(lo_c, w_max)
            k = tla.where(tla.cmp(hi_c, raw, mode="eq"), hi_c, trash)
            k = tla.where(m_ge, k, trash)
        k = tla.where(m_le, k, trash)
        if valid_v is not None:
            lane = tla.arange(
                base=(group_index * 4 + sub) * 64, order="increase", dtype=tla.Int32
            )
            k = tla.where(tla.cmp(lane, valid_v, mode="lt"), k, trash)
        k8 = k.to(tla.Int8, _CAST_TRUNC)
        tla.tile_view(
            ub_keys, tla.make_shape(64), tla.make_coord(group_index * 4 + sub)
        ).store(k8, _PACK4)


def _hist_groups(ub_keys, ngroups):
    """把 ub_keys 里 ngroups 个 256-key 组做 dhistv2，返回 i16 累加寄存器。"""
    hist_lo = tla.full(0, dtype=tla.Int16)
    hist_hi = tla.full(0, dtype=tla.Int16)
    for g in tla.range(ngroups):
        keys = tla.tile_view(
            ub_keys, tla.make_shape(GROUP_ELE), tla.make_coord(g)
        ).load()
        hist_lo = tla.histograms(keys, hist_lo, _HIST_BIN0)
        hist_hi = tla.histograms(keys, hist_hi, _HIST_BIN1)
    return hist_lo, hist_hi


def _store_h16(hist_lo, hist_hi, ub_h16):
    """两份 i16 直方图寄存器落到 UB（自然桶序）。"""
    tla.tile_view(ub_h16, tla.make_shape(128), tla.make_coord(0)).store(hist_lo)
    tla.tile_view(ub_h16, tla.make_shape(128), tla.make_coord(1)).store(hist_hi)


def _acc_from_h16(ub_h16, ub_acc):
    """UB 上的 i16 直方图 → `DIST_UNPACK_B16` 零扩展成 i32 → 累加到 UB 累加器。"""
    for i in range(4):
        reg16 = tla.tile_view(ub_h16, tla.make_shape(64), tla.make_coord(i)).load(
            _UNPACK16
        )
        wide = reg16.bitcast(tla.Int32)
        acc_t = tla.tile_view(ub_acc, tla.make_shape(64), tla.make_coord(i))
        acc_t.store(tla.add(acc_t.load(), wide))


def _zero_acc(ub_acc, nreg=4, dtype=None):
    """把 UB 上的累加器清零。"""
    zero = tla.full(0, dtype=tla.Int32 if dtype is None else dtype)
    for i in range(nreg):
        tla.tile_view(ub_acc, tla.make_shape(64), tla.make_coord(i)).store(zero)


def _store_acc_f32(ub_acc, ub_out, nreg=4):
    """i32 计数 → f32 → UB 输出 tile（自然顺序）。"""
    for i in range(nreg):
        reg = tla.tile_view(ub_acc, tla.make_shape(64), tla.make_coord(i)).load()
        tla.tile_view(ub_out, tla.make_shape(64), tla.make_coord(i)).store(
            reg.to(tla.Float32, _CAST_RINT)
        )


def _cast_to_out(ub_acc, ub_out, nreg, dtype_out):
    """f32 累加器按输出 dtype 一次 cast 写回 UB。"""
    for i in range(nreg):
        reg = tla.tile_view(ub_acc, tla.make_shape(64), tla.make_coord(i)).load()
        tla.tile_view(ub_out, tla.make_shape(64), tla.make_coord(i)).store(
            reg.to(dtype_out, _CAST_F16)
        )


@tla.kernel
def histc_hist(
    gm_x: tla.Tensor,
    gm_acc: tla.Tensor,
    gm_range: tla.Tensor,
    bins: tla.Constexpr[int],
    bins_pad: tla.Constexpr[int],
    is_f16: tla.Constexpr[int],
    n_main: tla.Int32,
) -> None:
    """主直方图 kernel：每核处理连续区间，窗口内 dhistv2 累加后写 partials 行。

    设计点：
    - 每核只处理 `[bid*chunk, (bid+1)*chunk)` 一段连续元素，chunk 按 256 对齐，块间无
      重叠、无跨核通信；partials 每核一行，写回是非原子的普通 MTE3。
    - tile 取 16K 元素（64 KB UB），使 1M 输入在 64 核下正好 1 tile/核，barrier 次数最少。
    - 每 256 元素一组：4×64 lane 折算成桶索引 → `DIST_PACK4_B32` 落 256B key →
      dhistv2（BIN0/BIN1）在寄存器内累加，i16 计数经 `DIST_UNPACK_B16` 零扩展成 i32
      后累加进 UB 累加器；key 落 UB 与 dhistv2 读之间必须 `pipe_barrier`。
    - 三处 SIMD region 依次是「生成 key / dhistv2 / 累加」，用 mutex 与 MTE2 握手
      （`mtx_tile` 在 MTE2 上 lock→copy→unlock，在 VECTOR 上 lock→计算→unlock），
      单缓冲因此天然串行，不需要额外释放 flag。
    """
    tile_ele = TILE_ELE
    group_ele = GROUP_ELE
    window = WINDOW
    passes = (bins + window - 1) // window
    single = bins <= window
    dtype_x = tla.Float16 if is_f16 else tla.Float32

    n = n_main
    bid = tla.arch.block_idx()
    nblk = tla.arch.block_num()

    chunk = (n + nblk - 1) // nblk
    chunk = ((chunk + group_ele - 1) // group_ele) * group_ele
    start = bid * chunk
    end = start + chunk
    end = min(end, n)
    end = max(end, start)
    n_tiles = (end - start + tile_ele - 1) // tile_ele

    ub_x = tla.make_tensor(
        tla.allocate(tile_ele, dtype_x, tla.AddressSpace.ub, 256),
        tla.make_layout(tla.make_shape(tile_ele), tla.make_stride(1)),
    )
    ub_keys = _ub(tla.allocate(tile_ele, tla.Int8, tla.AddressSpace.ub, 256), tile_ele)
    ub_h16 = _ub(tla.allocate(256, tla.Int16, tla.AddressSpace.ub, 256), 256)
    ub_acc = _ub(tla.allocate(256, tla.Int32, tla.AddressSpace.ub, 256), 256)
    ub_out = _ub(tla.allocate(256, tla.Float32, tla.AddressSpace.ub, 256), 256)

    # ub_x / ub_keys 由 MTE2 生产、VECTOR 消费：用一把 mutex 同时表达
    # 「写完好读」与「读完才能覆盖」，等价于原来的 MTE2↔VECTOR 双向 flag。
    mtx_tile = tla.mutex(resource="histc_tile", id=0)

    with tla.vector():
        for w in tla.range(passes):
            base = w * window
            with tla.vec.func(mode="simd"):
                _zero_acc(ub_acc)
            tla.pipe_barrier(tla.pipes.ALL)
            for ti in tla.range(n_tiles):
                here = start + ti * tile_ele
                rem = end - here
                rem = min(rem, tile_ele)
                gm_tile = tla.make_tensor(
                    gm_x.ptr + here.to(tla.Int64),
                    tla.make_layout(
                        tla.make_shape(tile_ele),
                        tla.make_stride(1),
                        origin_shape=tla.make_shape(rem),
                    ),
                )
                ub_tile = _dyn_tensor(ub_x, tile_ele, rem)
                ngroups = (rem + group_ele - 1) // group_ele

                mtx_tile.lock(pipe=tla.pipes.MTE2)
                tla.copy(ub_tile, gm_tile)
                mtx_tile.unlock(pipe=tla.pipes.MTE2)
                mtx_tile.lock(pipe=tla.pipes.VECTOR)
                with tla.vec.func(mode="simd"):
                    lo_v = tla.full(gm_range[0], dtype=tla.Float32)
                    hi_v = tla.full(gm_range[1], dtype=tla.Float32)
                    delta_v = tla.sub(hi_v, lo_v)
                    bins_v = tla.full(float(bins), dtype=tla.Float32)
                    for g in tla.range(ngroups):
                        _emit_keys(
                            ub_tile,
                            g,
                            ub_keys,
                            lo_v,
                            hi_v,
                            bins_v,
                            delta_v,
                            bins,
                            base,
                            single,
                            is_f16,
                        )
                tla.pipe_barrier(tla.pipes.ALL)
                with tla.vec.func(mode="simd"):
                    hist_lo = tla.full(0, dtype=tla.Int16)
                    hist_hi = tla.full(0, dtype=tla.Int16)
                    for g in tla.range(ngroups):
                        keys = tla.tile_view(
                            ub_keys, tla.make_shape(group_ele), tla.make_coord(g)
                        ).load()
                        hist_lo = tla.histograms(keys, hist_lo, _HIST_BIN0)
                        hist_hi = tla.histograms(keys, hist_hi, _HIST_BIN1)
                    _store_h16(hist_lo, hist_hi, ub_h16)
                tla.pipe_barrier(tla.pipes.ALL)
                with tla.vec.func(mode="simd"):
                    _acc_from_h16(ub_h16, ub_acc)
                mtx_tile.unlock(pipe=tla.pipes.VECTOR)

            # acc(i32) -> f32 -> ub_out -> partials[bid, base:base+window]
            with tla.vec.func(mode="simd"):
                _store_acc_f32(ub_acc, ub_out, 4)
            tla.pipe_barrier(tla.pipes.ALL)
            dst = _row_tensor(gm_acc, bid * bins_pad + base, window)
            src = tla.tile_view(ub_out, tla.make_shape(window), tla.make_coord(0))
            tla.copy(dst, src)
            tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def histc_tail(
    gm_stage: tla.Tensor,
    gm_acc: tla.Tensor,
    gm_range: tla.Tensor,
    bins: tla.Constexpr[int],
    bins_pad: tla.Constexpr[int],
    row: tla.Constexpr[int],
    is_f16: tla.Constexpr[int],
) -> None:
    """全局尾块（< 256 元素）：宿主已把尾块与 lo-1 填充拼进 256 元素 staging，单核处理。

    仅在「需要多趟窗口（bins > 255）」或「N < 256」时使用：前者要按窗口把计数写到
    partials 的不同列段，后者无法安全整组读取输入的 256 元素窗口。其它情况尾块直接
    并进 `histc_reduce`，省掉这次 launch。
    """
    group_ele = GROUP_ELE
    window = WINDOW
    passes = (bins + window - 1) // window
    single = bins <= window
    dtype_x = tla.Float16 if is_f16 else tla.Float32

    ub_tail = tla.make_tensor(
        tla.allocate(group_ele, dtype_x, tla.AddressSpace.ub, 256),
        tla.make_layout(tla.make_shape(group_ele), tla.make_stride(1)),
    )
    ub_keys = _ub(
        tla.allocate(group_ele, tla.Int8, tla.AddressSpace.ub, 256), group_ele
    )
    ub_h16 = _ub(tla.allocate(256, tla.Int16, tla.AddressSpace.ub, 256), 256)
    ub_acc = _ub(tla.allocate(256, tla.Int32, tla.AddressSpace.ub, 256), 256)
    ub_out = _ub(tla.allocate(256, tla.Float32, tla.AddressSpace.ub, 256), 256)

    # staging 一次搬运后整段只读：MTE2 生产、VECTOR 消费一把 mutex 即可
    mtx_stage = tla.mutex(resource="tail_stage", id=0)

    with tla.vector():
        # staging 在宿主侧是 (1, 256) 的 2-D 张量，这里用 2-D 视图搬运
        gm2 = tla.make_tensor(
            gm_stage.ptr,
            tla.make_layout(
                tla.make_shape(1, group_ele), tla.make_stride(group_ele, 1)
            ),
        )
        ub2 = tla.make_tensor(
            ub_tail.ptr,
            tla.make_layout(
                tla.make_shape(1, group_ele), tla.make_stride(group_ele, 1)
            ),
        )
        mtx_stage.lock(pipe=tla.pipes.MTE2)
        tla.copy(ub2, gm2)
        mtx_stage.unlock(pipe=tla.pipes.MTE2)
        mtx_stage.lock(pipe=tla.pipes.VECTOR)
        for w in tla.range(passes):
            base = w * window
            with tla.vec.func(mode="simd"):
                lo_v = tla.full(gm_range[0], dtype=tla.Float32)
                hi_v = tla.full(gm_range[1], dtype=tla.Float32)
                delta_v = tla.sub(hi_v, lo_v)
                bins_v = tla.full(float(bins), dtype=tla.Float32)
                _zero_acc(ub_acc)
                _emit_keys(
                    ub_tail,
                    0,
                    ub_keys,
                    lo_v,
                    hi_v,
                    bins_v,
                    delta_v,
                    bins,
                    base,
                    single,
                    is_f16,
                )
            tla.pipe_barrier(tla.pipes.ALL)
            with tla.vec.func(mode="simd"):
                hist_lo = tla.full(0, dtype=tla.Int16)
                hist_hi = tla.full(0, dtype=tla.Int16)
                keys = tla.tile_view(
                    ub_keys, tla.make_shape(group_ele), tla.make_coord(0)
                ).load()
                hist_lo = tla.histograms(keys, hist_lo, _HIST_BIN0)
                hist_hi = tla.histograms(keys, hist_hi, _HIST_BIN1)
                _store_h16(hist_lo, hist_hi, ub_h16)
            tla.pipe_barrier(tla.pipes.ALL)
            with tla.vec.func(mode="simd"):
                _acc_from_h16(ub_h16, ub_acc)
            tla.pipe_barrier(tla.pipes.ALL)
            with tla.vec.func(mode="simd"):
                _store_acc_f32(ub_acc, ub_out, 4)
            tla.pipe_barrier(tla.pipes.ALL)
            tla.copy(
                _row_tensor(gm_acc, row * bins_pad + base, window),
                tla.tile_view(ub_out, tla.make_shape(window), tla.make_coord(0)),
            )
            tla.pipe_barrier(tla.pipes.ALL)
        mtx_stage.unlock(pipe=tla.pipes.VECTOR)


@tla.kernel
def histc_range(gm_x: tla.Tensor, gm_pm: tla.Tensor) -> None:
    """每核求区间 min/max（仅 f32 autorange），写 partials_r 的两个平面。

    设计点：预置 `x[0]` 填充 tile 缓冲，非对齐尾块的残留 lane 只会是 `x` 自身元素，
    不会污染 min/max——这样就不必在 SIMD region 内使用动态 mask；跨核收口交给宿主侧
    两个 device 侧 `min()/max()`，不再多起一个 kernel。
    """
    tile_ele = TILE_ELE
    per_reg = 64
    dtype_x = tla.Float32
    big = 3.4028234663852886e38

    n = gm_x.origin_shape[0]
    bid = tla.arch.block_idx()
    nblk = tla.arch.block_num()
    chunk = (n + nblk - 1) // nblk
    start = bid * chunk
    end = start + chunk
    end = min(end, n)
    end = max(end, start)
    n_tiles = (end - start + tile_ele - 1) // tile_ele

    ub_x = tla.make_tensor(
        tla.allocate(tile_ele, dtype_x, tla.AddressSpace.ub, 256),
        tla.make_layout(tla.make_shape(tile_ele), tla.make_stride(1)),
    )
    ub_pair = _ub(tla.allocate(128, tla.Float32, tla.AddressSpace.ub, 256), 128)

    # ub_x 由 MTE2 生产、VECTOR 消费；先让 VECTOR 持锁做 seed，再交给 MTE2 覆盖
    mtx_x = tla.mutex(resource="range_tile", id=0)

    with tla.vector():
        # 用 x[0] 预填 tile 缓冲：非对齐尾块残留的 lane 只会是 x 自身的元素，
        # 不会污染 min/max（避免在 SIMD region 内使用动态 mask）。
        mtx_x.lock(pipe=tla.pipes.VECTOR)
        with tla.vec.func(mode="simd"):
            seed_x = tla.full(gm_x[0], dtype=dtype_x)
            for i in range(tile_ele // 64):
                tla.tile_view(ub_x, tla.make_shape(64), tla.make_coord(i)).store(seed_x)
            # 空核（无 tile）也要写出恒等值，避免 ub_pair 未初始化
            tla.tile_view(ub_pair, tla.make_shape(64), tla.make_coord(0)).store(
                tla.full(big, dtype=tla.Float32)
            )
            tla.tile_view(ub_pair, tla.make_shape(64), tla.make_coord(1)).store(
                tla.full(-big, dtype=tla.Float32)
            )
        tla.pipe_barrier(tla.pipes.ALL)
        mtx_x.unlock(pipe=tla.pipes.VECTOR)
        for ti in tla.range(n_tiles):
            here = start + ti * tile_ele
            rem = end - here
            rem = min(rem, tile_ele)
            gm_tile = tla.make_tensor(
                gm_x.ptr + here.to(tla.Int64),
                tla.make_layout(
                    tla.make_shape(tile_ele),
                    tla.make_stride(1),
                    origin_shape=tla.make_shape(rem),
                ),
            )
            ub_tile = _dyn_tensor(ub_x, tile_ele, rem)
            nregs = (rem + per_reg - 1) // per_reg
            mtx_x.lock(pipe=tla.pipes.MTE2)
            tla.copy(ub_tile, gm_tile)
            mtx_x.unlock(pipe=tla.pipes.MTE2)
            mtx_x.lock(pipe=tla.pipes.VECTOR)
            with tla.vec.func(mode="simd"):
                acc_max = tla.full(-big, dtype=tla.Float32)
                acc_min = tla.full(big, dtype=tla.Float32)
                for g in tla.range(nregs):
                    xr = tla.tile_view(
                        ub_tile, tla.make_shape(64), tla.make_coord(g)
                    ).load()
                    acc_max = tla.max(acc_max, xr)
                    acc_min = tla.min(acc_min, xr)
                tla.tile_view(ub_pair, tla.make_shape(64), tla.make_coord(0)).store(
                    tla.full(
                        acc_min.reduce(
                            tla.ReductionOp.MIN,
                            mask=tla.create_mask(
                                pattern=tla.mask.ALL, dtype=tla.Float32
                            ),
                        ),
                        dtype=tla.Float32,
                    )
                )
                tla.tile_view(ub_pair, tla.make_shape(64), tla.make_coord(1)).store(
                    tla.full(
                        acc_max.reduce(
                            tla.ReductionOp.MAX,
                            mask=tla.create_mask(
                                pattern=tla.mask.ALL, dtype=tla.Float32
                            ),
                        ),
                        dtype=tla.Float32,
                    )
                )
            mtx_x.unlock(pipe=tla.pipes.VECTOR)
        tla.pipe_barrier(tla.pipes.ALL)

        tla.copy(
            _row_tensor(gm_pm, bid * nblk, nblk),
            tla.tile_view(ub_pair, tla.make_shape(64), tla.make_coord(0)),
        )
        tla.copy(
            _row_tensor(gm_pm, nblk * nblk + bid * nblk, nblk),
            tla.tile_view(ub_pair, tla.make_shape(64), tla.make_coord(1)),
        )
        tla.pipe_barrier(tla.pipes.ALL)


def _sum_rows(ub_blk, ub_acc, nrow, nsub):
    """把 UB 上 [nrow, 256] 的 partials 块按列求和到 ub_acc（simd region 内使用）。"""
    for j in range(nsub):
        acc = tla.full(0.0, dtype=tla.Float32)
        for r in range(nrow):
            reg = tla.tile_view(
                ub_blk, tla.make_shape(1, 64), tla.make_coord(r, j)
            ).load()
            acc = tla.add(acc, reg)
        tla.tile_view(ub_acc, tla.make_shape(64), tla.make_coord(j)).store(acc)


def _add_f32_from_i32(ub_src, ub_acc, nreg):
    """把 UB 上的 i32 计数（≤256 lane）分片转 f32 加到 f32 累加器（simd region 内使用）。"""
    for i in range(nreg):
        wide = tla.tile_view(ub_src, tla.make_shape(64), tla.make_coord(i)).load()
        acc_t = tla.tile_view(ub_acc, tla.make_shape(64), tla.make_coord(i))
        acc_t.store(tla.add(acc_t.load(), wide.to(tla.Float32, _CAST_RINT)))


@tla.kernel
def histc_reduce(
    gm_partials: tla.Tensor,
    gm_out: tla.Tensor,
    gm_x: tla.Tensor,
    gm_range: tla.Tensor,
    bins: tla.Constexpr[int],
    bins_pad: tla.Constexpr[int],
    nrow: tla.Constexpr[int],
    has_tail: tla.Constexpr[int],
    tail_off: tla.Int32,
    tail_n: tla.Int32,
    is_f16: tla.Constexpr[int],
) -> None:
    """单核把 partials 的 nrow 行按列求和（整块 2-D 搬运），按输出 dtype 一次 cast。

    `has_tail` 时本核顺带算全局尾块：直接从 `gm_x[tail_off : tail_off+256]` 搬
    一组（`tail_off` 为 256 对齐的尾部起点，最多向后越界 < 256 元素），再用
    `valid_v=tail_n` 把越界/重复的 lane 置哨兵键。尾块计数在 cast 之前加进同一
    列空间，省掉一次 launch、一次宿主 staging 拷贝与 partials 的第 nblk 行。
    仅在 `bins <= 255`（单趟窗口）且 `n >= 256` 时启用。

    注：曾试过把列切成 4 段分给 4 个 block 并行（列不重叠、无需跨核归并），实测
    反而更慢（histc-004 48 → 103 µs）——该环境里 launch/dispatch 固定开销远大于
    reduce 本身的计算，故保持 1 block。
    """
    ch = 256
    sub = 64
    nsub = ch // sub
    nchunks = (bins + ch - 1) // ch
    window = WINDOW
    single = bins <= window
    dtype_out = tla.Float16 if is_f16 else tla.Float32

    ub_blk = tla.make_tensor(
        tla.allocate(nrow * ch, tla.Float32, tla.AddressSpace.ub, 256),
        tla.make_layout(tla.make_shape(nrow, ch), tla.make_stride(ch, 1)),
    )
    ub_acc = _ub(tla.allocate(ch, tla.Float32, tla.AddressSpace.ub, 256), ch)
    ub_out = _ub(tla.allocate(ch, dtype_out, tla.AddressSpace.ub, 256), ch)
    ub_tail = _ub(tla.allocate(ch, tla.Float32, tla.AddressSpace.ub, 256), ch)
    ub_tkeys = _ub(tla.allocate(ch, tla.Int8, tla.AddressSpace.ub, 256), ch)
    ub_th16 = _ub(tla.allocate(ch, tla.Int16, tla.AddressSpace.ub, 256), ch)
    ub_tacc = _ub(tla.allocate(ch, tla.Int32, tla.AddressSpace.ub, 256), ch)

    # 两个 GM→UB 搬运（尾块、partials）串行使用各自的 UB：一把 mutex 表达
    # MTE2 生产 / VECTOR 消费的顺序即可
    mtx_tile = tla.mutex(resource="red_tile", id=0)

    with tla.vector():
        if has_tail:
            # 与 partials 的同模板 2-D 拷贝保持同一 memref 形状（避免 1-D/2-D 混用）
            gm2 = tla.make_tensor(
                gm_x.ptr + tail_off.to(tla.Int64),
                tla.make_layout(tla.make_shape(1, ch), tla.make_stride(ch, 1)),
            )
            ub2 = tla.make_tensor(
                ub_tail.ptr,
                tla.make_layout(tla.make_shape(1, ch), tla.make_stride(ch, 1)),
            )
            mtx_tile.lock(pipe=tla.pipes.MTE2)
            tla.copy(ub2, gm2)
            mtx_tile.unlock(pipe=tla.pipes.MTE2)
            mtx_tile.lock(pipe=tla.pipes.VECTOR)
            with tla.vec.func(mode="simd"):
                lo_v = tla.full(gm_range[0], dtype=tla.Float32)
                hi_v = tla.full(gm_range[1], dtype=tla.Float32)
                delta_v = tla.sub(hi_v, lo_v)
                bins_v = tla.full(float(bins), dtype=tla.Float32)
                _zero_acc(ub_tacc)
                _emit_keys(
                    ub_tail,
                    0,
                    ub_tkeys,
                    lo_v,
                    hi_v,
                    bins_v,
                    delta_v,
                    bins,
                    0,
                    single,
                    is_f16,
                    valid_v=tail_n,
                )
            tla.pipe_barrier(tla.pipes.ALL)
            with tla.vec.func(mode="simd"):
                hist_lo = tla.full(0, dtype=tla.Int16)
                hist_hi = tla.full(0, dtype=tla.Int16)
                keys = tla.tile_view(
                    ub_tkeys, tla.make_shape(ch), tla.make_coord(0)
                ).load()
                hist_lo = tla.histograms(keys, hist_lo, _HIST_BIN0)
                hist_hi = tla.histograms(keys, hist_hi, _HIST_BIN1)
                _store_h16(hist_lo, hist_hi, ub_th16)
            tla.pipe_barrier(tla.pipes.ALL)
            with tla.vec.func(mode="simd"):
                _acc_from_h16(ub_th16, ub_tacc)
            tla.pipe_barrier(tla.pipes.ALL)
            mtx_tile.unlock(pipe=tla.pipes.VECTOR)
        for c in tla.range(nchunks):
            col = c * ch
            gm_tile = tla.make_tensor(
                gm_partials.ptr + col.to(tla.Int64),
                tla.make_layout(tla.make_shape(nrow, ch), tla.make_stride(bins_pad, 1)),
            )
            ub_tile = tla.make_tensor(
                ub_blk.ptr,
                tla.make_layout(tla.make_shape(nrow, ch), tla.make_stride(ch, 1)),
            )
            mtx_tile.lock(pipe=tla.pipes.MTE2)
            tla.copy(ub_tile, gm_tile)
            mtx_tile.unlock(pipe=tla.pipes.MTE2)
            mtx_tile.lock(pipe=tla.pipes.VECTOR)
            with tla.vec.func(mode="simd"):
                _sum_rows(ub_blk, ub_acc, nrow, nsub)
            tla.pipe_barrier(tla.pipes.ALL)
            if has_tail:
                # has_tail 只在单 chunk（bins <= 255）下启用，c 恒为 0
                with tla.vec.func(mode="simd"):
                    _add_f32_from_i32(ub_tacc, ub_acc, nsub)
                tla.pipe_barrier(tla.pipes.ALL)
            with tla.vec.func(mode="simd"):
                _cast_to_out(ub_acc, ub_out, nsub, dtype_out)
            tla.pipe_barrier(tla.pipes.ALL)
            left = bins - col
            left = min(left, ch)
            dst = tla.make_tensor(
                gm_out.ptr + col.to(tla.Int64),
                tla.make_layout(
                    tla.make_shape(ch),
                    tla.make_stride(1),
                    origin_shape=tla.make_shape(left),
                ),
            )
            src = tla.make_tensor(
                ub_out.ptr,
                tla.make_layout(
                    tla.make_shape(ch),
                    tla.make_stride(1),
                    origin_shape=tla.make_shape(left),
                ),
            )
            mtx_tile.unlock(pipe=tla.pipes.VECTOR)
            tla.copy(dst, src)
            tla.pipe_barrier(tla.pipes.ALL)


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------


_ARTIFACTS = {}
_CACHED = {}
_CORE_CACHE = {}
_WRAP_CACHE = {}
_I32_CACHE = {}
_GEOM_CACHE = {}
_OUT_CACHE = {}
_F32_CACHE = {}
_TAIL_PAD = {}


def _wrap(host):
    """按 data_ptr 复用 `from_dlpack` 绑定（它本身拖住 host 张量，指针不会被复用）。"""
    key = (host.data_ptr(), host.numel(), host.dtype, host.device.index)
    got = _WRAP_CACHE.get(key)
    if got is None:
        if len(_WRAP_CACHE) > 8:
            _WRAP_CACHE.clear()
        got = tla.from_dlpack(host, layout_tag=tla.arch.RowMajor)
        _WRAP_CACHE[key] = got
    return got


def _i32(value):
    """缓存 Int32 标量包装，省掉每次 launch 的构造。"""
    got = _I32_CACHE.get(value)
    if got is None:
        got = tla.Int32(value)
        _I32_CACHE[value] = got
    return got


def _geom(bins):
    """bins_pad 只与 bins 有关，避免每次重算。"""
    got = _GEOM_CACHE.get(bins)
    if got is None:
        passes = (bins + WINDOW - 1) // WINDOW
        got = ((passes * WINDOW + 255) // 256) * 256
        _GEOM_CACHE[bins] = got
    return got


def _out_buffer(bins, device, dev_index):
    """复用 f32 输出缓冲（`histc_reduce` 会写满全部 bins 列，无需清零）。"""
    import torch

    key = (bins, dev_index, str(device))
    got = _OUT_CACHE.get(key)
    if got is None:
        if len(_OUT_CACHE) > 8:
            _OUT_CACHE.clear()
        host = torch.empty(bins, dtype=torch.float32, device=device)
        got = (host, tla.from_dlpack(host, layout_tag=tla.arch.RowMajor))
        _OUT_CACHE[key] = got
    return got


def _f32_stage(n, device, dev_index):
    """f16 输入用的 f32 升精度缓冲（复用，避免每次新建张量与重新绑定）。"""
    import torch

    key = ("f32_stage", n, dev_index)
    got = _F32_CACHE.get(key)
    if got is None:
        if len(_F32_CACHE) > 4:
            _F32_CACHE.clear()
        got = torch.empty(n, dtype=torch.float32, device=device)
        _F32_CACHE[key] = got
    return got


def _buffer(key, shape, dtype, device):
    """按 key 复用 GM 工作区，并返回其 tla 包装。"""
    import torch

    got = _CACHED.get(key)
    if got is None:
        host = torch.empty(shape, dtype=dtype, device=device)
        got = (host, tla.from_dlpack(host, layout_tag=tla.arch.RowMajor))
        _CACHED[key] = got
    return got[1]


def _range_const(lo, hi, device, dev_index):
    """显式范围：把 [lo, hi] 缓存成 2 元素 f32 GM 张量（同值复用，无 H2D 热开销）。"""
    import torch

    key = ("range_const", lo, hi, dev_index)
    got = _CACHED.get(key)
    if got is None:
        host = torch.tensor([lo, hi], dtype=torch.float32, device=device)
        got = (host, tla.from_dlpack(host, layout_tag=tla.arch.RowMajor))
        _CACHED[key] = got
    return got[1]


def _cores(device_index):
    import torch

    got = _CORE_CACHE.get(device_index)
    if got is None:
        got = int(torch.npu.get_device_properties(device_index).vector_core_num)
        _CORE_CACHE[device_index] = got
    return got


def _artifact(key, kernel, args, options="--npu-arch 3510"):
    got = _ARTIFACTS.get(key)
    if got is None:
        got = _Lean(tla.compile(kernel, *args, options=options))
        _ARTIFACTS[key] = got
    return got


class _Lean:
    """精简发射：缓存 流句柄 / launch payload / ABI 常量，只留一次 ACL launch。

    `tla.compile` 的 `__call__` 每次都会重新打包 payload 并查询当前流，
    在 host 绑定的场景下（32.8 µs 里只有 ≈3 µs 是真正的 ACL 调用）占了大头。
    运行时参数对象（张量包装、缓存的 Int32）在 `_CACHE` 里长期存活，且本对象
    持有 args 引用，因此用 `id()` 做 payload 缓存键是安全的。
    """

    def __init__(self, art):
        from catlass import execution as _execution
        from catlass.base_dsl.jit_executor import _resolved_dyn_ubuf_bytes

        self._art = art
        ex = art._get_executor()
        module = ex.jit_module
        self._launch = _execution.launch_kernel
        self._gen = module.execution_args.generate_launch_payload
        self._fh = ex.function_handle
        self._dev = ex.device
        self._dyn_ub = _resolved_dyn_ubuf_bytes(module, 0)
        self._pc = module.print_config
        self._lean_ok = self._pc.print_metadata is None
        self._stream = None
        self._key = None
        self._payload = None
        self._keep = None

    def __call__(self, *args, block_num=1):
        if not self._lean_ok:
            return self._art(*args, block_num=block_num)
        key = tuple(id(a) for a in args)
        if key != self._key:
            self._payload = self._gen(args)
            self._keep = args
            self._key = key
        if self._stream is None:
            from catlass.base_dsl.runtime.ascend_stream_adapter import current_stream

            self._stream = current_stream(self._dev)
        self._launch(
            function_handle=self._fh,
            stream=self._stream,
            block_num=block_num,
            payload=self._payload,
            dyn_ubuf_bytes=self._dyn_ub,
            print_config=self._pc,
            device_id=self._dev,
        )


def run(x, bins, min, max):
    """Return the histogram of ``x`` with ``bins`` buckets over ``[min, max]``."""
    import torch

    bins = int(bins)
    lo = float(min)
    hi = float(max)
    if x.dim() != 1:
        x = x.reshape(-1)
    if not x.is_contiguous():
        x = x.contiguous()
    if x.dtype not in (torch.float32, torch.float16):
        raise TypeError(f"histc supports float32/float16, got {x.dtype}")
    if bins < 1 or bins > MAX_BINS:
        raise ValueError(f"bins must be in [1, {MAX_BINS}]")

    out_dtype = x.dtype
    n = int(x.numel())
    if n == 0:
        return torch.zeros(bins, dtype=out_dtype, device=x.device)

    # f16 输入先在宿主侧升到 f32（kernel 只走 f32 路径），输出再一次性 cast 回 f16
    dev_index = x.device.index if x.device.index is not None else 0
    if x.dtype == torch.float16:
        stage = _f32_stage(n, x.device, dev_index)
        stage.copy_(x)
        x = stage
    autorange = 1 if (lo == 0.0 and hi == 0.0) else 0
    is_f16 = 0
    nblk = _cores(dev_index)
    bins_pad = _geom(bins)
    n_main = (n // GROUP_ELE) * GROUP_ELE
    # 尾块（n % 256）并进 histc_reduce，仅限单趟窗口（bins <= 255）且 n >= 256：
    # 多趟需要逐窗口写不同列段、n < 256 无法安全整组读取，仍走独立尾核
    has_tail = 1 if n > n_main else 0
    merged_tail = 1 if (has_tail and bins <= WINDOW and n >= GROUP_ELE) else 0
    nrow = nblk + (1 if (has_tail and not merged_tail) else 0)

    t_x = _wrap(x)
    out_host, t_out = _out_buffer(bins, x.device, dev_index)
    t_partials = _buffer(
        ("partials", nrow, bins_pad, dev_index),
        (nrow, bins_pad),
        torch.float32,
        x.device,
    )

    if autorange:
        t_range = _buffer(("range", dev_index), (2,), torch.float32, x.device)
        t_pm = _buffer(
            ("pm", nblk, dev_index), (2, nblk * nblk), torch.float32, x.device
        )
        if is_f16:
            raise NotImplementedError("autorange 暂只支持 float32")
        art = _artifact("range", histc_range, [t_x, t_pm])
        art(t_x, t_pm, block_num=nblk)
        # 跨核 min/max 收口用 2 个 device 侧归约（避免再起一个 kernel）
        pm_host = _CACHED[("pm", nblk, dev_index)][0]
        rng_host = _CACHED[("range", dev_index)][0]
        rng_host[0].copy_(pm_host[0].min())
        rng_host[1].copy_(pm_host[1].max())
    else:
        t_range = _range_const(lo, hi, x.device, dev_index)

    art = _artifact(
        ("hist", bins, is_f16),
        histc_hist,
        [t_x, t_partials, t_range, bins, bins_pad, is_f16, tla.Int32(n_main)],
    )
    n_main_s = _i32(n_main)
    art(t_x, t_partials, t_range, n_main_s, block_num=nblk)

    if has_tail and not merged_tail:
        # 多趟窗口（bins > 255）或 n < 256：仍用独立尾核 + 宿主 staging
        t_stage = _buffer(
            ("tail_stage", is_f16, dev_index), (1, GROUP_ELE), x.dtype, x.device
        )
        stage_host = _CACHED[("tail_stage", is_f16, dev_index)][0].view(-1)
        n_tail = n - n_main
        # 填充只需在「lo 或尾块长度变化」时做一次（热循环里同一 shape 可跳过）
        padded = _TAIL_PAD.get((is_f16, dev_index))
        if padded is None or padded[0] != lo or padded[1] != n_tail:
            stage_host.fill_(lo - 1.0)
            _TAIL_PAD[(is_f16, dev_index)] = (lo, n_tail)
        stage_host[:n_tail].copy_(x[n_main:])
        art = _artifact(
            ("tail", bins, is_f16),
            histc_tail,
            [t_stage, t_partials, t_range, bins, bins_pad, nblk, is_f16],
        )
        art(t_stage, t_partials, t_range, block_num=1)

    art = _artifact(
        ("reduce", bins, is_f16, merged_tail),
        histc_reduce,
        [
            t_partials,
            t_out,
            t_x,
            t_range,
            bins,
            bins_pad,
            nrow,
            merged_tail,
            tla.Int32(n_main),
            tla.Int32(0),
            is_f16,
        ],
    )
    art(t_partials, t_out, t_x, t_range, _i32(n_main), _i32(n - n_main), block_num=1)
    return out_host if out_dtype == torch.float32 else out_host.to(out_dtype)
