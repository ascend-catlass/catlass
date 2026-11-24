# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""NeoX RoPE on Q plus append K/V into a paged KV cache (fp32).

Per token ``t`` with ``slot = slot_mapping[t]``:

* for every Q head and every ``d < head_dim / 2``:
  ``q_out[t, d]        = q[t, d] * cos[t, d] - q[t, d + hd] * sin[t, d]``
  ``q_out[t, d + hd]   = q[t, d] * sin[t, d] + q[t, d + hd] * cos[t, d]``
* if ``slot >= 0``, copy the whole K/V token row into ``cache_k[slot]`` /
  ``cache_v[slot]``; a negative slot means "no write" (e.g. padded token).

高性能设计（相对"一线程一 token、逐元素标量 GM 访问"的基线实现）：

1. 批量 DMA 取代标量访存：每 ``rows_per_group`` 行（默认 4）对 q/cos/sin/k/v/q_out
   各发一次连续 MTE2/MTE3 搬移（1–4 KB/次）；标量 GM 访问次数从 O(元素数) 降到
   O(组数 × 张量数)，访存从"延迟受限"转为带宽受限。
2. 三 pipe 流水 + UB 双缓冲：``slots=2`` 组缓冲轮转，下一组的 MTE2 与当前组的
   VECTOR、MTE3 并行；``UB_BUDGET_BYTES`` 决定 ``rows_per_group``，保证两组常驻。
3. 向量化旋转：NeoX 前后半区各自连续，直接按 64 lane 取 tile 配对，无需跨 lane
   shuffle；同一行的 cos/sin 只读一次，在 ``HQ`` 个 head 间复用。
4. K/V 直通：K/V 只做 UB 中转、不参与向量运算，每 token 一次 1 KB 的 UB→GM 散播，
   与 Q 的写回共用同一条 MTE3。
5. 幂等尾组：末尾组回拉至以 ``T`` 结束，重复行重复写同值，省掉尾块特判分支。
6. 按需起核：``block_num = min(vector_core_num, 组数)``，小 shape 不唤醒空核。
7. 核内同步全部用 mutex（见下表），不使用 ``set_flag``/``wait_flag``；kernel 末尾
   的 ``pipe_barrier(ALL)`` 只负责退出前排空在途 DMA。

mutex 配对（每个 slot 一组，producer→consumer 一对一）：

| mutex | producer | consumer | 保护缓冲 |
| --- | --- | --- | --- |
| ``m_in`` | MTE2 | VECTOR | q / cos / sin |
| ``m_out`` | VECTOR | MTE3 | q_out 暂存 |
| ``m_kv`` | MTE2 | MTE3 | k / v |

q_out 使用独立暂存区而不是原地改写 q：同一块缓冲若跨越 MTE2→VECTOR→MTE3 三条
pipe，会串成一条三段锁链；独立暂存区让每个 mutex 保持一对一，锁链更短、更好验证，
UB 用量也仍在 2×4 行预算内。

组按 block 连续切分，末尾组回拉到以 ``T`` 结束，因此一行可能被处理两次；旋转与
cache 写入都是幂等的（同值重写），不需要尾块特判。

``head_dim % 16 != 0`` 时 token 行窄于 32 Byte DMA 粒度，自动回退到标量 SIMT
kernel（``rope_kv_cache_scalar_kernel``）。
"""

import catlass.tla as tla

THREADS = 128  # 标量回退路径的 SIMT 线程数
VL_LANES = 64  # fp32 单寄存器宽度（Ascend 950 AVE）
UB_BUDGET_BYTES = 224 * 1024  # 双槽缓冲总预算，留出 UB 上限（256 KB）的余量
MAX_ROWS_PER_GROUP = 4  # 实测最优：更深缓冲的锁链更长，反而变慢
SLOTS = 2  # 双缓冲：MTE2 / VECTOR / MTE3 三 pipe 覆盖式重叠


def _chunk_lanes(half):
    """取半个 head 的向量分块宽度：不超过一个寄存器，且整除 ``half``。

    这样每个旋转对的 lo/hi 半区都能落在同一组 lane 上，配对无需 shuffle；
    ``half <= VL_LANES`` 时一次算完整行，``D > 128`` 时按 64 lane 多趟。
    """
    if half <= VL_LANES:
        return half
    for lanes in range(VL_LANES, 0, -1):
        if half % lanes == 0:
            return lanes
    return 1


@tla.kernel
def rope_kv_cache_kernel(
    gm_q: tla.Tensor,  # [T, HQ * D] fp32
    gm_k: tla.Tensor,  # [T, HK * D] fp32
    gm_v: tla.Tensor,  # [T, HK * D] fp32
    gm_cos: tla.Tensor,  # [T, D / 2] fp32
    gm_sin: tla.Tensor,  # [T, D / 2] fp32
    gm_slot: tla.Tensor,  # [T] int32, negative disables the cache write
    gm_q_out: tla.Tensor,  # [T, HQ * D] fp32
    gm_ck: tla.Tensor,  # [S, HK * D] fp32, updated in place
    gm_cv: tla.Tensor,  # [S, HK * D] fp32, updated in place
    D: tla.Constexpr[int],
    HQ: tla.Constexpr[int],
    HK: tla.Constexpr[int],
    rows_per_group: tla.Constexpr[int],
    slots: tla.Constexpr[int],
) -> None:
    half = D // 2
    q_row = HQ * D
    kv_row = HK * D
    rows = rows_per_group
    chunk = _chunk_lanes(half)

    total = gm_q.origin_shape[0]
    block_count = tla.arch.block_num()
    block = tla.arch.block_idx()

    # 组按 block 连续切分；末尾组回拉到以 `total` 结束，重复行幂等。
    group_count = (total + rows - 1) // rows
    group_begin = block * group_count // block_count
    group_end = (block + 1) * group_count // block_count

    group_q = tla.make_shape(rows, q_row)
    group_cs = tla.make_shape(rows, half)
    group_kv = tla.make_shape(rows, kv_row)
    row_kv = tla.make_shape(1, kv_row)
    lane = tla.make_shape(1, chunk)

    # ---- 双槽 UB 缓冲 + 每槽三个 mutex（producer→consumer 一对一）----------
    # m_in : MTE2 -> VECTOR （q / cos / sin）
    # m_out: VECTOR -> MTE3 （q_out 暂存）
    # m_kv : MTE2 -> MTE3   （k / v，不经过向量）
    # 同一 mutex 在下一轮同槽被 producer 再次 lock 时，会等待上一轮 consumer
    # 释放，从而自动形成跨迭代的回绕依赖；无需额外的 set_flag / wait_flag。
    ub_q = [None] * slots
    ub_qo = [None] * slots
    ub_cos = [None] * slots
    ub_sin = [None] * slots
    ub_k = [None] * slots
    ub_v = [None] * slots
    m_in = [None] * slots
    m_out = [None] * slots
    m_kv = [None] * slots
    for slot in tla.range_constexpr(slots):
        ub_q[slot] = tla.make_tensor_like(
            tla.allocate(rows * q_row, tla.Float32, tla.AddressSpace.ub, 256),
            tla.tile_view(gm_q, group_q, tla.make_coord(0, 0)),
            tla.arch.RowMajor,
        )
        ub_qo[slot] = tla.make_tensor_like(
            tla.allocate(rows * q_row, tla.Float32, tla.AddressSpace.ub, 256),
            tla.tile_view(gm_q_out, group_q, tla.make_coord(0, 0)),
            tla.arch.RowMajor,
        )
        ub_cos[slot] = tla.make_tensor_like(
            tla.allocate(rows * half, tla.Float32, tla.AddressSpace.ub, 256),
            tla.tile_view(gm_cos, group_cs, tla.make_coord(0, 0)),
            tla.arch.RowMajor,
        )
        ub_sin[slot] = tla.make_tensor_like(
            tla.allocate(rows * half, tla.Float32, tla.AddressSpace.ub, 256),
            tla.tile_view(gm_sin, group_cs, tla.make_coord(0, 0)),
            tla.arch.RowMajor,
        )
        ub_k[slot] = tla.make_tensor_like(
            tla.allocate(rows * kv_row, tla.Float32, tla.AddressSpace.ub, 256),
            tla.tile_view(gm_k, group_kv, tla.make_coord(0, 0)),
            tla.arch.RowMajor,
        )
        ub_v[slot] = tla.make_tensor_like(
            tla.allocate(rows * kv_row, tla.Float32, tla.AddressSpace.ub, 256),
            tla.tile_view(gm_v, group_kv, tla.make_coord(0, 0)),
            tla.arch.RowMajor,
        )
        m_in[slot] = tla.mutex(resource="in", id=3 * slot)
        m_out[slot] = tla.mutex(resource="out", id=3 * slot + 1)
        m_kv[slot] = tla.mutex(resource="kv", id=3 * slot + 2)

    zero = tla.as_numeric(0)

    with tla.vector():
        for group_iter in tla.range((group_end - group_begin + slots - 1) // slots):
            for slot in tla.range_constexpr(slots):
                group = group_begin + group_iter * slots + slot
                if group < group_end:
                    # 末尾组对齐到 [total-rows, total)，与前一组合并覆盖尾部；
                    # 重复行写同值，因此不需要 partial-group 分支。
                    start = group * rows
                    last_start = total - rows
                    if start > last_start:
                        start = last_start

                    # --- MTE2: 整组搬入（每张量一次连续 DMA，1–4 KB）---------
                    m_in[slot].lock(pipe=tla.arch.MTE2)
                    tla.copy(
                        ub_q[slot],
                        tla.get_tile(gm_q, tla.make_coord(start, 0), group_q),
                    )
                    tla.copy(
                        ub_cos[slot],
                        tla.get_tile(gm_cos, tla.make_coord(start, 0), group_cs),
                    )
                    tla.copy(
                        ub_sin[slot],
                        tla.get_tile(gm_sin, tla.make_coord(start, 0), group_cs),
                    )
                    m_in[slot].unlock(pipe=tla.arch.MTE2)

                    # K/V 只经 UB 中转、由 MTE3 直接散播，因此与 MTE3 共用 m_kv：
                    # 下一轮 MTE2 覆盖同槽前，必然等到上一轮 MTE3 读完。
                    m_kv[slot].lock(pipe=tla.arch.MTE2)
                    tla.copy(
                        ub_k[slot],
                        tla.get_tile(gm_k, tla.make_coord(start, 0), group_kv),
                    )
                    tla.copy(
                        ub_v[slot],
                        tla.get_tile(gm_v, tla.make_coord(start, 0), group_kv),
                    )
                    m_kv[slot].unlock(pipe=tla.arch.MTE2)

                    # --- VECTOR: 逐 head/逐 64 lane 旋转 ---------------------
                    # lo 在 [h*D + c*chunk)，hi 在 [h*D + half + c*chunk)，两段都连续，
                    # 直接 load 成同宽 lane 做 4 乘 2 加减；cos/sin 只按 (row, c) 读。
                    m_in[slot].lock(pipe=tla.arch.VECTOR)
                    m_out[slot].lock(pipe=tla.arch.VECTOR)
                    with tla.vec.func(mode="simd"):
                        for row in tla.range_constexpr(rows):
                            for h in tla.range_constexpr(HQ):
                                for c in tla.range_constexpr(half // chunk):
                                    col = h * D + c * chunk
                                    lo = tla.get_tile(
                                        ub_q[slot], tla.make_coord(row, col), lane
                                    ).load()
                                    hi = tla.get_tile(
                                        ub_q[slot],
                                        tla.make_coord(row, col + half),
                                        lane,
                                    ).load()
                                    cos_v = tla.get_tile(
                                        ub_cos[slot],
                                        tla.make_coord(row, c * chunk),
                                        lane,
                                    ).load()
                                    sin_v = tla.get_tile(
                                        ub_sin[slot],
                                        tla.make_coord(row, c * chunk),
                                        lane,
                                    ).load()
                                    out_lo = tla.sub(
                                        tla.mul(lo, cos_v), tla.mul(hi, sin_v)
                                    )
                                    out_hi = tla.add(
                                        tla.mul(lo, sin_v), tla.mul(hi, cos_v)
                                    )
                                    tla.get_tile(
                                        ub_qo[slot], tla.make_coord(row, col), lane
                                    ).store(out_lo)
                                    tla.get_tile(
                                        ub_qo[slot],
                                        tla.make_coord(row, col + half),
                                        lane,
                                    ).store(out_hi)
                    m_out[slot].unlock(pipe=tla.arch.VECTOR)
                    m_in[slot].unlock(pipe=tla.arch.VECTOR)

                    # --- MTE3: 整组写回 Q，再逐 token 散播 K/V ---------------
                    # Q 一次连续 DMA 写完整组；K/V 目标行由 slot 决定，只能按行
                    # 各发一次 1 KB 级的 UB→GM 拷贝，但仍在 MTE3 上与 Q 重叠。
                    m_out[slot].lock(pipe=tla.arch.MTE3)
                    tla.copy(
                        tla.get_tile(gm_q_out, tla.make_coord(start, 0), group_q),
                        ub_qo[slot],
                    )
                    m_out[slot].unlock(pipe=tla.arch.MTE3)

                    m_kv[slot].lock(pipe=tla.arch.MTE3)
                    for row in tla.range_constexpr(rows):
                        dst = gm_slot[start + row].to(tla.Int32)
                        if dst >= zero:
                            tla.copy(
                                tla.tile_view(gm_ck, row_kv, tla.make_coord(dst, 0)),
                                tla.get_tile(
                                    ub_k[slot], tla.make_coord(row, 0), row_kv
                                ),
                            )
                            tla.copy(
                                tla.tile_view(gm_cv, row_kv, tla.make_coord(dst, 0)),
                                tla.get_tile(
                                    ub_v[slot], tla.make_coord(row, 0), row_kv
                                ),
                            )
                    m_kv[slot].unlock(pipe=tla.arch.MTE3)

        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def rope_kv_cache_scalar_kernel(
    gm_q: tla.Tensor,
    gm_k: tla.Tensor,
    gm_v: tla.Tensor,
    gm_cos: tla.Tensor,
    gm_sin: tla.Tensor,
    gm_slot: tla.Tensor,
    gm_q_out: tla.Tensor,
    gm_ck: tla.Tensor,
    gm_cv: tla.Tensor,
    D: tla.Constexpr[int],
    HQ: tla.Constexpr[int],
    HK: tla.Constexpr[int],
) -> None:
    """标量 SIMT 回退：``head_dim % 16 != 0`` 时 token 行不足 32 Byte DMA 粒度。

    纯标量路径不跨 pipe 复用缓冲（GM 直读直写 + 一个 SIMT 段），因此不需要 mutex；
    末尾 ``pipe_barrier(ALL)`` 保证退出前所有标量写回落在 GM。
    """
    t = gm_q.origin_shape[0]
    half = D // 2

    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=THREADS):
            tid, _, _ = tla.arch.thread_idx()
            start = tla.arch.block_idx() * THREADS + tid
            step = tla.arch.block_num() * THREADS

            for tok in tla.range(start, t, step):
                for h in tla.range(0, HQ, 1):
                    for d in tla.range(0, half, 1):
                        lo = h * D + d
                        hi = lo + half
                        cos_v = gm_cos[tok, d]
                        sin_v = gm_sin[tok, d]
                        lo_v = gm_q[tok, lo]
                        hi_v = gm_q[tok, hi]
                        gm_q_out[tok, lo] = lo_v * cos_v - hi_v * sin_v
                        gm_q_out[tok, hi] = lo_v * sin_v + hi_v * cos_v

                slot = gm_slot[tok]
                if slot >= 0:
                    for hk in tla.range(0, HK, 1):
                        for e in tla.range(0, D, 1):
                            col = hk * D + e
                            gm_ck[slot, col] = gm_k[tok, col]
                            gm_cv[slot, col] = gm_v[tok, col]

        tla.pipe_barrier(tla.pipes.ALL)


# ======================================================================================
# HOST side: shape checks, compile and launch.
# ======================================================================================

_COMPILED = {}
_CORES = {}


def _vector_cores(index):
    """缓存 vector_core_num：get_device_properties 单次约 25 us，不能进热路径。"""
    import torch

    count = _CORES.get(index)
    if count is None:
        count = int(torch.npu.get_device_properties(index).vector_core_num)
        _CORES[index] = count
    return count


def _group_rows(t, d, hq, hk):
    """每组行数：默认 4，按双槽 UB 预算收缩，且不超过 token 总数。

    单行 UB 占用 = (q + q_out + cos + sin + k + v) 字节。rows 只影响 UB 容量与
    流水深度，不影响结果；实测 4 行在带宽与锁链长度之间最优。
    """
    half = d // 2
    row_bytes = 2 * (hq * d + half + hk * d) * 4
    rows = min(MAX_ROWS_PER_GROUP, max(1, (UB_BUDGET_BYTES // SLOTS) // row_bytes))
    return min(rows, t)


def run(
    q,
    k,
    v,
    cos,
    sin,
    slot_mapping,
    cache_k,
    cache_v,
    num_q_heads,
    num_k_heads,
    head_dim,
):
    """Return rotated ``q`` of shape ``[T, num_q_heads * head_dim]``.

    ``cache_k`` / ``cache_v`` are written in place at ``slot_mapping[t]`` for
    every non-negative slot. ``cos`` / ``sin`` have shape ``[T, head_dim // 2]``.
    """
    import torch
    from catlass.tla.runtime import from_dlpack

    if q.dim() != 2 or k.dim() != 2 or v.dim() != 2:
        raise ValueError("q/k/v must be 2-D [T, heads * head_dim]")
    t = q.shape[0]
    d = int(head_dim)
    hq = int(num_q_heads)
    hk = int(num_k_heads)
    if d % 2 != 0:
        raise ValueError("head_dim must be even")
    if q.shape[1] != hq * d or k.shape[1] != hk * d or v.shape != k.shape:
        raise ValueError("q/k/v widths must match heads * head_dim")
    if tuple(cos.shape) != (t, d // 2) or tuple(sin.shape) != (t, d // 2):
        raise ValueError("cos/sin must be [T, head_dim // 2]")
    if tuple(slot_mapping.shape) != (t,):
        raise ValueError("slot_mapping must be [T]")
    if cache_k.dim() != 2 or cache_k.shape != cache_v.shape:
        raise ValueError("cache_k/cache_v must be 2-D of equal shape")
    if cache_k.shape[1] != hk * d:
        raise ValueError("cache width must be num_k_heads * head_dim")

    device = q.device
    for name, buf in (
        ("q", q),
        ("k", k),
        ("v", v),
        ("cos", cos),
        ("sin", sin),
        ("cache_k", cache_k),
        ("cache_v", cache_v),
    ):
        if buf.dtype != torch.float32:
            raise ValueError("{} must be float32".format(name))
        if not buf.is_contiguous():
            raise ValueError("{} must be contiguous".format(name))

    q_out = torch.empty_like(q)
    if t == 0:
        return q_out

    slot = slot_mapping.to(torch.int32).contiguous()

    device_index = int(str(device).split(":")[-1]) if ":" in str(device) else 0
    vector_cores = _vector_cores(device_index)

    def _t(buf):
        return from_dlpack(buf, layout_tag=tla.arch.RowMajor)

    q_t, k_t, v_t = _t(q), _t(k), _t(v)
    cos_t, sin_t, slot_t = _t(cos), _t(sin), _t(slot)
    qo_t, ck_t, cv_t = _t(q_out), _t(cache_k), _t(cache_v)
    tensors = (q_t, k_t, v_t, cos_t, sin_t, slot_t, qo_t, ck_t, cv_t)

    # token 行 >= 32 Byte 时走 DMA+SIMD 主路径；否则回退标量 SIMT。
    if d % 16 == 0:
        rows = _group_rows(t, d, hq, hk)
        key = ("fast", t, d, hq, hk, rows)
        artifact = _COMPILED.get(key)
        if artifact is None:
            artifact = tla.compile(
                rope_kv_cache_kernel,
                *tensors,
                d,
                hq,
                hk,
                rows,
                SLOTS,
                options="--npu-arch 3510",
            )
            _COMPILED[key] = artifact
        # 组数少于核数时不多起核：空核唤醒是固定开销，小 shape 下占主导。
        groups = (t + rows - 1) // rows
        blocks = min(vector_cores, groups)
        artifact(*tensors, block_num=blocks)
    else:
        key = ("scalar", t, d, hq, hk)
        artifact = _COMPILED.get(key)
        if artifact is None:
            artifact = tla.compile(
                rope_kv_cache_scalar_kernel,
                *tensors,
                d,
                hq,
                hk,
                options="--npu-arch 3510",
            )
            _COMPILED[key] = artifact
        artifact(*tensors, block_num=vector_cores)
    return q_out


if __name__ == "__main__":
    import torch
    import torch_npu

    torch.npu.set_device(0)
    torch.manual_seed(0)
    t, hq, hk, d, slots = 32, 4, 2, 64, 128
    q = torch.randn(t, hq * d, dtype=torch.float32, device="npu")
    k = torch.randn(t, hk * d, dtype=torch.float32, device="npu")
    v = torch.randn(t, hk * d, dtype=torch.float32, device="npu")
    cos = torch.rand(t, d // 2, dtype=torch.float32, device="npu")
    sin = torch.rand(t, d // 2, dtype=torch.float32, device="npu")
    slot = torch.arange(t, dtype=torch.int32, device="npu")
    ck = torch.zeros(slots, hk * d, dtype=torch.float32, device="npu")
    cv = torch.zeros(slots, hk * d, dtype=torch.float32, device="npu")

    q_out = run(q, k, v, cos, sin, slot, ck, cv, hq, hk, d)

    qr = q.cpu().view(t, hq, d)
    half = d // 2
    c = cos.cpu().unsqueeze(1)
    s = sin.cpu().unsqueeze(1)
    lo, hi = qr[..., :half], qr[..., half:]
    ref = torch.cat([lo * c - hi * s, lo * s + hi * c], dim=-1).view(t, hq * d)
    print("q max abs err:", (q_out.cpu() - ref).abs().max().item())
    print("cache_k err:", (ck.cpu()[:t] - k.cpu()).abs().max().item())
    print("cache_v err:", (cv.cpu()[:t] - v.cpu()).abs().max().item())
