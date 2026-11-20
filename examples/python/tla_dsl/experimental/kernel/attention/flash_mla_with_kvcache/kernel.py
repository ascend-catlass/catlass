# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Flash MLA decoding kernels: paged multi-head latent attention over a KV cache.

``task = (query row, head group)``, ``tile = KV_BLOCK=128 logical KV tokens``.

* AIC runs the Cube pipeline: one-tile QK lookahead (gather K(t+1) and issue
  QK(t+1) while the vector core consumes S(t)), paged K/V gather, QK over the
  576-wide cache (6 x 96 MMAD chunks, L0A/L0B ping-pong) and PV over the 512
  absorbed value features (four 128-wide halves, L0B ping-pong).
* AIV runs the online softmax over the valid columns, writes P back to L1 as
  fp16/bf16, rescales the running O and produces LSE.

The KV cache is either ``PA_BBND`` ``[pages, 128, 1, 576]`` or ``PA_NZ``
``[pages, 1, 36, 128, 16]``; ``mask_mode`` 0 is dense and 3 is right-down causal
over ``cache_seqlens[b]``.

``flash_mla_kvsplit_reduce_kernel`` is the combine pass of the KV-split variant:
it merges the f32 partial outputs and partial LSEs of ``kvsplit`` KV chunks into
the final fp16/bf16 output and the global LSE.
"""

from __future__ import annotations

import catlass.tla as tla


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


# =========================================================================
# HOST SIDE (single-source solution entry: run / capture_graph / compile)
# =========================================================================

# -----------------------------------------------------------------------------------------------------------
# flash_mla_with_kvcache · HOST side entry
#
# Validates the public contract, prepares the per-query request table, binds the
# CATLASS DSL executor once per (dtype, device, cache layout), and launches the
# single device kernel.
# -----------------------------------------------------------------------------------------------------------
