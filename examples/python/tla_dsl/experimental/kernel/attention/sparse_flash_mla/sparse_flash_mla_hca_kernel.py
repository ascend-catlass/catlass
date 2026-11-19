# ---------------------------------------------------------------------------
# sparse_flash_mla_hca_kernel.py — SparseFlashMlaHcaKernel（arch35, TND, HCA）
#
# 结构参照 ops-transformer/attention/sparse_flash_mla/op_kernel/arch35/
#   sparse_flash_mla_swa_kernel_arch35.h -> 本文件（入口/任务循环）——OT 分发表
#   只实例化 CsaKernel/SwaKernel 两类，HCA 是 SwaKernel 的 TEMPLATE_MODE=HCA
#   实例（PRELOAD_NUM=2，3 stage、尾排空 2），所以非 PA 实例的施工蓝本是本仓
#   src/sparse_flash_mla_swa_kernel.py 的闸门③三 stage 骨架（T2a）；
#   sparse_flash_mla_csa_block_cube_arch35.h   -> smla_csa_block_cube.py
#   sparse_flash_mla_csa_block_vector_arch35.h -> smla_csa_block_vector.py
# PA 实例（#019，kv_layout=PA_BBND）按 spec §3 允许偏离维持 csa 串行路径
#   （constexpr 折叠，字节表/指令流与改前一致），对齐前提见下方 load_k_pa 注。
#
# HCA = 两段 **连续** KV 源，都不带稀疏索引、都不走 AIV gather：
#   ori：`ComputeParamBatch` 的三模式窗口（#019 为 mask_mode=3）
#        start ← clip(row - pre, 0, kv)、end ← clip(row + next + 1, 0, kv)
#   cmp：`ComputeS2LoopInfo` 的 HCA 分支（kvcache.h:396-402）
#        end ← clip((row + 1 + nextTokensPerBatchCmp) / cmpRatio, 0, cmpSize)
#        nextTokensPerBatchCmp ← cmpSize*cmpRatio + residual - q_len（cmpMaskMode != 0）
#        nextTokensPerBatchCmp ← cmpSize*cmpRatio（cmpMaskMode == 0）
# 两段都按 s2BaseSize=128 切 tile，共用同一套 running max/sum/O 重标定与 3 槽 L1 K 环。
#
# PA_BBND 时两段各自有自己的 block table / blockSize / stride（CalcS2Coord 会切换
# blockTableGm、curKvGm、kvCacheBlockSize），cube 侧走 `load_k_pa`（GmCopyInToL1PA 语义）。
# `load_k_pa` 已对齐 OT GmCopyInToL1PA（CopyInL1.h:322-378）：收窗口起点、
# 逐页按 `remainRowCnt` 截断、页大小无上限，故 band 模式（mask4，start>0）与任意页大小
# 均支持。OT 侧同一个 SmlaCsaBlockCube 由 SWA/HCA 共用
# （sparse_flash_mla.cpp:44-76；csa_block_cube_arch35.h:298/:303），HCA 无独立缺口。
# 实测（本仓 #12/13/14/15/42/45/50/89 等 band+PA 例全 1.0）见 ITERATIONS.md §27.41。
# ---------------------------------------------------------------------------

from __future__ import annotations

import catlass.tla as tla

from smla_csa_block_cube import (
    iterate_bmm1,
    iterate_bmm2_fixpipe,
    iterate_bmm2_seg,
    load_k,
    load_k_pa,
    load_q,
    release_k,
    wait_p_ready,
)
from smla_csa_block_vector import (
    bmm2_data_copy_out,
    lse_copy_out,
    clean_output,
    process_vec1,
    process_vec2_seg,
    vec_init_from_sinks,
)
from smla_csa_common import (
    compute_need_init,
    KV_BLOCK,
    KV_SLOTS,
    alloc_buffers,
    declare_sync,
    init_sync,
    init_sync_preload,
    static_range,
    sync_all_cores_aic,
    sync_all_cores_aiv,
)

# T2a：GS1 预取深度（OT swa_kernel_arch35.h:77 PRELOAD_NUM = 2，HCA 同实例同
# 值）。S/P/dm 双缓冲槽数 = 全局 gtile%2；drain 步数同值。
PRELOAD_NUM = 2


@tla.kernel
def sparse_flash_mla_hca_kernel(
    q: tla.Tensor,  # [T*H, D]  TND 展平
    ori_kv: tla.Tensor,  # [KV, D]；PA_BBND 时展平为 (blockNum*blockSize*n2, D)
    cmp_kv: tla.Tensor,  # [CMP, D]；同上
    ori_bt: tla.Tensor,  # [B, maxBlocks] int32 展平（非 PA 时 1 元素占位）
    cmp_bt: tla.Tensor,  # [B, maxBlocks] int32 展平（非 PA 时 1 元素占位）
    cu_q: tla.Tensor,  # [B+1] int32
    cu_kv: tla.Tensor,  # [B+1] int32
    cu_cmp: tla.Tensor,  # [B+1] int32
    resid: tla.Tensor,  # [B] int32
    sinks: tla.Tensor,  # [H] fp32
    out: tla.Tensor,  # [T*H, D]
    lse: tla.Tensor,  # [T*H] fp32 softmax_lse（enable_lse=0 时传 1 元素占位）
    tiling: tla.Tensor,  # [4] int32 预留
    softmax_scale: tla.Float32,
    head_num: tla.Constexpr[int],
    head_dim: tla.Constexpr[int],
    num_seqs: tla.Constexpr[int],
    cmp_ratio: tla.Constexpr[int],
    cmp_mask_mode: tla.Constexpr[int],
    ori_mask_mode: tla.Constexpr[int],
    ori_win_left: tla.Constexpr[int],
    ori_win_right: tla.Constexpr[int],
    ori_pa_block_size: tla.Constexpr[int],
    ori_pa_stride: tla.Constexpr[int],
    ori_pa_max_blocks: tla.Constexpr[int],
    ori_pa_pages: tla.Constexpr[int],
    cmp_pa_block_size: tla.Constexpr[int],
    cmp_pa_stride: tla.Constexpr[int],
    cmp_pa_max_blocks: tla.Constexpr[int],
    cmp_pa_pages: tla.Constexpr[int],
    g_split: tla.Constexpr[int],
    has_sinks: tla.Constexpr[int],
    enable_lse: tla.Constexpr[int],  # R37：softmax_lse 落盘开关
) -> None:
    c1 = 1
    c2 = 2
    dtype_in = q.ptr.dtype
    # IS_SPLIT_G（tiling splitG = gSize > 64）：任务空间 = token × G 半区，
    # m 为半区 G 宽度，ori 行数/cmp 端点仍按 token 行 tok 计算。
    #
    # 奇数 G（#013 h=65）按 OT 口径：
    #   halfG = (gSize+1)/2 = 33，goIdx = (aicIdx%2==0)?0:halfG
    #   （common/smla_kernel_common_arch35.h:271-278）；
    #   mRealSize = s1RealSize*gSplitSize（kvcache.h:200-203），两个半区分别为 33/32；
    #   fixpipe mSize = Align2Func(mRealSize)（csa_block_cube_arch35.h:440/493），
    #   所以 S/Otmp 的 view 行数按 34 声明（halfG 为奇数时 +1 对齐）。
    # AIV 仍按 17/17 行处理，丢弃行由 bmm2_data_copy_out 逐行判定 valid_m 后不写 GM。
    odd_g = (g_split == 2) and (head_num % g_split != 0)
    half_g = (head_num + 1) // 2 if g_split == 2 else head_num
    # Align2(halfG)：33 -> 34；偶数 halfG 时就是 halfG 本身。
    # 用 Python 条件表达式（trace 期取值），避免被当成 dynamic if 分支。
    m = (half_g + (half_g % 2)) if odd_g else (head_num // g_split)
    # AIV 两个子块行数：firstHalf=(m+1)/2、second=m-firstHalf（OT 同名量）；
    # rows 取前半（m 恒为偶数时两半相等）。
    rows = m - m // 2
    row_cap = m if odd_g else None
    # 供 helper 内部按 constexpr-int 折叠的开关（kernel 层不再写 `if odd_g:`）：
    #   odd_mode=1  -> bmm2_data_copy_out 走逐行 valid 判定
    #   sink_cap>0  -> vec_init_from_sinks 整条 sinks 一次搬入 UB 后逐行取
    #                  （host 侧把 sinks 补齐到 8 元素/32B 倍数）
    odd_mode = 1 if odd_g else 0
    sink_cap = ((head_num + 7) // 8) * 8 if odd_g else 0

    sync = declare_sync(need_lse_war=enable_lse == 1)
    # T2a：非 PA 实例走流水，需要 F6/F11 双槽套件（l1_p/l0c_s/ub_s/ub_max/
    # ub_sum/ub_dm 连续双槽，OT l1PBuffers[2]/bmm1Buffers[2]/softmaxExpBufs
    # [taskIdMod2] 同款）；PA 串行实例保持单槽字节表与改前逐字节一致。
    use_pa = 1 if (ori_pa_block_size > 0 or cmp_pa_block_size > 0) else 0
    bufs = alloc_buffers(
        dtype_in, row_cap, (sink_cap if sink_cap > 0 else None), preload=(use_pa == 0)
    )
    init_sync(sync, pre_l0c2=(use_pa == 0))
    if use_pa == 0:
        # F6：流水信用预置（s_free 1->2；F14 后 p_free 回执已不存在）。
        init_sync_preload(sync)
    # F4（OT swa_kernel_arch35.h:154 CleanOutput，HCA 与 SWA 同构）
    need_init = compute_need_init(cu_q, cu_kv, num_seqs, ori_mask_mode, 0)
    clean_output(
        sync,
        bufs,
        out,
        lse,
        need_init,
        cu_q[num_seqs] * head_num * head_dim,
        0,
        enable_lse,
        dtype_in,
        head_dim=head_dim,
    )

    # F27 补（#046 间歇性非逐位）：OT swa_kernel_arch35.h:344-346 的 Process()
    # 入口屏障（HCA 与 SWA 同构；「SyncAll Cube和Vector都需要调用」）。
    # 分布式 init 清零必须在任何核开始写 O 之前全部完成，否则迟到的 init 会把
    # 其它核已写好的行清 0——正控已逐元素复现、且加屏障后正控不再生效
    # （ITERATIONS.md §4.12）。条件与 OT 相同：needInit=1 时同步。
    if need_init == 1:
        with tla.cube():
            sync_all_cores_aic()
        with tla.vector():
            sync_all_cores_aiv()

    total_tasks = cu_q[num_seqs] * g_split
    task0 = tla.arch.block_idx()
    q_bn = tla.arch.block_num()
    for task in tla.range(task0, total_tasks, q_bn):
        # ---- 定位 (seq, row)；行号按 token（tok = task // g_split）----
        tok = task // g_split
        seq = tla.as_numeric(0)
        rowq = tok
        for bb in static_range(1, num_seqs):
            if tok >= cu_q[bb]:
                seq = bb
                rowq = tok - cu_q[bb]
        q_len = cu_q[seq + 1] - cu_q[seq]
        kv_len = cu_kv[seq + 1] - cu_kv[seq]
        cmp_len = cu_cmp[seq + 1] - cu_cmp[seq]

        # ---- ori：ComputeParamBatch 三模式（kvcache.h:96-114）----
        causal_offset = kv_len - q_len
        next_tok = kv_len  # mode 0（full）默认值
        pre_tok = q_len
        if ori_mask_mode == 3:
            next_tok = causal_offset
            pre_tok = q_len
        elif ori_mask_mode == 4:
            if ori_win_left == -1:
                pre_tok = q_len
            else:
                pre_tok = ori_win_left - causal_offset
            if ori_win_right == -1:
                next_tok = kv_len
            else:
                next_tok = causal_offset + ori_win_right
            if pre_tok > q_len:
                pre_tok = q_len
        ori_start = rowq - pre_tok
        if ori_start < 0:
            ori_start = tla.as_numeric(0)
        if ori_start > kv_len:
            ori_start = kv_len
        ori_end = rowq + next_tok + 1
        if ori_end < 0:
            ori_end = tla.as_numeric(0)
        if ori_end > kv_len:
            ori_end = kv_len
        ori_lines = ori_end - ori_start
        if ori_lines < 0:
            ori_lines = tla.as_numeric(0)
        n_ori_tiles = (ori_lines + KV_BLOCK - 1) // KV_BLOCK

        # ---- cmp：ComputeS2LoopInfo 的 HCA 分支（kvcache.h:396-402）----
        next_cmp = cmp_len * cmp_ratio  # cmpMaskMode == 0 的默认值
        if cmp_mask_mode != 0:
            next_cmp = cmp_len * cmp_ratio + resid[seq] - q_len
        cmp_end = (rowq + 1 + next_cmp) // cmp_ratio
        if cmp_end < 0:
            cmp_end = tla.as_numeric(0)
        if cmp_end > cmp_len:
            cmp_end = cmp_len
        n_cmp_tiles = (cmp_end + KV_BLOCK - 1) // KV_BLOCK

        n_tiles = n_ori_tiles + n_cmp_tiles
        kv_row0 = cu_kv[seq] + ori_start
        cmp_row0 = cu_cmp[seq]

        # IS_SPLIT_G：第 (task % g_split) 个 G 半区的首行 = goIdx
        #   偶数 G：两半等宽，g_off = half*m，q/out 行基址 = task*m（原行为）；
        #   奇数 G：g_off = half*halfG，行基址 = tok*gSize + g_off（OT
        #           csa_block_cube_arch35.h:267-275、kvcache.h:236-250）。
        #   两式可合并：task*m = (tok*g_split+half)*m = tok*gSize + half*m，
        #   偶数 G 时 m==halfG、奇数 G 时 halfG 就是半区宽度，故
        #   q_row_base = tok*gSize + half*halfG 对所有情形等价。
        # valid_m 同理：第一半 halfG、第二半 gSize-halfG（偶数 G 时两式都等于 m）。
        half = task - tok * g_split
        g_off = half * half_g
        valid_m = half_g
        if half != 0:
            valid_m = head_num - half_g
        q_row_base = tok * head_num + g_off

        if use_pa:
            # ---- PA 实例：csa 串行路径原样保留（spec §3 允许偏离；指令流与
            # 改前逐条一致，单槽 vec1/vec2、局部 K 键控在逐 tile 全排空下
            # 自洽）----
            load_q(sync, bufs, q, task, m, dtype_in, row_base=q_row_base)
            vec_init_from_sinks(
                sync, bufs, sinks, task, m, rows, dtype_in, g_off, has_sinks, sink_cap
            )

            for t in tla.range(n_tiles):
                slot = t - (t // KV_SLOTS) * KV_SLOTS
                n_valid = tla.as_numeric(0)
                n_load = tla.as_numeric(0)
                if t < n_ori_tiles:
                    tile_start = t * KV_BLOCK
                    n_valid = ori_lines - tile_start
                    if n_valid > KV_BLOCK:
                        n_valid = tla.as_numeric(KV_BLOCK)
                    n_load = kv_len - ori_start - tile_start
                    if n_load > KV_BLOCK:
                        n_load = tla.as_numeric(KV_BLOCK)
                else:
                    ct = t - n_ori_tiles
                    n_valid = cmp_end - ct * KV_BLOCK
                    if n_valid > KV_BLOCK:
                        n_valid = tla.as_numeric(KV_BLOCK)
                    n_load = cmp_len - ct * KV_BLOCK
                    if n_load > KV_BLOCK:
                        n_load = tla.as_numeric(KV_BLOCK)
                if t < n_ori_tiles:
                    if ori_pa_block_size > 0:
                        load_k_pa(
                            sync,
                            bufs,
                            ori_kv,
                            ori_bt,
                            seq,
                            ori_start + t * KV_BLOCK,
                            n_load,
                            slot,
                            dtype_in,
                            ori_pa_block_size,
                            ori_pa_stride,
                            ori_pa_max_blocks,
                            ori_pa_pages,
                        )
                    else:
                        load_k(sync, bufs, ori_kv, kv_row0, t, n_load, slot, dtype_in)
                else:
                    ct2 = t - n_ori_tiles
                    if cmp_pa_block_size > 0:
                        load_k_pa(
                            sync,
                            bufs,
                            cmp_kv,
                            cmp_bt,
                            seq,
                            ct2 * KV_BLOCK,
                            n_load,
                            slot,
                            dtype_in,
                            cmp_pa_block_size,
                            cmp_pa_stride,
                            cmp_pa_max_blocks,
                            cmp_pa_pages,
                        )
                    else:
                        load_k(
                            sync, bufs, cmp_kv, cmp_row0, ct2, n_load, slot, dtype_in
                        )
                iterate_bmm1(sync, bufs, slot, m, t + 1 == n_tiles, t == 0, dtype_in)
                process_vec1(sync, bufs, softmax_scale, n_valid, m, rows, dtype_in)
                wait_p_ready(sync)
                for seg in (0, 1, 2, 3):
                    iterate_bmm2_seg(sync, bufs, seg, slot, m, dtype_in)
                # F17 步骤 B：4 段 mmad -> 每 tile 一次整块 fixpipe -> 4 段 vec2
                # （OT IterateBmm2 同构：fixpipe 频次 5 -> 2/tile）。
                iterate_bmm2_fixpipe(sync, bufs, m)
                for seg in (0, 1, 2, 3):
                    process_vec2_seg(sync, bufs, seg, m, rows, dtype_in)
                release_k(sync, slot)

            bmm2_data_copy_out(
                sync,
                bufs,
                out,
                task,
                m,
                rows,
                dtype_in,
                row_base=q_row_base,
                valid_rows=valid_m,
                odd_mode=odd_mode,
            )
            if enable_lse == 1:
                # R37：串行 PA 路径只用一个槽（slot 0），行基址/有效行同 out
                lse_copy_out(
                    sync,
                    bufs,
                    lse,
                    task,
                    m,
                    rows,
                    slot=0,
                    row_base=q_row_base,
                    valid_rows=(valid_m if odd_mode == 1 else None),
                )
        else:
            # ---- T2a 闸门③ 任务内三 stage 流水（蓝本 swa_kernel 的 F11 两级
            # stage；跨任务连续化留待二轮按 #015/#018 vs #016 的分管道数据
            # 裁决）----
            # body i（i = 0..n_tiles+drain-1）：
            #   stage0 = tile i   （AIC MTE2 装载，ori/cmp 双源按 tile 定）
            #   stage1 = tile i-1 （AIV vec1 + AIC bmm1；任务首 tile 先 vec_init）
            #   stage2 = tile i-2 （AIC bmm2×4+fixpipe + AIV vec2×4 + release；
            #                      任务末 tile 就地 copy_out，闸门④ 折叠）
            # 每任务 drain=2 尾排空（任务边界退回串行交接，head/tail 损耗
            # ≈2/task_len：015/018 为 64 tile/任务 ≈3%，016 为 8 tile/任务
            # ≈25%——预注册预期 016 本版吃不满，见 F21e §2）。
            # 槽键控全部全局 gtile（K %3、S/P/dm %2）：g_k = 本核在 task 之前
            # 的真实 tile 总数，由 O(K) 前缀游走重算（swa 同款，AIC/AIV 冗余
            # 执行、数值一致；不用外层 loop-carry——无上板先例，F7 R7 推演见
            # ITERATIONS.md）。
            g_k = tla.as_numeric(0)
            for ct in tla.range(task0, total_tasks, q_bn):
                ct_tok = ct // g_split
                ct_seq = tla.as_numeric(0)
                ct_rowq = ct_tok
                # 目标名避开宿主任务体的 bb（swa 同款：头部 static 循环同名
                # 目标会把 bb 误判成本环 carry）。
                for cbb in tla.range(1, num_seqs):
                    if ct_tok >= cu_q[cbb]:
                        ct_seq = cbb
                        ct_rowq = ct_tok - cu_q[cbb]
                ct_qlen = cu_q[ct_seq + 1] - cu_q[ct_seq]
                ct_kvlen = cu_kv[ct_seq + 1] - cu_kv[ct_seq]
                ct_cmplen = cu_cmp[ct_seq + 1] - cu_cmp[ct_seq]
                ct_coff = ct_kvlen - ct_qlen
                ct_next = ct_kvlen
                ct_pre = ct_qlen
                if ori_mask_mode == 3:
                    ct_next = ct_coff
                    ct_pre = ct_qlen
                elif ori_mask_mode == 4:
                    if ori_win_left == -1:
                        ct_pre = ct_qlen
                    else:
                        ct_pre = ori_win_left - ct_coff
                    if ori_win_right == -1:
                        ct_next = ct_kvlen
                    else:
                        ct_next = ct_coff + ori_win_right
                    if ct_pre > ct_qlen:
                        ct_pre = ct_qlen
                ct_end = ct_rowq + ct_next + 1
                if ct_end < 0:
                    ct_end = tla.as_numeric(0)
                if ct_end > ct_kvlen:
                    ct_end = ct_kvlen
                ct_start = ct_rowq - ct_pre
                if ct_start < 0:
                    ct_start = tla.as_numeric(0)
                if ct_start > ct_kvlen:
                    ct_start = ct_kvlen
                ct_ori_lines = ct_end - ct_start
                if ct_ori_lines < 0:
                    ct_ori_lines = tla.as_numeric(0)
                ct_n_ori = (ct_ori_lines + KV_BLOCK - 1) // KV_BLOCK
                ct_ncmp_next = ct_cmplen * cmp_ratio
                if cmp_mask_mode != 0:
                    ct_ncmp_next = ct_cmplen * cmp_ratio + resid[ct_seq] - ct_qlen
                ct_cmp_end = (ct_rowq + 1 + ct_ncmp_next) // cmp_ratio
                if ct_cmp_end < 0:
                    ct_cmp_end = tla.as_numeric(0)
                if ct_cmp_end > ct_cmplen:
                    ct_cmp_end = ct_cmplen
                ct_n_cmp = (ct_cmp_end + KV_BLOCK - 1) // KV_BLOCK
                if ct < task:
                    g_k = g_k + ct_n_ori + ct_n_cmp

            # 闸门② Q 槽序（swa 同款：本核第 n 个真实任务占 {2n,2n+1} mod 3
            # 两半槽；空任务不装不释、槽位闲置）。任务内流水版 Q 在任务头装
            # （无跨任务预取，深度 1 握手 wait x1 / 末 tile bmm1 set x1）。
            n_task = (task - task0) // q_bn
            q_two_n = n_task + n_task
            q_slot_a = q_two_n - (q_two_n // 3) * 3
            q_two_n1 = q_two_n + 1
            q_slot_b = q_two_n1 - (q_two_n1 // 3) * 3
            if n_tiles > 0:
                load_q(
                    sync,
                    bufs,
                    q,
                    task,
                    m,
                    dtype_in,
                    row_base=q_row_base,
                    slot_a=q_slot_a,
                    slot_b=q_slot_b,
                )

            drain = 0
            if n_tiles > 0:
                drain = PRELOAD_NUM
            for i in tla.range(n_tiles + drain):
                x = g_k + i
                # ---- stage 0（AIC）：tile i 装载 ----
                if i < n_tiles:
                    slot_k0 = x - (x // KV_SLOTS) * KV_SLOTS
                    n_load0 = tla.as_numeric(0)
                    if i < n_ori_tiles:
                        n_load0 = kv_len - ori_start - i * KV_BLOCK
                        if n_load0 < 0:
                            n_load0 = tla.as_numeric(0)
                        if n_load0 > KV_BLOCK:
                            n_load0 = tla.as_numeric(KV_BLOCK)
                        load_k(
                            sync, bufs, ori_kv, kv_row0, i, n_load0, slot_k0, dtype_in
                        )
                    else:
                        n_load0 = cmp_len - (i - n_ori_tiles) * KV_BLOCK
                        if n_load0 < 0:
                            n_load0 = tla.as_numeric(0)
                        if n_load0 > KV_BLOCK:
                            n_load0 = tla.as_numeric(KV_BLOCK)
                        load_k(
                            sync,
                            bufs,
                            cmp_kv,
                            cmp_row0,
                            i - n_ori_tiles,
                            n_load0,
                            slot_k0,
                            dtype_in,
                        )
                # ---- stage 1：tile i-1 的 vec1（AIV）与 bmm1（AIC）----
                if i >= c1:
                    if i != n_tiles + c1:
                        j1 = i - c1
                        n_valid1 = tla.as_numeric(0)
                        if j1 < n_ori_tiles:
                            n_valid1 = ori_lines - j1 * KV_BLOCK
                            if n_valid1 > KV_BLOCK:
                                n_valid1 = tla.as_numeric(KV_BLOCK)
                        else:
                            n_valid1 = cmp_end - (j1 - n_ori_tiles) * KV_BLOCK
                            if n_valid1 > KV_BLOCK:
                                n_valid1 = tla.as_numeric(KV_BLOCK)
                        xg1 = x - c1
                        slot_k1 = xg1 - (xg1 // KV_SLOTS) * KV_SLOTS
                        slot_s1 = xg1 - (xg1 // PRELOAD_NUM) * PRELOAD_NUM
                        # F11 sum/max 链式读槽：非首 tile 读 (xg1-1)%2，任务
                        # 首 tile 读 vec_init 重灌的初值槽（xg1%2 = slot_s1）。
                        slot_sum_rd = slot_s1
                        if j1 != 0:
                            slot_sum_rd = slot_s1 + 1
                            if slot_sum_rd > 1:
                                slot_sum_rd = slot_sum_rd - 2
                        if j1 == 0:
                            # F11 配套②：只灌 max/sum 不清 O（O 启动由 vec2
                            # fresh 覆盖接管，数值恒等 0*dm=0）。
                            vec_init_from_sinks(
                                sync,
                                bufs,
                                sinks,
                                task,
                                m,
                                rows,
                                dtype_in,
                                g_off,
                                has_sinks,
                                sink_cap,
                                slot_sum=slot_s1,
                                slot_max=slot_s1,
                                clear_o=0,
                            )
                        process_vec1(
                            sync,
                            bufs,
                            softmax_scale,
                            n_valid1,
                            m,
                            rows,
                            dtype_in,
                            slot_s=slot_s1,
                            slot_p=slot_s1,
                            preload=1,
                            slot_sum_rd=slot_sum_rd,
                            slot_max_rd=slot_sum_rd,
                        )
                        is_first1 = j1 == 0
                        is_last1 = j1 + c1 == n_tiles
                        iterate_bmm1(
                            sync,
                            bufs,
                            slot_k1,
                            m,
                            is_last1,
                            is_first1,
                            dtype_in,
                            slot_s=slot_s1,
                            q_slot_a=q_slot_a,
                            q_slot_b=q_slot_b,
                        )
                # ---- stage 2：tile i-2 的 bmm2+fixpipe+vec2（+release/copy_out）----
                if i >= c2:
                    j2 = i - c2
                    xg2 = x - c2
                    slot_k2 = xg2 - (xg2 // KV_SLOTS) * KV_SLOTS
                    slot_p2 = xg2 - (xg2 // PRELOAD_NUM) * PRELOAD_NUM
                    wait_p_ready(sync)
                    for seg in (0, 1, 2, 3):
                        iterate_bmm2_seg(
                            sync, bufs, seg, slot_k2, m, dtype_in, slot_p=slot_p2
                        )
                    # F17 步骤 B：4 段 mmad -> 每 tile 一次整块 fixpipe -> 4 段
                    # vec2（OT IterateBmm2 同构：fixpipe 频次 5 -> 2/tile）。
                    iterate_bmm2_fixpipe(sync, bufs, m)
                    # F11 配套②：任务首 tile 的 O 启动走 fresh（O = PV）。
                    # vec.func region 内 fresh 直接作 Bool 谓词（i32 0/1 phi
                    # 会报 Expected scalar Bool predicate——F18 期已知形态）。
                    fresh_v2 = j2 == 0
                    for seg in (0, 1, 2, 3):
                        process_vec2_seg(
                            sync,
                            bufs,
                            seg,
                            m,
                            rows,
                            dtype_in,
                            fresh_v2,
                            slot_dm=slot_p2,
                        )
                    release_k(sync, slot_k2)
                    # 闸门④ 折叠：owner 末 tile 的 vec2 链后就地 copy_out
                    # （swa 同位；HCA 多带任务级参数 row_base/valid_rows/
                    # odd_mode 与 sum 读槽 slot_p2）。
                    if j2 + c1 == n_tiles:
                        bmm2_data_copy_out(
                            sync,
                            bufs,
                            out,
                            task,
                            m,
                            rows,
                            dtype_in,
                            row_base=q_row_base,
                            valid_rows=valid_m,
                            odd_mode=odd_mode,
                            slot_sum=slot_p2,
                        )
                        if enable_lse == 1:
                            # R37：终态 max/sum 在 slot_p2，行基址/有效行同 out
                            lse_copy_out(
                                sync,
                                bufs,
                                lse,
                                task,
                                m,
                                rows,
                                slot=slot_p2,
                                row_base=q_row_base,
                                valid_rows=(valid_m if odd_mode == 1 else None),
                            )
            # 空任务（n_tiles==0）语义按 OT：swa_kernel_arch35.h 的
            # `if (smlaS1NoNeedCalc || smaS2NoNeedCalc) continue;` 整条跳过，
            # 不产生 vec2/copy_out；空行输出由 CleanOutput（needInit）补零，
            # 两者配套（compute_need_init 的 OT 原式 s1Size > s2Size）。故流水
            # 版 n_tiles==0 不 copy_out 是忠实于 OT 的形态；串行 PA 分支保留的无
            # 条件 copy_out 是本仓自有偏离，记账为 spec §3 允许偏离的一部分
            # （hca 10 例每 token tile ≥6，不可达）。
