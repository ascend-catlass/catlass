# ---------------------------------------------------------------------------
# sparse_flash_mla_swa_kernel.py — SparseFlashMlaSwaKernel（arch35, TND 稠密）
#
# 结构参照 ops-transformer/attention/sparse_flash_mla/op_kernel/arch35/
#   sparse_flash_mla_swa_kernel_arch35.h  -> 本文件（入口/任务循环）
#   sparse_flash_mla_csa_block_cube_arch35.h   -> smla_csa_block_cube.py
#   sparse_flash_mla_csa_block_vector_arch35.h -> smla_csa_block_vector.py
#
# 固定 tile：s1BaseSize=64, s2BaseSize=128, dSize=512, qHalfNum=2。
# 任务：一个 (query token, head 组)，M = head_num（metadata 的 groupSize）。
# 可见列（ori_mask_mode=3，causal）：[0, kv_len - q_len + row + 1)。
# PA_BBND：入口收 ori_block_table，K 走 load_k_pa（与 OT SwaKernel 共用
# SmlaCsaBlockCube 的 GmCopyInToL1PA，见 sparse_flash_mla.cpp:44-76）。
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
    bc_div_cur,
    bc_merge_prev,
    bc_reinit_stats,
    bc_stage_cur,
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

# F6：GS1 预取深度（OT swa_kernel_arch35.h:77 PRELOAD_NUM = 2）。S/ P 双缓冲
# 槽数 = tile%2（OT bmm1Buffers[2] :79 / l1PBuffers[2] :80）；循环排空步数同值。
PRELOAD_NUM = 2


@tla.kernel
def sparse_flash_mla_swa_kernel(
    q: tla.Tensor,  # [T, H, D]  TND，与 Q_T 同 dtype
    kv: tla.Tensor,  # [KV, 1, D] TND；PA_BBND 时展平为 (blockNum*blockSize*n2, D)
    ori_bt: tla.Tensor,  # [B, maxPages] int32 展平（非 PA 时 1 元素占位）
    sinks: tla.Tensor,  # [H] fp32
    cu_q: tla.Tensor,  # [B+1] int32
    cu_kv: tla.Tensor,  # [B+1] int32
    out: tla.Tensor,  # [T, H, D] TND
    lse: tla.Tensor,  # [T*H] fp32 softmax_lse（enable_lse=0 时传 1 元素占位）
    tiling: tla.Tensor,  # [4] int32 预留（compiled-in 常量走 Constexpr）
    bc_ws: tla.Tensor,  # F3 staging（每 AIV 一槽 fp32；=0 传 1 元素占位）
    softmax_scale: tla.Float32,
    head_num: tla.Constexpr[int],
    head_dim: tla.Constexpr[int],
    num_seqs: tla.Constexpr[int],
    ori_mask_mode: tla.Constexpr[int],
    ori_win_left: tla.Constexpr[int],
    ori_win_right: tla.Constexpr[int],
    ori_pa_block_size: tla.Constexpr[int],
    ori_pa_stride: tla.Constexpr[int],
    ori_pa_max_blocks: tla.Constexpr[int],
    ori_pa_pages: tla.Constexpr[int],
    g_split: tla.Constexpr[int],
    has_sinks: tla.Constexpr[int],
    enable_lse: tla.Constexpr[int],  # R37：softmax_lse 落盘开关
    is_batch_consistency: tla.Constexpr[int],  # F3：OT tiling key 第 7 位
) -> None:
    c0 = 0
    c1 = 1
    c2 = 2
    dtype_in = q.ptr.dtype
    # IS_SPLIT_G：任务空间 = token × G 半区
    m = head_num // g_split
    # AIV 两个子块行数：firstHalf=(m+1)/2、second=m-firstHalf（OT 同名量）；
    # rows 取前半（偶数 m 即 m/2，奇数 m 时第二子块写 0 行，见 bmm2_data_copy_out）。
    rows = m - m // 2
    # R37：OT goIdx = half*halfG（奇 head_num 时 halfG ≠ m），lse 行基址用
    half_g = head_num // g_split if head_num % g_split == 0 else head_num // g_split + 1

    sync = declare_sync(need_lse_war=enable_lse == 1)
    # E-F3g（真根因）：bc 暂存容量必须随 is_batch_consistency 传入——此前恒用
    # 默认 bc=0，bc=1 实例的 ub_bc_prev/max/sum/w 只有 1 元素占位，merge 的
    # MTE2 拷入按 rows8/rows×128 视图越界写、V 读回残留（clamp=1 无 merge 链
    # 所以正确、staging 槽用的是真分配的 ub_max/sum 所以正确，全观测自洽）。
    # =0 实例传 0，字节表与改前逐字节一致。
    bufs = alloc_buffers(dtype_in, bc=is_batch_consistency, preload=True)
    # F3：is_batch_consistency=1 实例预置 1 个 bc_v2m 信用（merge 首段消费）；
    # =0 实例不预置。F6：init_sync_preload 追加流水信用（s_free 1->2；F14 步骤
    # A 后 p_free 回执已去除，见 common 注；拆独立 jit 是因为 flag 对象不能跨
    # 前端 if 边界，见 common 注）。
    init_sync(sync, pre_bc=(is_batch_consistency == 1), pre_l0c2=True)
    init_sync_preload(sync)
    # F4（OT swa_kernel_arch35.h:154 CleanOutput）
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
    # 入口屏障（「SyncAll Cube和Vector都需要调用」）。分布式 init 清零必须在任何
    # 核开始写 O 之前全部完成，否则迟到的 init 会把其它核已写好的行清 0——
    # 正控已逐元素复现、且加屏障后正控不再生效（ITERATIONS.md §4.12）。
    # 条件与 OT 相同：needInit=1 时同步。
    if need_init == 1:
        with tla.cube():
            sync_all_cores_aic()
        with tla.vector():
            sync_all_cores_aiv()

    total_tasks = cu_q[num_seqs] * g_split
    # 闸门③ 全局 gtile 连续流水：body x 处理 tile x（stage0）/ x-1（stage1）/
    # x-2（stage2），tile 序跨任务连续；每任务不再垫 PRELOAD_NUM 个 body，
    # 仅核上最后一个有 tile 的任务多跑 2 个 body 作 kernel 尾排空。g_k = 本核
    # 在 task 之前的真实 tile 总数（gtile 前缀），g_total = 本核真实 tile 总数。
    # stage1/stage2 的跨任务站点（i=0/1）按 owner 重算标量（P=前一真实任务、
    # PP=前前；虚拟标量环，无存储，推演见 ITERATIONS.md F7 R7）。
    task0 = tla.arch.block_idx()
    q_bn = tla.arch.block_num()
    task_range = tla.range(task0, total_tasks, q_bn)
    for task in task_range:
        # ---- 定位 (seq, row)；TND 的 cu_seqlens_q 累加表 ----
        tok = task // g_split
        seq = tla.as_numeric(0)
        rowq = tok
        for bb in static_range(1, num_seqs):
            if tok >= cu_q[bb]:
                seq = bb
                rowq = tok - cu_q[bb]
        q_len = cu_q[seq + 1] - cu_q[seq]
        kv_len = cu_kv[seq + 1] - cu_kv[seq]
        # 可见区间：kvcache.h:96-114 ComputeParamBatch 三模式（与 os/hca 同式）
        causal_offset = kv_len - q_len
        next_tok = kv_len  # mode 0（full）
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
        end_idx = rowq + next_tok + 1
        if end_idx < 0:
            end_idx = tla.as_numeric(0)
        if end_idx > kv_len:
            end_idx = kv_len
        start_idx = rowq - pre_tok
        if start_idx < 0:
            start_idx = tla.as_numeric(0)
        if start_idx > kv_len:
            start_idx = kv_len
        vis = end_idx - start_idx
        if vis < 0:
            vis = tla.as_numeric(0)
        num_tiles = (vis + KV_BLOCK - 1) // KV_BLOCK
        # F3-Iter23：OT SWA BC 的 baseBlockNumPerReductionBlock
        # （swa_kernel_arch35.h:428-437：raw=ceil(totalLoad/32)、
        #  rb=round_up(raw,s2BaseSize=128)、B=rb/128，rb==0 时 B=1；
        #  totalLoad=本任务 ori KV 可见行数，SWA 用例无 cmp 段）。
        # 001-004/007 vis≤4096 → B=1（每 tile 独立进链，改前语义）；
        # 005/006 vis>4096 → B=2（两 tile 一规约块：偶 tile 保 raw PV、
        # 奇 tile FlashUpdateLastNew 块内合并后才进 staged 链）。
        raw_rb = (vis + 31) // 32
        rb = ((raw_rb + 127) // 128) * 128
        base_block_num = rb // 128
        if rb == 0:
            base_block_num = tla.as_numeric(1)
        kv_row0 = cu_kv[seq] + start_idx
        g_off = (task - tok * g_split) * m

        # ---- 闸门③ 前缀游走（O(K)，AIC/AIV 冗余执行、数值一致；动态循环
        # 标量 loop-carry 已上板证实）----
        # g_k/g_total 累加 + 快照 P（ct<task 的第一个真实任务）、PP（第二个）、
        # NX（ct>task 的第一个真实任务）与 any_after（尾排空判据）。空任务
        # （num_tiles==0）不占 tile 也不占 body，P/PP/NX 自动跳过（OT :451-453
        # 同位语义）。
        g_k = tla.as_numeric(0)
        g_total = tla.as_numeric(0)
        any_after = 0
        nx_done = 0
        p_task = task0
        p_nt = 1
        p_vis = KV_BLOCK
        p_bbn = 1
        p_goff = 0
        pp_task = task0
        pp_nt = 1
        pp_vis = KV_BLOCK
        pp_bbn = 1
        pp_goff = 0
        nx_task = task0
        for ct in tla.range(task0, total_tasks, q_bn):
            ct_tok = ct // g_split
            ct_seq = tla.as_numeric(0)
            ct_rowq = ct_tok
            # 内层用动态 tla.range、目标名避开宿主任务体的 bb（头部 static 循环
            # 同名目标会把 bb 误判成本环 carry，读点在 num_seqs==1 时未绑定 →
            # UnboundLocalError；bb 在本体内是死变量，改名即消除）。cu_q[cbb]
            # 动态标量读与 cu_q[seq]（任务头）同型。
            for cbb in tla.range(1, num_seqs):
                if ct_tok >= cu_q[cbb]:
                    ct_seq = cbb
                    ct_rowq = ct_tok - cu_q[cbb]
            ct_qlen = cu_q[ct_seq + 1] - cu_q[ct_seq]
            ct_kvlen = cu_kv[ct_seq + 1] - cu_kv[ct_seq]
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
            ct_vis = ct_end - ct_start
            if ct_vis < 0:
                ct_vis = tla.as_numeric(0)
            ct_nt = (ct_vis + KV_BLOCK - 1) // KV_BLOCK
            ct_rrb = (ct_vis + 31) // 32
            ct_rb = ((ct_rrb + 127) // 128) * 128
            ct_bbn = ct_rb // 128
            if ct_rb == 0:
                ct_bbn = tla.as_numeric(1)
            ct_goff = (ct - ct_tok * g_split) * m
            g_total = g_total + ct_nt
            if ct < task:
                g_k = g_k + ct_nt
                if ct_nt > 0:
                    # P/PP 滚动记录（last-wins）：P=最近非空前驱、PP=次近非空
                    # （OT prevTask/prevPrevTask 同语义）。初版 first-wins 锁死
                    # 会把 P 钉在最早前驱（通常 task0），stage1/stage2 的 i<c
                    # 跨任务站点 owner 键随之坍缩到 task0——copy_out/dump 键全
                    # 错而数据路径正确（#008 clamp=2 dump 取证：站点数据与键
                    # 分家，cos=1.0000 闭环）。x>=c 守卫保证站点只在 g_k>=c 时
                    # 出现，此刻 P/PP 必已被滚动覆盖，初值不参与语义。
                    pp_task = p_task
                    pp_nt = p_nt
                    pp_vis = p_vis
                    pp_bbn = p_bbn
                    pp_goff = p_goff
                    p_task = ct
                    p_nt = ct_nt
                    p_vis = ct_vis
                    p_bbn = ct_bbn
                    p_goff = ct_goff
            if ct > task:
                if ct_nt > 0:
                    any_after = 1
                    if nx_done == 0:
                        nx_task = ct
                        nx_done = 1

        # 闸门②（OT l1QBufId 每任务 +qHalfNum=2 游标，csa_block_cube_arch35.h
        # :425）：本核第 n 个真实任务占 {2n, 2n+1} mod 3 两半槽。Q 环只在真实
        # 任务上推进（装 Q 的站点见头装与 stage1 末尾预取；空任务既不装也不
        # 释放，n 按任务序派生——空任务槽位闲置无害）。
        n_task = (task - task0) // q_bn
        q_two_n = n_task + n_task
        q_slot_a = q_two_n - (q_two_n // 3) * 3
        q_two_n1 = q_two_n + 1
        q_slot_b = q_two_n1 - (q_two_n1 // 3) * 3
        # 首个真实任务在任务头装 Q（OT IterateLoadQK isFirstLoop）；后续真实
        # 任务的 Q 由前一真实任务末 tile 的 stage1 预取（见 tile 循环 is_last1
        # 站点——跨任务后该站点落在下一真实任务的首 body）。信用为深度 1
        # 握手（init x1 / load wait x1 / release set x1——多信用 x2/x3 在循环
        # 语境下挂死，见账本 R6；深度 1 与深度 2 的闸点相同，预取距离不变）。
        if g_k == 0:
            if num_tiles > 0:
                load_q(
                    sync, bufs, q, task, m, dtype_in, slot_a=q_slot_a, slot_b=q_slot_b
                )

        # F6+闸门③④（OT swa_kernel_arch35.h:390-560 ProcessMainFlow，跨任务
        # 连续化）：body x = g_k+i 全局编号，tile x 的 stage0 在 body x、
        # stage1 在 body x+1、stage2 在 body x+2；本任务只跑 num_tiles 个 body
        # （+末真实任务 drain=2 尾排空），跨任务的 stage1/stage2 由 owner 重算
        # 标量（虚拟标量环，无存储）。槽派生全部改全局 gtile 键控（K 槽 %3、
        # S/P 槽 %2——局部键控在任务边界 N_T≢0 (mod 3) 时 K 信用必死锁，推演
        # 见 F7 R7）。
        # F11 源序交换（撤销 D-F6b，对齐 OT swa_kernel_arch35.h:478-517 发射
        # 序 LoadQK(x) → vec1/bmm1(x-1) → vec2/bmm2(x-2)）：body 内先发
        # stage1（tile x-1 的 reinit+vec1+bmm1+预取 Q）、再发 stage2（tile
        # x-2 的 bmm2/vec2+BC 链+copy_out）。逐 tile 账（R11）：两侧 AIC 工作
        # 量持平（0.97×）而 wall 1.87×、重叠 12% vs OT 97%，根因是 stage2
        # 先发让 bmm1(x-1) 排在长 vec2(x-2) 链之后、AIC 空等的锯齿。交换后
        # 同步结构不变（两组 op 无数据依赖、flag 集不相交：vec1 用
        # s_ready/s_free/p_ready/p_free，vec2+链用 o_ready/o_free+bc_*；AIC
        # 的 bmm1/bmm2 资源不相交），发射序只重排不改依赖——死锁推演仍见
        # ITERATIONS.md Iter 23 F6 节与 F7 R7 节。三项数值配套（缺一逐位即
        # 挂）：① dm 双槽（OT softmaxExpBufs[taskIdMod2] 同键 xg%2）——单
        # 缓冲下交换后 vec1(x-1) 写 dm 会覆盖 vec2(x-2) 尚未读的 dm(x-2)；
        # ② vec_init 不再清 ub_o（OT InitVec1SoftmaxFromSinks 只灌 max/sum，
        # csa_block_vector_arch35.h:768-795）——交换后它先于同 body 的
        # vec2(P 末)+copy_out(P)，清零会抹掉 P 末 running O；O 启动改由
        # vec2 fresh 覆盖接管（fresh=is_first_base2，任务首 tile 走 O=PV，
        # 与「O=0 后 O*dm+Otmp」数值恒等）；③ copy_out/vec2 链块内序不变，
        # ub_o 全宽每 tile 重写、无跨 tile 残留依赖。
        drain = 0
        if num_tiles > 0:
            if any_after == 0:
                drain = PRELOAD_NUM
        for i in tla.range(num_tiles + drain):
            x = g_k + i
            # ---- stage 0（AIC）：tile x = 本任务第 i 个 tile ----
            if i < num_tiles:
                n_load0 = kv_len - start_idx - i * KV_BLOCK
                if n_load0 < 0:
                    n_load0 = tla.as_numeric(0)
                if n_load0 > KV_BLOCK:
                    n_load0 = tla.as_numeric(KV_BLOCK)
                slot_k0 = x - (x // KV_SLOTS) * KV_SLOTS
                # PA_BBND（OT swa_kernel_arch35.h:50/:67/:122 的 oriBlockTable 面）：
                # 与 CSA 同一个 GmCopyInToL1PA 语义（CopyInL1.h:322-378），逐页
                # 按 remainRowCnt 截断。start_row 用序列内逻辑行（不含 cu_kv 基址），
                # 页号 = 序列内行 / blockSize（页表按 seq 分行）。
                if ori_pa_block_size > 0:
                    load_k_pa(
                        sync,
                        bufs,
                        kv,
                        ori_bt,
                        seq,
                        start_idx + i * KV_BLOCK,
                        n_load0,
                        slot_k0,
                        dtype_in,
                        ori_pa_block_size,
                        ori_pa_stride,
                        ori_pa_max_blocks,
                        ori_pa_pages,
                    )
                else:
                    load_k(sync, bufs, kv, kv_row0, i, n_load0, slot_k0, dtype_in)
            # ---- stage 1：tile x-1 的 vec1（AIV）与 bmm1（AIC）。owner 二选一
            # （i>=1 本任务 local j1=i-1；i==0 → x-1 = G_T-1 = P 末 tile，is_last
            # 恒真——本任务自己的末 tile 在下一真实任务的首 body 或 drain 落到
            # 该站点）。守卫 i != num_tiles+c1：末真实任务 drain 的第二个 body
            # （x = g_total+1）无 stage1 tile；其余 body 的 x-1 恒为真实 tile。
            if x >= c1:
                if i != num_tiles + c1:
                    j1 = i - c1
                    nt1 = num_tiles
                    vis1 = vis
                    bbn1 = base_block_num
                    goff1 = g_off
                    task1 = task
                    nta1 = n_task
                    pft_task = nx_task
                    pft_ok = nx_done
                    if i < c1:
                        j1 = p_nt - 1
                        nt1 = p_nt
                        vis1 = p_vis
                        bbn1 = p_bbn
                        goff1 = p_goff
                        task1 = p_task
                        nta1 = (p_task - task0) // q_bn
                        # 跨任务站点：owner=P 的末 tile 释放后，预取目标就是
                        # 本任务（P 的后继真实任务 == 当前任务）。
                        pft_task = task
                        pft_ok = 1
                    # owner 任务的两半 Q 槽（{2n, 2n+1} mod 3，n 按任务序
                    # 含空任务派生，与头装/预取的槽键一致；空任务槽位闲置）。
                    qa1 = nta1 + nta1
                    q_slot_a1 = qa1 - (qa1 // 3) * 3
                    qb1 = qa1 + 1
                    q_slot_b1 = qb1 - (qb1 // 3) * 3
                    n_valid1 = vis1 - j1 * KV_BLOCK
                    if n_valid1 > KV_BLOCK:
                        n_valid1 = tla.as_numeric(KV_BLOCK)
                    xg1 = x - c1
                    slot_k1 = xg1 - (xg1 // KV_SLOTS) * KV_SLOTS
                    slot_s1 = xg1 - (xg1 // PRELOAD_NUM) * PRELOAD_NUM
                    # F11 sum/max 链式读槽：非首 tile 读 (x-1)%2（slot_s1+1 取
                    # 模），任务首 tile 读 vec_init 重灌的初值槽（x%2 =
                    # slot_s1）。标量 phi 形态同 n_load0 钳制先例。
                    slot_sum_rd = slot_s1
                    if j1 != 0:
                        slot_sum_rd = slot_s1 + 1
                        if slot_sum_rd > 1:
                            slot_sum_rd = slot_sum_rd - 2
                        # F11·bc：BC 块首 tile（j1!=0 但块内序 0）读 bc_reinit_stats
                        # 的重灌槽 = 本 tile 槽 slot_s1（reinit 紧随其后写同槽；
                        # 值语义：max 重灌 MIN_F → exp(old-new)=0、sum 链断点由
                        # 块首 fresh 接管，与非首 tile 读 (x-1)%2 链槽并列）。
                        if is_batch_consistency == 1:
                            if j1 - (j1 // bbn1) * bbn1 == 0:
                                slot_sum_rd = slot_s1
                    is_first1 = j1 == 0
                    is_last1 = j1 + c1 == nt1
                    # 闸门④ 折叠：任务首 tile 的 vec1 前重灌 running 态
                    # （OT InitVec1SoftmaxFromSinks / ProcessVec1 isFirstSoftmaxBase
                    # 同位，csa_block_vector_arch35.h:940-957）。原任务头串行
                    # 段拆除。F11 交换后 vec_init(T 首) 先于同 body 的
                    # vec2(P 末)+copy_out(P) 发射：vec_init 只灌 ub_max/ub_sum
                    # （O 区清零已拆除，O 启动由 vec2 fresh 覆盖接管），而
                    # vec2(P 末) 不读 max/sum、copy_out(P) 读的 ub_o 由其后
                    # 的 vec2(P 末) 重写——running 态交接仍 == 串行序。
                    if is_first1:
                        vec_init_from_sinks(
                            sync,
                            bufs,
                            sinks,
                            task1,
                            m,
                            rows,
                            dtype_in,
                            goff1,
                            has_sinks,
                            slot_sum=slot_s1,
                            slot_max=slot_s1,
                            clear_o=0,
                        )
                    # BC 块内标量预初始化（前端规则）；bc=0 编译期消除。
                    tid_in_block1 = j1
                    is_first_base1 = j1 == 0
                    reduce_block_id1 = j1
                    # F3 BC（Iter23 配对）：每个规约块首 tile 重灌 vec1 初值
                    # （OT ProcessVec1 :945-948 isFirstSoftmaxBase 重灌/
                    # NoUpdate；语义与串行版逐条一致，t -> j1）。
                    if is_batch_consistency == 1:
                        tid_in_block1 = j1 - (j1 // bbn1) * bbn1
                        is_first_base1 = tid_in_block1 == 0
                        reduce_block_id1 = j1 // bbn1
                        if is_first_base1:
                            if j1 != 0:
                                # F11·bc：重灌本 tile 槽（xg1%2 = slot_s1）——
                                # 紧随的 process_vec1 以 slot_max_rd/slot_sum_rd
                                # = slot_s1 读到重灌值。
                                bc_reinit_stats(bufs, rows, slot_s1)
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
                    iterate_bmm1(
                        sync,
                        bufs,
                        slot_k1,
                        m,
                        is_last1,
                        is_first1,
                        dtype_in,
                        slot_s=slot_s1,
                        q_slot_a=q_slot_a1,
                        q_slot_b=q_slot_b1,
                    )
                    # 闸门② 预取（owner 解析后的统一站点）：owner 末 tile 的
                    # bmm1 已归还 Q 槽信用（iterate_bmm1 is_last_tile 分支 set
                    # q_l0a_free），紧随其后的 load_q wait x1 闸在该释放上
                    # （程序序同 body）。预取目标：跨任务站点=本任务（槽
                    # q_slot_a/b == P 头的 q_slot_a_next）；local 站点=下一
                    # 真实任务 NX（末真实任务 nx_done=0 自然不发）。深度 1
                    # 握手；MTE1 in-order 保证 L1->L0 读完成后才放行 MTE2 重写。
                    if is_last1:
                        if pft_ok == 1:
                            pft_n = (pft_task - task0) // q_bn
                            pft_2n = pft_n + pft_n
                            pft_sa = pft_2n - (pft_2n // 3) * 3
                            pft_2n1 = pft_2n + 1
                            pft_sb = pft_2n1 - (pft_2n1 // 3) * 3
                            load_q(
                                sync,
                                bufs,
                                q,
                                pft_task,
                                m,
                                dtype_in,
                                slot_a=pft_sa,
                                slot_b=pft_sb,
                            )
            # ---- stage 2：tile x-2 的 bmm2/vec2/BC 链（F11 源序交换后在
            # stage 1 之后发射，OT 序见上注）。owner 三选一：i>=2 本任务
            # local（j2=i-2）；i==1 → x-2 =
            # G_T-1 = P 末 tile（is_last 恒真）；i==0 → x-2 = G_T-2 = P 的
            # local N_P-2（N_P>=2）或 PP 末 tile（N_P==1）。守卫只需 x>=c2：
            # 循环上界保证 x-2 <= g_total-1 恒为真实 tile（推演见 F7 R7）。
            if x >= c2:
                j2 = i - c2
                nt2 = num_tiles
                vis2 = vis
                bbn2 = base_block_num
                goff2 = g_off
                task2 = task
                if i < c2:
                    if i == c1:
                        j2 = p_nt - 1
                        nt2 = p_nt
                        vis2 = p_vis
                        bbn2 = p_bbn
                        goff2 = p_goff
                        task2 = p_task
                    else:
                        if p_nt >= 2:
                            j2 = p_nt - 2
                            nt2 = p_nt
                            vis2 = p_vis
                            bbn2 = p_bbn
                            goff2 = p_goff
                            task2 = p_task
                        else:
                            j2 = pp_nt - 1
                            nt2 = pp_nt
                            vis2 = pp_vis
                            bbn2 = pp_bbn
                            goff2 = pp_goff
                            task2 = pp_task
                n_valid2 = vis2 - j2 * KV_BLOCK
                if n_valid2 > KV_BLOCK:
                    n_valid2 = tla.as_numeric(KV_BLOCK)
                xg2 = x - c2
                slot_k2 = xg2 - (xg2 // KV_SLOTS) * KV_SLOTS
                slot_p2 = xg2 - (xg2 // PRELOAD_NUM) * PRELOAD_NUM
                # BC 块内标量（动态 if 要求先初始化：Tla 前端规则）。bc=0 实例
                # 下列值无使用者（引用都在 constexpr if 内，编译期消除）。
                tid_in_block2 = j2
                is_first_base2 = j2 == 0
                reduce_block_id2 = j2
                # F11：fresh 语义扩为「任务首 tile ∪ bc 块首」= is_first_base2
                # （bc=0 时即 j2==0）。任务首 tile 的 O 启动由 fresh 覆盖接管
                # （vec_init_from_sinks 不再清 O，对齐 OT InitVec1SoftmaxFromSinks
                # 只灌 max/sum；数值恒等：0*dm=0），vec2(P 末)+copy_out(P) 与
                # vec_init(T 首) 的 ub_o 交接冲突随之消除——源序交换的前提之一。
                # bc=1 时 is_first_base2 即原 fresh 判据（块首 tile），行为不变。
                fresh_v2 = is_first_base2
                if is_batch_consistency == 1:
                    tid_in_block2 = j2 - (j2 // bbn2) * bbn2
                    is_first_base2 = tid_in_block2 == 0
                    reduce_block_id2 = j2 // bbn2
                    # F3 BC（Iter23 配对）：vec2 fresh DataCopy 仅块首 tile（OT
                    # smlaIsFirstVec2Base=isFirstBase，
                    # csa_block_vector_arch35.h:1079-1084）——块首 tile raw 未归
                    # 一 PV 驻留 ub_o，块内后续 tile FlashUpdateLastNew/
                    # FlashUpdateNew 块内合并（fresh=0 路径）。bc=1 时为动态值，
                    # seg 循环与 iterate_bmm2_seg 的同步结构保持静态（同步不在
                    # 动态分支内）。
                    fresh_v2 = is_first_base2
                wait_p_ready(sync)
                # F17 步骤 B：4 段 mmad -> 每 tile 一次整块 fixpipe -> 4 段
                # vec2（OT IterateBmm2 同构：fixpipe 频次 5 -> 2/tile；两段
                # 循环的同步结构仍保持静态，bc=1 的动态 fresh 不落在同步上）。
                for seg in (0, 1, 2, 3):
                    iterate_bmm2_seg(
                        sync, bufs, seg, slot_k2, m, dtype_in, slot_p=slot_p2
                    )
                iterate_bmm2_fixpipe(sync, bufs, m)
                for seg in (0, 1, 2, 3):
                    process_vec2_seg(
                        sync, bufs, seg, m, rows, dtype_in, fresh_v2, slot_dm=slot_p2
                    )
                # F14 步骤 A：p_release 已去除（p_free 回执删除；OT 同款无回
                # 执，安全链 = s_ready 数据依赖 + AIC 串行程序序 + MTE1 FIFO）。
                # F3 BC 跨块归一化/staging/合并链（=0 实例整块不存在）——语义
                # 与串行版逐条一致（t -> t2）：isLastBase 时 bc_div_cur（块内
                # LastDivNew）、reduceBlockId>0 时 bc_merge_prev、非末 tile
                # bc_stage_cur；B=1 时退化为逐 tile 语义。
                # F11·bc：三个函数的 max/sum 全部接本 tile 槽 slot_p2 = xg2%2
                # （div 读 sum(t)；merge 就地 RMW max(t)/sum(t)；stage 读
                # merge 写回的同槽）——F11 序下这些 op 晚于 vec1(t+1) 执行，
                # 单槽会被 vec1(t+1)/reinit(t+1) 的写（异槽 (t+1)%2）错开，
                # 槽键恢复逐值等价。
                if is_batch_consistency == 1:
                    is_last_base2 = tid_in_block2 == bbn2 - 1
                    if j2 + c1 == nt2:
                        # 动态 if 两臂须同型（Bool）：then 臂条件已保证为真。
                        is_last_base2 = j2 + c1 == nt2
                    if is_last_base2:
                        if nt2 != 1:
                            bc_div_cur(bufs, rows, slot_p2)
                        if reduce_block_id2 != 0:
                            bc_merge_prev(sync, bufs, bc_ws, rows, slot_p2)
                        if j2 + c1 != nt2:
                            bc_stage_cur(sync, bufs, bc_ws, rows, slot_p2)
                release_k(sync, slot_k2)
                # 闸门④ 折叠：owner 末 tile 的 vec2 链后就地 copy_out
                # （OT CopyOutAttentionOut@ProcessVec2 isLastS2 同位，
                # csa_block_vector_arch35.h:1138/:1266）。原任务尾串行段拆除；
                # 与后续 body 的 load/vec1 重叠。
                if j2 + c1 == nt2:
                    bmm2_data_copy_out(
                        sync, bufs, out, task2, m, rows, dtype_in, slot_sum=slot_p2
                    )
                    if enable_lse == 1:
                        # R37：终态 max/sum 在 slot_p2；行基址 = OT goIdx
                        lse_copy_out(
                            sync,
                            bufs,
                            lse,
                            task2,
                            m,
                            rows,
                            slot=slot_p2,
                            row_base=(task2 // g_split) * head_num
                            + (task2 - (task2 // g_split) * g_split) * half_g,
                        )
