# ---------------------------------------------------------------------------
# sparse_flash_mla_csa_kernel.py — SparseFlashMlaCsaKernel（arch35, TND, CSA）
#
# 结构参照 ops-transformer/attention/sparse_flash_mla/op_kernel/arch35/
#   sparse_flash_mla_csa_kernel_arch35.h -> 本文件（入口 / 扁平流水任务循环）
#   sparse_flash_mla_csa_block_cube_arch35.h   -> smla_csa_block_cube.py
#   sparse_flash_mla_csa_block_vector_arch35.h -> smla_csa_block_vector.py
#
# CSA = 两段 KV 源：
#   ori：连续的因果区间（oriMaskMode=3，oriWinLeft/Right=-1）
#        [0, kv_len - q_len + row + 1)  （kvcache.h:99-115 的 pre/next tokens 代入后化简）
#   cmp：压缩 KV 的稀疏索引
#        lines = Min(clip((row + 1 + cmp_len*cmp_ratio + residual - q_len)/cmp_ratio,
#                         0, cmp_len), Min(topk[token], idxWidth))
#        （kvcache.h:332 ComputeS2LoopInfo + :86-93 nextTokensPerBatchCmp）
# 两段共用同一套 softmax 状态（running max / sum / O 重标定），不能分开归一化。
#
# 任务划分与 OT 一致：qSNumInOneBlock=1 ⇒ 一个 task = 一个 token × head_num 行。
# 稀疏行的搬运在 AIV 侧（16 行/8 行两级 gather 到 GM 暂存槽），cube 再从暂存连续读。
#
# ==== F27：扁平 4 深流水（PRELOAD_NUM=3）====
# OT ProcessMainLoop（csa_kernel_arch35.h:466-700）把 (bnIdx, gS1Index, s2LoopCount)
# 三元组展平成单一 taskId 递增流，每个 s2 迭代体里按滞后发射 5 个站点：
#   vec0(t)   —— lag 0（本 task 的稀疏 gather，AIV）
#   load(t-1) —— lag 1（AIC MTE2：Q 首装 / K 段，索引 runInfo[(taskId+3)%4]）
#   vec1(t-2) + bmm1(t-2) —— lag 2（runInfo[(taskId+2)%4]）
#   bmm2(t-3) + vec2(t-3) —— lag 3（runInfo[(taskId+1)%4]）
# 本文件用同一套结构：每个 task 的 tile 展开成 (i, x) 的扁平行号 x，body 内
# 依次发射上述 5 站，站点边界一律用"该 tile 是否存在"的判据（见 stage 注释）；
# 每站的参数从深度 4 的 RunInfo 环 r0/r1/r2/r3 取（入队时刻冻结，AIC/AIV 都不
# 重算 owner 的 seq/len/tile 号）。尾排空由"哨兵步"承担：核上最后一个真实 task
# 之后以 n_tiles=0 的哨兵步补 PRELOAD_NUM 个 body（OT :557
# gS1LoopEnd = gs1LoopEndIdx + PRELOAD_NUM），哨兵步不重算参数、只消费环里
# 残留的冻结值（等价 OT 的 (taskId+k)%4 陈旧槽）。
# 逐行对照表、环轮转推演与不变量见 ITERATIONS.md F27。
# ---------------------------------------------------------------------------

from __future__ import annotations

import catlass.tla as tla

from smla_csa_block_cube import (
    iterate_bmm1,
    iterate_bmm2_fixpipe,
    iterate_bmm2_seg,
    load_k,
    load_k_pa,
    load_k_ptr,
    load_q,
    release_k,
    split_g_mte2_pair,
    wait_kv_stage,
    wait_p_ready,
)
from smla_csa_block_vector import (
    clean_output,
    bmm2_data_copy_out,
    lse_copy_out,
    configure_trace,
    gather_kv_rows,
    gather_kv_rows_pa_pairwise,
    gather_kv_rows_ring,
    gather_kv_rows_vec,
    get_kv_phy_addr_for_kv_type,
    process_vec1,
    process_vec2_seg,
    vec_init_from_sinks,
)
from smla_csa_common import (
    compute_need_init,
    KV_BLOCK,
    KV_SLOTS,
    P_SLOTS,
    alloc_buffers,
    declare_sync,
    init_sync,
    init_sync_preload,
    static_range,
    sync_all_cores_aic,
    sync_all_cores_aiv,
)

KV_STAGE_STRIDE = KV_BLOCK * 512  # 一个暂存槽的元素数（128 行 x 512）
WS_BLOCKS = 32  # 暂存区的按核分片数（950PR 最多 32 个 AIC）

# OT PRELOAD_NUM（csa_kernel_arch35.h:557/:564 的尾排空深度）
PRELOAD_NUM = 3

# 非 PA 稀疏地址走索引环（r10 T1-1 已采纳）：省逐行 GM 表读；PA（kv_layout==2）
# 仍走原表路径，其 IR 与改前逐条相同（PA #25 控制臂保持 kernel.o 逐字节不变）。
# PA 例走内联页数学 + 成对落位（R35 已采纳），不再建表。


@tla.kernel
def sparse_flash_mla_csa_kernel(
    q: tla.Tensor,  # [T*H, D]  TND 展平
    ori_kv: tla.Tensor,  # [KV, D]；PA_BBND 时展平为 (blockNum*blockSize*n2, D)
    cmp_kv: tla.Tensor,  # [CMP, D]；同上
    ori_bt: tla.Tensor,  # [B, maxBlocks] int32（非 PA 时 1 元素占位）
    cmp_bt: tla.Tensor,  # [B, maxBlocks] int32（非 PA 时 1 元素占位）
    idx: tla.Tensor,  # [T*idx_width] int32 展平
    topk: tla.Tensor,  # [T] int32
    cu_q: tla.Tensor,  # [B+1] int32
    cu_kv: tla.Tensor,  # [B+1] int32
    cu_cmp: tla.Tensor,  # [B+1] int32
    resid: tla.Tensor,  # [B] int32
    sinks: tla.Tensor,  # [H] fp32
    out: tla.Tensor,  # [T*H, D]
    lse: tla.Tensor,  # [T*H] fp32 softmax_lse（enable_lse=0 时传 1 元素占位）
    tiling: tla.Tensor,  # [4] int32；F27：tiling[1] = IS_SPLIT_G 的 MTE2 屏障配额
    # （OT FA_S2_MAX_NUM，host 计算；g_split==1 时不读）
    ws: tla.Tensor,  # 稀疏 gather 的 GM 暂存槽：KV_SLOTS * 128 * 512
    cmp_phy: tla.Tensor,  # F2：cmp 段物理地址表 [T*cmp_phy_width] int64（=0 时 1 元素占位）
    softmax_scale: tla.Float32,
    head_num: tla.Constexpr[int],
    head_dim: tla.Constexpr[int],
    num_seqs: tla.Constexpr[int],
    idx_width: tla.Constexpr[int],
    ori_mask_mode: tla.Constexpr[int],
    ori_win_left: tla.Constexpr[int],
    ori_win_right: tla.Constexpr[int],
    cmp_mask_mode: tla.Constexpr[int],
    cmp_ratio: tla.Constexpr[int],
    ori_pa_block_size: tla.Constexpr[int],
    ori_pa_stride: tla.Constexpr[int],
    ori_pa_max_blocks: tla.Constexpr[int],
    ori_pa_pages: tla.Constexpr[int],
    cmp_pa_block_size: tla.Constexpr[int],
    cmp_pa_stride: tla.Constexpr[int],
    cmp_pa_max_blocks: tla.Constexpr[int],
    g_split: tla.Constexpr[int],
    has_sinks: tla.Constexpr[int],
    enable_lse: tla.Constexpr[int],  # R37：softmax_lse 落盘开关
    topk_value_mode: tla.Constexpr[int],  # F5: 1=TOPK_INDEX, 2=TOPK_OFFSET
    # ---- F2（IS_VEC_S2PHYADDR，tiling.cpp:2498-2499/:2572 的第 8 模板参数语义）----
    # kv_layout 取 OT SMLA_LAYOUT（sparse_flash_mla_common.h:34-37）：BSND=0/TND=1/PA_BBND=2
    kv_layout: tla.Constexpr[int],
    is_vec_s2phyaddr: tla.Constexpr[int],
    cmp_phy_width: tla.Constexpr[int],  # Align128(idx_width)，表行宽（int64 元素）
) -> None:
    dtype_in = q.ptr.dtype
    # IS_SPLIT_G：任务空间 = token × G 半区，m 为半区 G 宽度
    m = head_num // g_split
    # AIV 两个子块行数：firstHalf=(m+1)/2、second=m-firstHalf（OT 同名量）；
    # rows 取前半（偶数 m 即 m/2，奇数 m 时第二子块写 0 行，见 bmm2_data_copy_out）。
    rows = m - m // 2
    # R37：OT goIdx = half*halfG（kvcache.h:250 IS_SPLIT_G 偏移）；奇 head_num 时
    # halfG=(h+1)/2 ≠ m，lse 行基址不能用 task*m。
    half_g = head_num // g_split if head_num % g_split == 0 else head_num // g_split + 1

    # 共用 gather 的 kernel 必须经同一入口（F45）：CSA 不实现共享槽臂，显式声明 False。
    configure_trace(g_split, False, False, KV_SLOTS, "csa")

    # T1-1（r10）：非 PA 去表实例。kv_layout 是 Constexpr ⇒ 此处为 Python 折叠，
    # 不产生运行期判据。
    # r12：PA（kv_layout==2）改走内联页数学 + OT 成对落位，不再建表/读表；
    # use_vec_gather 即「走预计算物理地址表（vec / ring）」的实例门。
    use_vec_gather = (is_vec_s2phyaddr == 1) and (kv_layout != 2)
    idx_ring_on = use_vec_gather and kv_layout != 2

    sync = declare_sync(need_idx_ring=idx_ring_on, need_lse_war=enable_lse == 1)
    # F2：=1 实例才分配 phy-addr 生成的 UB 暂存（ub_phy_idx/bt/addr）
    # F27：preload=True 打开流水所需的双槽（l1_p/l0c_s/ub_s/max/sum/dm——OT
    # l1PBuffers[2] / bmm1Buffers[2] / softmax*Bufs[2] 的字节表本仓已在
    # alloc_buffers 内按 preload 分支声明）。
    bufs = alloc_buffers(
        dtype_in,
        phy=cmp_phy_width if use_vec_gather else 0,
        phy_bt=cmp_pa_max_blocks if use_vec_gather else 0,
        preload=True,
        idx_ring=idx_ring_on,
    )
    init_sync(
        sync,
        pre_phy_vs=use_vec_gather and not idx_ring_on,
        pre_l0c2=True,
        pre_idx_ring=idx_ring_on,
    )
    init_sync_preload(sync)
    # F4（OT csa_kernel_arch35.h:165 CleanOutput）：Init 阶段整张输出清零
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

    # F27 补（#046 间歇性非逐位）：OT csa_kernel_arch35.h:444-449 的 Process()
    # 入口屏障（「SyncAll Cube和Vector都需要调用」）。分布式 init 清零必须在任何
    # 核开始写 O 之前全部完成，否则迟到的 init 会把其它核已写好的行清 0——
    # 正控已逐元素复现、且加上本屏障后正控不再生效（ITERATIONS.md §4.12）。
    # 条件与 OT 相同：IS_VEC_S2PHYADDR=1 无条件；否则 needInit=1 时同步。
    if is_vec_s2phyaddr == 1:
        with tla.cube():
            sync_all_cores_aic()
        with tla.vector():
            sync_all_cores_aiv()
    else:
        if need_init == 1:
            with tla.cube():
                sync_all_cores_aic()
            with tla.vector():
                sync_all_cores_aiv()

    total_tasks = cu_q[num_seqs] * g_split
    task0 = tla.arch.block_idx()
    q_bn = tla.arch.block_num()

    # ---- F27：IS_SPLIT_G 的 MTE2 屏障配额（OT FA_S2_MAX_NUM）----
    # OT 每核的 smlaMaxS2LoopCnt 来自 metadata 的 FA_S2_MAX_NUM（= 各核
    # coreCache.s2Loop 的最大值，sparse_flash_mla_metadata_aicpu.cpp:1420/:1595），
    # 每个真实 load 消费 1（:637-638）、每个被跳过的 load 也消费 1（:645-648），
    # 末尾把剩余配额用等量 set+wait 排空（:674-680）。本仓把该配额经 tiling[1]
    # 传入、用"预算排空"实现：真实 load 与跳过 load 各消费 1，末尾按剩余量补发，
    # 于是每核恰好发出配额条 —— 条数恒等只依赖"所有核读到同一个配额值"，
    # 不依赖 host 算得准（算错只改变同步密度，不会挂死）。
    # g_split==1 实例（无 split-G 指令形态）不读 tiling[1]、不发出任何一条。
    s2_budget = tla.as_numeric(0)
    c_one = 1
    if g_split == 2:
        s2_budget = tiling[c_one]

    # ---- F27 RunInfo 环：OT smla35RunInfo[4]，深度 4 ----
    # 每站的参数在"入队时刻"冻结进 r0（= OT SetRunInfo(smla35RunInfo[taskId%4])），
    # 消费站直接读环，不再按 owner 重算。环用轮转寄存器组实现：body 开始前
    #   r1 = tile(x-1) 的参数、r2 = tile(x-2)、r3 = tile(x-3)
    # body 末尾 r3<-r2、r2<-r1，真实 tile 才再覆盖 r1<-r0（哨兵步/drain 不覆盖，
    # 与 OT (taskId+k)%4 的陈旧槽语义逐条一致）。
    # 每槽 8 个标量：task / j（task 内 tile 号）/ nt（task 内 tile 数）/
    # n_valid / n_load / kv_row0 / is_cmp / seq。
    # g_off（G 轴偏移）不占槽：由 task 派生（task - (task//g_split)*g_split）*m。
    r0_task = tla.as_numeric(0)
    r0_j = tla.as_numeric(0)
    r0_nt = tla.as_numeric(1)
    r0_nvalid = tla.as_numeric(0)
    r0_nload = tla.as_numeric(0)
    r0_kvrow0 = tla.as_numeric(0)
    r0_iscmp = tla.as_numeric(0)
    r0_seq = tla.as_numeric(0)
    r0_start = tla.as_numeric(0)
    r1_task = tla.as_numeric(0)
    r1_j = tla.as_numeric(0)
    r1_nt = tla.as_numeric(1)
    r1_nvalid = tla.as_numeric(0)
    r1_nload = tla.as_numeric(0)
    r1_kvrow0 = tla.as_numeric(0)
    r1_iscmp = tla.as_numeric(0)
    r1_seq = tla.as_numeric(0)
    r1_start = tla.as_numeric(0)
    r2_task = tla.as_numeric(0)
    r2_j = tla.as_numeric(0)
    r2_nt = tla.as_numeric(1)
    r2_nvalid = tla.as_numeric(0)
    r2_nload = tla.as_numeric(0)
    r2_kvrow0 = tla.as_numeric(0)
    r2_iscmp = tla.as_numeric(0)
    r2_seq = tla.as_numeric(0)
    r2_start = tla.as_numeric(0)
    r3_task = tla.as_numeric(0)
    r3_j = tla.as_numeric(0)
    r3_nt = tla.as_numeric(1)
    r3_nvalid = tla.as_numeric(0)
    r3_nload = tla.as_numeric(0)
    r3_kvrow0 = tla.as_numeric(0)
    r3_iscmp = tla.as_numeric(0)
    r3_seq = tla.as_numeric(0)
    r3_start = tla.as_numeric(0)

    # 本核已消费的真实 tile 数：进入某个 task 时 = 该 task 的全局前缀 g_k，
    # 任务循环退出后 = 本核的 g_total（尾排空的 x 起点）。O(1) 累加，
    # 不做 OT 之外的前缀重扫。
    g_done = tla.as_numeric(0)
    # 暂存区按核分片（OT: v0ResGm 来自 workspace[blockIdx * perBlockSize]）。
    # 不分片时 28 个 block 会同时往同一组槽里写，AIC 读到的行由时序决定。
    ws_base = tla.arch.block_idx() * (KV_SLOTS * KV_STAGE_STRIDE)

    # 本核的 task 步数（ceil）+1 个哨兵步：哨兵步 = OT 的 PRELOAD_NUM 尾排空。
    n_steps = (total_tasks - task0 + q_bn - 1) // q_bn
    if n_steps < 0:
        n_steps = tla.as_numeric(0)

    for step in tla.range(n_steps + 1):
        task = task0 + step * q_bn

        # 任务级标量先给默认值：哨兵步（step == n_steps）跳过任务头，n_tiles 归 0，
        # 站点判据退化为 drain 形态；站点 1/2/3 的参数一律取自 RunInfo 环，故任务级
        # 标量取默认值不影响排空（站点 0 与 Q 预取在哨兵步按判据不执行）。
        n_tiles = tla.as_numeric(0)
        seq = tla.as_numeric(0)
        tok = tla.as_numeric(0)
        rowq = tla.as_numeric(0)
        start_idx = tla.as_numeric(0)
        vis = tla.as_numeric(0)
        kv_len = tla.as_numeric(0)
        n_ori_tiles = tla.as_numeric(0)
        cmp_lines = tla.as_numeric(0)
        kv_row0 = tla.as_numeric(0)
        cmp_row0 = tla.as_numeric(0)

        if step < n_steps:
            # ---- 定位 (seq, row)；稀疏索引按 token 行 tok = task // g_split ----
            tok = task // g_split
            rowq = tok
            for bb in static_range(1, num_seqs):
                if tok >= cu_q[bb]:
                    seq = bb
                    rowq = tok - cu_q[bb]
            q_len = cu_q[seq + 1] - cu_q[seq]
            kv_len = cu_kv[seq + 1] - cu_kv[seq]
            cmp_len = cu_cmp[seq + 1] - cu_cmp[seq]

            # ---- ori：ComputeParamBatch 三模式（kvcache.h:96-114）后
            #      start = clip(row - pre, 0, kv)、end = clip(row + next + 1, 0, kv) ----
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
            n_ori_tiles = (vis + KV_BLOCK - 1) // KV_BLOCK

            # ---- cmp：稀疏行数（kvcache.h:332 + :86-93）----
            sparse_count = topk[tok]
            if sparse_count > idx_width:
                sparse_count = tla.as_numeric(idx_width)
            if sparse_count < 0:
                sparse_count = tla.as_numeric(0)
            # nextTokensPerBatchCmp（kvcache.h:86-93）：cmpMaskMode==0 时不减 q_len、不加 resid
            next_cmp = cmp_len * cmp_ratio  # cmpMaskMode == 0 的默认值
            if cmp_mask_mode != 0:
                next_cmp = cmp_len * cmp_ratio + resid[seq] - q_len
            cmp_end = (rowq + 1 + next_cmp) // cmp_ratio
            if cmp_end < 0:
                cmp_end = tla.as_numeric(0)
            if cmp_end > cmp_len:
                cmp_end = cmp_len
            cmp_lines = cmp_end
            if cmp_lines > sparse_count:
                cmp_lines = sparse_count

            n_tiles = n_ori_tiles + (cmp_lines + KV_BLOCK - 1) // KV_BLOCK
            kv_row0 = cu_kv[seq] + start_idx
            cmp_row0 = cu_cmp[seq]

        # ---- 本 task 的 body 数：真实 task = n_tiles；哨兵步 = PRELOAD_NUM
        #      （空核——本核一个真实 tile 都没有——不补，避免越界 load）----
        n_body = n_tiles
        if step == n_steps:
            if g_done > 0:
                n_body = n_tiles + PRELOAD_NUM

        for i in tla.range(n_body):
            x = g_done + i

            # ================================================================
            # 站点 0（lag 0，AIV）：vec0(x) —— 稀疏 gather 落到 GM 暂存槽
            # OT ProcessVec0(runInfo[smlaRunInfoIdx])（:606），参数 = 本 body 入队的
            # 冻结值（j < n_tiles 才存在真实 tile；哨兵步不执行）
            # ================================================================
            if i < n_tiles:
                # ---- 入队：OT SetRunInfo(smla35RunInfo[taskId%4])（:600）----
                j0 = i
                nv0 = tla.as_numeric(0)
                nl0 = tla.as_numeric(0)
                if j0 < n_ori_tiles:
                    nv0 = vis - j0 * KV_BLOCK
                    if nv0 > KV_BLOCK:
                        nv0 = tla.as_numeric(KV_BLOCK)
                    nl0 = kv_len - start_idx - j0 * KV_BLOCK
                    if nl0 < 0:
                        nl0 = tla.as_numeric(0)
                    if nl0 > KV_BLOCK:
                        nl0 = tla.as_numeric(KV_BLOCK)
                else:
                    nv0 = cmp_lines - (j0 - n_ori_tiles) * KV_BLOCK
                    if nv0 > KV_BLOCK:
                        nv0 = tla.as_numeric(KV_BLOCK)
                    nl0 = nv0
                    if nl0 < 0:
                        nl0 = tla.as_numeric(0)
                r0_task = task
                r0_j = j0
                r0_nt = n_tiles
                r0_nvalid = nv0
                r0_nload = nl0
                r0_kvrow0 = kv_row0
                r0_start = start_idx
                r0_seq = seq
                r0_iscmp = tla.as_numeric(0)
                if j0 >= n_ori_tiles:
                    r0_iscmp = tla.as_numeric(1)

                # vec0：仅 cmp tile 走 AIV gather（ori tile 由 AIC 直读 ori_kv）
                if j0 >= n_ori_tiles:
                    base_line0 = (j0 - n_ori_tiles) * KV_BLOCK
                    slot0 = x - (x // KV_SLOTS) * KV_SLOTS
                    # T1-1（r10）：索引环 2 槽按 x 轮转（每个 AIV 自己的 UB 环）
                    islot = x - (x // 2) * 2
                    # AIV 侧 gather：同一 tile 的 n_load 行按 CeilDiv(n,2) 分给两个子块
                    if kv_layout == 2 and not use_vec_gather:
                        # r12：PA 内联页数学 + OT CopyInKvSparse 成对落位
                        gather_kv_rows_pa_pairwise(
                            sync,
                            bufs,
                            ws,
                            cmp_kv,
                            idx,
                            tok,
                            idx_width,
                            base_line0,
                            nl0,
                            slot0,
                            KV_STAGE_STRIDE,
                            ws_base,
                            block_table=cmp_bt,
                            bt_seq=seq,
                            bt_max=cmp_pa_max_blocks,
                            pa_block_size=cmp_pa_block_size,
                            pa_stride=cmp_pa_stride,
                            topk_value_mode=topk_value_mode,
                        )
                    elif is_vec_s2phyaddr == 1:
                        # T1-1：索引行 DMA -> UB 环 -> 标量读 + 基址加法
                        gather_kv_rows_ring(
                            sync,
                            bufs,
                            ws,
                            cmp_kv,
                            idx,
                            tok,
                            idx_width,
                            cmp_row0,
                            base_line0,
                            nl0,
                            slot0,
                            KV_STAGE_STRIDE,
                            islot,
                            ws_base,
                        )
                    else:
                        gather_kv_rows(
                            sync,
                            bufs,
                            ws,
                            cmp_kv,
                            idx,
                            tok,
                            idx_width,
                            base_line0,
                            nl0,
                            cmp_row0,
                            slot0,
                            KV_STAGE_STRIDE,
                            ws_base,
                            block_table=cmp_bt,
                            bt_seq=seq,
                            bt_max=cmp_pa_max_blocks,
                            pa_block_size=cmp_pa_block_size,
                            pa_stride=cmp_pa_stride,
                            topk_value_mode=topk_value_mode,
                        )

            # ================================================================
            # 站点 1（lag 1，AIC MTE2）：load(x-1)
            # OT IterateLoadQK(runInfo[(taskId+3)%4], isFirstLoop)（:628），
            # 发射条件 taskId>0 && smlaNotLastTwoLoop（:627）⇔ 本仓 "tile(x-1) 存在"
            # ⇔ x>=1 && i < n_tiles+1（哨兵步 n_tiles=0 时退化为只跑 i=0，
            # 正好是 OT 尾排空的第一个 body）。
            # 不满足该条件的迭代（只可能出现在尾排空）走 OT :642-648 的跳过分支：
            # IS_SPLIT_G 下 smlaMaxS2LoopCnt-- 并补一对核间 MTE2 屏障。
            # ================================================================
            if x >= 1:
                if i < n_tiles + 1:
                    xg1 = x - 1
                    slot_k1 = xg1 - (xg1 // KV_SLOTS) * KV_SLOTS
                    # OT IterateLoadQK 首迭代的 CopyQGmToL1（:331-333，isFirstLoop
                    # 是整个核的第一次 load，不是每 task）——只负责本核首 tile 的 Q；
                    # 其余 task 的 Q 一律由站点 2 的 notLastTwoLoop 预取（下一段）。
                    # 目标 task 取环里的 r1（= 正在 load 的 tile 的 task，OT 的
                    # CopyQGmToL1(runInfo) 同款）：首 tile 的 task 未必是 task0
                    # （本核前几个 task 可能没有 tile）。
                    if xg1 == 0:
                        qn1 = (r1_task - task0) // q_bn
                        q2n1 = qn1 + qn1
                        qa1 = q2n1 - (q2n1 // 3) * 3
                        q2n1b = q2n1 + 1
                        qb1 = q2n1b - (q2n1b // 3) * 3
                        load_q(
                            sync, bufs, q, r1_task, m, dtype_in, slot_a=qa1, slot_b=qb1
                        )
                    # OT :347-348 / :357-358：两条 load 分支之前各一对屏障
                    if g_split == 2:
                        if s2_budget > 0:
                            s2_budget = s2_budget - 1
                            split_g_mte2_pair(sync)
                    if r1_iscmp == 1:
                        # 稀疏：等两个 AIV 把行写进 GM 暂存槽（OT v0ResGm 跨核等待）
                        wait_kv_stage(sync)
                        load_k_ptr(
                            sync,
                            bufs,
                            ws.ptr + ws_base + slot_k1 * KV_STAGE_STRIDE,
                            r1_nload,
                            slot_k1,
                            dtype_in,
                        )
                    else:
                        if ori_pa_block_size > 0:
                            load_k_pa(
                                sync,
                                bufs,
                                ori_kv,
                                ori_bt,
                                r1_seq,
                                r1_start + r1_j * KV_BLOCK,
                                r1_nload,
                                slot_k1,
                                dtype_in,
                                ori_pa_block_size,
                                ori_pa_stride,
                                ori_pa_max_blocks,
                                ori_pa_pages,
                            )
                        else:
                            load_k(
                                sync,
                                bufs,
                                ori_kv,
                                r1_kvrow0,
                                r1_j,
                                r1_nload,
                                slot_k1,
                                dtype_in,
                            )
                else:
                    if g_split == 2:
                        if s2_budget > 0:
                            s2_budget = s2_budget - 1
                            split_g_mte2_pair(sync)

            # ================================================================
            # 站点 2（lag 2）：vec1(x-2)（AIV）+ bmm1(x-2)（AIC）
            # OT ProcessVec1((taskId+2)%4)（:612-615，条件 taskId>1 && smlaNotLast）
            # 与 IterateBmm1(bmm1Slot, smlaNotLastTwoLoop, Next=(taskId+3)%4,
            # runInfo2=(taskId+2)%4)（:653-660，同条件）⇔ "tile(x-2) 存在"
            # ⇔ x>=2 && i < n_tiles+2（哨兵步：i=0,1）。
            # 发射序 vec1 -> bmm1 与 OT 一致（AIV 先 vec1、AIC 先 bmm1，各自单流
            # 程序序不变；跨核依赖由 s_ready/s_free 信用手递）。
            # vec1 前的 reinit（任务首 tile 的 max=sinks/sum=1）与 Q 预取都在本站点。
            # ================================================================
            if x >= 2:
                if i < n_tiles + 2:
                    xg2 = x - 2
                    slot_s2 = xg2 - (xg2 // P_SLOTS) * P_SLOTS
                    slot_k2 = xg2 - (xg2 // KV_SLOTS) * KV_SLOTS
                    # sum/max 的读槽：任务首 tile 读本 tile 槽（vec_init 重灌的初值），
                    # 其余读上一 tile 槽（(x-1)%2 链式接力；F11 键）
                    slot_sum_rd = slot_s2
                    if r2_j > 0:
                        slot_sum_rd = 1 - slot_s2
                    # Q 3 槽环（闸门②）：宿主 task 的两半槽号 = 本核内 task 序号 n 的
                    # {2n, 2n+1} mod 3（与 load_q 侧同一派生式）
                    q_nta2 = (r2_task - task0) // q_bn
                    q_two2 = q_nta2 + q_nta2
                    qa2 = q_two2 - (q_two2 // 3) * 3
                    q_two2b = q_two2 + 1
                    qb2 = q_two2b - (q_two2b // 3) * 3
                    if r2_j == 0:
                        # OT InitVec1SoftmaxFromSinks（任务首 tile）：只灌 max/sum，
                        # O 由本站点后一位的 vec2 fresh 覆盖启动（clear_o=0，F11）
                        goff2 = (r2_task - (r2_task // g_split) * g_split) * m
                        vec_init_from_sinks(
                            sync,
                            bufs,
                            sinks,
                            r2_task,
                            m,
                            rows,
                            dtype_in,
                            goff2,
                            has_sinks,
                            slot_sum=slot_s2,
                            slot_max=slot_s2,
                            clear_o=0,
                        )
                    process_vec1(
                        sync,
                        bufs,
                        softmax_scale,
                        r2_nvalid,
                        m,
                        rows,
                        dtype_in,
                        slot_s=slot_s2,
                        slot_p=slot_s2,
                        preload=1,
                        slot_sum_rd=slot_sum_rd,
                        slot_max_rd=slot_sum_rd,
                    )
                    iterate_bmm1(
                        sync,
                        bufs,
                        slot_k2,
                        m,
                        r2_j + 1 == r2_nt,
                        r2_j == 0,
                        dtype_in,
                        slot_s=slot_s2,
                        q_slot_a=qa2,
                        q_slot_b=qb2,
                    )
                    # OT IterateBmm1Impl 末尾（:422-429）：宿主 tile 是本 task 末 tile 时，
                    # 归还 Q 两半槽信用并（notLastTwoLoop）预取"下一 task"的 Q。
                    # 目标 task 由环派生 = r2_task + q_bn（环里 x-1 恒为"宿主 tile 的
                    # 直接后继"，其 task 就是下一 task，OT CopyQGmToL1(runInfoNext)
                    # 的 runInfoNext = (taskId+3)%4 同款）。守卫 = "tile(x-1) 存在"
                    # ⇔ i < n_tiles+1：真实 body 恒真；哨兵步只有 i=0 可能命中
                    # （= OT extra=0 的 notLastTwoLoop 分支，末 task 仅 1 个 tile 时
                    # 正是它负责装该 task 的 Q），i=1/2 对应 OT 的 extra>=1 → 不发。
                    if r2_j + 1 == r2_nt:
                        if i < n_tiles + 1:
                            qn_p = q_nta2 + 1
                            q2n_p = qn_p + qn_p
                            qap = q2n_p - (q2n_p // 3) * 3
                            q2np_b = q2n_p + 1
                            qbp = q2np_b - (q2np_b // 3) * 3
                            load_q(
                                sync,
                                bufs,
                                q,
                                r2_task + q_bn,
                                m,
                                dtype_in,
                                slot_a=qap,
                                slot_b=qbp,
                            )

            # ================================================================
            # 站点 3（lag 3）：bmm2(x-3) + vec2(x-3) + 末 tile copy_out（AIC+AIV）
            # OT IterateBmm2(runInfo3=(taskId+1)%4)（:666-672，条件 taskId>2）
            # ⇔ "tile(x-3) 存在" ⇔ x>=3（body 内恒有 i>=0 ⇒ 只要 x>=3 就一定存在）。
            # copy_out（OT Bmm2DataCopyOut）本仓折叠到 owner task 末 tile 的 vec2 之后
            # （OT 每 tile 结束后 at::Bmm2DataCopyOut 同款，位置与串行版等价）。
            # ================================================================
            if x >= 3:
                xg3 = x - 3
                slot_k3 = xg3 - (xg3 // KV_SLOTS) * KV_SLOTS
                # P 槽 / dm 槽 / sum 归一化槽同键 xg%2（F6/F11）
                slot_p3 = xg3 - (xg3 // P_SLOTS) * P_SLOTS
                wait_p_ready(sync)
                for seg in (0, 1, 2, 3):
                    iterate_bmm2_seg(
                        sync, bufs, seg, slot_k3, m, dtype_in, slot_p=slot_p3
                    )
                # F17 步骤 B：4 段 mmad -> 每 tile 一次整块 fixpipe -> 4 段 vec2
                iterate_bmm2_fixpipe(sync, bufs, m)
                # fresh 判据（OT smlaIsFirstVec2Base）：任务首 tile。必须是 Bool——
                # process_vec2_seg 的 fresh 是运行期谓词（Int 会报 Expected scalar
                # Bool predicate, got i32）
                fresh_v3 = r3_j == 0
                for seg in (0, 1, 2, 3):
                    process_vec2_seg(
                        sync, bufs, seg, m, rows, dtype_in, fresh_v3, slot_dm=slot_p3
                    )
                release_k(sync, slot_k3)
                if r3_j + 1 == r3_nt:
                    bmm2_data_copy_out(
                        sync, bufs, out, r3_task, m, rows, dtype_in, slot_sum=slot_p3
                    )
                    if enable_lse == 1:
                        # R37：终态 max/sum 在 slot_p3；行基址 = OT goIdx
                        lse_copy_out(
                            sync,
                            bufs,
                            lse,
                            r3_task,
                            m,
                            rows,
                            slot=slot_p3,
                            row_base=(r3_task // g_split) * head_num
                            + (r3_task - (r3_task // g_split) * g_split) * half_g,
                        )

            # ---- 环轮转（body 末尾）：r3<-r2、r2<-r1、真实 tile 才 r1<-r0 ----
            rt = r2_task
            r3_task = rt
            rt = r2_j
            r3_j = rt
            rt = r2_nt
            r3_nt = rt
            rt = r2_nvalid
            r3_nvalid = rt
            rt = r2_nload
            r3_nload = rt
            rt = r2_kvrow0
            r3_kvrow0 = rt
            rt = r2_iscmp
            r3_iscmp = rt
            rt = r2_start
            r3_start = rt
            rt = r2_seq
            r3_seq = rt
            r2_task = r1_task
            r2_j = r1_j
            r2_nt = r1_nt
            r2_nvalid = r1_nvalid
            r2_nload = r1_nload
            r2_kvrow0 = r1_kvrow0
            r2_start = r1_start
            r2_iscmp = r1_iscmp
            r2_seq = r1_seq
            if i < n_tiles:
                r1_task = r0_task
                r1_j = r0_j
                r1_nt = r0_nt
                r1_nvalid = r0_nvalid
                r1_nload = r0_nload
                r1_kvrow0 = r0_kvrow0
                r1_start = r0_start
                r1_iscmp = r0_iscmp
                r1_seq = r0_seq

        # 本 task 结束：累计本核 tile 前缀（哨兵步 n_tiles=0，不再变化）
        g_done = g_done + n_tiles

    # ---- 预算排空（OT :674-680）：把剩余配额用等量屏障对补发，保证每核条数恒等 ----
    if g_split == 2:
        for _k in tla.range(s2_budget):
            split_g_mte2_pair(sync)
