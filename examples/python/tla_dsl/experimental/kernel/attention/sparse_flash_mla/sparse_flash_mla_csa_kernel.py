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
    iterate_bmm2_cube64,
    load_k,
    load_k_pa,
    load_k_ptr,
    load_q,
    load_q_half0,
    load_q_half1,
    release_k,
    split_g_mte2_pair,
    wait_kv_stage,
    wait_p_ready,
)
from smla_csa_block_vector import (
    clean_output,
    fd_merge_task,
    fd_store_partial,
    fd_unit_guard,
    lse_copy_out,
    configure_trace,
    configure_pa,
    gather_kv_rows,
    gather_kv_rows_l1,
    gather_kv_rows_pa_l1,
    tnd_l1_s3_stage4a,
    tnd_l1_s3_stage4b,
    tnd_l1_s3_stage0_async,
    tnd_l1_s3_stage25_full,
    pa_l1_s3_stage25_full,
    gather_kv_rows_pa_pairwise,
    gather_kv_rows_ring,
    gather_kv_rows_vec,
    get_kv_phy_addr_for_kv_type,
    process_vec1,
    process_vec2_full_opt,
    process_vec2_seg,
    vec_init_from_sinks,
    wait_kv_free,
)
from smla_csa_common import (
    compute_need_init,
    KV_BLOCK,
    KV_SLOTS,
    P_SLOTS,
    alloc_buffers,
    declare_sync,
    init_sync,
    init_sync_kv_ring,
    init_sync_preload,
    static_range,
    sync_all_cores_aic,
    sync_all_cores_aiv,
)

KV_STAGE_STRIDE = KV_BLOCK * 512  # 一个暂存槽的元素数（128 行 x 512）
WS_BLOCKS = 32  # 暂存区的按核分片数（950PR 最多 32 个 AIC）

# E17/E23：PA cmp 段 AIV 直写 L1（extern 换形）的开关。
# E23 的 rot2+pitch17 extern 已达 121 GB/s/核，但直写路径在生产 A/B 上
# 52.66/52.57 µs，仍慢于 base 的 49.9/50.2（尾部/交棒结构成本），故默认关闭；
# 仅后续实验（如半环+rot2 组合）显式置 1。
PA_L1_ENABLE = 0

# OT PRELOAD_NUM（csa_kernel_arch35.h:557/:564 的尾排空深度）
PRELOAD_NUM = 3

# E7：host 规划表的每项 int32 字段数：[task, tile_start, tile_end, ws_slot,
# merge_flag, units_in_task, unit_index_in_task, task_base_ws]。kernel 只读前 4 项。
FD_TAB_STRIDE = 8


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
    ws_fd: tla.Tensor,  # E4：FD 部分量 fp32 槽（FD_SPLITS==1 时 1 元素占位）
    cmp_phy: tla.Tensor,  # F2：cmp 段物理地址表 [T*cmp_phy_width] int64（=0 时 1 元素占位）
    # E7：host 的 unit 规划表（int32，FD_TAB_STRIDE 项/unit）。FD_PLAN==0 时
    # 1 元素占位；FD_PLAN==1 时 unit 的 task/tile 区间/ws 槽一律从表里读。
    fd_tab: tla.Tensor,
    # E13：task 参数表（int32，TP_STRIDE 项/task）。TP_ON==0 时 1 元素占位。
    tp: tla.Tensor,
    kv_zero: tla.Tensor,  # F19：常零 GM 源（[KV_BLOCK, D]，只读；KV tail 清零）
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
    # E3：1 = 相邻两个 token 合成一个 task（m=2*head_num）。host 只在
    # TND/mask3/非 PA/g_split==1/head_num==32 且每个 seq token 数为偶数时置 1。
    TOK2: tla.Constexpr[int],
    # E4：FlashDecoding KV 分片数（1 = 关闭，路径与改前逐字节一致）。
    FD_SPLITS: tla.Constexpr[int],
    # E4：归并的 D 维分摊数（每项 512/FD_MRG 列；1 = 单核整宽）。
    FD_MRG: tla.Constexpr[int],
    # E7：1 = unit 由 host 规划表驱动（task / [tile_start,tile_end) / ws 槽从
    # fd_tab 读；FD_SPLITS 此时表示「每 task 的最大 unit 数」= 归并展开宽度，
    # 也是 ws 槽布局 task*FD_SPLITS+slot 的步长）。0 = E6 的 (task,split) 等分。
    FD_PLAN: tla.Constexpr[int],
    # E9：1 = 锚点规划（表 field4 = 1 标在 anchor 段）。anchor 段的未归一化 O
    # 留在 ub_o 不落盘；屏障后由 anchor 所在核自己做归并（读其余段的部分量）。
    FD_ANCHOR: tla.Constexpr[int],
    # E9（实验插桩，默认 1/1/1 = 与 base 等价）：仅用于成本分解臂，
    # 关掉对应动作后输出不再正确，只做计时。
    FD_STORE: tla.Constexpr[int],
    FD_MERGE: tla.Constexpr[int],
    FD_GUARD: tla.Constexpr[int],
    # E10：1 = 归并统计段跨行向量化（整 VL 覆盖 rows 行，tail mask 屏蔽）。
    # host 只在 tasks <= cores（归并时存在空闲 AIV）的 FD 例置 1；#022 等
    # tasks > cores 的例保持 0，旧指令流不变。
    FD_MRG_VEC: tla.Constexpr[int],
    # E10：1 = 使用 host 归并项表（可变 MRG：单 split task 整宽、双 split
    # task D 半分），项表附在 fd_tab 的 unit 表之后，项数在 tiling[0]。
    FD_ITEM_TAB: tla.Constexpr[int],
    # E13：needInit 由 host 按 kernel 原式算出后以 Constexpr 下传（==0 时整段
    # 计算与分支折叠掉；语义与 compute_need_init 相同）。
    NEED_INIT: tla.Constexpr[int],
    # E13：1 = task 参数表驱动（去掉循环内 cu_q/cu_kv/cu_cmp/topk/resid 的
    # 串行标量读）。host 只在值域可打包时置 1；0 时指令流与改前一致。
    TP_ON: tla.Constexpr[int],
    # 0 = cm1; 1 = Q L2 bypass(cm4) 小T模式; 2 = Q L2 bypass(cm4) + 多token分槽预取(pre3+early_h0)。
    Q_L2_BYPASS: tla.Constexpr[int] = 0,
    SWIZ_ON: tla.Constexpr[int] = 1,
) -> None:
    dtype_in = q.ptr.dtype
    # E4：FD 是否启用（Constexpr ⇒ 所有新增指令在 ==1 时被折叠掉）。
    fd_on = FD_SPLITS > 1
    # E7：规划表驱动（FD_PLAN==0 时全部折叠，与 E6 指令流一致）。
    fd_plan = FD_PLAN == 1
    # E9：锚点规划（FD_ANCHOR==0 时全部折叠）。
    fd_anchor = FD_ANCHOR == 1
    # E9：成本分解臂开关（默认全 1，指令流与 base 一致）。
    fd_store_on = fd_on and FD_STORE == 1
    fd_merge_on = fd_on and FD_MERGE == 1
    fd_guard_on = fd_on and FD_GUARD == 1
    fd_off = FD_SPLITS == 1
    swiz_on = (SWIZ_ON == 1) and fd_off and (g_split == 1) and (TOK2 == 0)
    q_pre_on = fd_off and (g_split == 1) and (TOK2 == 0) and (NEED_INIT == 0)
    # IS_SPLIT_G：任务空间 = token × G 半区，m 为半区 G 宽度
    m = head_num // g_split
    if tla.const_expr(TOK2 == 1):
        # 两个 token 的 head 行拼成一个 M tile：token A 行 0..h-1、token B 行 h..2h-1
        m = 2 * head_num
    # AIV 两个子块行数：firstHalf=(m+1)/2、second=m-firstHalf（OT 同名量）；
    # rows 取前半（偶数 m 即 m/2，奇数 m 时第二子块写 0 行，见 bmm2_data_copy_out）。
    rows = m - m // 2
    # R37：OT goIdx = half*halfG（kvcache.h:250 IS_SPLIT_G 偏移）；奇 head_num 时
    # halfG=(h+1)/2 ≠ m，lse 行基址不能用 task*m。
    half_g = head_num // g_split if head_num % g_split == 0 else head_num // g_split + 1

    # 共用 gather 的 kernel 必须经同一入口（F45）：CSA 不实现共享槽臂，显式声明 False。
    configure_trace(g_split, False, False, KV_SLOTS, "csa")
    # R42 步一：PA 页数学的追踪期常量（仅 cmp 段走 PA 页；ori 由 AIC 直读）
    configure_pa(cmp_pa_block_size, cmp_pa_stride, topk_value_mode)

    # T1-1（r10）：非 PA 去表实例。kv_layout 是 Constexpr ⇒ 此处为 Python 折叠，
    # 不产生运行期判据。
    # r12：PA（kv_layout==2）改走内联页数学 + OT 成对落位，不再建表/读表；
    # use_vec_gather 即「走预计算物理地址表（vec / ring）」的实例门。
    use_vec_gather = (is_vec_s2phyaddr == 1) and (kv_layout != 2)
    _pa_pow2 = (cmp_pa_block_size > 0) and (
        (cmp_pa_block_size & (cmp_pa_block_size - 1)) == 0
    )
    pa_l1_s3 = (
        kv_layout == 2
        and _pa_pow2
        and not use_vec_gather
        and TOK2 == 0
        and FD_SPLITS == 1
        and g_split == 1
        and (ori_mask_mode == 3 or ori_mask_mode == 4)
    )
    idx_ring_on = (use_vec_gather and kv_layout != 2) or pa_l1_s3
    # 步 A（本轮）：R42 的 AIV 直写 L1 从 cmp-only 扩展到 **ori 稠密段**，并按
    # 竞品 N64 做法让 AIC 完全不搬 KV（只做 L1->L0）。启用范围 = 本组输的
    # N1<=64 非 PA CSA（g_split==1 且 m>1：021/022/026/028）；#023/#027
    # （g_split==2）、#025（PA）、#024（m=1，已优于 target）保持 AIC 直读
    # （kv_ring_on=False ⇒ 指令流回到步二形态）。
    # 步三（cmp-only）证据见 /home/l50064451/smla_probe/results/ot_src_r42/step3/。
    # 步 A 实测（device 3，022，3x10 calls）：**判负，默认关闭**。
    #   ori 也走 AIV 后 AIC MTE2 从 182.7 降到 1.0（Q-only，目标达成），
    #   AIC MTE1/MAC 也从 180.9/170.1 降到 165.3/166.6；但 AIV 的 ND->NZ
    #   成为新墙：软件 VF 版 aiv_vec=275.1、Dur=494.4（+126%）；改用
    #   ("ub","RowMajor","l1","zN") 硬件转换版 aiv_mte3=352.1、aic_mte1=403、
    #   Dur=638.1。两者都远差于步二 218.6。故 KV_L1_DIRECT 回退为 False，
    #   保持步二可用状态；实现与证据保留在
    #   /home/l50064451/smla_probe/results/ot_src_allv/stepA/{vf_fail,mte3_fail}/。
    KV_L1_DIRECT = False
    kv_ring_on = (
        KV_L1_DIRECT
        and idx_ring_on
        and is_vec_s2phyaddr == 1
        and ori_pa_block_size == 0
        and g_split == 1
        # 只打本组「输的」N1<=64 例（021/022/026/028）。#023/#027 是
        # g_split==2，#025 是 PA，均已在上面排除；#024 是 m=1（单 head）
        # 且当前已优于 target——AIV gather 无摊薄收益，保持 AIC 直读不动。
        and m > 1
    )
    # E17：PA cmp 段 AIV 直写 L1（extern 换形）。只作用于生产 A/B 形态
    # （PA + 内联页数学 + 与 D11 同组 constexpr + ori_mask_mode==4）；
    # #025（PA，mask3）与其余实例保持 base 指令流。与 R42 的 kv_ring 区别：
    # 仅 cmp tile 直写，ori tile 仍由 AIC `load_k_pa` 搬运；ND->zN 由 VECTOR 的
    # extern VSSTB 完成（不再用 MTE3 硬件 ND->NZ）。
    pa_l1_on = pa_l1_s3 or (
        PA_L1_ENABLE == 1
        and (
            kv_layout == 2
            and not use_vec_gather
            and TOK2 == 0
            and FD_SPLITS == 1
            and g_split == 1
            and not kv_ring_on
            and ori_mask_mode == 4
        )
    )
    tnd_l1_on = (
        kv_layout != 2
        and idx_ring_on
        and TOK2 == 0
        and FD_SPLITS == 1
        and g_split == 1
        and not kv_ring_on
        and (ori_mask_mode == 3 or ori_mask_mode == 4)
    )
    split_l1_on = tnd_l1_on or pa_l1_s3
    cmp_l1_on = pa_l1_on or tnd_l1_on
    # 两形态共用 R42 的同步对象/UB 分配/init（AIC 侧差异见站点 1/2/3/4）。
    ring_sync = kv_ring_on or cmp_l1_on

    sync = declare_sync(
        need_idx_ring=idx_ring_on,
        need_lse_war=enable_lse == 1,
        need_kv_ring=ring_sync,
        need_pa_l1=cmp_l1_on,
        need_kvz_db=split_l1_on,
        opt_cube64=True,
        opt_cube128=((m <= 32) and (g_split == 1)),
        opt_l2m=True,
        q_l2_bypass=(Q_L2_BYPASS >= 1),
        q_pre3_hint=(Q_L2_BYPASS >= 2),
    )
    # F2：=1 实例才分配 phy-addr 生成的 UB 暂存（ub_phy_idx/bt/addr）
    # F27：preload=True 打开流水所需的双槽（l1_p/l0c_s/ub_s/max/sum/dm——OT
    # l1PBuffers[2] / bmm1Buffers[2] / softmax*Bufs[2] 的字节表本仓已在
    # alloc_buffers 内按 preload 分支声明）。
    bufs = alloc_buffers(
        dtype_in,
        phy=0 if tnd_l1_on else (cmp_phy_width if use_vec_gather else 0),
        phy_bt=0 if tnd_l1_on else (cmp_pa_max_blocks if use_vec_gather else 0),
        preload=True,
        idx_ring=idx_ring_on,
        kv_ring=ring_sync,
        pa_l1=cmp_l1_on,
        alias_o16=split_l1_on,
        kvz_db=split_l1_on,
        fd_splits=FD_SPLITS,
        opt_cube64=True,
        opt_cube128=((m <= 32) and (g_split == 1)),
        opt_vec=True,
    )
    init_sync(
        sync,
        pre_phy_vs=use_vec_gather and not idx_ring_on,
        pre_l0c2=True,
        pre_idx_ring=idx_ring_on,
        pre_kv_ring=ring_sync,
    )
    if ring_sync:
        # R42：ND/zN 的 WAR 首拍信用（kv_free 走"后继条件"发送，不预置）。
        init_sync_kv_ring(sync, pa_l1=cmp_l1_on, kvz_db=split_l1_on)
    init_sync_preload(sync)
    # F4（OT csa_kernel_arch35.h:165 CleanOutput）：Init 阶段整张输出清零
    need_init = NEED_INIT  # E13：host 同式预算（Constexpr）
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
    # E13：need_init==0 时没有任何核做 init 清零，入口全核屏障无保护对象
    # （屏障只为「清零 vs 其它核写 O」的跨核可见性），故按 need_init 折叠。
    if need_init == 1:
        with tla.cube():
            sync_all_cores_aic()
        with tla.vector():
            sync_all_cores_aiv()

    total_tasks = cu_q[num_seqs] * g_split
    if tla.const_expr(TOK2 == 1):
        # 每个 task 覆盖两个 token（host 已保证每个 seq token 数为偶数）
        total_tasks = total_tasks // 2
    if tla.const_expr(fd_on):
        # E4：扁平流改按「执行单元」推进 = (task, split)，task 内部 tile 区间
        # 由 split 切分（见步头）。FD_SPLITS==1 时本式折叠掉。
        # E7：规划模式下单元数由 host 表长度给出（tiling[2]），不再是
        # task×FD_SPLITS；若 table 里仍有空区间，循环照常逐 unit 走。
        if tla.const_expr(fd_plan):
            total_tasks = tiling[2]
        else:
            total_tasks = total_tasks * FD_SPLITS
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
    r1_step = tla.as_numeric(0)
    r1_j = tla.as_numeric(0)
    r1_nt = tla.as_numeric(1)
    r1_nvalid = tla.as_numeric(0)
    r1_nload = tla.as_numeric(0)
    r1_kvrow0 = tla.as_numeric(0)
    r1_iscmp = tla.as_numeric(0)
    r1_seq = tla.as_numeric(0)
    r1_start = tla.as_numeric(0)
    r2_task = tla.as_numeric(0)
    r2_step = tla.as_numeric(0)
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
    # E3：第二个 token（行 rows..m-1）的每 tile 有效列数。TOK2==0 实例不使用
    # （常量赋值无指令，DCE 后指令流不变）；必须无条件预初始化——DSL 前端要求
    # 动态 if 内赋值的变量在使用前已定义。
    r0_nvalid_b = tla.as_numeric(0)
    r1_nvalid_b = tla.as_numeric(0)
    r2_nvalid_b = tla.as_numeric(0)
    r3_nvalid_b = tla.as_numeric(0)
    # E4：单元号（Q 槽/预取调度的流序号）与单元内首 tile 的全局 tile 号。
    # FD_SPLITS==1 实例的赋值与轮转都在 const_expr 分支里折叠掉，指令流不变。
    r0_u = tla.as_numeric(0)
    r1_u = tla.as_numeric(0)
    r2_u = tla.as_numeric(0)
    r3_u = tla.as_numeric(0)
    # E7：单元的部分量落盘槽号（规划模式 = task*FD_SPLITS+slot，legacy = unit）。
    r0_ws = tla.as_numeric(0)
    r1_ws = tla.as_numeric(0)
    r2_ws = tla.as_numeric(0)
    r3_ws = tla.as_numeric(0)
    r0_lo = tla.as_numeric(0)
    r1_lo = tla.as_numeric(0)
    r2_lo = tla.as_numeric(0)
    r3_lo = tla.as_numeric(0)
    # E9：锚点标志不入 RunInfo 环 —— 落盘站点直接用 r3_u（环里已有）从规划表
    # field4 读，避免新增 loop-carried 标量（FD_ANCHOR==0 时该表达式整体折叠，
    # 非锚点实例的指令流与 base 逐条一致）。

    # 本核已消费的真实 tile 数：进入某个 task 时 = 该 task 的全局前缀 g_k，
    # 任务循环退出后 = 本核的 g_total（尾排空的 x 起点）。O(1) 累加，
    # 不做 OT 之外的前缀重扫。
    g_done = tla.as_numeric(0)
    # E6：单元切换的 WAR 闸门标志（上一单元 fd_store 的 MTE3 已发出 → 下个
    # body 顶部等它读完，再允许本单元的 vec1/vec2 重写 ub_o/统计槽）。
    # 全部放在 const_expr(fd_on) 内：FD_SPLITS==1 实例连标量初始化都不生成，
    # 保证 FD 关闭时 MLIR 与 base 一致（E4 的「指令流不变」约束）。
    # E6：Q 信用链是否已启动（本核首个真实 tile 的 Q 已由站点 1 的 xg1==0
    # 路径装入）。启动后，空的 FD 单元（n_tiles==0）也必须完成一次
    # 「消费 q_l1_ready + 归还 q_l0a_free + 预取下一单元 Q」的链，否则信用
    # 逐空单元泄漏，3 次后 load_q 永久阻塞（E4 BN=1 顺序 25~30 unit 的挂死形态）。
    q_chain = tla.as_numeric(0)
    # Q 信用链用的标量（前端要求：跨动态 if 使用前先定义在循环外）。
    qc_nxt_u = tla.as_numeric(0)
    qc_nq = tla.as_numeric(0)
    qc_qa = tla.as_numeric(0)
    qc_qb = tla.as_numeric(0)
    # 暂存区按核分片（OT: v0ResGm 来自 workspace[blockIdx * perBlockSize]）。
    # 不分片时 28 个 block 会同时往同一组槽里写，AIC 读到的行由时序决定。
    ws_base = tla.arch.block_idx() * (KV_SLOTS * KV_STAGE_STRIDE)

    # 本核的 task 步数（ceil）+1 个哨兵步：哨兵步 = OT 的 PRELOAD_NUM 尾排空。
    task_start = task0
    n_steps = tla.as_numeric(0)
    if tla.const_expr(swiz_on):
        lcore = task0
        base_per = total_tasks // q_bn
        rem_t = total_tasks - base_per * q_bn
        if lcore < rem_t:
            n_steps = base_per + 1
            task_start = lcore * (base_per + 1)
        else:
            n_steps = base_per
            task_start = rem_t * (base_per + 1) + (lcore - rem_t) * base_per
    else:
        n_steps = (total_tasks - task0 + q_bn - 1) // q_bn
        if n_steps < 0:
            n_steps = tla.as_numeric(0)

    for step in tla.range(n_steps + 1):
        if tla.const_expr(swiz_on):
            task = task_start + step
        else:
            task = task0 + step * q_bn
        # E4：扁平流中的「单元号」固定为 unit；真实 task（token / pair）由它派生。
        # FD_SPLITS==1 时 task 保持原值、fd_split 恒 0，指令流与改前逐字节一致。
        unit = task
        fd_split = tla.as_numeric(0)
        # E7：规划表读出的本 unit 参数（哨兵步不读，默认值足够——无 body）。
        plan_lo = tla.as_numeric(0)
        plan_hi = tla.as_numeric(0)
        plan_ws = tla.as_numeric(0)
        if tla.const_expr(fd_on):
            if tla.const_expr(fd_plan):
                if step < n_steps:
                    plan_lo = fd_tab[unit * FD_TAB_STRIDE + 1]
                    plan_hi = fd_tab[unit * FD_TAB_STRIDE + 2]
                    plan_ws = fd_tab[unit * FD_TAB_STRIDE + 3]
                    task = fd_tab[unit * FD_TAB_STRIDE]
            else:
                task = unit // FD_SPLITS
                fd_split = unit - task * FD_SPLITS

        # 任务级标量先给默认值：哨兵步（step == n_steps）跳过任务头，n_tiles 归 0，
        # 站点判据退化为 drain 形态；站点 1/2/3 的参数一律取自 RunInfo 环，故任务级
        # 标量取默认值不影响排空（站点 0 与 Q 预取在哨兵步按判据不执行）。
        n_tiles = tla.as_numeric(0)
        j_lo = tla.as_numeric(0)
        j_hi = tla.as_numeric(0)
        fd_per = tla.as_numeric(0)
        seq = tla.as_numeric(0)
        tok = tla.as_numeric(0)
        rowq = tla.as_numeric(0)
        start_idx = tla.as_numeric(0)
        vis = tla.as_numeric(0)
        vis_b = tla.as_numeric(0)
        kv_len = tla.as_numeric(0)
        n_ori_tiles = tla.as_numeric(0)
        cmp_lines = tla.as_numeric(0)
        cmp_lines_b = tla.as_numeric(0)
        kv_row0 = tla.as_numeric(0)
        cmp_row0 = tla.as_numeric(0)

        if step < n_steps:
            # ---- 定位 (seq, row)；稀疏索引按 token 行 tok = task // g_split ----
            tok = task // g_split
            if tla.const_expr(TOK2 == 1):
                # E3：task t 覆盖 token 2t（A）与 2t+1（B）；tok 取 A 的全局行号
                tok = tok * 2
            if tla.const_expr(TP_ON == 1):
                # E13：task 参数表（host 按本 kernel 原式逐 token 算好并打包）：
                # 去掉 seq 扫描与 cu_q/cu_kv/cu_cmp/topk/resid 的串行 GM 标量读。
                tp_base = task * 8
                tp_w0 = tp[tp_base]
                tp_w1 = tp[tp_base + 1]
                tp_w2 = tp[tp_base + 2]
                tp_w3 = tp[tp_base + 3]
                vis = tp_w0 - (tp_w0 // 65536) * 65536
                vis_b = tp_w0 // 65536
                cmp_lines = tp_w1 - (tp_w1 // 65536) * 65536
                cmp_lines_b = tp_w1 // 65536
                n_ori_tiles = tp_w2 - (tp_w2 // 65536) * 65536
                seq = tp_w2 // 65536
                start_idx = tp_w3 - (tp_w3 // 65536) * 65536
                kv_len = tp_w3 // 65536
                kv_row0 = tp[tp_base + 4]
                cmp_row0 = tp[tp_base + 5]
            else:
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
                if tla.const_expr(TOK2 == 1):
                    # 第二个 token（rowq+1）：mask3 下 pre=q_len，而 rowq+1 <= q_len-1，
                    # 故 start 同为 clip(...)=0；end 多 1（pair 内 A 行号必为偶数，
                    # rowq+1=q_len 不可能），被 kv_len 裁剪时退化为相同 vis。
                    end_idx_b = rowq + 1 + next_tok + 1
                    if end_idx_b < 0:
                        end_idx_b = tla.as_numeric(0)
                    if end_idx_b > kv_len:
                        end_idx_b = kv_len
                    vis_b = end_idx_b - start_idx
                    if vis_b < 0:
                        vis_b = tla.as_numeric(0)
                    if vis_b > vis:
                        n_ori_tiles = (vis_b + KV_BLOCK - 1) // KV_BLOCK

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

                if tla.const_expr(TOK2 == 1):
                    # 第二个 token 的 cmp 行数独立计算（topk 按 token 行取）
                    sparse_count_b = topk[tok + 1]
                    if sparse_count_b > idx_width:
                        sparse_count_b = tla.as_numeric(idx_width)
                    if sparse_count_b < 0:
                        sparse_count_b = tla.as_numeric(0)
                    cmp_end_b = (rowq + 2 + next_cmp) // cmp_ratio
                    if cmp_end_b < 0:
                        cmp_end_b = tla.as_numeric(0)
                    if cmp_end_b > cmp_len:
                        cmp_end_b = cmp_len
                    cmp_lines_b = cmp_end_b
                    if cmp_lines_b > sparse_count_b:
                        cmp_lines_b = sparse_count_b
                kv_row0 = cu_kv[seq] + start_idx
                cmp_row0 = cu_cmp[seq]

            if tla.const_expr(TOK2 == 1):
                n_tiles = (
                    n_ori_tiles
                    + (cmp_lines + KV_BLOCK - 1) // KV_BLOCK
                    + (cmp_lines_b + KV_BLOCK - 1) // KV_BLOCK
                )
            else:
                n_tiles = n_ori_tiles + (cmp_lines + KV_BLOCK - 1) // KV_BLOCK

            # ---- E4：把 task 的全局 tile 区间交给本 split ----
            # 区间 = [split*ceil(T/S), min((split+1)*ceil(T/S), T))；split 可能为空
            # （n_tiles=0，不产生 body，其部分量由 host 预清零的 GM 表达：
            # max=0/sum=0/O=0，归并侧用 s_i>0 剔除，见 fd_merge_task）。
            if tla.const_expr(fd_on):
                if tla.const_expr(fd_plan):
                    # E7：区间直接取 host 规划的 [tile_start, tile_end)。host 与
                    # 本段用同一组公式（_csa_task_tiles / 本处 n_tiles）求总数，
                    # 保证 hi <= T；这里不做动态钳位（E4 已证明钳位会引入
                    # AIC/AIV 分歧）。空区间（lo==hi）沿用 E6 的空 unit 语义。
                    j_lo = plan_lo
                    j_hi = plan_hi
                    n_tiles = j_hi - j_lo
                else:
                    # 均衡、无分支的 tile 区间划分：lo=floor(s*T/S)、hi=floor((s+1)*T/S)，
                    # 各 split 不重叠且并集恰为 [0,T)，hi-lo ∈ {floor,ceil}。
                    # （动态 if 钳位赋值 n_tiles 的写法实测会引入 AIC/AIV 分歧，禁用。）
                    j_lo = (fd_split * n_tiles) // FD_SPLITS
                    j_hi = ((fd_split + 1) * n_tiles) // FD_SPLITS
                    n_tiles = j_hi - j_lo

        # ---- 本 task 的 body 数：真实 task = n_tiles；哨兵步 = PRELOAD_NUM
        #      （空核——本核一个真实 tile 都没有——不补，避免越界 load）----
        n_body = n_tiles
        if step == n_steps:
            if g_done > 0:
                n_body = n_tiles + PRELOAD_NUM

        for i in tla.range(n_body):
            x = g_done + i

            # E6 闸门不在这里：放在 fd_store 之后的同一个 body 末尾
            # （见站点 3 的 fd_unit_guard 调用），避免引入跨 body 的标志变量。

            # 槽号 / 索引环槽号在 body 顶部无条件算一次（站点 0 与站点 4 共用）。
            # 前端 _reject_unsupported_dynamic_if_new_defs 是**语法层**检查：
            # 动态 if 体内首次定义、又在其它动态 if 里使用的变量会硬失败。
            # 纯标量算术（每 body 两条 div/rem），对任何路径都无实质开销。
            slot0 = x - (x // KV_SLOTS) * KV_SLOTS
            islot = x - (x // 2) * 2

            # ================================================================
            # 站点 0（lag 0，AIV）：vec0(x) —— 稀疏 gather 落到 GM 暂存槽
            # OT ProcessVec0(runInfo[smlaRunInfoIdx])（:606），参数 = 本 body 入队的
            # 冻结值（j < n_tiles 才存在真实 tile；哨兵步不执行）
            # ================================================================
            if i < n_tiles:
                # ---- 入队：OT SetRunInfo(smla35RunInfo[taskId%4])（:600）----
                j0 = i
                if tla.const_expr(fd_on):
                    # 单元内 tile 下标 = 全局 tile 号（split 区间的起点 j_lo）
                    j0 = i + j_lo
                nv0 = tla.as_numeric(0)
                nl0 = tla.as_numeric(0)
                if tla.const_expr(TOK2 == 1):
                    # E3：tile 有效性分两半（A=行 0..rows-1、B=行 rows..m-1）。
                    # ori tile 两半共享同一 K 窗口，计数相差 1；cmp 部分拆成
                    # 独立的 A/B tile，各自只激活自己那一半。
                    nv0a = tla.as_numeric(0)
                    nv0b = tla.as_numeric(0)
                    if j0 < n_ori_tiles:
                        nv0a = vis - j0 * KV_BLOCK
                        if nv0a > KV_BLOCK:
                            nv0a = tla.as_numeric(KV_BLOCK)
                        if nv0a < 0:
                            nv0a = tla.as_numeric(0)
                        nv0b = vis_b - j0 * KV_BLOCK
                        if nv0b > KV_BLOCK:
                            nv0b = tla.as_numeric(KV_BLOCK)
                        if nv0b < 0:
                            nv0b = tla.as_numeric(0)
                        nl0 = kv_len - start_idx - j0 * KV_BLOCK
                        if nl0 < 0:
                            nl0 = tla.as_numeric(0)
                        if nl0 > KV_BLOCK:
                            nl0 = tla.as_numeric(KV_BLOCK)
                    else:
                        kk0 = j0 - n_ori_tiles
                        na_tiles0 = (cmp_lines + KV_BLOCK - 1) // KV_BLOCK
                        if kk0 < na_tiles0:
                            nv0a = cmp_lines - kk0 * KV_BLOCK
                            if nv0a > KV_BLOCK:
                                nv0a = tla.as_numeric(KV_BLOCK)
                            if nv0a < 0:
                                nv0a = tla.as_numeric(0)
                            nl0 = nv0a
                        else:
                            nv0b = cmp_lines_b - (kk0 - na_tiles0) * KV_BLOCK
                            if nv0b > KV_BLOCK:
                                nv0b = tla.as_numeric(KV_BLOCK)
                            if nv0b < 0:
                                nv0b = tla.as_numeric(0)
                            nl0 = nv0b
                    nv0 = nv0a
                    r0_nvalid_b = nv0b
                else:
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
                if tla.const_expr(fd_on):
                    # 单元的全局 tile 区间 [j_lo, j_hi)：首 tile 判据用 r*_lo，
                    # 末 tile 判据沿用 r*_nt == 区间右端（不存在的分支在 ==1 折叠）。
                    r0_u = unit
                    # E7：落盘槽号（legacy 下就是 unit = task*S+split；规划下取表）。
                    r0_ws = unit
                    if tla.const_expr(fd_plan):
                        r0_ws = plan_ws
                    r0_lo = j_lo
                    r0_nt = j_hi
                r0_nvalid = nv0
                r0_nload = nl0
                r0_kvrow0 = kv_row0
                r0_start = start_idx
                r0_seq = seq
                r0_iscmp = tla.as_numeric(0)
                if j0 >= n_ori_tiles:
                    r0_iscmp = tla.as_numeric(1)

                # vec0：仅 cmp tile 走 AIV gather（ori tile 由 AIC 直读 ori_kv）
                # R42：kv_ring 实例整段不发射——KV 的写挪到站点 4（必须在站点 3
                # 的 release 之后，零余量 3 槽的相位要求）。关闭时本条件只折叠
                # 掉站点 0 的 gather，指令流与步二逐条一致。
                if tla.const_expr(split_l1_on):
                    if r0_iscmp == 1:
                        tnd_l1_s3_stage0_async(
                            sync,
                            bufs,
                            idx,
                            tok,
                            idx_width,
                            (r0_j - n_ori_tiles) * KV_BLOCK,
                            r0_nload,
                            islot,
                        )
                if (not ring_sync) and j0 >= n_ori_tiles:
                    # 前端要求：动态分支内赋值的变量在使用前先定义
                    tok_sel = tok
                    base_line0 = 0
                    if tla.const_expr(TOK2 == 1):
                        # cmp tile 归属：前 na_tiles 块属于 token A，其余属于 B。
                        kk1 = j0 - n_ori_tiles
                        na_tiles1 = (cmp_lines + KV_BLOCK - 1) // KV_BLOCK
                        tok_sel = tok
                        base_line0 = kk1 * KV_BLOCK
                        if kk1 >= na_tiles1:
                            tok_sel = tok + 1
                            base_line0 = (kk1 - na_tiles1) * KV_BLOCK
                    else:
                        tok_sel = tok
                        base_line0 = (j0 - n_ori_tiles) * KV_BLOCK
                    # AIV 侧 gather：同一 tile 的 n_load 行按 CeilDiv(n,2) 分给两个子块
                    if kv_layout == 2 and not use_vec_gather:
                        # r12：PA 内联页数学 + OT CopyInKvSparse 成对落位
                        gather_kv_rows_pa_pairwise(
                            sync,
                            bufs,
                            ws,
                            cmp_kv,
                            idx,
                            tok_sel,
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
                            tok_sel,
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
                            tok_sel,
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
                        if tla.const_expr(swiz_on):
                            qn1 = r1_step
                        else:
                            qn1 = (r1_task - task0) // q_bn
                        if tla.const_expr(fd_on):
                            # Q 槽按单元流序号（同一 task 的多个 split 各占一组槽）
                            qn1 = (r1_u - task0) // q_bn
                        q2n1 = qn1 + qn1
                        qa1 = q2n1 - (q2n1 // 3) * 3
                        q2n1b = q2n1 + 1
                        qb1 = q2n1b - (q2n1b // 3) * 3
                        load_q(
                            sync, bufs, q, r1_task, m, dtype_in, slot_a=qa1, slot_b=qb1
                        )
                        if tla.const_expr(fd_on):
                            # E6：本核首个真实 tile 的 Q 已装入 ⇒ 后续空单元
                            # 走 Q 信用链（见 step 末尾）。
                            q_chain = tla.as_numeric(1)
                    # OT :347-348 / :357-358：两条 load 分支之前各一对屏障
                    if g_split == 2:
                        if s2_budget > 0:
                            s2_budget = s2_budget - 1
                            split_g_mte2_pair(sync)
                    if r1_iscmp == 1:
                        # R42：kv_ring 实例下 cmp tile 已由 AIV 直写 L1，AIC 不做
                        # 任何 KV 搬运（也不再有 GM 暂存槽 / wait_kv_stage /
                        # load_k_ptr）。就绪等待改在 bmm1 的 MTE1 上（wait_kv_ring）。
                        # E17：pa_l1 同样跳过 cmp 的搬运，但 ori 仍由 AIC 直读
                        # （见下面的 else 分支，未被 ring_sync 折叠）。
                        if not ring_sync:
                            # 稀疏：等两个 AIV 把行写进 GM 暂存槽（OT v0ResGm 跨核等待）
                            wait_kv_stage(sync)
                            load_k_ptr(
                                sync,
                                bufs,
                                ws.ptr + ws_base + slot_k1 * KV_STAGE_STRIDE,
                                r1_nload,
                                slot_k1,
                                dtype_in,
                                kv_zero.ptr,
                                1,
                            )
                    else:
                        # 步 A：kv_ring 实例下 ori 也由 AIV 直写 L1（站点 4），
                        # AIC 不做 MTE2 KV 搬运；就绪等待在 bmm1 的 MTE1 上。
                        if not kv_ring_on:
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
                                    kv_zero.ptr,
                                    1,
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
                                    kv_zero.ptr,
                                    1,
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
                    is_first2 = r2_j == 0
                    if tla.const_expr(fd_on):
                        is_first2 = r2_j == r2_lo
                    slot_sum_rd = slot_s2
                    if tla.const_expr(fd_on):
                        if not is_first2:
                            slot_sum_rd = 1 - slot_s2
                    else:
                        if r2_j > 0:
                            slot_sum_rd = 1 - slot_s2
                    # Q 3 槽环（闸门②）：宿主 task 的两半槽号 = 本核内 task 序号 n 的
                    # {2n, 2n+1} mod 3（与 load_q 侧同一派生式）
                    if tla.const_expr(swiz_on):
                        q_nta2 = r2_step
                    else:
                        q_nta2 = (r2_task - task0) // q_bn
                    if tla.const_expr(fd_on):
                        q_nta2 = (r2_u - task0) // q_bn
                    q_two2 = q_nta2 + q_nta2
                    qa2 = q_two2 - (q_two2 // 3) * 3
                    q_two2b = q_two2 + 1
                    qb2 = q_two2b - (q_two2b // 3) * 3
                    if is_first2:
                        # OT InitVec1SoftmaxFromSinks（任务首 tile）：只灌 max/sum，
                        # O 由本站点后一位的 vec2 fresh 覆盖启动（clear_o=0，F11）
                        goff2 = (r2_task - (r2_task // g_split) * g_split) * m
                        # E4：每个 split 的局部量不含 sinks（归并里只加一次），
                        # 故首 tile 初值走 has_sinks=0 的 (MIN_F, 0)。
                        hs2 = has_sinks
                        if tla.const_expr(fd_on):
                            hs2 = 0
                        vec_init_from_sinks(
                            sync,
                            bufs,
                            sinks,
                            r2_task,
                            m,
                            rows,
                            dtype_in,
                            goff2,
                            hs2,
                            slot_sum=slot_s2,
                            slot_max=slot_s2,
                            clear_o=0,
                            sink_reuse=1 if TOK2 == 1 else 0,
                        )
                    if tla.const_expr(TOK2 == 1):
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
                            tail_n2=r2_nvalid_b,
                        )
                    else:
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
                        is_first2,
                        dtype_in,
                        slot_s=slot_s2,
                        q_slot_a=qa2,
                        q_slot_b=qb2,
                        kv_ring=kv_ring_on,
                        iscmp=r2_iscmp,
                        pa_l1=cmp_l1_on,
                    )
                    # OT IterateBmm1Impl 末尾（:422-429）：宿主 tile 是本 task 末 tile 时，
                    # 归还 Q 两半槽信用并（notLastTwoLoop）预取"下一 task"的 Q。
                    # 目标 task 由环派生 = r2_task + q_bn（环里 x-1 恒为"宿主 tile 的
                    # 直接后继"，其 task 就是下一 task，OT CopyQGmToL1(runInfoNext)
                    # 的 runInfoNext = (taskId+3)%4 同款）。守卫 = "tile(x-1) 存在"
                    # ⇔ i < n_tiles+1：真实 body 恒真；哨兵步只有 i=0 可能命中
                    # （= OT extra=0 的 notLastTwoLoop 分支，末 task 仅 1 个 tile 时
                    # 正是它负责装该 task 的 Q），i=1/2 对应 OT 的 extra>=1 → 不发。
                    if tla.const_expr(q_pre_on):
                        if (
                            (r2_j == 0 and r2_nt >= 2)
                            if tla.const_expr(sync.opt_q_pre3)
                            else (r2_j + 2 == r2_nt)
                        ):
                            nxt_step0 = r2_step + 1
                            if nxt_step0 < n_steps:
                                nxt_t_pre = (
                                    (task_start + nxt_step0)
                                    if tla.const_expr(swiz_on)
                                    else (r2_task + q_bn)
                                )
                                qn_p0 = q_nta2 + 1
                                q2n_p0 = qn_p0 + qn_p0
                                qap0 = q2n_p0 - (q2n_p0 // 3) * 3
                                load_q_half0(
                                    sync, bufs, q, nxt_t_pre, m, dtype_in, slot_a=qap0
                                )
                    if r2_j + 1 == r2_nt:
                        if i < n_tiles + 1:
                            qn_p = q_nta2 + 1
                            q2n_p = qn_p + qn_p
                            qap = q2n_p - (q2n_p // 3) * 3
                            q2np_b = q2n_p + 1
                            qbp = q2np_b - (q2np_b // 3) * 3
                            if tla.const_expr(fd_on):
                                # 下一单元（流序号 +q_bn）的真实 task；越界不发。
                                # E7：规划模式的 task 也从表里读（预定义，避免动态
                                # if 内首次定义）。
                                nxt_u = r2_u + q_bn
                                nxt_task = tla.as_numeric(0)
                                if nxt_u < total_tasks:
                                    nxt_task = nxt_u // FD_SPLITS
                                    if tla.const_expr(fd_plan):
                                        nxt_task = fd_tab[nxt_u * FD_TAB_STRIDE]
                                    load_q(
                                        sync,
                                        bufs,
                                        q,
                                        nxt_task,
                                        m,
                                        dtype_in,
                                        slot_a=qap,
                                        slot_b=qbp,
                                    )
                            else:
                                nxt_step2 = r2_step + 1
                                nxt_t2 = (
                                    (task_start + nxt_step2)
                                    if tla.const_expr(swiz_on)
                                    else (r2_task + q_bn)
                                )
                                if tla.const_expr(q_pre_on):
                                    if r2_nt >= 2:
                                        load_q_half1(
                                            sync,
                                            bufs,
                                            q,
                                            nxt_t2,
                                            m,
                                            dtype_in,
                                            slot_b=qbp,
                                        )
                                    else:
                                        load_q(
                                            sync,
                                            bufs,
                                            q,
                                            nxt_t2,
                                            m,
                                            dtype_in,
                                            slot_a=qap,
                                            slot_b=qbp,
                                        )
                                else:
                                    load_q(
                                        sync,
                                        bufs,
                                        q,
                                        nxt_t2,
                                        m,
                                        dtype_in,
                                        slot_a=qap,
                                        slot_b=qbp,
                                    )

            # ================================================================
            if tla.const_expr(pa_l1_s3):
                if i < n_tiles:
                    if r0_iscmp == 1:
                        pa_l1_s3_stage25_full(
                            sync,
                            bufs,
                            cmp_kv,
                            kv_zero,
                            cmp_bt,
                            tok,
                            idx_width,
                            seq,
                            cmp_pa_max_blocks,
                            (r0_j - n_ori_tiles) * KV_BLOCK,
                            r0_nload,
                            islot,
                        )
            elif tla.const_expr(tnd_l1_on):
                if i < n_tiles:
                    if r0_iscmp == 1:
                        tnd_l1_s3_stage25_full(
                            sync,
                            bufs,
                            cmp_kv,
                            kv_zero,
                            tok,
                            idx_width,
                            cmp_row0,
                            (r0_j - n_ori_tiles) * KV_BLOCK,
                            r0_nload,
                            islot,
                        )
            if tla.const_expr(split_l1_on):
                if i < n_tiles:
                    if r0_iscmp == 1:
                        if x >= 3:
                            wait_kv_free(sync)
                        tnd_l1_s3_stage4a(sync, bufs, slot0, dtype_in)
                        if x >= 3:
                            if r3_j + 1 == r3_nt:
                                tnd_l1_s3_stage4b(
                                    sync, bufs, slot0, dtype_in, n_rows=r0_nload
                                )
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
                fresh_v3 = r3_j == 0
                if tla.const_expr(fd_on):
                    fresh_v3 = r3_j == r3_lo
                if tla.const_expr(fd_off):
                    iterate_bmm2_cube64(
                        sync,
                        bufs,
                        slot_k3,
                        m,
                        dtype_in,
                        slot_p=slot_p3,
                        dst_is_o=fresh_v3,
                    )
                    process_vec2_full_opt(
                        sync,
                        bufs,
                        out,
                        r3_task,
                        m,
                        rows,
                        dtype_in,
                        fresh_v3,
                        r3_j + 1 == r3_nt,
                        slot_dm=slot_p3,
                        o16_aliased=split_l1_on,
                    )
                else:
                    iterate_bmm2_cube64(
                        sync, bufs, slot_k3, m, dtype_in, slot_p=slot_p3
                    )
                    for seg in (0, 1, 2, 3):
                        process_vec2_seg(
                            sync,
                            bufs,
                            seg,
                            m,
                            rows,
                            dtype_in,
                            fresh_v3,
                            slot_dm=slot_p3,
                        )
                # 步 A：ori 与 cmp 都由 AIV 直写 L1，故后继 tile（x）存在即归还
                # `kv_free` token；与站点 4 的 AIV 消费条件（i < n_tiles）逐条对齐。
                free_tok3 = 0
                if ring_sync:
                    free_tok3 = tla.as_numeric(0)
                    if i < n_tiles:
                        free_tok3 = tla.as_numeric(1)
                        if cmp_l1_on:
                            # E17：只有后继 tile x 是 cmp 才归还 kv_free——
                            # AIV 站点 4 也只在 cmp 且 x>=3 时消费，两边条数恒等。
                            # ori tile 的 L1 槽由 AIC 自己读写，intra k_l0b_free 已覆盖。
                            free_tok3 = r0_iscmp
                release_k(sync, slot_k3, kv_ring=ring_sync, free_tok=free_tok3)
                if r3_j + 1 == r3_nt:
                    if tla.const_expr(fd_store_on):
                        # E4：本单元末 tile —— 落未归一化 O + max/sum，归并留到
                        # 全核屏障之后（split 0）。
                        fd_store_partial(
                            sync,
                            bufs,
                            ws_fd,
                            r3_ws,
                            m,
                            rows,
                            dtype_in,
                            slot_sum=slot_p3,
                            # E9：anchor 单元（field4=1）不落 O，留 ub_o 给归并。
                            store_o=(1 - fd_tab[r3_u * FD_TAB_STRIDE + 4])
                            if tla.const_expr(fd_anchor)
                            else 1,
                        )
                        # E6 闸门：紧跟在 fd_store 之后（同 body、程序序在其
                        # MTE3 解锁之后）等该次 MTE3 读完成。下一个 body 的
                        # vec1/vec2 才会重写 ub_o/统计槽；V 管道与 mutex 指令
                        # 同序，因此这一等就挡住了整段后续 V 写。
                        # 与放在「下个 body 顶部」等价，但不引入跨 body 标志
                        # 变量（FD=1 实例的 MLIR 因此只多 Q 链的标量）。
                        if tla.const_expr(fd_guard_on):
                            fd_unit_guard(sync)
                    elif tla.const_expr(fd_off):
                        if enable_lse == 1:
                            # R37：终态 max/sum 在 slot_p3；行基址 = OT goIdx
                            if tla.const_expr(TOK2 == 1):
                                # 两个 token 的 lse 行 = task*2*head_num（= task*m），
                                # 行 0..rows-1 属 token A、rows..m-1 属 token B。
                                lse_copy_out(
                                    sync,
                                    bufs,
                                    lse,
                                    r3_task,
                                    m,
                                    rows,
                                    slot=slot_p3,
                                    row_base=r3_task * m,
                                )
                            else:
                                lse_copy_out(
                                    sync,
                                    bufs,
                                    lse,
                                    r3_task,
                                    m,
                                    rows,
                                    slot=slot_p3,
                                    row_base=(r3_task // g_split) * head_num
                                    + (r3_task - (r3_task // g_split) * g_split)
                                    * half_g,
                                )

            # ================================================================
            # 站点 4（R42，AIV，位于 lag 3 之后）：cmp(x) 的 KV 直写 L1 槽
            # 相位：kv_free 的 set 来自本 body 站点 3 的 release(x-3)——同槽
            # (x%3)，零余量 3 槽成立；不得前移到站点 0（会与本站点成环）。
            # 守卫必须写：站点 3 的守卫是 x>=3，哨兵 body 也会释放真实 tile；
            # 漏掉 i<n_tiles 会为幻影 tile 消费 kv_free（静默挂死）。
            # 只有 cmp tile 写 L1；ori tile 完全不经本站点（料理由 AIC 直读）。
            # x<3 的槽本核从未用过，跳过 wait；x>=3 消费 AIC 站点 3 的 token。
            # 参数从 r0_* 重算（站点 0 的 base_line0 只在 cmp 分支里）。
            # ================================================================
            if ring_sync:
                if i < n_tiles:
                    if cmp_l1_on:
                        # E17 / TND-L1：只写 cmp tile；AIC 只在后继 tile 是 cmp 时归还
                        # kv_free，故 AIV 也只在 cmp 且 x>=3 时消费（零余量 3 槽同款）。
                        if r0_iscmp == 1:
                            if tla.const_expr(split_l1_on):
                                if x < 3:
                                    tnd_l1_s3_stage4b(
                                        sync, bufs, slot0, dtype_in, n_rows=r0_nload
                                    )
                                else:
                                    if r3_j + 1 != r3_nt:
                                        tnd_l1_s3_stage4b(
                                            sync, bufs, slot0, dtype_in, n_rows=r0_nload
                                        )
                            else:
                                if x >= 3:
                                    wait_kv_free(sync)
                                gather_kv_rows_pa_l1(
                                    sync,
                                    bufs,
                                    cmp_kv,
                                    idx,
                                    tok,
                                    idx_width,
                                    (r0_j - n_ori_tiles) * KV_BLOCK,
                                    r0_nload,
                                    slot0,
                                    dtype_in,
                                    block_table=cmp_bt,
                                    bt_seq=seq,
                                    bt_max=cmp_pa_max_blocks,
                                    pa_block_size=cmp_pa_block_size,
                                    pa_stride=cmp_pa_stride,
                                    topk_value_mode=topk_value_mode,
                                )
                    else:
                        # x<3 的槽本核从未被占用，无需等待；x>=3 消费 AIC 在站点 3
                        # 归还的 token（与 AIC 的 free_tok3 条件同源，条数恒等）。
                        if x >= 3:
                            wait_kv_free(sync)
                        if r0_iscmp == 1:
                            gather_kv_rows_l1(
                                sync,
                                bufs,
                                cmp_kv,
                                idx,
                                tok,
                                idx_width,
                                cmp_row0,
                                (r0_j - n_ori_tiles) * KV_BLOCK,
                                r0_nload,
                                slot0,
                                islot,
                                dtype_in,
                            )
                        else:
                            # 步 A：ori 稠密段同一套 AIV 环。行地址 =
                            # (r0_kvrow0 + j*KV_BLOCK + line)*HEAD_DIM，与 AIC 原
                            # load_k 的同式；无索引/页表。
                            gather_kv_rows_l1(
                                sync,
                                bufs,
                                ori_kv,
                                idx,
                                tok,
                                idx_width,
                                r0_kvrow0,
                                r0_j * KV_BLOCK,
                                r0_nload,
                                slot0,
                                islot,
                                dtype_in,
                                dense=True,
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
            if tla.const_expr(TOK2 == 1):
                rt = r2_nvalid_b
                r3_nvalid_b = rt
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
            r2_step = r1_step
            r2_j = r1_j
            r2_nt = r1_nt
            r2_nvalid = r1_nvalid
            if tla.const_expr(TOK2 == 1):
                r2_nvalid_b = r1_nvalid_b
            r2_nload = r1_nload
            r2_kvrow0 = r1_kvrow0
            r2_start = r1_start
            r2_iscmp = r1_iscmp
            r2_seq = r1_seq
            if tla.const_expr(fd_on):
                # E4：单元号/区间左端随环轮转（与 r*_task 同相位）。
                r3_u = r2_u
                r3_lo = r2_lo
                r3_ws = r2_ws
                r2_u = r1_u
                r2_lo = r1_lo
                r2_ws = r1_ws
            if i < n_tiles:
                r1_task = r0_task
                r1_step = step
                r1_j = r0_j
                r1_nt = r0_nt
                r1_nvalid = r0_nvalid
                if tla.const_expr(TOK2 == 1):
                    r1_nvalid_b = r0_nvalid_b
                r1_nload = r0_nload
                r1_kvrow0 = r0_kvrow0
                r1_start = r0_start
                r1_iscmp = r0_iscmp
                r1_seq = r0_seq
                if tla.const_expr(fd_on):
                    r1_u = r0_u
                    r1_lo = r0_lo
                    r1_ws = r0_ws

        # 本 task 结束：累计本核 tile 前缀（哨兵步 n_tiles=0，不再变化）
        g_done = g_done + n_tiles

        if tla.const_expr(fd_on):
            # E6：空执行单元（split 无 tile）的 Q 信用链。
            # 空单元没有 body ⇒ 不会消费上一单元预取的 Q（q_l1_ready 残留）、
            # 也不会归还那次 load_q 拿走的 q_l0a_free 信用，更不会为下一单元
            # 预取 Q。前两条每次泄漏 1 个信用（init 3，3 个空单元后 load_q 永久
            # 阻塞）；后一条让下一单元读到错误的 Q。这里补做同样的三件事。
            # q_chain==0（本核还没有真实 tile）时无待消费的 Q，不能消费。
            if step < n_steps:
                if n_tiles == 0:
                    if q_chain == 1:
                        with tla.cube():
                            tla.wait_flag(sync.q_l1_ready)
                            tla.set_flag(sync.q_l0a_free)
                        qc_nxt_u = unit + q_bn
                        qc_task = tla.as_numeric(0)
                        if qc_nxt_u < total_tasks:
                            qc_nq = (qc_nxt_u - task0) // q_bn
                            qc_qa = qc_nq + qc_nq
                            qc_qa = qc_qa - (qc_qa // 3) * 3
                            qc_qb = qc_nq + qc_nq + 1
                            qc_qb = qc_qb - (qc_qb // 3) * 3
                            qc_task = qc_nxt_u // FD_SPLITS
                            if tla.const_expr(fd_plan):
                                qc_task = fd_tab[qc_nxt_u * FD_TAB_STRIDE]
                            load_q(
                                sync,
                                bufs,
                                q,
                                qc_task,
                                m,
                                dtype_in,
                                slot_a=qc_qa,
                                slot_b=qc_qb,
                            )

    # ---- 预算排空（OT :674-680）：把剩余配额用等量屏障对补发，保证每核条数恒等 ----
    if g_split == 2:
        for _k in tla.range(s2_budget):
            split_g_mte2_pair(sync)

    # ================================================================
    # E4：全核屏障 + FlashDecoding 归并
    # 所有部分量（未归一化 O / max / sum）已在各核单元末落 GM。SyncAll<false>
    # 需要 AIC 与 AIV 全体参与（dav_3510 实现见 smla_csa_common 顶部注释），
    # 因此两侧各发一次；屏障之后每个 task 的 split 0 单元负责归并并写最终
    # out/lse（其余 split 不再写输出）。
    # ================================================================
    if tla.const_expr(fd_on):
        with tla.vector():
            # 本核全部部分量 MTE3 落盘完成的程序序标记：m_o16 在每次
            # fd_store_partial 里以 MTE3 加/解锁；末尾再取一次锁即等最后一个
            # MTE3 完成，之后才进全核屏障（跨核可见性）。
            sync.m_o16.lock(pipe=tla.pipes.MTE3)
            sync.m_o16.unlock(pipe=tla.pipes.MTE3)
        with tla.cube():
            sync_all_cores_aic()
        with tla.vector():
            sync_all_cores_aiv()
    if tla.const_expr(fd_merge_on):
        if tla.const_expr(fd_anchor):
            # E9 锚点模式：归并项 = 本核自己的 anchor 单元（field4=1）。它的
            # 未归一化 O 仍在本核 ub_o 里，归并只读其余段的部分量；MRG 恒 1。
            mrg_steps = (total_tasks - task0 + q_bn - 1) // q_bn
            if mrg_steps < 0:
                mrg_steps = tla.as_numeric(0)
            for mstep in tla.range(mrg_steps):
                mrg_u = task0 + mstep * q_bn
                if mrg_u < total_tasks:
                    if fd_tab[mrg_u * FD_TAB_STRIDE + 4] == 1:
                        fd_merge_task(
                            sync,
                            bufs,
                            ws_fd,
                            sinks,
                            out,
                            lse,
                            fd_tab[mrg_u * FD_TAB_STRIDE],
                            0,
                            m,
                            rows,
                            dtype_in,
                            FD_SPLITS,
                            1,
                            has_sinks,
                            enable_lse,
                            sink_reuse=1 if TOK2 == 1 else 0,
                            own_in_ub=1,
                        )
        else:
            # 归并项 = (task, part)，part 覆盖 512/FD_MRG 列；按核 round-robin
            # 分配到全部 AIC block（空核也参与，小 task 数时填满空闲核）。
            # E7：规划模式下单元数 != task*FD_SPLITS，真实 task 数由 host 经
            # tiling[3] 给出；归并仍是「屏障后按 task」执行（与 E6 同构，无单
            # unit 串行），每个 task 的 ws 槽是连续的 task*FD_SPLITS + [0,k)。
            if tla.const_expr(FD_ITEM_TAB == 1):
                # E10：host 归并项表（tiling[0] 项，附在 fd_tab 的 unit 表之后）。
                # 每项 [task, part, mode]：mode 0 = task 只有 1 个有效 split，
                # 整宽（mrg=1, splits=1，不读空槽）；mode 1 = 2 个有效 split，
                # D 半分（mrg=2, splits=2）。项数由 host 控制在 <= cores，
                # 每核至多 1 项，临界归并量减半。
                mrg_total = tiling[0]
                item_base = tiling[2] * FD_TAB_STRIDE
            else:
                if tla.const_expr(fd_plan):
                    mrg_total = tiling[3] * FD_MRG
                else:
                    mrg_total = (total_tasks // FD_SPLITS) * FD_MRG
            mrg_steps = (mrg_total - task0 + q_bn - 1) // q_bn
            if mrg_steps < 0:
                mrg_steps = tla.as_numeric(0)
            for mstep in tla.range(mrg_steps):
                mrg_item = task0 + mstep * q_bn
                if mrg_item < mrg_total:
                    if tla.const_expr(FD_ITEM_TAB == 1):
                        it = item_base + mrg_item * 4
                        mrg_task = fd_tab[it]
                        mrg_part = fd_tab[it + 1]
                        mrg_mode = fd_tab[it + 2]
                        if mrg_mode == 0:
                            fd_merge_task(
                                sync,
                                bufs,
                                ws_fd,
                                sinks,
                                out,
                                lse,
                                mrg_task,
                                mrg_part,
                                m,
                                rows,
                                dtype_in,
                                1,
                                1,
                                has_sinks,
                                enable_lse,
                                sink_reuse=1 if TOK2 == 1 else 0,
                                vec_stats=FD_MRG_VEC,
                                splits_ws=FD_SPLITS,
                            )
                        else:
                            fd_merge_task(
                                sync,
                                bufs,
                                ws_fd,
                                sinks,
                                out,
                                lse,
                                mrg_task,
                                mrg_part,
                                m,
                                rows,
                                dtype_in,
                                2,
                                2,
                                has_sinks,
                                enable_lse,
                                sink_reuse=1 if TOK2 == 1 else 0,
                                vec_stats=FD_MRG_VEC,
                                splits_ws=FD_SPLITS,
                            )
                    else:
                        mrg_task = mrg_item // FD_MRG
                        mrg_part = mrg_item - mrg_task * FD_MRG
                        fd_merge_task(
                            sync,
                            bufs,
                            ws_fd,
                            sinks,
                            out,
                            lse,
                            mrg_task,
                            mrg_part,
                            m,
                            rows,
                            dtype_in,
                            FD_SPLITS,
                            FD_MRG,
                            has_sinks,
                            enable_lse,
                            sink_reuse=1 if TOK2 == 1 else 0,
                            vec_stats=FD_MRG_VEC,
                        )
