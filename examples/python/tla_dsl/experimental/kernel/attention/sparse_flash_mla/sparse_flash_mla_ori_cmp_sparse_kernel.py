# ---------------------------------------------------------------------------
# sparse_flash_mla_ori_cmp_sparse_kernel.py — ORI_CMP_SPARSE（arch35, TND）
#
# 结构参照 ops-transformer/attention/sparse_flash_mla/op_kernel/arch35/
#   sparse_flash_mla_csa_kernel_arch35.h        -> 本文件（入口/任务循环）
#   sparse_flash_mla_csa_block_cube_arch35.h    -> smla_csa_block_cube.py
#   sparse_flash_mla_csa_block_vector_arch35.h  -> smla_csa_block_vector.py
#   sparse_flash_mla_kvcache.h                  -> 行数/区间公式
#
# 与 #022 CSA 的差别：ori 段也是稀疏索引（不是连续因果区间），故 ori tile 也走
# AIV 侧 16 行/8 行两级 gather；cmp 段与 CSA 相同。两段共用同一套 softmax 状态。
#
# PA_BBND（#045）：两段都是稀疏索引，所以两段都走 gather_kv_rows 的 PA 分支
# （block_table 按 seq 取页，phys*pa_stride + (pos%block_size)*HEAD_DIM）；
# TND 时三个 pa constexpr 均为 0，gather 退回 row_base+src 直接寻址。
#
# T1-1（r10，本变体 ot_r10_t1ocs）：把 csa/os 已采纳的「去表」机制接到 ocs。
# 非 PA 且 IS_VEC_S2PHYADDR=1 的实例不再在 prologue 生成 ori/cmp 两张 int64
# 物理地址表，改为 gather 时把索引行一次 MTE2 DMA 进 UB 2 槽环（gather_kv_rows_ring），
# 逐行标量读 (row_base+v)*HEAD_DIM。PA（kv_layout==2）仍走原表路径且 IR 不变。
# 落点与相位论证见 evidence/coord/r10_t1ocs/NOTES.md。
#
# 行数来源（sparse_flash_mla_kvcache.h）：
#   ori（:96-114 ComputeParamBatch + :378-388）：按 oriMaskMode 取 pre/next，
#        start ← 0、end ← Min(rangeLen, Min(oriTopkLength, oriIdxWidth), actualS2OriSize)
#   cmp（:86-93 + :410-420 ORI_CMP_SPARSE 分支）：
#        nextTokensPerBatchCmp ← actualS2CmpSize*cmpRatio + cmpResidual - actualS1Size
#        end ← clip((cubeSOuterOffset + s1RealSize + nextTokensPerBatchCmp)/cmpRatio,
#                   0, actualS2CmpSize)，再 Min(cmpSparseBlockCount)、Min(actualS2CmpSize)
#
# ==== F27：扁平 4 深流水（PRELOAD_NUM=3）====
# 与 CSA / OS kernel 同一套结构（OT 三个 family 共用 csa_kernel_arch35.h 的
# ProcessMainLoop :466-700）：(bnIdx, gS1Index, s2LoopCount) 展平为单一 taskId
# 递增流，每个 s2 迭代体按滞后发射 5 个站点：
#   vec0(t)（lag 0，AIV gather，ori/cmp 两源）→ load(t-1)（lag 1，AIC MTE2）
#   → vec1(t-2)+bmm1(t-2)（lag 2）→ bmm2(t-3)+vec2(t-3)（lag 3）
# 站点参数一律从深度 4 的 RunInfo 环（OT smla35RunInfo[4]）取，入队时刻冻结；
# 尾排空由哨兵步补 PRELOAD_NUM 个 body（OT :557 gS1LoopEnd += PRELOAD_NUM）。
# 全部 tile 都走 AIV gather ⇒ 环里不需要 csa 的 is_cmp / seq / kv_row0 三个字段
# （段归属与 gather 参数在 owner 自己的 body 里由任务头标量给出），环只留
# task / j / nt / n_valid / n_load 五个标量。
# 逐行对照表与不变量见 ITERATIONS.md F27。
# ---------------------------------------------------------------------------

from __future__ import annotations

import catlass.tla as tla

import smla_csa_block_vector as _bv  # 追踪期共享钳制配置

from smla_csa_block_cube import (
    iterate_bmm1,
    iterate_bmm2_fixpipe,
    iterate_bmm2_seg,
    load_k_ptr,
    load_q,
    pair_rendezvous,
    release_k,
    split_g_mte2_pair,
    wait_kv_stage,
    wait_p_ready,
)
from smla_csa_block_vector import (
    clean_output,
    bmm2_data_copy_out,
    lse_copy_out,
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
WS_BLOCKS = 32  # 暂存区按核分片数（OT v0ResGm = workspace[blockIdx*perBlock]）

# OT PRELOAD_NUM（csa_kernel_arch35.h:557/:564 的尾排空深度）
PRELOAD_NUM = 3

# 非 PA 稀疏地址走索引环（r10 T1-1 已采纳，与 csa / os kernel 同口径）：省逐行
# GM 表读 + prologue 整张物理地址表；PA（kv_layout==2）仍走原表路径，其 IR 与改前
# 逐条相同（PA 控制臂保持 kernel.o 对象级不变）。
# 注意：ring 分支落在 `is_vec_s2phyaddr == 1` 内侧（与 csa/os 同款门控），
# 因此 is_vec_s2phyaddr != 1 的实例完全不走 ring——它们走的是下面
# `else: gather_kv_rows(...)` 的原始逐行索引路径。
# PA 例走内联页数学 + 成对落位（R35 已采纳），不再建表。


@tla.kernel
def sparse_flash_mla_ori_cmp_sparse_kernel(
    q: tla.Tensor,  # [T*H, D]  TND 展平
    ori_kv: tla.Tensor,  # [KV, D]；PA_BBND 时展平为 (blockNum*blockSize*n2, D)
    cmp_kv: tla.Tensor,  # [CMP, D]；同上
    ori_bt: tla.Tensor,  # [B, maxBlocks] int32（非 PA 时传 1 元素占位）
    cmp_bt: tla.Tensor,  # [B, maxBlocks] int32（非 PA 时传 1 元素占位）
    ori_idx: tla.Tensor,  # [T*ori_idx_width] int32 展平
    ori_topk: tla.Tensor,  # [T] int32
    cmp_idx: tla.Tensor,  # [T*cmp_idx_width] int32 展平
    cmp_topk: tla.Tensor,  # [T] int32
    cu_q: tla.Tensor,  # [B+1] int32
    cu_kv: tla.Tensor,  # [B+1] int32
    cu_cmp: tla.Tensor,  # [B+1] int32
    resid: tla.Tensor,  # [B] int32
    sinks: tla.Tensor,  # [H] fp32
    out: tla.Tensor,  # [T*H, D]
    lse: tla.Tensor,  # [T*H] fp32 softmax_lse（enable_lse=0 时传 1 元素占位）
    tiling: tla.Tensor,  # [4] int32；F27：tiling[1] = IS_SPLIT_G 的 MTE2 屏障配额
    # （OT FA_S2_MAX_NUM，host 计算；g_split==1 时不读）
    ws: tla.Tensor,  # WS_BLOCKS * KV_SLOTS * 128 * 512
    ori_phy: tla.Tensor,  # F2：ori 段物理地址表 [T*ori_phy_width] int64（=0 时 1 元素占位）
    cmp_phy: tla.Tensor,  # F2：cmp 段物理地址表 [T*cmp_phy_width] int64（=0 时 1 元素占位）
    softmax_scale: tla.Float32,
    head_num: tla.Constexpr[int],
    head_dim: tla.Constexpr[int],
    num_seqs: tla.Constexpr[int],
    ori_idx_width: tla.Constexpr[int],
    cmp_idx_width: tla.Constexpr[int],
    ori_mask_mode: tla.Constexpr[int],
    ori_win_left: tla.Constexpr[int],
    ori_win_right: tla.Constexpr[int],
    cmp_mask_mode: tla.Constexpr[int],
    cmp_ratio: tla.Constexpr[int],
    # PA_BBND（layout_kv = PA_BBND 时 >0，TND 时全 0）：按 OT GetkeyOffset 的 PA 分支寻址
    #   phys ← block_table[seq*max_blocks + logical_pos // block_size]
    #   addr ← phys*pa_stride + (logical_pos % block_size)*HEAD_DIM
    ori_pa_block_size: tla.Constexpr[int],
    ori_pa_stride: tla.Constexpr[int],
    ori_pa_max_blocks: tla.Constexpr[int],
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
    ori_phy_width: tla.Constexpr[int],  # Align128(ori_idx_width)，表行宽（int64 元素）
    cmp_phy_width: tla.Constexpr[int],  # Align128(cmp_idx_width)，表行宽（int64 元素）
) -> None:
    dtype_in = q.ptr.dtype
    # IS_SPLIT_G（OT: tiling splitG = gSize > 64，common/smla_kernel_common_arch35.h:273-277）：
    # 一个 token 的 G 轴按 AIC 的 aicIdx%2 切成两半，halfG = ceil(gSize/2)，
    # 每半 mRealSize = s1RealSize * gSplitSize。任务空间因此是「token × G 半区」，
    # q / out / sinks 的展平行号 = task*m 仍成立（[T*H, D] 按 m 行一块连续）。
    m = head_num // g_split
    # AIV 两个子块行数：firstHalf=(m+1)/2、second=m-firstHalf（OT 同名量）；
    # rows 取前半（偶数 m 即 m/2，奇数 m 时第二子块写 0 行，见 bmm2_data_copy_out）。
    rows = m - m // 2
    # R37：OT goIdx = half*halfG（奇 head_num 时 halfG ≠ m），lse 行基址用
    half_g = head_num // g_split if head_num % g_split == 0 else head_num // g_split + 1

    # ---- 配对语义的**唯一**入口（F45 事故后的结构性收敛）----
    # `_PAIRED` 是本函数唯一的「类选择」读；共享臂的策略在这里折叠一次：
    # 模块开关 = 「愿意在配对时启用」，不是「无条件启用」。
    # configure_trace 仍保留一条绊线：任何调用方若把模块开关**原值**直接传进来
    # 而 g_split=1，会立刻编译失败（正是 F45 事故的形态）；gather 侧另有
    # require_share / verify_trace_closed 两道断言。注意 g_split 仍作为**数值**
    # 参与 task 映射（task // g_split 等），那类用法不是类选择，保留原样。
    _PAIRED = g_split == 2
    _PAIRED = _bv.configure_trace(g_split, _PAIRED, _PAIRED, KV_SLOTS, "ocs")

    # T1-1（r10）：非 PA 去表实例。kv_layout 是 Constexpr ⇒ 此处为 Python 折叠，
    # 不产生运行期判据。
    # r12：PA（kv_layout==2）改走内联页数学 + OT 成对落位，不再建表/读表
    use_vec_gather = (is_vec_s2phyaddr == 1) and (kv_layout != 2)
    idx_ring_on = use_vec_gather and kv_layout != 2

    # 共享暂存实例才声明真会合 flag（未用 flag 也会改变追踪结果，见 common 注）
    sync = declare_sync(
        need_rendezvous=_bv._TRACE_STAGE,
        need_idx_ring=idx_ring_on,
        need_lse_war=enable_lse == 1,
    )
    # F2：=1 实例才分配 phy-addr 生成的 UB 暂存；容量取 ori/cmp 两段的较大者
    # F27：preload=True 打开流水所需的双槽（l1_p/l0c_s/ub_s/max/sum/dm）。
    phy_cap = ori_phy_width if ori_phy_width >= cmp_phy_width else cmp_phy_width
    bt_cap = (
        ori_pa_max_blocks
        if ori_pa_max_blocks >= cmp_pa_max_blocks
        else cmp_pa_max_blocks
    )
    bufs = alloc_buffers(
        dtype_in,
        phy=phy_cap if use_vec_gather else 0,
        phy_bt=bt_cap if use_vec_gather else 0,
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
    # F4（OT csa_kernel_arch35.h:165 CleanOutput）：ORI_CMP_SPARSE 恒 needInit=1
    need_init = compute_need_init(cu_q, cu_kv, num_seqs, ori_mask_mode, 1)
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
    # 正控已逐元素复现、且加屏障后正控不再生效（ITERATIONS.md §4.12）。
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

    # ---- F29（路径 A，OT `GetKVPhyAddr` 保真）：prologue 表 Init + 一次块屏障 ----
    # OT 在**主循环之前**、`ASCEND_IS_AIV` 下按本核窗口调一次
    # `GetKVPhyAddr(hasLoad, bN2StartIdx, bN2EndIdx, gS1StartIdx, nextGs1Idx, …)`
    # （csa_kernel_arch35.h:177-192），函数内部再按 `vecCoreNum = IS_SPLIT_G?4:2`、
    # `smlaVecCoreIdx = aivIdx % 4` 把本窗口的 valid 行**均分给各 AIV**
    # （block_vector_arch35.h:1636-1700），随后一次 SyncAll；主循环只消费。
    # 我方 D1 任务划分下一个 token 会被 2 个 G 半区（block k/k+1）× 2 个子块
    # 各生成一遍（冗余 ≈ 4，见 ITERATIONS.md §14.1）：这里改成
    # 「全部 token 平铺滑片给各 AIV、每行恰好一个写者」+ 一次
    # `sync_all_cores_aiv()`，任务循环里的生成整段删除。
    # 滑片写法 = OT `InitOutputSingleCore` 同款（本变体 `init_output_single_core`
    # 已逐位验证）：`vec_core_num = block_num*2`、`vec_idx = block_idx*2 + sub_block`、
    # `single = ceil(total/vec_core_num)`、区间 `[vec_idx*single, +single)`。
    # 与 OT 的唯一口径差异：OT 按 valid 行数均分（跳过 lines==0 的 token），
    # 我方按 token 平铺——每行仍只有一个写者、内容不变，差别只在 Init 负载分布。
    # 尺寸（ITERATIONS.md §15.2）：表在宿主侧单独分配、按全体 token 足额，
    # 最大 #037 4 MiB，无需分段 Init。
    if is_vec_s2phyaddr == 1:
        # 下面的 SyncAll 无条件保留（=1 实例 AIC/AIV 各恰好一次，
        # #046 正控验证过，删一侧会挂死）。
        # OT `Process()` :443-449 的 `SyncAll<false>()`（注释原文「SyncAll
        # Cube和Vector都需要调用」）：**混合屏障，AIC 与 AIV 两侧都必须调用、
        # 各恰好一次**。我方用 F27 入口屏障同一对（`SyncAll<false>`），
        # 也是本 kernel 里已被 #046 正控验证过的唯一一处块屏障形态。
        # 反例（本轮实测）：只调向量侧的 `SyncAll<true>`（AIV-only）→ #047
        # 60 s 挂死；只加一侧或加两次同样挂死而非错值。
        with tla.cube():
            sync_all_cores_aic()
        with tla.vector():
            sync_all_cores_aiv()

    # ---- F27：IS_SPLIT_G 的 MTE2 屏障配额（OT FA_S2_MAX_NUM）----
    # 与 csa / os kernel 同一套"预算排空"：真实 load 与跳过 load 各消费 1，末尾
    # 把剩余配额用等量 set+wait 排空（OT :637-638 / :645-648 / :674-680），
    # 每核恰好发出配额条；条数恒等只依赖"所有核读到同一个配额值"。
    s2_budget = tla.as_numeric(0)
    c_one = 1
    if _PAIRED:
        s2_budget = tiling[c_one]

    # ---- F27 RunInfo 环：OT smla35RunInfo[4]，深度 4 ----
    # 每站参数在入队时刻冻结进 r0（= OT SetRunInfo(smla35RunInfo[taskId%4])）；
    # body 末尾 r3<-r2、r2<-r1，真实 tile 才再覆盖 r1<-r0（哨兵步不覆盖，
    # 对应 OT (taskId+k)%4 的陈旧槽语义）。每槽 5 个标量：
    # task / j（task 内 tile 号）/ nt（task 内 tile 数）/ n_valid / n_load。
    r0_task = tla.as_numeric(0)
    r0_j = tla.as_numeric(0)
    r0_nt = tla.as_numeric(1)
    r0_nvalid = tla.as_numeric(0)
    r0_nload = tla.as_numeric(0)
    r1_task = tla.as_numeric(0)
    r1_j = tla.as_numeric(0)
    r1_nt = tla.as_numeric(1)
    r1_nvalid = tla.as_numeric(0)
    r1_nload = tla.as_numeric(0)
    r2_task = tla.as_numeric(0)
    r2_j = tla.as_numeric(0)
    r2_nt = tla.as_numeric(1)
    r2_nvalid = tla.as_numeric(0)
    r2_nload = tla.as_numeric(0)
    r3_task = tla.as_numeric(0)
    r3_j = tla.as_numeric(0)
    r3_nt = tla.as_numeric(1)
    r3_nvalid = tla.as_numeric(0)
    r3_nload = tla.as_numeric(0)

    # 本核已消费的真实 tile 数（O(1) 累加）：进入某 task 时 = 其全局前缀，
    # 任务循环退出后 = 本核 g_total（尾排空的 x 起点）。
    g_done = tla.as_numeric(0)
    ws_base = tla.arch.block_idx() * (KV_SLOTS * KV_STAGE_STRIDE)
    # A 步一（共享钳制已采纳）：两个 G 半区（相邻 block）共用一份暂存，
    # 对应 OT `v0ResGm` 的 `totalOffset = v0ResSize*3*(aicIdx>>1)`（csa_kernel.h:429-431）。
    # 只有 g_split==2 时相邻 block 才处理同一 token，g_split==1 不配对。
    #
    # 共享槽环深度取 2×KV_SLOTS：配对核共用一块区域后，槽回收的**保守距离**由
    # 「本核 AIV 领先本核 AIC 的 2 个 tile + 配对错位 ~1 个 tile」决定，3 槽贴边。
    # 空间不变（配对后区域数减半：14×6 == 28×3），OT 的 3 槽是其 AIV 受
    # `WaitCrossCore` 逐轮约束后才够用的深度；我方 AIV 没有这条逐轮约束。
    # 纯 Python 常量（不放进动态 if：编译期绑定不能在动态作用域里赋值）
    # **环深也必须按 g_split==2 门控**：g_split=1 时 ws_base 仍是「每核 3 槽」，
    # 若环深取 6，槽 3..5 会落进下一核的区域 ⇒ 跨核别名（F45 事故：#049 bit 0.9436）。
    # 环深与配对起点都取自 configure_trace 的折叠结果：slots 在非配对实例恒为
    # KV_SLOTS（否则槽 3..5 会落进下一核区域，F45《#049 bit 0.9436》）。
    SHARE_SLOTS = _bv._TRACE_SLOTS
    if _bv._TRACE_STAGE:
        ws_base = (tla.arch.block_idx() // 2) * (SHARE_SLOTS * KV_STAGE_STRIDE)

    # 本核的 task 步数（ceil）+1 个哨兵步：哨兵步 = OT 的 PRELOAD_NUM 尾排空。
    n_steps = (total_tasks - task0 + q_bn - 1) // q_bn
    if n_steps < 0:
        n_steps = tla.as_numeric(0)

    # ---- 共享暂存（A 步二）：跨块会合的配额必须按**全局**步数算 ----
    # OT 的暂存保护是 `v0ResGm.WaitCrossCore()`（每个暂存槽一条独立 event id 的
    # 跨核信用），而 `CrossCoreSetFlag<0>(15)` 在**每次真实 load** 上都无条件发
    # （cube.h:249-253），配额 `qsmlaMaxS2LoopCnt` 只用于跳过分支与尾部排空。
    # 我方现在的形态把会合放进 `s2_budget > 0`：预算初值 = tiling[1] = 2，只在
    # 核启动时初始化一次 ⇒ 整个核只有前 2 次 load 有跨块会合，其余 tile 全靠
    # 6 槽环深度兜，实测 40% 行不逐位（读到配对核上一轮的旧象限，输出全有限、
    # 无 NaN —— 见 evidence/diag/ws_nan_f44n047.npz 的全覆盖哨兵）。
    # 修法：预算 = tiling[1] × **全局**步数（每核 2 × 饱和步数，条数恒等，mode-0
    # 全核屏障才不会错位），于是每个真实 load 都取到一次会合。
    # 不能用本核 n_steps（各核 9/10 不等 → 条数不等 → 全核屏障错配）。
    if _bv._TRACE_STAGE:
        n_steps_g = (total_tasks + q_bn - 1) // q_bn
        if n_steps_g < 0:
            n_steps_g = tla.as_numeric(0)
        if _PAIRED:
            s2_budget = tiling[c_one] * n_steps_g

    for step in tla.range(n_steps + 1):
        task = task0 + step * q_bn

        # 任务级标量先给默认值：哨兵步跳过任务头，n_tiles 归 0，站点判据退化为
        # drain 形态；站点 1/2/3 的参数一律取自 RunInfo 环。
        n_tiles = tla.as_numeric(0)
        n_ori_tiles = tla.as_numeric(0)
        seq = tla.as_numeric(0)
        tok = tla.as_numeric(0)
        rowq = tla.as_numeric(0)
        kv_len = tla.as_numeric(0)
        ori_lines = tla.as_numeric(0)
        cmp_lines = tla.as_numeric(0)
        kv_row0 = tla.as_numeric(0)
        cmp_row0 = tla.as_numeric(0)

        if step < n_steps:
            # ---- 定位 (seq, row)；稀疏索引/topk 按 token 行（tok），不是 G 半区 ----
            tok = task // g_split
            rowq = tok
            for bb in static_range(1, num_seqs):
                if tok >= cu_q[bb]:
                    seq = bb
                    rowq = tok - cu_q[bb]
            q_len = cu_q[seq + 1] - cu_q[seq]
            kv_len = cu_kv[seq + 1] - cu_kv[seq]
            cmp_len = cu_cmp[seq + 1] - cu_cmp[seq]

            # ---- ori 可见区间（ComputeParamBatch :96-114，逐式照搬）----
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
            range_len = end_idx - start_idx
            if range_len < 0:
                range_len = tla.as_numeric(0)

            # ---- ori 行数（ORI_CMP_SPARSE 分支：start=0，end=Min(...)）----
            ori_count = ori_topk[tok]
            if ori_count > ori_idx_width:
                ori_count = tla.as_numeric(ori_idx_width)
            if ori_count < 0:
                ori_count = tla.as_numeric(0)
            ori_lines = range_len
            if ori_lines > ori_count:
                ori_lines = ori_count
            if ori_lines > kv_len:
                ori_lines = kv_len
            if ori_lines < 0:
                ori_lines = tla.as_numeric(0)
            n_ori_tiles = (ori_lines + KV_BLOCK - 1) // KV_BLOCK

            # ---- cmp 行数（:86-93 + :410-420）----
            cmp_count = cmp_topk[tok]
            if cmp_count > cmp_idx_width:
                cmp_count = tla.as_numeric(cmp_idx_width)
            if cmp_count < 0:
                cmp_count = tla.as_numeric(0)
            # nextTokensPerBatchCmp（kvcache.h:86-93）：
            #   cmpMaskMode == 0 → actualS2CmpSize * cmpRatio（不减 q_len、不加 resid）
            #   否则            → actualS2CmpSize*cmpRatio + resid - actualS1Size
            next_cmp = cmp_len * cmp_ratio  # cmpMaskMode == 0 的默认值
            if cmp_mask_mode != 0:
                next_cmp = cmp_len * cmp_ratio + resid[seq] - q_len
            cmp_end = (rowq + 1 + next_cmp) // cmp_ratio
            if cmp_end < 0:
                cmp_end = tla.as_numeric(0)
            if cmp_end > cmp_len:
                cmp_end = cmp_len
            cmp_lines = cmp_end
            if cmp_lines > cmp_count:
                cmp_lines = cmp_count
            n_cmp_tiles = (cmp_lines + KV_BLOCK - 1) // KV_BLOCK

            n_tiles = n_ori_tiles + n_cmp_tiles
            kv_row0 = cu_kv[seq]
            cmp_row0 = cu_cmp[seq]

            # F29（路径 A）：ori / cmp 两段表行已在 prologue 的 Init 阶段由全体
            # AIV 分摊生成、并过一个块屏障（见函数头 total_tasks 之前），
            # 任务循环里**不再生成**，只按 tok/seq 消费（OT 同构）。

        # ---- 本 task 的 body 数：真实 task = n_tiles；哨兵步 = PRELOAD_NUM
        #      （空核不补，避免越界 load）----
        n_body = n_tiles
        if step == n_steps:
            if g_done > 0:
                n_body = n_tiles + PRELOAD_NUM

        for i in tla.range(n_body):
            x = g_done + i

            # ================================================================
            # 站点 0（lag 0，AIV）：vec0(x) —— 稀疏 gather 落到 GM 暂存槽
            # OT ProcessVec0(runInfo[smlaRunInfoIdx])（:606）+ SetRunInfo
            # (smla35RunInfo[taskId%4])（:600）；两段都是 gather。
            # ================================================================
            if i < n_tiles:
                j0 = i
                nv0 = tla.as_numeric(0)
                nl0 = tla.as_numeric(0)
                slot0 = x - (x // SHARE_SLOTS) * SHARE_SLOTS
                # T1-1（r10）：索引环 2 槽按 tile 序 x 轮转（每个 AIV 自己的 UB 环）。
                # ---- 双索引结构的相位论证（为什么 ori/cmp 可以共用同一对 event）----
                # 每个 tile 只会走 ori 段**或** cmp 段之一（下面的
                # `if j0 < n_ori_tiles: ... else: ...`），站点 0 在同一个 body 里
                # 至多发射一次 gather。环的握手是「wait free[slot] -> DMA ->
                # set/wait ready[slot] -> 逐行标量读 -> set free[slot]」：
                #   * ready（idx_m2s0/1，MTE2->SCALAR）每次调用 set 一次、wait 一次；
                #   * free （idx_s2m0/1，SCALAR->MTE2）每次调用 wait 一次、set 一次，
                #     且 init_sync(pre_idx_ring=True) 已预置 2 个槽各 1 份信用。
                # 同一 body 内只有一个 gather ⇒ 两次调用之间的配对只可能是
                # 「同一槽的上一次调用」（槽键 x%2，相邻两次调用必然异槽），
                # 且 AIV 程序序保证上一次调用早于下一次 ⇒ 事件 id 生命周期不重叠。
                # 故 ori/cmp 共用 idx_m2s0/1、idx_s2m0/1 与 csa/os 的相位一致，
                # 无需为两段各开一套 id（dst=SCALAR / dst=MTE2 的 0..7 也已排满）。
                # 环内 idx_width / token_row / row_base 按段传入：
                #   ori 段 -> ori_idx / ori_idx_width / kv_row0(=cu_kv[seq])
                #   cmp 段 -> cmp_idx / cmp_idx_width / cmp_row0(=cu_cmp[seq])
                islot = x - (x // 2) * 2
                if j0 < n_ori_tiles:
                    base_line0 = j0 * KV_BLOCK
                    nv0 = ori_lines - base_line0
                    if nv0 > KV_BLOCK:
                        nv0 = tla.as_numeric(KV_BLOCK)
                    nl0 = nv0
                    if nl0 < 0:
                        nl0 = tla.as_numeric(0)
                    if kv_layout == 2 and not use_vec_gather:
                        # r12：PA 内联页数学 + OT CopyInKvSparse 成对落位
                        gather_kv_rows_pa_pairwise(
                            sync,
                            bufs,
                            ws,
                            ori_kv,
                            ori_idx,
                            tok,
                            ori_idx_width,
                            base_line0,
                            nl0,
                            slot0,
                            KV_STAGE_STRIDE,
                            ws_base,
                            block_table=ori_bt,
                            bt_seq=seq,
                            bt_max=ori_pa_max_blocks,
                            pa_block_size=ori_pa_block_size,
                            pa_stride=ori_pa_stride,
                            topk_value_mode=topk_value_mode,
                        )
                    elif is_vec_s2phyaddr == 1:
                        # T1-1：索引行 DMA -> UB 环 -> 标量读 + 基址加法
                        gather_kv_rows_ring(
                            sync,
                            bufs,
                            ws,
                            ori_kv,
                            ori_idx,
                            tok,
                            ori_idx_width,
                            kv_row0,
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
                            ori_kv,
                            ori_idx,
                            tok,
                            ori_idx_width,
                            base_line0,
                            nl0,
                            kv_row0,
                            slot0,
                            KV_STAGE_STRIDE,
                            ws_base,
                            block_table=ori_bt,
                            bt_seq=seq,
                            bt_max=ori_pa_max_blocks,
                            pa_block_size=ori_pa_block_size,
                            pa_stride=ori_pa_stride,
                            topk_value_mode=topk_value_mode,
                        )
                else:
                    ct0 = j0 - n_ori_tiles
                    base_line0 = ct0 * KV_BLOCK
                    nv0 = cmp_lines - base_line0
                    if nv0 > KV_BLOCK:
                        nv0 = tla.as_numeric(KV_BLOCK)
                    nl0 = nv0
                    if nl0 < 0:
                        nl0 = tla.as_numeric(0)
                    if kv_layout == 2 and not use_vec_gather:
                        # r12：PA 内联页数学 + OT CopyInKvSparse 成对落位
                        gather_kv_rows_pa_pairwise(
                            sync,
                            bufs,
                            ws,
                            cmp_kv,
                            cmp_idx,
                            tok,
                            cmp_idx_width,
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
                        # （cmp 段：cmp_idx / cmp_idx_width / cmp_row0；与 ori 段共用
                        #  一对 event，相位论证见站点 0 的 islot 注释）
                        gather_kv_rows_ring(
                            sync,
                            bufs,
                            ws,
                            cmp_kv,
                            cmp_idx,
                            tok,
                            cmp_idx_width,
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
                            cmp_idx,
                            tok,
                            cmp_idx_width,
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
                r0_task = task
                r0_j = j0
                r0_nt = n_tiles
                r0_nvalid = nv0
                r0_nload = nl0

            # ================================================================
            # 站点 1（lag 1，AIC MTE2）：load(x-1)
            # OT IterateLoadQK(runInfo[(taskId+3)%4], isFirstLoop)（:628），
            # 条件 taskId>0 && smlaNotLastTwoLoop（:627）⇔ "tile(x-1) 存在"
            # ⇔ x>=1 && i < n_tiles+1（哨兵步 n_tiles=0 时只跑 i=0，正是 OT 尾
            # 排空的第一个 body）。不满足该条件的迭代走 OT :642-648 的跳过分支：
            # IS_SPLIT_G 下 smlaMaxS2LoopCnt-- 并补一对核间 MTE2 屏障。
            # ================================================================
            if x >= 1:
                if i < n_tiles + 1:
                    xg1 = x - 1
                    slot_k1 = xg1 - (xg1 // KV_SLOTS) * KV_SLOTS
                    # 暂存地址用共享环深度（与 AIV 侧 slot0 同一派生式）；
                    # L1 K 环仍 3 槽（slot_k1 不变）。非共享时是纯别名，不产生指令。
                    slot_st1 = slot_k1
                    if SHARE_SLOTS != KV_SLOTS:
                        slot_st1 = xg1 - (xg1 // SHARE_SLOTS) * SHARE_SLOTS
                    # OT IterateLoadQK 首迭代的 CopyQGmToL1（:331-333，isFirstLoop
                    # 是整个核的第一次 load，不是每 task）：只负责本核首 tile 的
                    # Q；其余 task 的 Q 由站点 2 的 notLastTwoLoop 预取。
                    if xg1 == 0:
                        qn1 = (r1_task - task0) // q_bn
                        q2n1 = qn1 + qn1
                        qa1 = q2n1 - (q2n1 // 3) * 3
                        q2n1b = q2n1 + 1
                        qb1 = q2n1b - (q2n1b // 3) * 3
                        load_q(
                            sync, bufs, q, r1_task, m, dtype_in, slot_a=qa1, slot_b=qb1
                        )
                    # OT :347-348 / :357-358：load 之前一对 MTE2 屏障
                    # A 步一（共享钳制）：OT 的顺序是「自有 AIV 等待 →
                    # 全 AIC 屏障 → 读」（csa_block_cube.h:249-253）。共享暂存时
                    # 必须先等本核 AIV，再进全 AIC 屏障，否则伙伴核的象限可能没写完。
                    if _bv._TRACE_STAGE:
                        wait_kv_stage(sync)
                        if s2_budget > 0:
                            s2_budget = s2_budget - 1
                            # OT 原形态跨块节流（csa_block_cube.h:249-253）
                            split_g_mte2_pair(sync)
                    else:
                        if _PAIRED:
                            if s2_budget > 0:
                                s2_budget = s2_budget - 1
                                split_g_mte2_pair(sync)
                        # 稀疏：等两个 AIV 把行写进 GM 暂存槽（OT v0ResGm 跨核等待）
                        wait_kv_stage(sync)
                    load_k_ptr(
                        sync,
                        bufs,
                        ws.ptr + ws_base + slot_st1 * KV_STAGE_STRIDE,
                        r1_nload,
                        slot_k1,
                        dtype_in,
                    )
                else:
                    if _PAIRED:
                        if s2_budget > 0:
                            s2_budget = s2_budget - 1
                            split_g_mte2_pair(sync)

            # ================================================================
            # 站点 2（lag 2）：vec1(x-2)（AIV）+ bmm1(x-2)（AIC）
            # OT ProcessVec1((taskId+2)%4)（:612-615）与
            # IterateBmm1(bmm1Slot, smlaNotLastTwoLoop, Next=(taskId+3)%4,
            # runInfo2=(taskId+2)%4)（:653-660），条件 taskId>1 && smlaNotLast
            # ⇔ "tile(x-2) 存在" ⇔ x>=2 && i < n_tiles+2（哨兵步：i=0,1）。
            # ================================================================
            if x >= 2:
                if i < n_tiles + 2:
                    xg2 = x - 2
                    slot_s2 = xg2 - (xg2 // P_SLOTS) * P_SLOTS
                    slot_k2 = xg2 - (xg2 // KV_SLOTS) * KV_SLOTS
                    # sum/max 读槽：任务首 tile 读本 tile 槽（vec_init 重灌的初值），
                    # 其余读上一 tile 槽（(x-1)%2 链式接力；F11 键）
                    slot_sum_rd = slot_s2
                    if r2_j > 0:
                        slot_sum_rd = 1 - slot_s2
                    # Q 3 槽环：宿主 task 的两半槽号 = 本核内 task 序号 n 的
                    # {2n, 2n+1} mod 3（与 load_q 侧同一派生式）
                    q_nta2 = (r2_task - task0) // q_bn
                    q_two2 = q_nta2 + q_nta2
                    qa2 = q_two2 - (q_two2 // 3) * 3
                    q_two2b = q_two2 + 1
                    qb2 = q_two2b - (q_two2b // 3) * 3
                    if r2_j == 0:
                        # OT InitVec1SoftmaxFromSinks（任务首 tile）：只灌 max/sum，
                        # O 由站点 3 的 vec2 fresh 覆盖启动（clear_o=0，F11）
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
                    # OT IterateBmm1Impl 末尾（:422-429）：宿主 tile 是本 task 末
                    # tile 时（notLastTwoLoop）预取"下一 task"的 Q。目标 task 由环
                    # 派生 = r2_task + q_bn（环里 x-1 恒为宿主 tile 的直接后继）。
                    # 守卫 = "tile(x-1) 存在" ⇔ i < n_tiles+1：哨兵步只有 i=0 可能
                    # 命中（= OT extra=0 分支，末 task 仅 1 个 tile 时正由它装 Q）。
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
            # 站点 3（lag 3）：bmm2(x-3) + vec2(x-3) + 末 tile copy_out
            # OT IterateBmm2(runInfo3=(taskId+1)%4)（:666-672），条件 taskId>2
            # ⇔ "tile(x-3) 存在" ⇔ x>=3（body 内 i>=0 恒成立）。
            # copy_out（OT Bmm2DataCopyOut）折叠到 owner task 末 tile 的 vec2
            # 之后，与串行版位置等价。
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
            r2_task = r1_task
            r2_j = r1_j
            r2_nt = r1_nt
            r2_nvalid = r1_nvalid
            r2_nload = r1_nload
            if i < n_tiles:
                r1_task = r0_task
                r1_j = r0_j
                r1_nt = r0_nt
                r1_nvalid = r0_nvalid
                r1_nload = r0_nload

        # 本 task 结束：累计本核 tile 前缀（哨兵步 n_tiles=0，不再变化）
        g_done = g_done + n_tiles

    # ---- 预算排空（OT :674-680）：把剩余配额用等量屏障对补发，保证每核条数恒等 ----
    if _PAIRED:
        for _k in tla.range(s2_budget):
            if _bv._TRACE_STAGE:
                pair_rendezvous(sync)
            else:
                split_g_mte2_pair(sync)

    # ---- F27 补·退出排空：保证出拷贝在核退出前全部退休 ----------------
    # 症状：核退出时最后一次出拷贝的 GM 写仍可能在飞（证据 ITERATIONS.md §4.4：
    # 每个核的最后一次 copy 都落在本核最后一个 body；p64/full50 坏的是 0..11 的
    # 第二 task，n09/n077 坏的是 1 task 核的唯一 task；整行 0 与「部分列不逐位」
    # 两种签名共存）。语义对齐 OT
    # `sparse_flash_mla_csa_block_vector_arch35.h:1515` 的
    # `FreeEvent ... WaitFlag<HardEvent::MTE3_V>(INNERCORE_STAGE2)`：set 落在
    # MTE3 管（排在所有已发出拷贝之后），wait 在 V 管消费，核退出前等到它。
    # 复用 init_mte3_v（MTE3->V，id 0，主循环内无其它使用，set/wait 成对自洽）。
    with tla.vector():
        tla.set_flag(sync.init_mte3_v)
        tla.wait_flag(sync.init_mte3_v)

    # ---- 追踪期闭合校验（F45）：承诺开共享臂却一次没发射 ⇒ 编译失败 ----
    _bv.verify_trace_closed("ocs")
