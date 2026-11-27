# ---------------------------------------------------------------------------
# smla_csa_block_vector.py — SmlaCsaBlockVector 的 DSL 对应
#
# 对照 ops-transformer/attention/sparse_flash_mla/op_kernel/arch35/
#   sparse_flash_mla_csa_block_vector_arch35.h
#     InitVec1SoftmaxFromSinks : 768-793   (sum=1.0, max=sinks)
#     ComputeVec1Softmax       : 748-765   (先乘 scale，再减 max，再 exp)
#     CopyVec1ResultToL1       : 796-815   (Cast CAST_ROUND -> UB P -> L1)
#     ProcessVec2              : 1055-1134 (UB fp32 FlashUpdate)
#     LastDivNew               : O / sum
#     Bmm2DataCopyOut          : 1277-1298 (CAST_ROUND -> GM)
#
# 行/列循环用 tla.range 动态循环（VF 静态展开会超 vf stack 6,144 B）。
# ---------------------------------------------------------------------------

from __future__ import annotations

import catlass.tla as tla

from smla_csa_common import (
    D_CHUNK,
    HEAD_DIM,
    IDX_RING_SLOTS,
    IDX_RING_STRIDE,
    INIT_POP_ELE,
    KV_BLOCK,
    KV_COPY_IN,
    KV_NZ_PITCH,
    KV_UNIT,
    MIN_F,
    Q_ROWS,
    STATE_SLOT_STRIDE,
    V_SEGS,
    static_range,
)

VL = 64
VL_B16 = 128


# ---------------------------------------------------------------------------
# F4：输出初始化 / 清零
#
# 逐函数对照：
#   attention/common/op_kernel/init_output.h:28-96
#       AttentionCommon::InitOutput<T, SYNC_ID, POP_BUF_START_ADDR,
#                                   POP_BUF_ELE_SIZE, ENABLE_LOCK>
#   arch35/sparse_flash_mla_csa_block_vector_arch35.h:1310-1348 InitOutputSingleCore
#   arch35/sparse_flash_mla_csa_block_vector_arch35.h:1350-1362 CleanOutput
#   arch35/sparse_flash_mla_common_arch35.h:48        INNERCORE_INITOUT_MTE3_V = 0
#   arch35/sparse_flash_mla_csa_block_vector_arch35.h:76-77 initOutputEventId
#   arch35/sparse_flash_mla_csa_kernel_arch35.h:293-319  needInit
#   arch35/sparse_flash_mla_swa_kernel_arch35.h:241-266  needInit（SWA/HCA 同构）
#
# 结构（与 OT 同）：AIV 按 vecIdx 把整张输出切成 vecCoreNum 段，每段先
# Duplicate 一次 popBuf，再循环 DataCopy，尾块 DataCopyPad。flag 用
# MTE3_V（保证上一次 MTE3 读完）与 V_MTE3（保证 Duplicate 写完）两份。
#
# 偏离（已记录在 ITERATIONS.md）：OT 在 Init 阶段做整张清零并随后
# SyncAll<false>() 屏障；TLA 没有公开的 block 级屏障，故这里由各 kernel 在
# 主循环之前调用一次（AIV 的第一条指令序列），时序上早于任何对 out 的写。
# ---------------------------------------------------------------------------
@tla.jit
def init_output_gm(
    sync, bufs, gm, base_elem, total_elem, pop_ele, fill_dtype, use_lse_buf=0
):
    """AttentionCommon::InitOutput 的等价翻译（单核那一段）。"""
    src = bufs.ub_init_lse if use_lse_buf == 1 else bufs.ub_init
    ub = tla.make_tensor(
        src, tla.make_layout(tla.make_shape(pop_ele), tla.make_stride(1))
    )
    # OT：SetFlag<MTE3_V>(SYNC_ID); WaitFlag<MTE3_V>(SYNC_ID)
    tla.set_flag(sync.init_mte3_v)
    tla.wait_flag(sync.init_mte3_v)
    with tla.vec.func(mode="simd"):
        zero = tla.full(0.0, dtype=fill_dtype)
        zmask, _ = tla.update_mask(VL, dtype=fill_dtype)
        # AscendC::Duplicate(popBuffer, initValue, POP_BUF_ELE_SIZE)
        for c in tla.range(pop_ele // VL):
            tla.tile_view(ub, tla.make_shape(VL), tla.make_coord(c)).store(
                zero,
                params=tla.params.UnalignStoreParams(),
                mask=zmask,
            )
        tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
    # OT：SetFlag<V_MTE3>(SYNC_ID); WaitFlag<V_MTE3>(SYNC_ID)
    tla.set_flag(sync.init_v_mte3)
    tla.wait_flag(sync.init_v_mte3)
    loop_cnt = total_elem // pop_ele
    tail = total_elem - loop_cnt * pop_ele
    for i in tla.range(loop_cnt):
        gm_blk = tla.make_tensor(
            gm.ptr + base_elem + i * pop_ele,
            tla.make_layout(tla.make_shape(pop_ele), tla.make_stride(1)),
        )
        tla.copy(gm_blk, ub)
    if tail > 0:
        # OT 尾块用 DataCopyPad(outGm[gmOffset], popBuffer, blockLen=tail*sizeof(T))
        gm_tail = tla.make_tensor(
            gm.ptr + base_elem + loop_cnt * pop_ele,
            tla.make_layout(tla.make_shape(tail), tla.make_stride(1)),
        )
        tla.copy(gm_tail, tla.tile_view(ub, tla.make_shape(tail), tla.make_coord(0)))
    tla.set_flag(sync.init_mte3_v)
    tla.wait_flag(sync.init_mte3_v)


def _lse_total_len(total_output, total_lse, head_dim):
    """softmax_lse 元素数 = tokens*head_num。

    R37：head_dim>0 的非 os 路径由 total_output//head_dim 在启用时才算
    （调用方传 0 占位）——若在调用点算 cu_q[num_seqs]*head_num，即使
    enable_lse=0 也会多发射一条 scalar_load+mul，破坏 lse=0 的 IR/缓存键。
    """
    d = int(getattr(head_dim, "value", head_dim))
    if d == 0:
        return total_lse
    return total_output // d


@tla.jit
def init_output_single_core(
    sync,
    bufs,
    attention_out,
    lse,
    total_output,
    total_lse,
    enable_lse,
    dtype_in,
    head_dim: tla.Constexpr[int] = 0,
):
    """InitOutputSingleCore（csa_block_vector_arch35.h:1310-1348）。

    只读 / 只在 `clean_output` 打开的 `tla.vector()` 作用域内调用
    （tla 不允许 vector 作用域嵌套）。
    """
    vec_core_num = tla.arch.block_num() * 2  # CV_RATIO * GetBlockNum()
    vec_idx = tla.arch.block_idx() * 2 + tla.arch.sub_block_idx()
    # ---- attentionOut ----
    single = (total_output + vec_core_num - 1) // vec_core_num
    off = vec_idx * single
    if off < total_output:
        actual = total_output - off
        if actual > single:
            actual = single
        init_output_gm(
            sync, bufs, attention_out, off, actual, INIT_POP_ELE, dtype_in, 0
        )
    # ---- softmaxLse（isSoftmaxLseEnable 时）----
    if enable_lse == 1:
        lse_len = _lse_total_len(total_output, total_lse, head_dim)
        single_l = (lse_len + vec_core_num - 1) // vec_core_num
        off_l = vec_idx * single_l
        if off_l < lse_len:
            actual_l = lse_len - off_l
            if actual_l > single_l:
                actual_l = single_l
            init_output_gm(
                sync, bufs, lse, off_l, actual_l, INIT_POP_ELE, tla.Float32, 1
            )


@tla.jit
def clean_output(
    sync,
    bufs,
    attention_out,
    lse,
    need_init,
    total_output,
    total_lse,
    enable_lse,
    dtype_in,
    head_dim: tla.Constexpr[int] = 0,
):
    """CleanOutput（csa_block_vector_arch35.h:1350-1362）：AIV only，needInit==1 才清。"""
    with tla.vector():
        if need_init == 1:
            init_output_single_core(
                sync,
                bufs,
                attention_out,
                lse,
                total_output,
                total_lse,
                enable_lse,
                dtype_in,
                head_dim=head_dim,
            )


@tla.jit
def _ub_s(bufs, rows, slot=0):
    # F6 双槽：连续分配（OT bmm1Buffers[2]），槽 stride = Q_ROWS//2*KV_BLOCK
    # （alloc 的 half 恒为 Q_ROWS//2：swa kernel 不传 row_cap）；slot=0（串行
    # 调用方默认）时偏移 0，指令流不变。
    return tla.make_tensor(
        bufs.ub_s + slot * (Q_ROWS // 2 * KV_BLOCK),
        tla.make_layout(tla.make_shape(rows, KV_BLOCK), tla.make_stride(KV_BLOCK, 1)),
    )


@tla.jit
def _ub_p(bufs, rows):
    # T2b：P 写改 zN 视图（参考 basic_mixed_store_zN.py:145 同款）：
    # make_tensor_like 从行主序视图派生**标准 zN（无 pad）**——rows=32 时
    # 列 fract 组距 = 32*16 = 512 el，物理 4096 el 全密（rows=17 时 pad16
    # 到 32 行，组距仍 512，物理 4096 <= half*KV_BLOCK 分配）。pass 2 用
    # (1,KV_BLOCK) tile_view coord (r2,0) 取行基址 r2*16，块存散射 8 个
    # 列 fract；block_stride 由 stride[1][1]//shape[1][0] 推导（单位
    # 16 el DataBlock，不手填）；L1 copy 走 zN->zN 路由直搬不做块内重排。
    # 与 OT vec1Srcstride=33 的唯一差异 = 每列块 16 el padding 省略（OT
    # 的 pad 从不被读：CopyVec1ResultToL1 每组只搬 halfM 块），数值无
    # 影响；此偏离进 T2b 账本，OT 同款 33 块距留作可选档。
    ub_rm = tla.make_tensor(
        bufs.ub_p,
        tla.make_layout(tla.make_shape(rows, KV_BLOCK), tla.make_stride(KV_BLOCK, 1)),
    )
    return tla.make_tensor_like(bufs.ub_p, ub_rm, tla.arch.zN)


@tla.jit
def _dm(bufs, rows, slot=0):
    # F11 dm 双槽（OT softmaxExpBufs[taskIdMod2] 同键：taskId = 全局 tile 序，
    # 槽 = xg%2）。连续分配（alloc half*2，preload 实例），槽 1 由
    # ptr + slot*Align8(rows) 寻址（_ub_s 同款 Pointer 算术）；slot=0（串行
    # 调用方默认）偏移 0，指令流不变。
    # F11·bc 修（Iter F11-bc2）：槽距必须取 Align8(rows) 而非 rows——
    # bc_stage_cur/dbg dump 以 (rows8, slot) 调本函数，旧式 slot*rows 距离下
    # rows ∈ {1,4} 时读方基址 slot*8 与写方基址 slot*rows 错位：奇 tile 的
    # staged max/sum 全为未初始化 UB 垃圾，merge 链自 tile 2 起全污染
    # （#004 ratio 0.3138 / #007 ratio 0.00017 的根因）。rows 为 8 的倍数时
    # Align8(rows)==rows，槽址逐位不变（#001/005/008 等已过例不受扰动）。
    _stride = STATE_SLOT_STRIDE if bufs.opt_vec else _bc_rows8(rows)
    return tla.make_tensor(
        bufs.ub_dm + slot * _stride,
        tla.make_layout(tla.make_shape(rows), tla.make_stride(1)),
    )


@tla.jit
def _sum(bufs, rows, slot=0):
    # F11 sum 双槽（OT softmaxSumBufs[2] 同构）：copy_out(P 末) 归一化读槽
    # (P末)%2 与 vec_init(T 首) 重灌槽 (T首)%2 = (P末+1)%2 异槽不互踩；
    # vec1(x-1) 读改写槽 (x-1)%2。分配/寻址同 _dm（槽距 Align8(rows)，见
    # _dm 内 F11·bc 修注释：slot*rows 与 rows8 读方错位是 #004/#007 根因）。
    _stride = STATE_SLOT_STRIDE if bufs.opt_vec else _bc_rows8(rows)
    return tla.make_tensor(
        bufs.ub_sum + slot * _stride,
        tla.make_layout(tla.make_shape(rows), tla.make_stride(1)),
    )


@tla.jit
def _max(bufs, rows, slot=0):
    # F11·bc max 双槽（OT softmaxMaxBufs[multiCoreIdxMod2] 先例，键同 sum =
    # 全局 tile 序 xg%2）：bc=1 时 merge_prev(t) 晚于 vec1(t+1) 执行（F11
    # body t+2 先 stage1 后 stage2），单槽下 lse_c 读到 max(t+1)；双槽后
    # vec1(t) 写槽 t%2、merge(t) 就地 RMW 槽 t%2、vec1(t+1)/reinit(t+1) 写
    # 异槽。bc=0 无 merge 实例，读写同槽退化为改前单槽语义；slot=0（串行
    # 调用方默认）偏移 0，指令流不变。分配/寻址同 _dm/_sum（槽距
    # Align8(rows)：rows ∈ {1,4} 时 slot*rows 与 bc_stage/dbg 的 rows8 读方
    # 错位，是 #004/#007 bc=1 全错的根因，Iter F11-bc2 修复）。
    _stride = STATE_SLOT_STRIDE if bufs.opt_vec else _bc_rows8(rows)
    return tla.make_tensor(
        bufs.ub_max + slot * _stride,
        tla.make_layout(tla.make_shape(rows), tla.make_stride(1)),
    )


@tla.jit
def _stats(ptr, rows):
    return tla.make_tensor(
        ptr, tla.make_layout(tla.make_shape(rows), tla.make_stride(1))
    )


@tla.jit
def vec_init_from_sinks(
    sync,
    bufs,
    sinks,
    task,
    m,
    rows,
    dtype_in,
    sink_off=0,
    has_sinks=1,
    sink_cap=0,
    slot_sum=0,
    slot_max=0,
    clear_o: tla.Constexpr[int] = 1,
    sink_reuse: tla.Constexpr[int] = 0,
):
    """每 task 初值：max=sinks，sum=1.0（R16）。

    F11 起 O 区清零拆除（docstring 见函数体内注释）：任务首 tile 的 O 由
    vec2 fresh 覆盖启动（OT InitVec1SoftmaxFromSinks 同款，只灌 max/sum）。

    sink_off 是 G 轴偏移：IS_SPLIT_G 时一个 token 的 G 被切成两半，
    第二半（aicIdx%2==1，goIdx=halfG）必须取 sinks[goIdx + sub*rows]。

    sink_cap>0 时走「奇数 G」路径（#013 h=65）：把整条 sinks（host 侧补齐到
    8 元素/32B 倍数）一次性搬到 UB，再按 goIdx+sub*rows+r 逐行取。
    这样与 OT 一致（sinks 在 UB 里连续，InitSoftmaxFromSinks 从 sinksOffset
    读 halfMRealSize 行，sparse_flash_mla_csa_block_vector_arch35.h:782-792），
    也避免 4B 粒度的 GM->UB 拷贝触发 MTE 对齐错误。
    越界的丢弃行读到的是补齐的 0（这些行不写 out）。

    has_sinks=0 对应算子未传 sinks（OT: isSinks=false，走
    ComputeVec1Softmax<false>）：初值 max = negativeFloatScalar、sum = 0。
    之后的 update 公式 exp(old_max-new_max) 在该初值下恒为 0，
    new_sum = 0*0 + Σexp 与 OT 的首 tile 直算逐位相同（0.0+x 不改变位型）。
    """
    c0 = 0
    with tla.vector():
        sub = tla.arch.sub_block_idx()
        # sink_len：奇数 G 用补齐后的整条 sinks，偶数 G 仍是本 AIV 的 rows 行。
        sink_len = sink_cap if sink_cap > 0 else rows
        ub_sink = _stats(bufs.ub_sink, sink_len)
        gm_sink_all = tla.make_tensor(
            sinks.ptr, tla.make_layout(tla.make_shape(sink_len), tla.make_stride(1))
        )
        if has_sinks == 1:
            sync.m_sink.lock(pipe=tla.pipes.MTE2)
            if sink_cap == 0:
                # m 为奇数时第二子块没有实义行（OT secondHalfMRealSize=0），
                # 其 sink 偏移会越界，钳回本 task 的首行以免读穿 sinks。
                s_off = sink_off + sub * rows
                if sink_reuse == 1:
                    # E3（TOK2）：m=2*head_num、rows=head_num，行 0..rows-1 与
                    # rows..m-1 分属两个 token，但 sink 索引都是 token 内 head
                    # 号 0..head_num-1，两个子块复用同一段 sinks（不按 sub 偏移，
                    # 否则 AIV1 会越界读 sinks[head_num..]）。
                    s_off = sink_off
                if sub * rows >= m:
                    s_off = sink_off
                gm_sink = tla.make_tensor(
                    sinks.ptr + s_off,
                    tla.make_layout(tla.make_shape(rows), tla.make_stride(1)),
                )
                tla.copy(ub_sink, gm_sink)
            else:
                # 整条 sinks 一次搬入（起点 0、长度 32B 对齐），逐行取由下游完成。
                tla.copy(ub_sink, gm_sink_all)
            sync.m_sink.unlock(pipe=tla.pipes.MTE2)
        sync.m_sink.lock(pipe=tla.pipes.VECTOR)
        # F11·bc：max 重灌槽 slot_max = 宿主 tile（T 首）的 xg%2——与 sum 重灌
        # 槽同键；bc=1 块首的 reinit 也写本 tile 槽（见 bc_reinit_stats）。
        ub_max = _max(bufs, rows, slot_max)
        # F11：sum 重灌槽 slot_sum = 宿主 tile（T 首）的 xg%2——与 vec1(T首)
        # 的写槽一致（非首 tile 的 vec1 读槽是 (x-1)%2，首 tile 读本槽初值）；
        # 同 body 后发的 copy_out(P 末) 读 (P末)%2 异槽不互踩。
        ub_sum = _sum(bufs, rows, slot_sum)
        # F11：O 清零改为 Constexpr 开关——swa 流水版（clear_o=0）任务首 tile
        # 的 O 由 vec2 fresh 覆盖接管（交换发射序的必要配套：vec_init 先于同
        # body 的 copy_out(P末)，清零会抹掉 P 末 running O）；os/串行调用方
        # （默认 1）保持清零启动，指令流与改前逐条一致。ub_o 视图无条件
        # 构造（纯地址算术，clear_o=0 时无使用者被 DCE；前端要求分支内
        # 使用的变量在分支前已初始化）。
        ub_o = tla.make_tensor(
            bufs.ub_o,
            tla.make_layout(
                tla.make_shape(rows, HEAD_DIM), tla.make_stride(HEAD_DIM, 1)
            ),
        )
        ub_sink_vl = tla.make_tensor(
            bufs.ub_sink, tla.make_layout(tla.make_shape(VL), tla.make_stride(1))
        )
        ub_max_vl = tla.make_tensor(
            bufs.ub_max + slot_max * STATE_SLOT_STRIDE,
            tla.make_layout(tla.make_shape(VL), tla.make_stride(1)),
        )
        ub_sum_vl = tla.make_tensor(
            bufs.ub_sum + slot_sum * STATE_SLOT_STRIDE,
            tla.make_layout(tla.make_shape(VL), tla.make_stride(1)),
        )
        with tla.vec.func(mode="simd"):
            preg = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            one_mask, _ = tla.update_mask(1, dtype=tla.Float32)
            brc = tla.params.NormalLoadParams(
                load_dist=tla.params.LoadDist.DIST_BRC_B32
            )
            zero = tla.full(0.0, dtype=tla.Float32)
            one = tla.full(1.0, dtype=tla.Float32)
            minreg = tla.full(MIN_F, dtype=tla.Float32)
            if tla.const_expr(bufs.opt_vec and clear_o == 0 and sink_cap == 0):
                rmask, _ = tla.update_mask(rows, dtype=tla.Float32)
                ub_max_v = tla.tile_view(
                    ub_max_vl, tla.make_shape(VL), tla.make_coord(0)
                )
                ub_sum_v = tla.tile_view(
                    ub_sum_vl, tla.make_shape(VL), tla.make_coord(0)
                )
                if has_sinks == 1:
                    ub_sink_v = tla.tile_view(
                        ub_sink_vl, tla.make_shape(VL), tla.make_coord(0)
                    )
                    ub_max_v.store(ub_sink_v.load(), mask=rmask)
                    ub_sum_v.store(one, mask=rmask)
                else:
                    ub_max_v.store(minreg, mask=rmask)
                    ub_sum_v.store(zero, mask=rmask)
                tla.local_mem_bar(
                    tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                )
            else:
                for r in tla.range(rows):
                    ub_max_r = tla.tile_view(
                        ub_max, tla.make_shape(1), tla.make_coord(r)
                    )
                    ub_sum_r = tla.tile_view(
                        ub_sum, tla.make_shape(1), tla.make_coord(r)
                    )
                    if has_sinks == 1:
                        # 奇数 G：UB 里是整条 sinks，行号 = goIdx + sub*rows + r
                        # （OT sinksOffset = goIdx + subBlockIdx*firstHalfMRealSize）。
                        # 偶数 G：UB 里已按 sub*rows 偏移，取第 r 行。
                        sink_row = (sink_off + sub * rows + r) if sink_cap > 0 else r
                        ub_sink_r = tla.tile_view(
                            ub_sink, tla.make_shape(1), tla.make_coord(sink_row)
                        )
                        ub_max_r.store(
                            ub_sink_r.load(params=brc),
                            params=tla.params.UnalignStoreParams(),
                            mask=one_mask,
                        )
                        ub_sum_r.store(
                            one, params=tla.params.UnalignStoreParams(), mask=one_mask
                        )
                    else:
                        ub_max_r.store(
                            minreg,
                            params=tla.params.UnalignStoreParams(),
                            mask=one_mask,
                        )
                        ub_sum_r.store(
                            zero, params=tla.params.UnalignStoreParams(), mask=one_mask
                        )
                    if clear_o == 1:
                        for c in tla.range(HEAD_DIM // VL):
                            ub_o_c = tla.tile_view(
                                ub_o, tla.make_shape(1, VL), tla.make_coord(r, c)
                            )
                            ub_o_c.store(zero, mask=preg)
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
        sync.m_sink.unlock(pipe=tla.pipes.VECTOR)


@tla.jit
def process_vec1(
    sync,
    bufs,
    scale,
    tail_n,
    m,
    rows,
    dtype_in,
    slot_s=0,
    slot_p=0,
    preload: tla.Constexpr[int] = 0,
    slot_sum_rd=0,
    slot_max_rd=0,
    tail_n2=-1,
):
    """softmax(update 分支) + P CAST_ROUND 写 L1。

    slot_s/slot_p：F6 流水版的 UB S 槽 / L1 P 槽（bmm1Buffers[t%2] /
    l1PBuffers[t%2]；串行调用方不传 = 槽 0，指令流不变）。
    preload=1（仅 swa kernel）：F14 步骤 A 后 p_free 回执已删（原「写 l1_p
    槽前等 p_free」的 wait 已不存在），槽踩写安全 = vec1(x) 写前的 s_ready
    wait 链 + AIC 程序序 + MTE1 FIFO（见下方 F14 注释）+ m_p mutex。
    =0 实例无此语义差异，指令流与改前逐条一致。
    """
    c0 = 0
    c1 = 1
    with tla.vector():
        sub = tla.arch.sub_block_idx()
        ub_s = _ub_s(bufs, rows, slot_s)
        ub_p = _ub_p(bufs, rows)
        # F11·bc：max 双槽——pass1 的 old_max 读槽 slot_max_rd（(x-1)%2 链式
        # / 首tile x%2 重灌槽，键同 slot_sum_rd），new_max 写槽 slot_s（x%2）；
        # pass2 的 mx 读刚写的槽 slot_s。bc=0 无 bc 链实例时两槽键相同路径
        # 退化为改前单槽语义（值恒等：读 (x-1)%2 槽仍是 vec1(x-1) 写的
        # max(x-1)）。
        ub_max = _max(bufs, rows, slot_s)
        ub_max_rd = _max(bufs, rows, slot_max_rd)
        # F11：sum 槽 = slot_s（xg1%2）——pass2 **写**槽（sum(x) 驻留 x%2）；
        # **读**槽 slot_sum_rd = (x-1)%2（sum(x-1)，链式 running 态接力），
        # 任务首 tile 特例 = x%2（vec_init 重灌的初值槽）——键由 kernel 侧
        # 按 j1==0 动态派生。OT softmaxSumBufs[multiCoreIdxMod2] 的读改写
        # 同槽即隐式「读上一 tile 槽、写本 tile 槽」（键每 tile 交替）。
        ub_sum = _sum(bufs, rows, slot_s)
        ub_sum_rd = _sum(bufs, rows, slot_sum_rd)
        # F11：dm 写槽 = slot_s（= xg1 % 2，OT softmaxExpBufs[taskIdMod2]
        # 同键）；vec2(xg2) 读槽 slot_dm = xg2 % 2——相邻 tile 奇偶交替，
        # 源序交换后 vec1(x-1) 写与 vec2(x-2) 读不再互踩。
        ub_dm = _dm(bufs, rows, slot_s)
        tla.cross_core_wait_flag(sync.s_ready, tla.pipes.VECTOR, aiv_id=0)
        tla.cross_core_wait_flag(sync.s_ready, tla.pipes.VECTOR, aiv_id=1)
        # F14 步骤 A（裁决采纳）：去除 p_free 回执（OT IterateBmm2Impl 对
        # l1PBuffers 单向 + 双槽 slack，无 C→V credit）。踩写安全链：vec1(x)
        # 写槽 x%2 前必 WAIT s_ready ← bmm1(x) 的 SET（FIX，S fixpipe 后）；
        # AIC 单核程序序 bmm2(x-2)（读槽 x%2 的 MTE1 LoadDataToL0A）→ … →
        # bmm1(x)，MTE1 队列 FIFO ⇒ bmm2(x-2) 的 P 读完成早于 bmm1(x) 的
        # SET ⇒ vec1(x) 的 MTE3 写不踩。领先量上界 2 tile 由该链封顶。
        sync.m_p.lock(pipe=tla.pipes.VECTOR)

        # E3（TOK2）：一个 task 的 m 行由两个 token 拼成，AIV0 处理 token A
        # （行 0..rows-1）、AIV1 处理 token B（行 rows..2rows-1），两半的有效列数
        # 各自独立。tail_n2 缺省 -1 表示单 token 实例：tn 恒等于 tail_n，
        # 指令流与改前逐条一致。
        tn = tail_n
        if tail_n2 >= 0:
            if sub == 0:
                tn = tail_n
            if sub == 1:
                tn = tail_n2
        n0 = tn
        if n0 > VL:
            n0 = tla.as_numeric(VL)
        n1 = tn - VL
        if n1 > VL:
            n1 = tla.as_numeric(VL)
        if n1 < 0:
            n1 = tla.as_numeric(0)
        ub_max_rd_vl = tla.make_tensor(
            bufs.ub_max + slot_max_rd * STATE_SLOT_STRIDE,
            tla.make_layout(tla.make_shape(VL), tla.make_stride(1)),
        )
        ub_max_wr_vl = tla.make_tensor(
            bufs.ub_max + slot_s * STATE_SLOT_STRIDE,
            tla.make_layout(tla.make_shape(VL), tla.make_stride(1)),
        )
        ub_sum_rd_vl = tla.make_tensor(
            bufs.ub_sum + slot_sum_rd * STATE_SLOT_STRIDE,
            tla.make_layout(tla.make_shape(VL), tla.make_stride(1)),
        )
        ub_sum_wr_vl = tla.make_tensor(
            bufs.ub_sum + slot_s * STATE_SLOT_STRIDE,
            tla.make_layout(tla.make_shape(VL), tla.make_stride(1)),
        )
        ub_dm_wr_vl = tla.make_tensor(
            bufs.ub_dm + slot_s * STATE_SLOT_STRIDE,
            tla.make_layout(tla.make_shape(VL), tla.make_stride(1)),
        )
        with tla.vec.func(mode="simd"):
            preg = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            preg_b16 = tla.create_mask(pattern=tla.mask.ALL, dtype=dtype_in)
            preg0, _ = tla.update_mask(n0, dtype=tla.Float32)
            preg1, _ = tla.update_mask(n1, dtype=tla.Float32)
            one_mask, _ = tla.update_mask(1, dtype=tla.Float32)
            minreg = tla.full(MIN_F, dtype=tla.Float32)
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
            brc = tla.params.NormalLoadParams(
                load_dist=tla.params.LoadDist.DIST_BRC_B32
            )
            # ---- F33：vec1 softmax 三路 s2RealSize 特化 -------------------
            # OT 锚点：sparse_flash_mla_csa_block_vector_arch35.h:752-775
            # （ComputeVec1Softmax 三路分派）；三臂实现分别在
            # vf_basic_block_{aligned128,unaligned64,unaligned128}_*_sfa.h。
            # 条件：① tail_n==128 ② tail_n<=64 ③ 64<tail_n<128（顺序同 OT，
            # ② 先于 ③）。tail_n ≡ OT s2RealSize；:938 的
            # s2RealSizeUpdate=s2RealSize 使第三路 OR 不引入新情形（§26.6）。
            # ① 臂语句与交付 c520323d 逐条相同（回归门：Dur / aiv_vec 不动）。
            # 注：条件必须内联在 if 处——DSL 不接受先赋给 Python 变量的动态条件。
            if tla.const_expr(bufs.opt_vec):
                # Native CATLASS-DSL 4-row interleaved single-pass softmax:
                # DIST_DINTLV_B32 splits even/odd lanes directly so cast_zero|cast_one
                # produces packed [0..127] without VSELR; tla.full broadcasts reduced max
                # in-register without a UB round-trip.
                rmask, _ = tla.update_mask(rows, dtype=tla.Float32)
                dintlv = tla.params.NormalLoadParams(
                    load_dist=tla.params.LoadDist.DIST_DINTLV_B32
                )
                ust = tla.params.UnalignStoreParams()
                block_store = tla.params.BlockStoreParams(
                    block_stride=ub_p.stride[1][1] // ub_p.shape[1][0]
                )
                old_max_vec = tla.tile_view(
                    ub_max_rd_vl, tla.make_shape(VL), tla.make_coord(0)
                ).load()
                old_sum_vec = tla.tile_view(
                    ub_sum_rd_vl, tla.make_shape(VL), tla.make_coord(0)
                ).load()
                tla.local_mem_bar(
                    tla.params.MemType.VEC_LOAD, tla.params.MemType.VEC_STORE
                )
                if tn == KV_BLOCK:
                    n4 = rows // 4
                    for g in tla.range(n4):
                        r0 = g * 4
                        r1 = r0 + 1
                        r2 = r0 + 2
                        r3 = r0 + 3
                        e0, o0 = tla.tile_view(
                            ub_s, tla.make_shape(1, KV_BLOCK), tla.make_coord(r0, 0)
                        ).load(dintlv)
                        om0 = tla.tile_view(
                            ub_max_rd, tla.make_shape(1), tla.make_coord(r0)
                        ).load(params=brc)
                        e1, o1 = tla.tile_view(
                            ub_s, tla.make_shape(1, KV_BLOCK), tla.make_coord(r1, 0)
                        ).load(dintlv)
                        om1 = tla.tile_view(
                            ub_max_rd, tla.make_shape(1), tla.make_coord(r1)
                        ).load(params=brc)
                        e2, o2 = tla.tile_view(
                            ub_s, tla.make_shape(1, KV_BLOCK), tla.make_coord(r2, 0)
                        ).load(dintlv)
                        om2 = tla.tile_view(
                            ub_max_rd, tla.make_shape(1), tla.make_coord(r2)
                        ).load(params=brc)
                        e3, o3 = tla.tile_view(
                            ub_s, tla.make_shape(1, KV_BLOCK), tla.make_coord(r3, 0)
                        ).load(dintlv)
                        om3 = tla.tile_view(
                            ub_max_rd, tla.make_shape(1), tla.make_coord(r3)
                        ).load(params=brc)

                        e0 = tla.mul(e0, scale, mask=preg)
                        o0 = tla.mul(o0, scale, mask=preg)
                        e1 = tla.mul(e1, scale, mask=preg)
                        o1 = tla.mul(o1, scale, mask=preg)
                        e2 = tla.mul(e2, scale, mask=preg)
                        o2 = tla.mul(o2, scale, mask=preg)
                        e3 = tla.mul(e3, scale, mask=preg)
                        o3 = tla.mul(o3, scale, mask=preg)

                        m0 = tla.max(
                            om0,
                            tla.full(
                                tla.max(e0, o0, mask=preg).reduce(
                                    tla.ReductionOp.MAX, mask=preg
                                ),
                                dtype=tla.Float32,
                            ),
                            mask=preg,
                        )
                        m1 = tla.max(
                            om1,
                            tla.full(
                                tla.max(e1, o1, mask=preg).reduce(
                                    tla.ReductionOp.MAX, mask=preg
                                ),
                                dtype=tla.Float32,
                            ),
                            mask=preg,
                        )
                        m2 = tla.max(
                            om2,
                            tla.full(
                                tla.max(e2, o2, mask=preg).reduce(
                                    tla.ReductionOp.MAX, mask=preg
                                ),
                                dtype=tla.Float32,
                            ),
                            mask=preg,
                        )
                        m3 = tla.max(
                            om3,
                            tla.full(
                                tla.max(e3, o3, mask=preg).reduce(
                                    tla.ReductionOp.MAX, mask=preg
                                ),
                                dtype=tla.Float32,
                            ),
                            mask=preg,
                        )

                        tla.tile_view(
                            ub_max, tla.make_shape(1), tla.make_coord(r0)
                        ).store(m0, params=ust, mask=one_mask)
                        tla.tile_view(
                            ub_max, tla.make_shape(1), tla.make_coord(r1)
                        ).store(m1, params=ust, mask=one_mask)
                        tla.tile_view(
                            ub_max, tla.make_shape(1), tla.make_coord(r2)
                        ).store(m2, params=ust, mask=one_mask)
                        tla.tile_view(
                            ub_max, tla.make_shape(1), tla.make_coord(r3)
                        ).store(m3, params=ust, mask=one_mask)

                        e0 = tla.exp(tla.sub(e0, m0, mask=preg), mask=preg)
                        o0 = tla.exp(tla.sub(o0, m0, mask=preg), mask=preg)
                        e1 = tla.exp(tla.sub(e1, m1, mask=preg), mask=preg)
                        o1 = tla.exp(tla.sub(o1, m1, mask=preg), mask=preg)
                        e2 = tla.exp(tla.sub(e2, m2, mask=preg), mask=preg)
                        o2 = tla.exp(tla.sub(o2, m2, mask=preg), mask=preg)
                        e3 = tla.exp(tla.sub(e3, m3, mask=preg), mask=preg)
                        o3 = tla.exp(tla.sub(o3, m3, mask=preg), mask=preg)

                        s0 = tla.add(e0, o0, mask=preg).reduce(
                            tla.ReductionOp.ADD, mask=preg
                        )
                        s1 = tla.add(e1, o1, mask=preg).reduce(
                            tla.ReductionOp.ADD, mask=preg
                        )
                        s2 = tla.add(e2, o2, mask=preg).reduce(
                            tla.ReductionOp.ADD, mask=preg
                        )
                        s3 = tla.add(e3, o3, mask=preg).reduce(
                            tla.ReductionOp.ADD, mask=preg
                        )

                        tla.tile_view(
                            ub_p, tla.make_shape(1, KV_BLOCK), tla.make_coord(r0, 0)
                        ).store(
                            tla.bitwise_or(
                                e0.to(dtype_in, cast_zero, mask=preg),
                                o0.to(dtype_in, cast_one, mask=preg),
                                mask=preg_b16,
                            ),
                            params=block_store,
                        )
                        tla.tile_view(
                            ub_p, tla.make_shape(1, KV_BLOCK), tla.make_coord(r1, 0)
                        ).store(
                            tla.bitwise_or(
                                e1.to(dtype_in, cast_zero, mask=preg),
                                o1.to(dtype_in, cast_one, mask=preg),
                                mask=preg_b16,
                            ),
                            params=block_store,
                        )
                        tla.tile_view(
                            ub_p, tla.make_shape(1, KV_BLOCK), tla.make_coord(r2, 0)
                        ).store(
                            tla.bitwise_or(
                                e2.to(dtype_in, cast_zero, mask=preg),
                                o2.to(dtype_in, cast_one, mask=preg),
                                mask=preg_b16,
                            ),
                            params=block_store,
                        )
                        tla.tile_view(
                            ub_p, tla.make_shape(1, KV_BLOCK), tla.make_coord(r3, 0)
                        ).store(
                            tla.bitwise_or(
                                e3.to(dtype_in, cast_zero, mask=preg),
                                o3.to(dtype_in, cast_one, mask=preg),
                                mask=preg_b16,
                            ),
                            params=block_store,
                        )

                        tla.tile_view(
                            ub_sum, tla.make_shape(1), tla.make_coord(r0)
                        ).store(s0, params=ust, mask=one_mask)
                        tla.tile_view(
                            ub_sum, tla.make_shape(1), tla.make_coord(r1)
                        ).store(s1, params=ust, mask=one_mask)
                        tla.tile_view(
                            ub_sum, tla.make_shape(1), tla.make_coord(r2)
                        ).store(s2, params=ust, mask=one_mask)
                        tla.tile_view(
                            ub_sum, tla.make_shape(1), tla.make_coord(r3)
                        ).store(s3, params=ust, mask=one_mask)
                    for rt in tla.range(rows - n4 * 4):
                        r = n4 * 4 + rt
                        et, ot = tla.tile_view(
                            ub_s, tla.make_shape(1, KV_BLOCK), tla.make_coord(r, 0)
                        ).load(dintlv)
                        omt = tla.tile_view(
                            ub_max_rd, tla.make_shape(1), tla.make_coord(r)
                        ).load(params=brc)
                        et = tla.mul(et, scale, mask=preg)
                        ot = tla.mul(ot, scale, mask=preg)
                        mt = tla.max(
                            omt,
                            tla.full(
                                tla.max(et, ot, mask=preg).reduce(
                                    tla.ReductionOp.MAX, mask=preg
                                ),
                                dtype=tla.Float32,
                            ),
                            mask=preg,
                        )
                        tla.tile_view(
                            ub_max, tla.make_shape(1), tla.make_coord(r)
                        ).store(mt, params=ust, mask=one_mask)
                        et = tla.exp(tla.sub(et, mt, mask=preg), mask=preg)
                        ot = tla.exp(tla.sub(ot, mt, mask=preg), mask=preg)
                        st = tla.add(et, ot, mask=preg).reduce(
                            tla.ReductionOp.ADD, mask=preg
                        )
                        tla.tile_view(
                            ub_p, tla.make_shape(1, KV_BLOCK), tla.make_coord(r, 0)
                        ).store(
                            tla.bitwise_or(
                                et.to(dtype_in, cast_zero, mask=preg),
                                ot.to(dtype_in, cast_one, mask=preg),
                                mask=preg_b16,
                            ),
                            params=block_store,
                        )
                        tla.tile_view(
                            ub_sum, tla.make_shape(1), tla.make_coord(r)
                        ).store(st, params=ust, mask=one_mask)
                else:
                    n_even = (tn + 1) // 2
                    n_odd = tn // 2
                    mask_even, _ = tla.update_mask(n_even, dtype=tla.Float32)
                    mask_odd, _ = tla.update_mask(n_odd, dtype=tla.Float32)
                    for r in tla.range(rows):
                        el, ol = tla.tile_view(
                            ub_s, tla.make_shape(1, KV_BLOCK), tla.make_coord(r, 0)
                        ).load(dintlv)
                        oml = tla.tile_view(
                            ub_max_rd, tla.make_shape(1), tla.make_coord(r)
                        ).load(params=brc)
                        el = tla.where(mask_even, tla.mul(el, scale, mask=preg), minreg)
                        ol = tla.where(mask_odd, tla.mul(ol, scale, mask=preg), minreg)
                        ml = tla.max(
                            oml,
                            tla.full(
                                tla.max(el, ol, mask=preg).reduce(
                                    tla.ReductionOp.MAX, mask=preg
                                ),
                                dtype=tla.Float32,
                            ),
                            mask=preg,
                        )
                        tla.tile_view(
                            ub_max, tla.make_shape(1), tla.make_coord(r)
                        ).store(ml, params=ust, mask=one_mask)
                        el = tla.exp(tla.sub(el, ml, mask=preg), mask=preg)
                        ol = tla.exp(tla.sub(ol, ml, mask=preg), mask=preg)
                        sl = tla.add(el, ol, mask=preg).reduce(
                            tla.ReductionOp.ADD, mask=preg
                        )
                        tla.tile_view(
                            ub_p, tla.make_shape(1, KV_BLOCK), tla.make_coord(r, 0)
                        ).store(
                            tla.bitwise_or(
                                el.to(dtype_in, cast_zero, mask=preg),
                                ol.to(dtype_in, cast_one, mask=preg),
                                mask=preg_b16,
                            ),
                            params=block_store,
                        )
                        tla.tile_view(
                            ub_sum, tla.make_shape(1), tla.make_coord(r)
                        ).store(sl, params=ust, mask=one_mask)
                tla.local_mem_bar(
                    tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                )
                new_max_vec = tla.tile_view(
                    ub_max_wr_vl, tla.make_shape(VL), tla.make_coord(0)
                ).load()
                dm_vec = tla.exp(
                    tla.sub(old_max_vec, new_max_vec, mask=rmask), mask=rmask
                )
                tla.tile_view(ub_dm_wr_vl, tla.make_shape(VL), tla.make_coord(0)).store(
                    dm_vec, mask=rmask
                )
                row_sum_vec = tla.tile_view(
                    ub_sum_wr_vl, tla.make_shape(VL), tla.make_coord(0)
                ).load()
                new_sum_vec = tla.add(
                    tla.mul(old_sum_vec, dm_vec, mask=rmask), row_sum_vec, mask=rmask
                )
                tla.tile_view(
                    ub_sum_wr_vl, tla.make_shape(VL), tla.make_coord(0)
                ).store(new_sum_vec, mask=rmask)
                tla.local_mem_bar(
                    tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                )
            else:
                if tn == KV_BLOCK:
                    # === 臂① EQ_128_SFA：交付原文 ===
                    # ---- pass 1: scale + mask + running max ----
                    for r in tla.range(rows):
                        s0 = tla.tile_view(
                            ub_s, tla.make_shape(1, VL), tla.make_coord(r, 0)
                        )
                        s1 = tla.tile_view(
                            ub_s, tla.make_shape(1, VL), tla.make_coord(r, 1)
                        )
                        v0 = tla.mul(s0.load(), scale, mask=preg)
                        v1 = tla.mul(s1.load(), scale, mask=preg)
                        v0 = tla.where(preg0, v0, minreg)
                        v1 = tla.where(preg1, v1, minreg)
                        s0.store(v0, mask=preg)
                        s1.store(v1, mask=preg)
                        tmax = tla.max(v0, v1, mask=preg).reduce(
                            tla.ReductionOp.MAX, mask=preg
                        )
                        ub_max_r = tla.tile_view(
                            ub_max, tla.make_shape(1), tla.make_coord(r)
                        )
                        ub_dm_r = tla.tile_view(
                            ub_dm, tla.make_shape(1), tla.make_coord(r)
                        )
                        # F11·bc：old_max 读读槽（max(x-1) 或重灌槽），new_max 写写槽。
                        old_max = tla.tile_view(
                            ub_max_rd, tla.make_shape(1), tla.make_coord(r)
                        ).load(params=brc)
                        new_max = tla.max(old_max, tmax, mask=preg)
                        ub_max_r.store(
                            new_max,
                            params=tla.params.UnalignStoreParams(),
                            mask=one_mask,
                        )
                        ub_dm_r.store(
                            tla.exp(tla.sub(old_max, new_max, mask=preg), mask=preg),
                            params=tla.params.UnalignStoreParams(),
                            mask=one_mask,
                        )
                    tla.local_mem_bar(
                        tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                    )
                    # ---- pass 2: exp(x - max), sum, cast -> P ----
                    # T2b：值构造 = 两个 64 宽分片分别 cast 到 ZERO/ONE 槽（spread 成
                    # [lo0,hi0,lo1,hi1,...] 交错对，v1 探针 mode0 数据段实测）→ Or →
                    # 寄存器 gather（vselr，perm = [0,2,...,126,1,3,...,127] 把交错对
                    # 拉直为 0..127 同序；v1 探针 mode3 的 group0/1/7 数据段与期望逐值
                    # 一致）→ 128 宽块存（vsstb / DATA_BLOCK_COPY，OT CastStoreExp128
                    # StoreAlign<T2,DATA_BLOCK_COPY> 同一条指令形态）。
                    # 【未验证】整缓冲哨兵判据（v2 探针灌 -1 后 exact=buf 全对）待
                    # 设备恢复后确认；store 目标已按参考规范改 zN 视图 +
                    # stride[1][1]//shape[1][0] 推导 block_stride。tsum 数值语义逐字
                    # 不变（rec0 + rec1 同序）。
                    preg_i = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Int16)
                    v = tla.arange(base=0, order="increase", dtype=tla.Int16)
                    hi_sel = tla.shift_right(v, 6, mask=preg_i)
                    perm_u = tla.sub(
                        tla.mul(v, 2, mask=preg_i),
                        tla.mul(hi_sel, 127, mask=preg_i),
                        mask=preg_i,
                    ).bitcast(tla.UInt16)
                    # block_stride 单位 = 16 el DataBlock，由 zN 视图布局推导
                    # （参考 basic_mixed_store_zN.py:150-152 同式），不手填：
                    # rows=32 时 = 512//16 = 32；rows=17 时 pad16 到 32 行仍 = 32。
                    block_store = tla.params.BlockStoreParams(
                        block_stride=ub_p.stride[1][1] // ub_p.shape[1][0]
                    )
                    for r2 in tla.range(rows):
                        ub_max_b = tla.tile_view(
                            ub_max, tla.make_shape(1), tla.make_coord(r2)
                        )
                        ub_sum_b = tla.tile_view(
                            ub_sum, tla.make_shape(1), tla.make_coord(r2)
                        )
                        ub_dm_b = tla.tile_view(
                            ub_dm, tla.make_shape(1), tla.make_coord(r2)
                        )
                        mx = ub_max_b.load(params=brc)
                        dm = ub_dm_b.load(params=brc)
                        # KV_BLOCK=128 = 两个 64 宽分片：各 exp -> sum 累加 tsum，
                        # cast ZERO/ONE 槽后 Or + gather 拉直，块存进组布局 ub_p。
                        ps0 = tla.tile_view(
                            ub_s, tla.make_shape(1, VL), tla.make_coord(r2, 0)
                        )
                        e0 = tla.exp(tla.sub(ps0.load(), mx, mask=preg), mask=preg)
                        rec0 = e0.reduce(tla.ReductionOp.ADD, mask=preg)
                        ps1 = tla.tile_view(
                            ub_s, tla.make_shape(1, VL), tla.make_coord(r2, 1)
                        )
                        e1 = tla.exp(tla.sub(ps1.load(), mx, mask=preg), mask=preg)
                        rec1 = e1.reduce(tla.ReductionOp.ADD, mask=preg)
                        # sum 结合序与交付树逐字一致：两个 64 宽分片各自 reduce、
                        # 再标量相加。lse 沿用同一个 tsum ⇒ lse 用的 sum 与 out
                        # 归一化分母逐位相同（out 已对 OT 逐位，故 lse 的 log(sum)
                        # 输入也逐位）。不用 OT vf 的 Add(even,odd)+单次 Reduce 序：
                        # 该序与交付树序不同，一旦替换会同时改动 out 分母。
                        tsum = tla.add(rec0, rec1, mask=preg)
                        d0 = e0.to(dtype_in, cast_zero, mask=preg)
                        d1 = e1.to(dtype_in, cast_one, mask=preg)
                        merged = tla.bitwise_or(d0, d1, mask=preg_b16)
                        tight = tla.gather(merged, perm_u)
                        tla.tile_view(
                            ub_p, tla.make_shape(1, KV_BLOCK), tla.make_coord(r2, 0)
                        ).store(tight, params=block_store)
                        old_sum = tla.tile_view(
                            ub_sum_rd, tla.make_shape(1), tla.make_coord(r2)
                        ).load(params=brc)
                        new_sum = tla.add(
                            tla.mul(old_sum, dm, mask=preg), tsum, mask=preg
                        )
                        ub_sum_b.store(
                            new_sum,
                            params=tla.params.UnalignStoreParams(),
                            mask=one_mask,
                        )
                    tla.local_mem_bar(
                        tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                    )
                else:
                    if tn <= VL:
                        # === 臂② GT_0_AND_LTE_64_SFA：第二分片整段跳过 ===
                        # ---- 臂② pass 1：单分片（OT Impl64 :51-58，第二分片整段跳过）----
                        for r in tla.range(rows):
                            s0 = tla.tile_view(
                                ub_s, tla.make_shape(1, VL), tla.make_coord(r, 0)
                            )
                            v0 = tla.mul(s0.load(), scale, mask=preg)
                            v0 = tla.where(preg0, v0, minreg)
                            s0.store(v0, mask=preg)
                            tmax = v0.reduce(tla.ReductionOp.MAX, mask=preg)
                            ub_max_r = tla.tile_view(
                                ub_max, tla.make_shape(1), tla.make_coord(r)
                            )
                            ub_dm_r = tla.tile_view(
                                ub_dm, tla.make_shape(1), tla.make_coord(r)
                            )
                            old_max = tla.tile_view(
                                ub_max_rd, tla.make_shape(1), tla.make_coord(r)
                            ).load(params=brc)
                            new_max = tla.max(old_max, tmax, mask=preg)
                            ub_max_r.store(
                                new_max,
                                params=tla.params.UnalignStoreParams(),
                                mask=one_mask,
                            )
                            ub_dm_r.store(
                                tla.exp(
                                    tla.sub(old_max, new_max, mask=preg), mask=preg
                                ),
                                params=tla.params.UnalignStoreParams(),
                                mask=one_mask,
                            )
                        tla.local_mem_bar(
                            tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                        )
                        # ---- 臂② pass 2：单分片 exp/sum/cast + gather ----
                        # P 仍按满 128 宽块存（高半 = gather 天然给的 0）：OT 的 64 宽 H 掩码存
                        # 在 OT 侧安全只因 cube K=s2RealSize；我方 bmm2 的 K 恒 128，若只写
                        # 64 宽会把陈旧高半算进 O（§26.6）。
                        preg_i = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Int16)
                        v = tla.arange(base=0, order="increase", dtype=tla.Int16)
                        hi_sel = tla.shift_right(v, 6, mask=preg_i)
                        perm_u = tla.sub(
                            tla.mul(v, 2, mask=preg_i),
                            tla.mul(hi_sel, 127, mask=preg_i),
                            mask=preg_i,
                        ).bitcast(tla.UInt16)
                        block_store = tla.params.BlockStoreParams(
                            block_stride=ub_p.stride[1][1] // ub_p.shape[1][0]
                        )
                        for r2 in tla.range(rows):
                            ub_max_b = tla.tile_view(
                                ub_max, tla.make_shape(1), tla.make_coord(r2)
                            )
                            ub_sum_b = tla.tile_view(
                                ub_sum, tla.make_shape(1), tla.make_coord(r2)
                            )
                            ub_dm_b = tla.tile_view(
                                ub_dm, tla.make_shape(1), tla.make_coord(r2)
                            )
                            mx = ub_max_b.load(params=brc)
                            dm = ub_dm_b.load(params=brc)
                            ps0 = tla.tile_view(
                                ub_s, tla.make_shape(1, VL), tla.make_coord(r2, 0)
                            )
                            e0 = tla.exp(tla.sub(ps0.load(), mx, mask=preg), mask=preg)
                            rec0 = e0.reduce(tla.ReductionOp.ADD, mask=preg)
                            d0 = e0.to(dtype_in, cast_zero, mask=preg)
                            tight = tla.gather(d0, perm_u)
                            tla.tile_view(
                                ub_p, tla.make_shape(1, KV_BLOCK), tla.make_coord(r2, 0)
                            ).store(tight, params=block_store)
                            old_sum = tla.tile_view(
                                ub_sum_rd, tla.make_shape(1), tla.make_coord(r2)
                            ).load(params=brc)
                            new_sum = tla.add(
                                tla.mul(old_sum, dm, mask=preg), rec0, mask=preg
                            )
                            ub_sum_b.store(
                                new_sum,
                                params=tla.params.UnalignStoreParams(),
                                mask=one_mask,
                            )
                        tla.local_mem_bar(
                            tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                        )
                    else:
                        # === 臂③ GT_64_AND_LTE_128_SFA：低半满 64 道不 Select
                        # （OT TailScaleStoreMax128 :86-101 低半 preg_all），
                        # 高半按 n-64 掩码 + Select；pass2 与臂①同一形态。===
                        # ---- pass 1: scale + mask + running max ----
                        for r in tla.range(rows):
                            s0 = tla.tile_view(
                                ub_s, tla.make_shape(1, VL), tla.make_coord(r, 0)
                            )
                            s1 = tla.tile_view(
                                ub_s, tla.make_shape(1, VL), tla.make_coord(r, 1)
                            )
                            v0 = tla.mul(s0.load(), scale, mask=preg)
                            v1 = tla.mul(s1.load(), scale, mask=preg)
                            v1 = tla.where(preg1, v1, minreg)
                            s0.store(v0, mask=preg)
                            s1.store(v1, mask=preg)
                            tmax = tla.max(v0, v1, mask=preg).reduce(
                                tla.ReductionOp.MAX, mask=preg
                            )
                            ub_max_r = tla.tile_view(
                                ub_max, tla.make_shape(1), tla.make_coord(r)
                            )
                            ub_dm_r = tla.tile_view(
                                ub_dm, tla.make_shape(1), tla.make_coord(r)
                            )
                            # F11·bc：old_max 读读槽（max(x-1) 或重灌槽），new_max 写写槽。
                            old_max = tla.tile_view(
                                ub_max_rd, tla.make_shape(1), tla.make_coord(r)
                            ).load(params=brc)
                            new_max = tla.max(old_max, tmax, mask=preg)
                            ub_max_r.store(
                                new_max,
                                params=tla.params.UnalignStoreParams(),
                                mask=one_mask,
                            )
                            ub_dm_r.store(
                                tla.exp(
                                    tla.sub(old_max, new_max, mask=preg), mask=preg
                                ),
                                params=tla.params.UnalignStoreParams(),
                                mask=one_mask,
                            )
                        tla.local_mem_bar(
                            tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                        )
                        # ---- pass 2: exp(x - max), sum, cast -> P ----
                        # T2b：值构造 = 两个 64 宽分片分别 cast 到 ZERO/ONE 槽（spread 成
                        # [lo0,hi0,lo1,hi1,...] 交错对，v1 探针 mode0 数据段实测）→ Or →
                        # 寄存器 gather（vselr，perm = [0,2,...,126,1,3,...,127] 把交错对
                        # 拉直为 0..127 同序；v1 探针 mode3 的 group0/1/7 数据段与期望逐值
                        # 一致）→ 128 宽块存（vsstb / DATA_BLOCK_COPY，OT CastStoreExp128
                        # StoreAlign<T2,DATA_BLOCK_COPY> 同一条指令形态）。
                        # 【未验证】整缓冲哨兵判据（v2 探针灌 -1 后 exact=buf 全对）待
                        # 设备恢复后确认；store 目标已按参考规范改 zN 视图 +
                        # stride[1][1]//shape[1][0] 推导 block_stride。tsum 数值语义逐字
                        # 不变（rec0 + rec1 同序）。
                        preg_i = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Int16)
                        v = tla.arange(base=0, order="increase", dtype=tla.Int16)
                        hi_sel = tla.shift_right(v, 6, mask=preg_i)
                        perm_u = tla.sub(
                            tla.mul(v, 2, mask=preg_i),
                            tla.mul(hi_sel, 127, mask=preg_i),
                            mask=preg_i,
                        ).bitcast(tla.UInt16)
                        # block_stride 单位 = 16 el DataBlock，由 zN 视图布局推导
                        # （参考 basic_mixed_store_zN.py:150-152 同式），不手填：
                        # rows=32 时 = 512//16 = 32；rows=17 时 pad16 到 32 行仍 = 32。
                        block_store = tla.params.BlockStoreParams(
                            block_stride=ub_p.stride[1][1] // ub_p.shape[1][0]
                        )
                        for r2 in tla.range(rows):
                            ub_max_b = tla.tile_view(
                                ub_max, tla.make_shape(1), tla.make_coord(r2)
                            )
                            ub_sum_b = tla.tile_view(
                                ub_sum, tla.make_shape(1), tla.make_coord(r2)
                            )
                            ub_dm_b = tla.tile_view(
                                ub_dm, tla.make_shape(1), tla.make_coord(r2)
                            )
                            mx = ub_max_b.load(params=brc)
                            dm = ub_dm_b.load(params=brc)
                            # KV_BLOCK=128 = 两个 64 宽分片：各 exp -> sum 累加 tsum，
                            # cast ZERO/ONE 槽后 Or + gather 拉直，块存进组布局 ub_p。
                            ps0 = tla.tile_view(
                                ub_s, tla.make_shape(1, VL), tla.make_coord(r2, 0)
                            )
                            e0 = tla.exp(tla.sub(ps0.load(), mx, mask=preg), mask=preg)
                            rec0 = e0.reduce(tla.ReductionOp.ADD, mask=preg)
                            ps1 = tla.tile_view(
                                ub_s, tla.make_shape(1, VL), tla.make_coord(r2, 1)
                            )
                            e1 = tla.exp(tla.sub(ps1.load(), mx, mask=preg), mask=preg)
                            rec1 = e1.reduce(tla.ReductionOp.ADD, mask=preg)
                            # 同 arm ①：交付树的 sum 结合序（两分片各自 reduce + 标量加）
                            tsum = tla.add(rec0, rec1, mask=preg)
                            d0 = e0.to(dtype_in, cast_zero, mask=preg)
                            d1 = e1.to(dtype_in, cast_one, mask=preg)
                            merged = tla.bitwise_or(d0, d1, mask=preg_b16)
                            tight = tla.gather(merged, perm_u)
                            tla.tile_view(
                                ub_p, tla.make_shape(1, KV_BLOCK), tla.make_coord(r2, 0)
                            ).store(tight, params=block_store)
                            old_sum = tla.tile_view(
                                ub_sum_rd, tla.make_shape(1), tla.make_coord(r2)
                            ).load(params=brc)
                            new_sum = tla.add(
                                tla.mul(old_sum, dm, mask=preg), tsum, mask=preg
                            )
                            ub_sum_b.store(
                                new_sum,
                                params=tla.params.UnalignStoreParams(),
                                mask=one_mask,
                            )
                        tla.local_mem_bar(
                            tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                        )
        # S 已消费，归还 FIX
        tla.cross_core_set_flag(sync.s_free, tla.pipes.VECTOR, aiv_id=0)
        tla.cross_core_set_flag(sync.s_free, tla.pipes.VECTOR, aiv_id=1)
        # ---- UB P -> L1 P（两个 AIV 各写 M 的一半）----
        # T2b：zN->zN copy（参考 basic_mixed_store_zN.py:174-181 同款：
        # L1 侧 parent zN + tile_view 半行，src 为 zN 视图，MTE3 按
        # 列 fract burst 直搬、不做块内重排）。L1 dst 声明与 T2a 逐字
        # 相同（parent zN (m,KV_BLOCK) + tile_view (rows,KV_BLOCK)
        # coord (sub,0)，物理列 fract 组距 align16(m)*16 el，T2a IR
        # 1024@m=64 已证）。src = _ub_p 的标准 zN 视图（无 pad，rows=32
        # 时组距 512 el 全密）；与 OT CopyVec1ResultToL1 的 DataCopy
        # 参数 {s2BaseSize/16, halfM, vec1Srcstride-halfM, Align16Func(m)-halfM}
        # 的差异仅 srcGap 1 块 -> 0 块（省略 16 el pad），数值无影响。
        # 旧路径（RowMajor->zN 重排 copy，13.8 GB/s）废弃；随机 q 时
        # out 偏差 ~8% 的教训保留——新路径两侧同序。
        # 【未验证】#018/#013 逐位与性能待设备恢复后实测。
        l1p = tla.make_tensor(
            bufs.l1_p + slot_p * (Q_ROWS * KV_BLOCK),
            tla.make_layout_with_tag((m, KV_BLOCK), dtype_in, tla.arch.zN),
        )
        # tile_view 的 coord 是 **tile 粒度**（前端乘以 shape 得到元素偏移），
        # 所以第 sub 个 rows 行块要写 make_coord(sub, 0)；写成 sub*rows 会指向
        # 第 sub*rows 个行块（rows=32、sub=1 时元素行偏移 1024），AIV1 的 P
        # 整块落到 l1_p 之外，l1_p 的后半行始终是未初始化数据。
        l1p_rows = tla.tile_view(
            l1p, tla.make_shape(rows, KV_BLOCK), tla.make_coord(sub, 0)
        )
        sync.m_p.unlock(pipe=tla.pipes.VECTOR)
        sync.m_p.lock(pipe=tla.pipes.MTE3)
        tla.set_flag(sync.p_data_ready)
        tla.wait_flag(sync.p_data_ready)
        tla.copy(l1p_rows, ub_p)
        sync.m_p.unlock(pipe=tla.pipes.MTE3)
        tla.cross_core_set_flag(sync.p_ready, tla.pipes.MTE3, aiv_id=0)
        tla.cross_core_set_flag(sync.p_ready, tla.pipes.MTE3, aiv_id=1)


@tla.jit
def fd_unit_guard(sync):
    """E6 修：FD 单元切换的 V 侧 WAR 闸门。

    每个 FD 执行单元末，`fd_store_partial` 用 m_o16 的 MTE3 阶段把 ub_o 与
    运行统计槽读走。下一单元的首个 vec1/vec2 会在同一 UB 上重写（同核多 unit
    时）。非 FD 路径由 `bmm2_data_copy_out` 自身的 lock(VECTOR) 兜住；FD 路径
    原先没有对应闸门，MTE3 读与下一单元的 V 写可重叠。

    实测形态：在 28 核上跑 #021 S=6（30 unit）、#028 S=4（60 unit）、
    #022 S=2（64 unit）、#026 S=2（40 unit）时，上一单元的部分量被下一单元
    的 vec 写污染（matched_ratio 0.47~0.99；task0 的行段最明显）。加闸门后
    全部 1.0。闸门只需在「上一单元 fd_store 已发出」与「本单元第一个危险 V
    写」之间建序，故由 kernel 在 fd_store_partial 之后、同一个 body 里调用
    一次（每个单元边界一次，不是每 tile 一次）。

    FD_SPLITS==1 实例不调用本函数，指令流与 base 一致。
    """
    with tla.vector():
        sync.m_o16.lock(pipe=tla.pipes.VECTOR)
        sync.m_o16.unlock(pipe=tla.pipes.VECTOR)


@tla.jit
def process_vec2_seg(sync, bufs, seg, m, rows, dtype_in, fresh=0, slot_dm=0):
    """O 累加：seg0 做 O = O*dm + Otmp，其余 O += Otmp（R12）。

    fresh=1（F3 BC 块首 tile，OT smlaIsFirstVec2Base，csa_block_vector_arch35.h
    :1083-1084）时直接 O = PV（DataCopy fresh；无 dm 乘、无旧 O 读）。
    fresh 参数带运行期语义（上板验证：vec.func 内运行时 if 两臂
    产值共用 store 可编译）——bc=1 实例传入动态 is_first_base，块内后续
    tile 走 FlashUpdateLastNew/FlashUpdateNew 块内合并；bc=0 实例恒传
    Python 0 ⇒ 分支编译期消除，指令流与改前逐条一致。

    F11：slot_dm = xg2 % 2（OT softmaxExpBufs[taskIdMod2] 同键）选 dm 读槽；
    fresh 语义扩为「任务首 tile ∪ bc 块首」（kernel 传 is_first_base2）——
    任务首 tile 的 O 启动由 fresh 覆盖接管（vec_init 不再清 O，数值恒等：
    0*dm=0）。bc=1 时 is_first_base2 即原 fresh 判据，行为不变。
    """
    c0 = 0
    with tla.vector():
        sub = tla.arch.sub_block_idx()
        ub_o = tla.make_tensor(
            bufs.ub_o,
            tla.make_layout(
                tla.make_shape(rows, HEAD_DIM), tla.make_stride(HEAD_DIM, 1)
            ),
        )
        ub_dm = _dm(bufs, rows, slot_dm)
        # F17 步骤 B：O 中转改单块整宽 ub_otmp_full（fixpipe 每 tile 一次
        # SPLIT_M 写入 (m,512)；本 AIV 读自己的半行 (rows,512)，列坐标
        # seg*(D_CHUNK//VL)+c 对齐 fixpipe 写入的列段）。
        ub_otmp = tla.make_tensor(
            bufs.ub_otmp_full,
            tla.make_layout(
                tla.make_shape(rows, HEAD_DIM), tla.make_stride(HEAD_DIM, 1)
            ),
        )
        # F17 步骤 B：o_ready 由 fixpipe 每 tile 发一次（原每段一次），只在
        # seg0 等待；seg1-3 与 seg0 同流程序，无需再等。
        if seg == 0:
            tla.cross_core_wait_flag(sync.o_ready, tla.pipes.VECTOR, aiv_id=0)
            tla.cross_core_wait_flag(sync.o_ready, tla.pipes.VECTOR, aiv_id=1)
        with tla.vec.func(mode="simd"):
            preg = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            brc = tla.params.NormalLoadParams(
                load_dist=tla.params.LoadDist.DIST_BRC_B32
            )
            # F11：fresh 分支提到 r/c 循环外（每 seg 判定一次，热循环内无
            # 分支）。两臂无跨臂产值（各自完整循环），无需预初始化；
            # else 臂指令流与改前逐条一致。fresh 是运行期 Bool（bc=0 时
            # j2==0、bc=1 时块首），直接作条件——Bool==Int 比较在 vec.func
            # region-body 内类型报错（Bool vs Int32），不能写 fresh == 1。
            if fresh:
                # fresh 臂：O = PV 原样落 UB（任务首 tile / bc 块首 tile，
                # OT DataCopy fresh：无 dm 乘、无旧 O 读）。
                for r in tla.range(rows):
                    for c in tla.range(D_CHUNK // VL):
                        col = seg * (D_CHUNK // VL) + c
                        dst = tla.tile_view(
                            ub_o, tla.make_shape(1, VL), tla.make_coord(r, col)
                        )
                        ot = tla.tile_view(
                            ub_otmp, tla.make_shape(1, VL), tla.make_coord(r, col)
                        )
                        dst.store(ot.load(), mask=preg)
            else:
                for r in tla.range(rows):
                    # F11·S3 逐段 op 对位（MLIR+V-stream 计数）：dm BRC 加载原
                    # 在 c 循环内逐 chunk 重发——每 row 发 D_CHUNK//VL 次 × 4
                    # seg = 每 tile 每 row 8 次 BRC；OT FlashUpdateNew 整行
                    # （dTemplateAlign64=512 列）一次 dm 广播。dm[r] 与 c 无关，
                    # 提到 r 层：每 row 1 次，每 tile 省 7×rows 次 BRC V-op 与
                    # 每 iter 1 次 reinterpret_cast+地址标量 op。
                    dm = tla.tile_view(
                        ub_dm, tla.make_shape(1), tla.make_coord(r)
                    ).load(params=brc)
                    for c in tla.range(D_CHUNK // VL):
                        col = seg * (D_CHUNK // VL) + c
                        dst = tla.tile_view(
                            ub_o, tla.make_shape(1, VL), tla.make_coord(r, col)
                        )
                        ot = tla.tile_view(
                            ub_otmp, tla.make_shape(1, VL), tla.make_coord(r, col)
                        )
                        # dm 是 **整行** 的重标定因子（OT: FlashUpdateNew 一次处理
                        # vec2MRealSize × dTemplateAlign64，即整行 512 列都乘 dm）。
                        # 分段实现里每个 seg 各自持有 128 列，所以每个 seg 都要乘一次；
                        # 只在 seg 0 乘会让 D 列 128..511 在换 tile 时漏乘，单 tile
                        # （dm≡1）时看不出来，多 tile 时表现为「只有 V seg 0 对」。
                        acc = dst.load()
                        acc = tla.mul(acc, dm, mask=preg)
                        acc = tla.add(acc, ot.load(), mask=preg)
                        dst.store(acc, mask=preg)
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
        # F17 步骤 B：o_free 由最后一个 seg 归还（ub_otmp_full 单缓冲，本 AIV
        # 的 4 段读全部完成才允许下一 tile 的 fixpipe 覆盖写；init_sync 预置
        # x2 供首 tile 的 fixpipe）。
        if seg == 3:
            tla.cross_core_set_flag(sync.o_free, tla.pipes.VECTOR, aiv_id=0)
            tla.cross_core_set_flag(sync.o_free, tla.pipes.VECTOR, aiv_id=1)


@tla.jit
def process_vec2_full_opt(
    sync,
    bufs,
    out_tensor,
    task,
    m,
    rows,
    dtype_in,
    fresh,
    is_last,
    slot_dm=0,
    o16_aliased: tla.Constexpr[bool] = False,
):
    """Fused PV accumulation + optional last-tile normalize & L2-bypass copy_out."""
    with tla.vector():
        sub = tla.arch.sub_block_idx()
        dm_ptr = bufs.ub_dm + slot_dm * STATE_SLOT_STRIDE
        sum_ptr = bufs.ub_sum + slot_dm * STATE_SLOT_STRIDE
        m_o16 = sync.m_o16
        ub_o_t = tla.make_tensor(
            bufs.ub_o,
            tla.make_layout(tla.make_shape(rows * HEAD_DIM), tla.make_stride(1)),
        )
        ub_ot_t = tla.make_tensor(
            bufs.ub_otmp_full,
            tla.make_layout(tla.make_shape(rows * HEAD_DIM), tla.make_stride(1)),
        )
        ub_o16_t = tla.make_tensor(
            bufs.ub_o16,
            tla.make_layout(
                tla.make_shape(rows, HEAD_DIM), tla.make_stride(HEAD_DIM, 1)
            ),
        )
        ub_dm_t = tla.make_tensor(
            dm_ptr, tla.make_layout(tla.make_shape(rows), tla.make_stride(1))
        )
        ub_sum_t = tla.make_tensor(
            sum_ptr, tla.make_layout(tla.make_shape(rows), tla.make_stride(1))
        )
        tla.cross_core_wait_flag(sync.o_ready, tla.pipes.VECTOR, aiv_id=0)
        tla.cross_core_wait_flag(sync.o_ready, tla.pipes.VECTOR, aiv_id=1)

        if is_last:
            m_o16.lock(pipe=tla.pipes.VECTOR)
            with tla.vec.func(mode="simd"):
                preg = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                preg_b16 = tla.create_mask(pattern=tla.mask.ALL, dtype=dtype_in)
                one_vec = tla.full(1.0, dtype=tla.Float32)
                brc = tla.params.NormalLoadParams(
                    load_dist=tla.params.LoadDist.DIST_BRC_B32
                )
                ld_u = tla.params.NormalLoadParams(
                    post_mode=tla.params.PostMode.POST_MODE_UPDATE
                )
                cast_zero = tla.params.CastParams(
                    reg_slot=tla.params.RegSlot.ZERO,
                    sat_mode=tla.params.SatMode.SAT,
                    round_mode=tla.params.RoundMode.CAST_ROUND,
                )
                pack_store = tla.params.NormalStoreParams(
                    store_dist=tla.params.StoreDist.DIST_PACK_B32
                )
                o_rd = tla.tile_view(ub_o_t, tla.make_shape(VL), tla.make_coord(0))
                ot_rd = tla.tile_view(ub_ot_t, tla.make_shape(VL), tla.make_coord(0))
                if fresh:
                    for r in tla.range(rows):
                        den_r = tla.tile_view(
                            ub_sum_t, tla.make_shape(1), tla.make_coord(r)
                        ).load(params=brc)
                        inv_r = tla.div(one_vec, den_r, mask=preg)
                        a0, o_rd = o_rd.load(ld_u)
                        a1, o_rd = o_rd.load(ld_u)
                        a2, o_rd = o_rd.load(ld_u)
                        a3, o_rd = o_rd.load(ld_u)
                        c0 = tla.mul(a0, inv_r, mask=preg).to(
                            dtype_in, cast_zero, mask=preg
                        )
                        c1 = tla.mul(a1, inv_r, mask=preg).to(
                            dtype_in, cast_zero, mask=preg
                        )
                        c2 = tla.mul(a2, inv_r, mask=preg).to(
                            dtype_in, cast_zero, mask=preg
                        )
                        c3 = tla.mul(a3, inv_r, mask=preg).to(
                            dtype_in, cast_zero, mask=preg
                        )
                        a4, o_rd = o_rd.load(ld_u)
                        a5, o_rd = o_rd.load(ld_u)
                        a6, o_rd = o_rd.load(ld_u)
                        a7, o_rd = o_rd.load(ld_u)
                        tla.tile_view(
                            ub_o16_t, tla.make_shape(1, VL), tla.make_coord(r, 0)
                        ).store(c0, pack_store, mask=preg_b16)
                        tla.tile_view(
                            ub_o16_t, tla.make_shape(1, VL), tla.make_coord(r, 1)
                        ).store(c1, pack_store, mask=preg_b16)
                        tla.tile_view(
                            ub_o16_t, tla.make_shape(1, VL), tla.make_coord(r, 2)
                        ).store(c2, pack_store, mask=preg_b16)
                        tla.tile_view(
                            ub_o16_t, tla.make_shape(1, VL), tla.make_coord(r, 3)
                        ).store(c3, pack_store, mask=preg_b16)
                        c4 = tla.mul(a4, inv_r, mask=preg).to(
                            dtype_in, cast_zero, mask=preg
                        )
                        c5 = tla.mul(a5, inv_r, mask=preg).to(
                            dtype_in, cast_zero, mask=preg
                        )
                        c6 = tla.mul(a6, inv_r, mask=preg).to(
                            dtype_in, cast_zero, mask=preg
                        )
                        c7 = tla.mul(a7, inv_r, mask=preg).to(
                            dtype_in, cast_zero, mask=preg
                        )
                        tla.tile_view(
                            ub_o16_t, tla.make_shape(1, VL), tla.make_coord(r, 4)
                        ).store(c4, pack_store, mask=preg_b16)
                        tla.tile_view(
                            ub_o16_t, tla.make_shape(1, VL), tla.make_coord(r, 5)
                        ).store(c5, pack_store, mask=preg_b16)
                        tla.tile_view(
                            ub_o16_t, tla.make_shape(1, VL), tla.make_coord(r, 6)
                        ).store(c6, pack_store, mask=preg_b16)
                        tla.tile_view(
                            ub_o16_t, tla.make_shape(1, VL), tla.make_coord(r, 7)
                        ).store(c7, pack_store, mask=preg_b16)
                else:
                    for r in tla.range(rows):
                        den_r = tla.tile_view(
                            ub_sum_t, tla.make_shape(1), tla.make_coord(r)
                        ).load(params=brc)
                        inv_r = tla.div(one_vec, den_r, mask=preg)
                        dm_r = tla.tile_view(
                            ub_dm_t, tla.make_shape(1), tla.make_coord(r)
                        ).load(params=brc)
                        a0, o_rd = o_rd.load(ld_u)
                        t0, ot_rd = ot_rd.load(ld_u)
                        a1, o_rd = o_rd.load(ld_u)
                        t1, ot_rd = ot_rd.load(ld_u)
                        a2, o_rd = o_rd.load(ld_u)
                        t2, ot_rd = ot_rd.load(ld_u)
                        a3, o_rd = o_rd.load(ld_u)
                        t3, ot_rd = ot_rd.load(ld_u)
                        c0 = tla.mul(
                            tla.add(tla.mul(a0, dm_r, mask=preg), t0, mask=preg),
                            inv_r,
                            mask=preg,
                        ).to(dtype_in, cast_zero, mask=preg)
                        c1 = tla.mul(
                            tla.add(tla.mul(a1, dm_r, mask=preg), t1, mask=preg),
                            inv_r,
                            mask=preg,
                        ).to(dtype_in, cast_zero, mask=preg)
                        c2 = tla.mul(
                            tla.add(tla.mul(a2, dm_r, mask=preg), t2, mask=preg),
                            inv_r,
                            mask=preg,
                        ).to(dtype_in, cast_zero, mask=preg)
                        c3 = tla.mul(
                            tla.add(tla.mul(a3, dm_r, mask=preg), t3, mask=preg),
                            inv_r,
                            mask=preg,
                        ).to(dtype_in, cast_zero, mask=preg)
                        a4, o_rd = o_rd.load(ld_u)
                        t4, ot_rd = ot_rd.load(ld_u)
                        a5, o_rd = o_rd.load(ld_u)
                        t5, ot_rd = ot_rd.load(ld_u)
                        a6, o_rd = o_rd.load(ld_u)
                        t6, ot_rd = ot_rd.load(ld_u)
                        a7, o_rd = o_rd.load(ld_u)
                        t7, ot_rd = ot_rd.load(ld_u)
                        tla.tile_view(
                            ub_o16_t, tla.make_shape(1, VL), tla.make_coord(r, 0)
                        ).store(c0, pack_store, mask=preg_b16)
                        tla.tile_view(
                            ub_o16_t, tla.make_shape(1, VL), tla.make_coord(r, 1)
                        ).store(c1, pack_store, mask=preg_b16)
                        tla.tile_view(
                            ub_o16_t, tla.make_shape(1, VL), tla.make_coord(r, 2)
                        ).store(c2, pack_store, mask=preg_b16)
                        tla.tile_view(
                            ub_o16_t, tla.make_shape(1, VL), tla.make_coord(r, 3)
                        ).store(c3, pack_store, mask=preg_b16)
                        c4 = tla.mul(
                            tla.add(tla.mul(a4, dm_r, mask=preg), t4, mask=preg),
                            inv_r,
                            mask=preg,
                        ).to(dtype_in, cast_zero, mask=preg)
                        c5 = tla.mul(
                            tla.add(tla.mul(a5, dm_r, mask=preg), t5, mask=preg),
                            inv_r,
                            mask=preg,
                        ).to(dtype_in, cast_zero, mask=preg)
                        c6 = tla.mul(
                            tla.add(tla.mul(a6, dm_r, mask=preg), t6, mask=preg),
                            inv_r,
                            mask=preg,
                        ).to(dtype_in, cast_zero, mask=preg)
                        c7 = tla.mul(
                            tla.add(tla.mul(a7, dm_r, mask=preg), t7, mask=preg),
                            inv_r,
                            mask=preg,
                        ).to(dtype_in, cast_zero, mask=preg)
                        tla.tile_view(
                            ub_o16_t, tla.make_shape(1, VL), tla.make_coord(r, 4)
                        ).store(c4, pack_store, mask=preg_b16)
                        tla.tile_view(
                            ub_o16_t, tla.make_shape(1, VL), tla.make_coord(r, 5)
                        ).store(c5, pack_store, mask=preg_b16)
                        tla.tile_view(
                            ub_o16_t, tla.make_shape(1, VL), tla.make_coord(r, 6)
                        ).store(c6, pack_store, mask=preg_b16)
                        tla.tile_view(
                            ub_o16_t, tla.make_shape(1, VL), tla.make_coord(r, 7)
                        ).store(c7, pack_store, mask=preg_b16)
                tla.local_mem_bar(
                    tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                )
            m_o16.unlock(pipe=tla.pipes.VECTOR)
            # When o16_aliased=True, ub_o16 aliases ub_otmp_full (not ub_o), so ub_o is
            # immediately free on PIPE_V for the next token's tile 0 (dst_is_o=True).
            tla.cross_core_set_flag(sync.o_free, tla.pipes.VECTOR, aiv_id=0)
            tla.cross_core_set_flag(sync.o_free, tla.pipes.VECTOR, aiv_id=1)
            m_o16.lock(pipe=tla.pipes.MTE3)
            if sub * rows < m:
                gm_o = tla.make_tensor(
                    out_tensor.ptr + task * (m * HEAD_DIM) + (sub * rows * HEAD_DIM),
                    tla.make_layout_with_tag(
                        (1, rows * HEAD_DIM), dtype_in, tla.arch.RowMajor
                    ),
                )
                ub_o16_1d = tla.make_tensor(
                    bufs.ub_o16,
                    tla.make_layout_with_tag(
                        (1, rows * HEAD_DIM), dtype_in, tla.arch.RowMajor
                    ),
                )
                tla.copy(gm_o, ub_o16_1d, cache_policy=tla.params.CachePolicy.DISABLE)
            m_o16.unlock(pipe=tla.pipes.MTE3)
        else:
            if fresh:
                if tla.const_expr(o16_aliased):
                    # Hardware MTE3 FIFO ensures prior token's copy_out from ub_o16
                    # (= ub_otmp_full[0:32KB]) completes before tile 1 writes ub_otmp_full.
                    tla.cross_core_set_flag(sync.o_free, tla.pipes.MTE3, aiv_id=0)
                    tla.cross_core_set_flag(sync.o_free, tla.pipes.MTE3, aiv_id=1)
                else:
                    tla.cross_core_set_flag(sync.o_free, tla.pipes.VECTOR, aiv_id=0)
                    tla.cross_core_set_flag(sync.o_free, tla.pipes.VECTOR, aiv_id=1)
            else:
                with tla.vec.func(mode="simd"):
                    preg = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                    brc = tla.params.NormalLoadParams(
                        load_dist=tla.params.LoadDist.DIST_BRC_B32
                    )
                    ld_u = tla.params.NormalLoadParams(
                        post_mode=tla.params.PostMode.POST_MODE_UPDATE
                    )
                    st_u = tla.params.NormalStoreParams(
                        post_mode=tla.params.PostMode.POST_MODE_UPDATE
                    )
                    o_rd = tla.tile_view(ub_o_t, tla.make_shape(VL), tla.make_coord(0))
                    o_wr = tla.tile_view(ub_o_t, tla.make_shape(VL), tla.make_coord(0))
                    ot_rd = tla.tile_view(
                        ub_ot_t, tla.make_shape(VL), tla.make_coord(0)
                    )
                    for r in tla.range(rows):
                        dm_r = tla.tile_view(
                            ub_dm_t, tla.make_shape(1), tla.make_coord(r)
                        ).load(params=brc)
                        a0, o_rd = o_rd.load(ld_u)
                        t0, ot_rd = ot_rd.load(ld_u)
                        a1, o_rd = o_rd.load(ld_u)
                        t1, ot_rd = ot_rd.load(ld_u)
                        a2, o_rd = o_rd.load(ld_u)
                        t2, ot_rd = ot_rd.load(ld_u)
                        a3, o_rd = o_rd.load(ld_u)
                        t3, ot_rd = ot_rd.load(ld_u)
                        r0 = tla.add(tla.mul(a0, dm_r, mask=preg), t0, mask=preg)
                        r1 = tla.add(tla.mul(a1, dm_r, mask=preg), t1, mask=preg)
                        r2 = tla.add(tla.mul(a2, dm_r, mask=preg), t2, mask=preg)
                        r3 = tla.add(tla.mul(a3, dm_r, mask=preg), t3, mask=preg)
                        a4, o_rd = o_rd.load(ld_u)
                        t4, ot_rd = ot_rd.load(ld_u)
                        a5, o_rd = o_rd.load(ld_u)
                        t5, ot_rd = ot_rd.load(ld_u)
                        a6, o_rd = o_rd.load(ld_u)
                        t6, ot_rd = ot_rd.load(ld_u)
                        a7, o_rd = o_rd.load(ld_u)
                        t7, ot_rd = ot_rd.load(ld_u)
                        o_wr = o_wr.store(r0, st_u, mask=preg)
                        o_wr = o_wr.store(r1, st_u, mask=preg)
                        o_wr = o_wr.store(r2, st_u, mask=preg)
                        o_wr = o_wr.store(r3, st_u, mask=preg)
                        r4 = tla.add(tla.mul(a4, dm_r, mask=preg), t4, mask=preg)
                        r5 = tla.add(tla.mul(a5, dm_r, mask=preg), t5, mask=preg)
                        r6 = tla.add(tla.mul(a6, dm_r, mask=preg), t6, mask=preg)
                        r7 = tla.add(tla.mul(a7, dm_r, mask=preg), t7, mask=preg)
                        o_wr = o_wr.store(r4, st_u, mask=preg)
                        o_wr = o_wr.store(r5, st_u, mask=preg)
                        o_wr = o_wr.store(r6, st_u, mask=preg)
                        o_wr = o_wr.store(r7, st_u, mask=preg)
                    tla.local_mem_bar(
                        tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                    )
                tla.cross_core_set_flag(sync.o_free, tla.pipes.VECTOR, aiv_id=0)
                tla.cross_core_set_flag(sync.o_free, tla.pipes.VECTOR, aiv_id=1)


# ---------------------------------------------------------------------------
# F3 batch-consistency（IS_BATCH_CONSISTENCY=1 实例专用，仅 swa kernel 调用；
# =0 实例 IR 不含下列任何一条）。OT 锚点：
#   csa_block_vector_arch35.h:940-975   ProcessVec1（BC 每 tile update）
#   csa_block_vector_arch35.h:779-782   InitVec1SoftmaxFromSinks !includeSink
#   csa_block_vector_arch35.h:1079-1133 ProcessVec2（BC fresh + LastDivNew）
#   csa_block_vector_arch35.h:1000-1090 ReduceIntraBlockAndStage
#   common/flash_decode.h:458-509       MergeStagedAndCurrentChunk
#   common/flash_decode.h:326-341       MergeTwoInputsWithLse
#   vf/vf_flash_decode_arch35.h         ComputeScaleValue_VF / ReduceFinalRes_*
# 偏离 D-F3a~f 见 ITERATIONS.md Iter 23。
# ---------------------------------------------------------------------------
def _bc_rows8(rows):
    """BC stats 区宽度：Align8(rows)（MTE GM<->UB 32B 对齐，D-F2a 审计同款；
    rows ∈ {1,4}（#004/#007）时 rows*4B 不足 32B）。"""
    return (rows + 7) // 8 * 8


def _bc_slot_stride(rows):
    """每 AIV 槽宽（fp32 元素）：[O rows*512][max rows8][sum rows8]。
    rows*512*4B 与 2*rows8*4B 均为 32B 倍数 => 槽基址恒 32B 对齐。"""
    return rows * HEAD_DIM + _bc_rows8(rows) * 2


@tla.jit
def bc_reinit_stats(bufs, rows, slot=0):
    """F3 BC t>=1 tile 的 vec1 初值重灌（OT InitVec1SoftmaxFromSinks 的
    !includeSink 分支，csa_block_vector_arch35.h:779-782：BC 下 includeSink 追加
    reduceBlockId==0 限定，t>=1 时 Duplicate max=negativeFloat / sum=0；随后每
    tile 都走 update 分支 ComputeVec1Softmax<true>，:945-958）。t==0 不调用
    （初值来自 vec_init_from_sinks：sinks→(sinks,1.0)，无 sinks→(negFloat,0)）。

    F11·bc：写槽 slot = 本 tile 的 xg%2（kernel 传 slot_s1）——紧随其后的
    vec1(t) 以首 tile 特例读槽 t%2（slot_max_rd/slot_sum_rd = slot_s1），
    拿到的正是本重灌值；前一 tile 的链槽 (t-1)%2 不被触碰。
    """
    with tla.vector():
        ub_max = _max(bufs, rows, slot)
        ub_sum = _sum(bufs, rows, slot)
        with tla.vec.func(mode="simd"):
            one_mask, _ = tla.update_mask(1, dtype=tla.Float32)
            zero = tla.full(0.0, dtype=tla.Float32)
            minreg = tla.full(MIN_F, dtype=tla.Float32)
            for r in tla.range(rows):
                tla.tile_view(ub_max, tla.make_shape(1), tla.make_coord(r)).store(
                    minreg, params=tla.params.UnalignStoreParams(), mask=one_mask
                )
                tla.tile_view(ub_sum, tla.make_shape(1), tla.make_coord(r)).store(
                    zero, params=tla.params.UnalignStoreParams(), mask=one_mask
                )
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)


@tla.jit
def bc_div_cur(bufs, rows, slot=0):
    """F3 BC 每 tile 的 LastDivNew（OT ProcessVec2 的 smlaIsFirstVec2Base 分支，
    csa_block_vector_arch35.h:1130-1133：LastDivNew(O, O, softmaxFinalSumBufs)）：
    当前 tile 归一化 O = raw O / 本 tile new_sum。merge 的 cur 侧与 stage 的槽内
    O 都取归一化形态（MergeStagedAndCurrentChunk 注释：previous/current 均为
    normalized，flash_decode.h:455-457）。

    F11·bc：读槽 slot = 本 tile 的 xg%2（kernel 传 slot_p2）——sum(t) 由
    vec1(t) 写槽 t%2，交换后 div(t) 晚于 vec1(t+1) 执行但 (t+1)%2 异槽。
    """
    with tla.vector():
        ub_o = tla.make_tensor(
            bufs.ub_o,
            tla.make_layout(
                tla.make_shape(rows, HEAD_DIM), tla.make_stride(HEAD_DIM, 1)
            ),
        )
        ub_sum = _sum(bufs, rows, slot)
        with tla.vec.func(mode="simd"):
            preg = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            brc = tla.params.NormalLoadParams(
                load_dist=tla.params.LoadDist.DIST_BRC_B32
            )
            for r in tla.range(rows):
                den = tla.tile_view(ub_sum, tla.make_shape(1), tla.make_coord(r)).load(
                    params=brc
                )
                for c2 in (0, 1, 2, 3, 4, 5, 6, 7):
                    dst = tla.tile_view(
                        ub_o, tla.make_shape(1, VL), tla.make_coord(r, c2)
                    )
                    dst.store(tla.div(dst.load(), den, mask=preg), mask=preg)
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)


@tla.jit
def bc_stage_cur(sync, bufs, bc_ws, rows, slot=0):
    """F3 BC 非末 tile 的 staging（OT StageVec2PartialOAndWait + StageVec1Lse +
    SetFlag MTE3_MTE2 INTRALSE/INTRAATTN 的合并形态，D-F3d：OT 的 LSE stage 在
    vec1 末、O stage 在 vec2 末分两处，本仓合并为 tile 尾一次拷出）。
    槽号 = block_idx*2 + sub_block_idx，每 AIV 一槽（OT (coreIdx<<1)+
    multiCoreIdxMod2 同款）；D-F3b：OT max/sum 每 row broadcast x8，本仓行内标量。
    同步：bc_v3m（V 写完 -> MTE3 拷出）屏障对；拷出后 bc_m3m2（供下 tile merge
    的 MTE2 读）+ bc_m3v（MTE3 完成 -> V 返回，OT AndWait 的 MTE3_V，防 t+1 的
    vec2 fresh 覆写 ub_o）。

    F11·bc：max/sum 读槽 slot = 本 tile 的 xg%2（kernel 传 slot_p2）——stage
    紧随 merge_prev(t) 之后，读到的是 merge 就地写回本 tile 槽的 merged 值。
    """
    with tla.vector():
        slot_ws = tla.arch.block_idx() * 2 + tla.arch.sub_block_idx()
        rows8 = _bc_rows8(rows)
        base = slot_ws * _bc_slot_stride(rows)
        # V 侧归一化/合并的 store 完成 -> MTE3 才能拷出（p_data_ready 屏障同款）
        tla.set_flag(sync.bc_v3m)
        tla.wait_flag(sync.bc_v3m)
        gm_o = tla.make_tensor(
            bc_ws.ptr + base,
            tla.make_layout(
                tla.make_shape(rows, HEAD_DIM), tla.make_stride(HEAD_DIM, 1)
            ),
        )
        tla.copy(
            gm_o,
            tla.make_tensor(
                bufs.ub_o,
                tla.make_layout(
                    tla.make_shape(rows, HEAD_DIM), tla.make_stride(HEAD_DIM, 1)
                ),
            ),
        )
        gm_max = tla.make_tensor(
            bc_ws.ptr + base + rows * HEAD_DIM,
            tla.make_layout(tla.make_shape(rows8), tla.make_stride(1)),
        )
        tla.copy(gm_max, _max(bufs, rows8, slot))
        gm_sum = tla.make_tensor(
            bc_ws.ptr + base + rows * HEAD_DIM + rows8,
            tla.make_layout(tla.make_shape(rows8), tla.make_stride(1)),
        )
        tla.copy(gm_sum, _sum(bufs, rows8, slot))
        # MTE3 管道序：set 在三笔 copy 之后 => 槽全部落 GM 才置起
        tla.set_flag(sync.bc_m3m2)  # 供下一 tile merge 的 MTE2 读（MTE3->MTE2）
        tla.set_flag(sync.bc_m3v)  # MTE3 完成 -> V 返回（源 UB 复用安全）
        tla.wait_flag(sync.bc_m3v)


@tla.jit
def bc_merge_seg(sync, bufs, bc_ws, rows, seg, rows8, base):
    """bc_merge_prev 的单列段（静态展开 4 次）：MTE2 拷入 prev O 的 seg 段
    （rows x D_CHUNK，行距 512）-> V 加权合并写回 ub_o 同列段。
    注意：不开 tla.vector 作用域（vector 不可嵌套），仅由 bc_merge_prev 的
    vector 作用域内调用。

    算术（OT ReduceFinalRes_const_VF，Mul 与 Add 分开两条非 FMA）：
      dst = w_prev * prevO_norm + w_cur * curO_norm（cur=ub_o 归一化后）
    权重 [w_prev | w_cur] 连续两段各 rows8（OT blockMaxUb/SumUb 的 [previous,
    current] 布局同款）。
    同步：bc_v2m 信用（V 读完上一段 ub_bc_prev -> 本段 MTE2 才能覆写；init 预置
    1）-> bc_m2v 屏障对（拷入完成 -> V 读）-> 段尾归还 bc_v2m。
    """
    tla.wait_flag(sync.bc_v2m)
    gm_seg = tla.make_tensor(
        bc_ws.ptr + base + seg * D_CHUNK,
        tla.make_layout(tla.make_shape(rows, D_CHUNK), tla.make_stride(HEAD_DIM, 1)),
    )
    tla.copy(
        tla.make_tensor(
            bufs.ub_bc_prev,
            tla.make_layout(tla.make_shape(rows, D_CHUNK), tla.make_stride(D_CHUNK, 1)),
        ),
        gm_seg,
    )
    tla.set_flag(sync.bc_m2v)
    tla.wait_flag(sync.bc_m2v)
    with tla.vec.func(mode="simd"):
        preg = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
        brc = tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_BRC_B32)
        ub_o = tla.make_tensor(
            bufs.ub_o,
            tla.make_layout(
                tla.make_shape(rows, HEAD_DIM), tla.make_stride(HEAD_DIM, 1)
            ),
        )
        ub_bc_prev = tla.make_tensor(
            bufs.ub_bc_prev,
            tla.make_layout(tla.make_shape(rows, D_CHUNK), tla.make_stride(D_CHUNK, 1)),
        )
        ub_bc_w = _stats(bufs.ub_bc_w, rows8 * 2)
        for r in tla.range(rows):
            w_p = tla.tile_view(ub_bc_w, tla.make_shape(1), tla.make_coord(r)).load(
                params=brc
            )
            w_c = tla.tile_view(
                ub_bc_w, tla.make_shape(1), tla.make_coord(rows8 + r)
            ).load(params=brc)
            for c2 in (0, 1):
                col = seg * (D_CHUNK // VL) + c2
                dst = tla.tile_view(ub_o, tla.make_shape(1, VL), tla.make_coord(r, col))
                prev = tla.tile_view(
                    ub_bc_prev, tla.make_shape(1, VL), tla.make_coord(r, c2)
                ).load()
                cur = dst.load()
                part = tla.mul(w_p, prev, mask=preg)
                dst.store(
                    tla.add(part, tla.mul(w_c, cur, mask=preg), mask=preg),
                    mask=preg,
                )
        tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
    tla.set_flag(sync.bc_v2m)


@tla.jit
def bc_merge_prev(sync, bufs, bc_ws, rows, slot=0):
    """F3 BC t>=1 tile 的跨 tile 合并（OT ReduceIntraBlockAndStage 的
    MergeStagedAndCurrentChunk 链；每 tile 左折叠一次：
    state = (prev merged) (+) cur，merged 以 (normalized O, merged_lse, 1.0)
    回存，flash_decode.h:459-462 注释同款）。

    算术对位 ComputeScaleValue_VF（vf_flash_decode_arch35.h）：
      m = max(lse_prev, lse_cur)；e_i = exp(lse_i - m)；num_i = s_i * e_i；
      Z = num_prev + num_cur；merged_lse = Log(Z) + m；w_i = num_i / Z
    （OT 从 -FLT_MAX 起 Max 与 0+ 起步 Add 在 lse/sum 恒有限非负时逐位恒等，
    D-F3e/f）。

    F11·bc：lse_c/s_c 读槽 slot = 本 tile 的 xg%2（kernel 传 slot_p2）——
    vec1(t) 写槽 t%2，交换后 merge(t) 晚于 vec1(t+1)/reinit(t+1)（写异槽
    (t+1)%2）执行；merged_lse/1.0 就地写回同槽 t%2，紧随的 stage_cur(t) 读
    同槽拿到 merged 值，随后 vec1(t+2) 写槽 (t+2)%2 = t%2 才覆写（其间唯一
    读者 stage_cur 已消费，staged 已落 GM）。
    """
    with tla.vector():
        slot_ws = tla.arch.block_idx() * 2 + tla.arch.sub_block_idx()
        rows8 = _bc_rows8(rows)
        base = slot_ws * _bc_slot_stride(rows)
        ub_bc_max = _stats(bufs.ub_bc_max, rows8)
        ub_bc_sum = _stats(bufs.ub_bc_sum, rows8)
        ub_max = _max(bufs, rows, slot)
        ub_sum = _sum(bufs, rows, slot)
        # ---- 等 stage 拷出完成（MTE3->MTE2），MTE2 拷入 prev max/sum ----
        tla.wait_flag(sync.bc_m3m2)
        gm_max = tla.make_tensor(
            bc_ws.ptr + base + rows * HEAD_DIM,
            tla.make_layout(tla.make_shape(rows8), tla.make_stride(1)),
        )
        gm_sum = tla.make_tensor(
            bc_ws.ptr + base + rows * HEAD_DIM + rows8,
            tla.make_layout(tla.make_shape(rows8), tla.make_stride(1)),
        )
        tla.copy(ub_bc_max, gm_max)
        tla.copy(ub_bc_sum, gm_sum)
        tla.set_flag(sync.bc_m2v)
        tla.wait_flag(sync.bc_m2v)
        # ---- V 逐行 merge 权重；merged_lse/1.0 回写 ub_max/ub_sum ----
        with tla.vec.func(mode="simd"):
            preg = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            brc = tla.params.NormalLoadParams(
                load_dist=tla.params.LoadDist.DIST_BRC_B32
            )
            one_mask, _ = tla.update_mask(1, dtype=tla.Float32)
            one = tla.full(1.0, dtype=tla.Float32)
            ub_bc_w = _stats(bufs.ub_bc_w, rows8 * 2)
            for r in tla.range(rows):
                lse_p = tla.tile_view(
                    ub_bc_max, tla.make_shape(1), tla.make_coord(r)
                ).load(params=brc)
                s_p = tla.tile_view(
                    ub_bc_sum, tla.make_shape(1), tla.make_coord(r)
                ).load(params=brc)
                lse_c = tla.tile_view(
                    ub_max, tla.make_shape(1), tla.make_coord(r)
                ).load(params=brc)
                s_c = tla.tile_view(ub_sum, tla.make_shape(1), tla.make_coord(r)).load(
                    params=brc
                )
                m_new = tla.max(lse_p, lse_c, mask=preg)
                e_p = tla.exp(tla.sub(lse_p, m_new, mask=preg), mask=preg)
                e_c = tla.exp(tla.sub(lse_c, m_new, mask=preg), mask=preg)
                num_p = tla.mul(s_p, e_p, mask=preg)
                num_c = tla.mul(s_c, e_c, mask=preg)
                z_sum = tla.add(num_p, num_c, mask=preg)
                w_p = tla.div(num_p, z_sum, mask=preg)
                w_c = tla.div(num_c, z_sum, mask=preg)
                merged_lse = tla.add(tla.log(z_sum, mask=preg), m_new, mask=preg)
                tla.tile_view(ub_max, tla.make_shape(1), tla.make_coord(r)).store(
                    merged_lse, params=tla.params.UnalignStoreParams(), mask=one_mask
                )
                tla.tile_view(ub_sum, tla.make_shape(1), tla.make_coord(r)).store(
                    one, params=tla.params.UnalignStoreParams(), mask=one_mask
                )
                tla.tile_view(ub_bc_w, tla.make_shape(1), tla.make_coord(r)).store(
                    w_p, params=tla.params.UnalignStoreParams(), mask=one_mask
                )
                tla.tile_view(
                    ub_bc_w, tla.make_shape(1), tla.make_coord(rows8 + r)
                ).store(w_c, params=tla.params.UnalignStoreParams(), mask=one_mask)
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
        # ---- per-seg prevO 拷入 + 加权合并（静态展开 4 段）----
        # 各段 vec.func 尾部自带 VEC_STORE->VEC_LOAD bar（覆盖 ub_o 段 store ->
        # 下一段 load 与终局 copyout 的 load），段外不能再发 local_mem_bar。
        for seg in (0, 1, 2, 3):
            bc_merge_seg(sync, bufs, bc_ws, rows, seg, rows8, base)


@tla.jit
def bmm2_data_copy_out(
    sync,
    bufs,
    out_tensor,
    task,
    m,
    rows,
    dtype_in,
    row_base=None,
    valid_rows=None,
    odd_mode=0,
    slot_sum=0,
):
    """O / sum -> CAST_ROUND -> GM（R13/R14）。

    odd_mode=1 时是奇数 G（#013 h=65）路径：m = Align2(halfG)（34），AIV 按
    17/17 行处理，两个半区 valid_rows 分别为 33/32；写出的行数取运行时上界
    `n_rows = clamp(valid_rows - sub*rows, 0, rows)`，逐行写 GM，丢弃行不写 out。
    行基址 row_base = tok*gSize + goIdx
    （OT kvcache.h:243-248 的 attentionOutOffset 以 goIdx*dSizeV 起、再加
    subBlockIdx*firstHalfMRealSize*dSizeV；每行对应 OT Bmm2DataCopyOut 的
    blockCount=vec2MRealSize 逐行 DataCopyPad，csa_block_vector_arch35.h:1277-1297）。
    偶数 m 传 odd_mode=0，保持原来的整块 copy 语义。
    """
    c0 = 0
    with tla.vector():
        sub = tla.arch.sub_block_idx()
        ub_o = tla.make_tensor(
            bufs.ub_o,
            tla.make_layout(
                tla.make_shape(rows, HEAD_DIM), tla.make_stride(HEAD_DIM, 1)
            ),
        )
        ub_o16 = tla.make_tensor(
            bufs.ub_o16,
            tla.make_layout_with_tag((rows, HEAD_DIM), dtype_in, tla.arch.RowMajor),
        )
        # F11：归一化分母读 sum 槽 slot_sum = (P末)%2（宿主 tile 的 xg%2）；
        # 同 body 先发的 vec_init(T首=P末+1) 重灌异槽，不互踩。
        ub_sum = _sum(bufs, rows, slot_sum)
        sync.m_o16.lock(pipe=tla.pipes.VECTOR)
        with tla.vec.func(mode="simd"):
            preg = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            preg_b16 = tla.create_mask(pattern=tla.mask.ALL, dtype=dtype_in)
            brc = tla.params.NormalLoadParams(
                load_dist=tla.params.LoadDist.DIST_BRC_B32
            )
            cast_zero = tla.params.CastParams(
                reg_slot=tla.params.RegSlot.ZERO,
                sat_mode=tla.params.SatMode.SAT,
                round_mode=tla.params.RoundMode.CAST_ROUND,
            )
            pack_store = tla.params.NormalStoreParams(
                store_dist=tla.params.StoreDist.DIST_PACK_B32
            )
            # F38：终局归一 div 与 CAST_ROUND 融合成一趟——f32 结果不再落 UB 再
            # 读回。逐 chunk 4 条 op（load / div / cast / pack-store），对照两趟的
            # 6 条；数值路径不变（同一个 Div 结果直接 CAST_ROUND，中间 f32 只是
            # 被旁路掉，没有改变任何舍入点）。
            for rf in tla.range(rows):
                denf = tla.tile_view(
                    ub_sum, tla.make_shape(1), tla.make_coord(rf)
                ).load(params=brc)
                for cf in tla.range(HEAD_DIM // VL):
                    srcf = tla.tile_view(
                        ub_o, tla.make_shape(1, VL), tla.make_coord(rf, cf)
                    )
                    d16f = tla.tile_view(
                        ub_o16, tla.make_shape(1, VL), tla.make_coord(rf, cf)
                    )
                    d16f.store(
                        tla.div(srcf.load(), denf, mask=preg).to(
                            dtype_in, cast_zero, mask=preg
                        ),
                        pack_store,
                        mask=preg_b16,
                    )
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
        sync.m_o16.unlock(pipe=tla.pipes.VECTOR)
        sync.m_o16.lock(pipe=tla.pipes.MTE3)
        if odd_mode == 0:
            # OT firstHalfMRealSize=(m+1)/2、secondHalf=m-firstHalf：m 为奇数时第二子块
            # 行数为 0（m=1 即 h=1 用例），不能写 GM —— 否则会覆盖下一个 token 的首行。
            # 偶数 m 下 sub*rows < m 恒真，与原来逐字等价。
            if sub * rows < m:
                gm_o = tla.make_tensor(
                    out_tensor.ptr + task * (m * HEAD_DIM) + (sub * rows * HEAD_DIM),
                    tla.make_layout_with_tag(
                        (rows, HEAD_DIM), dtype_in, tla.arch.RowMajor
                    ),
                )
                tla.copy(gm_o, ub_o16)
        else:
            # 奇数 G：只写本子块的 valid 行；行基址 row_base + sub*rows + r
            # （OT Bmm2DataCopyOut: blockCount=vec2MRealSize，dst 偏移
            #  attentionOutOffset + subBlockIdx*firstHalfMRealSize*dSizeV）。
            # 用运行时行数上界，不在循环体内放动态 if。
            n_rows = valid_rows - sub * rows
            if n_rows > rows:
                n_rows = tla.as_numeric(rows)
            if n_rows < 0:
                n_rows = tla.as_numeric(0)
            for r in tla.range(n_rows):
                gm_r = tla.make_tensor(
                    out_tensor.ptr + (row_base + sub * rows + r) * HEAD_DIM,
                    tla.make_layout_with_tag(
                        (1, HEAD_DIM), dtype_in, tla.arch.RowMajor
                    ),
                )
                src_r = tla.tile_view(
                    ub_o16, tla.make_shape(1, HEAD_DIM), tla.make_coord(r, c0)
                )
                tla.copy(gm_r, src_r)
        sync.m_o16.unlock(pipe=tla.pipes.MTE3)


@tla.jit
def lse_copy_out(
    sync, bufs, lse_tensor, task, m, rows, slot=0, row_base=None, valid_rows=None
):
    """softmax_lse = Log(sum) + max（OT CopyOutVec1Lse → ComputeLse<float>）。

    对照 sparse_flash_mla_csa_block_vector_arch35.h: CopyOutVec1Lse (896-919)、
    attention/common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_sfa.h:235
      ComputeLseVF: outLse = Log(sum) + max（fp32，无 epsilon）
    触发条件 (CopyOutVec1Lse)：isSoftmaxLseEnable && halfMRealSize>0 && s2LoopCount==s2LoopLimit
    ⇒ 在全部 tile 处理完之后写一次，写 halfMRealSize 个 fp32。
    偏移 (kvcache.h:249-267 TND)：n2o*s1Size*gSize + (seqOffset+sOuterOffset)*gSize
    + (subBlockIdx==1 ? firstHalfMRealSize : 0)；n2=1 时即 row_base + sub*rows。

    R37 修（os 非逐位根因，见 ITERATIONS §R37）：
    1) `slot` = 宿主 tile 的全局序 xg%2 —— running max/sum 的终态就在这个槽
       （与 bmm2_data_copy_out 归一化读的 slot_sum 同一个），原先固定读槽 0
       只在"末 tile 全局序为偶"时正确，是 030/037/033/040 等例的主因。
    2) 行数按 OT halfMRealSize：sub0 写 rows、sub1 写 m-rows（m 奇数时少一行），
       原先两个子块都写 rows 行，m=1（#032）时 sub1 会覆盖下一 token 的行
       （= 该例逐位率恰 0.5 的根因）。
    3) `row_base` = OT 的 (seqOffset+sOuterOffset)*gSize 行基址；奇 head_num +
       SPLIT_G 时 goIdx = half*halfG ≠ half*m（#013 h=65 类）。
    """
    c0 = 0
    with tla.vector():
        sub = tla.arch.sub_block_idx()
        # running max/sum 读槽必须与终态槽一致（R37 根因 1）
        ub_max = _max(bufs, rows, slot)
        ub_sum = _sum(bufs, rows, slot)
        # 复用 dm 本槽做落盘暂存：vec2 已在同迭代先读走本槽 dm（调用点在
        # bmm2_data_copy_out 之后），下一写者是 vec1(tile+2)（本槽下一迭代的
        # 下一轮），落盘后由 lse_mte3_v（MTE3→V）挡到搬完再继续（WAR）。
        ub_lse = _dm(bufs, rows, slot)
        with tla.vec.func(mode="simd"):
            preg = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            one_mask, _ = tla.update_mask(1, dtype=tla.Float32)
            brc = tla.params.NormalLoadParams(
                load_dist=tla.params.LoadDist.DIST_BRC_B32
            )
            for r in tla.range(rows):
                s = tla.tile_view(ub_sum, tla.make_shape(1), tla.make_coord(r)).load(
                    params=brc
                )
                mx = tla.tile_view(ub_max, tla.make_shape(1), tla.make_coord(r)).load(
                    params=brc
                )
                # 与 OT 的 ComputeLseVF 同序：先 Log(sum)，再 Add(max)
                val = tla.add(tla.log(s, mask=preg), mx, mask=preg)
                tla.tile_view(ub_lse, tla.make_shape(1), tla.make_coord(r)).store(
                    val,
                    params=tla.params.UnalignStoreParams(),
                    mask=one_mask,
                )
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
        tla.set_flag(sync.lse_v_mte3)
        tla.wait_flag(sync.lse_v_mte3)
        base = task * m if row_base is None else row_base
        if valid_rows is None and m % 2 == 0:
            # 偶数 m：两个子块各 rows 行，整块搬（与 OT 每子块一次 DataCopyPad 同形）
            gm_lse = tla.make_tensor(
                lse_tensor.ptr + base + sub * rows,
                tla.make_layout(tla.make_shape(rows), tla.make_stride(1)),
            )
            tla.copy(gm_lse, ub_lse)
        else:
            # 逐行按真实行数写：m 奇数（#032 h=1）或奇数 G（#013 h=65，m=Align2
            # (halfG) 为偶数但每半有效行 = halfG/gSize-halfG）。sub1 只写
            # valid_rows-rows 行，不能覆盖下一 token 的行。
            n_rows = (m if valid_rows is None else valid_rows) - sub * rows
            if n_rows > rows:
                n_rows = tla.as_numeric(rows)
            if n_rows < 0:
                n_rows = tla.as_numeric(0)
            for r in tla.range(n_rows):
                gm_r = tla.make_tensor(
                    lse_tensor.ptr + base + sub * rows + r,
                    tla.make_layout(tla.make_shape(1), tla.make_stride(1)),
                )
                src_r = tla.tile_view(ub_lse, tla.make_shape(1), tla.make_coord(r))
                tla.copy(gm_r, src_r)
        # WAR：dm 本槽下一次被 vec1 改写前，先等这次落盘搬完（MTE3→V 事件）
        tla.set_flag(sync.lse_mte3_v)
        tla.wait_flag(sync.lse_mte3_v)


# ---------------------------------------------------------------------------
# E4：FlashDecoding（KV 维分片 + 部分 softmax 归并；竞品 process_cube 的
# fd_start/fd_end 等价结构）
#
# 单元 = (task, split)，split 处理 task 的全局 tile 区间
#   [split*ceil(T/S), min((split+1)*ceil(T/S), T))
# （ori 与 cmp tile 同属一个全局 tile 序，按同一比例切分）。每个单元把
# 「未归一化 O（fp32）+ running max + running sum」写 GM workspace；全核屏障
# 后归并（数学见 fd_merge_task）。局部量不含 sinks（单元初值走 has_sinks=0 的
# (MIN_F, 0)），sink 只在归并里加一次。
#
# 归并按 D 维分摊：一个 (task, part) 归并项只处理 512/MRG 列；MRG>1 时同一
# task 的 O 累加分布到 MRG 个核（小 task 数场景填满空闲核）。
# ---------------------------------------------------------------------------
# 统计块宽度 = 整 VL（fp32 64 列）：UB 向量 load 必须是整 VL 宽。rows8<=32 时
# 尾部是未写区，归并侧用 tail mask 屏蔽。
FD_STAT_SLOT = 64


def fd_unit_stride(m, slot=FD_STAT_SLOT):
    """每个 unit 的 fp32 槽宽：[O m*HEAD_DIM][4 x slot 统计块]"""
    return m * HEAD_DIM + 4 * slot


@tla.jit
def fd_store_partial(sync, bufs, ws_fd, unit, m, rows, dtype_in, slot_sum=0, store_o=1):
    """单元末落部分量：未归一化 O（fp32）+ max + sum。

    GM 每 unit 一块（fp32 元素，长度 = fd_unit_stride）：
      [O: m x HEAD_DIM 行主序]
      [sub0 max: slot][sub0 sum: slot][sub1 max: slot][sub1 sum: slot]
    每个 AIV 子块只写自己的 rows 行与 rows8 个统计元素（其余 slot 尾部未写，
    归并侧 tail mask 屏蔽）。O 不做归一化（归并统一除）。

    E9：store_o=0（运行期标量，anchor 单元）时只写统计、不写 O —— anchor 的
    未归一化 O 留在 ub_o，由该核在屏障后自己归并（fd_merge_task own_in_ub）。
    """
    rows8 = (rows + 7) // 8 * 8
    slot = FD_STAT_SLOT
    stride = fd_unit_stride(m, slot)
    base = unit * stride
    with tla.vector():
        sub = tla.arch.sub_block_idx()
        ub_o = tla.make_tensor(
            bufs.ub_o,
            tla.make_layout(
                tla.make_shape(rows, HEAD_DIM), tla.make_stride(HEAD_DIM, 1)
            ),
        )
        _fd_st_stride = STATE_SLOT_STRIDE if bufs.opt_vec else _bc_rows8(rows)
        ub_max8 = _stats(bufs.ub_max + slot_sum * _fd_st_stride, rows8)
        ub_sum8 = _stats(bufs.ub_sum + slot_sum * _fd_st_stride, rows8)
        # V 程序序标记：空的 lock/unlock 位于本 tile 的 vec2 之后，MTE3 取得
        # 同一 mutex 即保证 O/统计已写定（与 bmm2_data_copy_out 共用 m_o16）。
        sync.m_o16.lock(pipe=tla.pipes.VECTOR)
        sync.m_o16.unlock(pipe=tla.pipes.VECTOR)
        sync.m_o16.lock(pipe=tla.pipes.MTE3)
        if store_o == 1:
            # 修（F10）：AIV 两子块行数 rows = m - m//2（前半）。m=1（h=1 的 CSA）时
            # sub1 的有效行数为 0，`sub*rows == m`；原实现仍无条件写 rows 行 O，
            # dst 偏移 base + sub*rows*HEAD_DIM == base + m*HEAD_DIM 正好压在本 unit
            # 的 partial 统计区（max/sum）上：sub1 用未写的 ub_o 覆盖 sub0 的
            # max/sum，归并侧 s_i 全 0 被剔除 → 输出整行 0；最后一个 unit 还会
            # 越界写 256 个 fp32。守卫 `sub*rows < m` 与 bmm2_data_copy_out 同义。
            # m 在追踪期是 constexpr 推导出的 Python int：偶数 m 下 sub*rows < m
            # 恒真，这里按奇偶**追踪期折叠**，偶数 m 实例（全部官方 CSA FD 例）
            # 的指令流与改前逐字一致，零性能影响。
            # 注：本守卫只覆盖 `sub*rows >= m`（即 m=1 的 sub1）；更大的奇数 m
            # 与 bmm2_data_copy_out 一样仍按前半行数整块搬，不在本次范围。
            o_dst = tla.make_tensor(
                ws_fd.ptr + base + sub * rows * HEAD_DIM,
                tla.make_layout(
                    tla.make_shape(rows, HEAD_DIM),
                    tla.make_stride(HEAD_DIM, 1),
                ),
            )
            if m % 2 == 0:
                tla.copy(o_dst, ub_o)
            else:
                if sub * rows < m:
                    tla.copy(o_dst, ub_o)
        tla.copy(
            tla.make_tensor(
                ws_fd.ptr + base + m * HEAD_DIM + sub * 2 * slot,
                tla.make_layout(tla.make_shape(rows8), tla.make_stride(1)),
            ),
            ub_max8,
        )
        tla.copy(
            tla.make_tensor(
                ws_fd.ptr + base + m * HEAD_DIM + sub * 2 * slot + slot,
                tla.make_layout(tla.make_shape(rows8), tla.make_stride(1)),
            ),
            ub_sum8,
        )
        sync.m_o16.unlock(pipe=tla.pipes.MTE3)


def _stat2d(ptr, slot):
    """统计块的 2-D (1, slot) 视图（整 VL load 的合法形态）。"""
    return tla.make_tensor(
        ptr, tla.make_layout(tla.make_shape(1, slot), tla.make_stride(slot, 1))
    )


@tla.jit
def fd_merge_task(
    sync,
    bufs,
    ws_fd,
    sinks,
    out_tensor,
    lse_tensor,
    task,
    part,
    m,
    rows,
    dtype_in,
    splits: tla.Constexpr[int],
    mrg: tla.Constexpr[int],
    has_sinks: tla.Constexpr[int],
    enable_lse: tla.Constexpr[int],
    sink_reuse: tla.Constexpr[int] = 0,
    own_in_ub: tla.Constexpr[int] = 0,
    vec_stats: tla.Constexpr[int] = 0,
    splits_ws: tla.Constexpr[int] = 0,
):
    """split-0 与 D 分片归并（task 的 part 段，两个 AIV 子块各写自己的 rows）。

    设各 split 的局部 max m_i、sum s_i、未归一化 O_i：
      M = max(max_i, sink)；w_i = exp(m_i - M)；
      den = Σ w_i s_i + exp(sink - M)；O = Σ w_i O_i / den；
      lse = log(den) + M。
    空 split（无有效 tile，GM 由 host 清零 ⇒ s_i=0）用 s_i>0 判定并从 M 剔除，
    避免其未写的 max 参与 max。den=max(den,1) 只作用于「全空」边界（非空时
    某 split 的 max 项 exp(0)=1 且 s_i>=1 ⇒ den>=1），不改变有效结果。
    精度：max/sum/O 全 fp32，最终 cast 一次（严格不低于串行实现）。
    """
    rows8 = (rows + 7) // 8 * 8
    slot = FD_STAT_SLOT
    stride = fd_unit_stride(m, slot)
    # E10：ws 槽步长（每 task 的 split 数上界）与本次实际归并的 split 数
    # 解耦：归并项表可对「只有 1 个 split」的 task 用 splits=1 读，避免读空槽。
    nws = splits if splits_ws == 0 else splits_ws
    base0 = task * nws * stride
    dlen = HEAD_DIM // mrg
    col0 = part * dlen
    o_max = 0
    o_sum = splits * slot
    o_e = 2 * splits * slot
    o_den = 3 * splits * slot
    o_sink = (3 * splits + 1) * slot
    with tla.vector():
        sub = tla.arch.sub_block_idx()
        # ---- MTE2：各 split 的 max/sum（+ sink 本子块的 rows 行）----
        # 同步一律走 m_o16 mutex（MTE2/V/MTE3 三段交接），不新增事件 id：CSA
        # 非 PA 实例的 (dst,id) 已被索引环/搬运链占满，借用 bc_* 会与其别名。
        sync.m_o16.lock(pipe=tla.pipes.MTE2)
        for i in static_range(splits):
            ubase = base0 + i * stride + m * HEAD_DIM + sub * 2 * slot
            tla.copy(
                _stat2d(bufs.ub_fd_stats + o_max + i * slot, slot),
                _stat2d(ws_fd.ptr + ubase, slot),
            )
            tla.copy(
                _stat2d(bufs.ub_fd_stats + o_sum + i * slot, slot),
                _stat2d(ws_fd.ptr + ubase + slot, slot),
            )
        if tla.const_expr(has_sinks == 1):
            # TOK2 两个 token 共用同一段 sinks（sink_reuse=1，按 token 内 head 号）；
            # 其余按 sub*rows 取本子块行。
            tla.copy(
                _stats(bufs.ub_fd_stats + o_sink, rows),
                tla.make_tensor(
                    sinks.ptr + (0 if sink_reuse == 1 else sub * rows),
                    tla.make_layout(tla.make_shape(rows), tla.make_stride(1)),
                ),
            )
        sync.m_o16.unlock(pipe=tla.pipes.MTE2)
        # 块视图（1-D 元素视图，逐行 shape(1)+brc 读取——与 bc_merge_prev 同款）
        ub_mx1 = tuple(
            _stats(bufs.ub_fd_stats + o_max + i * slot, slot)
            for i in static_range(splits)
        )
        ub_sm1 = tuple(
            _stats(bufs.ub_fd_stats + o_sum + i * slot, slot)
            for i in static_range(splits)
        )
        ub_ee = tuple(
            _stats(bufs.ub_fd_stats + o_e + i * slot, slot)
            for i in static_range(splits)
        )
        ub_den_t = _stats(bufs.ub_fd_stats + o_den, slot)
        ub_sink_t = _stats(bufs.ub_fd_stats + o_sink, slot)
        ub_max0 = _max(bufs, rows, 0)
        ub_sum0 = _sum(bufs, rows, 0)
        # ---- V：M / w_i / den（逐行 broadcast）----
        sync.m_o16.lock(pipe=tla.pipes.VECTOR)
        with tla.vec.func(mode="simd"):
            preg = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            one_mask, _ = tla.update_mask(1, dtype=tla.Float32)
            brc = tla.params.NormalLoadParams(
                load_dist=tla.params.LoadDist.DIST_BRC_B32
            )
            ustore = tla.params.UnalignStoreParams()
            minreg = tla.full(MIN_F, dtype=tla.Float32)
            zero = tla.full(0.0, dtype=tla.Float32)
            one = tla.full(1.0, dtype=tla.Float32)
            if tla.const_expr(vec_stats == 1):
                # E10：统计段跨行向量化。rows <= VL（FP32）时每个 split 的
                # max/sum 槽就是一段连续 rows 行的向量，一条 V 指令即可覆盖
                # 全部行；tail mask 屏蔽 rows 之后的未写区。数学过程与逐行版
                # 完全一致（同一 alive 判据、同一 max/exp/den 次序），只是把
                # 行维从标量循环改成向量 lanes。ub_ee 仍按行供 O 段 BRC 读。
                tail, _ = tla.update_mask(rows, dtype=tla.Float32)
                mrun = minreg
                for i in static_range(splits):
                    mx = ub_mx1[i].load()
                    sm = ub_sm1[i].load()
                    alive = tla.cmp(sm, zero, mode="gt", mask=tail)
                    meff = tla.where(alive, mx, minreg)
                    mrun = tla.max(mrun, meff, mask=tail)
                if tla.const_expr(has_sinks == 1):
                    sk = ub_sink_t.load()
                    mrun = tla.max(mrun, sk, mask=tail)
                den = zero
                for i in static_range(splits):
                    mx = ub_mx1[i].load()
                    sm = ub_sm1[i].load()
                    alive = tla.cmp(sm, zero, mode="gt", mask=tail)
                    meff = tla.where(alive, mx, minreg)
                    ei = tla.exp(tla.sub(meff, mrun, mask=tail), mask=tail)
                    ei = tla.where(alive, ei, zero)
                    ub_ee[i].store(ei, mask=tail)
                    den = tla.add(den, tla.mul(ei, sm, mask=tail), mask=tail)
                if tla.const_expr(has_sinks == 1):
                    sk = ub_sink_t.load()
                    den = tla.add(
                        den,
                        tla.exp(tla.sub(sk, mrun, mask=tail), mask=tail),
                        mask=tail,
                    )
                den = tla.max(den, one, mask=tail)
                ub_den_t.store(den, mask=tail)
                ub_max0.store(mrun, mask=tail)
                ub_sum0.store(den, mask=tail)
            else:
                for r in tla.range(rows):
                    mrun = minreg
                    for i in static_range(splits):
                        mx = tla.tile_view(
                            ub_mx1[i], tla.make_shape(1), tla.make_coord(r)
                        ).load(params=brc)
                        sm = tla.tile_view(
                            ub_sm1[i], tla.make_shape(1), tla.make_coord(r)
                        ).load(params=brc)
                        alive = tla.cmp(sm, zero, mode="gt", mask=preg)
                        meff = tla.where(alive, mx, minreg)
                        mrun = tla.max(mrun, meff, mask=preg)
                    if tla.const_expr(has_sinks == 1):
                        sk1 = tla.tile_view(
                            ub_sink_t, tla.make_shape(1), tla.make_coord(r)
                        ).load(params=brc)
                        mrun = tla.max(mrun, sk1, mask=preg)
                    den = zero
                    for i in static_range(splits):
                        mx = tla.tile_view(
                            ub_mx1[i], tla.make_shape(1), tla.make_coord(r)
                        ).load(params=brc)
                        sm = tla.tile_view(
                            ub_sm1[i], tla.make_shape(1), tla.make_coord(r)
                        ).load(params=brc)
                        alive = tla.cmp(sm, zero, mode="gt", mask=preg)
                        meff = tla.where(alive, mx, minreg)
                        ei = tla.exp(tla.sub(meff, mrun, mask=preg), mask=preg)
                        ei = tla.where(alive, ei, zero)
                        tla.tile_view(
                            ub_ee[i], tla.make_shape(1), tla.make_coord(r)
                        ).store(ei, params=ustore, mask=one_mask)
                        den = tla.add(den, tla.mul(ei, sm, mask=preg), mask=preg)
                    if tla.const_expr(has_sinks == 1):
                        sk2 = tla.tile_view(
                            ub_sink_t, tla.make_shape(1), tla.make_coord(r)
                        ).load(params=brc)
                        den = tla.add(
                            den,
                            tla.exp(tla.sub(sk2, mrun, mask=preg), mask=preg),
                            mask=preg,
                        )
                    den = tla.max(den, one, mask=preg)
                    tla.tile_view(ub_den_t, tla.make_shape(1), tla.make_coord(r)).store(
                        den, params=ustore, mask=one_mask
                    )
                    tla.tile_view(ub_max0, tla.make_shape(1), tla.make_coord(r)).store(
                        mrun, params=ustore, mask=one_mask
                    )
                    tla.tile_view(ub_sum0, tla.make_shape(1), tla.make_coord(r)).store(
                        den, params=ustore, mask=one_mask
                    )
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
        sync.m_o16.unlock(pipe=tla.pipes.VECTOR)
        # ---- E14：anchor 单趟归并（split0 在 ub_o，split1 落 otmp）----
        # 数学次序与逐 split 版逐元素一致：w0*O0 + w1*O1 后统一除 den、CAST 一次；
        # 省掉 acc 的两次整块 UB 往返与独立 cast 遍历（每块 12 -> 8 条 vec op）。
        # 视图构造放在分支内：非锚点路径（#021/#022/#028）的 MLIR 因此逐条不变。
        if tla.const_expr(own_in_ub == 1 and splits == 2 and mrg == 1):
            ub_o16 = tla.make_tensor(
                bufs.ub_o16,
                tla.make_layout_with_tag((rows, HEAD_DIM), dtype_in, tla.arch.RowMajor),
            )
            o16p = tla.make_tensor(
                bufs.ub_o16,
                tla.make_layout(
                    tla.make_shape(rows, dlen),
                    tla.make_stride(dlen, 1),
                    layoutTag=tla.arch.RowMajor,
                ),
            )
            # O1 = 非 anchor 段的部分量（O0 已在 ub_o，不再落盘/读回）。
            sync.m_o16.lock(pipe=tla.pipes.MTE2)
            ub_o1 = tla.make_tensor(
                bufs.ub_otmp_full,
                tla.make_layout(tla.make_shape(rows, dlen), tla.make_stride(dlen, 1)),
            )
            tla.copy(
                ub_o1,
                tla.make_tensor(
                    ws_fd.ptr + base0 + stride + sub * rows * HEAD_DIM + col0,
                    tla.make_layout(
                        tla.make_shape(rows, dlen), tla.make_stride(HEAD_DIM, 1)
                    ),
                ),
            )
            sync.m_o16.unlock(pipe=tla.pipes.MTE2)
            sync.m_o16.lock(pipe=tla.pipes.VECTOR)
            with tla.vec.func(mode="simd"):
                preg = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                preg_b16 = tla.create_mask(pattern=tla.mask.ALL, dtype=dtype_in)
                brc = tla.params.NormalLoadParams(
                    load_dist=tla.params.LoadDist.DIST_BRC_B32
                )
                cast_zero = tla.params.CastParams(
                    reg_slot=tla.params.RegSlot.ZERO,
                    sat_mode=tla.params.SatMode.SAT,
                    round_mode=tla.params.RoundMode.CAST_ROUND,
                )
                pack_store = tla.params.NormalStoreParams(
                    store_dist=tla.params.StoreDist.DIST_PACK_B32
                )
                ub_o0 = tla.make_tensor(
                    bufs.ub_o,
                    tla.make_layout(
                        tla.make_shape(rows, dlen), tla.make_stride(dlen, 1)
                    ),
                )
                for r in tla.range(rows):
                    e0 = tla.tile_view(
                        ub_ee[0], tla.make_shape(1), tla.make_coord(r)
                    ).load(params=brc)
                    e1 = tla.tile_view(
                        ub_ee[1], tla.make_shape(1), tla.make_coord(r)
                    ).load(params=brc)
                    den_r = tla.tile_view(
                        ub_den_t, tla.make_shape(1), tla.make_coord(r)
                    ).load(params=brc)
                    for c in tla.range(dlen // VL):
                        s0 = tla.tile_view(
                            ub_o0, tla.make_shape(1, VL), tla.make_coord(r, c)
                        )
                        s1 = tla.tile_view(
                            ub_o1, tla.make_shape(1, VL), tla.make_coord(r, c)
                        )
                        d16 = tla.tile_view(
                            o16p, tla.make_shape(1, VL), tla.make_coord(r, c)
                        )
                        acc = tla.add(
                            tla.mul(e0, s0.load(), mask=preg),
                            tla.mul(e1, s1.load(), mask=preg),
                            mask=preg,
                        )
                        d16.store(
                            tla.div(acc, den_r, mask=preg).to(
                                dtype_in, cast_zero, mask=preg
                            ),
                            pack_store,
                            mask=preg_b16,
                        )
                tla.local_mem_bar(
                    tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                )
            sync.m_o16.unlock(pipe=tla.pipes.VECTOR)
        else:
            # ---- 逐 split 载入 O_i 的本 D 段，acc += w_i * O_i ----
            for i in static_range(splits):
                if tla.const_expr(own_in_ub == 1 and i == 0):
                    # E9：本核就是该 task 的 anchor —— split 0 的未归一化 O 已在
                    # ub_o 里（fd_store_partial store_o=0 跳过落盘），直接当源用，
                    # 省一次 128KB 的 GM 往返。MRG 恒 1 时 dlen==HEAD_DIM，视图
                    # 与实际布局逐元素一致。
                    ub_o2 = tla.make_tensor(
                        bufs.ub_o,
                        tla.make_layout(
                            tla.make_shape(rows, dlen), tla.make_stride(dlen, 1)
                        ),
                    )
                else:
                    # MTE2 取锁 = V 已读完上一 split 的 ub_o（WAR），同时与 V 段互斥。
                    sync.m_o16.lock(pipe=tla.pipes.MTE2)
                    ub_o2 = tla.make_tensor(
                        bufs.ub_o,
                        tla.make_layout(
                            tla.make_shape(rows, dlen), tla.make_stride(dlen, 1)
                        ),
                    )
                    tla.copy(
                        ub_o2,
                        tla.make_tensor(
                            ws_fd.ptr
                            + base0
                            + i * stride
                            + sub * rows * HEAD_DIM
                            + col0,
                            tla.make_layout(
                                tla.make_shape(rows, dlen), tla.make_stride(HEAD_DIM, 1)
                            ),
                        ),
                    )
                    sync.m_o16.unlock(pipe=tla.pipes.MTE2)
                sync.m_o16.lock(pipe=tla.pipes.VECTOR)
                with tla.vec.func(mode="simd"):
                    preg = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                    brc = tla.params.NormalLoadParams(
                        load_dist=tla.params.LoadDist.DIST_BRC_B32
                    )
                    # 本 part 的列段视图：UB 根指针 + 静态形状（无动态 tile 坐标）
                    acc2 = tla.make_tensor(
                        bufs.ub_otmp_full,
                        tla.make_layout(
                            tla.make_shape(rows, dlen), tla.make_stride(dlen, 1)
                        ),
                    )
                    for r in tla.range(rows):
                        w = tla.tile_view(
                            ub_ee[i], tla.make_shape(1), tla.make_coord(r)
                        ).load(params=brc)
                        for c in tla.range(dlen // VL):
                            dst = tla.tile_view(
                                acc2, tla.make_shape(1, VL), tla.make_coord(r, c)
                            )
                            src = tla.tile_view(
                                ub_o2, tla.make_shape(1, VL), tla.make_coord(r, c)
                            )
                            val = tla.mul(w, src.load(), mask=preg)
                            if i == 0:
                                dst.store(val, mask=preg)
                            else:
                                dst.store(
                                    tla.add(dst.load(), val, mask=preg), mask=preg
                                )
                    tla.local_mem_bar(
                        tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                    )
                sync.m_o16.unlock(pipe=tla.pipes.VECTOR)
            # ---- O/den -> out（CAST_ROUND 一次；与 bmm2_data_copy_out 同款）----
            ub_o16 = tla.make_tensor(
                bufs.ub_o16,
                tla.make_layout_with_tag((rows, HEAD_DIM), dtype_in, tla.arch.RowMajor),
            )
            o16p = tla.make_tensor(
                bufs.ub_o16,
                tla.make_layout(
                    tla.make_shape(rows, dlen),
                    tla.make_stride(dlen, 1),
                    layoutTag=tla.arch.RowMajor,
                ),
            )
            sync.m_o16.lock(pipe=tla.pipes.VECTOR)
            with tla.vec.func(mode="simd"):
                preg = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                preg_b16 = tla.create_mask(pattern=tla.mask.ALL, dtype=dtype_in)
                brc = tla.params.NormalLoadParams(
                    load_dist=tla.params.LoadDist.DIST_BRC_B32
                )
                cast_zero = tla.params.CastParams(
                    reg_slot=tla.params.RegSlot.ZERO,
                    sat_mode=tla.params.SatMode.SAT,
                    round_mode=tla.params.RoundMode.CAST_ROUND,
                )
                pack_store = tla.params.NormalStoreParams(
                    store_dist=tla.params.StoreDist.DIST_PACK_B32
                )
                acc3 = tla.make_tensor(
                    bufs.ub_otmp_full,
                    tla.make_layout(
                        tla.make_shape(rows, dlen), tla.make_stride(dlen, 1)
                    ),
                )
                for rf in tla.range(rows):
                    den_r = tla.tile_view(
                        ub_den_t, tla.make_shape(1), tla.make_coord(rf)
                    ).load(params=brc)
                    for cf in tla.range(dlen // VL):
                        srcf = tla.tile_view(
                            acc3, tla.make_shape(1, VL), tla.make_coord(rf, cf)
                        )
                        d16f = tla.tile_view(
                            o16p, tla.make_shape(1, VL), tla.make_coord(rf, cf)
                        )
                        d16f.store(
                            tla.div(srcf.load(), den_r, mask=preg).to(
                                dtype_in, cast_zero, mask=preg
                            ),
                            pack_store,
                            mask=preg_b16,
                        )
                tla.local_mem_bar(
                    tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                )
            sync.m_o16.unlock(pipe=tla.pipes.VECTOR)
        sync.m_o16.lock(pipe=tla.pipes.MTE3)
        if sub * rows < m:
            if tla.const_expr(mrg == 1):
                # 整宽：直接用根视图（避免动态 coord 的 tile copy 走偏）。
                tla.copy(
                    tla.make_tensor(
                        out_tensor.ptr + task * (m * HEAD_DIM) + sub * rows * HEAD_DIM,
                        tla.make_layout_with_tag(
                            (rows, HEAD_DIM), dtype_in, tla.arch.RowMajor
                        ),
                    ),
                    ub_o16,
                )
            else:
                tla.copy(
                    tla.make_tensor(
                        out_tensor.ptr
                        + task * (m * HEAD_DIM)
                        + sub * rows * HEAD_DIM
                        + col0,
                        tla.make_layout(
                            tla.make_shape(rows, dlen),
                            tla.make_stride(HEAD_DIM, 1),
                        ),
                    ),
                    o16p,
                )
        sync.m_o16.unlock(pipe=tla.pipes.MTE3)
    # E6 修（#026 启用 FD 时编译失败）：lse_copy_out 自带 `with tla.vector()`，
    # 不能嵌在 fd_merge_task 的 vector 作用域内（前端禁止 cube/vector 嵌套）。
    # 归并已把合并后的 max/sum 写进 slot 0（ub_max0/ub_sum0），在作用域外调用
    # 语义相同（同核程序序），指令流其余部分不变。
    if tla.const_expr(enable_lse == 1):
        # lse 与 D 分段无关，只由 part 0 写一次。
        if part == 0:
            lse_copy_out(
                sync, bufs, lse_tensor, task, m, rows, slot=0, row_base=task * m
            )


# ---------------------------------------------------------------------------
# F2：IS_VEC_S2PHYADDR —— KV 物理地址预计算表（get_kv_phy_addr_vf.h 全量翻译）
#
# OT 结构：Init 阶段 AIV 按 aivIdx % vecCoreNum 两遍全局行划分，对每个 (bS1Idx)
# 行 CopyPaTableToUb + CopySparseIdxToUb → VF 每 128 列算 128 个 int64 物理地址
# （2 x 64-lane reg，Mull/AddC 64 位拆 low/high，DIST_INTLV_B32 写 UB）→
# CopyPhyAddrToGm 落 workspace 表（ori 段在前、cmp 段在后）→ SyncAll → 主循环
# GetkeyOffset 直接读表。6 个 INNERCORE_PHYADDR_* flag 在两个 AIV 子块间交接
# 共享 UB 暂存。
#
# 本仓对应（偏离 D-F2b/D-F2c 见 ITERATIONS.md）：D1 任务划分（每 task 独立处理
# 一个 token）没有 Init 阶段、也没有跨核屏障可用 ⇒ 改为「逐 token 生成」：每个
# task 在进入 tile 循环前为当前 tok 生成表行；两个 AIV 子块各自重复生成并重复
# 写同一 GM 行——写入内容逐字节相同，且每个子块只在自身 phy_m3v 屏障之后读表
# （读到的一定是自身已完成的写），同址并发写相同值不可能撕出第三种字节 ⇒ 安全。
# 6 个 INNERCORE_PHYADDR_* 在单子块内自用，退化为 3 对 set+wait 同 flag 屏障。
#
# 逐符号对照：
#   CopyPhyAddrToGm        smla_vector_common_arch35.h:26  -> copy_phy_addr_to_gm
#   CopyPaTableToUb        smla_vector_common_arch35.h:44  -> copy_pa_table_to_ub
#   CopySparseIdxToUb      smla_vector_common_arch35.h:56  -> copy_sparse_idx_to_ub
#   GetKVPhyAddrVfPA       get_kv_phy_addr_vf.h:184/208    -> get_kv_phy_addr_vf_pa(+_impl)
#   GetKVPhyAddrVfTND      get_kv_phy_addr_vf.h:283/307    -> get_kv_phy_addr_vf_tnd(+_impl)
#   GetKVPhyAddrVfBSND     get_kv_phy_addr_vf.h:391/415    -> get_kv_phy_addr_vf_bsnd(+_impl)
#   GetKVPhyAddrForKvType  csa_block_vector_arch35.h:1858  -> get_kv_phy_addr_for_kv_type
#   GetKVPhyAddr           csa_block_vector_arch35.h:146   -> get_kv_phy_addr
#   CalcPhyAddrValidInfo / CalcCurValidS2（csa_block_vector_arch35.h:1554-1651）
#       -> 行数公式已在各 kernel 的 lines 公式逐式翻译；OT validS2 在全部
#          cmpMaskMode/oriMaskMode 组合下 ≥ 本仓 lines（覆盖证明见
#          ITERATIONS.md Iter 20），故只生成 [0, lines) 列，其余列永不消费。
# ---------------------------------------------------------------------------

PHY_ELEMS_PER_LOOP = 128  # OT paElemsPerLoop：每轮 128 列稀疏索引
PHY_ELEMS_PER_REG = 64  # OT paElemsPerReg：每 reg 64 列（= VL）
PHY_ADDR_PER_LOOP = 256  # OT paAddrPerLoop：128 个 int64 地址的 u32 视图元素数
PHY_ADDR_PER_REG = 128  # OT paAddrPerReg：64 个 int64 地址的字节数（1024）；
# 本仓 store 按 64-u32 tile 切（interleave 输出各
# 64 lane），地址半宽寄存器无 128-lane i32 形态
PHY_INVALID = -1  # OT INVALID(0xFFFFFFFF) 的 Int32 位型

# F38（2026-11-17，卡 1 同窗 A/B）：
# bmm2_data_copy_out 的终局归一改成「div + CAST_ROUND 融合一趟」，
# 省掉 f32 中间结果落 UB 再读回（每 chunk 6 → 4 条 vec op）。
# 数值路径不变：同一个 Div 结果直接 CAST_ROUND；不动任何舍入模式或累加顺序。
# 证据：049 同进程交错 2×3 → Dur 206.69→203.30（**−3.39 µs，−1.64%**）、
# `aiv_vec` 67.16→63.14（−4.01），049 两 rep bit_exact 1.0。

# F31 ③（保真项）：2 行合并搬运，逐字对位 OT
# `SmlaCsaBlockVector::CopyInKvSparse`（sparse_flash_mla_csa_block_vector_arch35.h:495）：
#   for (i = 0; i < 8; i += 2) 取 tokenData[i], tokenData[i+1]
#   keySrcStride = |off1 - off0| * sizeof(KV_T) - combineBytes   （默认口径）
#   stride 溢出 / 为负 / sparseBlockSize>1 → 退回 CopyInSingleKv ×2
#   DataCopyExtParams：blockCount=2（尾块 1 + (off1>=0)）/ blockLen=combineBytes /
#                      dstStride=0 / 运行期 srcStride；起始 keyOffset = min(off0, off1)
#                      （IS_FULL）/ (off1>=0 && off1<off0) ? off1 : off0（尾块）
# 族别核对：OT 三个 family 共用同一份 block_vector 头，合并分支的开关是
# IS_VEC_S2PHYADDR + stride/sparseBlockSize 兜底，**不是族别**；我方该函数也只被
# is_vec_s2phyaddr==1 实例调用（50 例中 021–050），故与 OT 的分族一致。
# 合并**条件照抄 OT**（已回源码逐行核对 sparse_flash_mla_csa_block_vector_arch35.h:495-560）：
#   OT 侧唯一的开关是 `keySrcStride >= INT32_MAX || keySrcStride < 0 ||
#   sparseBlockSize > 1`；**没有**「两块物理相邻」或「同对齐」这类额外条件
#   （blockLen 恒为一行 dSizeVInput*sizeof(KV_T)=1024 B，天然 32B 对齐；源地址
#   由 DataCopyPad 自由寻址，只要求 stride 非负即不重叠）。
#   ⇒ 我方门限 = `|Δoff|*2 - 1024 >= 0` 且 `< INT32_MAX`，即
#      `HEAD_DIM <= |off_b - off_a| < 2^30`（元素数）。
#   **降序也合并**：起始取 min，UB 两行顺序随之互换（OT 默认口径的行为，不额外加
#     `off1 > off0` 门）。只有 IS_BATCH_CONSISTENCY（GetDeterministicLevel()==3）
#     才用有符号 stride 禁掉互换；本 workload tiling 默认 false。
# 尾块：只剩首行时用 blockCount=1 的合并描述符（src=off0，OT 侧 off1=-1）；
#   两行都无效时不发搬运（对应 OT 的提前 return）。stride 越界/过小时回落
#   CopyInSingleKv 语义的两次单行（保逻辑行序）。
# sparseBlockSize>1 回落：`_SPARSE_BLOCK_SIZE` 为编译期常量（OT 在 csa/swa 两处
#   硬编码为 1），取 >1 时整条 pair 分支静态退回逐行路径。
# ⚠ 步长口径：`phy_table` 存的是**元素**偏移，行差必须乘 2 B（ELEM_BYTES），
# 不是乘一整行字节数 1024；早期探针按 1024 算，把第二块源地址推到张量外
# 3.6 MB、读回 0，被误判成「嵌套形状下第二行丢失」（见 ITERATIONS.md §21.0）。
_PAIR_ELEM_BYTES = 2
# OT 在 csa_kernel_arch35.h:241 / swa_kernel_arch35.h:194 把 sparseBlockSize 硬编码为 1；
# 该常量 >1 时 OT 走 CopyInSingleKv×2，这里用同一门（编译期常量，静态分支）。
_SPARSE_BLOCK_SIZE = 1
# F31 ①c（OT 原形）：整块 8 条地址先读完 → 4 条合并拷；表读时机与 OT
# `GetRealS2Addr` 一致（整个 8 行块先读完，再在 `CopyInKvSparse` 循环发搬运）。
# gather 的搬运行按方向拆成两个拷贝点（升序/降序各自合并成一条 DataCopyPad）；
# 单拷贝点收敛（base=min(off_a,off_b)、stride=|Δ|*2-1024）的微内核 A/B 见
# ITERATIONS §27.31（IR 侧 −15 条分支 / −11 条 cast，+12 条算术），未采纳。
# 两个 G 半区共用一份 GM 暂存（OT `v0ResGm` 按 `aicIdx>>1`
# 分配）+ OT 的「自有 AIV 等待 → 全 AIC 跨块保护 → 读」顺序。
# 这是 OT 原形态的保真项，不是性能偏离——OT 在 `g_split=2` 下本来就共用暂存，
# 并由 `CrossCoreSetFlag<0>` 逐 load 节流保护。
# 选臂点在追踪期按 `g_split == 2` 折叠（对应 OT `IS_SPLIT_G` const_expr），
# `g_split==1` 的实例 IR 不变；**不得**在向量函数里加运行期守卫（历史 +40%）。

# 共享暂存下让「AIC 对的 4 个 AIV 各写 1/4 象限」
# （OT `CalProcessSize`，block_vector.h:754-777）。步一保留各写整份，步二起按象限切。
# 象限切分把配对重复的 gather 砍掉一半
# （#047 AIV 工作量 −24.9 us），是共享暂存的意义所在。

# ---------------------------------------------------------------------------
# 追踪期配置（F45 事故后的结构性收敛）
#
# 背景：共享暂存的全部语义都以「两个 G 半区配对」为前提（g_split==2）。此前象限切分
# 的判据直接留在 gather 内部，而 gather 看不见 g_split ⇒ 翻默认
# 后 g_split=1 的 #044 从 bit 1.0 掉到 0.0019（每核 4 个 AIV 各写 1/4 行）。
# 补救不能停留在「自觉重写一个镜像变量」：共用本 gather 的 kernel 有 ocs / os / csa
# 三个，任何一个漏写都会继承上一次追踪的残留值。
#
# 现在只有一条路径能定义配对语义：
#   kernel 体首（declare_sync 之前）调用 configure_trace(g_split, ...)，
#   由它按 constexpr 折叠出 TRACE.{paired,stage,quarter,slots}；
#   gather 只读 TRACE，不再读任何模块级策略。
# 两侧都有硬断言（追踪期 Python raise ⇒ 编译失败，而不是静默错误）：
#   - 非配对实例请求共享臂 ⇒ configure_trace 直接拒绝；
#   - 非配对实例走到共享路径 ⇒ require_share 拒绝（第二道防线）；
#   - 配对实例承诺开启共享臂却一次也没发射 ⇒ verify_trace_closed 拒绝。
# 约束：@tla.jit 的参数不带 Constexpr 性，不能用函数参数传这些 bool（会被当动态 if，
# 报 "dynamic Tla if requires variables..."），所以走模块级追踪期状态。


class TraceFidelityError(RuntimeError):
    """追踪期保真约束被违反。宁可编译失败，也不产出静默错误的 kernel。"""


# 追踪期状态必须是**模块级 bool**：TLA 只把模块级 Python bool 折叠成编译期常量，
# 对象属性（TRACE.quarter）会被当成动态 if，触发
# "dynamic Tla if requires variables used after the branch..."。
_TRACE_OWNER = None
_TRACE_GSPLIT = 0
_TRACE_PAIRED = False
_TRACE_STAGE = False
_TRACE_QUARTER = False
_TRACE_SLOTS = 0
_TRACE_EMITS = 0


def configure_trace(g_split, share_stage, share_quarter, kv_slots, owner):
    """**全工程唯一**的 g_split 类选择入口，返回 paired（g_split == 2）。

    `share_stage/share_quarter` 是「配对时启用」的策略：调用方在唯一类选择点
    折叠好再传进来。若有人把模块级策略**原值**直接传入而 g_split=1，这里的绊线
    会立刻让编译失败——那正是 F45 事故的形态。
    """
    global _TRACE_OWNER, _TRACE_GSPLIT, _TRACE_PAIRED
    global _TRACE_STAGE, _TRACE_QUARTER, _TRACE_SLOTS, _TRACE_EMITS
    _TRACE_OWNER = None
    _TRACE_GSPLIT = 0
    _TRACE_PAIRED = False
    _TRACE_STAGE = False
    _TRACE_QUARTER = False
    _TRACE_SLOTS = 0
    _TRACE_EMITS = 0
    if g_split not in (1, 2):
        raise TraceFidelityError(
            f"[{owner}] 拒绝编译：g_split={g_split}，本工程只定义 1/2 两类配对语义"
        )
    paired = g_split == 2
    if not paired and (share_stage or share_quarter):
        raise TraceFidelityError(
            f"[{owner}] 拒绝编译：共享槽臂在 g_split={g_split} 下被请求"
            f"（share_stage={share_stage}, share_quarter={share_quarter}）。"
            "共享暂存要求相邻两核处理同一 token：g_split=1 时发射会把每核 4 个 AIV "
            "的写入裁成 1/4 行，并把槽环撑到跨核别名（F45：《#044 bit 1.0→0.0019》）。"
            "若确需在 g_split=1 下共享，必须先单独实现配平与槽环，不得复用本机制。"
        )
    if share_quarter and not share_stage:
        raise TraceFidelityError(
            f"[{owner}] 拒绝编译：象限切分依赖共享暂存（share_quarter=True 而 "
            "share_stage=False）"
        )
    _TRACE_OWNER = owner
    _TRACE_GSPLIT = g_split
    _TRACE_PAIRED = paired
    _TRACE_STAGE = bool(share_stage) and paired
    _TRACE_QUARTER = bool(share_quarter) and paired
    _TRACE_SLOTS = kv_slots * 2 if _TRACE_STAGE else kv_slots
    return paired


def require_configured(site):
    """调用方必须先 configure_trace。TLA 前端禁止在被追踪函数体内构造异常，
    所以所有 raise 都收在普通 Python helper 里。"""
    if _TRACE_OWNER is None:
        raise TraceFidelityError(
            f"{site} 调用方未调用 configure_trace：共用本 gather 的 kernel "
            "必须在体首声明配对意图"
        )


def require_share(site):
    """共享槽路径的第二道防线：只有本 kernel 已声明配对且开着共享臂才放行。"""
    global _TRACE_EMITS
    if not (_TRACE_PAIRED and _TRACE_STAGE):
        raise TraceFidelityError(
            f"[{_TRACE_OWNER or '未配置'}] {site}: 共享槽路径在非配对/未开臂实例中"
            f"被追踪（paired={_TRACE_PAIRED}, stage={_TRACE_STAGE}）"
        )
    _TRACE_EMITS += 1


def verify_trace_closed(owner):
    """kernel 体末调用：承诺开臂就必须真的发射过，防止策略被静默忽略。"""
    if _TRACE_OWNER != owner:
        raise TraceFidelityError(
            f"[{owner}] 未调用 configure_trace（当前 owner={_TRACE_OWNER}）"
        )
    if _TRACE_STAGE and _TRACE_EMITS == 0:
        raise TraceFidelityError(
            f"[{owner}] 共享槽臂已开启，但整次追踪一次也没发射到 gather"
        )


# ---------------------------------------------------------------------------
# PA 页数学的追踪期常量（R42 步一）
#
# `@tla.jit` 的标量参数不带 Constexpr 性（见上方 1567 行），所以 PA 的
# block_size / stride / topk_value_mode 若沿形参读取，会在 IR 里留下运行期
# divsi、乘法和动态 if。PA block size 是 2 的幂（host `_vectorize_flag` 的前置，
# 但 `_pa_pairwise_fill` 只在 kv_layout==2 时被调用），这里改由模块级 Python
# 常量承载：kernel 体首（configure_trace 附近）用 constexpr 值调用一次
# `configure_pa(...)`，gather 只读这些常量 ⇒ `//2^k` 折叠成 shift、
# topk 分支在追踪期折叠、phys*stride 变成常量乘。
# 形参保留只为兼容现有调用点，`_pa_pairwise_fill` 不再读取它们。
# ---------------------------------------------------------------------------
_PA_BLOCK_SIZE = 0
_PA_BLOCK_SHIFT = -1
_PA_STRIDE = 0
_PA_TOPK_MODE = 1


def configure_pa(pa_block_size, pa_stride, topk_value_mode):
    """把 kernel 的 constexpr PA 参数折叠成追踪期常量（每次编译必须调用一次）。"""
    global _PA_BLOCK_SIZE, _PA_BLOCK_SHIFT, _PA_STRIDE, _PA_TOPK_MODE
    _PA_BLOCK_SIZE = int(pa_block_size)
    _PA_STRIDE = int(pa_stride)
    _PA_TOPK_MODE = int(topk_value_mode)
    if _PA_BLOCK_SIZE > 0 and (_PA_BLOCK_SIZE & (_PA_BLOCK_SIZE - 1)) == 0:
        _PA_BLOCK_SHIFT = _PA_BLOCK_SIZE.bit_length() - 1
    else:
        # 非 2 的幂只在非 PA 实例出现（此时 PA gather 不会被追踪）；
        # 若真被追踪则显式报错，避免静默走错误的除法口径。
        _PA_BLOCK_SHIFT = -1


def require_pa_configured(site):
    """PA 页数学常量必须是有效 2 的幂；在进入 tla.vector() 之前断言。"""
    if _PA_BLOCK_SHIFT < 0:
        raise TraceFidelityError(
            f"{site} 的 PA 常量未配置：block_size={_PA_BLOCK_SIZE}、"
            f"stride={_PA_STRIDE}（需在 kernel 体首调用 configure_pa）"
        )


def require_pa_page_aligned(pa_block_size, pa_stride, owner):
    """OS/OCS 的 ori/cmp 两段共用一份追踪期常量：两段口径必须一致。"""
    if _PA_BLOCK_SIZE == 0:
        # 非 PA 实例：gather_kv_rows_pa_pairwise 不会被追踪，无需校验。
        return
    if int(pa_block_size) != _PA_BLOCK_SIZE or int(pa_stride) != _PA_STRIDE:
        raise TraceFidelityError(
            f"[{owner}] ori/cmp 的 PA 页口径不一致："
            f"configure_pa=({_PA_BLOCK_SIZE}, {_PA_STRIDE})，"
            f"另一段=({int(pa_block_size)}, {int(pa_stride)})；"
            "共用追踪期常量要求 block_size/pa_stride 相同"
        )
    if _PA_BLOCK_SHIFT < 0:
        raise TraceFidelityError(
            f"[{owner}] PA block_size={_PA_BLOCK_SIZE} 不是 2 的幂，无法折叠成移位"
        )


# 共享暂存是「请求」：是否生效由 configure_trace
# 按 g_split 折叠后写入 _TRACE_*，gather 只读 _TRACE_*。单独 import 本模块不会产生
# 任何配对语义（_TRACE_* 默认全 False）。

_PAIR_SOURCE = r"""
#include <cstdint>
#include "kernel_operator.h"

extern "C" {

// 与 OT CopyInKvSparse 合并分支同形（blockCount=2 / dstStride=0 / 运行期 srcStride）。
// 按**字节**视图实现：OT 侧 KV_T 是模板（csa 族 fp16、os/ocs 族 bf16），
// 两者 blockLen 都是 dSizeVInput*sizeof(KV_T) 字节；用 uint8 视图 + 字节长度
// 可让同一份 extern 覆盖两种 dtype，发出的 DataCopyPad 描述符逐字段相同。
[aicore] __attribute__((always_inline)) void tla_pair_gm_to_ub(
    uint64_t src_gm_addr, uint64_t dst_ub_addr, int64_t block_len_bytes,
    int64_t src_stride_bytes, int64_t dst_byte_off, int64_t block_count) {
  AscendC::DataCopyExtParams intri;
  // OT：主路径 blockCount=2；尾块只剩首行时 blockCount=1（off1=-1）
  intri.blockCount = static_cast<uint32_t>(block_count);
  intri.blockLen = static_cast<uint32_t>(block_len_bytes);
  intri.dstStride = 0;
  intri.srcStride = src_stride_bytes;  // dav-c310: int64_t 字段
  AscendC::DataCopyPadExtParams<uint8_t> padParams;
  padParams.isPad = true;
  padParams.leftPadding = 0;
  padParams.rightPadding = 0;   // blockLen（一行 dSizeVInput*sizeof(KV_T)）已 32B 对齐
  padParams.paddingValue = 0;
  AscendC::LocalTensor<uint8_t> dst(
      AscendC::TPosition::VECCALC,
      static_cast<uint32_t>(dst_ub_addr) + static_cast<uint32_t>(dst_byte_off),
      static_cast<uint32_t>(block_len_bytes));
  AscendC::GlobalTensor<uint8_t> src;
  src.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(src_gm_addr),
                      static_cast<uint32_t>(block_len_bytes));
  AscendC::DataCopyPad(dst, src, intri, padParams);
}

}
"""


@tla.extern(source=_PAIR_SOURCE, name="tla_pair_gm_to_ub")
def pair_gm_to_ub(
    src: tla.Pointer[tla.UInt8, tla.AddressSpace.gm],
    dst_base: tla.Pointer[tla.UInt8, tla.AddressSpace.ub],
    block_len_bytes: tla.Int64,
    src_stride_bytes: tla.Int64,
    dst_byte_off: tla.Int64,
    block_count: tla.Int64,
) -> None: ...


@tla.jit
def copy_sparse_idx_to_ub(ub_idx, idx_tensor, gm_base, ncols):
    """CopySparseIdxToUb（smla_vector_common_arch35.h:56）。

    OT 用 DataCopyPad（blockLen=validS2*sizeof(int32)，任意宽度），GM 基址 =
    sparseIndicesGm[(bS1Idx+s1Idx) * sparseBlockCount]——行号×行宽的**平坦元素
    偏移**。TLA 的 gm->ub copy 要求 32B 对齐（= 8 个 int32），而 idx 行基址
    tok*idx_width 未必 8 列对齐（width ∈ {1,24,...}），故整行按标量 load/
    标量 store 逐列搬入（宽度无对齐要求），语义与 DataCopyPad 的
    blockLen=ncols*sizeof(int32) 完全一致（D-F2a 偏离）。ncols ≤ 512，标量
    循环开销可忽略。gm_base 由调用方传 `table_row * idx_row_width`（Iter 21：
    上一轮误传行号本身，读成拼接 idx 的线性下标——oracle 实证
    phy[tok][j]==idx_flat[tok+j]，见 evidence/audit/F2_CLOSURE.md §2）。
    """
    for j in tla.range(ncols):
        # ub_idx 已由调用方 make_tensor 成 1-D UB Tensor（__setitem__ 要求 Tensor）
        ub_idx[j] = idx_tensor[gm_base + j]


@tla.jit
def copy_pa_table_to_ub(ub_bt, bt_tensor, bt_base, bt_cols):
    """CopyPaTableToUb（smla_vector_common_arch35.h:44）：block table 行 GM→UB。

    OT：DataCopy(blkTableUb, gm+offset, blockLen=maxBlockNumPerBatch*4)。
    Iter 21 定版：标量逐列搬运（D-F2a 同款偏离）。此前对齐 tla.copy（MTE2）
    与 VF 里的 tla.gather(UB→Reg) 组合在真机必现 errcode 95 / subErrType 0x4
    「The set and wait instructions do not match」（仅 PA 用例崩：#025/034/045；
    二分证据见 evidence/audit/F2_CLOSURE.md §2——MTE2 拷入与 gather 单独在场
    均不崩，同 task 组合必崩；改标量搬运后 #025 逐位通过）。ub_bt 已由调用方
    make_tensor 成 1-D UB Tensor。
    """
    for j in tla.range(bt_cols):
        ub_bt[j] = bt_tensor[bt_base + j]


@tla.jit
def copy_phy_addr_to_gm(ub_addr, phy_table, table_off, ncols):
    """CopyPhyAddrToGm（smla_vector_common_arch35.h:26）。

    OT：DataCopy(kvPhyAddrGm[(bS1Idx+s1Idx)*alignedCount], kvPhyAddrUb,
    blockLen=ceil(validS2/128)*128*8/32)，GM/UB 都是 int64 视角。TLA 的
    gm<->ub copy 无 Int64 路由（_SUPPORTED_VECTOR_COPY_ROUTES）⇒ GM 指针
    recast 成 Int32 视图按字节搬运（u32 元素数 = ceil(ncols/128)*256），UB 侧
    ub_phy_addr 本就按 u32 分配；GM 张量 dtype 保持 Int64 不变（纯字节搬运，
    与 OT 的 int64 DataCopy 逐字节一致，D-F2a 偏离）。
    对齐审计：alignedCount ∈ {128,256,512} ⇒ GM 字节偏移恒为 1KiB 倍数；
    拷出长度 ceil128(ncols)*8B 亦为 32B 倍数。
    """
    n_i32 = ((ncols + PHY_ELEMS_PER_LOOP - 1) // PHY_ELEMS_PER_LOOP) * PHY_ADDR_PER_LOOP
    gm_i32 = tla.make_tensor(
        tla.recast_ptr(phy_table.ptr + table_off, dtype=tla.Int32),
        tla.make_layout(tla.make_shape(n_i32), tla.make_stride(1)),
    )
    tla.copy(
        gm_i32,
        tla.tile_view(ub_addr, tla.make_shape(n_i32), tla.make_coord(0)),
    )


@tla.jit
def gather_block_page(ub_bt, b, m, n_win):
    """OT DataCopyGather(blkTableUb, bidx) 的 TLA 替代（get_kv_phy_addr_vf.h
    的页表 gather 步）。

    tla.gather(UB→Reg) 的结果位宽 = x tile 宽度，而 i32 VectorSSA 寄存器物理
    上限 64 lane（256 B/寄存器），bt_max > 64 时无法一次取整行；按 OT 每 reg
    64 列的口径把 bt 行按 64 列分窗：窗 w 只对 b >= w*64 的列生效（where 归并，
    窗按升序扫描，最终值恰为所属窗），窗内下标夹取到 [0,64)。有效列的
    bidx < bt_max 必落在本窗内，夹取只作用于其它窗的 lane，不产生越界 UB 读；
    掩外列 bidx 恒 0（idx 先填 INVALID 再夹 0），同样落窗 0。
    注意 tla.tile_view 的 coord 是 tile 粒度（元素偏移 = coord*shape），窗 w
    的元素偏移 w*64 对应 tile 坐标 w。
    """
    zero = tla.full(0, dtype=tla.Int32)
    top = tla.full(PHY_ELEMS_PER_REG - 1, dtype=tla.Int32)
    g = tla.full(0, dtype=tla.Int32)
    for w in static_range(n_win):
        lo = w * PHY_ELEMS_PER_REG
        lo_v = tla.full(lo, dtype=tla.Int32)
        bw = tla.sub(b, lo_v)
        bw = tla.where(tla.cmp(bw, zero, "lt"), zero, bw)
        bw = tla.where(tla.cmp(bw, top, "gt"), top, bw)
        win = tla.tile_view(ub_bt, tla.make_shape(PHY_ELEMS_PER_REG), tla.make_coord(w))
        gw = tla.gather(win, bw, mask=m)
        g = tla.where(tla.cmp(b, lo_v, "ge"), gw, g)
    return g


@tla.jit
def get_kv_phy_addr_vf_pa_impl(
    ub_idx,
    ub_bt,
    ub_addr,
    ncols,
    n_chunks,
    shift_right_num,
    pa_block_size,
    kv_dim,
    kv_stride,
    bt_max,
):
    """GetKVPhyAddrVfPA 实现（get_kv_phy_addr_vf.h:208-281）。

    OT 每 128 列一轮（2 x 64-lane reg），逐列：
        sidx*sparseBlockSize(≡1) -> bidx = sidx >> shiftRightNum
        -> boff = bidx*blockSize -> toff = sidx - boff -> poff = toff*kvDim
        -> pblk = DataCopyGather(blkTableUb, bidx)
        -> Mull(pblk, kvStride) = (low, carry) -> Add low + poff -> AddC high
        -> 尾列 MERGING 填 INVALID -> Store DIST_INTLV_B32。
    TLA 替代（偏差 E-F2a/E-F2b/E-F2c，见 ITERATIONS.md）：
        - Mull/AddC 的 64 位进位按值域证明恒 0（pblk*kvStride + poff < 2^31，
          附录 A），单 i32 mul/add 精确；有效列 high 段写 0（与 OT AddC 的
          carry=0 一致），无效列低/高两半均填 INVALID（OT :172-175 对 Low/High
          各自 MERGING 0xFFFFFFFF，int64 = 全 F；Iter 21 修正——此前无效列
          高半误写 0，dump oracle 在 #025 INVALID 区 1024 项暴露）；
        - 无 DIST_INTLV_B32 store dist，改为 interleave(addr, 0) 后 r0/r1 两段
          连续 store（r0/r1 恰为交错序列的前/后半，等价位型）；
        - DataCopyGather 用 64 列分窗 gather + where 归并替代（E-F2c，见
          gather_block_page 注记）；
        - 尾块掩码按 reg 精确拆分（OT 两个 reg 共用 UpdateMask(s2Tail)，s2Tail>64
          时 reg1 高位列存 garbage——永不消费；我们补齐为逐 reg 掩码 + 掩外列
          地址段填 INVALID，消费面数值相同）；
        - src<0 / phys<0 夹 0（OT 依赖无效列不被消费；我们与内联路径在被消费列
          上逐位对齐）。
    """
    with tla.vec.func(mode="simd"):
        m_all = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Int32)
        zero = tla.full(0, dtype=tla.Int32)
        minus_one = tla.full(PHY_INVALID, dtype=tla.Int32)
        n_win = (bt_max + PHY_ELEMS_PER_REG - 1) // PHY_ELEMS_PER_REG
        for c in tla.range(n_chunks):
            rem = ncols - c * PHY_ELEMS_PER_LOOP
            n0 = rem
            if n0 > PHY_ELEMS_PER_REG:
                n0 = tla.as_numeric(PHY_ELEMS_PER_REG)
            if n0 < 0:
                n0 = tla.as_numeric(0)
            n1 = rem - PHY_ELEMS_PER_REG
            if n1 < 0:
                n1 = tla.as_numeric(0)
            # update_mask(64) 恰为全掩码 ⇒ 无需按尾块分支切换 ALL/尾掩码
            m0, _ = tla.update_mask(n0, dtype=tla.Int32)
            m1, _ = tla.update_mask(n1, dtype=tla.Int32)
            # tla.tile_view 的 coord 是 tile 粒度（元素偏移 = coord*shape）：
            # 64-u32 视图窗的元素基址 c*256+{0,64,128,192} 对应 tile 坐标
            # c*4+{0,1,2,3}（PHY_ADDR_PER_LOOP/PHY_ELEMS_PER_REG = 4）。
            addr_tile = c * (PHY_ADDR_PER_LOOP // PHY_ELEMS_PER_REG)
            # ---- reg0：列 [c*128, c*128+64) ----
            s0 = tla.tile_view(
                ub_idx,
                tla.make_shape(PHY_ELEMS_PER_REG),
                tla.make_coord(c * (PHY_ELEMS_PER_LOOP // PHY_ELEMS_PER_REG)),
            ).load()
            s0 = tla.where(m0, s0, minus_one)
            s0 = tla.where(tla.cmp(s0, zero, "lt"), zero, s0)
            b0 = tla.shift_right(s0, shift_right_num)  # sidx >> log2(bs)
            t0 = tla.sub(s0, tla.mul(b0, pa_block_size))  # toff = sidx-boff
            p0 = tla.mul(t0, kv_dim)  # poff = toff*kvDim
            g0 = gather_block_page(ub_bt, b0, m0, n_win)  # DataCopyGather
            g0 = tla.where(tla.cmp(g0, zero, "lt"), zero, g0)  # phys<0 夹 0
            a0 = tla.add(tla.mul(g0, kv_stride), p0)  # phys*kvStride+poff
            a0 = tla.where(m0, a0, minus_one)  # 掩外列地址段填 INVALID
            # 无效列低/高两个半宽都填 INVALID（OT :172-173 对 AddressLow/High
            # 各自 MERGING Duplicate 0xFFFFFFFF，int64 = 全 F）；有效列高半为
            # 进位 0（E-F2a 值域证明）
            h0 = tla.where(m0, zero, minus_one)
            lo0, hi0 = tla.interleave(a0, h0)  # [low,high] 交错
            tla.tile_view(
                ub_addr,
                tla.make_shape(PHY_ELEMS_PER_REG),
                tla.make_coord(addr_tile),
            ).store(lo0, mask=m_all)
            tla.tile_view(
                ub_addr,
                tla.make_shape(PHY_ELEMS_PER_REG),
                tla.make_coord(addr_tile + 1),
            ).store(hi0, mask=m_all)
            # ---- reg1：列 [c*128+64, c*128+128) ----
            s1 = tla.tile_view(
                ub_idx,
                tla.make_shape(PHY_ELEMS_PER_REG),
                tla.make_coord(c * (PHY_ELEMS_PER_LOOP // PHY_ELEMS_PER_REG) + 1),
            ).load()
            s1 = tla.where(m1, s1, minus_one)
            s1 = tla.where(tla.cmp(s1, zero, "lt"), zero, s1)
            b1 = tla.shift_right(s1, shift_right_num)
            t1 = tla.sub(s1, tla.mul(b1, pa_block_size))
            p1 = tla.mul(t1, kv_dim)
            g1 = gather_block_page(ub_bt, b1, m1, n_win)  # DataCopyGather
            g1 = tla.where(tla.cmp(g1, zero, "lt"), zero, g1)  # phys<0 夹 0
            a1 = tla.add(tla.mul(g1, kv_stride), p1)  # phys*kvStride+poff
            a1 = tla.where(m1, a1, minus_one)  # 掩外列地址段填 INVALID
            h1 = tla.where(m1, zero, minus_one)
            lo1, hi1 = tla.interleave(a1, h1)
            tla.tile_view(
                ub_addr,
                tla.make_shape(PHY_ELEMS_PER_REG),
                tla.make_coord(addr_tile + 2),
            ).store(lo1, mask=m_all)
            tla.tile_view(
                ub_addr,
                tla.make_shape(PHY_ELEMS_PER_REG),
                tla.make_coord(addr_tile + 3),
            ).store(hi1, mask=m_all)


@tla.jit
def get_kv_phy_addr_vf_pa(
    ub_idx, ub_bt, ub_addr, ncols, n_chunks, pa_block_size, kv_dim, kv_stride, bt_max
):
    """GetKVPhyAddrVfPA 包装（get_kv_phy_addr_vf.h:184-206）。

    OT 在此把 GM/UB 指针 recast 成 uint32 视角后调 impl；TLA 的 recast 在需要处
    （copy_phy_addr_to_gm）就地完成，包装只透传（OT 的 shiftRightNum 在
    ForKvType 派生，这里同为 Constexpr 派生）。
    """
    shift_right_num = int(pa_block_size).bit_length() - 1
    get_kv_phy_addr_vf_pa_impl(
        ub_idx,
        ub_bt,
        ub_addr,
        ncols,
        n_chunks,
        shift_right_num,
        pa_block_size,
        kv_dim,
        kv_stride,
        bt_max,
    )


@tla.jit
def get_kv_phy_addr_vf_tnd_impl(ub_idx, ub_addr, ncols, n_chunks, kv_prefix, kv_dim):
    """GetKVPhyAddrVfTND 实现（get_kv_phy_addr_vf.h:307-389）。

    OT：tmp = kvPrefix + sidx*sparseBlockSize(≡1)；Mull(tmp, kvDim) -> AddC；
    尾列 MERGING INVALID；Store DIST_INTLV_B32。替代同 PA impl 注记。
    """
    with tla.vec.func(mode="simd"):
        m_all = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Int32)
        zero = tla.full(0, dtype=tla.Int32)
        minus_one = tla.full(PHY_INVALID, dtype=tla.Int32)
        for c in tla.range(n_chunks):
            rem = ncols - c * PHY_ELEMS_PER_LOOP
            n0 = rem
            if n0 > PHY_ELEMS_PER_REG:
                n0 = tla.as_numeric(PHY_ELEMS_PER_REG)
            if n0 < 0:
                n0 = tla.as_numeric(0)
            n1 = rem - PHY_ELEMS_PER_REG
            if n1 < 0:
                n1 = tla.as_numeric(0)
            # update_mask(64) 恰为全掩码 ⇒ 无需按尾块分支切换 ALL/尾掩码
            m0, _ = tla.update_mask(n0, dtype=tla.Int32)
            m1, _ = tla.update_mask(n1, dtype=tla.Int32)
            # tla.tile_view 的 coord 是 tile 粒度（元素偏移 = coord*shape）：
            # s0/s1 元素基址 c*128+{0,64} → tile 坐标 c*2+{0,1}；
            # addr 元素基址 c*256+{0,64,128,192} → tile 坐标 c*4+{0,1,2,3}。
            addr_tile = c * (PHY_ADDR_PER_LOOP // PHY_ELEMS_PER_REG)
            # ---- reg0 ----
            s0 = tla.tile_view(
                ub_idx,
                tla.make_shape(PHY_ELEMS_PER_REG),
                tla.make_coord(c * (PHY_ELEMS_PER_LOOP // PHY_ELEMS_PER_REG)),
            ).load()
            s0 = tla.where(m0, s0, minus_one)
            s0 = tla.where(tla.cmp(s0, zero, "lt"), zero, s0)
            a0 = tla.mul(tla.add(s0, kv_prefix), kv_dim)  # OT 先加后乘
            # 无效列低/高两半都填 INVALID（OT :271-272，同 PA impl 注记）
            h0 = tla.where(m0, zero, minus_one)
            lo0, hi0 = tla.interleave(a0, h0)
            tla.tile_view(
                ub_addr,
                tla.make_shape(PHY_ELEMS_PER_REG),
                tla.make_coord(addr_tile),
            ).store(lo0, mask=m_all)
            tla.tile_view(
                ub_addr,
                tla.make_shape(PHY_ELEMS_PER_REG),
                tla.make_coord(addr_tile + 1),
            ).store(hi0, mask=m_all)
            # ---- reg1 ----
            s1 = tla.tile_view(
                ub_idx,
                tla.make_shape(PHY_ELEMS_PER_REG),
                tla.make_coord(c * (PHY_ELEMS_PER_LOOP // PHY_ELEMS_PER_REG) + 1),
            ).load()
            s1 = tla.where(m1, s1, minus_one)
            s1 = tla.where(tla.cmp(s1, zero, "lt"), zero, s1)
            a1 = tla.mul(tla.add(s1, kv_prefix), kv_dim)
            h1 = tla.where(m1, zero, minus_one)
            lo1, hi1 = tla.interleave(a1, h1)
            tla.tile_view(
                ub_addr,
                tla.make_shape(PHY_ELEMS_PER_REG),
                tla.make_coord(addr_tile + 2),
            ).store(lo1, mask=m_all)
            tla.tile_view(
                ub_addr,
                tla.make_shape(PHY_ELEMS_PER_REG),
                tla.make_coord(addr_tile + 3),
            ).store(hi1, mask=m_all)


@tla.jit
def get_kv_phy_addr_vf_tnd(ub_idx, ub_addr, ncols, n_chunks, kv_prefix, kv_dim):
    """GetKVPhyAddrVfTND 包装（get_kv_phy_addr_vf.h:283-305）：指针 recast 在
    需要处就地完成，包装只透传。"""
    get_kv_phy_addr_vf_tnd_impl(ub_idx, ub_addr, ncols, n_chunks, kv_prefix, kv_dim)


@tla.jit
def get_kv_phy_addr_vf_bsnd_impl(ub_idx, ub_addr, ncols, n_chunks, b_s2_base, kv_dim):
    """GetKVPhyAddrVfBSND 实现（get_kv_phy_addr_vf.h:415-493）。

    OT：Mull(sidx*sparseBlockSize(≡1), kvDim) -> Add low + bS2Base -> AddC high；
    （bS2Base = b*s2Size*dSize，OT 由调用方拆成 (low, high) u32 对传入；
    本仓 bS2Base = kv_prefix*kvDim < 2^31，进位恒 0）。替代同 PA impl 注记。
    """
    with tla.vec.func(mode="simd"):
        m_all = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Int32)
        zero = tla.full(0, dtype=tla.Int32)
        minus_one = tla.full(PHY_INVALID, dtype=tla.Int32)
        for c in tla.range(n_chunks):
            rem = ncols - c * PHY_ELEMS_PER_LOOP
            n0 = rem
            if n0 > PHY_ELEMS_PER_REG:
                n0 = tla.as_numeric(PHY_ELEMS_PER_REG)
            if n0 < 0:
                n0 = tla.as_numeric(0)
            n1 = rem - PHY_ELEMS_PER_REG
            if n1 < 0:
                n1 = tla.as_numeric(0)
            # update_mask(64) 恰为全掩码 ⇒ 无需按尾块分支切换 ALL/尾掩码
            m0, _ = tla.update_mask(n0, dtype=tla.Int32)
            m1, _ = tla.update_mask(n1, dtype=tla.Int32)
            # tla.tile_view 的 coord 是 tile 粒度（元素偏移 = coord*shape），
            # 换算口径同 TND impl（s 坐标 c*2+{0,1}，addr 坐标 c*4+{0,1,2,3}）。
            addr_tile = c * (PHY_ADDR_PER_LOOP // PHY_ELEMS_PER_REG)
            # ---- reg0 ----
            s0 = tla.tile_view(
                ub_idx,
                tla.make_shape(PHY_ELEMS_PER_REG),
                tla.make_coord(c * (PHY_ELEMS_PER_LOOP // PHY_ELEMS_PER_REG)),
            ).load()
            s0 = tla.where(m0, s0, minus_one)
            s0 = tla.where(tla.cmp(s0, zero, "lt"), zero, s0)
            a0 = tla.add(tla.mul(s0, kv_dim), b_s2_base)  # OT 先乘后加
            # 无效列低/高两半都填 INVALID（OT :378-379，同 PA impl 注记）
            h0 = tla.where(m0, zero, minus_one)
            lo0, hi0 = tla.interleave(a0, h0)
            tla.tile_view(
                ub_addr,
                tla.make_shape(PHY_ELEMS_PER_REG),
                tla.make_coord(addr_tile),
            ).store(lo0, mask=m_all)
            tla.tile_view(
                ub_addr,
                tla.make_shape(PHY_ELEMS_PER_REG),
                tla.make_coord(addr_tile + 1),
            ).store(hi0, mask=m_all)
            # ---- reg1 ----
            s1 = tla.tile_view(
                ub_idx,
                tla.make_shape(PHY_ELEMS_PER_REG),
                tla.make_coord(c * (PHY_ELEMS_PER_LOOP // PHY_ELEMS_PER_REG) + 1),
            ).load()
            s1 = tla.where(m1, s1, minus_one)
            s1 = tla.where(tla.cmp(s1, zero, "lt"), zero, s1)
            a1 = tla.add(tla.mul(s1, kv_dim), b_s2_base)
            h1 = tla.where(m1, zero, minus_one)
            lo1, hi1 = tla.interleave(a1, h1)
            tla.tile_view(
                ub_addr,
                tla.make_shape(PHY_ELEMS_PER_REG),
                tla.make_coord(addr_tile + 2),
            ).store(lo1, mask=m_all)
            tla.tile_view(
                ub_addr,
                tla.make_shape(PHY_ELEMS_PER_REG),
                tla.make_coord(addr_tile + 3),
            ).store(hi1, mask=m_all)


@tla.jit
def get_kv_phy_addr_vf_bsnd(ub_idx, ub_addr, ncols, n_chunks, kv_prefix, kv_dim):
    """GetKVPhyAddrVfBSND 包装（get_kv_phy_addr_vf.h:391-413）：OT 在此把调用方
    的 (low, high) u32 对重拼回 bS2Base；本仓直接传 kv_prefix，bS2Base =
    kv_prefix*kvDim 在 impl 外派生。"""
    b_s2_base = kv_prefix * kv_dim
    get_kv_phy_addr_vf_bsnd_impl(ub_idx, ub_addr, ncols, n_chunks, b_s2_base, kv_dim)


@tla.jit
def get_kv_phy_addr_for_kv_type(
    sync,
    bufs,
    phy_table,
    idx,
    bt,
    table_row,
    idx_row_width,
    ncols,
    aligned_cols,
    kv_layout,
    bt_seq,
    bt_max,
    pa_block_size,
    pa_stride,
    kv_prefix,
):
    """GetKVPhyAddrForKvType（csa_block_vector_arch35.h:1858-1959）逐 token 化简版。

    idx_row_width：sparse_indices 的行宽（OT sparseBlockCount）；staging 的 GM
    基址 = table_row * idx_row_width（OT (bS1Idx+s1Idx)*sparseBlockCount）。
    不能用 aligned_cols 代替（ocs cmp 行宽 24、Align128 后 128）。

    OT 契约：Init 阶段对每个 bS1Idx 行生成 alignedSparseBlockCount 列 int64 地址
    到 workspace（ori 段在前、cmp 段在后），随后 SyncAll；两个子块经 6 个
    INNERCORE_PHYADDR_* flag 共享 UB 暂存、分摊 s2Loop。本仓（D1 任务划分，
    无 Init 阶段 / 无跨核屏障）：每个 task 在消费该行之前为当前 tok 生成
    [0, ncols) 列（ncols = lines；OT CalcCurValidS2 在全部模式下 ≥ lines，
    覆盖证明见 ITERATIONS.md Iter 20），两个子块各自重复生成（写内容逐字节
    相同 + 各自 phy_m3s 之后才读 ⇒ 并发同址同值写撕读安全）。

    屏障序列（OT 的 SyncAll 一次性覆盖全部跨管道顺序，本仓无 SyncAll 且 TLA
    不做自动管道同步，逐条显式配对，flag 语义见 smla_csa_common.py Sync 注释）：
      wait phy_vs      （S）信用：上一轮 VF 已消费 staging，本轮才能写
      copy 页表（MTE2）/ 标量 idx（S）
      set phy_s2v+phy_m2v → wait 二者（V）：idx 与页表就绪后 VF 才读
      VF 写 ub_phy_addr
      set phy_v_mte3 + phy_vs（V）：表就绪可拷出 / staging 已消费可复用
      wait phy_v_mte3（MTE3）→ copy_phy_addr_to_gm
      set phy_m3v + phy_m3s（MTE3）
      wait phy_m3v（V）：ub_phy_addr 可复用；wait phy_m3s（S）：表行已落 GM，
      后续 gather_kv_rows_vec 的 row_off 标量 GM 读才可见

    kv_layout 取 OT SMLA_LAYOUT 枚举（sparse_flash_mla_common.h:34-37）：
    BSND=0 / TND=1 / PA_BBND=2。
    """
    with tla.vector():
        # 0) V -> S 信用（init_sync 在 =1 实例预置 1 个）：MTE2 页表拷入与标量
        #    idx 写都由 S 流发起，这一对屏障同时覆盖 ub_phy_bt / ub_phy_idx
        #    的跨 task 复用（上一轮 VF 消费完才允许本轮写）。
        tla.wait_flag(sync.phy_vs)
        # tla.allocate 返回的是指针：tile_view / __setitem__ 都要求 Tensor，
        # 先按容量包成 1-D UB Tensor 视图（容量 ≥ 任何被消费的下标）。
        ub_idx_t = tla.make_tensor(
            bufs.ub_phy_idx,
            tla.make_layout(tla.make_shape(aligned_cols), tla.make_stride(1)),
        )
        ub_addr_t = tla.make_tensor(
            bufs.ub_phy_addr,
            tla.make_layout(tla.make_shape(aligned_cols * 2), tla.make_stride(1)),
        )
        # bt 视图容量按 gather 窗宽 64 对齐（与 alloc_buffers 的 bt_cap 同一
        # 口径）：bt_max=32 时窗 tile (64,) 不越出视图/分配。
        bt_view = (
            ((bt_max + PHY_ELEMS_PER_REG - 1) // PHY_ELEMS_PER_REG) * PHY_ELEMS_PER_REG
            if bt_max > 0
            else PHY_ELEMS_PER_REG
        )
        ub_bt_t = tla.make_tensor(
            bufs.ub_phy_bt,
            tla.make_layout(tla.make_shape(bt_view), tla.make_stride(1)),
        )
        # 1) 暂存：PA 先拷页表行（OT CopyPaTableToUb），再拷 idx 行
        #    （OT CopySparseIdxToUb）
        if kv_layout == 2:
            copy_pa_table_to_ub(ub_bt_t, bt, bt_seq * bt_max, bt_max)
        copy_sparse_idx_to_ub(ub_idx_t, idx, table_row * idx_row_width, ncols)
        # 2) ready 屏障：S -> V（标量 UB idx 写完成）与 MTE2 -> V（页表拷入
        #    完成）是两条独立生产流，VF 读 ub_idx / ub_bt 前各等一对
        tla.set_flag(sync.phy_s2v)
        tla.set_flag(sync.phy_m2v)
        tla.wait_flag(sync.phy_m2v)
        tla.wait_flag(sync.phy_s2v)
        # 3) VF：每 128 列一轮写 ub_phy_addr
        n_chunks = (ncols + PHY_ELEMS_PER_LOOP - 1) // PHY_ELEMS_PER_LOOP
        if kv_layout == 2:
            get_kv_phy_addr_vf_pa(
                ub_idx_t,
                ub_bt_t,
                ub_addr_t,
                ncols,
                n_chunks,
                pa_block_size,
                HEAD_DIM,
                pa_stride,
                bt_max,
            )
        elif kv_layout == 1:
            get_kv_phy_addr_vf_tnd(
                ub_idx_t,
                ub_addr_t,
                ncols,
                n_chunks,
                kv_prefix,
                HEAD_DIM,
            )
        else:
            get_kv_phy_addr_vf_bsnd(
                ub_idx_t,
                ub_addr_t,
                ncols,
                n_chunks,
                kv_prefix,
                HEAD_DIM,
            )
        # 4) V -> MTE3 屏障（OT KVADDR_READY）+ V -> S 信用归还
        tla.set_flag(sync.phy_v_mte3)
        tla.set_flag(sync.phy_vs)
        tla.wait_flag(sync.phy_v_mte3)
        # 5) MTE3 拷出（OT CopyPhyAddrToGm）：GM 行偏移 = table_row*aligned_cols
        #    （int64 元素），OT 为 (bS1Idx+s1Idx)*alignedSparseBlockCount
        copy_phy_addr_to_gm(ub_addr_t, phy_table, table_row * aligned_cols, ncols)
        # 6) MTE3 -> V 屏障（OT KVADDR_FREE）：拷出完成后 ub_phy_addr 才能复用；
        #    MTE3 -> S 屏障：表行落 GM 后标量流才能读（缺它即 errcode 95 根因）
        tla.set_flag(sync.phy_m3v)
        tla.set_flag(sync.phy_m3s)
        tla.wait_flag(sync.phy_m3v)
        tla.wait_flag(sync.phy_m3s)


@tla.jit
def get_kv_phy_addr(
    sync,
    bufs,
    phy_table,
    idx,
    bt,
    table_row,
    idx_row_width,
    ncols,
    aligned_cols,
    kv_layout,
    bt_seq,
    bt_max,
    pa_block_size,
    pa_stride,
    kv_prefix,
):
    """GetKVPhyAddr（csa_block_vector_arch35.h:146）逐 token 化简版。

    OT：按 KV_LAYOUT_T 派发 ori/cmp 两个 workspace 段的 Init 阶段全表生成。
    本仓 workspace 偏移由 host 侧拆成每段一张表（ori_phy / cmp_phy 两个张量
    参数），行偏移 = tok*aligned_cols 在 ForKvType 内计算，故此入口与
    ForKvType 一一对应、仅保留符号位（OT 的 ori/cmp 段 workspace 连续布局
    拆成两张独立张量（or 段 / cmp 段各一），数值与偏移语义一致）。
    """
    get_kv_phy_addr_for_kv_type(
        sync,
        bufs,
        phy_table,
        idx,
        bt,
        table_row,
        idx_row_width,
        ncols,
        aligned_cols,
        kv_layout,
        bt_seq,
        bt_max,
        pa_block_size,
        pa_stride,
        kv_prefix,
    )


@tla.jit
def gather_kv_rows(
    sync,
    bufs,
    ws,
    kv_tensor,
    idx,
    token_row,
    width,
    base_line,
    n_rows,
    row_base,
    slot,
    slot_stride,
    ws_base=0,
    announce: bool = True,
    block_table=None,
    bt_seq=0,
    bt_max=0,
    pa_block_size=0,
    pa_stride=0,
    topk_value_mode=1,
):
    """稀疏 gather：按索引行号把 KV 行搬到 GM 暂存槽（OT ProcessSparseKv，AIV 侧）。

    结构参照 arch35 vector 的 ProcessSparseKv：
      - 每个 tile 的 n_rows 行按 CeilDiv(n,2) 切给两个 AIV 子块（AIV0 前半、AIV1 后半）；
      - 以 16 行为单位（KV_PROCESS_UNIT）：两次 8 行 GM->UB（KV_COPYIN_UNIT），再一次 UB->GM；
      - 无效/越界行在 OT 里用 -1 填充，这里改为夹到 [0, count-1]——这些列随后被
        process_vec1 的 tail 掩码置为 MIN_F、P=0，数值上与 -1 填充逐位等价。
    行号来源：
      - TND（pa_block_size == 0）：idx 直接是 KV 行号（OT GetkeyOffset 的 TND 分支）；
      - PA_BBND（pa_block_size > 0）：idx 是**逻辑** KV 位置，
        `phys = block_table[bt_seq*bt_max + pos//block_size]`、
        `off = phys*pa_stride + (pos%block_size)*HEAD_DIM`（OT GetkeyOffset 的 PA 分支，
        pa_stride = oriKvStride = kv.stride(0) = block_size*n2*d）。
      - PA_BBND 且 topk_value_mode == 2（TOPK_OFFSET_MODE）：idx 值**就是物理块号**，
        `physBlockIdx = blkTableIdx`，不查 block table（OT 同一函数 :444-446）。
      - is_vec_s2phyaddr == 1（F2，OT GetkeyOffset 的 vectorize 路径）：物理地址
        来自 AIV 预计算的 int64 表（get_kv_phy_addr_for_kv_type 生成），见下方
        gather_kv_rows_vec（jit 参数不带 Constexpr 性，TLA 动态 if 分支结果必须
        同构，故 =1/=0 拆成两个函数、由 kernel 层 constexpr 分支二选一）。

    announce=False 用于独立调试 kernel：不发"暂存就绪"跨核 flag（没有 AIC 接收方时，
    set/wait 不配对会让硬件报 "The set and wait instructions do not match"）。
    """
    # 结构性约束（F45）：调用方必须先 configure_trace。没有 owner 说明该 kernel
    # 绕过了配对语义入口 ⇒ 直接编译失败，而不是继承上一次追踪的残留状态。
    # 断言放在 `with tla.vector()` 之外：TLA 前端禁止在动态控制流里构造异常
    # （实测报 "user-defined class 'TraceFidelityError' cannot be constructed in
    # dynamic Tla control flow"）。
    require_configured("gather_kv_rows")
    if _TRACE_STAGE:
        require_share("gather_kv_rows")
    with tla.vector():
        sub = tla.arch.sub_block_idx()
        first = (n_rows + 1) // 2
        if _TRACE_QUARTER:
            # A 步二（OT CalProcessSize，block_vector.h:754-777）：共享暂存下，
            # AIC 对的 4 个 AIV 各写 1/4 象限。block 奇偶 = G 半区，sub = 子块。
            par = tla.arch.block_idx() % 2
            h0 = (first + 1) // 2
            h1 = first - h0
            second = n_rows - first
            s0 = (second + 1) // 2
            s1 = second - s0
            my_start = 0
            my_rows = h0
            if sub == 1:
                my_start = h0
                my_rows = h1
            if par == 1:
                my_start = first
                my_rows = s0
                if sub == 1:
                    my_start = first + s0
                    my_rows = s1
        else:
            my_start = 0
            my_rows = first
            if sub == 1:
                my_start = first
                my_rows = n_rows - first
        n_chunks = (my_rows + KV_UNIT - 1) // KV_UNIT
        for c in tla.range(n_chunks):
            line0 = my_start + c * KV_UNIT
            # 尾块只拷出有效行数（OT: CopyOutKvUb2Gm(dealRow=tail16Rows)）
            rows_out = my_rows - c * KV_UNIT
            if rows_out > KV_UNIT:
                rows_out = tla.as_numeric(KV_UNIT)
            if rows_out < 0:
                rows_out = tla.as_numeric(0)
            # OT 的 ping-pong：stage0OutBufs[2]，每 16 行块换一次
            if c % 2 == 0:
                ub_kv = tla.make_tensor(
                    bufs.ub_kv0,
                    tla.make_layout(
                        tla.make_shape(KV_UNIT, HEAD_DIM),
                        tla.make_stride(HEAD_DIM, 1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                # 信用等待必须在 GM->UB 之前：放在 16 次拷入之后不管用，
                # 上一块的 UB->GM 还在读，本块的 GM->UB 已经把行盖掉了
                # （实测表现为暂存槽里混进后一块的行）。
                tla.wait_flag(sync.kv_mte3_mte2_0)
                for i in static_range(KV_UNIT):
                    line = line0 + i
                    if line > n_rows - 1:
                        line = n_rows - 1
                    if line < 0:
                        line = 0
                    src = idx[token_row * width + base_line + line]
                    if src < 0:
                        src = 0
                    row_off = (row_base + src) * HEAD_DIM
                    if pa_block_size > 0:
                        blk = src // pa_block_size
                        off = src - blk * pa_block_size
                        # OT GetkeyOffset（csa_block_vector_arch35.h:441-447）：
                        #   int32_t physBlockIdx = blkTableIdx;              // 默认
                        #   if constexpr (TOPK_VALUE_MODE == TOPK_INDEX_MODE)
                        #       physBlockIdx = blockTableGm.GetValue(boIdx*maxBlockNumPerBatch
                        #                                            + blkTableIdx);
                        # TOPK_OFFSET(2)：索引值本身就是物理块号，不查 block table。
                        phys = blk
                        if topk_value_mode == 1:
                            phys = block_table[bt_seq * bt_max + blk]
                            if phys < 0:
                                phys = 0
                        row_off = phys * pa_stride + off * HEAD_DIM
                    else:
                        row_off = (row_base + src) * HEAD_DIM
                    gm_row = tla.make_tensor(
                        kv_tensor.ptr + row_off,
                        tla.make_layout(
                            tla.make_shape(1, HEAD_DIM),
                            tla.make_stride(HEAD_DIM, 1),
                            layoutTag=tla.arch.RowMajor,
                        ),
                    )
                    tla.copy(
                        tla.tile_view(
                            ub_kv, tla.make_shape(1, HEAD_DIM), tla.make_coord(i, 0)
                        ),
                        gm_row,
                    )
                tla.set_flag(sync.kv_mte2_mte3_0)
                tla.wait_flag(sync.kv_mte2_mte3_0)
                gm_chunk = tla.make_tensor(
                    ws.ptr + ws_base + slot * slot_stride + line0 * HEAD_DIM,
                    tla.make_layout(
                        tla.make_shape(rows_out, HEAD_DIM),
                        tla.make_stride(HEAD_DIM, 1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                tla.copy(
                    gm_chunk,
                    tla.tile_view(
                        ub_kv, tla.make_shape(rows_out, HEAD_DIM), tla.make_coord(0, 0)
                    ),
                )
                tla.set_flag(sync.kv_mte3_mte2_0)
            else:
                ub_kv = tla.make_tensor(
                    bufs.ub_kv1,
                    tla.make_layout(
                        tla.make_shape(KV_UNIT, HEAD_DIM),
                        tla.make_stride(HEAD_DIM, 1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                tla.wait_flag(sync.kv_mte3_mte2_1)
                for i in static_range(KV_UNIT):
                    line = line0 + i
                    if line > n_rows - 1:
                        line = n_rows - 1
                    if line < 0:
                        line = 0
                    src = idx[token_row * width + base_line + line]
                    if src < 0:
                        src = 0
                    row_off = (row_base + src) * HEAD_DIM
                    if pa_block_size > 0:
                        blk = src // pa_block_size
                        off = src - blk * pa_block_size
                        # OT GetkeyOffset（csa_block_vector_arch35.h:441-447）：
                        #   int32_t physBlockIdx = blkTableIdx;              // 默认
                        #   if constexpr (TOPK_VALUE_MODE == TOPK_INDEX_MODE)
                        #       physBlockIdx = blockTableGm.GetValue(boIdx*maxBlockNumPerBatch
                        #                                            + blkTableIdx);
                        # TOPK_OFFSET(2)：索引值本身就是物理块号，不查 block table。
                        phys = blk
                        if topk_value_mode == 1:
                            phys = block_table[bt_seq * bt_max + blk]
                            if phys < 0:
                                phys = 0
                        row_off = phys * pa_stride + off * HEAD_DIM
                    else:
                        row_off = (row_base + src) * HEAD_DIM
                    gm_row = tla.make_tensor(
                        kv_tensor.ptr + row_off,
                        tla.make_layout(
                            tla.make_shape(1, HEAD_DIM),
                            tla.make_stride(HEAD_DIM, 1),
                            layoutTag=tla.arch.RowMajor,
                        ),
                    )
                    tla.copy(
                        tla.tile_view(
                            ub_kv, tla.make_shape(1, HEAD_DIM), tla.make_coord(i, 0)
                        ),
                        gm_row,
                    )
                tla.set_flag(sync.kv_mte2_mte3_1)
                tla.wait_flag(sync.kv_mte2_mte3_1)
                gm_chunk = tla.make_tensor(
                    ws.ptr + ws_base + slot * slot_stride + line0 * HEAD_DIM,
                    tla.make_layout(
                        tla.make_shape(rows_out, HEAD_DIM),
                        tla.make_stride(HEAD_DIM, 1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                tla.copy(
                    gm_chunk,
                    tla.tile_view(
                        ub_kv, tla.make_shape(rows_out, HEAD_DIM), tla.make_coord(0, 0)
                    ),
                )
                tla.set_flag(sync.kv_mte3_mte2_1)
        if announce:
            # 伙伴核读本核象限的跨核可见性由 **AIC 侧的全核会合**（pair_rendezvous）
            # 保证，AIV 侧不需要额外 PipeBarrier<ALL>：实测加了它 Dur 117→126.8
            # （mte2 30.6→37.9），且当时并未修好逐位（根因是预算整核只初始化一次）。
            tla.cross_core_set_flag(sync.kv_stage_ready, tla.pipes.MTE3, aiv_id=0)
            tla.cross_core_set_flag(sync.kv_stage_ready, tla.pipes.MTE3, aiv_id=1)


# E23: rot2 multi-vector-register rotation + pitch17 (VSSTB block_stride=17).
# E15/E17 的逐对 VLDI->VSSTB 因复用同一 V0 形成串行链（61.2 GB/s/核）。
# 这里用 2 组 V0/V1 与 2 条在飞 VLDI（post-increment #64 = 1024 元素）：
# 反汇编为 `VLDI V0,[S64],#64,#0,#1; VLDI V1,[S65],#64,#0,#1; VSSTB V0; VSSTB V1`，
# 实测 121.2 GB/s/核。目的布局 pitch17（块距 272 元素 = 544 B）避开 pitch16 的
# UB bank 冲突（pitch16 时同等指令只有 47.9 GB/s/核）。
_E17_KV_NZ_SRC = r"""
#include <cstdint>
#include "c_api/asc_simd.h"

extern "C" {
__aicore__ __attribute__((always_inline)) void kv_nz_vsstb_p16(
    uint64_t src_ub, uint64_t dst_ub) {
  __ubuf__ bfloat16_t* s = reinterpret_cast<__ubuf__ bfloat16_t*>(src_ub);
  __ubuf__ bfloat16_t* d = reinterpret_cast<__ubuf__ bfloat16_t*>(dst_ub);
  vector_bool mask = asc_create_mask_b16(PAT_ALL);
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    __ubuf__ bfloat16_t* sp0 = s + q * 128;
    __ubuf__ bfloat16_t* sp1 = sp0 + 512;
    __ubuf__ bfloat16_t* dp0 = d + q * 2176;
    __ubuf__ bfloat16_t* dp1 = dp0 + 16;
#pragma unroll
    for (int rr = 0; rr < 8; ++rr) {
      vector_bfloat16_t v0, v1;
      vlds(v0, sp0, 1024, NORM, POST_UPDATE);
      vlds(v1, sp1, 1024, NORM, POST_UPDATE);
      asc_storealign(dp0, v0, 17, 0, mask);
      asc_storealign(dp1, v1, 17, 0, mask);
      dp0 += 32;
      dp1 += 32;
    }
  }
}
} // extern "C"
"""


@tla.extern(source=_E17_KV_NZ_SRC)
def kv_nz_vsstb_p16(
    src_ub: tla.Pointer[tla.BFloat16, tla.AddressSpace.ub],
    dst_ub: tla.Pointer[tla.BFloat16, tla.AddressSpace.ub],
) -> None: ...


@tla.jit
def _ub_zero_tail_rows(ub_kv_ptr, n_rows, my_start, chunk_id: tla.Constexpr[int]):
    """Zero tail padding rows [rv..16) locally in UB via native SIMD.

    Replaces repeated GM MTE2 reads from the single 1 KB _kv_zero address, which
    caused severe 64-AIV L2 bank contention on small-H/tail-heavy CSA workloads,
    while preserving 0*NaN safety for masked KV positions.
    """
    if n_rows < KV_BLOCK:
        line0 = my_start + chunk_id * KV_UNIT
        rv = n_rows - line0
        if rv < 0:
            rv = tla.as_numeric(0)
        if rv < KV_UNIT:
            z_t = tla.make_tensor(
                tla.recast_ptr(ub_kv_ptr, dtype=tla.Float32),
                tla.make_layout(
                    tla.make_shape(KV_UNIT, HEAD_DIM // 2),
                    tla.make_stride(HEAD_DIM // 2, 1),
                ),
            )
            with tla.vec.func(mode="simd"):
                preg = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                z = tla.full(0.0, dtype=tla.Float32)
                for r in tla.range(KV_UNIT - rv):
                    rr = rv + r
                    tla.tile_view(
                        z_t, tla.make_shape(1, VL), tla.make_coord(rr, 0)
                    ).store(z, mask=preg)
                    tla.tile_view(
                        z_t, tla.make_shape(1, VL), tla.make_coord(rr, 1)
                    ).store(z, mask=preg)
                    tla.tile_view(
                        z_t, tla.make_shape(1, VL), tla.make_coord(rr, 2)
                    ).store(z, mask=preg)
                    tla.tile_view(
                        z_t, tla.make_shape(1, VL), tla.make_coord(rr, 3)
                    ).store(z, mask=preg)
                tla.local_mem_bar(
                    tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                )


@tla.jit
def kv_nz_vsstb_p16_u8(src_ub, dst_ub):
    """Native CATLASS-DSL 16x512 ND->zN transpose with pitch=17 via post-update VLDS/VSSTB."""
    s_ptr = tla.recast_ptr(src_ub, dtype=tla.BFloat16)
    d_ptr = tla.recast_ptr(dst_ub, dtype=tla.BFloat16)
    s0_t = tla.make_tensor(
        s_ptr, tla.make_layout(tla.make_shape(16 * HEAD_DIM), tla.make_stride(1))
    )
    s1_t = tla.make_tensor(
        s_ptr + 128,
        tla.make_layout(tla.make_shape(16 * HEAD_DIM - 128), tla.make_stride(1)),
    )
    s2_t = tla.make_tensor(
        s_ptr + 256,
        tla.make_layout(tla.make_shape(16 * HEAD_DIM - 256), tla.make_stride(1)),
    )
    s3_t = tla.make_tensor(
        s_ptr + 384,
        tla.make_layout(tla.make_shape(16 * HEAD_DIM - 384), tla.make_stride(1)),
    )
    d0_t = tla.make_tensor(
        d_ptr, tla.make_layout(tla.make_shape(2176), tla.make_stride(1))
    )
    d1_t = tla.make_tensor(
        d_ptr + 2176, tla.make_layout(tla.make_shape(2176), tla.make_stride(1))
    )
    d2_t = tla.make_tensor(
        d_ptr + 4352, tla.make_layout(tla.make_shape(2176), tla.make_stride(1))
    )
    d3_t = tla.make_tensor(
        d_ptr + 6528, tla.make_layout(tla.make_shape(2176), tla.make_stride(1))
    )
    with tla.vec.func(mode="simd"):
        mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.BFloat16)
        ld_u = tla.params.NormalLoadParams(
            post_mode=tla.params.PostMode.POST_MODE_UPDATE,
            post_update_stride=HEAD_DIM,
        )
        st_u = tla.params.BlockStoreParams(
            block_stride=KV_NZ_PITCH,
            post_mode=tla.params.PostMode.POST_MODE_UPDATE,
            repeat_stride=1,
        )
        sp0 = tla.tile_view(s0_t, tla.make_shape(VL_B16), tla.make_coord(0))
        sp1 = tla.tile_view(s1_t, tla.make_shape(VL_B16), tla.make_coord(0))
        sp2 = tla.tile_view(s2_t, tla.make_shape(VL_B16), tla.make_coord(0))
        sp3 = tla.tile_view(s3_t, tla.make_shape(VL_B16), tla.make_coord(0))
        dp0 = tla.tile_view(d0_t, tla.make_shape(VL_B16), tla.make_coord(0))
        dp1 = tla.tile_view(d1_t, tla.make_shape(VL_B16), tla.make_coord(0))
        dp2 = tla.tile_view(d2_t, tla.make_shape(VL_B16), tla.make_coord(0))
        dp3 = tla.tile_view(d3_t, tla.make_shape(VL_B16), tla.make_coord(0))
        a0, sp0 = sp0.load(ld_u)
        a1, sp1 = sp1.load(ld_u)
        a2, sp2 = sp2.load(ld_u)
        a3, sp3 = sp3.load(ld_u)
        for _ in static_range(7):
            b0, sp0 = sp0.load(ld_u)
            dp0 = dp0.store(a0, st_u, mask=mask)
            b1, sp1 = sp1.load(ld_u)
            dp1 = dp1.store(a1, st_u, mask=mask)
            b2, sp2 = sp2.load(ld_u)
            dp2 = dp2.store(a2, st_u, mask=mask)
            b3, sp3 = sp3.load(ld_u)
            dp3 = dp3.store(a3, st_u, mask=mask)
            a0, sp0 = sp0.load(ld_u)
            dp0 = dp0.store(b0, st_u, mask=mask)
            a1, sp1 = sp1.load(ld_u)
            dp1 = dp1.store(b1, st_u, mask=mask)
            a2, sp2 = sp2.load(ld_u)
            dp2 = dp2.store(b2, st_u, mask=mask)
            a3, sp3 = sp3.load(ld_u)
            dp3 = dp3.store(b3, st_u, mask=mask)
        b0, sp0 = sp0.load(ld_u)
        dp0 = dp0.store(a0, st_u, mask=mask)
        b1, sp1 = sp1.load(ld_u)
        dp1 = dp1.store(a1, st_u, mask=mask)
        b2, sp2 = sp2.load(ld_u)
        dp2 = dp2.store(a2, st_u, mask=mask)
        b3, sp3 = sp3.load(ld_u)
        dp3 = dp3.store(a3, st_u, mask=mask)
        dp0 = dp0.store(b0, st_u, mask=mask)
        dp1 = dp1.store(b1, st_u, mask=mask)
        dp2 = dp2.store(b2, st_u, mask=mask)
        dp3 = dp3.store(b3, st_u, mask=mask)


@tla.jit
def gather_kv_rows_pa_l1(
    sync,
    bufs,
    kv_tensor,
    idx,
    token_row,
    width,
    base_line,
    n_rows,
    slot,
    dtype,
    block_table,
    bt_seq,
    bt_max,
    pa_block_size,
    pa_stride,
    topk_value_mode,
    announce: bool = True,
):
    """E17：PA cmp KV 由 AIV gather 到 UB(ND) -> extern VSSTB 换形到 UB zN(pitch16)
    -> MTE3 写入本核 L1 K 槽 band（zN(128,512)）。

    数据面（每 AIV 子块，16 行 band）与同步见 RUNLOG §2/§3：
      wait kv_v_m2X（ND 可写）-> MTE2 页数学填 ND -> set kv_nd_vX
      -> V: wait kvz_free + wait kv_nd_vX -> extern 换形 -> set kv_v_m2X + set kvz_v2m3
      -> MTE3: wait kvz_v2m3 -> copy UB(zN) -> L1(zN) band -> set kvz_free
    页数学/寻址与 `gather_kv_rows_pa_pairwise` 逐字相同（复用 `_pa_pairwise_fill`），
    唯一区别是落位从 GM 暂存槽改成 L1 K 槽的 zN band；AIC 不再搬 cmp KV。
    2 块 ND ping-pong（ub_kv0/1），1 块 zN 暂存（ub_kvz）。
    """
    require_configured("gather_kv_rows_pa_pairwise")
    require_pa_configured("gather_kv_rows_pa_pairwise")
    if _TRACE_STAGE:
        require_share("gather_kv_rows_pa_pairwise")
    with tla.vector():
        sub = tla.arch.sub_block_idx()
        # 16 行对齐的子块切分：band 是 zN 的写入单位（否则两 AIV 踩同一 band）。
        # 生产 A/B 的 cmp 是 64/128 行，两 AIV 各 32/64 行，与 base 同。
        first = ((n_rows + 1) // 2 + KV_UNIT - 1) // KV_UNIT * KV_UNIT
        if first > n_rows:
            first = n_rows
        my_start = 0
        my_rows = first
        if sub == 1:
            my_start = first
            my_rows = n_rows - first
        n_chunks = (my_rows + KV_UNIT - 1) // KV_UNIT
        pair_zero = tla.Int64(0)
        pair_one = tla.Int64(1)
        pair_hi = tla.Int64(1073741824)
        # L1 K 槽 zN 视图（与 cube.l1_v_tensor 同布局；V≡K 复用同一批字节）。
        slot_zn = tla.make_tensor(
            bufs.l1_k + slot * (HEAD_DIM * KV_BLOCK),
            tla.make_layout_with_tag((KV_BLOCK, HEAD_DIM), dtype, tla.arch.zN),
        )
        # E23: UB 目的为 pitch17 的 zNUnAlign(17,512)：块距 = 17*16 = 272 元素
        # （544 B），避开 pitch16（512 B）的 UB bank 冲突。zNUnAlign 关闭 M 轴
        # fractal 对齐，按 rows*elems_per_c0 推导块距，正好是 272。
        ub_zn_base = tla.make_tensor(
            bufs.ub_kvz,
            tla.make_layout(
                tla.make_shape(KV_NZ_PITCH, HEAD_DIM),
                tla.make_stride(HEAD_DIM, 1),
                layoutTag=tla.arch.RowMajor,
            ),
        )
        ub_zn = tla.make_tensor_like(bufs.ub_kvz, ub_zn_base, tla.arch.zNUnAlign)
        # 只拷真实 16 行（第 17 行是每个 16 列块的 pad datablock）。
        ub_zn_band = tla.tile_view(
            ub_zn, tla.make_shape(KV_UNIT, HEAD_DIM), tla.make_coord(0, 0)
        )
        for c in tla.range(n_chunks):
            line0 = my_start + c * KV_UNIT
            if c % 2 == 0:
                ub_kv = tla.make_tensor(
                    bufs.ub_kv0,
                    tla.make_layout(
                        tla.make_shape(KV_UNIT, HEAD_DIM),
                        tla.make_stride(HEAD_DIM, 1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                tla.wait_flag(sync.kv_v_m20)  # MTE2：ND 缓冲可写
                _pa_pairwise_fill(
                    ub_kv,
                    bufs.ub_kv0,
                    kv_tensor,
                    idx,
                    token_row,
                    width,
                    base_line,
                    n_rows,
                    line0,
                    block_table,
                    bt_seq,
                    bt_max,
                    pa_block_size,
                    pa_stride,
                    topk_value_mode,
                    pair_zero,
                    pair_one,
                    pair_hi,
                )
                tla.set_flag(sync.kv_nd_v0)  # MTE2 -> V：ND 已填好
                tla.wait_flag(sync.kvz_free)  # V：zN 缓冲可写（首个 band 预置 1）
                tla.wait_flag(sync.kv_nd_v0)  # V：ND 已填好
                with tla.vec.func(mode="simd"):
                    kv_nz_vsstb_p16(bufs.ub_kv0, bufs.ub_kvz)
                tla.set_flag(sync.kv_v_m20)  # V -> MTE2：ND 已读完
                tla.set_flag(sync.kvz_v2m3)  # V -> MTE3：zN 已写完
                tla.wait_flag(sync.kvz_v2m3)  # MTE3：zN 就绪
                tla.copy(
                    tla.tile_view(
                        slot_zn,
                        tla.make_shape(KV_UNIT, HEAD_DIM),
                        tla.make_coord(line0 // KV_UNIT, 0),
                    ),
                    ub_zn_band,
                )
                tla.set_flag(sync.kvz_free)  # MTE3 -> V：zN 已读完
            else:
                ub_kv = tla.make_tensor(
                    bufs.ub_kv1,
                    tla.make_layout(
                        tla.make_shape(KV_UNIT, HEAD_DIM),
                        tla.make_stride(HEAD_DIM, 1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                tla.wait_flag(sync.kv_v_m21)
                _pa_pairwise_fill(
                    ub_kv,
                    bufs.ub_kv1,
                    kv_tensor,
                    idx,
                    token_row,
                    width,
                    base_line,
                    n_rows,
                    line0,
                    block_table,
                    bt_seq,
                    bt_max,
                    pa_block_size,
                    pa_stride,
                    topk_value_mode,
                    pair_zero,
                    pair_one,
                    pair_hi,
                )
                tla.set_flag(sync.kv_nd_v1)
                tla.wait_flag(sync.kvz_free)
                tla.wait_flag(sync.kv_nd_v1)
                with tla.vec.func(mode="simd"):
                    kv_nz_vsstb_p16(bufs.ub_kv1, bufs.ub_kvz)
                tla.set_flag(sync.kv_v_m21)
                tla.set_flag(sync.kvz_v2m3)
                tla.wait_flag(sync.kvz_v2m3)
                tla.copy(
                    tla.tile_view(
                        slot_zn,
                        tla.make_shape(KV_UNIT, HEAD_DIM),
                        tla.make_coord(line0 // KV_UNIT, 0),
                    ),
                    ub_zn_band,
                )
                tla.set_flag(sync.kvz_free)
        if announce:
            # 编译器按 sub_block_idx 给两条 set 加 guard；每个 AIV 只发自己那条。
            tla.cross_core_set_flag(sync.kv_stage_ready, tla.pipes.MTE3, aiv_id=0)
            tla.cross_core_set_flag(sync.kv_stage_ready, tla.pipes.MTE3, aiv_id=1)


@tla.jit
def gather_kv_rows_pa_pairwise(
    sync,
    bufs,
    ws,
    kv_tensor,
    idx,
    token_row,
    width,
    base_line,
    n_rows,
    slot,
    slot_stride,
    ws_base=0,
    announce: bool = True,
    block_table=None,
    bt_seq=0,
    bt_max=0,
    pa_block_size=0,
    pa_stride=0,
    topk_value_mode=1,
):
    """PA_BBND 内联页数学 + OT `CopyInKvSparse` 的「成对 + 物理地址升序」落位。

    OT 准绳（sparse_flash_mla_csa_block_vector_arch35.h，只读）：
      * `GetRealCmpS2Idx`/`GetRealS2Addr` :340-426：每 8 行块先读 8 个索引，
        尾块只填有效前缀，其余 tokenData 保持 -1；
      * `GetkeyOffset` :430-468：`s2Idx < 0 ⇒ -1`（**不 clamp**）；PA 分支
        `phys = block_table[boIdx*maxBlockNumPerBatch + s2Idx/blockSize]`、
        `off = phys*paBlockStride + (s2Idx%blockSize)*dSizeVInput`；
      * `CopyInKvSparse` :495-560：`i += 2` 成对处理
          - 两个都 -1 ⇒ 整对不搬（无效项是后缀，等价于剩余对也都不搬）
          - `blockCount = 1 + (off1 >= 0)`（只有尾对可能为 1）
          - 起址 `(off1 > -1 && off1 < off0) ? off1 : off0`
            ⇒ **降序对在 UB 里两行交换**（与表路径 `gather_kv_rows_vec` 的
            arm1 升序 / arm2 降序逐位同义）
          - `srcStride = |off0-off1|*sizeof(KV_T) - combineBytes`
          - stride 为负/溢出 ⇒ 回落 `CopyInSingleKv × 2`（逻辑行序）
    本函数保留内联页数学（不建表、不读表），只在 PA 实例使用；非 PA 路径仍走
    `gather_kv_rows` / `gather_kv_rows_ring` / `gather_kv_rows_vec`（逐字不变）。

    子块切分、16 行 ping-pong、信用 flag 配对与 `gather_kv_rows` 完全相同；
    成对边界按**子块内位置**对齐（`my_start` 在 n_rows 为奇数时可能是奇数，
    此时会出现跨全局偶数行的 pair，OT 的子块切分同样如此）。
    """
    require_configured("gather_kv_rows_pa_pairwise")
    require_pa_configured("gather_kv_rows_pa_pairwise")
    if _TRACE_STAGE:
        require_share("gather_kv_rows_pa_pairwise")
    with tla.vector():
        sub = tla.arch.sub_block_idx()
        first = (n_rows + 1) // 2
        if _TRACE_QUARTER:
            # 同 gather_kv_rows（OT CalProcessSize，block_vector.h:754-777）
            par = tla.arch.block_idx() % 2
            h0 = (first + 1) // 2
            h1 = first - h0
            second = n_rows - first
            s0 = (second + 1) // 2
            s1 = second - s0
            my_start = 0
            my_rows = h0
            if sub == 1:
                my_start = h0
                my_rows = h1
            if par == 1:
                my_start = first
                my_rows = s0
                if sub == 1:
                    my_start = first + s0
                    my_rows = s1
        else:
            my_start = 0
            my_rows = first
            if sub == 1:
                my_start = first
                my_rows = n_rows - first
        n_chunks = (my_rows + KV_UNIT - 1) // KV_UNIT
        pair_zero = tla.Int64(0)
        pair_one = tla.Int64(1)
        pair_hi = tla.Int64(1073741824)  # |Δoff| 上界，同表路径 pair_hi
        for c in tla.range(n_chunks):
            line0 = my_start + c * KV_UNIT
            rows_out = my_rows - c * KV_UNIT
            if rows_out > KV_UNIT:
                rows_out = tla.as_numeric(KV_UNIT)
            if rows_out < 0:
                rows_out = tla.as_numeric(0)
            if c % 2 == 0:
                ub_kv = tla.make_tensor(
                    bufs.ub_kv0,
                    tla.make_layout(
                        tla.make_shape(KV_UNIT, HEAD_DIM),
                        tla.make_stride(HEAD_DIM, 1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                tla.wait_flag(sync.kv_mte3_mte2_0)
                _pa_pairwise_fill(
                    ub_kv,
                    bufs.ub_kv0,
                    kv_tensor,
                    idx,
                    token_row,
                    width,
                    base_line,
                    n_rows,
                    line0,
                    block_table,
                    bt_seq,
                    bt_max,
                    pa_block_size,
                    pa_stride,
                    topk_value_mode,
                    pair_zero,
                    pair_one,
                    pair_hi,
                )
                tla.set_flag(sync.kv_mte2_mte3_0)
                tla.wait_flag(sync.kv_mte2_mte3_0)
                gm_chunk = tla.make_tensor(
                    ws.ptr + ws_base + slot * slot_stride + line0 * HEAD_DIM,
                    tla.make_layout(
                        tla.make_shape(rows_out, HEAD_DIM),
                        tla.make_stride(HEAD_DIM, 1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                tla.copy(
                    gm_chunk,
                    tla.tile_view(
                        ub_kv, tla.make_shape(rows_out, HEAD_DIM), tla.make_coord(0, 0)
                    ),
                )
                tla.set_flag(sync.kv_mte3_mte2_0)
            else:
                ub_kv = tla.make_tensor(
                    bufs.ub_kv1,
                    tla.make_layout(
                        tla.make_shape(KV_UNIT, HEAD_DIM),
                        tla.make_stride(HEAD_DIM, 1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                tla.wait_flag(sync.kv_mte3_mte2_1)
                _pa_pairwise_fill(
                    ub_kv,
                    bufs.ub_kv1,
                    kv_tensor,
                    idx,
                    token_row,
                    width,
                    base_line,
                    n_rows,
                    line0,
                    block_table,
                    bt_seq,
                    bt_max,
                    pa_block_size,
                    pa_stride,
                    topk_value_mode,
                    pair_zero,
                    pair_one,
                    pair_hi,
                )
                tla.set_flag(sync.kv_mte2_mte3_1)
                tla.wait_flag(sync.kv_mte2_mte3_1)
                gm_chunk = tla.make_tensor(
                    ws.ptr + ws_base + slot * slot_stride + line0 * HEAD_DIM,
                    tla.make_layout(
                        tla.make_shape(rows_out, HEAD_DIM),
                        tla.make_stride(HEAD_DIM, 1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                tla.copy(
                    gm_chunk,
                    tla.tile_view(
                        ub_kv, tla.make_shape(rows_out, HEAD_DIM), tla.make_coord(0, 0)
                    ),
                )
                tla.set_flag(sync.kv_mte3_mte2_1)
        if announce:
            tla.cross_core_set_flag(sync.kv_stage_ready, tla.pipes.MTE3, aiv_id=0)
            tla.cross_core_set_flag(sync.kv_stage_ready, tla.pipes.MTE3, aiv_id=1)


@tla.jit
def _pa_pairwise_fill_tail(
    ub_kv,
    ub_base,
    kv_tensor,
    idx,
    token_row,
    width,
    base_line,
    n_rows,
    line0,
    block_table,
    bt_seq,
    bt_max,
    pa_block_size,
    pa_stride,
    topk_value_mode,
    pair_zero,
    pair_one,
    pair_hi,
):
    """尾块的 16 行块填充（R42 步二后仅 `line0 + 16 > n_rows` 时走这里）。

    保留原有 3/4 臂：逐行 validity（`src_raw < 0` / 越界行）、两行都无效不搬、
    `blockCount = 1 + v1`、`|Δ|` 越界回落 CopyInSingleKv×2。
    """
    for blk in static_range(KV_UNIT // 8):
        blk_base = 8 * blk
        # GetRealCmpS2Idx（:340-386）：8 条地址背靠背读完；尾块无效项 = -1
        addr_t = ()
        for ai in static_range(8):
            line = line0 + blk_base + ai
            # 越界行不读 idx（OT 侧 tokenData 保持 -1），仅用夹取后的行号安全读
            if line > n_rows - 1:
                line = n_rows - 1
            src_raw = idx[token_row * width + base_line + line]
            src = src_raw
            # GetkeyOffset 先判 s2Idx < 0（:432），之后才允许除页号；这里夹 0
            # 只为避免负下标读 block table，结果由下面的 -1 覆盖。
            # 注：合并成无分支算式实测更亏（+1.37 µs，#32），保留短分支。
            if src < 0:
                src = 0
            # R42 步一：页号用追踪期常量做算术 —— block size 是 2 的幂，
            # `//block_size` 折叠成 `>>shift`；topk 模式与 pa_stride 也是常量。
            # 形参 pa_block_size/pa_stride/topk_value_mode 仅为兼容调用点保留。
            blk_i = src >> _PA_BLOCK_SHIFT
            off_i = src - blk_i * _PA_BLOCK_SIZE
            # TOPK_INDEX_MODE 查页表；TOPK_OFFSET_MODE 索引值即物理块号（:441-447）
            phys = blk_i
            if _PA_TOPK_MODE == 1:
                phys = block_table[bt_seq * bt_max + blk_i]
                if phys < 0:
                    phys = 0
            off = phys * _PA_STRIDE + off_i * HEAD_DIM
            # GetkeyOffset：s2Idx < 0 ⇒ -1；尾块越界行 tokenData 保持 -1
            if src_raw < 0:
                off = -1
            if line0 + blk_base + ai > n_rows - 1:
                off = -1
            addr_t = addr_t + (off.to(tla.Int64),)
        # CopyInKvSparse（:495-560）：i += 2
        for p in static_range(4):
            row_a = blk_base + 2 * p
            off0 = addr_t[2 * p]
            off1 = addr_t[2 * p + 1]
            # 复用子表达式：v1 同时用于「有效计数 / blockCount / 起址选择」，
            # desc 同时用于取绝对值与起址符号，pair_delta 只算一次。
            v0 = (off0 >= pair_zero).to(tla.Int64)
            v1 = (off1 >= pair_zero).to(tla.Int64)
            if v0 + v1 > pair_zero:
                pair_delta = off1 - off0
                c_desc = (pair_delta < pair_zero).to(tla.Int64)
                pair_abs = pair_delta * (pair_one - c_desc - c_desc)
                # 起址：仅当 off1 有效且更小时取 off1（OT 的 `&&` 语义）
                sel = v1 * c_desc
                pair_base = off0 + sel * pair_delta
                block_count = pair_one + v1
                pair_stride = pair_abs * tla.Int64(_PAIR_ELEM_BYTES) - tla.Int64(
                    HEAD_DIM * _PAIR_ELEM_BYTES
                )
                if pair_stride >= pair_zero and pair_stride < pair_hi:
                    pair_gm_to_ub(
                        tla.recast_ptr(kv_tensor.ptr + pair_base, dtype=tla.UInt8),
                        tla.recast_ptr(ub_base, dtype=tla.UInt8),
                        HEAD_DIM * _PAIR_ELEM_BYTES,
                        pair_stride,
                        tla.Int64(row_a * HEAD_DIM * _PAIR_ELEM_BYTES),
                        block_count,
                    )
                else:
                    # 回落 CopyInSingleKv×2（:526-531）：逻辑行序，offset<0 自己 return
                    if off0 >= pair_zero:
                        gm_row_a = tla.make_tensor(
                            kv_tensor.ptr + off0,
                            tla.make_layout(
                                tla.make_shape(1, HEAD_DIM),
                                tla.make_stride(HEAD_DIM, 1),
                                layoutTag=tla.arch.RowMajor,
                            ),
                        )
                        tla.copy(
                            tla.tile_view(
                                ub_kv,
                                tla.make_shape(1, HEAD_DIM),
                                tla.make_coord(row_a, 0),
                            ),
                            gm_row_a,
                        )
                    if off1 >= pair_zero:
                        gm_row_b = tla.make_tensor(
                            kv_tensor.ptr + off1,
                            tla.make_layout(
                                tla.make_shape(1, HEAD_DIM),
                                tla.make_stride(HEAD_DIM, 1),
                                layoutTag=tla.arch.RowMajor,
                            ),
                        )
                        tla.copy(
                            tla.tile_view(
                                ub_kv,
                                tla.make_shape(1, HEAD_DIM),
                                tla.make_coord(row_a + 1, 0),
                            ),
                            gm_row_b,
                        )
            # 两行都无效：OT return（剩余对同样无效），不搬、保留 UB 残留


@tla.jit
def _pa_pairwise_fill_full(
    ub_base,
    kv_tensor,
    idx,
    token_row,
    width,
    base_line,
    line0,
    block_table,
    bt_seq,
    bt_max,
    pair_zero,
    pair_one,
):
    """满块快路（R42 步二）：本块 16 行全部有效（`line0+16 <= n_rows`）。

    满块内不再出现 `s2Idx < 0` 与越界行，因此去掉 validity 计算与
    CopyInSingleKv×2 回落：每对只做 2 次 idx 读 + min/max 选起址 + 1 条
    `pair_gm_to_ub(blockCount=2)`。与慢路在满块上逐位同义：
      - 起址取较小地址、步长 |Δoff|*2-1024（降序对在 UB 内两行交换）；
      - 页表读在满块内恒定非负，不再夹 0（OT GetkeyOffset 也不夹）。
    满块地址差远小于描述符上界（phys < 2^31 / stride 为页字节数），
    不再保留 slow 路的 `pair_stride < pair_hi` 分支。
    """
    for blk in static_range(KV_UNIT // 8):
        blk_base = 8 * blk
        addr_t = ()
        for ai in static_range(8):
            line = line0 + blk_base + ai
            src = idx[token_row * width + base_line + line]
            blk_i = src >> _PA_BLOCK_SHIFT
            off_i = src - blk_i * _PA_BLOCK_SIZE
            phys = blk_i
            if _PA_TOPK_MODE == 1:
                phys = block_table[bt_seq * bt_max + blk_i]
            addr_t = addr_t + ((phys * _PA_STRIDE + off_i * HEAD_DIM).to(tla.Int64),)
        for p in static_range(4):
            row_a = blk_base + 2 * p
            off0 = addr_t[2 * p]
            off1 = addr_t[2 * p + 1]
            pair_delta = off1 - off0
            c_desc = (pair_delta < pair_zero).to(tla.Int64)
            pair_abs = pair_delta * (pair_one - c_desc - c_desc)
            pair_base = off0 + c_desc * pair_delta
            pair_stride = pair_abs * tla.Int64(_PAIR_ELEM_BYTES) - tla.Int64(
                HEAD_DIM * _PAIR_ELEM_BYTES
            )
            pair_gm_to_ub(
                tla.recast_ptr(kv_tensor.ptr + pair_base, dtype=tla.UInt8),
                tla.recast_ptr(ub_base, dtype=tla.UInt8),
                HEAD_DIM * _PAIR_ELEM_BYTES,
                pair_stride,
                tla.Int64(row_a * HEAD_DIM * _PAIR_ELEM_BYTES),
                tla.Int64(2),
            )


@tla.jit
def _pa_pairwise_fill(
    ub_kv,
    ub_base,
    kv_tensor,
    idx,
    token_row,
    width,
    base_line,
    n_rows,
    line0,
    block_table,
    bt_seq,
    bt_max,
    pa_block_size,
    pa_stride,
    topk_value_mode,
    pair_zero,
    pair_one,
    pair_hi,
):
    """16 行块填充的分派（R42 步二）：满块走单臂快路，尾块保留 3/4 臂。

    追踪期常量（`_PA_BLOCK_SHIFT/_PA_STRIDE/_PA_TOPK_MODE`）由 kernel 体首的
    `configure_pa(...)` 写入；形参 pa_block_size/pa_stride/topk_value_mode 仅为
    兼容调用点保留。
    """
    if line0 + KV_UNIT <= n_rows:
        _pa_pairwise_fill_full(
            ub_base,
            kv_tensor,
            idx,
            token_row,
            width,
            base_line,
            line0,
            block_table,
            bt_seq,
            bt_max,
            pair_zero,
            pair_one,
        )
    else:
        _pa_pairwise_fill_tail(
            ub_kv,
            ub_base,
            kv_tensor,
            idx,
            token_row,
            width,
            base_line,
            n_rows,
            line0,
            block_table,
            bt_seq,
            bt_max,
            pa_block_size,
            pa_stride,
            topk_value_mode,
            pair_zero,
            pair_one,
            pair_hi,
        )


@tla.jit
def gather_kv_rows_vec(
    sync,
    bufs,
    ws,
    kv_tensor,
    phy_table,
    token_row,
    phy_width,
    base_line,
    n_rows,
    slot,
    slot_stride,
    ws_base=0,
    announce: bool = True,
):
    """F2 的 gather 消费者（IS_VEC_S2PHYADDR=1 实例）：与 gather_kv_rows 结构
    完全同构（CeilDiv(n,2) 子块切分 / 16 行 ping-pong / KV_COPY_IN 8 行 / 信用
    等待全部一致），唯一差别是行地址来自预计算的 int64 表：
        row_off = phy_table[tok * phy_width + base_line + line]
    （表行 tok 覆盖列 [0, lines)，clamp src<0→0 / phys<0→0 已在生成侧完成，
    与内联路径在全部被消费列上逐位一致——改前/改后同值证明见 ITERATIONS.md）。
    拆成独立函数的原因：@tla.jit 的参数不带 Constexpr 性，`if is_vec==1` 会被
    当成动态 if，TLA 要求动态 if 的分支结果同构（Int64 表值 vs Int32 行内计算
    无法合并），故由 kernel 层的 constexpr 分支二选一，=0 实例的 IR 里完全不
    含表读取路径（与 OT 的 =0 编译实例一致）。
    """
    require_configured("gather_kv_rows_vec")
    if _TRACE_STAGE:
        require_share("gather_kv_rows_vec")
    with tla.vector():
        sub = tla.arch.sub_block_idx()
        first = (n_rows + 1) // 2
        if _TRACE_QUARTER:
            # A 步二（OT CalProcessSize，block_vector.h:754-777）：共享暂存下，
            # AIC 对的 4 个 AIV 各写 1/4 象限。block 奇偶 = G 半区，sub = 子块。
            par = tla.arch.block_idx() % 2
            h0 = (first + 1) // 2
            h1 = first - h0
            second = n_rows - first
            s0 = (second + 1) // 2
            s1 = second - s0
            my_start = 0
            my_rows = h0
            if sub == 1:
                my_start = h0
                my_rows = h1
            if par == 1:
                my_start = first
                my_rows = s0
                if sub == 1:
                    my_start = first + s0
                    my_rows = s1
        else:
            my_start = 0
            my_rows = first
            if sub == 1:
                my_start = first
                my_rows = n_rows - first
        n_chunks = (my_rows + KV_UNIT - 1) // KV_UNIT
        for c in tla.range(n_chunks):
            line0 = my_start + c * KV_UNIT
            rows_out = my_rows - c * KV_UNIT
            if rows_out > KV_UNIT:
                rows_out = tla.as_numeric(KV_UNIT)
            if rows_out < 0:
                rows_out = tla.as_numeric(0)
            if c % 2 == 0:
                ub_kv = tla.make_tensor(
                    bufs.ub_kv0,
                    tla.make_layout(
                        tla.make_shape(KV_UNIT, HEAD_DIM),
                        tla.make_stride(HEAD_DIM, 1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                tla.wait_flag(sync.kv_mte3_mte2_0)
                if _SPARSE_BLOCK_SIZE == 1:
                    # F31 ①c：OT 原形是**两段叠加**，不是二选一：
                    #   GetRealS2Addr 每 8 行块先整块读 8 个地址
                    #   （`for i in 0..7: tokenData[i] = phyAddrGm64.GetValue(idx+i)`，
                    #    IS_FULL 快路不 clamp），CopyInKvSparse 随后以 `i += 2` 发
                    #   4 条合并 DataCopyPad。我方 KV_UNIT=16 = 2 个 OT 8 行块。
                    for blk in static_range(KV_UNIT // 8):
                        blk_base = 8 * blk
                        # 整块 8 条地址先背靠背读完（尾块按行 clamp，与逐行路径一致）
                        addr_t = ()
                        for ai in static_range(8):
                            aline = line0 + blk_base + ai
                            if aline > n_rows - 1:
                                aline = n_rows - 1
                            if aline < 0:
                                aline = 0
                            addr_t = addr_t + (
                                phy_table[token_row * phy_width + base_line + aline],
                            )
                        pair_lo = tla.Int64(HEAD_DIM)
                        pair_hi = tla.Int64(1073741824)
                        pair_zero = tla.Int64(0)
                        for p in static_range(4):
                            row_a = blk_base + 2 * p
                            off_a = addr_t[2 * p]
                            off_b = addr_t[2 * p + 1]
                            pair_delta = off_b - off_a
                            valid_a = row_a <= n_rows - 1 - line0
                            valid_b = valid_a and row_a + 1 <= n_rows - 1 - line0
                            # OT 默认口径（IS_BATCH_CONSISTENCY=false）：stride 取 |Δoff|、
                            # 起始取较小地址 ⇒ 升序与降序都合并，降序时 UB 两行互换；
                            # 门限 |Δ|*2-1024 ∈ [0, INT32_MAX)，否则 CopyInSingleKv×2。
                            if (
                                valid_b
                                and pair_delta >= pair_lo
                                and pair_delta < pair_hi
                            ):
                                pair_gm_to_ub(
                                    tla.recast_ptr(
                                        kv_tensor.ptr + off_a, dtype=tla.UInt8
                                    ),
                                    tla.recast_ptr(bufs.ub_kv0, dtype=tla.UInt8),
                                    HEAD_DIM * _PAIR_ELEM_BYTES,
                                    pair_delta * tla.Int64(_PAIR_ELEM_BYTES)
                                    - tla.Int64(HEAD_DIM * _PAIR_ELEM_BYTES),
                                    tla.Int64(row_a * HEAD_DIM * _PAIR_ELEM_BYTES),
                                    tla.Int64(2),
                                )
                            elif (
                                valid_b
                                and pair_zero - pair_delta >= pair_lo
                                and pair_zero - pair_delta < pair_hi
                            ):
                                # 降序：OT 取 min 起始（=off_b），UB 行序互换
                                pair_gm_to_ub(
                                    tla.recast_ptr(
                                        kv_tensor.ptr + off_b, dtype=tla.UInt8
                                    ),
                                    tla.recast_ptr(bufs.ub_kv0, dtype=tla.UInt8),
                                    HEAD_DIM * _PAIR_ELEM_BYTES,
                                    (pair_zero - pair_delta)
                                    * tla.Int64(_PAIR_ELEM_BYTES)
                                    - tla.Int64(HEAD_DIM * _PAIR_ELEM_BYTES),
                                    tla.Int64(row_a * HEAD_DIM * _PAIR_ELEM_BYTES),
                                    tla.Int64(2),
                                )
                            elif valid_b:
                                # stride 越界/过小：OT 回落 CopyInSingleKv×2
                                # （用它已读好的 tokenData，不重读表；保逻辑行序）
                                for ri in static_range(2):
                                    row_off = addr_t[2 * p + ri]
                                    gm_row = tla.make_tensor(
                                        kv_tensor.ptr + row_off,
                                        tla.make_layout(
                                            tla.make_shape(1, HEAD_DIM),
                                            tla.make_stride(HEAD_DIM, 1),
                                            layoutTag=tla.arch.RowMajor,
                                        ),
                                    )
                                    tla.copy(
                                        tla.tile_view(
                                            ub_kv,
                                            tla.make_shape(1, HEAD_DIM),
                                            tla.make_coord(row_a + ri, 0),
                                        ),
                                        gm_row,
                                    )
                            elif valid_a:
                                # 尾块只剩首行（OT 侧 off1=-1）：blockCount=1、src=off0；
                                # blockCount=1 时 stride 字段不参与搬运。
                                pair_gm_to_ub(
                                    tla.recast_ptr(
                                        kv_tensor.ptr + off_a, dtype=tla.UInt8
                                    ),
                                    tla.recast_ptr(bufs.ub_kv0, dtype=tla.UInt8),
                                    HEAD_DIM * _PAIR_ELEM_BYTES,
                                    pair_zero,
                                    tla.Int64(row_a * HEAD_DIM * _PAIR_ELEM_BYTES),
                                    tla.Int64(1),
                                )
                            # 两行都无效：OT 直接 return，不发搬运
                else:
                    for i in static_range(KV_UNIT):
                        line = line0 + i
                        if line > n_rows - 1:
                            line = n_rows - 1
                        if line < 0:
                            line = 0
                        # F2：物理地址来自预计算表（Int64 元素数偏移，ptr_add 接受 i64）
                        row_off = phy_table[token_row * phy_width + base_line + line]
                        gm_row = tla.make_tensor(
                            kv_tensor.ptr + row_off,
                            tla.make_layout(
                                tla.make_shape(1, HEAD_DIM),
                                tla.make_stride(HEAD_DIM, 1),
                                layoutTag=tla.arch.RowMajor,
                            ),
                        )
                        tla.copy(
                            tla.tile_view(
                                ub_kv, tla.make_shape(1, HEAD_DIM), tla.make_coord(i, 0)
                            ),
                            gm_row,
                        )
                tla.set_flag(sync.kv_mte2_mte3_0)
                tla.wait_flag(sync.kv_mte2_mte3_0)
                gm_chunk = tla.make_tensor(
                    ws.ptr + ws_base + slot * slot_stride + line0 * HEAD_DIM,
                    tla.make_layout(
                        tla.make_shape(rows_out, HEAD_DIM),
                        tla.make_stride(HEAD_DIM, 1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                tla.copy(
                    gm_chunk,
                    tla.tile_view(
                        ub_kv, tla.make_shape(rows_out, HEAD_DIM), tla.make_coord(0, 0)
                    ),
                )
                tla.set_flag(sync.kv_mte3_mte2_0)
            else:
                ub_kv = tla.make_tensor(
                    bufs.ub_kv1,
                    tla.make_layout(
                        tla.make_shape(KV_UNIT, HEAD_DIM),
                        tla.make_stride(HEAD_DIM, 1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                tla.wait_flag(sync.kv_mte3_mte2_1)
                if _SPARSE_BLOCK_SIZE == 1:
                    # F31 ①c：同 kv0 分支（读 8 地址 → 4 条合并拷贝，ub_kv1 目标）
                    for blk in static_range(KV_UNIT // 8):
                        blk_base = 8 * blk
                        addr_t = ()
                        for ai in static_range(8):
                            aline = line0 + blk_base + ai
                            if aline > n_rows - 1:
                                aline = n_rows - 1
                            if aline < 0:
                                aline = 0
                            addr_t = addr_t + (
                                phy_table[token_row * phy_width + base_line + aline],
                            )
                        pair_lo = tla.Int64(HEAD_DIM)
                        pair_hi = tla.Int64(1073741824)
                        pair_zero = tla.Int64(0)
                        for p in static_range(4):
                            row_a = blk_base + 2 * p
                            off_a = addr_t[2 * p]
                            off_b = addr_t[2 * p + 1]
                            pair_delta = off_b - off_a
                            valid_a = row_a <= n_rows - 1 - line0
                            valid_b = valid_a and row_a + 1 <= n_rows - 1 - line0
                            # 门限与降序口径同 kv0（OT 默认 IS_BATCH_CONSISTENCY=false）
                            if (
                                valid_b
                                and pair_delta >= pair_lo
                                and pair_delta < pair_hi
                            ):
                                pair_gm_to_ub(
                                    tla.recast_ptr(
                                        kv_tensor.ptr + off_a, dtype=tla.UInt8
                                    ),
                                    tla.recast_ptr(bufs.ub_kv1, dtype=tla.UInt8),
                                    HEAD_DIM * _PAIR_ELEM_BYTES,
                                    pair_delta * tla.Int64(_PAIR_ELEM_BYTES)
                                    - tla.Int64(HEAD_DIM * _PAIR_ELEM_BYTES),
                                    tla.Int64(row_a * HEAD_DIM * _PAIR_ELEM_BYTES),
                                    tla.Int64(2),
                                )
                            elif (
                                valid_b
                                and pair_zero - pair_delta >= pair_lo
                                and pair_zero - pair_delta < pair_hi
                            ):
                                # 降序：OT 取 min 起始（=off_b），UB 行序互换
                                pair_gm_to_ub(
                                    tla.recast_ptr(
                                        kv_tensor.ptr + off_b, dtype=tla.UInt8
                                    ),
                                    tla.recast_ptr(bufs.ub_kv1, dtype=tla.UInt8),
                                    HEAD_DIM * _PAIR_ELEM_BYTES,
                                    (pair_zero - pair_delta)
                                    * tla.Int64(_PAIR_ELEM_BYTES)
                                    - tla.Int64(HEAD_DIM * _PAIR_ELEM_BYTES),
                                    tla.Int64(row_a * HEAD_DIM * _PAIR_ELEM_BYTES),
                                    tla.Int64(2),
                                )
                            elif valid_b:
                                # stride 越界/过小：回落 CopyInSingleKv×2（用已读地址）
                                for ri in static_range(2):
                                    row_off = addr_t[2 * p + ri]
                                    gm_row = tla.make_tensor(
                                        kv_tensor.ptr + row_off,
                                        tla.make_layout(
                                            tla.make_shape(1, HEAD_DIM),
                                            tla.make_stride(HEAD_DIM, 1),
                                            layoutTag=tla.arch.RowMajor,
                                        ),
                                    )
                                    tla.copy(
                                        tla.tile_view(
                                            ub_kv,
                                            tla.make_shape(1, HEAD_DIM),
                                            tla.make_coord(row_a + ri, 0),
                                        ),
                                        gm_row,
                                    )
                            elif valid_a:
                                # 尾块只剩首行（OT 侧 off1=-1）：blockCount=1、src=off0
                                pair_gm_to_ub(
                                    tla.recast_ptr(
                                        kv_tensor.ptr + off_a, dtype=tla.UInt8
                                    ),
                                    tla.recast_ptr(bufs.ub_kv1, dtype=tla.UInt8),
                                    HEAD_DIM * _PAIR_ELEM_BYTES,
                                    pair_zero,
                                    tla.Int64(row_a * HEAD_DIM * _PAIR_ELEM_BYTES),
                                    tla.Int64(1),
                                )
                            # 两行都无效：OT 直接 return，不发搬运
                else:
                    for i in static_range(KV_UNIT):
                        line = line0 + i
                        if line > n_rows - 1:
                            line = n_rows - 1
                        if line < 0:
                            line = 0
                        row_off = phy_table[token_row * phy_width + base_line + line]
                        gm_row = tla.make_tensor(
                            kv_tensor.ptr + row_off,
                            tla.make_layout(
                                tla.make_shape(1, HEAD_DIM),
                                tla.make_stride(HEAD_DIM, 1),
                                layoutTag=tla.arch.RowMajor,
                            ),
                        )
                        tla.copy(
                            tla.tile_view(
                                ub_kv, tla.make_shape(1, HEAD_DIM), tla.make_coord(i, 0)
                            ),
                            gm_row,
                        )
                tla.set_flag(sync.kv_mte2_mte3_1)
                tla.wait_flag(sync.kv_mte2_mte3_1)
                gm_chunk = tla.make_tensor(
                    ws.ptr + ws_base + slot * slot_stride + line0 * HEAD_DIM,
                    tla.make_layout(
                        tla.make_shape(rows_out, HEAD_DIM),
                        tla.make_stride(HEAD_DIM, 1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                tla.copy(
                    gm_chunk,
                    tla.tile_view(
                        ub_kv, tla.make_shape(rows_out, HEAD_DIM), tla.make_coord(0, 0)
                    ),
                )
                tla.set_flag(sync.kv_mte3_mte2_1)
        if announce:
            # 伙伴核读本核象限的跨核可见性由 **AIC 侧的全核会合**（pair_rendezvous）
            # 保证，AIV 侧不需要额外 PipeBarrier<ALL>：实测加了它 Dur 117→126.8
            # （mte2 30.6→37.9），且当时并未修好逐位（根因是预算整核只初始化一次）。
            tla.cross_core_set_flag(sync.kv_stage_ready, tla.pipes.MTE3, aiv_id=0)
            tla.cross_core_set_flag(sync.kv_stage_ready, tla.pipes.MTE3, aiv_id=1)


# ---------------------------------------------------------------------------
# TND/BSND 16-row Burst Gather (Arm 1: ws, Arm 2: direct AIV->L1)
# ---------------------------------------------------------------------------
_TND_BURST_SRC = r"""
#include <cstdint>
#include "c_api/asc_simd.h"

extern "C" {

__aicore__ __attribute__((always_inline)) void tnd_gather_tail_rows_ub_l2_1(
    uint64_t kv_gm_addr,
    uint64_t ub_kv_addr,
    uint64_t ub_idx_addr,
    uint64_t kv_zero_addr,
    int64_t row_base,
    int64_t rows_valid) {
  __gm__ uint8_t* kv_base = reinterpret_cast<__gm__ uint8_t*>(kv_gm_addr);
  (void)kv_zero_addr;
  int32_t rv = static_cast<int32_t>(rows_valid);
  if (rv <= 0) return;
  __ubuf__ uint8_t* ub_base = reinterpret_cast<__ubuf__ uint8_t*>(ub_kv_addr);
  __ubuf__ int32_t* idx_ptr = reinterpret_cast<__ubuf__ int32_t*>(ub_idx_addr);
  uint64_t base_u = static_cast<uint64_t>(row_base);
#pragma unroll
  for (int p = 0; p < 8; ++p) {
    int32_t a = 2 * p;
    int32_t b = a + 1;
    if (b < rv) {
      int32_t i0 = idx_ptr[a];
      int32_t i1 = idx_ptr[b];
      uint64_t r0 = base_u + static_cast<uint64_t>((i0 < 0) ? 0 : i0);
      uint64_t r1 = base_u + static_cast<uint64_t>((i1 < 0) ? 0 : i1);
      uint64_t lo = (r0 < r1) ? r0 : r1;
      uint64_t hi = (r0 < r1) ? r1 : r0;
      uint64_t src_off = lo << 10;
      uint64_t src_stride = (hi - lo) << 10;
      copy_gm_to_ubuf_align_v2(
          reinterpret_cast<__ubuf__ bfloat16_t*>(ub_base + p * 2048),
          reinterpret_cast<__gm__ bfloat16_t*>(kv_base + src_off),
          0, 2, 1024, 0, 0, false, static_cast<uint8_t>(1), src_stride, 1024);
    } else if (a < rv) {
      int32_t i0 = idx_ptr[a];
      uint64_t r0 = base_u + static_cast<uint64_t>((i0 < 0) ? 0 : i0);
      copy_gm_to_ubuf_align_v2(
          reinterpret_cast<__ubuf__ bfloat16_t*>(ub_base + p * 2048),
          reinterpret_cast<__gm__ bfloat16_t*>(kv_base + (r0 << 10)),
          0, 1, 1024, 0, 0, false, static_cast<uint8_t>(1), 0, 0);
    }
  }
}

__aicore__ __attribute__((always_inline)) void pa_gather_16rows_mlp2_l2_1(
    uint64_t kv_gm_addr, uint64_t bt_row_addr, uint64_t ub_kv_addr, uint64_t ub_idx_addr,
    int64_t pa_block_shift, int64_t pa_block_mask, int64_t pa_stride_bytes, int64_t topk_mode) {
  __gm__ uint8_t* __restrict__ kv_base = reinterpret_cast<__gm__ uint8_t*>(kv_gm_addr);
  __gm__ int32_t* __restrict__ bt_row = reinterpret_cast<__gm__ int32_t*>(bt_row_addr);
  __ubuf__ uint8_t* __restrict__ dst_p = reinterpret_cast<__ubuf__ uint8_t*>(ub_kv_addr);
  __ubuf__ int32_t* __restrict__ idx_ptr = reinterpret_cast<__ubuf__ int32_t*>(ub_idx_addr);
  uint32_t sh = static_cast<uint32_t>(pa_block_shift);
  uint32_t msk = static_cast<uint32_t>(pa_block_mask);
  uint64_t sb = static_cast<uint64_t>(pa_stride_bytes);
  bool use_bt = (topk_mode == 1);
#pragma clang loop unroll(disable)
  for (int q = 0; q < 4; ++q) {
    int32_t i0 = idx_ptr[0], i1 = idx_ptr[1], i2 = idx_ptr[2], i3 = idx_ptr[3];
    idx_ptr += 4;
    uint32_t a0 = static_cast<uint32_t>((i0 < 0) ? 0 : i0);
    uint32_t a1 = static_cast<uint32_t>((i1 < 0) ? 0 : i1);
    uint32_t a2 = static_cast<uint32_t>((i2 < 0) ? 0 : i2);
    uint32_t a3 = static_cast<uint32_t>((i3 < 0) ? 0 : i3);
    int32_t p0, p1, p2, p3;
    if (use_bt) {
      p0 = bt_row[a0 >> sh]; p1 = bt_row[a1 >> sh];
      p2 = bt_row[a2 >> sh]; p3 = bt_row[a3 >> sh];
      p0 = (p0 < 0) ? 0 : p0; p1 = (p1 < 0) ? 0 : p1;
      p2 = (p2 < 0) ? 0 : p2; p3 = (p3 < 0) ? 0 : p3;
    } else {
      p0 = static_cast<int32_t>(a0 >> sh); p1 = static_cast<int32_t>(a1 >> sh);
      p2 = static_cast<int32_t>(a2 >> sh); p3 = static_cast<int32_t>(a3 >> sh);
    }
    uint64_t off0 = static_cast<uint64_t>(p0) * sb + (static_cast<uint64_t>(a0 & msk) << 10);
    uint64_t off1 = static_cast<uint64_t>(p1) * sb + (static_cast<uint64_t>(a1 & msk) << 10);
    uint64_t lo0 = (off0 < off1) ? off0 : off1;
    uint64_t hi0 = (off0 < off1) ? off1 : off0;
    copy_gm_to_ubuf_align_v2(
        reinterpret_cast<__ubuf__ bfloat16_t*>(dst_p),
        reinterpret_cast<__gm__ bfloat16_t*>(kv_base + lo0),
        0, 2, 1024, 0, 0, false, static_cast<uint8_t>(1), hi0 - lo0, 1024);

    uint64_t off2 = static_cast<uint64_t>(p2) * sb + (static_cast<uint64_t>(a2 & msk) << 10);
    uint64_t off3 = static_cast<uint64_t>(p3) * sb + (static_cast<uint64_t>(a3 & msk) << 10);
    uint64_t lo1 = (off2 < off3) ? off2 : off3;
    uint64_t hi1 = (off2 < off3) ? off3 : off2;
    copy_gm_to_ubuf_align_v2(
        reinterpret_cast<__ubuf__ bfloat16_t*>(dst_p + 2048),
        reinterpret_cast<__gm__ bfloat16_t*>(kv_base + lo1),
        0, 2, 1024, 0, 0, false, static_cast<uint8_t>(1), hi1 - lo1, 1024);
    dst_p += 4096;
  }
}

__aicore__ __attribute__((always_inline)) void pa_gather_tail_rows_ub_l2_1(
    uint64_t kv_gm_addr, uint64_t bt_row_addr, uint64_t ub_kv_addr, uint64_t ub_idx_addr,
    uint64_t kv_zero_addr, int64_t pa_block_shift, int64_t pa_block_mask,
    int64_t pa_stride_bytes, int64_t topk_mode, int64_t rows_valid) {
  __gm__ uint8_t* __restrict__ kv_base = reinterpret_cast<__gm__ uint8_t*>(kv_gm_addr);
  (void)kv_zero_addr;
  int32_t rv = static_cast<int32_t>(rows_valid);
  if (rv <= 0) return;
  __gm__ int32_t* __restrict__ bt_row = reinterpret_cast<__gm__ int32_t*>(bt_row_addr);
  __ubuf__ uint8_t* __restrict__ dst_p = reinterpret_cast<__ubuf__ uint8_t*>(ub_kv_addr);
  __ubuf__ int32_t* __restrict__ idx_ptr = reinterpret_cast<__ubuf__ int32_t*>(ub_idx_addr);
  uint32_t sh = static_cast<uint32_t>(pa_block_shift);
  uint32_t msk = static_cast<uint32_t>(pa_block_mask);
  uint64_t sb = static_cast<uint64_t>(pa_stride_bytes);
  bool use_bt = (topk_mode == 1);
#pragma clang loop unroll(disable)
  for (int p = 0; p < 8; ++p) {
    int32_t a = 2 * p;
    int32_t b = a + 1;
    if (b < rv) {
      int32_t i0 = idx_ptr[a];
      int32_t i1 = idx_ptr[b];
      uint32_t a0 = static_cast<uint32_t>((i0 < 0) ? 0 : i0);
      uint32_t a1 = static_cast<uint32_t>((i1 < 0) ? 0 : i1);
      int32_t p0 = use_bt ? bt_row[a0 >> sh] : static_cast<int32_t>(a0 >> sh);
      int32_t p1 = use_bt ? bt_row[a1 >> sh] : static_cast<int32_t>(a1 >> sh);
      p0 = (p0 < 0) ? 0 : p0;
      p1 = (p1 < 0) ? 0 : p1;
      uint64_t off0 = static_cast<uint64_t>(p0) * sb + (static_cast<uint64_t>(a0 & msk) << 10);
      uint64_t off1 = static_cast<uint64_t>(p1) * sb + (static_cast<uint64_t>(a1 & msk) << 10);
      uint64_t lo = (off0 < off1) ? off0 : off1;
      uint64_t hi = (off0 < off1) ? off1 : off0;
      copy_gm_to_ubuf_align_v2(
          reinterpret_cast<__ubuf__ bfloat16_t*>(dst_p),
          reinterpret_cast<__gm__ bfloat16_t*>(kv_base + lo),
          0, 2, 1024, 0, 0, false, static_cast<uint8_t>(1), hi - lo, 1024);
    } else if (a < rv) {
      int32_t i0 = idx_ptr[a];
      uint32_t a0 = static_cast<uint32_t>((i0 < 0) ? 0 : i0);
      int32_t p0 = use_bt ? bt_row[a0 >> sh] : static_cast<int32_t>(a0 >> sh);
      p0 = (p0 < 0) ? 0 : p0;
      uint64_t off0 = static_cast<uint64_t>(p0) * sb + (static_cast<uint64_t>(a0 & msk) << 10);
      copy_gm_to_ubuf_align_v2(
          reinterpret_cast<__ubuf__ bfloat16_t*>(dst_p),
          reinterpret_cast<__gm__ bfloat16_t*>(kv_base + off0),
          0, 1, 1024, 0, 0, false, static_cast<uint8_t>(1), 0, 0);
    }
    dst_p += 2048;
  }
}

__aicore__ __attribute__((always_inline)) void tnd_gather_16rows_mlp4_l2_1(
    uint64_t kv_gm_addr, uint64_t ub_kv_addr, uint64_t ub_idx_addr, int64_t row_base) {
  __gm__ uint8_t* __restrict__ kv_base =
      reinterpret_cast<__gm__ uint8_t*>(kv_gm_addr) + (static_cast<uint64_t>(row_base) << 10);
  __ubuf__ uint8_t* __restrict__ dst_p = reinterpret_cast<__ubuf__ uint8_t*>(ub_kv_addr);
  __ubuf__ int32_t* __restrict__ idx_ptr = reinterpret_cast<__ubuf__ int32_t*>(ub_idx_addr);
#pragma clang loop unroll(disable)
  for (int h = 0; h < 2; ++h) {
    int64_t i0 = idx_ptr[0], i1 = idx_ptr[1], i2 = idx_ptr[2], i3 = idx_ptr[3];
    int64_t i4 = idx_ptr[4], i5 = idx_ptr[5], i6 = idx_ptr[6], i7 = idx_ptr[7];
    idx_ptr += 8;
    int64_t a0 = (i0 < 0) ? 0 : i0, a1 = (i1 < 0) ? 0 : i1;
    int64_t a2 = (i2 < 0) ? 0 : i2, a3 = (i3 < 0) ? 0 : i3;
    int64_t a4 = (i4 < 0) ? 0 : i4, a5 = (i5 < 0) ? 0 : i5;
    int64_t a6 = (i6 < 0) ? 0 : i6, a7 = (i7 < 0) ? 0 : i7;
    int64_t lo0 = (a0 < a1) ? a0 : a1, hi0 = (a0 < a1) ? a1 : a0;
    int64_t lo1 = (a2 < a3) ? a2 : a3, hi1 = (a2 < a3) ? a3 : a2;
    int64_t lo2 = (a4 < a5) ? a4 : a5, hi2 = (a4 < a5) ? a5 : a4;
    int64_t lo3 = (a6 < a7) ? a6 : a7, hi3 = (a6 < a7) ? a7 : a6;
    copy_gm_to_ubuf_align_v2(
        reinterpret_cast<__ubuf__ bfloat16_t*>(dst_p),
        reinterpret_cast<__gm__ bfloat16_t*>(kv_base + (static_cast<uint64_t>(lo0) << 10)),
        0, 2, 1024, 0, 0, false, static_cast<uint8_t>(1), static_cast<uint64_t>(hi0 - lo0) << 10, 1024);
    copy_gm_to_ubuf_align_v2(
        reinterpret_cast<__ubuf__ bfloat16_t*>(dst_p + 2048),
        reinterpret_cast<__gm__ bfloat16_t*>(kv_base + (static_cast<uint64_t>(lo1) << 10)),
        0, 2, 1024, 0, 0, false, static_cast<uint8_t>(1), static_cast<uint64_t>(hi1 - lo1) << 10, 1024);
    copy_gm_to_ubuf_align_v2(
        reinterpret_cast<__ubuf__ bfloat16_t*>(dst_p + 4096),
        reinterpret_cast<__gm__ bfloat16_t*>(kv_base + (static_cast<uint64_t>(lo2) << 10)),
        0, 2, 1024, 0, 0, false, static_cast<uint8_t>(1), static_cast<uint64_t>(hi2 - lo2) << 10, 1024);
    copy_gm_to_ubuf_align_v2(
        reinterpret_cast<__ubuf__ bfloat16_t*>(dst_p + 6144),
        reinterpret_cast<__gm__ bfloat16_t*>(kv_base + (static_cast<uint64_t>(lo3) << 10)),
        0, 2, 1024, 0, 0, false, static_cast<uint8_t>(1), static_cast<uint64_t>(hi3 - lo3) << 10, 1024);
    dst_p += 8192;
  }
}

} // extern "C"
"""


@tla.extern(source=_TND_BURST_SRC, name="tnd_gather_tail_rows_ub_l2_1")
def tnd_gather_tail_rows_ub_l2_1(
    kv_gm: tla.Pointer[tla.UInt8, tla.AddressSpace.gm],
    ub_kv: tla.Pointer[tla.UInt8, tla.AddressSpace.ub],
    ub_idx: tla.Pointer[tla.Int32, tla.AddressSpace.ub],
    kv_zero: tla.Pointer[tla.UInt8, tla.AddressSpace.gm],
    row_base: tla.Int64,
    rows_valid: tla.Int64,
) -> None: ...


@tla.extern(source=_TND_BURST_SRC, name="pa_gather_16rows_mlp2_l2_1")
def _pa_gather_16rows_mlp2_l2_1(
    kv_gm: tla.Pointer[tla.UInt8, tla.AddressSpace.gm],
    bt_row: tla.Pointer[tla.Int32, tla.AddressSpace.gm],
    ub_kv: tla.Pointer[tla.UInt8, tla.AddressSpace.ub],
    ub_idx: tla.Pointer[tla.Int32, tla.AddressSpace.ub],
    pa_block_shift: tla.Int64,
    pa_block_mask: tla.Int64,
    pa_stride_bytes: tla.Int64,
    topk_mode: tla.Int64,
) -> None: ...


@tla.extern(source=_TND_BURST_SRC, name="pa_gather_tail_rows_ub_l2_1")
def _pa_gather_tail_rows_ub_l2_1(
    kv_gm: tla.Pointer[tla.UInt8, tla.AddressSpace.gm],
    bt_row: tla.Pointer[tla.Int32, tla.AddressSpace.gm],
    ub_kv: tla.Pointer[tla.UInt8, tla.AddressSpace.ub],
    ub_idx: tla.Pointer[tla.Int32, tla.AddressSpace.ub],
    kv_zero: tla.Pointer[tla.UInt8, tla.AddressSpace.gm],
    pa_block_shift: tla.Int64,
    pa_block_mask: tla.Int64,
    pa_stride_bytes: tla.Int64,
    topk_mode: tla.Int64,
    rows_valid: tla.Int64,
) -> None: ...


@tla.extern(source=_TND_BURST_SRC, name="tnd_gather_16rows_mlp4_l2_1")
def _tnd_gather_16rows_mlp4_l2_1(
    kv_gm: tla.Pointer[tla.UInt8, tla.AddressSpace.gm],
    ub_kv: tla.Pointer[tla.UInt8, tla.AddressSpace.ub],
    ub_idx: tla.Pointer[tla.Int32, tla.AddressSpace.ub],
    row_base: tla.Int64,
) -> None: ...


@tla.jit
def _pa_l1_gather_chunk_hoisted(
    bufs,
    kv_tensor,
    bt_row_ptr,
    kv_zero,
    ub_kv_ptr,
    my_start,
    ring_base,
    n_rows,
    chunk_id: tla.Constexpr[int],
):
    sh = tla.Int64(_PA_BLOCK_SHIFT)
    msk = tla.Int64(_PA_BLOCK_SIZE - 1)
    sb = tla.Int64(_PA_STRIDE * 2)
    tm = tla.Int64(_PA_TOPK_MODE)
    if n_rows == KV_BLOCK:
        _pa_gather_16rows_mlp2_l2_1(
            tla.recast_ptr(kv_tensor.ptr, dtype=tla.UInt8),
            bt_row_ptr,
            tla.recast_ptr(ub_kv_ptr, dtype=tla.UInt8),
            bufs.ub_idx_ring + (ring_base + chunk_id * KV_UNIT),
            sh,
            msk,
            sb,
            tm,
        )
    else:
        line0 = my_start + chunk_id * KV_UNIT
        rv = n_rows - line0
        if rv < 0:
            rv = tla.as_numeric(0)
        if rv > KV_UNIT:
            rv = tla.as_numeric(KV_UNIT)
        _pa_gather_tail_rows_ub_l2_1(
            tla.recast_ptr(kv_tensor.ptr, dtype=tla.UInt8),
            bt_row_ptr,
            tla.recast_ptr(ub_kv_ptr, dtype=tla.UInt8),
            bufs.ub_idx_ring + (ring_base + chunk_id * KV_UNIT),
            tla.recast_ptr(kv_zero.ptr, dtype=tla.UInt8),
            sh,
            msk,
            sb,
            tm,
            rv.to(tla.Int64),
        )


@tla.jit
def pa_l1_s3_stage25_full(
    sync,
    bufs,
    kv_tensor,
    kv_zero,
    block_table,
    token_row,
    idx_width,
    bt_seq,
    bt_max,
    base_line,
    n_rows,
    idx_slot,
):
    """PA R2 Stage 2.5 (after vec1): wait idx -> MTE2(0,1) -> VSSTB(0,1) -> MTE2(2,3) -> release idx."""
    with tla.vector():
        sub = tla.arch.sub_block_idx()
        my_start = sub * 64
        if n_rows > my_start:
            if idx_slot == 0:
                tla.wait_flag(sync.idx_m2s0)
            else:
                tla.wait_flag(sync.idx_m2s1)
        a0_abs = token_row * idx_width + base_line + my_start
        rem = a0_abs - (a0_abs // 8) * 8
        ring_base = idx_slot * IDX_RING_STRIDE + rem
        bt_row_ptr = block_table.ptr + bt_seq * bt_max

        tla.wait_flag(sync.kv_v_m20)
        _pa_l1_gather_chunk_hoisted(
            bufs,
            kv_tensor,
            bt_row_ptr,
            kv_zero,
            bufs.ub_kv0,
            my_start,
            ring_base,
            n_rows,
            0,
        )
        tla.set_flag(sync.kv_nd_v0)

        tla.wait_flag(sync.kv_v_m21)
        _pa_l1_gather_chunk_hoisted(
            bufs,
            kv_tensor,
            bt_row_ptr,
            kv_zero,
            bufs.ub_kv1,
            my_start,
            ring_base,
            n_rows,
            1,
        )
        tla.set_flag(sync.kv_nd_v1)

        tla.wait_flag(sync.kvz_free)
        tla.wait_flag(sync.kv_nd_v0)
        _ub_zero_tail_rows(bufs.ub_kv0, n_rows, my_start, 0)
        kv_nz_vsstb_p16_u8(
            tla.recast_ptr(bufs.ub_kv0, dtype=tla.UInt8),
            tla.recast_ptr(bufs.ub_kvz, dtype=tla.UInt8),
        )
        tla.set_flag(sync.kv_v_m20)
        tla.set_flag(sync.kvz_v2m3)
        tla.wait_flag(sync.kv_v_m20)
        _pa_l1_gather_chunk_hoisted(
            bufs,
            kv_tensor,
            bt_row_ptr,
            kv_zero,
            bufs.ub_kv0,
            my_start,
            ring_base,
            n_rows,
            2,
        )
        tla.set_flag(sync.kv_nd_v0)

        tla.wait_flag(sync.kvz_free1)
        tla.wait_flag(sync.kv_nd_v1)
        _ub_zero_tail_rows(bufs.ub_kv1, n_rows, my_start, 1)
        kv_nz_vsstb_p16_u8(
            tla.recast_ptr(bufs.ub_kv1, dtype=tla.UInt8),
            tla.recast_ptr(bufs.ub_kvz1, dtype=tla.UInt8),
        )
        tla.set_flag(sync.kv_v_m21)
        tla.set_flag(sync.kvz_v2m3_1)
        tla.wait_flag(sync.kv_v_m21)
        _pa_l1_gather_chunk_hoisted(
            bufs,
            kv_tensor,
            bt_row_ptr,
            kv_zero,
            bufs.ub_kv1,
            my_start,
            ring_base,
            n_rows,
            3,
        )
        tla.set_flag(sync.kv_nd_v1)

        if n_rows > my_start:
            if idx_slot == 0:
                tla.set_flag(sync.idx_s2m0)
            else:
                tla.set_flag(sync.idx_s2m1)


@tla.jit
def _tnd_l1_dma_idx(sync, bufs, idx, token_row, idx_width, base_line, n_rows, idx_slot):
    sub = tla.arch.sub_block_idx()
    my_start = sub * 64
    if n_rows == KV_BLOCK:
        a0_abs = token_row * idx_width + base_line + my_start
        rem = a0_abs - (a0_abs // 8) * 8
        a0 = a0_abs - rem
        need = rem + 64
        if idx_slot == 0:
            tla.wait_flag(sync.idx_s2m0)
        else:
            tla.wait_flag(sync.idx_s2m1)
        tla.copy(
            tla.tile_view(
                tla.make_tensor(
                    bufs.ub_idx_ring + idx_slot * IDX_RING_STRIDE,
                    tla.make_layout(
                        tla.make_shape(IDX_RING_STRIDE),
                        tla.make_stride(1),
                    ),
                ),
                tla.make_shape(need),
                tla.make_coord(0),
            ),
            tla.make_tensor(
                idx.ptr + a0,
                tla.make_layout(
                    tla.make_shape(need),
                    tla.make_stride(1),
                ),
            ),
        )
        if idx_slot == 0:
            tla.set_flag(sync.idx_m2s0)
        else:
            tla.set_flag(sync.idx_m2s1)
    else:
        my_valid = n_rows - my_start
        if my_valid < 0:
            my_valid = tla.as_numeric(0)
        if my_valid > 64:
            my_valid = tla.as_numeric(64)
        if my_valid > 0:
            a0_abs = token_row * idx_width + base_line + my_start
            rem = a0_abs - (a0_abs // 8) * 8
            a0 = a0_abs - rem
            cover = ((my_valid + 7) // 8) * 8
            room = idx_width - (base_line + my_start)
            if cover > room:
                cover = room
            need = rem + cover
            if idx_slot == 0:
                tla.wait_flag(sync.idx_s2m0)
            else:
                tla.wait_flag(sync.idx_s2m1)
            tla.copy(
                tla.tile_view(
                    tla.make_tensor(
                        bufs.ub_idx_ring + idx_slot * IDX_RING_STRIDE,
                        tla.make_layout(
                            tla.make_shape(IDX_RING_STRIDE),
                            tla.make_stride(1),
                        ),
                    ),
                    tla.make_shape(need),
                    tla.make_coord(0),
                ),
                tla.make_tensor(
                    idx.ptr + a0,
                    tla.make_layout(
                        tla.make_shape(need),
                        tla.make_stride(1),
                    ),
                ),
            )
            if idx_slot == 0:
                tla.set_flag(sync.idx_m2s0)
            else:
                tla.set_flag(sync.idx_m2s1)


@tla.jit
def _tnd_l1_gather_chunk_hoisted(
    bufs,
    kv_tensor,
    kv_zero,
    ub_kv_ptr,
    my_start,
    ring_base,
    row_base_i64,
    n_rows,
    chunk_id: tla.Constexpr[int],
):
    if n_rows == KV_BLOCK:
        _tnd_gather_16rows_mlp4_l2_1(
            tla.recast_ptr(kv_tensor.ptr, dtype=tla.UInt8),
            tla.recast_ptr(ub_kv_ptr, dtype=tla.UInt8),
            bufs.ub_idx_ring + (ring_base + chunk_id * KV_UNIT),
            row_base_i64,
        )
    else:
        line0 = my_start + chunk_id * KV_UNIT
        rv = n_rows - line0
        if rv < 0:
            rv = tla.as_numeric(0)
        if rv > KV_UNIT:
            rv = tla.as_numeric(KV_UNIT)
        tnd_gather_tail_rows_ub_l2_1(
            tla.recast_ptr(kv_tensor.ptr, dtype=tla.UInt8),
            tla.recast_ptr(ub_kv_ptr, dtype=tla.UInt8),
            bufs.ub_idx_ring + (ring_base + chunk_id * KV_UNIT),
            tla.recast_ptr(kv_zero.ptr, dtype=tla.UInt8),
            row_base_i64,
            rv.to(tla.Int64),
        )


@tla.jit
def tnd_l1_s3_stage0_async(
    sync, bufs, idx, token_row, idx_width, base_line, n_rows, idx_slot
):
    """R2 SCHED=3 Stage 0: async idx DMA only (zero scalar wait before vec1)."""
    with tla.vector():
        _tnd_l1_dma_idx(
            sync, bufs, idx, token_row, idx_width, base_line, n_rows, idx_slot
        )


@tla.jit
def tnd_l1_s3_stage25_full(
    sync,
    bufs,
    kv_tensor,
    kv_zero,
    token_row,
    idx_width,
    row_base,
    base_line,
    n_rows,
    idx_slot,
):
    """R2 SCHED=3 Stage 2.5 (after vec1): wait idx -> MTE2(0,1) -> VSSTB(0,1) -> MTE2(2,3) -> release idx."""
    with tla.vector():
        sub = tla.arch.sub_block_idx()
        my_start = sub * 64
        if n_rows > my_start:
            if idx_slot == 0:
                tla.wait_flag(sync.idx_m2s0)
            else:
                tla.wait_flag(sync.idx_m2s1)
        a0_abs = token_row * idx_width + base_line + my_start
        rem = a0_abs - (a0_abs // 8) * 8
        ring_base = idx_slot * IDX_RING_STRIDE + rem
        row_base_i64 = row_base.to(tla.Int64)

        tla.wait_flag(sync.kv_v_m20)
        _tnd_l1_gather_chunk_hoisted(
            bufs,
            kv_tensor,
            kv_zero,
            bufs.ub_kv0,
            my_start,
            ring_base,
            row_base_i64,
            n_rows,
            0,
        )
        tla.set_flag(sync.kv_nd_v0)

        tla.wait_flag(sync.kv_v_m21)
        _tnd_l1_gather_chunk_hoisted(
            bufs,
            kv_tensor,
            kv_zero,
            bufs.ub_kv1,
            my_start,
            ring_base,
            row_base_i64,
            n_rows,
            1,
        )
        tla.set_flag(sync.kv_nd_v1)

        tla.wait_flag(sync.kvz_free)
        tla.wait_flag(sync.kv_nd_v0)
        _ub_zero_tail_rows(bufs.ub_kv0, n_rows, my_start, 0)
        kv_nz_vsstb_p16_u8(
            tla.recast_ptr(bufs.ub_kv0, dtype=tla.UInt8),
            tla.recast_ptr(bufs.ub_kvz, dtype=tla.UInt8),
        )
        tla.set_flag(sync.kv_v_m20)
        tla.set_flag(sync.kvz_v2m3)
        tla.wait_flag(sync.kv_v_m20)
        _tnd_l1_gather_chunk_hoisted(
            bufs,
            kv_tensor,
            kv_zero,
            bufs.ub_kv0,
            my_start,
            ring_base,
            row_base_i64,
            n_rows,
            2,
        )
        tla.set_flag(sync.kv_nd_v0)

        tla.wait_flag(sync.kvz_free1)
        tla.wait_flag(sync.kv_nd_v1)
        _ub_zero_tail_rows(bufs.ub_kv1, n_rows, my_start, 1)
        kv_nz_vsstb_p16_u8(
            tla.recast_ptr(bufs.ub_kv1, dtype=tla.UInt8),
            tla.recast_ptr(bufs.ub_kvz1, dtype=tla.UInt8),
        )
        tla.set_flag(sync.kv_v_m21)
        tla.set_flag(sync.kvz_v2m3_1)
        tla.wait_flag(sync.kv_v_m21)
        _tnd_l1_gather_chunk_hoisted(
            bufs,
            kv_tensor,
            kv_zero,
            bufs.ub_kv1,
            my_start,
            ring_base,
            row_base_i64,
            n_rows,
            3,
        )
        tla.set_flag(sync.kv_nd_v1)

        if n_rows > my_start:
            if idx_slot == 0:
                tla.set_flag(sync.idx_s2m0)
            else:
                tla.set_flag(sync.idx_s2m1)


@tla.jit
def tnd_l1_s3_stage4a(sync, bufs, slot, dtype):
    """R2 Split Stage 4a: drain bands 0 & 1 to L1 immediately upon kv_free (BEFORE vec2)."""
    with tla.vector():
        sub = tla.arch.sub_block_idx()
        band0 = sub * 4
        slot_zn = tla.make_tensor(
            bufs.l1_k + slot * (HEAD_DIM * KV_BLOCK),
            tla.make_layout_with_tag((KV_BLOCK, HEAD_DIM), dtype, tla.arch.zN),
        )
        ub_zn_base0 = tla.make_tensor(
            bufs.ub_kvz,
            tla.make_layout(
                tla.make_shape(KV_NZ_PITCH, HEAD_DIM),
                tla.make_stride(HEAD_DIM, 1),
                layoutTag=tla.arch.RowMajor,
            ),
        )
        ub_zn0 = tla.make_tensor_like(bufs.ub_kvz, ub_zn_base0, tla.arch.zNUnAlign)
        ub_zn_band0 = tla.tile_view(
            ub_zn0, tla.make_shape(KV_UNIT, HEAD_DIM), tla.make_coord(0, 0)
        )
        ub_zn_base1 = tla.make_tensor(
            bufs.ub_kvz1,
            tla.make_layout(
                tla.make_shape(KV_NZ_PITCH, HEAD_DIM),
                tla.make_stride(HEAD_DIM, 1),
                layoutTag=tla.arch.RowMajor,
            ),
        )
        ub_zn1 = tla.make_tensor_like(bufs.ub_kvz1, ub_zn_base1, tla.arch.zNUnAlign)
        ub_zn_band1 = tla.tile_view(
            ub_zn1, tla.make_shape(KV_UNIT, HEAD_DIM), tla.make_coord(0, 0)
        )
        tla.wait_flag(sync.kvz_v2m3)
        tla.copy(
            tla.tile_view(
                slot_zn, tla.make_shape(KV_UNIT, HEAD_DIM), tla.make_coord(band0, 0)
            ),
            ub_zn_band0,
        )
        tla.set_flag(sync.kvz_free)
        tla.wait_flag(sync.kvz_v2m3_1)
        tla.copy(
            tla.tile_view(
                slot_zn, tla.make_shape(KV_UNIT, HEAD_DIM), tla.make_coord(band0 + 1, 0)
            ),
            ub_zn_band1,
        )
        tla.set_flag(sync.kvz_free1)


@tla.jit
def tnd_l1_s3_stage4b(sync, bufs, slot, dtype, n_rows=KV_BLOCK, announce: bool = True):
    """R2 Split Stage 4b: VSSTB+drain bands 2 & 3 + announce kv_stage_ready (AFTER vec2)."""
    with tla.vector():
        sub = tla.arch.sub_block_idx()
        my_start = sub * 64
        band0 = sub * 4
        slot_zn = tla.make_tensor(
            bufs.l1_k + slot * (HEAD_DIM * KV_BLOCK),
            tla.make_layout_with_tag((KV_BLOCK, HEAD_DIM), dtype, tla.arch.zN),
        )
        ub_zn_base0 = tla.make_tensor(
            bufs.ub_kvz,
            tla.make_layout(
                tla.make_shape(KV_NZ_PITCH, HEAD_DIM),
                tla.make_stride(HEAD_DIM, 1),
                layoutTag=tla.arch.RowMajor,
            ),
        )
        ub_zn0 = tla.make_tensor_like(bufs.ub_kvz, ub_zn_base0, tla.arch.zNUnAlign)
        ub_zn_band0 = tla.tile_view(
            ub_zn0, tla.make_shape(KV_UNIT, HEAD_DIM), tla.make_coord(0, 0)
        )
        ub_zn_base1 = tla.make_tensor(
            bufs.ub_kvz1,
            tla.make_layout(
                tla.make_shape(KV_NZ_PITCH, HEAD_DIM),
                tla.make_stride(HEAD_DIM, 1),
                layoutTag=tla.arch.RowMajor,
            ),
        )
        ub_zn1 = tla.make_tensor_like(bufs.ub_kvz1, ub_zn_base1, tla.arch.zNUnAlign)
        ub_zn_band1 = tla.tile_view(
            ub_zn1, tla.make_shape(KV_UNIT, HEAD_DIM), tla.make_coord(0, 0)
        )
        tla.wait_flag(sync.kvz_free)
        tla.wait_flag(sync.kv_nd_v0)
        _ub_zero_tail_rows(bufs.ub_kv0, n_rows, my_start, 2)
        kv_nz_vsstb_p16_u8(
            tla.recast_ptr(bufs.ub_kv0, dtype=tla.UInt8),
            tla.recast_ptr(bufs.ub_kvz, dtype=tla.UInt8),
        )
        tla.set_flag(sync.kv_v_m20)
        tla.set_flag(sync.kvz_v2m3)
        tla.wait_flag(sync.kvz_v2m3)
        tla.copy(
            tla.tile_view(
                slot_zn, tla.make_shape(KV_UNIT, HEAD_DIM), tla.make_coord(band0 + 2, 0)
            ),
            ub_zn_band0,
        )
        tla.set_flag(sync.kvz_free)

        tla.wait_flag(sync.kvz_free1)
        tla.wait_flag(sync.kv_nd_v1)
        _ub_zero_tail_rows(bufs.ub_kv1, n_rows, my_start, 3)
        kv_nz_vsstb_p16_u8(
            tla.recast_ptr(bufs.ub_kv1, dtype=tla.UInt8),
            tla.recast_ptr(bufs.ub_kvz1, dtype=tla.UInt8),
        )
        tla.set_flag(sync.kv_v_m21)
        tla.set_flag(sync.kvz_v2m3_1)
        tla.wait_flag(sync.kvz_v2m3_1)
        tla.copy(
            tla.tile_view(
                slot_zn, tla.make_shape(KV_UNIT, HEAD_DIM), tla.make_coord(band0 + 3, 0)
            ),
            ub_zn_band1,
        )
        tla.set_flag(sync.kvz_free1)
        if announce:
            tla.cross_core_set_flag(sync.kv_stage_ready, tla.pipes.MTE3, aiv_id=0)
            tla.cross_core_set_flag(sync.kv_stage_ready, tla.pipes.MTE3, aiv_id=1)


@tla.jit
def gather_kv_rows_ring(
    sync,
    bufs,
    ws,
    kv_tensor,
    idx,
    token_row,
    idx_width,
    row_base,
    base_line,
    n_rows,
    slot,
    slot_stride,
    idx_slot,
    ws_base=0,
    announce: bool = True,
):
    """T1-1（r10）：非 PA 稀疏地址「去表」的 gather（索引行 DMA -> UB 环 -> 标量读）。

    与 `gather_kv_rows_vec` 的交付配置**逐条同构**（`_SPARSE_BLOCK_SIZE=1`：
    每 8 行 4 条两行合并 pair 拷贝、CeilDiv(n,2) 子块切分、16 行 ping-pong、
    既有全部同步点、输出路径、ws 暂存形态），唯一变化是**行地址从哪里来**：

      旧（= 表路径）：per-task 生成的 int64 物理地址表落 GM，gather 每 8 行块
                     从 GM 表标量读 8 个 8 B 地址（V:2736 形态）
      新（本函数）  ：索引行一次 MTE2 DMA 进 UB 2 槽环（IDX_RING_STRIDE i32/槽），
                     逐行标量读 UB + 基址加法
                       row_off = (idx_clamped + row_base) * HEAD_DIM

    与表路径逐位等价：生成侧表值 = (max(idx,0) + kv_prefix) * HEAD_DIM，
    kv_prefix = cu_cmp[seq]（csa）/ cu_kv[seq]（os），TND 与 BSND 同值
    （BSND 的 bS2Base = kv_prefix*kv_dim）；本函数 `row_base` 收同一 cu 值。

    DMA 对齐：AIV 半区起点 my_start 与 tok*idx_width 都不保证 8 元素（32 B）对齐，
    源基址取 align_down(token_row*idx_width + base_line + my_start, 8)，槽内前
    rem 个元素为空位；槽宽 IDX_RING_STRIDE 覆盖 rem(≤7)+cover(≤64)=71。
    读下标上界 rem + min(64, my_rows) - 1 ≤ 70 < 71 ⇒ 不越出槽。

    同步（每个 AIV 自己的环；竞品 N:737-847 无同步，TLA 不做自动管道同步）：
      tile 头：wait free[idx_slot]（S->M2：上一轮同槽标量读已完成）
                -> MTE2 拷入索引行 -> set ready[idx_slot]（M2->S）
      tile 头紧随：wait ready[idx_slot]（读到本槽数据之前必须落地）
      tile 末    ：set free[idx_slot]（全部标量读已发出后放行下一轮同槽写）
      init 期预置 2 个 free 信用（init_sync(pre_idx_ring=True)）。
    读侧走 SCALAR pipe，不是 VECTOR：MTE2 写完的 UB 只能由 SCALAR flag 放行
    （见项目记录 tla-scalar-ub-read-needs-scalar-flag）。

    索引环与 ws 暂存**结构无关**（bufs.ub_idx_ring 是独立分配，不叠在 ws 上），
    T1-0a 摘除 ws 时本函数只需整段删掉「UB -> ws」段落与 kv_stage 通知。
    """
    require_configured("gather_kv_rows_vec")
    if _TRACE_STAGE:
        require_share("gather_kv_rows_vec")
    with tla.vector():
        sub = tla.arch.sub_block_idx()
        first = (n_rows + 1) // 2
        if _TRACE_QUARTER:
            # A 步二（OT CalProcessSize）：共享暂存下本 AIC 对的 4 个 AIV 各写 1/4
            par = tla.arch.block_idx() % 2
            h0 = (first + 1) // 2
            h1 = first - h0
            second = n_rows - first
            s0 = (second + 1) // 2
            s1 = second - s0
            my_start = 0
            my_rows = h0
            if sub == 1:
                my_start = h0
                my_rows = h1
            if par == 1:
                my_start = first
                my_rows = s0
                if sub == 1:
                    my_start = first + s0
                    my_rows = s1
        else:
            my_start = 0
            my_rows = first
            if sub == 1:
                my_start = first
                my_rows = n_rows - first
        n_chunks = (my_rows + KV_UNIT - 1) // KV_UNIT
        # ---- 索引行 DMA（每 tile 一次；2 槽环按 x%2 轮转）----
        # 放在 kv_mte3_mte2 信用等待之前：索引行搬运与上一块 UB->ws 的 MTE3
        # 写回重叠，不占 AIC/AIV 的 gather 关键段。
        a0_abs = token_row * idx_width + base_line + my_start
        rem = a0_abs - (a0_abs // 8) * 8  # 行内 8 元素（32 B）相位
        a0 = a0_abs - rem  # 32 B 对齐的源基址
        cover = n_chunks * KV_UNIT
        if cover > my_rows:
            cover = my_rows
        # 8 元素（32 B）对齐向上取整：DMA 覆盖行 [my_start-rem, my_start+cover)。
        # 表路径的 8 地址块会读到「本子块 my_rows 之后」的行（尾块 15/16 行、
        # 以及 pair 配对的第 16 行），该值决定尾 pair 走合并还是单行回落；
        # 环必须给出同一行的真实索引才能逐位同构，故多取 ≤7 个元素。
        # 上界只到本 token 的索引行尾（idx_width），绝不越出该行。
        cover = ((cover + 7) // 8) * 8
        room = idx_width - (base_line + my_start)
        if cover > room:
            cover = room
        need = rem + cover
        if n_chunks > 0:
            if idx_slot == 0:
                tla.wait_flag(sync.idx_s2m0)
            else:
                tla.wait_flag(sync.idx_s2m1)
            tla.copy(
                tla.tile_view(
                    tla.make_tensor(
                        bufs.ub_idx_ring + idx_slot * IDX_RING_STRIDE,
                        tla.make_layout(
                            tla.make_shape(IDX_RING_STRIDE),
                            tla.make_stride(1),
                        ),
                    ),
                    tla.make_shape(need),
                    tla.make_coord(0),
                ),
                tla.make_tensor(
                    idx.ptr + a0,
                    tla.make_layout(
                        tla.make_shape(need),
                        tla.make_stride(1),
                    ),
                ),
            )
            if idx_slot == 0:
                tla.set_flag(sync.idx_m2s0)
                tla.wait_flag(sync.idx_m2s0)
            else:
                tla.set_flag(sync.idx_m2s1)
                tla.wait_flag(sync.idx_m2s1)
        # 整环 1-D 视图 + 槽内基址（读下标 = ring_o + 行号，恒 >= 0）
        ring_t = tla.make_tensor(
            bufs.ub_idx_ring,
            tla.make_layout(
                tla.make_shape(IDX_RING_SLOTS * IDX_RING_STRIDE),
                tla.make_stride(1),
            ),
        )
        ring_o = idx_slot * IDX_RING_STRIDE + rem - my_start
        # 读行号上界 = min(槽内已落地最后一行, 表路径的 n_rows-1)：
        # 表路径把越过 tile 的行号 clamp 到 n_rows-1（读的是同一 tile 内的真实
        # 表项），环必须给同一行的真实索引；而 n_rows 之后的行表路径从不读，
        # 环也不允许读到槽外。
        last_row = my_start + cover - 1
        if last_row > n_rows - 1:
            last_row = n_rows - 1
        for c in tla.range(n_chunks):
            line0 = my_start + c * KV_UNIT
            rows_out = my_rows - c * KV_UNIT
            if rows_out > KV_UNIT:
                rows_out = tla.as_numeric(KV_UNIT)
            if rows_out < 0:
                rows_out = tla.as_numeric(0)
            if c % 2 == 0:
                ub_kv = tla.make_tensor(
                    bufs.ub_kv0,
                    tla.make_layout(
                        tla.make_shape(KV_UNIT, HEAD_DIM),
                        tla.make_stride(HEAD_DIM, 1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                # [T1-0a 摘除区：起点] UB->ws 信用（上一块 MTE3 写回完成）
                tla.wait_flag(sync.kv_mte3_mte2_0)
                # T1-0a 摘除区：[终点]（本块 gather 的 UB 缓冲本块内独占）
                for blk in static_range(KV_UNIT // 8):
                    blk_base = 8 * blk
                    # 整块 8 条地址先背靠背读完（与表路径同形，
                    # 只是来源换成 UB 环标量读 + 基址加法），尾块按行 clamp
                    addr_t = ()
                    for ai in static_range(8):
                        aline = line0 + blk_base + ai
                        # 只允许读到本槽已 DMA 落地、且表路径会读的行（last_row）；
                        # 越界的行回落到该行，避免拿环里的陈旧索引配出假 pair。
                        if aline > last_row:
                            aline = last_row
                        if aline < 0:
                            aline = 0
                        v = ring_t[ring_o + aline]
                        if v < 0:
                            v = 0
                        addr_t = addr_t + (
                            (row_base + v).to(tla.Int64) * tla.Int64(HEAD_DIM),
                        )
                    pair_lo = tla.Int64(HEAD_DIM)
                    pair_hi = tla.Int64(1073741824)
                    pair_zero = tla.Int64(0)
                    for p in static_range(4):
                        row_a = blk_base + 2 * p
                        off_a = addr_t[2 * p]
                        off_b = addr_t[2 * p + 1]
                        pair_delta = off_b - off_a
                        valid_a = row_a <= n_rows - 1 - line0
                        valid_b = valid_a and row_a + 1 <= n_rows - 1 - line0
                        if valid_b and pair_delta >= pair_lo and pair_delta < pair_hi:
                            pair_gm_to_ub(
                                tla.recast_ptr(kv_tensor.ptr + off_a, dtype=tla.UInt8),
                                tla.recast_ptr(bufs.ub_kv0, dtype=tla.UInt8),
                                HEAD_DIM * _PAIR_ELEM_BYTES,
                                pair_delta * tla.Int64(_PAIR_ELEM_BYTES)
                                - tla.Int64(HEAD_DIM * _PAIR_ELEM_BYTES),
                                tla.Int64(row_a * HEAD_DIM * _PAIR_ELEM_BYTES),
                                tla.Int64(2),
                            )
                        elif (
                            valid_b
                            and pair_zero - pair_delta >= pair_lo
                            and pair_zero - pair_delta < pair_hi
                        ):
                            # 降序：取 min 起始（= off_b），UB 行序互换
                            pair_gm_to_ub(
                                tla.recast_ptr(kv_tensor.ptr + off_b, dtype=tla.UInt8),
                                tla.recast_ptr(bufs.ub_kv0, dtype=tla.UInt8),
                                HEAD_DIM * _PAIR_ELEM_BYTES,
                                (pair_zero - pair_delta) * tla.Int64(_PAIR_ELEM_BYTES)
                                - tla.Int64(HEAD_DIM * _PAIR_ELEM_BYTES),
                                tla.Int64(row_a * HEAD_DIM * _PAIR_ELEM_BYTES),
                                tla.Int64(2),
                            )
                        elif valid_b:
                            # stride 越界/过小：回落单行 x2（用已读好的地址）
                            for ri in static_range(2):
                                row_off = addr_t[2 * p + ri]
                                gm_row = tla.make_tensor(
                                    kv_tensor.ptr + row_off,
                                    tla.make_layout(
                                        tla.make_shape(1, HEAD_DIM),
                                        tla.make_stride(HEAD_DIM, 1),
                                        layoutTag=tla.arch.RowMajor,
                                    ),
                                )
                                tla.copy(
                                    tla.tile_view(
                                        ub_kv,
                                        tla.make_shape(1, HEAD_DIM),
                                        tla.make_coord(row_a + ri, 0),
                                    ),
                                    gm_row,
                                )
                        elif valid_a:
                            # 尾块只剩首行：blockCount=1、src = off0
                            pair_gm_to_ub(
                                tla.recast_ptr(kv_tensor.ptr + off_a, dtype=tla.UInt8),
                                tla.recast_ptr(bufs.ub_kv0, dtype=tla.UInt8),
                                HEAD_DIM * _PAIR_ELEM_BYTES,
                                pair_zero,
                                tla.Int64(row_a * HEAD_DIM * _PAIR_ELEM_BYTES),
                                tla.Int64(1),
                            )
                # ==== [T1-0a 摘除区：起点] UB -> ws 暂存（GM 中转）====
                tla.set_flag(sync.kv_mte2_mte3_0)
                tla.wait_flag(sync.kv_mte2_mte3_0)
                gm_chunk = tla.make_tensor(
                    ws.ptr + ws_base + slot * slot_stride + line0 * HEAD_DIM,
                    tla.make_layout(
                        tla.make_shape(rows_out, HEAD_DIM),
                        tla.make_stride(HEAD_DIM, 1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                tla.copy(
                    gm_chunk,
                    tla.tile_view(
                        ub_kv, tla.make_shape(rows_out, HEAD_DIM), tla.make_coord(0, 0)
                    ),
                )
                tla.set_flag(sync.kv_mte3_mte2_0)
                # ==== [T1-0a 摘除区：终点] ====
            else:
                ub_kv = tla.make_tensor(
                    bufs.ub_kv1,
                    tla.make_layout(
                        tla.make_shape(KV_UNIT, HEAD_DIM),
                        tla.make_stride(HEAD_DIM, 1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                # [T1-0a 摘除区：起点] UB->ws 信用
                tla.wait_flag(sync.kv_mte3_mte2_1)
                # T1-0a 摘除区：[终点]
                for blk in static_range(KV_UNIT // 8):
                    blk_base = 8 * blk
                    addr_t = ()
                    for ai in static_range(8):
                        aline = line0 + blk_base + ai
                        # 只允许读到本槽已 DMA 落地、且表路径会读的行（last_row）；
                        # 越界的行回落到该行，避免拿环里的陈旧索引配出假 pair。
                        if aline > last_row:
                            aline = last_row
                        if aline < 0:
                            aline = 0
                        v = ring_t[ring_o + aline]
                        if v < 0:
                            v = 0
                        addr_t = addr_t + (
                            (row_base + v).to(tla.Int64) * tla.Int64(HEAD_DIM),
                        )
                    pair_lo = tla.Int64(HEAD_DIM)
                    pair_hi = tla.Int64(1073741824)
                    pair_zero = tla.Int64(0)
                    for p in static_range(4):
                        row_a = blk_base + 2 * p
                        off_a = addr_t[2 * p]
                        off_b = addr_t[2 * p + 1]
                        pair_delta = off_b - off_a
                        valid_a = row_a <= n_rows - 1 - line0
                        valid_b = valid_a and row_a + 1 <= n_rows - 1 - line0
                        if valid_b and pair_delta >= pair_lo and pair_delta < pair_hi:
                            pair_gm_to_ub(
                                tla.recast_ptr(kv_tensor.ptr + off_a, dtype=tla.UInt8),
                                tla.recast_ptr(bufs.ub_kv1, dtype=tla.UInt8),
                                HEAD_DIM * _PAIR_ELEM_BYTES,
                                pair_delta * tla.Int64(_PAIR_ELEM_BYTES)
                                - tla.Int64(HEAD_DIM * _PAIR_ELEM_BYTES),
                                tla.Int64(row_a * HEAD_DIM * _PAIR_ELEM_BYTES),
                                tla.Int64(2),
                            )
                        elif (
                            valid_b
                            and pair_zero - pair_delta >= pair_lo
                            and pair_zero - pair_delta < pair_hi
                        ):
                            pair_gm_to_ub(
                                tla.recast_ptr(kv_tensor.ptr + off_b, dtype=tla.UInt8),
                                tla.recast_ptr(bufs.ub_kv1, dtype=tla.UInt8),
                                HEAD_DIM * _PAIR_ELEM_BYTES,
                                (pair_zero - pair_delta) * tla.Int64(_PAIR_ELEM_BYTES)
                                - tla.Int64(HEAD_DIM * _PAIR_ELEM_BYTES),
                                tla.Int64(row_a * HEAD_DIM * _PAIR_ELEM_BYTES),
                                tla.Int64(2),
                            )
                        elif valid_b:
                            for ri in static_range(2):
                                row_off = addr_t[2 * p + ri]
                                gm_row = tla.make_tensor(
                                    kv_tensor.ptr + row_off,
                                    tla.make_layout(
                                        tla.make_shape(1, HEAD_DIM),
                                        tla.make_stride(HEAD_DIM, 1),
                                        layoutTag=tla.arch.RowMajor,
                                    ),
                                )
                                tla.copy(
                                    tla.tile_view(
                                        ub_kv,
                                        tla.make_shape(1, HEAD_DIM),
                                        tla.make_coord(row_a + ri, 0),
                                    ),
                                    gm_row,
                                )
                        elif valid_a:
                            pair_gm_to_ub(
                                tla.recast_ptr(kv_tensor.ptr + off_a, dtype=tla.UInt8),
                                tla.recast_ptr(bufs.ub_kv1, dtype=tla.UInt8),
                                HEAD_DIM * _PAIR_ELEM_BYTES,
                                pair_zero,
                                tla.Int64(row_a * HEAD_DIM * _PAIR_ELEM_BYTES),
                                tla.Int64(1),
                            )
                # ==== [T1-0a 摘除区：起点] UB -> ws 暂存（GM 中转）====
                tla.set_flag(sync.kv_mte2_mte3_1)
                tla.wait_flag(sync.kv_mte2_mte3_1)
                gm_chunk = tla.make_tensor(
                    ws.ptr + ws_base + slot * slot_stride + line0 * HEAD_DIM,
                    tla.make_layout(
                        tla.make_shape(rows_out, HEAD_DIM),
                        tla.make_stride(HEAD_DIM, 1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                tla.copy(
                    gm_chunk,
                    tla.tile_view(
                        ub_kv, tla.make_shape(rows_out, HEAD_DIM), tla.make_coord(0, 0)
                    ),
                )
                tla.set_flag(sync.kv_mte3_mte2_1)
                # ==== [T1-0a 摘除区：终点] ====
        # 本 tile 的标量读全部发出 -> 归还该槽的 free 信用（WAR）
        if n_chunks > 0:
            if idx_slot == 0:
                tla.set_flag(sync.idx_s2m0)
            else:
                tla.set_flag(sync.idx_s2m1)
        if announce:
            tla.cross_core_set_flag(sync.kv_stage_ready, tla.pipes.MTE3, aiv_id=0)
            tla.cross_core_set_flag(sync.kv_stage_ready, tla.pipes.MTE3, aiv_id=1)


@tla.jit
def wait_kv_free(sync):
    """R42：AIV 等本 cmp tile 的 L1 槽可写（cross AIC MTE1 -> AIV MTE3）。

    只在 cmp tile 且 x>=3 时消费（见 kernel 站点 4）；AIC 只在"后继 tile 是
    cmp"时归还（smla_csa_block_cube.release_k 的 free_tok）。两边条数恒等：
    每个 cmp tile 一收一发。x<3 的槽在本核从未被占用，无需信用。
    """
    with tla.vector():
        tla.cross_core_wait_flag(sync.kv_free, tla.pipes.MTE3, aiv_id=0)
        tla.cross_core_wait_flag(sync.kv_free, tla.pipes.MTE3, aiv_id=1)


@tla.jit
def gather_kv_rows_l1(
    sync,
    bufs,
    kv_tensor,
    idx,
    token_row,
    idx_width,
    row_base,
    base_line,
    n_rows,
    slot,
    idx_slot,
    dtype,
    dense: bool = False,
):
    """R42：cmp 稀疏 gather 由 AIV 直写本核 L1 槽（不再经 GM 暂存槽）。

    `dense`（新增，ori 稠密臂）：True 时本 tile 是 ori 连续窗口，没有索引行，
    行地址 = (row_base + base_line + line)*HEAD_DIM（与 cube.load_k 同式），
    其余数据面/同步与 cmp 臂完全一致。`dense` 是 Python bool，追踪期折叠，
    不产生运行期判据；False 时指令流与 cmp-only 形态一致。

    数据面（每个 16 行 band）：
      cmp：GM 8 条 pair 拷贝（2 行/条，与 gather_kv_rows_ring 同形）-> UB(ND, ub_kv0)
      ori：GM 一条 16x512 块拷贝（行连续，尾 band 按 n_rows 截断）-> UB(ND)
      两者 -> 一条 UB(RowMajor) -> L1(zN) 拷贝（MTE3 硬件 ND->NZ）
    AIC 侧不再有 MTE2 的 KV 搬运：它在 bmm1 里只等 `kv_stage_ready`（AIV->AIC）。
    实测（022）：软件 VF 重排版 aiv_vec=275us、硬件转换版 aiv_mte3=352us，
    两者都把 Dur 推到 494/638us（步二 218.6），故 KV_L1_DIRECT 默认 False。

    与 `gather_kv_rows_ring` 的差异：
      - 子块切分按 16 行对齐（zN band 以 16 行为发散单位；否则 AIV0 尾 band 与
        AIV1 首 band 是同一 band，两核踩写）。行集合不变，只改工作量分配。
      - 目的端从 ws(GM) 换成 L1 槽；`wait kv_free` 由调用方在每 tile 统一发
        （ori tile 也要消费信用），本函数只做 band 级 ND/zN WAR。
      - 索引仍来自 UB 2 槽环（T1-1），地址 = (row_base + idx) * HEAD_DIM。

    band 级同步（本 AIV 的 intra 环，见 common.declare_sync 的 need_kv_ring）：
      wait kv_nd_free（MTE3->MTE2：上一 band 的 L1 copy 已读完 ND）
      -> MTE2 填 ND -> set kv_a_loaded（MTE2->MTE3）
      -> wait kv_a_loaded -> MTE3 copy UB(ND)->L1(zN) -> set kv_nd_free
    """
    require_configured("gather_kv_rows_vec")
    if _TRACE_STAGE:
        require_share("gather_kv_rows_vec")
    with tla.vector():
        sub = tla.arch.sub_block_idx()
        # 16 行对齐的子块起点：AIV0 拿 ceil(CeilDiv(n,2)/16)*16 行，AIV1 拿其余。
        first = ((n_rows + 1) // 2 + KV_UNIT - 1) // KV_UNIT * KV_UNIT
        if first > n_rows:
            first = n_rows
        my_start = 0
        my_rows = first
        if sub == 1:
            my_start = first
            my_rows = n_rows - first
        n_chunks = (my_rows + KV_UNIT - 1) // KV_UNIT
        # ---- 索引行 DMA（每 tile 一次；2 槽环按 x%2 轮转）----
        # 稠密窗口没有索引行：本段（含 idx_m2s/idx_s2m）在 dense=True 的
        # 特化里一条不发（Python 折叠，不占指令）。前端语法层仍把 `dense` 当
        # 动态 if（要求 if 后使用的变量先初始化），故 ring_t/ring_o/last_row
        # 在此预置（dense=True 特化下不使用，语义不变）。
        ring_t = tla.make_tensor(
            bufs.ub_idx_ring,
            tla.make_layout(
                tla.make_shape(IDX_RING_SLOTS * IDX_RING_STRIDE),
                tla.make_stride(1),
            ),
        )
        ring_o = idx_slot * IDX_RING_STRIDE - my_start
        last_row = n_rows - 1
        if not dense:
            a0_abs = token_row * idx_width + base_line + my_start
            rem = a0_abs - (a0_abs // 8) * 8  # 行内 8 元素（32 B）相位
            a0 = a0_abs - rem  # 32 B 对齐的源基址
            cover = n_chunks * KV_UNIT
            if cover > my_rows:
                cover = my_rows
            cover = ((cover + 7) // 8) * 8
            room = idx_width - (base_line + my_start)
            if cover > room:
                cover = room
            need = rem + cover
            if n_chunks > 0:
                if idx_slot == 0:
                    tla.wait_flag(sync.idx_s2m0)
                else:
                    tla.wait_flag(sync.idx_s2m1)
                tla.copy(
                    tla.tile_view(
                        tla.make_tensor(
                            bufs.ub_idx_ring + idx_slot * IDX_RING_STRIDE,
                            tla.make_layout(
                                tla.make_shape(IDX_RING_STRIDE),
                                tla.make_stride(1),
                            ),
                        ),
                        tla.make_shape(need),
                        tla.make_coord(0),
                    ),
                    tla.make_tensor(
                        idx.ptr + a0,
                        tla.make_layout(
                            tla.make_shape(need),
                            tla.make_stride(1),
                        ),
                    ),
                )
                if idx_slot == 0:
                    tla.set_flag(sync.idx_m2s0)
                    tla.wait_flag(sync.idx_m2s0)
                else:
                    tla.set_flag(sync.idx_m2s1)
                    tla.wait_flag(sync.idx_m2s1)
            # 整环 1-D 视图 + 槽内基址（读下标 = ring_o + 行号，恒 >= 0）
            # ring_t 已在 if 之前建好（前端语法层要求），此处不再重复创建
            ring_o = idx_slot * IDX_RING_STRIDE + rem - my_start
            last_row = my_start + cover - 1
            if last_row > n_rows - 1:
                last_row = n_rows - 1
        # L1 槽 zN 视图（与 cube.l1_v_tensor 同布局；元素偏移 = slot*HEAD_DIM*KV_BLOCK）
        slot_zn = tla.make_tensor(
            bufs.l1_k + slot * (HEAD_DIM * KV_BLOCK),
            tla.make_layout_with_tag((KV_BLOCK, HEAD_DIM), dtype, tla.arch.zN),
        )
        # ND 中转（16 行 RowMajor，固定 ub_kv0）。ND->NZ 由 MTE3 的
        # UB(RowMajor)->L1(zN) 硬件转换完成，不再需要 ub_kvz / 软件 VF。
        ub_kv = tla.make_tensor(
            bufs.ub_kv0,
            tla.make_layout(
                tla.make_shape(KV_UNIT, HEAD_DIM),
                tla.make_stride(HEAD_DIM, 1),
                layoutTag=tla.arch.RowMajor,
            ),
        )
        for c in tla.range(n_chunks):
            line0 = my_start + c * KV_UNIT
            tla.wait_flag(sync.kv_nd_free)  # V->MTE2：ND 可写（c=0 用预置）
            if dense:
                # 步 A：ori 稠密窗口的行在 GM 连续，一条 16x512 的块拷贝即可
                # （尾 band 用运行期 cnt 截断到 n_rows，绝不越界读 GM）。
                cnt = KV_UNIT
                if line0 + KV_UNIT > n_rows:
                    cnt = n_rows - line0
                gm_band = tla.make_tensor(
                    kv_tensor.ptr + (row_base + base_line + line0) * HEAD_DIM,
                    tla.make_layout(
                        tla.make_shape(cnt, HEAD_DIM),
                        tla.make_stride(HEAD_DIM, 1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                tla.copy(
                    tla.tile_view(
                        ub_kv,
                        tla.make_shape(cnt, HEAD_DIM),
                        tla.make_coord(0, 0),
                    ),
                    gm_band,
                )
            else:
                for blk in static_range(KV_UNIT // 8):
                    blk_base = 8 * blk
                    addr_t = ()
                    for ai in static_range(8):
                        aline = line0 + blk_base + ai
                        if aline > last_row:
                            aline = last_row
                        if aline < 0:
                            aline = 0
                        # 前端语法层要求 if 之后使用的变量先初始化（dense 为 Python
                        # bool，追踪期折叠；初值不使用）。
                        v = tla.as_numeric(0)
                        if dense:
                            v = base_line + aline
                        else:
                            v = ring_t[ring_o + aline]
                            if v < 0:
                                v = 0
                        addr_t = addr_t + (
                            (row_base + v).to(tla.Int64) * tla.Int64(HEAD_DIM),
                        )
                    pair_lo = tla.Int64(HEAD_DIM)
                    pair_hi = tla.Int64(1073741824)
                    pair_zero = tla.Int64(0)
                    for p in static_range(4):
                        row_a = blk_base + 2 * p
                        off_a = addr_t[2 * p]
                        off_b = addr_t[2 * p + 1]
                        pair_delta = off_b - off_a
                        valid_a = row_a <= n_rows - 1 - line0
                        valid_b = valid_a and row_a + 1 <= n_rows - 1 - line0
                        if valid_b and pair_delta >= pair_lo and pair_delta < pair_hi:
                            pair_gm_to_ub(
                                tla.recast_ptr(kv_tensor.ptr + off_a, dtype=tla.UInt8),
                                tla.recast_ptr(bufs.ub_kv0, dtype=tla.UInt8),
                                HEAD_DIM * _PAIR_ELEM_BYTES,
                                pair_delta * tla.Int64(_PAIR_ELEM_BYTES)
                                - tla.Int64(HEAD_DIM * _PAIR_ELEM_BYTES),
                                tla.Int64(row_a * HEAD_DIM * _PAIR_ELEM_BYTES),
                                tla.Int64(2),
                            )
                        elif (
                            valid_b
                            and pair_zero - pair_delta >= pair_lo
                            and pair_zero - pair_delta < pair_hi
                        ):
                            pair_gm_to_ub(
                                tla.recast_ptr(kv_tensor.ptr + off_b, dtype=tla.UInt8),
                                tla.recast_ptr(bufs.ub_kv0, dtype=tla.UInt8),
                                HEAD_DIM * _PAIR_ELEM_BYTES,
                                (pair_zero - pair_delta) * tla.Int64(_PAIR_ELEM_BYTES)
                                - tla.Int64(HEAD_DIM * _PAIR_ELEM_BYTES),
                                tla.Int64(row_a * HEAD_DIM * _PAIR_ELEM_BYTES),
                                tla.Int64(2),
                            )
                        elif valid_b:
                            for ri in static_range(2):
                                row_off = addr_t[2 * p + ri]
                                gm_row = tla.make_tensor(
                                    kv_tensor.ptr + row_off,
                                    tla.make_layout(
                                        tla.make_shape(1, HEAD_DIM),
                                        tla.make_stride(HEAD_DIM, 1),
                                        layoutTag=tla.arch.RowMajor,
                                    ),
                                )
                                tla.copy(
                                    tla.tile_view(
                                        ub_kv,
                                        tla.make_shape(1, HEAD_DIM),
                                        tla.make_coord(row_a + ri, 0),
                                    ),
                                    gm_row,
                                )
                        elif valid_a:
                            pair_gm_to_ub(
                                tla.recast_ptr(kv_tensor.ptr + off_a, dtype=tla.UInt8),
                                tla.recast_ptr(bufs.ub_kv0, dtype=tla.UInt8),
                                HEAD_DIM * _PAIR_ELEM_BYTES,
                                pair_zero,
                                tla.Int64(row_a * HEAD_DIM * _PAIR_ELEM_BYTES),
                                tla.Int64(1),
                            )
            # ==== UB(ND) --MTE3--> L1 zN band（硬件 ND->NZ）====
            # ("ub","RowMajor","l1","zN") 是已注册的 vector copy 路由：MTE3 的
            # DataCopy 直接做 ND->NZ。R42 用软件 VF 逐行 x4 quarter 重排，022
            # 实测该 VF 单管 275us（Ori copy 占绝对多数时把 Dur 从 218 推到
            # 494）；改为硬件转换后 VF 段整段删除。
            tla.set_flag(sync.kv_a_loaded)  # MTE2 -> MTE3：ND band 已填好
            tla.wait_flag(sync.kv_a_loaded)
            tla.copy(
                tla.tile_view(
                    slot_zn,
                    tla.make_shape(KV_UNIT, HEAD_DIM),
                    tla.make_coord(line0 // KV_UNIT, 0),
                ),
                ub_kv,
            )
            # MTE3 -> MTE2：ND 已被 MTE3 读完，MTE2 可预取下一 band。
            # set 与 copy 同管（MTE3）、程序序在 copy 之后，天然是读完回执。
            tla.set_flag(sync.kv_nd_free)
        # 本 tile 的标量读全部发出 -> 归还索引槽 free 信用（WAR；dense 无索引环）
        if not dense:
            if n_chunks > 0:
                if idx_slot == 0:
                    tla.set_flag(sync.idx_s2m0)
                else:
                    tla.set_flag(sync.idx_s2m1)
        # 本 AIV 的 KV 半区已写进 L1 槽：通知 AIC（bmm1 的 MTE1 等本 flag）。
        # set 挂 MTE3：与上面的 L1 band copy 同管，程序序保证 copy 先发。
        tla.cross_core_set_flag(sync.kv_stage_ready, tla.pipes.MTE3, aiv_id=0)
        tla.cross_core_set_flag(sync.kv_stage_ready, tla.pipes.MTE3, aiv_id=1)
