# ---------------------------------------------------------------------------
# smla_csa_block_cube.py — SmlaCsaBlockCube 的 DSL 对应
#
# 对照 ops-transformer/attention/sparse_flash_mla/op_kernel/arch35/
#   sparse_flash_mla_csa_block_cube_arch35.h
#     IterateBmm1Impl : 384-510  (QK, 2x256 两次 MatmulKStatic, 每次再切 128)
#     IterateBmm2Impl :       (PV, K=s2RealSize, N=512 拆 4 段)
# 同步与空间见 MAP.md / SPACE-DIFF.md。
# ---------------------------------------------------------------------------

from __future__ import annotations

import catlass.tla as tla

from smla_csa_common import (
    D_CHUNK,
    HEAD_DIM,
    KV_BLOCK,
    L0_AB_SLOTS,
    L0_D,
    L0_D64,
    L0_D_CHUNKS,
    NUM_D_CHUNKS,
    Q_HALF,
    Q_ROWS,
    aic_fix_l0c_to_ub_uf3,
    aic_nd2nz_k4_cm1,
    static_range,
)


@tla.jit
def l1_q_tensor(ptr, dtype, rows):
    """L1 Q 的 zN 视图。

    行数必须与写入侧（load_q 里的 make_tensor_like(..., gm0, zN)，gm0 为 (m, Q_HALF)）
    完全一致：zN(R, D) 的 dim-block 步长是 R*16，R 取 64 而 m=32 时读写步长 1024 vs 512，
    读回来的是一半真实、一半未初始化（实测 S ~1e36）。m==Q_ROWS 时两种写法退化相同，
    所以此前只有 heads=64 是对的。
    """
    return tla.make_tensor(
        ptr, tla.make_layout_with_tag((rows, Q_HALF), dtype, tla.arch.zN)
    )


@tla.jit
def l1_k_tensor(ptr, dtype):
    return tla.make_tensor(
        ptr, tla.make_layout_with_tag((HEAD_DIM, KV_BLOCK), dtype, tla.arch.nZ)
    )


@tla.jit
def l1_v_tensor(ptr, dtype):
    # 同一批字节的 zN 视图（V≡K 复用；与 csa_block_cube 的 l1V 别名一致）
    return tla.make_tensor(
        ptr, tla.make_layout_with_tag((KV_BLOCK, HEAD_DIM), dtype, tla.arch.zN)
    )


@tla.jit
def l1_v_seg_tensor(ptr, dtype):
    """单段 V 视图：zN(KV_BLOCK, D_CHUNK)。

    在上等价于 l1_v_tensor 的 (0, seg) 子块——zN(128,128) 的段内地址公式
    (kv%16)*16 + (kv//16)*256 + (dim%16) + (dim//16)*2048 与父张量
    zN(128,512) 完全一致，所以指针 + seg*16384 等价于逻辑列偏移 seg*128。
    """
    return tla.make_tensor(
        ptr, tla.make_layout_with_tag((KV_BLOCK, D_CHUNK), dtype, tla.arch.zN)
    )


@tla.jit
def l1_v_d64_tensor(ptr, dtype):
    """64-wide V slice in L1: zN(KV_BLOCK=128, 64), offset d_tile * 8192 el."""
    return tla.make_tensor(
        ptr, tla.make_layout_with_tag((KV_BLOCK, 64), dtype, tla.arch.zN)
    )


@tla.jit
def l1_p_tensor(ptr, dtype, rows):
    """L1 P 的 zN 视图。行数同样必须与写入侧一致（见 l1_q_tensor 的说明）：
    写入侧按 (m, KV_BLOCK) 声明、两个 AIV 各写 rows=m/2 行；
    若这里固定成 zN(Q_ROWS=64)，m=32 时 dim-block 步长读写不一致。"""
    return tla.make_tensor(
        ptr, tla.make_layout_with_tag((rows, KV_BLOCK), dtype, tla.arch.zN)
    )


@tla.jit
def load_q(sync, bufs, q_tensor, task, m, dtype, row_base=None, slot_a=0, slot_b=1):
    """Q 装入 L1 的两半区（闸门② 3 槽环，OT l1QBufNum=3/qHalfNum=2）。

    slot_a/slot_b：本任务占用的两半槽号（任务 n 占 {2n, 2n+1} mod 3，调用方
    由游标派生；缺省 (0,1) 等价旧固定 l1_q0/l1_q1 形态）。槽隔离不靠按槽
    flag（事件 id 上限 0..7，dst=MTE2 八位已满），靠信用 + 程序序：q_l0a_free
    为单 flag 计数信用（init 3、每任务 wait x2 / 末 tile set x2），稳态余额
    1<->3；任何 copy 发出前本任务两槽的释放 set 都已在程序序先执行——slot_b
    由同 body 前面的 is_last_tile set 归还（MTE1 in-order 保证 L1->L0 读完成后
    才放行 MTE2 重写），slot_a 由更早任务归还；第二次 wait 必然闸在最近的
    释放上（余额 1 < 2），不会提前放行。

    row_base：Q 展平行基址（元素行）。默认 task*m（无 SPLIT_G 或偶数 G，两半
    等宽）；IS_SPLIT_G 且 gSize 为奇数时由调用方传 tok*gSize + goIdx（OT
    sparse_flash_mla_csa_block_cube_arch35.h:267-275 的 offsetCalculator
    以 goIdx 定位 G 半区首行）。
    """
    c0 = 0
    base = task * m if row_base is None else row_base
    with tla.cube():
        if tla.const_expr(sync.opt_q_pre3):
            _q_slot_free_wait(sync, slot_a)
            _copy_q_half(
                bufs,
                q_tensor.ptr + base * HEAD_DIM,
                slot_a,
                m,
                dtype,
                opt_l2m=sync.opt_l2m,
                q_l2_bypass=sync.q_l2_bypass,
            )
            _q_slot_ready_set(sync, slot_a)
            _q_slot_free_wait(sync, slot_b)
            _copy_q_half(
                bufs,
                q_tensor.ptr + base * HEAD_DIM + Q_HALF,
                slot_b,
                m,
                dtype,
                opt_l2m=sync.opt_l2m,
                q_l2_bypass=sync.q_l2_bypass,
            )
            _q_slot_ready_set(sync, slot_b)
        else:
            tla.wait_flag(sync.q_l0a_free)
            _copy_q_half(
                bufs,
                q_tensor.ptr + base * HEAD_DIM,
                slot_a,
                m,
                dtype,
                opt_l2m=sync.opt_l2m,
                q_l2_bypass=sync.q_l2_bypass,
            )
            _copy_q_half(
                bufs,
                q_tensor.ptr + base * HEAD_DIM + Q_HALF,
                slot_b,
                m,
                dtype,
                opt_l2m=sync.opt_l2m,
                q_l2_bypass=sync.q_l2_bypass,
            )
            tla.set_flag(sync.q_l1_ready)


@tla.jit
def _copy_q_half(bufs, gm_ptr, slot, m, dtype, opt_l2m=False, q_l2_bypass=False):
    gm_h = tla.make_tensor(
        gm_ptr,
        tla.make_layout(
            tla.make_shape(m, Q_HALF),
            tla.make_stride(HEAD_DIM, 1),
            layoutTag=tla.arch.RowMajor,
        ),
    )
    l1q = l1_q_tensor(_q_slot_ptr(bufs, slot), dtype, m)
    if tla.const_expr(opt_l2m and q_l2_bypass):
        tla.copy(l1q, gm_h, cache_policy=tla.params.CachePolicy.DISABLE)
    else:
        tla.copy(l1q, gm_h)


@tla.jit
def _q_slot_free_wait(sync, slot):
    if slot == 0:
        tla.wait_flag(sync.q_l0a_free)
    elif slot == 1:
        tla.wait_flag(sync.q_l0a_free1)
    else:
        tla.wait_flag(sync.q_l0a_free2)


@tla.jit
def _q_slot_free_set(sync, slot):
    if slot == 0:
        tla.set_flag(sync.q_l0a_free)
    elif slot == 1:
        tla.set_flag(sync.q_l0a_free1)
    else:
        tla.set_flag(sync.q_l0a_free2)


@tla.jit
def _q_slot_ready_set(sync, slot):
    if slot == 0:
        tla.set_flag(sync.q_l1_ready)
    elif slot == 1:
        tla.set_flag(sync.q_l1_ready1)
    else:
        tla.set_flag(sync.q_l1_ready2)


@tla.jit
def _q_slot_ready_wait(sync, slot):
    if slot == 0:
        tla.wait_flag(sync.q_l1_ready)
    elif slot == 1:
        tla.wait_flag(sync.q_l1_ready1)
    else:
        tla.wait_flag(sync.q_l1_ready2)


@tla.jit
def load_q_half0(sync, bufs, q_tensor, task, m, dtype, slot_a=0):
    base = task * m
    with tla.cube():
        if tla.const_expr(sync.opt_q_pre3):
            _q_slot_free_wait(sync, slot_a)
        _copy_q_half(
            bufs,
            q_tensor.ptr + base * HEAD_DIM,
            slot_a,
            m,
            dtype,
            opt_l2m=sync.opt_l2m,
            q_l2_bypass=sync.q_l2_bypass,
        )
        if tla.const_expr(sync.opt_q_pre3):
            _q_slot_ready_set(sync, slot_a)


@tla.jit
def load_q_half1(sync, bufs, q_tensor, task, m, dtype, slot_b=1):
    base = task * m
    with tla.cube():
        if tla.const_expr(sync.opt_q_pre3):
            _q_slot_free_wait(sync, slot_b)
        else:
            tla.wait_flag(sync.q_l0a_free)
        _copy_q_half(
            bufs,
            q_tensor.ptr + base * HEAD_DIM + Q_HALF,
            slot_b,
            m,
            dtype,
            opt_l2m=sync.opt_l2m,
            q_l2_bypass=sync.q_l2_bypass,
        )
        if tla.const_expr(sync.opt_q_pre3):
            _q_slot_ready_set(sync, slot_b)
        else:
            tla.set_flag(sync.q_l1_ready)


@tla.jit
def _k_slot_wait(sync, slot):
    """按 slot 等/放 K 槽（flag 不能动态选取，用互斥的动态 if 分支发出）。"""
    if slot == 0:
        tla.wait_flag(sync.k_l0b_free0)
    if slot == 1:
        tla.wait_flag(sync.k_l0b_free1)
    if slot == 2:
        tla.wait_flag(sync.k_l0b_free2)


@tla.jit
def _k_slot_free(sync, slot):
    if slot == 0:
        tla.set_flag(sync.k_l0b_free0)
    if slot == 1:
        tla.set_flag(sync.k_l0b_free1)
    if slot == 2:
        tla.set_flag(sync.k_l0b_free2)


@tla.jit
def _k_slot_ready_wait(sync, slot):
    if slot == 0:
        tla.wait_flag(sync.k_l1_ready0)
    if slot == 1:
        tla.wait_flag(sync.k_l1_ready1)
    if slot == 2:
        tla.wait_flag(sync.k_l1_ready2)


@tla.jit
def _k_slot_ready_set(sync, slot):
    if slot == 0:
        tla.set_flag(sync.k_l1_ready0)
    if slot == 1:
        tla.set_flag(sync.k_l1_ready1)
    if slot == 2:
        tla.set_flag(sync.k_l1_ready2)


@tla.jit
def _k_slot_ptr(bufs, slot):
    return bufs.l1_k + slot * (HEAD_DIM * KV_BLOCK)


# ---- 闸门② Q 3 槽环（OT l1QBufNum=3）：槽隔离靠 3 槽连续基址 + 游标寻址与
# 信用/程序序（见 load_q 注），不按槽开 flag——事件 id 上限 0..7、dst=MTE2
# 八位已满。此处只保留槽基址派生。


@tla.jit
def _q_slot_ptr(bufs, slot):
    return bufs.l1_q + slot * (Q_ROWS * Q_HALF)


# AIC 侧 kv_stage 信用消费粒度（OT 对应形态）：wait 两个 AIV 后才整块拷贝。
# AIV 侧信用条数不变（每子块每 task 一条），AIC 侧每条 load 仍消费恰好两条
# （id0/id1 各一），所以计数与改前完全一致，不存在信用泄漏。


@tla.jit
def load_k_ptr(sync, bufs, src_ptr, n_load, slot, dtype, zero_ptr=None, tail_zero=0):
    """GM(ND) -> L1(nZ)，按 dSize 分 4 段（CopyToL1Nd2Nz，s2RealSize 裁剪）。

    src_ptr 是"第 0 行第 0 列"的基地址：稠密路径指向 ori_kv，稀疏路径指向 GM 暂存槽。
    两种来源的行布局相同（每行 HEAD_DIM 个连续元素），所以拷贝代码完全复用。

    F32：稀疏路径在调用方 `wait_kv_stage` 等齐两个 AIV 的信用后整块拷贝
    n_load 行（OT 同款）。
    """
    with tla.cube():
        _k_slot_wait(sync, slot)
        l1k = l1_k_tensor(_k_slot_ptr(bufs, slot), dtype)
        if tla.const_expr(sync.opt_l2m):
            if n_load > 0:
                aic_nd2nz_k4_cm1(
                    tla.recast_ptr(_k_slot_ptr(bufs, slot), dtype=tla.UInt8),
                    tla.recast_ptr(src_ptr, dtype=tla.UInt8),
                    n_load,
                )
        else:
            for ck in (0, 1, 2, 3):
                gm_k = tla.make_tensor(
                    src_ptr + (ck * D_CHUNK),
                    tla.make_layout(
                        tla.make_shape(D_CHUNK, n_load),
                        tla.make_stride(1, HEAD_DIM),
                        layoutTag=tla.arch.ColumnMajor,
                    ),
                )
                k_tile = tla.tile_view(
                    l1k, tla.make_shape(D_CHUNK, n_load), tla.make_coord(ck, 0)
                )
                tla.copy(k_tile, gm_k)
        if tail_zero == 1:
            # F19：尾部行清零（见 `_load_k_ptr_nocredit`）
            n_tail = KV_BLOCK - n_load
            if n_tail > 0:
                for c in (0, 1, 2, 3):
                    gm_z = tla.make_tensor(
                        zero_ptr + (c * D_CHUNK),
                        tla.make_layout(
                            tla.make_shape(D_CHUNK, n_tail),
                            tla.make_stride(1, HEAD_DIM),
                            layoutTag=tla.arch.ColumnMajor,
                        ),
                    )
                    k_tail = tla.get_tile(
                        l1k,
                        tla.make_coord(c * D_CHUNK, n_load),
                        tla.make_shape(D_CHUNK, n_tail),
                    )
                    tla.copy(k_tail, gm_z)
        _k_slot_ready_set(sync, slot)


@tla.jit
def _load_k_ptr_nocredit(
    sync, bufs, src_ptr, n_load, slot, dtype, zero_ptr=None, tail_zero=0
):
    """稠密路径用的整块拷贝（不涉及 kv_stage 信用，永不等待）。

    与 `load_k_ptr` 的整块拷贝形态同构；单独成函数是为了让 `load_k` 的稠密调用
    不受 AIV1 等待影响（稠密路径没有 AIV gather，等下去会死锁）。

    F19（tail 清零）：本 tile 只写 [0, n_load) 行，[n_load, KV_BLOCK) 行是 L1
    残留。P 在这些位置被 mask 成精确 0，但 IEEE 0*NaN/Inf = NaN/Inf——若残留恰好
    是 NaN/Inf 位型（L1 不清零、跨上下文残留），PV 的整列输出会被污染，随后
    fp16 CAST_ROUND+SAT 落成精确 0（实测 `smla-ext-base-0434`，kv_len=64<128）。
    `zero_ptr` 非空时把尾部行清零（源为 host 提供的常零 GM 张量，只读）。
    """
    with tla.cube():
        _k_slot_wait(sync, slot)
        l1k = l1_k_tensor(_k_slot_ptr(bufs, slot), dtype)
        if tla.const_expr(sync.opt_l2m):
            if n_load > 0:
                aic_nd2nz_k4_cm1(
                    tla.recast_ptr(_k_slot_ptr(bufs, slot), dtype=tla.UInt8),
                    tla.recast_ptr(src_ptr, dtype=tla.UInt8),
                    n_load,
                )
        else:
            for ck in (0, 1, 2, 3):
                gm_k = tla.make_tensor(
                    src_ptr + (ck * D_CHUNK),
                    tla.make_layout(
                        tla.make_shape(D_CHUNK, n_load),
                        tla.make_stride(1, HEAD_DIM),
                        layoutTag=tla.arch.ColumnMajor,
                    ),
                )
                k_tile = tla.tile_view(
                    l1k, tla.make_shape(D_CHUNK, n_load), tla.make_coord(ck, 0)
                )
                tla.copy(k_tile, gm_k)
        if tail_zero == 1:
            n_tail = KV_BLOCK - n_load
            if n_tail > 0:
                for c in (0, 1, 2, 3):
                    gm_z = tla.make_tensor(
                        zero_ptr + (c * D_CHUNK),
                        tla.make_layout(
                            tla.make_shape(D_CHUNK, n_tail),
                            tla.make_stride(1, HEAD_DIM),
                            layoutTag=tla.arch.ColumnMajor,
                        ),
                    )
                    k_tail = tla.get_tile(
                        l1k,
                        tla.make_coord(c * D_CHUNK, n_load),
                        tla.make_shape(D_CHUNK, n_tail),
                    )
                    tla.copy(k_tail, gm_z)
        _k_slot_ready_set(sync, slot)


@tla.jit
def load_k(
    sync,
    bufs,
    kv_tensor,
    kv_row0,
    tile,
    n_load,
    slot,
    dtype,
    zero_ptr=None,
    tail_zero=0,
):
    """稠密路径：从 kv_row0 + tile*KV_BLOCK 行起连续读 n_load 行。

    不再自己开 tla.cube()：_load_k_ptr_nocredit 已经开了。混合 kernel 里
    tla.cube 区域不能嵌套（前端报 "tla.cube and tla.vector scopes must not be
    nested"）。

    zero_ptr：F19 的常零 GM 源（见 `_load_k_ptr_nocredit`）；None 时行为不变。
    """
    base = kv_row0 + tile * KV_BLOCK
    _load_k_ptr_nocredit(
        sync,
        bufs,
        kv_tensor.ptr + base * HEAD_DIM,
        n_load,
        slot,
        dtype,
        zero_ptr,
        tail_zero,
    )


@tla.jit
def load_k_pa(
    sync,
    bufs,
    kv_tensor,
    bt,
    seq,
    start_row,
    n_load,
    slot,
    dtype,
    pa_block_size,
    pa_stride,
    bt_max,
    pages_bound,
    zero_ptr=None,
    tail_zero=0,
):
    """PA_BBND 稠密路径：逻辑行 [start_row, +n_load) 经 block table 落成物理行。

    对照 OT `GmCopyInToL1PA`（`CopyInL1.h:322-358`，KVLAYOUT::BBH 分支）：

      curS2Idx = startPos.s2Offset;                       # :328 窗口起点
      while copyFinishRowCnt < copyRowNum:                # :330 按页逐轮
        blockIdOffset = curS2Idx / blockSize              # :331 页号
        remainRowCnt  = curS2Idx % blockSize              # :332 页内起始行
        copyRowCnt    = blockSize - remainRowCnt          # :336 本页可搬行数
        if copyFinishRowCnt + copyRowCnt > copyRowNum:    # :337-339 末页截断
            copyRowCnt = copyRowNum - copyFinishRowCnt
        源 = phys*pageStride + remainRowCnt*headDim       # :341-357

    因此支持**非对齐窗口起点**与**任意 blockSize**（不要求 ≤128、不要求整除 128、
    不要求 kv_len 是 blockSize 的整数倍）。`start_row` 是序列内逻辑行（不含
    cu_seqlens 基址），页号查表用 `bt[seq*bt_max + page]`（OT 的 bIdx 基偏移）。

    `pages_bound` 是「≤KV_BLOCK 行窗口最多跨几页」的静态上界：
    1 + ceil((KV_BLOCK-1)/blockSize)。循环按上界静态展开，超出部分由
    `done < n_load` 守卫空转，等价于 OT 的 while 条件。

    目的端坐标：L1 K 槽是 nZ(HEAD_DIM, KV_BLOCK) 分段布局（每段 [128 行 × 128 d]，
    段内行步长 = D_CHUNK），不能按 HEAD_DIM 前移指针，也不能用 ``tile_view`` 的
    tile 粒度 coord 传累计行数（前端会算成 coord*shape）。用 ``tla.get_tile`` 的
    **扁平元素偏移** (0, done) 一次取整宽 HEAD_DIM 子块，行偏移由布局自行换算。
    """
    with tla.cube():
        _k_slot_wait(sync, slot)
        l1k = l1_k_tensor(_k_slot_ptr(bufs, slot), dtype)
        done = tla.as_numeric(0)
        cur = start_row
        for _p in static_range(pages_bound):
            if done < n_load:
                page = cur // pa_block_size
                rem = cur - page * pa_block_size
                phys = bt[seq * bt_max + page]
                if phys < 0:
                    phys = tla.as_numeric(0)
                cnt = pa_block_size - rem
                if done + cnt > n_load:
                    cnt = n_load - done
                gm_k = tla.make_tensor(
                    kv_tensor.ptr + phys * pa_stride + rem * HEAD_DIM,
                    tla.make_layout(
                        tla.make_shape(HEAD_DIM, cnt),
                        tla.make_stride(1, HEAD_DIM),
                        layoutTag=tla.arch.ColumnMajor,
                    ),
                )
                k_tile = tla.get_tile(
                    l1k,
                    tla.make_coord(0, done),
                    tla.make_shape(HEAD_DIM, cnt),
                )
                tla.copy(k_tile, gm_k)
                done = done + cnt
                cur = cur + cnt
        if tail_zero == 1:
            n_tail = KV_BLOCK - n_load
            if n_tail > 0:
                gm_z = tla.make_tensor(
                    zero_ptr,
                    tla.make_layout(
                        tla.make_shape(HEAD_DIM, n_tail),
                        tla.make_stride(1, HEAD_DIM),
                        layoutTag=tla.arch.ColumnMajor,
                    ),
                )
                k_tail = tla.get_tile(
                    l1k, tla.make_coord(0, n_load), tla.make_shape(HEAD_DIM, n_tail)
                )
                tla.copy(k_tail, gm_z)
        _k_slot_ready_set(sync, slot)


@tla.jit
def wait_kv_stage(sync):
    """AIC 等 AIV 把稀疏行写进 GM 暂存槽（OT: Stage0 跨核同步）。

    F32：这里先等 AIV0（它负责行 [0, first)，first=(n+1)//2），再等 AIV1 写完后半；
    仍保证每条 load 消费 id0/id1 各一条信用，与 AIV 侧每 task 各发一条完全对齐。
    """
    with tla.cube():
        tla.cross_core_wait_flag(sync.kv_stage_ready, tla.pipes.MTE2, aiv_id=0)
        tla.cross_core_wait_flag(sync.kv_stage_ready, tla.pipes.MTE2, aiv_id=1)


@tla.jit
def wait_kv_ring(sync):
    """R42：AIC 在本 tile 的 bmm1 前等 AIV 把 KV 行写进 L1 槽。

    与 `wait_kv_stage` 的区别：语义是 L1 槽就绪（不再是 GM 暂存槽），消费管是
    MTE1（ring 形态下 AIC 的 MTE2 完全不碰 KV）。MTE1 上等 cross flag 有既有
    先例（`wait_p_ready`）。**不自开 tla.cube()**：唯一调用点在 `iterate_bmm1`
    的 cube 区域内部，region 不能嵌套。
    """
    tla.cross_core_wait_flag(sync.kv_stage_ready, tla.pipes.MTE1, aiv_id=0)
    tla.cross_core_wait_flag(sync.kv_stage_ready, tla.pipes.MTE1, aiv_id=1)


@tla.jit
def wait_p_ready(sync):
    with tla.cube():
        tla.cross_core_wait_flag(sync.p_ready, tla.pipes.MTE1, aiv_id=0)
        tla.cross_core_wait_flag(sync.p_ready, tla.pipes.MTE1, aiv_id=1)


@tla.jit
def iterate_bmm1(
    sync,
    bufs,
    slot,
    m,
    is_last_tile,
    first_tile,
    dtype,
    slot_s=0,
    q_slot_a=0,
    q_slot_b=1,
    kv_ring=False,
    iscmp=0,
    pa_l1=False,
):
    """QK：4 段 x [m,128]x[128,128]，段间 L0C fp32 累加（R1/R2）。

    slot_s：F6 流水版的 UB S 槽号（bmm1Buffers[t%2]；串行调用方不传 = 槽 0）。
    F10 步骤 1：slot_s 同时是 L0C S 槽号（OT INNERCORE_L0C GetNext ring(2) 同款，
    csa_block_cube_arch35.h:392-454 的 l0C.GetNext()）——l0c_s 扩成连续双槽后按
    slot_s 寻址，wait s_fix_done / set s_mm_ready / set s_fix_done 全部按槽配对
    （s_fix_done1/s_mm_ready1，id=3，仅 swa preload 实例发出）。MMAD(t) 只等
    fixpipe(t-2)（同槽），不再等 fixpipe(t-1)——CUBE 管道可领先 FIX 一 tile。
    s_ready/s_free 本就是信用制单 flag，流水下 AIC 的连续 set 由 s_free 反压限深
    （init 预置 2，AIC 领先 vec1 至多 2 tile），无需双 flag。
    q_slot_a/q_slot_b（闸门② Q 3 槽环）：本任务两半 Q 的槽号（缺省 (0,1)），
    只用于寻址（OT :402-403/:411-412 的按槽到达等待在本仓不设——load_q 两半
    拷完才统一置 q_l1_ready，首 tile 单次等待语义与改前一致）；末 tile 4 段
    消费完后对 q_l0a_free set x2 归还两槽信用（OT :423-424 末 chunk SetFlag
    MTE1->MTE2 两半槽）。MMAD 分段/累加序/两半消费顺序不变（硬条件 2，只改
    寻址与 flag 配对）。
    F10 步骤 2：L1→L0A/L0B 的 4 个 mutex 换成 AB 槽 flag 对（OT static_matmul.h
    MatmulKStatic 的 M_MTE1/MTE1_M 同款，L0A/L0B 合并单 bufferId）：MTE1 wait
    ab_free{slot}→copy A/B→set ab_ready{slot}；CUBE wait ab_ready{slot}→mmad→
    set ab_free{slot}。槽 0/1 ping-pong，MTE1 可领先 CUBE 一槽；跨 tile/bmm1↔
    bmm2 按槽轮转连续衔接（每对每 tile 2 wait/2 set，init 预置各 1 永不枯竭）。"""
    c0 = 0
    c1 = 1
    with tla.cube():
        # R42 / 步 A：kv_ring 实例下 ori 与 cmp 的 KV 槽都由 AIV 直写 L1，
        # AIC 不做任何 KV 搬运；每个 tile 都消费 1 个 intra 槽 free 信用
        # （站点 3 的 release 逐 tile 归还），就绪由 AIV 的 cross flag 给。
        if pa_l1:
            # E17：仅 cmp tile 由 AIV gather+extern 直写 L1；ori tile 仍由 AIC 的
            # 站点 1 `load_k_pa` 写入并以 k_l1_ready 交棒。按 per-tile iscmp 选择
            # 就绪信用：cmp -> 等 L1 槽 free（intra k_l0b_free，由 load 迁移到此处
            # 消费）+ cross kv_stage_ready；ori -> 等 MTE2 的 k_l1_ready。
            if iscmp == 1:
                _k_slot_wait(sync, slot)
                wait_kv_ring(sync)
            else:
                _k_slot_ready_wait(sync, slot)
        elif kv_ring:
            _k_slot_wait(sync, slot)
            wait_kv_ring(sync)
        else:
            _k_slot_ready_wait(sync, slot)
        if first_tile:
            if tla.const_expr(sync.opt_q_pre3):
                _q_slot_ready_wait(sync, q_slot_a)
            else:
                tla.wait_flag(sync.q_l1_ready)
        l1k = l1_k_tensor(_k_slot_ptr(bufs, slot), dtype)
        # F6 双槽：连续分配（OT bmm1Buffers[2]），槽 stride = Q_ROWS//2*KV_BLOCK。
        ub_s_full = tla.make_tensor(
            bufs.ub_s + slot_s * (Q_ROWS // 2 * KV_BLOCK),
            tla.make_layout(
                tla.make_shape(m, KV_BLOCK),
                tla.make_stride(KV_BLOCK, 1),
                layoutTag=tla.arch.RowMajor,
            ),
        )
        l0c_s = tla.make_tensor_like(
            bufs.l0c_s + slot_s * (Q_ROWS * KV_BLOCK), ub_s_full, tla.arch.L0Clayout
        )
        if tla.const_expr(not sync.opt_cube64):
            if slot_s == 0:
                tla.wait_flag(sync.s_fix_done)
            if slot_s == 1:
                tla.wait_flag(sync.s_fix_done1)
        if tla.const_expr(sync.opt_cube128):
            # Small-M (m <= 32): QK split D=512 into 4 x 128 with 2-slot L0A/L0B ring.
            for c in static_range(4):
                half = c // 2
                d_local = c % 2
                if tla.const_expr(c == 2 and sync.opt_q_pre3):
                    if first_tile:
                        _q_slot_ready_wait(sync, q_slot_b)
                l1q = l1_q_tensor(
                    _q_slot_ptr(bufs, q_slot_a if half == 0 else q_slot_b), dtype, m
                )
                q_chunk = tla.tile_view(
                    l1q, tla.make_shape(m, D_CHUNK), tla.make_coord(c0, d_local)
                )
                k_chunk = tla.tile_view(
                    l1k, tla.make_shape(D_CHUNK, KV_BLOCK), tla.make_coord(c, 0)
                )
                slot_l0 = c % 2
                l0a_ptr = bufs.l0a0 if slot_l0 == 0 else bufs.l0a1
                l0b_ptr = bufs.l0b0 if slot_l0 == 0 else bufs.l0b1
                l0_q = tla.make_tensor_like(l0a_ptr, q_chunk)
                l0_k = tla.make_tensor_like(l0b_ptr, k_chunk)
                if slot_l0 == 0:
                    tla.wait_flag(sync.ab_free0)
                    tla.copy(l0_q, q_chunk)
                    tla.copy(l0_k, k_chunk)
                    tla.set_flag(sync.ab_ready0)
                    tla.wait_flag(sync.ab_ready0)
                else:
                    tla.wait_flag(sync.ab_free1)
                    tla.copy(l0_q, q_chunk)
                    tla.copy(l0_k, k_chunk)
                    tla.set_flag(sync.ab_ready1)
                    tla.wait_flag(sync.ab_ready1)
                tla.mmad(
                    l0c_s,
                    l0_q,
                    l0_k,
                    init_c=(c == 0),
                    unit_flag=(0b11 if c == 3 else 0b10),
                    compute_order=tla.params.ComputeOrder.M_FIRST,
                )
                if slot_l0 == 0:
                    tla.set_flag(sync.ab_free0)
                else:
                    tla.set_flag(sync.ab_free1)
                if tla.const_expr(sync.opt_q_pre3 and c == 1):
                    if is_last_tile:
                        _q_slot_free_set(sync, q_slot_a)
                if tla.const_expr(sync.opt_q_pre3 and c == 3):
                    if is_last_tile:
                        _q_slot_free_set(sync, q_slot_b)
        elif tla.const_expr(sync.opt_cube64):
            # Track 2: QK split D=512 into 8 x 64 with 4-slot L0A & 4-slot L0B ring,
            # ComputeOrder.M_FIRST, unit_flag=0b10 (d=0..6) / 0b11 (d=7).
            for c in static_range(8):
                half = c // 4
                d_local = c % 4
                if tla.const_expr(c == 4 and sync.opt_q_pre3):
                    if first_tile:
                        _q_slot_ready_wait(sync, q_slot_b)
                l1q = l1_q_tensor(
                    _q_slot_ptr(bufs, q_slot_a if half == 0 else q_slot_b), dtype, m
                )
                q_chunk = tla.tile_view(
                    l1q, tla.make_shape(m, 64), tla.make_coord(c0, d_local)
                )
                k_chunk = tla.tile_view(
                    l1k, tla.make_shape(64, KV_BLOCK), tla.make_coord(c, 0)
                )
                slot_l0 = c % 4
                l0a_ptr = (
                    bufs.l0a0
                    if slot_l0 == 0
                    else (
                        bufs.l0a1
                        if slot_l0 == 1
                        else (bufs.l0a2 if slot_l0 == 2 else bufs.l0a3)
                    )
                )
                l0b_ptr = (
                    bufs.l0b0
                    if slot_l0 == 0
                    else (
                        bufs.l0b1
                        if slot_l0 == 1
                        else (bufs.l0b2 if slot_l0 == 2 else bufs.l0b3)
                    )
                )
                l0_q = tla.make_tensor_like(l0a_ptr, q_chunk)
                l0_k = tla.make_tensor_like(l0b_ptr, k_chunk)
                if slot_l0 == 0:
                    tla.wait_flag(sync.ab_free0)
                    tla.copy(l0_q, q_chunk)
                    tla.copy(l0_k, k_chunk)
                    tla.set_flag(sync.ab_ready0)
                    tla.wait_flag(sync.ab_ready0)
                elif slot_l0 == 1:
                    tla.wait_flag(sync.ab_free1)
                    tla.copy(l0_q, q_chunk)
                    tla.copy(l0_k, k_chunk)
                    tla.set_flag(sync.ab_ready1)
                    tla.wait_flag(sync.ab_ready1)
                elif slot_l0 == 2:
                    tla.wait_flag(sync.ab_free2)
                    tla.copy(l0_q, q_chunk)
                    tla.copy(l0_k, k_chunk)
                    tla.set_flag(sync.ab_ready2)
                    tla.wait_flag(sync.ab_ready2)
                else:
                    tla.wait_flag(sync.ab_free3)
                    tla.copy(l0_q, q_chunk)
                    tla.copy(l0_k, k_chunk)
                    tla.set_flag(sync.ab_ready3)
                    tla.wait_flag(sync.ab_ready3)
                tla.mmad(
                    l0c_s,
                    l0_q,
                    l0_k,
                    init_c=(c == 0),
                    unit_flag=(0b11 if c == 7 else 0b10),
                    compute_order=tla.params.ComputeOrder.M_FIRST,
                )
                if slot_l0 == 0:
                    tla.set_flag(sync.ab_free0)
                elif slot_l0 == 1:
                    tla.set_flag(sync.ab_free1)
                elif slot_l0 == 2:
                    tla.set_flag(sync.ab_free2)
                else:
                    tla.set_flag(sync.ab_free3)
                if tla.const_expr(sync.opt_q_pre3 and c == 3):
                    if is_last_tile:
                        _q_slot_free_set(sync, q_slot_a)
                if tla.const_expr(sync.opt_q_pre3 and c == 7):
                    if is_last_tile:
                        _q_slot_free_set(sync, q_slot_b)
        elif L0_D64:
            # 步 B：QK 沿 D 拆 8 段（D=64），L0A/L0B 4 槽 ring。
            for c in static_range(L0_D_CHUNKS):
                half = c // (Q_HALF // L0_D)
                d_local = c % (Q_HALF // L0_D)
                l1q = l1_q_tensor(
                    _q_slot_ptr(bufs, q_slot_a if half == 0 else q_slot_b), dtype, m
                )
                q_chunk = tla.tile_view(
                    l1q, tla.make_shape(m, L0_D), tla.make_coord(c0, d_local)
                )
                k_chunk = tla.tile_view(
                    l1k, tla.make_shape(L0_D, KV_BLOCK), tla.make_coord(c, 0)
                )
                slot_l0 = c % L0_AB_SLOTS
                l0a_ptr = (
                    bufs.l0a0
                    if slot_l0 == 0
                    else (
                        bufs.l0a1
                        if slot_l0 == 1
                        else (bufs.l0a2 if slot_l0 == 2 else bufs.l0a3)
                    )
                )
                l0b_ptr = (
                    bufs.l0b0
                    if slot_l0 == 0
                    else (
                        bufs.l0b1
                        if slot_l0 == 1
                        else (bufs.l0b2 if slot_l0 == 2 else bufs.l0b3)
                    )
                )
                l0_q = tla.make_tensor_like(l0a_ptr, q_chunk)
                l0_k = tla.make_tensor_like(l0b_ptr, k_chunk)
                # 深度 4 单信用对：token 按 FIFO，第 c 次 copy 等第 c-4 次
                # mmad 的 token（同一物理槽 c%4），故无需槽 2/3 的额外事件 id。
                tla.wait_flag(sync.ab_free0)
                tla.copy(l0_q, q_chunk)
                tla.copy(l0_k, k_chunk)
                tla.set_flag(sync.ab_ready0)
                tla.wait_flag(sync.ab_ready0)
                tla.mmad(l0c_s, l0_q, l0_k, init_c=(c == 0))
                tla.set_flag(sync.ab_free0)
        else:
            # 原步 A/步二形态（2 槽、4 x 128），IR 与改前逐条一致。
            for c in (0, 1, 2, 3):
                half = c // 2
                l1q = l1_q_tensor(
                    _q_slot_ptr(bufs, q_slot_a if half == 0 else q_slot_b), dtype, m
                )
                q_chunk = tla.tile_view(
                    l1q, tla.make_shape(m, D_CHUNK), tla.make_coord(c0, c % 2)
                )
                k_chunk = tla.tile_view(
                    l1k, tla.make_shape(D_CHUNK, KV_BLOCK), tla.make_coord(c, 0)
                )
                slot_l0 = c % 2
                l0a_ptr = bufs.l0a0 if slot_l0 == 0 else bufs.l0a1
                l0b_ptr = bufs.l0b0 if slot_l0 == 0 else bufs.l0b1
                l0_q = tla.make_tensor_like(l0a_ptr, q_chunk)
                l0_k = tla.make_tensor_like(l0b_ptr, k_chunk)
                if slot_l0 == 0:
                    tla.wait_flag(sync.ab_free0)
                    tla.copy(l0_q, q_chunk)
                    tla.copy(l0_k, k_chunk)
                    tla.set_flag(sync.ab_ready0)
                else:
                    tla.wait_flag(sync.ab_free1)
                    tla.copy(l0_q, q_chunk)
                    tla.copy(l0_k, k_chunk)
                    tla.set_flag(sync.ab_ready1)
                if slot_l0 == 0:
                    tla.wait_flag(sync.ab_ready0)
                else:
                    tla.wait_flag(sync.ab_ready1)
                tla.mmad(
                    l0c_s,
                    l0_q,
                    l0_k,
                    init_c=(c == 0),
                )
                if slot_l0 == 0:
                    tla.set_flag(sync.ab_free0)
                else:
                    tla.set_flag(sync.ab_free1)

        if tla.const_expr(sync.opt_cube64):
            tla.cross_core_wait_flag(sync.s_free, tla.pipes.FIX, aiv_id=0)
            tla.cross_core_wait_flag(sync.s_free, tla.pipes.FIX, aiv_id=1)
            if tla.const_expr(sync.opt_cube64 and (m == 1 or m % 2 == 0)):
                m_f_s = ((m + 15) // 16) * 16
                # m == 1: SPLIT_M with m_fix=2 sends row 0 to AIV0 and unused row 1
                # safely into AIV1's UB (AIV1 has sub*rows >= m and never writes GM).
                m_fix_s = 2 if m == 1 else m
                aic_fix_l0c_to_ub_uf3(
                    bufs.ub_s + slot_s * (Q_ROWS // 2 * KV_BLOCK),
                    bufs.l0c_s + slot_s * (Q_ROWS * KV_BLOCK),
                    KV_BLOCK,
                    m_fix_s,
                    KV_BLOCK,
                    m_f_s,
                )
            else:
                tla.copy(
                    ub_s_full,
                    l0c_s,
                    tla.params.CopyL0C2DstParams(
                        unit_flag=0b11,
                        l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M,
                    ),
                )
        else:
            if slot_s == 0:
                tla.set_flag(sync.s_mm_ready)
                tla.wait_flag(sync.s_mm_ready)
            if slot_s == 1:
                tla.set_flag(sync.s_mm_ready1)
                tla.wait_flag(sync.s_mm_ready1)
            tla.cross_core_wait_flag(sync.s_free, tla.pipes.FIX, aiv_id=0)
            tla.cross_core_wait_flag(sync.s_free, tla.pipes.FIX, aiv_id=1)
            tla.copy(
                ub_s_full,
                l0c_s,
                tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M),
            )
            if slot_s == 0:
                tla.set_flag(sync.s_fix_done)
            if slot_s == 1:
                tla.set_flag(sync.s_fix_done1)
        tla.cross_core_set_flag(sync.s_ready, tla.pipes.FIX, aiv_id=0)
        tla.cross_core_set_flag(sync.s_ready, tla.pipes.FIX, aiv_id=1)
        if tla.const_expr(not sync.opt_q_pre3):
            if is_last_tile:
                # 闸门②变体 B：归还 x1（原始握手）。
                tla.set_flag(sync.q_l0a_free)


@tla.jit
def iterate_bmm2_seg(sync, bufs, seg, slot_k, m, dtype, slot_p=0):
    """PV 单段：P[m,128] x V[128,128]，N 维第 seg 段（F17 步骤 B：只装载 +
    mmad；fixpipe 拆到 iterate_bmm2_fixpipe，每 tile 一次）。

    slot_p：F6 流水版的 L1 P 槽号（l1PBuffers[(t)%2]；串行调用方不传 = 槽 0）。
    F14 步骤 A：P 槽读完成不再归还（p_free 回执去除，OT IterateBmm2Impl 对
    l1PBuffers 单向 + 双槽 slack 同款；踩写安全链见 vector 侧 s_ready 注释）。
    F10 步骤 2：L1→L0A/L0B AB 槽 flag 对按 slot=seg%2 配对（数据与 flag 一致）。
    F17 步骤 B：mmad 目标改单槽 l0c_o 的第 seg 线性段
    make_tensor(l0c_o + seg*(m*D_CHUNK), (m,D_CHUNK), L0Clayout)（OT
    cL0Tensor[n*L0Coffset] 同构；探针 mode9/10 终判该形态成立）。段偏移必须落
    在 allocate 声明内——l0c_o 声明 Q_ROWS*HEAD_DIM=32768 元素，m=64 时 4 段
    8192 元素恰好占满。每窗口 init_c=True：窗口互不重叠、各自只被本 tile 的
    1 次 mmad 写（探针 t4/t9/t10 证实；L0C 不随 kernel 清零，n==0 形态会让
    窗口 1-3 累加跨 kernel 残留）。seg0 等 o_fix_done0（上一 tile 的 fixpipe
    释放 l0c_o，init 预置 1 供首 tile）；其余同步（o_mm_ready 闸、o_free 等
    待、整块 copy、o_fix_done/o_ready 归还）全部挪进 fixpipe。"""
    c0 = 0
    with tla.cube():
        # F6 双槽：连续分配（OT l1PBuffers[2]），槽 stride = Q_ROWS*KV_BLOCK。
        l1p = l1_p_tensor(bufs.l1_p + slot_p * (Q_ROWS * KV_BLOCK), dtype, m)
        v_seg = l1_v_seg_tensor(
            _k_slot_ptr(bufs, slot_k) + seg * (KV_BLOCK * D_CHUNK), dtype
        )
        # F17 步骤 B：线性段目标（探针 mode9/10 终判形态；L0C 视图偏移必须
        # 落在 allocate 声明范围内）。
        # L0C 段偏移必须按 n-fract 网格对齐：IR 实证 (64,128) L0Clayout =
        # shape<(16,4),(16,8)> stride<(16,256),(1,1024)>，n-fract stride =
        # ceil(m/16)*256，段内 8 个 n-fract → 段偏移 = seg*ceil(m/16)*2048 元素
        # （= seg*round16(m)*D_CHUNK）。m 是 16 的倍数时与 seg*(m*D_CHUNK) 相同
        # （#008 m=64 指令流逐位不变、探针 m=32/64 已证）；m<16 时旧式
        # seg*(m*128) 段间重叠（#007 m=8 ratio 恰 0.125=只有段 0 对）或破坏
        # fract 对齐（#004 m=1 aicore 异常）。
        m_f = ((m + 15) // 16) * 16
        l0c_seg = tla.make_tensor(
            bufs.l0c_o + seg * (m_f * D_CHUNK),
            tla.make_layout_with_tag((m, D_CHUNK), tla.Float32, tla.arch.L0Clayout),
        )
        # F17 步骤 B：l0c_o 单槽链——只 seg0 等 FIX 释放（窗口 0 与上一 tile
        # fixpipe 的读构成 WAR；窗口 1-3 依赖同 tile seg0 之后的程序序）。
        if seg == 0:
            tla.wait_flag(sync.o_fix_done0)
        if L0_D64:
            # 步 B：PV 的 K(=s2=128) 拆 2 段（64）；P 两片常驻 l0a0/l0a1
            # （seg0 装入，seg1-3 复用），V 按 (seg*2+kc)%4 轮转 4 个 L0B 槽。
            num_kc = KV_BLOCK // L0_D
            for kc in static_range(num_kc):
                slot = (seg * num_kc + kc) % L0_AB_SLOTS
                p_chunk = tla.tile_view(
                    l1p, tla.make_shape(m, L0_D), tla.make_coord(c0, kc)
                )
                v_chunk = tla.tile_view(
                    v_seg, tla.make_shape(L0_D, KV_BLOCK), tla.make_coord(kc, 0)
                )
                l0a_ptr = bufs.l0a0 if kc == 0 else bufs.l0a1
                load_p = seg == 0
                l0b_ptr = (
                    bufs.l0b0
                    if slot == 0
                    else (
                        bufs.l0b1
                        if slot == 1
                        else (bufs.l0b2 if slot == 2 else bufs.l0b3)
                    )
                )
                l0_p = tla.make_tensor_like(l0a_ptr, p_chunk)
                l0_v = tla.make_tensor_like(l0b_ptr, v_chunk)
                # 与 bmm1 共用的深度 4 单信用对：第 n 次 copy 等第 n-4 次
                # mmad 的 token（同一 L0B 槽 n%4）；P 只在 seg0 装、之后常驻。
                tla.wait_flag(sync.ab_free0)
                if load_p:
                    tla.copy(l0_p, p_chunk)
                tla.copy(l0_v, v_chunk)
                tla.set_flag(sync.ab_ready0)
                tla.wait_flag(sync.ab_ready0)
                tla.mmad(l0c_seg, l0_p, l0_v, init_c=(kc == 0), unit_flag=0)
                tla.set_flag(sync.ab_free0)
        else:
            # 原步 A/步二形态（2 槽、K 不拆），IR 与改前逐条一致。
            p_tile = tla.tile_view(
                l1p, tla.make_shape(m, KV_BLOCK), tla.make_coord(c0, c0)
            )
            slot = seg % 2
            l0a_ptr = bufs.l0a0 if slot == 0 else bufs.l0a1
            l0b_ptr = bufs.l0b0 if slot == 0 else bufs.l0b1
            l0_p = tla.make_tensor_like(l0a_ptr, p_tile)
            l0_v = tla.make_tensor_like(l0b_ptr, v_seg)
            if seg < 2:
                if slot == 0:
                    tla.wait_flag(sync.ab_free0)
                    tla.copy(l0_p, p_tile)
                    tla.copy(l0_v, v_seg)
                    tla.set_flag(sync.ab_ready0)
                else:
                    tla.wait_flag(sync.ab_free1)
                    tla.copy(l0_p, p_tile)
                    tla.copy(l0_v, v_seg)
                    tla.set_flag(sync.ab_ready1)
            else:
                if slot == 0:
                    tla.wait_flag(sync.ab_free0)
                    tla.copy(l0_v, v_seg)
                    tla.set_flag(sync.ab_ready0)
                else:
                    tla.wait_flag(sync.ab_free1)
                    tla.copy(l0_v, v_seg)
                    tla.set_flag(sync.ab_ready1)
            if slot == 0:
                tla.wait_flag(sync.ab_ready0)
            else:
                tla.wait_flag(sync.ab_ready1)
            tla.mmad(l0c_seg, l0_p, l0_v, init_c=True, unit_flag=0)
            if slot == 0:
                tla.set_flag(sync.ab_free0)
            else:
                tla.set_flag(sync.ab_free1)


@tla.jit
def iterate_bmm2_cube64(sync, bufs, slot_k, m, dtype, slot_p=0, dst_is_o=None):
    """64-wide N-split 8-slot L0C ring for BMM2 (PV).
    - K=128 unsplit, N=512 split into 8 x 64 (m<=64, k=128, n=64).
    - P(m,128) loaded once at d_tile=0 into dedicated l0a_p (16 KB).
    - V(128,64) rotated across 4-slot l0b0..3 ring (64 KB) with 4 independent flags.
    - 8 disjoint 16 KB L0C slots in bufs.l0c_o (128 KB), ComputeOrder.M_FIRST (matches BMM1 to
      avoid in-loop MOV CTRL flushes), init_c=True, unit_flag=0b11, overlapped per-d_tile FixPipe.
    - Supports dst_is_o (direct-to-ub_o on tile 0) to pair with process_vec2_full_opt.
    """
    c0 = 0
    with tla.cube():
        l1p = l1_p_tensor(bufs.l1_p + slot_p * (Q_ROWS * KV_BLOCK), dtype, m)
        p_tile = tla.tile_view(l1p, tla.make_shape(m, KV_BLOCK), tla.make_coord(c0, c0))
        l0_p = tla.make_tensor_like(bufs.l0a_p, p_tile)
        m_f = ((m + 15) // 16) * 16

        ub_dst_ptr = bufs.ub_otmp_full
        if dst_is_o is not None:
            if dst_is_o:
                ub_dst_ptr = bufs.ub_o
        ub_dst_full = tla.make_tensor(
            ub_dst_ptr,
            tla.make_layout(
                tla.make_shape(m, HEAD_DIM),
                tla.make_stride(HEAD_DIM, 1),
                layoutTag=tla.arch.RowMajor,
            ),
        )

        if tla.const_expr(sync.opt_cube128):
            l1v_full = l1_v_tensor(_k_slot_ptr(bufs, slot_k), dtype)
            for d_tile in static_range(4):
                v_d128 = tla.tile_view(
                    l1v_full,
                    tla.make_shape(KV_BLOCK, D_CHUNK),
                    tla.make_coord(c0, d_tile),
                )
                slot_b = d_tile % 2
                l0b_ptr = bufs.l0b0 if slot_b == 0 else bufs.l0b1
                l0_v = tla.make_tensor_like(l0b_ptr, v_d128)
                l0c_d = tla.make_tensor(
                    bufs.l0c_o + d_tile * (m_f * D_CHUNK),
                    tla.make_layout_with_tag(
                        (m, D_CHUNK), tla.Float32, tla.arch.L0Clayout
                    ),
                )
                if d_tile == 0:
                    tla.wait_flag(sync.ab_p_free)
                    tla.copy(l0_p, p_tile)
                    tla.set_flag(sync.ab_p_ready)
                if slot_b == 0:
                    tla.wait_flag(sync.ab_free0)
                    tla.copy(l0_v, v_d128)
                    tla.set_flag(sync.ab_ready0)
                else:
                    tla.wait_flag(sync.ab_free1)
                    tla.copy(l0_v, v_d128)
                    tla.set_flag(sync.ab_ready1)
                if d_tile == 0:
                    tla.wait_flag(sync.ab_p_ready)
                if slot_b == 0:
                    tla.wait_flag(sync.ab_ready0)
                else:
                    tla.wait_flag(sync.ab_ready1)
                tla.mmad(
                    l0c_d,
                    l0_p,
                    l0_v,
                    init_c=True,
                    unit_flag=0b11,
                    compute_order=tla.params.ComputeOrder.M_FIRST,
                )
                if slot_b == 0:
                    tla.set_flag(sync.ab_free0)
                else:
                    tla.set_flag(sync.ab_free1)
                if d_tile == 3:
                    tla.set_flag(sync.ab_p_free)

                if d_tile == 0:
                    tla.cross_core_wait_flag(sync.o_free, tla.pipes.FIX, aiv_id=0)
                    tla.cross_core_wait_flag(sync.o_free, tla.pipes.FIX, aiv_id=1)
                if tla.const_expr(sync.opt_cube64 and (m == 1 or m % 2 == 0)):
                    m_fix_o = 2 if m == 1 else m
                    aic_fix_l0c_to_ub_uf3(
                        ub_dst_ptr + d_tile * D_CHUNK,
                        bufs.l0c_o + d_tile * (m_f * D_CHUNK),
                        D_CHUNK,
                        m_fix_o,
                        HEAD_DIM,
                        m_f,
                    )
                else:
                    ub_dst_d = tla.tile_view(
                        ub_dst_full,
                        tla.make_shape(m, D_CHUNK),
                        tla.make_coord(c0, d_tile),
                    )
                    tla.copy(
                        ub_dst_d,
                        l0c_d,
                        tla.params.CopyL0C2DstParams(
                            unit_flag=0b11,
                            l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M,
                        ),
                    )
                if d_tile == 3:
                    tla.cross_core_set_flag(sync.o_ready, tla.pipes.FIX, aiv_id=0)
                    tla.cross_core_set_flag(sync.o_ready, tla.pipes.FIX, aiv_id=1)
        else:
            for d_tile in static_range(8):
                v_d64 = l1_v_d64_tensor(
                    _k_slot_ptr(bufs, slot_k) + d_tile * (KV_BLOCK * 64), dtype
                )
                slot_b = d_tile % 4
                l0b_ptr = (
                    bufs.l0b0
                    if slot_b == 0
                    else (
                        bufs.l0b1
                        if slot_b == 1
                        else (bufs.l0b2 if slot_b == 2 else bufs.l0b3)
                    )
                )
                l0_v = tla.make_tensor_like(l0b_ptr, v_d64)
                l0c_d = tla.make_tensor(
                    bufs.l0c_o + d_tile * (m_f * 64),
                    tla.make_layout_with_tag((m, 64), tla.Float32, tla.arch.L0Clayout),
                )
                if d_tile == 0:
                    tla.wait_flag(sync.ab_p_free)
                    tla.copy(l0_p, p_tile)
                    tla.set_flag(sync.ab_p_ready)
                if slot_b == 0:
                    tla.wait_flag(sync.ab_free0)
                    tla.copy(l0_v, v_d64)
                    tla.set_flag(sync.ab_ready0)
                elif slot_b == 1:
                    tla.wait_flag(sync.ab_free1)
                    tla.copy(l0_v, v_d64)
                    tla.set_flag(sync.ab_ready1)
                elif slot_b == 2:
                    tla.wait_flag(sync.ab_free2)
                    tla.copy(l0_v, v_d64)
                    tla.set_flag(sync.ab_ready2)
                else:
                    tla.wait_flag(sync.ab_free3)
                    tla.copy(l0_v, v_d64)
                    tla.set_flag(sync.ab_ready3)
                if d_tile == 0:
                    tla.wait_flag(sync.ab_p_ready)
                if slot_b == 0:
                    tla.wait_flag(sync.ab_ready0)
                elif slot_b == 1:
                    tla.wait_flag(sync.ab_ready1)
                elif slot_b == 2:
                    tla.wait_flag(sync.ab_ready2)
                else:
                    tla.wait_flag(sync.ab_ready3)
                tla.mmad(
                    l0c_d,
                    l0_p,
                    l0_v,
                    init_c=True,
                    unit_flag=0b11,
                    compute_order=tla.params.ComputeOrder.M_FIRST,
                )
                if slot_b == 0:
                    tla.set_flag(sync.ab_free0)
                elif slot_b == 1:
                    tla.set_flag(sync.ab_free1)
                elif slot_b == 2:
                    tla.set_flag(sync.ab_free2)
                else:
                    tla.set_flag(sync.ab_free3)
                if d_tile == 7:
                    tla.set_flag(sync.ab_p_free)

                if d_tile == 0:
                    tla.cross_core_wait_flag(sync.o_free, tla.pipes.FIX, aiv_id=0)
                    tla.cross_core_wait_flag(sync.o_free, tla.pipes.FIX, aiv_id=1)
                if tla.const_expr(sync.opt_cube64 and (m == 1 or m % 2 == 0)):
                    m_fix_o = 2 if m == 1 else m
                    aic_fix_l0c_to_ub_uf3(
                        ub_dst_ptr + d_tile * 64,
                        bufs.l0c_o + d_tile * (m_f * 64),
                        64,
                        m_fix_o,
                        HEAD_DIM,
                        m_f,
                    )
                else:
                    ub_dst_d = tla.tile_view(
                        ub_dst_full, tla.make_shape(m, 64), tla.make_coord(c0, d_tile)
                    )
                    tla.copy(
                        ub_dst_d,
                        l0c_d,
                        tla.params.CopyL0C2DstParams(
                            unit_flag=0b11,
                            l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M,
                        ),
                    )
                if d_tile == 7:
                    tla.cross_core_set_flag(sync.o_ready, tla.pipes.FIX, aiv_id=0)
                    tla.cross_core_set_flag(sync.o_ready, tla.pipes.FIX, aiv_id=1)


@tla.jit
def iterate_bmm2_fixpipe(sync, bufs, m):
    """F17 步骤 B：每 tile 一次的整块 fixpipe——4 段 mmad 完成后，把单槽
    l0c_o 的 (m,HEAD_DIM) fp32 一次 SPLIT_M 搬到 ub_otmp_full（OT IterateBmm2
    每 tile 一次 fixpipe 同款；探针 mode1 终判：整块 (64,512) SPLIT_M 成立，
    L0C 整块视图必须 make_tensor 直建——make_tensor_like 会 clamp 到 buf 声明
    形状，IR 实证 fixpipe 源被截成 memref<8192xf32>）。

    同步（每 tile 恰各一次，信用平衡推演见 ITERATIONS.md F17 节）：
    - set+wait o_mm_ready0：MMAD 结果对 FIX 可见的管线闸（原每段一对 → 现整
      块一对）；
    - cross_core_wait o_free x2：ub_otmp_full 单缓冲，等上一 tile 的 vec2
      读完（vec2 seg3 归还；init_sync 预置 x2 供首 tile）；
    - copy SPLIT_M 整块（每 AIV 半行 32x512 = 65,536 B = ub_otmp_full 全额）；
    - set o_fix_done0：释放 l0c_o 给下一 tile 的 seg0 mmad（init 预置 1）；
    - cross_core_set o_ready x2：vec2 seg0 消费。"""
    c0 = 0
    with tla.cube():
        ub_full = tla.make_tensor(
            bufs.ub_otmp_full,
            tla.make_layout(
                tla.make_shape(m, HEAD_DIM),
                tla.make_stride(HEAD_DIM, 1),
                layoutTag=tla.arch.RowMajor,
            ),
        )
        l0c_full = tla.make_tensor(
            bufs.l0c_o,
            tla.make_layout_with_tag((m, HEAD_DIM), tla.Float32, tla.arch.L0Clayout),
        )
        tla.set_flag(sync.o_mm_ready0)
        tla.wait_flag(sync.o_mm_ready0)
        tla.cross_core_wait_flag(sync.o_free, tla.pipes.FIX, aiv_id=0)
        tla.cross_core_wait_flag(sync.o_free, tla.pipes.FIX, aiv_id=1)
        tla.copy(
            ub_full,
            l0c_full,
            tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M),
        )
        tla.set_flag(sync.o_fix_done0)
        tla.cross_core_set_flag(sync.o_ready, tla.pipes.FIX, aiv_id=0)
        tla.cross_core_set_flag(sync.o_ready, tla.pipes.FIX, aiv_id=1)


@tla.jit
def release_k(sync, slot, kv_ring=False, free_tok=0):
    """V 段全部装入 L0B 后归还 K/V 的 L1 槽。

    R42（kv_ring=True）：除 intra 的 `k_l0b_free{slot}` 外，AIC 还要向 AIV 归还
    `kv_free` token（cross AIC MTE1 -> AIV MTE3），但**只在后继 tile 是 cmp 时**
    发（`free_tok` 由 kernel 在 body 内按"tile x 存在且 r0_iscmp==1"算出）。
    理由：AIV 只写 cmp tile；ori tile 的槽由 AIC 自己读写并由 intra 信用覆盖。
    每 cmp tile 恰好一收一发（AIV 侧同样只在 cmp 且 x>=3 时消费），ori 占多数
    时零额外跨核同步。set 挂 MTE1：与 `_k_slot_free` 同管，程序序保证本 tile
    的 L1->L0B 拷贝都发出后才 set。
    """
    with tla.cube():
        _k_slot_free(sync, slot)
        if kv_ring:
            if free_tok == 1:
                tla.cross_core_set_flag(sync.kv_free, tla.pipes.MTE1, aiv_id=0)
                tla.cross_core_set_flag(sync.kv_free, tla.pipes.MTE1, aiv_id=1)


# A 步五（已采纳，F45，2026-11-17）：共享暂存的跨块保护用 OT 原形态节流
# （`split_g_mte2_pair`，mode-0 MTE2 set/wait，与 `csa_block_cube.h:249-253`
# 字面同构）。OT 靠它 + 3 槽环 + 两核工作量恒等来兜底，不保证严格会合；
# 用全核会合替代属于过度强化，纠正它是保真修正。承载性实测：
#   - 去掉节流（`--mte2-barrier 0`）⇒ #047 13 轮 bit_exact 0.57–0.65、
#     #043 0.754，错的全是整半区 64 头 ⇒ 节流承载（不是"可有可无的配速"）。
#   - 真会合的 `PipeBarrier<ALL>` 是承载的（去掉 ⇒ bit 0.9657），但只在
#     真会合形态里存在；节流形态不发会合，故不需要它。


@tla.jit
def split_g_mte2_pair(sync):
    """F27：IS_SPLIT_G 的核间 MTE2 屏障对（全核 FFTS 屏障，OT flagId=15）。

    对照 OT `sparse_flash_mla_csa_kernel_arch35.h`：
      - :347-348（ProcessMainLoop 内 IterateLoadQK 之后/之前的 load 站点）
      - :357-358（同上，sparse 分支）
      - :642-648（hasLoad==0 或跳过的 load：smlaMaxS2LoopCnt-- 后 set+wait）
      - :674-680（AIC 末尾 for loopCnt < smlaMaxS2LoopCnt 的余量补发）
    每发出一次就与全部 AIC 对齐一轮；各核条数必须恒等，否则屏障计数不等会挂死。
    本仓由 kernel 侧"预算排空"保证：配额 = host 计算的 FA_S2_MAX_NUM（tiling[1]），
    每次真实 load / 跳过 load 各消费 1、末尾把剩余配额用等量的 pair 排空，
    于是每核恰好发出配额条（与 OT 语义同构，且对配额值本身不敏感——所有核用
    同一个值，恒等性不依赖 host 算得准）。
    """
    with tla.cube():
        tla.cross_core_set_flag(sync.s2_mte2, tla.pipes.MTE2)
        tla.cross_core_wait_flag(sync.s2_mte2, tla.pipes.MTE2)


@tla.jit
def pair_rendezvous(sync):
    """A 步 0b：真·全 AIC 会合（共享暂存的数据依赖必须靠它）。

    实测结论（8 块两向偏差对照）：

    | 形态 | set 管 | wait 管 | 结果 |
    |---|---|---|---|
    | `split_g_mte2_pair`（交付现形态） | MTE2 | MTE2 | **不是会合**：晚到核的
      行还没写，早到核读到 0（4/8 行错，两个偏差方向互为镜像） |
    | 本函数（参考 `ffn/sync_hard.py` 的 SyncAll 形态） | FIX | SCALAR |
      真会合：两向均 8/8 正确 |

    也就是说 `split_g_mte2_pair` 在交付里只起「同刻度节流」作用，
    **不能**给跨块数据依赖兜底；A（共享 v0Res 暂存）必须用这一条。
    形态与 `tla/kernel/matmul/fused_ffn/ffn/sync_hard.py` 的
    `_aic_arrive_handshake` 一致：mode-0 set(FIX) → wait(SCALAR)。
    FFTS 是全核屏障：**所有 block 调用次数必须相等**，仍由 kernel 侧配额排空保证。

    A 步四（已否决，F45-D）：**不能**去掉 `PipeBarrier<ALL>`。微探针
    `--mode 3` 在 8/8 成立给了错误信心：真机 #043 上去掉后 Dur 只 58.9→56.7（−2.2 us），
    但 **bit_exact 1.0 → 0.9657**。原因探针复现不了——AIC 上一轮对同一暂存槽的
    MTE2 读可能仍在飞，与伙伴核重写该槽构成 WAR，只有全管屏障能挡住。
    ⇒ 每条会合的代价不是「屏障指令」而是**会合本身**（等最慢核），形态不可再省。
    """
    with tla.cube():
        tla.pipe_barrier(tla.pipes.ALL)
        tla.cross_core_set_flag(sync.s2_rendezvous, tla.pipes.FIX)
        tla.cross_core_wait_flag(sync.s2_rendezvous, tla.pipes.SCALAR)


# F14 步骤 A：p_release 已删除（p_free 回执去除；OT IterateBmm2Impl 无 C→V
# credit，读完成归还由 AIC 程序序 + MTE1 FIFO + s_ready 链结构性保证）。
