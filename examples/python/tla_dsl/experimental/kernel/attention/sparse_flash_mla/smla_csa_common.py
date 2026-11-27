# ---------------------------------------------------------------------------
# smla_csa_common.py — 常量、缓冲分配、同步对象声明
#
# 结构参照 ops-transformer arch35 sparse_flash_mla:
#   空间：SPACE-DIFF.md（L1 524,288 B / L0A 32,768 / L0B 65,536 / L0C 262,144）
#   同步：MAP.md §同步（CROSSCORE_* 与 crossCoreMte2SyncFlagId）
#   数值：NUMERICS.md R1-R16
# ---------------------------------------------------------------------------

from __future__ import annotations

import builtins
from dataclasses import dataclass

import catlass.tla as tla


def static_range(start, stop=None):
    """前端静态循环序列（@tla.jit 内的 range 会被改写成动态 tla.range）。"""
    if stop is None:
        return tuple(builtins.range(int(start)))
    return tuple(builtins.range(int(start), int(stop)))


# ---------------------------------------------------------------- 全核屏障
# F27 补（#046 间歇性非逐位）：对照 OT csa_kernel_arch35.h:444-449
#   if constexpr (IS_VEC_S2PHYADDR) { AscendC::SyncAll<false>(); }
#   else if (needInit)              { AscendC::SyncAll<false>(); }
# （swa 同款：swa_kernel_arch35.h:344-346，仅 needInit 条件。）
# SyncAll<false> = 全 AIC+AIV 同步。dav_3510 实现在 1:2 mix 下是
#   AIC: wait 两组 AIV flag -> AIC-to-AIC(ffts) -> wait AIC flag -> set 两组 AIV flag
#   AIV: set AIV flag -> wait AIC flag
# （asc/impl/basic_api/dav_3510/kernel_operator_sync_impl.h:160-186）。
# TLA 公开面没有该原语，走 @tla.extern 直达 AscendC。
#
# 必须拆成两个符号（aic / aiv 各一份源码）：hivmc 会把两份 target 专属 bitcode
# 一起链接，而 extern "C" 符号同名时只保留一份定义（实测两份 .o 里都落成 AIC
# 分支 ⇒ AIV 去等只有 AIC 才 set 的 flag，必挂死）。按调用所在 region 决定
# 编译 target（_resolve_extern_targets 只看 usage 的 core_type），AIC 源码只编
# cube、AIV 源码只编 vector，g_coreType 才会各自折叠到正确分支。
_BLOCK_SYNC_AIC_SOURCE = r"""
#include "kernel_operator.h"

extern "C" {
[aicore] __attribute__((always_inline)) void tla_sync_all_cores_aic() {
  AscendC::SyncAll<false>();
}
}
"""

_BLOCK_SYNC_AIV_SOURCE = r"""
#include "kernel_operator.h"

extern "C" {
[aicore] __attribute__((always_inline)) void tla_sync_all_cores_aiv() {
  AscendC::SyncAll<false>();
}
}
"""


@tla.extern(source=_BLOCK_SYNC_AIC_SOURCE, name="tla_sync_all_cores_aic")
def sync_all_cores_aic() -> None: ...


@tla.extern(source=_BLOCK_SYNC_AIV_SOURCE, name="tla_sync_all_cores_aiv")
def sync_all_cores_aiv() -> None: ...


_AIC_EXT_SRC = r"""
#include "kernel_operator.h"

extern "C" {
[aicore] __attribute__((always_inline)) void aic_init_loop3_para() {
  set_loop3_para(1ULL);
  AscendC::PipeBarrier<PIPE_FIX>();
}

[aicore] __attribute__((always_inline)) void aic_fix_l0c_to_ub_uf3(
    uint64_t dst_ub_addr,
    uint64_t src_l0c_addr,
    int32_t n_size,
    int32_t m_size,
    int32_t dst_stride,
    int32_t src_stride) {
  __ubuf__ float* dst = reinterpret_cast<__ubuf__ float*>(dst_ub_addr);
  __cc__ float* src = reinterpret_cast<__cc__ float*>(src_l0c_addr);
  copy_matrix_cc_to_ub(
      dst, src,
      (uint8_t)0, (uint16_t)n_size, (uint16_t)m_size,
      (uint32_t)dst_stride, (uint16_t)src_stride,
      (uint8_t)1, false, (uint8_t)0, (uint8_t)3,
      (uint64_t)0, (uint8_t)0, false, true,
      (uint64_t)0, (uint8_t)0, false, false, (uint8_t)0,
      false, false, false, false, false, false);
}

[aicore] __attribute__((always_inline)) void aic_nd2nz_k4_cm1(
    uint64_t l1_dst_addr, uint64_t gm_src_addr, int32_t n_load) {
  uint64_t para = 1ULL | (1ULL << 16) | (128ULL << 32);
  set_mte2_nz_para(para);
  #pragma unroll
  for (int c = 0; c < 4; ++c) {
    copy_gm_to_cbuf_multi_nd2nz(
        reinterpret_cast<__cbuf__ half*>(l1_dst_addr + static_cast<uint64_t>(c) * 32768ULL),
        reinterpret_cast<__gm__ half*>(gm_src_addr + static_cast<uint64_t>(c) * 256ULL),
        (uint8_t)0, (uint64_t)1024, (uint8_t)1,
        (uint16_t)n_load, (uint16_t)128, (uint64_t)0, false);
  }
}
}
"""


@tla.extern(source=_AIC_EXT_SRC, name="aic_init_loop3_para")
def aic_init_loop3_para() -> None: ...


@tla.extern(source=_AIC_EXT_SRC, name="aic_fix_l0c_to_ub_uf3")
def aic_fix_l0c_to_ub_uf3(
    dst: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    src: tla.Pointer[tla.Float32, tla.AddressSpace.l0c],
    n_size: tla.Int32,
    m_size: tla.Int32,
    dst_stride: tla.Int32,
    src_stride: tla.Int32,
) -> None: ...


@tla.extern(source=_AIC_EXT_SRC, name="aic_nd2nz_k4_cm1")
def aic_nd2nz_k4_cm1(
    dst: tla.Pointer[tla.UInt8, tla.AddressSpace.l1],
    src: tla.Pointer[tla.UInt8, tla.AddressSpace.gm],
    n_load: tla.Int32,
) -> None: ...


# ---------------------------------------------------------------- 固定 tile
HEAD_DIM = 512  # dSize
Q_ROWS = 64  # Cube 的 M tile 上限（tiling.h:536 mBaseSize_）
KV_BLOCK = 128  # s2BaseSize（csa_kernel_arch35.h:160）
D_CHUNK = 128  # dBaseMatmulSize，QK 的 K 步长
NUM_D_CHUNKS = 4  # 2 x 256，每次 2 x 128
V_SEGS = 4  # PV 沿 N（dSize）拆 4 段
Q_HALF = 256  # dSize >> 1（qHalfNum=2）
KV_SLOTS = 3  # v0Res 3 槽环
P_SLOTS = 2  # l1P[2]
L0_SLOTS = 2

# ---- 步 B：L0A/L0B 4 槽 + D=64 tiling（竞品 N64 形态）----
# 竞品参数：L0A ring 4、L0B slots 4、D_F32_TILES=8（512/64）。
#   QK：8 个 D=64 chunk，L0A=(m,64) 8KB、L0B=(64,128) 16KB → 4 槽恰满
#       L0A 32KB / L0B 64KB（与原来 2×(16KB/32KB) 占用相同）。
#   PV：K(=s2=128) 拆 2 个 64 chunk；P=(m,64)、V=(64,128)，P 两片常驻
#       l0a0/l0a1 供 4 个 N 段复用，V 按 (seg*2+kc)%4 轮转 4 槽。
# 开关默认 False：步 A（ori AIV->L1）先在唯一基准上验证；置 True 启用步 B。
L0_D64 = False
STATE_SLOT_STRIDE = 64  # 64 fp32 = 256 B (1 VL, 256B aligned per state slot)
L0_AB_SLOTS = 4 if L0_D64 else 2
L0_D = 64 if L0_D64 else D_CHUNK
L0_D_CHUNKS = HEAD_DIM // L0_D  # 8（D=64）/ 4（D=128）

# R42：AIV 直写 L1 的 UB zN 暂存行距（竞品 kv_out_ub 的 17 行 pitch：16 行数据
# + 1 行 padding，见 CSA_KERNEL_SPEC.md §2）。只影响 UB 内部布局；L1 目的端仍是
# 标准 zN(HEAD_DIM, KV_BLOCK)（nZ 视图与之一一对应）。
KV_NZ_PITCH = 17

KV_UNIT = 16  # OT KV_PROCESS_UNIT：每次 UB->GM 拷出 16 行
KV_COPY_IN = 8  # OT KV_COPYIN_UNIT：每次 GM->UB 搬入 8 行
KV_STAGE_SLOTS = KV_SLOTS  # GM 暂存槽与 L1 K 环一一对应

# T1-1（r10）：非 PA 索引行 UB 环。OT 侧 sparse_indices_ub_ring = depth2 x 64 i32
# （N:448-455，256 B/槽）；本仓每 AIV 的一个 tile 最多读 64 行索引，但 AIV 半区
# 起点 my_start=(n_rows+1)//2 不保证 8 元素对齐（尾块），故 DMA 从 align_down(
# 全局列号, 8) 起搬，槽内多 7 个前导元素；实测覆盖上界 = rem(≤7) + cover(≤64)
# = 71 ⇒ 槽宽 72 i32（288 B，32 B 对齐），2 槽 576 B（UB 余量 12,384 B 内）。
IDX_RING_SLOTS = 2
IDX_RING_STRIDE = 72

MIN_F = -3.4028234663852886e38  # NEGATIVE_MIN_VALUE_FP32 = 0xFF7FFFFF

# F4：InitOutput 每次 DataCopy 的元素数。
# OT（init_output.h:28-96 的调用点 csa_block_vector_arch35.h:1310-1348）用
# ATTEN_OUT_POP_BUF_ELE_SIZE = 32KiB / sizeof(OUTPUT_T)（2 字节 dtype 时 16384）。
# 本仓取 512：TLA 内拿不到 dtype 的字节宽做「32KiB/sizeof(T)」的静态分块，
# 换成可枚举的固定粒度后循环次数变多，但 Duplicate 一次 + DataCopy 循环 + 尾块
# DataCopyPad 的结构与逐位结果不变。
INIT_POP_ELE = 512


@dataclass
class Sync:
    # 跨核 flag（mode=4，CROSSCORE_* 的 DSL 等价物）
    s_ready: object  # FIX -> VECTOR : QK 结果落到 UB
    s_free: object  # VECTOR -> FIX : UB S 已消费，可写下个 tile
    p_ready: object  # MTE3 -> MTE1 (AIC) : P 已写入 L1
    o_ready: object  # FIX -> VECTOR : PV 段落到 UB
    o_free: object  # VECTOR -> FIX : UB O tmp 已消费
    kv_stage_ready: object  # AIV MTE3 -> AIC MTE2 : 稀疏 gather 已写入 GM 暂存槽
    # 核内 flag
    # 闸门② Q 3 槽环：ready/free 保持单 flag（事件 id 上限 0..7，dst=MTE2 八个
    # 位已被占满，无法按槽各开一对），信用化——q_l0a_free init 预置 3 信用、
    # load_q wait x2、bmm1 末 tile set x2，每任务收发各 2 平衡；槽隔离由寻址
    # （3 槽连续基址 + 游标）与程序序保证（见 cube.load_q / iterate_bmm1 注）。
    q_l1_ready: object  # MTE2 -> MTE1 (slot 0)
    q_l0a_free: object  # MTE1 -> MTE2 (slot 0)
    q_l1_ready1: object  # MTE2 -> MTE1 (slot 1, opt_q_pre3)
    q_l1_ready2: object  # MTE2 -> MTE1 (slot 2, opt_q_pre3)
    q_l0a_free1: object  # MTE1 -> MTE2 (slot 1, opt_q_pre3)
    q_l0a_free2: object  # MTE1 -> MTE2 (slot 2, opt_q_pre3)
    k_l1_ready0: object
    k_l1_ready1: object
    k_l1_ready2: object
    k_l0b_free0: object
    k_l0b_free1: object
    k_l0b_free2: object
    s_mm_ready: object  # CUBE -> FIX
    s_mm_ready1: (
        object  # CUBE -> FIX（F10 步骤 1：S L0C 槽 1，仅 swa preload 实例发出）
    )
    s_fix_done: object  # FIX -> CUBE
    s_fix_done1: (
        object  # FIX -> CUBE（F10 步骤 1：S L0C 槽 1，仅 swa preload 实例发出）
    )
    o_mm_ready0: object
    o_mm_ready1: object
    o_fix_done0: object
    o_fix_done1: object
    p_data_ready: object  # VECTOR -> MTE3 (UB P 落盘)
    lse_v_mte3: object  # VECTOR -> MTE3 (LSE 落盘，OT: INNERCORE_LSE_V_MTE3)
    lse_mte3_v: object  # MTE3 -> VECTOR (R37：LSE 落盘 WAR，dm 槽复用前等搬完)
    # AIV 内 gather（OT: INNERCORE_STAGE0OUT_*，ping-pong 各一套 flag id）
    kv_mte2_mte3_0: object
    kv_mte2_mte3_1: object
    kv_mte3_mte2_0: object
    kv_mte3_mte2_1: object
    # R42（need_kv_ring 实例；其余实例为 None，不产生 flag op）：cmp 段 AIV 直写
    # L1 槽的 band 级 / tile 级同步。
    #   kv_a_loaded : MTE2 -> V   ，ND band 已填好
    #   kvz_v2m3    : V -> MTE3   ，zN band 已写完
    #   kvz_free    : MTE3 -> V   ，zN 已被 L1 copy 读完（预置 1）
    #   kv_nd_free  : V -> MTE2   ，ND 已被 VF 读完（预置 1）
    #   kv_free     : cross AIC MTE1 -> AIV MTE3，cmp tile 的 L1 槽可写
    #                 （"后继条件"信用：AIC 只在后继 tile 是 cmp 时归还，AIV
    #                  也只在 cmp 且 x>=3 时消费；不预置，x<3 的槽本核未用过）
    kv_a_loaded: object
    kvz_v2m3: object
    kvz_free: object
    kvz_v2m3_1: object
    kvz_free1: object
    kv_nd_free: object
    kv_free: object
    # E17（need_pa_l1 实例；其余 None）：PA cmp 段 AIV gather -> extern 换形 -> L1。
    # 与 R42 的区别是 ND->zN 由 VECTOR 的 extern VSSTB 完成（不再是 MTE3 硬件
    # ND->NZ），因此 ND 的"填好/读完"以 VECTOR 为对端，2 块 ND ping-pong 各一套。
    #   kv_nd_v{0,1} : MTE2 -> VECTOR，ND band 已填好
    #   kv_v_m2{0,1} : VECTOR -> MTE2，ND band 已被 extern 读完（预置 1）
    # zN 的 kvz_v2m3 / kvz_free 与 R42 复用（V->MTE3 / MTE3->V）。
    kv_nd_v0: object
    kv_nd_v1: object
    kv_v_m20: object
    kv_v_m21: object
    # 输出清零（OT: INNERCORE_INITOUT_MTE3_V = 0，MTE3_V / V_MTE3 同 id 双向）
    init_mte3_v: object
    init_v_mte3: object
    # F2 phy-addr 表生成（OT INNERCORE_PHYADDR_* 的本仓对应，均为核内 set+wait 屏障对；
    # OT 靠 Init 阶段结束后的 SyncAll 一次性覆盖下面全部跨管道顺序，本仓逐 token 化简
    # 没有 SyncAll，TLA 又不做自动管道同步，故逐条显式配对）：
    #   phy_m2v   : MTE2 -> VECTOR，页表行拷入完成后 VF 才能读 ub_phy_bt（OT SPARSEIDX_READY）
    #   phy_s2v   : SCALAR -> VECTOR，copy_sparse_idx_to_ub 的标量 UB idx 写完成后
    #               VF 才能读 ub_phy_idx（SCALAR_STORE -> VEC_LOAD）
    #   phy_v_mte3: VECTOR -> MTE3，VF 写完 ub_phy_addr 后 MTE3 才能拷出（OT KVADDR_READY）
    #   phy_m3v   : MTE3 -> VECTOR，MTE3 拷出完成后下一行才能复用 ub_phy_addr（OT KVADDR_FREE）
    #   phy_m3s   : MTE3 -> SCALAR，表行拷出（GM 写）完成后标量流才能读表——
    #               gather_kv_rows_vec 的 row_off 是 S 流 GM 读，缺这对屏障会读到旧表值，
    #               以垃圾地址发起 MTE2 KV 行拷贝（errcode 95，Iter20 smoke2/3 根因）
    #   phy_vs    : VECTOR -> SCALAR 信用（init_sync 在 =1 实例预置 1 个）：VF 消费完
    #               ub_phy_idx/ub_phy_bt 后下一轮生成才能写；MTE2 页表拷入与标量 idx 写
    #               都由 S 流发起，wait 放在生成入口即同时覆盖两个缓冲的跨 task 复用
    phy_m2v: object
    phy_s2v: object
    phy_v_mte3: object
    phy_m3v: object
    phy_m3s: object
    phy_vs: object
    # T1-1（r10，非 PA 去表）：索引行 DMA -> UB 2 槽环的事件对，只在
    # need_idx_ring=True 的实例创建（其余实例为 None ⇒ 不产生 flag op，
    # 交付 IR 与改前逐条相同；F44-A6 已证「未用的 flag 也会改变追踪结果」）。
    #   idx_m2s{0,1}: MTE2 -> SCALAR，槽 DMA 完成 -> 标量读放行（ready）。
    #                 标量读 MTE2 写的 UB 必须用 SCALAR 目的管（记忆
    #                 tla-scalar-ub-read-needs-scalar-flag；MTE2->VECTOR 会随机读 0）。
    #   idx_s2m{0,1}: SCALAR -> MTE2，本槽标量读完成 -> 下一轮写同槽放行（WAR）。
    # id 口径（硬件按 (目的管,id) 匹配，见 declare_sync 注释）：
    #   dst=SCALAR 已被 phy_m3s(0)/phy_vs(1) 占用 ⇒ m2s 取 2/3；
    #   dst=MTE2 csa 实例实际只用到 0..5（q_l0a 0、k_l0b 1-3、kv_m3m2 4-5），
    #   6/7 属于仅 bc=1 实例才 set/wait 的 bc_m3m2/bc_v2m ⇒ 生命周期不重叠，
    #   按 tla.flag 文档「显式 id 可复用，调用方保证生命周期不重叠」取 6/7。
    idx_m2s0: object
    idx_m2s1: object
    idx_s2m0: object
    idx_s2m1: object
    # F3 batch-consistency（OT INNERCORE_INTRALSE_MTE3_MTE2 / StageVec2PartialOAndWait
    # / MergeStagedAndCurrentChunk 的 V_MTE2 信用对的本仓对应；仅
    # is_batch_consistency=1 实例使用，=0 实例不发出任何一条）：
    #   bc_m3m2 : MTE3 -> MTE2，stage 拷出（O/max/sum 落 GM 槽）完成后，下一个 tile
    #             的 merge 才能发起 GM 读——生产者=本 AIV MTE3，消费者=本 AIV MTE2。
    #             无预置信用：首个 wait（tile 1）严格在首个 set（tile 0 stage）之后
    #             （OT INTRALSE/INTRAATTN 同款链）。
    #   bc_m3v  : MTE3 -> VECTOR，stage 拷出完成后 V 才继续（OT StageVec2PartialO
    #             AndWait 的 MTE3_V：「wait for MTE3 completion before the source
    #             UB can be reused」——防下一 tile 的 vec2 覆写 ub_o）。
    #   bc_m2v  : MTE2 -> VECTOR，prev max/sum/O 段拷入完成后 V 才读
    #             （OT MergeStagedAndCurrentChunk 的 mte2ToVId SetFlag/WaitFlag 对）。
    #   bc_v2m  : VECTOR -> MTE2，V 读完 ub_bc_prev（merged 完成）后 MTE2 才能覆写；
    #             信用模式，=1 实例 init_sync 预置 1 个（OT「Both V_MTE2 events
    #             must be set before the first call」的 vToMte2Id0/1 同款）。
    #   bc_v3m  : VECTOR -> MTE3，V 写完 ub_o/ub_max/ub_sum 后 MTE3 才能拷出
    #             （process_vec1:413 p_data_ready 的 set+wait 屏障对同款）。
    bc_m3m2: object
    bc_m3v: object
    bc_m2v: object
    bc_v2m: object
    bc_v3m: object
    # F6 流水版（仅 swa kernel preload 实例发出）：l1P 槽的读完成归还
    # （AIC MTE1 set -> AIV MTE3 wait，信用制，init 预置 2/AIV，见 init_sync）
    p_free: object
    # 核内 mutex
    m_l0a0: object
    m_l0a1: object
    m_l0b0: object
    m_l0b1: object
    m_o16: object
    m_p: object
    m_sink: object
    m_l0c_o0: object
    m_l0c_o1: object
    # F10 步骤 2：L0A/L0B AB 槽 flag 对（OT static_matmul.h 的 MTE1_M/M_MTE1 同款，
    # L0A/L0B 合并单 bufferId、成对加载成对消费）。ab_ready{i}：MTE1 搬完通知 CUBE；
    # ab_free{i}：CUBE mmad 用完释放给 MTE1。bmm1 段与 bmm2 段按 slot=段%2 共享同一
    # 对（OT RingBuffer curId 连续轮转同款）。init 预置 ab_free0/1 各 1（OT
    # cube_local_buffer.h:70-71 的 SetFlag<M_MTE1>(0)/(1)）。
    ab_ready0: object
    ab_ready1: object
    ab_free0: object
    ab_free1: object
    # 步 B（L0_D64）：4 槽 AB flag 对的槽 2/3。dst=CUBE 预算 0..5 已用，
    # 槽 2/3 取 6/7；dst=MTE1 同为 6/7。非 L0_D64 实例为 None（不发出）。
    ab_ready2: object
    ab_ready3: object
    ab_free2: object
    ab_free3: object
    ab_p_ready: object
    ab_p_free: object
    # F27：IS_SPLIT_G 的核间 MTE2 屏障（OT csa_kernel_arch35.h:347-348/:357-358 的
    # CrossCoreSetFlag<0,PIPE_MTE2>(15) / CrossCoreWaitFlag<0,PIPE_MTE2>(15)）。
    # mode=0 = FFTS 全核屏障（AscendC SyncAll<false> 语义：每个核 set 自己的到达位、
    # wait 等全部核到达；参考实现 tla/kernel/matmul/fused_ffn/ffn/sync_hard.py 的
    # ``cross_core_set_flag(aic_ffts, FIX)`` + ``cross_core_wait_flag(aic_ffts, SCALAR)``
    # 同样是单侧（AIC）set+wait、不需要 AIV 参与）。只有 g_split==2（split-G 指令
    # 形态）实例发出；各 AIC 发出条数由"预算排空"保证恒等（见 kernel 注）。
    s2_mte2: object
    # A 步 0b：真会合用的 mode-0 flag（set 挂 FIX / wait 挂 SCALAR）。
    # 与 s2_mte2 同 mode，但 s2_mte2 的 set/wait 都在 MTE2 上 —— 探针实测那**不是**
    # 会合（见 smla_csa_block_cube.pair_rendezvous）。
    s2_rendezvous: object
    # CSA-only optimization flags (default False so SWA/HCA/OS/OCS stay untouched)
    opt_cube64: bool = False
    opt_cube128: bool = False
    opt_l2m: bool = False
    opt_q_pre3: bool = False
    q_l2_bypass: bool = False


@dataclass
class Bufs:
    l1_p: object  # F6 流水版（swa preload 实例）为连续双槽基址（OT l1PBuffers[2]）；
    # 其余实例单槽。槽 1 = ptr + Q_ROWS*KV_BLOCK 元素。
    l1_q: object  # 闸门②：连续 3 槽基址（OT l1Q l1QBufNum=3 同款），
    # 槽 i = ptr + i*(Q_ROWS*Q_HALF) 元素；字节表不变
    # （3x32,768 B，每槽起点 32,768 B 整数倍保持 512 对齐）。
    l1_k: object  # l1K[3] 连续基址：3 x 131,072 B，slot 由偏移寻址
    l0a0: object
    l0a1: object
    l0b0: object
    l0b1: object
    # 步 B（L0_D64）：第 3/4 槽（L0A (Q_ROWS,64) 8KB、L0B (64,KV_BLOCK) 16KB）。
    # 非 L0_D64 实例为 None（不分配）。
    l0a2: object
    l0a3: object
    l0a_p: object
    l0b2: object
    l0b3: object
    l0c_s: object
    l0c_o: object  # F17 步骤 B：O 单槽 128 KB（Q_ROWS*HEAD_DIM fp32），
    # 4 个 N 窗口 mmad 各占 seg*(m*D_CHUNK) 线性段；
    # 旧 l0c_o0/o1 双槽（各 32 KB）删除。
    ub_s: object  # F6 流水版（swa preload 实例）为连续双槽基址（OT bmm1Buffers[2]，
    # 槽 stride = Q_ROWS//2*KV_BLOCK）；其余实例单槽。
    ub_p: object
    ub_otmp_full: object  # F17 步骤 B：O 整块中转（half*HEAD_DIM fp32 = 65,536 B），
    # 旧 ub_otmp0/1 双槽（各 half*D_CHUNK = 16 KB）删除。
    ub_o: object
    ub_o16: object
    ub_sink: object
    ub_max: object
    ub_sum: object
    ub_dm: object
    ub_kv0: object  # 稀疏 gather 的 UB 中转（KV_UNIT 行 x HEAD_DIM，ping-pong）
    ub_kv1: object
    # R42（kv_ring=True 实例）：AIV 直写 L1 前的 UB zN 暂存（KV_NZ_PITCH 行 pitch，
    # 17 x 512 dtype = 17,408 B）；其余实例为 None（不产生 allocate op）。
    ub_kvz: object
    ub_kvz1: object
    ub_init: object  # F4 输出清零的 DUPLICATE 中转（OT: ATTEN_OUT_POP_BUF）
    ub_init_lse: object  # F4 LSE 清零的 DUPLICATE 中转（OT: LSE_POP_BUF，fp32）
    # F2 phy-addr 表生成的 UB 暂存（OT: sparseIdxUb / blkTableUb / kvPhyAddrUb）
    ub_phy_idx: object  # sparse indices 行（Int32，alignedSparseBlockCount 上界 512）
    ub_phy_bt: object  # block table 行（Int32，maxBlockNumPerBatch 上界 256）
    ub_phy_addr: object  # 物理地址（Int32 视图；aligned*2 个 u32 = aligned 个 int64）
    # T1-1（r10）索引行 UB 环：2 槽 x IDX_RING_STRIDE 个 Int32。仅
    # alloc_buffers(idx_ring=True) 实例分配真实容量（576 B/AIV），其余实例为
    # None（不产生 allocate op ⇒ UB 字节表与改前逐字节一致）。环独立于 ws
    # 暂存结构，T1-0a 摘除 ws 时不受牵连。
    ub_idx_ring: object
    # F3 batch-consistency 的 UB 暂存（仅 is_batch_consistency=1 实例使用）：
    ub_bc_prev: object  # staged prevO 列段（rows x D_CHUNK fp32，OT partialOTmpUb 段）
    ub_bc_max: object  # staged prev lse（align8(rows) fp32，OT blockMaxUb slot0）
    ub_bc_sum: object  # staged prev sum（align8(rows) fp32，OT blockSumUb slot0）
    ub_bc_w: object  # 合并权重 [w_prev | w_cur]（2*align8(rows) fp32）
    # E4（FlashDecoding KV 分片）merge 暂存：[(2*SPLITS+2)*rows8] fp32。
    # 仅 fd_splits>1 实例分配；=1 实例为 None（不产生 allocate op ⇒ UB 字节表
    # 与改前逐字节一致）。
    ub_fd_stats: object
    # CSA-only optimization flags (default False so SWA/HCA/OS/OCS stay untouched)
    opt_cube64: bool = False
    opt_vec: bool = False


@tla.jit
def compute_need_init(cu_q, cu_kv, num_seqs, ori_mask_mode, ori_template):
    """needInit（csa_kernel_arch35.h:293-319；swa_kernel_arch35.h:241-266 同构）。

    OT 原式：
        needInit = 0;
        if (mode ∉ {ORI_SPARSE, ORI_CMP_SPARSE} && oriMaskMode != 0) {
            for b: s2Size = (PA ? seqused_ori_kv[b] : GetSmlaSeqLen(...));
                   s1Size = GetSmlaSeqLen(...);
                   expectQs = (TND ? cu_q 差分 : s1Size);
                   if (s1Size > s2Size || s1Size < expectQs) { needInit = 1; break; }
        } else { needInit = 1; }

    本仓输入只有 cu_seqlens：workload 里 seqused_* 与 cu 差分逐个相等，
    故 s2Size/s1Size 取 cu 差分；TND 下 expectQs == s1Size，于是
    「s1Size < expectQs」恒不成立，「s1Size > s2Size」保留。
    ori_template=1 对应 ORI_SPARSE/ORI_CMP_SPARSE（走 else，恒 1）。
    """
    need_init = tla.as_numeric(0)
    if ori_template == 1:
        need_init = tla.as_numeric(1)
    else:
        if ori_mask_mode == 0:
            need_init = tla.as_numeric(1)
        else:
            for b in static_range(num_seqs):
                s1 = cu_q[b + 1] - cu_q[b]
                s2 = cu_kv[b + 1] - cu_kv[b]
                if s1 > s2:
                    need_init = tla.as_numeric(1)
    return need_init


@tla.jit
def declare_sync(
    need_rendezvous: bool = False,
    need_idx_ring: bool = False,
    need_lse_war: bool = False,
    need_kv_ring: bool = False,
    need_pa_l1: bool = False,
    need_kvz_db: bool = False,
    opt_cube64: bool = False,
    opt_cube128: bool = False,
    opt_l2m: bool = False,
    q_l2_bypass: bool = False,
    q_pre3_hint: bool = False,
) -> Sync:
    """声明同步对象。

    need_rendezvous：共享暂存实例的真会合 flag（见下）。
    need_idx_ring（T1-1 r10）：非 PA 去表实例的索引环事件对。默认 False ⇒
    不创建任何新 flag（F44-A6：未用的 flag 也会改变追踪结果）。PA 控制臂因此
    与改前逐字节一致。
    need_kv_ring（R42）：cmp 段 AIV 直写 L1 槽的实例。GM 中转的 kv_m2m3/kv_m3m2
    两族在此实例里为 None（腾出 dst=MTE3 0/1、dst=MTE2 4/5），另建 band 级
    4 条 intra + 1 条 cross；k_l1_ready/k_l0b_free 两族**保留**（ori 仍由 AIC
    直读，需要 MTE2->MTE1 / MTE1->MTE2 的原生握手）。
    """

    def cross(name, mode=4):
        return tla.cross_flag(name, mode=mode)

    return Sync(
        s_ready=cross("smla_s_ready"),
        s_free=cross("smla_s_free"),
        p_ready=cross("smla_p_ready"),
        o_ready=cross("smla_o_ready"),
        o_free=cross("smla_o_free"),
        kv_stage_ready=cross("smla_kv_stage_ready"),
        # A 步 0b：真·全 AIC 会合（mode-0 FFTS，set 挂 FIX / wait 挂 SCALAR）。
        # 与 s2_mte2 的区别见 smla_csa_block_cube.pair_rendezvous 的注释：
        # s2_mte2 的 set/wait 都挂 MTE2，探针实测**不是会合**。
        # 只有共享暂存实例才声明：未用的 flag 也会改变追踪结果
        # （F44-A6 实测交付 key 由 b476e6b9 变为 854fb62c），故按需创建。
        s2_rendezvous=(
            cross("smla_s2_rendezvous", mode=0) if need_rendezvous else None
        ),
        # F3 竞态修复：DSL 对 event id 按 (src,dst) 管道对各自分配（每对从 0 递增），
        # 但硬件事件位按 (目的管, id) 匹配 ⇒ 跨管道对共享同一 (dst,id) 时互相释放
        # （实证：bc=1 实例最终 IR 中 bc_m2v(MTE2→V) 与 init_mte3_v(MTE3→V) 同为
        # EVENT_ID0，merge 的 wait 被 init 的 set 提前放行，MTE2 拷贝未落地）。
        # 全部 33 条 flag 显式指定 id，同一目的管内唯一：
        #   dst=MTE1: q_l1_ready 0, k_l1_ready 1-3 / dst=MTE2: q_l0a 0, k_l0b
        #             1-3, kv_m3m2 4-5, bc_m3m2 6, bc_v2m 7 / dst=MTE3: kv_m2m3
        #             0-1, p_data 2,
        #             lse 3, init_v_mte3 4, phy_v_mte3 5, bc_v3m 6, p_free 7
        #             （F6 流水版）/
        #   dst=V: init_mte3_v 0, phy_m2v 1, phy_s2v 2, bc_m2v 3, phy_m3v 4,
        #          bc_m3v 5 / dst=CUBE: s_fix_done 0-1（F10 步骤 1 S 槽环）,
        #          o_fix_done 1-2 /
        #   dst=FIX: s_mm_ready 0-1（F10 步骤 1 S 槽环）, o_mm_ready 1-2 /
        #   dst=S: phy_m3s 0, phy_vs 1 /
        #   dst=CUBE 补充（F10 步骤 2）：ab_ready 4-5（AB 槽 MTE1→CUBE）/
        #   dst=MTE1 补充（F10 步骤 2）：ab_free 4-5（AB 槽 CUBE→MTE1）
        q_l1_ready=tla.flag("smla_q_l1_ready", tla.pipes.MTE2, tla.pipes.MTE1, id=0),
        q_l0a_free=tla.flag("smla_q_l0a_free", tla.pipes.MTE1, tla.pipes.MTE2, id=0),
        q_l1_ready1=(
            tla.flag("smla_q_l1_ready1", tla.pipes.MTE2, tla.pipes.MTE1, id=4)
            if (opt_cube64 and q_pre3_hint)
            else None
        ),
        q_l1_ready2=(
            tla.flag("smla_q_l1_ready2", tla.pipes.MTE2, tla.pipes.MTE1, id=5)
            if (opt_cube64 and q_pre3_hint)
            else None
        ),
        q_l0a_free1=(
            tla.flag("smla_q_l0a_free1", tla.pipes.MTE1, tla.pipes.MTE2, id=4)
            if (opt_cube64 and q_pre3_hint)
            else None
        ),
        q_l0a_free2=(
            tla.flag("smla_q_l0a_free2", tla.pipes.MTE1, tla.pipes.MTE2, id=5)
            if (opt_cube64 and q_pre3_hint)
            else None
        ),
        k_l1_ready0=tla.flag("smla_k_l1_ready0", tla.pipes.MTE2, tla.pipes.MTE1, id=1),
        k_l1_ready1=tla.flag("smla_k_l1_ready1", tla.pipes.MTE2, tla.pipes.MTE1, id=2),
        k_l1_ready2=tla.flag("smla_k_l1_ready2", tla.pipes.MTE2, tla.pipes.MTE1, id=3),
        k_l0b_free0=tla.flag("smla_k_l0b_free0", tla.pipes.MTE1, tla.pipes.MTE2, id=1),
        k_l0b_free1=tla.flag("smla_k_l0b_free1", tla.pipes.MTE1, tla.pipes.MTE2, id=2),
        k_l0b_free2=tla.flag("smla_k_l0b_free2", tla.pipes.MTE1, tla.pipes.MTE2, id=3),
        s_mm_ready=tla.flag("smla_s_mm_ready", tla.pipes.CUBE, tla.pipes.FIX, id=0),
        s_mm_ready1=tla.flag("smla_s_mm_ready1", tla.pipes.CUBE, tla.pipes.FIX, id=3),
        s_fix_done=tla.flag("smla_s_fix_done", tla.pipes.FIX, tla.pipes.CUBE, id=0),
        s_fix_done1=tla.flag("smla_s_fix_done1", tla.pipes.FIX, tla.pipes.CUBE, id=3),
        o_mm_ready0=tla.flag("smla_o_mm_ready0", tla.pipes.CUBE, tla.pipes.FIX, id=1),
        o_mm_ready1=tla.flag("smla_o_mm_ready1", tla.pipes.CUBE, tla.pipes.FIX, id=2),
        o_fix_done0=tla.flag("smla_o_fix_done0", tla.pipes.FIX, tla.pipes.CUBE, id=1),
        o_fix_done1=tla.flag("smla_o_fix_done1", tla.pipes.FIX, tla.pipes.CUBE, id=2),
        p_data_ready=tla.flag(
            "smla_p_data_ready", tla.pipes.VECTOR, tla.pipes.MTE3, id=2
        ),
        lse_v_mte3=tla.flag("smla_lse_v_mte3", tla.pipes.VECTOR, tla.pipes.MTE3, id=3),
        lse_mte3_v=(
            tla.flag("smla_lse_mte3_v", tla.pipes.MTE3, tla.pipes.VECTOR, id=6)
            if need_lse_war
            else None
        ),
        # R42：need_kv_ring 实例不再经 GM 中转，m2m3/m3m2 两族不声明
        # （dst=MTE3 的 0/1、dst=MTE2 的 4/5 随之空出，见下面的新 flag）。
        kv_mte2_mte3_0=(
            tla.flag("smla_kv_m2m3_0", tla.pipes.MTE2, tla.pipes.MTE3, id=0)
            if not need_kv_ring
            else None
        ),
        kv_mte2_mte3_1=(
            tla.flag("smla_kv_m2m3_1", tla.pipes.MTE2, tla.pipes.MTE3, id=1)
            if not need_kv_ring
            else None
        ),
        kv_mte3_mte2_0=(
            tla.flag("smla_kv_m3m2_0", tla.pipes.MTE3, tla.pipes.MTE2, id=4)
            if not need_kv_ring
            else None
        ),
        kv_mte3_mte2_1=(
            tla.flag("smla_kv_m3m2_1", tla.pipes.MTE3, tla.pipes.MTE2, id=5)
            if not need_kv_ring
            else None
        ),
        # R42 band 级 4 条 intra：dst=V 取 1（MTE2->V）与 2（MTE3->V）——CSA
        # ring 实例不使用 phy_m2v(1)/phy_s2v(2)（索引走环、无物理表），也未启用
        # bc（bc_m2v 3 / bc_m3v 5 仅 swa kernel set/wait）；dst=MTE3 取 0（m2m3
        # 已腾出）；dst=MTE2 取 4（m3m2 已腾出）。dst=V 的 6 留给 lse_mte3_v
        # （#026 enable_lse=1），不得占用。
        # R42+步 A 修订：ND->NZ 改由 MTE3 的 ("ub","RowMajor","l1","zN") 硬件
        # 转换完成（软件 VF 在 022 上单管 275us，是 Dur 回退的根因）。
        #   kv_a_loaded : MTE2 -> MTE3 ，ND band 已填好（MTE3 等它再 copy）
        #   kv_nd_free  : MTE3 -> MTE2 ，MTE3 已读完 ND band（MTE2 可预取）
        # kvz_* 两条保留声明但不再 set/wait（zN 暂存退役）。
        kv_a_loaded=(
            tla.flag("smla_kv_a_loaded", tla.pipes.MTE2, tla.pipes.MTE3, id=1)
            if need_kv_ring
            else None
        ),
        kvz_v2m3=(
            tla.flag("smla_kvz_v2m3", tla.pipes.VECTOR, tla.pipes.MTE3, id=0)
            if (need_kv_ring or need_pa_l1)
            else None
        ),
        kvz_free=(
            tla.flag("smla_kvz_free", tla.pipes.MTE3, tla.pipes.VECTOR, id=2)
            if (need_kv_ring or need_pa_l1)
            else None
        ),
        kvz_v2m3_1=(
            tla.flag("smla_kvz_v2m3_1", tla.pipes.VECTOR, tla.pipes.MTE3, id=1)
            if need_kvz_db
            else None
        ),
        kvz_free1=(
            tla.flag("smla_kvz_free1", tla.pipes.MTE3, tla.pipes.VECTOR, id=4)
            if need_kvz_db
            else None
        ),
        # E17：PA 直写 L1 的 ND band 级 intra（2 块 ping-pong）。id 选取见
        # RUNLOG §3：dst=VECTOR 的 7 空闲、3 由 bc_m2v 占位但 PA CSA 从不 set/wait；
        # dst=MTE2 的 4/5 在 need_kv_ring(或本实例)下 kv_m3m2 为 None，空出。
        kv_nd_v0=(
            tla.flag("smla_kv_nd_v0", tla.pipes.MTE2, tla.pipes.VECTOR, id=7)
            if need_pa_l1
            else None
        ),
        kv_nd_v1=(
            tla.flag("smla_kv_nd_v1", tla.pipes.MTE2, tla.pipes.VECTOR, id=3)
            if need_pa_l1
            else None
        ),
        kv_v_m20=(
            tla.flag("smla_kv_v_m2_0", tla.pipes.VECTOR, tla.pipes.MTE2, id=4)
            if need_pa_l1
            else None
        ),
        kv_v_m21=(
            tla.flag("smla_kv_v_m2_1", tla.pipes.VECTOR, tla.pipes.MTE2, id=5)
            if need_pa_l1
            else None
        ),
        kv_nd_free=(
            tla.flag("smla_kv_nd_free", tla.pipes.MTE3, tla.pipes.MTE2, id=4)
            if need_kv_ring
            else None
        ),
        # cross：AIC MTE1 -> AIV MTE3（L1 槽 WAR 的归还，每 tile 1 token/AIV）
        kv_free=(cross("smla_kv_free") if need_kv_ring else None),
        init_mte3_v=tla.flag(
            "smla_init_mte3_v", tla.pipes.MTE3, tla.pipes.VECTOR, id=0
        ),
        init_v_mte3=tla.flag(
            "smla_init_v_mte3", tla.pipes.VECTOR, tla.pipes.MTE3, id=4
        ),
        phy_m2v=tla.flag("smla_phy_m2v", tla.pipes.MTE2, tla.pipes.VECTOR, id=1),
        phy_s2v=tla.flag("smla_phy_s2v", tla.pipes.SCALAR, tla.pipes.VECTOR, id=2),
        phy_v_mte3=tla.flag("smla_phy_v_mte3", tla.pipes.VECTOR, tla.pipes.MTE3, id=5),
        phy_m3v=tla.flag("smla_phy_m3v", tla.pipes.MTE3, tla.pipes.VECTOR, id=4),
        phy_m3s=tla.flag("smla_phy_m3s", tla.pipes.MTE3, tla.pipes.SCALAR, id=0),
        phy_vs=tla.flag("smla_phy_vs", tla.pipes.VECTOR, tla.pipes.SCALAR, id=1),
        # T1-1（r10）：索引环事件对，按 need_idx_ring 按需创建（见 Sync 注释）。
        idx_m2s0=(
            tla.flag("smla_idx_m2s0", tla.pipes.MTE2, tla.pipes.SCALAR, id=2)
            if need_idx_ring
            else None
        ),
        idx_m2s1=(
            tla.flag("smla_idx_m2s1", tla.pipes.MTE2, tla.pipes.SCALAR, id=3)
            if need_idx_ring
            else None
        ),
        idx_s2m0=(
            tla.flag("smla_idx_s2m0", tla.pipes.SCALAR, tla.pipes.MTE2, id=6)
            if need_idx_ring
            else None
        ),
        idx_s2m1=(
            tla.flag("smla_idx_s2m1", tla.pipes.SCALAR, tla.pipes.MTE2, id=7)
            if need_idx_ring
            else None
        ),
        bc_m3m2=tla.flag("smla_bc_m3m2", tla.pipes.MTE3, tla.pipes.MTE2, id=6),
        bc_m3v=tla.flag("smla_bc_m3v", tla.pipes.MTE3, tla.pipes.VECTOR, id=5),
        bc_m2v=tla.flag("smla_bc_m2v", tla.pipes.MTE2, tla.pipes.VECTOR, id=3),
        bc_v2m=tla.flag("smla_bc_v2m", tla.pipes.VECTOR, tla.pipes.MTE2, id=7),
        # F6 流水版：AIC(MTE1) -> AIV(MTE3) 的 l1P 槽读完成归还 flag。F14 步骤
        # A：回执已去除（set/wait 调用点删除，OT IterateBmm2Impl 同款无回执），
        # flag 对象保留占位（sync 结构布局不变）。跨核 flag 必须用
        # cross_flag(mode=4) 创建：cross_core_set/wait 只接受 TlaCrossFlag，
        # in-pipe tla.flag(...) 的句柄经 jit 参数边界会物化成 OpResult 而被拒
        # （实测 expected cross_flag, got OpResult）。
        p_free=cross("smla_p_free"),
        bc_v3m=tla.flag("smla_bc_v3m", tla.pipes.VECTOR, tla.pipes.MTE3, id=6),
        # F10 步骤 2：AB 槽 flag 对（dst=CUBE 已用 0-3 → 4/5；dst=MTE1 已用
        # 0-3 → 4/5）。ab_ready：MTE1→CUBE（数据就绪）；ab_free：CUBE→MTE1
        # （槽释放）。L0A/L0B 合并单 bufferId（A/B 成对）。
        ab_ready0=tla.flag(
            "smla_ab_ready0",
            tla.pipes.MTE1,
            tla.pipes.CUBE,
            id=(0 if opt_cube64 else 4),
        ),
        ab_ready1=tla.flag(
            "smla_ab_ready1",
            tla.pipes.MTE1,
            tla.pipes.CUBE,
            id=(1 if opt_cube64 else 5),
        ),
        ab_free0=tla.flag(
            "smla_ab_free0", tla.pipes.CUBE, tla.pipes.MTE1, id=(0 if opt_cube64 else 4)
        ),
        ab_free1=tla.flag(
            "smla_ab_free1", tla.pipes.CUBE, tla.pipes.MTE1, id=(1 if opt_cube64 else 5)
        ),
        ab_ready2=(
            tla.flag("smla_ab_ready2", tla.pipes.MTE1, tla.pipes.CUBE, id=2)
            if opt_cube64
            else None
        ),
        ab_ready3=(
            tla.flag("smla_ab_ready3", tla.pipes.MTE1, tla.pipes.CUBE, id=3)
            if opt_cube64
            else None
        ),
        ab_free2=(
            tla.flag("smla_ab_free2", tla.pipes.CUBE, tla.pipes.MTE1, id=2)
            if opt_cube64
            else None
        ),
        ab_free3=(
            tla.flag("smla_ab_free3", tla.pipes.CUBE, tla.pipes.MTE1, id=3)
            if opt_cube64
            else None
        ),
        ab_p_ready=(
            tla.flag("smla_ab_p_ready", tla.pipes.MTE1, tla.pipes.CUBE, id=4)
            if opt_cube64
            else None
        ),
        ab_p_free=(
            tla.flag("smla_ab_p_free", tla.pipes.CUBE, tla.pipes.MTE1, id=4)
            if opt_cube64
            else None
        ),
        opt_cube64=opt_cube64,
        opt_cube128=opt_cube128,
        opt_l2m=opt_l2m,
        opt_q_pre3=(opt_cube64 and q_pre3_hint),
        q_l2_bypass=q_l2_bypass,
        # F27：IS_SPLIT_G 核间 MTE2 屏障（全核，mode=0；无 aiv_id）
        s2_mte2=cross("smla_s2_mte2", mode=0),
        m_l0a0=tla.mutex("smla_l0a0", id=0),
        m_l0a1=tla.mutex("smla_l0a1", id=1),
        m_l0b0=tla.mutex("smla_l0b0", id=2),
        m_l0b1=tla.mutex("smla_l0b1", id=3),
        m_o16=tla.mutex("smla_o16", id=4),
        m_p=tla.mutex("smla_p", id=5),
        m_sink=tla.mutex("smla_sink", id=6),
        # L0C O 双槽互斥：cube(mm) 侧与 fix 侧用同一 id，才能形成 MMAD->FIX 依赖
        m_l0c_o0=tla.mutex("smla_l0c_o0", id=7),
        m_l0c_o1=tla.mutex("smla_l0c_o1", id=8),
    )


@tla.jit
def init_sync(
    sync: Sync,
    defer_ofree: bool = False,
    pre_phy_vs: bool = False,
    pre_bc: bool = False,
    pre_l0c2: bool = False,
    pre_idx_ring: bool = False,
    pre_kv_ring: bool = False,
) -> None:
    """预置 flag：等价于 cube_local_buffer.h 的初始 SetFlag 与 smla 的 prodlogue。

    pre_phy_vs：F2 IS_VEC_S2PHYADDR=1 实例预置 1 个 phy_vs 信用（V->S，首轮
    生成入口的 wait 有信用可消费，此后每轮生成 VF 后归还）；=0 实例不预置，
    指令流与改前逐条一致。
    pre_bc：F3 is_batch_consistency=1 实例预置 1 个 bc_v2m 信用（V->MTE2，首个
    merge 段的 wait 有信用可消费，此后每段 merge 末尾归还，OT
    MergeStagedAndCurrentChunk 的 vToMte2Id 预置同款）；=0 实例不预置。
    F6 流水版的信用预置拆到 init_sync_preload（不能放进 if：前端任何 if 边界
    都会把 sync 的 flag 对象物化成 OpResult，cross_core_set_flag 拒收——实测
    `if preload:`/`if tla.const_expr(preload):` 两种形态都报 expected cross_flag,
    got OpResult）。非 swa kernel 不调用 init_sync_preload，指令流与改前逐条
    一致。信用推演（AIC 可超前 AIV 至多 2 tile）见 ITERATIONS.md Iter 23 F6 节。
    """
    with tla.cube():
        # 闸门②变体 B（二分定位）：信用恢复原始握手 x1（寻址/合并保留）。
        tla.set_flag(sync.q_l0a_free)
        if tla.const_expr(sync.opt_q_pre3):
            tla.set_flag(sync.q_l0a_free1)
            tla.set_flag(sync.q_l0a_free2)
        if tla.const_expr(sync.opt_cube64):
            aic_init_loop3_para()
        tla.set_flag(sync.k_l0b_free0)
        tla.set_flag(sync.k_l0b_free1)
        tla.set_flag(sync.k_l0b_free2)
        # F10 步骤 2：AB 槽释放信用预置（OT cube_local_buffer.h:70-71 的
        # SetFlag<M_MTE1>(0)/(1) 同款；L0A/L0B 合并单 bufferId，每槽 1 份）。
        tla.set_flag(sync.ab_free0)
        tla.set_flag(sync.ab_free1)
        if tla.const_expr(sync.opt_cube128):
            # 2-slot L0A/L0B ring (ab_free0/1 already seeded above) + dedicated l0a_p
            tla.set_flag(sync.ab_p_free)
        elif tla.const_expr(sync.opt_cube64):
            # 4 independent 1-bit hardware flags for L0A/L0B slots 0..3 + 1 for l0a_p
            tla.set_flag(sync.ab_free2)
            tla.set_flag(sync.ab_free3)
            tla.set_flag(sync.ab_p_free)
        elif tla.const_expr(L0_D64):
            tla.set_flag(sync.ab_free0)
            tla.set_flag(sync.ab_free0)
            tla.set_flag(sync.ab_free0)
        tla.set_flag(sync.s_fix_done)
        if pre_l0c2:
            # F10 步骤 1（仅 swa preload 实例传 True）：S L0C 槽 1 的首个信用。
            # 槽平衡与槽 0 相同：bmm1(tile t) wait 槽 t%2、fixpipe(t) 后 set 槽 t%2，
            # 首两个 tile 各消费 1 个预置信用。静态 bool 分支（pre_bc 同款），非
            # preload 实例不发出，指令流不变。
            tla.set_flag(sync.s_fix_done1)
        tla.set_flag(sync.o_fix_done0)
        tla.set_flag(sync.o_fix_done1)
    with tla.vector():
        # gather 的 UB 复用信用（MTE3->MTE2），ping-pong 两块各一份
        # R42：need_kv_ring 实例这两族为 None（不走 GM 中转），跳过预置。
        if not pre_kv_ring:
            tla.set_flag(sync.kv_mte3_mte2_0)
            tla.set_flag(sync.kv_mte3_mte2_1)
        if pre_phy_vs:
            tla.set_flag(sync.phy_vs)
        if pre_idx_ring:
            # T1-1：两个槽各预置 1 个 free 信用（首个整环 DMA 的 wait 有信用可
            # 消费，此后每轮标量读完后归还）。对应竞品 ring 的初始可写状态
            # （N:448-455 无显式同步，本仓显式补齐，同 W27b 的 a_idx_free 预置）。
            tla.set_flag(sync.idx_s2m0)
            tla.set_flag(sync.idx_s2m1)
        if pre_bc:
            tla.set_flag(sync.bc_v2m)
        tla.cross_core_set_flag(sync.s_free, tla.pipes.VECTOR, aiv_id=0)
        tla.cross_core_set_flag(sync.s_free, tla.pipes.VECTOR, aiv_id=1)
        if not defer_ofree:
            tla.cross_core_set_flag(sync.o_free, tla.pipes.VECTOR, aiv_id=0)
            tla.cross_core_set_flag(sync.o_free, tla.pipes.VECTOR, aiv_id=1)


@tla.jit
def init_sync_kv_ring(sync: Sync, pa_l1: bool = False, kvz_db: bool = False) -> None:
    """R42：AIV 直写 L1 槽的流水信用预置；init_sync 之后调用。

    - kv_free（cross，AIC MTE1 -> AIV MTE3）：**不预置**。信用改为"后继条件"
      发送：AIC 在 body x 的站点 3 释放 tile x-3 时，只有当**同 body 的后继
      tile x 存在且是 cmp** 才 set 一个 token；AIV 也只在 tile x 是 cmp 且
      x>=3 时消费。两边条数恒等（每 cmp tile 一收一发），且 x<3 的槽在本核
      从未被用过，无需等待。ori tile（占多数）完全不碰该 flag。
    - kvz_free / kv_nd_free：zN 暂存退役后只用 kv_nd_free。首个 band 之前没有
      生产者，需要 1 个在途 token，否则第一个 band 的 MTE2 wait 永久阻塞。
      不能多于 1（多一个会提前覆写 ND，属静默踩写）。

    单独成函数（不并进 init_sync 的 if 分支）：cross set 落在动态 `if` 边界时
    flag 会被物化成 OpResult 而被 cross_core_set_flag 拒收。
    """
    with tla.vector():
        if pa_l1:
            # E17：2 块 ND 各 1 个可写信用；zN 暂存 1 个可写信用（首个 band
            # 之前没有生产者）。都恰 1 个，多了会提前覆写（静默踩写）。
            tla.set_flag(sync.kv_v_m20)
            tla.set_flag(sync.kv_v_m21)
            tla.set_flag(sync.kvz_free)
            if kvz_db:
                tla.set_flag(sync.kvz_free1)
        else:
            tla.set_flag(sync.kv_nd_free)


@tla.jit
def init_sync_preload(sync: Sync) -> None:
    """F6 swa kernel 流水版（PRELOAD_NUM=2）的流水信用预置；init_sync 之后调用。

    F14 步骤 A：p_free 回执已去除（OT IterateBmm2Impl 对 l1PBuffers 单向 +
    双槽 slack，无 C→V credit；vec1(0)/vec1(1) 首两 tile 无读方，无需初始
    信用。踩写安全链：vec1(x) 写槽 x%2 前 WAIT s_ready ← bmm1(x) SET，
    AIC 单核程序序 bmm2(x-2)→…→bmm1(x) + MTE1 FIFO ⇒ 读完成早于写）。
    s_free（V->FIX，bmm1 写 ub_s 槽前等 vec1 读完同槽）：流水下 AIC 的
      bmm1(0)/bmm1(1) 各消费 1 个初始信用（vec1(-1)/vec1(-2) 不存在），此后
      vec1(t) 归还、bmm1(t+2) 消费 ⇒ init_sync 的 1 份之外再 +1/AIV。
    """
    with tla.vector():
        tla.cross_core_set_flag(sync.s_free, tla.pipes.VECTOR, aiv_id=0)
        tla.cross_core_set_flag(sync.s_free, tla.pipes.VECTOR, aiv_id=1)


@tla.jit
def alloc_buffers(
    dtype_in,
    row_cap=None,
    sink_cap=None,
    phy=0,
    phy_bt=0,
    bc=0,
    preload=False,
    idx_ring=False,
    kv_ring=False,
    pa_l1=False,
    alias_o16=False,
    kvz_db=False,
    fd_splits=1,
    opt_cube64: bool = False,
    opt_cube128: bool = False,
    opt_vec: bool = False,
) -> Bufs:
    """按 OT 的字节表分配（SPACE-DIFF.md §1-§3）。

    row_cap 仅用于奇数 G（IS_SPLIT_G 且 gSize 为奇数）：此时各 AIV 的 S/Otmp
    view 按 Align2(mRealSize) 行声明（OT fixpipe mSize=Align2Func(mRealSize)），
    比默认 Q_ROWS/2 多 1~2 行；偶数 G 传 None，字节表与原先逐字节一致。
    sink_cap 同样只用于奇数 G：sinks 整条（Align8 补齐）一次搬入 UB 后逐行取，
    需要 head_num 上对齐的长度；偶数 G 传 None。
    phy（F2 IS_VEC_S2PHYADDR=1 实例）：各段 alignedSparseBlockCount 的最大值，
    即 ub_phy_idx / ub_phy_addr（u32 视图为 2*phy）的容量；phy_bt 为各段
    maxBlockNumPerBatch 的最大值。=0 实例传 0，只分配 1 元素占位（布局不变）。
    bc（F3 IS_BATCH_CONSISTENCY=1 实例）：staging merge 的 UB 暂存
    （ub_bc_prev/max/sum/w）；=0 实例只分配 1 元素占位（与 phy 同款处理），
    字节表在 =0 与改前逐字节一致。
    """
    # ---- L1：l1P[2] 32,768 | l1Q 3x32,768 | l1K(V) 3x131,072 = 524,288 B ----
    # F6：swa kernel 流水版（preload=True）补齐 OT l1PBuffers[2]——连续双槽
    # （+16,384 B，OT 字节表本就含两槽），槽 1 由 ptr + 8192 元素寻址；
    # 其余实例单槽，字节表不变。
    l1_p = tla.allocate(
        Q_ROWS * KV_BLOCK * (2 if preload else 1), dtype_in, tla.AddressSpace.l1, 512
    )
    # 闸门② Q 3 槽环：三槽合并为一段连续分配（元素总数 3*Q_ROWS*Q_HALF 与原
    # 三块独立分配逐字节一致，98,304 B，L1 合计仍 524,288 B——硬条件 1 无新增）；
    # 槽 i 由 ptr + i*(Q_ROWS*Q_HALF) 寻址（l1_k/l1_p 同款）。
    l1_q = tla.allocate(3 * Q_ROWS * Q_HALF, dtype_in, tla.AddressSpace.l1, 512)
    l1_k = tla.allocate(
        KV_SLOTS * HEAD_DIM * KV_BLOCK, dtype_in, tla.AddressSpace.l1, 512
    )
    # ---- L0A (64 KB on Ascend 950) & L0B (64 KB) ----
    if tla.const_expr(opt_cube128):
        # Small-M (m <= 32): 2 x (32,128) = 16 KB L0A Q ring + 1 x (32,128) = 8 KB
        # dedicated l0a_p (total 24 KB <= 64 KB L0A) + 2 x (128,128) = 64 KB L0B ring.
        # Halves MMAD (16 -> 8) and FixPipe (9 -> 5) launches per tile.
        l0a0 = tla.allocate(32 * D_CHUNK, dtype_in, tla.AddressSpace.l0a, 512)
        l0a1 = tla.allocate(32 * D_CHUNK, dtype_in, tla.AddressSpace.l0a, 512)
        l0a2 = None
        l0a3 = None
        l0a_p = tla.allocate(32 * KV_BLOCK, dtype_in, tla.AddressSpace.l0a, 512)
        l0b0 = tla.allocate(D_CHUNK * KV_BLOCK, dtype_in, tla.AddressSpace.l0b, 512)
        l0b1 = tla.allocate(D_CHUNK * KV_BLOCK, dtype_in, tla.AddressSpace.l0b, 512)
        l0b2 = None
        l0b3 = None
    elif tla.const_expr(opt_cube64):
        # Track 2: 4 x (64,64) = 32 KB for bmm1 Q ring + 1 x (64,128) = 16 KB
        # dedicated l0a_p for bmm2 P (total 48 KB <= 64 KB L0A, zero bmm1/bmm2 WAR).
        # L0B: 4 x (64,128) = 64 KB exact capacity (shared 4-slot ring for K and V).
        l0a0 = tla.allocate(Q_ROWS * 64, dtype_in, tla.AddressSpace.l0a, 512)
        l0a1 = tla.allocate(Q_ROWS * 64, dtype_in, tla.AddressSpace.l0a, 512)
        l0a2 = tla.allocate(Q_ROWS * 64, dtype_in, tla.AddressSpace.l0a, 512)
        l0a3 = tla.allocate(Q_ROWS * 64, dtype_in, tla.AddressSpace.l0a, 512)
        l0a_p = tla.allocate(Q_ROWS * KV_BLOCK, dtype_in, tla.AddressSpace.l0a, 512)
        l0b0 = tla.allocate(64 * KV_BLOCK, dtype_in, tla.AddressSpace.l0b, 512)
        l0b1 = tla.allocate(64 * KV_BLOCK, dtype_in, tla.AddressSpace.l0b, 512)
        l0b2 = tla.allocate(64 * KV_BLOCK, dtype_in, tla.AddressSpace.l0b, 512)
        l0b3 = tla.allocate(64 * KV_BLOCK, dtype_in, tla.AddressSpace.l0b, 512)
    elif tla.const_expr(L0_D64):
        l0a0 = tla.allocate(Q_ROWS * L0_D, dtype_in, tla.AddressSpace.l0a, 512)
        l0a1 = tla.allocate(Q_ROWS * L0_D, dtype_in, tla.AddressSpace.l0a, 512)
        l0a2 = tla.allocate(Q_ROWS * L0_D, dtype_in, tla.AddressSpace.l0a, 512)
        l0a3 = tla.allocate(Q_ROWS * L0_D, dtype_in, tla.AddressSpace.l0a, 512)
        l0a_p = None
        l0b0 = tla.allocate(L0_D * KV_BLOCK, dtype_in, tla.AddressSpace.l0b, 512)
        l0b1 = tla.allocate(L0_D * KV_BLOCK, dtype_in, tla.AddressSpace.l0b, 512)
        l0b2 = tla.allocate(L0_D * KV_BLOCK, dtype_in, tla.AddressSpace.l0b, 512)
        l0b3 = tla.allocate(L0_D * KV_BLOCK, dtype_in, tla.AddressSpace.l0b, 512)
    else:
        l0a0 = tla.allocate(Q_ROWS * D_CHUNK, dtype_in, tla.AddressSpace.l0a, 512)
        l0a1 = tla.allocate(Q_ROWS * D_CHUNK, dtype_in, tla.AddressSpace.l0a, 512)
        l0a2 = None
        l0a3 = None
        l0a_p = None
        l0b0 = tla.allocate(D_CHUNK * KV_BLOCK, dtype_in, tla.AddressSpace.l0b, 512)
        l0b1 = tla.allocate(D_CHUNK * KV_BLOCK, dtype_in, tla.AddressSpace.l0b, 512)
        l0b2 = None
        l0b3 = None
    # ---- L0C 262,144 B：S + O 双槽 ----
    # F10 步骤 1：swa preload 实例的 S 槽扩成连续双槽（OT INNERCORE_L0C GetNext
    # ring(2) 同款，65,536 B；L0C 合计 131,072 B < 262,144 上限）；其余实例单槽，
    # 字节表不变。槽 i = ptr + i*(Q_ROWS*KV_BLOCK) 元素（ub_s/l1_p 同款寻址）。
    l0c_s = tla.allocate(
        Q_ROWS * KV_BLOCK * (2 if preload else 1),
        tla.Float32,
        tla.AddressSpace.l0c,
        1024,
    )
    # F17 步骤 B：O 改单槽整块 128 KB（Q_ROWS*HEAD_DIM fp32 = 32,768 元素，
    # 4 个 N 窗口 mmad 各占 seg*(m*D_CHUNK) 线性段、每窗口 init_c=True，fixpipe
    # 每 tile 一次整块 SPLIT_M——OT IterateBmm2 同款；探针 mode4/9/10 + mode1
    # 终判成立）。L0C 合计 = l0c_s 双槽 65,536 B + l0c_o 131,072 B = 196,608 B
    # < 262,144 上限（preload 实例；非 preload 163,840 B）。
    l0c_o = tla.allocate(Q_ROWS * HEAD_DIM, tla.Float32, tla.AddressSpace.l0c, 1024)
    # ---- UB（每 AIV）----
    half = Q_ROWS // 2 if row_cap is None else row_cap
    # F6：swa 流水版的 UB S 双槽（OT bmm1Buffers[2]，连续分配，槽 1 由
    # ptr + half*KV_BLOCK 寻址，+half*KV_BLOCK fp32）；其余实例单槽不变。
    ub_s = tla.allocate(
        half * KV_BLOCK * (2 if preload else 1), tla.Float32, tla.AddressSpace.ub, 256
    )
    # T2b：ub_p 的消费视图改为标准 zN（无 pad，参考 basic_mixed_store_zN
    # 派生式）：物理容量 = pad16(rows)*128 el——rows=32（#018）时 4096 =
    # half*KV_BLOCK 恰满；rows=17（#013，half=row_cap=34）时 4096 <= 4352。
    # 分配式保持 T2a 原样不变。
    ub_p = tla.allocate(half * KV_BLOCK, dtype_in, tla.AddressSpace.ub, 256)
    # F17 步骤 B：ub_otmp0/1（2 x half*D_CHUNK = 32,768 B）合并为单块整宽
    # ub_otmp_full（half*HEAD_DIM fp32 = 65,536 B，每 AIV 半行 32 x 512）。
    # bc=0 每 AIV UB 合计 208,800 -> 241,568 B（余 12,384 B / 253,952 预算，
    # 不需要 --cce-disable-* 开关）；bc=1 实例 258,448 B > 253,952 但
    # < 262,144（验证形态，见账本 F17 节）。
    ub_otmp_full = tla.allocate(half * HEAD_DIM, tla.Float32, tla.AddressSpace.ub, 256)
    ub_o = tla.allocate(half * HEAD_DIM, tla.Float32, tla.AddressSpace.ub, 256)
    ub_o16 = (
        tla.recast_ptr(
            ub_otmp_full if opt_vec else ub_o,
            dtype=dtype_in,
        )
        if alias_o16
        else tla.allocate(half * HEAD_DIM, dtype_in, tla.AddressSpace.ub, 256)
    )
    # sinks：偶数 G 按 rows；奇数 G 存整条（host 补齐到 8 元素/32B 倍数）
    sink_len = half if sink_cap is None else sink_cap
    _sink_alloc = max(sink_len, STATE_SLOT_STRIDE) if opt_vec else sink_len
    ub_sink = tla.allocate(_sink_alloc, tla.Float32, tla.AddressSpace.ub, 256)
    # F11·bc：max 双槽（OT softmaxMaxBufs[multiCoreIdxMod2] 双槽先例，账本
    # F10/F11 节「softmaxSum/MaxBufs[2] 双槽是核私有工作区」记录）。bc=0 时
    # max 只有 vec1 链 + vec_init 读改写，单槽即可；bc=1 时块末 tile 的
    # bc_merge_prev 要读 max(t)（本 tile 的 running max），而交换后它晚于
    # vec1(t+1) 执行（body t+2：stage1 先发 vec1(t+1) 写槽 (t+1)%2，stage2
    # 才跑 merge(t)）——单槽下 lse_c 读到 max(t+1)。双槽链与 sum 同键
    # xg%2：vec1(t) 读 (t-1)%2 写 t%2、reinit/vec_init 重灌 t%2、merge(t)
    # 就地 RMW 槽 t%2。非 preload 实例（串行调用方）单槽不变。
    _state_el = STATE_SLOT_STRIDE if opt_vec else half
    ub_max = tla.allocate(
        _state_el * (2 if preload else 1), tla.Float32, tla.AddressSpace.ub, 256
    )
    # F11：sum 双槽（OT softmaxSumBufs[2] 同构）——bmm2_data_copy_out 就地
    # 用 ub_sum 归一化（div(src,den) 回写），交换后同 body 的 vec_init(T 首)
    # 先于 copy_out(P 末) 重灌 sum=1.0，非末任务输出全变 raw（实测：仅最后
    # task 全对）。槽键 xg%2 与 dm 同：vec1(x-1) 写槽 (x-1)%2、copy_out(P 末)
    # 读槽 (P末)%2、vec_init(T首=P末+1) 重灌异槽。非 preload 实例单槽不变。
    ub_sum = tla.allocate(
        _state_el * (2 if preload else 1), tla.Float32, tla.AddressSpace.ub, 256
    )
    # F11：dm 双槽（OT softmaxExpBufs[2]，taskIdMod2 = 全局 tile 序 %2 同键，
    # sparse_flash_mla_csa_block_vector_arch35.h:316/:1423-1425 连续分配先例
    # 同 ub_s：槽 1 由 ptr + rows 寻址）。源序交换（stage1 先于 stage2）后
    # vec1(x-1) 写 dm(x-1) 与 vec2(x-2) 读 dm(x-2) 相邻 tile 奇偶交替，
    # 单缓冲会互相覆盖；非 preload 实例（串行调用方）单槽不变。
    ub_dm = tla.allocate(
        _state_el * (2 if preload else 1), tla.Float32, tla.AddressSpace.ub, 256
    )
    # 稀疏 gather 中转：2 x 16 行 x HEAD_DIM（OT stage0OutBufs[2]，KV_PROCESS_UNIT）
    ub_kv0 = tla.allocate(KV_UNIT * HEAD_DIM, dtype_in, tla.AddressSpace.ub, 256)
    # R42：kv_ring 实例只用 ub_kv0 作 ND 中转（zN 走 ub_kvz），GM 中转路径整段
    # 不发射 ⇒ ub_kv1 不分配（省出 16,384 B 给 ub_kvz 的 17 行 pitch）。
    ub_kv1 = (
        tla.allocate(KV_UNIT * HEAD_DIM, dtype_in, tla.AddressSpace.ub, 256)
        if ((not kv_ring) or pa_l1)
        else None
    )
    # E23：PA 直写改用 pitch=17（zNUnAlign，块距 272 元素 = 544 B）以避开
    # pitch16（512 B 块距）在 UB 上的 bank 冲突；实测换形 47.9 -> 121.2 GB/s/核。
    ub_kvz = (
        tla.allocate(
            KV_NZ_PITCH * HEAD_DIM,
            dtype_in,
            tla.AddressSpace.ub,
            256,
        )
        if (kv_ring or pa_l1)
        else None
    )
    ub_kvz1 = (
        tla.allocate(
            KV_NZ_PITCH * HEAD_DIM,
            dtype_in,
            tla.AddressSpace.ub,
            256,
        )
        if kvz_db
        else None
    )
    # F4 清零中转（OT: ATTEN_OUT_POP_BUF / LSE_POP_BUF，各 32 KiB；
    # 这里取 INIT_POP_ELE 元素的更小粒度，见 smla_csa_block_vector.py 的偏离说明）
    ub_init = tla.allocate(INIT_POP_ELE, dtype_in, tla.AddressSpace.ub, 256)
    ub_init_lse = tla.allocate(INIT_POP_ELE, tla.Float32, tla.AddressSpace.ub, 256)
    # F2 phy-addr 生成暂存（OT sparseIdxUb / blkTableUb / kvPhyAddrUb）。
    # kvPhyAddrUb 按 u32 视图分配（TLA gm<->ub copy 无 i64 路由，拷出时按 u32 搬），
    # 容量 = aligned * 2 个 u32 = aligned 个 int64（OT 同容量）。
    phy_cap = phy if phy > 0 else 1
    bt_cap = phy_bt if phy_bt > 0 else 1
    # gather 窗宽 64（i32 寄存器上限，E-F2c）：bt 行按 64 列分窗读取，
    # bt_max 非 64 倍数（如 32）时把 UB 容量补齐到整窗，窗 tile 不越出分配。
    bt_cap = ((bt_cap + 63) // 64) * 64
    ub_phy_idx = tla.allocate(phy_cap, tla.Int32, tla.AddressSpace.ub, 256)
    ub_phy_bt = tla.allocate(bt_cap, tla.Int32, tla.AddressSpace.ub, 256)
    ub_phy_addr = tla.allocate(phy_cap * 2, tla.Int32, tla.AddressSpace.ub, 256)
    # T1-1（r10）：非 PA 索引环（2 x IDX_RING_STRIDE i32 = 576 B/AIV；32 B 对齐
    # 满足 MTE2 目的地址要求）。idx_ring=False 实例**不分配**（None）⇒ UB 字节表
    # 与改前逐字节一致（PA #25 控制臂靠这条保持对象级不变）。
    ub_idx_ring = (
        tla.allocate(
            IDX_RING_SLOTS * IDX_RING_STRIDE, tla.Int32, tla.AddressSpace.ub, 32
        )
        if idx_ring
        else None
    )
    # F3 BC staging 暂存。max/sum 容量按 Align8(rows) 补齐：MTE 的 GM<->UB 拷贝
    # 要求 32B 对齐（D-F2a 审计），rows ∈ {1,4}（#004 h=1 / #007 h=8）时 rows*4 B
    # 不足 32B，补齐后 GM 槽与 UB 两侧都按 align8 拷贝、merge 只读前 rows 行。
    bc_cap = half if bc == 1 else 1
    bc_cap8 = ((bc_cap + 7) // 8) * 8 if bc == 1 else 1
    ub_bc_prev = tla.allocate(bc_cap * D_CHUNK, tla.Float32, tla.AddressSpace.ub, 256)
    ub_bc_max = tla.allocate(bc_cap8, tla.Float32, tla.AddressSpace.ub, 256)
    ub_bc_sum = tla.allocate(bc_cap8, tla.Float32, tla.AddressSpace.ub, 256)
    ub_bc_w = tla.allocate(bc_cap8 * 2, tla.Float32, tla.AddressSpace.ub, 256)
    # E4：FD merge 暂存（仅 fd_splits>1 分配；=1 为 None，UB 字节表不变）。
    # 布局（按 AIV 子块，rows8=Align8(rows)）：
    #   [max_i: S*rows8][sum_i: S*rows8][e_i: S*rows8][den: rows8][sink: rows8]
    fd_cap8 = ((half + 7) // 8) * 8
    # 统计块宽度 = max(Align8(rows), 64)：UB 整向量 load 需 VL(fp32)=64 宽。
    fd_cap8 = ((fd_cap8 + 63) // 64) * 64
    ub_fd_stats = (
        tla.allocate(
            (3 * fd_splits + 2) * fd_cap8 + 256,
            tla.Float32,
            tla.AddressSpace.ub,
            256,
        )
        if fd_splits > 1
        else None
    )
    return Bufs(
        l1_p=l1_p,
        l1_q=l1_q,
        l1_k=l1_k,
        l0a0=l0a0,
        l0a1=l0a1,
        l0a2=l0a2,
        l0a3=l0a3,
        l0a_p=l0a_p,
        l0b0=l0b0,
        l0b1=l0b1,
        l0b2=l0b2,
        l0b3=l0b3,
        l0c_s=l0c_s,
        l0c_o=l0c_o,
        ub_s=ub_s,
        ub_p=ub_p,
        ub_otmp_full=ub_otmp_full,
        ub_o=ub_o,
        ub_o16=ub_o16,
        ub_sink=ub_sink,
        ub_max=ub_max,
        ub_sum=ub_sum,
        ub_dm=ub_dm,
        ub_kv0=ub_kv0,
        ub_kv1=ub_kv1,
        ub_kvz=ub_kvz,
        ub_kvz1=ub_kvz1,
        ub_init=ub_init,
        ub_init_lse=ub_init_lse,
        ub_phy_idx=ub_phy_idx,
        ub_phy_bt=ub_phy_bt,
        ub_phy_addr=ub_phy_addr,
        ub_idx_ring=ub_idx_ring,
        ub_bc_prev=ub_bc_prev,
        ub_bc_max=ub_bc_max,
        ub_bc_sum=ub_bc_sum,
        ub_bc_w=ub_bc_w,
        ub_fd_stats=ub_fd_stats,
        opt_cube64=opt_cube64,
        opt_vec=opt_vec,
    )
