# -----------------------------------------------------------------------------------------------------------
# Sparse flash attention (MLA-absorb) in CATLASS TLA DSL.
#
# Single-kernel mixed Cube/Vector pipeline: each tick gathers a KV tile of TILE_S2=128
# rows and overlaps QK, online softmax, PV and output accumulation across AIC/AIV,
# using a 3-slot L1 ring and a 2-slot L0C software pipeline. Supports BSND/TND query
# layouts, BSND/TND/PA_BSND KV layouts, optional RoPE, sinks and max/sum (LSE) outputs.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import collections

import torch
import torch_npu  # noqa: F401  -- registers torch.npu

import catlass.tla as tla
from catlass.tla.runtime import from_dlpack

# --- 模块级常量 -------------------------------------------------------------------------------
INT32_MAX = 2_147_483_647

INVALID_SPARSE_INDEX = -1
EMPTY_QUERY_ID = -1
SPARSE_MODE_DENSE = 0
SPARSE_MODE_RIGHT_DOWN_CAUSAL = 3

TILE_M = 64
TILE_S2 = 128
CUBE_K_TILE = 128
GATHER_CHUNK_ROWS = 16
AIV_ROW_CAPACITY = 32
AIV_COUNT = 2
SPARSE_INDEX_PAIR_SIZE = 2

# MLA-absorb 的 NoPE / RoPE 维度，本 kernel 固定为 512 / 64。
DIM_NOPE = 512
DIM_ROPE = 64
DIM_QK = DIM_NOPE + DIM_ROPE

PIPELINE_SLOT_COUNT = 3
PIPELINE_STAGE_COUNT = 4
PIPELINE_DRAIN_TICKS = PIPELINE_STAGE_COUNT - 1

BYTES_PER_16BIT_ELEMENT = 2
NZ_BLOCK_BYTES = 32
VECTOR_REGISTER_BITS = 2048
FP32_BITS = 32
VL_T = VECTOR_REGISTER_BITS // FP32_BITS
SOFTMAX_NEG_INF = -1.0e30
EMPTY_SOFTMAX_THRESHOLD = -1.0e29


# --- KV 行拷贝（gather 的逐行 / 成对写入） ----------------------------------------------------
def _kv_stage_view(stage, row, offset, width):
    """Element-granularity (1, width) window inside the UB stage buffer.

    `tla.tile_view` 的 coord 是瓦片粒度（内部会乘上 shape 再落成元素偏移），
    这里必须用 `tla.get_tile` 的元素偏移语义，否则 RoPE 列（offset=DIM_NOPE）
    会被算成 DIM_NOPE*DIM_ROPE 而越界。
    """
    return tla.get_tile(
        stage,
        tla.make_coord(row, offset),
        tla.make_shape(1, width),
    )


def _copy_kv_token(stage, key_token, rope_token, row):
    """Copy one sparse KV row with NoPE followed by RoPE in UB."""
    tla.copy(_kv_stage_view(stage, row, 0, DIM_NOPE), key_token)
    tla.copy(_kv_stage_view(stage, row, DIM_NOPE, DIM_ROPE), rope_token)


def _copy_kv_token_no_rope(stage, key_token, row):
    """Copy one sparse K row when RoPE is disabled."""
    tla.copy(_kv_stage_view(stage, row, 0, DIM_NOPE), key_token)


def _kv_pair_source(gm, first_row, row_gap, width):
    """Build the 2-row strided GM source of one pair-gather DMA.

    双行 gather 的物理形态是**一条**双行 DMA：首行基址 + 运行期行跨度（元素），
    行内连续，对应 `copy_gm2ub_align(dst+row*576, gm+first, 2, 1024, …,
    stride=gap*1024, 1152)` 的形态。

    **首行基址必须用根张量 `ptr + row*width`**，不能用 `get_tile(...).ptr`：
    `tla.tensor_ptr` 只回原张量基址，tile 偏移被丢掉（实测用 get_tile 写法会
    读成 src[0]/src[gap]，根张量 + 行偏移写法正确）。
    """
    return tla.make_tensor(
        gm.ptr + first_row * width,
        tla.make_layout(
            tla.make_shape(2, width),
            tla.make_stride(row_gap * width, 1),
            layoutTag=tla.arch.RowMajor,
        ),
    )


def _copy_kv_pair(stage, key, key_rope, first_row, row_gap, pair_index):
    """Copy two sparse KV rows with one NoPE DMA and one RoPE DMA.

    一次 gather 折成单条双行 DMA（2 行 + stride）。行跨度为 `row_gap * DIM_*`
    元素（`row_gap` 是两 token 的行距），目标侧行距由 stage0 父张量 pitch
    （`qk_dim`）给出，即 1152 B。`row_gap >= 1` 由调用点的
    `source_gap_bytes >= 0` 守卫保证（同 token 走单行回退）。

    目标视图的 coord 是**瓦片粒度**：行偏移 = pair_index * 2
    （漏乘会把两行写到同一位置，只落一半行）。
    """
    row0 = pair_index * SPARSE_INDEX_PAIR_SIZE
    tla.copy(
        tla.get_tile(stage, tla.make_coord(row0, 0), tla.make_shape(2, DIM_NOPE)),
        _kv_pair_source(key, first_row, row_gap, DIM_NOPE),
    )
    tla.copy(
        tla.get_tile(
            stage, tla.make_coord(row0, DIM_NOPE), tla.make_shape(2, DIM_ROPE)
        ),
        _kv_pair_source(key_rope, first_row, row_gap, DIM_ROPE),
    )


def _copy_kv_pair_no_rope(stage, key, first_row, row_gap, pair_index):
    """Copy two sparse K rows with one NoPE DMA when RoPE is disabled."""
    row0 = pair_index * SPARSE_INDEX_PAIR_SIZE
    tla.copy(
        tla.get_tile(stage, tla.make_coord(row0, 0), tla.make_shape(2, DIM_NOPE)),
        _kv_pair_source(key, first_row, row_gap, DIM_NOPE),
    )


# --- 掩码长度夹取与 NZ 物理步长 ------------------------------------------------------------
@tla.jit
def _clamp_nonneg(value: int) -> int:
    """Clamp a dynamic mask length before converting it to an unsigned value."""
    return 0 if value < 0 else value


def _nz_params(p_ub):
    """Derive NZ fractal addressing from p_ub physical layout stride.

    catlass.tla 把 NZ 物理排布放在 layout tag（`tla.arch.nZ`）里，张量本身只暴露
    逻辑 (M, N) 形状；这里按 C0=32B 的分形规则重建 physical stride 的四元组
    （数值等价：由逻辑形状 + C0 规则导出，而非读取已声明的物理 stride）。
    """
    n0 = NZ_BLOCK_BYTES // BYTES_PER_16BIT_ELEMENT
    m0 = n0
    rows = p_ub.origin_shape[0]
    s_m0 = n0
    s_n0 = 1
    s_m1 = m0 * s_m0
    s_n1 = (rows // m0) * s_m1
    return m0, s_m1, s_m0, s_n1 // (s_m0 // s_n0)


# ---------------------------------------------------------------------------------------------
# `_StableOnlineSoftmaxVector`（向量 softmax 基类）
# ---------------------------------------------------------------------------------------------
_CAST_EVEN = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ZERO,
    sat_mode=tla.params.SatMode.NOSAT,
    round_mode=tla.params.RoundMode.CAST_RINT,
)
_CAST_ODD = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ONE,
    sat_mode=tla.params.SatMode.NOSAT,
    round_mode=tla.params.RoundMode.CAST_RINT,
)


def _ub_tensor(shape, dtype):
    """Allocate a UB tile and land a row-major layout on it."""
    ptr = tla.allocate(shape, dtype, tla.AddressSpace.ub, 32)
    if len(shape) == 1:
        layout = tla.make_layout(tla.make_shape(shape[0]), tla.make_stride(1))
    else:
        layout = tla.make_layout(
            tla.make_shape(shape[0], shape[1]), tla.make_stride(shape[1], 1)
        )
    return tla.make_tensor(ptr, layout)


def _ub_nz_tensor(shape, dtype):
    """Allocate a UB tile with the NZ fractal physical layout."""
    ptr = tla.allocate(shape, dtype, tla.AddressSpace.ub, 32)
    layout = tla.make_layout_with_tag(
        (shape[0], shape[1]), dtype, layoutTag=tla.arch.nZ
    )
    return tla.make_tensor(ptr, layout)


def _vec_view(tensor, row, col, count):
    """Element-granularity 1-D vector view at (row, col) of a 2-D tensor."""
    return tla.get_tile(tensor, tla.make_coord(row, col), tla.make_shape(1, count))


def _row_view(tensor, row, col, count):
    """Runtime-row 1-D vector view (tile_view with a dynamic row coordinate)."""
    return tla.tile_view(tensor, tla.make_shape(1, count), tla.make_coord(row, col))


# --- 单条融合乘加（`asc_madd`） --------------------------------------------------------------
# tla 方言没有 fma，`@tla.extern` 可直达 `asc_madd` / `asc_fma`。
#
# 两条路线都试过：
#   1) 寄存器实参（返回 vector_float，TLA 侧 tla.mul/tla.add 换成 fma_reg）：
#      extern 的向量实参必须是满寄存器（f32 → 64 valid lanes），必须用
#      「运行期偏移指针 + 静态 layout」重建张量才能满足；但 TlaVectorRegionSimdPass
#      只允许**同一个 vec.func 里、同一块缓冲只被一个 extern 调用点引用**——
#      一行 8 段就会有 8 个调用点复用 res_o/pv，outline 直接放弃（"failed to outline
#      enclosing SIMD VF for extern"）。1 段能过、2 段就挂，与行循环无关。
#   2) 指针实参（本文件采用）：dst/pv 走 UB 指针，exp_b 仍由 TLA 侧
#      `DIST_BRC_B32` 载入后作为**满寄存器实参**传入（frontend 已记 64 valid lanes），
#      C 里自己展开 8 段 —— 每行只有**一个**调用点，outline 正常，且 `update_o`
#      的签名与调用点都不用改。C 侧用 asc_madd 得到单次舍入。
#      生成码形态：asc_loadalign_brc 一次 +
#      8×(asc_loadalign ×2 + asc_madd + asc_storealign)。
UPDATE_O_SOURCE = r"""
#include <cstdint>
#include "c_api/asc_simd.h"

// dst = dst * b + pv（逐 64 lane 单次舍入的 FMA）
// 软件流水：下一段的两条载入提前到本段 madd/store 之前发射，恢复被 region 边界切掉的预取。
extern "C" __aicore__ __attribute__((always_inline))
void update_o_row(uint64_t dst_addr, uint64_t pv_addr, vector_float b,
                  int32_t cols) {
    auto dst = reinterpret_cast<__ubuf__ float *>(dst_addr);
    auto pv = reinterpret_cast<__ubuf__ float *>(pv_addr);
    vector_bool mask = asc_create_mask_b32(PAT_ALL);
    if (cols <= 0) {
        return;
    }
    vector_float a, p, na, np, o;
    asc_loadalign(a, dst);
    asc_loadalign(p, pv);
    int32_t c = 0;
    for (; c + 64 < cols; c += 64) {
        asc_loadalign(na, dst + c + 64);
        asc_loadalign(np, pv + c + 64);
        o = a;
        asc_madd(o, b, p, mask);
        asc_storealign(dst + c, o, mask);
        a = na;
        p = np;
    }
    o = a;
    asc_madd(o, b, p, mask);
    asc_storealign(dst + c, o, mask);
}
"""


@tla.extern(source=UPDATE_O_SOURCE)
def update_o_row(
    dst: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    pv: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    b: tla.VectorSSA[tla.Float32],
    cols: tla.Int32,
) -> None: ...


class _StableOnlineSoftmaxVector:
    """Online softmax state, P packing and output accumulation."""

    def __init__(
        self,
        tile_vec_m,
        tile_n,
        tile_d,
        subblock_idx,
        dtype_16=tla.Float16,
        preload_num=PIPELINE_SLOT_COUNT,
        head_count=AIV_ROW_CAPACITY,
    ):
        self.tile_vec_m = tile_vec_m
        self.tile_m = tile_vec_m * 2
        self.tile_n = tile_n
        self.tile_d = tile_d
        self.subblock_idx = subblock_idx
        self.dtype_16 = dtype_16
        # `finish_o_cast` 用 `_output_tile` 的静态行数作搬运行数；tla 的
        # `.shape` / `.origin_shape` 在降级后都是运行时 Int32，不能当编译期
        # 循环上界，这里以构造期常量保留同一数值。
        self.head_count = head_count

        # 三组 running state 各 preload_num 槽；单块分配 + 槽内行偏移。
        self.sm_max_tb = _ub_tensor((preload_num * tile_vec_m,), tla.Float32)
        self.sm_sum_tb = _ub_tensor((preload_num * tile_vec_m,), tla.Float32)
        self.sm_exp_tb = _ub_tensor((preload_num * tile_vec_m,), tla.Float32)

        self.tmp_new_max = _ub_tensor((tile_vec_m,), tla.Float32)
        self.tmp_sum = _ub_tensor((tile_vec_m,), tla.Float32)
        self.res_o = _ub_tensor((tile_vec_m, tile_d), tla.Float32)

        # P 缓冲布局：每槽 8 条带，每带 tile_vec_m*16 个数据元素 + 16 pad
        # （32 行 → 512+16），带距 = 33 个 32B DataBlock = 528 元素；槽独立
        # 分配（每槽 8*528 = 4224 元素），不做 slot 交错。
        self.p_band = tile_vec_m * 16 + 16
        self.p_slot_elems = (tile_n // 16) * self.p_band
        self.p_block_stride = self.p_band // 16
        self.p_ub = _ub_tensor((2 * self.p_slot_elems,), self.dtype_16)
        # o_ub：depth=1，produce() 后固定使用槽 0。
        self.o_ub = _ub_tensor((tile_vec_m, tile_d), self.dtype_16)

        # AIV 侧核内同步：V 生产、MTE3 消费的片上缓冲各一把 mutex。
        # p_ub 之外的 stage0 / zero_row 归 kernel 类持有。
        self.mtx_p = tla.mutex("p_ub", id=8)
        self.mtx_o = tla.mutex("o_ub", id=9)

    # --- 槽游标 → 显式槽视图 -------------------------------------------------------------
    def state_slot(self, table, index):
        return tla.tile_view(
            table,
            tla.make_shape(self.tile_vec_m),
            tla.make_coord(index),
        )

    def state_base(self, table, index):
        """第 `index` 槽的基址视图（槽偏移折进 ptr）。

        行访问采用「缓冲区 + 行号」的一层偏移：把槽偏移折进基址后，行访问只剩
        一层 `tile_view(shape(1), coord(row))`；`tile_view(槽视图, shape(1),
        coord(row))` 这种两层嵌套在非 0 槽上读写不稳（实测）。基址必须在 vec
        区外构造：SIMD 区体内做 ptr 偏移会让区内的 `tla.load` 无法合法化。
        """
        return tla.make_tensor(
            table.ptr + index * self.tile_vec_m,
            tla.make_layout(tla.make_shape(self.tile_vec_m), tla.make_stride(1)),
        )

    def p_slot(self, index):
        # 每槽 8 带 × 528 元素独立存放；行内基址由 (i%16)*16+(i//16)*256
        # 给出，带距 528 由 BlockStoreParams 表达。
        return tla.make_tensor(
            self.p_ub.ptr + index * self.p_slot_elems,
            tla.make_layout(
                tla.make_shape(1, self.p_slot_elems),
                tla.make_stride(self.p_slot_elems, 1),
            ),
        )

    def _state_2d(self, table, index):
        """1-D 状态槽 → (vec_m, 1) 的 2-D 视图（重解释为二维布局）。"""
        return tla.make_tensor(
            table.ptr + index * self.tile_vec_m,
            tla.make_layout(tla.make_shape(self.tile_vec_m, 1), tla.make_stride(1, 1)),
        )

    def _state_flat(self, table, index):
        """1-D 状态槽视图（UB→UB 整槽拷贝用；2-D (m,1) 行 pitch 只有 4 B）。"""
        return tla.make_tensor(
            table.ptr + index * self.tile_vec_m,
            tla.make_layout(tla.make_shape(self.tile_vec_m), tla.make_stride(1)),
        )

    # --- softmax_first：本 Query 首块 softmax ---------------------------------------------
    @tla.jit
    def softmax_first(self, qk_ch, actual_n, scale, m_axis_triple, p_slot):
        """Initialize running max/sum and unnormalized weights for the first tile."""
        # 行级读写走基址 + 单层偏移；基址在 vec 区外构造。
        max_base = self.state_base(self.sm_max_tb, m_axis_triple)
        sum_base = self.state_base(self.sm_sum_tb, m_axis_triple)

        p_ub = self.p_slot(p_slot)
        tile_n = self.tile_n
        # 布局常量：16×16 分形，带内 256 元素。
        m0, s_m1, s_m0 = 16, 256, 16
        block_stride = self.p_block_stride
        # The score buffer keeps a fixed TILE_S2 row pitch for dynamic N.
        rows = self.tile_vec_m
        src_row_stride = self.tile_n
        # V 写 p_ub，MTE3 在 store_p 读。
        self.mtx_p.lock(pipe=tla.arch.VECTOR)
        with tla.vec.func(mode="simd"):
            pass_a_row = self._pass_a_row  # tla.range 体内不得调用 host 对象方法
            for row in tla.range(rows):  # 运行期行循环
                full, _ = tla.update_mask(VL_T, dtype=tla.Float32)
                half0_mask, _ = tla.update_mask(actual_n, dtype=tla.Float32)
                half1_mask, _ = tla.update_mask(
                    _clamp_nonneg(actual_n - VL_T), dtype=tla.Float32
                )
                pass_a_row(
                    qk_ch,
                    scale,
                    max_base,
                    row,
                    src_row_stride,
                    VL_T,
                    half0_mask,
                    half1_mask,
                    full,
                )
        # pass A 与 fold 各自一个 SIMD 区，屏障放在 fold 区的开头
        # （`local_mem_bar` 只允许出现在 `vec.func` 内）。同一区内「先 vstore 行
        # max、再 vload 广播读回」实测会被重排：fold 会读到
        # `_cross_init_state` 的初值（实测 max 恒 0、sum=Σexp(S)）。
        with tla.vec.func(mode="simd"):
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
            self._softmax_fold_loop(
                qk_ch,
                p_ub,
                max_base,
                sum_base,
                actual_n,
                rows,
                src_row_stride,
                self.tile_n,
                m0,
                s_m1,
                s_m0,
                block_stride,
            )
        self.mtx_p.unlock(pipe=tla.arch.VECTOR)

    # --- softmax_rest：后续 n-tile 的 softmax ----------------------------------------------
    @tla.jit
    def softmax_rest(self, qk_ch, actual_n, scale, m_axis_triple, tile_triple, p_slot):
        """Non-first n-tile: rescale running max/sum, P = exp(S - new_max)."""
        sm_max = self.state_slot(self.sm_max_tb, m_axis_triple)
        sm_sum = self.state_slot(self.sm_sum_tb, m_axis_triple)
        sm_exp = self.state_slot(self.sm_exp_tb, tile_triple)

        p_ub = self.p_slot(p_slot)
        tile_n = self.tile_n
        # 布局常量：16×16 分形，带内 256 元素。
        m0, s_m1, s_m0 = 16, 256, 16
        block_stride = self.p_block_stride
        # The score buffer keeps a fixed TILE_S2 row pitch for dynamic N.
        rows = self.tile_vec_m
        src_row_stride = self.tile_n
        # V 写 p_ub，MTE3 在 store_p 读。
        self.mtx_p.lock(pipe=tla.arch.VECTOR)
        with tla.vec.func(mode="simd"):
            pass_a_row = self._pass_a_row  # tla.range 体内不得调用 host 对象方法
            for row in tla.range(rows):  # 运行期行循环
                full, _ = tla.update_mask(VL_T, dtype=tla.Float32)
                half0_mask, _ = tla.update_mask(actual_n, dtype=tla.Float32)
                half1_mask, _ = tla.update_mask(
                    _clamp_nonneg(actual_n - VL_T), dtype=tla.Float32
                )
                pass_a_row(
                    qk_ch,
                    scale,
                    self.tmp_new_max,
                    row,
                    src_row_stride,
                    VL_T,
                    half0_mask,
                    half1_mask,
                    full,
                )
        # 三段拆区（见 `softmax_first`）：pass A / max 合并 / fold，
        # 屏障放在后两段的区开头。
        with tla.vec.func(mode="simd"):
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
            rowmask, _ = tla.update_mask(rows, dtype=tla.Float32)
            nm = tla.max(
                tla.tile_view(sm_max, tla.make_shape(rows), tla.make_coord(0)).load(),
                tla.tile_view(
                    self.tmp_new_max, tla.make_shape(rows), tla.make_coord(0)
                ).load(),
                mask=rowmask,
            )
            tla.tile_view(
                self.tmp_new_max, tla.make_shape(rows), tla.make_coord(0)
            ).store(nm, mask=rowmask)
        with tla.vec.func(mode="simd"):
            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
            rowmask, _ = tla.update_mask(rows, dtype=tla.Float32)
            self._softmax_fold_loop(
                qk_ch,
                p_ub,
                self.tmp_new_max,
                self.tmp_sum,
                actual_n,
                rows,
                src_row_stride,
                self.tile_n,
                m0,
                s_m1,
                s_m0,
                block_stride,
            )
            self._softmax_rest_tail(sm_max, sm_sum, sm_exp, rowmask)
        self.mtx_p.unlock(pipe=tla.arch.VECTOR)

    # --- store_p：p_ub → L1 的逐带写出 ---------------------------------------------------
    @tla.jit
    def store_p(self, p_l1_ch, p_slot):
        """p_ub -> the explicit per-AIV M tile of the shared L1 slot.

        AIV 分片按行区间 [subblock*AIV_ROW_CAPACITY, +AIV_ROW_CAPACITY) 表达，
        并写进目标视图（同一数据分区，显式行偏移而非引擎参数）。

        在 (G,TILE_S2) zN 父块里，这半块**不是**连续区间、也不能用
        `tile_view(coord(sub*32,0))` 表达：`tile_view` 会按半块自身 32 行的分形
        重算 stride，落成非法 L1 地址（实测 AIC 全核 "set and wait do not
        match" + l1 error info 0x1459…）。

        zN 物理地址 = (i%16)*16 + (i//16)*256 + (j%16) + (j//16)*(G//16)*256，
        因此按 16 列一带逐带写：带 j1 里本 AIV 的 32 行正好是父块中连续 512 个
        元素（两个分形），起点 `sub*512 + j1*(G//16)*256`。逐带 (32,16) zN 目标
        与整块 MTE2 拷贝逐值一致（实测 maxdiff=0）。
        """
        rows = self.tile_vec_m
        band_elems = rows * 16  # 每带数据元素：32*16 = 512
        dst_band = (self.tile_m // 16) * 256  # L1 zN 带距：64/16*256 = 1024
        # 源布局为带主序（带距 528），store_p 逐带 8 条 zN->zN copy，
        # 每条 1 KiB burst。
        slot_base = self.p_ub.ptr + p_slot * self.p_slot_elems
        # MTE3 读 V 写好的 p_ub。
        self.mtx_p.lock(pipe=tla.arch.MTE3)
        for j1 in tla.range_constexpr(self.tile_n // 16):
            tla.copy(
                tla.make_tensor(
                    p_l1_ch.ptr + self.subblock_idx * band_elems + j1 * dst_band,
                    tla.make_layout_with_tag((rows, 16), self.dtype_16, tla.arch.zN),
                ),
                tla.make_tensor(
                    slot_base + j1 * self.p_band,
                    tla.make_layout_with_tag((rows, 16), self.dtype_16, tla.arch.zN),
                ),
            )
        self.mtx_p.unlock(pipe=tla.arch.MTE3)

    # --- init_o：首块 PV 初始化输出累加器 --------------------------------------------------
    @tla.jit
    def init_o(self, pv_ch):
        """First PV tile: initialize the FP32 output accumulator directly."""
        tla.copy(
            tla.tile_view(
                self.res_o,
                tla.make_shape(self.tile_vec_m, self.tile_d),
                tla.make_coord(0, 0),
            ),
            pv_ch,
        )

    # --- update_o：非首块 PV 的在线累加 ---------------------------------------------------
    @tla.jit
    def update_o(self, pv_ch, sm_exp_buf):
        """Non-first PV tile: res_o = res_o * exp(old_max-new_max) + P_i @ V_i."""
        tile_d = self.tile_d
        res_o = tla.tile_view(
            self.res_o,
            tla.make_shape(self.tile_vec_m, tile_d),
            tla.make_coord(0, 0),
        )
        pv = tla.tile_view(
            pv_ch, tla.make_shape(self.tile_vec_m, tile_d), tla.make_coord(0, 0)
        )
        with tla.vec.func(mode="simd"):
            # 行循环保留为运行期循环（`for (i<32)`），列方向由 extern 内展开。
            for row in tla.range(self.tile_vec_m):
                exp_b = tla.tile_view(
                    sm_exp_buf, tla.make_shape(1), tla.make_coord(row)
                ).load(
                    tla.params.NormalLoadParams(
                        load_dist=tla.params.LoadDist.DIST_BRC_B32
                    )
                )
                # 整行交给 extern，内部每 64 lane 一条 `asc_madd`（单次舍入），
                # 替代 `tla.mul` + `tla.add` 两条。res_o/pv 的视图 coord 为 (0,0)，
                # 故 `.ptr` 即缓冲基址（tile_view 的 ptr 不折叠视图偏移，
                # 见 TlaLowerPtrPass.cpp:30-40），行偏移在此显式加上。
                update_o_row(
                    res_o.ptr + row * tile_d,
                    pv.ptr + row * tile_d,
                    exp_b,
                    tile_d,
                )

    # --- _softmax_fold_loop：逐行写非归一化权重与指数和 -----------------------------------
    @tla.jit
    def _softmax_fold_loop(
        self,
        qk_ch,
        p_ub,
        max_base,
        sum_base,
        actual_n,
        rows,
        src_row_stride,
        tile_n,
        m0,
        s_m1,
        s_m0,
        block_stride,
    ):
        """Write unnormalized weight rows and their exponential sums."""
        tile_f16 = self.dtype_16
        fold_row = self._softmax_fold_row  # tla.range 体内不得调用 host 对象方法
        for row in tla.range(rows):  # 运行期行循环
            full, _ = tla.update_mask(VL_T, dtype=tla.Float32)
            b16_full, _ = tla.update_mask(self.tile_n, dtype=tile_f16)
            ve_mask, _ = tla.update_mask((actual_n + 1) // 2, dtype=tla.Float32)
            vo_mask, _ = tla.update_mask(actual_n // 2, dtype=tla.Float32)
            b16, _ = tla.update_mask(actual_n, dtype=tile_f16)
            base = row * src_row_stride
            nz_off = (row // m0) * s_m1 + (row % m0) * s_m0
            ve, vo, mx = fold_row(
                qk_ch,
                p_ub,
                base,
                nz_off,
                max_base,
                row,
                ve_mask,
                vo_mask,
                b16,
                b16_full,
                block_stride,
            )
            # vreduce_sum(vadd(ve, vo)) → reduce(ADD)
            rsum = tla.add(ve, vo, mask=full).reduce(tla.ReductionOp.ADD, mask=ve_mask)
            tla.tile_view(sum_base, tla.make_shape(1), tla.make_coord(row)).store(
                rsum,
                params=tla.params.NormalStoreParams(
                    store_dist=tla.params.StoreDist.DIST_FIRST_ELEMENT_B32
                ),
            )
        tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)

    # --- _softmax_fold_row：单行 exp / 打包 / 指数和计算 ----------------------------------
    @tla.jit
    def _softmax_fold_row(
        self,
        qk_ch,
        p_ub,
        base,
        nz_off,
        max_base,
        row,
        ve_mask,
        vo_mask,
        b16,
        b16_full,
        block_stride,
    ):
        """Store unnormalized NZ weights and return the FP32 exponential halves."""
        mx = tla.tile_view(max_base, tla.make_shape(1), tla.make_coord(row)).load(
            tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_BRC_B32)
        )
        ve, vo = _vec_view(qk_ch, row, 0, VL_T).load(
            tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_DINTLV_B32)
        )
        # vexp_sub(x, mx) = exp(x - mx)（拆成 sub + exp 两条 tla 运算）
        ve = tla.exp(tla.sub(ve, mx, mask=ve_mask), mask=ve_mask)
        vo = tla.exp(tla.sub(vo, mx, mask=vo_mask), mask=vo_mask)
        he = ve.to(self.dtype_16, params=_CAST_EVEN, mask=ve_mask)
        ho = vo.to(self.dtype_16, params=_CAST_ODD, mask=vo_mask)
        merged = tla.bitwise_or(he, ho, mask=b16)
        # 一行 128 个 bf16 = 8 个 32B DataBlock，块距 33（528 元素），
        # 每带 1,024 B burst（对应 asc_storealign(p + t, v, 33, 0, mask_b16(128))）。
        _vec_view(p_ub, 0, nz_off, self.tile_n).store(
            merged,
            mask=b16_full,
            params=tla.params.BlockStoreParams(block_stride=block_stride),
        )
        return ve, vo, mx

    # --- _softmax_rest_tail：在线 softmax running state 更新 -------------------------------
    @tla.jit
    def _softmax_rest_tail(self, sm_max, sm_sum, sm_exp, rowmask):
        """Online-softmax running-state update."""
        rows = self.tile_vec_m
        old_max = tla.tile_view(sm_max, tla.make_shape(rows), tla.make_coord(0)).load()
        new_max = tla.tile_view(
            self.tmp_new_max, tla.make_shape(rows), tla.make_coord(0)
        ).load()
        se = tla.exp(
            tla.sub(old_max, new_max, mask=rowmask), mask=rowmask
        )  # exp(old-new)
        tla.tile_view(sm_exp, tla.make_shape(rows), tla.make_coord(0)).store(
            se, mask=rowmask
        )
        tla.tile_view(sm_max, tla.make_shape(rows), tla.make_coord(0)).store(
            new_max, mask=rowmask
        )
        old_sum = tla.tile_view(sm_sum, tla.make_shape(rows), tla.make_coord(0)).load()
        new_sum = tla.tile_view(
            self.tmp_sum, tla.make_shape(rows), tla.make_coord(0)
        ).load()
        ss = tla.add(tla.mul(old_sum, se, mask=rowmask), new_sum, mask=rowmask)
        tla.tile_view(sm_sum, tla.make_shape(rows), tla.make_coord(0)).store(
            ss, mask=rowmask
        )


# ---------------------------------------------------------------------------------------------
# `_SFAOnlineSoftmaxVector`（尾块安全 max + 输出归一化）
# ---------------------------------------------------------------------------------------------
class _SFAOnlineSoftmaxVector(_StableOnlineSoftmaxVector):
    """Online softmax with tail-safe maxima for signed scores."""

    # --- finish_o_cast：输出归一化与类型转换 ----------------------------------------------
    @tla.jit
    def finish_o_cast(
        self, pv_ch, sm_sum_buf, output_heads, output_head, sm_exp_buf=None
    ):
        """Normalize the final PV result and cast it to the output type."""
        tile_d = self.tile_d
        res_o = tla.tile_view(
            self.res_o,
            tla.make_shape(self.tile_vec_m, tile_d),
            tla.make_coord(0, 0),
        )
        pv = tla.tile_view(
            pv_ch, tla.make_shape(self.tile_vec_m, tile_d), tla.make_coord(0, 0)
        )
        o_ub = tla.tile_view(
            self.o_ub, tla.make_shape(self.tile_vec_m, tile_d), tla.make_coord(0, 0)
        )
        # V 写 o_ub、MTE3 随后读它搬 GM；锁句柄先提为局部名。
        mtx_o = self.mtx_o
        # heads<32 的档位用运行期行循环截断；其余档位保留静态满行循环。
        # 实测：64/128 头档位若也换成运行期循环会稳定慢约 4%（运行期循环
        # 抑制了 SIMD 体展开），而收益只来自 heads<32 的小例，因此按静态
        # head_count 二选一。
        head_count = self.head_count
        subblock_idx = self.subblock_idx
        if tla.const_expr(head_count < self.tile_vec_m):
            nrows = head_count - subblock_idx * self.tile_vec_m
            if nrows > self.tile_vec_m:
                nrows = tla.as_numeric(self.tile_vec_m)
            if nrows < 0:
                nrows = tla.as_numeric(0)
            mtx_o.lock(pipe=tla.arch.VECTOR)
            with tla.vec.func(mode="simd"):
                full, _ = tla.update_mask(VL_T, dtype=tla.Float32)
                b16_full, _ = tla.update_mask(2 * VL_T, dtype=self.dtype_16)
                zero = tla.full(0.0, tla.Float32, mask=full)
                one = tla.full(1.0, tla.Float32, mask=full)
                for row in tla.range(nrows):  # 只跑本 AIV 的有效 head 行
                    sum_b = tla.tile_view(
                        sm_sum_buf, tla.make_shape(1), tla.make_coord(row)
                    ).load(
                        tla.params.NormalLoadParams(
                            load_dist=tla.params.LoadDist.DIST_BRC_B32
                        )
                    )
                    is_zero = tla.cmp(sum_b, 0.0, "eq", mask=full)
                    # vselect(one, sum_b, cond_mask=is_zero)：cond 为真取第一操作数
                    safe_sum = tla.where(is_zero, one, sum_b)
                    if tla.const_expr(sm_exp_buf is not None):
                        exp_b = tla.tile_view(
                            sm_exp_buf, tla.make_shape(1), tla.make_coord(row)
                        ).load(
                            tla.params.NormalLoadParams(
                                load_dist=tla.params.LoadDist.DIST_BRC_B32
                            )
                        )
                    for col in tla.range_constexpr(0, tile_d, 2 * VL_T):
                        # 逐 64 列 `vload` + `vstore_pack(b32_to_b16)` 的落位等价物：
                        # f32->f16 是 2x 窄化，单个 64 lane 结果的落点由 reg_slot 决定
                        # （ZERO=偶数位、ONE=奇数位），因此先按 DINTLV 取同一 128 列
                        # 跨度的偶数位/奇数位两半，再用 ZERO/ONE 各自归位后合并
                        # （与 `_softmax_fold_row` 的 p_ub 打包同一形态）。
                        cur_e, cur_o = _vec_view(pv, row, col, VL_T).load(
                            tla.params.NormalLoadParams(
                                load_dist=tla.params.LoadDist.DIST_DINTLV_B32
                            )
                        )
                        if tla.const_expr(sm_exp_buf is not None):
                            pre_e, pre_o = _vec_view(res_o, row, col, VL_T).load(
                                tla.params.NormalLoadParams(
                                    load_dist=tla.params.LoadDist.DIST_DINTLV_B32
                                )
                            )
                            # vmadd(pre, exp_b, cur) = pre * exp_b + cur
                            cur_e = tla.add(
                                tla.mul(pre_e, exp_b, mask=full), cur_e, mask=full
                            )
                            cur_o = tla.add(
                                tla.mul(pre_o, exp_b, mask=full), cur_o, mask=full
                            )
                        val_e = tla.div(cur_e, safe_sum, mask=full)
                        val_e = tla.where(is_zero, zero, val_e)
                        lo = val_e.to(self.dtype_16, params=_CAST_EVEN, mask=full)
                        val_o = tla.div(cur_o, safe_sum, mask=full)
                        val_o = tla.where(is_zero, zero, val_o)
                        hi = val_o.to(self.dtype_16, params=_CAST_ODD, mask=full)
                        packed = tla.bitwise_or(lo, hi, mask=b16_full)
                        _vec_view(o_ub, row, col, 2 * VL_T).store(packed, mask=b16_full)

            # Keep the UB producer and GM consumer in the same JIT region.
            # 搬运行数是编译期展开的静态 head 数：`_output_tile(...,
            # rows=head_count)` 传入。tla 的 `origin_shape[0]` 是运行时 Int32，
            # 若改成动态 `tla.range`，MTE3 拷贝会在部分 query 上整段不发射
            # （实测同一 case 两次运行的 NaN 行集合不同，attn 按 query 随机缺写），
            # 因此这里回到静态循环。
            mtx_o.unlock(pipe=tla.arch.VECTOR)
        else:
            mtx_o.lock(pipe=tla.arch.VECTOR)
            with tla.vec.func(mode="simd"):
                full, _ = tla.update_mask(VL_T, dtype=tla.Float32)
                b16_full, _ = tla.update_mask(2 * VL_T, dtype=self.dtype_16)
                zero = tla.full(0.0, tla.Float32, mask=full)
                one = tla.full(1.0, tla.Float32, mask=full)
                for row in tla.range(self.tile_vec_m):  # 静态满行循环
                    sum_b = tla.tile_view(
                        sm_sum_buf, tla.make_shape(1), tla.make_coord(row)
                    ).load(
                        tla.params.NormalLoadParams(
                            load_dist=tla.params.LoadDist.DIST_BRC_B32
                        )
                    )
                    is_zero = tla.cmp(sum_b, 0.0, "eq", mask=full)
                    # vselect(one, sum_b, cond_mask=is_zero)：cond 为真取第一操作数
                    safe_sum = tla.where(is_zero, one, sum_b)
                    if tla.const_expr(sm_exp_buf is not None):
                        exp_b = tla.tile_view(
                            sm_exp_buf, tla.make_shape(1), tla.make_coord(row)
                        ).load(
                            tla.params.NormalLoadParams(
                                load_dist=tla.params.LoadDist.DIST_BRC_B32
                            )
                        )
                    for col in tla.range_constexpr(0, tile_d, 2 * VL_T):
                        # 逐 64 列 `vload` + `vstore_pack(b32_to_b16)` 的落位等价物：
                        # f32->f16 是 2x 窄化，单个 64 lane 结果的落点由 reg_slot 决定
                        # （ZERO=偶数位、ONE=奇数位），因此先按 DINTLV 取同一 128 列
                        # 跨度的偶数位/奇数位两半，再用 ZERO/ONE 各自归位后合并
                        # （与 `_softmax_fold_row` 的 p_ub 打包同一形态）。
                        cur_e, cur_o = _vec_view(pv, row, col, VL_T).load(
                            tla.params.NormalLoadParams(
                                load_dist=tla.params.LoadDist.DIST_DINTLV_B32
                            )
                        )
                        if tla.const_expr(sm_exp_buf is not None):
                            pre_e, pre_o = _vec_view(res_o, row, col, VL_T).load(
                                tla.params.NormalLoadParams(
                                    load_dist=tla.params.LoadDist.DIST_DINTLV_B32
                                )
                            )
                            # vmadd(pre, exp_b, cur) = pre * exp_b + cur
                            cur_e = tla.add(
                                tla.mul(pre_e, exp_b, mask=full), cur_e, mask=full
                            )
                            cur_o = tla.add(
                                tla.mul(pre_o, exp_b, mask=full), cur_o, mask=full
                            )
                        val_e = tla.div(cur_e, safe_sum, mask=full)
                        val_e = tla.where(is_zero, zero, val_e)
                        lo = val_e.to(self.dtype_16, params=_CAST_EVEN, mask=full)
                        val_o = tla.div(cur_o, safe_sum, mask=full)
                        val_o = tla.where(is_zero, zero, val_o)
                        hi = val_o.to(self.dtype_16, params=_CAST_ODD, mask=full)
                        packed = tla.bitwise_or(lo, hi, mask=b16_full)
                        _vec_view(o_ub, row, col, 2 * VL_T).store(packed, mask=b16_full)

            # Keep the UB producer and GM consumer in the same JIT region.
            # 搬运行数是编译期展开的静态 head 数：`_output_tile(...,
            # rows=head_count)` 传入。tla 的 `origin_shape[0]` 是运行时 Int32，
            # 若改成动态 `tla.range`，MTE3 拷贝会在部分 query 上整段不发射
            # （实测同一 case 两次运行的 NaN 行集合不同，attn 按 query 随机缺写），
            # 因此这里回到静态循环。
            mtx_o.unlock(pipe=tla.arch.VECTOR)
        head_rows = self.head_count
        group_rows = self.tile_m
        # 动态 if 内不得出现 `self.`（前端限制）：先提到本地名字。
        vec_m = self.tile_vec_m
        subblock_idx = self.subblock_idx
        o_ub = self.o_ub
        for part in tla.range_constexpr((head_rows + group_rows - 1) // group_rows):
            for sub in tla.range_constexpr(2):
                rows = head_rows - part * group_rows - sub * vec_m
                if tla.const_expr(rows > vec_m):
                    rows = vec_m
                if tla.const_expr(rows > 0):
                    row_start = part * group_rows + sub * vec_m
                    if output_head == part * group_rows and subblock_idx == sub:
                        # 整块 (rows, tile_d) 搬运在本 case 上会静默丢失（多数
                        # query 的 GM 行保持未写），逐行搬运（与
                        # `_write_empty_query` 同形态）实测正常。按 32 行一条
                        # UB→GM 拷贝（get_tile 元素偏移，块行数静态）。
                        nchunks = (rows + vec_m - 1) // vec_m
                        for c in tla.range_constexpr(nchunks):
                            n = rows - c * vec_m
                            if tla.const_expr(n > vec_m):
                                n = vec_m
                            dst = tla.get_tile(
                                output_heads,
                                tla.make_coord(row_start + c * vec_m, 0),
                                tla.make_shape(n, tile_d),
                            )
                            src = tla.get_tile(
                                o_ub,
                                tla.make_coord(c * vec_m, 0),
                                tla.make_shape(n, tile_d),
                            )
                            mtx_o.lock(pipe=tla.arch.MTE3)
                            tla.copy(dst, src)
                            mtx_o.unlock(pipe=tla.arch.MTE3)

    # --- _pass_a_row：单行缩放与尾块安全 max 归约 ------------------------------------------
    @tla.jit
    def _pass_a_row(
        self,
        qk_ch,
        scale,
        max_base,
        row,
        row_stride,
        vector_lanes,
        half0_mask,
        half1_mask,
        full_mask,
    ):
        """Scale one score row and reduce the tail-safe running max."""
        base = row * row_stride
        v0 = tla.mul(_vec_view(qk_ch, row, 0, VL_T).load(), scale, mask=half0_mask)
        v1 = tla.mul(
            _vec_view(qk_ch, row, vector_lanes, VL_T).load(),
            scale,
            mask=half1_mask,
        )
        _vec_view(qk_ch, row, 0, VL_T).store(v0, mask=half0_mask)
        _vec_view(qk_ch, row, vector_lanes, VL_T).store(v1, mask=half1_mask)
        neg_inf = tla.full(SOFTMAX_NEG_INF, tla.Float32, mask=full_mask)
        max0 = tla.where(half0_mask, v0, neg_inf)
        max1 = tla.where(half1_mask, v1, neg_inf)
        rmax = tla.max(max0, max1, mask=full_mask).reduce(
            tla.ReductionOp.MAX, mask=full_mask
        )
        # Initialize an empty first tile without replacing later running maxima.
        if tla.const_expr(max_base is not self.tmp_new_max):
            # 分支前后操作数道数必须一致：`rmax` 是 reduce 出来的 1 道标量，
            # 这里也用 1 道掩码，避免 `where` 把结果广播回 64 道后在
            # `DIST_FIRST_ELEMENT_B32` 落盘时丢值（实测：非 0 槽的 running max
            # 写不进槽内，fold 读到的是槽里的残留 UB 垃圾）。
            zeros = tla.full(0.0, tla.Float32, mask=full_mask)
            all_masked = tla.cmp(rmax, EMPTY_SOFTMAX_THRESHOLD, "le", mask=full_mask)
            rmax = tla.where(all_masked, zeros, rmax)
        tla.tile_view(max_base, tla.make_shape(1), tla.make_coord(row)).store(
            rmax,
            params=tla.params.NormalStoreParams(
                store_dist=tla.params.StoreDist.DIST_FIRST_ELEMENT_B32
            ),
        )


# ---------------------------------------------------------------------------------------------
# `_SparseMatmul`（Cube 阶段与 L1/L0 缓冲）
# ---------------------------------------------------------------------------------------------
def _l1_tensor(shape, dtype, tag=None):
    ptr = tla.allocate(shape, dtype, tla.AddressSpace.l1, 512)
    if tag is None:
        layout = tla.make_layout(
            tla.make_shape(shape[0], shape[1]), tla.make_stride(shape[1], 1)
        )
    else:
        layout = tla.make_layout_with_tag(shape, dtype, layoutTag=tag)
    return tla.make_tensor(ptr, layout)


def _l0_ptr(shape, dtype, space, align):
    """分配 L0 指针；同形布局视图由调用方按需另行构造。"""
    return tla.allocate(shape, dtype, space, align)


class _SparseMatmul:
    """Cube matmul stages and their L1/L0 buffers."""

    def __init__(self, group_size, dtype_16, has_rope):
        self.group_size = group_size
        self.has_rope = has_rope

        # nd2nz / fixpipe：tla 侧由 layout tag 与目标视图承担。

        # QK 每槽 K；PV 最后一次 L1 读之后回收（depth=PIPELINE_SLOT_COUNT）。
        self.right_root = [
            _l1_tensor((TILE_S2, DIM_NOPE), dtype_16, tla.arch.zN)
            for _ in range(PIPELINE_SLOT_COUNT)
        ]
        # k_rope_l1：KV 的 RoPE 列（feature 512..575）单独进一块 16 KiB L1。
        # QK 的 RoPE 尾巴从这块 L1 取 k^T（见 compute_qk）。
        self.k_rope_l1 = None
        if tla.const_expr(has_rope):
            self.k_rope_l1 = _l1_tensor((TILE_S2, DIM_ROPE), dtype_16, tla.arch.zN)
        self.q_nope_l1 = _l1_tensor((group_size, DIM_NOPE), dtype_16, tla.arch.zN)
        self.q_rope_l1 = None
        if tla.const_expr(has_rope):
            self.q_rope_l1 = _l1_tensor((group_size, DIM_ROPE), dtype_16, tla.arch.zN)

        self.dtype_16 = dtype_16
        # 动态槽号只能选 ptr：跨 `if` 传张量会丢掉可解析的 backing ptr
        # （实测报 `tensor_ptr source must resolve through ...`），
        # 因此列表里只存 ptr，张量在取用时用 make_tensor 重建。
        self.right_ptrs = [tile.ptr for tile in self.right_root]

        # QK 的 B 操作数是 k^T，而 `tla.mmad` 要求 rhs 为 (K,N)。L1→L0B 的
        # 就地转置通过把同一块 L1 KV 字节按 nZ 重新解释成 (dim, token) 视图再
        # 拷到 L0B 来完成，得到的就是 k^T（实测 4 段 × 5 种 valid_n 全部
        # max_abs 0.0，含 17/33 这种非 16 倍数）。首维偏移用
        # `ptr + feature*TILE_S2` 施加（nZ 下一个 dim 组的跨度是 TILE_S2*16 元素）。
        # 因此不需要 GM→L1 的 transpose 暂存。
        self.mtx_kt = tla.mutex("l1_kt", id=12)
        # KV L1 装载按 128 列一段拆成 4 次 MTE2 拷贝，每段一把 QK 读信用。
        # 若 `mtx_kt`（MTE1 侧）的锁跨 4 段，必须等整块 128 KiB + RoPE 拷完
        # 才放行第一条 mmad；拆段后第 0 段的 mmad 只等第 0 段（32 KiB）落 L1，
        # 其余 3 段与 4 条 QK mmad 重叠。`mtx_kt` 只承担 RoPE 段的信用，
        # `mtx_kv` 承担 PV 的整块 V 信用。
        self.mtx_ktseg = [
            tla.mutex("l1_ktseg0", id=15),
            tla.mutex("l1_ktseg1", id=16),
            tla.mutex("l1_ktseg2", id=17),
            tla.mutex("l1_ktseg3", id=18),
        ]

        # L0A/L0B/L0C double buffers: 32 KiB / 64 KiB / 256 KiB total.
        self.l0a_ptrs = []
        for _ in range(3):
            ptr = _l0_ptr((group_size, TILE_S2), dtype_16, tla.AddressSpace.l0a, 512)
            self.l0a_ptrs.append(ptr)
        self.l0b_ptrs = []
        for _ in range(2):
            ptr = _l0_ptr((TILE_S2, TILE_S2), dtype_16, tla.AddressSpace.l0b, 512)
            self.l0b_ptrs.append(ptr)
        self.l0c_ptrs = []
        for _ in range(2):
            ptr = _l0_ptr(
                (group_size, DIM_NOPE), tla.Float32, tla.AddressSpace.l0c, 1024
            )
            self.l0c_ptrs.append(ptr)

        # 核内同步：框架的自动同步（`auto_sync="v0"`）对本 kernel 的动态槽
        # 选择不成立（实测 get_buf/rls_buf 计数不配对），改为按缓冲显式 mutex，
        # 用法同 basic_matmul_mutex.py：生产/消费两侧各自在自己流水线上
        # lock/unlock 同一 mutex。
        # mutex id 在 AIC 上取 0..6（AIV 侧取 8..12，避免与 AIC 号段混用）。
        self.mtx_qn = tla.mutex("l1_qn", id=0)
        self.mtx_qr = tla.mutex("l1_qr", id=1)
        self.mtx_kv = tla.mutex("l1_kv", id=2)
        # 同一块 KV L1 有两个读侧：QK 取 k^T、PV 取 v。两侧各用一把 mutex
        # （QK 用 id=12 的 mtx_kt），保证每把 mutex 只有一个生产段和一个消费段，
        # 不会因为两个读者交替 lock/unlock 把 mutex 配对错位。
        # mtx_kr 保留自己的 id 号段；RoPE 块与 NoPE 块在同一段 MTE2 里一起
        # 发布（mtx_kt），不再单独握手。
        self.mtx_kr = tla.mutex("l1_kr", id=3)
        # l0a/l0b 每段轮转到下一块，一个 matmul 的 4 段因此是 0/1 交替。
        # 这里用两块缓冲 + 每块一把 mutex 实现同一结构；索引在展开循环里是
        # Python 常量，可直接下标。
        self.mtx_l0a = [
            tla.mutex("l0a0", id=4),
            tla.mutex("l0a1", id=5),
            tla.mutex("l0a2", id=14),
        ]
        self.mtx_l0b = [tla.mutex("l0b0", id=6), tla.mutex("l0b1", id=7)]
        # l0c 双槽：FixPipe 读完该槽后才把它归还给写侧。这里用一对核内 mutex
        # 实现这条握手：
        #   写侧  lock(CUBE) -> mmad 链 -> unlock(CUBE)
        #   FixPipe lock(FIX) -> copy -> unlock(FIX)
        # 不带 unit flag 时 `matmul_add_ub` 的 ENABLE_UNIT_FLAG=False 分支就是
        # 这条形态（CUBE→FIX 显式 mutex + SPLIT_M copy）。
        self.mtx_l0c = [tla.mutex("l0c0", id=19), tla.mutex("l0c1", id=20)]

    @tla.jit
    def _pick2(self, items, slot):
        """动态槽号选 2 槽张量：tla 的 Python 列表不能用动态下标，改 if 选择。"""
        out = items[0]
        if slot != 0:
            out = items[1]
        return out

    @tla.jit
    def _l0c_wait_free(self, slot):
        """写侧等该 L0C 槽空闲（FIX 已读完上一任）。"""
        if slot == 0:
            self.mtx_l0c[0].lock(pipe=tla.arch.CUBE)
        else:
            self.mtx_l0c[1].lock(pipe=tla.arch.CUBE)

    @tla.jit
    def _l0c_signal_ready(self, slot):
        """写侧交棒：该 L0C 槽已写完。"""
        if slot == 0:
            self.mtx_l0c[0].unlock(pipe=tla.arch.CUBE)
        else:
            self.mtx_l0c[1].unlock(pipe=tla.arch.CUBE)

    @tla.jit
    def _l0c_wait_ready(self, slot):
        """FixPipe 侧等该 L0C 槽写完。"""
        if slot == 0:
            self.mtx_l0c[0].lock(pipe=tla.arch.FIX)
        else:
            self.mtx_l0c[1].lock(pipe=tla.arch.FIX)

    @tla.jit
    def _l0c_signal_free(self, slot):
        """FixPipe 侧交棒：该 L0C 槽已读完。"""
        if slot == 0:
            self.mtx_l0c[0].unlock(pipe=tla.arch.FIX)
        else:
            self.mtx_l0c[1].unlock(pipe=tla.arch.FIX)

    @tla.jit
    def _pick3(self, items, slot):
        out = items[0]
        if slot == 1:
            out = items[1]
        elif slot == 2:
            out = items[2]
        return out

    # --- 槽位张量重建：动态 if 只传 ptr，张量由 make_tensor 重建 -------------------
    def right_tile(self, slot):
        """三槽 L1 环的显式槽位。"""
        return tla.make_tensor(
            self._pick3(self.right_ptrs, slot % PIPELINE_SLOT_COUNT),
            tla.make_layout_with_tag(
                (TILE_S2, DIM_NOPE), self.dtype_16, layoutTag=tla.arch.zN
            ),
        )

    def l0a_tile(self, slot):
        return tla.make_tensor(
            self._pick3(self.l0a_ptrs, slot),
            tla.make_layout_with_tag(
                (self.group_size, TILE_S2), self.dtype_16, layoutTag=tla.arch.zN
            ),
        )

    def l0b_tile(self, slot):
        return tla.make_tensor(
            self._pick2(self.l0b_ptrs, slot),
            tla.make_layout_with_tag(
                (TILE_S2, TILE_S2), self.dtype_16, layoutTag=tla.arch.nZ
            ),
        )

    def l0c_tile(self, slot):
        return tla.make_tensor(
            self._pick2(self.l0c_ptrs, slot),
            tla.make_layout_with_tag(
                (self.group_size, DIM_NOPE), tla.Float32, layoutTag=tla.arch.L0Clayout
            ),
        )

    # load_q：把本 Query 的 Q / Q-RoPE 装入 L1
    def load_q(self, query_token, query_rope_token):
        # MTE2 写 L1，MTE1 在 compute_qk 读同一缓冲。
        self.mtx_qn.lock(pipe=tla.arch.MTE2)
        tla.copy(self.q_nope_l1, query_token)
        self.mtx_qn.unlock(pipe=tla.arch.MTE2)
        if tla.const_expr(self.has_rope):
            self.mtx_qr.lock(pipe=tla.arch.MTE2)
            tla.copy(self.q_rope_l1, query_rope_token)
            self.mtx_qr.unlock(pipe=tla.arch.MTE2)

    # load_sparse_kv：把 KV tile 装入 L1（按 feature 分段）
    def load_sparse_kv(self, workspace_tile, valid_n, slot):
        nope = tla.get_tile(
            workspace_tile,
            tla.make_coord(0, 0),
            tla.make_shape(valid_n, DIM_NOPE),
        )
        dest = self.right_tile(slot)
        # MTE2 写 L1 KV，MTE1 在 compute_qk（k^T 视图）/ compute_pv（v）读。
        # 这里顺带把 RoPE 列（offset=DIM_NOPE）装进 k_rope_l1。
        self.mtx_kv.lock(pipe=tla.arch.MTE2)
        self.mtx_kt.lock(pipe=tla.arch.MTE2)
        # 按 feature 段拆成 4 次拷贝，每段在自己的信用上发布；QK 的第 seg
        # 条 mmad 只需等该段。段宽 128 与 compute_qk 的 feature 步长一致。
        for seg, feature in enumerate((0, 128, 256, 384)):
            self.mtx_ktseg[seg].lock(pipe=tla.arch.MTE2)
            tla.copy(
                tla.get_tile(
                    dest,
                    tla.make_coord(0, feature),
                    tla.make_shape(valid_n, 128),
                ),
                tla.get_tile(
                    workspace_tile,
                    tla.make_coord(0, feature),
                    tla.make_shape(valid_n, 128),
                ),
            )
            self.mtx_ktseg[seg].unlock(pipe=tla.arch.MTE2)
        if tla.const_expr(self.has_rope):
            rope = tla.get_tile(
                workspace_tile,
                tla.make_coord(0, DIM_NOPE),
                tla.make_shape(valid_n, DIM_ROPE),
            )
            tla.copy(self.k_rope_l1, rope)
        # 两个块都落 L1 之后再放 QK 的读侧进去（写侧 unlock 即发布）。
        self.mtx_kt.unlock(pipe=tla.arch.MTE2)
        self.mtx_kv.unlock(pipe=tla.arch.MTE2)
        return dest

    # compute_qk：4 段 NoPE + 可选 RoPE 尾巴的 QK 累加
    def compute_qk(self, right_slot, valid_n, slot):
        """Accumulate four NoPE chunks and an optional 64-feature RoPE tail."""
        m = self.group_size
        # 先取到一块空闲 L0C 槽：显式等该槽上一任被 FixPipe 读完。
        # mutex 初值自带空槽令牌，无需在 tick 循环前预置。
        self._l0c_wait_free(slot)
        score = _bounded_local(self.l0c_tile(slot), (m, valid_n), (m, TILE_S2))
        # L1 retains the TILE_S2 pitch when valid_n is smaller.
        # Q 的 L1 缓冲在一个 Query 的 4 段里被反复读，段与段之间如果解锁，
        # 下一个 Query 的 `load_q`（MTE2）就能在中间把 q_nope_l1 覆盖掉（两个 pipe
        # 只在 mutex 上排队）。这里把 MTE1 侧的读锁扩到整个 4 段，与 load_q 的一次
        # MTE2 加锁严格交替。
        self.mtx_qn.lock(pipe=tla.arch.MTE1)
        # k^T 的读信用按段拆（见 load_sparse_kv）；这里不整块加锁，
        # 每段只等自己那 32 KiB 落 L1，RoPE 尾巴仍用 mtx_kt。
        for seg, feature in enumerate((0, 128, 256, 384)):
            width = 128
            # l0a/l0b 每段轮转一块：4 段依次落在 0/1/0/1。
            # 索引是 Python 常量，可直接下标取 mutex。
            buf = seg % 2
            q = tla.get_tile(
                self.q_nope_l1,
                tla.make_coord(0, feature),
                tla.make_shape(m, width),
            )
            # L0 视图必须携带动态 N（valid_n）：`make_tensor_like` 只继承
            # like 张量的静态 shape，会把 mmad 的 N 钉成 TILE_S2，与只写了
            # valid_n 行的 L1→L0 拷贝对不上（实测 CUBE 报
            # "The set and wait instructions do not match"）。改用与
            # compute_pv 相同的 `_bounded_local(<L0 张量>, ...)` 写法。
            a = _bounded_local(
                self.l0a_tile(buf), (m, width), (self.group_size, TILE_S2)
            )
            # B = k^T（rhs 必须是 (K=width, N=valid_n)）。L1→L0B 的就地转置
            # 通过把同一块 L1 KV 字节按 nZ 重新解释成 (dim, token) 视图完成 ——
            # zN 的 [dim_group][token][16] 与 nZ 的 (dim, token) 是同一份字节 ——
            # 再按 (width, valid_n) 收窄后拷到 L0B，得到的就是 k^T（实测 4 段 ×
            # 5 种 valid_n，含 17/33 这类非 16 倍数，全部通过）。首维偏移用 ptr
            # 步进施加：nZ 下一个 dim 组（16 个 feature）的跨度是 TILE_S2*16 个元素。
            kt_root = tla.make_tensor(
                right_slot.ptr + feature * TILE_S2,
                tla.make_layout_with_tag(
                    (DIM_NOPE, TILE_S2), self.dtype_16, tla.arch.nZ
                ),
            )
            kt = _bounded_local(kt_root, (width, valid_n), (DIM_NOPE, TILE_S2))
            b = tla.make_tensor(
                self._pick2(self.l0b_ptrs, buf),
                tla.make_layout_with_tag((width, valid_n), self.dtype_16, tla.arch.nZ),
            )
            # L1→L0 由 MTE1 写、L0 由 CUBE 读；两侧各自在自己的流水线上加锁。
            self.mtx_l0a[buf].lock(pipe=tla.arch.MTE1)
            tla.copy(a, q)
            self.mtx_l0a[buf].unlock(pipe=tla.arch.MTE1)
            self.mtx_l0b[buf].lock(pipe=tla.arch.MTE1)
            # 本段 K^T 的 L1 数据到位即可下发 L0B 拷贝（不整块等）。
            self.mtx_ktseg[seg].lock(pipe=tla.arch.MTE1)
            tla.copy(b, kt)
            self.mtx_ktseg[seg].unlock(pipe=tla.arch.MTE1)
            self.mtx_l0b[buf].unlock(pipe=tla.arch.MTE1)
            # unit flag：统一置 0b00（与 store_score 的 FixPipe 参数一致）。
            unit = 0b00
            self.mtx_l0a[buf].lock(pipe=tla.arch.CUBE)
            self.mtx_l0b[buf].lock(pipe=tla.arch.CUBE)
            tla.mmad(score, a, b, init_c=(feature == 0), unit_flag=unit)
            self.mtx_l0b[buf].unlock(pipe=tla.arch.CUBE)
            self.mtx_l0a[buf].unlock(pipe=tla.arch.CUBE)
        self.mtx_qn.unlock(pipe=tla.arch.MTE1)
        if tla.const_expr(self.has_rope):
            q = tla.get_tile(
                self.q_rope_l1,
                tla.make_coord(0, 0),
                tla.make_shape(m, DIM_ROPE),
            )
            a = _bounded_local(
                self.l0a_tile(0), (m, DIM_ROPE), (self.group_size, TILE_S2)
            )
            # RoPE 尾巴走同一套：从 k_rope_l1 取 k^T，把这块 16 KiB L1 按 nZ
            # 重解释成 (DIM_ROPE, token) 视图。
            kt_root = tla.make_tensor(
                self.k_rope_l1.ptr,
                tla.make_layout_with_tag(
                    (DIM_ROPE, TILE_S2), self.dtype_16, tla.arch.nZ
                ),
            )
            kt = _bounded_local(kt_root, (DIM_ROPE, valid_n), (DIM_ROPE, TILE_S2))
            b = tla.make_tensor(
                self._pick2(self.l0b_ptrs, 0),
                tla.make_layout_with_tag(
                    (DIM_ROPE, valid_n), self.dtype_16, tla.arch.nZ
                ),
            )
            self.mtx_qr.lock(pipe=tla.arch.MTE1)
            self.mtx_l0a[0].lock(pipe=tla.arch.MTE1)
            tla.copy(a, q)
            self.mtx_l0a[0].unlock(pipe=tla.arch.MTE1)
            self.mtx_qr.unlock(pipe=tla.arch.MTE1)
            self.mtx_l0b[0].lock(pipe=tla.arch.MTE1)
            self.mtx_kt.lock(pipe=tla.arch.MTE1)
            tla.copy(b, kt)
            self.mtx_kt.unlock(pipe=tla.arch.MTE1)
            self.mtx_l0b[0].unlock(pipe=tla.arch.MTE1)
            self.mtx_l0a[0].lock(pipe=tla.arch.CUBE)
            self.mtx_l0b[0].lock(pipe=tla.arch.CUBE)
            tla.mmad(score, a, b, init_c=False, unit_flag=0b00)
            self.mtx_l0b[0].unlock(pipe=tla.arch.CUBE)
            self.mtx_l0a[0].unlock(pipe=tla.arch.CUBE)
        self._l0c_signal_ready(slot)

    # compute_pv：P @ V 的 4 段累加
    def compute_pv(self, right_slot, p_l1, valid_n, slot):
        """[G,valid_n] @ [valid_n,512], retaining static NZ storage pitch."""
        m = self.group_size
        k_dim = valid_n
        p = _bounded_local(p_l1, (m, k_dim), (m, TILE_S2))
        # compute_pv 只写一次 L0A（此后 4 段共用），L0B 则每段写一块。
        # P 由 AIV 的 MTE3 写 L1、AIC 的 MTE1 读：
        # 跨核侧由 p_ready/p_free 负责，这里只加核内 MTE1→CUBE 的 L0 锁。
        # Pick the L0 buffer that continues the 2-slot rotation past the QK
        # produces. QK does 0,1,0,1 plus a rope tail whose buffer alternates:
        # with rope the next index is 1 (L0A[1], L0B 1,0,1,0), without rope
        # (4 produces) it is 0 (L0A[0], L0B 0,1,0,1).
        # P gets a dedicated third L0A buffer that QK never produces into
        # (L0A has 64 KiB; QK uses 2x16 KiB, the third 16 KiB is free). The L0B
        # 2-slot rotation keeps the same phase: rope -> 1,0,1,0; no rope -> 0,1,0,1.
        a_buf = 2
        l0b_phase = 1 if self.has_rope else 0
        a = _bounded_local(self.l0a_tile(a_buf), (m, k_dim), (m, TILE_S2))
        self._l0c_wait_free(slot)
        self.mtx_l0a[a_buf].lock(pipe=tla.arch.MTE1)
        tla.copy(a, p)
        self.mtx_l0a[a_buf].unlock(pipe=tla.arch.MTE1)
        self.mtx_l0a[a_buf].lock(pipe=tla.arch.CUBE)
        pv_segs = (0, 128, 256, 384)
        # V 的 L1 槽被 4 段共用，段间解锁会让下一块 tile 的 MTE2 写入在
        # 中途覆盖它（实测 q0 首块对得上、后续 query 的 PV 逐渐退化成无关数据）。
        # 把 MTE1 侧读锁扩到整个 4 段，与 load_sparse_kv 的一次 MTE2 写严格交替。
        self.mtx_kv.lock(pipe=tla.arch.MTE1)
        for seg, feature in enumerate(pv_segs):
            # L0B keeps the same rotation phase (decoupled from the P buffer).
            buf = (1 - (seg % 2)) if l0b_phase == 1 else (seg % 2)
            v_root = tla.get_tile(
                right_slot,
                tla.make_coord(0, feature),
                tla.make_shape(TILE_S2, TILE_S2),
            )
            v = _bounded_local(v_root, (k_dim, TILE_S2), (TILE_S2, TILE_S2))
            b = _bounded_local(self.l0b_tile(buf), (k_dim, TILE_S2), (TILE_S2, TILE_S2))
            # L1→L0B 的转置由 L0B 的 nZ tag 承担。
            self.mtx_l0b[buf].lock(pipe=tla.arch.MTE1)
            tla.copy(b, v)
            self.mtx_l0b[buf].unlock(pipe=tla.arch.MTE1)
            # 4 段结果写进宽 (TILE_M,DIM_NOPE) L0C 的 4 个独立块（每块
            # (TILE_M,TILE_S2)）。在该缓冲上用列坐标取子块会被硬件判成非法
            # CUBE 指令（AIC errcode 11 / "set and wait do not match"；
            # 实测按整块访问正常），因此按 32 KiB 段偏移重建张量。
            seg_idx = feature // TILE_S2
            dst = tla.make_tensor(
                self._pick2(self.l0c_ptrs, slot) + seg_idx * (m * TILE_S2),
                tla.make_layout_with_tag((m, TILE_S2), tla.Float32, tla.arch.L0Clayout),
            )
            # 与 compute_qk 相同：统一 unit_flag=0b00。
            unit = 0b00
            self.mtx_l0b[buf].lock(pipe=tla.arch.CUBE)
            tla.mmad(dst, a, b, init_c=True, unit_flag=unit)
            self.mtx_l0b[buf].unlock(pipe=tla.arch.CUBE)
        self.mtx_kv.unlock(pipe=tla.arch.MTE1)
        self.mtx_l0a[a_buf].unlock(pipe=tla.arch.CUBE)
        self._l0c_signal_ready(slot)

    # store_score：L0C 分数按 AIV 拆分落 UB
    @tla.jit
    def store_score(self, qk_ub_base, valid_n, slot):
        """Split L0C score rows across AIVs while preserving TILE_S2 pitch."""
        # FixPipe 的 dst 必须按整块 (G, valid_n) 声明，两个 AIV 各落一半
        # （`L0C2UBMode.SPLIT_M`）。只声明 (G/2, valid_n) 时 AIC 侧仍按整块写，
        # 每张 AIV 的 UB 槽被写进 G 行 —— 越界踩坏相邻缓冲，表现为 PV 段卡死。
        dst = tla.make_tensor(
            qk_ub_base,
            tla.make_layout(
                tla.make_shape(self.group_size, valid_n),
                tla.make_stride(TILE_S2, 1),
            ),
        )
        src = _bounded_local(
            self.l0c_tile(slot), (self.group_size, valid_n), (self.group_size, TILE_S2)
        )
        # fixpipe 侧按统一 M→F 参数落盘。
        score_params = tla.params.CopyL0C2DstParams(
            l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M
        )
        self._l0c_wait_ready(slot)
        tla.copy(dst, src, score_params)
        self._l0c_signal_free(slot)

    # store_pv：L0C PV 结果整体搬出到 UB
    def store_pv(self, pv_ub_base, slot=0):
        """L0C PV result -> the cross-core pv_ub buffer, all G rows.

        L0C 槽号由调用方显式传入。

        四段 PV mmad 写进宽 (TILE_M,DIM_NOPE) L0C 的四个 32 KiB 段，再由
        **一条**宽 FixPipe 取走（nSize=512 / mSize=64 / dstStride=512 /
        srcStride=64）。四段（各自 (64,128) L0Clayout，基址 +seg*8192）与一条
        (64,512) L0Clayout 是同一幅字节图（flat = (n//16)*64*16 + (m//16)*256 +
        (m%16)*16 + n%16），因此这里合成一条宽拷贝，落点形状/布局与消费侧
        (vec_m, DIM_NOPE) 行主序一致。
        """
        params = tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M)
        self._l0c_wait_ready(slot)
        dst = tla.make_tensor(
            pv_ub_base,
            tla.make_layout(
                tla.make_shape(self.group_size, DIM_NOPE),
                tla.make_stride(DIM_NOPE, 1),
            ),
        )
        tla.copy(dst, self.l0c_tile(slot), params)
        self._l0c_signal_free(slot)


# ---------------------------------------------------------------------------------------------
# 原生存储视图辅助（均为普通 Python 函数，无 @jit）
# ---------------------------------------------------------------------------------------------
def _native_head_tile(gm, layout, query_id, head_offset, rows, heads_per_token):
    """Fix one physical token in native BSND/TND storage; tile the G,D axes.

    tla 的 DMA 只认二维元素视图，因此把 "某一 token 的 head 区间" 折算成
    (query_id * heads_per_token + head_offset, 0) 起始、rows 行的二维视图。
    宿主统一按 (rows, D) 绑定 GM，heads_per_token 由调用方以 Constexpr 传入。
    """
    origin = gm.origin_shape
    dim = origin[-1]
    if tla.const_expr(layout == "BSND"):
        token = query_id
    else:
        token = query_id
    return tla.get_tile(
        gm,
        tla.make_coord(token * heads_per_token + head_offset, 0),
        tla.make_shape(rows, dim),
    )


def _align16_py(value: int) -> int:
    return (value + 15) // 16 * 16


def _bounded_local(buffer, shape, capacity, offset=0, stride=None):
    """Preserve the allocation pitch while narrowing the runtime extent.

    tla 里源张量自身携带 stride，`get_tile` 只裁剪 origin_shape，因此等价于
    固定物理 pitch、只收窄逻辑 extent；`capacity` / `stride` 仅在偏移
    合法性检查中使用。
    """
    if stride is not None:
        row_stride = stride[-1] if isinstance(stride, tuple) else stride
        if row_stride < 0:
            raise ValueError(f"row stride must be non-negative, got {row_stride}")
    if len(shape) == 1:
        return tla.get_tile(
            buffer, tla.make_coord(offset, 0), tla.make_shape(1, shape[0])
        )
    return tla.get_tile(
        buffer, tla.make_coord(offset, 0), tla.make_shape(shape[0], shape[1])
    )


# ---------------------------------------------------------------------------------------------
# `_SparseFlashAttentionKernel`（AIC/AIV 同体 tick 管线 + 各 stage）
#
# 单一 `__call__` 由 AIC/AIV 同体执行。catlass.tla 只能显式划分
# `with tla.cube():` / `with tla.vector():`，因此：
#   * 标量流水线（query_id / valid_kv_count / ...）保持单份、放在 region 之外，两条核都执行；
#   * 设备动作按所属核放进对应 region，跨核依赖显式补 cross flag。
# ---------------------------------------------------------------------------------------------
class _SparseFlashAttentionKernel:
    """Process the query rows assigned to one AIC head group."""

    def __init__(
        self,
        group_size,
        sparse_count,
        head_chunks,
        queries_per_group,
        dtype_16,
        use_sinks,
        return_lse,
        head_count,
        total_queries,
        batch_count,
        layout_query,
        layout_kv,
        sparse_mode,
        has_rope,
        q_seq,
        kv_seq,
        kv_pages,
        kv_capacity,
        has_used_q,
        has_used_kv,
    ):
        self.group_size = group_size
        self.vec_m = (group_size + 1) // 2
        self.sparse_count = sparse_count
        self.head_chunks = head_chunks
        self.queries_per_group = queries_per_group
        self.dtype_16 = dtype_16
        self.use_sinks = use_sinks
        self.return_lse = return_lse
        self.head_count = head_count
        self.total_queries = total_queries
        self.batch_count = batch_count
        self.layout_query = layout_query
        self.layout_kv = layout_kv
        self.sparse_mode = sparse_mode
        self.has_rope = has_rope
        self.q_seq = q_seq
        self.kv_seq = kv_seq
        self.kv_pages = kv_pages
        self.kv_capacity = kv_capacity
        self.has_used_q = has_used_q
        self.has_used_kv = has_used_kv
        self.qk_dim = DIM_QK if has_rope else DIM_NOPE
        self.block_idx = tla.arch.block_idx()
        self.subblock_idx = tla.arch.sub_block_idx()

        self.matmul = _SparseMatmul(group_size, dtype_16, has_rope)
        self.vector = _SFAOnlineSoftmaxVector(
            self.vec_m,
            TILE_S2,
            DIM_NOPE,
            self.subblock_idx,
            dtype_16=dtype_16,
            preload_num=PIPELINE_SLOT_COUNT,
            head_count=head_count,
        )

        # 槽位缓冲：FixPipe 的 dst 要按整块 (G, N) 声明、由 SPLIT_M 分给
        # 两个 AIV，因此 qk_ub 改成两个独立分配（总空间不变）以便拿到各自的 ptr。
        self.qk_ub = _ub_tensor((self.vec_m, TILE_S2), tla.Float32)
        self.qk_ub_alt = _ub_tensor((self.vec_m, TILE_S2), tla.Float32)
        self.qk_ub_ptrs = (self.qk_ub.ptr, self.qk_ub_alt.ptr)
        self.pv_ub = _ub_tensor((self.vec_m, DIM_NOPE), tla.Float32)
        self.pv_ub_ptr = self.pv_ub.ptr
        self.p_l1 = _l1_tensor((group_size, TILE_S2), dtype_16, tla.arch.zN)
        self.stage0 = _ub_tensor((2 * GATHER_CHUNK_ROWS, self.qk_dim), dtype_16)
        self.sink_state = _ub_tensor((self.vec_m, 1), tla.Float32)
        self.lse_max_state = _ub_tensor((self.vec_m, 1), tla.Float32)
        self.lse_sum_state = _ub_tensor((self.vec_m, 1), tla.Float32)
        self.zero_row = _ub_tensor((1, DIM_NOPE), dtype_16)

        # 每条跨核生产者-消费者边一对 cross flag，一槽一个。
        self.ws_ready = []
        self.ws_free = []
        for i in range(PIPELINE_SLOT_COUNT):
            self.ws_ready.append(tla.cross_flag(f"ws_ready{i}", mode=2))
            self.ws_free.append(tla.cross_flag(f"ws_free{i}", mode=2))
        self.qk_ready = [tla.cross_flag(f"qk_ready{i}", mode=2) for i in range(2)]
        self.qk_free = [tla.cross_flag(f"qk_free{i}", mode=2) for i in range(2)]
        self.p_ready = tla.cross_flag("p_ready", mode=2)
        self.p_free = tla.cross_flag("p_free", mode=2)
        self.pv_ready = tla.cross_flag("pv_ready", mode=2)
        self.pv_free = tla.cross_flag("pv_free", mode=2)

        # AIV 侧核内同步：MTE2 生产、MTE3 消费的片上缓冲。
        # stage0 是**按槽**两把 MTE2/MTE3 互斥（`asc_lock` 的 id 取槽奇偶，
        # `asc_lock(PIPE_MTE2, (uint8_t)(slot != 0))`），单把锁会把
        # 「本 chunk 的 UB→GM flush」与「下一 chunk 的 GM→UB gather」串成一条链，
        # 双缓冲失去意义。
        self.mtx_stage0 = (
            tla.mutex("stage0_0", id=10),
            tla.mutex("stage0_1", id=13),
        )
        self.mtx_zero = tla.mutex("zero_row", id=11)

    # --- 槽位视图 -------------------------------------------------------------------------
    @tla.jit
    def qk_ub_base(self, slot):
        """qk_ub 槽基址（ptr）：FixPipe dst 与 AIV 读视图共用。"""
        base = self.qk_ub_ptrs[0]
        if slot != 0:
            base = self.qk_ub_ptrs[1]
        return base

    @tla.jit
    def qk_slot(self, slot):
        return tla.make_tensor(
            self.qk_ub_base(slot),
            tla.make_layout(
                tla.make_shape(self.vec_m, TILE_S2), tla.make_stride(TILE_S2, 1)
            ),
        )

    def pv_slot(self):
        return tla.tile_view(
            self.pv_ub,
            tla.make_shape(self.vec_m, DIM_NOPE),
            tla.make_coord(0, 0),
        )

    def stage0_slot(self, slot):
        return tla.tile_view(
            self.stage0,
            tla.make_shape(GATHER_CHUNK_ROWS, self.qk_dim),
            tla.make_coord(slot, 0),
        )

    # --- cross flag 的静态槽选择（tla 的 flag 句柄不能用动态下标） -------------------------
    @tla.jit
    def _set3(self, flags, slot, pipe):
        if slot == 0:
            tla.cross_core_set_flag(flags[0], pipe)
        elif slot == 1:
            tla.cross_core_set_flag(flags[1], pipe)
        else:
            tla.cross_core_set_flag(flags[2], pipe)

    @tla.jit
    def _wait3(self, flags, slot, pipe):
        if slot == 0:
            tla.cross_core_wait_flag(flags[0], pipe)
        elif slot == 1:
            tla.cross_core_wait_flag(flags[1], pipe)
        else:
            tla.cross_core_wait_flag(flags[2], pipe)

    @tla.jit
    def _set2(self, flags, slot, pipe):
        if slot == 0:
            tla.cross_core_set_flag(flags[0], pipe)
        else:
            tla.cross_core_set_flag(flags[1], pipe)

    @tla.jit
    def _wait2(self, flags, slot, pipe):
        if slot == 0:
            tla.cross_core_wait_flag(flags[0], pipe)
        else:
            tla.cross_core_wait_flag(flags[1], pipe)

    # --- _fill_softmax_sum_one：把 running sum 置 1 ----------------------------------------
    @tla.jit
    def _fill_softmax_sum_one(self, state_slot):
        with tla.vec.func(mode="simd"):
            mask, _ = tla.update_mask(self.vec_m, dtype=tla.Float32)
            ones = tla.full(1.0, tla.Float32, mask=mask)
            self.vector.state_slot(self.vector.sm_sum_tb, state_slot).store(
                ones, mask=mask
            )

    # --- _store_softmax_state：把 max/sum 状态经 UB 中转落 GM ------------------------------
    @tla.jit
    def _store_softmax_state(self, max_token, sum_token, state_slot):
        # UB→UB 整槽拷贝放在下面的 V 锁内，保持 V 侧排空语义。
        # UB→GM 同样避开 4 B 行 pitch：(32,1) f32 的二维落盘实测只写出零散
        # 4 行段（`written_per_block` 4/24/56…），换成 1-D 连续视图后是整段 32 行。
        #
        # 两处偏移口径不同：
        # * UB 源恒取本 AIV 的 `lse_*_state` 第 0 行起的 32 行——每个 AIV 的
        #   UB 状态表本来就只有 `AIV_ROW_CAPACITY` 行（`lse_max_state`
        #   声明为 `(vec_m, 1)`）。
        # * GM 侧的 32 行半段偏移由调用方 `_emit_stats` 加进 `token`，语义等于
        #   按 `split_axis=0 + part_id=subblock_idx` 切分两个 AIV。
        stride1 = tla.make_stride(1)
        rows = tla.make_shape(self.vector.tile_vec_m)
        # GM 源取上面的 `lse_*_state` 中转（先 sm→lse 的 UB→UB，再由 lse 落
        # GM）。两条路径落 GM 的字节相同（32 行 f32），中转让上面的 UB→UB
        # 拷贝成为有效写，避免死代码。
        # 用 `mtx_o` 的 V→MTE3 交接替代两条 `pipe_barrier(ALL)`。
        # 语义与 `finish_o_cast` 的 o_ub 交接同形：V 侧在锁内排空（含上面
        # 两条 UB→UB 拷贝与 softmax 阶段的 sm_*_tb 写），MTE3 侧再拿同一把
        # 锁读 sm_*_tb 落 GM（对应 asc_sync_intra_wait(PIPE_V,2) +
        # asc_lock(PIPE_MTE3,7)）。
        mtx_o = self.vector.mtx_o
        mtx_o.lock(pipe=tla.arch.VECTOR)
        tla.copy(
            self.vector._state_flat(self.lse_max_state, 0),
            self.vector._state_flat(self.vector.sm_max_tb, state_slot),
        )
        tla.copy(
            self.vector._state_flat(self.lse_sum_state, 0),
            self.vector._state_flat(self.vector.sm_sum_tb, state_slot),
        )
        mtx_o.unlock(pipe=tla.arch.VECTOR)
        src_layout = tla.make_layout(
            tla.make_shape(1, self.vector.tile_vec_m),
            tla.make_stride(self.vector.tile_vec_m, 1),
        )
        mtx_o.lock(pipe=tla.arch.MTE3)
        tla.copy(
            max_token,
            tla.make_tensor(
                self.lse_max_state.ptr,
                src_layout,
            ),
        )
        tla.copy(
            sum_token,
            tla.make_tensor(
                self.lse_sum_state.ptr,
                src_layout,
            ),
        )
        mtx_o.unlock(pipe=tla.arch.MTE3)

    # --- _kv_token：按存储布局取一条物理 KV 行 --------------------------------------------
    @tla.jit
    def _kv_token(self, gm, token):
        """Select one physical KV row without flattening paged strides.

        宿主把 GM 绑成 (rows, D) 的 2-D 视图，KV 只有 1 个 head，
        因此 BSND 的 `token // S, token % S` 在扁平行号上恒等于 `token`。
        """
        if tla.const_expr(self.layout_kv == "TND"):
            row = token
        else:
            row = (token // self.kv_seq) * self.kv_seq + token % self.kv_seq
        return tla.get_tile(
            gm,
            tla.make_coord(row, 0),
            tla.make_shape(1, gm.origin_shape[-1]),
        )

    # --- pair 双行 DMA 需要的行号（与 `_kv_token` 同一套 BSND/TND 换算） ------------------
    @tla.jit
    def _kv_row(self, token):
        if tla.const_expr(self.layout_kv == "TND"):
            return token
        return (token // self.kv_seq) * self.kv_seq + token % self.kv_seq

    # --- _workspace_tile：本 block 的 GM workspace 环形槽 --------------------------------
    @tla.jit
    def _workspace_tile(self, workspace, task_slot):
        """Return one slot from this block's GM workspace ring.

        槽偏移 = `block_idx * 3 + task_slot` 是**动态**值，用 `get_tile` 建视图时
        该偏移在 `tla.copy` 里会被丢掉，于是所有 block 的 `load_sparse_kv` 都会
        把 block 0 的 KV 搬进自己的 L1（实测错位时 PV 结果与 `P(q4) @ K(q0)`
        的余弦相似度 0.90）。这里改成 ptr + 偏移的行主序视图。
        """
        base = workspace.ptr + (self.block_idx * PIPELINE_SLOT_COUNT + task_slot) * (
            TILE_S2 * self.qk_dim
        )
        return tla.make_tensor(
            base,
            tla.make_layout(
                tla.make_shape(TILE_S2, self.qk_dim),
                tla.make_stride(self.qk_dim, 1),
                layoutTag=tla.arch.RowMajor,
            ),
        )

    # --- _find_valid_end：二分查找有效前缀的结束位置 -------------------------------------
    @tla.jit
    def _find_valid_end(self, indices, query_id, planned_end):
        """Find the first padding entry in a valid-prefix sparse row."""
        lo = tla.as_numeric(0)
        hi = planned_end
        while lo < hi:
            mid = (lo + hi) // 2
            if tla.as_numeric(indices[query_id, mid]) == INVALID_SPARSE_INDEX:
                hi = mid
            else:
                lo = mid + 1
        return lo

    # --- _stage_gather：按行区间 gather 一块稀疏 KV tile --------------------------------
    @tla.jit
    def _stage_gather(
        self,
        slot,
        query_id,
        sparse_idx,
        tile_rows,
        batch,
        key,
        key_rope,
        indices,
        table,
        cu_kv,
        workspace,
        cursor_begin,
        cursor_end,
        notify,
    ):
        """Gather this AIV's rows [cursor_begin, cursor_end) of one sparse tile.

        gather 拆成可分段发射的形态，调用方在 `gather_lo..gather_mid` 与
        `gather_mid..gather_hi` 之间插入 Stage 2 的向量段；只有 `notify != 0`
        的那次（第二半）在末尾交棒 `ws_ready`。
        """
        # 动态 while/if 体内不得出现 `self.`（前端限制），全部提升为局部名。
        kv_token = self._kv_token
        kv_row = self._kv_row
        qk_dim = self.qk_dim
        has_rope = self.has_rope
        subblock_idx = self.subblock_idx
        block_idx = self.block_idx
        # `get_tile` 套 `get_tile` 的**动态**行偏移在落盘时会退化，三个 task
        # 槽的 gather 结果会全部落回 slot 0（实测 slot 1/2 全程全零、softmax
        # 读到零分数）。这里按 ptr + 槽偏移重建目标张量（与 `_workspace_tile`
        # 同一形态），槽偏移与行偏移都直接进地址。
        slot_base = (block_idx * PIPELINE_SLOT_COUNT + slot) * (TILE_S2 * qk_dim)
        # 动态 while 体内不得出现 `self.method(...)`（前端限制），先提升为局部名。
        kv_address = self._kv_address
        kv_pages = self.kv_pages
        layout_kv = self.layout_kv
        kv_seq = self.kv_seq
        # PA 单页（kv_pages == 1）特化——页索引恒 0，页号读由逐行（topk 次/查询）
        # 提到段级一次；kv_capacity = kv_seq ⇒ kv_idx < kv_seq，地址复合折成
        # page_base + kv_idx。
        pa_page_base = tla.as_numeric(0)
        if tla.const_expr(kv_pages == 1 and layout_kv == "PA_BSND"):
            pa_page_base = tla.as_numeric(table[batch, 0]) * kv_seq
        stage0_slot = self.stage0_slot
        set3 = self._set3
        mtx_stage0_0 = self.mtx_stage0[0]
        mtx_stage0_1 = self.mtx_stage0[1]
        local_cursor = cursor_begin
        while local_cursor < cursor_end:
            chunk_rows = cursor_end - local_cursor
            if chunk_rows > GATHER_CHUNK_ROWS:
                chunk_rows = tla.as_numeric(GATHER_CHUNK_ROWS)
            chunk_slot = (local_cursor // GATHER_CHUNK_ROWS) % 2
            stage = stage0_slot(chunk_slot)
            # MTE2 写 stage0、MTE3 随后读它搬 workspace；锁按槽取。
            if chunk_slot == 0:
                mtx_stage0_0.lock(pipe=tla.arch.MTE2)
            else:
                mtx_stage0_1.lock(pipe=tla.arch.MTE2)
            row = tla.as_numeric(0)
            while row < chunk_rows:
                index_pos = sparse_idx + local_cursor + row
                kv_idx0 = tla.as_numeric(indices[query_id, index_pos])
                if tla.const_expr(kv_pages == 1 and layout_kv == "PA_BSND"):
                    token0 = pa_page_base + kv_idx0
                else:
                    token0 = kv_address(key, table, cu_kv, batch, kv_idx0)
                if row + SPARSE_INDEX_PAIR_SIZE <= chunk_rows:
                    kv_idx1 = tla.as_numeric(indices[query_id, index_pos + 1])
                    if tla.const_expr(kv_pages == 1 and layout_kv == "PA_BSND"):
                        token1 = pa_page_base + kv_idx1
                    else:
                        token1 = kv_address(key, table, cu_kv, batch, kv_idx1)
                    token_gap = token1 - token0 if token0 < token1 else token0 - token1
                    source_gap_bytes = (
                        (token_gap - 1) * DIM_NOPE * BYTES_PER_16BIT_ELEMENT
                    )
                    if source_gap_bytes >= 0 and source_gap_bytes < INT32_MAX:
                        first_token = token0 if token0 < token1 else token1
                        second_token = token1 if token0 < token1 else token0
                        pair_index = row // SPARSE_INDEX_PAIR_SIZE
                        if tla.const_expr(has_rope):
                            row_first = kv_row(first_token)
                            _copy_kv_pair(
                                stage,
                                key,
                                key_rope,
                                row_first,
                                kv_row(second_token) - row_first,
                                pair_index,
                            )
                        else:
                            row_first = kv_row(first_token)
                            _copy_kv_pair_no_rope(
                                stage,
                                key,
                                row_first,
                                kv_row(second_token) - row_first,
                                pair_index,
                            )
                    else:
                        if tla.const_expr(has_rope):
                            _copy_kv_token(
                                stage,
                                kv_token(key, token0),
                                kv_token(key_rope, token0),
                                row,
                            )
                            _copy_kv_token(
                                stage,
                                kv_token(key, token1),
                                kv_token(key_rope, token1),
                                row + 1,
                            )
                        else:
                            _copy_kv_token_no_rope(stage, kv_token(key, token0), row)
                            _copy_kv_token_no_rope(
                                stage, kv_token(key, token1), row + 1
                            )
                    row = row + SPARSE_INDEX_PAIR_SIZE
                else:
                    if tla.const_expr(has_rope):
                        _copy_kv_token(
                            stage,
                            kv_token(key, token0),
                            kv_token(key_rope, token0),
                            row,
                        )
                    else:
                        _copy_kv_token_no_rope(stage, kv_token(key, token0), row)
                    row = row + 1
            if chunk_slot == 0:
                mtx_stage0_0.unlock(pipe=tla.arch.MTE2)
            else:
                mtx_stage0_1.unlock(pipe=tla.arch.MTE2)
            ready = stage
            dst = tla.make_tensor(
                workspace.ptr + slot_base + local_cursor * qk_dim,
                tla.make_layout(
                    tla.make_shape(GATHER_CHUNK_ROWS, qk_dim),
                    tla.make_stride(qk_dim, 1),
                    layoutTag=tla.arch.RowMajor,
                ),
            )
            src = _bounded_local(
                ready,
                (chunk_rows, qk_dim),
                (GATHER_CHUNK_ROWS, qk_dim),
                stride=(qk_dim, 1),
            )
            if chunk_slot == 0:
                mtx_stage0_0.lock(pipe=tla.arch.MTE3)
            else:
                mtx_stage0_1.lock(pipe=tla.arch.MTE3)
            tla.copy(dst, src)
            if chunk_slot == 0:
                mtx_stage0_0.unlock(pipe=tla.arch.MTE3)
            else:
                mtx_stage0_1.unlock(pipe=tla.arch.MTE3)
            local_cursor = local_cursor + chunk_rows
        if notify != 0:
            if tile_rows > 0:
                # Notify AIC only after the whole tile's workspace writes are issued.
                set3(self.ws_ready, slot, tla.arch.MTE3)

    # --- _stage_qk：从 gather 好的 KV tile 计算 QK ---------------------------------------
    @tla.jit
    def _stage_qk(self, slot, workspace, valid_n, qk_slot):
        """Compute QK scores from the gathered KV tile."""
        # Wait for the AIV workspace write before reading it on AIC.
        self._wait3(self.ws_ready, slot, tla.arch.MTE2)
        workspace_tile = self._workspace_tile(workspace, slot)
        shared_k = self.matmul.load_sparse_kv(workspace_tile, valid_n, slot)
        self.matmul.compute_qk(shared_k, valid_n, qk_slot)
        # qk_ub 是双槽缓冲：写槽前必须等该槽上一任被 softmax 消费。
        # 空槽令牌在 `__call__` 进入 tick 循环前预置。
        self._wait2(self.qk_free, qk_slot, tla.arch.FIX)
        self.matmul.store_score(self.qk_ub_base(qk_slot), valid_n, qk_slot)
        self._set3(self.ws_free, slot, tla.arch.MTE2)
        self._set2(self.qk_ready, qk_slot, tla.arch.FIX)

    # --- _store_p：把 p_ub 写入共享 L1 -----------------------------------------------
    @tla.jit
    def _store_p(self, p_slot):
        # P 的 L1 槽只有一块：写 L1 前必须等 AIC 消费完上一块 P。
        # 空槽令牌在 `__call__` 进入 tick 循环前预置。
        tla.cross_core_wait_flag(self.p_free, tla.arch.MTE3)
        self.vector.store_p(self.p_l1, p_slot)
        tla.cross_core_set_flag(self.p_ready, tla.arch.MTE3)

    # --- _stage_pv：P @ V 与结果搬出 ------------------------------------------------
    @tla.jit
    def _stage_pv(self, k_slot, valid_n, pv_slot):
        tla.cross_core_wait_flag(self.p_ready, tla.arch.MTE1)
        shared_k = self.matmul.right_tile(k_slot)
        self.matmul.compute_pv(shared_k, self.p_l1, valid_n, pv_slot)
        tla.cross_core_set_flag(self.p_free, tla.arch.MTE1)
        # pv_ub 单槽：store 前等 AIV 消费完上一块。
        tla.cross_core_wait_flag(self.pv_free, tla.arch.FIX)
        self.matmul.store_pv(self.pv_ub_ptr, pv_slot)
        tla.cross_core_set_flag(self.pv_ready, tla.arch.FIX)

    # --- _cross_init_state：重置一个状态槽的 running max/sum -----------------------------
    @tla.jit
    def _cross_init_state(self, state_slot, valid_count):
        """Reset running softmax max and sum for one state slot."""
        with tla.vec.func(mode="simd"):
            rows, _ = tla.update_mask(self.vec_m, dtype=tla.Float32)
            zero = tla.full(0.0, tla.Float32, mask=rows)
            # `vdups(SOFTMAX_NEG_INF if valid_count > 0 else 0.0)`：
            # `tla.full` 只收字面量，动态条件改写成同值的选择分支。
            initial = tla.full(0.0, tla.Float32, mask=rows)
            if valid_count > 0:
                initial = tla.full(SOFTMAX_NEG_INF, tla.Float32, mask=rows)
            self.vector.state_slot(self.vector.sm_max_tb, state_slot).store(
                initial, mask=rows
            )
            self.vector.state_slot(self.vector.sm_sum_tb, state_slot).store(
                zero, mask=rows
            )

    # --- _query_tile / _output_tile：原生布局下的 Q / 输出二维视图 -----------------------
    @tla.jit
    def _query_tile(self, gm, query_id, head_offset):
        return _native_head_tile(
            gm,
            self.layout_query,
            query_id,
            head_offset,
            self.group_size,
            self.head_count,
        )

    @tla.jit
    def _output_tile(self, gm, query_id, head_offset, rows):
        return _native_head_tile(
            gm, self.layout_query, query_id, head_offset, rows, self.head_count
        )

    # --- _sequence_length：used 优先，其次 packed，最后 capacity -------------------------
    @tla.jit
    def _sequence_length(self, cu, used, batch, capacity, packed, has_used):
        """Prefer the used length; otherwise use the packed length or capacity."""
        length = capacity
        if tla.const_expr(packed):
            length = tla.as_numeric(cu[batch + 1, 0]) - tla.as_numeric(cu[batch, 0])
        if tla.const_expr(has_used):
            length = tla.as_numeric(used[batch, 0])
        return length

    # --- _q_length：Query 序列长度 -------------------------------------------------------
    @tla.jit
    def _q_length(self, query, cu, used, batch):
        capacity = self.q_seq
        return self._sequence_length(
            cu, used, batch, capacity, self.layout_query == "TND", self.has_used_q
        )

    # --- _kv_capacity / _kv_length：KV 容量与有效长度 -----------------------------------
    @tla.jit
    def _kv_capacity(self, key, table):
        """Return the logical KV capacity for the active storage layout."""
        return self.kv_capacity

    @tla.jit
    def _kv_length(self, key, table, cu, used, batch):
        capacity = self._kv_capacity(key, table)
        return self._sequence_length(
            cu, used, batch, capacity, self.layout_kv == "TND", self.has_used_kv
        )

    # --- _query_coords：扁平 Query ID → (batch, position) --------------------------------
    @tla.jit
    def _query_coords(self, query, cu, query_id):
        """Map a flattened Query ID to its batch and position."""
        batch = tla.as_numeric(0)
        position = query_id
        if tla.const_expr(self.layout_query == "BSND"):
            batch = query_id // self.q_seq
            position = query_id % self.q_seq
        else:
            # Repeated boundaries represent empty packed sequences.  `cu` is
            # non-decreasing, so the linear scan's "last b with
            # cu[b] <= q < cu[b+1]" is the largest b with cu[b] <= q; find it
            # by binary search.  The upper-bound test keeps the out-of-range
            # behaviour of the scan (no match -> batch stays 0).
            lo = tla.as_numeric(0)
            hi = tla.as_numeric(self.batch_count) - 1
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if tla.as_numeric(cu[mid, 0]) <= query_id:
                    lo = mid
                else:
                    hi = mid - 1
            if tla.as_numeric(cu[lo, 0]) <= query_id and query_id < tla.as_numeric(
                cu[lo + 1, 0]
            ):
                batch = lo
                position = query_id - tla.as_numeric(cu[lo, 0])
        return batch, position

    # --- _kv_address：逻辑 KV 位置 → 物理行号 --------------------------------------------
    @tla.jit
    def _kv_address(self, key, table, cu, batch, kv_idx):
        """Map one logical KV position to a physical row."""
        token = kv_idx
        if tla.const_expr(self.layout_kv == "BSND"):
            token = batch * self.kv_seq + kv_idx
        elif tla.const_expr(self.layout_kv == "TND"):
            token = tla.as_numeric(cu[batch, 0]) + kv_idx
        else:
            page = tla.as_numeric(table[batch, kv_idx // self.kv_seq])
            token = page * self.kv_seq + kv_idx % self.kv_seq
        return token

    # --- _prepare_sparse_query：解析 Query 的 batch / KV 上界 / 稀疏扫描界 ---------------
    @tla.jit
    def _prepare_sparse_query(
        self, query, key, table, cu_q, cu_kv, used_q, used_kv, query_id
    ):
        """Resolve this Query's batch, visible KV limit, and sparse scan bound."""
        sparse_count = self.sparse_count
        sparse_mode = self.sparse_mode
        batch, position = self._query_coords(query, cu_q, query_id)
        q_length = self._q_length(query, cu_q, used_q, batch)
        kv_length = self._kv_length(key, table, cu_kv, used_kv, batch)
        kv_end = kv_length
        if tla.const_expr(sparse_mode == SPARSE_MODE_RIGHT_DOWN_CAUSAL):
            kv_end = kv_length - q_length + position + 1
        is_query_active = tla.as_numeric(0)
        if position < q_length and kv_end >= 0:
            is_query_active = tla.as_numeric(1)
        if tla.const_expr(sparse_mode == SPARSE_MODE_RIGHT_DOWN_CAUSAL):
            if kv_end == 0:
                is_query_active = tla.as_numeric(0)
        sparse_end = tla.as_numeric(0)
        if is_query_active != 0:
            sparse_end = kv_end
            if sparse_count < sparse_end:
                sparse_end = tla.as_numeric(sparse_count)
        return batch, kv_length, kv_end, sparse_end, is_query_active

    # --- _init_query_state：初始化 Query 的 softmax 状态与可选 sink ---------------------
    @tla.jit
    def _init_query_state(self, sinks, head_offset, count, state_slot):
        """Initialize a Query's softmax state and optional sink."""
        # 动态 if 内不得出现 `self.`（前端限制）。
        state_flat = self.vector._state_flat
        sink_state = self.sink_state
        sm_max_tb = self.vector.sm_max_tb
        fill_softmax_sum_one = self._fill_softmax_sum_one
        use_sinks = self.use_sinks
        head_chunks = self.head_chunks
        head_count = self.head_count
        subblock_idx = self.subblock_idx
        vec_m = self.vec_m
        self._cross_init_state(state_slot, count)
        if tla.const_expr(use_sinks):
            # A negative count marks an inactive Query: do not load its sink.
            if count >= 0:
                # Limit sink loads to valid rows in a partial head group.
                for part in tla.range_constexpr(head_chunks):
                    for sub in tla.range_constexpr(2):
                        # `rows` 必须在 Python 层就是常量，后面的
                        # `tla.const_expr(rows > 0)` 才能折叠。若用
                        # `tla.as_numeric(AIV_ROW_CAPACITY)` 夹取，会把 `rows`
                        # 变成运行时标量，凡是 `head_count > 32` 且开 sinks 的
                        # 档位（heads=65）直接报 "requires a compile-time
                        # Python value"。
                        rows = min(
                            vec_m,
                            max(0, head_count - part * self.group_size - sub * vec_m),
                        )
                        if tla.const_expr(rows > 0):
                            row_start = part * self.group_size + sub * vec_m
                            if (
                                head_offset == part * self.group_size
                                and subblock_idx == sub
                            ):
                                # 这两次拷贝的先后必须显式建立：GM→UB 走 MTE2、
                                # 随后并行 UB→UB 拷贝，且前面的
                                # `_cross_init_state` 走 V pipe。实测（空 kv +
                                # sinks）偶发 max 停在 `_cross_init_state` 的初值
                                # 0.0（负 sink 行与参考差 1.0，而 sum 仍是 1.0）。
                                # 两个 barrier 建立这两条依赖；位置必须留在本分支
                                # 内——放到 `if count >= 0` 之外会把该区降级成
                                # 动态分支，`tla.const_expr(rows > 0)` 随即报
                                # "requires a compile-time Python value"（实测）。
                                tla.pipe_barrier(tla.pipes.ALL)
                                # GM→UB 的 sink 读同样走 1-D 连续视图：
                                # (rows,1) f32 的二维搬运行 pitch 只有 4 B。
                                sink_flat = state_flat(sink_state, 0)
                                flat = tla.make_layout(
                                    tla.make_shape(rows), tla.make_stride(1)
                                )
                                tla.copy(
                                    tla.make_tensor(sink_flat.ptr, flat),
                                    tla.make_tensor(sinks.ptr + row_start, flat),
                                )
                                # UB→UB 部分槽拷贝同样避开 4 B 行 pitch：按 1-D
                                # 元素视图搬 `rows` 个 f32。
                                max_flat = state_flat(sm_max_tb, state_slot)
                                tla.pipe_barrier(tla.pipes.ALL)
                                tla.copy(
                                    tla.make_tensor(max_flat.ptr, flat),
                                    tla.make_tensor(sink_flat.ptr, flat),
                                )
                fill_softmax_sum_one(state_slot)

    # --- _emit_stats：把本轮状态写入 GM 统计输出 -----------------------------------------
    @tla.jit
    def _emit_stats(self, maxima, sums, query_id, head_offset, state_slot):
        if tla.const_expr(self.return_lse):
            # `maxima[query_id]` 的行宽是补齐后的 `head_chunks*TILE_M`，
            # 不是实际 head 数（head 非 64 倍数时两者不同）。
            stats_row = query_id * (self.head_chunks * self.group_size) + head_offset
            # 64 行 `max_token` 按 `split_axis=0 + part_id=subblock_idx` 拆给两个
            # AIV；这里显式把本 AIV 的半段偏移加进 GM 侧地址。
            # 地址必须从 `maxima`/`sums` 的基地址起算：`get_tile` 结果经
            # `.ptr` 后只保留静态偏移，动态行列偏移会丢失（实测两个 block 的
            # 落盘都退回到基址前 32 行）。
            # GM 侧状态表按 `(halves, AIV_ROW_CAPACITY)` 的宽行布局落盘。
            # (32,1) / (1,32) 这类 “行 pitch 只有 4 B”的目标会被搬运指令按
            # 32 B 行宽重排（实测：只落 4 行、其余写坏），所以把同一批 64 个数
            # 改成两个各 32 列的行，等效于把 AIV 半段变成“行”、把 head 变成“列”。
            # 元素总数不变（每 query 仍是 head_chunks*64 个 f32）。
            half = (query_id * self.head_chunks + head_offset // self.group_size) * 2
            row = half + self.subblock_idx
            mb = tla.get_tile(
                maxima, tla.make_coord(row, 0), tla.make_shape(1, self.vec_m)
            )
            sb = tla.get_tile(
                sums, tla.make_coord(row, 0), tla.make_shape(1, self.vec_m)
            )
            self._store_softmax_state(mb, sb, state_slot)

    # --- _write_empty_query：空 Query 写零输出 -------------------------------------------
    @tla.jit
    def _write_empty_query(self, output, query_id, head_offset):
        # 动态 if 内不得出现 `self.`（前端限制）。
        output_tile = self._output_tile
        vec_m = self.vec_m
        subblock_idx = self.subblock_idx
        head_count = self.head_count
        dtype_16 = self.dtype_16
        zero_row_base = self.zero_row
        mtx_zero = self.mtx_zero
        zero_row = tla.tile_view(
            zero_row_base, tla.make_shape(1, DIM_NOPE), tla.make_coord(0, 0)
        )
        # V 写 zero_row、MTE3 随后逐行搬 GM。
        mtx_zero.lock(pipe=tla.arch.VECTOR)
        with tla.vec.func(mode="simd"):
            mask, _ = tla.update_mask(2 * VL_T, dtype=dtype_16)
            zeros = tla.full(0.0, dtype_16, mask=mask)
            for col in tla.range_constexpr(0, DIM_NOPE, 2 * VL_T):
                _vec_view(zero_row, 0, col, 2 * VL_T).store(zeros, mask=mask)
        mtx_zero.unlock(pipe=tla.arch.VECTOR)
        # head = head_offset + subblock*32 + row >= row，故 row >= head_count 的行
        # 不可能满足 head < head_count；head_count 是 trace 期常量，可直接裁掉
        # 这些行，免去 32 行展开在前导里构造永远不成立的地址/守卫。
        empty_rows = min(vec_m, head_count)
        for row in tla.range_constexpr(empty_rows):
            head = head_offset + subblock_idx * vec_m + row
            if head < head_count:
                mtx_zero.lock(pipe=tla.arch.MTE3)
                tla.copy(output_tile(output, query_id, head, 1), zero_row)
                mtx_zero.unlock(pipe=tla.arch.MTE3)

    # --- _stage_softmax：在线 softmax 并输出非归一化权重 --------------------------------
    @tla.jit
    def _stage_softmax(self, first, valid_n, slot, state_slot, scale, p_slot, qk_slot):
        """Update online softmax and stage unnormalized weights for PV."""
        # 动态 if 内不得出现 `self.`（前端限制）。
        softmax_first = self.vector.softmax_first
        softmax_rest = self.vector.softmax_rest
        use_sinks = self.use_sinks
        wait2 = self._wait2
        set2 = self._set2
        store_p = self._store_p
        # `slot` 是 sm_exp 的 3 槽轮转；QK 缓冲只有 2 槽，索引必须由调用方按
        # tile 序号给出（qk_ub 双槽，槽由入队顺序决定）。
        qk = self.qk_slot(qk_slot)
        wait2(self.qk_ready, qk_slot, tla.arch.VECTOR)
        if tla.const_expr(not use_sinks):
            if first != 0:
                softmax_first(qk, valid_n, scale, state_slot, p_slot)
            else:
                softmax_rest(qk, valid_n, scale, state_slot, slot, p_slot)
        else:
            softmax_rest(qk, valid_n, scale, state_slot, slot, p_slot)
        set2(self.qk_free, qk_slot, tla.arch.VECTOR)
        store_p(p_slot)

    # --- _stage_output：累加或收尾一块 PV 输出 -------------------------------------------
    @tla.jit
    def _stage_output(
        self,
        first,
        last,
        valid_n,
        slot,
        query_id,
        head_offset,
        output,
        maxima,
        sums,
    ):
        """Accumulate or finalize one PV tile, then close the Query if needed."""
        # 动态 if 内不得出现 `self.`（前端限制）。
        finish_o_cast = self.vector.finish_o_cast
        init_o = self.vector.init_o
        update_o = self.vector.update_o
        write_empty_query = self._write_empty_query
        emit_stats = self._emit_stats
        state_slot_of = self.vector.state_slot
        output_tile = self._output_tile
        head_count = self.head_count
        pv_slot_of = self.pv_slot
        sm_sum_tb = self.vector.sm_sum_tb
        sm_exp_tb = self.vector.sm_exp_tb
        res_o = self.vector.res_o
        pv_ready = self.pv_ready
        pv_free = self.pv_free
        state_slot = query_id % PIPELINE_SLOT_COUNT
        dst = output_tile(output, query_id, 0, head_count)
        # stats 的 UB→GM 落盘提前到输出写之前（只改发射顺序）。stats 的两条 128 B
        # 拷贝发出后，drain 尾部等待被输出的 MTE3 写遮挡；锁配对（mtx_o V/MTE3）
        # 与 flag 计数不变。
        if last != 0:
            emit_stats(maxima, sums, query_id, head_offset, state_slot)
        if valid_n > 0:
            pv = pv_slot_of()
            tla.cross_core_wait_flag(pv_ready, tla.arch.VECTOR)
            if last != 0:
                if first != 0:
                    finish_o_cast(
                        pv,
                        state_slot_of(sm_sum_tb, state_slot),
                        dst,
                        head_offset,
                    )
                else:
                    finish_o_cast(
                        pv,
                        state_slot_of(sm_sum_tb, state_slot),
                        dst,
                        head_offset,
                        state_slot_of(sm_exp_tb, slot),
                    )
            elif first != 0:
                init_o(pv)
            else:
                update_o(pv, state_slot_of(sm_exp_tb, slot))
            tla.cross_core_set_flag(pv_free, tla.arch.VECTOR)
        if last != 0:
            if valid_n == 0:
                if first != 0:
                    write_empty_query(output, query_id, head_offset)
                else:
                    # An empty final task can carry only the list terminator.
                    finish_o_cast(
                        res_o,
                        state_slot_of(sm_sum_tb, state_slot),
                        dst,
                        head_offset,
                    )

    # --- __call__ 的 tick 流水 -----------------------------------------------------------
    # 编译期 trip count == 1 档位的静态展开体。独立成 @tla.jit 方法，
    # 因为同一函数内同时出现静态展开段与动态循环时，前端会拒绝两处
    # 共用的编译期 origin 标记（实测 NameError: free variable
    # __tladsl_<name>_compiletime_origin_*）。状态在方法内自建，方法无返回值：
    # 原循环之后只做抽干令牌等待，不再读取任何 tick 状态。
    @tla.jit
    def _tick_loop_unrolled(
        self,
        output,
        query,
        key,
        indices,
        query_rope,
        key_rope,
        sinks,
        maxima,
        sums,
        workspace,
        table,
        cu_q,
        cu_kv,
        used_q,
        used_kv,
        scale,
    ):
        core_group = self.block_idx // self.head_chunks
        head_offset = self.block_idx % self.head_chunks * self.group_size

        # Determine the query range assigned to this core.
        first_query = core_group * self.queries_per_group
        query_end = first_query + self.queries_per_group
        if query_end > self.total_queries:
            query_end = tla.as_numeric(self.total_queries)
        current_query = first_query
        query_count = query_end - first_query

        # 动态 tick 循环体内不能出现 `self.method(...)`（前端只允许对句柄调用方法），
        # 先把所有 stage 入口提升为局部名再进循环（tla 报错原文的 "Hoist the call"）。
        prepare_sparse_query = self._prepare_sparse_query
        find_valid_end = self._find_valid_end
        stage_gather = self._stage_gather
        wait3 = self._wait3
        load_q = self.matmul.load_q
        query_tile = self._query_tile
        stage_qk = self._stage_qk
        init_query_state = self._init_query_state
        stage_softmax = self._stage_softmax
        stage_pv = self._stage_pv
        stage_output = self._stage_output

        subblock_idx = self.subblock_idx

        # Initialize sparse-block traversal and sequence-boundary state.
        sparse_idx = tla.as_numeric(0)
        is_first_block = tla.as_numeric(1)
        batch = tla.as_numeric(0)
        sparse_end = tla.as_numeric(0)
        is_query_active = tla.as_numeric(0)

        # 4 级流水线状态：每字段 3 级显式移位。
        stage_0_query_id = tla.as_numeric(EMPTY_QUERY_ID)
        stage_1_query_id = tla.as_numeric(EMPTY_QUERY_ID)
        stage_2_query_id = tla.as_numeric(EMPTY_QUERY_ID)
        stage_3_query_id = tla.as_numeric(EMPTY_QUERY_ID)
        stage_0_valid = tla.as_numeric(0)
        stage_1_valid = tla.as_numeric(0)
        stage_2_valid = tla.as_numeric(0)
        stage_3_valid = tla.as_numeric(0)
        stage_0_first = tla.as_numeric(0)
        stage_1_first = tla.as_numeric(0)
        stage_2_first = tla.as_numeric(0)
        stage_3_first = tla.as_numeric(0)
        stage_0_last = tla.as_numeric(0)
        stage_1_last = tla.as_numeric(0)
        stage_2_last = tla.as_numeric(0)
        stage_3_last = tla.as_numeric(0)
        stage_0_active = tla.as_numeric(0)
        stage_1_active = tla.as_numeric(0)
        stage_2_active = tla.as_numeric(0)
        stage_3_active = tla.as_numeric(0)

        # 工作段：原工作循环唯一的 1 次迭代（tick ≡ 0）。
        utick = 0
        # DelayLine 移位：tap(k) == stage_k
        stage_3_query_id = stage_2_query_id
        stage_2_query_id = stage_1_query_id
        stage_1_query_id = stage_0_query_id
        stage_3_valid = stage_2_valid
        stage_2_valid = stage_1_valid
        stage_1_valid = stage_0_valid
        stage_3_first = stage_2_first
        stage_2_first = stage_1_first
        stage_1_first = stage_0_first
        stage_3_last = stage_2_last
        stage_2_last = stage_1_last
        stage_1_last = stage_0_last
        stage_3_active = stage_2_active
        stage_2_active = stage_1_active
        stage_1_active = stage_0_active

        # Stage 0: gather a KV tile and register its task.
        valid_kv_count = tla.as_numeric(0)
        gather_active = tla.as_numeric(0)
        gather_slot = tla.as_numeric(0)
        gather_query = tla.as_numeric(0)
        gather_index = tla.as_numeric(0)
        gather_rows = tla.as_numeric(0)
        gather_lo = tla.as_numeric(0)
        gather_mid = tla.as_numeric(0)
        gather_hi = tla.as_numeric(0)
        if current_query < query_end:
            if is_first_block != 0:
                sparse_idx = tla.as_numeric(0)
                is_query_active = tla.as_numeric(0)
                sparse_end = tla.as_numeric(0)
                (
                    batch,
                    unused_kv_length,
                    unused_kv_end,
                    planned_sparse_end,
                    is_query_active,
                ) = prepare_sparse_query(
                    query,
                    key,
                    table,
                    cu_q,
                    cu_kv,
                    used_q,
                    used_kv,
                    current_query,
                )
                sparse_end = find_valid_end(indices, current_query, planned_sparse_end)

            # Gather at most one TILE_S2-sized block.
            if sparse_idx < sparse_end:
                tile_rows = sparse_end - sparse_idx
                if tile_rows > TILE_S2:
                    tile_rows = tla.as_numeric(TILE_S2)
                valid_kv_count = tile_rows
                gather_active = tla.as_numeric(1)
                gather_slot = utick % PIPELINE_SLOT_COUNT
                gather_query = current_query
                gather_index = sparse_idx
                gather_rows = tile_rows
                # 本 AIV 的行区间 [gather_lo, gather_hi) 与中点（按 chunk 对齐）。
                # 切分逻辑原样保留，只是变成标量侧的两段发射。
                rows_per_aiv_round = AIV_COUNT * SPARSE_INDEX_PAIR_SIZE
                pair_loops = (tile_rows + rows_per_aiv_round - 1) // rows_per_aiv_round
                split = pair_loops * SPARSE_INDEX_PAIR_SIZE
                if split > tile_rows:
                    split = tile_rows
                aligned_split = (
                    (split + GATHER_CHUNK_ROWS - 1) // GATHER_CHUNK_ROWS
                ) * GATHER_CHUNK_ROWS
                if aligned_split < tile_rows:
                    split = aligned_split
                else:
                    split = tile_rows
                gather_lo = tla.as_numeric(0)
                gather_hi = tile_rows
                if subblock_idx != 0:
                    gather_lo = split
                if subblock_idx == 0:
                    gather_hi = split
                gather_mid = gather_lo + (
                    ((gather_hi - gather_lo) // 2 // GATHER_CHUNK_ROWS)
                    * GATHER_CHUNK_ROWS
                )
                with tla.vector():
                    # 三槽环形工作区的空槽令牌由 AIC 在循环前预置，
                    # 因此每次 gather 都无条件等待。
                    wait3(self.ws_free, gather_slot, tla.arch.MTE3)
                    stage_gather(
                        gather_slot,
                        gather_query,
                        gather_index,
                        gather_rows,
                        batch,
                        key,
                        key_rope,
                        indices,
                        table,
                        cu_kv,
                        workspace,
                        gather_lo,
                        gather_mid,
                        tla.as_numeric(0),
                    )
                sparse_idx = sparse_idx + tile_rows

            # The last block lets the gather stage move to the next Query.
            is_last_block = tla.as_numeric(0)
            if sparse_idx >= sparse_end:
                is_last_block = tla.as_numeric(1)
            stage_0_query_id = current_query
            stage_0_valid = valid_kv_count
            stage_0_first = is_first_block
            stage_0_last = is_last_block
            stage_0_active = is_query_active
            is_first_block = tla.as_numeric(0)
            if is_last_block != 0:
                current_query = current_query + 1
                is_first_block = tla.as_numeric(1)
        else:
            # Fill unused ticks so older tasks can finish the pipeline.
            stage_0_query_id = tla.as_numeric(EMPTY_QUERY_ID)
            stage_0_valid = tla.as_numeric(0)
            stage_0_first = tla.as_numeric(0)
            stage_0_last = tla.as_numeric(0)
            stage_0_active = tla.as_numeric(0)

        # Stage 2 的向量段（init / softmax / store_p）提前到 gather 的两半
        # 之间发射。它消费 utick-2 的 tile，与本轮 gather 无数据依赖；向量执行与
        # gather 的标量链重叠后，store_p 的 V→MTE3 交棒等待被吸收。
        if tla.const_expr(utick >= 2):
            if stage_2_query_id >= 0:
                state_slot = stage_2_query_id % PIPELINE_SLOT_COUNT
                if stage_2_first != 0:
                    with tla.vector():
                        init_query_state(
                            sinks,
                            head_offset,
                            (
                                stage_2_valid
                                if stage_2_active != 0
                                else tla.as_numeric(-1)
                            ),
                            state_slot,
                        )
                if stage_2_valid > 0:
                    with tla.vector():
                        stage_softmax(
                            stage_2_first,
                            stage_2_valid,
                            (utick - 2) % PIPELINE_SLOT_COUNT,
                            state_slot,
                            scale,
                            (utick - 2) % 2,
                            (utick - 2) % 2,
                        )

        # gather 下半：续发余下的 chunk，并在最后交棒 ws_ready。
        if gather_active != 0:
            with tla.vector():
                stage_gather(
                    gather_slot,
                    gather_query,
                    gather_index,
                    gather_rows,
                    batch,
                    key,
                    key_rope,
                    indices,
                    table,
                    cu_kv,
                    workspace,
                    gather_mid,
                    gather_hi,
                    tla.as_numeric(1),
                )

        # Stage 1: compute QK for the tile gathered one utick ago.
        if tla.const_expr(utick >= 1):
            if stage_1_query_id >= 0 and stage_1_valid > 0:
                with tla.cube():
                    if stage_1_first != 0:
                        query_rope_tile = None
                        if tla.const_expr(self.has_rope):
                            query_rope_tile = query_tile(
                                query_rope, stage_1_query_id, head_offset
                            )
                        load_q(
                            query_tile(query, stage_1_query_id, head_offset),
                            query_rope_tile,
                        )
                    stage_qk(
                        (utick - 1) % PIPELINE_SLOT_COUNT,
                        workspace,
                        stage_1_valid,
                        (utick - 1) % 2,
                    )

        # Stage 2 cube 段：PV（向量段已在 gather 两半之间发射）。
        if tla.const_expr(utick >= 2):
            if stage_2_query_id >= 0:
                if stage_2_valid > 0:
                    with tla.cube():
                        stage_pv(
                            (utick - 2) % PIPELINE_SLOT_COUNT,
                            stage_2_valid,
                            (utick - 2) % 2,
                        )

        # Stage 3: accumulate or finalize the tile gathered three ticks ago.
        if tla.const_expr(utick >= 3):
            if stage_3_query_id >= 0:
                with tla.vector():
                    stage_output(
                        stage_3_first,
                        stage_3_last,
                        stage_3_valid,
                        (utick - 3) % PIPELINE_SLOT_COUNT,
                        stage_3_query_id,
                        head_offset,
                        output,
                        maxima,
                        sums,
                    )

        # 抽干段 1/3：原抽干循环的第 1 次迭代（drain_tick ≡ 1）。
        udrain_tick = 1
        # DelayLine 移位：tap(k) == stage_k
        stage_3_query_id = stage_2_query_id
        stage_2_query_id = stage_1_query_id
        stage_1_query_id = stage_0_query_id
        stage_3_valid = stage_2_valid
        stage_2_valid = stage_1_valid
        stage_1_valid = stage_0_valid
        stage_3_first = stage_2_first
        stage_2_first = stage_1_first
        stage_1_first = stage_0_first
        stage_3_last = stage_2_last
        stage_2_last = stage_1_last
        stage_1_last = stage_0_last
        stage_3_active = stage_2_active
        stage_2_active = stage_1_active
        stage_1_active = stage_0_active

        # 空 tick：老任务继续走完流水线（原 else 分支原样）。
        stage_0_query_id = tla.as_numeric(EMPTY_QUERY_ID)
        stage_0_valid = tla.as_numeric(0)
        stage_0_first = tla.as_numeric(0)
        stage_0_last = tla.as_numeric(0)
        stage_0_active = tla.as_numeric(0)

        # Stage 2 的向量段（init / softmax / store_p）提前到 gather 的两半
        # 之间发射。它消费 tick-2 的 tile，与本轮 gather 无数据依赖；向量执行与
        # gather 的标量链重叠后，store_p 的 V→MTE3 交棒等待被吸收。
        if tla.const_expr(udrain_tick >= 2):
            if stage_2_query_id >= 0:
                d_state_slot = stage_2_query_id % PIPELINE_SLOT_COUNT
                if stage_2_first != 0:
                    with tla.vector():
                        init_query_state(
                            sinks,
                            head_offset,
                            (
                                stage_2_valid
                                if stage_2_active != 0
                                else tla.as_numeric(-1)
                            ),
                            d_state_slot,
                        )
                if stage_2_valid > 0:
                    with tla.vector():
                        stage_softmax(
                            stage_2_first,
                            stage_2_valid,
                            (udrain_tick - 2) % PIPELINE_SLOT_COUNT,
                            d_state_slot,
                            scale,
                            (udrain_tick - 2) % 2,
                            (udrain_tick - 2) % 2,
                        )

        # Stage 1: compute QK for the tile gathered one tick ago.
        if tla.const_expr(udrain_tick >= 1):
            if stage_1_query_id >= 0 and stage_1_valid > 0:
                with tla.cube():
                    if stage_1_first != 0:
                        d_query_rope_tile = None
                        if tla.const_expr(self.has_rope):
                            d_query_rope_tile = query_tile(
                                query_rope, stage_1_query_id, head_offset
                            )
                        load_q(
                            query_tile(query, stage_1_query_id, head_offset),
                            d_query_rope_tile,
                        )
                    stage_qk(
                        (udrain_tick - 1) % PIPELINE_SLOT_COUNT,
                        workspace,
                        stage_1_valid,
                        (udrain_tick - 1) % 2,
                    )

        # Stage 2 cube 段：PV（向量段已在 gather 两半之间发射）。
        if tla.const_expr(udrain_tick >= 2):
            if stage_2_query_id >= 0:
                if stage_2_valid > 0:
                    with tla.cube():
                        stage_pv(
                            (udrain_tick - 2) % PIPELINE_SLOT_COUNT,
                            stage_2_valid,
                            (udrain_tick - 2) % 2,
                        )

        # Stage 3: accumulate or finalize the tile gathered three ticks ago.
        if tla.const_expr(udrain_tick >= 3):
            if stage_3_query_id >= 0:
                with tla.vector():
                    stage_output(
                        stage_3_first,
                        stage_3_last,
                        stage_3_valid,
                        (udrain_tick - 3) % PIPELINE_SLOT_COUNT,
                        stage_3_query_id,
                        head_offset,
                        output,
                        maxima,
                        sums,
                    )

        # 抽干段 2/3：原抽干循环的第 2 次迭代（drain_tick ≡ 2）。
        udrain_tick = 2
        # DelayLine 移位：tap(k) == stage_k
        stage_3_query_id = stage_2_query_id
        stage_2_query_id = stage_1_query_id
        stage_1_query_id = stage_0_query_id
        stage_3_valid = stage_2_valid
        stage_2_valid = stage_1_valid
        stage_1_valid = stage_0_valid
        stage_3_first = stage_2_first
        stage_2_first = stage_1_first
        stage_1_first = stage_0_first
        stage_3_last = stage_2_last
        stage_2_last = stage_1_last
        stage_1_last = stage_0_last
        stage_3_active = stage_2_active
        stage_2_active = stage_1_active
        stage_1_active = stage_0_active

        # 空 tick：老任务继续走完流水线（原 else 分支原样）。
        stage_0_query_id = tla.as_numeric(EMPTY_QUERY_ID)
        stage_0_valid = tla.as_numeric(0)
        stage_0_first = tla.as_numeric(0)
        stage_0_last = tla.as_numeric(0)
        stage_0_active = tla.as_numeric(0)

        # Stage 2 的向量段（init / softmax / store_p）提前到 gather 的两半
        # 之间发射。它消费 tick-2 的 tile，与本轮 gather 无数据依赖；向量执行与
        # gather 的标量链重叠后，store_p 的 V→MTE3 交棒等待被吸收。
        if tla.const_expr(udrain_tick >= 2):
            if stage_2_query_id >= 0:
                d_state_slot = stage_2_query_id % PIPELINE_SLOT_COUNT
                if stage_2_first != 0:
                    with tla.vector():
                        init_query_state(
                            sinks,
                            head_offset,
                            (
                                stage_2_valid
                                if stage_2_active != 0
                                else tla.as_numeric(-1)
                            ),
                            d_state_slot,
                        )
                if stage_2_valid > 0:
                    with tla.vector():
                        stage_softmax(
                            stage_2_first,
                            stage_2_valid,
                            (udrain_tick - 2) % PIPELINE_SLOT_COUNT,
                            d_state_slot,
                            scale,
                            (udrain_tick - 2) % 2,
                            (udrain_tick - 2) % 2,
                        )

        # Stage 1: compute QK for the tile gathered one tick ago.
        if tla.const_expr(udrain_tick >= 1):
            if stage_1_query_id >= 0 and stage_1_valid > 0:
                with tla.cube():
                    if stage_1_first != 0:
                        d_query_rope_tile = None
                        if tla.const_expr(self.has_rope):
                            d_query_rope_tile = query_tile(
                                query_rope, stage_1_query_id, head_offset
                            )
                        load_q(
                            query_tile(query, stage_1_query_id, head_offset),
                            d_query_rope_tile,
                        )
                    stage_qk(
                        (udrain_tick - 1) % PIPELINE_SLOT_COUNT,
                        workspace,
                        stage_1_valid,
                        (udrain_tick - 1) % 2,
                    )

        # Stage 2 cube 段：PV（向量段已在 gather 两半之间发射）。
        if tla.const_expr(udrain_tick >= 2):
            if stage_2_query_id >= 0:
                if stage_2_valid > 0:
                    with tla.cube():
                        stage_pv(
                            (udrain_tick - 2) % PIPELINE_SLOT_COUNT,
                            stage_2_valid,
                            (udrain_tick - 2) % 2,
                        )

        # Stage 3: accumulate or finalize the tile gathered three ticks ago.
        if tla.const_expr(udrain_tick >= 3):
            if stage_3_query_id >= 0:
                with tla.vector():
                    stage_output(
                        stage_3_first,
                        stage_3_last,
                        stage_3_valid,
                        (udrain_tick - 3) % PIPELINE_SLOT_COUNT,
                        stage_3_query_id,
                        head_offset,
                        output,
                        maxima,
                        sums,
                    )

        # 抽干段 3/3：原抽干循环的第 3 次迭代（drain_tick ≡ 3）。
        udrain_tick = 3
        # DelayLine 移位：tap(k) == stage_k
        stage_3_query_id = stage_2_query_id
        stage_2_query_id = stage_1_query_id
        stage_1_query_id = stage_0_query_id
        stage_3_valid = stage_2_valid
        stage_2_valid = stage_1_valid
        stage_1_valid = stage_0_valid
        stage_3_first = stage_2_first
        stage_2_first = stage_1_first
        stage_1_first = stage_0_first
        stage_3_last = stage_2_last
        stage_2_last = stage_1_last
        stage_1_last = stage_0_last
        stage_3_active = stage_2_active
        stage_2_active = stage_1_active
        stage_1_active = stage_0_active

        # 空 tick：老任务继续走完流水线（原 else 分支原样）。
        stage_0_query_id = tla.as_numeric(EMPTY_QUERY_ID)
        stage_0_valid = tla.as_numeric(0)
        stage_0_first = tla.as_numeric(0)
        stage_0_last = tla.as_numeric(0)
        stage_0_active = tla.as_numeric(0)

        # Stage 2 的向量段（init / softmax / store_p）提前到 gather 的两半
        # 之间发射。它消费 tick-2 的 tile，与本轮 gather 无数据依赖；向量执行与
        # gather 的标量链重叠后，store_p 的 V→MTE3 交棒等待被吸收。
        if tla.const_expr(udrain_tick >= 2):
            if stage_2_query_id >= 0:
                d_state_slot = stage_2_query_id % PIPELINE_SLOT_COUNT
                if stage_2_first != 0:
                    with tla.vector():
                        init_query_state(
                            sinks,
                            head_offset,
                            (
                                stage_2_valid
                                if stage_2_active != 0
                                else tla.as_numeric(-1)
                            ),
                            d_state_slot,
                        )
                if stage_2_valid > 0:
                    with tla.vector():
                        stage_softmax(
                            stage_2_first,
                            stage_2_valid,
                            (udrain_tick - 2) % PIPELINE_SLOT_COUNT,
                            d_state_slot,
                            scale,
                            (udrain_tick - 2) % 2,
                            (udrain_tick - 2) % 2,
                        )

        # Stage 1: compute QK for the tile gathered one tick ago.
        if tla.const_expr(udrain_tick >= 1):
            if stage_1_query_id >= 0 and stage_1_valid > 0:
                with tla.cube():
                    if stage_1_first != 0:
                        d_query_rope_tile = None
                        if tla.const_expr(self.has_rope):
                            d_query_rope_tile = query_tile(
                                query_rope, stage_1_query_id, head_offset
                            )
                        load_q(
                            query_tile(query, stage_1_query_id, head_offset),
                            d_query_rope_tile,
                        )
                    stage_qk(
                        (udrain_tick - 1) % PIPELINE_SLOT_COUNT,
                        workspace,
                        stage_1_valid,
                        (udrain_tick - 1) % 2,
                    )

        # Stage 2 cube 段：PV（向量段已在 gather 两半之间发射）。
        if tla.const_expr(udrain_tick >= 2):
            if stage_2_query_id >= 0:
                if stage_2_valid > 0:
                    with tla.cube():
                        stage_pv(
                            (udrain_tick - 2) % PIPELINE_SLOT_COUNT,
                            stage_2_valid,
                            (udrain_tick - 2) % 2,
                        )

        # Stage 3: accumulate or finalize the tile gathered three ticks ago.
        if tla.const_expr(udrain_tick >= 3):
            if stage_3_query_id >= 0:
                with tla.vector():
                    stage_output(
                        stage_3_first,
                        stage_3_last,
                        stage_3_valid,
                        (udrain_tick - 3) % PIPELINE_SLOT_COUNT,
                        stage_3_query_id,
                        head_offset,
                        output,
                        maxima,
                        sums,
                    )

    # `work_ticks == 2` 档位的全静态展开体。
    # 结构同 `_tick_loop_unrolled`：状态在方法内自建，方法无返回值，
    # 原循环之后只做抽干令牌等待。2 段工作 tick + 3 段抽干 tick 全部写死，
    # 归纳变量换成字面量（抽干段用全局 tick 号 work_ticks+i，槽位相位才连续），
    # `tick >= N` 守卫用 tla.const_expr 在编译期求值，循环结构整体消失。
    @tla.jit
    def _tick_loop_unrolled_w2(
        self,
        output,
        query,
        key,
        indices,
        query_rope,
        key_rope,
        sinks,
        maxima,
        sums,
        workspace,
        table,
        cu_q,
        cu_kv,
        used_q,
        used_kv,
        scale,
    ):
        core_group = self.block_idx // self.head_chunks
        head_offset = self.block_idx % self.head_chunks * self.group_size

        # Determine the query range assigned to this core.
        first_query = core_group * self.queries_per_group
        query_end = first_query + self.queries_per_group
        if query_end > self.total_queries:
            query_end = tla.as_numeric(self.total_queries)
        current_query = first_query
        query_count = query_end - first_query

        # 动态 tick 循环体内不能出现 `self.method(...)`（前端只允许对句柄调用方法），
        # 先把所有 stage 入口提升为局部名再进循环（tla 报错原文的 "Hoist the call"）。
        prepare_sparse_query = self._prepare_sparse_query
        find_valid_end = self._find_valid_end
        stage_gather = self._stage_gather
        wait3 = self._wait3
        load_q = self.matmul.load_q
        query_tile = self._query_tile
        stage_qk = self._stage_qk
        init_query_state = self._init_query_state
        stage_softmax = self._stage_softmax
        stage_pv = self._stage_pv
        stage_output = self._stage_output

        subblock_idx = self.subblock_idx

        # Initialize sparse-block traversal and sequence-boundary state.
        sparse_idx = tla.as_numeric(0)
        is_first_block = tla.as_numeric(1)
        batch = tla.as_numeric(0)
        sparse_end = tla.as_numeric(0)
        is_query_active = tla.as_numeric(0)

        # 4 级流水线状态：每字段 3 级显式移位。
        stage_0_query_id = tla.as_numeric(EMPTY_QUERY_ID)
        stage_1_query_id = tla.as_numeric(EMPTY_QUERY_ID)
        stage_2_query_id = tla.as_numeric(EMPTY_QUERY_ID)
        stage_3_query_id = tla.as_numeric(EMPTY_QUERY_ID)
        stage_0_valid = tla.as_numeric(0)
        stage_1_valid = tla.as_numeric(0)
        stage_2_valid = tla.as_numeric(0)
        stage_3_valid = tla.as_numeric(0)
        stage_0_first = tla.as_numeric(0)
        stage_1_first = tla.as_numeric(0)
        stage_2_first = tla.as_numeric(0)
        stage_3_first = tla.as_numeric(0)
        stage_0_last = tla.as_numeric(0)
        stage_1_last = tla.as_numeric(0)
        stage_2_last = tla.as_numeric(0)
        stage_3_last = tla.as_numeric(0)
        stage_0_active = tla.as_numeric(0)
        stage_1_active = tla.as_numeric(0)
        stage_2_active = tla.as_numeric(0)
        stage_3_active = tla.as_numeric(0)

        # 工作段 1/2：原工作循环的第 1 次迭代（tick ≡ 0）。
        if tla.const_expr(True):
            utick = 0
            # DelayLine 移位：tap(k) == stage_k
            stage_3_query_id = stage_2_query_id
            stage_2_query_id = stage_1_query_id
            stage_1_query_id = stage_0_query_id
            stage_3_valid = stage_2_valid
            stage_2_valid = stage_1_valid
            stage_1_valid = stage_0_valid
            stage_3_first = stage_2_first
            stage_2_first = stage_1_first
            stage_1_first = stage_0_first
            stage_3_last = stage_2_last
            stage_2_last = stage_1_last
            stage_1_last = stage_0_last
            stage_3_active = stage_2_active
            stage_2_active = stage_1_active
            stage_1_active = stage_0_active

            # Stage 0: gather a KV tile and register its task.
            valid_kv_count = tla.as_numeric(0)
            gather_active = tla.as_numeric(0)
            gather_slot = tla.as_numeric(0)
            gather_query = tla.as_numeric(0)
            gather_index = tla.as_numeric(0)
            gather_rows = tla.as_numeric(0)
            gather_lo = tla.as_numeric(0)
            gather_mid = tla.as_numeric(0)
            gather_hi = tla.as_numeric(0)
            if current_query < query_end:
                if is_first_block != 0:
                    sparse_idx = tla.as_numeric(0)
                    is_query_active = tla.as_numeric(0)
                    sparse_end = tla.as_numeric(0)
                    (
                        batch,
                        unused_kv_length,
                        unused_kv_end,
                        planned_sparse_end,
                        is_query_active,
                    ) = prepare_sparse_query(
                        query,
                        key,
                        table,
                        cu_q,
                        cu_kv,
                        used_q,
                        used_kv,
                        current_query,
                    )
                    sparse_end = find_valid_end(
                        indices, current_query, planned_sparse_end
                    )

                # Gather at most one TILE_S2-sized block.
                if sparse_idx < sparse_end:
                    tile_rows = sparse_end - sparse_idx
                    if tile_rows > TILE_S2:
                        tile_rows = tla.as_numeric(TILE_S2)
                    valid_kv_count = tile_rows
                    gather_active = tla.as_numeric(1)
                    gather_slot = utick % PIPELINE_SLOT_COUNT
                    gather_query = current_query
                    gather_index = sparse_idx
                    gather_rows = tile_rows
                    # 本 AIV 的行区间 [gather_lo, gather_hi) 与中点（按 chunk 对齐）。
                    # 切分逻辑原样保留，只是变成标量侧的两段发射。
                    rows_per_aiv_round = AIV_COUNT * SPARSE_INDEX_PAIR_SIZE
                    pair_loops = (
                        tile_rows + rows_per_aiv_round - 1
                    ) // rows_per_aiv_round
                    split = pair_loops * SPARSE_INDEX_PAIR_SIZE
                    if split > tile_rows:
                        split = tile_rows
                    aligned_split = (
                        (split + GATHER_CHUNK_ROWS - 1) // GATHER_CHUNK_ROWS
                    ) * GATHER_CHUNK_ROWS
                    if aligned_split < tile_rows:
                        split = aligned_split
                    else:
                        split = tile_rows
                    gather_lo = tla.as_numeric(0)
                    gather_hi = tile_rows
                    if subblock_idx != 0:
                        gather_lo = split
                    if subblock_idx == 0:
                        gather_hi = split
                    gather_mid = gather_lo + (
                        ((gather_hi - gather_lo) // 2 // GATHER_CHUNK_ROWS)
                        * GATHER_CHUNK_ROWS
                    )
                    with tla.vector():
                        # 三槽环形工作区的空槽令牌由 AIC 在循环前预置，
                        # 因此每次 gather 都无条件等待。
                        wait3(self.ws_free, gather_slot, tla.arch.MTE3)
                        stage_gather(
                            gather_slot,
                            gather_query,
                            gather_index,
                            gather_rows,
                            batch,
                            key,
                            key_rope,
                            indices,
                            table,
                            cu_kv,
                            workspace,
                            gather_lo,
                            gather_mid,
                            tla.as_numeric(0),
                        )
                    sparse_idx = sparse_idx + tile_rows

                # The last block lets the gather stage move to the next Query.
                is_last_block = tla.as_numeric(0)
                if sparse_idx >= sparse_end:
                    is_last_block = tla.as_numeric(1)
                stage_0_query_id = current_query
                stage_0_valid = valid_kv_count
                stage_0_first = is_first_block
                stage_0_last = is_last_block
                stage_0_active = is_query_active
                is_first_block = tla.as_numeric(0)
                if is_last_block != 0:
                    current_query = current_query + 1
                    is_first_block = tla.as_numeric(1)
            else:
                # Fill unused ticks so older tasks can finish the pipeline.
                stage_0_query_id = tla.as_numeric(EMPTY_QUERY_ID)
                stage_0_valid = tla.as_numeric(0)
                stage_0_first = tla.as_numeric(0)
                stage_0_last = tla.as_numeric(0)
                stage_0_active = tla.as_numeric(0)

            # Stage 2 的向量段（init / softmax / store_p）提前到 gather 的两半
            # 之间发射。它消费 utick-2 的 tile，与本轮 gather 无数据依赖；向量执行与
            # gather 的标量链重叠后，store_p 的 V→MTE3 交棒等待被吸收。
            if tla.const_expr(utick >= 2):
                if stage_2_query_id >= 0:
                    state_slot = stage_2_query_id % PIPELINE_SLOT_COUNT
                    if stage_2_first != 0:
                        with tla.vector():
                            init_query_state(
                                sinks,
                                head_offset,
                                (
                                    stage_2_valid
                                    if stage_2_active != 0
                                    else tla.as_numeric(-1)
                                ),
                                state_slot,
                            )
                    if stage_2_valid > 0:
                        with tla.vector():
                            stage_softmax(
                                stage_2_first,
                                stage_2_valid,
                                (utick - 2) % PIPELINE_SLOT_COUNT,
                                state_slot,
                                scale,
                                (utick - 2) % 2,
                                (utick - 2) % 2,
                            )

            # gather 下半：续发余下的 chunk，并在最后交棒 ws_ready。
            if gather_active != 0:
                with tla.vector():
                    stage_gather(
                        gather_slot,
                        gather_query,
                        gather_index,
                        gather_rows,
                        batch,
                        key,
                        key_rope,
                        indices,
                        table,
                        cu_kv,
                        workspace,
                        gather_mid,
                        gather_hi,
                        tla.as_numeric(1),
                    )

            # Stage 1: compute QK for the tile gathered one utick ago.
            if tla.const_expr(utick >= 1):
                if stage_1_query_id >= 0 and stage_1_valid > 0:
                    with tla.cube():
                        if stage_1_first != 0:
                            query_rope_tile = None
                            if tla.const_expr(self.has_rope):
                                query_rope_tile = query_tile(
                                    query_rope, stage_1_query_id, head_offset
                                )
                            load_q(
                                query_tile(query, stage_1_query_id, head_offset),
                                query_rope_tile,
                            )
                        stage_qk(
                            (utick - 1) % PIPELINE_SLOT_COUNT,
                            workspace,
                            stage_1_valid,
                            (utick - 1) % 2,
                        )

            # Stage 2 cube 段：PV（向量段已在 gather 两半之间发射）。
            if tla.const_expr(utick >= 2):
                if stage_2_query_id >= 0:
                    if stage_2_valid > 0:
                        with tla.cube():
                            stage_pv(
                                (utick - 2) % PIPELINE_SLOT_COUNT,
                                stage_2_valid,
                                (utick - 2) % 2,
                            )

            # Stage 3: accumulate or finalize the tile gathered three ticks ago.
            if tla.const_expr(utick >= 3):
                if stage_3_query_id >= 0:
                    with tla.vector():
                        stage_output(
                            stage_3_first,
                            stage_3_last,
                            stage_3_valid,
                            (utick - 3) % PIPELINE_SLOT_COUNT,
                            stage_3_query_id,
                            head_offset,
                            output,
                            maxima,
                            sums,
                        )

        # 工作段 2/2：原工作循环的第 2 次迭代（tick ≡ 1）。
        if tla.const_expr(True):
            utick = 1
            # DelayLine 移位：tap(k) == stage_k
            stage_3_query_id = stage_2_query_id
            stage_2_query_id = stage_1_query_id
            stage_1_query_id = stage_0_query_id
            stage_3_valid = stage_2_valid
            stage_2_valid = stage_1_valid
            stage_1_valid = stage_0_valid
            stage_3_first = stage_2_first
            stage_2_first = stage_1_first
            stage_1_first = stage_0_first
            stage_3_last = stage_2_last
            stage_2_last = stage_1_last
            stage_1_last = stage_0_last
            stage_3_active = stage_2_active
            stage_2_active = stage_1_active
            stage_1_active = stage_0_active

            # Stage 0: gather a KV tile and register its task.
            valid_kv_count = tla.as_numeric(0)
            gather_active = tla.as_numeric(0)
            gather_slot = tla.as_numeric(0)
            gather_query = tla.as_numeric(0)
            gather_index = tla.as_numeric(0)
            gather_rows = tla.as_numeric(0)
            gather_lo = tla.as_numeric(0)
            gather_mid = tla.as_numeric(0)
            gather_hi = tla.as_numeric(0)
            if current_query < query_end:
                if is_first_block != 0:
                    sparse_idx = tla.as_numeric(0)
                    is_query_active = tla.as_numeric(0)
                    sparse_end = tla.as_numeric(0)
                    (
                        batch,
                        unused_kv_length,
                        unused_kv_end,
                        planned_sparse_end,
                        is_query_active,
                    ) = prepare_sparse_query(
                        query,
                        key,
                        table,
                        cu_q,
                        cu_kv,
                        used_q,
                        used_kv,
                        current_query,
                    )
                    sparse_end = find_valid_end(
                        indices, current_query, planned_sparse_end
                    )

                # Gather at most one TILE_S2-sized block.
                if sparse_idx < sparse_end:
                    tile_rows = sparse_end - sparse_idx
                    if tile_rows > TILE_S2:
                        tile_rows = tla.as_numeric(TILE_S2)
                    valid_kv_count = tile_rows
                    gather_active = tla.as_numeric(1)
                    gather_slot = utick % PIPELINE_SLOT_COUNT
                    gather_query = current_query
                    gather_index = sparse_idx
                    gather_rows = tile_rows
                    # 本 AIV 的行区间 [gather_lo, gather_hi) 与中点（按 chunk 对齐）。
                    # 切分逻辑原样保留，只是变成标量侧的两段发射。
                    rows_per_aiv_round = AIV_COUNT * SPARSE_INDEX_PAIR_SIZE
                    pair_loops = (
                        tile_rows + rows_per_aiv_round - 1
                    ) // rows_per_aiv_round
                    split = pair_loops * SPARSE_INDEX_PAIR_SIZE
                    if split > tile_rows:
                        split = tile_rows
                    aligned_split = (
                        (split + GATHER_CHUNK_ROWS - 1) // GATHER_CHUNK_ROWS
                    ) * GATHER_CHUNK_ROWS
                    if aligned_split < tile_rows:
                        split = aligned_split
                    else:
                        split = tile_rows
                    gather_lo = tla.as_numeric(0)
                    gather_hi = tile_rows
                    if subblock_idx != 0:
                        gather_lo = split
                    if subblock_idx == 0:
                        gather_hi = split
                    gather_mid = gather_lo + (
                        ((gather_hi - gather_lo) // 2 // GATHER_CHUNK_ROWS)
                        * GATHER_CHUNK_ROWS
                    )
                    with tla.vector():
                        # 三槽环形工作区的空槽令牌由 AIC 在循环前预置，
                        # 因此每次 gather 都无条件等待。
                        wait3(self.ws_free, gather_slot, tla.arch.MTE3)
                        stage_gather(
                            gather_slot,
                            gather_query,
                            gather_index,
                            gather_rows,
                            batch,
                            key,
                            key_rope,
                            indices,
                            table,
                            cu_kv,
                            workspace,
                            gather_lo,
                            gather_mid,
                            tla.as_numeric(0),
                        )
                    sparse_idx = sparse_idx + tile_rows

                # The last block lets the gather stage move to the next Query.
                is_last_block = tla.as_numeric(0)
                if sparse_idx >= sparse_end:
                    is_last_block = tla.as_numeric(1)
                stage_0_query_id = current_query
                stage_0_valid = valid_kv_count
                stage_0_first = is_first_block
                stage_0_last = is_last_block
                stage_0_active = is_query_active
                is_first_block = tla.as_numeric(0)
                if is_last_block != 0:
                    current_query = current_query + 1
                    is_first_block = tla.as_numeric(1)
            else:
                # Fill unused ticks so older tasks can finish the pipeline.
                stage_0_query_id = tla.as_numeric(EMPTY_QUERY_ID)
                stage_0_valid = tla.as_numeric(0)
                stage_0_first = tla.as_numeric(0)
                stage_0_last = tla.as_numeric(0)
                stage_0_active = tla.as_numeric(0)

            # Stage 2 的向量段（init / softmax / store_p）提前到 gather 的两半
            # 之间发射。它消费 utick-2 的 tile，与本轮 gather 无数据依赖；向量执行与
            # gather 的标量链重叠后，store_p 的 V→MTE3 交棒等待被吸收。
            if tla.const_expr(utick >= 2):
                if stage_2_query_id >= 0:
                    state_slot = stage_2_query_id % PIPELINE_SLOT_COUNT
                    if stage_2_first != 0:
                        with tla.vector():
                            init_query_state(
                                sinks,
                                head_offset,
                                (
                                    stage_2_valid
                                    if stage_2_active != 0
                                    else tla.as_numeric(-1)
                                ),
                                state_slot,
                            )
                    if stage_2_valid > 0:
                        with tla.vector():
                            stage_softmax(
                                stage_2_first,
                                stage_2_valid,
                                (utick - 2) % PIPELINE_SLOT_COUNT,
                                state_slot,
                                scale,
                                (utick - 2) % 2,
                                (utick - 2) % 2,
                            )

            # gather 下半：续发余下的 chunk，并在最后交棒 ws_ready。
            if gather_active != 0:
                with tla.vector():
                    stage_gather(
                        gather_slot,
                        gather_query,
                        gather_index,
                        gather_rows,
                        batch,
                        key,
                        key_rope,
                        indices,
                        table,
                        cu_kv,
                        workspace,
                        gather_mid,
                        gather_hi,
                        tla.as_numeric(1),
                    )

            # Stage 1: compute QK for the tile gathered one utick ago.
            if tla.const_expr(utick >= 1):
                if stage_1_query_id >= 0 and stage_1_valid > 0:
                    with tla.cube():
                        if stage_1_first != 0:
                            query_rope_tile = None
                            if tla.const_expr(self.has_rope):
                                query_rope_tile = query_tile(
                                    query_rope, stage_1_query_id, head_offset
                                )
                            load_q(
                                query_tile(query, stage_1_query_id, head_offset),
                                query_rope_tile,
                            )
                        stage_qk(
                            (utick - 1) % PIPELINE_SLOT_COUNT,
                            workspace,
                            stage_1_valid,
                            (utick - 1) % 2,
                        )

            # Stage 2 cube 段：PV（向量段已在 gather 两半之间发射）。
            if tla.const_expr(utick >= 2):
                if stage_2_query_id >= 0:
                    if stage_2_valid > 0:
                        with tla.cube():
                            stage_pv(
                                (utick - 2) % PIPELINE_SLOT_COUNT,
                                stage_2_valid,
                                (utick - 2) % 2,
                            )

            # Stage 3: accumulate or finalize the tile gathered three ticks ago.
            if tla.const_expr(utick >= 3):
                if stage_3_query_id >= 0:
                    with tla.vector():
                        stage_output(
                            stage_3_first,
                            stage_3_last,
                            stage_3_valid,
                            (utick - 3) % PIPELINE_SLOT_COUNT,
                            stage_3_query_id,
                            head_offset,
                            output,
                            maxima,
                            sums,
                        )

        # 抽干段 1/3：原抽干循环的第 1 次迭代（drain_tick ≡ 2 = work_ticks+0）。
        if tla.const_expr(True):
            udrain_tick = 2
            # DelayLine 移位：tap(k) == stage_k
            stage_3_query_id = stage_2_query_id
            stage_2_query_id = stage_1_query_id
            stage_1_query_id = stage_0_query_id
            stage_3_valid = stage_2_valid
            stage_2_valid = stage_1_valid
            stage_1_valid = stage_0_valid
            stage_3_first = stage_2_first
            stage_2_first = stage_1_first
            stage_1_first = stage_0_first
            stage_3_last = stage_2_last
            stage_2_last = stage_1_last
            stage_1_last = stage_0_last
            stage_3_active = stage_2_active
            stage_2_active = stage_1_active
            stage_1_active = stage_0_active

            # 空 tick：老任务继续走完流水线（原 else 分支原样）。
            stage_0_query_id = tla.as_numeric(EMPTY_QUERY_ID)
            stage_0_valid = tla.as_numeric(0)
            stage_0_first = tla.as_numeric(0)
            stage_0_last = tla.as_numeric(0)
            stage_0_active = tla.as_numeric(0)

            # Stage 2 的向量段（init / softmax / store_p）提前到 gather 的两半
            # 之间发射。它消费 tick-2 的 tile，与本轮 gather 无数据依赖；向量执行与
            # gather 的标量链重叠后，store_p 的 V→MTE3 交棒等待被吸收。
            if tla.const_expr(udrain_tick >= 2):
                if stage_2_query_id >= 0:
                    d_state_slot = stage_2_query_id % PIPELINE_SLOT_COUNT
                    if stage_2_first != 0:
                        with tla.vector():
                            init_query_state(
                                sinks,
                                head_offset,
                                (
                                    stage_2_valid
                                    if stage_2_active != 0
                                    else tla.as_numeric(-1)
                                ),
                                d_state_slot,
                            )
                    if stage_2_valid > 0:
                        with tla.vector():
                            stage_softmax(
                                stage_2_first,
                                stage_2_valid,
                                (udrain_tick - 2) % PIPELINE_SLOT_COUNT,
                                d_state_slot,
                                scale,
                                (udrain_tick - 2) % 2,
                                (udrain_tick - 2) % 2,
                            )

            # Stage 1: compute QK for the tile gathered one tick ago.
            if tla.const_expr(udrain_tick >= 1):
                if stage_1_query_id >= 0 and stage_1_valid > 0:
                    with tla.cube():
                        if stage_1_first != 0:
                            d_query_rope_tile = None
                            if tla.const_expr(self.has_rope):
                                d_query_rope_tile = query_tile(
                                    query_rope, stage_1_query_id, head_offset
                                )
                            load_q(
                                query_tile(query, stage_1_query_id, head_offset),
                                d_query_rope_tile,
                            )
                        stage_qk(
                            (udrain_tick - 1) % PIPELINE_SLOT_COUNT,
                            workspace,
                            stage_1_valid,
                            (udrain_tick - 1) % 2,
                        )

            # Stage 2 cube 段：PV（向量段已在 gather 两半之间发射）。
            if tla.const_expr(udrain_tick >= 2):
                if stage_2_query_id >= 0:
                    if stage_2_valid > 0:
                        with tla.cube():
                            stage_pv(
                                (udrain_tick - 2) % PIPELINE_SLOT_COUNT,
                                stage_2_valid,
                                (udrain_tick - 2) % 2,
                            )

            # Stage 3: accumulate or finalize the tile gathered three ticks ago.
            if tla.const_expr(udrain_tick >= 3):
                if stage_3_query_id >= 0:
                    with tla.vector():
                        stage_output(
                            stage_3_first,
                            stage_3_last,
                            stage_3_valid,
                            (udrain_tick - 3) % PIPELINE_SLOT_COUNT,
                            stage_3_query_id,
                            head_offset,
                            output,
                            maxima,
                            sums,
                        )

        # 抽干段 2/3：原抽干循环的第 2 次迭代（drain_tick ≡ 3 = work_ticks+1）。
        if tla.const_expr(True):
            udrain_tick = 3
            # DelayLine 移位：tap(k) == stage_k
            stage_3_query_id = stage_2_query_id
            stage_2_query_id = stage_1_query_id
            stage_1_query_id = stage_0_query_id
            stage_3_valid = stage_2_valid
            stage_2_valid = stage_1_valid
            stage_1_valid = stage_0_valid
            stage_3_first = stage_2_first
            stage_2_first = stage_1_first
            stage_1_first = stage_0_first
            stage_3_last = stage_2_last
            stage_2_last = stage_1_last
            stage_1_last = stage_0_last
            stage_3_active = stage_2_active
            stage_2_active = stage_1_active
            stage_1_active = stage_0_active

            # 空 tick：老任务继续走完流水线（原 else 分支原样）。
            stage_0_query_id = tla.as_numeric(EMPTY_QUERY_ID)
            stage_0_valid = tla.as_numeric(0)
            stage_0_first = tla.as_numeric(0)
            stage_0_last = tla.as_numeric(0)
            stage_0_active = tla.as_numeric(0)

            # Stage 2 的向量段（init / softmax / store_p）提前到 gather 的两半
            # 之间发射。它消费 tick-2 的 tile，与本轮 gather 无数据依赖；向量执行与
            # gather 的标量链重叠后，store_p 的 V→MTE3 交棒等待被吸收。
            if tla.const_expr(udrain_tick >= 2):
                if stage_2_query_id >= 0:
                    d_state_slot = stage_2_query_id % PIPELINE_SLOT_COUNT
                    if stage_2_first != 0:
                        with tla.vector():
                            init_query_state(
                                sinks,
                                head_offset,
                                (
                                    stage_2_valid
                                    if stage_2_active != 0
                                    else tla.as_numeric(-1)
                                ),
                                d_state_slot,
                            )
                    if stage_2_valid > 0:
                        with tla.vector():
                            stage_softmax(
                                stage_2_first,
                                stage_2_valid,
                                (udrain_tick - 2) % PIPELINE_SLOT_COUNT,
                                d_state_slot,
                                scale,
                                (udrain_tick - 2) % 2,
                                (udrain_tick - 2) % 2,
                            )

            # Stage 1: compute QK for the tile gathered one tick ago.
            if tla.const_expr(udrain_tick >= 1):
                if stage_1_query_id >= 0 and stage_1_valid > 0:
                    with tla.cube():
                        if stage_1_first != 0:
                            d_query_rope_tile = None
                            if tla.const_expr(self.has_rope):
                                d_query_rope_tile = query_tile(
                                    query_rope, stage_1_query_id, head_offset
                                )
                            load_q(
                                query_tile(query, stage_1_query_id, head_offset),
                                d_query_rope_tile,
                            )
                        stage_qk(
                            (udrain_tick - 1) % PIPELINE_SLOT_COUNT,
                            workspace,
                            stage_1_valid,
                            (udrain_tick - 1) % 2,
                        )

            # Stage 2 cube 段：PV（向量段已在 gather 两半之间发射）。
            if tla.const_expr(udrain_tick >= 2):
                if stage_2_query_id >= 0:
                    if stage_2_valid > 0:
                        with tla.cube():
                            stage_pv(
                                (udrain_tick - 2) % PIPELINE_SLOT_COUNT,
                                stage_2_valid,
                                (udrain_tick - 2) % 2,
                            )

            # Stage 3: accumulate or finalize the tile gathered three ticks ago.
            if tla.const_expr(udrain_tick >= 3):
                if stage_3_query_id >= 0:
                    with tla.vector():
                        stage_output(
                            stage_3_first,
                            stage_3_last,
                            stage_3_valid,
                            (udrain_tick - 3) % PIPELINE_SLOT_COUNT,
                            stage_3_query_id,
                            head_offset,
                            output,
                            maxima,
                            sums,
                        )

        # 抽干段 3/3：原抽干循环的第 3 次迭代（drain_tick ≡ 4 = work_ticks+2）。
        if tla.const_expr(True):
            udrain_tick = 4
            # DelayLine 移位：tap(k) == stage_k
            stage_3_query_id = stage_2_query_id
            stage_2_query_id = stage_1_query_id
            stage_1_query_id = stage_0_query_id
            stage_3_valid = stage_2_valid
            stage_2_valid = stage_1_valid
            stage_1_valid = stage_0_valid
            stage_3_first = stage_2_first
            stage_2_first = stage_1_first
            stage_1_first = stage_0_first
            stage_3_last = stage_2_last
            stage_2_last = stage_1_last
            stage_1_last = stage_0_last
            stage_3_active = stage_2_active
            stage_2_active = stage_1_active
            stage_1_active = stage_0_active

            # 空 tick：老任务继续走完流水线（原 else 分支原样）。
            stage_0_query_id = tla.as_numeric(EMPTY_QUERY_ID)
            stage_0_valid = tla.as_numeric(0)
            stage_0_first = tla.as_numeric(0)
            stage_0_last = tla.as_numeric(0)
            stage_0_active = tla.as_numeric(0)

            # Stage 2 的向量段（init / softmax / store_p）提前到 gather 的两半
            # 之间发射。它消费 tick-2 的 tile，与本轮 gather 无数据依赖；向量执行与
            # gather 的标量链重叠后，store_p 的 V→MTE3 交棒等待被吸收。
            if tla.const_expr(udrain_tick >= 2):
                if stage_2_query_id >= 0:
                    d_state_slot = stage_2_query_id % PIPELINE_SLOT_COUNT
                    if stage_2_first != 0:
                        with tla.vector():
                            init_query_state(
                                sinks,
                                head_offset,
                                (
                                    stage_2_valid
                                    if stage_2_active != 0
                                    else tla.as_numeric(-1)
                                ),
                                d_state_slot,
                            )
                    if stage_2_valid > 0:
                        with tla.vector():
                            stage_softmax(
                                stage_2_first,
                                stage_2_valid,
                                (udrain_tick - 2) % PIPELINE_SLOT_COUNT,
                                d_state_slot,
                                scale,
                                (udrain_tick - 2) % 2,
                                (udrain_tick - 2) % 2,
                            )

            # Stage 1: compute QK for the tile gathered one tick ago.
            if tla.const_expr(udrain_tick >= 1):
                if stage_1_query_id >= 0 and stage_1_valid > 0:
                    with tla.cube():
                        if stage_1_first != 0:
                            d_query_rope_tile = None
                            if tla.const_expr(self.has_rope):
                                d_query_rope_tile = query_tile(
                                    query_rope, stage_1_query_id, head_offset
                                )
                            load_q(
                                query_tile(query, stage_1_query_id, head_offset),
                                d_query_rope_tile,
                            )
                        stage_qk(
                            (udrain_tick - 1) % PIPELINE_SLOT_COUNT,
                            workspace,
                            stage_1_valid,
                            (udrain_tick - 1) % 2,
                        )

            # Stage 2 cube 段：PV（向量段已在 gather 两半之间发射）。
            if tla.const_expr(udrain_tick >= 2):
                if stage_2_query_id >= 0:
                    if stage_2_valid > 0:
                        with tla.cube():
                            stage_pv(
                                (udrain_tick - 2) % PIPELINE_SLOT_COUNT,
                                stage_2_valid,
                                (udrain_tick - 2) % 2,
                            )

            # Stage 3: accumulate or finalize the tile gathered three ticks ago.
            if tla.const_expr(udrain_tick >= 3):
                if stage_3_query_id >= 0:
                    with tla.vector():
                        stage_output(
                            stage_3_first,
                            stage_3_last,
                            stage_3_valid,
                            (udrain_tick - 3) % PIPELINE_SLOT_COUNT,
                            stage_3_query_id,
                            head_offset,
                            output,
                            maxima,
                            sums,
                        )

    @tla.jit
    def __call__(
        self,
        output,
        query,
        key,
        indices,
        query_rope,
        key_rope,
        sinks,
        maxima,
        sums,
        workspace,
        table,
        cu_q,
        cu_kv,
        used_q,
        used_kv,
        scale,
    ):
        # Map this core to a query group and a head tile.
        core_group = self.block_idx // self.head_chunks
        head_offset = self.block_idx % self.head_chunks * self.group_size

        # Determine the query range assigned to this core.
        first_query = core_group * self.queries_per_group
        query_end = first_query + self.queries_per_group
        if query_end > self.total_queries:
            query_end = tla.as_numeric(self.total_queries)
        current_query = first_query
        query_count = query_end - first_query

        # 动态 tick 循环体内不能出现 `self.method(...)`（前端只允许对句柄调用方法），
        # 先把所有 stage 入口提升为局部名再进循环（tla 报错原文的 "Hoist the call"）。
        prepare_sparse_query = self._prepare_sparse_query
        find_valid_end = self._find_valid_end
        stage_gather = self._stage_gather
        wait3 = self._wait3
        load_q = self.matmul.load_q
        query_tile = self._query_tile
        stage_qk = self._stage_qk
        init_query_state = self._init_query_state
        stage_softmax = self._stage_softmax
        stage_pv = self._stage_pv
        stage_output = self._stage_output

        subblock_idx = self.subblock_idx

        # Initialize sparse-block traversal and sequence-boundary state.
        sparse_idx = tla.as_numeric(0)
        is_first_block = tla.as_numeric(1)
        batch = tla.as_numeric(0)
        sparse_end = tla.as_numeric(0)
        is_query_active = tla.as_numeric(0)

        # 4 级流水线状态：每字段 3 级显式移位。
        stage_0_query_id = tla.as_numeric(EMPTY_QUERY_ID)
        stage_1_query_id = tla.as_numeric(EMPTY_QUERY_ID)
        stage_2_query_id = tla.as_numeric(EMPTY_QUERY_ID)
        stage_3_query_id = tla.as_numeric(EMPTY_QUERY_ID)
        stage_0_valid = tla.as_numeric(0)
        stage_1_valid = tla.as_numeric(0)
        stage_2_valid = tla.as_numeric(0)
        stage_3_valid = tla.as_numeric(0)
        stage_0_first = tla.as_numeric(0)
        stage_1_first = tla.as_numeric(0)
        stage_2_first = tla.as_numeric(0)
        stage_3_first = tla.as_numeric(0)
        stage_0_last = tla.as_numeric(0)
        stage_1_last = tla.as_numeric(0)
        stage_2_last = tla.as_numeric(0)
        stage_3_last = tla.as_numeric(0)
        stage_0_active = tla.as_numeric(0)
        stage_1_active = tla.as_numeric(0)
        stage_2_active = tla.as_numeric(0)
        stage_3_active = tla.as_numeric(0)

        # Each tick gathers a new KV tile while older tiles move through
        # QK, softmax/PV, and output. The final three ticks drain the pipeline.
        #
        # `query_count` 从 `block_idx` 算出的是每个 core 实际分到的 query 数。
        # tla 前端把 `tla.range` 的**动态**上界降级后，非首/尾 block 的循环体
        # 没有跑到 stage 3（实测：block 1..4 的 stage 0 已消费掉 query，但
        # `_stage_output` 一次都没进）。这里改用 constexpr `queries_per_group`
        # 作上界：它按 `_calculate_core_partition` 的定义是所有分组的上界，
        # 多出的 tick 走 `current_query >= query_end` 的空分支，循环次数变成
        # 编译期常量。
        total_ticks = (
            self.queries_per_group * ((self.sparse_count + TILE_S2 - 1) // TILE_S2)
            + PIPELINE_DRAIN_TICKS
        )
        # `total_ticks` 是每个 block 的 tile 数上界（每 tick 最多启动一个 tile），
        # 跑完 `work_ticks` 个 tick 后本 block 的所有 tile 都已启动，
        # `current_query >= query_end` 恒成立；原循环在最后 `PIPELINE_DRAIN_TICKS`
        # 个 tick 里走的一直是空分支。因此把循环拆成「工作循环 + 抽干循环」，
        # 抽干循环的循环体里不再出现 Stage 0（prepare/find_valid_end/gather 前置体），
        # 语义不变。
        work_ticks = total_ticks - PIPELINE_DRAIN_TICKS
        # “初始空槽”语义：空闲侧先预置 depth 个令牌，tick 循环结束后同一侧
        # 再抽干，使每个 cross flag 的 set/wait 计数严格相等（计数不为零时
        # 硬件报 "The set and wait instructions do not match"）。
        with tla.cube():
            tla.cross_core_set_flag(self.ws_free[0], tla.arch.MTE2)
            tla.cross_core_set_flag(self.ws_free[1], tla.arch.MTE2)
            tla.cross_core_set_flag(self.ws_free[2], tla.arch.MTE2)
            tla.cross_core_set_flag(self.p_free, tla.arch.MTE1)
        with tla.vector():
            tla.cross_core_set_flag(self.qk_free[0], tla.arch.VECTOR)
            tla.cross_core_set_flag(self.qk_free[1], tla.arch.VECTOR)
            tla.cross_core_set_flag(self.pv_free, tla.arch.VECTOR)

        # trip count 为编译期常量时的静态展开。
        # `work_ticks = queries_per_group * ceil(sparse_count/TILE_S2)` 只由两个
        # `tla.Constexpr` 算出，所以 `tla.const_expr(work_ticks == 1)` 是编译期判定。
        # 展开体与循环第 k 次迭代逐条对应：同一语句、同一 DelayLine 移位，归纳变量换成
        # 字面量、守卫写成 `tla.const_expr(<字面量比较>)`（代入后结果与循环内相同，但不再发射）。
        if tla.const_expr(work_ticks == 1):
            self._tick_loop_unrolled(
                output,
                query,
                key,
                indices,
                query_rope,
                key_rope,
                sinks,
                maxima,
                sums,
                workspace,
                table,
                cu_q,
                cu_kv,
                used_q,
                used_kv,
                scale,
            )
        elif tla.const_expr(work_ticks == 2):
            # work_ticks == 2 的全静态展开。
            self._tick_loop_unrolled_w2(
                output,
                query,
                key,
                indices,
                query_rope,
                key_rope,
                sinks,
                maxima,
                sums,
                workspace,
                table,
                cu_q,
                cu_kv,
                used_q,
                used_kv,
                scale,
            )
        else:
            # 原路径：work_ticks 不为 1/2 时不展开（大 trip count 例保持原形态）。
            for tick in tla.range(0, work_ticks, 1):
                # DelayLine 移位：tap(k) == stage_k
                stage_3_query_id = stage_2_query_id
                stage_2_query_id = stage_1_query_id
                stage_1_query_id = stage_0_query_id
                stage_3_valid = stage_2_valid
                stage_2_valid = stage_1_valid
                stage_1_valid = stage_0_valid
                stage_3_first = stage_2_first
                stage_2_first = stage_1_first
                stage_1_first = stage_0_first
                stage_3_last = stage_2_last
                stage_2_last = stage_1_last
                stage_1_last = stage_0_last
                stage_3_active = stage_2_active
                stage_2_active = stage_1_active
                stage_1_active = stage_0_active

                # Stage 0: gather a KV tile and register its task.
                valid_kv_count = tla.as_numeric(0)
                gather_active = tla.as_numeric(0)
                gather_slot = tla.as_numeric(0)
                gather_query = tla.as_numeric(0)
                gather_index = tla.as_numeric(0)
                gather_rows = tla.as_numeric(0)
                gather_lo = tla.as_numeric(0)
                gather_mid = tla.as_numeric(0)
                gather_hi = tla.as_numeric(0)
                if current_query < query_end:
                    if is_first_block != 0:
                        sparse_idx = tla.as_numeric(0)
                        is_query_active = tla.as_numeric(0)
                        sparse_end = tla.as_numeric(0)
                        (
                            batch,
                            unused_kv_length,
                            unused_kv_end,
                            planned_sparse_end,
                            is_query_active,
                        ) = prepare_sparse_query(
                            query,
                            key,
                            table,
                            cu_q,
                            cu_kv,
                            used_q,
                            used_kv,
                            current_query,
                        )
                        sparse_end = find_valid_end(
                            indices, current_query, planned_sparse_end
                        )

                    # Gather at most one TILE_S2-sized block.
                    if sparse_idx < sparse_end:
                        tile_rows = sparse_end - sparse_idx
                        if tile_rows > TILE_S2:
                            tile_rows = tla.as_numeric(TILE_S2)
                        valid_kv_count = tile_rows
                        gather_active = tla.as_numeric(1)
                        gather_slot = tick % PIPELINE_SLOT_COUNT
                        gather_query = current_query
                        gather_index = sparse_idx
                        gather_rows = tile_rows
                        # 本 AIV 的行区间 [gather_lo, gather_hi) 与中点（按 chunk 对齐）。
                        # 切分逻辑原样保留，只是变成标量侧的两段发射。
                        rows_per_aiv_round = AIV_COUNT * SPARSE_INDEX_PAIR_SIZE
                        pair_loops = (
                            tile_rows + rows_per_aiv_round - 1
                        ) // rows_per_aiv_round
                        split = pair_loops * SPARSE_INDEX_PAIR_SIZE
                        if split > tile_rows:
                            split = tile_rows
                        aligned_split = (
                            (split + GATHER_CHUNK_ROWS - 1) // GATHER_CHUNK_ROWS
                        ) * GATHER_CHUNK_ROWS
                        if aligned_split < tile_rows:
                            split = aligned_split
                        else:
                            split = tile_rows
                        gather_lo = tla.as_numeric(0)
                        gather_hi = tile_rows
                        if subblock_idx != 0:
                            gather_lo = split
                        if subblock_idx == 0:
                            gather_hi = split
                        gather_mid = gather_lo + (
                            ((gather_hi - gather_lo) // 2 // GATHER_CHUNK_ROWS)
                            * GATHER_CHUNK_ROWS
                        )
                        with tla.vector():
                            # 三槽环形工作区的空槽令牌由 AIC 在循环前预置，
                            # 因此每次 gather 都无条件等待。
                            wait3(self.ws_free, gather_slot, tla.arch.MTE3)
                            stage_gather(
                                gather_slot,
                                gather_query,
                                gather_index,
                                gather_rows,
                                batch,
                                key,
                                key_rope,
                                indices,
                                table,
                                cu_kv,
                                workspace,
                                gather_lo,
                                gather_mid,
                                tla.as_numeric(0),
                            )
                        sparse_idx = sparse_idx + tile_rows

                    # The last block lets the gather stage move to the next Query.
                    is_last_block = tla.as_numeric(0)
                    if sparse_idx >= sparse_end:
                        is_last_block = tla.as_numeric(1)
                    stage_0_query_id = current_query
                    stage_0_valid = valid_kv_count
                    stage_0_first = is_first_block
                    stage_0_last = is_last_block
                    stage_0_active = is_query_active
                    is_first_block = tla.as_numeric(0)
                    if is_last_block != 0:
                        current_query = current_query + 1
                        is_first_block = tla.as_numeric(1)
                else:
                    # Fill unused ticks so older tasks can finish the pipeline.
                    stage_0_query_id = tla.as_numeric(EMPTY_QUERY_ID)
                    stage_0_valid = tla.as_numeric(0)
                    stage_0_first = tla.as_numeric(0)
                    stage_0_last = tla.as_numeric(0)
                    stage_0_active = tla.as_numeric(0)

                # Stage 2 的向量段（init / softmax / store_p）提前到 gather 的两半
                # 之间发射。它消费 tick-2 的 tile，与本轮 gather 无数据依赖；向量执行与
                # gather 的标量链重叠后，store_p 的 V→MTE3 交棒等待被吸收。
                if tick >= 2:
                    if stage_2_query_id >= 0:
                        state_slot = stage_2_query_id % PIPELINE_SLOT_COUNT
                        if stage_2_first != 0:
                            with tla.vector():
                                init_query_state(
                                    sinks,
                                    head_offset,
                                    (
                                        stage_2_valid
                                        if stage_2_active != 0
                                        else tla.as_numeric(-1)
                                    ),
                                    state_slot,
                                )
                        if stage_2_valid > 0:
                            with tla.vector():
                                stage_softmax(
                                    stage_2_first,
                                    stage_2_valid,
                                    (tick - 2) % PIPELINE_SLOT_COUNT,
                                    state_slot,
                                    scale,
                                    (tick - 2) % 2,
                                    (tick - 2) % 2,
                                )

                # gather 下半：续发余下的 chunk，并在最后交棒 ws_ready。
                if gather_active != 0:
                    with tla.vector():
                        stage_gather(
                            gather_slot,
                            gather_query,
                            gather_index,
                            gather_rows,
                            batch,
                            key,
                            key_rope,
                            indices,
                            table,
                            cu_kv,
                            workspace,
                            gather_mid,
                            gather_hi,
                            tla.as_numeric(1),
                        )

                # Stage 1: compute QK for the tile gathered one tick ago.
                if tick >= 1:
                    if stage_1_query_id >= 0 and stage_1_valid > 0:
                        with tla.cube():
                            if stage_1_first != 0:
                                query_rope_tile = None
                                if tla.const_expr(self.has_rope):
                                    query_rope_tile = query_tile(
                                        query_rope, stage_1_query_id, head_offset
                                    )
                                load_q(
                                    query_tile(query, stage_1_query_id, head_offset),
                                    query_rope_tile,
                                )
                            stage_qk(
                                (tick - 1) % PIPELINE_SLOT_COUNT,
                                workspace,
                                stage_1_valid,
                                (tick - 1) % 2,
                            )

                # Stage 2 cube 段：PV（向量段已在 gather 两半之间发射）。
                if tick >= 2:
                    if stage_2_query_id >= 0:
                        if stage_2_valid > 0:
                            with tla.cube():
                                stage_pv(
                                    (tick - 2) % PIPELINE_SLOT_COUNT,
                                    stage_2_valid,
                                    (tick - 2) % 2,
                                )

                # Stage 3: accumulate or finalize the tile gathered three ticks ago.
                if tick >= 3:
                    if stage_3_query_id >= 0:
                        with tla.vector():
                            stage_output(
                                stage_3_first,
                                stage_3_last,
                                stage_3_valid,
                                (tick - 3) % PIPELINE_SLOT_COUNT,
                                stage_3_query_id,
                                head_offset,
                                output,
                                maxima,
                                sums,
                            )

            # 抽干循环：与工作循环逐条相同，唯独 Stage 0 的整段
            # （prepare_sparse_query / find_valid_end / gather 前置体）不存在。
            for drain_tick in tla.range(work_ticks, total_ticks, 1):
                # DelayLine 移位：tap(k) == stage_k
                stage_3_query_id = stage_2_query_id
                stage_2_query_id = stage_1_query_id
                stage_1_query_id = stage_0_query_id
                stage_3_valid = stage_2_valid
                stage_2_valid = stage_1_valid
                stage_1_valid = stage_0_valid
                stage_3_first = stage_2_first
                stage_2_first = stage_1_first
                stage_1_first = stage_0_first
                stage_3_last = stage_2_last
                stage_2_last = stage_1_last
                stage_1_last = stage_0_last
                stage_3_active = stage_2_active
                stage_2_active = stage_1_active
                stage_1_active = stage_0_active

                # 空 tick：老任务继续走完流水线（原 else 分支原样）。
                stage_0_query_id = tla.as_numeric(EMPTY_QUERY_ID)
                stage_0_valid = tla.as_numeric(0)
                stage_0_first = tla.as_numeric(0)
                stage_0_last = tla.as_numeric(0)
                stage_0_active = tla.as_numeric(0)

                # Stage 2 的向量段（init / softmax / store_p）提前到 gather 的两半
                # 之间发射。它消费 tick-2 的 tile，与本轮 gather 无数据依赖；向量执行与
                # gather 的标量链重叠后，store_p 的 V→MTE3 交棒等待被吸收。
                if drain_tick >= 2:
                    if stage_2_query_id >= 0:
                        d_state_slot = stage_2_query_id % PIPELINE_SLOT_COUNT
                        if stage_2_first != 0:
                            with tla.vector():
                                init_query_state(
                                    sinks,
                                    head_offset,
                                    (
                                        stage_2_valid
                                        if stage_2_active != 0
                                        else tla.as_numeric(-1)
                                    ),
                                    d_state_slot,
                                )
                        if stage_2_valid > 0:
                            with tla.vector():
                                stage_softmax(
                                    stage_2_first,
                                    stage_2_valid,
                                    (drain_tick - 2) % PIPELINE_SLOT_COUNT,
                                    d_state_slot,
                                    scale,
                                    (drain_tick - 2) % 2,
                                    (drain_tick - 2) % 2,
                                )

                # Stage 1: compute QK for the tile gathered one tick ago.
                if drain_tick >= 1:
                    if stage_1_query_id >= 0 and stage_1_valid > 0:
                        with tla.cube():
                            if stage_1_first != 0:
                                d_query_rope_tile = None
                                if tla.const_expr(self.has_rope):
                                    d_query_rope_tile = query_tile(
                                        query_rope, stage_1_query_id, head_offset
                                    )
                                load_q(
                                    query_tile(query, stage_1_query_id, head_offset),
                                    d_query_rope_tile,
                                )
                            stage_qk(
                                (drain_tick - 1) % PIPELINE_SLOT_COUNT,
                                workspace,
                                stage_1_valid,
                                (drain_tick - 1) % 2,
                            )

                # Stage 2 cube 段：PV（向量段已在 gather 两半之间发射）。
                if drain_tick >= 2:
                    if stage_2_query_id >= 0:
                        if stage_2_valid > 0:
                            with tla.cube():
                                stage_pv(
                                    (drain_tick - 2) % PIPELINE_SLOT_COUNT,
                                    stage_2_valid,
                                    (drain_tick - 2) % 2,
                                )

                # Stage 3: accumulate or finalize the tile gathered three ticks ago.
                if drain_tick >= 3:
                    if stage_3_query_id >= 0:
                        with tla.vector():
                            stage_output(
                                stage_3_first,
                                stage_3_last,
                                stage_3_valid,
                                (drain_tick - 3) % PIPELINE_SLOT_COUNT,
                                stage_3_query_id,
                                head_offset,
                                output,
                                maxima,
                                sums,
                            )

        # 抽干预置令牌：生产者侧多等 depth 次，使计数归零。
        with tla.cube():
            self._wait2(self.qk_free, 0, tla.arch.FIX)
            self._wait2(self.qk_free, 1, tla.arch.FIX)
            tla.cross_core_wait_flag(self.pv_free, tla.arch.FIX)
        with tla.vector():
            self._wait3(self.ws_free, 0, tla.arch.MTE3)
            self._wait3(self.ws_free, 1, tla.arch.MTE3)
            self._wait3(self.ws_free, 2, tla.arch.MTE3)
            tla.cross_core_wait_flag(self.p_free, tla.arch.MTE3)


# ---------------------------------------------------------------------------------------------
# 设备 kernel 入口：把宿主传入的布局/形状常量折成编译期参数，构造 kernel 对象并执行。
# ---------------------------------------------------------------------------------------------
# Channel/Buffer 抽象不带核内同步（`produce()/consume()` 只选槽），框架的自动同步
# 对动态槽选择不成立（实测 AIV 侧 get_buf/rls_buf 计数不配对、AIC 报 CUBE
# "The set and wait instructions do not match"），因此逐缓冲显式 mutex。
@tla.kernel
def sparse_flash_attention_kernel(
    output: tla.Tensor,
    query: tla.Tensor,
    key: tla.Tensor,
    indices: tla.Tensor,
    query_rope: tla.Tensor,
    key_rope: tla.Tensor,
    sinks: tla.Tensor,
    maxima: tla.Tensor,
    sums: tla.Tensor,
    workspace: tla.Tensor,
    table: tla.Tensor,
    cu_q: tla.Tensor,
    cu_kv: tla.Tensor,
    used_q: tla.Tensor,
    used_kv: tla.Tensor,
    scale: tla.Float32,
    group_size: tla.Constexpr[int],
    sparse_count: tla.Constexpr[int],
    head_chunks: tla.Constexpr[int],
    queries_per_group: tla.Constexpr[int],
    total_queries: tla.Constexpr[int],
    batch_count: tla.Constexpr[int],
    has_rope: tla.Constexpr[bool],
    use_sinks: tla.Constexpr[bool],
    return_lse: tla.Constexpr[bool],
    layout_query_tnd: tla.Constexpr[bool],
    layout_kv_pa: tla.Constexpr[bool],
    layout_kv_tnd: tla.Constexpr[bool],
    sparse_mode: tla.Constexpr[int],
    q_seq: tla.Constexpr[int],
    kv_seq: tla.Constexpr[int],
    kv_pages: tla.Constexpr[int],
    kv_capacity: tla.Constexpr[int],
    head_count: tla.Constexpr[int],
    has_used_q: tla.Constexpr[bool],
    has_used_kv: tla.Constexpr[bool],
    dtype_bf16: tla.Constexpr[bool],
) -> None:
    # 类字段（缓冲）+ __call__ 的 tick 流水。
    # rank-2 扁平绑定丢失 rank/形状，宿主把 layout 字符串与各容量作为
    # Constexpr 传入。
    layout_query = "TND" if layout_query_tnd else "BSND"
    layout_kv = "PA_BSND" if layout_kv_pa else ("TND" if layout_kv_tnd else "BSND")
    # 宿主按 `query.dtype` 选 bfloat16/float16；tla 的 dtype 必须是编译期常量，
    # 改为等价的一个 Constexpr 布尔。
    dtype_16 = tla.BFloat16 if dtype_bf16 else tla.Float16

    kern = _SparseFlashAttentionKernel(
        group_size,
        sparse_count,
        head_chunks,
        queries_per_group,
        dtype_16,
        use_sinks,
        return_lse,
        head_count,
        total_queries,
        batch_count,
        layout_query,
        layout_kv,
        sparse_mode,
        has_rope,
        q_seq,
        kv_seq,
        kv_pages,
        kv_capacity,
        has_used_q,
        has_used_kv,
    )
    kern(
        output,
        query,
        key,
        indices,
        query_rope,
        key_rope,
        sinks,
        maxima,
        sums,
        workspace,
        table,
        cu_q,
        cu_kv,
        used_q,
        used_kv,
        scale,
    )


# ---------------------------------------------------------------------------------------------
# 宿主侧
# ---------------------------------------------------------------------------------------------
NPU_ARCH = "--npu-arch 3510"

# Bounded FIFO (cap 64) keyed by `id(value)`: an unbounded cache would keep every
# launch's output/workspace tensors alive forever.
_BINDING_FIFO_CAP = 64
_BINDING_CACHE = collections.OrderedDict()


def _tla_tensor(value):
    """Bind a torch tensor zero-copy as a 2-D dynamic-layout TLA tensor.

    tla 只接受 rank<=2 的绑定（实测 rank 3/4 报 "cannot derive layout
    metadata"），因此统一按 `(numel // last_dim, last_dim)` 扁平绑定；
    `last_dim` 与任何 `origin_shape` 下的 stride 1 维一致。

    1-D 张量（`cu_seqlens_*` / `seqused_*` / 占位张量）按 `(numel, 1)` 绑定，
    因此按 `cu[batch, 0]` 的二维写法访问。若按 `(1, numel)` 绑定，
    `cu[batch, 0]` 会取到第 `batch*numel` 个元素（越界），TND 用例因此把每条
    query 判成空行（实测：TND 1 token 的最小用例输出全 0，BSND 同形状逐位相同）。
    """
    key = (id(value), int(value.data_ptr()))
    hit = _BINDING_CACHE.get(key)
    if hit is not None and hit[0] is value:
        return hit[1]
    if value.dim() == 1:
        last = 1
    else:
        last = int(value.shape[-1]) if value.dim() else 1
    rows = int(value.numel()) // last if last else 1
    bound = from_dlpack(
        value, layout_tag=tla.arch.RowMajor, origin_shape=(rows, last)
    ).mark_layout_dynamic()
    if key not in _BINDING_CACHE and len(_BINDING_CACHE) >= _BINDING_FIFO_CAP:
        _BINDING_CACHE.popitem(last=False)
    _BINDING_CACHE[key] = (value, bound)
    return bound


_PLACEHOLDER_CACHE = {}


def _placeholder(device, dtype):
    """Return a shared 1-element zero tensor per (device, dtype).

    Allocating a fresh `torch.zeros(1, ...)` at every `launch()` would make the
    NPU lower one `aclnnInplaceZero_ZerosLikeAiCore` kernel each (3 per
    `run()`, 4.84-6.66 us/call).  These placeholders are only ever *bound* --
    every real access to the 8 optional tensors sits inside a `tla.const_expr`
    guard that is closed whenever the caller passed `None`, so both the read and
    the store path are folded away (IR: 0 `%argN` references for every
    guard-closed slot).  A placeholder is therefore never dereferenced, so
    reusing one zero buffer is bitwise equivalent.  It also keeps `id()` stable,
    so `_BINDING_CACHE` stops accumulating one entry per call.
    """
    key = (str(device), dtype)
    hit = _PLACEHOLDER_CACHE.get(key)
    if hit is None:
        hit = torch.zeros(1, dtype=dtype, device=device)
        _PLACEHOLDER_CACHE[key] = hit
    return hit


def _calculate_core_partition(total, heads, core_num):
    """Return head-chunk count, queries per group, and launch block count."""
    head_chunks = (heads + TILE_M - 1) // TILE_M
    core_groups = core_num // head_chunks
    if core_groups == 0:
        raise ValueError(
            f"Head tiles ({head_chunks}) exceed available cube cores ({core_num})"
        )
    queries_per_group = (total + core_groups - 1) // core_groups
    used_core_groups = (total + queries_per_group - 1) // queries_per_group
    return head_chunks, queries_per_group, used_core_groups * head_chunks


def _head_group_rows(heads, head_chunks):
    """M rows per head chunk, padded to 16 and capped at TILE_M."""
    per_chunk = (heads + head_chunks - 1) // head_chunks
    return min(TILE_M, ((per_chunk + 15) // 16) * 16)


_EXECUTOR_CACHE = {}


class _SparseFlashAttentionLauncher:
    """Hold JIT launch configuration for native-layout tensors."""

    def __init__(
        self,
        heads,
        sparse_count,
        total_queries,
        head_chunks,
        queries_per_group,
        dtype,
        block_dim,
        use_sinks,
        return_lse,
        layout_query,
        layout_kv,
        batch,
        sparse_mode,
        has_rope=True,
    ):
        self.heads = heads
        self.sparse_count = sparse_count
        self.total_queries = total_queries
        self.head_chunks = head_chunks
        self.queries_per_group = queries_per_group
        self.dtype = dtype
        self.block_dim = block_dim
        self.use_sinks = use_sinks
        self.return_lse = return_lse
        self.layout_query = layout_query
        self.layout_kv = layout_kv
        self.batch = batch
        self.sparse_mode = sparse_mode
        self.has_rope = has_rope

    # Keep this argument order aligned with the kernel and launcher call sites.
    def launch(
        self,
        output,
        query,
        key,
        indices,
        query_rope,
        key_rope,
        sinks,
        maxima,
        sums,
        workspace,
        table,
        cu_q,
        cu_kv,
        used_q,
        used_kv,
        scale,
        *,
        compile_only=False,
    ):
        """Bind the tla tensors, compile the kernel once, then execute it.

        tla 侧对应 `@tla.kernel` 单函数（类在 kernel 体内构造），block_dim 由
        `executor(..., block_num=...)` 给出。
        """
        device = query.device
        dtype = query.dtype
        # 宿主按 `query.dtype == torch.bfloat16` 选 bfloat16。
        dtype_bf16 = bool(dtype == torch.bfloat16)

        # rank-2 绑定后设备侧读不到原生 rank，形状常量在宿主侧取。
        if self.layout_query == "TND":
            q_seq = self.total_queries
        else:
            q_seq = int(query.shape[1])
        kv_pages = 1
        if self.layout_kv == "BSND":
            kv_seq = int(key.shape[1])
            kv_capacity = kv_seq
        elif self.layout_kv == "PA_BSND":
            kv_seq = int(key.shape[1])
            kv_pages = int(table.shape[1])
            kv_capacity = kv_pages * kv_seq
        else:
            kv_seq = int(key.shape[1])
            kv_capacity = int(key.shape[0])
        has_used_q = used_q is not None
        has_used_kv = used_kv is not None
        placeholder_16 = _placeholder(device, dtype)
        placeholder_32 = _placeholder(device, torch.float32)
        placeholder_i32 = _placeholder(device, torch.int32)

        t_o = _tla_tensor(output)
        t_q = _tla_tensor(query)
        t_k = _tla_tensor(key)
        t_idx = _tla_tensor(indices)
        t_qr = _tla_tensor(query_rope if self.has_rope else placeholder_16)
        t_kr = _tla_tensor(key_rope if self.has_rope else placeholder_16)
        t_sinks = _tla_tensor(sinks if sinks is not None else placeholder_32)
        t_max = _tla_tensor(maxima)
        t_sum = _tla_tensor(sums)
        t_ws = _tla_tensor(workspace)
        t_tab = _tla_tensor(table if table is not None else placeholder_i32)
        t_cuq = _tla_tensor(cu_q if cu_q is not None else placeholder_i32)
        t_cukv = _tla_tensor(cu_kv if cu_kv is not None else placeholder_i32)
        t_seq_q = _tla_tensor(used_q if used_q is not None else placeholder_i32)
        t_seq_kv = _tla_tensor(used_kv if used_kv is not None else placeholder_i32)
        scale_t = tla.Float32(scale)

        # M 行数按每个 head chunk 的宽度上取整到 16（封顶 64）；调度仍按 TILE_M=64。
        group_rows = _head_group_rows(self.heads, self.head_chunks)
        # 全部 constexpr 进缓存键：形状/布局不同的用例不能复用同一个 executor。
        key_cache = (
            str(dtype),
            str(device),
            group_rows,
            self.sparse_count,
            self.head_chunks,
            self.queries_per_group,
            self.total_queries,
            self.batch,
            self.has_rope,
            self.use_sinks,
            self.return_lse,
            self.layout_query,
            self.layout_kv,
            self.sparse_mode,
            q_seq,
            kv_seq,
            kv_pages,
            kv_capacity,
            self.heads,
            has_used_q,
            has_used_kv,
            dtype_bf16,
        )
        executor = _EXECUTOR_CACHE.get(key_cache)
        if executor is None:
            executor = tla.compile(
                sparse_flash_attention_kernel,
                t_o,
                t_q,
                t_k,
                t_idx,
                t_qr,
                t_kr,
                t_sinks,
                t_max,
                t_sum,
                t_ws,
                t_tab,
                t_cuq,
                t_cukv,
                t_seq_q,
                t_seq_kv,
                scale_t,
                group_rows,
                self.sparse_count,
                self.head_chunks,
                self.queries_per_group,
                self.total_queries,
                self.batch,
                self.has_rope,
                self.use_sinks,
                self.return_lse,
                self.layout_query == "TND",
                self.layout_kv == "PA_BSND",
                self.layout_kv == "TND",
                self.sparse_mode,
                q_seq,
                kv_seq,
                kv_pages,
                kv_capacity,
                self.heads,
                has_used_q,
                has_used_kv,
                dtype_bf16,
                options=NPU_ARCH,
            )
            _EXECUTOR_CACHE[key_cache] = executor

        if compile_only:
            return

        executor(
            t_o,
            t_q,
            t_k,
            t_idx,
            t_qr,
            t_kr,
            t_sinks,
            t_max,
            t_sum,
            t_ws,
            t_tab,
            t_cuq,
            t_cukv,
            t_seq_q,
            t_seq_kv,
            scale_t,
            block_num=self.block_dim,
        )


def _native_sparse_launch(
    query,
    key,
    query_rope,
    key_rope,
    sparse_indices,
    table,
    cu_q,
    cu_kv,
    used_q,
    used_kv,
    sinks,
    scale,
    return_lse,
    layout,
    layout_kv,
    sparse_mode,
    *,
    compile_only=False,
):
    """Allocate outputs and workspace, then launch the configured kernel."""
    has_rope = query_rope is not None
    batch = int(query.shape[0]) if layout == "BSND" else int(cu_q.numel()) - 1
    total = (
        int(query.shape[0]) * int(query.shape[1])
        if layout == "BSND"
        else int(query.shape[0])
    )
    heads = int(query.shape[-2])
    index_capacity = int(sparse_indices.shape[-1])
    indices = sparse_indices.view(total, index_capacity)
    core_num = torch.npu.get_device_properties(query.device).cube_core_num
    head_chunks, queries_per_group, block_dim = _calculate_core_partition(
        total, heads, core_num
    )
    # kernel M = 每个 head chunk 宽度 ceil16（封顶 64）；per-AIV half = group_rows/2。
    group_rows = _head_group_rows(heads, head_chunks)
    aiv_rows = group_rows // 2
    output = torch.empty(query.shape, dtype=query.dtype, device=query.device)

    # Reserve padded max/sum output buffers for all head tiles.
    # 核内按 (2*head_chunks, 32) 的宽行切分落盘，元素总数与
    # (total, head_chunks*64) 完全一致；宿主再还原成 (total, heads) 语义。
    stats_rows = total * head_chunks * 2
    max_out = torch.empty(
        (stats_rows, aiv_rows),
        dtype=torch.float32,
        device=query.device,
    )
    sum_out = torch.empty(
        (stats_rows, aiv_rows),
        dtype=torch.float32,
        device=query.device,
    )

    sink_view = None if sinks is None else sinks.view(heads, 1)
    # Each core reuses its three workspace slots across pipeline ticks.
    qk_dim = DIM_QK if has_rope else DIM_NOPE
    workspace = torch.empty(
        (
            block_dim * PIPELINE_SLOT_COUNT * TILE_S2,
            qk_dim,
        ),
        dtype=query.dtype,
        device=query.device,
    )
    launcher = _SparseFlashAttentionLauncher(
        heads,
        index_capacity,
        total,
        head_chunks,
        queries_per_group,
        torch.bfloat16 if query.dtype == torch.bfloat16 else torch.float16,
        block_dim,
        sinks is not None,
        return_lse,
        layout,
        layout_kv,
        batch,
        sparse_mode,
        has_rope,
    )
    launcher.launch(
        output,
        query,
        key,
        indices,
        query_rope,
        key_rope,
        sink_view,
        max_out,
        sum_out,
        workspace,
        table,
        cu_q,
        cu_kv,
        used_q,
        used_kv,
        scale,
        compile_only=compile_only,
    )

    if return_lse:
        max_out = max_out[:stats_rows].view(total, head_chunks * group_rows)[:, :heads]
        # sum_out 与 max_out 同尺寸（(stats_rows, AIV_ROW_CAPACITY) 的填充缓冲），
        # 早先漏了同样的裁剪 + view，BSND 分支会拿 16384 个元素去 reshape 成
        # (batch, seq, heads) 而报 "shape '[2, 4, 16]' is invalid"。
        sum_out = sum_out[:stats_rows].view(total, head_chunks * group_rows)[:, :heads]
        if layout == "BSND":
            # Restore [B, 1, S1, H] stats from the flattened query axis.
            max_out = max_out.view(batch, int(query.shape[1]), heads).unsqueeze(1)
            sum_out = sum_out.view(batch, int(query.shape[1]), heads).unsqueeze(1)
        else:
            max_out = max_out.unsqueeze(0)
            sum_out = sum_out.unsqueeze(0)
    else:
        max_out = torch.empty((0,), dtype=torch.float32, device=query.device)
        sum_out = torch.empty((0,), dtype=torch.float32, device=query.device)
    return output, max_out, sum_out


def sparse_flash_attention(
    query,
    key,
    value,
    sparse_indices,
    block_table=None,
    query_rope=None,
    key_rope=None,
    sinks=None,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    scale_value=1.0,
    sparse_block_size=1,
    layout_query="BSND",
    layout_kv="BSND",
    sparse_mode=0,
    pre_tokens=2**63 - 1,
    next_tokens=2**63 - 1,
    attention_mode=2,
    return_softmax_lse=False,
    *,
    compile_only=False,
):
    """Run sparse MLA attention for valid-prefix, minus-one-padded indices."""
    if attention_mode != 2 or sparse_block_size != 1:
        raise ValueError("SFA requires attention_mode=2 and sparse_block_size=1")
    if pre_tokens != 2**63 - 1 or next_tokens != 2**63 - 1:
        raise ValueError("pre_tokens and next_tokens must keep their default values")
    if sparse_mode not in (SPARSE_MODE_DENSE, SPARSE_MODE_RIGHT_DOWN_CAUSAL):
        raise ValueError("sparse_mode must be 0 or 3")
    if (layout_query, layout_kv) not in {
        ("BSND", "BSND"),
        ("BSND", "PA_BSND"),
        ("TND", "TND"),
        ("TND", "PA_BSND"),
    }:
        raise ValueError("unsupported query/KV layout combination")
    if layout_kv != "PA_BSND" and block_table is not None:
        raise ValueError("block_table is only valid for PA_BSND")
    if layout_kv == "PA_BSND" and return_softmax_lse:
        raise ValueError("PA_BSND does not support softmax max/sum outputs")
    if layout_query != "TND" and cu_seqlens_q is not None:
        raise ValueError("cu_seqlens_q is only valid for TND query")
    if layout_kv != "TND" and cu_seqlens_kv is not None:
        raise ValueError("cu_seqlens_kv is only valid for TND KV")
    if not isinstance(value, torch.Tensor) or value.shape != key.shape:
        raise ValueError("value must be a tensor with the same shape as key")
    if value.dtype != query.dtype or not value.is_contiguous():
        raise ValueError("value must be contiguous and share query dtype")
    if int(sparse_indices.shape[-1]) <= 0:
        raise ValueError("sparse_indices K must be positive")

    return _native_sparse_launch(
        query,
        key,
        query_rope,
        key_rope,
        sparse_indices,
        block_table,
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        sinks,
        scale_value,
        return_softmax_lse,
        layout_query,
        layout_kv,
        sparse_mode,
        compile_only=compile_only,
    )


def run(
    query,
    key,
    value,
    sparse_indices,
    block_table=None,
    query_rope=None,
    key_rope=None,
    sinks=None,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    scale_value=1.0,
    sparse_block_size=1,
    layout_query="BSND",
    layout_kv="BSND",
    sparse_mode=0,
    pre_tokens=2**63 - 1,
    next_tokens=2**63 - 1,
    attention_mode=2,
    return_softmax_lse=False,
    *,
    compile_only=False,
):
    """Harness entry point; same arguments as `sparse_flash_attention`."""
    return sparse_flash_attention(
        query,
        key,
        value,
        sparse_indices,
        block_table,
        query_rope,
        key_rope,
        sinks,
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        scale_value,
        sparse_block_size,
        layout_query,
        layout_kv,
        sparse_mode,
        pre_tokens,
        next_tokens,
        attention_mode,
        return_softmax_lse,
        compile_only=compile_only,
    )
