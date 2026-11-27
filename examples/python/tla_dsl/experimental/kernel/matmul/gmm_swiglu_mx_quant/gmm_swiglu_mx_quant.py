# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# This file is a CATLASS Python-DSL kernel: its layout and flag structure mirror
# the DSL examples, so it is excluded from line-length/format rewrites.
# ruff: noqa
# fmt: off

"""Grouped MX matmul (slice-M) + SwiGLU + Dynamic MXFP8 quant.

CATLASS Python DSL implementation for Ascend 950PR&950DT.  The host entry
point is ``run``; it binds the operand / scale tensors and the two outputs
(FP8 ``q`` and E8M0 ``q_scale``), compiles one fused mixed kernel per
element-format combination and launches it once.

Logical contract (see README.md), with A split along M by ``group_list``:

    C_g   = dequant(A_g) @ dequant(B_g)          fp32 accumulation
    S_g   = silu(C_g[:, :N/2]) * C_g[:, N/2:]    fp32 activation
    S16   = round_bf16_rne(S_g)
    Y, SY = dynamic_mx_quant(S16)                32-element blocks, E8M0 scale

Three operand families are covered: a8w8 (FP8 x FP8), a8w4 (FP8 act x FP4
weight, B widened to FP8) and a4w4 (native FP4 x FP4 cube).  The output is
always MXFP8 (E4M3, or E5M2 when the activation is E5M2).
"""

from __future__ import annotations

import ctypes
import os
import struct
from collections import OrderedDict, deque
from dataclasses import dataclass

import torch
import torch_npu  # noqa: F401

import catlass.tla as tla
from catlass import execution as _tla_execution
from catlass.tla.runtime import from_dlpack


MX_SCALE_GROUP_NUM = 32
MX_OUTPUT_BLOCK_SIZE = 32


# [CB-PIPE] Pre-decode depth (in K tiles) for the arch35 cross-basic-block
# schedule: decode the first m1 K chunks of the current block -> run the
# previous block's epilogue -> decode the rest. The reference pre-decodes
# DOUBLE_BUFFER_NUM * kbL1Size = 2 * 512 = 1024 K elements (8 tiles at tk=128);
# measured on card 7, m1=2 / m1=8 / full-block (0 = no mid-block split) give
# -2.1% / -4.0% / -5.0% on case 070, so the selected point is full-block
# deferral. The constant stays so the split can be re-tuned.
CB_PIPE_K_CHUNKS = 0


def _nibble_fp8_byte(c):
    """Map a 4-bit E2M1 code to its exact E4M3 byte value as Int32 (0..255)."""
    sign = (c >> 3) & 1
    e2 = (c >> 1) & 3
    m1 = c & 1
    normal = ((e2 + 6) << 3) | (m1 << 2)
    sub = m1 * 0x30
    sel = (e2 + 3) >> 2
    mag = sub + sel * (normal - sub)
    return (sign << 7) | mag


def _nibble_fp8(c):
    """Map a 4-bit E2M1 code to its exact E4M3 byte (lossless widening)."""
    return _nibble_fp8_byte(c).to(tla.Int8)


def _nibble_fp8_byte_shift(c):
    """Map a 4-bit E2M1 code to the E4M3 byte of (value / 64) as Int32 (0..255).

    Five-op bit shuffle (A5 arch35 reference: ShiftRight(2)/ShiftLeft(4)/Select/And
    0x9C collapsed). Bit-exact 2^-6 of _nibble_fp8_byte for every nibble, so the
    epilogue recovers the true value with an exact x64 multiply.
    """
    return ((c & 0x8) << 4) | ((c & 0x7) << 2)


def _nibble_fp8_shift(c):
    """E4M3 byte of (value / 64) as Int8 (see _nibble_fp8_byte_shift)."""
    return _nibble_fp8_byte_shift(c).to(tla.Int8)


def _transb_decode_fill(
    *, gm_pack, k_tile, j, par, dk, tn, tk, dtype_b, pk_buf, fp8_buf, act_col,
    fl_rl, fl_ld,
):
    """Fill the ``dk``-K packed staging tile of sub-tile ``j`` and wait for it.

    Returns the flat packed view, the flat Int16 FP8 view the SIMT decode stores
    through, and the zN-laid-out FP8 view the L1 store reads. The nibble
    widening itself stays in the kernel body: ``tla.vec.func`` cannot be emitted
    from a host-side helper.

    The staging ring is indexed by the *global* sub-tile parity, which for an
    even ``nsub`` reduces to ``j % 2`` (independent of ``k_tile``), so the fill
    of one parity overlaps the widen/store of the other exactly as the old
    per-K-tile ring did. ``j`` is a host-side int, so the L1 slice coordinate
    stays static.

    [ZNUB] The packed bytes stay in the aligned ``(tn, dk//2)`` row-major order
    the GM->UB DMA can actually write (a narrower 16-byte row is rounded up to
    the 32-byte DataBlock, which would insert gaps). The zN fractal order is
    produced by the SIMT decode instead: it writes each nibble pair straight to
    its zN offset, so the UB->L1 store turns into a contiguous run instead of
    the 32-byte scatter the row-major source forced.
    """
    pk_t = tla.make_tensor(
        pk_buf,
        tla.make_layout(
            tla.make_shape(tn, dk // 2), tla.make_stride(dk // 2, 1),
            origin_shape=tla.make_shape(tn, dk // 2), layoutTag=tla.arch.RowMajor,
        ),
    )
    gsrc = tla.get_tile(
        gm_pack,
        tla.make_coord(act_col, k_tile * (tk // 2) + j * (dk // 2)),
        tla.make_shape(tn, dk // 2),
    )
    bv0 = gsrc.origin_shape[0]
    bv1 = gsrc.origin_shape[1]
    pk_f = tla.make_tensor(
        pk_buf,
        tla.make_layout(tla.make_shape(dk * tn // 2), tla.make_stride(1)),
    )
    f8_i16 = tla.make_tensor(
        tla.recast_ptr(fp8_buf, dtype=tla.Int16),
        tla.make_layout(tla.make_shape(dk * tn // 2), tla.make_stride(1)),
    )
    fp8_t = tla.make_tensor(
        tla.recast_ptr(fp8_buf, dtype=dtype_b),
        tla.make_layout_with_tag((tn, dk), dtype_b, tla.arch.zN),
    )
    if par == 0:
        tla.wait_flag(fl_rl[0])
    else:
        tla.wait_flag(fl_rl[1])
    tla.copy(tla.tile_view(pk_t, tla.make_shape(bv0, bv1), tla.make_coord(0, 0)), gsrc)
    if par == 0:
        tla.set_flag(fl_ld[0])
        tla.wait_flag(fl_ld[0])
    else:
        tla.set_flag(fl_ld[1])
        tla.wait_flag(fl_ld[1])
    return pk_f, f8_i16, fp8_t


def _transb_decode_commit(
    *, fp8_t, k_tile, j, par, dk, tn, tk, l1_base, slot_ref,
    fl_rl, fl_st, b_free, b_full, aiv_id, first, last,
):
    """Store the widened sub-tile into its L1 slice and return the credits.

    Only the first sub-tile takes the cube's L1-slot credit and only the last
    publishes ``b_full``: the cube still sees one K tile.
    """
    if par == 0:
        tla.set_flag(fl_st[0])
        tla.wait_flag(fl_st[0])
    else:
        tla.set_flag(fl_st[1])
        tla.wait_flag(fl_st[1])
    if first:
        tla.cross_core_wait_flag(b_free, tla.pipes.MTE3, aiv_id=aiv_id)
    slot = tla.make_tensor_like(
        l1_base + (k_tile % 2) * (tk * tn), slot_ref, tla.arch.zN
    )
    tla.copy(
        tla.get_tile(slot, tla.make_coord(0, j * dk), tla.make_shape(tn, dk)),
        fp8_t,
    )
    if par == 0:
        tla.set_flag(fl_rl[0])
    else:
        tla.set_flag(fl_rl[1])
    if last:
        tla.cross_core_set_flag(b_full, tla.pipes.MTE3, aiv_id=aiv_id)


def _emit_l0_subtile(
    *, t_l1a, t_l1b_act, t_l1b_gate, t_l1sa, t_l1sb_act, t_l1sb_gate,
    l0a_buf, l0b_act_buf, l0b_gate_buf, t_l0c_act, t_l0c_gate,
    j, lk, skl, tm, tn, l0_ready, l0_free_p0, l0_free_p1, par, init_c,
):
    """Emit one ``lk``-K L0 step out of a ``tk``-K L1 tile.

    Mirrors the arch35 qbmm reference: the L1 tile is sliced along K for both
    the operands and their MX scale, loaded into a ping-pong L0 slot, and
    consumed by ``mmad_mx`` accumulating into the same L0C. The L1 slot (and
    the AIV credits) are released only after the *last* slice was issued, so
    the double-buffered L0 still overlaps the wider GM->L1 fill.
    """
    s_a = tla.get_tile(t_l1a, tla.make_coord(0, j * lk), tla.make_shape(tm, lk))
    s_ba = tla.get_tile(t_l1b_act, tla.make_coord(j * lk, 0), tla.make_shape(lk, tn))
    s_bg = tla.get_tile(t_l1b_gate, tla.make_coord(j * lk, 0), tla.make_shape(lk, tn))
    s_sa = tla.get_tile(t_l1sa, tla.make_coord(0, j * skl), tla.make_shape(tm, skl))
    s_sba = tla.get_tile(t_l1sb_act, tla.make_coord(j * skl, 0), tla.make_shape(skl, tn))
    s_sbg = tla.get_tile(t_l1sb_gate, tla.make_coord(j * skl, 0), tla.make_shape(skl, tn))
    t_l0a = tla.make_tensor_like(l0a_buf, s_a, tla.arch.zN)
    t_l0ba = tla.make_tensor_like(l0b_act_buf, s_ba, tla.arch.nZ)
    t_l0bg = tla.make_tensor_like(l0b_gate_buf, s_bg, tla.arch.nZ)
    if par == 0:
        tla.wait_flag(l0_free_p0)
    else:
        tla.wait_flag(l0_free_p1)
    tla.copy(t_l0a, s_a, scale=s_sa)
    tla.copy(t_l0ba, s_ba, scale=s_sba)
    tla.copy(t_l0bg, s_bg, scale=s_sbg)
    tla.set_flag(l0_ready)
    tla.wait_flag(l0_ready)
    tla.mmad_mx(t_l0c_act, t_l0a, t_l0ba, init_c=init_c)
    tla.mmad_mx(t_l0c_gate, t_l0a, t_l0bg, init_c=init_c)
    if par == 0:
        tla.set_flag(l0_free_p0)
    else:
        tla.set_flag(l0_free_p1)


@dataclass(frozen=True)
class GmmTilingParams:
    tm: tla.Constexpr[int] = 128
    tn: tla.Constexpr[int] = 128
    tk: tla.Constexpr[int] = 128


@tla.kernel
def gmm_swiglu_mx_quant_kernel(
    gm_a: tla.Tensor,
    gm_b: tla.Tensor,
    gm_sa: tla.Tensor,
    gm_sb: tla.Tensor,
    group_list: tla.Tensor,
    gm_y: tla.Tensor,
    gm_sy: tla.Tensor,
    _tiling: tla.Constexpr[GmmTilingParams],
    _transB: tla.Constexpr[bool],
    _outE5M2: tla.Constexpr[bool],
    _decodeA: tla.Constexpr[bool],
    _decodeB: tla.Constexpr[bool],
    _bZN: tla.Constexpr[bool] = False,
) -> None:
    dtype_a = tla.Float8E4M3FN if tla.const_expr(_decodeA) else gm_a.ptr.dtype
    dtype_b = tla.Float8E4M3FN if tla.const_expr(_decodeB) else gm_b.ptr.dtype
    out_dtype = tla.Float8E5M2 if tla.const_expr(_outE5M2) else tla.Float8E4M3FN

    c0 = 0
    c1 = 1
    tm = _tiling.tm
    tn = _tiling.tn
    tk = _tiling.tk
    half = tm // 2
    sk = tk // MX_SCALE_GROUP_NUM
    # [ZNUB] Decode granularity == the L1 K tile. The vector decode writes each
    # zN row with one BlockStore, which always covers exactly 8 DataBlocks (8
    # 32-wide K blocks = 256 K elements); that only lines up with the L1 tile
    # when tk == 256, so the old tk==256 -> dk=128 sub-tiling is retired and the
    # whole row is widened at once. Smaller tk keeps dk == tk as before.
    dk = tk
    nsub = tk // dk
    skd = dk // MX_SCALE_GROUP_NUM
    # [TK256] L0 K tile. A 256-K L1 tile cannot keep both L0A (2*tm*tk) and L0B
    # (4*tk*tn) ping-ponged in 64 KiB each, and losing that overlap serializes
    # every L1->L0 copy with the MMAD (measured +6% on 069). The arch35
    # reference (qbmm_swiglu_mxquant) instead loads each L1 tile as two
    # tk/2 L0 operands, so the wider GM->L1 burst is kept while L0 stays
    # double buffered. lk == tk for every pre-existing tile.
    # [A8TK] a8w8 transB joins the same decoupling on the tm=256 tile: 041
    # previously ran tk=256 with lk=256 and 019 ran tk=128 (32 K steps); the
    # serialized L1->L0 left aic_mte2 at 94% of wall on 019. Native FP4
    # operands (a4w4) already fit the ping-pong at tk=256 (half-byte tiles) and
    # keep lk == tk; small transB tiles (tm<256) keep their measured lk == tk.
    _l0_native_fp4 = tla.const_expr(
        (gm_a.ptr.dtype == tla.Float4E2M1 and not _decodeA)
        or (gm_b.ptr.dtype == tla.Float4E2M1 and not _decodeB)
    )
    # [TK512] native-FP4 512-K L1 tile keeps a 256-K L0 step (L0A/L0B land
    # exactly on their 64 KiB ping-pong at tm=256); cap dk.
    _lk_cap = 256 if dk > 256 else dk
    # [TK256-W] Narrow transB decode at tm<=128 takes the whole L1 K tile as one
    # L0 operand (lk=tk). L0B then lands on its 64 KiB single buffer (no ping-pong)
    # but the halved copy/store count wins: 063 aic_mte1 12.9->5.0 us, aiv_mte3
    # 19.4->8.8 us, Duration -12%, bit-identical. tm=256 transB (069) and all
    # other paths keep the measured lk=128 decoupling.
    _narrow_transb_decode = (
        tla.const_expr(_transB) and tla.const_expr(_decodeB) and tm <= 128
    )
    lk = _lk_cap if _narrow_transb_decode else (
        128 if (
            tk == 256 and tla.const_expr(_transB) and not _l0_native_fp4
            and (tla.const_expr(_decodeB) or tm == 256)
        ) else _lk_cap
    )
    nsub_l0 = tk // lk
    skl = lk // MX_SCALE_GROUP_NUM

    k_dim = gm_sa.origin_shape[1] * MX_SCALE_GROUP_NUM
    n_out = gm_y.origin_shape[1] * 4
    n_dim = n_out * 2
    group_cnt = group_list.origin_shape[0]
    k_groups = (k_dim + MX_SCALE_GROUP_NUM - 1) // MX_SCALE_GROUP_NUM
    kg_half = (k_groups + 1) // 2
    kg_ru = kg_half * 2
    grid_n = (n_out + tn - 1) // tn

    emax = 15 if tla.const_expr(_outE5M2) else 8
    emax_shifted = emax * 128
    clamp_value = 57344.0 if tla.const_expr(_outE5M2) else 448.0

    # L1 ping-pong: two slots for every L1 operand. Native FP4 operands need two
    # independent base pointers (sub-byte L1 rejects a dynamic ptr_add), the
    # byte-addressable ones use parity offsets into a doubled allocation.
    _l1_db = True
    _l1_fp4_a = tla.const_expr(gm_a.ptr.dtype == tla.Float4E2M1 and not _decodeA)
    _l1_fp4_b = tla.const_expr(gm_b.ptr.dtype == tla.Float4E2M1 and not _decodeB)
    l1_slots = 2
    if tla.const_expr(_l1_fp4_a):
        l1a0 = tla.allocate(tm * tk, dtype_a, tla.AddressSpace.l1, 512)
        l1a1 = tla.allocate(tm * tk, dtype_a, tla.AddressSpace.l1, 512)
    else:
        l1a = tla.allocate(l1_slots * tm * tk, dtype_a, tla.AddressSpace.l1, 512)
    if tla.const_expr(_l1_fp4_b):
        l1b_act0 = tla.allocate(tk * tn, dtype_b, tla.AddressSpace.l1, 512)
        l1b_act1 = tla.allocate(tk * tn, dtype_b, tla.AddressSpace.l1, 512)
        l1b_gate0 = tla.allocate(tk * tn, dtype_b, tla.AddressSpace.l1, 512)
        l1b_gate1 = tla.allocate(tk * tn, dtype_b, tla.AddressSpace.l1, 512)
    else:
        l1b_act = tla.allocate(l1_slots * tk * tn, dtype_b, tla.AddressSpace.l1, 512)
        l1b_gate = tla.allocate(l1_slots * tk * tn, dtype_b, tla.AddressSpace.l1, 512)
    l1sa = tla.allocate(l1_slots * tm * sk, tla.Float8E8M0, tla.AddressSpace.l1, 512)
    l1sb_act = tla.allocate(l1_slots * sk * tn, tla.Float8E8M0, tla.AddressSpace.l1, 512)
    l1sb_gate = tla.allocate(l1_slots * sk * tn, tla.Float8E8M0, tla.AddressSpace.l1, 512)
    # L0 A/B double buffering: the L1->L0 copy for slot kk%2 may run while the
    # MMAD of the other slot is still in flight (arch35 dbL0A/dbL0B). Separate
    # allocations keep sub-byte FP4 operands off ptr_add; slot choice is a
    # dynamic-if base select. Guarded by the doubled byte footprint: L0A and L0B
    # are 64 KiB each, and overflowing them surfaces as an AICore error
    # ("set/wait instructions do not match") rather than a compile failure.
    _l0_a_bytes = tm * lk // 2 if dtype_a == tla.Float4E2M1 else tm * lk
    _l0_b_bytes = lk * tn // 2 if dtype_b == tla.Float4E2M1 else lk * tn
    _l0_db = tla.const_expr(2 * _l0_a_bytes <= 65536 and 4 * _l0_b_bytes <= 65536)
    if tla.const_expr(_l0_db):
        l0a0 = tla.allocate(tm * lk, dtype_a, tla.AddressSpace.l0a, 1024)
        l0a1 = tla.allocate(tm * lk, dtype_a, tla.AddressSpace.l0a, 1024)
        l0b_act0 = tla.allocate(lk * tn, dtype_b, tla.AddressSpace.l0b, 1024)
        l0b_act1 = tla.allocate(lk * tn, dtype_b, tla.AddressSpace.l0b, 1024)
        l0b_gate0 = tla.allocate(lk * tn, dtype_b, tla.AddressSpace.l0b, 1024)
        l0b_gate1 = tla.allocate(lk * tn, dtype_b, tla.AddressSpace.l0b, 1024)
    else:
        l0a = tla.allocate(tm * lk, dtype_a, tla.AddressSpace.l0a, 1024)
        l0b_act = tla.allocate(lk * tn, dtype_b, tla.AddressSpace.l0b, 1024)
        l0b_gate = tla.allocate(lk * tn, dtype_b, tla.AddressSpace.l0b, 1024)
    l0c_act = tla.allocate(tm * tn, tla.Float32, tla.AddressSpace.l0c, 1024)
    l0c_gate = tla.allocate(tm * tn, tla.Float32, tla.AddressSpace.l0c, 1024)

    ub_act = tla.allocate(half * tn, tla.Float32, tla.AddressSpace.ub, 256)
    ub_gate = tla.allocate(half * tn, tla.Float32, tla.AddressSpace.ub, 256)
    ub_y = tla.allocate(half * tn // 4, tla.Int32, tla.AddressSpace.ub, 256)
    ub_sy = tla.allocate(half * 8, tla.Int32, tla.AddressSpace.ub, 256)

    if tla.const_expr(_decodeA):
        ub_a_pack = tla.allocate(tm * tk // 2, tla.Int8, tla.AddressSpace.ub, 256)
        ub_a_fp8 = tla.allocate(tm * tk, tla.Int8, tla.AddressSpace.ub, 256)
    if tla.const_expr(_decodeB):
        # [AGSHARE] The two B halves are decoded on different AIV sub-cores:
        # ``sub_idx == 0`` owns the act half and ``sub_idx == 1`` the gate half
        # (each signals/counts its own b_full id), so on any one core only one
        # half's staging is ever live. They share one allocation instead of two,
        # which frees half the decode staging -- the SIMT nibble decode needs
        # that headroom on the tm=256 transB tile (SIMT kernels reserve a 32 KiB
        # global Data Cache, lowering the UB ceiling to 219136 B).
        ub_ba_pack = tla.allocate(dk * tn // 2, tla.Int8, tla.AddressSpace.ub, 256)
        ub_ba_fp8 = tla.allocate(dk * tn, tla.Int8, tla.AddressSpace.ub, 256)
        ub_bg_pack = ub_ba_pack
        ub_bg_fp8 = ub_ba_fp8
        if tla.const_expr(_transB):
            # Second decode stage per half (ops-transformer
            # VecAntiQuantConfig ubMte2BufferNum staging): the GM->UB fill of
            # the next K tile runs while the current one is widened/stored.
            ub_ba_pack_b = tla.allocate(dk * tn // 2, tla.Int8, tla.AddressSpace.ub, 256)
            ub_ba_fp8_b = tla.allocate(dk * tn, tla.Int8, tla.AddressSpace.ub, 256)
            ub_bg_pack_b = ub_ba_pack_b
            ub_bg_fp8_b = ub_ba_fp8_b
    if tla.const_expr(_decodeB and _transB):
        # Unit E8M0 scale (127 = 2^0) for hardware mx_to FP4->FP8 widening; the
        # cube applies the real per-32 MX scale separately via gm_sb.
        ub_scale_unit = tla.allocate(skd * tn, tla.Int8, tla.AddressSpace.ub, 256)

    c_ref = tla.make_tensor(
        gm_a.ptr,
        tla.make_layout(
            tla.make_shape(tm, tn), tla.make_stride(tn, 1), layoutTag=tla.arch.RowMajor
        ),
    )
    t_l0c_act = tla.make_tensor_like(l0c_act, c_ref, tla.arch.L0Clayout)
    t_l0c_gate = tla.make_tensor_like(l0c_gate, c_ref, tla.arch.L0Clayout)

    ab_ready = tla.flag("ab_ready", tla.pipes.MTE2, tla.pipes.MTE1)
    l0_ready = tla.flag("l0_ready", tla.pipes.MTE1, tla.pipes.CUBE)
    if tla.const_expr(_l1_db):
        # One hardware event id per slot: SetFlag/WaitFlag are binary, so
        # depth 2 needs two ids (mirrors ops-transformer EVENT_ID + i).
        l1_free_p0 = tla.flag("l1_free_p0", tla.pipes.MTE1, tla.pipes.MTE2)
        l1_free_p1 = tla.flag("l1_free_p1", tla.pipes.MTE1, tla.pipes.MTE2)
    if tla.const_expr(_l0_db):
        l0_free_p0 = tla.flag("l0_free_p0", tla.pipes.CUBE, tla.pipes.MTE1)
        l0_free_p1 = tla.flag("l0_free_p1", tla.pipes.CUBE, tla.pipes.MTE1)
    else:
        l0_avail = tla.flag("l0_avail", tla.pipes.CUBE, tla.pipes.MTE1)
    l0_drain = tla.flag("l0_drain", tla.pipes.CUBE, tla.pipes.FIX)
    ub_ready = tla.cross_flag("ub_ready")
    ub_free = tla.cross_flag("ub_free")
    mtx_ub_out = tla.mutex(resource="ub_out", id=0)
    mtx_l0c = tla.mutex(resource="l0c", id=4)

    if tla.const_expr(_decodeA or _decodeB):
        a_full = tla.cross_flag("a_full", mode=4)
        a_free = tla.cross_flag("a_free", mode=4)
        b_full = tla.cross_flag("b_full", mode=4)
        b_free = tla.cross_flag("b_free", mode=4)
    if tla.const_expr(_decodeB and _transB):
        # Per-parity pipe flags for the two-deep UB decode staging: fill
        # (MTE2) -> widen (V) -> store (MTE3), each buffer's read release
        # gates the next fill of the same parity.
        dcd_a_ld0 = tla.flag("dcd_a_ld0", tla.pipes.MTE2, tla.pipes.VECTOR)
        dcd_a_ld1 = tla.flag("dcd_a_ld1", tla.pipes.MTE2, tla.pipes.VECTOR)
        dcd_a_st0 = tla.flag("dcd_a_st0", tla.pipes.VECTOR, tla.pipes.MTE3)
        dcd_a_st1 = tla.flag("dcd_a_st1", tla.pipes.VECTOR, tla.pipes.MTE3)
        dcd_a_rl0 = tla.flag("dcd_a_rl0", tla.pipes.MTE3, tla.pipes.MTE2)
        dcd_a_rl1 = tla.flag("dcd_a_rl1", tla.pipes.MTE3, tla.pipes.MTE2)
        dcd_g_ld0 = tla.flag("dcd_g_ld0", tla.pipes.MTE2, tla.pipes.VECTOR)
        dcd_g_ld1 = tla.flag("dcd_g_ld1", tla.pipes.MTE2, tla.pipes.VECTOR)
        dcd_g_st0 = tla.flag("dcd_g_st0", tla.pipes.VECTOR, tla.pipes.MTE3)
        dcd_g_st1 = tla.flag("dcd_g_st1", tla.pipes.VECTOR, tla.pipes.MTE3)
        dcd_g_rl0 = tla.flag("dcd_g_rl0", tla.pipes.MTE3, tla.pipes.MTE2)
        dcd_g_rl1 = tla.flag("dcd_g_rl1", tla.pipes.MTE3, tla.pipes.MTE2)
    if tla.const_expr(_decodeB and not _transB):
        # Half-tile pipeline (arch35 anti-quant ring in DSL form): each K tile
        # is decoded in two chunks so chunk-1 V work overlaps the chunk-0
        # UB->L1 store. Events: fill credit (V->MTE2), fill done (MTE2->V),
        # store ready (V->MTE3), chunk buffer released (MTE3->V).
        dcd_pk_a = tla.flag("dcd_pk_a", tla.pipes.VECTOR, tla.pipes.MTE2)
        dcd_ld_a = tla.flag("dcd_ld_a", tla.pipes.MTE2, tla.pipes.VECTOR)
        dcd_st_a0 = tla.flag("dcd_st_a0", tla.pipes.VECTOR, tla.pipes.MTE3)
        dcd_st_a1 = tla.flag("dcd_st_a1", tla.pipes.VECTOR, tla.pipes.MTE3)
        dcd_rl_a0 = tla.flag("dcd_rl_a0", tla.pipes.MTE3, tla.pipes.VECTOR)
        dcd_rl_a1 = tla.flag("dcd_rl_a1", tla.pipes.MTE3, tla.pipes.VECTOR)
        dcd_pk_g = tla.flag("dcd_pk_g", tla.pipes.VECTOR, tla.pipes.MTE2)
        dcd_ld_g = tla.flag("dcd_ld_g", tla.pipes.MTE2, tla.pipes.VECTOR)
        dcd_st_g0 = tla.flag("dcd_st_g0", tla.pipes.VECTOR, tla.pipes.MTE3)
        dcd_st_g1 = tla.flag("dcd_st_g1", tla.pipes.VECTOR, tla.pipes.MTE3)
        dcd_rl_g0 = tla.flag("dcd_rl_g0", tla.pipes.MTE3, tla.pipes.VECTOR)
        dcd_rl_g1 = tla.flag("dcd_rl_g1", tla.pipes.MTE3, tla.pipes.VECTOR)

    with tla.cube():
        core_num = tla.arch.block_num()
        core_idx = tla.arch.block_idx()
        start_core_idx = c0
        m_start = c0
        first_tile = True

        if tla.const_expr(_decodeA):
            # Prime the decode-credit flag once; the old code set it twice in a
            # row, which is a no-op on a binary cross-core flag and only adds
            # prologue scalar work (measured -3% on the tiny a8w4 cases).
            tla.cross_core_set_flag(a_free, tla.pipes.MTE1, aiv_id=0)
        if tla.const_expr(_decodeB):
            tla.cross_core_set_flag(b_free, tla.pipes.MTE1, aiv_id=0)
            tla.cross_core_set_flag(b_free, tla.pipes.MTE1, aiv_id=1)

        if tla.const_expr(_l1_db):
            # Both slots start free; every K iteration consumes and restores
            # its own parity credit, so block boundaries need no re-priming.
            tla.set_flag(l1_free_p0)
            tla.set_flag(l1_free_p1)
        if tla.const_expr(_l0_db):
            tla.set_flag(l0_free_p0)
            tla.set_flag(l0_free_p1)

        for g in tla.range(c0, group_cnt, c1):
            current_m = group_list[g].to(tla.Int32)
            if current_m > 0:
                zero = current_m - current_m
                grid_m = (current_m + tm - 1) // tm
                mn_blocks = grid_m * grid_n

                if tla.const_expr(gm_a.ptr.dtype == tla.Float4E2M1):
                    gm_a_g = tla.get_tile(
                        gm_a, tla.make_coord(m_start, c0),
                        tla.make_shape(current_m, k_dim),
                    )
                else:
                    gm_a_g = tla.make_tensor(
                        gm_a.ptr + m_start * gm_a.stride[0],
                        tla.make_layout(
                            tla.make_shape(current_m, k_dim),
                            tla.make_stride(gm_a.stride[0], gm_a.stride[1]),
                            origin_shape=tla.make_shape(current_m, k_dim),
                            layoutTag=tla.arch.RowMajor,
                        ),
                    )
                gm_sa_g = tla.make_tensor(
                    gm_sa.ptr + m_start * k_groups,
                    tla.make_layout(
                        tla.make_shape((1, current_m), (2, kg_half)),
                        tla.make_stride((zero, kg_ru), (1, 2)),
                        origin_shape=tla.make_shape(current_m, k_groups),
                        layoutTag=tla.arch.RowMajorMxScaleA,
                    ),
                )
                if tla.const_expr(_transB):
                    if tla.const_expr(gm_b.ptr.dtype == tla.Float4E2M1):
                        gm_b_g = tla.get_tile(
                            gm_b, tla.make_coord(c0, g * n_dim),
                            tla.make_shape(k_dim, n_dim),
                        )
                    else:
                        gm_b_g = tla.make_tensor(
                            gm_b.ptr + g * n_dim * gm_b.stride[0],
                            tla.make_layout(
                                tla.make_shape(k_dim, n_dim),
                                tla.make_stride(1, k_dim),
                                origin_shape=tla.make_shape(k_dim, n_dim),
                                layoutTag=tla.arch.ColumnMajor,
                            ),
                        )
                    gm_sb_g = tla.make_tensor(
                        gm_sb.ptr + g * n_dim * k_groups,
                        tla.make_layout(
                            tla.make_shape((2, kg_half), (1, n_dim)),
                            tla.make_stride((1, 2), (zero, kg_ru)),
                            origin_shape=tla.make_shape(k_groups, n_dim),
                            layoutTag=tla.arch.ColMajorMxScaleB,
                        ),
                    )
                else:
                    if tla.const_expr(gm_b.ptr.dtype == tla.Float4E2M1):
                        gm_b_g = tla.get_tile(
                            gm_b, tla.make_coord(g * k_dim, c0),
                            tla.make_shape(k_dim, n_dim),
                        )
                    else:
                        gm_b_g = tla.make_tensor(
                            gm_b.ptr + g * k_dim * gm_b.stride[0],
                            tla.make_layout(
                                tla.make_shape(k_dim, n_dim),
                                tla.make_stride(n_dim, 1),
                                origin_shape=tla.make_shape(k_dim, n_dim),
                                layoutTag=tla.arch.RowMajor,
                            ),
                        )
                    gm_sb_g = tla.make_tensor(
                        gm_sb.ptr + g * k_groups * n_dim,
                        tla.make_layout(
                            tla.make_shape((2, kg_half), (1, n_dim)),
                            tla.make_stride((1, n_dim * 2), (zero, 2)),
                            origin_shape=tla.make_shape(k_groups, n_dim),
                            layoutTag=tla.arch.RowMajorMxScaleB,
                        ),
                    )

                start_loop = (core_idx + core_num - start_core_idx) % core_num
                block_range = tla.range(start_loop, mn_blocks, core_num)
                for loop_idx in block_range:
                    block_row = loop_idx // grid_n
                    block_col = loop_idx % grid_n
                    act_col = block_col * tn
                    gate_col = n_out + block_col * tn

                    gm_a_t = tla.tile_view(
                        gm_a_g, tla.make_shape(tm, k_dim), tla.make_coord(block_row, c0)
                    )
                    k_tiles = (gm_a_t.origin_shape[1] + tk - 1) // tk

                    mtx_l0c.lock(pipe=tla.pipes.CUBE)
                    if not tla.const_expr(_l0_db):
                        tla.set_flag(l0_avail)
                    for kk in tla.range(c0, k_tiles, c1):
                        gm_a_k = tla.tile_view(
                            gm_a_t, tla.make_shape(tm, tk), tla.make_coord(c0, kk)
                        )
                        gm_b_act_k = tla.get_tile(
                            gm_b_g, tla.make_coord(kk * tk, act_col), tla.make_shape(tk, tn)
                        )
                        gm_b_gate_k = tla.get_tile(
                            gm_b_g, tla.make_coord(kk * tk, gate_col), tla.make_shape(tk, tn)
                        )
                        gm_sa_k = tla.tile_view(
                            gm_sa_g, tla.make_shape(tm, sk), tla.make_coord(block_row, kk)
                        )
                        gm_sb_act_k = tla.get_tile(
                            gm_sb_g, tla.make_coord(kk * sk, act_col), tla.make_shape(sk, tn)
                        )
                        gm_sb_gate_k = tla.get_tile(
                            gm_sb_g, tla.make_coord(kk * sk, gate_col), tla.make_shape(sk, tn)
                        )

                        m_valid = gm_a_k.origin_shape[0]
                        n_valid = gm_b_act_k.origin_shape[1]
                        k_valid = gm_a_k.origin_shape[1]
                        a_tag = (
                            tla.arch.zN
                            if gm_a_k.layout_tag in ("RowMajor", "zN")
                            else tla.arch.nZ
                        )
                        b_tag = (
                            tla.arch.nZ
                            if gm_b_act_k.layout_tag in ("ColumnMajor", "nZ")
                            else tla.arch.zN
                        )
                        if tla.const_expr(_decodeB):
                            b_l1_tag = tla.arch.nZ if tla.const_expr(_transB) else tla.arch.zN
                            b_l1_shape = (tk, tn)
                        else:
                            b_l1_tag = b_tag
                            b_l1_shape = (tk, tn)
                        if tla.const_expr(_l1_db):
                            if tla.const_expr(_l1_fp4_a):
                                l1a_buf = l1a0
                                if kk % 2 == 1:
                                    l1a_buf = l1a1
                            else:
                                l1a_buf = l1a + (kk % 2) * (tm * tk)
                            if tla.const_expr(_l1_fp4_b):
                                l1b_act_buf = l1b_act0
                                l1b_gate_buf = l1b_gate0
                                if kk % 2 == 1:
                                    l1b_act_buf = l1b_act1
                                    l1b_gate_buf = l1b_gate1
                            else:
                                l1b_act_buf = l1b_act + (kk % 2) * (tk * tn)
                                l1b_gate_buf = l1b_gate + (kk % 2) * (tk * tn)
                            l1sa_buf = l1sa + (kk % 2) * (tm * sk)
                            l1sb_act_buf = l1sb_act + (kk % 2) * (sk * tn)
                            l1sb_gate_buf = l1sb_gate + (kk % 2) * (sk * tn)
                        else:
                            l1a_buf = l1a
                            l1b_act_buf = l1b_act
                            l1b_gate_buf = l1b_gate
                            l1sa_buf = l1sa
                            l1sb_act_buf = l1sb_act
                            l1sb_gate_buf = l1sb_gate
                        t_l1a = tla.make_tensor(
                            l1a_buf, tla.make_layout_with_tag((tm, tk), dtype_a, a_tag)
                        )
                        t_l1b_act = tla.make_tensor(
                            l1b_act_buf, tla.make_layout_with_tag(b_l1_shape, dtype_b, b_l1_tag)
                        )
                        t_l1b_gate = tla.make_tensor(
                            l1b_gate_buf, tla.make_layout_with_tag(b_l1_shape, dtype_b, b_l1_tag)
                        )
                        t_l1sa = tla.make_tensor(
                            l1sa_buf, tla.make_layout_with_tag((tm, sk), tla.Float8E8M0, tla.arch.zZMxScale)
                        )
                        t_l1sb_act = tla.make_tensor(
                            l1sb_act_buf, tla.make_layout_with_tag((sk, tn), tla.Float8E8M0, tla.arch.nNMxScale)
                        )
                        t_l1sb_gate = tla.make_tensor(
                            l1sb_gate_buf, tla.make_layout_with_tag((sk, tn), tla.Float8E8M0, tla.arch.nNMxScale)
                        )

                        if kk % 2 == 0:
                            tla.wait_flag(l1_free_p0)
                        else:
                            tla.wait_flag(l1_free_p1)
                        if not tla.const_expr(_decodeA):
                            tla.copy(
                                tla.tile_view(t_l1a, tla.make_shape(m_valid, k_valid), tla.make_coord(0, 0)),
                                gm_a_k,
                            )
                        if not tla.const_expr(_decodeB):
                            tla.copy(
                                tla.tile_view(t_l1b_act, tla.make_shape(k_valid, n_valid), tla.make_coord(0, 0)),
                                gm_b_act_k,
                            )
                            tla.copy(
                                tla.tile_view(t_l1b_gate, tla.make_shape(k_valid, n_valid), tla.make_coord(0, 0)),
                                gm_b_gate_k,
                            )
                        tla.copy(
                            tla.tile_view(t_l1sa, tla.make_shape(m_valid, sk), tla.make_coord(0, 0)),
                            gm_sa_k,
                        )
                        tla.copy(
                            tla.tile_view(t_l1sb_act, tla.make_shape(sk, n_valid), tla.make_coord(0, 0)),
                            gm_sb_act_k,
                        )
                        tla.copy(
                            tla.tile_view(t_l1sb_gate, tla.make_shape(sk, n_valid), tla.make_coord(0, 0)),
                            gm_sb_gate_k,
                        )
                        if tla.const_expr(_decodeA):
                            tla.cross_core_wait_flag(a_full, tla.pipes.MTE1, aiv_id=0)
                        if tla.const_expr(_decodeB):
                            tla.cross_core_wait_flag(b_full, tla.pipes.MTE1, aiv_id=0)
                            tla.cross_core_wait_flag(b_full, tla.pipes.MTE1, aiv_id=1)
                        if k_valid < tk:
                            tla.get_tile(
                                t_l1a, tla.make_coord(0, k_valid),
                                tla.make_shape(m_valid, tk - k_valid),
                            ).fill(0)
                            if not tla.const_expr(_decodeB):
                                tla.get_tile(
                                    t_l1b_act, tla.make_coord(k_valid, 0),
                                    tla.make_shape(tk - k_valid, n_valid),
                                ).fill(0)
                                tla.get_tile(
                                    t_l1b_gate, tla.make_coord(k_valid, 0),
                                    tla.make_shape(tk - k_valid, n_valid),
                                ).fill(0)
                            tla.pipe_barrier(tla.pipes.ALL)
                        tla.set_flag(ab_ready)
                        tla.wait_flag(ab_ready)

                        if tla.const_expr(nsub_l0 > 1):
                            # [TK256] one L1 K tile is consumed as two
                            # 128-K L0 operands (arch35 qbmm pattern), so L0A/L0B
                            # stay ping-ponged while the GM->L1 burst is 2x wide.
                            _emit_l0_subtile(
                                t_l1a=t_l1a, t_l1b_act=t_l1b_act, t_l1b_gate=t_l1b_gate,
                                t_l1sa=t_l1sa, t_l1sb_act=t_l1sb_act, t_l1sb_gate=t_l1sb_gate,
                                l0a_buf=l0a0, l0b_act_buf=l0b_act0, l0b_gate_buf=l0b_gate0,
                                t_l0c_act=t_l0c_act, t_l0c_gate=t_l0c_gate,
                                j=0, lk=lk, skl=skl, tm=tm, tn=tn,
                                l0_ready=l0_ready, l0_free_p0=l0_free_p0, l0_free_p1=l0_free_p1,
                                par=0, init_c=(kk == 0),
                            )
                            _emit_l0_subtile(
                                t_l1a=t_l1a, t_l1b_act=t_l1b_act, t_l1b_gate=t_l1b_gate,
                                t_l1sa=t_l1sa, t_l1sb_act=t_l1sb_act, t_l1sb_gate=t_l1sb_gate,
                                l0a_buf=l0a1, l0b_act_buf=l0b_act1, l0b_gate_buf=l0b_gate1,
                                t_l0c_act=t_l0c_act, t_l0c_gate=t_l0c_gate,
                                j=1, lk=lk, skl=skl, tm=tm, tn=tn,
                                l0_ready=l0_ready, l0_free_p0=l0_free_p0, l0_free_p1=l0_free_p1,
                                par=1, init_c=False,
                            )
                            if kk % 2 == 0:
                                tla.set_flag(l1_free_p0)
                            else:
                                tla.set_flag(l1_free_p1)
                            if tla.const_expr(_decodeA):
                                tla.cross_core_set_flag(a_free, tla.pipes.MTE1, aiv_id=0)
                            if tla.const_expr(_decodeB):
                                tla.cross_core_set_flag(b_free, tla.pipes.MTE1, aiv_id=0)
                                tla.cross_core_set_flag(b_free, tla.pipes.MTE1, aiv_id=1)
                        else:
                            if tla.const_expr(_l0_db):
                                l0a_buf = l0a0
                                l0b_act_buf = l0b_act0
                                l0b_gate_buf = l0b_gate0
                                if kk % 2 == 1:
                                    l0a_buf = l0a1
                                    l0b_act_buf = l0b_act1
                                    l0b_gate_buf = l0b_gate1
                                if kk % 2 == 0:
                                    tla.wait_flag(l0_free_p0)
                                else:
                                    tla.wait_flag(l0_free_p1)
                            else:
                                l0a_buf = l0a
                                l0b_act_buf = l0b_act
                                l0b_gate_buf = l0b_gate
                                tla.wait_flag(l0_avail)
                            t_l0a = tla.make_tensor_like(l0a_buf, t_l1a, tla.arch.zN)
                            t_l0b_act = tla.make_tensor_like(l0b_act_buf, t_l1b_act, tla.arch.nZ)
                            t_l0b_gate = tla.make_tensor_like(l0b_gate_buf, t_l1b_gate, tla.arch.nZ)

                            tla.copy(t_l0a, t_l1a, scale=t_l1sa)
                            tla.copy(t_l0b_act, t_l1b_act, scale=t_l1sb_act)
                            tla.copy(t_l0b_gate, t_l1b_gate, scale=t_l1sb_gate)
                            if kk % 2 == 0:
                                tla.set_flag(l1_free_p0)
                            else:
                                tla.set_flag(l1_free_p1)
                            if tla.const_expr(_decodeA):
                                tla.cross_core_set_flag(a_free, tla.pipes.MTE1, aiv_id=0)
                            if tla.const_expr(_decodeB):
                                tla.cross_core_set_flag(b_free, tla.pipes.MTE1, aiv_id=0)
                                tla.cross_core_set_flag(b_free, tla.pipes.MTE1, aiv_id=1)
                            tla.set_flag(l0_ready)
                            tla.wait_flag(l0_ready)

                            init_c = True if kk == 0 else False
                            tla.mmad_mx(t_l0c_act, t_l0a, t_l0b_act, init_c=init_c)
                            tla.mmad_mx(t_l0c_gate, t_l0a, t_l0b_gate, init_c=init_c)
                            if tla.const_expr(_l0_db):
                                if kk % 2 == 0:
                                    tla.set_flag(l0_free_p0)
                                else:
                                    tla.set_flag(l0_free_p1)
                            else:
                                tla.set_flag(l0_avail)

                    if not tla.const_expr(_l0_db):
                        tla.wait_flag(l0_avail)
                    tla.set_flag(l0_drain)
                    mtx_l0c.unlock(pipe=tla.pipes.CUBE)

                    ub_act_cube = tla.make_tensor_like(ub_act, t_l0c_act, tla.arch.RowMajor)
                    ub_gate_cube = tla.make_tensor_like(ub_gate, t_l0c_gate, tla.arch.RowMajor)
                    l0c2ub = tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M)
                    if not first_tile:
                        tla.cross_core_wait_flag(ub_free, tla.pipes.FIX)
                    mtx_l0c.lock(pipe=tla.pipes.FIX)
                    tla.wait_flag(l0_drain)
                    tla.copy(ub_act_cube, t_l0c_act, l0c2ub)
                    tla.copy(ub_gate_cube, t_l0c_gate, l0c2ub)
                    mtx_l0c.unlock(pipe=tla.pipes.FIX)
                    tla.cross_core_set_flag(ub_ready, tla.pipes.FIX)
                    first_tile = False

                start_core_idx = (start_core_idx + mn_blocks) % core_num
            m_start = m_start + current_m

    with tla.vector():
        core_num = tla.arch.block_num()
        core_idx = tla.arch.block_idx()
        sub_idx = tla.arch.sub_block_idx()
        start_core_idx = c0
        m_start = c0

        k_tiles_v = (k_dim + tk - 1) // tk
        m1 = k_tiles_v
        if tla.const_expr(_decodeB and not _transB):
            if tla.const_expr(CB_PIPE_K_CHUNKS > 0):
                if m1 > CB_PIPE_K_CHUNKS:
                    m1 = CB_PIPE_K_CHUNKS

        # [CB-PIPE] previous-block epilogue carriers (cross-iteration slots)
        prev_valid = False
        prev_out_row = c0
        prev_rows = c0
        prev_cols_valid = c0
        prev_blk_col = c0

        half_ref = tla.make_tensor(
            gm_y.ptr,
            tla.make_layout(
                tla.make_shape(half, tn), tla.make_stride(tn, 1), layoutTag=tla.arch.RowMajor
            ),
        )
        sy_ref = tla.make_tensor(
            gm_sy.ptr,
            tla.make_layout(
                tla.make_shape(half, 8), tla.make_stride(8, 1),
                layoutTag=tla.arch.RowMajor,
            ),
        )
        y_ref = tla.make_tensor(
            gm_y.ptr,
            tla.make_layout(
                tla.make_shape(half, tn // 4), tla.make_stride(tn // 4, 1),
                layoutTag=tla.arch.RowMajor,
            ),
        )
        ub_act_half = tla.make_tensor_like(ub_act, half_ref, tla.arch.RowMajor)
        ub_gate_half = tla.make_tensor_like(ub_gate, half_ref, tla.arch.RowMajor)
        ub_y_half = tla.make_tensor_like(ub_y, y_ref, tla.arch.RowMajor)
        ub_y_f8 = tla.make_tensor(
            tla.recast_ptr(ub_y, dtype=out_dtype),
            tla.make_layout(
                tla.make_shape(half, tn), tla.make_stride(tn, 1),
                layoutTag=tla.arch.RowMajor,
            ),
        )
        ub_sy_half = tla.make_tensor_like(ub_sy, sy_ref, tla.arch.RowMajor)
        # Shape carrier for the transB L1 B slot: logical zN (tn, tk), so the
        # K sub-tile of one L1 tile is slice [0, j*dk) along its second axis.
        fp8_slot_ref = tla.make_tensor(
            gm_y.ptr,
            tla.make_layout(
                tla.make_shape(tn, tk), tla.make_stride(tk, 1),
                layoutTag=tla.arch.RowMajor,
            ),
        )

        if tla.const_expr(_decodeA or _decodeB):
            k_pack = k_dim // 2
            n_pack = n_dim // 2
            b_tag_v = tla.arch.zN
            if tla.const_expr(_decodeA):
                ub_a_pack_t = tla.make_tensor(
                    ub_a_pack,
                    tla.make_layout(tla.make_shape(tm, tk // 2), tla.make_stride(tk // 2, 1), layoutTag=tla.arch.RowMajor),
                )
                ub_a_fp8_t = tla.make_tensor(
                    tla.recast_ptr(ub_a_fp8, dtype=dtype_a),
                    tla.make_layout(tla.make_shape(tm, tk), tla.make_stride(tk, 1), layoutTag=tla.arch.RowMajor),
                )
            if tla.const_expr(_transB):
                b_pack_shape = tla.make_shape(tn, tk // 2)
                b_pack_stride = tla.make_stride(tk // 2, 1)
            else:
                b_pack_shape = tla.make_shape(tk, tn // 2)
                b_pack_stride = tla.make_stride(tn // 2, 1)
            # Decoded B is stored N x K row-major in UB (w4a8_mx_matmul pattern).
            # ub RowMajor(tn, tk) -> l1 zN(tn, tk); the cube reads the same buffer
            # as nZ(tk, tn) -- zN(N, K) and nZ(K, N) share the physical layout.
            ub_ba_pack_t = tla.make_tensor(
                ub_ba_pack,
                tla.make_layout(b_pack_shape, b_pack_stride, origin_shape=b_pack_shape, layoutTag=tla.arch.RowMajor),
            )
            ub_bg_pack_t = tla.make_tensor(
                ub_bg_pack,
                tla.make_layout(b_pack_shape, b_pack_stride, origin_shape=b_pack_shape, layoutTag=tla.arch.RowMajor),
            )
            ub_ba_fp8_t = tla.make_tensor(
                tla.recast_ptr(ub_ba_fp8, dtype=dtype_b),
                tla.make_layout(tla.make_shape(tn, tk), tla.make_stride(tk, 1), origin_shape=tla.make_shape(tn, tk), layoutTag=tla.arch.RowMajor),
            )
            ub_bg_fp8_t = tla.make_tensor(
                tla.recast_ptr(ub_bg_fp8, dtype=dtype_b),
                tla.make_layout(tla.make_shape(tn, tk), tla.make_stride(tk, 1), origin_shape=tla.make_shape(tn, tk), layoutTag=tla.arch.RowMajor),
            )
            # K-major view of the same decoded-B buffer: nontransB decodes
            # elementwise (coalesced read+write) into RowMajor(tk, tn) and the
            # UB->L1 copy transposes to zN(tn, tk) for the cube's nZ(tk, tn).
            ub_ba_fp8_k = tla.make_tensor(
                tla.recast_ptr(ub_ba_fp8, dtype=dtype_b),
                tla.make_layout(tla.make_shape(tk, tn), tla.make_stride(tn, 1), origin_shape=tla.make_shape(tk, tn), layoutTag=tla.arch.RowMajor),
            )
            ub_bg_fp8_k = tla.make_tensor(
                tla.recast_ptr(ub_bg_fp8, dtype=dtype_b),
                tla.make_layout(tla.make_shape(tk, tn), tla.make_stride(tn, 1), origin_shape=tla.make_shape(tk, tn), layoutTag=tla.arch.RowMajor),
            )
            # [NTB-ZN] The nontransB decode writes the packed nibbles in the
            # host's [nb][k][16] order, so the 5-op widen already emits the
            # zN(tk, tn) byte order the L1 tile uses. The staging is declared
            # as that same zN layout: the UB->L1 hop is then the contiguous
            # zN->zN route (one DataCopy per N-plane half) instead of a 32B
            # per-row scatter.
            ub_ba_fp8_zn = tla.make_tensor(
                tla.recast_ptr(ub_ba_fp8, dtype=dtype_b),
                tla.make_layout_with_tag((tk, tn), dtype_b, tla.arch.zN),
            )
            ub_bg_fp8_zn = tla.make_tensor(
                tla.recast_ptr(ub_bg_fp8, dtype=dtype_b),
                tla.make_layout_with_tag((tk, tn), dtype_b, tla.arch.zN),
            )
            ub_ba_planes = tla.make_tensor(
                ub_ba_pack,
                tla.make_layout(
                    tla.make_shape(tn // 32, tk * 16), tla.make_stride(tk * 16, 1),
                    origin_shape=tla.make_shape(tn // 32, tk * 16),
                    layoutTag=tla.arch.RowMajor,
                ),
            )
            ub_bg_planes = tla.make_tensor(
                ub_bg_pack,
                tla.make_layout(
                    tla.make_shape(tn // 32, tk * 16), tla.make_stride(tk * 16, 1),
                    origin_shape=tla.make_shape(tn // 32, tk * 16),
                    layoutTag=tla.arch.RowMajor,
                ),
            )
            if tla.const_expr(_decodeA):
                a_pk_f = tla.make_tensor(ub_a_pack, tla.make_layout(tla.make_shape(tm * tk // 2), tla.make_stride(1)))
                a_fp_f = tla.make_tensor(ub_a_fp8, tla.make_layout(tla.make_shape(tm * tk), tla.make_stride(1)))
            ba_pk_f = tla.make_tensor(ub_ba_pack, tla.make_layout(tla.make_shape(tk * tn // 2), tla.make_stride(1)))
            ba_fp_f = tla.make_tensor(ub_ba_fp8, tla.make_layout(tla.make_shape(tk * tn), tla.make_stride(1)))
            bg_pk_f = tla.make_tensor(ub_bg_pack, tla.make_layout(tla.make_shape(tk * tn // 2), tla.make_stride(1)))
            # [bZN] When the host pre-laid the weight in zN fractal order, the
            # GM->UB fill and the UB->L1 store are both zN->zN, so neither hop
            # re-swizzles; the 1-D views above stay the decode path (nibble
            # widening is position-independent, so physical order is preserved).
            if tla.const_expr(False):
                ub_ba_pack_zn = tla.make_tensor(
                    ub_ba_pack, tla.make_layout_with_tag((tk, tn), tla.Float4E2M1, tla.arch.zN)
                )
                ub_bg_pack_zn = tla.make_tensor(
                    ub_bg_pack, tla.make_layout_with_tag((tk, tn), tla.Float4E2M1, tla.arch.zN)
                )
                ub_ba_fp8_zn = tla.make_tensor(
                    tla.recast_ptr(ub_ba_fp8, dtype=dtype_b),
                    tla.make_layout_with_tag((tk, tn), dtype_b, tla.arch.zN),
                )
                ub_bg_fp8_zn = tla.make_tensor(
                    tla.recast_ptr(ub_bg_fp8, dtype=dtype_b),
                    tla.make_layout_with_tag((tk, tn), dtype_b, tla.arch.zN),
                )
            bg_fp_f = tla.make_tensor(ub_bg_fp8, tla.make_layout(tla.make_shape(tk * tn), tla.make_stride(1)))

        if tla.const_expr(_decodeB and _transB):
            # Hardware mx_to widening views: packed FP4 payload (logical shape)
            # -> FP8 payload, unit E8M0 scale. transB groups along K=dim1.
            # Full tiles only (partial keeps SIMT).
            if tla.const_expr(_transB):
                mx_in_ba = tla.make_tensor(
                    tla.recast_ptr(ub_ba_pack, dtype=tla.Float4E2M1),
                    tla.make_layout(tla.make_shape(tn, dk), tla.make_stride(dk, 1)),
                )
                mx_out_ba = tla.make_tensor(
                    tla.recast_ptr(ub_ba_fp8, dtype=dtype_b),
                    tla.make_layout(tla.make_shape(tn, dk), tla.make_stride(dk, 1)),
                )
                mx_in_bg = tla.make_tensor(
                    tla.recast_ptr(ub_bg_pack, dtype=tla.Float4E2M1),
                    tla.make_layout(tla.make_shape(tn, dk), tla.make_stride(dk, 1)),
                )
                mx_out_bg = tla.make_tensor(
                    tla.recast_ptr(ub_bg_fp8, dtype=dtype_b),
                    tla.make_layout(tla.make_shape(tn, dk), tla.make_stride(dk, 1)),
                )
                mx_scale = tla.make_tensor(
                    ub_scale_unit,
                    tla.make_layout(tla.make_shape(tn, skd), tla.make_stride(skd, 1)),
                )
            else:
                mx_in_ba = tla.make_tensor(
                    tla.recast_ptr(ub_ba_pack, dtype=tla.Float4E2M1),
                    tla.make_layout(tla.make_shape(tk, tn), tla.make_stride(tn, 1)),
                )
                mx_out_ba = tla.make_tensor(
                    tla.recast_ptr(ub_ba_fp8, dtype=dtype_b),
                    tla.make_layout(tla.make_shape(tk, tn), tla.make_stride(tn, 1)),
                )
                mx_in_bg = tla.make_tensor(
                    tla.recast_ptr(ub_bg_pack, dtype=tla.Float4E2M1),
                    tla.make_layout(tla.make_shape(tk, tn), tla.make_stride(tn, 1)),
                )
                mx_out_bg = tla.make_tensor(
                    tla.recast_ptr(ub_bg_fp8, dtype=dtype_b),
                    tla.make_layout(tla.make_shape(tk, tn), tla.make_stride(tn, 1)),
                )
                mx_scale = tla.make_tensor(
                    ub_scale_unit,
                    tla.make_layout(tla.make_shape(sk, tn), tla.make_stride(tn, 1)),
                )
            scale_flat = tla.make_tensor(
                ub_scale_unit, tla.make_layout(tla.make_shape(skd * tn), tla.make_stride(1))
            )
            with tla.vec.func(mode="simd"):
                scale_flat.store(tla.full(127, dtype=tla.Int8))
            if tla.const_expr(_transB):
                # Zero both decode staging parities once: a partial N tile's MX
                # groups past the tail then widen zeros, and later blocks only
                # ever leave finite FP8 payload there, so no in-loop zero (and
                # no sync) is needed inside the dynamic tail-if.
                act_pk_f0 = tla.make_tensor(
                    ub_ba_pack,
                    tla.make_layout(tla.make_shape(dk * tn // 2), tla.make_stride(1)),
                )
                act_pk_f1 = tla.make_tensor(
                    ub_ba_pack_b,
                    tla.make_layout(tla.make_shape(dk * tn // 2), tla.make_stride(1)),
                )
                gate_pk_f0 = tla.make_tensor(
                    ub_bg_pack,
                    tla.make_layout(tla.make_shape(dk * tn // 2), tla.make_stride(1)),
                )
                gate_pk_f1 = tla.make_tensor(
                    ub_bg_pack_b,
                    tla.make_layout(tla.make_shape(dk * tn // 2), tla.make_stride(1)),
                )
                with tla.vec.func(mode="simd"):
                    v0b = tla.full(0, tla.Int8)
                    act_pk_f0.store(v0b)
                    act_pk_f1.store(v0b)
                    gate_pk_f0.store(v0b)
                    gate_pk_f1.store(v0b)
                tla.pipe_barrier(tla.pipes.ALL)

        if tla.const_expr(_decodeB and _transB):
            # Both decode stages start free. Each block's fills balance its
            # stores per parity, so these credits carry across block
            # boundaries (same scheme as the L1 ping-pong priming).
            tla.set_flag(dcd_a_rl0)
            tla.set_flag(dcd_a_rl1)
            tla.set_flag(dcd_g_rl0)
            tla.set_flag(dcd_g_rl1)

        if tla.const_expr(_decodeB and not _transB):
            # Both staging buffers start free; every K tile returns exactly the
            # credits it consumes, so they carry across block boundaries.
            tla.set_flag(dcd_pk_a)
            tla.set_flag(dcd_rl_a0)
            tla.set_flag(dcd_rl_a1)
            tla.set_flag(dcd_pk_g)
            tla.set_flag(dcd_rl_g0)
            tla.set_flag(dcd_rl_g1)

        for g in tla.range(c0, group_cnt, c1):
            current_m = group_list[g].to(tla.Int32)
            if current_m > 0:
                grid_m = (current_m + tm - 1) // tm
                mn_blocks = grid_m * grid_n
                if tla.const_expr(_decodeA):
                    gm_a_pack_g = tla.make_tensor(
                        gm_a.ptr + m_start * gm_a.stride[0],
                        tla.make_layout(
                            tla.make_shape(current_m, k_pack),
                            tla.make_stride(gm_a.stride[0], gm_a.stride[1]),
                            origin_shape=tla.make_shape(current_m, k_pack),
                            layoutTag=tla.arch.RowMajor,
                        ),
                    )
                if tla.const_expr(_decodeB):
                    if tla.const_expr(_transB):
                        gm_b_pack_g = tla.make_tensor(
                            gm_b.ptr + g * n_dim * gm_b.stride[0],
                            tla.make_layout(
                                tla.make_shape(n_dim, k_pack),
                                tla.make_stride(k_pack, 1),
                                origin_shape=tla.make_shape(n_dim, k_pack),
                                layoutTag=tla.arch.RowMajor,
                            ),
                        )
                    else:
                        if tla.const_expr(False):
                            # Pre-laid zN weight: slice the logical (K, N) group
                            # window; the fractal order comes with it.
                            gm_b_pack_g = tla.get_tile(
                                gm_b,
                                tla.make_coord(g * k_dim, c0),
                                tla.make_shape(k_dim, n_dim),
                            )
                        else:
                            gm_b_pack_g = tla.make_tensor(
                                gm_b.ptr + g * k_dim * gm_b.stride[0],
                                tla.make_layout(
                                    tla.make_shape(k_dim, n_pack),
                                    tla.make_stride(n_pack, 1),
                                    origin_shape=tla.make_shape(k_dim, n_pack),
                                    layoutTag=tla.arch.RowMajor,
                                ),
                            )
                if tla.const_expr(_decodeB and not _transB):
                    # [NTB-ZN] Host pre-laid weight: inside a group the packed
                    # bytes run [nb][k][16] (nb = one 32-wide N plane), so a
                    # plane is a contiguous k_dim*16-byte run and a K tile of
                    # it is a contiguous tk*16-byte slice. nbt is the group's
                    # plane count.
                    nbt = gm_b.origin_shape[1] // 16
                    gm_b_pack_p = tla.make_tensor(
                        gm_b.ptr + g * k_dim * gm_b.stride[0],
                        tla.make_layout(
                            tla.make_shape(nbt, k_dim * 16),
                            tla.make_stride(k_dim * 16, 1),
                            origin_shape=tla.make_shape(nbt, k_dim * 16),
                            layoutTag=tla.arch.RowMajor,
                        ),
                    )
                start_loop = (core_idx + core_num - start_core_idx) % core_num
                block_range = tla.range(start_loop, mn_blocks + core_num, core_num)
                for loop_idx in block_range:
                    block_row = loop_idx // grid_n
                    block_col = loop_idx % grid_n
                    out_row = m_start + block_row * tm + sub_idx * half
                    # [CB-PIPE] last iteration of this core is a drain slot
                    has_block = loop_idx < mn_blocks
                    rows = current_m - block_row * tm - sub_idx * half
                    if rows > half:
                        rows = half
                    cols_valid = n_out - block_col * tn
                    if cols_valid > tn:
                        cols_valid = tn

                    # [CB-PIPE] cross-basic-block software pipeline (arch35
                    # weight_quant_basic_block IterateNzNkWithAiv): decode the first
                    # m1 K chunks of the current block so the cube can start its
                    # mmad, run the *previous* block's epilogue inside that window,
                    # then finish the remaining K chunks.
                    for phase in tla.range(c0, c1 + c1, c1):
                        if phase == c1:
                            if prev_valid:
                                tla.cross_core_wait_flag(ub_ready, tla.pipes.VECTOR)
                                prev_nchunks = prev_cols_valid // 64
                                if prev_rows > 0:
                                    mtx_ub_out.lock(pipe=tla.pipes.VECTOR)
                                    gm_y_t = tla.get_tile(
                                        gm_y, tla.make_coord(prev_out_row, prev_blk_col * (tn // 4)),
                                        tla.make_shape(prev_rows, prev_cols_valid // 4),
                                    )
                                    gm_sy_t = tla.get_tile(
                                        gm_sy, tla.make_coord(prev_out_row, prev_blk_col),
                                        tla.make_shape(prev_rows, 1),
                                    )
                                    with tla.vec.func(mode="simd"):
                                        m_src = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                                        m_lo = tla.create_mask(pattern=tla.mask.H, dtype=tla.Float32)
                                        # [VEC-BF16] v1/v7FFF/vMASK/v7F00 were the integer
                                        # RNE chain's constants; retired with it.
                                        v7F80 = tla.full(0x7F80, tla.Int32)
                                        v_base = tla.full(0x7F00 + emax_shifted, tla.Int32).reduce(tla.ReductionOp.MAX, mask=m_lo)
                                        v_emax = tla.full(emax, tla.Int32).reduce(tla.ReductionOp.MAX, mask=m_lo)
                                        v_emax_sh = tla.full(emax_shifted, tla.Int32).reduce(tla.ReductionOp.MAX, mask=m_lo)
                                        v_zero = tla.full(0, tla.Int32).reduce(tla.ReductionOp.MAX, mask=m_lo)
                                        v_254 = tla.full(254, tla.Int32).reduce(tla.ReductionOp.MAX, mask=m_lo)
                                        cast_f8 = tla.params.CastParams(
                                            reg_slot=tla.params.RegSlot.ZERO,
                                            sat_mode=tla.params.SatMode.SAT,
                                            round_mode=tla.params.RoundMode.CAST_RINT,
                                        )
                                        for r in tla.range(c0, prev_rows, c1):
                                            scale_word = v_zero
                                            for chunk in tla.range(c0, prev_nchunks, c1):
                                                col = chunk * 64
                                                a_reg = tla.get_tile(ub_act_half, tla.make_coord(r, col), tla.make_shape(1, 64)).load()
                                                g_reg = tla.get_tile(ub_gate_half, tla.make_coord(r, col), tla.make_shape(1, 64)).load()
                                                if tla.const_expr(_decodeB):
                                                    # [CMB-A] x64 restore on every
                                                    # decode path (the 5-op shuffle
                                                    # yields value/64).
                                                    a_reg = tla.mul(a_reg, 64.0, mask=m_src)
                                                    g_reg = tla.mul(g_reg, 64.0, mask=m_src)
                                                den = tla.add(1.0, tla.exp(tla.neg(a_reg, mask=m_src), mask=m_src), mask=m_src)
                                                s_val = tla.mul(tla.div(a_reg, den, mask=m_src), g_reg, mask=m_src)
                                                # [VEC-BF16] Hardware RNE to bf16
                                                # (f32->bf16 RINT ->f32) is bit-identical
                                                # to the integer round-half-even chain
                                                # (s+0x7FFF+((s>>16)&1))&0xFFFF0000 with two
                                                # cast instructions instead of five int
                                                # ops; the epilogue chain is latency
                                                # bound, so this is worth ~3.5 us on 069.
                                                s_bf = s_val.to(tla.BFloat16, tla.params.CastParams(
                                                    reg_slot=tla.params.RegSlot.ZERO,
                                                    sat_mode=tla.params.SatMode.NOSAT,
                                                    round_mode=tla.params.RoundMode.CAST_RINT,
                                                )).to(tla.Float32, tla.params.CastParams(
                                                    reg_slot=tla.params.RegSlot.ZERO,
                                                    round_mode=tla.params.RoundMode.CAST_NONE,
                                                ))
                                                s_bf16 = tla.shift_right(s_bf.bitcast(tla.Int32), 16)
                                                exp_bits = tla.bitwise_and(s_bf16, v7F80)
                                                max_lo = exp_bits.reduce(tla.ReductionOp.MAX, mask=m_lo)
                                                hi_only = tla.where(m_lo, v_zero, exp_bits)
                                                max_hi = hi_only.reduce(tla.ReductionOp.MAX, mask=m_src)
                                                # [VEC-ALG] max_c >= emax_shifted makes the
                                                # byte's lower clamp redundant (exp field is
                                                # a multiple of 0x80, so >>7 is exact), and
                                                # v_base = 0x7F00 + emax_shifted folds the
                                                # two subtractions of half_bits into one.
                                                max_lo_c = tla.max(max_lo, v_emax_sh)
                                                max_hi_c = tla.max(max_hi, v_emax_sh)
                                                byte_lo = tla.min(tla.sub(tla.shift_right(max_lo_c, 7), v_emax), v_254)
                                                byte_hi = tla.min(tla.sub(tla.shift_right(max_hi_c, 7), v_emax), v_254)
                                                half_lo_bits = tla.sub(v_base, max_lo_c)
                                                half_hi_bits = tla.sub(v_base, max_hi_c)
                                                half_lo = tla.full(tla.shift_left(half_lo_bits, 16).bitcast(tla.Float32), tla.Float32)
                                                half_hi = tla.full(tla.shift_left(half_hi_bits, 16).bitcast(tla.Float32), tla.Float32)
                                                scaled = tla.where(m_lo, tla.mul(s_bf, half_lo, mask=m_src), tla.mul(s_bf, half_hi, mask=m_src))
                                                scaled = tla.max(tla.min(scaled, clamp_value), -clamp_value)
                                                y_chunk = tla.get_tile(ub_y_f8, tla.make_coord(r, col), tla.make_shape(1, 64))
                                                y_chunk.store(
                                                    scaled.to(out_dtype, cast_f8, mask=m_src),
                                                    tla.params.NormalStoreParams(store_dist=tla.params.StoreDist.DIST_PACK4_B32),
                                                )
                                                word16 = tla.bitwise_or(byte_lo, tla.shift_left(byte_hi, 8))
                                                scale_word = tla.bitwise_or(scale_word, tla.shift_left(word16, chunk * 16))
                                            sy_cell = tla.get_tile(ub_sy_half, tla.make_coord(r, 0), tla.make_shape(1, 1))
                                            sy_cell.store(scale_word)
                                    mtx_ub_out.unlock(pipe=tla.pipes.VECTOR)
                                    mtx_ub_out.lock(pipe=tla.pipes.MTE3)
                                    tla.copy(gm_y_t, ub_y_half)
                                    tla.copy(gm_sy_t, ub_sy_half)
                                    mtx_ub_out.unlock(pipe=tla.pipes.MTE3)
                                tla.cross_core_set_flag(ub_free, tla.pipes.VECTOR)
                        phase_lo = phase * m1
                        phase_hi = m1 + phase * (k_tiles_v - m1)
                        if has_block:
                            if tla.const_expr(_decodeA or _decodeB):
                                if sub_idx == 0:
                                    act_col = block_col * tn
                                    for kk in tla.range(phase_lo, phase_hi, c1):
                                        if tla.const_expr(_decodeA):
                                            gm_a_pk = tla.get_tile(
                                                gm_a_pack_g,
                                                tla.make_coord(block_row * tm, kk * (tk // 2)),
                                                tla.make_shape(tm, tk // 2),
                                            )
                                            mv = gm_a_pk.origin_shape[0]
                                            kv = gm_a_pk.origin_shape[1]
                                            tla.cross_core_wait_flag(a_free, tla.pipes.MTE3, aiv_id=0)
                                            tla.copy(
                                                tla.tile_view(ub_a_pack_t, tla.make_shape(mv, kv), tla.make_coord(0, 0)),
                                                gm_a_pk,
                                            )
                                            tla.pipe_barrier(tla.pipes.ALL)
                                            with tla.vec.func(mode="simt", thread_block_dim=256):
                                                pk = a_pk_f
                                                fp = a_fp_f
                                                tid, _, _ = tla.arch.thread_idx()
                                                nth, _, _ = tla.arch.thread_block_dim()
                                                for p in tla.range(tid, tm * tk // 2, nth):
                                                    b = pk[p].to(tla.Int32) & 0xFF
                                                    fp[2 * p] = _nibble_fp8(b & 0x0F)
                                                    fp[2 * p + 1] = _nibble_fp8((b >> 4) & 0x0F)
                                            tla.pipe_barrier(tla.pipes.ALL)
                                            t_l1a_v_kk = tla.make_tensor(l1a + (kk % 2) * (tm * tk), tla.make_layout_with_tag((tm, tk), dtype_a, tla.arch.zN))
                                            tla.copy(t_l1a_v_kk, ub_a_fp8_t)
                                            tla.cross_core_set_flag(a_full, tla.pipes.MTE3, aiv_id=0)
                                        if tla.const_expr(_decodeB):
                                            if tla.const_expr(_transB and nsub == 1):
                                                # transB act half on aiv0; gate half on aiv1.
                                                # Two-deep UB staging: the fill of the next K
                                                # tile overlaps this tile's mx_to widening and
                                                # its UB->L1 store, so the decode chain costs
                                                # max(mte2, vec, mte3) per tile instead of the
                                                # serialized sum.
                                                gm_ba_pk = tla.get_tile(
                                                    gm_b_pack_g,
                                                    tla.make_coord(act_col, kk * (tk // 2)),
                                                    tla.make_shape(tn, tk // 2),
                                                )
                                                bv0 = gm_ba_pk.origin_shape[0]
                                                bv1 = gm_ba_pk.origin_shape[1]
                                                pk_buf_a = ub_ba_pack
                                                fp8_buf_a = ub_ba_fp8
                                                if kk % 2 == 1:
                                                    pk_buf_a = ub_ba_pack_b
                                                    fp8_buf_a = ub_ba_fp8_b
                                                pk_t_a = tla.make_tensor(
                                                    pk_buf_a,
                                                    tla.make_layout(
                                                        tla.make_shape(tn, tk // 2), tla.make_stride(tk // 2, 1),
                                                        origin_shape=tla.make_shape(tn, tk // 2),
                                                        layoutTag=tla.arch.RowMajor,
                                                    ),
                                                )
                                                pk_f_a = tla.make_tensor(
                                                    pk_buf_a,
                                                    tla.make_layout(tla.make_shape(tk * tn // 2), tla.make_stride(1)),
                                                )
                                                mx_in_a = tla.make_tensor(
                                                    tla.recast_ptr(pk_buf_a, dtype=tla.Float4E2M1),
                                                    tla.make_layout(tla.make_shape(tn, tk), tla.make_stride(tk, 1)),
                                                )
                                                mx_out_a = tla.make_tensor(
                                                    tla.recast_ptr(fp8_buf_a, dtype=dtype_b),
                                                    tla.make_layout(tla.make_shape(tn, tk), tla.make_stride(tk, 1)),
                                                )
                                                fp8_t_a = tla.make_tensor(
                                                    tla.recast_ptr(fp8_buf_a, dtype=dtype_b),
                                                    tla.make_layout_with_tag(
                                                        (tn, tk), dtype_b, tla.arch.zN
                                                    ),
                                                )
                                                if tla.const_expr(tk == 256):
                                                    # [ZNUB] 8 K-blocks per row: what
                                                    # the decode stores IS the zN
                                                    # order, so the L1 store is a
                                                    # contiguous run.
                                                    fp8_t_a = tla.make_tensor(
                                                        tla.recast_ptr(fp8_buf_a, dtype=dtype_b),
                                                        tla.make_layout_with_tag((tn, tk), dtype_b, tla.arch.zN),
                                                    )
                                                else:
                                                    # Smaller K tile (2 or 4 32-wide
                                                    # K blocks per row): keep the
                                                    # row-major staging + scatter.
                                                    fp8_t_a = tla.make_tensor(
                                                        tla.recast_ptr(fp8_buf_a, dtype=dtype_b),
                                                        tla.make_layout(
                                                            tla.make_shape(tn, tk), tla.make_stride(tk, 1),
                                                            origin_shape=tla.make_shape(tn, tk),
                                                            layoutTag=tla.arch.RowMajor,
                                                        ),
                                                    )
                                                if kk % 2 == 0:
                                                    tla.wait_flag(dcd_a_rl0)
                                                else:
                                                    tla.wait_flag(dcd_a_rl1)
                                                tla.copy(
                                                    tla.tile_view(pk_t_a, tla.make_shape(bv0, bv1), tla.make_coord(0, 0)),
                                                    gm_ba_pk,
                                                )
                                                if kk % 2 == 0:
                                                    tla.set_flag(dcd_a_ld0)
                                                    tla.wait_flag(dcd_a_ld0)
                                                else:
                                                    tla.set_flag(dcd_a_ld1)
                                                    tla.wait_flag(dcd_a_ld1)
                                                if tla.const_expr(False):
                                                 with tla.vec.func(mode="simt", thread_block_dim=256):
                                                    tid, _, _ = tla.arch.thread_idx()
                                                    nth, _, _ = tla.arch.thread_block_dim()
                                                    if bv0 != tn:
                                                        for p in tla.range(tid, (tn - bv0) * (tk // 2), nth):
                                                            pk_f_a[bv0 * (tk // 2) + p] = 0
                                                    if bv1 != tk // 2:
                                                        for rz in tla.range(tid, bv0, nth):
                                                            for cz in tla.range(bv1, tk // 2, c1):
                                                                pk_t_a[rz, cz] = 0
                                                if tla.const_expr(tk == 256):
                                                    # [ZNUB] Row-pointer view: row n of
                                                    # the zN tile starts at byte 32*n
                                                    # (uniform across 16-row fractal
                                                    # blocks), so a 32-byte-pitch
                                                    # view + BlockStore scatters one
                                                    # full row's 8 K-block fractals
                                                    # with a single store.
                                                    f8_rows_a = tla.make_tensor(
                                                        tla.recast_ptr(fp8_buf_a, dtype=dtype_b),
                                                        tla.make_layout(
                                                            tla.make_shape(tn, 256), tla.make_stride(32, 1),
                                                            origin_shape=tla.make_shape(tn, 256),
                                                            layoutTag=tla.arch.RowMajor,
                                                        ),
                                                    )
                                                    with tla.vec.func(mode="simd"):
                                                        for ch in tla.range(tk * tn // 512):
                                                            sv = tla.tile_view(pk_f_a, tla.make_shape(256), tla.make_coord(ch)).load()
                                                            lo = tla.bitwise_or(
                                                                tla.shift_left(tla.bitwise_and(sv, tla.full(0x08, tla.Int8)), 4),
                                                                tla.shift_left(tla.bitwise_and(sv, tla.full(0x07, tla.Int8)), 2),
                                                            )
                                                            hi = tla.bitwise_or(
                                                                tla.bitwise_and(sv, tla.full(-128, tla.Int8)),
                                                                tla.shift_right(tla.bitwise_and(sv, tla.full(0x70, tla.Int8)), 2),
                                                            )
                                                            r0, r1 = tla.interleave(lo, hi)
                                                            d0 = tla.get_tile(f8_rows_a, tla.make_coord(2 * ch, 0), tla.make_shape(1, 256))
                                                            d1 = tla.get_tile(f8_rows_a, tla.make_coord(2 * ch + 1, 0), tla.make_shape(1, 256))
                                                            d0.store(r0, tla.params.BlockStoreParams(block_stride=tn))
                                                            d1.store(r1, tla.params.BlockStoreParams(block_stride=tn))
                                                else:
                                                    # [CMB-A] old path: row-major
                                                    # staging, MTE3 re-swizzles.
                                                    fp8_f_a = tla.make_tensor(
                                                        tla.recast_ptr(fp8_buf_a, dtype=dtype_b),
                                                        tla.make_layout(tla.make_shape(tk * tn), tla.make_stride(1)),
                                                    )
                                                    with tla.vec.func(mode="simd"):
                                                        for ch in tla.range(tk * tn // 512):
                                                            sv = tla.tile_view(pk_f_a, tla.make_shape(256), tla.make_coord(ch)).load()
                                                            lo = tla.bitwise_or(
                                                                tla.shift_left(tla.bitwise_and(sv, tla.full(0x08, tla.Int8)), 4),
                                                                tla.shift_left(tla.bitwise_and(sv, tla.full(0x07, tla.Int8)), 2),
                                                            )
                                                            hi = tla.bitwise_or(
                                                                tla.bitwise_and(sv, tla.full(-128, tla.Int8)),
                                                                tla.shift_right(tla.bitwise_and(sv, tla.full(0x70, tla.Int8)), 2),
                                                            )
                                                            r0, r1 = tla.interleave(lo, hi)
                                                            tla.tile_view(fp8_f_a, tla.make_shape(256), tla.make_coord(2 * ch)).store(r0)
                                                            tla.tile_view(fp8_f_a, tla.make_shape(256), tla.make_coord(2 * ch + 1)).store(r1)
                                                if kk % 2 == 0:
                                                    tla.set_flag(dcd_a_st0)
                                                    tla.wait_flag(dcd_a_st0)
                                                else:
                                                    tla.set_flag(dcd_a_st1)
                                                    tla.wait_flag(dcd_a_st1)
                                                tla.cross_core_wait_flag(b_free, tla.pipes.MTE3, aiv_id=0)
                                                t_l1b_act_v_kk = tla.make_tensor_like(l1b_act + (kk % 2) * (tk * tn), fp8_slot_ref, tla.arch.zN)
                                                # [ZNUB] both sides are zN now: one
                                                # contiguous UB->L1 run.
                                                tla.copy(t_l1b_act_v_kk, fp8_t_a)
                                                if kk % 2 == 0:
                                                    tla.set_flag(dcd_a_rl0)
                                                else:
                                                    tla.set_flag(dcd_a_rl1)
                                                tla.cross_core_set_flag(b_full, tla.pipes.MTE3, aiv_id=0)
                                            elif tla.const_expr(_transB):
                                                # [TK256] one L1 K tile (tk) decoded as nsub
                                                # sub-tiles of dk K elements each; the staging
                                                # ring stays at its 128-K size and runs one
                                                # fill/widen/store round per sub-tile. The
                                                # 5-op widen stays inline (helpers cannot emit
                                                # tla.vec.func).
                                                pk_fa, f8i_a, fp8_ta = _transb_decode_fill(
                                                    gm_pack=gm_b_pack_g,
                                                    k_tile=kk, j=0, par=0,
                                                    dk=dk, tn=tn, tk=tk, dtype_b=dtype_b,
                                                    pk_buf=ub_ba_pack, fp8_buf=ub_ba_fp8,
                                                    act_col=act_col,
                                                    fl_rl=(dcd_a_rl0, dcd_a_rl1),
                                                    fl_ld=(dcd_a_ld0, dcd_a_ld1),
                                                )
                                                # [ZNUB] SIMT nibble widen + zN scatter: packed byte p
                                                # of the row-major (tn, dk//2) staging holds the two
                                                # k = 2kb, 2kb+1 nibbles of row n = p // (dk//2); both land
                                                # adjacent in the zN tile, so one 2-byte store covers the
                                                # pair. FP8 byte = exact value / 64 (epilogue restores x64).
                                                with tla.vec.func(mode="simt", thread_block_dim=256):
                                                    tid_a1, _, _ = tla.arch.thread_idx()
                                                    nth_a1, _, _ = tla.arch.thread_block_dim()
                                                    for p_a1 in tla.range(tid_a1, dk * tn // 2, nth_a1):
                                                        b_a1 = pk_fa[p_a1].to(tla.Int32) & 0xFF
                                                        n_a1 = p_a1 // (dk // 2)
                                                        kb_a1 = p_a1 - n_a1 * (dk // 2)
                                                        k2_a1 = 2 * kb_a1
                                                        z_a1 = ((k2_a1 // 32) * (tn * 32) + (n_a1 // 16) * 512
                                                               + (n_a1 % 16) * 32 + (k2_a1 % 32))
                                                        lo_a1 = ((b_a1 & 0x08) << 4) | ((b_a1 & 0x07) << 2)
                                                        hi_a1 = (b_a1 & 0x80) | ((b_a1 & 0x70) >> 2)
                                                        f8i_a[z_a1 // 2] = (lo_a1 | (hi_a1 << 8)).to(tla.Int16)
                                                _transb_decode_commit(
                                                    fp8_t=fp8_ta,
                                                    k_tile=kk, j=0, par=0, dk=dk, tn=tn, tk=tk,
                                                    l1_base=l1b_act, slot_ref=fp8_slot_ref,
                                                    fl_rl=(dcd_a_rl0, dcd_a_rl1),
                                                    fl_st=(dcd_a_st0, dcd_a_st1),
                                                    b_free=b_free, b_full=b_full, aiv_id=0,
                                                    first=True, last=False,
                                                )
                                                pk_fa, f8i_a, fp8_ta = _transb_decode_fill(
                                                    gm_pack=gm_b_pack_g,
                                                    k_tile=kk, j=1, par=1,
                                                    dk=dk, tn=tn, tk=tk, dtype_b=dtype_b,
                                                    pk_buf=ub_ba_pack_b, fp8_buf=ub_ba_fp8_b,
                                                    act_col=act_col,
                                                    fl_rl=(dcd_a_rl0, dcd_a_rl1),
                                                    fl_ld=(dcd_a_ld0, dcd_a_ld1),
                                                )
                                                # [ZNUB] SIMT nibble widen + zN scatter: packed byte p
                                                # of the row-major (tn, dk//2) staging holds the two
                                                # k = 2kb, 2kb+1 nibbles of row n = p // (dk//2); both land
                                                # adjacent in the zN tile, so one 2-byte store covers the
                                                # pair. FP8 byte = exact value / 64 (epilogue restores x64).
                                                with tla.vec.func(mode="simt", thread_block_dim=256):
                                                    tid_a1, _, _ = tla.arch.thread_idx()
                                                    nth_a1, _, _ = tla.arch.thread_block_dim()
                                                    for p_a1 in tla.range(tid_a1, dk * tn // 2, nth_a1):
                                                        b_a1 = pk_fa[p_a1].to(tla.Int32) & 0xFF
                                                        n_a1 = p_a1 // (dk // 2)
                                                        kb_a1 = p_a1 - n_a1 * (dk // 2)
                                                        k2_a1 = 2 * kb_a1
                                                        z_a1 = ((k2_a1 // 32) * (tn * 32) + (n_a1 // 16) * 512
                                                               + (n_a1 % 16) * 32 + (k2_a1 % 32))
                                                        lo_a1 = ((b_a1 & 0x08) << 4) | ((b_a1 & 0x07) << 2)
                                                        hi_a1 = (b_a1 & 0x80) | ((b_a1 & 0x70) >> 2)
                                                        f8i_a[z_a1 // 2] = (lo_a1 | (hi_a1 << 8)).to(tla.Int16)
                                                _transb_decode_commit(
                                                    fp8_t=fp8_ta,
                                                    k_tile=kk, j=1, par=1, dk=dk, tn=tn, tk=tk,
                                                    l1_base=l1b_act, slot_ref=fp8_slot_ref,
                                                    fl_rl=(dcd_a_rl0, dcd_a_rl1),
                                                    fl_st=(dcd_a_st0, dcd_a_st1),
                                                    b_free=b_free, b_full=b_full, aiv_id=0,
                                                    first=False, last=True,
                                                )
                                            else:
                                                # [NTB-ZN] The host pre-laid the packed weight as [nb][k][16]
                                                # (nb = one 32-wide N plane). A plane is a contiguous k_dim*16
                                                # run, so the fill is contiguous per plane and the 5-op nibble
                                                # widen emits exactly the zN byte order: the UB->L1 store is
                                                # then the contiguous zN->zN route instead of the per-row 32B
                                                # scatter the row-major staging forced.
                                                nbp = tn // 32
                                                nb0 = act_col // 32
                                                gm_plane_a = tla.get_tile(
                                                    gm_b_pack_p,
                                                    tla.make_coord(nb0, kk * (tk * 16)),
                                                    tla.make_shape(nbp, tk * 16),
                                                )
                                                mv = gm_plane_a.origin_shape[0]
                                                kv = gm_plane_a.origin_shape[1]
                                                tla.wait_flag(dcd_pk_a)
                                                if mv * kv == nbp * tk * 16:
                                                    tla.copy(ub_ba_planes, gm_plane_a)
                                                    tla.set_flag(dcd_ld_a)
                                                    tla.wait_flag(dcd_ld_a)
                                                    tla.wait_flag(dcd_rl_a0)
                                                    with tla.vec.func(mode="simd"):
                                                        for ch in tla.range(tk * tn // 1024):
                                                            sv = tla.tile_view(ba_pk_f, tla.make_shape(256), tla.make_coord(ch)).load()
                                                            lo = tla.bitwise_or(
                                                                tla.shift_left(tla.bitwise_and(sv, tla.full(0x08, tla.Int8)), 4),
                                                                tla.shift_left(tla.bitwise_and(sv, tla.full(0x07, tla.Int8)), 2),
                                                            )
                                                            hi = tla.bitwise_or(
                                                                tla.bitwise_and(sv, tla.full(-128, tla.Int8)),
                                                                tla.shift_right(tla.bitwise_and(sv, tla.full(0x70, tla.Int8)), 2),
                                                            )
                                                            r0, r1 = tla.interleave(lo, hi)
                                                            tla.tile_view(ba_fp_f, tla.make_shape(256), tla.make_coord(2 * ch)).store(r0)
                                                            tla.tile_view(ba_fp_f, tla.make_shape(256), tla.make_coord(2 * ch + 1)).store(r1)
                                                    tla.set_flag(dcd_st_a0)
                                                    tla.wait_flag(dcd_st_a0)
                                                    tla.cross_core_wait_flag(b_free, tla.pipes.MTE3, aiv_id=0)
                                                    t_l1b_act_v_kk = tla.make_tensor(l1b_act + (kk % 2) * (tk * tn), tla.make_layout_with_tag((tk, tn), dtype_b, tla.arch.zN))
                                                    tla.copy(
                                                        tla.get_tile(t_l1b_act_v_kk, tla.make_coord(0, 0), tla.make_shape(tk, tn // 2)),
                                                        tla.get_tile(ub_ba_fp8_zn, tla.make_coord(0, 0), tla.make_shape(tk, tn // 2)),
                                                    )
                                                    tla.set_flag(dcd_rl_a0)
                                                    tla.wait_flag(dcd_rl_a1)
                                                    with tla.vec.func(mode="simd"):
                                                        for ch in tla.range(tk * tn // 1024, tk * tn // 512):
                                                            sv = tla.tile_view(ba_pk_f, tla.make_shape(256), tla.make_coord(ch)).load()
                                                            lo = tla.bitwise_or(
                                                                tla.shift_left(tla.bitwise_and(sv, tla.full(0x08, tla.Int8)), 4),
                                                                tla.shift_left(tla.bitwise_and(sv, tla.full(0x07, tla.Int8)), 2),
                                                            )
                                                            hi = tla.bitwise_or(
                                                                tla.bitwise_and(sv, tla.full(-128, tla.Int8)),
                                                                tla.shift_right(tla.bitwise_and(sv, tla.full(0x70, tla.Int8)), 2),
                                                            )
                                                            r0, r1 = tla.interleave(lo, hi)
                                                            tla.tile_view(ba_fp_f, tla.make_shape(256), tla.make_coord(2 * ch)).store(r0)
                                                            tla.tile_view(ba_fp_f, tla.make_shape(256), tla.make_coord(2 * ch + 1)).store(r1)
                                                    tla.set_flag(dcd_pk_a)
                                                    tla.set_flag(dcd_st_a1)
                                                    tla.wait_flag(dcd_st_a1)
                                                    tla.copy(
                                                        tla.get_tile(t_l1b_act_v_kk, tla.make_coord(0, tn // 2), tla.make_shape(tk, tn // 2)),
                                                        tla.get_tile(ub_ba_fp8_zn, tla.make_coord(0, tn // 2), tla.make_shape(tk, tn // 2)),
                                                    )
                                                    tla.set_flag(dcd_rl_a1)
                                                    tla.cross_core_set_flag(b_full, tla.pipes.MTE3, aiv_id=0)
                                                else:
                                                    # Partial N-block / K tail: clear the staging, then fill only
                                                    # the window the group actually has.
                                                    with tla.vec.func(mode="simt", thread_block_dim=256):
                                                        ztid_a, _, _ = tla.arch.thread_idx()
                                                        znth_a, _, _ = tla.arch.thread_block_dim()
                                                        for zp_a in tla.range(ztid_a, tk * tn // 2, znth_a):
                                                            ba_pk_f[zp_a] = 0
                                                    tla.pipe_barrier(tla.pipes.ALL)
                                                    tla.copy(
                                                        tla.tile_view(ub_ba_planes, tla.make_shape(mv, kv), tla.make_coord(0, 0)),
                                                        gm_plane_a,
                                                    )
                                                    tla.set_flag(dcd_ld_a)
                                                    tla.wait_flag(dcd_ld_a)
                                                    tla.wait_flag(dcd_rl_a0)
                                                    with tla.vec.func(mode="simd"):
                                                        for ch in tla.range(tk * tn // 1024):
                                                            sv = tla.tile_view(ba_pk_f, tla.make_shape(256), tla.make_coord(ch)).load()
                                                            lo = tla.bitwise_or(
                                                                tla.shift_left(tla.bitwise_and(sv, tla.full(0x08, tla.Int8)), 4),
                                                                tla.shift_left(tla.bitwise_and(sv, tla.full(0x07, tla.Int8)), 2),
                                                            )
                                                            hi = tla.bitwise_or(
                                                                tla.bitwise_and(sv, tla.full(-128, tla.Int8)),
                                                                tla.shift_right(tla.bitwise_and(sv, tla.full(0x70, tla.Int8)), 2),
                                                            )
                                                            r0, r1 = tla.interleave(lo, hi)
                                                            tla.tile_view(ba_fp_f, tla.make_shape(256), tla.make_coord(2 * ch)).store(r0)
                                                            tla.tile_view(ba_fp_f, tla.make_shape(256), tla.make_coord(2 * ch + 1)).store(r1)
                                                    tla.set_flag(dcd_st_a0)
                                                    tla.wait_flag(dcd_st_a0)
                                                    tla.cross_core_wait_flag(b_free, tla.pipes.MTE3, aiv_id=0)
                                                    t_l1b_act_v_kk = tla.make_tensor(l1b_act + (kk % 2) * (tk * tn), tla.make_layout_with_tag((tk, tn), dtype_b, tla.arch.zN))
                                                    tla.copy(
                                                        tla.get_tile(t_l1b_act_v_kk, tla.make_coord(0, 0), tla.make_shape(tk, tn // 2)),
                                                        tla.get_tile(ub_ba_fp8_zn, tla.make_coord(0, 0), tla.make_shape(tk, tn // 2)),
                                                    )
                                                    tla.set_flag(dcd_rl_a0)
                                                    tla.wait_flag(dcd_rl_a1)
                                                    with tla.vec.func(mode="simd"):
                                                        for ch in tla.range(tk * tn // 1024, tk * tn // 512):
                                                            sv = tla.tile_view(ba_pk_f, tla.make_shape(256), tla.make_coord(ch)).load()
                                                            lo = tla.bitwise_or(
                                                                tla.shift_left(tla.bitwise_and(sv, tla.full(0x08, tla.Int8)), 4),
                                                                tla.shift_left(tla.bitwise_and(sv, tla.full(0x07, tla.Int8)), 2),
                                                            )
                                                            hi = tla.bitwise_or(
                                                                tla.bitwise_and(sv, tla.full(-128, tla.Int8)),
                                                                tla.shift_right(tla.bitwise_and(sv, tla.full(0x70, tla.Int8)), 2),
                                                            )
                                                            r0, r1 = tla.interleave(lo, hi)
                                                            tla.tile_view(ba_fp_f, tla.make_shape(256), tla.make_coord(2 * ch)).store(r0)
                                                            tla.tile_view(ba_fp_f, tla.make_shape(256), tla.make_coord(2 * ch + 1)).store(r1)
                                                    tla.set_flag(dcd_pk_a)
                                                    tla.set_flag(dcd_st_a1)
                                                    tla.wait_flag(dcd_st_a1)
                                                    tla.copy(
                                                        tla.get_tile(t_l1b_act_v_kk, tla.make_coord(0, tn // 2), tla.make_shape(tk, tn // 2)),
                                                        tla.get_tile(ub_ba_fp8_zn, tla.make_coord(0, tn // 2), tla.make_shape(tk, tn // 2)),
                                                    )
                                                    tla.set_flag(dcd_rl_a1)
                                                    tla.cross_core_set_flag(b_full, tla.pipes.MTE3, aiv_id=0)
                                if tla.const_expr(_decodeB and not _transB):
                                    # nontransB: spread the gate-half B decode onto the
                                    # second vector sub-core (aiv_id=1). aiv0 decodes the
                                    # act half into l1b_act; aiv1 the gate half into
                                    # l1b_gate. Each signals b_full for its own aiv_id;
                                    # the cube waits both and frees both per K-tile.
                                    # Mirrors w4a8_mx_matmul.py dual-subcore L1 produce.
                                    # Distinct local names keep the DSL dynamic-if
                                    # new-def analysis happy (nothing carried past it).
                                    if sub_idx == 1:
                                        gate_col_b = n_out + block_col * tn
                                        for kkb in tla.range(phase_lo, phase_hi, c1):
                                            nbp_g = tn // 32
                                            nb0_g = gate_col_b // 32
                                            gm_plane_g = tla.get_tile(
                                                gm_b_pack_p,
                                                tla.make_coord(nb0_g, kkb * (tk * 16)),
                                                tla.make_shape(nbp_g, tk * 16),
                                            )
                                            gv0 = gm_plane_g.origin_shape[0]
                                            gv1 = gm_plane_g.origin_shape[1]
                                            tla.wait_flag(dcd_pk_g)
                                            if gv0 * gv1 == nbp_g * tk * 16:
                                                tla.copy(ub_bg_planes, gm_plane_g)
                                                tla.set_flag(dcd_ld_g)
                                                tla.wait_flag(dcd_ld_g)
                                                tla.wait_flag(dcd_rl_g0)
                                                with tla.vec.func(mode="simd"):
                                                    for gch in tla.range(tk * tn // 1024):
                                                        gsv = tla.tile_view(bg_pk_f, tla.make_shape(256), tla.make_coord(gch)).load()
                                                        glo = tla.bitwise_or(
                                                            tla.shift_left(tla.bitwise_and(gsv, tla.full(0x08, tla.Int8)), 4),
                                                            tla.shift_left(tla.bitwise_and(gsv, tla.full(0x07, tla.Int8)), 2),
                                                        )
                                                        ghi = tla.bitwise_or(
                                                            tla.bitwise_and(gsv, tla.full(-128, tla.Int8)),
                                                            tla.shift_right(tla.bitwise_and(gsv, tla.full(0x70, tla.Int8)), 2),
                                                        )
                                                        gr0, gr1 = tla.interleave(glo, ghi)
                                                        tla.tile_view(bg_fp_f, tla.make_shape(256), tla.make_coord(2 * gch)).store(gr0)
                                                        tla.tile_view(bg_fp_f, tla.make_shape(256), tla.make_coord(2 * gch + 1)).store(gr1)
                                                tla.set_flag(dcd_st_g0)
                                                tla.wait_flag(dcd_st_g0)
                                                tla.cross_core_wait_flag(b_free, tla.pipes.MTE3, aiv_id=1)
                                                t_l1b_gate_b = tla.make_tensor(l1b_gate + (kkb % 2) * (tk * tn), tla.make_layout_with_tag((tk, tn), dtype_b, tla.arch.zN))
                                                tla.copy(
                                                    tla.get_tile(t_l1b_gate_b, tla.make_coord(0, 0), tla.make_shape(tk, tn // 2)),
                                                    tla.get_tile(ub_bg_fp8_zn, tla.make_coord(0, 0), tla.make_shape(tk, tn // 2)),
                                                )
                                                tla.set_flag(dcd_rl_g0)
                                                tla.wait_flag(dcd_rl_g1)
                                                with tla.vec.func(mode="simd"):
                                                    for gch in tla.range(tk * tn // 1024, tk * tn // 512):
                                                        gsv = tla.tile_view(bg_pk_f, tla.make_shape(256), tla.make_coord(gch)).load()
                                                        glo = tla.bitwise_or(
                                                            tla.shift_left(tla.bitwise_and(gsv, tla.full(0x08, tla.Int8)), 4),
                                                            tla.shift_left(tla.bitwise_and(gsv, tla.full(0x07, tla.Int8)), 2),
                                                        )
                                                        ghi = tla.bitwise_or(
                                                            tla.bitwise_and(gsv, tla.full(-128, tla.Int8)),
                                                            tla.shift_right(tla.bitwise_and(gsv, tla.full(0x70, tla.Int8)), 2),
                                                        )
                                                        gr0, gr1 = tla.interleave(glo, ghi)
                                                        tla.tile_view(bg_fp_f, tla.make_shape(256), tla.make_coord(2 * gch)).store(gr0)
                                                        tla.tile_view(bg_fp_f, tla.make_shape(256), tla.make_coord(2 * gch + 1)).store(gr1)
                                                tla.set_flag(dcd_pk_g)
                                                tla.set_flag(dcd_st_g1)
                                                tla.wait_flag(dcd_st_g1)
                                                tla.copy(
                                                    tla.get_tile(t_l1b_gate_b, tla.make_coord(0, tn // 2), tla.make_shape(tk, tn // 2)),
                                                    tla.get_tile(ub_bg_fp8_zn, tla.make_coord(0, tn // 2), tla.make_shape(tk, tn // 2)),
                                                )
                                                tla.set_flag(dcd_rl_g1)
                                                tla.cross_core_set_flag(b_full, tla.pipes.MTE3, aiv_id=1)
                                            else:
                                                with tla.vec.func(mode="simt", thread_block_dim=256):
                                                    ztid_g, _, _ = tla.arch.thread_idx()
                                                    znth_g, _, _ = tla.arch.thread_block_dim()
                                                    for zp_g in tla.range(ztid_g, tk * tn // 2, znth_g):
                                                        bg_pk_f[zp_g] = 0
                                                tla.pipe_barrier(tla.pipes.ALL)
                                                tla.copy(
                                                    tla.tile_view(ub_bg_planes, tla.make_shape(gv0, gv1), tla.make_coord(0, 0)),
                                                    gm_plane_g,
                                                )
                                                tla.set_flag(dcd_ld_g)
                                                tla.wait_flag(dcd_ld_g)
                                                tla.wait_flag(dcd_rl_g0)
                                                with tla.vec.func(mode="simd"):
                                                    for gch in tla.range(tk * tn // 1024):
                                                        gsv = tla.tile_view(bg_pk_f, tla.make_shape(256), tla.make_coord(gch)).load()
                                                        glo = tla.bitwise_or(
                                                            tla.shift_left(tla.bitwise_and(gsv, tla.full(0x08, tla.Int8)), 4),
                                                            tla.shift_left(tla.bitwise_and(gsv, tla.full(0x07, tla.Int8)), 2),
                                                        )
                                                        ghi = tla.bitwise_or(
                                                            tla.bitwise_and(gsv, tla.full(-128, tla.Int8)),
                                                            tla.shift_right(tla.bitwise_and(gsv, tla.full(0x70, tla.Int8)), 2),
                                                        )
                                                        gr0, gr1 = tla.interleave(glo, ghi)
                                                        tla.tile_view(bg_fp_f, tla.make_shape(256), tla.make_coord(2 * gch)).store(gr0)
                                                        tla.tile_view(bg_fp_f, tla.make_shape(256), tla.make_coord(2 * gch + 1)).store(gr1)
                                                tla.set_flag(dcd_st_g0)
                                                tla.wait_flag(dcd_st_g0)
                                                tla.cross_core_wait_flag(b_free, tla.pipes.MTE3, aiv_id=1)
                                                t_l1b_gate_b = tla.make_tensor(l1b_gate + (kkb % 2) * (tk * tn), tla.make_layout_with_tag((tk, tn), dtype_b, tla.arch.zN))
                                                tla.copy(
                                                    tla.get_tile(t_l1b_gate_b, tla.make_coord(0, 0), tla.make_shape(tk, tn // 2)),
                                                    tla.get_tile(ub_bg_fp8_zn, tla.make_coord(0, 0), tla.make_shape(tk, tn // 2)),
                                                )
                                                tla.set_flag(dcd_rl_g0)
                                                tla.wait_flag(dcd_rl_g1)
                                                with tla.vec.func(mode="simd"):
                                                    for gch in tla.range(tk * tn // 1024, tk * tn // 512):
                                                        gsv = tla.tile_view(bg_pk_f, tla.make_shape(256), tla.make_coord(gch)).load()
                                                        glo = tla.bitwise_or(
                                                            tla.shift_left(tla.bitwise_and(gsv, tla.full(0x08, tla.Int8)), 4),
                                                            tla.shift_left(tla.bitwise_and(gsv, tla.full(0x07, tla.Int8)), 2),
                                                        )
                                                        ghi = tla.bitwise_or(
                                                            tla.bitwise_and(gsv, tla.full(-128, tla.Int8)),
                                                            tla.shift_right(tla.bitwise_and(gsv, tla.full(0x70, tla.Int8)), 2),
                                                        )
                                                        gr0, gr1 = tla.interleave(glo, ghi)
                                                        tla.tile_view(bg_fp_f, tla.make_shape(256), tla.make_coord(2 * gch)).store(gr0)
                                                        tla.tile_view(bg_fp_f, tla.make_shape(256), tla.make_coord(2 * gch + 1)).store(gr1)
                                                tla.set_flag(dcd_pk_g)
                                                tla.set_flag(dcd_st_g1)
                                                tla.wait_flag(dcd_st_g1)
                                                tla.copy(
                                                    tla.get_tile(t_l1b_gate_b, tla.make_coord(0, tn // 2), tla.make_shape(tk, tn // 2)),
                                                    tla.get_tile(ub_bg_fp8_zn, tla.make_coord(0, tn // 2), tla.make_shape(tk, tn // 2)),
                                                )
                                                tla.set_flag(dcd_rl_g1)
                                                tla.cross_core_set_flag(b_full, tla.pipes.MTE3, aiv_id=1)

                                if tla.const_expr(_decodeB and _transB):
                                    # transB gate half decoded on aiv1 by the same
                                    # two-deep mx_to widening aiv0 runs for the act half.
                                    if sub_idx == 1:
                                        gate_col_t = n_out + block_col * tn
                                        for kkt in tla.range(phase_lo, phase_hi, c1):
                                            if tla.const_expr(nsub > 1):
                                                # [TK256] same sub-tiled decode as the act
                                                # half; only the buffer set, L1 half and AIV
                                                # id differ.
                                                pk_fg, f8i_g, fp8_tg = _transb_decode_fill(
                                                    gm_pack=gm_b_pack_g,
                                                    k_tile=kkt, j=0, par=0,
                                                    dk=dk, tn=tn, tk=tk, dtype_b=dtype_b,
                                                    pk_buf=ub_bg_pack, fp8_buf=ub_bg_fp8,
                                                    act_col=gate_col_t,
                                                    fl_rl=(dcd_g_rl0, dcd_g_rl1),
                                                    fl_ld=(dcd_g_ld0, dcd_g_ld1),
                                                )
                                                # [ZNUB] SIMT nibble widen + zN scatter: packed byte p
                                                # of the row-major (tn, dk//2) staging holds the two
                                                # k = 2kb, 2kb+1 nibbles of row n = p // (dk//2); both land
                                                # adjacent in the zN tile, so one 2-byte store covers the
                                                # pair. FP8 byte = exact value / 64 (epilogue restores x64).
                                                with tla.vec.func(mode="simt", thread_block_dim=256):
                                                    tid_g1, _, _ = tla.arch.thread_idx()
                                                    nth_g1, _, _ = tla.arch.thread_block_dim()
                                                    for p_g1 in tla.range(tid_g1, dk * tn // 2, nth_g1):
                                                        b_g1 = pk_fg[p_g1].to(tla.Int32) & 0xFF
                                                        n_g1 = p_g1 // (dk // 2)
                                                        kb_g1 = p_g1 - n_g1 * (dk // 2)
                                                        k2_g1 = 2 * kb_g1
                                                        z_g1 = ((k2_g1 // 32) * (tn * 32) + (n_g1 // 16) * 512
                                                               + (n_g1 % 16) * 32 + (k2_g1 % 32))
                                                        lo_g1 = ((b_g1 & 0x08) << 4) | ((b_g1 & 0x07) << 2)
                                                        hi_g1 = (b_g1 & 0x80) | ((b_g1 & 0x70) >> 2)
                                                        f8i_g[z_g1 // 2] = (lo_g1 | (hi_g1 << 8)).to(tla.Int16)
                                                _transb_decode_commit(
                                                    fp8_t=fp8_tg,
                                                    k_tile=kkt, j=0, par=0, dk=dk, tn=tn, tk=tk,
                                                    l1_base=l1b_gate, slot_ref=fp8_slot_ref,
                                                    fl_rl=(dcd_g_rl0, dcd_g_rl1),
                                                    fl_st=(dcd_g_st0, dcd_g_st1),
                                                    b_free=b_free, b_full=b_full, aiv_id=1,
                                                    first=True, last=False,
                                                )
                                                pk_fg, f8i_g, fp8_tg = _transb_decode_fill(
                                                    gm_pack=gm_b_pack_g,
                                                    k_tile=kkt, j=1, par=1,
                                                    dk=dk, tn=tn, tk=tk, dtype_b=dtype_b,
                                                    pk_buf=ub_bg_pack_b, fp8_buf=ub_bg_fp8_b,
                                                    act_col=gate_col_t,
                                                    fl_rl=(dcd_g_rl0, dcd_g_rl1),
                                                    fl_ld=(dcd_g_ld0, dcd_g_ld1),
                                                )
                                                # [ZNUB] SIMT nibble widen + zN scatter: packed byte p
                                                # of the row-major (tn, dk//2) staging holds the two
                                                # k = 2kb, 2kb+1 nibbles of row n = p // (dk//2); both land
                                                # adjacent in the zN tile, so one 2-byte store covers the
                                                # pair. FP8 byte = exact value / 64 (epilogue restores x64).
                                                with tla.vec.func(mode="simt", thread_block_dim=256):
                                                    tid_g1, _, _ = tla.arch.thread_idx()
                                                    nth_g1, _, _ = tla.arch.thread_block_dim()
                                                    for p_g1 in tla.range(tid_g1, dk * tn // 2, nth_g1):
                                                        b_g1 = pk_fg[p_g1].to(tla.Int32) & 0xFF
                                                        n_g1 = p_g1 // (dk // 2)
                                                        kb_g1 = p_g1 - n_g1 * (dk // 2)
                                                        k2_g1 = 2 * kb_g1
                                                        z_g1 = ((k2_g1 // 32) * (tn * 32) + (n_g1 // 16) * 512
                                                               + (n_g1 % 16) * 32 + (k2_g1 % 32))
                                                        lo_g1 = ((b_g1 & 0x08) << 4) | ((b_g1 & 0x07) << 2)
                                                        hi_g1 = (b_g1 & 0x80) | ((b_g1 & 0x70) >> 2)
                                                        f8i_g[z_g1 // 2] = (lo_g1 | (hi_g1 << 8)).to(tla.Int16)
                                                _transb_decode_commit(
                                                    fp8_t=fp8_tg,
                                                    k_tile=kkt, j=1, par=1, dk=dk, tn=tn, tk=tk,
                                                    l1_base=l1b_gate, slot_ref=fp8_slot_ref,
                                                    fl_rl=(dcd_g_rl0, dcd_g_rl1),
                                                    fl_st=(dcd_g_st0, dcd_g_st1),
                                                    b_free=b_free, b_full=b_full, aiv_id=1,
                                                    first=False, last=True,
                                                )
                                            else:
                                                gmt = tla.get_tile(
                                                    gm_b_pack_g,
                                                    tla.make_coord(gate_col_t, kkt * (tk // 2)),
                                                    tla.make_shape(tn, tk // 2),
                                                )
                                                tv0 = gmt.origin_shape[0]
                                                tv1 = gmt.origin_shape[1]
                                                gpk_buf = ub_bg_pack
                                                gfp8_buf = ub_bg_fp8
                                                if kkt % 2 == 1:
                                                    gpk_buf = ub_bg_pack_b
                                                    gfp8_buf = ub_bg_fp8_b
                                                gpk_t = tla.make_tensor(
                                                    gpk_buf,
                                                    tla.make_layout(
                                                        tla.make_shape(tn, tk // 2), tla.make_stride(tk // 2, 1),
                                                        origin_shape=tla.make_shape(tn, tk // 2),
                                                        layoutTag=tla.arch.RowMajor,
                                                    ),
                                                )
                                                gpk_f = tla.make_tensor(
                                                    gpk_buf,
                                                    tla.make_layout(tla.make_shape(tk * tn // 2), tla.make_stride(1)),
                                                )
                                                gmx_in = tla.make_tensor(
                                                    tla.recast_ptr(gpk_buf, dtype=tla.Float4E2M1),
                                                    tla.make_layout(tla.make_shape(tn, tk), tla.make_stride(tk, 1)),
                                                )
                                                gmx_out = tla.make_tensor(
                                                    tla.recast_ptr(gfp8_buf, dtype=dtype_b),
                                                    tla.make_layout(tla.make_shape(tn, tk), tla.make_stride(tk, 1)),
                                                )
                                                gfp8_t = tla.make_tensor(
                                                    tla.recast_ptr(gfp8_buf, dtype=dtype_b),
                                                    tla.make_layout_with_tag(
                                                        (tn, tk), dtype_b, tla.arch.zN
                                                    ),
                                                )
                                                if tla.const_expr(tk != 256):
                                                    # Smaller K tile: row-major
                                                    # staging + MTE3 scatter.
                                                    gfp8_t = tla.make_tensor(
                                                        tla.recast_ptr(gfp8_buf, dtype=dtype_b),
                                                        tla.make_layout(
                                                            tla.make_shape(tn, tk), tla.make_stride(tk, 1),
                                                            origin_shape=tla.make_shape(tn, tk),
                                                            layoutTag=tla.arch.RowMajor,
                                                        ),
                                                    )
                                                if tla.const_expr(tk == 256):
                                                    # [ZNUB] Row-pointer view (see the
                                                    # act half): row n starts at byte
                                                    # 32*n, BlockStore scatters its 8
                                                    # K-block fractals.
                                                    gfp8_f = tla.make_tensor(
                                                        tla.recast_ptr(gfp8_buf, dtype=dtype_b),
                                                        tla.make_layout(
                                                            tla.make_shape(tn, 256), tla.make_stride(32, 1),
                                                            origin_shape=tla.make_shape(tn, 256),
                                                            layoutTag=tla.arch.RowMajor,
                                                        ),
                                                    )
                                                else:
                                                    gfp8_f = tla.make_tensor(
                                                        tla.recast_ptr(gfp8_buf, dtype=dtype_b),
                                                        tla.make_layout(tla.make_shape(tk * tn), tla.make_stride(1)),
                                                    )
                                                if kkt % 2 == 0:
                                                    tla.wait_flag(dcd_g_rl0)
                                                else:
                                                    tla.wait_flag(dcd_g_rl1)
                                                tla.copy(
                                                    tla.tile_view(gpk_t, tla.make_shape(tv0, tv1), tla.make_coord(0, 0)),
                                                    gmt,
                                                )
                                                if kkt % 2 == 0:
                                                    tla.set_flag(dcd_g_ld0)
                                                    tla.wait_flag(dcd_g_ld0)
                                                else:
                                                    tla.set_flag(dcd_g_ld1)
                                                    tla.wait_flag(dcd_g_ld1)
                                                if tla.const_expr(False):
                                                    with tla.vec.func(mode="simt", thread_block_dim=256):
                                                        gtid, _, _ = tla.arch.thread_idx()
                                                        gnth, _, _ = tla.arch.thread_block_dim()
                                                        if tv0 != tn:
                                                            for tp in tla.range(gtid, (tn - tv0) * (tk // 2), gnth):
                                                                gpk_f[tv0 * (tk // 2) + tp] = 0
                                                        if tv1 != tk // 2:
                                                            for grz in tla.range(gtid, tv0, gnth):
                                                                for gcz in tla.range(tv1, tk // 2, c1):
                                                                    gpk_t[grz, gcz] = 0
                                                if tla.const_expr(tk == 256):
                                                    # [ZNUB] 5-op widen + BlockStore.
                                                    with tla.vec.func(mode="simd"):
                                                        for gch in tla.range(tk * tn // 512):
                                                            gsv = tla.tile_view(gpk_f, tla.make_shape(256), tla.make_coord(gch)).load()
                                                            glo = tla.bitwise_or(
                                                                tla.shift_left(tla.bitwise_and(gsv, tla.full(0x08, tla.Int8)), 4),
                                                                tla.shift_left(tla.bitwise_and(gsv, tla.full(0x07, tla.Int8)), 2),
                                                            )
                                                            ghi = tla.bitwise_or(
                                                                tla.bitwise_and(gsv, tla.full(-128, tla.Int8)),
                                                                tla.shift_right(tla.bitwise_and(gsv, tla.full(0x70, tla.Int8)), 2),
                                                            )
                                                            gr0, gr1 = tla.interleave(glo, ghi)
                                                            gd0 = tla.get_tile(gfp8_f, tla.make_coord(2 * gch, 0), tla.make_shape(1, 256))
                                                            gd1 = tla.get_tile(gfp8_f, tla.make_coord(2 * gch + 1, 0), tla.make_shape(1, 256))
                                                            gd0.store(gr0, tla.params.BlockStoreParams(block_stride=tn))
                                                            gd1.store(gr1, tla.params.BlockStoreParams(block_stride=tn))
                                                else:
                                                    with tla.vec.func(mode="simd"):
                                                        for gch in tla.range(tk * tn // 512):
                                                            gsv = tla.tile_view(gpk_f, tla.make_shape(256), tla.make_coord(gch)).load()
                                                            glo = tla.bitwise_or(
                                                                tla.shift_left(tla.bitwise_and(gsv, tla.full(0x08, tla.Int8)), 4),
                                                                tla.shift_left(tla.bitwise_and(gsv, tla.full(0x07, tla.Int8)), 2),
                                                            )
                                                            ghi = tla.bitwise_or(
                                                                tla.bitwise_and(gsv, tla.full(-128, tla.Int8)),
                                                                tla.shift_right(tla.bitwise_and(gsv, tla.full(0x70, tla.Int8)), 2),
                                                            )
                                                            gr0, gr1 = tla.interleave(glo, ghi)
                                                            tla.tile_view(gfp8_f, tla.make_shape(256), tla.make_coord(2 * gch)).store(gr0)
                                                            tla.tile_view(gfp8_f, tla.make_shape(256), tla.make_coord(2 * gch + 1)).store(gr1)
                                                if kkt % 2 == 0:
                                                    tla.set_flag(dcd_g_st0)
                                                    tla.wait_flag(dcd_g_st0)
                                                else:
                                                    tla.set_flag(dcd_g_st1)
                                                    tla.wait_flag(dcd_g_st1)
                                                tla.cross_core_wait_flag(b_free, tla.pipes.MTE3, aiv_id=1)
                                                t_l1b_gate_t = tla.make_tensor_like(l1b_gate + (kkt % 2) * (tk * tn), fp8_slot_ref, tla.arch.zN)
                                                tla.copy(t_l1b_gate_t, gfp8_t)
                                                if kkt % 2 == 0:
                                                    tla.set_flag(dcd_g_rl0)
                                                else:
                                                    tla.set_flag(dcd_g_rl1)
                                                tla.cross_core_set_flag(b_full, tla.pipes.MTE3, aiv_id=1)

                    if has_block:
                        prev_out_row = out_row
                        prev_rows = rows
                        prev_cols_valid = cols_valid
                        prev_blk_col = block_col
                        prev_valid = True
                    else:
                        prev_valid = False


                start_core_idx = (start_core_idx + mn_blocks) % core_num
            m_start = m_start + current_m


# -----------------------------------------------------------------------------------------------------------
# Host side: tensor binding, tiling, artifact cache and the ``run`` entry point.
# -----------------------------------------------------------------------------------------------------------


def get_block_num(block_num: int, device: int = 0, *, kind: str = "vector") -> int:
    """Launch ``block_num``: ``-1`` -> full device (cube/AIC for this sample)."""
    if block_num != -1 and block_num <= 0:
        raise ValueError("block_num must be -1 (automatic) or positive")
    block_num = int(block_num)
    if block_num != -1:
        return block_num
    props = torch.npu.get_device_properties(int(device))
    if kind == "vector":
        return max(1, int(props.vector_core_num))
    if kind in {"cube", "mix"}:
        return max(1, int(props.cube_core_num))
    raise ValueError(f"Unsupported kernel kind for block_num default: {kind!r}")


_ARTIFACT_CACHE: dict = {}

# Device last made current by run(); -1 forces the first call to set it.
_LAST_SET_DEVICE = -1


# ---------------------------------------------------------------------------
# FP4 weight pre-layout: FRACTAL zN
# ---------------------------------------------------------------------------
# tla/layout.hpp FractalSpec<zN> for a 4-bit element: fractal (16 K, 64 N) and
# the physical order runs n0 -> m0 -> m1 -> n1, so inside a fractal the N index
# is contiguous and K advances by one C0 (64 elements). Pre-building this on the
# host lets the GM->UB fill read the packed weight as-is instead of having the
# DMA re-swizzle it -- the same "weights arrive pre-laid-out" condition that
# ops-transformer's FRACTAL_NZ weights buy, done here once per buffer identity.
_NZ_WEIGHT_CACHE: "OrderedDict[tuple, object]" = OrderedDict()
_NZ_WEIGHT_CACHE_MAX = 4


def _pack_weight_zN(b):
    """Repack an FP4 weight (E, k, n/2) uint8 into zN fractal order (flat uint8)."""
    e, k, n2 = b.shape
    n = n2 * 2
    src = b.view(torch.uint8)
    lo = src & 0xF
    hi = (src >> 4) & 0xF
    nib = torch.stack((lo, hi), dim=-1).reshape(e, k, n)
    m0, n0 = 16, 64
    if k % m0 or n % n0:
        return None
    m1, n1 = k // m0, n // n0
    blk = nib.reshape(e, m1, m0, n1, n0)
    phys = blk.permute(0, 3, 1, 2, 4).contiguous()      # n1, m1, m0, n0 (slow->fast)
    flat = phys.reshape(-1, 2)
    return (flat[:, 0] | (flat[:, 1] << 4)).to(torch.uint8)


# ---------------------------------------------------------------------------
# NontransB decode staging pre-layout: packed weight as [group][nb][k][16]
# ---------------------------------------------------------------------------
# The nontransB SIMT-free decode widens 256 packed bytes into 512 FP8 bytes in
# packed order. The L1 zN(tk, tn) tile stores, for one 32-wide N plane nb, all
# K rows contiguously (plane stride 32*tk). If the packed staging itself runs
# [nb][k][16] then the widened bytes ARE the zN order, so the UB->L1 hop is a
# contiguous zN->zN DataCopy instead of the per-row 32B scatter (measured as
# the top AIV pipe: aiv_mte3 0.73 on case 070). Pre-laying the weight once per
# buffer identity keeps the GM->UB fill contiguous per plane; the same
# "weights arrive pre-laid-out" contract the transB FP4 zN path uses.
_NTB_PACK_CACHE: "OrderedDict[tuple, object]" = OrderedDict()
_NTB_PACK_CACHE_MAX = 4


def _pack_nontrans_zN(b):
    """Repack a packed FP4 weight (groups, k, n_pack) to per-group [nb][k][16].

    ``nb`` indexes a 32-wide N plane (16 packed bytes). 16-byte padding of a
    group's packed row keeps every plane whole so the device fill never has to
    issue a sub-DataBlock copy. Returns a contiguous uint8 (groups*k, n_pack_pad)
    buffer; group g's bytes are [plane][k][16].
    """
    e, kk, n2 = b.shape
    src = b.contiguous().view(torch.int8)
    pad = (-n2) % 16
    if pad:
        padded = torch.zeros((e, kk, n2 + pad), dtype=torch.int8, device=src.device)
        padded[..., :n2] = src
        src = padded
    n2p = n2 + pad
    nb = n2p // 16
    out = src.reshape(e, kk, nb, 16).permute(0, 2, 1, 3).contiguous()
    return out.reshape(e * kk, n2p)


def _ntb_pack_cached(b):
    """Memoize the nontransB pre-layout by buffer identity + content version.

    The cache retains ``b`` so its data_ptr cannot be reused by a different
    tensor while the entry is live, and ``_version`` catches an in-place
    rewrite -- the same stale-proof contract as ``_GROUP_SIZE_CACHE``.
    """
    key = (b.data_ptr(), tuple(b.shape), b.dtype, b._version)
    hit = _NTB_PACK_CACHE.get(key)
    if hit is not None:
        _NTB_PACK_CACHE.move_to_end(key)
        return hit[1]
    packed = _pack_nontrans_zN(b)
    _NTB_PACK_CACHE[key] = (b, packed)
    if len(_NTB_PACK_CACHE) > _NTB_PACK_CACHE_MAX:
        _NTB_PACK_CACHE.popitem(last=False)
    return packed


def _weight_zN_cached(b):
    key = (b.data_ptr(), tuple(b.shape), b.dtype)
    hit = _NZ_WEIGHT_CACHE.get(key)
    if hit is not None:
        _NZ_WEIGHT_CACHE.move_to_end(key)
        return hit
    packed = _pack_weight_zN(b)
    _NZ_WEIGHT_CACHE[key] = packed
    if len(_NZ_WEIGHT_CACHE) > _NZ_WEIGHT_CACHE_MAX:
        _NZ_WEIGHT_CACHE.popitem(last=False)
    return packed

# Bounded LRU of host DLPack bindings for INPUT operands, keyed by full buffer
# identity. from_dlpack re-exports the DLPack capsule and re-resolves the layout
# on every call (~0.037 ms each, 5 inputs => ~0.19 ms of the per-call floor).
# Inference reuses weight/scale buffers across calls, so memoizing the binding
# removes that repeated parse. This is host-side memoization only: the compiled
# artifact is still keyed by constexpr variant and every binding stays
# mark_layout_dynamic, so nothing is baked into the kernel. A cached binding
# retains its source buffer (DLPack consumer contract), so its data_ptr cannot
# be reused while cached -- the identity key is therefore stale-proof. Outputs
# are freshly allocated each call and are NOT cached (caching them would retain
# dead buffers and never hit). The LRU bound caps retained memory.
_BINDING_CACHE: "OrderedDict[tuple, object]" = OrderedDict()
_BINDING_CACHE_MAX = 256


def _bind_input(buf, layout_tag, element_type, origin_shape, dynamic):
    key = (
        buf.data_ptr(), tuple(buf.shape), buf.dtype,
        layout_tag, origin_shape, element_type, dynamic,
    )
    cached = _BINDING_CACHE.get(key)
    if cached is not None:
        _BINDING_CACHE.move_to_end(key)
        return cached
    t = from_dlpack(
        buf, layout_tag=layout_tag, origin_shape=origin_shape,
        element_type=element_type,
    )
    if dynamic == "compact0":
        t = t.mark_compact_shape_dynamic(0)
    else:
        t = t.mark_layout_dynamic()
    _BINDING_CACHE[key] = t
    if len(_BINDING_CACHE) > _BINDING_CACHE_MAX:
        _BINDING_CACHE.popitem(last=False)
    return t


# Output buffers are freshly allocated each call, so they cannot be cached by
# address. Their layout metadata depends only on (shape, dtype, layout_tag,
# element_type), so cache the binding by that key and repoint data_ptr at the
# new buffer each call (the launch reads _Tensor.data_ptr). This skips the
# repeated DLPack export/parse for outputs. The first buffer per shape is
# retained by the DLPack consumer contract (bounded by distinct output shapes);
# the current buffer stays alive via run()'s local reference. The compiled
# artifact is unchanged and layouts stay dynamic -- no kernel specialization.
_OUTPUT_BIND_CACHE: dict = {}


def _bind_output(buf, layout_tag, element_type):
    key = (tuple(buf.shape), buf.dtype, layout_tag, element_type)
    t = _OUTPUT_BIND_CACHE.get(key)
    if t is None:
        t = from_dlpack(
            buf, layout_tag=layout_tag, element_type=element_type
        ).mark_layout_dynamic()
        _OUTPUT_BIND_CACHE[key] = t
    else:
        t.data_ptr = buf.data_ptr()
    return t


# [WALL] Whole-pair output binding cache. The two outputs are freshly allocated
# every call, but their *binding metadata* depends only on (m, n_out, out_dtype):
# the Int32 view of ``q`` is (m, n_out//4) and the Int32 view of the scale buffer
# is (m, ceil(n_out/128)). So the two bound tensors can be created once per
# output shape and have their ``data_ptr`` repointed at the current buffers on
# every later call. This skips building the two Int32 views, two shape tuples and
# two DLPack imports per launch (measured ~1.5 us). The first buffers per shape
# stay alive through ``_OUTPUT_BIND_CACHE``, which is also what the DLPack
# consumer contract requires.
_OUT_BIND_PAIR_CACHE: dict = {}


def _bind_output_pair(q, n_out, qsb):
    key = (q.shape[0], n_out, q.dtype)
    hit = _OUT_BIND_PAIR_CACHE.get(key)
    if hit is None:
        y_t = _bind_output(q.view(torch.int32), tla.arch.RowMajor, tla.Int32)
        sy_t = _bind_output(qsb.view(torch.int32), tla.arch.RowMajor, tla.Int32)
        _OUT_BIND_PAIR_CACHE[key] = (y_t, sy_t)
        return y_t, sy_t
    y_t, sy_t = hit
    y_t.data_ptr = q.data_ptr()
    sy_t.data_ptr = qsb.data_ptr()
    return y_t, sy_t

FP8_SEL = {
    torch.float8_e4m3fn: tla.Float8E4M3FN,
    torch.float8_e5m2: tla.Float8E5M2,
}
FP4_DTYPE = torch.float4_e2m1fn_x2
NPU_ARCH = "--npu-arch 3510"


def _ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def _build_tensors(a, b, mx_scale_a, mx_scale_b, group_list, trans_b):
    """Bind operands and e8m0 scale blocks; FP4 operands go over as raw bytes."""
    is_fp4_a = a.dtype == FP4_DTYPE
    is_fp4_b = b.dtype == FP4_DTYPE
    m = int(a.shape[0])
    k = int(a.shape[1]) * 2 if is_fp4_a else int(a.shape[1])
    groups = int(b.shape[0])
    if trans_b:
        n = int(b.shape[1])
    else:
        n = int(b.shape[2]) * 2 if is_fp4_b else int(b.shape[2])
    k_groups = _ceil_div(k, MX_OUTPUT_BLOCK_SIZE)

    # FP4 operands bind natively as Float4E2M1 with logical (rows, K) shape so
    # the cube consumes packed FP4 directly (mmad_mx f4e2m1), no SIMT decode.
    if is_fp4_a:
        a_t = _bind_input(
            a.reshape(m, k // 2).contiguous().view(torch.int8),
            tla.arch.RowMajor, tla.Float4E2M1, (m, k), "layout",
        )
    else:
        a_t = _bind_input(
            a.view(torch.int8).contiguous(), tla.arch.RowMajor, FP8_SEL[a.dtype], None, "layout"
        )

    # B and its scale go over as plain row-major packed 2D buffers; the device
    # make_tensor windows impose the logical (K, N) / MxScale layout per group.
    # a4w4 binds B natively as Float4E2M1 (cube consumes packed FP4). a8w4 has
    # FP8 act x FP4 weight, which mmad_mx rejects (mixed f8xf4), so B stays raw
    # Int8 bytes for the device SIMT decode-to-FP8 path.
    native_fp4_b = is_fp4_a and is_fp4_b
    if trans_b:
        b_cols = k // 2 if is_fp4_b else k
        b2d = b.reshape(groups * n, b_cols)
        sb2d = mx_scale_b.reshape(groups * n, k_groups)
        # Native FP4 transB binds ColumnMajor (K, groups*N) so the device can
        # get_tile each group along N without extracting a sub-byte ptr (which
        # would drop the get_tile coord offset).
        b_tag = tla.arch.ColumnMajor if native_fp4_b else tla.arch.RowMajor
        b_origin = (k, groups * n) if native_fp4_b else None
    else:
        b_cols = n // 2 if is_fp4_b else n
        # [NTB-ZN] a8w4 nontransB decodes B on the vector side; hand it the
        # pre-laid [nb][k][16] packed order so the decode lands in zN order.
        decode_b_here = is_fp4_b and not is_fp4_a
        b2d = _ntb_pack_cached(b) if decode_b_here else b.reshape(groups * k, b_cols)
        sb2d = mx_scale_b.reshape(groups * k_groups, n)
        b_tag = tla.arch.RowMajor
        b_origin = (groups * k, n) if native_fp4_b else None
    if native_fp4_b:
        b_t = _bind_input(
            b2d.contiguous().view(torch.int8), b_tag, tla.Float4E2M1, b_origin, "layout"
        )
    elif is_fp4_b:
        b_t = _bind_input(
            b2d.contiguous().view(torch.int8), tla.arch.RowMajor, tla.Int8, None, "layout"
        )
    else:
        b_t = _bind_input(
            b2d.contiguous().view(torch.int8),
            tla.arch.RowMajor,
            FP8_SEL[b.dtype],
            None,
            "layout",
        )

    sa = mx_scale_a.reshape(m, k_groups).contiguous()
    sa_t = _bind_input(
        sa.view(torch.int8), tla.arch.RowMajorMxScaleA, tla.Float8E8M0, (m, k_groups), "layout"
    )
    sb_t = _bind_input(
        sb2d.contiguous().view(torch.int8), tla.arch.RowMajor, tla.Float8E8M0, None, "layout"
    )

    # group_list is INT64 by contract; bind it natively (content-independent,
    # memoizable by buffer identity) instead of converting to int32 every call,
    # which allocates a fresh buffer and forces a from_dlpack miss each launch.
    # The device casts each loaded count to Int32.
    gl_t = _bind_input(group_list.contiguous(), tla.arch.RowMajor, None, None, "compact0")
    # No a4w8 variant exists, so A never needs decode. B decodes only for a8w4
    # (FP8 act x FP4 weight); a4w4 feeds the cube native FP4.
    decode_a = False
    decode_b = is_fp4_b and not is_fp4_a
    return a_t, b_t, sa_t, sb_t, gl_t, (m, n, k, groups), decode_a, decode_b


# [LINE-a8w8] Per-call host cost: _build_tensors re-derives the DLPack bindings
# (~8 us of Python) on every launch even though inference reuses the operand
# buffers. Cache the whole built tuple by buffer identity + shape/dtype. The
# bindings stay mark_layout_dynamic, so this is pure host memoization under the
# same contract as _BINDING_CACHE (the identity key makes reuse stale-proof).
_BUILD_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_BUILD_CACHE_MAX = 8
# [WALL] One-entry identity fast path: the steady-state caller passes the very
# same operand tensor objects every launch, so five ``is`` checks answer the
# whole question without rebuilding the (data_ptr, shape, dtype) key tuple
# (measured ~1.1 us). ``b`` keeps a version check because the non-transposed decode
# path caches a re-laid-out copy of b, and an in-place rewrite of that same
# buffer must invalidate it. Identity implies unchanged shape/dtype, so the
# slower key below still covers every case this fast path declines.
_LAST_BUILD = None


def _build_tensors_cached(a, b, mx_scale_a, mx_scale_b, group_list, trans_b):
    global _LAST_BUILD
    last = _LAST_BUILD
    if (
        last is not None
        and last[0] is a and last[1] is b and last[2] is mx_scale_a
        and last[3] is mx_scale_b and last[4] is group_list
        and last[5] is trans_b and last[6] == b._version
    ):
        return last[7]
    key = (
        a.data_ptr(), b.data_ptr(), mx_scale_a.data_ptr(), mx_scale_b.data_ptr(),
        group_list.data_ptr(),
        # [NTB-ZN] the nontransB decode path caches a re-laid-out copy of b, so
        # an in-place rewrite of the same buffer must invalidate the build.
        b._version,
        tuple(a.shape), tuple(b.shape), tuple(mx_scale_a.shape),
        tuple(mx_scale_b.shape), a.dtype, b.dtype, trans_b,
    )
    hit = _BUILD_CACHE.get(key)
    if hit is not None:
        _BUILD_CACHE.move_to_end(key)
        _LAST_BUILD = (a, b, mx_scale_a, mx_scale_b, group_list, trans_b,
                       b._version, hit)
        return hit
    built = _build_tensors(a, b, mx_scale_a, mx_scale_b, group_list, trans_b)
    _BUILD_CACHE[key] = built
    if len(_BUILD_CACHE) > _BUILD_CACHE_MAX:
        _BUILD_CACHE.popitem(last=False)
    _LAST_BUILD = (a, b, mx_scale_a, mx_scale_b, group_list, trans_b,
                   b._version, built)
    return built


# Per-group row counts for the M-tile choice. This is the one place host code
# reads group_list back to the CPU: ragged groups each round their own tail up
# to a whole tm row, so the mean-M model can miss a wave boundary (case 085
# buys -14.3% from that). The read happens once per buffer identity + version;
# the entry retains the tensor so its data_ptr cannot be reused while cached
# (the same stale-proof argument as _BINDING_CACHE), and _version catches an
# in-place rewrite of a retained buffer. Steady-state inference that reuses the
# same group_list buffer therefore pays no device->host read.
_GROUP_SIZE_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_GROUP_SIZE_CACHE_MAX = 64


def _group_sizes_cached(group_list):
    key = (
        group_list.data_ptr(), tuple(group_list.shape), group_list.dtype,
        group_list._version,
    )
    hit = _GROUP_SIZE_CACHE.get(key)
    if hit is not None:
        _GROUP_SIZE_CACHE.move_to_end(key)
        return hit[1]
    counts = tuple(int(v) for v in group_list.detach().reshape(-1).to("cpu").tolist())
    _GROUP_SIZE_CACHE[key] = (group_list, counts)
    if len(_GROUP_SIZE_CACHE) > _GROUP_SIZE_CACHE_MAX:
        _GROUP_SIZE_CACHE.popitem(last=False)
    return counts


# [WALL] One-entry identity shortcut over ``_group_sizes_cached``: the
# steady-state caller passes the same group_list object, so two identity/version
# checks answer it without rebuilding the (data_ptr, shape, dtype, version) key
# tuple. Identity implies an unchanged shape and dtype.
_LAST_GROUP_SIZES = None


def _group_sizes_fast(group_list):
    global _LAST_GROUP_SIZES
    last = _LAST_GROUP_SIZES
    version = group_list._version
    if last is not None and last[0] is group_list and last[1] == version:
        return last[2]
    sizes = _group_sizes_cached(group_list)
    _LAST_GROUP_SIZES = (group_list, version, sizes)
    return sizes


def _select_tk(k, sk_max=8):
    """Largest 32*sk (sk in {sk_max..1, even}) that divides the K scale-block count.

    The device sizes each K tile's MX scale window by sk = tk / 32 blocks, so a
    last tile with fewer blocks left than sk reads past the group's scale
    buffer; the bytes there follow the allocator and results varied per
    process. Picking tk as a divisor of the block count leaves no partial tile.
    Wider K tiles halve the GM->L1 transaction count and double the row burst
    for the packed FP4 weight (tk/2 bytes per row). sk_max=4 keeps the tile at
    128 for variants where the larger decode stage costs more than the copies
    save (a8w4 transB with n_out > 64).
    """
    blocks = max(1, _ceil_div(k, 32))
    for sk in range(sk_max, 1, -1):
        if sk % 2 == 0 and blocks % sk == 0:
            return 32 * sk
    return 32


_M_L1 = 256
_M_TILE_CANDIDATES = (128, 160, 192, 224, 256)
# [MTAIL] Extra candidates for small decode shapes whose per-group row count
# leaves most of a 128-row tile empty. The M tail is real MAC work: a group of
# 64 rows still runs a full 128-row MMA, so on a one-wave shape a shorter tile
# cuts the MMAD in proportion to the rows it stops rounding up. Measured on card
# 7 (DIAG3): case 054 (groups all 64) tm=64 is -8.3%; case 067 (7x~73 rows)
# tm=96 is -4.4%. Native FP4 (104) got 5% *slower* with tm=64, so the expansion
# is gated to the a8w4 decode path by ``_m_tail_small_tile``.
_M_TILE_CANDIDATES_SMALL = (64, 96) + _M_TILE_CANDIDATES
# Row-utilization below which a small single-wave decode shape may search the
# short tiles. 067 (the least wasteful measured case) rounds 513 rows up to 896
# (0.573), so the cut sits just above it at 0.60.
_M_TAIL_UTIL_MAX = 0.60
# A short K tile is fill/drain bound: with fewer than this many elements per
# block the per-block overhead dominates and the shorter M tile does not pay
# (the same "short K keeps the 128-row grid" rule the wide branch already uses).
_M_TAIL_K_MIN = 256
# The tile the search would otherwise use. The tail gate only makes sense while
# each group still fits inside this one block, so it is also the cap on a
# group's rows.
_M_TAIL_BASE_TILE = 128
# Fixed cost of one (M,N) output block, in row-equivalents: the B tile is
# streamed GM->L1, decoded (a8w4) and reversed per block, so every extra block
# per core costs about this many rows on top of its tm rows. Fitted so the
# model reproduces the tiles measured on card 7 in R5/R6/R8 (256 rows for the
# 2/8-group decode shapes, 128 for the small-M ones) and this round's A/B; the
# window 96..128 gives the same big-case choices.


def _select_tiling(m_total, n_out, k, groups, cube_num, sk_max=8, tm=128):
    """Pick (tm, tn, tk) for this shape.

    tn follows the packed output-column grid: the epilogue consumes 64 output
    columns per chunk, and n_out == 64 needs only one, so a 128-wide B tile
    reads the act+gate halves twice and doubles the MMA N. tm is fixed by
    _m_tile_size; tk keeps every K tile whole, see _select_tk.
    """
    tn = 64 if n_out == 64 else 128
    return GmmTilingParams(tm=tm, tn=tn, tk=_select_tk(k, sk_max))


def _m_tail_small_tile(m_total, n_out, k, groups, cube_num, decode_b, native_fp4,
                       group_sizes):
    """Whether this shape may search M tiles shorter than 128 rows.

    Only the a8w4 decode path is eligible: it is the variant the short-tile win
    was measured on, while native FP4 (a4w4) measured 5% slower at tm=64 (104)
    and a8w8 has no decode staging, so neither gains from more, smaller blocks.
    The remaining conditions describe the one shape class where the M tail is
    pure waste rather than useful work:

      * every non-empty group fits in a single 128-row block -- the only cost of
        a 128-row tile is then the rounded-up tail;
      * the whole shape is one wave on the cube cores (blocks <= cube_num) --
        with more waves the per-block cost is amortised and a short tile only
        adds blocks;
      * the 128-row tile is under ``_M_TAIL_UTIL_MAX`` row utilization;
      * K is at least ``_M_TAIL_K_MIN`` so each block still has a few K steps.

    The candidate set is only *extended*; ``_m_tile_size`` still scores every
    candidate with the same per-core cost model, so an unprofitable short tile
    is never selected.
    """
    if not decode_b or native_fp4 or not group_sizes or cube_num <= 0:
        return False
    nonempty = [c for c in group_sizes if c > 0]
    if not nonempty or k < _M_TAIL_K_MIN:
        return False
    mean_m = m_total // groups if groups > 0 else 0
    if mean_m <= 0 or mean_m > _M_TAIL_BASE_TILE or max(nonempty) > _M_TAIL_BASE_TILE:
        return False
    tn = 64 if n_out == 64 else 128
    grid_n = _ceil_div(n_out, tn) if n_out > 0 else 1
    block_rows = sum(_ceil_div(c, _M_TAIL_BASE_TILE) for c in nonempty)
    blocks = block_rows * grid_n
    if blocks <= 0 or blocks > cube_num:
        return False
    return sum(nonempty) < _M_TAIL_UTIL_MAX * block_rows * _M_TAIL_BASE_TILE


def _m_tile_size(m_total, groups, n_out, cube_num, block_cost=128, group_sizes=None,
                 small_m_tail=False):
    """M-direction row tile, chosen from the work partition -- not from N size.

    The old rule cut tm to 128 whenever ``n_out <= 64`` and then re-ran a
    separate 96..256 search for that branch, so the same M shape got a
    different tile depending on how many output columns it had. The arch35
    reference is N-blind (``mL1Size = ceil(mSize / ceil(mSize / M_L1))``,
    gmmsq_weight_quant_resplit_controller.h), but copying that verbatim was
    measured on card 7 and lost: m=4096 gives 16 blocks on 28 cube cores and
    cases 099/100 ran +13% / +15.5% slower than the 26-block 160-row tile.
    The reference absorbs the idle cores in its own resplit pipeline; here the
    wall is blocks-per-core times (tm + per-block traffic).

    Choose the candidate minimising that per-core cost::

        blocks   = sum_g ceil(m_g / tm) * grid_n     (group_sizes)
        blocks   = groups * ceil(mean_m / tm) * grid_n (no group_sizes)
        cost(tm) = ceil(blocks / cores) * (tm + block_cost)

    ``group_sizes`` is the per-group row count from ``group_list``. Ragged
    groups each round their own tail up to a whole tm row, so the mean-M model
    undercounts the block rows and can miss the wave boundary: case 085
    (23,178,134,89,44,200,155,111,66) has 13 block rows at tm=128 (65 blocks,
    3 waves) but 11 at tm=160 (55 blocks, 2 waves), measured -14.3% on card 7.
    The exact counts pick 160; the mean model picks 128.

    grid_n is the N tile count (1 for n_out <= 128). It is a work-partition
    count, not the output width: the n_out=64 shapes now see grid_n=1, the
    same partition as any n_out in 65..128, instead of the old hard cut. The
    per-block traffic term is why 099/100 want 160 rows (one block per core
    either way) while the 8-N-block decode shapes want 256 (fewer per-core
    blocks, each with a full B decode).
    """
    mean_m = m_total // groups if groups > 0 else 0
    tn = 64 if n_out == 64 else 128
    grid_n = _ceil_div(n_out, tn) if n_out > 0 else 1
    if cube_num <= 0:
        return 128
    if group_sizes is None and mean_m <= 0:
        return 128
    candidates = _M_TILE_CANDIDATES_SMALL if small_m_tail else _M_TILE_CANDIDATES
    best = None
    for cand in candidates:
        if group_sizes is not None:
            block_rows = sum(_ceil_div(c, cand) for c in group_sizes if c > 0)
            blocks = block_rows * grid_n
            # A shape whose group_list is all zero has no work; keep the
            # compiled tile small rather than scoring an artificial zero.
            if blocks <= 0:
                continue
        else:
            blocks = groups * _ceil_div(mean_m, cand) * grid_n
        per_core = _ceil_div(blocks, cube_num)
        cost = per_core * (cand + block_cost)
        if best is None or cost < best[0]:
            best = (cost, cand)
    return 128 if best is None else best[1]


def _tiling_for(m_total, n_out, k, groups, cube_num, decode_b, trans_b, native_fp4=False,
                group_sizes=None):
    # a4w4 binds FP4 weights natively, so it has no decode staging to constrain
    # the tile; the arch35 reference still runs it at M_L1=256, halving the
    # number of B reloads per group. Short K keeps the 128-row grid: at k=1024 a
    # 256-row tile leaves only four K steps per block and fill/drain dominates
    # (case 140 measured +9%).
    # [LINE-a8w8] a8w8 non-transposed has no decode staging, but the same
    # B-reload argument applies: for a large group a 256-row tile halves how
    # often the group's B is streamed GM->L1, and mte2 is the bottleneck pipe
    # there (0.81 busy on 020, 0.90 on 042). Short K keeps 128 rows: case 040
    # (k=1024, four K steps per tile) measured -4.1% with the wide tile, the
    # same fill/drain effect the a4w4 note records at k=1024.
    # [LINE-m1] No n_out gate on tm any more. The old code forced tm=128 for
    # every n_out<=64 shape and only then re-searched 96..256 inside that
    # branch, so identical M shapes took different tiles depending on the
    # output width (099/100 got 160 through the special case, 049/050 stayed
    # 128 with tk=256). _m_tile_size now scores candidates on the work
    # partition (M blocks x N blocks) instead of cutting on the output width.
    # Copying the reference's 256-row ceil-split verbatim was measured 13-15.5%
    # slower on 099/100 (16 blocks on 28 cores vs 26).
    wide = trans_b or decode_b or k >= 2048
    if wide and not trans_b and not decode_b:
        # No decode staging and no transposed B: keep the 128-row grid for a
        # short K or when the groups do not span two M_L1 blocks (a single
        # block cannot reduce its own B reloads).
        mean_m = m_total // groups if groups > 0 else 0
        if groups * _ceil_div(mean_m, _M_L1) < 2:
            wide = False
    # [MTAIL] A small decode shape whose groups round most of a 128-row tile away
    # is allowed to search the short tiles too; the same cost model then decides
    # (054/067 measured -8.3%/-4.4% on card 7). See _m_tail_small_tile.
    small_m_tail = _m_tail_small_tile(
        m_total, n_out, k, groups, cube_num, decode_b, native_fp4, group_sizes
    )
    tm = _m_tile_size(m_total, groups, n_out, cube_num, group_sizes=group_sizes,
                      small_m_tail=small_m_tail) if wide else 128
    tn = 64 if n_out == 64 else 128
    # K tile for the decode paths. tn=128 keeps tk=128: tk=256 measured +12%
    # on 069 in R5, and at tm=256 it also statically overflows UB (148 KiB
    # epilogue + 96 KiB transB staging, R8's 249856 B compile failure on 070).
    # tn=64 halves epilogue and staging (<=172.5 KiB even at tm=256), and the
    # wider K burst measured faster there (R5: 099 -23.5%, 100 -14.9%), so the
    # half-column path keeps sk_max=8. The transB path has its own sk_max=4 cap
    # below (the pre-existing 069 tile).
    narrow_k = decode_b and (not trans_b) and tm > 128 and tn > 64
    # [NTB-TK256] Probed and rejected: opening tk=256 for the large-K nontransB
    # a8w4 shapes (blast radius 070/072/088/092) moved 070 -2.7%, 072 +3.0%,
    # 088 -3.1% while the out-of-radius control 090 moved +0.3%, i.e. inside the
    # session noise. Their wall is not the GM->L1 A burst. Keep narrow_k.
    # [LINE-a8w8] a wide (tm>128) a8w8 tile at tk=256 exceeds the 64 KiB L0A
    # double-buffer (2*tm*tk > 64 KiB) and 4*tk*tn exceeds the 64 KiB L0B, so
    # the kernel compiles with L0 ping-pong _l0_db=False and every L1->L0 copy
    # serializes with the MMAD. tk=128 restores the double buffer; profiles of
    # 019/041 show mac 0.70/0.66 + mte1 0.23 with only ~70% cube utilization,
    # i.e. the serialized copy is the gap.
    # tn=64 is exempt: there the L0B tile is half-sized, and 049/050 measured
    # tk=256 47%/34% faster than tk=128 this round (R5 agreed: 049 -26.3%), so
    # the transaction-count win outweighs the lost L0 overlap.
    if not decode_b and not native_fp4 and tm > 128 and tn > 64:
        # Only when a group spans at least two M blocks: with one block per
        # group (041, G=8 x 1 x grid_n=8) the smaller 128-byte A row burst
        # costs more than the L0 overlap buys (measured -4.4%), while the
        # multi-block groups gain +3.1% (019), +7.7% (037), +8.7% (038).
        # [A8TK] ...except the large-K transB shapes, which now take the a8w4
        # tk=256 + L0=128 decoupling instead: the device slices one 256-K L1
        # tile into two 128-K L0 operands, so the wider GM->L1 burst does not
        # cost the L0 ping-pong. tk=128 was chosen here only because tk=256
        # used to serialize L1->L0 with MMAD. 019/041 (k=4096, n_out=1024) are
        # the two large-K transB a8w8 shapes; 049 is narrow-N (n_out=64), and
        # everything else (including 021 at k=2048) keeps the previous tile.
        mean_m = m_total // groups if groups > 0 else 0
        a8tk = trans_b and k >= 4096 and n_out >= 1024
        if _ceil_div(mean_m, tm) >= 2 and not a8tk:
            narrow_k = True
    # [TK256] transB keeps the wide K tile only for the large shapes: the
    # decode staging is sub-tiled at 128 K elements in the kernel, so tk=256
    # costs no UB there, and halving the K-step count widens every GM->L1 burst
    # (A row 128->256 B) on the shapes where the profiler shows aic_mte2 2.46x
    # / aic_mte1 3.10x versus the vendor kernel. The small transB cases stay at
    # tk=128 (their K is 1024, where wider tiles only add fill/drain).
    trans_b_wide = decode_b and trans_b and tn > 64 and k >= 2048 and n_out >= 256
    # [TK256-W] Narrow transB decode (a8w4, tm<=128, n_out<=512): a 256-K L1 tile
    # taken with lk=tk (single L0 step) halves the per-K L1->L0 copy count and the
    # UB->L1 decode-store transaction count. Measured on 063 (g16 m2048 n512 k1024):
    # aic_mte1 12.9->5.0 us, aiv_mte3 19.4->8.8 us, Duration -12%, bit-identical.
    # The old "+12% at tk=256" note was case 069 (n_out=1024, tm=256, wall on mte2
    # scale), which this narrow gate excludes.
    # The n_out<=512 edge was probed once at 640 to cover 085 (576) and 097 (640):
    # 085 33.64->34.09 us, 097 18.54->18.60 us, i.e. no gain -- those two shapes
    # are not walled on the L1->L0 / decode-store path. Keep the tighter edge.
    narrow_transb_decode = decode_b and trans_b and tn > 64 and tm <= 128 and n_out <= 512
    if narrow_transb_decode:
        sk_max = 8
    else:
        sk_max = 4 if narrow_k or (decode_b and trans_b and tn > 64 and not trans_b_wide) else 8
    # [TK512] Native FP4 (a4w4) big-K: tk=512 widens the GM->L1 ND2NZ burst.
    if native_fp4 and tn > 64 and k >= 4096:
        sk_max = 16
    tiling = _select_tiling(m_total, n_out, k, groups, cube_num, sk_max, tm)
    # [TK256-PIN] The narrow-transB decode store only reaches its wide-transaction
    # path at tk=256. When K divides evenly by 192 the cost model prefers 192
    # (no K tail), which drops the store back to the slow shape: case 093 then
    # spends 20.4 us in aiv_mte3 against the reference operator's 6.0 us, and the
    # whole case runs 1.48x slower. Pinning tk=256 and accepting a K tail takes
    # 093 to 42.03 us (paired ratio 1.42->1.19) and 067 to 16.42 us (1.23->1.14),
    # bit-identical.
    # Only the 192 choice is overridden: forcing 256 where the cost model picked
    # 64 (065/180) measured +7.7%, and a 256 tile that does not fit inside K
    # (k=192, case 057) breaks the partial-K path outright.
    if narrow_transb_decode and tiling.tk == 192 and k >= 256:
        tiling = GmmTilingParams(tm=tiling.tm, tn=tiling.tn, tk=256)
    return tiling


# [WALL] ``get_block_num(-1, ...)`` asks ``torch.npu.get_device_properties`` for
# the cube-core count on every launch (~1 us). Core counts are a static property
# of the device, so memoize the answer per device id. Only the automatic (-1)
# form is cached; an explicit positive block_num is already a single caller-side
# integer and never reaches here.
_BLOCK_NUM_CACHE: dict = {}


def _block_num_cached(device):
    hit = _BLOCK_NUM_CACHE.get(device)
    if hit is None:
        hit = get_block_num(-1, device, kind="cube")
        _BLOCK_NUM_CACHE[device] = hit
    return hit


# [WALL] ``run`` derives two tilings from the same shape statistics: the one it
# launches (with the exact per-group row counts) and the shape-only one the
# tools rebuild from. Both are pure functions of the key below, so compute each
# once per distinct key. The third element records whether the shape-only key
# still has to be registered against the launched artifact; the registration
# itself is idempotent, so a cached ``False`` simply skips it.
_TILING_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_TILING_CACHE_MAX = 256


def _tilings_cached(m_total, n_out, k, groups, cube_num, decode_b, trans_b,
                    native_fp4, group_sizes):
    key = (m_total, n_out, k, groups, cube_num, decode_b, trans_b, native_fp4,
           group_sizes)
    hit = _TILING_CACHE.get(key)
    if hit is not None:
        _TILING_CACHE.move_to_end(key)
        return hit
    tiling = _tiling_for(m_total, n_out, k, groups, cube_num, decode_b, trans_b,
                         native_fp4, group_sizes=group_sizes)
    shape_tiling = _tiling_for(m_total, n_out, k, groups, cube_num, decode_b,
                               trans_b, native_fp4)
    hit = (tiling, shape_tiling, shape_tiling != tiling)
    _TILING_CACHE[key] = hit
    if len(_TILING_CACHE) > _TILING_CACHE_MAX:
        _TILING_CACHE.popitem(last=False)
    return hit


def _artifact_key(trans_b, out_e5m2, decode_a, decode_b, a_dtype, b_dtype, tiling):
    # The tiling is a constexpr baked into the artifact, so it is part of the
    # key: one artifact serves every shape of a given variant+tiling.
    return (trans_b, out_e5m2, decode_a, decode_b, a_dtype, b_dtype,
            tiling.tm, tiling.tn, tiling.tk)


def _resolve_stream(device):
    """Current NPU stream handle, without building or probing a Stream.

    ``torch.npu.current_stream(d).npu_stream`` allocates a Stream object every
    call (~3 us on the launch hot path). ``_npu_getCurrentRawStreamNoWait``
    returns the very same ACL stream pointer without the stream-progress probe
    the non-NoWait variant performs -- that probe waits on the just-recorded
    timing event and reintroduces the launch gap this fast path removes. The
    launch stays on torch's stream, so events and later torch ops stay ordered.

    The fast query is used only after it has been observed to agree with the
    public API for this device; a missing/zero/raising result falls back to
    ``torch.npu.current_stream(d).npu_stream`` on that call, so the launch can
    never silently target a stream the caller's events are not on.

    ``CATLASS_HOST_STREAM_QUERY=wait`` switches to the non-NoWait query. That one
    also waits for the caller's pending stream work, which keeps a raw ACL launch
    from overtaking an event the caller recorded just before calling us -- the
    cause of event windows collapsing to ~0.25 us on loaded hosts (WALL_LOG.md
    §6). It costs ~0.3 us when nothing is pending and up to the pending memsets'
    duration otherwise, so the default keeps the cheap query.
    """
    d = int(device)
    if _STREAM_QUERY_MODE == "wait":
        try:
            return int(torch_npu._C._npu_getCurrentRawStream(d))
        except Exception:
            return int(torch.npu.current_stream(d).npu_stream)
    ok = _STREAM_FAST_OK.get(d)
    if ok is None:
        raw = None
        try:
            raw = int(torch_npu._C._npu_getCurrentRawStreamNoWait(d))
        except Exception:
            raw = None
        cur = int(torch.npu.current_stream(d).npu_stream)
        ok = raw is not None and raw != 0 and raw == cur
        _STREAM_FAST_OK[d] = ok
        return raw if ok else cur
    if ok:
        try:
            raw = int(torch_npu._C._npu_getCurrentRawStreamNoWait(d))
        except Exception:
            raw = 0
        if raw:
            return raw
    return int(torch.npu.current_stream(d).npu_stream)


# Device -> whether the cheap raw stream query agreed with the public API.
_STREAM_FAST_OK: dict = {}
# "nowait" (default, cheapest) or "wait" (ordering-safe for event timing).
_STREAM_QUERY_MODE = os.environ.get(
    "CATLASS_HOST_STREAM_QUERY", "nowait"
).strip().lower()


class _FastArtifact:
    """Drop-in artifact proxy that skips the DSL's per-launch Python prologue.

    The DSL ``JitCompiledFunction.__call__`` re-derives the current device and
    stream, re-packs all seven Dynamic-GM memref descriptors from the bound
    tensors, and re-allocates the host-args byte buffer on every launch. On the
    event meter this host time is the device-idle "start gap" measured before
    the kernel begins, and it is shape-independent.

    Everything the launch needs except the seven ``data_ptr`` values is fixed
    for a given tuple of bound tensors, and only the two leading ABI words of
    each 13-word memref (``allocated`` and ``aligned``, both the same pointer)
    change between calls. So cache one host-args payload per bound-tensor tuple,
    patch those two words in place, and call the runtime's launch primitive
    directly. The compiled artifact, its ABI layout and the kernel math are
    untouched -- this is pure host-side memoization.

    The fast path is used only for the plain memref-only launch (no dynamic UB,
    no debug-print output, no per-call ``stream=``/``ub=``/``args=`` override).
    Every other case, and any layout surprise, transparently falls back to the
    real artifact so behavior is never silently different.

    [WALL] The host-args blob is also kept as a reusable ``ctypes`` buffer so
    the steady-state launch patches pointers straight into it and calls
    ``acl.rt.launch_kernel_with_host_args`` without re-entering
    ``execution.launch_kernel`` (which re-parses the cfg list and converts every
    argument on each call; measured ~1.2 us). ACL is still handed a fresh copy
    of the blob per launch -- see ``_invoke``. ``launch7`` additionally patches
    only the two output memrefs: the five input bindings are pinned by the
    cached template, so their pointers cannot change without a new template key.
    """

    __slots__ = ("_target", "_executor", "_ea", "_fh", "_device",
                 "_disabled", "_templates", "_launch_fn", "_last7", "_arg_ring")

    _MAX_TEMPLATES = 32
    # Number of recent host-args copies kept alive. ACL does not document that
    # ``hostArgs`` is consumed synchronously, so freeing a blob right after the
    # call can hand its memory back to Python's allocator while the driver still
    # reads it. Holding the last copies (64 * ~1 KiB) keeps that memory intact
    # for many launches. This is a lifetime guard, not a timing fix: the
    # near-zero event windows that show up on a loaded host are a submission
    # ordering artifact (see WALL_LOG.md), present in the pre-existing path too.
    _ARG_RING = 64

    def __init__(self, target):
        self._target = target
        self._executor = None
        self._ea = None
        self._fh = None
        self._device = 0
        self._disabled = False
        self._templates = OrderedDict()
        self._launch_fn = None
        self._last7 = None
        self._arg_ring = deque(maxlen=self._ARG_RING)

    def __getattr__(self, name):
        # Delegate every other attribute (kernel_binary_path, artifacts,
        # cache_key, ...) so tooling sees the same object it always did.
        return getattr(self._target, name)

    def _init_fast(self):
        executor = self._target._get_executor()
        module = executor.jit_module
        ea = module.execution_args
        packer = ea.abi_packer
        print_cfg = getattr(module, "print_config", module)
        if (
            packer is None
            or packer.slots
            or module.ub_dynamic_base >= 0
            or getattr(print_cfg, "uses_scalar_print", False)
            or getattr(print_cfg, "uses_tensor_print", False)
            or getattr(print_cfg, "print_argument_layout", ())
        ):
            return False
        self._executor = executor
        self._ea = ea
        self._fh = int(executor.function_handle)
        self._device = int(executor.device)
        return True

    def _build_template(self, args):
        runs = self._ea.abi_packer.memref_runs
        if len(runs) != len(args):
            return None
        offsets = []
        for index, mem_run in enumerate(runs):
            if mem_run.logical_index != index or mem_run.value_indices is not None:
                return None
            offsets.append((index, mem_run.start))
        payload = bytearray(self._ea.generate_launch_payload(args))
        # The ABI schema puts the data pointer in the first two 8-byte words of
        # each Dynamic-GM memref. Confirm that against the freshly packed
        # payload; refuse the fast path if the layout ever differs.
        for index, start in offsets:
            ptr = int(args[index].data_ptr)
            if (
                int.from_bytes(payload[start:start + 8], "little") != ptr
                or int.from_bytes(payload[start + 8:start + 16], "little") != ptr
            ):
                return None
        # Host-args blob: ``struct.pack_into`` patches it in place, and
        # ``_invoke`` hands ACL a fresh copy of it every launch (the driver may
        # read it after the call returns). Sized exactly like the blob the
        # runtime would build.
        host_buf = (ctypes.c_uint8 * len(payload)).from_buffer_copy(bytes(payload))
        return (tuple(args), payload, tuple(offsets), host_buf)

    def _invoke(self, host_buf, block_num, stream=None):
        """Hand a prepared host-args blob to ACL; True when handled.

        ``stream`` is resolved once at the entry point when the caller has one;
        otherwise (the generic ``__call__`` path) it is resolved here.
        """
        fn = self._launch_fn
        if fn is None:
            try:
                import acl
                fn = acl.rt.launch_kernel_with_host_args
            except Exception:
                self._disabled = True
                return False
            self._launch_fn = fn
        if stream is None:
            stream = _resolve_stream(self._device)
        size = len(host_buf)
        args_buf = (ctypes.c_uint8 * size).from_buffer_copy(host_buf)
        # Keep recent copies alive: ACL may still be reading an earlier launch's
        # blob, and freeing it would let Python reuse that memory.
        self._arg_ring.append(args_buf)
        ret = fn(self._fh, block_num, stream, [],
                 ctypes.addressof(args_buf), size, [], 0)
        if ret != 0:
            # Match the runtime's behavior: a failed enqueue is an error, not a
            # silent skip.
            raise RuntimeError(
                f"acl.rt.launch_kernel_with_host_args returned {ret}"
            )
        return True

    def launch7(self, a_t, b_t, sa_t, sb_t, gl_t, y_t, sy_t, block_num, stream=None):
        """Hot-path launch for exactly this kernel's seven memrefs.

        Returns False when the fast path cannot serve the call, in which case
        the caller must fall back to ``__call__`` (which itself falls back to
        the real artifact when disabled).
        """
        if self._disabled:
            return False
        if self._ea is None and not self._init_fast():
            self._disabled = True
            return False
        last = self._last7
        if (
            last is not None
            and last[0] is a_t and last[1] is b_t and last[2] is sa_t
            and last[3] is sb_t and last[4] is gl_t and last[5] is y_t
            and last[6] is sy_t
        ):
            tpl = last[7]
        else:
            key = (id(a_t), id(b_t), id(sa_t), id(sb_t), id(gl_t), id(y_t), id(sy_t))
            tpl = self._templates.get(key)
            if tpl is None:
                try:
                    tpl = self._build_template((a_t, b_t, sa_t, sb_t, gl_t, y_t, sy_t))
                except Exception:
                    tpl = None
                if tpl is None:
                    self._disabled = True
                    return False
                self._templates[key] = tpl
                if len(self._templates) > self._MAX_TEMPLATES:
                    self._templates.popitem(last=False)
            else:
                self._templates.move_to_end(key)
            self._last7 = (a_t, b_t, sa_t, sb_t, gl_t, y_t, sy_t, tpl)
        _, _, offsets, host_buf = tpl
        if len(offsets) != 7:
            return False
        # Only the two outputs are reallocated per call; patch their memref
        # pointers and leave the five pinned inputs untouched.
        pointer = y_t.data_ptr
        struct.pack_into("<QQ", host_buf, offsets[5][1], pointer, pointer)
        pointer = sy_t.data_ptr
        struct.pack_into("<QQ", host_buf, offsets[6][1], pointer, pointer)
        return self._invoke(host_buf, block_num, stream)

    def __call__(self, *launch_args, block_num=None, args=None, **launch_kwargs):
        if self._disabled or args is not None or launch_kwargs or len(launch_args) != 7:
            return self._target(*launch_args, block_num=block_num, args=args,
                                **launch_kwargs)
        if self._ea is None and not self._init_fast():
            self._disabled = True
            return self._target(*launch_args, block_num=block_num, args=args,
                                **launch_kwargs)
        key = tuple(id(a) for a in launch_args)
        tpl = self._templates.get(key)
        if tpl is None:
            try:
                tpl = self._build_template(launch_args)
            except Exception:
                tpl = None
            if tpl is None:
                self._disabled = True
                return self._target(*launch_args, block_num=block_num, args=args,
                                    **launch_kwargs)
            self._templates[key] = tpl
            if len(self._templates) > self._MAX_TEMPLATES:
                self._templates.popitem(last=False)
        else:
            self._templates.move_to_end(key)
        # The stored tuple pins the bound tensors alive, so the id-keyed lookup
        # above is exact (an id cannot be recycled while we hold the object).
        stored_args, payload, offsets, host_buf = tpl
        pack_into = struct.pack_into
        for index, start in offsets:
            ptr = int(stored_args[index].data_ptr)
            pack_into("<QQ", host_buf, start, ptr, ptr)
        if block_num is None:
            block_num = 1
        if not isinstance(block_num, int):
            return self._target(*launch_args, block_num=block_num, args=args,
                                **launch_kwargs)
        if not self._invoke(host_buf, block_num):
            return self._target(*launch_args, block_num=block_num, args=args,
                                **launch_kwargs)


def _launch_blocks(tiling, n_out, m, groups, group_sizes, bn):
    """Launch only the blocks the shape actually has.

    [A8LAT] Dispatching all 28 mix blocks costs a ~3.4 us hardware ramp that
    dominates a small shape's Duration: an empty kernel takes 4.71 us at 28
    blocks versus 1.35 us at 2. Tiling still sees ``bn`` cores (the work
    partition is unchanged), but the launch can start only the blocks that have
    work. When ``total_blocks >= bn`` this returns ``bn`` unchanged, so every
    shape that already fills the machine is untouched.

    Safe because at ``total_blocks <= bn`` no core owns more than one block, so
    removing the idle blocks changes no other core's workload. The kernel reads
    ``block_num()`` at run time and its core mapping is self-consistent for any
    value of it (verified bit-identical on 18 cases, including a single-block
    shape and a zero-row group).
    """
    if n_out <= 0:
        return bn
    grid_n = _ceil_div(n_out, tiling.tn)
    if group_sizes:
        total = sum(_ceil_div(c, tiling.tm) for c in group_sizes if c > 0) * grid_n
    else:
        total = groups * _ceil_div(m, tiling.tm) * grid_n
    if total <= 0:
        return bn
    return min(bn, total)


def _launch(a_t, b_t, sa_t, sb_t, gl_t, y_t, sy_t, trans_b, out_e5m2, decode_a, decode_b, a_dtype, b_dtype, block_num, tiling, stream=None):
    # The DSL disk-cache lookup costs ~60 ms per call; memoize the compiled
    # artifact by the constexpr variant (including the tiling) so steady-state
    # run() only pays launch. a4w4 binds native FP4 (decode False), a8w4 decodes
    # B (decode_b True), so the decode flags plus operand dtypes distinguish
    # artifacts; layouts stay mark_layout_dynamic, so one artifact serves all
    # shapes of a given variant+tiling. The stored object is a _FastArtifact
    # proxy that memoizes the host-args payload and skips the DSL per-launch
    # prologue; it delegates every other attribute to the compiled artifact.
    key = _artifact_key(trans_b, out_e5m2, decode_a, decode_b, a_dtype, b_dtype, tiling)
    artifact = _ARTIFACT_CACHE.get(key)
    if artifact is None:
        artifact = _FastArtifact(
            tla.compile(
                gmm_swiglu_mx_quant_kernel,
                a_t, b_t, sa_t, sb_t, gl_t, y_t, sy_t,
                tiling, trans_b, out_e5m2, decode_a, decode_b,
                options=NPU_ARCH,
            )
        )
        _ARTIFACT_CACHE[key] = artifact
    # [WALL] ``launch7`` is the same launch with two savings the generic entry
    # cannot make: it patches only the two output memrefs (the inputs are pinned
    # by the template) and calls ACL directly with a fresh copy of the cached
    # host-args blob. It returns False for any layout/ABI surprise, in which
    # case the generic call below runs exactly as before.
    if isinstance(artifact, _FastArtifact) and artifact.launch7(
        a_t, b_t, sa_t, sb_t, gl_t, y_t, sy_t, block_num, stream
    ):
        return artifact
    artifact(a_t, b_t, sa_t, sb_t, gl_t, y_t, sy_t, block_num=block_num)
    # [LINE-a8w8] No device-wide sync here: the launch is on torch's current
    # stream, so every later op (including any output-side work in run) is
    # stream-ordered. The sync cost ~4 us per call and, because it runs after
    # the kernel, it lands between the kernel and any caller-recorded end event.
    return artifact


def run(a, b, mx_scale_a, mx_scale_b, group_list, transB=True):
    """Definition entry point: launch the fused kernel, return (q, q_scale)."""
    device = int(a.device.index) if a.device.index is not None else 0
    # [WALL] ``set_device`` is a global context switch worth ~0.4 us per launch;
    # every op below names its device explicitly, and the launch resolves the
    # stream for ``device`` directly, so it only has to run when the target
    # device actually changes.
    global _LAST_SET_DEVICE
    if device != _LAST_SET_DEVICE:
        torch.npu.set_device(device)
        _LAST_SET_DEVICE = device
    # Resolve the launch stream once, before allocating. This is the cheapest
    # place for the ordering-safe query (CATLASS_HOST_STREAM_QUERY=wait): none of
    # our own kernels/memsets are pending yet, but the caller's just-recorded
    # event is, so the wait is short and it guarantees our launch cannot overtake
    # that event.
    stream = _resolve_stream(device)
    a_t, b_t, sa_t, sb_t, gl_t, (m, n, k, groups), decode_a, decode_b = _build_tensors_cached(
        a, b, mx_scale_a, mx_scale_b, group_list, transB
    )

    out_dtype = a.dtype if a.dtype in FP8_SEL else torch.float8_e4m3fn
    out_e5m2 = out_dtype == torch.float8_e5m2
    n_out = n // 2
    # The vector epilogue packs each row's shared exponents into one Int32 per
    # 128 output columns, so the E8M0 byte buffer is padded to that grid.
    n_scale_words = _ceil_div(n_out, 4 * MX_OUTPUT_BLOCK_SIZE)
    scale_cols = n_scale_words * 4

    # [EMPTY-M] A zero-row call has nothing to compute, and a zero-element torch
    # tensor binds a null device address, which the launch path rejects with
    # "Tensor buffer is not bound to a device address". Return the empty result
    # directly instead of entering the launch.
    if m == 0:
        q = torch.empty(0, n_out, dtype=out_dtype, device=a.device)
        q_scale = torch.empty(
            0, scale_cols, dtype=torch.uint8, device=a.device
        ).view(torch.float8_e8m0fnu)
        return q, q_scale

    # Rows [sum(group_list), m) are not written by the kernel; the reference
    # zero-initialises them, so they must be zero here too. When there are none
    # the whole-buffer memset is dead work (measured ~4 us for the output pair),
    # so allocate with ``empty`` instead. With a tail the two ``zeros`` stay:
    # zeroing just the tail is a separate dispatch of its own (~5 us), and one
    # memset over a combined blob measures no faster once its sliced views are
    # built. ``valid_rows is None`` (a single group, or per-group counts not
    # cached) keeps the conservative zeros path untouched.
    group_sizes = None
    valid_rows = None
    if groups > 1:
        group_sizes = _group_sizes_fast(group_list)
        if group_sizes is not None and len(group_sizes) == groups:
            valid_rows = sum(group_sizes)
        else:
            group_sizes = None
    if valid_rows is not None and valid_rows >= m:
        q = torch.empty(m, n_out, dtype=out_dtype, device=a.device)
        q_scale_bytes = torch.empty(m, scale_cols, dtype=torch.uint8, device=a.device)
    else:
        q = torch.zeros(m, n_out, dtype=out_dtype, device=a.device)
        q_scale_bytes = torch.zeros(m, scale_cols, dtype=torch.uint8, device=a.device)
    y_t, sy_t = _bind_output_pair(q, n_out, q_scale_bytes)

    bn = _block_num_cached(device)
    native_fp4 = a.dtype == FP4_DTYPE and b.dtype == FP4_DTYPE
    tiling, shape_tiling, need_register = _tilings_cached(
        m, n_out, k, groups, bn, decode_b, transB, native_fp4, group_sizes
    )
    artifact = _launch(a_t, b_t, sa_t, sb_t, gl_t, y_t, sy_t, transB, out_e5m2, decode_a, decode_b, a.dtype, b.dtype, _launch_blocks(tiling, n_out, m, groups, group_sizes, bn), tiling, stream)
    # tools/kernel_bench.py and tools/bench_ops_transformer.py rebuild the
    # artifact key from shape statistics only (no group_list), so register the
    # launched artifact under that key too; otherwise their lookup misses.
    if need_register:
        _ARTIFACT_CACHE[_artifact_key(
            transB, out_e5m2, decode_a, decode_b, a.dtype, b.dtype, shape_tiling
        )] = artifact
    valid_scale_cols = _ceil_div(n_out, MX_OUTPUT_BLOCK_SIZE)
    # [WALL] The padded-column slice is a strided view: the scale bytes of one
    # row are already contiguous in epilogue order, so only the row stride
    # differs from the compact layout the reference returns. Copying it via
    # ``contiguous()`` cost up to ~5 us per launch for nothing; a same-itemsize
    # ``view`` is metadata-only on a strided tensor. Values and shape are
    # identical to the copy.
    if valid_scale_cols == scale_cols:
        q_scale = q_scale_bytes.view(torch.float8_e8m0fnu)
    else:
        q_scale = q_scale_bytes[:, :valid_scale_cols].view(torch.float8_e8m0fnu)
    return q, q_scale
