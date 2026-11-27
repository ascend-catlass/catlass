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

from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch_npu  # noqa: F401

import catlass.tla as tla
from catlass.tla.runtime import from_dlpack


MX_SCALE_GROUP_NUM = 32
MX_OUTPUT_BLOCK_SIZE = 32


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

    l1a = tla.allocate((2 if tla.const_expr(_decodeA) else 1) * tm * tk, dtype_a, tla.AddressSpace.l1, 512)
    l1b_act = tla.allocate((2 if tla.const_expr(_decodeB) else 1) * tk * tn, dtype_b, tla.AddressSpace.l1, 512)
    l1b_gate = tla.allocate((2 if tla.const_expr(_decodeB) else 1) * tk * tn, dtype_b, tla.AddressSpace.l1, 512)
    l1sa = tla.allocate(tm * sk, tla.Float8E8M0, tla.AddressSpace.l1, 512)
    l1sb_act = tla.allocate(sk * tn, tla.Float8E8M0, tla.AddressSpace.l1, 512)
    l1sb_gate = tla.allocate(sk * tn, tla.Float8E8M0, tla.AddressSpace.l1, 512)
    l0a = tla.allocate(tm * tk, dtype_a, tla.AddressSpace.l0a, 1024)
    l0b_act = tla.allocate(tk * tn, dtype_b, tla.AddressSpace.l0b, 1024)
    l0b_gate = tla.allocate(tk * tn, dtype_b, tla.AddressSpace.l0b, 1024)
    l0c_act = tla.allocate(tm * tn, tla.Float32, tla.AddressSpace.l0c, 1024)
    l0c_gate = tla.allocate(tm * tn, tla.Float32, tla.AddressSpace.l0c, 1024)

    ub_act = tla.allocate(half * tn, tla.Float32, tla.AddressSpace.ub, 256)
    ub_gate = tla.allocate(half * tn, tla.Float32, tla.AddressSpace.ub, 256)
    ub_y = tla.allocate(half * tn // 4, tla.Int32, tla.AddressSpace.ub, 256)
    ub_sy = tla.allocate(half * 8, tla.Int32, tla.AddressSpace.ub, 256)

    if tla.const_expr(_decodeA or _decodeB):
        ub_a_pack = tla.allocate(tm * tk // 2, tla.Int8, tla.AddressSpace.ub, 256)
        ub_a_fp8 = tla.allocate(tm * tk, tla.Int8, tla.AddressSpace.ub, 256)
        ub_ba_pack = tla.allocate(tk * tn // 2, tla.Int8, tla.AddressSpace.ub, 256)
        ub_ba_fp8 = tla.allocate(tk * tn, tla.Int8, tla.AddressSpace.ub, 256)
        ub_bg_pack = tla.allocate(tk * tn // 2, tla.Int8, tla.AddressSpace.ub, 256)
        ub_bg_fp8 = tla.allocate(tk * tn, tla.Int8, tla.AddressSpace.ub, 256)
    if tla.const_expr(_decodeB and _transB):
        # Unit E8M0 scale (127 = 2^0) for hardware mx_to FP4->FP8 widening; the
        # cube applies the real per-32 MX scale separately via gm_sb.
        ub_scale_unit = tla.allocate(sk * tn, tla.Int8, tla.AddressSpace.ub, 256)

    c_ref = tla.make_tensor(
        gm_a.ptr,
        tla.make_layout(
            tla.make_shape(tm, tn), tla.make_stride(tn, 1), layoutTag=tla.arch.RowMajor
        ),
    )
    t_l0c_act = tla.make_tensor_like(l0c_act, c_ref, tla.arch.L0Clayout)
    t_l0c_gate = tla.make_tensor_like(l0c_gate, c_ref, tla.arch.L0Clayout)

    mtx_l1 = tla.mutex("l1_ab", id=1)
    mtx_l0 = tla.mutex("l0_ab", id=2)
    ub_ready = tla.cross_flag("ub_ready")
    ub_free = tla.cross_flag("ub_free")
    mtx_ub_out = tla.mutex(resource="ub_out", id=0)
    mtx_l0c = tla.mutex(resource="l0c", id=4)

    if tla.const_expr(_decodeA or _decodeB):
        a_full = tla.cross_flag("a_full", mode=4)
        a_free = tla.cross_flag("a_free", mode=4)
        b_full = tla.cross_flag("b_full", mode=4)
        b_free = tla.cross_flag("b_free", mode=4)

    with tla.cube():
        core_num = tla.arch.block_num()
        core_idx = tla.arch.block_idx()
        start_core_idx = c0
        m_start = c0
        first_tile = True

        if tla.const_expr(_decodeA):
            tla.cross_core_set_flag(a_free, tla.pipes.MTE1, aiv_id=0)
            tla.cross_core_set_flag(a_free, tla.pipes.MTE1, aiv_id=0)
        if tla.const_expr(_decodeB):
            tla.cross_core_set_flag(b_free, tla.pipes.MTE1, aiv_id=0)
            tla.cross_core_set_flag(b_free, tla.pipes.MTE1, aiv_id=0)

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
                        if tla.const_expr(_decodeA):
                            l1a_buf = l1a + (kk % 2) * (tm * tk)
                        else:
                            l1a_buf = l1a
                        if tla.const_expr(_decodeB):
                            l1b_act_buf = l1b_act + (kk % 2) * (tk * tn)
                            l1b_gate_buf = l1b_gate + (kk % 2) * (tk * tn)
                        else:
                            l1b_act_buf = l1b_act
                            l1b_gate_buf = l1b_gate
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
                            l1sa, tla.make_layout_with_tag((tm, sk), tla.Float8E8M0, tla.arch.zZMxScale)
                        )
                        t_l1sb_act = tla.make_tensor(
                            l1sb_act, tla.make_layout_with_tag((sk, tn), tla.Float8E8M0, tla.arch.nNMxScale)
                        )
                        t_l1sb_gate = tla.make_tensor(
                            l1sb_gate, tla.make_layout_with_tag((sk, tn), tla.Float8E8M0, tla.arch.nNMxScale)
                        )

                        mtx_l1.lock(pipe=tla.pipes.MTE2)
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
                        mtx_l1.unlock(pipe=tla.pipes.MTE2)
                        mtx_l1.lock(pipe=tla.pipes.MTE1)

                        t_l0a = tla.make_tensor_like(l0a, t_l1a, tla.arch.zN)
                        t_l0b_act = tla.make_tensor_like(l0b_act, t_l1b_act, tla.arch.nZ)
                        t_l0b_gate = tla.make_tensor_like(l0b_gate, t_l1b_gate, tla.arch.nZ)

                        mtx_l0.lock(pipe=tla.pipes.MTE1)
                        tla.copy(t_l0a, t_l1a, scale=t_l1sa)
                        tla.copy(t_l0b_act, t_l1b_act, scale=t_l1sb_act)
                        tla.copy(t_l0b_gate, t_l1b_gate, scale=t_l1sb_gate)
                        mtx_l1.unlock(pipe=tla.pipes.MTE1)
                        if tla.const_expr(_decodeA):
                            tla.cross_core_set_flag(a_free, tla.pipes.MTE1, aiv_id=0)
                        if tla.const_expr(_decodeB):
                            tla.cross_core_set_flag(b_free, tla.pipes.MTE1, aiv_id=0)
                        mtx_l0.unlock(pipe=tla.pipes.MTE1)

                        init_c = True if kk == 0 else False
                        mtx_l0.lock(pipe=tla.pipes.CUBE)
                        tla.mmad_mx(t_l0c_act, t_l0a, t_l0b_act, init_c=init_c)
                        tla.mmad_mx(t_l0c_gate, t_l0a, t_l0b_gate, init_c=init_c)
                        mtx_l0.unlock(pipe=tla.pipes.CUBE)

                    mtx_l1.lock(pipe=tla.pipes.MTE2)
                    mtx_l1.unlock(pipe=tla.pipes.MTE2)
                    mtx_l0.lock(pipe=tla.pipes.MTE1)
                    mtx_l0.unlock(pipe=tla.pipes.MTE1)
                    mtx_l0c.unlock(pipe=tla.pipes.CUBE)

                    ub_act_cube = tla.make_tensor_like(ub_act, t_l0c_act, tla.arch.RowMajor)
                    ub_gate_cube = tla.make_tensor_like(ub_gate, t_l0c_gate, tla.arch.RowMajor)
                    l0c2ub = tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M)
                    if not first_tile:
                        tla.cross_core_wait_flag(ub_free, tla.pipes.FIX)
                    mtx_l0c.lock(pipe=tla.pipes.FIX)
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
        ub_sy_half = tla.make_tensor_like(ub_sy, sy_ref, tla.arch.RowMajor)

        if tla.const_expr(_decodeA or _decodeB):
            k_tiles_v = (k_dim + tk - 1) // tk
            k_pack = k_dim // 2
            n_pack = n_dim // 2
            b_tag_v = tla.arch.zN
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
            a_pk_f = tla.make_tensor(ub_a_pack, tla.make_layout(tla.make_shape(tm * tk // 2), tla.make_stride(1)))
            a_fp_f = tla.make_tensor(ub_a_fp8, tla.make_layout(tla.make_shape(tm * tk), tla.make_stride(1)))
            ba_pk_f = tla.make_tensor(ub_ba_pack, tla.make_layout(tla.make_shape(tk * tn // 2), tla.make_stride(1)))
            ba_fp_f = tla.make_tensor(ub_ba_fp8, tla.make_layout(tla.make_shape(tk * tn), tla.make_stride(1)))
            bg_pk_f = tla.make_tensor(ub_bg_pack, tla.make_layout(tla.make_shape(tk * tn // 2), tla.make_stride(1)))
            bg_fp_f = tla.make_tensor(ub_bg_fp8, tla.make_layout(tla.make_shape(tk * tn), tla.make_stride(1)))
            # Word-wide views for the nontransB full-tile SIMT decode: Int32 packs
            # 4 packed bytes (8 nibbles) per load; Int16 packs 2 decoded FP8 bytes
            # per store, amortizing loop induction and address arithmetic.
            ba_pk_w = tla.make_tensor(tla.recast_ptr(ub_ba_pack, dtype=tla.Int32), tla.make_layout(tla.make_shape(tk * tn // 8), tla.make_stride(1)))
            ba_fp_h = tla.make_tensor(tla.recast_ptr(ub_ba_fp8, dtype=tla.Int16), tla.make_layout(tla.make_shape(tk * tn // 2), tla.make_stride(1)))
            bg_pk_w = tla.make_tensor(tla.recast_ptr(ub_bg_pack, dtype=tla.Int32), tla.make_layout(tla.make_shape(tk * tn // 8), tla.make_stride(1)))
            bg_fp_h = tla.make_tensor(tla.recast_ptr(ub_bg_fp8, dtype=tla.Int16), tla.make_layout(tla.make_shape(tk * tn // 2), tla.make_stride(1)))

        if tla.const_expr(_decodeB and _transB):
            # Hardware mx_to widening views: packed FP4 payload (logical shape)
            # -> FP8 payload, unit E8M0 scale. transB groups along K=dim1.
            # Full tiles only (partial keeps SIMT).
            if tla.const_expr(_transB):
                mx_in_ba = tla.make_tensor(
                    tla.recast_ptr(ub_ba_pack, dtype=tla.Float4E2M1),
                    tla.make_layout(tla.make_shape(tn, tk), tla.make_stride(tk, 1)),
                )
                mx_out_ba = tla.make_tensor(
                    tla.recast_ptr(ub_ba_fp8, dtype=dtype_b),
                    tla.make_layout(tla.make_shape(tn, tk), tla.make_stride(tk, 1)),
                )
                mx_in_bg = tla.make_tensor(
                    tla.recast_ptr(ub_bg_pack, dtype=tla.Float4E2M1),
                    tla.make_layout(tla.make_shape(tn, tk), tla.make_stride(tk, 1)),
                )
                mx_out_bg = tla.make_tensor(
                    tla.recast_ptr(ub_bg_fp8, dtype=dtype_b),
                    tla.make_layout(tla.make_shape(tn, tk), tla.make_stride(tk, 1)),
                )
                mx_scale = tla.make_tensor(
                    ub_scale_unit,
                    tla.make_layout(tla.make_shape(tn, sk), tla.make_stride(sk, 1)),
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
            if tla.arch.sub_block_idx() == 0:
                scale_flat = tla.make_tensor(
                    ub_scale_unit, tla.make_layout(tla.make_shape(sk * tn), tla.make_stride(1))
                )
                with tla.vec.func(mode="simd"):
                    scale_flat.store(tla.full(127, dtype=tla.Int8))

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
                        gm_b_pack_g = tla.make_tensor(
                            gm_b.ptr + g * k_dim * gm_b.stride[0],
                            tla.make_layout(
                                tla.make_shape(k_dim, n_pack),
                                tla.make_stride(n_pack, 1),
                                origin_shape=tla.make_shape(k_dim, n_pack),
                                layoutTag=tla.arch.RowMajor,
                            ),
                        )
                start_loop = (core_idx + core_num - start_core_idx) % core_num
                block_range = tla.range(start_loop, mn_blocks, core_num)
                for loop_idx in block_range:
                    block_row = loop_idx // grid_n
                    block_col = loop_idx % grid_n
                    out_row = m_start + block_row * tm + sub_idx * half

                    if tla.const_expr(_decodeA or _decodeB):
                        if sub_idx == 0:
                            act_col = block_col * tn
                            gate_col = n_out + block_col * tn
                            for kk in tla.range(c0, k_tiles_v, c1):
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
                                    if tla.const_expr(_transB):
                                        # transB: hardware mx_to FP4->FP8 widening (fast).
                                        gm_ba_pk = tla.get_tile(
                                            gm_b_pack_g,
                                            tla.make_coord(act_col, kk * (tk // 2)),
                                            tla.make_shape(tn, tk // 2),
                                        )
                                        gm_bg_pk = tla.get_tile(
                                            gm_b_pack_g,
                                            tla.make_coord(gate_col, kk * (tk // 2)),
                                            tla.make_shape(tn, tk // 2),
                                        )
                                        bv0 = gm_ba_pk.origin_shape[0]
                                        bv1 = gm_ba_pk.origin_shape[1]
                                        tla.cross_core_wait_flag(b_free, tla.pipes.MTE3, aiv_id=0)
                                        with tla.vec.func(mode="simt", thread_block_dim=256):
                                            tid, _, _ = tla.arch.thread_idx()
                                            nth, _, _ = tla.arch.thread_block_dim()
                                            if bv0 != tn:
                                                for p in tla.range(tid, tk * tn // 2, nth):
                                                    ba_pk_f[p] = 0
                                            if bv1 != tk // 2:
                                                for q in tla.range(tid, tk * tn // 2, nth):
                                                    ba_pk_f[q] = 0
                                        tla.pipe_barrier(tla.pipes.ALL)
                                        tla.copy(
                                            tla.tile_view(ub_ba_pack_t, tla.make_shape(bv0, bv1), tla.make_coord(0, 0)),
                                            gm_ba_pk,
                                        )
                                        tla.pipe_barrier(tla.pipes.ALL)
                                        tla.experimental.mx_to((mx_in_ba, mx_scale), (mx_out_ba, mx_scale))
                                        tla.pipe_barrier(tla.pipes.ALL)
                                        t_l1b_act_v_kk = tla.make_tensor_like(l1b_act + (kk % 2) * (tk * tn), ub_ba_fp8_t, tla.arch.zN)
                                        tla.copy(t_l1b_act_v_kk, ub_ba_fp8_t)
                                        with tla.vec.func(mode="simt", thread_block_dim=256):
                                            tid, _, _ = tla.arch.thread_idx()
                                            nth, _, _ = tla.arch.thread_block_dim()
                                            if bv0 != tn:
                                                for p in tla.range(tid, tk * tn // 2, nth):
                                                    bg_pk_f[p] = 0
                                            if bv1 != tk // 2:
                                                for q in tla.range(tid, tk * tn // 2, nth):
                                                    bg_pk_f[q] = 0
                                        tla.pipe_barrier(tla.pipes.ALL)
                                        tla.copy(
                                            tla.tile_view(ub_bg_pack_t, tla.make_shape(bv0, bv1), tla.make_coord(0, 0)),
                                            gm_bg_pk,
                                        )
                                        tla.pipe_barrier(tla.pipes.ALL)
                                        tla.experimental.mx_to((mx_in_bg, mx_scale), (mx_out_bg, mx_scale))
                                        tla.pipe_barrier(tla.pipes.ALL)
                                        t_l1b_gate_v_kk = tla.make_tensor_like(l1b_gate + (kk % 2) * (tk * tn), ub_bg_fp8_t, tla.arch.zN)
                                        tla.copy(t_l1b_gate_v_kk, ub_bg_fp8_t)
                                        tla.cross_core_set_flag(b_full, tla.pipes.MTE3, aiv_id=0)
                                    else:
                                        # nontransB: mx_to K-major grouping is pathological
                                        # (~25x slower); keep SIMT elementwise nibble decode.
                                        gm_ba_pk = tla.get_tile(
                                            gm_b_pack_g,
                                            tla.make_coord(kk * tk, act_col // 2),
                                            tla.make_shape(tk, tn // 2),
                                        )
                                        gm_bg_pk = tla.get_tile(
                                            gm_b_pack_g,
                                            tla.make_coord(kk * tk, gate_col // 2),
                                            tla.make_shape(tk, tn // 2),
                                        )
                                        bv0 = gm_ba_pk.origin_shape[0]
                                        bv1 = gm_ba_pk.origin_shape[1]
                                        tla.cross_core_wait_flag(b_free, tla.pipes.MTE3, aiv_id=0)
                                        tla.copy(
                                            tla.tile_view(ub_ba_pack_t, tla.make_shape(bv0, bv1), tla.make_coord(0, 0)),
                                            gm_ba_pk,
                                        )
                                        tla.pipe_barrier(tla.pipes.ALL)
                                        with tla.vec.func(mode="simt", thread_block_dim=256):
                                            pk = ba_pk_f
                                            fp = ba_fp_f
                                            pk_w = ba_pk_w
                                            fp_h = ba_fp_h
                                            tid, _, _ = tla.arch.thread_idx()
                                            nth, _, _ = tla.arch.thread_block_dim()
                                            if bv0 == tk:
                                                for w in tla.range(tid, tk * tn // 8, nth):
                                                    word = pk_w[w]
                                                    b0 = word & 0xFF
                                                    b1 = (word >> 8) & 0xFF
                                                    b2 = (word >> 16) & 0xFF
                                                    b3 = (word >> 24) & 0xFF
                                                    fp_h[4 * w] = (_nibble_fp8_byte(b0 & 0x0F) | (_nibble_fp8_byte((b0 >> 4) & 0x0F) << 8)).to(tla.Int16)
                                                    fp_h[4 * w + 1] = (_nibble_fp8_byte(b1 & 0x0F) | (_nibble_fp8_byte((b1 >> 4) & 0x0F) << 8)).to(tla.Int16)
                                                    fp_h[4 * w + 2] = (_nibble_fp8_byte(b2 & 0x0F) | (_nibble_fp8_byte((b2 >> 4) & 0x0F) << 8)).to(tla.Int16)
                                                    fp_h[4 * w + 3] = (_nibble_fp8_byte(b3 & 0x0F) | (_nibble_fp8_byte((b3 >> 4) & 0x0F) << 8)).to(tla.Int16)
                                            else:
                                                for r in tla.range(tid, tk * tn // 2, nth):
                                                    b = pk[r].to(tla.Int32) & 0xFF
                                                    if r // (tn // 2) < bv0:
                                                        fp[2 * r] = _nibble_fp8(b & 0x0F)
                                                        fp[2 * r + 1] = _nibble_fp8((b >> 4) & 0x0F)
                                                    else:
                                                        fp[2 * r] = 0
                                                        fp[2 * r + 1] = 0
                                        tla.pipe_barrier(tla.pipes.ALL)
                                        t_l1b_act_v_kk = tla.make_tensor(l1b_act + (kk % 2) * (tk * tn), tla.make_layout_with_tag((tk, tn), dtype_b, tla.arch.zN))
                                        tla.copy(t_l1b_act_v_kk, ub_ba_fp8_k)
                                        tla.copy(
                                            tla.tile_view(ub_bg_pack_t, tla.make_shape(bv0, bv1), tla.make_coord(0, 0)),
                                            gm_bg_pk,
                                        )
                                        tla.pipe_barrier(tla.pipes.ALL)
                                        with tla.vec.func(mode="simt", thread_block_dim=256):
                                            pk = bg_pk_f
                                            fp = bg_fp_f
                                            pk_w = bg_pk_w
                                            fp_h = bg_fp_h
                                            tid, _, _ = tla.arch.thread_idx()
                                            nth, _, _ = tla.arch.thread_block_dim()
                                            if bv0 == tk:
                                                for w in tla.range(tid, tk * tn // 8, nth):
                                                    word = pk_w[w]
                                                    b0 = word & 0xFF
                                                    b1 = (word >> 8) & 0xFF
                                                    b2 = (word >> 16) & 0xFF
                                                    b3 = (word >> 24) & 0xFF
                                                    fp_h[4 * w] = (_nibble_fp8_byte(b0 & 0x0F) | (_nibble_fp8_byte((b0 >> 4) & 0x0F) << 8)).to(tla.Int16)
                                                    fp_h[4 * w + 1] = (_nibble_fp8_byte(b1 & 0x0F) | (_nibble_fp8_byte((b1 >> 4) & 0x0F) << 8)).to(tla.Int16)
                                                    fp_h[4 * w + 2] = (_nibble_fp8_byte(b2 & 0x0F) | (_nibble_fp8_byte((b2 >> 4) & 0x0F) << 8)).to(tla.Int16)
                                                    fp_h[4 * w + 3] = (_nibble_fp8_byte(b3 & 0x0F) | (_nibble_fp8_byte((b3 >> 4) & 0x0F) << 8)).to(tla.Int16)
                                            else:
                                                for r in tla.range(tid, tk * tn // 2, nth):
                                                    b = pk[r].to(tla.Int32) & 0xFF
                                                    if r // (tn // 2) < bv0:
                                                        fp[2 * r] = _nibble_fp8(b & 0x0F)
                                                        fp[2 * r + 1] = _nibble_fp8((b >> 4) & 0x0F)
                                                    else:
                                                        fp[2 * r] = 0
                                                        fp[2 * r + 1] = 0
                                        tla.pipe_barrier(tla.pipes.ALL)
                                        t_l1b_gate_v_kk = tla.make_tensor(l1b_gate + (kk % 2) * (tk * tn), tla.make_layout_with_tag((tk, tn), dtype_b, tla.arch.zN))
                                        tla.copy(t_l1b_gate_v_kk, ub_bg_fp8_k)
                                        tla.cross_core_set_flag(b_full, tla.pipes.MTE3, aiv_id=0)

                    tla.cross_core_wait_flag(ub_ready, tla.pipes.VECTOR)

                    rows = current_m - block_row * tm - sub_idx * half
                    if rows > half:
                        rows = half
                    cols_valid = n_out - block_col * tn
                    if cols_valid > tn:
                        cols_valid = tn
                    nchunks = cols_valid // 64

                    if rows > 0:
                        mtx_ub_out.lock(pipe=tla.pipes.VECTOR)
                        gm_y_t = tla.get_tile(
                            gm_y, tla.make_coord(out_row, block_col * (tn // 4)),
                            tla.make_shape(rows, cols_valid // 4),
                        )
                        gm_sy_t = tla.get_tile(
                            gm_sy, tla.make_coord(out_row, block_col),
                            tla.make_shape(rows, 1),
                        )
                        with tla.vec.func(mode="simd"):
                            m_src = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                            m_lo = tla.create_mask(pattern=tla.mask.H, dtype=tla.Float32)
                            m_i32 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Int32)
                            m_i32_16 = tla.create_mask(pattern=tla.mask.VL16, dtype=tla.Int32)
                            v1 = tla.full(1, tla.Int32)
                            v7FFF = tla.full(0x7FFF, tla.Int32)
                            vMASK = tla.full(-65536, tla.Int32)
                            v7F80 = tla.full(0x7F80, tla.Int32)
                            v7F00 = tla.full(0x7F00, tla.Int32).reduce(tla.ReductionOp.MAX, mask=m_lo)
                            v_emax = tla.full(emax, tla.Int32).reduce(tla.ReductionOp.MAX, mask=m_lo)
                            v_emax_sh = tla.full(emax_shifted, tla.Int32).reduce(tla.ReductionOp.MAX, mask=m_lo)
                            v_zero = tla.full(0, tla.Int32).reduce(tla.ReductionOp.MAX, mask=m_lo)
                            v_254 = tla.full(254, tla.Int32).reduce(tla.ReductionOp.MAX, mask=m_lo)
                            v7FFFFFFF = tla.full(0x7FFFFFFF, tla.Int32)
                            vFF = tla.full(0xFF, tla.Int32)
                            v7FFFFF = tla.full(0x7FFFFF, tla.Int32)
                            zero32 = tla.full(0, tla.Int32)
                            trunc_i32 = tla.params.CastParams(
                                reg_slot=tla.params.RegSlot.ZERO,
                                sat_mode=tla.params.SatMode.NOSAT,
                                round_mode=tla.params.RoundMode.CAST_TRUNC,
                            )
                            round_f32 = tla.params.CastParams(
                                reg_slot=tla.params.RegSlot.ZERO,
                                sat_mode=tla.params.SatMode.NOSAT,
                                round_mode=tla.params.RoundMode.CAST_ROUND,
                            )
                            for r in tla.range(c0, rows, c1):
                                scale_word = v_zero
                                for chunk in tla.range(c0, nchunks, c1):
                                    col = chunk * 64
                                    a_reg = tla.get_tile(ub_act_half, tla.make_coord(r, col), tla.make_shape(1, 64)).load()
                                    g_reg = tla.get_tile(ub_gate_half, tla.make_coord(r, col), tla.make_shape(1, 64)).load()
                                    den = tla.add(1.0, tla.exp(tla.neg(a_reg, mask=m_src), mask=m_src), mask=m_src)
                                    s_val = tla.mul(tla.div(a_reg, den, mask=m_src), g_reg, mask=m_src)
                                    s_bits = s_val.bitcast(tla.Int32)
                                    s_round = tla.bitwise_and(
                                        tla.add(tla.add(s_bits, v7FFF), tla.bitwise_and(tla.shift_right(s_bits, 16), v1)),
                                        vMASK,
                                    )
                                    s_bf16 = tla.shift_right(s_round, 16)
                                    s_bf = s_round.bitcast(tla.Float32)
                                    exp_bits = tla.bitwise_and(s_bf16, v7F80)
                                    max_lo = exp_bits.reduce(tla.ReductionOp.MAX, mask=m_lo)
                                    hi_only = tla.where(m_lo, v_zero, exp_bits)
                                    max_hi = hi_only.reduce(tla.ReductionOp.MAX, mask=m_src)
                                    byte_lo = tla.min(tla.max(tla.sub(tla.shift_right(max_lo, 7), v_emax), v_zero), v_254)
                                    byte_hi = tla.min(tla.max(tla.sub(tla.shift_right(max_hi, 7), v_emax), v_zero), v_254)
                                    max_lo_c = tla.max(max_lo, v_emax_sh)
                                    max_hi_c = tla.max(max_hi, v_emax_sh)
                                    half_lo_bits = tla.sub(v7F00, tla.sub(max_lo_c, v_emax_sh))
                                    half_hi_bits = tla.sub(v7F00, tla.sub(max_hi_c, v_emax_sh))
                                    half_lo = tla.full(tla.shift_left(half_lo_bits, 16).bitcast(tla.Float32), tla.Float32)
                                    half_hi = tla.full(tla.shift_left(half_hi_bits, 16).bitcast(tla.Float32), tla.Float32)
                                    scaled = tla.where(m_lo, tla.mul(s_bf, half_lo, mask=m_src), tla.mul(s_bf, half_hi, mask=m_src))
                                    scaled = tla.max(tla.min(scaled, clamp_value), -clamp_value)
                                    bits = scaled.bitcast(tla.Int32)
                                    sign = tla.shift_left(tla.bitwise_and(tla.shift_right(bits, 31), v1), 7)
                                    abs_bits = tla.bitwise_and(bits, v7FFFFFFF)
                                    exp_field = tla.bitwise_and(tla.shift_right(abs_bits, 23), vFF)
                                    mant = tla.bitwise_and(abs_bits, v7FFFFF)
                                    if tla.const_expr(_outE5M2):
                                        mant2 = tla.shift_right(
                                            tla.add(tla.add(mant, tla.full(0x1FFFFF, tla.Int32)),
                                                    tla.bitwise_and(tla.shift_right(mant, 21), v1)), 21)
                                        code_exp = tla.sub(exp_field, 112)
                                        carry = tla.cmp(mant2, 4, "ge", mask=m_src)
                                        code_exp = tla.where(carry, tla.add(code_exp, 1), code_exp)
                                        mant2 = tla.where(carry, tla.sub(mant2, 4), mant2)
                                        normal_code = tla.bitwise_or(tla.shift_left(code_exp, 2), mant2)
                                        normal_code = tla.where(tla.cmp(code_exp, 31, "ge", mask=m_src), tla.full(0x7B, tla.Int32), normal_code)
                                        normal_code = tla.min(normal_code, tla.full(0x7B, tla.Int32))
                                        sub_f = tla.mul(tla.abs(scaled, mask=m_src), 65536.0)
                                        sub_i = sub_f.to(tla.Int32, trunc_i32, m_src)
                                        frac = tla.sub(sub_f, sub_i.to(tla.Float32, round_f32, m_src))
                                        up = tla.where(tla.cmp(frac, 0.5, "gt", mask=m_src), v1, zero32)
                                        up = tla.where(tla.cmp(frac, 0.5, "eq", mask=m_src), tla.bitwise_and(sub_i, v1), up)
                                        sub_code = tla.min(tla.add(sub_i, up), tla.full(4, tla.Int32))
                                        mag = tla.where(tla.cmp(exp_field, 113, "ge", mask=m_src), normal_code, sub_code)
                                    else:
                                        mant3 = tla.shift_right(
                                            tla.add(tla.add(mant, tla.full(0x7FFFF, tla.Int32)),
                                                    tla.bitwise_and(tla.shift_right(mant, 20), v1)), 20)
                                        code_exp = tla.sub(exp_field, 120)
                                        carry = tla.cmp(mant3, 8, "ge", mask=m_src)
                                        code_exp = tla.where(carry, tla.add(code_exp, 1), code_exp)
                                        mant3 = tla.where(carry, tla.sub(mant3, 8), mant3)
                                        normal_code = tla.bitwise_or(tla.shift_left(code_exp, 3), mant3)
                                        normal_code = tla.where(tla.cmp(code_exp, 16, "ge", mask=m_src), tla.full(0x7E, tla.Int32), normal_code)
                                        normal_code = tla.min(normal_code, tla.full(0x7E, tla.Int32))
                                        sub_f = tla.mul(tla.abs(scaled, mask=m_src), 512.0)
                                        sub_i = sub_f.to(tla.Int32, trunc_i32, m_src)
                                        frac = tla.sub(sub_f, sub_i.to(tla.Float32, round_f32, m_src))
                                        up = tla.where(tla.cmp(frac, 0.5, "gt", mask=m_src), v1, zero32)
                                        up = tla.where(tla.cmp(frac, 0.5, "eq", mask=m_src), tla.bitwise_and(sub_i, v1), up)
                                        sub_code = tla.min(tla.add(sub_i, up), tla.full(8, tla.Int32))
                                        mag = tla.where(tla.cmp(exp_field, 121, "ge", mask=m_src), normal_code, sub_code)
                                    code = tla.bitwise_or(sign, mag)
                                    even1, odd1 = tla.deinterleave(code, zero32)
                                    p0, p2 = tla.deinterleave(even1, zero32)
                                    p1, p3 = tla.deinterleave(odd1, zero32)
                                    word = tla.bitwise_or(
                                        tla.bitwise_or(p0, tla.shift_left(p1, 8)),
                                        tla.bitwise_or(tla.shift_left(p2, 16), tla.shift_left(p3, 24)),
                                    )
                                    y_chunk = tla.get_tile(ub_y_half, tla.make_coord(r, col // 4), tla.make_shape(1, 16))
                                    y_chunk.store(word, mask=m_i32_16)
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

                start_core_idx = (start_core_idx + mn_blocks) % core_num
            m_start = m_start + current_m

# -----------------------------------------------------------------------------------------------------------
# Host-side calling code: tensor binding, artifact memoization and single launch.
# -----------------------------------------------------------------------------------------------------------

# Memoized compiled artifacts keyed by constexpr variant (see _launch).
_ARTIFACT_CACHE: dict = {}

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
        b2d = b.reshape(groups * k, b_cols)
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
    return a_t, b_t, sa_t, sb_t, gl_t, m, n, k, groups, decode_a, decode_b


def _launch(a_t, b_t, sa_t, sb_t, gl_t, y_t, sy_t, trans_b, out_e5m2, decode_a, decode_b, a_dtype, b_dtype, block_num):
    tiling = GmmTilingParams()
    # The DSL disk-cache lookup costs ~60 ms per call; memoize the compiled
    # artifact by the constexpr variant so steady-state run() only pays launch.
    # a4w4 binds native FP4 (decode False), a8w4 decodes B (decode_b True), so
    # the decode flags plus operand dtypes distinguish artifacts; layouts stay
    # mark_layout_dynamic, so one artifact serves all shapes of a given variant.
    key = (trans_b, out_e5m2, decode_a, decode_b, a_dtype, b_dtype)
    artifact = _ARTIFACT_CACHE.get(key)
    if artifact is None:
        artifact = tla.compile(
            gmm_swiglu_mx_quant_kernel,
            a_t, b_t, sa_t, sb_t, gl_t, y_t, sy_t,
            tiling, trans_b, out_e5m2, decode_a, decode_b,
            options=NPU_ARCH,
        )
        _ARTIFACT_CACHE[key] = artifact
    artifact(a_t, b_t, sa_t, sb_t, gl_t, y_t, sy_t, block_num=block_num)
    torch.npu.synchronize()


def run(a, b, mx_scale_a, mx_scale_b, group_list, transB=True):
    """Definition entry point: launch the fused kernel, return (q, q_scale)."""
    device = int(a.device.index) if a.device.index is not None else 0
    torch.npu.set_device(device)
    built = _build_tensors(a, b, mx_scale_a, mx_scale_b, group_list, transB)
    a_t, b_t, sa_t, sb_t, gl_t, m, n, k, groups, decode_a, decode_b = built

    out_dtype = a.dtype if a.dtype in FP8_SEL else torch.float8_e4m3fn
    out_e5m2 = out_dtype == torch.float8_e5m2
    n_out = n // 2
    q = torch.zeros(m, n_out, dtype=out_dtype, device=a.device)
    # The vector epilogue packs each row's shared exponents into one Int32 per
    # 128 output columns, so the E8M0 byte buffer is padded to that grid.
    n_scale_words = _ceil_div(n_out, 4 * MX_OUTPUT_BLOCK_SIZE)
    q_scale_bytes = torch.zeros(m, n_scale_words * 4, dtype=torch.uint8, device=a.device)
    y_t = _bind_output(q.view(torch.int32), tla.arch.RowMajor, tla.Int32)
    sy_t = _bind_output(q_scale_bytes.view(torch.int32), tla.arch.RowMajor, tla.Int32)

    bn = max(1, int(torch.npu.get_device_properties(device).cube_core_num))
    _launch(a_t, b_t, sa_t, sb_t, gl_t, y_t, sy_t, transB, out_e5m2, decode_a, decode_b, a.dtype, b.dtype, bn)
    valid_scale_cols = _ceil_div(n_out, MX_OUTPUT_BLOCK_SIZE)
    q_scale = q_scale_bytes[:, :valid_scale_cols].contiguous().view(torch.float8_e8m0fnu)
    return q, q_scale
