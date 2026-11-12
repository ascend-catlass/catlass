# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch
import catlass.tla as tla
from catlass.tla.runtime import from_dlpack

TM = 128
TN = 128
TK = 64
L0_TK = 32
EVG_UB_STAGES = 2
UB_NODES = 3
VECTOR_ELE = 256


@tla.kernel
def matmul_add_chain_kernel(
    gm_q: tla.Tensor,
    gm_k: tla.Tensor,
    gm_v: tla.Tensor,
    gm_x1: tla.Tensor,
    gm_x2: tla.Tensor,
    gm_out: tla.Tensor,
    gm_wks: tla.Tensor,
    _tm: tla.Constexpr[int],
    _tn: tla.Constexpr[int],
    _tk: tla.Constexpr[int],
    _l0_tk: tla.Constexpr[int],
    _evg_stages: tla.Constexpr[int],
    UB_SLOT_ELEMS: tla.Constexpr[int],
    SIMD_LANES: tla.Constexpr[int],
) -> None:
    c0 = 0
    c1 = 1
    reg_m = 1
    byte_per_c0 = 32

    dtype_a = gm_q.ptr.dtype
    DTYPE_C = tla.Float32
    dtype_io = gm_x1.ptr.dtype

    m = gm_q.origin_shape[0]
    k1 = gm_q.origin_shape[1]
    n1 = gm_k.origin_shape[1]
    n2 = gm_v.origin_shape[1]

    mutex_l1a0 = tla.mutex("l1a0", id=0)
    mutex_l1b0 = tla.mutex("l1b0", id=1)
    mutex_l1a1 = tla.mutex("l1a1", id=2)
    mutex_l1b1 = tla.mutex("l1b1", id=3)
    mutex_l0a0 = tla.mutex("l0a0", id=4)
    mutex_l0b0 = tla.mutex("l0b0", id=5)
    mutex_l0a1 = tla.mutex("l0a1", id=6)
    mutex_l0b1 = tla.mutex("l0b1", id=7)
    mutex_l0c = tla.mutex("l0c", id=8)

    mutex_ub_aux0 = tla.mutex("ub_aux0", id=9)
    mutex_ub_out0 = tla.mutex("ub_out0", id=10)
    mutex_ub_aux1 = tla.mutex("ub_aux1", id=11)
    mutex_ub_out1 = tla.mutex("ub_out1", id=12)

    aic_finish = tla.cross_flag("aic_finish", mode=4)
    aiv_finish = tla.cross_flag("aiv_finish", mode=4)
    v1_published = tla.cross_flag("v1_published", mode=4)

    l1a0_ptr = tla.allocate(_tm * _tk, dtype_a, tla.AddressSpace.l1, 512)
    l1a1_ptr = tla.allocate(_tm * _tk, dtype_a, tla.AddressSpace.l1, 512)
    l1b0_ptr = tla.allocate(_tk * _tn, dtype_a, tla.AddressSpace.l1, 512)
    l1b1_ptr = tla.allocate(_tk * _tn, dtype_a, tla.AddressSpace.l1, 512)

    l0a0_ptr = tla.allocate(_tm * _l0_tk, dtype_a, tla.AddressSpace.l0a, 512)
    l0a1_ptr = tla.allocate(_tm * _l0_tk, dtype_a, tla.AddressSpace.l0a, 512)
    l0b0_ptr = tla.allocate(_l0_tk * _tn, dtype_a, tla.AddressSpace.l0b, 512)
    l0b1_ptr = tla.allocate(_l0_tk * _tn, dtype_a, tla.AddressSpace.l0b, 512)

    l0c_ptr = tla.allocate(_tm * _tn, DTYPE_C, tla.AddressSpace.l0c, 1024)

    ub_acc_ptr0 = tla.allocate(UB_SLOT_ELEMS, dtype_io, tla.AddressSpace.ub, 256)
    ub_aux_ptr0 = tla.allocate(UB_SLOT_ELEMS, dtype_io, tla.AddressSpace.ub, 256)
    ub_out_ptr0 = tla.allocate(UB_SLOT_ELEMS, dtype_io, tla.AddressSpace.ub, 256)
    ub_acc_ptr1 = ub_acc_ptr0
    ub_aux_ptr1 = ub_aux_ptr0
    ub_out_ptr1 = ub_out_ptr0
    if tla.const_expr(_evg_stages >= 2):
        ub_acc_ptr1 = tla.allocate(UB_SLOT_ELEMS, dtype_io, tla.AddressSpace.ub, 256)
        ub_aux_ptr1 = tla.allocate(UB_SLOT_ELEMS, dtype_io, tla.AddressSpace.ub, 256)
        ub_out_ptr1 = tla.allocate(UB_SLOT_ELEMS, dtype_io, tla.AddressSpace.ub, 256)

    grid_m = (m + _tm - 1) // _tm
    k1_l1_count = (k1 + _tk - 1) // _tk
    n1_count = (n1 + _tn - 1) // _tn
    k2_l1_count = (n1 + _tk - 1) // _tk
    n2_count = (n2 + _tn - 1) // _tn

    with tla.cube():
        l1_buf_idx = c0
        l0_buf_idx = c0

        block_range = tla.range(tla.arch.block_idx(), grid_m, tla.arch.block_num())
        for block_linear in block_range:
            m_tile = block_linear

            gm_q_strip = tla.tile_view(
                gm_q, tla.make_shape(_tm, k1), tla.make_coord(m_tile, c0)
            )
            for n1i in tla.range(c0, n1_count, c1):
                gm_k_tile = tla.tile_view(
                    gm_k, tla.make_shape(k1, _tn), tla.make_coord(c0, n1i)
                )
                gm_d_tile = tla.tile_view(
                    gm_wks, tla.make_shape(_tm, _tn), tla.make_coord(m_tile, n1i)
                )
                l0_c = tla.make_tensor_like(l0c_ptr, gm_d_tile)

                tla.cross_core_wait_flag(aiv_finish, tla.arch.FIX, aiv_id=0)
                tla.cross_core_wait_flag(aiv_finish, tla.arch.FIX, aiv_id=1)

                mutex_l0c.lock(pipe=tla.arch.CUBE)

                for k_l1 in tla.range(c0, k1_l1_count, c1):
                    gm_a_l1 = tla.tile_view(
                        gm_q_strip,
                        tla.make_shape(_tm, _tk),
                        tla.make_coord(c0, k_l1),
                    )
                    gm_b_l1 = tla.tile_view(
                        gm_k_tile,
                        tla.make_shape(_tk, _tn),
                        tla.make_coord(k_l1, c0),
                    )
                    l1_a = tla.make_tensor_like(
                        l1a0_ptr if (l1_buf_idx == c0) else l1a1_ptr, gm_a_l1
                    )
                    l1_b = tla.make_tensor_like(
                        l1b0_ptr if (l1_buf_idx == c0) else l1b1_ptr, gm_b_l1
                    )
                    if l1_buf_idx == c0:
                        mutex_l1a0.lock(pipe=tla.arch.MTE2)
                    else:
                        mutex_l1a1.lock(pipe=tla.arch.MTE2)
                    tla.copy(l1_a, gm_a_l1)
                    if l1_buf_idx == c0:
                        mutex_l1a0.unlock(pipe=tla.arch.MTE2)
                    else:
                        mutex_l1a1.unlock(pipe=tla.arch.MTE2)

                    if l1_buf_idx == c0:
                        mutex_l1b0.lock(pipe=tla.arch.MTE2)
                    else:
                        mutex_l1b1.lock(pipe=tla.arch.MTE2)
                    tla.copy(l1_b, gm_b_l1)
                    if l1_buf_idx == c0:
                        mutex_l1b0.unlock(pipe=tla.arch.MTE2)
                    else:
                        mutex_l1b1.unlock(pipe=tla.arch.MTE2)

                    k_l0_count = (l1_a.origin_shape[1] + _l0_tk - 1) // _l0_tk
                    for k_l0 in tla.range(c0, k_l0_count, c1):
                        l1_a_l0 = tla.tile_view(
                            l1_a,
                            tla.make_shape(_tm, _l0_tk),
                            tla.make_coord(c0, k_l0),
                        )
                        l1_b_l0 = tla.tile_view(
                            l1_b,
                            tla.make_shape(_l0_tk, _tn),
                            tla.make_coord(k_l0, c0),
                        )
                        l0_a = tla.make_tensor_like(
                            l0a0_ptr if (l0_buf_idx == c0) else l0a1_ptr, l1_a_l0
                        )
                        l0_b = tla.make_tensor_like(
                            l0b0_ptr if (l0_buf_idx == c0) else l0b1_ptr, l1_b_l0
                        )
                        if k_l0 == 0:
                            if l1_buf_idx == c0:
                                mutex_l1a0.lock(pipe=tla.arch.MTE1)
                            else:
                                mutex_l1a1.lock(pipe=tla.arch.MTE1)
                        if l0_buf_idx == c0:
                            mutex_l0a0.lock(pipe=tla.arch.MTE1)
                        else:
                            mutex_l0a1.lock(pipe=tla.arch.MTE1)
                        tla.copy(l0_a, l1_a_l0)
                        if l0_buf_idx == c0:
                            mutex_l0a0.unlock(pipe=tla.arch.MTE1)
                        else:
                            mutex_l0a1.unlock(pipe=tla.arch.MTE1)
                        if k_l0 == k_l0_count - 1:
                            if l1_buf_idx == c0:
                                mutex_l1a0.unlock(pipe=tla.arch.MTE1)
                            else:
                                mutex_l1a1.unlock(pipe=tla.arch.MTE1)

                        if k_l0 == 0:
                            if l1_buf_idx == c0:
                                mutex_l1b0.lock(pipe=tla.arch.MTE1)
                            else:
                                mutex_l1b1.lock(pipe=tla.arch.MTE1)
                        if l0_buf_idx == c0:
                            mutex_l0b0.lock(pipe=tla.arch.MTE1)
                        else:
                            mutex_l0b1.lock(pipe=tla.arch.MTE1)
                        tla.copy(l0_b, l1_b_l0)
                        if l0_buf_idx == c0:
                            mutex_l0b0.unlock(pipe=tla.arch.MTE1)
                        else:
                            mutex_l0b1.unlock(pipe=tla.arch.MTE1)
                        if k_l0 == k_l0_count - 1:
                            if l1_buf_idx == c0:
                                mutex_l1b0.unlock(pipe=tla.arch.MTE1)
                            else:
                                mutex_l1b1.unlock(pipe=tla.arch.MTE1)

                        if l0_buf_idx == c0:
                            mutex_l0a0.lock(pipe=tla.arch.CUBE)
                            mutex_l0b0.lock(pipe=tla.arch.CUBE)
                        else:
                            mutex_l0a1.lock(pipe=tla.arch.CUBE)
                            mutex_l0b1.lock(pipe=tla.arch.CUBE)

                        unit_flag = 0
                        if (k_l1 == k1_l1_count - 1) and (k_l0 == k_l0_count - 1):
                            unit_flag = 0b11
                        else:
                            unit_flag = 0b10
                        init_c = True if k_l1 == 0 and k_l0 == 0 else False
                        tla.mmad(l0_c, l0_a, l0_b, init_c=init_c, unit_flag=unit_flag)
                        if l0_buf_idx == c0:
                            mutex_l0a0.unlock(pipe=tla.arch.CUBE)
                            mutex_l0b0.unlock(pipe=tla.arch.CUBE)
                        else:
                            mutex_l0a1.unlock(pipe=tla.arch.CUBE)
                            mutex_l0b1.unlock(pipe=tla.arch.CUBE)
                        l0_buf_idx = c1 - l0_buf_idx
                    l1_buf_idx = c1 - l1_buf_idx

                mutex_l0c.unlock(pipe=tla.arch.CUBE)
                mutex_l0c.lock(pipe=tla.arch.FIX)
                tla.copy(
                    gm_d_tile,
                    l0_c,
                    tla.params.CopyL0C2DstParams(unit_flag=0b11),
                )
                mutex_l0c.unlock(pipe=tla.arch.FIX)
                tla.cross_core_set_flag(aic_finish, tla.arch.FIX, aiv_id=0)
                tla.cross_core_set_flag(aic_finish, tla.arch.FIX, aiv_id=1)

            tla.cross_core_wait_flag(v1_published, tla.arch.MTE2, aiv_id=0)
            tla.cross_core_wait_flag(v1_published, tla.arch.MTE2, aiv_id=1)

            gm_w_strip = tla.tile_view(
                gm_wks, tla.make_shape(_tm, n1), tla.make_coord(m_tile, c0)
            )
            for n2i in tla.range(c0, n2_count, c1):
                gm_v_tile = tla.tile_view(
                    gm_v, tla.make_shape(n1, _tn), tla.make_coord(c0, n2i)
                )
                gm_o_tile = tla.tile_view(
                    gm_out, tla.make_shape(_tm, _tn), tla.make_coord(m_tile, n2i)
                )
                l0_c2 = tla.make_tensor_like(l0c_ptr, gm_o_tile)

                tla.cross_core_wait_flag(aiv_finish, tla.arch.FIX, aiv_id=0)
                tla.cross_core_wait_flag(aiv_finish, tla.arch.FIX, aiv_id=1)

                mutex_l0c.lock(pipe=tla.arch.CUBE)

                for k2_l1 in tla.range(c0, k2_l1_count, c1):
                    gm_a2_l1 = tla.tile_view(
                        gm_w_strip,
                        tla.make_shape(_tm, _tk),
                        tla.make_coord(c0, k2_l1),
                    )
                    gm_b2_l1 = tla.tile_view(
                        gm_v_tile,
                        tla.make_shape(_tk, _tn),
                        tla.make_coord(k2_l1, c0),
                    )
                    l1_a2 = tla.make_tensor_like(
                        l1a0_ptr if (l1_buf_idx == c0) else l1a1_ptr, gm_a2_l1
                    )
                    l1_b2 = tla.make_tensor_like(
                        l1b0_ptr if (l1_buf_idx == c0) else l1b1_ptr, gm_b2_l1
                    )
                    if l1_buf_idx == c0:
                        mutex_l1a0.lock(pipe=tla.arch.MTE2)
                    else:
                        mutex_l1a1.lock(pipe=tla.arch.MTE2)
                    tla.copy(l1_a2, gm_a2_l1)
                    if l1_buf_idx == c0:
                        mutex_l1a0.unlock(pipe=tla.arch.MTE2)
                    else:
                        mutex_l1a1.unlock(pipe=tla.arch.MTE2)

                    if l1_buf_idx == c0:
                        mutex_l1b0.lock(pipe=tla.arch.MTE2)
                    else:
                        mutex_l1b1.lock(pipe=tla.arch.MTE2)
                    tla.copy(l1_b2, gm_b2_l1)
                    if l1_buf_idx == c0:
                        mutex_l1b0.unlock(pipe=tla.arch.MTE2)
                    else:
                        mutex_l1b1.unlock(pipe=tla.arch.MTE2)

                    k2_l0_count = (l1_a2.origin_shape[1] + _l0_tk - 1) // _l0_tk
                    for k2_l0 in tla.range(c0, k2_l0_count, c1):
                        l1_a2_l0 = tla.tile_view(
                            l1_a2,
                            tla.make_shape(_tm, _l0_tk),
                            tla.make_coord(c0, k2_l0),
                        )
                        l1_b2_l0 = tla.tile_view(
                            l1_b2,
                            tla.make_shape(_l0_tk, _tn),
                            tla.make_coord(k2_l0, c0),
                        )
                        l0_a2 = tla.make_tensor_like(
                            l0a0_ptr if (l0_buf_idx == c0) else l0a1_ptr, l1_a2_l0
                        )
                        l0_b2 = tla.make_tensor_like(
                            l0b0_ptr if (l0_buf_idx == c0) else l0b1_ptr, l1_b2_l0
                        )
                        if k2_l0 == 0:
                            if l1_buf_idx == c0:
                                mutex_l1a0.lock(pipe=tla.arch.MTE1)
                            else:
                                mutex_l1a1.lock(pipe=tla.arch.MTE1)
                        if l0_buf_idx == c0:
                            mutex_l0a0.lock(pipe=tla.arch.MTE1)
                        else:
                            mutex_l0a1.lock(pipe=tla.arch.MTE1)
                        tla.copy(l0_a2, l1_a2_l0)
                        if l0_buf_idx == c0:
                            mutex_l0a0.unlock(pipe=tla.arch.MTE1)
                        else:
                            mutex_l0a1.unlock(pipe=tla.arch.MTE1)
                        if k2_l0 == k2_l0_count - 1:
                            if l1_buf_idx == c0:
                                mutex_l1a0.unlock(pipe=tla.arch.MTE1)
                            else:
                                mutex_l1a1.unlock(pipe=tla.arch.MTE1)

                        if k2_l0 == 0:
                            if l1_buf_idx == c0:
                                mutex_l1b0.lock(pipe=tla.arch.MTE1)
                            else:
                                mutex_l1b1.lock(pipe=tla.arch.MTE1)
                        if l0_buf_idx == c0:
                            mutex_l0b0.lock(pipe=tla.arch.MTE1)
                        else:
                            mutex_l0b1.lock(pipe=tla.arch.MTE1)
                        tla.copy(l0_b2, l1_b2_l0)
                        if l0_buf_idx == c0:
                            mutex_l0b0.unlock(pipe=tla.arch.MTE1)
                        else:
                            mutex_l0b1.unlock(pipe=tla.arch.MTE1)
                        if k2_l0 == k2_l0_count - 1:
                            if l1_buf_idx == c0:
                                mutex_l1b0.unlock(pipe=tla.arch.MTE1)
                            else:
                                mutex_l1b1.unlock(pipe=tla.arch.MTE1)

                        if l0_buf_idx == c0:
                            mutex_l0a0.lock(pipe=tla.arch.CUBE)
                            mutex_l0b0.lock(pipe=tla.arch.CUBE)
                        else:
                            mutex_l0a1.lock(pipe=tla.arch.CUBE)
                            mutex_l0b1.lock(pipe=tla.arch.CUBE)

                        unit_flag2 = 0
                        if (k2_l1 == k2_l1_count - 1) and (k2_l0 == k2_l0_count - 1):
                            unit_flag2 = 0b11
                        else:
                            unit_flag2 = 0b10
                        init_c2 = True if k2_l1 == 0 and k2_l0 == 0 else False
                        tla.mmad(
                            l0_c2, l0_a2, l0_b2, init_c=init_c2, unit_flag=unit_flag2
                        )
                        if l0_buf_idx == c0:
                            mutex_l0a0.unlock(pipe=tla.arch.CUBE)
                            mutex_l0b0.unlock(pipe=tla.arch.CUBE)
                        else:
                            mutex_l0a1.unlock(pipe=tla.arch.CUBE)
                            mutex_l0b1.unlock(pipe=tla.arch.CUBE)
                        l0_buf_idx = c1 - l0_buf_idx
                    l1_buf_idx = c1 - l1_buf_idx

                mutex_l0c.unlock(pipe=tla.arch.CUBE)
                mutex_l0c.lock(pipe=tla.arch.FIX)
                tla.copy(
                    gm_o_tile,
                    l0_c2,
                    tla.params.CopyL0C2DstParams(unit_flag=0b11),
                )
                mutex_l0c.unlock(pipe=tla.arch.FIX)
                tla.cross_core_set_flag(aic_finish, tla.arch.FIX, aiv_id=0)
                tla.cross_core_set_flag(aic_finish, tla.arch.FIX, aiv_id=1)

        tla.pipe_barrier(tla.pipes.ALL)

    with tla.vector():
        aiv_sub_idx = tla.arch.sub_block_idx()
        block_range = tla.range(tla.arch.block_idx(), grid_m, tla.arch.block_num())

        tla.cross_core_set_flag(aiv_finish, tla.arch.VECTOR, aiv_id=0)
        tla.cross_core_set_flag(aiv_finish, tla.arch.VECTOR, aiv_id=1)

        for block_linear in block_range:
            m_tile = block_linear
            ub_stage_idx = c0

            for n1i in tla.range(c0, n1_count, c1):
                tla.cross_core_wait_flag(aic_finish, tla.arch.MTE2, aiv_id=0)
                tla.cross_core_wait_flag(aic_finish, tla.arch.MTE2, aiv_id=1)

                gm_x_tile = tla.tile_view(
                    gm_x1, tla.make_shape(_tm, _tn), tla.make_coord(m_tile, n1i)
                )
                gm_d_tile = tla.tile_view(
                    gm_wks, tla.make_shape(_tm, _tn), tla.make_coord(m_tile, n1i)
                )
                actual_m = gm_x_tile.origin_shape[0]
                actual_n = gm_x_tile.origin_shape[1]
                aiv_m_req = (actual_m + 1) // 2
                gm_x_half = tla.tile_view(
                    gm_x_tile,
                    tla.make_shape(aiv_m_req, actual_n),
                    tla.make_coord(aiv_sub_idx, c0),
                )
                gm_d_half = tla.tile_view(
                    gm_d_tile,
                    tla.make_shape(aiv_m_req, actual_n),
                    tla.make_coord(aiv_sub_idx, c0),
                )
                aiv_m = gm_x_half.origin_shape[0]
                aiv_n = gm_x_half.origin_shape[1]

                ub_row_stride = ((aiv_n + byte_per_c0 - 1) // byte_per_c0) * byte_per_c0
                max_rows = UB_SLOT_ELEMS // ub_row_stride
                if max_rows < 1:
                    max_rows = 1
                n_epilogue_row_tiles = (aiv_m + max_rows - 1) // max_rows

                for tile_i in tla.range(c0, n_epilogue_row_tiles, c1):
                    gm_acc_rt = tla.tile_view(
                        gm_d_half,
                        tla.make_shape(max_rows, aiv_n),
                        tla.make_coord(tile_i, c0),
                    )
                    gm_x_rt = tla.tile_view(
                        gm_x_half,
                        tla.make_shape(max_rows, aiv_n),
                        tla.make_coord(tile_i, c0),
                    )
                    tile_rows = gm_x_rt.origin_shape[0]
                    tile_cols = gm_x_rt.origin_shape[1]
                    ub_layout = tla.make_layout(
                        tla.make_shape(tile_rows, tile_cols),
                        tla.make_stride(ub_row_stride, 1),
                        layoutTag=tla.arch.RowMajor,
                    )
                    n_simd_strips = (tile_cols + SIMD_LANES - 1) // SIMD_LANES
                    if tla.const_expr(_evg_stages >= 2):
                        use_ub_stage1 = ub_stage_idx != c0
                    else:
                        use_ub_stage1 = False
                    ub_acc = tla.make_tensor(
                        ub_acc_ptr1 if use_ub_stage1 else ub_acc_ptr0, ub_layout
                    )
                    ub_addend = tla.make_tensor(
                        ub_aux_ptr1 if use_ub_stage1 else ub_aux_ptr0, ub_layout
                    )
                    ub_result = tla.make_tensor(
                        ub_out_ptr1 if use_ub_stage1 else ub_out_ptr0, ub_layout
                    )

                    if use_ub_stage1:
                        mutex_ub_aux1.lock(pipe=tla.arch.MTE2)
                    else:
                        mutex_ub_aux0.lock(pipe=tla.arch.MTE2)
                    tla.copy(ub_acc, gm_acc_rt)
                    tla.copy(ub_addend, gm_x_rt)
                    if use_ub_stage1:
                        mutex_ub_aux1.unlock(pipe=tla.arch.MTE2)
                        mutex_ub_aux1.lock(pipe=tla.arch.VECTOR)
                        mutex_ub_out1.lock(pipe=tla.arch.VECTOR)
                    else:
                        mutex_ub_aux0.unlock(pipe=tla.arch.MTE2)
                        mutex_ub_aux0.lock(pipe=tla.arch.VECTOR)
                        mutex_ub_out0.lock(pipe=tla.arch.VECTOR)

                    with tla.vec.func(mode="simd"):
                        for row_i in tla.range(c0, tile_rows, c1):
                            remaining = tile_cols
                            for col_j in tla.range(c0, n_simd_strips, c1):
                                c_chunk = tla.tile_view(
                                    ub_acc,
                                    tla.make_shape(reg_m, SIMD_LANES),
                                    tla.make_coord(row_i, col_j),
                                )
                                addend_chunk = tla.tile_view(
                                    ub_addend,
                                    tla.make_shape(reg_m, SIMD_LANES),
                                    tla.make_coord(row_i, col_j),
                                )
                                result_chunk = tla.tile_view(
                                    ub_result,
                                    tla.make_shape(reg_m, SIMD_LANES),
                                    tla.make_coord(row_i, col_j),
                                )
                                tail, remaining = tla.update_mask(
                                    remaining, dtype=dtype_io
                                )
                                result_chunk.store(
                                    c_chunk.load() + addend_chunk.load(), mask=tail
                                )

                    if use_ub_stage1:
                        mutex_ub_aux1.unlock(pipe=tla.arch.VECTOR)
                        mutex_ub_out1.unlock(pipe=tla.arch.VECTOR)
                        mutex_ub_out1.lock(pipe=tla.arch.MTE3)
                    else:
                        mutex_ub_aux0.unlock(pipe=tla.arch.VECTOR)
                        mutex_ub_out0.unlock(pipe=tla.arch.VECTOR)
                        mutex_ub_out0.lock(pipe=tla.arch.MTE3)
                    tla.copy(gm_acc_rt, ub_result)
                    if use_ub_stage1:
                        mutex_ub_out1.unlock(pipe=tla.arch.MTE3)
                    else:
                        mutex_ub_out0.unlock(pipe=tla.arch.MTE3)

                    if tla.const_expr(_evg_stages >= 2):
                        ub_stage_idx = c1 - ub_stage_idx

                tla.cross_core_set_flag(aiv_finish, tla.arch.VECTOR, aiv_id=0)
                tla.cross_core_set_flag(aiv_finish, tla.arch.VECTOR, aiv_id=1)

            tla.cross_core_set_flag(v1_published, tla.arch.MTE3, aiv_id=0)
            tla.cross_core_set_flag(v1_published, tla.arch.MTE3, aiv_id=1)

            for n2i in tla.range(c0, n2_count, c1):
                tla.cross_core_wait_flag(aic_finish, tla.arch.MTE2, aiv_id=0)
                tla.cross_core_wait_flag(aic_finish, tla.arch.MTE2, aiv_id=1)

                gm_x2_tile = tla.tile_view(
                    gm_x2, tla.make_shape(_tm, _tn), tla.make_coord(m_tile, n2i)
                )
                gm_o_tile_v = tla.tile_view(
                    gm_out, tla.make_shape(_tm, _tn), tla.make_coord(m_tile, n2i)
                )
                actual2_m = gm_x2_tile.origin_shape[0]
                actual2_n = gm_x2_tile.origin_shape[1]
                aiv2_m_req = (actual2_m + 1) // 2
                gm_x2_half = tla.tile_view(
                    gm_x2_tile,
                    tla.make_shape(aiv2_m_req, actual2_n),
                    tla.make_coord(aiv_sub_idx, c0),
                )
                gm_o_half = tla.tile_view(
                    gm_o_tile_v,
                    tla.make_shape(aiv2_m_req, actual2_n),
                    tla.make_coord(aiv_sub_idx, c0),
                )
                aiv2_m = gm_x2_half.origin_shape[0]
                aiv2_n = gm_x2_half.origin_shape[1]

                ub2_row_stride = (
                    (aiv2_n + byte_per_c0 - 1) // byte_per_c0
                ) * byte_per_c0
                max_rows2 = UB_SLOT_ELEMS // ub2_row_stride
                if max_rows2 < 1:
                    max_rows2 = 1
                n_row_tiles2 = (aiv2_m + max_rows2 - 1) // max_rows2

                for tile2_i in tla.range(c0, n_row_tiles2, c1):
                    gm_acc2_rt = tla.tile_view(
                        gm_o_half,
                        tla.make_shape(max_rows2, aiv2_n),
                        tla.make_coord(tile2_i, c0),
                    )
                    gm_x2_rt = tla.tile_view(
                        gm_x2_half,
                        tla.make_shape(max_rows2, aiv2_n),
                        tla.make_coord(tile2_i, c0),
                    )
                    tile_rows2 = gm_x2_rt.origin_shape[0]
                    tile_cols2 = gm_x2_rt.origin_shape[1]
                    ub_layout2 = tla.make_layout(
                        tla.make_shape(tile_rows2, tile_cols2),
                        tla.make_stride(ub2_row_stride, 1),
                        layoutTag=tla.arch.RowMajor,
                    )
                    n_simd_strips2 = (tile_cols2 + SIMD_LANES - 1) // SIMD_LANES
                    if tla.const_expr(_evg_stages >= 2):
                        use_stage1_p2 = ub_stage_idx != c0
                    else:
                        use_stage1_p2 = False
                    ub_acc2 = tla.make_tensor(
                        ub_acc_ptr1 if use_stage1_p2 else ub_acc_ptr0, ub_layout2
                    )
                    ub_addend2 = tla.make_tensor(
                        ub_aux_ptr1 if use_stage1_p2 else ub_aux_ptr0, ub_layout2
                    )
                    ub_result2 = tla.make_tensor(
                        ub_out_ptr1 if use_stage1_p2 else ub_out_ptr0, ub_layout2
                    )

                    if use_stage1_p2:
                        mutex_ub_aux1.lock(pipe=tla.arch.MTE2)
                    else:
                        mutex_ub_aux0.lock(pipe=tla.arch.MTE2)
                    tla.copy(ub_acc2, gm_acc2_rt)
                    tla.copy(ub_addend2, gm_x2_rt)
                    if use_stage1_p2:
                        mutex_ub_aux1.unlock(pipe=tla.arch.MTE2)
                        mutex_ub_aux1.lock(pipe=tla.arch.VECTOR)
                        mutex_ub_out1.lock(pipe=tla.arch.VECTOR)
                    else:
                        mutex_ub_aux0.unlock(pipe=tla.arch.MTE2)
                        mutex_ub_aux0.lock(pipe=tla.arch.VECTOR)
                        mutex_ub_out0.lock(pipe=tla.arch.VECTOR)

                    with tla.vec.func(mode="simd"):
                        for row2_i in tla.range(c0, tile_rows2, c1):
                            remaining2 = tile_cols2
                            for col2_j in tla.range(c0, n_simd_strips2, c1):
                                c2_chunk = tla.tile_view(
                                    ub_acc2,
                                    tla.make_shape(reg_m, SIMD_LANES),
                                    tla.make_coord(row2_i, col2_j),
                                )
                                addend2_chunk = tla.tile_view(
                                    ub_addend2,
                                    tla.make_shape(reg_m, SIMD_LANES),
                                    tla.make_coord(row2_i, col2_j),
                                )
                                result2_chunk = tla.tile_view(
                                    ub_result2,
                                    tla.make_shape(reg_m, SIMD_LANES),
                                    tla.make_coord(row2_i, col2_j),
                                )
                                tail2, remaining2 = tla.update_mask(
                                    remaining2, dtype=dtype_io
                                )
                                result2_chunk.store(
                                    c2_chunk.load() + addend2_chunk.load(), mask=tail2
                                )

                    if use_stage1_p2:
                        mutex_ub_aux1.unlock(pipe=tla.arch.VECTOR)
                        mutex_ub_out1.unlock(pipe=tla.arch.VECTOR)
                        mutex_ub_out1.lock(pipe=tla.arch.MTE3)
                    else:
                        mutex_ub_aux0.unlock(pipe=tla.arch.VECTOR)
                        mutex_ub_out0.unlock(pipe=tla.arch.VECTOR)
                        mutex_ub_out0.lock(pipe=tla.arch.MTE3)
                    tla.copy(gm_acc2_rt, ub_result2)
                    if use_stage1_p2:
                        mutex_ub_out1.unlock(pipe=tla.arch.MTE3)
                    else:
                        mutex_ub_out0.unlock(pipe=tla.arch.MTE3)

                    if tla.const_expr(_evg_stages >= 2):
                        ub_stage_idx = c1 - ub_stage_idx

                tla.cross_core_set_flag(aiv_finish, tla.arch.VECTOR, aiv_id=0)
                tla.cross_core_set_flag(aiv_finish, tla.arch.VECTOR, aiv_id=1)

        tla.pipe_barrier(tla.pipes.ALL)


_COMPILE_CACHE = {}
_TENSOR_CACHE = {}
_PROPS_CACHE = {}


def _make_tla_tensor(buf):
    key = (buf.data_ptr(), tuple(buf.shape), buf.dtype)
    wrapped = _TENSOR_CACHE.get(key)
    if wrapped is None:
        if len(_TENSOR_CACHE) > 4096:
            _TENSOR_CACHE.clear()
        wrapped = from_dlpack(
            buf.contiguous(), layout_tag=tla.arch.RowMajor
        ).mark_layout_dynamic()
        _TENSOR_CACHE[key] = wrapped
    return wrapped


def _cube_core_num(dev_idx: int) -> int:
    cached = _PROPS_CACHE.get(dev_idx)
    if cached is None:
        cached = max(1, int(torch.npu.get_device_properties(dev_idx).cube_core_num))
        _PROPS_CACHE[dev_idx] = cached
    return cached


def _ub_slot_elems(elem_bytes: int) -> int:
    budget = tla.arch.get_capacity_in_bytes(tla.AddressSpace.ub)
    slot = budget // UB_NODES // EVG_UB_STAGES // elem_bytes
    align_elems = 256 // elem_bytes
    return slot // align_elems * align_elems


def run(q, k, v, x1, x2):
    m, k1 = q.shape
    n1 = k.shape[1]
    n2 = v.shape[1]
    device = q.device
    elem_bytes = q.element_size()

    key = (int(m), int(k1), int(n1), int(n2), q.dtype)
    entry = _COMPILE_CACHE.get(key)
    if entry is None:
        out = torch.empty((m, n2), dtype=q.dtype, device=device)
        wks = torch.empty((m, n1), dtype=q.dtype, device=device)
        tensors = tuple(_make_tla_tensor(t) for t in (q, k, v, x1, x2, out, wks))
        artifact = tla.compile(
            matmul_add_chain_kernel,
            *tensors,
            TM,
            TN,
            TK,
            L0_TK,
            EVG_UB_STAGES,
            _ub_slot_elems(elem_bytes),
            VECTOR_ELE // elem_bytes,
            options=(
                "--npu-arch 3510"
                " --cce-disable-asc-reserved-ubuf"
                " --cce-disable-vf-stack-reserved-ubuf"
            ),
        )
        entry = (artifact, out, wks)
        _COMPILE_CACHE[key] = entry
    artifact, out, wks = entry

    tensors = tuple(_make_tla_tensor(t) for t in (q, k, v, x1, x2, out, wks))
    grid_m = (m + TM - 1) // TM
    dev_idx = device.index if device.index is not None else torch.npu.current_device()
    artifact(*tensors, block_num=min(grid_m, _cube_core_num(dev_idx)))
    return out
