# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import catlass.tla as tla

# Tile shape.  BM/BN is the cube output tile, BH the L1 K tile, SK the L0 K step.
BM = 256
BN = 256
BH = 128
# SK=64 fills L0A/L0B exactly when double buffered (2 x 256x64x2B = 64 KB each) and
# halves the number of inner-loop iterations versus SK=32, which matters because the
# per-tile cube-side scalar cost is a measurable share of the wall.  SK=128 does not
# fit the L0 double buffer and regresses by 20%.
SK = 64
ROWS_PER_SUB = BM // 2  # one AIV owns half of the L0C rows (SPLIT_M)
VL = 64  # fp32 vector register lanes

# Finite stand-in for -inf: exp(MIN_F - max) underflows to 0 without producing NaN.
MIN_F = -3.4028234663852886e38


@tla.kernel
def msrs_main_kernel(
    x: tla.Tensor,  # [M, H] bf16 RowMajor
    w: tla.Tensor,  # [H, V] bf16 RowMajor (host pre-transposes [V, H])
    tile_max: tla.Tensor,  # [M, NV] fp32 per V-segment partial max
    tile_sum: tla.Tensor,  # [M, NV] fp32 per V-segment partial exp-sum
    inv_temperature: float,
    m_real: int,
    v_real: int,
    NV: tla.Constexpr[int],
) -> None:
    c0 = 0
    c1 = 1

    m = x.origin_shape[0]
    h = x.origin_shape[1]
    v = w.origin_shape[1]

    grid_m = (m + BM - 1) // BM
    h_tiles = (h + BH - 1) // BH
    k_sub = (BH + SK - 1) // SK
    nv_tiles = tla.as_numeric((v + BN - 1) // BN)
    tiles_per_seg = (nv_tiles + NV - 1) // NV
    total_blocks = grid_m * NV

    dtype_in = x.ptr.dtype

    # C/V crossing: one cross flag handed back and forth (VECTOR -> FIX -> VECTOR).
    exchange = tla.cross_flag("exchange")
    # Same-core buffers are shared by their producer and consumer pipes.
    mutex_l1a0 = tla.mutex("l1a0", id=0)
    mutex_l1b0 = tla.mutex("l1b0", id=1)
    mutex_l1a1 = tla.mutex("l1a1", id=2)
    mutex_l1b1 = tla.mutex("l1b1", id=3)
    mutex_l0a0 = tla.mutex("l0a0", id=4)
    mutex_l0b0 = tla.mutex("l0b0", id=5)
    mutex_l0a1 = tla.mutex("l0a1", id=6)
    mutex_l0b1 = tla.mutex("l0b1", id=7)
    mutex_l0c = tla.mutex("l0c", id=8)

    l1a0_ptr = tla.allocate(BM * BH, dtype_in, tla.AddressSpace.l1, 512)
    l1b0_ptr = tla.allocate(BH * BN, dtype_in, tla.AddressSpace.l1, 512)
    l1a1_ptr = tla.allocate(BM * BH, dtype_in, tla.AddressSpace.l1, 512)
    l1b1_ptr = tla.allocate(BH * BN, dtype_in, tla.AddressSpace.l1, 512)
    l0a0_ptr = tla.allocate(BM * SK, dtype_in, tla.AddressSpace.l0a, 512)
    l0b0_ptr = tla.allocate(SK * BN, dtype_in, tla.AddressSpace.l0b, 512)
    l0a1_ptr = tla.allocate(BM * SK, dtype_in, tla.AddressSpace.l0a, 512)
    l0b1_ptr = tla.allocate(SK * BN, dtype_in, tla.AddressSpace.l0b, 512)
    l0c_ptr = tla.allocate(BM * BN, tla.Float32, tla.AddressSpace.l0c, 512)

    # Per-AIV UB: one SPLIT_M half of the L0C tile plus the statistics working set.
    ubz_ptr = tla.allocate(ROWS_PER_SUB * BN, tla.Float32, tla.AddressSpace.ub, 256)
    ub_rmax_ptr = tla.allocate(ROWS_PER_SUB, tla.Float32, tla.AddressSpace.ub, 256)
    ub_rsum_ptr = tla.allocate(ROWS_PER_SUB, tla.Float32, tla.AddressSpace.ub, 256)

    # ---------------------------------------------------------------- cube: GEMM
    # Complexity per tile: O(BM * BN * H) MAC; the statistics stay on the AIV side.
    with tla.cube():
        ub_z_full = tla.make_tensor(
            ubz_ptr, tla.make_layout(tla.make_shape(BM, BN), tla.make_stride(BN, 1))
        )
        l0_c = tla.make_tensor_like(l0c_ptr, ub_z_full, tla.arch.L0Clayout)

        # Two L1 stages so the GM->L1 load of the next K tile runs under the
        # MTE1 + MAC of the current one (L1 is the pipeline bottleneck otherwise).
        for blk in tla.range(tla.arch.block_idx(), total_blocks, tla.arch.block_num()):
            # Segment-major: consecutive blocks are consecutive M tiles inside one
            # V segment, so all concurrently running cores sweep the same rhs
            # stream together (L2 sharing) instead of 28 disjoint 160 MB regions.
            seg = blk // grid_m
            mb = blk - seg * grid_m
            vt_lo = seg * tiles_per_seg
            vt_hi = vt_lo + tiles_per_seg
            if vt_hi > nv_tiles:
                vt_hi = nv_tiles

            if vt_lo < vt_hi:
                for vt in tla.range(vt_lo, vt_hi):
                    # Prologue: stage the first K tile into L1 buffer 0.  The L0C
                    # mutex lock is deferred to just before the first MAC so this
                    # GM->L1 load overlaps the previous tile's FixPipe drain.
                    gm_a_p = tla.tile_view(
                        x, tla.make_shape(BM, BH), tla.make_coord(mb, c0)
                    )
                    gm_b_p = tla.tile_view(
                        w, tla.make_shape(BH, BN), tla.make_coord(c0, vt)
                    )
                    l1_a_p = tla.make_tensor_like(l1a0_ptr, gm_a_p, tla.arch.zN)
                    l1_b_p = tla.make_tensor_like(l1b0_ptr, gm_b_p, tla.arch.zN)
                    mutex_l1a0.lock(pipe=tla.arch.MTE2)
                    mutex_l1b0.lock(pipe=tla.arch.MTE2)
                    tla.copy(l1_a_p, gm_a_p)
                    tla.copy(l1_b_p, gm_b_p)
                    mutex_l1a0.unlock(pipe=tla.arch.MTE2)
                    mutex_l1b0.unlock(pipe=tla.arch.MTE2)

                    # First MAC of this V tile must not start before the previous
                    # tile's L0C->UB copy has consumed L0C.
                    mutex_l0c.lock(pipe=tla.arch.CUBE)

                    buf = tla.as_numeric(c0)
                    for hb in tla.range(c0, h_tiles, c1):
                        if buf == c0:
                            mutex_l1a0.lock(pipe=tla.arch.MTE1)
                            mutex_l1b0.lock(pipe=tla.arch.MTE1)
                        else:
                            mutex_l1a1.lock(pipe=tla.arch.MTE1)
                            mutex_l1b1.lock(pipe=tla.arch.MTE1)

                        # Prefetch the next K tile into the other stage.
                        if hb + c1 < h_tiles:
                            gm_a_n = tla.tile_view(
                                x,
                                tla.make_shape(BM, BH),
                                tla.make_coord(mb, hb + c1),
                            )
                            gm_b_n = tla.tile_view(
                                w,
                                tla.make_shape(BH, BN),
                                tla.make_coord(hb + c1, vt),
                            )
                            next_a_ptr = l1a1_ptr if buf == c0 else l1a0_ptr
                            next_b_ptr = l1b1_ptr if buf == c0 else l1b0_ptr
                            l1_a_n = tla.make_tensor_like(
                                next_a_ptr, gm_a_n, tla.arch.zN
                            )
                            l1_b_n = tla.make_tensor_like(
                                next_b_ptr, gm_b_n, tla.arch.zN
                            )
                            if buf == c0:
                                mutex_l1a1.lock(pipe=tla.arch.MTE2)
                                mutex_l1b1.lock(pipe=tla.arch.MTE2)
                            else:
                                mutex_l1a0.lock(pipe=tla.arch.MTE2)
                                mutex_l1b0.lock(pipe=tla.arch.MTE2)
                            tla.copy(l1_a_n, gm_a_n)
                            tla.copy(l1_b_n, gm_b_n)
                            if buf == c0:
                                mutex_l1a1.unlock(pipe=tla.arch.MTE2)
                                mutex_l1b1.unlock(pipe=tla.arch.MTE2)
                            else:
                                mutex_l1a0.unlock(pipe=tla.arch.MTE2)
                                mutex_l1b0.unlock(pipe=tla.arch.MTE2)

                        cur_a_ptr = l1a0_ptr if buf == c0 else l1a1_ptr
                        cur_b_ptr = l1b0_ptr if buf == c0 else l1b1_ptr
                        gm_a_c = tla.tile_view(
                            x, tla.make_shape(BM, BH), tla.make_coord(mb, hb)
                        )
                        gm_b_c = tla.tile_view(
                            w, tla.make_shape(BH, BN), tla.make_coord(hb, vt)
                        )
                        l1_a = tla.make_tensor_like(cur_a_ptr, gm_a_c, tla.arch.zN)
                        l1_b = tla.make_tensor_like(cur_b_ptr, gm_b_c, tla.arch.zN)

                        # L0 double buffer: stage chunk 0, then prefetch the next
                        # chunk while the MAC consumes the current one.
                        l1_a_p0 = tla.tile_view(
                            l1_a, tla.make_shape(BM, SK), tla.make_coord(c0, c0)
                        )
                        l1_b_p0 = tla.tile_view(
                            l1_b, tla.make_shape(SK, BN), tla.make_coord(c0, c0)
                        )
                        l0_a_p = tla.make_tensor_like(l0a0_ptr, l1_a_p0, tla.arch.zN)
                        l0_b_p = tla.make_tensor_like(l0b0_ptr, l1_b_p0, tla.arch.nZ)
                        mutex_l0a0.lock(pipe=tla.arch.MTE1)
                        mutex_l0b0.lock(pipe=tla.arch.MTE1)
                        tla.copy(l0_a_p, l1_a_p0)
                        tla.copy(l0_b_p, l1_b_p0)
                        mutex_l0a0.unlock(pipe=tla.arch.MTE1)
                        mutex_l0b0.unlock(pipe=tla.arch.MTE1)

                        lbuf = tla.as_numeric(c0)
                        for sk in tla.range(c0, k_sub, c1):
                            if lbuf == c0:
                                mutex_l0a0.lock(pipe=tla.arch.CUBE)
                                mutex_l0b0.lock(pipe=tla.arch.CUBE)
                            else:
                                mutex_l0a1.lock(pipe=tla.arch.CUBE)
                                mutex_l0b1.lock(pipe=tla.arch.CUBE)

                            # Next-chunk views are built unconditionally (they are pure
                            # metadata); only the guarded copy below touches them.
                            l1_a_q = tla.tile_view(
                                l1_a,
                                tla.make_shape(BM, SK),
                                tla.make_coord(c0, sk + c1),
                            )
                            l1_b_q = tla.tile_view(
                                l1_b,
                                tla.make_shape(SK, BN),
                                tla.make_coord(sk + c1, c0),
                            )
                            nxt_a_ptr = l0a1_ptr if lbuf == c0 else l0a0_ptr
                            nxt_b_ptr = l0b1_ptr if lbuf == c0 else l0b0_ptr
                            l0_a_n = tla.make_tensor_like(
                                nxt_a_ptr, l1_a_q, tla.arch.zN
                            )
                            l0_b_n = tla.make_tensor_like(
                                nxt_b_ptr, l1_b_q, tla.arch.nZ
                            )
                            if sk + c1 < k_sub:
                                if lbuf == c0:
                                    mutex_l0a1.lock(pipe=tla.arch.MTE1)
                                    mutex_l0b1.lock(pipe=tla.arch.MTE1)
                                else:
                                    mutex_l0a0.lock(pipe=tla.arch.MTE1)
                                    mutex_l0b0.lock(pipe=tla.arch.MTE1)
                                tla.copy(l0_a_n, l1_a_q)
                                tla.copy(l0_b_n, l1_b_q)
                                if lbuf == c0:
                                    mutex_l0a1.unlock(pipe=tla.arch.MTE1)
                                    mutex_l0b1.unlock(pipe=tla.arch.MTE1)
                                else:
                                    mutex_l0a0.unlock(pipe=tla.arch.MTE1)
                                    mutex_l0b0.unlock(pipe=tla.arch.MTE1)

                            cur_l0a_ptr = l0a0_ptr if lbuf == c0 else l0a1_ptr
                            cur_l0b_ptr = l0b0_ptr if lbuf == c0 else l0b1_ptr
                            l1_a_k = tla.tile_view(
                                l1_a, tla.make_shape(BM, SK), tla.make_coord(c0, sk)
                            )
                            l1_b_k = tla.tile_view(
                                l1_b, tla.make_shape(SK, BN), tla.make_coord(sk, c0)
                            )
                            l0_a = tla.make_tensor_like(
                                cur_l0a_ptr, l1_a_k, tla.arch.zN
                            )
                            l0_b = tla.make_tensor_like(
                                cur_l0b_ptr, l1_b_k, tla.arch.nZ
                            )

                            init_c = True if hb == c0 and sk == c0 else False
                            unit_flag = (
                                0b11
                                if (hb + c1 >= h_tiles) and (sk + c1 >= k_sub)
                                else 0b10
                            )
                            tla.mmad(
                                l0_c, l0_a, l0_b, init_c=init_c, unit_flag=unit_flag
                            )
                            if lbuf == c0:
                                mutex_l0a0.unlock(pipe=tla.arch.CUBE)
                                mutex_l0b0.unlock(pipe=tla.arch.CUBE)
                            else:
                                mutex_l0a1.unlock(pipe=tla.arch.CUBE)
                                mutex_l0b1.unlock(pipe=tla.arch.CUBE)
                            lbuf = c1 - lbuf

                        if buf == c0:
                            mutex_l1a0.unlock(pipe=tla.arch.MTE1)
                            mutex_l1b0.unlock(pipe=tla.arch.MTE1)
                        else:
                            mutex_l1a1.unlock(pipe=tla.arch.MTE1)
                            mutex_l1b1.unlock(pipe=tla.arch.MTE1)
                        buf = c1 - buf

                    # L0C (fp32 [BM, BN]) -> both AIV UB halves.  unit_flag=3 closes
                    # the accumulation chain so FixPipe observes a complete tile.
                    mutex_l0c.unlock(pipe=tla.arch.CUBE)
                    mutex_l0c.lock(pipe=tla.arch.FIX)
                    tla.cross_core_wait_flag(exchange, tla.arch.FIX)
                    tla.copy(
                        ub_z_full,
                        l0_c,
                        tla.params.CopyL0C2DstParams(
                            unit_flag=3,
                            l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M,
                        ),
                    )
                    tla.cross_core_set_flag(exchange, tla.arch.FIX)
                    mutex_l0c.unlock(pipe=tla.arch.FIX)

    with tla.vector():
        sub = tla.arch.sub_block_idx()
        ub_z = tla.make_tensor(
            ubz_ptr,
            tla.make_layout(tla.make_shape(ROWS_PER_SUB, BN), tla.make_stride(BN, 1)),
        )
        ub_rmax = tla.make_tensor(
            ub_rmax_ptr,
            tla.make_layout(tla.make_shape(ROWS_PER_SUB), tla.make_stride(1)),
        )
        ub_rsum = tla.make_tensor(
            ub_rsum_ptr,
            tla.make_layout(tla.make_shape(ROWS_PER_SUB), tla.make_stride(1)),
        )

        # Open the exchange so the first FixPipe hand-off can proceed.
        tla.cross_core_set_flag(exchange, tla.arch.VECTOR)

        for blk in tla.range(tla.arch.block_idx(), total_blocks, tla.arch.block_num()):
            seg = blk // grid_m
            mb = blk - seg * grid_m
            sub_row0 = mb * BM + sub * ROWS_PER_SUB
            vt_lo = seg * tiles_per_seg
            vt_hi = vt_lo + tiles_per_seg
            if vt_hi > nv_tiles:
                vt_hi = nv_tiles

            if vt_lo < vt_hi:
                with tla.vec.func(mode="simd"):
                    preg_all = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                    # A mask only covers VL (=64) lanes, so the per-row statistics
                    # are initialised in VL-row chunks; with a single 64-lane store
                    # only rows below 64 would be initialised when BM > 128.
                    for i0 in tla.range(c0, ROWS_PER_SUB // VL, c1):
                        tla.tile_view(
                            ub_rmax, tla.make_shape(VL), tla.make_coord(i0)
                        ).store(tla.full(MIN_F, dtype=tla.Float32), mask=preg_all)
                        tla.tile_view(
                            ub_rsum, tla.make_shape(VL), tla.make_coord(i0)
                        ).store(tla.full(0.0, dtype=tla.Float32), mask=preg_all)

                brc = tla.params.NormalLoadParams(
                    load_dist=tla.params.LoadDist.DIST_BRC_B32
                )

                for vt in tla.range(vt_lo, vt_hi):
                    v0 = vt * BN
                    ncols = tla.as_numeric(v_real - v0)
                    if ncols > BN:
                        ncols = tla.as_numeric(BN)
                    tla.cross_core_wait_flag(exchange, tla.arch.VECTOR)

                    # Online update per row: tile max first, then the rescale factor
                    # and the exp-sum, all from the registers loaded for this tile.
                    # Complexity: O(BM * BN) vector work per tile.
                    with tla.vec.func(mode="simd"):
                        preg_all = tla.create_mask(
                            pattern=tla.mask.ALL, dtype=tla.Float32
                        )
                        one_mask, _ = tla.update_mask(1, dtype=tla.Float32)
                        minv = tla.full(MIN_F, dtype=tla.Float32)
                        # Chunk masks depend only on the tile width, so they are built
                        # once per tile instead of once per row.
                        lane1 = ncols - 0 * VL
                        lane2 = ncols - 1 * VL
                        lane3 = ncols - 2 * VL
                        lane4 = ncols - 3 * VL
                        if lane1 > VL:
                            lane1 = tla.as_numeric(VL)
                        if lane2 > VL:
                            lane2 = tla.as_numeric(VL)
                        if lane3 > VL:
                            lane3 = tla.as_numeric(VL)
                        if lane4 > VL:
                            lane4 = tla.as_numeric(VL)
                        if lane2 < 0:
                            lane2 = tla.as_numeric(0)
                        if lane3 < 0:
                            lane3 = tla.as_numeric(0)
                        if lane4 < 0:
                            lane4 = tla.as_numeric(0)
                        pm1, _ = tla.update_mask(lane1, dtype=tla.Float32)
                        pm2, _ = tla.update_mask(lane2, dtype=tla.Float32)
                        pm3, _ = tla.update_mask(lane3, dtype=tla.Float32)
                        pm4, _ = tla.update_mask(lane4, dtype=tla.Float32)

                        for r1 in tla.range(ROWS_PER_SUB):
                            # One scale + mask per chunk; the four registers feed both
                            # the max pass and the exp pass.
                            ra0 = tla.tile_view(
                                ub_z, tla.make_shape(1, VL), tla.make_coord(r1, 0)
                            ).load()
                            ra0 = tla.mul(ra0, inv_temperature, mask=preg_all)
                            ra0 = tla.where(pm1, ra0, minv)
                            ra1 = tla.tile_view(
                                ub_z, tla.make_shape(1, VL), tla.make_coord(r1, 1)
                            ).load()
                            ra1 = tla.mul(ra1, inv_temperature, mask=preg_all)
                            ra1 = tla.where(pm2, ra1, minv)
                            ra2 = tla.tile_view(
                                ub_z, tla.make_shape(1, VL), tla.make_coord(r1, 2)
                            ).load()
                            ra2 = tla.mul(ra2, inv_temperature, mask=preg_all)
                            ra2 = tla.where(pm3, ra2, minv)
                            ra3 = tla.tile_view(
                                ub_z, tla.make_shape(1, VL), tla.make_coord(r1, 3)
                            ).load()
                            ra3 = tla.mul(ra3, inv_temperature, mask=preg_all)
                            ra3 = tla.where(pm4, ra3, minv)

                            mxv = tla.max(ra0, ra1, mask=preg_all)
                            mxv = tla.max(mxv, ra2, mask=preg_all)
                            mxv = tla.max(mxv, ra3, mask=preg_all)
                            tmax = mxv.reduce(tla.ReductionOp.MAX, mask=preg_all)
                            old_max = tla.tile_view(
                                ub_rmax, tla.make_shape(1), tla.make_coord(r1)
                            ).load(params=brc)
                            new_max = tla.max(old_max, tmax, mask=preg_all)
                            tla.tile_view(
                                ub_rmax, tla.make_shape(1), tla.make_coord(r1)
                            ).store(
                                new_max, tla.params.UnalignStoreParams(), mask=one_mask
                            )
                            tla.local_mem_bar(
                                tla.params.MemType.VEC_STORE,
                                tla.params.MemType.VEC_LOAD,
                            )
                            maxb = tla.tile_view(
                                ub_rmax, tla.make_shape(1), tla.make_coord(r1)
                            ).load(params=brc)

                            sv = tla.exp(
                                tla.sub(ra0, maxb, mask=preg_all), mask=preg_all
                            )
                            sv = tla.add(
                                sv,
                                tla.exp(
                                    tla.sub(ra1, maxb, mask=preg_all), mask=preg_all
                                ),
                                mask=preg_all,
                            )
                            sv = tla.add(
                                sv,
                                tla.exp(
                                    tla.sub(ra2, maxb, mask=preg_all), mask=preg_all
                                ),
                                mask=preg_all,
                            )
                            sv = tla.add(
                                sv,
                                tla.exp(
                                    tla.sub(ra3, maxb, mask=preg_all), mask=preg_all
                                ),
                                mask=preg_all,
                            )
                            tile_exp_sum = sv.reduce(tla.ReductionOp.ADD, mask=preg_all)
                            old_sum = tla.tile_view(
                                ub_rsum, tla.make_shape(1), tla.make_coord(r1)
                            ).load(params=brc)
                            delta = tla.where(
                                one_mask,
                                tla.sub(old_max, maxb, mask=preg_all),
                                minv,
                            )
                            scale = tla.exp(delta, mask=preg_all)
                            new_sum = tla.add(
                                tile_exp_sum,
                                tla.mul(old_sum, scale, mask=preg_all),
                                mask=preg_all,
                            )
                            tla.tile_view(
                                ub_rsum, tla.make_shape(1), tla.make_coord(r1)
                            ).store(
                                new_sum, tla.params.UnalignStoreParams(), mask=one_mask
                            )

                    tla.pipe_barrier(tla.pipes.ALL)
                    tla.cross_core_set_flag(exchange, tla.arch.VECTOR)

                # Publish this segment's partial statistics.  Fresh names keep the
                # dynamic-loop def/use checker happy (names defined inside the tile
                # loop must not be read after it).
                with tla.vec.func(mode="simt", thread_block_dim=ROWS_PER_SUB):
                    ptid, _, _ = tla.arch.thread_idx()
                    pgrow = sub_row0 + ptid
                    if pgrow < m_real:
                        tile_max[pgrow, seg] = ub_rmax[ptid]
                        tile_sum[pgrow, seg] = ub_rsum[ptid]


@tla.kernel
def msrs_merge_kernel(
    tile_max: tla.Tensor,  # [M, NV] fp32
    tile_sum: tla.Tensor,  # [M, NV] fp32
    row_max: tla.Tensor,  # [M] fp32
    row_sum: tla.Tensor,  # [M] fp32
    nv_tiles: int,
    NV: tla.Constexpr[int],
) -> None:
    m = row_max.origin_shape[0]
    tiles_per_seg = (nv_tiles + NV - 1) // NV

    # Complexity: O(M * NV) scalar work, NV is a small compile-time constant.
    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=128):
            tid, _, _ = tla.arch.thread_idx()
            tbd, _, _ = tla.arch.thread_block_dim()
            gstart = tla.arch.block_idx() * tbd + tid
            gstride = tla.arch.block_num() * tbd
            for mr in tla.range(gstart, m, gstride):
                mx = tla.as_numeric(MIN_F)
                for ms1 in tla.range(0, NV):
                    lo1 = ms1 * tiles_per_seg
                    if lo1 < nv_tiles:
                        value1 = tile_max[mr, ms1]
                        if value1 > mx:
                            mx = value1
                total = tla.as_numeric(0.0)
                for ms2 in tla.range(0, NV):
                    lo2 = ms2 * tiles_per_seg
                    if lo2 < nv_tiles:
                        total = total + tile_sum[mr, ms2] * tla.exp(
                            tile_max[mr, ms2] - mx
                        )
                row_max[mr] = mx
                row_sum[mr] = total


# ======================================================================================
# HOST side: tiling choice, workspace, compile and launch.
# ======================================================================================

_COMPILED = {}
# The right operand is read-only across calls, so its [V, H] -> [H, V] (plus
# zero padding) transform is cached.  The key carries the buffer address, shape,
# dtype and torch version counter, so an in-place update invalidates the entry
# instead of serving a stale buffer.
_W_CACHE = {}


def v_pad_target(w, bn):
    rows = int(w.shape[0])
    return ((rows + bn - 1) // bn) * bn


def _w_hv(w, v_pad, h_pad):
    import torch

    v, h = int(w.shape[0]), int(w.shape[1])
    key = (
        w.data_ptr(),
        tuple(w.shape),
        str(w.dtype),
        int(getattr(w, "_version", 0)),
        v_pad,
        h_pad,
    )
    cached = _W_CACHE.get(key)
    if cached is not None:
        return cached
    if v_pad != v or h_pad != h:
        padded = torch.zeros((v_pad, h_pad), dtype=w.dtype, device=w.device)
        padded[:v, :h] = w
    else:
        padded = w
    result = padded.t().contiguous()
    _W_CACHE.clear()
    _W_CACHE[key] = result
    return result


def _core_counts(index):
    import torch

    props = torch.npu.get_device_properties(index)
    return int(props.cube_core_num), int(props.vector_core_num)


def _tla_tensor(buffer, layout_tag):
    from catlass.tla.runtime import from_dlpack

    # Static layout: the SIMT statistics regions require statically shaped
    # buffers, so one artifact is compiled per (M, H, V, NV) tuple.
    return from_dlpack(buffer, layout_tag=layout_tag)


def run(x, w, temperature):
    """Return ``(row_max, row_sum)``; see README.md for semantics."""
    import torch

    m, h = x.shape
    v = w.shape[0]
    device = x.device

    row_max = torch.empty(m, dtype=torch.float32, device=device)
    row_sum = torch.empty(m, dtype=torch.float32, device=device)
    if m == 0:
        return row_max, row_sum

    bm, bn = BM, BN

    inv_temperature = 1.0 / float(temperature)

    grid_m = (m + bm - 1) // bm
    nv_tiles = (v + bn - 1) // bn
    device_index = int(str(x.device).split(":")[-1]) if ":" in str(x.device) else 0
    cube_cores, vector_cores = _core_counts(device_index)
    # Wave quantization: grid = grid_m * nv blocks must land on a core-count
    # multiple, otherwise the last wave runs with idle cores (64 blocks on 28
    # cores = 2.29 waves -> 76% cube utilization).  Pick the smallest V-segment
    # count whose block count wastes the fewest slots.
    nv = 1
    best_waste = None
    cand = 1
    cand_max = min(nv_tiles, 64)
    while cand <= cand_max:
        total = grid_m * cand
        waves = (total + cube_cores - 1) // cube_cores
        waste = float(waves * cube_cores - total) / float(total)
        if best_waste is None or waste < best_waste - 1e-12:
            nv = cand
            best_waste = waste
        cand = cand + 1

    tile_max = torch.empty((m, nv), dtype=torch.float32, device=device)
    tile_sum = torch.empty_like(tile_max)

    # Partial GM->L1 tiles have no working copy route, so every GEMM operand is
    # zero-padded up to whole tiles.  Zero padding is exact for M (rows beyond the
    # row count are dropped by the m_real guard) and for H (extra K terms
    # contribute 0); the padded V columns are masked out by v_real.
    m_pad = ((m + bm - 1) // bm) * bm
    h_pad = ((h + BH - 1) // BH) * BH
    if m_pad != m or h_pad != h:
        x_pad = torch.zeros((m_pad, h_pad), dtype=x.dtype, device=device)
        x_pad[:m, :h] = x
    else:
        x_pad = x

    x_t = _tla_tensor(x_pad, tla.arch.RowMajor)
    # The DSL copy has no ColumnMajor-GM -> zN-L1 route, so the [V, H] operand is
    # transposed once to a [H, V] row-major form (cached across calls, see
    # README.md).  A partial (V % BN != 0) rhs tile also has no working GM->L1
    # route, so the operand is zero-padded up to a BN multiple; the padded columns
    # are excluded from max/exp-sum by the kernel mask (v_real below).
    w_hv = _w_hv(w, v_pad_target(w, bn), h_pad)
    w_t = _tla_tensor(w_hv, tla.arch.RowMajor)
    tmax_t = _tla_tensor(tile_max, tla.arch.RowMajor)
    tsum_t = _tla_tensor(tile_sum, tla.arch.RowMajor)
    rmax_t = _tla_tensor(row_max, tla.arch.RowMajor)
    rsum_t = _tla_tensor(row_sum, tla.arch.RowMajor)

    key = (m, h, v, nv)
    entry = _COMPILED.get(key)
    if entry is None:
        stats_artifact = tla.compile(
            msrs_main_kernel,
            x_t,
            w_t,
            tmax_t,
            tsum_t,
            inv_temperature,
            m,
            v,
            nv,
            options="--npu-arch 3510",
        )
        merge_artifact = tla.compile(
            msrs_merge_kernel,
            tmax_t,
            tsum_t,
            rmax_t,
            rsum_t,
            nv_tiles,
            nv,
            options="--npu-arch 3510",
        )
        entry = (stats_artifact, merge_artifact, cube_cores, vector_cores)
        _COMPILED[key] = entry
    stats_artifact, merge_artifact, cube_cores, vector_cores = entry

    stats_artifact(
        x_t,
        w_t,
        tmax_t,
        tsum_t,
        inv_temperature,
        m,
        v,
        block_num=cube_cores,
    )
    merge_artifact(
        tmax_t,
        tsum_t,
        rmax_t,
        rsum_t,
        nv_tiles,
        block_num=vector_cores,
    )
    return row_max, row_sum
