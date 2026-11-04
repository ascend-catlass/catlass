# -----------------------------------------------------------------------------------------------------------
# CATLASS Python DSL kernels for matmul_softmax_stats.
#
# Two kernels:
#   main  : GEMM + online softmax statistics (z never lands in GM)
#   merge : fold the per-V-segment partial (max, sum) into the final statistics
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
def mss_main_kernel(
    hidden: tla.Tensor,  # [M, H] bf16 RowMajor
    weight: tla.Tensor,  # [H, V] bf16 RowMajor (host pre-transposes [V, H])
    part_max: tla.Tensor,  # [M, NV] fp32 per V-segment partial max
    part_sum: tla.Tensor,  # [M, NV] fp32 per V-segment partial exp-sum
    inv_temperature: float,
    tiles_per_seg: int,
    m_real: int,
    v_real: int,
    NV: tla.Constexpr[int],
) -> None:
    c0 = 0
    c1 = 1

    m = hidden.origin_shape[0]
    h = hidden.origin_shape[1]
    v = weight.origin_shape[1]

    grid_m = (m + BM - 1) // BM
    h_tiles = (h + BH - 1) // BH
    k_sub = (BH + SK - 1) // SK
    nv_tiles = tla.as_numeric((v + BN - 1) // BN)
    total_blocks = grid_m * NV

    dtype_in = hidden.ptr.dtype

    # C/V crossing: one cross flag handed back and forth (VECTOR -> FIX -> VECTOR).
    exchange = tla.cross_flag("exchange")
    # Same-core producer/consumer flags.
    l1a_ready0 = tla.flag("l1a_ready0", tla.arch.MTE2, tla.arch.MTE1)
    l1b_ready0 = tla.flag("l1b_ready0", tla.arch.MTE2, tla.arch.MTE1)
    l1a_free0 = tla.flag("l1a_free0", tla.arch.MTE1, tla.arch.MTE2)
    l1b_free0 = tla.flag("l1b_free0", tla.arch.MTE1, tla.arch.MTE2)
    l1a_ready1 = tla.flag("l1a_ready1", tla.arch.MTE2, tla.arch.MTE1)
    l1b_ready1 = tla.flag("l1b_ready1", tla.arch.MTE2, tla.arch.MTE1)
    l1a_free1 = tla.flag("l1a_free1", tla.arch.MTE1, tla.arch.MTE2)
    l1b_free1 = tla.flag("l1b_free1", tla.arch.MTE1, tla.arch.MTE2)
    l0_ready0 = tla.flag("l0_ready0", tla.arch.MTE1, tla.arch.CUBE)
    l0_ready1 = tla.flag("l0_ready1", tla.arch.MTE1, tla.arch.CUBE)
    l0a_free0 = tla.flag("l0a_free0", tla.arch.CUBE, tla.arch.MTE1)
    l0a_free1 = tla.flag("l0a_free1", tla.arch.CUBE, tla.arch.MTE1)
    l0b_free0 = tla.flag("l0b_free0", tla.arch.CUBE, tla.arch.MTE1)
    l0b_free1 = tla.flag("l0b_free1", tla.arch.CUBE, tla.arch.MTE1)
    mmad_done = tla.flag("mmad_done", tla.arch.CUBE, tla.arch.FIX)
    l0c_free = tla.flag("l0c_free", tla.arch.FIX, tla.arch.CUBE)

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
        tla.set_flag(l1a_free0)
        tla.set_flag(l1b_free0)
        tla.set_flag(l1a_free1)
        tla.set_flag(l1b_free1)
        tla.set_flag(l0a_free0)
        tla.set_flag(l0b_free0)
        tla.set_flag(l0a_free1)
        tla.set_flag(l0b_free1)
        tla.set_flag(l0c_free)

        for blk in tla.range(tla.arch.block_idx(), total_blocks, tla.arch.block_num()):
            # Segment-major: consecutive blocks are consecutive M tiles inside one
            # V segment, so all concurrently running cores sweep the same weight
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
                    # free wait is deferred to just before the first MAC so this
                    # GM->L1 load overlaps the previous tile's FixPipe drain.
                    gm_a_p = tla.tile_view(
                        hidden, tla.make_shape(BM, BH), tla.make_coord(mb, c0)
                    )
                    gm_b_p = tla.tile_view(
                        weight, tla.make_shape(BH, BN), tla.make_coord(c0, vt)
                    )
                    l1_a_p = tla.make_tensor_like(l1a0_ptr, gm_a_p, tla.arch.zN)
                    l1_b_p = tla.make_tensor_like(l1b0_ptr, gm_b_p, tla.arch.zN)
                    tla.wait_flag(l1a_free0)
                    tla.wait_flag(l1b_free0)
                    tla.copy(l1_a_p, gm_a_p)
                    tla.copy(l1_b_p, gm_b_p)
                    tla.set_flag(l1a_ready0)
                    tla.set_flag(l1b_ready0)

                    # First MAC of this V tile must not start before the previous
                    # tile's L0C->UB copy has consumed L0C.
                    tla.wait_flag(l0c_free)

                    buf = tla.as_numeric(c0)
                    for hb in tla.range(c0, h_tiles, c1):
                        if buf == c0:
                            tla.wait_flag(l1a_ready0)
                            tla.wait_flag(l1b_ready0)
                        else:
                            tla.wait_flag(l1a_ready1)
                            tla.wait_flag(l1b_ready1)

                        # Prefetch the next K tile into the other stage.
                        if hb + c1 < h_tiles:
                            gm_a_n = tla.tile_view(
                                hidden,
                                tla.make_shape(BM, BH),
                                tla.make_coord(mb, hb + c1),
                            )
                            gm_b_n = tla.tile_view(
                                weight,
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
                                tla.wait_flag(l1a_free1)
                                tla.wait_flag(l1b_free1)
                            else:
                                tla.wait_flag(l1a_free0)
                                tla.wait_flag(l1b_free0)
                            tla.copy(l1_a_n, gm_a_n)
                            tla.copy(l1_b_n, gm_b_n)
                            if buf == c0:
                                tla.set_flag(l1a_ready1)
                                tla.set_flag(l1b_ready1)
                            else:
                                tla.set_flag(l1a_ready0)
                                tla.set_flag(l1b_ready0)

                        cur_a_ptr = l1a0_ptr if buf == c0 else l1a1_ptr
                        cur_b_ptr = l1b0_ptr if buf == c0 else l1b1_ptr
                        gm_a_c = tla.tile_view(
                            hidden, tla.make_shape(BM, BH), tla.make_coord(mb, hb)
                        )
                        gm_b_c = tla.tile_view(
                            weight, tla.make_shape(BH, BN), tla.make_coord(hb, vt)
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
                        tla.wait_flag(l0a_free0)
                        tla.wait_flag(l0b_free0)
                        tla.copy(l0_a_p, l1_a_p0)
                        tla.copy(l0_b_p, l1_b_p0)
                        tla.set_flag(l0_ready0)

                        lbuf = tla.as_numeric(c0)
                        for sk in tla.range(c0, k_sub, c1):
                            if lbuf == c0:
                                tla.wait_flag(l0_ready0)
                            else:
                                tla.wait_flag(l0_ready1)

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
                                    tla.wait_flag(l0a_free1)
                                    tla.wait_flag(l0b_free1)
                                else:
                                    tla.wait_flag(l0a_free0)
                                    tla.wait_flag(l0b_free0)
                                tla.copy(l0_a_n, l1_a_q)
                                tla.copy(l0_b_n, l1_b_q)
                                if lbuf == c0:
                                    tla.set_flag(l0_ready1)
                                else:
                                    tla.set_flag(l0_ready0)

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
                                tla.set_flag(l0a_free0)
                                tla.set_flag(l0b_free0)
                            else:
                                tla.set_flag(l0a_free1)
                                tla.set_flag(l0b_free1)
                            lbuf = c1 - lbuf

                        if buf == c0:
                            tla.set_flag(l1a_free0)
                            tla.set_flag(l1b_free0)
                        else:
                            tla.set_flag(l1a_free1)
                            tla.set_flag(l1b_free1)
                        buf = c1 - buf

                    # L0C (fp32 [BM, BN]) -> both AIV UB halves.  unit_flag=3 closes
                    # the accumulation chain so FixPipe observes a complete tile.
                    tla.set_flag(mmad_done)
                    tla.wait_flag(mmad_done)
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
                    tla.set_flag(l0c_free)

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

                    # Pass 1: tile row max + the rescale factor of the online update.
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
                            tile_sum = sv.reduce(tla.ReductionOp.ADD, mask=preg_all)
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
                                tile_sum,
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
                        part_max[pgrow, seg] = ub_rmax[ptid]
                        part_sum[pgrow, seg] = ub_rsum[ptid]


@tla.kernel
def mss_merge_kernel(
    part_max: tla.Tensor,  # [M, NV] fp32
    part_sum: tla.Tensor,  # [M, NV] fp32
    local_max: tla.Tensor,  # [M] fp32
    exp_sum: tla.Tensor,  # [M] fp32
    nv_tiles: int,
    tiles_per_seg: int,
    NV: tla.Constexpr[int],
) -> None:
    m = local_max.origin_shape[0]

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
                        value1 = part_max[mr, ms1]
                        if value1 > mx:
                            mx = value1
                total = tla.as_numeric(0.0)
                for ms2 in tla.range(0, NV):
                    lo2 = ms2 * tiles_per_seg
                    if lo2 < nv_tiles:
                        total = total + part_sum[mr, ms2] * tla.exp(
                            part_max[mr, ms2] - mx
                        )
                local_max[mr] = mx
                exp_sum[mr] = total


# ======================================================================================
# HOST side: tiling choice, workspace, compile and launch.
# ======================================================================================

_COMPILED = {}
# The LM-head shard is read-only while decoding, so its [V, H] -> [H, V]
# (plus zero padding) transform is cached across calls.  The key carries the
# buffer address, shape, dtype and torch version counter, so an in-place update
# of the shard invalidates the entry instead of serving stale weights.
_WEIGHT_CACHE = {}


def v_pad_target(weight, bn):
    rows = int(weight.shape[0])
    return ((rows + bn - 1) // bn) * bn


def _weight_hv(weight, v_pad, h_pad):
    import torch

    local_vocab, hidden_size = int(weight.shape[0]), int(weight.shape[1])
    key = (
        weight.data_ptr(),
        tuple(weight.shape),
        str(weight.dtype),
        int(getattr(weight, "_version", 0)),
        v_pad,
        h_pad,
    )
    cached = _WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached
    if v_pad != local_vocab or h_pad != hidden_size:
        padded = torch.zeros((v_pad, h_pad), dtype=weight.dtype, device=weight.device)
        padded[:local_vocab, :hidden_size] = weight
    else:
        padded = weight
    result = padded.t().contiguous()
    _WEIGHT_CACHE.clear()
    _WEIGHT_CACHE[key] = result
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


def run(hidden, weight, temperature):
    """Return ``(local_max, exp_sum)``; see README.md for semantics."""
    import torch

    num_tokens, hidden_size = hidden.shape
    local_vocab = weight.shape[0]
    device = hidden.device

    local_max = torch.empty(num_tokens, dtype=torch.float32, device=device)
    exp_sum = torch.empty(num_tokens, dtype=torch.float32, device=device)
    if num_tokens == 0:
        return local_max, exp_sum

    bm, bn = BM, BN

    inv_temperature = 1.0 / float(temperature)

    grid_m = (num_tokens + bm - 1) // bm
    nv_tiles = (local_vocab + bn - 1) // bn
    device_index = (
        int(str(hidden.device).split(":")[-1]) if ":" in str(hidden.device) else 0
    )
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
    tiles_per_seg = (nv_tiles + nv - 1) // nv

    part_max = torch.empty((num_tokens, nv), dtype=torch.float32, device=device)
    part_sum = torch.empty_like(part_max)

    # Partial GM->L1 tiles have no working copy route, so every GEMM operand is
    # zero-padded up to whole tiles.  Zero padding is exact for M (rows beyond the
    # token count are dropped) and for H (extra K terms contribute 0); the padded
    # V columns are masked out by v_real.
    m_pad = ((num_tokens + bm - 1) // bm) * bm
    h_pad = ((hidden_size + BH - 1) // BH) * BH
    if m_pad != num_tokens or h_pad != hidden_size:
        hidden_use = torch.zeros((m_pad, h_pad), dtype=hidden.dtype, device=device)
        hidden_use[:num_tokens, :hidden_size] = hidden
    else:
        hidden_use = hidden

    h_t = _tla_tensor(hidden_use, tla.arch.RowMajor)
    # The DSL copy has no ColumnMajor-GM -> zN-L1 route, so the [V, H] shard is
    # transposed to a [H, V] row-major operand once per call (see README.md).  A
    # partial (V % BN != 0) weight tile also has no working GM->L1 route, so the
    # shard is zero-padded up to a BN multiple; the padded columns are excluded
    # from max/exp-sum by the kernel mask (v_real below).
    weight_hv = _weight_hv(weight, v_pad_target(weight, bn), h_pad)
    w_t = _tla_tensor(weight_hv, tla.arch.RowMajor)
    pmax_t = _tla_tensor(part_max, tla.arch.RowMajor)
    psum_t = _tla_tensor(part_sum, tla.arch.RowMajor)
    emax_t = _tla_tensor(local_max, tla.arch.RowMajor)
    esum_t = _tla_tensor(exp_sum, tla.arch.RowMajor)

    key = (num_tokens, hidden_size, local_vocab, nv)
    entry = _COMPILED.get(key)
    if entry is None:
        stats_artifact = tla.compile(
            mss_main_kernel,
            h_t,
            w_t,
            pmax_t,
            psum_t,
            inv_temperature,
            tiles_per_seg,
            num_tokens,
            local_vocab,
            nv,
            options="--npu-arch 3510",
        )
        merge_artifact = tla.compile(
            mss_merge_kernel,
            pmax_t,
            psum_t,
            emax_t,
            esum_t,
            nv_tiles,
            tiles_per_seg,
            nv,
            options="--npu-arch 3510",
        )
        entry = (stats_artifact, merge_artifact, cube_cores, vector_cores)
        _COMPILED[key] = entry
    stats_artifact, merge_artifact, cube_cores, vector_cores = entry

    stats_artifact(
        h_t,
        w_t,
        pmax_t,
        psum_t,
        inv_temperature,
        tiles_per_seg,
        num_tokens,
        local_vocab,
        block_num=cube_cores,
    )
    merge_artifact(
        pmax_t,
        psum_t,
        emax_t,
        esum_t,
        nv_tiles,
        tiles_per_seg,
        block_num=vector_cores,
    )
    return local_max, exp_sum
