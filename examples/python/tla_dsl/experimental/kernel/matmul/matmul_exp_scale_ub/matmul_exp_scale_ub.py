# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""GEMM with a per-row ``exp`` / scale epilogue and in-kernel bf16 writeback.

For ``x [M, H]``, ``weight [V, H]``, ``row_logsumexp [M]`` and ``row_scale [M]``:

    Y[m, n] = bf16_rne(exp(float32(x[m, :] @ weight[n, :].T) - row_logsumexp[m]) * row_scale[m])

One ``@tla.kernel``, one launch, Cube + Vector MIX:

    cube    : K-blocked MMAD of one [BM, N_EXT] fp32 tile in L0C
    fixpipe : L0C -> the UB of both AIVs of the core (SPLIT_M, half the rows each)
    vector  : per-row epilogue in UB -> bf16 narrowing -> MTE3 writeback

``V`` may be any positive size.  The last tile column is a *tail* when
``V % BN != 0``: it keeps ``V_REAL = V - (n_tiles - 1) * BN`` valid columns and
its cube-side N extent ``N_EXT`` is rounded up to a multiple of 16 (one MMAD /
L0B fractal).  Full tiles use ``N_EXT = BN``.

Structure versus payload
------------------------
The cube pipeline, the ``exchange`` hand-off, the SPLIT_M fixpipe and the bf16
writeback are the *structure* of this sample and are meant to be reused as is.
The vector-side math (subtract a row scalar, ``exp``, multiply a row scalar) is
the *payload*: replace it wholesale, but keep the per-row scalar shape (SIMT row
scalars -> SIMD broadcast) so the surrounding paths stay exercised.

Four facts the structure relies on; each of them has hung or corrupted a run:

1. ``tla.tile_view`` takes *tile* coordinates.  The front end multiplies them by
   the tile shape to obtain the element offset, so passing an element row or
   column number inflates the offset by the tile size and drives the DMA off
   the buffer (errcode 82 / 507015, or a hang).  The GM output tile below uses
   the alternative, ``tla.make_tensor(ptr + element_offset, layout(...))``,
   whose offset is element-granular.
2. ``origin_shape`` *does* crop a DMA (sentinel probe on Ascend 950PR with DSL
   2.2.1: a UB -> GM and GM -> UB copy with shape 64 rows and origin 32 moved
   exactly 32 rows; the cube-side GM -> L1 ND2NZ read exactly the origin
   columns).  A cropped read leaves the rest of the destination tile
   *undefined* (observed as zeros for L1), so the valid extent is never
   inferred from memory: the tail tile is sized explicitly (``N_EXT`` for the
   cube, ``V_REAL`` for fixpipe / vector / writeback) and its B tile is kept
   inside the weight buffer by the host (see ``run``).
3. MMAD never completes when ``ceil(acc_M / 16) > ceil(A_M / 16)`` (same for B
   versus the N extent of the C tile): the MAC asks for more 16-row fractals
   than MTE1 loaded, the cube core spins at 100 % and no error code is raised.  Zero
   padding A to whole ``BM`` tiles keeps the counts equal.  A B tile cropped by
   its ``origin_shape`` hangs the same way: ``weight`` must therefore be sliced
   from its logical ``[H, V]`` (column-major) view so that a ``[BK, N_EXT]`` tile
   never extends past ``H``.
4. A ``DIST_PACK_B32`` store writes the low bf16 half of each B32 register slot,
   and it needs all 128 mask bits of the fragment to commit its 64 values: with
   only the low 64 lanes set it writes 32, leaving stale bytes in the rest of
   the tile.  The bf16 tile is therefore stored with an all-true mask, and the
   lanes past ``V_REAL`` are steered into a row pitch padded to a whole SIMD
   chunk (``N_UB``) that the MTE3 copy never reads, instead of being masked off
   at the store.
"""

import catlass.tla as tla

# Tile shape.  BM/BN is the cube output tile, BK the L1 K tile, SK the L0 K step.
BM = 128
BN = 256
BK = 128
SK = 32
HALF = BM // 2  # rows one AIV owns after the SPLIT_M fixpipe
VL = 64  # fp32 vector register lanes
SUB_K = BK // SK  # L0 K steps inside one L1 K tile
FRACTAL = 16  # MMAD fractal; the tail tile's N extent is rounded up to it


def _b_tile(gm_w, kt, nt, n_ext):
    """[BK, n_ext] B tile at tile (kt, nt) of the column-major [H, V_w] weight.

    tile coordinates are scaled by the tile shape (fact 1), so a direct
    [BK, n_ext] view at column tile ``nt`` would start at column ``nt * n_ext``.
    The first [BK, BN] view places the tile at ``nt * BN``; the second one only
    narrows its width to the pass's N extent.
    """
    return tla.tile_view(
        tla.tile_view(gm_w, tla.make_shape(BK, BN), tla.make_coord(kt, nt)),
        tla.make_shape(BK, n_ext),
        tla.make_coord(0, 0),
    )


@tla.kernel
def matmul_exp_scale_ub_kernel(
    gm_x: tla.Tensor,  # [M_pad, H] bf16 RowMajor, padded rows are zero
    gm_w: tla.Tensor,  # [V_w, H] bf16 buffer bound as ColumnMajor == logical [H, V_w] (W^T); V_w = RoundUp(V, 16)
    gm_lse: tla.Tensor,  # [M_pad] fp32, padded rows hold the neutral 0
    gm_scale: tla.Tensor,  # [M_pad] fp32, padded rows hold the neutral 1
    gm_y: tla.Tensor,  # [M_pad, V] bf16 RowMajor
) -> None:
    c0 = 0
    c1 = 1

    # The host tensors are static, so these extents are compile-time Python ints
    # and the per-pass tile geometry below is static as well.
    m_pad = gm_x.origin_shape[0]
    h = gm_x.origin_shape[1]
    v = gm_y.origin_shape[1]
    m_tiles = m_pad // BM
    n_tiles = (v + BN - 1) // BN
    k_tiles = h // BK
    v_tail = v - (n_tiles - 1) * BN  # valid columns of the last tile column
    n_tail = (v_tail + FRACTAL - 1) // FRACTAL * FRACTAL  # its cube-side N extent

    # Two static passes over the task list, both strided by block_idx so the cube
    # and the vector region of one core walk the same tasks in the same order:
    #   pass 0: the full tile columns (nt < n_tiles - 1); empty when V <= BN
    #   pass 1: the last tile column, one per mt; a full column when V % BN == 0
    # Each entry is (task count, tiles per row in the pass, nt offset, N_EXT, V_REAL).
    passes = (
        (m_tiles * (n_tiles - 1), max(n_tiles - 1, 1), 0, BN, BN),
        (m_tiles, 1, n_tiles - 1, n_tail, v_tail),
    )

    dtype_in = gm_x.ptr.dtype

    # VECTOR -> FIX -> VECTOR: the fp32 staging half must not be overwritten by
    # the next fixpipe before this AIV has consumed it.
    exchange = tla.cross_flag("exchange")
    # VECTOR <-> MTE3 on the bf16 output tile.
    mutex_out = tla.mutex("ub_out", id=9)
    mutex_l1a0 = tla.mutex("l1a0", id=0)
    mutex_l1b0 = tla.mutex("l1b0", id=1)
    mutex_l1a1 = tla.mutex("l1a1", id=2)
    mutex_l1b1 = tla.mutex("l1b1", id=3)
    mutex_l0a0 = tla.mutex("l0a0", id=4)
    mutex_l0b0 = tla.mutex("l0b0", id=5)
    mutex_l0a1 = tla.mutex("l0a1", id=6)
    mutex_l0b1 = tla.mutex("l0b1", id=7)
    mutex_l0c = tla.mutex("l0c", id=8)

    l1a0_ptr = tla.allocate(BM * BK, dtype_in, tla.AddressSpace.l1, 512)
    l1b0_ptr = tla.allocate(BK * BN, dtype_in, tla.AddressSpace.l1, 512)
    l1a1_ptr = tla.allocate(BM * BK, dtype_in, tla.AddressSpace.l1, 512)
    l1b1_ptr = tla.allocate(BK * BN, dtype_in, tla.AddressSpace.l1, 512)
    l0a0_ptr = tla.allocate(BM * SK, dtype_in, tla.AddressSpace.l0a, 512)
    l0b0_ptr = tla.allocate(SK * BN, dtype_in, tla.AddressSpace.l0b, 512)
    l0a1_ptr = tla.allocate(BM * SK, dtype_in, tla.AddressSpace.l0a, 512)
    l0b1_ptr = tla.allocate(SK * BN, dtype_in, tla.AddressSpace.l0b, 512)
    # L0C accumulators are fed to tla.mmad, which needs 1024-byte alignment.
    l0c_ptr = tla.allocate(BM * BN, tla.Float32, tla.AddressSpace.l0c, 1024)

    # Per-AIV UB: the SPLIT_M half of the fp32 tile, the bf16 output half and two
    # row-scalar vectors.  Allocated for the full BN width; the tail pass views
    # them with their own (static) N extent.
    ubc_ptr = tla.allocate(HALF * BN, tla.Float32, tla.AddressSpace.ub, 256)
    uby_ptr = tla.allocate(HALF * BN, tla.BFloat16, tla.AddressSpace.ub, 256)
    ublse_ptr = tla.allocate(HALF, tla.Float32, tla.AddressSpace.ub, 256)
    ubscale_ptr = tla.allocate(HALF, tla.Float32, tla.AddressSpace.ub, 256)

    # ------------------------------------------------------------------ cube
    with tla.cube():
        for p in tla.range_constexpr(0, 2):
            tasks, tiles_per_row, nt_off, n_ext, v_real = passes[p]
            if tla.const_expr(tasks > 0):
                for task in tla.range(
                    tla.arch.block_idx(), tasks, tla.arch.block_num()
                ):
                    mt = task // tiles_per_row
                    nt = task - mt * tiles_per_row + nt_off

                    # Built per pass: the tail tile is [BM, n_ext] wide, so the
                    # MMAD n equals the L0B fractal count (fact 3).
                    ub_c_full = tla.make_tensor(
                        ubc_ptr,
                        tla.make_layout(
                            tla.make_shape(BM, n_ext), tla.make_stride(n_ext, 1)
                        ),
                    )
                    l0_c = tla.make_tensor_like(l0c_ptr, ub_c_full, tla.arch.L0Clayout)
                    # The fixpipe is sized by the destination origin: nSize =
                    # RoundUp(v_real, 8) for the tail, so lanes past V_REAL never
                    # reach UB (they are the zero-padded weight columns).
                    ub_c_fix = (
                        ub_c_full
                        if v_real == n_ext
                        else tla.make_tensor(
                            ubc_ptr,
                            tla.make_layout(
                                tla.make_shape(BM, n_ext),
                                tla.make_stride(n_ext, 1),
                                origin_shape=tla.make_shape(BM, v_real),
                            ),
                        )
                    )

                    # Prologue: stage the first K tile of A and B.  A is RowMajor
                    # [M, H] -> zN; B is the column-major [H, V] view of weight -> nZ.
                    gm_a_p = tla.tile_view(
                        gm_x, tla.make_shape(BM, BK), tla.make_coord(mt, c0)
                    )
                    gm_b_p = _b_tile(gm_w, c0, nt, n_ext)
                    l1_a_p = tla.make_tensor_like(l1a0_ptr, gm_a_p, tla.arch.zN)
                    l1_b_p = tla.make_tensor_like(l1b0_ptr, gm_b_p, tla.arch.nZ)
                    mutex_l1a0.lock(pipe=tla.pipes.MTE2)
                    mutex_l1b0.lock(pipe=tla.pipes.MTE2)
                    tla.copy(l1_a_p, gm_a_p)
                    tla.copy(l1_b_p, gm_b_p)
                    mutex_l1a0.unlock(pipe=tla.pipes.MTE2)
                    mutex_l1b0.unlock(pipe=tla.pipes.MTE2)

                    # L0C is free only after the previous tile's fixpipe has consumed it.
                    mutex_l0c.lock(pipe=tla.pipes.CUBE)

                    buf = tla.as_numeric(c0)
                    for kt in tla.range(c0, k_tiles, c1):
                        if buf == c0:
                            mutex_l1a0.lock(pipe=tla.pipes.MTE1)
                            mutex_l1b0.lock(pipe=tla.pipes.MTE1)
                        else:
                            mutex_l1a1.lock(pipe=tla.pipes.MTE1)
                            mutex_l1b1.lock(pipe=tla.pipes.MTE1)

                        # Prefetch the next K tile into the other stage.
                        if kt + c1 < k_tiles:
                            gm_a_n = tla.tile_view(
                                gm_x,
                                tla.make_shape(BM, BK),
                                tla.make_coord(mt, kt + c1),
                            )
                            gm_b_n = _b_tile(gm_w, kt + c1, nt, n_ext)
                            next_a_ptr = l1a1_ptr if buf == c0 else l1a0_ptr
                            next_b_ptr = l1b1_ptr if buf == c0 else l1b0_ptr
                            l1_a_n = tla.make_tensor_like(
                                next_a_ptr, gm_a_n, tla.arch.zN
                            )
                            l1_b_n = tla.make_tensor_like(
                                next_b_ptr, gm_b_n, tla.arch.nZ
                            )
                            if buf == c0:
                                mutex_l1a1.lock(pipe=tla.pipes.MTE2)
                                mutex_l1b1.lock(pipe=tla.pipes.MTE2)
                            else:
                                mutex_l1a0.lock(pipe=tla.pipes.MTE2)
                                mutex_l1b0.lock(pipe=tla.pipes.MTE2)
                            tla.copy(l1_a_n, gm_a_n)
                            tla.copy(l1_b_n, gm_b_n)
                            if buf == c0:
                                mutex_l1a1.unlock(pipe=tla.pipes.MTE2)
                                mutex_l1b1.unlock(pipe=tla.pipes.MTE2)
                            else:
                                mutex_l1a0.unlock(pipe=tla.pipes.MTE2)
                                mutex_l1b0.unlock(pipe=tla.pipes.MTE2)

                        cur_a_ptr = l1a0_ptr if buf == c0 else l1a1_ptr
                        cur_b_ptr = l1b0_ptr if buf == c0 else l1b1_ptr
                        gm_a_c = tla.tile_view(
                            gm_x, tla.make_shape(BM, BK), tla.make_coord(mt, kt)
                        )
                        gm_b_c = _b_tile(gm_w, kt, nt, n_ext)
                        l1_a = tla.make_tensor_like(cur_a_ptr, gm_a_c, tla.arch.zN)
                        l1_b = tla.make_tensor_like(cur_b_ptr, gm_b_c, tla.arch.nZ)

                        # Two L0 stages so the MTE1 copy of the next SK chunk overlaps the
                        # current MAC.
                        l1_a_p0 = tla.tile_view(
                            l1_a, tla.make_shape(BM, SK), tla.make_coord(c0, c0)
                        )
                        l1_b_p0 = tla.tile_view(
                            l1_b, tla.make_shape(SK, n_ext), tla.make_coord(c0, c0)
                        )
                        l0_a_p = tla.make_tensor_like(l0a0_ptr, l1_a_p0, tla.arch.zN)
                        l0_b_p = tla.make_tensor_like(l0b0_ptr, l1_b_p0, tla.arch.nZ)
                        mutex_l0a0.lock(pipe=tla.pipes.MTE1)
                        mutex_l0b0.lock(pipe=tla.pipes.MTE1)
                        tla.copy(l0_a_p, l1_a_p0)
                        tla.copy(l0_b_p, l1_b_p0)
                        mutex_l0a0.unlock(pipe=tla.pipes.MTE1)
                        mutex_l0b0.unlock(pipe=tla.pipes.MTE1)

                        lbuf = tla.as_numeric(c0)
                        for sk in tla.range(c0, SUB_K, c1):
                            if lbuf == c0:
                                mutex_l0a0.lock(pipe=tla.pipes.CUBE)
                                mutex_l0b0.lock(pipe=tla.pipes.CUBE)
                            else:
                                mutex_l0a1.lock(pipe=tla.pipes.CUBE)
                                mutex_l0b1.lock(pipe=tla.pipes.CUBE)

                            # Next-chunk views are pure metadata and built unconditionally;
                            # only the guarded copy below touches them.
                            l1_a_q = tla.tile_view(
                                l1_a,
                                tla.make_shape(BM, SK),
                                tla.make_coord(c0, sk + c1),
                            )
                            l1_b_q = tla.tile_view(
                                l1_b,
                                tla.make_shape(SK, n_ext),
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
                            if sk + c1 < SUB_K:
                                if lbuf == c0:
                                    mutex_l0a1.lock(pipe=tla.pipes.MTE1)
                                    mutex_l0b1.lock(pipe=tla.pipes.MTE1)
                                else:
                                    mutex_l0a0.lock(pipe=tla.pipes.MTE1)
                                    mutex_l0b0.lock(pipe=tla.pipes.MTE1)
                                tla.copy(l0_a_n, l1_a_q)
                                tla.copy(l0_b_n, l1_b_q)
                                if lbuf == c0:
                                    mutex_l0a1.unlock(pipe=tla.pipes.MTE1)
                                    mutex_l0b1.unlock(pipe=tla.pipes.MTE1)
                                else:
                                    mutex_l0a0.unlock(pipe=tla.pipes.MTE1)
                                    mutex_l0b0.unlock(pipe=tla.pipes.MTE1)

                            cur_l0a_ptr = l0a0_ptr if lbuf == c0 else l0a1_ptr
                            cur_l0b_ptr = l0b0_ptr if lbuf == c0 else l0b1_ptr
                            l1_a_k = tla.tile_view(
                                l1_a, tla.make_shape(BM, SK), tla.make_coord(c0, sk)
                            )
                            l1_b_k = tla.tile_view(
                                l1_b, tla.make_shape(SK, n_ext), tla.make_coord(sk, c0)
                            )
                            l0_a = tla.make_tensor_like(
                                cur_l0a_ptr, l1_a_k, tla.arch.zN
                            )
                            l0_b = tla.make_tensor_like(
                                cur_l0b_ptr, l1_b_k, tla.arch.nZ
                            )

                            # The first fragment initialises the accumulator; the last
                            # one closes the MMAD chain for the fixpipe (unit_flag bit 0).
                            init_c = True if kt == c0 and sk == c0 else False
                            unit_flag = (
                                0b11
                                if (kt + c1 >= k_tiles) and (sk + c1 >= SUB_K)
                                else 0b10
                            )
                            tla.mmad(
                                l0_c, l0_a, l0_b, init_c=init_c, unit_flag=unit_flag
                            )

                            if lbuf == c0:
                                mutex_l0a0.unlock(pipe=tla.pipes.CUBE)
                                mutex_l0b0.unlock(pipe=tla.pipes.CUBE)
                            else:
                                mutex_l0a1.unlock(pipe=tla.pipes.CUBE)
                                mutex_l0b1.unlock(pipe=tla.pipes.CUBE)
                            lbuf = c1 - lbuf

                        if buf == c0:
                            mutex_l1a0.unlock(pipe=tla.pipes.MTE1)
                            mutex_l1b0.unlock(pipe=tla.pipes.MTE1)
                        else:
                            mutex_l1a1.unlock(pipe=tla.pipes.MTE1)
                            mutex_l1b1.unlock(pipe=tla.pipes.MTE1)
                        buf = c1 - buf

                    # Hand the fp32 tile to the two AIV halves.  unit_flag=3 closes the
                    # accumulation chain so the fixpipe observes a complete tile.
                    mutex_l0c.unlock(pipe=tla.pipes.CUBE)
                    mutex_l0c.lock(pipe=tla.pipes.FIX)
                    tla.cross_core_wait_flag(exchange, tla.pipes.FIX)
                    tla.copy(
                        ub_c_fix,
                        l0_c,
                        tla.params.CopyL0C2DstParams(
                            unit_flag=3,
                            l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M,
                        ),
                    )
                    tla.cross_core_set_flag(exchange, tla.pipes.FIX)
                    mutex_l0c.unlock(pipe=tla.pipes.FIX)

    # ---------------------------------------------------------------- vector
    with tla.vector():
        sub = tla.arch.sub_block_idx()
        ub_lse = tla.make_tensor(
            ublse_ptr, tla.make_layout(tla.make_shape(HALF), tla.make_stride(1))
        )
        ub_scale = tla.make_tensor(
            ubscale_ptr, tla.make_layout(tla.make_shape(HALF), tla.make_stride(1))
        )

        # fp32 -> bf16 must round ties to even, matching GEMM / ND2NZ rounding.
        out_cast = tla.params.CastParams(
            reg_slot=tla.params.RegSlot.ZERO,
            sat_mode=tla.params.SatMode.NOSAT,
            round_mode=tla.params.RoundMode.CAST_RINT,
        )
        # PACK_B32 compacts the low bf16 half of each B32 register slot on the
        # way to UB, so no deinterleave is needed for a RowMajor store.  The
        # store mask must stay all-true: a PACK_B32 store of a [1, 64] fp32
        # fragment needs all 128 bf16 mask bits to write its 64 values (with
        # only the low 64 bits set it writes 32), so the tail lanes are kept out
        # of GM by the ub_y row padding below, not by a partial store mask.
        pack_store = tla.params.NormalStoreParams(
            store_dist=tla.params.StoreDist.DIST_PACK_B32
        )
        # One per-row scalar broadcast to every SIMD lane.
        broadcast_load = tla.params.NormalLoadParams(
            load_dist=tla.params.LoadDist.DIST_BRC_B32
        )

        # Open the exchange so the first fixpipe hand-off can proceed.
        tla.cross_core_set_flag(exchange, tla.pipes.VECTOR)

        # The vector region advances with the same block_idx / block_num and the
        # same two passes as the cube region (block_num is the cube core count);
        # sub_block_idx picks the SPLIT_M half this AIV owns.
        for p in tla.range_constexpr(0, 2):
            tasks, tiles_per_row, nt_off, n_ext, v_real = passes[p]
            if tla.const_expr(tasks > 0):
                # ub_c is v_real wide at the n_ext (fixpipe) row pitch.  ub_y rows
                # are padded to whole SIMD chunks (n_ub bf16, 32-byte aligned):
                # the lanes past v_real of the last chunk are stored into that
                # padding, which the MTE3 copy (v_real columns) never reads.
                n_ub = (v_real + VL - 1) // VL * VL
                ub_c = tla.make_tensor(
                    ubc_ptr,
                    tla.make_layout(
                        tla.make_shape(HALF, v_real), tla.make_stride(n_ext, 1)
                    ),
                )
                ub_y = tla.make_tensor(
                    uby_ptr,
                    tla.make_layout(
                        tla.make_shape(HALF, v_real), tla.make_stride(n_ub, 1)
                    ),
                )
                n_chunks = (v_real + VL - 1) // VL

                for task in tla.range(
                    tla.arch.block_idx(), tasks, tla.arch.block_num()
                ):
                    mt = task // tiles_per_row
                    nt = task - mt * tiles_per_row + nt_off
                    row0 = mt * BM + sub * HALF
                    tla.cross_core_wait_flag(exchange, tla.pipes.VECTOR)

                    # One thread per row: a 4-byte pitch is not a legal DMA tile, so the
                    # per-row scalars are read as scalars.
                    with tla.vec.func(mode="simt", thread_block_dim=HALF):
                        rtid, _, _ = tla.arch.thread_idx()
                        ub_lse[rtid] = gm_lse[row0 + rtid]
                        ub_scale[rtid] = gm_scale[row0 + rtid]
                    tla.pipe_barrier(tla.pipes.ALL)

                    # ---- payload: per-row epilogue, replace as a whole --------------
                    mutex_out.lock(pipe=tla.pipes.VECTOR)
                    with tla.vec.func(mode="simd"):
                        store_lanes = tla.create_mask(
                            pattern=tla.mask.ALL, dtype=tla.BFloat16
                        )
                        for row in tla.range(HALF):
                            lse_row = tla.tile_view(
                                ub_lse, tla.make_shape(1), tla.make_coord(row)
                            ).load(params=broadcast_load)
                            scale_row = tla.tile_view(
                                ub_scale, tla.make_shape(1), tla.make_coord(row)
                            ).load(params=broadcast_load)
                            # Lane j of chunk cb is valid iff cb * VL + j < v_real; the
                            # loop-carried counter yields an all-true mask for every
                            # full chunk and the tail mask for the last one, so lanes
                            # past V_REAL are not computed (their raw content is
                            # stored into the ub_y padding and never written to GM).
                            remaining = v_real
                            for cb in tla.range(c0, n_chunks, c1):
                                frag = tla.tile_view(
                                    ub_c, tla.make_shape(1, VL), tla.make_coord(row, cb)
                                ).load()
                                lanes, remaining = tla.update_mask(
                                    remaining, dtype=tla.Float32
                                )
                                frag = tla.sub(frag, lse_row, mask=lanes)
                                frag = tla.exp(frag, mask=lanes)
                                frag = tla.mul(frag, scale_row, mask=lanes)
                                tla.tile_view(
                                    ub_y, tla.make_shape(1, VL), tla.make_coord(row, cb)
                                ).store(
                                    frag.to(tla.BFloat16, out_cast, lanes),
                                    pack_store,
                                    mask=store_lanes,
                                )
                    mutex_out.unlock(pipe=tla.pipes.VECTOR)
                    # ---- end of payload ----------------------------------------------

                    tla.pipe_barrier(tla.pipes.ALL)

                    # Element-offset GM view for this AIV half: the base pointer already
                    # carries the half-tile row offset, so no tile-index scaling is
                    # involved (contrast with the tile coordinates used in the cube).
                    # Only the v_real valid columns are written; the row pitch is V.
                    gm_y_half = tla.make_tensor(
                        gm_y.ptr
                        + row0.to(tla.Int64) * tla.Int64(v)
                        + (nt * BN).to(tla.Int64),
                        tla.make_layout(
                            tla.make_shape(HALF, v_real), tla.make_stride(v, 1)
                        ),
                    )
                    mutex_out.lock(pipe=tla.pipes.MTE3)
                    tla.copy(gm_y_half, ub_y)
                    mutex_out.unlock(pipe=tla.pipes.MTE3)
                    tla.pipe_barrier(tla.pipes.ALL)
                    # Hand the fp32 staging half back to the fixpipe.
                    tla.cross_core_set_flag(exchange, tla.pipes.VECTOR)


# ======================================================================================
# HOST side: padding, compile and launch.
# ======================================================================================

_COMPILED = {}


def _cube_cores(index):
    import torch

    return int(torch.npu.get_device_properties(index).cube_core_num)


def run(x, weight, row_logsumexp, row_scale):
    """Return ``Y`` ``[M, V]`` bf16 = ``bf16(exp(x @ weight.T - lse) * scale)``.

    ``x`` is ``[M, H]`` bf16, ``weight`` is ``[V, H]`` bf16, and
    ``row_logsumexp`` / ``row_scale`` are ``[M]`` fp32.  ``H`` must be a multiple
    of ``BK``; ``V`` may be any positive size (``V % BN != 0`` is a tail tile).
    Rows are padded to ``BM`` on the host only when ``M`` is not tile aligned, so
    every GEMM tile is complete; padded rows are zero / neutral and are dropped
    from the returned view.
    """
    import torch
    from catlass.tla.runtime import from_dlpack

    if x.dim() != 2 or x.dtype != torch.bfloat16:
        raise ValueError("x must be 2-D bfloat16 [M, H]")
    if weight.dim() != 2 or weight.dtype != torch.bfloat16:
        raise ValueError("weight must be 2-D bfloat16 [V, H]")
    if weight.shape[1] != x.shape[1]:
        raise ValueError(
            f"weight must share H with x, got {weight.shape[1]} vs {x.shape[1]}"
        )
    m, h = x.shape
    v = weight.shape[0]
    for name, t in (("row_logsumexp", row_logsumexp), ("row_scale", row_scale)):
        if t.dim() != 1 or t.dtype != torch.float32 or t.shape[0] != m:
            raise ValueError(f"{name} must be 1-D float32 with M entries")
    if h == 0 or h % BK != 0:
        raise ValueError(f"H must be a positive multiple of BK={BK}, got {h}")
    if m == 0 or v == 0:
        return torch.empty((m, v), dtype=torch.bfloat16, device=x.device)

    device = x.device
    device_index = int(str(device).split(":")[-1]) if ":" in str(device) else 0
    # The launch goes through the ACL context of the current device; make it the
    # device of ``x`` before allocating anything (otherwise the runtime reports an
    # invalid handle on launch).
    torch.npu.set_device(device_index)

    # Pad rows to whole BM tiles.  MMAD needs ceil(acc_M / 16) == ceil(A_M / 16)
    # (see fact 3 in the module docstring), so a partial A tile is not an option:
    # the zero rows keep every tile complete and give the tail rows well-defined
    # values (exp(0 - 0) * 1 = 1).  When M already covers whole tiles the inputs
    # are bound as they are: no extra allocation and no copy (``contiguous`` is a
    # no-op on a contiguous tensor), the same rule as the V-side padding below.
    m_pad = (m + BM - 1) // BM * BM
    if m_pad != m:
        x_buf = torch.zeros((m_pad, h), dtype=torch.bfloat16, device=device)
        x_buf[:m] = x
        lse_buf = torch.zeros((m_pad,), dtype=torch.float32, device=device)
        lse_buf[:m] = row_logsumexp
        scale_buf = torch.ones((m_pad,), dtype=torch.float32, device=device)
        scale_buf[:m] = row_scale
    else:
        x_buf = x.contiguous()
        lse_buf = row_logsumexp.contiguous()
        scale_buf = row_scale.contiguous()
    y_buf = torch.empty((m_pad, v), dtype=torch.bfloat16, device=device)

    x_t = from_dlpack(x_buf, layout_tag=tla.arch.RowMajor)
    # The [V, H] row-major buffer is the column-major [H, V] matrix W^T, which is
    # exactly the logical shape the cube needs for B: a [BK, N_EXT] tile is [K, N].
    # No host transpose is required.
    #
    # V tail: the last tile column spans N_EXT = RoundUp(V_real, 16) columns, and
    # fact 3 forbids a B tile cropped by its origin, so B must offer N_EXT
    # columns.  When V is a multiple of 16 that is already true (N_EXT == V_real);
    # otherwise the weight rows (== B columns) are zero-padded to RoundUp(V, 16).
    # The padded columns only produce exp(-lse) * scale in lanes the kernel never
    # writes.  This is the only case run() copies the weight.
    w = weight.contiguous()
    v_w = (v + FRACTAL - 1) // FRACTAL * FRACTAL
    if v_w != v:
        w_pad = torch.zeros((v_w, h), dtype=torch.bfloat16, device=device)
        w_pad[:v] = w
        w = w_pad
    w_t = from_dlpack(w, layout_tag=tla.arch.ColumnMajor)
    lse_t = from_dlpack(lse_buf, layout_tag=tla.arch.RowMajor)
    scale_t = from_dlpack(scale_buf, layout_tag=tla.arch.RowMajor)
    y_t = from_dlpack(y_buf, layout_tag=tla.arch.RowMajor)

    cube_cores = _cube_cores(device_index)

    key = (device_index, m_pad, h, v)
    artifact = _COMPILED.get(key)
    if artifact is None:
        artifact = tla.compile(
            matmul_exp_scale_ub_kernel,
            x_t,
            w_t,
            lse_t,
            scale_t,
            y_t,
            options="--npu-arch 3510",
        )
        _COMPILED[key] = artifact

    # A MIX kernel is launched on the cube cores; each brings its two AIVs.
    artifact(x_t, w_t, lse_t, scale_t, y_t, block_num=cube_cores)
    return y_buf[:m]
