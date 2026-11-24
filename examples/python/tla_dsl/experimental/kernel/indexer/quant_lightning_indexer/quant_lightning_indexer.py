"""quant_lightning_indexer on CATLASS DSL (catlass.tla), Ascend 950.

Two kernels, launched in order:

* ``qli_scores`` (AIC/AIV mixed): MXFP4 QK and the per-head ReLU/weighted
  reduction into a row-major bf16 score matrix.
* ``qli_topk`` (fused AIV): monotonic u16 keys, two-level CDF selection and
  emission in one kernel, with three device-wide ``tla_sync_all()`` barriers
  replacing the stage launches.

Score semantics
---------------
    qk[r, h, s] = bf16(sum_d decode(q[r, h])[d] * decode(k[s, 0])[d])
    score[r, s] = fold_h bf16(bf16(max(qk[r, h, s], 0) * bf16(w[r, h])) + acc)

The reduction rounds to bf16 after every head, so the epilogue keeps that exact
rounding order (bf16(qk), ReLU, bf16(w), bf16(acc + qk*w)) instead of using a
fused multiply-add. Top-k is exact: the high-byte CDF picks the 256-wide bin
that contains rank K and the low-byte CDF resolves the exact key inside it.

High-performance design points
------------------------------
S1:
* One (query tile, page group) task per AI core (``groups = block_num //
  tiles_q`` rounds down), so no core owns a second task.
* AIC/AIV double buffering over ``raw_ub0/1`` and ``score_ub0/1``: the cube
  runs one page ahead of the vector epilogue, one ``ready``/``free`` cross flag
  pair per slot.
* Two 64-column accumulator chains are interleaved, and scores are written with
  ``DIST_PACK_B32`` straight from the accumulator registers.
* Explicit ``tla.mutex`` ownership of every on-chip buffer replaces
  ``auto_sync="v0"``; the AIV releases ``raw_ub`` as soon as the SIMD block
  retires, before the slower UB->GM copy.

S2:
* Row sharding: ``share = block_num // rows`` (capped at 8) splits one row's
  key blocks over several AIVs, so ``rows=12`` still keeps all 56 AIVs busy.
* Four stages in one kernel, separated by three ``tla_sync_all()`` barriers;
  only shard partials (``rpart``/``cpart``/``scal``) are handed over, through
  DMA because ``SyncAll`` does not make scalar GM writes visible across blocks.
* Emit compacts in registers: ``gt`` plus ranked ``eq`` (ACCUMULATE histogram +
  gather), ``squeeze`` for values and lane offsets, chunk-fixed slots, and a
  short SIMT pass for the global prefix.

Synchronization scope
---------------------
``tla.mutex`` orders pipes inside a single AIC or AIV only, so the AIC->AIV
page hand-off in S1 stays on ``tla.cross_flag`` (mode 4, addressed per AIV):
a mutex has no "held at kernel start" state and readiness must be an event.
"""

from __future__ import annotations

import catlass.tla as tla


MX_SCALE_GROUP_NUM = 32


@tla.kernel
def qli_scores(
    q: tla.Tensor,  # fp4 packed [rows*32, 128]
    k: tla.Tensor,  # fp4 packed, column-major (128, pages*page)
    sq: tla.Tensor,  # e8m0 [rows*32, 4] row-major
    sk: tla.Tensor,  # e8m0 column-major (4, pages*page)
    w: tla.Tensor,  # f32 [rows, 32]
    block_table: tla.Tensor,  # i32 [1, pages]
    scores: tla.Tensor,  # bf16 [rows, pages*page]
    page_size: tla.Constexpr[int],
    rows: tla.Constexpr[int],
    query_heads: tla.Constexpr[int],
    head_dim: tla.Constexpr[int],
    tile_rows: tla.Constexpr[int],
) -> None:
    dtype_a = q.ptr.dtype
    dtype_b = k.ptr.dtype

    tile_m = tile_rows * query_heads  # 4 * 32 = 128
    pages = block_table.origin_shape[1]
    max_blocks = pages * page_size
    sk_cols = (head_dim // MX_SCALE_GROUP_NUM + 1) // 2 * 2
    k_tiles = (head_dim + 128 - 1) // 128

    # --- On-chip buffers ---------------------------------------------------
    # L1/L0 tiles are single buffered: one MMAD is in flight at a time and the
    # L0C->UB (FIX) copy is the only consumer of L0C. Cross-core double
    # buffering lives in UB (raw/score slots) where it buys cube/vector overlap.
    l1_q = tla.allocate(tile_m * head_dim, dtype_a, tla.AddressSpace.l1, 512)
    l1_k = tla.allocate(head_dim * page_size, dtype_b, tla.AddressSpace.l1, 512)
    l1_sq = tla.allocate(tile_m * sk_cols, tla.Float8E8M0, tla.AddressSpace.l1, 512)
    l1_sk = tla.allocate(sk_cols * page_size, tla.Float8E8M0, tla.AddressSpace.l1, 512)
    l0_q = tla.allocate(tile_m * head_dim, dtype_a, tla.AddressSpace.l0a, 1024)
    l0_k = tla.allocate(head_dim * page_size, dtype_b, tla.AddressSpace.l0b, 1024)
    l0_c = tla.allocate(tile_m * page_size, tla.Float32, tla.AddressSpace.l0c, 1024)
    raw_ub0 = tla.allocate(tile_m * page_size, tla.Float32, tla.AddressSpace.ub, 256)
    raw_ub1 = tla.allocate(tile_m * page_size, tla.Float32, tla.AddressSpace.ub, 256)
    raw_slots = (raw_ub0, raw_ub1)
    score_slots = (
        tla.allocate(
            (tile_m // 2) // query_heads * page_size,
            tla.BFloat16,
            tla.AddressSpace.ub,
            256,
        ),
        tla.allocate(
            (tile_m // 2) // query_heads * page_size,
            tla.BFloat16,
            tla.AddressSpace.ub,
            256,
        ),
    )
    w_ub = tla.allocate(tile_rows * query_heads, tla.Float32, tla.AddressSpace.ub, 256)

    # --- Intra-core buffer ownership (mutex) --------------------------------
    # One mutex per buffer, locked on each pipe that touches it: MTE2/MTE1 for
    # the GM->L1->L0 loads, CUBE/FIX for L0C, VECTOR/MTE3 for the score slots,
    # MTE2/VECTOR for the weight tile. Ids are explicit (0..31 is mandatory).
    mutex_l1q = tla.mutex(resource="qli_l1q", id=0)
    mutex_l1sq = tla.mutex(resource="qli_l1sq", id=1)
    mutex_l1k = tla.mutex(resource="qli_l1k", id=2)
    mutex_l1sk = tla.mutex(resource="qli_l1sk", id=3)
    mutex_l0q = tla.mutex(resource="qli_l0q", id=4)
    mutex_l0k = tla.mutex(resource="qli_l0k", id=5)
    mutex_l0c = tla.mutex(resource="qli_l0c", id=6)
    mutex_w = tla.mutex(resource="qli_w", id=7)
    score_mutexes = (
        tla.mutex(resource="qli_score0", id=8),
        tla.mutex(resource="qli_score1", id=9),
    )

    # --- Cross-core page hand-off (event, not lock) -------------------------
    # mutex cannot cross AIC/AIV, and it has no initially-held state, so the
    # page token stays a cross flag: AIC raises ``ready`` after the FIX copy,
    # each AIV lowers ``free`` after the last read of that slot.
    ready_flags = (
        tla.cross_flag("qli_ready_0", mode=4),
        tla.cross_flag("qli_ready_1", mode=4),
    )
    free_flags = (
        tla.cross_flag("qli_free_0", mode=4),
        tla.cross_flag("qli_free_1", mode=4),
    )
    tiles_q = (rows + tile_rows - 1) // tile_rows
    groups = tla.arch.block_num() // tiles_q
    if groups < 1:
        groups = 1
    if groups > pages:
        groups = pages
    per_group = (pages + groups - 1) // groups
    task_count = tiles_q * groups

    with tla.cube():
        for cube_task in tla.range(
            tla.arch.block_idx(), task_count, tla.arch.block_num()
        ):
            cube_qt = cube_task // groups
            cube_pg = cube_task % groups
            page_begin = cube_pg * per_group
            page_end = page_begin + per_group
            if page_end > pages:
                page_end = pages
            gm_q = tla.tile_view(
                q, tla.make_shape(tile_m, head_dim), tla.make_coord(cube_qt, 0)
            )
            gm_sq = tla.tile_view(
                sq, tla.make_shape(tile_m, sk_cols), tla.make_coord(cube_qt, 0)
            )
            l1_q_t = tla.make_tensor_like(l1_q, gm_q, tla.arch.zN)
            l1_sq_t = tla.make_tensor_like(l1_sq, gm_sq, tla.arch.zZMxScale)
            mutex_l1q.lock(pipe=tla.arch.MTE2)
            tla.copy(l1_q_t, gm_q)
            mutex_l1q.unlock(pipe=tla.arch.MTE2)
            mutex_l1sq.lock(pipe=tla.arch.MTE2)
            tla.copy(l1_sq_t, gm_sq)
            mutex_l1sq.unlock(pipe=tla.arch.MTE2)
            l0_q_t = tla.make_tensor_like(l0_q, l1_q_t, tla.arch.zN)
            # Q and its E8M0 scale move L1->L0A once per task; the three
            # mutexes state that both L1 buffers are consumed and L0A written.
            mutex_l1q.lock(pipe=tla.arch.MTE1)
            mutex_l1sq.lock(pipe=tla.arch.MTE1)
            mutex_l0q.lock(pipe=tla.arch.MTE1)
            tla.copy(l0_q_t, l1_q_t, scale=l1_sq_t)
            mutex_l0q.unlock(pipe=tla.arch.MTE1)
            mutex_l1sq.unlock(pipe=tla.arch.MTE1)
            mutex_l1q.unlock(pipe=tla.arch.MTE1)
            l0_c_t = tla.make_tensor_like(l0_c, gm_q, tla.arch.L0Clayout)
            page_n = page_end - page_begin
            pair_n = (page_n + 1) // 2
            for pair_i in tla.range(0, pair_n, 1):
                for slot in (0, 1):
                    cube_page = page_begin + pair_i * 2 + slot
                    if cube_page >= page_end:
                        cube_page = page_end - 1
                    # Wait until both AIVs have consumed the previous page of
                    # this slot before the FIX copy overwrites raw_ub[slot].
                    tla.cross_core_wait_flag(free_flags[slot], tla.arch.FIX, aiv_id=0)
                    tla.cross_core_wait_flag(free_flags[slot], tla.arch.FIX, aiv_id=1)
                    physical_page = block_table[0, cube_page]
                    gm_k = tla.tile_view(
                        k,
                        tla.make_shape(head_dim, page_size),
                        tla.make_coord(0, physical_page),
                    )
                    gm_sk = tla.tile_view(
                        sk,
                        tla.make_shape(sk_cols, page_size),
                        tla.make_coord(0, physical_page),
                    )
                    l1_k_t = tla.make_tensor_like(l1_k, gm_k, tla.arch.nZ)
                    l1_sk_t = tla.make_tensor_like(l1_sk, gm_sk, tla.arch.nNMxScale)
                    mutex_l1k.lock(pipe=tla.arch.MTE2)
                    tla.copy(l1_k_t, gm_k)
                    mutex_l1k.unlock(pipe=tla.arch.MTE2)
                    mutex_l1sk.lock(pipe=tla.arch.MTE2)
                    tla.copy(l1_sk_t, gm_sk)
                    mutex_l1sk.unlock(pipe=tla.arch.MTE2)

                    l0_k_t = tla.make_tensor_like(l0_k, l1_k_t, tla.arch.nZ)
                    mutex_l1k.lock(pipe=tla.arch.MTE1)
                    mutex_l1sk.lock(pipe=tla.arch.MTE1)
                    mutex_l0k.lock(pipe=tla.arch.MTE1)
                    tla.copy(l0_k_t, l1_k_t, scale=l1_sk_t)
                    mutex_l0k.unlock(pipe=tla.arch.MTE1)
                    mutex_l1sk.unlock(pipe=tla.arch.MTE1)
                    mutex_l1k.unlock(pipe=tla.arch.MTE1)

                    # mmad_mx is not recognised by mutex_guard, so lock the
                    # three L0 tiles explicitly on the CUBE pipe.
                    mutex_l0q.lock(pipe=tla.arch.CUBE)
                    mutex_l0k.lock(pipe=tla.arch.CUBE)
                    mutex_l0c.lock(pipe=tla.arch.CUBE)
                    tla.mmad_mx(l0_c_t, l0_q_t, l0_k_t, init_c=True)
                    mutex_l0c.unlock(pipe=tla.arch.CUBE)
                    mutex_l0k.unlock(pipe=tla.arch.CUBE)
                    mutex_l0q.unlock(pipe=tla.arch.CUBE)

                    # L0C -> UB on FIX; the mutex makes the next MMAD wait for
                    # this fixpipe read before it overwrites L0C.
                    raw_ub_t = tla.make_tensor(
                        raw_slots[slot],
                        tla.make_layout(
                            tla.make_shape(tile_m, page_size),
                            tla.make_stride(page_size, 1),
                            layoutTag=tla.arch.RowMajor,
                        ),
                    )
                    mutex_l0c.lock(pipe=tla.arch.FIX)
                    tla.copy(
                        raw_ub_t,
                        l0_c_t,
                        tla.params.CopyL0C2DstParams(
                            l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M
                        ),
                    )
                    mutex_l0c.unlock(pipe=tla.arch.FIX)
                    tla.cross_core_set_flag(ready_flags[slot], tla.arch.FIX, aiv_id=0)
                    tla.cross_core_set_flag(ready_flags[slot], tla.arch.FIX, aiv_id=1)
        tla.pipe_barrier(tla.pipes.ALL)

    with tla.vector():
        sub = tla.arch.sub_block_idx()
        # Both slots start free, matching the first two AIC waits.
        for prime_slot in (0, 1):
            tla.cross_core_set_flag(free_flags[prime_slot], tla.arch.VECTOR, aiv_id=0)
            tla.cross_core_set_flag(free_flags[prime_slot], tla.arch.VECTOR, aiv_id=1)
        for vector_task in tla.range(
            tla.arch.block_idx(), task_count, tla.arch.block_num()
        ):
            vector_qt = vector_task // groups
            vector_pg = vector_task % groups
            page_begin = vector_pg * per_group
            page_end = page_begin + per_group
            if page_end > pages:
                page_end = pages
            row_start = vector_qt * tile_rows
            aiv_rows = rows - row_start
            if aiv_rows > tile_rows:
                aiv_rows = tile_rows
            aiv_row_start = row_start + sub * ((aiv_rows + 1) // 2)
            aiv_m = aiv_rows - sub * ((aiv_rows + 1) // 2)
            if aiv_m > (tile_rows + 1) // 2:
                aiv_m = (tile_rows + 1) // 2
            gm_w = tla.make_tensor(
                w.ptr + aiv_row_start * query_heads,
                tla.make_layout(
                    tla.make_shape(tile_rows, query_heads),
                    tla.make_stride(query_heads, 1),
                    origin_shape=tla.make_shape(aiv_m, query_heads),
                    layoutTag=tla.arch.RowMajor,
                ),
            )
            w_ub_t = tla.make_tensor(
                w_ub,
                tla.make_layout(
                    tla.make_shape(tile_rows, query_heads),
                    tla.make_stride(query_heads, 1),
                    origin_shape=tla.make_shape(aiv_m, query_heads),
                ),
            )
            mutex_w.lock(pipe=tla.arch.MTE2)
            tla.copy(w_ub_t, gm_w)
            mutex_w.unlock(pipe=tla.arch.MTE2)
            w_flat = tla.make_tensor(
                w_ub,
                tla.make_layout(
                    tla.make_shape(tile_rows * query_heads), tla.make_stride(1)
                ),
            )
            # Hold the weight tile on VECTOR for the whole task: it is read
            # (broadcast) by every page of this task, so MTE2 must not reload
            # it until the last page is done.
            mutex_w.lock(pipe=tla.arch.VECTOR)
            with tla.vec.func(mode="simd"):
                w_cast_rint = tla.params.CastParams(
                    reg_slot=tla.params.RegSlot.ZERO,
                    sat_mode=tla.params.SatMode.SAT,
                    round_mode=tla.params.RoundMode.CAST_RINT,
                )
                w_cast_wide = tla.params.CastParams(
                    reg_slot=tla.params.RegSlot.ZERO,
                    sat_mode=tla.params.SatMode.NOSAT,
                    round_mode=tla.params.RoundMode.CAST_NONE,
                )
                w_mask_f, _ = tla.update_mask(64, tla.Float32)
                w_mask_b, _ = tla.update_mask(128, tla.BFloat16)
                for w_chunk in tla.range_constexpr(
                    0, (tile_rows * query_heads + 63) // 64, 1
                ):
                    wv_t = tla.tile_view(
                        w_flat, tla.make_shape(64), tla.make_coord(w_chunk)
                    )
                    wv = wv_t.load()
                    wv_bf = wv.to(tla.BFloat16, w_cast_rint, w_mask_f)
                    wv = wv_bf.to(tla.Float32, w_cast_wide, w_mask_b)
                    wv_t.store(wv, mask=w_mask_f)
            page_n = page_end - page_begin
            pair_n = (page_n + 1) // 2
            for pair_i in tla.range(0, pair_n, 1):
                for slot in (0, 1):
                    vector_page = page_begin + pair_i * 2 + slot
                    if vector_page >= page_end:
                        vector_page = page_end - 1
                    tla.cross_core_wait_flag(
                        ready_flags[slot], tla.arch.VECTOR, aiv_id=0
                    )
                    tla.cross_core_wait_flag(
                        ready_flags[slot], tla.arch.VECTOR, aiv_id=1
                    )
                    raw_aiv = tla.make_tensor(
                        raw_slots[slot],
                        tla.make_layout(
                            tla.make_shape(tile_m // 2, page_size),
                            tla.make_stride(page_size, 1),
                            origin_shape=tla.make_shape(aiv_m * query_heads, page_size),
                        ),
                    )
                    score_ub_view = tla.make_tensor(
                        score_slots[slot],
                        tla.make_layout(
                            tla.make_shape(tile_m // 2 // query_heads, page_size),
                            tla.make_stride(page_size, 1),
                            origin_shape=tla.make_shape(aiv_m, page_size),
                        ),
                    )
                    # Vector owns score_ub[slot] until the stores retire; the
                    # same mutex on MTE3 orders the DMA read after them.
                    mutex_score = score_mutexes[slot]
                    mutex_score.lock(pipe=tla.arch.VECTOR)
                    with tla.vec.func(mode="simd"):
                        zero_f = tla.full(0.0, tla.Float32)
                        cast_rint = tla.params.CastParams(
                            reg_slot=tla.params.RegSlot.ZERO,
                            sat_mode=tla.params.SatMode.SAT,
                            round_mode=tla.params.RoundMode.CAST_RINT,
                        )
                        cast_wide = tla.params.CastParams(
                            reg_slot=tla.params.RegSlot.ZERO,
                            sat_mode=tla.params.SatMode.NOSAT,
                            round_mode=tla.params.RoundMode.CAST_NONE,
                        )
                        pack_store = tla.params.NormalStoreParams(
                            store_dist=tla.params.StoreDist.DIST_PACK_B32
                        )
                        mask_f0, _ = tla.update_mask(64, tla.Float32)
                        mask_b0, _ = tla.update_mask(128, tla.BFloat16)
                        if (page_size + 63) // 64 == 2:
                            active1 = page_size - 64
                            if active1 > 64:
                                active1 = 64
                            mask_f1, _ = tla.update_mask(active1, tla.Float32)
                            mask_b1, _ = tla.update_mask(active1 * 2, tla.BFloat16)
                            for score_row in tla.range(0, aiv_m, 1):
                                # Two independent 64-column chains keep the
                                # vector pipeline busy on every head.
                                acc0 = tla.full(0.0, tla.Float32)
                                acc1 = tla.full(0.0, tla.Float32)
                                for head in tla.range_constexpr(0, query_heads, 1):
                                    w_reg = tla.tile_view(
                                        w_flat,
                                        tla.make_shape(1),
                                        tla.make_coord(score_row * query_heads + head),
                                    ).load(
                                        params=tla.params.NormalLoadParams(
                                            load_dist=tla.params.LoadDist.DIST_BRC_B32
                                        )
                                    )
                                    h0 = tla.tile_view(
                                        raw_aiv,
                                        tla.make_shape(1, 64),
                                        tla.make_coord(
                                            score_row * query_heads + head, 0
                                        ),
                                    )
                                    qk0 = h0.load()
                                    qk0_bf = qk0.to(tla.BFloat16, cast_rint, mask_f0)
                                    qk0 = qk0_bf.to(tla.Float32, cast_wide, mask_b0)
                                    qk0 = tla.max(qk0, zero_f, mask=mask_f0)
                                    acc0 = tla.add(
                                        acc0,
                                        tla.mul(qk0, w_reg, mask=mask_f0),
                                        mask=mask_f0,
                                    )
                                    acc0_bf = acc0.to(tla.BFloat16, cast_rint, mask_f0)
                                    acc0 = acc0_bf.to(tla.Float32, cast_wide, mask_b0)
                                    h1 = tla.tile_view(
                                        raw_aiv,
                                        tla.make_shape(1, 64),
                                        tla.make_coord(
                                            score_row * query_heads + head, 1
                                        ),
                                    )
                                    qk1 = h1.load()
                                    qk1_bf = qk1.to(tla.BFloat16, cast_rint, mask_f1)
                                    qk1 = qk1_bf.to(tla.Float32, cast_wide, mask_b1)
                                    qk1 = tla.max(qk1, zero_f, mask=mask_f1)
                                    acc1 = tla.add(
                                        acc1,
                                        tla.mul(qk1, w_reg, mask=mask_f1),
                                        mask=mask_f1,
                                    )
                                    acc1_bf = acc1.to(tla.BFloat16, cast_rint, mask_f1)
                                    acc1 = acc1_bf.to(tla.Float32, cast_wide, mask_b1)
                                tla.tile_view(
                                    score_ub_view,
                                    tla.make_shape(1, 64),
                                    tla.make_coord(score_row, 0),
                                ).store(
                                    acc0.to(tla.BFloat16, cast_rint, mask_f0),
                                    pack_store,
                                    mask=mask_b0,
                                )
                                tla.tile_view(
                                    score_ub_view,
                                    tla.make_shape(1, 64),
                                    tla.make_coord(score_row, 1),
                                ).store(
                                    acc1.to(tla.BFloat16, cast_rint, mask_f1),
                                    pack_store,
                                    mask=mask_b1,
                                )
                        else:
                            for score_row in tla.range(0, aiv_m, 1):
                                acc0 = tla.full(0.0, tla.Float32)
                                for head in tla.range_constexpr(0, query_heads, 1):
                                    w_reg = tla.tile_view(
                                        w_flat,
                                        tla.make_shape(1),
                                        tla.make_coord(score_row * query_heads + head),
                                    ).load(
                                        params=tla.params.NormalLoadParams(
                                            load_dist=tla.params.LoadDist.DIST_BRC_B32
                                        )
                                    )
                                    h0 = tla.tile_view(
                                        raw_aiv,
                                        tla.make_shape(1, 64),
                                        tla.make_coord(
                                            score_row * query_heads + head, 0
                                        ),
                                    )
                                    qk0 = h0.load()
                                    qk0_bf = qk0.to(tla.BFloat16, cast_rint, mask_f0)
                                    qk0 = qk0_bf.to(tla.Float32, cast_wide, mask_b0)
                                    qk0 = tla.max(qk0, zero_f, mask=mask_f0)
                                    acc0 = tla.add(
                                        acc0,
                                        tla.mul(qk0, w_reg, mask=mask_f0),
                                        mask=mask_f0,
                                    )
                                    acc0_bf = acc0.to(tla.BFloat16, cast_rint, mask_f0)
                                    acc0 = acc0_bf.to(tla.Float32, cast_wide, mask_b0)
                                tla.tile_view(
                                    score_ub_view,
                                    tla.make_shape(1, 64),
                                    tla.make_coord(score_row, 0),
                                ).store(
                                    acc0.to(tla.BFloat16, cast_rint, mask_f0),
                                    pack_store,
                                    mask=mask_b0,
                                )
                    mutex_score.unlock(pipe=tla.arch.VECTOR)
                    # raw_ub[slot] is no longer read past this point: release
                    # the slot before the slower UB->GM copy of score_ub.
                    tla.cross_core_set_flag(free_flags[slot], tla.arch.VECTOR, aiv_id=0)
                    tla.cross_core_set_flag(free_flags[slot], tla.arch.VECTOR, aiv_id=1)
                    score_gm = tla.make_tensor(
                        scores.ptr
                        + aiv_row_start * max_blocks
                        + vector_page * page_size,
                        tla.make_layout(
                            tla.make_shape(aiv_m, page_size),
                            tla.make_stride(max_blocks, 1),
                            origin_shape=tla.make_shape(aiv_m, page_size),
                            layoutTag=tla.arch.RowMajor,
                        ),
                    )
                    score_ub_t = tla.make_tensor(
                        score_slots[slot],
                        tla.make_layout(
                            tla.make_shape(tile_m // 2 // query_heads, page_size),
                            tla.make_stride(page_size, 1),
                            origin_shape=tla.make_shape(aiv_m, page_size),
                        ),
                    )
                    mutex_score.lock(pipe=tla.arch.MTE3)
                    tla.copy(score_gm, score_ub_t)
                    mutex_score.unlock(pipe=tla.arch.MTE3)
            mutex_w.unlock(pipe=tla.arch.VECTOR)
        tla.pipe_barrier(tla.pipes.ALL)


# --- S2: fused top-k selection ---------------------------------------------

_SYNC_ALL_SOURCE = r"""
#include "kernel_operator.h"
extern "C" {
[aicore] __attribute__((always_inline)) void tla_sync_all() {
    AscendC::SyncAll();
}
}
"""


@tla.extern(source=_SYNC_ALL_SOURCE, name="tla_sync_all")
def tla_sync_all() -> None: ...


CHUNK = 128
MAX_BLOCK_CHUNKS = 128
SIMT_THREADS = 256
N_BINS = 256
MAX_SHARE = 8
EMIT_CHUNKS = 32
N_SCAL = 8


@tla.kernel
def qli_topk(
    scores: tla.Tensor,  # bf16 [rows, max_k]
    visible: tla.Tensor,  # i32  [rows]
    offsets: tla.Tensor,  # i32  [rows]
    keys: tla.Tensor,  # u16  [rows, max_k]
    rpart: tla.Tensor,  # i32  [rows, MAX_SHARE, 512]
    scal: tla.Tensor,  # i32  [rows, N_SCAL]
    cpart: tla.Tensor,  # i16  [rows, 2*n_chunks]
    out_idx: tla.Tensor,  # i32  [rows, topk]
    out_val: tla.Tensor,  # bf16 [rows, topk]
    rows: tla.Constexpr[int],
    max_k: tla.Constexpr[int],
    topk: tla.Constexpr[int],
) -> None:
    """Fused S2: keys -> low -> cnt -> emit in one launch.

    The four stages keep their split of the work but replace the launch
    boundaries with ``tla_sync_all()`` device-wide barriers, so the shard
    partials (``rpart``/``cpart``/``scal``) stay hot and only one kernel
    launch is needed. ``auto_sync`` is left unset because ``auto_sync='v0'``
    rejects ``tla.call_extern``; every intra-core transfer therefore carries
    an explicit ``tla.pipe_barrier``.
    """
    n_chunks_1 = max_k // CHUNK
    block_chunks_1 = n_chunks_1 if n_chunks_1 < MAX_BLOCK_CHUNKS else MAX_BLOCK_CHUNKS
    block_elems_1 = block_chunks_1 * CHUNK
    n_blocks_1 = n_chunks_1 // block_chunks_1

    scores_ub_1 = tla.allocate(block_elems_1, tla.BFloat16, tla.AddressSpace.ub, 256)
    keys_ub_1 = tla.allocate(block_elems_1, tla.UInt16, tla.AddressSpace.ub, 256)
    hist_ub_1 = tla.allocate(
        n_blocks_1 * N_BINS + 64, tla.Int16, tla.AddressSpace.ub, 256
    )
    shard_ub_1 = tla.allocate(N_BINS + 16, tla.Int32, tla.AddressSpace.ub, 256)

    flat_layout_1 = tla.make_layout(tla.make_shape(block_elems_1), tla.make_stride(1))
    scores_flat_1 = tla.make_tensor(scores_ub_1, flat_layout_1)
    keys_flat_1 = tla.make_tensor(keys_ub_1, flat_layout_1)
    hist_t_1 = tla.make_tensor(
        hist_ub_1,
        tla.make_layout(tla.make_shape(n_blocks_1 * N_BINS + 64), tla.make_stride(1)),
    )
    shard_t_1 = tla.make_tensor(
        shard_ub_1,
        tla.make_layout(tla.make_shape(N_BINS + 16), tla.make_stride(1)),
    )

    n_chunks_2 = max_k // CHUNK
    block_chunks_2 = n_chunks_2 if n_chunks_2 < MAX_BLOCK_CHUNKS else MAX_BLOCK_CHUNKS
    block_elems_2 = block_chunks_2 * CHUNK
    n_blocks_2 = n_chunks_2 // block_chunks_2

    keys_ub_2 = tla.allocate(block_elems_2, tla.UInt16, tla.AddressSpace.ub, 256)
    hist_ub_2 = tla.allocate(
        n_blocks_2 * N_BINS + 64, tla.Int16, tla.AddressSpace.ub, 256
    )
    cdf_ub_2 = tla.allocate(N_BINS + 16, tla.Int32, tla.AddressSpace.ub, 256)
    shard_ub_2 = tla.allocate(N_BINS + 16, tla.Int32, tla.AddressSpace.ub, 256)

    keys_flat_2 = tla.make_tensor(
        keys_ub_2, tla.make_layout(tla.make_shape(block_elems_2), tla.make_stride(1))
    )
    hist_t_2 = tla.make_tensor(
        hist_ub_2,
        tla.make_layout(tla.make_shape(n_blocks_2 * N_BINS + 64), tla.make_stride(1)),
    )
    cdf_t_2 = tla.make_tensor(
        cdf_ub_2,
        tla.make_layout(tla.make_shape(N_BINS + 16), tla.make_stride(1)),
    )
    cdf_u16_t_2 = tla.make_tensor(
        tla.recast_ptr(cdf_ub_2, dtype=tla.UInt16),
        tla.make_layout(tla.make_shape(2 * N_BINS + 32), tla.make_stride(1)),
    )
    shard_t_2 = tla.make_tensor(
        shard_ub_2,
        tla.make_layout(tla.make_shape(N_BINS + 16), tla.make_stride(1)),
    )
    rpart_flat_2 = tla.make_tensor(
        rpart.ptr,
        tla.make_layout(tla.make_shape(rows * MAX_SHARE * 512), tla.make_stride(1)),
    )

    n_chunks_3 = max_k // CHUNK
    block_chunks_3 = n_chunks_3 if n_chunks_3 < MAX_BLOCK_CHUNKS else MAX_BLOCK_CHUNKS
    block_elems_3 = block_chunks_3 * CHUNK
    n_blocks_3 = n_chunks_3 // block_chunks_3
    gt_slot_3 = 0
    eq_slot_3 = n_chunks_3

    keys_ub_3 = tla.allocate(block_elems_3, tla.UInt16, tla.AddressSpace.ub, 256)
    cdf_ub_3 = tla.allocate(N_BINS + 16, tla.Int32, tla.AddressSpace.ub, 256)
    hist_ub_3 = tla.allocate(2 * n_chunks_3 + 64, tla.Int16, tla.AddressSpace.ub, 256)

    keys_flat_3 = tla.make_tensor(
        keys_ub_3, tla.make_layout(tla.make_shape(block_elems_3), tla.make_stride(1))
    )
    cdf_t_3 = tla.make_tensor(
        cdf_ub_3,
        tla.make_layout(tla.make_shape(N_BINS + 16), tla.make_stride(1)),
    )
    cdf_u16_t_3 = tla.make_tensor(
        tla.recast_ptr(cdf_ub_3, dtype=tla.UInt16),
        tla.make_layout(tla.make_shape(2 * N_BINS + 32), tla.make_stride(1)),
    )
    hist_t_3 = tla.make_tensor(
        hist_ub_3,
        tla.make_layout(tla.make_shape(2 * n_chunks_3 + 64), tla.make_stride(1)),
    )
    rpart_flat_3 = tla.make_tensor(
        rpart.ptr,
        tla.make_layout(tla.make_shape(rows * MAX_SHARE * 512), tla.make_stride(1)),
    )
    cpart_flat_3 = tla.make_tensor(
        cpart.ptr,
        tla.make_layout(tla.make_shape(rows * 2 * n_chunks_3), tla.make_stride(1)),
    )

    n_chunks_4 = max_k // CHUNK
    emit_chunks_4 = n_chunks_4 if n_chunks_4 < EMIT_CHUNKS else EMIT_CHUNKS
    block_elems_4 = emit_chunks_4 * CHUNK
    n_blocks_4 = n_chunks_4 // emit_chunks_4
    gt_slot_4 = 0
    eq_slot_4 = n_chunks_4
    off_slot_4 = 0
    quo_slot_4 = n_chunks_4
    cnt_slot_4 = 2 * n_chunks_4
    PAD_4 = 128

    count_ub_4 = tla.allocate(3 * n_chunks_4 + 32, tla.Int32, tla.AddressSpace.ub, 256)
    cbuf_ub_4 = tla.allocate(2 * n_chunks_4 + 32, tla.Int16, tla.AddressSpace.ub, 256)
    ekeys_ub_4 = tla.allocate(block_elems_4, tla.UInt16, tla.AddressSpace.ub, 256)
    escore_ub_4 = tla.allocate(block_elems_4, tla.BFloat16, tla.AddressSpace.ub, 256)
    tmp_idx_ub_4 = tla.allocate(block_elems_4, tla.Int16, tla.AddressSpace.ub, 256)
    tmp_val_ub_4 = tla.allocate(block_elems_4, tla.BFloat16, tla.AddressSpace.ub, 256)
    part_ub_4 = tla.allocate(2 * 258, tla.Int32, tla.AddressSpace.ub, 256)
    thr_ub_4 = tla.allocate(8, tla.Int32, tla.AddressSpace.ub, 256)
    idx_stage_ub_4 = tla.allocate(topk + PAD_4, tla.Int32, tla.AddressSpace.ub, 256)
    val_stage_ub_4 = tla.allocate(topk + PAD_4, tla.BFloat16, tla.AddressSpace.ub, 256)

    cbuf_t_4 = tla.make_tensor(
        cbuf_ub_4,
        tla.make_layout(tla.make_shape(2 * n_chunks_4 + 32), tla.make_stride(1)),
    )
    cnt_t_4 = tla.make_tensor(
        count_ub_4,
        tla.make_layout(tla.make_shape(3 * n_chunks_4 + 32), tla.make_stride(1)),
    )
    cnt_u16_t_4 = tla.make_tensor(
        tla.recast_ptr(count_ub_4, dtype=tla.UInt16),
        tla.make_layout(tla.make_shape(8 * n_chunks_4 + 64), tla.make_stride(1)),
    )
    partg_t_4 = tla.make_tensor(
        part_ub_4, tla.make_layout(tla.make_shape(258), tla.make_stride(1))
    )
    parte_t_4 = tla.make_tensor(
        tla.recast_ptr(part_ub_4 + 258, dtype=tla.Int32),
        tla.make_layout(tla.make_shape(258), tla.make_stride(1)),
    )
    thr_i32_t_4 = tla.make_tensor(
        thr_ub_4, tla.make_layout(tla.make_shape(8), tla.make_stride(1))
    )
    thr_u16_t_4 = tla.make_tensor(
        tla.recast_ptr(thr_ub_4, dtype=tla.UInt16),
        tla.make_layout(tla.make_shape(16), tla.make_stride(1)),
    )
    ekeys_flat_4 = tla.make_tensor(
        ekeys_ub_4, tla.make_layout(tla.make_shape(block_elems_4), tla.make_stride(1))
    )
    escore_flat_4 = tla.make_tensor(
        escore_ub_4, tla.make_layout(tla.make_shape(block_elems_4), tla.make_stride(1))
    )
    tmp_idx_t_4 = tla.make_tensor(
        tmp_idx_ub_4, tla.make_layout(tla.make_shape(block_elems_4), tla.make_stride(1))
    )
    tmp_val_t_4 = tla.make_tensor(
        tmp_val_ub_4, tla.make_layout(tla.make_shape(block_elems_4), tla.make_stride(1))
    )
    idx_stage_t_4 = tla.make_tensor(
        idx_stage_ub_4,
        tla.make_layout(tla.make_shape(topk + PAD_4), tla.make_stride(1)),
    )
    val_stage_t_4 = tla.make_tensor(
        val_stage_ub_4,
        tla.make_layout(tla.make_shape(topk + PAD_4), tla.make_stride(1)),
    )

    plan_threads_4 = n_chunks_4 if n_chunks_4 < SIMT_THREADS else SIMT_THREADS
    plan_per_4 = (n_chunks_4 + plan_threads_4 - 1) // plan_threads_4

    scal_ub_f = tla.allocate(16, tla.Int32, tla.AddressSpace.ub, 256)
    scal_t_f = tla.make_tensor(
        scal_ub_f, tla.make_layout(tla.make_shape(16), tla.make_stride(1))
    )

    with tla.vector():
        share_1 = tla.arch.block_num() // rows
        if share_1 < 1:
            share_1 = 1
        if share_1 > MAX_SHARE:
            share_1 = MAX_SHARE
        per_shard_1 = (n_blocks_1 + share_1 - 1) // share_1
        row_1 = tla.arch.block_idx() // share_1
        shard_1 = tla.arch.block_idx() % share_1
        if row_1 >= rows:
            row_1 = 0
        b_begin_1 = shard_1 * per_shard_1
        b_end_1 = b_begin_1 + per_shard_1
        if b_end_1 > n_blocks_1:
            b_end_1 = n_blocks_1
        c_begin_1 = b_begin_1 * block_chunks_1
        c_end_1 = b_end_1 * block_chunks_1
        if c_end_1 > n_chunks_1:
            c_end_1 = n_chunks_1
        vis_1 = visible[row_1]
        scores_row_1 = tla.make_tensor(
            scores.ptr + row_1 * max_k,
            tla.make_layout(tla.make_shape(max_k), tla.make_stride(1)),
        )
        keys_row_1 = tla.make_tensor(
            keys.ptr + row_1 * max_k,
            tla.make_layout(tla.make_shape(max_k), tla.make_stride(1)),
        )
        if tla.arch.block_idx() < rows * share_1:
            for blk_1 in tla.range(b_begin_1, b_end_1, 1):
                tla.copy(
                    tla.tile_view(
                        scores_flat_1, tla.make_shape(block_elems_1), tla.make_coord(0)
                    ),
                    tla.tile_view(
                        scores_row_1,
                        tla.make_shape(block_elems_1),
                        tla.make_coord(blk_1),
                    ),
                )
                tla.pipe_barrier(tla.pipes.ALL)
                with tla.vec.func(mode="simd"):
                    zero_u16_1 = tla.full(0, tla.Int16).bitcast(tla.UInt16)
                    all_one_u16_1 = tla.full(-1, tla.Int16).bitcast(tla.UInt16)
                    sign_u16_1 = tla.full(-32768, tla.Int16).bitcast(tla.UInt16)
                    for ch0_1 in tla.range(0, block_chunks_1, 1):
                        j0_1 = blk_1 * block_elems_1 + ch0_1 * CHUNK
                        bf0_1 = tla.tile_view(
                            scores_flat_1,
                            tla.make_shape(CHUNK),
                            tla.make_coord(ch0_1),
                        ).load()
                        bits0_1 = bf0_1.bitcast(tla.UInt16)
                        neg0_1 = tla.cmp(bits0_1, sign_u16_1, "ge")
                        flip0_1 = tla.where(neg0_1, all_one_u16_1, sign_u16_1)
                        key0_1 = tla.bitwise_xor(bits0_1, flip0_1)
                        width0_1 = vis_1 - j0_1
                        if width0_1 > CHUNK:
                            width0_1 = CHUNK
                        if width0_1 < 0:
                            width0_1 = 0
                        keep0_1, _ = tla.update_mask(width0_1, tla.Int16)
                        key0_1 = tla.where(keep0_1, key0_1, zero_u16_1)
                        tla.tile_view(
                            keys_flat_1,
                            tla.make_shape(CHUNK),
                            tla.make_coord(ch0_1),
                        ).store(key0_1, mask=keep0_1)
                tla.pipe_barrier(tla.pipes.ALL)
                with tla.vec.func(mode="simd"):
                    all_u8_1 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.UInt8)
                    low_half_1 = tla.create_mask(pattern=tla.mask.H, dtype=tla.UInt8)
                    acc_lo_1 = tla.params.HistogramsParams(
                        mode=tla.params.HistogramsBinType.BIN0,
                        type=tla.params.HistogramsType.ACCUMULATE,
                    )
                    acc_hi_1 = tla.params.HistogramsParams(
                        mode=tla.params.HistogramsBinType.BIN1,
                        type=tla.params.HistogramsType.ACCUMULATE,
                    )
                    hist_lo_1 = tla.full(0, tla.Int16)
                    hist_hi_1 = tla.full(0, tla.Int16)
                    for pair_1 in tla.range(0, block_chunks_1 // 2, 1):
                        k0_1 = tla.tile_view(
                            keys_flat_1,
                            tla.make_shape(CHUNK),
                            tla.make_coord(pair_1 * 2),
                        ).load()
                        k1_1 = tla.tile_view(
                            keys_flat_1,
                            tla.make_shape(CHUNK),
                            tla.make_coord(pair_1 * 2 + 1),
                        ).load()
                        hb0_1 = tla.pack(
                            tla.shift_right(k0_1.bitcast(tla.Int16), 8),
                            part=tla.RegPart.LOW,
                        )
                        hb1_1 = tla.pack(
                            tla.shift_right(k1_1.bitcast(tla.Int16), 8),
                            part=tla.RegPart.HIGH,
                        )
                        high_1 = tla.where(low_half_1, hb0_1, hb1_1)
                        hist_lo_1 = tla.histograms(
                            high_1, hist_lo_1, acc_lo_1, mask=all_u8_1
                        )
                        hist_hi_1 = tla.histograms(
                            high_1, hist_hi_1, acc_hi_1, mask=all_u8_1
                        )
                    tla.tile_view(
                        hist_t_1, tla.make_shape(128), tla.make_coord(blk_1 * 2)
                    ).store(hist_lo_1)
                    tla.tile_view(
                        hist_t_1, tla.make_shape(128), tla.make_coord(blk_1 * 2 + 1)
                    ).store(hist_hi_1)
                tla.pipe_barrier(tla.pipes.ALL)
                tla.copy(
                    tla.tile_view(
                        keys_row_1, tla.make_shape(block_elems_1), tla.make_coord(blk_1)
                    ),
                    tla.tile_view(
                        keys_flat_1, tla.make_shape(block_elems_1), tla.make_coord(0)
                    ),
                )
                tla.pipe_barrier(tla.pipes.ALL)
            with tla.vec.func(mode="simt", thread_block_dim=SIMT_THREADS):
                tid_1, _, _ = tla.arch.thread_idx()
                acc_1 = tla.Int32(0)
                for sb_1 in tla.range(b_begin_1, b_end_1, 1):
                    acc_1 = acc_1 + hist_t_1[sb_1 * N_BINS + tid_1].to(tla.Int32)
                shard_t_1[tid_1] = acc_1
            tla.pipe_barrier(tla.pipes.ALL)
            tla.copy(
                tla.make_tensor(
                    rpart.ptr + (row_1 * MAX_SHARE + shard_1) * 512,
                    tla.make_layout(tla.make_shape(N_BINS), tla.make_stride(1)),
                ),
                tla.tile_view(shard_t_1, tla.make_shape(N_BINS), tla.make_coord(0)),
            )
        tla.pipe_barrier(tla.pipes.ALL)
        # Stage hand-off: drain, device-wide barrier, drain again. The partials
        # were written with DMA above, so they are visible to every block here
        # (a scalar GM store would not be).
        tla_sync_all()
        tla.pipe_barrier(tla.pipes.ALL)

        share_2 = tla.arch.block_num() // rows
        if share_2 < 1:
            share_2 = 1
        if share_2 > MAX_SHARE:
            share_2 = MAX_SHARE
        per_shard_2 = (n_blocks_2 + share_2 - 1) // share_2
        row_2 = tla.arch.block_idx() // share_2
        shard_2 = tla.arch.block_idx() % share_2
        if row_2 >= rows:
            row_2 = 0
        b_begin_2 = shard_2 * per_shard_2
        b_end_2 = b_begin_2 + per_shard_2
        if b_end_2 > n_blocks_2:
            b_end_2 = n_blocks_2
        c_begin_2 = b_begin_2 * block_chunks_2
        c_end_2 = b_end_2 * block_chunks_2
        if c_end_2 > n_chunks_2:
            c_end_2 = n_chunks_2
        keys_row_2 = tla.make_tensor(
            keys.ptr + row_2 * max_k,
            tla.make_layout(tla.make_shape(max_k), tla.make_stride(1)),
        )
        scal_row_2 = tla.make_tensor(
            scal.ptr + row_2 * N_SCAL,
            tla.make_layout(tla.make_shape(N_SCAL), tla.make_stride(1)),
        )
        if tla.arch.block_idx() < rows * share_2:
            with tla.vec.func(mode="simt", thread_block_dim=SIMT_THREADS):
                tid_2, _, _ = tla.arch.thread_idx()
                acc_2 = tla.Int32(0)
                for sh_2 in tla.range(0, share_2, 1):
                    acc_2 = (
                        acc_2 + rpart_flat_2[(row_2 * MAX_SHARE + sh_2) * 512 + tid_2]
                    )
                cdf_t_2[tid_2] = acc_2
                tla.arch.sync_threads()
                if tid_2 == 0:
                    total_2 = cdf_t_2[N_BINS - 1]
                    rank_2 = total_2 - tla.Int32(topk) + 1
                    found_2 = tla.Int32(-1)
                    prev_2 = tla.Int32(0)
                    for bin_i_2 in tla.range(0, N_BINS, 1):
                        if found_2 < 0 and cdf_t_2[bin_i_2] >= rank_2:
                            found_2 = bin_i_2
                            if bin_i_2 > 0:
                                prev_2 = cdf_t_2[bin_i_2 - 1]
                    cdf_t_2[N_BINS + 1] = found_2
                    cdf_t_2[N_BINS + 2] = rank_2 - prev_2
                    scal_t_f[0] = total_2
                    scal_t_f[1] = rank_2
                    scal_t_f[2] = rank_2 - prev_2
                    scal_t_f[3] = found_2
            tla.pipe_barrier(tla.pipes.ALL)
            tla.copy(
                tla.tile_view(scal_row_2, tla.make_shape(N_SCAL), tla.make_coord(0)),
                tla.tile_view(scal_t_f, tla.make_shape(N_SCAL), tla.make_coord(0)),
            )
            tla.pipe_barrier(tla.pipes.ALL)
            for blk_2 in tla.range(b_begin_2, b_end_2, 1):
                tla.copy(
                    tla.tile_view(
                        keys_flat_2, tla.make_shape(block_elems_2), tla.make_coord(0)
                    ),
                    tla.tile_view(
                        keys_row_2, tla.make_shape(block_elems_2), tla.make_coord(blk_2)
                    ),
                )
                tla.pipe_barrier(tla.pipes.ALL)
                with tla.vec.func(mode="simd"):
                    low_half_2 = tla.create_mask(pattern=tla.mask.H, dtype=tla.UInt8)
                    acc_lo_2 = tla.params.HistogramsParams(
                        mode=tla.params.HistogramsBinType.BIN0,
                        type=tla.params.HistogramsType.ACCUMULATE,
                    )
                    acc_hi_2 = tla.params.HistogramsParams(
                        mode=tla.params.HistogramsBinType.BIN1,
                        type=tla.params.HistogramsType.ACCUMULATE,
                    )
                    h_scalar_2 = tla.tile_view(
                        cdf_u16_t_2,
                        tla.make_shape(1),
                        tla.make_coord((N_BINS + 1) * 2),
                    ).load(
                        params=tla.params.NormalLoadParams(
                            load_dist=tla.params.LoadDist.DIST_BRC_B16
                        )
                    )
                    hb_low_2 = tla.pack(h_scalar_2, part=tla.RegPart.LOW)
                    hb_high_2 = tla.pack(h_scalar_2, part=tla.RegPart.HIGH)
                    hb_2 = tla.where(low_half_2, hb_low_2, hb_high_2)
                    hist_lo_2 = tla.full(0, tla.Int16)
                    hist_hi_2 = tla.full(0, tla.Int16)
                    for pair_2 in tla.range(0, block_chunks_2 // 2, 1):
                        k0_2 = tla.tile_view(
                            keys_flat_2,
                            tla.make_shape(CHUNK),
                            tla.make_coord(pair_2 * 2),
                        ).load()
                        k1_2 = tla.tile_view(
                            keys_flat_2,
                            tla.make_shape(CHUNK),
                            tla.make_coord(pair_2 * 2 + 1),
                        ).load()
                        hb0_2 = tla.pack(
                            tla.shift_right(k0_2.bitcast(tla.Int16), 8),
                            part=tla.RegPart.LOW,
                        )
                        hb1_2 = tla.pack(
                            tla.shift_right(k1_2.bitcast(tla.Int16), 8),
                            part=tla.RegPart.HIGH,
                        )
                        high_2 = tla.where(low_half_2, hb0_2, hb1_2)
                        lb0_2 = tla.pack(k0_2.bitcast(tla.Int16), part=tla.RegPart.LOW)
                        lb1_2 = tla.pack(k1_2.bitcast(tla.Int16), part=tla.RegPart.HIGH)
                        low_2 = tla.where(low_half_2, lb0_2, lb1_2)
                        match_2 = tla.cmp(high_2, hb_2, "eq")
                        hist_lo_2 = tla.histograms(
                            low_2, hist_lo_2, acc_lo_2, mask=match_2
                        )
                        hist_hi_2 = tla.histograms(
                            low_2, hist_hi_2, acc_hi_2, mask=match_2
                        )
                    tla.tile_view(
                        hist_t_2, tla.make_shape(128), tla.make_coord(blk_2 * 2)
                    ).store(hist_lo_2)
                    tla.tile_view(
                        hist_t_2, tla.make_shape(128), tla.make_coord(blk_2 * 2 + 1)
                    ).store(hist_hi_2)
                tla.pipe_barrier(tla.pipes.ALL)
            with tla.vec.func(mode="simt", thread_block_dim=SIMT_THREADS):
                tid2_2, _, _ = tla.arch.thread_idx()
                acc2_2 = tla.Int32(0)
                for sb2_2 in tla.range(b_begin_2, b_end_2, 1):
                    acc2_2 = acc2_2 + hist_t_2[sb2_2 * N_BINS + tid2_2].to(tla.Int32)
                shard_t_2[tid2_2] = acc2_2
            tla.pipe_barrier(tla.pipes.ALL)
            tla.copy(
                tla.make_tensor(
                    rpart.ptr + (row_2 * MAX_SHARE + shard_2) * 512 + N_BINS,
                    tla.make_layout(tla.make_shape(N_BINS), tla.make_stride(1)),
                ),
                tla.tile_view(shard_t_2, tla.make_shape(N_BINS), tla.make_coord(0)),
            )
        tla.pipe_barrier(tla.pipes.ALL)
        # Second hand-off: the exact threshold and the low-byte CDF partials
        # are now visible to the counting stage on every block.
        tla_sync_all()
        tla.pipe_barrier(tla.pipes.ALL)

        share_3 = tla.arch.block_num() // rows
        if share_3 < 1:
            share_3 = 1
        if share_3 > MAX_SHARE:
            share_3 = MAX_SHARE
        per_shard_3 = (n_blocks_3 + share_3 - 1) // share_3
        row_3 = tla.arch.block_idx() // share_3
        shard_3 = tla.arch.block_idx() % share_3
        if row_3 >= rows:
            row_3 = 0
        b_begin_3 = shard_3 * per_shard_3
        b_end_3 = b_begin_3 + per_shard_3
        if b_end_3 > n_blocks_3:
            b_end_3 = n_blocks_3
        c_begin_3 = b_begin_3 * block_chunks_3
        c_end_3 = b_end_3 * block_chunks_3
        if c_end_3 > n_chunks_3:
            c_end_3 = n_chunks_3
        keys_row_3 = tla.make_tensor(
            keys.ptr + row_3 * max_k,
            tla.make_layout(tla.make_shape(max_k), tla.make_stride(1)),
        )
        scal_row_3 = tla.make_tensor(
            scal.ptr + row_3 * N_SCAL,
            tla.make_layout(tla.make_shape(N_SCAL), tla.make_stride(1)),
        )
        if tla.arch.block_idx() < rows * share_3:
            tla.copy(
                tla.tile_view(scal_t_f, tla.make_shape(N_SCAL), tla.make_coord(0)),
                tla.tile_view(scal_row_3, tla.make_shape(N_SCAL), tla.make_coord(0)),
            )
            tla.pipe_barrier(tla.pipes.ALL)
            with tla.vec.func(mode="simt", thread_block_dim=SIMT_THREADS):
                tid_3, _, _ = tla.arch.thread_idx()
                acc_3 = tla.Int32(0)
                for sh_3 in tla.range(0, share_3, 1):
                    acc_3 = (
                        acc_3
                        + rpart_flat_3[
                            (row_3 * MAX_SHARE + sh_3) * 512 + N_BINS + tid_3
                        ]
                    )
                cdf_t_3[tid_3] = acc_3
                tla.arch.sync_threads()
                if tid_3 == 0:
                    rank_3 = scal_t_f[2]
                    high_3 = scal_t_f[3]
                    found_3 = tla.Int32(-1)
                    for bin_i_3 in tla.range(0, N_BINS, 1):
                        if found_3 < 0 and cdf_t_3[bin_i_3] >= rank_3:
                            found_3 = bin_i_3
                    thr_3 = (high_3 << 8) + found_3
                    cdf_t_3[N_BINS + 4] = thr_3
                    scal_t_f[4] = found_3
                    scal_t_f[5] = thr_3
            tla.pipe_barrier(tla.pipes.ALL)
            tla.copy(
                tla.tile_view(scal_row_3, tla.make_shape(N_SCAL), tla.make_coord(0)),
                tla.tile_view(scal_t_f, tla.make_shape(N_SCAL), tla.make_coord(0)),
            )
            tla.pipe_barrier(tla.pipes.ALL)
            for blk_3 in tla.range(b_begin_3, b_end_3, 1):
                tla.copy(
                    tla.tile_view(
                        keys_flat_3, tla.make_shape(block_elems_3), tla.make_coord(0)
                    ),
                    tla.tile_view(
                        keys_row_3, tla.make_shape(block_elems_3), tla.make_coord(blk_3)
                    ),
                )
                tla.pipe_barrier(tla.pipes.ALL)
                with tla.vec.func(mode="simd"):
                    all_i16_3 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Int16)
                    first_i16_3 = tla.params.NormalStoreParams(
                        store_dist=tla.params.StoreDist.DIST_FIRST_ELEMENT_B16
                    )
                    one_3 = tla.full(1, tla.Int16)
                    zero_3 = tla.full(0, tla.Int16)
                    thr_reg_3 = tla.tile_view(
                        cdf_u16_t_3,
                        tla.make_shape(1),
                        tla.make_coord((N_BINS + 4) * 2),
                    ).load(
                        params=tla.params.NormalLoadParams(
                            load_dist=tla.params.LoadDist.DIST_BRC_B16
                        )
                    )
                    for ch_3 in tla.range(0, block_chunks_3, 1):
                        k_3 = tla.tile_view(
                            keys_flat_3,
                            tla.make_shape(CHUNK),
                            tla.make_coord(ch_3),
                        ).load()
                        gt_3 = tla.cmp(k_3, thr_reg_3, "gt")
                        eq_3 = tla.cmp(k_3, thr_reg_3, "eq")
                        gcnt_3 = tla.where(gt_3, one_3, zero_3).reduce(
                            tla.ReductionOp.ADD, mask=all_i16_3
                        )
                        ecnt_3 = tla.where(eq_3, one_3, zero_3).reduce(
                            tla.ReductionOp.ADD, mask=all_i16_3
                        )
                        slot_3 = blk_3 * block_chunks_3 + ch_3
                        tla.tile_view(
                            hist_t_3,
                            tla.make_shape(1),
                            tla.make_coord(gt_slot_3 + slot_3),
                        ).store(gcnt_3, first_i16_3)
                        tla.tile_view(
                            hist_t_3,
                            tla.make_shape(1),
                            tla.make_coord(eq_slot_3 + slot_3),
                        ).store(ecnt_3, first_i16_3)
                tla.pipe_barrier(tla.pipes.ALL)
            with tla.vec.func(mode="simt", thread_block_dim=SIMT_THREADS):
                tid2_3, _, _ = tla.arch.thread_idx()
                for ch2_3 in tla.range(c_begin_3 + tid2_3, c_end_3, SIMT_THREADS):
                    cpart_flat_3[row_3 * 2 * n_chunks_3 + gt_slot_3 + ch2_3] = hist_t_3[
                        gt_slot_3 + ch2_3
                    ]
                    cpart_flat_3[row_3 * 2 * n_chunks_3 + eq_slot_3 + ch2_3] = hist_t_3[
                        eq_slot_3 + ch2_3
                    ]
        tla.pipe_barrier(tla.pipes.ALL)
        # Third hand-off: per-chunk gt/eq counts and their shard partials are
        # complete, so the emit stage can compute the global prefix.
        tla_sync_all()
        tla.pipe_barrier(tla.pipes.ALL)

        if tla.arch.block_idx() < rows:
            row_4 = tla.arch.block_idx()
            vis_4 = visible[row_4]
            idx_offset_4 = offsets[row_4]
            scores_row_4 = tla.make_tensor(
                scores.ptr + row_4 * max_k,
                tla.make_layout(tla.make_shape(max_k), tla.make_stride(1)),
            )
            keys_row_4 = tla.make_tensor(
                keys.ptr + row_4 * max_k,
                tla.make_layout(tla.make_shape(max_k), tla.make_stride(1)),
            )
            scal_row_4 = tla.make_tensor(
                scal.ptr + row_4 * N_SCAL,
                tla.make_layout(tla.make_shape(N_SCAL), tla.make_stride(1)),
            )
            tla.copy(
                tla.tile_view(
                    cbuf_t_4, tla.make_shape(2 * n_chunks_4), tla.make_coord(0)
                ),
                tla.make_tensor(
                    cpart.ptr + row_4 * 2 * n_chunks_4,
                    tla.make_layout(tla.make_shape(2 * n_chunks_4), tla.make_stride(1)),
                ),
            )
            tla.pipe_barrier(tla.pipes.ALL)
            tla.copy(
                tla.tile_view(scal_t_f, tla.make_shape(N_SCAL), tla.make_coord(0)),
                tla.tile_view(scal_row_4, tla.make_shape(N_SCAL), tla.make_coord(0)),
            )
            tla.pipe_barrier(tla.pipes.ALL)
            with tla.vec.func(mode="simt", thread_block_dim=SIMT_THREADS):
                tid_4, _, _ = tla.arch.thread_idx()
                if tid_4 == 0:
                    thr_i32_t_4[0] = scal_t_f[5]
                if tid_4 < plan_threads_4:
                    gsum_4 = tla.Int32(0)
                    esum_4 = tla.Int32(0)
                    for k0_4 in tla.range_constexpr(0, plan_per_4, 1):
                        pc0_4 = tid_4 * plan_per_4 + k0_4
                        if pc0_4 < n_chunks_4:
                            gsum_4 = gsum_4 + cbuf_t_4[gt_slot_4 + pc0_4].to(tla.Int32)
                            esum_4 = esum_4 + cbuf_t_4[eq_slot_4 + pc0_4].to(tla.Int32)
                    partg_t_4[tid_4] = gsum_4
                    parte_t_4[tid_4] = esum_4
                tla.arch.sync_threads()
                if tid_4 == 0:
                    acc_g_4 = tla.Int32(0)
                    acc_e_4 = tla.Int32(0)
                    for k1_4 in tla.range_constexpr(0, plan_threads_4, 1):
                        cur_g_4 = partg_t_4[k1_4]
                        cur_e_4 = parte_t_4[k1_4]
                        partg_t_4[k1_4] = acc_g_4
                        parte_t_4[k1_4] = acc_e_4
                        acc_g_4 = acc_g_4 + cur_g_4
                        acc_e_4 = acc_e_4 + cur_e_4
                    partg_t_4[257] = acc_g_4
                tla.arch.sync_threads()
                if tid_4 < plan_threads_4:
                    pref_e_4 = parte_t_4[tid_4]
                    budget_4 = tla.Int32(topk) - partg_t_4[257]
                    base_e_4 = pref_e_4 if pref_e_4 < budget_4 else budget_4
                    pref_g_4 = partg_t_4[tid_4] + base_e_4
                    for k2_4 in tla.range_constexpr(0, plan_per_4, 1):
                        pc_4 = tid_4 * plan_per_4 + k2_4
                        if pc_4 < n_chunks_4:
                            gt_4 = cbuf_t_4[gt_slot_4 + pc_4].to(tla.Int32)
                            eqv_4 = cbuf_t_4[eq_slot_4 + pc_4].to(tla.Int32)
                            left_4 = budget_4 - pref_e_4
                            quota_4 = tla.Int32(0)
                            if left_4 > 0:
                                quota_4 = eqv_4 if eqv_4 < left_4 else left_4
                            cnt_t_4[off_slot_4 + pc_4] = pref_g_4
                            cnt_t_4[quo_slot_4 + pc_4] = quota_4
                            cnt_t_4[cnt_slot_4 + pc_4] = gt_4 + quota_4
                            pref_g_4 = pref_g_4 + gt_4 + quota_4
                            pref_e_4 = pref_e_4 + eqv_4
            tla.pipe_barrier(tla.pipes.ALL)
            with tla.vec.func(mode="simd"):
                pad_i32_4 = tla.full(-1, tla.Int32)
                pad_bf_4 = tla.full(-128, tla.Int16).bitcast(tla.BFloat16)
                for fp_4 in tla.range_constexpr(0, (topk + 63) // 64, 1):
                    tla.tile_view(
                        idx_stage_t_4, tla.make_shape(64), tla.make_coord(fp_4)
                    ).store(pad_i32_4)
                for fq_4 in tla.range_constexpr(0, (topk + 127) // 128, 1):
                    tla.tile_view(
                        val_stage_t_4, tla.make_shape(128), tla.make_coord(fq_4)
                    ).store(pad_bf_4)
            tla.pipe_barrier(tla.pipes.ALL)
            if vis_4 <= topk:
                with tla.vec.func(mode="simt", thread_block_dim=SIMT_THREADS):
                    tid1_4, _, _ = tla.arch.thread_idx()
                    for j1_4 in tla.range(tid1_4, vis_4, SIMT_THREADS):
                        idx_stage_t_4[j1_4] = j1_4 + idx_offset_4
                        val_stage_t_4[j1_4] = scores_row_4[j1_4]
                tla.pipe_barrier(tla.pipes.ALL)
            if vis_4 > topk:
                for blk_4 in tla.range(0, n_blocks_4, 1):
                    tla.copy(
                        tla.tile_view(
                            ekeys_flat_4,
                            tla.make_shape(block_elems_4),
                            tla.make_coord(0),
                        ),
                        tla.tile_view(
                            keys_row_4,
                            tla.make_shape(block_elems_4),
                            tla.make_coord(blk_4),
                        ),
                    )
                    tla.copy(
                        tla.tile_view(
                            escore_flat_4,
                            tla.make_shape(block_elems_4),
                            tla.make_coord(0),
                        ),
                        tla.tile_view(
                            scores_row_4,
                            tla.make_shape(block_elems_4),
                            tla.make_coord(blk_4),
                        ),
                    )
                    tla.pipe_barrier(tla.pipes.ALL)
                    with tla.vec.func(mode="simd"):
                        thr16_4 = tla.tile_view(
                            thr_u16_t_4, tla.make_shape(1), tla.make_coord(0)
                        ).load(
                            params=tla.params.NormalLoadParams(
                                load_dist=tla.params.LoadDist.DIST_BRC_B16
                            )
                        )
                        lane16_4 = tla.arange(dtype=tla.Int16)
                        lane_f16_4 = lane16_4.bitcast(tla.Float16)
                        fill255_4 = tla.full(255, tla.Int16)
                        low_u8_4 = tla.create_mask(pattern=tla.mask.H, dtype=tla.UInt8)
                        all_u8_4 = tla.create_mask(
                            pattern=tla.mask.ALL, dtype=tla.UInt8
                        )
                        acc_bin0_4 = tla.params.HistogramsParams(
                            mode=tla.params.HistogramsBinType.BIN0,
                            type=tla.params.HistogramsType.ACCUMULATE,
                        )
                        for ch_4 in tla.range_constexpr(0, emit_chunks_4, 1):
                            gch_4 = blk_4 * emit_chunks_4 + ch_4
                            quota16_4 = tla.tile_view(
                                cnt_u16_t_4,
                                tla.make_shape(1),
                                tla.make_coord((quo_slot_4 + gch_4) * 2),
                            ).load(
                                params=tla.params.NormalLoadParams(
                                    load_dist=tla.params.LoadDist.DIST_BRC_B16
                                )
                            )
                            kreg_4 = tla.tile_view(
                                ekeys_flat_4,
                                tla.make_shape(CHUNK),
                                tla.make_coord(ch_4),
                            ).load()
                            sreg_4 = tla.tile_view(
                                escore_flat_4,
                                tla.make_shape(CHUNK),
                                tla.make_coord(ch_4),
                            ).load()
                            gt16_4 = tla.cmp(kreg_4, thr16_4, "gt")
                            eq16_4 = tla.cmp(kreg_4, thr16_4, "eq")
                            p8_4 = tla.where(
                                low_u8_4,
                                tla.pack(
                                    tla.where(eq16_4, lane16_4, fill255_4),
                                    part=tla.RegPart.LOW,
                                ),
                                tla.pack(fill255_4, part=tla.RegPart.HIGH),
                            )
                            rank16_4 = tla.gather(
                                tla.histograms(
                                    p8_4,
                                    tla.full(0, tla.Int16),
                                    acc_bin0_4,
                                    mask=all_u8_4,
                                ),
                                lane16_4.bitcast(tla.UInt16),
                            ).bitcast(tla.UInt16)
                            take16_4 = tla.bitwise_or(
                                gt16_4,
                                tla.bitwise_and(
                                    eq16_4, tla.cmp(rank16_4, quota16_4, "le")
                                ),
                            )
                            tla.tile_view(
                                tmp_val_t_4, tla.make_shape(128), tla.make_coord(ch_4)
                            ).store(
                                tla.squeeze(
                                    sreg_4.bitcast(tla.Float16), take16_4
                                ).bitcast(tla.BFloat16)
                            )
                            tla.tile_view(
                                tmp_idx_t_4, tla.make_shape(128), tla.make_coord(ch_4)
                            ).store(
                                tla.squeeze(lane_f16_4, take16_4).bitcast(tla.Int16)
                            )

                    tla.pipe_barrier(tla.pipes.ALL)
                    with tla.vec.func(mode="simt", thread_block_dim=SIMT_THREADS):
                        tc_4, _, _ = tla.arch.thread_idx()
                        if tc_4 < emit_chunks_4:
                            gch2_4 = blk_4 * emit_chunks_4 + tc_4
                            ccnt_4 = cnt_t_4[cnt_slot_4 + gch2_4]
                            cpos_4 = cnt_t_4[off_slot_4 + gch2_4]
                            gbase2_4 = idx_offset_4 + gch2_4 * CHUNK
                            for i0_4 in tla.range(0, ccnt_4, 1):
                                idx_stage_t_4[cpos_4 + i0_4] = gbase2_4 + tmp_idx_t_4[
                                    tc_4 * CHUNK + i0_4
                                ].to(tla.Int32)
                                val_stage_t_4[cpos_4 + i0_4] = tmp_val_t_4[
                                    tc_4 * CHUNK + i0_4
                                ]
                    tla.pipe_barrier(tla.pipes.ALL)
            tla.pipe_barrier(tla.pipes.ALL)
            out_idx_row_4 = tla.make_tensor(
                out_idx.ptr + row_4 * topk,
                tla.make_layout(tla.make_shape(topk), tla.make_stride(1)),
            )
            out_val_row_4 = tla.make_tensor(
                out_val.ptr + row_4 * topk,
                tla.make_layout(tla.make_shape(topk), tla.make_stride(1)),
            )
            idx_full_4 = (topk // 8) * 8
            val_full_4 = (topk // 16) * 16
            if idx_full_4 > 0:
                tla.copy(
                    tla.tile_view(
                        out_idx_row_4, tla.make_shape(idx_full_4), tla.make_coord(0)
                    ),
                    tla.tile_view(
                        idx_stage_t_4, tla.make_shape(idx_full_4), tla.make_coord(0)
                    ),
                )
            if val_full_4 > 0:
                tla.copy(
                    tla.tile_view(
                        out_val_row_4, tla.make_shape(val_full_4), tla.make_coord(0)
                    ),
                    tla.tile_view(
                        val_stage_t_4, tla.make_shape(val_full_4), tla.make_coord(0)
                    ),
                )
            if idx_full_4 < topk or val_full_4 < topk:
                with tla.vec.func(mode="simt", thread_block_dim=SIMT_THREADS):
                    tid2_4, _, _ = tla.arch.thread_idx()
                    for j2_4 in tla.range(tid2_4, topk, SIMT_THREADS):
                        if j2_4 >= idx_full_4:
                            out_idx_row_4[j2_4] = idx_stage_t_4[j2_4]
                        if j2_4 >= val_full_4:
                            out_val_row_4[j2_4] = val_stage_t_4[j2_4]
