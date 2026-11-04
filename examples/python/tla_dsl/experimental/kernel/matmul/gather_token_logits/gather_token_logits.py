# -----------------------------------------------------------------------------------------------------------
# CATLASS Python DSL kernels for gather_token_logits.
#
# One kernel:
#   gather : GEMM + gather of the selected logits (z never lands in GM)
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


@tla.kernel
def gtl_gather_kernel(
    hidden: tla.Tensor,  # [M, H] bf16 RowMajor
    weight: tla.Tensor,  # [H, V] bf16 RowMajor (host pre-transposes [V, H])
    lid: tla.Tensor,  # [M, K] int32 local vocab index, -1 when not owned
    payload: tla.Tensor,  # [M, K] fp32, zeroed by host
    inv_temperature: float,
    tiles_per_seg: int,
    m_real: int,
    v_real: int,
    K: tla.Constexpr[int],
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
    lid_ready = tla.flag("lid_ready", tla.arch.MTE2, tla.arch.VECTOR)

    l1a0_ptr = tla.allocate(BM * BH, dtype_in, tla.AddressSpace.l1, 512)
    l1b0_ptr = tla.allocate(BH * BN, dtype_in, tla.AddressSpace.l1, 512)
    l1a1_ptr = tla.allocate(BM * BH, dtype_in, tla.AddressSpace.l1, 512)
    l1b1_ptr = tla.allocate(BH * BN, dtype_in, tla.AddressSpace.l1, 512)
    l0a0_ptr = tla.allocate(BM * SK, dtype_in, tla.AddressSpace.l0a, 512)
    l0b0_ptr = tla.allocate(SK * BN, dtype_in, tla.AddressSpace.l0b, 512)
    l0a1_ptr = tla.allocate(BM * SK, dtype_in, tla.AddressSpace.l0a, 512)
    l0b1_ptr = tla.allocate(SK * BN, dtype_in, tla.AddressSpace.l0b, 512)
    l0c_ptr = tla.allocate(BM * BN, tla.Float32, tla.AddressSpace.l0c, 512)

    # Per-AIV UB: one SPLIT_M half of the L0C tile plus the gather working set.
    ubz_ptr = tla.allocate(ROWS_PER_SUB * BN, tla.Float32, tla.AddressSpace.ub, 256)
    ublid_ptr = tla.allocate(ROWS_PER_SUB * K, tla.Int32, tla.AddressSpace.ub, 256)

    # ---------------------------------------------------------------- cube: GEMM
    # Complexity per tile: O(BM * BN * H) MAC; the gather stays on the AIV side.
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
        ub_z_flat = tla.make_tensor(
            ubz_ptr,
            tla.make_layout(tla.make_shape(ROWS_PER_SUB * BN), tla.make_stride(1)),
        )
        ub_lid = tla.make_tensor(
            ublid_ptr,
            tla.make_layout(tla.make_shape(ROWS_PER_SUB, K), tla.make_stride(K, 1)),
        )
        ub_lid_flat = tla.make_tensor(
            ublid_ptr,
            tla.make_layout(tla.make_shape(ROWS_PER_SUB * K), tla.make_stride(1)),
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
                # Stage this block's local ids once: O(BM * K) per block.
                gm_lid = tla.tile_view(
                    lid,
                    tla.make_shape(ROWS_PER_SUB, K),
                    tla.make_coord(mb * 2 + sub, c0),
                )
                tla.copy(ub_lid, gm_lid)
                tla.set_flag(lid_ready)
                tla.wait_flag(lid_ready)

                for vt in tla.range(vt_lo, vt_hi):
                    v0 = vt * BN
                    ncols = tla.as_numeric(v_real - v0)
                    if ncols > BN:
                        ncols = tla.as_numeric(BN)
                    tla.cross_core_wait_flag(exchange, tla.arch.VECTOR)

                    # Gather: one thread per row, K scalar range checks per V tile.
                    # Complexity: O(BM * K) per tile, O(M * K) gathered reads overall.
                    with tla.vec.func(mode="simt", thread_block_dim=ROWS_PER_SUB):
                        tid, _, _ = tla.arch.thread_idx()
                        grow = sub_row0 + tid
                        if grow < m_real:
                            vhi = v0 + ncols
                            for k in tla.range(c0, K, c1):
                                loc = ub_lid_flat[tid * K + k]
                                # Non-owned ids keep the -1 sentinel, so the lower
                                # bound alone excludes them; the host zero-fills
                                # payload, so there is nothing to write for them.
                                if loc >= v0:
                                    if loc < vhi:
                                        payload[grow, k] = (
                                            ub_z_flat[tid * BN + (loc - v0)]
                                            * inv_temperature
                                        )

                    tla.pipe_barrier(tla.pipes.ALL)
                    tla.cross_core_set_flag(exchange, tla.arch.VECTOR)


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

    # Static layout: the SIMT gather regions require statically shaped buffers,
    # so one artifact is compiled per (M, H, V, K, NV) tuple.
    return from_dlpack(buffer, layout_tag=layout_tag)


def run(hidden, weight, topk_ids, temperature, rank):
    """Return ``logits`` ``[M, K]``; see README.md for semantics."""
    import torch

    num_tokens, hidden_size = hidden.shape
    local_vocab = weight.shape[0]
    topk = topk_ids.shape[1]
    device = hidden.device

    # Non-owned (i, k) stay 0, so the gather kernel has nothing to write for them.
    logits = torch.zeros((num_tokens, topk), dtype=torch.float32, device=device)
    if num_tokens == 0:
        return logits

    bm, bn = BM, BN

    vocab_start = int(rank) * local_vocab
    vocab_end = vocab_start + local_vocab
    inv_temperature = 1.0 / float(temperature)

    owned = (topk_ids >= vocab_start) & (topk_ids < vocab_end)
    local_ids = torch.where(
        owned, topk_ids - vocab_start, torch.full_like(topk_ids, -1)
    )
    lid = local_ids.to(torch.int32).contiguous()

    grid_m = (num_tokens + bm - 1) // bm
    nv_tiles = (local_vocab + bn - 1) // bn
    device_index = (
        int(str(hidden.device).split(":")[-1]) if ":" in str(hidden.device) else 0
    )
    cube_cores, _ = _core_counts(device_index)
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
    if m_pad != num_tokens:
        lid_use = torch.full((m_pad, topk), -1, dtype=torch.int32, device=device)
        lid_use[:num_tokens] = lid
    else:
        lid_use = lid

    h_t = _tla_tensor(hidden_use, tla.arch.RowMajor)
    # The DSL copy has no ColumnMajor-GM -> zN-L1 route, so the [V, H] shard is
    # transposed to a [H, V] row-major operand once per call (see README.md).  A
    # partial (V % BN != 0) weight tile also has no working GM->L1 route, so the
    # shard is zero-padded up to a BN multiple; the padded columns are excluded
    # from the gather by v_real below.
    weight_hv = _weight_hv(weight, v_pad_target(weight, bn), h_pad)
    w_t = _tla_tensor(weight_hv, tla.arch.RowMajor)
    lid_t = _tla_tensor(lid_use, tla.arch.RowMajor)
    payload_t = _tla_tensor(logits, tla.arch.RowMajor)

    key = (num_tokens, hidden_size, local_vocab, topk, nv)
    entry = _COMPILED.get(key)
    if entry is None:
        gather_artifact = tla.compile(
            gtl_gather_kernel,
            h_t,
            w_t,
            lid_t,
            payload_t,
            inv_temperature,
            tiles_per_seg,
            num_tokens,
            local_vocab,
            topk,
            nv,
            options="--npu-arch 3510",
        )
        entry = (gather_artifact, cube_cores)
        _COMPILED[key] = entry
    gather_artifact, cube_cores = entry

    gather_artifact(
        h_t,
        w_t,
        lid_t,
        payload_t,
        inv_temperature,
        tiles_per_seg,
        num_tokens,
        local_vocab,
        block_num=cube_cores,
    )
    return logits
