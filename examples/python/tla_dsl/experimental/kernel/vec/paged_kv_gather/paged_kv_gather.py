# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Paged KV gather: gather K/V rows from a block table into contiguous decode output.

    for each (batch b, position t < seq_len[b]):
        blk = block_table[b, t // block_size]
        off = t % block_size
        out_k[b, t, :] = cache_k[blk, off, :]
        out_v[b, t, :] = cache_v[blk, off, :]

A block-table entry maps ``BS`` consecutive positions to ``BS`` consecutive cache
rows.  The kernel therefore works on *tiles* of ``TR`` positions with
``TR | BS`` and ``TR | S``: inside a tile the source rows are contiguous in the
cache and the destination rows are contiguous in the output, so the whole tile
(K and V) travels with one DMA.  ``TR`` is picked on the host as the largest
common divisor of ``BS`` and ``S`` that still leaves room for several in-flight
tile buffers in UB, which turns the transfer into a multi-KiB copy and amortises
the block-table scalar load over ``TR`` rows.

Only the last tile of a sequence can be partially valid (``TR | S`` makes the
tail a length issue, not an address issue).  That tile is copied row by row with
a per-row ``t < seq_len[b]`` check.  Positions past ``seq_len[b]``, tiles behind
a negative table entry and tiles fully past the sequence are written as zeros
from a zeroed UB tile, so the host may hand over an ``empty`` output and the
kernel produces exactly the documented zero fill.  The GM side stays dense
(``stride = HD``), so ``HD`` does not have to be 32Byte-aligned.

High-performance design points
------------------------------
1. **Tile DMA instead of row DMA.** A tile is contiguous on both sides, so one
   MTE2 and one MTE3 instruction move the whole ``TR x HD`` tile. This divides
   the DMA issue count and the block-table scalar load by ``TR``.
2. **Multi-buffered pipeline.** ``BUFS`` independent K/V tile pairs, each
   guarded by its own mutex, let the load of tile ``i + 1`` overlap the store of
   tile ``i``; a slot is refilled only after its previous store drains.
3. **Padding only inside UB.** UB rows are padded to ``N_PAD`` (a multiple of
   64 lanes) so a single row of a tile starts 32Byte-aligned even when ``HD * 4``
   is not; the GM side stays dense, so any ``HD`` works.
4. **Tail tile fallback.** Only the last tile of a sequence can be partially
   valid; it is copied row by row after checking ``t0 + r < seq_lens[b]``.
5. **Kernel-side zero fill.** Tiles behind a negative table entry, tiles past
   the sequence and tail rows past ``seq_lens[b]`` are drained from a pre-zeroed
   UB tile, so the host allocates an ``empty`` output.
6. **Intra-core synchronization is all mutex.** Per-slot mutexes carry the
   MTE2 -> MTE3 hand-off, and the zero tile's VECTOR -> MTE3 hand-off uses
   lock/unlock on one mutex instead of a flag pair.
7. **Per-core tile split.** Each block owns one contiguous range of tile ids and
   unrolls ``BUFS`` slots; empty blocks skip through the ``task < total`` guard.
"""

import catlass.tla as tla

VL_ELE = 64  # fp32 vector length (256 B / 4 B)
UB_BYTES = 262144  # per-AIV UB capacity
MAX_BUFS = 4  # staging depth: one K/V tile pair per buffer


@tla.kernel
def paged_kv_gather_kernel(
    gm_ck: tla.Tensor,  # [NB * BS, HD] fp32
    gm_cv: tla.Tensor,  # [NB * BS, HD] fp32
    gm_bt: tla.Tensor,  # [B, MB] int32 block table, negative disables
    gm_lens: tla.Tensor,  # [B] int32 valid lengths
    gm_ok: tla.Tensor,  # [B * S, HD] fp32
    gm_ov: tla.Tensor,  # [B * S, HD] fp32
    BS: tla.Constexpr[int],
    S: tla.Constexpr[int],
    HD: tla.Constexpr[int],
    N_PAD: tla.Constexpr[int],
    TR: tla.Constexpr[int],
    BUFS: tla.Constexpr[int],
) -> None:
    bsz = gm_lens.origin_shape[0]
    c0 = 0
    tiles_b = (S + TR - 1) // TR

    # One K tile and one V tile per buffer: the two gathers of a tile are issued
    # back to back, so the pair travels through the pipeline as a unit and both
    # outputs come out of the same UB residency.
    k_ptr = [None] * BUFS
    v_ptr = [None] * BUFS
    mtx = [None] * BUFS
    for bi in tla.range_constexpr(BUFS):
        k_ptr[bi] = tla.allocate(TR * N_PAD, tla.Float32, tla.AddressSpace.ub, 256)
        v_ptr[bi] = tla.allocate(TR * N_PAD, tla.Float32, tla.AddressSpace.ub, 256)
        mtx[bi] = tla.mutex(resource="pkg_tile", id=bi)

    # UB rows are padded to ``N_PAD`` so that a row start inside a tile is
    # always 32-byte aligned even when ``HD`` is not (row-wise tail copies index
    # single rows of this buffer); the GM side stays dense.
    tile_ub = tla.make_layout(
        tla.make_shape(TR, HD),
        tla.make_stride(N_PAD, 1),
        origin_shape=tla.make_shape(TR, HD),
    )
    tile_gm = tla.make_layout(
        tla.make_shape(TR, HD),
        tla.make_stride(HD, 1),
        origin_shape=tla.make_shape(TR, HD),
    )
    row_gm = tla.make_layout(
        tla.make_shape(1, HD),
        tla.make_stride(HD, 1),
        origin_shape=tla.make_shape(1, HD),
    )
    ub_k = [None] * BUFS
    ub_v = [None] * BUFS
    for bj in tla.range_constexpr(BUFS):
        ub_k[bj] = tla.make_tensor(k_ptr[bj], tile_ub)
        ub_v[bj] = tla.make_tensor(v_ptr[bj], tile_ub)

    # Zero fill source: one tile's worth of zeros, written once by the vector
    # pipe.  Rows that must not be gathered are drained from here, which lets the
    # host allocate an uninitialised output.
    zero_ptr = tla.allocate(TR * N_PAD, tla.Float32, tla.AddressSpace.ub, 256)
    zero_pad = tla.make_tensor(
        zero_ptr,
        tla.make_layout(
            tla.make_shape(TR, N_PAD),
            tla.make_stride(N_PAD, 1),
            origin_shape=tla.make_shape(TR, N_PAD),
        ),
    )
    zero_ub = tla.make_tensor(
        zero_ptr,
        tla.make_layout(
            tla.make_shape(TR, HD),
            tla.make_stride(N_PAD, 1),
            origin_shape=tla.make_shape(TR, HD),
        ),
    )
    # Vector -> MTE3 hand-off for the zero tile: the vector fill acquires the
    # mutex first and releases it after the last store, so the first MTE3 read of
    # the zero tile waits for the whole fill (same ordering a flag pair gave,
    # without an extra flag resource).
    zero_mtx = tla.mutex(resource="pkg_zero", id=0)

    gid = tla.arch.block_idx()
    aiv_num = tla.arch.block_num()
    total = bsz * tiles_b
    per = (total + aiv_num - 1) // aiv_num
    begin = gid * per

    with tla.vector():
        zero_mtx.lock(pipe=tla.arch.VECTOR)
        with tla.vec.func(mode="simd"):
            for r in tla.range_constexpr(TR):
                for c in tla.range_constexpr(N_PAD // VL_ELE):
                    tla.tile_view(
                        zero_pad,
                        tla.make_shape(1, VL_ELE),
                        tla.make_coord(r, c),
                    ).store(tla.full(0.0, tla.Float32))
        zero_mtx.unlock(pipe=tla.arch.VECTOR)
        zero_mtx.lock(pipe=tla.arch.MTE3)
        zero_mtx.unlock(pipe=tla.arch.MTE3)

        for grp in tla.range((per + BUFS - 1) // BUFS):
            for bi in tla.range_constexpr(BUFS):
                # Static slot index ``bi`` keeps the buffer/mutex lists Python
                # indexed while the tile id stays dynamic: slot ``bi`` is only
                # touched again ``BUFS`` tiles later, which is what gives the
                # load/store overlap.
                task = begin + grp * BUFS + bi
                if task < total:
                    b = task // tiles_b
                    t0 = (task - b * tiles_b) * TR
                    dst = (
                        b.to(tla.Int64) * tla.Int64(S) + t0.to(tla.Int64)
                    ) * tla.Int64(HD)
                    blk = gm_bt[b, t0 // BS]
                    if blk < c0:
                        # Disabled entries read as zeros.
                        tla.copy(tla.make_tensor(gm_ok.ptr + dst, tile_gm), zero_ub)
                        tla.copy(tla.make_tensor(gm_ov.ptr + dst, tile_gm), zero_ub)
                    else:
                        if t0 < gm_lens[b]:
                            off0 = t0 - (t0 // BS) * BS
                            if t0 + TR <= gm_lens[b]:
                                # Full tile: one DMA per operand, source and
                                # destination rows are both contiguous.
                                row = blk.to(tla.Int64) * tla.Int64(BS) + off0.to(
                                    tla.Int64
                                )
                                src = row * tla.Int64(HD)
                                mtx[bi].lock(pipe=tla.arch.MTE2)
                                tla.copy(
                                    ub_k[bi], tla.make_tensor(gm_ck.ptr + src, tile_gm)
                                )
                                tla.copy(
                                    ub_v[bi], tla.make_tensor(gm_cv.ptr + src, tile_gm)
                                )
                                mtx[bi].unlock(pipe=tla.arch.MTE2)
                                mtx[bi].lock(pipe=tla.arch.MTE3)
                                tla.copy(
                                    tla.make_tensor(gm_ok.ptr + dst, tile_gm), ub_k[bi]
                                )
                                tla.copy(
                                    tla.make_tensor(gm_ov.ptr + dst, tile_gm), ub_v[bi]
                                )
                                mtx[bi].unlock(pipe=tla.arch.MTE3)
                            else:
                                # Tail tile: the rows with ``t0 + r < seq_len[b]``
                                # are gathered, the remaining lines are zeroed.
                                mtx[bi].lock(pipe=tla.arch.MTE2)
                                for r in tla.range_constexpr(TR):
                                    if t0 + r < gm_lens[b]:
                                        row = blk.to(tla.Int64) * tla.Int64(BS) + (
                                            off0 + r
                                        ).to(tla.Int64)
                                        src = row * tla.Int64(HD)
                                        ub_rk = tla.tile_view(
                                            ub_k[bi],
                                            tla.make_shape(1, HD),
                                            tla.make_coord(r, 0),
                                        )
                                        ub_rv = tla.tile_view(
                                            ub_v[bi],
                                            tla.make_shape(1, HD),
                                            tla.make_coord(r, 0),
                                        )
                                        tla.copy(
                                            ub_rk,
                                            tla.make_tensor(gm_ck.ptr + src, row_gm),
                                        )
                                        tla.copy(
                                            ub_rv,
                                            tla.make_tensor(gm_cv.ptr + src, row_gm),
                                        )
                                mtx[bi].unlock(pipe=tla.arch.MTE2)
                                mtx[bi].lock(pipe=tla.arch.MTE3)
                                for r in tla.range_constexpr(TR):
                                    row_dst = dst + tla.Int64(r * HD)
                                    if t0 + r < gm_lens[b]:
                                        ub_rk = tla.tile_view(
                                            ub_k[bi],
                                            tla.make_shape(1, HD),
                                            tla.make_coord(r, 0),
                                        )
                                        ub_rv = tla.tile_view(
                                            ub_v[bi],
                                            tla.make_shape(1, HD),
                                            tla.make_coord(r, 0),
                                        )
                                        tla.copy(
                                            tla.make_tensor(
                                                gm_ok.ptr + row_dst, row_gm
                                            ),
                                            ub_rk,
                                        )
                                        tla.copy(
                                            tla.make_tensor(
                                                gm_ov.ptr + row_dst, row_gm
                                            ),
                                            ub_rv,
                                        )
                                    else:
                                        ub_zk = tla.tile_view(
                                            zero_ub,
                                            tla.make_shape(1, HD),
                                            tla.make_coord(r, 0),
                                        )
                                        tla.copy(
                                            tla.make_tensor(
                                                gm_ok.ptr + row_dst, row_gm
                                            ),
                                            ub_zk,
                                        )
                                        tla.copy(
                                            tla.make_tensor(
                                                gm_ov.ptr + row_dst, row_gm
                                            ),
                                            ub_zk,
                                        )
                                mtx[bi].unlock(pipe=tla.arch.MTE3)
                        else:
                            # Whole tile past the sequence: zeros only.
                            tla.copy(tla.make_tensor(gm_ok.ptr + dst, tile_gm), zero_ub)
                            tla.copy(tla.make_tensor(gm_ov.ptr + dst, tile_gm), zero_ub)


# ======================================================================================
# HOST side: shape checks, compile and launch.
# ======================================================================================

_COMPILED = {}


def _vector_cores(index):
    import torch

    return int(torch.npu.get_device_properties(index).vector_core_num)


def _pick_tile(bs, max_seq, hd):
    """Choose ``(N_PAD, TR, BUFS)``: padded row, tile rows and in-flight pairs.

    ``TR`` must divide both ``BS`` and ``S`` so that a tile never straddles a
    block-table entry nor the end of the output.  The zero-fill tile and at least
    two K/V tile pairs have to fit in UB.
    """
    budget = UB_BYTES - 16384
    n_pad = (hd + VL_ELE - 1) // VL_ELE * VL_ELE
    # zero tile + one K/V pair, all padded to ``n_pad`` per row
    if 3 * n_pad * 4 > budget:
        raise ValueError(
            "one K/V row plus the zero tile does not fit in UB; HD is too large"
        )
    tr_max = max(1, budget // (3 * n_pad * 4))
    tr = 1
    for d in range(1, min(bs, max_seq, tr_max) + 1):
        if bs % d == 0 and max_seq % d == 0:
            tr = d
    room = budget - tr * n_pad * 4
    bufs = max(1, min(MAX_BUFS, room // (2 * tr * n_pad * 4)))
    return n_pad, tr, bufs


def run(cache_k, cache_v, block_table, seq_lens, max_seq):
    """Gather ``[B, max_seq, HD]`` K/V from a paged cache.

    ``cache_k`` / ``cache_v`` are ``[NB, BS, HD]``; ``block_table`` is
    ``[B, MB]`` int32 and ``seq_lens`` is ``[B]`` int32.
    """
    import torch
    from catlass.tla.runtime import from_dlpack

    if cache_k.dim() != 3 or cache_k.shape != cache_v.shape:
        raise ValueError("cache_k / cache_v must be 3-D [NB, BS, HD] with equal shapes")
    nb, bs, hd = cache_k.shape
    bsz, mb = block_table.shape
    if tuple(seq_lens.shape) != (bsz,):
        raise ValueError("seq_lens must be [B]")
    if max_seq <= 0 or max_seq > mb * bs:
        raise ValueError("max_seq must be in (0, MB * BS]")

    device = cache_k.device
    out_k = torch.empty((bsz * max_seq, hd), dtype=torch.float32, device=device)
    out_v = torch.empty((bsz * max_seq, hd), dtype=torch.float32, device=device)
    if bsz == 0 or max_seq == 0:
        return out_k.view(bsz, max_seq, hd), out_v.view(bsz, max_seq, hd)

    device_index = int(str(device).split(":")[-1]) if ":" in str(device) else 0

    ck = cache_k.reshape(-1, hd)
    cv = cache_v.reshape(-1, hd)
    bt_i32 = (
        block_table if block_table.dtype == torch.int32 else block_table.to(torch.int32)
    )
    ln_i32 = seq_lens if seq_lens.dtype == torch.int32 else seq_lens.to(torch.int32)

    key = (nb * bs, hd, bsz, mb, max_seq)
    plan = _COMPILED.get(key)
    if plan is None:
        vector_cores = _vector_cores(device_index)
        n_pad, tr, bufs = _pick_tile(bs, max_seq, hd)
        ck_t = from_dlpack(ck.contiguous(), layout_tag=tla.arch.RowMajor)
        cv_t = from_dlpack(cv.contiguous(), layout_tag=tla.arch.RowMajor)
        bt_t = from_dlpack(bt_i32.contiguous(), layout_tag=tla.arch.RowMajor)
        ln_t = from_dlpack(ln_i32.contiguous(), layout_tag=tla.arch.RowMajor)
        ok_t = from_dlpack(out_k, layout_tag=tla.arch.RowMajor)
        ov_t = from_dlpack(out_v, layout_tag=tla.arch.RowMajor)
        artifact = tla.compile(
            paged_kv_gather_kernel,
            ck_t,
            cv_t,
            bt_t,
            ln_t,
            ok_t,
            ov_t,
            bs,
            max_seq,
            hd,
            n_pad,
            tr,
            bufs,
            options="--npu-arch 3510",
        )
        plan = {
            "artifact": artifact,
            "cores": vector_cores,
            "key": key,
        }
        _COMPILED[key] = plan

    plan["artifact"](
        from_dlpack(ck.contiguous(), layout_tag=tla.arch.RowMajor),
        from_dlpack(cv.contiguous(), layout_tag=tla.arch.RowMajor),
        from_dlpack(bt_i32.contiguous(), layout_tag=tla.arch.RowMajor),
        from_dlpack(ln_i32.contiguous(), layout_tag=tla.arch.RowMajor),
        from_dlpack(out_k, layout_tag=tla.arch.RowMajor),
        from_dlpack(out_v, layout_tag=tla.arch.RowMajor),
        block_num=plan["cores"],
    )
    return out_k.view(bsz, max_seq, hd), out_v.view(bsz, max_seq, hd)


if __name__ == "__main__":
    import torch
    import torch_npu

    torch.npu.set_device(0)
    torch.manual_seed(0)

    nb, bs, hd = 8, 16, 128
    bsz, max_seq = 4, 40
    mb = (max_seq + bs - 1) // bs + 1

    cache_k = torch.randn(nb, bs, hd, dtype=torch.float32, device="npu")
    cache_v = torch.randn(nb, bs, hd, dtype=torch.float32, device="npu")
    bt = torch.stack([torch.randperm(nb, device="npu")[:mb] for _ in range(bsz)]).to(
        torch.int32
    )
    lens = torch.tensor([40, 17, 33, 1], dtype=torch.int32, device="npu")

    ok, ov = run(cache_k, cache_v, bt, lens, max_seq)

    ck, cv = cache_k.cpu(), cache_v.cpu()
    btc, lensc = bt.cpu(), lens.cpu()
    ref_k = torch.zeros(bsz, max_seq, hd)
    ref_v = torch.zeros(bsz, max_seq, hd)
    for b in range(bsz):
        for t in range(int(lensc[b])):
            blk = int(btc[b, t // bs])
            off = t % bs
            ref_k[b, t] = ck[blk, off]
            ref_v[b, t] = cv[blk, off]

    print("k max abs err:", (ok.cpu() - ref_k).abs().max().item())
    print("v max abs err:", (ov.cpu() - ref_v).abs().max().item())
