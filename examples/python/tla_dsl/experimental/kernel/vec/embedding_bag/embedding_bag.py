# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Row gather (``index_select``) and fixed-length ``embedding_bag`` in fp32.

``index_select``::

    y[i, :] = x[idx[i], :]

``embedding_bag`` with ``L`` indices per bag::

    out[b, :] = sum_{l < L} weight[indices[b, l], :]

Both operators move whole rows, so the kernel reads the row id as a *scalar*
(``gm_idx[r]``, a dynamic coordinate load) and then issues one row DMA per
output row instead of walking the elements thread by thread. Both entry points
are emitted from the single ``row_gather_kernel``: ``ACC=False`` copies one row
per output (``L = 1``), ``ACC=True`` sums ``L`` rows into a UB accumulator
before the output DMA.

High-performance design points
------------------------------
1. **Row-granular DMA instead of a per-element scalar loop.** Each output row
   reads its row id as a scalar and then moves the whole row with one DMA
   (``index_select`` adds one UB -> GM store), so a 16 KiB row travels at copy
   bandwidth instead of one element per thread.
2. **Multi-buffered pipeline.** ``BUFS`` row buffers are chained with a
   per-buffer MTE2 -> VECTOR (-> MTE3 for the copy mode) mutex, so the DMA of
   row ``i + 1`` overlaps the copy/accumulation of row ``i``; a buffer is
   refilled only after its consumer pipe releases it.
3. **UB accumulator.** In accumulation mode row 0 is DMA'd straight into the
   accumulator and rows 1..L-1 are folded in on the vector pipe, so the ``L``
   source rows are read exactly once and each bag emits one output DMA with no
   intermediate GM tensor.
4. **Accumulator reuse across bags.** VECTOR releases the accumulator mutex
   after the last add and MTE3 releases it after the output DMA, so the next
   bag's refill is ordered behind the previous store without extra sync
   objects.
5. **Masked tail lanes.** A UB row is padded to ``N_PAD = ceil(D / 64) * 64``
   so every vector chunk is a full 64-lane access; the trailing lanes of the
   last chunk are masked with ``update_mask`` when ``D % 64 != 0``, while the
   GM side stays dense (``stride = D``).
6. **Intra-core synchronization is all mutex.** Row buffers and the accumulator
   use ``tla.mutex`` only; no ``set_flag``/``wait_flag`` is emitted, and both
   modes share the same kernel body.
"""

import catlass.tla as tla

VL_ELE = 64  # fp32 vector length (256 B / 4 B)
UB_BYTES = 262144  # per-AIV UB capacity, used to size the row pipeline


@tla.kernel
def row_gather_kernel(
    gm_w: tla.Tensor,  # [R, D] fp32
    gm_idx: tla.Tensor,  # [N, L] int32
    gm_out: tla.Tensor,  # [N, D] fp32
    D: tla.Constexpr[int],
    N_PAD: tla.Constexpr[int],
    TAIL: tla.Constexpr[int],
    L: tla.Constexpr[int],
    BUFS: tla.Constexpr[int],
    ACC: tla.Constexpr[bool],
) -> None:
    n_out = gm_idx.origin_shape[0]

    w_ptr = [None] * BUFS
    mtx = [None] * BUFS
    for bi in tla.range_constexpr(BUFS):
        w_ptr[bi] = tla.allocate(N_PAD, tla.Float32, tla.AddressSpace.ub, 256)
        mtx[bi] = tla.mutex(resource="rg_row", id=bi)
    if tla.const_expr(ACC):
        acc_ptr = tla.allocate(N_PAD, tla.Float32, tla.AddressSpace.ub, 256)
        # The accumulator is written by the vector pipe and drained by one MTE3
        # copy per output row. ``acc_mtx`` carries the two orderings: VECTOR
        # acquires and releases it after the last add (so the MTE3 lock below
        # waits for the adds), and the MTE3 release orders that copy before the
        # next output row's MTE2 refill of the same buffer (a block may own
        # several bags when B > cores).
        acc_mtx = tla.mutex(resource="rg_acc", id=0)

    row_ub = tla.make_layout(
        tla.make_shape(1, D),
        tla.make_stride(N_PAD, 1),
        origin_shape=tla.make_shape(1, D),
    )
    row_gm = tla.make_layout(
        tla.make_shape(1, D),
        tla.make_stride(D, 1),
        origin_shape=tla.make_shape(1, D),
    )
    ub_row = [None] * BUFS
    for bj in tla.range_constexpr(BUFS):
        ub_row[bj] = tla.make_tensor(w_ptr[bj], row_ub)
    if tla.const_expr(ACC):
        ub_acc = tla.make_tensor(acc_ptr, row_ub)

    n_chunks = D // VL_ELE
    gid = tla.arch.block_idx()
    aiv_num = tla.arch.block_num()

    with tla.vector():
        if tla.const_expr(ACC):
            # Bags are independent, so one bag is owned by one block; the blocks
            # stride over the bag axis and the row pipeline inside a bag is the
            # same BUFS-deep chain as the copy mode.
            for out_row in tla.range(gid, n_out, aiv_num):
                # Row 0 overwrites the accumulator, so no zero-fill pass is needed.
                src0 = gm_w.ptr + gm_idx[out_row, 0].to(tla.Int64) * tla.Int64(D)
                acc_mtx.lock(pipe=tla.pipes.MTE2)
                tla.copy(ub_acc, tla.make_tensor(src0, row_gm))
                acc_mtx.unlock(pipe=tla.pipes.MTE2)

                # Wait for the row-0 transfer before the vector pipe reads it.
                acc_mtx.lock(pipe=tla.pipes.VECTOR)
                acc_mtx.unlock(pipe=tla.pipes.VECTOR)

                # Rows 1 .. L-1 are folded into the accumulator, one buffer each.
                for grp in tla.range((L - 1 + BUFS - 1) // BUFS):
                    for bi in tla.range_constexpr(BUFS):
                        row = 1 + grp * BUFS + bi
                        if row < L:
                            src = gm_w.ptr + gm_idx[out_row, row].to(
                                tla.Int64
                            ) * tla.Int64(D)
                            mtx[bi].lock(pipe=tla.pipes.MTE2)
                            tla.copy(ub_row[bi], tla.make_tensor(src, row_gm))
                            mtx[bi].unlock(pipe=tla.pipes.MTE2)

                            mtx[bi].lock(pipe=tla.pipes.VECTOR)
                            with tla.vec.func(mode="simd"):
                                mask = tla.create_mask(
                                    pattern=tla.mask.ALL, dtype=tla.Float32
                                )
                                if tla.const_expr(TAIL != 0):
                                    tail_mask, _ = tla.update_mask(
                                        TAIL, dtype=tla.Float32
                                    )
                                for c1 in tla.range(n_chunks):
                                    a1 = tla.tile_view(
                                        ub_acc,
                                        tla.make_shape(1, VL_ELE),
                                        tla.make_coord(0, c1),
                                    ).load()
                                    r1 = tla.tile_view(
                                        ub_row[bi],
                                        tla.make_shape(1, VL_ELE),
                                        tla.make_coord(0, c1),
                                    ).load()
                                    tla.tile_view(
                                        ub_acc,
                                        tla.make_shape(1, VL_ELE),
                                        tla.make_coord(0, c1),
                                    ).store(tla.add(a1, r1, mask=mask))
                                if tla.const_expr(TAIL != 0):
                                    at = tla.tile_view(
                                        ub_acc,
                                        tla.make_shape(1, VL_ELE),
                                        tla.make_coord(0, n_chunks),
                                    ).load()
                                    rt = tla.tile_view(
                                        ub_row[bi],
                                        tla.make_shape(1, VL_ELE),
                                        tla.make_coord(0, n_chunks),
                                    ).load()
                                    tla.tile_view(
                                        ub_acc,
                                        tla.make_shape(1, VL_ELE),
                                        tla.make_coord(0, n_chunks),
                                    ).store(tla.add(at, rt, mask=tail_mask))
                                # The next row reads what this one just stored in UB.
                                tla.local_mem_bar(
                                    tla.params.MemType.VEC_STORE,
                                    tla.params.MemType.VEC_LOAD,
                                )
                            mtx[bi].unlock(pipe=tla.pipes.VECTOR)

                acc_mtx.lock(pipe=tla.pipes.VECTOR)
                acc_mtx.unlock(pipe=tla.pipes.VECTOR)
                dst = gm_out.ptr + out_row.to(tla.Int64) * tla.Int64(D)
                acc_mtx.lock(pipe=tla.pipes.MTE3)
                tla.copy(tla.make_tensor(dst, row_gm), ub_acc)
                acc_mtx.unlock(pipe=tla.pipes.MTE3)
        else:
            # Copy mode (``L == 1``): each block owns a contiguous slice of the
            # output rows and every row runs a full MTE2 -> MTE3 chain, so the
            # gather of row i + 1 overlaps the store of row i.
            per = (n_out + aiv_num - 1) // aiv_num
            begin = gid * per
            for grp in tla.range((per + BUFS - 1) // BUFS):
                for bi in tla.range_constexpr(BUFS):
                    r = begin + grp * BUFS + bi
                    if r < n_out:
                        src = gm_w.ptr + gm_idx[r, 0].to(tla.Int64) * tla.Int64(D)
                        dst = gm_out.ptr + r.to(tla.Int64) * tla.Int64(D)
                        mtx[bi].lock(pipe=tla.pipes.MTE2)
                        tla.copy(ub_row[bi], tla.make_tensor(src, row_gm))
                        mtx[bi].unlock(pipe=tla.pipes.MTE2)
                        mtx[bi].lock(pipe=tla.pipes.MTE3)
                        tla.copy(tla.make_tensor(dst, row_gm), ub_row[bi])
                        mtx[bi].unlock(pipe=tla.pipes.MTE3)


# ======================================================================================
# HOST side: shape checks, compile and launch.
# ======================================================================================

_COMPILED = {}

_CORES = []


def _vector_cores(index):
    import torch

    if not _CORES:
        _CORES.append(int(torch.npu.get_device_properties(index).vector_core_num))
    return _CORES[0]


def _pipeline(n_pad):
    """Return ``(N_PAD, BUFS)`` for a row width of ``n_pad`` floats in UB.

    ``BUFS`` rows of ``n_pad`` floats are kept in flight (plus one accumulator
    row for ``embedding_bag`` is accounted for by the caller); the depth is
    capped at 8 because the row DMAs are short and more buffers stop helping.
    """
    budget = UB_BYTES - 8192
    if 4 * n_pad > budget:
        raise ValueError("one padded row does not fit in UB")
    bufs = max(1, min(8, budget // (4 * n_pad)))
    return n_pad, bufs


def _launch(acc, table, sel, out, d, n_pad, n_idx, bufs, cores):
    """Compile (once per shape/mode) and launch ``row_gather_kernel``."""
    from catlass.tla.runtime import from_dlpack

    key = ("rg", acc, sel.shape[0], n_idx, d)
    plan = _COMPILED.get(key)
    if plan is None:
        w_t = from_dlpack(table, layout_tag=tla.arch.RowMajor)
        i_t = from_dlpack(sel, layout_tag=tla.arch.RowMajor)
        o_t = from_dlpack(out, layout_tag=tla.arch.RowMajor)
        artifact = tla.compile(
            row_gather_kernel,
            w_t,
            i_t,
            o_t,
            d,
            n_pad,
            d - (d // VL_ELE) * VL_ELE,
            n_idx,
            bufs,
            acc,
            options="--npu-arch 3510",
        )
        plan = {"artifact": artifact, "cores": cores, "key": key}
        _COMPILED[key] = plan

    plan["artifact"](
        from_dlpack(table, layout_tag=tla.arch.RowMajor),
        from_dlpack(sel, layout_tag=tla.arch.RowMajor),
        from_dlpack(out, layout_tag=tla.arch.RowMajor),
        block_num=plan["cores"],
    )


def run_index_select(x, idx):
    """Return ``x[idx]`` (row gather) as a new ``[M, D]`` fp32 tensor."""
    import torch

    if x.dtype != torch.float32 or x.dim() != 2:
        raise ValueError("x must be a 2-D float32 tensor")
    if idx.dim() != 1:
        raise ValueError("idx must be 1-D")

    r, d = x.shape
    m = idx.shape[0]
    device = x.device
    sel = idx.to(torch.int32).contiguous().view(m, 1)

    out = torch.empty((m, d), dtype=torch.float32, device=device)
    if m == 0:
        return out

    xc = x.contiguous()
    if ("rg", False, m, 1, d) not in _COMPILED:
        # Validated once per shape: the reduction below synchronises the device,
        # which would otherwise cost more than the gather itself.
        if int(sel.min()) < 0 or int(sel.max()) >= r:
            raise ValueError("idx out of range [0, {})".format(r))
    n_pad, bufs = _pipeline((d + VL_ELE - 1) // VL_ELE * VL_ELE)
    _launch(False, xc, sel, out, d, n_pad, 1, bufs, _vector_cores(0))
    return out


def run_embedding_bag(weight, indices):
    """Return the fp32 sum of ``weight[indices[b, l]]`` over ``l`` for every bag."""
    import torch

    if weight.dtype != torch.float32 or weight.dim() != 2:
        raise ValueError("weight must be a 2-D float32 tensor")
    if indices.dim() != 2:
        raise ValueError("indices must be 2-D [B, L]")

    r, d = weight.shape
    b, seq_len = indices.shape
    device = weight.device
    sel = indices.to(torch.int32).contiguous()

    out = torch.empty((b, d), dtype=torch.float32, device=device)
    if b == 0:
        return out
    if seq_len == 0:
        out.zero_()
        return out

    wc = weight.contiguous()
    if ("rg", True, b, seq_len, d) not in _COMPILED:
        # Validated once per shape: the reduction below synchronises the device.
        if int(sel.min()) < 0 or int(sel.max()) >= r:
            raise ValueError("indices out of range [0, {})".format(r))
    n_pad = (d + VL_ELE - 1) // VL_ELE * VL_ELE
    # One accumulator row plus at least one staging row must fit in UB.
    if 2 * 4 * n_pad > UB_BYTES - 8192:
        raise ValueError("D is too large: a padded row does not fit in UB")
    bufs = max(1, min(8, (UB_BYTES - 8192 - 4 * n_pad) // (4 * n_pad)))
    _launch(True, wc, sel, out, d, n_pad, seq_len, bufs, _vector_cores(0))
    return out


if __name__ == "__main__":
    import torch
    import torch_npu

    torch.npu.set_device(0)
    torch.manual_seed(0)

    for d in (96, 100):
        x = torch.randn(128, d, dtype=torch.float32, device="npu")
        idx = torch.randint(0, 128, (37,), dtype=torch.int32, device="npu")
        y = run_index_select(x, idx)
        print(
            "index_select D={} err:".format(d),
            (y.cpu() - x.cpu()[idx.cpu().long()]).abs().max().item(),
        )

        w = torch.randn(64, d, dtype=torch.float32, device="npu")
        ind = torch.randint(0, 64, (16, 5), dtype=torch.int32, device="npu")
        out = run_embedding_bag(w, ind)
        ref = w.cpu()[ind.cpu().long()].sum(dim=1)
        print("embedding_bag D={} err:".format(d), (out.cpu() - ref).abs().max().item())
