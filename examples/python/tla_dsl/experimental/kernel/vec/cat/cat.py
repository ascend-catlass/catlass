# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""High-performance 2-D concatenation for Ascend AIV.

There is no direct GM-to-GM copy, so concatenation is staged through UB. For up
to three inputs, one fused launch covers each direction:

* ``cat_dim0_3_kernel`` - each input is a contiguous destination segment.  The
  fused kernel uses measured 32 KiB source chunks and one mutex per staging
  buffer; only the trailing partial chunk needs a shorter DMA.
* ``cat_dim1_3_kernel`` - rows interleave, so the copy is a *strided 2-D DMA*:
  one batch of ``BR`` rows is read contiguously and written back with row
  stride ``NT``.  Tail rows are copied one row at a time.

The old per-input kernels remain as the fallback for more than three inputs.

Every GM alignment is supported: the source/destination row length, destination
column offset and row strides may be non-multiples of 32 B.  The MTE accepts
arbitrary element offsets and byte counts; only UB accesses must be 32 B
aligned, which the 256 B-aligned staging buffer guarantees.

Performance design:
* One fused launch avoids host launch count growing with the number of inputs.
* One 32 KiB DMA pair per task balances AIV waves against enough bytes in
  flight to approach HBM bandwidth.
* ``dim=0`` assigns contiguous chunks round-robin to AIV cores, so every core
  has one producer/consumer pair and no cross-core reduction.
* ``dim=1`` batches up to 64 rows into one strided 2-D DMA.  This turns many
  per-row copies into one source copy plus one destination copy.
* Each staging buffer owns one mutex.  MTE2 publishes GM->UB data; MTE3 consumes
  it and releases the same buffer for the next iteration.  No all-pipe barrier
  is needed because the dependency is scoped to that physical UB resource.
"""

import ctypes
import functools
import inspect

import torch
import torch_npu

import catlass.tla as tla
from catlass.tla.runtime import _Tensor

# The legacy fallback keeps a large tile.  The fused path uses 8192 elements
# (32 KiB), selected by the measured small/tail/large-shape sweep.
FLAT_CHUNK = 49152
STAGE_ELEMS = 49152
MAX_BATCH_ROWS = 64


@tla.kernel
def cat_dim0_kernel(
    gm_src: tla.Tensor,  # [M*N] fp32, contiguous
    gm_dst: tla.Tensor,  # [MT*N] fp32
    DST_ELEM: tla.Constexpr[int],  # flat element offset in the destination
    TOTAL: tla.Constexpr[int],  # elements to copy
) -> None:
    gid = tla.arch.block_idx()
    aiv_num = tla.arch.block_num()

    ptr = tla.allocate(FLAT_CHUNK, tla.Float32, tla.AddressSpace.ub, 256)
    # One mutex guards the one physical staging buffer.  Full chunks and the
    # tail deliberately share it, so tail work also waits for any preceding
    # MTE3 consumption before overwriting the same UB address range.
    mutex_ub = tla.mutex(resource="cat_dim0_ub", id=0)
    full_lay = tla.make_layout(
        tla.make_shape(1, FLAT_CHUNK),
        tla.make_stride(FLAT_CHUNK, 1),
        origin_shape=tla.make_shape(1, FLAT_CHUNK),
    )
    rem = TOTAL % FLAT_CHUNK
    rem_lay = tla.make_layout(
        tla.make_shape(1, rem if rem else 1),
        tla.make_stride(rem if rem else 1, 1),
        origin_shape=tla.make_shape(1, rem if rem else 1),
    )
    ub_full = tla.make_tensor(ptr, full_lay)
    ub_rem = tla.make_tensor(ptr, rem_lay)
    n_units = TOTAL // FLAT_CHUNK

    with tla.vector():
        # Round-robin ownership keeps each core's sequence contiguous in the
        # stream.  Each unit is 192 KiB and has exactly one reader/writer.
        for u in tla.range(gid, n_units, aiv_num):
            off = tla.Int64(u) * tla.Int64(FLAT_CHUNK)
            mutex_ub.lock(pipe=tla.arch.MTE2)
            tla.copy(ub_full, tla.make_tensor(gm_src.ptr + off, full_lay))
            mutex_ub.unlock(pipe=tla.arch.MTE2)

            mutex_ub.lock(pipe=tla.arch.MTE3)
            tla.copy(
                tla.make_tensor(gm_dst.ptr + tla.Int64(DST_ELEM) + off, full_lay),
                ub_full,
            )
            mutex_ub.unlock(pipe=tla.arch.MTE3)

        # The partial tail is assigned to one block only.  Avoiding a launch per
        # row/chunk keeps the expensive path at exactly one extra short DMA.
        if tla.const_expr(rem != 0):
            for t in tla.range(gid, 1, aiv_num):
                mutex_ub.lock(pipe=tla.arch.MTE2)
                tla.copy(
                    ub_rem,
                    tla.make_tensor(
                        gm_src.ptr + tla.Int64(n_units * FLAT_CHUNK), rem_lay
                    ),
                )
                mutex_ub.unlock(pipe=tla.arch.MTE2)

                mutex_ub.lock(pipe=tla.arch.MTE3)
                tla.copy(
                    tla.make_tensor(
                        gm_dst.ptr + tla.Int64(DST_ELEM + n_units * FLAT_CHUNK),
                        rem_lay,
                    ),
                    ub_rem,
                )
                mutex_ub.unlock(pipe=tla.arch.MTE3)


@tla.kernel
def cat_dim1_kernel(
    gm_src: tla.Tensor,  # [M, NS] fp32
    gm_dst: tla.Tensor,  # [M, NT] fp32
    M: tla.Constexpr[int],
    NS: tla.Constexpr[int],
    NT: tla.Constexpr[int],
    COL: tla.Constexpr[int],
    BR: tla.Constexpr[int],  # rows per batch
) -> None:
    gid = tla.arch.block_idx()
    aiv_num = tla.arch.block_num()

    ptr = tla.allocate(BR * NS, tla.Float32, tla.AddressSpace.ub, 256)
    # Group and tail copies alias this UB buffer, so both use the same mutex.
    # MTE2 -> MTE3 publishes a completed row group; MTE3 -> next MTE2 prevents
    # the next group from overwriting data while the previous store is active.
    mutex_ub = tla.mutex(resource="cat_dim1_ub", id=0)
    ub_lay = tla.make_layout(
        tla.make_shape(BR, NS),
        tla.make_stride(NS, 1),
        origin_shape=tla.make_shape(BR, NS),
    )
    dst_lay = tla.make_layout(
        tla.make_shape(BR, NS),
        tla.make_stride(NT, 1),
        origin_shape=tla.make_shape(BR, NS),
    )
    ub = tla.make_tensor(ptr, ub_lay)
    row_lay = tla.make_layout(
        tla.make_shape(1, NS),
        tla.make_stride(NS, 1),
        origin_shape=tla.make_shape(1, NS),
    )
    ub_row = tla.make_tensor(ptr, row_lay)

    n_groups = M // BR
    tail = M % BR

    with tla.vector():
        # One source DMA reads BR contiguous rows; one strided destination DMA
        # writes those rows into the output column slice.  Batching amortizes
        # descriptor setup and converts BR row transactions into two DMAs.
        for b in tla.range(gid, n_groups, aiv_num):
            src_off = tla.Int64(b) * tla.Int64(BR * NS)
            dst_off = tla.Int64(b) * tla.Int64(BR * NT) + tla.Int64(COL)
            mutex_ub.lock(pipe=tla.arch.MTE2)
            tla.copy(
                ub,
                tla.make_tensor(gm_src.ptr + src_off, ub_lay),
            )
            mutex_ub.unlock(pipe=tla.arch.MTE2)

            mutex_ub.lock(pipe=tla.arch.MTE3)
            tla.copy(
                tla.make_tensor(gm_dst.ptr + dst_off, dst_lay),
                ub,
            )
            mutex_ub.unlock(pipe=tla.arch.MTE3)

        # Tail rows remain one DMA each.  There are at most BR-1 <= 63 rows, so
        # this bounded serial path does not dominate any meaningful batch.
        if tla.const_expr(tail != 0):
            for t in tla.range(gid, tail, aiv_num):
                mutex_ub.lock(pipe=tla.arch.MTE2)
                tla.copy(
                    ub_row,
                    tla.make_tensor(
                        gm_src.ptr
                        + (t + tla.Int32(n_groups * BR)).to(tla.Int64) * tla.Int64(NS),
                        row_lay,
                    ),
                )
                mutex_ub.unlock(pipe=tla.arch.MTE2)

                mutex_ub.lock(pipe=tla.arch.MTE3)
                tla.copy(
                    tla.make_tensor(
                        gm_dst.ptr
                        + (t + tla.Int32(n_groups * BR)).to(tla.Int64) * tla.Int64(NT)
                        + tla.Int64(COL),
                        row_lay,
                    ),
                    ub_row,
                )
                mutex_ub.unlock(pipe=tla.arch.MTE3)


@tla.kernel
def cat_dim0_3_kernel(
    gm_src0: tla.Tensor,
    gm_src1: tla.Tensor,
    gm_src2: tla.Tensor,
    gm_dst: tla.Tensor,
    TOTAL0: tla.Constexpr[int],
    TOTAL1: tla.Constexpr[int],
    TOTAL2: tla.Constexpr[int],
    CHUNK: tla.Constexpr[int],
) -> None:
    """Copy up to three contiguous inputs with one AIV launch.

    Empty slots are compiled away.  One launch replaces two or three scalar
    launches while preserving the 32 KiB DMA tile and mutex hand-off.
    """
    gid = tla.arch.block_idx()
    aiv_num = tla.arch.block_num()

    ptr = tla.allocate(CHUNK, tla.Float32, tla.AddressSpace.ub, 256)
    mutex_ub = tla.mutex(resource="cat_dim0_ub_3", id=0)
    full_lay = tla.make_layout(
        tla.make_shape(1, CHUNK),
        tla.make_stride(CHUNK, 1),
        origin_shape=tla.make_shape(1, CHUNK),
    )
    ub_full = tla.make_tensor(ptr, full_lay)

    rem0 = TOTAL0 % CHUNK
    rem1 = TOTAL1 % CHUNK
    rem2 = TOTAL2 % CHUNK
    rem_lay0 = tla.make_layout(
        tla.make_shape(1, rem0 if rem0 else 1),
        tla.make_stride(rem0 if rem0 else 1, 1),
        origin_shape=tla.make_shape(1, rem0 if rem0 else 1),
    )
    rem_lay1 = tla.make_layout(
        tla.make_shape(1, rem1 if rem1 else 1),
        tla.make_stride(rem1 if rem1 else 1, 1),
        origin_shape=tla.make_shape(1, rem1 if rem1 else 1),
    )
    rem_lay2 = tla.make_layout(
        tla.make_shape(1, rem2 if rem2 else 1),
        tla.make_stride(rem2 if rem2 else 1, 1),
        origin_shape=tla.make_shape(1, rem2 if rem2 else 1),
    )
    ub_rem0 = tla.make_tensor(ptr, rem_lay0)
    ub_rem1 = tla.make_tensor(ptr, rem_lay1)
    ub_rem2 = tla.make_tensor(ptr, rem_lay2)
    n_units0 = TOTAL0 // CHUNK
    n_units1 = TOTAL1 // CHUNK
    n_units2 = TOTAL2 // CHUNK

    with tla.vector():
        if tla.const_expr(TOTAL0 != 0):
            for u in tla.range(gid, n_units0, aiv_num):
                off = tla.Int64(u) * tla.Int64(CHUNK)
                mutex_ub.lock(pipe=tla.arch.MTE2)
                tla.copy(ub_full, tla.make_tensor(gm_src0.ptr + off, full_lay))
                mutex_ub.unlock(pipe=tla.arch.MTE2)
                mutex_ub.lock(pipe=tla.arch.MTE3)
                tla.copy(tla.make_tensor(gm_dst.ptr + off, full_lay), ub_full)
                mutex_ub.unlock(pipe=tla.arch.MTE3)
            if tla.const_expr(rem0 != 0):
                for t in tla.range(gid, 1, aiv_num):
                    mutex_ub.lock(pipe=tla.arch.MTE2)
                    tla.copy(
                        ub_rem0,
                        tla.make_tensor(
                            gm_src0.ptr + tla.Int64(n_units0 * CHUNK),
                            rem_lay0,
                        ),
                    )
                    mutex_ub.unlock(pipe=tla.arch.MTE2)
                    mutex_ub.lock(pipe=tla.arch.MTE3)
                    tla.copy(
                        tla.make_tensor(
                            gm_dst.ptr + tla.Int64(n_units0 * CHUNK),
                            rem_lay0,
                        ),
                        ub_rem0,
                    )
                    mutex_ub.unlock(pipe=tla.arch.MTE3)

        if tla.const_expr(TOTAL1 != 0):
            dst1 = tla.Int64(TOTAL0)
            for u in tla.range(gid, n_units1, aiv_num):
                off = tla.Int64(u) * tla.Int64(CHUNK)
                mutex_ub.lock(pipe=tla.arch.MTE2)
                tla.copy(ub_full, tla.make_tensor(gm_src1.ptr + off, full_lay))
                mutex_ub.unlock(pipe=tla.arch.MTE2)
                mutex_ub.lock(pipe=tla.arch.MTE3)
                tla.copy(tla.make_tensor(gm_dst.ptr + dst1 + off, full_lay), ub_full)
                mutex_ub.unlock(pipe=tla.arch.MTE3)
            if tla.const_expr(rem1 != 0):
                for t in tla.range(gid, 1, aiv_num):
                    mutex_ub.lock(pipe=tla.arch.MTE2)
                    tla.copy(
                        ub_rem1,
                        tla.make_tensor(
                            gm_src1.ptr + tla.Int64(n_units1 * CHUNK),
                            rem_lay1,
                        ),
                    )
                    mutex_ub.unlock(pipe=tla.arch.MTE2)
                    mutex_ub.lock(pipe=tla.arch.MTE3)
                    tla.copy(
                        tla.make_tensor(
                            gm_dst.ptr + dst1 + tla.Int64(n_units1 * CHUNK),
                            rem_lay1,
                        ),
                        ub_rem1,
                    )
                    mutex_ub.unlock(pipe=tla.arch.MTE3)

        if tla.const_expr(TOTAL2 != 0):
            dst2 = tla.Int64(TOTAL0 + TOTAL1)
            for u in tla.range(gid, n_units2, aiv_num):
                off = tla.Int64(u) * tla.Int64(CHUNK)
                mutex_ub.lock(pipe=tla.arch.MTE2)
                tla.copy(ub_full, tla.make_tensor(gm_src2.ptr + off, full_lay))
                mutex_ub.unlock(pipe=tla.arch.MTE2)
                mutex_ub.lock(pipe=tla.arch.MTE3)
                tla.copy(tla.make_tensor(gm_dst.ptr + dst2 + off, full_lay), ub_full)
                mutex_ub.unlock(pipe=tla.arch.MTE3)
            if tla.const_expr(rem2 != 0):
                for t in tla.range(gid, 1, aiv_num):
                    mutex_ub.lock(pipe=tla.arch.MTE2)
                    tla.copy(
                        ub_rem2,
                        tla.make_tensor(
                            gm_src2.ptr + tla.Int64(n_units2 * CHUNK),
                            rem_lay2,
                        ),
                    )
                    mutex_ub.unlock(pipe=tla.arch.MTE2)
                    mutex_ub.lock(pipe=tla.arch.MTE3)
                    tla.copy(
                        tla.make_tensor(
                            gm_dst.ptr + dst2 + tla.Int64(n_units2 * CHUNK),
                            rem_lay2,
                        ),
                        ub_rem2,
                    )
                    mutex_ub.unlock(pipe=tla.arch.MTE3)


@tla.kernel
def cat_dim1_3_kernel(
    gm_src0: tla.Tensor,
    gm_src1: tla.Tensor,
    gm_src2: tla.Tensor,
    gm_dst: tla.Tensor,
    M: tla.Constexpr[int],
    NS0: tla.Constexpr[int],
    NS1: tla.Constexpr[int],
    NS2: tla.Constexpr[int],
    NT: tla.Constexpr[int],
    COL0: tla.Constexpr[int],
    COL1: tla.Constexpr[int],
    COL2: tla.Constexpr[int],
    BR0: tla.Constexpr[int],
    BR1: tla.Constexpr[int],
    BR2: tla.Constexpr[int],
    BUF_ELEMS: tla.Constexpr[int],
) -> None:
    """Copy up to three strided column slices with one AIV launch."""
    gid = tla.arch.block_idx()
    aiv_num = tla.arch.block_num()

    ptr = tla.allocate(BUF_ELEMS, tla.Float32, tla.AddressSpace.ub, 256)
    mutex_ub = tla.mutex(resource="cat_dim1_ub_3", id=0)

    with tla.vector():
        if tla.const_expr(NS0 != 0 and M != 0):
            lay0 = tla.make_layout(
                tla.make_shape(BR0, NS0),
                tla.make_stride(NS0, 1),
                origin_shape=tla.make_shape(BR0, NS0),
            )
            dst0 = tla.make_layout(
                tla.make_shape(BR0, NS0),
                tla.make_stride(NT, 1),
                origin_shape=tla.make_shape(BR0, NS0),
            )
            row0 = tla.make_layout(
                tla.make_shape(1, NS0),
                tla.make_stride(NS0, 1),
                origin_shape=tla.make_shape(1, NS0),
            )
            ub0 = tla.make_tensor(ptr, lay0)
            ub_row0 = tla.make_tensor(ptr, row0)
            groups0 = M // BR0
            tail0 = M % BR0
            for b in tla.range(gid, groups0, aiv_num):
                src_off = tla.Int64(b) * tla.Int64(BR0 * NS0)
                dst_off = tla.Int64(b) * tla.Int64(BR0 * NT) + tla.Int64(COL0)
                mutex_ub.lock(pipe=tla.arch.MTE2)
                tla.copy(ub0, tla.make_tensor(gm_src0.ptr + src_off, lay0))
                mutex_ub.unlock(pipe=tla.arch.MTE2)
                mutex_ub.lock(pipe=tla.arch.MTE3)
                tla.copy(tla.make_tensor(gm_dst.ptr + dst_off, dst0), ub0)
                mutex_ub.unlock(pipe=tla.arch.MTE3)
            if tla.const_expr(tail0 != 0):
                for t in tla.range(gid, tail0, aiv_num):
                    row = t + tla.Int32(groups0 * BR0)
                    mutex_ub.lock(pipe=tla.arch.MTE2)
                    tla.copy(
                        ub_row0,
                        tla.make_tensor(
                            gm_src0.ptr + row.to(tla.Int64) * tla.Int64(NS0),
                            row0,
                        ),
                    )
                    mutex_ub.unlock(pipe=tla.arch.MTE2)
                    mutex_ub.lock(pipe=tla.arch.MTE3)
                    tla.copy(
                        tla.make_tensor(
                            gm_dst.ptr
                            + row.to(tla.Int64) * tla.Int64(NT)
                            + tla.Int64(COL0),
                            row0,
                        ),
                        ub_row0,
                    )
                    mutex_ub.unlock(pipe=tla.arch.MTE3)

        if tla.const_expr(NS1 != 0 and M != 0):
            lay1 = tla.make_layout(
                tla.make_shape(BR1, NS1),
                tla.make_stride(NS1, 1),
                origin_shape=tla.make_shape(BR1, NS1),
            )
            dst1 = tla.make_layout(
                tla.make_shape(BR1, NS1),
                tla.make_stride(NT, 1),
                origin_shape=tla.make_shape(BR1, NS1),
            )
            row1 = tla.make_layout(
                tla.make_shape(1, NS1),
                tla.make_stride(NS1, 1),
                origin_shape=tla.make_shape(1, NS1),
            )
            ub1 = tla.make_tensor(ptr, lay1)
            ub_row1 = tla.make_tensor(ptr, row1)
            groups1 = M // BR1
            tail1 = M % BR1
            for b in tla.range(gid, groups1, aiv_num):
                src_off = tla.Int64(b) * tla.Int64(BR1 * NS1)
                dst_off = tla.Int64(b) * tla.Int64(BR1 * NT) + tla.Int64(COL1)
                mutex_ub.lock(pipe=tla.arch.MTE2)
                tla.copy(ub1, tla.make_tensor(gm_src1.ptr + src_off, lay1))
                mutex_ub.unlock(pipe=tla.arch.MTE2)
                mutex_ub.lock(pipe=tla.arch.MTE3)
                tla.copy(tla.make_tensor(gm_dst.ptr + dst_off, dst1), ub1)
                mutex_ub.unlock(pipe=tla.arch.MTE3)
            if tla.const_expr(tail1 != 0):
                for t in tla.range(gid, tail1, aiv_num):
                    row = t + tla.Int32(groups1 * BR1)
                    mutex_ub.lock(pipe=tla.arch.MTE2)
                    tla.copy(
                        ub_row1,
                        tla.make_tensor(
                            gm_src1.ptr + row.to(tla.Int64) * tla.Int64(NS1),
                            row1,
                        ),
                    )
                    mutex_ub.unlock(pipe=tla.arch.MTE2)
                    mutex_ub.lock(pipe=tla.arch.MTE3)
                    tla.copy(
                        tla.make_tensor(
                            gm_dst.ptr
                            + row.to(tla.Int64) * tla.Int64(NT)
                            + tla.Int64(COL1),
                            row1,
                        ),
                        ub_row1,
                    )
                    mutex_ub.unlock(pipe=tla.arch.MTE3)

        if tla.const_expr(NS2 != 0 and M != 0):
            lay2 = tla.make_layout(
                tla.make_shape(BR2, NS2),
                tla.make_stride(NS2, 1),
                origin_shape=tla.make_shape(BR2, NS2),
            )
            dst2 = tla.make_layout(
                tla.make_shape(BR2, NS2),
                tla.make_stride(NT, 1),
                origin_shape=tla.make_shape(BR2, NS2),
            )
            row2 = tla.make_layout(
                tla.make_shape(1, NS2),
                tla.make_stride(NS2, 1),
                origin_shape=tla.make_shape(1, NS2),
            )
            ub2 = tla.make_tensor(ptr, lay2)
            ub_row2 = tla.make_tensor(ptr, row2)
            groups2 = M // BR2
            tail2 = M % BR2
            for b in tla.range(gid, groups2, aiv_num):
                src_off = tla.Int64(b) * tla.Int64(BR2 * NS2)
                dst_off = tla.Int64(b) * tla.Int64(BR2 * NT) + tla.Int64(COL2)
                mutex_ub.lock(pipe=tla.arch.MTE2)
                tla.copy(ub2, tla.make_tensor(gm_src2.ptr + src_off, lay2))
                mutex_ub.unlock(pipe=tla.arch.MTE2)
                mutex_ub.lock(pipe=tla.arch.MTE3)
                tla.copy(tla.make_tensor(gm_dst.ptr + dst_off, dst2), ub2)
                mutex_ub.unlock(pipe=tla.arch.MTE3)
            if tla.const_expr(tail2 != 0):
                for t in tla.range(gid, tail2, aiv_num):
                    row = t + tla.Int32(groups2 * BR2)
                    mutex_ub.lock(pipe=tla.arch.MTE2)
                    tla.copy(
                        ub_row2,
                        tla.make_tensor(
                            gm_src2.ptr + row.to(tla.Int64) * tla.Int64(NS2),
                            row2,
                        ),
                    )
                    mutex_ub.unlock(pipe=tla.arch.MTE2)
                    mutex_ub.lock(pipe=tla.arch.MTE3)
                    tla.copy(
                        tla.make_tensor(
                            gm_dst.ptr
                            + row.to(tla.Int64) * tla.Int64(NT)
                            + tla.Int64(COL2),
                            row2,
                        ),
                        ub_row2,
                    )
                    mutex_ub.unlock(pipe=tla.arch.MTE3)


# ======================================================================================
# HOST side
# ======================================================================================

_COMPILED = {}
_BINDINGS = {}
_LAUNCH_PLANS = {}
_CALL_PLANS = {}
_CALL_OUTPUTS = {}
_CORES = {}
_LAUNCH_ACCEPTS_PRINT = None
_MAX_BINDINGS = 4096
_MAX_PLANS = 512
_MAX_CALL_PLANS = 256
_MAX_CALL_OUTPUTS = 512


def reset_host_cache():
    """Drop pointer-keyed bindings and launch plans."""
    _COMPILED.clear()
    _BINDINGS.clear()
    _LAUNCH_PLANS.clear()
    _CALL_PLANS.clear()
    _CALL_OUTPUTS.clear()


def _device_index(device):
    return 0 if device.index is None else int(device.index)


def _cores(device_index):
    cores = _CORES.get(device_index)
    if cores is None:
        cores = int(torch.npu.get_device_properties(device_index).vector_core_num)
        _CORES[device_index] = cores
    return cores


def _current_stream(device_index):
    try:
        return int(torch_npu._C._npu_getCurrentRawStreamNoWait(device_index))
    except AttributeError:
        return int(torch.npu.current_stream(device_index).npu_stream)


def _call_key(xs, dim):
    """Return a stable key for the fully validated, contiguous hot path."""
    try:
        if any(not x.is_contiguous() for x in xs):
            return None
        device_index = _device_index(xs[0].device)
        return (
            dim,
            device_index,
            tuple(
                (
                    x.data_ptr(),
                    x._version,
                    x.dtype,
                    tuple(x.shape),
                    tuple(x.stride()),
                    x.storage_offset(),
                )
                for x in xs
            ),
        )
    except Exception:
        return None


def _bind(tensor, rows, cols):
    """Build a dense FP32 TLA tensor without repeated DLPack conversion."""
    key = (tensor.data_ptr(), rows, cols)
    bound = _BINDINGS.get(key)
    if bound is None:
        bound = _Tensor(
            (rows, cols),
            tla.Float32,
            origin_shape=(rows, cols),
            coord=(0, 0),
            stride=(cols, 1),
            layout_tag=tla.arch.RowMajor,
            data_ptr=tensor.data_ptr(),
        )
        if len(_BINDINGS) >= _MAX_BINDINGS:
            _BINDINGS.clear()
        _BINDINGS[key] = bound
    return bound


def _acquire_output(rows, cols, device, device_index, stream):
    """Use the native caching allocator; it is faster than a Python pool."""
    return torch.empty((rows, cols), dtype=torch.float32, device=device)


def _resolve_launcher(artifact):
    """Resolve the reusable host-args launcher of the installed DSL runtime."""
    try:
        from catlass import execution as execution_mod
        from catlass.base_dsl.jit_executor import _resolved_dyn_ubuf_bytes

        executor = artifact._get_executor()
        module = executor.jit_module
        if getattr(module, "uses_scalar_print", False) or getattr(
            module, "uses_tensor_print", False
        ):
            return None
        return (
            execution_mod,
            module.execution_args,
            _resolved_dyn_ubuf_bytes(module, 0),
            executor.function_handle,
            module,
        )
    except Exception:
        return None


def _make_launcher(artifact, tensors, block_num, stream):
    """Pre-pack host arguments once for a stable pointer/stream tuple."""
    plumbing = _resolve_launcher(artifact)
    if plumbing is None:
        return None
    execution_mod, execution_args, dyn_ub, function_handle, module = plumbing
    try:
        payload = execution_args.generate_launch_payload(tensors)
    except Exception:
        return None
    if payload is None:
        return None
    global _LAUNCH_ACCEPTS_PRINT
    if _LAUNCH_ACCEPTS_PRINT is None:
        parameters = inspect.signature(execution_mod.launch_kernel).parameters
        _LAUNCH_ACCEPTS_PRINT = "uses_scalar_print" in parameters
    try:
        import acl

        host_buf = (ctypes.c_uint8 * len(payload)).from_buffer_copy(payload)
        host_addr = ctypes.addressof(host_buf)
        cfg = [] if dyn_ub <= 0 else [{"id": 2, "value": int(dyn_ub)}]
        payload_bytes = len(payload)
        launch_acl = acl.rt.launch_kernel_with_host_args
        placeholders = []
        function_handle = int(function_handle)
        block_num = int(block_num)
        stream = int(stream)

        def direct_launch():
            ret = launch_acl(
                function_handle,
                block_num,
                stream,
                cfg,
                host_addr,
                payload_bytes,
                placeholders,
                0,
            )
            if ret:
                raise RuntimeError(
                    "acl.rt.launch_kernel_with_host_args failed: {}".format(ret)
                )

        # Keep the host buffer alive through the closure.
        direct_launch._host_buf = host_buf
        return direct_launch
    except Exception:
        pass
    kwargs = {
        "function_handle": function_handle,
        "stream": stream,
        "block_num": block_num,
        "payload": payload,
        "dyn_ubuf_bytes": dyn_ub,
    }
    if _LAUNCH_ACCEPTS_PRINT:
        kwargs["uses_scalar_print"] = getattr(module, "uses_scalar_print", False)
    return functools.partial(execution_mod.launch_kernel, **kwargs)


def _launch(kernel, tensors, extra_args, key, device_index):
    """Launch with pointer-keyed bindings and a pre-packed host argument blob."""
    stream = _current_stream(device_index)
    compile_key = (
        device_index,
        key,
        tuple(tuple(tensor.shape) for tensor in tensors),
    )
    launch_key = (
        compile_key,
        stream,
        tuple(tensor.data_ptr() for tensor in tensors),
    )
    launch_plan = _LAUNCH_PLANS.get(launch_key)
    if launch_plan is None:
        artifact = _COMPILED.get(compile_key)
        if artifact is None:
            bound = tuple(
                _bind(tensor, tensor.shape[0], tensor.shape[1]) for tensor in tensors
            )
            artifact = tla.compile(
                kernel, *bound, *extra_args, options="--npu-arch 3510"
            )
            if len(_COMPILED) >= _MAX_PLANS:
                _COMPILED.clear()
            _COMPILED[compile_key] = artifact
        bound = tuple(
            _bind(tensor, tensor.shape[0], tensor.shape[1]) for tensor in tensors
        )
        block_num = _cores(device_index)
        launch_plan = {
            "artifact": artifact,
            "inputs": bound,
            "block_num": block_num,
            "launcher": _make_launcher(artifact, bound, block_num, stream),
        }
        if len(_LAUNCH_PLANS) >= _MAX_PLANS:
            _LAUNCH_PLANS.clear()
        _LAUNCH_PLANS[launch_key] = launch_plan

    launcher = launch_plan["launcher"]
    if launcher is not None:
        launcher()
    else:
        launch_plan["artifact"](
            *launch_plan["inputs"], block_num=launch_plan["block_num"], stream=stream
        )
    return launch_plan


def _launch_checked(kernel, tensors, extra_args, key, device_index):
    """Launch a fallback kernel and surface failures with input context."""
    try:
        launch_plan = _launch(kernel, tensors, extra_args, key, device_index)
    except Exception as exc:
        raise RuntimeError(
            "{} failed for launch key {!r}".format(kernel.__name__, key)
        ) from exc
    if not isinstance(launch_plan, dict) or launch_plan.get("artifact") is None:
        raise RuntimeError(
            "{} returned an invalid launch plan for key {!r}".format(
                kernel.__name__, key
            )
        )


def _remember_call_plan(call_key, rows, cols, device, device_index, empty):
    """Cache only shape metadata; never retain input tensors."""
    if call_key is None:
        return
    if len(_CALL_PLANS) >= _MAX_CALL_PLANS:
        _CALL_PLANS.clear()
        _CALL_OUTPUTS.clear()
    _CALL_PLANS[call_key] = {
        "rows": rows,
        "cols": cols,
        "device": device,
        "device_index": device_index,
        "empty": empty,
    }


def _launcher_from_plan(plan, stream):
    launcher = plan["launcher"]
    if launcher is not None:
        return launcher
    return lambda: plan["artifact"](
        *plan["inputs"],
        block_num=plan["block_num"],
        stream=stream,
    )


def _remember_call_launch(call_key, stream, out, plan):
    """Cache a launcher by input metadata, stream and output pointer."""
    if call_key is None:
        return
    if len(_CALL_OUTPUTS) >= _MAX_CALL_OUTPUTS:
        _CALL_OUTPUTS.clear()
    _CALL_OUTPUTS[(call_key, stream, out.data_ptr())] = _launcher_from_plan(
        plan, stream
    )


def _choose_flat_chunk(total_elems, cores):
    """Use 32 KiB transfers; measured best across small, tail and 3M-element cases."""
    return 8192


def _batch_rows(width, rows=None, cores=None, budget=STAGE_ELEMS):
    by_budget = max(1, min(MAX_BATCH_ROWS, budget // max(1, width)))
    if rows is None:
        return by_budget
    return max(1, min(by_budget, rows))


def cat(xs, dim=0):
    """``torch.cat(xs, dim)`` for 2-D fp32 tensors, ``dim in {0, 1}``."""
    if not xs:
        raise ValueError("xs must not be empty")
    if dim not in (0, 1):
        raise ValueError("dim must be 0 or 1")
    call_key = _call_key(xs, dim)
    call_plan = _CALL_PLANS.get(call_key)
    preallocated_out = None
    if call_plan is not None:
        device_index = call_plan["device_index"]
        stream = _current_stream(device_index)
        out = torch.empty(
            (call_plan["rows"], call_plan["cols"]),
            dtype=torch.float32,
            device=call_plan["device"],
        )
        if call_plan["empty"]:
            return out
        launch = _CALL_OUTPUTS.get((call_key, stream, out.data_ptr()))
        if launch is not None:
            launch()
            return out
        preallocated_out = out

    xs = [x if x.is_contiguous() else x.contiguous() for x in xs]
    for i, x in enumerate(xs):
        if x.dtype != torch.float32 or x.dim() != 2:
            raise ValueError("xs[{}] must be 2-D float32".format(i))
    device = xs[0].device
    if any(x.device != device for x in xs):
        raise ValueError("all tensors in xs must be on the same device")
    device_index = _device_index(device)
    stream = _current_stream(device_index)

    if dim == 0:
        n = xs[0].shape[1]
        if any(x.shape[1] != n for x in xs):
            raise ValueError("cat(dim=0) needs the same number of columns")
        rows = 0
        for x in xs:
            rows += x.shape[0]
        out = (
            preallocated_out
            if preallocated_out is not None
            else _acquire_output(rows, n, device, device_index, stream)
        )
        if rows == 0 or n == 0:
            _remember_call_plan(call_key, rows, n, device, device_index, True)
            return out
        totals = [x.numel() for x in xs]
        if len(xs) <= 3:
            dummy = next(x for x in xs if x.numel() != 0)
            safe_xs = [x if x.numel() != 0 else dummy for x in xs]
            padded = safe_xs + [dummy] * (3 - len(xs))
            total_args = totals + [0] * (3 - len(totals))
            cores = _cores(device_index)
            chunk = _choose_flat_chunk(sum(totals), cores)
            plan = _launch(
                cat_dim0_3_kernel,
                (padded[0], padded[1], padded[2], out),
                total_args + [chunk],
                ("cat0_3", tuple(total_args), chunk),
                device_index,
            )
            _remember_call_plan(call_key, rows, n, device, device_index, False)
            _remember_call_launch(call_key, stream, out, plan)
            return out

        dst_elem = 0
        for x, total in zip(xs, totals):
            if total:
                _launch_checked(
                    cat_dim0_kernel,
                    (x, out),
                    [dst_elem, total],
                    ("cat0", total, dst_elem),
                    device_index,
                )
            dst_elem += total
        return out

    m = xs[0].shape[0]
    if any(x.shape[0] != m for x in xs):
        raise ValueError("cat(dim=1) needs the same number of rows")
    n_total = 0
    for x in xs:
        n_total += x.shape[1]
    out = (
        preallocated_out
        if preallocated_out is not None
        else _acquire_output(m, n_total, device, device_index, stream)
    )
    if m == 0 or n_total == 0:
        _remember_call_plan(call_key, m, n_total, device, device_index, True)
        return out

    widths = [x.shape[1] for x in xs]
    columns = []
    col = 0
    for ns in widths:
        if ns > STAGE_ELEMS:
            raise ValueError(
                "cat(dim=1) supports row width <= {} elements, got {}".format(
                    STAGE_ELEMS, ns
                )
            )
        columns.append(col)
        col += ns
    cores = _cores(device_index)
    brs = [_batch_rows(ns, rows=m, cores=cores) for ns in widths]

    if len(xs) <= 3:
        dummy = next(x for x in xs if x.numel() != 0)
        safe_xs = [x if x.numel() != 0 else dummy for x in xs]
        padded = safe_xs + [dummy] * (3 - len(xs))
        width_args = widths + [0] * (3 - len(widths))
        column_args = columns + [n_total] * (3 - len(columns))
        br_args = brs + [1] * (3 - len(brs))
        buf_elems = max(1, max(br * ns for br, ns in zip(br_args, width_args)))
        plan = _launch(
            cat_dim1_3_kernel,
            (padded[0], padded[1], padded[2], out),
            [m, *width_args, n_total, *column_args, *br_args, buf_elems],
            (
                "cat1_3",
                m,
                tuple(width_args),
                n_total,
                tuple(column_args),
                tuple(br_args),
                buf_elems,
            ),
            device_index,
        )
        _remember_call_plan(call_key, m, n_total, device, device_index, False)
        _remember_call_launch(call_key, stream, out, plan)
        return out

    for x, ns, col, br in zip(xs, widths, columns, brs):
        if ns:
            _launch_checked(
                cat_dim1_kernel,
                (x, out),
                [m, ns, n_total, col, br],
                ("cat1", m, ns, n_total, col, br),
                device_index,
            )
    return out


def run(xs, dim=0):
    """Return ``torch.cat(xs, dim)`` for a list of 2-D FP32 tensors."""
    return cat(xs, dim)


__all__ = ["run", "cat", "reset_host_cache"]


# ======================================================================================
# self check
# ======================================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    torch.manual_seed(0)
    failures = 0

    cases0 = (
        ((37, 1024), (23, 1024)),
        ((37, 1001), (23, 1001)),
        ((1, 3), (2, 3)),
        ((0, 8), (5, 8)),
        ((13, 4097), (7, 4097)),
    )
    for shapes in cases0:
        xs = [torch.randn(*s, device="npu:{}".format(args.device)) for s in shapes]
        out = run(xs, 0)
        ref = torch.cat(xs, 0)
        passed = torch.equal(out, ref)
        failures += not passed
        print("cat0", shapes, "passed={}".format(passed))

    cases1 = (
        ((37, 512), (37, 512)),
        ((37, 250), (37, 1001)),
        ((5, 1), (5, 63)),
        ((9, 999), (9, 3), (9, 17)),
        ((13, 4096), (13, 1)),
    )
    for shapes in cases1:
        xs = [torch.randn(*s, device="npu:{}".format(args.device)) for s in shapes]
        out = run(xs, 1)
        ref = torch.cat(xs, 1)
        passed = torch.equal(out, ref)
        failures += not passed
        print("cat1", shapes, "passed={}".format(passed))

    if failures:
        raise SystemExit(1)
