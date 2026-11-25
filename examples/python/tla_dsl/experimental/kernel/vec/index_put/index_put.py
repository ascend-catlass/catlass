# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""1-D IndexPut / Scatter add atom, fp32.

Covers KernelBench level1 ``IndexPut`` (and the duplicate-index half of
``Scatter``)::

    out = x.clone()
    out[index[i]] (accumulate ? += : =) values[i]

Performance design points (why this is fast):

1. **4-byte atomics instead of 32-byte sectors.**  ``tla.copy`` with
   ``AtomicMode.ADD`` lowers to ``SetAtomicAdd + DataCopyPad + SetAtomicNone``;
   ``DataCopyPad`` moves exactly the tile's bytes.  A one-element tile therefore
   adds a single f32 at ``out[index[i]]``, so an index costs 4 B of atomic
   traffic instead of a whole 32 B sector.
2. **Vector-produced tiles.**  Each element sits at lane 0 of its own 32-byte UB
   slot, filled by one ``tla.scatter`` on the vector pipe.  A vector-produced
   tile is ordered against the MTE3 atomic read by the 2.2.1 flags; scalar UB
   stores are not (they lose/duplicate updates).
3. **Stage is never reused.**  Every element owns a distinct 32-byte slot in the
   per-block ``stage`` buffer, so no clear can race with an in-flight atomic
   read, even when blocks run in several waves.
4. **Contiguous slices per core, no index DMA.**  Each block walks
   ``[lo, lo + CHUNK)``; the index is read as a scalar and values arrive with one
   64-element DMA per batch.  Padding to a multiple of 64 is zero-filled, and
   the ``gidx < m`` mask turns the tail adds into harmless ``+0``.
5. **Mutex-pipelined variant for large M.**  ``scatter_add_atomic_pipe_kernel``
   double buffers the value batch and hands buffer ownership over with mutexes,
   so MTE2 (next values) and VECTOR (next scatter) overlap with the in-flight
   atomic DMAs instead of draining them every 64 elements.
6. **Thin host path.**  Input/output descriptors are cached by
   ``(data_ptr, numel, dtype)``, padding uses a cached zero buffer + one
   ``copy_`` (which also performs the dtype cast), so a call issues only
   ``x.clone()`` plus the scatter kernel.

Device paths share one public ``run`` entry:

* ``scatter_overwrite_kernel`` (accumulate=False): every AI vector core walks a
  strided slice of ``i`` inside a SIMT region and stores ``values[i]`` at
  ``out[index[i]]``.  Duplicate indices are last-writer-wins, which is the
  documented torch semantics for ``accumulate=False``.
* ``scatter_add_atomic_kernel`` / ``scatter_add_atomic_pipe_kernel``
  (accumulate=True): see points 1-5 above.  Duplicate indices accumulate
  correctly because the hardware add is atomic.

The original implementation walked all ``M`` indices in a single block.  Both
paths above remove that serial bottleneck while keeping fp32 accumulation.
"""

import os
import sys

import catlass.tla as tla

_VEC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _VEC_DIR not in sys.path:
    sys.path.append(_VEC_DIR)

_SIMT_THREADS = (64, 4, 2)
_BATCH = 64  # one full f32 vector register of atomic tiles
_LANE = 8  # 32 B UB slot per atomic tile = 8 x f32
_INT32_MAX_N = 1 << 31
_SMALL_M = 512
_DESC_CACHE = {}


@tla.kernel
def scatter_overwrite_kernel(
    gm_val: tla.Tensor,  # [M] fp32
    gm_idx: tla.Tensor,  # [M] int32
    gm_out: tla.Tensor,  # [N] fp32, pre-filled with x
) -> None:
    """out[index[i]] = values[i] over the whole launch (duplicates undefined).

    One SIMT thread per index: ``gid``/``stride`` spread ``M`` across every
    block's threads, so the write of ``values[i]`` is a single scalar GM store
    with no UB staging and no cross-core sync.  Last write wins on duplicates.
    """
    block_id = tla.arch.block_idx()
    nblocks = tla.arch.block_num()
    m = gm_val.origin_shape[0]

    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=_SIMT_THREADS):
            tx, ty, tz = tla.arch.thread_idx()
            dx, dy, dz = tla.arch.thread_block_dim()
            tid = tx + ty * dx + tz * dx * dy
            tdim = dx * dy * dz
            gid = block_id * tdim + tid
            stride = nblocks * tdim
            for i in tla.range(gid, m, stride):
                gm_out[gm_idx[i]] = gm_val[i]
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def scatter_add_atomic_kernel(
    gm_val: tla.Tensor,  # [M] fp32, padded to a multiple of 64
    gm_idx: tla.Tensor,  # [M] int32, padded with zeros
    gm_out: tla.Tensor,  # [N] fp32, pre-filled with x
    CHUNK: tla.Constexpr[int],
    STAGE: tla.Constexpr[int],
) -> None:
    """out[index[i]] += values[i] via 4-byte UB->GM atomic adds, all cores.

    Each block owns ``CHUNK`` elements.  A batch of 64 values is scattered into
    64 separate 32-byte UB slots, one value at lane 0 of each slot, then copied
    with ``AtomicMode.ADD``: ``DataCopyPad`` moves only ``blockLen = 4`` bytes, so
    the hardware adds the single f32 at ``out[index[i]]``.  The stage buffer is
    never reused inside a block, so no clear can race with an in-flight atomic
    read.
    """
    m = gm_val.origin_shape[0]
    block_id = tla.arch.block_idx()
    lo = block_id * CHUNK
    hi = lo + CHUNK

    mutex_stage = tla.mutex("ub_stage", id=0)

    val_ptr = tla.allocate(_BATCH, tla.Float32, tla.AddressSpace.ub, 32)
    stage_ptr = tla.allocate(STAGE, tla.Float32, tla.AddressSpace.ub, 32)
    val_ub = tla.make_tensor(
        val_ptr,
        tla.make_layout(
            tla.make_shape(_BATCH), tla.make_stride(1), layoutTag=tla.arch.RowMajor
        ),
    )
    stage = tla.make_tensor(
        stage_ptr,
        tla.make_layout(
            tla.make_shape(STAGE), tla.make_stride(1), layoutTag=tla.arch.RowMajor
        ),
    )

    with tla.vector():
        if lo < m:
            # Zero every 32B slot once: the atomic copy moves the whole tile,
            # and the 28 B beyond lane 0 must add 0 to the neighbours.  The
            # first VECTOR unlock below also publishes this clear to MTE3.
            with tla.vec.func(mode="simd"):
                for z in tla.range(STAGE // 64):
                    tla.tile_view(stage, tla.make_shape(64), tla.make_coord(z)).store(
                        tla.full(0.0, dtype=tla.Float32)
                    )

            for bi in tla.range(CHUNK // _BATCH):
                base = lo + bi * _BATCH
                # One 64-element value DMA; the index is only read as a scalar.
                val_gm = tla.make_tensor(
                    gm_val.ptr + base,
                    tla.make_layout(
                        tla.make_shape(_BATCH),
                        tla.make_stride(1),
                        layoutTag=tla.arch.RowMajor,
                    ),
                )
                tla.copy(val_ub, val_gm)
                tla.pipe_barrier(tla.pipes.ALL)

                # Vector-produced atomic tiles: lane j -> slot j, lane 0.  The
                # scatter's mask drops the padded tail, leaving those slots 0.
                batch_stage = tla.tile_view(
                    stage, tla.make_shape(_BATCH * _LANE), tla.make_coord(bi)
                )
                mutex_stage.lock(pipe=tla.arch.VECTOR)
                with tla.vec.func(mode="simd"):
                    vv = tla.tile_view(
                        val_ub, tla.make_shape(_BATCH), tla.make_coord(0)
                    ).load()
                    lane = tla.arange(base=0, dtype=tla.Int32)
                    offs = tla.mul(lane, tla.full(_LANE, dtype=tla.Int32))
                    gidx = tla.add(tla.full(base, dtype=tla.Int32), lane)
                    msk = tla.cmp(gidx, tla.full(m, dtype=tla.Int32), "lt")
                    tla.scatter(batch_stage, vv, offs.bitcast(tla.UInt32), mask=msk)
                    tla.local_mem_bar(
                        tla.params.MemType.VEC_ALL, tla.params.MemType.VEC_ALL
                    )
                mutex_stage.unlock(pipe=tla.arch.VECTOR)

                # One 4-byte atomic add per element, at out[index[i]].  The
                # MTE3 barrier drains this batch before the next one reuses
                # val_ub (the stage slots themselves are never reused).
                mutex_stage.lock(pipe=tla.arch.MTE3)
                for j in tla.range(_BATCH):
                    if base + j < hi:  # noqa: SIM102 - keep the two scalar guards separate for tracing
                        if base + j < m:
                            k = gm_idx[base + j].to(tla.Int32)
                            tla.copy(
                                tla.tile_view(
                                    gm_out, tla.make_shape(1), tla.make_coord(k)
                                ),
                                tla.tile_view(
                                    stage,
                                    tla.make_shape(1),
                                    tla.make_coord(bi * _BATCH * _LANE + j * _LANE),
                                ),
                                tla.params.CopyUbToGmParams(
                                    atomic_mode=tla.params.AtomicMode.ADD
                                ),
                            )
                mutex_stage.unlock(pipe=tla.arch.MTE3)
                tla.pipe_barrier(tla.pipes.MTE3)


@tla.kernel
def scatter_add_atomic_pipe_kernel(
    gm_val: tla.Tensor,  # [M] fp32, padded to a multiple of 128
    gm_idx: tla.Tensor,  # [M] int32/int64, padded with zeros
    gm_out: tla.Tensor,  # [N] fp32, pre-filled with x
    CHUNK: tla.Constexpr[int],
    STAGE: tla.Constexpr[int],
) -> None:
    """4-byte atomic adds with a mutex-guarded two-buffer pipeline.

    ``scatter_add_atomic_kernel`` re-synchronises every 64 elements with
    ``pipe_barrier(ALL)``; that barrier also drains the in-flight atomic DMAs,
    so MTE3 cannot overlap with the next value DMA / vector scatter.

    Here the value batch is double buffered and buffer ownership is handed over
    with per-buffer mutexes, keeping all three pipes busy:

    * ``ub_val0/1``: MTE2 fills, VECTOR consumes.  The next MTE2 fill of the
      same buffer waits until VECTOR releases it, so the value DMA can run
      ahead of the atomic writes.
    * ``ub_stage0/1``: VECTOR scatters the atomic tiles, MTE3 consumes.  The
      next VECTOR scatter into the same buffer waits until MTE3 has issued the
      previous batch of adds.

    Only used when ``CHUNK`` is a multiple of ``2 * _BATCH``, so the loop can be
    unrolled into buffer 0 / buffer 1 pairs without a runtime buffer index.
    """
    m = gm_val.origin_shape[0]
    lo = tla.arch.block_idx() * CHUNK
    hi = lo + CHUNK

    mutex_val0 = tla.mutex("ub_val0", id=0)
    mutex_val1 = tla.mutex("ub_val1", id=1)
    mutex_stage0 = tla.mutex("ub_stage0", id=2)
    mutex_stage1 = tla.mutex("ub_stage1", id=3)

    val0_ptr = tla.allocate(_BATCH, tla.Float32, tla.AddressSpace.ub, 32)
    val1_ptr = tla.allocate(_BATCH, tla.Float32, tla.AddressSpace.ub, 32)
    stage_ptr = tla.allocate(STAGE, tla.Float32, tla.AddressSpace.ub, 32)
    val0 = tla.make_tensor(
        val0_ptr,
        tla.make_layout(
            tla.make_shape(_BATCH), tla.make_stride(1), layoutTag=tla.arch.RowMajor
        ),
    )
    val1 = tla.make_tensor(
        val1_ptr,
        tla.make_layout(
            tla.make_shape(_BATCH), tla.make_stride(1), layoutTag=tla.arch.RowMajor
        ),
    )
    stage = tla.make_tensor(
        stage_ptr,
        tla.make_layout(
            tla.make_shape(STAGE), tla.make_stride(1), layoutTag=tla.arch.RowMajor
        ),
    )

    with tla.vector():
        if lo < m:
            # Lane 0 of every 32B slot must be zero outside the valid values:
            # the tile is copied whole, so the padding lanes are added as 0.
            with tla.vec.func(mode="simd"):
                for z in tla.range(STAGE // 64):
                    tla.tile_view(stage, tla.make_shape(64), tla.make_coord(z)).store(
                        tla.full(0.0, dtype=tla.Float32)
                    )
            for b2 in tla.range(CHUNK // (2 * _BATCH)):
                bi = 2 * b2
                base = lo + bi * _BATCH

                # ---- batch 2*b2: value buffer 0, stage buffer 0 ----
                mutex_val0.lock(pipe=tla.arch.MTE2)
                tla.copy(
                    val0,
                    tla.make_tensor(
                        gm_val.ptr + base,
                        tla.make_layout(
                            tla.make_shape(_BATCH),
                            tla.make_stride(1),
                            layoutTag=tla.arch.RowMajor,
                        ),
                    ),
                )
                mutex_val0.unlock(pipe=tla.arch.MTE2)

                mutex_val0.lock(pipe=tla.arch.VECTOR)
                mutex_stage0.lock(pipe=tla.arch.VECTOR)
                with tla.vec.func(mode="simd"):
                    vv = tla.tile_view(
                        val0, tla.make_shape(_BATCH), tla.make_coord(0)
                    ).load()
                    lane = tla.arange(base=0, dtype=tla.Int32)
                    offs = tla.mul(lane, tla.full(_LANE, dtype=tla.Int32))
                    gidx = tla.add(tla.full(base, dtype=tla.Int32), lane)
                    msk = tla.cmp(gidx, tla.full(m, dtype=tla.Int32), "lt")
                    tla.scatter(
                        tla.tile_view(
                            stage, tla.make_shape(_BATCH * _LANE), tla.make_coord(bi)
                        ),
                        vv,
                        offs.bitcast(tla.UInt32),
                        mask=msk,
                    )
                    tla.local_mem_bar(
                        tla.params.MemType.VEC_ALL, tla.params.MemType.VEC_ALL
                    )
                mutex_stage0.unlock(pipe=tla.arch.VECTOR)
                mutex_val0.unlock(pipe=tla.arch.VECTOR)

                mutex_stage0.lock(pipe=tla.arch.MTE3)
                for j in tla.range(_BATCH):
                    if base + j < hi:  # noqa: SIM102 - keep the two scalar guards separate for tracing
                        if base + j < m:
                            k = gm_idx[base + j].to(tla.Int32)
                            tla.copy(
                                tla.tile_view(
                                    gm_out, tla.make_shape(1), tla.make_coord(k)
                                ),
                                tla.tile_view(
                                    stage,
                                    tla.make_shape(1),
                                    tla.make_coord(bi * _BATCH * _LANE + j * _LANE),
                                ),
                                tla.params.CopyUbToGmParams(
                                    atomic_mode=tla.params.AtomicMode.ADD
                                ),
                            )
                mutex_stage0.unlock(pipe=tla.arch.MTE3)

                # ---- batch 2*b2+1: value buffer 1, stage buffer 1 ----
                base1 = base + _BATCH
                mutex_val1.lock(pipe=tla.arch.MTE2)
                tla.copy(
                    val1,
                    tla.make_tensor(
                        gm_val.ptr + base1,
                        tla.make_layout(
                            tla.make_shape(_BATCH),
                            tla.make_stride(1),
                            layoutTag=tla.arch.RowMajor,
                        ),
                    ),
                )
                mutex_val1.unlock(pipe=tla.arch.MTE2)

                mutex_val1.lock(pipe=tla.arch.VECTOR)
                mutex_stage1.lock(pipe=tla.arch.VECTOR)
                with tla.vec.func(mode="simd"):
                    vv1 = tla.tile_view(
                        val1, tla.make_shape(_BATCH), tla.make_coord(0)
                    ).load()
                    lane1 = tla.arange(base=0, dtype=tla.Int32)
                    offs1 = tla.mul(lane1, tla.full(_LANE, dtype=tla.Int32))
                    gidx1 = tla.add(tla.full(base1, dtype=tla.Int32), lane1)
                    msk1 = tla.cmp(gidx1, tla.full(m, dtype=tla.Int32), "lt")
                    tla.scatter(
                        tla.tile_view(
                            stage,
                            tla.make_shape(_BATCH * _LANE),
                            tla.make_coord(bi + 1),
                        ),
                        vv1,
                        offs1.bitcast(tla.UInt32),
                        mask=msk1,
                    )
                    tla.local_mem_bar(
                        tla.params.MemType.VEC_ALL, tla.params.MemType.VEC_ALL
                    )
                mutex_stage1.unlock(pipe=tla.arch.VECTOR)
                mutex_val1.unlock(pipe=tla.arch.VECTOR)

                mutex_stage1.lock(pipe=tla.arch.MTE3)
                for j2 in tla.range(_BATCH):
                    if base1 + j2 < hi:  # noqa: SIM102 - keep the two scalar guards separate for tracing
                        if base1 + j2 < m:
                            k2 = gm_idx[base1 + j2].to(tla.Int32)
                            tla.copy(
                                tla.tile_view(
                                    gm_out, tla.make_shape(1), tla.make_coord(k2)
                                ),
                                tla.tile_view(
                                    stage,
                                    tla.make_shape(1),
                                    tla.make_coord(
                                        (bi + 1) * _BATCH * _LANE + j2 * _LANE
                                    ),
                                ),
                                tla.params.CopyUbToGmParams(
                                    atomic_mode=tla.params.AtomicMode.ADD
                                ),
                            )
                mutex_stage1.unlock(pipe=tla.arch.MTE3)
            tla.pipe_barrier(tla.pipes.MTE3)


@tla.kernel
def scatter_add_small_kernel(
    gm_val: tla.Tensor,  # [M] fp32
    gm_idx: tla.Tensor,  # [M] int64
    gm_out: tla.Tensor,  # [N] fp32
) -> None:
    """Serial fallback for N >= 2**31 (index does not fit int32)."""
    m = gm_val.origin_shape[0]

    with tla.vector():
        for i in tla.range(m):
            gm_out[gm_idx[i]] = gm_out[gm_idx[i]] + gm_val[i]
        tla.pipe_barrier(tla.pipes.ALL)


# ======================================================================================
# HOST side
# ======================================================================================

_PLANS = {}
_CORES = {}
_DTYPE = {
    "torch.float32": tla.Float32,
    "torch.int64": tla.Int64,
    "torch.int32": tla.Int32,
}


def _vector_cores(index):
    import torch

    cores = _CORES.get(index)
    if cores is None:
        cores = int(torch.npu.get_device_properties(index).vector_core_num)
        _CORES[index] = cores
    return cores


def _bind(tensor):
    ptr = tla.make_ptr(
        _DTYPE[str(tensor.dtype)], int(tensor.data_ptr()), tla.AddressSpace.gm
    )
    layout = tla.make_layout(
        tla.make_shape(tensor.numel()),
        tla.make_stride(1),
        layoutTag=tla.arch.RowMajor,
    )
    return tla.make_tensor(ptr, layout)


def _desc(tensor, cache=True):
    """Lightweight pointer descriptor; stable input buffers are reused."""
    if not cache:
        return _bind(tensor)
    key = (tensor.data_ptr(), tensor.numel(), tensor.dtype)
    desc = _DESC_CACHE.get(key)
    if desc is None:
        if len(_DESC_CACHE) > 256:
            _DESC_CACHE.clear()
        desc = _bind(tensor)
        _DESC_CACHE[key] = desc
    return desc


_ATOMIC_CHUNK_CAP = 4096


def _atomic_plan(mp, cores):
    """Blocks/chunk/stage sized so the per-block stage buffer stays under 128 KiB."""
    blocks = max(cores, (mp + _ATOMIC_CHUNK_CAP - 1) // _ATOMIC_CHUNK_CAP)
    chunk = (mp + blocks - 1) // blocks
    chunk = ((chunk + _BATCH - 1) // _BATCH) * _BATCH
    stage = chunk * _LANE
    return blocks, chunk, stage


_PAD_CACHE = {}


def _pad_to(tensor, m, mp, dtype):
    """Cached zero-padded buffer; only the first ``m`` slots are rewritten.

    Keyed by ``(mp, m, dtype)`` so a reused buffer's tail is always the zeros it
    was created with -- the kernel masks by the padded length, so a stale tail
    would be added into ``out``.  The per-call cost is one ``copy_`` (which also
    does the int64 -> int32 cast when asked), replacing ``zeros`` + ``cat``.
    """
    import torch

    key = (mp, m, dtype, str(tensor.device))
    buf = _PAD_CACHE.get(key)
    if buf is None:
        if len(_PAD_CACHE) > 32:
            _PAD_CACHE.clear()
        buf = torch.zeros(mp, dtype=dtype, device=tensor.device)
        _PAD_CACHE[key] = buf
    buf[:m].copy_(tensor)
    return buf


def _get_plan(kind, m, n, args, cores):
    key = (kind, m, n)
    plan = _PLANS.get(key)
    if plan is None:
        artifact = tla.compile(*args, options="--npu-arch 3510")
        plan = {"artifact": artifact, "cores": cores}
        _PLANS[key] = plan
    return plan


def run(x, index, values, accumulate=True):
    """``out = x``; ``out[index[i]] += values[i]`` (or ``=`` when accumulate is False)."""
    import torch

    if x.dtype != torch.float32 or values.dtype != torch.float32:
        raise ValueError("x and values must be float32")
    if x.dim() != 1:
        raise ValueError("only 1-D x is supported")
    if values.shape != index.shape:
        raise ValueError("values and index must have the same shape")
    if index.dtype not in (torch.int64, torch.int32):
        raise ValueError("index must be int32/int64")

    device_index = x.device.index if x.device.index is not None else 0
    val = values.contiguous()
    out = x.contiguous().clone()
    m = index.numel()
    if m == 0:
        return out
    cores = _vector_cores(device_index)

    if not accumulate:
        idx = index.contiguous()
        plan = _get_plan(
            ("ow", idx.dtype),
            m,
            out.numel(),
            (scatter_overwrite_kernel, _desc(val), _desc(idx), _desc(out)),
            cores,
        )
        plan["artifact"](_desc(val), _desc(idx), _desc(out), block_num=cores)
        return out

    if out.numel() >= _INT32_MAX_N:
        idx64 = index.to(torch.int64).contiguous()
        plan = _get_plan(
            "add_small",
            m,
            out.numel(),
            (scatter_add_small_kernel, _desc(val), _desc(idx64), _desc(out)),
            cores,
        )
        plan["artifact"](_desc(val), _desc(idx64), _desc(out), block_num=1)
        return out

    idx = index.contiguous()
    if m <= _SMALL_M:
        plan = _get_plan(
            ("add_small32", idx.dtype),
            m,
            out.numel(),
            (scatter_add_small_kernel, _desc(val), _desc(idx), _desc(out)),
            cores,
        )
        plan["artifact"](_desc(val), _desc(idx), _desc(out), block_num=1)
        return out
    pad = (-m) % _BATCH
    if pad:
        mp = m + pad
        idx = _pad_to(idx, m, mp, idx.dtype)
        val = _pad_to(val, m, mp, torch.float32)
    else:
        mp = m
    blocks, chunk, stage = _atomic_plan(mp, cores)
    kernel = (
        scatter_add_atomic_pipe_kernel
        if chunk % (2 * _BATCH) == 0
        else scatter_add_atomic_kernel
    )
    plan = _get_plan(
        ("add_atomic", chunk, idx.dtype, kernel.__name__),
        mp,
        out.numel(),
        (kernel, _desc(val), _desc(idx), _desc(out), chunk, stage),
        blocks,
    )
    plan["artifact"](_desc(val), _desc(idx), _desc(out), block_num=blocks)
    return out
