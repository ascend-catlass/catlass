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

# One thread per row: each thread performs K scalar loads from that row.
ROWS_PER_BLOCK = 128


@tla.kernel
def gather_kernel(
    x: tla.Tensor,  # [M, V] fp32 RowMajor
    index: tla.Tensor,  # [M, K] int32 column index, negative for unused slots
    out: tla.Tensor,  # [M, K] fp32, zeroed by host
    K: tla.Constexpr[int],
) -> None:
    c0 = 0
    c1 = 1

    m = x.origin_shape[0]
    v = x.origin_shape[1]

    with tla.vector():
        with tla.vec.func(mode="simt", thread_block_dim=ROWS_PER_BLOCK):
            tid, _, _ = tla.arch.thread_idx()
            start = tla.arch.block_idx() * ROWS_PER_BLOCK + tid
            step = tla.arch.block_num() * ROWS_PER_BLOCK

            # Complexity: O(M * K) scalar loads overall; every row is visited by
            # exactly one thread, so no reduction or cross-thread sync is needed.
            for row in tla.range(start, m, step):
                for k in tla.range(c0, K, c1):
                    loc = index[row, k]
                    # Negative slots are unused and the host zero-fills out, so
                    # there is nothing to write for them; the upper bound keeps
                    # the scalar load inside the row.
                    if loc >= c0:
                        if loc < v:
                            out[row, k] = x[row, loc]

        tla.pipe_barrier(tla.pipes.ALL)


# ======================================================================================
# HOST side: index preparation, compile and launch.
# ======================================================================================

_COMPILED = {}


def _vector_cores(index):
    import torch

    return int(torch.npu.get_device_properties(index).vector_core_num)


def _tla_tensor(buffer):
    from catlass.tla.runtime import from_dlpack

    # The SIMT body takes statically shaped buffers, so one artifact is compiled
    # per (M, V, K) tuple.
    return from_dlpack(buffer, layout_tag=tla.arch.RowMajor)


def run(x, index):
    """Return ``out`` ``[M, K]``: ``x[i, index[i, k]]``, 0 where the index is negative."""
    import torch

    m, v = x.shape
    k = index.shape[1]
    device = x.device

    # Unused slots stay 0, so the kernel has nothing to write for them.
    out = torch.zeros((m, k), dtype=torch.float32, device=device)
    if m == 0:
        return out

    idx = index.to(torch.int32).contiguous()

    device_index = int(str(device).split(":")[-1]) if ":" in str(device) else 0
    vector_cores = _vector_cores(device_index)

    x_t = _tla_tensor(x.contiguous())
    idx_t = _tla_tensor(idx)
    out_t = _tla_tensor(out)

    key = (m, v, k)
    artifact = _COMPILED.get(key)
    if artifact is None:
        artifact = tla.compile(
            gather_kernel,
            x_t,
            idx_t,
            out_t,
            k,
            options="--npu-arch 3510",
        )
        _COMPILED[key] = artifact

    artifact(
        x_t,
        idx_t,
        out_t,
        block_num=vector_cores,
    )
    return out
