# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Torch golden for Flash MLA decoding.

A straightforward chunked implementation of the same contract: gather the valid
KV pages through ``block_table``, run the QK product, apply the mask, take the
softmax in fp32 and reduce PV.  It runs on whatever device the inputs carry, so
the example can check the kernel against it on the device (fast) or on the CPU
(a fully independent path, for cross-checking).
"""

from __future__ import annotations

import torch

from .constants import (
    CACHE_LAYOUTS,
    HEAD_DIM_QK,
    HEAD_DIM_V,
    KV_HEAD_NUM,
    MASK_NONE,
    MASK_RIGHT_DOWN_CAUSAL,
    PAGE_SIZE,
    QUERY_CHUNK,
)


def _check_cache_shape(cache_shape, cache_layout):
    if cache_layout == "PA_BBND":
        expected = (PAGE_SIZE, KV_HEAD_NUM, HEAD_DIM_QK)
        if tuple(cache_shape[1:]) != expected:
            raise ValueError(
                "PA_BBND cache must have shape "
                "[num_blocks, 128, 1, 576], got {}".format(tuple(cache_shape))
            )
    elif cache_layout == "PA_NZ":
        expected = (KV_HEAD_NUM, HEAD_DIM_QK // 16, PAGE_SIZE, 16)
        if tuple(cache_shape[1:]) != expected:
            raise ValueError(
                "PA_NZ cache must have shape "
                "[num_blocks, 1, 36, 128, 16], got {}".format(tuple(cache_shape))
            )
    else:
        raise ValueError("cache_layout must be one of {}".format(CACHE_LAYOUTS))


def _gather_kv(k_cache, block_table_row, kv_len, cache_layout):
    """Collect the first kv_len tokens following the paged block table."""
    num_pages = (kv_len + PAGE_SIZE - 1) // PAGE_SIZE
    physical_pages = block_table_row[:num_pages].long()
    if cache_layout == "PA_BBND":
        blocks = k_cache.index_select(0, physical_pages).squeeze(2)
    else:
        blocks = k_cache.index_select(0, physical_pages).squeeze(1)
        blocks = blocks.permute(0, 2, 1, 3).reshape(num_pages, PAGE_SIZE, HEAD_DIM_QK)
    return blocks.reshape(-1, HEAD_DIM_QK)[:kv_len]


def flash_mla_with_kvcache_golden(
    q,
    k_cache,
    block_table,
    cu_seqlens_q,
    cache_seqlens,
    softmax_scale=None,
    mask_mode=MASK_NONE,
    cache_layout="PA_BBND",
):
    if q.ndim != 3:
        raise ValueError("q must have shape [total_q, num_heads, 576]")
    if q.shape[2] != HEAD_DIM_QK:
        raise ValueError("q head dimension must be 576")
    if q.shape[1] not in (64, 96):
        raise ValueError("q num_heads must be 64 or 96")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("q dtype must be float16 or bfloat16")
    if k_cache.dtype != q.dtype:
        raise ValueError("k_cache dtype must match q dtype")
    if block_table.ndim != 2:
        raise ValueError("block_table must be two-dimensional")
    batch = block_table.shape[0]
    if cu_seqlens_q.shape != (batch + 1,):
        raise ValueError("cu_seqlens_q must contain batch + 1 entries")
    if cache_seqlens.shape != (batch,):
        raise ValueError("cache_seqlens must contain one entry per request")
    _check_cache_shape(k_cache.shape, cache_layout)
    if mask_mode not in (MASK_NONE, MASK_RIGHT_DOWN_CAUSAL):
        raise ValueError("mask_mode must be 0 or 3")
    if softmax_scale is None:
        softmax_scale = 1.0
    if int(cu_seqlens_q[0].item()) != 0:
        raise ValueError("cu_seqlens_q must start at zero")
    if int(cu_seqlens_q[-1].item()) != q.shape[0]:
        raise ValueError("cu_seqlens_q must end at the total query length")
    if torch.any(cu_seqlens_q[1:] < cu_seqlens_q[:-1]):
        raise ValueError("cu_seqlens_q must be non-decreasing")

    total_q, num_heads, _ = q.shape
    output = torch.zeros(
        (total_q, num_heads, HEAD_DIM_V), dtype=q.dtype, device=q.device
    )
    lse = torch.full(
        (total_q, num_heads), float("-inf"), dtype=torch.float32, device=q.device
    )

    for batch_idx in range(batch):
        q_start = int(cu_seqlens_q[batch_idx].item())
        q_end = int(cu_seqlens_q[batch_idx + 1].item())
        q_len = q_end - q_start
        kv_len = int(cache_seqlens[batch_idx].item())
        if q_len <= 0 or kv_len <= 0:
            continue
        if mask_mode == MASK_RIGHT_DOWN_CAUSAL and kv_len < q_len:
            raise ValueError(
                "right-down causal mask requires cache length >= query length"
            )

        kv = _gather_kv(k_cache, block_table[batch_idx], kv_len, cache_layout).float()
        keys = kv
        values = kv[:, :HEAD_DIM_V]
        queries = q[q_start:q_end].float()

        for chunk_start in range(0, q_len, QUERY_CHUNK):
            chunk_end = min(chunk_start + QUERY_CHUNK, q_len)
            scores = torch.matmul(queries[chunk_start:chunk_end], keys.transpose(0, 1))
            scores = scores * softmax_scale
            if mask_mode == MASK_RIGHT_DOWN_CAUSAL:
                query_positions = (kv_len - q_len) + torch.arange(
                    chunk_start, chunk_end, device=q.device
                )
                key_positions = torch.arange(kv_len, device=q.device)
                visible = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
                scores = scores.masked_fill(~visible.unsqueeze(1), float("-inf"))

            row_max = scores.amax(dim=-1)
            finite = torch.isfinite(row_max)
            safe_max = torch.where(finite, row_max, torch.zeros_like(row_max))
            weights = torch.exp(scores - safe_max.unsqueeze(-1))
            row_sum = weights.sum(dim=-1)
            valid = finite & (row_sum > 0)
            lse_chunk = torch.where(
                valid,
                safe_max + torch.log(row_sum),
                torch.full_like(row_sum, float("-inf")),
            )
            weights = torch.where(
                valid.unsqueeze(-1),
                weights / row_sum.clamp_min(1).unsqueeze(-1),
                torch.zeros_like(weights),
            )
            out_chunk = torch.matmul(weights, values)
            output[q_start + chunk_start : q_start + chunk_end] = out_chunk.to(q.dtype)
            lse[q_start + chunk_start : q_start + chunk_end] = lse_chunk

    # Official layout_out=NTD: (num_heads, total_q, head_dim_v).
    output = output.transpose(0, 1).contiguous()
    lse = lse.transpose(0, 1).contiguous()
    return output, lse
