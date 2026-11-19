# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Flash MLA decoding with a paged KV cache: end-to-end example.

Multi-head latent attention decodes against a paged cache that stores 576
latent features per token; the first 512 of them are the absorbed values, so the
same page feeds both the QK product and the PV product:

.. math::

    S_{h,q,k} = \\frac{Q_{h,q} \\cdot K_k}{\\sqrt{576}}, \\qquad
    O_{h,q} = \\sum_k \\operatorname{softmax}(S)_{h,q,k} \\cdot V_k

``cache_seqlens`` makes the cache per-request variable length, and
``mask_mode=3`` applies the right-down causal mask over those lengths. Only the
query side is variable length (``cu_seqlens_q`` batches the requests), which is
the decode shape: ``total_q`` query rows in total, one or a few per request.

The operator is a single mixed Cube/Vector kernel with a one-tile QK lookahead;
when neither the query rows nor the heads fill the device, the host splits the
query rows over head groups and, for short inputs, the KV range over chunks
(``kvsplit``) and reduces the f32 partials in a second kernel.

Supported product: Ascend 950PR/950DT.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys

_DSL_EXAMPLE_PATH = str(
    next(p for p in Path(__file__).resolve().parents if (p / "helpers").is_dir())
)
if _DSL_EXAMPLE_PATH not in sys.path:
    sys.path.append(_DSL_EXAMPLE_PATH)

_REPO_PATH = str(Path(_DSL_EXAMPLE_PATH).parents[2])
if _REPO_PATH not in sys.path:
    sys.path.append(_REPO_PATH)

import catlass.tla as tla
import torch
import torch_npu  # noqa: F401  -- registers torch.npu
from catlass.tla.runtime import from_dlpack

from examples.python.tla_dsl.tla.kernel.attention.flash_mla_with_kvcache.constants import (
    CACHE_LAYOUT_BBND,
    CACHE_LAYOUT_NZ,
    CACHE_LAYOUTS,
    DTYPE_NAMES,
    HEAD_CHOICES,
    HEAD_DIM_QK,
    HEAD_DIM_QK as D_QK,
    HEAD_DIM_V,
    HEAD_DIM_V as D_V,
    KV_BLOCK,
    KV_HEAD_NUM,
    MASK_MODES,
    MASK_NONE,
    MASK_RIGHT_DOWN_CAUSAL,
    NPU_ARCH,
    PAGE_SIZE,
    Q_BLOCK,
)
from examples.python.tla_dsl.tla.kernel.attention.flash_mla_with_kvcache.golden import (
    flash_mla_with_kvcache_golden,
)
from examples.python.tla_dsl.tla.kernel.attention.flash_mla_with_kvcache.kernel import (
    flash_mla_kvsplit_reduce_kernel,
    flash_mla_with_kvcache_kernel,
)

_EXECUTOR_CACHE = {}
# Inputs are usually re-used across decode steps/benchmark trials. Wrapping a
# tensor through DLPack dominates the host path, so keep the wrapper keyed by
# the exact tensor object and its storage pointer.
_TENSOR_CACHE = {}
_TENSOR_CACHE_LIMIT = 32
_PLAN_CACHE = {}
_PLAN_CACHE_LIMIT = 32


def _tla_tensor(value):
    return from_dlpack(
        value, layout_tag=arch_row_major(), origin_shape=(value.numel(),)
    ).mark_layout_dynamic()


def _tla_tensor_cached(value):
    key = (value.data_ptr(), value.dtype, value.numel(), str(value.device))
    cached = _TENSOR_CACHE.get(key)
    if cached is not None and cached[0] is value:
        return cached[1]
    wrapped = _tla_tensor(value)
    if len(_TENSOR_CACHE) >= _TENSOR_CACHE_LIMIT:
        _TENSOR_CACHE.clear()
    _TENSOR_CACHE[key] = (value, wrapped)
    return wrapped


def arch_row_major():
    return tla.arch.RowMajor


def _check_cache_shape(k_cache, cache_layout):
    if cache_layout == "PA_BBND":
        if tuple(k_cache.shape[1:]) != (PAGE_SIZE, 1, D_QK):
            raise ValueError("PA_BBND k_cache must have shape [pages, 128, 1, 576]")
        return int(k_cache.shape[1]), int(k_cache.stride(0))
    if cache_layout == "PA_NZ":
        if tuple(k_cache.shape[1:]) != (1, D_QK // 16, PAGE_SIZE, 16):
            raise ValueError("PA_NZ k_cache must have shape [pages, 1, 36, 128, 16]")
        return int(k_cache.shape[3]), int(k_cache.stride(0))
    raise ValueError("cache_layout must be PA_BBND or PA_NZ")


def _check_capacity_shapes(q, block_table, cu_seqlens_q, cache_seqlens):
    """Validate the length-tensor shapes; these costs are host-only.

    The plan cache key only captures shapes, strides, dtypes and scalars, so a
    cache hit must still reject mismatched length tensors instead of indexing
    them blindly.
    """
    if q.ndim != 3:
        raise ValueError("q must have shape [total_q, num_heads, 576]")
    if block_table.ndim != 2:
        raise ValueError("block_table must be two-dimensional")
    batch = int(block_table.shape[0])
    if cu_seqlens_q.shape != (batch + 1,):
        raise ValueError("cu_seqlens_q must contain batch + 1 entries")
    if cache_seqlens.shape != (batch,):
        raise ValueError("cache_seqlens must contain one entry per request")


def _capacity_signature(cu_seqlens_q, cache_seqlens):
    """Identify the current contents of the two length tensors cheaply.

    Reading their values costs one device reduction plus a host sync, so the
    value checks are skipped while both the storage and the in-place update
    counter are unchanged.
    """
    return (
        cu_seqlens_q.data_ptr(),
        cu_seqlens_q._version,
        cache_seqlens.data_ptr(),
        cache_seqlens._version,
    )


def _check_capacity_values(q, block_table, cu_seqlens_q, cache_seqlens):
    """Validate the two capacity invariants that read tensor values.

    Without them a too large cache_seqlens or a mismatched cu_seqlens_q would
    drive the device kernel past the block_table and output bounds.
    """
    if int(cu_seqlens_q[-1]) != int(q.shape[0]):
        raise ValueError("cu_seqlens_q[-1] must equal q.shape[0]")
    if int(cache_seqlens.max()) > int(block_table.shape[1]) * PAGE_SIZE:
        raise ValueError("cache_seqlens must not exceed block_table pages * page_size")


def _plan_key(
    q, k_cache, block_table, cu_seqlens_q, cache_seqlens, scale, mask_mode, cache_layout
):
    return (
        tuple(q.shape),
        q.dtype,
        q.stride(),
        tuple(k_cache.shape),
        k_cache.dtype,
        k_cache.stride(),
        tuple(block_table.shape),
        block_table.dtype,
        block_table.stride(),
        tuple(cu_seqlens_q.shape),
        cu_seqlens_q.dtype,
        tuple(cache_seqlens.shape),
        cache_seqlens.dtype,
        str(q.device),
        str(k_cache.device),
        str(block_table.device),
        str(cu_seqlens_q.device),
        str(cache_seqlens.device),
        cache_layout,
        int(mask_mode),
        float(scale),
    )


def flash_mla_with_kvcache(
    q,
    k_cache,
    block_table,
    cu_seqlens_q,
    cache_seqlens,
    softmax_scale=None,
    mask_mode=MASK_NONE,
    cache_layout="PA_BBND",
):
    scale_value = 1.0 if softmax_scale is None else float(softmax_scale)
    # Shape-level checks are host-only; the two invariants that read tensor
    # values cost one device reduction, so they run on a plan miss and whenever
    # the length tensors changed.
    _check_capacity_shapes(q, block_table, cu_seqlens_q, cache_seqlens)
    capacity_sig = _capacity_signature(cu_seqlens_q, cache_seqlens)
    plan_key = _plan_key(
        q,
        k_cache,
        block_table,
        cu_seqlens_q,
        cache_seqlens,
        scale_value,
        mask_mode,
        cache_layout,
    )
    plan = _PLAN_CACHE.get(plan_key)

    if plan is None:
        if q.shape[2] != D_QK:
            raise ValueError("q head dimension must be 576")
        if q.shape[1] not in (64, 96):
            raise ValueError("q num_heads must be 64 or 96")
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("q dtype must be float16 or bfloat16")
        if not q.is_contiguous():
            raise ValueError("q must be contiguous")
        if k_cache.dtype != q.dtype:
            raise ValueError("k_cache dtype must match q dtype")
        if block_table.dtype != torch.int32:
            raise ValueError("block_table dtype must be int32")
        if cu_seqlens_q.dtype != torch.int32 or cache_seqlens.dtype != torch.int32:
            raise ValueError("sequence-length tensors must be int32")
        if not block_table.is_contiguous():
            raise ValueError("block_table must be contiguous")
        if not cu_seqlens_q.is_contiguous() or not cache_seqlens.is_contiguous():
            raise ValueError("sequence-length tensors must be contiguous")
        if mask_mode not in (MASK_NONE, MASK_RIGHT_DOWN_CAUSAL):
            raise ValueError("mask_mode must be 0 or 3")
        _, kv_blk_stride = _check_cache_shape(k_cache, cache_layout)

        total_q, heads, _ = q.shape
        batch = int(block_table.shape[0])
        num_pages = int(block_table.shape[1])
        _check_capacity_values(q, block_table, cu_seqlens_q, cache_seqlens)
        device_properties = torch_npu.npu.get_device_properties(q.device)
        cube_core_num = int(device_properties.cube_core_num)
        if cube_core_num < 1:
            raise RuntimeError("NPU device reports no available Cube cores")
        # N=64：两个相邻 query token 合并成一个 task（M=128），K/V 每 token
        # 只加载一次；只有 task 数仍能铺满 Cube 核（total_q >= 2*cores）时才启用，
        # 避免 T=1/8 这类小 batch 把并行度砍半。N=96：一个 token 一个 task（M=96）。
        # 合并后 task 数 = total_q/2；要求至少 4 个完整的 task 波次避免尾部失衡。
        paired = heads <= Q_BLOCK and total_q >= 8 * cube_core_num
        # 低并行度补核（纯 DSL 调度层）：total_q 极少时（decode 形态）任务数远小于
        # 核数，把一个 query 的 head 维切成 hgroups 组，多核并行处理同一 query 的
        # 不同 head 段。每组至少 16 个 head（MMA 原生 M 粒度），heads 需整除。
        hgroups = 1
        if not paired and total_q <= 4:
            budget = max(1, cube_core_num // max(1, total_q))
            for g in range(min(budget, heads // 16), 1, -1):
                if heads % g == 0 and heads // g >= 16:
                    hgroups = g
                    break
        # KV-split：低并行度（task 数远不够铺满核）时把每个 task 的 KV tile
        # 序列切成 kvsplit 段并行算，再由 kernel B 归并。要求每个 chunk 至少
        # 2 个 tile，否则并行收益被归并开销吃掉。
        ksplit = 1
        if not paired:
            tiles_max = (int(cache_seqlens.max()) + KV_BLOCK - 1) // KV_BLOCK
            base_tasks = max(1, total_q * hgroups)
            want = max(1, cube_core_num // base_tasks)
            for k in (8, 4, 2):
                if k <= want and 2 * k <= tiles_max:
                    ksplit = k
                    break
        lse_group = 1
        if ksplit > 1:
            # kernel B 的 partial-LSE 一次读 lse_group 个 head（32B 对齐），
            # 故 lane 数下限是 8；再取「2 的幂且整除 heads」。
            lg = max(8, (total_q * heads) // 64)
            lg = min(lg, 16, heads)
            while lg > 8 and (heads % lg != 0 or (lg & (lg - 1)) != 0):
                lg -= 1
            if heads % lg != 0:
                lg = 8
            lse_group = lg
        q_block = (
            2 * heads
            if paired
            else (
                heads // hgroups
                if hgroups > 1
                else (Q_BLOCK if heads <= Q_BLOCK else 96)
            )
        )
        # PV 宽度固定 128：L0B 用两个 32 KiB slot 做 V 分段 ping-pong，
        # 与 QK 的 K 分段共用同一组 slot。
        pv_width = 128
        executor_key = (
            str(q.dtype),
            str(q.device),
            cache_layout,
            heads,
            q_block,
            hgroups,
            ksplit,
            lse_group,
        )
        executor = _EXECUTOR_CACHE.get(executor_key)
        if executor is None:
            probe_out = torch.empty(
                (heads, total_q, D_V), dtype=q.dtype, device=q.device
            )
            probe_lse = torch.empty(
                (total_q, heads), dtype=torch.float32, device=q.device
            )
            executor = tla_compile(
                q,
                k_cache,
                probe_out,
                probe_lse,
                block_table,
                cu_seqlens_q,
                cache_seqlens,
                scale_value,
                total_q,
                batch,
                heads,
                num_pages,
                kv_blk_stride,
                mask_mode,
                q_block,
                hgroups,
                ksplit,
                lse_group,
                pv_width,
                cache_layout,
            )
            _EXECUTOR_CACHE[executor_key] = executor

        plan = {
            "batch": batch,
            "total_q": total_q,
            "heads": heads,
            "kv_blk_stride": kv_blk_stride,
            "num_pages": num_pages,
            "executor": executor,
            "kvsplit": ksplit,
            "lse_group": lse_group,
            "launch_blocks": min(total_q * hgroups * ksplit, cube_core_num),
            "launch_blocks_b": min(
                max(1, total_q * (heads // lse_group)), cube_core_num
            ),
            "dummy_po": torch.empty((8,), dtype=torch.float32, device=q.device),
            "dummy_pl": torch.empty((8,), dtype=torch.float32, device=q.device),
            "scale_t": tla_float32(scale_value),
            "total_q_t": tla_int32(total_q),
            "batch_t": tla_int32(batch),
            "heads_t": tla_int32(heads),
            "pages_t": tla_int32(num_pages),
            "blk_stride_t": tla_int32(kv_blk_stride),
            "mask_t": tla_int32(int(mask_mode)),
            "capacity_sig": capacity_sig,
        }
        if len(_PLAN_CACHE) >= _PLAN_CACHE_LIMIT:
            _PLAN_CACHE.clear()
        _PLAN_CACHE[plan_key] = plan
    elif plan["capacity_sig"] != capacity_sig:
        _check_capacity_values(q, block_table, cu_seqlens_q, cache_seqlens)
        plan["capacity_sig"] = capacity_sig

    total_q = plan["total_q"]
    heads = plan["heads"]
    # Official layout_out=NTD: (heads, total_q, D_V); LSE: (heads, total_q).
    output = torch.empty((heads, total_q, D_V), dtype=q.dtype, device=q.device)
    lse_tnd = torch.empty((total_q, heads), dtype=torch.float32, device=q.device)
    if total_q == 0 or plan["batch"] == 0:
        return output, lse_tnd.transpose(0, 1)

    q_t = _tla_tensor_cached(q)
    k_t = _tla_tensor_cached(k_cache)
    o_t = _tla_tensor(output)
    lse_t = _tla_tensor(lse_tnd)
    bt_t = _tla_tensor_cached(block_table)
    cu_t = _tla_tensor_cached(cu_seqlens_q)
    kvlen_t = _tla_tensor_cached(cache_seqlens)

    ksplit = plan["kvsplit"]
    if ksplit > 1:
        po = torch.empty(
            (ksplit * heads * total_q * D_V,),
            dtype=torch.float32,
            device=q.device,
        )
        pl = torch.empty(
            (ksplit * total_q * heads,),
            dtype=torch.float32,
            device=q.device,
        )
        po_t = _tla_tensor(po)
        pl_t = _tla_tensor(pl)
    else:
        po_t = _tla_tensor(plan["dummy_po"])
        pl_t = _tla_tensor(plan["dummy_pl"])
    ex_a = plan["executor"]
    if ksplit > 1:
        ex_a, ex_b = ex_a
    ex_a(
        q_t,
        k_t,
        o_t,
        lse_t,
        po_t,
        pl_t,
        bt_t,
        cu_t,
        kvlen_t,
        plan["scale_t"],
        plan["total_q_t"],
        plan["batch_t"],
        plan["heads_t"],
        plan["pages_t"],
        plan["blk_stride_t"],
        plan["mask_t"],
        block_num=plan["launch_blocks"],
    )
    if ksplit > 1:
        ex_b(
            po_t,
            pl_t,
            o_t,
            lse_t,
            plan["total_q_t"],
            plan["heads_t"],
            block_num=plan["launch_blocks_b"],
        )
    # Official contract: attn_out is NTD (heads, total_q, DV) and
    # softmax_lse is (heads, total_q). The kernel writes LSE in TND; returning
    # the transposed view avoids a per-call copy. Callers that need a packed
    # buffer can request `.contiguous()`.
    return output, lse_tnd.transpose(0, 1)


def tla_compile(
    q,
    k_cache,
    output,
    lse_tnd,
    block_table,
    cu_seqlens_q,
    cache_seqlens,
    scale_value,
    total_q,
    batch,
    heads,
    num_pages,
    kv_blk_stride,
    mask_mode,
    q_block,
    hgroups,
    kvsplit,
    lse_group,
    pv_width,
    cache_layout,
):
    dummy = torch.empty((8,), dtype=torch.float32, device=q.device)
    probe_po = (
        torch.empty(
            (kvsplit * heads * total_q * D_V,), dtype=torch.float32, device=q.device
        )
        if kvsplit > 1
        else dummy
    )
    probe_pl = (
        torch.empty((kvsplit * heads * total_q,), dtype=torch.float32, device=q.device)
        if kvsplit > 1
        else dummy
    )
    ex_a = tla.compile(
        flash_mla_with_kvcache_kernel,
        _tla_tensor_cached(q),
        _tla_tensor_cached(k_cache),
        _tla_tensor(output),
        _tla_tensor(lse_tnd),
        _tla_tensor(probe_po),
        _tla_tensor(probe_pl),
        _tla_tensor_cached(block_table),
        _tla_tensor_cached(cu_seqlens_q),
        _tla_tensor_cached(cache_seqlens),
        tla_float32(scale_value),
        tla_int32(total_q),
        tla_int32(batch),
        tla_int32(heads),
        tla_int32(num_pages),
        tla_int32(kv_blk_stride),
        tla_int32(int(mask_mode)),
        D_QK,
        D_V,
        q_block,
        hgroups,
        kvsplit,
        KV_BLOCK,
        pv_width,
        0 if cache_layout == "PA_BBND" else 1,
        options=NPU_ARCH,
    )
    if kvsplit == 1:
        return ex_a
    ex_b = tla.compile(
        flash_mla_kvsplit_reduce_kernel,
        _tla_tensor(probe_po),
        _tla_tensor(probe_pl),
        _tla_tensor(output),
        _tla_tensor(lse_tnd),
        tla_int32(total_q),
        tla_int32(heads),
        D_V,
        kvsplit,
        lse_group,
        options=NPU_ARCH,
    )
    return (ex_a, ex_b)


def tla_int32(value):
    return tla.Int32(value)


def tla_float32(value):
    return tla.Float32(value)


# -----------------------------------------------------------------------------------------------------------
# Example driver: build inputs, launch, check against the golden.
# -----------------------------------------------------------------------------------------------------------


@dataclass
class _Case:
    """One example case: a batch of requests and the operator selectors."""

    batch: int
    total_q: int
    kv_len: int
    heads: int
    dtype: str
    layout: str
    mask: int


#: The suite the battery replays. Between them the cases cover the decode shape
#: (one query row), the paired two-token path, the 96-head path, both cache
#: layouts, both mask modes, both dtypes, a partial last page and a KV-split case.
_CASES = (
    _Case(1, 1, 512, 64, "float16", CACHE_LAYOUT_BBND, MASK_NONE),
    _Case(1, 8, 128, 64, "float16", CACHE_LAYOUT_BBND, MASK_NONE),
    _Case(1, 64, 512, 64, "float16", CACHE_LAYOUT_BBND, MASK_NONE),
    _Case(2, 512, 1024, 64, "float16", CACHE_LAYOUT_BBND, MASK_NONE),
    _Case(1, 64, 512, 96, "float16", CACHE_LAYOUT_NZ, MASK_NONE),
    _Case(1, 64, 512, 64, "float16", CACHE_LAYOUT_BBND, MASK_RIGHT_DOWN_CAUSAL),
    _Case(2, 130, 383, 96, "bfloat16", CACHE_LAYOUT_NZ, MASK_RIGHT_DOWN_CAUSAL),
    _Case(2, 512, 2048, 64, "bfloat16", CACHE_LAYOUT_BBND, MASK_NONE),
)

#: Share of the elements that must land inside ``atol``, the same rule the
#: operator's frozen workload applies.
MATCHED_RATIO = 0.999


@dataclass
class _Inputs:
    """The operator inputs plus the per-request lengths they were built from."""

    q: torch.Tensor  # [total_q, heads, 576]
    k_cache: torch.Tensor  # [pages, ...] in the requested layout
    block_table: torch.Tensor  # [batch, pages_per_request] int32
    cu_seqlens_q: torch.Tensor  # [batch + 1] int32
    cache_seqlens: torch.Tensor  # [batch] int32
    softmax_scale: float


def _torch_dtype(name):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError("dtype must be float16 or bfloat16")


def _build_inputs(case, args):
    """Allocate the case's inputs.

    Every request gets its own physical pages, then the page order is shuffled:
    a kernel that ignored ``block_table`` and walked the cache linearly would
    still read valid memory, but the wrong tokens.
    """

    dtype = _torch_dtype(case.dtype)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    q_lens = [case.total_q // case.batch] * case.batch
    for index in range(case.total_q % case.batch):
        q_lens[index] += 1
    if case.mask == MASK_RIGHT_DOWN_CAUSAL and any(
        case.kv_len < q_len for q_len in q_lens
    ):
        raise ValueError("mask_mode=3 needs kv_len >= every request's query length")

    pages = max(1, -(-case.kv_len // PAGE_SIZE))
    num_blocks = case.batch * pages
    if case.layout == CACHE_LAYOUT_BBND:
        cache_shape = (num_blocks, PAGE_SIZE, KV_HEAD_NUM, HEAD_DIM_QK)
    else:
        cache_shape = (num_blocks, KV_HEAD_NUM, HEAD_DIM_QK // 16, PAGE_SIZE, 16)

    q = torch.randn(case.total_q, case.heads, HEAD_DIM_QK, generator=generator)
    k_cache = torch.randn(cache_shape, generator=generator)

    order = torch.randperm(num_blocks, generator=generator).tolist()
    block_table = [order[row * pages : (row + 1) * pages] for row in range(case.batch)]

    cu = [0]
    for q_len in q_lens:
        cu.append(cu[-1] + q_len)
    if cu[-1] != case.total_q:
        raise AssertionError("query split does not add up")

    device = torch.device("npu:%d" % args.device)
    return _Inputs(
        q=q.to(dtype).npu(device),
        k_cache=k_cache.to(dtype).npu(device),
        block_table=torch.tensor(block_table, dtype=torch.int32).npu(device),
        cu_seqlens_q=torch.tensor(cu, dtype=torch.int32).npu(device),
        cache_seqlens=torch.tensor([case.kv_len] * case.batch, dtype=torch.int32).npu(
            device
        ),
        softmax_scale=args.scale if args.scale > 0 else 1.0 / math.sqrt(HEAD_DIM_QK),
    )


def _check_output(result, expected, atol):
    """Return (matched_ratio, max_abs_error) of ``result`` against ``expected``."""

    diff = (result.float() - expected.float()).abs()
    if diff.numel() == 0:
        return 1.0, 0.0
    matched = float((diff <= atol).float().mean())
    return matched, float(diff.max())


def _check_lse(result, expected, atol):
    """Return (max_abs_error, same_invalid_mask) over the rows both call finite.

    A row with no visible token is ``-inf`` on both sides, so comparing the two
    infinities would only produce NaNs; the two masks are compared instead.
    """

    both_finite = torch.isfinite(result) & torch.isfinite(expected)
    same_mask = bool((torch.isfinite(result) == torch.isfinite(expected)).all())
    if not bool(both_finite.any()):
        return 0.0, same_mask
    error = (result[both_finite].float() - expected[both_finite].float()).abs()
    return float(error.max()), same_mask


def _launch(inputs, case):
    return flash_mla_with_kvcache(
        inputs.q,
        inputs.k_cache,
        inputs.block_table,
        inputs.cu_seqlens_q,
        inputs.cache_seqlens,
        inputs.softmax_scale,
        case.mask,
        case.layout,
    )


def _min_bytes(case):
    """Bytes one launch must move at least: latent pages, q, out and lse.

    V reuses the 576-wide latent page that K already needs, so a page is counted
    once.  With ``mask_mode=3`` some rows see fewer than ``kv_len`` tokens, so
    this is an upper bound there.
    """

    element = 2  # both fp16 and bf16 are two bytes wide
    kv = case.batch * case.kv_len * HEAD_DIM_QK * element
    q = case.total_q * case.heads * HEAD_DIM_QK * element
    out = case.total_q * case.heads * HEAD_DIM_V * element
    lse = case.total_q * case.heads * 4
    return kv + q + out + lse


def _measure_latency(case, args, inputs):
    """Time ``--perf-iters`` consecutive launches, one event pair per launch."""

    output, lse = _launch(inputs, case)
    for _ in range(max(0, args.perf_warmup - 1)):
        output, lse = _launch(inputs, case)
    torch.npu.synchronize(args.device)

    starts = [torch.npu.Event(enable_timing=True) for _ in range(args.perf_iters)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(args.perf_iters)]
    for start, end in zip(starts, ends):
        start.record()
        output, lse = _launch(inputs, case)
        end.record()
    torch.npu.synchronize(args.device)

    # The timed launches must still produce the declared output shapes: a silent
    # shape change would make the latency and bandwidth numbers meaningless.
    if tuple(output.shape) != (case.heads, case.total_q, HEAD_DIM_V):
        raise RuntimeError("timed launch returned an unexpected output shape")
    if tuple(lse.shape) != (case.heads, case.total_q):
        raise RuntimeError("timed launch returned an unexpected lse shape")

    samples = sorted(start.elapsed_time(end) for start, end in zip(starts, ends))
    mean_ms = sum(samples) / len(samples)
    total_bytes = _min_bytes(case)
    return {
        "latency_scope": "eager_end_to_end_including_host",
        "latency_us_mean": mean_ms * 1000.0,
        "latency_us_min": samples[0] * 1000.0,
        "latency_us_max": samples[-1] * 1000.0,
        "bytes_moved": total_bytes,
        "effective_gbps": total_bytes / mean_ms / 1e6,
        "tokens_per_s": case.total_q / (mean_ms * 1e-3),
        "perf_warmup": args.perf_warmup,
        "perf_iters": len(samples),
    }


def _run_case(case, args):
    inputs = _build_inputs(case, args)
    result, lse = _launch(inputs, case)
    torch.npu.synchronize(args.device)

    record = {
        "name": "B%d_T%d_KV%d_H%d_%s_%s_M%d"
        % (
            case.batch,
            case.total_q,
            case.kv_len,
            case.heads,
            case.dtype,
            case.layout,
            case.mask,
        ),
        "batch": case.batch,
        "total_q": case.total_q,
        "kv_len": case.kv_len,
        "heads": case.heads,
        "dtype": case.dtype,
        "cache_layout": case.layout,
        "mask_mode": case.mask,
    }
    if args.perf or args.perf_only:
        record.update(_measure_latency(case, args, inputs))
    if args.perf_only:
        record["passed"] = None
        return record

    golden = (
        (
            inputs.q.cpu(),
            inputs.k_cache.cpu(),
            inputs.block_table.cpu(),
            inputs.cu_seqlens_q.cpu(),
            inputs.cache_seqlens.cpu(),
        )
        if args.golden_device == "cpu"
        else (
            inputs.q,
            inputs.k_cache,
            inputs.block_table,
            inputs.cu_seqlens_q,
            inputs.cache_seqlens,
        )
    )
    expected_out, expected_lse = flash_mla_with_kvcache_golden(
        golden[0],
        golden[1],
        golden[2],
        golden[3],
        golden[4],
        inputs.softmax_scale,
        case.mask,
        case.layout,
    )
    expected_out = expected_out.to(result.device)
    expected_lse = expected_lse.to(lse.device)

    atol = (
        args.atol if args.atol > 0 else (0.01 if case.dtype == "bfloat16" else 0.0025)
    )
    matched, max_abs = _check_output(result, expected_out, atol)
    lse_error, lse_mask_ok = _check_lse(lse, expected_lse, atol)
    lse_ok = lse_mask_ok and lse_error <= max(atol * 10.0, 0.02)

    record.update(
        {
            "atol": atol,
            "matched_ratio": matched,
            "out_max_abs_error": max_abs,
            "lse_max_abs_error": lse_error,
            "passed": bool(matched >= MATCHED_RATIO and lse_ok),
        }
    )
    return record


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch", "--b", type=int, default=1)
    parser.add_argument("--total-q", "--t", type=int, default=512)
    parser.add_argument("--kv-len", "--k", type=int, default=1024)
    parser.add_argument("--heads", "--h", type=int, default=64, choices=HEAD_CHOICES)
    parser.add_argument("--dtype", choices=DTYPE_NAMES, default="float16")
    parser.add_argument("--layout", choices=CACHE_LAYOUTS, default=CACHE_LAYOUT_BBND)
    parser.add_argument(
        "--mask",
        type=int,
        choices=MASK_MODES,
        default=MASK_NONE,
        help="0: dense, 3: right-down causal over cache_seqlens",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.0,
        help="softmax scale; 0 uses 1/sqrt(576)",
    )
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument(
        "--golden-device",
        choices=("npu", "cpu"),
        default="npu",
        help="where the golden runs; cpu is the slow independent cross-check",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=0.0,
        help="absolute tolerance; 0 uses 0.0025 (fp16) or 0.01 (bf16)",
    )
    parser.add_argument(
        "--perf", action="store_true", help="also report device latency"
    )
    parser.add_argument(
        "--perf-only",
        "--perf_only",
        dest="perf_only",
        action="store_true",
        help="only measure latency, skip the golden check",
    )
    parser.add_argument("--perf-iters", type=int, default=20)
    parser.add_argument(
        "--perf-warmup", type=int, default=5, help="launches discarded before timing"
    )
    parser.add_argument(
        "--cases", action="store_true", help="validate the boundary-case suite"
    )
    args = parser.parse_args(argv)
    if args.device < 0:
        parser.error("--device must be nonnegative")
    if args.batch < 1 or args.total_q < 1 or args.kv_len < 1:
        parser.error("--batch, --total-q and --kv-len must be positive")
    if args.batch > args.total_q:
        parser.error("--batch must not exceed --total-q")
    if not math.isfinite(args.scale) or args.scale < 0:
        parser.error("--scale must be finite and nonnegative")
    if not math.isfinite(args.atol) or args.atol < 0:
        parser.error("--atol must be finite and nonnegative")
    if args.perf_iters < 1:
        parser.error("--perf-iters must be positive")
    if args.perf_warmup < 0:
        parser.error("--perf-warmup must be nonnegative")
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.npu.set_device(args.device)
    cases = (
        _CASES
        if args.cases
        else (
            _Case(
                args.batch,
                args.total_q,
                args.kv_len,
                args.heads,
                args.dtype,
                args.layout,
                args.mask,
            ),
        )
    )
    failed = 0
    for case in cases:
        result = _run_case(case, args)
        failed += 1 if result["passed"] is False else 0
        print(json.dumps(result), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
