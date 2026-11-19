# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Fused QKV split + per-head RMSNorm + RoPE, a pure AIV kernel.

One kernel does all three stages of the vLLM ``split_qkv_rmsnorm_rope`` pass:

* split the fused row into ``[q | k | v]``;
* per-head RMSNorm (no mean removal) in fp32, then scale by the head weight
  (plus optional bias) and round to bfloat16;
* NeoX RoPE on the first ``rope_dim`` lanes of each head, in fp32 with a single
  bfloat16 rounding.

The structure follows the pingpong pattern of
``dsl_tutorials/tutorial_matrix_add/matrix_add_v2.py``: a dynamic loop walks
``rows_per_group``-row groups of one block's chunk, and inside it a *static*
slot loop indexes several staging buffer sets, each with its own mutexes, so the
asynchronous MTE2 / MTE3 of one slot overlap the vector work of another.

Run it from the repository root::

    python examples/python/tla_dsl/experimental/kernel/vec/\
split_qkv_rmsnorm_rope/split_qkv_rmsnorm_rope.py --device 0
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys
import threading
from pathlib import Path

_DSL_EXAMPLE_PATH = str(
    next(p for p in Path(__file__).resolve().parents if (p / "helpers").is_dir())
)

if _DSL_EXAMPLE_PATH not in sys.path:
    sys.path.append(_DSL_EXAMPLE_PATH)

import torch

import catlass.tla as tla
from catlass.tla.runtime import _Tensor, from_dlpack

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------


_CAST_EVEN = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ZERO,
    sat_mode=tla.params.SatMode.NOSAT,
    round_mode=tla.params.RoundMode.CAST_ROUND,
)
_CAST_ODD = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ONE,
    sat_mode=tla.params.SatMode.NOSAT,
    round_mode=tla.params.RoundMode.CAST_ROUND,
)
_CAST_NONE = tla.params.CastParams()
_LOAD_UNPACK16 = tla.params.NormalLoadParams(
    load_dist=tla.params.LoadDist.DIST_UNPACK_B16
)
_STORE_PACK32 = tla.params.NormalStoreParams(
    store_dist=tla.params.StoreDist.DIST_PACK_B32
)

# Total staging budget shared by the pingpong slots.
_DEFAULT_GROUP_UB_BUDGET = 236 * 1024
# Wide rows (>~30 KB per staged row) measure faster with a shallower footprint:
# on the 200-case workload a 228 KB three-slot layout lost up to 17% against a
# 152 KB two-slot one, while narrow rows need the deeper pingpong instead.
_WIDE_ROW_BYTES = 30 * 1024
_WIDE_ROW_UB_BUDGET = 160 * 1024
_DEFAULT_MAX_ROWS_PER_GROUP = 8
_SLOTS = 2
_FUSED_ROPE_MIN_ROWS = 1024
# UB tile base-address granularity (AVE vector buffers).
_UB_ALIGN_BYTES = 256


def _group_ub_budget(row_bytes):
    return (
        _WIDE_ROW_UB_BUDGET
        if row_bytes >= _WIDE_ROW_BYTES
        else _DEFAULT_GROUP_UB_BUDGET
    )


def default_slots(q_hidden, kv_hidden):
    """Pick the pingpong depth: deepest that still stages >= 2 rows per slot."""
    row_bytes = (q_hidden + 2 * kv_hidden + q_hidden + kv_hidden) * 2
    budget = _group_ub_budget(row_bytes)
    for candidate in (4, 3, 2):
        if (budget // candidate) // row_bytes >= 2:
            return candidate
    return 2


def default_rows_per_group(q_hidden, kv_hidden, slots=_SLOTS):
    row_bytes = (q_hidden + 2 * kv_hidden + q_hidden + kv_hidden) * 2
    per_slot = _group_ub_budget(row_bytes) // slots
    return min(_DEFAULT_MAX_ROWS_PER_GROUP, max(1, per_slot // row_bytes))


def _norm_head(
    src,
    dst,
    in_head,
    out_head,
    w_even,
    w_odd,
    b_even,
    b_odd,
    head_dim,
    eps,
    all_lanes,
    one_vec,
    has_bias,
):
    """RMSNorm + weight + bias for one head, written back as bf16."""
    x = tla.tile_view(src, tla.make_shape(head_dim), tla.make_coord(in_head)).load()
    x_even = x.to(tla.Float32, _CAST_EVEN)
    x_odd = x.to(tla.Float32, _CAST_ODD)

    squares = tla.add(tla.mul(x_even, x_even), tla.mul(x_odd, x_odd))
    square_sum = tla.full(
        squares.reduce(tla.ReductionOp.ADD, mask=all_lanes), dtype=tla.Float32
    )
    variance = tla.add(tla.mul(square_sum, 1.0 / head_dim), eps)
    inv_rms = tla.div(one_vec, tla.sqrt(variance))

    norm_even = tla.mul(tla.mul(x_even, inv_rms), w_even)
    norm_odd = tla.mul(tla.mul(x_odd, inv_rms), w_odd)
    if tla.const_expr(has_bias):
        norm_even = tla.add(norm_even, b_even)
        norm_odd = tla.add(norm_odd, b_odd)

    normalized = tla.add(
        norm_even.to(tla.BFloat16, _CAST_EVEN),
        norm_odd.to(tla.BFloat16, _CAST_ODD),
    )
    tla.tile_view(dst, tla.make_shape(head_dim), tla.make_coord(out_head)).store(
        normalized
    )


def _norm_rope_full(
    src,
    dst,
    in_head,
    out_head,
    w0,
    w1,
    b0,
    b1,
    cos_v,
    sin_v,
    head_dim,
    eps,
    all_lanes,
    one_vec,
    has_bias,
):
    """Fuse RMSNorm and full-width NeoX RoPE without an intermediate UB pass.

    Two contiguous 64-lane half-head loads already expose the NeoX pair, so no
    lane permutation is needed.  The normalized values are rounded to bf16
    before rotation to preserve the benchmark's rounding point.
    """
    lanes = 64
    in_tile = in_head * (head_dim // lanes)
    out_tile = out_head * (head_dim // lanes)
    x0 = (
        tla.tile_view(src, tla.make_shape(lanes), tla.make_coord(in_tile))
        .load(_LOAD_UNPACK16)
        .to(tla.Float32, _CAST_NONE)
    )
    x1 = (
        tla.tile_view(src, tla.make_shape(lanes), tla.make_coord(in_tile + 1))
        .load(_LOAD_UNPACK16)
        .to(tla.Float32, _CAST_NONE)
    )

    squares = tla.add(tla.mul(x0, x0), tla.mul(x1, x1))
    square_sum = tla.full(
        squares.reduce(tla.ReductionOp.ADD, mask=all_lanes), dtype=tla.Float32
    )
    variance = tla.add(tla.mul(square_sum, 1.0 / head_dim), eps)
    inv_rms = tla.div(one_vec, tla.sqrt(variance))

    n0 = tla.mul(tla.mul(x0, inv_rms), w0)
    n1 = tla.mul(tla.mul(x1, inv_rms), w1)
    if tla.const_expr(has_bias):
        n0 = tla.add(n0, b0)
        n1 = tla.add(n1, b1)

    n0 = n0.to(tla.BFloat16, _CAST_EVEN).to(tla.Float32, _CAST_NONE)
    n1 = n1.to(tla.BFloat16, _CAST_EVEN).to(tla.Float32, _CAST_NONE)
    o0 = tla.sub(tla.mul(n0, cos_v), tla.mul(n1, sin_v))
    o1 = tla.add(tla.mul(n1, cos_v), tla.mul(n0, sin_v))

    tla.tile_view(dst, tla.make_shape(lanes), tla.make_coord(out_tile)).store(
        o0.to(tla.BFloat16, _CAST_EVEN), _STORE_PACK32
    )
    tla.tile_view(dst, tla.make_shape(lanes), tla.make_coord(out_tile + 1)).store(
        o1.to(tla.BFloat16, _CAST_EVEN), _STORE_PACK32
    )


def _rope_head(dst, head, head_dim, half, cos_even, cos_odd, sin_even, sin_odd):
    """NeoX RoPE on the first ``2 * half`` lanes of one head (fp32, one rounding).

    ``half`` must divide ``head_dim`` so the head base stays tile aligned; the
    public entry rejects shapes that break that assumption.
    """
    first = (head * head_dim) // half
    x1 = tla.tile_view(dst, tla.make_shape(half), tla.make_coord(first)).load()
    x2 = tla.tile_view(dst, tla.make_shape(half), tla.make_coord(first + 1)).load()

    x1_even, x1_odd = x1.to(tla.Float32, _CAST_EVEN), x1.to(tla.Float32, _CAST_ODD)
    x2_even, x2_odd = x2.to(tla.Float32, _CAST_EVEN), x2.to(tla.Float32, _CAST_ODD)

    low_even = tla.sub(tla.mul(x1_even, cos_even), tla.mul(x2_even, sin_even))
    low_odd = tla.sub(tla.mul(x1_odd, cos_odd), tla.mul(x2_odd, sin_odd))
    high_even = tla.add(tla.mul(x2_even, cos_even), tla.mul(x1_even, sin_even))
    high_odd = tla.add(tla.mul(x2_odd, cos_odd), tla.mul(x1_odd, sin_odd))

    low = tla.add(
        low_even.to(tla.BFloat16, _CAST_EVEN), low_odd.to(tla.BFloat16, _CAST_ODD)
    )
    high = tla.add(
        high_even.to(tla.BFloat16, _CAST_EVEN), high_odd.to(tla.BFloat16, _CAST_ODD)
    )

    tla.tile_view(dst, tla.make_shape(half), tla.make_coord(first)).store(low)
    tla.tile_view(dst, tla.make_shape(half), tla.make_coord(first + 1)).store(high)


def _weight_parity(ub_weight, ub_bias, head_shape):
    weight = tla.tile_view(ub_weight, head_shape, tla.make_coord(0)).load()
    bias = tla.tile_view(ub_bias, head_shape, tla.make_coord(0)).load()
    return (
        weight.to(tla.Float32, _CAST_EVEN),
        weight.to(tla.Float32, _CAST_ODD),
        bias.to(tla.Float32, _CAST_EVEN),
        bias.to(tla.Float32, _CAST_ODD),
    )


def _cos_sin_parity(ub_cos_sin, row, half_shape):
    cos = tla.tile_view(ub_cos_sin, half_shape, tla.make_coord(row * 2)).load()
    sin = tla.tile_view(ub_cos_sin, half_shape, tla.make_coord(row * 2 + 1)).load()
    return (
        cos.to(tla.Float32, _CAST_EVEN),
        cos.to(tla.Float32, _CAST_ODD),
        sin.to(tla.Float32, _CAST_EVEN),
        sin.to(tla.Float32, _CAST_ODD),
    )


@tla.kernel
def split_qkv_rmsnorm_rope_kernel(
    gm_input: tla.Tensor,
    gm_cache: tla.Tensor,
    gm_positions: tla.Tensor,
    gm_q_weight: tla.Tensor,
    gm_k_weight: tla.Tensor,
    gm_q_bias: tla.Tensor,
    gm_k_bias: tla.Tensor,
    gm_q: tla.Tensor,
    gm_k: tla.Tensor,
    gm_v: tla.Tensor,
    q_hidden: tla.Constexpr[int],
    kv_hidden: tla.Constexpr[int],
    head_dim: tla.Constexpr[int],
    rope_dim: tla.Constexpr[int],
    eps: tla.Constexpr[float],
    has_bias: tla.Constexpr[bool],
    rows_per_group: tla.Constexpr[int],
    slots: tla.Constexpr[int],
    fuse_rope: tla.Constexpr[bool],
) -> None:
    total_hidden = q_hidden + 2 * kv_hidden
    q_heads = q_hidden // head_dim
    kv_heads = kv_hidden // head_dim
    staged_heads = total_hidden // head_dim
    half_rope = rope_dim // 2
    kv_per_row = total_hidden // kv_hidden
    v_tile = (q_hidden + kv_hidden) // kv_hidden

    num_rows = gm_input.origin_shape[0] // total_hidden
    block_id = tla.arch.block_idx()
    num_blocks = tla.arch.block_num()

    full_groups = num_rows // rows_per_group
    tail_rows = num_rows - full_groups * rows_per_group
    # Balanced partition: block b owns [b*F//B, (b+1)*F//B). Rounding a
    # ceil-based quota up to the slot count would let a block also cover the
    # next block's first group, so two AIVs would write the same output rows.
    my_start = (block_id * full_groups) // num_blocks
    my_stop = ((block_id + 1) * full_groups) // num_blocks
    group_iters = (my_stop - my_start + slots - 1) // slots

    group_in_shape = tla.make_shape(rows_per_group * total_hidden)
    group_q_shape = tla.make_shape(rows_per_group * q_hidden)
    group_kv_shape = tla.make_shape(rows_per_group * kv_hidden)
    group_cs_shape = tla.make_shape(rows_per_group * rope_dim)
    row_in_shape = tla.make_shape(total_hidden)
    row_q_shape = tla.make_shape(q_hidden)
    row_kv_shape = tla.make_shape(kv_hidden)
    row_cs_shape = tla.make_shape(rope_dim)
    head_shape = tla.make_shape(head_dim)
    half_shape = tla.make_shape(half_rope)

    # ---- pingpong staging buffers, one set per static slot -------------------
    row_bufs = [None] * slots
    q_bufs = [None] * slots
    k_bufs = [None] * slots
    cs_bufs = [None] * slots
    m_row = [None] * slots
    m_q = [None] * slots
    m_k = [None] * slots
    for slot in tla.range_constexpr(slots):
        row_bufs[slot] = tla.make_tensor_like(
            tla.allocate(
                rows_per_group * total_hidden,
                tla.BFloat16,
                tla.AddressSpace.ub,
                _UB_ALIGN_BYTES,
            ),
            tla.tile_view(gm_input, group_in_shape, tla.make_coord(0)),
            tla.arch.RowMajor,
        )
        q_bufs[slot] = tla.make_tensor_like(
            tla.allocate(
                rows_per_group * q_hidden,
                tla.BFloat16,
                tla.AddressSpace.ub,
                _UB_ALIGN_BYTES,
            ),
            tla.tile_view(gm_q, group_q_shape, tla.make_coord(0)),
            tla.arch.RowMajor,
        )
        k_bufs[slot] = tla.make_tensor_like(
            tla.allocate(
                rows_per_group * kv_hidden,
                tla.BFloat16,
                tla.AddressSpace.ub,
                _UB_ALIGN_BYTES,
            ),
            tla.tile_view(gm_k, group_kv_shape, tla.make_coord(0)),
            tla.arch.RowMajor,
        )
        cs_bufs[slot] = tla.make_tensor_like(
            tla.allocate(
                rows_per_group * rope_dim,
                tla.BFloat16,
                tla.AddressSpace.ub,
                _UB_ALIGN_BYTES,
            ),
            tla.tile_view(gm_cache, group_cs_shape, tla.make_coord(0)),
            tla.arch.RowMajor,
        )
        m_row[slot] = tla.mutex(resource="row", id=3 * slot)
        m_q[slot] = tla.mutex(resource="q", id=3 * slot + 1)
        m_k[slot] = tla.mutex(resource="k", id=3 * slot + 2)

    ub_q_weight = tla.make_tensor_like(
        tla.allocate(head_dim, tla.BFloat16, tla.AddressSpace.ub, _UB_ALIGN_BYTES),
        tla.tile_view(gm_q_weight, head_shape, tla.make_coord(0)),
        tla.arch.RowMajor,
    )
    ub_k_weight = tla.make_tensor_like(
        tla.allocate(head_dim, tla.BFloat16, tla.AddressSpace.ub, _UB_ALIGN_BYTES),
        tla.tile_view(gm_k_weight, head_shape, tla.make_coord(0)),
        tla.arch.RowMajor,
    )
    ub_q_bias = tla.make_tensor_like(
        tla.allocate(head_dim, tla.BFloat16, tla.AddressSpace.ub, _UB_ALIGN_BYTES),
        tla.tile_view(gm_q_bias, head_shape, tla.make_coord(0)),
        tla.arch.RowMajor,
    )
    ub_k_bias = tla.make_tensor_like(
        tla.allocate(head_dim, tla.BFloat16, tla.AddressSpace.ub, _UB_ALIGN_BYTES),
        tla.tile_view(gm_k_bias, head_shape, tla.make_coord(0)),
        tla.arch.RowMajor,
    )

    with tla.vector():
        tla.copy(ub_q_weight, tla.tile_view(gm_q_weight, head_shape, tla.make_coord(0)))
        tla.copy(ub_k_weight, tla.tile_view(gm_k_weight, head_shape, tla.make_coord(0)))
        if tla.const_expr(has_bias):
            tla.copy(ub_q_bias, tla.tile_view(gm_q_bias, head_shape, tla.make_coord(0)))
            tla.copy(ub_k_bias, tla.tile_view(gm_k_bias, head_shape, tla.make_coord(0)))
        tla.pipe_barrier(tla.pipes.ALL)

        for group_iter in tla.range(group_iters):
            for slot in tla.range_constexpr(slots):
                group = my_start + group_iter * slots + slot
                if group < my_stop:
                    row_ub = row_bufs[slot]
                    q_ub = q_bufs[slot]
                    k_ub = k_bufs[slot]
                    cs_ub = cs_bufs[slot]

                    # --- MTE2: stage the group ------------------------------
                    m_row[slot].lock(pipe=tla.arch.MTE2)
                    tla.copy(
                        row_ub,
                        tla.tile_view(gm_input, group_in_shape, tla.make_coord(group)),
                    )
                    for pos_iter in tla.range(rows_per_group):
                        pos = gm_positions[group * rows_per_group + pos_iter].to(
                            tla.Int32
                        )
                        tla.copy(
                            tla.tile_view(
                                cs_ub, row_cs_shape, tla.make_coord(pos_iter)
                            ),
                            tla.tile_view(gm_cache, row_cs_shape, tla.make_coord(pos)),
                        )
                    m_row[slot].unlock(pipe=tla.arch.MTE2)

                    # --- VECTOR: RMSNorm + RoPE -----------------------------
                    m_row[slot].lock(pipe=tla.arch.VECTOR)
                    m_q[slot].lock(pipe=tla.arch.VECTOR)
                    m_k[slot].lock(pipe=tla.arch.VECTOR)
                    with tla.vec.func(mode="simd"):
                        qw_even, qw_odd, qb_even, qb_odd = _weight_parity(
                            ub_q_weight, ub_q_bias, head_shape
                        )
                        kw_even, kw_odd, kb_even, kb_odd = _weight_parity(
                            ub_k_weight, ub_k_bias, head_shape
                        )
                        all_lanes = tla.create_mask(
                            pattern=tla.mask.ALL, dtype=tla.Float32
                        )
                        one_vec = tla.full(1.0, dtype=tla.Float32)
                        for proc_row in tla.range(rows_per_group):
                            p_in_base = proc_row * staged_heads
                            p_out_base = proc_row * q_heads
                            p_k_base = proc_row * kv_heads
                            cos_even, cos_odd, sin_even, sin_odd = _cos_sin_parity(
                                cs_ub, proc_row, half_shape
                            )
                            if tla.const_expr(fuse_rope):
                                fqw0 = (
                                    tla.tile_view(
                                        ub_q_weight,
                                        tla.make_shape(64),
                                        tla.make_coord(0),
                                    )
                                    .load(_LOAD_UNPACK16)
                                    .to(tla.Float32, _CAST_NONE)
                                )
                                fqw1 = (
                                    tla.tile_view(
                                        ub_q_weight,
                                        tla.make_shape(64),
                                        tla.make_coord(1),
                                    )
                                    .load(_LOAD_UNPACK16)
                                    .to(tla.Float32, _CAST_NONE)
                                )
                                fkw0 = (
                                    tla.tile_view(
                                        ub_k_weight,
                                        tla.make_shape(64),
                                        tla.make_coord(0),
                                    )
                                    .load(_LOAD_UNPACK16)
                                    .to(tla.Float32, _CAST_NONE)
                                )
                                fkw1 = (
                                    tla.tile_view(
                                        ub_k_weight,
                                        tla.make_shape(64),
                                        tla.make_coord(1),
                                    )
                                    .load(_LOAD_UNPACK16)
                                    .to(tla.Float32, _CAST_NONE)
                                )
                                fqb0 = fqw0
                                fqb1 = fqw1
                                fkb0 = fkw0
                                fkb1 = fkw1
                                if tla.const_expr(has_bias):
                                    fqb0 = (
                                        tla.tile_view(
                                            ub_q_bias,
                                            tla.make_shape(64),
                                            tla.make_coord(0),
                                        )
                                        .load(_LOAD_UNPACK16)
                                        .to(tla.Float32, _CAST_NONE)
                                    )
                                    fqb1 = (
                                        tla.tile_view(
                                            ub_q_bias,
                                            tla.make_shape(64),
                                            tla.make_coord(1),
                                        )
                                        .load(_LOAD_UNPACK16)
                                        .to(tla.Float32, _CAST_NONE)
                                    )
                                    fkb0 = (
                                        tla.tile_view(
                                            ub_k_bias,
                                            tla.make_shape(64),
                                            tla.make_coord(0),
                                        )
                                        .load(_LOAD_UNPACK16)
                                        .to(tla.Float32, _CAST_NONE)
                                    )
                                    fkb1 = (
                                        tla.tile_view(
                                            ub_k_bias,
                                            tla.make_shape(64),
                                            tla.make_coord(1),
                                        )
                                        .load(_LOAD_UNPACK16)
                                        .to(tla.Float32, _CAST_NONE)
                                    )
                                fcos = (
                                    tla.tile_view(
                                        cs_ub,
                                        tla.make_shape(64),
                                        tla.make_coord(proc_row * 2),
                                    )
                                    .load(_LOAD_UNPACK16)
                                    .to(tla.Float32, _CAST_NONE)
                                )
                                fsin = (
                                    tla.tile_view(
                                        cs_ub,
                                        tla.make_shape(64),
                                        tla.make_coord(proc_row * 2 + 1),
                                    )
                                    .load(_LOAD_UNPACK16)
                                    .to(tla.Float32, _CAST_NONE)
                                )
                                for fq_head in tla.range(q_heads):
                                    _norm_rope_full(
                                        row_ub,
                                        q_ub,
                                        p_in_base + fq_head,
                                        p_out_base + fq_head,
                                        fqw0,
                                        fqw1,
                                        fqb0,
                                        fqb1,
                                        fcos,
                                        fsin,
                                        head_dim,
                                        eps,
                                        all_lanes,
                                        one_vec,
                                        has_bias,
                                    )
                                for fk_head in tla.range(kv_heads):
                                    _norm_rope_full(
                                        row_ub,
                                        k_ub,
                                        p_in_base + q_heads + fk_head,
                                        p_k_base + fk_head,
                                        fkw0,
                                        fkw1,
                                        fkb0,
                                        fkb1,
                                        fcos,
                                        fsin,
                                        head_dim,
                                        eps,
                                        all_lanes,
                                        one_vec,
                                        has_bias,
                                    )
                            else:
                                for q_head in tla.range(q_heads):
                                    _norm_head(
                                        row_ub,
                                        q_ub,
                                        p_in_base + q_head,
                                        p_out_base + q_head,
                                        qw_even,
                                        qw_odd,
                                        qb_even,
                                        qb_odd,
                                        head_dim,
                                        eps,
                                        all_lanes,
                                        one_vec,
                                        has_bias,
                                    )
                                for k_head in tla.range(kv_heads):
                                    _norm_head(
                                        row_ub,
                                        k_ub,
                                        p_in_base + q_heads + k_head,
                                        p_k_base + k_head,
                                        kw_even,
                                        kw_odd,
                                        kb_even,
                                        kb_odd,
                                        head_dim,
                                        eps,
                                        all_lanes,
                                        one_vec,
                                        has_bias,
                                    )
                                tla.local_mem_bar(
                                    tla.params.MemType.VEC_STORE,
                                    tla.params.MemType.VEC_LOAD,
                                )
                                for rq_head in tla.range(q_heads):
                                    _rope_head(
                                        q_ub,
                                        p_out_base + rq_head,
                                        head_dim,
                                        half_rope,
                                        cos_even,
                                        cos_odd,
                                        sin_even,
                                        sin_odd,
                                    )
                                for rk_head in tla.range(kv_heads):
                                    _rope_head(
                                        k_ub,
                                        p_k_base + rk_head,
                                        head_dim,
                                        half_rope,
                                        cos_even,
                                        cos_odd,
                                        sin_even,
                                        sin_odd,
                                    )
                    m_k[slot].unlock(pipe=tla.arch.VECTOR)
                    m_q[slot].unlock(pipe=tla.arch.VECTOR)
                    m_row[slot].unlock(pipe=tla.arch.VECTOR)

                    # --- MTE3: write Q / K / V back -------------------------
                    m_q[slot].lock(pipe=tla.arch.MTE3)
                    tla.copy(
                        tla.tile_view(gm_q, group_q_shape, tla.make_coord(group)), q_ub
                    )
                    m_q[slot].unlock(pipe=tla.arch.MTE3)
                    m_k[slot].lock(pipe=tla.arch.MTE3)
                    tla.copy(
                        tla.tile_view(gm_k, group_kv_shape, tla.make_coord(group)), k_ub
                    )
                    m_k[slot].unlock(pipe=tla.arch.MTE3)
                    m_row[slot].lock(pipe=tla.arch.MTE3)
                    for v_row in tla.range(rows_per_group):
                        tla.copy(
                            tla.tile_view(
                                gm_v,
                                row_kv_shape,
                                tla.make_coord(group * rows_per_group + v_row),
                            ),
                            tla.tile_view(
                                row_ub,
                                row_kv_shape,
                                tla.make_coord(v_row * kv_per_row + v_tile),
                            ),
                        )
                    m_row[slot].unlock(pipe=tla.arch.MTE3)

        # Drain the group pipeline before slot 0 is reused for the tail rows.
        tla.pipe_barrier(tla.pipes.ALL)

        # Leftover rows (< rows_per_group) are handled by block 0 on slot 0.
        if block_id == 0:
            if tail_rows > 0:
                for tail_iter in tla.range(tail_rows):
                    t_row = full_groups * rows_per_group + tail_iter
                    t_pos = gm_positions[t_row].to(tla.Int32)
                    tla.copy(
                        row_bufs[0],
                        tla.tile_view(gm_input, row_in_shape, tla.make_coord(t_row)),
                    )
                    tla.copy(
                        tla.tile_view(cs_bufs[0], row_cs_shape, tla.make_coord(0)),
                        tla.tile_view(gm_cache, row_cs_shape, tla.make_coord(t_pos)),
                    )
                    tla.pipe_barrier(tla.pipes.ALL)

                    with tla.vec.func(mode="simd"):
                        t_all_lanes = tla.create_mask(
                            pattern=tla.mask.ALL, dtype=tla.Float32
                        )
                        t_one_vec = tla.full(1.0, dtype=tla.Float32)
                        t_qw_even, t_qw_odd, t_qb_even, t_qb_odd = _weight_parity(
                            ub_q_weight, ub_q_bias, head_shape
                        )
                        t_kw_even, t_kw_odd, t_kb_even, t_kb_odd = _weight_parity(
                            ub_k_weight, ub_k_bias, head_shape
                        )
                        t_cos_even, t_cos_odd, t_sin_even, t_sin_odd = _cos_sin_parity(
                            cs_bufs[0], 0, half_shape
                        )
                        if tla.const_expr(fuse_rope):
                            tfqw0 = (
                                tla.tile_view(
                                    ub_q_weight, tla.make_shape(64), tla.make_coord(0)
                                )
                                .load(_LOAD_UNPACK16)
                                .to(tla.Float32, _CAST_NONE)
                            )
                            tfqw1 = (
                                tla.tile_view(
                                    ub_q_weight, tla.make_shape(64), tla.make_coord(1)
                                )
                                .load(_LOAD_UNPACK16)
                                .to(tla.Float32, _CAST_NONE)
                            )
                            tfkw0 = (
                                tla.tile_view(
                                    ub_k_weight, tla.make_shape(64), tla.make_coord(0)
                                )
                                .load(_LOAD_UNPACK16)
                                .to(tla.Float32, _CAST_NONE)
                            )
                            tfkw1 = (
                                tla.tile_view(
                                    ub_k_weight, tla.make_shape(64), tla.make_coord(1)
                                )
                                .load(_LOAD_UNPACK16)
                                .to(tla.Float32, _CAST_NONE)
                            )
                            tfqb0 = tfqw0
                            tfqb1 = tfqw1
                            tfkb0 = tfkw0
                            tfkb1 = tfkw1
                            if tla.const_expr(has_bias):
                                tfqb0 = (
                                    tla.tile_view(
                                        ub_q_bias,
                                        tla.make_shape(64),
                                        tla.make_coord(0),
                                    )
                                    .load(_LOAD_UNPACK16)
                                    .to(tla.Float32, _CAST_NONE)
                                )
                                tfqb1 = (
                                    tla.tile_view(
                                        ub_q_bias,
                                        tla.make_shape(64),
                                        tla.make_coord(1),
                                    )
                                    .load(_LOAD_UNPACK16)
                                    .to(tla.Float32, _CAST_NONE)
                                )
                                tfkb0 = (
                                    tla.tile_view(
                                        ub_k_bias,
                                        tla.make_shape(64),
                                        tla.make_coord(0),
                                    )
                                    .load(_LOAD_UNPACK16)
                                    .to(tla.Float32, _CAST_NONE)
                                )
                                tfkb1 = (
                                    tla.tile_view(
                                        ub_k_bias,
                                        tla.make_shape(64),
                                        tla.make_coord(1),
                                    )
                                    .load(_LOAD_UNPACK16)
                                    .to(tla.Float32, _CAST_NONE)
                                )
                            tfcos = (
                                tla.tile_view(
                                    cs_bufs[0],
                                    tla.make_shape(64),
                                    tla.make_coord(0),
                                )
                                .load(_LOAD_UNPACK16)
                                .to(tla.Float32, _CAST_NONE)
                            )
                            tfsin = (
                                tla.tile_view(
                                    cs_bufs[0],
                                    tla.make_shape(64),
                                    tla.make_coord(1),
                                )
                                .load(_LOAD_UNPACK16)
                                .to(tla.Float32, _CAST_NONE)
                            )
                            for tfq_head in tla.range(q_heads):
                                _norm_rope_full(
                                    row_bufs[0],
                                    q_bufs[0],
                                    tfq_head,
                                    tfq_head,
                                    tfqw0,
                                    tfqw1,
                                    tfqb0,
                                    tfqb1,
                                    tfcos,
                                    tfsin,
                                    head_dim,
                                    eps,
                                    t_all_lanes,
                                    t_one_vec,
                                    has_bias,
                                )
                            for tfk_head in tla.range(kv_heads):
                                _norm_rope_full(
                                    row_bufs[0],
                                    k_bufs[0],
                                    q_heads + tfk_head,
                                    tfk_head,
                                    tfkw0,
                                    tfkw1,
                                    tfkb0,
                                    tfkb1,
                                    tfcos,
                                    tfsin,
                                    head_dim,
                                    eps,
                                    t_all_lanes,
                                    t_one_vec,
                                    has_bias,
                                )
                        else:
                            for t_q_head in tla.range(q_heads):
                                _norm_head(
                                    row_bufs[0],
                                    q_bufs[0],
                                    t_q_head,
                                    t_q_head,
                                    t_qw_even,
                                    t_qw_odd,
                                    t_qb_even,
                                    t_qb_odd,
                                    head_dim,
                                    eps,
                                    t_all_lanes,
                                    t_one_vec,
                                    has_bias,
                                )
                            for t_k_head in tla.range(kv_heads):
                                _norm_head(
                                    row_bufs[0],
                                    k_bufs[0],
                                    q_heads + t_k_head,
                                    t_k_head,
                                    t_kw_even,
                                    t_kw_odd,
                                    t_kb_even,
                                    t_kb_odd,
                                    head_dim,
                                    eps,
                                    t_all_lanes,
                                    t_one_vec,
                                    has_bias,
                                )
                            tla.local_mem_bar(
                                tla.params.MemType.VEC_STORE,
                                tla.params.MemType.VEC_LOAD,
                            )
                            for t_rq_head in tla.range(q_heads):
                                _rope_head(
                                    q_bufs[0],
                                    t_rq_head,
                                    head_dim,
                                    half_rope,
                                    t_cos_even,
                                    t_cos_odd,
                                    t_sin_even,
                                    t_sin_odd,
                                )
                            for t_rk_head in tla.range(kv_heads):
                                _rope_head(
                                    k_bufs[0],
                                    t_rk_head,
                                    head_dim,
                                    half_rope,
                                    t_cos_even,
                                    t_cos_odd,
                                    t_sin_even,
                                    t_sin_odd,
                                )
                    tla.pipe_barrier(tla.pipes.ALL)

                    tla.copy(
                        tla.tile_view(gm_q, row_q_shape, tla.make_coord(t_row)),
                        q_bufs[0],
                    )
                    tla.copy(
                        tla.tile_view(gm_k, row_kv_shape, tla.make_coord(t_row)),
                        k_bufs[0],
                    )
                    tla.copy(
                        tla.tile_view(gm_v, row_kv_shape, tla.make_coord(t_row)),
                        tla.tile_view(
                            row_bufs[0], row_kv_shape, tla.make_coord(v_tile)
                        ),
                    )
                    tla.pipe_barrier(tla.pipes.ALL)

        tla.pipe_barrier(tla.pipes.ALL)


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------

_NPU_ARCH = "--npu-arch 3510"
_ARTIFACTS = {}

_TLA_DTYPE = {
    torch.bfloat16: tla.BFloat16,
    torch.float16: tla.Float16,
    torch.float32: tla.Float32,
    torch.int64: tla.Int64,
    torch.int32: tla.Int32,
}


_VECTOR_CORES: dict[int, int] = {}


def _vector_core_num(device) -> int:
    """Core count for ``device``.

    ``get_device_properties`` is a runtime query worth ~25 us per call, which is
    a fifth of the whole kernel at small token counts, so it is cached.
    """
    index = device.index if device.index is not None else torch.npu.current_device()
    cores = _VECTOR_CORES.get(index)
    if cores is None:
        cores = int(torch.npu.get_device_properties(index).vector_core_num)
        _VECTOR_CORES[index] = cores
    return cores


# Resolving the current stream through torch costs ~11 us per launch, which is
# 12% of the whole operator at small token counts. It is therefore resolved once
# per (thread, device) and reused.
#
# Caveat: a launch issued inside ``with torch.npu.stream(s)`` would keep using
# the stream that was current on the first call. Set SQR_STREAM_CACHE=0 to query
# every launch (correct for multi-stream callers, ~11 us slower), or call
# reset_stream_cache() after switching streams.
_STREAM_CACHE_ENABLED = os.environ.get("SQR_STREAM_CACHE", "1") != "0"
_STREAMS: dict[tuple[int, int], int] = {}


def reset_stream_cache() -> None:
    """Forget the cached launch stream (call after switching NPU streams)."""
    _STREAMS.clear()


def _launch_stream(device):
    """Cached launch stream handle, or None to let the runtime query it."""
    if not _STREAM_CACHE_ENABLED:
        return None
    index = device.index if device.index is not None else torch.npu.current_device()
    key = (threading.get_ident(), index)
    stream = _STREAMS.get(key)
    if stream is None:
        stream = int(torch.npu.current_stream(index).npu_stream)
        _STREAMS[key] = stream
    return stream


# The launch payload is just the packed argument blob (ten buffer addresses for
# this kernel). Packing it costs ~9.5 us per call and only depends on those
# addresses plus the dynamic extents they carry, so consecutive calls on the
# same buffers can reuse it.
_PAYLOADS: dict[tuple, tuple] = {}
_MAX_PAYLOADS = 256
_LAUNCH_ACCEPTS_PRINT: bool | None = None


def _launch_payload(
    execution_mod,
    function_handle,
    stream,
    block_num,
    payload,
    dyn_ub,
    module,
):
    """Call the versioned runtime launcher, which changed its kwargs in 2.2.1."""
    global _LAUNCH_ACCEPTS_PRINT
    if _LAUNCH_ACCEPTS_PRINT is None:
        parameters = inspect.signature(execution_mod.launch_kernel).parameters
        _LAUNCH_ACCEPTS_PRINT = "uses_scalar_print" in parameters
    kwargs = {
        "function_handle": function_handle,
        "stream": stream,
        "block_num": block_num,
        "payload": payload,
        "dyn_ubuf_bytes": dyn_ub,
    }
    if _LAUNCH_ACCEPTS_PRINT:
        kwargs["uses_scalar_print"] = getattr(module, "uses_scalar_print", False)
    execution_mod.launch_kernel(**kwargs)


def _resolve_launcher(artifact):
    """Return ``(executor, payload, dyn_ubuf_bytes)`` plumbing or None.

    Only the pieces ``JitCompiledFunction.__call__`` itself uses are reused; any
    surprise in the runtime internals disables the shortcut instead of failing.
    """
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
            executor,
            module.execution_args,
            _resolved_dyn_ubuf_bytes(module, 0),
            executor.function_handle,
            module,
        )
    except Exception:
        return None


def _launch(artifact, tensors, sigs, block_num, stream):
    """Launch ``artifact`` on ``tensors``, reusing the packed payload if possible."""
    key = None
    if stream is not None:
        # ``sigs`` carries (address, elements, dtype) per buffer: the element
        # count must be in the key because the caching allocator may hand the
        # same address to a differently sized request and a sliced caller can
        # keep one address while changing the row count. A stale payload would
        # read/write past the buffer or skip output rows.
        key = (id(artifact),) + sigs
        cached = _PAYLOADS.get(key)
        if cached is not None and cached[0] is artifact:
            execution_mod, function_handle, payload, dyn_ub, module = cached[1]
            _launch_payload(
                execution_mod,
                function_handle,
                stream,
                block_num,
                payload,
                dyn_ub,
                module,
            )
            return

    if key is not None:
        plumbing = _resolve_launcher(artifact)
        if plumbing is not None:
            execution_mod, executor, execution_args, dyn_ub, function_handle, module = (
                plumbing
            )
            try:
                payload = execution_args.generate_launch_payload(tensors)
            except Exception:
                payload = None
            if payload is not None:
                if len(_PAYLOADS) >= _MAX_PAYLOADS:
                    _PAYLOADS.clear()
                _PAYLOADS[key] = (
                    artifact,
                    (execution_mod, function_handle, payload, dyn_ub, module),
                )
                _launch_payload(
                    execution_mod,
                    function_handle,
                    stream,
                    block_num,
                    payload,
                    dyn_ub,
                    module,
                )
                return

    if stream is None:
        artifact(*tensors, block_num=block_num)
    else:
        artifact(*tensors, block_num=block_num, stream=stream)


_BINDINGS = {}
_ZERO_BIAS = {}
_MAX_BINDINGS = 4096


def _bind(tensor, sig=None):
    """Bind a contiguous torch buffer as a 1-D TLA host tensor with a dynamic row.

    ``from_dlpack`` costs ~70 us per tensor (capsule parse + ownership
    hand-off), which dwarfs the kernel at these sizes, so the host tensor is
    built directly from the buffer address. The caller keeps the torch tensor
    alive for the whole launch, so no DLPack ownership is needed. Every GM
    buffer is dynamic on purpose: the lowering shares one outlined copy helper
    per (dtype, layout), so mixing static and dynamic roots of the same dtype
    gives that helper two incompatible signatures.

    Bindings are memoized by ``(address, elements, dtype)`` (``sig`` when the
    caller already computed it). A binding carries no buffer ownership, so a
    cached entry stays valid when PyTorch hands the same address to a later
    allocation of the same size and dtype (the common case for the operator's
    output buffers).
    """
    if not tensor.is_contiguous():
        raise ValueError("kernel buffers must be contiguous")
    if sig is None:
        sig = (tensor.data_ptr(), tensor.numel(), tensor.dtype)
    cached = _BINDINGS.get(sig)
    if cached is not None:
        return cached
    dtype = _TLA_DTYPE.get(tensor.dtype)
    if dtype is None:
        view = from_dlpack(tensor, layout_tag=tla.arch.RowMajor)
        view = view.mark_compact_shape_dynamic(0)
    else:
        elements = tensor.numel()
        view = _Tensor(
            (elements,),
            dtype,
            origin_shape=(elements,),
            coord=(0,),
            stride=(1,),
            layout_tag=tla.arch.RowMajor,
            data_ptr=tensor.data_ptr(),
            flat_shape=(elements,),
            flat_stride=(1,),
        )
        view.mark_compact_shape_dynamic(0)
    if len(_BINDINGS) < _MAX_BINDINGS:
        _BINDINGS[sig] = view
    return view


def _zero_bias(head_dim, dtype, device):
    key = (head_dim, dtype, str(device))
    bias = _ZERO_BIAS.get(key)
    if bias is None:
        bias = torch.zeros(head_dim, dtype=dtype, device=device)
        _ZERO_BIAS[key] = bias
    return bias


def _get_artifact(
    q_hidden,
    kv_hidden,
    head_dim,
    rope_dim,
    eps,
    has_bias,
    rows_per_group,
    slots,
    fuse_rope,
    tensors,
):
    key = (
        q_hidden,
        kv_hidden,
        head_dim,
        rope_dim,
        float(eps),
        bool(has_bias),
        int(rows_per_group),
        int(slots),
        bool(fuse_rope),
    )
    artifact = _ARTIFACTS.get(key)
    if artifact is None:
        artifact = tla.compile(
            split_qkv_rmsnorm_rope_kernel,
            *tensors,
            q_hidden,
            kv_hidden,
            head_dim,
            rope_dim,
            float(eps),
            bool(has_bias),
            int(rows_per_group),
            int(slots),
            bool(fuse_rope),
            options=_NPU_ARCH,
        )
        _ARTIFACTS[key] = artifact
    return artifact


def split_qkv_rmsnorm_rope(
    input: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    eps: float,
    q_bias: torch.Tensor | None = None,
    k_bias: torch.Tensor | None = None,
    block_num: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total_hidden = q_hidden_size + 2 * kv_hidden_size
    rope_dim = cos_sin_cache.shape[-1]

    # Kernel contract. ``_rope_head`` addresses the rotated halves with tile
    # indices, so ``head_dim`` must split into whole ``rope_dim // 2`` tiles;
    # otherwise the head base would not be tile aligned and the host has to
    # reject the shape instead of silently reading the wrong lanes.
    # The kernel derives its row count from the input extent, so a padded or
    # narrower last dimension would make it write past (or short of) the
    # q/k/v outputs; both that and a mismatched positions length are rejected.
    if input.dim() != 2 or input.shape[1] != total_hidden:
        raise ValueError(f"input must be 2-D with last dim {total_hidden}")
    if positions.numel() != input.shape[0]:
        raise ValueError("positions must have one entry per input row")
    if input.dtype is not torch.bfloat16:
        raise ValueError("only bfloat16 input is supported")
    if head_dim != 128:
        raise ValueError("the Triton benchmark registers head_dim == 128 only")
    if rope_dim <= 0 or rope_dim % 2 or rope_dim > head_dim:
        raise ValueError("rope_dim must be positive, even and <= head_dim")
    if head_dim % (rope_dim // 2):
        raise ValueError("head_dim must be a multiple of rope_dim // 2")
    if q_hidden_size % kv_hidden_size or kv_hidden_size % head_dim:
        raise ValueError("kv_hidden_size must divide q_hidden_size")
    if (q_bias is None) != (k_bias is None):
        raise ValueError("q_bias and k_bias must be provided together")
    num_rows = input.shape[0]
    has_bias = q_bias is not None

    q_output = torch.empty(
        num_rows, q_hidden_size, dtype=input.dtype, device=input.device
    )
    k_output = torch.empty(
        num_rows, kv_hidden_size, dtype=input.dtype, device=input.device
    )
    v_output = torch.empty(
        num_rows, kv_hidden_size, dtype=input.dtype, device=input.device
    )
    if num_rows == 0:
        return q_output, k_output, v_output

    if has_bias:
        q_bias_arg, k_bias_arg = q_bias, k_bias
    else:
        zero_bias = _zero_bias(head_dim, input.dtype, input.device)
        q_bias_arg, k_bias_arg = zero_bias, zero_bias

    # Rows are runtime: input / positions carry a dynamic first extent, the
    # outputs and the rotary cache are fully described by their static shapes.
    # Buffers are passed as-is: the binding only reads address / elements /
    # dtype, so the eight ``reshape`` calls this used to make were pure cost.
    buffers = (
        input,
        cos_sin_cache,
        positions,
        q_weight,
        k_weight,
        q_bias_arg,
        k_bias_arg,
        q_output,
        k_output,
        v_output,
    )
    # One signature pass feeds both caches: the binding cache and the payload
    # cache are keyed by the same (address, elements, dtype) triple, so walking
    # all ten tensors twice would double the per-call host cost.
    sigs = tuple((t.data_ptr(), t.numel(), t.dtype) for t in buffers)
    bound = [_bind(tensor, sig) for tensor, sig in zip(buffers, sigs)]
    input_t, cache_t, positions_t, qw_t, kw_t, qb_t, kb_t, q_t, k_t, v_t = bound

    tensors = [input_t, cache_t, positions_t, qw_t, kw_t, qb_t, kb_t, q_t, k_t, v_t]
    slots = default_slots(q_hidden_size, kv_hidden_size)
    rows_per_group = default_rows_per_group(q_hidden_size, kv_hidden_size, slots)
    fuse_rope = rope_dim == head_dim and num_rows >= _FUSED_ROPE_MIN_ROWS
    artifact = _get_artifact(
        q_hidden_size,
        kv_hidden_size,
        head_dim,
        rope_dim,
        eps,
        has_bias,
        rows_per_group,
        slots,
        fuse_rope,
        tensors,
    )
    stream = _launch_stream(input.device)
    cores = _vector_core_num(input.device) if block_num is None else int(block_num)
    _launch(
        artifact,
        tensors,
        sigs,
        cores,
        stream,
    )
    return q_output, k_output, v_output


# ---------------------------------------------------------------------------
# Golden and driver
# ---------------------------------------------------------------------------


def golden(
    input,
    cos_sin_cache,
    positions,
    q_weight,
    k_weight,
    q_hidden_size,
    kv_hidden_size,
    head_dim,
    eps,
    q_bias=None,
    k_bias=None,
):
    """Torch replica of the kernel's numeric path (fp32 math, bf16 rounding)."""
    tokens = input.shape[0]
    rope_dim = cos_sin_cache.shape[-1]
    half = rope_dim // 2
    q_heads = q_hidden_size // head_dim
    kv_heads = kv_hidden_size // head_dim

    x = input.float()
    q_in = x[:, :q_hidden_size].reshape(tokens, q_heads, head_dim)
    k_in = x[:, q_hidden_size : q_hidden_size + kv_hidden_size].reshape(
        tokens, kv_heads, head_dim
    )
    v_out = input[:, q_hidden_size + kv_hidden_size :]

    def rmsnorm(part, weight, bias):
        inv = 1.0 / torch.sqrt((part * part).mean(-1, keepdim=True) + eps)
        normed = part * inv * weight.float()
        if bias is not None:
            normed = normed + bias.float()
        return normed.to(torch.bfloat16)

    def rope(part):
        out = part.float().clone()
        cos = cos_sin_cache[positions, :half].float()[:, None, :]
        sin = cos_sin_cache[positions, half:].float()[:, None, :]
        x1 = part[:, :, :half].float()
        x2 = part[:, :, half:rope_dim].float()
        out[:, :, :half] = (x1 * cos - x2 * sin).to(torch.bfloat16).float()
        out[:, :, half:rope_dim] = (x2 * cos + x1 * sin).to(torch.bfloat16).float()
        return out.to(torch.bfloat16).reshape(tokens, -1)

    q = rope(rmsnorm(q_in, q_weight, q_bias))
    k = rope(rmsnorm(k_in, k_weight, k_bias))
    return q, k, v_out


def _make_inputs(args):
    torch.npu.manual_seed(args.seed)
    q_hidden = args.q_heads * args.head_dim
    kv_hidden = args.kv_heads * args.head_dim
    total = q_hidden + 2 * kv_hidden
    dtype = torch.bfloat16

    inputs = {
        "input": torch.randn(args.tokens, total, dtype=torch.float32).to(dtype).npu(),
        "positions": torch.randint(
            0, args.cache_rows, (args.tokens,), dtype=torch.int64, device="npu"
        ),
        "q_weight": torch.randn(args.head_dim, dtype=torch.float32).to(dtype).npu(),
        "k_weight": torch.randn(args.head_dim, dtype=torch.float32).to(dtype).npu(),
        "q_hidden_size": q_hidden,
        "kv_hidden_size": kv_hidden,
        "head_dim": args.head_dim,
        "eps": args.eps,
    }
    inv_freq = 1.0 / (
        1e6 ** (torch.arange(0, args.rope_dim, 2, dtype=torch.float32) / args.rope_dim)
    )
    angles = torch.outer(torch.arange(args.cache_rows, dtype=torch.float32), inv_freq)
    inputs["cos_sin_cache"] = (
        torch.cat((angles.cos(), angles.sin()), -1).to(dtype).npu()
    )
    if args.bias:
        inputs["q_bias"] = (
            torch.randn(args.head_dim, dtype=torch.float32).to(dtype).npu()
        )
        inputs["k_bias"] = (
            torch.randn(args.head_dim, dtype=torch.float32).to(dtype).npu()
        )
    return inputs


def run(args: argparse.Namespace) -> int:
    import torch_npu  # noqa: F401

    from helpers import get_block_num

    if args.rope_dim > args.head_dim or args.rope_dim % 2:
        raise SystemExit("--rope-dim must be even and <= --head-dim")
    torch.npu.set_device(args.device)

    inputs = _make_inputs(args)
    expected = golden(**inputs)
    block_num = get_block_num(args.block_num, args.device, kind="vector")
    got = split_qkv_rmsnorm_rope(**inputs, block_num=block_num)
    torch.npu.synchronize()

    q_hidden = inputs["q_hidden_size"]
    kv_hidden = inputs["kv_hidden_size"]
    print(
        f"--- tokens={args.tokens} q_heads={args.q_heads} kv_heads={args.kv_heads} "
        f"head_dim={args.head_dim} rope_dim={args.rope_dim} bias={args.bias} "
        f"blocks={block_num} ---"
    )

    passed = True
    for name, actual, want in zip("qkv", got, expected):
        diff = (actual.float() - want.float()).abs()
        within = diff <= 1e-3 + 2.0**-7 * want.float().abs()
        ratio = float(within.float().mean())
        passed = passed and ratio >= 0.999
        print(
            f"accuracy out={name} pass={ratio >= 0.999} matched={ratio:.6f} "
            f"max_abs={float(diff.max()):.6f}"
        )
    slots = default_slots(q_hidden, kv_hidden)
    rows_per_group = default_rows_per_group(q_hidden, kv_hidden, slots)
    print(
        f"validation pass={passed} "
        f"q={inputs['q_hidden_size']} kv={inputs['kv_hidden_size']} "
        f"slots={slots} rows_per_group={rows_per_group}"
    )
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fused QKV split + RMSNorm + RoPE (pure AIV)."
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument(
        "--rope-dim", type=int, default=128, help="rotary width, even and <= head_dim"
    )
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--bias", action="store_true", help="apply q_bias / k_bias")
    parser.add_argument("--cache-rows", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--block-num",
        type=int,
        default=-1,
        help="Launch block count; -1 = full vector_core_num (AIV)",
    )
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
