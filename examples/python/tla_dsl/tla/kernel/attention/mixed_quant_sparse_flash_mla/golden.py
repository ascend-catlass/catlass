# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CPU FP32 semantic reference and BF16 output checks for MixedQuantSparseFlashMla.

The reference decodes the same 608-byte quantised KV records as the kernel, builds
each query row's visible column set (dense window, compressed top-k, or explicit
index list), and evaluates softmax attention in FP32 on the CPU. Outputs stay in
FP32; ``assert_bf16_output_matches_fp32`` adds the local BF16 rounding allowance
when comparing with the kernel's BF16 result.
"""

from __future__ import annotations

import math

import torch

REC_BYTES = 608
ROPE_DIM = 64
NOPE_DIM = 448
HEAD_DIM = 512
SCALE_STRIDE = 8  # seven per-group scales plus one pad element


def _decode_records(records: torch.Tensor) -> torch.Tensor:
    """Return FP32 ``[rows, 512]`` latent KV from 608-byte quantised records.

    A record holds 64 BF16 RoPE values, 448 FP8 E4M3 latent values covered by
    seven per-token-group scales, and padding. Output columns are
    ``[dequantised latent (448) | RoPE (64)]``.
    """
    raw = records.detach().reshape(-1, REC_BYTES).to(device="cpu")
    latent = (
        raw[:, 128:576]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .float()
        .reshape(-1, NOPE_DIM // 64, 64)
    )
    scale = (
        raw[:, 576:590]
        .contiguous()
        .view(torch.bfloat16)
        .float()
        .reshape(-1, NOPE_DIM // 64, 1)
    )
    rope = raw[:, :128].contiguous().view(torch.bfloat16).float()
    return torch.cat(((latent * scale).reshape(-1, NOPE_DIM), rope), dim=-1)


def _layout_rows(kv: torch.Tensor, block_table, layout: str) -> torch.Tensor:
    """Map ``(batch, position)`` to a flat row index in the KV pool."""
    if layout == "PA_BBND":
        if block_table is None:
            raise ValueError("PA_BBND requires a block table")
        pages, page_rows = int(kv.shape[0]), int(kv.shape[1])
        table = block_table.detach().to(device="cpu", dtype=torch.long)

        def rows(batch: int, positions: torch.Tensor) -> torch.Tensor:
            page = table[batch][positions // page_rows].clamp_min(0)
            return page * page_rows + positions % page_rows

        return rows, pages * page_rows
    if layout == "BSND":
        length = int(kv.shape[1])

        def rows(batch: int, positions: torch.Tensor) -> torch.Tensor:
            return batch * length + positions

        return rows, int(kv.shape[0]) * length
    length = int(kv.shape[0])

    def rows(_batch: int, positions: torch.Tensor) -> torch.Tensor:
        return positions

    return rows, length


def _window(start: int, stop: int, total: int) -> torch.Tensor:
    start = max(0, start)
    stop = min(stop, total)
    if stop <= start:
        return torch.empty(0, dtype=torch.long)
    return torch.arange(start, stop, dtype=torch.long)


def _len_of(seqused, cu, batch: int, fallback: int) -> int:
    if seqused is not None:
        return int(seqused.reshape(-1)[batch])
    if cu is not None:
        cu = cu.reshape(-1)
        return int(cu[batch + 1] - cu[batch])
    return fallback


@torch.no_grad()
def mixed_quant_sparse_flash_mla_golden_cpu(
    q: torch.Tensor,
    quant_mode: int,
    ori_kv: torch.Tensor,
    cmp_kv: torch.Tensor | None = None,
    ori_sparse_indices: torch.Tensor | None = None,
    cmp_sparse_indices: torch.Tensor | None = None,
    ori_block_table: torch.Tensor | None = None,
    cmp_block_table: torch.Tensor | None = None,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_ori_kv: torch.Tensor | None = None,
    cu_seqlens_cmp_kv: torch.Tensor | None = None,
    seqused_q: torch.Tensor | None = None,
    seqused_ori_kv: torch.Tensor | None = None,
    seqused_cmp_kv: torch.Tensor | None = None,
    cmp_residual_kv: torch.Tensor | None = None,
    ori_topk_length: torch.Tensor | None = None,
    cmp_topk_length: torch.Tensor | None = None,
    sinks: torch.Tensor | None = None,
    rope_head_dim: int | None = None,
    softmax_scale: float | None = None,
    cmp_ratio: int = 1,
    ori_mask_mode: int = 0,
    cmp_mask_mode: int = 0,
    ori_win_left: int = -1,
    ori_win_right: int = -1,
    layout_q: str = "BSND",
    layout_kv: str = "BSND",
    topk_value_mode: int = 1,
):
    """Return FP32 ``(out, lse)`` for one call.

    ``out`` is ``[T1, N1, 512]`` and ``lse`` is ``[T1, N1]``, both on CPU. Only
    ``quant_mode=1`` (608-byte records) is modelled. The visible column set of a
    query row follows ``ori_mask_mode`` (0 = full, 3 = right-down causal,
    4 = sliding window) plus the compressed top-k / explicit index lists; sinks
    join the softmax denominator but contribute no value column.
    """
    if int(quant_mode) != 1:
        raise NotImplementedError("the golden reference models quant_mode=1 only")
    if int(cmp_ratio) < 1:
        raise ValueError("cmp_ratio must be positive")
    batch = (
        int(q.shape[0])
        if q.ndim == 4
        else (
            int(cu_seqlens_q.reshape(-1).numel()) - 1 if cu_seqlens_q is not None else 1
        )
    )
    if layout_q == "BSND":
        q_rows = (
            q.detach()
            .to(device="cpu", dtype=torch.float32)
            .reshape(-1, q.shape[-2], HEAD_DIM)
        )
        q_len = int(q.shape[1])
    else:
        q_rows = (
            q.detach()
            .to(device="cpu", dtype=torch.float32)
            .reshape(-1, q.shape[-2], HEAD_DIM)
        )
        q_len = q_rows.shape[0] // batch
    heads = int(q_rows.shape[1])
    total = q_rows.shape[0]
    scale = 1.0 / math.sqrt(HEAD_DIM) if softmax_scale is None else float(softmax_scale)
    if rope_head_dim is not None and int(rope_head_dim) != ROPE_DIM:
        raise NotImplementedError("only rope_head_dim=64 is implemented")

    ori = _decode_records(ori_kv)
    ori_rows, _ = _layout_rows(ori_kv, ori_block_table, layout_kv)
    cmp_rows = None
    cmp = None
    if cmp_kv is not None:
        cmp = _decode_records(cmp_kv)
        cmp_rows, _ = _layout_rows(cmp_kv, cmp_block_table, layout_kv)
    kv_all = torch.cat((ori, cmp), dim=0) if cmp is not None else ori

    sink = (
        sinks.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        if sinks is not None
        else None
    )
    out = torch.empty((total, heads, HEAD_DIM), dtype=torch.float32)
    lse = torch.empty((total, heads), dtype=torch.float32)

    ori_idx_cpu = (
        ori_sparse_indices.detach().to(device="cpu")
        if ori_sparse_indices is not None
        else None
    )
    cmp_idx_cpu = (
        cmp_sparse_indices.detach().to(device="cpu")
        if cmp_sparse_indices is not None
        else None
    )
    ori_len_all = (
        ori_topk_length.detach().to(device="cpu").reshape(-1)
        if ori_topk_length is not None
        else None
    )
    cmp_len_all = (
        cmp_topk_length.detach().to(device="cpu").reshape(-1)
        if cmp_topk_length is not None
        else None
    )

    for row in range(total):
        seq = row // q_len
        pos = row % q_len
        cur_q = _len_of(seqused_q, cu_seqlens_q, seq, q_len)
        n_ori = _len_of(seqused_ori_kv, cu_seqlens_ori_kv, seq, ori.shape[0])
        if ori_idx_cpu is not None:
            ids = (
                ori_idx_cpu[row][: int(ori_len_all[row])]
                if ori_len_all is not None
                else ori_idx_cpu[row]
            )
            ids = ids[ids >= 0].long()
        elif int(ori_mask_mode) == 0:
            ids = _window(0, n_ori, n_ori)
        elif int(ori_mask_mode) == 3:
            ids = _window(0, pos + (n_ori - cur_q) + 1, n_ori)
        else:
            end = pos + (n_ori - cur_q) + 1 + int(ori_win_right)
            ids = _window(
                end - (int(ori_win_left) + 1) - int(ori_win_right), end, n_ori
            )
        if layout_kv == "TND" and cu_seqlens_ori_kv is not None:
            ids = ids + int(cu_seqlens_ori_kv.reshape(-1)[seq])
        columns = [ori_rows(seq, ids)]
        if cmp is not None:
            if cmp_idx_cpu is not None:
                keep = (
                    cmp_idx_cpu[row][: int(cmp_len_all[row])]
                    if cmp_len_all is not None
                    else cmp_idx_cpu[row]
                )
                keep = keep[keep >= 0].long()
            elif int(cmp_mask_mode) == 0:
                keep = _window(
                    0,
                    _len_of(seqused_cmp_kv, cu_seqlens_cmp_kv, seq, cmp.shape[0]),
                    cmp.shape[0],
                )
            else:
                keep = _window(
                    0,
                    _len_of(seqused_cmp_kv, cu_seqlens_cmp_kv, seq, cmp.shape[0]),
                    cmp.shape[0],
                )
            if layout_kv == "TND" and cu_seqlens_cmp_kv is not None:
                keep = keep + int(cu_seqlens_cmp_kv.reshape(-1)[seq])
            columns.append(cmp_rows(seq, keep) + ori.shape[0])
        rows = torch.cat(columns) if len(columns) > 1 else columns[0]
        if rows.numel() == 0:
            raise ValueError("a query row has no visible KV column")
        keys = kv_all.index_select(0, rows)
        logits = q_rows[row] @ keys.T * scale
        if sink is not None:
            logits = torch.cat((logits, sink[:, None]), dim=-1)
        lse[row] = torch.logsumexp(logits, dim=-1)
        prob = logits.softmax(dim=-1)
        if sink is not None:
            prob = prob[:, :-1]
        out[row] = prob @ keys
    return out, lse


@torch.no_grad()
def bf16_output_rounding_bound(expected_fp32: torch.Tensor) -> torch.Tensor:
    """Return the local BF16 half-ULP allowance as a CPU FP64 tensor.

    BF16 has eight significand bits and a smallest positive subnormal of
    ``2**-133``. Nonfinite reference elements get zero allowance and must be
    checked separately.
    """
    if (
        not isinstance(expected_fp32, torch.Tensor)
        or expected_fp32.dtype != torch.float32
    ):
        raise TypeError("expected_fp32 must be a float32 tensor")
    expected = expected_fp32.detach().to(device="cpu", dtype=torch.float64)
    finite = torch.isfinite(expected)
    magnitude = torch.where(finite, expected.abs(), 0.0)
    _, exponent = torch.frexp(magnitude)
    half_ulp_exponent = torch.where(magnitude == 0, -134, exponent - 9).clamp_min(-134)
    allowance = torch.ldexp(torch.ones_like(expected), half_ulp_exponent)
    return torch.where(finite, allowance, 0.0)


@torch.no_grad()
def assert_bf16_output_matches_fp32(
    actual: torch.Tensor,
    expected_fp32: torch.Tensor,
    *,
    actual_lse: torch.Tensor | None = None,
    expected_lse: torch.Tensor | None = None,
    atol: float = 2e-2,
    rtol: float = 2e-2,
    lse_atol: float = 2e-3,
    lse_rtol: float = 2e-3,
) -> None:
    """Raise ``AssertionError`` when ``actual`` exceeds the allowed error.

    ``actual`` is the kernel output (BF16 or FP32). The allowance is the local
    half-ULP of the FP32 reference plus ``atol + rtol * |reference|``: the PV
    product accumulates up to a few thousand BF16 keys in a Cube/Vector order
    that differs from the CPU reference, so the same ``atol/rtol`` pair as the
    operator's own accuracy harness is used. LSE is compared in FP32.
    """
    got = actual.detach().to(device="cpu", dtype=torch.float64)
    ref = expected_fp32.detach().to(device="cpu", dtype=torch.float64)
    if got.shape != ref.shape:
        raise AssertionError(
            f"shape mismatch: {tuple(got.shape)} vs {tuple(ref.shape)}"
        )
    bound = bf16_output_rounding_bound(expected_fp32) + atol + rtol * ref.abs()
    bad = (~torch.isfinite(got)) | (torch.abs(got - ref) > bound)
    if bool(bad.any()):
        raise AssertionError(f"{int(bad.sum())} output elements exceed the BF16 bound")
    if actual_lse is not None and expected_lse is not None:
        got_l = actual_lse.detach().to(device="cpu", dtype=torch.float64)
        ref_l = expected_lse.detach().to(device="cpu", dtype=torch.float64)
        ok = torch.isclose(got_l, ref_l, atol=lse_atol, rtol=lse_rtol)
        if not bool(ok.all()):
            raise AssertionError(
                f"{int((~ok).sum())} LSE elements exceed atol={lse_atol}, rtol={lse_rtol}"
            )
