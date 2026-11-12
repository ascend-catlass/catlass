# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CPU FP32 semantic reference and BF16 output checks for BlockMeanAttentionScore."""

from __future__ import annotations

import math

import torch

from .constants import BLOCK_TOKENS


@torch.no_grad()
def block_mean_attention_score_golden_cpu(
    q: torch.Tensor,
    k: torch.Tensor,
    variable_block_size_16: torch.Tensor,
    *,
    query_chunk_size: int = 256,
) -> torch.Tensor:
    """Return FP32 ``[H, B, B]`` mean probabilities on the CPU.

    Counts describe a valid prefix of each 16-slot block, shared by Q and K.
    Softmax covers every valid Key in the sequence. Query averaging precedes
    Key averaging; a pair involving either empty block returns NaN.

    Only ``query_chunk_size * S`` scores are materialized at once per head.
    Set the chunk size to S for a full-matrix reference. Inputs
    on another device are copied to CPU before reference computation.
    """
    if q.ndim != 3 or q.shape != k.shape:
        raise ValueError("q and k must have the same [H, S, D] shape")
    heads, slots, dim = q.shape
    if heads < 1 or slots < BLOCK_TOKENS or slots % BLOCK_TOKENS or dim < 1:
        raise ValueError(
            f"H and D must be positive; S must be a positive multiple of {BLOCK_TOKENS}"
        )
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise TypeError("q and k must be bfloat16")
    blocks = slots // BLOCK_TOKENS
    if variable_block_size_16.shape != (blocks,):
        raise ValueError(f"counts must have shape [S // {BLOCK_TOKENS}]")
    if variable_block_size_16.dtype != torch.int32:
        raise TypeError("counts must be int32")
    if (
        isinstance(query_chunk_size, bool)
        or not isinstance(query_chunk_size, int)
        or query_chunk_size < BLOCK_TOKENS
        or query_chunk_size % BLOCK_TOKENS
    ):
        raise ValueError(
            f"query_chunk_size must be a positive multiple of {BLOCK_TOKENS}"
        )

    counts = variable_block_size_16.detach().cpu()
    if bool(((counts < 0) | (counts > BLOCK_TOKENS)).any()):
        raise ValueError(f"counts must be in [0, {BLOCK_TOKENS}]")
    counts_f32 = counts.float()
    valid = (torch.arange(BLOCK_TOKENS)[None, :] < counts[:, None]).reshape(slots)
    q_f32 = q.detach().to(device="cpu", dtype=torch.float32)
    k_f32 = k.detach().to(device="cpu", dtype=torch.float32)
    output = torch.empty((heads, blocks, blocks), dtype=torch.float32)
    scale = 1.0 / math.sqrt(dim)

    for head in range(heads):
        for start in range(0, slots, query_chunk_size):
            stop = min(start + query_chunk_size, slots)
            first_block, stop_block = start // BLOCK_TOKENS, stop // BLOCK_TOKENS
            pair_valid = valid[start:stop, None] & valid[None, :]
            scores = torch.matmul(q_f32[head, start:stop], k_f32[head].T)
            scores.mul_(scale).masked_fill_(~pair_valid, float("-inf"))
            probability = scores.softmax(dim=-1)
            probability.masked_fill_(~pair_valid, 0.0)

            block_probability = probability.reshape(
                stop_block - first_block, BLOCK_TOKENS, blocks, BLOCK_TOKENS
            )
            query_mean = block_probability.sum(dim=1)
            query_mean.div_(counts_f32[first_block:stop_block, None, None])
            output[head, first_block:stop_block] = (
                query_mean.sum(dim=-1) / counts_f32[None, :]
            )

    return output


@torch.no_grad()
def bf16_output_rounding_bound(expected_fp32: torch.Tensor) -> torch.Tensor:
    """Return the local BF16 half-ULP allowance as a CPU FP64 tensor.

    BF16 has eight significand bits and a smallest positive subnormal of
    ``2**-133``. Nonfinite reference elements have zero allowance and must be
    checked separately. Keeping the reference in FP32 preserves the oracle;
    FP64 here only makes the validation bound and its propagation precise.
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
    rtol: float = 1e-4,
    atol: float = 1e-7,
) -> float:
    """Check BF16 output using the FP32 tolerance plus final rounding only.

    Shape and NaN positions must agree exactly, as must infinity positions and
    signs. Finite elements use ``atol + rtol * abs(expected)`` plus the local
    BF16 half ULP. No conversion or approximation is applied to the FP32
    oracle. Validation runs on CPU and returns the largest finite absolute
    error, or zero if there are no finite elements.
    """
    if not isinstance(actual, torch.Tensor) or actual.dtype != torch.bfloat16:
        raise TypeError("actual must be a bfloat16 tensor")
    if (
        not isinstance(expected_fp32, torch.Tensor)
        or expected_fp32.dtype != torch.float32
    ):
        raise TypeError("expected_fp32 must be a float32 tensor")
    if actual.shape != expected_fp32.shape:
        raise AssertionError(
            f"Output shapes differ: {actual.shape} != {expected_fp32.shape}"
        )
    if not math.isfinite(rtol) or not math.isfinite(atol) or rtol < 0 or atol < 0:
        raise ValueError("rtol and atol must be finite and nonnegative")

    observed = actual.detach().to(device="cpu", dtype=torch.float64)
    expected = expected_fp32.detach().to(device="cpu", dtype=torch.float64)
    if not torch.equal(torch.isnan(observed), torch.isnan(expected)):
        raise AssertionError("Output NaN positions differ from the FP32 reference")
    expected_inf = torch.isinf(expected)
    if not torch.equal(torch.isinf(observed), expected_inf):
        raise AssertionError("Output infinity positions differ from the FP32 reference")
    if not torch.equal(
        torch.signbit(observed[expected_inf]), torch.signbit(expected[expected_inf])
    ):
        raise AssertionError("Output infinity signs differ from the FP32 reference")

    finite = torch.isfinite(expected)
    if not bool(torch.isfinite(observed[finite]).all()):
        raise AssertionError("Nonfinite output at a finite FP32 reference element")
    if not bool(finite.any()):
        return 0.0
    error = (observed[finite] - expected[finite]).abs()
    allowance = atol + rtol * expected[finite].abs()
    allowance = allowance + bf16_output_rounding_bound(expected_fp32)[finite]
    failed = error > allowance
    if bool(failed.any()):
        worst = torch.argmax(error - allowance)
        raise AssertionError(
            "BF16 output exceeds FP32 arithmetic tolerance plus final BF16 rounding: "
            f"actual={observed[finite][worst].item():.9g}, "
            f"expected={expected[finite][worst].item():.9g}, "
            f"abs_error={error[worst].item():.9g}, allowance={allowance[worst].item():.9g}"
        )
    return error.max().item()
