# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.


import torch
from torch import Tensor


def fag_tla(
    dout: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    out: Tensor,
    softmax_lse: Tensor,
    actual_seq_qlen: Tensor,
    actual_seq_kvlen: Tensor,
    num_heads: int,
    num_key_value_heads: int,
    is_deterministic: bool = True,
):
    """Run CATLASS flash attention gradient (FAG) on NPU tensors (TLA style).

    Source: example 87_fag_tla.

    Computes the backward pass of flash attention: given the forward cached
    results (``out`` and ``softmax_lse``) and the output gradient ``dout``,
    returns ``(dq, dk, dv)``.

    Args:
        dout: Output gradient in TND layout ``(total_q_tokens, num_heads, head_dim)``.
        query: Query tensor ``(total_q_tokens, num_heads, head_dim)``.
        key: Key tensor ``(total_kv_tokens, num_key_value_heads, head_dim)``.
        value: Value tensor ``(total_kv_tokens, num_key_value_heads, v_head_dim)``.
        out: Forward output tensor, same shape as ``query``.
        softmax_lse: Forward log-sum-exp, fp32, per-batch head-major flat layout
            ``[batch][num_heads][s1_b]`` (as read by the FAG TND kernel).
        actual_seq_qlen: Prefix-sum (cumsum) of per-batch Q lengths, int64.
        actual_seq_kvlen: Prefix-sum (cumsum) of per-batch KV lengths, int64.
        num_heads: Number of query heads.
        num_key_value_heads: Number of KV heads.
        is_deterministic: Use deterministic DQKV path when ``True``.

    Returns:
        Tuple ``(dq, dk, dv)`` with the same shapes as ``query``, ``key``, ``value``.
    """
    return torch.ops.catlass.fag_tla(
        dout,
        query,
        key,
        value,
        out,
        softmax_lse,
        actual_seq_qlen,
        actual_seq_kvlen,
        num_heads,
        num_key_value_heads,
        is_deterministic,
    )