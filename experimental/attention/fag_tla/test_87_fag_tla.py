# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.


import os
import sys

_CATLASS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_OPTEST_ROOT = os.path.join(_CATLASS_ROOT, "tests", "optest")
for _p in (os.path.join(_OPTEST_ROOT, "tests"), _OPTEST_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest
import torch
import torch_npu

from common import only_on_2201


def _reference_forward(q, k, v, scale):
    """CPU fp32 full attention forward on a single batch: returns (out, lse).

    q/k/v: (L, H, D) / (S, H, D) / (S, H, D) fp32.
    """
    scores = torch.einsum("lhd,shd->lhs", q, k) * scale  # (L, H, S)
    m = scores.max(dim=-1, keepdim=True).values
    probs = torch.exp(scores - m)
    attn = probs / probs.sum(dim=-1, keepdim=True)
    out = torch.einsum("lhs,shd->lhd", attn, v)  # (L, H, D)
    lse = m.squeeze(-1) + torch.log(probs.sum(dim=-1))  # (L, H)
    return out, lse


def _reference_fag_backward(dout, q, k, v, q_lens, kv_lens):
    """Per-batch CPU fp32 attention backward: returns (dq, dk, dv) fp32.

    Mirrors the FAG kernel semantics: scale = 1/sqrt(D), no mask, no dropout.
    """
    scale = 1.0 / (q.shape[-1] ** 0.5)
    dqs, dks, dvs = [], [], []
    start_q, start_k = 0, 0
    for i in range(len(q_lens)):
        q_len = int(q_lens[i])
        kv_len = int(kv_lens[i])
        qb = q[start_q : start_q + q_len].float().requires_grad_()
        kb = k[start_k : start_k + kv_len].float().requires_grad_()
        vb = v[start_k : start_k + kv_len].float().requires_grad_()
        dob = dout[start_q : start_q + q_len].float()

        scores = torch.einsum("lhd,shd->lhs", qb, kb) * scale
        attn = torch.softmax(scores, dim=-1)
        ob = torch.einsum("lhs,shd->lhd", attn, vb)
        ob.backward(dob)

        dqs.append(qb.grad)
        dks.append(kb.grad)
        dvs.append(vb.grad)
        start_q += q_len
        start_k += kv_len

    return torch.cat(dqs, 0), torch.cat(dks, 0), torch.cat(dvs, 0)


def _check_close(res, ref, name, rtol=5e-2, atol=5e-2):
    """Dual-criterion check mirroring the reference verify_result.py.

    C1: mean/max relative error below threshold; C2: error ratio <= 0.15.
    """
    res_f = res.float()
    ref_f = ref.float()
    rel = (res_f - ref_f).abs() / (ref_f.abs() + 1e-7)
    mere = rel.mean().item()
    mare = rel.max().item()
    diff = (res_f - ref_f).abs()
    err_ratio = (diff > atol + rtol * ref_f.abs()).float().mean().item()

    c1 = mere < 1e-3 and mare < 1e-2
    c2 = err_ratio <= 0.15
    print(f"[{name}] mere={mere:.6f} mare={mare:.6f} err_ratio={err_ratio:.4f} c1={c1} c2={c2}")
    return c1 or c2


@only_on_2201
def test_fag_tla_backward():
    """Compare FAG (flash attention gradient) against a PyTorch reference backward."""
    import torch_catlass

    torch.manual_seed(2)
    batch = 2
    q_lens = [128, 192]
    kv_lens = [128, 192]
    num_heads = 4
    kv_heads = 4  # MHA in the first cut
    head_dim = 128

    total_q = sum(q_lens)
    total_kv = sum(kv_lens)

    q = torch.randn(total_q, num_heads, head_dim, dtype=torch.float16)
    k = torch.randn(total_kv, kv_heads, head_dim, dtype=torch.float16)
    v = torch.randn(total_kv, kv_heads, head_dim, dtype=torch.float16)
    dout = torch.randn(total_q, num_heads, head_dim, dtype=torch.float16)

    # Prefix-sum (batch-size, not batch+1) tables used by the FAG kernel.
    cu_q = torch.cumsum(torch.tensor(q_lens, dtype=torch.int64), dim=0)
    cu_k = torch.cumsum(torch.tensor(kv_lens, dtype=torch.int64), dim=0)

    # Forward reference on CPU to obtain out and softmax_lse.
    scale = 1.0 / (head_dim**0.5)
    out_parts, lse_parts = [], []
    start_q, start_k = 0, 0
    for i in range(batch):
        qb = q[start_q : start_q + q_lens[i]].float()
        kb = k[start_k : start_k + kv_lens[i]].float()
        vb = v[start_k : start_k + kv_lens[i]].float()
        ob, lb = _reference_forward(qb, kb, vb, scale)
        out_parts.append(ob)
        lse_parts.append(lb)
        start_q += q_lens[i]
        start_k += kv_lens[i]
    out_ref = torch.cat(out_parts, 0)
    # The FAG kernel (TND) reads softmax_lse with per-batch head-major layout
    # [b][head][s1_b] (1 float per row). h == 1 makes token-major and head-major
    # identical, so the mismatch only shows up with num_heads > 1.
    softmax_lse_ref = torch.cat(
        [lb.transpose(0, 1).contiguous().view(-1) for lb in lse_parts], 0
    )

    # ---- move inputs to NPU ----
    dout_npu = dout.to("npu")
    q_npu = q.to("npu")
    k_npu = k.to("npu")
    v_npu = v.to("npu")
    out_npu = out_ref.to(torch.float16).to("npu")
    softmax_lse_npu = softmax_lse_ref.to("npu")
    cu_q_npu = cu_q.to("npu")
    cu_k_npu = cu_k.to("npu")

    dq, dk, dv = torch_catlass.fag_tla(
        dout_npu,
        q_npu,
        k_npu,
        v_npu,
        out_npu,
        softmax_lse_npu,
        cu_q_npu,
        cu_k_npu,
        num_heads,
        kv_heads,
        is_deterministic=True,
    )

    assert dq.shape == q.shape
    assert dk.shape == k.shape
    assert dv.shape == v.shape
    assert dq.dtype == torch.float16
    assert dq.device.type == "npu"

    dq_ref, dk_ref, dv_ref = _reference_fag_backward(dout, q, k, v, q_lens, kv_lens)

    ok_dq = _check_close(dq.cpu(), dq_ref, "dq")
    ok_dk = _check_close(dk.cpu(), dk_ref, "dk")
    ok_dv = _check_close(dv.cpu(), dv_ref, "dv")
    torch.npu.synchronize()
    assert ok_dq and ok_dk and ok_dv, "FAG backward not close to reference"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])