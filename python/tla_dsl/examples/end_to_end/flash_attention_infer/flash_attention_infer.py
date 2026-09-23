# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import catlass.tla as tla
from catlass.tla.runtime import from_dlpack

from kernel import flash_attention_infer_kernel

import argparse


@dataclass
class TileMaskParams:
    """FA kernel 所需的 tile mask 输入格式"""

    tile_range: torch.Tensor  # (B, Tq) int32 — 每 Q tile 的 KV 循环上界
    tile_compute_bp: torch.Tensor  # (B, Tq, Wk) int32 — bit=1: 该 tile 有可见元素
    fine_mask_bp: torch.Tensor  # (B, Tq, Wk) int32 — bit=1: 需行级右边界 mask
    maskr: torch.Tensor  # (B, Sq) int32 — 行右边界(k >= maskr 被 mask)


def generate_fa_mask(mask_mode, batch, seq_q, seq_k, tile_m=128, tile_n=128):
    Tq = (seq_q + tile_m - 1) // tile_m
    Tk = (seq_k + tile_n - 1) // tile_n
    Wk = (Tk + 31) // 32

    if mask_mode == "none":
        maskr_1d = torch.full((seq_q,), seq_k, dtype=torch.int32)
        tile_range = torch.full((batch, Tq), Tk, dtype=torch.int32)
        tile_compute_bp = torch.zeros((batch, Tq, Wk), dtype=torch.int32)
        fine_mask_bp = torch.zeros((batch, Tq, Wk), dtype=torch.int32)
        for kt in range(Tk):
            tile_compute_bp[:, :, kt // 32] |= 1 << (kt % 32)
        dense = torch.ones((batch, seq_q, seq_k), dtype=torch.bool)
    else:  # causal
        prefix = seq_k - seq_q
        # 行右边界(不含): 列 >= maskr 的被遮蔽
        maskr_1d = (torch.arange(seq_q) + prefix + 1).clamp(max=seq_k).to(torch.int32)
        dense_2d = torch.arange(seq_k).unsqueeze(0) < maskr_1d.unsqueeze(1)  # (Sq, Sk)
        dense = dense_2d.unsqueeze(0).expand(batch, -1, -1).contiguous()

        tile_range = torch.zeros((batch, Tq), dtype=torch.int32)
        tile_compute_bp = torch.zeros((batch, Tq, Wk), dtype=torch.int32)
        fine_mask_bp = torch.zeros((batch, Tq, Wk), dtype=torch.int32)
        for qt in range(Tq):
            q_start = qt * tile_m
            q_end = min(q_start + tile_m, seq_q)
            # 该 Q tile 的最大可见列(最宽行的右边界)
            max_vis = min(int(maskr_1d[q_end - 1]), seq_k)
            tr = (max_vis + tile_n - 1) // tile_n  # 最后一个有可见元素的 tile
            tile_range[:, qt] = tr
            for kt in range(tr):
                tile_compute_bp[:, qt, kt // 32] |= 1 << (kt % 32)
                # 精细 mask: tile 跨越因果边界(最窄行看不全此 tile)
                k_end = (kt + 1) * tile_n
                min_vis = int(maskr_1d[q_start])  # 最窄行右边界
                if k_end > min_vis:
                    fine_mask_bp[:, qt, kt // 32] |= 1 << (kt % 32)

    maskr = maskr_1d.unsqueeze(0).expand(batch, seq_q).contiguous()
    tm = TileMaskParams(
        tile_range=tile_range,
        tile_compute_bp=tile_compute_bp,
        fine_mask_bp=fine_mask_bp,
        maskr=maskr,
    )
    return tm, dense


def get_block_num(block_num: int, device: int = 0, *, kind: str = "mix") -> int:
    """Get launch block_num. -1 means full-device launch."""
    if int(block_num) != -1:
        return max(1, int(block_num))
    props = torch.npu.get_device_properties(int(device))
    if kind == "vector":
        return max(1, int(props.vector_core_num))
    if kind in {"cube", "mix"}:
        return max(1, int(props.cube_core_num))
    raise ValueError(f"Unsupported kernel kind for block_num default: {kind!r}")


def _require_torch_npu(device_id: int) -> Any:
    """检查 torch_npu 依赖"""
    try:
        import torch_npu
    except ImportError as exc:
        raise SystemExit("FA run requires torch_npu.") from exc
    torch.npu.set_device(device_id)
    return torch


def create_tla_tensor(buf, layout_tag=tla.arch.RowMajor):
    """用 from_dlpack 包装 torch NPU tensor 为 tla.Tensor"""
    contiguous = buf.contiguous()
    t = from_dlpack(contiguous, layout_tag=layout_tag)
    t._torch_storage = contiguous
    return t


@dataclass
class RunResult:
    name: str
    passed: bool
    info: str = ""
    sentinel_info: str = ""


@dataclass
class _KernelOutput:
    """_run_kernel_core 的返回值"""

    out_buf: torch.Tensor  # (total_q*D,) fp16 NPU buffer
    total_q: int
    D: int
    kernel_time_s: float
    display_name: str
    tilemask: TileMaskParams
    query: torch.Tensor  # BSND: (B, max_sq, H, D) / TND: (Σq_lens, H, D) fp16 NPU
    key: torch.Tensor  # BSND: (B, max_kv, Hkv, D) / TND: (Σkv_lens, Hkv, D)
    value: torch.Tensor  # BSND: (B, max_kv, Hkv, D) / TND: (Σkv_lens, Hkv, D)
    kv_lens_list: list[int]  # 每 batch 的 kv 长度
    q_lens_list: list[int]  # 每 batch 的 q 长度
    input_format: str = "BSND"  # "BSND" 或 "TND"
    cu_seqlens_q: torch.Tensor | None = None  # TND: (B+1,) q 累加序列长度
    cu_seqlens_k: torch.Tensor | None = None  # TND: (B+1,) kv 累加序列长度


def _run_kernel_core(
    args: argparse.Namespace,
    mask_mode: str,
    mask_key: str,
    *,
    seq_q: int,
    seq_k: int,
    batch_size: int = 1,
    num_heads: int = 1,
    kv_heads: int = 1,
    head_dim: int = 128,
    tile_m: int = 128,
    tile_n: int = 128,
    op_dtype: torch.dtype = torch.float16,
    input_format: str = "BSND",
    q_lens_list: list[int] | None = None,
    kv_lens_list_override: list[int] | None = None,
) -> _KernelOutput:
    """TileMask 生成 → DSL kernel 编译执行"""
    is_tnd = input_format == "TND"

    B, H, Hkv, D = batch_size, num_heads, kv_heads, head_dim

    if q_lens_list is None:
        q_lens_list = [seq_q] * B
    if kv_lens_list_override is None:
        kv_lens_list_override = [seq_k] * B

    max_sq = max(q_lens_list)
    max_kv = max(kv_lens_list_override)
    is_uniform = all(v == q_lens_list[0] for v in q_lens_list) and all(
        v == kv_lens_list_override[0] for v in kv_lens_list_override
    )
    var_tag = "" if is_uniform else "_var"

    torch_mod = _require_torch_npu(args.device)
    device = "npu"
    torch_mod.manual_seed(144)

    display_name = (
        f"{mask_key}_{input_format}{var_tag}_B{B}_H{H}_D{D}_Sq{max_sq}_Sk{max_kv}"
    )

    if not is_uniform:
        print(f"  q_lens: {q_lens_list}")
        print(f"  kv_lens: {kv_lens_list_override}")

    kv_lens_list = list(kv_lens_list_override)
    if mask_key == "causal" or mask_key == "none":
        _mask_tensors = {"seq_lens": torch.tensor(kv_lens_list, dtype=torch.int32)}
    else:
        raise ValueError(f"Unknown mask_key: {mask_key}")

    tilemask, _ = generate_fa_mask(
        "none" if mask_mode == "none" else "causal",
        batch=B,
        seq_q=max_sq,
        seq_k=max_kv,
        tile_m=tile_m,
        tile_n=tile_n,
    )

    if is_tnd:
        # TND 格式：(Σlens, H, D) 3D 变长 — QKV 放在 CPU，kernel 前再迁 NPU
        Tq_total = sum(q_lens_list)
        Tk_total = sum(kv_lens_list_override)
        query_raw = torch.randn((Tq_total, H, D), dtype=torch.float32)
        key_raw = torch.randn((Tk_total, Hkv, D), dtype=torch.float32)
        value_raw = torch.randn((Tk_total, Hkv, D), dtype=torch.float32)
        query = query_raw.to(op_dtype)
        key = key_raw.to(op_dtype)
        value = value_raw.to(op_dtype)
        total_q = Tq_total * H
        total_k = Tk_total * Hkv
        _cum_q = 0
        _cum_q_vals = [0]
        for v in q_lens_list:
            _cum_q += v
            _cum_q_vals.append(_cum_q)
        cu_seqlens_q = torch.tensor(_cum_q_vals, dtype=torch.int32)
        _cum_k = 0
        _cum_k_vals = [0]
        for v in kv_lens_list_override:
            _cum_k += v
            _cum_k_vals.append(_cum_k)
        cu_seqlens_k = torch.tensor(_cum_k_vals, dtype=torch.int32)
        actual_q_vals = list(_cum_q_vals)
        actual_kv_vals = list(_cum_k_vals)
        query_4d = query_raw
    else:
        # BSND 格式：(B, max_sq, H, D) 4D 定长 — QKV 放在 CPU，kernel 前再迁 NPU
        query_raw = torch.randn((B, max_sq, H, D), dtype=torch.float32)
        key_raw = torch.randn((B, max_kv, Hkv, D), dtype=torch.float32)
        value_raw = torch.randn((B, max_kv, Hkv, D), dtype=torch.float32)
        query = query_raw.to(op_dtype)
        key = key_raw.to(op_dtype)
        value = value_raw.to(op_dtype)
        total_q = B * max_sq * H
        total_k = B * max_kv * Hkv
        cu_seqlens_q = None
        cu_seqlens_k = None
        actual_q_vals = [max_sq] * B
        actual_kv_vals = [max_kv] * B
        query_4d = query_raw

    kv_lens_npu = torch.tensor(kv_lens_list_override, dtype=torch.int32)
    _scale_value = 1.0 / (head_dim**0.5)

    # Q/K/V 迁到 NPU（仅 kernel 需要，golden 用 CPU 版本）
    query_2d = query.reshape(total_q, D).contiguous().to(device)
    key_2d = key.reshape(total_k, D).contiguous().to(device)
    value_2d = value.reshape(total_k, D).contiguous().to(device)

    _tla_dtype = tla.BFloat16 if op_dtype == torch.bfloat16 else tla.Float16
    out_buf = torch.full(
        (total_q, D), getattr(args, "sentinel", -7.0), dtype=op_dtype, device=device
    )

    tla_query = create_tla_tensor(query_2d, tla.arch.RowMajor)
    tla_key = create_tla_tensor(key_2d, tla.arch.RowMajor)
    tla_value = create_tla_tensor(value_2d, tla.arch.RowMajor)

    _paged = bool(getattr(args, "paged", False))
    _mnb = 0
    if _paged:
        # ===== PagedAttention: cache [numBlocks, Hkv, bs, D] + 打乱块表; kernel 物理页寻址 =====
        _bs = tile_n
        assert int(getattr(args, "blocksize", _bs)) == _bs, (
            "--blocksize 必须等于 kvBaseTile(tile_n)"
        )
        _is_tnd = key.dim() == 3  # TND: (total_k, Hkv, D) vs BSND: (B, max_kv, Hkv, D)
        _hkv = key.shape[1] if _is_tnd else key.shape[2]
        _cu_k = (
            cu_seqlens_k.tolist() if (_is_tnd and cu_seqlens_k is not None) else None
        )
        B_tot = len(kv_lens_list_override)
        _mnb = (max_kv + _bs - 1) // _bs
        num_blocks = B_tot * _mnb
        if getattr(args, "pa_table", "shuffle") == "identity":
            block_table = (
                torch.arange(num_blocks, dtype=torch.int32).reshape(B_tot, _mnb).clone()
            )
        else:
            _gtab = torch.Generator().manual_seed(7)
            block_table = (
                torch.randperm(num_blocks, generator=_gtab)
                .reshape(B_tot, _mnb)
                .to(torch.int32)
            )
        k_cache = torch.zeros(num_blocks, _hkv, _bs, D, dtype=op_dtype)
        v_cache = torch.zeros_like(k_cache)
        for _b in range(B_tot):
            _sk = int(kv_lens_list_override[_b])
            _koff = (
                int(_cu_k[_b]) if _cu_k else 0
            )  # TND: 该 batch 在扁平张量中的起始偏移
            for _i in range(_mnb):
                _lo, _hi = _i * _bs, min((_i + 1) * _bs, _sk)
                if _is_tnd:
                    # TND: key 是 3D (total_k, Hkv, D), 用偏移切片
                    k_cache[block_table[_b, _i], :, : _hi - _lo, :] = key[
                        _koff + _lo : _koff + _hi
                    ].permute(1, 0, 2)
                    v_cache[block_table[_b, _i], :, : _hi - _lo, :] = value[
                        _koff + _lo : _koff + _hi
                    ].permute(1, 0, 2)
                else:
                    # BSND: key 是 4D (B, max_kv, Hkv, D), 按 batch 索引
                    k_cache[block_table[_b, _i], :, : _hi - _lo, :] = key[
                        _b, _lo:_hi, :, :
                    ].permute(1, 0, 2)
                    v_cache[block_table[_b, _i], :, : _hi - _lo, :] = value[
                        _b, _lo:_hi, :, :
                    ].permute(1, 0, 2)
        tla_key = create_tla_tensor(
            k_cache.reshape(-1, D).contiguous().to(device), tla.arch.RowMajor
        )
        tla_value = create_tla_tensor(
            v_cache.reshape(-1, D).contiguous().to(device), tla.arch.RowMajor
        )
        tla_block_table = create_tla_tensor(
            block_table.reshape(-1).contiguous().to(device), tla.arch.RowMajor
        )
    else:
        tla_block_table = create_tla_tensor(
            torch.zeros(1, dtype=torch.int32, device=device), tla.arch.RowMajor
        )
    tla_output = create_tla_tensor(out_buf, tla.arch.RowMajor)

    uniform_q_seqlen = q_lens_list[0] if is_uniform else 0
    uniform_kv_seqlen = kv_lens_list_override[0] if is_uniform else 0
    uniform_tasks_per_batch = (
        H * ((uniform_q_seqlen + tile_m - 1) // tile_m) if is_uniform else 0
    )

    # tiling: FA 21字段(fa_tiling) + qFormat/kvFormat
    import fa_tiling as _fa_tiling

    _fa_td = _fa_tiling.compute_tiling(
        batch=len(kv_lens_list_override),
        num_heads=num_heads,
        kv_heads=kv_heads,
        q_seqlen_list=list(
            q_lens_list if q_lens_list else [max_sq] * len(kv_lens_list_override)
        ),
        kv_seqlen_list=list(kv_lens_list_override),
        head_dim=head_dim,
        q_base_tile=tile_m,
        kv_base_tile=tile_n,
    )
    tiling_int_list = (
        _fa_tiling.pack_tiling_int(_fa_td)
        + [
            1 if input_format == "BSND" else 0,  # qFormat
            1 if input_format == "BSND" else 0,
        ]  # kvFormat
    )
    tiling_scale_list = [_scale_value]

    Tq = (max_sq + tile_m - 1) // tile_m
    Tk = (max_kv + tile_n - 1) // tile_n
    Wk = (Tk + 31) // 32

    tla_tiling_int = create_tla_tensor(
        torch.tensor(tiling_int_list, dtype=torch.int32, device=device),
        tla.arch.RowMajor,
    )
    tla_tiling_scale = create_tla_tensor(
        torch.tensor(tiling_scale_list, dtype=torch.float32, device=device),
        tla.arch.RowMajor,
    )
    tla_actual_q = create_tla_tensor(
        torch.tensor(actual_q_vals, dtype=torch.int32, device=device), tla.arch.RowMajor
    )
    tla_actual_kv = create_tla_tensor(
        torch.tensor(actual_kv_vals, dtype=torch.int32, device=device),
        tla.arch.RowMajor,
    )
    tla_tile_range = create_tla_tensor(
        tilemask.tile_range.contiguous().to(device).reshape(-1), tla.arch.RowMajor
    )
    tla_tile_compute = create_tla_tensor(
        tilemask.tile_compute_bp.contiguous().to(device).to(torch.int32).reshape(-1),
        tla.arch.RowMajor,
    )
    tla_fine_mask = create_tla_tensor(
        tilemask.fine_mask_bp.contiguous().to(device).to(torch.int32).reshape(-1),
        tla.arch.RowMajor,
    )
    tla_maskr = create_tla_tensor(
        tilemask.maskr.contiguous().to(device).reshape(-1), tla.arch.RowMajor
    )

    _is_fp16 = op_dtype == torch.float16

    artifact = tla.compile(
        flash_attention_infer_kernel,
        tla_query,
        tla_key,
        tla_value,
        tla_output,
        tla_tiling_int,
        tla_tiling_scale,
        tla_actual_q,
        tla_actual_kv,
        tla_tile_range,
        tla_tile_compute,
        tla_fine_mask,
        tla_maskr,
        tla_block_table,
        _is_fp16,
        uniform_q_seqlen,
        uniform_kv_seqlen,
        uniform_tasks_per_batch,
        _paged,
        _mnb,
        head_dim,
        options="--npu-arch 3510",
    )

    import time as _time

    _t0 = _time.perf_counter()
    _rep = max(1, int(getattr(args, "launch_repeat", 1)))
    for _ in range(_rep):
        artifact(
            tla_query,
            tla_key,
            tla_value,
            tla_output,
            tla_tiling_int,
            tla_tiling_scale,
            tla_actual_q,
            tla_actual_kv,
            tla_tile_range,
            tla_tile_compute,
            tla_fine_mask,
            tla_maskr,
            tla_block_table,
            block_num=get_block_num(args.block_num, args.device),
        )
    torch.npu.synchronize()
    _dt = _time.perf_counter() - _t0

    return _KernelOutput(
        out_buf=out_buf,
        total_q=total_q,
        D=D,
        kernel_time_s=_dt,
        display_name=display_name,
        tilemask=tilemask,
        query=query,
        key=key,
        value=value,
        kv_lens_list=list(kv_lens_list_override),
        q_lens_list=list(q_lens_list),
        input_format=input_format,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
    )


def _fa_style_bm(query, key, value, scale, kv_lens_per_batch, custom_mask=None):
    """标杆: CPU模拟FA"""
    B, Sq, Hq, D = query.shape
    Hkv = key.shape[2]
    dt = query.dtype
    FILL = -3e38  # FA 的 MIN_VALUE
    out = torch.empty_like(query)
    for b in range(B):
        kv_b = int(kv_lens_per_batch[b])
        q_b = query[b].permute(1, 0, 2)  # [Hq, Sq, D] f16
        k_b = key[b][:kv_b].permute(1, 0, 2)
        v_b = value[b][:kv_b].permute(1, 0, 2)
        if Hkv != Hq:
            g = Hq // Hkv
            k_b = k_b.repeat_interleave(g, dim=0)
            v_b = v_b.repeat_interleave(g, dim=0)
        gl = gm = go = None
        for kv_start in range(0, kv_b, 128):
            sub_len = min(128, kv_b - kv_start)
            sub_k = k_b[:, kv_start : kv_start + sub_len, :]
            sub_v = v_b[:, kv_start : kv_start + sub_len, :]
            qk = torch.matmul(q_b, sub_k.transpose(-2, -1)).float() * scale
            if custom_mask is not None:
                m = custom_mask[b, :Sq, kv_start : kv_start + sub_len].bool()
                qk = qk.masked_fill(m.unsqueeze(0), FILL)
            lm = torch.max(qk, dim=-1, keepdim=True)[0]
            if kv_start == 0:
                hm = lm
                dm = torch.zeros_like(lm)
            else:
                hm = torch.maximum(gm, lm)
                dm = gm - hm
            gm = hm
            sim = torch.exp(qk - hm)
            row_sum = torch.sum(sim, dim=-1, keepdim=True)
            p = sim.to(dt)
            lo = torch.matmul(p, sub_v).float()
            if kv_start == 0:
                gl = row_sum
                go = lo
            else:
                dm = torch.exp(dm)
                gl = gl * dm + row_sum
                go = go * dm + lo
        out[b] = torch.nan_to_num(go / gl, nan=0.0).permute(1, 0, 2).to(query.dtype)
    return out


def compute_golden_torch_tnd(
    query, key, value, scale, cu_seqlens_q=None, cu_seqlens_k=None, custom_mask=None
):
    """TND 变长格式 golden: 逐 batch 切片, f32 全量 attention."""
    T_q, H_q, D = query.shape
    T_k, H_kv, _ = key.shape
    k, v = key, value
    if H_kv != H_q:
        g = H_q // H_kv
        k = k.repeat_interleave(g, dim=1)
        v = v.repeat_interleave(g, dim=1)
    q_hnd = query.permute(1, 0, 2).float()
    k_hnd = k.permute(1, 0, 2).float()
    v_hnd = v.permute(1, 0, 2).float()
    scores = torch.matmul(q_hnd, k_hnd.transpose(-2, -1)) * scale
    if cu_seqlens_k is None:
        cu_seqlens_k = cu_seqlens_q
    nb = len(cu_seqlens_q) - 1
    tnd_mask = torch.ones((T_q, T_k), dtype=torch.bool, device=query.device)
    for i in range(nb):
        qs, qe = int(cu_seqlens_q[i]), int(cu_seqlens_q[i + 1])
        ks, ke = int(cu_seqlens_k[i]), int(cu_seqlens_k[i + 1])
        if custom_mask is not None:
            local = custom_mask[i, : qe - qs, : ke - ks].to(
                dtype=torch.bool, device=query.device
            )
            tnd_mask[qs:qe, ks:ke] = local
        else:
            tnd_mask[qs:qe, ks:ke] = False
    scores = scores.masked_fill(tnd_mask.unsqueeze(0), float("-inf"))
    row_max = scores.max(dim=-1, keepdim=True).values
    row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
    probs = torch.exp(scores - row_max)
    probs = torch.where(torch.isfinite(scores), probs, torch.zeros_like(probs))
    denom = probs.sum(dim=-1, keepdim=True)
    attn = torch.where(
        denom > 0, probs / denom.clamp_min(1e-30), torch.zeros_like(probs)
    )
    out = torch.matmul(attn, v_hnd)
    return out.permute(1, 0, 2).contiguous().to(dtype=query.dtype)


def compute_golden_torch_bsnd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    kv_lens_per_batch,
    custom_mask: torch.Tensor = None,
) -> torch.Tensor:
    """基准 Golden（BSND 定长布局），与 compute_golden_torch_tnd 数值等价，但按 batch 独立计算。"""
    B, Sq, H_q, D = query.shape
    H_kv = key.shape[2]
    out = torch.empty_like(query)

    for b in range(B):
        kv_b = int(kv_lens_per_batch[b])
        q_b = query[b]  # (Sq, H_q, D)
        k_b = key[b][:kv_b]  # (kv_b, H_kv, D)
        v_b = value[b][:kv_b]  # (kv_b, H_kv, D)

        if H_kv != H_q:
            g = H_q // H_kv
            k_b = k_b.repeat_interleave(g, dim=1)
            v_b = v_b.repeat_interleave(g, dim=1)

        q_hnd = q_b.permute(1, 0, 2).float()  # (H_q, Sq, D)
        k_hnd = k_b.permute(1, 0, 2).float()  # (H_q, kv_b, D)
        v_hnd = v_b.permute(1, 0, 2).float()

        scores = torch.matmul(q_hnd, k_hnd.transpose(-2, -1)) * scale

        if custom_mask is not None:
            m = custom_mask[b, :Sq, :kv_b].to(dtype=torch.bool, device=query.device)
            scores = scores.masked_fill(m.unsqueeze(0), float("-inf"))

        row_max = scores.max(dim=-1, keepdim=True).values
        row_max = torch.where(
            torch.isfinite(row_max), row_max, torch.zeros_like(row_max)
        )
        probs = torch.exp(scores - row_max)
        probs = torch.where(torch.isfinite(scores), probs, torch.zeros_like(probs))
        denom = probs.sum(dim=-1, keepdim=True)
        attn = torch.where(
            denom > 0, probs / denom.clamp_min(1e-30), torch.zeros_like(probs)
        )

        o_hnd = torch.matmul(attn, v_hnd)
        out[b] = o_hnd.permute(1, 0, 2).to(query.dtype)
    return out


@dataclass
class GoldenInput:
    mask: torch.Tensor
    scale: float
    op_dtype: torch.dtype
    kv_lens: torch.Tensor


def main() -> int:
    args = parse_args()
    rst = run(args)
    print_result(rst, args)
    return 0 if rst.passed else 1


def run(args: argparse.Namespace) -> RunResult:
    """执行kernel，根据perf_only决定是否进入校验阶段"""
    torch_mod = _require_torch_npu(args.device)

    op_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    print(
        f"--- BATCH=({args.batch},{args.qseqlen},{args.kvseqlen}) "
        f"HEAD=({args.headnum},{args.kvheadnum}) "
        f"HEAD_DIM={args.head_dim} "
        f"dtype={args.dtype} mask={args.mask} sentinel={args.sentinel} ---"
    )
    ko = _run_kernel_core(
        args, args.mask, args.mask,
        seq_q=args.qseqlen,
        seq_k=args.kvseqlen,
        batch_size=args.batch,
        num_heads=args.headnum,
        kv_heads=args.kvheadnum,
        head_dim=args.head_dim,
        input_format=args.format,
        op_dtype=op_dtype,
    )
    return _validate_kernel_output(args, ko)


def run_debug(
    args: argparse.Namespace, mask_mode: str, mask_key: str, **kwargs
) -> RunResult:
    """保留原调试入口及自定义tile、变长序列等参数的转发能力"""
    input_format = kwargs.pop("input_format", "BSND")
    ko = _run_kernel_core(
        args, mask_mode, mask_key, input_format=input_format, **kwargs
    )
    return _validate_kernel_output(args, ko)


def _validate_kernel_output(args: argparse.Namespace, ko: _KernelOutput) -> RunResult:
    """共用原调试入口的长度统计、perf_only判断和校验流程"""
    # 保持原顺序：先统计长度，再判断是否跳过校验
    max_sq = max(ko.q_lens_list)
    max_kv = max(ko.kv_lens_list)
    batch = len(ko.kv_lens_list)
    if bool(getattr(args, "perf_only", False)):
        return RunResult(name="perf_only", passed=True, info="verify skipped")
    return run_validation(
        ko,
        mask_mode=args.mask,
        args=args,
        sequence_shape=(max_sq, max_kv, batch),
    )


def run_validation(
    ko: _KernelOutput, *, mask_mode: str, args: argparse.Namespace,
    sequence_shape: tuple[int, int, int] | None = None,
) -> RunResult:
    """只组织校验阶段，具体计算交给各阶段函数"""
    if ko.input_format not in {"TND", "BSND"}:
        raise ValueError(f"Unknown input_format: {ko.input_format}")
    inputs = prepare_cpu_golden_input(ko, mask_mode, sequence_shape=sequence_shape)
    if ko.input_format == "TND":
        golden = compute_golden_tnd(ko, inputs)
        reference = compute_reference_tnd(ko, inputs)
    else:
        golden = compute_golden_bsnd(ko, inputs)
        reference = compute_reference_bsnd(ko, inputs)
    return compare(
        ko.out_buf, golden, reference,
        op_dtype=inputs.op_dtype,
        name=ko.display_name,
        kernel_time_s=ko.kernel_time_s,
        total_q=ko.total_q,
        head_dim=ko.D,
        args=args,
    )


def prepare_cpu_golden_input(
    ko: _KernelOutput, mask_mode: str, *,
    sequence_shape: tuple[int, int, int] | None = None,
) -> GoldenInput:
    """准备两种格式共用的Mask、scale和KV长度，沿用原来的Mask规则"""
    # 复用run的长度统计，独立调用时就地统计
    if sequence_shape is None:
        sequence_shape = (max(ko.q_lens_list), max(ko.kv_lens_list), len(ko.kv_lens_list))
    max_sq, max_kv, batch = sequence_shape
    prefix = max_kv - max_sq if mask_mode == "causal" else max_kv
    cols = torch.arange(max_kv)
    rows = torch.arange(max_sq)
    visible = (
        (cols.unsqueeze(0) <= rows.unsqueeze(1) + prefix)
        .unsqueeze(0)
        .expand(batch, -1, -1)
    )
    return GoldenInput(
        mask=~visible,  # True表示屏蔽
        scale=1.0 / math.sqrt(float(ko.D)),
        op_dtype=ko.query.dtype,
        kv_lens=torch.tensor(ko.kv_lens_list, dtype=torch.int32),
    )


def compute_golden_tnd(ko: _KernelOutput, inputs: GoldenInput) -> torch.Tensor:
    """沿用TND原golden的计算与dtype舍入，再转为CPU FP32比较"""
    golden = compute_golden_torch_tnd(
        ko.query, ko.key, ko.value, inputs.scale,
        cu_seqlens_q=ko.cu_seqlens_q,
        cu_seqlens_k=ko.cu_seqlens_k,
        custom_mask=inputs.mask,
    )
    return golden.reshape(ko.total_q, ko.D).cpu().float()


def compute_golden_bsnd(ko: _KernelOutput, inputs: GoldenInput) -> torch.Tensor:
    """沿用BSND原golden的计算与dtype舍入，再转为CPU FP32比较"""
    golden = compute_golden_torch_bsnd(
        ko.query, ko.key, ko.value, inputs.scale,
        kv_lens_per_batch=inputs.kv_lens,
        custom_mask=inputs.mask,
    )
    return golden.reshape(ko.total_q, ko.D).cpu().float()


def compute_reference_tnd(ko: _KernelOutput, inputs: GoldenInput) -> torch.Tensor:
    """逐batch将TND适配为BSND，计算分块参考并按原顺序拼接"""
    ref_parts = []
    cu_q = (
        ko.cu_seqlens_q.tolist() if ko.cu_seqlens_q is not None else [0, ko.total_q]
    )
    cu_k = (
        ko.cu_seqlens_k.tolist()
        if ko.cu_seqlens_k is not None
        else [0, ko.key.shape[0]]
    )
    for bi in range(len(cu_q) - 1):
        qs, qe = int(cu_q[bi]), int(cu_q[bi + 1])
        ks, ke = int(cu_k[bi]), int(cu_k[bi + 1])
        if qe <= qs:
            continue
        q_b = ko.query[qs:qe].unsqueeze(0)  # (1, sq, H, D)
        k_b = ko.key[ks:ke].unsqueeze(0)
        v_b = ko.value[ks:ke].unsqueeze(0)
        m_b = (
            inputs.mask[bi : bi + 1, : qe - qs, : ke - ks]
            if inputs.mask is not None
            else None
        )
        ref_b = _fa_style_bm(
            q_b, k_b, v_b, inputs.scale, kv_lens_per_batch=[ke - ks], custom_mask=m_b
        )
        ref_parts.append(ref_b.reshape(-1, ko.D))
    return torch.cat(ref_parts, dim=0).cpu().float()


def compute_reference_bsnd(ko: _KernelOutput, inputs: GoldenInput) -> torch.Tensor:
    """计算BSND的分块参考，并统一结果形状"""
    reference = _fa_style_bm(
        ko.query, ko.key, ko.value, inputs.scale,
        kv_lens_per_batch=inputs.kv_lens,
        custom_mask=inputs.mask,
    )
    return reference.reshape(ko.total_q, ko.D).cpu().float()


def compute_error_metrics(
    actual: torch.Tensor, golden: torch.Tensor
) -> tuple[float, float, float]:
    """返回最大相对误差、平均相对误差和均方根误差"""
    diff = (actual - golden).abs()
    relative = diff / (golden.abs() + 1e-7)
    return relative.max().item(), relative.mean().item(), torch.sqrt((diff**2).mean()).item()


def check_sentinel(output: torch.Tensor, sentinel: float) -> str:
    """仅生成输出填充值诊断，不参与精度通过判定"""
    sentinel_t = torch.full_like(output, sentinel)
    unchanged = torch.isclose(output, sentinel_t, rtol=0.0, atol=1e-2)
    return (
        f"O unchanged (sentinel)? {bool(unchanged.all())} "
        f"changed_count={int((~unchanged).sum().item())} / {output.numel()}"
    )


def compare(
    output: torch.Tensor,
    golden: torch.Tensor,
    reference: torch.Tensor,
    *,
    op_dtype: torch.dtype,
    name: str,
    kernel_time_s: float,
    total_q: int,
    head_dim: int,
    args: argparse.Namespace,
) -> RunResult:
    """相对原golden衡量kernel和分块参考的误差，沿用原来的通过阈值"""
    kernel_flat = output.reshape(total_q, head_dim).cpu().float()
    num_mare, num_mere, num_rmse = compute_error_metrics(kernel_flat, golden)
    den_mare, den_mere, den_rmse = compute_error_metrics(reference, golden)
    floor = 2 ** (-7) if op_dtype == torch.float16 else 2 ** (-6)
    ratio_rmse = num_rmse / max(den_rmse, floor)
    ratio_mare = num_mare / max(den_mare, floor)
    ratio_mere = num_mere / max(den_mere, floor)
    passed = ratio_mare <= 2.0 and ratio_mere <= 1.2 and ratio_rmse <= 1.2
    sentinel_info = check_sentinel(kernel_flat, getattr(args, "sentinel", -7.0))
    return RunResult(
        name=name,
        passed=passed,
        info=f"kernel={kernel_time_s:.3f}s",
        sentinel_info=sentinel_info,
    )


def print_result(rst: RunResult, args: argparse.Namespace) -> None:
    """由main打印原有结果日志，保留perf_only时的空sentinel行"""
    print(
        f"host=torch_npu BATCH={args.batch} Q_SEQ={args.qseqlen} KV_SEQ={args.kvseqlen} "
        f"HEAD_NUM={args.headnum} KV_HEAD_NUM={args.kvheadnum} HEAD_DIM={args.head_dim}"
    )
    print(rst.sentinel_info)
    print(f"passed={rst.passed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FlashAttention Infer kernel")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--qseqlen", type=int, default=117, help="Q 序列长度")
    parser.add_argument("--kvseqlen", type=int, default=512, help="KV 序列长度")
    parser.add_argument("--headnum", type=int, default=8, help="Q 头数")
    parser.add_argument("--kvheadnum", type=int, default=1, help="KV 头数")
    parser.add_argument("--batch", type=int, default=1, help="batch size")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument(
        "--mask",
        choices=("none", "causal"),
        default="causal",
        help="mask: none=全可见; causal=chunk-prefill 底对齐因果",
    )
    parser.add_argument("--format", choices=("BSND", "TND"), default="BSND")
    parser.add_argument("--block-num", type=int, default=-1, help="-1=自动取满核")
    parser.add_argument(
        "--paged",
        action="store_true",
        help="PagedAttention: cache+块表, kernel 侧物理页寻址",
    )
    parser.add_argument(
        "--sentinel",
        type=float,
        default=-7.0,
        help="O 的初始值，用于检测 kernel 是否真正写入",
    )
    parser.add_argument(
        "--perf-only",
        "--perf_only",
        dest="perf_only",
        action="store_true",
        help="仅执行 kernel，跳过 golden 和精度验证",
    )
    args = parser.parse_args()
    args.head_dim = 128

    return args



if __name__ == "__main__":
    raise SystemExit(main())


