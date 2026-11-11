# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Host runner for fused FFN (Up Matmul + Act + SyncAll + Down Matmul).
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Sample dir (``ffn`` package) + local ``host/`` helpers.
# Layout: <repo>/examples/python/tla_dsl/tla/kernel/matmul/fused_ffn/
_EXAMPLE_DIR = Path(__file__).resolve().parent
for _p in (str(_EXAMPLE_DIR), str(_EXAMPLE_DIR / "host")):
    if _p not in sys.path:
        sys.path.append(_p)

import catlass.tla as tla
import torch
import torch_npu
from catlass.types import dtype_size_bytes
from ffn.buffers import apply_streamk_tilings, share_tilings
from ffn.config import (
    BIAS_CUBE_BT,
    BIAS_NONE,
    DOWN_BASIC,
    DOWN_FULL_A,
    DOWN_FULL_B,
    DOWN_STREAMK,
    LAYOUT_CANONICAL,
    LAYOUT_LINEAR,
    SWIGLU_DUAL,
    SWIGLU_SINGLE,
    UP_L0C2UB,
    build_ffn_config,
)
from ffn.kernels.ffn_basic import ffn_basic_kernel
from ffn.policies.act import parse_activation
from ffn.stages.up_l0c2ub_silu import (
    EVG_UB_NODES,
    EVG_UB_STAGES,
    L0C_SIZE,
    UB_SIZE,
    swiglu_b_cpp_eligible,
    swiglu_dual_ub_bytes,
    up_fixpipe_acc_elems,
)
from ffn.tiling import (
    choose_block_num,
    choose_ffn_host_tilings,
    choose_swizzle_down,
    choose_swizzle_up,
    maybe_force_dual_full_l0_tk,
    prefer_down_streamk,
    resolve_swiglu_mode,
    select_swiglu_tilings,
    set_swiglu_up_tn,
    sk_workspace_elems,
)
from golden import golden_ffn
from swiglu_pack import swiglu_simd_lanes
from tensor_utils import compute_ub_slot_elems, create_tla_tensor, get_block_num

# Host smoke = ATK ``mixed_tolerance_bm_v2`` (device-iso golden).
_MIXED_TOL_V2 = {
    torch.float16: dict(
        rtol=1.0e-3, atol=1.0e-3, ratio=0.99, max_abs=1e-1, ulp_mult=32
    ),
    torch.bfloat16: dict(
        rtol=7.8e-3, atol=7.8e-3, ratio=0.99, max_abs=1e-1, ulp_mult=32
    ),
}

VECTOR_ELE = 256


def _spacing_towards_pos_inf(
    remote_fp32: torch.Tensor, ulp_dtype: torch.dtype
) -> torch.Tensor:
    """ULP spacing in ``ulp_dtype`` (fp16/bf16), measured in fp32.

    ``nextafter`` must run on the target dtype; casting to fp32 first would
    yield fp32 ULPs and falsely reject large-magnitude 1-ULP differences.
    """
    cur = remote_fp32.to(dtype=ulp_dtype)
    nxt = torch.nextafter(cur, torch.full_like(cur, torch.inf))
    return (nxt.float() - cur.float()).abs().clamp_min(torch.finfo(torch.float32).tiny)


def _mixed_tolerance_bm_v2(y: torch.Tensor, ref: torch.Tensor) -> tuple[bool, float]:
    """ATK ``mixed_tolerance_bm_v2``: matched_ratio + abs/ULP hard gate."""
    thr = _MIXED_TOL_V2.get(y.dtype)
    if thr is None:
        thr = _MIXED_TOL_V2[torch.float16]
    yf = y.detach().cpu().float().reshape(-1)
    rf = ref.detach().cpu().float().reshape(-1)
    abs_err = (yf - rf).abs()
    matched = float(
        torch.isclose(yf, rf, rtol=thr["rtol"], atol=thr["atol"]).float().mean().item()
    )
    spacing = _spacing_towards_pos_inf(rf.reshape(y.shape), y.dtype).reshape(-1)
    soft = (abs_err <= thr["max_abs"]) | (abs_err <= thr["ulp_mult"] * spacing)
    ok = matched >= thr["ratio"] and bool(soft.all().item())
    return ok, matched


# Ascend950 WithoutQue: disable ASC/VF stack UB reserves (product default).
_NPU_COMPILE_OPTS = "--npu-arch 3510 --cce-disable-asc-reserved-ubuf --cce-disable-vf-stack-reserved-ubuf"


def _down_mode_tag(use_sk: bool, down_full: int) -> str:
    if use_sk:
        return "streamk"
    if int(down_full) == 1:
        return "full_a"
    if int(down_full) == 2:
        return "full_b"
    return "basic"


def _dtype_map(name: str):
    # x is fp16/bf16 only (f32 rejected in run_once).
    return {"f16": torch.float16, "bf16": torch.bfloat16}[name]


def _ub_slots(
    dtype_name: str,
    *,
    sk_ub_reserve: int = 0,
    acc_elems: int | None = None,
    acc_slots: int = 1,
) -> tuple[int, int]:
    # L0C→UB FixPipe acc: per-slot × slots (dual = 2×tile-sized).
    n_slots = max(1, int(acc_slots))
    acc_bytes = int(acc_elems) * 4 * n_slots if acc_elems is not None else L0C_SIZE // 2
    stages = (
        1
        if (
            n_slots > 1
            or (acc_elems is not None and int(acc_elems) > L0C_SIZE // 2 // 4)
        )
        else EVG_UB_STAGES
    )
    budget = UB_SIZE - acc_bytes - max(0, int(sk_ub_reserve)) - 1024
    return (
        compute_ub_slot_elems(
            dtype_name,
            nodes=EVG_UB_NODES,
            stages=stages,
            budget_bytes=max(0, budget),
        ),
        VECTOR_ELE // 4,
    )


def _dtype_name_from_torch(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "f16"
    if dtype == torch.bfloat16:
        return "bf16"
    raise ValueError(f"dtype {dtype} is not supported (x is fp16/bf16 only)")


def run_from_tensors(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    *,
    activation: str = "silu",
    bias1: torch.Tensor | None = None,
    bias2: torch.Tensor | None = None,
    layout: str = "linear",
    block_num: int = -1,
    device: int | None = None,
    swiglu_mode_force: str = "auto",
    warmup: int = 0,
    iters: int = 0,
) -> torch.Tensor:
    """Compile and launch fused FFN from host tensors; return ``y`` on device.

    Weight layouts match ``run_once``:
      silu/gelu linear: w1 [H,K], w2 [N,H]
      swiglu linear:    w1 [2H,K], w2 [N,H]
      canonical:        w1 [K,H|2H], w2 [H,N] (SwiGLU rejected)
    Optional ``bias1`` / ``bias2`` enable Cube BT (paired).
    """
    if device is None:
        if x.device.type == "npu" and x.device.index is not None:
            device = int(x.device.index)
        else:
            device = int(torch.npu.current_device())
    torch.npu.set_device(int(device))

    activation = str(activation).strip().lower()
    act_pol = parse_activation(activation)
    layout_id = (
        LAYOUT_CANONICAL if str(layout).lower() == "canonical" else LAYOUT_LINEAR
    )
    use_bias = bias1 is not None
    if use_bias != (bias2 is not None):
        raise ValueError("bias1 and bias2 must both be set or both be None")

    # Move to CPU for host tiling / pad; single H2D later.
    x = x.detach().to("cpu").contiguous()
    w1 = w1.detach().to("cpu").contiguous()
    w2 = w2.detach().to("cpu").contiguous()
    b1 = None if bias1 is None else bias1.detach().to("cpu").contiguous()
    b2 = None if bias2 is None else bias2.detach().to("cpu").contiguous()

    dtype = x.dtype
    dtype_name = _dtype_name_from_torch(dtype)
    dtype_bytes = int(dtype_size_bytes(dtype_name))
    m, k = int(x.shape[0]), int(x.shape[1])
    if layout_id == LAYOUT_CANONICAL:
        up_n, n = int(w1.shape[1]), int(w2.shape[1])
        h_from_w2 = int(w2.shape[0])
    else:
        up_n, n = int(w1.shape[0]), int(w2.shape[0])
        h_from_w2 = int(w2.shape[1])
    # Prefer w2's H: silu/gelu up_n==H; swiglu up_n==2H.
    if activation == "swiglu":
        if up_n != 2 * h_from_w2:
            raise ValueError(
                f"swiglu expects w1 out-dim=2H and w2 H; got up_n={up_n}, H={h_from_w2}"
            )
        h = h_from_w2
    else:
        if up_n != h_from_w2:
            raise ValueError(
                f"{activation} expects w1 out-dim == w2 H; got {up_n} vs {h_from_w2}"
            )
        h = h_from_w2
    up_n = int(act_pol.up_out_width(h))
    bias_mode = BIAS_CUBE_BT if use_bias else BIAS_NONE
    if use_bias:
        bias_torch_dtype = b1.dtype
        if int(b1.numel()) != int(up_n) or int(b2.numel()) != int(n):
            raise ValueError(
                f"bias shapes must be bias1[{up_n}], bias2[{n}]; got {tuple(b1.shape)}, {tuple(b2.shape)}"
            )
    else:
        bias_torch_dtype = torch.float32

    y = torch.zeros(m, n, dtype=dtype, device="cpu")
    hidden = torch.zeros(m, h, dtype=dtype, device="cpu")

    cube_max = get_block_num(-1, device=int(device), kind="cube")
    bn_force = None
    tu, td, down_full = choose_ffn_host_tilings(
        m,
        k,
        h,
        n,
        cube_max=cube_max,
        up_n=up_n,
        has_bias=use_bias,
        dtype_bytes=dtype_bytes,
    )
    if down_full != 0:
        tiling_up, tiling_down = tu, td
    else:
        tiling_up, tiling_down, _ = share_tilings(tu, td, m=m, h=up_n, n=n, down_k=h)
    swiglu_mode = SWIGLU_SINGLE
    w1_up_dev = None
    b1_up_dev = None
    b1_dev = b1
    if activation == "swiglu":
        tiling_up, tiling_down = select_swiglu_tilings(
            m, k, h, n, tiling_up, tiling_down, cube_max=cube_max
        )
        if down_full == 0:
            tiling_up, tiling_down, _ = share_tilings(
                tiling_up, tiling_down, m=m, h=up_n, n=n, down_k=h, target_bd=1
            )
        swiglu_mode = resolve_swiglu_mode(tiling_up, force=swiglu_mode_force)
        if layout_id == LAYOUT_CANONICAL:
            raise ValueError("SwiGLU + Canonical is not supported")
        if swiglu_mode == SWIGLU_DUAL:
            up_n = int(h)
            tu, td, down_full = choose_ffn_host_tilings(
                m,
                k,
                h,
                n,
                cube_max=cube_max,
                up_n=up_n,
                has_bias=use_bias,
                dtype_bytes=dtype_bytes,
            )
            if down_full != 0:
                tiling_up, tiling_down = tu, td
            else:
                tiling_up, tiling_down, _ = share_tilings(
                    tu, td, m=m, h=up_n, n=n, down_k=h
                )
            tiling_up = maybe_force_dual_full_l0_tk(tiling_up)
            w1_dev = w1[:h].contiguous()
            w1_up_dev = w1[h:].contiguous()
            if use_bias:
                b1_dev = b1[:h].contiguous()
                b1_up_dev = b1[h:].contiguous()
        else:
            tiling_up = set_swiglu_up_tn(tiling_up, int(tiling_up.l1_tn))
            _up_n = int(tiling_up.l1_tn)
            swiglu_b_cpp_eligible(
                _up_n,
                int(tiling_up.l1_tk),
                2 * int(h),
                use_bias=use_bias,
            )
            w1_dev = w1
    else:
        w1_dev = w1

    use_sk = False
    if down_full == 0:
        use_sk = prefer_down_streamk(
            m, h, n, tiling_down, cube_max, activation, has_bias=use_bias
        )
    m_run, n_run = int(m), int(n)
    b2_dev = b2
    if use_sk:
        tiling_up, tiling_down, _ = apply_streamk_tilings(
            tiling_up,
            m=m,
            h=up_n,
            n=n,
            down_k=h,
            has_bias=use_bias,
            aic_num=cube_max,
        )
        tm, tn = int(tiling_down.l1_tm), int(tiling_down.l1_tn)
        m_run = ((int(m) + tm - 1) // tm) * tm
        n_run = ((int(n) + tn - 1) // tn) * tn
        if m_run != m or n_run != n:
            x_pad = torch.zeros(m_run, k, dtype=dtype, device="cpu")
            x_pad[:m, :].copy_(x)
            x = x_pad
            y = torch.zeros(m_run, n_run, dtype=dtype, device="cpu")
            hidden = torch.zeros(m_run, h, dtype=dtype, device="cpu")
            if layout_id == LAYOUT_CANONICAL:
                w2_pad = torch.zeros(h, n_run, dtype=dtype, device="cpu")
                w2_pad[:, :n].copy_(w2)
            else:
                w2_pad = torch.zeros(n_run, h, dtype=dtype, device="cpu")
                w2_pad[:n, :].copy_(w2)
            w2 = w2_pad
            if use_bias:
                b2_pad = torch.zeros(n_run, dtype=bias_torch_dtype, device="cpu")
                b2_pad[:n].copy_(b2)
                b2_dev = b2_pad
    if use_sk:
        down_mode = DOWN_STREAMK
    elif down_full == 1:
        down_mode = DOWN_FULL_A
    elif down_full == 2:
        down_mode = DOWN_FULL_B
    else:
        down_mode = DOWN_BASIC
    cfg = build_ffn_config(
        activation,
        up_path=UP_L0C2UB,
        bias_mode=bias_mode,
        down_mode=down_mode,
        layout=layout_id,
        swiglu_mode=swiglu_mode,
    )
    if activation == "swiglu" and swiglu_mode == SWIGLU_SINGLE:
        from dataclasses import replace as _dc_replace

        cfg = _dc_replace(
            cfg,
            swiglu_b_cpp=True,
            swiglu_h=int(h),
            swiglu_k=int(k),
            swiglu_m=int(m_run),
            swiglu_n=int(n_run),
        )

    if activation == "swiglu" and swiglu_mode == SWIGLU_SINGLE:
        wks = torch.zeros(m_run, 2 * h, dtype=dtype, device="cpu")
    else:
        wks = torch.zeros(1, 1, dtype=dtype, device="cpu")
    if activation == "swiglu" and swiglu_mode == SWIGLU_DUAL:
        gate = torch.zeros(m_run, h, dtype=torch.float32, device="cpu")
    else:
        gate = torch.zeros(1, 1, dtype=torch.float32, device="cpu")

    bn = (
        int(bn_force)
        if bn_force is not None
        else (
            get_block_num(block_num, device=int(device), kind="cube")
            if int(block_num) != -1
            else choose_block_num(
                m_run,
                k,
                up_n,
                n_run,
                tiling_up,
                tiling_down,
                cube_max,
                down_streamk=use_sk,
                down_k=h,
                swiglu_single_n_align32=(
                    activation == "swiglu" and swiglu_mode == SWIGLU_SINGLE
                ),
                has_bias=use_bias,
            )
        )
    )
    if use_sk:
        sk_elems = sk_workspace_elems(bn, tiling_down)
        sk_ws = torch.zeros(int(bn) * 256, 256, dtype=torch.float32, device="cpu")
        if sk_ws.numel() != sk_elems:
            raise ValueError(
                f"StreamK workspace size mismatch: {sk_ws.numel()} != {sk_elems}"
            )
    else:
        sk_ws = torch.zeros(1, 1, dtype=torch.float32, device="cpu")

    x = x.contiguous().npu()
    w1_dev = w1_dev.contiguous().npu()
    if w1_up_dev is not None:
        w1_up_dev = w1_up_dev.contiguous().npu()
    w2 = w2.contiguous().npu()
    y = y.contiguous().npu()
    hidden = hidden.contiguous().npu()
    wks = wks.contiguous().npu()
    gate = gate.contiguous().npu()
    sk_ws = sk_ws.contiguous().npu()
    if use_bias:
        b1_dev = b1_dev.contiguous().npu()
        if b1_up_dev is not None:
            b1_up_dev = b1_up_dev.contiguous().npu()
        b2_dev = b2_dev.contiguous().npu()

    tx = create_tla_tensor(x, "row")
    w_layout = "row" if layout_id == LAYOUT_CANONICAL else "col"

    def _bias_to_bt(t: torch.Tensor) -> torch.Tensor:
        return t.float() if t.dtype != torch.float32 else t

    tw1t = create_tla_tensor(w1_dev, w_layout)
    if w1_up_dev is None:
        tw1_up = tw1t
    else:
        tw1_up = create_tla_tensor(w1_up_dev, w_layout)
    tw2t = create_tla_tensor(w2, w_layout)
    ty = create_tla_tensor(y, "row")
    th = create_tla_tensor(hidden, "row")
    twks = create_tla_tensor(wks, "row")
    tgate = create_tla_tensor(gate, "row")
    tsk = create_tla_tensor(sk_ws, "row")
    if use_bias:
        b1_bt = _bias_to_bt(b1_dev).contiguous().npu()
        b2_bt = _bias_to_bt(b2_dev).contiguous().npu()
        tb1 = create_tla_tensor(b1_bt, "vector")
        if b1_up_dev is None:
            tb1_up = tb1
        else:
            tb1_up = create_tla_tensor(
                _bias_to_bt(b1_up_dev).contiguous().npu(), "vector"
            )
        tb2 = create_tla_tensor(b2_bt, "vector")
    else:
        b1_dummy = torch.zeros(1, dtype=torch.float32, device="cpu").npu()
        b2_dummy = torch.zeros(1, dtype=torch.float32, device="cpu").npu()
        tb1 = create_tla_tensor(b1_dummy, "vector")
        tb1_up = tb1
        tb2 = create_tla_tensor(b2_dummy, "vector")

    sk_ub = 0
    if use_sk:
        from ffn.stages.down_streamk import sk_aiv_ub_reserve_bytes

        sk_ub = sk_aiv_ub_reserve_bytes(
            int(tiling_down.l1_tn),
            dtype_size_bytes(dtype_name),
            int(tiling_down.l1_tm),
        )
    acc_elems = up_fixpipe_acc_elems(int(tiling_up.l0_tm), int(tiling_up.l0_tn))
    acc_slots = 1
    if activation == "swiglu" and swiglu_mode == SWIGLU_DUAL:
        acc_slots = 2
        acc_elems = up_fixpipe_acc_elems(
            int(tiling_up.l0_tm), int(tiling_up.l0_tn), compact=True
        )
        dual_acc_bytes = swiglu_dual_ub_bytes(
            int(tiling_up.l0_tm), int(tiling_up.l0_tn)
        )
        if dual_acc_bytes + max(0, int(sk_ub)) + 1024 > int(UB_SIZE):
            raise ValueError(
                "SwiGLU dual UB budget exceeded: "
                f"2×compact acc={dual_acc_bytes}B "
                f"(l0={tiling_up.l0_tm}×{tiling_up.l0_tn}) + "
                f"sk_reserve={sk_ub}B > UB={UB_SIZE}B; "
                "shrink L0 tile or use --swiglu-mode single"
            )
    ub_slot, simd = _ub_slots(
        dtype_name,
        sk_ub_reserve=sk_ub,
        acc_elems=acc_elems or None,
        acc_slots=acc_slots,
    )
    if ub_slot < 1:
        raise ValueError(
            "UB out-slot empty after acc/SK reserve: "
            f"acc_elems={acc_elems}×{acc_slots}, sk_ub={sk_ub}, "
            f"dtype={dtype_name}; shrink tiling or drop StreamK"
        )
    if activation == "swiglu" and swiglu_mode == SWIGLU_SINGLE:
        simd = swiglu_simd_lanes(int(tiling_up.l1_tn))
    sw_up = choose_swizzle_up(
        m_run,
        up_n,
        l1_tm=int(tiling_up.l1_tm),
        l1_tn=int(tiling_up.l1_tn),
        block_num=int(bn),
        forbid_n_split=(activation == "swiglu" and swiglu_mode == SWIGLU_SINGLE),
    )
    sw_down = choose_swizzle_down(
        m_run,
        n_run,
        l1_tm=int(tiling_down.l1_tm),
        l1_tn=int(tiling_down.l1_tn),
        block_num=int(bn),
        forbid_n_split=(int(down_mode) == DOWN_FULL_B),
    )
    block_dim_ct = int(bn)
    mode_tag = (
        "dual"
        if (activation == "swiglu" and swiglu_mode == SWIGLU_DUAL)
        else ("single" if activation == "swiglu" else "-")
    )

    artifact = tla.compile(
        ffn_basic_kernel,
        tx,
        tw1t,
        tw1_up,
        tw2t,
        ty,
        th,
        twks,
        tgate,
        tsk,
        tb1,
        tb1_up,
        tb2,
        cfg,
        tiling_up,
        tiling_down,
        sw_up,
        sw_down,
        ub_slot,
        simd,
        block_dim_ct,
        acc_elems,
        options=_NPU_COMPILE_OPTS,
    )

    def _launch() -> None:
        artifact(
            tx,
            tw1t,
            tw1_up,
            tw2t,
            ty,
            th,
            twks,
            tgate,
            tsk,
            tb1,
            tb1_up,
            tb2,
            block_num=bn,
        )

    _launch()
    torch.npu.synchronize()

    bias_tag = "cube_bt" if use_bias else "none"
    if int(warmup) > 0 or int(iters) > 0:
        print("FFN_DSL_READY", flush=True)
        for _ in range(int(warmup)):
            _launch()
        torch.npu.synchronize()
        for _ in range(int(iters)):
            _launch()
        torch.npu.synchronize()
        print(
            f"DSL bench done shape=({m},{k},{h},{n}) dtype={dtype_name} "
            f"act={activation} bias={bias_tag} "
            f"down_mode={_down_mode_tag(use_sk, down_full)} "
            f"warmup={warmup} iters={iters}",
            flush=True,
        )
        return y[:m, :n].contiguous()

    # Keep host log parity with run_once for smoke callers.
    print(
        f"shape=({m},{k},{h},{n}) dtype={dtype_name} act={activation} "
        f"layout={'canonical' if layout_id == LAYOUT_CANONICAL else 'linear'} "
        f"swiglu={mode_tag} bias={bias_tag} up_path={UP_L0C2UB} "
        f"down_mode={_down_mode_tag(use_sk, down_full)} "
        f"block_num={bn} tiling_up=({tiling_up.l1_tm},{tiling_up.l1_tn},{tiling_up.l1_tk}) "
        f"tiling_down=({tiling_down.l1_tm},{tiling_down.l1_tn},{tiling_down.l1_tk}) "
        f"acc_slots={acc_slots} acc_elems={acc_elems}"
    )
    return y[:m, :n].contiguous()


def run_once(
    m: int,
    k: int,
    h: int,
    n: int,
    dtype_name: str,
    block_num: int,
    activation: str = "silu",
    seed: int = 0,
    skip_golden: bool = False,
    device: int = 0,
    use_bias: bool = False,
    bias_dtype: str = "fp32",
    layout: str = "linear",
    swiglu_mode_force: str = "auto",
    warmup: int = 0,
    iters: int = 0,
) -> bool:
    torch.npu.set_device(int(device))
    torch.manual_seed(seed)
    if str(dtype_name).lower() == "f32":
        raise ValueError("dtype f32 is not supported (x is fp16/bf16 only)")
    dtype = _dtype_map(dtype_name)
    act_pol = parse_activation(activation)
    up_n = act_pol.up_out_width(h)
    layout_id = (
        LAYOUT_CANONICAL if str(layout).lower() == "canonical" else LAYOUT_LINEAR
    )
    bdt = str(bias_dtype).lower()
    if bdt in ("same", "x"):
        bias_torch_dtype = dtype
    elif bdt in ("fp32", "f32", "float32"):
        bias_torch_dtype = torch.float32
    else:
        raise ValueError(f"bias_dtype={bias_dtype!r}; use 'fp32' or 'same' (=x dtype)")

    # Same input amplitude as ATK fused_ffn delivery (MindIE-scale).
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    x = torch.randn(m, k, generator=g, dtype=dtype) * 0.3
    if layout_id == LAYOUT_CANONICAL:
        w1 = torch.randn(k, up_n, generator=g, dtype=dtype) * 0.05
        w2 = torch.randn(h, n, generator=g, dtype=dtype) * 0.05
    else:
        w1 = torch.randn(up_n, k, generator=g, dtype=dtype) * 0.05
        w2 = torch.randn(n, h, generator=g, dtype=dtype) * 0.05
    if use_bias:
        b1 = (torch.randn(up_n, generator=g) * 0.1).to(dtype=bias_torch_dtype)
        b2 = (torch.randn(n, generator=g) * 0.1).to(dtype=bias_torch_dtype)
    else:
        b1 = None
        b2 = None

    bench_mode = int(warmup) > 0 or int(iters) > 0
    ref = None
    if not skip_golden and not bench_mode:
        if layout_id == LAYOUT_CANONICAL:
            w1_g = w1.transpose(0, 1).contiguous()
            w2_g = w2.transpose(0, 1).contiguous()
        else:
            w1_g, w2_g = w1, w2
        ref = golden_ffn(activation, x, w1_g, w2_g, dtype, bias1=b1, bias2=b2)

    y = run_from_tensors(
        x,
        w1,
        w2,
        activation=activation,
        bias1=b1,
        bias2=b2,
        layout=layout,
        block_num=block_num,
        device=device,
        swiglu_mode_force=swiglu_mode_force,
        warmup=warmup,
        iters=iters,
    )

    if skip_golden or bench_mode:
        return True

    passed, matched = _mixed_tolerance_bm_v2(y, ref)
    print(f"passed={passed} matched_ratio={matched:.6f}")
    return bool(passed)


def run(args: argparse.Namespace) -> int:
    ok = run_once(
        args.m,
        args.k,
        args.h,
        args.n,
        args.dtype,
        args.block_num,
        activation=args.activation,
        skip_golden=args.skip_golden,
        device=args.device,
        use_bias=args.bias,
        bias_dtype=args.bias_dtype,
        layout=args.layout,
        swiglu_mode_force=args.swiglu_mode,
        warmup=args.warmup,
        iters=args.iters,
    )
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Fused FFN end-to-end example (Up Matmul + Act + SyncAll + Down Matmul)"
        )
    )
    p.add_argument("--device", type=int, default=0, help="NPU device id")
    p.add_argument("--m", type=int, default=128)
    p.add_argument("--k", type=int, default=256)
    p.add_argument("--h", type=int, default=512)
    p.add_argument("--n", type=int, default=256)
    p.add_argument(
        "--dtype",
        choices=("f16", "bf16"),
        default="f16",
        help="Input/output dtype (fp16/bf16)",
    )
    p.add_argument(
        "--activation",
        choices=("silu", "gelu", "swiglu"),
        default="silu",
        help="Activation: silu / gelu / swiglu (Up path is always L0C→UB)",
    )
    p.add_argument(
        "--layout",
        choices=("linear", "canonical"),
        default="linear",
        help="Weight layout: linear [N,K] ColMajor / canonical [K,N] RowMajor",
    )
    p.add_argument(
        "--bias",
        action="store_true",
        help="Enable Cube BT bias (paired bias1/bias2; first-K mmad)",
    )
    p.add_argument(
        "--bias-dtype",
        choices=("fp32", "same"),
        default="fp32",
        help="Bias dtype: fp32 or same as --dtype (device BT uses fp32 path)",
    )
    p.add_argument(
        "--swiglu-mode",
        choices=("auto", "single", "dual"),
        default="auto",
        help="SwiGLU path: auto select single/dual; force single/dual for smoke",
    )
    p.add_argument("--block-num", type=int, default=-1)
    p.add_argument(
        "--skip-golden",
        action="store_true",
        help="Skip fp32 reference + allclose (launch/perf smoke only)",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="msprof bench: warm launches after prime (skips golden)",
    )
    p.add_argument(
        "--iters",
        type=int,
        default=0,
        help="msprof bench: timed launches after warmup (skips golden)",
    )
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
