# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""GDN-2 backward for CATLASS Python DSL: host interface and device kernels.

The public entry point is ``gdn2_bwd``: it validates the BNSD inputs, derives
the schedule (grids, core counts, group splits), allocates the seven outputs
and the per-shape workspace, binds them with DLPack, resolves the five
compiled artifacts from a kernel+dtype keyed LRU cache, and launches K1..K5 on
the current stream of the input device.  ``layout="BSND"`` accepts the same
tensors as ``[B, T, H, D]`` and transposes them to the internal BNSD layout on
the host (outside the kernels); the six token-major gradients are transposed
back before return.

Layout is BNSD, identical to ``reference.py``::

    q, k          [B, HK, T, 128]
    v, w_gate, do [B, HV, T, 128]
    g, b          [B, HG, T, 128]    HG == HB, both HK or both HV
    initial_state [B, HV, 128, 128]  fp32 or None (K-first)

The public entry point returns ``(dq, dk, dv, db, dw, dg, dh0)``; the first six
use the dtype of the corresponding input, ``dh0`` is fp32.

Device stages (each stage's GM slots are listed where it is launched):

K1 ``prep_kernel``   grid = B*HF*NT : exp(gc), exp(-gc), q*eg, k*e^-g,
                                     k*eg, b*eg, b*k*eg, k*exp(gl-gc),
                                     Aq, A = (I+L)^-1, W = A@(b*k*eg)
K2 ``fwd_kernel``    grid = B*HV    : U = A@(w*v), v_new = U - W@H,
                                     H' = exp(gl)*H + Kt^T@v_new
K3 ``bwd_kernel``    grid = B*HV    : dh recurrence, chunks in reverse order
K4 ``grad_kernel``   grid = B*HV*NT : dv_new, dAq, dq, dk, dv, dw, db, dgc
K5 ``reduce_kernel`` grid = B*HK*NT : GVA reduce + reverse cumsum + dtype cast

K2/K3 also have ``*_pair`` variants that interleave two ``(b, hv)`` state chains
per block (R16) and a chunk-level software pipeline (R11); both are launched by
the host section below under the same stream protocol as the base kernels.

Every GM tensor is bound with a fixed one-dimensional shape (128 elements) and
addressed with explicit ``Int64`` offsets derived from the runtime shape
arguments, so a single compiled artifact serves every ``T``.  Each kernel is
self-contained: it reads only this module's constants and helpers, never a
mutable module-level tile/shape configuration.
"""

from __future__ import annotations

import math
import numbers
import os
import threading
import time
from collections import OrderedDict
from threading import Lock

import torch
import torch_npu  # noqa: F401
from catlass import tla

BT = 64
KD = 128  # K = V = 128
# R16: interleave two chains per K2 block (see ``fwd_kernel_pair``).  Set
# ``GDN_PAIR=0`` to launch the historical one-chain-per-block kernel.
_PAIR = os.environ.get("GDN_PAIR", "1") != "0"


# --------------------------------------------------------------------------- #
# IR helpers (evaluated at trace time)
# --------------------------------------------------------------------------- #
def sub(x, row, col, rows, cols):
    return tla.get_tile(x, tla.make_coord(row, col), tla.make_shape(rows, cols))


def tensor(ptr, rows, cols, dtype=tla.Float32, tag=tla.arch.RowMajor):
    return tla.make_tensor(
        tla.recast_ptr(ptr, dtype=dtype),
        tla.make_layout_with_tag((rows, cols), dtype, tag),
    )


def gm_view(x, offset, rows, cols, pitch):
    """RowMajor view of a GM buffer with the given row pitch."""
    return tla.make_tensor(
        x.ptr + offset,
        tla.make_layout(
            tla.make_shape(rows, cols),
            tla.make_stride(pitch, 1),
            layoutTag=tla.arch.RowMajor,
        ),
    )


def gm_view_t(x, offset, rows, cols, pitch):
    """Transposed view: logical (rows, cols) over a (cols, rows) row-major
    buffer with the given pitch."""
    return tla.make_tensor(
        x.ptr + offset,
        tla.make_layout(
            tla.make_shape(rows, cols),
            tla.make_stride(1, pitch),
            layoutTag=tla.arch.ColumnMajor,
        ),
    )


CAST_ROUND = tla.params.CastParams(
    sat_mode=tla.params.SatMode.NOSAT, round_mode=tla.params.RoundMode.CAST_ROUND
)
PLAIN_CAST = tla.params.CastParams()
UNPACK16 = tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_UNPACK_B16)
PACK16 = tla.params.NormalStoreParams(store_dist=tla.params.StoreDist.DIST_PACK_B32)
HF32_OFF = tla.params.HF32Mode.HF32_DISABLE
# Per-tensor fixpipe scale used by K4 to fold the ``-1`` of ``dW = -dv_new@H^T``
# into the L0C->GM relay of ``dv_new`` (exact fp32 negation: the multiply is by
# a power of two, so every bit pattern is preserved up to the sign bit).
NEG_SCALE = tla.params.CopyL0C2DstParams(
    quant_mode=tla.params.QuantMode.PER_TENSOR, quant_scale=-1.0
)
# The five GVA partials (``r_dq/r_dks/r_dkt/r_dkbg/r_dr``) are stored per value
# head into the global (b, hv, chunk) slot (``sbase``) and summed by K5 in
# ascending hv order, so the reduction order is fixed run to run.


def ld_f32(tile, mode):
    """Load a 64-wide tile as 64 fp32 lanes.  mode: 0=f32, 1=bf16, 2=f16."""
    if mode == 0:
        return tile.load()
    # Both 16-bit modes must use the unpacking load distribution: a plain
    # load keeps two halves packed per 32-bit lane, so the fp32 cast reads
    # the pair as one fp32 word (garbage, including NaN bit patterns).
    if mode == 1:
        return tile.load(UNPACK16).to(tla.Float32, PLAIN_CAST)
    return tile.load(UNPACK16).to(tla.Float32, PLAIN_CAST)


def st_low(tile, value, mode):
    if mode == 0:
        tile.store(value)
    elif mode == 1:
        tile.store(value.to(tla.BFloat16, CAST_ROUND), PACK16)
    else:
        tile.store(value.to(tla.Float16, CAST_ROUND), PACK16)


def native_dtype(mode):
    if mode == 0:
        return tla.Float32
    if mode == 1:
        return tla.BFloat16
    return tla.Float16


def to_out(ptr, rows, cols, mode):
    """On-chip tensor view for writing one output/precision cast."""
    return tensor(ptr, rows, cols, native_dtype(mode))


# --------------------------------------------------------------------------- #
# K1: preparation
# --------------------------------------------------------------------------- #
# UB offsets in Float32 elements.
U_NAT1 = 0
U_NAT2 = 8192
U_GC = 16384
U_F1 = 24576
U_F2 = 32768
U_S0 = 40960
U_S1 = 45056
U_VEC = 49152
U_TOTAL = 49664

# L1 offsets in Float32 elements.
L_I = 0
L_N = 4096
L_P = 8192
L_PN = 12288
L_OUT = 16384
L_X = 20480
L_Y = 28672
L_TOTAL = 36864

# L0C offsets in Float32 elements.
C_S = 0
C_A = 4096
C_P = 8192
C_W = 12288
C_TOTAL = 20480


@tla.jit
def zero_tail(tile, valid, mode):
    """Zero rows [valid, 64) of a 64x128 native staging tile."""
    if valid < 64:
        with tla.vec.func(mode="simd"):
            z = tla.full(0.0, tla.Float32)
            for pr in tla.range(valid, 64):
                for pc in tla.range_constexpr(2):
                    st_low(sub(tile, pr, pc * 64, 1, 64), z, mode)


@tla.kernel(auto_sync="v0")
def prep_kernel(
    gm_q: tla.Tensor,
    gm_k: tla.Tensor,
    gm_g: tla.Tensor,
    gm_b: tla.Tensor,
    w_eg: tla.Tensor,
    w_eng: tla.Tensor,
    w_qs: tla.Tensor,
    w_ks: tla.Tensor,
    w_kb: tla.Tensor,
    w_ke: tla.Tensor,
    w_be: tla.Tensor,
    w_kt: tla.Tensor,
    w_wy: tla.Tensor,
    w_aq: tla.Tensor,
    w_am: tla.Tensor,
    w_nm: tla.Tensor,
    w_egl: tla.Tensor,
    t_xq: tla.Tensor,
    t_xk: tla.Tensor,
    t_ys: tla.Tensor,
    gm_id: tla.Tensor,
    length: int,
    hk: int,
    hg: int,
    hb: int,
    hf: int,
    factors: int,
    QMODE: tla.Constexpr[int],
    GMODE: tla.Constexpr[int],
    BMODE: tla.Constexpr[int],
) -> None:
    ub = tla.allocate(U_TOTAL, tla.Float32, tla.AddressSpace.ub, 256)
    lp = tla.allocate(L_TOTAL, tla.Float32, tla.AddressSpace.l1, 512)
    l0a = tla.allocate(16384, tla.Float32, tla.AddressSpace.l0a, 512)
    l0b = tla.allocate(16384, tla.Float32, tla.AddressSpace.l0b, 512)
    l0c = tla.allocate(C_TOTAL, tla.Float32, tla.AddressSpace.l0c, 1024)

    vf = tla.cross_flag("k1v", mode=2)
    cf = tla.cross_flag("k1c", mode=2)
    nf = tla.cross_flag("k1n", mode=2)

    nblk = tla.arch.block_num()
    count = (factors - tla.arch.block_idx() + nblk - 1) // nblk

    with tla.cube():
        ref64 = gm_view(w_nm, tla.Int64(0), 64, 64, 64)
        ref128 = gm_view(w_kb, tla.Int64(0), 64, KD, KD)
        refy = gm_view_t(t_ys, tla.Int64(0), KD, 64, KD)
        l1_i = tla.make_tensor_like(lp + L_I, ref64, tla.arch.zN)
        tla.copy(l1_i, gm_view(gm_id, tla.Int64(0), 64, 64, 64))
        l1_x = tla.make_tensor_like(lp + L_X, ref128, tla.arch.zN)
        l1_y = tla.make_tensor_like(lp + L_Y, refy, tla.arch.nZ)
        l1_n = tla.make_tensor_like(lp + L_N, ref64, tla.arch.zN)
        l1_p0 = tla.make_tensor_like(lp + L_P, ref64, tla.arch.zN)
        l1_p1 = tla.make_tensor_like(lp + L_PN, ref64, tla.arch.zN)
        l1_out = tla.make_tensor_like(lp + L_OUT, ref64, tla.arch.zN)
        l1_b128 = tla.make_tensor_like(lp + L_Y, ref128, tla.arch.zN)
        for cstep in tla.range(count):
            f = tla.arch.block_idx() + cstep * nblk
            fo = tla.Int64(f) * tla.Int64(64) * tla.Int64(KD)
            fo64 = tla.Int64(f) * tla.Int64(64) * tla.Int64(64)

            # ---- scores: Aq = Xq @ Ys^T, Lmat = Xk @ Ys^T ----
            tla.cross_core_wait_flag(vf, tla.pipes.MTE2)
            c_s = tensor(l0c + C_S, 64, 64, tla.Float32, tla.arch.L0Clayout)
            for product in tla.range_constexpr(2):
                if tla.const_expr(product == 0):
                    src_x = t_xq
                    dst_s = U_S0
                else:
                    src_x = t_xk
                    dst_s = U_S1
                tla.copy(l1_x, gm_view(src_x, fo, 64, KD, KD))
                tla.copy(l1_y, gm_view_t(t_ys, fo, KD, 64, KD))
                tla.copy(tla.make_tensor_like(l0a, l1_x, tla.arch.zN), l1_x)
                tla.copy(tla.make_tensor_like(l0b, l1_y, tla.arch.nZ), l1_y)
                tla.mmad(
                    c_s,
                    tla.make_tensor_like(l0a, l1_x, tla.arch.zN),
                    tla.make_tensor_like(l0b, l1_y, tla.arch.nZ),
                    init_c=True,
                    hf32_mode=tla.params.HF32Mode.HF32_NEAREST_EVEN,
                )
                dst_t = tensor(ub + dst_s, 64, 64)
                tla.copy(
                    dst_t,
                    c_s,
                    tla.params.CopyL0C2DstParams(
                        l0c2ub_mode=tla.params.L0C2UBMode.NO_SPLIT_VEC_0
                    ),
                )
                tla.copy(
                    dst_t,
                    c_s,
                    tla.params.CopyL0C2DstParams(
                        l0c2ub_mode=tla.params.L0C2UBMode.NO_SPLIT_VEC_1
                    ),
                )
            tla.cross_core_set_flag(cf, tla.pipes.FIX)

            # ---- A = (I + L)^-1 by doubling on N = -strict_tril(L) ----
            tla.cross_core_wait_flag(nf, tla.pipes.MTE2)
            tla.copy(l1_n, gm_view(w_nm, fo64, 64, 64, 64))
            tla.copy(l1_p0, gm_view(w_nm, fo64, 64, 64, 64))
            c_a = tensor(l0c + C_A, 64, 64, tla.Float32, tla.arch.L0Clayout)
            c_p = tensor(l0c + C_P, 64, 64, tla.Float32, tla.arch.L0Clayout)
            # out is I plus N
            tla.copy(tla.make_tensor_like(l0a, l1_n, tla.arch.zN), l1_n)
            tla.copy(tla.make_tensor_like(l0b, l1_i, tla.arch.nZ), l1_i)
            tla.mmad(
                c_a,
                tla.make_tensor_like(l0a, l1_n, tla.arch.zN),
                tla.make_tensor_like(l0b, l1_i, tla.arch.nZ),
                init_c=True,
                hf32_mode=tla.params.HF32Mode.HF32_NEAREST_EVEN,
            )
            tla.copy(tla.make_tensor_like(l0a, l1_i, tla.arch.zN), l1_i)
            tla.mmad(
                c_a,
                tla.make_tensor_like(l0a, l1_i, tla.arch.zN),
                tla.make_tensor_like(l0b, l1_i, tla.arch.nZ),
                init_c=False,
                hf32_mode=tla.params.HF32Mode.HF32_NEAREST_EVEN,
            )
            tla.copy(l1_out, c_a)
            for step in tla.range_constexpr(5):
                if tla.const_expr(step % 2 == 0):
                    l1_src = l1_p0
                    l1_dst = l1_p1
                else:
                    l1_src = l1_p1
                    l1_dst = l1_p0
                tla.copy(tla.make_tensor_like(l0a, l1_src, tla.arch.zN), l1_src)
                tla.copy(tla.make_tensor_like(l0b, l1_src, tla.arch.nZ), l1_src)
                tla.mmad(
                    c_p,
                    tla.make_tensor_like(l0a, l1_src, tla.arch.zN),
                    tla.make_tensor_like(l0b, l1_src, tla.arch.nZ),
                    init_c=True,
                    hf32_mode=tla.params.HF32Mode.HF32_NEAREST_EVEN,
                )
                tla.copy(l1_dst, c_p)
                tla.copy(tla.make_tensor_like(l0a, l1_out, tla.arch.zN), l1_out)
                tla.copy(tla.make_tensor_like(l0b, l1_dst, tla.arch.nZ), l1_dst)
                tla.mmad(
                    c_a,
                    tla.make_tensor_like(l0a, l1_out, tla.arch.zN),
                    tla.make_tensor_like(l0b, l1_dst, tla.arch.nZ),
                    init_c=False,
                    hf32_mode=tla.params.HF32Mode.HF32_NEAREST_EVEN,
                )
                tla.copy(l1_out, c_a)
            tla.copy(gm_view(w_am, fo64, 64, 64, 64), c_a)

            # ---- W = A @ kbg ----
            tla.copy(l1_b128, gm_view(w_kb, fo, 64, KD, KD))
            tla.copy(tla.make_tensor_like(l0a, l1_out, tla.arch.zN), l1_out)
            tla.copy(tla.make_tensor_like(l0b, l1_b128, tla.arch.nZ), l1_b128)
            c_w = tensor(l0c + C_W, 64, KD, tla.Float32, tla.arch.L0Clayout)
            tla.mmad(
                c_w,
                tla.make_tensor_like(l0a, l1_out, tla.arch.zN),
                tla.make_tensor_like(l0b, l1_b128, tla.arch.nZ),
                init_c=True,
                hf32_mode=tla.params.HF32Mode.HF32_NEAREST_EVEN,
            )
            tla.copy(gm_view(w_wy, fo, 64, KD, KD), c_w)
        tla.pipe_barrier(tla.pipes.ALL)

    with tla.vector():
        natg = tensor(ub + U_NAT1, 64, KD, native_dtype(GMODE))
        natq = tensor(ub + U_NAT1, 64, KD, native_dtype(QMODE))
        natb = tensor(ub + U_NAT2, 64, KD, native_dtype(BMODE))
        gd = tensor(ub + U_GC, 64, KD)
        f1 = tensor(ub + U_F1, 64, KD)
        f2 = tensor(ub + U_F2, 64, KD)
        s0 = tensor(ub + U_S0, 64, 64)
        s1 = tensor(ub + U_S1, 64, 64)
        vear = tensor(ub + U_VEC, 1, KD)
        vera = tensor(ub + U_VEC + 128, 1, KD)
        vegl = tensor(ub + U_VEC + 256, 1, KD)
        for vstep in tla.range(count):
            f = tla.arch.block_idx() + vstep * nblk
            fo = tla.Int64(f) * tla.Int64(64) * tla.Int64(KD)
            fo64 = tla.Int64(f) * tla.Int64(64) * tla.Int64(64)
            nblk_t = (length + 63) // 64
            blk = f % nblk_t
            head = f // nblk_t
            batch = head // hf
            fh = head % hf
            kh = fh // (hf // hk)
            gh = fh
            if hg == hk:
                gh = kh
            bh = fh
            if hb == hk:
                bh = kh
            valid = length - blk * 64
            if valid > 64:
                valid = 64
            goff = (
                (tla.Int64(batch) * tla.Int64(hg) + tla.Int64(gh)) * tla.Int64(length)
                + tla.Int64(blk) * tla.Int64(64)
            ) * tla.Int64(KD)
            boff = (
                (tla.Int64(batch) * tla.Int64(hb) + tla.Int64(bh)) * tla.Int64(length)
                + tla.Int64(blk) * tla.Int64(64)
            ) * tla.Int64(KD)
            qoff = (
                (tla.Int64(batch) * tla.Int64(hk) + tla.Int64(kh)) * tla.Int64(length)
                + tla.Int64(blk) * tla.Int64(64)
            ) * tla.Int64(KD)

            # --- gc = cumulative sum of g over the 64 rows ---
            zero_tail(natg, valid, GMODE)
            tla.copy(sub(natg, 0, 0, valid, KD), gm_view(gm_g, goff, valid, KD, KD))
            with tla.vec.func(mode="simd"):
                for pc in tla.range_constexpr(2):
                    acc = tla.full(0.0, tla.Float32)
                    for r in tla.range(64):
                        acc = acc + ld_f32(sub(natg, r, pc * 64, 1, 64), GMODE)
                        sub(gd, r, pc * 64, 1, 64).store(acc)
            # R37: ``gc`` is not read by any later kernel (K2..K5 recompute what
            # they need from ``eg``/``eng``), so the GM tile is not stored.

            # --- anchors: aref = gc[32]; ear = exp(-aref); era = exp(aref) ---
            with tla.vec.func(mode="simd"):
                for pc in tla.range_constexpr(2):
                    aref = sub(gd, 32, pc * 64, 1, 64).load()
                    sub(vear, 0, pc * 64, 1, 64).store(tla.exp(tla.neg(aref)))
                    sub(vera, 0, pc * 64, 1, 64).store(tla.exp(aref))
                    gl = sub(gd, 63, pc * 64, 1, 64).load()
                    sub(vegl, 0, pc * 64, 1, 64).store(tla.exp(gl))
            tla.copy(gm_view(w_egl, tla.Int64(f) * tla.Int64(KD), 1, KD, KD), vegl)

            # --- q: qs = q*exp(gc), and xq = qs*exp(-aref) ---
            zero_tail(natq, valid, QMODE)
            tla.copy(sub(natq, 0, 0, valid, KD), gm_view(gm_q, qoff, valid, KD, KD))
            with tla.vec.func(mode="simd"):
                for pc in tla.range_constexpr(2):
                    ear = sub(vear, 0, pc * 64, 1, 64).load()
                    for r in tla.range(64):
                        qf = ld_f32(sub(natq, r, pc * 64, 1, 64), QMODE)
                        eg = tla.exp(sub(gd, r, pc * 64, 1, 64).load())
                        qs = qf * eg
                        sub(f1, r, pc * 64, 1, 64).store(qs)
                        sub(f2, r, pc * 64, 1, 64).store(qs * ear)
            tla.copy(gm_view(w_qs, fo, 64, KD, KD), f1)
            tla.copy(gm_view(t_xq, fo, 64, KD, KD), f2)

            # --- k and b: ks, kt, ys, ke, be, kb ---
            zero_tail(natq, valid, QMODE)
            tla.copy(sub(natq, 0, 0, valid, KD), gm_view(gm_k, qoff, valid, KD, KD))
            zero_tail(natb, valid, BMODE)
            tla.copy(sub(natb, 0, 0, valid, KD), gm_view(gm_b, boff, valid, KD, KD))
            with tla.vec.func(mode="simd"):
                for pc in tla.range_constexpr(2):
                    egl = sub(vegl, 0, pc * 64, 1, 64).load()
                    for r in tla.range(64):
                        kf = ld_f32(sub(natq, r, pc * 64, 1, 64), QMODE)
                        eng = tla.exp(tla.neg(sub(gd, r, pc * 64, 1, 64).load()))
                        ks = kf * eng
                        sub(f1, r, pc * 64, 1, 64).store(ks)
                        sub(f2, r, pc * 64, 1, 64).store(ks * egl)
            tla.copy(gm_view(w_ks, fo, 64, KD, KD), f1)
            tla.copy(gm_view(w_kt, fo, 64, KD, KD), f2)
            with tla.vec.func(mode="simd"):
                for pc in tla.range_constexpr(2):
                    era = sub(vera, 0, pc * 64, 1, 64).load()
                    for r in tla.range(64):
                        kf = ld_f32(sub(natq, r, pc * 64, 1, 64), QMODE)
                        eng = tla.exp(tla.neg(sub(gd, r, pc * 64, 1, 64).load()))
                        sub(f1, r, pc * 64, 1, 64).store((kf * eng) * era)
                        sub(f2, r, pc * 64, 1, 64).store(
                            kf * tla.exp(sub(gd, r, pc * 64, 1, 64).load())
                        )
            tla.copy(gm_view(t_ys, fo, 64, KD, KD), f1)
            tla.copy(gm_view(w_ke, fo, 64, KD, KD), f2)
            with tla.vec.func(mode="simd"):
                for pc in tla.range_constexpr(2):
                    for r in tla.range(64):
                        kf = ld_f32(sub(natq, r, pc * 64, 1, 64), QMODE)
                        bf = ld_f32(sub(natb, r, pc * 64, 1, 64), BMODE)
                        be = bf * tla.exp(sub(gd, r, pc * 64, 1, 64).load())
                        sub(f1, r, pc * 64, 1, 64).store(be)
                        sub(f2, r, pc * 64, 1, 64).store(be * kf)
            tla.copy(gm_view(w_be, fo, 64, KD, KD), f1)
            tla.copy(gm_view(w_kb, fo, 64, KD, KD), f2)
            with tla.vec.func(mode="simd"):
                for pc in tla.range_constexpr(2):
                    ear = sub(vear, 0, pc * 64, 1, 64).load()
                    for r in tla.range(64):
                        kf = ld_f32(sub(natq, r, pc * 64, 1, 64), QMODE)
                        bf = ld_f32(sub(natb, r, pc * 64, 1, 64), BMODE)
                        eg = tla.exp(sub(gd, r, pc * 64, 1, 64).load())
                        sub(f1, r, pc * 64, 1, 64).store(((bf * kf) * eg) * ear)
            tla.copy(gm_view(t_xk, fo, 64, KD, KD), f1)

            # --- exp(gc) / exp(-gc) copies (used by K3/K4) ---
            with tla.vec.func(mode="simd"):
                for r in tla.range(64):
                    for pc in tla.range_constexpr(2):
                        gcr = sub(gd, r, pc * 64, 1, 64).load()
                        sub(f1, r, pc * 64, 1, 64).store(tla.exp(gcr))
                        sub(f2, r, pc * 64, 1, 64).store(tla.exp(tla.neg(gcr)))
            tla.copy(gm_view(w_eg, fo, 64, KD, KD), f1)
            tla.copy(gm_view(w_eng, fo, 64, KD, KD), f2)
            tla.cross_core_set_flag(vf, tla.pipes.MTE3)

            # --- mask the two score tiles ---
            tla.cross_core_wait_flag(cf, tla.pipes.VECTOR)
            with tla.vec.func(mode="simd"):
                idx = tla.arange(0, dtype=tla.Int32)
                z = tla.full(0.0, tla.Float32)
                for r in tla.range(64):
                    a = sub(s0, r, 0, 1, 64).load()
                    sub(s0, r, 0, 1, 64).store(tla.where(tla.cmp(idx, r, "gt"), z, a))
                    b = sub(s1, r, 0, 1, 64).load()
                    sub(s1, r, 0, 1, 64).store(
                        tla.where(tla.cmp(idx, r, "lt"), z - b, z)
                    )
            tla.copy(gm_view(w_aq, fo64, 64, 64, 64), s0)
            tla.copy(gm_view(w_nm, fo64, 64, 64, 64), s1)
            tla.cross_core_set_flag(nf, tla.pipes.MTE3)
        tla.pipe_barrier(tla.pipes.ALL)


# --------------------------------------------------------------------------- #
# K2: sequential state forward
# --------------------------------------------------------------------------- #
V2_H = 0  # 16384  128x128 : H slot, also the w*v staging area
V2_P = 16384  # 16384  128x128 : Kt^T @ v_new
V2_U = 32768  # 8192   [64,128] : U = A@(w*v)
V2_W = 40960  # 8192   [64,128] : W@H
V2_V = 49152  # 512    exp(gl)
V2_TOTAL = 49664

V2_LA64 = 0  # 4096
V2_LA128 = 4096  # 8192
V2_LB = 12288  # 8192
V2_LH = 20480  # 16384  128x128
V2_LTOTAL = 36864

V2_CU = 0  # 8192  [64,128]
V2_CW = 8192  # 8192  [64,128]
V2_CH0 = 16384  # 8192  [64,128]
V2_CH1 = 24576  # 8192  [64,128]
V2_CTOTAL = 32768


@tla.kernel(auto_sync="v0")
def fwd_kernel(
    gm_v: tla.Tensor,
    gm_w: tla.Tensor,
    w_am: tla.Tensor,
    w_wy: tla.Tensor,
    w_kt: tla.Tensor,
    w_egl: tla.Tensor,
    hbuf: tla.Tensor,
    vnbuf: tla.Tensor,
    s_wv: tla.Tensor,
    s_u: tla.Tensor,
    s_wh: tla.Tensor,
    s_hp: tla.Tensor,
    length: int,
    hk: int,
    hv: int,
    hf: int,
    ntask: int,
    task0: int,
    QMODE: tla.Constexpr[int],
) -> None:
    ub = tla.allocate(V2_TOTAL, tla.Float32, tla.AddressSpace.ub, 256)
    lp = tla.allocate(V2_LTOTAL, tla.Float32, tla.AddressSpace.l1, 512)
    l0a = tla.allocate(16384, tla.Float32, tla.AddressSpace.l0a, 512)
    l0b = tla.allocate(16384, tla.Float32, tla.AddressSpace.l0b, 512)
    l0c = tla.allocate(V2_CTOTAL, tla.Float32, tla.AddressSpace.l0c, 1024)

    f1 = tla.cross_flag("k2a", mode=2)
    f2 = tla.cross_flag("k2b", mode=2)
    f3 = tla.cross_flag("k2c", mode=2)
    f4 = tla.cross_flag("k2d", mode=2)

    nblk = tla.arch.block_num()
    count = (ntask - tla.arch.block_idx() + nblk - 1) // nblk
    nt = (length + 63) // 64

    with tla.cube():
        ref64 = gm_view(w_am, tla.Int64(0), 64, 64, 64)
        ref128 = gm_view(w_wy, tla.Int64(0), 64, KD, KD)
        refh = gm_view(hbuf, tla.Int64(0), KD, KD, KD)
        refkt = gm_view_t(w_kt, tla.Int64(0), 64, 64, KD)
        l1_a64 = tla.make_tensor_like(lp + V2_LA64, ref64, tla.arch.zN)
        l1_a128 = tla.make_tensor_like(lp + V2_LA128, ref128, tla.arch.zN)
        l1_kt = tla.make_tensor_like(lp + V2_LA128, refkt, tla.arch.nZ)
        l1_b = tla.make_tensor_like(lp + V2_LB, ref128, tla.arch.zN)
        l1_h = tla.make_tensor_like(lp + V2_LH, refh, tla.arch.zN)
        c_u = tensor(l0c + V2_CU, 64, KD, tla.Float32, tla.arch.L0Clayout)
        c_w = tensor(l0c + V2_CW, 64, KD, tla.Float32, tla.arch.L0Clayout)
        c_h0 = tensor(l0c + V2_CH0, 64, KD, tla.Float32, tla.arch.L0Clayout)
        c_h1 = tensor(l0c + V2_CH1, 64, KD, tla.Float32, tla.arch.L0Clayout)
        for cstep in tla.range(count):
            t = task0 + tla.arch.block_idx() + cstep * nblk
            b = t // hv
            vi = t % hv
            fh = vi
            if hf == hk:
                fh = vi // (hv // hk)
            n0 = tla.Int64(t) * tla.Int64(nt)
            n0h = tla.Int64(t) * (tla.Int64(nt) + tla.Int64(1))
            for n in tla.range(nt):
                fi = tla.Int64(b * hf) + tla.Int64(fh)
                fo = (fi * tla.Int64(nt) + tla.Int64(n)) * tla.Int64(64) * tla.Int64(KD)
                fo64 = (
                    (fi * tla.Int64(nt) + tla.Int64(n)) * tla.Int64(64) * tla.Int64(64)
                )
                hbase = (n0h + tla.Int64(n)) * tla.Int64(KD) * tla.Int64(KD)
                hnext = hbase + tla.Int64(KD) * tla.Int64(KD)
                vnbase = (n0 + tla.Int64(n)) * tla.Int64(64) * tla.Int64(KD)
                sbase = tla.Int64(t) * tla.Int64(64) * tla.Int64(KD)

                # ---- U = A@(w*v), W@H ----
                tla.cross_core_wait_flag(f1, tla.pipes.MTE2)
                tla.copy(l1_a64, gm_view(w_am, fo64, 64, 64, 64))
                tla.copy(l1_b, gm_view(s_wv, sbase, 64, KD, KD))
                tla.copy(tla.make_tensor_like(l0a, l1_a64, tla.arch.zN), l1_a64)
                tla.copy(tla.make_tensor_like(l0b, l1_b, tla.arch.nZ), l1_b)
                tla.mmad(
                    c_u,
                    tla.make_tensor_like(l0a, l1_a64, tla.arch.zN),
                    tla.make_tensor_like(l0b, l1_b, tla.arch.nZ),
                    init_c=True,
                    hf32_mode=tla.params.HF32Mode.HF32_NEAREST_EVEN,
                )
                tla.copy(l1_a128, gm_view(w_wy, fo, 64, KD, KD))
                tla.copy(l1_h, gm_view(hbuf, hbase, KD, KD, KD))
                tla.copy(tla.make_tensor_like(l0a, l1_a128, tla.arch.zN), l1_a128)
                tla.copy(tla.make_tensor_like(l0b, l1_h, tla.arch.nZ), l1_h)
                tla.mmad(
                    c_w,
                    tla.make_tensor_like(l0a, l1_a128, tla.arch.zN),
                    tla.make_tensor_like(l0b, l1_h, tla.arch.nZ),
                    init_c=True,
                    hf32_mode=tla.params.HF32Mode.HF32_NEAREST_EVEN,
                )
                tla.copy(gm_view(s_u, sbase, 64, KD, KD), c_u)
                tla.copy(gm_view(s_wh, sbase, 64, KD, KD), c_w)
                tla.cross_core_set_flag(f2, tla.pipes.FIX)

                # ---- Kt^T @ v_new ----
                tla.cross_core_wait_flag(f3, tla.pipes.MTE2)
                tla.copy(l1_kt, gm_view_t(w_kt, fo, 64, 64, KD))
                tla.copy(l1_b, gm_view(vnbuf, vnbase, 64, KD, KD))
                tla.copy(tla.make_tensor_like(l0a, l1_kt, tla.arch.zN), l1_kt)
                tla.copy(tla.make_tensor_like(l0b, l1_b, tla.arch.nZ), l1_b)
                tla.mmad(
                    c_h0,
                    tla.make_tensor_like(l0a, l1_kt, tla.arch.zN),
                    tla.make_tensor_like(l0b, l1_b, tla.arch.nZ),
                    init_c=True,
                    hf32_mode=tla.params.HF32Mode.HF32_NEAREST_EVEN,
                )
                tla.copy(l1_kt, gm_view_t(w_kt, fo + tla.Int64(64), 64, 64, KD))
                tla.copy(tla.make_tensor_like(l0a, l1_kt, tla.arch.zN), l1_kt)
                tla.mmad(
                    c_h1,
                    tla.make_tensor_like(l0a, l1_kt, tla.arch.zN),
                    tla.make_tensor_like(l0b, l1_b, tla.arch.nZ),
                    init_c=True,
                    hf32_mode=tla.params.HF32Mode.HF32_NEAREST_EVEN,
                )
                hp0 = tla.Int64(t) * tla.Int64(KD) * tla.Int64(KD)
                tla.copy(gm_view(s_hp, hp0, 64, KD, KD), c_h0)
                tla.copy(
                    gm_view(s_hp, hp0 + tla.Int64(64) * tla.Int64(KD), 64, KD, KD), c_h1
                )
                tla.cross_core_set_flag(f4, tla.pipes.FIX)
        tla.pipe_barrier(tla.pipes.ALL)

    with tla.vector():
        nat_w = tensor(ub + V2_H, 64, KD, native_dtype(QMODE))
        nat_v = tensor(ub + V2_H + 4096, 64, KD, native_dtype(QMODE))
        ub_wv = tensor(ub + V2_H + 8192, 64, KD)
        ub_u = tensor(ub + V2_H, 64, KD)
        ub_w = tensor(ub + V2_H + 8192, 64, KD)
        ub_h = tensor(ub + V2_H, KD, KD)
        ub_p = tensor(ub + V2_P, KD, KD)
        vgl = tensor(ub + V2_V, 1, KD)
        for vstep in tla.range(count):
            t = task0 + tla.arch.block_idx() + vstep * nblk
            b = t // hv
            vi = t % hv
            fh = vi
            if hf == hk:
                fh = vi // (hv // hk)
            n0 = tla.Int64(t) * tla.Int64(nt)
            n0h = tla.Int64(t) * (tla.Int64(nt) + tla.Int64(1))
            for n in tla.range(nt):
                fi = tla.Int64(b * hf) + tla.Int64(fh)
                hbase = (n0h + tla.Int64(n)) * tla.Int64(KD) * tla.Int64(KD)
                hnext = hbase + tla.Int64(KD) * tla.Int64(KD)
                vnbase = (n0 + tla.Int64(n)) * tla.Int64(64) * tla.Int64(KD)
                sbase = tla.Int64(t) * tla.Int64(64) * tla.Int64(KD)
                voff = (
                    (tla.Int64(b) * tla.Int64(hv) + tla.Int64(vi)) * tla.Int64(length)
                    + tla.Int64(n) * tla.Int64(64)
                ) * tla.Int64(KD)
                valid = length - n * 64
                if valid > 64:
                    valid = 64

                # ---- w*v ----
                zero_tail(nat_w, valid, QMODE)
                zero_tail(nat_v, valid, QMODE)
                tla.copy(
                    sub(nat_w, 0, 0, valid, KD), gm_view(gm_w, voff, valid, KD, KD)
                )
                tla.copy(
                    sub(nat_v, 0, 0, valid, KD), gm_view(gm_v, voff, valid, KD, KD)
                )
                with tla.vec.func(mode="simd"):
                    for r in tla.range(64):
                        for pc in tla.range_constexpr(2):
                            wf = ld_f32(sub(nat_w, r, pc * 64, 1, 64), QMODE)
                            vf = ld_f32(sub(nat_v, r, pc * 64, 1, 64), QMODE)
                            sub(ub_wv, r, pc * 64, 1, 64).store(wf * vf)
                tla.copy(gm_view(s_wv, sbase, 64, KD, KD), ub_wv)
                tla.cross_core_set_flag(f1, tla.pipes.MTE3)

                # ---- v_new = U - W@H ----
                tla.cross_core_wait_flag(f2, tla.pipes.MTE2)
                tla.copy(ub_u, gm_view(s_u, sbase, 64, KD, KD))
                tla.copy(ub_w, gm_view(s_wh, sbase, 64, KD, KD))
                with tla.vec.func(mode="simd"):
                    for r in tla.range(64):
                        for pc in tla.range_constexpr(2):
                            uu = sub(ub_u, r, pc * 64, 1, 64).load()
                            ww = sub(ub_w, r, pc * 64, 1, 64).load()
                            sub(ub_u, r, pc * 64, 1, 64).store(uu - ww)
                tla.copy(gm_view(vnbuf, vnbase, 64, KD, KD), ub_u)
                tla.cross_core_set_flag(f3, tla.pipes.MTE3)

                # ---- H' = exp(gl)*H + Kt^T@v_new ----
                tla.cross_core_wait_flag(f4, tla.pipes.MTE2)
                tla.copy(
                    vgl,
                    gm_view(
                        w_egl,
                        (fi * tla.Int64(nt) + tla.Int64(n)) * tla.Int64(KD),
                        1,
                        KD,
                        KD,
                    ),
                )
                tla.copy(
                    ub_p,
                    gm_view(
                        s_hp, tla.Int64(t) * tla.Int64(KD) * tla.Int64(KD), KD, KD, KD
                    ),
                )
                tla.copy(sub(ub_h, 0, 0, 64, KD), gm_view(hbuf, hbase, 64, KD, KD))
                tla.copy(
                    sub(ub_h, 64, 0, 64, KD),
                    gm_view(hbuf, hbase + tla.Int64(64) * tla.Int64(KD), 64, KD, KD),
                )
                with tla.vec.func(mode="simd"):
                    for r in tla.range(KD):
                        for pc in tla.range_constexpr(2):
                            gl = sub(vgl, 0, pc * 64, 1, 64).load()
                            hh = sub(ub_h, r, pc * 64, 1, 64).load()
                            pp = sub(ub_p, r, pc * 64, 1, 64).load()
                            sub(ub_h, r, pc * 64, 1, 64).store(hh * gl + pp)
                tla.copy(gm_view(hbuf, hnext, 64, KD, KD), sub(ub_h, 0, 0, 64, KD))
                tla.copy(
                    gm_view(hbuf, hnext + tla.Int64(64) * tla.Int64(KD), 64, KD, KD),
                    sub(ub_h, 64, 0, 64, KD),
                )
        tla.pipe_barrier(tla.pipes.ALL)


# --------------------------------------------------------------------------- #
# K2p (R16): two independent chains interleaved inside one block
# --------------------------------------------------------------------------- #
# ``fwd_kernel`` pays one fixed cross-core latency per chunk for *one* chain.
# With 32 chains on 28 cube cores a launch needs two full rounds, so the tail
# chains re-pay the whole latency story.  Here one block owns **two** disjoint
# chains and the six stage sequences are interleaved: chain B's cube/vector
# work fills the flag wait of chain A, so one latency chain covers two chains.
# No new on-chip buffer: H/P live in GM between chunks (``hbuf``/``s_hp``), and
# the UB/L1/L0C tiles of the two chains are reused in program order (auto_sync
# inserts the WAR waits, exactly as the two chains of one chunk do today).
# ``j`` is a ``range_constexpr`` index (a Python int at trace time), so the
# stage body is unrolled twice and only ``t`` and the flag instance differ.
@tla.kernel(auto_sync="v0")
def fwd_kernel_pair(
    gm_v: tla.Tensor,
    gm_w: tla.Tensor,
    w_am: tla.Tensor,
    w_wy: tla.Tensor,
    w_kt: tla.Tensor,
    w_egl: tla.Tensor,
    hbuf: tla.Tensor,
    vnbuf: tla.Tensor,
    s_wv: tla.Tensor,
    s_u: tla.Tensor,
    s_wh: tla.Tensor,
    s_hp: tla.Tensor,
    length: int,
    hk: int,
    hv: int,
    hf: int,
    ntask: int,
    task0: int,
    QMODE: tla.Constexpr[int],
) -> None:
    ub = tla.allocate(V2_TOTAL, tla.Float32, tla.AddressSpace.ub, 256)
    lp = tla.allocate(V2_LTOTAL, tla.Float32, tla.AddressSpace.l1, 512)
    l0a = tla.allocate(16384, tla.Float32, tla.AddressSpace.l0a, 512)
    l0b = tla.allocate(16384, tla.Float32, tla.AddressSpace.l0b, 512)
    l0c = tla.allocate(V2_CTOTAL, tla.Float32, tla.AddressSpace.l0c, 1024)

    fl1 = (tla.cross_flag("k2p1a", mode=2), tla.cross_flag("k2p1b", mode=2))
    fl2 = (tla.cross_flag("k2p2a", mode=2), tla.cross_flag("k2p2b", mode=2))
    fl3 = (tla.cross_flag("k2p3a", mode=2), tla.cross_flag("k2p3b", mode=2))
    fl4 = (tla.cross_flag("k2p4a", mode=2), tla.cross_flag("k2p4b", mode=2))
    fl5 = (tla.cross_flag("k2p5a", mode=2), tla.cross_flag("k2p5b", mode=2))

    nblk = tla.arch.block_num()
    npair = ntask // 2
    count = (npair - tla.arch.block_idx() + nblk - 1) // nblk
    nt = (length + 63) // 64

    with tla.cube():
        ref64 = gm_view(w_am, tla.Int64(0), 64, 64, 64)
        ref128 = gm_view(w_wy, tla.Int64(0), 64, KD, KD)
        refh = gm_view(hbuf, tla.Int64(0), KD, KD, KD)
        refkt = gm_view_t(w_kt, tla.Int64(0), 64, 64, KD)
        refkt128 = gm_view_t(w_kt, tla.Int64(0), KD, 64, KD)
        l1_a64 = tla.make_tensor_like(lp + V2_LA64, ref64, tla.arch.zN)
        l1_a128 = tla.make_tensor_like(lp + V2_LA128, ref128, tla.arch.zN)
        l1_kt = tla.make_tensor_like(lp + V2_LA128, refkt, tla.arch.nZ)
        l1_kt128 = tla.make_tensor_like(lp + V2_LA128, refkt128, tla.arch.nZ)
        l1_b = tla.make_tensor_like(lp + V2_LB, ref128, tla.arch.zN)
        l1_h = tla.make_tensor_like(lp + V2_LH, refh, tla.arch.zN)
        c_u = tensor(l0c + V2_CU, 64, KD, tla.Float32, tla.arch.L0Clayout)
        c_w = tensor(l0c + V2_CW, 64, KD, tla.Float32, tla.arch.L0Clayout)
        # R16b: one 128x128 MMAD for Kt^T@v_new instead of two 64-row halves.
        # Same MACs, but one L0 copy + one MMAD fewer per chain per chunk, and
        # the MMAD slots are the serialised resource in the paired kernel.
        # ``c_h`` covers the two former half tiles (V2_CH0..V2_CH1, contiguous).
        c_h = tensor(l0c + V2_CH0, KD, KD, tla.Float32, tla.arch.L0Clayout)
        for cstep in tla.range(count):
            ta = task0 + (tla.arch.block_idx() + cstep * nblk) * 2
            # ---- R11 prologue: U(0) = A@(w*v)(0) ----
            for j in tla.range_constexpr(2):
                t = ta + j
                b = t // hv
                vi = t % hv
                fh = vi
                if hf == hk:
                    fh = vi // (hv // hk)
                fi = tla.Int64(b * hf) + tla.Int64(fh)
                fo64 = fi * tla.Int64(nt) * tla.Int64(64) * tla.Int64(64)
                wvoff = tla.Int64(t) * tla.Int64(2) * tla.Int64(64) * tla.Int64(KD)
                sbase = tla.Int64(t) * tla.Int64(2) * tla.Int64(64) * tla.Int64(KD)
                tla.cross_core_wait_flag(fl1[j], tla.pipes.MTE2)
                tla.copy(l1_a64, gm_view(w_am, fo64, 64, 64, 64))
                tla.copy(l1_b, gm_view(s_wv, wvoff, 64, KD, KD))
                tla.copy(tla.make_tensor_like(l0a, l1_a64, tla.arch.zN), l1_a64)
                tla.copy(tla.make_tensor_like(l0b, l1_b, tla.arch.nZ), l1_b)
                tla.mmad(
                    c_u,
                    tla.make_tensor_like(l0a, l1_a64, tla.arch.zN),
                    tla.make_tensor_like(l0b, l1_b, tla.arch.nZ),
                    init_c=True,
                    hf32_mode=tla.params.HF32Mode.HF32_NEAREST_EVEN,
                )
                tla.copy(gm_view(s_u, sbase, 64, KD, KD), c_u)
            for n in tla.range(nt):
                # ---- R11: W@H  (H'(n) comes from the previous chunk, fl5) ----
                for j in tla.range_constexpr(2):
                    t = ta + j
                    b = t // hv
                    vi = t % hv
                    fh = vi
                    if hf == hk:
                        fh = vi // (hv // hk)
                    fi = tla.Int64(b * hf) + tla.Int64(fh)
                    fo = (
                        (fi * tla.Int64(nt) + tla.Int64(n))
                        * tla.Int64(64)
                        * tla.Int64(KD)
                    )
                    hbase = (
                        (tla.Int64(t) * (tla.Int64(nt) + tla.Int64(1)) + tla.Int64(n))
                        * tla.Int64(KD)
                        * tla.Int64(KD)
                    )
                    sbase = tla.Int64(t) * tla.Int64(64) * tla.Int64(KD)
                    tla.cross_core_wait_flag(fl5[j], tla.pipes.MTE2)
                    tla.copy(l1_a128, gm_view(w_wy, fo, 64, KD, KD))
                    tla.copy(l1_h, gm_view(hbuf, hbase, KD, KD, KD))
                    tla.copy(tla.make_tensor_like(l0a, l1_a128, tla.arch.zN), l1_a128)
                    tla.copy(tla.make_tensor_like(l0b, l1_h, tla.arch.nZ), l1_h)
                    tla.mmad(
                        c_w,
                        tla.make_tensor_like(l0a, l1_a128, tla.arch.zN),
                        tla.make_tensor_like(l0b, l1_h, tla.arch.nZ),
                        init_c=True,
                        hf32_mode=tla.params.HF32Mode.HF32_NEAREST_EVEN,
                    )
                    tla.copy(gm_view(s_wh, sbase, 64, KD, KD), c_w)
                    tla.cross_core_set_flag(fl2[j], tla.pipes.FIX)
                # ---- Kt^T @ v_new ----
                for j in tla.range_constexpr(2):
                    t = ta + j
                    b = t // hv
                    vi = t % hv
                    fh = vi
                    if hf == hk:
                        fh = vi // (hv // hk)
                    fi = tla.Int64(b * hf) + tla.Int64(fh)
                    fo = (
                        (fi * tla.Int64(nt) + tla.Int64(n))
                        * tla.Int64(64)
                        * tla.Int64(KD)
                    )
                    vnbase = (
                        (tla.Int64(t) * tla.Int64(nt) + tla.Int64(n))
                        * tla.Int64(64)
                        * tla.Int64(KD)
                    )
                    tla.cross_core_wait_flag(fl3[j], tla.pipes.MTE2)
                    tla.copy(l1_kt128, gm_view_t(w_kt, fo, KD, 64, KD))
                    tla.copy(l1_b, gm_view(vnbuf, vnbase, 64, KD, KD))
                    tla.copy(tla.make_tensor_like(l0a, l1_kt128, tla.arch.zN), l1_kt128)
                    tla.copy(tla.make_tensor_like(l0b, l1_b, tla.arch.nZ), l1_b)
                    tla.mmad(
                        c_h,
                        tla.make_tensor_like(l0a, l1_kt128, tla.arch.zN),
                        tla.make_tensor_like(l0b, l1_b, tla.arch.nZ),
                        init_c=True,
                        hf32_mode=tla.params.HF32Mode.HF32_NEAREST_EVEN,
                    )
                    hp0 = tla.Int64(t) * tla.Int64(KD) * tla.Int64(KD)
                    tla.copy(gm_view(s_hp, hp0, KD, KD, KD), c_h)
                    tla.cross_core_set_flag(fl4[j], tla.pipes.FIX)
                # ---- R11: U(n+1) = A@(w*v)(n+1), issued before the next H wait ----
                n1 = n + 1
                if n1 >= nt:
                    n1 = nt - 1
                for j in tla.range_constexpr(2):
                    t = ta + j
                    b = t // hv
                    vi = t % hv
                    fh = vi
                    if hf == hk:
                        fh = vi // (hv // hk)
                    fi = tla.Int64(b * hf) + tla.Int64(fh)
                    fo64 = (
                        (fi * tla.Int64(nt) + tla.Int64(n1))
                        * tla.Int64(64)
                        * tla.Int64(64)
                    )
                    slot = (
                        (tla.Int64(t) * tla.Int64(2) + tla.Int64(n1 % 2))
                        * tla.Int64(64)
                        * tla.Int64(KD)
                    )
                    tla.cross_core_wait_flag(fl1[j], tla.pipes.MTE2)
                    tla.copy(l1_a64, gm_view(w_am, fo64, 64, 64, 64))
                    tla.copy(l1_b, gm_view(s_wv, slot, 64, KD, KD))
                    tla.copy(tla.make_tensor_like(l0a, l1_a64, tla.arch.zN), l1_a64)
                    tla.copy(tla.make_tensor_like(l0b, l1_b, tla.arch.nZ), l1_b)
                    tla.mmad(
                        c_u,
                        tla.make_tensor_like(l0a, l1_a64, tla.arch.zN),
                        tla.make_tensor_like(l0b, l1_b, tla.arch.nZ),
                        init_c=True,
                        hf32_mode=tla.params.HF32Mode.HF32_NEAREST_EVEN,
                    )
                    tla.copy(gm_view(s_u, slot, 64, KD, KD), c_u)
        tla.pipe_barrier(tla.pipes.ALL)

    with tla.vector():
        nat_w = tensor(ub + V2_H, 64, KD, native_dtype(QMODE))
        nat_v = tensor(ub + V2_H + 4096, 64, KD, native_dtype(QMODE))
        ub_wv = tensor(ub + V2_H + 8192, 64, KD)
        ub_u = tensor(ub + V2_H, 64, KD)
        ub_w = tensor(ub + V2_H + 8192, 64, KD)
        ub_h = tensor(ub + V2_H, KD, KD)
        ub_p = tensor(ub + V2_P, KD, KD)
        vgl = tensor(ub + V2_V, 1, KD)
        for vstep in tla.range(count):
            ta = task0 + (tla.arch.block_idx() + vstep * nblk) * 2
            # ---- R11 prologue: w*v(0) ----
            for j in tla.range_constexpr(2):
                t = ta + j
                valid = length
                if valid > 64:
                    valid = 64
                voff = tla.Int64(t) * tla.Int64(length) * tla.Int64(KD)
                wvoff = tla.Int64(t) * tla.Int64(2) * tla.Int64(64) * tla.Int64(KD)
                zero_tail(nat_w, valid, QMODE)
                zero_tail(nat_v, valid, QMODE)
                tla.copy(
                    sub(nat_w, 0, 0, valid, KD), gm_view(gm_w, voff, valid, KD, KD)
                )
                tla.copy(
                    sub(nat_v, 0, 0, valid, KD), gm_view(gm_v, voff, valid, KD, KD)
                )
                with tla.vec.func(mode="simd"):
                    for r in tla.range(64):
                        for pc in tla.range_constexpr(2):
                            wf = ld_f32(sub(nat_w, r, pc * 64, 1, 64), QMODE)
                            vf = ld_f32(sub(nat_v, r, pc * 64, 1, 64), QMODE)
                            sub(ub_wv, r, pc * 64, 1, 64).store(wf * vf)
                tla.copy(gm_view(s_wv, wvoff, 64, KD, KD), ub_wv)
                tla.cross_core_set_flag(fl1[j], tla.pipes.MTE3)
            # H(0) is the host-written initial state: prime the H flag once.
            for j in tla.range_constexpr(2):
                tla.cross_core_set_flag(fl5[j], tla.pipes.MTE3)
            for n in tla.range(nt):
                valid = length - n * 64
                if valid > 64:
                    valid = 64
                # ---- R11: w*v(n+1) prefetch (off the H chain) ----
                n1 = n + 1
                if n1 >= nt:
                    n1 = nt - 1
                for j in tla.range_constexpr(2):
                    t = ta + j
                    valid1 = length - n1 * 64
                    if valid1 > 64:
                        valid1 = 64
                    voff = (
                        tla.Int64(t) * tla.Int64(length) + tla.Int64(n1) * tla.Int64(64)
                    ) * tla.Int64(KD)
                    wvoff = (
                        (tla.Int64(t) * tla.Int64(2) + tla.Int64(n1 % 2))
                        * tla.Int64(64)
                        * tla.Int64(KD)
                    )
                    zero_tail(nat_w, valid1, QMODE)
                    zero_tail(nat_v, valid1, QMODE)
                    tla.copy(
                        sub(nat_w, 0, 0, valid1, KD),
                        gm_view(gm_w, voff, valid1, KD, KD),
                    )
                    tla.copy(
                        sub(nat_v, 0, 0, valid1, KD),
                        gm_view(gm_v, voff, valid1, KD, KD),
                    )
                    with tla.vec.func(mode="simd"):
                        for r in tla.range(64):
                            for pc in tla.range_constexpr(2):
                                wf = ld_f32(sub(nat_w, r, pc * 64, 1, 64), QMODE)
                                vf = ld_f32(sub(nat_v, r, pc * 64, 1, 64), QMODE)
                                sub(ub_wv, r, pc * 64, 1, 64).store(wf * vf)
                    tla.copy(gm_view(s_wv, wvoff, 64, KD, KD), ub_wv)
                    tla.cross_core_set_flag(fl1[j], tla.pipes.MTE3)
                # ---- v_new = U - W@H ----
                for j in tla.range_constexpr(2):
                    t = ta + j
                    sbase = tla.Int64(t) * tla.Int64(64) * tla.Int64(KD)
                    vnbase = (
                        (tla.Int64(t) * tla.Int64(nt) + tla.Int64(n))
                        * tla.Int64(64)
                        * tla.Int64(KD)
                    )
                    tla.cross_core_wait_flag(fl2[j], tla.pipes.MTE2)
                    sb_u = (
                        (tla.Int64(t) * tla.Int64(2) + tla.Int64(n % 2))
                        * tla.Int64(64)
                        * tla.Int64(KD)
                    )
                    tla.copy(ub_u, gm_view(s_u, sb_u, 64, KD, KD))
                    tla.copy(ub_w, gm_view(s_wh, sbase, 64, KD, KD))
                    with tla.vec.func(mode="simd"):
                        for r in tla.range(64):
                            for pc in tla.range_constexpr(2):
                                uu = sub(ub_u, r, pc * 64, 1, 64).load()
                                ww = sub(ub_w, r, pc * 64, 1, 64).load()
                                sub(ub_u, r, pc * 64, 1, 64).store(uu - ww)
                    tla.copy(gm_view(vnbuf, vnbase, 64, KD, KD), ub_u)
                    tla.cross_core_set_flag(fl3[j], tla.pipes.MTE3)
                # ---- H' = exp(gl)*H + Kt^T@v_new ----
                for j in tla.range_constexpr(2):
                    t = ta + j
                    b = t // hv
                    vi = t % hv
                    fh = vi
                    if hf == hk:
                        fh = vi // (hv // hk)
                    fi = tla.Int64(b * hf) + tla.Int64(fh)
                    hbase = (
                        (tla.Int64(t) * (tla.Int64(nt) + tla.Int64(1)) + tla.Int64(n))
                        * tla.Int64(KD)
                        * tla.Int64(KD)
                    )
                    hnext = hbase + tla.Int64(KD) * tla.Int64(KD)
                    tla.cross_core_wait_flag(fl4[j], tla.pipes.MTE2)
                    tla.copy(
                        vgl,
                        gm_view(
                            w_egl,
                            (fi * tla.Int64(nt) + tla.Int64(n)) * tla.Int64(KD),
                            1,
                            KD,
                            KD,
                        ),
                    )
                    tla.copy(
                        ub_p,
                        gm_view(
                            s_hp,
                            tla.Int64(t) * tla.Int64(KD) * tla.Int64(KD),
                            KD,
                            KD,
                            KD,
                        ),
                    )
                    tla.copy(sub(ub_h, 0, 0, 64, KD), gm_view(hbuf, hbase, 64, KD, KD))
                    tla.copy(
                        sub(ub_h, 64, 0, 64, KD),
                        gm_view(
                            hbuf, hbase + tla.Int64(64) * tla.Int64(KD), 64, KD, KD
                        ),
                    )
                    with tla.vec.func(mode="simd"):
                        gl0 = sub(vgl, 0, 0, 1, 64).load()
                        gl1 = sub(vgl, 0, 64, 1, 64).load()
                        for r in tla.range(KD):
                            hh0 = sub(ub_h, r, 0, 1, 64).load()
                            pp0 = sub(ub_p, r, 0, 1, 64).load()
                            sub(ub_h, r, 0, 1, 64).store(hh0 * gl0 + pp0)
                            hh1 = sub(ub_h, r, 64, 1, 64).load()
                            pp1 = sub(ub_p, r, 64, 1, 64).load()
                            sub(ub_h, r, 64, 1, 64).store(hh1 * gl1 + pp1)
                    tla.copy(gm_view(hbuf, hnext, 64, KD, KD), sub(ub_h, 0, 0, 64, KD))
                    tla.copy(
                        gm_view(
                            hbuf, hnext + tla.Int64(64) * tla.Int64(KD), 64, KD, KD
                        ),
                        sub(ub_h, 64, 0, 64, KD),
                    )
                    if n + 1 < nt:
                        tla.cross_core_set_flag(fl5[j], tla.pipes.MTE3)
        tla.pipe_barrier(tla.pipes.ALL)


# --------------------------------------------------------------------------- #
# K3: sequential state backward (chunks in reverse order)
# --------------------------------------------------------------------------- #
K3_NAT = 0  # 4096  native do staging
K3_DOP = 4096  # 8192  do' = scale*do
K3_P = 12288  # 8192  dh  [128,64]
K3_A = 20480  # 8192  Qs^T @ do'
K3_B = 28672  # 8192  W^T  @ dv_new
K3_T = 36864  # 8192  dv_new [64,128]
K3_VEC = 45056  # 128
K3_TOTAL = 45184

K3_LA64 = 0  # 4096
K3_LA128 = 4096  # 8192
K3_LB = 12288  # 8192
K3_LH = 20480  # 16384
K3_LTOTAL = 36864

K3_CDVN = 0  # 8192  [64,128]
K3_CA = 8192  # 16384 [128,128]
K3_CB = 24576  # 16384 [128,128]
K3_CTOTAL = 40960


@tla.kernel(auto_sync="v0")
def bwd_kernel(
    gm_do: tla.Tensor,
    w_aq: tla.Tensor,
    w_qs: tla.Tensor,
    w_wy: tla.Tensor,
    w_kt: tla.Tensor,
    w_egl: tla.Tensor,
    dhbuf: tla.Tensor,
    s_dop: tla.Tensor,
    s_dvn: tla.Tensor,
    s_dvnk: tla.Tensor,
    s_dha: tla.Tensor,
    s_dhb: tla.Tensor,
    length: int,
    hk: int,
    hv: int,
    hf: int,
    ntask: int,
    task0: int,
    scale: float,
    QMODE: tla.Constexpr[int],
) -> None:
    ub = tla.allocate(K3_TOTAL, tla.Float32, tla.AddressSpace.ub, 256)
    lp = tla.allocate(K3_LTOTAL, tla.Float32, tla.AddressSpace.l1, 512)
    l0a = tla.allocate(8192, tla.Float32, tla.AddressSpace.l0a, 512)
    l0b = tla.allocate(16384, tla.Float32, tla.AddressSpace.l0b, 512)
    l0c = tla.allocate(K3_CTOTAL, tla.Float32, tla.AddressSpace.l0c, 1024)

    f1 = tla.cross_flag("k3a", mode=2)
    f2 = tla.cross_flag("k3b", mode=2)
    f3 = tla.cross_flag("k3c", mode=2)
    f4 = tla.cross_flag("k3d", mode=2)

    nblk = tla.arch.block_num()
    count = (ntask - tla.arch.block_idx() + nblk - 1) // nblk
    nt = (length + 63) // 64

    with tla.cube():
        ref64 = gm_view(w_aq, tla.Int64(0), 64, 64, 64)
        ref128 = gm_view(w_kt, tla.Int64(0), 64, KD, KD)
        refh = gm_view(dhbuf, tla.Int64(0), KD, KD, KD)
        refa = gm_view_t(w_qs, tla.Int64(0), KD, 64, KD)
        l1_a64 = tla.make_tensor_like(lp + K3_LA64, ref64, tla.arch.nZ)
        l1_a128 = tla.make_tensor_like(lp + K3_LA128, ref128, tla.arch.zN)
        l1_aT = tla.make_tensor_like(lp + K3_LA128, refa, tla.arch.nZ)
        l1_b = tla.make_tensor_like(lp + K3_LB, ref128, tla.arch.zN)
        l1_h = tla.make_tensor_like(lp + K3_LH, refh, tla.arch.zN)
        c_dvn = tensor(l0c + K3_CDVN, 64, KD, tla.Float32, tla.arch.L0Clayout)
        c_a = tensor(l0c + K3_CA, KD, KD, tla.Float32, tla.arch.L0Clayout)
        c_b = tensor(l0c + K3_CB, KD, KD, tla.Float32, tla.arch.L0Clayout)
        for cstep in tla.range(count):
            t = task0 + tla.arch.block_idx() + cstep * nblk
            b = t // hv
            vi = t % hv
            fh = vi
            if hf == hk:
                fh = vi // (hv // hk)
            n0h = tla.Int64(t) * (tla.Int64(nt) + tla.Int64(1))
            sbase = tla.Int64(t) * tla.Int64(64) * tla.Int64(KD)
            for step in tla.range(nt):
                n = nt - 1 - step
                fi = tla.Int64(b * hf) + tla.Int64(fh)
                fo = (fi * tla.Int64(nt) + tla.Int64(n)) * tla.Int64(64) * tla.Int64(KD)
                fo64 = (
                    (fi * tla.Int64(nt) + tla.Int64(n)) * tla.Int64(64) * tla.Int64(64)
                )
                hprev = (
                    (n0h + tla.Int64(n) + tla.Int64(1)) * tla.Int64(KD) * tla.Int64(KD)
                )
                hcur = (n0h + tla.Int64(n)) * tla.Int64(KD) * tla.Int64(KD)

                # ---- dv_new = Aq^T@do' + Kt@dh_out ----
                tla.cross_core_wait_flag(f1, tla.pipes.MTE2)
                tla.copy(l1_a64, gm_view_t(w_aq, fo64, 64, 64, 64))
                tla.copy(l1_b, gm_view(s_dop, sbase, 64, KD, KD))
                tla.copy(tla.make_tensor_like(l0a, l1_a64, tla.arch.zN), l1_a64)
                tla.copy(tla.make_tensor_like(l0b, l1_b, tla.arch.nZ), l1_b)
                tla.mmad(
                    c_dvn,
                    tla.make_tensor_like(l0a, l1_a64, tla.arch.zN),
                    tla.make_tensor_like(l0b, l1_b, tla.arch.nZ),
                    init_c=True,
                    hf32_mode=tla.params.HF32Mode.HF32_NEAREST_EVEN,
                )
                tla.copy(l1_a128, gm_view(w_kt, fo, 64, KD, KD))
                tla.copy(l1_h, gm_view(dhbuf, hprev, KD, KD, KD))
                tla.copy(tla.make_tensor_like(l0a, l1_a128, tla.arch.zN), l1_a128)
                tla.copy(tla.make_tensor_like(l0b, l1_h, tla.arch.nZ), l1_h)
                tla.mmad(
                    c_dvn,
                    tla.make_tensor_like(l0a, l1_a128, tla.arch.zN),
                    tla.make_tensor_like(l0b, l1_h, tla.arch.nZ),
                    init_c=False,
                    hf32_mode=tla.params.HF32Mode.HF32_NEAREST_EVEN,
                )
                tla.copy(gm_view(s_dvnk, sbase, 64, KD, KD), c_dvn)
                tla.cross_core_set_flag(f2, tla.pipes.FIX)

                # ---- dh = exp(gl)*dh_out + Qs^T@do' - W^T@dv_new ----
                tla.cross_core_wait_flag(f3, tla.pipes.MTE2)
                tla.copy(l1_aT, gm_view_t(w_qs, fo, KD, 64, KD))
                tla.copy(l1_b, gm_view(s_dop, sbase, 64, KD, KD))
                tla.copy(tla.make_tensor_like(l0a, l1_aT, tla.arch.zN), l1_aT)
                tla.copy(tla.make_tensor_like(l0b, l1_b, tla.arch.nZ), l1_b)
                tla.mmad(
                    c_a,
                    tla.make_tensor_like(l0a, l1_aT, tla.arch.zN),
                    tla.make_tensor_like(l0b, l1_b, tla.arch.nZ),
                    init_c=True,
                    hf32_mode=tla.params.HF32Mode.HF32_NEAREST_EVEN,
                )
                tla.copy(gm_view(s_dha, hcur, KD, KD, KD), c_a)
                tla.copy(l1_aT, gm_view_t(w_wy, fo, KD, 64, KD))
                tla.copy(l1_b, gm_view(s_dvn, sbase, 64, KD, KD))
                tla.copy(tla.make_tensor_like(l0a, l1_aT, tla.arch.zN), l1_aT)
                tla.copy(tla.make_tensor_like(l0b, l1_b, tla.arch.nZ), l1_b)
                tla.mmad(
                    c_b,
                    tla.make_tensor_like(l0a, l1_aT, tla.arch.zN),
                    tla.make_tensor_like(l0b, l1_b, tla.arch.nZ),
                    init_c=True,
                    hf32_mode=tla.params.HF32Mode.HF32_NEAREST_EVEN,
                )
                tla.copy(gm_view(s_dhb, hcur, KD, KD, KD), c_b)
                tla.cross_core_set_flag(f4, tla.pipes.FIX)
        tla.pipe_barrier(tla.pipes.ALL)

    with tla.vector():
        nat_do = tensor(ub + K3_NAT, 64, KD, native_dtype(QMODE))
        ub_dop = tensor(ub + K3_DOP, 64, KD)
        ub_p = tensor(ub + K3_P, KD, 64)
        ub_a = tensor(ub + K3_A, KD, 64)
        ub_b = tensor(ub + K3_B, KD, 64)
        ub_t = tensor(ub + K3_T, 64, KD)
        vgl = tensor(ub + K3_VEC, 1, 64)
        for vstep in tla.range(count):
            t = task0 + tla.arch.block_idx() + vstep * nblk
            b = t // hv
            vi = t % hv
            fh = vi
            if hf == hk:
                fh = vi // (hv // hk)
            n0h = tla.Int64(t) * (tla.Int64(nt) + tla.Int64(1))
            sbase = tla.Int64(t) * tla.Int64(64) * tla.Int64(KD)
            for step in tla.range(nt):
                n = nt - 1 - step
                fi = tla.Int64(b * hf) + tla.Int64(fh)
                hcur = (n0h + tla.Int64(n)) * tla.Int64(KD) * tla.Int64(KD)
                dooff = (
                    (tla.Int64(b) * tla.Int64(hv) + tla.Int64(vi)) * tla.Int64(length)
                    + tla.Int64(n) * tla.Int64(64)
                ) * tla.Int64(KD)
                valid = length - n * 64
                if valid > 64:
                    valid = 64

                # ---- do' = scale*do ----
                zero_tail(nat_do, valid, QMODE)
                tla.copy(
                    sub(nat_do, 0, 0, valid, KD), gm_view(gm_do, dooff, valid, KD, KD)
                )
                with tla.vec.func(mode="simd"):
                    for r in tla.range(64):
                        for pc in tla.range_constexpr(2):
                            dv = ld_f32(sub(nat_do, r, pc * 64, 1, 64), QMODE)
                            sub(ub_dop, r, pc * 64, 1, 64).store(dv * scale)
                tla.copy(gm_view(s_dop, sbase, 64, KD, KD), ub_dop)
                tla.cross_core_set_flag(f1, tla.pipes.MTE3)

                # ---- dv_new ----
                tla.cross_core_wait_flag(f2, tla.pipes.MTE2)
                tla.copy(ub_t, gm_view(s_dvnk, sbase, 64, KD, KD))
                tla.copy(gm_view(s_dvn, sbase, 64, KD, KD), ub_t)
                tla.cross_core_set_flag(f3, tla.pipes.MTE3)

                # ---- dh ----
                tla.cross_core_wait_flag(f4, tla.pipes.MTE2)
                for half in tla.range_constexpr(2):
                    hcol = hcur + tla.Int64(half * 64)
                    hprev = hcur + tla.Int64(KD * KD) + tla.Int64(half * 64)
                    tla.copy(ub_a, gm_view(s_dha, hcol, KD, 64, KD))
                    tla.copy(ub_b, gm_view(s_dhb, hcol, KD, 64, KD))
                    tla.copy(ub_p, gm_view(dhbuf, hprev, KD, 64, KD))
                    tla.copy(
                        vgl,
                        gm_view(
                            w_egl,
                            (fi * tla.Int64(nt) + tla.Int64(n)) * tla.Int64(KD)
                            + tla.Int64(half * 64),
                            1,
                            64,
                            KD,
                        ),
                    )
                    with tla.vec.func(mode="simd"):
                        for r in tla.range(KD):
                            gl = sub(vgl, 0, 0, 1, 64).load()
                            pp = sub(ub_p, r, 0, 1, 64).load()
                            aa = sub(ub_a, r, 0, 1, 64).load()
                            bb = sub(ub_b, r, 0, 1, 64).load()
                            sub(ub_p, r, 0, 1, 64).store(pp * gl + aa - bb)
                    tla.copy(gm_view(dhbuf, hcol, KD, 64, KD), ub_p)
        tla.pipe_barrier(tla.pipes.ALL)


# --------------------------------------------------------------------------- #
# K3p (R16): two independent chains interleaved inside one block
# --------------------------------------------------------------------------- #
# Same transformation as ``fwd_kernel_pair``: the two chains' ``dv_new`` and
# ``dh`` stage sequences are interleaved so one cross-core latency chain covers
# two chains, halving the block count (16 blocks for 32 chains = one wave).
@tla.kernel(auto_sync="v0")
def bwd_kernel_pair(
    gm_do: tla.Tensor,
    w_aq: tla.Tensor,
    w_qs: tla.Tensor,
    w_wy: tla.Tensor,
    w_kt: tla.Tensor,
    w_egl: tla.Tensor,
    dhbuf: tla.Tensor,
    s_dop: tla.Tensor,
    s_dvn: tla.Tensor,
    s_dvnk: tla.Tensor,
    s_dha: tla.Tensor,
    s_dhb: tla.Tensor,
    length: int,
    hk: int,
    hv: int,
    hf: int,
    ntask: int,
    task0: int,
    scale: float,
    QMODE: tla.Constexpr[int],
) -> None:
    ub = tla.allocate(K3_TOTAL, tla.Float32, tla.AddressSpace.ub, 256)
    lp = tla.allocate(K3_LTOTAL, tla.Float32, tla.AddressSpace.l1, 512)
    l0a = tla.allocate(8192, tla.Float32, tla.AddressSpace.l0a, 512)
    l0b = tla.allocate(16384, tla.Float32, tla.AddressSpace.l0b, 512)
    l0c = tla.allocate(K3_CTOTAL, tla.Float32, tla.AddressSpace.l0c, 1024)

    fl1 = (tla.cross_flag("k3p1a", mode=2), tla.cross_flag("k3p1b", mode=2))
    fl4 = (tla.cross_flag("k3p4a", mode=2), tla.cross_flag("k3p4b", mode=2))
    fl5 = (tla.cross_flag("k3p5a", mode=2), tla.cross_flag("k3p5b", mode=2))

    nblk = tla.arch.block_num()
    npair = ntask // 2
    count = (npair - tla.arch.block_idx() + nblk - 1) // nblk
    nt = (length + 63) // 64

    with tla.cube():
        ref64 = gm_view(w_aq, tla.Int64(0), 64, 64, 64)
        ref128 = gm_view(w_kt, tla.Int64(0), 64, KD, KD)
        refh = gm_view(dhbuf, tla.Int64(0), KD, KD, KD)
        refa = gm_view_t(w_qs, tla.Int64(0), KD, 64, KD)
        l1_a64 = tla.make_tensor_like(lp + K3_LA64, ref64, tla.arch.nZ)
        l1_a128 = tla.make_tensor_like(lp + K3_LA128, ref128, tla.arch.zN)
        l1_aT = tla.make_tensor_like(lp + K3_LA128, refa, tla.arch.nZ)
        l1_b = tla.make_tensor_like(lp + K3_LB, ref128, tla.arch.zN)
        l1_h = tla.make_tensor_like(lp + K3_LH, refh, tla.arch.zN)
        c_dvn = tensor(l0c + K3_CDVN, 64, KD, tla.Float32, tla.arch.L0Clayout)
        c_a = tensor(l0c + K3_CA, KD, KD, tla.Float32, tla.arch.L0Clayout)
        c_b = tensor(l0c + K3_CB, KD, KD, tla.Float32, tla.arch.L0Clayout)
        for cstep in tla.range(count):
            ta = task0 + (tla.arch.block_idx() + cstep * nblk) * 2
            for step in tla.range(nt):
                n = nt - 1 - step
                # ---- dv_new = Aq^T@do' + Kt@dh_out  (both chains) ----
                for j in tla.range_constexpr(2):
                    t = ta + j
                    b = t // hv
                    vi = t % hv
                    fh = vi
                    if hf == hk:
                        fh = vi // (hv // hk)
                    fi = tla.Int64(b * hf) + tla.Int64(fh)
                    fo = (
                        (fi * tla.Int64(nt) + tla.Int64(n))
                        * tla.Int64(64)
                        * tla.Int64(KD)
                    )
                    fo64 = (
                        (fi * tla.Int64(nt) + tla.Int64(n))
                        * tla.Int64(64)
                        * tla.Int64(64)
                    )
                    hprev = (
                        (
                            tla.Int64(t) * (tla.Int64(nt) + tla.Int64(1))
                            + tla.Int64(n)
                            + tla.Int64(1)
                        )
                        * tla.Int64(KD)
                        * tla.Int64(KD)
                    )
                    sbase = tla.Int64(t) * tla.Int64(64) * tla.Int64(KD)
                    dopbase = (
                        (tla.Int64(t) * tla.Int64(nt) + tla.Int64(n))
                        * tla.Int64(64)
                        * tla.Int64(KD)
                    )
                    tla.cross_core_wait_flag(fl1[j], tla.pipes.MTE2)
                    tla.cross_core_wait_flag(fl5[j], tla.pipes.MTE2)
                    tla.copy(l1_a64, gm_view_t(w_aq, fo64, 64, 64, 64))
                    tla.copy(l1_b, gm_view(s_dop, dopbase, 64, KD, KD))
                    tla.copy(tla.make_tensor_like(l0a, l1_a64, tla.arch.zN), l1_a64)
                    tla.copy(tla.make_tensor_like(l0b, l1_b, tla.arch.nZ), l1_b)
                    tla.mmad(
                        c_dvn,
                        tla.make_tensor_like(l0a, l1_a64, tla.arch.zN),
                        tla.make_tensor_like(l0b, l1_b, tla.arch.nZ),
                        init_c=True,
                        hf32_mode=tla.params.HF32Mode.HF32_NEAREST_EVEN,
                    )
                    tla.copy(l1_a128, gm_view(w_kt, fo, 64, KD, KD))
                    tla.copy(l1_h, gm_view(dhbuf, hprev, KD, KD, KD))
                    tla.copy(tla.make_tensor_like(l0a, l1_a128, tla.arch.zN), l1_a128)
                    tla.copy(tla.make_tensor_like(l0b, l1_h, tla.arch.nZ), l1_h)
                    tla.mmad(
                        c_dvn,
                        tla.make_tensor_like(l0a, l1_a128, tla.arch.zN),
                        tla.make_tensor_like(l0b, l1_h, tla.arch.nZ),
                        init_c=False,
                        hf32_mode=tla.params.HF32Mode.HF32_NEAREST_EVEN,
                    )
                    tla.copy(gm_view(s_dvnk, sbase, 64, KD, KD), c_dvn)
                # ---- dh = exp(gl)*dh_out + Qs^T@do' - W^T@dv_new ----
                for j in tla.range_constexpr(2):
                    t = ta + j
                    b = t // hv
                    vi = t % hv
                    fh = vi
                    if hf == hk:
                        fh = vi // (hv // hk)
                    fi = tla.Int64(b * hf) + tla.Int64(fh)
                    fo = (
                        (fi * tla.Int64(nt) + tla.Int64(n))
                        * tla.Int64(64)
                        * tla.Int64(KD)
                    )
                    hcur = (
                        (tla.Int64(t) * (tla.Int64(nt) + tla.Int64(1)) + tla.Int64(n))
                        * tla.Int64(KD)
                        * tla.Int64(KD)
                    )
                    sbase = tla.Int64(t) * tla.Int64(64) * tla.Int64(KD)
                    dopbase = (
                        (tla.Int64(t) * tla.Int64(nt) + tla.Int64(n))
                        * tla.Int64(64)
                        * tla.Int64(KD)
                    )
                    tla.copy(l1_aT, gm_view_t(w_qs, fo, KD, 64, KD))
                    tla.copy(l1_b, gm_view(s_dop, dopbase, 64, KD, KD))
                    tla.copy(tla.make_tensor_like(l0a, l1_aT, tla.arch.zN), l1_aT)
                    tla.copy(tla.make_tensor_like(l0b, l1_b, tla.arch.nZ), l1_b)
                    tla.mmad(
                        c_a,
                        tla.make_tensor_like(l0a, l1_aT, tla.arch.zN),
                        tla.make_tensor_like(l0b, l1_b, tla.arch.nZ),
                        init_c=True,
                        hf32_mode=tla.params.HF32Mode.HF32_NEAREST_EVEN,
                    )
                    tla.copy(gm_view(s_dha, hcur, KD, KD, KD), c_a)
                    tla.copy(l1_aT, gm_view_t(w_wy, fo, KD, 64, KD))
                    tla.copy(l1_b, gm_view(s_dvnk, sbase, 64, KD, KD))
                    tla.copy(tla.make_tensor_like(l0a, l1_aT, tla.arch.zN), l1_aT)
                    tla.copy(tla.make_tensor_like(l0b, l1_b, tla.arch.nZ), l1_b)
                    tla.mmad(
                        c_b,
                        tla.make_tensor_like(l0a, l1_aT, tla.arch.zN),
                        tla.make_tensor_like(l0b, l1_b, tla.arch.nZ),
                        init_c=True,
                        hf32_mode=tla.params.HF32Mode.HF32_NEAREST_EVEN,
                    )
                    tla.copy(gm_view(s_dhb, hcur, KD, KD, KD), c_b)
                    tla.cross_core_set_flag(fl4[j], tla.pipes.FIX)
        tla.pipe_barrier(tla.pipes.ALL)

    with tla.vector():
        nat_do = tensor(ub + K3_NAT, 64, KD, native_dtype(QMODE))
        ub_dop = tensor(ub + K3_DOP, 64, KD)
        ub_p = tensor(ub + K3_P, KD, 64)
        ub_a = tensor(ub + K3_A, KD, 64)
        ub_b = tensor(ub + K3_B, KD, 64)
        vgl = tensor(ub + K3_VEC, 1, 64)
        for vstep in tla.range(count):
            ta = task0 + (tla.arch.block_idx() + vstep * nblk) * 2
            # ---- R11 prologue: do'(nt-1) and the dh(nt) ready flag ----
            n0 = nt - 1
            valid0 = length - n0 * 64
            if valid0 > 64:
                valid0 = 64
            for j in tla.range_constexpr(2):
                t = ta + j
                dooff = (
                    tla.Int64(t) * tla.Int64(length) + tla.Int64(n0) * tla.Int64(64)
                ) * tla.Int64(KD)
                dopbase = (
                    (tla.Int64(t) * tla.Int64(nt) + tla.Int64(n0))
                    * tla.Int64(64)
                    * tla.Int64(KD)
                )
                zero_tail(nat_do, valid0, QMODE)
                tla.copy(
                    sub(nat_do, 0, 0, valid0, KD), gm_view(gm_do, dooff, valid0, KD, KD)
                )
                with tla.vec.func(mode="simd"):
                    for r in tla.range(64):
                        for pc in tla.range_constexpr(2):
                            dv = ld_f32(sub(nat_do, r, pc * 64, 1, 64), QMODE)
                            sub(ub_dop, r, pc * 64, 1, 64).store(dv * scale)
                tla.copy(gm_view(s_dop, dopbase, 64, KD, KD), ub_dop)
                tla.cross_core_set_flag(fl1[j], tla.pipes.MTE3)
            # dh(nt) is the host-written initial state: prime the dh flag once.
            for j in tla.range_constexpr(2):
                tla.cross_core_set_flag(fl5[j], tla.pipes.MTE3)
            for step in tla.range(nt):
                n = nt - 1 - step
                valid = length - n * 64
                if valid > 64:
                    valid = 64
                # ---- R11: do'(n-1) prefetch (off the dh chain) ----
                n1 = n - 1
                if n1 < 0:
                    n1 = 0
                valid1 = length - n1 * 64
                if valid1 > 64:
                    valid1 = 64
                for j in tla.range_constexpr(2):
                    t = ta + j
                    dooff = (
                        tla.Int64(t) * tla.Int64(length) + tla.Int64(n1) * tla.Int64(64)
                    ) * tla.Int64(KD)
                    dopbase = (
                        (tla.Int64(t) * tla.Int64(nt) + tla.Int64(n1))
                        * tla.Int64(64)
                        * tla.Int64(KD)
                    )
                    zero_tail(nat_do, valid1, QMODE)
                    tla.copy(
                        sub(nat_do, 0, 0, valid1, KD),
                        gm_view(gm_do, dooff, valid1, KD, KD),
                    )
                    with tla.vec.func(mode="simd"):
                        for r in tla.range(64):
                            for pc in tla.range_constexpr(2):
                                dv = ld_f32(sub(nat_do, r, pc * 64, 1, 64), QMODE)
                                sub(ub_dop, r, pc * 64, 1, 64).store(dv * scale)
                    tla.copy(gm_view(s_dop, dopbase, 64, KD, KD), ub_dop)
                    if n1 < n:
                        tla.cross_core_set_flag(fl1[j], tla.pipes.MTE3)
                # ---- dh ----
                for j in tla.range_constexpr(2):
                    t = ta + j
                    b = t // hv
                    vi = t % hv
                    fh = vi
                    if hf == hk:
                        fh = vi // (hv // hk)
                    fi = tla.Int64(b * hf) + tla.Int64(fh)
                    hcur = (
                        (tla.Int64(t) * (tla.Int64(nt) + tla.Int64(1)) + tla.Int64(n))
                        * tla.Int64(KD)
                        * tla.Int64(KD)
                    )
                    tla.cross_core_wait_flag(fl4[j], tla.pipes.MTE2)
                    for half in tla.range_constexpr(2):
                        hcol = hcur + tla.Int64(half * 64)
                        hprev = hcur + tla.Int64(KD * KD) + tla.Int64(half * 64)
                        tla.copy(ub_a, gm_view(s_dha, hcol, KD, 64, KD))
                        tla.copy(ub_b, gm_view(s_dhb, hcol, KD, 64, KD))
                        tla.copy(ub_p, gm_view(dhbuf, hprev, KD, 64, KD))
                        tla.copy(
                            vgl,
                            gm_view(
                                w_egl,
                                (fi * tla.Int64(nt) + tla.Int64(n)) * tla.Int64(KD)
                                + tla.Int64(half * 64),
                                1,
                                64,
                                KD,
                            ),
                        )
                        with tla.vec.func(mode="simd"):
                            gl = sub(vgl, 0, 0, 1, 64).load()
                            for r in tla.range(KD):
                                pp = sub(ub_p, r, 0, 1, 64).load()
                                aa = sub(ub_a, r, 0, 1, 64).load()
                                bb = sub(ub_b, r, 0, 1, 64).load()
                                sub(ub_p, r, 0, 1, 64).store(pp * gl + aa - bb)
                        tla.copy(gm_view(dhbuf, hcol, KD, 64, KD), ub_p)
                    if n > 0:
                        tla.cross_core_set_flag(fl5[j], tla.pipes.MTE3)
        tla.pipe_barrier(tla.pipes.ALL)


# --------------------------------------------------------------------------- #
# K4: per-chunk gradients (cube heavy, masks only on the vector side)
# --------------------------------------------------------------------------- #
K4_H1 = 0  # 4096  native do / w staging
K4_F1 = 4096  # 8192  do' / w*v
K4_H2 = 12288  # 4096  native w / v
K4_H3 = 16384  # 4096  native v
K4_F2 = 20480  # 8192  w*v / duv
K4_OU = 28672  # 4096  output cast staging
K4_S = 32768  # 4096  [64,64] score mask
K4_TOTAL = 36864

K4_DVN = 0  # 8192  [64,128] zN  dv_new
K4_DW = 8192  # 8192  [64,128] zN  dW
K4_A128 = 16384  # 8192  [64,128] zN
K4_A64 = 24576  # 4096  [64,64]  zN
K4_DA = 28672  # 4096  [64,64]  zN  dA
K4_DLT = 32768  # 4096  [64,64]  zN  dLt
K4_B = 36864  # 8192  [64,128] zN
K4_NZ = 45056  # 8192  [128,64] nZ
K4_H = 53248  # 16384 [128,128] zN
K4_HT = 69632  # 16384 [128,128] nZ
K4_AMC = 86016  # 4096  [64,64]  nZ  cached A^T (loaded once, used 4x)
K4_KSC = 90112  # 8192  [64,128] zN  cached ks
K4_KBC = 98304  # 8192  [128,64] nZ  cached kb^T
K4_NVN = 106496  # 8192  [64,128] zN  -dv_new relayed through GM (R8)
K4_LTOTAL = 114688

K4_CDQ = 0  # 8192
K4_CDK = 8192  # 8192
K4_C1 = 16384  # 8192  dv_new -> dkbg
K4_C2 = 24576  # 4096  dAq -> dLt
K4_CKT = 28672  # 8192
K4_CDW = 36864  # 8192
K4_CDA = 45056  # 4096
K4_CDUV = 49152  # 8192
K4_CDL = 57344  # 4096
K4_CTOTAL = 61440


@tla.kernel(auto_sync="v0")
def grad_kernel(
    gm_v: tla.Tensor,
    gm_w: tla.Tensor,
    gm_do: tla.Tensor,
    gm_dv: tla.Tensor,
    gm_dw: tla.Tensor,
    w_aq: tla.Tensor,
    w_am: tla.Tensor,
    w_qs: tla.Tensor,
    w_ks: tla.Tensor,
    w_kb: tla.Tensor,
    w_kt: tla.Tensor,
    hbuf: tla.Tensor,
    vnbuf: tla.Tensor,
    dhbuf: tla.Tensor,
    s_dop: tla.Tensor,
    s_wv: tla.Tensor,
    s_daqr: tla.Tensor,
    s_daq: tla.Tensor,
    s_dlr: tla.Tensor,
    s_dl: tla.Tensor,
    r_dq: tla.Tensor,
    r_dks: tla.Tensor,
    r_dkt: tla.Tensor,
    r_duv: tla.Tensor,
    r_dkbg: tla.Tensor,
    r_dr: tla.Tensor,
    gm_nid: tla.Tensor,
    s_nar: tla.Tensor,
    length: int,
    hk: int,
    hv: int,
    hf: int,
    ntask: int,
    scale: float,
    QMODE: tla.Constexpr[int],
) -> None:
    ub = tla.allocate(K4_TOTAL, tla.Float32, tla.AddressSpace.ub, 256)
    lp = tla.allocate(K4_LTOTAL, tla.Float32, tla.AddressSpace.l1, 512)
    l0a = tla.allocate(8192, tla.Float32, tla.AddressSpace.l0a, 512)
    l0b = tla.allocate(16384, tla.Float32, tla.AddressSpace.l0b, 512)
    l0c = tla.allocate(K4_CTOTAL, tla.Float32, tla.AddressSpace.l0c, 1024)

    fa = tla.cross_flag("k4a", mode=2)
    fb = tla.cross_flag("k4b", mode=2)
    fc = tla.cross_flag("k4c", mode=2)
    fd = tla.cross_flag("k4d", mode=2)
    fe = tla.cross_flag("k4e", mode=2)
    ff = tla.cross_flag("k4f", mode=2)

    nblk = tla.arch.block_num()
    count = (ntask - tla.arch.block_idx() + nblk - 1) // nblk
    nt = (length + 63) // 64
    # R37: K4's task order is ``(b, hk, chunk)`` major and value head minor, so
    # the ``ratio`` blocks that read one K1 slot set are launched next to each
    # other and hit the same L2 lines.  With ``ratio == 1`` this is the identity.
    ratio = hv // hk

    with tla.cube():
        ref64 = gm_view(w_aq, tla.Int64(0), 64, 64, 64)
        ref128 = gm_view(w_kt, tla.Int64(0), 64, KD, KD)
        refh = gm_view(hbuf, tla.Int64(0), KD, KD, KD)
        refT64 = gm_view_t(w_aq, tla.Int64(0), 64, 64, 64)
        refT128 = gm_view_t(w_qs, tla.Int64(0), KD, 64, KD)
        l1_dvn = tla.make_tensor_like(lp + K4_DVN, ref128, tla.arch.zN)
        l1_dw = tla.make_tensor_like(lp + K4_DW, ref128, tla.arch.zN)
        l1_a128 = tla.make_tensor_like(lp + K4_A128, ref128, tla.arch.zN)
        l1_a64 = tla.make_tensor_like(lp + K4_A64, ref64, tla.arch.zN)
        l1_da = tla.make_tensor_like(lp + K4_DA, ref64, tla.arch.zN)
        l1_dlt = tla.make_tensor_like(lp + K4_DLT, ref64, tla.arch.zN)
        l1_b = tla.make_tensor_like(lp + K4_B, ref128, tla.arch.zN)
        l1_nz = tla.make_tensor_like(lp + K4_NZ, refT128, tla.arch.nZ)
        l1_nz64 = tla.make_tensor_like(lp + K4_NZ, refT64, tla.arch.nZ)
        l1_h = tla.make_tensor_like(lp + K4_H, refh, tla.arch.zN)
        l1_ht = tla.make_tensor_like(lp + K4_HT, refh, tla.arch.nZ)
        l1_amT = tla.make_tensor_like(lp + K4_AMC, refT64, tla.arch.nZ)
        l1_ks = tla.make_tensor_like(lp + K4_KSC, ref128, tla.arch.zN)
        l1_kbT = tla.make_tensor_like(lp + K4_KBC, refT128, tla.arch.nZ)
        l1_nvn = tla.make_tensor_like(lp + K4_NVN, ref128, tla.arch.zN)
        c_dq = tensor(l0c + K4_CDQ, 64, KD, tla.Float32, tla.arch.L0Clayout)
        c_dk = tensor(l0c + K4_CDK, 64, KD, tla.Float32, tla.arch.L0Clayout)
        c_1 = tensor(l0c + K4_C1, 64, KD, tla.Float32, tla.arch.L0Clayout)
        c_2 = tensor(l0c + K4_C2, 64, 64, tla.Float32, tla.arch.L0Clayout)
        c_kt = tensor(l0c + K4_CKT, 64, KD, tla.Float32, tla.arch.L0Clayout)
        c_dw = tensor(l0c + K4_CDW, 64, KD, tla.Float32, tla.arch.L0Clayout)
        c_da = tensor(l0c + K4_CDA, 64, 64, tla.Float32, tla.arch.L0Clayout)
        c_duv = tensor(l0c + K4_CDUV, 64, KD, tla.Float32, tla.arch.L0Clayout)
        c_dl = tensor(l0c + K4_CDL, 64, 64, tla.Float32, tla.arch.L0Clayout)
        for cstep in tla.range(count):
            ti = tla.arch.block_idx() + cstep * nblk
            g = ti // ratio
            j = ti % ratio
            b = g // (hk * nt)
            remg = g % (hk * nt)
            n = remg % nt
            vi = (remg // nt) * ratio + j
            # canonical per-value-head task id: every state/output slot address
            # below is still computed from ``t``, so the layout is unchanged.
            t = (b * hv + vi) * nt + n
            fh = vi
            if hf == hk:
                fh = vi // (hv // hk)
            fi = tla.Int64(b * hf) + tla.Int64(fh)
            fchunk = fi * tla.Int64(nt) + tla.Int64(n)
            fo = fchunk * tla.Int64(64) * tla.Int64(KD)
            fo64 = fchunk * tla.Int64(64) * tla.Int64(64)
            tt = tla.Int64(t)
            sbase = tt * tla.Int64(64) * tla.Int64(KD)
            sbase64 = tt * tla.Int64(64) * tla.Int64(64)
            hbase = (
                (
                    (tla.Int64(b * hv) + tla.Int64(vi)) * (tla.Int64(nt) + tla.Int64(1))
                    + tla.Int64(n)
                )
                * tla.Int64(KD)
                * tla.Int64(KD)
            )
            dhbase = hbase + tla.Int64(KD) * tla.Int64(KD)
            vnbase = tt * tla.Int64(64) * tla.Int64(KD)
            narbase = tt * tla.Int64(KD) * tla.Int64(KD)

            # ---- PH2 ----
            tla.cross_core_wait_flag(fa, tla.pipes.MTE2)
            # dv_new = Aq^T@do' + Kt@dh_out
            tla.copy(l1_nz64, gm_view_t(w_aq, fo64, 64, 64, 64))
            tla.copy(l1_b, gm_view(s_dop, sbase, 64, KD, KD))
            tla.copy(tla.make_tensor_like(l0a, l1_nz64, tla.arch.zN), l1_nz64)
            tla.copy(tla.make_tensor_like(l0b, l1_b, tla.arch.nZ), l1_b)
            tla.mmad(
                c_1,
                tla.make_tensor_like(l0a, l1_nz64, tla.arch.zN),
                tla.make_tensor_like(l0b, l1_b, tla.arch.nZ),
                init_c=True,
                hf32_mode=tla.params.HF32Mode.HF32_NEAREST_EVEN,
            )
            tla.copy(l1_a128, gm_view(w_kt, fo, 64, KD, KD))
            tla.copy(l1_h, gm_view(dhbuf, dhbase, KD, KD, KD))
            tla.copy(tla.make_tensor_like(l0a, l1_a128, tla.arch.zN), l1_a128)
            tla.copy(tla.make_tensor_like(l0b, l1_h, tla.arch.nZ), l1_h)
            tla.mmad(
                c_1,
                tla.make_tensor_like(l0a, l1_a128, tla.arch.zN),
                tla.make_tensor_like(l0b, l1_h, tla.arch.nZ),
                init_c=False,
                hf32_mode=tla.params.HF32Mode.HF32_NEAREST_EVEN,
            )
            tla.copy(l1_dvn, c_1)
            # -dv_new relayed through GM: the fixpipe per-tensor -1.0 scale is
            # an exact negation, so ``dW = -dv_new@H^T`` needs no second MMAD
            # against -I.  The read side sits after ``wait_flag(fc)``, and this
            # write is drained by ``set_flag(fb, FIX)`` below, so the
            # cube->vector->cube flag chain orders write before read.
            tla.copy(gm_view(s_nar, narbase, 64, KD, KD), c_1, params=NEG_SCALE)
            # dAq raw = do'@v_new^T   (do' is still in l1_b: loaded for dv_new)
            tla.copy(l1_nz, gm_view_t(vnbuf, vnbase, KD, 64, KD))
            tla.copy(tla.make_tensor_like(l0a, l1_b, tla.arch.zN), l1_b)
            tla.copy(tla.make_tensor_like(l0b, l1_nz, tla.arch.nZ), l1_nz)
            tla.mmad(
                c_2,
                tla.make_tensor_like(l0a, l1_b, tla.arch.zN),
                tla.make_tensor_like(l0b, l1_nz, tla.arch.nZ),
                init_c=True,
                hf32_mode=tla.params.HF32Mode.HF32_NEAREST_EVEN,
            )
            tla.copy(gm_view(s_daqr, sbase64, 64, 64, 64), c_2)
            # dq = do'@H^T
            tla.copy(l1_ht, gm_view_t(hbuf, hbase, KD, KD, KD))
            tla.copy(tla.make_tensor_like(l0a, l1_b, tla.arch.zN), l1_b)
            tla.copy(tla.make_tensor_like(l0b, l1_ht, tla.arch.nZ), l1_ht)
            tla.mmad(
                c_dq,
                tla.make_tensor_like(l0a, l1_b, tla.arch.zN),
                tla.make_tensor_like(l0b, l1_ht, tla.arch.nZ),
                init_c=True,
                hf32_mode=tla.params.HF32Mode.HF32_NEAREST_EVEN,
            )
            tla.cross_core_set_flag(fb, tla.pipes.FIX)

            # ---- PH4 ----
            tla.cross_core_wait_flag(fc, tla.pipes.MTE2)
            tla.copy(l1_a64, gm_view(s_daq, sbase64, 64, 64, 64))
            tla.copy(l1_ks, gm_view(w_ks, fo, 64, KD, KD))
            tla.copy(tla.make_tensor_like(l0a, l1_a64, tla.arch.zN), l1_a64)
            tla.copy(tla.make_tensor_like(l0b, l1_ks, tla.arch.nZ), l1_ks)
            tla.mmad(
                c_dq,
                tla.make_tensor_like(l0a, l1_a64, tla.arch.zN),
                tla.make_tensor_like(l0b, l1_ks, tla.arch.nZ),
                init_c=False,
                hf32_mode=tla.params.HF32Mode.HF32_NEAREST_EVEN,
            )
            tla.copy(gm_view(r_dq, sbase, 64, KD, KD), c_dq)
            # dks comes from dAq transpose times Qs
            tla.copy(l1_nz64, gm_view_t(s_daq, sbase64, 64, 64, 64))
            tla.copy(l1_b, gm_view(w_qs, fo, 64, KD, KD))
            tla.copy(tla.make_tensor_like(l0a, l1_nz64, tla.arch.zN), l1_nz64)
            tla.copy(tla.make_tensor_like(l0b, l1_b, tla.arch.nZ), l1_b)
            tla.mmad(
                c_dk,
                tla.make_tensor_like(l0a, l1_nz64, tla.arch.zN),
                tla.make_tensor_like(l0b, l1_b, tla.arch.nZ),
                init_c=True,
                hf32_mode=HF32_OFF,
            )
            tla.copy(gm_view(r_dks, sbase, 64, KD, KD), c_dk)
            # dW = -dv_new@H^T  (l1_ht still holds H^T from PH2: reuse it; the
            # negated dv_new comes from the GM relay, not from an extra MMAD)
            tla.copy(l1_nvn, gm_view(s_nar, narbase, 64, KD, KD))
            tla.copy(tla.make_tensor_like(l0a, l1_nvn, tla.arch.zN), l1_nvn)
            tla.copy(tla.make_tensor_like(l0b, l1_ht, tla.arch.nZ), l1_ht)
            tla.mmad(
                c_dw,
                tla.make_tensor_like(l0a, l1_nvn, tla.arch.zN),
                tla.make_tensor_like(l0b, l1_ht, tla.arch.nZ),
                init_c=True,
                hf32_mode=tla.params.HF32Mode.HF32_NEAREST_EVEN,
            )
            tla.copy(l1_dw, c_dw)
            # dk_tail comes from v_new times dh_out transpose
            tla.copy(l1_a128, gm_view(vnbuf, vnbase, 64, KD, KD))
            tla.copy(l1_ht, gm_view_t(dhbuf, dhbase, KD, KD, KD))
            tla.copy(tla.make_tensor_like(l0a, l1_a128, tla.arch.zN), l1_a128)
            tla.copy(tla.make_tensor_like(l0b, l1_ht, tla.arch.nZ), l1_ht)
            tla.mmad(
                c_kt,
                tla.make_tensor_like(l0a, l1_a128, tla.arch.zN),
                tla.make_tensor_like(l0b, l1_ht, tla.arch.nZ),
                init_c=True,
                hf32_mode=HF32_OFF,
            )
            tla.copy(gm_view(r_dkt, sbase, 64, KD, KD), c_kt)
            # dA adds dv_new times w*v transpose and dW times kbg transpose
            tla.copy(l1_nz, gm_view_t(s_wv, sbase, KD, 64, KD))
            tla.copy(tla.make_tensor_like(l0a, l1_dvn, tla.arch.zN), l1_dvn)
            tla.copy(tla.make_tensor_like(l0b, l1_nz, tla.arch.nZ), l1_nz)
            tla.mmad(
                c_da,
                tla.make_tensor_like(l0a, l1_dvn, tla.arch.zN),
                tla.make_tensor_like(l0b, l1_nz, tla.arch.nZ),
                init_c=True,
                hf32_mode=tla.params.HF32Mode.HF32_NEAREST_EVEN,
            )
            tla.copy(l1_kbT, gm_view_t(w_kb, fo, KD, 64, KD))
            tla.copy(tla.make_tensor_like(l0a, l1_dw, tla.arch.zN), l1_dw)
            tla.copy(tla.make_tensor_like(l0b, l1_kbT, tla.arch.nZ), l1_kbT)
            tla.mmad(
                c_da,
                tla.make_tensor_like(l0a, l1_dw, tla.arch.zN),
                tla.make_tensor_like(l0b, l1_kbT, tla.arch.nZ),
                init_c=False,
                hf32_mode=tla.params.HF32Mode.HF32_NEAREST_EVEN,
            )
            tla.copy(l1_da, c_da)
            # duv = A^T@dv_new   (A^T cached in l1_amT, reused by PH4/PH6)
            tla.copy(l1_amT, gm_view_t(w_am, fo64, 64, 64, 64))
            tla.copy(tla.make_tensor_like(l0a, l1_amT, tla.arch.zN), l1_amT)
            tla.copy(tla.make_tensor_like(l0b, l1_dvn, tla.arch.nZ), l1_dvn)
            tla.mmad(
                c_duv,
                tla.make_tensor_like(l0a, l1_amT, tla.arch.zN),
                tla.make_tensor_like(l0b, l1_dvn, tla.arch.nZ),
                init_c=True,
                hf32_mode=tla.params.HF32Mode.HF32_NEAREST_EVEN,
            )
            tla.copy(gm_view(r_duv, sbase, 64, KD, KD), c_duv)
            # dLt comes from A transpose times dA
            tla.copy(tla.make_tensor_like(l0a, l1_amT, tla.arch.zN), l1_amT)
            tla.copy(tla.make_tensor_like(l0b, l1_da, tla.arch.nZ), l1_da)
            tla.mmad(
                c_dl,
                tla.make_tensor_like(l0a, l1_amT, tla.arch.zN),
                tla.make_tensor_like(l0b, l1_da, tla.arch.nZ),
                init_c=True,
                hf32_mode=tla.params.HF32Mode.HF32_NEAREST_EVEN,
            )
            tla.copy(l1_dlt, c_dl)
            # dL is minus dLt times A transpose
            tla.copy(tla.make_tensor_like(l0a, l1_dlt, tla.arch.zN), l1_dlt)
            tla.copy(tla.make_tensor_like(l0b, l1_amT, tla.arch.nZ), l1_amT)
            tla.mmad(
                c_dl,
                tla.make_tensor_like(l0a, l1_dlt, tla.arch.zN),
                tla.make_tensor_like(l0b, l1_amT, tla.arch.nZ),
                init_c=True,
                hf32_mode=tla.params.HF32Mode.HF32_NEAREST_EVEN,
            )
            tla.copy(gm_view(s_dlr, sbase64, 64, 64, 64), c_dl)
            tla.cross_core_set_flag(fd, tla.pipes.FIX)

            # ---- PH6 ----
            tla.cross_core_wait_flag(fe, tla.pipes.MTE2)
            tla.copy(tla.make_tensor_like(l0a, l1_amT, tla.arch.zN), l1_amT)
            tla.copy(tla.make_tensor_like(l0b, l1_dw, tla.arch.nZ), l1_dw)
            tla.mmad(
                c_1,
                tla.make_tensor_like(l0a, l1_amT, tla.arch.zN),
                tla.make_tensor_like(l0b, l1_dw, tla.arch.nZ),
                init_c=True,
                hf32_mode=tla.params.HF32Mode.HF32_NEAREST_EVEN,
            )
            tla.copy(l1_a64, gm_view(s_dl, sbase64, 64, 64, 64))
            tla.copy(tla.make_tensor_like(l0a, l1_a64, tla.arch.zN), l1_a64)
            tla.copy(tla.make_tensor_like(l0b, l1_ks, tla.arch.nZ), l1_ks)
            tla.mmad(
                c_1,
                tla.make_tensor_like(l0a, l1_a64, tla.arch.zN),
                tla.make_tensor_like(l0b, l1_ks, tla.arch.nZ),
                init_c=False,
                hf32_mode=HF32_OFF,
            )
            tla.copy(gm_view(r_dkbg, sbase, 64, KD, KD), c_1)
            # dR comes from dL transpose times kbg
            tla.copy(l1_nz64, gm_view_t(s_dl, sbase64, 64, 64, 64))
            tla.copy(l1_b, gm_view(w_kb, fo, 64, KD, KD))
            tla.copy(tla.make_tensor_like(l0a, l1_nz64, tla.arch.zN), l1_nz64)
            tla.copy(tla.make_tensor_like(l0b, l1_b, tla.arch.nZ), l1_b)
            tla.mmad(
                c_kt,
                tla.make_tensor_like(l0a, l1_nz64, tla.arch.zN),
                tla.make_tensor_like(l0b, l1_b, tla.arch.nZ),
                init_c=True,
                hf32_mode=tla.params.HF32Mode.HF32_NEAREST_EVEN,
            )
            tla.copy(gm_view(r_dr, sbase, 64, KD, KD), c_kt)
            tla.cross_core_set_flag(ff, tla.pipes.FIX)
        tla.pipe_barrier(tla.pipes.ALL)

    with tla.vector():
        nat_a = tensor(ub + K4_H1, 64, KD, native_dtype(QMODE))
        ub_a = tensor(ub + K4_F1, 64, KD)
        nat_b = tensor(ub + K4_H2, 64, KD, native_dtype(QMODE))
        nat_c = tensor(ub + K4_H3, 64, KD, native_dtype(QMODE))
        ub_b = tensor(ub + K4_F2, 64, KD)
        ou_bf = tensor(ub + K4_OU, 64, KD, native_dtype(QMODE))
        smat = tensor(ub + K4_S, 64, 64)
        for vstep in tla.range(count):
            ti = tla.arch.block_idx() + vstep * nblk
            g = ti // ratio
            j = ti % ratio
            b = g // (hk * nt)
            remg = g % (hk * nt)
            n = remg % nt
            vi = (remg // nt) * ratio + j
            t = (b * hv + vi) * nt + n
            tt = tla.Int64(t)
            sbase = tt * tla.Int64(64) * tla.Int64(KD)
            sbase64 = tt * tla.Int64(64) * tla.Int64(64)
            boff = (
                (tla.Int64(b) * tla.Int64(hv) + tla.Int64(vi)) * tla.Int64(length)
                + tla.Int64(n) * tla.Int64(64)
            ) * tla.Int64(KD)
            valid = length - n * 64
            if valid > 64:
                valid = 64

            # ---- PH1: do' and w*v ----
            zero_tail(nat_a, valid, QMODE)
            tla.copy(sub(nat_a, 0, 0, valid, KD), gm_view(gm_do, boff, valid, KD, KD))
            with tla.vec.func(mode="simd"):
                for r in tla.range(64):
                    for pc in tla.range_constexpr(2):
                        dv = ld_f32(sub(nat_a, r, pc * 64, 1, 64), QMODE)
                        sub(ub_a, r, pc * 64, 1, 64).store(dv * scale)
            tla.copy(gm_view(s_dop, sbase, 64, KD, KD), ub_a)
            zero_tail(nat_b, valid, QMODE)
            tla.copy(sub(nat_b, 0, 0, valid, KD), gm_view(gm_w, boff, valid, KD, KD))
            zero_tail(nat_c, valid, QMODE)
            tla.copy(sub(nat_c, 0, 0, valid, KD), gm_view(gm_v, boff, valid, KD, KD))
            with tla.vec.func(mode="simd"):
                for r in tla.range(64):
                    for pc in tla.range_constexpr(2):
                        wf = ld_f32(sub(nat_b, r, pc * 64, 1, 64), QMODE)
                        vf = ld_f32(sub(nat_c, r, pc * 64, 1, 64), QMODE)
                        sub(ub_b, r, pc * 64, 1, 64).store(wf * vf)
            tla.copy(gm_view(s_wv, sbase, 64, KD, KD), ub_b)
            tla.cross_core_set_flag(fa, tla.pipes.MTE3)

            # ---- PH3: mask dAq ----
            tla.cross_core_wait_flag(fb, tla.pipes.MTE2)
            tla.copy(smat, gm_view(s_daqr, sbase64, 64, 64, 64))
            with tla.vec.func(mode="simd"):
                idx = tla.arange(0, dtype=tla.Int32)
                z = tla.full(0.0, tla.Float32)
                for r in tla.range(64):
                    a = sub(smat, r, 0, 1, 64).load()
                    sub(smat, r, 0, 1, 64).store(tla.where(tla.cmp(idx, r, "gt"), z, a))
            tla.copy(gm_view(s_daq, sbase64, 64, 64, 64), smat)
            tla.cross_core_set_flag(fc, tla.pipes.MTE3)

            # ---- PH5: mask dL, and dv/dw ----
            tla.cross_core_wait_flag(fd, tla.pipes.MTE2)
            tla.copy(smat, gm_view(s_dlr, sbase64, 64, 64, 64))
            with tla.vec.func(mode="simd"):
                idx2 = tla.arange(0, dtype=tla.Int32)
                z2 = tla.full(0.0, tla.Float32)
                for r in tla.range(64):
                    b2 = sub(smat, r, 0, 1, 64).load()
                    sub(smat, r, 0, 1, 64).store(
                        tla.where(tla.cmp(idx2, r, "lt"), z2 - b2, z2)
                    )
            tla.copy(gm_view(s_dl, sbase64, 64, 64, 64), smat)
            tla.copy(ub_b, gm_view(r_duv, sbase, 64, KD, KD))
            with tla.vec.func(mode="simd"):
                for r in tla.range(64):
                    for pc in tla.range_constexpr(2):
                        wf = ld_f32(sub(nat_b, r, pc * 64, 1, 64), QMODE)
                        du = sub(ub_b, r, pc * 64, 1, 64).load()
                        st_low(sub(ou_bf, r, pc * 64, 1, 64), wf * du, QMODE)
            tla.copy(gm_view(gm_dv, boff, valid, KD, KD), sub(ou_bf, 0, 0, valid, KD))
            with tla.vec.func(mode="simd"):
                for r in tla.range(64):
                    for pc in tla.range_constexpr(2):
                        vf = ld_f32(sub(nat_c, r, pc * 64, 1, 64), QMODE)
                        du = sub(ub_b, r, pc * 64, 1, 64).load()
                        st_low(sub(ou_bf, r, pc * 64, 1, 64), vf * du, QMODE)
            tla.copy(gm_view(gm_dw, boff, valid, KD, KD), sub(ou_bf, 0, 0, valid, KD))
            tla.cross_core_set_flag(fe, tla.pipes.MTE3)
        tla.pipe_barrier(tla.pipes.ALL)


# --------------------------------------------------------------------------- #
# K5: GVA reduce + reverse cumsum of dgc + dtype cast
# --------------------------------------------------------------------------- #
K5_T0 = 0  # 4096  acc dq
K5_T1 = 4096  # 4096  acc dk
K5_T2 = 8192  # 4096  acc db
K5_T3 = 12288  # 4096  acc dgc / dg
K5_T4 = 16384  # 4096  r_* staging; row-63 strip aliases its first 2048
K5_FEG = 20480  # 4096  exp(gc)
K5_FQS = 24576  # 4096  q*exp(-gc)
K5_FENG = 28672  # 4096  exp(-gc)
K5_FKS = 32768  # 4096  k*exp(-gc)
K5_FKT = 36864  # 4096  k*exp(gl-gc)
K5_FBE = 40960  # 4096  b*exp(gc)
K5_FKE = 45056  # 4096  k*exp(gc)
K5_FKB = 49152  # 4096  b*k*exp(gc)
K5_EGL = 53248  # 64    exp(gl)
K5_T10 = 53312  # 64    row 63 correction
K5_T9 = 53376  # 4096  db/dg output cast staging
K5_T12 = 57472  # 4096  dq/dk cast staging
K5_S63 = 61568  # 64    row-63 reduced scalars (one per K channel)
K5_TOTAL = 61632


@tla.jit
def k5_finish(t3, t10, t2, t9, gdb, gdg, off, valid, mode):
    """Apply the row-63 correction, reverse-cumsum dgc, cast and store db/dg."""
    with tla.vec.func(mode="simd"):
        r63 = sub(t3, 63, 0, 1, 64).load() + sub(t10, 0, 0, 1, 64).load()
        sub(t3, 63, 0, 1, 64).store(r63)
        tot = tla.full(0.0, tla.Float32)
        for ra in tla.range(64):
            tot = tot + sub(t3, ra, 0, 1, 64).load()
        pre = tla.full(0.0, tla.Float32)
        for rb in tla.range(64):
            val = sub(t3, rb, 0, 1, 64).load()
            st_low(sub(t9, rb, 0, 1, 64), tot - pre, mode)
            pre = pre + val
    tla.copy(gm_view(gdg, off, valid, 64, KD), sub(t9, 0, 0, valid, 64))
    with tla.vec.func(mode="simd"):
        for row_L1346_L1346 in tla.range(64):
            st_low(
                sub(t9, row_L1346_L1346, 0, 1, 64),
                sub(t2, row_L1346_L1346, 0, 1, 64).load(),
                mode,
            )
    tla.copy(gm_view(gdb, off, valid, 64, KD), sub(t9, 0, 0, valid, 64))


@tla.kernel(auto_sync="v0")
def reduce_kernel(
    gm_dq: tla.Tensor,
    gm_dk: tla.Tensor,
    gm_db: tla.Tensor,
    gm_dg: tla.Tensor,
    w_eg: tla.Tensor,
    w_eng: tla.Tensor,
    w_qs: tla.Tensor,
    w_ks: tla.Tensor,
    w_kb: tla.Tensor,
    w_ke: tla.Tensor,
    w_be: tla.Tensor,
    w_kt: tla.Tensor,
    w_egl: tla.Tensor,
    hbuf: tla.Tensor,
    dhbuf: tla.Tensor,
    r_dq: tla.Tensor,
    r_dks: tla.Tensor,
    r_dkt: tla.Tensor,
    r_dkbg: tla.Tensor,
    r_dr: tla.Tensor,
    length: int,
    hk: int,
    hg: int,
    hv: int,
    hf: int,
    nratio: int,
    ntask: int,
    REDGB: tla.Constexpr[int],
    DQMODE: tla.Constexpr[int],
    DBMODE: tla.Constexpr[int],
) -> None:
    ub = tla.allocate(K5_TOTAL, tla.Float32, tla.AddressSpace.ub, 256)
    t0 = tensor(ub + K5_T0, 64, 64)
    t1 = tensor(ub + K5_T1, 64, 64)
    t2 = tensor(ub + K5_T2, 64, 64)
    t3 = tensor(ub + K5_T3, 64, 64)
    t4 = tensor(ub + K5_T4, 64, 64)
    feg = tensor(ub + K5_FEG, 64, 64)
    fqs = tensor(ub + K5_FQS, 64, 64)
    feng = tensor(ub + K5_FENG, 64, 64)
    fks = tensor(ub + K5_FKS, 64, 64)
    fkt = tensor(ub + K5_FKT, 64, 64)
    fbe = tensor(ub + K5_FBE, 64, 64)
    fke = tensor(ub + K5_FKE, 64, 64)
    fkb = tensor(ub + K5_FKB, 64, 64)
    t9 = tensor(ub + K5_T9, 64, 64, native_dtype(DBMODE))
    t10 = tensor(ub + K5_T10, 1, 64)
    vgl = tensor(ub + K5_EGL, 1, 64)
    t12 = tensor(ub + K5_T12, 64, 64, native_dtype(DQMODE))
    # fp32 aliases for the row-63 strip pass, which runs after the fifth r_*
    # term and before k5_finish rewrites t9: 32 K rows x all 128 V columns of
    # H (t4's region) and of dH (t9's region).
    th63 = tensor(ub + K5_T4, 32, KD)
    tdh63 = tensor(ub + K5_T9, 32, KD)
    # 1-D scratch for the 64 reduced row scalars, one lane per K channel.
    s63v = tla.make_tensor(
        tla.recast_ptr(ub + K5_S63, dtype=tla.Float32),
        tla.make_layout(tla.make_shape(64), tla.make_stride(1)),
    )

    nblk = tla.arch.block_num()
    count = (ntask - tla.arch.block_idx() + nblk - 1) // nblk
    nt = (length + 63) // 64

    with tla.vector():
        for vstep_L1403_L1403 in tla.range(count):
            u = tla.arch.block_idx() + vstep_L1403_L1403 * nblk
            b = u // (hk * nt)
            rem = u % (hk * nt)
            khi = rem // nt
            n = rem % nt
            valid = length - n * 64
            if valid > 64:
                valid = 64
            # R6 term-major layout: one pass over the factor tiles per half.
            # With REDGB == 1 (HF == HK) every value head of this k head reads
            # the same factors, so they are loaded once and reused by all
            # ``nratio`` heads; only the per-hv r_* tiles and H/dH are re-read.
            for half_L1412 in tla.range_constexpr(2):
                c0 = half_L1412 * 64
                with tla.vec.func(mode="simd"):
                    z = tla.full(0.0, tla.Float32)
                    for row_L1416 in tla.range(64):
                        sub(t0, row_L1416, 0, 1, 64).store(z)
                        sub(t1, row_L1416, 0, 1, 64).store(z)
                if tla.const_expr(REDGB == 1):
                    with tla.vec.func(mode="simd"):
                        z2 = tla.full(0.0, tla.Float32)
                        for row_L1422 in tla.range(64):
                            sub(t2, row_L1422, 0, 1, 64).store(z2)
                            sub(t3, row_L1422, 0, 1, 64).store(z2)
                        sub(t10, 0, 0, 1, 64).store(z2)
                    fc_L1400 = (tla.Int64(b * hf) + tla.Int64(khi)) * tla.Int64(
                        nt
                    ) + tla.Int64(n)
                    fb_L1400 = fc_L1400 * tla.Int64(64) * tla.Int64(KD) + tla.Int64(c0)
                    tla.copy(feg, gm_view(w_eg, fb_L1400, 64, 64, KD))
                    tla.copy(fqs, gm_view(w_qs, fb_L1400, 64, 64, KD))
                    tla.copy(feng, gm_view(w_eng, fb_L1400, 64, 64, KD))
                    tla.copy(fks, gm_view(w_ks, fb_L1400, 64, 64, KD))
                    tla.copy(fkt, gm_view(w_kt, fb_L1400, 64, 64, KD))
                    tla.copy(fbe, gm_view(w_be, fb_L1400, 64, 64, KD))
                    tla.copy(fke, gm_view(w_ke, fb_L1400, 64, 64, KD))
                    tla.copy(fkb, gm_view(w_kb, fb_L1400, 64, 64, KD))
                    tla.copy(
                        vgl,
                        gm_view(
                            w_egl, fc_L1400 * tla.Int64(KD) + tla.Int64(c0), 1, 64, KD
                        ),
                    )

                for j_L1427 in tla.range(nratio):
                    hvi = khi * nratio + j_L1427
                    fh = hvi
                    if tla.const_expr(REDGB == 1):
                        fh = khi
                    fchunk = (tla.Int64(b * hf) + tla.Int64(fh)) * tla.Int64(
                        nt
                    ) + tla.Int64(n)
                    fbase = fchunk * tla.Int64(64) * tla.Int64(KD) + tla.Int64(c0)
                    hvchunk = (tla.Int64(b * hv) + tla.Int64(hvi)) * tla.Int64(
                        nt
                    ) + tla.Int64(n)
                    hbase = hvchunk * tla.Int64(64) * tla.Int64(KD) + tla.Int64(c0)
                    # H (chunk-start state, slot n) and dH (this chunk's
                    # dh_out, slot n+1) at row 0 / column 0: this half's K
                    # channels c0..c0+63 are added as a row offset below.
                    hrow63 = (
                        (
                            (tla.Int64(b * hv) + tla.Int64(hvi))
                            * (tla.Int64(nt) + tla.Int64(1))
                            + tla.Int64(n)
                        )
                        * tla.Int64(KD)
                        * tla.Int64(KD)
                    )
                    dhrow63 = hrow63 + tla.Int64(KD) * tla.Int64(KD)
                    if tla.const_expr(REDGB == 0):
                        with tla.vec.func(mode="simd"):
                            z3 = tla.full(0.0, tla.Float32)
                            for row_L1445 in tla.range(64):
                                sub(t2, row_L1445, 0, 1, 64).store(z3)
                                sub(t3, row_L1445, 0, 1, 64).store(z3)
                            sub(t10, 0, 0, 1, 64).store(z3)
                        # per-value-head factors: same slots, loaded per head
                        tla.copy(feg, gm_view(w_eg, fbase, 64, 64, KD))
                        tla.copy(fqs, gm_view(w_qs, fbase, 64, 64, KD))
                        tla.copy(feng, gm_view(w_eng, fbase, 64, 64, KD))
                        tla.copy(fks, gm_view(w_ks, fbase, 64, 64, KD))
                        tla.copy(fkt, gm_view(w_kt, fbase, 64, 64, KD))
                        tla.copy(fbe, gm_view(w_be, fbase, 64, 64, KD))
                        tla.copy(fke, gm_view(w_ke, fbase, 64, 64, KD))
                        tla.copy(fkb, gm_view(w_kb, fbase, 64, 64, KD))
                        tla.copy(
                            vgl,
                            gm_view(
                                w_egl, fchunk * tla.Int64(KD) + tla.Int64(c0), 1, 64, KD
                            ),
                        )

                    # accumulate the dq and dgc contributions (q is qs times eng)
                    tla.copy(t4, gm_view(r_dq, hbase, 64, 64, KD))
                    with tla.vec.func(mode="simd"):
                        for rowq in tla.range(64):
                            aq = sub(t4, rowq, 0, 1, 64).load()
                            sub(t0, rowq, 0, 1, 64).store(
                                sub(t0, rowq, 0, 1, 64).load()
                                + aq * sub(feg, rowq, 0, 1, 64).load()
                            )
                            sub(t3, rowq, 0, 1, 64).store(
                                sub(t3, rowq, 0, 1, 64).load()
                                + sub(fqs, rowq, 0, 1, 64).load() * aq
                            )
                    # dk_score = eng * dks ; dk += ; dgc -= ks * dk_score
                    tla.copy(t4, gm_view(r_dks, hbase, 64, 64, KD))
                    with tla.vec.func(mode="simd"):
                        for rowk in tla.range(64):
                            ak = sub(t4, rowk, 0, 1, 64).load()
                            sub(t1, rowk, 0, 1, 64).store(
                                sub(t1, rowk, 0, 1, 64).load()
                                + ak * sub(feng, rowk, 0, 1, 64).load()
                            )
                            sub(t3, rowk, 0, 1, 64).store(
                                sub(t3, rowk, 0, 1, 64).load()
                                - sub(fks, rowk, 0, 1, 64).load() * ak
                            )
                    # dk_tail contributes through exp(gl-gc), i.e. eng times egl,
                    # and dgc loses the dk_tail times k_tail term
                    tla.copy(t4, gm_view(r_dkt, hbase, 64, 64, KD))
                    with tla.vec.func(mode="simd"):
                        rsum = tla.full(0.0, tla.Float32)
                        gl_L1500 = sub(vgl, 0, 0, 1, 64).load()
                        for rowt in tla.range(64):
                            dt = sub(t4, rowt, 0, 1, 64).load()
                            at = dt * sub(fkt, rowt, 0, 1, 64).load()
                            rsum = rsum + at
                            sub(t1, rowt, 0, 1, 64).store(
                                sub(t1, rowt, 0, 1, 64).load()
                                + dt * (sub(feng, rowt, 0, 1, 64).load() * gl_L1500)
                            )
                            sub(t3, rowt, 0, 1, 64).store(
                                sub(t3, rowt, 0, 1, 64).load() - at
                            )
                        sub(t10, 0, 0, 1, 64).store(sub(t10, 0, 0, 1, 64).load() + rsum)
                    # dkbg terms: dk += be*., db += ke*., dgc += kb*.
                    tla.copy(t4, gm_view(r_dkbg, hbase, 64, 64, KD))
                    with tla.vec.func(mode="simd"):
                        for row_L1491 in tla.range(64):
                            ab = sub(t4, row_L1491, 0, 1, 64).load()
                            sub(t1, row_L1491, 0, 1, 64).store(
                                sub(t1, row_L1491, 0, 1, 64).load()
                                + ab * sub(fbe, row_L1491, 0, 1, 64).load()
                            )
                            sub(t2, row_L1491, 0, 1, 64).store(
                                sub(t2, row_L1491, 0, 1, 64).load()
                                + ab * sub(fke, row_L1491, 0, 1, 64).load()
                            )
                            sub(t3, row_L1491, 0, 1, 64).store(
                                sub(t3, row_L1491, 0, 1, 64).load()
                                + ab * sub(fkb, row_L1491, 0, 1, 64).load()
                            )
                    # dR terms: dk += eng*r_dr ; dgc -= ks*r_dr
                    tla.copy(t4, gm_view(r_dr, hbase, 64, 64, KD))
                    with tla.vec.func(mode="simd"):
                        for row_L1515 in tla.range(64):
                            ad = sub(t4, row_L1515, 0, 1, 64).load()
                            sub(t1, row_L1515, 0, 1, 64).store(
                                sub(t1, row_L1515, 0, 1, 64).load()
                                + ad * sub(feng, row_L1515, 0, 1, 64).load()
                            )
                            sub(t3, row_L1515, 0, 1, 64).store(
                                sub(t3, row_L1515, 0, 1, 64).load()
                                - sub(fks, row_L1515, 0, 1, 64).load() * ad
                            )
                    # row 63: dgc[L-1, k] += exp(gl[k]) * sum_v H[k, v] * dH[k, v]
                    # H is the chunk-start state (slot n), dH is *this chunk's*
                    # dh_out (slot n+1) and the reduction axis is V -- the
                    # contiguous axis of the row-major state.  Each of the 64 K
                    # channels of this half is therefore one 128-wide row dot:
                    # two 32-row strips with all V columns, two 64-lane products
                    # summed in registers, one ADD reduce, and the scalar stored
                    # into lane k of s63v with the unaligned 1-lane store used by
                    # examples/end_to_end/flash_attention_infer.
                    for sb_L1527 in tla.range(2):
                        tla.copy(
                            th63,
                            gm_view(
                                hbuf,
                                hrow63
                                + (tla.Int64(c0) + tla.Int64(sb_L1527 * 32))
                                * tla.Int64(KD),
                                32,
                                KD,
                                KD,
                            ),
                        )
                        tla.copy(
                            tdh63,
                            gm_view(
                                dhbuf,
                                dhrow63
                                + (tla.Int64(c0) + tla.Int64(sb_L1527 * 32))
                                * tla.Int64(KD),
                                32,
                                KD,
                                KD,
                            ),
                        )
                        with tla.vec.func(mode="simd"):
                            all63 = tla.create_mask(
                                pattern=tla.mask.ALL, dtype=tla.Float32
                            )
                            one63 = tla.update_mask(1, dtype=tla.Float32)[0]
                            for rr_L1536 in tla.range(32):
                                acc2 = (
                                    sub(th63, rr_L1536, 0, 1, 64).load()
                                    * sub(tdh63, rr_L1536, 0, 1, 64).load()
                                )
                                acc2 = (
                                    acc2
                                    + sub(th63, rr_L1536, 64, 1, 64).load()
                                    * sub(tdh63, rr_L1536, 64, 1, 64).load()
                                )
                                sc63 = acc2.reduce(tla.ReductionOp.ADD, mask=all63)
                                tla.tile_view(
                                    s63v,
                                    tla.make_shape(1),
                                    tla.make_coord(sb_L1527 * 32 + rr_L1536),
                                ).store(
                                    sc63,
                                    params=tla.params.UnalignStoreParams(),
                                    mask=one63,
                                )
                    with tla.vec.func(mode="simd"):
                        # order the unaligned lane stores above against this load
                        tla.local_mem_bar(
                            tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                        )
                        sub(t10, 0, 0, 1, 64).store(
                            sub(t10, 0, 0, 1, 64).load()
                            + sub(vgl, 0, 0, 1, 64).load()
                            * tla.tile_view(
                                s63v, tla.make_shape(64), tla.make_coord(0)
                            ).load()
                        )
                    if tla.const_expr(REDGB == 0):
                        goff = (
                            (tla.Int64(b) * tla.Int64(hg) + tla.Int64(hvi))
                            * tla.Int64(length)
                            + tla.Int64(n) * tla.Int64(64)
                        ) * tla.Int64(KD) + tla.Int64(c0)
                        k5_finish(t3, t10, t2, t9, gm_db, gm_dg, goff, valid, DBMODE)
                # dq / dk outputs
                qoff = (
                    (tla.Int64(b) * tla.Int64(hk) + tla.Int64(khi)) * tla.Int64(length)
                    + tla.Int64(n) * tla.Int64(64)
                ) * tla.Int64(KD) + tla.Int64(c0)
                with tla.vec.func(mode="simd"):
                    for srowq in tla.range(64):
                        st_low(
                            sub(t12, srowq, 0, 1, 64),
                            sub(t0, srowq, 0, 1, 64).load(),
                            DQMODE,
                        )
                tla.copy(gm_view(gm_dq, qoff, valid, 64, KD), sub(t12, 0, 0, valid, 64))
                with tla.vec.func(mode="simd"):
                    for srowk in tla.range(64):
                        st_low(
                            sub(t12, srowk, 0, 1, 64),
                            sub(t1, srowk, 0, 1, 64).load(),
                            DQMODE,
                        )
                tla.copy(gm_view(gm_dk, qoff, valid, 64, KD), sub(t12, 0, 0, valid, 64))
                if tla.const_expr(REDGB == 1):
                    goff2 = (
                        (tla.Int64(b) * tla.Int64(hg) + tla.Int64(khi))
                        * tla.Int64(length)
                        + tla.Int64(n) * tla.Int64(64)
                    ) * tla.Int64(KD) + tla.Int64(c0)
                    k5_finish(t3, t10, t2, t9, gm_db, gm_dg, goff2, valid, DBMODE)
        tla.pipe_barrier(tla.pipes.ALL)


# --------------------------------------------------------------------------- #
# Device ABI: dtype codes and workspace slot names
# --------------------------------------------------------------------------- #
# ``mode`` is the per-tensor dtype code the kernels branch on (0=fp32,
# 1=bf16, 2=fp16); it is passed as a ``tla.Constexpr`` argument to every kernel
# and selects the load/store/cast path in ``ld_f32``/``st_low``.  K5's two
# epilogues take the same codes.
_DT_MODE = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}

# GM workspace slots consumed by the kernels.  ``_W_NAMES`` are 128-wide
# ``[F*64, 128]`` fp32 tiles, ``_W64_NAMES`` are 64-wide ``[F*64, 64]`` fp32
# tiles; both are indexed per factor ``f = (b*HF + hf)*NT + n``.  The name order
# is the binding order used by ``_Plan`` below and must match the
# kernel signatures.  R37: ``gc`` was dropped -- K1 computed it in UB for the
# exp() factors but no kernel ever read the GM copy back.
_W_NAMES = (
    "eg",
    "eng",
    "qs",
    "ks",
    "kb",
    "ke",
    "be",
    "kt",
    "wy",
    "xq",
    "xk",
    "ys",
)
_W64_NAMES = ("aq", "am", "nm")

# --------------------------------------------------------------------------- #
# Host interface: validation, workspace / LRU caches, launches, public entry  #
# --------------------------------------------------------------------------- #
_ARTIFACT_CACHE = OrderedDict()
_ARTIFACT_CACHE_LIMIT = 64
_ARTIFACT_CACHE_LOCK = Lock()


def _mode(t):
    try:
        return _DT_MODE[t.dtype]
    except KeyError:  # pragma: no cover
        raise TypeError(f"unsupported dtype {t.dtype}")


def _bind(t):
    return tla.from_dlpack(
        t.detach().reshape(-1)[:KD].reshape(1, KD), layout_tag=tla.arch.RowMajor
    )


def _artifact(kernel, key, bindings, scalars, consts=()):
    with _ARTIFACT_CACHE_LOCK:
        art = _ARTIFACT_CACHE.get(key)
        if art is None:
            art = tla.compile(
                kernel, *bindings, *scalars, *consts, options="--npu-arch 3510"
            )
            _ARTIFACT_CACHE[key] = art
            if len(_ARTIFACT_CACHE) > _ARTIFACT_CACHE_LIMIT:
                _ARTIFACT_CACHE.popitem(last=False)
        else:
            _ARTIFACT_CACHE.move_to_end(key)
    return art


def validate(q, k, v, g, b, w_gate, do, scale, initial_state):
    tens = (q, k, v, g, b, w_gate, do)
    if any(not isinstance(x, torch.Tensor) for x in tens):
        raise TypeError("q, k, v, g, b, w_gate, do must be tensors")
    if any(x.ndim != 4 for x in tens):
        raise ValueError("all inputs must be rank 4 in BNSD layout")
    B, HK, T, K = q.shape
    HV = v.shape[1]
    if K != KD or T < 1 or B < 1 or HK < 1 or not 1 <= HV <= 96:
        raise ValueError("require B,T,HK >= 1, K=128, HV<=96")
    if HV % HK or HV // HK not in (1, 2, 3, 4):
        raise ValueError("HV / HK must be 1, 2, 3 or 4")
    if k.shape != q.shape or v.shape != (B, HV, T, KD):
        raise ValueError("invalid q/k/v shape")
    if do.shape != v.shape or w_gate.shape != v.shape:
        raise ValueError("do and w_gate must match v")
    for name, x in (("g", g), ("b", b)):
        if x.shape[1] not in (HK, HV) or x.shape != (B, x.shape[1], T, KD):
            raise ValueError(f"{name} must be [B, HK|HV, T, 128]")
    if g.shape[1] != b.shape[1]:
        # K5 uses one head count for both db and dg, and the reference only
        # defines a shared HG; a mixed pair would store past the shorter
        # tensor.  Reject before any allocation or launch.
        raise ValueError("g and b must share one head count (HG == HB)")
    if any(x.dtype != q.dtype for x in (k,)):
        raise TypeError("q and k must share a dtype")
    if any(x.dtype != v.dtype for x in (w_gate, do)):
        raise TypeError("v, w_gate and do must share a dtype")
    for x in tens:
        if not x.is_contiguous():
            raise ValueError("all tensors must be contiguous")
        if x.device.type != "npu" or x.device != q.device:
            raise ValueError("all tensors must be on the same npu device")
    if g.dtype != b.dtype:
        raise TypeError("g and b must share a dtype")
    if initial_state is not None:
        if initial_state.dtype != torch.float32:
            raise TypeError("initial_state must be fp32")
        if initial_state.shape != (B, HV, KD, KD):
            raise ValueError("initial_state must be [B, HV, 128, 128]")
    if scale is None:
        scale = KD**-0.5
    if (
        isinstance(scale, bool)
        or not isinstance(scale, numbers.Real)
        or not math.isfinite(scale)
    ):
        raise ValueError("scale must be a finite real scalar or None")
    return float(scale)


class Ctx:
    """Allocated workspace plus launch descriptors for one call."""

    __slots__ = (
        "B",
        "HK",
        "HV",
        "HG",
        "HB",
        "HF",
        "T",
        "NT",
        "F",
        "ratio",
        "q",
        "k",
        "v",
        "g",
        "b",
        "w",
        "do",
        "scale",
        "initial_state",
        "ws",
        "outs",
        "device",
        "cores",
        "vcores",
        "streams",
        "plan",
        "entry",
    )


# Per-shape workspace cache.  ``_bind`` (``tla.from_dlpack``) costs ~24 us per
# tensor and one ``G.run`` binds 93 of them (~2.3 ms host time); the first
# launch's share lands inside the benchmark's device window because the device
# is idle when the call starts.  Every binding except the 14 input / output
# tensors belongs to a workspace buffer whose address is fixed per shape, so
# the workspace and its bindings are built once per (device, stream, shape) and
# replayed; only ``h`` slot 0 (``initial_state``) and ``dh`` slot ``NT`` are
# re-initialised per call.  This is *runtime* state: ``_ARTIFACT_CACHE`` still
# keys on kernel + dtypes only and never stores a device pointer.
#
# Capacity is bounded by **total device bytes**, not by an entry count.  The
# 50-case workload is two thirds small shapes (17-72 MiB of workspace each) and
# one third large ones (001 is 4.44 GiB, 021 is 6.46 GiB), so a fixed count
# either thrashes the small shapes when they alternate (every call would then
# re-allocate and re-bind, ~2.9 ms of host time) or lets several large
# workspaces pin ~10 GiB.  The 12 GiB default holds 001 and 021 together; a
# call that would leave less than 2 GiB of device memory free is not cached at
# all, and ``GDN2_WS_CACHE_MB`` overrides the cap (0 disables the cache).
_PLAN_CACHE = OrderedDict()
_PLAN_CACHE_BYTES = int(float(os.environ.get("GDN2_WS_CACHE_MB", "12288")) * 2**20)
_PLAN_CACHE_LOCK = Lock()
# Host-only counters, read by the probes in ``tools/``; no kernel touches them.
_WS_STATS = {"hit": 0, "miss": 0, "evict": 0, "skip": 0, "bytes": 0}
_STREAM_CACHE = {}
_CORES_CACHE = {}
_VCORES_CACHE = {}


# Workspace slot names per kernel, in binding order.  They live at module
# level so every ``_Plan`` comprehension stays on a single line.
_WS_PREP = (
    "eg",
    "eng",
    "qs",
    "ks",
    "kb",
    "ke",
    "be",
    "kt",
    "wy",
    "aq",
    "am",
    "nm",
    "egl",
    "xq",
    "xk",
    "ys",
    "ident",
)

_WS_FWD = (
    "am",
    "wy",
    "kt",
    "egl",
    "h",
    "vn",
    "wv",
    "u",
    "wh",
    "hp",
)

_WS_BWD = (
    "aq",
    "qs",
    "wy",
    "kt",
    "egl",
    "dh",
    "dop",
    "dvn",
    "dvnk",
    "dha",
    "dhb",
)

_WS_GRAD = (
    "aq",
    "am",
    "qs",
    "ks",
    "kb",
    "kt",
    "h",
    "vn",
    "dh",
    "dop",
    "wv2",
    "daqr",
    "daq",
    "dlr",
    "dl",
    "r_dq",
    "r_dks",
    "r_dkt",
    "r_duv",
    "r_dkbg",
    "r_dr",
    "nid",
)

_WS_REDUCE = (
    "eg",
    "eng",
    "qs",
    "ks",
    "kb",
    "ke",
    "be",
    "kt",
    "egl",
    "h",
    "dh",
    "r_dq",
    "r_dks",
    "r_dkt",
    "r_dkbg",
    "r_dr",
)


class _Plan:
    """Binding lists for one workspace: everything that is shape-stable."""

    __slots__ = ("ws", "prep", "fwd", "bwd", "grad", "reduce")

    def __init__(self, ws):
        b = _bind
        self.ws = ws
        self.prep = [b(ws[n]) for n in _WS_PREP]
        self.fwd = [b(ws[n]) for n in _WS_FWD]
        self.bwd = [b(ws[n]) for n in _WS_BWD]
        self.grad = [b(ws[n]) for n in _WS_GRAD]
        self.reduce = [b(ws[n]) for n in _WS_REDUCE]


class _Entry:
    """Cached workspace plus the host lease that guards its reuse.

    ``owner`` is the thread currently driving a call against this workspace
    (``None`` when free).  It is held only for the duration of one call, so a
    second thread that reaches the same bucket while the first is still
    enqueuing gets a private workspace instead of interleaving its ``h``/``dh``
    writes with the first call's launches.  On the single-threaded path the
    lease is two uncontended lock round-trips and never blocks.
    """

    __slots__ = ("plan", "ws", "nbytes", "lock", "owner")

    def __init__(self, plan, ws, nbytes):
        self.plan = plan
        self.ws = ws
        self.nbytes = nbytes
        self.lock = Lock()
        self.owner = None


def _lease(ent):
    """Claim ``ent`` for this call; ``False`` means another thread owns it.

    A lease held by *this* thread is stale by construction (a thread cannot be
    inside two calls at once), so it is taken over: an exception that skipped
    ``_unlease`` can never wedge the bucket.  A lease whose owner thread has
    exited is taken over as well, for the same reason.
    """
    tid = threading.get_ident()
    with ent.lock:
        if ent.owner is None or ent.owner == tid:
            ent.owner = tid
            return True
        if ent.owner not in {t.ident for t in threading.enumerate()}:
            ent.owner = tid
            return True
    return False


def _unlease(ent):
    with ent.lock:
        ent.owner = None


def _ws_nbytes(ws):
    """Device bytes held by one workspace: the unit of the LRU budget."""
    return sum(v.numel() * v.element_size() for v in ws.values() if torch.is_tensor(v))


# Prefer the non-blocking raw getter: ``_npu_getCurrentRawStream`` was measured
# to stall for up to 2 ms when the stream has work in flight, which lands right
# inside the next call's device window (BSND was ~10% slower before this).
_RAW_STREAM = getattr(torch_npu._C, "_npu_getCurrentRawStreamNoWait", None)


def _stream_key(dev):
    """Cheap identity of the stream this call will be enqueued on.

    ``torch.npu.current_stream()`` builds a Python ``Stream`` object and costs
    ~5.5 us, which is ~1% of a T=512 call; the raw handle is the same identity
    for ~0.13 us and never blocks.  The public call stays as the fallback when
    the private getter is unavailable or changes shape.
    """
    global _RAW_STREAM
    if _RAW_STREAM is None:
        return torch.npu.current_stream()
    idx = dev.index
    if idx is None:
        idx = torch.npu.current_device()
    try:
        return _RAW_STREAM(idx)
    except Exception:  # pragma: no cover - private API changed shape
        _RAW_STREAM = None
        return torch.npu.current_stream()


# Keep at least this much device memory available after admitting a workspace,
# so caching never squeezes another job sharing the card.
_WS_RESERVE_BYTES = 2 << 30
# ``torch.npu.mem_get_info`` costs ~0.5 ms on this part (it queries the device),
# so it must stay off the per-call path: it runs once per *insert* and a "too
# low" verdict is remembered for this long instead of being re-queried on every
# miss while memory stays tight.
_WS_MEM_COOLDOWN = 0.25
_WS_MEM = {"low": False, "t": -1e9}


def _ws_free_bytes(dev):
    """Bytes a new workspace could use without squeezing other jobs.

    ``mem_get_info`` counts only what the device still has free; blocks this
    process reserved and then released are reusable by torch too, so add the
    allocator pool back.  Evicting a cached workspace moves its bytes into that
    pool, which is what makes the LRU retry below work.
    """
    free, _total = torch.npu.mem_get_info(dev)
    return free + torch.npu.memory_reserved(dev) - torch.npu.memory_allocated(dev)


def _admit(ent, dev):
    """Cache ``ent`` only if ``_WS_RESERVE_BYTES`` stays available afterwards.

    Called *after* the workspace is allocated, so this is the same predicate as
    ``free_before < entry_bytes + reserve -> do not cache``: whatever the
    allocation consumed is already reflected in the free number.  On shortage
    the idle LRU entries are evicted first (their bytes go back to the pool)
    and the check is retried; if it still fails the entry stays transient.
    """
    if _PLAN_CACHE_BYTES <= 0:
        return False
    now = time.monotonic()
    if _WS_MEM["low"] and now - _WS_MEM["t"] < _WS_MEM_COOLDOWN:
        return False  # remembered shortage: do not query the device again
    avail = _ws_free_bytes(dev)
    if avail >= _WS_RESERVE_BYTES:
        _WS_MEM["low"], _WS_MEM["t"] = False, now
        return True
    for key in list(_PLAN_CACHE):
        old = _PLAN_CACHE[key]
        with old.lock:
            if old.owner is not None:
                continue
        _PLAN_CACHE.pop(key)
        _WS_STATS["evict"] += 1
        _WS_STATS["bytes"] -= old.nbytes
        avail += old.nbytes
        if avail >= _WS_RESERVE_BYTES:
            _WS_MEM["low"], _WS_MEM["t"] = False, now
            return True
    _WS_MEM["low"], _WS_MEM["t"] = True, now
    return False


def _trim_cache():
    """Drop least-recently-used idle workspaces until the byte budget is met.

    A workspace leased by an in-flight call is skipped: its buffers may still
    be read by kernels the caller has already enqueued, and freeing them would
    let the allocator hand the memory to a later op.  If every candidate is
    busy the cache stays over budget until the next insert.
    """
    total = sum(e.nbytes for e in _PLAN_CACHE.values())
    while total > _PLAN_CACHE_BYTES and len(_PLAN_CACHE) > 1:
        victim = None
        for key, ent in _PLAN_CACHE.items():
            with ent.lock:
                busy = ent.owner is not None
            if not busy:
                victim = key
                break
        if victim is None:
            break
        old = _PLAN_CACHE.pop(victim)
        total -= old.nbytes
        _WS_STATS["evict"] += 1
    _WS_STATS["bytes"] = total


def _workspace(ctx, dev, stream=None):
    """(binding plan, workspace) for this shape, allocated on first use.

    The key covers every dimension ``_alloc_ws`` reads -- ``B``, ``HV``,
    ``HF`` and ``NT`` (``T`` never sizes a buffer, ``ceil(T/BT)`` does) -- plus
    the device and the stream the call runs on.  The stream is part of the key
    so that an overlapping call on another stream gets its own buffers: a
    cached workspace may still be read by the previous call's kernels, and
    sharing it across streams would need a cross-stream wait on every call.
    """
    key = (str(dev), stream, ctx.B, ctx.HV, ctx.HF, ctx.NT)
    with _PLAN_CACHE_LOCK:
        ent = _PLAN_CACHE.get(key)
        if ent is not None:
            _PLAN_CACHE.move_to_end(key)
            _WS_STATS["hit"] += 1
            return ent
    ws = _alloc_ws(ctx, dev)
    ent = _Entry(_Plan(ws), ws, _ws_nbytes(ws))
    with _PLAN_CACHE_LOCK:
        cur = _PLAN_CACHE.get(key)
        if cur is not None:  # another thread built this bucket meanwhile
            _PLAN_CACHE.move_to_end(key)
            _WS_STATS["hit"] += 1
            return cur
        _WS_STATS["miss"] += 1
        if _admit(ent, dev):
            _PLAN_CACHE[key] = ent
            _trim_cache()
        else:
            # Low on device memory (or caching disabled): this call keeps its
            # private workspace and drops it on return.
            _WS_STATS["skip"] += 1
    return ent


def _ws_cache_stats():
    """Snapshot of the workspace cache for the host probes."""
    with _PLAN_CACHE_LOCK:
        return dict(
            _WS_STATS,
            bytes=sum(e.nbytes for e in _PLAN_CACHE.values()),
            entries=len(_PLAN_CACHE),
            budget=_PLAN_CACHE_BYTES,
            limit_mb=_PLAN_CACHE_BYTES / 2**20,
        )


def _alloc_ws(ctx, dev):
    """Allocate every device buffer whose extent only depends on the shape."""
    ws = {}
    for name in _W_NAMES:
        ws[name] = torch.empty((ctx.F * BT, KD), dtype=torch.float32, device=dev)
    for name in _W64_NAMES:
        ws[name] = torch.empty((ctx.F * BT, 64), dtype=torch.float32, device=dev)
    ws["egl"] = torch.empty((ctx.F, KD), dtype=torch.float32, device=dev)
    # The five GVA partials (``r_dq/r_dks/r_dkt/r_dkbg/r_dr``) are per *value*
    # head, so K5 can sum them in a fixed ascending order; they share the
    # ``(b, hv, chunk)`` slot layout with ``dop`` (same bytes as before).
    nhvc = ctx.B * ctx.HV * ctx.NT
    for name in ("dop", "wv2", "r_dq", "r_dks", "r_dkt", "r_duv", "r_dkbg", "r_dr"):
        ws[name] = torch.empty((nhvc * BT, KD), dtype=torch.float32, device=dev)
    for name in ("daqr", "daq", "dlr", "dl"):
        ws[name] = torch.empty((nhvc * BT, 64), dtype=torch.float32, device=dev)
    ws["h"] = torch.empty(
        (ctx.B * ctx.HV * (ctx.NT + 1) * KD, KD), dtype=torch.float32, device=dev
    )
    ws["vn"] = torch.empty(
        (ctx.B * ctx.HV * ctx.NT * BT, KD), dtype=torch.float32, device=dev
    )
    ws["dh"] = torch.empty(
        (ctx.B * ctx.HV * (ctx.NT + 1) * KD, KD), dtype=torch.float32, device=dev
    )
    ws["ident"] = torch.eye(64, dtype=torch.float32, device=dev)
    ws["nid"] = -torch.eye(KD, dtype=torch.float32, device=dev)
    # R11: the pair kernel indexes ``wv`` (w*v relay) and ``u`` (U relay)
    # by chunk parity, so they need two slots per task; ``wh`` is consumed
    # by v_new inside the same chunk and keeps one.  The non-pair kernel
    # keeps addressing slot 0 of each task, so its layout is unchanged.
    for name in ("wv", "u"):
        ws[name] = torch.empty(
            (2 * ctx.B * ctx.HV * BT, KD), dtype=torch.float32, device=dev
        )
    ws["wh"] = torch.empty((ctx.B * ctx.HV * BT, KD), dtype=torch.float32, device=dev)
    # ``dvn``/``dvnk`` are addressed once per *task* (K2/K3 use
    # ``sbase = t*BT*KD`` with ``t < B*HV``), so this B*HV slot layout is all
    # they need.  ``dop`` must NOT be re-allocated here: K4's vector pass writes
    # it at the *global* (task, chunk) slot ``t = (b*HV + vi)*NT + n``, i.e.
    # ``nhvc = B*HV*NT`` slots (see ``sbase`` in grad_kernel).  Shrinking it to
    # the K3 per-task layout made every K4 launch write ``(NT-1)*B*HV`` slots
    # (= (NT-1)*B*HV*BT*KD*4 bytes) past the end of the tensor and trample
    # whatever the allocator had placed next (a workspace-overlap bug).
    for name in ("dvn", "dvnk"):
        ws[name] = torch.empty(
            (ctx.B * ctx.HV * BT, KD), dtype=torch.float32, device=dev
        )
    # K3 relays the 128x128 L0C tiles through GM at dh-slot index
    # ``(t*(NT+1) + n) * KD*KD`` (one slot per (task, chunk)), so these buffers
    # need the same NT+1 slot layout as ``h``/``dh``.  Allocating only B*HV slots
    # made K3 write up to slot (B*HV-1)*(NT+1)+NT: past NT~12 the write ran off
    # the last mapped allocator segment and the AIC faulted (a relay-slot
    # overflow, fixed by sizing the relay by the whole chain).
    for name in ("dha", "dhb"):
        ws[name] = torch.empty(
            (ctx.B * ctx.HV * (ctx.NT + 1) * KD, KD),
            dtype=torch.float32,
            device=dev,
        )
    ws["hp"] = torch.empty((ctx.B * ctx.HV * KD, KD), dtype=torch.float32, device=dev)
    return ws


def _cores(dev):
    key = str(dev)
    n = _CORES_CACHE.get(key)
    if n is None:
        with torch.npu.device(dev):
            n = torch.npu.get_device_properties(dev).cube_core_num
        _CORES_CACHE[key] = n
    return n


def _vcores(dev):
    """R20: vector-only kernels (K5) get one block per AIV, not per AIC.

    ``cube_core_num`` is 28 on this part but ``vector_core_num`` is 56, and a
    ``tla.vector()``-only kernel maps one block to one AIV.  Launching K5 with
    the cube count left half the vector cores idle (measured -26.4% on 001 when
    fixed; the grid only partitions tasks, so results are bitwise identical).
    """
    key = str(dev)
    n = _VCORES_CACHE.get(key)
    if n is None:
        with torch.npu.device(dev):
            props = torch.npu.get_device_properties(dev)
        n = getattr(props, "vector_core_num", 0) or props.cube_core_num
        _VCORES_CACHE[key] = n
    return n


def _streams_for(ctx, ngroups):
    """Reusable per-device streams: creating them costs host time per call."""
    key = (str(ctx.device), ngroups)
    streams = _STREAM_CACHE.get(key)
    if streams is None:
        streams = [torch.npu.Stream(device=ctx.device) for _ in range(ngroups)]
        _STREAM_CACHE[key] = streams
    return streams


def _setup(q, k, v, g, b, w_gate, do, scale, initial_state):
    scale = validate(q, k, v, g, b, w_gate, do, scale, initial_state)
    ctx = Ctx()
    ctx.q, ctx.k, ctx.v, ctx.g, ctx.b, ctx.w, ctx.do = q, k, v, g, b, w_gate, do
    ctx.scale = scale
    ctx.initial_state = initial_state
    ctx.device = q.device
    ctx.B, ctx.HK, ctx.T, _ = q.shape
    ctx.HV = v.shape[1]
    ctx.HG = g.shape[1]
    ctx.HB = b.shape[1]
    ctx.HF = max(ctx.HK, ctx.HG, ctx.HB)
    ctx.NT = (ctx.T + BT - 1) // BT
    ctx.F = ctx.B * ctx.HF * ctx.NT
    ctx.ratio = ctx.HV // ctx.HK
    # ``HF != HK`` (g/b given per *value* head, i.e. ``HG == HV``) is supported
    # by K5's ``REDGB == 0`` branch: the dq/dk accumulators survive the loop
    # over the ``ratio`` value heads of one k head (GVA sum), while db/dg are
    # finished *inside* that loop at the per-hv offset ``goff`` (no GVA sum).
    # It needs no extra host buffer.
    dev = q.device
    stream = _stream_key(dev)
    ent = _workspace(ctx, dev, stream)
    if _lease(ent):
        plan, ws = ent.plan, ent.ws
        ctx.entry = ent
    else:
        # Same (device, stream, shape) bucket is being driven by another
        # thread right now: give this call private buffers rather than let two
        # calls interleave their ``h``/``dh`` writes on one workspace.  Rare,
        # and the private buffers are dropped when the call returns.
        ws = _alloc_ws(ctx, dev)
        plan = _Plan(ws)
        ctx.entry = None
    ctx.ws = ws
    ctx.plan = plan
    try:
        # Only ``h`` slot 0 (the initial state) and ``dh`` slot ``NT`` (the
        # incoming gradient) are read before being written: K2 fills
        # ``h[1..NT]`` in order, K3 fills ``dh[0..NT-1]`` in reverse.  Zeroing the
        # whole 2 x B*HV*(NT+1) x 128 x 128 fp32 (= 2 x 371 MiB on 001) cost
        # 0.44 ms of pure device time.
        hstate = ws["h"].view(ctx.B * ctx.HV, ctx.NT + 1, KD * KD)
        if initial_state is not None:
            hstate[:, 0].copy_(initial_state.reshape(ctx.B * ctx.HV, KD * KD))
        else:
            hstate[:, 0].zero_()
        ws["dh"].view(ctx.B * ctx.HV, ctx.NT + 1, KD * KD)[:, ctx.NT].zero_()
        outs = {}
        outs["dq"] = torch.empty(q.shape, dtype=q.dtype, device=dev)
        outs["dk"] = torch.empty(k.shape, dtype=k.dtype, device=dev)
        outs["dv"] = torch.empty(v.shape, dtype=v.dtype, device=dev)
        outs["db"] = torch.empty(b.shape, dtype=b.dtype, device=dev)
        outs["dw"] = torch.empty(w_gate.shape, dtype=w_gate.dtype, device=dev)
        outs["dg"] = torch.empty(g.shape, dtype=g.dtype, device=dev)
        ws["vout_dv"] = outs["dv"]
        ws["vout_dw"] = outs["dw"]
        ctx.outs = outs
        ctx.cores = _cores(dev)
        # R20: K5 is vector-only, so its grid is the AIV count (56), not the AIC
        # count (28).  Mixed kernels keep ``ctx.cores``.
        ctx.vcores = _vcores(dev)
    except BaseException:
        if ctx.entry is not None:
            _unlease(ctx.entry)
        raise
    return ctx


def launch_prep(ctx, compile_only=False):
    qm, gm, bm = _mode(ctx.q), _mode(ctx.g), _mode(ctx.b)
    bindings = [
        _bind(ctx.q),
        _bind(ctx.k),
        _bind(ctx.g),
        _bind(ctx.b),
        *ctx.plan.prep,
    ]
    scalars = [ctx.T, ctx.HK, ctx.HG, ctx.HB, ctx.HF, ctx.F]
    consts = [qm, gm, bm]
    key = ("prep", ctx.q.dtype, ctx.g.dtype, ctx.b.dtype)
    art = _artifact(prep_kernel, key, bindings, scalars, consts)
    if not compile_only:
        art(*bindings, *scalars, block_num=min(ctx.cores, ctx.F))
    return art


def launch_fwd(ctx, task0=0, ntask=None, compile_only=False):
    if ntask is None:
        ntask = ctx.B * ctx.HV
    bindings = [_bind(ctx.v), _bind(ctx.w), *ctx.plan.fwd]
    scalars = [ctx.T, ctx.HK, ctx.HV, ctx.HF, ntask, task0]
    consts = [_mode(ctx.v)]
    key = ("fwd", ctx.v.dtype)
    art = _artifact(fwd_kernel, key, bindings, scalars, consts)
    if not compile_only:
        art(*bindings, *scalars, block_num=min(ctx.cores, ntask))
    return art


def launch_fwd_pair(ctx, task0=0, ntask=None, compile_only=False):
    if ntask is None:
        ntask = ctx.B * ctx.HV
    bindings = [_bind(ctx.v), _bind(ctx.w), *ctx.plan.fwd]
    scalars = [ctx.T, ctx.HK, ctx.HV, ctx.HF, ntask, task0]
    consts = [_mode(ctx.v)]
    key = ("fwd_pair", ctx.v.dtype)
    art = _artifact(fwd_kernel_pair, key, bindings, scalars, consts)
    if not compile_only:
        art(*bindings, *scalars, block_num=min(ctx.cores, ntask // 2))
    return art


def launch_bwd(ctx, task0=0, ntask=None, compile_only=False):
    if ntask is None:
        ntask = ctx.B * ctx.HV
    bindings = [_bind(ctx.do), *ctx.plan.bwd]
    scalars = [ctx.T, ctx.HK, ctx.HV, ctx.HF, ntask, task0, ctx.scale]
    consts = [_mode(ctx.do)]
    key = ("bwd", ctx.do.dtype)
    art = _artifact(bwd_kernel, key, bindings, scalars, consts)
    if not compile_only:
        art(*bindings, *scalars, block_num=min(ctx.cores, ntask))
    return art


def launch_bwd_pair(ctx, task0=0, ntask=None, compile_only=False):
    if ntask is None:
        ntask = ctx.B * ctx.HV
    bindings = [_bind(ctx.do), *ctx.plan.bwd]
    scalars = [ctx.T, ctx.HK, ctx.HV, ctx.HF, ntask, task0, ctx.scale]
    consts = [_mode(ctx.do)]
    key = ("bwd_pair", ctx.do.dtype)
    art = _artifact(bwd_kernel_pair, key, bindings, scalars, consts)
    if not compile_only:
        art(*bindings, *scalars, block_num=min(ctx.cores, ntask // 2))
    return art


def launch_grad(ctx, compile_only=False):
    bindings = [
        _bind(ctx.v),
        _bind(ctx.w),
        _bind(ctx.do),
        _bind(ctx.outs["dv"]),
        _bind(ctx.outs["dw"]),
        *ctx.plan.grad,
        _bind(ctx.ws["dha"]),
    ]
    nhvc = ctx.B * ctx.HV * ctx.NT
    scalars = [ctx.T, ctx.HK, ctx.HV, ctx.HF, nhvc, ctx.scale]
    consts = [_mode(ctx.do)]
    key = ("grad", ctx.v.dtype, ctx.do.dtype)
    art = _artifact(grad_kernel, key, bindings, scalars, consts)
    if not compile_only:
        art(*bindings, *scalars, block_num=min(ctx.cores, nhvc))
    return art


def launch_reduce(ctx, compile_only=False):
    outs = ctx.outs
    bindings = [
        _bind(outs["dq"]),
        _bind(outs["dk"]),
        _bind(outs["db"]),
        _bind(outs["dg"]),
        *ctx.plan.reduce,
    ]
    ntask = ctx.B * ctx.HK * ctx.NT
    scalars = [ctx.T, ctx.HK, ctx.HG, ctx.HV, ctx.HF, ctx.ratio, ntask]
    consts = [
        1 if ctx.HF == ctx.HK else 0,
        _mode(ctx.q),
        _mode(ctx.g),
    ]
    key = ("reduce", ctx.q.dtype, ctx.g.dtype, ctx.HF == ctx.HK)
    art = _artifact(reduce_kernel, key, bindings, scalars, consts)
    if not compile_only:
        art(*bindings, *scalars, block_num=min(ctx.vcores, ntask))
    return art


def dh0_of(ctx):
    ws = ctx.ws
    return ws["dh"].view(ctx.B, ctx.HV, ctx.NT + 1, KD, KD)[:, :, 0].contiguous()


def run_state(ctx):
    """K2+K3: when there are more (b,hv) chains than cube cores, split the
    chains into groups and run each group's forward+backward chain on its own
    stream.  The groups are independent (disjoint h/dh slots, dh starts at 0
    for every chain), so the streams overlap and the second wave of K2 fills
    the cores freed by the first group's K3 instead of running 4 chains on 28
    cores.  With ``ntask <= cores`` this is exactly the historical single
    launch (no streams, no behaviour change).

    R16: when the chains are past one wave (``ntask > cores``) and even, the
    forward pass runs ``fwd_kernel_pair`` -- two chains interleaved per block,
    so ``ntask/2`` blocks fit in one wave.  ``GDN_PAIR=0`` restores the
    historical one-chain-per-block launch.
    """
    ntask = ctx.B * ctx.HV
    cores = ctx.cores
    if ntask <= cores:
        _ = launch_fwd(ctx)
        _ = launch_bwd(ctx)
        return
    if _PAIR and ntask % 2 == 0:
        npair = ntask // 2
        ngroups = (npair + cores - 1) // cores
        if ngroups == 1:
            _ = launch_fwd_pair(ctx, 0, ntask)
            _ = launch_bwd_pair(ctx, 0, ntask)
            return
        base, rem = divmod(npair, ngroups)
        cur = torch.npu.current_stream()
        streams = _streams_for(ctx, ngroups)
        for s in streams:
            s.wait_stream(cur)
        lo = 0
        for gi, s in enumerate(streams):
            sz = base + (1 if gi < rem else 0)
            with torch.npu.stream(s):
                _ = launch_fwd_pair(ctx, 2 * lo, 2 * sz)
                _ = launch_bwd_pair(ctx, 2 * lo, 2 * sz)
            lo += sz
        for s in streams:
            cur.wait_stream(s)
        return
    ngroups = (ntask + cores - 1) // cores
    base, rem = divmod(ntask, ngroups)
    cur = torch.npu.current_stream()
    # ``_streams_for`` caches the stream objects per device: building them on
    # every call costs host time and they are stateless between launches.
    streams = _streams_for(ctx, ngroups)
    for s in streams:
        s.wait_stream(cur)
    lo = 0
    for gi, s in enumerate(streams):
        sz = base + (1 if gi < rem else 0)
        with torch.npu.stream(s):
            _ = launch_fwd(ctx, lo, sz)
            _ = launch_bwd(ctx, lo, sz)
        lo += sz
    for s in streams:
        cur.wait_stream(s)


def _bnsd(x):
    """``[B, T, H, D]`` -> contiguous ``[B, H, T, D]`` for the BNSD kernels."""
    return x.transpose(1, 2).contiguous()


def gdn2_bwd(q, k, v, g, b, w_gate, do, scale=None, initial_state=None, layout="BNSD"):
    """GDN-2 backward for one contiguous batch.

    Parameters follow ``reference.py``: ``q/k`` are ``[B, HK, T, 128]``,
    ``v/w_gate/do`` are ``[B, HV, T, 128]``, ``g/b`` both take ``HK`` or both
    take ``HV`` heads (they must share one head count, ``HG == HB``), and
    ``initial_state`` is an optional fp32 K-first ``[B, HV, 128, 128]`` tensor.
    ``scale`` defaults to ``K**-0.5``.

    ``layout`` selects the token axis: ``"BNSD"`` (default) keeps the shapes
    above, ``"BSND"`` takes every input as ``[B, T, H, 128]`` and returns the
    six token-major gradients as ``[B, T, H, 128]``.  The transposes are host
    side ``transpose(1, 2).contiguous()`` copies outside the kernels, so a
    BSND call pays for 13 extra contiguous copies (7 inputs, 6 outputs);
    ``dh0`` stays ``[B, HV, 128, 128]`` in both layouts and ``initial_state``
    keeps its layout-independent K-first semantics.

    Returns ``(dq, dk, dv, db, dw, dg, dh0)``; the first six match the dtype of
    their input and none of them aliases an input.  Inputs are not modified and
    no autograd graph is built.  The five kernels are enqueued on the current
    stream of the input device; the caller synchronises when it needs results.
    """
    if layout not in ("BNSD", "BSND"):
        raise ValueError(f"layout must be 'BNSD' or 'BSND', got {layout!r}")
    if layout == "BSND":
        tens = (q, k, v, g, b, w_gate, do)
        if any(not isinstance(x, torch.Tensor) for x in tens):
            raise TypeError("q, k, v, g, b, w_gate, do must be tensors")
        q, k, v, g, b, w_gate, do = (_bnsd(x) for x in tens)
    ctx = _setup(q, k, v, g, b, w_gate, do, scale, initial_state)
    try:
        _ = launch_prep(ctx)
        run_state(ctx)
        _ = launch_grad(ctx)
        _ = launch_reduce(ctx)
        outs = ctx.outs
        grads = (outs["dq"], outs["dk"], outs["dv"], outs["db"], outs["dw"], outs["dg"])
        if layout == "BSND":
            grads = tuple(_bnsd(x) for x in grads)
        # ``dh0_of`` reads the workspace, so it stays inside the lease too.
        dh0 = dh0_of(ctx)
    finally:
        if ctx.entry is not None:
            _unlease(ctx.entry)
    return (*grads, dh0)


def reference_shapes(ctx):
    return ctx.B, ctx.HK, ctx.HV, ctx.T


# ``run`` is the name the verification tools use; the public entry point
# is ``gdn2_bwd``.
run = gdn2_bwd

__all__ = ["gdn2_bwd"]
