# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L0C→UB b8 + unit_flag: pad cube extents so Fixpipe and mmad fractal counts match.

Background
----------
With ``CopyL0C2DstParams.unit_flag != 0`` and a 1-byte UB destination, Fixpipe
rounds the write extent up to a 32-element block (RowMajor: N, ColumnMajor: M)
while mmad under unitFlag still counts fractals by 16. If the logical tile is
only 16-aligned (e.g. M=33→RoundUp 64 on Fixpipe vs 48 on mmad), Fixpipe waits
for fractals mmad never produced and the core hangs.

This sample keeps ``unit_flag=0b11`` and shows the pad workaround:

* **ColumnMajor** write: round **M** up to 32 for L1A / L0A / L0C / UB.
  Build L1A / L0A with the **aligned** layout; ``get_tile`` + GM→L1 copies only
  the **valid** M; L1→L0A / ``tla.mmad`` / Fixpipe use the aligned frame.
* **RowMajor** write: same on **N** for L1B / L0B / L0C / UB (L0B stays ``nZ``).

GM stores only the valid ``(m, n)`` crop.

Demo (the hang shape from review): ``--m 33 --n 44 --k 64 --layout-c col --dtype-c i8``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_DSL_EXAMPLE_PATH = str(
    next(p for p in Path(__file__).resolve().parents if (p / "helpers").is_dir())
)

if _DSL_EXAMPLE_PATH not in sys.path:
    sys.path.append(_DSL_EXAMPLE_PATH)

import argparse

import torch
import torch_npu
from helpers import TilingParams, create_tla_tensor, get_block_num, compare

import catlass.tla as tla
from catlass.params import CopyL0C2DstParams, L0C2UBMode, QuantMode

# Single L1/L0 tile large enough for m_align/n_align ≤ 64.
_TILE = TilingParams(l1_tm=64, l1_tn=64, l1_tk=64, l0_tm=64, l0_tn=64, l0_tk=64)
_UNIT_ALIGN = 32


def _round_up(v: int, align: int) -> int:
    return ((int(v) + align - 1) // align) * align


@tla.kernel
def basic_mixed_l0c2ub_b8_unitflag_pad_kernel(
    mem_a: tla.Tensor,
    mem_b: tla.Tensor,
    mem_c: tla.Tensor,
    m_valid: tla.Constexpr[int],
    n_valid: tla.Constexpr[int],
    k_dim: tla.Constexpr[int],
    m_cube: tla.Constexpr[int],
    n_cube: tla.Constexpr[int],
    trans_c: tla.Constexpr[bool],
    quant_scale: float,
    quant_offset: int,
):
    """One-block mixed matmul; cube frame may be larger than the GM-valid tile."""
    c0 = 0
    dtype_a = mem_a.ptr.dtype
    dtype_b = mem_b.ptr.dtype
    dtype_c = mem_c.ptr.dtype
    dst_is_b8 = dtype_c in (tla.Int8, tla.UInt8)
    acc_dtype = tla.Float32

    l1_full = tla.flag("l1_full", tla.arch.MTE2, tla.arch.MTE1)
    l1_free = tla.flag("l1_free", tla.arch.MTE1, tla.arch.MTE2)
    l0_full = tla.flag("l0_full", tla.arch.MTE1, tla.arch.CUBE)
    l0_free = tla.flag("l0_free", tla.arch.CUBE, tla.arch.MTE1)
    l0c_available = tla.flag("l0c_available", tla.arch.FIX, tla.arch.CUBE)
    fix_done = tla.cross_flag("fix_done", mode=4)
    fix_ready = tla.cross_flag("fix_ready", mode=4)

    # zN / nZ / L0C fractal storage needs 16-aligned extents; size buffers by
    # the L0 tile so a valid M/N of 33 still fits the packed layout.
    l1a_ptr = tla.allocate(_TILE.l0_tm * _TILE.l0_tk, dtype_a, tla.AddressSpace.l1, 512)
    l1b_ptr = tla.allocate(_TILE.l0_tk * _TILE.l0_tn, dtype_b, tla.AddressSpace.l1, 512)
    l0a_ptr = tla.allocate(
        _TILE.l0_tm * _TILE.l0_tk, dtype_a, tla.AddressSpace.l0a, 512
    )
    l0b_ptr = tla.allocate(
        _TILE.l0_tk * _TILE.l0_tn, dtype_b, tla.AddressSpace.l0b, 512
    )
    l0c_ptr = tla.allocate(
        _TILE.l0_tm * _TILE.l0_tn, acc_dtype, tla.AddressSpace.l0c, 1024
    )
    ub_ptr = tla.allocate(_TILE.l0_tm * _TILE.l0_tn, dtype_c, tla.AddressSpace.ub, 256)

    c_tag = tla.arch.ColumnMajor if trans_c else tla.arch.RowMajor

    with tla.cube():
        tla.set_flag(l1_free)
        tla.set_flag(l0_free)
        tla.set_flag(l0c_available)

        gm_a = tla.tile_view(
            mem_a, tla.make_shape(m_valid, k_dim), tla.make_coord(c0, c0)
        )
        gm_b = tla.tile_view(
            mem_b, tla.make_shape(k_dim, n_valid), tla.make_coord(c0, c0)
        )

        tla.wait_flag(l1_free)
        # ColumnMajor: aligned L1A; get_tile + GM→L1 loads only valid M.
        if tla.const_expr(trans_c):
            l1_a = tla.make_tensor(
                l1a_ptr,
                tla.make_layout_with_tag((m_cube, k_dim), dtype_a, tla.arch.zN),
            )
            l1_a_load = tla.get_tile(
                l1_a, tla.make_coord(c0, c0), tla.make_shape(m_valid, k_dim)
            )
            tla.copy(l1_a_load, gm_a)
        else:
            l1_a = tla.make_tensor_like(l1a_ptr, gm_a, tla.arch.zN)
            tla.copy(l1_a, gm_a)

        # RowMajor: aligned L1B; get_tile + GM→L1 loads only valid N.
        if tla.const_expr(not trans_c):
            l1_b = tla.make_tensor(
                l1b_ptr,
                tla.make_layout_with_tag((k_dim, n_cube), dtype_b, tla.arch.zN),
            )
            l1_b_load = tla.get_tile(
                l1_b, tla.make_coord(c0, c0), tla.make_shape(k_dim, n_valid)
            )
            tla.copy(l1_b_load, gm_b)
        else:
            l1_b = tla.make_tensor_like(l1b_ptr, gm_b, tla.arch.zN)
            tla.copy(l1_b, gm_b)
        tla.set_flag(l1_full)

        tla.wait_flag(l1_full)
        tla.wait_flag(l0_free)

        # L1→L0: full cube frame (aligned on the write axis) for mmad.
        if tla.const_expr(trans_c):
            l0_a = tla.make_tensor(
                l0a_ptr,
                tla.make_layout_with_tag((m_cube, k_dim), dtype_a, tla.arch.zN),
            )
            tla.copy(l0_a, l1_a)
        else:
            l0_a = tla.make_tensor_like(l0a_ptr, l1_a, tla.arch.zN)
            tla.copy(l0_a, l1_a)

        if tla.const_expr(not trans_c):
            l0_b = tla.make_tensor(
                l0b_ptr,
                tla.make_layout_with_tag((k_dim, n_cube), dtype_b, tla.arch.nZ),
            )
            tla.copy(l0_b, l1_b)  # L1 zN → L0 nZ
        else:
            l0_b = tla.make_tensor_like(l0b_ptr, l1_b, tla.arch.nZ)
            tla.copy(l0_b, l1_b)

        tla.set_flag(l1_free)
        tla.set_flag(l0_full)

        # L0C / UB share the same aligned cube frame as Fixpipe.
        l0_c = tla.make_tensor(
            l0c_ptr,
            tla.make_layout_with_tag((m_cube, n_cube), acc_dtype, tla.arch.L0Clayout),
        )
        tla.wait_flag(l0_full)
        tla.wait_flag(l0c_available)
        tla.mmad(l0_c, l0_a, l0_b, init_c=True, unit_flag=0b11)
        tla.set_flag(l0_free)

        ub_c = tla.make_tensor(
            ub_ptr,
            tla.make_layout(
                tla.make_shape(m_cube, n_cube),
                (tla.make_stride(1, m_cube) if trans_c else tla.make_stride(n_cube, 1)),
                layoutTag=c_tag,
            ),
        )
        tla.cross_core_wait_flag(fix_ready, tla.arch.FIX, aiv_id=0)
        tla.copy(
            ub_c,
            l0_c,
            params=CopyL0C2DstParams(
                unit_flag=0b11,
                l0c2ub_mode=L0C2UBMode.NO_SPLIT_VEC_0,
                quant_mode=QuantMode.PER_TENSOR,
                quant_scale=quant_scale,
                quant_offset=quant_offset if dst_is_b8 else None,
            ),
        )
        tla.set_flag(l0c_available)
        tla.cross_core_set_flag(fix_done, tla.arch.FIX, aiv_id=0)

        tla.cross_core_wait_flag(fix_ready, tla.arch.FIX, aiv_id=0)
        tla.wait_flag(l1_free)
        tla.wait_flag(l0_free)
        tla.wait_flag(l0c_available)
        tla.pipe_barrier(tla.pipes.ALL)

    with tla.vector():
        vec_idx = tla.arch.sub_block_idx()
        if vec_idx == c0:
            tla.cross_core_set_flag(fix_ready, tla.arch.MTE3, aiv_id=0)

            # Fixpipe Col UB is (n_cube, m_cube) RowMajor; Row is (m_cube, n_cube).
            # Store only the GM-valid crop.
            if tla.const_expr(trans_c):
                out_rows, out_cols = n_valid, m_valid
                ub_pitch = m_cube
                gm_pitch = mem_c.stride[1]
            else:
                out_rows, out_cols = m_valid, n_valid
                ub_pitch = n_cube
                gm_pitch = mem_c.stride[0]

            gm_out = tla.make_tensor(
                mem_c.ptr,
                tla.make_layout(
                    tla.make_shape(out_rows, out_cols),
                    tla.make_stride(gm_pitch, 1),
                    layoutTag=tla.arch.RowMajor,
                ),
                coord=tla.make_coord(c0, c0),
            )
            ub_src = tla.make_tensor(
                ub_ptr,
                tla.make_layout(
                    tla.make_shape(out_rows, out_cols),
                    tla.make_stride(ub_pitch, 1),
                    layoutTag=tla.arch.RowMajor,
                ),
            )
            tla.cross_core_wait_flag(fix_done, tla.arch.MTE3, aiv_id=0)
            tla.copy(gm_out, ub_src)
            tla.cross_core_set_flag(fix_ready, tla.arch.MTE3, aiv_id=0)

        tla.pipe_barrier(tla.pipes.ALL)


_INT_DST_RANGES = {
    torch.int8: (-128, 127),
    torch.uint8: (0, 255),
}


def prepare_quant_scale(x: float) -> float:
    t = torch.tensor(x, dtype=torch.float32)
    bits = t.view(torch.uint32) & 0xFFFFE000
    return float(bits.view(torch.float32).item())


def per_tensor_quant_golden(
    acc: torch.Tensor,
    dst_dtype: torch.dtype,
    quant_scale: float,
    quant_offset: int = 0,
) -> torch.Tensor:
    scaled = acc.float() * torch.tensor(quant_scale, dtype=torch.float32)
    lo, hi = _INT_DST_RANGES[dst_dtype]
    quantized = torch.round(torch.clamp(scaled, -256, 255)).to(torch.int16)
    quantized += int(quant_offset)
    return torch.round(torch.clamp(quantized, lo, hi)).to(dst_dtype)


def run(args: argparse.Namespace) -> int:
    if min(args.m, args.n, args.k) <= 0:
        raise ValueError(f"M/N/K must be positive; received {args.m}/{args.n}/{args.k}")
    if max(args.m, args.n, args.k) > 64:
        raise ValueError("b8 unitflag pad requires 1 <= M, N, K <= 64")

    m, n, k = int(args.m), int(args.n), int(args.k)
    if k > _TILE.l1_tk:
        raise SystemExit(f"k must fit one L1 tile (k<={_TILE.l1_tk}); got {k}")
    if m > _TILE.l1_tm or n > _TILE.l1_tn:
        raise SystemExit(
            f"m/n must fit one L1 tile (m<={_TILE.l1_tm}, n<={_TILE.l1_tn}); "
            f"got m={m} n={n}"
        )

    trans_c = args.layout_c == "col"
    # Align only the Fixpipe write axis to 32 for unit_flag + b8.
    if trans_c:
        m_cube, n_cube = _round_up(m, _UNIT_ALIGN), n
    else:
        m_cube, n_cube = m, _round_up(n, _UNIT_ALIGN)
    if m_cube > _TILE.l0_tm or n_cube > _TILE.l0_tn:
        raise SystemExit(
            f"aligned cube frame ({m_cube},{n_cube}) exceeds L0 tile "
            f"({_TILE.l0_tm},{_TILE.l0_tn})"
        )

    torch.npu.set_device(args.device)

    dtype_c = torch.int8 if args.dtype_c == "i8" else torch.uint8
    print(
        f"--- mnk=({m},{n},{k}) layout_c={args.layout_c} "
        f"cube_frame=({m_cube},{n_cube}) dtype_c={args.dtype_c} "
        f"quant_scale={args.quant_scale} ---"
    )

    a = torch.rand(m, k, dtype=torch.float32) * 10.0 - 5.0
    b = torch.rand(k, n, dtype=torch.float32) * 10.0 - 5.0
    acc_ref = a @ b
    host_golden = per_tensor_quant_golden(
        acc_ref, dtype_c, args.quant_scale, args.quant_offset
    )

    c = torch.zeros(m, n, dtype=dtype_c)
    if trans_c:
        c = c.permute(1, 0)
    a_n = a.contiguous().npu()
    b_n = b.contiguous().npu()
    c_n = c.contiguous().npu()
    a_t = create_tla_tensor(a_n, "row")
    b_t = create_tla_tensor(b_n, "row")
    c_t = create_tla_tensor(c_n, args.layout_c)

    artifact = tla.compile(
        basic_mixed_l0c2ub_b8_unitflag_pad_kernel,
        a_t,
        b_t,
        c_t,
        m,
        n,
        k,
        m_cube,
        n_cube,
        trans_c,
        args.quant_scale,
        args.quant_offset,
        options="--npu-arch 3510",
    )
    block_num = get_block_num(args.block_num, args.device, kind="mix")
    artifact(
        a_t,
        b_t,
        c_t,
        args.quant_scale,
        args.quant_offset,
        block_num=block_num,
    )
    torch.npu.synchronize()

    result = c_n.detach().cpu()
    if trans_c:
        result = result.transpose(0, 1).contiguous()
    max_delta = (
        (result.to(torch.int32) - host_golden.to(torch.int32)).abs().max().item()
    )
    passed = max_delta <= 1
    print(f"passed={passed} max_abs_delta={max_delta} cache_key={artifact.cache_key}")
    if not passed:
        print("result", result)
        print("host_golden", host_golden)
        _ = compare  # keep import used if helpers change
    return 0 if passed else 1


def main() -> int:
    torch.manual_seed(0)
    parser = argparse.ArgumentParser(
        description="L0C→UB b8 unit_flag pad workaround (align M for col / N for row)."
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--m", type=int, default=33, help="valid M (col pads to %%32)")
    parser.add_argument("--n", type=int, default=44, help="valid N (row pads to %%32)")
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--layout-c", choices=("row", "col"), default="col")
    parser.add_argument("--dtype-c", choices=("i8", "u8"), default="i8")
    parser.add_argument("--quant-scale", type=float, default=0.5)
    parser.add_argument("--quant-offset", type=int, default=0)
    parser.add_argument("--block-num", type=int, default=1)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
