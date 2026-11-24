# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Basic Mixed with fixpipe per-tensor quant: Kernel + Host in one file.

The mixed counterpart of ``basic_mmad/basic_matmul_fixpipe_per_tensor.py``: the
same arguments and reference models, plus what the mixed pipeline demands -- a
``TilingParams`` block, a ``trans_c`` destination switch and ``block_num`` taken
from the cube core count.

Destination dtypes match the L0C→UB ``PER_TENSOR`` row of the fixpipe route
table: f32 / f16 / bf16 / i8 / u8 / f8e4m3fn / hif8 (fp8/hif8 need an f32
accumulator). ``quant_scale`` is a runtime scalar; ``quant_offset`` applies
only to i8/u8.

L0C→UB 1-byte destinations with ``unit_flag != 0`` require the UB write
extent 32-aligned (RowMajor ``N``, ColumnMajor ``M``); this sample keeps
``unit_flag=0b11`` and uses a 128-wide tile so the check holds. Unaligned
tiles must drop to ``unit_flag=0`` and pair mmad with flag sync.

``--layout-c col`` stores the result ColumnMajor: the L0C transfer then runs
through the transposing (nz2dn) fixpipe, so the tile is written transposed and
quantized in one pass.

The kernel walks the grid of tiles -- no K split, no swizzle -- and hands each
task block to one AIV by parity. That split is why the fixpipe names
``NO_SPLIT_VEC_0`` / ``NO_SPLIT_VEC_1``: the transfer, not the cross flag,
decides which AIV's UB the tile lands in. K has to fit one L1 tile.

UB is allocated once: each AIV has its own address space, so the same offset
resolves to a private tile on either side. Its row pitch is the full tile
extent, not the clipped tile's own extent; only the problem's own rows and
columns are stored to GM.
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
from helpers import TilingParams

import catlass.tla as tla
from catlass.params import CopyL0C2DstParams, L0C2UBMode, QuantMode

# Smaller than TilingParams' default (256x256): the tile has to fit UB for the
# fixpipe, and nothing below a tile is split. 128 keeps 1-byte + unit_flag
# extents 32-aligned (RowMajor N / ColumnMajor M).
MIXED_TILING = TilingParams(
    l1_tm=128, l1_tn=128, l1_tk=64, l0_tm=128, l0_tn=128, l0_tk=64
)


@tla.kernel
def basic_mixed_fixpipe_per_tensor_kernel(
    mem_a: tla.Tensor,
    mem_b: tla.Tensor,
    mem_c: tla.Tensor,
    _tiling: tla.Constexpr[TilingParams],
    acc_is_int: tla.Constexpr[bool],
    relu_enable: tla.Constexpr[bool],
    trans_c: tla.Constexpr[bool],
    quant_scale: float,
    quant_offset: int,
):
    c0 = 0

    dtype_a = mem_a.ptr.dtype
    dtype_b = mem_b.ptr.dtype
    dtype_c = mem_c.ptr.dtype
    dst_is_b8 = dtype_c in (tla.Int8, tla.UInt8)
    acc_dtype = tla.Int32 if acc_is_int else tla.Float32

    m = mem_a.origin_shape[0]
    n = mem_b.origin_shape[1]
    k = mem_a.origin_shape[1]

    tm = _tiling.l1_tm
    tn = _tiling.l1_tn
    tk = _tiling.l1_tk
    l0m = _tiling.l0_tm
    l0n = _tiling.l0_tn
    l0k = _tiling.l0_tk

    grid_m = (m + tm - 1) // tm
    grid_n = (n + tn - 1) // tn
    total_blocks = grid_m * grid_n

    # Two flags per stage buffer: "data is there" and "buffer is free again".
    # The ``free`` side is seeded before the loop and drained after it.
    l1_full = tla.flag("l1_full", tla.arch.MTE2, tla.arch.MTE1)
    l1_free = tla.flag("l1_free", tla.arch.MTE1, tla.arch.MTE2)
    l0_full = tla.flag("l0_full", tla.arch.MTE1, tla.arch.CUBE)
    l0_free = tla.flag("l0_free", tla.arch.CUBE, tla.arch.MTE1)
    # The cross flags below are only waited on after the MMAD, too late for L0C;
    # this intra-core flag is what keeps the next MMAD off the accumulator.
    l0c_available = tla.flag("l0c_available", tla.arch.FIX, tla.arch.CUBE)

    # One flag pair for both AIVs. mode 4 keeps a separate counter per AIV, so
    # ``aiv_id`` on each crossing picks the channel -- a shared counter (mode 2)
    # could not express "vec0 has drained its UB but vec1 has not".
    fix_done = tla.cross_flag("fix_done", mode=4)
    fix_ready = tla.cross_flag("fix_ready", mode=4)

    l1a_ptr = tla.allocate(tm * tk, dtype_a, tla.AddressSpace.l1, 512)
    l1b_ptr = tla.allocate(tk * tn, dtype_b, tla.AddressSpace.l1, 512)
    l0a_ptr = tla.allocate(l0m * l0k, dtype_a, tla.AddressSpace.l0a, 512)
    l0b_ptr = tla.allocate(l0k * l0n, dtype_b, tla.AddressSpace.l0b, 512)
    # L0C accumulators are fed to tla.mmad, which needs 1024-byte alignment.
    l0c_ptr = tla.allocate(l0m * l0n, acc_dtype, tla.AddressSpace.l0c, 1024)
    # One allocation serves both AIVs: each has its own UB address space, so the
    # same offset resolves to a private tile on either side.
    ub_ptr = tla.allocate(tm * tn, dtype_c, tla.AddressSpace.ub, 256)

    c_tag = tla.arch.ColumnMajor if trans_c else tla.arch.RowMajor

    with tla.cube():
        tla.set_flag(l1_free)
        tla.set_flag(l0_free)
        tla.set_flag(l0c_available)

        block_range = tla.range(
            tla.arch.block_idx(), total_blocks, tla.arch.block_num()
        )
        for block_linear in block_range:
            block_row = block_linear // grid_n
            block_col = block_linear % grid_n

            # tile_view crops at the tensor's edge; nothing is padded on GM.
            gm_a = tla.tile_view(
                mem_a, tla.make_shape(tm, k), tla.make_coord(block_row, c0)
            )
            gm_b = tla.tile_view(
                mem_b, tla.make_shape(k, tn), tla.make_coord(c0, block_col)
            )
            gm_c = tla.tile_view(
                mem_c, tla.make_shape(tm, tn), tla.make_coord(block_row, block_col)
            )

            l1_a = tla.make_tensor_like(l1a_ptr, gm_a, tla.arch.zN)
            l1_b = tla.make_tensor_like(l1b_ptr, gm_b, tla.arch.zN)
            tla.wait_flag(l1_free)
            tla.copy(l1_a, gm_a)
            tla.copy(l1_b, gm_b)
            tla.set_flag(l1_full)

            tla.wait_flag(l1_full)
            tla.wait_flag(l0_free)
            l0_a = tla.make_tensor_like(l0a_ptr, l1_a, tla.arch.zN)
            l0_b = tla.make_tensor_like(l0b_ptr, l1_b, tla.arch.nZ)
            tla.copy(l0_a, l1_a)
            tla.copy(l0_b, l1_b)
            tla.set_flag(l1_free)
            tla.set_flag(l0_full)

            l0_c = tla.make_tensor_like(l0c_ptr, gm_c, tla.arch.L0Clayout)
            tla.wait_flag(l0_full)
            tla.wait_flag(l0c_available)
            # 0b11: the fixpipe starts off this MMAD's own completion signal.
            tla.mmad(l0_c, l0_a, l0_b, init_c=True, unit_flag=0b11)
            tla.set_flag(l0_free)

            # The fixpipe writes rows on a tile-wide pitch, not on the clipped
            # tile's extent and not on the GM stride: RowMajor drains (m, n) at a
            # tn pitch, ColumnMajor transposes and drains (m, n) at a tm pitch.
            ub_c = tla.make_tensor(
                ub_ptr,
                tla.make_layout(
                    tla.make_shape(gm_c.origin_shape[0], gm_c.origin_shape[1]),
                    (tla.make_stride(1, tm) if trans_c else tla.make_stride(tn, 1)),
                    layoutTag=c_tag,
                ),
            )

            # Even blocks belong to vec0, odd ones to vec1.
            # unit_flag=0b11 is safe here because MIXED_TILING keeps N/M
            # 32-aligned for 1-byte UB destinations.
            if block_linear % 2 == 0:
                tla.cross_core_wait_flag(fix_ready, tla.arch.FIX, aiv_id=0)
                tla.copy(
                    ub_c,
                    l0_c,
                    params=CopyL0C2DstParams(
                        unit_flag=0b11,
                        l0c2ub_mode=L0C2UBMode.NO_SPLIT_VEC_0,
                        relu_enable=relu_enable,
                        quant_mode=QuantMode.PER_TENSOR,
                        quant_scale=quant_scale,
                        quant_offset=quant_offset if dst_is_b8 else None,
                    ),
                )
                tla.set_flag(l0c_available)
                tla.cross_core_set_flag(fix_done, tla.arch.FIX, aiv_id=0)
            else:
                tla.cross_core_wait_flag(fix_ready, tla.arch.FIX, aiv_id=1)
                tla.copy(
                    ub_c,
                    l0_c,
                    params=CopyL0C2DstParams(
                        unit_flag=0b11,
                        l0c2ub_mode=L0C2UBMode.NO_SPLIT_VEC_1,
                        relu_enable=relu_enable,
                        quant_mode=QuantMode.PER_TENSOR,
                        quant_scale=quant_scale,
                        quant_offset=quant_offset if dst_is_b8 else None,
                    ),
                )
                tla.set_flag(l0c_available)
                tla.cross_core_set_flag(fix_done, tla.arch.FIX, aiv_id=1)

        # One credit per channel, standing in for an already-spent "UB is free".
        tla.cross_core_wait_flag(fix_ready, tla.arch.FIX, aiv_id=0)
        tla.cross_core_wait_flag(fix_ready, tla.arch.FIX, aiv_id=1)
        tla.wait_flag(l1_free)
        tla.wait_flag(l0_free)
        tla.wait_flag(l0c_available)
        tla.pipe_barrier(tla.pipes.ALL)

    with tla.vector():
        vec_idx = tla.arch.sub_block_idx()

        # A cross flag carries one direction only, so each AIV publishes its own
        # UB as free once before the loop, on its own channel. Publishing on
        # MTE3, the pipe that drains UB to GM, keeps every "UB is free" on one
        # pipe: the set then orders against that AIV's own MTE3 stores.
        if vec_idx == c0:
            tla.cross_core_set_flag(fix_ready, tla.arch.MTE3, aiv_id=0)
        else:
            tla.cross_core_set_flag(fix_ready, tla.arch.MTE3, aiv_id=1)

        block_range = tla.range(
            tla.arch.block_idx(), total_blocks, tla.arch.block_num()
        )
        for block_linear in block_range:
            block_row = block_linear // grid_n
            block_col = block_linear % grid_n

            gm_c = tla.tile_view(
                mem_c, tla.make_shape(tm, tn), tla.make_coord(block_row, block_col)
            )
            m_tile = gm_c.origin_shape[0]
            n_tile = gm_c.origin_shape[1]

            # A transposing fixpipe leaves UB holding the tile as (n, m), a
            # RowMajor destination (m, n); both are row-major either way.
            out_rows = n_tile if trans_c else m_tile
            out_cols = m_tile if trans_c else n_tile
            out_pitch = mem_c.stride[1] if trans_c else mem_c.stride[0]
            out_row_off = block_col * tn if trans_c else block_row * tm
            out_col_off = block_row * tm if trans_c else block_col * tn
            gm_out = tla.make_tensor(
                mem_c.ptr,
                tla.make_layout(
                    tla.make_shape(out_rows, out_cols),
                    tla.make_stride(out_pitch, 1),
                    layoutTag=tla.arch.RowMajor,
                ),
                coord=tla.make_coord(out_row_off, out_col_off),
            )

            # UB is pitched on the full tile extent, the destination on its own.
            ub_src = tla.make_tensor(
                ub_ptr,
                tla.make_layout(
                    tla.make_shape(out_rows, out_cols),
                    tla.make_stride(
                        tm if trans_c else tn,
                        1,
                    ),
                    layoutTag=tla.arch.RowMajor,
                ),
            )
            # The store out of UB is an MTE3 operation, so both halves of the
            # handshake must ride MTE3: the wait, and the set that tells the
            # fixpipe the tile is drained. A set on another pipe would be issued
            # in parallel with the copy, letting the next fixpipe write UB while
            # this store still reads it.
            if vec_idx == c0:
                if block_linear % 2 == c0:
                    tla.cross_core_wait_flag(fix_done, tla.arch.MTE3, aiv_id=0)
                    tla.copy(gm_out, ub_src)
                    tla.cross_core_set_flag(fix_ready, tla.arch.MTE3, aiv_id=0)
            else:
                if block_linear % 2 == 1:
                    tla.cross_core_wait_flag(fix_done, tla.arch.MTE3, aiv_id=1)
                    tla.copy(gm_out, ub_src)
                    tla.cross_core_set_flag(fix_ready, tla.arch.MTE3, aiv_id=1)

        tla.pipe_barrier(tla.pipes.ALL)


_TORCH_DTYPES = {
    "f32": torch.float32,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "i8": torch.int8,
    "u8": torch.uint8,
    "f8e4m3fn": torch.float8_e4m3fn,
    "hif8": torch.uint8,
}

_FP8_HIF8_TLA = {
    "f8e4m3fn": tla.Float8E4M3FN,
    "hif8": tla.HiFloat8,
}

_DST_DTYPES = {
    "f32": ("f32", "f16", "bf16", "i8", "u8", "f8e4m3fn", "hif8"),
    "i32": ("f16", "bf16", "i8", "u8"),
}

_INT_DST_RANGES = {
    torch.int8: (-128, 127),
    torch.uint8: (0, 255),
}


def prepare_quant_scale(x: float) -> float:
    """The fixpipe scale is 1:sign 8:exponent 10:mantissa, so f32 drops 13 bits."""
    t = torch.tensor(x, dtype=torch.float32)
    bits = t.view(torch.uint32)
    bits = bits & 0xFFFFE000
    return float(bits.view(torch.float32).item())


def per_tensor_quant_golden(
    acc: torch.Tensor,
    dst_dtype: torch.dtype,
    quant_scale: float,
    quant_offset: int = 0,
    relu: bool = False,
    simulate_npu: bool = False,
) -> torch.Tensor:
    if simulate_npu:
        scaled = acc.float() * prepare_quant_scale(quant_scale)
    else:
        scaled = acc.float() * torch.tensor(quant_scale, dtype=torch.float32)

    if relu:
        # Normal ReLU: M1 for positive raw data, M2 = 0 for negative raw data.
        # The branch is chosen by the sign of the *accumulator*.
        scaled = torch.where(acc > 0, scaled, torch.zeros_like(scaled))

    if dst_dtype == torch.float32:
        return scaled
    elif dst_dtype in (torch.float16, torch.bfloat16):
        return scaled.to(dst_dtype)
    elif dst_dtype in (torch.int8, torch.uint8):
        lo, hi = _INT_DST_RANGES[dst_dtype]
        quantized = torch.round(torch.clamp(scaled, -256, 255)).to(torch.int16)
        quantized += int(quant_offset)
        return torch.round(torch.clamp(quantized, lo, hi)).to(dst_dtype)
    elif dst_dtype == torch.float8_e4m3fn:
        return scaled.to(torch.float8_e4m3fn)
    raise RuntimeError(f"unsupported quant dst_dtype {dst_dtype}")


def _load_encode_hif8_rna():
    """Load ``encode_hif8_rna`` without mutating ``sys.path`` (CodeCheck)."""
    import importlib.util

    here = Path(__file__).resolve()
    ref_path = None
    for parent in here.parents:
        candidate = (
            parent
            / "python"
            / "tla_dsl"
            / "tests"
            / "end_to_end"
            / "_lowp_reference.py"
        )
        if candidate.is_file():
            ref_path = candidate
            break
    if ref_path is None:
        raise RuntimeError(
            "cannot locate python/tla_dsl/tests/end_to_end/_lowp_reference.py "
            "from this example; keep the repo layout intact for --dtype-c hif8"
        )
    spec = importlib.util.spec_from_file_location("_lowp_reference_hif8", ref_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load HiF8 reference module from {ref_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.encode_hif8_rna


def per_tensor_quant_golden_hif8(
    acc: torch.Tensor,
    quant_scale: float,
    relu: bool = False,
    simulate_npu: bool = False,
) -> torch.Tensor:
    """Host reference for QF322HIF8_PRE (Half to Away / RNA)."""
    encode_hif8_rna = _load_encode_hif8_rna()

    if simulate_npu:
        scaled = acc.float() * prepare_quant_scale(quant_scale)
    else:
        scaled = acc.float() * torch.tensor(quant_scale, dtype=torch.float32)
    if relu:
        scaled = torch.where(acc > 0, scaled, torch.zeros_like(scaled))
    flat = scaled.detach().cpu().reshape(-1).tolist()
    codes = [encode_hif8_rna(float(v), saturate=True) for v in flat]
    return torch.tensor(codes, dtype=torch.uint8).reshape(acc.shape)


def run(args: argparse.Namespace) -> int:
    from helpers import create_tla_tensor, get_block_num, compare, compute_error_metrics

    torch.npu.set_device(args.device)

    acc_dtype = "i32" if args.dtype_ab == "i8" else "f32"
    acc_is_int = acc_dtype == "i32"

    dst_choices = _DST_DTYPES[acc_dtype]
    if args.dtype_c not in dst_choices:
        raise SystemExit(
            f"acc {acc_dtype} supports --dtype-c in {dst_choices}, got {args.dtype_c}"
        )

    dtype_a = _TORCH_DTYPES[args.dtype_ab]
    dtype_b = dtype_a  # the two operands share one dtype
    dtype_c = _TORCH_DTYPES[args.dtype_c]
    layout_a, layout_b = "row", "row"
    layout_c = args.layout_c
    is_hif8_dst = args.dtype_c == "hif8"
    is_f8_dst = args.dtype_c == "f8e4m3fn"

    if args.k > MIXED_TILING.l1_tk:
        raise SystemExit(
            f"k must fit one L1 tile (k<={MIXED_TILING.l1_tk}); got k={args.k}"
        )
    # 1-byte + unit_flag≠0: static RowMajor N / ColumnMajor M must be %32.
    align_extent = args.m if layout_c == "col" else args.n
    if args.dtype_c in ("i8", "u8", "f8e4m3fn", "hif8") and align_extent % 32 != 0:
        raise SystemExit(
            f"L0C→UB 1-byte + unit_flag requires "
            f"{'M' if layout_c == 'col' else 'N'} % 32 == 0; got {align_extent}"
        )

    print(
        f"--- mnk=({args.m},{args.n},{args.k}) "
        f"layout={layout_a}/{layout_b}/{layout_c} "
        f"dtype={args.dtype_ab}/{args.dtype_ab} acc={acc_dtype} dst={args.dtype_c} "
        f"relu={bool(args.relu_enable)} "
        f"quant_scale={args.quant_scale} ---"
    )

    if acc_is_int:  # i8*i8
        a = torch.randint(-8, 8, (args.m, args.k), dtype=dtype_a, device="cpu")
        b = torch.randint(-8, 8, (args.k, args.n), dtype=dtype_b, device="cpu")
        acc_ref = a.to(torch.int32) @ b.to(torch.int32)
    else:
        a = torch.rand(args.m, args.k, dtype=dtype_a, device="cpu") * 10.0 - 5.0
        b = torch.rand(args.k, args.n, dtype=dtype_b, device="cpu") * 10.0 - 5.0
        acc_ref = a.float() @ b.float()

    if is_hif8_dst:
        host_golden = per_tensor_quant_golden_hif8(
            acc_ref,
            args.quant_scale,
            relu=bool(args.relu_enable),
            simulate_npu=False,
        )
        host_cpu = per_tensor_quant_golden_hif8(
            acc_ref,
            args.quant_scale,
            relu=bool(args.relu_enable),
            simulate_npu=True,
        )
    else:
        host_golden = per_tensor_quant_golden(
            acc_ref,
            dtype_c,
            args.quant_scale,
            args.quant_offset,
            relu=bool(args.relu_enable),
            simulate_npu=False,
        )
        host_cpu = per_tensor_quant_golden(
            acc_ref,
            dtype_c,
            args.quant_scale,
            args.quant_offset,
            relu=bool(args.relu_enable),
            simulate_npu=True,
        )

    c = torch.zeros(args.m, args.n, dtype=dtype_c, device="cpu")

    a = (a.contiguous() if layout_a == "row" else a.permute(1, 0).contiguous()).npu()
    b = (b.contiguous() if layout_b == "row" else b.permute(1, 0).contiguous()).npu()
    # A ColumnMajor (m, n) result is stored as an (n, m) buffer, which the
    # transposing fixpipe fills on its way out of L0C.
    trans_c = layout_c == "col"
    if trans_c:
        c = c.permute(1, 0)
    c = c.contiguous().npu()
    a_tensor = create_tla_tensor(a, layout_a)
    b_tensor = create_tla_tensor(b, layout_b)
    c_elem = _FP8_HIF8_TLA.get(args.dtype_c)
    c_tensor = create_tla_tensor(c, layout_c, element_type=c_elem)

    runtime_args = [a_tensor, b_tensor, c_tensor]
    artifact = tla.compile(
        basic_mixed_fixpipe_per_tensor_kernel,
        *runtime_args,
        MIXED_TILING,
        acc_is_int,
        bool(args.relu_enable),
        trans_c,
        args.quant_scale,
        args.quant_offset,
        options="--npu-arch 3510",
    )
    launch_args = [args.quant_scale, args.quant_offset]
    print(f"kernel.o={artifact.kernel_binary_path}")

    block_num = get_block_num(args.block_num, args.device, kind="mix")
    artifact(*runtime_args, *launch_args, block_num=block_num)
    torch.npu.synchronize()

    result = c.detach().cpu()
    if trans_c:
        result = result.transpose(0, 1).contiguous()
    if is_hif8_dst:
        max_delta = (
            (result.to(torch.int32) - host_golden.to(torch.int32)).abs().max().item()
        )
        max_delta_cpu = (
            (result.to(torch.int32) - host_cpu.to(torch.int32)).abs().max().item()
        )
        passed = max_delta <= 1 or max_delta_cpu <= 1
        print(
            f"passed={passed} max_abs_delta={max_delta} "
            f"max_abs_delta_cpu={max_delta_cpu} cache_key={artifact.cache_key}"
        )
        return 0 if passed else 1
    if is_f8_dst:
        res_f = result.to(torch.float32)
        gold_f = host_golden.to(torch.float32)
        cpu_f = host_cpu.to(torch.float32)
        passed1 = compare(res_f, gold_f, args.k)
        metrics = compute_error_metrics(
            res_f,
            cpu_f,
            gold_f,
            mare_threshold=10.0,
            mere_threshold=2.0,
            rmse_threshold=2.0,
        )
        byte_delta = (
            (
                result.view(torch.uint8).to(torch.int32)
                - host_golden.view(torch.uint8).to(torch.int32)
            )
            .abs()
            .max()
            .item()
        )
        passed = passed1 or metrics.passed or byte_delta <= 1
        print(f"passed={passed} byte_delta={byte_delta} cache_key={artifact.cache_key}")
        return 0 if passed else 1
    if dtype_c in (torch.int8, torch.uint8):
        max_delta = (
            (result.to(torch.int32) - host_golden.to(torch.int32)).abs().max().item()
        )
        passed = max_delta <= 1
        print(
            f"passed={passed} max_abs_delta={max_delta} cache_key={artifact.cache_key}"
        )
        return 0 if passed else 1

    passed1 = compare(result, host_golden, args.k)
    metrics = compute_error_metrics(
        result,
        host_cpu,
        host_golden,
        mare_threshold=10.0,
        mere_threshold=2.0,
        rmse_threshold=2.0,
    )
    passed = passed1 or metrics.passed
    print(f"passed={passed} cache_key={artifact.cache_key}")
    if not passed:
        print(
            f"max_diff={(result - host_golden).abs().max().item():.4f} "
            f"mare_ratio={metrics.mare_ratio:.4f}, "
            f"mere_ratio={metrics.mere_ratio:.4f}, "
            f"rmse_ratio={metrics.rmse_ratio:.4f}, "
            f"small_cnt_ratio={metrics.small_cnt_ratio:.4f}"
        )
        print("result", result)
        print("host_golden", host_golden)
    return 0 if passed else 1


def main() -> int:
    torch.manual_seed(0)
    parser = argparse.ArgumentParser(
        "basic_mixed fixpipe per-tensor quant, optionally combined with relu."
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--m", type=int, default=MIXED_TILING.l1_tm)
    parser.add_argument("--n", type=int, default=MIXED_TILING.l1_tn)
    parser.add_argument("--k", type=int, default=MIXED_TILING.l1_tk)
    parser.add_argument("--layout-c", choices=("row", "col"), default="row")
    parser.add_argument("--dtype-ab", choices=("f32", "i8"), default="f32")
    parser.add_argument(
        "--dtype-c",
        choices=("f32", "f16", "bf16", "i8", "u8", "f8e4m3fn", "hif8"),
        default="f16",
        help=f"valid dtype-c: f32->{_DST_DTYPES['f32']} i32->{_DST_DTYPES['i32']}",
    )
    parser.add_argument(
        "--quant-scale",
        type=float,
        default=torch.rand(1).item(),
        help="default uniform(0.0, 1.0)",
    )
    parser.add_argument(
        "--quant-offset",
        type=int,
        default=torch.randint(-5, 5, (1,), dtype=torch.int32).item(),
        help="only take effects when dst i8/u8, default randint[-5, 5)",
    )
    parser.add_argument(
        "--relu-enable", type=int, choices=(0, 1), default=0, help="default 0"
    )
    parser.add_argument("--block-num", type=int, default=-1)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
