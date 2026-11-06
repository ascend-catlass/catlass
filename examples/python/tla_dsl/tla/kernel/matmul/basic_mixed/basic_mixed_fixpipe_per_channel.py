# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Basic Mixed with fixpipe per-channel quant: Kernel + Host in one file.

The mixed counterpart of ``basic_mmad/basic_matmul_fixpipe_per_channel.py``: the
same dequant tensor, plus what the mixed pipeline demands -- a ``TilingParams``
block, a ``trans_c`` destination switch and ``block_num`` taken from the cube
core count.

Where per-tensor applies one scalar to the whole tile, per-channel applies one
scale *per output column*. The scale is a tensor, not a kernel argument: it is
packed into one uint64 word per channel and staged GM -> L1, where the fixpipe
reads it as its dequant tensor. That staging is per task block, because each
block covers a different run of columns.

The destination is float only -- f32 / f16 / bf16.

``--layout-c col`` stores the result ColumnMajor: the L0C transfer then runs
through the transposing (nz2dn) fixpipe, so the tile is written transposed and
dequantized in one pass. The dequant is applied to the tile before the
transpose, so the scale still runs along N either way.

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
# fixpipe, and nothing below a tile is split.
MIXED_TILING = TilingParams(
    l1_tm=128, l1_tn=128, l1_tk=64, l0_tm=128, l0_tn=128, l0_tk=64
)


@tla.kernel
def basic_mixed_fixpipe_per_channel_kernel(
    mem_a: tla.Tensor,
    mem_b: tla.Tensor,
    mem_c: tla.Tensor,
    mem_quant: tla.Tensor,
    _tiling: tla.Constexpr[TilingParams],
    acc_is_int: tla.Constexpr[bool],
    relu_enable: tla.Constexpr[bool],
    trans_c: tla.Constexpr[bool],
):
    c0 = 0
    c1 = 1

    dtype_a = mem_a.ptr.dtype
    dtype_b = mem_b.ptr.dtype
    dtype_c = mem_c.ptr.dtype
    quant_dtype = mem_quant.ptr.dtype
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
    # The dequant tile is one run per block, so MTE2 rewrites it every iteration
    # and FIX reads it: same full/free pair as any staged buffer.
    quant_full = tla.flag("quant_full", tla.arch.MTE2, tla.arch.FIX)
    quant_free = tla.flag("quant_free", tla.arch.FIX, tla.arch.MTE2)

    # One flag pair for both AIVs. mode 4 keeps a separate counter per AIV, so
    # ``aiv_id`` on each crossing picks the channel -- a shared counter (mode 2)
    # could not express "vec0 has drained its UB but vec1 has not".
    fix_done = tla.cross_flag("fix_done", mode=4)
    fix_ready = tla.cross_flag("fix_ready", mode=4)

    l1a_ptr = tla.allocate(tm * tk, dtype_a, tla.AddressSpace.l1, 512)
    l1b_ptr = tla.allocate(tk * tn, dtype_b, tla.AddressSpace.l1, 512)
    l1_quant_ptr = tla.allocate(tn, quant_dtype, tla.AddressSpace.l1, 512)
    l0a_ptr = tla.allocate(l0m * l0k, dtype_a, tla.AddressSpace.l0a, 512)
    l0b_ptr = tla.allocate(l0k * l0n, dtype_b, tla.AddressSpace.l0b, 512)
    l0c_ptr = tla.allocate(l0m * l0n, acc_dtype, tla.AddressSpace.l0c, 512)
    # One allocation serves both AIVs: each has its own UB address space, so the
    # same offset resolves to a private tile on either side.
    ub_ptr = tla.allocate(tm * tn, dtype_c, tla.AddressSpace.ub, 256)

    c_tag = tla.arch.ColumnMajor if trans_c else tla.arch.RowMajor

    with tla.cube():
        tla.set_flag(l1_free)
        tla.set_flag(l0_free)
        tla.set_flag(l0c_available)
        tla.set_flag(quant_free)

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
            # This block's run of the scale: one word per column, cropped at the
            # tensor's last column. VectorLayout is rank-1, so it is one extent.
            gm_quant = tla.tile_view(
                mem_quant, tla.make_shape(tn), tla.make_coord(block_col)
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

            # u64 has no make_tensor_like route, so the L1 dequant tile is built
            # explicitly: rank-1 VectorLayout with a literal unit stride.
            l1_quant = tla.make_tensor(
                l1_quant_ptr,
                tla.make_layout(
                    tla.make_shape(gm_c.origin_shape[1]),
                    tla.make_stride(c1),
                    layoutTag=tla.arch.VectorLayout,
                ),
            )
            tla.wait_flag(quant_free)
            tla.copy(l1_quant, gm_quant)
            tla.set_flag(quant_full)
            tla.wait_flag(quant_full)

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
            if block_linear % 2 == 0:
                tla.cross_core_wait_flag(fix_ready, tla.arch.FIX, aiv_id=0)
                tla.copy(
                    ub_c,
                    l0_c,
                    params=CopyL0C2DstParams(
                        unit_flag=0b11,
                        l0c2ub_mode=L0C2UBMode.NO_SPLIT_VEC_0,
                        relu_enable=relu_enable,
                        quant_mode=QuantMode.PER_CHANNEL,
                        quant_tensor=l1_quant,
                    ),
                )
                tla.set_flag(l0c_available)
                tla.set_flag(quant_free)
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
                        quant_mode=QuantMode.PER_CHANNEL,
                        quant_tensor=l1_quant,
                    ),
                )
                tla.set_flag(l0c_available)
                tla.set_flag(quant_free)
                tla.cross_core_set_flag(fix_done, tla.arch.FIX, aiv_id=1)

        # One credit per channel, standing in for an already-spent "UB is free".
        tla.cross_core_wait_flag(fix_ready, tla.arch.FIX, aiv_id=0)
        tla.cross_core_wait_flag(fix_ready, tla.arch.FIX, aiv_id=1)
        tla.wait_flag(l1_free)
        tla.wait_flag(l0_free)
        tla.wait_flag(l0c_available)
        tla.wait_flag(quant_free)
        tla.pipe_barrier(tla.pipes.ALL)

    with tla.vector():
        vec_idx = tla.arch.sub_block_idx()

        # A cross flag carries one direction only, so each AIV publishes its own
        # UB as free once before the loop, on its own channel.
        if vec_idx == c0:
            tla.cross_core_set_flag(fix_ready, tla.arch.MTE2, aiv_id=0)
        else:
            tla.cross_core_set_flag(fix_ready, tla.arch.MTE2, aiv_id=1)

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
            if vec_idx == c0:
                if block_linear % 2 == c0:
                    tla.cross_core_wait_flag(fix_done, tla.arch.MTE3, aiv_id=0)
                    tla.copy(gm_out, ub_src)
                    tla.cross_core_set_flag(fix_ready, tla.arch.MTE2, aiv_id=0)
            else:
                if block_linear % 2 == 1:
                    tla.cross_core_wait_flag(fix_done, tla.arch.MTE3, aiv_id=1)
                    tla.copy(gm_out, ub_src)
                    tla.cross_core_set_flag(fix_ready, tla.arch.MTE2, aiv_id=1)

        tla.pipe_barrier(tla.pipes.ALL)


_TORCH_DTYPES = {
    "f32": torch.float32,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "i8": torch.int8,
}

_DST_DTYPES = {
    "f32": ("f32", "f16", "bf16"),
    "i32": ("f16", "bf16"),
}


def prepare_quant_scale(scale: torch.Tensor) -> torch.Tensor:
    """The fixpipe scale is 1:sign 8:exponent 10:mantissa, so f32 drops 13 bits."""
    bits = scale.to(torch.float32).contiguous().view(torch.uint32)
    return (bits & 0xFFFFE000).view(torch.float32)


def encode_per_channel_quant_tensor(scale: torch.Tensor) -> torch.Tensor:
    """Pack one f32 scale per output channel into the uint64 word the fixpipe
    dequant tensor takes, matching ``copy_l0c_to_gm.hpp``'s ``quantBits``: the
    scale bits sit in [31:0].

    Packed through int64 because torch has no left shift for uint64; the word
    stays positive, so the two carry the same value.
    """
    bits = scale.to(torch.float32).contiguous().view(torch.uint32).to(torch.int64)
    return bits.to(torch.uint64)


def per_channel_quant_golden(
    acc: torch.Tensor,
    dst_dtype: torch.dtype,
    quant_scale: torch.Tensor,
    relu: bool = False,
    simulate_npu: bool = False,
) -> torch.Tensor:
    # ``quant_scale`` is one scale per output channel, broadcast across rows.
    scale = quant_scale.to(torch.float32).reshape(1, -1)
    if simulate_npu:
        scaled = acc.float() * prepare_quant_scale(scale)
    else:
        scaled = acc.float() * scale

    if relu:
        # Normal ReLU: M1 for positive raw data, M2 = 0 for negative raw data.
        # The branch is chosen by the sign of the *accumulator*.
        scaled = torch.where(acc > 0, scaled, torch.zeros_like(scaled))

    if dst_dtype == torch.float32:
        return scaled
    elif dst_dtype in (torch.float16, torch.bfloat16):
        return scaled.to(dst_dtype)
    raise RuntimeError(f"unsupported quant dst_dtype {dst_dtype}")


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

    if args.k > MIXED_TILING.l1_tk:
        raise SystemExit(
            f"k must fit one L1 tile (k<={MIXED_TILING.l1_tk}); got k={args.k}"
        )

    print(
        f"--- mnk=({args.m},{args.n},{args.k}) "
        f"layout={layout_a}/{layout_b}/{layout_c} "
        f"dtype={args.dtype_ab}/{args.dtype_ab} acc={acc_dtype} dst={args.dtype_c} "
        f"relu={bool(args.relu_enable)} ---"
    )

    if acc_is_int:  # i8*i8
        a = torch.randint(-8, 8, (args.m, args.k), dtype=dtype_a, device="cpu")
        b = torch.randint(-8, 8, (args.k, args.n), dtype=dtype_b, device="cpu")
        acc_ref = a.to(torch.int32) @ b.to(torch.int32)
    else:
        a = torch.rand(args.m, args.k, dtype=dtype_a, device="cpu") * 10.0 - 5.0
        b = torch.rand(args.k, args.n, dtype=dtype_b, device="cpu") * 10.0 - 5.0
        acc_ref = a.float() @ b.float()

    # One fp32 scale per output channel, row-major; packed into the uint64 word
    # the fixpipe dequant tensor takes and handed over as a flat VectorLayout run.
    quant_scale = torch.rand(1, args.n, dtype=torch.float32, device="cpu") * 2.0 - 1.0

    host_golden = per_channel_quant_golden(
        acc_ref,
        dtype_c,
        quant_scale,
        relu=bool(args.relu_enable),
        simulate_npu=False,
    )
    host_cpu = per_channel_quant_golden(
        acc_ref,
        dtype_c,
        quant_scale,
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
    # The dequant runs along N before the transpose, so the scale is not
    # transposed with the destination.
    quant = encode_per_channel_quant_tensor(quant_scale).reshape(-1).contiguous().npu()
    a_tensor = create_tla_tensor(a, layout_a)
    b_tensor = create_tla_tensor(b, layout_b)
    c_tensor = create_tla_tensor(c, layout_c)
    quant_tensor = create_tla_tensor(quant, "vector", element_type=tla.UInt64)

    runtime_args = [a_tensor, b_tensor, c_tensor, quant_tensor]
    artifact = tla.compile(
        basic_mixed_fixpipe_per_channel_kernel,
        *runtime_args,
        MIXED_TILING,
        acc_is_int,
        bool(args.relu_enable),
        trans_c,
        options="--npu-arch 3510",
    )
    print(f"kernel.o={artifact.kernel_binary_path}")

    block_num = get_block_num(args.block_num, args.device, kind="mix")
    artifact(*runtime_args, block_num=block_num)
    torch.npu.synchronize()

    result = c.detach().cpu()
    if trans_c:
        result = result.transpose(0, 1).contiguous()
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
        "basic_mixed fixpipe per-channel quant, optionally combined with relu."
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--m", type=int, default=MIXED_TILING.l1_tm)
    parser.add_argument("--n", type=int, default=MIXED_TILING.l1_tn)
    parser.add_argument("--k", type=int, default=MIXED_TILING.l1_tk)
    parser.add_argument("--layout-c", choices=("row", "col"), default="row")
    parser.add_argument("--dtype-ab", choices=("f32", "i8"), default="f32")
    parser.add_argument(
        "--dtype-c",
        choices=("f32", "f16", "bf16"),
        default="f16",
        help=f"valid dtype-c: f32->{_DST_DTYPES['f32']} i32->{_DST_DTYPES['i32']}",
    )
    parser.add_argument(
        "--relu-enable", type=int, choices=(0, 1), default=0, help="default 0"
    )
    parser.add_argument("--block-num", type=int, default=-1)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
