# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Basic MMAD with fixpipe per-channel quant: Kernel + Host in one file.

Same double-buffered pipeline as ``basic_matmul_fixpipe_per_tensor.py``; the only
difference is the L0C -> GM transfer, which runs the fixpipe with
``QuantMode.PER_CHANNEL``: one scale *per output channel* is applied to the tile
on the way out, instead of a single scalar for the whole tile.

Unlike the per-tensor scale, the per-channel scale is a tensor. It lives in GM
(one entry per output channel, row-major) and is staged into L1 up front -- the
fixpipe consumes it as its dequant tensor -- before the L0C -> GM copy. This is
the same shape of data flow as
``include/catlass/gemm/block/block_mmad_pingpong_dequant_tla.hpp``'s
``copyL0CToDst(tensorC, tensorL0C, tensorL1Quant)``.

Each per-channel entry travels to the fixpipe as a uint64 built the same way the
per-tensor ``quantBits`` scalar is (see ``copy_l0c_to_gm.hpp``): the f32 scale
bits sit in [31:0], the sign selector for an 8-bit destination in bit 46, and the
9-bit signed offset in [45:37]. There is one such word per output column.

A and B share one dtype -- f32 or i8 -- and the L0C accumulator follows it (f32
for the float MMAD route, i32 for the integer one). The legal
(accumulator -> destination) pairs are the ``PER_CHANNEL`` row of
``catlass/core_api.py``'s fixpipe route table.
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
import torch_npu  # noqa: F401
from helpers import TilingParams

import catlass.tla as tla


@tla.kernel
def basic_mmad_fixpipe_per_channel_kernel(
    gm_a: tla.Tensor,
    gm_b: tla.Tensor,
    gm_c: tla.Tensor,
    gm_quant: tla.Tensor,
    _tiling: tla.Constexpr[TilingParams],
    acc_is_int: tla.Constexpr[bool],
    relu_enable: tla.Constexpr[bool],
) -> None:
    c0 = 0
    c1 = 1

    dtype_a = gm_a.ptr.dtype
    dtype_b = gm_b.ptr.dtype
    quant_dtype = gm_quant.ptr.dtype

    m = gm_a.origin_shape[0]
    n = gm_b.origin_shape[1]
    k = gm_a.origin_shape[1]

    l1a0_data_ready = tla.flag("l1a0_data_ready", tla.arch.MTE2, tla.arch.MTE1)
    l1a1_data_ready = tla.flag("l1a1_data_ready", tla.arch.MTE2, tla.arch.MTE1)
    l1b0_data_ready = tla.flag("l1b0_data_ready", tla.arch.MTE2, tla.arch.MTE1)
    l1b1_data_ready = tla.flag("l1b1_data_ready", tla.arch.MTE2, tla.arch.MTE1)
    l1a0_available = tla.flag("l1a0_available", tla.arch.MTE1, tla.arch.MTE2)
    l1a1_available = tla.flag("l1a1_available", tla.arch.MTE1, tla.arch.MTE2)
    l1b0_available = tla.flag("l1b0_available", tla.arch.MTE1, tla.arch.MTE2)
    l1b1_available = tla.flag("l1b1_available", tla.arch.MTE1, tla.arch.MTE2)
    l0a0_available = tla.flag("l0a0_available", tla.arch.CUBE, tla.arch.MTE1)
    l0a1_available = tla.flag("l0a1_available", tla.arch.CUBE, tla.arch.MTE1)
    l0b0_available = tla.flag("l0b0_available", tla.arch.CUBE, tla.arch.MTE1)
    l0b1_available = tla.flag("l0b1_available", tla.arch.CUBE, tla.arch.MTE1)
    l0_ab_data_ready = tla.flag("l0_ab_data_ready", tla.arch.MTE1, tla.arch.CUBE)
    # The per-channel scale is staged GM -> L1 before the fixpipe consumes it:
    # MTE2 writes the L1 tile, FIX reads it as the dequant tensor.
    quant_data_ready = tla.flag("quant_data_ready", tla.arch.MTE2, tla.arch.FIX)
    quant_available = tla.flag("quant_available", tla.arch.FIX, tla.arch.MTE2)

    l1a0_ptr = tla.allocate(
        _tiling.l1_tm * _tiling.l1_tk, dtype_a, tla.AddressSpace.l1, 512
    )
    l1a1_ptr = tla.allocate(
        _tiling.l1_tm * _tiling.l1_tk, dtype_a, tla.AddressSpace.l1, 512
    )
    l1b0_ptr = tla.allocate(
        _tiling.l1_tk * _tiling.l1_tn, dtype_b, tla.AddressSpace.l1, 512
    )
    l1b1_ptr = tla.allocate(
        _tiling.l1_tk * _tiling.l1_tn, dtype_b, tla.AddressSpace.l1, 512
    )
    # One dequant entry per output channel of the tile.
    l1_quant_ptr = tla.allocate(_tiling.l1_tn, quant_dtype, tla.AddressSpace.l1, 512)

    l0a0_ptr = tla.allocate(
        _tiling.l0_tm * _tiling.l0_tk, dtype_a, tla.AddressSpace.l0a, 512
    )
    l0a1_ptr = tla.allocate(
        _tiling.l0_tm * _tiling.l0_tk, dtype_a, tla.AddressSpace.l0a, 512
    )
    l0b0_ptr = tla.allocate(
        _tiling.l0_tk * _tiling.l0_tn, dtype_b, tla.AddressSpace.l0b, 512
    )
    l0b1_ptr = tla.allocate(
        _tiling.l0_tk * _tiling.l0_tn, dtype_b, tla.AddressSpace.l0b, 512
    )

    # Integer route: an i8 x i8 MMAD accumulates into an i32 L0C, not fp32.
    acc_dtype = tla.Int32 if acc_is_int else tla.Float32
    l0c_ptr = tla.allocate(
        _tiling.l0_tm * _tiling.l0_tn, acc_dtype, tla.AddressSpace.l0c, 1024
    )

    grid_m = (m + _tiling.l1_tm - 1) // _tiling.l1_tm
    grid_n = (n + _tiling.l1_tn - 1) // _tiling.l1_tn
    total_blocks = grid_m * grid_n

    with tla.cube():
        tla.set_flag(l1a0_available)
        tla.set_flag(l1a1_available)
        tla.set_flag(l1b0_available)
        tla.set_flag(l1b1_available)
        tla.set_flag(l0a0_available)
        tla.set_flag(l0a1_available)
        tla.set_flag(l0b0_available)
        tla.set_flag(l0b1_available)
        tla.set_flag(quant_available)

        runtime_zero = tla.as_numeric(0)
        l1_buf_idx = runtime_zero
        l0_buf_idx = runtime_zero

        block_range = tla.range(
            tla.arch.block_idx(), total_blocks, tla.arch.block_num()
        )
        for block_linear in block_range:
            block_row = block_linear // grid_n
            block_col = block_linear % grid_n
            gm_a_by_core = tla.tile_view(
                gm_a, tla.make_shape(_tiling.l1_tm, k), tla.make_coord(block_row, c0)
            )
            gm_b_by_core = tla.tile_view(
                gm_b, tla.make_shape(k, _tiling.l1_tn), tla.make_coord(c0, block_col)
            )
            gm_c_by_core = tla.tile_view(
                gm_c,
                tla.make_shape(_tiling.l1_tm, _tiling.l1_tn),
                tla.make_coord(block_row, block_col),
            )
            # This block's slice of the per-channel scale: one entry per column,
            # addressed as a flat run. VectorLayout is rank-1, so the tile is a
            # single extent starting at this block's first column.
            gm_quant_by_core = tla.tile_view(
                gm_quant,
                tla.make_shape(_tiling.l1_tn),
                tla.make_coord(block_col),
            )

            k_block = gm_a_by_core.origin_shape[1]
            k_l1_count = (k_block + _tiling.l1_tk - 1) // _tiling.l1_tk
            k_l1_range = tla.range(c0, k_l1_count, c1)

            l0_c = tla.make_tensor_like(l0c_ptr, gm_c_by_core)
            # u64 is not a make_tensor_like-able element type, so build the L1
            # quant tile explicitly. VectorLayout is rank-1 with a literal unit
            # stride, which is what the flat GM -> L1 route requires.
            l1_quant = tla.make_tensor(
                l1_quant_ptr,
                tla.make_layout(
                    tla.make_shape(_tiling.l1_tn),
                    tla.make_stride(c1),
                    layoutTag=tla.arch.VectorLayout,
                ),
            )

            for k_l1 in k_l1_range:
                gm_a_by_l1 = tla.tile_view(
                    gm_a_by_core,
                    tla.make_shape(_tiling.l1_tm, _tiling.l1_tk),
                    tla.make_coord(c0, k_l1),
                )
                gm_b_by_l1 = tla.tile_view(
                    gm_b_by_core,
                    tla.make_shape(_tiling.l1_tk, _tiling.l1_tn),
                    tla.make_coord(k_l1, c0),
                )

                l1_a = tla.make_tensor_like(
                    l1a0_ptr if (l1_buf_idx == c0) else l1a1_ptr, gm_a_by_l1
                )
                l1_b = tla.make_tensor_like(
                    l1b0_ptr if (l1_buf_idx == c0) else l1b1_ptr, gm_b_by_l1
                )
                if l1_buf_idx == c0:
                    tla.wait_flag(l1a0_available)
                else:
                    tla.wait_flag(l1a1_available)
                tla.copy(l1_a, gm_a_by_l1)
                if l1_buf_idx == c0:
                    tla.set_flag(l1a0_data_ready)
                else:
                    tla.set_flag(l1a1_data_ready)

                if l1_buf_idx == c0:
                    tla.wait_flag(l1b0_available)
                else:
                    tla.wait_flag(l1b1_available)
                tla.copy(l1_b, gm_b_by_l1)
                if l1_buf_idx == c0:
                    tla.set_flag(l1b0_data_ready)
                else:
                    tla.set_flag(l1b1_data_ready)

                k_l0_count = (l1_a.origin_shape[1] + _tiling.l0_tk - 1) // _tiling.l0_tk
                k_l0_range = tla.range(c0, k_l0_count, c1)

                for k_l0 in k_l0_range:
                    l1_a_by_l0 = tla.tile_view(
                        l1_a,
                        tla.make_shape(_tiling.l0_tm, _tiling.l0_tk),
                        tla.make_coord(c0, k_l0),
                    )
                    l1_b_by_l0 = tla.tile_view(
                        l1_b,
                        tla.make_shape(_tiling.l0_tk, _tiling.l0_tn),
                        tla.make_coord(k_l0, c0),
                    )

                    l0_a = tla.make_tensor_like(
                        l0a0_ptr if (l0_buf_idx == c0) else l0a1_ptr, l1_a_by_l0
                    )
                    l0_b = tla.make_tensor_like(
                        l0b0_ptr if (l0_buf_idx == c0) else l0b1_ptr, l1_b_by_l0
                    )
                    if k_l0 == 0:
                        if l1_buf_idx == c0:
                            tla.wait_flag(l1a0_data_ready)
                        else:
                            tla.wait_flag(l1a1_data_ready)

                    if l0_buf_idx == c0:
                        tla.wait_flag(l0a0_available)
                    else:
                        tla.wait_flag(l0a1_available)
                    tla.copy(l0_a, l1_a_by_l0)
                    if k_l0 == k_l0_count - 1:
                        if l1_buf_idx == c0:
                            tla.set_flag(l1a0_available)
                        else:
                            tla.set_flag(l1a1_available)

                    if k_l0 == 0:
                        if l1_buf_idx == c0:
                            tla.wait_flag(l1b0_data_ready)
                        else:
                            tla.wait_flag(l1b1_data_ready)
                    if l0_buf_idx == c0:
                        tla.wait_flag(l0b0_available)
                    else:
                        tla.wait_flag(l0b1_available)
                    tla.copy(l0_b, l1_b_by_l0)
                    if k_l0 == k_l0_count - 1:
                        if l1_buf_idx == c0:
                            tla.set_flag(l1b0_available)
                        else:
                            tla.set_flag(l1b1_available)

                    tla.set_flag(l0_ab_data_ready)
                    tla.wait_flag(l0_ab_data_ready)

                    # 0b10 marks every MMAD but the last of the tile, 0b11 the
                    # last one: the fixpipe reads the final flag and starts on
                    # its own, with no CUBE -> FIX round trip.
                    unit_flag = (
                        0b11
                        if (k_l1 == k_l1_count - 1) and (k_l0 == k_l0_count - 1)
                        else 0b10
                    )
                    init_c = k_l1 == 0 and k_l0 == 0
                    tla.mmad(l0_c, l0_a, l0_b, init_c=init_c, unit_flag=unit_flag)

                    if l0_buf_idx == c0:
                        tla.set_flag(l0a0_available)
                        tla.set_flag(l0b0_available)
                    else:
                        tla.set_flag(l0a1_available)
                        tla.set_flag(l0b1_available)
                    l0_buf_idx = c1 - l0_buf_idx
                l1_buf_idx = c1 - l1_buf_idx

            # Stage this block's per-channel scale into L1, then let the fixpipe
            # read it as the L0C -> GM dequant tensor.
            tla.wait_flag(quant_available)
            tla.copy(l1_quant, gm_quant_by_core)
            tla.set_flag(quant_data_ready)
            tla.wait_flag(quant_data_ready)

            tla.copy(
                gm_c_by_core,
                l0_c,
                tla.params.CopyL0C2DstParams(
                    unit_flag=0b11,
                    relu_enable=relu_enable,
                    quant_mode=tla.params.QuantMode.PER_CHANNEL,
                    quant_tensor=l1_quant,
                ),
            )
            tla.set_flag(quant_available)

        tla.wait_flag(l1a0_available)
        tla.wait_flag(l1a1_available)
        tla.wait_flag(l1b0_available)
        tla.wait_flag(l1b1_available)
        tla.wait_flag(l0a0_available)
        tla.wait_flag(l0a1_available)
        tla.wait_flag(l0b0_available)
        tla.wait_flag(l0b1_available)
        tla.wait_flag(quant_available)
        tla.pipe_barrier(tla.pipes.ALL)


_TORCH_DTYPES = {
    "f32": torch.float32,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "i32": torch.int32,
    "i8": torch.int8,
    "u8": torch.uint8,
}

_DST_DTYPES = {
    "f32": ("f32", "f16", "bf16", "i8", "u8"),
    "i32": ("f16", "bf16", "i8", "u8"),
}

torch.manual_seed(0)


def prepare_quant_scale(x: torch.Tensor) -> torch.Tensor:
    """quant_scale is not standard f32, 1:sign 8:exponent 10:mantissa, need to
    truncate low 13 bits of f32. Works element-wise on a scale tensor."""
    t = x.to(torch.float32).contiguous()
    bits = t.view(torch.uint32)
    bits = bits & 0xFFFFE000  # truncate low 13 bits
    return bits.view(torch.float32)


def encode_per_channel_quant_tensor(
    scale: torch.Tensor, dst_dtype: torch.dtype, offset: torch.Tensor | None = None
) -> torch.Tensor:
    """Pack one f32 scale per output channel into the uint64 word the fixpipe
    dequant tensor takes, matching ``copy_l0c_to_gm.hpp``'s ``quantBits``:

    - [31:0]  : the f32 scale bits
    - [46]    : sign selector for an 8-bit destination (1: int8, 0: uint8)
    - [45:37] : 9-bit signed offset, for an 8-bit destination only

    ``offset`` is a per-channel tensor (broadcast like ``scale``); fp32/fp16/bf16
    destinations carry no offset and pass ``None``.
    """
    scale = scale.to(torch.float32).contiguous()
    # Pack in int64, then hand back uint64. Every field lands below bit 47, so the
    # word stays positive and int64 carries it without loss -- and torch has no
    # left shift for uint64 (`lshift_cpu` is unimplemented), which the offset
    # field needs. The 9-bit offset is masked before shifting, which is also what
    # turns a negative one into its two's complement field.
    bits = scale.view(torch.uint32).to(torch.int64) & 0xFFFFFFFF
    if dst_dtype == torch.int8:
        bits = bits | (1 << 46)
    if offset is not None and dst_dtype in (torch.int8, torch.uint8):
        offset_i64 = offset.to(torch.int64).reshape(bits.shape) & 0x1FF
        bits = bits | (offset_i64 << 37)
    return bits.to(torch.uint64)


# Integer destinations saturate at their own range.
_INT_DST_RANGES = {
    torch.int8: (-128, 127),
    torch.uint8: (0, 255),
}


def per_channel_quant_golden(
    acc: torch.Tensor,
    dst_dtype: torch.dtype,
    quant_scale: torch.Tensor,
    quant_offset: torch.Tensor | None = None,
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
    elif dst_dtype in (torch.int8, torch.uint8):
        lo, hi = _INT_DST_RANGES[dst_dtype]
        # Fixpipe int8/uint8 order : scale (M1/M2 by acc sign, handled above)
        # -> intermediate saturation to the 16-bit field [-256, 255] with rounding
        # -> add the 9-bit signed offset
        # -> saturate to the destination range (int8 [-128,127] / uint8 [0,255]).
        quantized = torch.round(torch.clamp(scaled, -256, 255)).to(torch.int16)
        if quant_offset is not None:
            quantized = quantized + quant_offset.to(torch.int16).reshape(1, -1)
        return torch.round(torch.clamp(quantized, lo, hi)).to(dst_dtype)
    raise RuntimeError(f"unsupported quant dst_dtype {dst_dtype}")


def run(args: argparse.Namespace) -> int:
    from helpers import compare, compute_error_metrics, create_tla_tensor, get_block_num

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

    # One fp32 scale per output channel, row-major. For an 8-bit destination
    # there is also a per-channel 9-bit signed offset; f32/f16/bf16 carry none.
    quant_scale = torch.rand(1, args.n, dtype=torch.float32, device="cpu") * 2.0 - 1.0
    if dtype_c in (torch.int8, torch.uint8):
        quant_offset = torch.randint(
            -10, 10, (1, args.n), dtype=torch.int64, device="cpu"
        )
    else:
        quant_offset = None

    host_golden = per_channel_quant_golden(
        acc_ref,
        dtype_c,
        quant_scale,
        quant_offset,
        relu=bool(args.relu_enable),
        simulate_npu=False,
    )
    host_cpu = per_channel_quant_golden(
        acc_ref,
        dtype_c,
        quant_scale,
        quant_offset,
        relu=bool(args.relu_enable),
        simulate_npu=True,
    )

    # Pack each column's scale (and offset, when present) into one uint64 -- one
    # word per output channel, as one contiguous run. The flat GM -> L1 route is
    # keyed on VectorLayout, so the buffer is handed over as a rank-1 tile.
    quant_bits = encode_per_channel_quant_tensor(quant_scale, dtype_c, quant_offset)

    c = torch.zeros(args.m, args.n, dtype=dtype_c, device="cpu")

    a = (a.contiguous() if layout_a == "row" else a.permute(1, 0).contiguous()).npu()
    b = (b.contiguous() if layout_b == "row" else b.permute(1, 0).contiguous()).npu()
    # A ColumnMajor (m, n) result is stored as an (n, m) buffer; the fixpipe
    # writes it transposed on its way out of L0C (nz2dn), so the kernel body is
    # unchanged -- the route follows the destination tensor's layout tag. The
    # per-channel scale folds into the same transfer, so the transposed store
    # carries the dequant too.
    if layout_c == "col":
        c = c.permute(1, 0)
    c = c.contiguous().npu()
    quant = quant_bits.reshape(-1).contiguous().npu()
    a_tensor = create_tla_tensor(a, layout_a)
    b_tensor = create_tla_tensor(b, layout_b)
    c_tensor = create_tla_tensor(c, layout_c)
    quant_tensor = create_tla_tensor(quant, "vector", element_type=tla.UInt64)

    artifact = tla.compile(
        basic_mmad_fixpipe_per_channel_kernel,
        a_tensor,
        b_tensor,
        c_tensor,
        quant_tensor,
        TilingParams(
            l1_tm=128, l0_tm=128
        ),  # tiling: L1: (128, 256, 128); L0: (128, 256, 32)
        acc_is_int,
        bool(args.relu_enable),
        options="--npu-arch 3510",
    )
    print(f"kernel.o={artifact.kernel_binary_path}")

    block_num = get_block_num(args.block_num, args.device, kind="cube")
    artifact(
        a_tensor,
        b_tensor,
        c_tensor,
        quant_tensor,
        block_num=block_num,
    )
    torch.npu.synchronize()

    result = c.detach().cpu()
    if layout_c == "col":
        result = result.transpose(0, 1).contiguous()
    if dtype_c in (torch.int8, torch.uint8):
        max_delta = (
            (result.to(torch.int32) - host_golden.to(torch.int32)).abs().max().item()
        )
        passed = max_delta <= 1
        print(
            f"passed={passed} max_abs_delta={max_delta} cache_key={artifact.cache_key}"
        )
        return 0 if passed else 1
    else:
        # Same acceptance as the per-tensor example: the accumulation-aware
        # `compare` against the high-precision golden, or the error-metric ratios
        # against the same-precision CPU result -- whichever passes. The ratios
        # use the per-tensor thresholds, since per-channel dequant accumulates the
        # same way.
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
                f"rmse_ratio={metrics.rmse_ratio:.4f}"
            )
        return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser("fixpipe per-channel quant combined with relu.")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--dtype-ab", choices=("f32", "i8"), default="f32")
    parser.add_argument(
        "--dtype-c",
        choices=("f32", "f16", "bf16", "i8", "u8"),
        default="f16",
        help=f"valid dtype-c: f32->{_DST_DTYPES['f32']} i32->{_DST_DTYPES['i32']}",
    )
    parser.add_argument(
        "--relu-enable", type=int, choices=(0, 1), default=0, help="default 0"
    )
    parser.add_argument("--block-num", type=int, default=-1)
    # `col` sends the L0C -> GM store through the transposing (nz2dn) fixpipe,
    # which then applies the per-channel dequant to the transposed tile.
    parser.add_argument("--layout-c", choices=("row", "col"), default="row")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
