# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Basic MMAD chain with a per-channel quantized L0C -> L1 store.

Same single-block pipeline as ``basic_matmul_l0c2l1_fixpipe_per_tensor.py`` --
two chained MMADs over one L1 tile set, ``C = A @ B`` then ``E = C @ D``, with
the first result leaving L0C through the fixpipe quantized -- except that the
fixpipe runs ``QuantMode.PER_CHANNEL``: one scale *per output channel* instead
of a single scalar for the whole tile.

A and B share one dtype (f32 or i8) and so share the L0C accumulator type (f32
or i32). C and D share the quantized dtype, which is also the dtype of the L1
tile the fixpipe writes and of the L0A operand of the second MMAD. The final E
goes L0C -> GM without any fixpipe parameters, so it keeps the second
accumulator's own dtype.

Quantized L0C -> L1 stores exist for the zN layout only, which is the layout
``make_tensor_like`` tags below.

The per-channel scale is a tensor rather than a runtime scalar: it lives in GM as
one uint64 word per output channel and is staged into L1 up front, because the
fixpipe takes it as its dequant tensor. Each word is packed the same way the
per-tensor ``quantBits`` scalar is (see ``copy_l0c_to_l1.hpp``): the f32 scale
bits sit in [31:0], the sign selector for an 8-bit destination in bit 46, and the
9-bit signed offset in [45:37]. There is one such word per output column of the
tile, so the staged tile is exactly ``N_L1`` words long, while the host builds
the problem's ``N_DIM`` words.

With ``relu_enable`` set, the fixpipe also applies a normal ReLU on the way out:
accumulator values at or below zero contribute zero, positive ones keep their
scaled value. The branch is selected by the sign of the raw accumulator, before
the 8-bit clamp and the offset.

The first MMAD and the quantized store run on the unit flag (``0b11``: the last
MMAD of the tile plus the fixpipe self-start it enables), so the store starts
from the accumulator's own completion signal. The second MMAD carries no unit
flag: its store to GM is ordered by the explicit CUBE -> FIX flag.
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

import catlass.tla as tla

M_DIM = 50
N_DIM = 60
K_DIM = 64

M_L1 = 64
N_L1 = 64
K_L1 = 64

# First mmad: C = A(M,K) @ B(K,N), accumulated in L0C.
L1A_SIZE = M_L1 * K_L1
L1B_SIZE = K_L1 * N_L1
L0A1_SIZE = M_L1 * K_L1
L0B1_SIZE = K_L1 * N_L1

# Quantized L0C -> L1 store target, and the L0A operand of the second mmad.
L1C_SIZE = M_L1 * N_L1

# Second mmad: E = C(M,N) @ D(N,N)
L1D_SIZE = N_L1 * N_L1
L0A2_SIZE = M_L1 * N_L1
L0B2_SIZE = N_L1 * N_L1

# One L0C accumulator per mmad: the first holds the f32/i32 sum of A@B, the
# second the sum of C@D. They differ in dtype, so they cannot share a buffer.
L0C1_SIZE = M_L1 * N_L1
L0C2_SIZE = M_L1 * N_L1

# One u64 dequant word per output channel of the quantized tile.
L1_QUANT_SIZE = N_L1


DESCRIPTION = (
    "Basic MMAD L0C->L1 chain with a per-channel quantized store: "
    "E = quant(A@B) @ D; AB one dtype, CD one dtype, cube-only, optional relu."
)


# ---------------------------------------------------------------------------
# Golden
# ---------------------------------------------------------------------------

# Integer destinations saturate at their own range.
_INT_DST_RANGES = {
    torch.int8: (-128, 127),
    torch.uint8: (0, 255),
}


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
    dequant tensor takes, matching ``copy_l0c_to_l1.hpp``'s ``quantBits``:

    - [31:0]  : the f32 scale bits
    - [46]    : sign selector for an 8-bit destination (1: int8, 0: uint8)
    - [45:37] : 9-bit signed offset, for an 8-bit destination only

    ``offset`` is a per-channel tensor (broadcast like ``scale``); fp16/bf16
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


def per_channel_quant_golden(
    acc: torch.Tensor,
    dst_dtype: torch.dtype,
    quant_scale: torch.Tensor,
    quant_offset: torch.Tensor | None = None,
    relu: bool = False,
) -> torch.Tensor:
    """Mirror the fixpipe order: scale (with the optional relu), 8-bit clamp,
    offset, destination clamp.

    Only the int8 destination runs both clamps; the 16-bit destinations apply
    the scale and let the cast round.
    """
    # ``quant_scale`` is one scale per output channel of the problem, broadcast
    # across rows.
    scale = quant_scale.to(torch.float32).reshape(1, -1)
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


def golden(a, b, d, dst_dtype: torch.dtype, quant_scale, quant_offset, relu=False):
    """C = A@B in the first accumulator's precision, then quantize per channel
    (optionally with relu), then C@D.

    Returns ``(c, e)``: the quantized L1 tile and the GM result, both at the
    problem shape.
    """
    if a.dtype == torch.int8:
        # The integer MMAD accumulates A@B exactly.
        acc = (a.to(torch.int32) @ b.to(torch.int32)).float()
    else:
        acc = a.float() @ b.float()

    c = per_channel_quant_golden(acc, dst_dtype, quant_scale, quant_offset, relu=relu)

    if c.dtype == torch.int8:
        # The second MMAD is an i8 x i8 -> i32 GEMM on the quantized tile.
        return c, c.to(torch.int32) @ d.to(torch.int32)
    return c, c.float() @ d.float()


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


@tla.kernel
def basic_matmul_l0c2l1_fixpipe_per_channel(
    mem_a: tla.Tensor,
    mem_b: tla.Tensor,
    mem_d: tla.Tensor,
    mem_e: tla.Tensor,
    mem_quant: tla.Tensor,
    relu_enable: tla.Constexpr[bool],
) -> None:
    dtype_a = mem_a.ptr.dtype  # f32 or i8; A and B share it
    dtype_c = mem_d.ptr.dtype  # quantized C, and D: f16, bf16 or i8
    dtype_e = mem_e.ptr.dtype  # second accumulator: f32, or i32 on the i8 route
    dtype_quant = mem_quant.ptr.dtype  # the packed dequant words: u64

    src_is_int = dtype_a == tla.Int8
    acc1 = tla.Int32 if src_is_int else tla.Float32

    l1a0_data_ready = tla.flag("l1a0_data_ready", tla.arch.MTE2, tla.arch.MTE1)
    l1b0_data_ready = tla.flag("l1b0_data_ready", tla.arch.MTE2, tla.arch.MTE1)
    l1a0_available = tla.flag("l1a0_available", tla.arch.MTE1, tla.arch.MTE2)
    l1b0_available = tla.flag("l1b0_available", tla.arch.MTE1, tla.arch.MTE2)
    l0a0_available = tla.flag("l0a0_available", tla.arch.CUBE, tla.arch.MTE1)
    l0b0_available = tla.flag("l0b0_available", tla.arch.CUBE, tla.arch.MTE1)
    l0_ab_data_ready = tla.flag("l0_ab_data_ready", tla.arch.MTE1, tla.arch.CUBE)
    l0_loaded = tla.flag("l0_loaded", tla.arch.MTE1, tla.arch.CUBE)
    mmad_done = tla.flag("mmad_done", tla.arch.CUBE, tla.arch.FIX)
    fixl1_done = tla.flag("fixl1_done", tla.arch.FIX, tla.arch.MTE1)
    l1d_done = tla.flag("l1d_done", tla.arch.MTE2, tla.arch.MTE1)
    # The per-channel scale is staged GM -> L1 before the fixpipe consumes it:
    # MTE2 writes the L1 tile, FIX reads it as the dequant tensor.
    quant_ready = tla.flag("quant_ready", tla.arch.MTE2, tla.arch.FIX)

    l1a_ptr = tla.allocate(L1A_SIZE, dtype_a, tla.AddressSpace.l1, 512)
    l1b_ptr = tla.allocate(L1B_SIZE, dtype_a, tla.AddressSpace.l1, 512)
    l1c_ptr = tla.allocate(L1C_SIZE, dtype_c, tla.AddressSpace.l1, 512)
    l1d_ptr = tla.allocate(L1D_SIZE, dtype_c, tla.AddressSpace.l1, 512)
    l1_quant_ptr = tla.allocate(L1_QUANT_SIZE, dtype_quant, tla.AddressSpace.l1, 512)

    # The two mmads need their own L0A/L0B buffers: the operand dtype changes
    # between them whenever A/B are f32 and the quantized C is half or int8.
    l0a1_ptr = tla.allocate(L0A1_SIZE, dtype_a, tla.AddressSpace.l0a, 512)
    l0b1_ptr = tla.allocate(L0B1_SIZE, dtype_a, tla.AddressSpace.l0b, 512)
    l0a2_ptr = tla.allocate(L0A2_SIZE, dtype_c, tla.AddressSpace.l0a, 512)
    l0b2_ptr = tla.allocate(L0B2_SIZE, dtype_c, tla.AddressSpace.l0b, 512)

    l0c1_ptr = tla.allocate(L0C1_SIZE, acc1, tla.AddressSpace.l0c, 1024)
    l0c2_ptr = tla.allocate(L0C2_SIZE, dtype_e, tla.AddressSpace.l0c, 1024)

    with tla.cube():
        # ---- stage the per-channel dequant tensor ----------------------
        # VectorLayout is rank-1 with a literal unit stride, which is what the
        # flat GM -> L1 route for the u64 dequant words requires.
        gm_quant = tla.tile_view(
            mem_quant, tla.make_shape(L1_QUANT_SIZE), tla.make_coord(0)
        )
        l1_quant = tla.make_tensor_like(l1_quant_ptr, gm_quant, tla.arch.VectorLayout)
        tla.copy(l1_quant, gm_quant)
        tla.set_flag(quant_ready)

        # ---- C = A @ B -------------------------------------------------
        gm_a = tla.tile_view(mem_a, tla.make_shape(M_L1, K_L1), tla.make_coord(0, 0))
        gm_b = tla.tile_view(mem_b, tla.make_shape(K_L1, N_L1), tla.make_coord(0, 0))
        gm_e = tla.tile_view(mem_e, tla.make_shape(M_L1, N_L1), tla.make_coord(0, 0))

        tla.set_flag(l1a0_available)
        tla.set_flag(l1b0_available)
        tla.set_flag(l0a0_available)
        tla.set_flag(l0b0_available)

        l1_a = tla.make_tensor_like(l1a_ptr, gm_a)
        l1_b = tla.make_tensor_like(l1b_ptr, gm_b)

        tla.wait_flag(l1a0_available)
        tla.copy(l1_a, gm_a)
        tla.set_flag(l1a0_data_ready)

        tla.wait_flag(l1b0_available)
        tla.copy(l1_b, gm_b)
        tla.set_flag(l1b0_data_ready)

        l1_a_l0 = tla.tile_view(l1_a, tla.make_shape(M_L1, K_L1), tla.make_coord(0, 0))
        l1_b_l0 = tla.tile_view(l1_b, tla.make_shape(K_L1, N_L1), tla.make_coord(0, 0))
        l0a_a = tla.make_tensor_like(l0a1_ptr, l1_a_l0)
        l0b_b = tla.make_tensor_like(l0b1_ptr, l1_b_l0)

        tla.wait_flag(l1a0_data_ready)
        tla.wait_flag(l0a0_available)
        tla.copy(l0a_a, l1_a_l0)
        tla.set_flag(l1a0_available)

        tla.wait_flag(l1b0_data_ready)
        tla.wait_flag(l0b0_available)
        tla.copy(l0b_b, l1_b_l0)
        tla.set_flag(l1b0_available)

        tla.set_flag(l0_ab_data_ready)
        tla.wait_flag(l0_ab_data_ready)

        l0c_c = tla.make_tensor_like(l0c1_ptr, gm_e)
        tla.mmad(l0c_c, l0a_a, l0b_b, init_c=True, unit_flag=0b11)
        tla.set_flag(l0a0_available)
        tla.set_flag(l0b0_available)

        # ---- C -> L1, quantized per channel ----------------------------
        # L0Clayout -> zN is the only quantized L0C -> L1 route: the fixpipe
        # applies this column's scale (and, for int8, its offset) on the way
        # out, reading both from the packed word staged above.
        tla.wait_flag(quant_ready)
        l1_c = tla.make_tensor_like(l1c_ptr, l0c_c, tla.arch.zN)
        tla.copy(
            l1_c,
            l0c_c,
            tla.params.CopyL0C2DstParams(
                unit_flag=0b11,
                relu_enable=relu_enable,
                quant_mode=tla.params.QuantMode.PER_CHANNEL,
                quant_tensor=l1_quant,
            ),
        )
        tla.set_flag(fixl1_done)
        tla.wait_flag(fixl1_done)

        # ---- E = C @ D -------------------------------------------------
        gm_d = tla.tile_view(mem_d, tla.make_shape(N_L1, N_L1), tla.make_coord(0, 0))
        l1_d = tla.make_tensor_like(l1d_ptr, gm_d, tla.arch.zN)
        tla.copy(l1_d, gm_d)
        tla.set_flag(l1d_done)
        tla.wait_flag(l1d_done)

        l1_c_l0 = tla.tile_view(l1_c, tla.make_shape(M_L1, N_L1), tla.make_coord(0, 0))
        l1_d_l0 = tla.tile_view(l1_d, tla.make_shape(N_L1, N_L1), tla.make_coord(0, 0))
        l0a_c = tla.make_tensor_like(l0a2_ptr, l1_c_l0, tla.arch.zN)
        l0b_d = tla.make_tensor_like(l0b2_ptr, l1_d_l0, tla.arch.nZ)
        tla.copy(l0a_c, l1_c_l0)
        tla.copy(l0b_d, l1_d_l0)
        tla.set_flag(l0_loaded)
        tla.wait_flag(l0_loaded)

        l0c_e = tla.make_tensor_like(
            l0c2_ptr, l0c_c, tla.arch.L0Clayout
        )  # the two mmads produce the same tile shape
        # No unit_flag: the CUBE -> MTE3 dependency for the GM store is the
        # explicit mmad_done flag, so nothing consumes the accumulate-last hint.
        tla.mmad(l0c_e, l0a_c, l0b_d, init_c=True)
        tla.set_flag(mmad_done)
        tla.wait_flag(mmad_done)

        # No fixpipe parameters: E keeps the second accumulator's dtype.
        tla.copy(gm_e, l0c_e)

        tla.wait_flag(l1a0_available)
        tla.wait_flag(l1b0_available)
        tla.wait_flag(l0a0_available)
        tla.wait_flag(l0b0_available)
        tla.pipe_barrier(tla.pipes.ALL)


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------

_TORCH_DTYPES = {
    "f32": torch.float32,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "i8": torch.int8,
}


def build_inputs(args):
    src_dtype = _TORCH_DTYPES[args.src_dtype]
    quant_dtype = _TORCH_DTYPES[args.quant_dtype]
    e_dtype = torch.int32 if quant_dtype == torch.int8 else torch.float32

    if src_dtype == torch.int8:
        # Small magnitudes keep the quantized tile out of int8 saturation and the
        # second i8 x i8 GEMM well inside i32.
        a = torch.randint(-4, 5, (M_DIM, K_DIM), dtype=torch.int8, device="cpu")
        b = torch.randint(-4, 5, (K_DIM, N_DIM), dtype=torch.int8, device="cpu")
    else:
        a = torch.rand((M_DIM, K_DIM), dtype=torch.float32, device="cpu") * 10.0 - 5.0
        b = torch.rand((K_DIM, N_DIM), dtype=torch.float32, device="cpu") * 10.0 - 5.0

    if quant_dtype == torch.int8:
        d = torch.randint(-8, 9, (N_DIM, N_DIM), dtype=torch.int8, device="cpu")
    else:
        d = (
            torch.rand((N_DIM, N_DIM), dtype=torch.float32, device="cpu") * 10.0 - 5.0
        ).to(quant_dtype)

    e = torch.zeros((M_DIM, N_DIM), dtype=e_dtype, device="cpu")
    return a, b, d, e


def build_quant(args):
    scale = torch.rand(1, N_DIM, dtype=torch.float32, device="cpu") * 2.0 - 1.0
    scale = prepare_quant_scale(scale).contiguous()

    if args.quant_dtype != "i8":
        offset = None
    else:
        # The field is a 9-bit signed value, so -256..255 is the meaningful
        # range; keep the draws small enough to stay well inside it.
        offset = torch.randint(-10, 10, (1, N_DIM), dtype=torch.int64, device="cpu")
    return scale, offset


def run(args: argparse.Namespace) -> int:
    import torch_npu

    from helpers import (
        create_tla_tensor,
        compare,
    )

    torch.npu.set_device(args.device)
    torch.manual_seed(0)

    a, b, d, e = build_inputs(args)
    scale, offset = build_quant(args)
    dst_dtype = _TORCH_DTYPES[args.quant_dtype]
    _, e_ref = golden(a, b, d, dst_dtype, scale, offset, relu=bool(args.relu_enable))
    ref = e_ref

    quant_bits = encode_per_channel_quant_tensor(scale, dst_dtype, offset)

    a_n, b_n, d_n, e_n = (t.contiguous().npu() for t in (a, b, d, e))
    quant_n = quant_bits.reshape(-1).contiguous().npu()
    a_t, b_t, d_t, e_t = (create_tla_tensor(t, "row") for t in (a_n, b_n, d_n, e_n))
    # The flat GM -> L1 route is keyed on VectorLayout, so the buffer is handed
    # over as a rank-1 tile of packed uint64 words.
    quant_t = create_tla_tensor(quant_n, "vector", element_type=tla.UInt64)

    artifact = tla.compile(
        basic_matmul_l0c2l1_fixpipe_per_channel,
        a_t,
        b_t,
        d_t,
        e_t,
        quant_t,
        bool(args.relu_enable),
        options="--npu-arch 3510",
    )
    artifact(
        a_t,
        b_t,
        d_t,
        e_t,
        quant_t,
        block_num=args.block_num,
    )
    torch.npu.synchronize()

    got = e_n.detach().cpu()
    if got.dtype == torch.int32:
        # Both operands of the second mmad are integers, so that GEMM is exact:
        # any delta comes from the quantized C tile, one int8 step for every
        # element whose rounding went the other way. The hardware rounds the
        # scaled accumulator the way torch.round does (the 0.5-tie case was
        # checked on silicon), but on the --src-dtype f32 route the first MMAD
        # accumulates in f32 in a different order than torch's matmul, so a
        # single element can land on the other side of a tie. Quantized int
        # outputs therefore allow one LSB.
        delta = (got - ref).abs()
        max_delta = int(delta.max().item())
        passed = max_delta <= 1
        detail = (
            f"max_abs_delta={max_delta} "
            f"mismatched={int((delta != 0).sum())}/{delta.numel()}"
        )
    else:
        passed = compare(got, ref, max(K_DIM, N_DIM))
        detail = ""

    if offset is None:
        offset_desc = "none"
    else:
        offset_desc = f"per-channel([{int(offset.min())}, {int(offset.max())}])"
    print(
        f"src_dtype={args.src_dtype} quant_dtype={args.quant_dtype} "
        f"scale=[{scale.min().item():.6f}, {scale.max().item():.6f}] "
        f"offset={offset_desc} relu={bool(args.relu_enable)}"
    )
    print(f"passed={passed} {detail} cache_key={artifact.cache_key}")
    print(f"kernel.o={artifact.kernel_binary_path}")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--block-num", type=int, default=1)
    parser.add_argument("--src-dtype", choices=("f32", "i8"), default="f32")
    parser.add_argument("--quant-dtype", choices=("f16", "bf16", "i8"), default="f16")
    parser.add_argument(
        "--relu-enable", type=int, choices=(0, 1), default=0, help="default 0"
    )
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
