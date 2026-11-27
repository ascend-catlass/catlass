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

from pathlib import Path
from typing import Any

import catlass.tla as tla

from vector_op_harness import (
    DirectVectorOpConfig,
    DirectVectorOpHarness,
    vector_kernel_config,
)

# Multi-op masked vector kernel. A single fixed kernel runs add/sub/mul/div in one
# vector loop. Each masked op result is laid over the kept lanes, so lane i holds
# `op(a, b)` on kept lanes and `0` elsewhere (the masked binary intrinsics zero
# the inactive lanes), verifiable without preloading the output buffer.
#
# The UB buffers are allocated tight (32B aligned, no slack), so the partial last
# chunk must not be written out of bounds. This is handled *inside* the loop by a
# loop-carried tail counter: `remaining` starts at VECTOR_ELE and each iteration
# `tail, remaining = tla.update_mask(remaining, dtype=...)` (lowered to
# ave.hir.plt) yields a mask whose lane j is active iff j < remaining — all-true
# for full chunks, partial for the last — and decrements `remaining` by the lane
# count (256B / dtype size). Every store is bounded by this `tail` mask.
#
# Each op exercises a masked COMPUTE (so masked-out lanes hold 0), using a
# different mask source; the store is bounded by `tail` for the partial last
# chunk:
#   add: pattern H (first half)
#   sub: pattern Q (first quarter)
#   mul: pattern M4 (multiples of 4)
#   div: pattern H (first half)
# (lane == element index within a VL-wide chunk).
#
# A fifth output exercises `tla.where` (lowered to ave.hir.vsel): it selects,
# per lane, between the *unmasked* add and sub results using a pattern-H mask, so
# lane i holds (a+b) on the first half of each chunk and (a-b) on the second
# half. Every lane is defined (no zeroing), only the tail store bound applies.

VECTOR_ELE = 400
VL_ELE = 64
LOOPS = (VECTOR_ELE + VL_ELE - 1) // VL_ELE
ALL_DTYPES = ("i16", "i32", "f16", "f32")

_KERNEL_DTYPE = tla.Float32
_KERNEL_ELEMENT_BYTES = 4
_KERNEL_SHAPE = (VECTOR_ELE,)


@tla.kernel
def masked_binary(
    mem_a: tla.Tensor,
    mem_b: tla.Tensor,
    mem_radd: tla.Tensor,
    mem_rsub: tla.Tensor,
    mem_rmul: tla.Tensor,
    mem_rdiv: tla.Tensor,
    mem_rsel: tla.Tensor,
) -> None:
    ub_loaded = tla.flag("ub_loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    vec_done = tla.flag("vec_done", tla.pipes.VECTOR, tla.pipes.MTE3)

    a_gm = tla.tile_view(mem_a, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    b_gm = tla.tile_view(mem_b, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    radd_gm = tla.tile_view(mem_radd, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    rsub_gm = tla.tile_view(mem_rsub, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    rmul_gm = tla.tile_view(mem_rmul, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    rdiv_gm = tla.tile_view(mem_rdiv, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    rsel_gm = tla.tile_view(mem_rsel, tla.make_shape(VECTOR_ELE), tla.make_coord(0))

    a_ub = _make_ub_tensor(a_gm)
    b_ub = _make_ub_tensor(b_gm)
    radd_ub = _make_ub_tensor(radd_gm)
    rsub_ub = _make_ub_tensor(rsub_gm)
    rmul_ub = _make_ub_tensor(rmul_gm)
    rdiv_ub = _make_ub_tensor(rdiv_gm)
    rsel_ub = _make_ub_tensor(rsel_gm)

    with tla.vector():
        tla.copy(a_ub, a_gm)
        tla.copy(b_ub, b_gm)

        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)
        with tla.vec.func(mode="simd"):
            # Single loop over all chunks. The tail is handled *inside* the loop by a
            # loop-carried counter `remaining` (seeded with VECTOR_ELE): each
            # iteration `tla.update_mask` yields a `tail` mask (lane j active iff
            # j < remaining) and decrements `remaining` by the lane count. This is
            # all-true for fully in-bounds chunks and partial for the last one, so the
            # tight 32B-aligned buffers are never written out of bounds. Every store
            # is bounded by `tail`.
            remaining = VECTOR_ELE
            for i in tla.range(LOOPS):
                a_t = _chunk(a_ub, i)
                b_t = _chunk(b_ub, i)
                radd_t = _chunk(radd_ub, i)
                rsub_t = _chunk(rsub_ub, i)
                rmul_t = _chunk(rmul_ub, i)
                rdiv_t = _chunk(rdiv_ub, i)
                rsel_t = _chunk(rsel_ub, i)

                av = a_t.load()
                bv = b_t.load()

                tail, remaining = tla.update_mask(
                    remaining, dtype=_KERNEL_DTYPE
                )  # tail bound
                m_add = tla.create_mask(
                    pattern=tla.mask.H, dtype=_KERNEL_DTYPE
                )  # first half
                m_sub = tla.create_mask(
                    pattern=tla.mask.Q, dtype=_KERNEL_DTYPE
                )  # first quarter
                m_mul = tla.create_mask(
                    pattern=tla.mask.M4, dtype=_KERNEL_DTYPE
                )  # multiples of 4
                m_div = tla.create_mask(
                    pattern=tla.mask.H, dtype=_KERNEL_DTYPE
                )  # first half
                m_sel = tla.create_mask(
                    pattern=tla.mask.H, dtype=_KERNEL_DTYPE
                )  # first half

                radd_t.store(tla.add(av, bv, mask=m_add), mask=tail)
                rsub_t.store(tla.sub(av, bv, mask=m_sub), mask=tail)
                rmul_t.store(tla.mul(av, bv, mask=m_mul), mask=tail)
                rdiv_t.store(tla.div(av, bv, mask=m_div), mask=tail)

                # Select between the unmasked add/sub results: first half of each
                # chunk takes a+b, second half takes a-b (lowers to ave.hir.vsel).
                sum_v = tla.add(av, bv)
                diff_v = tla.sub(av, bv)
                rsel_t.store(tla.where(m_sel, sum_v, diff_v), mask=tail)

        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)

        tla.copy(radd_gm, radd_ub)
        tla.copy(rsub_gm, rsub_ub)
        tla.copy(rmul_gm, rmul_ub)
        tla.copy(rdiv_gm, rdiv_ub)
        tla.copy(rsel_gm, rsel_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def _make_ub_tensor(like_tensor: Any) -> Any:
    # Tight 32-byte alignment (one UB block): no per-buffer slack, so the partial
    # last chunk must be tail-masked (see kernel). 512/256 alignment is no longer
    # used here.
    ptr = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 32)
    return tla.make_tensor_like(ptr, like_tensor, tla.arch.RowMajor)


def _chunk(tensor: Any, chunk_idx: Any) -> Any:
    return tla.tile_view(tensor, tla.make_shape(VL_ELE), tla.make_coord(chunk_idx))


def _operator_specs() -> dict[str, dict[str, Any]]:
    return {
        "masked_binary": {
            "default_atol": 1e-3,
        },
    }


def _is_unsupported_case(op_name: str, dtype_name: str) -> bool:
    del op_name, dtype_name
    return False


def _print_skip(op_name: str, dtype_name: str, shape: tuple[int, ...]) -> None:
    del shape
    print(f"skip op={op_name} dtype={dtype_name}: unsupported case")


def _set_kernel_config(
    op_name: str, dtype_name: str, shape: tuple[int, ...] | None = None
) -> tuple[type[Any], Any, float | int]:
    global VL_ELE, LOOPS, VECTOR_ELE, _KERNEL_DTYPE, _KERNEL_ELEMENT_BYTES
    global _KERNEL_SHAPE
    if op_name not in _operator_specs():
        raise SystemExit("unknown masked-binary operator")
    config = vector_kernel_config(dtype_name, shape, ALL_DTYPES)
    VECTOR_ELE = config.vector_elements
    _KERNEL_SHAPE = shape if shape is not None else (VECTOR_ELE,)
    VL_ELE = config.lanes
    LOOPS = config.loops
    _KERNEL_DTYPE = config.tla_dtype
    _KERNEL_ELEMENT_BYTES = config.element_bytes
    return config.tla_dtype, config.torch_dtype, config.default_sentinel


def _make_inputs(args: Any, dtype_name: str, torch: Any) -> tuple[Any, ...]:
    _, dtype, _ = _set_kernel_config(args.op, dtype_name, args.shape)
    device = "npu"
    if dtype_name in {"i16", "i32"}:
        arange = torch.arange(VECTOR_ELE, dtype=torch.int32, device=device)
        a = ((arange % 23) - 7).to(dtype)
        b = ((arange % 5) + 1).to(dtype)  # nonzero divisor
        return a, b
    idx = torch.arange(VECTOR_ELE, dtype=torch.float32, device=device)
    a = ((idx % 13.0) * 0.5 + 1.0).to(dtype)
    b = ((idx % 6.0) * 0.25 + 1.0).to(dtype)  # nonzero divisor
    return a, b


def _merge(torch: Any, res: Any, background: Any, keep: Any) -> Any:
    return torch.where(keep, res, background)


def _expected(op_name: str, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
    del op_name
    import torch

    a, b = inputs
    add_r = a + b
    sub_r = a - b
    mul_r = a * b
    if a.is_floating_point():
        div_r = a / b
    else:
        div_r = a.div(b, rounding_mode="trunc")

    # Per-op keep masks matching the kernel's masks (lane = idx % VL_ELE).
    lane = torch.arange(VECTOR_ELE, device=a.device) % VL_ELE
    keep_add = lane < (VL_ELE // 2)  # pattern H (first half)
    keep_sub = lane < (VL_ELE // 4)  # pattern Q (first quarter)
    keep_mul = (lane % 4) == 0  # pattern M4 (multiples of 4)
    keep_div = lane < (VL_ELE // 2)  # pattern H (first half)
    keep_sel = lane < (VL_ELE // 2)  # pattern H (first half -> add, else sub)

    # Masked-out lanes are zeroed by the masked binary intrinsics (see kernel).
    zero = torch.zeros_like(a)
    # The select output has every lane defined: add_r where the H mask is active,
    # sub_r elsewhere (no zeroing) -- matches tla.where / ave.hir.vsel.
    sel_r = _merge(torch, add_r, sub_r, keep_sel)
    return (
        _merge(torch, add_r, zero, keep_add),
        _merge(torch, sub_r, zero, keep_sub),
        _merge(torch, mul_r, zero, keep_mul),
        _merge(torch, div_r, zero, keep_div),
        sel_r,
    )


HARNESS = DirectVectorOpHarness(
    DirectVectorOpConfig(
        description="Compile and run a multi-op masked vector binary kernel.",
        kernel=masked_binary,
        all_dtypes=ALL_DTYPES,
        operator_specs=_operator_specs,
        set_kernel_config=_set_kernel_config,
        get_vector_elements=lambda: VECTOR_ELE,
        get_kernel_shape=lambda: _KERNEL_SHAPE,
        make_inputs=_make_inputs,
        expected=_expected,
        unsupported_case=_is_unsupported_case,
        print_skip=_print_skip,
        script_path=Path(__file__).resolve(),
        float_dtypes=frozenset({"f32", "f16"}),
        output_count=5,
    )
)


def main() -> int:
    return HARNESS.main()


if __name__ == "__main__":
    raise SystemExit(main())
