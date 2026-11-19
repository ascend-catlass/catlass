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

VECTOR_ELE = 400
VL_ELE = 64
LOOPS = (VECTOR_ELE + VL_ELE - 1) // VL_ELE
ALL_DTYPES = ("f16", "bf16", "f32", "i32", "i16", "i8", "u8", "u16", "u32")
_UNSIGNED_BITS = {"u8": 8, "u16": 16, "u32": 32}

_KERNEL_DTYPE = tla.Int32
_KERNEL_ELEMENT_BYTES = 4
_KERNEL_SHAPE = (VECTOR_ELE,)
_SUPPORTS_REG_NOT = True


@tla.kernel
def bitwise_ops(
    mem_a: tla.Tensor,
    mem_b: tla.Tensor,
    mem_mask_not: tla.Tensor,
    mem_mask_and: tla.Tensor,
    mem_mask_or: tla.Tensor,
    mem_mask_xor: tla.Tensor,
    mem_reg_not: tla.Tensor,
    mem_reg_and: tla.Tensor,
    mem_reg_or: tla.Tensor,
    mem_reg_xor: tla.Tensor,
) -> None:
    ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    vec_done = tla.flag("vec_done", tla.arch.VECTOR, tla.arch.MTE3)

    a_gm = tla.tile_view(mem_a, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    b_gm = tla.tile_view(mem_b, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    mask_not_gm = tla.tile_view(
        mem_mask_not, tla.make_shape(VECTOR_ELE), tla.make_coord(0)
    )
    mask_and_gm = tla.tile_view(
        mem_mask_and, tla.make_shape(VECTOR_ELE), tla.make_coord(0)
    )
    mask_or_gm = tla.tile_view(
        mem_mask_or, tla.make_shape(VECTOR_ELE), tla.make_coord(0)
    )
    mask_xor_gm = tla.tile_view(
        mem_mask_xor, tla.make_shape(VECTOR_ELE), tla.make_coord(0)
    )
    reg_not_gm = tla.tile_view(
        mem_reg_not, tla.make_shape(VECTOR_ELE), tla.make_coord(0)
    )
    reg_and_gm = tla.tile_view(
        mem_reg_and, tla.make_shape(VECTOR_ELE), tla.make_coord(0)
    )
    reg_or_gm = tla.tile_view(mem_reg_or, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    reg_xor_gm = tla.tile_view(
        mem_reg_xor, tla.make_shape(VECTOR_ELE), tla.make_coord(0)
    )

    a_ub = _make_ub_tensor(a_gm)
    b_ub = _make_ub_tensor(b_gm)
    mask_not_ub = _make_ub_tensor(mask_not_gm)
    mask_and_ub = _make_ub_tensor(mask_and_gm)
    mask_or_ub = _make_ub_tensor(mask_or_gm)
    mask_xor_ub = _make_ub_tensor(mask_xor_gm)
    reg_not_ub = _make_ub_tensor(reg_not_gm)
    reg_and_ub = _make_ub_tensor(reg_and_gm)
    reg_or_ub = _make_ub_tensor(reg_or_gm)
    reg_xor_ub = _make_ub_tensor(reg_xor_gm)

    with tla.vector():
        tla.copy(a_ub, a_gm)
        tla.copy(b_ub, b_gm)

        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)
        with tla.vec.func(mode="simd"):
            if _SUPPORTS_REG_NOT:
                remaining = VECTOR_ELE
                for i in tla.range(LOOPS):
                    a_t = _chunk(a_ub, i)
                    b_t = _chunk(b_ub, i)
                    mask_not_t = _chunk(mask_not_ub, i)
                    mask_and_t = _chunk(mask_and_ub, i)
                    mask_or_t = _chunk(mask_or_ub, i)
                    mask_xor_t = _chunk(mask_xor_ub, i)
                    reg_not_t = _chunk(reg_not_ub, i)
                    reg_and_t = _chunk(reg_and_ub, i)
                    reg_or_t = _chunk(reg_or_ub, i)
                    reg_xor_t = _chunk(reg_xor_ub, i)

                    av = a_t.load()
                    bv = b_t.load()
                    tail, remaining = tla.update_mask(remaining, dtype=_KERNEL_DTYPE)
                    m_h = tla.create_mask(pattern=tla.mask.H, dtype=_KERNEL_DTYPE)
                    m_q = tla.create_mask(pattern=tla.mask.Q, dtype=_KERNEL_DTYPE)
                    m_m4 = tla.create_mask(pattern=tla.mask.M4, dtype=_KERNEL_DTYPE)

                    reg_not = tla.bitwise_not(av, mask=tail)
                    zero = tla.bitwise_and(av, reg_not)
                    reg_and = tla.bitwise_and(av, bv, mask=tail)
                    reg_or = tla.bitwise_or(av, bv)
                    reg_xor = tla.bitwise_xor(av, bv, mask=tail)

                    mask_not = tla.bitwise_not(m_q)
                    mask_and = tla.bitwise_and(m_h, m_m4, mask=tail)
                    mask_or = tla.bitwise_or(m_q, m_m4)
                    mask_xor = tla.bitwise_xor(m_h, m_m4, mask=tail)

                    mask_not_t.store(tla.where(mask_not, av, zero), mask=tail)
                    mask_and_t.store(tla.where(mask_and, av, zero), mask=tail)
                    mask_or_t.store(tla.where(mask_or, av, zero), mask=tail)
                    mask_xor_t.store(tla.where(mask_xor, av, zero), mask=tail)
                    reg_not_t.store(reg_not, mask=tail)
                    reg_and_t.store(reg_and, mask=tail)
                    reg_or_t.store(reg_or, mask=tail)
                    reg_xor_t.store(reg_xor, mask=tail)
            else:
                remaining = VECTOR_ELE
                for i in tla.range(LOOPS):
                    a_t = _chunk(a_ub, i)
                    b_t = _chunk(b_ub, i)
                    mask_not_t = _chunk(mask_not_ub, i)
                    mask_and_t = _chunk(mask_and_ub, i)
                    mask_or_t = _chunk(mask_or_ub, i)
                    mask_xor_t = _chunk(mask_xor_ub, i)
                    reg_not_t = _chunk(reg_not_ub, i)
                    reg_and_t = _chunk(reg_and_ub, i)
                    reg_or_t = _chunk(reg_or_ub, i)
                    reg_xor_t = _chunk(reg_xor_ub, i)

                    av = a_t.load()
                    bv = b_t.load()
                    tail, remaining = tla.update_mask(remaining, dtype=_KERNEL_DTYPE)
                    m_h = tla.create_mask(pattern=tla.mask.H, dtype=_KERNEL_DTYPE)
                    m_q = tla.create_mask(pattern=tla.mask.Q, dtype=_KERNEL_DTYPE)
                    m_m4 = tla.create_mask(pattern=tla.mask.M4, dtype=_KERNEL_DTYPE)

                    zero = tla.bitwise_xor(av, av)
                    reg_and = tla.bitwise_and(av, bv, mask=tail)
                    reg_or = tla.bitwise_or(av, bv)
                    reg_xor = tla.bitwise_xor(av, bv, mask=tail)

                    mask_not = tla.bitwise_not(m_q, mask=tail)
                    mask_and = tla.bitwise_and(m_h, m_m4, mask=tail)
                    mask_or = tla.bitwise_or(m_q, m_m4)
                    mask_xor = tla.bitwise_xor(m_h, m_m4, mask=tail)

                    mask_not_t.store(tla.where(mask_not, av, zero), mask=tail)
                    mask_and_t.store(tla.where(mask_and, av, zero), mask=tail)
                    mask_or_t.store(tla.where(mask_or, av, zero), mask=tail)
                    mask_xor_t.store(tla.where(mask_xor, av, zero), mask=tail)
                    reg_not_t.store(av, mask=tail)
                    reg_and_t.store(reg_and, mask=tail)
                    reg_or_t.store(reg_or, mask=tail)
                    reg_xor_t.store(reg_xor, mask=tail)

        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)

        tla.copy(mask_not_gm, mask_not_ub)
        tla.copy(mask_and_gm, mask_and_ub)
        tla.copy(mask_or_gm, mask_or_ub)
        tla.copy(mask_xor_gm, mask_xor_ub)
        tla.copy(reg_not_gm, reg_not_ub)
        tla.copy(reg_and_gm, reg_and_ub)
        tla.copy(reg_or_gm, reg_or_ub)
        tla.copy(reg_xor_gm, reg_xor_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def _make_ub_tensor(like_tensor: Any) -> Any:
    ptr = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 32)
    return tla.make_tensor_like(ptr, like_tensor, tla.arch.RowMajor)


def _chunk(tensor: Any, chunk_idx: Any) -> Any:
    return tla.tile_view(tensor, tla.make_shape(VL_ELE), tla.make_coord(chunk_idx))


def _operator_specs() -> dict[str, dict[str, Any]]:
    return {"bitwise_ops": {"default_atol": 0}}


def _is_unsupported_case(op_name: str, dtype_name: str) -> bool:
    del op_name
    return dtype_name not in ALL_DTYPES


def _print_skip(op_name: str, dtype_name: str, shape: tuple[int, ...]) -> None:
    del shape
    print(f"skip op={op_name} dtype={dtype_name}: unsupported case")


def _set_kernel_config(
    op_name: str, dtype_name: str, shape: tuple[int, ...] | None = None
) -> tuple[type[Any], Any, float | int]:
    global VL_ELE, LOOPS, VECTOR_ELE, _KERNEL_DTYPE, _KERNEL_ELEMENT_BYTES
    global _KERNEL_SHAPE, _SUPPORTS_REG_NOT
    if op_name not in _operator_specs():
        raise SystemExit("unknown bitwise operator")
    config = vector_kernel_config(dtype_name, shape, ALL_DTYPES)
    VECTOR_ELE = config.vector_elements
    _KERNEL_SHAPE = shape if shape is not None else (VECTOR_ELE,)
    VL_ELE = config.lanes
    LOOPS = config.loops
    _KERNEL_DTYPE = config.tla_dtype
    _KERNEL_ELEMENT_BYTES = config.element_bytes
    # Unsigned and bf16 cases use the fallback path, which does not compute register bitwise NOT.
    _SUPPORTS_REG_NOT = dtype_name != "bf16" and dtype_name not in _UNSIGNED_BITS
    return config.tla_dtype, config.torch_dtype, config.default_sentinel


def _make_unsigned_inputs(torch: Any, dtype_name: str, dtype: Any) -> tuple[Any, Any]:
    bits = _UNSIGNED_BITS[dtype_name]
    idx = torch.arange(VECTOR_ELE, dtype=torch.int64)

    # Mask for the most significant bit.
    msb = 1 << (bits - 1)

    # Modulus used to keep values within the unsigned range.
    modulus = 1 << bits

    # Generate different base bit patterns.
    base_a = idx * 13
    base_b = idx * 7 + 5

    # Flip the highest bit on alternating elements of a.
    flip_a = (idx % 2) * msb

    # Flip the highest bit on alternating pairs of elements of b.
    flip_b = ((idx // 2) % 2) * msb

    # Apply the bit flips and constrain values to the target bit width.
    a = (base_a ^ flip_a) % modulus
    b = (base_b ^ flip_b) % modulus

    return a.to(dtype).to("npu"), b.to(dtype).to("npu")


def _make_inputs(args: Any, dtype_name: str, torch: Any) -> tuple[Any, ...]:
    _, dtype, _ = _set_kernel_config(args.op, dtype_name, args.shape)
    device = "npu"
    if dtype_name in _UNSIGNED_BITS:
        return _make_unsigned_inputs(torch, dtype_name, dtype)
    arange = torch.arange(VECTOR_ELE, dtype=torch.int64, device=device)
    if dtype_name in {"f16", "bf16", "f32"}:
        base = (arange % 31).to(torch.float32) / 64.0
        a = (1.0 + base).to(dtype)
        b = (1.0 + ((arange * 3 + 5) % 31).to(torch.float32) / 64.0).to(dtype)
        return a, b
    a = ((arange * 13 % 127) - 63).to(dtype)
    b = ((arange * 7 % 113) - 47).to(dtype)
    return a, b


def _select(torch: Any, x: Any, keep: Any) -> Any:
    return torch.where(keep, x, torch.zeros_like(x))


def _raw_int_dtype(torch: Any, x: Any) -> Any | None:
    # PyTorch CPU bitwise ops may not accept unsigned 16/32-bit tensors; use same-width signed views.
    # Viewing preserves each element's bits; it does not convert the unsigned value numerically.
    return {
        torch.float32: torch.int32,
        torch.float16: torch.int16,
        torch.bfloat16: torch.int16,
        torch.uint8: torch.int8,
        torch.uint16: torch.int16,
        torch.uint32: torch.int32,
    }.get(x.dtype)


def _bitwise_unary(torch: Any, x: Any) -> Any:
    raw_dtype = _raw_int_dtype(torch, x)
    if raw_dtype is None:
        return torch.bitwise_not(x)
    return torch.bitwise_not(x.view(raw_dtype)).view(x.dtype)


def _bitwise_binary(torch: Any, op: Any, x: Any, y: Any) -> Any:
    raw_dtype = _raw_int_dtype(torch, x)
    if raw_dtype is None:
        return op(x, y)
    return op(x.view(raw_dtype), y.view(raw_dtype)).view(x.dtype)


def _expected(op_name: str, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
    del op_name
    import torch

    a, b = inputs
    # CPU reference operations support unsigned storage through signed bit views.
    if a.dtype in (torch.uint8, torch.uint16, torch.uint32):
        a, b = a.cpu(), b.cpu()
        raw_dtype = _raw_int_dtype(torch, a)
        a_raw = a.view(raw_dtype)
        # Build the same H/Q/M4 lane masks as the kernel, then apply them to the raw bit patterns.
        lane = torch.arange(VECTOR_ELE) % VL_ELE
        h = lane < (VL_ELE // 2)
        q = lane < (VL_ELE // 4)
        m4 = (lane % 4) == 0
        return (
            _select(torch, a_raw, ~q).view(a.dtype),
            _select(torch, a_raw, h & m4).view(a.dtype),
            _select(torch, a_raw, q | m4).view(a.dtype),
            _select(torch, a_raw, h ^ m4).view(a.dtype),
            a,
            _bitwise_binary(torch, torch.bitwise_and, a, b),
            _bitwise_binary(torch, torch.bitwise_or, a, b),
            _bitwise_binary(torch, torch.bitwise_xor, a, b),
        )
    lane = torch.arange(VECTOR_ELE, device=a.device) % VL_ELE
    h = lane < (VL_ELE // 2)
    q = lane < (VL_ELE // 4)
    m4 = (lane % 4) == 0
    return (
        _select(torch, a, ~q),
        _select(torch, a, h & m4),
        _select(torch, a, q | m4),
        _select(torch, a, h ^ m4),
        _bitwise_unary(torch, a) if _SUPPORTS_REG_NOT else a,
        _bitwise_binary(torch, torch.bitwise_and, a, b),
        _bitwise_binary(torch, torch.bitwise_or, a, b),
        _bitwise_binary(torch, torch.bitwise_xor, a, b),
    )


HARNESS = DirectVectorOpHarness(
    DirectVectorOpConfig(
        description="Compile and run MaskSSA and VectorSSA bitwise kernels.",
        kernel=bitwise_ops,
        all_dtypes=ALL_DTYPES,
        operator_specs=_operator_specs,
        set_kernel_config=_set_kernel_config,
        get_vector_elements=lambda: VECTOR_ELE,
        get_kernel_shape=lambda: _KERNEL_SHAPE,
        make_inputs=_make_inputs,
        expected=_expected,
        unsupported_case=_is_unsupported_case,
        print_skip=_print_skip,
        script_path=Path(__file__),
        float_dtypes=frozenset({"f16", "bf16", "f32"}),
        output_count=8,
    )
)


def main() -> int:
    return HARNESS.main()


if __name__ == "__main__":
    raise SystemExit(main())
