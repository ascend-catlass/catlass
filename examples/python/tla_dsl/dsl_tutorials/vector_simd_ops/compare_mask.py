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
from typing import Any, Callable

import catlass.tla as tla

from vector_op_harness import (
    DirectVectorOpConfig,
    DirectVectorOpHarness,
    vector_kernel_config,
)

# End-to-end coverage for ``tla.cmp`` / compare masks:
#   - all six vector-vector modes (lt/le/gt/ge/eq/ne) via ``tla.where``
#   - vector-scalar compares against 0.0 (gt/ge)
#   - input mask on ``tla.cmp`` (pattern H)
#   - cmp result driving masked ``tla.add`` plus masked ``tla.store``
#   - static ``tile_view(coord=0)`` reference vs dynamic loop chunks (shape metadata
#     may differ while tile width stays VL-wide)

# A multiple of every VL width (32/64/128/256), so 64-bit registers stay full
# and the 64-bit store verifier accepts every chunk.
VECTOR_ELE = 512
VL_ELE = 64
LOOPS = (VECTOR_ELE + VL_ELE - 1) // VL_ELE
ALL_DTYPES = ("f32", "u8", "u16", "u32", "i8", "i16", "bf16", "i64", "u64")
_UNSIGNED = frozenset({"u8", "u16", "u32", "u64"})
_64BIT = frozenset({"i64", "u64"})
# Values above each signed boundary so VCmp vs VCmpU disagree.
_UNSIGNED_BIAS = {"u8": 128, "u16": 32768, "u32": 0x80000000}

_KERNEL_DTYPE = tla.Float32
_KERNEL_ELEMENT_BYTES = 4
_KERNEL_SHAPE = (VECTOR_ELE,)
_CMP_OP: Callable[[Any, Any], Any] | None = None
_OP_NAME: str = ""
_BATCH_OPS = ("vector_vector_lt",) * 4


@tla.kernel
def compare_mask(mem_a: tla.Tensor, mem_b: tla.Tensor, mem_out: tla.Tensor) -> None:
    ub_loaded = tla.flag("ub_loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    vec_done = tla.flag("vec_done", tla.pipes.VECTOR, tla.pipes.MTE3)

    a_gm = tla.tile_view(mem_a, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    b_gm = tla.tile_view(mem_b, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    out_gm = tla.tile_view(mem_out, tla.make_shape(VECTOR_ELE), tla.make_coord(0))

    a_ub = _make_ub_tensor(a_gm)
    b_ub = _make_ub_tensor(b_gm)
    out_ub = _make_ub_tensor(out_gm)

    with tla.vector():
        tla.copy(a_ub, a_gm)
        tla.copy(b_ub, b_gm)

        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)
        with tla.vec.func(mode="simd"):
            a_static = tla.tile_view(a_ub, tla.make_shape(VL_ELE), tla.make_coord(0))
            a_ref = a_static.load()
            remaining = VECTOR_ELE
            for i in tla.range(LOOPS):
                a_tile = _chunk(a_ub, i)
                b_tile = _chunk(b_ub, i)
                out_tile = _chunk(out_ub, i)

                av = a_tile.load()
                bv = b_tile.load()
                tail, remaining = tla.update_mask(remaining, dtype=_KERNEL_DTYPE)
                zero = tla.sub(av, av)

                if _OP_NAME == "static_dynamic_lt":
                    m = tla.cmp(a_ref, bv, "lt")
                    _store_result(out_tile, tla.where(m, bv, zero), tail)
                elif _OP_NAME == "cmp_masked_fused":
                    m_lt = tla.cmp(av, bv, "lt")
                    m_ge = tla.cmp(av, 0.0, "ge")
                    fused = tla.add(av, bv, mask=m_lt)
                    _store_result(out_tile, tla.where(m_ge, fused, zero), tail)
                else:
                    _store_result(out_tile, _CMP_OP(av, bv), tail)

        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)

        tla.copy(out_gm, out_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def _batch_compare_chunk(
    a_ref: Any,
    a_tile: Any,
    b_tile: Any,
    out_tile: Any,
    tail: Any,
    op_name: str,
) -> None:
    av = a_tile.load()
    bv = b_tile.load()
    zero = tla.sub(av, av)
    if op_name == "static_dynamic_lt":
        result = tla.where(tla.cmp(a_ref, bv, "lt"), bv, zero)
    elif op_name == "cmp_masked_fused":
        m_lt = tla.cmp(av, bv, "lt")
        m_ge = tla.cmp(av, 0.0, "ge")
        result = tla.where(m_ge, tla.add(av, bv, mask=m_lt), zero)
    else:
        result = _operator_specs()[op_name]["op"](av, bv)
    _store_result(out_tile, result, tail)


@tla.kernel
def compare_mask_batch(
    mem_a: tla.Tensor, mem_b: tla.Tensor, mem_out: tla.Tensor
) -> None:
    ub_loaded = tla.flag("ub_loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    vec_done = tla.flag("vec_done", tla.pipes.VECTOR, tla.pipes.MTE3)
    block_idx = tla.arch.block_idx()
    a_gm = tla.tile_view(mem_a, tla.make_shape(VECTOR_ELE), tla.make_coord(block_idx))
    b_gm = tla.tile_view(mem_b, tla.make_shape(VECTOR_ELE), tla.make_coord(block_idx))
    out_gm = tla.tile_view(
        mem_out, tla.make_shape(VECTOR_ELE), tla.make_coord(block_idx)
    )
    a_ub = _make_ub_tensor(a_gm)
    b_ub = _make_ub_tensor(b_gm)
    out_ub = _make_ub_tensor(out_gm)

    with tla.vector():
        tla.copy(a_ub, a_gm)
        tla.copy(b_ub, b_gm)
        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)
        with tla.vec.func(mode="simd"):
            a_static = tla.tile_view(a_ub, tla.make_shape(VL_ELE), tla.make_coord(0))
            a_ref = a_static.load()
            remaining = VECTOR_ELE
            for i in tla.range(LOOPS):
                a_tile = _chunk(a_ub, i)
                b_tile = _chunk(b_ub, i)
                out_tile = _chunk(out_ub, i)
                tail, remaining = tla.update_mask(remaining, dtype=_KERNEL_DTYPE)
                if block_idx == 0:
                    _batch_compare_chunk(
                        a_ref, a_tile, b_tile, out_tile, tail, _BATCH_OPS[0]
                    )
                elif block_idx == 1:
                    _batch_compare_chunk(
                        a_ref, a_tile, b_tile, out_tile, tail, _BATCH_OPS[1]
                    )
                elif block_idx == 2:
                    _batch_compare_chunk(
                        a_ref, a_tile, b_tile, out_tile, tail, _BATCH_OPS[2]
                    )
                else:
                    _batch_compare_chunk(
                        a_ref, a_tile, b_tile, out_tile, tail, _BATCH_OPS[3]
                    )
        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)
        tla.copy(out_gm, out_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def _store_result(out_tile: Any, result: Any, tail: Any) -> None:
    # 64-bit masked stores hit the same AVE library-call ABI limitation; the
    # tutorial keeps a full 32-lane register, so store 64-bit results unmasked.
    if _KERNEL_ELEMENT_BYTES == 8:
        out_tile.store(result)
    else:
        out_tile.store(result, mask=tail)


def _make_ub_tensor(like_tensor: Any) -> Any:
    ptr = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    return tla.make_tensor_like(ptr, like_tensor, tla.arch.RowMajor)


def _chunk(tensor: Any, chunk_idx: Any) -> Any:
    return tla.tile_view(tensor, tla.make_shape(VL_ELE), tla.make_coord(chunk_idx))


def _vector_vector_where(mode: str) -> Callable[[Any, Any], Any]:
    def op(lhs: Any, rhs: Any) -> Any:
        return tla.where(tla.cmp(lhs, rhs, mode), lhs, rhs)

    return op


def _operator_specs() -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {
        "vector_scalar_gt": {
            "op": lambda lhs, rhs: tla.where(
                tla.cmp(lhs, 0.0, "gt"), lhs, tla.sub(rhs, rhs)
            ),
            "default_atol": 1e-5,
        },
        "vector_scalar_ge": {
            "op": lambda lhs, rhs: tla.where(
                tla.cmp(lhs, 0.0, "ge"), lhs, tla.sub(rhs, rhs)
            ),
            "default_atol": 1e-5,
        },
        "masked_vector_vector_lt": {
            "op": lambda lhs, rhs: tla.where(
                tla.cmp(
                    lhs,
                    rhs,
                    "lt",
                    mask=tla.create_mask(pattern=tla.mask.H, dtype=_KERNEL_DTYPE),
                ),
                lhs,
                rhs,
            ),
            "default_atol": 1e-5,
        },
        "cmp_masked_fused": {
            "op": None,
            "default_atol": 1e-5,
        },
        "static_dynamic_lt": {
            "op": None,
            "default_atol": 1e-5,
        },
    }
    for mode in ("lt", "le", "gt", "ge", "eq", "ne"):
        specs[f"vector_vector_{mode}"] = {
            "op": _vector_vector_where(mode),
            "default_atol": 1e-5,
        }
    return specs


def _require_full_64bit_registers(dtype_name: str, config: Any) -> None:
    # The 64-bit register store requires a full 32-lane register. Keep the
    # tutorial shape a multiple of the lane count instead of failing in the IR.
    if dtype_name in _64BIT and config.vector_elements % config.lanes:
        raise SystemExit(
            f"compare_mask.py: {dtype_name} requires --shape to be a multiple "
            f"of {config.lanes} (one full 64-bit register)"
        )


def _is_unsupported_case(op_name: str, dtype_name: str) -> bool:
    # Unsigned only covers vector-vector cmp (VCmpU); scalar/fused ops use f32 0.0.
    # 64-bit compare+select is covered by vector-vector only; the fused paths
    # also exercise 64-bit vector arithmetic, which is outside this case set.
    if dtype_name in _UNSIGNED or dtype_name in _64BIT:
        return not op_name.startswith("vector_vector_")
    return False


def _print_skip(op_name: str, dtype_name: str, shape: tuple[int, ...]) -> None:
    del shape
    print(f"skip op={op_name} dtype={dtype_name}: unsupported case")


def _set_kernel_config(
    op_name: str, dtype_name: str, shape: tuple[int, ...] | None = None
) -> tuple[type[Any], Any, float | int]:
    global VL_ELE, LOOPS, VECTOR_ELE, _KERNEL_DTYPE, _KERNEL_ELEMENT_BYTES
    global _KERNEL_SHAPE, _CMP_OP, _OP_NAME
    specs = _operator_specs()
    if op_name not in specs:
        choices = ", ".join(sorted(specs))
        raise SystemExit(
            f"unknown compare-mask operator {op_name!r}; expected one of: {choices}"
        )
    config = vector_kernel_config(dtype_name, shape, ALL_DTYPES)
    _require_full_64bit_registers(dtype_name, config)
    VECTOR_ELE = config.vector_elements
    _KERNEL_SHAPE = shape if shape is not None else (VECTOR_ELE,)
    VL_ELE = config.lanes
    LOOPS = config.loops
    _KERNEL_DTYPE = config.tla_dtype
    _KERNEL_ELEMENT_BYTES = config.element_bytes
    _OP_NAME = op_name
    _CMP_OP = specs[op_name]["op"]
    return config.tla_dtype, config.torch_dtype, config.default_sentinel


def _configure_batch(
    ops: tuple[str, ...], dtype_name: str, shape: tuple[int, ...]
) -> tuple[type[Any], Any, float | int]:
    global VL_ELE, LOOPS, VECTOR_ELE, _KERNEL_DTYPE, _KERNEL_ELEMENT_BYTES
    global _KERNEL_SHAPE, _BATCH_OPS
    if not 1 <= len(ops) <= 4:
        raise SystemExit("compare-mask batch requires one to four operations")
    config = vector_kernel_config(dtype_name, shape, ALL_DTYPES)
    _require_full_64bit_registers(dtype_name, config)
    VECTOR_ELE = config.vector_elements
    _KERNEL_SHAPE = shape
    VL_ELE = config.lanes
    LOOPS = config.loops
    _KERNEL_DTYPE = config.tla_dtype
    _KERNEL_ELEMENT_BYTES = config.element_bytes
    _BATCH_OPS = ops + (ops[-1],) * (4 - len(ops))
    return config.tla_dtype, config.torch_dtype, config.default_sentinel


def _make_inputs(args: Any, dtype_name: str, torch: Any) -> tuple[Any, Any]:
    _, dtype, _ = _set_kernel_config(args.op, dtype_name, args.shape)
    device = "npu"
    # NPU arange lacks uint; build on CPU then .to(npu).
    idx = torch.arange(VECTOR_ELE, dtype=torch.int64)
    if dtype_name in _UNSIGNED:
        bias = _UNSIGNED_BIAS.get(dtype_name)
        if bias is None:
            # u64: keep inputs inside the int64 range so the CPU golden can
            # compare them through .to(torch.int64); torch has no uint64 lt.
            a64 = idx % 31
            b64 = torch.flip(a64, dims=[0]) + 1
        else:
            a64 = (idx % 50) + (bias - 10)
            b64 = torch.flip(a64, dims=[0])
        return a64.to(dtype).to(device), b64.to(dtype).to(device)
    if dtype_name in ("i8", "i16", "i64"):
        a64 = (idx % 31) - 15
        b64 = torch.flip(a64, dims=[0]) + 1
        return a64.to(dtype).to(device), b64.to(dtype).to(device)
    idx = idx.to(dtype=dtype, device=device)
    a = ((idx % 31) - 15.0) * 0.5
    b = torch.flip(a, dims=[0]) + 0.25
    return a, b


def _chunk_expected(a: Any, b: Any, op_name: str, torch: Any) -> Any:
    if op_name.startswith("vector_vector_"):
        mode = op_name.removeprefix("vector_vector_")
        # Unsigned: compare in int64 so values above the signed bound stay correct.
        # Float: keep native compare — casting to int64 truncates fractions.
        if a.dtype in (torch.uint8, torch.uint16, torch.uint32, torch.uint64):
            lhs, rhs = a.to(torch.int64), b.to(torch.int64)
        else:
            lhs, rhs = a, b
        # torch.where has no uint64 kernel; select in the int64 view and cast back.
        wa, wb = (lhs, rhs) if a.dtype == torch.uint64 else (a, b)
        if mode == "lt":
            out = torch.where(lhs < rhs, wa, wb)
        elif mode == "le":
            out = torch.where(lhs <= rhs, wa, wb)
        elif mode == "gt":
            out = torch.where(lhs > rhs, wa, wb)
        elif mode == "ge":
            out = torch.where(lhs >= rhs, wa, wb)
        elif mode == "eq":
            out = torch.where(lhs == rhs, wa, wb)
        elif mode == "ne":
            out = torch.where(lhs != rhs, wa, wb)
        else:
            raise AssertionError(mode)
        return out.to(torch.uint64) if a.dtype == torch.uint64 else out
    if op_name == "vector_scalar_gt":
        return torch.where(a > 0.0, a, torch.zeros_like(a))
    if op_name == "vector_scalar_ge":
        return torch.where(a >= 0.0, a, torch.zeros_like(a))
    if op_name == "masked_vector_vector_lt":
        lane = torch.arange(a.numel(), device=a.device) % VL_ELE
        active = lane < (VL_ELE // 2)
        return torch.where((a < b) & active, a, b)
    if op_name == "cmp_masked_fused":
        m_lt = a < b
        partial = torch.where(m_lt, a + b, torch.zeros_like(a))
        m_ge = a >= 0.0
        return torch.where(m_ge, partial, torch.zeros_like(a))
    if op_name == "static_dynamic_lt":
        n = b.numel()
        mask = a[:n] < b
        return torch.where(mask, b, torch.zeros_like(b))
    raise AssertionError(op_name)


def _iter_chunks(tensor: Any, chunk_size: int) -> list[Any]:
    return [tensor[i : i + chunk_size] for i in range(0, tensor.numel(), chunk_size)]


def _expected(op_name: str, inputs: tuple[Any, ...]) -> Any:
    import torch

    a_full, b_full = inputs
    if op_name != "static_dynamic_lt":
        a_chunks = _iter_chunks(a_full, VL_ELE)
        b_chunks = _iter_chunks(b_full, VL_ELE)
        out_chunks = [
            _chunk_expected(a_chunks[i], b_chunks[i], op_name, torch)
            for i in range(len(a_chunks))
        ]
        return torch.cat(out_chunks, dim=0)

    a_ref = _iter_chunks(a_full, VL_ELE)[0]
    out_chunks = [
        _chunk_expected(a_ref, chunk, op_name, torch)
        for chunk in _iter_chunks(b_full, VL_ELE)
    ]
    return torch.cat(out_chunks, dim=0)


HARNESS = DirectVectorOpHarness(
    DirectVectorOpConfig(
        description="Compile and run vector compare-mask kernels.",
        kernel=compare_mask,
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
        float_dtypes=frozenset({"f32", "bf16"}),
        input_count=2,
        batch_kernel=compare_mask_batch,
        configure_batch=_configure_batch,
    )
)


def main() -> int:
    return HARNESS.main()


if __name__ == "__main__":
    raise SystemExit(main())
