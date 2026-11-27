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
    DTYPES,
    vector_kernel_config,
)

# Four end-to-end cases cover reg-reg and reg-scalar shifts. B64 reg-reg
# examples include counts around the 64-bit boundary for the CPU reference.
VECTOR_ELE = 400
VL_ELE = 64
LOOPS = (VECTOR_ELE + VL_ELE - 1) // VL_ELE
SHIFT_BITS = 4
_VV64_TILE_ELE = 1024
ALL_DTYPES = ("i32", "i16", "i8", "u8", "u16", "u32", "i64", "u64")

_KERNEL_DTYPE = tla.Int32
_AMOUNT_DTYPE = tla.Int32
_IS_VECTOR_SHIFT = True
_MASK_PATTERN = tla.mask.ALL
_IS_64_BIT = False
_KERNEL_SHAPE = (VECTOR_ELE,)
_SHIFT_OP = "shift_left"


@tla.kernel
def shift_op(
    mem_source: tla.Tensor,
    mem_shift: tla.Tensor,
    mem_output: tla.Tensor,
) -> None:
    ub_loaded = tla.flag("ub_loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    vec_done = tla.flag("vec_done", tla.pipes.VECTOR, tla.pipes.MTE3)

    source_gm = tla.tile_view(mem_source, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    shift_gm = tla.tile_view(mem_shift, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    output_gm = tla.tile_view(mem_output, tla.make_shape(VECTOR_ELE), tla.make_coord(0))

    if tla.const_expr(
        _IS_64_BIT and _IS_VECTOR_SHIFT and VECTOR_ELE >= 2 * _VV64_TILE_ELE
    ):
        source_ub = _make_vv64_ub_tensor(_KERNEL_DTYPE)
        shift_ub = _make_vv64_ub_tensor(_AMOUNT_DTYPE)
        output_ub = _make_vv64_ub_tensor(_KERNEL_DTYPE)
        _shift_vv64_pipeline(
            source_gm, shift_gm, output_gm, source_ub, shift_ub, output_ub
        )
        return

    source_ub, source_ub_copy = _make_ub_tensor(source_gm, _KERNEL_DTYPE)
    output_ub, output_ub_copy = _make_ub_tensor(output_gm, _KERNEL_DTYPE)
    if tla.const_expr(_IS_VECTOR_SHIFT):
        shift_ub, shift_ub_copy = _make_ub_tensor(shift_gm, _AMOUNT_DTYPE)

    with tla.vector():
        tla.copy(source_ub_copy, source_gm)
        if tla.const_expr(_IS_VECTOR_SHIFT):
            tla.copy(shift_ub_copy, shift_gm)
        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)

        with tla.vec.func(mode="simd"):
            remaining = VECTOR_ELE
            for i in tla.range(LOOPS):
                source_tile = _chunk(source_ub, i)
                output_tile = _chunk(output_ub, i)
                source_reg = source_tile.load()
                if tla.const_expr(_IS_VECTOR_SHIFT):
                    shift_reg = _chunk(shift_ub, i).load()
                if tla.const_expr(VECTOR_ELE % VL_ELE == 0):
                    tail = tla.create_mask(pattern=_MASK_PATTERN, dtype=_KERNEL_DTYPE)
                else:
                    tail, remaining = tla.update_mask(remaining, dtype=_KERNEL_DTYPE)

                # The public API dispatches to reg-reg or reg-scalar TLA ops
                # according to the type of its second argument.
                if tla.const_expr(_SHIFT_OP == "shift_left"):
                    result = tla.shift_left(source_reg, shift_reg, mask=tail)
                elif tla.const_expr(_SHIFT_OP == "shift_right"):
                    result = tla.shift_right(source_reg, shift_reg, mask=tail)
                elif tla.const_expr(_SHIFT_OP == "shift_lefts"):
                    result = tla.shift_left(source_reg, SHIFT_BITS, mask=tail)
                else:
                    result = tla.shift_right(source_reg, SHIFT_BITS, mask=tail)
                if tla.const_expr(_IS_64_BIT or VECTOR_ELE % VL_ELE == 0):
                    output_tile.store(result)
                else:
                    output_tile.store(result, mask=tail)

        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)
        tla.copy(output_gm, output_ub_copy)
        tla.pipe_barrier(tla.pipes.ALL)


@tla.jit
def _shift_vv64_pipeline(
    source_gm: Any,
    shift_gm: Any,
    output_gm: Any,
    source_ub: Any,
    shift_ub: Any,
    output_ub: Any,
) -> None:
    ready = (
        tla.flag("vv64_ready_even", tla.pipes.MTE2, tla.pipes.VECTOR),
        tla.flag("vv64_ready_odd", tla.pipes.MTE2, tla.pipes.VECTOR),
    )
    input_free = (
        tla.flag("vv64_input_free_even", tla.pipes.VECTOR, tla.pipes.MTE2),
        tla.flag("vv64_input_free_odd", tla.pipes.VECTOR, tla.pipes.MTE2),
    )
    done = (
        tla.flag("vv64_done_even", tla.pipes.VECTOR, tla.pipes.MTE3),
        tla.flag("vv64_done_odd", tla.pipes.VECTOR, tla.pipes.MTE3),
    )
    output_free = (
        tla.flag("vv64_output_free_even", tla.pipes.MTE3, tla.pipes.VECTOR),
        tla.flag("vv64_output_free_odd", tla.pipes.MTE3, tla.pipes.VECTOR),
    )
    tile_count = (VECTOR_ELE + _VV64_TILE_ELE - 1) // _VV64_TILE_ELE
    registers_per_tile = _VV64_TILE_ELE // VL_ELE
    with tla.vector():
        _load_vv_tile(source_ub, shift_ub, source_gm, shift_gm, 0)
        tla.set_flag(ready[0])
        for tile in tla.range_constexpr(tile_count):
            slot = tile % 2
            # A reused input slot must remain intact until its VF has read it.
            if tla.const_expr(tile + 1 < tile_count):
                if tla.const_expr(tile >= 1):
                    tla.wait_flag(input_free[(tile + 1) % 2])
                _load_vv_tile(source_ub, shift_ub, source_gm, shift_gm, tile + 1)
                tla.set_flag(ready[(tile + 1) % 2])
            tla.wait_flag(ready[slot])
            # The previous output in this slot must have reached GM before
            # the next VF can overwrite it.
            if tla.const_expr(tile >= 2):
                tla.wait_flag(output_free[slot])
            with tla.vec.func(mode="simd"):
                mask = tla.create_mask(pattern=_MASK_PATTERN, dtype=_KERNEL_DTYPE)
                remaining = VECTOR_ELE - tile * _VV64_TILE_ELE
                for i in tla.range(registers_per_tile):
                    chunk = slot * registers_per_tile + i
                    if tla.const_expr(
                        tile == tile_count - 1 and VECTOR_ELE % _VV64_TILE_ELE != 0
                    ):
                        tail, remaining = tla.update_mask(
                            remaining, dtype=_KERNEL_DTYPE
                        )
                        active = tla.bitwise_and(mask, tail)
                    else:
                        active = mask
                    source = _chunk(source_ub, chunk).load()
                    amount = _chunk(shift_ub, chunk).load()
                    if tla.const_expr(_SHIFT_OP == "shift_left"):
                        result = tla.shift_left(source, amount, mask=active)
                    else:
                        result = tla.shift_right(source, amount, mask=active)
                    _chunk(output_ub, chunk).store(result)
            if tla.const_expr(tile + 2 < tile_count):
                tla.set_flag(input_free[slot])
            tla.set_flag(done[slot])
            tla.wait_flag(done[slot])
            tla.copy(_dma_tile(output_gm, tile), _dma_tile(output_ub, slot))
            if tla.const_expr(tile + 2 < tile_count):
                tla.set_flag(output_free[slot])
        tla.pipe_barrier(tla.pipes.ALL)


def _dma_tile(tensor: Any, tile: int) -> Any:
    return tla.tile_view(tensor, tla.make_shape(_VV64_TILE_ELE), tla.make_coord(tile))


def _load_vv_tile(
    source_ub: Any, shift_ub: Any, source_gm: Any, shift_gm: Any, tile: int
) -> None:
    tla.copy(_dma_tile(source_ub, tile % 2), _dma_tile(source_gm, tile))
    tla.copy(_dma_tile(shift_ub, tile % 2), _dma_tile(shift_gm, tile))


def _make_vv64_ub_tensor(dtype: Any) -> Any:
    length = 2 * _VV64_TILE_ELE
    ptr = tla.allocate(length, dtype, tla.AddressSpace.ub, 256)
    layout = tla.make_layout(shape=tla.make_shape(length), stride=tla.make_stride(1))
    return tla.make_tensor(ptr, layout)


def _make_ub_tensor(like_tensor: Any, dtype: Any) -> tuple[Any, Any]:
    length = LOOPS * VL_ELE
    ptr = tla.allocate(length, dtype, tla.AddressSpace.ub, 256)
    copy_view = tla.make_tensor_like(ptr, like_tensor, tla.arch.RowMajor)
    if _IS_64_BIT and length != VECTOR_ELE:
        # Compute on full registers while copying only the logical GM extent.
        layout = tla.make_layout(
            shape=tla.make_shape(length), stride=tla.make_stride(1)
        )
        return tla.make_tensor(ptr, layout), copy_view
    return copy_view, copy_view


def _chunk(tensor: Any, chunk_idx: Any) -> Any:
    return tla.tile_view(tensor, tla.make_shape(VL_ELE), tla.make_coord(chunk_idx))


def _operator_specs() -> dict[str, dict[str, Any]]:
    # Keep the underlying four TLA op names as CLI case names. The scalar
    # cases still exercise the intentionally unified tla.shift_left/right API.
    return {
        "shift_left": {"default_atol": 0},
        "shift_right": {"default_atol": 0},
        "shift_lefts": {"default_atol": 0},
        "shift_rights": {"default_atol": 0},
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
    global VECTOR_ELE, VL_ELE, LOOPS, _KERNEL_DTYPE, _KERNEL_SHAPE, _SHIFT_OP
    global _AMOUNT_DTYPE, _IS_VECTOR_SHIFT, _IS_64_BIT
    if op_name not in _operator_specs():
        choices = ", ".join(sorted(_operator_specs()))
        raise SystemExit(
            f"unknown shift operator {op_name!r}; expected one of: {choices}"
        )
    config = vector_kernel_config(dtype_name, shape, ALL_DTYPES)
    VECTOR_ELE = config.vector_elements
    VL_ELE = config.lanes
    LOOPS = config.loops
    _KERNEL_DTYPE = config.tla_dtype
    _KERNEL_SHAPE = shape if shape is not None else (VECTOR_ELE,)
    _SHIFT_OP = op_name
    _AMOUNT_DTYPE = DTYPES["i" + dtype_name[1:]].tla_dtype
    _IS_VECTOR_SHIFT = not op_name.endswith("s")
    _IS_64_BIT = config.element_bytes == 8
    return config.tla_dtype, config.torch_dtype, config.default_sentinel


def _make_inputs(args: Any, dtype_name: str, torch: Any) -> tuple[Any, Any]:
    _, dtype, _ = _set_kernel_config(args.op, dtype_name, args.shape)
    spec = DTYPES[dtype_name]
    count_dtype = getattr(torch, DTYPES["i" + dtype_name[1:]].torch_dtype_name)
    bits = 8 * spec.element_bytes
    high = 1 << (bits - 1)
    if dtype_name.startswith("u"):
        values = [0, 1, high - 1, high, high + 1, 2 * high - 1]
    else:
        values = [0, 1, high - 1, -high, -1]
    source = torch.tensor(
        [values[i % len(values)] for i in range(VECTOR_ELE)], dtype=dtype
    )
    if bits == 64 and _IS_VECTOR_SHIFT:
        counts = (0, 1, 31, 32, 63, 64, 65, 128, 32767, SHIFT_BITS)
        shift = torch.tensor(
            [counts[i % len(counts)] for i in range(VECTOR_ELE)], dtype=count_dtype
        )
    else:
        shift = torch.full((VECTOR_ELE,), SHIFT_BITS, dtype=count_dtype)
    return source.npu(), shift.npu()


def reference_shift(op_name: str, source: Any, amounts: Any, scalar: int) -> Any:
    """Compute reference bits for the tutorial's count range using Python integers."""
    import torch

    data = source.cpu()
    bits = data.element_size() * 8
    mask = (1 << bits) - 1
    signed = data.dtype in {torch.int8, torch.int16, torch.int32, torch.int64}
    counts = (
        [scalar] * data.numel() if op_name.endswith("s") else amounts.cpu().tolist()
    )
    values = []
    for x, count in zip(data.tolist(), counts, strict=True):
        if op_name.startswith("shift_left"):
            value = (x << count) & mask if count < bits else 0
            if signed and value >= 1 << (bits - 1):
                value -= 1 << bits
        else:
            value = x >> min(count, bits)
        values.append(value)
    return torch.tensor(values, dtype=data.dtype)


def _expected(op_name: str, inputs: tuple[Any, ...]) -> Any:
    source, shift = inputs
    return reference_shift(op_name, source, shift, SHIFT_BITS)


HARNESS = DirectVectorOpHarness(
    DirectVectorOpConfig(
        description="Compile and run vector left/right shifts with scalar or per-lane counts.",
        kernel=shift_op,
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
        float_dtypes=frozenset(),
        input_count=2,
    )
)


def main() -> int:
    return HARNESS.main()


if __name__ == "__main__":
    raise SystemExit(main())
