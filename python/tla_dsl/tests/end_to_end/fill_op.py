# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compile and run L1 Tensor.fill cases, checking bit patterns and neighboring blocks."""

from __future__ import annotations

import argparse
import math
import re
from contextlib import redirect_stdout
from io import StringIO

import catlass.tla as tla

# 16 fractal rows and three C0 columns; all allocations occupy 1536 bytes.
L1_ROWS = 16
L1_C0_COLUMNS = 3
L1_BLOCK_COUNT = L1_ROWS * L1_C0_COLUMNS
SENTINEL = 0x5A39C671
DTYPE_SPECS = {
    "i16": (tla.Int16, 16),
    "u16": (tla.UInt16, 16),
    "f16": (tla.Float16, 16),
    "bf16": (tla.BFloat16, 16),
    "i32": (tla.Int32, 32),
    "u32": (tla.UInt32, 32),
    "f32": (tla.Float32, 32),
}
CASES = ("full", "subtile", "gap", "dynamic", "empty", "c0-boundary")


@tla.kernel
def fill_l1_kernel(
    gm_input: tla.Tensor,
    row_start: tla.Int32,
    col_start: tla.Int32,
    row_extent: tla.Int32,
    col_extent: tla.Int32,
    dtype: tla.Constexpr,
    layout_tag: tla.Constexpr,
    c0_elements: tla.Constexpr,
    values: tla.Constexpr,
) -> None:
    # Initialize every target dtype through the same byte-storage copy.
    l1_ptr = tla.allocate(L1_BLOCK_COUNT * 32, tla.Int8, tla.AddressSpace.l1, 512)
    t_l1_copy = tla.make_tensor_like(l1_ptr, gm_input, tla.arch.zN)
    l1_fill_ptr = tla.recast_ptr(l1_ptr, dtype=dtype)
    if tla.const_expr(layout_tag == tla.arch.zN):
        shape = (L1_ROWS, L1_C0_COLUMNS * c0_elements)
        coord = tla.make_coord(row_start, col_start)
        extent = tla.make_shape(row_extent, col_extent)
    else:
        shape = (L1_C0_COLUMNS * c0_elements, L1_ROWS)
        coord = tla.make_coord(col_start, row_start)
        extent = tla.make_shape(col_extent, row_extent)
    t_l1 = tla.make_tensor(
        l1_fill_ptr, tla.make_layout_with_tag(shape, dtype, layout_tag)
    )
    t_l1_fill = tla.get_tile(t_l1, coord, extent)
    fill_done = tla.flag("fill_readback", tla.arch.MTE2, tla.arch.MTE1)
    with tla.cube():
        tla.copy(t_l1_copy, gm_input)
        # Order the DMA sentinel write before the overlapping Fill write.
        tla.pipe_barrier(tla.arch.MTE2)
        # Each core checks one constant, using its own L1 allocation.
        for value_idx in tla.range_constexpr(len(values)):
            if tla.arch.block_idx() == value_idx:
                t_l1_fill.fill(values[value_idx])
        tla.set_flag(fill_done)
        tla.wait_flag(fill_done)
        for block_idx in tla.range_constexpr(L1_BLOCK_COUNT):
            l1_print_ptr = tla.recast_ptr(l1_ptr + block_idx * 32, dtype=tla.UInt32)
            t_l1_print = tla.make_tensor(
                l1_print_ptr,
                tla.make_layout_with_tag((1, 8), tla.UInt32, tla.arch.zN),
            )
            tla.print(t_l1_print, 8)


def _case_region(case: str, c0_elements: int) -> tuple[int, int, int, int]:
    # Coordinates are (fractal row, C0 column), swapped for nZ in the kernel.
    return {
        "full": (0, 0, L1_ROWS, L1_C0_COLUMNS * c0_elements),
        "subtile": (4, c0_elements, 8, c0_elements),
        "gap": (3, 0, 5, 3 * c0_elements),
        "dynamic": (2, c0_elements, 11, 2 * c0_elements),
        "empty": (4, c0_elements + 1, 8, 0),
        "c0-boundary": (4, 1, 8, c0_elements + 1),
    }[case]


def _fill_values(dtype_name: str) -> list[tuple[int | float, int]]:
    # Expected bits are independent of the Fill host encoder.
    if dtype_name == "i16":
        return [(0, 0), (-2, 0xFFFE)]
    if dtype_name == "u16":
        return [(0, 0), (0x8001, 0x8001), (0xFFFF, 0xFFFF)]
    if dtype_name == "i32":
        return [(0, 0), (-2, 0xFFFFFFFE)]
    if dtype_name == "u32":
        return [(0, 0), (0x80000001, 0x80000001), (0xFFFFFFFF, 0xFFFFFFFF)]
    one_and_half_bits, negative_two_bits, negative_zero_bits, inf_bits = {
        "f16": (0x3E00, 0xC000, 0x8000, 0x7C00),
        "bf16": (0x3FC0, 0xC000, 0x8000, 0x7F80),
        "f32": (0x3FC00000, 0xC0000000, 0x80000000, 0x7F800000),
    }[dtype_name]
    return [
        (0, 0),
        (1.5, one_and_half_bits),
        (-2.0, negative_two_bits),
        (-0.0, negative_zero_bits),
        (math.inf, inf_bits),
        (math.nan, inf_bits | (inf_bits >> 5)),
    ]


def _verify_output(
    output: str,
    dtype_name: str,
    layout: str,
    region: tuple[int, int, int, int],
    value_bits: list[tuple[int | float, int]],
) -> None:
    element_bits = DTYPE_SPECS[dtype_name][1]
    c0_elements = 256 // element_bits
    row_start, col_start, row_extent, col_extent = region
    records = []
    for line in output.splitlines():
        if line.startswith("tla.print "):
            records.append(line.strip())
    expected_records = L1_BLOCK_COUNT * len(value_bits)
    if len(records) != expected_records:
        raise AssertionError(
            f"expected {expected_records} C0 records, got {len(records)}"
        )
    seen_records = set()
    for record in records:
        match = re.fullmatch(
            r"tla\.print call=(\d+) block=(\d+) dtype=u32 (?:position=\S+ )?"
            r"shape=\[1,8\] count=8 values=\[([0-9, ]+)\]",
            record,
        )
        if match is None:
            raise AssertionError(f"malformed L1 record: {record}")
        # Call IDs identify C0 blocks; core IDs identify the tested constants.
        block_idx, value_idx = int(match[1]), int(match[2])
        identity = (value_idx, block_idx)
        if (
            identity in seen_records
            or not 0 <= block_idx < L1_BLOCK_COUNT
            or not 0 <= value_idx < len(value_bits)
        ):
            raise AssertionError(f"duplicate or unexpected C0 record {identity}")
        seen_records.add(identity)
        value, bits = value_bits[value_idx]
        words = [int(v.strip()) for v in match[3].split(",")]
        if len(words) != 8:
            raise AssertionError(f"incomplete C0 block {block_idx}")
        c0_column, fractal_row = divmod(block_idx, L1_ROWS)
        # Test block starts to model C0 alignment independently of the device helper.
        filled = (
            row_extent > 0
            and col_extent > 0
            and row_start <= fractal_row < row_start + row_extent
            and col_start <= c0_column * c0_elements < col_start + col_extent
        )
        if not filled:
            if words != [SENTINEL] * 8:
                raise AssertionError(
                    f"neighbor block {block_idx} was overwritten: {words}"
                )
            continue
        if element_bits == 16:
            expected = [bits | (bits << 16)] * 8
        else:
            expected = [bits] * 8
        if isinstance(value, float) and math.isnan(value):
            exponent_mask, mantissa_mask = {
                "f16": (0x7C00, 0x03FF),
                "bf16": (0x7F80, 0x007F),
                "f32": (0x7F800000, 0x007FFFFF),
            }[dtype_name]
            elements = (
                words
                if element_bits == 32
                else [half for word in words for half in (word & 0xFFFF, word >> 16)]
            )
            if any(
                v & exponent_mask != exponent_mask or not v & mantissa_mask
                for v in elements
            ):
                raise AssertionError(
                    f"invalid NaN pattern in C0 block {block_idx}: {words}"
                )
        elif words != expected:
            raise AssertionError(
                f"{dtype_name}/{layout} C0 block {block_idx}: {words} != {expected}"
            )


def run(args: argparse.Namespace) -> int:
    from catlass.base_dsl.runtime.ascend_debug_fifo import _debug_bus_support_error

    reason = _debug_bus_support_error()
    if reason:
        print(f"SKIP: L1 debug readback unavailable: {reason}")
        return 77

    import torch
    import torch_npu  # noqa: F401
    from catlass.tla.runtime import from_dlpack

    torch.npu.set_device(args.device)
    input_tensor = torch.full(
        (L1_ROWS, L1_C0_COLUMNS * 8),
        SENTINEL,
        dtype=torch.int32,
        device=f"npu:{args.device}",
    )
    gm_input = from_dlpack(input_tensor.view(torch.int8), layout_tag=tla.arch.RowMajor)
    dtype_names = tuple(DTYPE_SPECS) if args.all_dtypes else (args.dtype,)
    layouts = ("zn", "nz") if args.all_layouts else (args.layout,)
    case_names = CASES if args.all_cases else (args.case,)
    passed = 0
    for dtype_name in dtype_names:
        dtype, element_bits = DTYPE_SPECS[dtype_name]
        c0_elements = 256 // element_bits
        for layout in layouts:
            layout_tag = tla.arch.zN if layout == "zn" else tla.arch.nZ
            value_bits = _fill_values(dtype_name)
            if not args.all_values:
                # One nonzero pattern is enough for the default on-device check.
                value_bits = value_bits[1:2]
            values = tuple(value for value, _ in value_bits)
            compile_args = (
                gm_input,
                *(tla.Int32(v) for v in _case_region("full", c0_elements)),
            )
            artifact = tla.compile(
                fill_l1_kernel,
                *compile_args,
                dtype,
                layout_tag,
                c0_elements,
                values,
                options="--npu-arch 3510",
            )
            for case_name in case_names:
                region = _case_region(case_name, c0_elements)
                runtime_args = (gm_input, *(tla.Int32(v) for v in region))
                captured = StringIO()
                with redirect_stdout(captured):
                    artifact(*runtime_args, block_num=len(values))
                    torch.npu.synchronize()
                _verify_output(
                    captured.getvalue(), dtype_name, layout, region, value_bits
                )
                logical_shape = (
                    (L1_ROWS, L1_C0_COLUMNS * c0_elements)
                    if layout == "zn"
                    else (L1_C0_COLUMNS * c0_elements, L1_ROWS)
                )
                for value, bits in value_bits:
                    passed += 1
                    pattern = (
                        "NaN (payload unspecified)"
                        if isinstance(value, float) and math.isnan(value)
                        else f"0x{bits:08x}"
                    )
                    print(
                        f"PASS dtype={dtype_name} layout={layout} case={case_name} "
                        f"shape={logical_shape} value={value!r} bits={pattern}"
                    )
    print(f"L1 Fill: {passed} passed, 0 failed, 0 skipped")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dtype", choices=DTYPE_SPECS, default="f16")
    parser.add_argument("--all-dtypes", action="store_true")
    parser.add_argument(
        "--all-values",
        action="store_true",
        help="Check all constants; the default checks one nonzero value per dtype.",
    )
    parser.add_argument("--layout", choices=("zn", "nz"), default="zn")
    parser.add_argument("--all-layouts", action="store_true")
    parser.add_argument("--case", choices=CASES, default="full")
    parser.add_argument("--all-cases", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
