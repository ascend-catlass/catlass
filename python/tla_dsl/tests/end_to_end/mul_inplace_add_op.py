# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Device checks for mul_inplace_add, SSA inputs and zeroed inactive lanes."""

from __future__ import annotations

import argparse
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

import catlass.tla as tla

GUARD = 32
DTYPES = {
    "f16": tla.Float16,
    "bf16": tla.BFloat16,
    "f32": tla.Float32,
    "i16": tla.Int16,
    "u16": tla.UInt16,
    "i32": tla.Int32,
    "u32": tla.UInt32,
}


def make_kernel(dtype, *, masked, source_lanes, sparse=False, iterations=1):
    lanes = 2048 // dtype.width
    capacity = GUARD + lanes + GUARD

    @tla.kernel
    def compute(
        accumulator: tla.Tensor,
        left: tla.Tensor,
        right: tla.Tensor,
        destination: tla.Tensor,
        saved: tla.Tensor,
        count: tla.Int32,
        branch: tla.Int32,
    ):
        acc_ptr = tla.allocate(capacity, dtype, tla.AddressSpace.ub, 256)
        left_ptr = tla.allocate(capacity, dtype, tla.AddressSpace.ub, 256)
        right_ptr = tla.allocate(capacity, dtype, tla.AddressSpace.ub, 256)
        out_ptr = tla.allocate(capacity, dtype, tla.AddressSpace.ub, 256)
        saved_ptr = tla.allocate(capacity, dtype, tla.AddressSpace.ub, 256)
        acc_ub = tla.make_tensor_like(acc_ptr, accumulator, tla.arch.RowMajor)
        left_ub = tla.make_tensor_like(left_ptr, left, tla.arch.RowMajor)
        right_ub = tla.make_tensor_like(right_ptr, right, tla.arch.RowMajor)
        out_ub = tla.make_tensor_like(out_ptr, destination, tla.arch.RowMajor)
        saved_ub = tla.make_tensor_like(saved_ptr, saved, tla.arch.RowMajor)
        layout = tla.make_layout(tla.make_shape(source_lanes), tla.make_stride(1))
        acc_tile = tla.make_tensor(acc_ptr + GUARD, layout)
        left_tile = tla.make_tensor(left_ptr + GUARD, layout)
        right_tile = tla.make_tensor(right_ptr + GUARD, layout)
        output_tile = tla.make_tensor(
            out_ptr + GUARD,
            tla.make_layout(tla.make_shape(source_lanes), tla.make_stride(1)),
        )
        old_tile = tla.make_tensor(
            saved_ptr + GUARD,
            tla.make_layout(tla.make_shape(source_lanes), tla.make_stride(1)),
        )
        ready = tla.flag("ready", tla.pipes.MTE2, tla.pipes.VECTOR)
        done = tla.flag("done", tla.pipes.VECTOR, tla.pipes.MTE3)
        with tla.vector():
            tla.copy(acc_ub, accumulator)
            tla.copy(left_ub, left)
            tla.copy(right_ub, right)
            tla.copy(out_ub, destination)
            tla.copy(saved_ub, saved)
            tla.set_flag(ready)
            tla.wait_flag(ready)
            with tla.vec.func(mode="simd"):
                acc = acc_tile.load()
                lhs = left_tile.load()
                rhs = right_tile.load()
                if tla.const_expr(masked):
                    active, _ = tla.update_mask(count, dtype=dtype)
                    if tla.const_expr(sparse):
                        active = tla.bitwise_and(
                            active, tla.create_mask(pattern=tla.mask.M3, dtype=dtype)
                        )
                    result = tla.arch.mul_inplace_add(acc, lhs, rhs, mask=active)
                else:
                    result = tla.arch.mul_inplace_add(acc, lhs, rhs)
                if tla.const_expr(iterations > 1):
                    for step in tla.range(iterations - 1):
                        if branch > 0:
                            step_mask, _ = tla.update_mask(count + step, dtype=dtype)
                            result = tla.arch.mul_inplace_add(
                                result, lhs, rhs, mask=step_mask
                            )
                output_tile.store(result)
                # Old acc is used after the operation to test preservation.
                old_tile.store(acc)
            tla.set_flag(done)
            tla.wait_flag(done)
            tla.copy(destination, out_ub)
            tla.copy(saved, saved_ub)
            tla.pipe_barrier(tla.pipes.ALL)

    return compute


def _rounded_fma(acc, lhs, rhs, finfo):
    # Binary64 represents the product of two finite FP32 (or narrower) inputs exactly.
    product = lhs * rhs
    total = product + acc
    if total == 0.0 or not math.isfinite(total):
        return total  # Preserve IEEE signed zero, infinity and NaN.

    magnitude = abs(Fraction(product) + Fraction(acc))
    # The denominator is a power of two. Subnormals use the spacing at min normal.
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    spacing = Fraction(max(math.ldexp(1.0, exponent), finfo.tiny) * finfo.eps)
    nearest = float(round(magnitude / spacing) * spacing)  # Ties to even.
    return math.copysign(nearest if nearest <= finfo.max else math.inf, total)


def reference(accumulator, left, right, token):
    import torch

    if token.startswith(("i", "u")):
        bits = int(token[1:])
        modulus = 1 << bits
        rows = [
            (int(a) + int(b) * int(c)) % modulus
            for a, b, c in zip(accumulator.tolist(), left.tolist(), right.tolist())
        ]
        if token.startswith("i"):
            half = modulus >> 1
            rows = [value - modulus if value >= half else value for value in rows]
        return torch.tensor(rows, dtype=accumulator.dtype)
    finfo = torch.finfo(accumulator.dtype)
    rows = [
        _rounded_fma(a, b, c, finfo)
        for a, b, c in zip(accumulator.tolist(), left.tolist(), right.tolist())
    ]
    return torch.tensor(rows, dtype=accumulator.dtype)


def main():
    import torch
    import torch_npu  # noqa: F401

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--dtype", choices=DTYPES, default="f32")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    token = args.dtype
    dtype = DTYPES[token]
    torch_dtype = {
        "f16": torch.float16,
        "bf16": torch.bfloat16,
        "f32": torch.float32,
        "i16": torch.int16,
        "u16": torch.uint16,
        "i32": torch.int32,
        "u32": torch.uint32,
    }[token]
    torch.npu.set_device(args.device)
    device = f"npu:{args.device}"
    lanes = 2048 // dtype.width
    capacity = 2 * GUARD + lanes
    rows = []
    # Reuse each compiled object with updated values/counts. A physical full
    # output store checks ZEROING; no output mask can hide inactive lanes.
    for kind, source_lanes, iterations in (
        ("default", lanes, 1),
        ("tail", 13, 1),
        ("prefix", lanes, 1),
        ("sparse", lanes, 1),
        ("flow", lanes, 3),
    ):
        kernel = make_kernel(
            dtype,
            masked=kind in ("prefix", "sparse"),
            source_lanes=source_lanes,
            sparse=kind == "sparse",
            iterations=iterations,
        )
        compiled = None
        for repeat in range(2 if kind != "prefix" else 3):
            generator = np.random.default_rng(3510 + repeat)
            values = [
                torch.tensor(generator.integers(-40, 40, capacity), dtype=torch.float32)
                for _ in range(3)
            ]
            if token.startswith("u"):
                values = [value.abs() for value in values]
            source_cpu = [value.to(torch_dtype) for value in values]
            if token.startswith(("i", "u")):
                bits = int(token[1:])
                source_cpu[0][GUARD] = (1 << (bits - (token[0] == "i"))) - 1
                source_cpu[1][GUARD] = 2
                source_cpu[2][GUARD] = 3
                # High-bit products exercise signed/unsigned overflow alike.
                limit = (1 << (bits - (token[0] == "i"))) - 1
                for offset, value in enumerate((limit, limit - 1, 1 << (bits - 2), 0)):
                    source_cpu[1][GUARD + 1 + offset] = value
                    source_cpu[2][GUARD + 1 + offset] = value

            else:
                mantissa = {"f32": 23, "f16": 10, "bf16": 7}[token]
                special = [
                    0.0,
                    -0.0,
                    float("nan"),
                    float("inf"),
                    -float("inf"),
                    torch.finfo(torch_dtype).max,
                    torch.finfo(torch_dtype).tiny,
                    2.0 ** (-mantissa - ({"f32": 126, "f16": 14, "bf16": 126}[token])),
                ]
                for source in source_cpu:
                    source[GUARD + 1 : GUARD + 1 + len(special)] = torch.tensor(
                        special, dtype=torch_dtype
                    )
                source_cpu[0][GUARD] = -1
                source_cpu[1][GUARD] = 1 + 2**-mantissa
                source_cpu[2][GUARD] = 1 - 2**-mantissa
                if token in ("f16", "bf16"):
                    # The product is a midpoint. The small addends would vanish in FP32.
                    start = GUARD + 1 + len(special)
                    midpoints = slice(start, start + 3)
                    source_cpu[0][midpoints] = torch.tensor(
                        [-(2**-14), 0.0, 2**-14], dtype=torch_dtype
                    )
                    source_cpu[1][midpoints] = 1 + 2**-mantissa
                    source_cpu[2][midpoints] = 1536
            initial = torch.full((capacity,), 7, dtype=torch_dtype)
            inputs = [value.to(device) for value in source_cpu]
            output = initial.to(device)
            saved = initial.to(device)
            views = [
                tla.from_dlpack(value, element_type=dtype, layout_tag=tla.arch.RowMajor)
                for value in (*inputs, output, saved)
            ]
            count = [0, 13, lanes][repeat] if kind == "prefix" else lanes - 3
            branch = repeat
            if compiled is None:
                compiled = tla.compile(kernel, *views, count, branch)
            compiled(*views, count, branch, block_num=1)
            torch.npu.synchronize()
            acc, left, right = [value[GUARD : GUARD + lanes] for value in source_cpu]
            expected = reference(acc, left, right, token)
            active = np.arange(lanes) < (
                count if kind in ("prefix", "sparse") else source_lanes
            )
            if kind == "sparse":
                active &= np.arange(lanes) % 3 == 0
            # Indexing unsupported unsigned Torch dtypes via host raw arrays.
            expected_np = (
                expected.float().numpy() if token == "bf16" else expected.numpy()
            )
            expected_np[~active] = 0
            expected = torch.from_numpy(expected_np).to(torch_dtype)
            if kind == "flow" and branch > 0:
                # The first loop iteration clears the tail. The next reactivates
                # one lane, which must accumulate from zero.
                for step in range(iterations - 1):
                    expected = reference(expected, left, right, token)
                    expected[count + step :] = 0
            expected_output = initial.clone()
            expected_output[GUARD : GUARD + source_lanes] = expected[:source_lanes]
            actual = output.cpu()
            if token.startswith("f") or token == "bf16":
                torch.testing.assert_close(
                    actual.float(),
                    expected_output.float(),
                    rtol=0,
                    atol=0,
                    equal_nan=True,
                )
                zeros = expected_output.float() == 0
                torch.testing.assert_close(
                    torch.signbit(actual.float()[zeros]),
                    torch.signbit(expected_output.float()[zeros]),
                )
            else:
                np.testing.assert_array_equal(actual.numpy(), expected_output.numpy())
            expected_saved = initial.clone()
            expected_saved[GUARD : GUARD + source_lanes] = acc[:source_lanes]
            np.testing.assert_array_equal(
                saved.cpu().view(torch.uint8).numpy(),
                expected_saved.view(torch.uint8).numpy(),
            )
            for actual_input, original in zip(inputs, source_cpu):
                np.testing.assert_array_equal(
                    actual_input.cpu().view(torch.uint8).numpy(),
                    original.view(torch.uint8).numpy(),
                )
            rows.append(
                {
                    "case": kind,
                    "repeat": repeat,
                    "count": count,
                    "dtype": token,
                    "cache_key": compiled.cache_key,
                }
            )
        print(f"{token} {kind}: passed", flush=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"passed={len(rows)} dtype={token}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
