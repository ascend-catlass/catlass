# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Device regression for FP4 bitwise ops on packed pair carriers.

``f4e2m1x2`` / ``f4e1m2x2`` hold two FP4 values per byte, so a vector lane is
one byte and a 256-byte register holds 256 pairs. Bitwise ops act on the whole
pair, and the generic byte-granular load/store and predicate paths carry them:
partial tiles keep the bytes past their valid pairs untouched.

Needs an Ascend 950PR device, so pytest does not collect this directory -- see
tests/conftest.py. Run it directly on a box that has one, passing a free card:
``python fp4x2_bitwise_ops.py --batch-run --device <id>``.
"""

from __future__ import annotations

import argparse
import itertools
from typing import Any

import numpy as np

from catlass import tla
from catlass.tla.runtime import from_dlpack, make_fake_tensor

REGISTER_PAIRS = 256
TOTAL_PAIRS = 4 * REGISTER_PAIRS
ALL_DTYPES = ("f4e2m1x2", "f4e1m2x2")
ALL_OPS = (
    "bitwise_or",
    "bitwise_and",
    "bitwise_xor",
    "bitwise_not",
    "chained",
    "mixed",
)
BATCH_VALID_PAIRS = (1, 2, 63, 64, 255, 256)
BATCH_PAIR_OFFSETS = (0, 32)
_FP4_PAIR_DTYPE = tla.Float4E2M1x2
_CHAINED = False
_OP = "bitwise_or"
_VALID_PAIRS = 256
_PAIR_OFFSET = 0
_SENTINEL = 0xA5


@tla.kernel
def fp4x2_bitwise_ops(
    mem_a: tla.Tensor,
    mem_b: tla.Tensor,
    mem_c: tla.Tensor,
    mem_out: tla.Tensor,
) -> None:
    loaded = tla.flag("fp4x2_loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    done = tla.flag("fp4x2_done", tla.pipes.VECTOR, tla.pipes.MTE3)
    a_gm = tla.tile_view(mem_a, tla.make_shape(TOTAL_PAIRS), tla.make_coord(0))
    b_gm = tla.tile_view(mem_b, tla.make_shape(TOTAL_PAIRS), tla.make_coord(0))
    c_gm = tla.tile_view(mem_c, tla.make_shape(TOTAL_PAIRS), tla.make_coord(0))
    out_gm = tla.tile_view(mem_out, tla.make_shape(TOTAL_PAIRS), tla.make_coord(0))

    # Copies stay byte-denominated: one pair carrier is exactly one byte.
    a_ptr = tla.allocate(TOTAL_PAIRS, tla.Int8, tla.AddressSpace.ub, 256)
    b_ptr = tla.allocate(TOTAL_PAIRS, tla.Int8, tla.AddressSpace.ub, 256)
    c_ptr = tla.allocate(TOTAL_PAIRS, tla.Int8, tla.AddressSpace.ub, 256)
    out_ptr = tla.allocate(TOTAL_PAIRS, tla.Int8, tla.AddressSpace.ub, 256)
    a_bytes = tla.make_tensor_like(a_ptr, a_gm, tla.arch.RowMajor)
    b_bytes = tla.make_tensor_like(b_ptr, b_gm, tla.arch.RowMajor)
    c_bytes = tla.make_tensor_like(c_ptr, c_gm, tla.arch.RowMajor)
    out_bytes = tla.make_tensor_like(out_ptr, out_gm, tla.arch.RowMajor)
    # Compute views count packed pairs; recast_ptr performs no conversion.
    tile_layout = tla.make_layout(
        tla.make_shape(REGISTER_PAIRS),
        tla.make_stride(1),
        origin_shape=tla.make_shape(_VALID_PAIRS),
    )

    with tla.vector():
        tla.copy(a_bytes, a_gm)
        tla.copy(b_bytes, b_gm)
        tla.copy(c_bytes, c_gm)
        tla.copy(out_bytes, out_gm)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        with tla.vec.func(mode="simd"):
            # One register covers 256 pairs; the offset counts pairs, not bytes.
            for chunk in tla.range_constexpr(0, 2):
                coord = tla.make_coord(_PAIR_OFFSET + chunk * REGISTER_PAIRS)
                a_tile = tla.make_tensor(
                    tla.recast_ptr(a_ptr, dtype=_FP4_PAIR_DTYPE), tile_layout, coord
                )
                b_tile = tla.make_tensor(
                    tla.recast_ptr(b_ptr, dtype=_FP4_PAIR_DTYPE), tile_layout, coord
                )
                out_tile = tla.make_tensor(
                    tla.recast_ptr(out_ptr, dtype=_FP4_PAIR_DTYPE), tile_layout, coord
                )
                lhs = a_tile.load()
                if tla.const_expr(_OP == "bitwise_not"):
                    result = tla.bitwise_not(lhs)
                else:
                    rhs = b_tile.load()
                    if tla.const_expr(_OP == "bitwise_and"):
                        result = tla.bitwise_and(lhs, rhs)
                    elif tla.const_expr(_OP == "bitwise_xor"):
                        result = tla.bitwise_xor(lhs, rhs)
                    elif tla.const_expr(_OP == "mixed"):
                        result = tla.bitwise_or(
                            tla.bitwise_not(
                                tla.bitwise_xor(tla.bitwise_and(lhs, rhs), lhs)
                            ),
                            rhs,
                        )
                    else:
                        result = tla.bitwise_or(lhs, rhs)
                if tla.const_expr(_CHAINED):
                    c_tile = tla.make_tensor(
                        tla.recast_ptr(c_ptr, dtype=_FP4_PAIR_DTYPE), tile_layout, coord
                    )
                    result = tla.bitwise_or(result, c_tile.load())
                out_tile.store(result)
        tla.set_flag(done)
        tla.wait_flag(done)
        tla.copy(out_gm, out_bytes)
        tla.pipe_barrier(tla.pipes.ALL)


def make_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lane = np.arange(TOTAL_PAIRS, dtype=np.uint32)
    lo = lane & 15
    hi = (lane >> 4) & 15
    # In each register both nibble positions cover every 16x16 input pair;
    # independent permutations make high/low outputs differ, exposing mixing.
    lhs = (lo | ((hi ^ 3) << 4)).astype(np.uint8)
    rhs = (hi | (((lo * 5 + 7) & 15) << 4)).astype(np.uint8)
    third = (((lane * 7 + 3) & 15) | (((lane * 5 + 1) & 15) << 4)).astype(np.uint8)
    return lhs, rhs, third


def expected(
    lhs: np.ndarray,
    rhs: np.ndarray,
    third: np.ndarray,
    *,
    valid_pairs: int = 256,
    pair_offset: int = 0,
    op: str = "bitwise_or",
) -> np.ndarray:
    result = np.full(TOTAL_PAIRS, _SENTINEL, dtype=np.uint8)
    chained = op == "chained"
    if chained:
        op = "bitwise_or"
    if op == "bitwise_not":
        payload = np.bitwise_not(lhs)
    elif op == "mixed":
        payload = np.bitwise_or(
            np.bitwise_not(np.bitwise_xor(np.bitwise_and(lhs, rhs), lhs)), rhs
        )
    else:
        binary = {
            "bitwise_or": np.bitwise_or,
            "bitwise_and": np.bitwise_and,
            "bitwise_xor": np.bitwise_xor,
        }[op]
        payload = binary(lhs, rhs)
    if chained:
        payload = np.bitwise_or(payload, third)
    for chunk in range(2):
        start = pair_offset + chunk * REGISTER_PAIRS
        result[start : start + valid_pairs] = payload[start : start + valid_pairs]
    return result


def _configure(dtype: str, op: str, valid_pairs: int, pair_offset: int) -> None:
    global _FP4_PAIR_DTYPE, _CHAINED, _VALID_PAIRS, _PAIR_OFFSET, _OP
    _FP4_PAIR_DTYPE = {
        "f4e2m1x2": tla.Float4E2M1x2,
        "f4e1m2x2": tla.Float4E1M2x2,
    }[dtype]
    # "chained" is the three-input OR chain, not a fourth bitwise operation.
    _CHAINED = op == "chained"
    _OP = "bitwise_or" if _CHAINED else op
    _VALID_PAIRS = valid_pairs
    _PAIR_OFFSET = pair_offset


def _compile(args: argparse.Namespace, tensors: tuple[Any, ...]) -> Any:
    return tla.compile(fp4x2_bitwise_ops, *tensors, options=f"--npu-arch {args.arch}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("op", nargs="?", choices=ALL_OPS, default="bitwise_or")
    parser.add_argument("--dtype", choices=ALL_DTYPES, default=ALL_DTYPES[0])
    parser.add_argument("--all-dtypes", action="store_true")
    parser.add_argument(
        "--batch-run",
        action="store_true",
        help="Test both formats and all bitwise operations at all boundary lengths and offsets",
    )
    parser.add_argument("--valid-pairs", type=int, default=256)
    parser.add_argument("--pair-offset", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--arch", default="3510")
    args = parser.parse_args()
    if not 1 <= args.valid_pairs <= REGISTER_PAIRS:
        parser.error(f"--valid-pairs must be between 1 and {REGISTER_PAIRS}")
    if (
        args.pair_offset < 0
        or args.pair_offset % 32
        or args.pair_offset > TOTAL_PAIRS - 2 * REGISTER_PAIRS
    ):
        parser.error(
            "--pair-offset must be a non-negative multiple of 32 with room for two register reads"
        )
    dtypes = ALL_DTYPES if args.all_dtypes else (args.dtype,)
    cases = (
        tuple(
            itertools.product(
                ALL_DTYPES,
                ALL_OPS,
                BATCH_VALID_PAIRS,
                BATCH_PAIR_OFFSETS,
            )
        )
        if args.batch_run
        else tuple(
            (dtype, args.op, args.valid_pairs, args.pair_offset) for dtype in dtypes
        )
    )
    if args.compile_only:
        tensors = tuple(
            make_fake_tensor(tla.Int8, (TOTAL_PAIRS,), (1,)) for _ in range(4)
        )
        for dtype, op, valid_pairs, pair_offset in cases:
            _configure(dtype, op, valid_pairs, pair_offset)
            artifact = _compile(args, tensors)
            print(
                f"compile_ok=True dtype={dtype} op={op} valid_pairs={valid_pairs} pair_offset={pair_offset} kernel.o={artifact.kernel_binary_path}"
            )
        return 0

    import torch
    import torch_npu  # noqa: F401 -- register the backend in the device entry

    torch.npu.set_device(args.device)
    inputs = make_inputs()
    for dtype, op, valid_pairs, pair_offset in cases:
        golden = expected(
            *inputs,
            valid_pairs=valid_pairs,
            pair_offset=pair_offset,
            op=op,
        )
        _configure(dtype, op, valid_pairs, pair_offset)
        host = (*inputs, np.full(TOTAL_PAIRS, _SENTINEL, dtype=np.uint8))
        device = tuple(torch.from_numpy(x.view(np.int8).copy()).to("npu") for x in host)
        tensors = tuple(from_dlpack(x, layout_tag=tla.arch.RowMajor) for x in device)
        artifact = _compile(args, tensors)
        artifact(*tensors, block_num=1)
        torch.npu.synchronize()
        actual = device[-1].cpu().numpy().view(np.uint8)
        if not np.array_equal(actual, golden):
            bad = np.flatnonzero(actual != golden)
            index = int(bad[0])
            print(
                f"dtype={dtype} op={op} valid_pairs={valid_pairs} pair_offset={pair_offset} mismatch_byte={index} actual={actual[index]} expected={golden[index]}"
            )
            return 1
        print(
            f"compile_ok=True launch_ok=True device={args.device} dtype={dtype} op={op} valid_pairs={valid_pairs} pair_offset={pair_offset} bytes_equal=True"
        )
    if args.batch_run:
        print(f"batch_ok=True cases={len(cases)} device={args.device}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
