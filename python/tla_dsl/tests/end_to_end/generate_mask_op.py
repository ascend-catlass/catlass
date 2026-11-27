# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end check for generate_mask: register data block -> predicate lanes.

The kernel loads one full register, turns one data block of it into a mask with
``VectorSSA.generate_mask``, and selects between two inputs under that mask. The
register is a bit array ordered by element then by in-element bit, so ``offset``
picks a ``count``-bit block (64 bits for b32, 128 for b16) and lane ``L`` of the
mask is that block's bit ``L``. The CPU reference slices the same way.

``offset`` is a compile-time constant (the instruction encodes it as an immediate), so the
driver compiles one kernel per offset rather than dispatching on a runtime id.

Example: python generate_mask_op.py --run --dtype f32 --case low --device 0
Example: python generate_mask_op.py --batch-run --device 0
"""

from __future__ import annotations

import argparse
from typing import Any

import catlass.tla as tla

_CASES = ("low", "high", "middle")
_VECTOR_ELE = 64
_DTYPE: Any = tla.Float32
_OFFSET = 0
_OFFSETS = {"low": 0, "middle": 4, "high": 31}
_OFFSETS_F16 = {"low": 0, "middle": 4, "high": 15}


@tla.kernel
def generate_mask_op(mem_a: tla.Tensor, mem_b: tla.Tensor, mem_y: tla.Tensor) -> None:
    """Select ``a`` where the generated mask is active, else ``b``.

    ``offset`` is a compile-time ``Constexpr`` (the instruction's ``part`` is an immediate),
    so each offset is its own kernel specialization; the driver compiles one
    kernel per case rather than dispatching on a runtime value.
    """
    loaded = tla.flag("loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    done = tla.flag("done", tla.pipes.VECTOR, tla.pipes.MTE3)
    a_gm = tla.tile_view(mem_a, tla.make_shape(_VECTOR_ELE), tla.make_coord(0))
    b_gm = tla.tile_view(mem_b, tla.make_shape(_VECTOR_ELE), tla.make_coord(0))
    y_gm = tla.tile_view(mem_y, tla.make_shape(_VECTOR_ELE), tla.make_coord(0))
    a_ptr = tla.allocate(_VECTOR_ELE, _DTYPE, tla.AddressSpace.ub, 256)
    b_ptr = tla.allocate(_VECTOR_ELE, _DTYPE, tla.AddressSpace.ub, 256)
    y_ptr = tla.allocate(_VECTOR_ELE, _DTYPE, tla.AddressSpace.ub, 256)
    a_ub = tla.make_tensor_like(a_ptr, a_gm, tla.arch.RowMajor)
    b_ub = tla.make_tensor_like(b_ptr, b_gm, tla.arch.RowMajor)
    y_ub = tla.make_tensor_like(y_ptr, y_gm, tla.arch.RowMajor)

    with tla.vector():
        tla.copy(a_ub, a_gm)
        tla.copy(b_ub, b_gm)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        with tla.vec.func(mode="simd"):
            a_tile = tla.tile_view(a_ub, tla.make_shape(_VECTOR_ELE), tla.make_coord(0))
            b_tile = tla.tile_view(b_ub, tla.make_shape(_VECTOR_ELE), tla.make_coord(0))
            y_tile = tla.tile_view(y_ub, tla.make_shape(_VECTOR_ELE), tla.make_coord(0))
            a = a_tile.load()
            b = b_tile.load()
            mask = a.generate_mask(_OFFSET)
            y_tile.store(tla.where(mask, a, b))
        tla.set_flag(done)
        tla.wait_flag(done)
        tla.copy(y_gm, y_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def _offset_for(case: str) -> int:
    table = _OFFSETS_F16 if _DTYPE is tla.Float16 else _OFFSETS
    return table[case]


def _set_config(dtype: str, case: str) -> None:
    global _VECTOR_ELE, _DTYPE, _OFFSET
    _DTYPE = {"f16": tla.Float16, "f32": tla.Float32}[dtype]
    _VECTOR_ELE = 128 if dtype == "f16" else 64
    _OFFSET = _offset_for(case)


def _int_view(tensor: Any, dtype: str) -> Any:
    """Reinterpret a float tensor as its same-width integer bit pattern."""
    import torch

    return tensor.view(torch.int16 if dtype == "f16" else torch.int32)


def _bit_patterns(dtype: str) -> Any:
    """Deterministic finite bit patterns whose every bit varies across lanes.

    Building the operands as raw bit patterns (then viewing them as floats)
    keeps the CPU reference exact: the tested bit is a literal register bit, not
    a property of some float value. Exponent fields stay in a finite range and
    ``b`` flips one mantissa bit of ``a`` so the two operands never coincide.
    """
    import torch

    count = 128 if dtype == "f16" else 64
    lanes = torch.arange(count, dtype=torch.int64)
    mantissa_mask = 0x03FF if dtype == "f16" else 0x007FFFFF
    mantissa = (lanes * 2654435761) & mantissa_mask
    sign = (lanes & 1) << (15 if dtype == "f16" else 31)
    if dtype == "f16":
        exponent = (12 + (lanes * 7) % 6) << 10  # 12..17, finite
    else:
        exponent = (117 + (lanes * 37) % 21) << 23  # 117..137, finite
    a_bits = exponent | mantissa | sign
    # Flip one mantissa bit: keeps the value finite and guarantees a != b.
    b_bits = a_bits ^ (0x0100 if dtype == "f16" else 0x00010000)
    as_int = {"f16": torch.int16, "f32": torch.int32}[dtype]
    as_float = {"f16": torch.float16, "f32": torch.float32}[dtype]
    return a_bits.to(as_int).view(as_float), b_bits.to(as_int).view(as_float)


def _expected(a: Any, b: Any, dtype: str, offset: int) -> Any:
    """CPU reference: one data block of the register drives the mask.

    The register is a bit array ordered by element then by in-element bit (LSB
    first). With ``count`` elements of ``width`` bits, ``offset`` selects the
    ``count``-bit block starting at register bit ``count * offset``; output lane
    ``L`` is that block's bit ``L``. The comparison runs on integer bit patterns
    so the float value never round-trips.
    """
    import torch

    width = 16 if dtype == "f16" else 32
    count = 128 if dtype == "f16" else 64
    raw = _int_view(a, dtype).to(torch.int64)
    reg_bit = count * offset + torch.arange(count, dtype=torch.int64)
    bits = (raw[reg_bit // width] >> (reg_bit % width)) & 1
    return torch.where(bits.bool(), a, b)


def _run_case(dtype: str, case: str) -> bool:
    """Compile one kernel for one dtype/offset pair and check its output.

    Each offset is a separate specialization: ``generate_mask``'s offset is a
    compile-time constant that must reach the instruction as an immediate, so the driver
    cannot pick the offset from a runtime block id.
    """
    import torch

    _set_config(dtype, case)
    count = 128 if dtype == "f16" else 64
    offset = _offset_for(case)
    a, b = _bit_patterns(dtype)
    expected = _expected(a, b, dtype, offset)
    x_a = a.to("npu")
    x_b = b.to("npu")
    y = torch.full((count,), 123.0, dtype=a.dtype, device="npu")
    tla_a = tla.from_dlpack(x_a, layout_tag=tla.arch.RowMajor)
    tla_b = tla.from_dlpack(x_b, layout_tag=tla.arch.RowMajor)
    tla_y = tla.from_dlpack(y, layout_tag=tla.arch.RowMajor)
    artifact = tla.compile(generate_mask_op, tla_a, tla_b, tla_y, options="--npu-arch 3510")
    artifact(tla_a, tla_b, tla_y, block_num=1)
    torch.npu.synchronize()

    actual = y.cpu()
    ok = torch.equal(actual, expected)
    print(f"op=generate_mask dtype={dtype} case={case} offset={offset} passed={ok}")
    if not ok:
        print(f"result   = {actual.tolist()}")
        print(f"expected = {expected.tolist()}")
    print(f"kernel.o path={artifact.kernel_binary_path}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true", help="run one dtype/offset case")
    mode.add_argument(
        "--batch-run", action="store_true",
        help="run all offsets; both dtypes unless --dtype is set",
    )
    parser.add_argument("--dtype", choices=("f16", "f32"))
    parser.add_argument("--case", choices=_CASES)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if args.batch_run and args.case is not None:
        parser.error("--case cannot be combined with --batch-run")

    import torch
    import torch_npu  # noqa: F401 -- registers the NPU backend

    torch.npu.set_device(args.device)
    if args.batch_run:
        dtypes = (args.dtype,) if args.dtype else ("f16", "f32")
        failed = 0
        for dtype in dtypes:
            for case in _CASES:
                failed += not _run_case(dtype, case)
        total = len(dtypes) * len(_CASES)
        print(
            f"batch summary: passed={total - failed} failed={failed} "
            f"dtypes={len(dtypes)} total={total}"
        )
        return 0 if failed == 0 else 1
    return 0 if _run_case(args.dtype or "f32", args.case or _CASES[0]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
