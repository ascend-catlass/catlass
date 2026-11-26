# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""``tla.fast_div_get_divisor`` end-to-end sample.

For a fixed unsigned divisor it precomputes the asc-devkit ``(magic, shift)``
pair that fast division consumes as ``(x + mulhi_w(x, magic)) >> shift``. The 32-bit
variant evaluates the closed form directly; the 64-bit variant runs the software
long division. Both run on the scalar unit inside a ``tla.vector()`` region.

Each launch reads the divisors from GM (runtime path) and also computes one
compile-time literal, writing ``magic`` and ``shift`` back in the same unsigned
width. The host compares them against a Python mirror of the C++ helper.

Run on device:

    python tests/end_to_end/fast_div_get_divisor.py --device 0
"""

from __future__ import annotations

import argparse

import catlass.tla as tla

_INT32_MAX = (1 << 31) - 1
_INT64_MAX = (1 << 63) - 1
_MASK64 = (1 << 64) - 1

# Divisors covering 1, powers of two (the power-of-two shift path), odd values,
# the 32-bit boundary and the largest legal divisor of each variant.
_VALUES32 = (1, 2, 3, 5, 6, 7, 8, 16, 100, 1024, 65535, 1000003, _INT32_MAX)
_VALUES64 = (
    _INT32_MAX + 1,
    1 << 32,
    (1 << 40) + 1,
    3**20,
    (1 << 62) + 1,
    _INT64_MAX,
)

_LITERAL32 = _INT32_MAX
_LITERAL64 = _INT64_MAX


def _get_uint_div_magic(dividend: int, divisor: int) -> int:
    """Bit-exact mirror of the scalar long division."""
    quotient = 0
    remainder = dividend
    for _ in range(64):
        quotient = (quotient << 1) & _MASK64
        borrow = 1 if remainder & (1 << 63) else 0
        remainder = (remainder << 1) & _MASK64
        if borrow:
            remainder = (_MASK64 - divisor + 1 + remainder) & _MASK64
            quotient |= 1
        elif remainder >= divisor:
            remainder -= divisor
            quotient |= 1
    return (quotient + 1) & _MASK64


def _reference(divisor: int, width: int) -> tuple[int, int]:
    """Mirror of the scalar bitcode helper, in Python."""
    pos = divisor.bit_length()
    cnt1 = bin(divisor).count("1")
    shift = pos - 1 if cnt1 == 1 else pos
    if width == 32:
        magic = (((1 << 32) * ((1 << shift) - divisor)) // divisor + 1) & 0xFFFFFFFF
        return magic, shift
    if shift < 64:
        dividend = (1 << shift) - divisor
    elif shift == 64:
        dividend = _MASK64 - divisor + 1
    else:  # pragma: no cover - the supported divisor range never reaches this
        return 0, shift
    return _get_uint_div_magic(dividend, divisor), shift


@tla.kernel
def _fast_div32(src: tla.Tensor, magic: tla.Tensor, shift: tla.Tensor) -> None:
    with tla.vector():
        for i in range(len(_VALUES32)):
            fd = tla.fast_div_get_divisor(src[i])
            magic[i] = fd.magic
            shift[i] = fd.shift
        lit = tla.fast_div_get_divisor(tla.UInt32(_LITERAL32))
        magic[len(_VALUES32)] = lit.magic
        shift[len(_VALUES32)] = lit.shift
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def _fast_div64(src: tla.Tensor, magic: tla.Tensor, shift: tla.Tensor) -> None:
    with tla.vector():
        for i in range(len(_VALUES64)):
            fd = tla.fast_div_get_divisor(src[i])
            magic[i] = fd.magic
            shift[i] = fd.shift
        lit = tla.fast_div_get_divisor(tla.UInt64(_LITERAL64))
        magic[len(_VALUES64)] = lit.magic
        shift[len(_VALUES64)] = lit.shift
        tla.pipe_barrier(tla.pipes.ALL)


def _to_unsigned(bits: list[int], width: int) -> list[int]:
    mask = (1 << width) - 1
    return [value & mask for value in bits]


def _run_variant(torch, from_dlpack, width: int) -> bool:
    if width == 32:
        values = (*_VALUES32, _LITERAL32)
        dtype, signed, kernel = torch.uint32, torch.int32, _fast_div32
    else:
        values = (*_VALUES64, _LITERAL64)
        dtype, signed, kernel = torch.uint64, torch.int64, _fast_div64

    src_bits = torch.tensor(values, dtype=signed, device="npu")
    src = from_dlpack(src_bits.view(dtype), layout_tag=tla.arch.RowMajor)

    count = len(values)
    # NPU does not accept a uint32/uint64 allocation, so allocate the signed
    # carrier and hand the kernel a view with the unsigned element type.
    magic_bits = torch.zeros((count,), dtype=signed, device="npu")
    shift_bits = torch.zeros((count,), dtype=signed, device="npu")
    magic_t = from_dlpack(magic_bits.view(dtype), layout_tag=tla.arch.RowMajor)
    shift_t = from_dlpack(shift_bits.view(dtype), layout_tag=tla.arch.RowMajor)

    artifact = tla.compile(kernel, src, magic_t, shift_t, options="--npu-arch 3510")
    artifact(src, magic_t, shift_t, block_num=1)
    torch.npu.synchronize()

    expected = [_reference(value, width) for value in values]
    actual_magic = _to_unsigned(magic_bits.cpu().tolist(), width)
    actual_shift = _to_unsigned(shift_bits.cpu().tolist(), width)
    exp_magic = [m for m, _ in expected]
    exp_shift = [s for _, s in expected]
    ok = actual_magic == exp_magic and actual_shift == exp_shift
    print(f"fast_div width={width} values={count} passed={ok}")
    print(f"fast_div width={width} kernel.o path={artifact.kernel_binary_path}")
    if not ok:
        for value, (m, s), am, as_ in zip(
            values, expected, actual_magic, actual_shift
        ):
            if (m, s) != (am, as_):
                print(f"  divisor={value} expected=({m},{s}) actual=({am},{as_})")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    import torch
    import torch_npu  # noqa: F401

    from catlass.tla.runtime import from_dlpack

    torch.npu.set_device(args.device)
    failed = 0
    for width in (32, 64):
        failed += not _run_variant(torch, from_dlpack, width)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
