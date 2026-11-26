# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""SIMD fast ``divmod`` end-to-end sample (separate from ``fast_div_get_divisor``).

``fd = tla.fast_div_get_divisor(divisor)`` precomputes the asc-devkit magic/shift
pair; then ``x // fd`` and ``x % fd`` lower to the software-mulhi BC helpers
``_mlir_ciface_tla_simd_fast_{div,mod}_{32,64}``. Both the dividend and the
divisor must be at most ``INT32_MAX`` (32-bit) / ``INT64_MAX`` (64-bit), the
domain the add-form magic is exact on.

Run on device:

    python tests/end_to_end/fast_divmod_simd.py --device 0
"""

from __future__ import annotations

import argparse

import catlass.tla as tla

_INT32_MAX = (1 << 31) - 1
_INT64_MAX = (1 << 63) - 1


def _pairs(width: int) -> list[tuple[int, int]]:
    """(dividend, divisor) pairs covering powers of two, odd values and edges."""
    if width == 32:
        divisors = (1, 2, 3, 7, 8, 100, 1000003, _INT32_MAX)
        limit = _INT32_MAX
    else:
        divisors = (1, 2, _INT32_MAX + 1, 1 << 32, (1 << 40) + 1, 3**20, _INT64_MAX)
        limit = _INT64_MAX

    pairs: list[tuple[int, int]] = []
    for divisor in divisors:
        candidates = (0, 1, divisor - 1, divisor, divisor + 1, limit, limit // 2)
        for dividend in candidates:
            if 0 <= dividend <= limit:
                pairs.append((dividend, divisor))
    return pairs


_PAIRS32 = _pairs(32)
_PAIRS64 = _pairs(64)


@tla.kernel
def _divmod32(
    src_x: tla.Tensor, src_d: tla.Tensor, out_q: tla.Tensor, out_r: tla.Tensor
) -> None:
    with tla.vector():
        for i in range(len(_PAIRS32)):
            fd = tla.fast_div_get_divisor(src_d[i])
            out_q[i] = src_x[i] // fd
            out_r[i] = src_x[i] % fd
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def _divmod64(
    src_x: tla.Tensor, src_d: tla.Tensor, out_q: tla.Tensor, out_r: tla.Tensor
) -> None:
    with tla.vector():
        for i in range(len(_PAIRS64)):
            fd = tla.fast_div_get_divisor(src_d[i])
            out_q[i] = src_x[i] // fd
            out_r[i] = src_x[i] % fd
        tla.pipe_barrier(tla.pipes.ALL)


def _run_variant(torch, from_dlpack, width: int) -> bool:
    pairs = _PAIRS32 if width == 32 else _PAIRS64
    if width == 32:
        signed, unsigned, kernel = torch.int32, torch.uint32, _divmod32
    else:
        signed, unsigned, kernel = torch.int64, torch.uint64, _divmod64

    xs = [x for x, _ in pairs]
    ds = [d for _, d in pairs]
    count = len(pairs)

    # NPU does not accept uint allocations; allocate the signed carrier and hand
    # the kernel views with the unsigned element type.
    x_bits = torch.tensor(xs, dtype=signed, device="npu")
    d_bits = torch.tensor(ds, dtype=signed, device="npu")
    q_bits = torch.zeros((count,), dtype=signed, device="npu")
    r_bits = torch.zeros((count,), dtype=signed, device="npu")

    src_x = from_dlpack(x_bits.view(unsigned), layout_tag=tla.arch.RowMajor)
    src_d = from_dlpack(d_bits.view(unsigned), layout_tag=tla.arch.RowMajor)
    out_q = from_dlpack(q_bits.view(unsigned), layout_tag=tla.arch.RowMajor)
    out_r = from_dlpack(r_bits.view(unsigned), layout_tag=tla.arch.RowMajor)

    artifact = tla.compile(
        kernel, src_x, src_d, out_q, out_r, options="--npu-arch 3510"
    )
    artifact(src_x, src_d, out_q, out_r, block_num=1)
    torch.npu.synchronize()

    expected_q = [x // d for x, d in pairs]
    expected_r = [x % d for x, d in pairs]
    ok_q = q_bits.cpu().tolist() == expected_q
    ok_r = r_bits.cpu().tolist() == expected_r
    ok = ok_q and ok_r
    print(f"fast_divmod width={width} pairs={count} passed={ok}")
    print(f"fast_divmod width={width} kernel.o path={artifact.kernel_binary_path}")
    if not ok:
        actual_q = q_bits.cpu().tolist()
        actual_r = r_bits.cpu().tolist()
        for (x, d), eq, er, aq, ar in zip(
            pairs, expected_q, expected_r, actual_q, actual_r
        ):
            if (eq, er) != (aq, ar):
                print(f"  x={x} d={d} expected=({eq},{er}) actual=({aq},{ar})")
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
