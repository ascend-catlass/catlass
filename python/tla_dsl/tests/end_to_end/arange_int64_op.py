# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compile and verify both orders of tla.arange with one Int64 register."""

from __future__ import annotations

import argparse

import catlass.tla as tla
from catlass.tla.runtime import from_dlpack

LANES = 32
INT64_SIGN = 1 << 63
BASES = (0, -13, (1 << 32) - 7, -(1 << 40), INT64_SIGN - 8, -INT64_SIGN)


@tla.kernel
def arange_int64(
    dst: tla.Tensor, base: tla.Int64, order: tla.Constexpr, literal: tla.Constexpr
) -> None:
    ub_ptr = tla.allocate(LANES, tla.Int64, tla.AddressSpace.ub, 256)
    ub = tla.make_tensor_like(ub_ptr, dst, tla.arch.RowMajor)
    done = tla.flag("arange_done", tla.pipes.VECTOR, tla.pipes.MTE3)
    with tla.vector():
        with tla.vec.func(mode="simd"):
            if tla.const_expr(literal):
                values = tla.arange((1 << 32) - 7, order=order, dtype=tla.Int64)
            else:
                values = tla.arange(base, order=order, dtype=tla.Int64)
            ub.store(values)
        tla.set_flag(done)
        tla.wait_flag(done)
        tla.copy(dst, ub)


def main() -> None:
    import torch
    import torch_npu  # noqa: F401

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    torch.npu.set_device(args.device)

    output = torch.empty(LANES, dtype=torch.int64, device="npu")
    dst = from_dlpack(output, layout_tag=tla.arch.RowMajor, origin_shape=(LANES,))
    for order in ("increase", "decrease"):
        offsets = range(LANES) if order == "increase" else range(LANES - 1, -1, -1)
        for literal, bases in ((False, BASES), (True, ((1 << 32) - 7,))):
            artifact = tla.compile(
                arange_int64,
                dst,
                tla.Int64(0),
                order,
                literal,
                options="--npu-arch 3510",
            )
            for base in bases:
                artifact(dst, tla.Int64(base), block_num=1)
                torch.npu.synchronize()
                actual = output.cpu().tolist()
                expected = [
                    (base + lane + INT64_SIGN) % (2 * INT64_SIGN) - INT64_SIGN
                    for lane in offsets
                ]
                if actual != expected:
                    raise RuntimeError(
                        f"Int64 arange mismatch for {order}, literal={literal}, base={base}: "
                        f"expected {expected}, got {actual}"
                    )
                print(f"PASS {order} literal={literal} base={base}")


if __name__ == "__main__":
    main()
