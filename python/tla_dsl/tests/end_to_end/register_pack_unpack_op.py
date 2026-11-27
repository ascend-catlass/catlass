# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Check 64-bit vector Pack and predicate UnPack on Ascend 950PR."""

from __future__ import annotations

import argparse

from catlass import tla


@tla.kernel
def _rearrange(
    src: tla.Tensor,
    dst: tla.Tensor,
    src_dtype: tla.Constexpr[type],
    src_storage: tla.Constexpr[type],
    dst_storage: tla.Constexpr[type],
    part: tla.Constexpr[tla.RegPart],
) -> None:
    loaded = tla.flag("loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    computed = tla.flag("computed", tla.pipes.VECTOR, tla.pipes.MTE3)
    src_ptr = tla.allocate(2048 // src_storage.width, src_storage, tla.AddressSpace.ub, 256)
    dst_ptr = tla.allocate(2048 // dst_storage.width, dst_storage, tla.AddressSpace.ub, 256)
    ub_src = tla.make_tensor_like(src_ptr, src, tla.arch.RowMajor)
    ub_dst = tla.make_tensor_like(dst_ptr, dst, tla.arch.RowMajor)
    with tla.vector():
        tla.copy(ub_src, src)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        with tla.vec.func(mode="simd"):
            value = ub_src.load()
            if tla.const_expr(src_dtype.width == 64):
                result = tla.pack(value.bitcast(src_dtype), part=part).bitcast(dst_storage)
            else:
                pred = tla.cmp(value, 0, mode="gt")
                expanded = tla.unpack(pred, part=part)
                result = tla.where(expanded, tla.full(1, dst_storage), tla.full(0, dst_storage))
            ub_dst.store(result)
        tla.set_flag(computed)
        tla.wait_flag(computed)
        tla.copy(dst, ub_dst)
        tla.pipe_barrier(tla.pipes.ALL)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    # The high and low words differ. Vary the words across the register to
    # catch reordering and include signed boundaries and all-ones low words.
    b64 = torch.tensor(
        [0, -1, -(1 << 63), (1 << 63) - 1]
        + [-((i + 1) << 32) + (0x80000000 ^ (i * 131)) for i in range(28)],
        dtype=torch.int64,
    )
    cases = [
        ("pack-i64", b64, tla.Int64, tla.Int32),
        ("pack-u64", b64, tla.UInt64, tla.Int32),
        ("unpack-mask256", ((torch.arange(256) * 5) % 13 - 6).to(torch.int8), tla.Int8, tla.Int16),
        ("unpack-mask128", ((torch.arange(128) * 5) % 13 - 6).to(torch.int16), tla.Int16, tla.Int32),
    ]
    torch_dtype = {tla.Int16: torch.int16, tla.Int32: torch.int32}
    for name, host_src, src_dtype, dst_storage in cases:
        pack = src_dtype.width == 64
        dst_lanes = host_src.numel() * 2 if pack else host_src.numel() // 2
        half_lanes = host_src.numel() if pack else dst_lanes
        src_storage = tla.Int64 if pack else src_dtype
        dst_dtype = torch_dtype[dst_storage]
        src = host_src.npu()
        bound_src = tla.from_dlpack(src, layout_tag=tla.arch.RowMajor)
        for part in (tla.RegPart.LOW, tla.RegPart.HIGH):
            dst = torch.empty(dst_lanes, dtype=dst_dtype, device="npu")
            bound_dst = tla.from_dlpack(dst, layout_tag=tla.arch.RowMajor)
            artifact = tla.compile(
                _rearrange, bound_src, bound_dst, src_dtype, src_storage, dst_storage, part, options="--npu-arch 3510"
            )
            artifact(bound_src, bound_dst, block_num=1)
            torch.npu.synchronize()
            selected = slice(0, half_lanes) if part is tla.RegPart.LOW else slice(half_lanes, None)
            if pack:
                torch.testing.assert_close(dst.cpu()[selected], (host_src & 0xFFFFFFFF).to(dst_dtype))
            else:
                torch.testing.assert_close(dst.cpu(), (host_src[selected] > 0).to(dst_dtype))
            print(f"PASS {name}/{part.value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
