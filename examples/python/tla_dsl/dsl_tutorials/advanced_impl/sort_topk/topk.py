# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""TopK 算子：排序后取前 k 个。

复用 sort.py 的 _pass 排序，再用 lane < K 的掩码 squeeze 压出前 k 个。
取最大需降序，故排序前整位取反、排完再取反。

K 是编译期常量，按 u8 位模式构造（K>127 取 K-256），K==256 用全 1 掩码。

    python examples/python/tla_dsl/dsl_tutorials/advanced_impl/sort_topk/topk.py --device 0 --k 8
"""

from __future__ import annotations

import argparse
import os
import sys

import catlass.tla as tla

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sort import _pass as _sort_pass

N = 256  # 一个向量寄存器 256 字节
K = 8  # 编译期 K
IS_LARGEST = True
# K 的 u8 位模式
K_BITS = K if K <= 127 else K - 256
KEEP_ALL = K == N


@tla.kernel
def topk_kernel(mem_keys: tla.Tensor, mem_val: tla.Tensor, mem_idx: tla.Tensor) -> None:
    loaded = tla.flag("l", tla.arch.MTE2, tla.arch.VECTOR)
    done = tla.flag("d", tla.arch.VECTOR, tla.arch.MTE3)
    kg = tla.tile_view(mem_keys, tla.make_shape(N), tla.make_coord(0))
    vg = tla.tile_view(mem_val, tla.make_shape(N), tla.make_coord(0))
    ig = tla.tile_view(mem_idx, tla.make_shape(N), tla.make_coord(0))

    def ubuf():
        return tla.make_tensor_like(
            tla.allocate(N, tla.UInt8, tla.AddressSpace.ub, 256), kg, tla.arch.RowMajor
        )

    ku = ubuf()
    t0 = ubuf()
    t1 = ubuf()
    t2 = ubuf()
    t3 = ubuf()
    t4 = ubuf()
    dv = ubuf()
    di = ubuf()
    with tla.vector():
        tla.copy(ku, kg)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        with tla.vec.func(mode="simd"):

            def TV(x):
                return tla.tile_view(x, tla.make_shape(N), tla.make_coord(0))

            keys = TV(ku).load()

            # 与 sort.py 的 _pass 约定一致
            k0u8 = tla.full(0, dtype=tla.Int8).bitcast(tla.UInt8)
            k1u8 = tla.full(1, dtype=tla.Int8).bitcast(tla.UInt8)
            k2u8 = tla.full(2, dtype=tla.Int8).bitcast(tla.UInt8)
            k3u8 = tla.full(3, dtype=tla.Int8).bitcast(tla.UInt8)
            b0u16 = tla.full(0, dtype=tla.Int16).bitcast(tla.UInt16)
            b1u16 = tla.full(1, dtype=tla.Int16).bitcast(tla.UInt16)
            b2u16 = tla.full(2, dtype=tla.Int16).bitcast(tla.UInt16)
            b3u16 = tla.full(3, dtype=tla.Int16).bitcast(tla.UInt16)
            consts = (k0u8, k1u8, k2u8, k3u8, k3u8, b0u16, b1u16, b2u16, b3u16)
            m_all128 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.UInt16)
            zero16 = tla.full(0, dtype=tla.Int16)
            lane_u16 = tla.arange(base=0, order="increase", dtype=tla.Int16).bitcast(
                tla.UInt16
            )
            bins = lane_u16
            lane_u8 = tla.arange(base=0, order="increase", dtype=tla.Int8).bitcast(
                tla.UInt8
            )

            # 排序；取最大需降序，故前后取反
            work = keys
            if IS_LARGEST:
                work = tla.bitwise_not(keys.bitcast(tla.Int8)).bitcast(tla.UInt8)
            TV(t0).store(lane_u8)
            tla.local_mem_bar(tla.params.MemType.VEC_ALL, tla.params.MemType.VEC_ALL)
            _sort_pass(
                work, TV(t0), TV(t1), 0, consts, m_all128, lane_u16, bins, zero16
            )
            _sort_pass(
                work, TV(t1), TV(t2), 2, consts, m_all128, lane_u16, bins, zero16
            )
            _sort_pass(
                work, TV(t2), TV(t3), 4, consts, m_all128, lane_u16, bins, zero16
            )
            _sort_pass(
                work, TV(t3), TV(t4), 6, consts, m_all128, lane_u16, bins, zero16
            )
            perm = TV(t4).load()
            sorted_v = tla.gather(work, perm)
            if IS_LARGEST:
                sorted_v = tla.bitwise_not(sorted_v.bitcast(tla.Int8)).bitcast(
                    tla.UInt8
                )

            # 取前 k 个；K==256 时 lane<256 恒真，用全 1 掩码
            if KEEP_ALL:
                keep = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.UInt8)
            else:
                keep = tla.cmp(
                    lane_u8, tla.full(K_BITS, dtype=tla.Int8).bitcast(tla.UInt8), "lt"
                )
            TV(dv).store(tla.squeeze(sorted_v, keep))
            TV(di).store(tla.squeeze(perm, keep))
        tla.set_flag(done)
        tla.wait_flag(done)
        tla.copy(vg, dv)
        tla.copy(ig, di)
        tla.pipe_barrier(tla.pipes.ALL)


def main() -> int:
    global IS_LARGEST, K, KEEP_ALL, K_BITS
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--smallest", type=int, default=0, help="1 = select the smallest")
    a = ap.parse_args()
    K = a.k
    if not (1 <= K <= N):
        raise SystemExit(f"--k must be in [1, {N}]; got {K}")
    K_BITS = K if K <= 127 else K - 256
    KEEP_ALL = K == N
    IS_LARGEST = not bool(a.smallest)
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(a.device)
    keys = torch.randint(0, 256, (N,), dtype=torch.int32)
    src = keys.to(torch.uint8).to(device="npu")
    val = torch.zeros((N,), dtype=torch.uint8, device="npu")
    idx = torch.zeros((N,), dtype=torch.uint8, device="npu")
    ts = tla.from_dlpack(src.contiguous(), layout_tag=tla.arch.RowMajor)
    tv = tla.from_dlpack(val.contiguous(), layout_tag=tla.arch.RowMajor)
    ti = tla.from_dlpack(idx.contiguous(), layout_tag=tla.arch.RowMajor)
    tla.compile(topk_kernel, ts, tv, ti, options="--npu-arch 3510")(
        ts, tv, ti, block_num=1
    )
    torch.npu.synchronize()
    got = val.cpu().to(torch.int64)[:K]
    gidx = idx.cpu().to(torch.int64)[:K]
    exp = torch.sort(keys, descending=IS_LARGEST).values.to(torch.int64)[:K]
    # 失配即抛错，使脚本以非 0 退出（battery 断言 rc == 0）
    torch.testing.assert_close(got, exp, rtol=0, atol=0)
    torch.testing.assert_close(keys[gidx].to(torch.int64), exp, rtol=0, atol=0)
    print(f"topk k={K} largest={IS_LARGEST}: values=OK indices=OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
