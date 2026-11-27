# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Sort 算子：LSD 基数排序，2-bit 一轮。

每个 pass 三步：桶起始偏移（histograms 的 FREQUENCY 减 ACCUMULATE）、
桶内压缩（cmp 出桶掩码 + squeeze 压缩索引）、落桶（scatter 到桶基址）。
一个字节轮跑 4 个 pass（shift 0/2/4/6）。

带符号与降序通过翻转映射成无符号可比值，排完翻回。

    python examples/python/tla_dsl/dsl_tutorials/advanced_impl/sort_topk/sort.py --device 0
"""

from __future__ import annotations

import argparse

import catlass.tla as tla

# 一个向量寄存器 256 字节
N = 256

# 编译期配置
KEY_BITS = 8
SIGNED = False
IS_FLOAT = False
DESCENDING = False

DTYPE_SPECS = {
    "u8": (8, False, False),
    "i8": (8, True, False),
}


def _pass(keys, src_tile, dst_tile, shift, consts, m_all128, lane_u16, bins, zero16):
    """一个 2-bit pass：桶偏移 -> 桶内压缩 -> 落桶。"""
    idx = src_tile.load()  # u8 256：当前索引序列
    cur = tla.gather(keys, idx)  # u8 256：按索引取 key
    c0_, c1_, c2_, c3_, k3u8, b0u16, b1u16, b2u16, b3u16 = consts
    shr = tla.shift_right(cur.bitcast(tla.Int8), tla.full(shift, dtype=tla.Int8))
    digit = tla.bitwise_and(shr.bitcast(tla.UInt8), k3u8)  # u8 256：2-bit digit

    # 桶起始偏移
    freq = tla.histograms(
        digit,
        tla.full(0, dtype=tla.Int16),
        tla.params.HistogramsParams(
            mode=tla.params.HistogramsBinType.BIN0,
            type=tla.params.HistogramsType.FREQUENCY,
        ),
        mask=tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Int8),
    )
    cum = tla.histograms(
        digit,
        tla.full(0, dtype=tla.Int16),
        tla.params.HistogramsParams(
            mode=tla.params.HistogramsBinType.BIN0,
            type=tla.params.HistogramsType.ACCUMULATE,
        ),
        mask=tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Int8),
    )
    excl = tla.sub(cum, freq)  # u16 128：桶起始偏移

    # 桶内压缩 + 落桶
    for kb, cu8 in ((b0u16, c0_), (b1u16, c1_), (b2u16, c2_), (b3u16, c3_)):
        hot = tla.cmp(bins, kb, "eq")  # 128 桶上的掩码
        cnt_b = tla.where(hot, freq, zero16).reduce(tla.ReductionOp.ADD, mask=m_all128)
        base_b = tla.where(hot, excl, zero16).reduce(tla.ReductionOp.ADD, mask=m_all128)
        m_b = tla.cmp(digit, cu8, "eq")  # u8 256 掩码
        compact = tla.squeeze(idx, m_b)  # GatherMask<STORE_REG>
        # u8 scatter 消费偶 lane，每桶分两批
        lo_pack = tla.unpack(compact, part=tla.RegPart.LOW).bitcast(tla.UInt8)
        hi_pack = tla.unpack(compact, part=tla.RegPart.HIGH).bitcast(tla.UInt8)
        base_u16 = tla.full(base_b, dtype=tla.Int16).bitcast(tla.UInt16)
        take1 = tla.cmp(
            lane_u16, tla.full(cnt_b, dtype=tla.Int16).bitcast(tla.UInt16), "lt"
        )
        tgt1 = tla.add(base_u16, lane_u16)
        tla.scatter(dst_tile, lo_pack, tgt1, mask=take1)
        c128u16 = tla.full(128, dtype=tla.Int16).bitcast(tla.UInt16)
        take2 = tla.cmp(
            tla.add(lane_u16, c128u16),
            tla.full(cnt_b, dtype=tla.Int16).bitcast(tla.UInt16),
            "lt",
        )
        tgt2 = tla.add(tla.add(base_u16, c128u16), lane_u16)
        tla.scatter(dst_tile, hi_pack, tgt2, mask=take2)
    tla.local_mem_bar(tla.params.MemType.VEC_ALL, tla.params.MemType.VEC_ALL)


def _twiddle_in(v, sign_bit, all_f, zero):
    """入方向翻转：映射成无符号可比值。"""
    u = v.bitcast(tla.UInt8)
    if IS_FLOAT:
        sg = tla.bitwise_and(u, sign_bit.bitcast(tla.UInt8))
        mv = tla.where(
            tla.cmp(sg, zero.bitcast(tla.UInt8), "ne"),
            all_f.bitcast(tla.UInt8),
            sign_bit.bitcast(tla.UInt8),
        )
        out = tla.bitwise_xor(u, mv)
    elif SIGNED:
        out = tla.bitwise_xor(u, sign_bit.bitcast(tla.UInt8))
    else:
        out = u
    if DESCENDING:
        out = tla.bitwise_not(out.bitcast(tla.Int8)).bitcast(tla.UInt8)
    return out


def _twiddle_out(v, sign_bit, all_f, zero):
    """出方向翻转：还原。"""
    u = v.bitcast(tla.UInt8)
    if DESCENDING:
        u = tla.bitwise_not(u.bitcast(tla.Int8)).bitcast(tla.UInt8)
    if IS_FLOAT:
        sg = tla.bitwise_and(u, sign_bit.bitcast(tla.UInt8))
        mv = tla.where(
            tla.cmp(sg, zero.bitcast(tla.UInt8), "eq"),
            all_f.bitcast(tla.UInt8),
            sign_bit.bitcast(tla.UInt8),
        )
        return tla.bitwise_xor(u, mv)
    if SIGNED:
        return tla.bitwise_xor(u, sign_bit.bitcast(tla.UInt8))
    return u


@tla.kernel
def sort_kernel(mem_keys: tla.Tensor, mem_val: tla.Tensor, mem_idx: tla.Tensor) -> None:
    """每字节轮 4 个 2-bit pass。"""
    loaded = tla.flag("l", tla.pipes.MTE2, tla.pipes.VECTOR)
    done = tla.flag("d", tla.pipes.VECTOR, tla.pipes.MTE3)
    kg = tla.tile_view(mem_keys, tla.make_shape(N), tla.make_coord(0))
    vg = tla.tile_view(mem_val, tla.make_shape(N), tla.make_coord(0))
    ig = tla.tile_view(mem_idx, tla.make_shape(N), tla.make_coord(0))

    def u8buf():
        return tla.make_tensor_like(
            tla.allocate(N, tla.UInt8, tla.AddressSpace.ub, 256), kg, tla.arch.RowMajor
        )

    ku = u8buf()  # 输入 key
    du = u8buf()  # 排序后的值
    iu = u8buf()  # 排序后的索引
    # 5 块独立缓冲：pass 不能轮转复用
    t0 = u8buf()
    t1 = u8buf()
    t2 = u8buf()
    t3 = u8buf()
    t4 = u8buf()
    with tla.vector():
        tla.copy(ku, kg)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        with tla.vec.func(mode="simd"):
            keys = tla.tile_view(ku, tla.make_shape(N), tla.make_coord(0)).load()
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

            def TV(t):
                return tla.tile_view(t, tla.make_shape(N), tla.make_coord(0))

            lane_u8 = tla.arange(base=0, order="increase", dtype=tla.Int8).bitcast(
                tla.UInt8
            )

            # 符号位 0x80
            sign_bit = tla.full(-128, dtype=tla.Int8)
            all_f = tla.bitwise_not(tla.full(0, dtype=tla.Int8)).bitcast(tla.UInt8)
            zero8 = tla.full(0, dtype=tla.Int8)
            keys = _twiddle_in(keys, sign_bit, all_f, zero8)

            TV(t0).store(lane_u8)
            tla.local_mem_bar(tla.params.MemType.VEC_ALL, tla.params.MemType.VEC_ALL)

            # 4 个 pass，shift 编译期常量
            _pass(keys, TV(t0), TV(t1), 0, consts, m_all128, lane_u16, bins, zero16)
            _pass(keys, TV(t1), TV(t2), 2, consts, m_all128, lane_u16, bins, zero16)
            _pass(keys, TV(t2), TV(t3), 4, consts, m_all128, lane_u16, bins, zero16)
            _pass(keys, TV(t3), TV(t4), 6, consts, m_all128, lane_u16, bins, zero16)

            fi = TV(t4).load()
            TV(du).store(_twiddle_out(tla.gather(keys, fi), sign_bit, all_f, zero8))
            TV(iu).store(fi)
        tla.set_flag(done)
        tla.wait_flag(done)
        tla.copy(vg, du)
        tla.copy(ig, iu)
        tla.pipe_barrier(tla.pipes.ALL)


def main() -> int:
    global DESCENDING, IS_FLOAT, KEY_BITS, SIGNED
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--dtype", default="u8", choices=sorted(DTYPE_SPECS))
    ap.add_argument("--desc", type=int, default=0, help="1 = descending")
    a = ap.parse_args()
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(a.device)
    KEY_BITS, SIGNED, IS_FLOAT = DTYPE_SPECS[a.dtype]
    DESCENDING = bool(a.desc)
    if SIGNED:
        keys = torch.randint(-128, 128, (N,), dtype=torch.int32)
        src = (keys & 0xFF).to(torch.uint8).to(device="npu")
    else:
        keys = torch.randint(0, 256, (N,), dtype=torch.int32)
        src = keys.to(torch.uint8).to(device="npu")
    val = torch.zeros((N,), dtype=torch.uint8, device="npu")
    idx = torch.zeros((N,), dtype=torch.uint8, device="npu")
    ts = tla.from_dlpack(src.contiguous(), layout_tag=tla.arch.RowMajor)
    tv = tla.from_dlpack(val.contiguous(), layout_tag=tla.arch.RowMajor)
    ti = tla.from_dlpack(idx.contiguous(), layout_tag=tla.arch.RowMajor)
    tla.compile(sort_kernel, ts, tv, ti, options="--npu-arch 3510")(
        ts, tv, ti, block_num=1
    )
    torch.npu.synchronize()
    if SIGNED:
        got = val.cpu().to(torch.int8).to(torch.int64)
        gsrc = (src.cpu() & 0xFF).to(torch.int8).to(torch.int64)
    else:
        got = val.cpu().to(torch.int64)
        gsrc = src.cpu().to(torch.int64)
    gidx = idx.cpu().to(torch.int64)
    exp = torch.sort(keys, descending=DESCENDING).values.to(torch.int64)
    # 失配抛错，非 0 退出
    torch.testing.assert_close(got, exp, rtol=0, atol=0)
    torch.testing.assert_close(gsrc[gidx].to(torch.int64), exp, rtol=0, atol=0)
    print(f"[{a.dtype}] sort desc={DESCENDING}: values=OK indices=OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
