# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Constant pad of the last dim: ``F.pad(x, (left, right), value=...)``.

The DSL has no GM->GM copy, so the row is staged in UB once.  Instead of
shifting the payload inside UB (which would need an unaligned vector store),
the output row is written as three disjoint pieces:

* the ``left`` pad cells  - a batched strided 2-D DMA from a UB pad buffer;
* the payload            - a batched strided 2-D DMA (row stride ``W``);
* the ``right`` pad cells - a batched strided 2-D DMA from the same buffer.

Every GM write may start at an arbitrary element offset and have an arbitrary
byte length (the MTE handles both); the UB staging buffers stay 256 B aligned,
so any ``N``, ``left`` and ``right`` is supported, including values that are
not multiples of the 8-element (32 B) alignment.  The result is a dense
``[M, W]`` buffer (no strided view), and the pad cells cost two extra DMAs per
*row batch* rather than per row.

高性能设计：

1. 一个 block 一次处理至多 ``BR`` 行，payload 和两侧 pad 都按行组批量 DMA，
   把逐行小搬运的指令开销摊薄到整组。
2. GM 写允许任意元素偏移，因此把一行拆成“左 pad + payload + 右 pad”三段互不
   重叠写，避免在 UB 内做非 32 B 对齐的整行平移。
3. 左右 pad 缓冲在每个 block 只由 VECTOR 填充一次，之后 MTE3 反复读取；同步只
   发生在这个只读发布边界，不进入每个 payload DMA。
4. 核间按 ``gid + k * aiv_num`` 领取独立行组，没有跨核归约或全局屏障。
"""

import catlass.tla as tla

VL_ELE = 64  # 每个 fp32 vector register 的 lane 数；pad 缓冲行宽按此对齐
STAGE_ELEMS = 61440  # 240 KiB total staging budget for two payload slots
SLOT_ELEMS = STAGE_ELEMS // 2


@tla.kernel
def pad_kernel(
    gm_src: tla.Tensor,  # [M, N] fp32
    gm_dst: tla.Tensor,  # [M, W] fp32, W = N + PL + PR
    M: tla.Constexpr[int],
    N: tla.Constexpr[int],
    W: tla.Constexpr[int],
    PL: tla.Constexpr[int],  # left padding (elements)
    PR: tla.Constexpr[int],  # right padding (elements)
    LS: tla.Constexpr[int],  # left transfer width (max(PL, 1))
    RS: tla.Constexpr[int],  # right transfer width (max(PR, 1))
    PLA: tla.Constexpr[int],  # left pad-buffer row width (multiple of VL_ELE)
    PRA: tla.Constexpr[int],  # right pad-buffer row width (multiple of VL_ELE)
    PADV: tla.Constexpr[float],
    BR: tla.Constexpr[int],  # rows per batch
) -> None:
    gid = tla.arch.block_idx()
    aiv_num = tla.arch.block_num()

    # ---- payload staging：O(BR * N) 的连续 GM→UB 搬运 ----
    # The strided GM->UB copy may touch a few elements past the last staged
    # row, so the allocation carries a guard tail that no tensor view covers.
    # 实测跨步批量拷贝会多写少量元素；256 元素 guard 防止踩坏紧邻的 pad 缓冲。
    stage0_ptr = tla.allocate(BR * N + 256, tla.Float32, tla.AddressSpace.ub, 256)
    stage1_ptr = tla.allocate(BR * N + 256, tla.Float32, tla.AddressSpace.ub, 256)
    ub_lay = tla.make_layout(
        tla.make_shape(BR, N),
        tla.make_stride(N, 1),
        origin_shape=tla.make_shape(BR, N),
    )
    dst_lay = tla.make_layout(
        tla.make_shape(BR, N),
        tla.make_stride(W, 1),
        origin_shape=tla.make_shape(BR, N),
    )
    ub0 = tla.make_tensor(stage0_ptr, ub_lay)
    ub1 = tla.make_tensor(stage1_ptr, ub_lay)

    row_lay = tla.make_layout(
        tla.make_shape(1, N),
        tla.make_stride(N, 1),
        origin_shape=tla.make_shape(1, N),
    )
    row_dst_lay = tla.make_layout(
        tla.make_shape(1, N),
        tla.make_stride(W, 1),
        origin_shape=tla.make_shape(1, N),
    )
    ub_row = tla.make_tensor(stage0_ptr, row_lay)

    # ---- pad-cell buffers：BR 行 × 64-lane 对齐行宽，只填充一次 ----
    l_ptr = tla.allocate(BR * PLA, tla.Float32, tla.AddressSpace.ub, 256)
    r_ptr = tla.allocate(BR * PRA, tla.Float32, tla.AddressSpace.ub, 256)
    lfill_lay = tla.make_layout(
        tla.make_shape(1, BR * PLA),
        tla.make_stride(BR * PLA, 1),
        origin_shape=tla.make_shape(1, BR * PLA),
    )
    rfill_lay = tla.make_layout(
        tla.make_shape(1, BR * PRA),
        tla.make_stride(BR * PRA, 1),
        origin_shape=tla.make_shape(1, BR * PRA),
    )
    ub_lfill = tla.make_tensor(l_ptr, lfill_lay)
    ub_rfill = tla.make_tensor(r_ptr, rfill_lay)
    lb_src_lay = tla.make_layout(
        tla.make_shape(BR, LS),
        tla.make_stride(PLA, 1),
        origin_shape=tla.make_shape(BR, LS),
    )
    lb_dst_lay = tla.make_layout(
        tla.make_shape(BR, LS),
        tla.make_stride(W, 1),
        origin_shape=tla.make_shape(BR, LS),
    )
    rb_src_lay = tla.make_layout(
        tla.make_shape(BR, RS),
        tla.make_stride(PRA, 1),
        origin_shape=tla.make_shape(BR, RS),
    )
    rb_dst_lay = tla.make_layout(
        tla.make_shape(BR, RS),
        tla.make_stride(W, 1),
        origin_shape=tla.make_shape(BR, RS),
    )
    ub_lb = tla.make_tensor(l_ptr, lb_src_lay)
    ub_rb = tla.make_tensor(r_ptr, rb_src_lay)
    # one-row destination shapes for the tail rows
    l_row_dst_lay = tla.make_layout(
        tla.make_shape(1, LS),
        tla.make_stride(W, 1),
        origin_shape=tla.make_shape(1, LS),
    )
    r_row_dst_lay = tla.make_layout(
        tla.make_shape(1, RS),
        tla.make_stride(W, 1),
        origin_shape=tla.make_shape(1, RS),
    )

    # 左右 pad 缓冲是两个独立资源，各自使用独立 mutex/id。
    # 生产者是 VECTOR，消费者是 MTE3；不能用同一 mutex 合并两条依赖链。
    mtx_lpad = tla.mutex(resource="pad_left", id=0)
    mtx_rpad = tla.mutex(resource="pad_right", id=1)
    mtx_stage0 = tla.mutex(resource="payload_stage0", id=2)
    mtx_stage1 = tla.mutex(resource="payload_stage1", id=3)

    n_groups = M // BR
    tail = M % BR

    with tla.vector():
        # 性能设计点：pad 缓冲只填一次；行宽为 VL_ELE 的整数倍，vector store
        # 始终满足 32 B 对齐。无对应 padding 的一侧完全不分配额外 pipe 工作。
        if tla.const_expr(PL != 0):
            mtx_lpad.lock(pipe=tla.arch.VECTOR)
            with tla.vec.func(mode="simd"):
                pv = tla.full(PADV, dtype=tla.Float32)
                for fi in tla.range(BR * PLA // VL_ELE):
                    tla.tile_view(
                        ub_lfill, tla.make_shape(1, VL_ELE), tla.make_coord(0, fi)
                    ).store(pv)
            mtx_lpad.unlock(pipe=tla.arch.VECTOR)

        if tla.const_expr(PR != 0):
            mtx_rpad.lock(pipe=tla.arch.VECTOR)
            with tla.vec.func(mode="simd"):
                pv = tla.full(PADV, dtype=tla.Float32)
                for fj in tla.range(BR * PRA // VL_ELE):
                    tla.tile_view(
                        ub_rfill, tla.make_shape(1, VL_ELE), tla.make_coord(0, fj)
                    ).store(pv)
            mtx_rpad.unlock(pipe=tla.arch.VECTOR)

        # 消费者锁覆盖整个只读生命周期，而不是每个行组重复加锁/解锁。
        # MTE3 会等到对应 VECTOR unlock 后才读取已填好的 pad 缓冲。
        if tla.const_expr(PL != 0):
            mtx_lpad.lock(pipe=tla.arch.MTE3)
        if tla.const_expr(PR != 0):
            mtx_rpad.lock(pipe=tla.arch.MTE3)

        # ---- 双槽流水：每个 block 每次登记两个行组 ----
        # 先为两个槽提交 MTE2 load，再依次提交 MTE3 三段写。不同槽使用独立
        # mutex，因此后一槽的 load 可与前一槽的 store 重叠。
        groups_per_block = n_groups // aiv_num
        extra_groups = n_groups % aiv_num
        group_count = groups_per_block + (1 if gid < extra_groups else 0)
        group_begin = gid * groups_per_block + (
            gid if gid < extra_groups else extra_groups
        )
        for local_pair in tla.range((group_count + 1) // 2):
            for slot in tla.range_constexpr(2):
                b = group_begin + local_pair * 2 + slot
                if local_pair * 2 + slot < group_count:
                    stage = ub0 if slot == 0 else ub1
                    mtx_stage = mtx_stage0 if slot == 0 else mtx_stage1
                    mtx_stage.lock(pipe=tla.arch.MTE2)
                    tla.copy(
                        stage,
                        tla.make_tensor(
                            gm_src.ptr + tla.Int64(b) * tla.Int64(BR * N),
                            ub_lay,
                        ),
                    )
                    mtx_stage.unlock(pipe=tla.arch.MTE2)

            for slot in tla.range_constexpr(2):
                b = group_begin + local_pair * 2 + slot
                if local_pair * 2 + slot < group_count:
                    stage = ub0 if slot == 0 else ub1
                    mtx_stage = mtx_stage0 if slot == 0 else mtx_stage1
                    mtx_stage.lock(pipe=tla.arch.MTE3)
                    tla.copy(
                        tla.make_tensor(
                            gm_dst.ptr
                            + tla.Int64(b) * tla.Int64(BR * W)
                            + tla.Int64(PL),
                            dst_lay,
                        ),
                        stage,
                    )
                    if tla.const_expr(PL != 0):
                        tla.copy(
                            tla.make_tensor(
                                gm_dst.ptr + tla.Int64(b) * tla.Int64(BR * W),
                                lb_dst_lay,
                            ),
                            ub_lb,
                        )
                    if tla.const_expr(PR != 0):
                        tla.copy(
                            tla.make_tensor(
                                gm_dst.ptr
                                + tla.Int64(b) * tla.Int64(BR * W)
                                + tla.Int64(PL + N),
                                rb_dst_lay,
                            ),
                            ub_rb,
                        )
                    mtx_stage.unlock(pipe=tla.arch.MTE3)

        # ---- 尾行：只处理 M % BR 行；各核交错领取，避免少数核串行收尾 ----
        if tla.const_expr(tail != 0):
            for t in tla.range(gid, tail, aiv_num):
                mtx_stage0.lock(pipe=tla.arch.MTE2)
                tla.copy(
                    ub_row,
                    tla.make_tensor(
                        gm_src.ptr
                        + (t + tla.Int32(n_groups * BR)).to(tla.Int64) * tla.Int64(N),
                        row_lay,
                    ),
                )
                mtx_stage0.unlock(pipe=tla.arch.MTE2)
                mtx_stage0.lock(pipe=tla.arch.MTE3)
                tla.copy(
                    tla.make_tensor(
                        gm_dst.ptr
                        + (t + tla.Int32(n_groups * BR)).to(tla.Int64) * tla.Int64(W)
                        + tla.Int64(PL),
                        row_dst_lay,
                    ),
                    ub_row,
                )
                if tla.const_expr(PL != 0):
                    tla.copy(
                        tla.make_tensor(
                            gm_dst.ptr
                            + (t + tla.Int32(n_groups * BR)).to(tla.Int64)
                            * tla.Int64(W),
                            l_row_dst_lay,
                        ),
                        tla.tile_view(
                            ub_lfill, tla.make_shape(1, LS), tla.make_coord(0, 0)
                        ),
                    )
                if tla.const_expr(PR != 0):
                    tla.copy(
                        tla.make_tensor(
                            gm_dst.ptr
                            + (t + tla.Int32(n_groups * BR)).to(tla.Int64)
                            * tla.Int64(W)
                            + tla.Int64(PL + N),
                            r_row_dst_lay,
                        ),
                        tla.tile_view(
                            ub_rfill, tla.make_shape(1, RS), tla.make_coord(0, 0)
                        ),
                    )
                mtx_stage0.unlock(pipe=tla.arch.MTE3)

        if tla.const_expr(PR != 0):
            mtx_rpad.unlock(pipe=tla.arch.MTE3)
        if tla.const_expr(PL != 0):
            mtx_lpad.unlock(pipe=tla.arch.MTE3)


# ======================================================================================
# HOST side
# ======================================================================================

_COMPILED = {}
_RUN_CACHE = {}
_ROWMAJOR = tla.arch.RowMajor
_CORES = {}
_DEVICES = {}


def _device_index(device):
    name = str(device)
    index = _DEVICES.get(name)
    if index is None:
        index = int(name.split(":")[-1]) if ":" in name else 0
        _DEVICES[name] = index
    return index


def _cores(device_index):
    cores = _CORES.get(device_index)
    if cores is None:
        import torch

        cores = int(torch.npu.get_device_properties(device_index).vector_core_num)
        _CORES[device_index] = cores
    return cores


def _launch_plan(plan, ta, tb, block_num):
    da = plan["a"]
    db = plan["b"]
    ptr = int(ta.data_ptr())
    if da.data_ptr != ptr:
        da.data_ptr = ptr
    ptr = int(tb.data_ptr())
    if db.data_ptr != ptr:
        db.data_ptr = ptr
    plan["artifact"](da, db, block_num=block_num)


def _launch(kernel, ta, tb, extra_args, key, block_num):
    """Compile and cache one launch plan, then execute it."""
    from catlass.tla.runtime import from_dlpack

    plan = _COMPILED.get(key)
    if plan is None:
        da = from_dlpack(ta, layout_tag=_ROWMAJOR)
        db = from_dlpack(tb, layout_tag=_ROWMAJOR)
        artifact = tla.compile(kernel, da, db, *extra_args, options="--npu-arch 3510")
        plan = {
            "artifact": artifact,
            "a": da,
            "b": db,
        }
        _COMPILED[key] = plan
    _launch_plan(plan, ta, tb, block_num)
    return plan


def _round_up(x, m):
    return (x + m - 1) // m * m


def pad(x, left, right, value=0.0):
    """Constant-pad the last dim of a 2-D fp32 tensor (dense ``[M, N+left+right]``)."""
    import torch

    if x.dtype != torch.float32 or x.dim() != 2:
        raise ValueError("x must be 2-D float32")
    if left < 0 or right < 0:
        raise ValueError("left/right must be >= 0")
    x = x if x.is_contiguous() else x.contiguous()
    m, n = x.shape
    device_index = _device_index(x.device)
    cache_key = (device_index, m, n, left, right, float(value))
    cached = _RUN_CACHE.get(cache_key)
    if cached is not None:
        w, plan, block_num = cached
        out = x.new_empty((m, w))
        _launch_plan(plan, x, out, block_num)
        return out
    w = n + left + right
    if m == 0 or n == 0:
        return torch.full((m, w), float(value), dtype=torch.float32, device=x.device)
    ls = max(left, 1)
    rs = max(right, 1)
    pla = max(_round_up(left, VL_ELE), VL_ELE)
    pra = max(_round_up(right, VL_ELE), VL_ELE)
    if n + pla + pra > SLOT_ELEMS:
        raise ValueError(
            f"row width {n} plus padded row buffers requires "
            f"{n + pla + pra} elements, exceeding the UB slot capacity "
            f"of {SLOT_ELEMS}"
        )
    core_num = _cores(device_index)
    capacity_br = max(1, min(32, SLOT_ELEMS // (n + pla + pra)))
    # 低 M 时不要用一个超大行组把任务压到少数核；按满载核数反推 BR，
    # 使 M < 56 时自然退化为每块 1 行，M 较大时仍以 UB 容量上限为准。
    balance_br = max(1, (m + core_num - 1) // core_num)
    br = min(capacity_br, balance_br)
    row_groups = (m + br - 1) // br
    block_num = min(core_num, row_groups)
    out = x.new_empty((m, w))
    args = [m, n, w, left, right, ls, rs, pla, pra, float(value), br]
    plan = _launch(
        pad_kernel,
        x,
        out,
        args,
        ("pad", device_index) + tuple(args),
        block_num,
    )
    _RUN_CACHE[cache_key] = (w, plan, block_num)
    return out


# ======================================================================================
# self check
# ======================================================================================

if __name__ == "__main__":
    import argparse
    import itertools

    import torch
    import torch_npu  # noqa: F401

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--shard", default="0/1", help="shard index/count")
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    shard_index, shard_count = (int(value) for value in args.shard.split("/"))
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        parser.error("invalid --shard")

    torch.npu.set_device(args.device)
    configs = (
        ((37, 1001), (0, 0), -2.5),
        ((37, 1001), (3, 5), -2.5),
        ((37, 63), (1, 1), 0.0),
        ((9, 999), (7, 9), 1.25),
        ((13, 4096), (64, 64), -2.5),
        ((5, 63), (13, 17), 4.0),
        ((1, 1), (1, 1), -2.5),
        ((11, 1001), (65, 3), -2.5),
        ((113, 7), (65, 66), 3.125),
        ((65, 1000), (5, 0), -11.0),
    )
    selected = [
        config
        for index, config in enumerate(configs)
        if index % shard_count == shard_index
    ]
    passed = 0
    total = 0
    for repeat, ((m, n), (left, right), value) in itertools.product(
        range(args.repeats), selected
    ):
        torch.manual_seed(repeat * 131 + m + n + left + right)
        x = torch.randn(m, n, dtype=torch.float32, device=f"npu:{args.device}")
        out = pad(x, left, right, value=value)
        ref = torch.nn.functional.pad(x.cpu(), (left, right), value=value)
        if not torch.equal(out.cpu(), ref):
            raise RuntimeError(
                f"mismatch: m={m} n={n} left={left} right={right} repeat={repeat}"
            )
        passed += 1
        total += 1
    print(
        f"pad self-check: {passed}/{total} passed (shard {shard_index}/{shard_count})"
    )
