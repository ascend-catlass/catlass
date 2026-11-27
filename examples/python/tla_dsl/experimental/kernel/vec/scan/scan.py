# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Row-wise inclusive prefix sum along N, with optional segment reset (fp32).

    y[r, j] = sum_{k = seg_start(j)}^{j} x[r, k]

``seg_start(j)`` is ``0`` for a plain cumulative sum and the last position
``k <= j`` with ``head[r, k] != 0`` for the segmented variant (``head`` is
usually a cu_seqlens / chunk-boundary mask).

性能设计要点：

1. **tile = 一行 + per-buffer mutex 的三段流水**：每个块认领一段连续行，每行一次
   MTE2 搬入、一次 MTE3 搬出；``BUFS`` 个行缓冲用 per-buffer mutex 串起来，搬入 /
   计算 / 搬出由 mutex 而不是程序序决定何时执行，因此 MTE2 搬第 r+1 行、VECTOR 扫
   第 r 行、MTE3 写第 r-1 行三段重叠，扫描这种纯向量算子不再被 DMA 串行化。
2. **64 lane Hillis-Steele 树 + 寄存器内移位**：一个 64-lane chunk 原地扫描，六步
   ``shift up by d``（d = 1,2,4,8,16,32）全部用 ``(lane - d) mod 64`` 的寄存器
   gather + 恒等填充完成，不走 UB 往返，也没有 shared-memory 冲突。
3. **跨 chunk 只做一次归约**：chunk 按序推进，把上一块 lane 63 的值折叠进下一块
   lane 0，跨块传播的代价是每个 chunk 一次 ``reduce``，而不是整行重扫。
4. **尾块按完整 64-lane 向量处理**：UB 行按 ``N_PAD = ceil(N/64)*64`` 补齐，GM 侧
   保持稠密 ``N``；末块先清零 padding lane 再用全 lane mask 扫描（收窄 mask 会让
   向量长度变化、破坏 64-lane gather），最后用 ``update_mask(TAIL)`` 只落盘前
   ``TAIL`` 个 lane。``N < 64`` 时 ``n_chunks == 0``，整个算子就退化成这一条尾块
   路径，无需额外分支。
5. **GM 起始地址不要求 32 B 对齐**：一行的 GM 基址是 ``r * N * 4`` 字节，尾块路径
   只对 UB 侧做 256 B 对齐，GM 侧的 ``N`` 可以是任意正整数（例如 65、999、1000），
   不对齐的行首由 DMA 自己处理。
6. **段式扫描零额外遍历**：segmented 变体把 ``(value, head-flag)`` 作为算子
   ``op((a,ha),(b,hb)) = hb ? (b,1) : (a+b,ha)`` 的二元组，与普通扫描共用同一棵
   树，每个 chunk 只多一组 ``where/cmp``，不增加内存往返；``HAS_HEAD == 0`` 时这
   部分在编译期整体消除。
"""

import functools
import inspect
import threading

import torch

import catlass.tla as tla
from catlass.tla.runtime import _Tensor, from_dlpack

VL_ELE = 64  # fp32 vector length (256 B / 4 B)
UB_BYTES = 262144  # per-AIV UB capacity, used to size the row pipeline
_CAST_NONE = tla.params.CastParams()


# --- kernel-side helpers -----------------------------------------------------


def _shift_up(v, d, lane_i, lane_f):
    """``v2[j] = v[j - d]`` for ``j >= d`` and ``0`` (scan identity) elsewhere."""
    wrapped = tla.add(lane_i, VL_ELE - d)
    idx = tla.bitwise_and(wrapped, tla.full(float(VL_ELE - 1), dtype=tla.Int32))
    return tla.where(
        tla.cmp(lane_f, d, "ge"),
        tla.gather(v, idx),
        tla.full(0.0, dtype=tla.Float32),
    )


def _head_flag(h):
    """Normalize a loaded head chunk to ``0.0`` / ``1.0``.

    The contract is "every non-zero entry starts a segment", so any non-zero
    value (not just ``1.0``) must raise the flag; the scan below compares flags
    against ``1.0`` and also propagates them with ``where`` / ``reduce``.
    """
    return tla.where(
        tla.cmp(h, 0.0, "ne"),
        tla.full(1.0, dtype=tla.Float32),
        tla.full(0.0, dtype=tla.Float32),
    )


def _scan_sum(v, lane_i, lane_f, mask):
    """Inclusive Hillis-Steele prefix sum of one 64-lane chunk."""
    w = tla.add(v, _shift_up(v, 1, lane_i, lane_f), mask=mask)
    w = tla.add(w, _shift_up(w, 2, lane_i, lane_f), mask=mask)
    w = tla.add(w, _shift_up(w, 4, lane_i, lane_f), mask=mask)
    w = tla.add(w, _shift_up(w, 8, lane_i, lane_f), mask=mask)
    w = tla.add(w, _shift_up(w, 16, lane_i, lane_f), mask=mask)
    w = tla.add(w, _shift_up(w, 32, lane_i, lane_f), mask=mask)
    return w


def _scan_seg(v, h, lane_i, lane_f, mask):
    """Segmented inclusive prefix sum over ``(value, head-flag)`` pairs."""
    av = tla.where(
        tla.cmp(h, 1.0, "eq"),
        v,
        tla.add(v, _shift_up(v, 1, lane_i, lane_f), mask=mask),
    )
    ah = tla.where(tla.cmp(h, 1.0, "eq"), h, _shift_up(h, 1, lane_i, lane_f))
    av = tla.where(
        tla.cmp(ah, 1.0, "eq"),
        av,
        tla.add(av, _shift_up(av, 2, lane_i, lane_f), mask=mask),
    )
    ah = tla.where(tla.cmp(ah, 1.0, "eq"), ah, _shift_up(ah, 2, lane_i, lane_f))
    av = tla.where(
        tla.cmp(ah, 1.0, "eq"),
        av,
        tla.add(av, _shift_up(av, 4, lane_i, lane_f), mask=mask),
    )
    ah = tla.where(tla.cmp(ah, 1.0, "eq"), ah, _shift_up(ah, 4, lane_i, lane_f))
    av = tla.where(
        tla.cmp(ah, 1.0, "eq"),
        av,
        tla.add(av, _shift_up(av, 8, lane_i, lane_f), mask=mask),
    )
    ah = tla.where(tla.cmp(ah, 1.0, "eq"), ah, _shift_up(ah, 8, lane_i, lane_f))
    av = tla.where(
        tla.cmp(ah, 1.0, "eq"),
        av,
        tla.add(av, _shift_up(av, 16, lane_i, lane_f), mask=mask),
    )
    ah = tla.where(tla.cmp(ah, 1.0, "eq"), ah, _shift_up(ah, 16, lane_i, lane_f))
    av = tla.where(
        tla.cmp(ah, 1.0, "eq"),
        av,
        tla.add(av, _shift_up(av, 32, lane_i, lane_f), mask=mask),
    )
    ah = tla.where(tla.cmp(ah, 1.0, "eq"), ah, _shift_up(ah, 32, lane_i, lane_f))
    return av, ah


@tla.kernel
def scan_kernel(
    gm_x: tla.Tensor,  # [M, N] fp32
    gm_head: tla.Tensor,  # [M, N] fp32, non-zero starts a new segment (unused when HAS_HEAD == 0)
    gm_y: tla.Tensor,  # [M, N] fp32
    N: tla.Constexpr[int],
    N_PAD: tla.Constexpr[int],
    TAIL: tla.Constexpr[int],
    HAS_HEAD: tla.Constexpr[int],
    BUFS: tla.Constexpr[int],
) -> None:
    m = gm_x.origin_shape[0]

    # One mutex per row buffer guards a whole (load -> scan -> store) round trip:
    # MTE2 unlocks it when the GM->UB copy retires, VECTOR then owns the buffer,
    # and MTE3 takes it over once the store retires. Because the three stages of
    # consecutive buffers are issued back to back, the pipes overlap.
    x_ptr = [None] * BUFS
    h_ptr = [None] * BUFS
    row_mutex = [None] * BUFS
    for bi in tla.range_constexpr(BUFS):
        x_ptr[bi] = tla.allocate(N_PAD, tla.Float32, tla.AddressSpace.ub, 256)
        if tla.const_expr(HAS_HEAD == 1):
            h_ptr[bi] = tla.allocate(N_PAD, tla.Float32, tla.AddressSpace.ub, 256)
        row_mutex[bi] = tla.mutex(resource="sc_row", id=bi)

    # UB rows are padded to a whole 64-lane vector; the GM side stays dense (N),
    # so a row can start at any byte offset (N need not be 32 B aligned).
    row_ub = tla.make_layout(
        tla.make_shape(1, N),
        tla.make_stride(N_PAD, 1),
        origin_shape=tla.make_shape(1, N),
    )
    row_gm = tla.make_layout(
        tla.make_shape(1, N),
        tla.make_stride(N, 1),
        origin_shape=tla.make_shape(1, N),
    )
    ub_x = [None] * BUFS
    ub_h = [None] * BUFS
    for bj in tla.range_constexpr(BUFS):
        ub_x[bj] = tla.make_tensor(x_ptr[bj], row_ub)
        if tla.const_expr(HAS_HEAD == 1):
            ub_h[bj] = tla.make_tensor(h_ptr[bj], row_ub)

    n_chunks = N // VL_ELE
    gid = tla.arch.block_idx()
    aiv_num = tla.arch.block_num()
    # Each block owns a contiguous span of rows, so its GM accesses walk forward
    # and every row is a single contiguous DMA in and out.
    rows_per_block = (m + aiv_num - 1) // aiv_num
    row_begin = gid * rows_per_block

    with tla.vector():
        for grp in tla.range((rows_per_block + BUFS - 1) // BUFS):
            for bi in tla.range_constexpr(BUFS):
                r = row_begin + grp * BUFS + bi
                if r < m:
                    row_off = r.to(tla.Int64) * tla.Int64(N)
                    gm_row_x = tla.make_tensor(gm_x.ptr + row_off, row_gm)
                    gm_row_y = tla.make_tensor(gm_y.ptr + row_off, row_gm)

                    row_mutex[bi].lock(pipe=tla.pipes.MTE2)
                    tla.copy(ub_x[bi], gm_row_x)
                    if tla.const_expr(HAS_HEAD == 1):
                        gm_row_h = tla.make_tensor(gm_head.ptr + row_off, row_gm)
                        tla.copy(ub_h[bi], gm_row_h)
                    row_mutex[bi].unlock(pipe=tla.pipes.MTE2)

                    row_mutex[bi].lock(pipe=tla.pipes.VECTOR)
                    # 行扫描：chunk 顺序推进，chunk 内 64 lane 树形扫描，跨 chunk 只
                    # 折叠一次 lane63 -> lane0 的进位。
                    with tla.vec.func(mode="simd"):
                        mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                        if tla.const_expr(TAIL != 0):
                            tail_mask, _ = tla.update_mask(TAIL, dtype=tla.Float32)
                        else:
                            tail_mask = mask

                        lane_i = tla.arange(base=0, order="increase", dtype=tla.Int32)
                        lane_f = lane_i.to(tla.Float32, _CAST_NONE, mask)
                        # lane0 / lane63 选择子：进位交接点与进位来源。
                        first = tla.cmp(lane_f, 0.0, "eq")
                        last = tla.cmp(lane_f, float(VL_ELE - 1), "eq")

                        carry = tla.full(0.0, dtype=tla.Float32)
                        hcarry = tla.full(0.0, dtype=tla.Float32)

                        for c in tla.range(n_chunks):
                            xc = tla.tile_view(
                                ub_x[bi],
                                tla.make_shape(1, VL_ELE),
                                tla.make_coord(0, c),
                            )
                            v = xc.load()
                            if tla.const_expr(HAS_HEAD == 1):
                                hc = tla.tile_view(
                                    ub_h[bi],
                                    tla.make_shape(1, VL_ELE),
                                    tla.make_coord(0, c),
                                )
                                h = _head_flag(hc.load())
                                v = tla.where(
                                    first,
                                    tla.where(
                                        tla.cmp(h, 1.0, "eq"),
                                        v,
                                        tla.add(v, carry, mask=mask),
                                    ),
                                    v,
                                )
                                h = tla.where(
                                    first,
                                    tla.where(tla.cmp(h, 1.0, "eq"), h, hcarry),
                                    h,
                                )
                                v, h = _scan_seg(v, h, lane_i, lane_f, mask)
                                xc.store(v)
                                carry = tla.full(
                                    v.reduce(tla.ReductionOp.MAX, mask=last),
                                    dtype=tla.Float32,
                                )
                                hcarry = tla.full(
                                    h.reduce(tla.ReductionOp.MAX, mask=last),
                                    dtype=tla.Float32,
                                )
                            else:
                                v = tla.where(first, tla.add(v, carry, mask=mask), v)
                                v = _scan_sum(v, lane_i, lane_f, mask)
                                xc.store(v)
                                carry = tla.full(
                                    v.reduce(tla.ReductionOp.MAX, mask=last),
                                    dtype=tla.Float32,
                                )

                        if tla.const_expr(TAIL != 0):
                            xt = tla.tile_view(
                                ub_x[bi],
                                tla.make_shape(1, VL_ELE),
                                tla.make_coord(0, n_chunks),
                            )
                            vt = xt.load()
                            # padding lane 先清零，再用全 lane mask 扫描（收窄 mask 会
                            # 缩短向量、破坏 64 lane gather），最后按 TAIL masked store。
                            pad = tla.cmp(lane_f, float(TAIL), "ge")
                            vt = tla.where(pad, tla.full(0.0, dtype=tla.Float32), vt)
                            if tla.const_expr(HAS_HEAD == 1):
                                ht_view = tla.tile_view(
                                    ub_h[bi],
                                    tla.make_shape(1, VL_ELE),
                                    tla.make_coord(0, n_chunks),
                                )
                                ht = _head_flag(ht_view.load())
                                ht = tla.where(
                                    pad, tla.full(0.0, dtype=tla.Float32), ht
                                )
                                vt = tla.where(
                                    first,
                                    tla.where(
                                        tla.cmp(ht, 1.0, "eq"),
                                        vt,
                                        tla.add(vt, carry, mask=mask),
                                    ),
                                    vt,
                                )
                                ht = tla.where(
                                    first,
                                    tla.where(tla.cmp(ht, 1.0, "eq"), ht, hcarry),
                                    ht,
                                )
                                vt, ht = _scan_seg(vt, ht, lane_i, lane_f, mask)
                            else:
                                vt = tla.where(first, tla.add(vt, carry, mask=mask), vt)
                                vt = _scan_sum(vt, lane_i, lane_f, mask)
                            xt.store(vt, mask=tail_mask)
                    row_mutex[bi].unlock(pipe=tla.pipes.VECTOR)

                    row_mutex[bi].lock(pipe=tla.pipes.MTE3)
                    tla.copy(gm_row_y, ub_x[bi])
                    row_mutex[bi].unlock(pipe=tla.pipes.MTE3)

        tla.pipe_barrier(tla.pipes.ALL)


# ======================================================================================
# HOST side: shape checks, compile and launch.
# ======================================================================================
# 性能设计要点：小 shape 的端到端耗时基本由 host 侧启动开销决定（from_dlpack 约
# 47 us/张量、get_device_properties 约 285 us、current_stream 约 11 us、payload 打包
# 约 10 us，而 kernel 本身只有几微秒），因此这里做三级缓存：
#
#   _COMPILED  按 (M, N, 是否分段) 缓存编译产物（含 BUFS 选择）；
#   _BINDINGS  按 (data_ptr, rows, cols) 直接构造 host 张量，绕开 from_dlpack；
#   _FAST      按 (张量地址, M, N, 是否分段) 缓存「绑定张量 + 预打包 payload +
#              partial 启动器」，命中时只剩一次字典查找 + 一次函数调用。
#
# 设备索引、stream、vector core 数同样缓存（键用 int，不用 str(device)）。

_COMPILED = {}
_VECTOR_CORES = {}
_BINDINGS = {}
_DUMMY = {}
_FAST = {}
_MAX_BINDINGS = 4096
_MAX_FAST = 256


def _vector_cores(index):
    cores = _VECTOR_CORES.get(index)
    if cores is None:
        cores = int(torch.npu.get_device_properties(index).vector_core_num)
        _VECTOR_CORES[index] = cores
    return cores


def _bind(buf, rows, cols):
    """Bind a contiguous fp32 torch buffer as a 2-D TLA host tensor."""
    sig = (buf.data_ptr(), rows, cols)
    view = _BINDINGS.get(sig)
    if view is not None:
        return view
    if buf.dtype is not torch.float32:
        return from_dlpack(buf, layout_tag=tla.arch.RowMajor)
    view = _Tensor(
        (rows, cols),
        tla.Float32,
        origin_shape=(rows, cols),
        coord=(0, 0),
        stride=(cols, 1),
        layout_tag=tla.arch.RowMajor,
        data_ptr=buf.data_ptr(),
    )
    if len(_BINDINGS) < _MAX_BINDINGS:
        _BINDINGS[sig] = view
    return view


def _dummy(device):
    key = device.index
    buf = _DUMMY.get(key)
    if buf is None:
        buf = torch.ones((1, 1), dtype=torch.float32, device=device)
        _DUMMY[key] = buf
    return buf


_STREAMS = {}
_LAUNCH_ACCEPTS_PRINT = None


def reset_stream_cache():
    """Forget the cached streams and the launch plans bound to them."""
    _STREAMS.clear()
    _FAST.clear()


def _launch_stream(index):
    key = (threading.get_ident(), index)
    stream = _STREAMS.get(key)
    if stream is None:
        stream = int(torch.npu.current_stream(index).npu_stream)
        _STREAMS[key] = stream
    return stream


def _resolve_launcher(artifact):
    try:
        from catlass import execution as execution_mod
        from catlass.base_dsl.jit_executor import _resolved_dyn_ubuf_bytes

        executor = artifact._get_executor()
        module = executor.jit_module
        if getattr(module, "uses_scalar_print", False) or getattr(
            module, "uses_tensor_print", False
        ):
            return None
        return (
            execution_mod,
            module.execution_args,
            _resolved_dyn_ubuf_bytes(module, 0),
            executor.function_handle,
            module,
        )
    except Exception:
        return None


def _make_launcher(artifact, tensors, block_num, stream):
    """Return a zero-argument launcher that reuses the packed payload."""
    plumbing = _resolve_launcher(artifact)
    if plumbing is None:
        return None
    execution_mod, execution_args, dyn_ub, function_handle, module = plumbing
    try:
        payload = execution_args.generate_launch_payload(tensors)
    except Exception:
        return None
    if payload is None:
        return None

    global _LAUNCH_ACCEPTS_PRINT
    if _LAUNCH_ACCEPTS_PRINT is None:
        parameters = inspect.signature(execution_mod.launch_kernel).parameters
        _LAUNCH_ACCEPTS_PRINT = "uses_scalar_print" in parameters
    kwargs = {
        "function_handle": function_handle,
        "stream": stream,
        "block_num": block_num,
        "payload": payload,
        "dyn_ubuf_bytes": dyn_ub,
    }
    if _LAUNCH_ACCEPTS_PRINT:
        kwargs["uses_scalar_print"] = getattr(module, "uses_scalar_print", False)
    return functools.partial(execution_mod.launch_kernel, **kwargs)


def _pipeline(n_pad, has_head):
    """Pick the number of in-flight row buffers from the UB budget.

    One buffer is ``n_pad`` fp32 values (plus the same amount for the segment
    mask), so a wide row naturally gets a shallow pipeline and a narrow row gets
    the full four buffers.
    """
    per_buf = 4 * n_pad * (2 if has_head else 1)
    return max(1, min(4, (UB_BYTES - 8192) // per_buf))


def run(x, head=None):
    """Return the inclusive prefix sum of ``x`` along the last axis (fp32).

    ``head`` is an optional ``[M, N]`` tensor/mask whose non-zero entries start a
    new segment, i.e. the accumulation is reset at that position.
    """
    if x.dtype is not torch.float32:
        raise ValueError("x must be float32")
    if x.dim() != 2:
        raise ValueError("x must be 2-D [M, N]")
    m, n = x.shape
    n_pad = (n + VL_ELE - 1) // VL_ELE * VL_ELE
    tail = n - (n // VL_ELE) * VL_ELE

    if not x.is_contiguous():
        x = x.contiguous()
    device = x.device
    has_head = 0 if head is None else 1
    if has_head:
        if tuple(head.shape) != (m, n):
            raise ValueError("head must have the same shape as x")
        if head.dtype is torch.float32 and head.is_contiguous():
            hbuf = head
        else:
            hbuf = (head != 0).to(torch.float32).contiguous()
    else:
        hbuf = _dummy(device)

    out = torch.empty((m, n), dtype=torch.float32, device=device)
    if m == 0:
        return out

    key = (x.data_ptr(), hbuf.data_ptr(), out.data_ptr(), m, n, has_head)
    plan = _FAST.get(key)
    if plan is not None:
        launch = plan[0]
        if launch is not None:
            launch()
        else:
            plan[1](*plan[2], block_num=plan[3], stream=plan[4])
        return out

    bufs = _pipeline(n_pad, has_head)
    compile_key = (m, n, has_head)
    entry = _COMPILED.get(compile_key)
    if entry is None:
        artifact = tla.compile(
            scan_kernel,
            _bind(x, m, n),
            _bind(hbuf, m, n) if has_head else _bind(hbuf, 1, 1),
            _bind(out, m, n),
            n,
            n_pad,
            tail,
            has_head,
            bufs,
            options="--npu-arch 3510",
        )
        entry = artifact
        _COMPILED[compile_key] = entry
    artifact = entry

    device_index = device.index
    cores = _vector_cores(device_index)
    stream = _launch_stream(device_index)
    tensors = (
        _bind(x, m, n),
        _bind(hbuf, m, n) if has_head else _bind(hbuf, 1, 1),
        _bind(out, m, n),
    )
    plan = (
        _make_launcher(artifact, tensors, cores, stream),
        artifact,
        tensors,
        cores,
        stream,
    )
    if len(_FAST) >= _MAX_FAST:
        _FAST.clear()
    _FAST[key] = plan

    launch = plan[0]
    if launch is not None:
        launch()
    else:
        artifact(*tensors, block_num=cores, stream=stream)
    return out


if __name__ == "__main__":
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    torch.manual_seed(0)

    # 对齐 / 非 32 B 对齐 / 小于一个向量的 N 都要走通，并覆盖 M 非整除核数。
    for m, n in ((64, 512), (37, 1000), (129, 65), (5, 63), (1, 1), (33, 4096)):
        x = torch.randn(m, n, dtype=torch.float32, device="npu")
        y = run(x)
        err = (y.cpu() - torch.cumsum(x.cpu(), dim=-1)).abs().max().item()
        print("cumsum", (m, n), "err", err)

    for m, n in ((32, 256), (17, 300), (9, 61), (3, 1000)):
        x = torch.randn(m, n, dtype=torch.float32, device="npu")
        # Segment starts every 37 columns plus a few random ones.
        head = torch.zeros(m, n, dtype=torch.float32, device="npu")
        head[:, ::37] = 1.0
        head[torch.randint(0, m, (4,)), torch.randint(0, n, (4,))] = 1.0
        y = run(x, head)

        xc, hc = x.cpu(), head.cpu()
        ref = torch.zeros_like(xc)
        acc = torch.zeros(m)
        for j in range(n):
            acc = torch.where(hc[:, j] != 0, xc[:, j], acc + xc[:, j])
            ref[:, j] = acc
        err = (y.cpu() - ref).abs().max().item()
        print("segmented", (m, n), "err", err)

    # head 的契约是「非 0 即段首」，非 1.0 的非零值同样要触发清零。
    m, n = 8, 200
    x = torch.randn(m, n, dtype=torch.float32, device="npu")
    head = torch.zeros(m, n, dtype=torch.float32, device="npu")
    head[:, ::29] = 0.5
    head[0, 7] = -2.0
    head[1, 133] = 3.5
    y = run(x, head)
    xc, hc = x.cpu(), head.cpu()
    ref = torch.zeros_like(xc)
    acc = torch.zeros(m)
    for j in range(n):
        acc = torch.where(hc[:, j] != 0, xc[:, j], acc + xc[:, j])
        ref[:, j] = acc
    print("segmented non-binary head err", (y.cpu() - ref).abs().max().item())
