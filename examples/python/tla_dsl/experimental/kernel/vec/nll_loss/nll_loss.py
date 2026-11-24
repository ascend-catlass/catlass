# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Fused FP32 NLLLoss / CrossEntropy forward for Ascend AIV.

The implementation keeps the C-wide work on device and the tiny row reduction
on host:

    direct:  loss_i = -x[i, t_i] * w[t_i]
    logits:  loss_i = (lse_i - x[i, t_i]) * w[t_i]

High-performance design:

* rows are striped over AIV blocks, so every row has one owner and no
  cross-block reduction or atomic update is needed;
* one C-element row is staged in UB and scanned in 64-FP32-lane chunks. The
  logits path performs one stable max, one exp/sum, and one scalar reduction;
* x and loss use separate mutexes. MTE2->VECTOR owns the input row, while
  VECTOR->MTE3 owns the one-element output slot. The two resources can move
  independently, and the next row naturally back-pressures on buffer reuse;
* ``ignore_index`` is sanitized on host because the DSL has no scalar select.
  The kernel therefore never sees an out-of-range target, and host performs
  only the O(N) final reduction;
* compiled artifacts and vector-core counts are cached by shape and device.

Constraints: ``x`` and ``weight`` are FP32, ``target`` is int64, ``C`` is
positive, and supported targets are in ``[0, C)`` or equal to
``ignore_index``. Non-64-multiple rows use a masked tail chunk.
"""

import functools
import inspect
import threading

import catlass.tla as tla
from catlass.tla.runtime import _Tensor

VL_ELE = 64


@tla.kernel
def nll_loss_kernel(
    gm_x: tla.Tensor,  # [N, C] fp32
    gm_t: tla.Tensor,  # [N] int64, ignore_index already rewritten to 0
    gm_w: tla.Tensor,  # [C] fp32 (unused when HAS_WEIGHT == 0)
    gm_loss: tla.Tensor,  # [N] fp32
    C: tla.Constexpr[int],
    C_PAD: tla.Constexpr[int],
    TAIL: tla.Constexpr[int],
    HAS_WEIGHT: tla.Constexpr[int],
    FROM_LOGITS: tla.Constexpr[int],
) -> None:
    rows = gm_x.origin_shape[0]
    n_chunks = C // VL_ELE

    # Two row slots form a ping-pong pipeline. While VECTOR scans slot i, MTE2
    # can fill slot 1-i and MTE3 can drain the previous one-element result.
    # Input and output ownership are separate, so the scalar drain never blocks
    # the next row's max/exp/sum scan.
    x_mutex0 = tla.mutex(resource="nll_x0", id=0)
    x_mutex1 = tla.mutex(resource="nll_x1", id=1)
    loss_mutex0 = tla.mutex(resource="nll_loss0", id=2)
    loss_mutex1 = tla.mutex(resource="nll_loss1", id=3)

    # The GM row is dense, while the UB rows are padded to a complete 64-lane
    # vector so the final masked chunk can be loaded without over-reading GM.
    x_ptr0 = tla.allocate(C_PAD, tla.Float32, tla.AddressSpace.ub, 256)
    x_ptr1 = tla.allocate(C_PAD, tla.Float32, tla.AddressSpace.ub, 256)
    loss_ptr0 = tla.allocate(1, tla.Float32, tla.AddressSpace.ub, 256)
    loss_ptr1 = tla.allocate(1, tla.Float32, tla.AddressSpace.ub, 256)

    row_layout = tla.make_layout(
        tla.make_shape(1, C),
        tla.make_stride(C_PAD, 1),
        origin_shape=tla.make_shape(1, C),
    )
    one_layout = tla.make_layout(
        tla.make_shape(1, 1),
        tla.make_stride(1, 1),
        origin_shape=tla.make_shape(1, 1),
    )
    ub_x0 = tla.make_tensor(x_ptr0, row_layout)
    ub_x1 = tla.make_tensor(x_ptr1, row_layout)
    ub_loss0 = tla.make_tensor(loss_ptr0, one_layout)
    ub_loss1 = tla.make_tensor(loss_ptr1, one_layout)

    gid = tla.arch.block_idx()
    aiv_num = tla.arch.block_num()

    with tla.vector():
        buf_idx = tla.as_numeric(0)
        for r in tla.range(gid, rows, aiv_num):
            ub_x = ub_x0 if buf_idx == 0 else ub_x1
            ub_loss = ub_loss0 if buf_idx == 0 else ub_loss1
            x_mutex = x_mutex0 if buf_idx == 0 else x_mutex1
            loss_mutex = loss_mutex0 if buf_idx == 0 else loss_mutex1
            row_off = r.to(tla.Int64) * tla.Int64(C)
            gm_row = tla.make_tensor(gm_x.ptr + row_off, row_layout)
            gm_out = tla.make_tensor(gm_loss.ptr + r.to(tla.Int64), one_layout)

            if tla.const_expr(FROM_LOGITS == 1):
                x_mutex.lock(pipe=tla.arch.MTE2)
                tla.copy(ub_x, gm_row)
                x_mutex.unlock(pipe=tla.arch.MTE2)
                x_mutex.lock(pipe=tla.arch.VECTOR)
            loss_mutex.lock(pipe=tla.arch.VECTOR)
            with tla.vec.func(mode="simd"):
                mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                tail_mask = None
                if tla.const_expr(TAIL != 0):
                    tail_mask, _ = tla.update_mask(TAIL, dtype=tla.Float32)

                if tla.const_expr(FROM_LOGITS == 1):
                    amax = tla.full(-3.4e38, dtype=tla.Float32)
                    # Keep long-C rows as a dynamic loop. Static unrolling of
                    # every 64-lane chunk overflows the VF stack for C >= 4096.
                    for c_max in tla.range(n_chunks):
                        xc_max = tla.tile_view(
                            ub_x,
                            tla.make_shape(1, VL_ELE),
                            tla.make_coord(0, c_max),
                        )
                        amax = tla.max(amax, xc_max.load(), mask=mask)
                    max_b = tla.full(
                        amax.reduce(tla.ReductionOp.MAX, mask=mask),
                        dtype=tla.Float32,
                    )
                    if tla.const_expr(TAIL != 0):
                        xc_tail = tla.tile_view(
                            ub_x,
                            tla.make_shape(1, VL_ELE),
                            tla.make_coord(0, n_chunks),
                        )
                        tail_max = tla.full(
                            tla.max(
                                tla.full(-3.4e38, dtype=tla.Float32),
                                xc_tail.load(),
                                mask=tail_mask,
                            ).reduce(tla.ReductionOp.MAX, mask=tail_mask),
                            dtype=tla.Float32,
                        )
                        max_b = tla.max(max_b, tail_max, mask=mask)

                    acc = tla.full(0.0, dtype=tla.Float32)
                    for c_sum in tla.range(n_chunks):
                        xc_sum = tla.tile_view(
                            ub_x,
                            tla.make_shape(1, VL_ELE),
                            tla.make_coord(0, c_sum),
                        )
                        acc = tla.add(
                            acc,
                            tla.exp(
                                tla.sub(xc_sum.load(), max_b, mask=mask),
                                mask=mask,
                            ),
                            mask=mask,
                        )
                    sum_b = tla.full(
                        acc.reduce(tla.ReductionOp.ADD, mask=mask), dtype=tla.Float32
                    )
                    if tla.const_expr(TAIL != 0):
                        xc_tail = tla.tile_view(
                            ub_x,
                            tla.make_shape(1, VL_ELE),
                            tla.make_coord(0, n_chunks),
                        )
                        tail_sum = tla.full(
                            tla.exp(
                                tla.sub(xc_tail.load(), max_b, mask=tail_mask),
                                mask=tail_mask,
                            ).reduce(tla.ReductionOp.ADD, mask=tail_mask),
                            dtype=tla.Float32,
                        )
                        sum_b = tla.add(sum_b, tail_sum, mask=mask)
                    lse = tla.add(max_b, tla.log(sum_b, mask=mask), mask=mask)
                    tgt = gm_x[r, gm_t[r]]
                    loss = tla.sub(lse, tla.full(tgt, dtype=tla.Float32), mask=mask)
                else:
                    loss = tla.neg(tla.full(gm_x[r, gm_t[r]], dtype=tla.Float32))

                if tla.const_expr(HAS_WEIGHT == 1):
                    loss = tla.mul(
                        loss, tla.full(gm_w[gm_t[r]], dtype=tla.Float32), mask=mask
                    )

                ub_loss.store(loss.reduce(tla.ReductionOp.MAX, mask=mask))
            loss_mutex.unlock(pipe=tla.arch.VECTOR)
            if tla.const_expr(FROM_LOGITS == 1):
                x_mutex.unlock(pipe=tla.arch.VECTOR)

            loss_mutex.lock(pipe=tla.arch.MTE3)
            tla.copy(gm_out, ub_loss)
            loss_mutex.unlock(pipe=tla.arch.MTE3)
            buf_idx = tla.as_numeric(1) - buf_idx

        tla.pipe_barrier(tla.pipes.ALL)


# ======================================================================================
# HOST side
# ======================================================================================

_COMPILED = {}
_VECTOR_CORES = {}
_BINDINGS = {}
_LAUNCH_PLANS = {}
_FAST_RUNS = {}
_LAST_RUN = None
_STREAM_OBJECTS = {}
_TARGET_SAFE = {}
_SCRATCH_LOSS = {}
_DUMMY_WEIGHT = {}
_LAUNCH_ACCEPTS_PRINT = None
_MAX_BINDINGS = 512
_MAX_LAUNCH_PLANS = 512
_MAX_TARGET_SAFE = 256
_MAX_SCRATCH_LOSS = 64
_MAX_STREAM_OBJECTS = 64


def _vector_cores(index):
    import torch

    cores = _VECTOR_CORES.get(index)
    if cores is None:
        cores = int(torch.npu.get_device_properties(index).vector_core_num)
        _VECTOR_CORES[index] = cores
    return cores


def _current_stream(device_index):
    """Query the current stream and cache only reusable Stream wrappers."""
    import torch
    import torch_npu

    try:
        raw = torch_npu._C._npu_getCurrentStream(device_index)
    except Exception:
        return int(torch.npu.current_stream(device_index).npu_stream)
    stream = _STREAM_OBJECTS.get(raw)
    if stream is None:
        if len(_STREAM_OBJECTS) >= _MAX_STREAM_OBJECTS:
            _STREAM_OBJECTS.clear()
        stream = torch.npu.Stream(
            stream_id=raw[0],
            device_index=raw[1],
            device_type=raw[2],
        )
        _STREAM_OBJECTS[raw] = stream
    return int(stream.npu_stream)


def _bind(tensor, shape, dtype, stride):
    """Build a dense TLA descriptor directly from the tensor storage."""
    key = (tensor.data_ptr(), shape, stride, str(dtype))
    bound = _BINDINGS.get(key)
    if bound is None:
        bound = _Tensor(
            shape,
            dtype,
            origin_shape=shape,
            coord=tuple(0 for _ in shape),
            stride=stride,
            layout_tag=tla.arch.RowMajor,
            data_ptr=tensor.data_ptr(),
        )
        if len(_BINDINGS) >= _MAX_BINDINGS:
            _BINDINGS.clear()
        _BINDINGS[key] = bound
    return bound


def _prepare_target(target, c, ignore_index):
    """Sanitize ignored targets once per tensor version."""
    import torch

    key = (
        id(target),
        target.data_ptr(),
        int(target._version),
        target.numel(),
        c,
        ignore_index,
    )
    cached = _TARGET_SAFE.get(key)
    if cached is not None:
        return cached[:3]

    t_valid = target.to(torch.int64).contiguous()
    valid = torch.ones_like(t_valid, dtype=torch.bool)
    if ignore_index is not None:
        valid = t_valid != ignore_index
    invalid = (t_valid < 0) | (t_valid >= c)
    if ignore_index is not None:
        invalid = invalid & valid
    if torch.any(invalid):
        raise ValueError("target values must be in [0, C) or equal ignore_index")
    has_ignored = bool((~valid).any().item())
    t_safe = (
        torch.where(valid, t_valid, torch.zeros_like(t_valid))
        if has_ignored
        else t_valid
    )

    if len(_TARGET_SAFE) >= _MAX_TARGET_SAFE:
        _TARGET_SAFE.clear()
    _TARGET_SAFE[key] = (t_safe, valid, has_ignored, target)
    return t_safe, valid, has_ignored


def _scratch_loss(device, n, reduction, device_index, stream):
    """Reuse the row-result buffer for scalar reductions on one stream."""
    import torch

    if reduction == "none":
        return torch.empty(n, dtype=torch.float32, device=device)
    key = (
        threading.get_ident(),
        device_index,
        n,
        stream,
    )
    loss = _SCRATCH_LOSS.get(key)
    if loss is None:
        loss = torch.empty(n, dtype=torch.float32, device=device)
        if len(_SCRATCH_LOSS) >= _MAX_SCRATCH_LOSS:
            _SCRATCH_LOSS.clear()
        _SCRATCH_LOSS[key] = loss
    return loss


def _dummy_weight(device, device_index):
    import torch

    weight = _DUMMY_WEIGHT.get(device_index)
    if weight is None:
        weight = torch.zeros(1, dtype=torch.float32, device=device)
        _DUMMY_WEIGHT[device_index] = weight
    return weight


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
    """Pre-pack the CATLASS 2.2.1 launch payload for repeated calls."""
    plumbing = _resolve_launcher(artifact)
    if plumbing is None:
        return None
    execution_mod, args, dyn_ub, function_handle, module = plumbing
    try:
        payload = args.generate_launch_payload(tensors)
    except Exception:
        return None
    if payload is None:
        return None

    try:
        global _LAUNCH_ACCEPTS_PRINT
        if _LAUNCH_ACCEPTS_PRINT is None:
            _LAUNCH_ACCEPTS_PRINT = (
                "uses_scalar_print"
                in inspect.signature(execution_mod.launch_kernel).parameters
            )
    except Exception:
        return None
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


def _finish_reduction(
    loss,
    valid,
    has_ignored,
    has_weight,
    weight,
    t_safe,
    n,
    reduction,
):
    import torch

    if reduction == "none":
        if has_ignored:
            return torch.where(valid, loss, torch.zeros_like(loss))
        return loss
    if reduction == "sum":
        return loss[valid].sum() if has_ignored else loss.sum()
    numerator = loss[valid].sum() if has_ignored else loss.sum()
    if has_weight:
        target_weights = weight[t_safe[valid]] if has_ignored else weight[t_safe]
        return numerator / target_weights.sum()
    count = valid.sum() if has_ignored else n
    return numerator / count


def run(x, target, weight=None, ignore_index=-100, reduction="mean", from_logits=False):
    """NLLLoss / CrossEntropy over ``[N, C]``; ``reduction`` in {none, sum, mean}."""
    global _LAST_RUN
    import torch

    if x.dtype != torch.float32:
        raise ValueError("x must be float32")
    if x.dim() != 2:
        raise ValueError("x must be 2-D [N, C]")
    if target.dim() != 1 or target.numel() != x.shape[0]:
        raise ValueError("target must be [N]")
    if target.device != x.device:
        raise ValueError("target must be on the same device as x")
    n, c = x.shape
    if c < 1:
        raise ValueError("C must be >= 1")
    if reduction not in ("none", "sum", "mean"):
        raise ValueError("reduction must be none/sum/mean")

    device = x.device
    device_index = int(str(device).split(":")[-1]) if ":" in str(device) else 0
    stream = _current_stream(device_index)
    x_version = int(x._version)
    weight_version = int(weight._version) if weight is not None else -1

    last = _LAST_RUN
    if (
        last is not None
        and last["input_x"] is x
        and last["input_target"] is target
        and last["input_weight"] is weight
        and last["x_version"] == x_version
        and last["target_version"] == int(target._version)
        and last["weight_version"] == weight_version
        and last["ignore_index"] == ignore_index
        and last["reduction"] == reduction
        and last["from_logits"] == bool(from_logits)
        and last["thread"] == threading.get_ident()
        and last["stream"] == stream
    ):
        last["launch"]()
        loss = last["loss"]
        if reduction == "sum":
            return loss.sum() if not last["has_ignored"] else loss[last["valid"]].sum()
        if reduction == "mean":
            if not last["has_weight"]:
                return (
                    loss.mean()
                    if not last["has_ignored"]
                    else loss[last["valid"]].mean()
                )
            numerator = (
                loss.sum() if not last["has_ignored"] else loss[last["valid"]].sum()
            )
            target_weights = (
                last["weight"][last["t_safe"]]
                if not last["has_ignored"]
                else last["weight"][last["t_safe"][last["valid"]]]
            )
            return numerator / target_weights.sum()
        return _finish_reduction(
            loss,
            last["valid"],
            last["has_ignored"],
            last["has_weight"],
            last["weight"],
            last["t_safe"],
            last["n"],
            reduction,
        )

    fast_key = (
        id(x),
        id(target),
        id(weight),
        tuple(x.shape),
        tuple(target.shape),
        None if weight is None else tuple(weight.shape),
        x_version,
        int(target._version),
        weight_version,
        ignore_index,
        reduction,
        bool(from_logits),
        stream,
    )
    fast_plan = _FAST_RUNS.get(fast_key) if reduction != "none" else None
    if fast_plan is not None:
        fast_plan["launch"]()
        return _finish_reduction(
            fast_plan["loss"],
            fast_plan["valid"],
            fast_plan["has_ignored"],
            fast_plan["has_weight"],
            fast_plan["weight"],
            fast_plan["t_safe"],
            fast_plan["n"],
            reduction,
        )

    c_pad = (c + VL_ELE - 1) // VL_ELE * VL_ELE
    tail = c - (c // VL_ELE) * VL_ELE

    xc = x.contiguous()
    t_safe, valid, has_ignored = _prepare_target(target, c, ignore_index)

    has_weight = 0 if weight is None else 1
    if has_weight:
        if weight.numel() != c:
            raise ValueError("weight must have C elements")
        if weight.device != x.device:
            raise ValueError("weight must be on the same device as x")
        w = weight.to(torch.float32).contiguous()
    else:
        w = _dummy_weight(device, device_index)

    if n == 0:
        loss = torch.empty(0, dtype=torch.float32, device=device)
        return loss if reduction == "none" else loss.sum()

    loss = _scratch_loss(device, n, reduction, device_index, stream)

    key = (n, c, has_weight, int(from_logits), device_index)
    artifact = _COMPILED.get(key)
    if artifact is None:
        x_t = _bind(xc, (n, c), tla.Float32, (c, 1))
        t_t = _bind(t_safe, (n,), tla.Int64, (1,))
        w_shape = (c,) if has_weight else (1,)
        w_t = _bind(w, w_shape, tla.Float32, (1,))
        l_t = _bind(loss, (n,), tla.Float32, (1,))
        artifact = tla.compile(
            nll_loss_kernel,
            x_t,
            t_t,
            w_t,
            l_t,
            c,
            c_pad,
            tail,
            has_weight,
            int(from_logits),
            options="--npu-arch 3510",
        )
        _COMPILED[key] = artifact

    launch_key = (
        xc.data_ptr(),
        t_safe.data_ptr(),
        w.data_ptr(),
        loss.data_ptr(),
        n,
        c,
        has_weight,
        int(from_logits),
        device_index,
        stream,
    )
    plan = _LAUNCH_PLANS.get(launch_key)
    if plan is None:
        x_t = _bind(xc, (n, c), tla.Float32, (c, 1))
        t_t = _bind(t_safe, (n,), tla.Int64, (1,))
        w_shape = (c,) if has_weight else (1,)
        w_t = _bind(w, w_shape, tla.Float32, (1,))
        l_t = _bind(loss, (n,), tla.Float32, (1,))
        cores = _vector_cores(device_index)
        tensors = (x_t, t_t, w_t, l_t)
        plan = {
            "artifact": artifact,
            "tensors": tensors,
            "cores": cores,
            "launcher": _make_launcher(
                artifact,
                tensors,
                cores,
                stream,
            ),
        }
        if len(_LAUNCH_PLANS) >= _MAX_LAUNCH_PLANS:
            _LAUNCH_PLANS.clear()
        _LAUNCH_PLANS[launch_key] = plan

    if plan["launcher"] is not None:
        plan["launcher"]()
    else:
        plan["artifact"](
            *plan["tensors"],
            block_num=plan["cores"],
        )

    if reduction != "none":
        if len(_FAST_RUNS) >= _MAX_LAUNCH_PLANS:
            _FAST_RUNS.clear()
        new_plan = {
            "x": xc,
            "target": t_safe,
            "weight": w,
            "loss": loss,
            "valid": valid,
            "t_safe": t_safe,
            "has_ignored": has_ignored,
            "has_weight": has_weight,
            "n": n,
            "input_x": x,
            "input_target": target,
            "input_weight": weight,
            "x_version": x_version,
            "target_version": int(target._version),
            "weight_version": weight_version,
            "ignore_index": ignore_index,
            "reduction": reduction,
            "from_logits": bool(from_logits),
            "thread": threading.get_ident(),
            "stream": stream,
            "launch": plan["launcher"]
            if plan["launcher"] is not None
            else functools.partial(
                plan["artifact"],
                *plan["tensors"],
                block_num=plan["cores"],
            ),
        }
        _FAST_RUNS[fast_key] = new_plan
        _LAST_RUN = new_plan
    return _finish_reduction(
        loss,
        valid,
        has_ignored,
        has_weight,
        w,
        t_safe,
        n,
        reduction,
    )


if __name__ == "__main__":
    import torch
    import torch_npu
    import torch.nn.functional as F

    torch.npu.set_device(0)
    torch.manual_seed(0)

    def check_error(name, error, limit):
        print(name, error)
        if not error <= limit:
            raise RuntimeError(f"{name}: {error} > {limit}")

    n, c = 129, 256
    logits = torch.randn(n, c, dtype=torch.float32, device="npu")
    target = torch.randint(0, c, (n,), device="npu")
    target[::7] = -100

    y = run(logits, target, from_logits=True)
    ref = F.cross_entropy(logits.cpu(), target.cpu(), ignore_index=-100)
    check_error("cross_entropy mean err", abs(y.item() - ref.item()), 2e-5)

    # Repeated calls exercise _LAST_RUN and _FAST_RUNS with the same buffers.
    y_again = run(logits, target, from_logits=True)
    y_sum = run(logits, target, from_logits=True, reduction="sum")
    ref_sum = F.cross_entropy(
        logits.cpu(), target.cpu(), ignore_index=-100, reduction="sum"
    )
    y_cached = run(logits, target, from_logits=True)
    torch.npu.synchronize()
    check_error(
        "cached-path mean err",
        max(abs(y_again.item() - ref.item()), abs(y_cached.item() - ref.item())),
        2e-5,
    )
    check_error("cached-path sum err", abs(y_sum.item() - ref_sum.item()), 2e-5)

    other_stream = torch.npu.Stream(device=logits.device)
    with torch.npu.stream(other_stream):
        y_stream = run(logits, target, from_logits=True)
    torch.npu.synchronize()
    check_error("non-default stream err", abs(y_stream.item() - ref.item()), 2e-5)

    # x is a non-contiguous view and weight needs an FP32 copy. Updating either
    # source must invalidate the cached plan instead of reusing stale copies.
    wide = torch.randn(n, 2 * c, dtype=torch.float32, device="npu")
    x_view = wide[:, :c]
    x_ref = x_view.clone()
    w_half = torch.randn(c, dtype=torch.float16, device="npu")
    y = run(x_view, target, weight=w_half, from_logits=True)
    ref = F.cross_entropy(x_ref.cpu(), target.cpu(), weight=w_half.float().cpu())
    x_view.copy_(torch.randn_like(x_view))
    w_half.copy_(torch.randn_like(w_half))
    y_updated = run(x_view, target, weight=w_half, from_logits=True)
    ref_updated = F.cross_entropy(
        x_view.cpu(), target.cpu(), weight=w_half.float().cpu()
    )
    torch.npu.synchronize()
    check_error("source-update initial err", abs(y.item() - ref.item()), 2e-5)
    check_error(
        "source-update refreshed err",
        abs(y_updated.item() - ref_updated.item()),
        2e-5,
    )

    logp = F.log_softmax(logits.cpu(), dim=-1).npu()
    y = run(logp, target)
    ref = F.nll_loss(logp.cpu(), target.cpu(), ignore_index=-100)
    check_error("nll_loss mean err", abs(y.item() - ref.item()), 2e-5)

    w = torch.randn(c, dtype=torch.float32, device="npu")
    y = run(logits, target, weight=w, from_logits=True, reduction="sum")
    ref = F.cross_entropy(
        logits.cpu(), target.cpu(), weight=w.cpu(), ignore_index=-100, reduction="sum"
    )
    check_error("cross_entropy weighted sum err", abs(y.item() - ref.item()), 1e-3)

    y = run(logits, target, weight=w, from_logits=True)
    ref = F.cross_entropy(logits.cpu(), target.cpu(), weight=w.cpu(), ignore_index=-100)
    check_error("cross_entropy weighted mean err", abs(y.item() - ref.item()), 2e-5)

    y = run(logits, target, from_logits=True, reduction="none")
    ref = F.cross_entropy(
        logits.cpu(), target.cpu(), ignore_index=-100, reduction="none"
    )
    check_error(
        "cross_entropy none err",
        (y.cpu() - ref).abs().max().item(),
        2e-5,
    )
