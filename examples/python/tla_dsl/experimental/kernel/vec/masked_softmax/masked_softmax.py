# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Row-wise numerically stable softmax with an optional additive mask/bias.

    y[r] = softmax(scale * (x[r] + add[r]))

The additive term is the usual pre-softmax bias (``0`` for keep, a large
negative number for mask-off).

High-performance design:

* One AIV block owns a complete row. Row max and sum therefore never require
  a cross-block reduction or a GM round trip.
* The row is copied into UB once. Scale+bias, exp+sum, and normalize are three
  register passes over that same buffer, so the normalized values and both
  statistics stay on chip.
* All arithmetic is FP32 and every tile is a full 64-lane vector. ``SCALE`` and
  ``HAS_ADD`` are compile-time constants, allowing the host and add branches to
  disappear from the generated kernel.
* Rows are strided across all vector cores instead of assigning contiguous row
  groups to a core. This keeps the work distribution balanced when M is not a
  multiple of the AIV count.
"""

import weakref

import catlass.tla as tla

VL_ELE = 64  # One FP32 vector register covers 64 lanes (256 B).
MIN_VALUE = -3.0e38


@tla.kernel
def masked_softmax_kernel(
    gm_x: tla.Tensor,  # [M, N] fp32
    gm_add: tla.Tensor,  # [M, N] fp32 additive bias (unused when HAS_ADD == 0)
    gm_y: tla.Tensor,  # [M, N] fp32
    N: tla.Constexpr[int],
    SCALE: tla.Constexpr[float],
    HAS_ADD: tla.Constexpr[int],
) -> None:
    m = gm_x.origin_shape[0]

    # Each mutex represents ownership of one physical UB resource, not a pipe
    # pair. This lets MTE2, VECTOR and MTE3 express fill -> consume -> drain
    # without integer-flag bookkeeping.
    mutex_x = tla.mutex(resource="masked_softmax_x", id=0)
    mutex_add = tla.mutex(resource="masked_softmax_add", id=1)

    # Two physical UB rows at most: input/logits and the optional additive term.
    # The x buffer is reused for scaled logits, exponentials and final output.
    x_ptr = tla.allocate(N, tla.Float32, tla.AddressSpace.ub, 256)
    a_ptr = tla.allocate(N, tla.Float32, tla.AddressSpace.ub, 256)
    row_layout = tla.make_layout(
        tla.make_shape(1, N),
        tla.make_stride(N, 1),
        origin_shape=tla.make_shape(1, N),
    )
    ub_x = tla.make_tensor(x_ptr, row_layout)
    ub_add = tla.make_tensor(a_ptr, row_layout)

    n_chunks = N // VL_ELE
    gid = tla.arch.block_idx()
    aiv_num = tla.arch.block_num()

    with tla.vector():
        for r in tla.range(gid, m, aiv_num):
            # A static 1xN view keeps the GM access contiguous and lets the
            # compiler fold the row stride into a single scalar address.
            row_off = r.to(tla.Int64) * tla.Int64(N)
            gm_row_x = tla.make_tensor(gm_x.ptr + row_off, row_layout)
            gm_row_y = tla.make_tensor(gm_y.ptr + row_off, row_layout)

            # MTE2 owns ub_x until both x and (when present) add are resident.
            mutex_x.lock(pipe=tla.arch.MTE2)
            tla.copy(ub_x, gm_row_x)
            mutex_x.unlock(pipe=tla.arch.MTE2)
            if tla.const_expr(HAS_ADD == 1):
                gm_row_add = tla.make_tensor(gm_add.ptr + row_off, row_layout)
                mutex_add.lock(pipe=tla.arch.MTE2)
                tla.copy(ub_add, gm_row_add)
                mutex_add.unlock(pipe=tla.arch.MTE2)

            # Acquire both inputs before the three-pass register computation.
            mutex_x.lock(pipe=tla.arch.VECTOR)
            if tla.const_expr(HAS_ADD == 1):
                mutex_add.lock(pipe=tla.arch.VECTOR)
            with tla.vec.func(mode="simd"):
                mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)

                # Pass 1: y = scale * (x + add); track the row max. Scaling
                # happens before the max so the sum below uses the same domain.
                mx = tla.full(MIN_VALUE, dtype=tla.Float32)
                for c in tla.range_constexpr(n_chunks):
                    xc = tla.tile_view(
                        ub_x, tla.make_shape(1, VL_ELE), tla.make_coord(0, c)
                    )
                    reg = xc.load()
                    if tla.const_expr(HAS_ADD == 1):
                        ac = tla.tile_view(
                            ub_add, tla.make_shape(1, VL_ELE), tla.make_coord(0, c)
                        )
                        reg = tla.add(reg, ac.load(), mask=mask)
                    reg = tla.mul(reg, SCALE, mask=mask)
                    xc.store(reg)
                    mx = tla.max(mx, reg, mask=mask)
                max_b = tla.full(
                    mx.reduce(tla.ReductionOp.MAX, mask=mask), dtype=tla.Float32
                )

                tla.local_mem_bar(
                    tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                )

                # Pass 2: e = exp(y - max); accumulate the row sum. exp is
                # evaluated directly from the scaled buffer, avoiding a second
                # GM read of x.
                acc = tla.full(0.0, dtype=tla.Float32)
                for c in tla.range_constexpr(n_chunks):
                    xc = tla.tile_view(
                        ub_x, tla.make_shape(1, VL_ELE), tla.make_coord(0, c)
                    )
                    e = tla.exp(tla.sub(xc.load(), max_b, mask=mask), mask=mask)
                    xc.store(e)
                    acc = tla.add(acc, e, mask=mask)
                den = tla.full(
                    acc.reduce(tla.ReductionOp.ADD, mask=mask), dtype=tla.Float32
                )
                inv = tla.div(tla.full(1.0, dtype=tla.Float32), den, mask=mask)

                tla.local_mem_bar(
                    tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                )

                # Pass 3: normalize by multiplying with one reciprocal.
                # This replaces N divisions by N multiplies and one division.
                for c in tla.range_constexpr(n_chunks):
                    xc = tla.tile_view(
                        ub_x, tla.make_shape(1, VL_ELE), tla.make_coord(0, c)
                    )
                    xc.store(tla.mul(xc.load(), inv, mask=mask))

            # Release add after its last VECTOR read. ub_x remains owned until
            # MTE3 has drained it because the normalized row aliases ub_x.
            if tla.const_expr(HAS_ADD == 1):
                mutex_add.unlock(pipe=tla.arch.VECTOR)
            mutex_x.unlock(pipe=tla.arch.VECTOR)

            # MTE3 takes ownership of ub_x and releases it after the row store,
            # which also protects the next iteration's MTE2 fill.
            mutex_x.lock(pipe=tla.arch.MTE3)
            tla.copy(gm_row_y, ub_x)
            mutex_x.unlock(pipe=tla.arch.MTE3)


# ======================================================================================
# HOST side: shape checks, compile and launch.
# ======================================================================================

_COMPILED = {}
_NULL_ADD = {}
_INPUT_WRAPPERS = {}


def _tla_input(tensor, layout_tag):
    """Reuse a TLA view for a stable input tensor without changing its data."""
    from catlass.tla.runtime import from_dlpack

    key = id(tensor)
    signature = (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        str(tensor.device),
        layout_tag,
    )
    cached = _INPUT_WRAPPERS.get(key)
    if cached is not None and cached[0]() is tensor and cached[1] == signature:
        return cached[2]
    wrapped = from_dlpack(tensor, layout_tag=layout_tag)
    # Close over local references so the callback stays valid during shutdown
    # without using a mutable object as a function default.
    wrappers = _INPUT_WRAPPERS
    cache_key = key

    def drop(dead_ref):
        entry = wrappers.get(cache_key)
        if entry is not None and entry[0] is dead_ref:
            wrappers.pop(cache_key, None)

    wrappers[cache_key] = (weakref.ref(tensor, drop), signature, wrapped)
    return wrapped


def _vector_cores(index):
    import torch

    return int(torch.npu.get_device_properties(index).vector_core_num)


def run(x, add=None, scale=1.0):
    """Return softmax(scale * (x + add)) over the last axis (fp32)."""
    import torch

    if x.dtype != torch.float32:
        raise ValueError("x must be float32")
    if x.dim() != 2:
        raise ValueError("x must be 2-D [M, N]")
    m, n = x.shape
    if n % VL_ELE != 0:
        raise ValueError("N must be a multiple of {}, got {}".format(VL_ELE, n))

    device = x.device
    device_index = int(str(device).split(":")[-1]) if ":" in str(device) else 0
    has_add = 0 if add is None else 1
    if has_add:
        if add.shape != x.shape:
            raise ValueError("add must have the same shape as x")
        add = add.to(torch.float32).contiguous()
    else:
        add = _NULL_ADD.get(device_index)
        if add is None:
            add = torch.zeros((1, 1), dtype=torch.float32, device=device)
            _NULL_ADD[device_index] = add

    # The output is a distinct allocation so callers may reuse the input x.
    out = torch.empty((m, n), dtype=torch.float32, device=device)
    if m == 0:
        return out

    vector_cores = _vector_cores(device_index)

    x_t = _tla_input(x.contiguous(), tla.arch.RowMajor)
    add_t = _tla_input(add, tla.arch.RowMajor)
    out_t = _tla_input(out, tla.arch.RowMajor)

    # SCALE and HAS_ADD are compile-time constants in the kernel. Both must be
    # part of the cache key, otherwise a later call can silently reuse a kernel
    # specialized for a different numerical mode.
    key = (m, n, float(scale), has_add)
    artifact = _COMPILED.get(key)
    if artifact is None:
        artifact = tla.compile(
            masked_softmax_kernel,
            x_t,
            add_t,
            out_t,
            n,
            float(scale),
            has_add,
            options="--npu-arch 3510",
        )
        _COMPILED[key] = artifact

    artifact(x_t, add_t, out_t, block_num=vector_cores)
    return out


def main():
    import argparse

    import torch
    import torch_npu

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    torch.manual_seed(args.seed)
    device = "npu:{}".format(args.device)
    x = torch.randn(args.m, args.n, dtype=torch.float32, device=device)
    add = torch.zeros(args.m, args.n, dtype=torch.float32, device=device)
    add[:, args.n * 3 // 4 :] = -1.0e9
    scale = 1.0 / (args.n**0.5)

    y = run(x, add, scale)
    ref = torch.softmax((x + add) * scale, dim=-1)
    torch.npu.synchronize()
    max_error = (y - ref).abs().max().item()
    print("shape={} dtype={} max_abs_err={:.6g}".format(y.shape, y.dtype, max_error))
    return 0 if max_error <= 1.0e-4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
