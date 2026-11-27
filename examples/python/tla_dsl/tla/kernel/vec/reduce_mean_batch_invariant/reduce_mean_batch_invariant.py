# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CATLASS DSL reduce_mean (batch invariant): AIV kernel plus the host entry.

Operator semantics: ``x[batch, m, n]`` with ``axis in {1, 2}``; the batch axis is
never reduced.  Every output element is produced from a single row (``axis == 2``)
or a single batch element (``axis == 1``) with a fixed serial reduction order, so
a row gives bit-identical results whatever batch it is grouped into.  ``x`` /
``out`` are bound as 2-D row-major views and only the outermost extent is dynamic,
so each compiled variant serves every row count:

* ``axis == 2`` -> ``x`` is ``[batch * m, n]``, ``out`` is ``[batch * m, 1]``;
  one task stages ``row_tile`` whole rows in UB, accumulates each row in a 64-lane
  f32 register, writes the ``row_tile`` scalars into a UB result row and flushes
  them with a single contiguous copy.  The rows left over by ``row_tile`` are
  handled by a per-row tail loop.
* ``axis == 1`` -> ``x`` is ``[batch * m, n]``, ``out`` is ``[batch, n]``; one
  batch element and one column chunk per pass, 32 input rows per UB tile,
  and the column accumulator lives in UB because every stage needs its own
  vector region.

f32 chunks are 64 lanes (one full register).  f16 / bf16 use a 128-element
chunk. Widen the even and odd source lanes with ``RegSlot.ZERO`` and
``RegSlot.ONE``. Accumulate each parity in f32, then recombine the results.
Each narrowing cast writes one parity of a 128-slot register and zeros the
other. Add the two narrowed halves before storing them contiguously.

Axis and precision are specialized at compile time. The dtype of the bound input
tensor controls UB allocations and casts for all three supported dtypes.
"""

import argparse

import torch
import torch_npu  # noqa: F401

import catlass.tla as tla


@tla.kernel
def reduce_mean_kernel(
    mem_x: tla.Tensor,
    mem_z: tla.Tensor,
    inv: float,
    tile_or_batch_rows: int,
    full_rows: int,
    axis: tla.Constexpr[int],
    is_lowp: tla.Constexpr[bool],
) -> None:
    dtype = mem_x.ptr.dtype
    align = 256
    if tla.const_expr(is_lowp):
        chunk = 128
    else:
        chunk = 64

    loaded = tla.mutex("x_ub", id=0)
    stored = tla.mutex("z_ub", id=1)

    if tla.const_expr(axis == 2):
        row_tile = tile_or_batch_rows
        row_cap = 16384
        n = mem_x.shape[1]
        rows = mem_x.shape[0]
        ntasks = full_rows // row_tile

        x_ptr = tla.allocate(row_cap, dtype, tla.AddressSpace.ub, align)
        # ``make_tensor_like`` gives the 1-column result tile a 32 B row pitch, so
        # 64 result rows need 2 KiB at the resulting row pitch.
        z_ptr = tla.allocate(chunk * 8, dtype, tla.AddressSpace.ub, align)

        with tla.vector():
            for task in tla.range(tla.arch.block_idx(), ntasks, tla.arch.block_num()):
                base = task * row_tile
                x_gm = tla.get_tile(
                    mem_x, tla.make_coord(base, 0), tla.make_shape(row_tile, n)
                )
                z_gm = tla.get_tile(
                    mem_z, tla.make_coord(base, 0), tla.make_shape(row_tile, 1)
                )
                x_ub = tla.make_tensor_like(x_ptr, x_gm, tla.arch.RowMajor)
                z_ub = tla.make_tensor_like(z_ptr, z_gm, tla.arch.RowMajor)
                loaded.lock(pipe=tla.pipes.MTE2)
                tla.copy(x_ub, x_gm)
                loaded.unlock(pipe=tla.pipes.MTE2)
                loaded.lock(pipe=tla.pipes.VECTOR)
                stored.lock(pipe=tla.pipes.VECTOR)
                for r in tla.range(0, row_tile, 1):
                    with tla.vec.func(mode="simd"):
                        acc = tla.full(0.0, tla.Float32)
                        zero = tla.full(0.0, tla.Float32)
                        for off in tla.range(0, n, chunk):
                            sub = tla.get_tile(
                                x_ub, tla.make_coord(r, off), tla.make_shape(1, chunk)
                            )
                            if tla.const_expr(is_lowp):
                                rest = n - off
                                even, _ = tla.update_mask(rest - rest // 2, tla.Float32)
                                odd, _ = tla.update_mask(rest // 2, tla.Float32)
                                src = sub.load()
                                acc = tla.add(
                                    acc,
                                    tla.where(
                                        even,
                                        src.to(
                                            tla.Float32,
                                            tla.params.CastParams(
                                                reg_slot=tla.params.RegSlot.ZERO
                                            ),
                                        ),
                                        zero,
                                    ),
                                )
                                acc = tla.add(
                                    acc,
                                    tla.where(
                                        odd,
                                        src.to(
                                            tla.Float32,
                                            tla.params.CastParams(
                                                reg_slot=tla.params.RegSlot.ONE
                                            ),
                                        ),
                                        zero,
                                    ),
                                )
                            else:
                                tail, _ = tla.update_mask(n - off, tla.Float32)
                                acc = tla.add(acc, tla.where(tail, sub.load(), zero))
                        all_lanes = tla.create_mask(
                            pattern=tla.mask.ALL, dtype=tla.Float32
                        )
                        total = tla.mul(
                            acc.reduce(tla.ReductionOp.ADD, mask=all_lanes), inv
                        )
                        z_row = tla.get_tile(
                            z_ub, tla.make_coord(r, 0), tla.make_shape(1, 1)
                        )
                        if tla.const_expr(is_lowp):
                            z_row.store(
                                total.to(
                                    dtype,
                                    tla.params.CastParams(
                                        reg_slot=tla.params.RegSlot.ZERO,
                                        sat_mode=tla.params.SatMode.NOSAT,
                                    ),
                                )
                            )
                        else:
                            z_row.store(total)
                stored.unlock(pipe=tla.pipes.VECTOR)
                loaded.unlock(pipe=tla.pipes.VECTOR)
                stored.lock(pipe=tla.pipes.MTE3)
                tla.copy(z_gm, z_ub)
                stored.unlock(pipe=tla.pipes.MTE3)
                tla.pipe_barrier(tla.pipes.ALL)
            for tail_row in tla.range(
                full_rows + tla.arch.block_idx(), rows, tla.arch.block_num()
            ):
                x_tail_gm = tla.get_tile(
                    mem_x, tla.make_coord(tail_row, 0), tla.make_shape(1, n)
                )
                z_tail_gm = tla.get_tile(
                    mem_z, tla.make_coord(tail_row, 0), tla.make_shape(1, 1)
                )
                x_tail_ub = tla.make_tensor_like(x_ptr, x_tail_gm, tla.arch.RowMajor)
                z_tail_ub = tla.make_tensor_like(z_ptr, z_tail_gm, tla.arch.RowMajor)
                loaded.lock(pipe=tla.pipes.MTE2)
                tla.copy(x_tail_ub, x_tail_gm)
                loaded.unlock(pipe=tla.pipes.MTE2)
                loaded.lock(pipe=tla.pipes.VECTOR)
                stored.lock(pipe=tla.pipes.VECTOR)
                with tla.vec.func(mode="simd"):
                    acc2 = tla.full(0.0, tla.Float32)
                    zero2 = tla.full(0.0, tla.Float32)
                    for off2 in tla.range(0, n, chunk):
                        sub2 = tla.get_tile(
                            x_tail_ub, tla.make_coord(0, off2), tla.make_shape(1, chunk)
                        )
                        if tla.const_expr(is_lowp):
                            rest2 = n - off2
                            even2, _ = tla.update_mask(rest2 - rest2 // 2, tla.Float32)
                            odd2, _ = tla.update_mask(rest2 // 2, tla.Float32)
                            src2 = sub2.load()
                            acc2 = tla.add(
                                acc2,
                                tla.where(
                                    even2,
                                    src2.to(
                                        tla.Float32,
                                        tla.params.CastParams(
                                            reg_slot=tla.params.RegSlot.ZERO
                                        ),
                                    ),
                                    zero2,
                                ),
                            )
                            acc2 = tla.add(
                                acc2,
                                tla.where(
                                    odd2,
                                    src2.to(
                                        tla.Float32,
                                        tla.params.CastParams(
                                            reg_slot=tla.params.RegSlot.ONE
                                        ),
                                    ),
                                    zero2,
                                ),
                            )
                        else:
                            tail2, _ = tla.update_mask(n - off2, tla.Float32)
                            acc2 = tla.add(acc2, tla.where(tail2, sub2.load(), zero2))
                    all2 = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                    total2 = tla.mul(acc2.reduce(tla.ReductionOp.ADD, mask=all2), inv)
                    if tla.const_expr(is_lowp):
                        z_tail_ub.store(
                            total2.to(
                                dtype,
                                tla.params.CastParams(
                                    reg_slot=tla.params.RegSlot.ZERO,
                                    sat_mode=tla.params.SatMode.NOSAT,
                                ),
                            )
                        )
                    else:
                        z_tail_ub.store(total2)
                stored.unlock(pipe=tla.pipes.VECTOR)
                loaded.unlock(pipe=tla.pipes.VECTOR)
                stored.lock(pipe=tla.pipes.MTE3)
                tla.copy(z_tail_gm, z_tail_ub)
                stored.unlock(pipe=tla.pipes.MTE3)
                tla.pipe_barrier(tla.pipes.ALL)
            tla.pipe_barrier(tla.pipes.ALL)
    else:
        rows_per_batch = tile_or_batch_rows
        tile_rows = 32
        n = mem_x.shape[1]
        batch = mem_z.shape[0]

        nchunks = (n + chunk - 1) // chunk
        x_ptr = tla.allocate(tile_rows * chunk, dtype, tla.AddressSpace.ub, align)
        acc_ptr = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, align)
        if tla.const_expr(is_lowp):
            acc_o_ptr = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, align)
        z_ptr = tla.allocate(chunk, dtype, tla.AddressSpace.ub, align)

        acc_layout = tla.make_layout(
            tla.make_shape(1, 64),
            tla.make_stride(64, 1),
            layoutTag=tla.arch.RowMajor,
        )
        acc_ub = tla.make_tensor(acc_ptr, acc_layout)
        if tla.const_expr(is_lowp):
            acc_o_ub = tla.make_tensor(acc_o_ptr, acc_layout)
            z_full_ub = tla.make_tensor(
                z_ptr,
                tla.make_layout(
                    tla.make_shape(1, chunk),
                    tla.make_stride(chunk, 1),
                    layoutTag=tla.arch.RowMajor,
                ),
            )
        else:
            # The DSL collects region captures before specializing the body.
            # These aliases are never accessed in the FP32 variant.
            acc_o_ub = acc_ub
            z_full_ub = acc_ub

        with tla.vector():
            for task in tla.range(
                tla.arch.block_idx(), batch * nchunks, tla.arch.block_num()
            ):
                for unit in tla.range_constexpr(0, 1):
                    b = task // nchunks
                    c = (task - b * nchunks) * chunk
                    z_gm = tla.get_tile(
                        mem_z, tla.make_coord(b, c), tla.make_shape(1, chunk)
                    )
                    if tla.const_expr(is_lowp):
                        z_ub = z_full_ub
                    else:
                        z_ub = tla.make_tensor_like(z_ptr, z_gm, tla.arch.RowMajor)
                    with tla.vec.func(mode="simd"):
                        acc_ub.store(tla.full(0.0, tla.Float32))
                        if tla.const_expr(is_lowp):
                            acc_o_ub.store(tla.full(0.0, tla.Float32))
                    for i in tla.range(0, full_rows, tile_rows):
                        x_gm = tla.get_tile(
                            mem_x,
                            tla.make_coord(b * rows_per_batch + i, c),
                            tla.make_shape(tile_rows, chunk),
                        )
                        x_ub = tla.make_tensor_like(x_ptr, x_gm, tla.arch.RowMajor)
                        loaded.lock(pipe=tla.pipes.MTE2)
                        tla.copy(x_ub, x_gm)
                        loaded.unlock(pipe=tla.pipes.MTE2)
                        loaded.lock(pipe=tla.pipes.VECTOR)
                        with tla.vec.func(mode="simd"):
                            if tla.const_expr(is_lowp):
                                rest = n - c
                                even, _ = tla.update_mask(rest - rest // 2, tla.Float32)
                                odd, _ = tla.update_mask(rest // 2, tla.Float32)
                            else:
                                tail, _ = tla.update_mask(n - c, tla.Float32)
                            zero = tla.full(0.0, tla.Float32)
                            acc = acc_ub.load()
                            if tla.const_expr(is_lowp):
                                acc_o = acc_o_ub.load()
                            for r in tla.range_constexpr(0, tile_rows):
                                sub = tla.get_tile(
                                    x_ub, tla.make_coord(r, 0), tla.make_shape(1, chunk)
                                )
                                src = sub.load()
                                if tla.const_expr(is_lowp):
                                    acc = tla.add(
                                        acc,
                                        tla.where(
                                            even,
                                            src.to(
                                                tla.Float32,
                                                tla.params.CastParams(
                                                    reg_slot=tla.params.RegSlot.ZERO
                                                ),
                                            ),
                                            zero,
                                        ),
                                    )
                                    acc_o = tla.add(
                                        acc_o,
                                        tla.where(
                                            odd,
                                            src.to(
                                                tla.Float32,
                                                tla.params.CastParams(
                                                    reg_slot=tla.params.RegSlot.ONE
                                                ),
                                            ),
                                            zero,
                                        ),
                                    )
                                else:
                                    acc = tla.add(acc, tla.where(tail, src, zero))
                            acc_ub.store(acc)
                            if tla.const_expr(is_lowp):
                                acc_o_ub.store(acc_o)
                        loaded.unlock(pipe=tla.pipes.VECTOR)
                        tla.pipe_barrier(tla.pipes.ALL)
                    for tail_row in tla.range(full_rows, rows_per_batch, 1):
                        x_tail_gm = tla.get_tile(
                            mem_x,
                            tla.make_coord(b * rows_per_batch + tail_row, c),
                            tla.make_shape(1, chunk),
                        )
                        x_tail_ub = tla.make_tensor_like(
                            x_ptr, x_tail_gm, tla.arch.RowMajor
                        )
                        loaded.lock(pipe=tla.pipes.MTE2)
                        tla.copy(x_tail_ub, x_tail_gm)
                        loaded.unlock(pipe=tla.pipes.MTE2)
                        loaded.lock(pipe=tla.pipes.VECTOR)
                        with tla.vec.func(mode="simd"):
                            if tla.const_expr(is_lowp):
                                rest2 = n - c
                                even2, _ = tla.update_mask(
                                    rest2 - rest2 // 2, tla.Float32
                                )
                                odd2, _ = tla.update_mask(rest2 // 2, tla.Float32)
                                zero2 = tla.full(0.0, tla.Float32)
                                src2 = x_tail_ub.load()
                                ae2 = tla.add(
                                    acc_ub.load(),
                                    tla.where(
                                        even2,
                                        src2.to(
                                            tla.Float32,
                                            tla.params.CastParams(
                                                reg_slot=tla.params.RegSlot.ZERO
                                            ),
                                        ),
                                        zero2,
                                    ),
                                )
                                ao2 = tla.add(
                                    acc_o_ub.load(),
                                    tla.where(
                                        odd2,
                                        src2.to(
                                            tla.Float32,
                                            tla.params.CastParams(
                                                reg_slot=tla.params.RegSlot.ONE
                                            ),
                                        ),
                                        zero2,
                                    ),
                                )
                                acc_ub.store(ae2)
                                acc_o_ub.store(ao2)
                            else:
                                tail2, _ = tla.update_mask(n - c, tla.Float32)
                                zero2 = tla.full(0.0, tla.Float32)
                                acc2 = tla.add(
                                    acc_ub.load(),
                                    tla.where(tail2, x_tail_ub.load(), zero2),
                                )
                                acc_ub.store(acc2)
                        loaded.unlock(pipe=tla.pipes.VECTOR)
                        tla.pipe_barrier(tla.pipes.ALL)
                    stored.lock(pipe=tla.pipes.VECTOR)
                    with tla.vec.func(mode="simd"):
                        if tla.const_expr(is_lowp):
                            ze = tla.mul(acc_ub.load(), inv).to(
                                dtype,
                                tla.params.CastParams(
                                    reg_slot=tla.params.RegSlot.ZERO,
                                    sat_mode=tla.params.SatMode.NOSAT,
                                ),
                            )
                            zo = tla.mul(acc_o_ub.load(), inv).to(
                                dtype,
                                tla.params.CastParams(
                                    reg_slot=tla.params.RegSlot.ONE,
                                    sat_mode=tla.params.SatMode.NOSAT,
                                ),
                            )
                            z_ub.store(tla.add(ze, zo))
                        else:
                            z_ub.store(tla.mul(acc_ub.load(), inv))
                    stored.unlock(pipe=tla.pipes.VECTOR)
                    stored.lock(pipe=tla.pipes.MTE3)
                    tla.copy(z_gm, z_ub)
                    stored.unlock(pipe=tla.pipes.MTE3)
                    tla.pipe_barrier(tla.pipes.ALL)
            tla.pipe_barrier(tla.pipes.ALL)


# ======================================================================================
# HOST side: shape planning, per-shape compile cache and launch.
# ======================================================================================

_BLOCK_CAP = 48
_TILE_ROWS = 32
_TILE_MAX = 64
_ROW_CAP = 16384
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_COMPILED = {}
_PLANS = {}
_BIND = None
_BOUND = {}
_LAUNCH = {}
_FAST = None


def _binder():
    """Return a function binding a torch NPU tensor as a launchable runtime view.

    ``tla.from_dlpack`` builds a DLPack capsule on every call (~0.05 ms); the
    runtime tensor can also be constructed straight from the torch data pointer
    (~0.03 ms).  The fallback keeps the public path when the internal
    constructor is unavailable.  Every path stores only the address, so the
    caller must keep the torch tensor alive until the launch finishes.
    """
    global _BIND
    if _BIND is not None:
        return _BIND

    try:
        from catlass.tla.runtime import _Tensor
    except ImportError:
        _Tensor = None

    if _Tensor is None:

        def bind(tensor, shape):
            view = tla.from_dlpack(tensor, layout_tag=tla.arch.RowMajor)
            view.mark_compact_shape_dynamic(mode=0)
            return view

    else:
        tla_dtypes = {
            torch.float32: tla.Float32,
            torch.float16: tla.Float16,
            torch.bfloat16: tla.BFloat16,
        }

        def bind(tensor, shape):
            view = _Tensor(
                shape,
                tla_dtypes[tensor.dtype],
                data_ptr=int(tensor.data_ptr()),
                origin_shape=shape,
                layout_tag=tla.arch.RowMajor,
            )
            view.mark_compact_shape_dynamic(mode=0)
            return view

    _BIND = bind
    return bind


def _bound(tensor, shape, bind):
    """Reuse the runtime view for an address that has already been bound."""
    key = (int(tensor.data_ptr()), shape, tensor.dtype)
    view = _BOUND.get(key)
    if view is None:
        if len(_BOUND) > 256:
            _BOUND.clear()
        view = bind(tensor, shape)
        _BOUND[key] = view
    return view


def _validate(x, axis):
    if x.ndim != 3:
        raise ValueError("x must have shape [batch, m, n]")
    axis = int(axis)
    if axis not in (1, 2):
        raise ValueError("axis must be 1 or 2; the batch axis is never reduced")
    if any(int(dim) <= 0 for dim in x.shape):
        raise ValueError("all dimensions of x must be non-empty")
    if x.dtype not in _DTYPES:
        raise ValueError("unsupported dtype: {}".format(x.dtype))
    return axis


def _build_plan(x, axis):
    """Resolve the kernel variant, GM view shapes, scalars and grid once per shape."""
    axis = _validate(x, axis)
    batch, m, n = (int(dim) for dim in x.shape)
    rows = batch * m

    if axis == 2:
        if n > _ROW_CAP:
            raise ValueError(
                "n={} exceeds the staged row capacity {}".format(n, _ROW_CAP)
            )
        out_shape = (batch, m)
        x_view = (rows, n)
        z_view = (rows, 1)
        # The staged tile gets a UB row stride rounded up to 32 B, so a tile of
        # ``row_tile`` rows needs ``row_tile * align_up(n)`` elements; sizing the
        # tile by the compact width overflows the UB buffer for unaligned ``n``.
        align = 32 // (4 if x.dtype == torch.float32 else 2)
        padded_n = (n + align - 1) // align * align
        row_tile = max(1, min(_TILE_MAX, _ROW_CAP // padded_n))
        full_rows = rows - rows % row_tile
        scalars = (1.0 / n, row_tile, full_rows)
        blocks = min(_BLOCK_CAP, max(1, full_rows // row_tile))
    else:
        out_shape = (batch, n)
        x_view = (rows, n)
        z_view = (batch, n)
        scalars = (1.0 / m, m, m - m % _TILE_ROWS)
        chunk = 64 if x.dtype == torch.float32 else 128
        blocks = min(_BLOCK_CAP, batch * ((n + chunk - 1) // chunk))
    return axis, x_view, z_view, out_shape, scalars, blocks


def _plan(x, axis):
    key = (tuple(x.shape), x.dtype, int(axis))
    plan = _PLANS.get(key)
    if plan is None:
        plan = _build_plan(x, axis)
        _PLANS[key] = plan
    return plan


def _compiled(axis, x_buffer, z_buffer, x_view, z_view, scalars):
    key = (axis, x_view[1], x_buffer.dtype)
    compiled = _COMPILED.get(key)
    if compiled is None:
        bind = _binder()
        compiled = tla.compile(
            reduce_mean_kernel,
            bind(x_buffer, x_view),
            bind(z_buffer, z_view),
            *scalars,
            axis,
            x_buffer.dtype != torch.float32,
            options="--npu-arch 3510",
        )
        _COMPILED[key] = compiled
    return compiled


def compile_only(x, axis):
    """Compile the selected kernel variant without launching it."""
    axis, x_view, z_view, _, scalars, _ = _plan(x, axis)
    x_buffer = torch.empty(x_view, dtype=x.dtype, device=x.device)
    z_buffer = torch.empty(z_view, dtype=x.dtype, device=x.device)
    return _compiled(axis, x_buffer, z_buffer, x_view, z_view, scalars)


def _fast_primitives():
    """Private launch primitives, or ``None`` when the fast path is unavailable.

    ``JitExecutor.__call__`` rebuilds the ABI payload, re-resolves the dynamic UB
    window, re-reads the device and allocates a result object on every launch,
    and ``execution.launch_kernel`` then rebuilds the launch config and copies
    the payload into a fresh host buffer.  Only the stream handle can change
    between two launches of the same tensor addresses, so everything else is
    built once per argument tuple and the ACL call is issued directly.
    """
    global _FAST
    if _FAST is None:
        try:
            import ctypes

            import acl
            from catlass.base_dsl.jit_executor import _resolved_dyn_ubuf_bytes
            from catlass.base_dsl.runtime.ascend import _launch_config, check_acl_errors

            _FAST = (
                acl,
                ctypes,
                check_acl_errors,
                _launch_config,
                _resolved_dyn_ubuf_bytes,
            )
        except Exception:  # noqa: BLE001
            _FAST = False
    return _FAST or None


def _prepare_launch(primitives, compiled, block_num, args):
    """One-off launch state for ``args``: packed payload, host buffer, config."""
    acl, ctypes, check_errors, launch_config, resolved_dyn_ubuf = primitives
    executor = compiled._get_executor()
    module = executor.jit_module
    print_config = module.print_config
    if (
        print_config.print_metadata is not None
        or print_config.uses_scalar_print
        or print_config.uses_tensor_print
    ):
        return None
    payload = module.execution_args.generate_launch_payload(args)
    host_buffer = (ctypes.c_uint8 * len(payload)).from_buffer_copy(payload)
    return (
        acl.rt.launch_kernel_with_host_args,
        check_errors,
        int(executor.function_handle),
        int(block_num),
        launch_config(resolved_dyn_ubuf(module, 0)),
        ctypes.addressof(host_buffer),
        len(payload),
        host_buffer,
        int(executor.device),
    )


def _launch(compiled, block_num, args):
    """Launch ``compiled`` on the current stream with a prebuilt host payload."""
    primitives = _fast_primitives()
    if primitives is None:
        compiled(*args, block_num=block_num)
        return
    key = (compiled, args)
    entry = _LAUNCH.get(key)
    if entry is None:
        if len(_LAUNCH) > 64:
            _LAUNCH.clear()
        entry = _prepare_launch(primitives, compiled, block_num, args)
        if entry is None:
            compiled(*args, block_num=block_num)
            return
        _LAUNCH[key] = entry
    launch, check_errors, handle, blocks, config, address, size, keepalive, device = (
        entry
    )
    stream = int(torch.npu.current_stream(device).npu_stream)
    check_errors(
        launch(handle, blocks, stream, config, address, size, [], 0),
        "acl.rt.launch_kernel_with_host_args",
    )


def run(x, axis):
    """Return the batch-invariant mean of ``x`` over ``axis`` with batch preserved."""
    if not x.is_contiguous():
        x = x.contiguous()
    axis, x_view, z_view, out_shape, scalars, blocks = _plan(x, axis)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    bind = _binder()
    compiled = _compiled(axis, x, out, x_view, z_view, scalars)
    _launch(
        compiled, blocks, (_bound(x, x_view, bind), _bound(out, z_view, bind)) + scalars
    )
    return out


def main() -> int:
    """Compile and run one shape from the command line, checked against torch."""
    parser = argparse.ArgumentParser(
        description="Compile and run a batch-invariant reduce_mean."
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--dtype", choices=("f32", "f16", "bf16"), default="f16")
    parser.add_argument("--axis", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    dtype = {
        "f32": torch.float32,
        "f16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.dtype]
    x = torch.randn(args.batch, args.m, args.n, dtype=dtype, device="npu")

    out = run(x, args.axis)
    torch.npu.synchronize()

    # The kernel accumulates in f32 but in a different order than torch, so the
    # comparison uses the tolerance the README documents.
    ref = x.float().mean(dim=args.axis)
    passed = torch.allclose(out.float(), ref, rtol=1e-2, atol=1e-2)
    print(f"passed={passed} axis={args.axis} dtype={args.dtype} shape={tuple(x.shape)}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
