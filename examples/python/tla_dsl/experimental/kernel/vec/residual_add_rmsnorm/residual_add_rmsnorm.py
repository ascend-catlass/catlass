# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Residual add + row-wise RMSNorm implemented with the CATLASS Python DSL.

    t[r, :] = x[r, :] + residual[r, :]
    y[r, :] = t[r, :] / sqrt(mean(t[r, :]^2) + eps) * weight

The file intentionally keeps the Device and Host implementations together so
the entry point and the launch contract can be reviewed as one unit.

Device-side high-performance design:

1. Continuous multi-row tiles: each AIV block owns a contiguous span of
   ``TILE_ROWS`` row tiles, so one tile is one contiguous GM-to-UB transfer
   and one contiguous UB-to-GM transfer.
2. Three-stage pipeline: ``BUFS`` tile buffers use one mutex per buffer. The
   same mutex moves ownership from MTE2 to VECTOR and then to MTE3, forming a
   load -> compute -> store hand-off while also providing refill back-pressure.
3. Row amortization: the optional weight is loaded once per block and stays
   resident in UB for every processing row. This removes one N-sized weight
   transfer per row.
4. Reduction amortization: each row accumulates squares in 64 FP32 lanes and
   performs only one reduction after the dynamic chunk loop.
5. Tail specialization: the UB row uses ``N_PAD = ceil(N / 64) * 64`` while GM
   stays dense. A single ``update_mask`` tail path is folded away for aligned
   ``N`` at compile time.

Host-side high-performance design:

1. Compilation is cached by ``(M, N, eps, affine, tile_rows, bufs)``.
2. TLA bindings are cached by buffer address and logical shape, avoiding a
   repeated DLPack conversion in repeated benchmark calls.
3. Device properties, AIV block count, stream and prepared launch payloads are
   cached by device index, so steady-state calls avoid rebuilding the ABI tree.
4. Every call returns fresh output storage; the PyTorch allocator can reuse
   addresses safely without making run results alias each other.
"""

import functools
import inspect
import math
import threading

import torch
import torch_npu  # noqa: F401

import catlass.tla as tla
from catlass.tla.runtime import _Tensor

VL_ELE = 64
UB_BYTES = 262144


def _chunk(tensor, row, col):
    return tla.tile_view(tensor, tla.make_shape(1, VL_ELE), tla.make_coord(row, col))


def _row_rstd(sq_sum, INV_N, EPS, mask, tail_acc=None, tail_mask=None):
    """Reduce lane-wise squares once and broadcast the row rstd."""
    total = tla.full(sq_sum.reduce(tla.ReductionOp.ADD, mask=mask), dtype=tla.Float32)
    if tail_acc is not None:
        total = tla.add(
            total,
            tla.full(
                tail_acc.reduce(tla.ReductionOp.ADD, mask=tail_mask),
                dtype=tla.Float32,
            ),
            mask=mask,
        )
    mean_square = tla.mul(total, INV_N, mask=mask)
    return tla.div(
        tla.full(1.0, dtype=tla.Float32),
        tla.sqrt(tla.add(mean_square, EPS, mask=mask), mask=mask),
        mask=mask,
    )


@tla.kernel
def residual_add_rmsnorm_kernel(
    gm_x: tla.Tensor,
    gm_residual: tla.Tensor,
    gm_weight: tla.Tensor,
    gm_output: tla.Tensor,
    N: tla.Constexpr[int],
    N_PAD: tla.Constexpr[int],
    TAIL: tla.Constexpr[int],
    INV_N: tla.Constexpr[float],
    EPS: tla.Constexpr[float],
    HAS_AFFINE: tla.Constexpr[int],
    TILE_ROWS: tla.Constexpr[int],
    BUFS: tla.Constexpr[int],
) -> None:
    """Fused FP32 residual add and row RMSNorm.

    Complexity: O(M * N) arithmetic and O(M * N) GM traffic. Each row is
    reduced inside one AIV block, so no cross-block reduction is required.
    """
    m = gm_x.origin_shape[0]
    gid = tla.arch.block_idx()
    aiv_num = tla.arch.block_num()
    n_chunks = N // VL_ELE

    # One mutex per tile buffer is the ownership token for the complete
    # MTE2 -> VECTOR -> MTE3 cycle:
    #   MTE2:   lock -> load x/residual -> unlock
    #   VECTOR: lock -> add/reduce/normalize -> unlock
    #   MTE3:   lock -> write output -> unlock
    # The next MTE2 refill waits for MTE3 to return the token, so in-flight
    # buffers progress independently while stale data cannot be overwritten.
    tile_mutex = [None] * BUFS
    x_ptr = [None] * BUFS
    residual_ptr = [None] * BUFS
    for bi in tla.range_constexpr(BUFS):
        x_ptr[bi] = tla.allocate(
            TILE_ROWS * N_PAD, tla.Float32, tla.AddressSpace.ub, 256
        )
        residual_ptr[bi] = tla.allocate(
            TILE_ROWS * N_PAD, tla.Float32, tla.AddressSpace.ub, 256
        )
        tile_mutex[bi] = tla.mutex(resource="rrn_tile", id=bi)

    # The optional affine weight is resident for the whole block. Its mutex
    # hands the first MTE2 fill to VECTOR and lets every owned row reuse the
    # same UB copy instead of reloading N elements per row.
    weight_mutex = tla.mutex(resource="rrn_weight", id=31)
    weight_ptr = tla.allocate(N_PAD, tla.Float32, tla.AddressSpace.ub, 256)

    row_layout = tla.make_layout(
        tla.make_shape(1, N),
        tla.make_stride(N_PAD, 1),
        origin_shape=tla.make_shape(1, N),
    )
    tile_layout = tla.make_layout(
        tla.make_shape(TILE_ROWS, N),
        tla.make_stride(N_PAD, 1),
        origin_shape=tla.make_shape(TILE_ROWS, N),
    )
    gm_tile_layout = tla.make_layout(
        tla.make_shape(TILE_ROWS, N),
        tla.make_stride(N, 1),
        origin_shape=tla.make_shape(TILE_ROWS, N),
    )
    ub_x = [None] * BUFS
    ub_residual = [None] * BUFS
    for bj in tla.range_constexpr(BUFS):
        ub_x[bj] = tla.make_tensor(x_ptr[bj], tile_layout)
        ub_residual[bj] = tla.make_tensor(residual_ptr[bj], tile_layout)
    ub_weight = tla.make_tensor(weight_ptr, row_layout)

    # TILE_ROWS is selected only when it divides M, so every tile has a full
    # origin shape and no out-of-range GM access is possible.
    n_tiles = m // TILE_ROWS
    tiles_per_block = (n_tiles + aiv_num - 1) // aiv_num
    tile_begin = gid * tiles_per_block

    with tla.vector():
        if tile_begin < n_tiles:
            if tla.const_expr(HAS_AFFINE == 1):
                weight_mutex.lock(pipe=tla.pipes.MTE2)
                tla.copy(ub_weight, gm_weight)
                weight_mutex.unlock(pipe=tla.pipes.MTE2)

            # Issue one group of independent buffers at a time. Mutex waits are
            # pipe-local, so MTE2 can fill the next buffer while VECTOR processes
            # the current one and MTE3 drains a completed one.
            for grp in tla.range((tiles_per_block + BUFS - 1) // BUFS):
                for bi in tla.range_constexpr(BUFS):
                    tile = tile_begin + grp * BUFS + bi
                    if tile < n_tiles:
                        tile_offset = tile.to(tla.Int64) * tla.Int64(TILE_ROWS * N)
                        gm_x_tile = tla.make_tensor(
                            gm_x.ptr + tile_offset, gm_tile_layout
                        )
                        gm_residual_tile = tla.make_tensor(
                            gm_residual.ptr + tile_offset, gm_tile_layout
                        )
                        gm_output_tile = tla.make_tensor(
                            gm_output.ptr + tile_offset, gm_tile_layout
                        )

                        tile_mutex[bi].lock(pipe=tla.pipes.MTE2)
                        tla.copy(ub_x[bi], gm_x_tile)
                        tla.copy(ub_residual[bi], gm_residual_tile)
                        tile_mutex[bi].unlock(pipe=tla.pipes.MTE2)

                        tile_mutex[bi].lock(pipe=tla.pipes.VECTOR)
                        if tla.const_expr(HAS_AFFINE == 1):
                            weight_mutex.lock(pipe=tla.pipes.VECTOR)

                        with tla.vec.func(mode="simd"):
                            mask = tla.create_mask(
                                pattern=tla.mask.ALL, dtype=tla.Float32
                            )
                            tail_mask = None
                            if tla.const_expr(TAIL != 0):
                                tail_mask, _ = tla.update_mask(TAIL, dtype=tla.Float32)

                            # Rows in a tile share the resident weight. Each row
                            # accumulates squares in 64 FP32 lanes, then performs
                            # one scalar reduction after all chunks are consumed.
                            for row_in_tile in tla.range_constexpr(TILE_ROWS):
                                sq_sum = tla.full(0.0, dtype=tla.Float32)
                                for c1 in tla.range(n_chunks):
                                    x_head_chunk = _chunk(ub_x[bi], row_in_tile, c1)
                                    residual_head_chunk = _chunk(
                                        ub_residual[bi], row_in_tile, c1
                                    )
                                    residual_sum = tla.add(
                                        x_head_chunk.load(),
                                        residual_head_chunk.load(),
                                        mask=mask,
                                    )
                                    x_head_chunk.store(residual_sum)
                                    sq_sum = tla.add(
                                        sq_sum,
                                        tla.mul(residual_sum, residual_sum, mask=mask),
                                        mask=mask,
                                    )

                                # GM remains dense. Only the UB row is padded to
                                # N_PAD, and the masked tail accumulator keeps
                                # padded lanes out of both statistics and output.
                                tail_sum = tla.full(0.0, dtype=tla.Float32)
                                if tla.const_expr(TAIL != 0):
                                    x_tail_head = _chunk(
                                        ub_x[bi], row_in_tile, n_chunks
                                    )
                                    residual_tail = _chunk(
                                        ub_residual[bi], row_in_tile, n_chunks
                                    )
                                    residual_tail_sum = tla.add(
                                        x_tail_head.load(),
                                        residual_tail.load(),
                                        mask=tail_mask,
                                    )
                                    x_tail_head.store(residual_tail_sum)
                                    tail_sum = tla.add(
                                        tail_sum,
                                        tla.mul(
                                            residual_tail_sum,
                                            residual_tail_sum,
                                            mask=tail_mask,
                                        ),
                                        mask=tail_mask,
                                    )
                                    rstd = _row_rstd(
                                        sq_sum,
                                        INV_N,
                                        EPS,
                                        mask,
                                        tail_sum,
                                        tail_mask,
                                    )
                                else:
                                    rstd = _row_rstd(sq_sum, INV_N, EPS, mask)

                                # The first pass stored t into the same UB tile;
                                # order the VEC store before the second-pass load.
                                tla.local_mem_bar(
                                    tla.params.MemType.VEC_STORE,
                                    tla.params.MemType.VEC_LOAD,
                                )

                                for c2 in tla.range(n_chunks):
                                    x_scale_chunk = _chunk(ub_x[bi], row_in_tile, c2)
                                    scaled_chunk = tla.mul(
                                        x_scale_chunk.load(), rstd, mask=mask
                                    )
                                    if tla.const_expr(HAS_AFFINE == 1):
                                        scaled_chunk = tla.mul(
                                            scaled_chunk,
                                            _chunk(ub_weight, 0, c2).load(),
                                            mask=mask,
                                        )
                                    x_scale_chunk.store(scaled_chunk)
                                if tla.const_expr(TAIL != 0):
                                    x_tail_scale = _chunk(
                                        ub_x[bi], row_in_tile, n_chunks
                                    )
                                    scaled_tail = tla.mul(
                                        x_tail_scale.load(), rstd, mask=tail_mask
                                    )
                                    if tla.const_expr(HAS_AFFINE == 1):
                                        scaled_tail = tla.mul(
                                            scaled_tail,
                                            _chunk(ub_weight, 0, n_chunks).load(),
                                            mask=tail_mask,
                                        )
                                    x_tail_scale.store(scaled_tail)

                        if tla.const_expr(HAS_AFFINE == 1):
                            weight_mutex.unlock(pipe=tla.pipes.VECTOR)
                        tile_mutex[bi].unlock(pipe=tla.pipes.VECTOR)

                        tile_mutex[bi].lock(pipe=tla.pipes.MTE3)
                        tla.copy(gm_output_tile, ub_x[bi])
                        tile_mutex[bi].unlock(pipe=tla.pipes.MTE3)

        tla.pipe_barrier(tla.pipes.ALL)


# ======================================================================================
# HOST side: validation, UB-based tiling and cached compile/launch.
# ======================================================================================

_COMPILED = {}
_BINDINGS = {}
_FAST_PLANS = {}
_LAUNCH_PLANS = {}
_VECTOR_CORES = {}
_DUMMY_WEIGHT = {}
_STREAMS = {}
_MAX_FAST_PLANS = 512
_MAX_LAUNCH_PLANS = 512
_LAUNCH_ACCEPTS_PRINT = None


def _vector_cores(device_index):
    cores = _VECTOR_CORES.get(device_index)
    if cores is None:
        cores = int(torch.npu.get_device_properties(device_index).vector_core_num)
        _VECTOR_CORES[device_index] = cores
    return cores


def _current_stream(device_index):
    key = (threading.get_ident(), device_index)
    stream = _STREAMS.get(key)
    if stream is None:
        stream = int(torch.npu.current_stream(device_index).npu_stream)
        _STREAMS[key] = stream
    return stream


def _acquire_output(x, m, n):
    """Allocate a fresh result so derived views cannot be overwritten implicitly."""
    return torch.empty((m, n), dtype=torch.float32, device=x.device)


def _bind(tensor, shape):
    rows, cols = shape
    key = (tensor.data_ptr(), rows, cols)
    bound = _BINDINGS.get(key)
    if bound is None:
        # DLPack conversion costs tens of microseconds per tensor. All public
        # inputs are validated as contiguous FP32, so the row-major descriptor
        # can be built directly. The caller keeps the backing tensor alive.
        bound = _Tensor(
            (rows, cols),
            tla.Float32,
            origin_shape=(rows, cols),
            coord=(0, 0),
            stride=(cols, 1),
            layout_tag=tla.arch.RowMajor,
            data_ptr=tensor.data_ptr(),
        )
        _BINDINGS[key] = bound
    return bound


def _resolve_launcher(artifact):
    """Resolve the reusable advanced launcher used by CATLASS 2.2.1."""
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
    """Pre-pack a launch payload; fall back to the public artifact call on mismatch."""
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


def _pipeline(m, n_pad):
    """Maximize rows per weight load, then fill remaining UB with tile buffers."""
    weight_bytes = 4 * n_pad
    reserved_bytes = 8192
    tile_rows = 1
    for candidate in (4, 2):
        tile_bytes = 2 * 4 * candidate * n_pad
        if m % candidate == 0 and 2 * tile_bytes + weight_bytes <= UB_BYTES:
            tile_rows = candidate
            break
    tile_bytes = 2 * 4 * tile_rows * n_pad
    available = UB_BYTES - weight_bytes - reserved_bytes
    bufs = max(1, min(4, available // tile_bytes))
    return tile_rows, bufs


def reset_stream_cache():
    """Drop address- and stream-keyed launch plans after switching streams."""
    _BINDINGS.clear()
    _FAST_PLANS.clear()
    _LAUNCH_PLANS.clear()
    _STREAMS.clear()


def _prepare(x, residual, weight, eps):
    """Validate inputs, select tiling and return a compiled launch plan."""
    if not isinstance(x, torch.Tensor) or not isinstance(residual, torch.Tensor):
        raise ValueError("x and residual must be torch tensors")
    if x.dtype != torch.float32:
        raise ValueError("x must be float32")
    if residual.dtype != torch.float32:
        raise ValueError("residual must be float32")
    if x.dim() != 2:
        raise ValueError("x must be 2-D [M, N]")
    if residual.shape != x.shape:
        raise ValueError("residual must have the same shape as x")
    m, n = x.shape
    if n < VL_ELE:
        raise ValueError("N must be >= {}, got {}".format(VL_ELE, n))
    if (
        not isinstance(eps, (int, float))
        or not math.isfinite(float(eps))
        or float(eps) < 0.0
    ):
        raise ValueError("eps must be a finite non-negative scalar")

    has_affine = 0 if weight is None else 1
    if has_affine:
        if not isinstance(weight, torch.Tensor) or weight.shape != (n,):
            raise ValueError("weight must have shape [N]")
        if weight.dtype != torch.float32:
            raise ValueError("weight must be float32")

    device = x.device
    if device.type != "npu":
        raise ValueError("x must be an npu tensor, got {}".format(device))
    if residual.device != device:
        raise ValueError("residual must be on the same device as x")
    if has_affine and weight.device != device:
        raise ValueError("weight must be on the same device as x")
    device_index = (
        device.index if device.index is not None else torch.npu.current_device()
    )

    if not x.is_contiguous():
        x = x.contiguous()
    if not residual.is_contiguous():
        residual = residual.contiguous()
    if has_affine and not weight.is_contiguous():
        weight = weight.contiguous()

    stream = _current_stream(device_index)
    output = _acquire_output(x, m, n)
    if m == 0:
        return {
            "artifact": None,
            "output": output,
            "inputs": (x, residual, None, output),
            "block_num": 0,
        }

    if not has_affine:
        dummy = _DUMMY_WEIGHT.get((device_index, n))
        if dummy is None:
            dummy = torch.zeros((n,), dtype=torch.float32, device=device)
            _DUMMY_WEIGHT[(device_index, n)] = dummy
        weight_tensor = dummy
    else:
        weight_tensor = weight

    launch_key = (
        x.data_ptr(),
        residual.data_ptr(),
        weight_tensor.data_ptr(),
        output.data_ptr(),
        m,
        n,
        float(eps),
        has_affine,
        device_index,
        stream,
    )
    launch_plan = _LAUNCH_PLANS.get(launch_key)
    if launch_plan is not None:
        return {
            "artifact": launch_plan["artifact"],
            "output": output,
            "inputs": launch_plan["inputs"],
            "block_num": launch_plan["block_num"],
            "launcher": launch_plan["launcher"],
        }

    fast_key = (
        x.data_ptr(),
        residual.data_ptr(),
        weight_tensor.data_ptr(),
        m,
        n,
        float(eps),
        has_affine,
        device_index,
    )
    fast_plan = _FAST_PLANS.get(fast_key)
    if fast_plan is not None:
        output_t = _bind(output, (m, n))
        plan = {
            "artifact": fast_plan["artifact"],
            "inputs": fast_plan["inputs"] + (output_t,),
            "block_num": fast_plan["block_num"],
        }
        plan["launcher"] = _make_launcher(
            plan["artifact"], plan["inputs"], plan["block_num"], stream
        )
        if len(_LAUNCH_PLANS) >= _MAX_LAUNCH_PLANS:
            _LAUNCH_PLANS.clear()
        _LAUNCH_PLANS[launch_key] = plan
        return {
            "artifact": plan["artifact"],
            "output": output,
            "inputs": plan["inputs"],
            "block_num": plan["block_num"],
            "launcher": plan["launcher"],
        }

    n_pad = (n + VL_ELE - 1) // VL_ELE * VL_ELE
    tail = n - (n // VL_ELE) * VL_ELE
    tile_rows, bufs = _pipeline(m, n_pad)
    vector_cores = _vector_cores(device_index)
    block_num = max(1, min(vector_cores, m // tile_rows))

    compile_key = (
        m,
        n,
        float(eps),
        has_affine,
        tile_rows,
        bufs,
        device_index,
    )
    artifact = _COMPILED.get(compile_key)
    if artifact is None:
        x_t = _bind(x, (m, n))
        residual_t = _bind(residual, (m, n))
        weight_t = _bind(weight_tensor, (1, n))
        output_t = _bind(output, (m, n))
        artifact = tla.compile(
            residual_add_rmsnorm_kernel,
            x_t,
            residual_t,
            weight_t,
            output_t,
            n,
            n_pad,
            tail,
            1.0 / float(n),
            float(eps),
            has_affine,
            tile_rows,
            bufs,
            options="--npu-arch 3510",
        )
        _COMPILED[compile_key] = artifact

    x_t = _bind(x, (m, n))
    residual_t = _bind(residual, (m, n))
    weight_t = _bind(weight_tensor, (1, n))
    output_t = _bind(output, (m, n))
    plan = {
        "artifact": artifact,
        "inputs": (x_t, residual_t, weight_t),
        "block_num": block_num,
    }
    if len(_FAST_PLANS) >= _MAX_FAST_PLANS:
        _FAST_PLANS.clear()
    _FAST_PLANS[fast_key] = plan
    if len(_LAUNCH_PLANS) >= _MAX_LAUNCH_PLANS:
        _LAUNCH_PLANS.clear()
    _LAUNCH_PLANS[launch_key] = {
        "artifact": artifact,
        "inputs": (x_t, residual_t, weight_t, output_t),
        "block_num": block_num,
        "launcher": _make_launcher(
            artifact,
            (x_t, residual_t, weight_t, output_t),
            block_num,
            stream,
        ),
    }
    return {
        "artifact": artifact,
        "output": output,
        "inputs": (x_t, residual_t, weight_t, output_t),
        "block_num": block_num,
        "launcher": _LAUNCH_PLANS[launch_key]["launcher"],
    }


def run(x, residual, weight=None, eps=1e-6):
    """Return ``RMSNorm(x + residual) * weight`` as FP32 ``[M, N]``."""
    plan = _prepare(x, residual, weight, eps)
    artifact = plan["artifact"]
    if artifact is not None:
        launcher = plan.get("launcher")
        if launcher is not None:
            launcher()
        else:
            artifact(*plan["inputs"], block_num=plan["block_num"])
    output = plan["output"]
    return output


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-affine", action="store_true")
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    torch.manual_seed(args.seed)
    device = "npu:{}".format(args.device)
    x = torch.randn(args.m, args.n, dtype=torch.float32, device=device)
    residual = torch.randn_like(x)
    weight = (
        None
        if args.no_affine
        else torch.randn(args.n, dtype=torch.float32, device=device)
    )

    output = run(x, residual, weight)
    t = x + residual
    reference = t / torch.sqrt(t.square().mean(dim=-1, keepdim=True) + 1e-6)
    if weight is not None:
        reference = reference * weight
    torch.npu.synchronize()
    max_error = (output - reference).abs().max().item()
    print(
        "shape={} dtype={} max_abs_err={:.6g}".format(
            output.shape, output.dtype, max_error
        )
    )
    return 0 if max_error <= 1.0e-4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
