# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import catlass.tla as tla
from catlass.params import (
    LoadDist,
    NormalLoadParams,
    UnalignLoadParams,
    NormalStoreParams,
    UnalignStoreParams,
)

from vector_op_harness import (
    DirectVectorOpConfig,
    DirectVectorOpHarness,
    shape_label,
    vector_kernel_config,
)

VECTOR_ELE = 400
VL_ELE = 64
LOOPS = (VECTOR_ELE + VL_ELE - 1) // VL_ELE
ALL_DTYPES = ("i8", "i16", "i32", "f16", "f32", "bf16")

_KERNEL_DTYPE = tla.Float32
_KERNEL_ELEMENT_BYTES = 4
_KERNEL_SHAPE = (VECTOR_ELE,)
_BINARY_OP: Callable[[Any, Any], Any] | None = None
_X_LOAD_PARAMS: NormalLoadParams | UnalignLoadParams = NormalLoadParams()
_X_STORE_PARAMS: NormalStoreParams | UnalignStoreParams = NormalStoreParams()
_X_TILE_ELE = 0
_STORE_LOADED_X_ONLY = False
# Safe import-time placeholders. _configure_batch() replaces them before
# compilation, and the harness launches only the number of requested blocks.
_BATCH_OPS = ("add", "add", "add", "add")


@tla.kernel
def binary_op(mem_x: tla.Tensor, mem_y: tla.Tensor, mem_z: tla.Tensor) -> None:
    ub_loaded = tla.flag("ub_loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    vec_done = tla.flag("vec_done", tla.pipes.VECTOR, tla.pipes.MTE3)

    x_gm = tla.tile_view(mem_x, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    y_gm = tla.tile_view(mem_y, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    z_gm = tla.tile_view(mem_z, tla.make_shape(VECTOR_ELE), tla.make_coord(0))

    x_ub_ptr = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    y_ub_ptr = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    z_ub_ptr = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)

    x_ub = tla.make_tensor_like(x_ub_ptr, x_gm, tla.arch.RowMajor)
    y_ub = tla.make_tensor_like(y_ub_ptr, y_gm, tla.arch.RowMajor)
    z_ub = tla.make_tensor_like(z_ub_ptr, z_gm, tla.arch.RowMajor)

    with tla.vector():
        tla.copy(x_ub, x_gm)
        tla.copy(y_ub, y_gm)

        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)
        with tla.vec.func(mode="simd"):
            x_extent = _X_TILE_ELE if _X_TILE_ELE > 0 else VL_ELE
            for i in tla.range(LOOPS):
                x_tile = tla.tile_view(
                    x_ub, tla.make_shape(x_extent), tla.make_coord(i)
                )
                y_tile = tla.tile_view(y_ub, tla.make_shape(VL_ELE), tla.make_coord(i))
                z_tile = tla.tile_view(z_ub, tla.make_shape(VL_ELE), tla.make_coord(i))

                if _STORE_LOADED_X_ONLY:
                    z_tile.store(x_tile.load(_X_LOAD_PARAMS), _X_STORE_PARAMS)
                else:
                    z_tile.store(
                        _BINARY_OP(x_tile.load(_X_LOAD_PARAMS), y_tile.load()),
                        _X_STORE_PARAMS,
                    )

        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)

        tla.copy(z_gm, z_ub)

        tla.pipe_barrier(tla.pipes.ALL)


def _batch_binary_chunk(
    x_ub: Any, y_ub: Any, z_ub: Any, chunk_idx: Any, op_name: str
) -> None:
    spec = _operator_specs()[op_name]
    x_extent = spec.get("x_tile_ele", 0) or VL_ELE
    x_tile = tla.tile_view(x_ub, tla.make_shape(x_extent), tla.make_coord(chunk_idx))
    y_tile = tla.tile_view(y_ub, tla.make_shape(VL_ELE), tla.make_coord(chunk_idx))
    z_tile = tla.tile_view(z_ub, tla.make_shape(VL_ELE), tla.make_coord(chunk_idx))
    x_load = spec.get("x_load", NormalLoadParams())
    x_store = spec.get("x_store", NormalStoreParams())
    if spec.get("store_loaded_x_only", False):
        z_tile.store(x_tile.load(x_load), x_store)
    else:
        z_tile.store(spec["op"](x_tile.load(x_load), y_tile.load()), x_store)


@tla.kernel
def binary_op_batch(mem_x: tla.Tensor, mem_y: tla.Tensor, mem_z: tla.Tensor) -> None:
    ub_loaded = tla.flag("ub_loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    vec_done = tla.flag("vec_done", tla.pipes.VECTOR, tla.pipes.MTE3)
    block_idx = tla.arch.block_idx()

    x_gm = tla.tile_view(mem_x, tla.make_shape(VECTOR_ELE), tla.make_coord(block_idx))
    y_gm = tla.tile_view(mem_y, tla.make_shape(VECTOR_ELE), tla.make_coord(block_idx))
    z_gm = tla.tile_view(mem_z, tla.make_shape(VECTOR_ELE), tla.make_coord(block_idx))
    x_ub_ptr = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    y_ub_ptr = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    z_ub_ptr = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    x_ub = tla.make_tensor_like(x_ub_ptr, x_gm, tla.arch.RowMajor)
    y_ub = tla.make_tensor_like(y_ub_ptr, y_gm, tla.arch.RowMajor)
    z_ub = tla.make_tensor_like(z_ub_ptr, z_gm, tla.arch.RowMajor)

    with tla.vector():
        tla.copy(x_ub, x_gm)
        tla.copy(y_ub, y_gm)
        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)
        with tla.vec.func(mode="simd"):
            for i in tla.range(LOOPS):
                if block_idx == 0:
                    _batch_binary_chunk(x_ub, y_ub, z_ub, i, _BATCH_OPS[0])
                elif block_idx == 1:
                    _batch_binary_chunk(x_ub, y_ub, z_ub, i, _BATCH_OPS[1])
                elif block_idx == 2:
                    _batch_binary_chunk(x_ub, y_ub, z_ub, i, _BATCH_OPS[2])
                else:
                    _batch_binary_chunk(x_ub, y_ub, z_ub, i, _BATCH_OPS[3])
        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)
        tla.copy(z_gm, z_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def _operator_specs() -> dict[str, dict[str, Any]]:
    return {
        "add": {
            "op": lambda lhs, rhs: lhs + rhs,
            "default_atol": 1e-4,
            "nonzero_rhs": False,
        },
        "sub": {
            "op": lambda lhs, rhs: lhs - rhs,
            "default_atol": 1e-4,
            "nonzero_rhs": False,
        },
        "mul": {
            "op": lambda lhs, rhs: lhs * rhs,
            "default_atol": 1e-4,
            "nonzero_rhs": False,
        },
        "div": {
            "op": lambda lhs, rhs: lhs / rhs,
            "default_atol": 1e-3,
            "nonzero_rhs": True,
        },
        "max": {
            "op": lambda lhs, rhs: tla.max(lhs, rhs),
            "default_atol": 1e-4,
            "nonzero_rhs": False,
        },
        "min": {
            "op": lambda lhs, rhs: tla.min(lhs, rhs),
            "default_atol": 1e-4,
            "nonzero_rhs": False,
        },
        "add_unalign": {
            "op": lambda lhs, rhs: lhs + rhs,
            "default_atol": 0,
            "x_load": UnalignLoadParams(),
            "x_store": UnalignStoreParams(),
        },
        "add_brc_b32": {
            "op": lambda lhs, rhs: lhs + rhs,
            "default_atol": 1e-4,
            "x_load": NormalLoadParams(load_dist=LoadDist.DIST_BRC_B32),
            "x_tile_ele": 1,
            "dtypes": ("f32",),
        },
    }


def _is_unsupported_case(op_name: str, dtype_name: str) -> bool:
    spec = _operator_specs().get(op_name, {})
    allowed = spec.get("dtypes")
    if allowed is not None and dtype_name not in allowed:
        return True
    if dtype_name == "i8" and op_name in {"mul", "div"}:
        return True
    if dtype_name == "bf16" and op_name == "div":
        return True
    return False


def _print_skip(op_name: str, dtype_name: str, shape: tuple[int, ...]) -> None:
    spec = _operator_specs().get(op_name, {})
    allowed = spec.get("dtypes")
    if allowed is not None and dtype_name not in allowed:
        reason = f"{op_name} only supports {', '.join(allowed)}"
    elif dtype_name == "i8" and op_name in {"mul", "div"}:
        reason = "i8 is not supported for mul/div"
    elif dtype_name == "bf16" and op_name == "div":
        reason = "bf16 vector div is not supported by AVE vdiv"
    else:
        reason = "unsupported case"
    print(f"skip op={op_name} dtype={dtype_name} shape={shape_label(shape)}: {reason}")


def _set_kernel_config(
    op_name: str, dtype_name: str, shape: tuple[int, ...] | None = None
) -> tuple[type[Any], Any, float | int]:
    global VL_ELE, LOOPS, VECTOR_ELE, _KERNEL_DTYPE, _KERNEL_ELEMENT_BYTES
    global \
        _KERNEL_SHAPE, \
        _BINARY_OP, \
        _X_LOAD_PARAMS, \
        _X_STORE_PARAMS, \
        _X_TILE_ELE, \
        _STORE_LOADED_X_ONLY
    specs = _operator_specs()
    if op_name not in specs:
        choices = ", ".join(sorted(specs))
        raise SystemExit(
            f"unknown binary operator {op_name!r}; expected one of: {choices}"
        )
    spec = specs[op_name]
    allowed = spec.get("dtypes", ALL_DTYPES)
    config = vector_kernel_config(dtype_name, shape, allowed)
    VECTOR_ELE = config.vector_elements
    _KERNEL_SHAPE = shape if shape is not None else (VECTOR_ELE,)
    VL_ELE = config.lanes
    LOOPS = config.loops
    _KERNEL_DTYPE = config.tla_dtype
    _KERNEL_ELEMENT_BYTES = config.element_bytes
    _BINARY_OP = spec["op"]
    _X_LOAD_PARAMS = spec.get("x_load", NormalLoadParams())
    _X_STORE_PARAMS = spec.get("x_store", NormalStoreParams())
    _X_TILE_ELE = spec.get("x_tile_ele", 0)
    _STORE_LOADED_X_ONLY = spec.get("store_loaded_x_only", False)
    return config.tla_dtype, config.torch_dtype, config.default_sentinel


def _configure_batch(
    ops: tuple[str, ...], dtype_name: str, shape: tuple[int, ...]
) -> tuple[type[Any], Any, float | int]:
    global VL_ELE, LOOPS, VECTOR_ELE, _KERNEL_DTYPE, _KERNEL_ELEMENT_BYTES
    global _KERNEL_SHAPE, _BATCH_OPS
    if not 1 <= len(ops) <= 4:
        raise SystemExit("binary batch requires one to four operations")
    config = vector_kernel_config(dtype_name, shape, ALL_DTYPES)
    VECTOR_ELE = config.vector_elements
    _KERNEL_SHAPE = shape
    VL_ELE = config.lanes
    LOOPS = config.loops
    _KERNEL_DTYPE = config.tla_dtype
    _KERNEL_ELEMENT_BYTES = config.element_bytes
    _BATCH_OPS = ops + (ops[-1],) * (4 - len(ops))
    return config.tla_dtype, config.torch_dtype, config.default_sentinel


def _make_inputs(args: Any, dtype_name: str, torch: Any) -> tuple[Any, Any]:
    _, _, _ = _set_kernel_config(args.op, dtype_name, args.shape)
    dtype = vector_kernel_config(
        dtype_name, args.shape, _operator_specs()[args.op].get("dtypes", ALL_DTYPES)
    ).torch_dtype
    device = "npu"
    if dtype_name in {"i8", "i16", "i32"}:
        arange = torch.arange(VECTOR_ELE, dtype=torch.int32, device=device)
        x = ((arange % 97) - 48).to(dtype)
        y = ((arange % 31) + (1 if args.op == "div" else -15)).to(dtype)
        if args.op == "div":
            y = y.clamp_min(1)
        return x, y
    x = torch.arange(VECTOR_ELE, dtype=dtype, device=device) + 1
    y = (torch.arange(VECTOR_ELE, dtype=dtype, device=device) * 0.5) + 2
    return x, y


def _expected(op_name: str, inputs: tuple[Any, ...]) -> Any:
    x, y = inputs
    if op_name == "add_brc_b32":
        out = y.clone()
        for i in range(LOOPS):
            start = i * VL_ELE
            end = min(start + VL_ELE, x.numel())
            out[start:end] = x[i] + y[start:end]
        return out
    if op_name in {"add", "add_unalign"}:
        return x + y
    if op_name == "sub":
        return x - y
    if op_name == "mul":
        return x * y
    if op_name == "div":
        if not x.is_floating_point():
            return x.div(y, rounding_mode="trunc")
        return x / y
    if op_name == "max":
        return x.maximum(y)
    if op_name == "min":
        return x.minimum(y)
    raise AssertionError(op_name)


HARNESS = DirectVectorOpHarness(
    DirectVectorOpConfig(
        description="Compile and run a vector binary op.",
        kernel=binary_op,
        all_dtypes=ALL_DTYPES,
        operator_specs=_operator_specs,
        set_kernel_config=_set_kernel_config,
        get_vector_elements=lambda: VECTOR_ELE,
        get_kernel_shape=lambda: _KERNEL_SHAPE,
        make_inputs=_make_inputs,
        expected=_expected,
        unsupported_case=_is_unsupported_case,
        print_skip=_print_skip,
        script_path=Path(__file__).resolve(),
        float_dtypes=frozenset({"f32", "f16", "bf16"}),
        input_count=2,
        batch_kernel=binary_op_batch,
        configure_batch=_configure_batch,
    )
)


def main() -> int:
    return HARNESS.main()


if __name__ == "__main__":
    raise SystemExit(main())
