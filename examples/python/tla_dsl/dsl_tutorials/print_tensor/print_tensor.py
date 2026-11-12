# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compile and run typed GM, UB, L1, or L0C tensor ``tla.print`` C310 cases."""

from __future__ import annotations

import argparse
import re
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, Callable, NamedTuple

import catlass.tla as tla
import sys


SOURCE_SHAPE = (8, 4)
UB_SHAPE = (4, 8)
L1_OUTER_DIMENSION = 64
L0C_SHAPE = (16, 16)
L0C_LARGE_ORIGIN_SHAPE = (32, 32)
CAPACITY_SHAPE = (262_112,)


class _DTypeSpec(NamedTuple):
    token: str
    torch_dtype: str
    tla_dtype: str
    values: tuple[float | int, ...]


class _RuntimeInput(NamedTuple):
    value: Any
    owner: Any


_FLOAT_VALUES = (
    0.0,
    -0.0,
    1.0,
    -2.5,
    float("nan"),
    float("inf"),
    float("-inf"),
    3.25,
) * 2


def _integer_values(minimum: int, maximum: int) -> tuple[int, ...]:
    return (minimum, maximum, 0, -1 if minimum < 0 else 1) * 4


DTYPE_SPECS = {
    "f16": _DTypeSpec("f16", "float16", "Float16", _FLOAT_VALUES),
    "f32": _DTypeSpec("f32", "float32", "Float32", _FLOAT_VALUES),
    "i8": _DTypeSpec("i8", "int8", "Int8", _integer_values(-128, 127)),
    "i16": _DTypeSpec("i16", "int16", "Int16", _integer_values(-32768, 32767)),
    "i32": _DTypeSpec(
        "i32", "int32", "Int32", _integer_values(-2147483648, 2147483647)
    ),
    "u8": _DTypeSpec("u8", "uint8", "UInt8", _integer_values(0, 255)),
    "u16": _DTypeSpec("u16", "uint16", "UInt16", _integer_values(0, 65535)),
    "u32": _DTypeSpec("u32", "uint32", "UInt32", _integer_values(0, 4294967295)),
}
EXPECTED_VALUES = list(DTYPE_SPECS["f32"].values)
_UNSIGNED_ITEMSIZE = {"u16": 2, "u32": 4}
_ELEMENT_BYTES = {
    "f16": 2,
    "f32": 4,
    "i8": 1,
    "i16": 2,
    "i32": 4,
    "u8": 1,
    "u16": 2,
    "u32": 4,
}
_SIGNED_STAGING_DTYPES = {"u8": "Int8", "u16": "Int16", "u32": "Int32"}
_KERNEL_DTYPE: Any = None
_KERNEL_COPY_DTYPE: Any = None
_KERNEL_ELEMENT_BYTES = 4
_KERNEL_UNSIGNED = False
_KERNEL_L0C_INPUT_DTYPE: Any = None
_KERNEL_L0C_ACC_DTYPE: Any = None
_KERNEL_L1_COPY_DTYPE: Any = None
_KERNEL_L1_COPY_ELEMENT_BYTES = 1
_KERNEL_LOCAL_CASE = "base"
_KERNEL_LOCAL_CALLS = 1
_KERNEL_L1_LAYOUT: Any = None


def _l1_shape(layout: Any, element_bytes: int) -> tuple[int, int]:
    c0_elements = 32 // element_bytes
    if layout in ("zn", tla.arch.zN):
        return (L1_OUTER_DIMENSION, c0_elements)
    return (c0_elements, L1_OUTER_DIMENSION)


def _repeat_values(
    values: tuple[float | int, ...], count: int
) -> tuple[float | int, ...]:
    repeats = (count + len(values) - 1) // len(values)
    return (values * repeats)[:count]


def _ub_row_major_layout() -> Any:
    return tla.make_layout(
        shape=tla.make_shape(*UB_SHAPE),
        stride=tla.make_stride(UB_SHAPE[1], 1),
    )


def _ub_copy_layout() -> Any:
    return tla.make_layout(
        shape=tla.make_shape(1, 32),
        stride=tla.make_stride(32, 1),
    )


def _configure_ub_kernel(spec: _DTypeSpec) -> None:
    global _KERNEL_COPY_DTYPE, _KERNEL_DTYPE, _KERNEL_ELEMENT_BYTES
    global _KERNEL_UNSIGNED
    _KERNEL_DTYPE = getattr(tla, spec.tla_dtype)
    staging_dtype = _SIGNED_STAGING_DTYPES.get(spec.token, spec.tla_dtype)
    _KERNEL_COPY_DTYPE = getattr(tla, staging_dtype)
    _KERNEL_ELEMENT_BYTES = _ELEMENT_BYTES[spec.token]
    _KERNEL_UNSIGNED = spec.token in _SIGNED_STAGING_DTYPES


def _configure_l0c_kernel(spec: _DTypeSpec) -> None:
    global _KERNEL_DTYPE, _KERNEL_L0C_ACC_DTYPE, _KERNEL_L0C_INPUT_DTYPE
    _KERNEL_DTYPE = getattr(tla, spec.tla_dtype)
    if spec.token.startswith(("i", "u")):
        _KERNEL_L0C_INPUT_DTYPE = tla.Int8
        _KERNEL_L0C_ACC_DTYPE = tla.Int32
    else:
        _KERNEL_L0C_INPUT_DTYPE = tla.Float32
        _KERNEL_L0C_ACC_DTYPE = tla.Float32


def _configure_l1_kernel(spec: _DTypeSpec) -> None:
    global _KERNEL_DTYPE, _KERNEL_ELEMENT_BYTES, _KERNEL_L1_COPY_DTYPE
    global _KERNEL_L1_COPY_ELEMENT_BYTES
    global _KERNEL_UNSIGNED
    _KERNEL_DTYPE = getattr(tla, spec.tla_dtype)
    _KERNEL_ELEMENT_BYTES = _ELEMENT_BYTES[spec.token]
    _KERNEL_L1_COPY_DTYPE = (
        _KERNEL_DTYPE if spec.token in ("f16", "f32", "i8") else tla.Int8
    )
    _KERNEL_L1_COPY_ELEMENT_BYTES = (
        _KERNEL_ELEMENT_BYTES if spec.token in ("f16", "f32", "i8") else 1
    )
    _KERNEL_UNSIGNED = spec.token.startswith("u")


def _dynamic_row_major_layout(rows: Any) -> Any:
    return tla.make_layout(
        shape=tla.make_shape(rows, SOURCE_SHAPE[1]),
        stride=tla.make_stride(SOURCE_SHAPE[1], 1),
    )


@tla.kernel
def print_tensor_aiv_kernel(value: tla.Tensor) -> None:
    with tla.vector():
        tla.print(value, 16)


@tla.kernel
def print_tensor_aiv_two_calls_kernel(value: tla.Tensor) -> None:
    with tla.vector():
        tla.print(value, 16)
        tla.print(value, 8)


@tla.kernel
def print_tensor_aiv_dynamic_control_flow_kernel(
    value: tla.Tensor, enabled: tla.Int32, repeats: tla.Int32
) -> None:
    """Print a tensor from runtime ``if`` and ``scf.for`` control flow.

    ``enabled`` deliberately permits an execution with no native print record;
    ``repeats`` makes the one static print site emit the record repeatedly.
    """
    with tla.vector():
        if enabled != 0:
            for _ in tla.range(0, repeats, 1):
                tla.print(value, 16)


@tla.kernel
def print_tensor_dynamic_aiv_kernel(
    value: tla.Tensor, rows: tla.Int32, length: tla.Int32
) -> None:
    tensor = tla.make_tensor(value.ptr, _dynamic_row_major_layout(rows))
    with tla.vector():
        tla.print(tensor, length)


@tla.kernel
def print_tensor_capacity_aiv_kernel(value: tla.Tensor) -> None:
    with tla.vector():
        tla.print(value, CAPACITY_SHAPE[0])


@tla.kernel
def print_tensor_ub_base_kernel(value: tla.Tensor) -> None:
    loaded, copy_ub, gm, print_ub = _prepare_ub_tensors(value, 0)
    with tla.vector():
        _print_ub_tensor(loaded, copy_ub, gm, print_ub, calls=1)


@tla.kernel
def print_tensor_ub_base_two_calls_kernel(value: tla.Tensor) -> None:
    loaded, copy_ub, gm, print_ub = _prepare_ub_tensors(value, 0)
    with tla.vector():
        _print_ub_tensor(loaded, copy_ub, gm, print_ub, calls=2)


@tla.kernel
def print_tensor_ub_aligned_offset_kernel(value: tla.Tensor) -> None:
    loaded, copy_ub, gm, print_ub = _prepare_ub_tensors(
        value, 32 // _KERNEL_ELEMENT_BYTES
    )
    with tla.vector():
        _print_ub_tensor(loaded, copy_ub, gm, print_ub, calls=1)


def _prepare_ub_tensors(
    value: tla.Tensor, element_offset: int
) -> tuple[Any, Any, Any, Any]:
    loaded = tla.flag("print_ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    allocation = tla.allocate(
        32 + element_offset, _KERNEL_COPY_DTYPE, tla.AddressSpace.ub, 256
    )
    copy_ptr = allocation + element_offset
    gm_ptr = value.ptr
    print_ptr = copy_ptr
    if _KERNEL_UNSIGNED:
        gm_ptr = tla.recast_ptr(gm_ptr, dtype=_KERNEL_COPY_DTYPE)
        print_ptr = tla.recast_ptr(print_ptr, dtype=_KERNEL_DTYPE)
    copy_layout = _ub_copy_layout()
    gm = tla.make_tensor(gm_ptr, copy_layout)
    copy_ub = tla.make_tensor(copy_ptr, copy_layout)
    print_ub = tla.make_tensor(print_ptr, _ub_row_major_layout())
    return loaded, copy_ub, gm, print_ub


def _print_ub_tensor(
    loaded: Any, copy_ub: Any, gm: Any, print_ub: Any, *, calls: int
) -> None:
    tla.copy(copy_ub, gm)
    tla.set_flag(loaded)
    tla.wait_flag(loaded)
    tla.print(print_ub, 16)
    if calls == 2:
        tla.print(print_ub, 8)


@tla.kernel
def print_tensor_ub_dynamic_kernel(
    value: tla.Tensor, rows: tla.Int32, length: tla.Int32
) -> None:
    loaded = tla.flag("print_ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    ptr = tla.allocate(32, tla.Float32, tla.AddressSpace.ub, 256)
    copy_layout = _ub_row_major_layout()
    gm = tla.make_tensor(value.ptr, copy_layout)
    ub = tla.make_tensor(ptr, copy_layout)
    dynamic_ub = tla.make_tensor(ptr, _dynamic_row_major_layout(rows))
    with tla.vector():
        tla.copy(ub, gm)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        tla.print(dynamic_ub, length)


def _make_l1_source(
    value: tla.Tensor, layout: Any, dtype: Any, element_bytes: int
) -> Any:
    source_ptr = tla.recast_ptr(value.ptr, dtype=dtype)
    source_shape = _l1_shape(layout, element_bytes)
    if layout == tla.arch.zN:
        source_layout = tla.make_layout(
            shape=tla.make_shape(*source_shape),
            stride=tla.make_stride(source_shape[1], 1),
        )
    else:
        source_layout = tla.make_layout(
            shape=tla.make_shape(*source_shape),
            stride=tla.make_stride(1, source_shape[0]),
            layoutTag=tla.arch.ColumnMajor,
        )
    return tla.make_tensor(source_ptr, source_layout)


def _prepare_l1_tensors(
    value: tla.Tensor, layout: Any, byte_offset: int
) -> tuple[Any, Any]:
    element_offset = byte_offset // _KERNEL_L1_COPY_ELEMENT_BYTES
    allocation_elements = (L1_OUTER_DIMENSION * 32) // _KERNEL_L1_COPY_ELEMENT_BYTES
    allocation = tla.allocate(
        allocation_elements + element_offset,
        _KERNEL_L1_COPY_DTYPE,
        tla.AddressSpace.l1,
        256,
    )
    copy_ptr = allocation + element_offset
    print_ptr = tla.recast_ptr(copy_ptr, dtype=_KERNEL_DTYPE)
    return (
        tla.make_tensor_like(
            copy_ptr,
            _make_l1_source(
                value, layout, _KERNEL_L1_COPY_DTYPE, _KERNEL_L1_COPY_ELEMENT_BYTES
            ),
            layout,
        ),
        tla.make_tensor_like(
            print_ptr,
            _make_l1_source(value, layout, _KERNEL_DTYPE, _KERNEL_ELEMENT_BYTES),
            layout,
        ),
    )


def _l1_tile(
    tensor: tla.Tensor,
    layout: Any,
    outer_coord: int,
    element_bytes: int | None = None,
) -> Any:
    if element_bytes is None:
        element_bytes = _KERNEL_ELEMENT_BYTES
    c0_elements = 32 // element_bytes
    if layout == tla.arch.zN:
        shape = tla.make_shape(16, c0_elements)
        coord = tla.make_coord(outer_coord, 0)
    else:
        shape = tla.make_shape(c0_elements, 16)
        coord = tla.make_coord(0, outer_coord)
    return tla.tile_view(tensor, shape, coord)


@tla.jit
def _print_l1(
    value: tla.Tensor,
    *,
    layout: tla.Constexpr[Any],
    outer_coord: tla.Constexpr[int],
    calls: tla.Constexpr[int],
    block_dependent: tla.Constexpr[bool] = False,
) -> None:
    loaded = tla.flag("print_l1_loaded", tla.arch.MTE2, tla.arch.MTE1)
    copy_l1, print_l1 = _prepare_l1_tensors(value, layout, 0)
    source = _make_l1_source(
        value, layout, _KERNEL_L1_COPY_DTYPE, _KERNEL_L1_COPY_ELEMENT_BYTES
    )
    with tla.cube():
        if outer_coord:
            copy_l1 = _l1_tile(
                copy_l1, layout, outer_coord, _KERNEL_L1_COPY_ELEMENT_BYTES
            )
            source = _l1_tile(
                source, layout, outer_coord, _KERNEL_L1_COPY_ELEMENT_BYTES
            )
            print_l1 = _l1_tile(print_l1, layout, outer_coord)
        tla.copy(copy_l1, source)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        if block_dependent:
            if tla.arch.block_idx() == 0:
                tla.print(_l1_tile(print_l1, layout, 0), 8)
            else:
                tla.print(_l1_tile(print_l1, layout, 2), 8)
        else:
            tla.print(print_l1, 8)
            if calls == 2:
                tla.print(print_l1, 4)


@tla.kernel
def print_tensor_l1_kernel(value: tla.Tensor) -> None:
    _print_l1(
        value,
        layout=_KERNEL_L1_LAYOUT,
        outer_coord=2 if _KERNEL_LOCAL_CASE == "aligned-offset" else 0,
        calls=_KERNEL_LOCAL_CALLS,
        block_dependent=_KERNEL_LOCAL_CASE == "block-dependent",
    )


@tla.jit
def _print_l0c(
    lhs: tla.Tensor,
    rhs: tla.Tensor,
    *,
    element_offset: tla.Constexpr[int],
    calls: tla.Constexpr[int],
    dimension: tla.Constexpr[int] = 16,
) -> None:
    l1_loaded = tla.flag("print_l0c_l1_loaded", tla.arch.MTE2, tla.arch.MTE1)
    l0_loaded = tla.flag("print_l0c_l0_loaded", tla.arch.MTE1, tla.arch.CUBE)
    mm_completed = tla.flag("print_l0c_mmad_done", tla.arch.CUBE, tla.arch.FIX)
    element_count = dimension * dimension
    l1a_ptr = tla.allocate(
        element_count, _KERNEL_L0C_INPUT_DTYPE, tla.AddressSpace.l1, 512
    )
    l1b_ptr = tla.allocate(
        element_count, _KERNEL_L0C_INPUT_DTYPE, tla.AddressSpace.l1, 512
    )
    l0a_ptr = tla.allocate(
        element_count, _KERNEL_L0C_INPUT_DTYPE, tla.AddressSpace.l0a, 512
    )
    l0b_ptr = tla.allocate(
        element_count, _KERNEL_L0C_INPUT_DTYPE, tla.AddressSpace.l0b, 512
    )
    l0c_allocation = tla.allocate(
        element_count + element_offset,
        _KERNEL_L0C_ACC_DTYPE,
        tla.AddressSpace.l0c,
        1024,
    )
    l0c_ptr = l0c_allocation + element_offset
    with tla.cube():
        l1_a = tla.make_tensor_like(l1a_ptr, lhs, tla.arch.zN)
        l1_b = tla.make_tensor_like(l1b_ptr, rhs, tla.arch.zN)
        tla.copy(l1_a, lhs)
        tla.copy(l1_b, rhs)
        tla.set_flag(l1_loaded)
        tla.wait_flag(l1_loaded)
        l0_a = tla.make_tensor_like(l0a_ptr, l1_a, tla.arch.zN)
        l0_b = tla.make_tensor_like(l0b_ptr, l1_b, tla.arch.nZ)
        l0_c = tla.make_tensor_like(l0c_ptr, lhs, tla.arch.L0Clayout)
        tla.copy(l0_a, l1_a)
        tla.copy(l0_b, l1_b)
        tla.set_flag(l0_loaded)
        tla.wait_flag(l0_loaded)
        tla.mmad(l0_c, l0_a, l0_b, init_c=True)
        tla.set_flag(mm_completed)
        tla.wait_flag(mm_completed)
        print_ptr = tla.recast_ptr(l0c_ptr, dtype=_KERNEL_DTYPE)
        print_l0_c = tla.make_tensor_like(print_ptr, lhs, tla.arch.L0Clayout)
        if dimension == 32:
            print_l0_c = tla.tile_view(
                print_l0_c,
                tla.make_shape(16, 16),
                tla.make_coord(1, 0),
            )
        tla.print(print_l0_c, 256)
        if calls == 2:
            tla.print(print_l0_c, 256)


@tla.kernel
def print_tensor_l0c_kernel(lhs: tla.Tensor, rhs: tla.Tensor) -> None:
    _print_l0c(
        lhs,
        rhs,
        element_offset=256 if _KERNEL_LOCAL_CASE == "aligned-offset" else 0,
        calls=_KERNEL_LOCAL_CALLS,
        dimension=32 if _KERNEL_LOCAL_CASE == "larger-origin" else 16,
    )


def _kernel(args: argparse.Namespace) -> Callable[..., None]:
    global _KERNEL_L1_LAYOUT, _KERNEL_LOCAL_CALLS, _KERNEL_LOCAL_CASE
    if args.storage == "l0c":
        if args.arch_scope != "aic.c310":
            raise tla.TlaExecutionError("L0C tensor tla.print requires AIC")
        if (args.case, args.calls) not in {
            ("base", 1),
            ("base", 2),
            ("aligned-offset", 1),
            ("larger-origin", 1),
        }:
            raise tla.TlaExecutionError("unsupported L0C tensor case/call combination")
        _KERNEL_LOCAL_CASE, _KERNEL_LOCAL_CALLS = args.case, args.calls
        return print_tensor_l0c_kernel
    if args.storage == "l1":
        if args.arch_scope != "aic.c310":
            raise tla.TlaExecutionError("L1 tensor tla.print requires AIC")
        if (args.layout, args.case, args.calls) not in {
            ("zn", "base", 1),
            ("zn", "base", 2),
            ("zn", "block-dependent", 1),
            ("zn", "aligned-offset", 1),
            ("nz", "base", 1),
            ("nz", "base", 2),
            ("nz", "aligned-offset", 1),
        }:
            raise tla.TlaExecutionError("unsupported L1 tensor layout/case")
        _KERNEL_L1_LAYOUT = tla.arch.zN if args.layout == "zn" else tla.arch.nZ
        _KERNEL_LOCAL_CASE, _KERNEL_LOCAL_CALLS = args.case, args.calls
        return print_tensor_l1_kernel
    if args.case == "dynamic-control-flow":
        if args.storage != "gm":
            raise tla.TlaExecutionError(
                "dynamic-control-flow tensor tla.print requires GM storage"
            )
        if args.calls != 1:
            raise tla.TlaExecutionError(
                "the dynamic-control-flow case supports one static print site"
            )
        return print_tensor_aiv_dynamic_control_flow_kernel
    if args.storage == "ub":
        kernels = {
            ("base", 1): print_tensor_ub_base_kernel,
            ("base", 2): print_tensor_ub_base_two_calls_kernel,
            ("aligned-offset", 1): print_tensor_ub_aligned_offset_kernel,
            ("dynamic", 1): print_tensor_ub_dynamic_kernel,
        }
        try:
            return kernels[(args.case, args.calls)]
        except KeyError as exc:
            raise tla.TlaExecutionError(
                "unsupported UB tensor case/call combination"
            ) from exc
    if args.case == "capacity":
        if args.calls != 1:
            raise tla.TlaExecutionError("the capacity case supports one call")
        return print_tensor_capacity_aiv_kernel
    if args.dynamic_shape:
        if args.calls != 1:
            raise tla.TlaExecutionError("the dynamic GM case supports one call")
        return print_tensor_dynamic_aiv_kernel
    kernels = {
        1: print_tensor_aiv_kernel,
        2: print_tensor_aiv_two_calls_kernel,
    }
    try:
        return kernels[args.calls]
    except KeyError as exc:
        raise tla.TlaExecutionError(
            "tensor tla.print example supports --calls 1 or 2"
        ) from exc


def _format_record(
    spec_or_values: _DTypeSpec | list[float | int] = DTYPE_SPECS["f32"],
    *,
    values: tuple[float | int, ...] | list[float | int] | None = None,
    shape: tuple[int, ...] = SOURCE_SHAPE,
    subblock: int | None = 0,
) -> str:
    from catlass.execution import _format_print_tensor_record

    if isinstance(spec_or_values, _DTypeSpec):
        spec = spec_or_values
        record_values = spec.values if values is None else values
    else:
        spec = DTYPE_SPECS["f32"]
        record_values = spec_or_values if values is None else values
    return _format_print_tensor_record(
        record_values,
        shape=shape,
        dtype=spec.token,
        subblock=subblock,
    )


def _public_records(output: str) -> list[str]:
    return [
        line.strip()
        for line in output.splitlines()
        if line.strip().startswith("tla.print ")
    ]


def _verify_public_output(
    output: str,
    spec: _DTypeSpec = DTYPE_SPECS["f32"],
    *,
    values: tuple[float | int, ...] | list[float | int] | None = None,
    shape: tuple[int, ...] = SOURCE_SHAPE,
    subblock: int | None = 0,
) -> str:
    expected = _format_record(spec, values=values, shape=shape, subblock=subblock)
    records = _public_records(output)
    if records != [expected]:
        raise tla.TlaExecutionError(
            "tensor tla.print native initialization or decoding failed: "
            f"expected exactly {expected!r}, got {records!r}"
        )
    return expected


def _verify_multi_record_public_output(
    output: str,
    *,
    calls: int,
    block_count: int,
    spec: _DTypeSpec = DTYPE_SPECS["f32"],
    values: tuple[float | int, ...] | list[float | int] | None = None,
    values_by_identity: dict[
        tuple[int, int, int | None], tuple[float | int, ...] | list[float | int]
    ]
    | None = None,
    shape: tuple[int, ...] = SOURCE_SHAPE,
    first_count: int = 16,
    second_count: int | None = None,
    subblocks: tuple[int | None, ...] = (0,),
) -> str:
    from catlass.execution import _format_print_tensor_record

    record_values = spec.values if values is None else values
    records = _public_records(output)
    expected_identities = (
        set(values_by_identity)
        if values_by_identity is not None
        else {
            (call, block, subblock)
            for call in range(calls)
            for block in range(block_count)
            for subblock in subblocks
        }
    )
    seen: set[tuple[int, int, int | None]] = set()
    expected = []
    for record in records:
        match = re.match(
            r"^tla\.print call=(\d+) block=(\d+) "
            r"dtype=\S+(?: position=\S+)?(?: subblock=(\d+))? ",
            record,
        )
        if match is None:
            raise tla.TlaExecutionError(
                f"tensor tla.print has malformed record {record!r}"
            )
        call = int(match.group(1))
        block = int(match.group(2))
        subblock = int(match.group(3)) if match.group(3) is not None else None
        identity = (call, block, subblock)
        if identity not in expected_identities or identity in seen:
            raise tla.TlaExecutionError(
                f"tensor tla.print has unexpected record identity {identity!r}"
            )
        seen.add(identity)
        count = (
            first_count
            if call == 0
            else (second_count if second_count is not None else first_count // 2)
        )
        identity_values = (
            record_values
            if values_by_identity is None
            else values_by_identity[identity]
        )
        expected.append(
            _format_print_tensor_record(
                identity_values[:count],
                shape=shape,
                dtype=spec.token,
                call=call,
                block=block,
                subblock=subblock,
            )
        )
    if seen != expected_identities or records != expected:
        raise tla.TlaExecutionError(
            f"tensor tla.print expected {expected_identities!r}, got {records!r}"
        )
    return "\n".join(expected)


def _verify_dynamic_control_flow_public_output(
    output: str,
    spec: _DTypeSpec = DTYPE_SPECS["f32"],
    *,
    values: tuple[float | int, ...] | list[float | int] | None = None,
    shape: tuple[int, ...] = SOURCE_SHAPE,
) -> str:
    """Validate best-effort records from a dynamic tensor-print site.

    A disabled branch legitimately produces no record; a loop may produce the
    same static print site more than once.  Each emitted public record must
    nevertheless retain the exact dtype, location, shape, count, and values.
    """
    expected = _format_record(spec, values=values, shape=shape)
    records = _public_records(output)
    malformed = [record for record in records if record != expected]
    if malformed:
        raise tla.TlaExecutionError(
            "dynamic-control-flow tensor tla.print has malformed record(s): "
            f"expected {expected!r}, got {malformed!r}"
        )
    return "\n".join(records)


def _make_external_unsigned_input(
    torch: Any,
    spec: _DTypeSpec,
    *,
    values: tuple[float | int, ...],
    shape: tuple[int, ...],
) -> _RuntimeInput:
    import numpy as np

    byte_view = np.asarray(
        values, dtype=np.dtype(f"<u{_UNSIGNED_ITEMSIZE[spec.token]}")
    ).view(np.uint8)
    owner = (
        torch.from_numpy(byte_view.copy())
        .to(device="npu", dtype=torch.uint8)
        .contiguous()
    )
    # Unsigned element types without a native torch dtype: bind the byte buffer
    # via from_dlpack and rely on kernel/print paths that accept the storage view.
    value = tla.from_dlpack(
        owner,
        layout_tag=tla.arch.RowMajor,
        origin_shape=shape,
    )
    return _RuntimeInput(value, owner)


def _make_runtime_input(
    torch: Any,
    spec: _DTypeSpec,
    *,
    values: tuple[float | int, ...],
    shape: tuple[int, ...],
) -> _RuntimeInput:
    torch_dtype = getattr(torch, spec.torch_dtype, None)
    if torch_dtype is not None:
        try:
            owner = (
                torch.tensor(values, dtype=torch_dtype, device="npu")
                .reshape(shape)
                .contiguous()
            )
            return _RuntimeInput(
                tla.from_dlpack(owner, layout_tag=tla.arch.RowMajor), owner
            )
        except (AttributeError, RuntimeError, TypeError):
            if spec.token not in _UNSIGNED_ITEMSIZE:
                raise
    if spec.token not in _UNSIGNED_ITEMSIZE:
        raise tla.TlaExecutionError(
            f"torch does not expose the required {spec.torch_dtype} dtype"
        )
    return _make_external_unsigned_input(torch, spec, values=values, shape=shape)


def _compile(
    args: argparse.Namespace,
    kernel: Callable[..., None],
    kernel_args: tuple[Any, ...],
) -> Any:
    return tla.compile(kernel, *kernel_args, options="--npu-arch 3510")


def _prepare_l1_source(
    args: argparse.Namespace, torch: Any, spec: _DTypeSpec
) -> tuple[Any, Any, list[float | int]]:
    import numpy as np

    typed_shape = _l1_shape(args.layout, _ELEMENT_BYTES[spec.token])
    copy_shape = _l1_shape(args.layout, _KERNEL_L1_COPY_ELEMENT_BYTES)
    typed_dtype = np.dtype(spec.torch_dtype)
    if args.case == "block-dependent":
        typed = np.arange(typed_shape[0] * typed_shape[1], dtype=typed_dtype).reshape(
            typed_shape
        )
    else:
        typed = np.asarray(
            _repeat_values(spec.values, typed_shape[0] * typed_shape[1]),
            dtype=typed_dtype,
        ).reshape(typed_shape)
    staged = spec.token not in ("f16", "f32", "i8")
    if staged and args.layout == "nz":
        physical = typed.T.copy().view(np.uint8).reshape(copy_shape[1], copy_shape[0]).T
    else:
        physical = typed.view(np.uint8).reshape(copy_shape) if staged else typed
    layout_tag = tla.arch.RowMajor
    if args.layout == "nz":
        physical = physical.T.copy()
        layout_tag = tla.arch.ColumnMajor
    owner = torch.from_numpy(physical.copy()).to(device="npu").contiguous()
    value = tla.from_dlpack(owner, layout_tag=layout_tag)
    if args.case == "block-dependent":
        expected = typed[0, :8].tolist()
    elif args.case == "aligned-offset" and args.layout == "zn":
        expected = typed[32, :8].tolist()
    elif args.case == "aligned-offset":
        expected = typed[:8, 32].tolist()
    elif args.layout == "zn":
        expected = typed[0, :8].tolist()
    else:
        expected = typed[:8, 0].tolist()
    return owner, value, expected


def _run_spec(args: argparse.Namespace, torch: Any, spec: _DTypeSpec) -> None:
    _configure_ub_kernel(spec)
    if args.storage == "l0c":
        _configure_l0c_kernel(spec)
    elif args.storage == "l1":
        _configure_l1_kernel(spec)
    kernel = _kernel(args)
    source_shape = (
        L0C_LARGE_ORIGIN_SHAPE
        if args.storage == "l0c" and args.case == "larger-origin"
        else L0C_SHAPE
        if args.storage == "l0c"
        else _l1_shape(args.layout, _ELEMENT_BYTES[spec.token])
        if args.storage == "l1"
        else (CAPACITY_SHAPE if args.case == "capacity" else SOURCE_SHAPE)
    )
    if args.case == "capacity":
        source = (
            torch.arange(CAPACITY_SHAPE[0], dtype=torch.float32, device="npu")
            .reshape(CAPACITY_SHAPE)
            .contiguous()
        )
        value = tla.from_dlpack(source, layout_tag=tla.arch.RowMajor)
        expected_values: list[float | int] = list(map(float, range(CAPACITY_SHAPE[0])))
    elif args.storage == "l0c":
        import numpy as np

        integer_view = spec.token.startswith(("i", "u"))
        input_dtype = torch.int8 if integer_view else torch.float32
        dimension = source_shape[0]
        lhs_source = torch.eye(dimension, dtype=input_dtype, device="npu")
        rhs_values = (
            torch.arange(dimension * dimension, dtype=torch.int32).reshape(source_shape)
            % 64
        )
        rhs_source = rhs_values.to(dtype=input_dtype, device="npu").contiguous()
        lhs_value = tla.from_dlpack(lhs_source, layout_tag=tla.arch.RowMajor)
        rhs_value = tla.from_dlpack(rhs_source, layout_tag=tla.arch.RowMajor)
        source = rhs_source
        value = lhs_value
        second_value = rhs_value
        accumulator_dtype = np.int32 if integer_view else np.float32
        expected_tile = (
            rhs_values[16:32, :16]
            if args.case == "larger-origin"
            else rhs_values[:16, :16]
        )
        expected_values = (
            np.asarray(expected_tile.flatten().tolist(), dtype=accumulator_dtype)
            .view(np.dtype(spec.torch_dtype))[:256]
            .tolist()
        )
    elif args.storage == "l1":
        source, value, expected_values = _prepare_l1_source(args, torch, spec)
    else:
        runtime_input = _make_runtime_input(
            torch,
            spec,
            values=spec.values * 2,
            shape=SOURCE_SHAPE,
        )
        source = runtime_input.owner
        value = runtime_input.value
        expected_values = list(spec.values)
        if args.layout == "column-major":
            source = source.detach().cpu().permute(1, 0).contiguous().npu()
            value = tla.from_dlpack(source, layout_tag=tla.arch.ColumnMajor)
            expected_values = source.flatten()[: len(spec.values)].tolist()
    if args.case == "dynamic-control-flow":
        kernel_args = (value, tla.Int32(args.enabled), tla.Int32(args.repeats))
    elif (args.storage == "ub" and args.case == "dynamic") or (
        args.storage == "gm" and args.dynamic_shape
    ):
        kernel_args = (value, tla.Int32(SOURCE_SHAPE[0]), tla.Int32(16))
    else:
        kernel_args = (value, second_value) if args.storage == "l0c" else (value,)
    executor = _compile(args, kernel, kernel_args)
    captured = StringIO()
    with redirect_stdout(captured):
        executor(*kernel_args, block_num=args.block_num)
    output_shape = (
        L0C_SHAPE
        if args.storage == "l0c"
        else (
            _l1_shape(args.layout, _ELEMENT_BYTES[spec.token])
            if args.case not in ("aligned-offset", "block-dependent")
            else (
                (16, 32 // _ELEMENT_BYTES[spec.token])
                if args.layout == "zn"
                else (32 // _ELEMENT_BYTES[spec.token], 16)
            )
        )
        if args.storage == "l1"
        else UB_SHAPE
        if args.storage == "ub" and args.case != "dynamic"
        else source_shape
    )
    if args.case == "dynamic-control-flow":
        rendered = _verify_dynamic_control_flow_public_output(
            captured.getvalue(),
            spec,
            values=expected_values,
            shape=output_shape,
        )
    elif args.case == "block-dependent":
        c0_elements = 32 // _ELEMENT_BYTES[spec.token]
        rendered = _verify_multi_record_public_output(
            captured.getvalue(),
            calls=2,
            block_count=2,
            spec=spec,
            values_by_identity={
                (0, 0, None): [float(value) for value in range(8)],
                (1, 1, None): [float(32 * c0_elements + value) for value in range(8)],
            },
            shape=output_shape,
            first_count=8,
            second_count=8,
            subblocks=(None,),
        )
    elif args.calls == 1 and args.block_num == 1:
        rendered = _verify_public_output(
            captured.getvalue(),
            spec,
            values=expected_values,
            shape=output_shape,
            subblock=None if args.storage in ("l1", "l0c") else 0,
        )
    else:
        rendered = _verify_multi_record_public_output(
            captured.getvalue(),
            calls=args.calls,
            block_count=args.block_num,
            spec=spec,
            values=expected_values,
            shape=output_shape,
            first_count=256
            if args.storage == "l0c"
            else 8
            if args.storage == "l1"
            else 16,
            second_count=256 if args.storage == "l0c" else None,
            subblocks=(None,) if args.storage in ("l1", "l0c") else (0,),
        )
    print(rendered)
    print(f"case dtype={spec.token} compile_ok=True")
    print(f"case dtype={spec.token} launch_ok=True")
    print(f"case dtype={spec.token} output_ok=True")


def run(args: argparse.Namespace) -> int:
    if args.case == "capacity" and (
        args.storage != "gm" or args.dynamic_shape or args.block_num != 1
    ):
        raise tla.TlaExecutionError(
            "the capacity case requires static GM printing with --block-num 1"
        )
    if args.layout == "column-major" and (
        args.storage != "gm" or args.case != "base" or args.dynamic_shape
    ):
        raise tla.TlaExecutionError(
            "the column-major case requires static GM base printing"
        )
    if (
        args.case in ("capacity", "dynamic")
        or args.dynamic_shape
        or args.layout == "column-major"
    ) and (args.all_dtypes or args.dtype != "f32"):
        raise tla.TlaExecutionError(
            "capacity, dynamic-shape, and column-major cases require --dtype f32"
        )
    if args.case == "dynamic-control-flow" and (
        args.storage != "gm"
        or args.dynamic_shape
        or args.layout != "row-major"
        or args.block_num != 1
        or args.dtype != "f32"
        or args.all_dtypes
        or args.calls != 1
        or args.repeats < 0
    ):
        raise tla.TlaExecutionError(
            "the dynamic-control-flow case requires static f32 GM printing, "
            "--block-num 1, one static call, and non-negative --repeats"
        )
    if args.case == "block-dependent" and (
        args.storage != "l1"
        or args.layout != "zn"
        or args.arch_scope != "aic.c310"
        or args.block_num != 2
        or args.dtype != "f32"
        or args.all_dtypes
        or args.calls != 1
    ):
        raise tla.TlaExecutionError(
            "the block-dependent case requires f32 zN L1 printing, "
            "--block-num 2, and --calls 1"
        )

    import torch
    import torch_npu

    torch.npu.set_device(args.device)
    specs = DTYPE_SPECS.values() if args.all_dtypes else (DTYPE_SPECS[args.dtype],)
    for spec in specs:
        _run_spec(args, torch, spec)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    dtype = parser.add_mutually_exclusive_group()
    dtype.add_argument("--dtype", choices=tuple(DTYPE_SPECS), default="f32")
    dtype.add_argument("--all-dtypes", action="store_true")
    parser.add_argument("--arch-scope", default="aiv.c310")
    parser.add_argument("--storage", choices=("gm", "ub", "l1", "l0c"), default="gm")
    parser.add_argument(
        "--layout",
        choices=("row-major", "column-major", "zn", "nz"),
        default="row-major",
    )
    parser.add_argument(
        "--case",
        choices=(
            "base",
            "aligned-offset",
            "block-dependent",
            "larger-origin",
            "dynamic",
            "dynamic-control-flow",
            "capacity",
        ),
        default="base",
    )
    parser.add_argument("--dynamic-shape", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--block-num", type=int, default=1)
    parser.add_argument("--calls", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--enabled",
        type=int,
        choices=(0, 1),
        default=1,
        help="Runtime branch predicate for --case dynamic-control-flow.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="Runtime loop trip count for --case dynamic-control-flow.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not args.run:
        raise SystemExit("pass --run")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
