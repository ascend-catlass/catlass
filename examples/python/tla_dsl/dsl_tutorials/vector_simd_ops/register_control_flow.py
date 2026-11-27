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
from typing import Any

import catlass.tla as tla

from vector_op_harness import (
    DirectVectorOpConfig,
    DirectVectorOpHarness,
    vector_kernel_config,
)

VECTOR_ELE = 64
VL_ELE = 64
ALL_DTYPES = ("f32",)

_KERNEL_DTYPE = tla.Float32
_KERNEL_SHAPE = (VECTOR_ELE,)


@tla.kernel
def register_control_flow(mem_src: tla.Tensor, mem_out: tla.Tensor) -> None:
    """Carry a VectorSSA and MaskSSA together through ``scf.for``."""
    ub_loaded = tla.flag("ub_loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    vec_done = tla.flag("vec_done", tla.pipes.VECTOR, tla.pipes.MTE3)

    src_ub_ptr = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    out_ub_ptr = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    src_ub = tla.make_tensor_like(src_ub_ptr, mem_src, tla.arch.RowMajor)
    out_ub = tla.make_tensor_like(out_ub_ptr, mem_out, tla.arch.RowMajor)

    with tla.vector():
        tla.copy(src_ub, mem_src)
        # Initialize masked-out lanes so their expected values are deterministic.
        tla.copy(out_ub, mem_out)

        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)
        with tla.vec.func(mode="simd"):
            value = src_ub.load()
            mask = tla.create_mask(pattern=tla.mask.H, dtype=_KERNEL_DTYPE)

            # Both values become scf.for iter_args/results. Two inversions return
            # the predicate to H while keeping MaskSSA live across the loop edge.
            for _ in tla.range(2):
                value = tla.abs(value)
                mask = tla.bitwise_not(mask)

            out_ub.store(value, mask=mask)

        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)
        tla.copy(mem_out, out_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def _operator_specs() -> dict[str, dict[str, Any]]:
    return {"register_carriers": {"default_atol": 1e-5}}


def _set_kernel_config(
    op_name: str, dtype_name: str, shape: tuple[int, ...] | None = None
) -> tuple[type[Any], Any, float]:
    global VECTOR_ELE, VL_ELE, _KERNEL_DTYPE, _KERNEL_SHAPE
    if op_name not in _operator_specs():
        raise SystemExit("unknown register control-flow operator")
    config = vector_kernel_config(dtype_name, shape, ALL_DTYPES)
    if config.vector_elements != config.lanes:
        raise SystemExit(
            "register_control_flow requires exactly one physical vector register"
        )
    VECTOR_ELE = config.vector_elements
    VL_ELE = config.lanes
    _KERNEL_DTYPE = config.tla_dtype
    _KERNEL_SHAPE = shape if shape is not None else (VECTOR_ELE,)
    return config.tla_dtype, config.torch_dtype, 0.0


def _make_inputs(args: Any, dtype_name: str, torch: Any) -> tuple[Any, ...]:
    _, dtype, _ = _set_kernel_config(args.op, dtype_name, args.shape)
    src = torch.arange(VECTOR_ELE, dtype=dtype, device="npu") - (VECTOR_ELE // 2)
    return (src,)


def _expected(op_name: str, inputs: tuple[Any, ...]) -> Any:
    del op_name
    (src,) = inputs
    expected = src.new_zeros(src.shape)
    expected[: VL_ELE // 2] = src[: VL_ELE // 2].abs()
    return expected


def _is_unsupported_case(op_name: str, dtype_name: str) -> bool:
    del op_name
    return dtype_name not in ALL_DTYPES


def _print_skip(op_name: str, dtype_name: str, shape: tuple[int, ...]) -> None:
    del shape
    print(f"skip op={op_name} dtype={dtype_name}: unsupported case")


HARNESS = DirectVectorOpHarness(
    DirectVectorOpConfig(
        description="Compile and run VectorSSA/MaskSSA control-flow carriers.",
        kernel=register_control_flow,
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
        float_dtypes=frozenset({"f32"}),
    )
)


def main() -> int:
    return HARNESS.main()


if __name__ == "__main__":
    raise SystemExit(main())
