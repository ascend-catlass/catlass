# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""tla.unsqueeze end-to-end sample: mask exclusive prefix indices.

Battery case: vector-unsqueeze-op. The result of M4 (lanes 0, 4, 8, ...) is
ceil(lane / 4); all six supported integer dtypes are checked.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

# On-device sample under python/tla_dsl/tests/end_to_end; battery case
# vector-unsqueeze-op. Reuse the vector_simd_ops harness.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_VECTOR_SIMD_OPS = str(
    _REPO_ROOT / "examples" / "python" / "tla_dsl" / "dsl_tutorials" / "vector_simd_ops"
)

if _VECTOR_SIMD_OPS not in sys.path:
    # append (not insert(0)) to avoid shadowing installed packages (G.PSL.03).
    sys.path.append(_VECTOR_SIMD_OPS)

import catlass.tla as tla

from vector_op_harness import (
    DirectVectorOpConfig,
    DirectVectorOpHarness,
    vector_kernel_config,
)

VECTOR_ELE = 64
VL_ELE = 64
LOOPS = 1
ALL_DTYPES = ("i8", "i16", "i32", "u8", "u16", "u32")

_KERNEL_DTYPE = tla.Int8
_KERNEL_SHAPE = (VECTOR_ELE,)
_MASK_PATTERN = tla.mask.M4
_TORCH_DTYPE: Any = None


@tla.kernel
def unsqueeze_op(mem_dst: tla.Tensor) -> None:
    vec_done = tla.flag("vec_done", tla.pipes.VECTOR, tla.pipes.MTE3)

    dst_gm = tla.tile_view(mem_dst, tla.make_shape(VECTOR_ELE), tla.make_coord(0))

    dst_ub_ptr = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    dst_ub = tla.make_tensor_like(dst_ub_ptr, dst_gm, tla.arch.RowMajor)

    with tla.vector():
        with tla.vec.func(mode="simd"):
            remaining = VECTOR_ELE
            for i in tla.range(LOOPS):
                dst_tile = tla.tile_view(
                    dst_ub, tla.make_shape(VL_ELE), tla.make_coord(i)
                )
                tail, remaining = tla.update_mask(remaining, dtype=_KERNEL_DTYPE)
                m4 = tla.create_mask(pattern=_MASK_PATTERN, dtype=_KERNEL_DTYPE)
                active = tla.bitwise_and(m4, tail)
                dst_tile.store(tla.unsqueeze(active, _KERNEL_DTYPE), mask=tail)

        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)

        tla.copy(dst_gm, dst_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def _operator_specs() -> dict[str, dict[str, Any]]:
    return {
        "unsqueeze": {
            "default_atol": 0,
        },
    }


def _is_unsupported_case(op_name: str, dtype_name: str) -> bool:
    del op_name, dtype_name
    return False


def _print_skip(op_name: str, dtype_name: str, shape: tuple[int, ...]) -> None:
    del shape
    print(f"skip op={op_name} dtype={dtype_name}: unsupported case")


def _set_kernel_config(
    op_name: str, dtype_name: str, shape: tuple[int, ...] | None = None
) -> tuple[type[Any], Any, float | int]:
    global VL_ELE, LOOPS, VECTOR_ELE, _KERNEL_DTYPE, _KERNEL_SHAPE, _TORCH_DTYPE
    if op_name not in _operator_specs():
        raise SystemExit("unknown unsqueeze operator")
    config = vector_kernel_config(dtype_name, shape, ALL_DTYPES)
    VECTOR_ELE = config.vector_elements
    _KERNEL_SHAPE = shape if shape is not None else (VECTOR_ELE,)
    VL_ELE = config.lanes
    LOOPS = max(1, config.loops)
    _KERNEL_DTYPE = config.tla_dtype
    _TORCH_DTYPE = config.torch_dtype
    return config.tla_dtype, config.torch_dtype, config.default_sentinel


def _make_inputs(args: Any, dtype_name: str, torch: Any) -> tuple[Any, ...]:
    del args, dtype_name, torch
    # tla.unsqueeze has no vector operand: the mask alone defines the result.
    return ()


def _expected(op_name: str, inputs: tuple[Any, ...]) -> Any:
    del op_name, inputs
    import torch

    # Exclusive prefix over the M4 mask (active lanes 0, 4, 8, ...):
    # lane i holds the number of active bits at lanes < i, i.e. ceil(i / 4).
    out = torch.zeros(VECTOR_ELE, dtype=torch.int64)
    for i in range(LOOPS):
        start = i * VL_ELE
        if start >= VECTOR_ELE:
            break
        end = min(start + VL_ELE, VECTOR_ELE)
        lane = torch.arange(end - start)
        out[start:end] = (lane + 3) // 4
    return out.to(_TORCH_DTYPE)


HARNESS = DirectVectorOpHarness(
    DirectVectorOpConfig(
        description="Compile and run a vector unsqueeze kernel (M4 mask).",
        kernel=unsqueeze_op,
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
        float_dtypes=frozenset(),
        input_count=0,
    )
)


def main() -> int:
    return HARNESS.main()


if __name__ == "__main__":
    raise SystemExit(main())
