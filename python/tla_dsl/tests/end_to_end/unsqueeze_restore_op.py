# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""tla.unsqueeze + tla.gather tail restore end-to-end sample.

Battery case: vector-unsqueeze-restore. 96 Int32 elements: the first 64 are a
full register, the last 32 are a cropped tail tile. squeeze -> unsqueeze ->
gather restores the input; the tail source/result keep 32 valid lanes while
the unsqueeze index covers the full 64-lane register.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

# On-device sample under python/tla_dsl/tests/end_to_end; battery case
# vector-unsqueeze-restore. Reuse the vector_simd_ops harness.
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

TOTAL_ELE = 96
BLOCK_ELE = 64
TAIL_ELE = 32
TAIL_COORD = 2
ALL_DTYPES = ("i32",)

_KERNEL_DTYPE = tla.Int32
_INDEX_DTYPE = tla.UInt32


@tla.kernel
def unsqueeze_restore_op(mem_src: tla.Tensor, mem_dst: tla.Tensor) -> None:
    ub_loaded = tla.flag("ub_loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    vec_done = tla.flag("vec_done", tla.pipes.VECTOR, tla.pipes.MTE3)

    src_gm = tla.tile_view(mem_src, tla.make_shape(TOTAL_ELE), tla.make_coord(0))
    dst_gm = tla.tile_view(mem_dst, tla.make_shape(TOTAL_ELE), tla.make_coord(0))

    src_ub_ptr = tla.allocate(TOTAL_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    dst_ub_ptr = tla.allocate(TOTAL_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    src_ub = tla.make_tensor_like(src_ub_ptr, src_gm, tla.arch.RowMajor)
    dst_ub = tla.make_tensor_like(dst_ub_ptr, dst_gm, tla.arch.RowMajor)

    with tla.vector():
        tla.copy(src_ub, src_gm)

        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)
        with tla.vec.func(mode="simd"):
            # Full block: 64 lanes.
            src0 = tla.tile_view(src_ub, tla.make_shape(BLOCK_ELE), tla.make_coord(0))
            dst0 = tla.tile_view(dst_ub, tla.make_shape(BLOCK_ELE), tla.make_coord(0))
            values0 = src0.load()
            keep0, _ = tla.update_mask(BLOCK_ELE, dtype=_KERNEL_DTYPE)
            packed0 = tla.squeeze(values0, keep0)
            index0 = tla.unsqueeze(keep0, _INDEX_DTYPE)
            dst0.store(tla.gather(packed0, index0), mask=keep0)

            # Tail block: 32 valid lanes, full-register (64-lane) index.
            src1 = tla.tile_view(src_ub, tla.make_shape(TAIL_ELE), tla.make_coord(TAIL_COORD))
            dst1 = tla.tile_view(dst_ub, tla.make_shape(TAIL_ELE), tla.make_coord(TAIL_COORD))
            values1 = src1.load()
            keep1, _ = tla.update_mask(TAIL_ELE, dtype=_KERNEL_DTYPE)
            packed1 = tla.squeeze(values1, keep1)
            index1 = tla.unsqueeze(keep1, _INDEX_DTYPE)
            dst1.store(tla.gather(packed1, index1), mask=keep1)

        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)

        tla.copy(dst_gm, dst_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def _operator_specs() -> dict[str, dict[str, Any]]:
    return {
        "restore": {
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
    del shape
    if op_name not in _operator_specs():
        raise SystemExit("unknown unsqueeze restore operator")
    config = vector_kernel_config(dtype_name, (TOTAL_ELE,), ALL_DTYPES)
    return config.tla_dtype, config.torch_dtype, config.default_sentinel


def _make_inputs(args: Any, dtype_name: str, torch: Any) -> tuple[Any, ...]:
    del args, dtype_name
    idx = torch.arange(TOTAL_ELE, dtype=torch.int64)
    return ((idx % 50) - 20).to(torch.int32).to("npu"),


def _expected(op_name: str, inputs: tuple[Any, ...]) -> Any:
    del op_name
    return inputs[0].detach().cpu().clone()


HARNESS = DirectVectorOpHarness(
    DirectVectorOpConfig(
        description="Restore a squeezed register with unsqueeze + gather on a 96-element tail.",
        kernel=unsqueeze_restore_op,
        all_dtypes=ALL_DTYPES,
        operator_specs=_operator_specs,
        set_kernel_config=_set_kernel_config,
        get_vector_elements=lambda: TOTAL_ELE,
        get_kernel_shape=lambda: (TOTAL_ELE,),
        make_inputs=_make_inputs,
        expected=_expected,
        unsupported_case=_is_unsupported_case,
        print_skip=_print_skip,
        script_path=Path(__file__).resolve(),
        float_dtypes=frozenset(),
        input_count=1,
    )
)


def main() -> int:
    return HARNESS.main()


if __name__ == "__main__":
    raise SystemExit(main())
