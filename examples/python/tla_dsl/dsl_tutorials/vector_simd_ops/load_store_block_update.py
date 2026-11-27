# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PostMode UPDATE e2e: Block ``tile.load`` / ``tile.store`` (``DATA_BLOCK_COPY``).

``add`` — contiguous Block UPDATE sliding window (``block_stride=1``). Each trip
gathers/scatters 8 consecutive DataBlocks (= one VL of f32/f16), then advances
``repeat_stride`` DataBlocks via the POST pointer. Full trips are unmasked;
when ``VECTOR_ELE % VL != 0``, one epilogue uses ``update_mask(tail)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import catlass.tla as tla
from catlass.params import BlockLoadParams, BlockStoreParams, PostMode

from vector_op_harness import (
    DirectVectorOpConfig,
    DirectVectorOpHarness,
    vector_kernel_config,
)

DATA_BLOCK_BYTES = 32
VECTOR_ELE = 400
VL_ELE = 64
LOOPS = (VECTOR_ELE + VL_ELE - 1) // VL_ELE
ALL_DTYPES = ("f32", "f16")

_KERNEL_DTYPE = tla.Float32
_KERNEL_ELEMENT_BYTES = 4
_KERNEL_SHAPE = (VECTOR_ELE,)
_POST_DB = 8

_LOAD_U = BlockLoadParams(
    block_stride=1,
    post_mode=PostMode.POST_MODE_UPDATE,
    repeat_stride=_POST_DB,
)
_STORE_U = BlockStoreParams(
    block_stride=1,
    post_mode=PostMode.POST_MODE_UPDATE,
    repeat_stride=_POST_DB,
)


def _post_db(vl: int, element_bytes: int) -> int:
    return (vl * element_bytes) // DATA_BLOCK_BYTES


@tla.kernel
def load_store_block_update_op(
    mem_x: tla.Tensor, mem_y: tla.Tensor, mem_z: tla.Tensor
) -> None:
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
            full_loops = VECTOR_ELE // VL_ELE
            tail_elems = VECTOR_ELE % VL_ELE
            x_tile = tla.tile_view(x_ub, tla.make_shape(VL_ELE), tla.make_coord(0))
            y_tile = tla.tile_view(y_ub, tla.make_shape(VL_ELE), tla.make_coord(0))
            z_tile = tla.tile_view(z_ub, tla.make_shape(VL_ELE), tla.make_coord(0))
            for _ in tla.range(full_loops):
                x_reg, x_tile = x_tile.load(_LOAD_U)
                y_reg, y_tile = y_tile.load(_LOAD_U)
                z_tile = z_tile.store(tla.add(x_reg, y_reg), _STORE_U)
            if tail_elems != 0:
                x_tail, x_tile = x_tile.load(_LOAD_U)
                y_tail, y_tile = y_tile.load(_LOAD_U)
                tail, _ = tla.update_mask(tail_elems, dtype=_KERNEL_DTYPE)
                z_tile = z_tile.store(tla.add(x_tail, y_tail), _STORE_U, mask=tail)
        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)
        tla.copy(z_gm, z_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def _operator_specs() -> dict[str, dict[str, Any]]:
    return {"add": {"op": "add", "dtypes": ALL_DTYPES, "default_atol": 1e-4}}


def _set_kernel_config(
    op_name: str, dtype_name: str, shape: tuple[int, ...] | None = None
) -> tuple[type[Any], Any, float | int]:
    global VL_ELE, LOOPS, VECTOR_ELE, _KERNEL_DTYPE, _KERNEL_ELEMENT_BYTES
    global _KERNEL_SHAPE, _LOAD_U, _STORE_U, _POST_DB
    del op_name
    config = vector_kernel_config(dtype_name, shape, ALL_DTYPES)
    VECTOR_ELE = config.vector_elements
    _KERNEL_SHAPE = shape if shape is not None else (VECTOR_ELE,)
    VL_ELE = config.lanes
    LOOPS = config.loops
    _KERNEL_DTYPE = config.tla_dtype
    _KERNEL_ELEMENT_BYTES = config.element_bytes
    _POST_DB = _post_db(VL_ELE, _KERNEL_ELEMENT_BYTES)
    if _POST_DB < 1:
        raise SystemExit(
            f"Block UPDATE needs VL*elem_bytes >= {DATA_BLOCK_BYTES}, "
            f"got VL={VL_ELE} elem_bytes={_KERNEL_ELEMENT_BYTES}"
        )
    _LOAD_U = BlockLoadParams(
        block_stride=1,
        post_mode=PostMode.POST_MODE_UPDATE,
        repeat_stride=_POST_DB,
    )
    _STORE_U = BlockStoreParams(
        block_stride=1,
        post_mode=PostMode.POST_MODE_UPDATE,
        repeat_stride=_POST_DB,
    )
    return config.tla_dtype, config.torch_dtype, config.default_sentinel


def _make_inputs(args: Any, dtype_name: str, torch: Any) -> tuple[Any, Any]:
    _, dtype, _ = _set_kernel_config("add", dtype_name, args.shape)
    x = torch.arange(VECTOR_ELE, dtype=dtype, device="npu") + 1
    y = (torch.arange(VECTOR_ELE, dtype=dtype, device="npu") * 0.5) + 2
    return x, y


def _expected(_op_name: str, inputs: tuple[Any, ...]) -> Any:
    x, y = inputs
    return x + y


HARNESS = DirectVectorOpHarness(
    DirectVectorOpConfig(
        description="Block PostMode UPDATE load/store sliding-window add E2E",
        kernel=load_store_block_update_op,
        all_dtypes=ALL_DTYPES,
        operator_specs=_operator_specs,
        set_kernel_config=_set_kernel_config,
        get_vector_elements=lambda: VECTOR_ELE,
        get_kernel_shape=lambda: _KERNEL_SHAPE,
        make_inputs=_make_inputs,
        expected=_expected,
        unsupported_case=lambda _o, _d: False,
        print_skip=lambda o, d, s: print(f"skip op={o} dtype={d} shape={s}"),
        script_path=Path(__file__).resolve(),
        float_dtypes=frozenset({"f32", "f16"}),
        input_count=2,
    )
)


def main() -> int:
    return HARNESS.main()


if __name__ == "__main__":
    raise SystemExit(main())
