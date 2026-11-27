# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end sweep for the tile.load distribution modes added by issue 519.

One script, one generic kernel, six operators selected by the positional ``op``
argument (``brc_b16``, ``us_b16``, ``unpack_b16``, ``e2b_b16``, ``e2b_b32``,
``blk``). Each loop iteration loads ``SRC_VIEW_ELE`` source elements from a
tile view and stores the ``VL``-wide distributed register produced by the
selected ``LoadDist`` mode; only the per-op view width, load params, and the
host-side reference transform differ.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import catlass.tla as tla
from catlass.params import LoadDist, NormalLoadParams

from vector_op_harness import (
    DirectVectorOpConfig,
    DirectVectorOpHarness,
    vector_kernel_config,
)

VECTOR_ELE = 512
ALL_DTYPES = ("i16", "i32")

# Per-operator metadata: backend LoadDist enum and the currently supported
# element type (b16 modes are exercised with i16, e2b_b32 with i32).
_OP_DISTS: dict[str, LoadDist] = {
    "brc_b16": LoadDist.DIST_BRC_B16,
    "us_b16": LoadDist.DIST_US_B16,
    "unpack_b16": LoadDist.DIST_UNPACK_B16,
    "e2b_b16": LoadDist.DIST_E2B_B16,
    "e2b_b32": LoadDist.DIST_E2B_B32,
    "blk": LoadDist.DIST_BLK,
}
_OP_DTYPES: dict[str, str] = {
    "brc_b16": "i16",
    "us_b16": "i16",
    "unpack_b16": "i16",
    "e2b_b16": "i16",
    "e2b_b32": "i32",
    "blk": "i16",
}

# Per-op module state refreshed by _set_kernel_config before each compile.
VL_ELE = 128
LOOPS = VECTOR_ELE // VL_ELE
SRC_VIEW_ELE = 1
SRC_ELE = LOOPS
OUT_VALID_ELE = LOOPS * VL_ELE

_KERNEL_DTYPE = tla.Int16
_KERNEL_ELEMENT_BYTES = 2
_KERNEL_SHAPE = (VECTOR_ELE,)
_KERNEL_SENTINEL: int = -7
_LOAD_PARAMS = NormalLoadParams(load_dist=LoadDist.DIST_BRC_B16)


@tla.kernel
def load_dist_ops_kernel(mem_src: tla.Tensor, mem_dst: tla.Tensor) -> None:
    ub_loaded = tla.flag("ub_loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    vec_done = tla.flag("vec_done", tla.pipes.VECTOR, tla.pipes.MTE3)

    src_gm = tla.tile_view(mem_src, tla.make_shape(SRC_ELE), tla.make_coord(0))
    # Copy back only the lanes the kernel writes; host-initialized sentinel
    # remains in the unused tail of mem_dst.
    dst_gm = tla.tile_view(mem_dst, tla.make_shape(OUT_VALID_ELE), tla.make_coord(0))

    src_ub = _make_ub_tensor(src_gm, SRC_ELE)
    dst_ub = _make_ub_tensor(dst_gm, OUT_VALID_ELE)

    with tla.vector():
        tla.copy(src_ub, src_gm)

        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)

        with tla.vec.func(mode="simd"):
            for i in tla.range(LOOPS):
                # SRC_VIEW_ELE-wide view at block i: the selected dist reads
                # this many elements from the base and fills a VL register
                # (brc: 1 element broadcast; us/unpack: VL/2 elements;
                # e2b: one element per 32-byte DataBlock; blk: one DataBlock
                # replicated across all VL/DB DataBlocks).
                src_tile = tla.tile_view(
                    src_ub, tla.make_shape(SRC_VIEW_ELE), tla.make_coord(i)
                )
                dst_tile = tla.tile_view(
                    dst_ub, tla.make_shape(VL_ELE), tla.make_coord(i)
                )
                dist_reg = src_tile.load(_LOAD_PARAMS)
                dst_tile.store(dist_reg)

        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)

        tla.copy(dst_gm, dst_ub)

        tla.pipe_barrier(tla.pipes.ALL)


def _make_ub_tensor(like_tensor: Any, num_ele: int) -> Any:
    ptr = tla.allocate(num_ele, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    return tla.make_tensor_like(ptr, like_tensor, tla.arch.RowMajor)


def _src_view_ele(op_name: str, vl_ele: int) -> int:
    if op_name == "brc_b16":
        return 1
    if op_name in ("us_b16", "unpack_b16"):
        return vl_ele // 2
    if op_name in ("e2b_b16", "e2b_b32"):
        return 8  # VL/DB: one source element per 32-byte DataBlock
    # blk: one whole 32-byte DataBlock per loop.
    return vl_ele // 8


def _operator_specs() -> dict[str, dict[str, Any]]:
    return {
        op_name: {"default_atol": 0.0, "dtypes": (dtype,)}
        for op_name, dtype in _OP_DTYPES.items()
    }


def _is_unsupported_case(op_name: str, dtype_name: str) -> bool:
    required = _OP_DTYPES.get(op_name)
    return required is None or dtype_name != required


def _print_skip(op_name: str, dtype_name: str, shape: tuple[int, ...]) -> None:
    del shape
    print(
        f"skip op={op_name} dtype={dtype_name}: "
        f"{op_name} currently requires {_OP_DTYPES.get(op_name, '?')}"
    )


def _set_kernel_config(
    op_name: str, dtype_name: str, shape: tuple[int, ...] | None = None
) -> tuple[type[Any], Any, float | int]:
    global VL_ELE, LOOPS, SRC_VIEW_ELE, SRC_ELE, OUT_VALID_ELE
    global _KERNEL_DTYPE, _KERNEL_ELEMENT_BYTES, _KERNEL_SHAPE, _KERNEL_SENTINEL
    global _LOAD_PARAMS
    if op_name not in _OP_DISTS:
        raise SystemExit(f"unknown load_dist operator {op_name!r}")

    del shape
    required = _OP_DTYPES[op_name]
    if dtype_name != required:
        raise SystemExit(f"{op_name} currently requires {required}, got {dtype_name}")
    config = vector_kernel_config(dtype_name, (VECTOR_ELE,), (required,))
    VL_ELE = config.lanes
    LOOPS = VECTOR_ELE // VL_ELE
    SRC_VIEW_ELE = _src_view_ele(op_name, VL_ELE)
    SRC_ELE = LOOPS * SRC_VIEW_ELE
    OUT_VALID_ELE = LOOPS * VL_ELE
    _KERNEL_SHAPE = (VECTOR_ELE,)
    _KERNEL_DTYPE = config.tla_dtype
    _KERNEL_ELEMENT_BYTES = config.element_bytes
    _KERNEL_SENTINEL = config.default_sentinel
    _LOAD_PARAMS = NormalLoadParams(load_dist=_OP_DISTS[op_name])
    return config.tla_dtype, config.torch_dtype, config.default_sentinel


def _make_inputs(args: Any, dtype_name: str, torch: Any) -> tuple[Any, ...]:
    _, dtype, _ = _set_kernel_config(args.op, dtype_name, args.shape)
    device = "npu"
    # Distinct small values so the distributed pattern is easy to verify.
    src = torch.arange(SRC_ELE, dtype=torch.float32, device=device).to(dtype)
    return (src,)


def _expected(op_name: str, inputs: tuple[Any, ...]) -> Any:
    import torch

    (src,) = inputs
    # Kernel writes LOOPS*VL lanes and leaves the host sentinel in the rest.
    dst = torch.full(
        (VECTOR_ELE,), _KERNEL_SENTINEL, dtype=src.dtype, device=src.device
    )
    if op_name == "brc_b16":
        # Each source element is broadcast across a full VL register.
        distributed = torch.repeat_interleave(src, VL_ELE)
    elif op_name == "us_b16":
        # Each source element is repeated twice.
        distributed = torch.repeat_interleave(src, 2)
    elif op_name == "unpack_b16":
        # UNPK_B16 result: [e0, 0, e1, 0, ...] — each element followed by a zero.
        zeros = torch.zeros_like(src)
        distributed = torch.stack((src, zeros), dim=1).flatten()
    elif op_name in ("e2b_b16", "e2b_b32"):
        # Element j of each 8-element group is replicated across all lanes of
        # DataBlock j, i.e. each source element is repeated VL/DB times.
        distributed = torch.repeat_interleave(src, VL_ELE // 8)
    else:
        # blk: each loop reads one 32-byte DataBlock and replicates it across
        # all 8 DataBlocks of the VL register.
        rows = src.view(LOOPS, VL_ELE // 8)
        distributed = rows.repeat_interleave(8, dim=0).reshape(-1)
    dst[:OUT_VALID_ELE] = distributed[:OUT_VALID_ELE]
    return (dst,)


HARNESS = DirectVectorOpHarness(
    DirectVectorOpConfig(
        description=(
            "Compile and run tile.load distribution modes "
            "(brc_b16/us_b16/unpack_b16/e2b_b16/e2b_b32/blk)."
        ),
        kernel=load_dist_ops_kernel,
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
        input_count=1,
        output_count=1,
    )
)


def main() -> int:
    return HARNESS.main()


if __name__ == "__main__":
    raise SystemExit(main())
