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

# This kernel is hard-wired to a fixed control-flow structure of NUM_CHUNKS
# chunks over a fixed VECTOR_ELE elements; it is intentionally not size-generic.
VECTOR_ELE = 400
VL_ELE = 64
NUM_CHUNKS = 7
LOOPS = NUM_CHUNKS
# Only 4-byte dtypes (64 lanes) are supported: the fixed NUM_CHUNKS(=7) structure
# maps to 400 elements exactly when VL_ELE == 64 (7 * 64 = 448 >= 400, one partial
# tail chunk). 2-byte dtypes (128 lanes) would need 4 chunks for 400 elements, so
# the fixed 7 chunks would over-read uninitialized UB.
ALL_DTYPES = ("i32", "f32")

_KERNEL_DTYPE = tla.Float32
_KERNEL_ELEMENT_BYTES = 4
_KERNEL_SHAPE = (VECTOR_ELE,)


@tla.kernel
def multi_binary(
    mem_a: tla.Tensor,
    mem_b: tla.Tensor,
    mem_c: tla.Tensor,
    mem_d: tla.Tensor,
    mem_e: tla.Tensor,
    mem_t: tla.Tensor,
    mem_x: tla.Tensor,
) -> None:
    ub_loaded = tla.flag("ub_loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    vec_done = tla.flag("vec_done", tla.pipes.VECTOR, tla.pipes.MTE3)

    block_idx = tla.arch.block_idx()

    a_gm = tla.tile_view(mem_a, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    b_gm = tla.tile_view(mem_b, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    c_gm = tla.tile_view(mem_c, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    d_gm = tla.tile_view(mem_d, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    e_gm = tla.tile_view(mem_e, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    t_gm = tla.tile_view(mem_t, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    x_gm = tla.tile_view(mem_x, tla.make_shape(VECTOR_ELE), tla.make_coord(0))

    a_ub = _make_ub_tensor(a_gm)
    b_ub = _make_ub_tensor(b_gm)
    c_ub = _make_ub_tensor(c_gm)
    d_ub = _make_ub_tensor(d_gm)
    e_ub = _make_ub_tensor(e_gm)
    t_ub = _make_ub_tensor(t_gm)
    x_ub = _make_ub_tensor(x_gm)

    with tla.vector():
        tla.copy(a_ub, a_gm)
        tla.copy(b_ub, b_gm)
        tla.copy(c_ub, c_gm)
        tla.copy(d_ub, d_gm)
        tla.copy(e_ub, e_gm)

        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)
        with tla.vec.func(mode="simd"):
            # 2-iteration for-loop -> chunks 0, 1. Loop-body values are named
            # locally (lt/li/la/le/lx) so they are not treated as loop-carried
            # values across the dynamic tla.range boundary.
            for i in tla.range(2):
                lt_chunk, l_inter, la_chunk, le_chunk, lx_chunk = _chunk_compute(
                    a_ub, b_ub, c_ub, d_ub, e_ub, t_ub, x_ub, i
                )
                if block_idx == 0:
                    lx_chunk.store(la_chunk.load() + l_inter - le_chunk.load())
                else:
                    lt_chunk.store(la_chunk.load() * l_inter + le_chunk.load())

            # one operation outside the for loop -> chunk 2
            t_chunk, inter, a_chunk, e_chunk, x_chunk = _chunk_compute(
                a_ub, b_ub, c_ub, d_ub, e_ub, t_ub, x_ub, 2
            )
            if block_idx == 0:
                x_chunk.store(a_chunk.load() + inter - e_chunk.load())
            else:
                t_chunk.store(a_chunk.load() * inter + e_chunk.load())

            # 1-iteration for-loop -> chunk 3. Distinct loop-local names from the
            # first loop so neither loop's values are read after its own boundary.
            for j in tla.range(1):
                jt_chunk, j_inter, ja_chunk, je_chunk, jx_chunk = _chunk_compute(
                    a_ub, b_ub, c_ub, d_ub, e_ub, t_ub, x_ub, 3 + j
                )
                if block_idx == 0:
                    jx_chunk.store(ja_chunk.load() + j_inter - je_chunk.load())
                else:
                    jt_chunk.store(ja_chunk.load() * j_inter + je_chunk.load())

            # straight-line code again (outside any for loop) -> chunk 4
            t_chunk, inter, a_chunk, e_chunk, x_chunk = _chunk_compute(
                a_ub, b_ub, c_ub, d_ub, e_ub, t_ub, x_ub, 4
            )
            if block_idx == 0:
                x_chunk.store(a_chunk.load() + inter - e_chunk.load())
            else:
                t_chunk.store(a_chunk.load() * inter + e_chunk.load())

            # straight-line -> chunk 5
            t_chunk, inter, a_chunk, e_chunk, x_chunk = _chunk_compute(
                a_ub, b_ub, c_ub, d_ub, e_ub, t_ub, x_ub, 5
            )
            if block_idx == 0:
                x_chunk.store(a_chunk.load() + inter - e_chunk.load())
            else:
                t_chunk.store(a_chunk.load() * inter + e_chunk.load())

            # straight-line -> chunk 6
            t_chunk, inter, a_chunk, e_chunk, x_chunk = _chunk_compute(
                a_ub, b_ub, c_ub, d_ub, e_ub, t_ub, x_ub, 6
            )
            if block_idx == 0:
                x_chunk.store(a_chunk.load() + inter - e_chunk.load())
            else:
                t_chunk.store(a_chunk.load() * inter + e_chunk.load())

        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)

        if block_idx == 0:
            tla.copy(x_gm, x_ub)
        if block_idx == 1:
            tla.copy(t_gm, t_ub)

        tla.pipe_barrier(tla.pipes.ALL)


def _make_ub_tensor(like_tensor: Any) -> Any:
    alignment = 512 if _KERNEL_ELEMENT_BYTES == 8 else 256
    ptr = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, alignment)
    return tla.make_tensor_like(ptr, like_tensor, tla.arch.RowMajor)


def _chunk(tensor: Any, chunk_idx: Any) -> Any:
    return tla.tile_view(
        tensor,
        tla.make_shape(VL_ELE),
        tla.make_coord(chunk_idx),
    )


# Straight-line (no control flow) per-chunk loads + compute. Returns the tile
# handles and the intermediate so the kernel body can do the store and the
# block_idx-gated store itself: `if`/`for` stays in the AST-traced @tla.kernel
# body because this bare helper executes as ordinary Python staging code.
def _chunk_compute(
    a_ub: Any,
    b_ub: Any,
    c_ub: Any,
    d_ub: Any,
    e_ub: Any,
    t_ub: Any,
    x_ub: Any,
    idx: Any,
) -> tuple[Any, Any, Any, Any, Any]:
    a_chunk = _chunk(a_ub, idx)
    b_chunk = _chunk(b_ub, idx)
    c_chunk = _chunk(c_ub, idx)
    d_chunk = _chunk(d_ub, idx)
    e_chunk = _chunk(e_ub, idx)
    t_chunk = _chunk(t_ub, idx)
    x_chunk = _chunk(x_ub, idx)

    intermediate = b_chunk.load() * c_chunk.load() / d_chunk.load()
    return t_chunk, intermediate, a_chunk, e_chunk, x_chunk


def _operator_specs() -> dict[str, dict[str, Any]]:
    return {
        "multi_binary": {
            "default_atol": 1e-3,
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
    global VL_ELE, LOOPS, _KERNEL_DTYPE, _KERNEL_ELEMENT_BYTES
    global _KERNEL_SHAPE
    if op_name not in _operator_specs():
        raise SystemExit("unknown multi-binary operator")
    # The kernel is hard-wired to a fixed NUM_CHUNKS structure over VECTOR_ELE
    # (400) elements; the requested shape is ignored on purpose so the test always
    # runs at the fixed size. Only the dtype-dependent lane width changes.
    del shape
    config = vector_kernel_config(dtype_name, (VECTOR_ELE,), ALL_DTYPES)
    _KERNEL_SHAPE = (VECTOR_ELE,)
    VL_ELE = config.lanes
    LOOPS = NUM_CHUNKS
    _KERNEL_DTYPE = config.tla_dtype
    _KERNEL_ELEMENT_BYTES = config.element_bytes
    return config.tla_dtype, config.torch_dtype, config.default_sentinel


def _make_inputs(args: Any, dtype_name: str, torch: Any) -> tuple[Any, ...]:
    _, _, _ = _set_kernel_config(args.op, dtype_name, args.shape)
    dtype = vector_kernel_config(dtype_name, args.shape, ALL_DTYPES).torch_dtype
    device = "npu"
    if dtype_name == "i32":
        arange = torch.arange(VECTOR_ELE, dtype=torch.int32, device=device)
        return (
            ((arange % 29) + 5).to(dtype),
            ((arange % 7) + 2).to(dtype),
            ((arange % 5) + 1).to(dtype),
            ((arange % 9) + 1).to(dtype),
            ((arange % 3) + 1).to(dtype),
        )
    idx = torch.arange(VECTOR_ELE, dtype=torch.float32, device=device)
    return (
        ((idx % 17.0) * 0.25 + 1.0).to(dtype),
        ((idx % 7.0) * 0.5 + 1.0).to(dtype),
        ((idx % 5.0) * 0.25 + 1.0).to(dtype),
        ((idx % 9.0) * 0.5 + 1.0).to(dtype),
        ((idx % 3.0) * 0.25).to(dtype),
    )


def _expected(op_name: str, inputs: tuple[Any, ...]) -> tuple[Any, Any]:
    del op_name
    a, b, c, d, e = inputs
    if a.is_floating_point():
        intermediate = ((b * c).to(a.dtype) / d).to(a.dtype)
    else:
        product = (b * c).to(a.dtype)
        intermediate = product.div(d, rounding_mode="trunc").to(a.dtype)

    # Both blocks read the same inputs and the same intermediate (b*c/d); the
    # if/else only changes the operation per block id. Block 0 produces x with
    # one combination, block 1 produces t with a different combination.
    x_result = ((a + intermediate).to(a.dtype) - e).to(a.dtype)  # block 0 -> x
    t_result = ((a * intermediate).to(a.dtype) + e).to(a.dtype)  # block 1 -> t
    return t_result, x_result


HARNESS = DirectVectorOpHarness(
    DirectVectorOpConfig(
        description="Compile and run a multi-op vector binary graph.",
        kernel=multi_binary,
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
        float_dtypes=frozenset({"f32", "f16"}),
        output_count=2,
        # multi_binary gates its two outputs on block_idx (block 0 -> x, block 1
        # -> t), so both blocks must run for the result to be correct.
        launch_blocks=2,
    )
)


def main() -> int:
    return HARNESS.main()


if __name__ == "__main__":
    raise SystemExit(main())
