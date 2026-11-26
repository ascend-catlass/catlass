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
from catlass.params import (
    CastParams,
    RegSlot,
    RoundMode,
    SatMode,
    StoreDist,
    NormalStoreParams,
)

from vector_op_harness import (
    DirectVectorOpConfig,
    DirectVectorOpHarness,
    vector_kernel_config,
)

VECTOR_ELE = 256
VL_ELE = 64
LOOPS = (VECTOR_ELE + VL_ELE - 1) // VL_ELE
ALL_DTYPES = ("i32", "i16")
OUTPUT_DTYPES = ("i16", "i8")
# AscendC Pack is same-type DistVST. Narrowing is a prior cast; PACK then
# densifies the unpacked slots (MulCast / GenStoreL2 style).
PACK_CONFIGS = {
    "i32": ("i16", StoreDist.DIST_PACK_B32),
    "i16": ("i8", StoreDist.DIST_PACK_B16),
}

# int↔int casts truncate; RegSlot.ZERO selects the even/low part for 2x narrow.
_CAST_NARROW = CastParams(
    reg_slot=RegSlot.ZERO,
    sat_mode=SatMode.NOSAT,
    round_mode=RoundMode.CAST_TRUNC,
)

_KERNEL_DTYPE = tla.Int32
_KERNEL_TORCH_DTYPE = None
_KERNEL_SENTINEL = -7
_OUTPUT_DTYPE = tla.Int16

_OUTPUT_TORCH_DTYPE = None
_STORE_DIST = StoreDist.DIST_PACK_B32
_KERNEL_SHAPE = (VECTOR_ELE,)


def _make_ub_tensor(like_tensor: Any, dtype: type[Any]) -> Any:
    ptr = tla.allocate(VECTOR_ELE, dtype, tla.AddressSpace.ub, 256)
    return tla.make_tensor_like(ptr, like_tensor, tla.arch.RowMajor)


@tla.kernel
def store_pack(
    mem_a: tla.Tensor,
    mem_out: tla.Tensor,
) -> None:
    ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    vec_done = tla.flag("vec_done", tla.arch.VECTOR, tla.arch.MTE3)

    a_gm = tla.tile_view(mem_a, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    # Harness binds mem_out with the input dtype. Recast the pointer so GM/UB
    # output tensors use the narrowed dtype; PACK densifies into that layout.
    out_gm = tla.make_tensor(
        tla.recast_ptr(mem_out.ptr, dtype=_OUTPUT_DTYPE),
        tla.make_layout(
            shape=tla.make_shape(VECTOR_ELE),
            stride=tla.make_stride(1),
        ),
    )

    a_ub = _make_ub_tensor(a_gm, _KERNEL_DTYPE)
    out_ub = _make_ub_tensor(out_gm, _OUTPUT_DTYPE)

    with tla.vector():
        tla.arch.set_ctrl(
            tla.params.CTRLValue(
                saturation_strategy=tla.params.OverrideStrategy.USE_API
            )
        )
        tla.copy(a_ub, a_gm)

        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)
        with tla.vec.func(mode="simd"):
            for i in tla.range(LOOPS):
                a_tile = tla.tile_view(a_ub, tla.make_shape(VL_ELE), tla.make_coord(i))
                out_tile = tla.tile_view(
                    out_ub, tla.make_shape(VL_ELE), tla.make_coord(i)
                )

                a_reg = a_tile.load()
                # Same-type Pack: cast first (unpacked narrow lanes), then densify.
                narrow_reg = a_reg.to(_OUTPUT_DTYPE, _CAST_NARROW)
                out_tile.store(narrow_reg, NormalStoreParams(store_dist=_STORE_DIST))

        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)
        tla.copy(out_gm, out_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def _operator_specs() -> dict[str, dict[str, Any]]:
    return {
        "store_pack": {
            "default_atol": 0,
        }
    }


def _set_kernel_config(
    op_name: str,
    dtype_name: str,
    shape: tuple[int, ...] | None = None,
) -> tuple[type[Any], Any, float | int]:
    global VL_ELE, LOOPS, VECTOR_ELE, _KERNEL_DTYPE, _KERNEL_TORCH_DTYPE
    global _KERNEL_SENTINEL, _OUTPUT_DTYPE, _OUTPUT_TORCH_DTYPE, _STORE_DIST
    global _KERNEL_SHAPE
    specs = _operator_specs()
    if op_name not in specs:
        choices = ", ".join(sorted(specs))
        raise SystemExit(f"unknown op {op_name!r}; expected one of: {choices}")
    config = vector_kernel_config(dtype_name, shape, ALL_DTYPES)
    VECTOR_ELE = config.vector_elements
    _KERNEL_SHAPE = shape if shape is not None else (VECTOR_ELE,)
    VL_ELE = config.lanes
    LOOPS = config.loops
    _KERNEL_DTYPE = config.tla_dtype
    _KERNEL_TORCH_DTYPE = config.torch_dtype
    _KERNEL_SENTINEL = config.default_sentinel
    output_dtype_name, _STORE_DIST = PACK_CONFIGS[dtype_name]
    output_config = vector_kernel_config(
        output_dtype_name, _KERNEL_SHAPE, OUTPUT_DTYPES
    )
    _OUTPUT_DTYPE = output_config.tla_dtype
    _OUTPUT_TORCH_DTYPE = output_config.torch_dtype
    return config.tla_dtype, config.torch_dtype, config.default_sentinel


def _make_inputs(args: Any, dtype_name: str, torch: Any) -> tuple[Any, ...]:
    """Fill input data with common and corner testcases"""
    _, torch_dtype, _ = _set_kernel_config(args.op, dtype_name, args.shape)
    if dtype_name == "i32":
        values = (
            0,
            1,
            -1,
            33,
            67,
            32767,
            -32769,
            0x12345678,
            -0x01234567,
            0x7FFFFFFF,
            -0x80000000,
        )
    else:
        values = (
            0,
            1,
            -1,
            33,
            67,
            127,
            -129,
            0x1234,
            -0x1234,
            32767,
            -32768,
        )
    pattern = torch.tensor(values, dtype=torch_dtype, device="npu")
    repeats = (VECTOR_ELE + len(values) - 1) // len(values)
    a = pattern.repeat(repeats)[:VECTOR_ELE].contiguous()
    return (a,)


def _expected(_op_name: str, inputs: tuple[Any, ...]) -> Any:
    """Host reference for cast-then-PACK densify.

    AscendC path: narrow with trunc cast (unpacked slots), then same-type PACK
    writes contiguous narrow elements. The harness buffer is input-dtype sized;
    little-endian reinterpret packs consecutive narrow values into wide slots::

        input  (i32):  a[2k]        a[2k+1]
                       +-----------+-----------+
                       | cast16    | cast16    |
                       +-----------+-----------+
                            \\        /
        output (i32 slot):  cast(a[2k]) | cast(a[2k+1])<<16
    """
    import torch

    a = inputs[0]
    output_bits = 16 if _OUTPUT_TORCH_DTYPE == torch.int16 else 8
    output_mask = (1 << output_bits) - 1
    narrowed = a.to(dtype=_OUTPUT_TORCH_DTYPE)
    payload = torch.bitwise_and(narrowed.to(dtype=torch.int32), output_mask)

    # Harness owns an input-dtype output buffer; dense narrow store fills the
    # leading half (as wide words) and leaves the rest as sentinel.
    result = torch.full_like(a, _KERNEL_SENTINEL)
    pair_count = VECTOR_ELE // 2
    packed = torch.bitwise_or(
        payload[: pair_count * 2 : 2],
        torch.bitwise_left_shift(payload[1 : pair_count * 2 : 2], output_bits),
    )
    result[:pair_count] = packed.to(dtype=_KERNEL_TORCH_DTYPE)
    if VECTOR_ELE % 2:
        preserved_high = int(_KERNEL_SENTINEL) & ~output_mask
        result[pair_count] = payload[-1] | preserved_high
    return result


HARNESS = DirectVectorOpHarness(
    DirectVectorOpConfig(
        description=(
            "Cast-then-PACK densify: i32->i16/DIST_PACK_B32 and "
            "i16->i8/DIST_PACK_B16 (AscendC same-type StoreAlign)."
        ),
        kernel=store_pack,
        all_dtypes=ALL_DTYPES,
        operator_specs=_operator_specs,
        set_kernel_config=_set_kernel_config,
        get_vector_elements=lambda: VECTOR_ELE,
        get_kernel_shape=lambda: _KERNEL_SHAPE,
        make_inputs=_make_inputs,
        expected=_expected,
        unsupported_case=lambda _op, _dtype: False,
        print_skip=lambda _op, _dtype, _shape: None,
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
