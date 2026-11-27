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
from catlass.params import MaskLoadParams, MaskStoreParams
from catlass.types import dtype_size_bytes

from vector_op_harness import (
    DirectVectorOpConfig,
    DirectVectorOpHarness,
    vector_kernel_config,
)

# MaskSSA load/store round-trip:
#   create_mask(H) → store(..., MaskStoreParams) → load(MaskLoadParams)
#   → masked store of src
#
# Two fixed A5 widths (same number, different units):
#   VEC_REG_BYTES = 256   # vector register bytes
#   MASK_REG_BITS = 256   # physical MaskReg bits (DIST_NORM spill = 32B)
#
# --dtype is both companion and Mask UB dtype:
#   ELE        = VEC_REG_BYTES / sizeof(dtype)   # logical lanes / !tla.mask<N>
#   MASK_ELEMS = (MASK_REG_BITS/8) / sizeof      # typed UB slots for full physical spill
# Logical N (e.g. 64 for f32) only sizes the predicate SSA; UB buffer covers 256bit.

VEC_REG_BYTES = 256
MASK_REG_BITS = 256
ALL_DTYPES = ("f32", "f16", "i8", "u8")
_UNSIGNED = frozenset({"u8"})

_DTYPE: type[Any] = tla.Float32
_ELE = VEC_REG_BYTES // 4
_MASK_ELEMS = (MASK_REG_BITS // 8) // 4  # 32B / sizeof(f32) = 8
_SHAPE = (_ELE,)


def _apply_dtype(dtype_name: str) -> tuple[type[Any], Any, float | int]:
    global _DTYPE, _ELE, _MASK_ELEMS, _SHAPE
    cfg = vector_kernel_config(dtype_name, None, ALL_DTYPES)
    elem_bytes = dtype_size_bytes(dtype_name)
    _DTYPE = cfg.tla_dtype
    _ELE = VEC_REG_BYTES // elem_bytes
    _MASK_ELEMS = (MASK_REG_BITS // 8) // elem_bytes
    _SHAPE = (_ELE,)
    return cfg.tla_dtype, cfg.torch_dtype, 0.0


_apply_dtype("f32")


@tla.kernel
def load_store_mask(mem_src: tla.Tensor, mem_out: tla.Tensor) -> None:
    ub_loaded = tla.flag("ub_loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    vec_done = tla.flag("vec_done", tla.pipes.VECTOR, tla.pipes.MTE3)

    src_ub = _make_ub_tensor(mem_src)
    out_ub = _make_ub_tensor(mem_out)
    mask_ub = _make_mask_ub_tensor()

    with tla.vector():
        tla.copy(src_ub, mem_src)
        # Masked-out lanes must stay deterministic (0).
        tla.copy(out_ub, mem_out)

        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)
        with tla.vec.func(mode="simd"):
            value = src_ub.load()

            pattern = tla.create_mask(pattern=tla.mask.H, dtype=_DTYPE)
            mask_ub.store(pattern, MaskStoreParams())

            tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
            loaded = mask_ub.load(MaskLoadParams())
            out_ub.store(value, mask=loaded)

        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)

        tla.copy(mem_out, out_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def _make_ub_tensor(like_tensor: Any) -> Any:
    ptr = tla.allocate(_ELE, _DTYPE, tla.AddressSpace.ub, 32)
    return tla.make_tensor_like(ptr, like_tensor, tla.arch.RowMajor)


def _make_mask_ub_tensor() -> Any:
    ptr = tla.allocate(_MASK_ELEMS, _DTYPE, tla.AddressSpace.ub, 32)
    layout = tla.make_layout(
        shape=tla.make_shape(_MASK_ELEMS),
        stride=tla.make_stride(1),
    )
    return tla.make_tensor(ptr, layout)


def _operator_specs() -> dict[str, dict[str, Any]]:
    return {"load_store_mask": {"default_atol": 1e-5}}


def _is_unsupported_case(op_name: str, dtype_name: str) -> bool:
    del op_name
    return dtype_name not in ALL_DTYPES


def _print_skip(op_name: str, dtype_name: str, shape: tuple[int, ...]) -> None:
    del shape
    print(f"skip op={op_name} dtype={dtype_name}: unsupported dtype")


def _set_kernel_config(
    op_name: str, dtype_name: str, shape: tuple[int, ...] | None = None
) -> tuple[type[Any], Any, float | int]:
    del shape
    if op_name not in _operator_specs():
        raise SystemExit("unknown load_store_mask operator")
    if dtype_name not in ALL_DTYPES:
        raise SystemExit(
            f"unsupported dtype={dtype_name!r}; expected one of: "
            f"{', '.join(ALL_DTYPES)}"
        )
    return _apply_dtype(dtype_name)


def _make_inputs(args: Any, dtype_name: str, torch: Any) -> tuple[Any, ...]:
    _, dtype, _ = _set_kernel_config(args.op, dtype_name, args.shape)
    idx = torch.arange(_ELE, dtype=torch.int64)
    # u8: full 0..255 so GM↔UB copy cannot hide behind a signed twin.
    if dtype_name in _UNSIGNED:
        return (((idx % 256).to(dtype)).to("npu"),)
    return ((((idx % 31) - 15).to(dtype)).to("npu"),)


def _expected(op_name: str, inputs: tuple[Any, ...]) -> Any:
    del op_name
    (src,) = inputs
    expected = src.new_zeros(src.shape)
    expected[: _ELE // 2] = src[: _ELE // 2]
    return expected


HARNESS = DirectVectorOpHarness(
    DirectVectorOpConfig(
        description=(
            "Compile and run MaskLoadParams/MaskStoreParams round-trip; "
            "Mask UB uses the same dtype as the companion vector."
        ),
        kernel=load_store_mask,
        all_dtypes=ALL_DTYPES,
        operator_specs=_operator_specs,
        set_kernel_config=_set_kernel_config,
        get_vector_elements=lambda: _ELE,
        get_kernel_shape=lambda: _SHAPE,
        make_inputs=_make_inputs,
        expected=_expected,
        unsupported_case=_is_unsupported_case,
        print_skip=_print_skip,
        script_path=Path(__file__).resolve(),
        float_dtypes=frozenset({"f32", "f16"}),
    )
)


def main() -> int:
    return HARNESS.main()


if __name__ == "__main__":
    raise SystemExit(main())
