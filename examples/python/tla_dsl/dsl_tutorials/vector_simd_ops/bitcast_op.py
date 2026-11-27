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

# VectorSSA.bitcast exercise: a single-block, four-chunk f32 kernel that proves
# the two things a bit reinterpretation has to get right, with an f32 input and
# an f32 output so the reference stays exact.
# A signed and an unsigned 32-bit round trip must both preserve every input
# element. The unsigned route proves that the value remains valid through the
# vector region, not only at the frontend.
#
# The 16-bit routes cover both f16 -> i16 -> f16 and f16 -> bf16 -> f16. The
# output also contains the biased IEEE-754 exponent extracted through an i32
# reinterpretation, the use case that avoids a store and reload through UB.
#
# The output is the exponent plus the two zero-differences, so a wrong
# reinterpretation shows up as a mismatch rather than being masked by the
# exponent term.
VECTOR_ELE = 256
VL_ELE = 64
NUM_CHUNKS = 4
LOOPS = NUM_CHUNKS
ALL_DTYPES = ("f32",)

# IEEE-754 binary32: 23 mantissa bits below an 8-bit biased exponent.
_MANTISSA_BITS = 23
_EXPONENT_MASK = 0xFF

_KERNEL_DTYPE = tla.Float32
_KERNEL_SHAPE = (VECTOR_ELE,)

# Widen-in-place to f32. Used for both the i32 exponent field (a small positive
# integer, so exact in f32) and the f16 round-trip results below, hence the
# neutral name: the trait describes the destination, not one source type.
_TRAIT_WIDEN_F32 = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ZERO,
    sat_mode=tla.params.SatMode.NOSAT,
    round_mode=tla.params.RoundMode.CAST_ROUND,
)

# f32 -> f16 narrow, even lane. The inputs are exact in f16, so the narrow is
# lossless and the 16-bit round trips below stay exact identities.
_TRAIT_F32_TO_F16 = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ZERO,
    sat_mode=tla.params.SatMode.NOSAT,
    round_mode=tla.params.RoundMode.CAST_ROUND,
)


@tla.kernel
def bitcast_op(
    mem_a: tla.Tensor,
    mem_out: tla.Tensor,
) -> None:
    ub_loaded = tla.flag("ub_loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    vec_done = tla.flag("vec_done", tla.pipes.VECTOR, tla.pipes.MTE3)

    a_gm = tla.tile_view(mem_a, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    out_gm = tla.tile_view(mem_out, tla.make_shape(VECTOR_ELE), tla.make_coord(0))

    a_ub = _make_ub_tensor(a_gm)
    out_ub = _make_ub_tensor(out_gm)

    with tla.vector():
        tla.copy(a_ub, a_gm)

        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)
        with tla.vec.func(mode="simd"):
            for i in tla.range(NUM_CHUNKS):
                a_v = _chunk(a_ub, i).load()
                out_chunk = _chunk(out_ub, i)

                int_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Int32)

                # Identity: through i32 and back. Bit-exact, so the difference
                # is exactly zero and stays zero in f32 arithmetic.
                signed_rt = a_v.bitcast(tla.Int32).bitcast(tla.Float32)
                signed_delta = signed_rt - a_v

                # The same identity through an unsigned element type.
                unsigned_rt = a_v.bitcast(tla.UInt32).bitcast(tla.Float32)
                unsigned_delta = unsigned_rt - a_v

                # The 16-bit width class, reached by narrowing to f16 first so
                # the reinterpretation happens on a real 16-bit register. Both
                # f16 -> i16 -> f16 and f16 -> bf16 -> f16 are bit-exact, so
                # widening back to f32 reproduces a_h exactly and the two
                # deltas are zero. bf16 is included because it is the one pair
                # where a same-width reinterpretation changes the *meaning* of
                # the bits rather than just their type.
                a_h = a_v.to(tla.Float16, _TRAIT_F32_TO_F16)
                h_signed_rt = a_h.bitcast(tla.Int16).bitcast(tla.Float16)
                h_bf_rt = a_h.bitcast(tla.BFloat16).bitcast(tla.Float16)
                h_signed_delta = h_signed_rt.to(tla.Float32, _TRAIT_WIDEN_F32) - a_h.to(
                    tla.Float32, _TRAIT_WIDEN_F32
                )
                h_bf_delta = h_bf_rt.to(tla.Float32, _TRAIT_WIDEN_F32) - a_h.to(
                    tla.Float32, _TRAIT_WIDEN_F32
                )

                # The real use: read the biased exponent out of the bit pattern.
                bits = a_v.bitcast(tla.Int32)
                exponent_bits = tla.bitwise_and(
                    tla.shift_right(bits, _MANTISSA_BITS, mask=int_mask),
                    tla.full(_EXPONENT_MASK, tla.Int32),
                )
                exponent_f32 = exponent_bits.to(tla.Float32, _TRAIT_WIDEN_F32)

                out_chunk.store(
                    exponent_f32
                    + signed_delta
                    + unsigned_delta
                    + h_signed_delta
                    + h_bf_delta
                )

        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)

        tla.copy(out_gm, out_ub)

        tla.pipe_barrier(tla.pipes.ALL)


def _make_ub_tensor(like_tensor: Any) -> Any:
    ptr = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    return tla.make_tensor_like(ptr, like_tensor, tla.arch.RowMajor)


def _chunk(tensor: Any, chunk_idx: Any) -> Any:
    return tla.tile_view(
        tensor,
        tla.make_shape(VL_ELE),
        tla.make_coord(chunk_idx),
    )


def _operator_specs() -> dict[str, dict[str, Any]]:
    return {
        "bitcast": {
            # A reinterpretation and an integer exponent are both exact; nothing
            # here is allowed to drift.
            "default_atol": 0.0,
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
    global VL_ELE, LOOPS, _KERNEL_DTYPE
    global _KERNEL_SHAPE
    if op_name not in _operator_specs():
        raise SystemExit("unknown bitcast operator")
    del shape
    config = vector_kernel_config(dtype_name, (VECTOR_ELE,), ALL_DTYPES)
    _KERNEL_SHAPE = (VECTOR_ELE,)
    VL_ELE = config.lanes
    LOOPS = NUM_CHUNKS
    _KERNEL_DTYPE = config.tla_dtype
    return config.tla_dtype, config.torch_dtype, config.default_sentinel


def _make_inputs(args: Any, dtype_name: str, torch: Any) -> tuple[Any, ...]:
    _, _, _ = _set_kernel_config(args.op, dtype_name, args.shape)
    dtype = vector_kernel_config(dtype_name, args.shape, ALL_DTYPES).torch_dtype
    idx = torch.arange(VECTOR_ELE, dtype=torch.float32, device="npu")
    # Spread the values over several binades so the exponent field actually
    # varies, and keep them exactly representable so the reference is exact.
    values = ((idx % 16.0) + 1.0) * torch.exp2((idx % 5.0) - 2.0)
    return (values.to(dtype),)


def _expected(op_name: str, inputs: tuple[Any, ...]) -> Any:
    del op_name
    import torch

    (a,) = inputs
    # The two round trips are identities, so the reference is the exponent
    # alone -- read with torch's own reinterpretation of the same bits.
    bits = a.cpu().view(torch.int32)
    exponent = (bits >> _MANTISSA_BITS) & _EXPONENT_MASK
    return (exponent.to(torch.float32).to(a.device),)


HARNESS = DirectVectorOpHarness(
    DirectVectorOpConfig(
        description="VectorSSA.bitcast: same-width bit reinterpretation on a vector register.",
        kernel=bitcast_op,
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
        output_count=1,
    )
)


def main() -> int:
    return HARNESS.main()


if __name__ == "__main__":
    raise SystemExit(main())
