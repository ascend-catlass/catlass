# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""VectorSSA.to RoundMode CAST_NONE / CAST_RINT / CAST_ODD end-to-end sample.

Coverage beyond CAST_ROUND/FLOOR/CEIL/TRUNC:
  CAST_RINT  : f32 -> i32, ties to even
  CAST_NONE  : must be explicit; lossy f32 -> i32 equals RINT; lossless
               f16 -> f32 drops the round operand (identity for exact f16)
  CAST_ODD   : f32 -> f16 (hardware supports ODD on this narrow; exact-f16
               inputs make it a no-op so the delta is zero)

CAST_ROUND is also computed so the RINT/ROUND tie split (2.5 -> 2 vs 3) is
visible in the reference. Battery case: vector-cast-round-modes.

f32 output packing: rint result, plus a zero delta for lossy NONE vs RINT,
plus 1e-3 * ROUND, plus zero deltas for f32->f16 ODD and lossless NONE widen
when inputs are exact f16 values.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

# On-device sample under python/tla_dsl/tests/end_to_end; battery case
# vector-cast-round-modes. Reuse the vector_simd_ops harness.
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

VECTOR_ELE = 256
VL_ELE = 64
NUM_CHUNKS = 4
LOOPS = NUM_CHUNKS
ALL_DTYPES = ("f32",)

_KERNEL_DTYPE = tla.Float32
_KERNEL_SHAPE = (VECTOR_ELE,)

_TRAIT_RINT = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ZERO,
    sat_mode=tla.params.SatMode.NOSAT,
    round_mode=tla.params.RoundMode.CAST_RINT,
)
_TRAIT_NONE = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ZERO,
    sat_mode=tla.params.SatMode.NOSAT,
    round_mode=tla.params.RoundMode.CAST_NONE,
)
_TRAIT_ROUND = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ZERO,
    sat_mode=tla.params.SatMode.NOSAT,
    round_mode=tla.params.RoundMode.CAST_ROUND,
)
_TRAIT_ODD_F16 = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ZERO,
    sat_mode=tla.params.SatMode.NOSAT,
    round_mode=tla.params.RoundMode.CAST_ODD,
)
_TRAIT_NARROW_F16 = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ZERO,
    sat_mode=tla.params.SatMode.NOSAT,
    round_mode=tla.params.RoundMode.CAST_ROUND,
)
# int -> float widen for packing integer cast results back into the f32 output.
_TRAIT_I32_TO_F32 = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ZERO,
    sat_mode=tla.params.SatMode.NOSAT,
    round_mode=tla.params.RoundMode.CAST_ROUND,
)
# Lossless f16 -> f32: CAST_NONE must be explicit; lowering drops rnd on vextf.
_TRAIT_F16_TO_F32_NONE = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ZERO,
    sat_mode=tla.params.SatMode.NOSAT,
    round_mode=tla.params.RoundMode.CAST_NONE,
)


@tla.kernel
def cast_round_modes_op(
    mem_a: tla.Tensor,
    mem_out: tla.Tensor,
) -> None:
    ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    vec_done = tla.flag("vec_done", tla.arch.VECTOR, tla.arch.MTE3)

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

                rint_i = a_v.to(tla.Int32, _TRAIT_RINT).to(tla.Float32, _TRAIT_I32_TO_F32)
                none_i = a_v.to(tla.Int32, _TRAIT_NONE).to(tla.Float32, _TRAIT_I32_TO_F32)
                round_i = a_v.to(tla.Int32, _TRAIT_ROUND).to(
                    tla.Float32, _TRAIT_I32_TO_F32
                )

                # Lossless CAST_NONE widen, plus CAST_ODD on f32->f16 (exact
                # inputs keep ODD a no-op so both deltas stay zero).
                a_h = a_v.to(tla.Float16, _TRAIT_NARROW_F16)
                a_h_odd = a_v.to(tla.Float16, _TRAIT_ODD_F16)
                wide_none = a_h.to(tla.Float32, _TRAIT_F16_TO_F32_NONE)
                narrow_f32 = a_h.to(tla.Float32, _TRAIT_F16_TO_F32_NONE)
                odd_f32 = a_h_odd.to(tla.Float32, _TRAIT_F16_TO_F32_NONE)

                out_chunk.store(
                    rint_i
                    + (none_i - rint_i)
                    + round_i * tla.Float32(1.0e-3)
                    + (odd_f32 - narrow_f32)
                    + (wide_none - narrow_f32)
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


def _round_away(value: float) -> float:
    if value >= 0.0:
        return float(math.floor(value + 0.5))
    return float(math.ceil(value - 0.5))


def _operator_specs() -> dict[str, dict[str, Any]]:
    return {
        "cast_round_modes": {
            # Integer cast results packed back to f32 are exact here.
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
        raise SystemExit("unknown cast_round_modes operator")
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
    # Mix exact integers, positive/negative .5 ties, and exact f16 fractions so
    # RINT/ROUND/ODD diverge and the lossless NONE widen stays bit-exact.
    pattern = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            3.5,
            -0.5,
            -1.0,
            -1.5,
            -2.5,
            4.25,
            -4.25,
            8.0,
            -8.0,
        ],
        dtype=torch.float32,
        device="npu",
    )
    idx = torch.arange(VECTOR_ELE, device="npu")
    values = pattern[idx % pattern.numel()]
    return (values.to(dtype),)


def _expected(op_name: str, inputs: tuple[Any, ...]) -> Any:
    del op_name
    import torch

    (a,) = inputs
    host = a.detach().cpu().to(torch.float32)
    rint_vals = []
    round_vals = []
    for value in host.tolist():
        rint_vals.append(float(round(value)))
        round_vals.append(_round_away(value))
    rint_t = torch.tensor(rint_vals, dtype=torch.float32)
    round_t = torch.tensor(round_vals, dtype=torch.float32)
    # Lossy NONE and ODD/NONE widen deltas are zero for these inputs.
    out = rint_t + round_t * 1.0e-3
    return (out.to(a.device),)


HARNESS = DirectVectorOpHarness(
    DirectVectorOpConfig(
        description=(
            "VectorSSA.to RoundMode CAST_NONE/CAST_RINT/CAST_ODD: lossy NONE==RINT, "
            "lossless NONE widen, and f32->f16 ODD."
        ),
        kernel=cast_round_modes_op,
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
