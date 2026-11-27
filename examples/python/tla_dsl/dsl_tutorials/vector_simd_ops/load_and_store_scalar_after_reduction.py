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

import argparse
from pathlib import Path
from typing import Any

import catlass.tla as tla
import sys

DEMO_DIR = Path(__file__).resolve().parent

VECTOR_ELE = 64
VEC_FUNC_STORE_INDEX = 7
VECTOR_STORE_INDEX = 8


@tla.kernel
def load_and_store_scalar_after_reduction(
    mem_x: tla.Tensor,
    mem_stored: tla.Tensor,
    mem_reduced: tla.Tensor,
) -> None:
    """Exercise UB scalar accesses both inside and outside ``tla.vec.func``."""
    loaded = tla.flag("loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    vec_func_done = tla.flag("vec_func_done", tla.pipes.VECTOR, tla.pipes.SCALAR)
    scalar_to_mte3 = tla.flag("scalar_to_mte3", tla.pipes.SCALAR, tla.pipes.MTE3)

    x_gm = tla.tile_view(mem_x, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    stored_gm = tla.tile_view(mem_stored, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    reduced_gm = tla.tile_view(mem_reduced, tla.make_shape(1), tla.make_coord(0))

    x_ptr = tla.allocate(VECTOR_ELE, tla.Float32, tla.AddressSpace.ub, 256)
    stored_ptr = tla.allocate(VECTOR_ELE, tla.Float32, tla.AddressSpace.ub, 256)
    reduced_ptr = tla.allocate(1, tla.Float32, tla.AddressSpace.ub, 256)
    x_ub = tla.make_tensor_like(x_ptr, x_gm, tla.arch.RowMajor)
    stored_ub = tla.make_tensor_like(stored_ptr, stored_gm, tla.arch.RowMajor)
    reduced_ub = tla.make_tensor_like(reduced_ptr, reduced_gm, tla.arch.RowMajor)

    with tla.vector():
        tla.copy(x_ub, x_gm)
        tla.copy(stored_ub, x_gm)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        with tla.vec.func(mode="simd"):
            x_vec_tile = tla.tile_view(
                x_ub, tla.make_shape(VECTOR_ELE), tla.make_coord(0)
            )
            reduced_vec_tile = tla.tile_view(
                reduced_ub, tla.make_shape(1), tla.make_coord(0)
            )
            stored_vec_tile = tla.tile_view(
                stored_ub, tla.make_shape(VECTOR_ELE), tla.make_coord(0)
            )
            reduce_mask = tla.create_mask(
                pattern=tla.mask.ALL,
                dtype=tla.Float32,
            )
            reduced = x_vec_tile.load().reduce(
                tla.ReductionOp.ADD,
                mask=reduce_mask,
            )
            reduced_vec_tile.store(reduced)

            # Keep the reduction store, scalar load, and scalar store in the
            # same outlined helper. The barrier makes the reduction slot
            # visible to the scalar pipe before it is read.
            tla.local_mem_bar(
                tla.params.MemType.VEC_STORE,
                tla.params.MemType.SCALAR_LOAD,
            )
            vec_func_scalar = reduced_vec_tile[0]
            stored_vec_tile[VEC_FUNC_STORE_INDEX] = vec_func_scalar

        # local_mem_bar only orders accesses inside the helper. The outer
        # scalar pipe must also wait until the complete vector helper,
        # including its scalar store, has finished.
        tla.set_flag(vec_func_done)
        tla.wait_flag(vec_func_done)

        # Exercise a second UB scalar load/store pair directly in tla.vector,
        # outside tla.vec.func.
        reduced_scalar_tile = tla.tile_view(
            reduced_ub, tla.make_shape(1), tla.make_coord(0)
        )
        stored_scalar_tile = tla.tile_view(
            stored_ub, tla.make_shape(VECTOR_ELE), tla.make_coord(0)
        )
        vector_scalar = reduced_scalar_tile[0]
        stored_scalar_tile[VECTOR_STORE_INDEX] = vector_scalar

        # MTE3 must not read stored_ub before either scalar store is visible.
        tla.set_flag(scalar_to_mte3)
        tla.wait_flag(scalar_to_mte3)
        tla.copy(stored_gm, stored_ub)
        tla.copy(reduced_gm, reduced_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def _runtime_tensor(dev_buf: Any) -> Any:
    return tla.from_dlpack(
        dev_buf.contiguous(),
        layout_tag=tla.arch.RowMajor,
    )


def _compile(args: argparse.Namespace, *type_args: Any) -> Any:
    return tla.compile(
        load_and_store_scalar_after_reduction, *type_args, options="--npu-arch 3510"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reduce an f32 UB vector, then load/store the result as scalar SSA "
            "both inside tla.vec.func and directly inside tla.vector."
        )
    )
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    import torch
    import torch_npu

    torch.npu.set_device(args.device)
    x = torch.linspace(-17.0, 46.0, VECTOR_ELE, dtype=torch.float32, device="npu")
    stored_out = torch.full((VECTOR_ELE,), -999.0, dtype=torch.float32, device="npu")
    reduced_out = torch.full((1,), -999.0, dtype=torch.float32, device="npu")

    tla_x = _runtime_tensor(x)
    tla_stored = _runtime_tensor(stored_out)
    tla_reduced = _runtime_tensor(reduced_out)
    artifact = _compile(args, tla_x, tla_stored, tla_reduced)
    artifact(tla_x, tla_stored, tla_reduced, block_num=1)
    torch.npu.synchronize()

    expected_scalar = x.sum()
    expected_stored = x.clone()
    expected_stored[VEC_FUNC_STORE_INDEX] = expected_scalar
    expected_stored[VECTOR_STORE_INDEX] = expected_scalar

    reduced_ok = bool(
        torch.isclose(reduced_out[0], expected_scalar, rtol=0.0, atol=1e-4)
    )
    vec_func_store_ok = bool(
        torch.isclose(
            stored_out[VEC_FUNC_STORE_INDEX],
            expected_scalar,
            rtol=0.0,
            atol=1e-4,
        )
    )
    vector_store_ok = bool(
        torch.isclose(
            stored_out[VECTOR_STORE_INDEX],
            expected_scalar,
            rtol=0.0,
            atol=1e-4,
        )
    )
    stored_ok = bool(
        torch.isclose(stored_out, expected_stored, rtol=0.0, atol=1e-4).all()
    )

    print(
        "compile_ok=True host=torch_npu "
        "op=load_and_store_scalar_after_reduction dtype=f32"
    )
    print(f"kernel.o path={artifact.kernel_binary_path}")
    print("launch_ok=True")
    print(f"reduction UB slot equals expected scalar? {reduced_ok}")
    print(
        "tla.vec.func UB scalar load/store wrote index "
        f"{VEC_FUNC_STORE_INDEX}? {vec_func_store_ok}"
    )
    print(
        "tla.vector UB scalar load/store wrote index "
        f"{VECTOR_STORE_INDEX}? {vector_store_ok}"
    )
    print(f"complete stored output matches expected? {stored_ok}")
    print(f"expected scalar={float(expected_scalar.cpu())}")
    print(f"reduced scalar={float(reduced_out[0].cpu())}")
    print(
        "stored scalars="
        f"({float(stored_out[VEC_FUNC_STORE_INDEX].cpu())}, "
        f"{float(stored_out[VECTOR_STORE_INDEX].cpu())})"
    )
    print(f"stored_out[:9]={stored_out[:9].cpu()}")
    return (
        0 if reduced_ok and vec_func_store_ok and vector_store_ok and stored_ok else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
