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

import sys
import argparse
from pathlib import Path
from typing import Any

import catlass.tla as tla
from catlass.params import UnalignStoreParams

DEMO_DIR = Path(__file__).resolve().parent

VECTOR_ELE = 128
ELEMENT_BYTES = 4
_REDUCE_OP = tla.ReductionOp.ADD
_BATCH_REDUCE_OPS = (
    tla.ReductionOp.ADD,
    tla.ReductionOp.MAX,
    tla.ReductionOp.MIN,
)


@tla.kernel
def reduction_op(mem_x: tla.Tensor, mem_z: tla.Tensor) -> None:
    """Reduce 128 fp32 → two 1-element results.

    Tile 0 (first 64 ele): result at z coord 0 (aligned) → normal store.
    Tile 1 (last  64 ele): result at z coord 1 (unaligned) → UnalignStoreParams.
    """
    TILE_ELE = 64
    loaded = tla.flag("loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    done = tla.flag("done", tla.pipes.VECTOR, tla.pipes.MTE3)
    x_gm = tla.tile_view(mem_x, tla.make_shape(VECTOR_ELE), tla.make_coord(0))
    z_gm = tla.tile_view(mem_z, tla.make_shape(2), tla.make_coord(0))
    x_ptr = tla.allocate(VECTOR_ELE, tla.Float32, tla.AddressSpace.ub, 256)
    z_ptr = tla.allocate(2, tla.Float32, tla.AddressSpace.ub, 256)
    x_ub = tla.make_tensor_like(x_ptr, x_gm, tla.arch.RowMajor)
    z_ub = tla.make_tensor_like(z_ptr, z_gm, tla.arch.RowMajor)

    with tla.vector():
        tla.copy(x_ub, x_gm)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        with tla.vec.func(mode="simd"):
            # tile 0: first 64 elements, result at z coord 0 (aligned)
            x0 = tla.tile_view(x_ub, tla.make_shape(TILE_ELE), tla.make_coord(0))
            z0 = tla.tile_view(z_ub, tla.make_shape(1), tla.make_coord(0))
            reduce_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            z0.store(x0.load().reduce(_REDUCE_OP, mask=reduce_mask))
            # tile 1: last 64 elements, result at z coord 1 (NOT aligned)
            x1 = tla.tile_view(x_ub, tla.make_shape(TILE_ELE), tla.make_coord(1))
            z1 = tla.tile_view(z_ub, tla.make_shape(1), tla.make_coord(1))
            z1.store(
                x1.load().reduce(_REDUCE_OP, mask=reduce_mask),
                params=UnalignStoreParams(),
            )
        tla.set_flag(done)
        tla.wait_flag(done)
        tla.copy(z_gm, z_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def _batch_reduce_value(value: Any, mask: Any, op: Any) -> Any:
    return value.reduce(op, mask=mask)


@tla.kernel
def reduction_op_batch(mem_x: tla.Tensor, mem_z: tla.Tensor) -> None:
    tile_ele = 64
    block_idx = tla.arch.block_idx()
    loaded = tla.flag("loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    done = tla.flag("done", tla.pipes.VECTOR, tla.pipes.MTE3)
    x_gm = tla.tile_view(mem_x, tla.make_shape(VECTOR_ELE), tla.make_coord(block_idx))
    z_gm = tla.tile_view(mem_z, tla.make_shape(2), tla.make_coord(block_idx))
    x_ptr = tla.allocate(VECTOR_ELE, tla.Float32, tla.AddressSpace.ub, 256)
    z_ptr = tla.allocate(2, tla.Float32, tla.AddressSpace.ub, 256)
    x_ub = tla.make_tensor_like(x_ptr, x_gm, tla.arch.RowMajor)
    z_ub = tla.make_tensor_like(z_ptr, z_gm, tla.arch.RowMajor)
    with tla.vector():
        tla.copy(x_ub, x_gm)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        with tla.vec.func(mode="simd"):
            reduce_mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
            x0 = tla.tile_view(x_ub, tla.make_shape(tile_ele), tla.make_coord(0))
            z0 = tla.tile_view(z_ub, tla.make_shape(1), tla.make_coord(0))
            x1 = tla.tile_view(x_ub, tla.make_shape(tile_ele), tla.make_coord(1))
            z1 = tla.tile_view(z_ub, tla.make_shape(1), tla.make_coord(1))
            if block_idx == 0:
                z0.store(
                    _batch_reduce_value(x0.load(), reduce_mask, _BATCH_REDUCE_OPS[0])
                )
                z1.store(
                    _batch_reduce_value(x1.load(), reduce_mask, _BATCH_REDUCE_OPS[0]),
                    params=UnalignStoreParams(),
                )
            elif block_idx == 1:
                z0.store(
                    _batch_reduce_value(x0.load(), reduce_mask, _BATCH_REDUCE_OPS[1])
                )
                z1.store(
                    _batch_reduce_value(x1.load(), reduce_mask, _BATCH_REDUCE_OPS[1]),
                    params=UnalignStoreParams(),
                )
            else:
                z0.store(
                    _batch_reduce_value(x0.load(), reduce_mask, _BATCH_REDUCE_OPS[2])
                )
                z1.store(
                    _batch_reduce_value(x1.load(), reduce_mask, _BATCH_REDUCE_OPS[2]),
                    params=UnalignStoreParams(),
                )
        tla.set_flag(done)
        tla.wait_flag(done)
        tla.copy(z_gm, z_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def _set_op(op: str) -> None:
    global _REDUCE_OP
    _REDUCE_OP = {
        "add": tla.ReductionOp.ADD,
        "max": tla.ReductionOp.MAX,
        "min": tla.ReductionOp.MIN,
    }[op]


def _runtime_tensor(dev_buf: Any) -> Any:
    return tla.from_dlpack(
        dev_buf.contiguous(),
        layout_tag=tla.arch.RowMajor,
    )


def _expected(op: str, x: Any) -> Any:
    if op == "add":
        return x.sum().reshape(1)
    if op == "max":
        return x.max().reshape(1)
    return x.min().reshape(1)


def _compile(args: argparse.Namespace, *type_args: Any) -> Any:
    return tla.compile(
        reduction_op,
        *type_args,
        options="--npu-arch 3510",
    )


def _compile_batch(args: argparse.Namespace, *type_args: Any) -> Any:
    return tla.compile(
        reduction_op_batch,
        *type_args,
        options="--npu-arch 3510",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("op", choices=("add", "max", "min"), nargs="?")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--dtype", choices=("f32",), default="f32")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch-run", action="store_true")
    args = parser.parse_args()

    if args.batch_run:
        if args.op is not None:
            raise SystemExit("positional op cannot be combined with --batch-run")
        import torch
        import torch_npu

        torch.npu.set_device(args.device)
        tile_ele = 64
        base = torch.linspace(
            -17.0, 46.0, VECTOR_ELE, dtype=torch.float32, device="npu"
        )
        x = torch.cat((base, base, base))
        z = torch.full((6,), -999.0, dtype=torch.float32, device="npu")
        tla_x = _runtime_tensor(x)
        tla_z = _runtime_tensor(z)
        artifact = _compile_batch(args, tla_x, tla_z)
        artifact(tla_x, tla_z, block_num=3)
        torch.npu.synchronize()
        failed = 0
        for index, op_name in enumerate(("add", "max", "min")):
            actual = z[index * 2 : index * 2 + 2]
            expected = torch.cat(
                (
                    _expected(op_name, base[:tile_ele]),
                    _expected(op_name, base[tile_ele:]),
                )
            )
            ok = bool(torch.isclose(actual, expected, rtol=0.0, atol=1e-4).all())
            failed += not ok
            print(
                f"batch_case op={op_name} dtype=f32 passed={ok} "
                f"actual={actual.cpu().tolist()} expected={expected.cpu().tolist()}"
            )
        print(f"batch_kernel_ok={failed == 0} ops=add,max,min blocks=3")
        print(f"kernel.o path={artifact.kernel_binary_path}")
        return 0 if failed == 0 else 1
    if args.op is None:
        raise SystemExit("a positional op is required unless --batch-run is used")
    _set_op(args.op)
    if not args.run:
        raise SystemExit("pass --run")

    import torch
    import torch_npu

    torch.npu.set_device(args.device)

    TILE_ELE = 64
    x = torch.linspace(-17.0, 46.0, VECTOR_ELE, dtype=torch.float32, device="npu")
    z = torch.full((2,), -999.0, dtype=torch.float32, device="npu")
    tla_x = _runtime_tensor(x)
    tla_z = _runtime_tensor(z)
    artifact = _compile(args, tla_x, tla_z)
    artifact(tla_x, tla_z, block_num=1)
    torch.npu.synchronize()

    exp0 = _expected(args.op, x[:TILE_ELE])
    exp1 = _expected(args.op, x[TILE_ELE:])
    z0_cpu = z[0].cpu()
    z1_cpu = z[1].cpu()
    exp0_cpu = exp0.cpu()
    exp1_cpu = exp1.cpu()
    print(f"z[0] actual  = {z0_cpu.item():.6f}")
    print(f"z[0] expected= {exp0_cpu.item():.6f}")
    ok0 = bool(torch.isclose(z[0].reshape(1), exp0, rtol=0.0, atol=1e-4).all())
    print(f"z[1] actual  = {z1_cpu.item():.6f}")
    print(f"z[1] expected= {exp1_cpu.item():.6f}")
    ok1 = bool(torch.isclose(z[1].reshape(1), exp1, rtol=0.0, atol=1e-4).all())
    ok = ok0 and ok1
    print(f"tile0 (coord=0, aligned)   match? {ok0}")
    print(f"tile1 (coord=1, unalign)   match? {ok1}")

    print(f"compile_ok=True host=torch_npu op={args.op} dtype=f32 layout=row")
    print(f"kernel.o path={artifact.kernel_binary_path}")
    print("launch_ok=True")
    print(f"output equals expected reduction? {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
