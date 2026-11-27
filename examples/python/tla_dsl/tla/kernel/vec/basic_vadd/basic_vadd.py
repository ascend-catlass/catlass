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
from pathlib import Path

_DSL_EXAMPLE_PATH = str(
    next(p for p in Path(__file__).resolve().parents if (p / "helpers").is_dir())
)

if _DSL_EXAMPLE_PATH not in sys.path:
    sys.path.append(_DSL_EXAMPLE_PATH)

import argparse

import catlass.tla as tla
from catlass.tla.runtime import from_dlpack

# VECTOR_ELE is the static UB allocation upper bound (compile-time capacity).
# GM extents come from mark_compact_shape_dynamic host tensors (no Int32 n_ele).
VECTOR_ELE = 400
VL_ELE = 64
_KERNEL_DTYPE = tla.Float32

# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


@tla.kernel
def basic_vadd(gm_a: tla.Tensor, gm_b: tla.Tensor, gm_c: tla.Tensor) -> None:
    n_ele = gm_a.origin_shape[0]
    ub_loaded = tla.flag("ub_loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    vec_done = tla.flag("vec_done", tla.pipes.VECTOR, tla.pipes.MTE3)

    ub_ptr_a = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    ub_ptr_b = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    ub_ptr_c = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)

    ub_a = tla.make_tensor_like(ub_ptr_a, gm_a, tla.arch.RowMajor)
    ub_b = tla.make_tensor_like(ub_ptr_b, gm_b, tla.arch.RowMajor)
    ub_c = tla.make_tensor_like(ub_ptr_c, gm_c, tla.arch.RowMajor)

    with tla.vector():
        tla.copy(ub_a, gm_a)
        tla.copy(ub_b, gm_b)

        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)
        with tla.vec.func(mode="simd"):
            for i in tla.range((n_ele + VL_ELE - 1) // VL_ELE):
                ub_vl_a = tla.tile_view(ub_a, tla.make_shape(VL_ELE), tla.make_coord(i))
                ub_vl_b = tla.tile_view(ub_b, tla.make_shape(VL_ELE), tla.make_coord(i))
                ub_vl_c = tla.tile_view(ub_c, tla.make_shape(VL_ELE), tla.make_coord(i))

                reg_a = ub_vl_a.load()
                reg_b = ub_vl_b.load()
                reg_c = tla.add(reg_a, reg_b)
                ub_vl_c.store(reg_c)

        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)

        tla.copy(gm_c, ub_c)
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def basic_vadd_mutex(gm_a: tla.Tensor, gm_b: tla.Tensor, gm_c: tla.Tensor) -> None:
    n_ele = gm_a.origin_shape[0]
    mutex_ub_a = tla.mutex(resource="ub_a", id=0)
    mutex_ub_b = tla.mutex(resource="ub_b", id=1)
    mutex_ub_c = tla.mutex(resource="ub_c", id=2)

    ub_ptr_a = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    ub_ptr_b = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    ub_ptr_c = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)

    ub_a = tla.make_tensor_like(ub_ptr_a, gm_a, tla.arch.RowMajor)
    ub_b = tla.make_tensor_like(ub_ptr_b, gm_b, tla.arch.RowMajor)
    ub_c = tla.make_tensor_like(ub_ptr_c, gm_c, tla.arch.RowMajor)

    with tla.vector():
        mutex_ub_a.lock(pipe=tla.pipes.MTE2)
        tla.copy(ub_a, gm_a)
        mutex_ub_a.unlock(pipe=tla.pipes.MTE2)

        mutex_ub_b.lock(pipe=tla.pipes.MTE2)
        tla.copy(ub_b, gm_b)
        mutex_ub_b.unlock(pipe=tla.pipes.MTE2)

        mutex_ub_a.lock(pipe=tla.pipes.VECTOR)
        mutex_ub_b.lock(pipe=tla.pipes.VECTOR)
        mutex_ub_c.lock(pipe=tla.pipes.VECTOR)
        with tla.vec.func(mode="simd"):
            for i in tla.range((n_ele + VL_ELE - 1) // VL_ELE):
                ub_vl_a = tla.tile_view(ub_a, tla.make_shape(VL_ELE), tla.make_coord(i))
                ub_vl_b = tla.tile_view(ub_b, tla.make_shape(VL_ELE), tla.make_coord(i))
                ub_vl_c = tla.tile_view(ub_c, tla.make_shape(VL_ELE), tla.make_coord(i))

                reg_a = ub_vl_a.load()
                reg_b = ub_vl_b.load()
                reg_c = tla.add(reg_a, reg_b)
                ub_vl_c.store(reg_c)
        mutex_ub_c.unlock(pipe=tla.pipes.VECTOR)
        mutex_ub_b.unlock(pipe=tla.pipes.VECTOR)
        mutex_ub_a.unlock(pipe=tla.pipes.VECTOR)

        mutex_ub_c.lock(pipe=tla.pipes.MTE3)
        tla.copy(gm_c, ub_c)
        mutex_ub_c.unlock(pipe=tla.pipes.MTE3)
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def basic_vadd_mutex_with(gm_a: tla.Tensor, gm_b: tla.Tensor, gm_c: tla.Tensor) -> None:
    n_ele = gm_a.origin_shape[0]
    mutex_ub_a = tla.mutex(resource="ub_a", id=0)
    mutex_ub_b = tla.mutex(resource="ub_b", id=1)
    mutex_ub_c = tla.mutex(resource="ub_c", id=2)

    ub_ptr_a = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    ub_ptr_b = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    ub_ptr_c = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)

    ub_a = tla.make_tensor_like(ub_ptr_a, gm_a, tla.arch.RowMajor)
    ub_b = tla.make_tensor_like(ub_ptr_b, gm_b, tla.arch.RowMajor)
    ub_c = tla.make_tensor_like(ub_ptr_c, gm_c, tla.arch.RowMajor)

    with tla.vector():
        with tla.mutex_guard(mutex_ub_a):
            tla.copy(ub_a, gm_a)

        with tla.mutex_guard(mutex_ub_b):
            tla.copy(ub_b, gm_b)

        with tla.mutex_guard(mutex_ub_a, mutex_ub_b, mutex_ub_c):
            with tla.vec.func(mode="simd"):
                for i in tla.range((n_ele + VL_ELE - 1) // VL_ELE):
                    ub_vl_a = tla.tile_view(
                        ub_a, tla.make_shape(VL_ELE), tla.make_coord(i)
                    )
                    ub_vl_b = tla.tile_view(
                        ub_b, tla.make_shape(VL_ELE), tla.make_coord(i)
                    )
                    ub_vl_c = tla.tile_view(
                        ub_c, tla.make_shape(VL_ELE), tla.make_coord(i)
                    )

                    reg_a = ub_vl_a.load()
                    reg_b = ub_vl_b.load()
                    reg_c = tla.add(reg_a, reg_b)
                    ub_vl_c.store(reg_c)

        with tla.mutex_guard(mutex_ub_c):
            tla.copy(gm_c, ub_c)
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def basic_vadd_atomic_add(gm_a: tla.Tensor, gm_b: tla.Tensor, gm_c: tla.Tensor) -> None:
    """C = A + B via plain A store then atomic B add (single AIV block)."""
    ub_loaded = tla.flag("ub_loaded", tla.pipes.MTE2, tla.pipes.MTE3)

    ub_ptr_a = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    ub_ptr_b = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)

    ub_a = tla.make_tensor_like(ub_ptr_a, gm_a, tla.arch.RowMajor)
    ub_b = tla.make_tensor_like(ub_ptr_b, gm_b, tla.arch.RowMajor)

    with tla.vector():
        # To avoid possible race condition since every
        # launched block sees the same GM tiles,
        # Restrict this work to only one block.
        if tla.arch.block_idx() == 0:
            tla.copy(ub_a, gm_a)
            tla.copy(ub_b, gm_b)

            tla.set_flag(ub_loaded)
            tla.wait_flag(ub_loaded)

            # C = A (plain copy, overwrite c on GM)
            tla.copy(gm_c, ub_a)
            tla.pipe_barrier(tla.pipes.MTE3)

            tla.copy(
                gm_c,
                ub_b,
                tla.params.CopyUbToGmParams(atomic_mode=tla.params.AtomicMode.ADD),
            )
            tla.pipe_barrier(tla.pipes.MTE3)
        tla.pipe_barrier(tla.pipes.ALL)


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------


def golden(a, b):
    return a + b


def run(args: argparse.Namespace) -> int:
    import torch
    import torch_npu

    from helpers import (
        get_block_num,
        compare,
    )

    mod = sys.modules[__name__]
    dtype_name = args.dtype
    n_ele = int(args.n)
    if n_ele <= 0 or n_ele > VECTOR_ELE:
        raise SystemExit(f"--n={n_ele} out of range [1, {VECTOR_ELE}]")

    tla_of = {
        "f32": tla.Float32,
        "f16": tla.Float16,
        "i16": tla.Int16,
        "i32": tla.Int32,
        "i8": tla.Int8,
    }
    torch_of = {
        "f32": torch.float32,
        "f16": torch.float16,
        "i16": torch.int16,
        "i32": torch.int32,
        "i8": torch.int8,
    }
    vl_of = {"f32": 64, "f16": 128, "i16": 128, "i32": 64, "i8": 256}
    sentinel_of = {"f32": -7.0, "f16": -7.0, "i16": -7, "i32": -7, "i8": -101}

    mod.VL_ELE = vl_of[dtype_name]
    mod._KERNEL_DTYPE = tla_of[dtype_name]
    torch_dtype = torch_of[dtype_name]
    sentinel = sentinel_of[dtype_name]

    def create_tla_tensor(dev_buf):
        return from_dlpack(
            dev_buf.contiguous(), layout_tag=tla.arch.RowMajor
        ).mark_compact_shape_dynamic(0)

    if args.use_mutex:
        kernel = basic_vadd_mutex
    elif args.use_mutex_with:
        kernel = basic_vadd_mutex_with
    elif args.use_atomic_add:
        kernel = basic_vadd_atomic_add
    else:
        kernel = basic_vadd

    torch.npu.set_device(args.device)
    # Pure AIV kernel: default (-1) uses vector_core_num, not cube_core_num.
    block_num = get_block_num(args.block_num, args.device, kind="vector")
    print(f"--- dtype={dtype_name} n={n_ele} ---")

    if dtype_name in {"i8", "i16", "i32"}:
        # Integer tensors: use randint (torch.rand is float-only).
        # Keep i8 ranges small enough that a+b stays in int8.
        if dtype_name == "i8":
            a = torch.randint(-25, 26, (n_ele,), dtype=torch_dtype, device="npu")
            b = torch.randint(-15, 16, (n_ele,), dtype=torch_dtype, device="npu")
        else:
            a = torch.randint(-1000, 1001, (n_ele,), dtype=torch_dtype, device="npu")
            b = torch.randint(-1000, 1001, (n_ele,), dtype=torch_dtype, device="npu")
    else:
        torch.npu.manual_seed(0)
        a = torch.rand(n_ele, dtype=torch_dtype, device="npu") * 10.0 - 5.0
        b = torch.rand(n_ele, dtype=torch_dtype, device="npu") * 10.0 - 5.0
    c = torch.full((n_ele,), sentinel, dtype=torch_dtype, device="npu")
    expected = golden(a, b)

    tla_a, tla_b, tla_c = (
        create_tla_tensor(a),
        create_tla_tensor(b),
        create_tla_tensor(c),
    )
    artifact = tla.compile(kernel, tla_a, tla_b, tla_c, options="--npu-arch 3510")
    artifact(tla_a, tla_b, tla_c, block_num=block_num)
    torch.npu.synchronize()

    passed = compare(c, expected, rtol=0.0, atol=1e-4)

    print(f"passed={passed} cache_key={artifact.cache_key}")
    print(f"kernel.o={artifact.kernel_binary_path}")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile and run a vector add.")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--n", type=int, default=VECTOR_ELE)
    parser.add_argument(
        "--dtype", choices=("f32", "f16", "i8", "i16", "i32"), default="f32"
    )
    parser.add_argument(
        "--block-num",
        type=int,
        default=-1,
        help="Launch block count; -1 = full vector_core_num (AIV) for this pure-v kernel",
    )

    # Kernel selector
    parser_group = parser.add_mutually_exclusive_group()
    parser_group.add_argument("--use-mutex", action="store_true")
    parser_group.add_argument("--use-mutex-with", action="store_true")
    parser_group.add_argument("--use-atomic-add", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
