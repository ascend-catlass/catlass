# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end: ``tla.Constexpr`` Callable epilogue inlined at staging time.

Demonstrates Phase-1 language-boundary support:

- Kernel takes ``epilogue: tla.Constexpr`` (``def`` / ``lambda``).
- Staging calls the Python callable and emits device IR (e.g. ``tla.abs``).
- Callable is stripped from the launch ABI (same as scalar Constexpr).
- Launch uses ``tla.compile`` → ``JitCompiledFunction`` only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_DSL_EXAMPLE_PATH = str(
    next(p for p in Path(__file__).resolve().parents if (p / "helpers").is_dir())
)

if _DSL_EXAMPLE_PATH not in sys.path:
    sys.path.append(_DSL_EXAMPLE_PATH)

import catlass.tla as tla
from catlass.tla.runtime import from_dlpack

VECTOR_ELE = 400
VL_ELE = 64
_KERNEL_DTYPE = tla.Float32


def abs_epilogue(value):
    """Module-level helper used as a Constexpr Callable argument."""
    return tla.abs(value)


@tla.kernel
def transform_with_epilogue(
    gm_src: tla.Tensor,
    gm_dst: tla.Tensor,
    epilogue: tla.Constexpr,
) -> None:
    n_ele = gm_src.origin_shape[0]
    ub_loaded = tla.flag("ub_loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
    vec_done = tla.flag("vec_done", tla.pipes.VECTOR, tla.pipes.MTE3)

    ub_ptr_src = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    ub_ptr_dst = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    ub_src = tla.make_tensor_like(ub_ptr_src, gm_src, tla.arch.RowMajor)
    ub_dst = tla.make_tensor_like(ub_ptr_dst, gm_dst, tla.arch.RowMajor)

    with tla.vector():
        tla.copy(ub_src, gm_src)
        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)
        with tla.vec.func(mode="simd"):
            for i in tla.range((n_ele + VL_ELE - 1) // VL_ELE):
                src_tile = tla.tile_view(
                    ub_src, tla.make_shape(VL_ELE), tla.make_coord(i)
                )
                dst_tile = tla.tile_view(
                    ub_dst, tla.make_shape(VL_ELE), tla.make_coord(i)
                )
                # Staging-time Python call: epilogue body is inlined into device IR.
                dst_tile.store(epilogue(src_tile.load()))
        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)
        tla.copy(gm_dst, ub_dst)
        tla.pipe_barrier(tla.pipes.ALL)


def run(args: argparse.Namespace) -> int:
    import torch
    import torch_npu  # noqa: F401

    from helpers import compare, get_block_num

    n_ele = int(args.n)
    if n_ele <= 0 or n_ele > VECTOR_ELE:
        raise SystemExit(f"--n={n_ele} out of range [1, {VECTOR_ELE}]")

    torch.npu.set_device(args.device)
    block_num = get_block_num(args.block_num, args.device, kind="vector")
    print(f"--- epilogue=abs n={n_ele} block_num={block_num} ---")

    torch.npu.manual_seed(0)
    src = torch.rand(n_ele, dtype=torch.float32, device="npu") * 10.0 - 5.0
    dst = torch.full((n_ele,), -7.0, dtype=torch.float32, device="npu")
    expected = src.abs()

    def create_tla_tensor(dev_buf):
        return from_dlpack(
            dev_buf.contiguous(), layout_tag=tla.arch.RowMajor
        ).mark_compact_shape_dynamic(0)

    tla_src = create_tla_tensor(src)
    tla_dst = create_tla_tensor(dst)

    # Compile with Constexpr callable in type_args; launch without it.
    compiled = tla.compile(
        transform_with_epilogue,
        tla_src,
        tla_dst,
        abs_epilogue,
        options="--npu-arch 3510",
    )
    compiled(tla_src, tla_dst, block_num=block_num)

    torch.npu.synchronize()
    passed = compare(dst, expected, rtol=0.0, atol=1e-4)
    print(f"passed={passed} cache_key={compiled.cache_key}")
    print(f"kernel.o={compiled.kernel_binary_path}")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Constexpr Callable epilogue (Phase-1 language boundary)."
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--n", type=int, default=VECTOR_ELE)
    parser.add_argument(
        "--block-num",
        type=int,
        default=-1,
        help="Launch block count; -1 = full vector_core_num",
    )
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
