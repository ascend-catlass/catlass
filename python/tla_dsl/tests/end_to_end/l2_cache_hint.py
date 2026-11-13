# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""E2E watch dog for ``tla.copy(..., cache_policy=)`` on GM→UB.

Correctness is identical to NORMAL (the hint only tags GM addresses); this
sample guards the API path, IR attr, BC ABI, link and on-device run.

The kernel does not partition by ``block_idx``, so launch is restricted to a
single vector core (``--block-num=1``).

Battery case: ``l2-cache-hint-disable``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# On-device sample under python/tla_dsl/tests/end_to_end; reuse example helpers.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_DSL_EXAMPLE_PATH = str(_REPO_ROOT / "examples" / "python" / "tla_dsl")

if _DSL_EXAMPLE_PATH not in sys.path:
    # append (not insert(0)) to avoid shadowing installed packages (G.PSL.03).
    sys.path.append(_DSL_EXAMPLE_PATH)

import catlass.tla as tla  # noqa: E402
from catlass.tla.runtime import from_dlpack  # noqa: E402

VECTOR_ELE = 256
VL_ELE = 64
_KERNEL_DTYPE = tla.Float32


@tla.kernel
def copy_with_l2_cache_hint(gm_src: tla.Tensor, gm_dst: tla.Tensor) -> None:
    """GM→UB→GM identity copy with DISABLE hint on the GM→UB copy."""
    n_ele = gm_src.origin_shape[0]
    ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    vec_done = tla.flag("vec_done", tla.arch.VECTOR, tla.arch.MTE3)

    ub_ptr = tla.allocate(VECTOR_ELE, _KERNEL_DTYPE, tla.AddressSpace.ub, 256)
    ub = tla.make_tensor_like(ub_ptr, gm_src, tla.arch.RowMajor)

    with tla.vector():
        tla.copy(ub, gm_src, cache_policy=tla.params.CachePolicy.DISABLE)

        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)
        with tla.vec.func(mode="simd"):
            # Touch the data so the path is not a pure DMA noop elision.
            for i in tla.range((n_ele + VL_ELE - 1) // VL_ELE):
                ub_vl = tla.tile_view(ub, tla.make_shape(VL_ELE), tla.make_coord(i))
                reg = ub_vl.load()
                ub_vl.store(reg)

        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)

        tla.copy(gm_dst, ub)
        tla.pipe_barrier(tla.pipes.ALL)


def golden(src):
    return src.clone()


def run(args: argparse.Namespace) -> int:
    import torch
    import torch_npu  # noqa: F401

    from helpers import compare, get_block_num  # noqa: E402

    n_ele = int(args.n)
    if n_ele <= 0 or n_ele > VECTOR_ELE:
        raise SystemExit(f"--n={n_ele} out of range [1, {VECTOR_ELE}]")
    if int(args.block_num) != 1:
        raise SystemExit(
            f"--block-num={args.block_num} rejected: this kernel does not "
            "partition by block_idx; use --block-num=1"
        )

    torch.npu.set_device(args.device)
    block_num = get_block_num(args.block_num, args.device, kind="vector")
    print(f"--- l2_cache_hint DISABLE n={n_ele} block_num={block_num} ---")

    torch.npu.manual_seed(0)
    src = torch.rand(n_ele, dtype=torch.float32, device="npu") * 10.0 - 5.0
    dst = torch.full((n_ele,), -7.0, dtype=torch.float32, device="npu")
    expected = golden(src)

    def create_tla_tensor(dev_buf):
        return from_dlpack(
            dev_buf.contiguous(), layout_tag=tla.arch.RowMajor
        ).mark_compact_shape_dynamic(0)

    tla_src, tla_dst = create_tla_tensor(src), create_tla_tensor(dst)
    artifact = tla.compile(
        copy_with_l2_cache_hint, tla_src, tla_dst, options="--npu-arch 3510"
    )
    artifact(tla_src, tla_dst, block_num=block_num)
    torch.npu.synchronize()

    passed = compare(dst, expected, rtol=0.0, atol=1e-4)
    print(f"passed={passed} cache_key={artifact.cache_key}")
    print(f"kernel.o={artifact.kernel_binary_path}")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile and run GM copy with cache_policy=DISABLE."
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--n", type=int, default=VECTOR_ELE)
    parser.add_argument(
        "--block-num",
        type=int,
        default=1,
        help="Launch block count; must be 1 (kernel is not block-partitioned)",
    )
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
