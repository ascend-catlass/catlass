# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""gm_c = gm_a + gm_b with the length fixed at compile time.

Kernel body, compile-time configuration and the compile entry point live in one
class. ``n_ele`` is an instance attribute, so it folds to a constant while the
kernel is staged: each length gets its own binary and its own cache key.

Runnable on Ascend 950 (``--npu-arch 3510``):
    python dsl_tutorials/kernel_in_class/vector_add_with_static_length.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_DSL_EXAMPLE_PATH = str(
    next(p for p in Path(__file__).resolve().parents if (p / "helpers").is_dir())
)

if _DSL_EXAMPLE_PATH not in sys.path:
    sys.path.append(_DSL_EXAMPLE_PATH)

import catlass.tla as tla
from catlass.tla.runtime import from_dlpack

VL_ELE = 64


class VectorAddWithStaticLength:
    """gm_c = gm_a + gm_b; ``n_ele`` is fixed when the instance is built."""

    def __init__(self, n_ele: int, vl_ele: int = VL_ELE) -> None:
        if n_ele <= 0 or n_ele % vl_ele:
            raise ValueError(
                f"n_ele={n_ele} must be a positive multiple of vl_ele={vl_ele}"
            )
        self.n_ele = n_ele
        self.vl_ele = vl_ele

    @tla.jit
    def __call__(self, gm_a, gm_b, gm_c):
        self.kernel(gm_a, gm_b, gm_c)

    @tla.kernel
    def kernel(self, gm_a, gm_b, gm_c):
        ub_loaded = tla.flag("ub_loaded", tla.pipes.MTE2, tla.pipes.VECTOR)
        vec_done = tla.flag("vec_done", tla.pipes.VECTOR, tla.pipes.MTE3)

        ub_a = tla.make_tensor_like(
            tla.allocate(self.n_ele, tla.Float32, tla.AddressSpace.ub, 256),
            gm_a,
            tla.arch.RowMajor,
        )
        ub_b = tla.make_tensor_like(
            tla.allocate(self.n_ele, tla.Float32, tla.AddressSpace.ub, 256),
            gm_b,
            tla.arch.RowMajor,
        )
        ub_c = tla.make_tensor_like(
            tla.allocate(self.n_ele, tla.Float32, tla.AddressSpace.ub, 256),
            gm_c,
            tla.arch.RowMajor,
        )

        with tla.vector():
            tla.copy(ub_a, gm_a)
            tla.copy(ub_b, gm_b)
            tla.set_flag(ub_loaded)
            tla.wait_flag(ub_loaded)

            with tla.vec.func(mode="simd"):
                if tla.const_expr(self.n_ele == self.vl_ele):
                    vl = tla.make_shape(self.vl_ele)
                    at = tla.make_coord(0)
                    tla.tile_view(ub_c, vl, at).store(
                        tla.add(
                            tla.tile_view(ub_a, vl, at).load(),
                            tla.tile_view(ub_b, vl, at).load(),
                        )
                    )
                else:
                    for i in tla.range(self.n_ele // self.vl_ele):
                        vl = tla.make_shape(self.vl_ele)
                        at = tla.make_coord(i)
                        tla.tile_view(ub_c, vl, at).store(
                            tla.add(
                                tla.tile_view(ub_a, vl, at).load(),
                                tla.tile_view(ub_b, vl, at).load(),
                            )
                        )

            tla.set_flag(vec_done)
            tla.wait_flag(vec_done)

            tla.copy(gm_c, ub_c)
            tla.pipe_barrier(tla.pipes.ALL)


def main() -> int:
    import torch
    import torch_npu

    from helpers import compare, get_block_num

    torch.npu.set_device(0)
    torch.npu.manual_seed(0)
    block_num = get_block_num(-1, 0, kind="vector")

    def bind(buf):
        return from_dlpack(
            buf.contiguous(), layout_tag=tla.arch.RowMajor
        ).mark_compact_shape_dynamic(0)

    ok = True
    for n_ele in (64, 384):
        a = torch.rand(n_ele, dtype=torch.float32, device="npu") * 10.0 - 5.0
        b = torch.rand(n_ele, dtype=torch.float32, device="npu") * 10.0 - 5.0
        c = torch.full((n_ele,), -7.0, dtype=torch.float32, device="npu")
        tla_a, tla_b, tla_c = bind(a), bind(b), bind(c)

        compiled = tla.compile(
            VectorAddWithStaticLength(n_ele),
            tla_a,
            tla_b,
            tla_c,
            options="--npu-arch 3510",
        )
        compiled(tla_a, tla_b, tla_c, block_num=block_num)
        torch.npu.synchronize()

        passed = compare(c, a + b, rtol=0.0, atol=1e-4)
        ok = ok and passed
        print(f"n_ele={n_ele} passed={passed} cache_key={compiled.cache_key}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
