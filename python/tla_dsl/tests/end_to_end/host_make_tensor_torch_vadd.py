# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end Host bind with Torch via dual-mode ``tla.make_tensor`` (no DLPack).

Demonstrates the Kernel-aligned Host bind path against a small RowMajor vector
add, for both **static** and **dynamic** Host layouts:

```python
import catlass.tla as tla

ptr = tla.make_ptr(
    tla.Float16,
    int(buf.data_ptr()),
    tla.AddressSpace.gm,
)
layout = tla.make_layout(
    tla.make_shape(M, N),
    tla.make_stride(N, 1),
    layoutTag=tla.arch.RowMajor,
)
host = tla.make_tensor(ptr, layout)
# Dynamic (optional): same as from_dlpack — mark after static bind.
host = host.mark_layout_dynamic()
```

Battery case: ``host-make-tensor-torch-vadd``.
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

M_DIM = 32
N_DIM = 64
VECTOR_TILE_M = 32
VECTOR_TILE_N = 64
VECTOR_REG_TILE_M = 2
UB_TILE_BYTES = VECTOR_TILE_M * VECTOR_TILE_N * 2

DESCRIPTION = (
    "tla.make_ptr/make_layout + dual-mode tla.make_tensor Torch bind + vector add "
    "(no DLPack); runs static and dynamic Host layout modes."
)


@tla.kernel
def host_make_tensor_torch_vadd(
    lhs: tla.Tensor,
    rhs: tla.Tensor,
    out: tla.Tensor,
) -> None:
    ub_load_ready = tla.flag("ub_load_ready", tla.arch.VECTOR, tla.arch.MTE2)
    ub_loaded = tla.flag("ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    vec_done = tla.flag("vec_done", tla.arch.VECTOR, tla.arch.MTE3)

    a_ub_ptr = tla.allocate(UB_TILE_BYTES // 2, tla.Float16, tla.AddressSpace.ub, 256)
    b_ub_ptr = tla.allocate(UB_TILE_BYTES // 2, tla.Float16, tla.AddressSpace.ub, 256)
    c_ub_ptr = tla.allocate(UB_TILE_BYTES // 2, tla.Float16, tla.AddressSpace.ub, 256)

    with tla.vector():
        gm_a = tla.tile_view(
            lhs,
            tla.make_shape(VECTOR_TILE_M, VECTOR_TILE_N),
            tla.make_coord(0, 0),
        )
        gm_b = tla.tile_view(
            rhs,
            tla.make_shape(VECTOR_TILE_M, VECTOR_TILE_N),
            tla.make_coord(0, 0),
        )
        gm_c = tla.tile_view(
            out,
            tla.make_shape(VECTOR_TILE_M, VECTOR_TILE_N),
            tla.make_coord(0, 0),
        )

        a_ub = tla.make_tensor_like(a_ub_ptr, gm_a, tla.arch.RowMajor)
        b_ub = tla.make_tensor_like(b_ub_ptr, gm_b, tla.arch.RowMajor)
        c_ub = tla.make_tensor_like(c_ub_ptr, gm_c, tla.arch.RowMajor)

        tla.set_flag(ub_load_ready)
        tla.wait_flag(ub_load_ready)
        tla.copy(a_ub, gm_a)
        tla.copy(b_ub, gm_b)
        tla.set_flag(ub_loaded)
        tla.wait_flag(ub_loaded)

        for row_tile_idx in tla.range(0, VECTOR_TILE_M // VECTOR_REG_TILE_M, 1):
            with tla.vec.func(mode="simd"):
                a_chunk = tla.tile_view(
                    a_ub,
                    tla.make_shape(VECTOR_REG_TILE_M, VECTOR_TILE_N),
                    tla.make_coord(row_tile_idx, 0),
                )
                b_chunk = tla.tile_view(
                    b_ub,
                    tla.make_shape(VECTOR_REG_TILE_M, VECTOR_TILE_N),
                    tla.make_coord(row_tile_idx, 0),
                )
                c_chunk = tla.tile_view(
                    c_ub,
                    tla.make_shape(VECTOR_REG_TILE_M, VECTOR_TILE_N),
                    tla.make_coord(row_tile_idx, 0),
                )
                c_chunk.store(a_chunk.load() + b_chunk.load())

        tla.set_flag(vec_done)
        tla.wait_flag(vec_done)
        tla.copy(gm_c, c_ub)
        tla.pipe_barrier(tla.pipes.ALL)


def _bind_torch_row_major(buf, *, dynamic: bool) -> tla.Tensor:
    """Host Torch bind: ``tla.make_ptr`` + ``make_layout`` + dual-mode ``tla.make_tensor``.

    ``buf`` must already be a contiguous NPU RowMajor buffer whose logical shape
    matches ``(M_DIM, N_DIM)``. When ``dynamic`` is true, mark layout dynamic
    after bind (same post-bind step as ``from_dlpack``).
    """
    ptr = tla.make_ptr(
        tla.Float16,
        int(buf.data_ptr()),
        tla.AddressSpace.gm,
    )
    layout = tla.make_layout(
        tla.make_shape(int(buf.shape[0]), int(buf.shape[1])),
        tla.make_stride(int(buf.stride(0)), int(buf.stride(1))),
        layoutTag=tla.arch.RowMajor,
    )
    host = tla.make_tensor(ptr, layout)
    if dynamic:
        host = host.mark_layout_dynamic(leading_dim=1)
    return host


def _run_one(
    *,
    mode: str,
    device: int,
    block_num_arg: int,
    sentinel: float,
) -> bool:
    import torch
    import torch_npu

    from helpers import compare, get_block_num

    del torch_npu  # import required for torch.npu

    dynamic = mode == "dynamic"
    torch.npu.set_device(device)
    print(f"--- Host make_tensor Torch vadd mode={mode} mn=({M_DIM},{N_DIM}) ---")

    lhs_cpu = torch.rand(M_DIM, N_DIM, dtype=torch.float16) * 2.0 - 1.0
    rhs_cpu = torch.rand(M_DIM, N_DIM, dtype=torch.float16) * 2.0 - 1.0
    out_cpu = torch.full((M_DIM, N_DIM), sentinel, dtype=torch.float16)
    ref = (lhs_cpu.float() + rhs_cpu.float()).half()

    lhs = lhs_cpu.contiguous().npu()
    rhs = rhs_cpu.contiguous().npu()
    out = out_cpu.contiguous().npu()

    a_tensor = _bind_torch_row_major(lhs, dynamic=dynamic)
    b_tensor = _bind_torch_row_major(rhs, dynamic=dynamic)
    c_tensor = _bind_torch_row_major(out, dynamic=dynamic)

    if dynamic:
        if a_tensor._dynamic_shape_tree is None:
            raise RuntimeError("expected mark_layout_dynamic to set _dynamic_shape_tree")
        if a_tensor._dynamic_stride_tree != (None, 1):
            raise RuntimeError(
                f"expected _dynamic_stride_tree==(None, 1), got {a_tensor._dynamic_stride_tree!r}"
            )

    artifact = tla.compile(
        host_make_tensor_torch_vadd,
        a_tensor,
        b_tensor,
        c_tensor,
        options="--npu-arch 3510",
    )
    block_num = get_block_num(block_num_arg, device, kind="vector")
    if block_num_arg < 0:
        block_num = 1
    artifact(a_tensor, b_tensor, c_tensor, block_num=block_num)
    torch.npu.synchronize()

    passed = compare(out.detach().cpu(), ref, rtol=1e-3, atol=1e-3)
    print(f"mode={mode} passed={passed} cache_key={artifact.cache_key}")
    print(f"kernel.o={artifact.kernel_binary_path}")
    return bool(passed)


def run(args: argparse.Namespace) -> int:
    mi, ni = int(args.m), int(args.n)
    if (mi, ni) != (M_DIM, N_DIM):
        raise SystemExit(
            f"this sample fixes M={M_DIM}, N={N_DIM} for the Host bind demo; got {(mi, ni)}"
        )

    modes = ("static", "dynamic") if args.mode == "both" else (args.mode,)
    results = [
        _run_one(
            mode=mode,
            device=args.device,
            block_num_arg=args.block_num,
            sentinel=args.sentinel,
        )
        for mode in modes
    ]
    return 0 if all(results) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--m", type=int, default=M_DIM)
    parser.add_argument("--n", type=int, default=N_DIM)
    parser.add_argument("--block-num", type=int, default=1)
    parser.add_argument("--sentinel", type=float, default=-9.0)
    parser.add_argument(
        "--mode",
        choices=("static", "dynamic", "both"),
        default="both",
        help="Host layout mode: static bind, mark_layout_dynamic after bind, or both.",
    )
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
