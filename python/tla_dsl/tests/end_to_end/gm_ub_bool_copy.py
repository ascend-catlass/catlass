# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""E2E accuracy: GM↔UB bool (i1) two-arg DataCopy roundtrip.

- rowmajor: GM RowMajor → UB → GM RowMajor.
- vector: GM VectorLayout → UB VectorLayout, then RowMajor view of the same UB
  bytes written back to a flat RowMajor GM (UB→GM has no VectorLayout route).
"""

from __future__ import annotations

import argparse

import catlass.tla as tla

_ROWS = 8
_COLS = 32
_N = _ROWS * _COLS
_ALIGN = 256

CASES = ("rowmajor", "vector")


@tla.kernel
def bool_rowmajor_roundtrip(gm_src: tla.Tensor, gm_dst: tla.Tensor) -> None:
    ub_ptr = tla.allocate(_N, tla.Bool, tla.AddressSpace.ub, _ALIGN)
    loaded = tla.flag("loaded", tla.arch.MTE2, tla.arch.MTE3)
    with tla.vector():
        ub = tla.make_tensor(
            ub_ptr,
            tla.make_layout(
                tla.make_shape(_ROWS, _COLS),
                tla.make_stride(_COLS, 1),
                origin_shape=tla.make_shape(_ROWS, _COLS),
                layoutTag=tla.arch.RowMajor,
            ),
            tla.make_coord(0, 0),
        )
        tla.copy(ub, gm_src)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        tla.copy(gm_dst, ub)
        tla.pipe_barrier(tla.pipes.ALL)


@tla.kernel
def bool_vector_roundtrip(gm_src: tla.Tensor, gm_dst: tla.Tensor) -> None:
    ub_ptr = tla.allocate(_N, tla.Bool, tla.AddressSpace.ub, _ALIGN)
    loaded = tla.flag("loaded", tla.arch.MTE2, tla.arch.MTE3)
    with tla.vector():
        ub_vec = tla.make_tensor(
            ub_ptr,
            tla.make_layout(
                tla.make_shape(_N),
                tla.make_stride(1),
                origin_shape=tla.make_shape(_N),
                layoutTag=tla.arch.VectorLayout,
            ),
            tla.make_coord(0),
        )
        tla.copy(ub_vec, gm_src)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        ub_rm = tla.make_tensor(
            ub_ptr,
            tla.make_layout(
                tla.make_shape(_N),
                tla.make_stride(1),
                origin_shape=tla.make_shape(_N),
                layoutTag=tla.arch.RowMajor,
            ),
            tla.make_coord(0),
        )
        tla.copy(gm_dst, ub_rm)
        tla.pipe_barrier(tla.pipes.ALL)


def run_case(case: str, device: int) -> None:
    import torch
    import torch_npu

    del torch_npu
    torch.npu.set_device(device)

    host = torch.tensor([((i * 3) % 5 != 0) for i in range(_N)], dtype=torch.bool)
    if case == "rowmajor":
        src_host = host.view(_ROWS, _COLS)
        dst_host = torch.zeros((_ROWS, _COLS), dtype=torch.bool)
        layout_src = tla.arch.RowMajor
        layout_dst = tla.arch.RowMajor
        kernel = bool_rowmajor_roundtrip
        origin_src = (_ROWS, _COLS)
        origin_dst = (_ROWS, _COLS)
    else:
        src_host = host.clone()
        dst_host = torch.zeros((_N,), dtype=torch.bool)
        layout_src = tla.arch.VectorLayout
        layout_dst = tla.arch.RowMajor
        kernel = bool_vector_roundtrip
        origin_src = (_N,)
        origin_dst = (_N,)

    src_dev = src_host.to(f"npu:{device}")
    dst_dev = dst_host.to(f"npu:{device}")
    gm_src = tla.from_dlpack(src_dev, layout_tag=layout_src, origin_shape=origin_src)
    gm_dst = tla.from_dlpack(dst_dev, layout_tag=layout_dst, origin_shape=origin_dst)
    artifact = tla.compile(kernel, gm_src, gm_dst, options="--npu-arch 3510")
    artifact(gm_src, gm_dst, block_num=1)
    torch.npu.synchronize()

    got = dst_dev.cpu().reshape(-1).to(torch.uint8)
    exp = host.reshape(-1).to(torch.uint8)
    if not torch.equal(got, exp):
        raise AssertionError(
            f"{case}: mismatch got={got[:16].tolist()} exp={exp[:16].tolist()}"
        )
    print(f"PASS {case} elems={_N}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--case", choices=(*CASES, "all"), default="all")
    args = parser.parse_args()
    names = CASES if args.case == "all" else (args.case,)
    for name in names:
        run_case(name, args.device)
    print(f"GM↔UB bool accuracy: {len(names)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
