# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Round-trip copy using two user-provided Ascend C external operations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_DSL_BASE_PATH = str(
    next(p for p in Path(__file__).resolve().parents if (p / "helpers").is_dir())
)
_DSL_PATH_ADDED = _DSL_BASE_PATH not in sys.path
if _DSL_PATH_ADDED:
    sys.path.append(_DSL_BASE_PATH)

import catlass.tla as tla
from catlass.tla.runtime import from_dlpack


TILE_ELE = 256

GM_TO_UB_SOURCE = r"""
#include <cstdint>
#include "kernel_operator.h"

extern "C" {

__aicore__ __attribute__((always_inline)) void tla_multi_gm_to_ub_f32(
    uint64_t src_gm_addr, uint64_t dst_ub_addr, int32_t count) {
  AscendC::GlobalTensor<float> src;
  src.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(src_gm_addr),
                      static_cast<uint32_t>(count));
  AscendC::LocalTensor<float> dst(AscendC::TPosition::VECCALC,
                                  static_cast<uint32_t>(dst_ub_addr),
                                  static_cast<uint32_t>(count));
  AscendC::DataCopy(dst, src, static_cast<uint32_t>(count));
}

} // extern "C"
"""

UB_TO_GM_SOURCE = r"""
#include <cstdint>
#include "kernel_operator.h"

extern "C" {

__aicore__ __attribute__((always_inline)) void tla_multi_ub_to_gm_f32(
    uint64_t src_ub_addr, uint64_t dst_gm_addr, int32_t count) {
  AscendC::LocalTensor<float> src(AscendC::TPosition::VECCALC,
                                  static_cast<uint32_t>(src_ub_addr),
                                  static_cast<uint32_t>(count));
  AscendC::GlobalTensor<float> dst;
  dst.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dst_gm_addr),
                      static_cast<uint32_t>(count));
  AscendC::DataCopy(dst, src, static_cast<uint32_t>(count));
}

} // extern "C"
"""


@tla.extern(source=GM_TO_UB_SOURCE, name="tla_multi_gm_to_ub_f32")
def extern_gm_to_ub_f32(
    gm_ptr: tla.Pointer[tla.Float32, tla.AddressSpace.gm],
    ub_ptr: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    ele_num: tla.Int32,
) -> None: ...


@tla.extern(source=UB_TO_GM_SOURCE, name="tla_multi_ub_to_gm_f32")
def extern_ub_to_gm_f32(
    ub_ptr: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    gm_ptr: tla.Pointer[tla.Float32, tla.AddressSpace.gm],
    ele_num: tla.Int32,
) -> None: ...


@tla.kernel
def extern_multi_ops(gm_src: tla.Tensor, gm_dst: tla.Tensor) -> None:
    """Copy GM -> UB -> GM using two different external operations."""

    ub_ptr = tla.allocate(TILE_ELE, tla.Float32, tla.AddressSpace.ub, 256)
    load_done = tla.flag("load_done", tla.arch.MTE2, tla.arch.MTE3)

    with tla.vector():
        extern_gm_to_ub_f32(gm_src.ptr, ub_ptr, TILE_ELE)
        tla.set_flag(load_done)
        tla.wait_flag(load_done)
        extern_ub_to_gm_f32(ub_ptr, gm_dst.ptr, TILE_ELE)
        tla.pipe_barrier(tla.pipes.ALL)


def run(device: int) -> int:
    """Compile, launch, and check that the round-trip copy preserves all values."""

    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(device)
    torch.npu.manual_seed(0)
    src = torch.rand(TILE_ELE, dtype=torch.float32, device="npu")
    dst = torch.empty_like(src)
    tla_src = from_dlpack(src, layout_tag=tla.arch.RowMajor)
    tla_dst = from_dlpack(dst, layout_tag=tla.arch.RowMajor)

    executor = tla.compile(
        extern_multi_ops,
        tla_src,
        tla_dst,
        options="--npu-arch 3510",
    )
    executor(tla_src, tla_dst, block_num=1)
    torch.npu.synchronize()
    torch.testing.assert_close(dst, src, rtol=0.0, atol=0.0)
    print(f"passed; kernel={executor.kernel_binary_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a two-extern GM-to-UB-to-GM copy."
    )
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    try:
        return run(args.device)
    finally:
        if _DSL_PATH_ADDED:
            sys.path.remove(_DSL_BASE_PATH)


if __name__ == "__main__":
    raise SystemExit(main())
