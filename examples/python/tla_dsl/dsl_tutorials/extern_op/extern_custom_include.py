# -------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -------------------------------------------------------------------------------------------
"""External operation that writes a value defined in a user header."""

from __future__ import annotations

import argparse
from pathlib import Path

import catlass.tla as tla
from catlass.tla.runtime import from_dlpack


CUSTOM_INCLUDE_DIR = Path(__file__).resolve().parent / "include"
EXPECTED_VALUE = 42

OP_SOURCE_CODE = r"""
#include "tla_custom_value.h"

extern "C" {

__aicore__ __attribute__((always_inline)) void tla_user_custom_include(
    uint64_t dst_gm_addr) {
  auto dst = reinterpret_cast<__gm__ int32_t *>(dst_gm_addr);
  dst[0] = tla_custom::kHeaderValue;
}

} // extern "C"
"""


@tla.extern(
    source=OP_SOURCE_CODE,
    name="tla_user_custom_include",
    include_dirs=[CUSTOM_INCLUDE_DIR],
)
def custom_include_op(
    dst: tla.Pointer[tla.Int32, tla.AddressSpace.gm],
) -> None: ...


@tla.kernel
def extern_custom_include(result: tla.Tensor) -> None:
    with tla.cube():
        custom_include_op(result.ptr)
        tla.pipe_barrier(tla.pipes.ALL)


def run(device: int = 0) -> None:
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(device)
    result = torch.zeros(1, dtype=torch.int32, device="npu")
    tla_result = from_dlpack(result, layout_tag=tla.arch.RowMajor)

    executor = tla.compile(
        extern_custom_include,
        tla_result,
        options="--npu-arch 3510",
    )
    executor(tla_result, block_num=1)
    torch.npu.synchronize()

    assert result.item() == EXPECTED_VALUE
    print(f"passed; kernel={executor.kernel_binary_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    run(parser.parse_args().device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
