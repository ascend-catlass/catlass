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

_DSL_EXAMPLE_PATH = str(Path(__file__).resolve().parents[2])

if _DSL_EXAMPLE_PATH not in sys.path:
    sys.path.append(_DSL_EXAMPLE_PATH)

import argparse
from dataclasses import dataclass

import catlass.tla as tla
from catlass.tla.runtime import from_dlpack
from matrix_add_v0 import matrix_add_v0
from matrix_add_v1 import matrix_add_v1
from matrix_add_v2 import matrix_add_v2
from matrix_add_v3 import matrix_add_v3
from scripts.gen_data import gen_golden_data_simple
from scripts.verify_result import verify_result

# 添加新版本时，在此注册对应的matrix_add kernel
KERNELS = {
    "v0": matrix_add_v0,
    "v1": matrix_add_v1,
    "v2": matrix_add_v2,
    "v3": matrix_add_v3,
}

# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VaddConfig:
    # Matrix dimensions and per-buffer UB capacity in elements
    m: int = 8192
    n: int = 8192
    dataBlockLen: int = 4096
    VL_LEN: int = 64
    ELEM_TYPE_LEN: int = 4
    buffer_num: int = 2

    @property
    def UB_STAGE_BYTES(self) -> int:
        # A/B/C每个缓冲区的容量，单位为字节
        return self.dataBlockLen * self.ELEM_TYPE_LEN

    @property
    def BANK_CONFLICT_DATA_COPY_BYTES(self) -> int:
        # v3每个缓冲区额外预留128字节，增加半行空间申请，用于调整Bank地址布局
        return self.UB_STAGE_BYTES + 128


# Arch 3510, default compiler options: 256 KiB minus 8 KiB reserved
# See python/tla_dsl/csrc/mlir/include/Passes/TlaSimtUbLimits.h
UB_BUDGET_BYTES = 248 * 1024


def validate_config(config: VaddConfig, version: str, block_num: int) -> None:
    if config.m <= 0 or config.n <= 0:
        raise ValueError("--m and --n must be positive")
    if block_num != -1 and block_num <= 0:
        raise ValueError("--block-num must be -1 or a positive integer")
    if config.dataBlockLen <= 0:
        raise ValueError("--dataBlockLen must be positive")

    buffer_count = 3 if version in ("v0", "v1") else 3 * config.buffer_num
    alignment = 128 if version == "v3" else 256
    buffer_bytes = (
        config.BANK_CONFLICT_DATA_COPY_BYTES
        if version == "v3"
        else config.UB_STAGE_BYTES
    )
    # Include the alignment gap before each allocation, but no trailing gap
    aligned_bytes = (buffer_bytes + alignment - 1) // alignment * alignment
    required_bytes = (buffer_count - 1) * aligned_bytes + buffer_bytes
    if required_bytes > UB_BUDGET_BYTES:
        raise ValueError(
            f"--dataBlockLen={config.dataBlockLen} requires {required_bytes} UB bytes "
            f"for {version}, exceeding the arch 3510 budget of {UB_BUDGET_BYTES} bytes"
        )


def create_tla_tensor(dev_buf):
    return from_dlpack(
        dev_buf.contiguous(), layout_tag=tla.arch.RowMajor
    ).mark_compact_shape_dynamic(0)


def run(args: argparse.Namespace) -> int:
    import numpy as np
    import torch
    import torch_npu

    from helpers.utils import get_block_num

    kernel = KERNELS[args.version]
    config = VaddConfig(
        m=int(args.m), n=int(args.n), dataBlockLen=int(args.dataBlockLen)
    )
    validate_config(config, args.version, args.block_num)

    torch.npu.set_device(args.device)
    block_num = get_block_num(args.block_num, args.device, kind="vector")

    # The scripts use input/ and output/ relative to the working directory
    np.random.seed(0)
    gen_golden_data_simple(config)
    device = f"npu:{args.device}"
    a = torch.from_numpy(
        np.fromfile("input/input_x.bin", dtype=np.float32).reshape(config.m, config.n)
    ).to(device)
    b = torch.from_numpy(
        np.fromfile("input/input_y.bin", dtype=np.float32).reshape(config.m, config.n)
    ).to(device)
    c = torch.full_like(a, -7.0)

    print(
        f"--- version={args.version} dtype={args.dtype} m={config.m} n={config.n} ---"
    )
    tla_a, tla_b, tla_c = (
        create_tla_tensor(a),
        create_tla_tensor(b),
        create_tla_tensor(c),
    )
    artifact = tla.compile(
        kernel, tla_a, tla_b, tla_c, config, options="--npu-arch 3510"
    )
    artifact(tla_a, tla_b, tla_c, block_num=block_num)
    torch.npu.synchronize()

    output_path = "output/output.bin"
    c.cpu().numpy().tofile(output_path)
    passed = verify_result(
        output_path, "output/golden.bin", expected_size=config.m * config.n
    )

    print(f"passed={passed} cache_key={artifact.cache_key}")
    print(f"kernel.o={artifact.kernel_binary_path}")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile and run scalar matrix add.")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--m", type=int, default=VaddConfig.m)
    parser.add_argument("--n", type=int, default=VaddConfig.n)
    parser.add_argument("--dataBlockLen", type=int, default=VaddConfig.dataBlockLen)
    parser.add_argument(
        "--version",
        choices=tuple(KERNELS),
        default="v0",
        help="Matrix add kernel version",
    )
    parser.add_argument("--dtype", choices=("f32",), default="f32")
    parser.add_argument(
        "--block-num",
        type=int,
        default=-1,
        help="Launch block count; -1 = full vector_core_num (AIV) for this pure-v kernel",
    )

    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
