#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


import argparse
import os
import numpy as np


def gen_golden_data_simple(vaddConfig):
    if vaddConfig.m <= 0 or vaddConfig.n <= 0:
        raise ValueError("m and n must be positive")
    input_x = np.random.uniform(1, 100, [vaddConfig.m, vaddConfig.n]).astype(np.float32)
    input_y = np.random.uniform(1, 100, [vaddConfig.m, vaddConfig.n]).astype(np.float32)
    golden = (input_x + input_y).astype(np.float32)
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    input_x.tofile("./input/input_x.bin")
    input_y.tofile("./input/input_y.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate FP32 matrix add inputs and golden")
    parser.add_argument("--m", type=int, default=8192)
    parser.add_argument("--n", type=int, default=8192)
    gen_golden_data_simple(parser.parse_args())
