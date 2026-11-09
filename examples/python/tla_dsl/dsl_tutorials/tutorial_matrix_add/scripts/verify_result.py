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


import sys
from pathlib import Path
import numpy as np


RELATIVE_TOL = 1e-4
ABSOLUTE_TOL = 1e-5


def verify_result(output, golden, *, expected_size=None):
    element_bytes = np.dtype(np.float32).itemsize
    if any(Path(path).stat().st_size % element_bytes for path in (output, golden)):
        print("[ERROR] file size must be a multiple of the FP32 element size")
        return False
    output = np.fromfile(output, dtype=np.float32).reshape(-1)
    golden = np.fromfile(golden, dtype=np.float32).reshape(-1)
    if output.size == 0 or output.size != golden.size:
        print(
            f"[ERROR] invalid element counts: output={output.size}, golden={golden.size}"
        )
        return False
    if expected_size is not None and output.size != expected_size:
        print(f"[ERROR] expected {expected_size} elements, got {output.size}")
        return False
    if not np.all(np.isfinite(output)) or not np.all(np.isfinite(golden)):
        print("[ERROR] output and golden must contain only finite values")
        return False
    different_element_results = np.isclose(
        output, golden, rtol=RELATIVE_TOL, atol=ABSOLUTE_TOL, equal_nan=False
    )
    different_element_indexes = np.flatnonzero(~different_element_results)
    for index in range(len(different_element_indexes)):
        real_index = different_element_indexes[index]
        golden_data = golden[real_index]
        output_data = output[real_index]
        print(
            "data index: %06d, expected: %-.9f, actual: %-.9f, rdiff: %-.6f"
            % (
                real_index,
                golden_data,
                output_data,
                abs(float(output_data) - float(golden_data)) / abs(float(golden_data))
                if golden_data != 0
                else float("inf"),
            )
        )
        if index == 100:
            break
    error_ratio = float(different_element_indexes.size) / golden.size
    print(
        "mismatched elements: %d / %d, error ratio: %.8f"
        % (different_element_indexes.size, golden.size, error_ratio)
    )
    return different_element_indexes.size == 0


if __name__ == "__main__":
    try:
        res = verify_result(sys.argv[1], sys.argv[2])
        if not res:
            raise ValueError("[ERROR] result error")
        else:
            print("test pass!")
    except Exception as e:
        print(e)
        sys.exit(1)
