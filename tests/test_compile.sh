#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

SCRIPT_PATH=$(dirname "$(realpath "$0")")
BUILD_SCRIPT_PATH=$(realpath "$SCRIPT_PATH"/../scripts/build.sh)

if [[ $# -ge 1 ]]; then
    CATLASS_ARCH="$1"
else
    CATLASS_ARCH=2201
fi

DEFAULT_BUILD_OPTIONS=(
    "--clean"
    "-DCATLASS_ARCH=${CATLASS_ARCH}"
)

# self contained include
bash "$BUILD_SCRIPT_PATH" "${DEFAULT_BUILD_OPTIONS[@]}" --tests test_self_contained_includes || exit 1

# msSanitizer
if [[ "$CATLASS_ARCH" != 3510 ]]; then
    bash "$BUILD_SCRIPT_PATH" "${DEFAULT_BUILD_OPTIONS[@]}" --enable_mssanitizer catlass_examples || exit 1
fi

# msopgen package build
if [[ "$CATLASS_ARCH" == 2201 ]]; then
    bash "$BUILD_SCRIPT_PATH" "${DEFAULT_BUILD_OPTIONS[@]}" -DSOC_VERSION=Ascend910B1 basic_matmul_aclnn || exit 1
fi

# example test. Do not replace or the test will fail.
bash "$BUILD_SCRIPT_PATH" "${DEFAULT_BUILD_OPTIONS[@]}" catlass_examples || exit 1
