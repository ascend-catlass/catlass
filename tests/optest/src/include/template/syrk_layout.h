/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#ifndef OPTEST_TEMPLATE_SYRK_LAYOUT_H
#define OPTEST_TEMPLATE_SYRK_LAYOUT_H

#include <ATen/Tensor.h>

namespace CatlassKernelWrapper {

/**
 * @brief Layout of a SYRK operand, in BLAS `trans` terms.
 *
 * A [M, K] logical operand is accepted either as a dense row-major buffer (N) or as the
 * transposed view of a dense row-major [K, M] buffer (T); the kernel reads both directly.
 */
enum class SyrkOperandLayout
{
    kRowMajor = 0,   ///< strides (K, 1): X is stored [M, K] row-major.
    kTransposed = 1, ///< strides (1, M): X is a view of a row-major [K, M] buffer.
    kUnsupported = 2
};

inline SyrkOperandLayout ClassifySyrkOperand(const at::Tensor& mat)
{
    const int64_t rank = mat.dim();
    if (rank != 2 && rank != 3) {
        return SyrkOperandLayout::kUnsupported;
    }
    const int64_t rows = mat.size(-2);
    const int64_t cols = mat.size(-1);
    const int64_t strideRow = mat.stride(-2);
    const int64_t strideCol = mat.stride(-1);
    // The kernel derives the batch stride from m * k, so a batched operand must be packed.
    if (rank == 3 && mat.stride(0) != rows * cols) {
        return SyrkOperandLayout::kUnsupported;
    }
    if (strideRow == cols && strideCol == 1) {
        return SyrkOperandLayout::kRowMajor;
    }
    if (strideRow == 1 && strideCol == rows) {
        return SyrkOperandLayout::kTransposed;
    }
    return SyrkOperandLayout::kUnsupported;
}

} // namespace CatlassKernelWrapper

#endif // OPTEST_TEMPLATE_SYRK_LAYOUT_H
