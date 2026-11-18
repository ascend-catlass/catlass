/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCENDC_STUB_KERNEL_STRUCT_TRANSPOSE_H
#define ASCENDC_STUB_KERNEL_STRUCT_TRANSPOSE_H

#include <cstdint>

namespace AscendC {

constexpr uint32_t NCHW_CONV_ADDR_LIST_SIZE = 16;

struct TransDataTo5HDParams {
    constexpr TransDataTo5HDParams() = default;

    constexpr TransDataTo5HDParams(
        const bool dstHighHalfIn, const bool srcHighHalfIn, const uint8_t repeatTimesIn, const uint16_t dstRepStrideIn,
        const uint16_t srcRepStrideIn)
        : dstHighHalf(dstHighHalfIn),
          srcHighHalf(srcHighHalfIn),
          repeatTimes(repeatTimesIn),
          dstRepStride(dstRepStrideIn),
          srcRepStride(srcRepStrideIn)
    {}

    bool dstHighHalf = false;
    bool srcHighHalf = false;
    uint8_t repeatTimes = 1;
    uint16_t dstRepStride = 0;
    uint16_t srcRepStride = 0;
};

} // namespace AscendC

#endif // ASCENDC_STUB_KERNEL_STRUCT_TRANSPOSE_H
