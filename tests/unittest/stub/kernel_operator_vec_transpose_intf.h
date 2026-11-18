/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCENDC_STUB_KERNEL_OPERATOR_VEC_TRANSPOSE_INTF_H
#define ASCENDC_STUB_KERNEL_OPERATOR_VEC_TRANSPOSE_INTF_H

#include <vector>

#include "kernel_tensor.h"
#include "kernel_struct_transpose.h"
#include "ascendc_logger.h"

namespace AscendC {

template <typename T>
inline void TransDataTo5HD(
    uint64_t dstList[NCHW_CONV_ADDR_LIST_SIZE], uint64_t srcList[NCHW_CONV_ADDR_LIST_SIZE],
    const TransDataTo5HDParams& params)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<T>()};
    const std::vector<uint64_t> dstAddrList(dstList, dstList + NCHW_CONV_ADDR_LIST_SIZE);
    const std::vector<uint64_t> srcAddrList(srcList, srcList + NCHW_CONV_ADDR_LIST_SIZE);
    const std::vector<Arg> args = {params.dstHighHalf, params.srcHighHalf, params.repeatTimes,
        params.dstRepStride, params.srcRepStride, Arg(dstAddrList), Arg(srcAddrList)};
    ASCENDC_LOG_CALL_T(TransDataTo5HD, argsT, args);
}

} // namespace AscendC

#endif // ASCENDC_STUB_KERNEL_OPERATOR_VEC_TRANSPOSE_INTF_H
