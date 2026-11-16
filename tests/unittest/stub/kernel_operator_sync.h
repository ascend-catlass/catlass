/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCENDC_STUB_KERNEL_OPERATOR_SYNC_H
#define ASCENDC_STUB_KERNEL_OPERATOR_SYNC_H

#include <cstdint>
#include <cstdio>
#include <vector>
#include <string>
#include "ascendc_logger.h"
#include "kernel_constants.h"

enum pipe_t
{
    PIPE_M = 0,
    PIPE_V = 1,
    PIPE_MTE1 = 2,
    PIPE_MTE2 = 3,
    PIPE_MTE3 = 4,
    PIPE_FIX = 5,
    PIPE_ALL = 6
};

namespace AscendC {

template <pipe_t pipe>
inline void PipeBarrier()
{
    ASCENDC_LOG_CALL_T(PipeBarrier, {Arg::MakeArgWithValue<pipe_t>(pipe)}, {});
}

template <int8_t startBit, int8_t endBit>
inline int64_t GetCtrlSpr()
{
    const std::vector<Arg> argsT = {Arg::MakeArgWithValue(startBit), Arg::MakeArgWithValue(endBit)};
    const std::vector<Arg> args = {};
    ASCENDC_LOG_CALL_T(GetCtrlSpr, argsT, args);
    return 0;
}

template <int8_t startBit, int8_t endBit>
inline void SetCtrlSpr(int64_t value)
{
    const std::vector<Arg> argsT = {Arg::MakeArgWithValue(startBit), Arg::MakeArgWithValue(endBit),
        Arg::MakeArgWithValue(value)};
    const std::vector<Arg> args = {};
    ASCENDC_LOG_CALL_T(SetCtrlSpr, argsT, args);
}

enum class HardEvent
{
    MTE1_MTE2,
    MTE2_MTE1,
    M_MTE1,
    MTE1_M,
    M_FIX,
    FIX_M,
    V_MTE2,
    MTE2_V,
    MTE3_MTE2,
    MTE2_MTE3,
    MTE3_V,
    V_MTE3
};

template <HardEvent event>
inline void SetFlag(int32_t eventId)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<HardEvent>()};
    const std::vector<Arg> args = {eventId};
    ASCENDC_LOG_CALL_T(SetFlag, argsT, args);
}

template <HardEvent event>
inline void WaitFlag(int32_t eventId)
{
    const std::vector<Arg> argsT = {Arg::MakeArg<HardEvent>()};
    const std::vector<Arg> args = {eventId};
    ASCENDC_LOG_CALL_T(WaitFlag, argsT, args);
}

} // namespace AscendC

#endif // ASCENDC_STUB_KERNEL_OPERATOR_SYNC_H
