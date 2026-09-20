/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @brief 自包含的 Kernel Runner 头文件。
 *
 * 提供 RunKernel<Kernel> —— 一站式 host 函数。
 *
 * Workspace 由外部（torch 层）统一管理，通过 CatlassSetWorkspaceAlloc 注入。
 * torch 层静态初始化器保证 g_catlassWorkspaceAlloc 始终有效。
 */

#ifndef OPTEST_KERNELS_COMMON_KERNEL_RUNNER_H
#define OPTEST_KERNELS_COMMON_KERNEL_RUNNER_H

#include <cstdint>
#include <type_traits>

#include <acl/acl.h>

#include "catlass/catlass.hpp"
#include "catlass/status.hpp"

#include "common/workspace_alloc.h"

#ifndef KERNEL_NAME
#define KERNEL_NAME undefined
#endif

#ifndef KERNEL_TYPE
#define KERNEL_TYPE
#endif

namespace Catlass {

template <class Kernel, class = void>
struct HasUbSize : std::false_type {};

template <class Kernel>
struct HasUbSize<Kernel, std::void_t<decltype(Kernel::UB_SIZE)>> : std::true_type {};

// 设备端 kernel 入口

template <class Kernel>
CATLASS_GLOBAL KERNEL_TYPE void KERNEL_NAME(typename Kernel::Params params)
{
    Kernel kernel;
    kernel(params);
}

// 一站式 host 启动

template <class Kernel>
inline void RunKernel(typename Kernel::Arguments args, aclrtStream stream, uint32_t coreNum)
{
    if (!Kernel::CanImplement(args)) {
        return;
    }
    size_t wsSize = Kernel::GetWorkspaceSize(args);
    uint8_t* ws = nullptr;
    if (wsSize > 0) {
        ws = g_catlassWorkspaceAlloc(wsSize);
    }
    auto params = Kernel::ToUnderlyingArguments(args, ws);
    if constexpr (HasUbSize<Kernel>::value) {
        KERNEL_NAME<Kernel><<<coreNum, Kernel::UB_SIZE, stream>>>(params);
    } else {
        KERNEL_NAME<Kernel><<<coreNum, nullptr, stream>>>(params);
    }

    if (ws != nullptr && g_catlassWorkspaceFree != nullptr && aclrtSynchronizeStream(stream) == ACL_ERROR_NONE) {
        g_catlassWorkspaceFree(ws, wsSize);
    }
}

} // namespace Catlass

#endif // OPTEST_KERNELS_COMMON_KERNEL_RUNNER_H
